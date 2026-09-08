"""A bounded three-call experiment with durable, inspectable provenance."""

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import threading
import time
import uuid

from .artifacts import image_record, import_parent, make_comparison, save_image, sha256, sync_directory, write_json
from .config import Config, MODEL_ID, adjustments, parse_args
from .costs import tracked_local_call


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dependencies() -> dict:
    return {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata["Name"]}


def code_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    files = [*sorted((root / "src/kusama").glob("*.py")), root / "scripts/first_edit.py", root / "pyproject.toml", root / "uv.lock"]
    return {str(path.relative_to(root)): sha256(path) for path in files if path.is_file()}


def publish_saved_image(source: Path, target: Path) -> None:
    """Publish a separate byte-identical copy without replacing any existing output."""
    temporary = target.with_suffix('.publish.tmp.png')
    with source.open('rb') as incoming, temporary.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    # An exclusive link preserves atomic publication without linking the mutable
    # canonical parent directly to its retained raw evidence.
    os.link(temporary, target)
    temporary.unlink()
    sync_directory(target.parent)


class Measurement:
    """Synchronized wall time and memory sampling; MPS has no true peak allocator API."""

    def __init__(self, device: str):
        self.device = device
        self.result = {"sample_interval_seconds": 0.25, "sampled_max_bytes": {}, "samples": 0, "measurement_errors": []}
        self.stop = threading.Event()

    def sample(self) -> None:
        from .backend import memory_snapshot
        try:
            values = memory_snapshot(self.device)
            self.result["after"] = values
            self.result.setdefault("before", values)
            self.result["samples"] += 1
            for key, value in values.items():
                if key.endswith("_bytes") and isinstance(value, (int, float)) and not isinstance(value, bool):
                    maxima = self.result["sampled_max_bytes"]
                    maxima[key] = max(value, maxima.get(key, value))
        except Exception as exc:
            if not self.result["measurement_errors"]:
                self.result["measurement_errors"].append(str(exc))

    def poll(self) -> None:
        while not self.stop.wait(0.25):
            self.sample()

    def __enter__(self):
        from .backend import reset_peak_memory, synchronize
        synchronize(self.device)
        reset_peak_memory(self.device)
        self.started = time.perf_counter()
        self.sample()
        self.thread = threading.Thread(target=self.poll, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        from .backend import synchronize
        try:
            synchronize(self.device)
        except Exception as sync_error:
            self.result["measurement_errors"].append(str(sync_error))
        self.stop.set()
        self.thread.join(timeout=2)
        self.sample()
        self.result["runtime_seconds"] = time.perf_counter() - self.started
        self.result["peak_note"] = "sampled_max_bytes contains observed sample maxima, not exact peaks. CUDA allocator peaks, when present in after, are native per-stage peaks; MPS has no native peak counter."


@contextmanager
def handle_termination():
    """SIGINT/SIGTERM preserve progress. SIGKILL/power loss leave the last running record."""
    old = None
    if threading.current_thread() is threading.main_thread():
        old = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM received")))
    try:
        yield
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def run_experiment(config: Config, backend_factory=None) -> tuple[Path, int]:
    """Backend injection is only for tests; the public CLI always calls real FLUX."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "_" + uuid.uuid4().hex[:8]
    run_dir = config.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = run_dir / "manifest.json"
    started = time.perf_counter()
    manifest = {
        "schema_version": 1, "run_id": run_id, "experiment_kind": "development",
        "status": "running", "started_at": utc_now(), "finished_at": None,
        "model": {"id": MODEL_ID, "requested_revision": config.model_revision, "resolved_revision": None, "frozen": True},
        "requested_config": config.to_dict(), "configuration_adjustments": adjustments(config),
        "prompts": {"parent": config.prompt if config.parent_image is None else None, "edit_instruction": config.edit_instruction},
        "seeds": {"parent": config.parent_seed if config.parent_image is None else None, "edits": list(config.edit_seeds)},
        "dependencies": dependencies(), "code_sha256": code_provenance(),
        "budget": {"planned_image_calls": 2 if config.parent_image else 3, "max_retries_per_call": config.max_retries, "maximum_image_attempts": (2 if config.parent_image else 3) * (config.max_retries + 1), "backend_request_attempts": 0, "actual_image_attempts": 0, "successful_image_calls": 0},
        "calls": [], "images": [], "failures": [], "backend": {}, "environment": {},
        "cost_accounting": {"root": str(config.output_dir.resolve()), "run_id": run_id},
        "observations": {"model_observations": [], "human_labels": [], "visual_review_status": "pending", "note": "Inference completion does not establish instruction-following. Human labels must be supplied by a human; assistant visual assessments belong in model_observations with provenance."},
        "comparison": None,
    }
    if not config.parent_image:
        manifest["calls"].append({"id": "parent", "kind": "generate", "seed": config.parent_seed, "prompt": config.prompt, "status": "planned", "attempts": []})
    for seed in config.edit_seeds:
        manifest["calls"].append({"id": f"edit_seed_{seed}", "kind": "edit", "seed": seed, "prompt": config.edit_instruction, "status": "planned", "attempts": []})

    def persist():
        manifest["runtime_seconds_so_far"] = time.perf_counter() - started
        write_json(manifest_path, manifest)

    def failure(stage: str, exc: BaseException, attempt: dict | None = None, *, phase: str | None = None):
        entry = {"stage": stage, "exception_type": type(exc).__name__, "message": str(exc), "timestamp": utc_now()}
        if phase is not None:
            entry['phase'] = phase
        manifest["failures"].append(entry)
        if attempt is not None:
            attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", failure=entry)
        print(f"{stage}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def backend_progress(details: dict):
        manifest["backend"].update(details)
        if details.get("resolved_revision"):
            manifest["model"]["resolved_revision"] = details["resolved_revision"]
        persist()

    backend = None
    parent = None
    interrupted = False
    persist()
    print(f"Run directory: {run_dir}", flush=True)
    try:
        # Preserve supplied input before network access or model loading can fail.
        if config.parent_image:
            manifest["parent_source"] = import_parent(config.parent_image, run_dir / "parent.png")
            parent = image_record(run_dir / "parent.png", image_id="parent", parent=None, seed=None, origin="provided_image")
            manifest["images"].append(parent)
            persist()
            if parent["size"] != [config.width, config.height]:
                raise ValueError(f"Provided parent has size {parent['size']}; pass matching --width and --height. The source was saved without resizing.")
        from .backend import FluxBackend, inspect_environment
        manifest["environment"] = inspect_environment()
        persist()
        backend = (backend_factory or FluxBackend)(config)
        device = config.device if config.device != "auto" else manifest["environment"].get("preferred_device", "cpu")
        # Loading measurements cover model downloads, loading and accelerator placement.
        load_measure = Measurement(device)
        try:
            with tracked_local_call(config.output_dir, run_id, 'flux:load', MODEL_ID,
                                    scope='model_loading') as cost:
                try:
                    with load_measure:
                        backend_progress(backend.prepare(on_progress=backend_progress))
                finally:
                    cost['measurements'] = load_measure.result
                    cost['runtime_seconds'] = load_measure.result.get('runtime_seconds')
        finally:
            manifest["load_measurements"] = load_measure.result
            persist()
        device = backend.execution_settings["device"]
        manifest["effective_settings"] = dict(backend.execution_settings)
        persist()
        for call in manifest["calls"]:
            if call["kind"] == "edit" and parent is None:
                call.update(status="skipped", reason="Parent generation did not produce a saved image")
                persist()
                continue
            call["settings"] = dict(backend.execution_settings)
            call["parent_image_id"] = parent["id"] if call["kind"] == "edit" else None
            call["parent_sha256"] = parent["sha256"] if call["kind"] == "edit" else None
            call["input_image_path"] = "parent.png" if call["kind"] == "edit" else None
            for number in range(1, config.max_retries + 2):
                raw_target = run_dir / f"{call['id']}_attempt_{number}_raw.png"
                attempt = {"number": number, "started_at": utc_now(), "status": "running", "pipeline_invoked": False,
                           "cost_call_id": f"{call['id']}:attempt:{number}",
                           "phase": "inference", "raw_output_path": raw_target.name, "raw_output_saved": False}
                call["attempts"].append(attempt)
                call["status"] = "running"
                persist()
                measure = Measurement(device)
                try:
                    parent_path = run_dir / "parent.png" if call["kind"] == "edit" else None
                    if parent_path and sha256(parent_path) != parent["sha256"]:
                        raise RuntimeError("Saved parent hash changed. Refusing an edit with the wrong input.")
                    print(f"{call['id']}: attempt {number}/{config.max_retries + 1}", flush=True)
                    with measure:
                        def inference_started():
                            attempt["pipeline_invoked"] = True
                            manifest["budget"]["actual_image_attempts"] += 1
                            persist()

                        manifest["budget"]["backend_request_attempts"] += 1
                        if hasattr(backend, "on_inference_start"):
                            backend.on_inference_start = inference_started
                        else:
                            # Test substitutes act as the pipeline boundary.
                            inference_started()
                        persist()
                        with tracked_local_call(config.output_dir, run_id,
                                f"{call['id']}:attempt:{number}", MODEL_ID,
                                scope='backend_request', metadata={'kind': call['kind'], 'seed': call['seed']}) as cost:
                            try:
                                image = backend.generate(call["prompt"], call["seed"], parent_path)
                            finally:
                                cost['measurements'] = {'pipeline_invoked': attempt['pipeline_invoked']}
                        # Once generation returned, downstream work must not trigger inference again.
                        attempt['phase'] = 'save_output'
                    attempt["prompt_token_count"] = getattr(backend, "last_prompt_token_count", None)
                    attempt["inference_measurements"] = measure.result
                    # Store model output before any downstream validation can fail.
                    if raw_target.exists():
                        raise FileExistsError(f'Refusing to overwrite retained model output: {raw_target.name}')
                    save_image(image, raw_target)
                    attempt.update(raw_output_saved=True, phase='register_output')
                    persist()
                    record = image_record(raw_target, image_id=call["id"], parent=parent if call["kind"] == "edit" else None, seed=call["seed"], origin="model_generation" if call["kind"] == "generate" else "model_edit")
                    attempt.update(raw_output_sha256=record['sha256'], phase='publish_output')
                    target = run_dir / f"{call['id']}.png"
                    publish_saved_image(raw_target, target)
                    record['path'] = target.name
                    manifest["images"].append(record)
                    call.update(status="succeeded", output_image_id=record["id"])
                    attempt.update(status="succeeded", phase="completed", output_sha256=record["sha256"])
                    manifest["budget"]["successful_image_calls"] += 1
                    if call["kind"] == "generate":
                        parent = record
                    if record["size"] != [config.width, config.height]:
                        # Do not silently resize or discard a real model output.
                        manifest.setdefault("output_warnings", []).append(f"{record['id']} returned size {record['size']}, requested {[config.width, config.height]}")
                    break
                except (Exception, KeyboardInterrupt) as exc:
                    failure(call["id"], exc, attempt, phase=attempt['phase'])
                    # A rename may have succeeded before its directory fsync failed.
                    attempt['raw_output_saved'] = raw_target.is_file()
                    call["status"] = attempt["status"]
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    if attempt['phase'] != 'inference':
                        attempt['status'] = call['status'] = 'postprocessing_failed'
                        break
                    if hasattr(backend, "recover_after_failure"):
                        try:
                            backend.recover_after_failure()
                        except Exception as recovery_error:
                            failure("recovery", recovery_error)
                    gc.collect()
                finally:
                    attempt["finished_at"] = utc_now()
                    attempt.setdefault("inference_measurements", measure.result)
                    persist()
    except (Exception, KeyboardInterrupt) as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        failure("experiment", exc)
    finally:
        for call in manifest["calls"]:
            if call["status"] in ("planned", "running"):
                call.update(status="skipped", reason="Run interrupted" if interrupted else "Experiment setup failed")
        if manifest["images"]:
            try:
                manifest["comparison"] = make_comparison(run_dir, manifest["images"], config.width, config.height)
            except Exception as exc:
                failure("comparison", exc)
        if backend and hasattr(backend, "close"):
            try:
                backend.close()
            except Exception as exc:
                failure("cleanup", exc)
        all_succeeded = all(c["status"] == "succeeded" for c in manifest["calls"])
        complete = all_succeeded and len(manifest["images"]) == 3 and manifest["comparison"] is not None
        manifest["status"] = "interrupted" if interrupted else "completed_with_failures" if complete and manifest["failures"] else "completed" if complete else "failed"
        manifest["finished_at"] = utc_now()
        manifest["runtime_seconds"] = time.perf_counter() - started
        persist()
    print(f"{manifest['status']}: {manifest_path}", flush=True)
    return manifest_path, 130 if interrupted else 0 if complete else 1


def main(argv: list[str] | None = None) -> int:
    config, check = parse_args(argv)
    if check:
        try:
            from .backend import inspect_environment
            # Import the complete selected pipeline and matching text/tokenizer classes.
            from diffusers import Flux2KleinPipeline
            from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
            result = inspect_environment()
            result["required_imports"] = [cls.__name__ for cls in (Flux2KleinPipeline, Qwen3ForCausalLM, Qwen2TokenizerFast)]
            result["dependencies"] = dependencies()
            print(json.dumps(result, indent=2))
            return 0
        except Exception as exc:
            print(f"Environment check failed: {exc}. Run `uv sync --locked --dev` using Python 3.11.", file=sys.stderr)
            return 2
    try:
        with handle_termination():
            _, code = run_experiment(config)
            return code
    except (OSError, ImportError) as exc:
        print(f"Unable to start the experiment: {exc}. Check output permissions and run `uv sync --locked --dev`.", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
