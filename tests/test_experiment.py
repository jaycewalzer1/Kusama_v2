"""Orchestration checks with synthetic fixtures; these are not model results."""

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from PIL import Image
import pytest

from kusama import backend as backend_module
from kusama import experiment as experiment_module
from kusama.config import Config, EDIT_INSTRUCTION, MODEL_REVISION, PARENT_PROMPT
from kusama.experiment import run_experiment
from kusama.costs import CostJournal


class FakeBackend:
    """Controlled test-only image producer that reloads every supplied input."""

    def __init__(self, config, *, fail_seeds=None, prepare_error=None, tamper_parent=False, preflight_fail_seeds=()):
        self.config = config
        self.fail_seeds = fail_seeds or {}
        self.prepare_error = prepare_error
        self.tamper_parent = tamper_parent
        self.preflight_fail_seeds = preflight_fail_seeds
        self.on_inference_start = None
        self.calls = []
        self.inputs = []
        self.counts = Counter()
        self.closed = False
        self.recoveries = 0
        self.execution_settings = {
            "device": "cpu", "precision": "fp32", "offload": "none",
            "width": config.width, "height": config.height,
            "num_inference_steps": config.steps, "guidance_scale": config.guidance,
            "max_sequence_length": config.max_sequence_length,
            "resolved_revision": config.model_revision,
            "test_fixture": True,
        }

    def prepare(self, on_progress):
        on_progress({"resolved_revision": self.config.model_revision, "status": "loading"})
        if self.prepare_error:
            raise self.prepare_error
        return {**self.execution_settings, "status": "ready"}

    def generate(self, prompt, seed, parent_path=None):
        self.calls.append((prompt, seed, parent_path))
        self.counts[seed] += 1
        if seed in self.preflight_fail_seeds:
            raise ValueError("Synthetic prompt preflight failure")
        if parent_path is not None:
            with Image.open(parent_path) as image:
                image.load()
                reloaded = image.convert("RGB")
            self.inputs.append({
                "path": parent_path, "image": reloaded,
                "pixels": reloaded.tobytes(),
                "sha256": hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            })
            # A pipeline may modify its input object. A later request must still
            # start from the persisted parent, rather than this modified object.
            reloaded.putpixel((0, 0), (255, 255, 255))
            if self.tamper_parent:
                Image.new("RGB", reloaded.size, "black").save(parent_path)
        if self.on_inference_start is not None:
            self.on_inference_start()
        if self.counts[seed] <= self.fail_seeds.get(seed, 0):
            raise RuntimeError(f"Synthetic failure for seed {seed}")
        return Image.new("RGB", (self.config.width, self.config.height), (seed, 10, 20))

    def recover_after_failure(self):
        self.recoveries += 1

    def close(self):
        self.closed = True


@pytest.fixture
def config(tmp_path):
    return Config(output_dir=tmp_path / "runs", cache_dir=tmp_path / "cache",
                  device="cpu", precision="fp32", width=64, height=64)


@pytest.fixture(autouse=True)
def lightweight_measurements(monkeypatch):
    """Exercise real timing/persistence without probing GPUs or loading weights."""
    monkeypatch.setattr(backend_module, "inspect_environment", lambda: {"preferred_device": "cpu", "test_fixture": True})
    monkeypatch.setattr(backend_module, "synchronize", lambda device: None)
    monkeypatch.setattr(backend_module, "reset_peak_memory", lambda device: None)
    monkeypatch.setattr(backend_module, "memory_snapshot", lambda device: {"process_rss_bytes": 1024, "device": device})


def run_fixture(config, **kwargs):
    backend = FakeBackend(config, **kwargs)
    manifest_path, code = run_experiment(config, backend_factory=lambda config: backend)
    return backend, manifest_path, json.loads(manifest_path.read_text()), code


def assert_readable_record(run_dir, record):
    path = run_dir / record["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    with Image.open(path) as image:
        image.load()
        assert image.format == "PNG"
        assert list(image.size) == record["size"]


def test_three_saved_outputs_share_one_independent_parent_and_exact_provenance(config):
    backend, manifest_path, manifest, code = run_fixture(config)
    assert code == 0
    assert manifest["status"] == "completed"
    assert manifest["experiment_kind"] == "development"
    assert manifest["model"]["requested_revision"] == MODEL_REVISION
    assert manifest["model"]["resolved_revision"] == MODEL_REVISION
    assert manifest["prompts"] == {"parent": PARENT_PROMPT, "edit_instruction": EDIT_INSTRUCTION}
    assert manifest["seeds"] == {"parent": 17, "edits": [23, 41]}
    assert manifest["budget"] == {
        "planned_image_calls": 3, "max_retries_per_call": 0,
        "maximum_image_attempts": 3, "actual_image_attempts": 3,
        "successful_image_calls": 3, "backend_request_attempts": 3,
    }
    assert [call[1] for call in backend.calls] == [17, 23, 41]
    assert backend.calls[0] == (PARENT_PROMPT, 17, None)
    assert all(call[0] == EDIT_INSTRUCTION for call in backend.calls[1:])
    parent, edit_a, edit_b = manifest["images"]
    assert [record["path"] for record in manifest["images"]] == ["parent.png", "edit_seed_23.png", "edit_seed_41.png"]
    for record in manifest["images"]:
        assert_readable_record(manifest_path.parent, record)
    assert parent["parent_image_id"] is None
    assert parent["seed"] == 17
    for record in (edit_a, edit_b):
        assert record["parent_image_id"] == parent["id"]
        assert record["parent_sha256"] == parent["sha256"]
    assert len(backend.inputs) == 2
    assert backend.inputs[0]["image"] is not backend.inputs[1]["image"]
    assert backend.inputs[0]["pixels"] == backend.inputs[1]["pixels"] == Image.new("RGB", (64, 64), (17, 10, 20)).tobytes()
    assert all(item["path"] == manifest_path.parent / "parent.png" for item in backend.inputs)
    assert all(item["sha256"] == parent["sha256"] for item in backend.inputs)
    calls = manifest["calls"]
    assert calls[1]["settings"] == calls[2]["settings"] == manifest["effective_settings"]
    assert all(call["parent_sha256"] == parent["sha256"] for call in calls[1:])
    assert all(call["input_image_path"] == "parent.png" for call in calls[1:])
    assert all(call["status"] == "succeeded" and len(call["attempts"]) == 1 for call in calls)
    for call in calls:
        measurement = call["attempts"][0]["inference_measurements"]
        assert measurement["runtime_seconds"] >= 0
        assert measurement["samples"] >= 2
        assert measurement["sampled_max_bytes"]["process_rss_bytes"] == 1024
    assert manifest["dependencies"]["pillow"]
    assert manifest["code_sha256"]["src/kusama/experiment.py"]
    assert manifest["runtime_seconds"] >= manifest["load_measurements"]["runtime_seconds"]
    assert manifest["finished_at"] >= manifest["started_at"]
    assert manifest["observations"]["human_labels"] == []
    assert manifest["observations"]["visual_review_status"] == "pending"
    assert manifest["failures"] == []
    assert manifest["comparison"]["partial"] is False
    assert_readable_record(manifest_path.parent, manifest["comparison"])
    assert backend.closed
    with CostJournal(config.output_dir, read_only=True) as costs:
        accounting = costs.summary(manifest['run_id'])
    assert accounting['calls'] == 4  # Model loading plus three backend requests.
    assert accounting['open_calls'] == 0
    assert accounting['currencies']['USD']['total'] is None
    assert all(record['finish']['outcome'] == 'succeeded' for record in accounting['records'])
    assert manifest['cost_accounting']['run_id'] == manifest['run_id']


def test_reuse_parent_runs_only_two_edits_and_preserves_original(config, tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "green").save(source)
    original_bytes = source.read_bytes()
    config = replace(config, parent_image=source)
    backend, manifest_path, manifest, code = run_fixture(config)
    assert code == 0
    assert [call[1] for call in backend.calls] == [23, 41]
    assert all(call[2] == manifest_path.parent / "parent.png" for call in backend.calls)
    assert manifest["budget"]["planned_image_calls"] == manifest["budget"]["actual_image_attempts"] == 2
    assert manifest["seeds"]["parent"] is None
    assert manifest["images"][0]["origin"] == "provided_image"
    assert manifest["prompts"]["parent"] is None  # Imported images have no known generation prompt.
    assert manifest["parent_source"]["sha256"] == hashlib.sha256(original_bytes).hexdigest()
    assert source.read_bytes() == original_bytes
    assert backend.inputs[0]["pixels"] == backend.inputs[1]["pixels"]
    assert manifest["images"][0]["seed"] is None


def test_one_failed_edit_preserves_parent_and_successful_sibling(config):
    backend, manifest_path, manifest, code = run_fixture(config, fail_seeds={23: 1})
    assert code == 1
    assert manifest["status"] == "failed"
    assert [call[1] for call in backend.calls] == [17, 23, 41]
    assert [call["status"] for call in manifest["calls"]] == ["succeeded", "failed", "succeeded"]
    assert [image["path"] for image in manifest["images"]] == ["parent.png", "edit_seed_41.png"]
    assert manifest["budget"]["actual_image_attempts"] == 3
    assert manifest["budget"]["successful_image_calls"] == 2
    assert manifest["failures"][0]["stage"] == "edit_seed_23"
    assert "Synthetic failure" in manifest["failures"][0]["message"]
    assert manifest["comparison"]["partial"] is True
    for record in [*manifest["images"], manifest["comparison"]]:
        assert_readable_record(manifest_path.parent, record)
    assert backend.closed


def test_retry_is_bounded_uses_same_seed_and_records_failed_attempt(config):
    config = replace(config, max_retries=1)
    backend, _, manifest, code = run_fixture(config, fail_seeds={23: 1})
    assert code == 0
    assert manifest["status"] == "completed_with_failures"
    assert [call[1] for call in backend.calls] == [17, 23, 23, 41]
    assert manifest["budget"]["maximum_image_attempts"] == 6
    assert manifest["budget"]["actual_image_attempts"] == 4
    assert [attempt["status"] for attempt in manifest["calls"][1]["attempts"]] == ["failed", "succeeded"]
    assert [attempt["number"] for attempt in manifest["calls"][1]["attempts"]] == [1, 2]
    assert manifest["calls"][1]["attempts"][0]["failure"] == manifest["failures"][0]
    assert len({item["sha256"] for item in backend.inputs}) == 1
    assert backend.recoveries == 1
    with CostJournal(config.output_dir, read_only=True) as costs:
        attempts = [record for record in costs.summary(manifest['run_id'])['records']
                    if record['call_id'].startswith('edit_seed_23:')]
    assert [record['finish']['outcome'] for record in attempts] == ['failed', 'succeeded']
    assert all(record['finish']['runtime_seconds'] >= 0 for record in attempts)


@pytest.mark.parametrize('failed_id,expected_seeds', [('parent', [17]), ('edit_seed_23', [17, 23, 41])])
def test_post_save_registration_failure_never_rerenders_or_overwrites_raw(config, monkeypatch, failed_id, expected_seeds):
    original_record = experiment_module.image_record
    retained = {}
    def failed_registration(path, **kwargs):
        if kwargs['image_id'] == failed_id:
            retained['path'], retained['bytes'] = path, path.read_bytes()
            raise OSError('Injected image registration failure')
        return original_record(path, **kwargs)
    monkeypatch.setattr(experiment_module, 'image_record', failed_registration)
    backend, manifest_path, manifest, code = run_fixture(replace(config, max_retries=2))
    assert code == 1
    assert [call[1] for call in backend.calls] == expected_seeds
    assert manifest['budget']['actual_image_attempts'] == len(expected_seeds)
    assert backend.recoveries == 0
    failed_call = next(call for call in manifest['calls'] if call['id'] == failed_id)
    assert failed_call['status'] == 'postprocessing_failed'
    assert len(failed_call['attempts']) == 1
    attempt = failed_call['attempts'][0]
    assert attempt['status'] == 'postprocessing_failed'
    assert attempt['failure']['phase'] == 'register_output'
    assert attempt['raw_output_saved'] is True
    assert retained['path'] == manifest_path.parent/attempt['raw_output_path']
    assert retained['path'].name == f'{failed_id}_attempt_1_raw.png'
    assert retained['path'].read_bytes() == retained['bytes']
    assert not (manifest_path.parent/f'{failed_id}.png').exists()
    with Image.open(retained['path']) as raw:
        assert raw.getpixel((0, 0)) == (failed_call['seed'], 10, 20)
    if failed_id == 'parent':
        assert [call['status'] for call in manifest['calls'][1:]] == ['skipped', 'skipped']
    else:
        assert manifest['calls'][2]['status'] == 'succeeded'
        assert manifest['comparison']['partial'] is True
        for image in manifest['images']:
            assert_readable_record(manifest_path.parent, image)


def test_save_failure_after_raw_rename_retains_bytes_without_retry(config, monkeypatch):
    original_save = experiment_module.save_image
    retained = {}
    def fail_after_save(image, path):
        original_save(image, path)
        if path.name == 'edit_seed_23_attempt_1_raw.png':
            retained['path'], retained['bytes'] = path, path.read_bytes()
            raise OSError('Injected directory fsync failure after rename')
    monkeypatch.setattr(experiment_module, 'save_image', fail_after_save)
    backend, _, manifest, code = run_fixture(replace(config, max_retries=2))
    assert code == 1
    assert [call[1] for call in backend.calls] == [17, 23, 41]
    assert retained['path'].read_bytes() == retained['bytes']
    attempt = manifest['calls'][1]['attempts'][0]
    assert attempt['status'] == 'postprocessing_failed'
    assert attempt['failure']['phase'] == 'save_output'
    assert attempt['raw_output_saved'] is True


def test_comparison_failure_retains_all_raw_and_canonical_outputs(config, monkeypatch):
    def failed_comparison(*args, **kwargs):
        raise OSError('Injected comparison failure')
    monkeypatch.setattr(experiment_module, 'make_comparison', failed_comparison)
    backend, manifest_path, manifest, code = run_fixture(replace(config, max_retries=2))
    assert code == 1
    assert [call[1] for call in backend.calls] == [17, 23, 41]
    assert manifest['budget']['actual_image_attempts'] == 3
    assert manifest['failures'][-1]['stage'] == 'comparison'
    for call, record in zip(manifest['calls'], manifest['images'], strict=True):
        assert call['status'] == 'succeeded'
        assert len(call['attempts']) == 1
        attempt = call['attempts'][0]
        raw = manifest_path.parent/attempt['raw_output_path']
        canonical = manifest_path.parent/record['path']
        assert raw.read_bytes() == canonical.read_bytes()
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == attempt['raw_output_sha256'] == record['sha256']


def test_canonical_parent_changes_do_not_change_retained_raw(config):
    _, manifest_path, manifest, _ = run_fixture(config, tamper_parent=True)
    attempt = manifest['calls'][0]['attempts'][0]
    raw = manifest_path.parent/attempt['raw_output_path']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == attempt['raw_output_sha256']
    assert raw.read_bytes() != (manifest_path.parent/'parent.png').read_bytes()


def test_parent_failure_exhausts_retry_bound_then_skips_both_edits(config):
    config = replace(config, max_retries=2)
    backend, manifest_path, manifest, code = run_fixture(config, fail_seeds={17: 99})
    assert code == 1
    assert [call[1] for call in backend.calls] == [17, 17, 17]
    assert [call["status"] for call in manifest["calls"]] == ["failed", "skipped", "skipped"]
    assert manifest["budget"]["actual_image_attempts"] == 3
    assert len(manifest["failures"]) == 3
    assert manifest["images"] == []
    assert manifest["comparison"] is None
    assert not list(manifest_path.parent.glob("*.png"))


@pytest.mark.parametrize("parent_kind", ["missing", "invalid"])
def test_invalid_parent_fails_before_backend_construction(config, tmp_path, parent_kind):
    source = tmp_path / "input.png"
    if parent_kind == "invalid":
        source.write_text("This is not an image.")
    invoked = []
    manifest_path, code = run_experiment(replace(config, parent_image=source), backend_factory=lambda cfg: invoked.append(cfg))
    manifest = json.loads(manifest_path.read_text())
    assert code == 1
    assert invoked == []
    assert manifest["status"] == "failed"
    assert manifest["budget"]["actual_image_attempts"] == 0
    assert manifest["images"] == []
    assert all(call["status"] == "skipped" for call in manifest["calls"])
    assert manifest["failures"][0]["stage"] == "experiment"


def test_model_setup_failure_preserves_supplied_parent(config, tmp_path):
    source = tmp_path / "parent.png"
    Image.new("RGB", (64, 64), "purple").save(source)
    backend, manifest_path, manifest, code = run_fixture(replace(config, parent_image=source), prepare_error=RuntimeError("No model access"))
    assert code == 1
    assert backend.calls == []
    assert backend.closed
    assert len(manifest["images"]) == 1
    assert_readable_record(manifest_path.parent, manifest["images"][0])
    assert manifest["model"]["resolved_revision"] == MODEL_REVISION
    assert manifest["failures"][0]["message"] == "No model access"
    assert manifest["comparison"]["partial"] is True


def test_parent_hash_change_blocks_second_edit_before_pipeline(config):
    backend, _, manifest, code = run_fixture(config, tamper_parent=True)
    assert code == 1
    assert [call[1] for call in backend.calls] == [17, 23]
    assert manifest["budget"]["actual_image_attempts"] == 2
    final_attempt = manifest["calls"][2]["attempts"][0]
    assert final_attempt["pipeline_invoked"] is False
    assert "Saved parent hash changed" in final_attempt["failure"]["message"]


def test_backend_preflight_failure_does_not_count_as_image_inference(config):
    backend, _, manifest, code = run_fixture(config, preflight_fail_seeds=(23,))
    assert code == 1
    assert [call[1] for call in backend.calls] == [17, 23, 41]
    assert manifest["budget"]["backend_request_attempts"] == 3
    assert manifest["budget"]["actual_image_attempts"] == 2
    assert manifest["calls"][1]["attempts"][0]["pipeline_invoked"] is False
    assert manifest["calls"][2]["status"] == "succeeded"


def test_repeated_runs_use_unique_directories_and_do_not_overwrite(config):
    _, first_path, first, _ = run_fixture(config)
    previous = {path.name: path.read_bytes() for path in first_path.parent.iterdir()}
    _, second_path, second, _ = run_fixture(config)
    assert first_path.parent != second_path.parent
    assert first["run_id"] != second["run_id"]
    assert {path.name: path.read_bytes() for path in first_path.parent.iterdir()} == previous
