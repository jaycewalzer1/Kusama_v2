"""One frozen Qwen direction, one exact FLUX edit, then a pending human review."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import psutil
import time
import uuid

from PIL import Image

from .artifacts import import_parent, save_image, sha256, write_json
from .backend import FluxBackend
from .config import Config, parse_args
from .costs import tracked_local_call
from .controlled import load_plan, run as run_edit
from .director import DirectorConfig, QwenDirector, validate_proposal
from .experiment import Measurement, code_provenance, handle_termination, publish_saved_image, utc_now


def propose(parent: Path, intention: str, config: Config, director_config: DirectorConfig,
            *, seed: int = 23, director_factory=QwenDirector) -> Path:
    if not isinstance(intention, str) or not intention.strip() or len(intention) > 4000:
        raise ValueError('Intention must contain 1–4000 characters')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('Seed must be an integer in [0, 2**63)')
    parent = Path(parent).resolve()
    with Image.open(parent) as image:
        if any(d < 64 or d % 16 for d in image.size):
            raise ValueError('Parent dimensions must be multiples of 16 and at least 64')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_qwen_' + uuid.uuid4().hex[:8]
    directory = config.output_dir.resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'kind': 'qwen_directed_edit', 'status': 'directing',
        'started_at': utc_now(), 'intention': intention.strip(), 'origin': 'model_proposal',
        'cost_accounting': {'root': str(config.output_dir.resolve()), 'run_id': run_id},
        'frozen_weights': True, 'training_performed': False, 'max_calls': 1, 'model_calls': 0,
        'director_calls': 0, 'max_director_calls': 1, 'visual_review': 'pending',
        'human_labels': [], 'model_observations': [], 'code': code_provenance(),
        'director_process': {'pid': psutil.Process().pid, 'created_at': psutil.Process().create_time()},
        'director_config': {k: str(v) if isinstance(v, Path) else v for k, v in asdict(director_config).items()}}
    started = time.monotonic()
    director = None

    def persist():
        manifest['runtime_seconds'] = time.monotonic() - started
        write_json(directory / 'manifest.json', manifest)

    print(f'Direction packet: {directory}', flush=True)
    persist()
    try:
        manifest['parent_source'] = import_parent(parent, directory / 'parent.png')
        with Image.open(directory / 'parent.png') as image:
            size = image.size
        manifest['parent_sha256'] = sha256(directory / 'parent.png')
        director = director_factory(director_config)

        def progress(settings):
            manifest['director_backend'] = settings
            persist()

        with tracked_local_call(config.output_dir, run_id, 'director:load',
                                'Qwen/Qwen3-VL-8B-Instruct', scope='model_loading'):
            director.prepare(on_progress=progress)
        # A failed/uncertain director call is retained, never automatically repeated.
        manifest['director_calls'] = 1
        persist()
        measurement = (Measurement(director.execution_settings['device'])
                       if director_factory is QwenDirector else None)
        try:
            with tracked_local_call(config.output_dir, run_id, 'director:0',
                                    'Qwen/Qwen3-VL-8B-Instruct') as cost:
                try:
                    with measurement if measurement is not None else nullcontext():
                        result = director.propose(directory / 'parent.png', intention.strip())
                finally:
                    if measurement is not None:
                        cost['measurements'] = measurement.result
                        cost['runtime_seconds'] = measurement.result.get('runtime_seconds')
        finally:
            if measurement is not None:
                manifest['director_measurement'] = measurement.result
        write_json(directory / 'direction.json', result)
        proposal = validate_proposal(result['proposal'])
        if result['input_image_sha256'] != manifest['parent_sha256']:
            raise ValueError('Director proposal is bound to a different parent')
        save_image(Image.new('L', size, 255), directory / 'allowed_mask.png')
        plan = {'parent': 'parent.png', 'parent_sha256': manifest['parent_sha256'],
            'instruction': proposal['edit_instruction'], 'target': proposal['specific_change'],
            'allowed_mask': 'allowed_mask.png', 'method': 'whole', 'seed': seed,
            'max_calls': 1, 'author': 'Qwen/Qwen3-VL-8B-Instruct (frozen director)',
            'visual_checks': [f"Did this change occur: {proposal['specific_change']}?",
                f"Did the intended effect improve: {proposal['expected_effect']}?",
                *[f'Was this preserved: {item}?' for item in proposal['preservation_constraints']],
                'Assess finished quality and development potential separately.']}
        write_json(directory / 'plan.json', plan)
        load_plan(directory / 'plan.json')
        manifest['packet_hashes'] = {name: sha256(directory / name) for name in
            ('parent.png', 'direction.json', 'allowed_mask.png', 'plan.json')}
        manifest.update(status='direction_ready', proposal=proposal,
            preservation_policy='Whole-image FLUX edit; artistic preservation constraints require visual review')
    except BaseException as exc:
        if getattr(exc, 'result', None) is not None:
            write_json(directory / 'direction.json', exc.result)
        manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'director_failed',
                        error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if director is not None:
            try:
                director.close()
            except Exception as exc:
                manifest['cleanup_error'] = f'{type(exc).__name__}: {exc}'
                manifest['status'] = 'director_cleanup_failed'
                persist()
                raise
        persist()
    return directory


def execute(directory: Path, config: Config | None = None, *, backend_factory=FluxBackend) -> Path:
    """Reuse the saved direction; ledger recovery prevents duplicate FLUX calls."""
    directory = Path(directory).resolve()
    with (directory / '.execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This direction is already executing') from exc
        manifest = json.loads((directory / 'manifest.json').read_text())
        required_files = {'parent.png', 'direction.json', 'allowed_mask.png', 'plan.json'}
        if (manifest.get('kind') != 'qwen_directed_edit'
                or not isinstance(manifest.get('packet_hashes'), dict)
                or set(manifest['packet_hashes']) != required_files):
            raise ValueError('A complete saved Qwen direction packet is required')
        for name, expected in manifest['packet_hashes'].items():
            if sha256(directory / name) != expected:
                raise ValueError(f'Saved direction packet changed: {name}')
        result = json.loads((directory / 'direction.json').read_text())
        proposal = validate_proposal(result['proposal'])
        plan, *_ = load_plan(directory / 'plan.json')
        if (plan['instruction'] != proposal['edit_instruction'] or plan['max_calls'] != 1
                or plan['method'] != 'whole' or plan.get('fallbacks')
                or result['input_image_sha256'] != plan['parent_sha256']):
            raise ValueError('Plan must execute the exact saved Qwen instruction once')
        if manifest.get('status') == 'director_cleanup_failed':
            origin = manifest.get('director_process')
            if not origin:
                raise ValueError('Cannot verify release after director cleanup failed')
            try:
                still_running = psutil.Process(origin['pid']).create_time() == origin['created_at']
            except psutil.NoSuchProcess:
                still_running = False
            if still_running:
                raise ValueError('Director cleanup failed; exit the original process before execution')
        child = directory / 'edit'
        if (child / 'job.json').exists() and config is not None:
            raise ValueError('Existing edit uses frozen settings; omit backend flags')
        manifest['status'] = 'editing'
        manifest.setdefault('director_runtime_seconds', manifest.get('runtime_seconds', 0))
        write_json(directory / 'manifest.json', manifest)
        try:
            if (child / 'job.json').exists():
                run_edit(resume_dir=child, backend_factory=backend_factory)
            else:
                run_edit(directory / 'plan.json', replace(config or Config(), output_dir=directory),
                         backend_factory=backend_factory, job_id='edit')
            edit_manifest = json.loads((child / 'manifest.json').read_text())
            manifest.update(status=edit_manifest['status'], model_calls=edit_manifest['model_calls'],
                flux_manifest='edit/manifest.json', flux_backend=edit_manifest['attempts'][0].get('backend')
                if edit_manifest['attempts'] else None)
            published = manifest.setdefault('published_artifacts', {})
            for source, target in (('attempt_0_result.png', 'edit.png'),
                                   ('attempt_0_comparison.png', 'comparison.png')):
                if not (child / source).exists():
                    continue
                if not (directory / target).exists():
                    publish_saved_image(child / source, directory / target)
                if sha256(directory / target) != sha256(child / source):
                    raise ValueError(f'Published artifact changed: {target}')
                published[target] = sha256(directory / target)
            manifest['review_path'] = 'edit/review.json' if (child / 'review.json').exists() else None
            manifest['finished_at'] = utc_now()
            manifest.pop('error', None)
        except BaseException as exc:
            manifest.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'execution_failed',
                            error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            # The controlled executor persists charged calls even on SIGTERM.
            # Keep the dashboard's outer record aligned after an interrupted call.
            if (child / 'manifest.json').exists():
                try:
                    child_state = json.loads((child / 'manifest.json').read_text())
                    manifest.update(model_calls=child_state['model_calls'],
                        flux_manifest='edit/manifest.json', flux_status=child_state['status'],
                        flux_runtime_seconds=child_state.get('runtime_seconds', 0))
                    manifest['runtime_seconds'] = (manifest['director_runtime_seconds']
                                                   + manifest['flux_runtime_seconds'])
                except (OSError, ValueError, KeyError) as exc:
                    manifest['child_state_error'] = str(exc)
            write_json(directory / 'manifest.json', manifest)
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        'New directions also require --parent-image PATH. Shared FLUX flags include '
        '--device, --offload, --vae-tiling, --cache-dir, --output-dir, --steps, '
        '--precision, and --local-files-only; see kusama-first-edit --help. '
        'The saved parent determines image dimensions.'))
    parser.add_argument('--intention')
    parser.add_argument('--propose-only', action='store_true')
    parser.add_argument('--execute', type=Path, help='Execute/recover a saved packet without calling Qwen')
    parser.add_argument('--seed', type=int, default=23)
    parser.add_argument('--director-device', choices=['auto', 'mps', 'cuda', 'cpu'], default='auto')
    parser.add_argument('--director-precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    parser.add_argument('--director-local-files-only', action='store_true')
    args, remaining = parser.parse_known_args(argv)
    if args.execute and (args.intention or args.propose_only):
        parser.error('--execute accepts a saved packet, without --intention or --propose-only')
    config, check = parse_args(remaining) if remaining or not args.execute else (None, False)
    if check:
        parser.error('Use kusama-first-edit --check-environment for hardware diagnostics')
    try:
        with handle_termination():
            if args.execute:
                directory = execute(args.execute, config)
            else:
                if not args.intention or config.parent_image is None:
                    parser.error('Supply --parent-image and --intention')
                director_config = DirectorConfig(cache_dir=config.cache_dir,
                    device=args.director_device, precision=args.director_precision,
                    local_files_only=args.director_local_files_only)
                directory = propose(config.parent_image, args.intention, config, director_config, seed=args.seed)
                if not args.propose_only:
                    execute(directory, config)
        print(directory, flush=True)
        status = json.loads((directory / 'manifest.json').read_text())['status']
        return 0 if status in ('direction_ready', 'needs_review') else 1
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'kusama-direct: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
