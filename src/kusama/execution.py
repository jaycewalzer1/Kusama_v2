"""Restart-safe controlled edits; SQLite is authoritative, files are evidence."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
import uuid

from PIL import Image, ImageChops, ImageDraw

from .artifacts import save_image, sha256, write_json
from .backend import FluxBackend
from .config import Config, MODEL_ID
from .costs import tracked_local_call
from .experiment import code_provenance, dependencies, Measurement
from .ledger import BudgetExceeded, Ledger
from .regions import region_masks


def restore_config(record):
    values = dict(record)
    for key in ('cache_dir', 'output_dir', 'parent_image'):
        if values.get(key) is not None:
            values[key] = Path(values[key])
    values['edit_seeds'] = tuple(values['edit_seeds'])
    return Config(**values)


def canonical_config(config):
    return replace(config, cache_dir=config.cache_dir.resolve(), output_dir=config.output_dir.resolve(),
                   parent_image=config.parent_image.resolve() if config.parent_image else None)


def verify_artifacts(ledger, job_id):
    directory = Path(ledger.get_job(job_id)['directory'])
    for record in ledger.artifacts(job_id):
        target = directory / record['path']
        if not target.is_file() or sha256(target) != record['sha256']:
            raise ValueError(f'Immutable artifact missing or changed: {target}')


def _save_once(image, path):
    if path.exists():
        with Image.open(path) as saved:
            saved.load()
            if saved.mode != image.mode or saved.size != image.size or saved.tobytes() != image.tobytes():
                raise ValueError(f'Existing generated image differs from frozen computation: {path}')
    else:
        save_image(image, path)


def _save_json_once(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing snapshot differs from frozen source: {path}')
    else:
        write_json(path, value)


def _copy_snapshot_file(source, target, expected_sha):
    """Preserve the approved file bytes, including PNG encoding and metadata."""
    if sha256(source) != expected_sha:
        raise ValueError('Source parent changed before snapshot copying')
    if target.exists():
        if sha256(target) != expected_sha:
            raise ValueError('Existing parent snapshot differs from the approved source')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='wb', dir=target.parent,
                prefix='.parent-snapshot-', delete=False) as output:
            temporary = Path(output.name)
            with source.open('rb') as original:
                shutil.copyfileobj(original, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256(temporary) != expected_sha or sha256(source) != expected_sha:
            raise ValueError('Source parent changed during snapshot copying')
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_source(plan_path):
    from .controlled import load_plan
    plan = json.loads(plan_path.read_text())
    hashes = {key: sha256(plan_path.parent / plan[key])
              for key in ('parent', 'allowed_mask', 'object_mask', 'secondary_mask') if plan.get(key)}
    loaded = load_plan(plan_path)
    if plan != loaded[0] or any(sha256(plan_path.parent / plan[key]) != value for key, value in hashes.items()):
        raise ValueError('Plan or source changed during preflight')
    return loaded, hashes


def _snapshot(directory, source_path, loaded):
    from .controlled import binary_mask
    plan, parent, allowed, obj, cutout, crop, methods = loaded
    execution_plan = dict(plan, parent='parent.png', allowed_mask='allowed_mask.png')
    _copy_snapshot_file(source_path.parent / plan['parent'], directory / 'parent.png', plan['parent_sha256'])
    _save_once(allowed, directory / 'allowed_mask.png')
    if obj is not None:
        _save_once(obj, directory / 'object_mask.png')
        execution_plan['object_mask'] = 'object_mask.png'
    if plan.get('secondary_mask'):
        _save_once(binary_mask(source_path.parent / plan['secondary_mask'], parent.size), directory / 'secondary_mask.png')
        execution_plan['secondary_mask'] = 'secondary_mask.png'
    _save_json_once(directory / 'plan.json', plan)
    _save_json_once(directory / 'execution_plan.json', execution_plan)


def _finish(directory, index, attempt, parent, allowed, obj, cutout, masks, plan, config,
            registered_verifications=None):
    from . import controlled as edits
    l, t, r, b = attempt['crop']
    with Image.open(directory / f'attempt_{index}_raw.png') as saved:
        saved.load()
        if saved.size != (r-l, b-t) or saved.mode != 'RGB':
            attempt.update(status='failed', error='Unexpected model output dimensions or mode; raw output retained')
            return
        raw = saved.copy()
    method = attempt['method']
    expanded = parent.copy()
    expanded.paste(raw, (l, t))
    attempt['raw_preservation'] = edits.preservation(parent, expanded, allowed)
    blend = masks['blend'] if method == 'geometric' else masks['local']
    result = expanded if method == 'whole' else Image.composite(expanded, parent, blend)
    if method == 'geometric':
        result = Image.alpha_composite(result.convert('RGBA'), cutout).convert('RGB')
        attempt['geometry'] = {'status': 'construction_verified', 'scale': plan['scale'],
            'anchor': plan['anchor'], 'original_mask_bbox': obj.getbbox(),
            'transformed_alpha_bbox': cutout.getchannel('A').getbbox(),
            'transformed_core_bbox': cutout.getchannel('A').point(lambda x: 255 if x >= 128 else 0).getbbox(),
            'limitation': 'Supplied cutout transform, not semantic segmentation or background repair quality'}
    elif cutout is not None:
        attempt['geometry'] = {'status': 'requires_independent_measurement'}
    target = directory / f'attempt_{index}_result.png'
    _save_once(result, target)
    # A retained result is the evidence used for every resumed downstream check.
    with Image.open(target) as image:
        result = image.copy()
    comparison = Image.new('RGB', (parent.width * 2, parent.height + 40), 'white')
    comparison.paste(parent, (0, 40)); comparison.paste(result, (parent.width, 40))
    labels = ImageDraw.Draw(comparison)
    labels.text((12, 10), 'Original', fill='black')
    labels.text((parent.width + 12, 10), f"{method} - seed {plan['seed']} - pending review", fill='black')
    _save_once(comparison, directory / f'attempt_{index}_comparison.png')
    _save_once(ImageChops.difference(parent, result), directory / f'attempt_{index}_difference.png')
    attempt['preservation'] = edits.preservation(parent, result, allowed)
    attempt['effective_preservation'] = edits.preservation(parent, result, masks['effective'] if method == 'geometric' else masks['local'])
    attempt['repair_metrics'] = edits.repair_metrics(parent, result, masks['repair'])
    verification_path = directory / f'attempt_{index}_verification.json'
    if registered_verifications is None:
        registered_verifications = set()
    if verification_path.exists() and verification_path.name in registered_verifications:
        verification = json.loads(verification_path.read_text())
        if verification['image_sha256'] != sha256(target):
            raise ValueError('Saved verification is bound to another image')
    else:
        try:
            verification, overlay = edits.verify(parent, result, obj, plan, config)
        except Exception as exc:
            verification, overlay = {'status': 'uncertain', 'error': f'{type(exc).__name__}: {exc}'}, None
        verification['image_sha256'] = sha256(target)
        if overlay is not None:
            _save_once(overlay, directory / f'attempt_{index}_detections.png')
        write_json(verification_path, verification)
    registered_verifications.add(verification_path.name)
    attempt['verification'] = verification
    attempt['nonempty_edit'] = attempt['preservation'].get('total_changed_pixels', 0) > 0
    attempt['status'] = 'needs_review' if (attempt['preservation']['passed'] and attempt['effective_preservation']['passed']
        and attempt['nonempty_edit'] and verification['status'] != 'fail') else 'rejected'


def run_job(plan_path=None, config=None, backend_factory=FluxBackend, call_guard=None, **kwargs):
    ledger = kwargs.pop('ledger', None)
    if ledger is not None:
        return _run_job(plan_path, config, backend_factory, call_guard, ledger=ledger, **kwargs)
    resume = kwargs.get('resume_dir')
    if resume is not None:
        database = Path(json.loads((Path(resume) / 'job.json').read_text())['database'])
    elif config is not None:
        database = config.output_dir / 'jobs.sqlite3'
    else:
        raise ValueError('Supply a plan and configuration, or resume_dir')
    with Ledger(database) as owned:
        return _run_job(plan_path, config, backend_factory, call_guard, ledger=owned, **kwargs)


def _run_job(plan_path=None, config=None, backend_factory=FluxBackend, call_guard=None, *,
            resume_dir=None, job_id=None, ledger=None, parent_job_id=None):
    """Resume only saved computation. A charged attempt without bytes is uncertain."""
    from .controlled import load_plan, binary_mask
    if resume_dir is not None:
        directory = Path(resume_dir).resolve()
        descriptor = json.loads((directory / 'job.json').read_text())
        ledger = ledger or Ledger(Path(descriptor['database']))
        job_id = descriptor['job_id']
        job = ledger.get_job(job_id)
        if Path(job['directory']) != directory:
            raise ValueError('Resume directory does not match registered job')
        stored = restore_config(job['spec']['config'])
        if config is not None and canonical_config(config).to_dict() != stored.to_dict():
            raise ValueError('Resume uses frozen configuration; start a new job to change settings')
        config = stored
        if plan_path is not None:
            loaded, hashes = _read_source(Path(plan_path).resolve())
            if loaded[0] != job['spec']['plan'] or hashes != job['spec']['source_hashes']:
                raise ValueError('Resume plan differs from frozen job')
    else:
        if plan_path is None or config is None:
            raise ValueError('A new job requires a plan and configuration')
        config = canonical_config(config)
        plan_path = Path(plan_path).resolve()
        loaded, hashes = _read_source(plan_path)
        job_id = job_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_controlled_' + uuid.uuid4().hex[:8]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', job_id):
            raise ValueError('job-id must contain 1–128 letters, digits, underscores or hyphens')
        directory = config.output_dir / job_id
        ledger = ledger or Ledger(config.output_dir / 'jobs.sqlite3')
        spec = {'schema_version': 1, 'kind': 'controlled_edit', 'plan': loaded[0], 'config': config.to_dict(),
                'source_plan': str(plan_path), 'source_hashes': hashes}
        job = ledger.create_job(job_id, spec, loaded[0]['max_calls'], directory, parent_id=parent_job_id)
    with ledger.lock(job_id):
        directory.mkdir(parents=True, exist_ok=True)
        if not (directory / 'job.json').exists():
            write_json(directory / 'job.json', {'schema_version': 1, 'job_id': job_id,
                'database': str(ledger.path)})
        verify_artifacts(ledger, job_id)
        job = ledger.get_job(job_id)
        if not job['payload'].get('snapshot_complete'):
            source = Path(job['spec']['source_plan'])
            loaded, hashes = _read_source(source)
            if loaded[0] != job['spec']['plan'] or hashes != job['spec']['source_hashes']:
                raise ValueError('Original source changed before snapshot completed')
            _snapshot(directory, source, loaded)
            for name in ('job.json', 'plan.json', 'execution_plan.json', 'parent.png', 'allowed_mask.png', 'object_mask.png', 'secondary_mask.png'):
                if (directory / name).exists():
                    ledger.register_artifact(job_id, name)
            ledger.update_job(job_id, payload={'snapshot_complete': True})
        plan, parent, allowed, obj, cutout, crop, methods = load_plan(directory / 'execution_plan.json')
        secondary = binary_mask(directory / plan['secondary_mask'], parent.size) if plan.get('secondary_mask') else None
        masks = region_masks(allowed, obj, cutout, plan.get('repair_policy'), secondary)
        for name, mask in masks.items():
            _save_once(mask, directory / f'region_{name}.png')
        if cutout is not None:
            _save_once(cutout, directory / 'transformed_object.png')
        job = ledger.get_job(job_id)
        manifest = job['payload'].get('manifest') or {'schema_version': 3, 'job_id': job_id,
            'cost_accounting': {'root': str(ledger.path.parent), 'run_id': job_id},
            'development_experiment': True, 'status': 'running', 'plan_sha256': sha256(directory / 'plan.json'),
            'source_parent_sha256': plan['parent_sha256'], 'router': 'explicit plan' if plan['method'] != 'auto' else 'deterministic capability policy v1',
            'methods': methods, 'config': config.to_dict(), 'dependencies': dependencies(), 'code': code_provenance(),
            'model_calls': 0, 'attempts': [], 'model_observations': [], 'human_labels': [], 'visual_review': 'pending'}
        manifest['attempts'] = [attempt for index in range(len(methods)) if (attempt := ledger.get_attempt(job_id, index)) is not None]
        verified_paths = {record['path'] for record in ledger.artifacts(job_id)
                          if re.fullmatch(r'attempt_\d+_verification.json', record['path'])}

        def register_raw(index, target):
            recorded = ledger.get_attempt(job_id, index)
            if recorded is None or not recorded['reserved']:
                raise ValueError(f'Raw model output has no inference reservation: {target}')
            ledger.register_artifact(job_id, target.name, attempt_index=index, kind='raw_model_output')

        def persist():
            for target in sorted(directory.glob('*.png')):
                raw_match = re.fullmatch(r'attempt_(\d+)_raw.png', target.name)
                if raw_match:
                    register_raw(int(raw_match[1]), target)
                else:
                    ledger.register_artifact(job_id, target.name)
            for name in sorted(verified_paths):
                ledger.register_artifact(job_id, name)
            manifest['artifacts'] = {record['path']: record['sha256'] for record in ledger.artifacts(job_id) if record['path'].endswith('.png')}
            manifest['model_calls'] = ledger.call_count(job_id)
            for index, attempt in enumerate(manifest['attempts']):
                ledger.save_attempt(job_id, index, attempt, status=attempt['status'])
            ledger.update_job(job_id, status=manifest['status'], payload={'manifest': manifest})
            write_json(directory / 'manifest.json', manifest)
            ledger.export_events(directory / 'events.jsonl', job_id=job_id, recover=True)

        if manifest['status'] in ('needs_review', 'rejected', 'budget_exhausted', 'uncertain'):
            persist()
            return directory
        manifest['status'] = 'running'
        manifest.setdefault('sessions', []).append({'started_at': datetime.now(timezone.utc).isoformat(), 'code': code_provenance()})
        started = time.monotonic()
        persist()
        try:
            for index, method in enumerate(methods):
                attempt = manifest['attempts'][index] if index < len(manifest['attempts']) else None
                if attempt and attempt['status'] in ('rejected', 'failed'):
                    continue
                if attempt and attempt['status'] == 'needs_review':
                    manifest.update(status='needs_review', candidate=f'attempt_{index}_result.png')
                    break
                raw_path = directory / f'attempt_{index}_raw.png'
                charged = bool(attempt and ledger.get_attempt(job_id, index).get('reserved'))
                if charged and not raw_path.exists():
                    attempt.update(status='uncertain', reason='Reserved inference has no saved output; no automatic rerun')
                    manifest['status'] = 'uncertain'
                    break
                if not raw_path.exists() and ledger.call_count(job_id) >= plan['max_calls']:
                    manifest['status'] = 'budget_exhausted'
                    break
                if attempt is None:
                    attempt = {'method': method, 'seed': plan['seed'], 'status': 'running', 'pipeline_invoked': False,
                               'cost_call_id': f'attempt:{index}'}
                    manifest['attempts'].append(attempt)
                attempt['status'] = 'running'
                region = [0, 0, *parent.size] if method == 'whole' else crop
                l, t, r, b = region
                attempt['crop'] = region
                attempt['prompt'] = plan['repair_prompt'] if method == 'geometric' else plan['instruction']
                backend = None
                phase = 'preparation'
                attempt_start = time.monotonic()
                persist()
                try:
                    if not raw_path.exists():
                        source_path = directory / f'attempt_{index}_input.png'
                        _save_once(parent.crop(region), source_path)
                        active_mask = masks['repair'] if method == 'geometric' else masks['local']
                        mask_path = directory / f'attempt_{index}_mask.png'
                        _save_once(active_mask.crop(region), mask_path)
                        attempt['active_mask_pixels'] = active_mask.histogram()[255]
                        persist()
                        backend = backend_factory(replace(config, width=r-l, height=b-t))
                        backend.use_inpainting = method != 'whole'
                        def progress(settings):
                            attempt['backend'] = settings
                            persist()
                        with tracked_local_call(ledger.path.parent, job_id,
                                f"attempt:{index}:load:session:{len(manifest['sessions'])}",
                                MODEL_ID, scope='model_loading'):
                            backend.prepare(on_progress=progress)
                        def reserve_call():
                            if call_guard is not None:
                                call_guard()
                            if not ledger.reserve_call(job_id, index):
                                raise RuntimeError('Attempt already reserved; refusing duplicate inference')
                            attempt['pipeline_invoked'] = True
                            persist()
                        backend.on_inference_start = reserve_call
                        measurement = Measurement(backend.execution_settings['device']) if backend_factory is FluxBackend else None
                        phase = 'inference'
                        try:
                            with tracked_local_call(ledger.path.parent, job_id, f'attempt:{index}', MODEL_ID,
                                                    scope='backend_request') as cost:
                                try:
                                    with measurement if measurement is not None else nullcontext():
                                        raw = backend.generate(attempt['prompt'], plan['seed'], source_path,
                                            **({} if method == 'whole' else {'mask_path': mask_path}))
                                finally:
                                    cost['measurements'] = {'pipeline_invoked': attempt['pipeline_invoked']}
                                    if measurement is not None:
                                        cost['measurements'].update(measurement.result)
                                        cost['runtime_seconds'] = measurement.result.get('runtime_seconds')
                        finally:
                            if measurement is not None:
                                attempt['measurement'] = measurement.result
                        phase = 'postprocessing'
                        save_image(raw, raw_path)
                        # Commit bytes before cleanup, verification, comparisons or labels.
                        register_raw(index, raw_path)
                        ledger.save_attempt(job_id, index, attempt, checkpoint={'raw_saved': True})
                        persist()
                        backend.close()
                        backend = None
                    else:
                        phase = 'postprocessing'
                        register_raw(index, raw_path)
                        attempt['recovered_saved_output'] = True
                    _finish(directory, index, attempt, parent, allowed, obj, cutout, masks, plan, config, verified_paths)
                    if attempt['status'] == 'needs_review':
                        manifest.update(status='needs_review', candidate=f'attempt_{index}_result.png')
                        if not (directory / 'review.json').exists():
                            write_json(directory / 'review.json', {'image_sha256': sha256(directory / manifest['candidate']),
                                'status': 'pending', 'geometry': attempt.get('geometry', {'status': 'not_requested'}),
                                'independent_verification': attempt['verification'], 'questions': plan['visual_checks'],
                                'model_observations': [], 'human_labels': []})
                        break
                except BudgetExceeded as exc:
                    attempt.update(status='budget_exhausted', error=f'{type(exc).__name__}: {exc}')
                    manifest['status'] = 'budget_exhausted'
                    break
                except Exception as exc:
                    attempt.update(status='postprocessing_failed' if phase == 'postprocessing' else 'failed',
                        error=f'{type(exc).__name__}: {exc}')
                    if phase == 'postprocessing':
                        manifest['status'] = 'postprocessing_failed'
                        break
                except BaseException as exc:
                    attempt.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
                    raise
                finally:
                    attempt['runtime_seconds'] = attempt.get('runtime_seconds', 0) + time.monotonic() - attempt_start
                    if backend is not None:
                        try:
                            backend.close()
                        except Exception as exc:
                            attempt['cleanup_error'] = str(exc)
                    persist()
            if manifest['status'] == 'running':
                manifest['status'] = 'rejected'
        except BaseException:
            manifest['status'] = 'interrupted'
            raise
        finally:
            elapsed = time.monotonic() - started
            manifest['runtime_seconds'] = manifest.get('runtime_seconds', 0) + elapsed
            manifest['sessions'][-1]['runtime_seconds'] = elapsed
            persist()
        return directory
