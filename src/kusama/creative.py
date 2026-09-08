"""Resumable brief → reference-informed Qwen directions → FLUX → human review.

Both pretrained models remain frozen. SQLite reservations prevent repeat model
calls after a crash; raw responses and source bytes remain inspectable artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
import uuid

from .artifacts import image_record, save_image, sha256, write_json
from .backend import FluxBackend
from .config import Config, MODEL_ID as FLUX_MODEL_ID
from .comparison_design import diagnostic_pairs, design_metadata
from .costs import CostJournal, tracked_local_call
from .director import (DirectorConfig, QwenDirector, MODEL_ID as QWEN_MODEL_ID,
                       DirectorOutputError, validate_directions, validate_observation, _unique_object,
                       MAX_RESPONSE_CHARACTERS)
from .execution import canonical_config, restore_config
from .experiment import Measurement, code_provenance, handle_termination, utc_now
from .ledger import Ledger
from .references import ReferencePacket, load_packet, snapshot_packet, selected_challenges
from .reviews import ReviewStore, ReviewNotFound


class UncertainInvocation(RuntimeError):
    """A reserved call has no saved output and must not run automatically again."""


def _settings(config):
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _save_json_evidence(ledger, job_id, directory, name, value, *, index=None):
    target = directory / name
    if target.exists():
        if json.loads(target.read_text()) != value:
            raise ValueError(f'Saved evidence differs: {name}')
    else:
        write_json(target, value)
    ledger.register_artifact(job_id, name, attempt_index=index, kind='model_response' if index is not None else 'snapshot')


def _new_spec(packet_path, config, director_config, directions, realizations, seed, observe, batch_size=1, note=''):
    if type(directions) is not int or not 1 <= directions <= 12:
        raise ValueError('Choose 1–12 directions')
    if realizations != 2:
        raise ValueError('This experiment requires exactly two realizations per direction')
    if type(batch_size) is not int or not 1 <= batch_size <= 3:
        raise ValueError('Choose a Qwen batch size of 1–3; one direction per call is the default')
    if not isinstance(note, str) or len(note) > 2000:
        raise ValueError('Experiment note must be text of at most 2000 characters')
    if type(seed) is not int or not 0 <= seed < 2**63 - directions * realizations:
        raise ValueError('Base seed must leave room for every unique realization seed')
    if (any(type(n) is not int or n < 64 or n % 16 for n in [config.width, config.height])
            or type(config.steps) is not int or config.steps < 1
            or not math.isfinite(config.guidance) or config.guidance < 1):
        raise ValueError('Invalid generation dimensions, steps or guidance')
    if config.parent_image is not None:
        raise ValueError('Initial creative generation has no parent image; use an edit workflow after human selection')
    if config.max_retries:
        raise ValueError('Creative calls have no automatic retries; use max_retries=0')
    packet, hashes = load_packet(packet_path)
    selected_challenges(packet, [f'd{i + 1:02d}' for i in range(directions)])
    return {'schema_version': 1, 'kind': 'reference_creative_experiment',
            'source_packet': str(Path(packet_path).resolve()), 'source_hashes': hashes,
            'packet': packet, 'config': config.to_dict(), 'director_config': _settings(director_config),
            'direction_count': directions, 'realizations': realizations,
            'base_seed': seed, 'observe': bool(observe), 'batch_size': batch_size, 'note': note}


def _stored_director_config(values):
    return DirectorConfig(**{**values, 'cache_dir': Path(values['cache_dir'])})


def run(packet_path=None, config=None, director_config=None, *, directions=12, realizations=2,
        base_seed=11001, observe=False, stage='all', resume_dir=None, job_id=None,
        revalidate_saved=False, batch_size=1, note='', retry_rejected_observers=0,
        continue_rejected_observers=False,
        director_factory=QwenDirector, backend_factory=FluxBackend):
    """Public CLI uses real frozen models; factories are solely for offline tests."""
    if stage not in {'all', 'direct', 'render', 'observe'}:
        raise ValueError('stage must be all, direct, render or observe')
    if revalidate_saved and (resume_dir is None or stage not in {'all', 'direct', 'observe'}):
        raise ValueError('Saved-response revalidation requires --resume with direct, observe or all')
    if type(retry_rejected_observers) is not int or retry_rejected_observers not in (0, 1):
        raise ValueError('Observer retries must be 0 or 1 per rejected pair')
    if retry_rejected_observers and (resume_dir is None or stage not in {'all', 'observe'}):
        raise ValueError('Observer retries require --resume with observe or all')
    if type(continue_rejected_observers) is not bool:
        raise ValueError('continue_rejected_observers must be boolean')
    if continue_rejected_observers and (resume_dir is None or stage not in {'all', 'observe'}):
        raise ValueError('Continuing rejected observers requires --resume with observe or all')
    if resume_dir is not None:
        directory = Path(resume_dir).resolve()
        descriptor = json.loads((directory / 'job.json').read_text())
        if packet_path is not None or config is not None or director_config is not None:
            raise ValueError('Resume uses frozen inputs and settings; omit packet and configuration overrides')
        ledger_path, job_id = Path(descriptor['database']), descriptor['job_id']
    else:
        if packet_path is None or config is None:
            raise ValueError('A new experiment requires a reference packet and generation configuration')
        if stage not in {'all', 'direct'}:
            raise ValueError('Start with direct/all; render/observe require --resume')
        config = canonical_config(config)
        director_config = director_config or DirectorConfig(config.cache_dir, device=config.device,
            precision=config.precision, local_files_only=config.local_files_only,
            max_new_tokens=2048, max_image_pixels=262144)
        job_id = job_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_creative_' + uuid.uuid4().hex[:8]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', job_id):
            raise ValueError('Job ID requires 1–100 letters, digits, underscores or hyphens')
        directory = config.output_dir / job_id
        ledger_path = config.output_dir / 'jobs.sqlite3'
    with Ledger(ledger_path) as ledger:
        if resume_dir is None:
            spec = _new_spec(packet_path, config, director_config, directions, realizations, base_seed, observe, batch_size, note)
            batch_count = math.ceil(directions / spec['batch_size'])
            budget = batch_count + directions * realizations + (directions if observe else 0)
            ledger.create_job(job_id, spec, budget, directory)
        job = ledger.get_job(job_id)
        if Path(job['directory']) != directory or job['spec']['kind'] != 'reference_creative_experiment':
            raise ValueError('Resume directory or experiment kind does not match registered job')
        spec = job['spec']
        config = restore_config(spec['config'])
        director_config = _stored_director_config(spec['director_config'])
        with ledger.lock(job_id), handle_termination():
            directory.mkdir(parents=True, exist_ok=True)
            _save_json_evidence(ledger, job_id, directory, 'job.json',
                {'schema_version': 1, 'job_id': job_id, 'database': str(ledger.path)})
            ledger.verify_artifacts(job_id)
            if not job['payload'].get('snapshot_complete'):
                packet = snapshot_packet(Path(spec['source_packet']), directory, spec['source_hashes'])
                for target in [directory / 'source_packet.json', directory / 'packet.json', *sorted((directory / 'references').iterdir())]:
                    ledger.register_artifact(job_id, target.relative_to(directory), kind='reference_source')
                ledger.update_job(job_id, payload={'snapshot_complete': True})
            packet = ReferencePacket.model_validate_json((directory / 'packet.json').read_text()).model_dump()
            manifest = ledger.get_job(job_id)['payload'].get('manifest') or {
                'schema_version': 1, 'kind': 'reference_creative_experiment', 'run_id': job_id,
                'status': 'pending', 'created_at': utc_now(), 'brief': packet['brief'], 'title': packet['title'],
                'packet': 'packet.json', 'source_hashes': spec['source_hashes'],
                'config': config.to_dict(), 'director_config': _settings(director_config),
                'models_frozen': True, 'training_enabled': False,
                'direction_count': spec['direction_count'], 'realizations': 2,
                'batch_size': spec['batch_size'], 'note': spec.get('note', ''),
                'composition_scaffolding': {
                    'origin': 'operator_authored_scaffold' if packet.get('composition_challenges') else 'none',
                    'source': 'packet.json',
                    'direction_ids': sorted(packet.get('composition_challenges', {})),
                    'model_role': 'Qwen formulates realizations of assigned challenges' if packet.get('composition_challenges') else 'Qwen proposes directions from the brief and references',
                },
                'direction_batches': [], 'directions': [], 'images': [], 'observations': [],
                'human_review': {'status': 'pending', 'advancement': 'explicit human decision only'},
                'cost_accounting': {'root': str(ledger.path.parent), 'run_id': job_id},
                'planned_calls': job['max_calls'], 'sessions': [],
                'reference_policy': 'Reference images and sourced text condition Qwen. FLUX receives exact saved text prompts; initial generations have no image parent.',
                'distinctness_policy': 'The prompt requests distinct pictorial interventions; validation rejects identical titles, core moves or prompts. Substantive distinctness still requires human review.',
            }
            manifest['sessions'].append({'started_at': utc_now(), 'stage': stage,
                'revalidate_saved': bool(revalidate_saved),
                'retry_rejected_observers': retry_rejected_observers, 'code_sha256': code_provenance()})
            manifest['sessions'][-1]['continue_rejected_observers'] = continue_rejected_observers
            if manifest.get('observer_recovery'):
                ledger.verify_artifacts(manifest['observer_recovery']['job_id'])

            def persist(status=None):
                if status:
                    manifest['status'] = status
                manifest['updated_at'] = utc_now()
                manifest['model_calls'] = ledger.call_count(job_id)
                related = [job_id]
                recovery = manifest.get('observer_recovery')
                if recovery:
                    related.append(recovery['job_id'])
                    recovery['model_calls'] = ledger.call_count(recovery['job_id'])
                    manifest['total_model_calls'] = manifest['model_calls'] + recovery['model_calls']
                    manifest['authorized_maximum_model_calls'] = manifest['planned_calls'] + recovery['max_calls']
                    manifest['cost_accounting']['run_ids'] = related
                with CostJournal(ledger.path.parent) as costs:
                    manifest['costs'] = costs.summary(run_ids=related) if recovery else costs.summary(job_id)
                ledger.update_job(job_id, status=manifest['status'], payload={'manifest': manifest})
                write_json(directory / 'manifest.json', manifest)
                ledger.export_events(directory / 'events.jsonl', job_id=job_id, recover=True)

            print(f'Creative experiment: {directory}', flush=True)
            persist()
            try:
                if stage in {'all', 'direct'}:
                    _direct(ledger, job_id, directory, spec, packet, manifest, persist,
                            director_config, director_factory, revalidate_saved)
                if stage in {'all', 'render', 'observe'}:
                    # The aggregate may have survived a crash before artifact
                    # registration. Reconstruct it from charged Qwen evidence;
                    # never trust an unchecked aggregate as executable prompts.
                    manifest['directions'] = _load_saved_directions(
                        ledger, job_id, directory, spec, packet, revalidate_saved)
                if stage in {'all', 'render'}:
                    _render(ledger, job_id, directory, spec, packet, manifest, persist, config, backend_factory)
                if stage == 'observe' or (stage == 'all' and spec['observe']):
                    if not spec['observe']:
                        raise ValueError('Observer was not included in this experiment budget; start with --observe')
                    _observe(ledger, job_id, directory, spec, packet, manifest, persist,
                             director_config, director_factory, revalidate_saved, retry_rejected_observers,
                             continue_rejected_observers)
                # Historical failures remain in events and immutable responses;
                # the current projection should reflect this successful stage.
                manifest.pop('error', None)
                if len(manifest['images']) == spec['direction_count'] * 2:
                    _register_review(directory, ledger.path.parent, job_id, packet, manifest)
                    persist('needs_human_review')
                else:
                    persist('directions_ready')
                return directory
            except BaseException as exc:
                if isinstance(exc, UncertainInvocation):
                    status = 'uncertain'
                elif isinstance(exc, KeyboardInterrupt):
                    status = 'interrupted'
                else:
                    status = 'failed'
                manifest['error'] = {'type': type(exc).__name__, 'message': str(exc), 'at': utc_now()}
                persist(status)
                raise


def _parse_missing_observer_array_close(raw):
    """Recognize two complete observer entries missing only their array closer.

    Decode JSON tokens instead of searching text, so quoted lookalikes cannot
    establish an edit location. The resulting complete document must parse
    strictly; no other insertion or suffix recovery is combined with this policy.
    """
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)

    def whitespace(index):
        while index < len(raw) and raw[index] in ' \t\r\n':
            index += 1
        return index

    def character(index, expected):
        index = whitespace(index)
        if index >= len(raw) or raw[index] != expected:
            raise ValueError('Not the supported two-entry observer array-closure case')
        return index + 1

    def value(index):
        return decoder.raw_decode(raw, idx=whitespace(index))

    index = character(0, '{')
    key, index = value(index)
    if key != 'visible_observations':
        raise ValueError('Array-closure recovery requires visible_observations as the first field')
    index = character(character(index, ':'), '[')
    first, index = value(index)
    second, index = value(character(index, ','))
    for entry in (first, second):
        if (not isinstance(entry, dict) or set(entry) != {'candidate_id', 'observations'}
                or not isinstance(entry['candidate_id'], str) or not entry['candidate_id'].strip()
                or not isinstance(entry['observations'], list) or not 1 <= len(entry['observations']) <= 4
                or any(not isinstance(point, str) or not point.strip() for point in entry['observations'])):
            raise ValueError('Array-closure recovery requires two complete observation entries')
    offset = whitespace(index)
    key, index = value(character(offset, ','))
    if key != 'finished_quality':
        raise ValueError('Array-closure recovery requires finished_quality immediately after the two entries')
    character(index, ':')
    parsed = json.loads(raw[:offset] + ']' + raw[offset:], object_pairs_hook=_unique_object)
    expected = {'visible_observations', 'finished_quality', 'development_potential', 'revision', 'uncertainty'}
    if not isinstance(parsed, dict) or set(parsed) != expected or parsed['visible_observations'] != [first, second]:
        raise ValueError('Array-closure recovery requires the exact observer envelope')
    return parsed, {'parser_policy': 'observer_two_entries_missing_array_closer_v1', 'ignored_suffix': '',
                    'insertion_edits': [{'offset': offset, 'text': ']', 'unit': 'unicode_codepoints'}]}


def _parse_revalidation_response(raw):
    """Explicit recovery applies only documented, bounded JSON syntax policies.

    The ordinary inference parser remains strict. This helper never rewrites the
    saved model text or discards data, another object, commentary or a preamble.
    """
    if not isinstance(raw, str) or not raw.strip() or len(raw) > MAX_RESPONSE_CHARACTERS:
        raise ValueError('Saved response has no bounded raw JSON to revalidate')
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique_object)
        return parsed, {'parser_policy': 'strict_json_v1', 'ignored_suffix': ''}
    except json.JSONDecodeError:
        # JSON's standard leading whitespace is permitted; other prefixes are
        # rejected rather than searching the response for a convenient object.
        start = len(raw) - len(raw.lstrip(' \t\r\n'))
        try:
            parsed, end = json.JSONDecoder(object_pairs_hook=_unique_object).raw_decode(raw, idx=start)
        except json.JSONDecodeError:
            return _parse_missing_observer_array_close(raw)
        suffix = raw[end:]
        if (not isinstance(parsed, dict) or not suffix
                or any(character not in '}]' and not character.isspace() for character in suffix)):
            raise ValueError('Revalidation permits only one complete JSON object followed by surplus closing delimiters or whitespace')
        return parsed, {'parser_policy': 'complete_object_surplus_closing_delimiters_v1',
                        'ignored_suffix': suffix}


def _recover_response(ledger, job_id, directory, index, filename, validator, allow_revalidation=False):
    """A saved response is recovered without another model invocation."""
    path = directory / filename
    attempt = ledger.get_attempt(job_id, index)
    if path.exists():
        if attempt is None or not attempt['reserved']:
            raise ValueError(f'Model response has no invocation reservation: {filename}')
        ledger.register_artifact(job_id, filename, attempt_index=index, kind='model_response')
        result = json.loads(path.read_text())
        if result.get('output') is None:
            sidecar_path = path.with_suffix('.revalidated.json')
            if not sidecar_path.exists() and not allow_revalidation:
                raise ValueError(f'Saved Qwen response is invalid; retained at {path}; no automatic replacement')
            # Revalidation is a new parser decision over the exact retained raw
            # response. Never repair text, replace model evidence, or invoke Qwen.
            raw = result.get('raw_response')
            decoded, parser = _parse_revalidation_response(raw)
            parsed = validator(decoded)
            original_hash = sha256(path)
            if sidecar_path.exists():
                sidecar = json.loads(sidecar_path.read_text())
                if (sidecar.get('original_response_path') != filename
                        or sidecar.get('original_response_sha256') != original_hash
                        or sidecar.get('output') != parsed):
                    raise ValueError('Saved revalidation is not bound to the exact original model response')
                if ('parser_policy' in sidecar or parser['parser_policy'] != 'strict_json_v1') and any(
                        sidecar.get(key) != value for key, value in parser.items()):
                    raise ValueError('Saved revalidation parser policy or edits differ from the original response')
            else:
                sidecar = {'schema_version': 1, 'kind': 'model_response_revalidation',
                    'original_response_path': filename, 'original_response_sha256': original_hash,
                    'original_parse_error': result.get('parse_error'),
                    'validator_code_sha256': code_provenance(), 'validated_at': utc_now(),
                    'policy': 'Explicit parser revalidation of unchanged saved raw JSON; no model call or source rewrite. Only recorded bounded syntax edits are applied to parsing input; descriptive content is never invented or revised.',
                    **parser,
                    'output': parsed}
                write_json(sidecar_path, sidecar)
            ledger.register_artifact(job_id, sidecar_path.name, attempt_index=index, kind='model_response_revalidation')
            result = {**result, 'output': parsed, 'revalidation': {
                'path': sidecar_path.name, 'sha256': sha256(sidecar_path),
                'original_response_sha256': original_hash, **parser}}
        validator(result['output'])
        ledger.save_attempt(job_id, index, {'response_path': filename}, status='succeeded')
        return result
    if attempt and attempt['reserved']:
        ledger.save_attempt(job_id, index, status='uncertain')
        raise UncertainInvocation(f'Call {index} was reserved but has no saved response; no automatic rerun')
    return None


def _load_saved_directions(ledger, job_id, directory, spec, packet, allow_revalidation=False):
    prior = []
    source_ids = {ref['id'] for ref in packet['references']}
    for batch_index, start in enumerate(range(0, spec['direction_count'], spec['batch_size'])):
        ids = [f'd{i + 1:02d}' for i in range(start, min(start + spec['batch_size'], spec['direction_count']))]
        validator = lambda value: validate_directions(value, ids, source_ids, prior)
        result = _recover_response(ledger, job_id, directory, batch_index,
            f'director_batch_{batch_index:02d}.json', validator, allow_revalidation)
        if result is None:
            raise ValueError('Complete reserved Qwen direction responses are required before rendering/observing')
        prior.extend(result['output']['directions'])
    _save_json_evidence(ledger, job_id, directory, 'directions.json', {'directions': prior})
    return prior


def _qwen_call(ledger, job_id, directory, index, filename, kind, invoke, validator):
    recovered = _recover_response(ledger, job_id, directory, index, filename, validator)
    if recovered is not None:
        return recovered
    ledger.save_attempt(job_id, index, {'kind': kind, 'response_path': filename}, status='planned')
    if not ledger.reserve_call(job_id, index):
        raise UncertainInvocation('Qwen call already reserved')
    try:
        with tracked_local_call(ledger.path.parent, job_id, f'attempt:{index}', QWEN_MODEL_ID, scope=kind):
            result = invoke()
            _save_json_evidence(ledger, job_id, directory, filename, result, index=index)
        validator(result['output'])
        ledger.save_attempt(job_id, index, {'response_path': filename}, status='succeeded')
        return result
    except BaseException as exc:
        result = getattr(exc, 'result', None)
        if result is not None and not (directory / filename).exists():
            _save_json_evidence(ledger, job_id, directory, filename, result, index=index)
        ledger.save_attempt(job_id, index, {'error': f'{type(exc).__name__}: {exc}'}, status='failed')
        raise


def _prepare_model(model, ledger, job_id, scope, manifest, persist):
    def progress(settings):
        manifest.setdefault('backend_status', {})[scope] = settings
        persist()
    load_id = f'load:{scope}:{uuid.uuid4().hex}'
    with tracked_local_call(ledger.path.parent, job_id, load_id,
                           FLUX_MODEL_ID if scope == 'flux' else QWEN_MODEL_ID, scope='model_load'):
        model.prepare(on_progress=progress)
    manifest.setdefault('effective_settings', {})[scope] = dict(model.execution_settings)
    persist()


def _direct(ledger, job_id, directory, spec, packet, manifest, persist, config, factory, revalidate_saved=False):
    count, size = spec['direction_count'], spec['batch_size']
    prior, batches, director = [], [], None
    try:
        for batch_index, start in enumerate(range(0, count, size)):
            ids = [f'd{i + 1:02d}' for i in range(start, min(start + size, count))]
            filename = f'director_batch_{batch_index:02d}.json'
            validator = lambda value: validate_directions(value, ids, {r['id'] for r in packet['references']}, prior)
            result = _recover_response(ledger, job_id, directory, batch_index, filename, validator, revalidate_saved)
            if result is None:
                if director is None:
                    director = factory(config)
                    _prepare_model(director, ledger, job_id, 'director', manifest, persist)
                print(f'Qwen directions {ids[0]}–{ids[-1]}', flush=True)
                persist('directing')
                result = _qwen_call(ledger, job_id, directory, batch_index, filename, 'direction_generation',
                    lambda: director.propose_directions(packet, directory, ids, prior=prior), validator)
            prior.extend(result['output']['directions'])
            batches.append({'path': filename, 'sha256': sha256(directory / filename), 'direction_ids': ids,
                            **({'revalidation': result['revalidation']} if result.get('revalidation') else {})})
            manifest.update(directions=list(prior), direction_batches=list(batches))
            persist()
        _save_json_evidence(ledger, job_id, directory, 'directions.json', {'directions': prior})
        persist('directions_ready')
    finally:
        if director is not None:
            director.close()


def _render(ledger, job_id, directory, spec, packet, manifest, persist, config, factory):
    from PIL import Image
    start = math.ceil(spec['direction_count'] / spec['batch_size'])
    backend, images = None, []
    try:
        for direction_index, direction in enumerate(manifest['directions']):
            for realization in range(2):
                image_offset = direction_index * 2 + realization
                index, seed = start + image_offset, spec['base_seed'] + image_offset
                candidate_id = f"{direction['id']}-r{realization + 1}"
                filename = f'{candidate_id}.png'
                path = directory / filename
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError(f'Candidate output must stay inside its run directory: {filename}')
                attempt = ledger.get_attempt(job_id, index)
                if path.exists():
                    if attempt is None or not attempt['reserved']:
                        raise ValueError(f'Saved generation has no reservation: {filename}')
                elif attempt and attempt['reserved']:
                    ledger.save_attempt(job_id, index, status='uncertain')
                    raise UncertainInvocation(f'{candidate_id} has a reservation but no saved image; no automatic rerun')
                else:
                    if backend is None:
                        backend = factory(config)
                        _prepare_model(backend, ledger, job_id, 'flux', manifest, persist)
                    ledger.save_attempt(job_id, index, {'kind': 'initial_generation', 'candidate_id': candidate_id,
                        'direction_id': direction['id'], 'prompt': direction['flux_prompt'], 'seed': seed,
                        'parent_image_id': None, 'source_relationships': direction['source_relationships'],
                        'raw_output_path': filename, 'settings': dict(backend.execution_settings)}, status='planned')
                    if not ledger.reserve_call(job_id, index):
                        raise UncertainInvocation(f'Generation already reserved: {candidate_id}')
                    print(f'FLUX {candidate_id} / {spec["direction_count"] * 2} images; seed {seed}', flush=True)
                    persist('rendering')
                    measurement = Measurement(backend.execution_settings['device']) if factory is FluxBackend else None
                    try:
                        with tracked_local_call(ledger.path.parent, job_id, f'attempt:{index}', FLUX_MODEL_ID) as cost:
                            with measurement if measurement is not None else nullcontext():
                                image = backend.generate(direction['flux_prompt'], seed)
                            if measurement is not None:
                                cost['measurements'] = measurement.result
                                cost['runtime_seconds'] = measurement.result['runtime_seconds']
                            if path.exists():
                                raise ValueError(f'Refusing to overwrite model output: {filename}')
                            save_image(image, path)
                        ledger.save_attempt(job_id, index, {'measurements': measurement.result if measurement else {},
                            'prompt_token_count': getattr(backend, 'last_prompt_token_count', None)}, status='output_saved')
                    except BaseException as exc:
                        ledger.save_attempt(job_id, index, {'error': f'{type(exc).__name__}: {exc}'}, status='failed')
                        raise
                # Register before downstream checks so retained bytes survive rejection.
                ledger.register_artifact(job_id, filename, attempt_index=index, kind='raw_model_output')
                record = image_record(path, image_id=candidate_id, parent=None, seed=seed, origin='model_generation')
                if record['mode'] != 'RGB' or record['size'] != [config.width, config.height]:
                    raise ValueError(f'Unexpected image format/size retained at {path}')
                record.update(direction_id=direction['id'], realization=realization + 1,
                    label=f"{direction['title']} / realization {realization + 1}",
                    prompt=direction['flux_prompt'], direction_path='directions.json',
                    source_relationships=direction['source_relationships'], packet_sha256=sha256(directory / 'packet.json'))
                ledger.save_attempt(job_id, index, {'output': record}, status='succeeded')
                images.append(record)
                manifest['images'] = list(images)
                persist()
        _save_json_evidence(ledger, job_id, directory, 'candidates.json', {'candidates': images})
        _register_review(directory, ledger.path.parent, job_id, packet, manifest)
        persist('needs_human_review')
    finally:
        if backend is not None:
            backend.close()


def _rejected_observer(ledger, job_id, directory, index, filename, validator):
    """Identify known saved schema rejection; missing/uncertain evidence is ineligible."""
    path = directory / filename
    attempt = ledger.get_attempt(job_id, index)
    if not path.is_file() or attempt is None or not attempt['reserved']:
        return None
    if path.with_suffix('.revalidated.json').exists():
        return None  # Failed revalidation binding checks must never become retries.
    artifact = next((item for item in ledger.artifacts(job_id) if item['path'] == filename), None)
    if (artifact is None or artifact['sha256'] != sha256(path)
            or artifact['attempt_index'] != index or artifact['kind'] != 'model_response'):
        return None
    result = json.loads(path.read_text())
    if (result.get('output') is not None or not isinstance(result.get('parse_error'), str)
            or not isinstance(result.get('raw_response'), str) or not result['raw_response'].strip()
            or result.get('inference_error')):
        return None
    try:
        decoded, _ = _parse_revalidation_response(result['raw_response'])
        validator(decoded)
    except (ValueError, TypeError, RecursionError):
        return {'execution_job_id': job_id, 'attempt_index': index, 'path': filename,
                'sha256': artifact['sha256'], 'parse_error': result['parse_error']}
    # Valid content under an explicit syntax policy is ineligible for a new
    # model call or a rejection diagnostic. Preserve the original parse error
    # until the operator enables revalidation.
    return None


def _ensure_observer_recovery(ledger, job_id, directory, spec, manifest, config):
    recovery_id = job_id + '_observer_recovery'
    recovery_dir = directory / 'observer_recovery'
    candidates = [{key: image[key] for key in ('id', 'path', 'sha256', 'direction_id')}
                  for image in manifest['images']]
    recovery_spec = {'schema_version': 1, 'kind': 'observer_schema_recovery',
        'source_job_id': job_id, 'packet_sha256': sha256(directory / 'packet.json'),
        'candidates': candidates, 'director_config': _settings(config),
        'eligible_direction_ids': [d['id'] for d in manifest['directions']],
        'max_retries_per_pair': 1,
        'authorization': {'operator_flag': '--retry-rejected-observers 1',
            'scope': 'At most one new observer invocation per pair with a retained schema-rejected original response; uncertain missing outputs are excluded.'}}
    # Deliberately no ledger parent: this is a separately authorized budget,
    # linked by source_job_id without changing the immutable original allowance.
    ledger.create_job(recovery_id, recovery_spec, spec['direction_count'], recovery_dir)
    recovery_dir.mkdir(parents=True, exist_ok=True)
    _save_json_evidence(ledger, recovery_id, recovery_dir, 'job.json',
        {'schema_version': 1, 'job_id': recovery_id, 'database': str(ledger.path)})
    ledger.verify_artifacts(recovery_id)
    manifest['observer_recovery'] = {'job_id': recovery_id, 'directory': 'observer_recovery',
        'max_calls': spec['direction_count'], 'max_retries_per_pair': 1,
        'model_calls': ledger.call_count(recovery_id)}
    return recovery_id, recovery_dir


def _persist_recovery_job(ledger, recovery_id, recovery_dir, status):
    job = ledger.get_job(recovery_id)
    indices = [row[0] for row in ledger.connection.execute(
        'SELECT attempt_index FROM attempts WHERE job_id=? ORDER BY attempt_index', (recovery_id,))]
    value = {'schema_version': 1, 'kind': 'observer_schema_recovery', 'job_id': recovery_id,
        'source_job_id': job['spec']['source_job_id'], 'status': status,
        'max_calls': job['max_calls'], 'max_retries_per_pair': 1,
        'model_calls': ledger.call_count(recovery_id),
        'attempts': [ledger.get_attempt(recovery_id, index) for index in indices],
        'cost_accounting': {'root': str(ledger.path.parent), 'run_id': recovery_id}}
    ledger.update_job(recovery_id, status=status, payload={'manifest': value})
    write_json(recovery_dir / 'manifest.json', value)
    ledger.export_events(recovery_dir / 'events.jsonl', job_id=recovery_id, recover=True)


def _observe(ledger, job_id, directory, spec, packet, manifest, persist, config, factory,
             revalidate_saved=False, retry_rejected_observers=0, continue_rejected_observers=False):
    if len(manifest['images']) != spec['direction_count'] * 2:
        raise ValueError('Every candidate must be saved before the observer runs')
    start = math.ceil(spec['direction_count'] / spec['batch_size']) + spec['direction_count'] * 2
    director, observations, rejections, processed = None, [], [], set()
    prior_status = manifest.get('observer_status', {})
    continue_known = continue_rejected_observers or (
        prior_status.get('status') == 'partial' and prior_status.get('attempted_pairs') == spec['direction_count'])

    def ready_director():
        nonlocal director
        if director is None:
            director = factory(config)
            _prepare_model(director, ledger, job_id, 'observer', manifest, persist)
        return director

    def invoke_observer(owner, target_dir, attempt_index, name, direction, candidates, validator):
        try:
            return _qwen_call(ledger, owner, target_dir, attempt_index, name, 'candidate_observation',
                lambda: director.observe_candidates(packet['brief'], direction, candidates, directory), validator)
        except DirectorOutputError:
            # The explicit parser flag also covers evidence saved during this
            # invocation. It never permits new inference or invented content.
            if revalidate_saved:
                return _recover_response(ledger, owner, target_dir, attempt_index, name, validator, True)
            raise

    def progress(final=False):
        valid = {entry['direction_id'] for entry in observations}
        unresolved = [d['id'] for d in manifest['directions'] if d['id'] in processed and d['id'] not in valid]
        manifest['observations'] = list(observations)
        manifest['observer_rejections'] = list(rejections)
        manifest['observer_status'] = {
            'status': ('partial' if unresolved else 'complete') if final else 'running',
            'attempted_pairs': len(processed), 'valid_pairs': len(valid),
            'rejected_pairs': len(unresolved), 'unresolved_direction_ids': unresolved,
            'rejected_response_count': len(rejections)}
        persist()

    def record_rejection(direction_id, rejection, role, *, prefix='', supersedes=None):
        record = {**rejection, 'direction_id': direction_id, 'status': 'schema_rejected', 'role': role}
        if prefix:
            record['path'] = str(Path(prefix) / record['path'])
        if supersedes:
            record['supersedes'] = supersedes
        identity = (record['execution_job_id'], record['attempt_index'])
        if not any((r['execution_job_id'], r['attempt_index']) == identity for r in rejections):
            rejections.append(record)
        processed.add(direction_id)
        progress()

    try:
        for direction_index, direction in enumerate(manifest['directions']):
            candidates = [r for r in manifest['images'] if r['direction_id'] == direction['id']]
            if any(sha256(directory / c['path']) != c['sha256'] for c in candidates):
                raise ValueError('Observer candidate bytes changed before inference')
            index = start + direction_index
            filename = f"observer_{direction['id']}.json"
            validator = lambda value: validate_observation(value, [c['id'] for c in candidates])
            owner_id, owner_index, source_path, supersedes = job_id, index, filename, None
            result = None
            try:
                result = _recover_response(ledger, job_id, directory, index, filename, validator, revalidate_saved)
                if result is None:
                    ready_director()
                    persist('observing')
                    print(f'Qwen observes {direction["id"]}', flush=True)
                    result = invoke_observer(job_id, directory, index, filename, direction, candidates, validator)
            except Exception:
                rejected = _rejected_observer(ledger, job_id, directory, index, filename, validator)
                if rejected is None:
                    raise  # Missing output, inference failure or integrity error remains blocking.
                record_rejection(direction['id'], rejected, 'original')
                recovery = manifest.get('observer_recovery')
                if not retry_rejected_observers and not recovery:
                    if continue_known:
                        continue
                    raise
                recovery_id, recovery_dir = _ensure_observer_recovery(ledger, job_id, directory, spec, manifest, config)
                retry_filename = f"observer_{direction['id']}_retry_1.json"
                owner_id, owner_index = recovery_id, direction_index
                source_path = str((recovery_dir / retry_filename).relative_to(directory))
                supersedes = rejected
                with ledger.lock(recovery_id):
                    try:
                        ledger.verify_artifacts(recovery_id)
                        result = _recover_response(ledger, recovery_id, recovery_dir, direction_index,
                            retry_filename, validator, revalidate_saved)
                        if result is None:
                            # A stored allowance permits reuse of existing evidence;
                            # only the fresh operator retry flag starts a new call.
                            if not retry_rejected_observers:
                                if continue_known:
                                    continue
                                raise ValueError('This rejected pair has no saved retry; a fresh --retry-rejected-observers 1 flag is required for another model call')
                            ready_director()
                            ledger.save_attempt(recovery_id, direction_index, {
                                'kind': 'candidate_observation', 'direction_id': direction['id'],
                                'candidate_ids': [c['id'] for c in candidates], 'supersedes': rejected,
                                'response_path': retry_filename}, status='planned')
                            _persist_recovery_job(ledger, recovery_id, recovery_dir, 'running')
                            print(f'Qwen retries rejected observer {direction["id"]}: 1/1 new call', flush=True)
                            result = invoke_observer(recovery_id, recovery_dir, direction_index,
                                retry_filename, direction, candidates, validator)
                        _persist_recovery_job(ledger, recovery_id, recovery_dir, 'running')
                    except Exception:
                        _persist_recovery_job(ledger, recovery_id, recovery_dir, 'failed')
                        retry_rejection = _rejected_observer(ledger, recovery_id, recovery_dir,
                            direction_index, retry_filename, validator)
                        if retry_rejection is None:
                            raise
                        record_rejection(direction['id'], retry_rejection, 'retry',
                            prefix='observer_recovery', supersedes=rejected)
                        if continue_known:
                            continue
                        raise
                    except BaseException:
                        _persist_recovery_job(ledger, recovery_id, recovery_dir, 'failed')
                        raise
            # Only schema-valid content enters the canonical observations list.
            record = {'direction_id': direction['id'], 'path': source_path,
                      'sha256': sha256(directory / source_path), 'output': result['output'],
                      'execution_job_id': owner_id, 'attempt_index': owner_index,
                      'authority': 'model prediction; not a human vote or advancement'}
            if supersedes:
                record['supersedes'] = supersedes
            if result.get('revalidation'):
                revalidation = dict(result['revalidation'])
                if owner_id != job_id:
                    revalidation['path'] = str(Path('observer_recovery') / revalidation['path'])
                record['revalidation'] = revalidation
            observations.append(record)
            processed.add(direction['id'])
            progress()
        _save_json_evidence(ledger, job_id, directory, 'observations.json', {'observations': observations})
        _save_json_evidence(ledger, job_id, directory, 'observer_rejections.json', {'rejections': rejections})
        if manifest.get('observer_recovery'):
            _persist_recovery_job(ledger, manifest['observer_recovery']['job_id'], directory / 'observer_recovery', 'completed')
        progress(final=True)
    finally:
        if director is not None:
            director.close()


def _register_review(directory, root, job_id, packet, manifest):
    candidates = [{'id': r['id'], 'path': str(directory / r['path']), 'label': r['label'],
                   'direction_id': r['direction_id'], 'parent_candidate_id': None} for r in manifest['images']]
    ids = [d['id'] for d in manifest['directions']]
    pairs, design = diagnostic_pairs(ids), design_metadata(ids)
    store = ReviewStore(root)
    try:
        existing = store.collection(job_id)
    except ReviewNotFound:
        existing = None
    if existing is not None and existing['pairs'] != pairs:
        # A code update may change the default schedule. Never migrate an
        # already registered collection, even if it has no votes yet.
        from collections import Counter
        pairs = existing['pairs']
        design = {'schema_version': 1, 'design': 'existing_immutable_collection',
            'pair_count': len(pairs), 'planned_dimension_questions': len(pairs) * 2,
            'comparisons_per_candidate': dict(Counter(c for p in pairs for c in p['candidate_ids'])),
            'note': 'The previously registered comparison plan is preserved; no collection or votes were migrated.'}
    store.register_collection(job_id, title=packet['title'], brief=packet['brief'],
                              candidates=candidates, pairs=pairs)
    manifest['human_review'].update(collection_id=job_id, pair_count=len(pairs),
        design_metadata=design, planned_dimension_questions=design['planned_dimension_questions'],
        sampling_note=design['note'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--revalidate-saved', action='store_true',
                        help='Explicitly revalidate unchanged rejected JSON into a linked sidecar; never rerun the model')
    parser.add_argument('--retry-rejected-observers', type=int, choices=[0, 1], default=0,
                        help='Authorize one separately budgeted new call per saved schema-rejected observer pair; never retry uncertain missing outputs')
    parser.add_argument('--continue-rejected-observers', action='store_true',
                        help='Finish the original observer pass while retaining known schema failures as separate diagnostics; do not start new retries')
    parser.add_argument('--stage', choices=['all', 'direct', 'render', 'observe'], default='all')
    parser.add_argument('--job-id')
    parser.add_argument('--directions', type=int, default=12)
    parser.add_argument('--batch-size', type=int, choices=[1, 2, 3], default=1,
                        help='Qwen directions per call; default 1 avoids multi-direction response truncation')
    parser.add_argument('--note', default='', help='Recorded reason or context for this new experiment')
    parser.add_argument('--realizations', type=int, default=2)
    parser.add_argument('--base-seed', type=int, default=11001)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--cache-dir', type=Path, default=Path('.cache/huggingface'))
    parser.add_argument('--output-dir', type=Path, default=Path('artifacts'))
    parser.add_argument('--device', choices=['auto', 'mps', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'bf16', 'fp16', 'fp32'], default='auto')
    parser.add_argument('--offload', choices=['none', 'model', 'sequential'], default='none')
    parser.add_argument('--local-files-only', action='store_true')
    parser.add_argument('--vae-tiling', action='store_true')
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--guidance', type=float, default=4.0)
    parser.add_argument('--director-max-image-pixels', type=int, default=262144)
    args = parser.parse_args(argv)
    if args.resume:
        # Explicit flags would look like overrides but be ignored; reject them.
        import sys
        supplied = list(argv) if argv is not None else sys.argv[1:]
        allowed = {'--resume', '--stage', '--revalidate-saved', '--retry-rejected-observers', '--continue-rejected-observers'}
        if any(token.startswith('--') and token.split('=')[0] not in allowed for token in supplied):
            parser.error('Resume permits stage and explicit recovery flags; original model inputs and settings remain frozen')
        directory = run(resume_dir=args.resume, stage=args.stage, revalidate_saved=args.revalidate_saved,
                        retry_rejected_observers=args.retry_rejected_observers,
                        continue_rejected_observers=args.continue_rejected_observers)
    else:
        if not args.packet:
            parser.error('Supply --packet or --resume')
        if args.revalidate_saved:
            parser.error('--revalidate-saved requires --resume')
        if args.retry_rejected_observers:
            parser.error('--retry-rejected-observers requires --resume')
        if args.continue_rejected_observers:
            parser.error('--continue-rejected-observers requires --resume')
        config = Config(cache_dir=args.cache_dir.resolve(), output_dir=args.output_dir.resolve(),
            device=args.device, precision=args.precision, offload=args.offload,
            local_files_only=args.local_files_only, vae_tiling=args.vae_tiling,
            width=args.width, height=args.height, steps=args.steps, guidance=args.guidance)
        director_config = DirectorConfig(config.cache_dir, device=config.device, precision=config.precision,
            local_files_only=config.local_files_only, max_new_tokens=2048, max_image_pixels=args.director_max_image_pixels)
        directory = run(args.packet, config, director_config, directions=args.directions,
            realizations=args.realizations, base_seed=args.base_seed, observe=args.observe,
            stage=args.stage, job_id=args.job_id, batch_size=args.batch_size, note=args.note)
    print(f'Saved: {directory}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
