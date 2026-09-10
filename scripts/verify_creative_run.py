#!/usr/bin/env python3
"""Verify a completed 12×2 frozen-model diagnostic without inference or voting.

Usage: uv run python scripts/verify_creative_run.py artifacts/RUN_DIRECTORY
Writes two new immutable evidence files only after every check passes. This
verifies recorded technical integrity, not artistic quality or cultural success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile

from PIL import Image, ImageOps

from kusama.artifacts import sha256, sync_directory
from kusama.costs import CostJournal
from kusama.creative import _parse_revalidation_response
from kusama.director import validate_directions, validate_observation, _unique_object
from kusama.ledger import Ledger
from kusama.references import ReferencePacket, prompt_context, selected_challenges
from kusama.reviews import ReviewStore


FLUX = ('black-forest-labs/FLUX.2-klein-base-4B', 'a3b4f4849157f664bdbc776fd7453c2783562f4d')
QWEN = ('Qwen/Qwen3-VL-8B-Instruct', '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b')
EARLIER_PILOTS = ['20260908T015317_creative_c514a8c7', '20260908T020935_creative_6cd3bb17']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)


def checked_path(directory, relative):
    supplied = directory / relative
    require(not Path(relative).is_absolute(), f'Expected relative artifact path: {relative}')
    path = supplied.resolve()
    require(path.is_relative_to(directory) and path.is_file(), f'Artifact escaped run or is missing: {relative}')
    require(not any(p.is_symlink() for p in [supplied, *supplied.parents] if p.is_relative_to(directory)),
            f'Artifact contains a symlink: {relative}')
    return path


def check_frozen(settings, expected, label):
    require(settings.get('model_id') == expected[0], f'{label}: wrong model ID')
    for key in ('requested_revision', 'model_revision', 'resolved_revision'):
        require(settings.get(key) == expected[1], f'{label}: wrong or missing {key}')
    require(type(settings.get('trainable_parameter_count')) is int and settings['trainable_parameter_count'] == 0,
            f'{label}: trainable parameter count must explicitly be zero')
    require(settings.get('status') == 'ready', f'{label}: model did not report ready')
    if expected == QWEN:
        require(settings.get('training') is False and settings.get('training_enabled') is False
                and settings.get('inference_mode') is True, f'{label}: frozen inference flags missing')
        require(settings.get('model_class') == 'Qwen3VLForConditionalGeneration', f'{label}: unexpected model class')
    else:
        require(settings.get('is_distilled') is False, f'{label}: base FLUX required')
        modules = settings.get('frozen_modules', {})
        require(set(modules) == {'vae', 'text_encoder', 'transformer'}, f'{label}: incomplete frozen module report')
        for name, module in modules.items():
            require(type(module.get('trainable_parameter_count')) is int and module['trainable_parameter_count'] == 0
                    and module.get('training') is False, f'{label}: {name} was not frozen')


def check_inputs(response, expected, max_pixels):
    """Verify hashes of the actual source bytes recorded by Qwen, plus rendition sizes."""
    inputs = response['metadata']['input_images']
    require([item['id'] for item in inputs] == list(expected), 'Qwen input image identities/order differ')
    for item in inputs:
        path = expected[item['id']]
        require(Path(item['path']).resolve() == path and item['sha256'] == sha256(path),
                f"Qwen input bytes differ: {item['id']}")
        with Image.open(path) as original:
            original.load()
            require(item['original_size'] == list(original.size), 'Qwen original dimensions differ')
            image = ImageOps.exif_transpose(original).convert('RGB')
        if image.width * image.height > max_pixels:
            scale = math.sqrt(max_pixels / (image.width * image.height))
            image.thumbnail((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
        require(item['director_image_size'] == list(image.size), 'Qwen bounded rendition dimensions differ')
    require(type(response['metadata'].get('generated_tokens')) is int
            and response['metadata']['generated_tokens'] > 0, 'Missing actual Qwen generation token count')


def recorded_attempts(ledger, job_id):
    rows = ledger.connection.execute(
        'SELECT attempt_index FROM attempts WHERE job_id=? ORDER BY attempt_index', (job_id,)).fetchall()
    attempts = {row[0]: ledger.get_attempt(job_id, row[0]) for row in rows}
    require(len(attempts) == ledger.call_count(job_id)
            and all(a['reserved'] and a['status'] in ('succeeded', 'failed') for a in attempts.values()),
            f'Every retained attempt must have a terminal reservation: {job_id}')
    return attempts


def response_output(directory, artifacts, filename, index, validator):
    require(filename in artifacts, f'Unregistered model response: {filename}')
    require(artifacts[filename]['attempt_index'] == index and artifacts[filename]['kind'] == 'model_response',
            f'Wrong response attempt binding: {filename}')
    path = checked_path(directory, filename)
    response = read_json(path)
    raw = response.get('raw_response')
    require(isinstance(raw, str) and raw.strip(), f'Missing raw model response: {filename}')
    sidecar_name = None
    if response.get('output') is None:
        sidecar_name = path.with_suffix('.revalidated.json').name
        require(sidecar_name in artifacts and artifacts[sidecar_name]['attempt_index'] == index
                and artifacts[sidecar_name]['kind'] == 'model_response_revalidation',
                f'Missing registered revalidation: {filename}')
        sidecar = read_json(checked_path(directory, sidecar_name))
        parsed, parser = _parse_revalidation_response(raw)
        output = validator(parsed)
        require(sidecar.get('original_response_path') == filename
                and sidecar.get('original_response_sha256') == sha256(path)
                and sidecar.get('output') == output, f'Revalidation changed source or output: {filename}')
        require(parser['parser_policy'] in ('strict_json_v1', 'complete_object_surplus_closing_delimiters_v1',
                                           'observer_two_entries_missing_array_closer_v1'),
                'Unrecognized revalidation parser policy')
        if any(key in sidecar for key in parser) or parser['parser_policy'] != 'strict_json_v1':
            require(all(key in sidecar and sidecar[key] == value for key, value in parser.items()),
                    'Revalidation parser evidence differs')
        require(sidecar.get('insertion_edits', []) == parser.get('insertion_edits', []),
                'Revalidation insertion evidence differs')
        if parser['parser_policy'] == 'observer_two_entries_missing_array_closer_v1':
            edits = parser['insertion_edits']
            require(len(edits) == 1 and set(edits[0]) == {'offset', 'text', 'unit'}
                    and type(edits[0]['offset']) is int and 0 <= edits[0]['offset'] < len(raw)
                    and edits[0]['text'] == ']' and edits[0]['unit'] == 'unicode_codepoints'
                    and parser['ignored_suffix'] == '', 'Observer recovery is not one recorded array-closer insertion')
            repaired = raw[:edits[0]['offset']] + ']' + raw[edits[0]['offset']:]
            require(validator(json.loads(repaired, object_pairs_hook=_unique_object)) == output,
                    'Recorded insertion does not reconstruct the validated observer output')
    else:
        output = validator(json.loads(raw, object_pairs_hook=_unique_object))
        require(output == response['output'], f'Parsed output differs from raw response: {filename}')
    check_frozen(response['metadata']['settings'], QWEN, filename)
    return response, output, sidecar_name


def review_counts(store, collection_id, images, expected_pairs, directory):
    collection = store.collection(collection_id)
    require(len(collection['candidates']) == 24 and collection['pairs'] == expected_pairs,
            'Review collection does not match the frozen 108-pair plan')
    candidates = {candidate['id']: candidate for candidate in collection['candidates']}
    require(set(candidates) == set(images), 'Review candidate identities differ')
    for identifier, image in images.items():
        saved = candidates[identifier]
        require(saved['sha256'] == image['sha256'] and saved['size'] == image['size']
                and saved['direction_id'] == image['direction_id'] and saved['parent_candidate_id'] is None,
                f'Review candidate binding differs: {identifier}')
        require(store.candidate_path(collection_id, identifier) == checked_path(directory, image['path']), 'Review image path differs')
    pairs = {pair['id']: set(pair['candidate_ids']) for pair in expected_pairs}
    history = store.history(collection_id)
    coverage = set()
    for vote in history['votes']:
        require(vote['collection_id'] == collection_id and vote['pair_id'] in pairs, 'Human vote has invalid collection/pair')
        order = vote['candidate_order']
        require(set(order) == {'A', 'B'} and {v['id'] for v in order.values()} == pairs[vote['pair_id']],
                'Human vote candidate order does not bind to its pair')
        for item in order.values():
            require(item['sha256'] == images[item['id']]['sha256'], 'Human vote references changed image bytes')
        for dimension in ('finished_quality', 'development_potential'):
            answer = vote[dimension]
            require(answer['choice'] in ('A', 'B', 'tie', 'unsure'), 'Human vote has invalid choice')
            preferred = order[answer['choice']]['id'] if answer['choice'] in ('A', 'B') else None
            require(answer['preferred_candidate_id'] == preferred, 'Human vote preferred image differs from displayed side')
        coverage.add((vote['reviewer'], vote['pair_id']))
    for advance in history['advancements']:
        require(advance['collection_id'] == collection_id and advance['candidate_id'] in images
                and advance['image_sha256'] == images[advance['candidate_id']]['sha256'], 'Advancement image binding differs')
    return {'review_records': len(history['votes']), 'dimension_answers_including_revisions': 2 * len(history['votes']),
            'unique_reviewer_pair_comparisons': len(coverage),
            'reviewed_pairs_by_reviewer': dict(Counter(reviewer for reviewer, _ in coverage)),
            'advancement_records': len(history['advancements']),
            'note': 'Read-only snapshot of actual human records; counts can grow. Revisions are retained and not counted as new unique comparisons.'}


def exclusive_json(path, value):
    """Publish complete bytes atomically with an exclusive link; never replace evidence."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                prefix='.verification-', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify(run_directory):
    directory = Path(run_directory).resolve()
    descriptor = read_json(directory / 'job.json')
    job_id = descriptor['job_id']
    database = Path(descriptor['database']).resolve()
    require(database.is_file() and database.parent == directory.parent, 'Unexpected or missing ledger location')
    require(not any((directory / name).exists() or (directory / name).is_symlink() for name in ('verification.json', 'cost-summary.json')),
            'Verification outputs already exist; refusing to overwrite')
    with Ledger(database) as ledger, ledger.lock(job_id):
        job = ledger.get_job(job_id)
        require(Path(job['directory']) == directory and job['status'] == 'needs_human_review', 'Run is incomplete or mismatched')
        manifest = read_json(directory / 'manifest.json')
        require(manifest == job['payload']['manifest'], 'Manifest differs from the authoritative ledger projection')
        require(Path(manifest['cost_accounting']['root']).resolve() == database.parent
                and manifest['cost_accounting']['run_id'] == job_id, 'Manifest monetary journal binding differs')
        spec = job['spec']
        require((spec['direction_count'], spec['realizations'], spec['batch_size'], spec['observe']) == (12, 2, 1, True),
                'This verification requires 12 single-direction calls, 24 images and 12 observers')
        require(job['max_calls'] == manifest['planned_calls'] == 48,
                'The original 48-call budget must remain unchanged')
        attempts = recorded_attempts(ledger, job_id)
        require(manifest['model_calls'] == len(attempts) == 48 and set(attempts) == set(range(48)),
                'All 48 original calls, including all 12 observer pairs, must have terminal reservations')
        require(all(attempts[index]['status'] == 'succeeded' for index in range(36)),
                'All 12 director and 24 image attempts must have succeeded')
        require(manifest['models_frozen'] is True and manifest['training_enabled'] is False, 'Frozen-model manifest flags differ')
        artifact_rows = ledger.verify_artifacts(job_id)
        artifacts = {record['path']: record for record in artifact_rows}
        jobs = {job_id: job}
        attempts_by_job = {job_id: attempts}
        artifacts_by_job = {job_id: artifacts}
        directories = {job_id: directory}
        require({'source_packet.json', 'packet.json', 'directions.json', 'candidates.json', 'observations.json'} <= set(artifacts),
                'Missing registered aggregate/source evidence')
        source = ReferencePacket.model_validate_json((directory / 'source_packet.json').read_text()).model_dump()
        packet = ReferencePacket.model_validate_json((directory / 'packet.json').read_text()).model_dump()
        require(sha256(directory / 'source_packet.json') == spec['source_hashes']['packet'], 'Original packet bytes differ')
        require(source == spec['packet'] and manifest['source_hashes'] == spec['source_hashes'], 'Source packet metadata differs')
        expected_source_images = {}
        original_refs = {ref['id']: ref for ref in source['references']}
        require({ref['id'] for ref in packet['references']} == set(original_refs), 'Snapshot changed the reference set')
        for ref in packet['references']:
            original = original_refs[ref['id']]
            require({k: v for k, v in ref.items() if k != 'image_path'} ==
                    {k: v for k, v in original.items() if k != 'image_path'}, 'Snapshot changed reference context')
            require(bool(ref['image_path']) == bool(original['image_path']), 'Snapshot changed image availability')
            if ref['image_path']:
                path = checked_path(directory, ref['image_path'])
                require(ref['image_path'] in artifacts and sha256(path) == spec['source_hashes']['images'][ref['id']],
                        'Copied reference image bytes differ')
                expected_source_images[ref['id']] = path
        require(set(expected_source_images) == set(spec['source_hashes']['images']), 'Copied source image set differs')
        require({k: v for k, v in source.items() if k != 'references'} ==
                {k: v for k, v in packet.items() if k != 'references'}, 'Snapshot changed brief or challenge data')
        config = spec['config']
        require(manifest['config'] == config and config['parent_image'] is None, 'Generation config or parent differs')
        require(config['model_revision'] == FLUX[1] and spec['director_config']['model_revision'] == QWEN[1], 'Pinned spec revisions differ')
        for name, expected in [('director', QWEN), ('flux', FLUX), ('observer', QWEN)]:
            check_frozen(manifest['effective_settings'][name], expected, name)
        ids, directions, revalidated = [f'd{i:02d}' for i in range(1, 13)], [], []
        for index, identifier in enumerate(ids):
            filename = f'director_batch_{index:02d}.json'
            response, output, sidecar = response_output(directory, artifacts, filename, index,
                lambda value: validate_directions(value, [identifier], set(original_refs), directions))
            request = response['metadata']['request']
            require(request['packet'] == prompt_context(packet) and request['requested_direction_ids'] == [identifier],
                    'Qwen director received a different brief/reference packet or direction ID')
            require(request.get('assigned_composition_challenges', {}) == selected_challenges(packet, [identifier]),
                    'Qwen composition challenge differs from frozen packet')
            check_inputs(response, expected_source_images, spec['director_config']['max_image_pixels'])
            require(attempts[index]['kind'] == 'direction_generation' and attempts[index]['response_path'] == filename,
                    'Direction attempt mapping differs')
            directions.extend(output['directions'])
            if sidecar:
                revalidated.append(sidecar)
        require(read_json(directory / 'directions.json') == {'directions': directions}
                and manifest['directions'] == directions, 'Direction aggregate differs from actual Qwen responses')
        images = manifest['images']
        require(len(images) == 24 and read_json(directory / 'candidates.json') == {'candidates': images}, 'Candidate aggregate differs')
        indexed_images = {image['id']: image for image in images}
        require(len(indexed_images) == 24 and len({image['seed'] for image in images}) == 24, 'Candidate IDs/seeds are not unique')
        for offset, image in enumerate(images):
            direction, realization = directions[offset // 2], offset % 2 + 1
            identifier = f"{direction['id']}-r{realization}"
            attempt, path = attempts[12 + offset], checked_path(directory, image['path'])
            require(image['id'] == identifier and path.name == f'{identifier}.png', 'Image order or filename differs')
            require(artifacts[image['path']]['attempt_index'] == 12 + offset and artifacts[image['path']]['kind'] == 'raw_model_output',
                    'Image artifact attempt binding differs')
            require(attempt['output'] == image and attempt['kind'] == 'initial_generation'
                    and attempt['candidate_id'] == identifier and attempt['direction_id'] == direction['id'], 'Image attempt mapping differs')
            require(image['seed'] == attempt['seed'] == spec['base_seed'] + offset, 'Seed differs from frozen realization seed')
            require(image['prompt'] == attempt['prompt'] == direction['flux_prompt'], 'FLUX prompt differs from exact Qwen direction')
            require(image['source_relationships'] == attempt['source_relationships'] == direction['source_relationships'],
                    'Image source relationships differ')
            require(image['parent_image_id'] is None and image['parent_sha256'] is None and attempt['parent_image_id'] is None,
                    'Initial generation unexpectedly has a parent')
            require(image['packet_sha256'] == sha256(directory / 'packet.json') and sha256(path) == image['sha256'], 'Image/source hash differs')
            with Image.open(path) as opened:
                opened.load()
                require(opened.format == 'PNG' and opened.mode == image['mode'] == 'RGB'
                        and list(opened.size) == image['size'] == [config['width'], config['height']], 'Image mode/dimensions differ')
            check_frozen(attempt['settings'], FLUX, identifier)
            for setting, frozen_key in [('width', 'width'), ('height', 'height'), ('num_inference_steps', 'steps'), ('guidance_scale', 'guidance')]:
                require(attempt['settings'][setting] == config[frozen_key], 'Per-call FLUX settings differ from frozen config')
            require(type(attempt.get('prompt_token_count')) is int and 0 < attempt['prompt_token_count'] <= config['max_sequence_length'],
                    'Missing/invalid non-truncated FLUX prompt token measurement')
        observations = manifest['observations']
        require(isinstance(observations, list) and len(observations) <= 12
                and read_json(directory / 'observations.json') == {'observations': observations},
                'Accepted observer aggregate differs')
        accepted_by_direction = {record['direction_id']: record for record in observations}
        require(len(accepted_by_direction) == len(observations)
                and list(accepted_by_direction) == [identifier for identifier in ids if identifier in accepted_by_direction],
                'Accepted observer directions are duplicated, unknown or out of order')
        recovery = manifest.get('observer_recovery')
        if recovery:
            recovery_id = recovery['job_id']
            require(recovery_id == f'{job_id}_observer_recovery' and recovery['directory'] == 'observer_recovery',
                    'Unexpected observer recovery identity or location')
            recovery_directory = (directory / recovery['directory']).resolve()
            require(recovery_directory.parent == directory and not (directory / recovery['directory']).is_symlink(),
                    'Observer recovery directory escaped the run or is a symlink')
            recovery_job = ledger.get_job(recovery_id)
            recovery_spec = recovery_job['spec']
            require(read_json(recovery_directory / 'job.json') ==
                    {'schema_version': 1, 'job_id': recovery_id, 'database': str(database)},
                    'Recovery ledger descriptor differs')
            require(Path(recovery_job['directory']) == recovery_directory and recovery_job['parent_id'] is None,
                    'Recovery must have its own bounded job without charging the immutable primary budget')
            require(recovery_spec['kind'] == 'observer_schema_recovery' and recovery_spec['source_job_id'] == job_id
                    and recovery_spec['packet_sha256'] == sha256(directory / 'packet.json')
                    and recovery_spec['director_config'] == spec['director_config'], 'Recovery source/config binding differs')
            expected_candidates = [{key: image[key] for key in ('id', 'path', 'sha256', 'direction_id')} for image in images]
            require(recovery_spec['candidates'] == expected_candidates, 'Recovery spec changed candidate identities or bytes')
            eligible = recovery_spec['eligible_direction_ids']
            require(isinstance(eligible, list) and len(eligible) == len(set(eligible))
                    and set(eligible) <= set(ids), 'Recovery contains unknown or duplicate eligible directions')
            require(recovery_spec['authorization']['operator_flag'] == '--retry-rejected-observers 1'
                    and recovery_spec['max_retries_per_pair'] == recovery['max_retries_per_pair'] == 1
                    and recovery_job['max_calls'] == recovery['max_calls'] == 12,
                    'Recovery lacks the explicit bounded authorization')
            recovery_attempts = recorded_attempts(ledger, recovery_id)
            require(set(recovery_attempts) <= set(range(12))
                    and recovery['model_calls'] == len(recovery_attempts) <= 12,
                    'Recovery call count or direction slots differ')
            recovery_manifest = read_json(recovery_directory / 'manifest.json')
            require(recovery_job['status'] == recovery_manifest['status'] == 'completed'
                    and recovery_manifest == recovery_job['payload']['manifest']
                    and recovery_manifest['model_calls'] == len(recovery_attempts)
                    and recovery_manifest['attempts'] == list(recovery_attempts.values()),
                    'Recovery manifest differs from its completed authoritative job')
            jobs[recovery_id], attempts_by_job[recovery_id] = recovery_job, recovery_attempts
            directories[recovery_id] = recovery_directory
            artifacts_by_job[recovery_id] = {row['path']: row for row in ledger.verify_artifacts(recovery_id)}
            require(manifest['total_model_calls'] == sum(len(value) for value in attempts_by_job.values())
                    and manifest['authorized_maximum_model_calls'] == 60,
                    'Aggregate invocation count or separately authorized maximum differs')
            require(manifest['cost_accounting']['run_ids'] == [job_id, recovery_id],
                    'Manifest cost coverage omits or duplicates the recovery job')
        selected_observers, superseded_observers = set(), set()
        observer_bindings = []
        for offset, direction in enumerate(directions):
            if direction['id'] not in accepted_by_direction:
                continue
            candidate_ids = [f"{direction['id']}-r1", f"{direction['id']}-r2"]
            summary = accepted_by_direction[direction['id']]
            owner_id = summary.get('execution_job_id', job_id)
            require(owner_id in jobs, 'Observer response belongs to an unbound execution job')
            index = summary.get('attempt_index', 36 + offset if owner_id == job_id else None)
            expected_index = 36 + offset if owner_id == job_id else offset
            require(index == expected_index and index in attempts_by_job[owner_id], 'Observer work-slot index differs')
            owner_directory, owner_artifacts = directories[owner_id], artifacts_by_job[owner_id]
            filename = f"observer_{direction['id']}.json" if owner_id == job_id else f"observer_{direction['id']}_retry_1.json"
            response_path = checked_path(owner_directory, filename)
            require(checked_path(directory, summary['path']) == response_path, 'Observer summary path differs from owning job')
            response, output, sidecar = response_output(owner_directory, owner_artifacts, filename, index,
                lambda value: validate_observation(value, candidate_ids))
            attempt = attempts_by_job[owner_id][index]
            require(attempt['kind'] == 'candidate_observation' and attempt['status'] == 'succeeded'
                    and attempt['response_path'] == filename, 'Observer attempt binding differs')
            request = response['metadata']['request']
            require(request['brief'] == packet['brief'] and request['direction'] == direction
                    and [request['candidate_a'], request['candidate_b']] == candidate_ids, 'Observer received different candidates/brief/direction')
            check_inputs(response, {identifier: checked_path(directory, indexed_images[identifier]['path']) for identifier in candidate_ids},
                         spec['director_config']['max_image_pixels'])
            require(summary['direction_id'] == direction['id']
                    and summary['sha256'] == sha256(response_path) and summary['output'] == output,
                    'Observer summary differs from saved model response')
            selected_observers.add((owner_id, index))
            if owner_id != job_id:
                original_index, original_name = 36 + offset, f"observer_{direction['id']}.json"
                original_path = checked_path(directory, original_name)
                original = read_json(original_path)
                supersedes = {'execution_job_id': job_id, 'attempt_index': original_index,
                              'path': original_name, 'sha256': sha256(original_path),
                              'parse_error': original.get('parse_error')}
                require(summary['supersedes'] == attempt['supersedes'] == supersedes
                        and attempt['candidate_ids'] == candidate_ids and attempt['direction_id'] == direction['id']
                        and direction['id'] in recovery_spec['eligible_direction_ids'], 'Recovery supersession binding differs')
                require(original_index in attempts and attempts[original_index]['status'] == 'failed'
                        and attempts[original_index]['kind'] == 'candidate_observation'
                        and attempts[original_index]['response_path'] == original_name
                        and artifacts[original_name]['attempt_index'] == original_index
                        and artifacts[original_name]['kind'] == 'model_response', 'Original rejected observer evidence is not retained')
                require(original.get('output') is None and isinstance(original.get('raw_response'), str)
                        and original['raw_response'].strip() and original.get('parse_error'), 'Rejected observer raw/error evidence is missing')
                original_request = original['metadata']['request']
                require(original_request['brief'] == packet['brief'] and original_request['direction'] == direction
                        and [original_request['candidate_a'], original_request['candidate_b']] == candidate_ids,
                        'Rejected observer input binding differs from the recovered pair')
                check_frozen(original['metadata']['settings'], QWEN, original_name)
                check_inputs(original, {identifier: checked_path(directory, indexed_images[identifier]['path']) for identifier in candidate_ids},
                             spec['director_config']['max_image_pixels'])
                superseded_observers.add((job_id, original_index))
            else:
                require(not summary.get('supersedes'), 'Ordinary observer unexpectedly supersedes other evidence')
            observer_bindings.append({'direction_id': direction['id'], 'candidate_ids': candidate_ids,
                                     'execution_job_id': owner_id, 'attempt_index': index,
                                     'path': summary['path'], 'sha256': summary['sha256']})
            if sidecar:
                revalidated.append(str((owner_directory / sidecar).relative_to(directory)))
        rejections = manifest.get('observer_rejections', [])
        if rejections or 'observer_rejections' in manifest:
            require('observer_rejections.json' in artifacts
                    and read_json(directory / 'observer_rejections.json') == {'rejections': rejections},
                    'Rejected observer aggregate is missing or differs')
        rejected_observers = set()
        for rejection in rejections:
            require(rejection['direction_id'] in ids, 'Rejected observer direction is unknown')
            offset = ids.index(rejection['direction_id'])
            direction = directions[offset]
            candidate_ids = [f"{direction['id']}-r1", f"{direction['id']}-r2"]
            owner, index = rejection['execution_job_id'], rejection['attempt_index']
            require(owner in jobs and index in attempts_by_job[owner], 'Rejected observer reservation is unknown')
            role = 'original' if owner == job_id else 'retry'
            filename = f"observer_{direction['id']}.json" if role == 'original' else f"observer_{direction['id']}_retry_1.json"
            require(index == (36 + offset if role == 'original' else offset)
                    and rejection['role'] == role and rejection['status'] == 'schema_rejected',
                    'Rejected observer role or work slot differs')
            attempt = attempts_by_job[owner][index]
            owner_artifacts = artifacts_by_job[owner]
            path = checked_path(directories[owner], filename)
            require(attempt['status'] == 'failed' and attempt['kind'] == 'candidate_observation'
                    and attempt['response_path'] == filename and filename in owner_artifacts
                    and owner_artifacts[filename]['attempt_index'] == index
                    and owner_artifacts[filename]['kind'] == 'model_response'
                    and checked_path(directory, rejection['path']) == path
                    and rejection['sha256'] == sha256(path), 'Rejected response ownership or byte binding differs')
            response = read_json(path)
            require(response.get('output') is None and isinstance(response.get('raw_response'), str)
                    and response['raw_response'].strip() and isinstance(response.get('parse_error'), str)
                    and response['parse_error'] and rejection['parse_error'] == response['parse_error']
                    and not response.get('inference_error') and not path.with_suffix('.revalidated.json').exists(),
                    'Rejected observer lacks retained schema-failure evidence or has conflicting revalidation')
            try:
                decoded, _ = _parse_revalidation_response(response['raw_response'])
                validate_observation(decoded, candidate_ids)
            except (ValueError, TypeError, RecursionError):
                pass
            else:
                raise ValueError('A rejected observer now validates; explicit revalidation is required before final reporting')
            request = response['metadata']['request']
            require(request['brief'] == packet['brief'] and request['direction'] == direction
                    and [request['candidate_a'], request['candidate_b']] == candidate_ids,
                    'Rejected observer received different candidates/brief/direction')
            check_frozen(response['metadata']['settings'], QWEN, str(path.relative_to(directory)))
            check_inputs(response, {identifier: checked_path(directory, indexed_images[identifier]['path']) for identifier in candidate_ids},
                         spec['director_config']['max_image_pixels'])
            if role == 'retry':
                original_name = f"observer_{direction['id']}.json"
                original = read_json(checked_path(directory, original_name))
                supersedes = {'execution_job_id': job_id, 'attempt_index': 36 + offset,
                              'path': original_name, 'sha256': sha256(directory / original_name),
                              'parse_error': original.get('parse_error')}
                require(rejection['supersedes'] == attempt['supersedes'] == supersedes
                        and attempts[36 + offset]['status'] == 'failed'
                        and attempt['candidate_ids'] == candidate_ids and attempt['direction_id'] == direction['id']
                        and direction['id'] in recovery_spec['eligible_direction_ids'],
                        'Rejected retry does not bind to the failed original pair')
            else:
                require(not rejection.get('supersedes'), 'Original rejection unexpectedly supersedes another call')
            require((owner, index) not in rejected_observers, 'Rejected observer response is duplicated')
            rejected_observers.add((owner, index))
        all_observers = {(owner, index) for owner, values in attempts_by_job.items()
                         for index, attempt in values.items() if attempt['kind'] == 'candidate_observation'}
        require(all_observers == selected_observers | rejected_observers
                and not selected_observers & rejected_observers and superseded_observers <= rejected_observers,
                'Observer calls are duplicated, unbound or missing from accepted/rejected diagnostics')
        unresolved = [identifier for identifier in ids if identifier not in accepted_by_direction]
        observer_status = {'status': 'partial' if unresolved else 'complete', 'attempted_pairs': 12,
                           'valid_pairs': len(observations), 'rejected_pairs': len(unresolved),
                           'unresolved_direction_ids': unresolved, 'rejected_response_count': len(rejections)}
        require(manifest['observer_status'] == observer_status, 'Observer contract coverage differs from recorded evidence')
        if unresolved:
            require(any(session.get('continue_rejected_observers') is True for session in manifest['sessions']),
                    'Partial observer completion lacks explicit diagnostic-continuation provenance')
        succeeded_calls = sum(attempt['status'] == 'succeeded' for values in attempts_by_job.values() for attempt in values.values())
        failed_calls = sum(attempt['status'] == 'failed' for values in attempts_by_job.values() for attempt in values.values())
        require(succeeded_calls == 36 + len(observations) and failed_calls == len(rejected_observers)
                and sum(map(len, attempts_by_job.values())) == succeeded_calls + failed_calls,
                'Successful and rejected invocation counts differ from actual work results')
        review = manifest.get('human_review') or {}
        human = {'votes': 0, 'advancements': 0}
        if review.get('collection_id'):
            store = ReviewStore(database.parent)
            collection = store.collection(review['collection_id'])
            human = review_counts(store, review['collection_id'], indexed_images, collection['pairs'], directory)
        pilots = [*EARLIER_PILOTS, *jobs]
        require(len(set(pilots)) == len(pilots) == 2 + len(jobs),
                'Expected both earlier pilots and every current execution job, exactly once')
        with CostJournal(database.parent, read_only=True) as costs:
            per_pilot = {identifier: costs.summary(identifier) for identifier in pilots}
            aggregate = costs.summary(run_ids=pilots)
        require(aggregate['open_calls'] == 0 and all(s['calls'] > 0 for s in per_pilot.values()), 'Costs incomplete: open operations or missing pilot records')
        for record in aggregate['records']:
            fee = record['finish']['cost']['components']['provider_api']
            require(record['start']['execution'] == 'local' and fee['amount'] == 0 and fee['status'] == 'actual',
                    'Recorded local provider API fee differs from the expected actual zero fee')
        pilot_model_calls = {}
        for pilot in pilots:
            ledger.get_job(pilot)
            reserved = ledger.connection.execute('SELECT a.attempt_index FROM attempts a JOIN reservations r ON r.attempt_id=a.attempt_id WHERE a.job_id=?', (pilot,)).fetchall()
            expected_cost_ids = {f'attempt:{row[0]}' for row in reserved}
            recorded_cost_ids = {r['call_id'] for r in per_pilot[pilot]['records'] if r['call_id'].startswith('attempt:')}
            require(recorded_cost_ids == expected_cost_ids, f'Cost records do not cover every reserved invocation in pilot {pilot}')
            pilot_model_calls[pilot] = len(reserved)
        for owner, values in attempts_by_job.items():
            current_calls = {r['call_id']: r for r in per_pilot[owner]['records'] if r['call_id'].startswith('attempt:')}
            require(set(current_calls) == {f'attempt:{index}' for index in values},
                    'Current execution costs do not cover every model reservation exactly once')
            for index, attempt in values.items():
                model = FLUX[0] if attempt['kind'] == 'initial_generation' else QWEN[0]
                require(current_calls[f'attempt:{index}']['start']['model'] == model, 'Cost/model identity differs')
                if (owner, index) in rejected_observers:
                    require(current_calls[f'attempt:{index}']['finish']['outcome'] == 'failed',
                            'Rejected observer cost outcome was not retained')
        failed_cost_calls = {identifier: sum(record['finish']['outcome'] == 'failed'
                            for record in summary['records'] if record['call_id'].startswith('attempt:'))
                            for identifier, summary in per_pilot.items()}
        captured = datetime.now(timezone.utc).isoformat()
        cost_snapshot = {'schema_version': 1, 'captured_at': captured, 'pilot_run_ids': pilots,
            'earlier_diagnostic_run_ids': EARLIER_PILOTS, 'current_execution_job_ids': list(jobs),
            'source_database': str(database.parent / 'costs.sqlite3'), 'model_invocations_by_pilot': pilot_model_calls,
            'total_model_invocations_all_pilots': sum(pilot_model_calls.values()),
            'failed_model_cost_outcomes_by_job': failed_cost_calls, 'aggregate': aggregate,
            'per_pilot': {identifier: {key: value for key, value in summary.items() if key != 'records'}
                          for identifier, summary in per_pilot.items()},
            'note': 'Includes failed and revalidated operation costs from both earlier pilots and every current execution/recovery job. A later parser revalidation can make a work slot succeed while its original cost outcome remains failed. Runtime is recorded scope time, not total project elapsed time. Unknown operating cost is not zero.'}
        report = {'schema_version': 1, 'kind': 'technical_integrity_verification', 'status': 'passed',
            'verified_at': captured, 'run_id': job_id, 'run_directory': str(directory),
            'scope': 'Recorded technical integrity only; no artistic-quality, cultural-authenticity or human-preference verdict.',
            'models': {'flux': {'id': FLUX[0], 'revision': FLUX[1], 'trainable_parameters': 0},
                       'qwen': {'id': QWEN[0], 'revision': QWEN[1], 'trainable_parameters': 0}},
            'counts': {'directions': 12, 'images': 24, 'original_observer_pair_attempts': 12,
                       'valid_observer_responses': len(observations), 'rejected_observer_responses': len(rejections),
                       'unresolved_observer_pairs': len(unresolved), 'planned_original_work_slots': 48,
                       'reserved_succeeded_model_calls': succeeded_calls,
                       'retained_failed_model_attempts': failed_calls,
                       'total_reserved_model_invocations_current_experiment': sum(map(len, attempts_by_job.values())),
                       'reserved_model_invocations_by_execution_job': {owner: len(values) for owner, values in attempts_by_job.items()},
                       'authorized_maximum_model_invocations_current_experiment': sum(item['max_calls'] for item in jobs.values()),
                       'review_pairs': 108, 'planned_dimension_questions': 216, 'verified_artifacts_before_report': len(artifacts)},
            'human_history_at_verification': human, 'response_revalidation_sidecars': revalidated,
            'observer_contract': observer_status,
            'observer_contract_note': 'The diagnostic pass attempted every original pair. Only schema-valid accepted responses are model judgments; retained rejections do not fill missing observations or preferences. Technical integrity does not establish observer reliability.',
            'observer_execution_bindings': observer_bindings,
            'retained_rejected_observer_calls': [{'execution_job_id': owner, 'attempt_index': index}
                                                for owner, index in sorted(rejected_observers)],
            'verified_artifacts_by_execution_job': {owner: len(rows) for owner, rows in artifacts_by_job.items()},
            'composition_scaffolding': manifest.get('composition_scaffolding'),
            'source_hashes': spec['source_hashes'], 'candidate_hashes': {key: value['sha256'] for key, value in indexed_images.items()},
            'checks': ['Ledger artifact hashes', 'Exact frozen model IDs/revisions and zero trainable counts',
                'Copied source bytes and brief/challenge metadata', 'Raw/revalidated Qwen direction response bindings',
                'RGB image dimensions, unique seeds, exact prompts and absent initial parents',
                'All 12 original observer attempts; separate accepted/rejected schema evidence and input-image byte hashes',
                'Bounded observer recovery, superseded raw responses and unchanged primary call budget',
                'Preserved historical review schedule and actual human history bindings',
                'Both diagnostic pilots and all current execution/recovery cost records; no open cost operations'],
            'limitation': 'Qwen input hashes bind the original source-image bytes and recorded rendition dimensions. Processor tensors were not retained; this script does not reconstruct or re-execute model inference.',
            'verification_script_sha256': sha256(Path(__file__).resolve()), 'cost_snapshot': 'cost-summary.json'}
        # Recheck evidence after the complete inspection, before publishing either file.
        for owner in jobs:
            ledger.verify_artifacts(owner)
        # No output is written until every check above has passed.
        exclusive_json(directory / 'cost-summary.json', cost_snapshot)
        ledger.register_artifact(job_id, 'cost-summary.json', kind='cost_snapshot')
        report['cost_snapshot_sha256'] = sha256(directory / 'cost-summary.json')
        exclusive_json(directory / 'verification.json', report)
        ledger.register_artifact(job_id, 'verification.json', kind='technical_verification')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_directory', type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.run_directory)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(1, f'Verification failed; no success claim: {exc}\n')
    print(json.dumps({'status': result['status'], 'run_id': result['run_id'], 'counts': result['counts'],
                      'verification': str(args.run_directory.resolve() / 'verification.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
