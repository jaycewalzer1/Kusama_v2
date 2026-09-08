"""Integration/recovery tests use deterministic substitutes, never model downloads."""
from dataclasses import replace
import json
from pathlib import Path

from PIL import Image
import pytest

from kusama.config import Config
from kusama.creative import run, UncertainInvocation
from kusama.director import DirectorConfig
from kusama.ledger import Ledger, SpecMismatch
from kusama.references import load_packet


LIFETIMES = []


def directions(ids):
    return {'directions': [{'id': identifier, 'title': f'Painting {identifier}',
        'core_move': f'Frame the subject using negative space {identifier}.',
        'flux_prompt': f'Original painting {identifier}: a self-possessed figure in a fragmented landscape.',
        'expected_effect': 'The open space should express absence while the figure conveys continuity.',
        'source_relationships': [{'source_id': 'source', 'relationship': 'Transform the visual layers into gaps.'}]}
        for identifier in ids]}


class FakeDirector:
    calls = 0
    def __init__(self, config):
        self.config = config
        self.execution_settings = {'device': 'cpu', 'status': 'ready', 'training_enabled': False}
    def prepare(self, on_progress=None):
        LIFETIMES.append('qwen open')
        if on_progress:
            on_progress(self.execution_settings)
    def propose_directions(self, packet, directory, expected_ids, prior):
        type(self).calls += 1
        assert all((directory / ref['image_path']).exists() for ref in packet['references'] if ref['image_path'])
        return {'output': directions(expected_ids), 'raw_response': json.dumps(directions(expected_ids)),
                'metadata': {'prior_count': len(prior)}}
    def observe_candidates(self, brief, direction, candidates, directory):
        type(self).calls += 1
        output = {
            'visible_observations': [{'candidate_id': c['id'], 'observations': ['A red painted field is visible.']} for c in candidates],
            'finished_quality': {'preference': 'tie', 'reason': 'Both have equivalent surface resolution.'},
            'development_potential': {'preference': 'candidate_b', 'reason': 'A visible gap could be expanded.'},
            'revision': {'candidate_id': candidates[1]['id'], 'instruction': 'Expand the gap while preserving the figure.',
                         'expected_effect': 'The absence should become more legible.'},
            'uncertainty': 'Cultural interpretation cannot be established by the pixels alone.'}
        return {'output': output, 'raw_response': json.dumps(output)}
    def close(self):
        LIFETIMES.append('qwen close')


class FakeFlux:
    calls = 0
    def __init__(self, config):
        self.config = config
        self.execution_settings = {'device': 'cpu', 'status': 'ready', 'trainable_parameter_count': 0}
    def prepare(self, on_progress=None):
        LIFETIMES.append('flux open')
        if on_progress:
            on_progress(self.execution_settings)
    def generate(self, prompt, seed):
        type(self).calls += 1
        assert not any(s == 'qwen open' for s in LIFETIMES[-1:])
        return Image.new('RGB', (self.config.width, self.config.height), (seed % 256, 2, 3))
    def close(self):
        LIFETIMES.append('flux close')


@pytest.fixture
def setup(tmp_path):
    LIFETIMES.clear()
    FakeDirector.calls = FakeFlux.calls = 0
    reference = tmp_path / 'reference.jpg'
    Image.new('RGB', (64, 64), 'red').save(reference)
    packet = {'schema_version': 1, 'id': 'test', 'title': 'A test inquiry', 'brief': 'Make a painting about absence and continuity.',
        'references': [{'id': 'source', 'title': 'Reference title', 'url': 'https://example.org/source',
            'provenance': 'A source record.', 'permitted_uses': 'Test fixture', 'source_family': 'Museum',
            'context': 'A factual source note.', 'relationship_to_brief': 'A proposed analogy.', 'image_path': 'reference.jpg'}],
        'creative_constraints': ['Do not fabricate glyph meanings.']}
    source = tmp_path / 'packet.json'
    source.write_text(json.dumps(packet))
    config = Config(cache_dir=tmp_path / 'cache', output_dir=tmp_path / 'artifacts', device='cpu',
                    precision='fp32', width=64, height=64, steps=1)
    return source, config


def invoke(setup, **kwargs):
    source, config = setup
    kwargs.setdefault('batch_size', 3)  # Exercise compatibility with saved three-direction batches.
    return run(source, config, directions=3, observe=True, director_factory=FakeDirector,
               backend_factory=FakeFlux, **kwargs)


def test_brief_to_six_images_references_observer_review_and_no_training(setup):
    directory = invoke(setup)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_human_review'
    assert len(manifest['directions']) == 3
    assert len(manifest['images']) == 6
    assert len(manifest['observations']) == 3
    assert manifest['human_review']['pair_count'] == 15
    assert manifest['model_calls'] == 10  # one director, six FLUX, three observer
    assert manifest['training_enabled'] is False
    assert all(image['parent_image_id'] is None for image in manifest['images'])
    assert all(image['source_relationships'] for image in manifest['images'])
    assert LIFETIMES == ['qwen open', 'qwen close', 'flux open', 'flux close', 'qwen open', 'qwen close']
    assert FakeDirector.calls == 4 and FakeFlux.calls == 6
    assert (directory / 'references/source.jpg').read_bytes() == (setup[0].parent / 'reference.jpg').read_bytes()
    assert (setup[1].output_dir / 'reviews.sqlite3').exists()
    assert manifest['cost_accounting']['run_id'] == directory.name
    assert manifest['costs']


def test_resume_complete_run_does_not_load_models_or_repeat_calls(setup):
    directory = invoke(setup)
    LIFETIMES.clear()
    run(resume_dir=directory, director_factory=FakeDirector, backend_factory=FakeFlux)
    assert LIFETIMES == []
    assert (FakeDirector.calls, FakeFlux.calls) == (4, 6)


def test_stage_direct_then_render_then_observe_uses_frozen_snapshot(setup):
    directory = invoke(setup, stage='direct')
    assert FakeFlux.calls == 0
    (setup[0].parent / 'reference.jpg').unlink()
    setup[0].unlink()
    run(resume_dir=directory, stage='render', director_factory=FakeDirector, backend_factory=FakeFlux)
    assert FakeFlux.calls == 6 and FakeDirector.calls == 1
    run(resume_dir=directory, stage='observe', director_factory=FakeDirector, backend_factory=FakeFlux)
    assert FakeDirector.calls == 4


def test_twelve_directions_batched_with_prior_context(setup):
    source, config = setup
    directory = run(source, config, directions=12, stage='direct', batch_size=3, director_factory=FakeDirector)
    assert FakeDirector.calls == 4
    assert [json.loads((directory / f'director_batch_{i:02d}.json').read_text())['metadata']['prior_count']
            for i in range(4)] == [0, 3, 6, 9]
    assert len(json.loads((directory / 'directions.json').read_text())['directions']) == 12


def test_saved_flux_output_recovered_after_registration_crash(setup, monkeypatch):
    directory = invoke(setup, stage='direct')
    original = Ledger.register_artifact
    crashed = False
    def fail_once(self, job_id, path, *args, **kwargs):
        nonlocal crashed
        if str(path) == 'd01-r1.png' and not crashed:
            crashed = True
            raise OSError('Simulated crash after PNG saved')
        return original(self, job_id, path, *args, **kwargs)
    monkeypatch.setattr(Ledger, 'register_artifact', fail_once)
    with pytest.raises(OSError, match='PNG saved'):
        run(resume_dir=directory, stage='render', director_factory=FakeDirector, backend_factory=FakeFlux)
    assert FakeFlux.calls == 1
    monkeypatch.setattr(Ledger, 'register_artifact', original)
    run(resume_dir=directory, stage='render', director_factory=FakeDirector, backend_factory=FakeFlux)
    assert FakeFlux.calls == 6  # saved first image reused


def test_reserved_flux_without_bytes_is_uncertain_never_repeated(setup):
    directory = invoke(setup, stage='direct')
    class FailedFlux(FakeFlux):
        calls = 0
        def generate(self, prompt, seed):
            type(self).calls += 1
            raise RuntimeError('Inference lost before saving')
    with pytest.raises(RuntimeError, match='Inference lost'):
        run(resume_dir=directory, stage='render', director_factory=FakeDirector, backend_factory=FailedFlux)
    with pytest.raises(UncertainInvocation):
        run(resume_dir=directory, stage='render', director_factory=FakeDirector, backend_factory=FailedFlux)
    assert FailedFlux.calls == 1
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'uncertain'


def test_saved_director_response_recovered_after_registration_crash(setup, monkeypatch):
    original = Ledger.register_artifact
    def fail(self, job_id, path, *args, **kwargs):
        if path == 'director_batch_00.json':
            raise OSError('Simulated response registration crash')
        return original(self, job_id, path, *args, **kwargs)
    monkeypatch.setattr(Ledger, 'register_artifact', fail)
    with pytest.raises(OSError):
        invoke(setup, stage='direct', job_id='crash')
    assert FakeDirector.calls == 1
    monkeypatch.setattr(Ledger, 'register_artifact', original)
    run(resume_dir=setup[1].output_dir / 'crash', stage='direct', director_factory=FakeDirector)
    assert FakeDirector.calls == 1


def test_mutated_reference_snapshot_blocks_models(setup):
    directory = invoke(setup, stage='direct')
    (directory / 'references/source.jpg').write_bytes(b'tampered')
    with pytest.raises(SpecMismatch, match='missing or changed'):
        run(resume_dir=directory, stage='render', backend_factory=FakeFlux)
    assert FakeFlux.calls == 0


def test_invalid_director_output_saved_without_substitution_or_flux(setup):
    class BadDirector(FakeDirector):
        calls = 0
        def propose_directions(self, packet, directory, expected_ids, prior):
            from kusama.director import DirectorOutputError
            type(self).calls += 1
            raise DirectorOutputError('Bad JSON', {'output': None, 'raw_response': 'not valid'})
    source, config = setup
    with pytest.raises(Exception, match='Bad JSON'):
        run(source, config, directions=3, stage='all', job_id='bad', director_factory=BadDirector, backend_factory=FakeFlux)
    directory = config.output_dir / 'bad'
    assert json.loads((directory / 'director_batch_00.json').read_text())['raw_response'] == 'not valid'
    with pytest.raises(ValueError, match='no automatic replacement'):
        run(resume_dir=directory, director_factory=BadDirector, backend_factory=FakeFlux)
    assert BadDirector.calls == 1 and FakeFlux.calls == 0


def test_packet_requires_verified_local_images_and_unique_ids(setup):
    source, _ = setup
    value = json.loads(source.read_text())
    value['references'][0]['image_path'] = None
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='one to six'):
        load_packet(source)


@pytest.mark.parametrize('fixture_name,core_lengths,batch_size', [
    ('qwen_maya_directions_first_response.json', [390, 491, 552], 3),
    ('qwen_maya_single_direction_response.json', [917], 1),
])
def test_actual_qwen_response_revalidated_without_rewriting_or_model_call(setup, fixture_name, core_lengths, batch_size):
    from kusama.artifacts import sha256
    from kusama.director import DirectorOutputError
    fixture = json.loads((Path(__file__).parent / 'fixtures' / fixture_name).read_text())
    actual = json.loads(fixture['raw_response'])
    assert [len(d['core_move']) for d in actual['directions']] == core_lengths
    source, config = setup
    packet = json.loads(source.read_text())
    prototype = packet['references'][0]
    ids = sorted({r['source_id'] for d in actual['directions'] for r in d['source_relationships']})
    packet['references'] = [{**prototype, 'id': identifier,
        'image_path': prototype['image_path'] if i == 0 else None} for i, identifier in enumerate(ids)]
    source.write_text(json.dumps(packet))
    class FormerlyRejectedDirector(FakeDirector):
        calls = 0
        def propose_directions(self, packet, directory, expected_ids, prior):
            type(self).calls += 1
            raise DirectorOutputError(fixture['parse_error'], fixture)
    with pytest.raises(DirectorOutputError):
        run(source, config, directions=len(core_lengths), stage='direct', job_id='bounds', batch_size=batch_size,
            director_factory=FormerlyRejectedDirector, backend_factory=FakeFlux)
    directory = config.output_dir / 'bounds'
    evidence = directory / 'director_batch_00.json'
    original_bytes, original_hash = evidence.read_bytes(), sha256(evidence)
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        before_calls = ledger.call_count('bounds')
    run(resume_dir=directory, stage='direct', revalidate_saved=True,
        director_factory=FormerlyRejectedDirector, backend_factory=FakeFlux)
    sidecar = json.loads((directory / 'director_batch_00.revalidated.json').read_text())
    assert sidecar['original_response_sha256'] == original_hash
    assert sidecar['original_parse_error'] == fixture['parse_error']
    assert sidecar['validator_code_sha256']['src/kusama/director.py']
    assert sidecar['output'] == actual
    assert evidence.read_bytes() == original_bytes
    assert FormerlyRejectedDirector.calls == 1 and FakeFlux.calls == 0
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.call_count('bounds') == before_calls
        records = {r['path']: r for r in ledger.artifacts('bounds')}
        assert records['director_batch_00.json']['sha256'] == original_hash
        assert records['director_batch_00.revalidated.json']['kind'] == 'model_response_revalidation'
    # The accepted sidecar is reusable without repeating the explicit flag.
    run(resume_dir=directory, stage='direct', director_factory=FormerlyRejectedDirector)
    assert FormerlyRejectedDirector.calls == 1
    assert json.loads((directory / 'directions.json').read_text()) == actual


def test_revalidation_cannot_repair_bad_json_or_invade_flux_prompt_bound(setup):
    from kusama.director import DirectorOutputError
    source, config = setup
    class StillInvalidDirector(FakeDirector):
        calls = 0
        def propose_directions(self, packet, directory, expected_ids, prior):
            type(self).calls += 1
            output = directions(expected_ids)
            output['directions'][0]['flux_prompt'] = 'x' * 2001
            raise DirectorOutputError('Too long prompt', {'output': None, 'raw_response': json.dumps(output)})
    with pytest.raises(DirectorOutputError):
        run(source, config, directions=3, stage='direct', job_id='still-invalid', director_factory=StillInvalidDirector)
    directory = config.output_dir / 'still-invalid'
    with pytest.raises(ValueError, match='flux_prompt'):
        run(resume_dir=directory, stage='direct', revalidate_saved=True, director_factory=StillInvalidDirector)
    assert not (directory / 'director_batch_00.revalidated.json').exists()
    assert StillInvalidDirector.calls == 1


def test_twelve_by_two_registers_balanced_216_question_plan(setup):
    source, config = setup
    directory = run(source, config, directions=12, observe=True,
                    director_factory=FakeDirector, backend_factory=FakeFlux)
    manifest = json.loads((directory / 'manifest.json').read_text())
    review = manifest['human_review']
    assert review['pair_count'] == 108
    assert review['planned_dimension_questions'] == 216
    assert review['design_metadata']['design'] == 'balanced_108_v1'
    assert set(review['design_metadata']['comparisons_per_candidate'].values()) == {9}
    assert manifest['batch_size'] == 1
    assert manifest['planned_calls'] == 48 and manifest['model_calls'] == 48
    assert len(manifest['direction_batches']) == 12
    assert all(len(batch['direction_ids']) == 1 for batch in manifest['direction_batches'])
    from kusama.reviews import ReviewStore
    collection = ReviewStore(config.output_dir).collection(directory.name)
    assert len(collection['pairs']) == 108 and len(collection['candidates']) == 24


def test_existing_review_schedule_preserved_on_resume(setup, monkeypatch):
    import kusama.creative as creative
    original = creative.diagnostic_pairs
    monkeypatch.setattr(creative, 'diagnostic_pairs', lambda ids: original(ids)[:1])
    directory = invoke(setup)
    from kusama.reviews import ReviewStore
    store = ReviewStore(setup[1].output_dir)
    original_collection = store.collection(directory.name)
    assert len(original_collection['pairs']) == 1
    monkeypatch.setattr(creative, 'diagnostic_pairs', original)
    run(resume_dir=directory, director_factory=FakeDirector, backend_factory=FakeFlux)
    assert store.collection(directory.name) == original_collection
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['human_review']['design_metadata']['design'] == 'existing_immutable_collection'
    assert manifest['human_review']['planned_dimension_questions'] == 2


@pytest.mark.parametrize('batch_size', [0, 4, True, 1.5])
def test_invalid_batch_size_rejected_before_model_load(setup, batch_size):
    source, config = setup
    with pytest.raises(ValueError, match='batch size'):
        run(source, config, batch_size=batch_size, director_factory=FakeDirector)
    assert LIFETIMES == []


def test_one_direction_default_and_experiment_note_are_frozen(setup):
    source, config = setup
    directory = run(source, config, directions=3, stage='direct', note='Supersedes a truncated pilot.',
                    director_factory=FakeDirector)
    assert FakeDirector.calls == 3
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['note'] == 'Supersedes a truncated pilot.'
    assert manifest['batch_size'] == 1
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        spec = ledger.get_job(directory.name)['spec']
        assert spec['batch_size'] == 1 and spec['note'] == manifest['note']
    run(resume_dir=directory, stage='direct', director_factory=FakeDirector)
    assert FakeDirector.calls == 3


def test_verbose_observer_revalidation_does_not_repeat_observation_or_render(setup):
    from kusama.director import DirectorOutputError
    from kusama.artifacts import sha256
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_single_direction_response.json').read_text())
    prose = json.loads(fixture['raw_response'])['directions'][0]['core_move']
    class PreviouslyStrictObserver(FakeDirector):
        calls = 0
        def observe_candidates(self, brief, direction, candidates, directory):
            result = super().observe_candidates(brief, direction, candidates, directory)
            result['output']['visible_observations'][0]['observations'][0] = prose
            result['output']['finished_quality']['reason'] = prose
            result['output']['revision']['expected_effect'] = prose
            raw = json.dumps(result['output'])
            raise DirectorOutputError('Previous observer descriptive bound', {
                'output': None, 'raw_response': raw, 'parse_error': 'Previous observer descriptive bound'})
    source, config = setup
    with pytest.raises(DirectorOutputError):
        run(source, config, directions=1, observe=True, job_id='observer-bound',
            director_factory=PreviouslyStrictObserver, backend_factory=FakeFlux)
    directory = config.output_dir / 'observer-bound'
    original = directory / 'observer_d01.json'
    original_bytes, digest = original.read_bytes(), sha256(original)
    assert PreviouslyStrictObserver.calls == 2 and FakeFlux.calls == 2
    run(resume_dir=directory, stage='observe', revalidate_saved=True,
        director_factory=PreviouslyStrictObserver, backend_factory=FakeFlux)
    assert PreviouslyStrictObserver.calls == 2 and FakeFlux.calls == 2
    assert original.read_bytes() == original_bytes
    sidecar = json.loads((directory / 'observer_d01.revalidated.json').read_text())
    assert sidecar['original_response_sha256'] == digest
    assert sidecar['output']['finished_quality']['reason'] == prose
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_human_review'
    assert manifest['observations'][0]['revalidation']['path'] == 'observer_d01.revalidated.json'


def test_all_challenge_coverage_validated_before_job_creation_or_model_load(setup):
    source, config = setup
    packet = json.loads(source.read_text())
    packet['composition_challenges'] = {'d01': 'A landscape built around an absent center.'}
    source.write_text(json.dumps(packet))
    with pytest.raises(ValueError, match='missing requested direction IDs'):
        run(source, config, directions=3, stage='direct', job_id='missing-challenges', director_factory=FakeDirector)
    assert FakeDirector.calls == 0 and LIFETIMES == []
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        with pytest.raises(KeyError):
            ledger.get_job('missing-challenges')


@pytest.mark.parametrize('challenges', [
    {'d00': 'Outside the direction range.'}, {'d13': 'Outside the direction range.'},
    {'../escaped': 'An invalid key.'}, {'d01': ' '}, {'d01': 'x' * 801}, {'d01': 123},
])
def test_composition_challenge_schema_rejects_invalid_ids_and_text(setup, challenges):
    source, _ = setup
    packet = json.loads(source.read_text())
    packet['composition_challenges'] = challenges
    source.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        load_packet(source)


def test_complete_scaffold_saved_immutably_with_operator_provenance(setup):
    source, config = setup
    packet = json.loads(source.read_text())
    packet['composition_challenges'] = {f'd{i:02d}': f'An operator spatial challenge {i}.' for i in range(1, 4)}
    source.write_text(json.dumps(packet))
    directory = run(source, config, directions=3, stage='direct', director_factory=FakeDirector)
    saved = json.loads((directory / 'packet.json').read_text())
    assert saved['composition_challenges'] == packet['composition_challenges']
    assert (directory / 'source_packet.json').read_bytes() == source.read_bytes()
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['composition_scaffolding']['origin'] == 'operator_authored_scaffold'
    assert manifest['composition_scaffolding']['direction_ids'] == ['d01', 'd02', 'd03']
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.get_job(directory.name)['spec']['packet']['composition_challenges'] == packet['composition_challenges']


def test_legacy_packet_normalized_shape_has_no_empty_scaffold(setup):
    packet, _ = load_packet(setup[0])
    assert 'composition_challenges' not in packet


def test_real_d10_surplus_closers_revalidated_without_changing_any_model_bytes(tmp_path):
    from kusama.creative import _recover_response
    from kusama.director import validate_directions
    from kusama.artifacts import sha256, write_json
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_d10_surplus_closers_response.json').read_text())
    decoded, end = json.JSONDecoder().raw_decode(fixture['raw_response'])
    assert fixture['raw_response'][end:] == ']}'
    assert [d['id'] for d in decoded['directions']] == ['d10']
    source_ids = {r['source_id'] for d in decoded['directions'] for r in d['source_relationships']}
    directory = tmp_path / 'retained-d10'
    directory.mkdir()
    evidence = directory / 'director_batch_09.json'
    write_json(evidence, fixture)
    original_bytes, digest = evidence.read_bytes(), sha256(evidence)
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('retained-d10', {'fixture': 'actual Qwen d10 response'}, 1, directory)
        ledger.reserve_call('retained-d10', 9)
        ledger.register_artifact('retained-d10', evidence.name, attempt_index=9, kind='model_response')
        validator = lambda value: validate_directions(value, ['d10'], source_ids)
        with pytest.raises(ValueError, match='no automatic replacement'):
            _recover_response(ledger, 'retained-d10', directory, 9, evidence.name, validator)
        recovered = _recover_response(ledger, 'retained-d10', directory, 9, evidence.name, validator, True)
        assert ledger.call_count('retained-d10') == 1
        assert recovered['output'] == decoded
        assert evidence.read_bytes() == original_bytes
        sidecar = json.loads((directory / 'director_batch_09.revalidated.json').read_text())
        assert sidecar['ignored_suffix'] == ']}'
        assert sidecar['parser_policy'] == 'complete_object_surplus_closing_delimiters_v1'
        assert sidecar['original_response_sha256'] == digest
        # A normal later resume reuses the explicit parser decision.
        assert _recover_response(ledger, 'retained-d10', directory, 9, evidence.name, validator)['output'] == decoded
        assert ledger.call_count('retained-d10') == 1
        # Suffix recovery cannot waive semantic identity or source constraints.
        with pytest.raises(ValueError, match='IDs'):
            _recover_response(ledger, 'retained-d10', directory, 9, evidence.name,
                lambda value: validate_directions(value, ['d11'], source_ids), True)


@pytest.mark.parametrize('suffix', [']}', ' \n}\t]\r', '}}}}', '\n])'])
def test_explicit_suffix_recovery_accepts_only_closing_delimiters(suffix):
    from kusama.creative import _parse_revalidation_response
    raw = '{"text":"A brace } and bracket ] inside a quoted string."}' + suffix
    if ')' in suffix:
        with pytest.raises(ValueError):
            _parse_revalidation_response(raw)
    else:
        value, parser = _parse_revalidation_response(raw)
        assert value['text'].endswith('quoted string.')
        assert parser['ignored_suffix'] == suffix


@pytest.mark.parametrize('raw', [
    '{"a":1} {"b":2}', '{"a":1}]} {"b":2}', '{"a":1}]} commentary',
    '{"a":1}]}123', '{"a":1}]}[]', 'prefix {"a":1}]}',
    '```json\n{"a":1}\n```', '{"a":', '[{"a":1}]]}',
    '{"a":1,"a":2}]}', '{"a":{"b":1,"b":2}}]}',
])
def test_explicit_suffix_recovery_rejects_data_prefixes_incomplete_json_and_duplicates(raw):
    from kusama.creative import _parse_revalidation_response
    with pytest.raises(ValueError):
        _parse_revalidation_response(raw)


class MissingCandidateOnce(FakeDirector):
    observer_calls = {}
    calls = 0
    def observe_candidates(self, brief, direction, candidates, directory):
        from kusama.director import DirectorOutputError
        response = super().observe_candidates(brief, direction, candidates, directory)
        count = type(self).observer_calls.get(direction['id'], 0) + 1
        type(self).observer_calls[direction['id']] = count
        if direction['id'] == 'd01' and count == 1:
            fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_observer_missing_candidate.json').read_text())
            raise DirectorOutputError(fixture['parse_error'], fixture)
        return response


def prepare_rejected_observer_run(setup, factory=MissingCandidateOnce):
    from kusama.director import DirectorOutputError
    factory.observer_calls = {}
    factory.calls = 0
    source, config = setup
    with pytest.raises(DirectorOutputError, match='both candidates'):
        run(source, config, directions=3, observe=True, job_id='retry-test',
            director_factory=factory, backend_factory=FakeFlux)
    return config.output_dir / 'retry-test'


def test_explicit_observer_retry_has_separate_budget_cost_and_source_bindings(setup):
    from kusama.artifacts import sha256
    from kusama.costs import CostJournal
    directory = prepare_rejected_observer_run(setup)
    original = directory / 'observer_d01.json'
    original_bytes, digest = original.read_bytes(), sha256(original)
    image_hashes = {p.name: sha256(p) for p in directory.glob('d??-r?.png')}
    with Ledger(setup[1].output_dir / 'jobs.sqlite3') as ledger:
        before = ledger.get_job('retry-test')
    run(resume_dir=directory, stage='observe', retry_rejected_observers=1,
        director_factory=MissingCandidateOnce, backend_factory=FakeFlux)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_human_review'
    assert manifest['planned_calls'] == manifest['model_calls'] == 12
    assert manifest['total_model_calls'] == 13
    assert manifest['authorized_maximum_model_calls'] == 15
    assert manifest['observer_recovery']['max_calls'] == 3
    assert manifest['observer_recovery']['model_calls'] == 1
    assert MissingCandidateOnce.observer_calls == {'d01': 2, 'd02': 1, 'd03': 1}
    assert FakeFlux.calls == 6
    assert original.read_bytes() == original_bytes
    assert {p.name: sha256(p) for p in directory.glob('d??-r?.png')} == image_hashes
    accepted = manifest['observations'][0]
    assert accepted['execution_job_id'] == 'retry-test_observer_recovery'
    assert accepted['attempt_index'] == 0
    assert accepted['path'] == 'observer_recovery/observer_d01_retry_1.json'
    assert accepted['supersedes']['sha256'] == digest
    assert accepted['supersedes']['attempt_index'] == 9
    with Ledger(setup[1].output_dir / 'jobs.sqlite3') as ledger:
        after = ledger.get_job('retry-test')
        assert after['max_calls'] == before['max_calls']
        assert after['spec_fingerprint'] == before['spec_fingerprint']
        assert ledger.get_attempt('retry-test', 9)['status'] == 'failed'
        child = ledger.get_job('retry-test_observer_recovery')
        assert child['parent_id'] is None
        assert child['status'] == 'completed' and child['max_calls'] == 3
        assert child['spec']['source_job_id'] == 'retry-test'
        assert child['spec']['packet_sha256'] == sha256(directory / 'packet.json')
        assert {c['path']: c['sha256'] for c in child['spec']['candidates']} == image_hashes
        assert ledger.get_attempt(child['job_id'], 0)['status'] == 'succeeded'
        assert ledger.get_attempt(child['job_id'], 0)['supersedes']['sha256'] == digest
        assert 'observer_d01_retry_1.json' in {a['path'] for a in ledger.artifacts(child['job_id'])}
        assert 'observer_recovery/observer_d01_retry_1.json' not in {a['path'] for a in ledger.artifacts('retry-test')}
    with CostJournal(setup[1].output_dir, read_only=True) as costs:
        assert costs.call('retry-test', 'attempt:9')['finish']['outcome'] == 'failed'
        assert costs.call('retry-test_observer_recovery', 'attempt:0')['finish']['outcome'] == 'succeeded'
    # Saved authorization and accepted responses resume without fresh invocations.
    run(resume_dir=directory, stage='observe', director_factory=MissingCandidateOnce, backend_factory=FakeFlux)
    assert MissingCandidateOnce.observer_calls == {'d01': 2, 'd02': 1, 'd03': 1}
    assert FakeFlux.calls == 6


def test_retry_output_saved_before_registration_is_recovered_without_reexecution(setup, monkeypatch):
    directory = prepare_rejected_observer_run(setup)
    original = Ledger.register_artifact
    crashed = False
    def crash_once(self, job_id, path, *args, **kwargs):
        nonlocal crashed
        if job_id.endswith('_observer_recovery') and path == 'observer_d01_retry_1.json' and not crashed:
            crashed = True
            raise OSError('Crash after retry response bytes were saved')
        return original(self, job_id, path, *args, **kwargs)
    monkeypatch.setattr(Ledger, 'register_artifact', crash_once)
    with pytest.raises(OSError, match='retry response'):
        run(resume_dir=directory, stage='observe', retry_rejected_observers=1,
            director_factory=MissingCandidateOnce, backend_factory=FakeFlux)
    assert MissingCandidateOnce.observer_calls['d01'] == 2
    monkeypatch.setattr(Ledger, 'register_artifact', original)
    run(resume_dir=directory, stage='observe', director_factory=MissingCandidateOnce, backend_factory=FakeFlux)
    assert MissingCandidateOnce.observer_calls['d01'] == 2
    assert FakeFlux.calls == 6


@pytest.mark.parametrize('retry_failure', ['schema', 'missing_output'])
def test_one_retry_limit_and_uncertain_retry_are_not_repeated(setup, retry_failure):
    from kusama.director import DirectorOutputError
    class FailedRetry(MissingCandidateOnce):
        def observe_candidates(self, brief, direction, candidates, directory):
            response = super().observe_candidates(brief, direction, candidates, directory)
            if direction['id'] == 'd01':
                if retry_failure == 'missing_output':
                    raise RuntimeError('Retry failed before any response was saved')
                output = response['output']
                output['visible_observations'] = output['visible_observations'][:1]
                raise DirectorOutputError('Retry still missing candidate', {
                    'raw_response': json.dumps(output), 'output': None, 'parse_error': 'Retry still missing candidate'})
            return response
    directory = prepare_rejected_observer_run(setup, FailedRetry)
    with pytest.raises((DirectorOutputError, RuntimeError)):
        run(resume_dir=directory, stage='observe', retry_rejected_observers=1,
            director_factory=FailedRetry, backend_factory=FakeFlux)
    assert FailedRetry.observer_calls['d01'] == 2
    expected = UncertainInvocation if retry_failure == 'missing_output' else ValueError
    with pytest.raises(expected):
        run(resume_dir=directory, stage='observe', retry_rejected_observers=1,
            director_factory=FailedRetry, backend_factory=FakeFlux)
    assert FailedRetry.observer_calls['d01'] == 2
    with Ledger(setup[1].output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.call_count('retry-test_observer_recovery') == 1
    assert FakeFlux.calls == 6


def test_missing_primary_observer_output_cannot_use_retry_authorization(setup):
    class LostObserver(FakeDirector):
        observer_calls = 0
        def observe_candidates(self, *args):
            type(self).observer_calls += 1
            raise RuntimeError('Original observer response was lost')
    source, config = setup
    with pytest.raises(RuntimeError, match='was lost'):
        run(source, config, directions=1, observe=True, job_id='lost-observer',
            director_factory=LostObserver, backend_factory=FakeFlux)
    directory = config.output_dir / 'lost-observer'
    with pytest.raises(UncertainInvocation):
        run(resume_dir=directory, stage='observe', retry_rejected_observers=1,
            director_factory=LostObserver, backend_factory=FakeFlux)
    assert LostObserver.observer_calls == 1
    assert not (directory / 'observer_recovery').exists()
    assert FakeFlux.calls == 2


def test_real_observer_missing_array_closer_has_one_recorded_syntax_insertion(tmp_path):
    from kusama.creative import _parse_revalidation_response, _recover_response
    from kusama.director import validate_observation
    from kusama.artifacts import sha256, write_json
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_observer_missing_array_closer.json').read_text())
    raw = fixture['raw_response']
    output, parser = _parse_revalidation_response(raw)
    validate_observation(output, ['d01-r1', 'd01-r2'])
    assert len(output['visible_observations']) == 2
    assert parser['parser_policy'] == 'observer_two_entries_missing_array_closer_v1'
    assert parser['ignored_suffix'] == ''
    assert len(parser['insertion_edits']) == 1
    edit = parser['insertion_edits'][0]
    assert edit['text'] == ']' and edit['unit'] == 'unicode_codepoints'
    assert raw[edit['offset']:].startswith(',"finished_quality"')
    assert json.loads(raw[:edit['offset']] + ']' + raw[edit['offset']:]) == output
    # All model statements are retained exactly; this test makes no aesthetic
    # or semantic correction to its possibly mistaken cross-candidate claims.
    assert output['finished_quality']['reason'] in raw
    directory = tmp_path / 'recovery'
    directory.mkdir()
    path = directory / 'observer_d01_retry_1.json'
    write_json(path, fixture)
    original_bytes, digest = path.read_bytes(), sha256(path)
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('recovery', {'fixture': 'actual observer retry response'}, 1, directory)
        ledger.reserve_call('recovery', 0)
        ledger.register_artifact('recovery', path.name, attempt_index=0, kind='model_response')
        validator = lambda value: validate_observation(value, ['d01-r1', 'd01-r2'])
        recovered = _recover_response(ledger, 'recovery', directory, 0, path.name, validator, True)
        assert recovered['output'] == output
        assert path.read_bytes() == original_bytes
        assert ledger.call_count('recovery') == 1
        sidecar = json.loads((directory / 'observer_d01_retry_1.revalidated.json').read_text())
        assert sidecar['original_response_sha256'] == digest
        assert sidecar['insertion_edits'] == parser['insertion_edits']
        assert sidecar['parser_policy'] == parser['parser_policy']
        assert _recover_response(ledger, 'recovery', directory, 0, path.name, validator)['output'] == output
        assert ledger.call_count('recovery') == 1


@pytest.mark.parametrize('corruption', [
    'missing_second_entry', 'incomplete_second_entry', 'string_lookalike', 'extra_object',
    'duplicate_key', 'another_missing_bracket', 'missing_outer_brace', 'commentary', 'wrong_next_key',
])
def test_missing_observer_closer_policy_rejects_other_structural_changes(corruption):
    from kusama.creative import _parse_revalidation_response
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_observer_missing_array_closer.json').read_text())
    raw = fixture['raw_response']
    if corruption == 'missing_second_entry':
        # A single-entry array cannot be syntactically turned into two entries.
        second = raw.index(',{"candidate_id":"d01-r2"')
        end = raw.index(',"finished_quality"')
        raw = raw[:second] + raw[end:]
    elif corruption == 'incomplete_second_entry':
        raw = raw.replace('manifest"]},"finished_quality"', 'manifest"],"finished_quality"')
    elif corruption == 'string_lookalike':
        raw = json.dumps({'text': raw})[:-1]  # The pattern is entirely inside a string, outer JSON incomplete.
    elif corruption == 'extra_object':
        raw += '{"other":1}'
    elif corruption == 'duplicate_key':
        raw = raw.replace('"uncertainty":', '"finished_quality":{},"uncertainty":')
    elif corruption == 'another_missing_bracket':
        raw = raw.replace('emotional void"]}', 'emotional void"}')
    elif corruption == 'missing_outer_brace':
        raw = raw[:-1]
    elif corruption == 'commentary':
        raw += ' commentary'
    else:
        raw = raw.replace(',"finished_quality":', ',"unrecognized_field":', 1)
    assert raw != fixture['raw_response']
    with pytest.raises(ValueError):
        _parse_revalidation_response(raw)


def test_missing_observer_closer_policy_still_requires_candidate_identity():
    from kusama.creative import _parse_revalidation_response
    from kusama.director import validate_observation
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_observer_missing_array_closer.json').read_text())
    parsed, _ = _parse_revalidation_response(fixture['raw_response'])
    with pytest.raises(ValueError, match='candidate binding'):
        validate_observation(parsed, ['different-image', 'd01-r2'])


def test_continue_rejected_observers_keeps_exhausted_retries_and_finishes_original_slots(setup):
    from kusama.director import DirectorOutputError
    from kusama.artifacts import sha256
    from kusama.costs import CostJournal

    class DiagnosticObserver(MissingCandidateOnce):
        def observe_candidates(self, brief, direction, candidates, directory):
            response = super().observe_candidates(brief, direction, candidates, directory)
            if direction['id'] in {'d02', 'd03'}:
                response['output']['visible_observations'] = response['output']['visible_observations'][:1]
                raise DirectorOutputError('Missing candidate B', {'output': None,
                    'raw_response': json.dumps(response['output']), 'parse_error': 'Missing candidate B'})
            return response

    directory = prepare_rejected_observer_run(setup, DiagnosticObserver)
    images = {path.name: sha256(path) for path in directory.glob('d??-r?.png')}
    with pytest.raises(DirectorOutputError):
        run(resume_dir=directory, stage='observe', retry_rejected_observers=1,
            director_factory=DiagnosticObserver, backend_factory=FakeFlux)
    assert DiagnosticObserver.observer_calls == {'d01': 2, 'd02': 2}
    retained = {path: path.read_bytes() for path in directory.rglob('observer_d*.json')}
    run(resume_dir=directory, stage='observe', continue_rejected_observers=True,
        revalidate_saved=True, director_factory=DiagnosticObserver, backend_factory=FakeFlux)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_human_review'
    assert manifest['observer_status'] == {
        'status': 'partial', 'attempted_pairs': 3, 'valid_pairs': 1, 'rejected_pairs': 2,
        'unresolved_direction_ids': ['d02', 'd03'], 'rejected_response_count': 4}
    assert [entry['direction_id'] for entry in manifest['observations']] == ['d01']
    assert [(entry['direction_id'], entry['role']) for entry in manifest['observer_rejections']] == [
        ('d01', 'original'), ('d02', 'original'), ('d02', 'retry'), ('d03', 'original')]
    assert json.loads((directory / 'observations.json').read_text()) == {'observations': manifest['observations']}
    assert json.loads((directory / 'observer_rejections.json').read_text()) == {'rejections': manifest['observer_rejections']}
    assert manifest['sessions'][-1]['continue_rejected_observers'] is True
    assert manifest['sessions'][-1]['retry_rejected_observers'] == 0
    assert manifest['model_calls'] == 12 and manifest['total_model_calls'] == 14
    assert manifest['observer_recovery']['model_calls'] == 2
    assert not (directory / 'observer_recovery/observer_d03_retry_1.json').exists()
    assert DiagnosticObserver.observer_calls == {'d01': 2, 'd02': 2, 'd03': 1}
    assert all(path.read_bytes() == content for path, content in retained.items())
    assert {path.name: sha256(path) for path in directory.glob('d??-r?.png')} == images
    with Ledger(setup[1].output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.get_job('retry-test_observer_recovery')['status'] == 'completed'
        for entry in manifest['observer_rejections']:
            assert ledger.get_attempt(entry['execution_job_id'], entry['attempt_index'])['status'] == 'failed'
            assert sha256(directory / entry['path']) == entry['sha256']
    with CostJournal(setup[1].output_dir, read_only=True) as costs:
        for entry in manifest['observer_rejections']:
            assert costs.call(entry['execution_job_id'], f"attempt:{entry['attempt_index']}")['finish']['outcome'] == 'failed'
    # A completed diagnostic projection reuses every byte, including failures.
    LIFETIMES.clear()
    run(resume_dir=directory, stage='observe', director_factory=DiagnosticObserver, backend_factory=FakeFlux)
    assert LIFETIMES == []
    assert DiagnosticObserver.observer_calls == {'d01': 2, 'd02': 2, 'd03': 1}
    assert FakeFlux.calls == 6


def test_continuation_can_complete_with_zero_valid_observations(setup):
    from kusama.director import DirectorOutputError
    class RejectedEveryPair(FakeDirector):
        def observe_candidates(self, *args):
            response = super().observe_candidates(*args)
            response['output']['visible_observations'].pop()
            raise DirectorOutputError('Both candidates required', {'output': None,
                'raw_response': json.dumps(response['output']), 'parse_error': 'Both candidates required'})
    source, config = setup
    directory = run(source, config, directions=3, observe=True, stage='direct', job_id='all-rejected',
        director_factory=RejectedEveryPair, backend_factory=FakeFlux)
    run(resume_dir=directory, stage='all', continue_rejected_observers=True,
        director_factory=RejectedEveryPair, backend_factory=FakeFlux)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_human_review'
    assert manifest['observations'] == []
    assert manifest['observer_status'] == {
        'status': 'partial', 'attempted_pairs': 3, 'valid_pairs': 0, 'rejected_pairs': 3,
        'unresolved_direction_ids': ['d01', 'd02', 'd03'], 'rejected_response_count': 3}
    assert not (directory / 'observer_recovery').exists()
    assert manifest['model_calls'] == 12


@pytest.mark.parametrize('when', ['new', 'saved'])
def test_diagnostic_continuation_still_halts_uncertain_original_output(setup, when):
    class LostObserver(FakeDirector):
        observer_calls = 0
        def observe_candidates(self, *args):
            type(self).observer_calls += 1
            raise RuntimeError('No saved model output')
    source, config = setup
    directory = run(source, config, directions=1, observe=True, stage='direct', job_id='uncertain-diagnostic',
        director_factory=LostObserver, backend_factory=FakeFlux)
    if when == 'saved':
        with pytest.raises(RuntimeError):
            run(resume_dir=directory, stage='all', director_factory=LostObserver, backend_factory=FakeFlux)
    with pytest.raises(RuntimeError if when == 'new' else UncertainInvocation):
        run(resume_dir=directory, stage='all', continue_rejected_observers=True,
            director_factory=LostObserver, backend_factory=FakeFlux)
    assert LostObserver.observer_calls == 1
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] in {'failed', 'uncertain'}
    assert manifest.get('observer_rejections', []) == []
    assert not (directory / 'observations.json').exists()


def test_explicit_revalidation_applies_to_new_observer_response_in_same_pass(setup):
    from kusama.director import DirectorOutputError
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_observer_missing_array_closer.json').read_text())
    class MissingArrayCloser(FakeDirector):
        observer_calls = 0
        def observe_candidates(self, brief, direction, candidates, directory):
            type(self).observer_calls += 1
            response = {**fixture, 'raw_response': fixture['raw_response'].replace('d01', direction['id'])}
            raise DirectorOutputError(response['parse_error'], response)
    source, config = setup
    directory = run(source, config, directions=3, observe=True, stage='direct', job_id='new-syntax-recovery',
        director_factory=MissingArrayCloser, backend_factory=FakeFlux)
    run(resume_dir=directory, stage='all', revalidate_saved=True, continue_rejected_observers=True,
        director_factory=MissingArrayCloser, backend_factory=FakeFlux)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['observer_status'] == {
        'status': 'complete', 'attempted_pairs': 3, 'valid_pairs': 3, 'rejected_pairs': 0,
        'unresolved_direction_ids': [], 'rejected_response_count': 0}
    assert MissingArrayCloser.observer_calls == 3
    assert manifest['model_calls'] == 12
    assert not (directory / 'observer_recovery').exists()
    for record in manifest['observations']:
        raw = json.loads((directory / record['path']).read_text())
        assert raw['output'] is None
        assert raw['raw_response'] == fixture['raw_response'].replace('d01', record['direction_id'])
        assert record['revalidation']['parser_policy'] == 'observer_two_entries_missing_array_closer_v1'
        assert len(record['output']['visible_observations']) == 2


def test_continue_rejected_observers_cli_flag_is_explicit_and_resume_only(monkeypatch, tmp_path):
    from kusama import creative
    calls = []
    monkeypatch.setattr(creative, 'run', lambda **kwargs: calls.append(kwargs) or tmp_path)
    creative.main(['--resume', str(tmp_path), '--stage', 'observe', '--continue-rejected-observers', '--revalidate-saved'])
    assert calls[0]['continue_rejected_observers'] is True
    assert calls[0]['revalidate_saved'] is True
    assert calls[0]['retry_rejected_observers'] == 0
    with pytest.raises(SystemExit):
        creative.main(['--packet', str(tmp_path / 'packet.json'), '--continue-rejected-observers'])
