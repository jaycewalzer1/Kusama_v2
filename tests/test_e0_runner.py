"""Offline E0 executor fixtures only; no GPU, real model, or invented human vote."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from kusama import e0_runner
from kusama.artifacts import sha256, write_json
from kusama.evaluation import prepare_protocol, protocol_status
from kusama.ledger import Ledger, SpecMismatch
from kusama.resources import OvernightConfig, SessionBudget, load_config
from kusama.reviews import ReviewStore, ReviewConflict


class FixtureIndex:
    root = None
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = {'index_id': 'fixture-index', 'encoder': {'revision': 'fixture'}}
        self.documents = []
        for i in range(2):
            path = self.directory / f'source{i}.png'
            self.documents.append({'id': f'source{i}', 'reference_id': f'source{i}', 'scope': 'whole',
                'path': str(path), 'sha256': sha256(path), 'source_family': f'family{i}',
                'reference': {'id': f'source{i}', 'title': f'Fixture reference {i}', 'description': 'Recorded fixture context',
                              'rights': {'image_display': {'status': 'allowed'}}}})
        self.by_id = {row['id']: row for row in self.documents}
    def query(self, text, **kwargs):
        return {'query': text, 'selected': [{'id': row['id']} for row in self.documents], 'fixture_only': True}
    def asset(self, identifier):
        return Path(self.by_id[identifier]['path'])


class FixtureBackend:
    calls, loads, mode = [], 0, 'success'
    def __init__(self, config):
        self.config = config
    def prepare(self):
        type(self).loads += 1
        return {'fixture_only': True, 'model_id': 'fixture-renderer'}
    def prompt_token_count(self, text):
        return 10
    def generate(self, instruction, seed, parent=None):
        if parent is not None:
            frozen = json.loads((parent.parents[1] / 'frozen.json').read_text())
            assert len(frozen['action_requests']) == 3  # Before any edit invocation.
            assert frozen['parent']['sha256'] == sha256(parent)
            assert frozen['edit_seeds'] == [23, 41]
        type(self).calls.append({'instruction': instruction, 'seed': seed, 'parent_sha256': sha256(parent) if parent else None})
        ordinal = len(type(self).calls)
        if self.mode == 'interrupt' and ordinal == 1:
            raise KeyboardInterrupt('fixture interruption')
        if self.mode == 'fail_one_edit' and ordinal == 2:
            raise RuntimeError('fixture technical render failure')
        if self.mode == 'fail_both' and ordinal in (2, 3):
            raise RuntimeError('fixture technical render failure')
        return Image.new('RGB', (self.config.width, self.config.height), (ordinal*11 % 255, seed % 255, 20))
    def close(self):
        pass


class FixtureDirector:
    loads = 0
    def __init__(self, config):
        pass
    def prepare(self):
        type(self).loads += 1
    def close(self):
        pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'overnight'
    root.mkdir()
    monkeypatch.setattr('kusama.resources.shutil.disk_usage', lambda p: SimpleNamespace(total=10**12, used=10**11, free=9*10**11))
    config = OvernightConfig.model_validate({**load_config().model_dump(), 'device': 'cpu',
              'render_profiles': {'fixture': {'width': 64, 'height': 64, 'steps': 1, 'guidance': 1.0,
                                              'precision': 'fp32', 'offload': 'none', 'vae_tiling': False}},
              'default_render_profile': 'fixture'})
    with SessionBudget(root, config):
        pass
    protocol = root / 'evaluation' / 'e0'
    protocol.mkdir(parents=True)
    identity = json.loads(Path('configs/e0.yaml').read_text())['protocol_id']
    holdout = protocol / 'fixture_holdouts.json'
    write_json(holdout, {'schema_version': 1, 'protocol_id': identity, 'briefs': [
        {'brief_id': f'e0-h{i+1:02}', 'brief': f'Synthetic sealed fixture {i}',
         'partition': 'sealed_test', 'authorship': 'assistant_provisional'} for i in range(8)]})
    prepare_protocol(protocol, holdout_file=holdout)
    index = root / 'index'
    index.mkdir()
    write_json(index / 'manifest.json', {'fixture_only': True})
    for i in range(2):
        Image.new('RGB', (64, 64), ('red', 'blue')[i]).save(index / f'source{i}.png')
    monkeypatch.setattr(e0_runner, 'RetrievalIndex', FixtureIndex)
    calls = []
    def multimodal(director, system, request, images, validate, **kwargs):
        assert len(images) == 3 and images[0][0] == 'parent'
        calls.append({'task': request['task'], 'parent_sha256': sha256(images[0][1]), 'sources': [row[0] for row in images[1:]]})
        return {'output': validate({'instruction': f'Fixture revision {len(calls)} changes the central shape.',
                 'intended_effect': 'Uncertain fixture prediction.', 'preserve': ['surrounding field'], 'uncertainty': 'May not improve.'}),
                'metadata': {'input_tokens': 10, 'generated_tokens': 30}, 'fixture_only': True}
    monkeypatch.setattr(e0_runner, '_multimodal_json', multimodal)
    FixtureBackend.calls, FixtureBackend.loads, FixtureBackend.mode = [], 0, 'success'
    FixtureDirector.loads = 0
    return root, protocol, index, calls


def execute(setup, **kwargs):
    root, protocol, index, _ = setup
    return e0_runner.run(protocol, index, root=root, backend_factory=FixtureBackend, director_factory=FixtureDirector, **kwargs)


def test_one_brief_all_actions_frozen_before_shared_seed_edits_and_exact_review_pool(setup):
    root, protocol, _, calls = setup
    result = execute(setup, brief_id='e0-d01')
    row = result['briefs'][0]
    assert row['generation_status'] == 'generation_complete' and row['parent_count'] == 1 and row['edit_count'] == 6
    assert len(FixtureBackend.calls) == 7 and len(calls) == 3
    assert len({call['parent_sha256'] for call in calls}) == 1
    assert all(call['sources'] == ['source0', 'source1'] for call in calls)
    assert [call['seed'] for call in FixtureBackend.calls[1:]] == [23, 41]*3
    assert len({call['parent_sha256'] for call in FixtureBackend.calls[1:]}) == 1
    assert row['human_judgments'] == row['human_pool_selections'] == 0
    store = ReviewStore(root)
    collection = store.collections()[0]
    assert collection['comparison_count'] == 6 and collection['candidate_count'] == 7 and collection['vote_count'] == 0
    pool = store.pool_assignment(collection['id'], 'reviewer-1')
    assert pool['complete'] is True and len(pool['candidates']) == 6
    assert all(candidate['id'] != 'e0-d01-parent' for candidate in pool['candidates'].values())
    assert protocol_status(protocol)['verified_edits'] == 6


def test_resume_reuses_all_evidence_without_any_model_load_or_call(setup):
    execute(setup, brief_id='e0-d01')
    first_loads = (FixtureBackend.loads, FixtureDirector.loads)
    first_calls = list(FixtureBackend.calls)
    result = execute(setup, brief_id='e0-d01')
    assert result['briefs'][0]['edit_count'] == 6
    assert FixtureBackend.calls == first_calls
    assert (FixtureBackend.loads, FixtureDirector.loads) == first_loads
    assert len(setup[3]) == 3
    with SessionBudget(setup[0]) as budget:
        assert budget.summary()['renderer_attempts'] == 7


def test_one_identical_technical_retry_is_counted_and_valid_output_not_retried(setup):
    FixtureBackend.mode = 'fail_one_edit'
    result = execute(setup, brief_id='e0-d01')
    assert result['briefs'][0]['edit_count'] == 6
    assert len(FixtureBackend.calls) == 8
    assert FixtureBackend.calls[1] == FixtureBackend.calls[2]
    assert protocol_status(setup[1])['recorded_edit_attempts'] == 7
    execute(setup, brief_id='e0-d01')
    assert len(FixtureBackend.calls) == 8


def test_unresolved_output_excludes_pair_and_standard_pool(setup):
    FixtureBackend.mode = 'fail_both'
    result = execute(setup, brief_id='e0-d01')
    assert result['briefs'][0]['generation_status'] == 'incomplete_outputs'
    assert result['briefs'][0]['edit_count'] == 5 and len(FixtureBackend.calls) == 8
    store = ReviewStore(setup[0])
    collection = store.collections()[0]
    assert collection['comparison_count'] == 5
    pool = store.pool_assignment(collection['id'], 'reviewer-1')
    assert pool['complete'] is False and pool['eligibility'] == 'incomplete_pool_excluded'
    execute(setup, brief_id='e0-d01')
    assert len(FixtureBackend.calls) == 8


def test_uncertain_execution_cannot_dispatch_any_retry(setup):
    FixtureBackend.mode = 'interrupt'
    result = execute(setup, brief_id='e0-d01')
    assert result['briefs'][0]['generation_status'] == 'uncertain_requires_reconciliation'
    FixtureBackend.mode = 'success'
    replay = execute(setup, brief_id='e0-d01')
    assert replay['briefs'][0]['generation_status'] == 'uncertain_requires_reconciliation'
    assert len(FixtureBackend.calls) == 1
    with SessionBudget(setup[0]) as budget:
        assert budget.summary()['renderer_attempts'] == 1 and len(budget.summary()['unsettled']) == 1


def test_default_partition_never_exposes_holdouts_and_missing_freeze_queues(setup):
    with pytest.raises(ValueError, match='partition'):
        execute(setup, brief_id='e0-h01')
    queued = execute(setup, brief_id='e0-h01', heldout=True)
    assert queued['status'] == 'queued_not_executable'
    assert FixtureBackend.calls == [] and setup[3] == []
    result = execute(setup)
    assert [row['brief_id'] for row in result['briefs']] == [f'e0-d{i+1:02}' for i in range(4)]
    assert len(FixtureBackend.calls) == 28 and len(setup[3]) == 12
    with pytest.raises(ValueError, match='fixtures cannot pass'):
        e0_runner.freeze_runtime(setup[1], setup[2], root=setup[0])


def test_corrupt_frozen_proposal_or_source_never_adds_inference(setup):
    execute(setup, brief_id='e0-d01')
    path = setup[1] / 'episodes' / 'e0-d01' / 'actions' / 'action-inquiry_relationship.json'
    value = json.loads(path.read_text())
    value['output']['instruction'] = 'Tampered execution request'
    write_json(path, value)
    with pytest.raises(SpecMismatch):
        execute(setup, brief_id='e0-d01')
    assert len(FixtureBackend.calls) == 7 and len(setup[3]) == 3


def test_unavailable_resources_pause_with_saved_evidence_not_missing_feedback(setup, monkeypatch):
    original = SessionBudget.reserve
    def limited(self, key, kind, *args, **kwargs):
        if kind == 'renderer' and self.summary()['renderer_attempts'] >= 2:
            from kusama.ledger import BudgetExceeded
            raise BudgetExceeded('Fixture cap exhausted')
        return original(self, key, kind, *args, **kwargs)
    monkeypatch.setattr(SessionBudget, 'reserve', limited)
    result = execute(setup, brief_id='e0-d01')
    assert result['briefs'][0]['generation_status'] == 'queued_resource_limit'
    state = json.loads((setup[1] / 'episodes' / 'e0-d01' / 'state.json').read_text())
    assert state['parent'] and len(state['children']) == 1
    assert state['human_judgments'] == 0 and len(FixtureBackend.calls) == 2


def test_aggregate_retry_quota_is_transactional_idempotent_and_not_reclaimed(setup):
    import sqlite3
    with SessionBudget(setup[0]) as budget:
        request = {'instruction': 'fixture', 'seed': 23}
        assert e0_runner._reserve_retry(budget, 'fixture-protocol', 'retry-1', request, limit=1)
        assert e0_runner._reserve_retry(budget, 'fixture-protocol', 'retry-1', request, limit=1)
        assert not e0_runner._reserve_retry(budget, 'fixture-protocol', 'retry-2', request, limit=1)
        with pytest.raises(SpecMismatch):
            e0_runner._reserve_retry(budget, 'fixture-protocol', 'retry-1', {**request, 'seed': 41}, limit=1)
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            budget.db.execute('DELETE FROM e0_retry_reservations')


def test_heldout_preflight_uses_seven_primaries_and_frozen_retry_reserve(setup):
    root, protocol, index_path, _ = setup
    index = FixtureIndex(index_path)
    evidence = root / 'fixture-throughput-evidence.json'
    write_json(evidence, {'fixture_only': True, 'test': 'admission arithmetic only'})
    freeze_path = root / 'fixture-runtime-freeze.json'
    with SessionBudget(root) as budget:
        identity = e0_runner._runtime_identity(budget, index, protocol)
        write_json(freeze_path, {**identity, 'protocol_id': json.loads((protocol/'manifest.json').read_text())['protocol_id'],
            'development_completed': True, **fixture_timing(),
            'model_load_seconds_per_brief': 20, 'retry_reserve': 25,
            'throughput_evidence': [{'path': str(evidence), 'sha256': sha256(evidence)}]})
        result = e0_runner.heldout_admission(protocol, index, budget, freeze_path)
        assert result['required_primary_attempts'] == 56
        assert result['remaining_retry_reserve'] == 25
        assert result['required_renderer_attempts'] == 81
        assert result['required_reasoning_tokens'] == 8*6*8192
        assert result['required_parent_attempts']==8 and result['required_edit_attempts']==48
        assert result['possible_schema_repair_calls']==24
        assert result['status'] == 'admitted'
        assert FixtureBackend.calls == []  # This tested arithmetic, no heldout execution.


def test_partial_resource_pause_resumes_with_new_counted_model_load(setup, monkeypatch):
    original = SessionBudget.reserve
    paused = {'enabled': True}
    def limited(self, key, kind, *args, **kwargs):
        if paused['enabled'] and kind == 'renderer' and self.summary()['renderer_attempts'] >= 2:
            from kusama.ledger import BudgetExceeded
            raise BudgetExceeded('Fixture temporary allocation pause')
        return original(self, key, kind, *args, **kwargs)
    monkeypatch.setattr(SessionBudget, 'reserve', limited)
    assert execute(setup, brief_id='e0-d01')['briefs'][0]['generation_status'] == 'queued_resource_limit'
    paused['enabled'] = False
    assert execute(setup, brief_id='e0-d01')['briefs'][0]['edit_count'] == 6
    assert len(FixtureBackend.calls) == 7
    assert FixtureBackend.loads == 3  # Parent, first edit stage, resumed edit stage.


def test_duplicate_action_requests_stop_before_any_edit_without_retry(setup, monkeypatch):
    original = e0_runner._multimodal_json
    def duplicate(*args, **kwargs):
        result = original(*args, **kwargs)
        result['output']['instruction'] = 'Identical fixture request'
        return result
    monkeypatch.setattr(e0_runner, '_multimodal_json', duplicate)
    result = execute(setup, brief_id='e0-d01')
    assert result['briefs'][0]['generation_status'] == 'duplicate_actions'
    assert len(FixtureBackend.calls) == 1 and len(setup[3]) == 3
    assert not (setup[1] / 'episodes' / 'e0-d01' / 'frozen.json').exists()


def test_all_protocol_reviewer_slots_match_exact_frozen_pair_and_pool_assignments(setup):
    execute(setup, brief_id='e0-d01')
    root, protocol, _, _ = setup
    store = ReviewStore(root)
    binding_path = next((protocol/'episodes'/'e0-d01').glob('review_assignment_binding-*.json'))
    binding = json.loads(binding_path.read_text())
    planned = {row['assignment_id']: row for row in e0_runner.verified_assignments(protocol) if row['brief_id'] == 'e0-d01'}
    assert len(binding['bindings']) == 21
    for row in binding['bindings']:
        original = planned[row['protocol_assignment_id']]
        assert row['actual_assignment_id'] == original['assignment_id']
        if row['kind'] == 'parent_edit':
            task = store.comparison(row['collection_id'], row['pair_id'], row['reviewer_slot'])
            assert {position: candidate['id'] for position, candidate in task['candidates'].items()} == original['candidate_order']
        else:
            task = store.pool_assignment(row['collection_id'], row['reviewer_slot'])
            assert [candidate['id'] for candidate in task['candidates'].values()] == original['candidate_ids']
        assert task['assignment_id'] == original['assignment_id']


def test_altered_frozen_plan_rejected_before_any_dispatch(setup):
    path = setup[1]/'review_assignments.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['candidate_order'] = dict(zip(('B', 'A'), rows[0]['candidate_order'].values()))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    with pytest.raises(ValueError, match='Frozen review assignments differ'):
        execute(setup, brief_id='e0-d01')
    assert FixtureBackend.calls == [] and FixtureDirector.loads == 0


def test_runtime_freezes_prompt_schema_code_and_review_identity(setup):
    root, protocol, index, _ = setup
    with SessionBudget(root) as budget:
        identity = e0_runner._runtime_identity(budget, FixtureIndex(index), protocol)
    assert identity['execution_policy']['action_system'] == e0_runner.ACTION_SYSTEM
    assert identity['execution_policy']['action_schema'] == e0_runner.ActionRequest.model_json_schema()
    assert identity['execution_policy']['scoped_code_sha256']['e0_runner.py'] == sha256(Path(e0_runner.__file__))
    assert identity['review_assignments_sha256'] == sha256(protocol/'review_assignments.jsonl')


def test_interrupted_qwen_action_is_explicit_uncertain_and_resume_does_not_retry(setup, monkeypatch):
    count = []
    def interrupted(*args, **kwargs):
        count.append(True)
        raise KeyboardInterrupt('Fixture interrupted Qwen')
    monkeypatch.setattr(e0_runner, '_multimodal_json', interrupted)
    assert execute(setup, brief_id='e0-d01')['briefs'][0]['generation_status'] == 'uncertain_requires_reconciliation'
    assert execute(setup, brief_id='e0-d01')['briefs'][0]['generation_status'] == 'uncertain_requires_reconciliation'
    assert len(count) == 1 and len(FixtureBackend.calls) == 1


@pytest.mark.parametrize('changed_field', ['instruction', 'seed', 'parent'])
def test_saved_charged_pixels_cannot_be_recovered_as_another_request(setup, changed_field):
    root = setup[0]
    folder = root/'recovery-probe'; folder.mkdir()
    original = {'instruction': 'Original fixture action', 'seed': 23, 'parent': None}
    changed = {**original, changed_field: {'instruction': 'Different action', 'seed': 41,
        'parent': {'path': str(root/'different-parent.png'), 'sha256': 'a'*64}}[changed_field]}
    with SessionBudget(root) as budget, Ledger(root/'jobs.sqlite3') as ledger:
        ledger.create_job('recovery_probe', {'fixture_only': True}, 1, folder)
        budget.reserve('recovery_probe:render:0', 'renderer')
        ledger.open_attempt('recovery_probe', 0, payload={'request': original})
        ledger.reserve_call('recovery_probe', 0)
        Image.new('RGB', (64, 64), 'red').save(folder/'image-00.png')
        renderer = e0_runner.RendererSession(budget, ledger, 'recovery_probe', folder,
            e0_runner.render_config(root, budget), FixtureBackend)
        try:
            with pytest.raises(SpecMismatch, match='charged render request changed'):
                renderer.attempt(0, changed)
            assert not (folder/'attempt-00.json').exists()
            assert ledger.get_attempt('recovery_probe', 0)['request'] == original
            assert budget.summary()['calls'][0]['status'] == 'running'
            recovered = renderer.attempt(0, original)
            assert recovered['status'] == 'completed' and recovered['request'] == original
            assert budget.summary()['renderer_attempts'] == 1
        finally:
            renderer.close()
    assert FixtureBackend.loads == 0 and FixtureBackend.calls == []


@pytest.mark.parametrize('contamination', ['manifest', 'unselected', 'selected_split', 'nested'])
def test_sealed_source_markers_block_before_retrieval_and_copy(setup, monkeypatch, contamination):
    index = FixtureIndex(setup[2])
    if contamination == 'manifest':
        index.manifest['source_pool'] = {'partition': 'heldout'}
    elif contamination == 'unselected':
        index.documents.append({'id': 'not_selected', 'split': 'sealed_test'})
    elif contamination == 'selected_split':
        index.documents[0]['split'] = 'test'
    else:
        index.documents[0]['reference']['lineage'] = {'sealed': True}
    monkeypatch.setattr(index, 'query', lambda *a, **k: pytest.fail('Contaminated pool must not affect ranking'))
    folder = setup[0]/'source-guard-probe'; folder.mkdir()
    with pytest.raises(ValueError, match='Sealed or test source pools'):
        e0_runner._references(index, 'Synthetic guard probe', folder, 23)
    assert not (folder/'sources').exists()
    assert FixtureBackend.loads == 0 and FixtureBackend.calls == []


def profile_fixture(setup, **changes):
    path = setup[0]/'fixture-e0-profile.json'
    path.write_text(json.dumps({'name': 'fixture_e0_384x512', 'width': 384, 'height': 512, 'steps': 1,
        'guidance': 1.0, 'precision': 'fp32', 'offload': 'none', 'vae_tiling': False, **changes}))
    return path


def test_named_e0_study_file_exact_declared_config_without_session_mutation(setup):
    config = load_config()
    budget = SimpleNamespace(root=setup[0], config=config)
    before = config.model_dump(mode='json')
    effective, metadata = e0_runner._effective_render_profile(budget, Path('configs/e0_study_384.json'))
    assert (effective.width, effective.height, effective.steps, effective.guidance) == (384, 512, 50, 4.0)
    assert (effective.precision, effective.offload, effective.vae_tiling) == ('bf16', 'model', True)
    assert metadata['name'] == 'flux_base4b_e0_384x512_v1'
    assert metadata['file_sha256'] == sha256(Path('configs/e0_study_384.json'))
    assert config.model_dump(mode='json') == before
    ordinary = e0_runner.render_config(setup[0], budget)
    assert (ordinary.width, ordinary.height) == (512, 640)


@pytest.mark.parametrize('changes', [{'width': 383}, {'height': '512'}, {'steps': True}, {'precision': 'auto'},
    {'offload': 'invented'}, {'vae_tiling': 'true'}, {'model_revision': 'a'*40}, {'renderer_attempt_cap': 999},
    {'guidance': float('inf')}, {'device': 'cuda'}])
def test_invalid_e0_profile_rejected_before_model_load_or_charge(setup, changes):
    path = profile_fixture(setup, **changes)
    with pytest.raises(ValueError): execute(setup, brief_id='e0-d01', render_profile=path)
    assert FixtureBackend.loads == FixtureDirector.loads == 0 and FixtureBackend.calls == []
    with SessionBudget(setup[0]) as budget: assert budget.summary()['renderer_attempts'] == 0


def test_explicit_profile_freezes_all_stages_all_briefs_and_resumes_without_calls(setup):
    path = profile_fixture(setup)
    result = execute(setup, brief_id='e0-d01', render_profile=path)
    assert result['briefs'][0]['edit_count'] == 6
    gate = json.loads(Path(result['briefs'][0]['render_profile_gate']['path']).read_text())
    assert gate['status'] == 'fixture_or_missing_live_timing' and gate['completed_primary_output_count'] == 7
    assert all(row['size'] == [384, 512] for row in gate['verified_images'])
    identity = json.loads((setup[1]/'episodes'/'e0-d01'/'runtime_identity.json').read_text())
    assert identity['render_profile']['file_sha256'] == sha256(path)
    assert json.loads((setup[1]/'render_profile.json').read_text())['render_config']['width'] == 384
    with Ledger(setup[0]/'jobs.sqlite3') as ledger:
        for stage in ('parent', 'actions', 'edits'):
            job = ledger.get_job('e0_overnight_v2_e0-d01_'+stage)
            assert job['spec']['runtime']['render_profile'] == identity['render_profile']
    first = len(FixtureBackend.calls), FixtureBackend.loads, FixtureDirector.loads
    execute(setup, brief_id='e0-d01', render_profile=path)
    assert (len(FixtureBackend.calls), FixtureBackend.loads, FixtureDirector.loads) == first
    for bid in ('e0-d01', 'e0-d02'):
        with pytest.raises(ValueError, match='Immutable artifact already differs'):
            execute(setup, brief_id=bid)  # Omitting explicit profile may not switch to session default.
    assert (len(FixtureBackend.calls), FixtureBackend.loads, FixtureDirector.loads) == first
    with SessionBudget(setup[0]) as budget:
        assert budget.config.default_render_profile == 'fixture'
        assert budget.config.render_profiles['fixture']['width'] == 64


def test_profile_file_byte_change_blocks_resume_even_when_settings_identical(setup):
    path = profile_fixture(setup)
    execute(setup, brief_id='e0-d01', render_profile=path)
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='Immutable artifact already differs'):
        execute(setup, brief_id='e0-d01', render_profile=path)
    assert len(FixtureBackend.calls) == 7


def test_profile_change_after_parent_blocks_before_qwen_load(setup, monkeypatch):
    path = profile_fixture(setup)
    original = FixtureBackend.generate
    def change_profile(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        path.write_text(path.read_text()+'\n')
        return result
    monkeypatch.setattr(FixtureBackend, 'generate', change_profile)
    with pytest.raises(SpecMismatch, match='profile file changed'):
        execute(setup, brief_id='e0-d01', render_profile=path)
    assert len(FixtureBackend.calls) == 1 and FixtureDirector.loads == 0


def test_heldout_admission_requires_identical_explicit_profile(setup):
    root, protocol, index_path, _ = setup
    profile = profile_fixture(setup)
    evidence, freeze = root/'fixture-evidence.json', root/'fixture-freeze.json'
    write_json(evidence, {'fixture_only': True, 'purpose': 'admission arithmetic only'})
    with SessionBudget(root) as budget:
        identity = e0_runner._runtime_identity(budget, FixtureIndex(index_path), protocol, profile)
        write_json(freeze, {**identity, 'protocol_id': 'e0_overnight_v2', 'development_completed': True,
            **fixture_timing(), 'model_load_seconds_per_brief': 20, 'retry_reserve': 0,
            'throughput_evidence': [{'path': str(evidence), 'sha256': sha256(evidence)}]})
        admitted = e0_runner.heldout_admission(protocol, FixtureIndex(index_path), budget, freeze, render_profile=profile)
        assert admitted['required_primary_attempts'] == 56 and admitted['status'] == 'admitted'
        with pytest.raises(SpecMismatch, match='identity differs'):
            e0_runner.heldout_admission(protocol, FixtureIndex(index_path), budget, freeze)
    assert FixtureBackend.calls == []


def test_freeze_runtime_refuses_mixed_development_profile_evidence(setup):
    profile = profile_fixture(setup)
    execute(setup, render_profile=profile)
    path = setup[1]/'episodes'/'e0-d01'/'runtime_identity.json'
    identity = json.loads(path.read_text()); identity['render_config']['width'] = 512
    write_json(path, identity)
    with pytest.raises(SpecMismatch, match='Development throughput used a different E0 render profile'):
        e0_runner.freeze_runtime(setup[1], setup[2], root=setup[0], render_profile=profile)


@pytest.mark.parametrize('command', ['run', 'freeze-runtime'])
def test_profile_cli_is_forwarded_to_both_entrypoints(setup, monkeypatch, command):
    received = []
    def capture(*args, **kwargs): received.append(kwargs); return {'fixture_only': True}
    monkeypatch.setattr(e0_runner, 'run' if command == 'run' else 'freeze_runtime', capture)
    assert e0_runner.main([command, '--protocol', str(setup[1]), '--index', str(setup[2]),
        '--root', str(setup[0]), '--render-profile', 'configs/e0_study_384.json']) == 0
    assert received[0]['render_profile'] == Path('configs/e0_study_384.json')


@pytest.mark.parametrize('kind', ['renderer', 'reasoning'])
def test_preload_uses_measured_reservation_time_and_never_loads_rejected_model(setup, monkeypatch, kind):
    from datetime import timedelta
    from kusama.ledger import BudgetExceeded
    root, protocol, index_path, _ = setup
    folder = root/('preload-'+kind); folder.mkdir()
    monkeypatch.setattr(SessionBudget, 'estimated_seconds', lambda self, requested_kind, supplied=0: 600)
    with SessionBudget(root) as budget, Ledger(root/'jobs.sqlite3') as ledger:
        budget.deadline = (e0_runner.now()+timedelta(minutes=budget.config.handoff_reserve_minutes, seconds=250)).isoformat()
        if kind == 'renderer':
            ledger.create_job('preload_renderer', {'fixture_only': True}, 1, folder)
            renderer = e0_runner.RendererSession(budget, ledger, 'preload_renderer', folder,
                e0_runner.render_config(root, budget), FixtureBackend)
            try:
                with pytest.raises(BudgetExceeded, match='before model load'):
                    renderer.attempt(0, {'instruction': 'Fixture only', 'seed': 23, 'parent': None})
            finally: renderer.close()
        else:
            parent = root/'fixture-parent.png'; Image.new('RGB', (64, 64), 'red').save(parent)
            refs = e0_runner._references(FixtureIndex(index_path), 'Fixture only', folder, 23)
            with pytest.raises(BudgetExceeded, match='before Qwen model load'):
                e0_runner._actions('Fixture only', 'e0-d01', {'path': str(parent), 'sha256': sha256(parent)}, refs,
                    folder, budget, ledger, 'fixture_preload', FixtureDirector)
        assert budget.summary()['renderer_attempts'] == 0 and budget.summary()['calls'] == []
    assert FixtureBackend.loads == FixtureDirector.loads == 0 and FixtureBackend.calls == []


def test_preload_adds_measured_model_load_margin(setup, monkeypatch):
    from kusama.costs import CostJournal
    class FixtureCosts:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def summary(self):
            return {'records': [
                {'start': {'model': 'FLUX fixture', 'scope': 'model_load'}, 'finish': {'runtime_seconds': 80}},
                {'start': {'model': 'Qwen fixture', 'scope': 'model_load'}, 'finish': {'runtime_seconds': 20}},
                {'start': {'model': 'FLUX fixture', 'scope': 'inference'}, 'finish': {'runtime_seconds': 900}}]}
    (setup[0]/'costs.sqlite3').touch()
    monkeypatch.setattr(e0_runner, 'CostJournal', FixtureCosts)
    budget = SimpleNamespace(root=setup[0], estimated_seconds=lambda kind, supplied: 300 if kind == 'renderer' else 100)
    assert e0_runner._preload_seconds(budget, 'renderer', 180) == 400
    assert e0_runner._preload_seconds(budget, 'reasoning', 90) == 125


def test_failed_duration_increases_forecast_without_becoming_a_successful_output():
    # Substitute timing envelopes only; these are never written to research artifacts.
    success = {'status': 'completed', 'backend': {'model_id': 'substitute-timing-contract'},
               'measurements': {'runtime_seconds': 100}, 'output': {'path': 'unused-fixture-path'}}
    failure = {**success, 'status': 'technical_failure', 'measurements': {'runtime_seconds': 420}, 'output': None}
    measured = [e0_runner._measured_render_time(row) for row in (success, failure)]
    assert max(measured) == 420
    assert 1.25*(56+25)*max(measured) == 42525
    assert failure['output'] is None and failure['status'] == 'technical_failure'
    assert e0_runner._measured_render_time({**failure, 'backend': {'model_id': 'fixture', 'fixture_only': True}}) is None
    for seconds in (None, 0, -1, float('nan'), float('inf'), True, '420'):
        assert e0_runner._measured_render_time({**failure, 'measurements': {'runtime_seconds': seconds}}) is None
    assert e0_runner._measured_render_time({**failure, 'status': 'uncertain'}) is None


def fixture_timing():
    return {'render_seconds':10,'parent_render_seconds':7,'edit_render_seconds':10,
        'reasoning_seconds':5,'schema_repair_seconds':6,'schema_repair_model_load_seconds':2,
        'schema_repair_max_per_action':1,'throughput_estimator_version':'e0_stage_max_with_bounded_schema_repairs_v2'}
