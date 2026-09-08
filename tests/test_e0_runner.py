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
    pool = store.pool_assignment(collection['id'], 'fixture-reviewer')
    assert pool['complete'] is True and len(pool['candidates']) == 6
    assert all(candidate['id'] != 'e0-d01-parent' for candidate in pool['candidates'])
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
    pool = store.pool_assignment(collection['id'], 'fixture-reviewer')
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
