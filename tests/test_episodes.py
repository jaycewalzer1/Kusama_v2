"""Episode wrappers with fixture inference only; no live models or human labels."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from kusama import episodes
from kusama.artifacts import sha256, write_json
from kusama.inspiration import example
from kusama.ledger import BudgetExceeded, Ledger, SpecMismatch
from kusama.resources import OvernightConfig, SessionBudget, load_config
from kusama.reviews import ReviewStore


class FixtureBackend:
    calls = []
    loads = 0
    mode = 'success'

    def __init__(self, config):
        self.config = config

    def prepare(self):
        type(self).loads += 1
        return {'device': 'cpu', 'fixture_only': True}

    def prompt_token_count(self, instruction):
        return 12

    def generate(self, instruction, seed):
        type(self).calls.append(seed)
        if self.mode == 'fail':
            raise RuntimeError('deliberate fixture failure')
        if self.mode == 'interrupt':
            raise KeyboardInterrupt('deliberate fixture interruption')
        return Image.new('RGB', (self.config.width, self.config.height), (seed % 255, 23, 41))

    def close(self):
        pass


class FixtureDirector:
    loads = 0
    def __init__(self, config):
        self.config = config
    def prepare(self):
        type(self).loads += 1
    def close(self):
        pass


@pytest.fixture(autouse=True)
def fixture_limits(monkeypatch):
    monkeypatch.setattr('kusama.resources.shutil.disk_usage', lambda p: SimpleNamespace(total=10**12, used=10**11, free=9*10**11))
    FixtureBackend.calls, FixtureBackend.loads, FixtureBackend.mode = [], 0, 'success'
    FixtureDirector.loads = 0


def source_run(tmp_path, *, cap=160, valid_count=6, allocation=None, repair_cap=1):
    root = tmp_path / 'artifacts'
    root.mkdir()
    values = {**load_config().model_dump(), 'device': 'cpu', 'renderer_attempt_cap': cap,
              'schema_repair_cap': repair_cap,
              'render_profiles': {'fixture': {'width': 64, 'height': 64, 'steps': 1, 'guidance': 1.0,
                                             'precision': 'fp32', 'offload': 'none', 'vae_tiling': False}},
              'default_render_profile': 'fixture'}
    if allocation is not None:
        values['episode_allocation'] = allocation
    config = OvernightConfig.model_validate(values)
    with SessionBudget(root, config):
        pass
    run_id = 'fixture_episode'
    directory = root / 'runs' / run_id
    (directory / 'sources').mkdir(parents=True)
    docs = []
    for i in range(2):
        path = directory / 'sources' / f'source{i}.png'
        Image.new('RGB', (64, 64), ('red', 'blue')[i]).save(path)
        reference = {'id': f'source{i}', 'title': f'Fixture source {i}', 'description': 'Fixture description',
                     'url': f'https://example.invalid/source{i}', 'field_provenance': {'fixture': True},
                     'rights': {'image_display': {'status': 'allowed'}, 'derivatives': {'status': 'allowed'}},
                     'source_family': f'fixture-family-{i}', 'maker': None, 'date': None, 'medium': None}
        docs.append({'id': f'source{i}', 'reference_id': f'source{i}', 'region_id': None,
                     'path': f'source{i}.png', 'sha256': sha256(path), 'reference': reference})
    spec = {'brief': 'Fixture connection test, not an artistic evaluation.', 'source_documents': docs,
            'source_pairs': [['source0', 'source1'] for _ in range(6)]}
    connections = []
    with Ledger(root / 'jobs.sqlite3') as ledger:
        ledger.create_job(run_id, spec, 12, directory)
        write_json(directory / 'job.json', {'job_id': run_id})
        write_json(directory / 'input.json', spec)
        ledger.register_artifact(run_id, 'input.json', kind='snapshot')
        for i in range(valid_count):
            proposal = example(f'c{i+1:02d}', docs)
            proposal.update(title=f'Fixture direction {i}', transformation=f'Fixture transformation {i}')
            connections.append(proposal)
            filename = f'proposal_c{i+1:02d}.json'
            write_json(directory / filename, {'output': proposal, 'metadata': {'fixture_only': True}})
            ledger.open_attempt(run_id, 2*i)
            ledger.reserve_call(run_id, 2*i)
            ledger.save_attempt(run_id, 2*i, status='completed', payload={'response': filename})
            ledger.register_artifact(run_id, filename, attempt_index=2*i, kind='response')
    write_json(directory / 'manifest.json', {'brief': spec['brief'], 'connections': connections})
    return root, directory, run_id


def fake_local_call(calls):
    def call(director, system, request, images, validate, **kwargs):
        calls.append({'request': request, 'images': images})
        value = {'instruction': 'Make a gap in the central band.', 'intended_change': 'An opening appears.',
                 'intended_effect': 'Connection may become less certain.', 'target': 'Central band',
                 'region': [.2, .2, .8, .8], 'preserve': ['Keep surrounding field'],
                 'uncertainty': 'The whole may not improve.'}
        return {'output': validate(value), 'metadata': {'input_tokens': 8, 'generated_tokens': 32}, 'fixture_only': True}
    return call


def test_generation_wrapper_saves_measured_images_and_resumes_without_calls(tmp_path):
    root, directory, run_id = source_run(tmp_path)
    first = episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert first['status'] == 'needs_human_review'
    assert len(first['images']) == first['model_calls'] == 6
    assert FixtureBackend.loads == 1 and len(FixtureBackend.calls) == 6
    hashes = []
    for row in first['images']:
        path = directory / row['path']
        assert path.is_file() and path.parent.name == 'generation'
        assert sha256(path) == row['sha256']
        assert row['parent_image_id'] is row['parent_sha256'] is None
        assert row['measurements']['runtime_seconds'] >= 0
        assert row['measurements']['samples'] >= 2
        hashes.append(row['sha256'])
    second = episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert [row['sha256'] for row in second['images']] == hashes
    assert len(FixtureBackend.calls) == 6 and FixtureBackend.loads == 1
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 6
    assert ReviewStore(root).collections()[0]['vote_count'] == 0
    with Ledger(root / 'jobs.sqlite3') as ledger:
        assert ledger.get_job(run_id)['directory'] != ledger.get_job(run_id + '_render')['directory']


def test_proposal_corruption_or_insufficient_directions_never_dispatch(tmp_path):
    root, directory, _ = source_run(tmp_path)
    path = directory / 'proposal_c01.json'
    value = json.loads(path.read_text())
    value['output']['operation'] = 'arbitrary_shell'
    write_json(path, value)
    with pytest.raises(SpecMismatch):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert FixtureBackend.calls == [] and FixtureBackend.loads == 0
    other = tmp_path / 'other'
    other.mkdir()
    root, directory, _ = source_run(other, valid_count=5)
    with pytest.raises(ValueError, match='six'):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert FixtureBackend.calls == []


def test_failed_images_stay_charged_and_are_not_retried_or_human_rejections(tmp_path):
    root, directory, _ = source_run(tmp_path)
    FixtureBackend.mode = 'fail'
    first = episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert first['status'] == 'partial' and first['images'] == []
    assert len(first['failures']) == 6
    assert all(row.get('human_label') is None for row in first['failures'])
    second = episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert second['images'] == [] and len(FixtureBackend.calls) == 6
    assert FixtureBackend.loads == 1
    assert ReviewStore(root).collections() == []
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 6


def test_interrupted_renderer_never_silently_repeats(tmp_path):
    root, directory, run_id = source_run(tmp_path)
    FixtureBackend.mode = 'interrupt'
    with pytest.raises(KeyboardInterrupt):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    FixtureBackend.mode = 'success'
    with pytest.raises(RuntimeError, match='Uncertain'):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert len(FixtureBackend.calls) == 1
    with Ledger(root / 'jobs.sqlite3') as ledger:
        assert ledger.call_count(run_id + '_render') == 1
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 1


def test_global_renderer_cap_leaves_partial_saved_bytes(tmp_path):
    root, directory, _ = source_run(tmp_path, cap=2)
    with pytest.raises(BudgetExceeded):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    state = json.loads((directory / 'generation.json').read_text())
    assert state['status'] == 'partial' and len(state['images']) == 2
    assert len(FixtureBackend.calls) == 2
    assert all(sha256(directory / row['path']) == row['sha256'] for row in state['images'])
    with pytest.raises(BudgetExceeded):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert len(FixtureBackend.calls) == 2


def test_local_proposal_parent_lineage_and_resume_after_generation_replay(tmp_path, monkeypatch):
    root, directory, run_id = source_run(tmp_path)
    episodes.render(directory, root=root, backend_factory=FixtureBackend)
    calls = []
    monkeypatch.setattr(episodes, '_multimodal_json', fake_local_call(calls))
    proposals = episodes.propose_revisions(directory, root=root, count=4, director_factory=FixtureDirector)
    assert len(proposals) == 4 and len(calls) == 4
    assert Counter(row['selection']['allocation'] for row in proposals) == {'ordinary': 2, 'protected': 2}
    for row, call in zip(proposals, calls):
        candidate = row['selection']['candidate']
        assert sha256(call['images'][0][1]) == candidate['sha256']
        assert row['human_label'] is None
    episodes.render(directory, root=root, backend_factory=FixtureBackend)
    replay = episodes.propose_revisions(directory, root=root, count=4, director_factory=FixtureDirector)
    assert replay == proposals and len(calls) == 4
    with Ledger(root / 'jobs.sqlite3') as ledger:
        assert ledger.call_count(run_id + '_local') == 4
        assert ledger.get_job(run_id + '_local')['directory'] != str(directory)


def test_episode_initial_allocation_cannot_be_overridden_by_global_capacity(tmp_path):
    allocation = {'initial': 5, 'ordinary': 4, 'protected_continuations': 4, 'finishing_intervention': 2}
    root, directory, _ = source_run(tmp_path, allocation=allocation)
    result = episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert len(result['images']) == len(FixtureBackend.calls) == 5
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 5


def test_local_proposal_allocation_cannot_spend_reserved_other_categories(tmp_path, monkeypatch):
    allocation = {'initial': 6, 'ordinary': 0, 'protected_continuations': 4, 'finishing_intervention': 2}
    root, directory, _ = source_run(tmp_path, allocation=allocation)
    episodes.render(directory, root=root, backend_factory=FixtureBackend)
    calls = []
    monkeypatch.setattr(episodes, '_multimodal_json', fake_local_call(calls))
    with pytest.raises((BudgetExceeded, ValueError), match='(?i)(allocation|ordinary|protected)'):
        episodes.propose_revisions(directory, root=root, count=4, director_factory=FixtureDirector)
    assert calls == []


def test_empty_observation_pool_makes_no_job_or_model_call(tmp_path):
    root, directory, run_id = source_run(tmp_path)
    write_json(directory / 'continuations.json', {'continuations': []})
    result = episodes.observe(directory, root=root, director_factory=FixtureDirector)
    assert result == []
    assert FixtureDirector.loads == 0
    state = json.loads((directory / 'observations.json').read_text())
    assert state['observations'] == []
    with SessionBudget(root) as budget:
        assert budget.summary()['reasoning_tokens_charged_or_reserved'] == 0


def test_no_repair_configuration_uses_only_four_primary_calls(tmp_path, monkeypatch):
    root, directory, run_id = source_run(tmp_path, repair_cap=0)
    episodes.render(directory, root=root, backend_factory=FixtureBackend)
    calls = []
    monkeypatch.setattr(episodes, '_multimodal_json', fake_local_call(calls))
    assert len(episodes.propose_revisions(directory, root=root, count=4, director_factory=FixtureDirector)) == 4
    assert len(calls) == 4
    with Ledger(root / 'jobs.sqlite3') as ledger:
        assert ledger.call_count(run_id + '_local') == 4
        assert ledger.get_job(run_id + '_local')['max_calls'] == 4


def test_corrupted_local_proposal_cannot_execute_continuation(tmp_path, monkeypatch):
    root, directory, _ = source_run(tmp_path)
    episodes.render(directory, root=root, backend_factory=FixtureBackend)
    calls = []
    monkeypatch.setattr(episodes, '_multimodal_json', fake_local_call(calls))
    episodes.propose_revisions(directory, root=root, count=1, director_factory=FixtureDirector)
    path = directory / 'local' / 'local_01.json'
    value = json.loads(path.read_text())
    value['output']['instruction'] = 'tampered after model execution'
    write_json(path, value)
    def forbidden(*args, **kwargs):
        pytest.fail('Corrupted local proposal must not reach continuation execution')
    monkeypatch.setattr(episodes, 'ContinuationStore', forbidden)
    with pytest.raises(SpecMismatch):
        episodes.revise(directory, root=root)
