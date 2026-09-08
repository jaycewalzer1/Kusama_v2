"""Fixture-only observation-informed continuation; no real model calls."""
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from kusama import episodes, finishing
from kusama.artifacts import sha256, write_json
from kusama.continuation import ContinuationStore
from kusama.director import DirectorOutputError
from kusama.ledger import BudgetExceeded, Ledger, SpecMismatch
from kusama.resources import SessionBudget, canonical
from test_episodes import source_run, FixtureBackend, FixtureDirector, fake_local_call


class EditBackend:
    calls = []
    mode = 'success'
    def __init__(self, config):
        self.config = config
    def prepare(self, on_progress):
        on_progress({'device': 'cpu', 'fixture_only': True})
    def generate(self, instruction, seed, parent, **kwargs):
        self.on_inference_start()
        type(self).calls.append({'seed': seed, 'parent_sha256': sha256(parent)})
        if self.mode == 'fail':
            raise RuntimeError('Fixture renderer failure after charge')
        if self.mode == 'interrupt':
            raise KeyboardInterrupt('Fixture interruption after charge')
        return Image.new('RGB', (self.config.width, self.config.height), (31, seed, 71))
    def close(self):
        pass


@pytest.fixture(autouse=True)
def isolated_fixture_state(monkeypatch):
    monkeypatch.setattr('kusama.resources.shutil.disk_usage', lambda p: SimpleNamespace(total=10**12, used=10**11, free=9*10**11))
    FixtureBackend.calls, FixtureBackend.loads, FixtureBackend.mode = [], 0, 'success'
    FixtureDirector.loads = 0
    EditBackend.calls, EditBackend.mode = [], 'success'


def developed_run(tmp_path, monkeypatch, *, observed=True, finishing_cap=2):
    allocation = {'initial': 6, 'ordinary': 4, 'protected_continuations': 4, 'finishing_intervention': finishing_cap}
    root, directory, run_id = source_run(tmp_path, allocation=allocation)
    episodes.render(directory, root=root, backend_factory=FixtureBackend)
    monkeypatch.setattr(episodes, '_multimodal_json', fake_local_call([]))
    proposals = episodes.propose_revisions(directory, root=root, director_factory=FixtureDirector)
    with Ledger(root / 'jobs.sqlite3') as ledger, SessionBudget(root) as budget:
        local = ledger.get_job(run_id + '_local')
        source = ledger.get_job(run_id)['spec']
        config = episodes.render_config(root, budget)
    rows = []
    store = ContinuationStore(root)
    for proposal in proposals:
        selected = proposal['selection']['candidate']
        connection = next(c for c in local['spec']['connections'] if c['id'] == selected['connection_id'])
        refs = [{'id': d['reference_id'], 'title': d['reference']['title'], 'uri': d['reference']['url'],
                 'provenance': 'Fixture source only', 'permitted_uses': 'Fixture tests',
                 'source_family': d['reference']['source_family']} for d in source['source_documents']]
        request = {'title': 'Fixture first revision', 'brief': source['brief'],
            'selection': {'origin': 'operator', 'path': str(directory / selected['path']),
                'image_sha256': selected['sha256'], 'author': 'Fixture seeded policy', 'reason': 'Fixture exploratory selection'},
            'inquiry': {'question': source['brief'], 'stakes': 'Fixture', 'position': 'Fixture', 'viewing_conditions': 'Screen'},
            'references': refs, 'connection_context': [connection],
            'revision': {'author': 'Fixture director substitute', 'origin': 'model_proposal',
                'instruction': 'Make a central gap', 'intended_change': 'Gap appears', 'target': 'Central band',
                'method': 'masked', 'region': [.2, .2, .8, .8], 'seeds': [23, 41],
                'preservation_constraints': ['Keep surroundings'], 'visual_checks': ['Inspect the whole image'],
                'source_ids': [r['id'] for r in refs], 'connection_id': connection['id'],
                'relationship': connection['shared_relationship'], 'whole_work_plan': connection['composition']['whole']}}
        identifier = run_id + '_' + proposal['id']
        store.prepare(identifier, request, config)
        state = store.execute(identifier, EditBackend, resource_root=root)
        rows.append({'id': identifier, 'allocation': proposal['selection']['allocation']})
        if observed:
            for child in state['outputs']:
                store.observe(identifier, child['id'], canonical({'visible_changes': ['A fixture color patch interrupts the field'],
                    'preservation_failures': ['The rectangular edge remains apparent'], 'ambiguity': 'Fixture only; no artistic judgment'}),
                    'Fixture observer substitute', 'model_observation')
    write_json(directory / 'continuations.json', {'continuations': rows})
    write_json(directory / 'observations.json', {'fixture_marker': 'Original observation plan must remain unchanged'})
    EditBackend.calls = []
    return root, directory, run_id, rows


def capture_model_calls(monkeypatch, *, reject_primary=False, no_visible_change=False):
    calls = []
    def inference(director, system, request, images, validate, **kwargs):
        calls.append({'tokens': director.config.max_new_tokens, 'request': request,
                      'images': [(label, str(path), sha256(path)) for label, path in images], 'sampling': kwargs.get('sampling')})
        if reject_primary and len(calls) == 1:
            result = {'parse_error': 'Fixture invalid first proposal', 'raw_response': '{bad fixture}',
                      'metadata': {'input_tokens': 8, 'generated_tokens': 4}, 'fixture_only': True}
            raise DirectorOutputError(result['parse_error'], result)
        value = ({'instruction': 'Break the hard rectangular boundary with a narrow gap.',
                  'intended_change': 'The prior rectangular boundary opens locally.',
                  'intended_effect': 'Whole-work enclosure may become less rigid.', 'target': 'Prior rectangular boundary',
                  'region': [.15, .15, .85, .85], 'preserve': ['Retain the surrounding field'],
                  'uncertainty': 'The composited edge may remain apparent.'}
                 if director.config.max_new_tokens == 512 else
                 {'visible_changes': [] if no_visible_change else ['The fixture patch changes'], 'preservation_failures': [],
                  'ambiguity': 'Fixture comparison only; quality remains unknown.'})
        return {'output': validate(value), 'raw_response': canonical(value),
                'metadata': {'input_tokens': 12, 'generated_tokens': 50}, 'fixture_only': True}
    monkeypatch.setattr(finishing, '_multimodal_json', inference)
    return calls


def test_observation_informed_protected_descendant_stays_within_16_and_resumes(tmp_path, monkeypatch):
    root, directory, run_id, rows = developed_run(tmp_path, monkeypatch)
    originals = {name: (directory / name).read_bytes() for name in ('continuations.json', 'observations.json')}
    protected = next(row for row in rows if row['allocation'] == 'protected')
    parent_state = ContinuationStore(root).get(protected['id'])
    parent = parent_state['outputs'][0]
    calls = capture_model_calls(monkeypatch)
    result = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert result['attempt_accounting']['total_charged_attempts'] == 16
    assert result['attempt_accounting']['episode_cap'] == 16
    assert result['parent_continuation_id'] == protected['id'] and result['parent']['sha256'] == parent['sha256']
    assert len(result['outputs']) == 2 and len(EditBackend.calls) == 2
    assert {row['parent_sha256'] for row in EditBackend.calls} == {parent['sha256']}
    assert [row['tokens'] for row in calls] == [512, 256, 256]
    assert len(calls[0]['images']) == 3 and calls[0]['images'][0][2] == parent['sha256']
    assert calls[0]['request']['prior_visible_observations']
    assert all(row['sampling'] is None for row in calls[1:])
    assert result['human_quality'] is result['human_readiness'] is None
    assert result['reviews']['votes'] == 0
    observation_count = len(ContinuationStore(root).get(run_id + '_finish')['observations'])
    completed_loads = FixtureDirector.loads
    replay = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert replay['attempt_accounting']['total_charged_attempts'] == 16
    assert len(calls) == 3 and len(EditBackend.calls) == 2
    assert FixtureDirector.loads == completed_loads
    assert len(ContinuationStore(root).get(run_id + '_finish')['observations']) == observation_count == 2
    inspection = finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
    assert inspection['observations'] == replay['observations'] and len(calls) == 3
    assert all((directory / name).read_bytes() == data for name, data in originals.items())
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 16
    with Ledger(root / 'jobs.sqlite3') as ledger:
        assert ledger.get_job(run_id + '_finishing')['directory'] == str(directory / 'finishing')
        assert ledger.get_job(run_id + '_finishing_observation')['directory'] == str(directory / 'finishing_observation')


@pytest.mark.parametrize('missing', ['observations', 'allocation'])
def test_missing_evidence_or_reserved_allocation_does_not_call_model(tmp_path, monkeypatch, missing):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch, observed=missing != 'observations',
                                             finishing_cap=0 if missing == 'allocation' else 2)
    calls = capture_model_calls(monkeypatch)
    with pytest.raises((ValueError, BudgetExceeded), match='(?i)(observation|finishing)'):
        finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert calls == [] and EditBackend.calls == []
    with Ledger(root / 'jobs.sqlite3') as ledger:
        with pytest.raises(KeyError): ledger.get_job(run_id + '_finishing')


def test_proposal_cutoff_excludes_later_observations_and_repair_is_bounded(tmp_path, monkeypatch):
    root, directory, run_id, rows = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch, reject_primary=True)
    first = finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
    assert len(calls) == 2 and first['status'] == 'proposed' and not first['outputs']
    selected = first['parent_continuation_id']
    ContinuationStore(root).observe(selected, first['parent']['id'], 'LATER OUTCOME MUST NOT ENTER EARLIER INPUT',
                                    'Fixture observer', 'model_observation')
    second = finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
    assert second['prior_observations'] == first['prior_observations'] and len(calls) == 2
    assert 'LATER OUTCOME' not in (directory / 'finishing/input.json').read_text()
    assert (directory / 'finishing/proposal.json').exists() and (directory / 'finishing/proposal_repair.json').exists()
    with Ledger(root / 'jobs.sqlite3') as ledger:
        assert ledger.call_count(run_id + '_finishing') == 2


def test_charged_failures_use_finishing_slots_without_quality_labels(tmp_path, monkeypatch):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch)
    EditBackend.mode = 'fail'
    first = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert first['outputs'] == [] and first['observations'] == []
    assert first['attempt_accounting']['total_charged_attempts'] == 16
    assert first['human_quality'] is None
    EditBackend.mode = 'success'
    second = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert second['outputs'] == [] and len(EditBackend.calls) == 2 and len(calls) == 1


def test_changed_parent_bytes_block_before_new_reasoning(tmp_path, monkeypatch):
    root, directory, run_id, rows = developed_run(tmp_path, monkeypatch)
    selected = next(row for row in rows if row['allocation'] == 'protected')
    state = ContinuationStore(root).get(selected['id'])
    Path(state['outputs'][0]['path']).write_bytes(b'changed')
    calls = capture_model_calls(monkeypatch)
    with pytest.raises(ValueError, match='changed'):
        finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert calls == [] and EditBackend.calls == []


def test_interrupted_finishing_charge_is_retained_without_repeating_that_seed(tmp_path, monkeypatch):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch)
    EditBackend.mode = 'interrupt'
    with pytest.raises(KeyboardInterrupt):
        finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert [row['seed'] for row in EditBackend.calls] == [23]
    EditBackend.mode = 'success'
    resumed = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert [row['seed'] for row in EditBackend.calls] == [23, 41]
    assert resumed['attempt_accounting']['total_charged_attempts'] == 16
    assert len(resumed['outputs']) == 1 and resumed['human_quality'] is None
    assert [row['tokens'] for row in calls] == [512, 256]
    replay = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert len(EditBackend.calls) == 2 and len(calls) == 2
    assert replay['outputs'] == resumed['outputs']
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 16


@pytest.mark.parametrize('stage', ['propose', 'observe'])
def test_reasoning_budget_preflight_stops_before_finishing_model_load(tmp_path, monkeypatch, stage):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch)
    if stage == 'observe':
        finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
        finishing.finish(directory, root=root, stage='execute', backend_factory=EditBackend)
    before = FixtureDirector.loads, len(calls)
    original = SessionBudget.summary
    def exhausted(self):
        return {**original(self), 'reasoning_tokens_remaining': 0}
    monkeypatch.setattr(SessionBudget, 'summary', exhausted)
    with pytest.raises(BudgetExceeded, match='reasoning cap exhausted before model load'):
        finishing.finish(directory, root=root, stage=stage, director_factory=FixtureDirector)
    assert (FixtureDirector.loads, len(calls)) == before


@pytest.mark.parametrize('orphan', ['global', 'uncharged_attempt', 'response'])
def test_orphan_finishing_evidence_blocks_before_model_prepare(tmp_path, monkeypatch, orphan):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch)
    def paused(*args, **kwargs):
        raise BudgetExceeded('Fixture pause before model loading')
    with monkeypatch.context() as patch:
        patch.setattr(finishing, 'preflight_model_load', paused)
        with pytest.raises(BudgetExceeded):
            finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
    job_id = run_id+'_finishing'
    with Ledger(root/'jobs.sqlite3') as ledger, SessionBudget(root) as budget:
        if orphan == 'global':
            budget.reserve(job_id+':reasoning:0', 'reasoning', max_tokens=8192)
        elif orphan == 'uncharged_attempt':
            ledger.open_attempt(job_id, 0, payload={'fixture_only': True})
        else:
            write_json(directory/'finishing/proposal.json', {'output': {'fixture_only': True}})
    before = FixtureDirector.loads
    with pytest.raises(SpecMismatch):
        finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
    assert FixtureDirector.loads == before and calls == [] and EditBackend.calls == []


def test_execute_only_cannot_use_mutated_proposal_summary(tmp_path, monkeypatch):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch)
    finishing.finish(directory, root=root, stage='propose', director_factory=FixtureDirector)
    with Ledger(root/'jobs.sqlite3') as ledger:
        state = ledger.get_job(run_id+'_finishing')['payload']['state']
        state['proposal']['instruction'] = 'A different instruction absent from the charged raw response'
        ledger.update_job(run_id+'_finishing', payload={'state': state})
    with pytest.raises(SpecMismatch, match='differs from its charged raw response'):
        finishing.finish(directory, root=root, stage='execute', backend_factory=EditBackend)
    assert len(calls) == 1 and EditBackend.calls == []
    assert not (root/'continuations'/(run_id+'_finish')).exists()


def test_no_visible_change_observations_remain_valid_and_do_not_become_quality_labels(tmp_path, monkeypatch):
    root, directory, run_id, _ = developed_run(tmp_path, monkeypatch)
    calls = capture_model_calls(monkeypatch, no_visible_change=True)
    result = finishing.finish(directory, root=root, director_factory=FixtureDirector, backend_factory=EditBackend)
    assert len(calls) == 3 and len(result['observations']) == 2
    assert all(row['observation']['visible_changes'] == [] and row['error'] is None for row in result['observations'])
    assert result['human_quality'] is result['human_readiness'] is None
    assert result['reviews']['votes'] == 0
