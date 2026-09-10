"""Controlled substitutes verify lineage/recovery; these are not artistic evidence."""
import json
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image
import pytest
from pydantic import ValidationError

from kusama.artifacts import sha256
from kusama.config import Config
from kusama.continuation import ContinuationStore, ContinuationRequest, read_continuations
from kusama.reviews import ReviewStore, ReviewConflict, ReviewNotFound
from kusama.studio import Studio, ProposalBatch


class Backend:
    calls = []
    def __init__(self, config):
        self.config = config
    def prepare(self, on_progress):
        on_progress({'device': 'cpu', 'test_double': True})
    def generate(self, prompt, seed, parent, **kwargs):
        self.on_inference_start()
        type(self).calls.append({'seed': seed, 'parent_sha256': sha256(parent)})
        return Image.new('RGB', (self.config.width, self.config.height), 'red' if seed == 23 else 'blue')
    def close(self):
        pass


def forbidden(config):
    pytest.fail('Saved output must be reused without another model load')


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / 'artifacts'
    root.mkdir()
    parent = root / 'initial.png'
    Image.new('RGB', (64, 64), 'white').save(parent)
    request = {'title': 'A continued inquiry', 'brief': 'How can an enclosure support and restrict?',
        'selection': {'origin': 'operator', 'path': str(parent), 'image_sha256': sha256(parent),
                      'author': 'test operator', 'reason': 'Declared plumbing experiment, no preference label'},
        'inquiry': {'question': 'What does the boundary do?', 'stakes': 'Ambivalence', 'position': 'A study',
                    'viewing_conditions': 'Screen'},
        'revision': {'author': 'test operator', 'origin': 'development_example',
                     'instruction': 'Interrupt the central boundary', 'intended_change': 'A more porous enclosure',
                     'target': 'central boundary', 'method': 'masked', 'region': [0.25, 0.25, 0.75, 0.75],
                     'seeds': [23, 41], 'preservation_constraints': ['Keep the surrounding image'],
                     'visual_checks': ['Did the boundary change?', 'Does the whole work still cohere?'],
                     'whole_work_plan': 'Keep figure and enclosure legible within the field'}}
    config = Config(output_dir=root, width=64, height=64, steps=1)
    Backend.calls = []
    return ContinuationStore(root), request, config


def test_prepare_freezes_explicit_selection_without_inference(setup):
    store, request, config = setup
    result = store.prepare('study', request, config)
    assert result['status'] == 'queued' and result['model_calls'] == 0 and Backend.calls == []
    assert result['selection']['origin'] == 'operator'
    assert result['reviews']['human_labels'] is None
    assert store.prepare('study', request, config) == result
    assert ReviewStore(store.root).collections() == []
    with sqlite3.connect(store.root / 'jobs.sqlite3') as db:
        assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] == 3
        assert db.execute('SELECT count(*) FROM reservations').fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match='immutable'):
            db.execute('DELETE FROM continuation_specs')


def test_two_seeds_use_same_parent_and_register_null_balanced_reviews(setup):
    store, request, config = setup
    before = store.prepare('study', request, config)
    result = store.execute('study', Backend)
    assert result['status'] == 'awaiting_selection'
    assert result['model_calls'] == 2 and len(result['outputs']) == 2
    assert {c['seed'] for c in Backend.calls} == {23, 41}
    assert {c['parent_sha256'] for c in Backend.calls} == {before['parent']['sha256']}
    assert all(c['parent_artifact_id'] == 'root' for c in result['outputs'])
    for artifact in result['outputs']:
        assert artifact['preservation']['outside_changed_pixels'] == 0
    assert store.execute('study', forbidden)['model_calls'] == 2
    assert len(Backend.calls) == 2
    reviews = ReviewStore(store.root)
    collection = reviews.collection(result['reviews']['collection_id'])
    assert len(collection['pairs']) == 2 and collection['schema_version'] == 2
    assert reviews.history(collection['id']) == {'votes': [], 'advancements': []}
    assignments = [reviews.comparison(collection['id'], p['id'], 'A person') for p in collection['pairs']]
    assert sum(c['candidates']['A']['id'] == 'image-001' for c in assignments) == 1
    assert len((store.root / 'continuations/study/review_assignments.jsonl').read_text().splitlines()) == 2
    assert (store.root / 'continuations/study/gallery.html').is_file()


def test_imports_only_real_explicit_advancement(setup):
    store, request, config = setup
    Image.new('RGB', (64, 64), 'yellow').save(store.root / 'other.png')
    reviews = ReviewStore(store.root)
    reviews.register_collection('original', title='Original', brief='Brief', candidates=[
        {'id': 'first', 'path': request['selection']['path']}, {'id': 'second', 'path': store.root / 'other.png'}])
    request['selection'] = {'origin': 'human_advancement', 'collection_id': 'original',
                            'advancement_submission_id': 'explicit-choice'}
    with pytest.raises(ReviewNotFound):
        store.prepare('human', request, config)
    assert not (store.root / 'jobs.sqlite3').exists()
    record = reviews.advance('original', {'reviewer': 'Actual named test reviewer', 'candidate_id': 'second',
                             'reason': 'Chosen explicitly in test fixture', 'submission_id': 'explicit-choice'})
    result = store.prepare('human', request, config)
    assert result['selection']['advancement_record'] == record
    assert result['parent']['sha256'] == sha256(store.root / 'other.png')
    assert reviews.history('original')['votes'] == []
    assert result['model_calls'] == 0


@pytest.mark.parametrize('change', ['hash', 'bounds', 'unsupported', 'unknown_source', 'same_seed'])
def test_invalid_inputs_do_not_queue_jobs(setup, change):
    store, request, config = setup
    if change == 'hash': request['selection']['image_sha256'] = '0' * 64
    if change == 'bounds': request['revision']['region'] = [0, 0, 1.1, 1]
    if change == 'unsupported': request['revision']['strength'] = 0.3
    if change == 'unknown_source': request['revision'].update(source_ids=['missing'], relationship='An analogy')
    if change == 'same_seed': request['revision']['seeds'] = [23, 23]
    with pytest.raises((ValueError, ReviewConflict)):
        store.prepare('bad', request, config)
    assert not (store.root / 'jobs.sqlite3').exists()


def test_changed_key_and_parent_bytes_fail(setup):
    store, request, config = setup
    result = store.prepare('study', request, config)
    request['revision']['instruction'] = 'A different edit'
    with pytest.raises(ValueError, match='frozen'):
        store.prepare('study', request, config)
    from pathlib import Path
    Path(result['parent']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed'):
        store.execute('study', forbidden)


def test_resume_uses_imported_parent_when_original_is_removed(setup):
    from pathlib import Path
    store, request, config = setup
    store.prepare('study', request, config)
    Path(request['selection']['path']).unlink()
    assert store.execute('study', Backend)['model_calls'] == 2


def test_recover_committed_raw_without_repeating_inference(setup, monkeypatch):
    import kusama.execution as execution
    store, request, config = setup
    store.prepare('study', request, config)
    original = execution._finish
    def failed(*args):
        raise OSError('postprocessing interrupted')
    monkeypatch.setattr(execution, '_finish', failed)
    first = store.execute('study', Backend)
    assert first['model_calls'] == 1 and first['reviews']['collection_id'] is None
    monkeypatch.setattr(execution, '_finish', original)
    result = store.execute('study', Backend)
    assert result['model_calls'] == 2 and len(Backend.calls) == 2
    assert len(result['outputs']) == 2


def test_uncertain_invocation_is_not_repeated(setup):
    store, request, config = setup
    store.prepare('study', request, config)
    class Uncertain(Backend):
        def generate(self, prompt, seed, parent, **kwargs):
            if seed == 23:
                self.on_inference_start()
                raise KeyboardInterrupt('possible invocation without bytes')
            return super().generate(prompt, seed, parent, **kwargs)
    with pytest.raises(KeyboardInterrupt):
        store.execute('study', Uncertain)
    result = store.execute('study', Backend)
    assert result['model_calls'] == 2
    assert [c['seed'] for c in Backend.calls] == [41]
    assert len(result['outputs']) == 1
    assert store.execute('study', forbidden)['model_calls'] == 2


def test_seed_variants_do_not_become_distinct_artistic_directions():
    common = {'id': 'a', 'title': 'One', 'intended_change': 'Same intervention',
              'preservation_constraints': ['Framing'], 'plan': 'a.json'}
    batch = {'id': 'batch', 'parent_artifact_id': 'root', 'author': 'operator', 'origin': 'operator',
             'directions': [common, common | {'id': 'b', 'plan': 'b.json'}]}
    with pytest.raises(ValidationError, match='distinct'):
        ProposalBatch.model_validate(batch)
    for direction in batch['directions']:
        direction['realization_group'] = 'same_action'
    assert len(ProposalBatch.model_validate(batch).directions) == 2


def test_empty_registry_read_is_read_only(tmp_path):
    assert read_continuations(tmp_path / 'absent') == []
    assert list(tmp_path.iterdir()) == []


def test_session_wrapper_counts_each_seed_and_reuses_completed_calls(setup):
    from kusama.resources import SessionBudget, load_config
    store, request, config = setup
    session_config = load_config().model_copy(update={'renderer_attempt_cap': 2})
    with SessionBudget(store.root, session_config):
        pass
    store.prepare('study', request, config)
    result = store.execute('study', Backend, resource_root=store.root)
    assert result['model_calls'] == 2
    with SessionBudget(store.root) as budget:
        state = budget.summary()
    assert state['renderer_attempts'] == 2 and state['renderer_remaining'] == 0
    assert {r['key'] for r in state['calls']} == {'continuation:study:seed:23', 'continuation:study:seed:41'}
    assert all(r['status'] == 'completed' for r in state['calls'])
    assert store.execute('study', forbidden, resource_root=store.root)['model_calls'] == 2
    with pytest.raises(ValueError, match='frozen'):
        store.execute('study', forbidden, resource_root=store.root / 'another-budget')


def test_workload_lock_prevents_model_loading_and_budget_exhaustion_stops_dispatch(setup):
    from kusama.resources import SessionBudget, load_config
    from kusama.ledger import JobLocked
    store, request, config = setup
    with SessionBudget(store.root, load_config().model_copy(update={'renderer_attempt_cap': 1})) as budget:
        store.prepare('study', request, config)
        with budget.workload_lock(), pytest.raises(JobLocked):
            store.execute('study', forbidden, resource_root=store.root)
        assert budget.summary()['renderer_attempts'] == 0
    result = store.execute('study', Backend, resource_root=store.root)
    assert len(Backend.calls) == 1 and result['model_calls'] == 1
    assert len(result['outputs']) == 1
    with SessionBudget(store.root) as budget:
        assert budget.summary()['renderer_attempts'] == 1
    pool = ReviewStore(store.root).pool_assignment(result['reviews']['collection_id'], 'Reviewer')
    assert not pool['complete'] and pool['eligibility'] == 'incomplete_pool_excluded'


def test_session_completion_waits_for_raw_commit_and_crash_never_repeats_seed(setup, monkeypatch):
    import kusama.execution as execution
    from kusama.resources import SessionBudget
    store, request, config = setup
    store.prepare('study', request, config)
    original = execution.save_image
    def interrupted_save(image, path):
        if path.name == 'attempt_0_raw.png':
            with SessionBudget(store.root) as budget:
                assert budget.summary()['calls'][0]['status'] == 'running'
            raise KeyboardInterrupt('Returned pixels were not yet committed')
        return original(image, path)
    monkeypatch.setattr(execution, 'save_image', interrupted_save)
    with pytest.raises(KeyboardInterrupt):
        store.execute('study', Backend, resource_root=store.root)
    audit = json.loads((store.root / 'continuations/study/session_accounting.json').read_text())
    assert audit['rows'][0]['evidence_status'] == 'uncertain'
    assert audit['rows'][0]['session_status_after'] == 'uncertain'
    assert not audit['rows'][0]['raw_file_present']
    monkeypatch.setattr(execution, 'save_image', original)
    result = store.execute('study', Backend, resource_root=store.root)
    assert [c['seed'] for c in Backend.calls] == [23, 41]
    assert result['model_calls'] == 2 and len(result['outputs']) == 1
    assert store.execute('study', forbidden, resource_root=store.root)['model_calls'] == 2


def test_registered_raw_recovery_and_known_failure_reconcile_without_extra_charge(setup, monkeypatch):
    import kusama.execution as execution
    from kusama.resources import SessionBudget
    store, request, config = setup
    store.prepare('study', request, config)
    finish = execution._finish
    monkeypatch.setattr(execution, '_finish', lambda *a, **k: (_ for _ in ()).throw(OSError('After raw commit')))
    first = store.execute('study', Backend, resource_root=store.root)
    assert first['model_calls'] == 1
    with SessionBudget(store.root) as budget:
        assert budget.summary()['calls'][0]['status'] == 'completed'
    monkeypatch.setattr(execution, '_finish', finish)
    class KnownFailure(Backend):
        def generate(self, *args, **kwargs):
            super().generate(*args, **kwargs)
            raise RuntimeError('Known failure after pipeline invocation')
    result = store.execute('study', KnownFailure, resource_root=store.root)
    assert len(result['outputs']) == 1 and result['model_calls'] == 2
    assert [c['seed'] for c in Backend.calls] == [23, 41]
    audit = store.reconcile_session('study')
    assert [r['session_status_after'] for r in audit['rows']] == ['completed', 'failed']
    assert audit['discrepancies'] == []
    assert store.execute('study', forbidden, resource_root=store.root)['model_calls'] == 2


def test_legacy_completed_without_raw_retains_history_and_discloses_discrepancy(setup):
    from kusama.resources import SessionBudget
    store, request, config = setup
    store.prepare('study', request, config)
    class LegacyPrematureCompletion(Backend):
        def generate(self, *args, **kwargs):
            super().generate(*args, **kwargs)
            with SessionBudget(store.root) as budget:
                budget.settle('continuation:study:seed:23', 'completed')
            raise KeyboardInterrupt('Legacy process stopped after premature settlement')
    with pytest.raises(KeyboardInterrupt):
        store.execute('study', LegacyPrematureCompletion, resource_root=store.root)
    audit = store.reconcile_session('study')
    assert audit['rows'][0]['session_status_after'] == 'completed'
    assert audit['rows'][0]['evidence_status'] == 'uncertain'
    assert len(audit['discrepancies']) == 1 and audit['human_quality'] is None
    result = store.execute('study', Backend, resource_root=store.root)
    assert [c['seed'] for c in Backend.calls] == [23, 41] and len(result['outputs']) == 1
    assert store.reconcile_session('study')['discrepancies'][0]['session_status_after'] == 'completed'


def test_dashboard_continuation_and_gallery_reads(setup):
    from kusama.dashboard import create_app
    store, request, config = setup
    store.prepare('study', request, config)
    result = store.execute('study', Backend)
    client = TestClient(create_app(store.root), base_url='http://localhost')
    assert client.get('/api/continuations').json()['continuations'][0]['id'] == 'study'
    assert client.get('/api/continuations/study').json()['model_calls'] == 2
    gallery = client.get('/api/continuations/study/gallery')
    assert gallery.status_code == 200 and '/api/projects/cont_study/artifacts/root' in gallery.text
    assert 'Human follow-up reviews: 0' in gallery.text
    response = client.get('/api/projects/cont_study/artifacts/root')
    import hashlib
    assert hashlib.sha256(response.content).hexdigest() == result['parent']['sha256']
    assert '<body class="gallery-page">' in gallery.text
    assert client.get('/api/sources').json() == {'sources': []}
    assert not (store.root / 'corpus').exists()


def test_durable_dispatch_is_idempotent_and_worker_uses_existing_batch(setup):
    from types import SimpleNamespace
    from kusama.continuation_dispatch import DispatchStore
    from kusama.resources import SessionBudget, load_config
    store, request, config = setup
    store.prepare('study', request, config)
    with SessionBudget(store.root, load_config().model_copy(update={'renderer_attempt_cap': 2})):
        pass
    launches = []
    def launcher(command, **kwargs):
        launches.append((command, kwargs))
        return SimpleNamespace(pid=12345)
    dispatch = DispatchStore(store.root)
    queued = dispatch.submit('study', {'submission_id': 'explicit-execute'}, launcher=launcher)
    assert queued['state'] == 'queued' and Backend.calls == []
    assert launches[0][1]['shell'] is False
    assert launches[0][0][-1] == 'explicit-execute'
    assert dispatch.submit('study', {'submission_id': 'explicit-execute'}, launcher=launcher)['submission_id'] == 'explicit-execute'
    assert len(launches) == 1
    with pytest.raises(ReviewConflict, match='existing submission ID'):
        dispatch.submit('study', {'submission_id': 'another'}, launcher=launcher)
    result = dispatch.work('study', 'explicit-execute', backend_factory=Backend)
    assert result['model_calls'] == 2 and dispatch.get('study')['state'] == 'completed'
    assert dispatch.work('study', 'explicit-execute', backend_factory=forbidden)['model_calls'] == 2
    assert dispatch.submit('study', {'submission_id': 'explicit-execute'}, launcher=launcher)['state'] == 'completed'
    assert len(launches) == 1
    with sqlite3.connect(store.root / 'jobs.sqlite3') as db, pytest.raises(sqlite3.IntegrityError, match='append-only'):
        db.execute('DELETE FROM continuation_dispatch_events')


def test_api_prepare_observe_and_busy_dispatch_do_not_infer(setup, monkeypatch):
    from kusama.dashboard import create_app
    from kusama.resources import SessionBudget, load_config
    from kusama.continuation_dispatch import DispatchStore
    store, request, config = setup
    headers = {'X-Kusama-Review': '1'}
    client = TestClient(create_app(store.root), base_url='http://localhost')
    body = {'id': 'api-study', 'request': request, 'renderer': config.to_dict()}
    assert client.post('/api/continuations', json=body).status_code == 403
    response = client.post('/api/continuations', json=body, headers=headers)
    assert response.status_code == 201 and response.json()['model_calls'] == 0
    assert client.get('/api/contracts').json()['ContinuationCreate']['title'] == 'ContinuationCreate'
    monkeypatch.setattr('kusama.continuation_dispatch.subprocess.Popen', lambda *a, **k: pytest.fail('Busy API must not launch a worker'))
    with SessionBudget(store.root, load_config()) as budget, budget.workload_lock():
        response = client.post('/api/continuations/api-study/execute', json={'submission_id': 'execute'}, headers=headers)
    assert response.status_code == 409 and DispatchStore(store.root).get('api-study') is None
    assert Backend.calls == []
    assert client.post('/api/continuations/missing/execute', json={'submission_id': 'execute'}, headers=headers).status_code == 404
    response = client.post('/api/continuations/api-study/observations', headers=headers,
        json={'artifact_id': 'root', 'note': 'Inspection of a software fixture only', 'author': 'test agent', 'origin': 'development_example'})
    assert response.status_code == 200 and response.json()['origin'] == 'development_example'
    assert client.get('/api/continuations/api-study/status').json()['dispatch'] is None
    assert ReviewStore(store.root).collections() == []
