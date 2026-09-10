"""Question-led UI dispatch; all inference and acquisition use local substitutes."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from kusama import panel_runs, research, research_render
from kusama.dashboard import create_app
from kusama.panel_runs import PanelRuns, StartRun
from kusama.resources import SessionBudget, load_config
from test_research_render import ResearchDirector, SourceBackend, Renderer, hypothesis

HEADERS = {'X-Kusama-Review': '1'}


@pytest.fixture
def studio(tmp_path, monkeypatch):
    from kusama import backend
    monkeypatch.setattr(backend, 'cached_prompt_token_count', lambda prompt, **kw: len(prompt.split()))
    config = load_config().model_copy(update={'device': 'cpu', 'cache_dir': str(tmp_path / 'cache'),
        'render_profiles': {'flux_base4b_study_512_v1': {'width': 64, 'height': 64, 'steps': 1, 'precision': 'fp32'}}})
    monkeypatch.setattr(panel_runs, 'load_config', lambda *args: config)
    launches = []
    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        return SimpleNamespace(pid=os.getpid())
    monkeypatch.setattr(panel_runs.subprocess, 'Popen', launch)
    Renderer.loads = 0; Renderer.calls = []; Renderer.fail_seed = None
    return PanelRuns(tmp_path / 'artifacts'), launches


def request(**changes):
    return StartRun(submission_id='question-1', question='Who gets to feel at home in a place that is disappearing?',
        context='Consider care, ownership and displacement without resolving their tension.',
        research_steps=4, directions=1, allow_network=True, study_development=False, **changes)


def runners(monkeypatch):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    from kusama import research_context
    # UI integration uses substitute models and an empty checkpoint cache.
    # Exact processor admission is covered separately by the token-admission tests.
    monkeypatch.setattr(research_context, 'admit_context',
        lambda context, config, **kw: research_context.fit_context(context, lambda _: 100,
            output_tokens=config.max_new_tokens))
    h = hypothesis('h1', ['s0'])
    actions = [{'action': 'search', 'query': 'boundary'}, {'action': 'open', 'source_id': 's0'},
               {'action': 'draft', 'hypothesis': h}, {'action': 'commit', 'hypothesis': h}]
    def model(director, system, payload, images, validator, **kwargs):
        value = {'reason': 'Fixture research decision', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(research, '_multimodal_json', model)
    return (lambda output, **kw: research.run_research(output, director_factory=ResearchDirector, sources_factory=SourceBackend, **kw),
            lambda output, **kw: research_render.run_render(output, backend_factory=Renderer, **kw))


def test_start_freezes_question_budget_and_deduplicates(studio):
    store, launches = studio
    req = request()
    record = store.start(req)
    directory = store.directory(record['run_id'])
    frozen = json.loads((directory / 'request.json').read_text())
    assert frozen['request'] == req.model_dump(mode='json')
    assert frozen['resource_config']['renderer_attempt_cap'] == 1
    assert frozen['resource_config']['reasoning_token_cap'] == 5 * 16384
    assert frozen['resource_config']['session_hours'] == 2
    assert not (directory / 'resources.sqlite3').exists(), 'Preparation cannot admit or load models'
    assert store.start(req)['run_id'] == record['run_id']
    assert len(launches) == 1
    with pytest.raises(ValueError, match='different'):
        store.start(req.model_copy(update={'question': 'Changed question'}))
    with pytest.raises(ValueError, match='online collection'):
        store.start(req.model_copy(update={'submission_id': 'offline', 'allow_network': False}))
    with pytest.raises(ValueError, match='named render profile'):
        store.start(req.model_copy(update={'submission_id': 'profile', 'render_profile': 'smaller'}))


def test_new_panel_studies_complete_current_grounded_pipeline(studio, monkeypatch):
    from kusama import research_context, study_development
    from test_research_grounded import Sources, analysis, hyp, observations
    from test_study_development import Renderer as StudyRenderer
    from test_study_cycle import cycle_actor
    store, _ = studio
    monkeypatch.setattr(research_context, 'admit_context',
        lambda context, config, **kw: research_context.fit_context(context, lambda _: 100,
            output_tokens=config.max_new_tokens))
    monkeypatch.setattr(study_development, 'admit_study_context',
        lambda *a: {'measurement': 'scripted fixture processor count', 'input_tokens': 100})
    # This scripted source actor covers the study cycle; the chance-exposure
    # integration below exercises real source storage with a fixture transport.
    req = StartRun(submission_id='three-studies', question='What can an enclosure do?', random_exposure_count=0)
    record = store.start(req); directory = store.directory(record['run_id'])
    frozen = json.loads((directory / 'request.json').read_text())
    assert frozen['resource_config']['renderer_attempt_cap'] == 6
    assert frozen['resource_config']['reasoning_token_cap'] == 42 * 16384
    assert frozen['resource_config']['per_call_total_tokens'] == 16384
    assert record['max_calls'] == 48
    actions = [analysis(), {'action': 'search', 'query': 'figure', 'mode': 'remote'},
        {'action': 'assess_candidates', 'candidate_scores': [{'candidate_id': sid, 'score': .8,
            'reason': 'The figure geometry addresses the question'} for sid in ('artic', 'met')]},
        {'action': 'acquire', 'candidate_id': 'artic'},
        {'action': 'inspect_images', 'reinspection': [observations('artic_canonical')]},
        {'action': 'acquire', 'candidate_id': 'met'},
        {'action': 'inspect_images', 'reinspection': [observations('met_canonical')]}]
    for i in range(3):
        h = hyp(); h.update(id=f'h{i}', composition=f'A central figure with {i+1} openings',
            transformation=f'Fixture transformation {i}', artistic_operation=f'Fixture relation {i}',
            study_reason=f'Investigate spatial relation {i}', distinct_from_previous=f'Different relation {i}')
        actions += [{'action': 'draft', 'hypothesis': h}, {'action': 'commit', 'hypothesis': h,
            'reinspection': [observations('artic_canonical'), observations('met_canonical')]}]
    seen = []
    def model(director, system, payload, images, validator, **kwargs):
        seen.append(payload)
        value = {'reason': 'Scripted current-protocol fixture', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value)}
    monkeypatch.setattr(research, '_multimodal_json', model)
    calls = cycle_actor(monkeypatch, [('continue', [0]), ('continue', [3]), ('review', [4])])
    StudyRenderer.calls = []; StudyRenderer.fail_seed = None
    store.execute(record['run_id'],
        research_runner=lambda output, **kw: research.run_research(output, director_factory=ResearchDirector,
            sources_factory=lambda root, *a, **kw: Sources(root), **kw),
        render_runner=lambda output, **kw: research_render.run_render(output, backend_factory=StudyRenderer,
            director_factory=ResearchDirector, **kw))
    result = store.get(record['run_id'])
    assert result['status'] == 'completed', result
    assert result['model_calls'] == 26  # 13 research + 3 studies + 3 inspections + 3 choices + 2 revisions/inspections
    assert len(calls) == 8 and len(StudyRenderer.calls) == 5
    with SessionBudget(directory) as budget:
        reasoning_calls = [call for call in budget.summary()['calls'] if call['kind'] == 'reasoning']
        assert len(reasoning_calls) == 21
        assert all(call['token_reservation'] == 16384 for call in reasoning_calls)
    assert seen[-1]['hypotheses'][0]['artistic_operation'] == 'Fixture relation 0'
    client = TestClient(create_app(store.root), base_url='http://localhost')
    run = next(row for row in client.get('/api/runs').json()['runs'] if row['name'] == record['run_id'])
    inspection = client.get(f"/api/runs/{run['id']}/inspect").json()
    assert len(inspection['images']) == 5, [im['name'] for im in inspection['images']]
    development = inspection['study_development'][0]
    assert development['status'] == 'complete' and development['choice']['action'] == 'review'
    assert development['cycles'][1]['observation']['explanation_gaps']
    assert development['human_review_requested']
    assert development['branches'][-1]['parent_image_id'] == development['branches'][-2]['id']
    assert not list(directory.rglob('reviews.sqlite3'))
    workflow = inspection['development_cycle']
    assert workflow['policy']['max_steps'] == 3
    assert workflow['initial_image_count'] == 3
    assert workflow['development']['human_review_requested']
    assert workflow['development_directory'] == 'render/study_development'
    assert client.get('/static/dashboard_cycle.js').status_code == 200


def test_panel_chance_leads_reach_actual_saved_prompt(studio, monkeypatch):
    from kusama import research_context
    from kusama.research_sources import ResearchSources
    from test_research_grounded import analysis
    from test_research_source_audit import record as museum_record, envelope
    from test_research_sources import Response
    store, _ = studio
    record = store.start(StartRun(submission_id='chance', question='What does a boundary make possible?'))
    directory = store.directory(record['run_id'])
    monkeypatch.setattr(research_context, 'admit_context',
        lambda context, config, **kw: research_context.fit_context(context, lambda _: 100))
    seen = []
    def model(director, system, payload, images, validator, **kwargs):
        seen.append(payload)
        value = analysis() if len(seen) == 1 else {'action': 'stop', 'reason': 'Fixture stops without choosing a source.'}
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(research, '_multimodal_json', model)
    requests = []
    def factory(root, config, **kwargs):
        def fetch(request, timeout):
            requests.append(request.full_url)
            provider = 'artic' if 'artic.edu' in request.full_url else 'met' if 'metmuseum.org' in request.full_url else 'cleveland'
            if provider == 'met':
                body = ({'total': 8, 'objectIDs': list(range(1, 9))} if '/search?' in request.full_url else
                        museum_record(provider, int(request.full_url.rsplit('/', 1)[-1])))
            else:
                body = envelope(provider, [museum_record(provider, i) for i in range(1, 9)])
            return Response(json.dumps(body).encode())
        return ResearchSources(root, config, opener=fetch, **kwargs)
    runner = lambda output, **kw: research.run_research(output, director_factory=ResearchDirector, sources_factory=factory, **kw)
    store.execute(record['run_id'], research_runner=runner)
    report = research.read_report(directory / 'research')
    assert report['status'] == 'model_stop' and report['reasoning_calls'] == 2
    assert report['network']['acquisitions'] == 0
    assert report['human_labels'] == 0
    assert 1 <= len(seen[1]['chance_candidates']) <= 2
    assert not seen[0]['chance_candidates']
    assert all(s['exposure'] == 'chance' for s in seen[1]['chance_candidates'])
    assert report['chance_exposure']['policy']['count'] == 2
    saved = (directory / 'research/decision-001.json').read_bytes()
    assert json.loads(saved)['payload']['chance_candidates'] == seen[1]['chance_candidates']
    calls = len(requests)
    assert runner(directory / 'research')['status'] == 'model_stop'
    assert len(requests) == calls
    assert (directory / 'research/decision-001.json').read_bytes() == saved


def test_queued_ui_uses_its_frozen_cycle_limits_and_preserves_evidence(studio):
    store, _ = studio
    record = store.start(StartRun(submission_id='queued-cycle', question='What can an enclosure do?',
        development_steps=5, development_research_steps=12))
    directory = store.directory(record['run_id'])
    before = (directory / 'request.json').read_bytes()
    client = TestClient(create_app(store.root), base_url='http://localhost')
    run = next(r for r in client.get('/api/runs').json()['runs'] if r['name'] == record['run_id'])
    view = client.get(f"/api/runs/{run['id']}/inspect").json()['development_cycle']
    assert view['policy']['max_steps'] == 5 and view['policy']['research_steps'] == 12
    assert view['development'] == {} and view['initial_image_ids'] == []
    assert (directory / 'request.json').read_bytes() == before
    (directory / 'request.json').write_bytes(before + b' ')
    result = client.get(f"/api/runs/{run['id']}/inspect").json()
    assert result['development_cycle'] is None and result['errors']


def test_stale_form_cannot_authorize_larger_undisplayed_cycle_budget(studio):
    store, launches = studio
    client = TestClient(create_app(store.root), base_url='http://localhost')
    body = StartRun(submission_id='new-budget', question='What can an enclosure do?').model_dump(mode='json')
    for missing in ('development_steps', 'development_research_steps', 'total_token_limit'):
        stale = {key: value for key, value in body.items() if key != missing}
        response = client.post('/api/start-run', json=stale, headers=HEADERS)
        assert response.status_code == 422 and 'Refresh the panel' in response.json()['detail']
    assert launches == [] and not (store.root / 'runs').exists()
    assert client.post('/api/start-run', json=body, headers=HEADERS).status_code == 202
    assert len(launches) == 1


def test_ui_pipeline_research_to_images_and_complete_inspection(studio, monkeypatch):
    store, launches = studio
    app = create_app(store.root)
    client = TestClient(app, base_url='http://localhost')
    body = request(images_per_direction=2).model_dump(mode='json')
    assert client.post('/api/start-run', json=body).status_code == 403
    assert client.post('/api/start-run', json=body, headers=HEADERS | {'Origin': 'http://evil.example'}).status_code == 403
    response = client.post('/api/start-run', json=body, headers=HEADERS)
    assert response.status_code == 202, response.text
    value = response.json(); run_id = value['run_id']; directory = store.directory(run_id)
    research_runner, render_runner = runners(monkeypatch)
    store.execute(run_id, research_runner=research_runner, render_runner=render_runner)
    assert store.get(run_id)['status'] == 'completed'
    assert store.get(run_id)['model_calls'] == 6
    assert json.loads((directory / 'manifest.json').read_text())['model_calls'] == 6
    assert len(Renderer.calls) == 2
    assert not (directory / 'render' / 'reviews.sqlite3').exists()
    assert not list(directory.rglob('reviews.sqlite3')), 'Question-led run does not create pairwise review infrastructure'
    with SessionBudget(directory) as budget:
        assert budget.summary()['renderer_attempts'] == 2
        assert len(budget.db.execute('SELECT * FROM calls').fetchall()) == 6
    inspection = client.get(f"/api/runs/{value['id']}/inspect")
    assert inspection.status_code == 200, inspection.text
    data = inspection.json()
    assert len(data['images']) == 2
    assert data['research'][0]['problem'] == body['question']
    assert len(data['research'][0]['steps']) == 4
    transcripts = data['research'][0]['transcripts']
    assert len(transcripts) == 4
    for transcript in transcripts:
        index = transcript['index']
        assert transcript['prompt'] == json.loads((directory / 'research' / f'decision-{index:03}.json').read_text())
        assert transcript['response'] == json.loads((directory / 'research' / f'response-{index:03}.json').read_text())
    assert not data['generations_without_images']
    assert not data['errors']
    assert data['research'][0]['evidence_images']
    image = client.get(data['research'][0]['evidence_images'][0]['url'])
    assert image.status_code == 200, image.text
    assert [item['at'] for item in data['activity']] == sorted(item['at'] for item in data['activity'])
    for entry in data['images']:
        assert client.get(entry['url']).headers['content-type'] == 'image/png'
        assert entry['record']['saved_action']['prompt'] == hypothesis('h1', ['s0'])['composition'] + '\n' + hypothesis('h1', ['s0'])['render_instruction']
    raw = next(file for file in data['files'] if file['name'].endswith('request.json'))
    assert client.get(raw['url']).status_code == 200
    assert client.get(f"/api/runs/{value['id']}").json()['model_calls'] == 6
    with pytest.raises(ValueError, match='already started'):
        store.execute(run_id, research_runner=research_runner, render_runner=render_runner)
    assert len(Renderer.calls) == 2
    with closing(sqlite3.connect(store.database)) as db:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('DELETE FROM panel_events')


def test_stop_queued_run_from_all_artifacts_and_preserve_other_runs(studio):
    store, _ = studio
    record = store.start(request())
    client = TestClient(create_app(store.root.parent), base_url='http://localhost')
    listing = client.get('/api/runs').json()['runs']
    entry = next(row for row in listing if row['name'] == record['run_id'])
    endpoint = f"/api/runs/{entry['id']}/stop"
    assert client.post(endpoint, json={}).status_code == 403
    assert client.post(endpoint, json={}, headers=HEADERS).status_code == 200
    assert store.get(record['run_id'])['stop_requested'] is True
    store.execute(record['run_id'])
    assert store.get(record['run_id'])['status'] == 'stopped'
    assert not (store.directory(record['run_id']) / 'resources.sqlite3').exists()
    assert Renderer.loads == 0


def test_stop_between_images_retains_first_and_never_dispatches_second(studio, monkeypatch):
    store, _ = studio
    record = store.start(request(images_per_direction=2))
    directory = store.directory(record['run_id'])
    research_runner, _ = runners(monkeypatch)
    class StopRenderer(Renderer):
        def generate(self, *args, **kwargs):
            image = super().generate(*args, **kwargs)
            store.stop(record['run_id'])
            return image
    def render_runner(output, **kw):
        return research_render.run_render(output, backend_factory=StopRenderer, **kw)
    store.execute(record['run_id'], research_runner=research_runner, render_runner=render_runner)
    assert store.get(record['run_id'])['status'] == 'stopped'
    assert len(Renderer.calls) == 1
    assert len(list((directory / 'render').glob('*.png'))) == 1
    with SessionBudget(directory) as budget:
        assert budget.summary()['renderer_attempts'] == 1


def test_changed_frozen_request_fails_before_admission(studio):
    store, _ = studio
    record = store.start(request())
    directory = store.directory(record['run_id'])
    frozen = json.loads((directory / 'request.json').read_text())
    frozen['request']['question'] = 'Replacement'
    (directory / 'request.json').write_text(json.dumps(frozen))
    with pytest.raises(Exception, match='Frozen inputs'):
        store.execute(record['run_id'])
    assert store.get(record['run_id'])['status'] == 'failed'
    assert not (directory / 'resources.sqlite3').exists()


def test_recursive_file_inspection_is_read_only_and_excludes_symlinks(tmp_path):
    root = tmp_path / 'artifacts'; run = root / 'run'; nested = run / 'saved'
    nested.mkdir(parents=True)
    (run / 'manifest.json').write_text('{"status":"completed"}')
    (nested / 'prompt.txt').write_text('Original instruction')
    secret = tmp_path / 'private.txt'; secret.write_text('Outside workspace')
    (nested / 'outside.txt').symlink_to(secret)
    client = TestClient(create_app(root), base_url='http://localhost')
    run_id = client.get('/api/runs').json()['runs'][0]['id']
    before = sorted(str(path) for path in root.rglob('*'))
    data = client.get(f'/api/runs/{run_id}/inspect').json()
    prompt = next(file for file in data['files'] if file['name'] == 'saved/prompt.txt')
    from datetime import datetime
    assert datetime.fromisoformat(prompt['modified_at']).timestamp() == pytest.approx((nested / 'prompt.txt').stat().st_mtime)
    assert client.get('/static/dashboard_feed.js').status_code == 200
    assert 'saved/prompt.txt' in [file['name'] for file in data['files']]
    assert 'saved/outside.txt' not in [file['name'] for file in data['files']]
    assert client.get(f'/api/runs/{run_id}/files/saved/outside.txt').status_code == 404
    assert before == sorted(str(path) for path in root.rglob('*'))


def test_inspection_shows_frozen_prompts_even_without_rendered_images(tmp_path):
    root = tmp_path / 'artifacts'
    directory = root / 'run'
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text('{"status":"partial"}')
    actions = [{'id': f'image-{i:03}', 'prompt': f'Saved prompt {i}', 'seed': i} for i in range(2)]
    (directory / 'render_plan.json').write_text(json.dumps({'actions': actions}))
    (directory / 'failure-000.json').write_text(json.dumps({'action': actions[0], 'error': 'Fixture renderer failure'}))
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with TestClient(create_app(root), base_url='http://localhost') as app:
        run_id = app.get('/api/runs').json()['runs'][0]['id']
        result = app.get(f'/api/runs/{run_id}/inspect').json()
    assert result['images'] == []
    assert [row['action'] for row in result['generations_without_images']] == actions
    assert result['generations_without_images'][0]['error'] == 'Fixture renderer failure'
    assert result['generations_without_images'][1]['error'] is None
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_online_is_default_and_local_selection_is_removed(studio):
    store, _ = studio
    client = TestClient(create_app(store.root), base_url='http://localhost')
    assert 'indexes' not in client.get('/api/start-options').json()
    html = client.get('/').text
    assert 'new-run-index' not in html and 'new-run-network' not in html
    body = request().model_dump(mode='json'); body.pop('allow_network')
    response = client.post('/api/start-run', json=body, headers=HEADERS)
    assert response.status_code == 202
    saved = json.loads((store.directory(response.json()['run_id']) / 'request.json').read_text())
    assert saved['request']['allow_network'] is True
    assert 'index_id' not in saved['request'] and 'reference_index' not in saved
    for invalid in ({'index_id': 'old-library'}, {'allow_network': False}):
        assert client.post('/api/start-run', json=body | invalid, headers=HEADERS).status_code == 422


@pytest.mark.parametrize('status', ['step_limit', 'stagnated', 'model_stop'])
def test_empty_terminal_research_is_partial_and_never_renders(studio, status):
    store, _ = studio
    record = store.start(request())
    def no_images(*args, **kwargs):
        pytest.fail('No committed hypothesis may dispatch rendering')
    store.execute(record['run_id'], research_runner=lambda *a, **kw: {'status': status, 'hypotheses': []},
                  render_runner=no_images)
    result = store.get(record['run_id'])
    assert result['status'] == 'partial' and status in result['progress_message']
    assert result['model_calls'] == 0


@pytest.mark.parametrize('provider', ['met', 'artic', 'cleveland'])
def test_online_provider_search_acquire_commit_render_end_to_end(studio, monkeypatch, provider):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    from kusama import research_context
    from kusama.research_sources import ResearchSources
    from test_research_source_audit import Transport
    store, _ = studio
    record = store.start(request(images_per_direction=2))
    directory = store.directory(record['run_id'])
    transport = Transport(provider)
    calls = []
    monkeypatch.setattr(research_context, 'admit_context',
        lambda context, config, **kw: research_context.fit_context(context, lambda _: 100,
            output_tokens=config.max_new_tokens))
    def model(director, system, payload, images, validator, **kwargs):
        if len(calls) == 0:
            value = {'action': 'search', 'mode': 'remote', 'provider': provider, 'query': 'portrait', 'limit': 1}
        elif len(calls) == 1:
            value = {'action': 'acquire', 'candidate_id': payload['last_tool_result']['sources'][0]['candidate_id']}
        else:
            assert images, 'Draft and commitment must see the acquired pixels'
            sid = payload['source_context'][0]['source_id']
            value = {'action': 'draft' if len(calls) == 2 else 'commit', 'hypothesis': hypothesis('h1', [sid])}
        calls.append(value)
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(research, '_multimodal_json', model)
    def sources(root, config, index_path=None):
        assert config['allow_network'] is True and index_path is None
        return ResearchSources(root, config, opener=transport)
    store.execute(record['run_id'],
        research_runner=lambda output, **kw: research.run_research(output, director_factory=ResearchDirector,
            sources_factory=sources, **kw),
        render_runner=lambda output, **kw: research_render.run_render(output, backend_factory=Renderer, **kw))
    result = store.get(record['run_id'])
    assert result['status'] == 'completed' and result['model_calls'] == 6
    assert len(Renderer.calls) == 2 and len(calls) == 4
    report = research.read_report(directory / 'research')
    assert report['network']['acquisitions'] == 1 and len(report['hypotheses']) == 1
    for step in report['steps'][2:]:
        assert step['render_prompt_admission']['input_tokens'] < 512
