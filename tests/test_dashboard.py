import json

from fastapi.testclient import TestClient
from PIL import Image

from kusama.dashboard import create_app
from kusama.ledger import Ledger


def saved_run(root, name='edit'):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / 'manifest.json').write_text(json.dumps({'status': 'needs_review', 'model_calls': 1,
        'runtime_seconds': 3.5, 'attempts': [{'verification': {'status': 'pass'}}]}))
    (directory / 'plan.json').write_text('{"max_calls": 2}')
    Image.new('RGB', (32, 32), 'red').save(directory / 'parent.png')
    return directory


def client(root):
    return TestClient(create_app(root), base_url='http://127.0.0.1')


def test_lists_runs_and_reads_outputs_without_changing_files(tmp_path):
    directory = saved_run(tmp_path)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with client(tmp_path) as app:
        row = app.get('/api/runs').json()['runs'][0]
        assert (row['name'], row['model_calls'], row['max_calls'], row['verification']) == ('edit', 1, 2, 'pass')
        detail = app.get(f"/api/runs/{row['id']}").json()
        image = next(a for a in detail['artifacts'] if a['name'] == 'parent.png')
        assert app.get(image['url']).content == before['parent.png']
        assert app.post('/api/runs').status_code == 405
        assert app.post('/api/sessions').status_code == 404
        assert app.post('/api/projects/project').status_code == 405
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    assert not (tmp_path / 'jobs.sqlite3').exists()


def test_broken_and_legacy_runs_remain_visible(tmp_path):
    saved_run(tmp_path)
    broken = tmp_path / 'broken'; broken.mkdir()
    (broken / 'manifest.json').write_text('{unfinished')
    legacy = tmp_path / 'legacy'; legacy.mkdir()
    (legacy / 'manifest.json').write_text(json.dumps({'status': 'failed', 'calls': [],
        'budget': {'actual_image_attempts': 2, 'maximum_image_attempts': 3},
        'failures': [{'message': 'out of memory'}]}))
    benchmark = tmp_path / 'suite'; benchmark.mkdir()
    (benchmark / 'benchmark.json').write_text(json.dumps({'status': 'completed', 'model_calls': 7, 'max_calls': 7,
        'summary': {'verification_counts': {'pass': 5, 'fail': 2}}}))
    with client(tmp_path) as app:
        rows = {row['name']: row for row in app.get('/api/runs').json()['runs']}
        assert len(rows) == 4
        assert rows['broken']['status'] == 'unreadable' and rows['broken']['error']
        assert rows['legacy']['model_calls'] == 2 and rows['legacy']['max_calls'] == 3
        assert rows['legacy']['kind'] == 'first_edit'
        assert rows['suite']['verification'] == {'pass': 5, 'fail': 2}


def test_ledger_status_and_budget_override_stale_manifest(tmp_path):
    directory = saved_run(tmp_path)
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('job', {}, 4, directory)
        ledger.reserve_call('job', 0)
        ledger.reserve_call('job', 1)
        ledger.update_job('job', status='uncertain')
        original_events = ledger.events()
    with client(tmp_path) as app:
        row = app.get('/api/runs').json()['runs'][0]
        assert (row['status'], row['model_calls'], row['max_calls']) == ('uncertain', 2, 4)
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        assert ledger.events() == original_events


def test_file_access_stays_inside_run_and_excludes_private_files(tmp_path):
    directory = saved_run(tmp_path / 'artifacts')
    secret = tmp_path / 'secret.json'; secret.write_text('{"secret":true}')
    (directory / 'linked.json').symlink_to(secret)
    (directory / '.env').write_text('TOKEN=secret')
    with client(tmp_path / 'artifacts') as app:
        run_id = app.get('/api/runs').json()['runs'][0]['id']
        files = app.get(f'/api/runs/{run_id}').json()['artifacts']
        assert not {'linked.json', '.env'} & {f['name'] for f in files}
        for name in ('linked.json', '.env', 'unknown.json', '..%2Fsecret.json'):
            assert app.get(f'/api/runs/{run_id}/files/{name}').status_code == 404
        assert app.get('/api/runs/unknown').status_code == 404


def test_loopback_requests_need_no_login_but_cross_origin_is_blocked(tmp_path):
    with client(tmp_path) as app:
        assert app.get('/api/runs').json() == {'runs': []}
        assert app.get('/api/runs', headers={'Origin': 'https://example.com'}).status_code == 403
        assert app.get('/api/runs', headers={'Host': 'example.com'}).status_code == 400
        assert app.get('/api/runs', headers={'Sec-Fetch-Site': 'cross-site'}).status_code == 403


def test_source_and_region_routes_verify_registered_bytes_and_rights(tmp_path):
    import sqlite3
    from kusama.artifacts import sha256
    root = tmp_path / 'corpus'
    root.mkdir()
    path = root / 'source.jpg'
    Image.new('RGB', (64, 64), 'green').save(path)
    asset = {'path': path.name, 'sha256': sha256(path)}
    rights = {'image_display': {'status': 'allowed'}, 'derivatives': {'status': 'allowed'}}
    reference = {'id': 'source', 'title': 'A source', 'view': asset, 'rights': rights}
    region = {'id': 'region', 'reference_id': 'source', 'crop': asset, 'rights': rights}
    with sqlite3.connect(root / 'corpus.sqlite3') as db:
        for table in ('references_v1', 'regions_v1', 'claims_v1'):
            db.execute(f'CREATE TABLE {table}(id TEXT PRIMARY KEY,payload TEXT)')
        db.execute('INSERT INTO references_v1 VALUES (?,?)', ('source', json.dumps(reference)))
        db.execute('INSERT INTO regions_v1 VALUES (?,?)', ('region', json.dumps(region)))
    with client(tmp_path) as app:
        assert app.get('/api/sources').json()['sources'][0]['id'] == 'source'
        assert app.get('/api/sources/source').json()['regions'][0]['id'] == 'region'
        assert app.get('/api/sources/source/image').content == path.read_bytes()
        assert app.get('/api/source-regions/region/image').status_code == 200
        path.write_bytes(b'changed')
        assert app.get('/api/source-regions/region/image').status_code == 409
        assert app.get('/api/sources/missing').status_code == 404
    with sqlite3.connect(root / 'corpus.sqlite3') as db:
        reference['rights']['image_display']['status'] = 'excluded'
        db.execute('UPDATE references_v1 SET payload=? WHERE id=?', (json.dumps(reference), 'source'))
    with client(tmp_path) as app:
        assert app.get('/api/sources/source/image').status_code == 403


def test_inspiration_images_are_registered_and_hash_checked(tmp_path):
    from kusama.artifacts import sha256
    directory = tmp_path / 'runs' / 'episode'
    directory.mkdir(parents=True)
    image = directory / 'image.png'
    Image.new('RGB', (64, 64), 'red').save(image)
    (directory / 'manifest.json').write_text(json.dumps({'kind': 'connection_inspiration', 'run_id': 'episode',
        'status': 'needs_review', 'images': [{'id': 'one', 'path': image.name, 'sha256': sha256(image)}]}))
    with client(tmp_path) as app:
        assert app.get('/api/inspirations').json()['inspirations'][0]['directory_id'] == 'episode'
        assert app.get('/api/inspirations/episode/images/one').content == image.read_bytes()
        assert app.get('/api/inspirations/episode/images/two').status_code == 404
        image.write_bytes(b'changed')
        assert app.get('/api/inspirations/episode/images/one').status_code == 409


def test_review_api_requires_local_json_and_preserves_independent_answers(tmp_path):
    from kusama.reviews import ReviewStore
    directory = saved_run(tmp_path)
    Image.new('RGB', (32, 32), 'blue').save(directory / 'result.png')
    store = ReviewStore(tmp_path)
    store.register_collection('study', title='Study', brief='Explore scale.', candidates=[
        {'id': 'parent', 'path': directory / 'parent.png'}, {'id': 'result', 'path': directory / 'result.png'}])
    headers = {'X-Kusama-Review': '1'}
    with client(tmp_path) as app:
        assert app.get('/').status_code == 200
        assert 'Which is better finished work?' in app.get('/').text
        assert 'Which idea deserves further development?' in app.get('/').text
        assert app.get('/api/reviews/collections').json()['collections'][0]['id'] == 'study'
        comparison = app.get('/api/reviews/collections/study/pairs/pair-001?reviewer=Jayce').json()
        body = {'reviewer': 'Jayce', 'assignment_id': comparison['assignment_id'], 'submission_id': 'first',
                'finished_quality': {'choice': 'tie'}, 'development_potential': {'choice': 'unsure'}}
        url = '/api/reviews/collections/study/pairs/pair-001/votes'
        assert app.post(url, json=body).status_code == 403
        assert app.post(url, json=body, headers=headers | {'Origin': 'https://example.org'}).status_code == 403
        assert app.post(url, json=body, headers=headers | {'Sec-Fetch-Site': 'cross-site'}).status_code == 403
        response = app.post(url, json=body, headers=headers)
        assert response.status_code == 200
        assert response.json()['finished_quality']['choice'] == 'tie'
        assert response.json()['development_potential']['choice'] == 'unsure'
        assert app.post(url, json=body, headers=headers).json() == response.json()
        assert len(store.history('study')['votes']) == 1
        assert not store.history('study')['advancements']
        choice = app.post('/api/reviews/collections/study/advance', headers=headers, json={
            'reviewer': 'Jayce', 'candidate_id': 'result', 'reason': 'Follow this idea.', 'submission_id': 'choice1'})
        assert choice.status_code == 200
        assert choice.json()['candidate_id'] == 'result'
        assert app.get('/api/reviews/collections/study/candidates/result').content == (directory / 'result.png').read_bytes()
        Image.new('RGB', (32, 32), 'green').save(directory / 'result.png')
        assert app.get('/api/reviews/collections/study/candidates/result').status_code == 409
        assert app.get('/api/reviews/collections/study/candidates/unknown').status_code == 404
        assert app.get('/api/reviews/collections/unknown').status_code == 404
    assert not (tmp_path / 'jobs.sqlite3').exists()


def test_no_review_database_created_by_reading_empty_dashboard(tmp_path):
    with client(tmp_path) as app:
        assert app.get('/api/reviews/collections').json() == {'collections': []}
    assert not (tmp_path / 'reviews.sqlite3').exists()


def test_run_cost_summary_keeps_unknown_total_and_reads_only_contained_root(tmp_path):
    from kusama.costs import CostJournal
    directory = saved_run(tmp_path)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['cost_accounting'] = {'root': str(tmp_path), 'run_id': 'cost-run'}
    manifest_path.write_text(json.dumps(manifest))
    with CostJournal(tmp_path) as costs:
        costs.start('cost-run', 'inference', 'FLUX')
        costs.finish('cost-run', 'inference', 2)
    with client(tmp_path) as app:
        run_id = app.get('/api/runs').json()['runs'][0]['id']
        cost = app.get(f'/api/runs/{run_id}').json()['monetary_cost']
        assert cost['currencies']['USD']['total'] is None
        assert cost['currencies']['USD']['known_subtotal'] == 0
        manifest['cost_accounting']['root'] = str(tmp_path.parent)
        manifest_path.write_text(json.dumps(manifest))
        assert app.get(f'/api/runs/{run_id}').json()['monetary_cost'] == {'coverage': 'no_cost_records'}


def test_studio_selection_requires_deliberate_local_request(tmp_path, monkeypatch):
    selections = []
    class FakeStudio:
        def __init__(self, root):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def select(self, project_id, episode_id, artifact_id, reason, author):
            selections.append((project_id, episode_id, artifact_id, reason, author))
            return {'status': 'awaiting_direction', 'current_artifact_id': artifact_id}
    monkeypatch.setattr('kusama.dashboard.Studio', FakeStudio)
    body = {'episode_id': 'episode', 'artifact_id': 'candidate', 'author': 'Jayce', 'reason': 'Develop this idea.'}
    with client(tmp_path) as app:
        assert app.post('/api/projects/project/select', json=body).status_code == 403
        assert not selections
        response = app.post('/api/projects/project/select', json=body, headers={'X-Kusama-Review': '1'})
        assert response.status_code == 200
        assert selections == [('project', 'episode', 'candidate', 'Develop this idea.', 'Jayce')]


def test_creative_run_shows_title_brief_scope_time_and_partial_image_gallery(tmp_path):
    from kusama.artifacts import sha256
    from kusama.costs import CostJournal
    directory = tmp_path / 'creative-run'
    directory.mkdir()
    manifest_path = directory / 'manifest.json'
    manifest = {'kind': 'reference_creative_experiment', 'status': 'rendering',
                'title': 'A study of interrupted memory', 'brief': 'Explore the feeling of a severed history.',
                'direction_count': 12, 'realizations': 2, 'planned_calls': 40, 'model_calls': 4,
                'images': [], 'costs': {'runtime_seconds_measured': 8},
                'cost_accounting': {'root': str(tmp_path), 'run_id': 'creative-run'}}
    manifest_path.write_text(json.dumps(manifest))
    with CostJournal(tmp_path) as costs:
        costs.start('creative-run', 'qwen-load', 'Qwen', scope='model_loading')
        costs.finish('creative-run', 'qwen-load', 2)
        costs.start('creative-run', 'direction-1', 'Qwen')
        costs.finish('creative-run', 'direction-1', 7)
        costs.start('creative-run', 'generation-1', 'FLUX')
    with client(tmp_path) as app:
        row = app.get('/api/runs').json()['runs'][0]
        assert row['kind'] == 'reference_creative_experiment'
        assert row['title'] == manifest['title'] and row['brief'] == manifest['brief']
        assert (row['image_count'], row['planned_images'], row['max_calls']) == (0, 24, 40)
        assert row['runtime_scope'] == 'recorded_operations' and row['runtime_seconds'] == 8
        detail = app.get(f"/api/runs/{row['id']}").json()
        assert detail['runtime_seconds'] == 9  # Read current accounting rather than stale snapshot.
        assert detail['monetary_cost']['open_calls'] == 1
        assert not [f for f in detail['artifacts'] if f['preview']]
        image_path = directory / 'direction-1-r1.png'
        Image.new('RGB', (32, 40), 'orange').save(image_path)
        manifest['images'] = [{'id': 'direction-1-r1', 'path': image_path.name,
                               'label': 'Interrupted memory / realization 1', 'sha256': sha256(image_path)}]
        manifest_path.write_text(json.dumps(manifest))
        detail = app.get(f"/api/runs/{row['id']}").json()
        previews = [f for f in detail['artifacts'] if f['preview']]
        assert detail['image_count'] == 1 and len(previews) == 1
        assert previews[0]['label'] == 'Interrupted memory / realization 1'
        assert app.get(previews[0]['url']).content == image_path.read_bytes()
        Image.new('RGB', (32, 40), 'purple').save(image_path)
        assert app.get(previews[0]['url']).status_code == 409


def test_creative_recovery_aggregates_only_explicit_cost_runs_and_total_call_budget(tmp_path):
    from kusama.costs import CostJournal
    directory = saved_run(tmp_path, 'creative-recovery')
    manifest = {'kind': 'reference_creative_experiment', 'status': 'needs_review',
                'title': 'Study with observer recovery', 'brief': 'An artistic inquiry.',
                'model_calls': 37, 'planned_calls': 48, 'total_model_calls': 49,
                'authorized_maximum_model_calls': 60, 'images': [],
                'observer_recovery': {'job_id': 'recovery', 'directory': 'observer-recovery', 'max_calls': 12},
                'cost_accounting': {'root': str(tmp_path), 'run_id': 'primary', 'run_ids': ['primary', 'recovery']}}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('primary', {}, 48, directory)
        for index in range(37):
            ledger.reserve_call('primary', index)
        ledger.update_job('primary', status='needs_review', payload={'manifest': manifest})
    policy = {'currency': 'USD', 'average_power_watts': 100, 'electricity_per_kwh': .20,
              'hardware_per_hour': 1, 'source': 'Synthetic dashboard test fixture.'}
    with CostJournal(tmp_path) as costs:
        for run_id, seconds in [('primary', 3600), ('recovery', 1800), ('unrelated', 7200)]:
            costs.start(run_id, 'inference', 'Qwen', policy=policy)
            costs.finish(run_id, 'inference', seconds)
    with client(tmp_path) as app:
        summary = app.get('/api/runs').json()['runs'][0]
        assert (summary['model_calls'], summary['max_calls']) == (49, 60)
        detail = app.get(f"/api/runs/{summary['id']}").json()
        assert (detail['model_calls'], detail['max_calls']) == (49, 60)
        assert detail['runtime_seconds'] == 5400
        assert detail['monetary_cost']['calls'] == 2
        assert {record['run_id'] for record in detail['monetary_cost']['records']} == {'primary', 'recovery'}
        assert abs(detail['monetary_cost']['currencies']['USD']['total'] - 1.53) < 1e-9
        assert detail['data']['planned_calls'] == 48 and detail['data']['model_calls'] == 37


def test_invalid_aggregation_ids_use_legacy_scope_and_never_all_journal_records(tmp_path):
    from kusama.costs import CostJournal
    directory = saved_run(tmp_path)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    with CostJournal(tmp_path) as costs:
        for run_id in ('primary', 'other'):
            costs.start(run_id, 'inference', 'FLUX')
            costs.finish(run_id, 'inference', 3)
    with client(tmp_path) as app:
        run_id = app.get('/api/runs').json()['runs'][0]['id']
        for invalid in ([], 'primary', ['primary', 'primary'], ['primary', ''], ['primary', 3], [None], [{}]):
            manifest['cost_accounting'] = {'root': str(tmp_path), 'run_id': 'primary', 'run_ids': invalid}
            manifest_path.write_text(json.dumps(manifest))
            detail = app.get(f'/api/runs/{run_id}').json()
            assert detail['monetary_cost']['calls'] == 1
            assert detail['monetary_cost']['records'][0]['run_id'] == 'primary'
        manifest['cost_accounting'] = {'root': str(tmp_path), 'run_ids': []}
        manifest_path.write_text(json.dumps(manifest))
        assert app.get(f'/api/runs/{run_id}').json()['monetary_cost'] == {'coverage': 'no_cost_records'}
        manifest['cost_accounting'] = {'root': str(tmp_path), 'run_ids': ['primary', 'other']}
        manifest_path.write_text(json.dumps(manifest))
        assert app.get(f'/api/runs/{run_id}').json()['monetary_cost']['calls'] == 2


def test_partial_observer_status_is_explicit_when_images_are_ready_for_human_review(tmp_path):
    directory = saved_run(tmp_path, 'partial-observer')
    status = {'status': 'partial', 'attempted_pairs': 12, 'valid_pairs': 8,
              'rejected_pairs': 4, 'unresolved_direction_ids': ['d02', 'd04', 'd07', 'd12'],
              'rejected_response_count': 5}
    manifest = {'kind': 'reference_creative_experiment', 'status': 'needs_human_review',
                'direction_count': 12, 'realizations': 2, 'images': [], 'observer_status': status,
                'observations': [{'direction_id': f'd{i}'} for i in range(8)],
                'observer_rejections': [{'response': i} for i in range(5)]}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    with client(tmp_path) as app:
        summary = app.get('/api/runs').json()['runs'][0]
        assert summary['status'] == 'needs_human_review'
        assert summary['observer_status'] == status
        assert summary['planned_observations'] == 12
        detail = app.get(f"/api/runs/{summary['id']}").json()
        assert detail['observer_status']['valid_pairs'] == 8
        assert detail['observer_status']['rejected_pairs'] == 4
        assert detail['observer_status']['rejected_response_count'] == 5
        assert len(detail['data']['observations']) == 8
        assert app.get('/api/reviews/collections').json() == {'collections': []}
    assert not (tmp_path / 'reviews.sqlite3').exists()


def test_creative_saved_technical_verification_matches_its_run_and_scope(tmp_path):
    directory = saved_run(tmp_path, 'verified-creative')
    (directory / 'manifest.json').write_text(json.dumps({'kind': 'reference_creative_experiment',
        'run_id': 'verified-creative', 'status': 'needs_human_review', 'images': []}))
    report_path = directory / 'verification.json'
    report = {'kind': 'technical_integrity_verification', 'run_id': 'verified-creative',
              'status': 'passed', 'verified_at': '2026-09-08T04:17:27Z',
              'scope': 'Recorded technical integrity only; no artistic-quality verdict.'}
    report_path.write_text(json.dumps(report))
    with client(tmp_path) as app:
        row = app.get('/api/runs').json()['runs'][0]
        assert row['verification'] == {'status': 'passed', 'scope': 'technical_integrity',
            'verified_at': '2026-09-08T04:17:27Z', 'report_file': 'verification.json'}
        detail = app.get(f"/api/runs/{row['id']}").json()
        assert detail['verification'] == row['verification']
        link = next(f for f in detail['artifacts'] if f['name'] == 'verification.json')
        assert app.get(link['url']).json() == report
        for overrides in ({'run_id': 'another-run'}, {'kind': 'artistic_quality_verification'}, {'status': 'unrecognized'}):
            report_path.write_text(json.dumps(report | overrides))
            assert app.get('/api/runs').json()['runs'][0]['verification'] is None
        report_path.write_text('{unfinished')
        assert app.get('/api/runs').json()['runs'][0]['verification'] is None
        report_path.unlink()
        external = tmp_path / 'external-report.json'
        external.write_text(json.dumps(report))
        report_path.symlink_to(external)
        assert app.get('/api/runs').json()['runs'][0]['verification'] is None
