"""Intention and decision contracts use fixtures, not artistic preference labels."""
from copy import deepcopy
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
import pytest

from kusama.artifacts import sha256
from kusama.dashboard import create_app
from kusama.reviews import ReviewConflict, ReviewStore
from test_continuation import setup, Backend, forbidden


def mapped(request):
    request['intention_map'] = {
        'parent_sha256': request['selection']['image_sha256'],
        'author': 'fixture operator', 'origin': 'development_example',
        'whole': 'Let enclosure and figure remain in tension',
        'elements': [
            {'id': 'boundary', 'role': 'An uncertain opening', 'region': [.25, .25, .75, .75],
             'subtlety': 'Noticeable after the figure', 'relationships': ['figure'],
             'must_survive': 'Continuity with the whole'},
            {'id': 'figure', 'role': 'Vulnerability', 'region': [0, 0, .25, .25],
             'subtlety': 'Small but visible', 'must_survive': 'Exact marks',
             'status': 'resolved', 'protect_pixels': True}],
        'target_ids': ['boundary'], 'operation': 'conceal',
        'attention_reason': 'Test whether the opening feels less reassuring',
        'expected_reading': 'A tentative hypothesis, not a viewer response'}
    return request


def decision(artifact_id='root', choice='select'):
    return {'submission_id': 'choice-1', 'artifact_id': artifact_id, 'choice': choice,
            'author': 'fixture operator', 'origin': 'development_example',
            'reason': 'Retain this branch for another test',
            'local_assessment': 'The boundary changed',
            'whole_assessment': 'Its meaning remains uncertain',
            'preservation_assessment': 'The figure pixels survived'}


def test_map_freezes_parent_and_survives_execution_resume(setup):
    store, request, config = setup
    mapped(request)
    prepared = store.prepare('mapped', request, config)
    assert store.prepare('mapped', request, config) == prepared
    assert prepared['request']['intention_map']['expected_reading'] == request['intention_map']['expected_reading']
    result = store.execute('mapped', Backend)
    assert result['model_calls'] == 2
    assert all(output['preservation']['outside_changed_pixels'] == 0 for output in result['outputs'])
    assert store.execute('mapped', forbidden)['model_calls'] == 2
    assert result['reviews']['human_labels'] is None
    request['intention_map']['whole'] = 'Changed after freeze'
    with pytest.raises(ValueError, match='frozen'):
        store.prepare('mapped', request, config)


@pytest.mark.parametrize('problem', ['parent', 'protected', 'whole', 'unknown', 'resolved', 'empty', 'off_target'])
def test_bad_maps_fail_before_job_registration(setup, problem):
    store, request, config = setup
    intentions = mapped(request)['intention_map']
    if problem == 'parent': intentions['parent_sha256'] = '0' * 64
    if problem == 'protected': intentions['elements'][1]['region'] = [0, 0, .5, .5]
    if problem == 'whole': request['revision'].update(method='whole', region=None)
    if problem == 'unknown': intentions['elements'][0]['relationships'] = ['missing']
    if problem == 'resolved': intentions['elements'][0]['status'] = 'leave_alone'
    if problem == 'empty': intentions['elements'][1]['region'] = [0, 0, .001, .001]
    if problem == 'off_target': request['revision']['region'] = [.8, .8, 1, 1]
    with pytest.raises(ValueError):
        store.prepare('bad', request, config)
    assert Backend.calls == [] and not (store.root / 'jobs.sqlite3').exists()


def test_linked_revision_requires_reason_to_reopen_resolved_detail(setup):
    store, request, config = setup
    mapped(request)
    store.prepare('first', request, config)
    result = store.execute('first', Backend)
    child = result['outputs'][0]
    next_request = deepcopy(request)
    next_request['selection'].update(path=child['path'], image_sha256=child['sha256'])
    intentions = next_request['intention_map']
    intentions.update(parent_sha256=child['sha256'], previous_continuation_id='first', target_ids=['figure'])
    intentions['elements'][1].update(status='unresolved', protect_pixels=False)
    next_request['revision']['region'] = [0, 0, .25, .25]
    with pytest.raises(ValueError, match='revisit reason'):
        store.prepare('second', next_request, config)
    intentions['revisit_reasons'] = {'figure': 'Revisit its scale to test vulnerability in the whole'}
    store.prepare('second', next_request, config)
    assert len(Backend.calls) == 2  # Preparation did not execute the next round.
    assert store.get('first')['request']['intention_map']['elements'][1]['protect_pixels'] is True
    second = store.execute('second', Backend)
    assert {c['parent_sha256'] for c in Backend.calls[2:]} == {child['sha256']}
    assert second['reviews']['human_labels'] is None


def test_link_rejects_unrelated_parent_and_unknown_revisit(setup):
    store, request, config = setup
    mapped(request)
    store.prepare('first', request, config)
    request['intention_map'].update(previous_continuation_id='first', revisit_reasons={'invented': 'Reason'})
    with pytest.raises(ValueError, match='previous intention'):
        store.prepare('bad', request, config)
    from PIL import Image
    parent = store.root / 'unrelated.png'
    Image.new('RGB', (64, 64), 'yellow').save(parent)
    request['selection'].update(path=str(parent), image_sha256=sha256(parent))
    request['intention_map'].update(parent_sha256=sha256(parent), revisit_reasons={})
    with pytest.raises(ValueError, match='intact previous'):
        store.prepare('bad', request, config)


def test_decision_is_attributed_idempotent_and_creates_no_reviews(setup):
    store, request, config = setup
    store.prepare('mapped', mapped(request), config)
    with pytest.raises(ValueError, match='Complete'):
        store.decide('mapped', decision())
    result = store.execute('mapped', Backend)
    data = decision(result['outputs'][0]['id'])
    record = store.decide('mapped', data)
    assert record['image_sha256'] == result['outputs'][0]['sha256']
    assert record['parent_sha256'] == result['parent']['sha256']
    assert record['intention_map_sha256']
    assert store.decide('mapped', data) == record
    with pytest.raises(ReviewConflict):
        store.decide('mapped', data | {'reason': 'Changed'})
    assert len(store.get('mapped')['decisions']) == 1
    assert ReviewStore(store.root).history(result['reviews']['collection_id']) == {'votes': [], 'advancements': []}
    assert len(Backend.calls) == 2


def test_stop_before_render_cancels_work_and_is_available_via_api(setup):
    store, request, config = setup
    store.prepare('mapped', mapped(request), config)
    client = TestClient(create_app(store.root), base_url='http://localhost')
    response = client.post('/api/continuations/mapped/decisions', json=decision(choice='stop'),
                           headers={'X-Kusama-Review': '1'})
    assert response.status_code == 200
    assert store.get('mapped')['status'] == 'stopped'
    assert store.execute('mapped', forbidden)['model_calls'] == 0
    assert Backend.calls == []


def test_local_views_use_same_original_bytes_and_matching_crop_bounds(setup):
    store, request, config = setup
    store.prepare('mapped', mapped(request), config)
    result = store.execute('mapped', Backend)
    hashes = [sha256(Path(a['path'])) for a in [result['parent'], *result['outputs']]]
    for web in (False, True):
        body = store.gallery_html('mapped', web=web)
        svgs = [ET.fromstring(svg) for svg in re.findall(r'<svg\b.*?</svg>', body, re.S)]
        assert len(svgs) == 4
        assert all(svg.attrib['viewBox'] == '16 16 32 32' for svg in svgs[1:])
        assert all('detail:' in svg.attrib['aria-label'] for svg in svgs[1:])
        assert 'not higher-resolution render passes' in body
        assert 'Intention map' in body and 'Vulnerability' in body
        if web:
            assert ' style=' not in body and '<style>' not in body
    assert [sha256(Path(a['path'])) for a in [result['parent'], *result['outputs']]] == hashes


def test_legacy_request_keeps_its_serialized_shape(setup):
    store, request, config = setup
    store.prepare('legacy', request, config)
    frozen = json.loads((store.root / 'continuations/legacy/inputs.json').read_text())
    assert 'intention_map' not in frozen['request']


def test_api_decision_respects_running_worker_lock(setup):
    from kusama.ledger import Ledger
    store, request, config = setup
    store.prepare('study', request, config)
    client = TestClient(create_app(store.root), base_url='http://localhost')
    with Ledger(store.root / 'jobs.sqlite3') as ledger, ledger.lock('continuation_study'):
        response = client.post('/api/continuations/study/decisions', json=decision(choice='stop'),
                               headers={'X-Kusama-Review': '1'})
    assert response.status_code == 409
    assert store.get('study')['decisions'] == []
