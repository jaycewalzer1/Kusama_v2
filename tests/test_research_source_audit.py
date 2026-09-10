"""Collection contract and full research-loop checks; no network or model inference."""
from copy import deepcopy
import io
import json
import urllib.error
import urllib.parse

import pytest

from kusama import research as r
from kusama.research_sources import ResearchSources
from test_corpus import cleveland, png
from test_research import setup, Director, hypothesis
from test_research_sources import Response


def record(provider, identifier=1):
    if provider == 'met':
        return {'objectID': identifier, 'title': 'Fixture portrait',
                'objectURL': f'https://www.metmuseum.org/art/collection/search/{identifier}',
                'primaryImageSmall': f'https://images.metmuseum.org/{identifier}.jpg',
                'isPublicDomain': True, 'artistDisplayName': 'Fixture maker',
                'objectDate': '1900', 'medium': 'Oil', 'culture': 'Documented culture'}
    if provider == 'artic':
        return {'id': identifier, 'title': 'Fixture portrait', 'image_id': 'fixture_image',
                'is_public_domain': True, 'artist_display': 'Fixture maker',
                'date_display': '1900', 'medium_display': 'Oil', 'place_of_origin': 'Documented place'}
    return cleveland(identifier)


def envelope(provider, rows):
    return {'data': rows, **({'config': {'iiif_url': 'https://www.artic.edu/iiif/2'}} if provider == 'artic' else {})}


class Transport:
    def __init__(self, provider):
        self.provider, self.urls = provider, []

    def __call__(self, request, timeout):
        url = request.full_url
        self.urls.append(url)
        if '.jpg' in url:
            return Response(png(), 'image/png')
        if self.provider == 'met':
            if '/search?' in url:
                assert urllib.parse.urlsplit(url).path == '/public/collection/v1.1/search'
                assert urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)['limit'] == ['1']
                return Response(json.dumps({'total': 1, 'objectIDs': [1]}).encode())
            return Response(json.dumps(record('met')).encode())
        return Response(json.dumps(envelope(self.provider, [record(self.provider)])).encode())


@pytest.mark.parametrize('provider', ['met', 'artic', 'cleveland'])
def test_remote_search_acquire_reopen_draft_commit_and_replay(setup, monkeypatch, provider):
    output, _, request, _, _ = setup
    request['source_config'] = {'allow_network': True, 'providers': [provider]}
    transport = Transport(provider)
    decisions, source_ids = [], []
    def model(director, system, payload, images, validator, **kwargs):
        index = len(decisions)
        if index == 0:
            value = {'action': 'search', 'query': 'portrait', 'mode': 'remote', 'provider': provider, 'limit': 1}
        elif index == 1:
            value = {'action': 'acquire', 'candidate_id': payload['last_tool_result']['sources'][0]['candidate_id']}
        elif index == 2:
            source_ids.append(payload['last_tool_result']['source_id'])
            value = {'action': 'open', 'source_id': source_ids[0]}
            assert images
        else:
            value = {'action': 'draft' if index == 3 else 'commit', 'hypothesis': hypothesis((source_ids[0],))}
            assert images
        decisions.append(value)
        return {'raw_response': json.dumps(value), 'output': validator(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(r, '_multimodal_json', model)
    def sources(root, config, index_path=None):
        return ResearchSources(root, config, index_path, opener=transport)
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=sources)
    assert report['status'] == 'hypotheses_ready'
    assert report['reasoning_calls'] == 5 and len(report['hypotheses']) == 1
    assert all(step['status'] == 'completed' for step in report['steps'])
    assert report['network']['acquisition_states'] == {'complete': 1}
    assert len(transport.urls) == (3 if provider == 'met' else 2)
    assert report['human_labels'] == 0 and report['fixture']
    before = list(transport.urls)
    again = r.run_research(output, director_factory=Director, sources_factory=sources)
    assert again['hypotheses'] == report['hypotheses']
    assert len(decisions) == 5 and transport.urls == before


@pytest.mark.parametrize('provider', ['met', 'artic', 'cleveland'])
@pytest.mark.parametrize('body', [[], {'error': 'upstream unavailable'}, {'data': {}, 'objectIDs': {}}])
def test_malformed_search_envelopes_are_not_empty_successes(tmp_path, provider, body):
    calls = []
    def transport(request, timeout):
        calls.append(request.full_url)
        return Response(json.dumps(body).encode())
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport)
    try:
        for _ in range(2):
            with pytest.raises(ValueError, match='Malformed .* search response'):
                backend.search('portrait', mode='remote', provider=provider)
        assert len(calls) == 1  # saved raw response is reused, never fetched again
        assert backend.summary()['candidates'] == 0
    finally:
        backend.close()


@pytest.mark.parametrize('provider', ['artic', 'cleveland'])
def test_bad_records_do_not_hide_valid_neighbors(tmp_path, provider):
    rows = [None, [], {'id': 99, 'title': 'Malformed'}, record(provider)]
    backend = ResearchSources(tmp_path, {'allow_network': True},
        opener=lambda *args, **kwargs: Response(json.dumps(envelope(provider, rows)).encode()))
    try:
        result = backend.search('portrait', mode='remote', provider=provider)
        assert len(result['exclusions']) == 3 and len(result['sources']) == 1
    finally:
        backend.close()


def test_met_missing_and_mismatched_records_keep_valid_results_without_retry(tmp_path):
    calls = []
    def transport(request, timeout):
        url = request.full_url
        calls.append(url)
        if '/search?' in url:
            return Response(json.dumps({'total': 4, 'objectIDs': [True, 1, 2, 3]}).encode())
        if url.endswith('/1'):
            raise urllib.error.HTTPError(url, 404, 'gone', {}, io.BytesIO(b'not found'))
        if url.endswith('/2'):
            return Response(json.dumps(record('met', 200)).encode())
        return Response(json.dumps(record('met', 3)).encode())
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport)
    try:
        result = backend.search('portrait', mode='remote', provider='met')
        assert len(result['sources']) == 1 and len(result['exclusions']) == 3
        assert len(calls) == 4
        repeated = backend.search('portrait', mode='remote', provider='met')
        assert repeated['sources'] == result['sources'] and len(calls) == 4
        assert backend.summary()['request_states'] == {'complete': 3, 'failed': 1}
    finally:
        backend.close()


def test_met_null_empty_results_are_supported(tmp_path):
    backend = ResearchSources(tmp_path, {'allow_network': True},
        opener=lambda *args, **kwargs: Response(b'{"total":0,"objectIDs":null}'))
    try:
        assert backend.search('unmatched', mode='remote', provider='met')['sources'] == []
    finally:
        backend.close()


@pytest.mark.parametrize('status', [429, 500, 503])
def test_http_failures_retain_charge_and_never_retry(tmp_path, status):
    calls = []
    def transport(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, status, 'fixture failure', {}, io.BytesIO(b'failure body'))
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport)
    try:
        with pytest.raises(urllib.error.HTTPError):
            backend.search('portrait', mode='remote', provider='cleveland')
        with pytest.raises(RuntimeError, match='no silent retry'):
            backend.search('portrait', mode='remote', provider='cleveland')
        assert len(calls) == 1
        state, charged, payload = backend.db.execute('SELECT state,charged_bytes,payload FROM requests').fetchone()
        payload = json.loads(payload)
        assert state == 'failed' and charged == backend.config.max_response_bytes + 1
        assert payload['http_status'] == status
        assert backend.store.asset_path(payload['raw_path']).read_bytes() == b'failure body'
    finally:
        backend.close()


def test_failed_http_body_read_still_records_original_failure(tmp_path):
    class BrokenBody:
        def read(self, *args):
            raise TimeoutError('body interrupted')
        def close(self):
            pass
    def transport(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 503, 'unavailable', {}, BrokenBody())
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport)
    try:
        with pytest.raises(urllib.error.HTTPError):
            backend.search('portrait', mode='remote', provider='cleveland')
        state, payload = backend.db.execute('SELECT state,payload FROM requests').fetchone()
        assert state == 'failed'
        assert json.loads(payload)['body_read_error'] == 'TimeoutError: body interrupted'
    finally:
        backend.close()


def test_discovery_query_is_not_candidate_relevance_and_acquired_state_is_visible(tmp_path):
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=Transport('cleveland'))
    try:
        result = backend.search('unicornzz', mode='remote', provider='cleveland', limit=1)
        candidate = result['sources'][0]
        assert {'maker', 'culture', 'date', 'medium', 'tags'} <= candidate.keys()
        assert backend.search('unicornzz')['sources'] == []
        assert backend.search(candidate['title'])['sources']
        acquired = backend.acquire(candidate['candidate_id'])
        repeated = backend.search('unicornzz', mode='remote', provider='cleveland', limit=1)['sources'][0]
        assert repeated['acquired'] and repeated['acquired_source_id'] == acquired['source_id']
    finally:
        backend.close()


def test_prompt_excerpt_bounds_do_not_modify_saved_source_text():
    source = {'source_id': 's1', 'description': 'x' * 20000, 'title': 't' * 1000,
              'maker': 'm' * 1000, 'tags': ['z' * 1000] * 100}
    original = deepcopy(source)
    tool = {'sources': [source]}
    compact = r._compact_tool(tool)['sources'][0]
    assert len(compact['description']) == 500 and len(compact['title']) == 200
    assert len(compact['maker']) == 200 and len(compact['tags']) == 8
    assert set(compact['prompt_excerpt']['truncated_fields']) == {'description', 'title', 'maker', 'tags'}
    recalled = r._compact_tool({'recalled': [{'index': 0, 'action': {}, 'tool_result': tool}]})
    assert recalled['recalled'][0]['tool_result']['sources'][0] == compact
    assert source == original


def test_snapshot_cannot_exceed_source_storage_budget(tmp_path):
    backend = ResearchSources(tmp_path, {'max_storage_bytes': 1024 * 1024})
    try:
        with pytest.raises(RuntimeError, match='storage byte budget'):
            backend._snapshot('fixture', {'text': 'x' * (2 * 1024 * 1024)})
        assert backend.db.execute('SELECT COUNT(*) FROM snapshots').fetchone()[0] == 0
    finally:
        backend.close()
