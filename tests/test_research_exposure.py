"""Offline tests of exposure policy; no claim about artistic usefulness."""
import json
import os
from pathlib import Path

import pytest

from kusama import research as r
from kusama.artifacts import sha256
from kusama.ledger import SpecMismatch
from kusama.panel_runs import StartRun
from kusama.research_context import project_context
from kusama.research_exposure import expose
from kusama.research_sources import ResearchSources
from test_corpus import cleveland, png
from test_research_grounded import analysis
from test_research_sources import Response
from test_retrieval import tiny_index


def policy(seed=23, count=2):
    return {'seed': seed, 'count': count, 'breadth_queries': ['textile', 'instrument', 'armor']}


class Transport:
    def __init__(self):
        self.urls = []

    def __call__(self, request, timeout):
        self.urls.append(request.full_url)
        if request.full_url.startswith('https://openaccess-cdn.clevelandart.org/'):
            return Response(png(), 'image/png')
        assert request.full_url.startswith('https://openaccess-api.clevelandart.org/')
        return Response(json.dumps({'data': [cleveland(i) for i in range(1, 9)]}).encode())


def test_online_draw_freezes_pool_and_reuses_on_resume(tmp_path):
    transport = Transport()
    config = {'allow_network': True, 'providers': ['cleveland'], 'max_requests': 1}
    with_backend = ResearchSources(tmp_path, config, opener=transport)
    first = expose(with_backend, policy())
    assert len(first['sources']) == 2
    assert first['eligible_count'] == 8
    assert len({s['reference_id'] for s in first['sources']}) == 2
    assert first['bootstrap']['query'] in policy()['breadth_queries']
    assert first['status'] == 'complete'
    assert all(s['exposure'] == 'chance' and s['relevance_score'] is None for s in first['sources'])
    assert with_backend.summary()['requests'] == 1
    assert with_backend.summary()['acquisitions'] == 0
    assert not with_backend.store.references()
    assert sha256(tmp_path / first['snapshot_path']) == first['snapshot_sha256']
    with_backend.close()
    resumed = ResearchSources(tmp_path, config, opener=transport)
    assert expose(resumed, policy()) == first
    assert len(transport.urls) == 1
    with pytest.raises(SpecMismatch, match='policy changed'):
        expose(resumed, policy(seed=24))
    resumed.close()


def test_seed_variation_deduplication_and_exclusions(tiny_index, tmp_path):
    index, _, _ = tiny_index
    outcomes = set()
    for seed in range(8):
        backend = ResearchSources(tmp_path / f'run-{seed}', index_path=index.directory)
        result = expose(backend, policy(seed, count=1))
        assert result['eligible_count'] == 3  # whole works, not index detail crops
        assert result['bootstrap'] is None
        assert backend.summary()['requests'] == 0
        outcomes.add(result['sources'][0]['source_id'])
        backend.close()
    assert len(outcomes) > 1
    backend = ResearchSources(tmp_path / 'excluded', {'providers': ['cleveland'],
        'excluded_source_ids': ['cleveland_1']}, index_path=index.directory)
    result = expose(backend, policy())
    assert all(s['reference_id'] != 'cleveland_1' for s in result['eligible_pool'])
    assert not any(s['source_id'].startswith('cleveland_1:') for s in result['sources'])
    backend.close()


def test_draw_ignores_relevance_and_no_network_without_permission(tmp_path):
    transport = Transport()
    online = ResearchSources(tmp_path / 'catalog', {'allow_network': True, 'providers': ['cleveland']}, opener=transport)
    # Same work from two different search queries must have only one pool slot.
    online.search('query one', mode='remote', provider='cleveland')
    online.search('query two', mode='remote', provider='cleveland')
    assert online.summary()['candidates'] == 16
    first = expose(online, policy())
    assert first['eligible_count'] == 8
    assert first['bootstrap'] is None and len(transport.urls) == 2
    online.close()
    offline = ResearchSources(tmp_path / 'offline', opener=lambda *a, **kw: pytest.fail('network disabled'))
    result = expose(offline, policy())
    assert result['status'] == 'empty' and result['sources'] == []
    assert offline.summary()['requests'] == 0
    offline.close()


def test_failed_and_interrupted_draws_are_not_retried(tmp_path, monkeypatch):
    calls = []
    def fail(request, timeout):
        calls.append(request.full_url)
        raise TimeoutError('uncertain request')
    config = {'allow_network': True, 'providers': ['cleveland']}
    backend = ResearchSources(tmp_path / 'failed', config, opener=fail)
    first = expose(backend, policy())
    assert first['status'] == 'failed' and first['sources'] == []
    assert 'TimeoutError' in first['error']
    assert expose(backend, policy()) == first and len(calls) == 1
    assert backend.summary()['charged_network_bytes'] > 0
    backend.close()
    backend = ResearchSources(tmp_path / 'interrupted', config, opener=Transport())
    def interrupt(*a, **kw):
        raise KeyboardInterrupt()
    monkeypatch.setattr(backend, '_snapshot', interrupt)
    with pytest.raises(KeyboardInterrupt):
        expose(backend, policy())
    assert backend.summary()['requests'] == 1
    backend.close()
    resumed = ResearchSources(tmp_path / 'interrupted', config, opener=lambda *a, **kw: pytest.fail('no retry'))
    with pytest.raises(SpecMismatch, match='unsettled'):
        expose(resumed, policy())
    resumed.close()


def test_changed_snapshot_is_rejected(tmp_path):
    backend = ResearchSources(tmp_path)
    result = expose(backend, policy())
    (tmp_path / result['snapshot_path']).write_text('{}')
    with pytest.raises(SpecMismatch, match='snapshot missing or changed'):
        expose(backend, policy())
    backend.close()


def test_plan_seed_and_model_selection_boundary(tmp_path):
    request = {'id': 'exposure', 'project_id': 'p', 'brief_id': 'b', 'problem': 'Who owns a boundary?',
        'authorship': 'operator', 'resource_root': str(tmp_path / 'resources'), 'max_steps': 24,
        'fixture': True, 'random_exposure_count': 2, 'random_exposure_seed': 0,
        'source_config': {'allow_network': True, 'providers': ['cleveland']}}
    root = tmp_path / 'run'
    spec = r.plan_research(root, request)
    assert r.plan_research(root, request) == spec and spec['exposure_policy']['seed'] == 0
    assert spec['max_reasoning_calls'] == 24
    assert not (tmp_path / 'resources').exists()
    assert not (root / 'sources/chance_exposure_started.json').exists()
    transport = Transport()
    backend = ResearchSources(root / 'sources', spec['source_config'], opener=transport)
    action = r.ResearchActionEnvelope.model_validate(analysis())
    row = r._execute_action(root, spec, r._state(spec, []), [], action, backend)
    steps = [row]
    state = r._state(spec, steps)
    payload = r._payload(spec, state, steps)
    assert len(payload['chance_candidates']) == 2
    sid = payload['chance_candidates'][0]['candidate_id']
    assert sid in state['known'] and not state['opened']
    assert payload['candidate_memory'] == []
    assert row['origin'] == 'fixture_actor'
    assert row['tool_result']['sources'][0]['exposure_origin'] == 'operator_authored_random_sampling'
    # Topic relevance is still required for acquisition, unless Qwen gives its
    # own explicit explanation of why an indirect source could help.
    with pytest.raises(ValueError, match='relevance gate'):
        r._execute_action(root, spec, state, steps, r.ResearchActionEnvelope.model_validate({
            'action': 'acquire', 'candidate_id': sid, 'reason': 'Investigate chance'}), backend)
    for stage in range(11):
        projected = project_context({'system': spec['policy'], 'payload': payload, 'images': []}, stage)
        cards = projected['payload']['chance_candidates']
        assert [c['source_id'] for c in cards] == [c['source_id'] for c in payload['chance_candidates']]
        assert all(c['exposure'] == 'chance' and c['exposure_origin'] == 'operator_authored_random_sampling' for c in cards)
        assert all(c['title'] and c['description'] for c in cards)
    reject = r._execute_action(root, spec, state, steps, r.ResearchActionEnvelope.model_validate({
        'action': 'reject', 'candidate_id': sid, 'reason': 'No useful connection to my inquiry'}), backend)
    state = r._state(spec, steps + [reject])
    remaining = r._payload(spec, state, steps + [reject])['chance_candidates']
    assert len(remaining) == 1 and all(c['candidate_id'] != sid for c in remaining)
    assert not state['hypotheses']
    # Replaying the analysis uses exactly the saved draw, with no second search.
    replay = r._execute_action(root, spec, r._state(spec, []), [], action, backend)
    assert replay['chance_exposure'] == row['chance_exposure'] and len(transport.urls) == 1
    # Model-authored indirect usefulness can permit acquisition. The draw alone
    # did not supply this rationale or authorize a hypothesis without inspection.
    selected = remaining[0]['candidate_id']
    acquired = r._execute_action(root, spec, state, steps + [reject], r.ResearchActionEnvelope.model_validate({
        'action': 'acquire', 'candidate_id': selected, 'reason': 'Inspect its material structure',
        'relevance_override': 'A carved vessel may suggest how containment makes a boundary; inspect before attributing a property.'}), backend)
    assert acquired['opened_sources'][0]['images']
    next_payload = r._payload(spec, r._state(spec, steps + [reject, acquired]), steps + [reject, acquired])
    assert next_payload['chance_candidates'] == []
    assert next_payload['vision_probe_required']
    backend.close()


def test_automatic_seeds_differ_and_off_is_frozen(tmp_path):
    request = {'id': 'seed', 'project_id': 'p', 'brief_id': 'b', 'problem': 'A question',
               'authorship': 'operator', 'resource_root': str(tmp_path / 'resources'), 'random_exposure_count': 2}
    first = r.plan_research(tmp_path / 'first', request)
    assert r.plan_research(tmp_path / 'first', request)['exposure_policy'] == first['exposure_policy']
    second = r.plan_research(tmp_path / 'second', request)
    assert first['exposure_policy']['seed'] != second['exposure_policy']['seed']
    disabled = r.plan_research(tmp_path / 'off', {**request, 'random_exposure_count': 0})
    assert disabled['exposure_policy'] is None
    assert StartRun(submission_id='new', question='A question').random_exposure_count == 2
    with pytest.raises(ValueError):
        StartRun(submission_id='new', question='A question', random_exposure_count=3)


@pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1',
                    reason='Explicit pinned local processor check; no inference')
def test_chance_shelf_and_dense_search_fit_actual_processor(tmp_path):
    from kusama.director import DirectorConfig
    from kusama.research_context import admit_context
    cache = Path(__file__).resolve().parents[1] / '.cache/huggingface'
    root = tmp_path / 'research'
    spec = r.plan_research(root, {'id': 'processor', 'project_id': 'p', 'brief_id': 'b',
        'problem': 'How can care become a way to hold someone captive?', 'authorship': 'fixture operator',
        'resource_root': str(tmp_path / 'unadmitted'), 'cache_dir': str(cache), 'device': 'cpu',
        'random_exposure_count': 2, 'random_exposure_seed': 1, 'study_development': True,
        'source_config': {'allow_network': True, 'providers': ['cleveland']}})
    backend = ResearchSources(root / 'sources', spec['source_config'], opener=Transport())
    row = r._execute_action(root, spec, r._state(spec, []), [], r.ResearchActionEnvelope.model_validate(analysis()), backend)
    hits = [{'source_id': f'{provider}-{i}', 'candidate_id': f'{provider}-{i}', 'provider': provider,
             'title': 'Documented source title ' * 10, 'description': 'Documented collection description. ' * 50,
             'medium': 'Carved material', 'maker': 'Documented maker', 'acquired': False}
            for provider in ('met', 'artic', 'cleveland') for i in range(8)]
    steps = [row, {'index': 1, 'action': {'action': 'search', 'query': 'figure', 'mode': 'remote'},
                   'tool_result': {'sources': hits}}]
    state = r._state(spec, steps)
    full = {'system': spec['policy'], 'payload': r._payload(spec, state, steps), 'images': []}
    admitted, count = admit_context(full, DirectorConfig(cache, device='cpu', max_new_tokens=1800, max_image_pixels=65536))
    assert count['input_tokens'] <= 6392 and count['reserved_output_tokens'] == 1800
    assert {c['source_id'] for c in admitted['payload']['chance_candidates']} == {c['source_id'] for c in full['payload']['chance_candidates']}
    assert all(c['exposure'] == 'chance' for c in admitted['payload']['chance_candidates'])
    assert admitted['payload']['problem'] == full['payload']['problem']
    assert not (tmp_path / 'unadmitted').exists()
    backend.close()
