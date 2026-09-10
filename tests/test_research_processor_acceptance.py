"""Opt-in production-path acceptance with real cached processors, no inference.

Run KUSAMA_PROCESSOR_ACCEPTANCE=1 uv run --locked pytest -q this_file.
Only model.generate, FLUX pixels and HTTP transport are replaced. Admission,
image tensors, generated-token decoding, JSON validation, source normalization,
acquisition, panel dispatch, provenance and resource ledgers are production code.
These are scripted software scenarios, never model capability measurements.
"""
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import urllib.error

import pytest
from fastapi.testclient import TestClient
import torch

from kusama import backend, director, panel_runs, research, research_render
from kusama.artifacts import sha256
from kusama.dashboard import create_app
from kusama.resources import SessionBudget, load_config
from kusama.research_sources import ResearchSources
from test_research_render import Renderer, hypothesis
from test_research_source_audit import Transport
from test_research_sources import Response

pytestmark = pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1',
    reason='Explicit processor acceptance requires pinned local Qwen/FLUX caches; no inference')
CACHE = Path(__file__).resolve().parents[1] / '.cache/huggingface'


@pytest.fixture(scope='module')
def processor():
    from transformers import AutoProcessor
    # Missing cache must fail, not silently skip this explicitly requested gate.
    return AutoProcessor.from_pretrained(director.MODEL_ID, revision=director.MODEL_REVISION,
        cache_dir=CACHE, local_files_only=True, trust_remote_code=False, use_fast=False)


def run_panel(tmp_path, monkeypatch, processor, choose, transport, *, steps=24, directions=1,
              grounded=False, random_exposure_count=0,
              question='How can a boundary shelter someone while also excluding them?',
              context='Operator-authored software diagnostic; scripted responses and images, no artistic judgment.'):
    from research_legacy import use_legacy_protocol
    if not grounded:
        use_legacy_protocol(monkeypatch)
    config = load_config().model_copy(update={'device': 'cpu', 'cache_dir': str(CACHE)})
    monkeypatch.setattr(panel_runs, 'load_config', lambda *a: config)
    store = panel_runs.PanelRuns(tmp_path / 'artifacts', coordinator=tmp_path / 'coordinator')
    client = TestClient(create_app(store.root), base_url='http://localhost')
    # Start through the same API as the form; prevent a separate real worker.
    monkeypatch.setattr(panel_runs.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(pid=os.getpid()))
    response = client.post('/api/start-run', headers={'X-Kusama-Review': '1'}, json={
        'submission_id': 'processor-gate', 'question': question, 'context': context,
        'research_steps': steps, 'directions': directions, 'images_per_direction': 2, 'study_development': False,
        'random_exposure_count': random_exposure_count, 'total_token_limit': 16384})
    assert response.status_code == 202, response.text
    entry = response.json(); output = store.directory(entry['run_id']); calls = []

    class TokenDirector:
        def __init__(self, config):
            self.config = config
            self.processor = processor
            self.execution_settings = {'status': 'ready', 'device': 'cpu', 'origin': 'scripted_token_fixture'}
            self.model = self
        def prepare(self):
            pass
        def close(self):
            pass
        def generate(self, input_ids, max_new_tokens, **kwargs):
            index = len(calls)
            frozen = json.loads((output / 'research' / f'decision-{index:03}.json').read_text())
            assert input_ids.shape[-1] == frozen['context_admission']['input_tokens'] <= frozen['context_admission']['input_token_limit']
            if frozen['images']:
                assert 'pixel_values' in kwargs and 'image_grid_thw' in kwargs
                assert len(kwargs['image_grid_thw']) == len(frozen['images'])
            value = choose(index, frozen['payload'], frozen['images'], output)
            raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            tokens = processor.tokenizer.encode(raw, add_special_tokens=False)
            # Simulate the real output cap, including invalid truncated JSON.
            emitted = tokens[:max_new_tokens]
            calls.append({'payload': frozen['payload'], 'images': frozen['images'], 'raw': raw,
                          'input_tokens': input_ids.shape[-1], 'output_tokens': len(emitted)})
            return torch.cat([input_ids, torch.tensor([emitted], dtype=torch.long)], dim=1)

    class TokenRenderer(Renderer):
        def prompt_token_count(self, prompt):
            return backend.cached_prompt_token_count(prompt, cache_dir=CACHE, revision=self.config.model_revision)
    TokenRenderer.loads = 0; TokenRenderer.calls = []; TokenRenderer.fail_seed = None
    def sources(root, cfg, index_path=None):
        return ResearchSources(root, cfg, index_path, opener=transport)
    try:
        store.execute(entry['run_id'],
            research_runner=lambda path, **kw: research.run_research(path, director_factory=TokenDirector,
                sources_factory=sources, **kw),
            render_runner=lambda path, **kw: research_render.run_render(path, backend_factory=TokenRenderer, **kw))
    except ValueError as exc:
        if 'Research stopped with response_rejected' not in str(exc):
            raise
    report = research.read_report(output / 'research')
    assert report['reasoning_calls'] == len(calls)
    assert len(calls) <= steps
    for index, call in enumerate(calls):
        saved = json.loads((output / 'research' / f'response-{index:03}.json').read_text())
        assert saved['metadata']['input_tokens'] == call['input_tokens']
        assert saved['metadata']['generated_tokens'] == call['output_tokens']
    with SessionBudget(output) as budget:
        assert not budget.summary()['unsettled']
        assert len(budget.summary()['calls']) == len(calls) + len(TokenRenderer.calls)
    # Terminal replay must work with every execution boundary forbidden.
    counts = (len(calls), len(transport.urls))
    replay = research.run_research(output / 'research', director_factory=TokenDirector, sources_factory=sources)
    assert replay['hypotheses'] == report['hypotheses']
    assert (len(calls), len(transport.urls)) == counts
    data = client.get(f"/api/runs/{entry['id']}/inspect").json()
    assert not data['errors']
    assert len(data['images']) == len(TokenRenderer.calls)
    for image in data['images']:
        assert client.get(image['url']).status_code == 200
    assert not list(output.rglob('reviews.sqlite3'))
    return report, calls, store.get(entry['run_id']), TokenRenderer.calls


def test_repeated_rewrite_incident_retains_charge_and_never_retries(tmp_path, monkeypatch, processor):
    incident = json.loads((Path(__file__).parent / 'fixtures/research/repeated_rewrite_queries.json').read_text())
    transport = Transport('artic')
    def choose(index, payload, images, output):
        if index == 2:
            assert payload['response_contract']['required_fields_by_action']['rewrite_queries'] == ['action', 'reason', 'queries']
            assert 'objects each containing query' in payload['last_tool_result']['tool_error']
        return incident['raw_responses'][index]
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, transport, grounded=True)
    assert report['status'] == 'response_rejected' and panel['status'] == 'failed'
    assert report['reasoning_calls'] == len(calls) == 3
    assert calls[2]['raw'] == incident['raw_responses'][2]
    assert 'Extra data: line 5 column 1 (char 580)' in report['steps'][2]['error']
    assert report['steps'][1]['status'] == 'tool_rejected'
    assert not transport.urls and not renders


def test_scripted_complete_rewrite_then_search_under_new_protocol(tmp_path, monkeypatch, processor):
    from test_research_grounded import analysis
    transport = Transport('artic')
    def choose(index, payload, images, output):
        if index == 0:
            return analysis()
        if index == 1:
            return {'action': 'rewrite_queries', 'queries': [
                {'query': 'portrait', 'field': 'subject'}, {'query': 'figure', 'field': 'object'},
                {'query': 'painting', 'field': 'technique'}], 'reason': 'Scripted protocol diagnostic.'}
        if index == 2:
            return {'action': 'search', 'mode': 'remote', 'query': payload['prepared_queries'][0]['query'], 'reason': 'Scripted query dispatch.'}
        return {'action': 'stop', 'reason': 'Software diagnostic complete; no artistic conclusion.'}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, transport, grounded=True)
    assert report['status'] == 'model_stop' and panel['status'] == 'partial'
    assert len(calls) == 4 and all(step['status'] == 'completed' for step in report['steps'])
    assert transport.urls and not renders


@pytest.mark.parametrize('token_limit', [8192, 16384])
def test_saved_21_candidate_overflow_fits_and_dispatches_once(tmp_path, processor, token_limit):
    from kusama import research_context
    from copy import deepcopy
    fixture = Path(__file__).parent / 'fixtures/research/search_context_overflow.json'
    full = json.loads(fixture.read_text())
    saved = deepcopy(full)
    cfg = director.DirectorConfig(CACHE, device='cpu', max_new_tokens=1800, max_image_pixels=65536,
                                  max_input_tokens=token_limit)
    holder = SimpleNamespace(config=cfg, processor=processor, execution_settings={'status': 'ready', 'device': 'cpu'})
    def count(value):
        result, _ = director._prepare_multimodal_inputs(holder, value['system'], value['payload'], [],
            allow_text_only=True, compact_json=True)
        return result['metadata']['input_tokens']
    assert count(research_context.project_context(full, 6)) > 8500
    admitted, admission = research_context.fit_context(full, count, combined_limit=token_limit)
    assert admission['input_tokens'] <= token_limit - 1800
    assert full == saved
    if token_limit == 8192:
        assert admission['stage'] >= 7
        page = admitted['payload']['last_tool_result']['sources_page']
        assert page['total'] == 21 and page['recall']['recall_steps'] == [1]
        assert page['next_offset'] == len(admitted['payload']['last_tool_result']['sources'])
        action = {**page['recall'], 'reason': 'Scripted next-page request.'}
    else:
        assert admission['stage'] < 7 and admission['input_tokens'] > 8192
        assert len(admitted['payload']['last_tool_result']['sources']) == 21
        action = {'action': 'stop', 'reason': 'Scripted larger-context verification.'}
    calls = []
    class Model:
        def generate(self, input_ids, **kwargs):
            calls.append(input_ids.shape[-1])
            reply = json.dumps(action)
            tokens = processor.tokenizer.encode(reply, add_special_tokens=False)
            return torch.cat([input_ids, torch.tensor([tokens], dtype=torch.long)], dim=1)
    holder.model = Model()
    result = director._multimodal_json(holder, admitted['system'], admitted['payload'], [],
        lambda v: research.ResearchActionEnvelope.model_validate(v).model_dump(mode='json'),
        total_token_limit=token_limit, allow_text_only=True, compact_json=True)
    assert calls == [admission['input_tokens']]
    assert result['output']['action'] == action['action']


def test_irreducible_panel_context_preserves_session_without_failure(tmp_path, monkeypatch, processor):
    from kusama import research_context
    def pause(full, config, **kwargs):
        return research_context.fit_context(full, lambda _: 9000)
    monkeypatch.setattr(research_context, 'admit_context', pause)
    def never_generate(*args):
        pytest.fail('Irreducible context must not dispatch generation')
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, never_generate,
        Transport('artic'), grounded=True)
    assert report['status'] == 'context_paused' and panel['status'] == 'partial'
    assert report['context_pause']['model_call_dispatched'] is False
    assert report['reasoning_calls'] == 0 and not calls and not renders


def test_context_pause_preserves_and_renders_earlier_committed_work(tmp_path, monkeypatch, processor):
    from kusama import research_context
    real_admit = research_context.admit_context
    def admit(full, config, **kwargs):
        if full['payload']['hypotheses']:
            return research_context.fit_context(full, lambda _: 9000)
        return real_admit(full, config, **kwargs)
    monkeypatch.setattr(research_context, 'admit_context', admit)
    canonical = None
    def choose(index, payload, images, output):
        nonlocal canonical
        if index == 0:
            return {'action': 'search', 'query': 'portrait', 'mode': 'remote', 'provider': 'artic', 'limit': 1}
        if index == 1:
            return {'action': 'acquire', 'candidate_id': payload['last_tool_result']['sources'][0]['candidate_id']}
        if canonical is None:
            canonical = payload['last_tool_result']['source_id']
        return {'action': 'draft' if index == 2 else 'commit', 'hypothesis': hypothesis('h1', [canonical])}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, Transport('artic'), directions=2)
    assert report['status'] == 'context_paused' and panel['status'] == 'partial'
    assert len(calls) == report['reasoning_calls'] == 4
    assert len(report['hypotheses']) == 1 and len(renders) == 2
    assert 'retained committed studies were rendered' in panel['progress_message']


@pytest.mark.parametrize('provider', ['met', 'artic', 'cleveland'])
@pytest.mark.parametrize('scenario', ['direct', 'acquired_stall', 'metadata_stall', 'http_recovery', 'invalid_acquire_arguments'])
def test_production_panel_paths(tmp_path, monkeypatch, processor, provider, scenario):
    transport = Transport(provider)
    candidate = None; canonical = None
    original = transport.__call__
    class FailingFirst:
        urls = transport.urls
        def __call__(self, req, timeout):
            if 'unavailable' in req.full_url:
                self.urls.append(req.full_url)
                raise urllib.error.HTTPError(req.full_url, 503, 'fixture unavailable', {}, io.BytesIO(b'failure'))
            return original(req, timeout)
    http = FailingFirst() if scenario == 'http_recovery' else transport
    shift = int(scenario == 'http_recovery')
    def choose(index, payload, images, output):
        nonlocal candidate, canonical
        if scenario == 'http_recovery' and index == 0:
            return {'action': 'search', 'mode': 'remote', 'provider': provider, 'query': 'unavailable', 'limit': 1}
        i = index - shift
        if i == 0:
            if shift:
                assert payload['last_tool_result']['tool_error']
            return {'action': 'search', 'mode': 'remote', 'provider': provider, 'query': 'portrait', 'limit': 1}
        if candidate is None:
            candidate = payload['last_tool_result']['sources'][0]['candidate_id']
        if scenario == 'metadata_stall' and 1 <= i <= 6:
            return {'action': 'note', 'note': 'Unresolved evidence; scripted inactivity'}
        acquire_at = 7 if scenario == 'metadata_stall' else 1
        if i == acquire_at:
            return {'action': 'acquire', 'candidate_id': candidate}
        if canonical is None:
            canonical = payload['last_tool_result']['source_id']
        assert images and images[0]['source_id'] == canonical
        if scenario == 'invalid_acquire_arguments':
            if i == 2:
                return {'action': 'acquire', 'source_id': canonical}
            if i == 3:
                assert 'acquire requires candidate_id' in payload['last_tool_result']['tool_error']
            return {'action': 'draft' if i == 3 else 'commit', 'hypothesis': hypothesis('h1', [canonical])}
        if scenario == 'acquired_stall' and 2 <= i <= 7:
            return {'action': 'search', 'mode': 'remote', 'provider': provider, 'query': 'portrait', 'limit': 1}
        draft_at = 8 if scenario in ('metadata_stall', 'acquired_stall') else 2
        if scenario != 'direct' and scenario != 'http_recovery':
            assert payload['research_progress']['phase'] == 'synthesis'
        return {'action': 'draft' if i == draft_at else 'commit', 'hypothesis': hypothesis('h1', [canonical])}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, http)
    assert report['status'] == 'hypotheses_ready' and len(report['hypotheses']) == 1
    assert len(calls) == (10 if 'stall' in scenario else 5 if shift or scenario == 'invalid_acquire_arguments' else 4)
    assert panel['status'] == 'completed' and len(renders) == 2
    assert report['network']['acquisitions'] == 1
    assert len(http.urls) == (3 if provider == 'met' else 2) + shift
    assert all(prompt == report['hypotheses'][0]['composition'] + '\n' + report['hypotheses'][0]['render_instruction'] for prompt, seed in renders)


@pytest.mark.parametrize('failure', ['missing_sources', 'truncated_json'])
def test_committed_direction_survives_later_rejected_draft(tmp_path, monkeypatch, processor, failure):
    transport = Transport('met')
    raw = (Path(__file__).parent / 'fixtures/research/missing_hypothesis_sources.json').read_text()
    selected = None
    def choose(index, payload, images, output):
        nonlocal selected
        if index == 0:
            return {'action': 'search', 'mode': 'remote', 'provider': 'met', 'query': 'portrait', 'limit': 1}
        if index == 1:
            return {'action': 'acquire', 'candidate_id': payload['last_tool_result']['sources'][0]['candidate_id']}
        if index in (2, 3):
            selected = images[0]['source_id']
            return {'action': 'draft' if index == 2 else 'commit', 'hypothesis': hypothesis('h1', [selected])}
        if index == 4:
            return raw if failure == 'missing_sources' else '{"action":"draft","hypothesis":'
        assert 'hypothesis.sources' in payload['last_tool_result']['tool_error']
        return {'action': 'stop', 'reason': 'Keep the committed direction; no second hypothesis.'}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, transport, directions=2)
    assert len(report['hypotheses']) == 1 and len(renders) == 2
    assert all(prompt == report['hypotheses'][0]['composition'] + '\n' + report['hypotheses'][0]['render_instruction'] for prompt, _ in renders)
    if failure == 'missing_sources':
        assert panel['status'] == 'completed' and report['steps'][4]['status'] == 'tool_rejected'
        assert calls[4]['raw'] == raw and len(calls) == 6
    else:
        assert panel['status'] == 'partial' and report['status'] == 'response_rejected'
        assert 'response_rejected' in panel['progress_message'] and len(calls) == 5


@pytest.mark.parametrize('raw', ['{"action":"invented"}', '{"action":"note","note":"' + 'word ' * 2000 + '"}'])
def test_real_decoder_rejects_invalid_and_output_truncated_json_without_retry(tmp_path, monkeypatch, processor, raw):
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor,
        lambda *a: raw, Transport('artic'))
    assert report['status'] == 'response_rejected' and len(calls) == 1
    assert panel['status'] == 'failed' and not renders
    assert report['steps'][0]['action'] is None and report['network']['requests'] == 0


@pytest.mark.skipif(not os.environ.get('KUSAMA_RECORDED_RESEARCH'), reason='Requires an explicitly selected local incident archive')
def test_recorded_incident_through_real_processors_and_source_backend(tmp_path, monkeypatch, processor):
    """Replay original choices/HTTP bytes; remap only timestamp-dependent candidate IDs.

    The two post-incident synthesis actions are assistant-authored fixtures. Passing
    this does not mean Qwen will choose them. Original run bytes remain unchanged.
    """
    original = Path(os.environ['KUSAMA_RECORDED_RESEARCH']).resolve()
    before = {p: sha256(p) for p in original.rglob('*') if p.is_file() and not p.name.endswith(('-wal', '-shm'))}
    original_report = research.read_report(original)
    assert original_report['status'] == 'stagnated' and original_report['reasoning_calls'] == 15
    source_root = original / 'sources'
    with research._read_db(source_root / 'research_sources.sqlite3') as db:
        requests = {url: json.loads(body) for url, body in db.execute('SELECT url,payload FROM requests')}
        candidates = {identifier: json.loads(body) for identifier, body in db.execute('SELECT id,payload FROM candidates')}
    class RecordedTransport:
        def __init__(self):
            self.urls = []
        def __call__(self, req, timeout):
            assert req.full_url in requests, 'Unexpected request must never escape to the network'
            self.urls.append(req.full_url)
            record = requests[req.full_url]
            raw = source_root / 'corpus' / record['path']
            assert sha256(raw) == record['sha256']
            return Response(raw.read_bytes(), record['content_type'])
    transport = RecordedTransport()
    def choose(index, payload, images, output):
        if index < 15:
            value = json.loads((original / f'response-{index:03}.json').read_text())['output']
            for key in ('source_id', 'candidate_id'):
                old = candidates.get(value.get(key))
                if old:
                    with research._read_db(output / 'research/sources/research_sources.sqlite3') as db:
                        matches = [identifier for identifier, raw in db.execute('SELECT id,payload FROM candidates')
                            if (row := json.loads(raw))['reference_id'] == old['reference_id'] and row['query'] == old['query']]
                    assert len(matches) == 1
                    value[key] = matches[0]
            return value
        assert payload['research_progress']['checkpoint_start'] == 15
        assert payload['research_progress']['phase'] == 'synthesis'
        assert images and images[0]['source_id'] == 'artic_16791'
        return {'action': 'draft' if index == 15 else 'commit', 'hypothesis': hypothesis('diagnostic', ['artic_16791'])}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, transport,
        question=original_report['problem'], context=json.loads((original / 'plan.json').read_text())['request']['prior_context'])
    assert report['status'] == 'hypotheses_ready' and len(calls) == 17
    assert panel['status'] == 'completed' and len(renders) == 2
    assert len(transport.urls) == 11 and report['network']['acquisitions'] == 1
    assert all(not row.get('tool_error') for row in report['steps'] if row['index'] not in (12, 13))
    assert before == {p: sha256(p) for p in before}


@pytest.mark.skipif(not os.environ.get('KUSAMA_RECORDED_HYPOTHESIS'), reason='Requires retained 13-call hypothesis incident')
def test_recorded_hypothesis_incident_complete_trace(tmp_path, monkeypatch, processor):
    original = Path(os.environ['KUSAMA_RECORDED_HYPOTHESIS']).resolve()
    original_report = research.read_report(original)
    assert original_report['status'] == 'response_rejected' and original_report['reasoning_calls'] == 13
    before = {p: sha256(p) for p in original.rglob('*.json')}
    source_root = original / 'sources'
    with research._read_db(source_root / 'research_sources.sqlite3') as db:
        requests = {url: json.loads(body) for url, body in db.execute('SELECT url,payload FROM requests')}
        candidates = {identifier: json.loads(body) for identifier, body in db.execute('SELECT id,payload FROM candidates')}
    class RecordedTransport:
        urls = []
        def __call__(self, req, timeout):
            assert req.full_url in requests
            self.urls.append(req.full_url)
            record = requests[req.full_url]
            path = source_root / 'corpus' / record['path']
            assert sha256(path) == record['sha256']
            return Response(path.read_bytes(), record['content_type'])
    def choose(index, payload, images, output):
        if index == 13:
            assert 'hypothesis.sources' in payload['last_tool_result']['tool_error']
            return {'action': 'stop', 'reason': 'Scripted terminal action after retained incident; not model evidence.'}
        raw = json.loads((original / f'response-{index:03}.json').read_text())['raw_response']
        value = json.loads(raw)
        old = candidates.get(value.get('candidate_id'))
        if old:
            with research._read_db(output / 'research/sources/research_sources.sqlite3') as db:
                matches = [identifier for identifier, body in db.execute('SELECT id,payload FROM candidates')
                           if (c := json.loads(body))['reference_id'] == old['reference_id'] and c['query'] == old['query']]
            assert len(matches) == 1
            value['candidate_id'] = matches[0]
            return value
        return raw
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, RecordedTransport(),
        directions=2, question=original_report['problem'], context='')
    assert panel['status'] == 'completed' and len(calls) == 14 and len(renders) == 2
    assert report['steps'][12]['status'] == 'tool_rejected'
    assert report['hypotheses'][0]['render_instruction'] == original_report['hypotheses'][0]['render_instruction']
    assert before == {p: sha256(p) for p in before}


@pytest.mark.skipif(not os.environ.get('KUSAMA_RECORDED_CONTEXT'), reason='Requires retained context-overflow incident')
@pytest.mark.parametrize('dense_metadata', [False, True])
def test_recorded_overflow_admits_without_changing_protected_inputs(processor, dense_metadata):
    from kusama.research_context import fit_context, project_context
    path = Path(os.environ['KUSAMA_RECORDED_CONTEXT'])
    source = json.loads(path.read_text())
    if dense_metadata:
        for entry in source['payload']['candidate_memory'] + source['payload']['last_tool_result']['sources']:
            entry['title'] = ('A long reference title with several distinct words and historical details ' * 4)[:200]
            entry['maker'] = ('An artist with a long documented attribution and geographic description ' * 4)[:200]
    before = sha256(path)
    config = json.loads((path.parent / 'plan.json').read_text())['director_config']
    config['cache_dir'] = Path(config['cache_dir'])
    holder = SimpleNamespace(config=director.DirectorConfig(**config), processor=processor, execution_settings={})
    def count(value):
        entries = [(im['source_id'] + ':' + im['scope'], Path(im['path'])) for im in value['images']]
        result, _ = director._prepare_multimodal_inputs(holder, value['system'], value['payload'], entries,
            allow_text_only=not entries, compact_json=True)
        return result['metadata']['input_tokens']
    # JSON key order in the preserved file may change token merges; both the
    # original live admission (6661) and this byte-preserved replay exceed 6392.
    assert count(project_context(source, 0)) > 6392
    projected, admission = fit_context(source, count)
    assert admission['input_tokens'] <= 6392 and admission['reserved_output_tokens'] == 1800
    assert admission['stage'] >= 1
    assert projected['images'] == source['images'] and projected['system'] == source['system']
    for key in ('problem', 'current_question', 'prior_context', 'pending_draft', 'hypotheses'):
        assert projected['payload'][key] == source['payload'][key]
    for key in ('candidate_memory',):
        assert {(r.get('source_id'), r.get('candidate_id')) for r in projected['payload'][key]} <= {
            (r.get('source_id'), r.get('candidate_id')) for r in source['payload'][key]}
    assert [(r.get('source_id'), r.get('candidate_id')) for r in projected['payload']['last_tool_result']['sources']] == [
        (r.get('source_id'), r.get('candidate_id')) for r in source['payload']['last_tool_result']['sources']]
    assert sha256(path) == before


@pytest.mark.parametrize('scenario', ['two_directions', 'oversized_prompt'])
def test_multiple_directions_and_render_prompt_redrafting(tmp_path, monkeypatch, processor, scenario):
    canonical = None
    def choose(index, payload, images, output):
        nonlocal canonical
        if index == 0:
            return {'action': 'search', 'query': 'portrait', 'mode': 'remote', 'provider': 'artic', 'limit': 1}
        if index == 1:
            return {'action': 'acquire', 'candidate_id': payload['last_tool_result']['sources'][0]['candidate_id']}
        if canonical is None:
            canonical = payload['last_tool_result']['source_id']
        if scenario == 'oversized_prompt':
            h = hypothesis('h1', [canonical])
            if index == 2:
                # Under the schema character cap and Qwen output cap, but above
                # the actual FLUX token limit. All three constraints matter.
                h['render_instruction'] = 'a ' * 600
            if index == 3:
                assert 'FLUX allows 512' in payload['last_tool_result']['tool_error']
            return {'action': 'draft' if index < 4 else 'commit', 'hypothesis': h}
        return {'action': 'draft' if index % 2 == 0 else 'commit',
                'hypothesis': hypothesis('h1' if index < 4 else 'h2', [canonical])}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, Transport('artic'),
        directions=2 if scenario == 'two_directions' else 1)
    assert panel['status'] == 'completed'
    assert len(renders) == (4 if scenario == 'two_directions' else 2)
    assert len(calls) == (6 if scenario == 'two_directions' else 5)
    assert all('a a a a a' not in prompt for prompt, seed in renders)


def test_six_image_pending_hypothesis_admits_with_real_processor(tmp_path, processor):
    from PIL import Image
    from kusama import research_context
    spec = research.plan_research(tmp_path / 'research', {
        'id': 'six-image-diagnostic', 'project_id': 'p', 'brief_id': 'b',
        'problem': 'How can forms of shelter also become exclusion?', 'authorship': 'operator software fixture',
        'resource_root': str(tmp_path / 'unadmitted'), 'max_sources_per_hypothesis': 6,
        'max_images_per_call': 6, 'cache_dir': str(CACHE), 'device': 'cpu'})
    images = []
    for index in range(6):
        path = tmp_path / f's{index}.png'
        Image.new('RGB', (512, 640), (index * 40, 100, 120)).save(path)
        images.append({'source_id': f's{index}', 'path': str(path), 'sha256': sha256(path)})
    state = research._state(spec, [])
    state['opened'] = {image['source_id']: {'source_id': image['source_id'], 'images': [image],
        'text': 'Source description. ' * 100, 'reference': {'title': 'Documented title', 'maker': 'Documented maker'}}
        for image in images}
    state['active'] = list(state['opened'])
    h = hypothesis('six_sources', state['active'])
    state['draft'] = {'step': 0, 'hypothesis': h}
    full = {'system': spec['policy'], 'payload': research._payload(spec, state, []), 'images': images}
    cfg = director.DirectorConfig(CACHE, device='cpu', max_new_tokens=1800, max_image_pixels=65536)
    admitted, count = research_context.admit_context(full, cfg)
    assert count['input_tokens'] <= 6392
    assert admitted['payload']['pending_draft'] == full['payload']['pending_draft']
    assert admitted['images'] == full['images']
    assert len(processor.tokenizer.encode(json.dumps({'action': 'commit', 'hypothesis': h}), add_special_tokens=False)) < 1800
    holder = SimpleNamespace(config=cfg, processor=processor, execution_settings={'status': 'processor_only'})
    actual, inputs = director._prepare_multimodal_inputs(holder, admitted['system'], admitted['payload'],
        [(image['source_id'], Path(image['path'])) for image in images], compact_json=True)
    assert actual['metadata']['input_tokens'] == count['input_tokens']
    assert len(inputs['image_grid_thw']) == 6
    assert not (tmp_path / 'unadmitted').exists()


def test_image_prompt_choice_through_cached_processor_panel_and_renderer(tmp_path, monkeypatch, processor):
    captured = []
    original_generate = Renderer.generate
    def generate(self, prompt, seed, parent_path=None, *, reference_images=None):
        assert parent_path is None and len(reference_images) == 1
        assert Path(reference_images[0]['path']).is_file()
        assert sha256(Path(reference_images[0]['path'])) == reference_images[0]['sha256']
        captured.append(reference_images)
        return original_generate(self, prompt, seed, parent_path)
    monkeypatch.setattr(Renderer, 'generate', generate)
    canonical = None
    def choose(index, payload, images, output):
        nonlocal canonical
        if index == 0:
            return {'action': 'search', 'mode': 'remote', 'provider': 'artic', 'query': 'portrait', 'limit': 1}
        if index == 1:
            return {'action': 'acquire', 'candidate_id': payload['last_tool_result']['sources'][0]['candidate_id']}
        canonical = images[0]['source_id']
        assert images[0]['image_index'] == 0
        assert payload['source_context'][0]['render_image_indices'] == [0]
        h = hypothesis('reference_shape', [canonical])
        h['render_images'] = [{'source_id': canonical, 'image_index': 0, 'purpose': 'Borrow the boundary contour'}]
        h['render_instruction'] = 'Paint a new opening using the boundary contour visible in image 1'
        return {'action': 'draft' if index == 2 else 'commit', 'hypothesis': h}
    report, calls, panel, renders = run_panel(tmp_path, monkeypatch, processor, choose, Transport('artic'))
    assert panel['status'] == 'completed' and len(captured) == len(renders) == 2
    assert report['hypotheses'][0]['render_images'][0]['source_id'] == canonical
    assert captured[0][0]['source_id'] == canonical
    assert all(c['input_tokens'] <= 6392 and c['output_tokens'] <= 1800 for c in calls)
