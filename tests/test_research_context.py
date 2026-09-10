"""Prompt admission tests use substitutes; no inference or downloaded tokenizer."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from kusama import research as r
from kusama import research_context as context_tools
from kusama.backend import BackendError
from kusama.director import DirectorConfig, _multimodal_json, _prepare_multimodal_inputs
from test_research import setup, Director


def context():
    return {'system': 'Original system policy', 'images': [{'source_id': 's1', 'path': '/saved/image'}],
            'payload': {'problem': 'Original problem', 'current_question': 'Current question',
                'prior_context': 'Operator context', 'output_schema': r.ResearchAction.model_json_schema(),
                'pending_draft': {'hypothesis': {'title': 'Exact title', 'render_instruction': 'Exact instruction'}},
                'last_tool_result': {'sources': [{'source_id': 's1', 'candidate_id': 'c1',
                    'title': 'Source title', 'description': 'Evidence ' * 2000, 'metadata_sha256': 'abc'}]},
                'source_context': [], 'recent_actions': [{'action': 'note', 'note': 'history'}] * 5,
                'notebook': [{'action': 'note', 'note': 'notebook'}] * 4}}


def test_projection_preserves_candidate_ids_and_workflow_constraints():
    source = context()
    source['payload']['candidate_memory'] = [{'source_id': 's0', 'candidate_id': 'c0',
        'first_seen_step': 0, 'title': 'Documented title', 'maker': 'Documented maker', 'description': 'x' * 240}]
    source['payload']['research_progress'] = {'phase': 'synthesis', 'allowed_actions': ['draft', 'stop']}
    for stage in range(5):
        projected = context_tools.project_context(source, stage)['payload']
        entry = projected['candidate_memory'][0]
        assert entry['source_id'] == 's0' and entry['candidate_id'] == 'c0' and entry['first_seen_step'] == 0
        assert entry['title'] == 'Documented title' and entry['maker'] == 'Documented maker'
        assert projected['research_progress'] == source['payload']['research_progress']


def test_deduplicated_metadata_is_still_available_by_exact_candidate_id():
    original = context()
    hit = original['payload']['last_tool_result']['sources'][0]
    hit.update(maker='Original maker', url='https://collection.example/object/1', acquired=False)
    original['payload']['candidate_memory'] = [
        {'source_id': 's1', 'candidate_id': 'c1', 'title': hit['title'], 'maker': hit['maker'], 'first_seen_step': 2},
        {'source_id': 's2', 'candidate_id': 'c2', 'title': 'Other title', 'maker': 'Other maker'}]
    saved = deepcopy(original)
    projected = context_tools.project_context(original, 5)
    shared, separate = projected['payload']['candidate_memory']
    assert shared['source_id'] == 's1' and shared['candidate_id'] == 'c1' and shared['first_seen_step'] == 2
    assert shared['details_in'] and 'title' not in shared and 'maker' not in shared
    card = projected['payload']['last_tool_result']['sources'][0]
    assert card['title'] == hit['title'] and card['maker'] == hit['maker'] and card['acquired'] is False
    assert separate == saved['payload']['candidate_memory'][1]
    for key in ('problem', 'current_question', 'prior_context', 'pending_draft'):
        assert projected['payload'][key] == saved['payload'][key]
    assert projected['images'] == saved['images'] and original == saved


def test_compact_serialization_preserves_every_request_field(tmp_path):
    seen = []
    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            text = messages[1]['content'][-1]['text']
            seen.append(text)
            return text
        def __call__(self, **kwargs):
            assert kwargs['truncation'] is False
            return {'input_ids': torch.zeros((1, 10), dtype=torch.long)}
    holder = SimpleNamespace(config=DirectorConfig(tmp_path), processor=Processor(), execution_settings={})
    request = {'text': 'Keep spaces inside this text.', 'nested': {'a': 1, 'b': None}}
    _prepare_multimodal_inputs(holder, 'policy', request, [], allow_text_only=True, compact_json=True)
    assert json.loads(seen[0]) == request
    assert len(seen[0]) < len(json.dumps(request))


def test_projection_fits_without_mutating_protected_context():
    original = context()
    saved = deepcopy(original)
    projected, admission = context_tools.fit_context(original,
        lambda value: len(json.dumps(value, ensure_ascii=False)) // 2)
    assert admission['stage'] > 0 and admission['input_tokens'] <= 6392
    assert admission['reserved_output_tokens'] == 1800
    assert original == saved
    assert projected['system'] == original['system'] and projected['images'] == original['images']
    for key in ('problem', 'current_question', 'prior_context', 'pending_draft'):
        assert projected['payload'][key] == original['payload'][key]
    source = projected['payload']['last_tool_result']['sources'][0]
    assert source['source_id'] == 's1' and source['candidate_id'] == 'c1'
    assert projected['payload']['prompt_projection']['omitted_or_shortened_fields']
    # Schema annotation removal must not remove the hypothesis's actual title field.
    assert 'title' in projected['payload']['output_schema']['$defs']['ResearchHypothesis']['properties']
    assert projected == context_tools.project_context(original, admission['stage'])


def test_irreducible_context_fails_without_changing_budget():
    with pytest.raises(context_tools.ContextCapacityPause, match='6392.*1800.*8192'):
        context_tools.fit_context(context(), lambda _: 9000)


def test_candidate_paging_is_complete_stable_and_provider_interleaved():
    sources = [{'candidate_id': f'{provider}-{i}', 'provider': provider, 'title': 'Full title'}
               for provider in ('met', 'artic', 'cleveland') for i in range(8)]
    saved = deepcopy(sources)
    collected, offset = [], 0
    while offset is not None:
        page, metadata = context_tools.candidate_page(sources, step=3, offset=offset, limit=5)
        collected.extend(item['candidate_id'] for item in page)
        assert metadata['total'] == 24 and metadata['offset'] == offset
        offset = metadata['next_offset']
    assert collected[:3] == ['met-0', 'artic-0', 'cleveland-0']
    assert len(collected) == len(set(collected)) == 24
    assert set(collected) == {item['candidate_id'] for item in sources}
    assert sources == saved
    page, metadata = context_tools.candidate_page(sources, step=3, offset=8, limit=8)
    narrowed = context_tools._bounded_tool({'sources': page, 'sources_page': metadata, 'result_step': 3}, 2)
    assert narrowed['sources_page']['offset'] == 8
    assert narrowed['sources_page']['next_offset'] == 10
    assert narrowed['sources_page']['total'] == 24
    assert narrowed['sources_page']['recall']['offset'] == 10


def test_paged_projection_preserves_essential_inputs_and_schema_constraints():
    original = context()
    original['payload']['last_tool_result']['result_step'] = 2
    original['payload']['last_tool_result']['sources'] *= 24
    saved = deepcopy(original)
    for stage, size in [(7, 8), (8, 4), (9, 2), (10, 1)]:
        projected = context_tools.project_context(original, stage)
        payload = projected['payload']
        assert len(payload['last_tool_result']['sources']) == size
        assert payload['last_tool_result']['sources_page']['total'] == 24
        for key in ('problem', 'current_question', 'prior_context', 'pending_draft'):
            assert payload[key] == original['payload'][key]
        assert projected['images'] == original['images']
        schema = payload['output_schema']
        assert schema['allOf'] == context_tools._schema_without_annotations(original['payload']['output_schema'])['allOf']
        title = schema['$defs']['ResearchHypothesis']['properties']['title']
        assert title['minLength'] == 1 and title['maxLength'] == 160
        assert schema['required'] == original['payload']['output_schema']['required']
    assert original == saved


def test_paged_text_retains_correct_next_offset_and_link_ids():
    tool = {'text': 'x' * 2200, 'offset': 2200, 'next_offset': 4400,
            'links': [{'id': 'link1', 'title': 'Link', 'url': 'https://example.org'}]}
    result = context_tools._bounded_tool(tool, 2)
    assert result['next_offset'] == 3200 and result['text_excerpt'] is True
    assert result['links'][0]['id'] == 'link1'


def test_capacity_pause_retains_charged_history_and_replays_without_dispatch(setup, monkeypatch):
    from kusama.resources import SessionBudget
    from kusama.ledger import Ledger
    from test_research import Sources, action
    output, root, request, calls, actions = setup
    request['fixture'] = False
    actions.append(action('search', query='lead'))
    def admit(full, config, **kwargs):
        return context_tools.fit_context(full, lambda _: 100 if not calls else 9000)
    monkeypatch.setattr(context_tools, 'admit_context', admit)
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert report['status'] == 'context_paused' and report['reasoning_calls'] == 1
    assert len(calls) == len(report['steps']) == 1 and Director.loads == 1
    assert report['context_pause']['model_call_dispatched'] is False
    assert report['last_interruption'] is None
    assert (output / 'context-pause-001.json').exists()
    assert (output / 'decision-001-source.json').exists()
    assert not (output / 'response-001.json').exists()
    with SessionBudget(root) as budget:
        assert len(budget.summary()['calls']) == 1 and not budget.summary()['unsettled']
    # Simulate a crash after registering the pause but before updating job status.
    with Ledger(output / 'jobs.sqlite3') as ledger:
        ledger.update_job('research_' + request['id'], status='running', payload={})
    monkeypatch.setattr(context_tools, 'admit_context', lambda *a, **k: pytest.fail('Paused replay recounted tokens'))
    again = r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert again['status'] == 'context_paused' and again['context_pause'] == report['context_pause']
    assert len(calls) == Director.loads == 1


def test_context_failure_precedes_model_load_and_call_reservation(setup, monkeypatch):
    output, root, request, _, _ = setup
    request['fixture'] = False
    r.plan_research(output, request)
    def failure(*args, **kwargs):
        raise ValueError('Context cannot fit; no model call was reserved or dispatched')
    monkeypatch.setattr(context_tools, 'admit_context', failure)
    with pytest.raises(ValueError, match='Context cannot fit'):
        r.run_research(output, director_factory=Director)
    assert Director.loads == 0
    report = r.read_report(output)
    assert report['reasoning_calls'] == 0 and report['resource_calls'] == []
    assert (output / 'decision-000-source.json').is_file()
    assert not (output / 'response-000.json').exists()


def test_actual_generation_receives_frozen_admitted_context_and_replay_does_not_recount(setup, monkeypatch):
    output, _, request, _, _ = setup
    request['fixture'] = False
    admissions, calls = [], []
    def admit(full, config, **kw):
        admissions.append(full)
        return context_tools.fit_context(full, lambda _: 100)
    def model(director, system, payload, images, validator, **kwargs):
        calls.append(payload)
        frozen = json.loads((output / 'decision-000.json').read_text())
        assert frozen['payload'] == payload and frozen['context_admission']['input_tokens'] == 100
        assert kwargs['compact_json'] is True and kwargs['total_token_limit'] == 8192
        return {'output': validator({'action': 'stop'}), 'raw_response': '{"action":"stop"}',
                'metadata': {'input_tokens': 100, 'generated_tokens': 10}}
    monkeypatch.setattr(context_tools, 'admit_context', admit)
    monkeypatch.setattr(r, '_multimodal_json', model)
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director)
    assert report['status'] == 'model_stop'
    r.run_research(output, director_factory=Director)
    assert len(admissions) == len(calls) == 1


def test_token_error_reports_actual_input_ceiling_and_never_generates(tmp_path):
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return 'fixture prompt'
        def __call__(self, **kwargs):
            return {'input_ids': torch.zeros((1, 6811), dtype=torch.long)}
    def forbidden(**kwargs):
        pytest.fail('Over-budget input must never reach generation')
    holder = SimpleNamespace(config=DirectorConfig(tmp_path, max_new_tokens=1800),
        model=SimpleNamespace(generate=forbidden), processor=Processor(),
        execution_settings={'status': 'ready', 'device': 'cpu'})
    with pytest.raises(BackendError, match='6811.*6392.*1800.*8192') as exc:
        _multimodal_json(holder, 'policy', {}, [], lambda x: x, total_token_limit=8192, allow_text_only=True)
    assert exc.value.result['metadata']['input_token_limit'] == 6392


def test_rich_pages_keep_full_evidence_and_recall_offsets():
    original = context()
    sources = [{'candidate_id': f'c{i}', 'source_id': f'c{i}', 'provider': 'museum',
                'title': f'Title {i}', 'description': 'Context. ' * 100 + 'Decisive closing evidence.'}
               for i in range(12)]
    original['payload'].update(context_projection_policy='rich_candidate_pages_v1',
        last_tool_result={'sources': sources, 'result_step': 3}, candidate_memory=deepcopy(sources))
    saved = deepcopy(original)
    projected = context_tools.project_context(original, 13)['payload']
    assert projected['last_tool_result']['sources'] == sources[:4]
    assert projected['last_tool_result']['sources_page']['recall'] == {
        'action': 'recall', 'recall_steps': [3], 'offset': 4, 'limit': 4}
    assert projected['candidate_memory'][0]['details_in'].startswith('last_tool_result')
    assert projected['candidate_memory'][4]['description'] == sources[4]['description']
    assert projected['candidate_memory'][8]['details_in'].startswith('Saved candidate')
    for key in ('problem', 'current_question', 'prior_context', 'pending_draft'):
        assert projected[key] == original['payload'][key]
    assert original == saved
    admitted, admission = context_tools.fit_context(original,
        lambda ctx: 99999 if 'prompt_projection' not in ctx['payload'] else 100)
    assert admission['stage'] == 11
    assert admitted['payload']['last_tool_result']['sources'][0]['description'] == sources[0]['description']
    recalled = context_tools._rich_tool({'sources': sources[4:8],
        'sources_page': {'total': 12, 'offset': 4}, 'result_step': 3}, 2)
    assert recalled['sources'] == sources[4:6]
    assert recalled['sources_page']['next_offset'] == 6


def test_full_description_compaction_is_explicit_and_propagates_to_recall():
    source = {'candidate_id': 'c1', 'description': 'evidence ' * 200}
    tool = {'sources': [source], 'recalled': [{'index': 3, 'tool_result': {'sources': [source]}}]}
    assert len(r._compact_tool(tool)['sources'][0]['description']) == 500
    rich = r._compact_tool(tool, description_limit=None)
    assert rich['sources'][0]['description'] == source['description']
    assert rich['recalled'][0]['tool_result']['sources'][0]['description'] == source['description']
