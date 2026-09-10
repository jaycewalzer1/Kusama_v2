"""Substitute research decisions; these tests establish no artistic ability."""
import json
from pathlib import Path

from PIL import Image
import pytest
from pydantic import ValidationError

from kusama import research as r
from kusama.artifacts import sha256
from kusama.resources import SessionBudget, load_config


class Director:
    loads = 0
    def __init__(self, config):
        self.config = config
    def prepare(self):
        type(self).loads += 1
    def close(self):
        pass


class Sources:
    opened = []
    def __init__(self, root, config, index_path=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def search(self, query, mode='text', limit=8, **kwargs):
        return {'sources': [{'source_id': f's{i}', 'title': f'Source {i}'} for i in range(4)],
                'query': query, 'mode': mode, 'policy': 'fixture'}
    def open(self, source_id):
        self.opened.append(source_id)
        path = self.root / f'{source_id}.png'
        if not path.exists():
            Image.new('RGB', (64, 64), 'red').save(path)
        return {'source_id': source_id, 'images': [{'path': str(path), 'sha256': sha256(path)}],
                'text': 'Documented fixture context; not artistic evidence', 'claims': [],
                'links': [], 'provenance': {'origin': 'fixture'},
                'reference': {'rights': {'image_display': {'status': 'allowed', 'license': 'fixture'}}}}
    def close(self):
        pass
    def summary(self):
        return {'fixture': True, 'requests': 0}


def hypothesis(ids=('s0',), **updates):
    return {'id': 'h1', 'title': 'A research experiment', 'question': 'An open question',
        'sources': [{'source_id': i, 'role': 'Test a relation', 'transferred_property': 'boundary',
                     'discarded_property': 'original subject'} for i in ids],
        'relationship': 'One form shelters another', 'transformation': 'Reverse the enclosure',
        'composition': 'A central enclosure with an opening', 'render_instruction': 'Paint an enclosure with an opening.',
        'visible_test': 'Compare the boundary', 'uncertainty': 'Viewer reading unknown', **updates}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    output, root = tmp_path / 'research', tmp_path / 'session'
    cfg = load_config().model_copy(update={'session_hours': 24, 'handoff_reserve_minutes': 0,
                                         'device': 'cpu', 'cache_dir': str(tmp_path / 'cache')})
    with SessionBudget(root, cfg):
        pass
    request = {'id': 'fixture', 'project_id': 'p', 'brief_id': 'b', 'problem': 'An open artistic question',
               'authorship': 'fixture operator', 'resource_root': str(root), 'device': 'cpu', 'precision': 'fp32',
               'max_steps': 12, 'max_hypotheses': 1, 'fixture': True, 'total_token_limit': 8192}
    calls, actions = [], []
    def model(director, system, payload, images, validator, **kwargs):
        index = len(calls)
        calls.append({'payload': payload, 'images': images, 'kwargs': kwargs})
        value = actions[index]
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(r, '_multimodal_json', model)
    Sources.opened = []; Director.loads = 0
    return output, root, request, calls, actions


def action(kind, **kwargs):
    return {'action': kind, 'reason': 'Fixture decision', **kwargs}


def run(setup):
    output, _, request, _, _ = setup
    r.plan_research(output, request)
    return r.run_research(output, director_factory=Director, sources_factory=Sources)


@pytest.mark.parametrize('ids', [('s0',), ('s0', 's1', 's2')])
def test_model_selects_one_or_three_sources_reopens_and_replays_without_calls(setup, ids):
    output, root, request, calls, actions = setup
    actions.append(action('search', query='enclosure'))
    actions.extend(action('open', source_id=i) for i in ids)
    actions.extend([action('draft', hypothesis=hypothesis(ids)), action('commit', hypothesis=hypothesis(ids))])
    report = run(setup)
    assert report['status'] == 'hypotheses_ready'
    assert len(report['hypotheses'][0]['sources']) == len(ids)
    assert [i[0] for i in calls[-1]['images']] == [f'{sid} [image_index=0]' for sid in ids]
    assert len(Sources.opened) == len(ids) * 2
    assert calls[0]['kwargs']['allow_text_only'] is True
    assert report['reasoning_calls'] == len(actions)
    assert len(r.selected_sources(output)) == len(ids)
    before = len(calls)
    again = r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert again['hypotheses'] == report['hypotheses'] and len(calls) == before
    assert len(again['resource_calls']) == before
    assert report['human_labels'] == 0 and not report['training_performed']
    assert r.read_report(output)['evidence_images']


def test_question_revision_rejected_leads_and_recall_are_attributed(setup):
    _, _, request, calls, actions = setup
    actions.extend([action('search', query='support'),
                    action('reject', source_id='s0', note='Too literal'),
                    action('revise_question', question='Can support become restraint?'),
                    action('recall', recall_steps=[0, 1]), action('stop')])
    report = run(setup)
    assert report['problem'] == request['problem']
    assert report['question'] == 'Can support become restraint?'
    assert [n['action'] for n in report['notebook']] == ['reject', 'revise_question']
    assert calls[-1]['payload']['last_tool_result']['recalled'][1]['action']['note'] == 'Too literal'


@pytest.mark.parametrize('kind', ['note', 'reject'])
@pytest.mark.parametrize('fields', [
    {'reason': 'These metadata results do not address the question.'},
    {'note': 'These metadata results do not address the question.'},
    {'reason': 'No relevant evidence yet.', 'note': 'Try a different query.'},
])
def test_notebook_accepts_model_text_in_either_field_without_rewriting(setup, kind, fields):
    output, _, _, calls, actions = setup
    supplied = {'action': kind, **fields}
    actions.extend([supplied, {'action': 'recall', 'recall_steps': [0]}, {'action': 'stop'}])
    report = run(setup)
    assert report['status'] == 'model_stop' and report['reasoning_calls'] == 3
    entry = report['notebook'][0]
    assert entry['reason'] == fields.get('reason')
    assert entry['note'] == fields.get('note', '')
    assert entry['origin'] == 'fixture_actor'
    assert calls[1]['payload']['notebook'][0].get('reason') == fields.get('reason')
    recalled = calls[2]['payload']['last_tool_result']['recalled'][0]['action']
    for field, text in fields.items():
        assert recalled[field] == text
    raw = json.loads((output / 'response-000.json').read_text())['raw_response']
    assert json.loads(raw) == supplied
    assert not report['hypotheses'] and report['human_labels'] == 0
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert len(calls) == 3


@pytest.mark.parametrize('kind', ['note', 'reject'])
@pytest.mark.parametrize('fields', [{}, {'note': '   '}, {'reason': None, 'note': ''}])
def test_empty_notebook_actions_are_not_invented(kind, fields):
    with pytest.raises(ValidationError, match='requires note or reason'):
        r.ResearchAction.model_validate({'action': kind, **fields})


def test_frozen_schema_discloses_conditional_action_requirements(setup):
    output, _, request, _, _ = setup
    rules = r.plan_research(output, request)['schema']['allOf']
    for kind, fields in r.ACTION_REQUIRED_FIELDS.items():
        rule = next(rule for rule in rules if rule['if']['properties']['action'].get('const') == kind
                    and 'mode' not in rule['if']['properties'])
        assert rule['then']['required'] == list(fields)
    notebook = next(rule for rule in rules if 'enum' in rule['if']['properties']['action'])
    assert notebook['if']['properties']['action']['enum'] == ['note', 'reject']
    assert [branch['required'] for branch in notebook['then']['anyOf']] == [['note'], ['reason']]


def test_unseen_source_and_commit_without_reinspection_fail_as_tool_results(setup):
    _, _, _, calls, actions = setup
    actions.extend([action('open', source_id='invented'), action('search', query='enclosure'),
                    action('open', source_id='s0'), action('commit', hypothesis=hypothesis()), action('stop')])
    report = run(setup)
    assert 'encountered' in report['steps'][0]['tool_error']
    assert 'reinspection' in report['steps'][3]['tool_error']
    assert report['hypotheses'] == [] and len(calls) == 5


def test_source_set_change_requires_fresh_draft(setup):
    _, _, _, _, actions = setup
    actions.extend([action('search', query='enclosure'), action('open', source_id='s0'),
                    action('open', source_id='s1'), action('draft', hypothesis=hypothesis()),
                    action('commit', hypothesis=hypothesis(('s0', 's1'))), action('stop')])
    report = run(setup)
    assert 'Changed source set' in report['steps'][4]['tool_error']
    assert not report['hypotheses']


def test_failed_response_charged_once_and_no_retry(setup, monkeypatch):
    output, _, _, calls, _ = setup
    monkeypatch.setattr(r, '_multimodal_json', lambda *a, **k: {'raw_response': '{}', 'parse_error': 'bad output'})
    report = run(setup)
    assert report['status'] == 'response_rejected' and report['reasoning_calls'] == 1
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert Director.loads == 1


def test_missing_reason_does_not_block_research_or_fabricate_rationale(setup):
    output, _, _, calls, actions = setup
    actions.extend([
        {'action': 'search', 'query': 'enclosure'},
        {'action': 'open', 'source_id': 's0'},
        {'action': 'draft', 'hypothesis': hypothesis()},
        {'action': 'commit', 'hypothesis': hypothesis()},
    ])
    report = run(setup)
    assert report['status'] == 'hypotheses_ready' and len(report['hypotheses']) == 1
    assert all(step['action']['reason'] is None for step in report['steps'])
    response = json.loads((output / 'response-000.json').read_text())
    assert 'reason' not in json.loads(response['raw_response'])
    assert 'reason' not in calls[0]['payload']['output_schema']['required']
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert len(calls) == 4


def test_missing_reason_still_preserves_provider_mode_rejection(setup):
    output, _, request, calls, actions = setup
    actions.extend([
        {'action': 'search', 'mode': 'historical', 'query': 'agency', 'limit': 8, 'provider': 'met'},
        {'action': 'stop'},
    ])
    r.plan_research(output, request)
    # The real source backend rejects this combination before any network call.
    report = r.run_research(output, director_factory=Director)
    assert report['status'] == 'model_stop' and len(calls) == 2
    assert report['steps'][0]['status'] == 'tool_rejected'
    assert 'Provider is only valid for remote search' in report['steps'][0]['tool_error']
    assert report['steps'][0]['action']['reason'] is None
    assert calls[1]['payload']['recent_actions'][0]['tool_error']
    recent = calls[1]['payload']['recent_actions'][0]['action']
    assert recent['mode'] == 'historical' and recent['provider'] == 'met' and recent['limit'] == 8
    assert report['network']['requests'] == 0


@pytest.mark.parametrize('value', [
    {'action': 'search'}, {'action': 'open'}, {'action': 'acquire'},
    {'action': 'follow', 'source_id': 's0'}, {'action': 'draft'}, {'action': 'commit'},
    {'action': 'note'}, {'action': 'reject'}, {'action': 'revise_question'},
    {'action': 'recall'}, {'action': 'search', 'query': 'agency', 'mode': 'remote'},
])
def test_optional_reason_keeps_operational_fields_required(value):
    with pytest.raises(ValidationError):
        r.ResearchAction.model_validate(value)


@pytest.mark.parametrize('fields, expected', [({}, 8), ({'limit': 8}, 8), ({'limit': 3}, 3)])
def test_search_limit_reaches_backend_and_frozen_evidence(setup, fields, expected):
    output, _, request, calls, actions = setup
    received = []
    class SearchSources(Sources):
        def search(self, query, mode='text', limit=8, **kwargs):
            received.append(limit)
            return super().search(query, mode, limit, **kwargs)
    actions.extend([action('search', query='enclosure', **fields), action('stop')])
    plan = r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=SearchSources)
    assert report['status'] == 'model_stop'
    assert received == [expected]
    assert report['steps'][0]['action']['limit'] == expected
    assert plan['schema']['properties']['limit']['maximum'] == 8
    assert calls[0]['payload']['search_capabilities'] == {
        'network_enabled': False, 'index_available': False,
        'clip_text_enabled': False, 'provider_mode': 'remote only', 'max_results': 8}
    r.run_research(output, director_factory=Director, sources_factory=SearchSources)
    assert len(calls) == 2 and received == [expected]


@pytest.mark.parametrize('fields', [
    {'limit': 0}, {'limit': 9}, {'limit': True}, {'limit': 2.5}, {'limit': '8'},
    {'limit': None}, {'invented_search_option': 8},
])
def test_invalid_search_fields_remain_charged_rejections_without_retry(setup, monkeypatch, fields):
    output, _, _, _, _ = setup
    raw = json.dumps(action('search', query='enclosure', **fields))
    def model(director, system, payload, images, validator, **kwargs):
        try:
            validator(json.loads(raw))
        except ValidationError as exc:
            return {'raw_response': raw, 'parse_error': str(exc)}
        pytest.fail('Invalid search response was accepted')
    monkeypatch.setattr(r, '_multimodal_json', model)
    report = run(setup)
    assert report['status'] == 'response_rejected' and report['reasoning_calls'] == 1
    assert report['steps'][0]['action'] is None and report['hypotheses'] == []
    response = output / 'response-000.json'
    original = response.read_bytes()
    assert json.loads(original)['raw_response'] == raw
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert Director.loads == 1 and response.read_bytes() == original


def test_missing_resource_session_never_creates_one(setup, tmp_path):
    output, _, request, _, _ = setup
    request['resource_root'] = str(tmp_path / 'not_admitted')
    r.plan_research(output, request)
    assert not Path(request['resource_root']).exists()
    with pytest.raises(ValueError, match='admitted'):
        r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert not Path(request['resource_root']).exists()


def test_fixture_never_runs_real_model(setup):
    output, _, request, _, _ = setup
    r.plan_research(output, request)
    with pytest.raises(ValueError, match='Fixture'):
        r.run_research(output)


def test_stolen_or_changed_evidence_cannot_replay(setup):
    output, _, _, _, actions = setup
    actions.extend([action('search', query='enclosure'), action('open', source_id='s0'),
                    action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    run(setup)
    source = r.selected_sources(output)[0]
    Image.new('RGB', (64, 64), 'blue').save(source['path'])
    with pytest.raises(Exception, match='changed'):
        r.read_report(output)
    with pytest.raises(Exception, match='changed'):
        r.run_research(output, director_factory=Director, sources_factory=Sources)


def test_terminal_stop_and_step_caps(setup):
    _, _, request, _, actions = setup
    request['max_steps'] = 2
    actions.extend([action('note', note='An unresolved possibility'), action('note', note='More questions')])
    report = run(setup)
    assert report['status'] == 'step_limit' and report['reasoning_calls'] == 2
    assert not report['hypotheses']


def test_enclosing_deadline_callback_runs_before_model_load(setup):
    output, _, request, _, _ = setup
    r.plan_research(output, request)
    def expired():
        raise r.BudgetExceeded('Enclosing loop deadline')
    with pytest.raises(r.BudgetExceeded):
        r.run_research(output, director_factory=Director, sources_factory=Sources, before_dispatch=expired)
    assert Director.loads == 0
    assert r.read_report(output)['reasoning_calls'] == 0


def test_uncertain_charged_call_does_not_dispatch_again(setup):
    output, root, request, _, _ = setup
    spec = r.plan_research(output, request)
    job_id = 'research_' + request['id']
    with r.Ledger(output / 'jobs.sqlite3') as ledger, SessionBudget(root) as budget:
        ledger.create_job(job_id, spec, request['max_steps'], output)
        budget.reserve(job_id + ':reasoning:0', 'reasoning', max_tokens=8192)
        ledger.open_attempt(job_id, 0)
        ledger.reserve_call(job_id, 0)
    with pytest.raises(RuntimeError, match='Uncertain'):
        r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert Director.loads == 0


def test_no_write_on_read_and_parent_hash_checked(setup, tmp_path):
    output, _, request, _, _ = setup
    parent = tmp_path / 'parent.png'; Image.new('RGB', (64, 64), 'white').save(parent)
    request['parent'] = {'path': str(parent), 'sha256': sha256(parent)}
    spec = r.plan_research(output, request)
    assert r.read_report(output)['status'] == 'planned'
    assert not (output / 'jobs.sqlite3').exists()
    (output / spec['snapshots']['parent']['path']).write_bytes(b'bad')
    with pytest.raises(Exception, match='changed'):
        r.read_report(output)


def test_later_passage_and_links_can_be_read_without_network(setup):
    output, _, request, calls, actions = setup
    class LongSources(Sources):
        def open(self, sid):
            value = super().open(sid)
            value['text'] = 'a' * 3000 + 'LATER DOCUMENTED PASSAGE'
            value['links'] = [{'link_id': f'l{i}', 'url': f'https://example.org/{i}'} for i in range(30)]
            return value
    actions.extend([action('search', query='context'), action('open', source_id='s0'),
                    action('read', source_id='s0', offset=3000, link_offset=12), action('stop')])
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=LongSources)
    assert report['steps'][2]['tool_result']['text'] == 'LATER DOCUMENTED PASSAGE'
    assert calls[-1]['payload']['last_tool_result']['links'][0]['link_id'] == 'l12'


def test_oversized_detail_open_remains_recoverable(setup):
    output, _, request, _, actions = setup
    request.update(max_images_per_call=1, max_sources_per_hypothesis=1)
    class DetailSources(Sources):
        def open(self, sid):
            value = super().open(sid)
            value['images'] *= 2
            return value
    actions.extend([action('search', query='detail'), action('open', source_id='s0'), action('stop')])
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=DetailSources)
    assert 'image budget' in report['steps'][1]['tool_error']
    assert report['status'] == 'model_stop'


def test_unseen_acquisition_not_dispatched_to_backend(setup):
    _, _, _, _, actions = setup
    actions.extend([action('acquire', candidate_id='unseen'), action('stop')])
    report = run(setup)
    assert 'encountered candidate' in report['steps'][0]['tool_error']


def test_report_resources_match_literal_run_id(setup):
    output, root, request, _, actions = setup
    request['id'] = 'a_b'
    actions.append(action('stop'))
    report = run(setup)
    with SessionBudget(root) as budget:
        budget.reserve('research_axb:reasoning:0', 'reasoning', max_tokens=8192)
        budget.settle('research_axb:reasoning:0', 'failed')
    assert len(r.read_report(output)['resource_calls']) == 1


def test_explicit_text_only_director_keeps_visual_guard(tmp_path):
    from types import SimpleNamespace
    import torch
    from kusama.director import _multimodal_json, DirectorConfig
    from kusama.backend import BackendError
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return 'fixture prompt'
        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return Batch(input_ids=torch.tensor([[1, 2]]))
        def batch_decode(self, *args, **kwargs):
            return ['{"action":"stop"}']
    processor = Processor()
    director = SimpleNamespace(model=SimpleNamespace(generate=lambda **kwargs: torch.tensor([[1, 2, 3]])),
        processor=processor, config=DirectorConfig(tmp_path, max_new_tokens=100),
        execution_settings={'status': 'ready', 'device': 'cpu'})
    with pytest.raises(BackendError, match='requires saved images'):
        _multimodal_json(director, 'system', {}, [], lambda x: x)
    result = _multimodal_json(director, 'system', {}, [], lambda x: x, allow_text_only=True)
    assert result['metadata']['input_mode'] == 'explicit_text_only_research'
    assert 'images' not in processor.kwargs
    image = tmp_path / 'source.png'; Image.new('RGB', (64, 64), 'white').save(image)
    with pytest.raises(BackendError, match='image tensors'):
        _multimodal_json(director, 'system', {}, [('source', image)], lambda x: x, allow_text_only=True)
