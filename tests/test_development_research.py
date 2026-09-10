"""Reference adoption, accounting and resume tests; substitutes only, no artistic claims."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from PIL import Image
import pytest

from kusama import autonomous as a
from kusama import research as r
from kusama import research_development as rd
from kusama.artifacts import sha256, write_json
from kusama.evidence import digest
from kusama.ledger import BudgetExceeded, SpecMismatch
from kusama.resources import SessionBudget
from test_autonomous import setup, Director, Backend


def enable(request, **updates):
    request['research'] = {'max_steps': 2, 'source_config': {'allow_network': False}, **updates}
    return request


def controller(monkeypatch, decision='keep'):
    original = a._multimodal_json
    decisions = []
    def model(director, system, payload, images, validator, **kwargs):
        if system != rd.RESEARCH_SYSTEM:
            return original(director, system, payload, images, validator, **kwargs)
        decisions.append((payload, images))
        value = {'decision': decision, 'reason': 'The actual boundary raises an unresolved relationship.',
                 'question': 'How can an enclosing boundary also expose vulnerability?' if decision == 'research' else ''}
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 50, 'generated_tokens': 20}}
    monkeypatch.setattr(a, '_multimodal_json', model)
    return decisions


def test_legacy_request_shape_and_limits_preserved(setup):
    _, output, request, _ = setup
    old = a.LoopRequest.model_validate(request).model_dump(mode='json')
    explicit_null = a.LoopRequest.model_validate({**request, 'research': None}).model_dump(mode='json')
    assert 'research' not in old and explicit_null == old
    spec = a.plan_loop(output, request)
    assert spec['request_sha256'] == digest(old)
    assert spec['version'] == a.VERSION
    assert spec['max_reasoning_calls'] == 6
    assert spec['max_reasoning_tokens'] == 6 * 8192
    assert 'research_inputs' not in spec and 'research_caps' not in spec
    assert set(spec['policy']) == {'proposal', 'observation', 'selection'}


def test_explicit_policy_expands_only_new_frozen_plans(setup, tmp_path):
    _, output, request, _ = setup
    sources = []
    for index in range(3):
        path = tmp_path / f'source-{index}.png'
        Image.new('RGB', (20, 20), 'blue').save(path)
        sources.append({'source_id': f'source_{index}', 'path': str(path), 'sha256': sha256(path)})
    request['references'] = sources
    with pytest.raises(ValueError, match='legacy'):
        a.LoopRequest.model_validate(request)
    enable(request, max_sources=3)
    spec = a.plan_loop(output, request)
    assert spec['version'] == a.RESEARCH_VERSION
    assert spec['max_reasoning_calls'] == 6 + 2 + 2 * 2
    assert spec['max_reasoning_tokens'] == spec['max_reasoning_calls'] * 8192
    assert spec['research_caps']['max_network_requests'] == 96
    assert spec['research_caps']['source_config_per_session']['allow_network'] is False
    assert len(spec['snapshots']) == 4
    assert 'research_controller' in spec['policy']
    assert a.preflight_loop(output) == spec


def test_invalid_source_config_fails_before_creating_plan(setup):
    _, output, request, _ = setup
    enable(request, source_config={'max_requests': -1})
    with pytest.raises(ValueError):
        a.plan_loop(output, request)
    assert not output.exists()


def test_keep_research_uses_actual_parent_and_prior_observations_without_dispatch(setup, monkeypatch):
    _, output, request, calls = setup
    enable(request)
    decisions = controller(monkeypatch)
    a.plan_loop(output, request)
    def forbidden(*args, **kwargs):
        raise AssertionError('Keep must not start a research session')
    report = a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=forbidden)
    assert report['reasoning_calls'] == 8 and report['research_reasoning_calls'] == 0
    assert len(decisions) == 2 and len(calls) == 6
    assert decisions[1][0]['parent_sha256'] == report['rounds'][0]['selected']['sha256']
    assert decisions[1][0]['prior_rounds'][0]['observations'] == report['rounds'][0]['observations']
    assert sha256(decisions[1][1][0][1]) == report['rounds'][1]['parent']['sha256']
    assert not (output / 'research').exists()
    assert report['human_labels'] == 0 and report['training_performed'] is False
    replay = a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=forbidden)
    assert replay['rounds'] == report['rounds'] and len(decisions) == 2 and len(calls) == 6


@pytest.fixture
def adopted(setup, monkeypatch, tmp_path):
    root, output, request, calls = setup
    enable(request)
    controller(monkeypatch, decision='research')
    source_path = tmp_path / 'new-source.png'
    Image.new('RGB', (40, 40), 'blue').save(source_path)
    sources = [{'source_id': 'new_source', 'path': str(source_path), 'sha256': sha256(source_path),
                'context': 'Attributed fixture context', 'provenance': {'author': 'fixture museum'}}]
    monkeypatch.setattr(r, 'selected_sources', lambda path: deepcopy(sources))
    sessions = []
    def runner(directory, *, director_factory, before_dispatch):
        before_dispatch()
        spec = json.loads((directory / 'plan.json').read_text())
        sessions.append(spec)
        with SessionBudget(root) as budget:
            keys = [f"research_{spec['request']['id']}:reasoning:{index}" for index in range(2)]
            for key in keys:
                assert budget.reserve(key, 'reasoning', max_tokens=8192)
                budget.settle(key, 'completed', actual_tokens=100)
            resource_calls = [row for row in budget.summary()['calls'] if row['key'] in keys]
        report = {'status': 'complete', 'reasoning_calls': 2, 'resource_calls': resource_calls,
            'hypotheses': [{'id': 'h1', 'question': 'Fixture inquiry', 'relationship': 'A proposed analogy',
                           'transformation': 'Fixture transformation', 'visible_test': 'Inspect the boundary',
                           'uncertainty': 'No artistic evaluation'}]}
        write_json(directory / 'report.json', report)
        return report
    a.plan_loop(output, request)
    report = a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
    return root, output, report, source_path, sessions, calls, runner


def test_adopted_sources_reopened_and_frozen_for_each_parent(adopted):
    root, output, report, original, sessions, calls, _ = adopted
    assert report['reasoning_calls'] == 12 and report['research_reasoning_calls'] == 4
    assert len(report['resource_calls']) == 12 + 4  # reasoning plus four render attempts
    assert len(sessions) == 2
    assert all(s['request']['resource_root'] == str(root) for s in sessions)
    assert sessions[1]['request']['parent']['sha256'] == report['rounds'][0]['selected']['sha256']
    assert sessions[0]['request']['problem'] == 'How can an enclosing boundary also expose vulnerability?'
    for row in report['rounds']:
        source = row['research']['sources'][0]
        assert source['path'] != str(original)
        assert source['sha256'] == source['reopened_sha256'] == sha256(Path(source['path']))
        assert source['provenance'] == {'author': 'fixture museum'}
        assert source['selection_origin'] == 'model_research'
    proposals = [payload for system, payload, _ in calls if system == a.PROPOSAL_SYSTEM]
    assert all(p['sources'][0]['source_id'] == 'new_source' for p in proposals)
    assert all(p['reference_research']['findings'][0]['relationship'] == 'A proposed analogy' for p in proposals)
    with SessionBudget(root) as budget:
        assert not budget.summary()['unsettled']


def test_replay_uses_saved_reopened_bytes_after_original_changes(adopted):
    _, output, report, original, sessions, calls, runner = adopted
    Image.new('RGB', (40, 40), 'black').save(original)
    count = len(calls)
    replay = a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
    assert replay['rounds'] == report['rounds']
    assert len(sessions) == 2 and len(calls) == count


def test_tampered_adopted_source_blocks_resume_before_dispatch(adopted):
    _, output, report, _, sessions, _, runner = adopted
    frozen = Path(report['rounds'][0]['research']['sources'][0]['path'])
    Image.new('RGB', (40, 40), 'black').save(frozen)
    with pytest.raises((ValueError, SpecMismatch)):
        a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
    assert len(sessions) == 2


def test_rejected_controller_charged_once_and_does_not_render(setup, monkeypatch):
    _, output, request, _ = setup
    enable(request)
    monkeypatch.setattr(a, '_multimodal_json', lambda *args, **kwargs: {
        'raw_response': '{}', 'output': None, 'parse_error': 'No research question'})
    a.plan_loop(output, request)
    report = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert report['status'] == 'research_controller_rejected'
    assert report['reasoning_calls'] == 1 and Backend.calls == 0
    a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert Director.loads == 1 and Backend.calls == 0


def test_actual_research_ledger_rejects_once_and_loop_accounts_shared_calls(setup, monkeypatch):
    root, output, request, _ = setup
    enable(request)
    controller(monkeypatch, decision='research')
    research_calls = []
    def rejected(*args, **kwargs):
        research_calls.append(True)
        return {'raw_response': '{}', 'output': None, 'parse_error': 'Fixture rejected research response'}
    monkeypatch.setattr(r, '_multimodal_json', rejected)
    a.plan_loop(output, request)
    report = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert report['status'] == 'research_no_selected_sources'
    assert report['reasoning_calls'] == 2 and report['research_reasoning_calls'] == 1
    assert report['rounds'][0]['research']['research_status'] == 'response_rejected'
    assert len(report['resource_calls']) == 2 and Backend.calls == 0
    a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert len(research_calls) == 1 and Director.loads == 2 and Backend.calls == 0
    with SessionBudget(root) as budget:
        assert len(budget.summary()['calls']) == 2


def test_research_honors_enclosing_wall_deadline(setup, monkeypatch):
    _, output, request, _ = setup
    enable(request)
    controller(monkeypatch, decision='research')
    a.plan_loop(output, request)
    def runner(directory, *, director_factory, before_dispatch):
        class ExpiredClock:
            @staticmethod
            def now(tz):
                return datetime.now(timezone.utc) + timedelta(days=2)
        monkeypatch.setattr(a, 'datetime', ExpiredClock)
        before_dispatch()
        pytest.fail('Research dispatch proceeded past development deadline')
    with pytest.raises(BudgetExceeded, match='wall-time'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
    assert Backend.calls == 0


@pytest.mark.parametrize('include_image', [False, True])
def test_selected_text_context_and_influence_survive_adoption_keep_and_resume(setup, monkeypatch, include_image):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    _, output, request, calls = setup
    enable(request, max_steps=6)
    original = a._multimodal_json
    controller_calls, research_calls = [], []
    def development(director, system, payload, images, validator, **kwargs):
        if system != rd.RESEARCH_SYSTEM:
            return original(director, system, payload, images, validator, **kwargs)
        value = {'decision': 'keep' if controller_calls else 'research',
            'reason': 'Investigate the historical meaning of a boundary.',
            'question': '' if controller_calls else 'How do accounts of enclosure complicate the visible boundary?'}
        controller_calls.append((payload, images))
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 50, 'generated_tokens': 20}}
    monkeypatch.setattr(a, '_multimodal_json', development)
    ids = ['text_source'] + (['image_source'] if include_image else [])
    chosen = {'id': 'h1', 'title': 'Contextual inquiry', 'question': 'What does this boundary imply?',
        'sources': [{'source_id': sid, 'role': 'Historical counterargument' if sid == 'text_source' else 'Visual comparison',
                     'transferred_property': 'Ambiguous protection', 'discarded_property': 'Original application'} for sid in ids],
        'relationship': 'A contextual analogy', 'transformation': 'Expose a contradiction in enclosure',
        'composition': 'An open-sided boundary', 'render_instruction': 'Paint an open-sided boundary.',
        'visible_test': 'Inspect the opening', 'uncertainty': 'Interpretation remains uncertain'}
    actions = [{'action': 'search', 'query': 'enclosure'}]
    actions += [{'action': 'open', 'source_id': sid} for sid in ids]
    actions += [{'action': action, 'hypothesis': chosen} for action in ('draft', 'commit')]
    def research_model(director, system, payload, images, validator, **kwargs):
        research_calls.append((payload, images))
        value = {'reason': 'Fixture research choice', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(r, '_multimodal_json', research_model)
    class ContextSources:
        def __init__(self, root, config, index_path=None):
            self.root = Path(root)
        def search(self, *args, **kwargs):
            return {'sources': [{'source_id': sid} for sid in ids]}
        def open(self, sid):
            if sid == 'text_source':
                path = self.root / 'historical-original.html'
                if not path.exists():
                    path.write_bytes(b'<p>Enclosure can express exclusion as well as protection.</p>')
                return {'source_id': sid, 'kind': 'context', 'images': [],
                    'text': 'Enclosure can express exclusion as well as protection.',
                    'title': 'Attributed historical account', 'url': 'https://example.org/account',
                    'provenance': {'raw_path': str(path), 'raw_sha256': sha256(path), 'author': 'Fixture source'},
                    'text_transform': 'Inert HTML converted to text', 'claims': [], 'links': []}
            path = self.root / 'reference.png'
            if not path.exists():
                Image.new('RGB', (40, 40), 'blue').save(path)
            return {'source_id': sid, 'images': [{'path': str(path), 'sha256': sha256(path)}],
                    'text': 'Fixture image context', 'claims': [], 'links': [], 'provenance': {},
                    'reference': {'maker': 'Fixture maker', 'date': '1900', 'medium': 'Ink'}}
        def close(self):
            pass
        def summary(self):
            return {'requests': 0, 'fixture': True}
    def runner(directory, *, director_factory, before_dispatch):
        return r.run_research(directory, director_factory=director_factory,
                              before_dispatch=before_dispatch, sources_factory=ContextSources)
    a.plan_loop(output, request)
    report = a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
    assert report['status'] == 'round_limit' and report['renderer_attempts'] == 4
    assert report['rounds'][0]['research']['status'] == 'researched'
    assert report['rounds'][1]['research']['status'] == 'keep'
    text = next(s for s in report['selected_sources'] if s.get('media_type') == 'text')
    assert len(report['selected_sources']) == len(ids)
    assert text['contribution']['role'] == 'Historical counterargument'
    assert Path(text['path']).read_text() == text['text']
    assert text['byte_evidence'] and sha256(Path(text['byte_evidence'][0]['path'])) == text['byte_evidence'][0]['sha256']
    proposals = [(payload, images) for system, payload, images in calls if system == a.PROPOSAL_SYSTEM]
    for payload, images in proposals:
        context = next(s for s in payload['sources'] if s['source_id'] == 'text_source')
        assert context['context'] == text['text']
        assert context['contribution']['role'] == 'Historical counterargument'
        assert all(label != 'text_source' for label, _ in images)
        if include_image:
            metadata = next(s for s in payload['sources'] if s['source_id'] == 'image_source')['documented_metadata']
            assert metadata['maker'] == 'Fixture maker' and metadata['date'] == '1900'
    assert all(label != 'text_source' for _, images in controller_calls for label, _ in images)
    before = len(research_calls), len(calls), len(controller_calls)
    Path(text['byte_evidence'][0]['original_path']).write_bytes(b'Original research raw file later changed')
    replay = a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
    assert replay['selected_sources'] == report['selected_sources']
    assert before == (len(research_calls), len(calls), len(controller_calls))
    Path(text['path']).write_text('Tampered adopted text')
    with pytest.raises((ValueError, SpecMismatch)):
        a.run_loop(output, director_factory=Director, backend_factory=Backend, research_runner=runner)
