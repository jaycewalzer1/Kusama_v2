"""Workflow failure regressions; substitute decisions are not artistic evidence."""
from copy import deepcopy

from kusama import research as r
from test_research import setup, action, hypothesis, Director, Sources, run


def test_candidate_evidence_survives_empty_search_and_false_notebook(setup):
    output, _, request, calls, actions = setup
    class Leads(Sources):
        def search(self, query, **kwargs):
            if query == 'empty':
                return {'sources': []}
            return {'sources': [{'source_id': 's0', 'candidate_id': 'c0', 'title': 'An overlooked source',
                'maker': 'Documented maker', 'description': 'Relevant source evidence', 'url': 'https://example.org/object/1'}]}
    actions.extend([action('search', query='lead'), action('search', query='empty'),
                    action('note', note='Unsupported model claim that no leads exist'),
                    action('open', source_id='s0'), action('draft', hypothesis=hypothesis()),
                    action('commit', hypothesis=hypothesis())])
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=Leads)
    retained = calls[3]['payload']['candidate_memory']
    assert retained[0]['source_id'] == 's0' and retained[0]['first_seen_step'] == 0
    assert retained[0]['description'] == 'Relevant source evidence'
    assert 'Unsupported' in calls[3]['payload']['notebook'][0]['note']
    assert calls[2]['payload']['recent_actions'][-1]['result_count'] == 0
    assert report['status'] == 'hypotheses_ready'


def test_repeated_remote_search_is_not_executed_again_and_can_recover(setup):
    output, _, request, calls, actions = setup
    searches = []
    class CountSources(Sources):
        def search(self, query, **kwargs):
            searches.append(query)
            return super().search(query, **kwargs)
    actions.extend([action('search', mode='remote', provider='artic', query='Portrait  collage'),
                    action('search', mode='remote', provider='artic', query='portrait collage'),
                    action('open', source_id='s0'), action('draft', hypothesis=hypothesis()),
                    action('commit', hypothesis=hypothesis())])
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=CountSources)
    assert searches == ['Portrait  collage']
    assert 'Repeated remote search' in report['steps'][1]['tool_error']
    assert 'Repeated remote search' in calls[2]['payload']['last_tool_result']['tool_error']
    assert report['reasoning_calls'] == 5 and report['status'] == 'hypotheses_ready'


def test_empty_search_loop_stops_before_spending_entire_budget_and_replays(setup):
    output, _, request, calls, actions = setup
    request['max_steps'] = 24
    class Empty(Sources):
        def search(self, *args, **kwargs):
            return {'sources': []}
    actions.extend(action('search', mode='remote', provider='artic', query=f'empty {i}') for i in range(6))
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=Empty)
    assert report['status'] == 'stagnated' and report['reasoning_calls'] == 6
    assert report['research_progress']['actions_without_progress'] == 6
    assert not report['hypotheses']
    r.run_research(output, director_factory=Director, sources_factory=Empty)
    assert len(calls) == 6


def test_same_object_with_new_candidate_id_does_not_reset_stagnation(setup):
    output, _, request, _, actions = setup
    request['max_steps'] = 24
    class SameObject(Sources):
        def search(self, query, **kwargs):
            return {'sources': [{'source_id': query, 'provider': 'artic', 'url': 'https://example.org/one'}]}
    actions.extend(action('search', query=f'candidate_{i}') for i in range(7))
    actions.extend(action('note', note='Still no new evidence') for _ in range(3))
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=SameObject)
    assert report['status'] == 'stagnated'
    assert report['research_progress']['candidate_count'] == 1


def test_last_three_steps_are_available_for_open_draft_commit(setup):
    _, _, request, calls, actions = setup
    request['max_steps'] = 4
    actions.extend([action('search', query='lead'), action('open', source_id='s0'),
                    action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    assert run(setup)['status'] == 'hypotheses_ready'
    assert 'search' not in calls[1]['payload']['research_progress']['allowed_actions']
    assert calls[3]['payload']['research_progress']['allowed_actions'] == ['stop', 'commit']


def test_final_search_does_not_dispatch_source_request(setup):
    output, _, request, _, actions = setup
    request['max_steps'] = 4
    searches = []
    class CountSources(Sources):
        def search(self, query, **kwargs):
            searches.append(query)
            return super().search(query, **kwargs)
    actions.extend([action('search', query='one'), action('search', query='two'), action('stop')])
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=CountSources)
    assert searches == ['one']
    assert 'reserved for synthesis' in report['steps'][1]['tool_error']
    assert not report['hypotheses']


def test_retention_is_bounded_and_never_changes_original_steps(setup):
    output, _, request, _, _ = setup
    spec = r.plan_research(output, request)
    steps = [{'index': i, 'action': {'action': 'search'}, 'tool_result': {'sources': [
        {'source_id': f's{i}', 'title': f'Source {i}', 'description': 'x' * 1000}]}}
        for i in range(20)]
    original = deepcopy(steps)
    progress, memory = r._research_progress(spec, r._state(spec, steps), steps)
    assert [s['source_id'] for s in memory] == [f's{i}' for i in list(range(8)) + list(range(16, 20))]
    assert progress['candidate_count'] == 20 and progress['retained_candidates'] == 12
    assert steps == original


def test_oversized_render_instruction_can_be_redrafted_before_commit(setup, monkeypatch):
    from kusama import backend
    _, _, request, _, actions = setup
    request['render_prompt_contract'] = {'model_revision': backend.MODEL_REVISION, 'max_sequence_length': 100}
    original = 'An oversized instruction'
    monkeypatch.setattr(backend, 'cached_prompt_token_count', lambda prompt, **kw: 150 if prompt == original else 30)
    actions.extend([action('search', query='lead'), action('open', source_id='s0'),
                    action('draft', hypothesis=hypothesis(render_instruction=original)),
                    action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    report = run(setup)
    assert 'FLUX allows 100' in report['steps'][2]['tool_error']
    assert report['steps'][2]['action']['hypothesis']['render_instruction'] == original
    assert report['hypotheses'][0]['render_instruction'] == hypothesis()['render_instruction']
    assert report['reasoning_calls'] == 5 and report['status'] == 'hypotheses_ready'
