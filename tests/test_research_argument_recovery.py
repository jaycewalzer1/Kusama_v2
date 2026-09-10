"""The real ten-call failure exposed argument validation at the wrong boundary."""
import json
from pathlib import Path

import pytest

from kusama import research as r
from test_research import setup, action, hypothesis, run, Director, Sources


@pytest.mark.parametrize('invalid', [
    {'action': 'acquire', 'source_id': 'met_268339'},  # exact live response
    {'action': 'search'}, {'action': 'open'}, {'action': 'follow', 'source_id': 's0'},
    {'action': 'read'}, {'action': 'draft'}, {'action': 'commit'},
    {'action': 'note'}, {'action': 'revise_question'}, {'action': 'recall'},
    {'action': 'search', 'mode': 'remote', 'query': 'portrait'},
])
def test_missing_tool_arguments_are_visible_and_recover_within_original_budget(setup, invalid):
    output, _, request, calls, actions = setup
    request['max_steps'] = 6
    actions.extend([action('search', query='lead'), action('open', source_id='s0'), invalid,
                    action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    report = run(setup)
    assert report['status'] == 'hypotheses_ready' and report['reasoning_calls'] == 5
    step = report['steps'][2]
    assert step['status'] == 'tool_rejected' and 'requires' in step['tool_error']
    assert calls[3]['payload']['last_tool_result']['tool_error'] == step['tool_error']
    assert json.loads(json.loads((output / 'response-002.json').read_text())['raw_response']) == invalid
    assert step['action']['candidate_id'] is None  # no invented operational input
    count = len(calls)
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert len(calls) == count


def test_invalid_argument_loop_uses_existing_stagnation_cap(setup):
    _, _, request, calls, actions = setup
    request['max_steps'] = 24
    actions.extend({'action': 'acquire', 'source_id': 'met_268339'} for _ in range(6))
    report = run(setup)
    assert report['status'] == 'stagnated' and len(calls) == 6
    assert not report['hypotheses'] and report['network']['requests'] == 0
    assert all(step['status'] == 'tool_rejected' for step in report['steps'])


INVALID_HYPOTHESES = [
    {k: v for k, v in hypothesis().items() if k != field}
    for field, definition in r.ResearchHypothesis.model_fields.items() if definition.is_required()
] + [hypothesis(sources=[]), hypothesis(sources='s0'),
     hypothesis(sources=[{'source_id': 's0'}]), hypothesis(ids=('s0', 's0')),
     hypothesis(render_instruction='x' * 1801), hypothesis(invented_field='untrusted'),
     json.loads((Path(__file__).parent / 'fixtures/research/missing_hypothesis_sources.json').read_text())['hypothesis']]


@pytest.mark.parametrize('invalid', INVALID_HYPOTHESES)
def test_invalid_nested_arguments_preserved_and_corrected_without_invention(setup, invalid):
    output, _, request, calls, actions = setup
    request['max_steps'] = 6
    actions.extend([action('search', query='lead'), action('open', source_id='s0'),
                    action('draft', hypothesis=invalid), action('draft', hypothesis=hypothesis()),
                    action('commit', hypothesis=hypothesis())])
    report = run(setup)
    assert report['status'] == 'hypotheses_ready' and len(calls) == 5
    assert report['steps'][2]['status'] == 'tool_rejected'
    assert report['steps'][2]['action']['hypothesis'] == invalid
    assert calls[3]['payload']['last_tool_result']['tool_error']
    assert json.loads(json.loads((output / 'response-002.json').read_text())['raw_response'])['hypothesis'] == invalid
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    assert len(calls) == 5


def test_invalid_nested_loop_preserves_committed_work_and_original_budget(setup):
    _, _, request, calls, actions = setup
    request.update(max_steps=24, max_hypotheses=2)
    actions.extend([action('search', query='lead'), action('open', source_id='s0'),
                    action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    actions.extend(action('draft', hypothesis={}) for _ in range(12))
    report = run(setup)
    assert report['status'] == 'stagnated' and len(report['hypotheses']) == 1
    assert len(calls) <= 13 and all(s['status'] == 'tool_rejected' for s in report['steps'][4:])


@pytest.mark.parametrize('render_images', [
    [{'source_id': 'unopened', 'image_index': 0, 'purpose': 'Borrow form'}],
    [{'source_id': 's0', 'image_index': 99, 'purpose': 'Borrow form'}],
    [{'source_id': 's0', 'image_index': 0, 'purpose': 'Borrow form'}] * 2,
    [{'source_id': 's0', 'image_index': 0, 'purpose': 'Borrow form', 'path': '/invented.png'}],
])
def test_invalid_image_attachment_returns_feedback_and_can_choose_text_only(setup, render_images):
    _, _, request, calls, actions = setup
    request['max_steps'] = 6
    actions.extend([action('search', query='lead'), action('open', source_id='s0'),
                    action('draft', hypothesis=hypothesis(render_images=render_images)),
                    action('draft', hypothesis=hypothesis(render_images=[])),
                    action('commit', hypothesis=hypothesis(render_images=[]))])
    result = run(setup)
    assert result['steps'][2]['status'] == 'tool_rejected'
    assert calls[3]['payload']['last_tool_result']['tool_error']
    assert result['hypotheses'][0]['render_images'] == []
    assert result['reasoning_calls'] == 5
