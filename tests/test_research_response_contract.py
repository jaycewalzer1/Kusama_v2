"""Protocol regressions; these do not establish better model compliance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from kusama import research as r
from kusama.director import _unique_object
from kusama.research_context import project_context
from test_research_grounded import session, analysis, execute, opened, hyp


def test_requirements_follow_available_actions_and_survive_projection(session):
    _, spec, steps, _ = session
    frozen_schema = deepcopy(spec['schema'])
    payload = r._payload(spec, r._state(spec, steps), steps)
    assert payload['response_contract']['required_fields_by_action'] == {
        'analyze_question': ['action', 'reason', 'question_analysis', 'queries'], 'stop': ['action', 'reason']}
    execute(session, analysis())
    payload = r._payload(spec, r._state(spec, steps), steps)
    required = payload['response_contract']['required_fields_by_action']
    assert required['rewrite_queries'] == ['action', 'reason', 'queries']
    assert required['search'] == ['action', 'reason', 'query', 'mode']
    assert required['stop'] == ['action', 'reason']
    assert 'analyze_question' not in required
    context = {'system': spec['policy'], 'payload': payload, 'images': []}
    for stage in range(7):
        projected = project_context(context, stage)
        assert projected['payload']['response_contract'] == payload['response_contract']
        assert projected['payload']['prepared_queries'] == analysis()['queries']
        assert projected['payload']['output_schema'] == frozen_schema
    assert spec['schema'] == frozen_schema


def test_pending_study_requirements_preserve_hypothesis_and_reinspection(session):
    opened(session)
    execute(session, {'action': 'draft', 'hypothesis': hyp()})
    _, spec, steps, _ = session
    payload = r._payload(spec, r._state(spec, steps), steps)
    required = payload['response_contract']['required_fields_by_action']
    assert set(required) == {'commit', 'draft', 'reject', 'search', 'stop'}
    assert required['commit'] == ['action', 'reason', 'hypothesis', 'reinspection']


def test_saved_repetition_is_not_reinterpreted_as_a_valid_action(session):
    fixture = json.loads((Path(__file__).parent / 'fixtures/research/repeated_rewrite_queries.json').read_text())
    raw = fixture['raw_responses'][2]
    assert hashlib.sha256(raw.encode()).hexdigest() == fixture['raw_response_sha256'][2]
    with pytest.raises(json.JSONDecodeError, match='Extra data: line 5 column 1'):
        json.loads(raw, object_pairs_hook=_unique_object)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    remainder = raw
    objects = []
    while remainder.strip():
        value, end = decoder.raw_decode(remainder.lstrip())
        objects.append(value)
        remainder = remainder.lstrip()[end:]
    assert len(objects) == 5 and all(value == objects[0] for value in objects)
    execute(session, analysis())
    before = len(session[3].calls)
    with pytest.raises(ValueError, match='3 to 5'):
        execute(session, objects[0])
    assert len(session[3].calls) == before


def test_query_error_feedback_names_payload_fields(session):
    execute(session, analysis())
    _, spec, steps, _ = session
    feedback = r._next_error('Search requires a prepared short query', spec, r._state(spec, steps), steps)
    assert 'queries: 3 to 5 objects each containing query' in feedback
    assert 'field (artist, subject, technique, object, work)' in feedback


def test_no_index_search_requires_explicit_remote_mode_before_source_dispatch(session):
    execute(session, analysis())
    with pytest.raises(ValueError, match='requires mode="remote"'):
        execute(session, {'action': 'search', 'query': 'figure'})
    assert not session[3].calls
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    assert len(session[3].calls) == 3


@pytest.mark.parametrize('count,reason_length', [(7, 20), (1, 241)])
def test_oversized_assessment_is_rejected_before_scoring(session, count, reason_length):
    execute(session, analysis())
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    before = len(session[2])
    with pytest.raises(ValueError, match='at most six candidates'):
        execute(session, {'action': 'assess_candidates', 'candidate_scores': [
            {'candidate_id': 'artic', 'score': .8, 'reason': 'x' * reason_length}
            for _ in range(count)]})
    assert len(session[2]) == before
    assert not r._state(session[1], session[2])['scores']


def test_draft_requires_acquired_and_inspected_pixels_in_new_plans(session):
    from test_research_grounded import observations
    from kusama.json_generation import generation_schema
    _, spec, steps, _ = session
    spec['draft_evidence_policy'] = 'acquired_and_inspected_v1'
    initial, _ = r._research_progress(spec, r._state(spec, steps), steps)
    assert initial['next'].startswith('analyze_question')
    execute(session, analysis())
    state = r._state(spec, steps)
    payload = r._payload(spec, state, steps)
    assert 'draft' not in payload['response_contract']['required_fields_by_action']
    assert 'draft' not in [v['properties']['action']['enum'][0] for v in generation_schema(payload)['anyOf']]
    with pytest.raises(ValueError, match='Action unavailable'):
        execute(session, {'action': 'draft', 'hypothesis': hyp()})
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    execute(session, {'action': 'assess_candidates', 'candidate_scores': [
        {'candidate_id': 'artic', 'score': .8, 'reason': 'Specific formal evidence'}]})
    execute(session, {'action': 'acquire', 'candidate_id': 'artic'})
    state = r._state(spec, steps)
    payload = r._payload(spec, state, steps)
    assert payload['vision_probe_required']
    assert 'draft' not in payload['response_contract']['required_fields_by_action']
    assert 'inspect_images' in payload['response_contract']['required_fields_by_action']
    execute(session, {'action': 'inspect_images', 'reinspection': [observations('artic_canonical')]})
    payload = r._payload(spec, r._state(spec, steps), steps)
    assert 'draft' in payload['response_contract']['required_fields_by_action']


def test_acquisition_eligibility_uses_current_question_scores(session):
    _, spec, steps, _ = session
    spec['acquisition_generation_policy'] = 'conditional_relevance_override_v1'
    execute(session, analysis())
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    state = r._state(spec, steps)
    assert r._payload(spec, state, steps)['acquisition_contract']['eligible_without_override'] == []
    execute(session, {'action': 'assess_candidates', 'candidate_scores': [
        {'candidate_id': 'artic', 'score': .8, 'reason': 'Formal evidence'},
        {'candidate_id': 'met', 'score': .2, 'reason': 'Limited relevance'}]})
    state = r._state(spec, steps)
    assert r._payload(spec, state, steps)['acquisition_contract']['eligible_without_override'] == ['artic']
    state['question'] = 'A different inquiry'
    assert r._payload(spec, state, steps)['acquisition_contract']['eligible_without_override'] == []
