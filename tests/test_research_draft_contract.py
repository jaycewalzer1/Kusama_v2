"""Preserved v27 rejection and prospective v28 software behavior, without inference."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from kusama import research as r, research_rules as rules
from kusama.json_generation import generation_schema, json_prefix_constraint, JSON_FORMAT
from kusama.research_context import project_context
from test_json_generation import tokenizer
from test_research_grounded import session, analysis, execute, observations, hyp

INCIDENT = json.loads((Path(__file__).parent / 'fixtures/research/unsupported_property_attribution.json').read_text())


def test_exact_failed_attributions_stay_rejected_with_specific_feedback():
    source = INCIDENT['source']
    opened = {source['source_id']: source}
    for response in INCIDENT['responses']:
        raw = response['raw_response']
        assert hashlib.sha256(raw.encode()).hexdigest() == response['raw_response_sha256']
        hypothesis = json.loads(raw)['hypothesis']
        before = deepcopy(hypothesis)
        errors = rules.grounding_errors(hypothesis, opened)
        assert len(errors) == 2 and all('entailment failed' in e for e in errors)
        feedback = rules.draft_feedback(hypothesis, opened)
        assert len(feedback['missing_requirements']) == 3
        for item in feedback['property_failures']:
            contribution = hypothesis['sources'][0]
            assert item['submitted_property'] == contribution[item['field']]
            cited = contribution[item['field'] + '_evidence']['claim_id']
            assert item['cited_text'] == next(c['claim'] for c in source['claims'] if c['id'] == cited)
            assert not item['cited_text_truncated']
        assert hypothesis == before


def test_feedback_survives_admission_and_fixture_revision_commits(session):
    execute(session, analysis())
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    execute(session, {'action': 'assess_candidates', 'candidate_scores': [
        {'candidate_id': 'met', 'score': .8, 'reason': 'Fixture source evidence'}]})
    backend = session[3]
    original_open = backend.open
    def open_source(sid):
        return {**original_open(sid), **deepcopy(INCIDENT['source'])}
    backend.open = open_source
    execute(session, {'action': 'acquire', 'candidate_id': 'met'})
    sid = INCIDENT['source']['source_id']
    execute(session, {'action': 'inspect_images', 'reinspection': [observations(sid)]})
    root, spec, steps, _ = session
    action = r.ResearchActionEnvelope.model_validate(json.loads(INCIDENT['responses'][0]['raw_response']))
    with pytest.raises(ValueError) as rejected:
        r._execute_action(root, spec, r._state(spec, steps), steps, action, backend)
    message = str(rejected.value)
    for expected in ('draft_feedback', 'entailment failed', 'formal_properties', 'visible_predicates', 'single_source_justification'):
        assert expected in message
    # Emulate the runner's retained error record; the original completion stays rejected.
    steps.append({'index': len(steps), 'action': action.model_dump(mode='json'),
                  'tool_error': message, 'status': 'tool_rejected', 'opened_sources': []})
    payload = r._payload(spec, r._state(spec, steps), steps)
    context = {'payload': payload, 'system': spec['policy'], 'images': []}
    for stage in range(16):
        projected = project_context(context, stage)
        assert projected['payload']['draft_feedback'] == payload['draft_feedback']
        assert projected['payload']['draft_contract'] == payload['draft_contract']
    # Explicitly synthetic software comparison, never written to the live run.
    revised = hyp((sid,))
    revised['sources'][0].update(
        transferred_property='low-relief spiral designs and small carved figures at each end',
        transferred_property_evidence={'claim_id': sid + '_claim1'},
        discarded_property="hung from the rafters of the chief's house",
        discarded_property_evidence={'claim_id': sid + '_claim3'})
    revised.update(single_source_justification='Fixture checks one source citation in isolation.',
        formal_properties=['low-relief spiral designs', 'small carved figures'],
        render_instruction='Paint low-relief spiral designs and small carved figures in red.')
    execute(session, {'action': 'draft', 'hypothesis': revised})
    committed = execute(session, {'action': 'commit', 'hypothesis': revised, 'reinspection': [observations(sid)]})
    assert committed['committed_hypothesis']['sources'][0]['transferred_property'] == revised['sources'][0]['transferred_property']
    assert steps[-3]['status'] == 'tool_rejected'


def generation_payload(session):
    schema = deepcopy(session[1]['schema'])
    return {'output_schema': schema, 'draft_contract': {'policy': session[1]['draft_contract_policy']},
            'response_contract': {'required_fields_by_action': {'draft': ['action', 'hypothesis', 'reason']}}}


def ordered_action(payload, value):
    """Serialize scripted comparisons in the same frozen required-field order."""
    defs = payload['output_schema']['$defs']
    def order(item, name):
        fields = defs[name]['required']
        return {key: item[key] for key in fields + [k for k in item if k not in fields] if key in item}
    value = deepcopy(value)
    value['sources'] = [order(c, 'ResearchContribution') for c in value['sources']]
    value = order(value, 'ResearchHypothesis')
    if len(value['sources']) == 1 and 'single_source_justification' in value:
        value = {'single_source_justification': value.pop('single_source_justification'), **value}
    return {'action': 'draft', 'hypothesis': value, 'reason': 'Scripted software check'}


def accepts(tokenizer, schema, value):
    guard = json_prefix_constraint(tokenizer, JSON_FORMAT, schema)
    sequence = [tokenizer.eos_token_id]
    for token in tokenizer.encode(json.dumps(value, separators=(',', ':')), add_special_tokens=False):
        if token not in guard(0, torch.tensor(sequence)):
            return False
        sequence.append(token)
    return tokenizer.eos_token_id in guard(0, torch.tensor(sequence))


@pytest.mark.parametrize('case', ['two_sources', 'one_justified', 'one_unjustified', 'no_evidence',
                                'empty_evidence', 'no_formal', 'no_predicates', 'malformed_predicate'])
def test_actual_token_filter_enforces_complete_draft(session, tokenizer, case):
    payload = generation_payload(session)
    original = deepcopy(payload)
    value = hyp(('artic',)) if case.startswith('one_') else hyp()
    if case == 'one_justified':
        value['single_source_justification'] = 'One source isolates this formal test.'
    if case == 'no_evidence':
        del value['sources'][0]['transferred_property_evidence']
    if case == 'empty_evidence':
        value['sources'][0]['transferred_property_evidence'] = {}
    if case == 'no_formal':
        value['formal_properties'] = []
    if case == 'no_predicates':
        value['visible_predicates'] = []
    if case == 'malformed_predicate':
        del value['visible_predicates'][0]['measure']
    valid = case in ('two_sources', 'one_justified')
    assert accepts(tokenizer, generation_schema(payload), ordered_action(payload, value)) is valid
    assert payload == original


def test_claim_quote_and_both_citations_remain_available(session, tokenizer):
    payload = generation_payload(session)
    value = hyp()
    value['sources'][0]['transferred_property_evidence'] = {'claim_id': 'artic_claim'}
    value['sources'][1]['transferred_property_evidence']['claim_id'] = 'met_claim'
    assert accepts(tokenizer, generation_schema(payload), ordered_action(payload, value))


def test_old_generation_payload_keeps_its_frozen_optional_fields(session, tokenizer):
    payload = generation_payload(session)
    payload.pop('draft_contract')
    payload['output_schema'] = r.ResearchAction.model_json_schema()
    value = hyp()
    for field in ('formal_properties', 'visible_predicates'):
        value.pop(field)
    for contribution in value['sources']:
        contribution.pop('transferred_property_evidence')
        contribution.pop('discarded_property_evidence')
    assert accepts(tokenizer, generation_schema(payload), ordered_action(payload, value))


@pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1',
                   reason='Requires explicitly selected cached Qwen tokenizer; no weights')
def test_pinned_qwen_tokenizer_admits_complete_draft_and_blocks_missing_requirements(session):
    from transformers import AutoTokenizer
    from kusama.director import MODEL_ID, MODEL_REVISION
    cache = Path(__file__).resolve().parents[1] / '.cache/huggingface'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
        cache_dir=cache, local_files_only=True, trust_remote_code=False)
    payload = generation_payload(session)
    schema = generation_schema(payload)
    value = hyp(('artic',))
    value['single_source_justification'] = 'A scripted one-source software check.'
    action = ordered_action(payload, value)
    assert len(tokenizer.encode(json.dumps(action), add_special_tokens=False)) < 1800
    assert accepts(tokenizer, schema, action)
    del value['single_source_justification']
    assert not accepts(tokenizer, schema, ordered_action(payload, value))
    # Saved model wording is retained, not replaced by the synthetic comparison.
    for response in INCIDENT['responses']:
        value = json.loads(response['raw_response'])['hypothesis']
        assert not accepts(tokenizer, schema, ordered_action(payload, value))
