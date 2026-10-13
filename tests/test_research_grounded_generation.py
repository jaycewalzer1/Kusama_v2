"""Prospective generation checks; scripted completions are not model outcomes."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest
from lmformatenforcer import JsonSchemaParser
from kusama.json_generation import generation_schema
from kusama.research_generation import EvidenceParser, POLICY
from kusama import research as r, research_rules as rules
from test_research_draft_contract import generation_payload, ordered_action, accepts
from test_research_grounded import session, hyp
from test_json_generation import tokenizer


def contract():
    return {'policy': POLICY, 'sources': {
        sid: {'text': 'A red figure stands without a crown. Blue sky above.',
              'claims': {sid + '_claim': 'A red figure stands without a crown.'}}
        for sid in ('artic', 'met')}}


def valid_hypothesis():
    value = hyp()
    for c in value['sources']:
        c.update(transferred_property='red figure stands without a crown',
                 transferred_property_evidence={'claim_id': c['source_id'] + '_claim'},
                 discarded_property='Blue sky', discarded_property_evidence={'quote': 'Blue sky above.'})
    return value


@pytest.mark.parametrize('case', ['valid', 'duplicate', 'unknown_source', 'wrong_claim',
    'unsupported', 'polarity', 'fake_quote', 'one_source', 'one_unjustified'])
def test_real_token_filter_blocks_source_and_attribution_failures(session, tokenizer, case):
    payload = generation_payload(session)
    payload['grounded_generation'] = contract()
    value = valid_hypothesis()
    if case == 'duplicate':
        value['sources'][1] = deepcopy(value['sources'][0])
    elif case == 'unknown_source':
        value['sources'][0]['source_id'] = 'invented'
    elif case == 'wrong_claim':
        value['sources'][0]['transferred_property_evidence']['claim_id'] = 'met_claim'
    elif case == 'unsupported':
        value['sources'][0]['transferred_property'] = 'red figure represents agency'
    elif case == 'polarity':
        value['sources'][0]['transferred_property'] = 'a crown'
    elif case == 'fake_quote':
        value['sources'][0]['discarded_property_evidence']['quote'] = 'Invented blue sky'
    elif case.startswith('one_'):
        value['sources'] = value['sources'][:1]
        if case == 'one_source':
            value['single_source_justification'] = 'One source isolates this fixture test.'
    before = deepcopy(payload)
    assert accepts(tokenizer, generation_schema(payload), ordered_action(payload, value)) == (case in ('valid', 'one_source'))
    assert payload == before


def feed_characters(value, evidence):
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    config = CharacterLevelParserConfig()
    config.alphabet += '“”é'
    parser = EvidenceParser(JsonSchemaParser({'type': 'object'}, config=config), evidence)
    for character in json.dumps(value, ensure_ascii=False, separators=(',', ':')):
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def test_escaped_quotes_unicode_and_all_supported_span_boundaries():
    text = '“Red” cloth, "blue" sky and café. No crown.'
    evidence = {'policy': POLICY, 'sources': {'s': {'text': text, 'claims': {}}}}
    for span in ('“Red” cloth', '"blue" sky', 'café', 'No crown'):
        c = {'source_id': 's', 'transferred_property_evidence': {'quote': text}, 'transferred_property': span}
        # Quote includes a negation: the runtime contract requires retaining it.
        expected = not rules.grounding_errors({'sources': [{**c, 'discarded_property': text,
            'discarded_property_evidence': {'quote': text}}]}, {'s': evidence['sources']['s']})
        assert feed_characters({'hypothesis': {'sources': [c]}}, evidence) == expected
    for span in ('“Red” cloth', '"blue" sky', 'café'):
        c = {'source_id': 's', 'transferred_property_evidence': {'quote': span}, 'transferred_property': span}
        assert feed_characters({'hypothesis': {'sources': [c]}}, evidence)


def test_no_sources_disable_draft_but_leave_stop(session, tokenizer):
    payload = generation_payload(session)
    payload['grounded_generation'] = {'policy': POLICY, 'sources': {}}
    payload['response_contract']['required_fields_by_action']['stop'] = ['action', 'reason']
    schema = generation_schema(payload)
    assert [v['properties']['action']['enum'] for v in schema['anyOf']] == [['stop']]
    assert accepts(tokenizer, schema, {'action': 'stop', 'reason': 'Evidence is insufficient.'})


def test_single_available_source_cannot_enter_multi_source_branch(session, tokenizer):
    payload = generation_payload(session)
    payload['grounded_generation'] = contract()
    del payload['grounded_generation']['sources']['met']
    schema = generation_schema(payload)
    assert len(schema['$defs']['ResearchHypothesis']['anyOf']) == 1
    value = valid_hypothesis()
    value['sources'] = value['sources'][:1]
    value['single_source_justification'] = 'Fixture one-source justification.'
    assert accepts(tokenizer, schema, ordered_action(payload, value))


def test_contract_survives_context_projection(session):
    from kusama.research_context import project_context
    payload = generation_payload(session)
    payload['grounded_generation'] = contract()
    for stage in range(16):
        assert project_context({'payload': payload, 'images': [], 'system': ''}, stage)['payload']['grounded_generation'] == contract()


@pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1', reason='Opt-in cached tokenizer only; no inference')
def test_pinned_qwen_grounded_generation(session):
    from transformers import AutoTokenizer
    from kusama.director import MODEL_ID, MODEL_REVISION
    cache = Path(__file__).resolve().parents[1] / '.cache/huggingface'
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache,
                                       local_files_only=True, trust_remote_code=False)
    payload = generation_payload(session)
    payload['grounded_generation'] = contract()
    schema = generation_schema(payload)
    value = valid_hypothesis()
    assert accepts(tok, schema, ordered_action(payload, value))
    value['sources'][1] = deepcopy(value['sources'][0])
    assert not accepts(tok, schema, ordered_action(payload, value))
    value = valid_hypothesis()
    value['sources'][0]['transferred_property'] = 'This visually embodies dissolution of the self.'
    assert not accepts(tok, schema, ordered_action(payload, value))


INCIDENT = json.loads((Path(__file__).parent / 'fixtures/research/grounded_generation_incident.json').read_text())


def test_exact_saved_incident_is_blocked_without_rewriting(session, tokenizer):
    raw = INCIDENT['raw_response']
    assert hashlib.sha256(raw.encode()).hexdigest() == INCIDENT['raw_response_sha256']
    source = INCIDENT['source']
    payload = generation_payload(session)
    payload['grounded_generation'] = {'policy': POLICY, 'sources': {source['source_id']: {
        'text': source['text'], 'claims': {c['id']: c['claim'] for c in source['claims']}}}}
    value = json.loads(raw)
    assert not accepts(tokenizer, generation_schema(payload), ordered_action(payload, value['hypothesis']))
    assert not feed_characters(value, payload['grounded_generation'])
    assert len(rules.grounding_errors(value['hypothesis'], {source['source_id']: source})) == 4
    with pytest.raises(ValueError, match='Reference IDs must be distinct'):
        r.ResearchHypothesis.model_validate(value['hypothesis'])


def test_generated_fixture_draft_and_commit_pass_existing_runtime(session, tokenizer):
    from test_research_grounded import opened, execute, observations
    opened(session)
    ids = ('artic_canonical', 'met_canonical')
    root, spec, steps, _ = session
    # No source text enters the grammar before image inspection.
    payload = r._payload(spec, r._state(spec, steps), steps)
    assert not payload['grounded_generation']['sources']
    execute(session, {'action': 'inspect_images', 'reinspection': [observations(ids[1])]})
    execute(session, {'action': 'open', 'source_id': ids[0]})
    execute(session, {'action': 'inspect_images', 'reinspection': [observations(ids[0])]})
    payload = r._payload(spec, r._state(spec, steps), steps)
    assert set(payload['grounded_generation']['sources']) == set(ids)
    payload['output_schema'] = spec['schema']
    value = hyp(ids)
    assert accepts(tokenizer, generation_schema(payload), ordered_action(payload, value))
    execute(session, {'action': 'draft', 'hypothesis': value})
    row = execute(session, {'action': 'commit', 'hypothesis': value,
                            'reinspection': [observations(sid) for sid in ids]})
    assert row['committed_hypothesis']['sources'][0]['source_id'] == ids[0]


def test_span_start_has_a_polarity_preserving_completion_with_escaped_length_bound():
    # A prefix that cannot reach its required negation within the property bound
    # must be unavailable from the start, rather than dead-ending mid-generation.
    text = '"' * 400 + ' without a crown'
    evidence = {'policy': POLICY, 'sources': {'s': {'text': text, 'claims': {'c': text}}}}
    parser = EvidenceParser(JsonSchemaParser({'type': 'object'}), evidence)
    choices = parser._spans(text, 500, polarity=True)
    assert 0 not in choices[0][1]
    assert choices[0][1]
    for offset in choices[0][1]:
        suffix = choices[0][0][offset:]
        assert suffix.index('without') + len('without') <= 500
