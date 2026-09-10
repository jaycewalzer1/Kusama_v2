"""Syntax enforcement regressions, without Qwen weights or new artistic work."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from kusama import director, research
from kusama.json_generation import JSON_FORMAT, json_prefix_constraint

FIXTURE = Path(__file__).parent / 'fixtures/research/missing_observation_bracket.json'
CACHE = Path(__file__).resolve().parents[1] / '.cache/huggingface'


def test_original_missing_bracket_remains_rejected():
    fixture = json.loads(FIXTURE.read_text())
    raw = fixture['raw_response']
    assert hashlib.sha256(raw.encode()).hexdigest() == fixture['raw_response_sha256']
    with pytest.raises(json.JSONDecodeError) as exc:
        json.loads(raw, object_pairs_hook=director._unique_object)
    assert str(exc.value) == fixture['parse_error']
    assert exc.value.pos == 873 and not fixture['reached_token_limit']


@pytest.fixture(scope='module')
def tokenizer():
    # A small real byte-level tokenizer keeps syntax tests portable. The pinned
    # Qwen tokenizer runs the identical cases under the processor acceptance gate.
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    core = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(alphabet)}, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    core.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=core, eos_token='<eos>')


def feed(tokenizer, raw):
    guard = json_prefix_constraint(tokenizer, JSON_FORMAT)
    sequence = [tokenizer.eos_token_id]  # Represents a prompt, not generated text.
    allowed = guard(0, torch.tensor(sequence))
    assert tokenizer.eos_token_id not in allowed
    tokens = tokenizer.encode(raw, add_special_tokens=False)
    for token in tokens:
        if token not in allowed:
            return guard, sequence, allowed, token
        sequence.append(token)
        allowed = guard(0, torch.tensor(sequence))
    return guard, sequence, allowed, None


@pytest.mark.parametrize('raw', [
    '{"action":"stop","reason":"No relevant material."}',
    json.dumps({'x': ['quoted "text"', 3, True, None, {'label': 'Òpó Ògògá — 女王'}]}),
    '{"observations":["a", "b", "c"],"supports":[]}',
    json.dumps({'x': list(range(30)), 'whitespace': 'ordinary spacing'}),
])
def test_complete_objects_allow_eos_and_block_second_object(tokenizer, raw):
    _, sequence, allowed, blocked = feed(tokenizer, raw)
    assert blocked is None, (tokenizer.decode([blocked]), tokenizer.decode(sequence))
    assert tokenizer.eos_token_id in allowed
    opening = tokenizer.encode('{', add_special_tokens=False)[0]
    assert opening not in allowed


def test_exact_incident_cannot_emit_its_mismatched_closing_token(tokenizer):
    raw = json.loads(FIXTURE.read_text())['raw_response']
    _, sequence, allowed, blocked = feed(tokenizer, raw)
    assert blocked is not None
    assert '}' in tokenizer.decode([blocked])
    assert tokenizer.eos_token_id not in allowed
    # An explicitly labelled synthetic comparison, never a recovered run output.
    synthetic = raw[:873] + ']' + raw[873:]
    _, _, allowed, blocked = feed(tokenizer, synthetic)
    assert blocked is None and tokenizer.eos_token_id in allowed


def test_invalid_contract_cannot_silently_disable_filter(tokenizer):
    with pytest.raises(ValueError, match='contract or dependency differs'):
        json_prefix_constraint(tokenizer, {**JSON_FORMAT, 'version': 'unknown'})


def test_whitespace_cannot_exhaust_the_completion(tokenizer):
    guard, sequence, allowed, blocked = feed(tokenizer, '{"x":[]\n\n')
    assert blocked is None
    newline = tokenizer.encode('\n', add_special_tokens=False)[0]
    assert newline not in allowed
    closing = tokenizer.encode('}', add_special_tokens=False)[0]
    assert closing in allowed
    sequence.append(closing)
    assert guard(0, torch.tensor(sequence)) == [tokenizer.eos_token_id]


def test_generation_schema_requires_arguments_before_bounded_reason(tokenizer):
    from kusama.json_generation import generation_schema
    payload = {'output_schema': research.ResearchAction.model_json_schema(),
        'response_contract': {'required_fields_by_action': {
            'analyze_question': ['action', 'reason', 'question_analysis', 'queries'],
            'stop': ['action', 'reason']}}}
    schema = generation_schema(payload)
    assert schema['anyOf'][0]['required'] == ['action', 'question_analysis', 'queries', 'reason']
    assert schema['anyOf'][1]['properties']['reason']['maxLength'] == 1000
    guard = json_prefix_constraint(tokenizer, JSON_FORMAT, schema)
    sequence = [tokenizer.eos_token_id]
    prefix = '{"action":"stop","reason":"' + 'x' * 1000
    for token in tokenizer.encode(prefix, add_special_tokens=False):
        assert token in guard(0, torch.tensor(sequence))
        sequence.append(token)
    allowed = guard(0, torch.tensor(sequence))
    assert tokenizer.encode('x', add_special_tokens=False)[0] not in allowed
    assert tokenizer.encode('"', add_special_tokens=False)[0] in allowed


def test_generation_schema_excludes_completed_searches_without_supplying_a_query():
    from kusama.json_generation import generation_schema
    payload = {'output_schema': research.ResearchAction.model_json_schema(),
        'response_contract': {'required_fields_by_action': {
            'search': ['action', 'reason', 'query', 'mode'], 'stop': ['action', 'reason']}},
        'search_choices': {'unused_prepared_queries': ['figure']}}
    schema = generation_schema(payload)
    assert schema['anyOf'][0]['properties']['query']['enum'] == ['figure']
    payload['search_choices']['unused_prepared_queries'] = []
    schema = generation_schema(payload)
    assert len(schema['anyOf']) == 1 and schema['anyOf'][0]['properties']['action']['enum'] == ['stop']


def test_transformers_generate_masks_premature_eos_and_wrong_bracket(tokenizer, monkeypatch):
    from transformers import GPT2Config, GPT2LMHeadModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
    raw = '{"x":[]}'
    scripted = tokenizer.encode(raw, add_special_tokens=False)
    eos = tokenizer.eos_token_id
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer), n_positions=32,
        n_embd=8, n_layer=1, n_head=1, eos_token_id=eos, pad_token_id=eos))
    def forward(input_ids, **kwargs):
        index = input_ids.shape[-1] - 1
        scores = torch.full((1, input_ids.shape[-1], len(tokenizer)), -100.0)
        scores[0, -1, scripted[index] if index < len(scripted) else eos] = 5
        scores[0, -1, eos] = 10  # Always prefer stopping, even inside the object.
        if index == 6:
            scores[0, -1, tokenizer.encode('}', add_special_tokens=False)[0]] = 9
        return CausalLMOutputWithPast(logits=scores)
    monkeypatch.setattr(model, 'forward', forward)
    generated = model.generate(input_ids=torch.tensor([[eos]]),
        attention_mask=torch.ones((1, 1), dtype=torch.long), use_cache=False,
        max_new_tokens=12, do_sample=False,
        prefix_allowed_tokens_fn=json_prefix_constraint(tokenizer, JSON_FORMAT))
    assert tokenizer.decode(generated[0, 1:], skip_special_tokens=True) == raw
    assert generated[0, -1] == eos


@pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1',
                   reason='Requires explicitly selected pinned local Qwen processor; no weights')
def test_pinned_qwen_tokens_block_the_saved_failure_and_preserve_unicode():
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(director.MODEL_ID, revision=director.MODEL_REVISION,
        cache_dir=CACHE, local_files_only=True, trust_remote_code=False, use_fast=False)
    test_exact_incident_cannot_emit_its_mismatched_closing_token(processor.tokenizer)
    test_complete_objects_allow_eos_and_block_second_object(processor.tokenizer,
        '{"x":["quoted \\\"text\\\"",3,true,null,{"label":"Òpó Ògògá — 女王"}]}')


def test_generation_routes_guard_and_preserves_strict_validation(tmp_path, tokenizer):
    from test_director import ProcessorStub, ModelStub
    processor = ProcessorStub('{"action":"stop","reason":"No relevant material."}')
    processor.apply_chat_template = lambda messages, **kw: 'fixture prompt'
    processor.tokenizer = tokenizer
    payload = {'output_schema': research.ResearchAction.model_json_schema(),
               'response_contract': {'required_fields_by_action': {'stop': ['action', 'reason']}}}
    holder = SimpleNamespace(config=director.DirectorConfig(tmp_path, max_new_tokens=1800,
        max_input_tokens=16384), processor=processor, model=ModelStub(),
        execution_settings={'status': 'ready', 'device': 'cpu'})
    result = director._multimodal_json(holder, 'system', payload, [],
        lambda v: research.ResearchActionEnvelope.model_validate(v).model_dump(mode='json'),
        allow_text_only=True, total_token_limit=16384, response_format=deepcopy(JSON_FORMAT))
    assert result['metadata']['response_format'] == JSON_FORMAT
    assert callable(holder.model.calls[0]['prefix_allowed_tokens_fn'])
    assert holder.model.calls[0]['max_new_tokens'] == 1800
    assert result['raw_response'] == processor.response
    # A backend ignoring the filter still cannot bypass the strict parser.
    processor.response = json.loads(FIXTURE.read_text())['raw_response']
    with pytest.raises(director.DirectorOutputError) as exc:
        director._multimodal_json(holder, 'system', payload, [], lambda v: v,
            allow_text_only=True, response_format=JSON_FORMAT)
    assert exc.value.result['raw_response'] == processor.response
    # Historical callers continue without a generation filter.
    processor.response = '{"action":"stop"}'
    director._multimodal_json(holder, 'system', {}, [], lambda v: v, allow_text_only=True)
    assert 'prefix_allowed_tokens_fn' not in holder.model.calls[-1]


@pytest.mark.parametrize('candidate,override,valid', [
    ('eligible', None, True), ('unscored', None, False),
    ('unscored', 'Specific formal evidence to investigate', True), ('unscored', '', False),
])
def test_acquisition_generation_requires_explicit_override_when_needed(tokenizer, candidate, override, valid):
    from kusama.json_generation import generation_schema
    schema = research.ResearchAction.model_json_schema()
    payload = {'output_schema': schema,
        'response_contract': {'required_fields_by_action': {'acquire': ['action', 'candidate_id', 'reason']}},
        'acquisition_contract': {'eligible_without_override': ['eligible']}}
    guard = json_prefix_constraint(tokenizer, JSON_FORMAT, generation_schema(payload))
    value = {'action': 'acquire', 'candidate_id': candidate}
    if override is not None:
        value['relevance_override'] = override
    value['reason'] = 'Model rationale'
    sequence = [tokenizer.eos_token_id]
    accepted = True
    for token in tokenizer.encode(json.dumps(value, separators=(',', ':')), add_special_tokens=False):
        if token not in guard(0, torch.tensor(sequence)):
            accepted = False
            break
        sequence.append(token)
    assert accepted is valid
    if valid:
        assert tokenizer.eos_token_id in guard(0, torch.tensor(sequence))
