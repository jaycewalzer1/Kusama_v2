"""Frozen syntax, action-schema and optional evidence filtering; never output repair.

Use LM Format Enforcer's core API: its bundled Transformers adapter imports an
obsolete tokenizer location in Transformers 5. This adapter uses the same token
prefix protocol without modifying either dependency.
"""
from hashlib import sha256
from importlib.metadata import version
from copy import deepcopy


JSON_FORMAT = {'kind': 'json_object', 'engine': 'lm-format-enforcer',
               'version': '0.11.3', 'scope': 'bounded_action_schema_v1',
               'max_consecutive_whitespaces': 2, 'max_json_array_length': 16384}
_tokenizer_cache = None


def tokenizer_data(tokenizer):
    """Cache the trie, not decision state; bind it to the actual tokenizer bytes."""
    from lmformatenforcer import TokenEnforcerTokenizerData
    global _tokenizer_cache
    special = frozenset(tokenizer.all_special_ids)
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos not in special:
        raise ValueError('JSON generation requires an explicit tokenizer EOS token')
    identity = (sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),
                eos, special, len(tokenizer))
    if _tokenizer_cache is not None and _tokenizer_cache[0] == identity:
        return _tokenizer_cache[1]
    zero = tokenizer.encode('0', add_special_tokens=False)[-1]
    def decode(ids):
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    regular = []
    for index in range(len(tokenizer)):
        if index not in special:
            after_zero = decode([zero, index])[1:]
            regular.append((index, after_zero, len(after_zero) > len(decode([index]))))
    data = TokenEnforcerTokenizerData(regular, lambda ids: decode(ids).rstrip('\ufffd'),
                                     eos, False, len(tokenizer))
    # LMFE's freetext shortcut omits legal tokens such as .\",\": they close
    # a string and cross into the next field. Keep a small supplemental trie
    # for these boundary tokens instead of forcing alternative segmentation.
    from lmformatenforcer.tokenizerprefixtree import TokenizerPrefixTreeNode
    data.boundary_tree = TokenizerPrefixTreeNode()
    for index, text, _ in regular:
        if not text.startswith('"') and ('"' in text[:-1] or '\\' in text[:-1]):
            node = data.boundary_tree
            for character in text:
                node = node.children.setdefault(character, TokenizerPrefixTreeNode())
            node.tokens.append(index)
    _tokenizer_cache = (identity, data)
    return data


def generation_schema(payload):
    """Compile frozen action requirements; final Pydantic/tool checks still apply."""
    schema = deepcopy(payload['output_schema'])
    if payload.get('draft_contract'):
        # LMFE does not implement nested if/then requirements. Compile the frozen
        # single-source condition into two ordinary alternatives, not invented text.
        hypothesis = schema['$defs']['ResearchHypothesis']
        condition = hypothesis.pop('allOf')[0]['then']
        single, multiple = deepcopy(hypothesis), deepcopy(hypothesis)
        single['properties']['sources']['maxItems'] = 1
        # Resolve the alternatives at their first key. Keeping both branches
        # alive through long source strings makes the tokenizer traversal costly.
        single['required'] = condition['required'] + single['required']
        for name, constraint in condition['properties'].items():
            single['properties'][name].update(constraint)
        multiple['properties']['sources']['minItems'] = 2
        schema['$defs']['ResearchHypothesis'] = {'anyOf': [single, multiple]}
        # Each citation may contain either key or both; both must be nonempty.
        contribution = schema['$defs']['ResearchContribution']
        for field in ('transferred_property_evidence', 'discarded_property_evidence'):
            citation = contribution['properties'][field]
            choices = citation.pop('anyOf')
            contribution['properties'][field] = {'anyOf': [{**deepcopy(citation), **choice} for choice in choices]}
    grounded = payload.get('grounded_generation')
    if grounded is not None:
        count = len(grounded['sources'])
        alternatives = schema['$defs']['ResearchHypothesis']['anyOf']
        alternatives[:] = [option for option in alternatives
                           if option['properties']['sources'].get('minItems', 1) <= count]
        for option in alternatives:
            option['properties']['sources']['maxItems'] = min(count, option['properties']['sources'].get('maxItems', 6))
    requirements = payload['response_contract']['required_fields_by_action']
    variants = []
    for action, required in requirements.items():
        if grounded is not None and not grounded['sources'] and action in ('draft', 'commit'):
            continue
        choices = payload.get('search_choices', {}).get('unused_prepared_queries')
        if action == 'search' and choices == []:
            continue
        variant = deepcopy(schema)
        variant.pop('$defs', None)
        conditions = variant.pop('allOf', [])
        variant['properties']['action'] = {'type': 'string', 'enum': [action]}
        if action == 'search' and choices is not None:
            variant['properties']['query']['enum'] = choices
        for condition in conditions:
            if condition.get('if', {}).get('properties', {}).get('action', {}).get('const') == action:
                for name, constraint in condition.get('then', {}).get('properties', {}).items():
                    variant['properties'][name].update(constraint)
        # Bound the reason using its existing field contract, and put action
        # arguments before it. This adds no model-authored content.
        variant['required'] = ['action'] + [name for name in required if name not in ('action', 'reason')] + ['reason']
        if action == 'acquire' and payload.get('acquisition_contract') is not None:
            variant['properties']['candidate_id'] = {'type': 'string', 'minLength': 1}
            eligible = payload['acquisition_contract']['eligible_without_override']
            if eligible:
                scored = deepcopy(variant)
                scored['properties']['candidate_id']['enum'] = eligible
                variants.append(scored)
            # The existing runtime rule already requires an explicit override for
            # unscored/low-relevance candidates. Preserve that field attribution.
            variant['required'].insert(-1, 'relevance_override')
            variant['properties']['relevance_override']['minLength'] = 1
        variants.append(variant)
    def clean(value):
        if isinstance(value, list):
            return [clean(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {key: clean(item) for key, item in value.items()
                if not (key == 'additionalProperties' and item is True)}
    result = clean({'anyOf': variants, '$defs': schema.get('$defs', {})})
    if grounded is not None:
        result['x-kusama-grounded-generation'] = deepcopy(grounded)
    return result


def json_prefix_constraint(tokenizer, contract, schema=None):
    """One new parser per response. Unknown contracts fail before generation."""
    from lmformatenforcer import JsonSchemaParser, TokenEnforcer
    from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
    if contract != JSON_FORMAT or version('lm-format-enforcer') != contract['version']:
        raise ValueError('Frozen JSON generation contract or dependency differs')
    data = tokenizer_data(tokenizer)
    # The library otherwise reads LMFE_* environment defaults. Freeze these
    # controls explicitly; the existing 16,384-character response cap is tighter.
    config = CharacterLevelParserConfig(alphabet=data.tokenizer_alphabet,
        force_json_field_order=schema is not None,
        max_consecutive_whitespaces=contract['max_consecutive_whitespaces'],
        max_json_array_length=contract['max_json_array_length'])
    parser = JsonSchemaParser(schema or {'type': 'object'}, config=config)
    if schema and schema.get('x-kusama-grounded-generation') is not None:
        from .research_generation import EvidenceParser
        parser = EvidenceParser(parser, schema['x-kusama-grounded-generation'])
    class BoundaryTokenEnforcer(TokenEnforcer):
        def _collect_allowed_tokens(self, parser, tree_node, allowed_tokens, shortcut_key):
            super()._collect_allowed_tokens(parser, tree_node, allowed_tokens, shortcut_key)
            if isinstance(shortcut_key, tuple) and shortcut_key[0] == 'json_freetext':
                super()._collect_allowed_tokens(parser, data.boundary_tree, allowed_tokens, None)
    enforcer = BoundaryTokenEnforcer(data, parser)
    parser.config = config

    def allowed(batch_id, input_ids):
        if batch_id != 0:
            raise ValueError('Research JSON filtering expects a single sequence')
        sequence = input_ids.tolist()
        result = enforcer.get_allowed_tokens(sequence).allowed_tokens
        # Research freezes num_beams=1 and returns one sequence. Retaining every
        # prior 150k-token allowlist would grow memory throughout long responses.
        key = tuple(sequence)
        enforcer.prefix_states = {key: enforcer.prefix_states[key]}
        if data.eos_token_id in result:
            return [data.eos_token_id]  # The one complete object ends this response.
        return result
    return allowed
