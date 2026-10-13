"""Stateful constraints on source attribution, not an artistic-content generator.

Intersect the syntax parser with opened-source IDs and contiguous evidence spans.
The model selects every source, citation, span and transformation. No completed
response is changed. State is immutable across token-trie branches.
"""
from copy import copy
import json
import re

from lmformatenforcer.characterlevelparser import CharacterLevelParser
from .research_rules import normalized, words

POLARITY = {'not', 'never', 'without', 'no'}
POLICY = 'distinct_sources_extractive_spans_v1'


def encoded(value):
    return json.dumps(value, ensure_ascii=False)[1:-1]


class EvidenceParser(CharacterLevelParser):
    def __init__(self, parser, contract):
        super().__init__(parser.config)
        if contract['policy'] != POLICY:
            raise ValueError('Unknown frozen evidence generation policy')
        self.parser = parser
        self.sources = contract['sources']
        self.stack = ()  # kind, path, current object key, array index
        self.values = {}
        self.in_string = False
        self.is_key = False
        self.escape = False
        self.buffer = ''
        self.path = ()
        self.restriction = None
        self.support = None
        self.required_polarity = set()

    def _path(self):
        if not self.stack:
            return ()
        kind, path, key, index = self.stack[-1]
        return path + ((key,) if kind == 'object' else (index,))

    def _choices(self, path, is_key):
        """Return (encoded string, viable offsets) sequences, or None for free text."""
        self.support = None
        self.required_polarity = set()
        if len(path) < 4 or path[:2] != ('hypothesis', 'sources'):
            return None
        base = path[:3]
        source = self.sources.get(self.values.get(base + ('source_id',)), {})
        if is_key:
            if len(path) == 4 and path[3].endswith('_property_evidence'):
                choices = (['claim_id'] if source.get('claims') else []) + (['quote'] if source.get('text') else [])
                return tuple((encoded(s), (0,)) for s in choices)
            return None
        field = path[3]
        if len(path) == 4 and field == 'source_id':
            used = {value for key, value in self.values.items()
                    if len(key) == 4 and key[:2] == ('hypothesis', 'sources') and key[3] == 'source_id'}
            return tuple((encoded(sid), (0,)) for sid in self.sources if sid not in used)
        if len(path) == 5 and field.endswith('_property_evidence'):
            if path[4] == 'claim_id':
                return tuple((encoded(s), (0,)) for s in source.get('claims', {}))
            if path[4] == 'quote':
                return self._spans(source.get('text', ''), 100000, polarity=False)
        if len(path) == 4 and field in ('transferred_property', 'discarded_property'):
            citation = base + (field + '_evidence',)
            claim = source.get('claims', {}).get(self.values.get(citation + ('claim_id',)), '')
            support = claim or self.values.get(citation + ('quote',), '')
            return self._spans(support, 700 if field == 'transferred_property' else 500, polarity=True)
        return None

    def _spans(self, support, maximum, *, polarity):
        self.support = support
        self.required_polarity = (words(support) & POLARITY) if polarity else set()
        # Only starts with a possible nonempty, polarity-preserving completion
        # within the existing field length are admitted. No span is selected here.
        ends = {word: [m.end() for m in re.finditer(r'\b' + word + r'\b', support, re.I)]
                for word in self.required_polarity}
        pieces = [encoded(character) for character in support]
        positions = [0]
        for piece in pieces:
            positions.append(positions[-1] + len(piece))
        offsets = []
        word_character = re.compile(r'\w')
        for start in range(len(support)):
            first = word_character.search(support, start)
            if first is None:
                continue
            minimum_end = first.end()
            for word, word_ends in ends.items():
                following = next((end for end in word_ends if start <= end - len(word)), None)
                if following is None:
                    break
                minimum_end = max(minimum_end, following)
            else:
                # The underlying grammar may count escaped characters. This
                # conservative bound keeps at least one legal closing point.
                if positions[minimum_end] - positions[start] <= maximum:
                    offsets.append(positions[start])
        return ((''.join(pieces), tuple(offsets)),)

    def _can_close_string(self):
        if self.restriction is None:
            return True
        if self.support is None:
            return any(len(text) in offsets for text, offsets in self.restriction)
        try:
            value = json.loads('"' + self.buffer + '"')
        except ValueError:
            return False
        return bool(normalized(value)) and normalized(value) in normalized(self.support) and not (
            self.required_polarity - words(value))

    def get_allowed_characters(self):
        allowed = self.parser.get_allowed_characters()
        if not self.in_string or self.restriction is None:
            return allowed
        next_chars = {text[i] for text, offsets in self.restriction for i in offsets if i < len(text)}
        if not self.escape and self._can_close_string():
            next_chars.add('"')
        return ''.join(c for c in allowed if c in next_chars)

    def add_character(self, character):
        result = copy(self)
        result.parser = self.parser.add_character(character)
        if self.in_string:
            closing = character == '"' and not self.escape
            if closing:
                value = json.loads('"' + self.buffer + '"')
                result.in_string = False
                result.restriction = None
                if self.is_key:
                    kind, path, _, index = self.stack[-1]
                    result.stack = self.stack[:-1] + ((kind, path, value, index),)
                elif self.path[:2] == ('hypothesis', 'sources'):
                    result.values = {**self.values, self.path: value}
            else:
                result.buffer = self.buffer + character
                result.escape = character == '\\' and not self.escape
                if self.restriction is not None:
                    result.restriction = tuple((text, tuple(i + 1 for i in offsets if i < len(text) and text[i] == character))
                                               for text, offsets in self.restriction)
            return result
        if character == '"':
            result.in_string = True
            result.escape = False
            result.buffer = ''
            result.is_key = bool(self.stack and self.stack[-1][0] == 'object' and self.stack[-1][2] is None)
            result.path = self.stack[-1][1] if result.is_key else self._path()
            result.restriction = result._choices(result.path, result.is_key)
        elif character in '{[':
            result.stack = self.stack + (('object' if character == '{' else 'array', self._path(), None, 0),)
        elif character in '}]':
            result.stack = self.stack[:-1]
        elif character == ',':
            kind, path, _, index = self.stack[-1]
            result.stack = self.stack[:-1] + ((kind, path, None, index + 1),)
        return result

    def can_end(self):
        return self.parser.can_end()

    def shortcut_key(self):
        # Freetext token shortcuts must never bypass evidence restrictions.
        return None if self.restriction is not None else self.parser.shortcut_key()
