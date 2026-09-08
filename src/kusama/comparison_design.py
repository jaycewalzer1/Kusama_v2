"""Deterministic diagnostic comparisons, balanced across both realizations.

Twelve directions receive 108 pairs / 216 independent dimension questions.
This is a predeclared comparison schedule, never a record of completed reviews.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import hashlib
import json
import re


DEFAULT_SEED = 'kusama-diagnostic-v1'


def _hash(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def _order(direction_ids, seed):
    if not isinstance(direction_ids, (list, tuple)) or not 1 <= len(direction_ids) <= 12:
        raise ValueError('Supply 1–12 direction IDs')
    if any(not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', identifier)
           for identifier in direction_ids):
        raise ValueError('Direction IDs require 1–64 letters, digits, underscores or hyphens')
    if len(set(direction_ids)) != len(direction_ids):
        raise ValueError('Direction IDs must be unique')
    if not isinstance(seed, str) or not seed.strip() or len(seed) > 200:
        raise ValueError('The design seed must contain 1–200 characters')
    # The immutable IDs/seed determine the cycle, not Qwen's generation batches
    # or any aesthetic ranking obtained after viewing the outputs.
    return sorted(direction_ids, key=lambda identifier: (_hash([seed, 'direction', identifier]), identifier))


def diagnostic_pairs(direction_ids, seed=DEFAULT_SEED):
    """Return ReviewStore pair dicts for IDs such as d01-r1 and d01-r2.

    All candidate pairs are used for 1–11 directions. For 12, a regular schedule
    gives each of the 24 candidates 9 comparisons, including its sibling.
    """
    order = _order(direction_ids, seed)
    candidates = [(identifier, realization) for identifier in order for realization in (1, 2)]
    pairs = []
    if len(order) != 12:
        pairs.extend(combinations(candidates, 2))
    else:
        for identifier in order:
            pairs.append(((identifier, 1), (identifier, 2)))
        # Thirty direction edges receive two matched comparisons. Each direction
        # has five such neighbors; each realization contributes exactly once.
        for i, j in combinations(range(12), 2):
            distance = min(j - i, 12 - (j - i))
            if distance in (1, 2, 6):
                matching = (1, 2) if distance == 1 else (2, 1)
                for realization in (1, 2):
                    pairs.append(((order[i], realization), (order[j], matching[realization - 1])))
        # The remaining 36 direction edges receive one comparison. Every vertex
        # has three outgoing and three incoming edges, split evenly by variant.
        for i in range(12):
            for delta in (3, 4, 5):
                j = (i + delta) % 12
                pairs.append(((order[i], 1 + i % 2), (order[j], 2 - j % 2)))
    normalized = [tuple(sorted(f'{direction}-r{realization}' for direction, realization in pair))
                  for pair in pairs]
    if len(set(normalized)) != len(normalized):
        raise ValueError('Comparison design produced duplicate candidate pairs')
    # Intermix within- and cross-direction pairs instead of presenting them in
    # construction order. ReviewStore separately randomizes A/B presentation.
    normalized.sort(key=lambda pair: (_hash([seed, 'pair_order', pair]), pair))
    return [{'id': f'pair-{index + 1:03}', 'candidate_ids': list(pair), 'kind': 'alternatives'}
            for index, pair in enumerate(normalized)]


def design_metadata(direction_ids, seed=DEFAULT_SEED):
    order = _order(direction_ids, seed)
    pairs = diagnostic_pairs(direction_ids, seed)
    degrees = Counter(candidate for pair in pairs for candidate in pair['candidate_ids'])
    return {
        'schema_version': 1,
        'design': 'balanced_108_v1' if len(order) == 12 else 'all_candidate_pairs_v1',
        'seed': seed,
        'direction_cycle': order,
        'direction_order_policy': 'SHA-256 permutation of immutable direction IDs and seed; independent of outputs and Qwen batch order.',
        'pair_order_policy': 'SHA-256 permutation of candidate pair identities and seed; A/B presentation randomized separately per reviewer.',
        'candidate_count': 2 * len(order),
        'pair_count': len(pairs),
        'planned_dimension_questions': 2 * len(pairs),
        'dimensions': ['finished_quality', 'development_potential'],
        'comparisons_per_candidate': dict(sorted(degrees.items())),
        'all_direction_pairs_covered': True,
        'note': 'Planned diagnostic coverage, not completed human judgments. Equal exposure does not establish an unbiased ranking or generalization to unseen briefs.',
    }
