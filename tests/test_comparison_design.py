"""Verify diagnostic coverage rather than mirroring its graph construction."""
from collections import Counter
from itertools import combinations

import pytest

from kusama.comparison_design import diagnostic_pairs, design_metadata


IDS = [f'd{i:02}' for i in range(1, 13)]


def test_twelve_directions_yield_216_questions_with_equal_exposure():
    pairs = diagnostic_pairs(IDS)
    assert len(pairs) == 108
    assert len({pair['id'] for pair in pairs}) == 108
    assert len({tuple(sorted(pair['candidate_ids'])) for pair in pairs}) == 108
    degrees = Counter(candidate for pair in pairs for candidate in pair['candidate_ids'])
    assert set(degrees) == {f'{direction}-r{realization}' for direction in IDS for realization in (1, 2)}
    assert set(degrees.values()) == {9}
    assert all(pair['kind'] == 'alternatives' for pair in pairs)
    within, direction_edges, realization_types = [], Counter(), Counter()
    for pair in pairs:
        (left, lr), (right, rr) = [candidate.rsplit('-r', 1) for candidate in pair['candidate_ids']]
        if left == right:
            within.append(left)
            assert {lr, rr} == {'1', '2'}
        else:
            direction_edges[tuple(sorted((left, right)))] += 1
            realization_types[tuple(sorted((lr, rr)))] += 1
    assert sorted(within) == IDS
    assert set(direction_edges) == set(combinations(IDS, 2))
    assert Counter(direction_edges.values()) == {1: 36, 2: 30}
    assert realization_types == {('1', '1'): 24, ('2', '2'): 24, ('1', '2'): 48}
    metadata = design_metadata(IDS)
    assert metadata['planned_dimension_questions'] == 216
    assert metadata['comparisons_per_candidate'] == dict(degrees)
    assert metadata['design'] == 'balanced_108_v1'


def test_seeded_schedule_is_stable_and_independent_of_input_batch_order():
    original = diagnostic_pairs(IDS, seed='experiment-one')
    assert diagnostic_pairs(list(reversed(IDS)), seed='experiment-one') == original
    assert diagnostic_pairs(IDS[4:] + IDS[:4], seed='experiment-one') == original
    assert diagnostic_pairs(IDS, seed='experiment-two') != original
    metadata = design_metadata(IDS)
    assert metadata['direction_cycle'] != IDS
    assert set(metadata['direction_cycle']) == set(IDS)
    assert 'not completed' in metadata['note']


@pytest.mark.parametrize('count', range(1, 12))
def test_smaller_experiments_compare_every_candidate_pair(count):
    ids = IDS[:count]
    pairs = diagnostic_pairs(ids)
    candidates = {f'{direction}-r{realization}' for direction in ids for realization in (1, 2)}
    assert {frozenset(pair['candidate_ids']) for pair in pairs} == {
        frozenset(pair) for pair in combinations(candidates, 2)}
    assert len(pairs) <= 300
    assert design_metadata(ids)['planned_dimension_questions'] == 2 * len(pairs)


@pytest.mark.parametrize('ids,seed', [([], 'x'), (IDS + ['d13'], 'x'), (['d01', 'd01'], 'x'),
    (['../escape'], 'x'), ([None], 'x'), ('d01', 'x'), (IDS, ''), (IDS, 12)])
def test_invalid_design_inputs_rejected_before_building_pairs(ids, seed):
    with pytest.raises(ValueError):
        diagnostic_pairs(ids, seed=seed)
