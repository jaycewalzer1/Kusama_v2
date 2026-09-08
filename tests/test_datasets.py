"""Synthetic test envelopes only; no fixtures enter released research datasets."""
import copy
import json

import pytest

from kusama.datasets import LearningEvent, build_snapshot, grouped_assignments, load_snapshot, review_events

CUTOFF = '2026-09-08T20:00:00+00:00'


def event(index=0, **overrides):
    return dict(event_id=f'event-{index}', kind='judge_sft', occurred_at='2026-09-08T12:00:00+00:00',
                decision_at='2026-09-08T11:00:00+00:00', project_id=f'p{index}', brief_id=f'b{index}', episode_id=f'e{index}',
                trajectory_id=f't{index}', label_origin='human', reviewer='fixture-reviewer', label='A',
                source_event_ids=[f'fixture-review-{index}'], input_events=[{'event_id': f'input-{index}',
                'occurred_at': '2026-09-08T10:00:00+00:00', 'episode_id': f'e{index}', 'data': {'brief': 'test fixture'}}],
                presented_images=[{'candidate_id': f'a{index}', 'sha256': f'hasha{index}'},
                                  {'candidate_id': f'b{index}', 'sha256': f'hashb{index}'}]) | overrides


def test_empty_human_export_truthful_and_immutable(tmp_path):
    report = build_snapshot([], tmp_path, cutoff=CUTOFF)
    assert report['eligible_count'] == 0
    manifest, splits = load_snapshot(report['snapshot_path'])
    assert splits == {'train': [], 'development': []}
    assert build_snapshot([], tmp_path, cutoff=CUTOFF) == report
    assert '0 eligible rows' in (tmp_path / report['snapshot_id'] / 'data_card.md').read_text()


def test_missing_synthetic_future_sealed_excluded(tmp_path):
    future = event(2)
    future['input_events'][0]['occurred_at'] = '2026-09-08T13:00:00+00:00'
    rows = [event(0, label_origin='synthetic'), event(1, label=None), future, event(3, sealed=True),
            event(4, occurred_at='2026-09-09T12:00:00+00:00')]
    report = build_snapshot(rows, tmp_path, cutoff=CUTOFF)
    assert report['eligible_count'] == 0
    assert report['rejected_count'] == 5
    assert report['rejection_reasons']['future_event_in_input'] == 1
    assert report['rejection_reasons']['not_genuine_human_label'] == 1


def test_grouped_60_20_20_before_task_expansion_and_tiny_data(tmp_path):
    rows = [event(i) for i in range(5)]
    report = build_snapshot(rows, tmp_path, cutoff=CUTOFF)
    assert report['split_counts'] == {'train': 3, 'development': 1, 'test': 1}
    small = build_snapshot(rows[:4], tmp_path, cutoff=CUTOFF)
    assert small['split_counts'] == {'insufficient_groups': 4}
    with pytest.raises(ValueError, match='Test labels'):
        load_snapshot(report['snapshot_path'], splits=('test',))


def test_related_projects_descendants_and_crops_are_grouped():
    rows = [event(0, lineage_ids=['parent-image']), event(1, lineage_ids=['parent-image']),
            event(2, source_family_ids=['artist'], source_parent_ids=['source-full']),
            event(3, source_family_ids=['artist'], source_parent_ids=['source-full'], sealed=True)]
    models = [LearningEvent.model_validate(r) for r in rows]
    ordinary = grouped_assignments(models)
    source = grouped_assignments(models, claim='source_family_transfer')
    assert ordinary['event-0']['split_group'] == ordinary['event-1']['split_group']
    assert ordinary['event-2']['split_group'] != ordinary['event-3']['split_group']
    assert source['event-2']['split'] == source['event-3']['split'] == 'sealed'


def test_sealed_parent_quarantines_unlabelled_linked_descendants(tmp_path):
    report = build_snapshot([event(0, sealed=True, label=None, lineage_ids=['shared']),
                             event(1, lineage_ids=['shared'])], tmp_path, cutoff=CUTOFF)
    assert report['eligible_count'] == 0
    assert report['rejection_reasons']['sealed_or_related_to_sealed'] == 2


def test_director_excludes_future_child_and_requires_real_support(tmp_path):
    row = event(0, kind='director_sft', target={'action': {'operation': 'stop'}},
                outcome={'realized': True, 'review_event_ids': ['human-review']})
    row['input_events'][0]['data']['child_image'] = 'forbidden'
    report = build_snapshot([row], tmp_path, cutoff=CUTOFF, kind='director_sft')
    assert report['rejection_reasons']['post_action_content_in_director_input'] == 1
    row['input_events'][0]['data'].pop('child_image')
    report = build_snapshot([row], tmp_path, cutoff=CUTOFF, kind='director_sft')
    assert report['eligible_count'] == 1  # Includes restraint, still too few groups to train.


def test_snapshot_hash_tampering_fails_and_no_missing_db_creation(tmp_path):
    assert review_events(tmp_path) == []
    assert not (tmp_path / 'reviews.sqlite3').exists()
    report = build_snapshot([event(i) for i in range(5)], tmp_path, cutoff=CUTOFF)
    (tmp_path / report['snapshot_id'] / 'train.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_snapshot(report['snapshot_path'])


def test_action_preferences_not_causal_from_trajectory_or_confounded(tmp_path):
    row = event(kind='action_preferences', target={'same_prior_state_sha256': 'state', 'matched_execution': True,
                'actions': [{'operation': 'stop'}, {'operation': 'edit'}]},
                outcome={'realized': True, 'preference_scope': 'whole_trajectory'})
    report = build_snapshot([row], tmp_path, cutoff=CUTOFF, kind='action_preferences')
    assert report['rejection_reasons']['missing_outcomes_or_noncausal_trajectory_preference'] == 1


def test_retrieval_requires_complete_disjoint_candidate_set(tmp_path):
    row = event(kind='retrieval', target={'query': 'fixture', 'candidate_ids': ['a', 'b'], 'chosen_ids': ['a'], 'rejected_ids': []}, outcome={'realized': True})
    report = build_snapshot([row], tmp_path, cutoff=CUTOFF, kind='retrieval')
    assert report['rejection_reasons']['invalid_retrieval_candidate_partition'] == 1


def test_test_review_event_cannot_be_retrieved_into_development_inputs(tmp_path):
    rows = [event(i) for i in range(5)]
    assignments = grouped_assignments([LearningEvent.model_validate(row) for row in rows])
    test_row = next(row for row in rows if assignments[row['event_id']]['split'] == 'test')
    train_row = next(row for row in rows if assignments[row['event_id']]['split'] == 'train')
    train_row['input_events'][0]['event_id'] = test_row['source_event_ids'][0]
    report = build_snapshot(rows, tmp_path, cutoff=CUTOFF)
    assert report['rejection_reasons']['heldout_review_or_event_in_input'] == 1


def test_ab_order_position_mismatch_rejected(tmp_path):
    row = event()
    row['presented_images'][0]['position'] = 'B'
    report = build_snapshot([row], tmp_path, cutoff=CUTOFF)
    assert report['rejection_reasons']['presented_order_position_mismatch'] == 1
