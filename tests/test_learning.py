"""CLIP selector mechanics fixtures; not evidence of artistic improvement."""
import json

import numpy as np
import pytest

from kusama.datasets import build_snapshot
from kusama.learning import (LABELS, classification_metrics, pair_features,
                             reversed_presentation, train_selector)
from test_datasets import CUTOFF, event


def test_normalized_clip_difference_and_order_mapping():
    feature = pair_features([3, 0], [0, 7])
    assert np.allclose(feature, [1, -1, 1, 1])
    reverse = pair_features([0, 7], [3, 0])
    assert np.allclose(reverse[:2], -feature[:2])
    assert np.allclose(reverse[2:], feature[2:])
    row = event()
    for label in LABELS:
        row['label'] = label
        reversed_row = reversed_presentation(row)
        assert reversed_presentation(reversed_row)['label'] == label
        assert reversed_row['event_id'] == row['event_id']
    with pytest.raises(ValueError, match='nonzero'):
        pair_features([0, 0], [1, 0])


def test_tie_and_abstention_metrics():
    metrics = classification_metrics([[0, 0, 0, 1], [0, 0, 1, 0]], ['unsure', 'tie'])
    assert metrics['brier'] == metrics['log_loss'] == 0


def test_empty_selector_exact_counts_and_no_checkpoint(tmp_path):
    report = build_snapshot([], tmp_path / 'datasets', cutoff=CUTOFF, kind='selector')
    result = train_selector(report['snapshot_path'], tmp_path / 'selector')
    assert result['status'] == 'insufficient_data'
    assert result['checkpoint'] is None
    assert result['counts']['train']['rows'] == 0
    assert not (tmp_path / 'selector' / 'selector.json').exists()


def test_one_class_does_not_manufacture_personalization_from_reversal(tmp_path):
    report = build_snapshot([event(i) for i in range(10)], tmp_path / 'datasets', cutoff=CUTOFF)
    result = train_selector(report['snapshot_path'], tmp_path / 'selector', execute=True)
    assert result['status'] == 'insufficient_data'
    assert result['personalizations']['fixture-reviewer:unspecified']['constant_baseline'] == 'A'
    assert result['checkpoint'] is None


def test_selector_real_solver_fixture_no_test_labels_loaded(tmp_path):
    rows = [event(i, label='A' if i % 2 == 0 else 'B') for i in range(15)]
    snapshot = build_snapshot(rows, tmp_path / 'datasets', cutoff=CUTOFF)
    embeddings = {}
    for row in rows:
        a, b = row['presented_images']
        embeddings[a['sha256']] = [1, 0] if row['label'] == 'A' else [0, 1]
        embeddings[b['sha256']] = [0, 1] if row['label'] == 'A' else [1, 0]
    cache = tmp_path / 'embeddings.json'
    cache.write_text(json.dumps({'representation': 'CLIP', 'model_id': 'fixture-only', 'revision': 'fixture',
                                 'preprocessing': 'test-mechanics-not-live-clip', 'embeddings': embeddings}))
    # A corrupt sealed file must never be read while selecting regularization.
    (tmp_path / 'datasets' / snapshot['snapshot_id'] / 'test.jsonl').write_text('DO NOT READ TEST LABELS')
    result = train_selector(snapshot['snapshot_path'], tmp_path / 'selector', embedding_cache=cache, execute=True)
    assert result['status'] == 'trained'
    metrics = result['personalizations']['fixture-reviewer:unspecified']
    assert metrics['development_metrics']['accuracy'] == 1
    assert metrics['augmented_rows'] == 2 * metrics['original_training_rows']
    assert result['test_evaluated'] is False
