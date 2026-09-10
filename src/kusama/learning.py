"""Human-grounded CLIP selector; no fabricated checkpoints."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .datasets import load_snapshot
from .evidence import file_hash, immutable_json

LABELS = ('A', 'B', 'tie', 'unsure')
REVERSE = {'A': 'B', 'B': 'A', 'tie': 'tie', 'unsure': 'unsure'}


def pair_features(e_a, e_b):
    a, b = np.asarray(e_a, dtype=np.float64), np.asarray(e_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('CLIP embeddings require matching finite vectors')
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        raise ValueError('CLIP embeddings must be nonzero')
    delta = a / np.linalg.norm(a) - b / np.linalg.norm(b)
    return np.concatenate((delta, np.abs(delta)))


def reversed_presentation(row):
    row = deepcopy(row)
    row['presented_images'] = [dict(image, position=position) for position, image in zip(('A', 'B'), reversed(row['presented_images']))]
    for event in row.get('model_input', []):
        if isinstance(event, dict) and 'candidate_order' in event:
            event['candidate_order'] = deepcopy(row['presented_images'])
    row['label'] = REVERSE[row['label']]
    row['augmentation_of'] = row['event_id']
    # Keep event and group identity: reversal is not an independent judgment.
    return row


def softmax(values):
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def classification_metrics(probabilities, labels):
    probabilities = np.asarray(probabilities)
    if not len(labels):
        return {'count': 0, 'log_loss': None, 'brier': None, 'accuracy': None}
    targets = np.asarray([LABELS.index(label) for label in labels])
    truth = np.eye(len(LABELS))[targets]
    return {'count': len(labels), 'log_loss': float(-np.log(np.maximum(probabilities[np.arange(len(labels)), targets], 1e-12)).mean()),
            'brier': float(np.square(probabilities-truth).sum(axis=1).mean()),
            'accuracy': float((probabilities.argmax(axis=1) == targets).mean())}


def _fit_logistic(x, labels, regularization, steps=1500):
    """Full-batch convex multinomial objective; intercept is not L2 penalized."""
    y = np.eye(len(LABELS))[[LABELS.index(label) for label in labels]]
    weights = np.zeros((x.shape[1], len(LABELS)))
    intercept = np.zeros(len(LABELS))
    # Lipschitz-safe step for softmax cross entropy and the L2 penalty.
    lipschitz = .5 * np.linalg.norm(np.column_stack([x, np.ones(len(x))]), ord=2) ** 2 / len(x) + regularization
    lr = 1 / max(lipschitz, 1e-6)
    for _ in range(steps):
        error = softmax(x @ weights + intercept) - y
        grad = x.T @ error / len(x) + regularization * weights
        bias_grad = error.mean(axis=0)
        weights -= lr * grad
        intercept -= lr * bias_grad
        if max(float(np.abs(grad).max()), float(np.abs(bias_grad).max())) < 1e-7:
            break
    return weights, intercept


def _counts(splits):
    return {split: {'rows': len(rows), 'independent_groups': len({r['split_group'] for r in rows}),
                    'human_review_events': len({event for r in rows for event in r['source_event_ids']})}
            for split, rows in splits.items()}


def _embedding_reader(cache):
    value = json.loads(Path(cache).read_text())
    if not all(value.get(key) for key in ('model_id', 'revision', 'preprocessing', 'embeddings')) or value.get('representation') != 'CLIP':
        raise ValueError('Embedding cache requires CLIP identity, pinned revision, preprocessing and image-hash keys')
    def features(row):
        return pair_features(*[value['embeddings'][image['sha256']] for image in row['presented_images']])
    return value, features


def train_selector(snapshot, output, *, embedding_cache=None, execute=False):
    if (Path(output) / 'result.json').exists():
        raise ValueError('Training output already has an immutable result; choose a new output directory')
    manifest, splits = load_snapshot(snapshot)
    if manifest['kind'] not in ('selector', 'judge_sft'):
        raise ValueError('Selector requires situated pairwise human labels')
    result = {'component': 'selector', 'snapshot_id': manifest['snapshot_id'], 'counts': _counts(splits),
              'status': 'insufficient_data', 'checkpoint': None, 'personalizations': {},
              'claims': 'No learned artistic advantage or test evaluation is implied.'}
    reviewers = sorted({(r['reviewer'], r['target'].get('dimension', 'unspecified')) for rows in splits.values() for r in rows})
    eligible = []
    for reviewer, dimension in reviewers:
        selected = {split: [r for r in rows if r['reviewer'] == reviewer and r['target'].get('dimension', 'unspecified') == dimension] for split, rows in splits.items()}
        train, dev = selected['train'], selected['development']
        classes = Counter(row['label'] for row in train)
        key = f'{reviewer}:{dimension}'
        if len({r['split_group'] for r in train}) < 3 or len({r['split_group'] for r in dev}) < 1 or len(classes) < 2:
            result['personalizations'][key] = {'status': 'unsupported_personalization', 'counts': _counts(selected),
                                               'constant_baseline': max(classes, key=classes.get) if classes else None,
                                               'missing_requirements': ['at least three train groups, one development group, and two original training classes']}
        else:
            eligible.append((key, train, dev))
    if not eligible:
        result['missing_requirements'] = ['genuine reviewer labels in at least five independent groups',
                                          'per reviewer/dimension: three train groups, one development group, two original label classes']
        immutable_json(Path(output) / 'result.json', result)
        return result
    if embedding_cache is None:
        result.update(status='capability_unavailable', missing_requirements=['pinned CLIP embedding cache keyed by reviewed image SHA-256'])
        immutable_json(Path(output) / 'result.json', result)
        return result
    descriptor, features = _embedding_reader(embedding_cache)
    if not execute:
        result.update(status='ready_not_executed', missing_requirements=['explicit --execute'], eligible_personalizations=len(eligible))
        immutable_json(Path(output) / 'result.json', result)
        return result
    models = {}
    for key, train, dev in eligible:
        augmented = train + [reversed_presentation(row) for row in train]
        x = np.asarray([features(row) for row in augmented])
        labels = [row['label'] for row in augmented]
        mean, scale = x.mean(axis=0), x.std(axis=0)
        scale[scale < 1e-8] = 1
        x = (x-mean)/scale
        dev_x = (np.asarray([features(row) for row in dev])-mean)/scale
        scored = []
        for l2 in (.001, .01, .1, 1.0, 10.0):
            weights, intercept = _fit_logistic(x, labels, l2)
            probabilities = softmax(dev_x @ weights + intercept)
            # Each brief/group has equal development weight despite row counts.
            losses = []
            for group in sorted({row['split_group'] for row in dev}):
                indexes = [i for i, row in enumerate(dev) if row['split_group'] == group]
                losses.append(classification_metrics(probabilities[indexes], [dev[i]['label'] for i in indexes])['log_loss'])
            scored.append((float(np.mean(losses)), l2, weights, intercept))
        loss, l2, weights, intercept = min(scored, key=lambda x: (x[0], x[1]))
        probabilities = softmax(dev_x @ weights + intercept)
        models[key] = {'mean': mean.tolist(), 'scale': scale.tolist(), 'weights': weights.tolist(), 'intercept': intercept.tolist(),
                       'regularization': l2, 'labels': LABELS}
        reverse_x = (np.asarray([features(reversed_presentation(row)) for row in dev])-mean)/scale
        reverse_p = softmax(reverse_x @ weights + intercept)[:, [1, 0, 2, 3]]
        favorites = [i for i, row in enumerate(dev) if row['label'] in ('A', 'B')]
        result['personalizations'][key] = {'status': 'trained', 'original_training_rows': len(train), 'augmented_rows': len(augmented),
                                          'original_review_events': len({eid for row in train for eid in row['source_event_ids']}),
                                          'independent_training_groups': len({row['split_group'] for row in train}),
                                          'grouped_development_log_loss': loss, 'regularization': l2,
                                          'development_metrics': classification_metrics(probabilities, [r['label'] for r in dev]),
                                          'order_reversal_mean_probability_error': float(np.abs(probabilities-reverse_p).mean()),
                                          'human_favorite_retention': float(np.mean([LABELS[probabilities[i].argmax()] == dev[i]['label'] for i in favorites])) if favorites else None,
                                          'retention_scope': 'pairwise development choices; not a six-candidate pool metric'}
    checkpoint = {'schema_version': 1, 'component': 'selector', 'snapshot_id': manifest['snapshot_id'],
                  'clip_descriptor': {k: descriptor[k] for k in ('model_id', 'revision', 'preprocessing', 'representation')},
                  'embedding_cache_sha256': file_hash(embedding_cache), 'models': models}
    path = immutable_json(Path(output) / 'selector.json', checkpoint)
    result.update(status='trained', checkpoint=str(path.resolve()), test_evaluated=False)
    immutable_json(Path(output) / 'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('component', choices=('selector',))
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    result = train_selector(args.snapshot, args.output, embedding_cache=args.embedding_cache, execute=args.execute)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] in ('trained', 'ready_not_executed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
