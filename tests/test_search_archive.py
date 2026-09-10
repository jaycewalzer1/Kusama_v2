"""Offline archive mechanics, not artistic quality or live CLIP evidence."""
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pytest

from kusama.evidence import digest, file_hash, immutable_json
from kusama.search_archive import Archive, fit_archive


def index_fixture(tmp_path, *, sealed=False, include_crop=True):
    root = tmp_path / 'index'
    root.mkdir()
    docs = []
    vectors = []
    for i in range(12):
        vector = np.zeros(512, dtype=np.float32)
        vector[i] = 1
        vectors.append(vector)
        docs.append({'id': f'reference-{i}', 'reference_id': f'reference-{i}', 'region_id': None,
                     'scope': 'whole', 'source_family': f'family-{i}', 'split_group': f'family-{i}',
                     'sha256': f'source-hash-{i}', 'reference': {'rights': {'image_display': {'status': 'allowed'}}},
                     'sealed': sealed and i == 0})
    if include_crop:
        docs.append({**docs[0], 'id': 'crop-0', 'region_id': 'crop-0', 'scope': 'part'})
        vectors.append(vectors[0])
    immutable_json(root / 'documents.json', {'documents': docs, 'claims': []})
    np.save(root / 'embeddings.npy', np.asarray(vectors), allow_pickle=False)
    identity = digest({'version': 'clip512-rrf60-v1', 'clip_revision': 'fixture-revision', 'documents': docs, 'claims': []})
    immutable_json(root / 'manifest.json', {'index_id': identity[:20], 'version': 'clip512-rrf60-v1', 'source_identity_sha256': identity,
                   'encoder': {'model': 'fixture-only-CLIP', 'revision': 'fixture-revision'},
                   'files': {name: file_hash(root / name) for name in ('documents.json', 'embeddings.npy')}})
    return root.rename(tmp_path / identity[:20])


def archive_fixture(tmp_path, *, judge_version='unavailable'):
    index = index_fixture(tmp_path)
    output = tmp_path / 'archive'
    fit_archive(index, output, centroid_count=4)
    return Archive(output, brief_id='fixture-brief', reviewer_context='fixture-context',
                   judge_version=judge_version, anchor_version='fixture-anchor')


def candidate(tmp_path, i, **extra):
    path = tmp_path / f'image-{i}.png'
    Image.new('RGB', (4, 4), color=(i * 11 % 255, 33, 79)).save(path)
    return {'candidate_id': f'candidate-{i}', 'path': str(path), 'sha256': file_hash(path), 'realized': True, **extra}


def vector(i=0):
    result = np.zeros(512, dtype=np.float32)
    result[i] = 1
    return result


def test_cvt_fit_once_full_sources_only_and_source_exclusions(tmp_path):
    index = index_fixture(tmp_path)
    manifest = fit_archive(index, tmp_path / 'archive', centroid_count=32, excluded_source_families=['family-0'])
    assert manifest['source_vector_count'] == 11
    assert manifest['centroid_count'] == 11
    assert 'reference-0' not in manifest['source_ids']
    assert 'crop-0' not in manifest['source_ids']
    assert fit_archive(index, tmp_path / 'archive', centroid_count=32, excluded_source_families=['family-0']) == manifest
    with pytest.raises(ValueError, match='Immutable'):
        fit_archive(index, tmp_path / 'archive', centroid_count=2)


def test_sealed_pool_and_candidate_rejected(tmp_path):
    index = index_fixture(tmp_path, sealed=True)
    with pytest.raises(ValueError, match='Sealed'):
        fit_archive(index, tmp_path / 'archive')
    other = tmp_path / 'other'
    other.mkdir()
    archive = archive_fixture(other)
    with pytest.raises(ValueError, match='Sealed'):
        archive.add(candidate(tmp_path, 0, partition='sealed_test'), vector())


def test_all_realized_candidates_retained_without_quality_proxy(tmp_path):
    archive = archive_fixture(tmp_path)
    for i in range(14):
        archive.add(candidate(tmp_path, i), vector(i % 12))
    pool = archive.active_pool()
    assert len(archive.candidates()) == 14
    assert pool['underlying_candidate_count'] == 14
    assert pool['favored'] == []
    assert pool['quality_available'] is False
    assert len(pool['protected']) == 2
    assert len(pool['different']) <= 2
    assert len(pool['exploratory_fallback']) <= 4
    assert all(row['quality'] is None for row in archive.candidates())
    assert any(event['record'].get('selection_basis') == 'seeded_exploratory_fallback_no_quality_claim' for event in archive.events())
    with sqlite3.connect(archive.database) as db:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('DELETE FROM candidates')


def test_compatible_quality_replaces_niche_and_invalid_quality_fails(tmp_path):
    archive = archive_fixture(tmp_path, judge_version='fixture-judge-v1')
    metadata = {'judge_version': 'fixture-judge-v1', 'reviewer_context': 'fixture-context', 'dimension': 'finished_quality'}
    archive.add(candidate(tmp_path, 0), vector(), quality=.2, quality_origin='model_prediction', quality_metadata=metadata)
    archive.add(candidate(tmp_path, 1), vector(), quality=.8, quality_origin='model_prediction', quality_metadata=metadata)
    archive.add(candidate(tmp_path, 2), vector(), quality=.1, quality_origin='model_prediction', quality_metadata=metadata)
    events = [event for event in archive.events() if event['event_type'] == 'candidate_retained']
    assert [event['record']['is_new_elite'] for event in events] == [True, True, False]
    assert len(archive.candidates()) == 3
    with pytest.raises(ValueError, match='incompatible'):
        archive.add(candidate(tmp_path, 3), vector(), quality=.9, quality_origin='model_prediction', quality_metadata={**metadata, 'judge_version': 'different'})
    with pytest.raises(ValueError, match='incompatible'):
        archive.add(candidate(tmp_path, 4, anchor_version='different-anchor'), vector())
    with pytest.raises(ValueError, match='Descriptor version'):
        Archive(archive.root, brief_id='x', reviewer_context='x', anchor_version='x', descriptor_version='different')


def test_protected_slots_not_reclaimed_by_later_favorites(tmp_path):
    archive = archive_fixture(tmp_path, judge_version='fixture-judge-v1')
    metadata = {'judge_version': 'fixture-judge-v1', 'reviewer_context': 'fixture-context', 'dimension': 'finished_quality'}
    for i in range(3):
        archive.add(candidate(tmp_path, i), vector(i), protected=True)
    initial = archive.active_pool()
    assert [row['candidate_id'] for row in initial['protected']] == ['candidate-0', 'candidate-1']
    assert len(archive.candidates()) == 3  # Third protection request retained, reservation denied.
    assert any(event['event_type'] == 'protection_capacity_exhausted' for event in archive.events())
    for i in range(3, 12):
        archive.add(candidate(tmp_path, i), vector(i), quality=float(i), quality_origin='model_prediction', quality_metadata=metadata)
    final = archive.active_pool()
    assert final['protected'] == initial['protected']
    assert len(final['favored']) <= 4
    chosen = [row['candidate_id'] for key in ('favored', 'exploratory_fallback', 'different', 'protected') for row in final[key]]
    assert len(chosen) == len(set(chosen)) <= 8


def test_archive_scope_and_idempotency(tmp_path):
    archive = archive_fixture(tmp_path)
    one = candidate(tmp_path, 0)
    assert archive.add(one, vector()) == archive.add(one, vector())
    assert len(archive.candidates()) == 1
    with pytest.raises(ValueError, match='different immutable'):
        archive.add(one, vector(1))
    independent = Archive(archive.root, brief_id='other-brief', reviewer_context='fixture-context', anchor_version='fixture-anchor')
    assert independent.candidates() == []
    with pytest.raises(ValueError, match='actually realized'):
        archive.add({**one, 'realized': False}, vector())
    with pytest.raises(ValueError, match='available compatible judge'):
        archive.add(candidate(tmp_path, 2), vector(), quality=.9, quality_origin='model_prediction')
