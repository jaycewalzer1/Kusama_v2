"""Small frozen CVT archive; descriptors measure coverage, never artistic quality."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np

from .evidence import digest, file_hash, immutable_json
from .retrieval import RetrievalIndex, normalized

ARCHIVE_VERSION = 'spherical-cvt-map-elites-v1'


def _sealed(value):
    if isinstance(value, dict):
        if value.get('sealed') is True or value.get('partition') in ('sealed', 'sealed_test', 'test', 'heldout') or value.get('split') in ('sealed', 'sealed_test', 'test', 'heldout'):
            return True
        return any(_sealed(v) for v in value.values())
    if isinstance(value, list):
        return any(_sealed(v) for v in value)
    return False


def fit_archive(index_dir, output, *, centroid_count=32, seed=23, excluded_source_families=(), allowed_reference_ids=None):
    """Fit once on eligible full references. Crops/experiment outcomes never fit niches."""
    if not 1 <= centroid_count <= 128:
        raise ValueError('A modest archive needs 1–128 requested centroids')
    index = RetrievalIndex(index_dir)
    if _sealed(index.manifest) or any(_sealed(doc) for doc in index.documents):
        raise ValueError('Sealed or test pools cannot fit development archive centroids')
    selected = [(i, doc) for i, doc in enumerate(index.documents) if doc['scope'] == 'whole'
                and doc['source_family'] not in excluded_source_families
                and (allowed_reference_ids is None or doc['reference_id'] in allowed_reference_ids)]
    if not selected:
        raise ValueError('No permitted full reference vectors available')
    if any(doc.get('reference', {}).get('rights', {}).get('image_display', {}).get('status') != 'allowed' for _, doc in selected):
        raise ValueError('Archive fitting requires explicit image-display permission')
    vectors = normalized(index.vectors[[i for i, _ in selected]])
    # Duplicate vectors do not inflate the number of distinct centroid supports.
    unique = np.unique(vectors, axis=0)
    count = min(centroid_count, len(unique))
    rng = np.random.default_rng(seed)
    centroids = unique[rng.choice(len(unique), count, replace=False)].copy()
    for iteration in range(100):
        nearest = (vectors @ centroids.T).argmax(axis=1)
        updated = centroids.copy()
        for niche in range(count):
            members = vectors[nearest == niche]
            if len(members):
                mean = members.mean(axis=0)
                if np.linalg.norm(mean) > 1e-12:
                    updated[niche] = normalized(mean)
        if np.allclose(updated, centroids, atol=1e-6):
            break
        centroids = updated
    root = Path(output)
    encoder = index.manifest['encoder']
    descriptor = {'model': encoder['model'], 'revision': encoder['revision'], 'dimensions': 512,
                  'normalized': True, 'index_version': index.manifest['version']}
    manifest = {'schema_version': 1, 'archive_version': ARCHIVE_VERSION, 'index_id': index.manifest['index_id'],
                'index_manifest_sha256': file_hash(Path(index_dir) / 'manifest.json'),
                'descriptor': descriptor, 'descriptor_version': digest(descriptor), 'seed': seed,
                'requested_centroids': centroid_count, 'centroid_count': count, 'source_vector_count': len(selected),
                'centroid_training': 'spherical k-means on permitted whole reference vectors; fixed before candidate search',
                'source_ids': [doc['reference_id'] for _, doc in selected],
                'source_hashes': {doc['reference_id']: doc['sha256'] for _, doc in selected},
                'source_families': sorted({doc['source_family'] for _, doc in selected}),
                'source_split_groups': sorted({doc['split_group'] for _, doc in selected}),
                'exclusions': {'source_families': sorted(excluded_source_families),
                               'allowed_reference_ids': sorted(allowed_reference_ids) if allowed_reference_ids is not None else None},
                'centroids_sha256': digest(centroids.tolist()),
                'meaning': 'Niches are coverage under this descriptor, not human quality or historical novelty.'}
    immutable_json(root / 'centroids.json', centroids.tolist())
    immutable_json(root / 'manifest.json', manifest)
    return manifest


class Archive:
    """Append-only candidates and retention events scoped to one comparison context."""
    def __init__(self, output, *, brief_id, reviewer_context, judge_version='unavailable', anchor_version,
                 descriptor_version=None):
        self.root = Path(output).resolve()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.centroids = np.asarray(json.loads((self.root / 'centroids.json').read_text()), dtype=np.float32)
        if digest(self.centroids.tolist()) != self.manifest['centroids_sha256']:
            raise ValueError('Frozen centroid hashes changed')
        if not brief_id or not reviewer_context or not anchor_version:
            raise ValueError('Brief, reviewer context, and anchor version are required')
        if descriptor_version is not None and descriptor_version != self.manifest['descriptor_version']:
            raise ValueError('Descriptor version is incompatible with archive centroids')
        self.context = {'brief_id': brief_id, 'reviewer_context': reviewer_context,
                        'judge_version': judge_version, 'anchor_version': anchor_version,
                        'descriptor_version': self.manifest['descriptor_version'], 'archive_version': ARCHIVE_VERSION}
        self.context_id = digest(self.context)
        self.database = self.root / 'archive.sqlite3'
        with closing(self._connect()) as db, db:
            db.executescript('''
              CREATE TABLE IF NOT EXISTS contexts(id TEXT PRIMARY KEY, specification TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS candidates(sequence INTEGER PRIMARY KEY, context_id TEXT NOT NULL,
                candidate_id TEXT NOT NULL, record TEXT NOT NULL, UNIQUE(context_id,candidate_id));
              CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY, context_id TEXT NOT NULL,
                event_type TEXT NOT NULL, candidate_id TEXT, record TEXT NOT NULL);
            ''')
            for table in ('contexts', 'candidates', 'events'):
                for operation in ('UPDATE', 'DELETE'):
                    db.execute(f"CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'archive evidence is append-only'); END")
            db.execute('INSERT OR IGNORE INTO contexts VALUES (?,?)', (self.context_id, json.dumps(self.context, sort_keys=True)))

    def _connect(self):
        db = sqlite3.connect(self.database, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _rows(self, db):
        return [json.loads(row[0]) for row in db.execute('SELECT record FROM candidates WHERE context_id=? ORDER BY sequence', (self.context_id,))]

    def candidates(self):
        with closing(self._connect()) as db:
            return self._rows(db)

    def _priority(self, candidate_id):
        return digest([self.manifest['seed'], self.context_id, candidate_id, 'declared-exploratory-retention'])

    def _elites(self, rows):
        elites = {}
        for row in rows:
            old = elites.get(row['niche'])
            if old is None:
                elites[row['niche']] = row
            elif row['quality'] is not None and old['quality'] is None:
                elites[row['niche']] = row
            elif row['quality'] is not None and old['quality'] is not None and row['quality_origin'] == old['quality_origin']:
                if row['quality'] > old['quality']:
                    elites[row['niche']] = row
            elif row['quality'] is None and old['quality'] is None and self._priority(row['candidate_id']) < self._priority(old['candidate_id']):
                elites[row['niche']] = row
        return elites

    def _protected(self, db):
        return [row[0] for row in db.execute("SELECT candidate_id FROM events WHERE context_id=? AND event_type='protected_slot_assigned' ORDER BY sequence", (self.context_id,))]

    def _event(self, db, event_type, candidate_id, record):
        db.execute('INSERT INTO events(context_id,event_type,candidate_id,record) VALUES(?,?,?,?)',
                   (self.context_id, event_type, candidate_id, json.dumps(record, sort_keys=True)))

    def add(self, candidate, vector, *, quality=None, quality_origin='unavailable', quality_metadata=None, protected=False):
        """Candidate requires id/path/sha256 and realized=True; all bytes are verified.

        Quality metadata must carry matching judge_version/reviewer_context/dimension.
        Missing quality enables explicitly seeded exploratory retention, not a proxy.
        """
        if not candidate.get('realized') or not candidate.get('candidate_id'):
            raise ValueError('Only an identified, actually realized candidate can enter the archive')
        if _sealed(candidate):
            raise ValueError('Sealed/test outcomes cannot enter a development search archive')
        for key in self.context:
            if key in candidate and candidate[key] != self.context[key]:
                raise ValueError(f'Candidate {key} incompatible with archive context')
        path = Path(candidate['path']).resolve()
        if file_hash(path) != candidate['sha256']:
            raise ValueError('Candidate image hash does not match realized bytes')
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        vector = normalized(vector)
        if vector.ndim != 1:
            raise ValueError('One candidate requires one descriptor')
        if quality is None:
            if quality_origin != 'unavailable':
                raise ValueError('Absent quality must have unavailable origin')
        else:
            if not np.isfinite(quality) or quality_origin not in ('human_judgment', 'model_prediction'):
                raise ValueError('Quality requires a finite value and explicit situated origin')
            if self.context['judge_version'] == 'unavailable' or not quality_metadata:
                raise ValueError('Quality requires an available compatible judge and metadata')
            if quality_metadata.get('judge_version') != self.context['judge_version'] or quality_metadata.get('reviewer_context') != self.context['reviewer_context'] or quality_metadata.get('dimension') != 'finished_quality':
                raise ValueError('Quality version/context/dimension incompatible')
        record = {'candidate_id': candidate['candidate_id'], 'candidate': {**candidate, 'path': str(path)},
                  'context': self.context, 'descriptor': vector.tolist(), 'niche': int((self.centroids @ vector).argmax()),
                  'quality': float(quality) if quality is not None else None, 'quality_origin': quality_origin,
                  'quality_metadata': quality_metadata, 'protected_requested': bool(protected)}
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT record FROM candidates WHERE context_id=? AND candidate_id=?', (self.context_id, candidate['candidate_id'])).fetchone()
            if previous:
                if json.loads(previous[0]) != record:
                    raise ValueError('Candidate ID already has different immutable evidence')
                return json.loads(previous[0])
            rows = self._rows(db)
            before = self._elites(rows).get(record['niche'])
            after = self._elites(rows + [record])[record['niche']]
            db.execute('INSERT INTO candidates(context_id,candidate_id,record) VALUES(?,?,?)',
                       (self.context_id, candidate['candidate_id'], json.dumps(record, sort_keys=True)))
            self._event(db, 'candidate_retained', candidate['candidate_id'],
                        {'niche': record['niche'], 'is_new_elite': after['candidate_id'] == record['candidate_id'],
                         'previous_elite': before['candidate_id'] if before else None,
                         'selection_basis': 'compatible_situated_quality' if quality is not None else 'seeded_exploratory_fallback_no_quality_claim'})
            if protected:
                assigned = self._protected(db)
                self._event(db, 'protected_slot_assigned' if len(assigned) < 2 else 'protection_capacity_exhausted',
                            candidate['candidate_id'], {'reserved_slots': 2, 'reclaim_allowed': False})
        return record

    def active_pool(self):
        """At most 4 favored/fallback + 2 different + 2 protected; no slot stealing."""
        with closing(self._connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            rows = self._rows(db)
            by_id = {row['candidate_id']: row for row in rows}
            protected_ids = self._protected(db)
            # Assign reserved exploration slots once; future quality cannot reclaim them.
            possibilities = sorted((row for row in rows if row['candidate_id'] not in protected_ids), key=lambda row: self._priority(row['candidate_id']))
            for row in possibilities[:max(0, 2-len(protected_ids))]:
                protected_ids.append(row['candidate_id'])
                self._event(db, 'protected_slot_assigned', row['candidate_id'], {'reserved_slots': 2, 'reclaim_allowed': False, 'basis': 'seeded_exploration'})
            protected = [by_id[identifier] for identifier in protected_ids]
            elites = list(self._elites(rows).values())
            remaining = [row for row in elites if row['candidate_id'] not in protected_ids]
            # Scores with different attribution are never mixed in one favored ranking.
            scored_origins = {row['quality_origin'] for row in remaining if row['quality'] is not None}
            compatible_origin = next(iter(scored_origins)) if len(scored_origins) == 1 else None
            favored = sorted((row for row in remaining if row['quality'] is not None and row['quality_origin'] == compatible_origin), key=lambda row: (-row['quality'], row['candidate_id']))[:4]
            chosen = {row['candidate_id'] for row in protected + favored}
            fallback = []
            if not favored:
                fallback = sorted((row for row in remaining if row['candidate_id'] not in chosen), key=lambda row: self._priority(row['candidate_id']))[:4]
                chosen.update(row['candidate_id'] for row in fallback)
            different = []
            pool = [row for row in rows if row['candidate_id'] not in chosen]
            while pool and len(different) < 2:
                anchors = protected + favored + fallback + different
                def distance(row):
                    return min(1-float(np.dot(row['descriptor'], anchor['descriptor'])) for anchor in anchors) if anchors else 0
                row = min(pool, key=lambda row: (-distance(row), self._priority(row['candidate_id'])))
                different.append(row)
                chosen.add(row['candidate_id'])
                pool = [item for item in pool if item['candidate_id'] != row['candidate_id']]
            compact = lambda items: [{'candidate_id': row['candidate_id'], 'niche': row['niche'],
                                      'quality': row['quality'], 'quality_origin': row['quality_origin'],
                                      'path': row['candidate']['path'], 'sha256': row['candidate']['sha256']} for row in items]
            result = {'context': self.context, 'context_id': self.context_id, 'favored': compact(favored),
                      'exploratory_fallback': compact(fallback), 'different': compact(different), 'protected': compact(protected),
                      'underlying_candidate_count': len(rows), 'occupied_niches': len(elites),
                      'reserved_capacity': {'favored_or_exploratory': 4, 'different': 2, 'protected': 2},
                      'quality_available': bool(favored), 'mixed_quality_origins': len(scored_origins) > 1,
                      'policy': 'Four compatible-quality favorites only when available, otherwise seeded exploration; two descriptor-different and two irrevocably protected slots. Coverage is not quality.'}
            self._event(db, 'active_pool_selected', None, result)
        return result

    def events(self):
        with closing(self._connect()) as db:
            return [{'sequence': row['sequence'], 'event_type': row['event_type'], 'candidate_id': row['candidate_id'],
                     'record': json.loads(row['record'])} for row in db.execute('SELECT * FROM events WHERE context_id=? ORDER BY sequence', (self.context_id,))]
