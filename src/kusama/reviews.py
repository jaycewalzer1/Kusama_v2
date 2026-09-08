"""Local human comparisons with immutable candidates and append-only judgments.

These records describe human preference, independently of technical validation.
Reading the queue never creates a database or manufactures a review.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import secrets
import sqlite3
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import sha256


Choice = Literal['A', 'B', 'tie', 'unsure', 'incomparable', 'decline']


class ReviewError(ValueError):
    pass


class ReviewNotFound(ReviewError):
    pass


class ReviewConflict(ReviewError):
    pass


class Answer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    choice: Choice | None
    confidence: Literal['low', 'medium', 'high'] | None = None


class SituatedAnswers(BaseModel):
    model_config = ConfigDict(extra='forbid')
    A: Literal['Y', 'N', 'U', 'decline'] | None = None
    B: Literal['Y', 'N', 'U', 'decline'] | None = None


class Exposure(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prior_exposure: Literal['none', 'images', 'statement', 'images_and_statement', 'unknown'] = 'unknown'
    context_revealed: bool = False
    overview_opened: bool = False
    duration_seconds: float | None = Field(default=None, ge=0, le=86400, allow_inf_nan=False)
    duration_source: Literal['browser_elapsed', 'reviewer_reported', 'unrecorded'] = 'unrecorded'


class Presentation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    condition: Literal['statement_free', 'supplied_context'] = 'statement_free'
    supplied_context: str = Field(default='', max_length=8000)
    context_author: str | None = Field(default=None, max_length=200)
    viewing_conditions: str = Field(default='Screen; actual device and environment not independently measured', max_length=2000)

    @model_validator(mode='after')
    def truthful_context(self):
        if self.condition == 'supplied_context' and (not self.supplied_context or not self.context_author):
            raise ValueError('Supplied context requires text and attributed author')
        if self.condition == 'statement_free' and self.supplied_context:
            raise ValueError('Statement-free assignments cannot contain a statement')
        return self


class Vote(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    reviewer: str = Field(min_length=1, max_length=200)
    assignment_id: str = Field(min_length=1, max_length=64)
    submission_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')
    finished_quality: Answer
    development_potential: Answer
    notes: str = Field(default='', max_length=8000)
    open_reading: str | None = Field(default=None, max_length=8000)
    felt_response: str | None = Field(default=None, max_length=8000)
    readiness: SituatedAnswers | None = None
    development_willingness: SituatedAnswers | None = None
    exposure: Exposure | None = None


class Advancement(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    reviewer: str = Field(min_length=1, max_length=200)
    candidate_id: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=8000)
    submission_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')


class PoolChoice(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    reviewer: str = Field(min_length=1, max_length=200)
    assignment_id: str = Field(min_length=1, max_length=64)
    submission_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,100}$')
    choice: Literal['candidate', 'none', 'unsure', 'decline']
    candidate_id: str | None = None
    criterion: Literal['finished_quality', 'development_potential'] = 'finished_quality'
    notes: str = Field(default='', max_length=8000)
    exposure: Exposure | None = None

    @model_validator(mode='after')
    def candidate_or_none(self):
        if (self.choice == 'candidate') != (self.candidate_id is not None):
            raise ValueError('Only a candidate choice may include a candidate ID')
        return self


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise ReviewError('Identifiers require 1–100 letters, digits, underscores or hyphens')
    return value


def _text(value, maximum=8000):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise ReviewError(f'Text must contain 1–{maximum} characters')
    return value.strip()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _frozen_assignments(rows, pairs, pool_ids):
    """Validate a complete externally frozen schedule without inventing reviewers."""
    if not isinstance(rows, list) or not rows:
        raise ReviewError('Frozen assignments require a nonempty list')
    pair_members = {pair['id']: set(pair['candidate_ids']) for pair in pairs}
    normalized, seen, identifiers, reviewers = [], set(), set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ReviewError('Frozen assignment must be an object')
        kind = row.get('kind')
        required = {'reviewer_slot', 'assignment_id', 'kind'} | ({'pair_id', 'candidate_order'} if kind == 'parent_edit' else {'candidate_ids'})
        if kind not in ('parent_edit', 'six_candidate_or_none') or set(row) != required:
            raise ReviewError('Unknown or missing frozen assignment fields')
        reviewer = _text(row['reviewer_slot'], 200)
        identifier = row['assignment_id']
        if not isinstance(identifier, str) or not re.fullmatch(r'[a-f0-9]{64}', identifier) or identifier in identifiers:
            raise ReviewError('Frozen assignment IDs require distinct lowercase SHA256 identifiers')
        key = (reviewer, kind, row.get('pair_id'))
        if key in seen:
            raise ReviewError('Frozen reviewer slots must be unique per comparison or pool')
        if kind == 'parent_edit':
            order = row['candidate_order']
            if (row['pair_id'] not in pair_members or not isinstance(order, dict) or set(order) != {'A', 'B'}
                    or set(order.values()) != pair_members[row['pair_id']]):
                raise ReviewError('Frozen A/B order must match the registered pair')
        else:
            order = row['candidate_ids']
            if not isinstance(order, list) or len(order) != len(pool_ids) or set(order) != set(pool_ids):
                raise ReviewError('Frozen pool order must contain each designated candidate exactly once')
        normalized.append({**row, 'reviewer_slot': reviewer})
        seen.add(key); identifiers.add(identifier); reviewers.add(reviewer)
    expected = {(reviewer, 'parent_edit', pair) for reviewer in reviewers for pair in pair_members}
    expected |= {(reviewer, 'six_candidate_or_none', None) for reviewer in reviewers}
    if seen != expected:
        raise ReviewError('Every frozen reviewer slot requires all registered pairs and one pool assignment')
    return sorted(normalized, key=lambda row: (row['reviewer_slot'], row['kind'], row.get('pair_id', '')))


class ReviewStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.database = self.root / 'reviews.sqlite3'

    def _connect(self, *, write=False):
        if self.database.is_symlink():
            raise ReviewError('Review database must not be a symlink')
        if write:
            self.root.mkdir(parents=True, exist_ok=True)
            db = sqlite3.connect(self.database, timeout=10)
            db.executescript('''
                CREATE TABLE IF NOT EXISTS review_collections (
                    id TEXT PRIMARY KEY, specification TEXT NOT NULL, random_seed TEXT NOT NULL,
                    created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_votes (
                    sequence INTEGER PRIMARY KEY, submission_id TEXT UNIQUE NOT NULL,
                    collection_id TEXT NOT NULL, pair_id TEXT NOT NULL, reviewer TEXT NOT NULL,
                    request TEXT NOT NULL, record TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_advancements (
                    sequence INTEGER PRIMARY KEY, submission_id TEXT UNIQUE NOT NULL,
                    collection_id TEXT NOT NULL, reviewer TEXT NOT NULL,
                    request TEXT NOT NULL, record TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS review_pool_votes (
                    sequence INTEGER PRIMARY KEY, submission_id TEXT UNIQUE NOT NULL,
                    collection_id TEXT NOT NULL, reviewer TEXT NOT NULL,
                    request TEXT NOT NULL, record TEXT NOT NULL);
            ''')
            for table in ('review_collections', 'review_votes', 'review_advancements', 'review_pool_votes'):
                for operation in ('UPDATE', 'DELETE'):
                    db.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                        BEFORE {operation} ON {table} BEGIN
                        SELECT RAISE(ABORT, 'human review records are append-only'); END''')
            db.commit()
        else:
            if not self.database.is_file():
                raise ReviewNotFound('No review collections registered')
            db = sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    def _image(self, value):
        source = Path(value)
        path = source if source.is_absolute() else self.root / source
        # Reject directory symlinks too: a changed link must never retarget a candidate.
        if not path.is_absolute() or not path.is_relative_to(self.root):
            raise ReviewError('Review images must be inside the artifacts directory')
        if any(part.is_symlink() for part in [path, *path.parents] if part != self.root.parent):
            raise ReviewError('Review image paths must not contain symlinks')
        path = path.resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ReviewError('Review image is missing or outside the artifacts directory')
        if path.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp'):
            raise ReviewError('Review candidate must be a PNG, JPEG or WebP image')
        with Image.open(path) as image:
            image.verify()
        return path

    def register_collection(self, collection_id, *, title, brief, candidates, pairs=None, presentation=None,
                            pool_expected_count=None, pool_candidate_ids=None, frozen_assignments=None):
        """Freeze image hashes and a comparison plan; identical registration is idempotent.

        Candidate: id, path, label; optional direction_id, parent_candidate_id, studio.
        Pair: id, candidate_ids (exactly two distinct IDs), kind (alternatives/before_after).
        Omitted pairs produces all unordered pairs. No review is created here.
        Frozen assignments name actual reviewer slots and fix exact A/B and pool orders;
        when supplied, unassigned reviewers cannot open or submit these tasks.
        """
        _id(collection_id)
        if not 2 <= len(candidates) <= 100:
            raise ReviewError('A collection needs 2–100 candidates')
        records = []
        for candidate in candidates:
            allowed = {'id', 'path', 'label', 'direction_id', 'parent_candidate_id', 'studio'}
            if set(candidate) - allowed:
                raise ReviewError('Unknown candidate fields')
            path = self._image(candidate['path'])
            with Image.open(path) as image:
                size = list(image.size)
            record = {'id': _id(candidate['id']), 'path': str(path.relative_to(self.root)),
                      'label': _text(candidate.get('label', candidate['id']), 1000),
                      'sha256': sha256(path), 'size': size,
                      'direction_id': candidate.get('direction_id'),
                      'parent_candidate_id': candidate.get('parent_candidate_id')}
            if candidate.get('studio'):
                studio = candidate['studio']
                if set(studio) != {'project_id', 'episode_id', 'artifact_id'}:
                    raise ReviewError('Studio linkage requires project_id, episode_id and artifact_id')
                record['studio'] = {key: _text(value, 200) for key, value in studio.items()}
            records.append(record)
        ids = {candidate['id'] for candidate in records}
        if len(ids) != len(records):
            raise ReviewError('Candidate IDs must be unique')
        for candidate in records:
            parent = candidate['parent_candidate_id']
            if parent is not None and (parent not in ids or parent == candidate['id']):
                raise ReviewError('Parent candidate must identify another candidate in this collection')
        if pairs is None:
            pairs = [{'id': f'pair-{index + 1:03}', 'candidate_ids': [a['id'], b['id']],
                      'kind': 'alternatives'} for index, (a, b) in enumerate(combinations(records, 2))]
        if not 1 <= len(pairs) <= 300:
            raise ReviewError('A collection needs 1–300 planned comparisons')
        normalized_pairs, pair_ids, compared = [], set(), set()
        for pair in pairs:
            if set(pair) - {'id', 'candidate_ids', 'kind'}:
                raise ReviewError('Unknown comparison fields')
            pair_id = _id(pair['id'])
            members = pair['candidate_ids']
            if not isinstance(members, (list, tuple)) or len(members) != 2 or len(set(members)) != 2 or not set(members) <= ids:
                raise ReviewError('Each comparison requires two distinct registered candidates')
            kind = pair.get('kind', 'alternatives')
            if kind not in ('alternatives', 'before_after'):
                raise ReviewError('Unknown comparison kind')
            if pair_id in pair_ids or tuple(sorted(members)) in compared:
                raise ReviewError('Comparison IDs and candidate pairs must be unique')
            pair_ids.add(pair_id)
            compared.add(tuple(sorted(members)))
            normalized_pairs.append({'id': pair_id, 'candidate_ids': list(members), 'kind': kind})
        specification = {'schema_version': 1, 'id': collection_id, 'title': _text(title, 1000),
                         'brief': _text(brief), 'candidates': records, 'pairs': normalized_pairs}
        if presentation is not None:
            specification.update(schema_version=2, presentation=Presentation.model_validate(presentation).model_dump())
        if pool_candidate_ids is not None:
            if (not isinstance(pool_candidate_ids, (list, tuple)) or not pool_candidate_ids
                    or len(set(pool_candidate_ids)) != len(pool_candidate_ids)
                    or not set(pool_candidate_ids) <= ids):
                raise ReviewError('Pool candidate IDs must be a nonempty unique subset of registered candidates')
            specification['pool_candidate_ids'] = list(pool_candidate_ids)
        pool_count = len(pool_candidate_ids) if pool_candidate_ids is not None else len(records)
        if pool_expected_count is not None:
            if type(pool_expected_count) is not int or not pool_count <= pool_expected_count <= 100:
                raise ReviewError('Expected pool count must include every designated pool candidate and be at most 100')
            specification['pool_expected_count'] = pool_expected_count
        if frozen_assignments is not None:
            specification.update(schema_version=3, frozen_assignments=_frozen_assignments(
                frozen_assignments, normalized_pairs, pool_candidate_ids if pool_candidate_ids is not None else ids))
        with closing(self._connect(write=True)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT specification FROM review_collections WHERE id=?', (collection_id,)).fetchone()
            if row:
                if row['specification'] != _json(specification):
                    raise ReviewConflict('This collection already exists with different images or comparison plan')
            else:
                db.execute('INSERT INTO review_collections VALUES (?,?,?,?)',
                           (collection_id, _json(specification), secrets.token_hex(32), _now()))
        return self.collection(collection_id)

    def _row(self, collection_id):
        with closing(self._connect()) as db:
            row = db.execute('SELECT * FROM review_collections WHERE id=?', (collection_id,)).fetchone()
        if row is None:
            raise ReviewNotFound('Unknown review collection')
        return row

    def collection(self, collection_id):
        row = self._row(collection_id)
        return json.loads(row['specification']) | {'created_at': row['created_at']}

    def collections(self, reviewer=None):
        if not self.database.exists():
            return []
        with closing(self._connect()) as db:
            rows = db.execute('SELECT * FROM review_collections ORDER BY created_at DESC').fetchall()
            result = []
            for row in rows:
                spec = json.loads(row['specification'])
                votes = db.execute('SELECT count(*) FROM review_votes WHERE collection_id=?', (spec['id'],)).fetchone()[0]
                reviewed = [r[0] for r in db.execute('SELECT DISTINCT pair_id FROM review_votes WHERE collection_id=? AND reviewer=?',
                                                   (spec['id'], reviewer or ''))]
                result.append({key: spec[key] for key in ('id', 'title', 'brief')} |
                              {'candidate_count': len(spec['candidates']), 'comparison_count': len(spec['pairs']),
                               'vote_count': votes, 'reviewed_pair_ids': reviewed, 'created_at': row['created_at']})
            return result

    def candidate_path(self, collection_id, candidate_id):
        collection = self.collection(collection_id)
        record = next((c for c in collection['candidates'] if c['id'] == candidate_id), None)
        if record is None:
            raise ReviewNotFound('Unknown review candidate')
        try:
            path = self._image(record['path'])
        except (ValueError, OSError) as exc:
            raise ReviewConflict('Review image is missing or changed') from exc
        if sha256(path) != record['sha256']:
            raise ReviewConflict('Review image changed since registration')
        return path

    def comparison(self, collection_id, pair_id, reviewer):
        reviewer = _text(reviewer, 200)
        row = self._row(collection_id)
        spec = json.loads(row['specification'])
        pair = next((p for p in spec['pairs'] if p['id'] == pair_id), None)
        if pair is None:
            raise ReviewNotFound('Unknown comparison')
        digest = hashlib.sha256(_json([row['random_seed'], pair_id, reviewer]).encode()).hexdigest()
        ids = list(pair['candidate_ids'])
        if spec.get('schema_version', 1) >= 2:
            # Alternate sides in a reviewer-specific shuffled pair schedule.
            # Every complete even-sized queue has exactly balanced orientation.
            schedule = sorted(spec['pairs'], key=lambda p: hashlib.sha256(
                _json([row['random_seed'], reviewer, p['id']]).encode()).hexdigest())
            index = next(i for i, p in enumerate(schedule) if p['id'] == pair_id)
            reverse = (index + int(hashlib.sha256(_json([row['random_seed'], reviewer]).encode()).hexdigest(), 16)) % 2
        else:
            reverse = int(digest, 16) % 2
        if reverse:
            ids.reverse()
        frozen = None
        if 'frozen_assignments' in spec:
            frozen = next((item for item in spec['frozen_assignments'] if item['reviewer_slot'] == reviewer
                           and item['kind'] == 'parent_edit' and item['pair_id'] == pair_id), None)
            if frozen is None:
                raise ReviewNotFound('Reviewer slot is not assigned to this frozen comparison')
            ids = [frozen['candidate_order'][position] for position in ('A', 'B')]
        candidates = {c['id']: c for c in spec['candidates']}
        for candidate_id in ids:
            self.candidate_path(collection_id, candidate_id)
        assignment_id = hashlib.sha256(_json([collection_id, pair_id, reviewer, ids, digest]).encode()).hexdigest()
        if frozen is not None:
            assignment_id = frozen['assignment_id']
        with closing(self._connect()) as db:
            reviews = [json.loads(row[0]) for row in db.execute(
                'SELECT record FROM review_votes WHERE collection_id=? AND pair_id=? AND reviewer=? ORDER BY sequence',
                (collection_id, pair_id, reviewer))]
        return {'collection_id': collection_id, 'pair_id': pair_id, 'kind': pair['kind'],
                'assignment_id': assignment_id, 'reviewer': reviewer,
                'candidates': {position: candidates[candidate_id] for position, candidate_id in zip(('A', 'B'), ids)},
                'presentation': spec.get('presentation'),
                'reviews': reviews}

    def submit_vote(self, collection_id, pair_id, vote: Vote | dict):
        vote = Vote.model_validate(vote)
        # Preserve retry compatibility with older submissions lacking new fields.
        requested = vote.model_dump()
        for field in ('open_reading', 'felt_response', 'readiness', 'development_willingness', 'exposure'):
            if requested[field] is None:
                requested.pop(field)
        comparison = self.comparison(collection_id, pair_id, vote.reviewer)
        if comparison['assignment_id'] != vote.assignment_id:
            raise ReviewConflict('Comparison assignment changed; reload the images before voting')
        answers = {}
        for dimension in ('finished_quality', 'development_potential'):
            answer = requested[dimension]
            choice = answer.get('choice')
            answer = {'choice': choice, 'confidence': answer.get('confidence')}
            answers[dimension] = answer | {'preferred_candidate_id':
                comparison['candidates'][choice]['id'] if choice in ('A', 'B') else None}
        record = {'schema_version': 1, 'type': 'human_pairwise_review', 'created_at': _now(),
                  'collection_id': collection_id, 'pair_id': pair_id, **requested, **answers,
                  'candidate_order': {position: {'id': c['id'], 'sha256': c['sha256']}
                                      for position, c in comparison['candidates'].items()}}
        if comparison.get('presentation') is not None:
            record['presentation'] = comparison['presentation']
        for dimension in ('readiness', 'development_willingness'):
            if requested.get(dimension) is not None:
                record[dimension] = {position: {'response': requested[dimension].get(position),
                    'candidate_id': c['id'], 'image_sha256': c['sha256']}
                    for position, c in comparison['candidates'].items()}
        with closing(self._connect(write=True)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM review_votes WHERE submission_id=?', (vote.submission_id,)).fetchone()
            if previous:
                if (previous['collection_id'], previous['pair_id'], previous['request']) != (collection_id, pair_id, _json(requested)):
                    raise ReviewConflict('Submission ID already used for a different review')
                return json.loads(previous['record'])
            db.execute('INSERT INTO review_votes(submission_id,collection_id,pair_id,reviewer,request,record) VALUES(?,?,?,?,?,?)',
                       (vote.submission_id, collection_id, pair_id, vote.reviewer, _json(requested), _json(record)))
        return record

    def advance(self, collection_id, advancement: Advancement | dict):
        """Record an explicit human choice; never triggers generation or implies a vote."""
        advancement = Advancement.model_validate(advancement)
        requested = advancement.model_dump()
        collection = self.collection(collection_id)
        candidate = next((c for c in collection['candidates'] if c['id'] == advancement.candidate_id), None)
        if candidate is None:
            raise ReviewNotFound('Unknown review candidate')
        self.candidate_path(collection_id, candidate['id'])
        record = {'schema_version': 1, 'type': 'human_advancement', 'created_at': _now(),
                  'collection_id': collection_id, **requested, 'image_sha256': candidate['sha256']}
        with closing(self._connect(write=True)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM review_advancements WHERE submission_id=?', (advancement.submission_id,)).fetchone()
            if previous:
                if (previous['collection_id'], previous['request']) != (collection_id, _json(requested)):
                    raise ReviewConflict('Submission ID already used for another advancement')
                return json.loads(previous['record'])
            db.execute('INSERT INTO review_advancements(submission_id,collection_id,reviewer,request,record) VALUES(?,?,?,?,?)',
                       (advancement.submission_id, collection_id, advancement.reviewer, _json(requested), _json(record)))
        return record

    def history(self, collection_id):
        self.collection(collection_id)
        with closing(self._connect()) as db:
            return {name: [json.loads(r[0]) for r in db.execute(
                f'SELECT record FROM {table} WHERE collection_id=? ORDER BY sequence', (collection_id,))]
                    for name, table in (('votes', 'review_votes'), ('advancements', 'review_advancements'))}

    def pool_assignment(self, collection_id, reviewer):
        reviewer = _text(reviewer, 200)
        row = self._row(collection_id)
        spec = json.loads(row['specification'])
        eligible_ids = set(spec.get('pool_candidate_ids', [c['id'] for c in spec['candidates']]))
        candidates = sorted([c for c in spec['candidates'] if c['id'] in eligible_ids], key=lambda c: hashlib.sha256(
            _json([row['random_seed'], reviewer, c['id'], 'pool']).encode()).hexdigest())
        frozen = None
        if 'frozen_assignments' in spec:
            frozen = next((item for item in spec['frozen_assignments'] if item['reviewer_slot'] == reviewer
                           and item['kind'] == 'six_candidate_or_none'), None)
            if frozen is None:
                raise ReviewNotFound('Reviewer slot is not assigned to this frozen pool')
            by_id = {candidate['id']: candidate for candidate in candidates}
            candidates = [by_id[identifier] for identifier in frozen['candidate_ids']]
        for candidate in candidates:
            self.candidate_path(collection_id, candidate['id'])
        ordered = [{key: c[key] for key in ('id', 'sha256')} for c in candidates]
        expected = spec.get('pool_expected_count', len(candidates))
        complete = len(candidates) == expected
        return {'schema_version': 1, 'collection_id': collection_id, 'reviewer': reviewer,
            'assignment_id': frozen['assignment_id'] if frozen is not None else hashlib.sha256(_json([collection_id, reviewer, ordered, 'pool']).encode()).hexdigest(),
            'candidates': {str(i + 1): c for i, c in enumerate(candidates)},
            'presentation': spec.get('presentation'), 'expected_count': expected,
            'complete': complete, 'eligibility': 'eligible' if complete else 'incomplete_pool_excluded',
            'human_label': None}

    def submit_pool_choice(self, collection_id, choice: PoolChoice | dict):
        choice = PoolChoice.model_validate(choice)
        assignment = self.pool_assignment(collection_id, choice.reviewer)
        if not assignment['complete']:
            raise ReviewConflict('Incomplete pools cannot receive standard pool selections')
        if assignment['assignment_id'] != choice.assignment_id:
            raise ReviewConflict('Pool assignment changed; reload before selecting')
        candidates = list(assignment['candidates'].values())
        chosen = next((c for c in candidates if c['id'] == choice.candidate_id), None)
        if choice.choice == 'candidate' and chosen is None:
            raise ReviewNotFound('Unknown candidate in this pool')
        requested = choice.model_dump()
        record = {'schema_version': 1, 'type': 'human_pool_selection', 'created_at': _now(),
            'collection_id': collection_id, **requested, 'candidate_order': {
                position: {'id': c['id'], 'sha256': c['sha256']} for position, c in assignment['candidates'].items()},
            'selected_image_sha256': chosen['sha256'] if chosen else None,
            'presentation': assignment['presentation'], 'complete_pool': True}
        with closing(self._connect(write=True)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            previous = db.execute('SELECT * FROM review_pool_votes WHERE submission_id=?', (choice.submission_id,)).fetchone()
            if previous:
                if (previous['collection_id'], previous['request']) != (collection_id, _json(requested)):
                    raise ReviewConflict('Submission ID already used for a different pool choice')
                return json.loads(previous['record'])
            db.execute('INSERT INTO review_pool_votes(submission_id,collection_id,reviewer,request,record) VALUES(?,?,?,?,?)',
                       (choice.submission_id, collection_id, choice.reviewer, _json(requested), _json(record)))
        return record

    def pool_history(self, collection_id):
        self.collection(collection_id)
        with closing(self._connect()) as db:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='review_pool_votes'").fetchone():
                return []
            return [json.loads(row[0]) for row in db.execute(
                'SELECT record FROM review_pool_votes WHERE collection_id=? ORDER BY sequence', (collection_id,))]
