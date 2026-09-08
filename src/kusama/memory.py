"""Decision-time experiment memory derived from read-only execution/review ledgers.

The memory is an append-only projection, never execution authority. Events keep
their actual recorded availability time. Current manifests are not historical
snapshots; they are deliberately not ingested. No model calls or learned ranks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MEMORY_VERSION = 'event-cutoff-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def timestamp(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('A timezone-aware decision/event timestamp is required')
    return value.astimezone(timezone.utc)


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class EvidenceFile(Contract):
    path: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    role: str


class MemoryEvent(Contract):
    schema_version: int = 1
    id: str
    episode_id: str
    partition: Literal['development', 'sealed_test', 'unclassified']
    kind: Literal['context', 'action', 'artifact', 'observation', 'human_review', 'failure', 'budget', 'execution']
    available_at: str
    source_database: str
    source_table: str
    source_event_id: str
    source_sequence: int | None = None
    source_row_sha256: str
    job_id: str | None = None
    origin: str
    source_ids: list[str] = Field(default_factory=list)
    source_families: list[str] = Field(default_factory=list)
    context: str = ''
    payload: dict = Field(default_factory=dict)
    evidence: list[EvidenceFile] = Field(default_factory=list)

    @field_validator('available_at')
    @classmethod
    def aware_time(cls, value):
        return timestamp(value).isoformat()


class MemoryQuery(Contract):
    brief: str = Field(min_length=1, max_length=16000)
    decision_at: str
    episode_id: str = Field(min_length=1)
    purpose: Literal['development', 'heldout'] = 'development'
    exclude_source_families: tuple[str, ...] = ()
    limit: int = Field(default=8, ge=1, le=32)

    @field_validator('decision_at')
    @classmethod
    def valid_cutoff(cls, value):
        moment = timestamp(value)
        if moment > datetime.now(timezone.utc):
            raise ValueError('Decision cutoff cannot be in the future')
        return moment.isoformat()


@contextmanager
def read_database(path):
    """A coherent SQLite snapshot; never instantiate a writer to inspect a run."""
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute('BEGIN')
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


def rows(connection, table):
    # Table names are fixed internal identifiers, not user/source text.
    exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return [dict(row) for row in connection.execute(f'SELECT * FROM {table}')] if exists else []


def _sources(spec):
    documents = spec.get('source_documents', []) + spec.get('references', [])
    documents += spec.get('request', {}).get('references', [])
    ids, families = set(), set()
    for item in documents:
        if isinstance(item, dict):
            if item.get('reference_id') or item.get('id'):
                ids.add(item.get('reference_id') or item['id'])
            if item.get('source_family'):
                families.add(item['source_family'])
    for action in spec.get('actions', []):
        ids.update(action.get('reference_ids', []))
    return sorted(ids), sorted(families)


def _partition(episode, spec):
    partition = spec.get('partition')
    if partition in ('sealed_test', 'heldout', 'sealed') or spec.get('sealed') is True or re.search(r'(?:^|[/_])e0-h\d+', episode):
        return 'sealed_test'
    if partition == 'development' or spec.get('kind') in {
        'connection_inspiration', 'connection_generation', 'connection_local_proposals',
        'selected_parent_continuation', 'visible_transition_observations'}:
        return 'development'
    if re.search(r'(?:^|[/_])e0-d\d+', episode):
        return 'development'
    return 'unclassified'


class ExperimentMemory:
    def __init__(self, root=Path('artifacts/overnight')):
        self.root = Path(root).resolve()
        self.directory = self.root / 'memory'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'memory.sqlite3'
        self.db = sqlite3.connect(self.path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS memory_events (
                id TEXT PRIMARY KEY, origin_key TEXT UNIQUE NOT NULL,
                row_sha256 TEXT NOT NULL, event_json TEXT NOT NULL, raw_row_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memory_builds (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, report_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS memory_queries (
                id TEXT PRIMARY KEY, created_at TEXT NOT NULL, snapshot_json TEXT NOT NULL);
        ''')
        for table in ('memory_events', 'memory_builds', 'memory_queries'):
            for operation in ('UPDATE', 'DELETE'):
                self.db.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                    BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'experiment memory is append-only'); END''')
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _append(self, event, raw):
        event = MemoryEvent.model_validate(event)
        key = canonical([event.source_database, event.source_table, event.source_event_id])
        encoded = canonical(event.model_dump())
        prior = self.db.execute('SELECT * FROM memory_events WHERE origin_key=?', (key,)).fetchone()
        if prior:
            if prior['row_sha256'] != event.source_row_sha256 or prior['event_json'] != encoded:
                raise ValueError('Previously ingested authority or projection changed: ' + key)
            return False
        self.db.execute('INSERT INTO memory_events VALUES(?,?,?,?,?)',
                        (event.id, key, event.source_row_sha256, encoded, canonical(raw)))
        return True

    def events(self):
        return [MemoryEvent.model_validate_json(row[0]) for row in self.db.execute('SELECT event_json FROM memory_events ORDER BY id')]

    def query(self, brief, decision_at, episode_id, purpose='development', exclude_source_families=(), limit=8):
        """Freeze the exact eligible set, rank and selected inputs before a decision.

        Heldout use is deliberately restricted to this episode's own preceding
        non-review state. Development use excludes sealed and unclassified runs.
        Genuine review outcomes are separate and never returned to heldout calls.
        """
        request = MemoryQuery(brief=brief, decision_at=decision_at, episode_id=episode_id,
                              purpose=purpose, exclude_source_families=tuple(sorted(set(exclude_source_families))), limit=limit)
        cutoff = timestamp(request.decision_at)
        tokens = set(re.findall(r'[a-z0-9]+', brief.lower())) - {'the', 'a', 'an', 'and', 'or', 'to', 'of', 'in', 'with', 'is'}
        eligible, rejected = [], Counter()
        wants_budget = bool(tokens & {'budget', 'cost', 'costs', 'runtime', 'tokens', 'charged', 'reservation'})
        for event in self.events():
            if timestamp(event.available_at) >= cutoff:
                rejected['at_or_after_cutoff'] += 1
                continue
            if purpose == 'heldout':
                if event.episode_id != episode_id or event.kind == 'human_review' or event.partition == 'unclassified':
                    rejected['heldout_boundary'] += 1
                    continue
            elif event.partition != 'development':
                rejected['sealed_or_unclassified'] += 1
                continue
            if set(event.source_families) & set(request.exclude_source_families):
                rejected['source_family'] += 1
                continue
            meaningful_artifact = event.kind == 'artifact' and (
                event.payload.get('kind') in ('image', 'image_record') or
                any(key in event.payload for key in ('children', 'parent')))
            if (event.kind in ('context', 'execution') or
                    (event.kind == 'budget' and not wants_budget) or
                    (event.kind == 'artifact' and not meaningful_artifact)):
                rejected['non_experience_event'] += 1
                continue
            # Inputs are inert JSON. No commands, expressions, or template execution.
            words = set(re.findall(r'[a-z0-9]+', canonical(event.payload).lower()))
            context_words = set(re.findall(r'[a-z0-9]+', event.context.lower()))
            overlap = sorted(tokens & (words | context_words))
            score = (2 * len(tokens & words) + .25 * len(tokens & context_words)) / max(1, len(tokens))
            if overlap:
                eligible.append((score, event, overlap))
            else:
                rejected['no_query_overlap'] += 1
        eligible.sort(key=lambda item: (-item[0], item[1].available_at, item[1].id))
        # Preserve realized evidence and failure cases when available; verbose
        # proposals alone should not consume the complete memory shortlist.
        required = []
        for predicate in (
            lambda event: event.kind == 'observation',
            lambda event: event.kind == 'artifact' and (event.payload.get('kind') == 'image' or 'children' in event.payload),
            lambda event: event.kind == 'failure',
        ):
            candidate = next((item for item in eligible if predicate(item[1])), None)
            if candidate is not None:
                required.append(candidate)
        required = required[:limit]
        selected = [item for item in eligible if item not in required][:limit - len(required)] + required
        authority_rows = {}
        for _, event, _ in selected:
            key = (event.source_database, event.source_table)
            if key not in authority_rows:
                if event.source_table not in {'events', 'studio_events', 'continuation_specs', 'review_votes', 'review_advancements', 'review_pool_votes', 'cost_events', 'resource_events'}:
                    raise ValueError('Unknown memory authority table')
                with read_database(event.source_database) as authority:
                    authority_rows[key] = rows(authority, event.source_table)
            matches = [row for row in authority_rows[key] if str(row.get('event_id', row.get('submission_id', row.get('sequence', row.get('id'))))) == event.source_event_id]
            if len(matches) != 1 or digest(matches[0]) != event.source_row_sha256:
                raise ValueError('Memory source authority changed or is missing')
        for _, event, _ in selected:
            for evidence in event.evidence:
                if not Path(evidence.path).is_file() or file_hash(evidence.path) != evidence.sha256:
                    raise ValueError('Memory evidence changed or is missing: ' + evidence.path)
        snapshot = {
            'schema_version': 1, 'memory_version': MEMORY_VERSION,
            'request': request.model_dump(mode='json'),
            'ranking_policy': 'payload token overlap weight 2, brief context weight .25; stable event-time/id ties; experience events; budget only on budget queries; reserve matched observation, actual image/transition and failure slots when available, in that priority order',
            'eligible_candidates': [{'id': event.id, 'score': score, 'matched_tokens': overlap,
                                     'available_at': event.available_at, 'source_row_sha256': event.source_row_sha256}
                                    for score, event, overlap in eligible],
            'selected': [event.model_dump(mode='json') for _, event, _ in selected],
            'cutoff_rule': 'available_at < decision_at; no wall-clock or file-mtime backdating',
            'heldout_rule': 'only own preceding state; no human review labels',
        }
        # Rejected future content/counts deliberately do not enter decision inputs
        # or snapshot identity: future ingestion cannot change a historical query.
        snapshot_id = digest(snapshot)
        snapshot = {'snapshot_id': snapshot_id, **snapshot}
        self.db.execute('INSERT OR IGNORE INTO memory_queries VALUES(?,?,?)',
                        (snapshot_id, datetime.now(timezone.utc).isoformat(), canonical(snapshot)))
        self.db.commit()
        target = self.directory / 'queries' / (snapshot_id + '.json')
        target.parent.mkdir(exist_ok=True)
        if target.exists() and json.loads(target.read_text()) != snapshot:
            raise ValueError('Immutable memory query export changed')
        if not target.exists():
            target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n')
        return snapshot


class _Builder:
    def __init__(self, memory):
        self.memory, self.root = memory, memory.root
        self.added = 0
        self.issues = []
        self.jobs, self.info, self.projects, self.continuations = {}, {}, {}, {}
        self.artifact_episodes = {}
        self.source_boundaries = {}

    def remember_artifact(self, sha, info):
        prior = self.artifact_episodes.get(sha)
        if prior and prior['episode'] != info['episode']:
            # Identical bytes in distinct episodes cannot identify a review or
            # parent uniquely; never let insertion order declassify sealed work.
            info = dict(episode='shared:' + sha,
                partition='sealed_test' if 'sealed_test' in (prior['partition'], info['partition']) else 'unclassified',
                context='', sources=sorted(set(prior['sources'] + info['sources'])),
                families=sorted(set(prior['families'] + info['families'])))
        self.artifact_episodes[sha] = info

    def evidence(self, path, expected, role):
        path = Path(path).resolve()
        if not expected or not path.is_file() or file_hash(path) != expected:
            raise ValueError('Missing or mismatched evidence: ' + str(path))
        return {'path': str(path), 'sha256': expected, 'role': role}

    def emit(self, database, table, raw, info, kind, payload, *, origin='execution_ledger', evidence=(), event_id=None, when=None, context=None):
        source_id = str(event_id if event_id is not None else raw.get('event_id', raw.get('sequence', raw.get('id'))))
        row_hash = digest(raw)
        value = dict(id=digest([str(database), table, source_id, row_hash, MEMORY_VERSION]),
            episode_id=info['episode'], partition=info['partition'], kind=kind,
            available_at=when or raw['created_at'], source_database=str(database), source_table=table,
            source_event_id=source_id, source_sequence=raw.get('sequence'), source_row_sha256=row_hash,
            job_id=raw.get('job_id'), origin=origin, source_ids=info['sources'], source_families=info['families'],
            context=info['context'] if context is None else context, payload=payload, evidence=list(evidence))
        if timestamp(value['available_at']) > datetime.now(timezone.utc):
            raise ValueError('Authority event has a future timestamp')
        self.added += self.memory._append(value, raw)

    def info_for(self, job_id, visiting=()):
        if job_id in self.info:
            return self.info[job_id]
        if job_id in visiting or job_id not in self.jobs:
            return dict(episode=str(job_id), partition='unclassified', context='', sources=[], families=[])
        row = self.jobs[job_id]
        spec = json.loads(row['spec_json'])
        directory = Path(row['directory'])
        # Directory membership and immutable spec lineage, not name similarity.
        run_root = self.root / 'runs'
        episode = None
        if directory.is_relative_to(run_root) and directory != run_root:
            episode = directory.relative_to(run_root).parts[0]
        elif spec.get('brief_id'):
            episode = spec['brief_id']
        elif 'episodes' in directory.parts:
            offset = directory.parts.index('episodes') + 1
            if offset < len(directory.parts):
                episode = directory.parts[offset]
        parent_id = spec.get('inspiration_job') or row.get('parent_id')
        inherited = self.info_for(parent_id, (*visiting, job_id)) if parent_id else None
        project_id = spec.get('project_id')
        if project_id in self.continuations:
            inherited = self.continuations[project_id]['info']
        if episode is None:
            episode = inherited['episode'] if inherited else job_id
        sources, families = _sources(spec)
        request = spec.get('request', {})
        context = spec.get('brief') or request.get('brief') or spec.get('whole_context') or ''
        partition = _partition(episode, spec)
        if inherited:
            sources = sorted(set(sources + inherited['sources']))
            families = sorted(set(families + inherited['families']))
            context = context or inherited['context']
            if partition == 'unclassified' or inherited['partition'] == 'sealed_test':
                partition = inherited['partition']
        info = dict(episode=episode, partition=partition, context=context, sources=sources, families=families)
        self.info[job_id] = info
        return info

    def jobs_database(self):
        database = self.root / 'jobs.sqlite3'
        if not database.is_file():
            return
        with read_database(database) as db:
            self.jobs = {row['job_id']: row for row in rows(db, 'jobs')}
            artifacts = rows(db, 'artifacts')
            events = sorted(rows(db, 'events'), key=lambda row: row['sequence'])
            self.projects = {row['id']: json.loads(row['data']) for row in rows(db, 'studio_projects')}
            continuation_rows = rows(db, 'continuation_specs')
            studio_events = sorted(rows(db, 'studio_events'), key=lambda row: row['sequence'])
        for job_id, row in self.jobs.items():
            if digest(json.loads(row['spec_json'])) != row['spec_fingerprint']:
                raise ValueError('Immutable job spec fingerprint mismatch: ' + job_id)
        for row in artifacts:
            if row.get('kind') == 'image':
                info = self.info_for(row['job_id'])
                if info['partition'] != 'unclassified':
                    self.remember_artifact(row['sha256'], info)
        for raw in continuation_rows:
            spec = json.loads(raw['specification'])
            selected = spec['selection']
            sources, families = _sources(spec)
            info = dict(self.artifact_episodes.get(selected['image_sha256']) or dict(
                episode='continuation:' + raw['id'], partition=_partition(raw['id'], spec),
                context=spec['request']['brief'], sources=sources, families=families))
            info['sources'] = sorted(set(info['sources'] + sources))
            info['families'] = sorted(set(info['families'] + families))
            self.continuations[spec['project_id']] = {'raw': raw, 'spec': spec, 'info': info}
        # Recompute descendants after authoritative continuation-parent mapping.
        self.info.clear()
        for job_id in self.jobs:
            self.info_for(job_id)
        for raw in events:
            try:
                self.job_event(database, raw)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.issues.append({'table': 'events', 'source_event_id': raw['event_id'], 'error': str(exc)})
        for value in self.continuations.values():
            raw, spec, info = value['raw'], value['spec'], value['info']
            try:
                parent = self.evidence(spec['selection']['path'], spec['selection']['image_sha256'], 'selected_parent')
                # A human advancement is a selection event, never a preference label.
                selection = {key: spec['selection'].get(key) for key in ('origin', 'author', 'reason', 'candidate_id', 'image_sha256', 'advancement_submission_id')}
                payload = {'continuation_id': spec['id'], 'parent': parent, 'selection': selection,
                           'action': spec['request']['revision'], 'max_calls': spec['max_calls']}
                self.emit(database, 'continuation_specs', raw, info, 'action', payload,
                          origin=spec['request']['revision']['origin'], evidence=[parent])
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.issues.append({'table': 'continuation_specs', 'source_event_id': raw['id'], 'error': str(exc)})
        for raw in studio_events:
            try:
                self.studio_event(database, raw)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.issues.append({'table': 'studio_events', 'source_event_id': raw['sequence'], 'error': str(exc)})
        self.source_boundaries[str(database)] = {'events_max_sequence': max([row['sequence'] for row in events], default=0),
            'studio_events_max_sequence': max([row['sequence'] for row in studio_events], default=0)}

    def job_event(self, database, raw):
        info = self.info_for(raw['job_id'])
        job = self.jobs[raw['job_id']]
        spec = json.loads(job['spec_json'])
        event_type, payload = raw['type'], json.loads(raw['payload_json'])
        kind, origin, evidence = 'execution', 'execution_ledger', []
        if event_type == 'job_created':
            kind = 'context'
            payload = {'job_kind': spec.get('kind'), 'max_calls': job['max_calls'],
                       'parent_job_id': job['parent_id'], 'spec_sha256': job['spec_fingerprint'],
                       'local_need': spec.get('local_need'), 'whole_context': spec.get('whole_context')}
            if spec.get('actions'):
                kind, origin = 'action', 'model_proposal' if spec.get('kind') == 'connection_generation' else 'recorded_input'
                payload['actions'] = spec['actions']
            if spec.get('plan'):
                kind, origin = 'action', 'recorded_input'
                payload['plan'] = spec['plan']
        elif event_type == 'artifact_registered':
            path = Path(job['directory']) / payload['path']
            evidence = [self.evidence(path, payload['sha256'], payload.get('kind') or 'artifact')]
            kind = 'artifact'
            payload = {**payload, 'path': str(path.resolve())}
            if payload.get('kind') == 'image':
                self.remember_artifact(payload['sha256'], info)
                index = payload.get('attempt_index')
                actions = spec.get('actions', [])
                if isinstance(index, int) and index < len(actions):
                    payload['action'] = actions[index]
                    payload['parent_artifact_id'] = actions[index].get('parent_artifact_id')
            elif path.suffix == '.json' and payload.get('kind') in ('response', 'model_response', 'image_record', 'snapshot'):
                data = json.loads(path.read_text())
                if data.get('parse_error') or data.get('inference_error'):
                    kind, origin = 'failure', 'model_response_validation'
                    payload['parse_error'] = data.get('parse_error')
                    payload['inference_error'] = data.get('inference_error')
                elif data.get('output'):
                    output = data['output']
                    if isinstance(output, dict) and 'visible_changes' in output:
                        kind, origin = 'observation', 'model_observation'
                    else:
                        kind, origin = 'action', 'model_proposal'
                    payload['output'] = output
                    if data.get('authorship'):
                        payload['authorship'] = data['authorship']
                elif payload.get('kind') == 'image_record':
                    payload['image'] = {key: data.get(key) for key in ('id', 'sha256', 'parent_image_id', 'parent_sha256', 'action', 'origin', 'connection_id')}
        elif event_type in ('call_reserved', 'child_call_reserved'):
            kind = 'budget'
            payload = {**payload, 'charge_state': 'reserved_even_if_execution_uncertain', 'max_job_calls': job['max_calls']}
        elif event_type in ('attempt_created', 'attempt_updated'):
            kind = 'failure' if payload.get('status') in ('failed', 'uncertain', 'interrupted', 'rejected') else 'execution'
            safe = payload.get('payload') or {}
            payload = {key: payload.get(key) for key in ('attempt_id', 'index', 'status')}
            payload.update({key: safe[key] for key in ('action', 'error', 'parse_error', 'inference_error', 'output') if key in safe})
        elif event_type == 'job_updated':
            # A mutable state projection may include sealed responses or reviews.
            # Only the actual event status is usable here; artifact events carry bytes.
            kind = 'failure' if payload.get('status') in ('failed', 'uncertain', 'interrupted', 'rejected') else 'execution'
            payload = {'status': payload.get('status')}
        else:
            payload = {'event_type': event_type}
        self.emit(database, 'events', raw, info, kind, payload, origin=origin, evidence=evidence)

    def studio_event(self, database, raw):
        details = json.loads(raw['details'])
        project = self.projects[raw['project_id']]
        continuation = self.continuations.get(raw['project_id'])
        info = continuation['info'] if continuation else dict(episode='studio:' + raw['project_id'],
            partition='unclassified', context='', sources=[], families=[])
        kind, origin, evidence = 'execution', 'execution_ledger', []
        if raw['type'] == 'artifact_observed':
            kind, origin = 'observation', details['origin']
            if origin not in ('operator', 'model_observation', 'development_example'):
                raise ValueError('Unknown observation origin')
            artifact = project['artifacts'][details['artifact_id']]
            evidence = [self.evidence(artifact['path'], details['image_sha256'], 'observed_image')]
        elif raw['type'] == 'action_finished':
            kind = 'artifact' if details['outputs'] else 'failure'
            action = project['actions'][details['action_id']]
            parent = project['artifacts'][action['parent_artifact_id']]
            evidence = [self.evidence(parent['path'], parent['sha256'], 'parent')]
            children = []
            for identifier in details['outputs']:
                child = project['artifacts'][identifier]
                image = self.evidence(child['path'], child['sha256'], 'actual_child')
                evidence.append(image)
                children.append({key: child.get(key) for key in ('id', 'path', 'sha256', 'parent_artifact_id', 'action_id', 'origin', 'execution_status', 'verification', 'preservation', 'effective_preservation')})
                self.remember_artifact(child['sha256'], info)
            details = {**details, 'parent': evidence[0], 'children': children}
        elif raw['type'] == 'action_interrupted':
            kind = 'failure'
        elif raw['type'] == 'batch_proposed':
            kind, origin = 'action', 'recorded_proposal'
        elif raw['type'] == 'artifact_decision':
            # Operator decision is not a genuine preference review record.
            origin = 'operator_decision'
        elif raw['type'] == 'project_created':
            kind = 'artifact'
            parent = project['artifacts']['root']
            evidence = [self.evidence(parent['path'], details['parent_sha256'], 'imported_parent')]
            details = {**details, 'parent': evidence[0]}
        elif raw['type'] not in ('inquiry_revised', 'reference_added', 'episode_started', 'action_started'):
            details = {'event_type': raw['type']}
        self.emit(database, 'studio_events', raw, info, kind, details, origin=origin, evidence=evidence)

    def reviews_database(self):
        database = self.root / 'reviews.sqlite3'
        if not database.is_file():
            return
        with read_database(database) as db:
            collections = {row['id']: json.loads(row['specification']) for row in rows(db, 'review_collections')}
            tables = {table: rows(db, table) for table in ('review_votes', 'review_advancements', 'review_pool_votes')}
        expected_types = {'review_votes': 'human_pairwise_review', 'review_advancements': 'human_advancement', 'review_pool_votes': 'human_pool_selection'}
        for table, records in tables.items():
            for raw in records:
                try:
                    record = json.loads(raw['record'])
                    if record.get('type') != expected_types[table] or record.get('submission_id') != raw['submission_id']:
                        raise ValueError('Review record type/submission provenance mismatch')
                    collection = collections[raw['collection_id']]
                    candidates = collection['candidates']
                    if isinstance(candidates, dict):
                        candidates = list(candidates.values())
                    used = set(item['sha256'] for item in record.get('candidate_order', {}).values())
                    if record.get('image_sha256'):
                        used.add(record['image_sha256'])
                    infos = [self.artifact_episodes[sha] for sha in used if sha in self.artifact_episodes]
                    if not used or len(infos) != len(used) or len({item['episode'] for item in infos}) != 1:
                        info = dict(episode='review:' + raw['collection_id'], partition='unclassified', context=collection.get('brief', ''), sources=[], families=[])
                    else:
                        info = dict(infos[0])
                        info['sources'] = sorted(set().union(*(set(item['sources']) for item in infos)))
                        info['families'] = sorted(set().union(*(set(item['families']) for item in infos)))
                        if any(item['partition'] == 'sealed_test' for item in infos):
                            info['partition'] = 'sealed_test'
                    evidence = [self.evidence(self.root / item['path'], item['sha256'], 'reviewed_candidate')
                                for item in candidates if item['sha256'] in used]
                    if len(evidence) != len(used):
                        raise ValueError('Review candidate hash lacks collection evidence')
                    self.emit(database, table, raw, info, 'human_review', record, origin='human_review_ledger',
                              evidence=evidence, event_id=raw['submission_id'], when=record['created_at'])
                except (ValueError, KeyError, OSError, TypeError) as exc:
                    self.issues.append({'table': table, 'source_event_id': raw['submission_id'], 'error': str(exc)})
        self.source_boundaries[str(database)] = {table + '_max_sequence': max([row['sequence'] for row in records], default=0) for table, records in tables.items()}

    def costs_database(self):
        database = self.root / 'costs.sqlite3'
        if not database.is_file():
            return
        with read_database(database) as db:
            records = rows(db, 'cost_events')
        for raw in records:
            try:
                payload = json.loads(raw['payload_json'])
                info = self.info_for(raw['run_id'])
                # Cost payloads contain measured runtime and explicit estimate/unknown
                # statuses, never turn unknown electricity/hardware into zero.
                self.emit(database, 'cost_events', raw, info, 'budget',
                          {'run_id': raw['run_id'], 'call_id': raw['call_id'], 'event_type': raw['event_type'], **payload},
                          origin='local_cost_journal')
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.issues.append({'table': 'cost_events', 'source_event_id': raw['sequence'], 'error': str(exc)})
        self.source_boundaries[str(database)] = {'max_sequence': max([row['sequence'] for row in records], default=0)}

    def resources_database(self):
        database = self.root / 'resources.sqlite3'
        if not database.is_file():
            return
        with read_database(database) as db:
            records = rows(db, 'resource_events')
        keys = {}
        for job_id, job in self.jobs.items():
            for action in json.loads(job['spec_json']).get('actions', []):
                if action.get('idempotency_key'):
                    keys[action['idempotency_key']] = self.info_for(job_id)
        for value in self.continuations.values():
            for key in value['spec'].get('invocation_keys', {}).values():
                keys[key] = value['info']
        for raw in records:
            try:
                payload = json.loads(raw['payload'])
                key = payload.get('key', '')
                info = keys.get(key) or self.info_for(key.rsplit(':reasoning:', 1)[0] if ':reasoning:' in key else None)
                self.emit(database, 'resource_events', raw, info, 'budget',
                          {'event_kind': raw['kind'], **payload}, origin='global_resource_ledger')
            except (ValueError, KeyError, OSError, TypeError) as exc:
                self.issues.append({'table': 'resource_events', 'source_event_id': raw['sequence'], 'error': str(exc)})
        self.source_boundaries[str(database)] = {'max_sequence': max([row['sequence'] for row in records], default=0)}


def build_memory(root=Path('artifacts/overnight')):
    with ExperimentMemory(root) as memory:
        builder = _Builder(memory)
        builder.jobs_database()
        builder.reviews_database()
        builder.costs_database()
        builder.resources_database()
        events = memory.events()
        report = {'memory_version': MEMORY_VERSION, 'root': str(memory.root), 'database': str(memory.path),
                  'events': len(events), 'new_events': builder.added,
                  'kinds': dict(Counter(event.kind for event in events)),
                  'partitions': dict(Counter(event.partition for event in events)),
                  'episodes': sorted(set(event.episode_id for event in events)),
                  'source_boundaries': builder.source_boundaries, 'issues': builder.issues,
                  'source_event_set_sha256': digest(sorted((event.id, event.source_row_sha256) for event in events))}
        report_id = digest(report)
        memory.db.execute('INSERT OR IGNORE INTO memory_builds VALUES(?,?,?)',
                          (report_id, datetime.now(timezone.utc).isoformat(), canonical(report)))
        memory.db.commit()
        directory = memory.directory / 'builds'
        directory.mkdir(exist_ok=True)
        (directory / (report_id + '.json')).write_text(json.dumps(report, indent=2) + '\n')
        return {'build_id': report_id, **report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--root', type=Path, default=Path('artifacts/overnight'))
    query = commands.add_parser('query')
    query.add_argument('--root', type=Path, default=Path('artifacts/overnight'))
    query.add_argument('--brief', required=True)
    query.add_argument('--decision-at', required=True)
    query.add_argument('--episode-id', required=True)
    query.add_argument('--purpose', choices=['development', 'heldout'], default='development')
    query.add_argument('--exclude-source-family', action='append', default=[])
    query.add_argument('--limit', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'build':
        result = build_memory(args.root)
    else:
        with ExperimentMemory(args.root) as memory:
            result = memory.query(args.brief, args.decision_at, args.episode_id, args.purpose, args.exclude_source_family, args.limit)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
