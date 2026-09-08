"""Project memory and a bounded, operator-directed studio loop.

The studio records artistic intentions and decisions separately from execution
checks. Rendering and crash recovery remain the controlled executor's job.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import sha256, write_json
from .backend import FluxBackend
from .controlled import run as run_edit
from .execution import _copy_snapshot_file, _read_source, _snapshot, canonical_config, restore_config
from .ledger import Ledger


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    schema_version: Literal[1] = 1


class Inquiry(Model):
    question: str = Field(min_length=1, max_length=4000)
    stakes: str = Field(min_length=1, max_length=4000)
    position: str = Field(min_length=1, max_length=4000)
    viewing_conditions: str = Field(min_length=1, max_length=2000)
    constraints: list[str] = Field(default_factory=list, max_length=32)
    unresolved: list[str] = Field(default_factory=list, max_length=32)


class Reference(Model):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    title: str = Field(min_length=1, max_length=1000)
    uri: str = Field(min_length=1, max_length=4000)
    provenance: str = Field(min_length=1, max_length=4000)
    permitted_uses: str = Field(min_length=1, max_length=2000)
    source_family: str = Field(min_length=1, max_length=1000)
    maker: str | None = None
    date: str | None = None
    medium: str | None = None
    context: str = ''


class Direction(Model):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    title: str = Field(min_length=1, max_length=1000)
    intended_change: str = Field(min_length=1, max_length=4000)
    preservation_constraints: list[str] = Field(min_length=1, max_length=32)
    source_ids: list[str] = Field(default_factory=list, max_length=32)
    relationship: str = ''
    relationship_kind: Literal['proposed_analogy', 'documented_relationship'] = 'proposed_analogy'
    plan: str = Field(min_length=1)

    @model_validator(mode='after')
    def check_sources(self):
        if self.source_ids and not self.relationship:
            raise ValueError('Referenced directions require a proposed relationship/transformation')
        if any(not value.strip() for value in self.preservation_constraints):
            raise ValueError('Preservation constraints must not be blank')
        return self


class ProposalBatch(Model):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    parent_artifact_id: str = Field(min_length=1)
    author: str = Field(min_length=1, max_length=200)
    origin: Literal['operator', 'model_proposal', 'development_example']
    directions: list[Direction] = Field(min_length=1, max_length=8)

    @model_validator(mode='after')
    def unique_directions(self):
        if len({a.id for a in self.directions}) != len(self.directions):
            raise ValueError('Direction IDs must be unique within a batch')
        if len({a.intended_change for a in self.directions}) != len(self.directions):
            raise ValueError('Directions must state distinct intended changes')
        return self


def now():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def identifier(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('IDs require 1–64 letters, digits, underscores or hyphens')
    return value


def required_text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('An author and a nonempty reason/observation are required')
    return value.strip()


def read_projects(root: Path, project_id=None):
    """Read-only dashboard view. Never creates a database or changes studio state."""
    database = Path(root).resolve() / 'jobs.sqlite3'
    if not database.exists():
        return []
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='studio_projects'").fetchone():
            return []
        query = 'SELECT data FROM studio_projects'
        params = ()
        if project_id is not None:
            query += ' WHERE id=?'
            params = (project_id,)
        projects = [json.loads(row[0]) for row in db.execute(query, params)]
        for project in projects:
            for episode in project['episodes'].values():
                episode['model_calls'] = db.execute('SELECT count(*) FROM reservation_charges WHERE job_id=?',
                                                   (episode['job_id'],)).fetchone()[0]
        return projects


class Studio:
    def __init__(self, root=Path('artifacts')):
        self.root = Path(root).resolve()
        self.ledger = Ledger(self.root / 'jobs.sqlite3')
        self.db = self.ledger.connection
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS studio_schema (version INTEGER NOT NULL);
            INSERT INTO studio_schema SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM studio_schema);
            CREATE TABLE IF NOT EXISTS studio_projects (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS studio_events (
                sequence INTEGER PRIMARY KEY, project_id TEXT NOT NULL REFERENCES studio_projects(id),
                type TEXT NOT NULL, details TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS studio_events_no_update BEFORE UPDATE ON studio_events
                BEGIN SELECT RAISE(ABORT, 'studio events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS studio_events_no_delete BEFORE DELETE ON studio_events
                BEGIN SELECT RAISE(ABORT, 'studio events are append-only'); END;
        ''')
        if self.db.execute('SELECT version FROM studio_schema').fetchone()[0] != 1:
            self.close()
            raise ValueError('Unsupported studio schema version')

    def __enter__(self):
        return self

    def close(self):
        self.ledger.close()

    def __exit__(self, *args):
        self.close()

    def get(self, project_id):
        row = self.db.execute('SELECT data FROM studio_projects WHERE id=?', (project_id,)).fetchone()
        if not row:
            raise ValueError(f'Unknown project: {project_id}')
        return json.loads(row[0])

    @contextmanager
    def locked(self, project_id):
        with self.ledger.lock('studio_project_' + identifier(project_id)):
            yield self.get(project_id)

    def _save(self, project, event, details):
        project['updated_at'] = now()
        with self.ledger._transaction():
            self.db.execute('INSERT INTO studio_projects VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data',
                            (project['id'], encoded(project)))
            self.db.execute('INSERT INTO studio_events(project_id,type,details,created_at) VALUES(?,?,?,?)',
                            (project['id'], event, encoded(details), now()))
        # These files are inspectable projections. SQLite above remains authoritative.
        directory = self.root / 'studio_projects' / project['id']
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'project.json', project)

    def create(self, project_id, name, inquiry, parent, author):
        project_id, author = identifier(project_id), required_text(author)
        inquiry = Inquiry.model_validate(inquiry).model_dump()
        parent = Path(parent).resolve()
        parent_hash = sha256(parent)
        with Image.open(parent) as image:
            image.load()
            if image.format != 'PNG' or image.mode != 'RGB':
                raise ValueError('Import an RGB PNG parent; normalization must happen before project creation')
            size = list(image.size)
        spec = dict(name=required_text(name), inquiry=inquiry, parent_sha256=parent_hash, author=author)
        with self.ledger.lock('studio_project_' + project_id):
            if self.db.execute('SELECT 1 FROM studio_projects WHERE id=?', (project_id,)).fetchone():
                project = self.get(project_id)
                if project['creation_spec'] != spec:
                    raise ValueError('Existing project ID has a different creation specification')
                self._artifact(project, 'root')
                return project
            directory = self.root / 'studio_projects' / project_id
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / 'parent.png'
            _copy_snapshot_file(parent, target, parent_hash)
            artifact = {'schema_version': 1, 'id': 'root', 'path': str(target), 'sha256': parent_hash,
                        'size': size, 'parent_artifact_id': None, 'action_id': None, 'origin': 'imported',
                        'source_path': str(parent), 'author': author}
            project = {'schema_version': 1, 'id': project_id, 'name': spec['name'], 'creation_spec': spec,
                'created_at': now(), 'current_artifact_id': 'root', 'inquiry_versions': [
                    {'version': 1, 'inquiry': inquiry, 'author': author, 'reason': 'Initial inquiry', 'created_at': now()}],
                'references': {}, 'artifacts': {'root': artifact}, 'episodes': {}, 'actions': {},
                'decisions': [], 'observations': []}
            self._save(project, 'project_created', {'parent_sha256': parent_hash, 'inquiry_version': 1, 'author': author})
            return project

    def revise(self, project_id, inquiry, reason, author):
        inquiry = Inquiry.model_validate(inquiry).model_dump()
        with self.locked(project_id) as project:
            version = {'version': len(project['inquiry_versions']) + 1, 'inquiry': inquiry,
                       'reason': required_text(reason), 'author': required_text(author), 'created_at': now()}
            project['inquiry_versions'].append(version)
            self._save(project, 'inquiry_revised', version)
            return version

    def add_reference(self, project_id, reference):
        reference = Reference.model_validate(reference).model_dump()
        with self.locked(project_id) as project:
            previous = project['references'].get(reference['id'])
            if previous:
                if previous != reference:
                    raise ValueError('Reference IDs are immutable; add a new ID for revised source information')
                return previous
            project['references'][reference['id']] = reference
            self._save(project, 'reference_added', reference)
            return reference

    def _artifact(self, project, artifact_id):
        artifact = project['artifacts'].get(artifact_id)
        if not artifact:
            raise ValueError(f'Unknown artifact in this project: {artifact_id}')
        path = Path(artifact['path'])
        if not path.is_file() or sha256(path) != artifact['sha256']:
            raise ValueError('Parent/output artifact is missing or changed; restore the recorded bytes')
        return artifact

    def start(self, project_id, episode_id, config, max_calls, max_actions=12, max_rounds=3):
        identifier(episode_id)
        for value, maximum in ((max_calls, 128), (max_actions, 64), (max_rounds, 16)):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError('Budgets require 1–128 calls, 1–64 actions, and 1–16 rounds')
        config = canonical_config(config)
        if config.output_dir != self.root:
            raise ValueError('Episode output directory must match the studio artifacts directory')
        with self.locked(project_id) as project:
            job_id = 'studio_' + fingerprint([project_id, episode_id])[:24]
            spec = {'schema_version': 1, 'kind': 'studio_episode', 'project_id': project_id,
                    'episode_id': episode_id, 'max_calls': max_calls, 'max_actions': max_actions,
                    'max_rounds': max_rounds, 'config': config.to_dict()}
            existing = project['episodes'].get(episode_id)
            if existing:
                if existing['spec'] != json.loads(encoded(spec)):
                    raise ValueError('Episode ID already has different immutable settings')
                return existing
            if any(e['status'] not in ('stopped', 'completed') for e in project['episodes'].values()):
                raise ValueError('Finish or stop the current episode before starting another')
            self._artifact(project, project['current_artifact_id'])
            directory = self.root / job_id
            self.ledger.create_job(job_id, spec, max_calls, directory)
            episode = {'schema_version': 1, 'id': episode_id, 'job_id': job_id, 'spec': spec,
                'status': 'awaiting_proposals', 'current_artifact_id': project['current_artifact_id'],
                'round': 0, 'batches': {}, 'action_ids': [], 'selected_round': 0, 'created_at': now()}
            project['episodes'][episode_id] = episode
            self._save(project, 'episode_started', {'episode_id': episode_id, 'spec': spec})
            return episode

    @staticmethod
    def _episode(project, episode_id):
        episode = project['episodes'].get(episode_id)
        if not episode:
            raise ValueError(f'Unknown episode in this project: {episode_id}')
        return episode

    def propose(self, project_id, episode_id, batch_path):
        batch_path = Path(batch_path).resolve()
        batch = ProposalBatch.model_validate_json(batch_path.read_text()).model_dump()
        with self.locked(project_id) as project:
            episode = self._episode(project, episode_id)
            existing = episode['batches'].get(batch['id'])
            if existing:
                if existing['proposal'] != batch:
                    raise ValueError('Batch ID already has a different proposal')
                return existing
            if episode['status'] != 'awaiting_proposals':
                raise ValueError('Execute the current batch and select a parent, or stop the episode, before proposing again')
            if batch['parent_artifact_id'] != episode['current_artifact_id']:
                raise ValueError('Proposal parent must be the selected current artifact; explicitly select alternatives first')
            parent = self._artifact(project, batch['parent_artifact_id'])
            spec = episode['spec']
            if episode['round'] >= spec['max_rounds'] or len(episode['action_ids']) + len(batch['directions']) > spec['max_actions']:
                raise ValueError('Episode round/action budget would be exceeded')
            prepared = []
            for direction in batch['directions']:
                if direction['id'] in project['actions']:
                    raise ValueError('Action IDs must be unique within the project')
                if set(direction['source_ids']) - project['references'].keys():
                    raise ValueError('Action references unknown source IDs')
                source = batch_path.parent / direction['plan']
                loaded, hashes = _read_source(source)
                if loaded[0]['parent_sha256'] != parent['sha256']:
                    raise ValueError('Executable plan parent does not match the selected artifact')
                prepared.append((direction, source, loaded, hashes))
            if sum(item[2][0]['max_calls'] for item in prepared) > spec['max_calls'] - self.ledger.call_count(episode['job_id']):
                raise ValueError('Declared plans, including fallbacks, exceed remaining episode call budget')
            episode['round'] += 1
            for direction, source, loaded, hashes in prepared:
                directory = self.root / 'studio_projects' / project_id / 'actions' / direction['id']
                directory.mkdir(parents=True, exist_ok=True)
                _snapshot(directory, source, loaded)
                frozen = directory / 'execution_plan.json'
                frozen_loaded, frozen_hashes = _read_source(frozen)
                child_id = 'studio_action_' + fingerprint([project_id, direction['id']])[:24]
                child_spec = {'schema_version': 1, 'kind': 'controlled_edit', 'plan': frozen_loaded[0],
                    'config': spec['config'], 'source_plan': str(frozen), 'source_hashes': frozen_hashes}
                self.ledger.create_job(child_id, child_spec, loaded[0]['max_calls'], self.root / child_id, parent_id=episode['job_id'])
                action = {'schema_version': 1, 'id': direction['id'], 'episode_id': episode_id,
                    'batch_id': batch['id'], 'round': episode['round'], 'direction': direction,
                    'parent_artifact_id': parent['id'], 'parent_sha256': parent['sha256'],
                    'inquiry_version': len(project['inquiry_versions']),
                    'inquiry': project['inquiry_versions'][-1]['inquiry'],
                    'references': [project['references'][ref] for ref in direction['source_ids']],
                    'author': batch['author'], 'origin': batch['origin'], 'status': 'proposed',
                    'plan_path': str(frozen), 'plan_sha256': sha256(frozen), 'source_hashes': hashes,
                    'job_id': child_id, 'run_dir': str(self.root / child_id), 'output_artifact_ids': [], 'created_at': now()}
                project['actions'][action['id']] = action
                episode['action_ids'].append(action['id'])
            record = {'proposal': batch, 'round': episode['round'], 'created_at': now()}
            episode['batches'][batch['id']] = record
            episode['status'] = 'queued'
            self._save(project, 'batch_proposed', {'episode_id': episode_id, 'batch': record})
            return record

    @staticmethod
    def _pending(project, episode):
        return [project['actions'][key] for key in episode['action_ids'] if project['actions'][key]['status']
                in ('proposed', 'running', 'interrupted', 'postprocessing_failed')]

    def _settle(self, project, episode):
        if self._pending(project, episode):
            episode['status'] = 'queued'
        elif episode['selected_round'] < episode['round'] and any(
                project['actions'][key]['output_artifact_ids'] for key in episode['action_ids']
                if project['actions'][key]['round'] == episode['round']):
            episode['status'] = 'awaiting_selection'
        elif (self.ledger.call_count(episode['job_id']) >= episode['spec']['max_calls']
              or episode['round'] >= episode['spec']['max_rounds']
              or len(episode['action_ids']) >= episode['spec']['max_actions']):
            episode['status'] = 'exhausted'
        else:
            episode['status'] = 'awaiting_proposals'

    def step(self, project_id, episode_id, backend_factory=FluxBackend):
        with self.locked(project_id) as project:
            episode = self._episode(project, episode_id)
            if episode['status'] in ('stopped', 'completed'):
                raise ValueError('Episode is closed')
            pending = self._pending(project, episode)
            if not pending:
                return episode
            action = pending[0]
            self._artifact(project, action['parent_artifact_id'])
            if sha256(Path(action['plan_path'])) != action['plan_sha256']:
                raise ValueError('Frozen action plan changed')
            action['status'], episode['status'] = 'running', 'running'
            self._save(project, 'action_started', {'action_id': action['id'], 'job_id': action['job_id']})
            try:
                directory = Path(action['run_dir'])
                kwargs = dict(ledger=self.ledger, parent_job_id=episode['job_id'], job_id=action['job_id'], backend_factory=backend_factory)
                if (directory / 'job.json').exists():
                    run_edit(resume_dir=directory, **kwargs)
                else:
                    run_edit(Path(action['plan_path']), restore_config(episode['spec']['config']), **kwargs)
                job = self.ledger.get_job(action['job_id'])
                manifest = job['payload']['manifest']
                self.ledger.verify_artifacts(action['job_id'])
                outputs = []
                for index, attempt in enumerate(manifest['attempts']):
                    name = f'attempt_{index}_result.png'
                    if name not in manifest['artifacts']:
                        continue
                    path = directory / name
                    artifact_id = action['id'] + ':' + str(index)
                    with Image.open(path) as image:
                        size = list(image.size)
                    record = {'schema_version': 1, 'id': artifact_id, 'path': str(path), 'sha256': manifest['artifacts'][name],
                        'size': size, 'parent_artifact_id': action['parent_artifact_id'], 'action_id': action['id'],
                        'origin': 'model_edit', 'execution_status': attempt['status'], 'verification': attempt.get('verification'),
                        'preservation': attempt.get('preservation'), 'effective_preservation': attempt.get('effective_preservation')}
                    previous = project['artifacts'].get(artifact_id)
                    if previous is not None and previous != record:
                        raise ValueError('Existing output artifact record changed')
                    project['artifacts'][artifact_id] = record
                    outputs.append(artifact_id)
                action.update(status=manifest['status'], output_artifact_ids=outputs,
                              model_calls=job['model_calls'], attempts=manifest['attempts'])
                self._settle(project, episode)
                self._save(project, 'action_finished', {'action_id': action['id'], 'status': action['status'], 'outputs': outputs})
            except BaseException as exc:
                action.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
                episode['status'] = 'interrupted'
                self._save(project, 'action_interrupted', {'action_id': action['id'], 'error': action['error']})
                raise
            return episode

    def run(self, project_id, episode_id, backend_factory=FluxBackend):
        # Each call drains only the already-declared batch. No model-driven taste selection.
        while True:
            project = self.get(project_id)
            episode = self._episode(project, episode_id)
            if episode['status'] in ('stopped', 'completed') or not self._pending(project, episode):
                return episode
            episode = self.step(project_id, episode_id, backend_factory)
            project = self.get(project_id)
            if any(a['status'] == 'postprocessing_failed' for a in self._pending(project, episode)):
                return episode

    def select(self, project_id, episode_id, artifact_id, reason, author):
        with self.locked(project_id) as project:
            episode = self._episode(project, episode_id)
            if episode['status'] in ('stopped', 'completed') or self._pending(project, episode):
                raise ValueError('Finish executing the batch before selection; closed episodes cannot change')
            artifact = self._artifact(project, artifact_id)
            decision = {'kind': 'select', 'episode_id': episode_id, 'artifact_id': artifact_id,
                        'reason': required_text(reason), 'author': required_text(author), 'created_at': now()}
            project['decisions'].append(decision)
            project['current_artifact_id'] = episode['current_artifact_id'] = artifact['id']
            episode['selected_round'] = episode['round']
            self._settle(project, episode)
            self._save(project, 'artifact_selected', decision)
            return episode

    def decide(self, project_id, artifact_id, decision, reason, author):
        if decision not in ('retain', 'reject'):
            raise ValueError('Decision must be retain or reject')
        with self.locked(project_id) as project:
            self._artifact(project, artifact_id)
            record = {'kind': decision, 'artifact_id': artifact_id, 'reason': required_text(reason),
                      'author': required_text(author), 'created_at': now()}
            project['decisions'].append(record)
            self._save(project, 'artifact_decision', record)
            return record

    def observe(self, project_id, artifact_id, note, author, origin):
        if origin not in ('operator', 'model_observation', 'development_example'):
            raise ValueError('Observation origin must be explicit')
        with self.locked(project_id) as project:
            artifact = self._artifact(project, artifact_id)
            record = {'artifact_id': artifact_id, 'image_sha256': artifact['sha256'], 'note': required_text(note),
                      'author': required_text(author), 'origin': origin, 'created_at': now()}
            project['observations'].append(record)
            self._save(project, 'artifact_observed', record)
            return record

    def close_episode(self, project_id, episode_id, reason, author, finish=False):
        with self.locked(project_id) as project:
            episode = self._episode(project, episode_id)
            if episode['status'] in ('stopped', 'completed'):
                return episode
            for action in self._pending(project, episode):
                action['status'] = 'cancelled'
            episode['status'] = 'completed' if finish else 'stopped'
            record = {'kind': 'finish' if finish else 'stop', 'episode_id': episode_id,
                      'artifact_id': episode['current_artifact_id'], 'reason': required_text(reason),
                      'author': required_text(author), 'created_at': now()}
            project['decisions'].append(record)
            self._save(project, 'episode_closed', record)
            return episode
