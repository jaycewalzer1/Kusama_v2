"""Question-led runs, explicit bounded authorization and durable local dispatch."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import sha256, write_json
from .evidence import digest, immutable_json
from .director import RESEARCH_TOTAL_TOKENS
from .ledger import BudgetExceeded, SpecMismatch
from .resources import OvernightConfig, SessionBudget, load_config
from .study_cycle import DevelopmentCycle, cycle_id

REPOSITORY = Path(__file__).resolve().parents[2]
ACTIVE = {'queued', 'researching', 'rendering', 'starting'}


class StartRun(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    submission_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    question: str = Field(min_length=1, max_length=8000)
    title: str = Field(default='', max_length=180)
    context: str = Field(default='', max_length=4000)
    seed_artists: list[str] = Field(default_factory=list, max_length=12)
    seed_works: list[str] = Field(default_factory=list, max_length=12)
    medium_constraint: str = Field(default='', max_length=300)
    render_profile: str = 'flux_base4b_study_512_v1'
    research_steps: int = Field(default=24, ge=4, le=48)
    total_token_limit: int = Field(default=RESEARCH_TOTAL_TOKENS, ge=8192, le=RESEARCH_TOTAL_TOKENS, strict=True)
    directions: int = Field(default=3, ge=1, le=6)
    images_per_direction: int = Field(default=1, ge=1, le=2)
    minutes: int = Field(default=120, ge=15, le=480)
    allow_network: Literal[True] = True
    study_development: bool = True
    development_steps: int = Field(default=3, ge=1, le=6)
    development_research_steps: int = Field(default=8, ge=4, le=24)
    random_exposure_count: int = Field(default=2, ge=0, le=2, strict=True)

    @model_validator(mode='after')
    def study_capacity(self):
        if self.study_development and self.directions < 3:
            raise ValueError('Artistic development requires capacity for at least three distinct studies; seed variants do not count')
        if self.study_development and self.directions * self.images_per_direction + self.development_steps > 12:
            raise ValueError('Initial images plus development actions must fit the 12-branch inspection capacity')
        return self


class StopRequested(BudgetExceeded):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def options():
    config = load_config(REPOSITORY / 'configs/overnight.yaml')
    return {'profiles': [{'id': key, **value} for key, value in config.render_profiles.items()],
            'device': config.device, 'default_profile': config.default_render_profile,
            'handoff_reserve_minutes': 5, 'total_token_limit': RESEARCH_TOTAL_TOKENS,
            'reserved_output_tokens': 1800}


def source_identity():
    return {str(path.relative_to(REPOSITORY)): sha256(path) for path in sorted((REPOSITORY / 'src/kusama').rglob('*.py'))} | {'uv.lock': sha256(REPOSITORY / 'uv.lock')}


class PanelRuns:
    def __init__(self, root, coordinator=None):
        self.root = Path(root).resolve()
        self.coordinator = Path(coordinator or root).resolve()
        self.database = self.root / 'panel_jobs.sqlite3'

    def directory(self, identifier):
        if not re.fullmatch(r'studio-[a-f0-9]{20}', identifier):
            raise ValueError('Unknown studio run')
        directory = self.root / 'runs' / identifier
        if not directory.resolve().is_relative_to(self.root) or any(p.is_symlink() for p in (directory, directory.parent)):
            raise ValueError('Run directory is outside its workspace')
        return directory

    def _connect(self):
        self.root.mkdir(parents=True, exist_ok=True)
        if self.database.is_symlink():
            raise ValueError('Panel ledger cannot be a symbolic link')
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        db.executescript('''CREATE TABLE IF NOT EXISTS panel_runs (
            id TEXT PRIMARY KEY, submission_id TEXT UNIQUE, request_hash TEXT NOT NULL,
            state TEXT NOT NULL, pid INTEGER, created_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS panel_events (
            sequence INTEGER PRIMARY KEY, run_id TEXT NOT NULL, at TEXT NOT NULL,
            state TEXT NOT NULL, detail TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS panel_events_no_update BEFORE UPDATE ON panel_events
            BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS panel_events_no_delete BEFORE DELETE ON panel_events
            BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;''')
        return db

    def get(self, identifier):
        directory = self.directory(identifier)
        record = json.loads((directory / 'manifest.json').read_text())
        with closing(sqlite3.connect(self.database.as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute('SELECT * FROM panel_runs WHERE id=?', (identifier,)).fetchone()
            record['events'] = [dict(event) for event in db.execute('SELECT * FROM panel_events WHERE run_id=? ORDER BY sequence', (identifier,))]
        if row:
            record['status'] = row['state']
            if row['pid'] and row['state'] in ACTIVE:
                try:
                    os.kill(row['pid'], 0)
                except PermissionError:
                    pass  # The worker exists, but belongs to a different permission context.
                except ProcessLookupError:
                    record['status'] = 'interrupted'
                    record['error'] = 'Worker exited before a final status. Saved evidence is retained; no automatic retry.'
        if row and not row['pid'] and row['state'] in ACTIVE and (datetime.now(timezone.utc) - datetime.fromisoformat(row['created_at'])).total_seconds() > 30:
            record['status'] = 'interrupted'
            record['error'] = 'Dispatch was interrupted before worker acknowledgement. Inspect saved evidence before any new run.'
        if (directory / 'stop_requested.json').exists():
            record['stop_requested'] = True
            if record['status'] in ACTIVE:
                record['progress_message'] = 'Stop requested. The current admitted action will finish; no next action will start.'
        record.update(self.call_totals(directory))
        return record

    @staticmethod
    def call_totals(directory):
        resource = directory / 'resources.sqlite3'
        if not resource.is_file() or resource.is_symlink():
            return {}
        with closing(sqlite3.connect(resource.as_uri() + '?mode=ro', uri=True)) as db:
            return {'model_calls': db.execute('SELECT count(*) FROM calls').fetchone()[0]}

    def event(self, identifier, state, detail):
        with closing(self._connect()) as db, db:
            db.execute('UPDATE panel_runs SET state=? WHERE id=?', (state, identifier))
            db.execute('INSERT INTO panel_events(run_id,at,state,detail) VALUES(?,?,?,?)', (identifier, now(), state, detail))
        path = self.directory(identifier) / 'manifest.json'
        record = json.loads(path.read_text())
        record.update(status=state, updated_at=now(), progress_message=detail)
        record.update(self.call_totals(path.parent))
        if state in ('failed', 'interrupted'):
            record['error'] = detail
        write_json(path, record)

    def start(self, request, *, launcher=None, authorization=None):
        request = StartRun.model_validate(request)
        value = request.model_dump(mode='json')
        identity = digest(value)
        if request.allow_network is not True:
            raise ValueError('New runs require online collection research')
        template = load_config(REPOSITORY / 'configs/overnight.yaml')
        if request.render_profile not in template.render_profiles:
            raise ValueError('Select an available named render profile')
        initial_images = request.directions * request.images_per_direction
        cycle = (DevelopmentCycle(max_steps=request.development_steps, research_steps=request.development_research_steps)
                 if request.study_development else None)
        renderer_cap = initial_images + (cycle.max_steps if cycle else 0)
        reasoning_cap = request.research_steps + initial_images + (2 * cycle.max_steps + 1
            + cycle.research_sessions * cycle.research_steps if cycle else 0)
        config = template.model_dump(mode='json') | {
            'session_hours': request.minutes / 60, 'handoff_reserve_minutes': 5,
            'renderer_attempt_cap': renderer_cap,
            'reasoning_token_cap': reasoning_cap * request.total_token_limit,
            'per_call_total_tokens': request.total_token_limit,
            'default_render_profile': request.render_profile,
            'cache_dir': str((REPOSITORY / template.cache_dir).resolve())}
        config = OvernightConfig.model_validate(config).model_dump(mode='json')
        identifier = 'studio-' + uuid.uuid4().hex[:20]
        directory = self.directory(identifier)
        with closing(self._connect()) as db:
            db.execute('BEGIN IMMEDIATE')
            prior = db.execute('SELECT * FROM panel_runs WHERE submission_id=?', (request.submission_id,)).fetchone()
            if prior:
                if prior['request_hash'] != identity:
                    raise ValueError('This submission ID already belongs to a different question or settings')
                return self.get(prior['id'])
            directory.mkdir(parents=True)
            frozen = {'request': value, 'resource_config': config,
                      'code_identity': source_identity(), 'created_at': now(),
                      'authorization': authorization or 'Operator pressed Start run with the displayed local inference, image, time and network limits.',
                      'authorship': {'question': 'operator', 'pipeline_and_defaults': 'operator-authored UI policy',
                                     'research_and_artistic_hypotheses': 'frozen_Qwen'}}
            immutable_json(directory / 'request.json', frozen)
            immutable_json(directory / 'request_identity.json', {'sha256': sha256(directory / 'request.json')})
            manifest = {'kind': 'artistic_studio_run', 'run_id': identifier, 'status': 'queued',
                        'title': request.title or request.question[:100], 'brief': request.question,
                        'context': request.context, 'created_at': now(),
                        'max_calls': reasoning_cap + renderer_cap,
                        'model_calls': 0, 'images': [], 'stages': ['research', 'render', 'inspect', 'development_cycle', 'review_or_stop'] if request.study_development else ['research', 'render'],
                        'cost_accounting': {'root': str(directory), 'run_ids': ['research_' + identifier, 'research_render_' + identifier]
                            + (['studies_' + identifier] + [prefix + cycle_id(identifier, i)
                                for i in range(cycle.max_steps) for prefix in ('research_', 'research_render_')] if cycle else [])},
                        'progress_message': 'Queued for the local studio worker.'}
            write_json(directory / 'manifest.json', manifest)
            db.execute('INSERT INTO panel_runs VALUES(?,?,?,?,?,?)', (identifier, request.submission_id, identity, 'queued', None, now()))
            db.execute('INSERT INTO panel_events(run_id,at,state,detail) VALUES(?,?,?,?)', (identifier, now(), 'queued', 'Run requested.'))
            db.commit()
        try:
            with (directory / 'worker.log').open('ab') as stream:
                child = (launcher or subprocess.Popen)([sys.executable, '-m', 'kusama.panel_runs', '--root', str(self.root),
                    '--coordinator', str(self.coordinator), '--id', identifier], cwd=REPOSITORY,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            with closing(self._connect()) as db, db:
                db.execute('UPDATE panel_runs SET pid=? WHERE id=?', (child.pid, identifier))
        except Exception as exc:
            self.event(identifier, 'failed', 'Worker could not start: ' + str(exc))
        return self.get(identifier)

    def stop(self, identifier):
        record = self.get(identifier)
        if record['status'] not in ACTIVE:
            return record
        immutable_json(self.directory(identifier) / 'stop_requested.json', {'run_id': identifier, 'action': 'stop_before_next_dispatch'})
        with closing(self._connect()) as db, db:
            db.execute('INSERT INTO panel_events(run_id,at,state,detail) VALUES(?,?,?,?)', (identifier, now(), 'stop_requested', 'Stop after the current admitted action.'))
        return record | {'stop_requested': True}

    def execute(self, identifier, *, research_runner=None, render_runner=None):
        from .research import plan_research, run_research
        from .research_render import plan_render, run_render
        from .config import Config
        directory = self.directory(identifier)
        # A worker may be invoked only once. An interrupted run requires evidence reconciliation.
        with (directory / '.execution.lock').open('a') as run_lock:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (directory / 'execution_started.json').exists():
                raise ValueError('This run already started; automatic redispatch is disabled')
            try:
                with closing(self._connect()) as db, db:
                    db.execute('UPDATE panel_runs SET pid=? WHERE id=?', (os.getpid(), identifier))
                self.coordinator.mkdir(parents=True, exist_ok=True)
                with (self.coordinator / '.panel_worker.lock').open('a') as shared:
                    # Panel workers queue without loading any model while another worker owns the slot.
                    fcntl.flock(shared, fcntl.LOCK_EX)
                    def check_stop():
                        if (directory / 'stop_requested.json').exists():
                            raise StopRequested('Stopped before the next action; completed work is preserved.')
                    check_stop()
                    frozen = json.loads((directory / 'request.json').read_text())
                    if sha256(directory / 'request.json') != json.loads((directory / 'request_identity.json').read_text())['sha256'] or source_identity() != frozen['code_identity']:
                        raise SpecMismatch('Frozen inputs or implementation changed; this run cannot be dispatched')
                    request = StartRun.model_validate(frozen['request'])
                    immutable_json(directory / 'execution_started.json', {'at': now()})
                    config = OvernightConfig.model_validate(frozen['resource_config'])
                    with SessionBudget(directory, config):
                        pass
                    render_dir = directory / 'render'
                    renderer = Config(output_dir=render_dir, cache_dir=Path(config.cache_dir), device=config.device,
                        local_files_only=True, max_retries=0, **config.render_profiles[request.render_profile])
                    self.event(identifier, 'researching', 'Exploring the question and inspecting references.')
                    plan_research(directory / 'research', {
                        'id': identifier, 'problem': request.question, 'authorship': frozen['authorship']['question'],
                        'project_id': identifier, 'brief_id': identifier, 'resource_root': str(directory),
                        'prior_context': request.context, 'seed_artists': request.seed_artists,
                        'seed_works': request.seed_works, 'medium_constraint': request.medium_constraint,
                        'source_config': {'allow_network': request.allow_network, 'max_requests': 48, 'max_acquisitions': 12},
                        'render_prompt_contract': {'model_revision': renderer.model_revision,
                                                   'max_sequence_length': renderer.max_sequence_length},
                        'max_steps': request.research_steps, 'max_hypotheses': request.directions,
                        'total_token_limit': config.per_call_total_tokens,
                        'study_development': request.study_development,
                        'random_exposure_count': request.random_exposure_count,
                        'max_wall_seconds': (request.minutes - 5) * 60,
                        'cache_dir': config.cache_dir, 'device': config.device})
                    research = (research_runner or run_research)(directory / 'research', before_dispatch=check_stop)
                    check_stop()
                    hypotheses = research.get('hypotheses', [])
                    rejected = research['status'] == 'response_rejected'
                    rejection_detail = ''
                    if rejected and research.get('steps'):
                        last = research['steps'][-1]
                        rejection_detail = f"response-{last['index']:03}.json: {last.get('error', 'Rejected response')}"
                    if (research['status'] not in ('hypotheses_ready', 'model_stop', 'step_limit', 'stagnated', 'context_paused')
                            and not (rejected and hypotheses)):
                        raise ValueError('Research stopped with ' + research['status'] + '; ' +
                                         (rejection_detail or 'inspect the retained responses.'))
                    if not hypotheses:
                        if research['status'] == 'context_paused':
                            self.event(identifier, 'partial', 'Research saved at the context limit. '
                                'Sources, notes and completed calls are retained; the oversized decision was not charged.')
                            return
                        self.event(identifier, 'partial', f"Research ended ({research['status']}) without a committed artistic direction. No images were generated.")
                        return
                    self.event(identifier, 'rendering', 'Generating images from the saved artistic directions.')
                    plan_render(render_dir, {'id': identifier, 'research_output': str(directory / 'research'),
                        'hypothesis_ids': [h['id'] for h in hypotheses],
                        'seeds': [23 + i for i in range(len(hypotheses) * request.images_per_direction)],
                        'resource_root': str(directory), 'render_profile_id': request.render_profile,
                        'render_config': renderer.to_dict(), 'selection_origin': 'all_model_hypotheses',
                        'selection_reason': 'Generate every provisional model study within the operator-selected cap.',
                        'study_development': request.study_development, 'register_review': False,
                        'development_cycle': DevelopmentCycle(max_steps=request.development_steps,
                            research_steps=request.development_research_steps).model_dump(mode='json') if request.study_development else None})
                    report = (render_runner or run_render)(render_dir, before_dispatch=check_stop,
                        on_stage=lambda message: self.event(identifier, 'rendering', message))
                    message = f"Saved {len(report.get('images', []))} images. Inspect the results and research below."
                    if report.get('study_development'):
                        development = report['study_development']
                        message = f"Saved {len(report.get('images', []))} initial study images. " + development['reason']
                    if rejected:
                        message = ('Research stopped with response_rejected; rendered only earlier committed directions. '
                                   + message + ' ' + rejection_detail)
                    if research['status'] == 'context_paused':
                        message = 'Research saved at the context limit; retained committed studies were rendered. ' + message
                    self.event(identifier, 'completed' if report['status'] == 'complete' and not rejected
                               and research['status'] != 'context_paused' else 'partial', message)
            except StopRequested as exc:
                self.event(identifier, 'stopped', str(exc))
            except BaseException as exc:
                self.event(identifier, 'interrupted' if not isinstance(exc, Exception) else 'failed', f'{type(exc).__name__}: {exc}')
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--id', required=True)
    args = parser.parse_args(argv)
    from .experiment import handle_termination
    with handle_termination():
        PanelRuns(args.root, args.coordinator).execute(args.id)


if __name__ == '__main__':
    main()
