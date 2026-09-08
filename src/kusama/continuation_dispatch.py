"""Durable localhost dispatch of already-frozen continuation IDs.

The HTTP process never owns model weights. A worker drains only an existing
studio batch, using its immutable inputs, process locks and inference ledger.
"""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

from pydantic import BaseModel, ConfigDict, Field

from .backend import FluxBackend
from .continuation import ContinuationStore, _identifier
from .ledger import BudgetExceeded, JobLocked, Ledger
from .resources import SessionBudget
from .reviews import ReviewConflict, ReviewNotFound


class DispatchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    submission_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')


def _now():
    return datetime.now(timezone.utc).isoformat()


class DispatchStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    @staticmethod
    def _schema(db):
        db.executescript('''
            CREATE TABLE IF NOT EXISTS continuation_dispatch_requests (
                continuation_id TEXT PRIMARY KEY, submission_id TEXT UNIQUE NOT NULL,
                state TEXT NOT NULL, pid INTEGER, log_path TEXT NOT NULL,
                requested_at TEXT NOT NULL, updated_at TEXT NOT NULL, detail TEXT);
            CREATE TABLE IF NOT EXISTS continuation_dispatch_events (
                sequence INTEGER PRIMARY KEY, continuation_id TEXT NOT NULL,
                state TEXT NOT NULL, created_at TEXT NOT NULL, detail TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS continuation_dispatch_events_no_update
                BEFORE UPDATE ON continuation_dispatch_events BEGIN
                SELECT RAISE(ABORT, 'dispatch events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS continuation_dispatch_events_no_delete
                BEFORE DELETE ON continuation_dispatch_events BEGIN
                SELECT RAISE(ABORT, 'dispatch events are append-only'); END;
        ''')

    def get(self, continuation_id):
        _identifier(continuation_id)
        database = self.root / 'jobs.sqlite3'
        if not database.is_file():
            return None
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
            db.row_factory = sqlite3.Row
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='continuation_dispatch_requests'").fetchone():
                return None
            row = db.execute('SELECT * FROM continuation_dispatch_requests WHERE continuation_id=?', (continuation_id,)).fetchone()
            if not row:
                return None
            return dict(row) | {'events': [dict(e) for e in db.execute(
                'SELECT state,created_at,detail FROM continuation_dispatch_events WHERE continuation_id=? ORDER BY sequence',
                (continuation_id,))]}

    @staticmethod
    def _state(ledger, continuation_id, state, detail):
        with ledger._transaction():
            ledger.connection.execute('UPDATE continuation_dispatch_requests SET state=?,updated_at=?,detail=? WHERE continuation_id=?',
                                      (state, _now(), detail, continuation_id))
            ledger.connection.execute('INSERT INTO continuation_dispatch_events(continuation_id,state,created_at,detail) VALUES(?,?,?,?)',
                                      (continuation_id, state, _now(), detail))

    def _preflight(self):
        with SessionBudget(self.root) as budget, budget.workload_lock():
            if budget.summary()['renderer_remaining'] < 1:
                raise BudgetExceeded('Session renderer cap exhausted')
            deadline = datetime.fromisoformat(budget.deadline) - timedelta(minutes=budget.config.handoff_reserve_minutes)
            if datetime.now(timezone.utc) >= deadline:
                raise BudgetExceeded('Execution window exhausted; handoff reserve retained')
            disk = shutil.disk_usage(self.root)
            if disk.free < max(budget.config.minimum_free_bytes, disk.total * budget.config.minimum_free_fraction):
                raise BudgetExceeded('Disk free-space floor reached')

    def submit(self, continuation_id, request: DispatchRequest | dict, *, launcher=None):
        continuation_id = _identifier(continuation_id)
        launcher = launcher or subprocess.Popen
        request = DispatchRequest.model_validate(request)
        store = ContinuationStore(self.root)
        spec = store._spec(continuation_id)
        if Path(spec.get('resource_root', self.root)).resolve() != self.root:
            raise ReviewConflict('Browser dispatch requires the dashboard artifact root to be the frozen session budget root')
        with Ledger(self.root / 'jobs.sqlite3') as ledger, ledger.lock('api_submit_' + continuation_id):
            self._schema(ledger.connection)
            previous = self.get(continuation_id)
            if previous:
                if previous['submission_id'] != request.submission_id:
                    raise ReviewConflict('This continuation already has a dispatch request; retry its existing submission ID')
                if previous['state'] == 'completed':
                    return previous
                try:
                    with ledger.lock('api_worker_' + continuation_id):
                        pass
                except JobLocked:
                    return previous
                # A just-launched process needs time to acquire its worker lease.
                if previous['state'] == 'queued' and (datetime.now(timezone.utc) - datetime.fromisoformat(previous['updated_at'])).total_seconds() < 10:
                    return previous
            current = store.get(continuation_id)
            settled = current['status'] in ('awaiting_selection', 'exhausted', 'completed', 'stopped')
            if not settled:
                self._preflight()
            directory = self.root / 'continuations' / continuation_id
            log_path = directory / f'dispatch_{request.submission_id}.log'
            if log_path.is_symlink() or any(p.is_symlink() for p in [directory, *directory.parents] if p.is_relative_to(self.root)):
                raise ReviewConflict('Dispatch output paths must not be symlinks')
            directory.mkdir(parents=True, exist_ok=True)
            if previous is None:
                try:
                    with ledger._transaction():
                        ledger.connection.execute('INSERT INTO continuation_dispatch_requests VALUES(?,?,?,?,?,?,?,?)',
                            (continuation_id, request.submission_id, 'queued', None, str(log_path), _now(), _now(), 'Explicit localhost execution request'))
                except sqlite3.IntegrityError as exc:
                    raise ReviewConflict('Submission ID is already used for another continuation') from exc
            if settled:
                self._state(ledger, continuation_id, 'completed', 'Existing frozen batch is already settled; no process or model call launched')
                return self.get(continuation_id)
            self._state(ledger, continuation_id, 'queued', 'Explicit request accepted; worker will reconcile the existing action ledgers')
            command = [sys.executable, '-m', 'kusama.continuation_dispatch', '--artifacts', str(self.root),
                       '--id', continuation_id, '--submission-id', request.submission_id]
            try:
                with log_path.open('ab') as log:
                    process = launcher(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, shell=False)
                with ledger._transaction():
                    ledger.connection.execute('UPDATE continuation_dispatch_requests SET pid=? WHERE continuation_id=?',
                                              (process.pid, continuation_id))
            except BaseException as exc:
                self._state(ledger, continuation_id, 'launch_failed', f'{type(exc).__name__}: {exc}')
                raise
            return self.get(continuation_id)

    def work(self, continuation_id, submission_id, *, backend_factory=FluxBackend):
        _identifier(continuation_id)
        with Ledger(self.root / 'jobs.sqlite3') as ledger, ledger.lock('api_worker_' + continuation_id):
            self._schema(ledger.connection)
            record = self.get(continuation_id)
            if not record or record['submission_id'] != submission_id:
                raise ReviewNotFound('Worker requires its exact registered dispatch request')
            if record['state'] == 'completed':
                return ContinuationStore(self.root).get(continuation_id)
            self._state(ledger, continuation_id, 'running', 'Worker holds the continuation dispatch lease')
            try:
                result = ContinuationStore(self.root).execute(continuation_id, backend_factory, resource_root=self.root)
                settled = result['status'] in ('awaiting_selection', 'exhausted', 'completed', 'stopped')
                self._state(ledger, continuation_id, 'completed' if settled else 'needs_attention',
                            f'Frozen batch execution status: {result["status"]}; human evaluation remains separate')
                return result
            except BaseException as exc:
                self._state(ledger, continuation_id, 'needs_attention', f'{type(exc).__name__}: {exc}')
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', required=True, type=Path)
    parser.add_argument('--id', required=True)
    parser.add_argument('--submission-id', required=True)
    args = parser.parse_args(argv)
    request = DispatchRequest(submission_id=args.submission_id)
    result = DispatchStore(args.artifacts).work(args.id, request.submission_id)
    print(json.dumps({'id': result['id'], 'status': result['status'], 'model_calls': result['model_calls']}))


if __name__ == '__main__':
    main()
