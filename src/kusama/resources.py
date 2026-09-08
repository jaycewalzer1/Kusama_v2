"""Session-wide dispatch limits, separate from per-job invocation ledgers.

Reservations are conservative until a durable completion records actual usage.
No existing reservation ever authorizes a second call. An uncertain reservation
stays charged; a new retry requires a new key and the owning job's retry policy.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import write_json
from .ledger import BudgetExceeded, JobLocked, SpecMismatch


def now():
    return datetime.now(timezone.utc)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class OvernightConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    session_hours: float = Field(default=8, gt=0, le=24)
    handoff_reserve_minutes: int = Field(default=45, ge=0)
    renderer_attempt_cap: int = Field(default=160, ge=0, le=160)
    reasoning_token_cap: int = Field(default=500000, ge=0, le=500000)
    per_call_total_tokens: int = Field(default=8192, gt=0, le=8192)
    paid_api_usd_cap: float = Field(default=0, ge=0, le=0)
    asset_byte_cap: int = Field(default=2 * 1024**3, gt=0, le=2 * 1024**3)
    minimum_free_bytes: int = Field(default=20 * 1024**3, ge=20 * 1024**3)
    minimum_free_fraction: float = Field(default=.15, ge=.15, le=.9)
    reference_target: int = Field(default=200, ge=1, le=300)
    reference_cap: int = Field(default=300, ge=1, le=300)
    download_attempt_cap: int = Field(default=3, ge=1, le=3)
    schema_repair_cap: int = Field(default=1, ge=0, le=1)
    active_gpu_workloads: Literal[1] = 1
    cache_dir: str = '.cache/huggingface'
    device: Literal['mps', 'cpu', 'cuda'] = 'mps'
    clip_revision: str = Field(pattern=r'^[0-9a-f]{40}$')
    render_profiles: dict[str, dict]
    default_render_profile: str
    episode_allocation: dict[str, int]
    full_episode_allocation: dict[str, int]
    shelf_allocation: dict[str, int]
    director: dict
    observer: dict
    seed: int = Field(default=23, ge=0)

    @model_validator(mode='after')
    def valid(self):
        if self.handoff_reserve_minutes >= self.session_hours * 60:
            raise ValueError('Handoff reserve must leave an execution window')
        if self.default_render_profile not in self.render_profiles:
            raise ValueError('Unknown render profile')
        if self.reference_target > self.reference_cap:
            raise ValueError('Target exceeds cap')
        for allocation in (self.episode_allocation, self.full_episode_allocation, self.shelf_allocation):
            if any(type(n) is not int or n < 0 for n in allocation.values()):
                raise ValueError('Allocations require nonnegative integer counts')
        if sum(self.episode_allocation.values()) > 16:
            raise ValueError('Default development episode may not exceed 16 attempts')
        return self


def load_config(path=Path('configs/overnight.yaml')):
    # JSON is a YAML subset; no permissive object construction or executable tags.
    return OvernightConfig.model_validate_json(Path(path).read_bytes())


class SessionBudget:
    def __init__(self, root, config=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'resources.sqlite3', timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS session (id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL,
                started_at TEXT NOT NULL, deadline TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, kind TEXT NOT NULL,
                status TEXT NOT NULL, token_reservation INTEGER NOT NULL, actual_tokens INTEGER,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resource_events (sequence INTEGER PRIMARY KEY,
                kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE TRIGGER IF NOT EXISTS resource_events_no_update BEFORE UPDATE ON resource_events
                BEGIN SELECT RAISE(ABORT, 'resource events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS resource_events_no_delete BEFORE DELETE ON resource_events
                BEGIN SELECT RAISE(ABORT, 'resource events are append-only'); END;
        ''')
        with self.transaction():
            row = self.db.execute('SELECT * FROM session WHERE id=1').fetchone()
            if row:
                self.config = OvernightConfig.model_validate_json(row['config'])
                if config is not None:
                    supplied = config if isinstance(config, OvernightConfig) else load_config(config)
                    if supplied != self.config:
                        raise SpecMismatch('Session configuration is frozen; do not silently change limits')
                self.started_at, self.deadline = row['started_at'], row['deadline']
            else:
                self.config = config if isinstance(config, OvernightConfig) else load_config(config or Path('configs/overnight.yaml'))
                start = now()
                self.started_at = start.isoformat()
                self.deadline = (start + timedelta(hours=self.config.session_hours)).isoformat()
                self.db.execute('INSERT INTO session VALUES(1,?,?,?)',
                    (canonical(self.config.model_dump()), self.started_at, self.deadline))
                self._event('session_created', {'config': self.config.model_dump(), 'deadline': self.deadline})
        self.export()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def _event(self, kind, payload):
        self.db.execute('INSERT INTO resource_events(kind,payload,created_at) VALUES(?,?,?)',
            (kind, canonical(payload), now().isoformat()))

    def summary(self):
        rows = [dict(r) for r in self.db.execute('SELECT * FROM calls ORDER BY created_at,key')]
        attempts = sum(r['kind'] == 'renderer' for r in rows)
        used_tokens = sum(r['actual_tokens'] if r['actual_tokens'] is not None else r['token_reservation'] for r in rows)
        return {'schema_version': 1, 'started_at': self.started_at, 'deadline': self.deadline,
                'config_sha256': hashlib.sha256(canonical(self.config.model_dump()).encode()).hexdigest(),
                'renderer_attempts': attempts, 'renderer_remaining': self.config.renderer_attempt_cap - attempts,
                'reasoning_tokens_measured': sum(r['actual_tokens'] or 0 for r in rows),
                'reasoning_tokens_charged_or_reserved': used_tokens,
                'reasoning_tokens_remaining': self.config.reasoning_token_cap - used_tokens,
                'unsettled': [r for r in rows if r['status'] in ('running', 'uncertain')],
                'calls': rows, 'paid_api_usd': 0, 'electricity_and_hardware_usd': None}

    def estimated_seconds(self, kind, supplied=0):
        """Use measured slow calls to protect handoff time, including edit latency.

        This changes admission only, never a frozen renderer or prompt setting.
        Model loads are excluded; a separate workload-level preflight may also
        reserve them. The caller's more conservative estimate always wins.
        """
        estimate = max(float(supplied), 300 if kind == 'renderer' else 90)
        path = self.root / 'costs.sqlite3'
        if path.exists():
            from .costs import CostJournal
            with CostJournal(self.root, read_only=True) as costs:
                for record in costs.summary()['records']:
                    start, finish = record['start'], record['finish']
                    model = start.get('model', '').casefold()
                    relevant = 'flux' in model if kind == 'renderer' else 'qwen' in model
                    if relevant and start.get('scope') in ('inference', 'backend_request') and finish:
                        estimate = max(estimate, 1.25 * (finish.get('runtime_seconds') or 0))
        return estimate

    def reserve(self, key, kind, max_tokens=0, estimated_seconds=0):
        if not isinstance(key, str) or not key.strip() or len(key) > 300:
            raise ValueError('A bounded stable invocation key is required')
        if kind not in ('renderer', 'reasoning'):
            raise ValueError('Unknown resource kind')
        if type(max_tokens) is not int or max_tokens < 0 or max_tokens > self.config.per_call_total_tokens:
            raise ValueError('Invalid per-call token reservation')
        if (kind == 'reasoning' and max_tokens == 0) or (kind == 'renderer' and max_tokens):
            raise ValueError('Reasoning must reserve tokens; renderer must not')
        if estimated_seconds < 0:
            raise ValueError('Estimate must be nonnegative')
        effective_seconds = self.estimated_seconds(kind, estimated_seconds)
        with self.transaction():
            old = self.db.execute('SELECT * FROM calls WHERE key=?', (key,)).fetchone()
            if old:
                if old['kind'] != kind or old['token_reservation'] != max_tokens:
                    raise SpecMismatch('Reservation key reused with different resources')
                return False
            dispatch_deadline = datetime.fromisoformat(self.deadline) - timedelta(minutes=self.config.handoff_reserve_minutes)
            if now() + timedelta(seconds=effective_seconds) >= dispatch_deadline:
                raise BudgetExceeded('Session execution window exhausted; handoff reserve retained')
            disk = shutil.disk_usage(self.root)
            if disk.free < max(self.config.minimum_free_bytes, disk.total * self.config.minimum_free_fraction):
                raise BudgetExceeded('Disk free-space floor reached')
            state = self.summary()
            if kind == 'renderer' and state['renderer_remaining'] < 1:
                raise BudgetExceeded('Session renderer attempt cap exhausted')
            if state['reasoning_tokens_remaining'] < max_tokens:
                raise BudgetExceeded('Session reasoning token cap exhausted')
            stamp = now().isoformat()
            self.db.execute('INSERT INTO calls VALUES(?,?,?, ?,NULL,?,?)', (key, kind, 'running', max_tokens, stamp, stamp))
            self._event('reserved', {'key': key, 'kind': kind, 'max_tokens': max_tokens,
                                     'estimated_seconds': effective_seconds,
                                     'time_policy': 'max(caller, conservative floor, 1.25 * measured slowest matching inference)'})
        self.export()
        return True

    def settle(self, key, status, actual_tokens=None):
        if status not in ('completed', 'failed', 'uncertain'):
            raise ValueError('Invalid completion state')
        with self.transaction():
            row = self.db.execute('SELECT * FROM calls WHERE key=?', (key,)).fetchone()
            if row is None:
                raise ValueError('Cannot settle an unreserved call')
            if actual_tokens is not None and (type(actual_tokens) is not int or actual_tokens < 0 or actual_tokens > row['token_reservation']):
                raise ValueError('Actual token usage exceeds reservation or is invalid')
            if row['status'] in ('completed', 'failed'):
                if row['status'] != status or row['actual_tokens'] != actual_tokens:
                    raise SpecMismatch('Completed accounting is immutable')
                return
            self.db.execute('UPDATE calls SET status=?,actual_tokens=?,updated_at=? WHERE key=?',
                (status, actual_tokens, now().isoformat(), key))
            self._event('settled', {'key': key, 'status': status, 'actual_tokens': actual_tokens})
        self.export()

    @contextmanager
    def workload_lock(self):
        # OS releases the exclusive device lease after a crash. Invocation
        # reservations survive, so a new worker must reconcile their outputs.
        with (self.root / 'gpu.lock').open('a+') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise JobLocked('Another assigned model workload is active') from exc
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def export(self):
        # All projections are serialized separately from the long GPU lease.
        # Read after taking this lock so a later export cannot overwrite a more
        # current projection with a snapshot captured before waiting.
        with (self.root / 'resource_export.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                self.db.execute('BEGIN')
                summary = self.summary()
                events = [dict(r) for r in self.db.execute('SELECT * FROM resource_events ORDER BY sequence')]
                self.db.execute('COMMIT')
                write_json(self.root / 'resource_summary.json', summary)
                with (self.root / 'resource_events.jsonl.tmp').open('w') as out:
                    for row in events:
                        out.write(canonical({**row, 'payload': json.loads(row['payload'])}) + '\n')
                (self.root / 'resource_events.jsonl.tmp').replace(self.root / 'resource_events.jsonl')
            finally:
                if self.db.in_transaction:
                    self.db.execute('ROLLBACK')
                fcntl.flock(lock, fcntl.LOCK_UN)
