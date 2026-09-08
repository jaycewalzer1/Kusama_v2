"""SQLite authority for resumable jobs, immutable artifacts, and inference budgets.

Hold ``lock(job_id)`` for an entire execution, but never a SQLite transaction
during inference. A reservation records a *possible* model invocation: it remains
charged after a crash and must not be used to repeat an uncertain invocation.
JSONL files are derived event exports; SQLite is always authoritative.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import uuid


SCHEMA_VERSION = 1


class LedgerError(RuntimeError):
    """A persisted execution invariant would be violated."""


class SpecMismatch(LedgerError):
    """A stable job or artifact identifier has different immutable contents."""


class BudgetExceeded(LedgerError):
    """An inference reservation would exceed a job or ancestor budget."""


class JobLocked(LedgerError):
    """Another executor holds this job's filesystem lock."""


class EventExportError(LedgerError):
    """An event export requires explicit recovery from the database."""


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Ledger:
    def __init__(self, db_path: Path):
        self.path = Path(db_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        version = self.connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, SCHEMA_VERSION):
            self.close()
            raise LedgerError(f"Unsupported ledger schema version: {version}")
        self.connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                schema_version INTEGER NOT NULL,
                spec_json TEXT NOT NULL,
                spec_fingerprint TEXT NOT NULL,
                max_calls INTEGER NOT NULL CHECK(max_calls > 0),
                directory TEXT NOT NULL UNIQUE,
                parent_id TEXT REFERENCES jobs(job_id),
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                attempt_index INTEGER NOT NULL CHECK(attempt_index >= 0),
                schema_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(job_id, attempt_index)
            );
            CREATE TABLE IF NOT EXISTS reservations (
                attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reservation_charges (
                attempt_id TEXT NOT NULL REFERENCES reservations(attempt_id),
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                PRIMARY KEY(attempt_id, job_id)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                attempt_index INTEGER,
                kind TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(job_id, path)
            );
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
            CREATE INDEX IF NOT EXISTS events_job ON events(job_id, sequence);
            CREATE INDEX IF NOT EXISTS charges_job ON reservation_charges(job_id);
            PRAGMA user_version=1;
            COMMIT;
        """)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def _transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _event(self, job_id, event_type, payload):
        self.connection.execute(
            "INSERT INTO events(event_id,schema_version,job_id,type,payload_json,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (uuid.uuid4().hex, SCHEMA_VERSION, job_id, event_type, _json(payload), _now()))

    def _job(self, job_id):
        row = self.connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown job: {job_id}")
        return row

    def create_job(self, job_id, spec, max_calls, directory, parent_id=None):
        """Create a stable job or validate every immutable field on reopening.

        Include the effective execution configuration and input hashes in ``spec``.
        A child must be registered before running it; all ancestors share its cost.
        """
        if not isinstance(job_id, str) or not job_id.strip():
            raise ValueError("job_id must be a nonempty string")
        if not isinstance(spec, dict):
            raise ValueError("spec must be an object")
        if type(max_calls) is not int or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        encoded = _json(spec)
        fingerprint = hashlib.sha256(encoded.encode()).hexdigest()
        directory = str(Path(directory).resolve())
        with self._transaction():
            row = self.connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is not None:
                expected = (fingerprint, max_calls, directory, parent_id)
                actual = (row['spec_fingerprint'], row['max_calls'], row['directory'], row['parent_id'])
                if actual != expected:
                    raise SpecMismatch(f"Job {job_id} does not match its immutable spec, budget, directory, or parent")
            else:
                if parent_id is not None:
                    self._job(parent_id)
                if self.connection.execute("SELECT 1 FROM jobs WHERE directory=?", (directory,)).fetchone():
                    raise SpecMismatch(f"Job directory is already registered: {directory}")
                now = _now()
                self.connection.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, SCHEMA_VERSION, encoded, fingerprint, max_calls, directory,
                     parent_id, "pending", "{}", now, now))
                self._event(job_id, "job_created", {"spec_fingerprint": fingerprint,
                    "max_calls": max_calls, "directory": directory, "parent_id": parent_id})
        return self.get_job(job_id)

    def open_job(self, job_id, *, spec, max_calls, directory, parent_job_id=None):
        return self.create_job(job_id, spec, max_calls, directory, parent_job_id)

    def get_job(self, job_id):
        result = dict(self._job(job_id))
        result['spec'] = json.loads(result.pop('spec_json'))
        result['payload'] = json.loads(result.pop('payload_json'))
        result['model_calls'] = self.call_count(job_id)
        return result

    def update_job(self, job_id, status=None, payload=None):
        if payload is not None and not isinstance(payload, dict):
            raise ValueError("job payload must be an object")
        with self._transaction():
            row = self._job(job_id)
            merged = json.loads(row['payload_json']) | (payload or {})
            encoded = _json(merged)
            new_status = status if status is not None else row['status']
            if not isinstance(new_status, str) or not new_status:
                raise ValueError("job status must be a nonempty string")
            if encoded != row['payload_json'] or new_status != row['status']:
                self.connection.execute("UPDATE jobs SET status=?,payload_json=?,updated_at=? WHERE job_id=?",
                    (new_status, encoded, _now(), job_id))
                self._event(job_id, "job_updated", {"status": new_status, "payload": merged})
        return self.get_job(job_id)

    @staticmethod
    def _attempt_id(job_id, index):
        if type(index) is not int or index < 0:
            raise ValueError("attempt index must be a nonnegative integer")
        return f"{job_id}:{index}"

    def get_attempt(self, job_id, index):
        attempt_id = self._attempt_id(job_id, index)
        row = self.connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            return None
        result = json.loads(row['payload_json'])
        result.update({key: row[key] for key in ('attempt_id', 'job_id', 'schema_version',
                       'status', 'created_at', 'updated_at')})
        result['index'] = row['attempt_index']
        result['checkpoint'] = json.loads(row['checkpoint_json'])
        result['reserved'] = self.connection.execute(
            "SELECT 1 FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone() is not None
        return result

    def _save_attempt(self, job_id, index, payload, status, checkpoint):
        attempt_id = self._attempt_id(job_id, index)
        row = self.connection.execute("SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
        if row is None:
            self._job(job_id)
        merged = (json.loads(row['payload_json']) if row else {}) | (payload or {})
        checkpoints = (json.loads(row['checkpoint_json']) if row else {}) | (checkpoint or {})
        new_status = status if status is not None else (
            (payload or {}).get('status') or (row['status'] if row else 'pending'))
        if not isinstance(new_status, str) or not new_status:
            raise ValueError("attempt status must be a nonempty string")
        encoded, encoded_checkpoints = _json(merged), _json(checkpoints)
        now = _now()
        if row is None:
            self.connection.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?)",
                (attempt_id, job_id, index, SCHEMA_VERSION, new_status, encoded,
                 encoded_checkpoints, now, now))
            event_type = 'attempt_created'
        elif (encoded, encoded_checkpoints, new_status) != (
                row['payload_json'], row['checkpoint_json'], row['status']):
            self.connection.execute(
                "UPDATE attempts SET status=?,payload_json=?,checkpoint_json=?,updated_at=? WHERE attempt_id=?",
                (new_status, encoded, encoded_checkpoints, now, attempt_id))
            event_type = 'attempt_updated'
        else:
            return
        self._event(job_id, event_type, {"attempt_id": attempt_id, "index": index,
                    "status": new_status, "payload": merged, "checkpoint": checkpoints})

    def save_attempt(self, job_id, index, payload=None, status=None, checkpoint=None):
        for name, value in (("payload", payload), ("checkpoint", checkpoint)):
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"attempt {name} must be an object")
        with self._transaction():
            self._save_attempt(job_id, index, payload, status, checkpoint)
        return self.get_attempt(job_id, index)

    def open_attempt(self, job_id, index, *, payload=None):
        return self.save_attempt(job_id, index, payload)

    def update_attempt(self, job_id, index, *, status=None, payload=None, checkpoint=None):
        return self.save_attempt(job_id, index, payload, status, checkpoint)

    def call_count(self, job_id):
        self._job(job_id)
        return self.connection.execute(
            "SELECT COUNT(*) FROM reservation_charges WHERE job_id=?", (job_id,)).fetchone()[0]

    def cost_summary(self, job_id, *, include_descendants=True):
        """Read separately journaled monetary scopes without modifying artifacts.

        Container jobs can aggregate descendant calls once; model-call budgets
        remain independent from these estimated/unknown currency amounts.
        """
        from .costs import CostJournal
        self._job(job_id)
        job_ids = [job_id]
        if include_descendants:
            job_ids = [row[0] for row in self.connection.execute("""
                WITH RECURSIVE descendants(job_id) AS (
                    SELECT job_id FROM jobs WHERE job_id=?
                    UNION ALL
                    SELECT child.job_id FROM jobs child
                    JOIN descendants parent ON child.parent_id=parent.job_id
                ) SELECT job_id FROM descendants
            """, (job_id,))]
        if not (self.path.parent / 'costs.sqlite3').exists():
            return {'schema_version': 1, 'run_id': job_id, 'included_run_ids': job_ids,
                    'calls': 0, 'open_calls': 0, 'currencies': {}, 'coverage': 'no_cost_records',
                    'note': 'No recorded monetary coverage; historical cost is unknown.'}
        with CostJournal(self.path.parent, read_only=True) as costs:
            result = costs.summary(run_ids=job_ids)
        result.update(run_id=job_id, included_run_ids=job_ids)
        return result

    def reserve_call(self, job_id, index):
        """Charge an attempt and all ancestor budgets together, exactly once.

        Return False for an existing reservation, including an uncertain or failed
        attempt. False is never authorization to repeat an inference call.
        """
        attempt_id = self._attempt_id(job_id, index)
        with self._transaction():
            self._job(job_id)
            if self.connection.execute("SELECT 1 FROM reservations WHERE attempt_id=?", (attempt_id,)).fetchone():
                return False
            charged_jobs = []
            current = job_id
            while current is not None:
                row = self._job(current)
                if self.call_count(current) >= row['max_calls']:
                    raise BudgetExceeded(f"Inference budget exhausted for job {current} ({row['max_calls']} calls)")
                charged_jobs.append(current)
                current = row['parent_id']
            self._save_attempt(job_id, index, None, "started", None)
            self.connection.execute("INSERT INTO reservations VALUES(?,?)", (attempt_id, _now()))
            self.connection.executemany("INSERT INTO reservation_charges VALUES(?,?)",
                [(attempt_id, charged_id) for charged_id in charged_jobs])
            self._event(job_id, "call_reserved", {"attempt_id": attempt_id,
                "index": index, "charged_jobs": charged_jobs})
            for charged_id in charged_jobs[1:]:
                self._event(charged_id, "child_call_reserved", {"attempt_id": attempt_id,
                    "child_job_id": job_id})
        return True

    def _artifact_path(self, job_id, path):
        directory = Path(self._job(job_id)['directory'])
        supplied = Path(path)
        absolute = (supplied if supplied.is_absolute() else directory / supplied).resolve()
        try:
            relative = absolute.relative_to(directory).as_posix()
        except ValueError as error:
            raise ValueError("Artifact must reside inside its job directory") from error
        if absolute == directory:
            raise ValueError("Artifact must be a file")
        return absolute, relative

    def register_artifact(self, job_id, relative_path, sha=None, *, attempt_index=None, kind=None):
        """Bind an existing file's relative path and content hash immutably."""
        absolute, relative = self._artifact_path(job_id, relative_path)
        actual_sha = _hash(absolute)
        if sha is not None and sha != actual_sha:
            raise SpecMismatch(f"Artifact does not match supplied hash: {relative}")
        artifact_id = hashlib.sha256(_json([job_id, relative]).encode()).hexdigest()
        with self._transaction():
            row = self.connection.execute("SELECT * FROM artifacts WHERE job_id=? AND path=?",
                                          (job_id, relative)).fetchone()
            if row is not None:
                if row['sha256'] != actual_sha:
                    raise SpecMismatch(f"Registered artifact changed: {relative}")
                if ((attempt_index is not None and row['attempt_index'] != attempt_index)
                        or (kind is not None and row['kind'] != kind)):
                    raise SpecMismatch(f"Registered artifact metadata changed: {relative}")
            else:
                if attempt_index is not None:
                    self._attempt_id(job_id, attempt_index)
                    if self.get_attempt(job_id, attempt_index) is None:
                        raise KeyError(f"Unknown attempt: {job_id}:{attempt_index}")
                self.connection.execute("INSERT INTO artifacts VALUES(?,?,?,?,?,?,?,?)",
                    (artifact_id, job_id, relative, actual_sha, SCHEMA_VERSION, attempt_index, kind, _now()))
                self._event(job_id, "artifact_registered", {"artifact_id": artifact_id,
                    "path": relative, "sha256": actual_sha, "attempt_index": attempt_index, "kind": kind})
        return dict(self.connection.execute("SELECT * FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone())

    def artifacts(self, job_id):
        self._job(job_id)
        return [dict(row) for row in self.connection.execute(
            "SELECT * FROM artifacts WHERE job_id=? ORDER BY path", (job_id,))]

    def verify_artifacts(self, job_id):
        records = self.artifacts(job_id)
        for record in records:
            path, _ = self._artifact_path(job_id, record['path'])
            if not path.is_file() or _hash(path) != record['sha256']:
                raise SpecMismatch(f"Registered artifact missing or changed: {record['path']}")
        return records

    @contextmanager
    def lock(self, job_id):
        """Nonblocking cross-process run lock; OS releases it after a crash."""
        lock_dir = self.path.parent / '.job-locks'
        lock_dir.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(_json([str(self.path), job_id]).encode()).hexdigest()
        # Lock files must persist: unlinking one allows two different locked inodes.
        with (lock_dir / f'{key}.lock').open('a+b') as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise JobLocked(f"Job {job_id} is already running") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def run_lock(self, job_id):
        return self.lock(job_id)

    def events(self, job_id=None):
        query, parameters = "SELECT * FROM events", ()
        if job_id is not None:
            self._job(job_id)
            query += " WHERE job_id=?"
            parameters = (job_id,)
        result = []
        for row in self.connection.execute(query + " ORDER BY sequence", parameters):
            event = dict(row)
            event['payload'] = json.loads(event.pop('payload_json'))
            result.append(event)
        return result

    def export_events(self, path, job_id=None, *, recover=False):
        """Append committed events, or explicitly recover a torn/mismatched log.

        Existing bytes must be a complete, exact prefix of the authoritative
        stream. Recovery uses an fsynced temporary file and atomic replacement.
        A crash at any export boundary is safe to retry with ``recover=True``.
        """
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_name(path.name + '.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            records = self.events(job_id)
            expected = ''.join(_json(event) + '\n' for event in records).encode('utf-8')
            existing = path.read_bytes() if path.exists() else b''
            valid_prefix = (not existing or existing.endswith(b'\n')) and expected.startswith(existing)
            if not valid_prefix and not recover:
                raise EventExportError(f"Event export {path} is incomplete or differs from SQLite; use recover=True")
            if not valid_prefix:
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(mode='wb', dir=path.parent,
                            prefix=f'.{path.name}.', delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(expected)
                        stream.flush()
                        os.fsync(stream.fileno())
                    os.replace(temporary, path)
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            elif existing != expected or not path.exists():
                with path.open('ab') as stream:
                    stream.write(expected[len(existing):])
                    stream.flush()
                    os.fsync(stream.fileno())
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return len(records)
