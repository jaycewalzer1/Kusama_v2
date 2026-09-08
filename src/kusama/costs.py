"""Append-only local inference cost estimates, separate from provider invoices.

No rate or power draw is assumed. A missing component keeps the total unknown.
The policy is snapshotted before each call, and a crash leaves an open event.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time


POLICY_ENV = 'KUSAMA_COST_POLICY'
COMPONENTS = ('provider_api', 'electricity', 'hardware')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be a finite nonnegative number')
    return float(value)


def validate_policy(policy=None):
    """Validate an explicitly supplied tariff; USD is only a reporting currency."""
    result = {'currency': 'USD', 'average_power_watts': None,
              'electricity_per_kwh': None, 'hardware_per_hour': None, 'source': None}
    policy = {} if policy is None else policy
    if not isinstance(policy, dict) or set(policy) - set(result):
        raise ValueError('Cost policy must be an object containing only documented fields')
    result.update(policy)
    if not isinstance(result['currency'], str) or not re.fullmatch('[A-Z]{3}', result['currency']):
        raise ValueError('currency must be a three-letter uppercase currency code')
    for key in ('average_power_watts', 'electricity_per_kwh', 'hardware_per_hour'):
        if result[key] is not None:
            result[key] = _number(result[key], key)
    supplied_rates = any(result[key] is not None for key in
                         ('average_power_watts', 'electricity_per_kwh', 'hardware_per_hour'))
    if supplied_rates and (not isinstance(result['source'], str) or not result['source'].strip()):
        raise ValueError('An explicit source/assumption note is required for cost estimates')
    if result['source'] is not None and not isinstance(result['source'], str):
        raise ValueError('source must be text or null')
    return result


def load_policy():
    path = os.environ.get(POLICY_ENV)
    return validate_policy(json.loads(Path(path).expanduser().read_text()) if path else None)


def estimate_local_cost(runtime_seconds, policy):
    policy = validate_policy(policy)
    seconds = None if runtime_seconds is None else _number(runtime_seconds, 'runtime_seconds')
    components = {
        'provider_api': {'amount': 0.0, 'status': 'actual',
                         'source': 'Local execution: no provider API invocation or invoice.'},
        'electricity': {'amount': None, 'status': 'unknown',
                        'source': 'Requires runtime, assumed average watts and electricity tariff.'},
        'hardware': {'amount': None, 'status': 'unknown',
                     'source': 'Requires runtime and an explicit hardware allocation rate.'},
    }
    if seconds is not None:
        hours = seconds / 3600
        if policy['average_power_watts'] is not None and policy['electricity_per_kwh'] is not None:
            components['electricity'] = {'amount': hours * policy['average_power_watts'] / 1000 * policy['electricity_per_kwh'],
                                         'status': 'estimated', 'source': policy['source']}
        if policy['hardware_per_hour'] is not None:
            components['hardware'] = {'amount': hours * policy['hardware_per_hour'],
                                      'status': 'estimated', 'source': policy['source']}
    known = sum(item['amount'] for item in components.values() if item['amount'] is not None)
    complete = all(item['amount'] is not None for item in components.values())
    return {'currency': policy['currency'], 'components': components, 'known_subtotal': known,
            'total': known if complete else None, 'complete': complete,
            'status': 'estimated' if complete else 'incomplete',
            'scope_note': 'Local compute estimate for the recorded runtime scope only; excludes labor, storage and research.'}


class CostJournal:
    """Independent run/call identifiers support both ledger and pre-ledger workflows."""
    def __init__(self, root: Path, *, read_only=False):
        self.path = Path(root).resolve() / 'costs.sqlite3'
        if read_only:
            self.connection = sqlite3.connect(f'{self.path.as_uri()}?mode=ro', uri=True, timeout=10)
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.executescript('''
                CREATE TABLE IF NOT EXISTS cost_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK(event_type IN ('started','finished')),
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, call_id, event_type)
                );
                CREATE TRIGGER IF NOT EXISTS costs_no_update BEFORE UPDATE ON cost_events
                    BEGIN SELECT RAISE(ABORT, 'cost events are append-only'); END;
                CREATE TRIGGER IF NOT EXISTS costs_no_delete BEFORE DELETE ON cost_events
                    BEGIN SELECT RAISE(ABORT, 'cost events are append-only'); END;
            ''')
        self.connection.row_factory = sqlite3.Row

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _append(self, run_id, call_id, event_type, payload, *, exclusive=False):
        if not all(isinstance(value, str) and value.strip() for value in (run_id, call_id)):
            raise ValueError('run_id and call_id must be nonempty strings')
        encoded = _json(payload)
        self.connection.execute('BEGIN IMMEDIATE')
        try:
            row = self.connection.execute('SELECT * FROM cost_events WHERE run_id=? AND call_id=? AND event_type=?',
                                          (run_id, call_id, event_type)).fetchone()
            if row is not None:
                if exclusive:
                    raise ValueError('Cost call already exists; refusing to execute it again')
                if row['payload_json'] != encoded:
                    raise ValueError('Cost event identity already has different immutable contents')
            else:
                self.connection.execute('INSERT INTO cost_events(run_id,call_id,event_type,payload_json,created_at) VALUES(?,?,?,?,?)',
                    (run_id, call_id, event_type, encoded, datetime.now(timezone.utc).isoformat()))
            self.connection.execute('COMMIT')
        except BaseException:
            self.connection.execute('ROLLBACK')
            raise
        return self.call(run_id, call_id)

    def start(self, run_id, call_id, model, *, policy=None, scope='inference', metadata=None, exclusive=False):
        if not isinstance(model, str) or not model.strip() or not isinstance(scope, str) or not scope.strip():
            raise ValueError('model and scope must be nonempty strings')
        return self._append(run_id, call_id, 'started', {'schema_version': 1, 'execution': 'local',
            'model': model, 'scope': scope, 'policy': load_policy() if policy is None else validate_policy(policy),
            'metadata': metadata or {}}, exclusive=exclusive)

    def finish(self, run_id, call_id, runtime_seconds=None, outcome='succeeded', *, measurements=None, evidence=None):
        if outcome not in ('succeeded', 'failed', 'interrupted', 'uncertain', 'recovered'):
            raise ValueError('Unsupported cost outcome')
        record = self.call(run_id, call_id)
        seconds = None if runtime_seconds is None else _number(runtime_seconds, 'runtime_seconds')
        return self._append(run_id, call_id, 'finished', {'runtime_seconds': seconds, 'outcome': outcome,
            'measurements': measurements or {}, 'evidence': evidence,
            'cost': estimate_local_cost(seconds, record['start']['policy'])})

    def call(self, run_id, call_id):
        rows = list(self.connection.execute('SELECT * FROM cost_events WHERE run_id=? AND call_id=? ORDER BY sequence',
                                           (run_id, call_id)))
        if not rows or rows[0]['event_type'] != 'started':
            raise KeyError(f'Unknown cost call: {run_id}/{call_id}')
        record = {'run_id': run_id, 'call_id': call_id, 'started_at': rows[0]['created_at'],
                  'start': json.loads(rows[0]['payload_json']), 'finish': None}
        if len(rows) > 1:
            record.update(finish=json.loads(rows[1]['payload_json']), finished_at=rows[1]['created_at'])
        return record

    def summary(self, run_id=None, *, run_ids=None):
        query = "SELECT run_id,call_id FROM cost_events WHERE event_type='started'"
        parameters = ()
        if run_id is not None and run_ids is not None:
            raise ValueError('Supply run_id or run_ids, not both')
        if run_id is not None:
            query += ' AND run_id=?'
            parameters = (run_id,)
        elif run_ids is not None:
            parameters = tuple(run_ids)
            query += ' AND run_id IN (' + ','.join('?' for _ in parameters) + ')'
        records = [self.call(row['run_id'], row['call_id']) for row in
                   self.connection.execute(query + ' ORDER BY sequence', parameters)]
        currencies = {}
        for record in records:
            finish = record['finish']
            cost = finish['cost'] if finish else estimate_local_cost(None, record['start']['policy'])
            currency = currencies.setdefault(cost['currency'], {'known_subtotal': 0.0, 'total': 0.0,
                'complete': True, 'incomplete_calls': 0, 'estimated_calls': 0,
                'components': {key: {'known_subtotal': 0.0, 'unknown_calls': 0} for key in COMPONENTS}})
            currency['known_subtotal'] += cost['known_subtotal']
            currency['complete'] = currency['complete'] and cost['complete']
            currency['incomplete_calls'] += int(not cost['complete'])
            currency['estimated_calls'] += int(any(c['status'] == 'estimated' for c in cost['components'].values()))
            for key, component in cost['components'].items():
                currency['components'][key]['known_subtotal'] += component['amount'] or 0.0
                currency['components'][key]['unknown_calls'] += int(component['amount'] is None)
            currency['total'] = currency['known_subtotal'] if currency['complete'] else None
        return {'schema_version': 1, 'run_id': run_id, 'calls': len(records),
                'open_calls': sum(record['finish'] is None for record in records),
                'runtime_seconds_measured': sum((record['finish'] or {}).get('runtime_seconds') or 0 for record in records),
                'currencies': currencies, 'records': records,
                'coverage': 'recorded_calls_only' if records else 'no_cost_records',
                'note': 'No records is unknown coverage, not zero cost. Currency totals include recorded scopes only.'}


@contextmanager
def tracked_local_call(root, run_id, call_id, model, *, policy=None, scope='inference', metadata=None):
    """Wrap one local operation; duplicate identity never authorizes re-execution.

    Set the yielded dict's measurements, runtime_seconds or outcome when needed.
    A process crash leaves the start record open and its total unknown.
    """
    with CostJournal(root) as journal:
        try:
            journal.call(run_id, call_id)
        except KeyError:
            pass
        else:
            raise ValueError('Cost call already exists; refusing to execute it again')
        journal.start(run_id, call_id, model, policy=policy, scope=scope, metadata=metadata, exclusive=True)
        started = time.monotonic()
        result = {}
        try:
            yield result
        except BaseException as exc:
            result['outcome'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            result['evidence'] = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            journal.finish(run_id, call_id, result.get('runtime_seconds', time.monotonic() - started),
                result.get('outcome', 'succeeded'), measurements=result.get('measurements'), evidence=result.get('evidence'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts'))
    parser.add_argument('--run-id')
    parser.add_argument('--reconcile-call', help='Close an open call as uncertain, without inventing runtime or running a model')
    parser.add_argument('--evidence', help='Required explanation for reconciliation')
    args = parser.parse_args(argv)
    if args.reconcile_call and (not args.run_id or not args.evidence):
        parser.error('Reconciliation requires --run-id and --evidence')
    with CostJournal(args.root) as journal:
        if args.reconcile_call:
            journal.finish(args.run_id, args.reconcile_call, outcome='uncertain', evidence=args.evidence)
        print(json.dumps(journal.summary(args.run_id), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
