"""Read-only integrity and accounting snapshot; never an artistic quality score."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from statistics import median

from PIL import Image

from .artifacts import sha256, write_json
from .costs import CostJournal


# These are sampled usage counters. Capacity, available/recommended memory,
# allocator reservations, and native peak counters have different meanings.
SAMPLED_USAGE_FIELDS = ('process_rss_bytes', 'mps_current_allocated_bytes',
                        'mps_driver_allocated_bytes', 'cuda_allocated_bytes')


def _valid_measurement(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0)


def _model_activity(records):
    """Aggregate the existing cost journal without exposing call contents."""
    grouped = {}
    for record in records:
        key = (record['start']['model'], record['start']['scope'])
        grouped.setdefault(key, []).append(record)
    groups = []
    for (model, scope), rows in sorted(grouped.items()):
        finished = [row['finish'] for row in rows if row['finish'] is not None]
        runtimes = [row['runtime_seconds'] for row in finished
                    if _valid_measurement(row.get('runtime_seconds'))]
        missing_runtime = sum(row.get('runtime_seconds') is None for row in finished)
        sampled = []
        for finish in finished:
            measurements = finish.get('measurements')
            maxima = measurements.get('sampled_max_bytes') if isinstance(measurements, dict) else None
            sampled.append(maxima if isinstance(maxima, dict) else {})
        memory = {}
        for field in SAMPLED_USAGE_FIELDS:
            values = [row[field] for row in sampled if _valid_measurement(row.get(field))]
            memory[field] = {'max_bytes': max(values) if values else None,
                             'measured_records': len(values),
                             'finished_records_without_valid_measurement': len(finished) - len(values)}
        groups.append({'model': model, 'scope': scope,
            'started_records': len(rows), 'completed_records': len(finished),
            'open_records': len(rows) - len(finished),
            'finish_outcomes': dict(sorted(Counter(row['outcome'] for row in finished).items())),
            'runtime_seconds': {'measured_records': len(runtimes),
                'missing_finished_records': missing_runtime,
                'invalid_finished_records': len(finished) - missing_runtime - len(runtimes),
                'unmeasured_open_records': len(rows) - len(finished),
                'min': min(runtimes) if runtimes else None,
                'median': median(runtimes) if runtimes else None,
                'max': max(runtimes) if runtimes else None,
                'sum': sum(runtimes) if runtimes else None},
            'sampled_memory': {'finished_records_with_any_valid_usage_measurement': sum(
                any(_valid_measurement(row.get(field)) for field in SAMPLED_USAGE_FIELDS) for row in sampled),
                'unmeasured_open_records': len(rows) - len(finished), 'fields': memory}})
    return {'source': 'CostJournal.summary records', 'groups': groups,
        'completed_definition': 'A recorded finish event, including failed, interrupted, uncertain or recovered outcomes; not necessarily a successful result.',
        'limitations': [
            'Recorded calls only; absent measurements and open calls are unknown, not zero runtime or memory usage.',
            'Runtime uses each finish.runtime_seconds in its recorded scope. Sums include measured failed calls, exclude missing times, and are not total elapsed session time.',
            'Exact model and scope groups may mix render profiles, dimensions, tokens, devices and loading state. Scope labels remain distinct; these aggregates are not comparable latency benchmarks.',
            'Memory figures are maxima of recorded sampled usage counters, not native peaks or guaranteed peak bounds; brief peaks between samples may be missed.',
            'Available, total and recommended RAM, allocator reservations and native peak counters are excluded. RSS and device allocation counters overlap and must not be added together.',
            'No raw cost records, prompts, image data, metadata or evidence text are returned.'
        ]}


@contextmanager
def _open(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('BEGIN')
    try:
        yield db
    finally:
        db.close()


def audit(root):
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError('An existing artifact root is required')
    issues, databases, jobs, images = [], [], [], {}
    for path in sorted(root.rglob('*.sqlite3')):
        if 'exports' in path.relative_to(root).parts:
            continue
        with _open(path) as db:
            integrity = [r[0] for r in db.execute('PRAGMA integrity_check')]
            foreign_keys = [list(r) for r in db.execute('PRAGMA foreign_key_check')]
            databases.append({'path': str(path.relative_to(root)), 'integrity': integrity,
                              'foreign_key_issues': foreign_keys})
            if integrity != ['ok'] or foreign_keys:
                issues.append({'database': str(path), 'error': 'SQLite integrity failure'})
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {'jobs', 'artifacts', 'attempts', 'reservations'} <= tables:
                continue
            for job in db.execute('SELECT * FROM jobs ORDER BY created_at,job_id'):
                identifier, directory = job['job_id'], Path(job['directory']).resolve()
                encoded = job['spec_json'].encode()
                if hashlib.sha256(encoded).hexdigest() != job['spec_fingerprint']:
                    issues.append({'job_id': identifier, 'error': 'Immutable specification hash mismatch'})
                records = list(db.execute('SELECT * FROM artifacts WHERE job_id=?', (identifier,)))
                verified = 0
                for record in records:
                    asset = (directory / record['path']).resolve()
                    try:
                        if not asset.is_relative_to(directory) or not directory.is_relative_to(root):
                            raise ValueError('Registered path escapes its job or artifact root')
                        if sha256(asset) != record['sha256']:
                            raise ValueError('Registered artifact hash mismatch')
                        verified += 1
                        if asset.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
                            with Image.open(asset) as image:
                                image.load()
                                images.setdefault(record['sha256'], {'sha256': record['sha256'],
                                    'width': image.width, 'height': image.height, 'mode': image.mode,
                                    'registered_paths': []})['registered_paths'].append(str(asset))
                    except (ValueError, OSError) as exc:
                        issues.append({'job_id': identifier, 'path': record['path'], 'error': str(exc)})
                attempts = Counter(r[0] for r in db.execute('SELECT status FROM attempts WHERE job_id=?', (identifier,)))
                direct_calls = db.execute('SELECT count(*) FROM reservations r JOIN attempts a ON a.attempt_id=r.attempt_id WHERE a.job_id=?', (identifier,)).fetchone()[0]
                charged_calls = db.execute('SELECT count(*) FROM reservation_charges WHERE job_id=?', (identifier,)).fetchone()[0]
                if charged_calls > job['max_calls']:
                    issues.append({'job_id': identifier, 'error': 'Ledger calls exceed frozen budget'})
                jobs.append({'id': identifier, 'database': str(path.relative_to(root)), 'status': job['status'],
                    'direct_calls': direct_calls, 'charged_calls_including_descendants': charged_calls,
                    'max_calls': job['max_calls'], 'attempt_states': dict(attempts),
                    'registered_artifacts': len(records), 'verified_artifacts': verified,
                    'spec_sha256': job['spec_fingerprint']})
    review = {'genuine_submitted_pair_votes': 0, 'submitted_pool_votes': 0, 'explicit_advancements': 0}
    review_path = root / 'reviews.sqlite3'
    if review_path.exists():
        with _open(review_path) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for key, table in [('genuine_submitted_pair_votes', 'review_votes'), ('submitted_pool_votes', 'review_pool_votes'), ('explicit_advancements', 'review_advancements')]:
                if table in tables:
                    review[key] = db.execute('SELECT count(*) FROM ' + table).fetchone()[0]
    resource = None
    if (root / 'resources.sqlite3').exists():
        with _open(root / 'resources.sqlite3') as db:
            session = db.execute('SELECT * FROM session WHERE id=1').fetchone()
            cfg = json.loads(session['config'])
            calls = list(db.execute('SELECT * FROM calls'))
            renderer = sum(r['kind'] == 'renderer' for r in calls)
            tokens = sum(r['actual_tokens'] if r['actual_tokens'] is not None else r['token_reservation'] for r in calls)
            resource = {'started_at': session['started_at'], 'deadline': session['deadline'],
                'renderer_attempts': renderer, 'renderer_remaining': cfg['renderer_attempt_cap'] - renderer,
                'reasoning_tokens_charged_or_reserved': tokens, 'reasoning_tokens_remaining': cfg['reasoning_token_cap'] - tokens,
                'states': dict(Counter(r['status'] for r in calls)),
                'calls_by_kind': {kind: {'total': sum(r['kind'] == kind for r in calls),
                    'states': dict(sorted(Counter(r['status'] for r in calls if r['kind'] == kind).items()))}
                    for kind in sorted({'renderer', 'reasoning'} | {r['kind'] for r in calls})},
                'unsettled_keys': [r['key'] for r in calls if r['status'] in ('running', 'uncertain')]}
            if renderer > cfg['renderer_attempt_cap'] or tokens > cfg['reasoning_token_cap']:
                issues.append({'error': 'Session resource cap exceeded'})
    cost = None
    if (root / 'costs.sqlite3').exists():
        with CostJournal(root, read_only=True) as journal:
            summary = journal.summary()
            cost = {key: value for key, value in summary.items() if key != 'records'}
            cost['model_activity'] = _model_activity(summary['records'])
    return {'schema_version': 1, 'audited_at': datetime.now(timezone.utc).isoformat(), 'root': str(root),
        'integrity_status': 'passed' if not issues else 'failed', 'issues': issues, 'databases': databases,
        'jobs': jobs, 'registered_unique_image_bytes': len(images), 'image_records': list(images.values()),
        'reviews': review, 'resources': resource, 'costs': cost,
        'scope': 'Consistent read transactions per database, not an atomic cross-database backup. Registered images include references, parents, masks, raw outputs and final outputs; this count is not renderer attempts.',
        'claim_boundary': 'Integrity and recorded execution only. Fixture tests, machine observations and queue creation do not establish artistic effectiveness or human preference. No sealed text or outcomes are included in this report.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/overnight'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = audit(args.root)
    output = args.output or args.root / 'integrity_audit.json'
    write_json(output, report)
    print(json.dumps({'report': str(output), 'integrity_status': report['integrity_status'],
                      'issues': report['issues'], 'jobs': len(report['jobs']),
                      'reviews': report['reviews'], 'resources': report['resources']}, indent=2))
    return int(bool(report['issues']))


if __name__ == '__main__':
    raise SystemExit(main())
