"""Durable suites share one transactional budget with their child edit jobs."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from .artifacts import sha256, write_json
from .backend import FluxBackend
from .controlled import run
from .execution import _read_source, _snapshot, canonical_config, restore_config
from .experiment import code_provenance
from .ledger import Ledger


def run_suite_job(path=None, config=None, runner=run, *, resume_dir=None, job_id=None, backend_factory=FluxBackend):
    from . import benchmark
    if resume_dir is not None:
        directory = Path(resume_dir).resolve()
        descriptor = json.loads((directory / 'job.json').read_text())
        database = Path(descriptor['database'])
        job_id = descriptor['job_id']
    else:
        if path is None or config is None:
            raise ValueError('A new suite requires a suite path and configuration')
        path = Path(path).resolve()
        config = canonical_config(config)
        suite = benchmark.load_suite(path)
        # Pin every source before the first child can spend budget.
        sources = [_read_source(path.parent / case['plan']) for case in suite['cases']]
        spec = {'schema_version': 1, 'kind': 'benchmark', 'suite': suite, 'source_suite': str(path),
                'inputs': [{'plan': data[0][0], 'hashes': data[1]} for data in sources], 'config': config.to_dict()}
        job_id = job_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_benchmark_' + uuid.uuid4().hex[:8]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', job_id):
            raise ValueError('job-id must contain 1–128 letters, digits, underscores or hyphens')
        directory = config.output_dir / job_id
        database = config.output_dir / 'jobs.sqlite3'
    with Ledger(database) as ledger:
        if resume_dir is not None:
            job = ledger.get_job(job_id)
            if Path(job['directory']) != directory or job['spec']['kind'] != 'benchmark':
                raise ValueError('Resume target is not this benchmark job')
            saved_config = restore_config(job['spec']['config'])
            if config is not None and canonical_config(config).to_dict() != saved_config.to_dict():
                raise ValueError('Resume uses frozen configuration')
            config = saved_config
            if path is not None:
                raise ValueError('Resume uses the saved suite; omit --suite')
        else:
            job = ledger.create_job(job_id, spec, suite['max_calls'], directory)
        with ledger.lock(job_id):
            directory.mkdir(parents=True, exist_ok=True)
            if not (directory / 'job.json').exists():
                write_json(directory / 'job.json', {'schema_version': 1, 'job_id': job_id, 'database': str(ledger.path)})
            ledger.verify_artifacts(job_id)
            job = ledger.get_job(job_id)
            suite = job['spec']['suite']
            if not job['payload'].get('snapshot_complete'):
                source_path = Path(job['spec']['source_suite'])
                if benchmark.load_suite(source_path) != suite:
                    raise ValueError('Suite changed before input snapshots completed')
                for index, case in enumerate(suite['cases']):
                    source = source_path.parent / case['plan']
                    loaded, hashes = _read_source(source)
                    if {'plan': loaded[0], 'hashes': hashes} != job['spec']['inputs'][index]:
                        raise ValueError('Suite input changed before snapshot completed')
                    target = directory / 'inputs' / str(index)
                    target.mkdir(parents=True, exist_ok=True)
                    _snapshot(target, source, loaded)
                if not (directory / 'suite.json').exists():
                    write_json(directory / 'suite.json', suite)
                for target in [directory / 'job.json', directory / 'suite.json', *sorted((directory / 'inputs').rglob('*'))]:
                    if target.is_file():
                        ledger.register_artifact(job_id, target.relative_to(directory))
                ledger.update_job(job_id, payload={'snapshot_complete': True})
            report = ledger.get_job(job_id)['payload'].get('report') or {
                'schema_version': 2, 'job_id': job_id, 'name': suite['name'], 'max_calls': suite['max_calls'],
                'model_calls': 0, 'status': 'running', 'cases': [], 'suite_sha256': sha256(directory / 'suite.json'),
                'config': config.to_dict(), 'code': code_provenance()}

            def persist():
                report['model_calls'] = ledger.call_count(job_id)
                ledger.update_job(job_id, status=report['status'], payload={'report': report})
                write_json(directory / 'benchmark.json', report)
                ledger.export_events(directory / 'events.jsonl', job_id=job_id, recover=True)

            started = time.monotonic()
            original_error = None
            active_case = None
            try:
                report['status'] = 'running'
                persist()
                for index, item in enumerate(suite['cases']):
                    if index < len(report['cases']):
                        case = report['cases'][index]
                    else:
                        case = {'id': item['id'], 'status': 'pending'}
                        report['cases'].append(case)
                    active_case = case
                    child_id = 'case_' + hashlib.sha256(f'{job_id}:{index}'.encode()).hexdigest()[:32]
                    child_dir = config.output_dir / child_id
                    plan_path = directory / 'inputs' / str(index) / 'execution_plan.json'
                    loaded, hashes = _read_source(plan_path)
                    child_spec = {'schema_version': 1, 'kind': 'controlled_edit', 'plan': loaded[0],
                        'config': config.to_dict(), 'source_plan': str(plan_path), 'source_hashes': hashes}
                    child = ledger.create_job(child_id, child_spec, loaded[0]['max_calls'], child_dir, parent_id=job_id)
                    # Register identity and location even if the runner is killed before returning.
                    case.update(job_id=child_id, run=str(child_dir), plan_sha256=sha256(plan_path))
                    if child['status'] in ('needs_review', 'rejected', 'budget_exhausted', 'uncertain'):
                        # The controlled entry point still checks retained artifact hashes.
                        pass
                    elif not child['model_calls'] and ledger.call_count(job_id) >= suite['max_calls']:
                        case['status'] = 'skipped_budget'
                        persist()
                        continue
                    case['status'] = 'running'
                    persist()
                    try:
                        kwargs = {'ledger': ledger, 'parent_job_id': job_id, 'job_id': child_id}
                        if runner is run:
                            kwargs['backend_factory'] = backend_factory
                        if (child_dir / 'job.json').exists():
                            output = runner(None, None, call_guard=None, resume_dir=child_dir, **kwargs)
                        else:
                            output = runner(plan_path, config, call_guard=None, **kwargs)
                        if Path(output).resolve() != child_dir:
                            raise ValueError('Runner returned a directory different from the registered child')
                        case['status'] = ledger.get_job(child_id)['status']
                    except Exception as exc:
                        case.update(status='failed', error=f'{type(exc).__name__}: {exc}')
                    persist()
                report['status'] = 'completed' if all(c['status'] in ('needs_review', 'rejected') for c in report['cases']) else 'completed_with_failures'
            except BaseException as exc:
                original_error = exc
                report.update(status='interrupted', error=f'{type(exc).__name__}: {exc}')
                if active_case is not None and active_case['status'] == 'running':
                    active_case['status'] = 'interrupted'
                raise
            finally:
                report['runtime_seconds'] = report.get('runtime_seconds', 0) + time.monotonic() - started
                if report.get('reporting_errors'):
                    report.setdefault('previous_reporting_errors', []).append(report.pop('reporting_errors'))
                report['reporting_status'] = 'running'
                persist()
                reporting_interrupt = None
                for stage, render in [('summary', benchmark.summarize), ('contact_sheet', benchmark.contact_sheet)]:
                    try:
                        render(directory, report)
                    except BaseException as exc:
                        benchmark.reporting_error(report, stage, exc)
                        if not isinstance(exc, Exception) and reporting_interrupt is None:
                            reporting_interrupt = exc
                    finally:
                        persist()
                report['reporting_status'] = 'completed_with_errors' if report.get('reporting_errors') else 'completed'
                persist()
                if reporting_interrupt is not None and original_error is None:
                    raise reporting_interrupt
            return directory
