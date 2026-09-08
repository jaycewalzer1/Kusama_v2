"""Local development runs and explicit, append-only human comparisons."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from fastapi.responses import FileResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .studio import Studio, read_projects
from .reviews import Advancement, ReviewConflict, ReviewError, ReviewNotFound, ReviewStore, Vote
from .artifacts import sha256


STATIC = Path(__file__).with_name('dashboard_static')
FILE_TYPES = {'.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.json': 'json',
              '.jsonl': 'log', '.log': 'log', '.txt': 'text', '.md': 'text', '.csv': 'text'}


class Runs:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def ledger_jobs(self):
        database = self.root / 'jobs.sqlite3'
        if not database.is_file():
            return {}
        try:
            # Never instantiate the writing ledger from a dashboard request.
            with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute('''SELECT j.directory,j.status,j.payload_json,j.max_calls,
                    (SELECT count(*) FROM reservation_charges r WHERE r.job_id=j.job_id) AS calls
                    FROM jobs j''').fetchall()
                return {row['directory']: dict(row) for row in rows}
        except sqlite3.Error:
            return {}

    def entries(self):
        jobs = self.ledger_jobs()
        entries = []
        if not self.root.is_dir():
            return entries
        for directory in self.root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or directory.name.startswith('.'):
                continue
            job = jobs.get(str(directory.resolve()))
            manifest = directory / 'benchmark.json'
            kind = 'benchmark'
            if not manifest.is_file():
                manifest, kind = directory / 'manifest.json', 'edit'
            if not manifest.is_file() and job is None:
                continue
            error = None
            data = {}
            if manifest.is_file() and not manifest.is_symlink():
                try:
                    data = json.loads(manifest.read_text())
                    if not isinstance(data, dict):
                        raise ValueError('Expected a JSON object')
                except (OSError, ValueError) as exc:
                    data, error = {}, f'Cannot read {manifest.name}: {exc}'
            if job is not None:
                try:
                    payload = json.loads(job['payload_json'])
                    data = payload.get('manifest') or payload.get('report') or data
                    data = dict(data, status=job['status'], model_calls=job['calls'])
                    if payload.get('report'):
                        kind = 'benchmark'
                except (ValueError, TypeError, AttributeError):
                    pass
            if data.get('kind') == 'reference_creative_experiment':
                kind = 'reference_creative_experiment'
            if 'calls' in data and isinstance(data.get('calls'), list):
                kind = 'first_edit'
            budget = data.get('budget') or {}
            attempts = data.get('attempts') or []
            last = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
            verification = last.get('verification', {}).get('status') if isinstance(last.get('verification'), dict) else None
            if kind == 'benchmark':
                verification = (data.get('summary') or {}).get('verification_counts')
            elif kind == 'reference_creative_experiment':
                report_path = directory / 'verification.json'
                if report_path.is_file() and not report_path.is_symlink():
                    try:
                        report = json.loads(report_path.read_text())
                        if (isinstance(report, dict) and data.get('run_id')
                                and report.get('run_id') == data['run_id']
                                and report.get('kind') == 'technical_integrity_verification'
                                and report.get('status') in ('passed', 'failed')):
                            verification = {'status': report['status'], 'scope': 'technical_integrity',
                                            'verified_at': report.get('verified_at'), 'report_file': 'verification.json'}
                    except (OSError, ValueError):
                        pass
            max_calls = job['max_calls'] if job else data.get('max_calls', data.get('planned_calls', budget.get('maximum_image_attempts')))
            if max_calls is None:
                try:
                    max_calls = json.loads((directory / 'plan.json').read_text()).get('max_calls')
                except (OSError, ValueError, AttributeError):
                    pass
            failures = data.get('failures') or []
            last_failure = failures[-1] if failures else None
            run_time = data.get('runtime_seconds', data.get('runtime_seconds_so_far'))
            runtime_scope = 'run' if run_time is not None else None
            if run_time is None and isinstance(data.get('costs'), dict):
                run_time = data['costs'].get('runtime_seconds_measured')
                runtime_scope = 'recorded_operations' if run_time is not None else None
            creative = kind == 'reference_creative_experiment'
            model_calls = data.get('model_calls', budget.get('actual_image_attempts'))
            if creative:
                if type(data.get('total_model_calls')) is int and data['total_model_calls'] >= 0:
                    model_calls = data['total_model_calls']
                if type(data.get('authorized_maximum_model_calls')) is int and data['authorized_maximum_model_calls'] >= 0:
                    max_calls = data['authorized_maximum_model_calls']
            planned_images = None
            if creative and isinstance(data.get('direction_count'), int) and isinstance(data.get('realizations'), int):
                planned_images = data['direction_count'] * data['realizations']
            entries.append({'id': hashlib.sha256(directory.name.encode()).hexdigest()[:20],
                'name': directory.name, 'title': data.get('title') if creative else None,
                'brief': data.get('brief') if creative else None,
                'image_count': len(data.get('images') or []) if creative else None, 'planned_images': planned_images,
                'kind': kind, 'status': data.get('status', 'unreadable'),
                'observer_status': data.get('observer_status') if creative and isinstance(data.get('observer_status'), dict) else None,
                'planned_observations': data.get('direction_count') if creative and isinstance(data.get('observer_status'), dict) else None,
                'model_calls': model_calls,
                'max_calls': max_calls, 'runtime_seconds': run_time, 'runtime_scope': runtime_scope,
                'verification': verification, 'error': error or data.get('error') or last.get('error') or last_failure,
                '_directory': directory, '_data': data, '_modified': manifest.stat().st_mtime if manifest.exists() else directory.stat().st_mtime})
        return sorted(entries, key=lambda row: row['_modified'], reverse=True)

    @staticmethod
    def summary(row):
        return {key: value for key, value in row.items() if not key.startswith('_')}

    def get(self, run_id):
        for row in self.entries():
            if row['id'] == run_id:
                return row
        raise HTTPException(404, 'Unknown run')

    def files(self, row):
        result = []
        images = {}
        if row['kind'] == 'reference_creative_experiment':
            for record in row['_data'].get('images') or []:
                if isinstance(record, dict) and isinstance(record.get('path'), str):
                    images[record['path']] = record
        for path in sorted(row['_directory'].iterdir()):
            if path.is_file() and not path.is_symlink() and not path.name.startswith('.') and path.suffix in FILE_TYPES:
                result.append({'name': path.name, 'url': f"/api/runs/{row['id']}/files/{quote(path.name, safe='')}",
                               'type': FILE_TYPES[path.suffix], 'preview': path.name in images,
                               'label': images.get(path.name, {}).get('label'),
                               'sha256': images.get(path.name, {}).get('sha256')})
        return result


class StudioSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    episode_id: str = Field(min_length=1, max_length=200)
    artifact_id: str = Field(min_length=1, max_length=200)
    author: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=8000)


def create_app(artifacts: Path = Path('artifacts')):
    runs = Runs(artifacts)
    reviews = ReviewStore(runs.root)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.headers.get('sec-fetch-site') == 'cross-site' or (origin and origin != f"http://{request.headers.get('host')}"):
            return Response('Local access only', status_code=403)
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and (request.url.path.startswith('/api/reviews/') or re.fullmatch(r'/api/projects/[^/]+/select', request.url.path)):
            if request.headers.get('x-kusama-review') != '1' or request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return Response('Local JSON review requests require X-Kusama-Review: 1', status_code=403)
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        return response

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/static/{name}')
    def static(name: str):
        if name not in ('dashboard.js', 'dashboard.css'):
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.exception_handler(ReviewError)
    async def review_error(request, exc):
        from fastapi.responses import JSONResponse
        status = 404 if isinstance(exc, ReviewNotFound) else 409 if isinstance(exc, ReviewConflict) else 422
        return JSONResponse({'detail': str(exc)}, status_code=status)

    @app.get('/api/reviews/collections')
    def review_collections(reviewer: str | None = None):
        return {'collections': reviews.collections(reviewer)}

    @app.get('/api/reviews/collections/{collection_id}')
    def review_collection(collection_id: str):
        return reviews.collection(collection_id)

    @app.get('/api/reviews/collections/{collection_id}/history')
    def review_history(collection_id: str):
        return reviews.history(collection_id)

    @app.get('/api/reviews/collections/{collection_id}/candidates/{candidate_id}')
    def review_image(collection_id: str, candidate_id: str):
        return FileResponse(reviews.candidate_path(collection_id, candidate_id))

    @app.get('/api/reviews/collections/{collection_id}/pairs/{pair_id}')
    def review_comparison(collection_id: str, pair_id: str, reviewer: str):
        return reviews.comparison(collection_id, pair_id, reviewer)

    @app.post('/api/reviews/collections/{collection_id}/pairs/{pair_id}/votes')
    def review_vote(collection_id: str, pair_id: str, vote: Vote):
        return reviews.submit_vote(collection_id, pair_id, vote)

    @app.post('/api/reviews/collections/{collection_id}/advance')
    def review_advance(collection_id: str, advancement: Advancement):
        return reviews.advance(collection_id, advancement)

    @app.get('/api/runs')
    def listing():
        return {'runs': [runs.summary(row) for row in runs.entries()]}

    @app.get('/api/projects')
    def projects():
        return {'projects': read_projects(runs.root)}

    @app.get('/api/projects/{project_id}')
    def project_detail(project_id: str):
        rows = read_projects(runs.root, project_id)
        if not rows:
            raise HTTPException(404, 'Unknown project')
        return rows[0]

    @app.post('/api/projects/{project_id}/select')
    def project_select(project_id: str, selection: StudioSelection):
        try:
            with Studio(runs.root) as studio:
                episode = studio.select(project_id, selection.episode_id, selection.artifact_id,
                                        selection.reason, selection.author)
            return {'episode': episode}
        except (ValueError, KeyError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/projects/{project_id}/artifacts/{artifact_id}')
    def project_artifact(project_id: str, artifact_id: str):
        record = project_detail(project_id)['artifacts'].get(artifact_id)
        if not record:
            raise HTTPException(404, 'Unknown artifact')
        path = Path(record['path'])
        if not path.resolve().is_relative_to(runs.root) or not path.is_file() or path.is_symlink():
            raise HTTPException(404)
        if sha256(path) != record['sha256']:
            raise HTTPException(409, 'Artifact changed')
        return FileResponse(path, media_type='image/png')

    @app.get('/api/runs/{run_id}')
    def detail(run_id: str):
        row = runs.get(run_id)
        accounting = row['_data'].get('cost_accounting')
        monetary = {'coverage': 'no_cost_records'}
        if isinstance(accounting, dict) and isinstance(accounting.get('root'), str) and accounting['root'].strip():
            cost_root = Path(accounting['root']).resolve()
            run_ids = accounting.get('run_ids')
            valid_run_ids = (isinstance(run_ids, list) and bool(run_ids)
                             and all(isinstance(value, str) and value.strip() for value in run_ids)
                             and len(set(run_ids)) == len(run_ids))
            primary_id = accounting.get('run_id')
            valid_primary = isinstance(primary_id, str) and bool(primary_id.strip())
            # Never call an unscoped summary: absent or malformed IDs must not
            # expose totals belonging to unrelated experiments in the same journal.
            if cost_root.is_relative_to(runs.root) and (valid_run_ids or valid_primary):
                try:
                    from .costs import CostJournal
                    with CostJournal(cost_root, read_only=True) as costs:
                        monetary = costs.summary(run_ids=run_ids) if valid_run_ids else costs.summary(primary_id)
                except (OSError, sqlite3.Error):
                    pass
        summary = runs.summary(row)
        if row['kind'] == 'reference_creative_experiment' and monetary.get('coverage') == 'recorded_calls_only':
            summary.update(runtime_seconds=monetary['runtime_seconds_measured'], runtime_scope='recorded_operations')
        return summary | {'artifacts': runs.files(row), 'data': row['_data'], 'monetary_cost': monetary}

    @app.get('/api/runs/{run_id}/files/{name}')
    def artifact(run_id: str, name: str):
        row = runs.get(run_id)
        file_record = next((item for item in runs.files(row) if item['name'] == name), None)
        if file_record is None:
            raise HTTPException(404, 'Unknown artifact')
        path = row['_directory'] / name
        if path.is_symlink() or path.resolve().parent != row['_directory'].resolve():
            raise HTTPException(404)
        if file_record.get('sha256') and sha256(path) != file_record['sha256']:
            raise HTTPException(409, 'Artifact changed since generation')
        media = None if FILE_TYPES[path.suffix] == 'image' else 'application/json' if path.suffix == '.json' else 'text/plain'
        return FileResponse(path, media_type=media)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local dashboard for Kusama runs and human review')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1 through 65535')
    import uvicorn
    print(f'Kusama dev dashboard: http://127.0.0.1:{args.port}/', flush=True)
    uvicorn.run(create_app(args.artifacts), host='127.0.0.1', port=args.port, access_log=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
