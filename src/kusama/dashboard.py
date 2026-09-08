"""Local development runs and explicit, append-only human comparisons."""
from __future__ import annotations

import argparse
from dataclasses import fields
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import quote
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, create_model
from fastapi.responses import FileResponse, HTMLResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware
from .studio import Studio, read_projects
from .reviews import Advancement, PoolChoice, ReviewConflict, ReviewError, ReviewNotFound, ReviewStore, Vote
from .artifacts import sha256
from .config import Config
from .continuation import ContinuationRequest, ContinuationStore, DevelopmentDecision
from .continuation_dispatch import DispatchRequest, DispatchStore
from .ledger import BudgetExceeded, JobLocked


STATIC = Path(__file__).with_name('dashboard_static')
FILE_TYPES = {'.png': 'image', '.jpg': 'image', '.jpeg': 'image', '.json': 'json',
              '.jsonl': 'log', '.log': 'log', '.txt': 'text', '.md': 'text', '.csv': 'text'}


RendererConfiguration = create_model('RendererConfiguration', __config__=ConfigDict(extra='forbid'),
                                    **{field.name: (field.type, field.default) for field in fields(Config)})


class ContinuationCreate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,40}$')
    request: ContinuationRequest
    renderer: RendererConfiguration


class ContinuationObservation(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    artifact_id: str = Field(min_length=1, max_length=100)
    note: str = Field(min_length=1, max_length=8000)
    author: str = Field(min_length=1, max_length=200)
    origin: Literal['operator', 'model_observation', 'development_example']


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
        directories = set(self.root.iterdir())
        nested = self.root / 'runs'
        if nested.is_dir() and not nested.is_symlink():
            directories.update(nested.iterdir())
        directories.update(Path(path) for path in jobs if Path(path).is_relative_to(self.root))
        for directory in directories:
            if (not directory.is_dir() or directory.name.startswith('.') or
                    any(p.is_symlink() for p in [directory, *directory.parents] if p.is_relative_to(self.root))):
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
                    data = payload.get('manifest') or payload.get('report') or payload.get('state') or data
                    data = dict(data, status=job['status'], model_calls=job['calls'])
                    if payload.get('report'):
                        kind = 'benchmark'
                except (ValueError, TypeError, AttributeError):
                    pass
            if data.get('kind') == 'reference_creative_experiment':
                kind = 'reference_creative_experiment'
            elif data.get('kind') in ('connection_inspiration', 'connection_generation'):
                kind = data['kind']
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
            entries.append({'id': hashlib.sha256(str(directory.relative_to(self.root)).encode()).hexdigest()[:20],
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
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and (request.url.path.startswith(('/api/reviews/', '/api/continuations')) or re.fullmatch(r'/api/projects/[^/]+/select', request.url.path)):
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

    @app.get('/api/reviews/collections/{collection_id}/pairs/{pair_id}/images/{position}')
    def assigned_image(collection_id: str, pair_id: str, position: str, reviewer: str):
        if position not in ('A', 'B'):
            raise HTTPException(404)
        assignment = reviews.comparison(collection_id, pair_id, reviewer)
        return FileResponse(reviews.candidate_path(collection_id, assignment['candidates'][position]['id']))

    @app.post('/api/reviews/collections/{collection_id}/pairs/{pair_id}/votes')
    def review_vote(collection_id: str, pair_id: str, vote: Vote):
        return reviews.submit_vote(collection_id, pair_id, vote)

    @app.post('/api/reviews/collections/{collection_id}/advance')
    def review_advance(collection_id: str, advancement: Advancement):
        return reviews.advance(collection_id, advancement)

    @app.get('/api/reviews/collections/{collection_id}/pool')
    def pool_assignment(collection_id: str, reviewer: str):
        return reviews.pool_assignment(collection_id, reviewer)

    @app.get('/api/reviews/collections/{collection_id}/pool/history')
    def pool_history(collection_id: str):
        return {'selections': reviews.pool_history(collection_id)}

    @app.get('/api/reviews/collections/{collection_id}/pool/images/{position}')
    def pool_image(collection_id: str, position: str, reviewer: str):
        assignment = reviews.pool_assignment(collection_id, reviewer)
        candidate = assignment['candidates'].get(position)
        if not candidate:
            raise HTTPException(404)
        return FileResponse(reviews.candidate_path(collection_id, candidate['id']))

    @app.post('/api/reviews/collections/{collection_id}/pool/votes')
    def pool_vote(collection_id: str, choice: PoolChoice):
        return reviews.submit_pool_choice(collection_id, choice)

    @app.get('/api/runs')
    def listing():
        return {'runs': [runs.summary(row) for row in runs.entries()]}

    @app.get('/api/projects')
    def projects():
        return {'projects': read_projects(runs.root)}

    @app.get('/api/continuations')
    def continuations():
        from .continuation import ContinuationStore, read_continuations
        store = ContinuationStore(runs.root)
        return {'continuations': [store.get(row['id']) for row in read_continuations(runs.root)]}

    @app.get('/api/contracts')
    def contracts():
        return {model.__name__: model.model_json_schema() for model in (
            ContinuationCreate, ContinuationRequest, ContinuationObservation, DevelopmentDecision, DispatchRequest, Vote, PoolChoice)}

    @app.post('/api/continuations', status_code=201)
    def continuation_prepare(proposal: ContinuationCreate):
        try:
            config = Config(**proposal.renderer.model_dump())
            if config.steps < 1 or not 1 <= config.max_sequence_length <= 512 or config.max_retries != 0:
                raise ValueError('Continuation requires positive steps, 1–512 prompt tokens and zero automatic retries')
            if not re.fullmatch('[0-9a-f]{40}', config.model_revision):
                raise ValueError('Renderer revision must be an immutable lowercase commit SHA')
            return ContinuationStore(runs.root).prepare(proposal.id, proposal.request, config)
        except ReviewError:
            raise
        except (ValueError, OSError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/continuations/{continuation_id}/execute', status_code=202)
    def continuation_execute(continuation_id: str, request: DispatchRequest):
        try:
            return DispatchStore(runs.root).submit(continuation_id, request)
        except (JobLocked, BudgetExceeded) as exc:
            raise HTTPException(409, str(exc)) from exc
        except ReviewError:
            raise
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/continuations/{continuation_id}/status')
    def continuation_status(continuation_id: str):
        return {'continuation': ContinuationStore(runs.root).get(continuation_id),
                'dispatch': DispatchStore(runs.root).get(continuation_id)}

    @app.post('/api/continuations/{continuation_id}/observations')
    def continuation_observe(continuation_id: str, observation: ContinuationObservation):
        try:
            return ContinuationStore(runs.root).observe(continuation_id, **observation.model_dump())
        except ReviewError:
            raise
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/continuations/{continuation_id}')
    def continuation_detail(continuation_id: str):
        from .continuation import ContinuationStore
        return ContinuationStore(runs.root).get(continuation_id)

    @app.post('/api/continuations/{continuation_id}/decisions')
    def continuation_decide(continuation_id: str, decision: DevelopmentDecision):
        try:
            return ContinuationStore(runs.root).decide(continuation_id, decision)
        except JobLocked as exc:
            raise HTTPException(409, str(exc)) from exc
        except ReviewError:
            raise
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/continuations/{continuation_id}/gallery')
    def continuation_gallery(continuation_id: str):
        from .continuation import ContinuationStore
        return HTMLResponse(ContinuationStore(runs.root).gallery_html(continuation_id, web=True))

    def corpus_rows(table, identifier=None):
        database = runs.root / 'corpus' / 'corpus.sqlite3'
        if not database.is_file() or database.is_symlink() or database.parent.is_symlink():
            return []
        with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
            sql, args = f'SELECT payload FROM {table}', ()
            if identifier is not None:
                sql += ' WHERE id=?'
                args = (identifier,)
            return [json.loads(row[0]) for row in db.execute(sql, args)]

    @app.get('/api/sources')
    def sources():
        return {'sources': [{k: row.get(k) for k in ('id', 'title', 'maker', 'provider', 'date', 'medium', 'source_family', 'url')}
                            for row in corpus_rows('references_v1')]}

    @app.get('/api/sources/{reference_id}')
    def source_detail(reference_id: str):
        records = corpus_rows('references_v1', reference_id)
        if not records:
            raise HTTPException(404, 'Unknown source')
        return {'reference': records[0],
                'regions': [r for r in corpus_rows('regions_v1') if r['reference_id'] == reference_id],
                'claims': [r for r in corpus_rows('claims_v1') if r.get('reference_id') == reference_id]}

    def registered_image(root, asset):
        path = root / asset['path']
        if (not path.resolve().is_relative_to(root) or not path.is_file()
                or any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(root))):
            raise HTTPException(404, 'Image is missing or outside its registered root')
        if sha256(path) != asset['sha256']:
            raise HTTPException(409, 'Registered image bytes changed')
        return FileResponse(path)

    @app.get('/api/sources/{reference_id}/image')
    def source_image(reference_id: str):
        ref = source_detail(reference_id)['reference']
        if ref['rights']['image_display']['status'] not in ('allowed', 'conditional'):
            raise HTTPException(403, 'Source display rights do not permit this use')
        return registered_image(runs.root / 'corpus', ref['view'])

    @app.get('/api/source-regions/{region_id}')
    def source_region(region_id: str):
        records = corpus_rows('regions_v1', region_id)
        if not records:
            raise HTTPException(404, 'Unknown source region')
        return records[0]

    @app.get('/api/source-regions/{region_id}/image')
    def source_region_image(region_id: str):
        region = source_region(region_id)
        if any(region['rights'][operation]['status'] not in ('allowed', 'conditional') for operation in ('image_display', 'derivatives')):
            raise HTTPException(403, 'Region display rights do not permit this use')
        return registered_image(runs.root / 'corpus', region['crop'])

    def inspiration_directory(run_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
            raise HTTPException(404)
        directory = runs.root / 'runs' / run_id
        if directory.is_symlink() or not directory.resolve().is_relative_to(runs.root):
            raise HTTPException(404)
        return directory

    @app.get('/api/inspirations')
    def inspirations():
        records = []
        parent = runs.root / 'runs'
        for path in sorted(parent.glob('*/manifest.json')) if parent.is_dir() else []:
            if path.is_symlink() or path.parent.is_symlink():
                continue
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if record.get('kind') in ('connection_inspiration', 'creative_episode'):
                records.append(record | {'directory_id': path.parent.name})
        return {'inspirations': records}

    @app.get('/api/inspirations/{run_id}')
    def inspiration_detail(run_id: str):
        directory = inspiration_directory(run_id)
        path = directory / 'manifest.json'
        if not path.is_file() or path.is_symlink():
            raise HTTPException(404)
        record = json.loads(path.read_text())
        for key in ('shelf', 'sources'):
            filename = record.get(key)
            if isinstance(filename, str) and filename in ('shelf.json', 'sources.json'):
                source = directory / filename
                if source.is_file() and not source.is_symlink():
                    record[key + '_data'] = json.loads(source.read_text())
        return record

    @app.get('/api/inspirations/{run_id}/images/{image_id}')
    def inspiration_image(run_id: str, image_id: str):
        record = inspiration_detail(run_id)
        asset = next((row for row in record.get('images', []) if row.get('id') == image_id), None)
        if asset is None:
            raise HTTPException(404, 'Unknown realized image')
        return registered_image(inspiration_directory(run_id), asset)

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
