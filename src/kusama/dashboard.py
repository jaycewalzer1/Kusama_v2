"""Local development panel for runs, research and controlled continuations."""
from __future__ import annotations

import argparse
from dataclasses import fields
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import os
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
from .reviews import ReviewConflict, ReviewError, ReviewNotFound
from .artifacts import sha256
from .config import Config
from .continuation import ContinuationRequest, ContinuationStore, DevelopmentDecision
from .continuation_dispatch import DispatchRequest, DispatchStore
from .ledger import BudgetExceeded, JobLocked, LedgerError
from .devtools import DOCS, artifact_directories, tools_catalog, system_inventory
from .panel_runs import PanelRuns, StartRun, options as start_options
from .run_inspection import inspect_run


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


class WorkflowPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    workflow: Literal['research', 'development', 'research_render']
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,40}$')
    request: dict


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
        jobs = {}
        for directory, files in artifact_directories(self.root):
            if 'jobs.sqlite3' not in files:
                continue
            try:
                with closing(sqlite3.connect((directory / 'jobs.sqlite3').as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
                    db.row_factory = sqlite3.Row
                    rows = db.execute("""SELECT j.directory,j.status,j.payload_json,j.max_calls,
                        (SELECT count(*) FROM reservation_charges r WHERE r.job_id=j.job_id) AS calls
                        FROM jobs j ORDER BY j.rowid""").fetchall()
                    for row in rows:
                        path = Path(row['directory']).resolve()
                        if path.is_relative_to(self.root) and path.is_relative_to(directory):
                            jobs[str(path)] = dict(row)
            except sqlite3.Error:
                continue
        return jobs

    def entries(self):
        jobs = self.ledger_jobs()
        entries = []
        if not self.root.is_dir():
            return entries
        directories = {directory for directory, files in artifact_directories(self.root)
                       if files & {'manifest.json', 'benchmark.json', 'plan.json', 'report.json', 'render_plan.json'}}
        directories.update(Path(path) for path in jobs)
        for directory in directories:
            if (not directory.is_dir() or directory.name.startswith('.') or
                    any(p.is_symlink() for p in [directory, *directory.parents] if p.is_relative_to(self.root))):
                continue
            job = jobs.get(str(directory.resolve()))
            manifest = directory / 'benchmark.json'
            kind = 'benchmark'
            if not manifest.is_file():
                manifest, kind = directory / 'manifest.json', 'edit'
            if not manifest.is_file():
                for filename in ('report.json', 'render_plan.json', 'plan.json'):
                    candidate = directory / filename
                    if candidate.is_file() and not candidate.is_symlink():
                        manifest = candidate
                        kind = 'development'
                        break
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
            if data.get('kind') == 'artistic_studio_run':
                try:
                    data = PanelRuns(directory.parent.parent).get(directory.name)
                    resource = directory / 'resources.sqlite3'
                    if resource.is_file() and not resource.is_symlink():
                        with closing(sqlite3.connect(resource.as_uri() + '?mode=ro', uri=True)) as db:
                            data['model_calls'] = db.execute('SELECT count(*) FROM calls').fetchone()[0]
                except (OSError, ValueError, sqlite3.Error) as exc:
                    data['error'] = str(exc)
            if data.get('kind'):
                kind = data['kind']
            elif str(data.get('version', '')).startswith('bounded-development'):
                kind = 'bounded_development'
            if manifest.name == 'plan.json':
                data.setdefault('status', 'prepared')
            if data.get('kind') == 'reference_creative_experiment':
                kind = 'reference_creative_experiment'
            elif data.get('kind') in ('connection_inspiration', 'connection_generation'):
                kind = data['kind']
            if 'calls' in data and isinstance(data.get('calls'), list):
                kind = 'first_edit'
            budget = data.get('budget') if isinstance(data.get('budget'), dict) else {}
            request_data = data.get('request') if isinstance(data.get('request'), dict) else {}
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
            model_calls = data.get('model_calls', data.get('reasoning_calls', budget.get('actual_image_attempts')))
            if creative:
                if type(data.get('total_model_calls')) is int and data['total_model_calls'] >= 0:
                    model_calls = data['total_model_calls']
                if type(data.get('authorized_maximum_model_calls')) is int and data['authorized_maximum_model_calls'] >= 0:
                    max_calls = data['authorized_maximum_model_calls']
            planned_images = None
            if creative and isinstance(data.get('direction_count'), int) and isinstance(data.get('realizations'), int):
                planned_images = data['direction_count'] * data['realizations']
            entries.append({'id': hashlib.sha256(str(directory.relative_to(self.root)).encode()).hexdigest()[:20],
                'name': directory.name, 'directory': str(directory.relative_to(self.root)),
                'title': data.get('title') or request_data.get('title'),
                'brief': data.get('brief') or request_data.get('problem'),
                'image_count': len(data.get('images') or []) if creative else None, 'planned_images': planned_images,
                'kind': kind, 'status': data.get('status', 'unreadable' if error else 'saved'),
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

    def files(self, row, recursive=False):
        result = []
        images = {}
        if row['kind'] == 'reference_creative_experiment':
            for record in row['_data'].get('images') or []:
                if isinstance(record, dict) and isinstance(record.get('path'), str):
                    images[record['path']] = record
        root = row['_directory']
        paths = (folder / name for folder, names in artifact_directories(root) for name in sorted(names)) if recursive else sorted(root.iterdir())
        for path in paths:
            if path.is_file() and not path.is_symlink() and not path.name.startswith('.') and path.suffix in FILE_TYPES:
                relative = str(path.relative_to(root))
                stat = path.stat()
                result.append({'name': relative, 'bytes': stat.st_size,
                               'modified_at': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
                               'url': f"/api/runs/{row['id']}/files/{quote(relative, safe='/')}",
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


class ResearchRuns:
    """Read-only research evidence, addressed by contained directory identities.

    Sessions can live in the research directory or inside a development run. No
    request opens the writing corpus or initializes a research/resource ledger.
    """

    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def directories(self):
        if not self.root.is_dir():
            return
        for parent, children, files in os.walk(self.root, followlinks=False):
            directory = Path(parent)
            children[:] = [name for name in children
                           if not name.startswith('.') and not (directory / name).is_symlink()]
            plan = directory / 'plan.json'
            if 'plan.json' not in files or plan.is_symlink():
                continue
            try:
                data = json.loads(plan.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get('kind') == 'autonomous_research':
                yield directory, data

    def identifier(self, directory):
        return hashlib.sha256(str(directory.relative_to(self.root)).encode()).hexdigest()[:20]

    def get(self, identifier):
        if re.fullmatch('[a-f0-9]{20}', identifier):
            for directory, plan in self.directories():
                if self.identifier(directory) == identifier:
                    return directory, plan
        raise HTTPException(404, 'Unknown research session')

    @staticmethod
    def report(directory, *, include_transcripts=True):
        from .research import read_report
        try:
            if any((directory / name).is_symlink() for name in
                   ('plan.json', 'plan_identity.json', 'jobs.sqlite3')):
                raise ValueError('Research evidence cannot use symbolic links')
            report = read_report(directory)
            if include_transcripts:
                from .run_inspection import research_transcripts
                report['transcripts'] = research_transcripts(directory, report)
            return report
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error, LedgerError) as exc:
            raise HTTPException(409, 'Research evidence cannot be verified: ' + str(exc)) from exc

    def entries(self):
        result = []
        for directory, plan in self.directories():
            request = plan.get('request') if isinstance(plan.get('request'), dict) else {}
            row = {'directory_id': self.identifier(directory),
                   'directory': str(directory.relative_to(self.root)),
                   'id': request.get('id'), 'problem': request.get('problem'),
                   'authorship': request.get('authorship'), 'status': 'unreadable'}
            try:
                report = self.report(directory, include_transcripts=False)
                row.update({key: report.get(key) for key in
                            ('id', 'problem', 'authorship', 'status', 'fixture', 'reasoning_calls')})
                row['steps'] = len(report.get('steps') or [])
                row['hypotheses'] = len(report.get('hypotheses') or [])
            except HTTPException as exc:
                row['error'] = exc.detail
            result.append(row)
        return sorted(result, key=lambda item: item['directory'])


def workspace_choices(root):
    choices = [{'directory': '.', 'label': 'All artifacts', 'url': '/'}]
    for directory, files in artifact_directories(root):
        if (directory == root or 'exports' in directory.relative_to(root).parts
                or any(part.startswith('studio-') for part in directory.relative_to(root).parts)
                or not files.intersection({'jobs.sqlite3', 'resources.sqlite3', 'corpus.sqlite3'})):
            continue
        if directory.name == 'corpus':
            directory = directory.parent
        if directory == root:
            continue
        relative = str(directory.relative_to(root))
        if any(row['directory'] == relative for row in choices):
            continue
        identifier = hashlib.sha256(relative.encode()).hexdigest()[:20]
        choices.append({'directory': relative, 'label': relative, 'url': f'/workspaces/{identifier}/'})
    return choices


def create_app(artifacts: Path = Path('artifacts'), *, _workspace_base=None, _prefix=''):
    runs = Runs(artifacts)
    workspace_base = Path(_workspace_base or artifacts).resolve()
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])

    @app.middleware('http')
    async def local_only(request: Request, call_next):
        origin = request.headers.get('origin')
        if request.headers.get('sec-fetch-site') == 'cross-site' or (origin and origin != f"http://{request.headers.get('host')}"):
            return Response('Local access only', status_code=403)
        if request.method not in ('GET', 'HEAD', 'OPTIONS') and (request.url.path.removeprefix(_prefix).startswith(('/api/continuations', '/api/plans', '/api/start-run')) or re.fullmatch(r'/api/(?:projects/[^/]+/select|runs/[^/]+/stop)', request.url.path.removeprefix(_prefix))):
            if request.headers.get('x-kusama-review') != '1' or request.headers.get('content-type', '').split(';')[0] != 'application/json':
                return Response('Local JSON development requests require X-Kusama-Review: 1', status_code=403)
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
        if name not in ('dashboard.js', 'dashboard_cycle.js', 'dashboard_feed.js', 'dashboard.css'):
            raise HTTPException(404)
        return FileResponse(STATIC / name)

    @app.exception_handler(ReviewError)
    async def review_error(request, exc):
        from fastapi.responses import JSONResponse
        status = 404 if isinstance(exc, ReviewNotFound) else 409 if isinstance(exc, ReviewConflict) else 422
        return JSONResponse({'detail': str(exc)}, status_code=status)

    @app.get('/api/runs')
    def listing():
        return {'runs': [runs.summary(row) for row in runs.entries()]}

    @app.get('/api/workspaces')
    def workspaces():
        return {'selected': _prefix or '/', 'workspaces': workspace_choices(workspace_base)}

    @app.get('/api/start-options')
    def run_options():
        return start_options()

    @app.post('/api/start-run', status_code=202)
    def start_run(request: StartRun):
        try:
            if request.study_development and not {'development_steps', 'development_research_steps', 'total_token_limit'} <= request.model_fields_set:
                raise ValueError('Refresh the panel to load the current token and development limits before starting. API requests must explicitly supply development_steps, development_research_steps and total_token_limit.')
            result = PanelRuns(runs.root, workspace_base).start(request)
            relative = 'runs/' + result['run_id']
            return result | {'id': hashlib.sha256(relative.encode()).hexdigest()[:20]}
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/runs/{run_id}/stop')
    def stop_run(run_id: str):
        try:
            row = runs.get(run_id)
            if row['kind'] != 'artistic_studio_run':
                raise ValueError('Only question-led studio runs support this stop control')
            directory = row['_directory']
            return PanelRuns(directory.parent.parent, workspace_base).stop(directory.name)
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/tools')
    def workflow_tools():
        return {'tools': tools_catalog()}

    @app.post('/api/plans', status_code=201)
    def workflow_plan(proposal: WorkflowPlan):
        from .autonomous import plan_loop
        from .research import plan_research
        from .research_render import plan_render
        planners = {'research': plan_research, 'development': plan_loop, 'research_render': plan_render}
        directory = runs.root / proposal.workflow / proposal.id
        if (not directory.resolve().is_relative_to(runs.root)
                or any(path.is_symlink() for path in (directory, directory.parent))):
            raise HTTPException(422, 'Plan directory must stay inside the workspace')
        if directory.exists():
            raise HTTPException(409, 'Plan ID already exists; use a new ID to preserve frozen inputs')
        try:
            result = planners[proposal.workflow](directory, proposal.request)
            return {'directory': str(directory), 'plan': result, 'status': 'prepared'}
        except (ValueError, OSError, TypeError, LedgerError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/system')
    def system():
        return system_inventory(runs.root)

    @app.get('/api/guides/{name:path}')
    def guide(name: str):
        allowed = {tool['guide'].removeprefix('/api/guides/') for tool in tools_catalog()}
        path = DOCS / name
        if name not in allowed or not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(DOCS):
            raise HTTPException(404, 'Unknown workflow guide')
        return FileResponse(path, media_type='text/plain')

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
        from .research import ResearchAction, ResearchHypothesis, ResearchRequest
        from .autonomous import LoopRequest
        from .research_render import ResearchRenderRequest
        from .research_sources import ResearchSourceConfig
        from .study_cycle import DevelopmentCycle, CycleChoice
        return {model.__name__: model.model_json_schema() for model in (
            ContinuationCreate, ContinuationRequest, ContinuationObservation, DevelopmentDecision, DispatchRequest, WorkflowPlan, StartRun,
            ResearchRequest, ResearchAction, ResearchHypothesis, ResearchSourceConfig, ResearchRenderRequest, LoopRequest, DevelopmentCycle, CycleChoice)}

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
        html = ContinuationStore(runs.root).gallery_html(continuation_id, web=True)
        html = html.replace('<h1>', '<body class="gallery-page"><h1>', 1).replace('</html>', '</body></html>')
        if _prefix:
            html = html.replace('"/api/', '"' + _prefix + '/api/').replace("'/api/", "'" + _prefix + '/api/')
        return HTMLResponse(html)

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

    research_runs = ResearchRuns(runs.root)

    @app.get('/api/research')
    def research_listing():
        return {'research': research_runs.entries()}

    @app.get('/api/research/{research_id}')
    def research_detail(research_id: str):
        directory, plan = research_runs.get(research_id)
        report = research_runs.report(directory)
        images = []
        for asset in report.get('evidence_images') or []:
            image_id = hashlib.sha256(json.dumps(asset, sort_keys=True).encode()).hexdigest()[:20]
            images.append(asset | {'id': image_id,
                                  'url': f'/api/research/{research_id}/images/{image_id}'})
        return report | {'directory_id': research_id, 'directory': str(directory.relative_to(runs.root)),
                         'frozen_plan': plan, 'evidence_images': images}

    @app.get('/api/research/{research_id}/images/{image_id}')
    def research_image(research_id: str, image_id: str):
        directory, _ = research_runs.get(research_id)
        report = research_detail(research_id)
        asset = next((row for row in report['evidence_images'] if row['id'] == image_id), None)
        if asset is None:
            raise HTTPException(404, 'Unknown research evidence image')
        rights = asset.get('rights') or {}
        permission = rights.get('image_display', rights)
        if permission.get('status') not in ('allowed', 'conditional'):
            raise HTTPException(403, 'Research source display rights do not permit this use')
        if not isinstance(asset.get('path'), str) or not isinstance(asset.get('sha256'), str):
            raise HTTPException(409, 'Research evidence image registration is incomplete')
        if Path(asset['path']).suffix.lower() not in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff'):
            raise HTTPException(403, 'Research evidence is not a supported raster image')
        return registered_image(directory, asset)

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

    @app.get('/api/runs/{run_id}/inspect')
    def inspect(run_id: str):
        return inspect_run(runs.get(run_id), runs)

    @app.get('/api/runs/{run_id}/files/{name:path}')
    def artifact(run_id: str, name: str):
        row = runs.get(run_id)
        file_record = next((item for item in runs.files(row, recursive=True) if item['name'] == name), None)
        if file_record is None:
            raise HTTPException(404, 'Unknown artifact')
        path = row['_directory'] / name
        if path.is_symlink() or not path.resolve().is_relative_to(row['_directory'].resolve()):
            raise HTTPException(404)
        if file_record.get('sha256') and sha256(path) != file_record['sha256']:
            raise HTTPException(409, 'Artifact changed since generation')
        media = None if FILE_TYPES[path.suffix] == 'image' else 'application/json' if path.suffix == '.json' else 'text/plain'
        return FileResponse(path, media_type=media)

    if _workspace_base is None:
        for choice in workspace_choices(workspace_base):
            if choice['url'] == '/':
                continue
            app.mount(choice['url'].rstrip('/'), create_app(workspace_base / choice['directory'],
                _workspace_base=workspace_base, _prefix=choice['url'].rstrip('/')))
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description='Local development panel for Kusama')
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
