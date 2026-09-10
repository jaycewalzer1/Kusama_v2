"""Read-only developer inventory and documented workflow entry points."""
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3

DOCS = Path(__file__).resolve().parents[2] / 'docs'

# Commands are copied, never interpreted or launched by the web process.
WORKFLOWS = (
    ('research', 'Independent research', 'Plan inquiries, inspect source encounters, and render selected hypotheses.', 'workflows/independent_research.md'),
    ('research-sources', 'Source catalog', 'Build metadata catalogs and acquire bounded source material.', 'workflows/independent_research.md'),
    ('ingest', 'Reference ingestion', 'Import collection records with provenance and rights.', 'workflows/source_connected_studio.md'),
    ('index', 'Reference retrieval', 'Build whole-image, detail, lexical and relationship indexes.', 'workflows/source_connected_studio.md'),
    ('inspire', 'Connection proposals', 'Develop inquiry-led relationships between references.', 'workflows/source_connected_studio.md'),
    ('run', 'Studio generation', 'Generate and develop source-connected artistic episodes.', 'workflows/source_connected_studio.md'),
    ('resume', 'Resume an episode', 'Continue using the original frozen inputs and invocation ledger.', 'operations/recovery.md'),
    ('develop', 'Autonomous development', 'Plan, preflight and execute bounded parent-to-child development.', 'workflows/bounded_development.md'),
    ('finish', 'Finishing', 'Prepare, execute and observe the reserved finishing stage.', 'workflows/continuation.md'),
    ('memory', 'Experiment memory', 'Build and query decision-time experiment memory.', 'data/experiment_memory.md'),
    ('dataset', 'Dataset export', 'Create grouped, provenance-bound learning snapshots.', 'data/experiment_memory.md'),
    ('train', 'Learning eligibility', 'Inspect eligibility and explicit training entry points.', 'models/model_and_training_status.md'),
    ('edit-benchmark', 'Edit diagnostics', 'Measure edit fulfillment, geometry and protected pixels.', 'evaluation/edit_benchmark.md'),
    ('verify', 'Evidence audit', 'Inspect integrity, lineage and recorded execution.', 'operations/implementation_status.md'),
    ('doctor', 'Environment check', 'Inspect devices, cached checkpoints and configured resources.', 'operations/implementation_status.md'),
    ('export', 'Evidence export', 'Export preserved artifacts and consistent database snapshots.', 'operations/recovery.md'),
)
EXTRA_WORKFLOWS = (
    ('kusama.continuation', 'Selected-parent continuation', 'Prepare a controlled revision from an attributed saved parent.', 'workflows/continuation.md'),
    ('kusama.creative', 'Reference creative loop', 'Generate directions from an operator-curated reference packet.', 'workflows/reference_loop.md'),
    ('kusama.directed', 'Visual director', 'Propose and execute a frozen Qwen-directed edit.', 'workflows/frozen_director.md'),
    ('kusama.controlled', 'Controlled editor', 'Whole, masked, crop and geometric editing controls.', 'workflows/controlled_editing.md'),
    ('kusama.studio_cli', 'Persistent studio', 'Manage inquiries, episodes, actions and branch selection.', 'workflows/studio.md'),
    ('kusama.experiment', 'Initial image experiment', 'Generate a parent and inspect saved edits.', 'experiments/verification.md'),
    ('kusama.benchmark', 'Verification benchmark', 'Run technical editing diagnostics and inspect failures.', 'evaluation/verification_benchmark.md'),
    ('kusama.content_probe', 'Content diagnostic', 'Run explicitly budgeted content capability probes.', 'evaluation/content_probe.md'),
)


def tools_catalog():
    from .cli import COMMAND_MODULES
    result = []
    for name, title, description, guide in WORKFLOWS:
        if name not in COMMAND_MODULES and name not in ('doctor', 'export'):
            continue
        result.append(dict(id=name, title=title, description=description,
                           command=f'uv run --locked python -m kusama.cli {name} --help', guide='/api/guides/' + guide))
    for module, title, description, guide in EXTRA_WORKFLOWS:
        result.append(dict(id=module, title=title, description=description,
                           command=f'uv run --locked python -m {module} --help', guide='/api/guides/' + guide))
    return result


def artifact_directories(root):
    """Skip hidden directories and links at every level, including authority files."""
    root = Path(root).resolve()
    for parent, children, files in os.walk(root, followlinks=False):
        directory = Path(parent)
        children[:] = [name for name in children if not name.startswith('.') and not (directory / name).is_symlink()]
        yield directory, {name for name in files if not name.startswith('.') and not (directory / name).is_symlink()}


def system_inventory(root):
    root = Path(root).resolve()
    sessions, memories = [], []
    for directory, files in artifact_directories(root):
        if 'resources.sqlite3' in files:
            item = {'directory': str(directory.relative_to(root))}
            try:
                with closing(sqlite3.connect((directory / 'resources.sqlite3').as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
                    db.row_factory = sqlite3.Row
                    row = db.execute('SELECT config,started_at,deadline FROM session WHERE id=1').fetchone()
                    if row is None:
                        raise ValueError('Resource session is not initialized')
                    config = json.loads(row['config'])
                    calls = [dict(call) for call in db.execute('SELECT kind,status,actual_tokens,token_reservation FROM calls')]
                    charged = sum(call['actual_tokens'] if call['actual_tokens'] is not None else call['token_reservation'] for call in calls)
                    item.update(started_at=row['started_at'], deadline=row['deadline'],
                        expired=datetime.fromisoformat(row['deadline']) <= datetime.now(timezone.utc),
                        renderer_attempts=sum(call['kind'] == 'renderer' for call in calls),
                        renderer_cap=config.get('renderer_attempt_cap'), reasoning_tokens_charged_or_reserved=charged,
                        reasoning_token_cap=config.get('reasoning_token_cap'),
                        unsettled_calls=sum(call['status'] in ('running', 'uncertain') for call in calls),
                        device=config.get('device'), default_render_profile=config.get('default_render_profile'),
                        render_profiles=config.get('render_profiles'))
            except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
                item['error'] = str(exc)
            sessions.append(item)
        if 'memory.sqlite3' in files:
            item = {'directory': str(directory.relative_to(root))}
            try:
                with closing(sqlite3.connect((directory / 'memory.sqlite3').as_uri() + '?mode=ro', uri=True, timeout=2)) as db:
                    item['events'] = db.execute('SELECT count(*) FROM memory_events').fetchone()[0]
                    item['builds'] = db.execute('SELECT count(*) FROM memory_builds').fetchone()[0]
                    item['queries'] = db.execute('SELECT count(*) FROM memory_queries').fetchone()[0]
            except sqlite3.Error as exc:
                item['error'] = str(exc)
            memories.append(item)
    return {'artifact_root': str(root), 'resource_sessions': sessions, 'experiment_memory': memories,
            'models': {'Qwen': 'frozen', 'FLUX': 'frozen'}, 'inference_started_by_inspection': False}
