"""Explicit saved-parent revisions through the existing studio and charge ledger.

Preparing freezes inputs but never calls a model. Executing drains only that
frozen batch. Human advancements are imported by submission ID; operator/model
choices remain separately attributed. Review registration never creates labels.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import html
import json
from pathlib import Path
import sqlite3
from typing import Literal

from PIL import Image, ImageChops, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import sha256, write_json
from .backend import FluxBackend
from .config import Config
from .controlled import load_plan
from .execution import canonical_config, restore_config
from .reviews import ReviewConflict, ReviewNotFound, ReviewStore
from .studio import Inquiry, Reference, Studio, encoded, fingerprint, now


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)


class ParentSelection(StrictModel):
    origin: Literal['human_advancement', 'operator', 'model_selection']
    collection_id: str | None = None
    advancement_submission_id: str | None = None
    path: Path | None = None
    image_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')
    candidate_id: str | None = None
    author: str | None = Field(default=None, min_length=1, max_length=200)
    reason: str | None = Field(default=None, min_length=1, max_length=8000)

    @model_validator(mode='after')
    def explicit_selection(self):
        if self.origin == 'human_advancement':
            if not self.collection_id or not self.advancement_submission_id:
                raise ValueError('Human selection requires an existing collection and advancement submission ID')
            if any(getattr(self, key) is not None for key in ('path', 'image_sha256', 'candidate_id', 'author', 'reason')):
                raise ValueError('Human advancement fields are resolved from the immutable review record')
        elif (not self.path or not self.image_sha256 or not self.author or not self.reason
              or self.collection_id or self.advancement_submission_id):
            raise ValueError('Operator/model selection needs a path, exact hash, author and reason, without a human advancement ID')
        return self


class Revision(StrictModel):
    author: str = Field(min_length=1, max_length=200)
    origin: Literal['operator', 'model_proposal', 'development_example']
    instruction: str = Field(min_length=1, max_length=8000)
    intended_change: str = Field(min_length=1, max_length=4000)
    target: str = Field(min_length=1, max_length=1000)
    method: Literal['whole', 'masked']
    region: tuple[float, float, float, float] | None = None
    target_region_id: str | None = Field(default=None, max_length=200)
    seeds: tuple[int, int] = (23, 41)
    preservation_constraints: list[str] = Field(min_length=1, max_length=32)
    visual_checks: list[str] = Field(min_length=1, max_length=32)
    source_ids: list[str] = Field(default_factory=list, max_length=32)
    connection_id: str | None = Field(default=None, max_length=200)
    relationship: str = Field(default='', max_length=8000)
    relationship_kind: Literal['proposed_analogy', 'documented_relationship'] = 'proposed_analogy'
    whole_work_plan: str = Field(min_length=1, max_length=8000)

    @model_validator(mode='after')
    def supported_operation(self):
        if len(set(self.seeds)) != 2 or any(type(s) is not int or not 0 <= s < 2**63 for s in self.seeds):
            raise ValueError('Two distinct nonnegative 63-bit seeds are required')
        if any(not item.strip() for item in self.preservation_constraints + self.visual_checks):
            raise ValueError('Preservation and visual checks cannot contain blank text')
        if self.method == 'masked':
            if self.region is None:
                raise ValueError('Masked edits require explicit normalized region bounds')
            l, t, r, b = self.region
            if not 0 <= l < r <= 1 or not 0 <= t < b <= 1:
                raise ValueError('Region bounds must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1')
        elif self.region is not None or self.target_region_id is not None:
            raise ValueError('Whole-image edits do not support claimed region isolation')
        if self.source_ids and not self.relationship:
            raise ValueError('Source references require an attributed relationship')
        return self


class ContinuationRequest(StrictModel):
    title: str = Field(min_length=1, max_length=1000)
    brief: str = Field(min_length=1, max_length=8000)
    selection: ParentSelection
    inquiry: Inquiry
    references: list[Reference] = Field(default_factory=list, max_length=32)
    connection_context: list[dict] = Field(default_factory=list, max_length=32)
    revision: Revision

    @model_validator(mode='after')
    def known_sources(self):
        ids = [r.id for r in self.references]
        if len(ids) != len(set(ids)) or set(self.revision.source_ids) - set(ids):
            raise ValueError('Reference IDs must be unique and every action source must resolve')
        return self


def _identifier(value):
    import re
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', value):
        raise ValueError('Continuation IDs require 1–40 letters, digits, underscores or hyphens')
    return value


def _ensure_json(path, data):
    normalized = json.loads(encoded(data))
    if path.exists():
        if json.loads(path.read_text()) != normalized:
            raise ValueError('Frozen continuation input changed: ' + str(path))
    else:
        write_json(path, normalized)


def _atomic_text(path, value):
    import os
    from .artifacts import sync_directory
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def read_continuations(root: Path, continuation_id=None):
    """Read the authoritative registry without creating a database."""
    root = Path(root).resolve()
    database = root / 'jobs.sqlite3'
    if not database.is_file():
        return []
    with closing(sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='continuation_specs'").fetchone():
            return []
        sql, params = 'SELECT specification,created_at FROM continuation_specs', ()
        if continuation_id is not None:
            sql += ' WHERE id=?'
            params = (continuation_id,)
        return [json.loads(row[0]) | {'created_at': row[1]} for row in db.execute(sql, params)]


class ContinuationStore:
    def __init__(self, root=Path('artifacts')):
        self.root = Path(root).resolve()

    def _selection(self, request):
        selected = request.selection.model_dump(mode='json', exclude_none=True)
        reviews = ReviewStore(self.root)
        if request.selection.origin == 'human_advancement':
            records = reviews.history(selected['collection_id'])['advancements']
            record = next((r for r in records if r['submission_id'] == selected['advancement_submission_id']), None)
            if record is None:
                raise ReviewNotFound('No explicit human advancement with this submission ID')
            path = reviews.candidate_path(record['collection_id'], record['candidate_id'])
            selected.update(path=str(path), image_sha256=record['image_sha256'],
                            candidate_id=record['candidate_id'], author=record['reviewer'],
                            reason=record['reason'], advancement_record=record)
        else:
            path = reviews._image(selected['path'])
            selected['path'] = str(path)
        if sha256(path) != selected['image_sha256']:
            raise ReviewConflict('Selected parent does not match the frozen image hash')
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.format != 'PNG':
                raise ValueError('Selected parent must already be RGB PNG; preserve original normalization lineage')
            if any(n < 64 or n % 16 for n in image.size):
                raise ValueError('Parent dimensions must be at least 64 and divisible by 16')
            selected['size'] = list(image.size)
        return selected

    def prepare(self, continuation_id, request: ContinuationRequest | dict, config: Config, *, resource_root=None):
        """Freeze an explicit selection and two seed plans. No model inference."""
        continuation_id = _identifier(continuation_id)
        request = ContinuationRequest.model_validate(request)
        config = canonical_config(config)
        if config.output_dir != self.root:
            raise ValueError('Configuration output_dir must match the continuation artifact root')
        selected = self._selection(request)
        if (config.width, config.height) != tuple(selected['size']):
            raise ValueError('Declared renderer width/height must match the saved parent')
        spec = {'schema_version': 1, 'kind': 'selected_parent_continuation', 'id': continuation_id,
                'project_id': 'cont_' + continuation_id, 'episode_id': 'revision',
                'request': request.model_dump(mode='json'), 'selection': selected,
                'config': config.to_dict(), 'max_calls': 2,
                'resource_root': str(Path(resource_root or self.root).resolve()),
                'invocation_keys': {str(seed): f'continuation:{continuation_id}:seed:{seed}'
                                    for seed in request.revision.seeds}}
        spec = json.loads(encoded(spec))
        with Studio(self.root) as studio, studio.ledger.lock('continuation_' + continuation_id):
            studio.db.executescript('''
                CREATE TABLE IF NOT EXISTS continuation_specs (
                    id TEXT PRIMARY KEY, specification TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TRIGGER IF NOT EXISTS continuation_specs_no_update BEFORE UPDATE ON continuation_specs
                    BEGIN SELECT RAISE(ABORT, 'continuation inputs are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS continuation_specs_no_delete BEFORE DELETE ON continuation_specs
                    BEGIN SELECT RAISE(ABORT, 'continuation inputs are immutable'); END;
            ''')
            with studio.ledger._transaction():
                prior = studio.db.execute('SELECT specification FROM continuation_specs WHERE id=?', (continuation_id,)).fetchone()
                if prior and json.loads(prior[0]) != spec:
                    raise ValueError('Continuation ID already has different frozen inputs')
                if prior is None:
                    studio.db.execute('INSERT INTO continuation_specs VALUES (?,?,?)', (continuation_id, encoded(spec), now()))
            self._materialize(studio, spec)
        return self.get(continuation_id)

    def _materialize(self, studio, spec):
        request = ContinuationRequest.model_validate(spec['request'])
        revision = request.revision
        project_id, episode_id = spec['project_id'], spec['episode_id']
        directory = self.root / 'continuations' / spec['id']
        directory.mkdir(parents=True, exist_ok=True)
        _ensure_json(directory / 'inputs.json', spec)
        if studio.db.execute('SELECT 1 FROM studio_projects WHERE id=?', (project_id,)).fetchone():
            project = studio.get(project_id)
            studio._artifact(project, 'root')
        else:
            project = studio.create(project_id, request.title, request.inquiry.model_dump(),
                                    spec['selection']['path'], spec['selection']['author'])
        for reference in request.references:
            studio.add_reference(project_id, reference.model_dump())
        studio.start(project_id, episode_id, restore_config(spec['config']), max_calls=2, max_actions=2, max_rounds=1)
        parent = project['artifacts']['root']
        with Image.open(parent['path']) as image:
            size = image.size
        mask = Image.new('L', size, 255 if revision.method == 'whole' else 0)
        if revision.region is not None:
            l, t, r, b = revision.region
            bounds = [int(l * size[0]), int(t * size[1]), int(r * size[0]), int(b * size[1])]
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError('Region is empty after mapping to parent pixels')
            ImageDraw.Draw(mask).rectangle((bounds[0], bounds[1], bounds[2] - 1, bounds[3] - 1), fill=255)
        mask_path = directory / 'allowed_mask.png'
        if mask_path.exists():
            with Image.open(mask_path) as existing:
                if existing.mode != 'L' or existing.size != size or ImageChops.difference(existing, mask).getbbox():
                    raise ValueError('Frozen continuation mask changed')
        else:
            mask.save(mask_path)
        directions = []
        for index, seed in enumerate(revision.seeds):
            action_id = f'variant_{index + 1}'
            plan = {'parent': parent['path'], 'parent_sha256': parent['sha256'], 'instruction': revision.instruction,
                    'target': revision.target, 'allowed_mask': str(mask_path), 'method': revision.method,
                    'seed': seed, 'max_calls': 1, 'visual_checks': revision.visual_checks, 'author': revision.author}
            path = directory / f'{action_id}.json'
            _ensure_json(path, plan)
            load_plan(path)
            directions.append({'id': action_id, 'title': f'Realization {index + 1}',
                'intended_change': revision.intended_change, 'preservation_constraints': revision.preservation_constraints,
                'source_ids': revision.source_ids, 'relationship': revision.relationship,
                'relationship_kind': revision.relationship_kind, 'realization_group': 'revision', 'plan': str(path)})
        batch = {'id': 'revision', 'parent_artifact_id': 'root', 'author': revision.author,
                 'origin': revision.origin, 'directions': directions}
        batch_path = directory / 'batch.json'
        _ensure_json(batch_path, batch)
        studio.propose(project_id, episode_id, batch_path)

    def _spec(self, continuation_id):
        values = read_continuations(self.root, _identifier(continuation_id))
        if not values:
            raise ReviewNotFound('Unknown continuation')
        return values[0]

    def get(self, continuation_id):
        from .studio import read_projects
        spec = self._spec(continuation_id)
        values = read_projects(self.root, spec['project_id'])
        project = values[0] if values else None
        episode = project['episodes'].get(spec['episode_id']) if project else None
        collection_id = 'continuation_' + continuation_id
        reviews = ReviewStore(self.root)
        try:
            history = reviews.history(collection_id)
            review_status = {'collection_id': collection_id, 'votes': len(history['votes']),
                             'advancements': len(history['advancements']), 'human_labels': history['votes'] or None}
        except ReviewNotFound:
            review_status = {'collection_id': None, 'votes': 0, 'advancements': 0, 'human_labels': None}
        outputs = [a for a in project['artifacts'].values() if a['id'] != 'root'] if project else []
        return spec | {'status': episode['status'] if episode else 'preparing',
            'model_calls': episode['model_calls'] if episode else 0,
            'parent': project['artifacts']['root'] if project else None, 'outputs': outputs,
            'observations': project['observations'] if project else [], 'reviews': review_status,
            'method_note': ('FLUX edits the full saved parent. Prompt requests do not guarantee local preservation.'
                if spec['request']['revision']['method'] == 'whole' else
                'FLUX proposes an edit from the full parent; a hard rectangular compositor preserves decoded pixels outside the declared mask. Meaning and boundaries inside it remain uncontrolled. Raw spillover is retained in each attempt.'),
            'artistic_effectiveness': None}

    def execute(self, continuation_id, backend_factory=FluxBackend, *, resource_root=None, resource_config=None):
        """Execute only saved actions; uncertain charged calls follow ledger recovery."""
        spec = self._spec(continuation_id)
        immutable = {k: v for k, v in spec.items() if k != 'created_at'}
        frozen_resource_root = Path(spec.get('resource_root', self.root)).resolve()
        if resource_root is not None and Path(resource_root).resolve() != frozen_resource_root:
            raise ValueError('Execution resource root differs from the frozen continuation budget root')
        with Studio(self.root) as studio, studio.ledger.lock('continuation_' + continuation_id):
            # Complete interrupted preparation using existing frozen inputs.
            self._materialize(studio, immutable)
            if resource_root is not None or backend_factory is FluxBackend:
                from .resources import SessionBudget
                from .ledger import BudgetExceeded, LedgerError
                with SessionBudget(frozen_resource_root, resource_config) as budget, budget.workload_lock():
                    def budgeted_factory(config):
                        if budget.summary()['renderer_remaining'] < 1:
                            raise BudgetExceeded('Session renderer cap exhausted before model loading')
                        backend = backend_factory(config)
                        generate = backend.generate
                        def budgeted_generate(prompt, seed, *args, **kwargs):
                            key = spec['invocation_keys'][str(seed)]
                            if not budget.reserve(key, 'renderer'):
                                raise LedgerError('Existing session reservation cannot authorize repeat inference: ' + key)
                            # Returning pixels is not a durable result. The
                            # controlled ledger commits raw bytes before the
                            # session reservation is reconciled below.
                            return generate(prompt, seed, *args, **kwargs)
                        backend.generate = budgeted_generate
                        return backend
                    try:
                        studio.run(spec['project_id'], spec['episode_id'], budgeted_factory)
                    finally:
                        self._reconcile_session(studio, spec, budget)
            else:
                # Custom factories are controlled substitutes or already budgeted
                # workers. Production CLI always uses the session wrapper above.
                studio.run(spec['project_id'], spec['episode_id'], backend_factory)
            project = studio.get(spec['project_id'])
            self._register_followup(project, spec)
        manifest = self.get(continuation_id)
        directory = self.root / 'continuations' / continuation_id
        write_json(directory / 'manifest.json', manifest)
        self.export_gallery(continuation_id)
        return manifest

    def _reconcile_session(self, studio, spec, budget):
        """Reconcile charges from committed action evidence, never authorize a retry."""
        project = studio.get(spec['project_id'])
        session = {row['key']: row for row in budget.summary()['calls']}
        rows = []
        for action_id in project['episodes'][spec['episode_id']]['action_ids']:
            action = project['actions'][action_id]
            job = studio.ledger.get_job(action['job_id'])
            seed = job['spec']['plan']['seed']
            key = spec['invocation_keys'][str(seed)]
            attempt = studio.ledger.get_attempt(action['job_id'], 0)
            registered = next((r for r in studio.ledger.artifacts(action['job_id'])
                               if r['path'] == 'attempt_0_raw.png' and r['kind'] == 'raw_model_output'), None)
            raw_path = Path(job['directory']) / 'attempt_0_raw.png'
            valid_raw = bool(registered and raw_path.is_file() and sha256(raw_path) == registered['sha256'])
            charged = bool(attempt and attempt['reserved'])
            if charged and valid_raw:
                evidence_status, reason = 'completed', 'Hash-verified raw bytes are registered in the charged action ledger'
            elif charged and attempt['status'] == 'failed' and not raw_path.exists():
                evidence_status, reason = 'failed', 'The charged action recorded a known failure without raw bytes'
            elif key in session or charged:
                evidence_status, reason = 'uncertain', 'Charge exists without committed raw evidence or a known terminal failure; never repeat automatically'
            else:
                evidence_status, reason = 'not_reserved', 'No session or local inference charge exists'
            current = session.get(key)
            before = current['status'] if current else None
            discrepancy = None
            if current:
                if before in ('completed', 'failed'):
                    if before != evidence_status:
                        discrepancy = 'Immutable settled session history differs from current local evidence; retain both and reconcile explicitly'
                elif before != evidence_status:
                    budget.settle(key, evidence_status)
            elif charged:
                discrepancy = 'The local charge has no shared-session reservation; no reservation is fabricated retroactively'
            after = (budget.summary()['calls'] if current else [])
            after_status = next((r['status'] for r in after if r['key'] == key), None)
            rows.append({'key': key, 'seed': seed, 'action_id': action_id, 'job_id': action['job_id'],
                'local_attempt_status': attempt['status'] if attempt else None, 'local_charged': charged,
                'registered_raw_sha256': registered['sha256'] if registered else None,
                'raw_hash_verified': valid_raw, 'raw_file_present': raw_path.exists(),
                'session_status_before': before, 'session_status_after': after_status,
                'evidence_status': evidence_status, 'reason': reason, 'discrepancy': discrepancy})
        report = {'schema_version': 1, 'continuation_id': spec['id'], 'origin': 'execution_accounting_audit',
            'audited_at': now(), 'rows': rows, 'discrepancies': [r for r in rows if r['discrepancy']],
            'policy': 'All reservations remain charged. Audit does not authorize inference or rewrite settled session history.',
            'human_quality': None}
        write_json(self.root / 'continuations' / spec['id'] / 'session_accounting.json', report)
        return report

    def reconcile_session(self, continuation_id):
        """Reconcile a stopped worker without loading a model or dispatching work."""
        from .resources import SessionBudget
        spec = self._spec(continuation_id)
        resource_root = Path(spec.get('resource_root', self.root)).resolve()
        with Studio(self.root) as studio, studio.ledger.lock('continuation_' + continuation_id):
            with SessionBudget(resource_root) as budget, budget.workload_lock():
                return self._reconcile_session(studio, spec, budget)

    def _register_followup(self, project, spec):
        if project['episodes'][spec['episode_id']]['status'] in ('queued', 'running', 'interrupted'):
            return None
        outputs = [a for a in project['artifacts'].values() if a['id'] != 'root']
        if not outputs:
            return None
        candidates = [{'id': 'image-001', 'path': project['artifacts']['root']['path'], 'label': 'Saved parent',
                       'studio': {'project_id': project['id'], 'episode_id': spec['episode_id'], 'artifact_id': 'root'}}]
        pairs = []
        for index, artifact in enumerate(outputs, 2):
            candidate_id = f'image-{index:03}'
            candidates.append({'id': candidate_id, 'path': artifact['path'], 'label': f'Realized revision {index - 1}',
                'parent_candidate_id': 'image-001', 'studio': {'project_id': project['id'],
                'episode_id': spec['episode_id'], 'artifact_id': artifact['id']}})
            pairs.append({'id': f'comparison-{index - 1:03}', 'candidate_ids': ['image-001', candidate_id], 'kind': 'before_after'})
        collection = ReviewStore(self.root).register_collection('continuation_' + spec['id'],
            title=spec['request']['title'], brief=spec['request']['brief'], candidates=candidates, pairs=pairs,
            presentation={'condition': 'statement_free', 'viewing_conditions': spec['request']['inquiry']['viewing_conditions']},
            pool_expected_count=3)
        queue = [{'schema_version': 1, 'collection_id': collection['id'], 'pair_id': p['id'],
            'candidate_ids': p['candidate_ids'], 'candidate_hashes': {c['id']: c['sha256'] for c in collection['candidates'] if c['id'] in p['candidate_ids']},
            'presentation': collection['presentation'], 'assignment_rule': 'reviewer-specific balanced stable A/B order',
            'human_labels': None} for p in collection['pairs']]
        directory = self.root / 'continuations' / spec['id']
        _atomic_text(directory / 'review_assignments.jsonl', ''.join(encoded(row) + '\n' for row in queue))
        return collection

    def observe(self, continuation_id, artifact_id, note, author, origin):
        spec = self._spec(continuation_id)
        with Studio(self.root) as studio:
            record = studio.observe(spec['project_id'], artifact_id, note, author, origin)
        self.export_gallery(continuation_id)
        return record

    def gallery_html(self, continuation_id, *, web=False):
        """Read-only gallery projection. Local export links preserve original bytes."""
        import os
        from urllib.parse import quote
        manifest = self.get(continuation_id)
        directory = self.root / 'continuations' / continuation_id
        esc = lambda value: html.escape(str(value), quote=True)
        cards = []
        parent_url = None
        for artifact in ([manifest['parent']] if manifest['parent'] else []) + manifest['outputs']:
            path = Path(artifact['path'])
            if sha256(path) != artifact['sha256']:
                raise ReviewConflict('Gallery artifact hash changed')
            url = esc(f'/api/projects/{quote(manifest["project_id"], safe="")}/artifacts/{quote(artifact["id"], safe="")}'
                      if web else os.path.relpath(path, directory))
            if artifact['id'] == 'root':
                parent_url = url
            cards.append(f'<figure><a href="{url}"><img src="{url}" alt="{esc(artifact["id"])}"></a><figcaption>{esc(artifact["id"])} · {esc(artifact.get("origin"))}<br>SHA-256 {esc(artifact["sha256"])}</figcaption></figure>')
        region = self._gallery_edit_region(manifest)
        preview = '<p>Prepared edit-region preview is available after preparation completes.</p>'
        if region is not None:
            width, height = region['size']
            left, top, right, bottom = region['bounds']
            protection = (f'{region["protected"]:,} of {width * height:,} pixels protected outside the rectangle.'
                          if region['protected'] else '0 protected pixels — the whole image may change.')
            warning = ' continuation-protection-warning' if region['protected'] == 0 else ''
            preview = f'''<section class="continuation-plan" aria-labelledby="planned-edit-title">
<h2 id="planned-edit-title">Planned edit region on the saved parent</h2>
<div class="continuation-plan-layout"><figure class="continuation-mask-preview">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="edit-region-title edit-region-description">
<title id="edit-region-title">Exact prepared edit region</title>
<desc id="edit-region-description">The complete saved parent with the allowed edit rectangle outlined. Pixel bounds: x {left} to {right}, y {top} to {bottom}; right and bottom are exclusive. {esc(protection)}</desc>
<image href="{parent_url}" x="0" y="0" width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>
<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="none" stroke="#101010" stroke-width="5" vector-effect="non-scaling-stroke"/>
<rect x="{left}" y="{top}" width="{right - left}" height="{bottom - top}" fill="none" stroke="#ffda44" stroke-width="2.5" stroke-dasharray="8 5" vector-effect="non-scaling-stroke"/>
</svg><figcaption>Yellow dashed boundary: allowed edit area. The entire {width} × {height} parent is shown; the overlay does not alter its saved bytes. <a href="{parent_url}">Open original parent</a></figcaption></figure>
<div class="continuation-plan-details"><p><strong>Prepared pixel bounds:</strong> x [{left}, {right}), y [{top}, {bottom}). Right and bottom bounds are exclusive.</p>
<p class="continuation-protection{warning}"><strong>{esc(protection)}</strong></p>
<p>This rectangle is the saved geometric mask, not an inferred object boundary or evidence that the requested subject lies inside it.</p>
<h3>Exact saved edit instruction</h3><pre class="continuation-instruction">{esc(manifest['request']['revision']['instruction'])}</pre>
<p class="continuation-mask-hash">Prepared mask SHA-256: {region['sha256']}</p></div></div></section>'''
        selection = manifest['selection']
        body = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(manifest['request']['title'])}</title>
<style>body{{font:18px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;background:#f2f0e9;color:#222}}section{{display:flex;gap:20px;flex-wrap:wrap}}figure{{margin:0;flex:1;min-width:260px}}img{{width:100%;max-height:700px;object-fit:contain}}figcaption{{font-size:12px;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.continuation-plan{{display:block;margin:28px 0}}.continuation-plan-layout{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,300px),1fr));gap:24px;margin-top:16px}}.continuation-mask-preview{{min-width:0}}.continuation-mask-preview svg{{display:block;width:100%;height:auto}}.continuation-plan-details{{min-width:0}}.continuation-instruction{{max-height:none;overflow:visible}}.continuation-mask-hash{{font-size:12px;overflow-wrap:anywhere}}.continuation-protection-warning{{border:2px solid #ae7100;padding:12px}}</style>
<h1>{esc(manifest['request']['title'])}</h1><p>Development gallery · provenance visible</p><p>{esc(manifest['request']['brief'])}</p>
<p>Parent selected by {esc(selection['author'])} ({esc(selection['origin'])}): {esc(selection['reason'])}</p>
<p>Requested change by {esc(manifest['request']['revision']['author'])}: {esc(manifest['request']['revision']['intended_change'])}</p>
<p>{esc(manifest['method_note'])}</p><p>Execution: {esc(manifest['status'])}. Charged model calls: {manifest['model_calls']}/2. Human follow-up reviews: {manifest['reviews']['votes']}. Artistic effectiveness remains unassessed until actual observations and situated review.</p>
{preview}
<section>{''.join(cards)}</section><h2>Observed results</h2><pre>{esc(encoded(manifest['observations']) if manifest['observations'] else 'No observations saved yet.')}</pre>
<details><summary>Frozen inquiry, sources and action</summary><pre>{esc(json.dumps(manifest['request'], indent=2, ensure_ascii=False))}</pre></details></html>'''
        if web:
            import re
            body = re.sub(r'<style>.*?</style>', '<link rel="stylesheet" href="/static/dashboard.css">', body, flags=re.S)
        return body

    def _gallery_edit_region(self, manifest):
        """Read the exact prepared mask; never infer a region from the instruction."""
        from .studio import read_projects
        projects = read_projects(self.root, manifest['project_id'])
        if not manifest['parent'] or not projects:
            return None
        actions = projects[0]['actions']
        if any(key not in actions for key in ('variant_1', 'variant_2')):
            return None
        mask_path = self.root / 'continuations' / manifest['id'] / 'allowed_mask.png'
        mask_hash = sha256(mask_path)
        if any(actions[key]['source_hashes'].get('allowed_mask') != mask_hash
               for key in ('variant_1', 'variant_2')):
            raise ReviewConflict('Gallery prepared mask hash changed')
        with Image.open(manifest['parent']['path']) as parent, Image.open(mask_path) as mask:
            if mask.mode != 'L' or mask.size != parent.size:
                raise ReviewConflict('Gallery prepared mask dimensions or mode changed')
            histogram = mask.histogram()
            bounds = mask.getbbox()
            if sum(histogram[1:255]) or bounds is None:
                raise ReviewConflict('Gallery requires the prepared binary rectangular mask')
            left, top, right, bottom = bounds
            if histogram[255] != (right - left) * (bottom - top):
                raise ReviewConflict('Gallery prepared mask is not rectangular')
            return {'size': mask.size, 'bounds': bounds, 'protected': histogram[0], 'sha256': mask_hash}

    def export_gallery(self, continuation_id):
        path = self.root / 'continuations' / continuation_id / 'gallery.html'
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_text(path, self.gallery_html(continuation_id))
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description='Explicit saved-parent continuation; preparation never performs inference.')
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    parser.add_argument('--resource-root', type=Path, default=None, help='Shared overnight budget; defaults to artifact root for real inference')
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--id', required=True)
    prepare.add_argument('--request', type=Path, required=True)
    prepare.add_argument('--config', type=Path, required=True, help='JSON object using the existing Config fields')
    for name in ('execute', 'show', 'gallery'):
        command = sub.add_parser(name)
        command.add_argument('--id', required=True)
    args = parser.parse_args(argv)
    store = ContinuationStore(args.artifacts)
    try:
        if args.command == 'prepare':
            config = restore_config(json.loads(args.config.read_text()))
            result = store.prepare(args.id, json.loads(args.request.read_text()), config, resource_root=args.resource_root)
        elif args.command == 'execute':
            result = store.execute(args.id, resource_root=args.resource_root)
        elif args.command == 'gallery':
            result = {'gallery': str(store.export_gallery(args.id))}
        else:
            result = store.get(args.id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
