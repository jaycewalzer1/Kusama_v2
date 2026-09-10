"""Explicit rendering of model-selected research hypotheses through frozen FLUX."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import shutil
from typing import Literal
import uuid

from PIL import Image
from pydantic import Field, model_validator

from .artifacts import image_record, save_image, sha256, write_json
from .backend import FluxBackend
from .config import MODEL_ID
from .costs import tracked_local_call
from .evidence import digest, immutable_json
from .execution import canonical_config, restore_config
from .experiment import Measurement
from .inspiration import preflight_model_load, require_charged_invocation
from .image_prompt import conditioning_record, resolve_selections, validate_assets
from .ledger import Ledger, SpecMismatch
from .resource_snapshot import session_snapshot
from .research import Strict, read_report
from .render_prompt import compose_prompt
from .resources import SessionBudget
from .reviews import ReviewStore
from .study_cycle import DevelopmentCycle

VERSION = 'research-render-v5-development-cycle'


def code_identity():
    base = Path(__file__).parent
    names = ['research_render.py', 'study_development.py', 'study_cycle.py', 'research_context.py', 'research_sources.py', 'render_prompt.py', 'research_judge.py', 'research_rules.py', 'director.py', 'research.py', 'image_prompt.py', 'backend.py', 'config.py',
             'execution.py', 'inspiration.py', 'resources.py', 'ledger.py',
             'artifacts.py', 'costs.py', 'experiment.py', 'evidence.py',
             'reviews.py', 'resource_snapshot.py']
    identity = {name: sha256(base / name) for name in names}
    lock = base.parents[1] / 'uv.lock'
    if lock.is_file():
        identity['uv.lock'] = sha256(lock)
    return identity


class ResearchRenderRequest(Strict):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,48}$')
    research_output: Path
    hypothesis_ids: list[str] = Field(min_length=1, max_length=12)
    seeds: list[int] = Field(min_length=1, max_length=24)
    resource_root: Path
    render_profile_id: str = Field(min_length=1, max_length=100)
    render_config: dict
    selection_origin: Literal['operator_selection', 'all_model_hypotheses']
    selection_reason: str = Field(min_length=1, max_length=1000)
    register_review: bool = True
    study_development: bool = False
    development_cycle: DevelopmentCycle | None = None

    @model_validator(mode='after')
    def selection(self):
        if self.development_cycle:
            if not self.study_development:
                raise ValueError('Development cycles require inspected provisional studies')
            if len(self.seeds) + self.development_cycle.max_steps > 12:
                raise ValueError('Initial images plus development actions must fit the 12-branch inspection capacity')
        if len(set(self.hypothesis_ids)) != len(self.hypothesis_ids):
            raise ValueError('Hypothesis IDs must be unique')
        if len(self.seeds) % len(self.hypothesis_ids) or len(self.seeds) // len(self.hypothesis_ids) not in (1, 2):
            raise ValueError('Supply exactly one or two distinct seeds per hypothesis')
        if len(set(self.seeds)) != len(self.seeds) or any(type(s) is not int or not 0 <= s < 2**63 for s in self.seeds):
            raise ValueError('Supply distinct nonnegative 63-bit integer seeds')
        return self


def plan_render(output, request):
    return _plan_render(output, request)


def _plan_render(output, request, *, research=None, parent=None, origin=None, research_plan=None):
    """Also freeze a model-authored follow-up from the charged study decision."""
    output = Path(output).resolve()
    request = ResearchRenderRequest.model_validate(request)
    value = request.model_dump(mode='json')
    if (output / 'render_plan.json').exists():
        spec = preflight_render(output)
        if spec['request_sha256'] != digest(value) or spec.get('origin') != origin:
            raise SpecMismatch('Frozen render request changed')
        return spec
    if output.exists() and any(output.iterdir()):
        raise ValueError('New render plan needs an empty directory')
    research = research if research is not None else read_report(request.research_output)
    available = {h['id']: h for h in research['hypotheses']}
    if set(request.hypothesis_ids) - available.keys():
        raise ValueError('Render only committed research hypotheses')
    if request.selection_origin == 'all_model_hypotheses' and set(request.hypothesis_ids) != available.keys():
        raise ValueError('Subset selection must be attributed to the operator')
    config = canonical_config(restore_config(request.render_config))
    if config.output_dir != output or config.max_retries or not config.local_files_only or config.parent_image:
        raise ValueError('Explicit local zero-retry initial generation config required with this output_dir')
    output.mkdir(parents=True, exist_ok=True)
    selected = deepcopy([available[i] for i in request.hypothesis_ids])
    if parent:
        original = Path(parent['path'])
        if sha256(original) != parent['sha256']:
            raise SpecMismatch('Selected study parent changed')
        target = output / 'inputs' / 'parent.png'
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(original, target)
        parent = {**parent, 'path': str(target), 'original_path': str(original)}
    for h in selected:
        resolve_selections(h.get('render_images', []), h['source_snapshots'])
    for h in selected:
        for source in h['source_snapshots']:
            for asset in source.get('images', []) + source.get('byte_evidence', []):
                original = Path(asset['path'])
                if sha256(original) != asset['sha256']:
                    raise SpecMismatch('Research evidence changed')
                target = output / 'evidence' / (asset['sha256'] + original.suffix.lower())
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copyfile(original, target)
                asset.update(original_path=str(original), path=str(target))
    per_hypothesis = len(request.seeds) // len(selected)
    actions = [{'id': f"{origin.get('image_prefix', 'followup')}-image-{i:03}" if origin else f'image-{i:03}', 'hypothesis_id': h['id'], 'prompt': compose_prompt(h),
                'prompt_components': {k: h[k] for k in ('composition', 'render_instruction')},
                'seed': request.seeds[i], 'source_ids': [s['source_id'] for s in h['sources']],
                'reference_images': resolve_selections(h.get('render_images', []), h['source_snapshots'])}
               for i, h in enumerate(h for h in selected for _ in range(per_hypothesis))]
    for action in actions:
        action['parent'] = parent
        action['conditioning'] = conditioning_record(action['reference_images'], parent=parent)
    research_plan = research_plan if research_plan is not None else json.loads((request.research_output / 'plan.json').read_text())
    spec = {'schema_version': 1, 'kind': 'research_render', 'version': VERSION,
        'request': value, 'request_sha256': digest(value), 'problem': research['problem'],
        'project_id': research_plan['request']['project_id'],
        'brief_id': research_plan['request']['brief_id'],
        'research_plan_sha256': research['plan_sha256'], 'hypotheses': selected,
        'director_config': research_plan['director_config'],
        'operator_context': {k: research_plan['request'].get(k) for k in
            ('authorship', 'prior_context', 'medium_constraint', 'seed_artists', 'seed_works')},
        'judge_calls': len(actions) if research_plan.get('progress_policy', {}).get('grounded_search') and not request.study_development and not origin else 0,
        'study_reasoning_calls': (len(actions) + 2 * request.development_cycle.max_steps + 1 if request.development_cycle
                                  else len(actions) + 2) if request.study_development else 0,
        'study_revision_cap': (request.development_cycle.max_steps if request.development_cycle else 1) if request.study_development else 0,
        'origin': origin,
        'actions': actions, 'render_config': config.to_dict(), 'fixture': research['fixture'],
        'code_sha256': code_identity(),
        'boundary': 'FLUX receives the saved text and only explicitly selected reference images. References are not lineage parents; image conditioning does not establish artistic success.'}
    if request.development_cycle:
        spec['cycle_research_template'] = research_plan['request']
        spec['cycle_source_config'] = research_plan['source_config']
    spec = json.loads(json.dumps(spec))
    immutable_json(output / 'render_plan.json', spec)
    immutable_json(output / 'render_identity.json', {'sha256': sha256(output / 'render_plan.json')})
    return spec


def preflight_render(output):
    output = Path(output).resolve()
    spec = json.loads((output / 'render_plan.json').read_text())
    if (json.loads((output / 'render_identity.json').read_text())['sha256'] != sha256(output / 'render_plan.json')
            or spec['code_sha256'] != code_identity() or spec['version'] != VERSION):
        raise SpecMismatch('Frozen research render plan or implementation changed')
    if digest(spec['request']) != spec['request_sha256']:
        raise SpecMismatch('Research render request changed')
    ResearchRenderRequest.model_validate(spec['request'])
    if spec.get('origin') and sha256(Path(spec['origin']['path'])) != spec['origin']['sha256']:
        raise SpecMismatch('Model study decision changed')
    for h in spec['hypotheses']:
        for source in h['source_snapshots']:
            for asset in source.get('images', []) + source.get('byte_evidence', []):
                path = Path(asset['path']).resolve()
                if not path.is_relative_to(output) or sha256(path) != asset['sha256']:
                    raise SpecMismatch('Frozen render source evidence changed')
    hypotheses = {h['id']: h for h in spec['hypotheses']}
    for action in spec['actions']:
        h = hypotheses[action['hypothesis_id']]
        if action['prompt'] != compose_prompt(h) or action['prompt_components'] != {k: h[k] for k in ('composition', 'render_instruction')}:
            raise SpecMismatch('Render prompt differs from the complete model-authored composition/instruction')
        parent = action.get('parent')
        if parent and (not Path(parent['path']).resolve().is_relative_to(output) or sha256(Path(parent['path'])) != parent['sha256']):
            raise SpecMismatch('Frozen study parent changed')
        expected = resolve_selections(h.get('render_images', []), h['source_snapshots'])
        if action['reference_images'] != expected or action['conditioning'] != conditioning_record(expected, parent=parent):
            raise SpecMismatch('Frozen image conditioning differs from model selection')
    return spec


def run_render(output, *, backend_factory=FluxBackend, before_dispatch=None, director_factory=None, on_stage=None):
    output = Path(output).resolve()
    spec = preflight_render(output)
    if spec['fixture'] and backend_factory is FluxBackend:
        raise ValueError('Fixture research cannot dispatch real FLUX')
    request = spec['request']; root = Path(request['resource_root']).resolve()
    session = session_snapshot(root)
    config = restore_config(spec['render_config'])
    if not spec['fixture']:
        profile = session['config']['render_profiles'].get(request['render_profile_id'])
        if profile is None or any(spec['render_config'].get(k) != v for k, v in profile.items()):
            raise SpecMismatch('Named render profile differs from the admitted session')
        if config.device != session['config']['device']:
            raise SpecMismatch('Renderer device differs from admitted session')
    immutable_json(output / 'resource_binding.json', {'snapshot': session, 'root': str(root)})
    job_id = 'research_render_' + request['id']
    image_origin = ('render_fixture' if spec['fixture'] else
                    'frozen_flux_edit' if any(a.get('parent') for a in spec['actions']) else 'frozen_flux_generation')
    images, failures = [], []
    with Ledger(output / 'jobs.sqlite3') as ledger, ledger.lock(job_id), SessionBudget(root) as budget:
        ledger.create_job(job_id, spec, len(spec['actions']), output)
        for name in ('render_plan.json', 'render_identity.json', 'resource_binding.json'):
            ledger.register_artifact(job_id, name, kind='research_render_plan')
        ledger.verify_artifacts(job_id)
        for index, action in enumerate(spec['actions']):
            if before_dispatch is not None:
                before_dispatch()
            path = output / (action['id'] + '.png')
            record_path = output / (action['id'] + '.json')
            attempt = ledger.get_attempt(job_id, index)
            key = f'{job_id}:renderer:{index}'
            if attempt is not None:
                require_charged_invocation(ledger, job_id, index, budget, key, 'renderer')
                if path.exists():
                    with Image.open(path) as im:
                        im.load()
                        if im.mode != 'RGB' or im.size != (config.width, config.height):
                            raise SpecMismatch('Uncertain raw render shape requires reconciliation')
                    record = {**image_record(path, image_id=action['id'], parent=action.get('parent'), seed=action['seed'], origin=image_origin),
                              'path': str(path), 'hypothesis_id': action['hypothesis_id'], 'source_ids': action['source_ids'],
                              'conditioning': action['conditioning']}
                    if record_path.exists():
                        saved = json.loads(record_path.read_text())
                        if saved['sha256'] != record['sha256']:
                            raise SpecMismatch('Research render output changed')
                        record = saved
                    else:
                        immutable_json(record_path, record)
                    ledger.register_artifact(job_id, path.name, kind='image')
                    ledger.register_artifact(job_id, record_path.name, kind='image_record')
                    ledger.save_attempt(job_id, index, status='completed')
                    budget.settle(key, 'completed'); images.append(record)
                    continue
                if attempt['status'] == 'failed':
                    failures.append({'action': action, 'attempt': attempt}); continue
                raise RuntimeError('Uncertain charged render; reconcile before another dispatch')
            if path.exists() or record_path.exists() or budget.db.execute('SELECT 1 FROM calls WHERE key=?', (key,)).fetchone():
                raise SpecMismatch('Unmatched render artifact/reservation requires reconciliation')
            backend = None
            with budget.workload_lock():
                try:
                    preflight_model_load(budget, 'renderer', estimated_seconds=180)
                    backend = backend_factory(config)
                    with tracked_local_call(root, job_id, 'load-' + uuid.uuid4().hex[:12], MODEL_ID, scope='model_load'):
                        backend.prepare()
                    if hasattr(backend, 'prompt_token_count') and backend.prompt_token_count(action['prompt']) > config.max_sequence_length:
                        raise ValueError('Saved instruction exceeds FLUX tokenizer capacity; no truncation')
                    if before_dispatch is not None:
                        before_dispatch()
                    validate_assets(action['reference_images'])
                    if not budget.reserve(key, 'renderer', estimated_seconds=180):
                        raise SpecMismatch('Existing render reservation cannot dispatch again')
                    ledger.open_attempt(job_id, index, payload={'action': action})
                    ledger.reserve_call(job_id, index)
                    try:
                        with tracked_local_call(root, job_id, action['id'], MODEL_ID) as cost:
                            with Measurement(config.device) as measure:
                                im = backend.generate(action['prompt'], action['seed'],
                                    **({'parent_path': Path(action['parent']['path'])} if action.get('parent') else {}),
                                    **({'reference_images': action['reference_images']} if action['reference_images'] else {}))
                                if im.mode != 'RGB' or im.size != (config.width, config.height):
                                    raise ValueError('Unexpected FLUX image shape or mode')
                                save_image(im, path)
                            cost['measurements'] = measure.result
                        record = {**image_record(path, image_id=action['id'], parent=action.get('parent'), seed=action['seed'], origin=image_origin),
                                  'path': str(path), 'hypothesis_id': action['hypothesis_id'],
                                  'source_ids': action['source_ids'], 'measurements': measure.result,
                                  'conditioning': action['conditioning']}
                        immutable_json(record_path, record)
                        ledger.register_artifact(job_id, path.name, kind='image')
                        ledger.register_artifact(job_id, record_path.name, kind='image_record')
                        ledger.save_attempt(job_id, index, status='completed'); budget.settle(key, 'completed')
                        images.append(record)
                    except Exception as exc:
                        if path.exists():
                            raise RuntimeError('Saved charged output requires reconciliation; no replacement render') from exc
                        failure = {'action': action, 'error': f'{type(exc).__name__}: {exc}', 'human_quality': None}
                        immutable_json(output / f'failure-{index:03}.json', failure)
                        ledger.register_artifact(job_id, f'failure-{index:03}.json', kind='failure')
                        ledger.save_attempt(job_id, index, status='failed', payload=failure); budget.settle(key, 'failed')
                        failures.append(failure)
                finally:
                    if backend is not None:
                        backend.close()
        report = {'kind': 'research_render', 'status': 'complete' if len(images) == len(spec['actions']) else 'partial',
                  'images': images, 'failures': failures, 'renderer_attempts': ledger.call_count(job_id),
                  'human_quality': None, 'training_performed': False, 'fixture': spec['fixture'], 'boundary': spec['boundary']}
        if len(images) >= 2 and request.get('register_review', True):
            collection = ReviewStore(output).register_collection(job_id, title='Research-led painting studies', brief=spec['problem'],
                candidates=[{'id': im['id'], 'path': im['path'], 'label': im['id'],
                             'direction_id': im['hypothesis_id']} for im in images],
                presentation={'condition': 'statement_free', 'viewing_conditions': 'Equal-size images; research explanations withheld.'},
                pool_expected_count=len(spec['actions']))
            report['review_collection_id'] = collection['id']
            report['review_root'] = str(output)
        ledger.update_job(job_id, status=report['status'])
        ledger.export_events(output / 'events.jsonl', job_id=job_id, recover=True)
        write_json(output / 'report.json', report)
    if spec.get('judge_calls') and images:
        from .research_judge import run_judge
        report['visible_tests'] = run_judge(output, spec, images, before_dispatch=before_dispatch)
        report['result'] = report['visible_tests']['status']
        report = {'result': report.pop('result'), **report}
        write_json(output / 'report.json', report)
    if request.get('study_development'):
        from .study_development import run_study_development
        kwargs = {'director_factory': director_factory} if director_factory else {}
        development = run_study_development(output, spec, images, backend_factory=backend_factory,
            before_dispatch=before_dispatch, on_stage=on_stage, **kwargs)
        report['study_development'] = development
        if development['status'] != 'complete':
            report['status'] = 'partial'
        write_json(output / 'report.json', report)
    return report
