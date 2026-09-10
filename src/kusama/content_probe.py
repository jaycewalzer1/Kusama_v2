"""Small frozen-model content diagnostic; no training, retries or human votes.

The Qwen leg deliberately tests text direction only, without archive retrieval.
Existing session, invocation, measurement and artifact machinery owns execution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import uuid

from pydantic import BaseModel, ConfigDict, Field

from .artifacts import sha256, write_json
from .backend import FluxBackend, _torch
from .config import Config
from .costs import CostJournal, tracked_local_call
from .director import DirectorConfig, QwenDirector, MODEL_ID, _unique_object
from .renderer_session import RendererSession
from .evidence import immutable_json
from .inspiration import model_call, preflight_model_load
from .ledger import Ledger, SpecMismatch
from .resources import SessionBudget, load_config

SYSTEM = ('Develop a self-contained image-generation instruction for the supplied artistic brief. '
          'Preserve its subject, critical position, bodily presentation and visible requirements. '
          'You may clarify composition and painting technique. Return only JSON with exactly '
          'instruction and intended_effect. Keep instruction under 140 words and intended_effect '
          'under 40 words. The intended effect is a prediction, not an observed result.')


class Proposal(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    instruction: str = Field(min_length=1, max_length=1800)
    intended_effect: str = Field(min_length=1, max_length=500)


def parse(raw):
    if len(raw) > 16384:
        raise ValueError('Response exceeds diagnostic bound')
    value = Proposal.model_validate(json.loads(raw, object_pairs_hook=_unique_object))
    if not value.instruction.strip() or not value.intended_effect.strip():
        raise ValueError('Empty proposal field')
    return value.model_dump()


def text_proposal(director, brief):
    """Explicit text-only use of the existing frozen VL checkpoint."""
    result = {'raw_response': None, 'output': None,
              'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': brief}],
              'metadata': {'settings': dict(director.execution_settings), 'input_images': [],
                           'modality': 'text_only', 'combined_token_limit': 8192}}
    try:
        tokenizer = director.processor.tokenizer
        prompt = tokenizer.apply_chat_template(result['messages'], tokenize=False, add_generation_prompt=True)
        result['prompt'] = prompt
        inputs = tokenizer([prompt], return_tensors='pt', truncation=False).to(director.execution_settings['device'])
        count = int(inputs['input_ids'].shape[-1])
        result['metadata']['input_tokens'] = count
        if count + director.config.max_new_tokens > 8192:
            raise ValueError('Combined token bound exceeded; no truncation')
        with _torch().inference_mode():
            generated = director.model.generate(**inputs, max_new_tokens=director.config.max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True, temperature=None, top_p=None, top_k=None)
        continuation = generated[:, count:]
        result['metadata'].update(generated_tokens=int(continuation.shape[-1]),
            reached_token_limit=int(continuation.shape[-1]) >= director.config.max_new_tokens)
        result['raw_response'] = tokenizer.batch_decode(continuation, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
    except Exception as exc:
        result['inference_error'] = f'{type(exc).__name__}: {exc}'
        exc.result = result
        raise
    try:
        result['output'] = parse(result['raw_response'])
    except (ValueError, TypeError, RecursionError) as exc:
        result['parse_error'] = str(exc)
    return result


def plan(root, suite_path):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('Plan requires a new directory; resume the existing plan explicitly')
    suite = json.loads(Path(suite_path).read_text())
    if len(suite['briefs']) != 4 or len({b['id'] for b in suite['briefs']}) != 4:
        raise ValueError('This diagnostic requires four distinct briefs')
    session = load_config().model_copy(update={'session_hours': 1.0, 'handoff_reserve_minutes': 5,
        'renderer_attempt_cap': 8, 'reasoning_token_cap': 32768, 'schema_repair_cap': 0})
    profile = 'flux_base4b_study_512_v1'
    config = Config(cache_dir=Path(session.cache_dir).resolve(), output_dir=root,
        device=session.device, local_files_only=True, max_retries=0,
        note='User-requested content capability diagnostic; fresh session, no overnight continuation',
        **session.render_profiles[profile])
    director = DirectorConfig(config.cache_dir, device=config.device, precision='bf16',
                              local_files_only=True, max_new_tokens=512)
    paths = ['content_probe.py', 'director.py', 'backend.py', 'renderer_session.py', 'inspiration.py',
             'resources.py', 'ledger.py', 'costs.py']
    spec = {'schema_version': 1, 'suite': suite, 'system': SYSTEM, 'schema': Proposal.model_json_schema(),
        'authorization': {'user_request': 'create and evaluate a test',
            'scope': 'New bounded local diagnostic: four Qwen calls, eight FLUX calls, zero retries or training.',
            'previous_session': 'Expired overnight session remains unchanged; no old jobs resumed.'},
        'render_profile': profile, 'render_config': config.to_dict(),
        'director_config': {**asdict(director), 'cache_dir': str(director.cache_dir)},
        'seed': 23, 'max_calls': 12, 'retry_cap': 0, 'human_labels': 0,
        'session_config': session.model_dump(mode='json'),
        'code_sha256': {p: sha256(Path(__file__).with_name(p)) for p in paths}}
    immutable_json(root/'plan.json', spec)
    print(json.dumps({'status': 'planned_not_executed', 'root': str(root), 'max_calls': 12}), flush=True)


def run(root):
    root = Path(root).resolve()
    spec = json.loads((root/'plan.json').read_text())
    if spec['system'] != SYSTEM or any(sha256(Path(__file__).with_name(p)) != h for p,h in spec['code_sha256'].items()):
        raise SpecMismatch('Frozen diagnostic implementation changed')
    from .resources import OvernightConfig
    session = OvernightConfig.model_validate(spec['session_config'])
    rc = dict(spec['render_config'])
    for key in ('cache_dir', 'output_dir'):
        rc[key] = Path(rc[key])
    rc['edit_seeds'] = tuple(rc['edit_seeds'])
    config = Config(**rc)
    dc = dict(spec['director_config']); dc['cache_dir'] = Path(dc['cache_dir'])
    qdir, rdir = root/'qwen', root/'render'
    qdir.mkdir(exist_ok=True); rdir.mkdir(exist_ok=True)
    with SessionBudget(root, session) as budget, Ledger(root/'jobs.sqlite3') as ledger, budget.workload_lock():
        ledger.create_job('content_probe', spec, 12, root)
        ledger.register_artifact('content_probe', 'plan.json', kind='immutable_plan')
        ledger.create_job('content_probe_qwen', spec, 4, qdir, parent_id='content_probe')
        ledger.create_job('content_probe_render', spec, 8, rdir, parent_id='content_probe')
        for job in ('content_probe', 'content_probe_qwen', 'content_probe_render'):
            ledger.verify_artifacts(job)
        director = QwenDirector(DirectorConfig(**dc))
        responses = []
        try:
            if any(ledger.get_attempt('content_probe_qwen', i) is None for i in range(4)):
                preflight_model_load(budget, 'reasoning', estimated_seconds=120)
                with tracked_local_call(root, 'content_probe_qwen', 'load-'+uuid.uuid4().hex[:12], MODEL_ID, scope='model_load'):
                    director.prepare()
            for i, brief in enumerate(spec['suite']['briefs']):
                print('Qwen: '+brief['id'], flush=True)
                result = model_call(ledger, 'content_probe_qwen', qdir, budget, i, f'response-{i:02}.json',
                    lambda b=brief: text_proposal(director, b['prompt']), device=config.device)
                responses.append(result)
                print(json.dumps({'brief': brief['id'], 'output': result.get('output'),
                    'error': result.get('parse_error') or result.get('inference_error')}), flush=True)
        finally:
            director.close()
        renderer = RendererSession(budget, ledger, 'content_probe_render', rdir, config, FluxBackend)
        results = []
        try:
            for i, (brief, proposal) in enumerate(zip(spec['suite']['briefs'], responses, strict=True)):
                for j, mode in enumerate(('direct', 'qwen')):
                    if mode == 'qwen' and not proposal.get('output'):
                        results.append({'brief_id': brief['id'], 'mode': mode, 'status': 'not_rendered_invalid_proposal'})
                        continue
                    instruction = brief['prompt'] if mode == 'direct' else proposal['output']['instruction']
                    request = {'brief_id': brief['id'], 'mode': mode, 'instruction': instruction,
                        'seed': spec['seed'], 'parent': None,
                        'prompt_authorship': 'assistant_operator' if mode == 'direct' else 'frozen_qwen',
                        'retry_authorization': 'none; zero retries in this diagnostic'}
                    immutable_json(rdir/f'input-{i*2+j:02}.json', request)
                    ledger.register_artifact('content_probe_render', f'input-{i*2+j:02}.json', kind='render_input')
                    print('FLUX: '+brief['id']+' / '+mode, flush=True)
                    record = renderer.attempt(i*2+j, request)
                    results.append({'brief_id': brief['id'], 'mode': mode, **record})
                    write_json(root/'manifest.json', {'status': 'running', 'results': results})
                    print(json.dumps({'brief': brief['id'], 'mode': mode, 'status': record['status']}), flush=True)
        finally:
            renderer.close()
        for job in ('content_probe', 'content_probe_qwen', 'content_probe_render'):
            ledger.verify_artifacts(job)
            ledger.update_job(job, status='execution_finished')
        with CostJournal(root, read_only=True) as costs:
            cost_summary = costs.summary()
        manifest = {'status': 'execution_finished_pending_assistant_visual_assessment',
            'plan_sha256': sha256(root/'plan.json'), 'results': results,
            'model_calls': ledger.call_count('content_probe'), 'resources': budget.summary(),
            'cost_accounting': cost_summary, 'human_labels': 0, 'training_performed': False}
        write_json(root/'manifest.json', manifest)
        ledger.export_events(root/'events.jsonl')
        print(json.dumps({'status': manifest['status'], 'model_calls': manifest['model_calls']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'run'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--suite', type=Path, default=Path('configs/evaluation/content_probe_v1.json'))
    args = parser.parse_args()
    plan(args.root, args.suite) if args.stage == 'plan' else run(args.root)


if __name__ == '__main__':
    main()
