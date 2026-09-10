"""Verify a completed real panel research process without dispatching any work.

This acceptance gate requires actual model metadata, source acquisition, visual
reinspection and saved renderer outputs. A clean empty stop is a failed outcome
for this gate. Artistic quality and human review remain separately assessed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path

import httpx
from PIL import Image

from kusama.artifacts import sha256
from kusama.evidence import digest
from kusama import research
from kusama.panel_runs import source_identity
from kusama.research_render import preflight_render


def current_implementation_outcomes(directory, identity):
    """A prior success cannot hide a later failure under the same implementation."""
    outcomes = []
    for request_path in sorted(Path(directory).parent.glob('studio-*/request.json')):
        frozen = json.loads(request_path.read_text())
        if frozen.get('code_identity') != identity:
            continue
        manifest_path = request_path.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        outcomes.append({'run': request_path.parent.name, 'status': manifest.get('status', 'missing_manifest')})
    return outcomes


def verify(directory, *, base_url=None):
    directory = Path(directory).resolve()
    checks = []
    def check(name, condition, detail=''):
        checks.append({'name': name, 'passed': bool(condition), 'detail': detail})
    def stage(name, fn):
        try:
            return fn()
        except Exception as exc:
            check(name, False, f'{type(exc).__name__}: {exc}')
            return None
    def read(path):
        return json.loads(Path(path).read_text())
    frozen = stage('panel_request_readable', lambda: read(directory / 'request.json'))
    manifest = stage('panel_manifest_readable', lambda: read(directory / 'manifest.json'))
    report = stage('research_evidence_integrity', lambda: research.read_report(directory / 'research'))
    if frozen and manifest:
        check('panel_request_seal', sha256(directory / 'request.json') == read(directory / 'request_identity.json')['sha256'])
        check('current_implementation', frozen['code_identity'] == source_identity())
        outcomes = current_implementation_outcomes(directory, frozen['code_identity'])
        check('current_implementation_outcomes', bool(outcomes) and all(
            item['status'] in ('completed', 'stopped') for item in outcomes), json.dumps(outcomes))
        check('panel_completed', manifest['status'] == 'completed', manifest['status'])
        req, limits = frozen['request'], frozen['resource_config']
        with research._read_db(directory / 'resources.sqlite3') as db:
            db.row_factory = sqlite3.Row
            calls = [dict(row) for row in db.execute('SELECT * FROM calls ORDER BY created_at')]
            session = db.execute('SELECT config,started_at,deadline FROM session WHERE id=1').fetchone()
        check('admission_matches_frozen_limits', json.loads(session['config']) == limits)
        check('manifest_call_count', manifest['model_calls'] == len(calls))
        check('all_calls_settled_successfully', all(row['status'] == 'completed' for row in calls),
              str({row['status'] for row in calls}))
        reasoning = [row for row in calls if row['kind'] == 'reasoning']
        rendering = [row for row in calls if row['kind'] == 'renderer']
        check('reasoning_call_cap', 0 < len(reasoning) <= req['research_steps'])
        check('renderer_call_cap', 0 < len(rendering) <= limits['renderer_attempt_cap'])
        check('reasoning_token_cap', sum(row['actual_tokens'] if row['actual_tokens'] is not None
              else row['token_reservation'] for row in reasoning) <= limits['reasoning_token_cap'])
        check('no_dispatch_after_deadline', all(row['created_at'] <= session['deadline'] for row in calls))
        if report:
            check('original_problem_preserved', report['problem'] == req['question'])
            check('real_research_not_fixture', report['fixture'] is False)
            check('committed_hypotheses_exist', 0 < len(report['hypotheses']) <= req['directions'])
            check('research_ledger_count', report['reasoning_calls'] == len(reasoning))
    if report:
        director_config = read(directory / 'research/plan.json')['director_config']
        check('no_rejected_response', all(s['status'] != 'response_rejected' for s in report['steps']))
        network = report.get('network') or {}
        check('live_acquisition_completed', network.get('acquisition_states', {}).get('complete', 0) > 0)
        check('no_uncertain_source_work', not any(network.get(key, {}).get('started', 0)
              for key in ('request_states', 'acquisition_states')))
        if network:
            for field, limit in [('requests','max_requests'), ('acquisitions','max_acquisitions'),
                                 ('charged_network_bytes','max_network_bytes'), ('candidates','max_candidates')]:
                check('source_cap_' + field, network[field] <= network['limits'][limit])
        for step in report['steps']:
            index = step['index']
            response = read(directory / 'research' / f'response-{index:03}.json')
            decision = read(directory / 'research' / f'decision-{index:03}.json')
            meta = response.get('metadata', {}); settings = meta.get('settings', {})
            check(f'qwen_runtime_{index}', settings.get('status') == 'ready'
                and settings.get('model_class') == 'Qwen3VLForConditionalGeneration'
                and settings.get('parameter_count', 0) > 0 and settings.get('trainable_parameter_count') == 0
                and settings.get('training') is False,
                'Requires real frozen-model metadata; scripted processor fixtures cannot pass.')
            check(f'pinned_runtime_{index}', settings.get('model_revision') == director_config['model_revision']
                and settings.get('resolved_revision') == director_config['model_revision']
                and settings.get('device') == director_config['device'])
            if frozen and manifest:
                invocation = next((row for row in reasoning if row['key'].endswith(f':reasoning:{index}')), None)
                check(f'token_charge_{index}', invocation is not None and invocation['actual_tokens'] ==
                      meta.get('input_tokens', 0) + meta.get('generated_tokens', 0))
            check(f'actual_token_admission_{index}', meta.get('input_tokens') == decision['context_admission']['input_tokens']
                  and 0 < meta.get('input_tokens', 0) <= decision['context_admission']['input_token_limit']
                  and 0 < meta.get('generated_tokens', 0) <= decision['context_admission']['reserved_output_tokens'])
        for hyp in report['hypotheses']:
            index = hyp['reinspection_step']; seen = report['steps'][index]['seen_images']
            expected = {(asset['source_id'], asset['sha256']) for source in hyp['source_snapshots']
                        for asset in source.get('images', [])}
            check('committed_source_pixels_' + hyp['id'], bool(expected)
                and expected <= {(image['source_id'], image['sha256']) for image in seen})
            check('model_authorship_' + hyp['id'], hyp['origin'] == 'model_proposal' and hyp['human_quality'] is None)
        def source_bytes():
            with research._read_db(directory / 'research/sources/research_sources.sqlite3') as db:
                for url, state, raw in db.execute('SELECT url,state,payload FROM requests'):
                    value = json.loads(raw)
                    if state == 'complete':
                        path = directory / 'research/sources/corpus' / value['path']
                        check('source_bytes_' + digest(url)[:12], sha256(path) == value['sha256'])
        stage('source_response_bytes', source_bytes)
    def renders():
        plan = preflight_render(directory / 'render')
        saved = read(directory / 'render/report.json')
        check('render_complete', saved['status'] == 'complete' and not saved['failures'])
        check('exact_hypothesis_selection', {h['id'] for h in plan['hypotheses']} == {h['id'] for h in report['hypotheses']})
        check('exact_image_count', len(saved['images']) == len(report['hypotheses']) * frozen['request']['images_per_direction'])
        check('exact_profile', plan['request']['render_profile_id'] == frozen['request']['render_profile']
              and all(plan['render_config'].get(k) == v for k, v in frozen['resource_config']['render_profiles'][frozen['request']['render_profile']].items()))
        with research._read_db(directory / 'render/jobs.sqlite3') as db:
            for path, expected in db.execute('SELECT path,sha256 FROM artifacts'):
                check('render_artifact_' + path, sha256(directory / 'render' / path) == expected)
        hypotheses = {h['id']: h for h in report['hypotheses']}
        for action, image in zip(plan['actions'], saved['images']):
            check('saved_prompt_' + action['id'], action['prompt'] == hypotheses[action['hypothesis_id']]['render_instruction'])
            with Image.open(image['path']) as im:
                im.load()
                check('rendered_pixels_' + image['id'], image['origin'] == 'frozen_flux_generation'
                      and im.mode == 'RGB' and list(im.size) == [plan['render_config']['width'], plan['render_config']['height']]
                      and sha256(Path(image['path'])) == image['sha256'] and image.get('parent_sha256') is None
                      and image['seed'] == action['seed'] and bool(image.get('measurements')))
        return saved
    rendered = stage('research_to_render_handoff', renders)
    def http_inspection():
        with httpx.Client(base_url=base_url, timeout=30) as client:
            response = client.get('/api/runs'); response.raise_for_status()
            row = next(row for row in response.json()['runs'] if row['name'] == directory.name)
            response = client.get(f"/api/runs/{row['id']}/inspect"); response.raise_for_status()
            inspection = response.json()
            check('inspection_without_errors', not inspection['errors'], str(inspection['errors']))
            check('inspection_outputs_visible', bool(rendered) and len(inspection['images']) == len(rendered['images']))
            check('inspection_sources_visible', bool(inspection['research'][0]['evidence_images']))
            for image in inspection['images'] + inspection['research'][0]['evidence_images']:
                response = client.get(image['url']); response.raise_for_status()
                check('image_endpoint_' + image['url'], response.headers['content-type'].startswith('image/'))
            names = {entry['name'] for entry in inspection['files']}
            check('inspectable_prompts_and_logs', {'request.json','worker.log','research/plan.json','render/render_plan.json'} <= names)
    if base_url:
        stage('live_http_inspection', http_inspection)
    else:
        check('live_http_inspection', False, 'Supply --base-url for the actual dashboard; filesystem checks alone are incomplete.')
    return {'kind': 'research_process_acceptance', 'run_directory': str(directory),
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'status': 'passed' if checks and all(c['passed'] for c in checks) else 'failed', 'checks': checks,
            'scope': 'Current panel: question → live research → acquired pixels → model hypotheses → FLUX → inspection.',
            'not_automated': ['artistic quality', 'human critique/selection', 'separately authorized selected-parent revision'],
            'calls_dispatched_by_verifier': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run_directory.resolve()):
        parser.error('Write acceptance evidence outside the frozen run directory')
    try:
        result = verify(args.run_directory, base_url=args.base_url)
    except Exception as exc:
        result = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}', 'calls_dispatched_by_verifier': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Each invocation needs a new result path so failed gates remain preserved.
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    failed = [c['name'] for c in result.get('checks', []) if not c['passed']]
    print(json.dumps({'status': result['status'], 'failed_checks': failed, 'error': result.get('error'), 'output': str(args.output)}))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
