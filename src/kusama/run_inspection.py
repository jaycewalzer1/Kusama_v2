"""Contained, read-only views joining the saved parts of one run."""
from pathlib import Path
import json
from datetime import datetime, timezone
import hashlib
from .devtools import artifact_directories
from .artifacts import sha256


def frozen_cycle_view(directory, folder, plan):
    """Read-only presentation of this run's own policy, never today's defaults."""
    if not isinstance(plan, dict) or not isinstance(plan.get('request', {}), dict):
        raise ValueError('Malformed render plan')
    policy = plan.get('request', {}).get('development_cycle')
    if not isinstance(policy, dict):
        return None
    identity = json.loads((folder / 'render_identity.json').read_text())
    if identity['sha256'] != sha256(folder / 'render_plan.json'):
        raise ValueError('Development workflow plan changed; its shape cannot be verified')
    return {'policy': policy, 'initial_image_count': len(plan.get('actions', [])),
            'initial_image_ids': [a['id'] for a in plan.get('actions', [])],
            'development_directory': str((folder / 'study_development').relative_to(directory)),
            'problem': plan.get('problem'), 'fixture': plan.get('fixture', False)}


def queued_cycle_view(directory):
    """A queued panel has frozen settings before its render plan exists."""
    path = directory / 'request.json'
    if not path.is_file():
        return None
    frozen = json.loads(path.read_text())
    request = frozen.get('request', {})
    if not request.get('study_development') or 'development_steps' not in request:
        return None
    identity = json.loads((directory / 'request_identity.json').read_text())
    if identity['sha256'] != sha256(path):
        raise ValueError('Frozen run settings changed; development limits cannot be verified')
    return {'policy': {'max_steps': request['development_steps'], 'research_sessions': 1,
                      'research_steps': request['development_research_steps']},
            'initial_image_count': request['directions'] * request['images_per_direction'],
            'initial_image_ids': [], 'development_directory': 'render/study_development',
            'problem': request['question'], 'fixture': False}


def research_transcripts(directory, report):
    """Display saved admitted inputs/replies after the research report verifies them.

    Keep this presentation projection separate from immutable step records. Never
    reconstruct a historical prompt from today's templates or the full context.
    """
    transcripts = []
    previous_draft = None
    for step in report.get('steps', []):
        index = step['index']
        transcript = {'index': index}
        for kind, prefix in (('prompt', 'decision'), ('response', 'response')):
            path = directory / f'{prefix}-{index:03}.json'
            if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                transcript[kind + '_error'] = 'Saved record is not a contained regular file.'
                continue
            try:
                value = json.loads(path.read_text())
                if not isinstance(value, dict):
                    raise ValueError('Expected a saved object')
                transcript[kind] = value
            except (OSError, ValueError) as exc:
                transcript[kind + '_error'] = 'Saved record unavailable: ' + str(exc)
        action = step.get('action') or {}
        if action.get('action') == 'draft' and action.get('hypothesis'):
            import difflib
            current = json.dumps(action['hypothesis'], indent=2, sort_keys=True, ensure_ascii=False).splitlines()
            if previous_draft is not None:
                transcript['draft_diff'] = '\n'.join(difflib.unified_diff(previous_draft, current, fromfile='previous draft', tofile='this draft', lineterm='')) or '(unchanged)'
            previous_draft = current
        admission = transcript.get('prompt', {}).get('context_admission', {})
        metadata = transcript.get('response', {}).get('metadata', {})
        transcript['tokens'] = {'input': metadata.get('input_tokens', admission.get('input_tokens')),
            'output': metadata.get('generated_tokens'), 'projection_stage': admission.get('stage'),
            'removed': transcript.get('prompt', {}).get('payload', {}).get('prompt_projection', {}).get('omitted_or_shortened_fields')}
        transcripts.append(transcript)
    return transcripts


def inspect_run(row, runs):
    directory = row['_directory']
    files = runs.files(row, recursive=True)
    by_path = {item['name']: item for item in files}
    research, timeline, stages = [], [], []
    study_development = []
    development_by_directory = {}
    cycle_view = None
    cycle_plan_unreadable = False
    errors = []
    image_records = {}
    generations_without_images = []
    if row['_data'].get('kind') == 'artistic_studio_run':
        from .panel_runs import PanelRuns
        try:
            record = PanelRuns(directory.parent.parent).get(directory.name)
            timeline.extend({'id': f"event-{item['sequence']}", 'label': item['state'].replace('_', ' '),
                             'detail': item['detail'], 'at': item['at'], 'origin': 'studio coordinator'} for item in record['events'])
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    for folder, names in artifact_directories(directory):
        if folder.name == 'study_development' and 'report.json' in names:
            try:
                saved = json.loads((folder / 'report.json').read_text())
                if saved.get('kind') == 'study_development':
                    study_development.append(saved)
                    development_by_directory[str(folder.relative_to(directory))] = saved
            except (OSError, ValueError, AttributeError):
                pass
        # Enrich output thumbnails with the exact saved prompt and lineage.
        actions, hypotheses = {}, {}
        if 'render_plan.json' in names:
            try:
                render_plan = json.loads((folder / 'render_plan.json').read_text())
                candidate = frozen_cycle_view(directory, folder, render_plan)
                if candidate is not None:
                    cycle_view = candidate
                actions = {item['id']: item for item in render_plan.get('actions', [])}
                hypotheses = {item['id']: item for item in render_plan.get('hypotheses', [])}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                cycle_plan_unreadable = True
                errors.append(f'{folder.name}: {exc}')
        for name in sorted(names):
            if Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                continue
            sidecar = (folder / name).with_suffix('.json')
            try:
                if sidecar.name in names:
                    record = json.loads(sidecar.read_text())
                    if isinstance(record, dict):
                        action = actions.get(Path(name).stem, {})
                        hypothesis = hypotheses.get(action.get('hypothesis_id'), {})
                        image_records[str((folder / name).relative_to(directory))] = {
                            'record': record | ({'saved_action': action} if action else {}),
                            'label': hypothesis.get('title') or record.get('label')}
            except (OSError, ValueError):
                pass
        for index, action in enumerate(actions.values()):
            if f"{action['id']}.png" in names:
                continue
            failure = {}
            failure_name = f'failure-{index:03}.json'
            if failure_name in names:
                try:
                    failure = json.loads((folder / failure_name).read_text())
                except (OSError, ValueError):
                    pass
            generations_without_images.append({
                'action': action, 'directory': str(folder.relative_to(directory)),
                'label': hypotheses.get(action.get('hypothesis_id'), {}).get('title') or action['id'],
                'error': failure.get('error') if isinstance(failure, dict) else None,
            })
        if 'events.jsonl' in names and 'plan.json' not in names:
            try:
                for line in (folder / 'events.jsonl').read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue  # The last line may still be being appended.
                    if event.get('type') not in ('job_created', 'job_updated', 'attempt_created', 'attempt_updated', 'call_reserved'):
                        continue
                    payload = event.get('payload', {})
                    timeline.append({'id': event.get('event_id'), 'at': event.get('created_at', ''),
                        'label': event['type'].replace('_', ' '), 'status': payload.get('status'),
                        'detail': payload.get('error') or payload.get('path') or '',
                        'origin': str(folder.relative_to(directory)) + ' · execution ledger', 'record': event})
            except (OSError, ValueError, AttributeError):
                pass
        if 'plan.json' in names:
            try:
                plan = json.loads((folder / 'plan.json').read_text())
                if isinstance(plan, dict) and plan.get('kind') == 'autonomous_research':
                    from .dashboard import ResearchRuns
                    from fastapi import HTTPException
                    try:
                        report = ResearchRuns.report(folder)
                        verified_images = []
                        for asset in report.get('evidence_images', []):
                            path = Path(asset['path'])
                            if not path.is_absolute():
                                path = folder / path
                            relative = str(path.relative_to(directory)) if path.is_relative_to(directory) else None
                            file = by_path.get(relative)
                            rights = asset.get('rights', {})
                            permission = rights.get('image_display', rights)
                            if file and permission.get('status') in ('allowed', 'conditional'):
                                # Use the existing verified source route, including saved display rights.
                                research_id = ResearchRuns(runs.root).identifier(folder)
                                image_id = hashlib.sha256(json.dumps(asset, sort_keys=True).encode()).hexdigest()[:20]
                                verified_images.append(asset | {'url': f'/api/research/{research_id}/images/{image_id}'})
                        report['evidence_images'] = verified_images
                        research.append(report | {'directory': str(folder.relative_to(directory)), 'frozen_plan': plan})
                    except HTTPException as exc:
                        errors.append(str(exc.detail))
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f'{folder.name}: {exc}')
        for name in sorted(names):
            if name.startswith(('round-', 'step-')) and name.endswith('.json'):
                path = folder / name
                try:
                    data = json.loads(path.read_text())
                    action = data.get('action') or {}
                    label = action.get('action', name.removesuffix('.json')) if isinstance(action, dict) else name
                    timeline.append({'id': str(path.relative_to(directory)), 'label': label.replace('_', ' '),
                        'detail': (action.get('reason') or action.get('note') or action.get('query') or '') if isinstance(action, dict) else '',
                        'origin': data.get('origin', 'model research' if name.startswith('step-') else 'development record'),
                        'at': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                        'status': data.get('status'), 'record': data, 'file': by_path.get(str(path.relative_to(directory)))})
                except (OSError, ValueError, AttributeError):
                    continue
        if folder != directory and names & {'manifest.json', 'report.json', 'plan.json', 'render_plan.json'}:
            selected = next(name for name in ('manifest.json','report.json','render_plan.json','plan.json') if name in names)
            try:
                data = json.loads((folder / selected).read_text())
                if isinstance(data, dict):
                    stages.append({'directory': str(folder.relative_to(directory)), 'status': data.get('status', 'saved'),
                                   'kind': data.get('kind'), 'file': by_path.get(str((folder / selected).relative_to(directory)))})
            except (OSError, ValueError):
                pass
    # Technical reference bytes stay in Files; the image gallery foregrounds actual outputs.
    images = [file | image_records.get(file['name'], {}) for file in files if file['type'] == 'image' and not any(part in ('sources', 'evidence', 'references', 'inputs') for part in Path(file['name']).parts)]
    if cycle_view is None and not cycle_plan_unreadable and row['_data'].get('kind') == 'artistic_studio_run':
        try:
            cycle_view = queued_cycle_view(directory)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            errors.append(str(exc))
    if cycle_view is not None:
        cycle_view['development'] = development_by_directory.get(cycle_view['development_directory'], {})
    return {'files': files, 'images': images, 'generations_without_images': generations_without_images,
            'study_development': study_development,
            'development_cycle': cycle_view,
            'research': research, 'activity': sorted(timeline, key=lambda item: item.get('at', '')),
            'stages': stages, 'errors': errors, 'request': row['_data'].get('request')}
