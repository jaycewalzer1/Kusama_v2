"""Frozen E0 planning and evidence accounting. Planning never dispatches inference."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONDITIONS = ('inquiry_relationship', 'ordinary_critique', 'exploratory')
SEEDS = (23, 41)
MISSING_RULE = {
    'technical_failure_identical_retries': 1,
    'uncertain_execution': 'reconcile_before_retry',
    'valid_output': 'never_replace_for_quality',
    'unresolved_pair': 'exclude_from_completed_pair_analysis',
    'incomplete_pool': 'inspect_separately_exclude_from_standard_pool_selections',
    'accounting': 'count_every_primary_and_retry_attempt',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def immutable_json(path, value):
    """Create once; an identical repeat is idempotent and changed bytes fail."""
    path = Path(path)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_text() != data:
            raise ValueError(f'Immutable artifact already differs: {path}')
    return path


class E0Config(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    protocol_id: str = Field(pattern=r'^[A-Za-z0-9_-]+$')
    development_briefs: list[str] = Field(min_length=4, max_length=4)
    reviewer_slots: list[str] = Field(min_length=3, max_length=3)
    shared_edit_seeds: tuple[int, int] = SEEDS
    holdout_count: Literal[8] = 8
    holdout_authorship: Literal['assistant_provisional'] = 'assistant_provisional'

    @model_validator(mode='after')
    def distinct(self):
        if len(set(self.reviewer_slots)) != 3 or len(set(self.shared_edit_seeds)) != 2:
            raise ValueError('Reviewer slots and edit seeds must be distinct')
        return self


def prepare_protocol(output_dir: Path, config_path: Path = Path('configs/e0.yaml'), *, holdout_file=None):
    config = E0Config.model_validate(json.loads(Path(config_path).read_text()))
    root = Path(output_dir)
    source = Path(holdout_file) if holdout_file is not None else root / 'sealed' / 'briefs.json'
    if not source.is_file():
        raise ValueError('Supply an external holdout file or pre-existing sealed/briefs.json; no holdout text is embedded in source code')
    sealed = json.loads(source.read_text())
    if not isinstance(sealed, dict) or sealed.get('protocol_id') != config.protocol_id or sealed.get('schema_version') != 1:
        raise ValueError('Sealed manifest protocol identity is incompatible')
    holdouts = sealed.get('briefs', [])
    expected = {f'e0-h{i+1:02}' for i in range(8)}
    if len(holdouts) != 8 or {row.get('brief_id') for row in holdouts} != expected:
        raise ValueError('Supply exactly eight distinct heldout brief IDs e0-h01 through e0-h08')
    if any(row.get('partition') != 'sealed_test' or not isinstance(row.get('brief'), str) or not row['brief'].strip()
           or len(row['brief']) > 4000 or row.get('authorship') != config.holdout_authorship for row in holdouts):
        raise ValueError('Sealed briefs require nonempty bounded text and explicit provisional authorship')
    if len({row['brief'].strip() for row in holdouts}) != 8:
        raise ValueError('Heldout briefs must be distinct')
    immutable_json(root / 'sealed' / 'briefs.json', sealed)
    briefs = [{'brief_id': f'e0-d{i+1:02}', 'brief': brief, 'partition': 'development',
               'authorship': 'user_supplied_development_example'} for i, brief in enumerate(config.development_briefs)]
    briefs += [{'brief_id': x['brief_id'], 'partition': 'sealed_test', 'brief_sha256': digest(x)} for x in holdouts]
    manifest = {'schema_version': 1, 'protocol_id': config.protocol_id, 'config': config.model_dump(mode='json'),
                'briefs': briefs, 'sealed_manifest_sha256': digest(sealed), 'conditions': CONDITIONS,
                'missing_output_rule': MISSING_RULE,
                'planned': {'parents': 12, 'edits': 72, 'images': 84, 'parent_edit_judgments': 216, 'pool_selections': 36},
                'status': 'planned_not_executable',
                'prerequisites': ['freeze reference and prior-memory snapshots', 'generate and hash one parent per brief',
                                  'freeze three validated action requests and shared seed pair before edits',
                                  'complete development then freeze technical settings before sealed dispatch',
                                  'reserve attempts, reasoning tokens, GPU time and wall time using measured throughput'],
                'claims': 'Provisional assistant-authored holdouts; no generated images or collected judgments implied.'}
    immutable_json(root / 'manifest.json', manifest)
    assignments = planned_assignments(manifest)
    target = root / 'review_assignments.jsonl'
    content = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in assignments)
    if target.exists() and target.read_text() != content:
        raise ValueError('Immutable review assignments differ')
    if not target.exists():
        with target.open('x') as stream:
            stream.write(content)
    return manifest


def planned_assignments(manifest):
    """Reproduce the predeclared schedule from public protocol fields only."""
    config = E0Config.model_validate(manifest['config'])
    if config.protocol_id != manifest['protocol_id']:
        raise ValueError('Protocol and configuration identity differ')
    assignments = []
    briefs = manifest['briefs']
    for brief in briefs:
        bid = brief['brief_id']
        children = [f'{bid}-{condition}-s{seed}' for condition in CONDITIONS for seed in config.shared_edit_seeds]
        for reviewer in config.reviewer_slots:
            for child in children:
                identity = [config.protocol_id, bid, reviewer, child]
                order = [f'{bid}-parent', child]
                if int(digest(identity), 16) % 2:
                    order.reverse()
                assignments.append({'assignment_id': digest(identity), 'protocol_id': config.protocol_id,
                                    'brief_id': bid, 'reviewer_slot': reviewer, 'kind': 'parent_edit',
                                    'candidate_order': dict(zip(('A', 'B'), order)),
                                    'dimensions': ['finished_quality', 'development_potential', 'readiness'],
                                    'status': 'awaiting_outputs_and_human_review'})
            ordered = sorted(children, key=lambda child: digest([config.protocol_id, reviewer, child, 'pool']))
            assignments.append({'assignment_id': digest([config.protocol_id, bid, reviewer, 'pool']),
                                'protocol_id': config.protocol_id, 'brief_id': bid, 'reviewer_slot': reviewer,
                                'kind': 'six_candidate_or_none', 'candidate_ids': ordered,
                                'allowed_none': True, 'status': 'awaiting_complete_pool_and_human_review'})
    return assignments


def verified_assignments(root):
    """Reject a changed, missing or incomplete review plan before execution."""
    root = Path(root)
    manifest = load_protocol(root)
    actual = [json.loads(line) for line in (root / 'review_assignments.jsonl').read_text().splitlines()]
    if actual != planned_assignments(manifest):
        raise ValueError('Frozen review assignments differ from the predeclared protocol')
    return actual


def load_protocol(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    sealed = json.loads((root / 'sealed' / 'briefs.json').read_text())
    if digest(sealed) != manifest['sealed_manifest_sha256']:
        raise ValueError('Sealed brief manifest changed')
    return manifest


def episode_brief(root, brief_id, *, purpose='development', episode_id=None):
    """The only accessor for sealed text, scoped to its own execution episode."""
    manifest = load_protocol(root)
    brief = next((row for row in manifest['briefs'] if row['brief_id'] == brief_id), None)
    if brief is None:
        raise ValueError('Unknown brief')
    if brief['partition'] == 'sealed_test':
        if purpose != 'heldout_execution' or episode_id != brief_id:
            raise ValueError('Sealed brief is only available to its own heldout_execution')
        sealed = json.loads((Path(root) / 'sealed' / 'briefs.json').read_text())
        return next(row for row in sealed['briefs'] if row['brief_id'] == brief_id)
    return brief


def freeze_episode(root, brief_id, *, parent, action_requests, render_profile,
                   reference_snapshot_sha256, memory_snapshot_sha256, development_freeze_sha256=None):
    manifest = load_protocol(root)
    brief = next((x for x in manifest['briefs'] if x['brief_id'] == brief_id), None)
    if brief is None:
        raise ValueError('Unknown brief')
    if brief['partition'] == 'sealed_test' and not development_freeze_sha256:
        raise ValueError('Heldout episodes require an explicit development configuration freeze')
    if set(action_requests) != set(CONDITIONS) or any(not v for v in action_requests.values()):
        raise ValueError('Freeze all three nonempty alternative action requests together')
    if not render_profile or not reference_snapshot_sha256 or not memory_snapshot_sha256:
        raise ValueError('Render profile and reference/memory snapshot hashes are required')
    parent_path = Path(parent['path']).resolve()
    if file_hash(parent_path) != parent['sha256']:
        raise ValueError('Parent bytes do not match frozen hash')
    from PIL import Image
    with Image.open(parent_path) as img:
        img.verify()
    value = {'schema_version': 1, 'protocol_id': manifest['protocol_id'], 'brief_id': brief_id,
             'partition': brief['partition'], 'parent': {'path': str(parent_path), 'sha256': parent['sha256']},
             'action_requests': action_requests, 'render_profile': render_profile,
             'edit_seeds': manifest['config']['shared_edit_seeds'],
             'reference_snapshot_sha256': reference_snapshot_sha256, 'memory_snapshot_sha256': memory_snapshot_sha256,
             'development_freeze_sha256': development_freeze_sha256, 'missing_output_rule': MISSING_RULE}
    value['request_sha256'] = digest(value)
    immutable_json(Path(root) / 'episodes' / brief_id / 'frozen.json', value)
    return value


def record_attempt(root, brief_id, condition, seed, *, attempt, status, reconciled=False,
                   output_path=None, execution_id, request_sha256):
    """Account for independently authorized execution; never triggers a retry."""
    episode_dir = Path(root) / 'episodes' / brief_id
    frozen = json.loads((episode_dir / 'frozen.json').read_text())
    if request_sha256 != frozen['request_sha256'] or condition not in CONDITIONS or seed not in frozen['edit_seeds']:
        raise ValueError('Attempt differs from frozen episode request')
    if attempt not in (1, 2) or status not in ('succeeded', 'technical_failure', 'uncertain'):
        raise ValueError('Invalid attempt/status')
    target = episode_dir / f'{condition}-s{seed}-attempt{attempt}.json'
    if attempt == 2:
        first = json.loads((episode_dir / f'{condition}-s{seed}-attempt1.json').read_text())
        if first['status'] == 'succeeded' or not reconciled:
            raise ValueError('Only a reconciled failure permits one identical retry')
        if first['execution_id'] == execution_id:
            raise ValueError('Retry must identify a separately counted execution')
    value = {'condition': condition, 'seed': seed, 'attempt': attempt, 'status': status,
             'reconciled': reconciled, 'execution_id': execution_id, 'request_sha256': request_sha256}
    if status == 'succeeded':
        if output_path is None:
            raise ValueError('A successful attempt requires actual saved image bytes')
        from PIL import Image
        path = Path(output_path).resolve()
        with Image.open(path) as img:
            img.verify()
        value['output'] = {'path': str(path), 'sha256': file_hash(path)}
    immutable_json(target, value)
    return value


def protocol_status(root):
    manifest = load_protocol(root)
    parents, edits, attempts, complete_briefs = 0, 0, 0, []
    failures = Counter()
    for brief in manifest['briefs']:
        folder = Path(root) / 'episodes' / brief['brief_id']
        if not (folder / 'frozen.json').exists():
            continue
        frozen = json.loads((folder / 'frozen.json').read_text())
        parent_valid = Path(frozen['parent']['path']).is_file() and file_hash(frozen['parent']['path']) == frozen['parent']['sha256']
        parents += int(parent_valid)
        completed = set()
        for path in folder.glob('*-attempt*.json'):
            record = json.loads(path.read_text())
            attempts += 1
            out = record.get('output')
            if record['status'] == 'succeeded' and out and Path(out['path']).is_file() and file_hash(out['path']) == out['sha256']:
                completed.add((record['condition'], record['seed']))
            else:
                failures[record['status']] += 1
        edits += len(completed)
        if parent_valid and len(completed) == 6:
            complete_briefs.append(brief['brief_id'])
    return {'protocol_id': manifest['protocol_id'], 'generation_status': 'complete' if parents == 12 and edits == 72 else 'incomplete',
            'human_evaluation_status': 'uncollected_in_protocol_store', 'verified_parents': parents, 'verified_edits': edits,
            'recorded_edit_attempts': attempts, 'attempt_failures': dict(failures), 'complete_pool_briefs': complete_briefs,
            'missing_images': 84-parents-edits, 'planned_parent_edit_assignments': 216, 'planned_pool_assignments': 36,
            'genuine_judgments_in_protocol_store': 0, 'genuine_pool_selections_in_protocol_store': 0,
            'note': 'Review assignments are plans. Submit genuine judgments through the review ledger; this protocol planner does not synthesize labels.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--config', type=Path, default=Path('configs/e0.yaml'))
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--holdout-file', type=Path)
    status = sub.add_parser('status')
    status.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_protocol(args.output, args.config, holdout_file=args.holdout_file) if args.command == 'prepare' else protocol_status(args.protocol)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
