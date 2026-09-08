"""Immutable, human-only datasets with decision-time inputs and grouped splits."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .evaluation import digest, file_hash, immutable_json
from .dataset_lineage import ADAPTER_VERSION, ReviewLineageResolver

KINDS = ('selector', 'judge_sft', 'director_sft', 'retrieval', 'action_preferences', 'continuations', 'practice_critique')


def timestamp(value):
    parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timestamps require explicit timezones')
    return parsed


class Evidence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str = Field(min_length=1)
    occurred_at: str
    episode_id: str = Field(min_length=1)
    sealed: bool = False
    data: dict[str, Any]


class LearningEvent(BaseModel):
    """Provenance envelope. Targets/reviews are separate from model input events."""
    model_config = ConfigDict(extra='forbid')
    event_id: str = Field(min_length=1)
    kind: Literal['selector', 'judge_sft', 'director_sft', 'retrieval', 'action_preferences', 'continuations', 'practice_critique']
    occurred_at: str
    decision_at: str
    project_id: str = Field(min_length=1)
    brief_id: str = Field(min_length=1)
    episode_id: str = Field(min_length=1)
    trajectory_id: str = Field(min_length=1)
    lineage_ids: list[str] = Field(default_factory=list)
    source_family_ids: list[str] = Field(default_factory=list)
    source_parent_ids: list[str] = Field(default_factory=list)
    source_image_hashes: list[str] = Field(default_factory=list)
    image_hashes: list[str] = Field(default_factory=list)
    sealed: bool = False
    label_origin: Literal['human', 'synthetic', 'derived', 'missing']
    reviewer: str | None = None
    label: Any = None
    source_event_ids: list[str] = Field(default_factory=list)
    input_events: list[Evidence] = Field(default_factory=list)
    target: dict[str, Any] = Field(default_factory=dict)
    presented_images: list[dict[str, Any]] = Field(default_factory=list)
    outcome: dict[str, Any] = Field(default_factory=dict)
    confounds: list[str] = Field(default_factory=list)
    # Explicit typed envelopes already supply their canonical IDs. Live reviews
    # additionally retain the resolver's exact ancestry proof outside inputs.
    lineage_provenance: dict[str, Any] = Field(default_factory=dict)


def _group_tokens(row, claim):
    tokens = [f'project:{row.project_id}', f'brief:{row.brief_id}', f'trajectory:{row.trajectory_id}']
    tokens += [f'lineage:{x}' for x in row.lineage_ids]
    tokens += [f'candidate-image:{x}' for x in row.image_hashes]
    if claim == 'source_family_transfer':
        tokens += [f'source-family:{x}' for x in row.source_family_ids]
        tokens += [f'source-parent:{x}' for x in row.source_parent_ids]
        tokens += [f'source-image:{x}' for x in row.source_image_hashes]
    return tokens


def grouped_assignments(rows, *, claim='project_transfer', seed=23):
    if claim not in ('project_transfer', 'source_family_transfer'):
        raise ValueError('Unknown transfer claim')
    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def union(a, b):
        parent[find(b)] = find(a)
    seen = {}
    for i, row in enumerate(rows):
        for token in _group_tokens(row, claim):
            if token in seen:
                union(i, seen[token])
            seen[token] = i
    members = {}
    for i, row in enumerate(rows):
        members.setdefault(find(i), []).append(row)
    groups = {digest(sorted(r.event_id for r in group)): group for group in members.values()}
    ordered = sorted(groups, key=lambda group: digest([seed, group]))
    sealed = {group for group, items in groups.items() if any(r.sealed for r in items)}
    quarantined = {group for group, items in groups.items() if any(
        r.lineage_provenance and r.lineage_provenance.get('status') != 'resolved' for r in items)} - sealed
    unsealed = [x for x in ordered if x not in sealed | quarantined]
    # Five independent groups is the minimum for a literal 3/1/1 partition.
    # Smaller datasets remain unassigned and cannot quietly become all-training.
    splits = {group: 'sealed' for group in sealed}
    splits.update({group: 'quarantined' for group in quarantined})
    if len(unsealed) < 5:
        splits.update({group: 'insufficient_groups' for group in unsealed})
    else:
        train_n = max(1, round(len(unsealed) * .6))
        dev_n = max(1, round(len(unsealed) * .2))
        train_n = min(train_n, len(unsealed) - dev_n - 1)
        for i, group in enumerate(unsealed):
            splits[group] = 'train' if i < train_n else 'development' if i < train_n + dev_n else 'test'
    return {r.event_id: {'split_group': group, 'split': splits[group]} for group, items in groups.items() for r in items}


def _forbidden_input_keys(value):
    forbidden = {'child_image', 'child_image_path', 'future_outcome', 'later_reviews', 'post_action_review', 'outcome', 'review_label', 'target', 'completion'}
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(_forbidden_input_keys(v) for v in value.values())
    if isinstance(value, list):
        return any(_forbidden_input_keys(v) for v in value)
    return False


def eligibility(row, *, cutoff, claim, assignment):
    reasons = []
    cutoff = timestamp(cutoff)
    decision = timestamp(row.decision_at)
    if timestamp(row.occurred_at) > cutoff or decision > cutoff:
        reasons.append('after_snapshot_cutoff')
    if row.label_origin != 'human':
        reasons.append('not_genuine_human_label')
    if row.label is None or not row.reviewer or not row.source_event_ids:
        reasons.append('missing_label_reviewer_or_provenance')
    if row.sealed or assignment['split'] == 'sealed':
        reasons.append('sealed_or_related_to_sealed')
    if assignment['split'] == 'quarantined':
        reasons.append('unresolved_or_related_review_lineage')
    if claim == 'source_family_transfer' and not row.source_family_ids:
        reasons.append('source_family_provenance_missing')
    if not row.input_events:
        reasons.append('missing_decision_time_inputs')
    for event in row.input_events:
        if timestamp(event.occurred_at) > decision:
            reasons.append('future_event_in_input')
        if event.sealed:
            reasons.append('sealed_event_in_input')
        if row.kind == 'director_sft' and _forbidden_input_keys(event.data):
            reasons.append('post_action_content_in_director_input')
    if row.kind in ('selector', 'judge_sft'):
        if row.label not in ('A', 'B', 'tie', 'unsure') or len(row.presented_images) != 2:
            reasons.append('invalid_pair_presentation_or_label')
        if len(row.presented_images) == 2:
            if any(not im.get('sha256') or not im.get('candidate_id') for im in row.presented_images):
                reasons.append('missing_presented_image_identity')
            elif len({im['candidate_id'] for im in row.presented_images}) != 2:
                reasons.append('duplicate_presented_candidate')
            if any(im.get('position', position) != position for im, position in zip(row.presented_images, ('A', 'B'))):
                reasons.append('presented_order_position_mismatch')
    if row.kind == 'director_sft':
        if not row.target.get('action') or not row.outcome.get('realized') or not row.outcome.get('review_event_ids'):
            reasons.append('missing_executable_action_real_outcome_or_support')
    if row.kind == 'retrieval':
        required = ('query', 'candidate_ids', 'chosen_ids', 'rejected_ids')
        if any(k not in row.target for k in required) or not row.outcome:
            reasons.append('incomplete_retrieval_candidate_set_or_outcome')
        elif set(row.target['chosen_ids']) & set(row.target['rejected_ids']) or set(row.target['chosen_ids']) | set(row.target['rejected_ids']) != set(row.target['candidate_ids']):
            reasons.append('invalid_retrieval_candidate_partition')
    if row.kind == 'action_preferences':
        if not row.target.get('same_prior_state_sha256') or not row.target.get('matched_execution') or len(row.target.get('actions', [])) != 2:
            reasons.append('unmatched_or_missing_action_alternatives')
        if not row.outcome.get('realized') or row.outcome.get('preference_scope') == 'whole_trajectory':
            reasons.append('missing_outcomes_or_noncausal_trajectory_preference')
        if 'uncontrolled_render_changes' in row.confounds:
            reasons.append('uncontrolled_render_changes')
    if row.kind in ('continuations', 'practice_critique'):
        if not all(row.target.get(k) for k in ('prior_work', 'competing_paths', 'budgets')) or not row.outcome.get('follow_up'):
            reasons.append('missing_continuation_context_or_follow_up')
    return sorted(set(reasons))


def build_snapshot(events, output_root, *, cutoff, claim='project_transfer', kind='judge_sft', seed=23):
    if kind not in KINDS:
        raise ValueError('Unknown dataset kind')
    timestamp(cutoff)
    rows = [LearningEvent.model_validate(event) for event in events]
    if len({row.event_id for row in rows}) != len(rows):
        raise ValueError('Source event IDs must be unique before task expansion')
    # Group all supplied events before filtering: a sealed parent must quarantine
    # related unsealed descendants, even when the sealed row has no label.
    assignments = grouped_assignments(rows, claim=claim, seed=seed)
    evidence_splits = {}
    for row in rows:
        for event_id in [row.event_id, *row.source_event_ids]:
            evidence_splits.setdefault(event_id, set()).add(assignments[row.event_id]['split'])
    admitted, rejected = [], []
    for row in rows:
        if row.kind != kind:
            continue
        reasons = eligibility(row, cutoff=cutoff, claim=claim, assignment=assignments[row.event_id])
        if any(evidence_splits.get(event.event_id, set()) & {'test', 'sealed'} for event in row.input_events):
            reasons.append('heldout_review_or_event_in_input')
        if any('quarantined' in evidence_splits.get(event.event_id, set()) for event in row.input_events):
            reasons.append('unresolved_review_lineage_event_in_input')
        if reasons:
            rejected.append({'event_id': row.event_id, 'reasons': reasons}
                            | ({'lineage_provenance': row.lineage_provenance} if row.lineage_provenance else {}))
            continue
        record = row.model_dump(mode='json') | assignments[row.event_id]
        record['model_input'] = [e.data for e in row.input_events]
        admitted.append(record)
    admitted.sort(key=lambda row: row['event_id'])
    rejected.sort(key=lambda row: row['event_id'])
    provenance = {'schema_version': 2, 'kind': kind, 'human_only': True, 'cutoff': cutoff, 'claim': claim,
                  'exporter_identity': {'review_adapter': ADAPTER_VERSION, 'datasets_sha256': file_hash(Path(__file__)),
                                        'lineage_resolver_sha256': file_hash(Path(__file__).with_name('dataset_lineage.py'))},
                  'split_seed': seed, 'source_event_manifest_sha256': digest([r.model_dump(mode='json') for r in rows]),
                  'source_event_count': len(rows), 'eligible_count': len(admitted), 'rejected_count': len(rejected),
                  'split_counts': dict(Counter(row['split'] for row in admitted)),
                  'independent_groups': len({row['split_group'] for row in admitted}),
                  'split_policy': 'Connected project/brief/trajectory/lineage/image groups; source families/parents/duplicates additionally connected for source transfer. At least five unsealed groups before 60/20/20.',
                  'inputs_policy': 'Only input_events available at decision_at; labels/outcomes remain separate targets; test rows never loaded by trainers.',
                  'review_lineage_policy': 'Live reviews require registered original-brief ancestry or an explicit collection-hash-bound curator declaration. Unknown/conflicting ancestry and connected rows are quarantined. Exact brief wording cannot establish semantic independence; differently worded versions may need curator grouping. Candidate IDs are scoped to their collection.',
                  'rejection_reasons': dict(Counter(reason for row in rejected for reason in row['reasons']))}
    snapshot_id = digest({'provenance': provenance, 'rows': admitted, 'rejections': rejected})[:24]
    root = Path(output_root) / snapshot_id
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for split in ('train', 'development', 'test', 'insufficient_groups'):
        data = ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in admitted if row['split'] == split)
        path = root / f'{split}.jsonl'
        if path.exists() and path.read_text() != data:
            raise ValueError('Immutable dataset bytes changed')
        if not path.exists():
            with path.open('x') as stream:
                stream.write(data)
        files[path.name] = file_hash(path)
    immutable_json(root / 'rejections.json', rejected)
    files['rejections.json'] = file_hash(root / 'rejections.json')
    manifest = provenance | {'snapshot_id': snapshot_id, 'files': files}
    immutable_json(root / 'manifest.json', manifest)
    card = f"# Dataset {snapshot_id}\n\nKind: {kind}. Claim: {claim}. Human-only. Cutoff: {cutoff}.\n\n{len(admitted)} eligible rows from {provenance['independent_groups']} independent groups; {len(rejected)} rejected. Split counts: {json.dumps(provenance['split_counts'], sort_keys=True)}.\n\n{provenance['split_policy']}\n\n{provenance['inputs_policy']}\n\n{provenance['review_lineage_policy']}\n\nExporter identity: {json.dumps(provenance['exporter_identity'], sort_keys=True)}.\n\nRejections: {json.dumps(provenance['rejection_reasons'], sort_keys=True)}. Small datasets remain insufficient_groups; these rows are not training data. Two questions, reversed image orders and seeds do not create independent human judgments. Common sources may be shared for project transfer only. No trained advantage is established. File hashes and source event IDs preserve provenance. Images remain at recorded artifact paths and require separately preserved bytes.\n"
    path = root / 'data_card.md'
    if path.exists() and path.read_text() != card:
        raise ValueError('Immutable data card changed')
    if not path.exists():
        with path.open('x') as stream:
            stream.write(card)
    return {'snapshot_path': str(root.resolve()), **manifest}


def review_events(artifact_root, *, kind='judge_sft', lineage_declarations=None):
    """Read actual ReviewStore votes only. Advancement/model observations are not votes."""
    if kind not in ('selector', 'judge_sft'):
        return []  # These require linked action/continuation provenance, not a guess.
    root = Path(artifact_root).resolve()
    path = root / 'reviews.sqlite3'
    if not path.exists():
        return []
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.execute('BEGIN')
        db.create_function('lineage_sha256', 1, lambda value: hashlib.sha256(value.encode()).hexdigest())
        votes = [json.loads(row[0]) for row in db.execute('SELECT record FROM review_votes ORDER BY sequence')]
        # Do not inspect unrelated collection definitions, especially sealed
        # studies. No votes means no candidate, brief or lineage reads at all.
        collection_ids = {vote['collection_id'] for vote in votes if vote.get('type') == 'human_pairwise_review'}
        # The original text is withheld until the ancestry/seal check passes.
        records = {cid: db.execute("SELECT json_remove(specification,'$.brief'),lineage_sha256(specification) FROM review_collections WHERE id=?", (cid,)).fetchone()
                   for cid in sorted(collection_ids)}
        specs = {cid: json.loads(record[0]) for cid, record in records.items()}
    if not specs:
        return []
    with ReviewLineageResolver(root, declarations=lineage_declarations) as resolver:
        lineage = {cid: resolver.resolve(collection, collection_sha256=records[cid][1]) for cid, collection in specs.items()}
    with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
        db.create_function('lineage_sha256', 1, lambda value: hashlib.sha256(value.encode()).hexdigest())
        briefs = {}
        for cid, ancestry in lineage.items():
            if ancestry['status'] != 'resolved' or ancestry['sealed']:
                continue
            row = db.execute("SELECT json_extract(specification,'$.brief') FROM review_collections WHERE id=? AND lineage_sha256(specification)=?", (cid, records[cid][1])).fetchone()
            if row is None:
                raise ValueError('Review collection changed during lineage resolution')
            briefs[cid] = row[0]
    events = []
    for vote in votes:
        if vote.get('type') != 'human_pairwise_review':
            continue
        collection = specs[vote['collection_id']]
        ancestry = lineage[collection['id']]
        candidates = {c['id']: c for c in collection['candidates']}
        images = []
        for position in ('A', 'B'):
            presented = vote['candidate_order'][position]
            candidate = candidates.get(presented['id'])
            if candidate is None or presented.get('sha256') != candidate['sha256']:
                raise ValueError('Recorded review presentation differs from the registered candidate identity')
            image_path = (root / candidate['path']).resolve()
            if not image_path.is_relative_to(root):
                raise ValueError('A reviewed image escapes the artifact root')
            if ancestry['status'] == 'resolved' and not ancestry['sealed'] and file_hash(image_path) != candidate['sha256']:
                raise ValueError('A reviewed image changed; refuse dataset export')
            images.append({'position': position, 'candidate_id': candidate['id'], 'path': str(image_path), 'sha256': candidate['sha256']})
        for dimension in ('finished_quality', 'development_potential'):
            event_id = f"{vote['submission_id']}:{dimension}"
            events.append({'event_id': event_id, 'kind': kind, 'occurred_at': vote['created_at'], 'decision_at': vote['created_at'],
                           'project_id': ancestry['project_id'], 'brief_id': ancestry['brief_id'], 'episode_id': collection['id'],
                           'trajectory_id': ancestry['trajectory_id'], 'lineage_ids': ancestry['lineage_ids'],
                           'lineage_provenance': ancestry, 'sealed': ancestry['sealed'],
                           'image_hashes': [im['sha256'] for im in images], 'label_origin': 'human', 'reviewer': vote['reviewer'],
                           'label': vote[dimension]['choice'], 'source_event_ids': [vote['submission_id']],
                           'presented_images': images,
                           'input_events': [{'event_id': f"presentation:{vote['assignment_id']}", 'occurred_at': vote['created_at'],
                                             'episode_id': collection['id'], 'data': {'brief': briefs[collection['id']], 'dimension': dimension,
                                             'reviewer': vote['reviewer'], 'candidate_order': images}}] if collection['id'] in briefs else [],
                           'target': {'dimension': dimension}, 'outcome': {'review_event_ids': [vote['submission_id']]}})
    return events


def load_snapshot(snapshot, *, splits=('train', 'development')):
    """Development API deliberately has no switch to expose test labels."""
    if any(split not in ('train', 'development', 'insufficient_groups') for split in splits):
        raise ValueError('Test labels are sealed from training, profiles, and checkpoint selection')
    root = Path(snapshot)
    manifest = json.loads((root / 'manifest.json').read_text())
    result = {}
    for split in splits:
        path = root / f'{split}.jsonl'
        if file_hash(path) != manifest['files'][path.name]:
            raise ValueError('Dataset hash mismatch')
        result[split] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in result[split]:
            if row['split'] != split or row['sealed'] or row['label_origin'] != 'human':
                raise ValueError('Invalid or sealed training row')
    groups = [{row['split_group'] for row in rows} for rows in result.values()]
    for i, a in enumerate(groups):
        if any(a & b for b in groups[i+1:]):
            raise ValueError('Split group leakage')
    return manifest, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument('--artifact-root', type=Path)
    source.add_argument('--events', type=Path)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--cutoff', required=True)
    build.add_argument('--claim', choices=('project_transfer', 'source_family_transfer'), default='project_transfer')
    build.add_argument('--kind', choices=KINDS, default='judge_sft')
    build.add_argument('--seed', type=int, default=23)
    build.add_argument('--lineage', type=Path, help='Explicit collection-hash-bound curator lineage declarations for live reviews')
    args = parser.parse_args(argv)
    if args.lineage and not args.artifact_root:
        parser.error('--lineage applies only to --artifact-root live review exports')
    declarations = json.loads(args.lineage.read_text()) if args.lineage else None
    events = review_events(args.artifact_root, kind=args.kind, lineage_declarations=declarations) if args.artifact_root else [json.loads(line) for line in args.events.read_text().splitlines() if line.strip()]
    result = build_snapshot(events, args.output, cutoff=args.cutoff, claim=args.claim, kind=args.kind, seed=args.seed)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
