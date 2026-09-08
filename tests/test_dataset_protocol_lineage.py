"""Public protocol/plan fixtures only; no real heldout texts or outcomes."""
import json

import pytest

from kusama.dataset_lineage import ReviewLineageResolver
from kusama.datasets import review_events
from kusama.evaluation import digest, file_hash
from kusama.ledger import Ledger
from test_dataset_lineage import assignments, collection, png


def protocol(root):
    folder = root / 'public-protocol'
    folder.mkdir()
    manifest = {'schema_version': 1, 'protocol_id': 'fixture-protocol', 'briefs': [
        {'brief_id': 'unpredictable-a', 'partition': 'development', 'brief': 'First development fixture.'},
        {'brief_id': 'unpredictable-b', 'partition': 'development', 'brief': 'Different development fixture.'},
        {'brief_id': 'opaque-sealed-slot', 'partition': 'sealed_test', 'brief_sha256': 'a' * 64}]}
    (folder / 'manifest.json').write_text(json.dumps(manifest))
    return folder, {'protocol_manifest_sha256': file_hash(folder / 'manifest.json'), 'policy': 'fixture-only'}


def protocol_episode(root, folder, runtime, bid, colors, *, sealed=False):
    directory = folder / 'episodes' / bid
    parent_dir = directory / 'parent'
    parent_paths = [png(parent_dir / f'image-{i}.png', color) for i, color in enumerate(colors[:2] if sealed else colors[:1])]
    with Ledger(root / 'jobs.sqlite3') as ledger:
        ledger.create_job('opaque-parent-' + bid, {'runtime': runtime, 'request': {'parent': None, 'instruction': 'Fixture only.'}}, 2, parent_dir)
        for path in parent_paths:
            ledger.register_artifact('opaque-parent-' + bid, path.name, kind='image')
        if sealed:
            # No runtime_identity.json, frozen.json, private brief or per-episode
            # text exists. Public partition admission must finish first.
            return parent_paths
        (directory / 'runtime_identity.json').write_text(json.dumps(runtime))
        frozen = {'schema_version': 1, 'protocol_id': 'fixture-protocol', 'brief_id': bid,
                  'partition': 'development', 'parent': {'path': str(parent_paths[0]), 'sha256': file_hash(parent_paths[0])},
                  'action_requests': {'fixture': {'instruction': 'Synthetic action text.'}}}
        frozen['request_sha256'] = digest(frozen)
        (directory / 'frozen.json').write_text(json.dumps(frozen))
        edit_dir = directory / 'edits'
        paths = [png(edit_dir / f'image-{i}.png', color) for i, color in enumerate(colors[1:])]
        ledger.create_job('opaque-edits-' + bid, {'runtime': runtime, 'frozen_request_sha256': frozen['request_sha256']}, 6, edit_dir)
        for path in paths:
            ledger.register_artifact('opaque-edits-' + bid, path.name, kind='image')
    return parent_paths + paths


def test_protocol_parent_and_disjoint_edits_share_one_public_brief_without_name_inference(tmp_path):
    folder, runtime = protocol(tmp_path)
    a = protocol_episode(tmp_path, folder, runtime, 'unpredictable-a', ['red', 'green', 'blue', 'yellow'])
    b = protocol_episode(tmp_path, folder, runtime, 'unpredictable-b', ['purple', 'pink'])
    collection(tmp_path, 'opaque-collection-1', a[:2], shared_ids=True)
    collection(tmp_path, 'opaque-collection-2', a[2:], shared_ids=True)
    collection(tmp_path, 'opaque-collection-3', b, shared_ids=True)
    events = review_events(tmp_path)
    assert all(e['lineage_provenance']['status'] == 'resolved' for e in events)
    groups = assignments(events)
    assert groups[events[0]['event_id']]['split_group'] == groups[events[2]['event_id']]['split_group']
    assert groups[events[0]['event_id']]['split_group'] != groups[events[4]['event_id']]['split_group']
    assert not set(events[0]['image_hashes']) & set(events[2]['image_hashes'])
    assert events[0]['project_id'] == events[2]['project_id']
    assert any(p['type'] == 'frozen_protocol_request' for p in events[2]['lineage_provenance']['proofs'])


def test_public_sealed_partition_stops_before_episode_files_or_image_bytes(tmp_path, monkeypatch):
    folder, runtime = protocol(tmp_path)
    paths = protocol_episode(tmp_path, folder, runtime, 'opaque-sealed-slot', ['red', 'blue'], sealed=True)
    spec, _ = collection(tmp_path, 'opaque-review', paths)
    import kusama.dataset_lineage as module
    actual_hash = module.file_hash
    hashed = []

    def public_only(path):
        assert path == folder / 'manifest.json', 'Sealed episode files and images must remain unread'
        hashed.append(path)
        return actual_hash(path)

    monkeypatch.setattr(module, 'file_hash', public_only)
    with ReviewLineageResolver(tmp_path) as resolver:
        result = resolver.resolve(spec)
    assert result['sealed'] and result['status'] == 'unresolved'
    assert hashed
    assert not any(p['type'] in ('frozen_protocol_runtime', 'original_brief_digest') for p in result['proofs'])
    events = review_events(tmp_path)
    assert all(e['sealed'] and not e['input_events'] for e in events)


@pytest.mark.parametrize('changed_file', ['manifest.json', 'episodes/unpredictable-a/runtime_identity.json', 'episodes/unpredictable-a/frozen.json'])
def test_protocol_authority_runtime_or_request_tamper_quarantines(tmp_path, changed_file):
    folder, runtime = protocol(tmp_path)
    paths = protocol_episode(tmp_path, folder, runtime, 'unpredictable-a', ['red', 'blue'])
    spec, _ = collection(tmp_path, 'opaque-review', paths, vote=False)
    path = folder / changed_file
    value = json.loads(path.read_text())
    value['tampered'] = True
    path.write_text(json.dumps(value))
    with ReviewLineageResolver(tmp_path) as resolver:
        result = resolver.resolve(spec)
    assert result['status'] != 'resolved'


def comparison_plan(root, *, partition='development'):
    folder = root / 'arbitrary-comparison-folder'
    folder.mkdir()
    modes = ['direct', 'grounded', 'critique', 'search']
    plan = {'schema_version': 1, 'comparison_id': 'fixture-comparison', 'partition': partition,
            'brief': {'brief_id': 'fixture-common-brief', 'partition': partition, 'brief': 'Common synthetic brief.'},
            'modes': modes}
    (folder / 'plan.json').write_text(json.dumps(plan))
    return folder, modes


def test_all_matched_modes_group_by_hash_bound_common_plan_with_opaque_collections(tmp_path):
    folder, modes = comparison_plan(tmp_path)
    for i, mode in enumerate(modes):
        directory = folder / mode / 'rendering'
        paths = [png(directory / f'{j}.png', (i * 50, 25 + j * 70, 90)) for j in range(2)]
        with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
            ledger.create_job('opaque-render-' + str(i), {'mode': mode, 'stage': 'rendering', 'plan_sha256': file_hash(folder / 'plan.json')}, 4, directory)
            for path in paths:
                ledger.register_artifact('opaque-render-' + str(i), path.name, kind='image')
        collection(tmp_path, 'workflow-' + digest(i)[:16], paths, shared_ids=True)
    events = review_events(tmp_path)
    assert all(e['lineage_provenance']['status'] == 'resolved' for e in events)
    assert len({e['project_id'] for e in events}) == 1
    assert len({row['split_group'] for row in assignments(events).values()}) == 1
    assert len({sha for e in events for sha in e['image_hashes']}) == 8


@pytest.mark.parametrize('variant', ['sealed', 'changed-plan', 'wrong-mode'])
def test_comparison_seal_plan_hash_and_registered_mode_membership_are_enforced(tmp_path, variant):
    folder, _ = comparison_plan(tmp_path, partition='sealed_test' if variant == 'sealed' else 'development')
    directory = folder / 'direct' / 'rendering'
    paths = [png(directory / f'{i}.png', color) for i, color in enumerate(['red', 'blue'])]
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('opaque-render', {'mode': 'wrong' if variant == 'wrong-mode' else 'direct', 'stage': 'rendering',
                          'plan_sha256': file_hash(folder / 'plan.json')}, 4, directory)
        for path in paths:
            ledger.register_artifact('opaque-render', path.name, kind='image')
    spec, _ = collection(tmp_path, 'opaque-review', paths, vote=False)
    if variant == 'changed-plan':
        (folder / 'plan.json').write_text('{}')
    with ReviewLineageResolver(tmp_path) as resolver:
        result = resolver.resolve(spec)
    assert result['status'] != 'resolved'
    assert result['sealed'] == (variant == 'sealed')
