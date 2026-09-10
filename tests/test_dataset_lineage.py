"""Synthetic images, execution records and human-shaped fixture votes only."""
import hashlib
import json
import shutil
import sqlite3

from PIL import Image
import pytest

from kusama.dataset_lineage import ADAPTER_VERSION, ReviewLineageResolver
from kusama.datasets import LearningEvent, build_snapshot, grouped_assignments, review_events
from kusama.evidence import digest, file_hash
from kusama.ledger import Ledger
from kusama.reviews import ReviewStore

CUTOFF = '2099-01-01T00:00:00+00:00'


def png(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (32, 32), color).save(path)
    return path


def source_run(root, job_id, brief, colors):
    directory = root / 'runs' / job_id
    directory.mkdir(parents=True)
    (directory / 'input.json').write_text(json.dumps({'brief': brief}))
    generation = directory / 'generation'
    generation.mkdir()
    with Ledger(root / 'jobs.sqlite3') as ledger:
        ledger.create_job(job_id, {'kind': 'connection_inspiration', 'brief': brief}, 2, directory)
        ledger.create_job(job_id + '_render', {'kind': 'connection_generation', 'inspiration_job': job_id,
                          'source_input_sha256': file_hash(directory / 'input.json')}, 2, generation)
        paths = [png(generation / f'image-{i}.png', color) for i, color in enumerate(colors)]
        for path in paths:
            ledger.register_artifact(job_id + '_render', path.name, kind='result')
    return paths


def branch(root, project_id, parent, colors):
    directory = root / 'studio_projects' / project_id
    directory.mkdir(parents=True)
    shutil.copyfile(parent, directory / 'parent.png')
    parent_hash = file_hash(parent)
    artifacts = {'root': {'id': 'root', 'origin': 'imported', 'parent_artifact_id': None,
                         'path': str(directory / 'parent.png'), 'sha256': parent_hash, 'source_path': str(parent)}}
    paths = []
    with Ledger(root / 'jobs.sqlite3') as ledger:
        episode = project_id + '_episode'
        ledger.create_job(episode, {'kind': 'studio_episode', 'project_id': project_id}, 4, directory)
        for i, color in enumerate(colors):
            child_id = project_id + f'_action_{i}'
            child_dir = root / child_id
            child_dir.mkdir()
            shutil.copyfile(parent, child_dir / 'parent.png')
            ledger.create_job(child_id, {'kind': 'controlled_edit',
                'plan': {'parent': 'parent.png', 'parent_sha256': parent_hash}}, 2, child_dir, parent_id=episode)
            child = png(child_dir / 'result.png', color)
            ledger.register_artifact(child_id, child.name, kind='result')
            artifacts[f'edit-{i}'] = {'id': f'edit-{i}', 'origin': 'generated', 'parent_artifact_id': 'root',
                                     'path': str(child), 'sha256': file_hash(child)}
            paths.append(child)
        # Only the minimal stable, authoritative Studio ancestry records are
        # fixtures here; no model outputs or research execution are claimed.
        ledger.connection.execute('CREATE TABLE IF NOT EXISTS studio_projects(id TEXT PRIMARY KEY,data TEXT NOT NULL)')
        ledger.connection.execute('INSERT INTO studio_projects VALUES (?,?)', (project_id, json.dumps({
            'id': project_id, 'creation_spec': {'parent_sha256': parent_hash}, 'artifacts': artifacts})))
    return paths


def collection(root, cid, paths, *, project_id=None, shared_ids=False, vote=True):
    store = ReviewStore(root)
    candidates = [{'id': f'image-{i:03}' if shared_ids else f'{cid}-image-{i}', 'path': str(path), 'label': 'Fixture image',
                   **({'studio': {'project_id': project_id, 'episode_id': 'revision', 'artifact_id': f'edit-{i}'}} if project_id else {})}
                  for i, path in enumerate(paths)]
    store.register_collection(cid, title='Synthetic review fixture', brief='This collection text is not lineage evidence.', candidates=candidates)
    if vote:
        assignment = store.comparison(cid, 'pair-001', 'fixture-reviewer')
        store.submit_vote(cid, 'pair-001', {'reviewer': 'fixture-reviewer', 'assignment_id': assignment['assignment_id'],
            'submission_id': 'fixture-' + cid, 'finished_quality': {'choice': 'A'}, 'development_potential': {'choice': 'B'}})
    with sqlite3.connect(root / 'reviews.sqlite3') as db:
        raw = db.execute('SELECT specification FROM review_collections WHERE id=?', (cid,)).fetchone()[0]
    return json.loads(raw), hashlib.sha256(raw.encode()).hexdigest()


def assignments(events):
    return grouped_assignments([LearningEvent.model_validate(event) for event in events])


def test_disjoint_reviewed_branches_and_descendant_share_original_brief(tmp_path):
    initial = source_run(tmp_path, 'original', 'One original fixture brief.', ['red', 'green', 'blue', 'white'])
    first = branch(tmp_path, 'branch-a', initial[0], ['yellow', 'orange'])
    second = branch(tmp_path, 'branch-b', initial[2], ['purple', 'pink'])
    descendant = branch(tmp_path, 'finishing', first[0], ['black', 'gray'])
    collection(tmp_path, 'initial', initial[:2])
    collection(tmp_path, 'first', first, project_id='branch-a')
    collection(tmp_path, 'second', second, project_id='branch-b')
    collection(tmp_path, 'descendant', descendant, project_id='finishing')
    events = review_events(tmp_path)
    pair_hashes = [set(event['image_hashes']) for event in events[::2]]
    assert all(not a & b for i, a in enumerate(pair_hashes) for b in pair_hashes[i + 1:])
    grouped = assignments(events)
    assert len({row['split_group'] for row in grouped.values()}) == 1
    assert all(e['lineage_provenance']['status'] == 'resolved' for e in events)
    assert all(e['brief_id'] == 'original-brief:' + digest('One original fixture brief.') for e in events)
    assert any(p['type'] == 'studio_artifact_records' for p in events[-1]['lineage_provenance']['proofs'])
    assert all('lineage_provenance' not in e['input_events'][0]['data'] for e in events)


def test_reused_collection_local_candidate_ids_do_not_merge_distinct_briefs(tmp_path):
    a = source_run(tmp_path, 'original-a', 'First independent fixture brief.', ['red', 'green'])
    b = source_run(tmp_path, 'original-b', 'Second independent fixture brief.', ['blue', 'yellow'])
    collection(tmp_path, 'a', a, shared_ids=True)
    collection(tmp_path, 'b', b, shared_ids=True)
    events = review_events(tmp_path)
    assert events[0]['presented_images'][0]['candidate_id'] in ('image-000', 'image-001')
    assert not set(events[0]['lineage_ids']) & set(events[2]['lineage_ids'])
    assert len({row['split_group'] for row in assignments(events).values()}) == 2


def test_export_rejects_vote_image_hash_that_disagrees_with_collection(tmp_path):
    paths = source_run(tmp_path, 'original', 'Fixture brief.', ['red', 'green'])
    collection(tmp_path, 'review', paths)
    # Simulate persisted corruption, explicitly bypassing the review ledger's
    # append-only guard in this isolated fixture. Never rewrite a real vote.
    with sqlite3.connect(tmp_path / 'reviews.sqlite3') as db:
        row = db.execute('SELECT sequence,record FROM review_votes').fetchone()
        vote = json.loads(row[1])
        vote['candidate_order']['A']['sha256'] = '0' * 64
        db.execute('DROP TRIGGER review_votes_no_update')
        db.execute('UPDATE review_votes SET record=? WHERE sequence=?', (json.dumps(vote), row[0]))
    with pytest.raises(ValueError, match='review presentation differs'):
        review_events(tmp_path)


def test_exact_frozen_brief_repeated_in_different_jobs_is_conservatively_grouped(tmp_path):
    a = source_run(tmp_path, 'attempt-a', 'Same exact fixture brief.', ['red', 'green'])
    b = source_run(tmp_path, 'attempt-b', 'Same exact fixture brief.', ['blue', 'yellow'])
    collection(tmp_path, 'a', a)
    collection(tmp_path, 'b', b)
    assert len({row['split_group'] for row in assignments(review_events(tmp_path)).values()}) == 1


def test_identical_imported_ancestor_bytes_connect_disjoint_reviewed_descendants(tmp_path):
    a = source_run(tmp_path, 'a', 'First fixture brief.', ['red', 'green'])
    b = source_run(tmp_path, 'b', 'A differently worded brief.', ['red', 'blue'])
    first = branch(tmp_path, 'first', a[0], ['purple', 'pink'])
    second = branch(tmp_path, 'second', b[0], ['yellow', 'orange'])
    collection(tmp_path, 'first', first, project_id='first')
    collection(tmp_path, 'second', second, project_id='second')
    events = review_events(tmp_path)
    assert not set(events[0]['image_hashes']) & set(events[2]['image_hashes'])
    assert events[0]['brief_id'] != events[2]['brief_id']
    assert len({row['split_group'] for row in assignments(events).values()}) == 1


def test_unknown_lineage_requires_exact_explicit_curator_declaration(tmp_path):
    paths = [png(tmp_path / f'{i}.png', color) for i, color in enumerate(['red', 'blue'])]
    _, spec_hash = collection(tmp_path, 'generic', paths)
    unresolved = review_events(tmp_path)
    assert all(e['lineage_provenance']['status'] == 'unresolved' and not e['input_events'] for e in unresolved)
    report = build_snapshot(unresolved, tmp_path / 'fixture-snapshots', cutoff=CUTOFF)
    assert report['eligible_count'] == 0
    assert report['rejection_reasons']['unresolved_or_related_review_lineage'] == 2
    declarations = {'schema_version': 1, 'author': 'fixture curator', 'reason': 'Explicit fixture origin metadata.',
                    'collections': {'generic': {'collection_sha256': spec_hash, 'project_id': 'fixture-project',
                                    'brief_id': 'fixture-brief', 'trajectory_id': 'fixture-trajectory', 'sealed': False}}}
    events = review_events(tmp_path, lineage_declarations=declarations)
    assert all(e['lineage_provenance']['status'] == 'resolved' and e['project_id'] == 'curated:fixture-project' for e in events)
    assert events[0]['lineage_provenance']['proofs'][-1]['manifest_sha256'] == digest(declarations)
    assert build_snapshot(events, tmp_path / 'fixture-snapshots', cutoff=CUTOFF)['eligible_count'] == 2
    declarations['collections']['generic']['collection_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='exact collection'):
        review_events(tmp_path, lineage_declarations=declarations)


def test_curator_can_group_differently_worded_versions_without_splitting_known_ancestry(tmp_path):
    a = source_run(tmp_path, 'short', 'Fixture brief.', ['red', 'green'])
    b = source_run(tmp_path, 'extended', 'Fixture brief. Added medium constraint.', ['blue', 'yellow'])
    _, a_hash = collection(tmp_path, 'a', a)
    _, b_hash = collection(tmp_path, 'b', b)
    declarations = {'schema_version': 1, 'author': 'fixture curator', 'reason': 'Declared versions of one original brief.',
                    'collections': {cid: {'collection_sha256': value, 'project_id': 'same-project', 'brief_id': 'same-brief',
                                         'trajectory_id': cid, 'sealed': False} for cid, value in [('a', a_hash), ('b', b_hash)]}}
    events = review_events(tmp_path, lineage_declarations=declarations)
    assert len({row['split_group'] for row in assignments(events).values()}) == 1
    assert len({proof['brief_sha256'] for e in events for proof in e['lineage_provenance']['proofs'] if proof['type'] == 'original_brief_digest'}) == 2


def test_conflicting_origins_quarantine_connected_rows_even_with_a_declaration(tmp_path):
    a = source_run(tmp_path, 'a', 'First fixture brief.', ['red', 'green'])
    b = source_run(tmp_path, 'b', 'Second fixture brief.', ['blue', 'yellow'])
    collection(tmp_path, 'a-only', a)
    _, mixed_hash = collection(tmp_path, 'mixed', [a[0], b[0]])
    declarations = {'schema_version': 1, 'author': 'fixture curator', 'reason': 'Cannot excuse conflicting origins.',
                    'collections': {'mixed': {'collection_sha256': mixed_hash, 'project_id': 'invented',
                                    'brief_id': 'invented', 'trajectory_id': 'invented', 'sealed': False}}}
    events = review_events(tmp_path, lineage_declarations=declarations)
    assert any(e['lineage_provenance']['status'] == 'ambiguous' for e in events)
    assert all(row['split'] == 'quarantined' for row in assignments(events).values())
    assert build_snapshot(events, tmp_path / 'fixture-snapshots', cutoff=CUTOFF)['eligible_count'] == 0


def test_broken_import_or_tampered_job_never_resolves(tmp_path):
    initial = source_run(tmp_path, 'original', 'Fixture brief.', ['red', 'green'])
    children = branch(tmp_path, 'branch', initial[0], ['blue', 'yellow'])
    spec, _ = collection(tmp_path, 'branch', children, project_id='branch', vote=False)
    initial[0].write_bytes(b'changed source bytes')
    with ReviewLineageResolver(tmp_path) as resolver:
        result = resolver.resolve(spec)
    assert result['status'] == 'ambiguous'
    assert any('hash_mismatch' in issue['reason'] for issue in result['issues'])
    with sqlite3.connect(tmp_path / 'jobs.sqlite3') as db:
        db.execute("UPDATE jobs SET spec_json='{}' WHERE job_id='branch_action_0'")
    with ReviewLineageResolver(tmp_path) as resolver:
        result = resolver.resolve(spec)
    assert any(issue['reason'] == 'lineage_job_spec_hash_mismatch' for issue in result['issues'])


def test_no_votes_does_not_read_unrelated_collection_text_or_create_a_job_database(tmp_path):
    with sqlite3.connect(tmp_path / 'reviews.sqlite3') as db:
        db.execute('CREATE TABLE review_votes(sequence INTEGER,record TEXT)')
        db.execute('CREATE TABLE review_collections(id TEXT,specification TEXT)')
        db.execute('INSERT INTO review_collections VALUES (?,?)', ('unrelated', 'invalid uninspected fixture text'))
    assert review_events(tmp_path) == []
    assert not (tmp_path / 'jobs.sqlite3').exists()


def test_new_snapshot_identity_discloses_resolver_without_replacing_prior_bytes(tmp_path):
    previous = tmp_path / 'prior-snapshot' / 'manifest.json'
    previous.parent.mkdir()
    previous.write_text('{"historical":"unchanged fixture"}')
    before = previous.read_bytes()
    current = build_snapshot([], tmp_path / 'fixture-snapshots', cutoff=CUTOFF)
    assert current['schema_version'] == 2
    assert current['exporter_identity']['review_adapter'] == ADAPTER_VERSION
    assert len(current['exporter_identity']['lineage_resolver_sha256']) == 64
    assert previous.read_bytes() == before


@pytest.mark.parametrize('partition', ['development', 'sealed_test'])
def test_protocol_jobs_remain_quarantined_without_public_protocol_resolution(tmp_path, partition):
    directory = tmp_path / 'opaque-job'
    paths = [png(directory / f'{i}.png', color) for i, color in enumerate(['red', 'blue'])]
    # Fixture shape only: the live E0 and mode runners bind protocol/runtime or
    # frozen plan hashes; their collection names cannot supply missing ancestry.
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('opaque-job', {'brief_id': 'fixture-study', 'runtime': {'protocol_id': 'fixture-protocol',
                          'partition': partition}, 'frozen_request_sha256': 'a' * 64}, 2, directory)
        for path in paths:
            ledger.register_artifact('opaque-job', path.name, kind='result')
    _, spec_hash = collection(tmp_path, 'opaque-workflow-hash', paths)
    declarations = {'schema_version': 1, 'author': 'fixture curator', 'reason': 'Cannot bypass unsupported protocol ancestry.',
        'collections': {'opaque-workflow-hash': {'collection_sha256': spec_hash, 'project_id': 'invented',
            'brief_id': 'invented', 'trajectory_id': 'invented', 'sealed': False}}}
    events = review_events(tmp_path, lineage_declarations=declarations)
    assert all(e['lineage_provenance']['status'] == 'unresolved' and not e['input_events'] for e in events)
    assert all(e['sealed'] == (partition == 'sealed_test') for e in events)
    assert build_snapshot(events, tmp_path / 'fixture-snapshots', cutoff=CUTOFF)['eligible_count'] == 0


def test_curator_cannot_hide_a_sealed_registered_original(tmp_path):
    paths = source_run(tmp_path, 'original', 'Synthetic sealed fixture brief.', ['red', 'green'])
    with sqlite3.connect(tmp_path / 'jobs.sqlite3') as db:
        spec = json.loads(db.execute("SELECT spec_json FROM jobs WHERE job_id='original'").fetchone()[0])
        spec['sealed'] = True
        wire = json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        db.execute("UPDATE jobs SET spec_json=?,spec_fingerprint=? WHERE job_id='original'", (wire, hashlib.sha256(wire.encode()).hexdigest()))
    _, spec_hash = collection(tmp_path, 'sealed-fixture', paths)
    declarations = {'schema_version': 1, 'author': 'fixture curator', 'reason': 'Cannot declassify a sealed ancestor.',
        'collections': {'sealed-fixture': {'collection_sha256': spec_hash, 'project_id': 'invented',
            'brief_id': 'invented', 'trajectory_id': 'invented', 'sealed': False}}}
    events = review_events(tmp_path, lineage_declarations=declarations)
    assert all(e['sealed'] and not e['input_events'] for e in events)
    assert all(not any(p['type'] == 'original_brief_digest' for p in e['lineage_provenance']['proofs']) for e in events)
    assert build_snapshot(events, tmp_path / 'fixture-snapshots', cutoff=CUTOFF)['eligible_count'] == 0
