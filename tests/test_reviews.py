import json
import sqlite3

from PIL import Image
import pytest
from pydantic import ValidationError

from kusama.reviews import ReviewConflict, ReviewError, ReviewNotFound, ReviewStore


def collection(root, name='experiment'):
    root.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (24, 32), 'red').save(root / 'parent.png')
    Image.new('RGB', (24, 32), 'blue').save(root / 'edit.png')
    store = ReviewStore(root)
    store.register_collection(name, title='An experiment', brief='Develop the visual tension.', candidates=[
        {'id': 'parent', 'path': root / 'parent.png', 'label': 'Original'},
        {'id': 'edit', 'path': root / 'edit.png', 'label': 'Revision', 'parent_candidate_id': 'parent'},
    ], pairs=[{'id': 'before-after', 'candidate_ids': ['parent', 'edit'], 'kind': 'before_after'}])
    return store


def vote_for(comparison, submission='vote1'):
    return {'reviewer': comparison['reviewer'], 'assignment_id': comparison['assignment_id'],
            'submission_id': submission, 'finished_quality': {'choice': 'A', 'confidence': 'high'},
            'development_potential': {'choice': 'B', 'confidence': 'low'}, 'notes': 'Stronger finish, different potential.'}


def test_empty_read_does_not_create_database(tmp_path):
    assert ReviewStore(tmp_path).collections() == []
    assert list(tmp_path.iterdir()) == []


def test_assignment_is_stable_and_randomized_by_reviewer(tmp_path):
    store = collection(tmp_path)
    a = store.comparison('experiment', 'before-after', 'Jayce')
    assert ReviewStore(tmp_path).comparison('experiment', 'before-after', 'Jayce') == a
    positions = {store.comparison('experiment', 'before-after', f'Reviewer {i}')['candidates']['A']['id'] for i in range(30)}
    assert positions == {'parent', 'edit'}
    assert store.history('experiment') == {'votes': [], 'advancements': []}


def test_two_independent_judgments_preserve_candidate_identity_and_append(tmp_path):
    store = collection(tmp_path)
    comparison = store.comparison('experiment', 'before-after', 'Jayce')
    first = store.submit_vote('experiment', 'before-after', vote_for(comparison))
    assert first['finished_quality']['preferred_candidate_id'] == comparison['candidates']['A']['id']
    assert first['development_potential']['preferred_candidate_id'] == comparison['candidates']['B']['id']
    assert first['candidate_order']['A']['sha256'] == comparison['candidates']['A']['sha256']
    assert store.submit_vote('experiment', 'before-after', vote_for(comparison)) == first
    update = vote_for(comparison, 'vote2')
    update['finished_quality'] = {'choice': 'tie'}
    update['development_potential'] = {'choice': 'unsure'}
    second = store.submit_vote('experiment', 'before-after', update)
    assert second['finished_quality']['preferred_candidate_id'] is None
    assert second['development_potential']['preferred_candidate_id'] is None
    assert second['finished_quality']['choice'] != second['development_potential']['choice']
    assert store.history('experiment')['votes'] == [first, second]
    assert store.collections('Jayce')[0]['reviewed_pair_ids'] == ['before-after']
    with sqlite3.connect(store.database) as db:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('DELETE FROM review_votes')
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute("UPDATE review_collections SET specification='{}'")


def test_incomplete_or_stale_reviews_rejected(tmp_path):
    store = collection(tmp_path)
    comparison = store.comparison('experiment', 'before-after', 'Jayce')
    vote = vote_for(comparison)
    del vote['development_potential']
    with pytest.raises(ValidationError):
        store.submit_vote('experiment', 'before-after', vote)
    vote = vote_for(comparison)
    vote['assignment_id'] = 'bad'
    with pytest.raises(ReviewConflict, match='assignment'):
        store.submit_vote('experiment', 'before-after', vote)
    vote = vote_for(comparison)
    store.submit_vote('experiment', 'before-after', vote)
    vote['notes'] = 'Different content'
    with pytest.raises(ReviewConflict, match='Submission ID'):
        store.submit_vote('experiment', 'before-after', vote)


def test_modified_or_retargeted_images_cannot_be_voted_on(tmp_path):
    root = tmp_path / 'artifacts'
    store = collection(root)
    comparison = store.comparison('experiment', 'before-after', 'Jayce')
    Image.new('RGB', (24, 32), 'green').save(root / 'edit.png')
    with pytest.raises(ReviewConflict, match='changed'):
        store.submit_vote('experiment', 'before-after', vote_for(comparison))
    (root / 'edit.png').unlink()
    outside = tmp_path / 'outside.png'
    Image.new('RGB', (24, 32), 'blue').save(outside)
    (root / 'edit.png').symlink_to(outside)
    with pytest.raises(ReviewConflict):
        store.candidate_path('experiment', 'edit')
    assert store.history('experiment')['votes'] == []


def test_registration_is_idempotent_but_not_replaceable(tmp_path):
    store = collection(tmp_path)
    before = store.collection('experiment')
    assert collection(tmp_path).collection('experiment') == before
    candidates = [{'id': 'a', 'path': tmp_path / 'parent.png'}, {'id': 'b', 'path': tmp_path / 'edit.png'}]
    with pytest.raises(ReviewConflict):
        store.register_collection('experiment', title='Changed', brief='Different brief', candidates=candidates)
    store.register_collection('second', title='Second', brief='A different comparison', candidates=candidates)
    assert len(store.collection('second')['pairs']) == 1


def test_registration_rejects_paths_outside_root_and_symlink_directories(tmp_path):
    root = tmp_path / 'artifacts'
    store = collection(root)
    outside = tmp_path / 'other'
    outside.mkdir()
    Image.new('RGB', (24, 32), 'yellow').save(outside / 'image.png')
    (root / 'linked').symlink_to(outside, target_is_directory=True)
    for path in (outside / 'image.png', root / '..' / 'other' / 'image.png', root / 'linked' / 'image.png'):
        with pytest.raises(ReviewError):
            store.register_collection('bad', title='Bad', brief='Bad', candidates=[
                {'id': 'a', 'path': root / 'parent.png'}, {'id': 'b', 'path': path}])


def test_advancement_is_explicit_and_separate_from_votes(tmp_path):
    store = collection(tmp_path)
    comparison = store.comparison('experiment', 'before-after', 'Jayce')
    store.submit_vote('experiment', 'before-after', vote_for(comparison))
    assert store.history('experiment')['advancements'] == []
    request = {'reviewer': 'Jayce', 'candidate_id': 'edit', 'reason': 'Explore its unusual structure.', 'submission_id': 'advance1'}
    record = store.advance('experiment', request)
    assert store.advance('experiment', request) == record
    assert record['image_sha256'] == next(c for c in store.collection('experiment')['candidates'] if c['id'] == 'edit')['sha256']
    assert store.history('experiment')['advancements'] == [record]
    assert not (tmp_path / 'jobs.sqlite3').exists()
    with sqlite3.connect(store.database) as db:
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            db.execute('DELETE FROM review_advancements')


def test_simultaneous_retry_saves_one_review(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store = collection(tmp_path)
    comparison = store.comparison('experiment', 'before-after', 'Jayce')
    vote = vote_for(comparison)
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(executor.map(lambda _: ReviewStore(tmp_path).submit_vote('experiment', 'before-after', vote), range(2)))
    assert records[0] == records[1]
    assert len(store.history('experiment')['votes']) == 1


def test_situated_review_labels_exposure_and_nulls_keep_order(tmp_path):
    store = collection(tmp_path)
    comparison = store.comparison('experiment', 'before-after', 'Jayce')
    vote = vote_for(comparison)
    vote.update(open_reading='Two kinds of enclosure', felt_response='Uncertain',
        readiness={'A': 'Y', 'B': None}, development_willingness={'A': 'N', 'B': 'U'},
        exposure={'prior_exposure': 'images', 'context_revealed': True,
                  'duration_seconds': 25.4, 'duration_source': 'browser_elapsed'})
    vote['finished_quality'] = {'choice': 'incomparable'}
    vote['development_potential'] = {'choice': None}
    result = store.submit_vote('experiment', 'before-after', vote)
    assert result['finished_quality']['choice'] == 'incomparable'
    assert result['finished_quality']['preferred_candidate_id'] is None
    assert result['development_potential']['choice'] is None
    assert result['readiness']['A']['candidate_id'] == comparison['candidates']['A']['id']
    assert result['readiness']['B']['response'] is None
    assert result['development_willingness']['B']['image_sha256'] == comparison['candidates']['B']['sha256']
    assert result['exposure']['duration_seconds'] == 25.4
    assert store.submit_vote('experiment', 'before-after', vote) == result
    assert store.history('experiment')['advancements'] == []


def test_new_balanced_protocol_does_not_change_existing_assignments(tmp_path):
    store = collection(tmp_path)
    before = store.comparison('experiment', 'before-after', 'Jayce')
    store.register_collection('protocol', title='Protocol', brief='Do not show this brief in statement-free condition',
        candidates=[
            {'id': 'a', 'path': tmp_path / 'parent.png'}, {'id': 'b', 'path': tmp_path / 'edit.png'}],
        presentation={'condition': 'statement_free'})
    assert store.comparison('experiment', 'before-after', 'Jayce') == before
    assert store.comparison('protocol', 'pair-001', 'Jayce')['presentation']['condition'] == 'statement_free'
    with pytest.raises(ValidationError, match='attributed author'):
        store.register_collection('badcontext', title='Bad', brief='Brief', candidates=[
            {'id': 'a', 'path': tmp_path / 'parent.png'}, {'id': 'b', 'path': tmp_path / 'edit.png'}],
            presentation={'condition': 'supplied_context', 'supplied_context': 'A statement'})


def test_pool_none_is_a_real_separate_immutable_choice(tmp_path):
    store = collection(tmp_path)
    assignment = store.pool_assignment('experiment', 'Viewer')
    request = {'reviewer': 'Viewer', 'assignment_id': assignment['assignment_id'],
               'submission_id': 'pool-1', 'choice': 'none'}
    assert assignment['human_label'] is None
    result = store.submit_pool_choice('experiment', request)
    assert result['type'] == 'human_pool_selection' and result['choice'] == 'none'
    assert result['selected_image_sha256'] is None
    assert store.submit_pool_choice('experiment', request) == result
    assert store.history('experiment') == {'votes': [], 'advancements': []}
    assert store.pool_history('experiment') == [result]
    candidate = assignment['candidates']['1']
    chosen = store.submit_pool_choice('experiment', request | {'submission_id': 'pool-2', 'choice': 'candidate', 'candidate_id': candidate['id']})
    assert chosen['selected_image_sha256'] == candidate['sha256']
    with sqlite3.connect(store.database) as db, pytest.raises(sqlite3.IntegrityError, match='append-only'):
        db.execute('DELETE FROM review_pool_votes')


def test_incomplete_pool_is_explicitly_excluded(tmp_path):
    store = collection(tmp_path)
    store.register_collection('partial', title='Partial', brief='Brief', candidates=[
        {'id': 'a', 'path': tmp_path / 'parent.png'}, {'id': 'b', 'path': tmp_path / 'edit.png'}], pool_expected_count=3)
    assignment = store.pool_assignment('partial', 'Viewer')
    assert assignment['eligibility'] == 'incomplete_pool_excluded'
    with pytest.raises(ReviewConflict, match='Incomplete'):
        store.submit_pool_choice('partial', {'reviewer': 'Viewer', 'assignment_id': assignment['assignment_id'],
            'submission_id': 'none-for-partial', 'choice': 'none'})
    assert store.pool_history('partial') == []


def test_pool_subset_excludes_parent_without_changing_pairwise_members(tmp_path):
    store = collection(tmp_path)
    candidates = [{'id': 'parent', 'path': tmp_path / 'parent.png'}]
    for i in range(6):
        candidates.append({'id': f'child{i}', 'path': tmp_path / 'edit.png', 'parent_candidate_id': 'parent'})
    children = [c['id'] for c in candidates[1:]]
    pairs = [{'id': f'pair{i}', 'candidate_ids': ['parent', child], 'kind': 'before_after'} for i, child in enumerate(children)]
    store.register_collection('e0', title='E0 pool mechanics', brief='Fixture', candidates=candidates, pairs=pairs,
                              pool_candidate_ids=children, pool_expected_count=6)
    assignment = store.pool_assignment('e0', 'Reviewer')
    assert assignment['complete'] and len(assignment['candidates']) == 6
    assert 'parent' not in {c['id'] for c in assignment['candidates'].values()}
    assert 'parent' in {c['id'] for c in store.comparison('e0', 'pair0', 'Reviewer')['candidates'].values()}
    with pytest.raises(ReviewNotFound):
        store.submit_pool_choice('e0', {'reviewer': 'Reviewer', 'assignment_id': assignment['assignment_id'],
            'submission_id': 'badparent', 'choice': 'candidate', 'candidate_id': 'parent'})
    store.register_collection('e0partial', title='Partial', brief='Fixture', candidates=candidates[:-1], pairs=pairs[:-1],
                              pool_candidate_ids=children[:-1], pool_expected_count=6)
    assert store.pool_assignment('e0partial', 'Reviewer')['eligibility'] == 'incomplete_pool_excluded'
