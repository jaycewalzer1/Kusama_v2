import json
import sqlite3

from PIL import Image
import pytest
from pydantic import ValidationError

from kusama.reviews import ReviewConflict, ReviewError, ReviewStore


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
