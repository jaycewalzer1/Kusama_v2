"""Decision-time memory fixtures; no real inference or human preference evidence."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from kusama import ledger as ledger_module
from kusama.ledger import Ledger
from kusama.memory import ExperimentMemory, MemoryQuery, build_memory, file_hash


BEFORE = '2025-01-01T10:00:00+00:00'
DECISION = '2025-01-01T11:00:00+00:00'
AFTER = '2025-01-01T12:00:00+00:00'


def make_job(root, monkeypatch, job_id='episode', partition='development', when=BEFORE):
    monkeypatch.setattr(ledger_module, '_now', lambda: when)
    directory = root / 'runs' / job_id
    directory.mkdir(parents=True)
    ledger = Ledger(root / 'jobs.sqlite3')
    spec = {'kind': 'connection_generation', 'partition': partition,
            'brief': 'Enclosure and support through a frame; care and control.',
            'source_documents': [{'reference_id': 'source_one', 'source_family': 'family_a'}],
            'actions': [{'id': 'child', 'kind': 'generate', 'operation': 'encloses',
                         'instruction': 'An enclosure supports a delicate frame.',
                         'reference_ids': ['source_one'], 'parent_artifact_id': None}]}
    ledger.create_job(job_id, spec, 2, directory)
    ledger.open_attempt(job_id, 0, payload={'fixture_only': True})
    return ledger, directory


def add_image(ledger, directory, job_id, name='child.png', color='red'):
    Image.new('RGB', (64, 64), color).save(directory / name)
    ledger.register_artifact(job_id, name, attempt_index=0, kind='image')
    return directory / name


def add_response(ledger, directory, job_id, filename, payload):
    (directory / filename).write_text(json.dumps(payload))
    ledger.register_artifact(job_id, filename, kind='response')


def test_actual_artifact_hash_and_source_database_remain_unchanged(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    image = add_image(ledger, directory, 'episode')
    ledger.close()
    source_hash = file_hash(tmp_path / 'jobs.sqlite3')
    report = build_memory(tmp_path)
    assert report['issues'] == []
    assert source_hash == file_hash(tmp_path / 'jobs.sqlite3')
    with ExperimentMemory(tmp_path) as memory:
        result = memory.query('enclosure support', DECISION, 'next')
        child = next(item for item in result['selected'] if item['kind'] == 'artifact')
        assert child['payload']['sha256'] == file_hash(image)
        assert child['evidence'][0]['path'] == str(image)
        assert child['source_ids'] == ['source_one']
        assert child['source_families'] == ['family_a']
        assert child['source_event_id'] and child['source_sequence']
        assert result['snapshot_id'] == memory.query('enclosure support', DECISION, 'next')['snapshot_id']
    assert build_memory(tmp_path)['new_events'] == 0


def test_future_child_and_future_observation_do_not_change_historical_candidate_set(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    add_image(ledger, directory, 'episode', 'first.png')
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        before = memory.query('enclosure support', DECISION, 'episode')
    monkeypatch.setattr(ledger_module, '_now', lambda: AFTER)
    add_image(ledger, directory, 'episode', 'future.png', 'blue')
    add_response(ledger, directory, 'episode', 'observation.json', {
        'output': {'visible_changes': ['The future enclosure supports the frame.'],
                   'preservation_failures': [], 'ambiguity': 'Fixture only.'}})
    ledger.close()
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        after = memory.query('enclosure support', DECISION, 'episode')
        assert after == before
        assert all(not str(item['payload'].get('path', '')).endswith('future.png') for item in after['selected'])
        assert all(item['kind'] != 'observation' for item in after['selected'])
        later = memory.query('enclosure support', '2025-01-01T13:00:00+00:00', 'episode')
        assert any(item['kind'] == 'observation' and item['origin'] == 'model_observation' for item in later['selected'])


def test_cutoff_is_exclusive_and_naive_or_future_decisions_rejected(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch, when=DECISION)
    add_image(ledger, directory, 'episode')
    ledger.close()
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        assert memory.query('enclosure', DECISION, 'episode')['selected'] == []
        with pytest.raises(ValueError, match='timezone-aware'):
            memory.query('enclosure', '2025-01-01T11:00:00', 'episode')
        with pytest.raises(ValueError, match='future'):
            memory.query('enclosure', (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), 'episode')


def test_heldout_only_own_preceding_state_and_development_excludes_sealed(tmp_path, monkeypatch):
    for episode, partition in [('dev', 'development'), ('sealed_a', 'sealed_test'), ('sealed_b', 'sealed_test')]:
        ledger, directory = make_job(tmp_path, monkeypatch, episode, partition)
        add_image(ledger, directory, episode, color={'dev': 'red', 'sealed_a': 'blue', 'sealed_b': 'green'}[episode])
        ledger.close()
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        heldout = memory.query('enclosure', DECISION, 'sealed_a', 'heldout')
        assert heldout['selected'] and {item['episode_id'] for item in heldout['selected']} == {'sealed_a'}
        dev = memory.query('enclosure', DECISION, 'new', 'development')
        assert dev['selected'] and {item['episode_id'] for item in dev['selected']} == {'dev'}
        assert memory.query('enclosure', DECISION, 'new', exclude_source_families=['family_a'])['selected'] == []


def test_failed_cases_preserved_without_model_verdict_becoming_human_review(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    add_image(ledger, directory, 'episode')
    add_response(ledger, directory, 'episode', 'failed.json', {'parse_error': 'Fixture invalid structure', 'raw_response': 'not valid json'})
    add_response(ledger, directory, 'episode', 'verdict.json', {'output': {'judgment': 'Fixture model likes enclosure'}})
    ledger.close()
    report = build_memory(tmp_path)
    assert report['kinds'].get('human_review', 0) == 0
    assert report['kinds']['failure'] == 1
    with ExperimentMemory(tmp_path) as memory:
        result = memory.query('enclosure', DECISION, 'next', limit=2)
        assert any(item['kind'] == 'failure' for item in result['selected'])
        assert all(item.origin != 'human_review_ledger' for item in memory.events())


def test_append_only_memory_and_changed_image_fail_closed(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    image = add_image(ledger, directory, 'episode')
    ledger.close()
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        result = memory.query('enclosure', DECISION, 'next')
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            memory.db.execute('DELETE FROM memory_events')
        memory.db.rollback()
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            memory.db.execute("UPDATE memory_queries SET snapshot_json='{}'")
        memory.db.rollback()
        image.write_bytes(b'changed')
        with pytest.raises(ValueError, match='evidence changed'):
            memory.query('enclosure', DECISION, 'next')
        assert result['snapshot_id']


def add_review_fixture(root, image, *, when=BEFORE):
    """Synthetic test ledger, clearly isolated from all real review stores."""
    with sqlite3.connect(root / 'reviews.sqlite3') as db:
        db.executescript('''
            CREATE TABLE review_collections(id TEXT, specification TEXT, random_seed TEXT, created_at TEXT);
            CREATE TABLE review_advancements(sequence INTEGER, submission_id TEXT, collection_id TEXT, reviewer TEXT, request TEXT, record TEXT);
        ''')
        collection = {'brief': 'Enclosure fixture', 'candidates': [{'id': 'fixture_child', 'path': str(image), 'sha256': file_hash(image)}]}
        record = {'type': 'human_advancement', 'submission_id': 'fixture_review', 'collection_id': 'fixture_collection',
                  'reviewer': 'fixture-human', 'created_at': when, 'candidate_id': 'fixture_child',
                  'image_sha256': file_hash(image), 'reason': 'Enclosure fixture continuation choice, not quality vote.'}
        db.execute('INSERT INTO review_collections VALUES(?,?,?,?)', ('fixture_collection', json.dumps(collection), 'fixture', BEFORE))
        db.execute('INSERT INTO review_advancements VALUES(?,?,?,?,?,?)', (1, 'fixture_review', 'fixture_collection', 'fixture-human', '{}', json.dumps(record)))


def test_genuine_review_rows_remain_separate_and_never_enter_heldout_input(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    image = add_image(ledger, directory, 'episode')
    ledger.close()
    add_review_fixture(tmp_path, image)
    report = build_memory(tmp_path)
    assert report['issues'] == [] and report['kinds']['human_review'] == 1
    with ExperimentMemory(tmp_path) as memory:
        reviews = [event for event in memory.events() if event.kind == 'human_review']
        assert reviews[0].payload['type'] == 'human_advancement'
        assert 'finished_quality' not in reviews[0].payload
        assert reviews[0].source_sequence == 1 and reviews[0].source_event_id == 'fixture_review'
        assert all(item['kind'] != 'human_review' for item in memory.query('enclosure', DECISION, 'episode', 'heldout')['selected'])


def test_future_human_review_does_not_change_prior_input(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    image = add_image(ledger, directory, 'episode')
    ledger.close()
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        prior = memory.query('enclosure', DECISION, 'episode')
    add_review_fixture(tmp_path, image, when=AFTER)
    build_memory(tmp_path)
    with ExperimentMemory(tmp_path) as memory:
        assert memory.query('enclosure', DECISION, 'episode') == prior


def test_query_input_schema_rejects_unsupported_controls():
    with pytest.raises(ValueError):
        MemoryQuery(brief='enclosure', decision_at=DECISION, episode_id='fixture', execute='rm -rf /')
    with pytest.raises(ValueError):
        MemoryQuery(brief='enclosure', decision_at=DECISION, episode_id='fixture', limit=33)


def test_continuation_actual_child_lineage_and_observation_are_not_backdated(tmp_path, monkeypatch):
    ledger, directory = make_job(tmp_path, monkeypatch)
    parent = add_image(ledger, directory, 'episode')
    child = directory / 'revision.png'
    Image.new('RGB', (64, 64), 'blue').save(child)
    spec = {'kind': 'selected_parent_continuation', 'id': 'revision', 'project_id': 'cont_revision',
            'selection': {'path': str(parent), 'image_sha256': file_hash(parent), 'origin': 'operator',
                          'author': 'fixture-operator', 'reason': 'Explore enclosure'},
            'request': {'brief': 'Enclosure and support', 'references': [],
                        'revision': {'origin': 'model_proposal', 'instruction': 'Enclose the frame', 'relationship': 'supports'}},
            'max_calls': 2}
    project = {'artifacts': {'root': {'path': str(parent), 'sha256': file_hash(parent)},
                            'variant:0': {'id': 'variant:0', 'path': str(child), 'sha256': file_hash(child),
                                          'parent_artifact_id': 'root', 'action_id': 'variant', 'origin': 'model_edit'}},
               'actions': {'variant': {'parent_artifact_id': 'root'}}}
    ledger.connection.executescript('''
        CREATE TABLE continuation_specs(id TEXT, specification TEXT, created_at TEXT);
        CREATE TABLE studio_projects(id TEXT, data TEXT);
        CREATE TABLE studio_events(sequence INTEGER, project_id TEXT, type TEXT, details TEXT, created_at TEXT);
    ''')
    ledger.connection.execute('INSERT INTO continuation_specs VALUES(?,?,?)', ('revision', json.dumps(spec), BEFORE))
    ledger.connection.execute('INSERT INTO studio_projects VALUES(?,?)', ('cont_revision', json.dumps(project)))
    ledger.connection.execute('INSERT INTO studio_events VALUES(?,?,?,?,?)', (1, 'cont_revision', 'project_created', json.dumps({'parent_sha256': file_hash(parent)}), BEFORE))
    ledger.connection.execute('INSERT INTO studio_events VALUES(?,?,?,?,?)', (2, 'cont_revision', 'action_finished', json.dumps({'action_id': 'variant', 'status': 'complete', 'outputs': ['variant:0']}), AFTER))
    ledger.connection.execute('INSERT INTO studio_events VALUES(?,?,?,?,?)', (3, 'cont_revision', 'artifact_observed', json.dumps({'artifact_id': 'variant:0', 'image_sha256': file_hash(child), 'note': 'Enclosure visibly supports a larger frame.', 'origin': 'model_observation', 'author': 'fixture-model'}), AFTER))
    ledger.close()
    report = build_memory(tmp_path)
    assert report['issues'] == []
    assert report['episodes'] == ['episode']
    with ExperimentMemory(tmp_path) as memory:
        prior = memory.query('enclosure support', DECISION, 'episode')
        assert all('children' not in event['payload'] and event['kind'] != 'observation' for event in prior['selected'])
        records = memory.events()
        transition = next(event for event in records if 'children' in event.payload)
        assert transition.payload['parent']['sha256'] == file_hash(parent)
        assert transition.payload['children'][0]['sha256'] == file_hash(child)
        assert transition.available_at == AFTER
        observation = next(event for event in records if event.kind == 'observation')
        assert observation.origin == 'model_observation' and observation.evidence[0].sha256 == file_hash(child)


def test_shared_image_cannot_declassify_sealed_review(tmp_path, monkeypatch):
    for episode, partition in [('sealed', 'sealed_test'), ('dev', 'development')]:
        ledger, directory = make_job(tmp_path, monkeypatch, episode, partition)
        image = add_image(ledger, directory, episode, color='red')
        ledger.close()
    add_review_fixture(tmp_path, image)
    report = build_memory(tmp_path)
    assert report['issues'] == []
    with ExperimentMemory(tmp_path) as memory:
        review = next(event for event in memory.events() if event.kind == 'human_review')
        assert review.partition == 'sealed_test'
        assert all(event['kind'] != 'human_review' for event in memory.query('enclosure', DECISION, 'dev')['selected'])
