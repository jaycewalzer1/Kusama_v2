import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import get_context
from pathlib import Path
import sqlite3

import pytest

from kusama.ledger import BudgetExceeded, EventExportError, JobLocked, Ledger, SpecMismatch


@pytest.fixture
def ledger(tmp_path):
    with Ledger(tmp_path / 'jobs.sqlite3') as result:
        yield result


def job(ledger, name='edit', budget=2, parent=None):
    directory = ledger.path.parent / name
    directory.mkdir(exist_ok=True)
    return ledger.create_job(name, {'seed': 23, 'method': 'masked'}, budget, directory, parent)


def _lock_worker(database, queue):
    with Ledger(database) as worker:
        try:
            with worker.lock('edit'):
                queue.put('acquired')
        except JobLocked:
            queue.put('locked')


def _reserve_worker(database, name):
    with Ledger(database) as worker:
        try:
            return worker.reserve_call(name, 0)
        except BudgetExceeded:
            return 'exhausted'


def test_job_identity_and_payload_survive_reopen(ledger):
    original = job(ledger)
    ledger.update_job('edit', 'running', {'candidate': 'result.png'})
    ledger.update_job('edit', payload={'elapsed_seconds': 2})
    with Ledger(ledger.path) as reopened:
        restored = job(reopened)
        assert restored['created_at'] == original['created_at']
        assert restored['spec_fingerprint'] == original['spec_fingerprint']
        assert restored['schema_version'] == 1
        assert restored['status'] == 'running'
        assert restored['payload'] == {'candidate': 'result.png', 'elapsed_seconds': 2}
        assert restored['model_calls'] == 0


@pytest.mark.parametrize('change', ['spec', 'budget', 'directory', 'parent'])
def test_reopening_cannot_change_immutable_job_fields(ledger, change):
    job(ledger, 'parent')
    original = job(ledger)
    spec, budget, directory, parent = original['spec'], 2, original['directory'], None
    if change == 'spec':
        spec = dict(spec, seed=24)
    elif change == 'budget':
        budget = 3
    elif change == 'directory':
        directory = Path(directory).with_name('different')
    else:
        parent = 'parent'
    with pytest.raises(SpecMismatch):
        ledger.create_job('edit', spec, budget, directory, parent)
    assert ledger.get_job('edit') == original
    assert len(ledger.events('edit')) == 1


def test_spec_fingerprint_ignores_dictionary_order(ledger):
    original = job(ledger)
    equivalent = {'method': 'masked', 'seed': 23}
    assert ledger.create_job('edit', equivalent, 2, original['directory']) == original


def test_job_directory_cannot_be_reassigned(ledger):
    original = job(ledger)
    with pytest.raises(SpecMismatch, match='directory'):
        ledger.create_job('other', {}, 2, original['directory'])


def test_attempt_checkpoint_merges_and_has_stable_identity(ledger):
    job(ledger)
    assert ledger.get_attempt('edit', 0) is None
    first = ledger.save_attempt('edit', 0, {'method': 'masked', 'status': 'running'})
    ledger.save_attempt('edit', 0, {'seed': 23}, checkpoint={'raw': 'attempt_0_raw.png'})
    ledger.save_attempt('edit', 0, {'status': 'needs_review'}, checkpoint={'verified': True})
    with Ledger(ledger.path) as reopened:
        restored = reopened.get_attempt('edit', 0)
    assert restored['attempt_id'] == first['attempt_id'] == 'edit:0'
    assert restored['created_at'] == first['created_at']
    assert restored['method'] == 'masked'
    assert restored['seed'] == 23
    assert restored['status'] == 'needs_review'
    assert restored['checkpoint'] == {'raw': 'attempt_0_raw.png', 'verified': True}
    assert restored['reserved'] is False


def test_reservation_is_idempotent_across_crash_and_reopen(ledger):
    job(ledger, budget=1)
    assert ledger.reserve_call('edit', 0) is True
    assert ledger.get_attempt('edit', 0)['status'] == 'started'
    with Ledger(ledger.path) as resumed:
        assert resumed.reserve_call('edit', 0) is False
        resumed.save_attempt('edit', 0, {'status': 'uncertain', 'reason': 'worker vanished'})
        assert resumed.reserve_call('edit', 0) is False
        with pytest.raises(BudgetExceeded):
            resumed.reserve_call('edit', 1)
        assert resumed.get_attempt('edit', 1) is None
        assert resumed.call_count('edit') == 1
    assert [e['type'] for e in ledger.events('edit')].count('call_reserved') == 1


def test_parent_and_child_budget_reservations_are_atomic(ledger):
    job(ledger, 'parent', budget=2)
    job(ledger, 'first', budget=1, parent='parent')
    job(ledger, 'second', budget=2, parent='parent')
    ledger.reserve_call('first', 0)
    with pytest.raises(BudgetExceeded, match='first'):
        ledger.reserve_call('first', 1)
    assert ledger.call_count('parent') == 1
    ledger.reserve_call('second', 0)
    with pytest.raises(BudgetExceeded, match='parent'):
        ledger.reserve_call('second', 1)
    assert ledger.call_count('first') == 1
    assert ledger.call_count('second') == 1
    assert ledger.call_count('parent') == 2
    assert ledger.get_attempt('second', 1) is None


def test_reservation_failure_rolls_back_attempt_and_all_budget_charges(ledger, monkeypatch):
    job(ledger, 'parent')
    job(ledger, 'edit', parent='parent')
    original_event = ledger._event

    def interrupted(job_id, kind, payload):
        original_event(job_id, kind, payload)
        if kind == 'child_call_reserved':
            raise KeyboardInterrupt('process stopped before transaction commit')

    monkeypatch.setattr(ledger, '_event', interrupted)
    with pytest.raises(KeyboardInterrupt):
        ledger.reserve_call('edit', 0)
    assert ledger.call_count('edit') == ledger.call_count('parent') == 0
    assert ledger.get_attempt('edit', 0) is None
    assert len(ledger.events()) == 2
    monkeypatch.setattr(ledger, '_event', original_event)
    assert ledger.reserve_call('edit', 0) is True


def test_concurrent_children_cannot_overspend_parent(ledger):
    job(ledger, 'parent', budget=1)
    job(ledger, 'first', parent='parent')
    job(ledger, 'second', parent='parent')
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda name: _reserve_worker(ledger.path, name), ['first', 'second']))
    assert results.count(True) == results.count('exhausted') == 1
    assert ledger.call_count('parent') == 1
    assert ledger.call_count('first') + ledger.call_count('second') == 1


def test_duplicate_concurrent_reservations_charge_once(ledger):
    job(ledger)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: _reserve_worker(ledger.path, 'edit'), range(2)))
    assert sorted(results) == [False, True]
    assert ledger.call_count('edit') == 1


def test_run_lock_excludes_other_process_without_holding_database_transaction(ledger):
    job(ledger)
    context = get_context('spawn')
    queue = context.Queue()
    with ledger.lock('edit'):
        assert not ledger.connection.in_transaction
        process = context.Process(target=_lock_worker, args=(ledger.path, queue))
        process.start()
        assert queue.get(timeout=10) == 'locked'
        process.join(timeout=10)
        assert process.exitcode == 0
        with Ledger(ledger.path) as other:
            job(other, 'independent')
            with other.lock('independent'):
                assert other.reserve_call('independent', 0)
    process = context.Process(target=_lock_worker, args=(ledger.path, queue))
    process.start()
    assert queue.get(timeout=10) == 'acquired'
    process.join(timeout=10)
    assert process.exitcode == 0
    queue.close()


def test_artifact_registration_is_idempotent_and_verifies_immutable_bytes(ledger):
    directory = Path(job(ledger)['directory'])
    raw = directory / 'raw.png'
    raw.write_bytes(b'first model output')
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    first = ledger.register_artifact('edit', 'raw.png', digest)
    assert ledger.register_artifact('edit', raw) == first
    assert first['path'] == 'raw.png'
    assert first['sha256'] == digest
    assert len(ledger.artifacts('edit')) == 1
    assert ledger.verify_artifacts('edit') == [first]
    raw.write_bytes(b'changed model output')
    with pytest.raises(SpecMismatch, match='changed'):
        ledger.register_artifact('edit', 'raw.png')
    with pytest.raises(SpecMismatch, match='changed'):
        ledger.verify_artifacts('edit')
    raw.unlink()
    with pytest.raises(SpecMismatch, match='missing'):
        ledger.verify_artifacts('edit')
    assert ledger.artifacts('edit') == [first]


def test_artifact_rejects_bad_hash_and_paths_outside_job(ledger):
    directory = Path(job(ledger)['directory'])
    (directory / 'raw.png').write_bytes(b'raw')
    with pytest.raises(SpecMismatch, match='supplied hash'):
        ledger.register_artifact('edit', 'raw.png', '0' * 64)
    outside = directory.parent / 'external.png'
    outside.write_bytes(b'outside')
    for path in ('../external.png', outside):
        with pytest.raises(ValueError, match='inside'):
            ledger.register_artifact('edit', path)
    (directory / 'symlink.png').symlink_to(outside)
    with pytest.raises(ValueError, match='inside'):
        ledger.register_artifact('edit', 'symlink.png')
    assert ledger.artifacts('edit') == []


def test_events_are_append_only_versioned_and_export_idempotently(ledger, tmp_path):
    job(ledger)
    path = tmp_path / 'events.jsonl'
    assert ledger.export_events(path, 'edit') == 1
    first = path.read_bytes()
    ledger.reserve_call('edit', 0)
    count = ledger.export_events(path, 'edit')
    assert path.read_bytes().startswith(first)
    contents = path.read_bytes()
    assert ledger.export_events(path, 'edit') == count
    assert path.read_bytes() == contents
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert events == ledger.events('edit')
    assert len({event['event_id'] for event in events}) == count
    assert all(event['schema_version'] == 1 for event in events)
    with pytest.raises(sqlite3.IntegrityError, match='append-only'):
        ledger.connection.execute("DELETE FROM events")
    with pytest.raises(sqlite3.IntegrityError, match='append-only'):
        ledger.connection.execute("UPDATE events SET type='changed'")


@pytest.mark.parametrize('damage', ['torn', 'mismatch', 'extra'])
def test_event_export_requires_explicit_recovery_from_sqlite(ledger, tmp_path, damage):
    job(ledger)
    ledger.reserve_call('edit', 0)
    path = tmp_path / 'events.jsonl'
    ledger.export_events(path)
    original = path.read_bytes()
    damaged = {'torn': original[:-7], 'mismatch': b'{}\n', 'extra': original + b'{}\n'}[damage]
    path.write_bytes(damaged)
    with pytest.raises(EventExportError, match='recover=True'):
        ledger.export_events(path)
    assert path.read_bytes() == damaged
    ledger.export_events(path, recover=True)
    assert path.read_bytes() == original


def test_failed_event_recovery_preserves_old_log_and_can_be_retried(ledger, tmp_path, monkeypatch):
    import kusama.ledger as module

    job(ledger)
    path = tmp_path / 'events.jsonl'
    path.write_bytes(b'{torn')
    replace = module.os.replace

    def failed_replace(*args):
        raise OSError('interrupted before atomic replacement')

    monkeypatch.setattr(module.os, 'replace', failed_replace)
    with pytest.raises(OSError):
        ledger.export_events(path, recover=True)
    assert path.read_bytes() == b'{torn'
    monkeypatch.setattr(module.os, 'replace', replace)
    ledger.export_events(path, recover=True)
    assert [json.loads(line) for line in path.read_text().splitlines()] == ledger.events()


def test_invalid_json_rolls_back_without_an_event(ledger):
    job(ledger)
    with pytest.raises(ValueError):
        ledger.save_attempt('edit', 0, {'metric': float('nan')})
    assert ledger.get_attempt('edit', 0) is None
    assert len(ledger.events()) == 1
