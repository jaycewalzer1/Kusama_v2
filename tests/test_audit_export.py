"""Persistence checks use local synthetic records, never model or human evidence."""
import json
from pathlib import Path
import sqlite3
import zipfile

from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.audit import audit, _model_activity
from kusama.cli import export_evidence
from kusama.costs import CostJournal
from kusama.ledger import Ledger


def test_audit_distinguishes_corruption_from_missing_evaluation(tmp_path):
    root = tmp_path / 'artifacts'
    job = root / 'run'
    job.mkdir(parents=True)
    image = job / 'fixture.png'
    Image.new('RGB', (16, 16), 'red').save(image)
    with Ledger(root / 'jobs.sqlite3') as ledger:
        ledger.create_job('fixture', {'fixture_only': True}, 1, job)
        ledger.register_artifact('fixture', image.name, kind='image')
    result = audit(root)
    assert result['integrity_status'] == 'passed'
    assert result['registered_unique_image_bytes'] == 1
    assert result['reviews']['genuine_submitted_pair_votes'] == 0
    assert result['jobs'][0]['direct_calls'] == 0
    image.write_bytes(b'corrupted saved evidence')
    result = audit(root)
    assert result['integrity_status'] == 'failed'
    assert result['issues'][0]['error'] == 'Registered artifact hash mismatch'


def test_export_includes_committed_wal_and_exact_bytes_without_mutating_inputs(tmp_path):
    root = tmp_path / 'artifacts'
    root.mkdir()
    db = sqlite3.connect(root / 'fixture.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY, value TEXT)')
    db.execute('INSERT INTO events(value) VALUES(?)', ('committed WAL evidence',))
    db.commit()
    original = root / 'immutable.json'
    original.write_text(json.dumps({'absolute_original_path': str(root / 'prior.png')}))
    expected = sha256(original)
    output = tmp_path / 'export'
    result = export_evidence(root, output)
    assert sha256(original) == expected
    with zipfile.ZipFile(result['archive']) as archive:
        manifest = json.loads(archive.read('export_manifest.json'))
        for record in manifest['files']:
            import hashlib
            assert hashlib.sha256(archive.read(record['archive_path'])).hexdigest() == record['sha256']
        archive.extract('artifacts/fixture.sqlite3', tmp_path / 'restored')
        assert not any(name.endswith(('-wal', '-shm')) for name in archive.namelist())
    with sqlite3.connect(tmp_path / 'restored/artifacts/fixture.sqlite3') as restored:
        assert restored.execute('SELECT value FROM events').fetchone()[0] == 'committed WAL evidence'
        assert restored.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert manifest['source_code_included'] is False
    assert manifest['models_included'] is False
    assert manifest['path_map'][str(original)] == 'artifacts/immutable.json'
    with pytest.raises(ValueError, match='fresh output'):
        export_evidence(root, output)
    db.close()


def test_audit_model_activity_counts_failed_open_and_unmeasured_calls(tmp_path):
    secret = 'DO_NOT_EXPORT_PROMPT_IMAGE_OR_EVIDENCE_CONTENT'
    with CostJournal(tmp_path) as journal:
        for call_id in ('success', 'failed', 'later', 'uncertain', 'open'):
            journal.start('fixture', call_id, 'Qwen/exact-model', scope='inference',
                          metadata={'prompt': secret, 'image': secret})
        journal.finish('fixture', 'success', 10, measurements={'sampled_max_bytes': {
            'process_rss_bytes': 1024, 'mps_current_allocated_bytes': 2048,
            'mps_driver_allocated_bytes': 4096, 'cuda_allocated_bytes': 0,
            'system_available_bytes': 10**12, 'system_total_bytes': 2 * 10**12,
            'mps_recommended_max_bytes': 10**12, 'cuda_reserved_bytes': 99999,
            'cuda_peak_allocated_bytes': 99999}}, evidence=secret)
        journal.finish('fixture', 'failed', 0, outcome='failed', measurements={
            'sampled_max_bytes': {'process_rss_bytes': 2048,
                'mps_current_allocated_bytes': '9999', 'mps_driver_allocated_bytes': -1,
                'cuda_allocated_bytes': True}}, evidence=secret)
        journal.finish('fixture', 'later', 30)
        journal.finish('fixture', 'uncertain', outcome='uncertain', measurements={
            'sampled_max_bytes': None, 'after': {'process_rss_bytes': 10**12}})
        journal.start('fixture', 'load', 'Qwen/exact-model', scope='model_load')
        journal.finish('fixture', 'load', 3)
        journal.start('fixture', 'different-case', 'qwen/exact-model', scope='inference')
        journal.finish('fixture', 'different-case', outcome='failed')
        before = journal.connection.execute('SELECT count(*) FROM cost_events').fetchone()[0]
    report = audit(tmp_path)
    assert report['integrity_status'] == 'passed'
    assert secret not in json.dumps(report)
    assert 'records' not in report['costs']
    activity = report['costs']['model_activity']
    groups = {(g['model'], g['scope']): g for g in activity['groups']}
    assert len(groups) == 3
    inference = groups['Qwen/exact-model', 'inference']
    assert {key: inference[key] for key in ('started_records', 'completed_records', 'open_records')} == {
        'started_records': 5, 'completed_records': 4, 'open_records': 1}
    assert inference['finish_outcomes'] == {'succeeded': 2, 'failed': 1, 'uncertain': 1}
    assert inference['runtime_seconds'] == {'measured_records': 3, 'missing_finished_records': 1,
        'invalid_finished_records': 0, 'unmeasured_open_records': 1,
        'min': 0, 'median': 10, 'max': 30, 'sum': 40}
    memory = inference['sampled_memory']
    assert memory['finished_records_with_any_valid_usage_measurement'] == 2
    assert memory['unmeasured_open_records'] == 1
    assert set(memory['fields']) == {'process_rss_bytes', 'mps_current_allocated_bytes',
                                     'mps_driver_allocated_bytes', 'cuda_allocated_bytes'}
    assert memory['fields']['process_rss_bytes'] == {'max_bytes': 2048, 'measured_records': 2,
        'finished_records_without_valid_measurement': 2}
    assert memory['fields']['mps_driver_allocated_bytes']['max_bytes'] == 4096
    assert memory['fields']['mps_current_allocated_bytes']['measured_records'] == 1
    assert memory['fields']['cuda_allocated_bytes']['max_bytes'] == 0
    assert memory['fields']['cuda_allocated_bytes']['measured_records'] == 1
    assert groups['Qwen/exact-model', 'model_load']['runtime_seconds']['sum'] == 3
    missing = groups['qwen/exact-model', 'inference']
    assert missing['runtime_seconds']['measured_records'] == 0
    assert all(missing['runtime_seconds'][key] is None for key in ('min', 'median', 'max', 'sum'))
    assert all(field['max_bytes'] is None for field in missing['sampled_memory']['fields'].values())
    assert 'not necessarily a successful result' in activity['completed_definition']
    limitations = ' '.join(activity['limitations'])
    assert 'not native peaks' in limitations and 'may mix render profiles' in limitations
    with CostJournal(tmp_path, read_only=True) as journal:
        assert journal.connection.execute('SELECT count(*) FROM cost_events').fetchone()[0] == before


def test_audit_open_only_and_empty_cost_journals_do_not_invent_measurements(tmp_path):
    with CostJournal(tmp_path):
        pass
    assert audit(tmp_path)['costs']['model_activity']['groups'] == []
    with CostJournal(tmp_path) as journal:
        journal.start('fixture', 'open', 'Exact/model', scope='model_loading')
    group = audit(tmp_path)['costs']['model_activity']['groups'][0]
    assert (group['started_records'], group['completed_records'], group['open_records']) == (1, 0, 1)
    assert group['finish_outcomes'] == {}
    assert group['runtime_seconds']['sum'] is None
    assert group['runtime_seconds']['unmeasured_open_records'] == 1
    assert all(field['max_bytes'] is None for field in group['sampled_memory']['fields'].values())


@pytest.mark.parametrize('invalid', [True, -1, '10', float('nan'), float('inf')])
def test_audit_invalid_runtime_values_are_not_measured(invalid):
    group = _model_activity([{'start': {'model': 'Fixture', 'scope': 'inference'},
        'finish': {'runtime_seconds': invalid, 'outcome': 'failed', 'measurements': {}}}])['groups'][0]
    assert group['runtime_seconds']['invalid_finished_records'] == 1
    assert group['runtime_seconds']['measured_records'] == 0
    assert group['runtime_seconds']['sum'] is None
    json.dumps(group, allow_nan=False)


def test_audit_resource_statuses_distinguish_renderer_and_reasoning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kusama.resources import SessionBudget, load_config
    monkeypatch.setattr('kusama.resources.shutil.disk_usage',
        lambda path: SimpleNamespace(total=10**12, used=10**11, free=9 * 10**11))
    with SessionBudget(tmp_path, load_config()) as budget:
        budget.reserve('render-success', 'renderer')
        budget.settle('render-success', 'completed')
        budget.reserve('render-failure', 'renderer')
        budget.settle('render-failure', 'failed')
        budget.reserve('reasoning-failure', 'reasoning', max_tokens=10)
        budget.settle('reasoning-failure', 'failed', actual_tokens=2)
        budget.reserve('reasoning-open', 'reasoning', max_tokens=10)
        budget.reserve('reasoning-uncertain', 'reasoning', max_tokens=10)
        budget.settle('reasoning-uncertain', 'uncertain')
    report = audit(tmp_path)
    assert report['integrity_status'] == 'passed'
    assert report['resources']['renderer_attempts'] == 2
    assert report['resources']['calls_by_kind'] == {
        'renderer': {'total': 2, 'states': {'completed': 1, 'failed': 1}},
        'reasoning': {'total': 3, 'states': {'failed': 1, 'running': 1, 'uncertain': 1}}}
    assert report['resources']['states'] == {'completed': 1, 'failed': 2, 'running': 1, 'uncertain': 1}
    assert report['resources']['reasoning_tokens_charged_or_reserved'] == 22
