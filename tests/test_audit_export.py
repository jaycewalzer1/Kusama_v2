"""Persistence checks use local synthetic records, never model or human evidence."""
import json
from pathlib import Path
import sqlite3
import zipfile

from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.audit import audit
from kusama.cli import export_evidence
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
