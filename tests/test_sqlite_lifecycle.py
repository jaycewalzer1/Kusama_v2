"""Short-lived evidence readers/writers release handles, including on failure."""
from contextlib import closing
import json
import sqlite3

import pytest

from kusama.cli import export_evidence
from kusama.resource_snapshot import session_snapshot
from kusama.resources import load_config
from kusama.retrieval import build_index
from test_retrieval import FixtureEncoder, tiny_index as tiny_index


@pytest.fixture
def connections(monkeypatch):
    opened = []
    connect = sqlite3.connect

    class TrackedConnection(sqlite3.Connection):
        closed = False

        def close(self):
            self.closed = True
            super().close()

    def tracked(*args, **kwargs):
        connection = connect(*args, **kwargs, factory=TrackedConnection)
        opened.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, 'connect', tracked)
    return opened


@pytest.mark.parametrize('valid', [True, False])
def test_session_snapshot_closes_on_success_and_failure(tmp_path, connections, valid):
    with closing(sqlite3.connect(tmp_path / 'resources.sqlite3')) as db, db:
        db.execute('CREATE TABLE session(id, config, started_at, deadline)')
        if valid:
            db.execute('INSERT INTO session VALUES(1, ?, ?, ?)', ('{}', 'start', 'end'))
    connections.clear()
    if valid:
        assert session_snapshot(tmp_path) == {'config': {}, 'started_at': 'start', 'deadline': 'end'}
    else:
        with pytest.raises(ValueError, match='not initialized'):
            session_snapshot(tmp_path)
    assert len(connections) == 1 and connections[0].closed




def test_export_closes_source_and_backup(tmp_path, connections):
    root = tmp_path / 'artifacts'
    root.mkdir()
    with closing(sqlite3.connect(root / 'evidence.sqlite3')) as db, db:
        db.execute('CREATE TABLE evidence(value)')
        db.execute('INSERT INTO evidence VALUES(42)')
    connections.clear()
    result = export_evidence(root, tmp_path / 'export')
    assert result['files'] == 1
    assert len(connections) == 2 and all(db.closed for db in connections)


def test_retrieval_closes_query_and_committed_index_database(tiny_index, tmp_path, connections):  # noqa: F811 - imported pytest fixture
    index, store, _ = tiny_index
    assert index.query('enclosure')['channels']['lexical']
    assert len(connections) == 1 and connections[0].closed
    connections.clear()
    rebuilt = build_index(store.root, tmp_path / 'rebuilt', config=load_config(), encoder=FixtureEncoder())
    assert connections and all(db.closed for db in connections)
    # The write transaction must still commit before hashes are frozen.
    with closing(sqlite3.connect(rebuilt / 'lexical.sqlite3')) as db:
        assert db.execute('SELECT count(*) FROM documents').fetchone()[0] == len(index.documents)
    manifest = json.loads((rebuilt / 'manifest.json').read_text())
    assert manifest['files'] == index.manifest['files']
