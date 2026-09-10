"""Read an existing authorization without initializing or extending it."""
from contextlib import closing
import json
from pathlib import Path
import sqlite3


def session_snapshot(root):
    path=Path(root).resolve()/'resources.sqlite3'
    if not path.is_file():raise ValueError('Existing authorized resource session required; inspection cannot create or extend one')
    with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row
        row=db.execute('SELECT config,started_at,deadline FROM session WHERE id=1').fetchone()
        if row is None:raise ValueError('Resource session is not initialized')
        return {'config':json.loads(row['config']),'started_at':row['started_at'],'deadline':row['deadline']}


