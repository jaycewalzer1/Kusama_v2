"""Shared immutable evidence and hashing primitives."""
import hashlib
import json
from pathlib import Path
from .artifacts import sha256


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    """Compatibility wrapper accepting both strings and paths."""
    return sha256(Path(path))


def immutable_json(path, value):
    """Create once; an identical repeat is idempotent and changed bytes fail."""
    path = Path(path)
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x') as stream:
            stream.write(data)
    except FileExistsError:
        if path.read_text() != data:
            raise ValueError(f'Immutable artifact already differs: {path}')
    return path


