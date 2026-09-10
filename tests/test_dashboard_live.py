"""Execute front-end polling behavior against controlled DOM/HTTP fixtures."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_live_progress_preserves_development_drafts_and_selection():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is required for the dashboard JavaScript behavior test')
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([node, str(root / 'tests' / 'dashboard_live_check.cjs')],
                            cwd=root, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
