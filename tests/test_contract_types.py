"""Keep the dashboard declarations synchronized with the server contracts."""
from pathlib import Path
import subprocess
import sys


def test_dashboard_contracts_match_python_schemas():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / 'scripts/generate_contract_types.py'), '--check'],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
