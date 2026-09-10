"""Command discovery stays cheap and forwarding preserves existing semantics."""
import subprocess
import sys

import pytest

from kusama import cli


def test_help_does_not_import_application_dependencies(tmp_path):
    result = subprocess.run(
        [sys.executable, '-c',
         'import sys; from kusama.cli import main; '
         'main(["--help"]); '
         'assert not ({"torch", "numpy", "pydantic", "PIL", "kusama.resources"} & sys.modules.keys())'],
        cwd=tmp_path, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Commands: doctor,' in result.stdout
    assert not list(tmp_path.iterdir())


def test_run_and_resume_forward_arguments_and_exit_status(monkeypatch):
    from kusama import episodes

    calls = []
    monkeypatch.setattr(episodes, 'main', lambda args: calls.append(args) or 7)
    for command in ('run', 'resume'):
        argv = [command, '--root', 'a directory with spaces', '--resume']
        assert cli.main(argv) == 7
        assert argv[0] == command
    assert calls == [['--root', 'a directory with spaces', '--resume']] * 2


def test_unknown_command_does_not_import_a_handler(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Unknown commands must not load arbitrary modules')

    monkeypatch.setattr(cli, 'import_module', forbidden)
    with pytest.raises(SystemExit, match='Unknown command: no-such-command'):
        cli.main(['no-such-command'])
