"""Crash-boundary contracts exercise a separate process, never real inference."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.config import Config
from kusama.controlled import run
from kusama.ledger import Ledger, SpecMismatch


@pytest.fixture
def source(tmp_path):
    Image.new('RGB', (64, 64), 'white').save(tmp_path / 'parent.png')
    Image.new('L', (64, 64), 255).save(tmp_path / 'mask.png')
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(dict(parent='parent.png', parent_sha256=sha256(tmp_path / 'parent.png'),
        instruction='Make the field red', target='field', allowed_mask='mask.png', method='masked',
        seed=23, max_calls=2, visual_checks=['Does it work?'])))
    return path, Config(output_dir=tmp_path / 'runs')


class Backend:
    calls = 0
    def __init__(self, config):
        self.config = config
    def prepare(self, on_progress):
        on_progress({'device': 'cpu', 'test_double': True})
    def generate(self, *args, **kwargs):
        self.on_inference_start()
        type(self).calls += 1
        return Image.new('RGB', (64, 64), 'red')
    def close(self):
        pass


def forbidden(config):
    pytest.fail('Recovery must not construct an inference backend')


def test_stable_job_reuses_output_and_rejects_changed_spec(source):
    path, config = source
    directory = run(path, config, Backend, job_id='stable')
    before = sha256(directory / 'attempt_0_raw.png')
    assert run(path, config, forbidden, job_id='stable') == directory
    assert run(resume_dir=directory, backend_factory=forbidden) == directory
    assert sha256(directory / 'attempt_0_raw.png') == before
    with pytest.raises(SpecMismatch):
        run(path, replace(config, steps=51), forbidden, job_id='stable')
    assert json.loads((directory / 'manifest.json').read_text())['model_calls'] == 1


def test_resume_uses_snapshots_and_rejects_changed_saved_bytes(source):
    path, config = source
    directory = run(path, config, Backend)
    (path.parent / 'parent.png').unlink()
    (path.parent / 'mask.png').unlink()
    assert run(resume_dir=directory, backend_factory=forbidden) == directory
    Image.new('RGB', (64, 64), 'blue').save(directory / 'attempt_0_result.png')
    with pytest.raises(ValueError, match='missing or changed'):
        run(resume_dir=directory, backend_factory=forbidden)


def test_postprocessing_failure_recovers_without_new_call(source, monkeypatch):
    import kusama.execution as execution
    path, config = source
    finish = execution._finish
    def broken(*args):
        raise OSError('comparison storage failed')
    monkeypatch.setattr(execution, '_finish', broken)
    directory = run(path, config, Backend)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'postprocessing_failed'
    raw_hash = sha256(directory / 'attempt_0_raw.png')
    monkeypatch.setattr(execution, '_finish', finish)
    run(resume_dir=directory, backend_factory=forbidden)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_review' and manifest['model_calls'] == 1
    assert manifest['attempts'][0]['recovered_saved_output']
    assert sha256(directory / 'attempt_0_raw.png') == raw_hash


CRASH_PROGRAM = r'''
import os, sys
from pathlib import Path
from PIL import Image
from kusama.config import Config
from kusama.controlled import run
import kusama.execution as execution
point = sys.argv[3]
save = execution.save_image
def save_and_exit(image, path):
    save(image, path)
    if (point == 'raw' and path.name == 'attempt_0_raw.png') or (point == 'result' and path.name == 'attempt_0_result.png'):
        os._exit(77)
execution.save_image = save_and_exit
class Backend:
    def __init__(self, config): pass
    def prepare(self, on_progress):
        on_progress({'device':'cpu','test_double':True})
        if point == 'before': os._exit(77)
    def generate(self, *args, **kwargs):
        self.on_inference_start()
        if point == 'reserved': os._exit(77)
        return Image.new('RGB',(64,64),'red')
    def close(self): pass
run(Path(sys.argv[1]), Config(output_dir=Path(sys.argv[2])), Backend, job_id='crash')
'''


@pytest.mark.parametrize('point', ['before', 'reserved', 'raw', 'result'])
def test_abrupt_process_exit_at_inference_boundaries(source, point):
    path, config = source
    child = subprocess.run([sys.executable, '-c', CRASH_PROGRAM, str(path), str(config.output_dir), point],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 77, child.stderr
    directory = config.output_dir.resolve() / 'crash'
    assert (directory / 'job.json').exists()
    result = run(resume_dir=directory, backend_factory=Backend if point == 'before' else forbidden)
    manifest = json.loads((result / 'manifest.json').read_text())
    assert manifest['model_calls'] == 1
    if point == 'reserved':
        assert manifest['status'] == 'uncertain'
        assert manifest['attempts'][0]['status'] == 'uncertain'
        assert not list(directory.glob('*raw.png'))
    else:
        assert manifest['status'] == 'needs_review'
        assert (directory / 'attempt_0_result.png').exists()
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.call_count('crash') == 1
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        assert sum(event['type'] == 'call_reserved' for event in events) == 1


def test_parent_budget_is_atomic_across_controlled_children(source):
    path, config = source
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        ledger.create_job('suite', {'test': True}, 1, config.output_dir / 'suite')
        run(path, config, Backend, job_id='one', ledger=ledger, parent_job_id='suite')
        directory = run(path, config, Backend, job_id='two', ledger=ledger, parent_job_id='suite')
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert manifest['model_calls'] == 0
        assert 'budget exhausted' in manifest['attempts'][0]['error']
        assert ledger.call_count('suite') == ledger.call_count('one') == 1
        assert ledger.call_count('two') == 0


def test_parent_snapshot_preserves_original_png_encoding_and_metadata(source):
    from PIL.PngImagePlugin import PngInfo

    path, config = source
    metadata = PngInfo()
    metadata.add_text('Description', 'Approved source with nondefault encoding')
    parent = path.parent / 'parent.png'
    Image.new('RGB', (64, 64), 'white').save(parent, compress_level=0, pnginfo=metadata)
    data = json.loads(path.read_text())
    data['parent_sha256'] = sha256(parent)
    path.write_text(json.dumps(data))
    original = parent.read_bytes()
    directory = run(path, config, Backend)
    assert (directory / 'parent.png').read_bytes() == original
    assert json.loads((directory / 'execution_plan.json').read_text())['parent_sha256'] == data['parent_sha256']
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_review'
    assert manifest['source_parent_sha256'] == data['parent_sha256']
    assert run(resume_dir=directory, backend_factory=forbidden) == directory


def test_unreserved_orphan_raw_is_rejected_before_adoption(source):
    path, config = source
    child = subprocess.run([sys.executable, '-c', CRASH_PROGRAM, str(path), str(config.output_dir), 'before'],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 77, child.stderr
    directory = config.output_dir / 'crash'
    Image.new('RGB', (64, 64), 'blue').save(directory / 'attempt_0_raw.png')
    with pytest.raises(ValueError, match='no inference reservation'):
        run(resume_dir=directory, backend_factory=forbidden)
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.call_count('crash') == 0
        assert 'attempt_0_raw.png' not in {record['path'] for record in ledger.artifacts('crash')}
        assert ledger.get_job('crash')['status'] != 'needs_review'


@pytest.mark.parametrize('name', ['allowed_mask.png', 'plan.json', 'execution_plan.json'])
def test_partial_snapshot_cannot_adopt_changed_unregistered_inputs(source, monkeypatch, name):
    import kusama.execution as execution

    path, config = source
    snapshot = execution._snapshot

    def interrupted(*args):
        snapshot(*args)
        raise KeyboardInterrupt('stopped before registering snapshot artifacts')

    monkeypatch.setattr(execution, '_snapshot', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(path, config, forbidden, job_id='partial')
    directory = config.output_dir / 'partial'
    target = directory / name
    if name.endswith('.png'):
        Image.new('L', (64, 64), 0).save(target)
    else:
        data = json.loads(target.read_text())
        data['instruction'] = 'A different, unapproved instruction'
        target.write_text(json.dumps(data))
    monkeypatch.setattr(execution, '_snapshot', snapshot)
    with pytest.raises(ValueError, match='differs from frozen'):
        run(resume_dir=directory, backend_factory=forbidden)
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.call_count('partial') == 0
        assert not ledger.get_job('partial')['payload'].get('snapshot_complete')


def test_orphan_result_must_match_computation_from_reserved_raw(source):
    path, config = source
    child = subprocess.run([sys.executable, '-c', CRASH_PROGRAM, str(path), str(config.output_dir), 'result'],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 77, child.stderr
    directory = config.output_dir / 'crash'
    Image.new('RGB', (64, 64), 'blue').save(directory / 'attempt_0_result.png')
    run(resume_dir=directory, backend_factory=forbidden)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'postprocessing_failed'
    assert manifest['model_calls'] == 1
    assert 'differs from frozen computation' in manifest['attempts'][0]['error']
    assert 'candidate' not in manifest


def test_unregistered_verification_is_recomputed_before_adoption(source, monkeypatch):
    import kusama.controlled as controlled

    path, config = source
    child = subprocess.run([sys.executable, '-c', CRASH_PROGRAM, str(path), str(config.output_dir), 'raw'],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 77, child.stderr
    directory = config.output_dir / 'crash'
    verification = directory / 'attempt_0_verification.json'
    verification.write_text(json.dumps({'status': 'pass', 'image_sha256': sha256(directory / 'attempt_0_raw.png'),
                                        'injected': True}))
    monkeypatch.setattr(controlled, 'verify', lambda *args: ({'status': 'fail', 'recomputed': True}, None))
    run(resume_dir=directory, backend_factory=forbidden)
    manifest = json.loads((directory / 'manifest.json').read_text())
    saved = json.loads(verification.read_text())
    assert manifest['status'] == 'rejected'
    assert manifest['model_calls'] == 1
    assert saved['status'] == 'fail' and saved['recomputed']
    assert 'injected' not in saved
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        record = next(record for record in ledger.artifacts('crash') if record['path'] == verification.name)
        assert record['sha256'] == sha256(verification)


def test_registered_verification_survives_later_reporting_failure(source, monkeypatch):
    import kusama.controlled as controlled
    import kusama.execution as execution

    path, config = source
    write = execution.write_json

    def fail_review(target, value):
        if target.name == 'review.json':
            raise OSError('review write interrupted')
        return write(target, value)

    monkeypatch.setattr(controlled, 'verify', lambda *args: ({'status': 'pass', 'registered': True}, None))
    monkeypatch.setattr(execution, 'write_json', fail_review)
    directory = run(path, config, Backend)
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'postprocessing_failed'
    original_hash = sha256(directory / 'attempt_0_verification.json')

    def no_verification(*args):
        pytest.fail('Registered verification must be reused')

    monkeypatch.setattr(execution, 'write_json', write)
    monkeypatch.setattr(controlled, 'verify', no_verification)
    run(resume_dir=directory, backend_factory=forbidden)
    assert sha256(directory / 'attempt_0_verification.json') == original_hash
    assert json.loads((directory / 'manifest.json').read_text())['status'] == 'needs_review'
