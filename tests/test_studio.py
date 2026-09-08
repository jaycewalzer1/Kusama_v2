import json
from pathlib import Path

from PIL import Image
import pytest
from fastapi.testclient import TestClient

from kusama.artifacts import sha256
from kusama.config import Config
from kusama.studio import Studio, read_projects
from kusama.dashboard import create_app
from kusama.studio_cli import main


INQUIRY = dict(question='How does scale change isolation?', stakes='Presence', position='Experiment', viewing_conditions='Screen')

class Backend:
    calls = 0
    def __init__(self, config): pass
    def prepare(self, on_progress): on_progress({'device': 'cpu', 'test_double': True})
    def generate(self, *args, **kwargs):
        self.on_inference_start()
        type(self).calls += 1
        return Image.new('RGB', (64, 64), 'red')
    def close(self): pass


def forbidden(config):
    pytest.fail('Must reuse saved output')

@pytest.fixture
def studio(tmp_path):
    Image.new('RGB', (64, 64), 'white').save(tmp_path / 'parent.png')
    Image.new('L', (64, 64), 255).save(tmp_path / 'mask.png')
    Backend.calls = 0
    with Studio(tmp_path / 'runs') as s:
        s.create('p', 'Project', INQUIRY, tmp_path / 'parent.png', 'test')
        s.start('p', 'e', Config(output_dir=s.root), max_calls=4, max_rounds=2)
        yield s


def batch(s, ids=('a', 'b', 'c'), parent='root', batch_id='one', calls=1):
    directory = s.root.parent
    artifact = s.get('p')['artifacts'][parent]
    directions = []
    for i, key in enumerate(ids):
        plan = dict(parent=artifact['path'], parent_sha256=artifact['sha256'], instruction=f'Change field {key}',
                    target='field', allowed_mask=str(directory / 'mask.png'), method='masked',
                    seed=23+i, max_calls=calls, visual_checks=['Inspect field'])
        (directory / f'{key}.json').write_text(json.dumps(plan))
        directions.append(dict(id=key, title=key, intended_change=f'Change {key}',
                               preservation_constraints=['Keep framing'], plan=f'{key}.json'))
    path = directory / f'{batch_id}.json'
    path.write_text(json.dumps(dict(id=batch_id, parent_artifact_id=parent, author='test',
                                    origin='development_example', directions=directions)))
    return path


def test_siblings_selection_and_continuation(studio):
    s = studio
    path = batch(s)
    first = s.propose('p', 'e', path)
    assert s.propose('p', 'e', path) == first
    with pytest.raises(ValueError): s.select('p', 'e', 'root', 'Early', 'test')
    assert s.run('p', 'e', Backend)['status'] == 'awaiting_selection'
    p = s.get('p')
    assert len(p['artifacts']) == 4 and Backend.calls == 3
    assert all(a['parent_artifact_id'] == 'root' for a in p['actions'].values())
    assert s.select('p', 'e', 'b:0', 'Continue this branch', 'test')['status'] == 'awaiting_proposals'
    s.decide('p', 'a:0', 'retain', 'Keep alternative', 'test')
    s.propose('p', 'e', batch(s, ('d',), 'b:0', 'two'))
    s.run('p', 'e', Backend)
    assert s.select('p', 'e', 'd:0', 'Done', 'test')['status'] == 'exhausted'
    p = s.get('p')
    assert p['artifacts']['d:0']['parent_artifact_id'] == 'b:0'
    assert len(p['artifacts']) == 5
    assert read_projects(s.root)[0]['episodes']['e']['model_calls'] == 4
    with pytest.raises(ValueError): s.propose('p', 'e', batch(s, ('f',), 'd:0', 'three'))
    s.close_episode('p', 'e', 'Enough', 'test', finish=True)
    with pytest.raises(ValueError): s.step('p', 'e', forbidden)


def test_frozen_inquiry_references_and_inputs(studio):
    s = studio
    ref = dict(id='ref', title='Source', uri='urn:test', provenance='Test fixture', permitted_uses='Tests', source_family='fixture')
    s.add_reference('p', ref)
    path = batch(s, ('a',))
    data = json.loads(path.read_text())
    data['directions'][0].update(source_ids=['ref'], relationship='Proposed contrast')
    path.write_text(json.dumps(data))
    s.propose('p', 'e', path)
    s.revise('p', INQUIRY | {'question': 'New question'}, 'Learning', 'test')
    with pytest.raises(ValueError): s.add_reference('p', ref | {'title': 'Changed'})
    (path.parent / 'a.json').unlink()
    s.run('p', 'e', Backend)
    action = s.get('p')['actions']['a']
    assert action['inquiry']['question'] == INQUIRY['question']
    assert action['inquiry_version'] == 1 and action['references'][0]['title'] == 'Source'
    s.observe('p', 'a:0', 'An observation', 'test', 'development_example')
    assert 'observations' not in action


@pytest.mark.parametrize('bad', ['budget', 'parent', 'reference'])
def test_invalid_batch_has_no_actions_or_calls(studio, bad):
    path = batch(studio, calls=2 if bad == 'budget' else 1)
    data = json.loads(path.read_text())
    if bad == 'parent': data['parent_artifact_id'] = 'missing'
    if bad == 'reference': data['directions'][-1].update(source_ids=['missing'], relationship='Test')
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): studio.propose('p', 'e', path)
    assert not studio.get('p')['actions'] and Backend.calls == 0


def test_stop_and_tamper(studio):
    s = studio
    s.propose('p', 'e', batch(s, ('a',)))
    s.close_episode('p', 'e', 'Stop now', 'test')
    assert s.run('p', 'e', forbidden)['status'] == 'stopped'
    assert s.get('p')['actions']['a']['status'] == 'cancelled'
    Path(s.get('p')['artifacts']['root']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError): s.decide('p', 'root', 'retain', 'Reason', 'test')


def test_recover_after_child_completed_before_project_commit(studio, monkeypatch):
    s = studio
    s.propose('p', 'e', batch(s, ('a',)))
    save = s._save
    def fail(project, event, details):
        if event == 'action_finished': raise OSError('projection interrupted')
        return save(project, event, details)
    monkeypatch.setattr(s, '_save', fail)
    with pytest.raises(OSError): s.step('p', 'e', Backend)
    monkeypatch.setattr(s, '_save', save)
    assert s.step('p', 'e', forbidden)['status'] == 'awaiting_selection'
    assert Backend.calls == 1


def test_postprocessing_recovery(studio, monkeypatch):
    import kusama.execution as execution
    s = studio
    s.propose('p', 'e', batch(s, ('a',)))
    finish = execution._finish
    def broken(*args): raise OSError('disk')
    monkeypatch.setattr(execution, '_finish', broken)
    s.run('p', 'e', Backend)
    monkeypatch.setattr(execution, '_finish', finish)
    assert s.run('p', 'e', forbidden)['status'] == 'awaiting_selection'
    assert Backend.calls == 1


def test_read_only_and_cli(studio, tmp_path, capsys):
    absent = tmp_path / 'absent'
    assert read_projects(absent) == [] and not absent.exists()
    assert main(['--artifacts', str(studio.root), 'show', '--project', 'p']) == 0
    assert json.loads(capsys.readouterr().out)['id'] == 'p'
    client = TestClient(create_app(studio.root), base_url='http://localhost')
    assert client.get('/api/projects').json()['projects'][0]['id'] == 'p'
    assert client.get('/api/projects/p').json()['current_artifact_id'] == 'root'
    assert client.post('/api/projects/p').status_code == 405
    assert client.get('/api/projects/p/artifacts/root').status_code == 200
    assert client.get('/api/projects/p/artifacts/missing').status_code == 404

@pytest.mark.parametrize('limit', ['max_rounds', 'max_actions'])
def test_independent_iteration_limits(studio, limit):
    s = studio
    s.close_episode('p', 'e', 'New limits', 'test')
    s.start('p', 'limited', Config(output_dir=s.root), max_calls=10, **{limit: 1})
    s.propose('p', 'limited', batch(s, ('a',)))
    s.run('p', 'limited', Backend)
    assert s.select('p', 'limited', 'a:0', 'Continue', 'test')['status'] == 'exhausted'
    assert s.ledger.call_count(s.get('p')['episodes']['limited']['job_id']) == 1


@pytest.mark.parametrize('point', ['reserved', 'raw'])
def test_process_exit_recovers_studio_state(studio, point):
    import subprocess
    import sys
    s = studio
    s.propose('p', 'e', batch(s, ('a',)))
    program = r'''
import os, sys
from pathlib import Path
from PIL import Image
from kusama.studio import Studio
import kusama.execution as execution
point = sys.argv[2]
save = execution.save_image
def crash_save(image, path):
    save(image, path)
    if point == 'raw' and path.name == 'attempt_0_raw.png': os._exit(77)
execution.save_image = crash_save
class Backend:
    def __init__(self, config): pass
    def prepare(self, on_progress): on_progress({'device':'cpu','test_double':True})
    def generate(self, *args, **kwargs):
        self.on_inference_start()
        if point == 'reserved': os._exit(77)
        return Image.new('RGB',(64,64),'red')
    def close(self): pass
with Studio(Path(sys.argv[1])) as studio:
    studio.step('p', 'e', Backend)
'''
    result = subprocess.run([sys.executable, '-c', program, str(s.root), point], capture_output=True, text=True, timeout=20)
    assert result.returncode == 77, result.stderr
    assert s.get('p')['actions']['a']['status'] == 'running'
    s.step('p', 'e', forbidden)
    p = s.get('p')
    assert s.ledger.call_count(p['episodes']['e']['job_id']) == 1
    assert p['actions']['a']['status'] == ('uncertain' if point == 'reserved' else 'needs_review')
    assert len(p['artifacts']) == (1 if point == 'reserved' else 2)
