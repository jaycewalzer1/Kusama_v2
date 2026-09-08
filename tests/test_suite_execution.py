import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.benchmark import run_suite
from kusama.config import Config
from kusama.ledger import Ledger


def source(tmp_path):
    Image.new('RGB', (64, 64), 'white').save(tmp_path / 'parent.png')
    Image.new('L', (64, 64), 255).save(tmp_path / 'mask.png')
    plan = dict(parent='parent.png', parent_sha256=sha256(tmp_path / 'parent.png'), instruction='Red field',
        target='field', allowed_mask='mask.png', method='masked', seed=23, max_calls=1, visual_checks=['Compelling?'])
    (tmp_path / 'plan.json').write_text(json.dumps(plan))
    suite = tmp_path / 'suite.json'
    suite.write_text(json.dumps(dict(name='Restart fixture', max_calls=2,
        cases=[{'id': 'first', 'plan': 'plan.json'}, {'id': 'second', 'plan': 'plan.json'}])))
    return suite, Config(output_dir=tmp_path / 'runs')


class Backend:
    def __init__(self, config): pass
    def prepare(self, on_progress): on_progress({'test_double': True})
    def generate(self, *args, **kwargs):
        self.on_inference_start()
        return Image.new('RGB', (64, 64), 'red')
    def close(self): pass


def forbidden(config):
    pytest.fail('Completed suite must not construct a backend')


def test_repeat_and_resume_reuse_registered_children(tmp_path):
    path, config = source(tmp_path)
    directory = run_suite(path, config, job_id='stable-suite', backend_factory=Backend)
    initial = json.loads((directory / 'benchmark.json').read_text())
    raw_hashes = {case['job_id']: sha256(Path(case['run']) / 'attempt_0_raw.png') for case in initial['cases']}
    assert run_suite(path, config, job_id='stable-suite', backend_factory=forbidden) == directory
    assert run_suite(resume_dir=directory, backend_factory=forbidden) == directory
    final = json.loads((directory / 'benchmark.json').read_text())
    assert final['status'] == 'completed' and final['model_calls'] == 2
    assert raw_hashes == {case['job_id']: sha256(Path(case['run']) / 'attempt_0_raw.png') for case in final['cases']}


CRASH = r'''
import os,sys
from pathlib import Path
from PIL import Image
from kusama.config import Config
from kusama.benchmark import run_suite
import kusama.execution as execution
point=sys.argv[3]
save=execution.save_image
def save_and_exit(image,path):
    save(image,path)
    if point=='raw' and path.name=='attempt_0_raw.png': os._exit(77)
execution.save_image=save_and_exit
class Backend:
    def __init__(self,config):pass
    def prepare(self,on_progress):on_progress({'test_double':True})
    def generate(self,*args,**kwargs):
        self.on_inference_start()
        if point=='reserved':os._exit(77)
        return Image.new('RGB',(64,64),'red')
    def close(self):pass
run_suite(Path(sys.argv[1]),Config(output_dir=Path(sys.argv[2])),job_id='crash-suite',backend_factory=Backend)
'''


@pytest.mark.parametrize('point', ['raw', 'reserved'])
def test_killed_suite_retains_child_link_and_global_budget(tmp_path, point):
    path, config = source(tmp_path)
    child = subprocess.run([sys.executable, '-c', CRASH, str(path), str(config.output_dir), point],
                           capture_output=True, text=True, timeout=20)
    assert child.returncode == 77, child.stderr
    directory = config.output_dir / 'crash-suite'
    interrupted = json.loads((directory / 'benchmark.json').read_text())
    assert interrupted['cases'][0]['run']
    with Ledger(config.output_dir / 'jobs.sqlite3') as ledger:
        assert ledger.call_count('crash-suite') == 1
        assert ledger.get_job(interrupted['cases'][0]['job_id'])['parent_id'] == 'crash-suite'
    # All remaining sources must come from snapshots, including not-yet-run children.
    path.unlink(); (tmp_path / 'parent.png').unlink(); (tmp_path / 'mask.png').unlink()
    run_suite(resume_dir=directory, backend_factory=Backend)
    final = json.loads((directory / 'benchmark.json').read_text())
    assert final['model_calls'] == 2
    assert final['cases'][0]['status'] == ('needs_review' if point == 'raw' else 'uncertain')
    assert final['cases'][1]['status'] == 'needs_review'
    assert final['status'] == ('completed' if point == 'raw' else 'completed_with_failures')


def test_failed_reports_recover_without_inference(tmp_path, monkeypatch):
    from kusama import benchmark
    path, config = source(tmp_path)
    original = benchmark.summarize
    def fail(*args): raise OSError('report file unavailable')
    monkeypatch.setattr(benchmark, 'summarize', fail)
    directory = run_suite(path, config, backend_factory=Backend)
    first = json.loads((directory / 'benchmark.json').read_text())
    assert first['status'] == 'completed' and first['reporting_status'] == 'completed_with_errors'
    monkeypatch.setattr(benchmark, 'summarize', original)
    run_suite(resume_dir=directory, backend_factory=forbidden)
    final = json.loads((directory / 'benchmark.json').read_text())
    assert final['model_calls'] == 2
    assert final['reporting_status'] == 'completed'
    assert final['previous_reporting_errors'][0][0]['stage'] == 'summary'


def test_ancestor_budget_exhaustion_is_not_reported_as_completed(tmp_path):
    path, config = source(tmp_path)
    spec = json.loads(path.read_text())
    spec.update(max_calls=1, cases=spec['cases'][:1])
    path.write_text(json.dumps(spec))
    plan_path = tmp_path / 'plan.json'
    plan = json.loads(plan_path.read_text())
    plan.update(max_calls=2, fallbacks=['masked'])
    plan_path.write_text(json.dumps(plan))
    class NoChange(Backend):
        def generate(self, *args, **kwargs):
            self.on_inference_start()
            return Image.new('RGB', (64, 64), 'white')
    directory = run_suite(path, config, backend_factory=NoChange)
    report = json.loads((directory / 'benchmark.json').read_text())
    assert report['model_calls'] == 1
    assert report['cases'][0]['status'] == 'budget_exhausted'
    assert report['status'] == 'completed_with_failures'


def test_resume_interruption_marks_current_case_not_last_saved_case(tmp_path):
    path, config = source(tmp_path)
    directory = run_suite(path, config, backend_factory=Backend)
    def stop(*args, **kwargs):
        raise KeyboardInterrupt('interrupt first resumed child')
    with pytest.raises(KeyboardInterrupt):
        run_suite(resume_dir=directory, runner=stop)
    report = json.loads((directory / 'benchmark.json').read_text())
    assert report['status'] == 'interrupted'
    assert [case['status'] for case in report['cases']] == ['interrupted', 'needs_review']
