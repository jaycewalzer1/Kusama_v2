import csv
import json
from pathlib import Path
import pytest
from PIL import Image
from kusama import benchmark
from kusama.benchmark import run_suite,load_suite, summarize, contact_sheet
from kusama.artifacts import sha256
from kusama.config import Config
from kusama.controlled import run


def suite(tmp_path):
    Image.new('RGB',(64,64),'white').save(tmp_path/'parent.png')
    Image.new('L',(64,64),255).save(tmp_path/'mask.png')
    plan=dict(parent='parent.png',parent_sha256=sha256(tmp_path/'parent.png'),instruction='edit',target='object',
              allowed_mask='mask.png',method='masked',seed=23,max_calls=1,visual_checks=['Natural?'])
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    path=tmp_path/'suite.json';path.write_text(json.dumps(dict(name='test',max_calls=1,cases=[{'id':'a','plan':'plan.json'},{'id':'b','plan':'plan.json'}])))
    return path


class Backend:
    def __init__(self,config):self.config=config
    def prepare(self,on_progress):on_progress({'test_double':True})
    def generate(self,*args,**kwargs):
        self.on_inference_start();return Image.new('RGB',(64,64),'red')
    def close(self):pass


def test_global_budget_and_reports(tmp_path):
    path=suite(tmp_path)
    def runner(path,config,call_guard,**kwargs):return run(path,config,Backend,call_guard,**kwargs)
    p=run_suite(path,Config(output_dir=tmp_path/'runs'),runner)
    report=json.loads((p/'benchmark.json').read_text())
    assert report['model_calls']==1
    assert report['cases'][1]['status']=='skipped_budget'
    assert (p/'results.csv').exists() and (p/'comparison.png').exists()
    assert report['summary']['verification_counts']['not_requested']==1


def test_suite_prevalidates_all_plans(tmp_path):
    path=suite(tmp_path);s=json.loads(path.read_text());s['cases'][1]['plan']='missing.json';path.write_text(json.dumps(s))
    with pytest.raises(OSError):load_suite(path)


def test_cleanup_failure_retains_raw_result_and_reports(tmp_path):
    class CleanupFailure(Backend):
        def close(self):
            raise RuntimeError('release failed')

    path = suite(tmp_path)
    def runner(path, config, call_guard, **kwargs):
        return run(path, config, CleanupFailure, call_guard, **kwargs)
    directory = run_suite(path, Config(output_dir=tmp_path/'runs'), runner)
    report = json.loads((directory/'benchmark.json').read_text())
    case = report['cases'][0]
    manifest = json.loads((Path(case['run'])/'manifest.json').read_text())
    attempt = manifest['attempts'][-1]
    assert attempt['status'] == 'postprocessing_failed' and 'verification' not in attempt
    assert attempt['error'] == 'RuntimeError: release failed'
    assert attempt['cleanup_error'] == 'release failed'
    assert report['model_calls'] == 1
    assert report['reporting_status'] == 'completed'
    assert report['summary']['verification_counts']['unavailable'] == 2
    with (directory/'results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]['attempt_status'] == 'postprocessing_failed'
    assert rows[0]['attempt_error'] == 'RuntimeError: release failed'
    assert rows[0]['verification'] == 'unavailable'
    assert (directory/'report.md').exists()
    assert sha256(Path(case['run'])/'attempt_0_raw.png') == manifest['artifacts']['attempt_0_raw.png']


def test_summary_preserves_metrics_when_cleanup_left_verification_missing(tmp_path):
    run_dir = tmp_path/'partial'; run_dir.mkdir()
    manifest = {'model_calls': 1, 'runtime_seconds': 1.5, 'attempts': [
        {'status': 'failed', 'error': 'RuntimeError: cleanup failed',
         'preservation': {'outside_changed_pixels': 0, 'total_changed_pixels': 80},
         'effective_preservation': {'outside_changed_pixels': 0}}]}
    (run_dir/'manifest.json').write_text(json.dumps(manifest))
    report = {'name': 'partial', 'status': 'completed', 'model_calls': 1, 'max_calls': 1,
              'cases': [{'id': 'partial', 'status': 'rejected', 'run': str(run_dir)}]}
    rows = summarize(tmp_path, report)
    assert rows[0]['protected_changed'] == rows[0]['effective_protected_changed'] == 0
    assert rows[0]['edit_pixels'] == 80
    assert rows[0]['verification'] == 'unavailable'
    assert rows[0]['attempt_error'] == 'RuntimeError: cleanup failed'
    assert report['summary']['verification_counts']['unavailable'] == 1


def test_summary_does_not_reuse_prior_attempt_verification(tmp_path):
    run_dir = tmp_path/'partial'; run_dir.mkdir()
    manifest = {'model_calls': 2, 'attempts': [
        {'status': 'rejected', 'preservation': {'outside_changed_pixels': 0},
         'verification': {'status': 'pass', 'measured_height_ratio': .5}},
        {'status': 'failed', 'error': 'result preview failed'}]}
    (run_dir/'manifest.json').write_text(json.dumps(manifest))
    report = {'name': 'partial', 'status': 'completed_with_failures', 'model_calls': 2, 'max_calls': 2,
              'cases': [{'id': 'partial', 'status': 'rejected', 'run': str(run_dir)}]}
    rows = summarize(tmp_path, report)
    assert rows[0]['attempt_index'] == 1
    assert rows[0]['attempt_status'] == 'failed'
    assert rows[0]['verification'] == 'unavailable'
    assert rows[0]['height_ratio'] == '' and rows[0]['protected_changed'] == ''
    assert rows[0]['seconds'] == ''
    assert report['summary']['verification_counts']['pass'] == 0


def test_each_report_failure_recorded_after_terminal_metadata(tmp_path, monkeypatch):
    def runner(path, config, call_guard, **kwargs):
        raise RuntimeError('original execution failure')
    def failed_summary(directory, report):
        durable = json.loads((directory/'benchmark.json').read_text())
        assert durable['status'] == 'completed_with_failures'
        assert durable['runtime_seconds'] >= 0
        assert all(c['error'] == 'RuntimeError: original execution failure' for c in durable['cases'])
        raise ValueError('summary failed')
    def failed_preview(directory, report):
        durable = json.loads((directory/'benchmark.json').read_text())
        assert durable['reporting_errors'][0]['stage'] == 'summary'
        raise OSError('preview failed')
    monkeypatch.setattr(benchmark, 'summarize', failed_summary)
    monkeypatch.setattr(benchmark, 'contact_sheet', failed_preview)
    directory = run_suite(suite(tmp_path), Config(output_dir=tmp_path/'runs'), runner)
    report = json.loads((directory/'benchmark.json').read_text())
    assert report['status'] == 'completed_with_failures'
    assert report['reporting_status'] == 'completed_with_errors'
    assert report['reporting_errors'] == [
        {'stage': 'summary', 'error': 'ValueError: summary failed'},
        {'stage': 'contact_sheet', 'error': 'OSError: preview failed'}]


def test_report_failures_do_not_mask_original_interruption(tmp_path, monkeypatch):
    def runner(path, config, call_guard, **kwargs):
        raise KeyboardInterrupt('original interruption')
    def failed_summary(directory, report):
        raise RuntimeError('summary failed')
    def interrupted_preview(directory, report):
        raise SystemExit('preview interruption')
    monkeypatch.setattr(benchmark, 'summarize', failed_summary)
    monkeypatch.setattr(benchmark, 'contact_sheet', interrupted_preview)
    output_dir = tmp_path/'runs'
    with pytest.raises(KeyboardInterrupt, match='original interruption'):
        run_suite(suite(tmp_path), Config(output_dir=output_dir), runner)
    report = json.loads(next(output_dir.glob('*/benchmark.json')).read_text())
    assert report['status'] == report['cases'][0]['status'] == 'interrupted'
    assert report['error'] == 'KeyboardInterrupt: original interruption'
    assert report['reporting_errors'] == [
        {'stage': 'summary', 'error': 'RuntimeError: summary failed'},
        {'stage': 'contact_sheet', 'error': 'SystemExit: preview interruption'}]


def test_bad_case_preview_does_not_discard_other_case_panels(tmp_path):
    cases = []
    for name in ['broken', 'good']:
        run_dir = tmp_path/name; run_dir.mkdir()
        (run_dir/'manifest.json').write_text(json.dumps({'attempts': [{}]}))
        (run_dir/'plan.json').write_text('{}')
        Image.new('RGB', (64, 64), 'white').save(run_dir/'parent.png')
        Image.new('RGB', (64, 64), 'red').save(run_dir/'attempt_0_result.png')
        cases.append({'id': name, 'run': str(run_dir)})
    (tmp_path/'broken'/'attempt_0_detections.png').write_text('damaged PNG')
    report = {'cases': cases}
    contact_sheet(tmp_path, report)
    assert report['reporting_errors'][0]['stage'] == 'contact_sheet_case'
    assert report['reporting_errors'][0]['case'] == 'broken'
    with Image.open(tmp_path/'comparison.png') as sheet:
        assert sheet.size == (960, 360)
        assert sheet.getpixel((330, 50)) == (255, 0, 0)
