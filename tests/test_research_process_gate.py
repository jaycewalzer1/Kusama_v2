"""A software pass must not masquerade as a successful unmocked process."""
import importlib.util
import json
import os
from pathlib import Path

import pytest

from test_panel_runs import studio, request, runners

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/verify_research_process.py'
spec = importlib.util.spec_from_file_location('verify_research_process', SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def test_complete_scripted_panel_is_not_a_live_acceptance_pass(studio, monkeypatch):
    store, _ = studio
    entry = store.start(request(images_per_direction=2))
    research_runner, render_runner = runners(monkeypatch)
    store.execute(entry['run_id'], research_runner=research_runner, render_runner=render_runner)
    assert store.get(entry['run_id'])['status'] == 'completed'
    directory = store.directory(entry['run_id'])
    before = {p: p.read_bytes() for p in directory.rglob('*.json')}
    result = gate.verify(directory)
    assert result['status'] == 'failed'
    failures = {item['name'] for item in result['checks'] if not item['passed']}
    assert {'qwen_runtime_0', 'qwen_runtime_1', 'live_acquisition_completed', 'live_http_inspection'} <= failures
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert result['calls_dispatched_by_verifier'] == 0


def test_empty_terminal_run_is_not_full_process_success(studio):
    store, _ = studio
    entry = store.start(request())
    # Production planning and panel reporting; simulate an unresolved outcome.
    store.execute(entry['run_id'], research_runner=lambda *a, **kw: {'status': 'model_stop', 'hypotheses': []})
    result = gate.verify(store.directory(entry['run_id']))
    assert result['status'] == 'failed'
    failures = {item['name'] for item in result['checks'] if not item['passed']}
    assert {'panel_completed', 'committed_hypotheses_exist', 'research_to_render_handoff'} <= failures


@pytest.mark.skipif(not os.environ.get('KUSAMA_LIVE_RESEARCH'),
    reason='Live process acceptance requires a completed unmocked run and actual dashboard URL')
def test_completed_unmocked_research_process():
    directory = Path(os.environ['KUSAMA_LIVE_RESEARCH'])
    base_url = os.environ.get('KUSAMA_LIVE_DASHBOARD')
    assert base_url, 'KUSAMA_LIVE_DASHBOARD must point at the running production dashboard'
    result = gate.verify(directory, base_url=base_url)
    assert result['status'] == 'passed', [c for c in result['checks'] if not c['passed']]


@pytest.mark.parametrize('status', ['failed', 'partial', 'researching', 'rendering'])
def test_prior_live_success_cannot_hide_current_implementation_failure(tmp_path, status):
    identity = {'source.py': 'current'}
    for name, state, code in [('studio-good', 'completed', identity), ('studio-new', status, identity),
                               ('studio-old', 'failed', {'source.py': 'previous'})]:
        p = tmp_path / name
        p.mkdir()
        (p / 'request.json').write_text(json.dumps({'code_identity': code}))
        (p / 'manifest.json').write_text(json.dumps({'status': state}))
    outcomes = gate.current_implementation_outcomes(tmp_path / 'studio-good', identity)
    assert outcomes == [{'run': 'studio-good', 'status': 'completed'}, {'run': 'studio-new', 'status': status}]
    assert not all(item['status'] in ('completed', 'stopped') for item in outcomes)
