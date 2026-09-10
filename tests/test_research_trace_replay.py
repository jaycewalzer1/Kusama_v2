"""Built last: regression on original recorded actions, never replacement responses."""
from pathlib import Path
from copy import deepcopy
import json
from kusama.research_replay import replay_trace

TRACE=Path(__file__).parent/'fixtures/research/trace_16_step.json'


def test_recorded_16_steps_rejected_by_new_rules():
    trace=json.loads(TRACE.read_text());before=deepcopy(trace)
    report=replay_trace(trace)
    assert trace==before
    assert report['result']=='PASS' and report['recorded_run_acceptance']=='FAIL'
    by_step={r['step']:r for r in report['steps']}
    for index in (5,8,9,11,12,13,15): assert by_step[index]['status']=='rejected',by_step[index]
    assert any('relevance gate' in e for e in by_step[3]['reasons'])
    assert any('grounding' in e or 'entailment' in e for e in by_step[16]['reasons'])
    assert any('unchanged since step' in e for e in by_step[9]['reasons'])
    assert report['causal_replay']['steps_executed']<16
    assert report['inference_calls']==report['network_requests']==report['render_calls']==0
    assert all(s['action']['reason'] is None for s in trace['steps'])
    assert all(any('nonempty reason' in e for e in s['reasons']) for s in report['steps'])
