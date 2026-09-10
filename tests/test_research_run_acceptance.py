"""Incident-shaped acceptance runs through real ledgers; no artistic judgments.

The model and source boundaries are substitutes here. Processor-backed panel runs
are in test_research_processor_acceptance.py and require the pinned local cache.
"""
import json

import pytest

from kusama import research as r
from kusama.ledger import Ledger
from test_research import setup, Director, Sources, action, hypothesis


class IncidentSources(Sources):
    def search(self, query, **kwargs):
        return super().search(query, **kwargs) if query == 'lead' else {'sources': []}


def execute(setup, backend=IncidentSources):
    output, _, request, _, _ = setup
    r.plan_research(output, request)
    return r.run_research(output, director_factory=Director, sources_factory=backend)


def acquired_then_idle(actions):
    actions.extend([action('search', query='lead'), action('open', source_id='s0')])
    actions.extend(action('search', mode='remote', provider='artic', query=f'empty {i}') for i in range(6))


def test_acquired_reference_gets_checkpoint_before_stagnation_and_can_commit(setup):
    output, _, request, calls, actions = setup
    request['max_steps'] = 24
    acquired_then_idle(actions)
    actions.extend([action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    report = execute(setup)
    assert report['status'] == 'hypotheses_ready'
    checkpoint = calls[8]['payload']['research_progress']
    assert checkpoint['phase'] == 'synthesis' and checkpoint['checkpoint_start'] == 8
    assert checkpoint['synthesis_actions_remaining'] == 3
    assert 'search' not in checkpoint['allowed_actions'] and 'draft' in checkpoint['allowed_actions']
    assert calls[8]['images'] and calls[9]['images']
    assert report['reasoning_calls'] == 10
    before = [p.read_bytes() for p in sorted(output.glob('response-*.json'))]
    r.run_research(output, director_factory=Director, sources_factory=IncidentSources)
    assert len(calls) == 10 and before == [p.read_bytes() for p in sorted(output.glob('response-*.json'))]


@pytest.mark.parametrize('response', ['note', 'open', 'stop'])
def test_checkpoint_is_bounded_even_when_model_ignores_it_or_reopens_same_source(setup, response):
    _, _, request, calls, actions = setup
    request['max_steps'] = 24
    acquired_then_idle(actions)
    choice = action(response, **({'note': 'Keep searching'} if response == 'note' else
                                {'source_id': 's0'} if response == 'open' else {'reason': 'Evidence inadequate'}))
    actions.extend([choice] * 3)
    report = execute(setup)
    assert report['status'] == ('model_stop' if response == 'stop' else 'stagnated')
    assert len(calls) == (9 if response == 'stop' else 11)
    assert not report['hypotheses'] and report['human_labels'] == 0
    if response != 'stop':
        assert [c['payload']['research_progress']['synthesis_actions_remaining'] for c in calls[8:]] == [3, 2, 1]
        assert report['steps'][-1]['tool_error']


def test_metadata_only_lead_can_be_acquired_drafted_and_committed_in_checkpoint(setup):
    _, _, request, calls, actions = setup
    request['max_steps'] = 24
    class Metadata(IncidentSources):
        def acquire(self, candidate_id):
            return self.open(candidate_id)
    actions.append(action('search', query='lead'))
    actions.extend(action('note', note='Evidence remains unresolved') for _ in range(6))
    actions.extend([action('acquire', candidate_id='s0'), action('draft', hypothesis=hypothesis()),
                    action('commit', hypothesis=hypothesis())])
    report = execute(setup, Metadata)
    assert report['status'] == 'hypotheses_ready' and len(calls) == 10
    assert not calls[7]['images'] and calls[8]['images']
    assert calls[9]['payload']['research_progress']['allowed_actions'] == ['stop', 'commit']


@pytest.mark.parametrize('terminal', ['model_stop', 'hypotheses_ready', 'response_rejected', 'stagnated'])
def test_crash_after_durable_terminal_step_cannot_dispatch_one_more_call(setup, monkeypatch, terminal):
    output, _, request, calls, actions = setup
    request['max_steps'] = 24
    if terminal == 'model_stop':
        actions.append(action('stop'))
    elif terminal == 'hypotheses_ready':
        actions.extend([action('search', query='lead'), action('open', source_id='s0'),
                        action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    elif terminal == 'response_rejected':
        def rejected(*args, **kwargs):
            calls.append('rejected')
            return {'raw_response': '{broken', 'parse_error': 'Invalid JSON'}
        monkeypatch.setattr(r, '_multimodal_json', rejected)
    else:
        actions.extend(action('search', query=f'empty {i}') for i in range(6))
    original = Ledger.update_job
    crashed = []
    def update(self, job_id, **kwargs):
        if kwargs.get('status') == terminal and not crashed:
            crashed.append(True)
            raise OSError('Injected crash after durable terminal step')
        return original(self, job_id, **kwargs)
    monkeypatch.setattr(Ledger, 'update_job', update)
    with pytest.raises(OSError, match='Injected crash'):
        execute(setup)
    count = len(calls)
    report = r.run_research(output, director_factory=Director, sources_factory=IncidentSources)
    assert len(calls) == count and report['status'] == terminal
    assert report['reasoning_calls'] == count


@pytest.mark.parametrize('max_steps', [4, 7, 8, 9, 10, 11, 12, 24, 48])
def test_synthesis_and_stagnation_share_total_budget_at_boundary_lengths(setup, monkeypatch, max_steps):
    output, _, request, calls, _ = setup
    request['max_steps'] = max_steps
    def choose(director, system, payload, images, validator, **kwargs):
        calls.append(payload)
        if len(calls) == 1:
            value = action('search', query='lead')
        elif payload['research_progress']['phase'] == 'research':
            value = action('note', note='Unresolved direction')
        elif not images and 'acquire' in payload['research_progress']['allowed_actions']:
            value = action('acquire', candidate_id='s0')
        elif 'draft' in payload['research_progress']['allowed_actions'] and not payload['pending_draft']:
            value = action('draft', hypothesis=hypothesis())
        elif 'commit' in payload['research_progress']['allowed_actions']:
            value = action('commit', hypothesis=hypothesis())
        else:
            value = action('stop')
        return {'raw_response': json.dumps(value), 'output': validator(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    class Acquisition(IncidentSources):
        def acquire(self, source_id):
            return self.open(source_id)
    monkeypatch.setattr(r, '_multimodal_json', choose)
    report = execute(setup, Acquisition)
    assert report['status'] == 'hypotheses_ready'
    assert len(calls) <= max_steps and len(calls) <= 10
    assert report['reasoning_calls'] == len(calls)


@pytest.mark.parametrize('boundary', ['response', 'step'])
@pytest.mark.parametrize('index', [0, 1, 7, 8])
def test_resume_across_source_checkpoint_and_draft_boundaries(setup, monkeypatch, boundary, index):
    output, _, request, calls, actions = setup
    request['max_steps'] = 24
    acquired_then_idle(actions)
    actions.extend([action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    interrupted = []
    if boundary == 'response':
        original = r._execute_action
        def execute_action(output, spec, state, steps, action, backend):
            if len(steps) == index and not interrupted:
                interrupted.append(True)
                raise r.BudgetExceeded('Injected boundary interruption')
            return original(output, spec, state, steps, action, backend)
        monkeypatch.setattr(r, '_execute_action', execute_action)
    else:
        original = Ledger.update_job
        def update(self, job_id, **kwargs):
            if (output / f'step-{index:03}.json').exists() and not interrupted:
                interrupted.append(True)
                raise OSError('Injected boundary interruption')
            return original(self, job_id, **kwargs)
        monkeypatch.setattr(Ledger, 'update_job', update)
    with pytest.raises((r.BudgetExceeded, OSError), match='Injected boundary'):
        execute(setup)
    saved = {p:p.read_bytes() for p in output.glob('response-*.json')}
    report = r.run_research(output, director_factory=Director, sources_factory=IncidentSources)
    assert report['status'] == 'hypotheses_ready' and report['reasoning_calls'] == len(calls) == 10
    assert all(p.read_bytes() == raw for p, raw in saved.items())
    assert report['research_progress']['checkpoint_start'] == 8
