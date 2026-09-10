"""Independent safety audit: frozen authority, resource admission, and recovery."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from kusama import autonomous as a
from kusama.artifacts import sha256
from kusama.ledger import BudgetExceeded, SpecMismatch
from kusama.memory import digest as memory_digest
from kusama.resources import SessionBudget
from test_autonomous import setup, Director, Backend


REJECTED = (ValueError, SpecMismatch)


@pytest.mark.parametrize('length', [201, 254])
def test_plan_rejects_authorship_exceeding_continuation_limit_before_writes(setup, length):
    root, output, request, calls = setup
    request['authorship'] = 'x' * length
    with pytest.raises(ValueError, match='authorship'):
        a.plan_loop(output, request)
    assert not output.exists()
    assert calls == [] and Director.loads == Backend.calls == 0
    with SessionBudget(root) as budget:
        assert budget.summary()['calls'] == []


def test_preflight_recovery_keeps_exact_operator_attribution_without_selection(setup, monkeypatch):
    from test_autonomous import proposal
    root, output, request, calls = setup
    request['authorship'] = 'x' * 200
    original = a._multimodal_json

    def model(director, system, payload, images, validator, **kwargs):
        if system == a.PROPOSAL_SYSTEM and not payload['prior_rounds']:
            value = proposal(region=[.4, .4, .6, .6])
            calls.append((system, payload, images))
            return {'raw_response': json.dumps(value), 'output': validator(value),
                    'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
        return original(director, system, payload, images, validator, **kwargs)

    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    first, second = result['rounds']
    assert first['failure_category'] == 'preflight_rejected' and first.get('selection') is None
    assert second['parent'] == first['parent']
    continuation = a.ContinuationStore(output).get(second['continuation_id'])
    selection = continuation['request']['selection']
    assert selection['origin'] == 'operator' and selection['author'] == request['authorship']
    assert selection['reason'] == 'Frozen initial input'
    assert continuation['request']['intention_map'].get('previous_continuation_id') is None
    assert result['reasoning_calls'] == len(calls) == 4
    assert result['renderer_attempts'] == Backend.calls == 2
    before = (Director.loads, Backend.calls, len(calls))
    a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert (Director.loads, Backend.calls, len(calls)) == before
    with SessionBudget(root) as budget:
        assert not budget.summary()['unsettled']


def test_explicit_model_parent_retention_survives_intervening_preflight_rejection(setup, monkeypatch):
    from test_autonomous import proposal
    _, output, request, calls = setup
    request.update(max_rounds=3, seeds=[17, 23, 41, 59, 71, 83])
    original = a._multimodal_json

    def model(director, system, payload, images, validator, **kwargs):
        if system == a.PROPOSAL_SYSTEM and len(payload['prior_rounds']) == 1:
            value = proposal(region=[.4, .4, .6, .6])
        elif system == a.SELECTION_SYSTEM:
            value = {'selected_id': 'parent', 'next_action': 'continue', 'reason': 'Keep the actual original parent',
                     'local_assessment': 'No established gain', 'whole_assessment': 'Retain the field',
                     'preservation_assessment': 'Parent pixels remain intact'}
        else:
            return original(director, system, payload, images, validator, **kwargs)
        calls.append((system, payload, images))
        return {'raw_response': json.dumps(value), 'output': validator(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}

    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    first, rejected, last = result['rounds']
    assert rejected['failure_category'] == 'preflight_rejected' and rejected.get('selection') is None
    assert first['selected']['sha256'] == last['parent']['sha256'] == request['parent']['sha256']
    continuation = a.ContinuationStore(output).get(last['continuation_id'])
    selection = continuation['request']['selection']
    assert selection['origin'] == 'model_selection' and selection['author'] == a.MODEL_ID
    assert selection['reason'] == 'Previous recorded model selection'
    assert continuation['request']['intention_map']['previous_continuation_id'] == first['continuation_id']
    assert result['renderer_attempts'] == Backend.calls == 4
    assert result['reasoning_calls'] == len(calls) == 7


def test_empty_resource_database_cannot_create_default_session(setup, tmp_path):
    _, output, request, _ = setup
    root = tmp_path / 'unadmitted'
    root.mkdir()
    database = root / 'resources.sqlite3'
    sqlite3.connect(database).close()
    request['resource_root'] = str(root)
    a.plan_loop(output, request)
    with pytest.raises(REJECTED, match='session|admit|resource'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert Director.loads == Backend.calls == 0
    with sqlite3.connect(database) as connection:
        exists = connection.execute("SELECT 1 FROM sqlite_master WHERE name='session'").fetchone()
        assert not exists or connection.execute('SELECT count(*) FROM session').fetchone()[0] == 0


def test_named_real_profile_cannot_claim_different_render_settings(setup):
    _, output, request, _ = setup
    request.update(fixture=False, render_profile_id='flux_base4b_1024_v1')
    # Deliberate mismatch: the fixture config is 64x64, 1 step, CPU/fp32.
    with pytest.raises(REJECTED, match='profile|render'):
        a.plan_loop(output, request)
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert Director.loads == Backend.calls == 0


@pytest.mark.parametrize('field', ['render_config', 'director_config', 'max_reasoning_calls', 'policy'])
def test_all_executable_plan_fields_are_frozen_before_first_run(setup, field):
    _, output, request, _ = setup
    a.plan_loop(output, request)
    path = output / 'plan.json'
    spec = json.loads(path.read_text())
    if field == 'render_config': spec[field]['steps'] += 1
    elif field == 'director_config': spec[field]['max_new_tokens'] += 1
    elif field == 'max_reasoning_calls': spec[field] += 3
    else: spec[field]['proposal'] = 'Changed after preparation'
    path.write_text(json.dumps(spec))
    with pytest.raises(REJECTED, match='changed|identity|frozen|mismatch|plan|configuration'):
        a.preflight_loop(output)
    assert Director.loads == Backend.calls == 0


def test_snapshot_cannot_replace_the_declared_parent_before_first_run(setup, tmp_path):
    _, output, request, _ = setup
    a.plan_loop(output, request)
    other = tmp_path / 'other.png'
    Image.new('RGB', (64, 64), 'blue').save(other)
    path = output / 'plan.json'
    spec = json.loads(path.read_text())
    spec['snapshots']['parent'].update(path=str(other), sha256=sha256(other))
    path.write_text(json.dumps(spec))
    with pytest.raises(REJECTED, match='changed|identity|frozen|mismatch|parent|plan'):
        a.preflight_loop(output)


def test_unregistered_round_file_cannot_fabricate_a_model_selection(setup, tmp_path):
    _, output, request, _ = setup
    spec = a.plan_loop(output, request)
    other = tmp_path / 'unseen.png'
    Image.new('RGB', (64, 64), 'blue').save(other)
    parent = {key: spec['snapshots']['parent'][key] for key in ('path', 'sha256')}
    forged = {'round': 0, 'parent': parent, 'candidates': [],
              'selected': {'path': str(other), 'sha256': sha256(other)},
              'origin': 'model_development', 'human_quality': None, 'status': 'model_stop'}
    (output / 'round-00.json').write_text(json.dumps(forged))
    with pytest.raises(REJECTED, match='round|registered|evidence|selection|artifact'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert Director.loads == Backend.calls == 0


def test_memory_snapshot_identity_detects_changed_selected_payload(setup):
    _, output, request, _ = setup
    event_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    decision_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    event = {'id': 'memory-event', 'episode_id': 'earlier-development', 'partition': 'development',
             'kind': 'observation', 'available_at': event_time, 'source_database': '/fixture/source.sqlite3',
             'source_table': 'studio_events', 'source_event_id': 'prior-event', 'source_row_sha256': 'a' * 64,
             'origin': 'model_observation', 'payload': {'observation': 'An uncertain relationship'}}
    snapshot = {'schema_version': 1, 'memory_version': 'event-cutoff-v1',
                'request': {'purpose': 'development', 'decision_at': decision_time},
                'eligible_candidates': [{'id': event['id'], 'available_at': event_time,
                                         'source_row_sha256': event['source_row_sha256']}],
                'selected': [event]}
    snapshot['snapshot_id'] = memory_digest(snapshot)
    changed = deepcopy(snapshot)
    changed['selected'][0]['payload']['observation'] = 'Altered result with the same source hash'
    request.update(memory_enabled=True, memory_snapshot=changed)
    with pytest.raises(REJECTED, match='memory|Memory|snapshot|identity|authority'):
        a.plan_loop(output, request)
    assert not output.exists()


def test_proposal_crossing_local_wall_limit_does_not_start_rendering(setup, monkeypatch):
    _, output, request, _ = setup
    request['max_wall_seconds'] = 1
    original = a._multimodal_json
    base = datetime.now(timezone.utc)

    class Clock(datetime):
        offset = timedelta()

        @classmethod
        def now(cls, tz=None):
            value = base + cls.offset
            return value if tz is not None else value.replace(tzinfo=None)

    monkeypatch.setattr(a, 'datetime', Clock)

    def model(*args, **kwargs):
        result = original(*args, **kwargs)
        Clock.offset = timedelta(seconds=2)
        return result

    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    try:
        result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    except BudgetExceeded:
        pass
    else:
        assert result['status'] == 'wall_limit'
        assert result['reasoning_calls'] == 1
    assert Backend.calls == 0


def test_uncertain_reasoning_call_is_charged_once_and_never_reloaded(setup, monkeypatch):
    root, output, request, _ = setup
    invoked = []

    def interrupted(*args, **kwargs):
        invoked.append(True)
        raise KeyboardInterrupt('Possible invocation, no committed response')

    monkeypatch.setattr(a, '_multimodal_json', interrupted)
    a.plan_loop(output, request)
    with pytest.raises(KeyboardInterrupt):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert len(invoked) == Director.loads == 1
    with pytest.raises(RuntimeError, match='Uncertain|uncertain'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert len(invoked) == Director.loads == 1
    assert Backend.calls == 0
    with SessionBudget(root) as budget:
        calls = budget.summary()['calls']
        assert len(calls) == 1 and calls[0]['status'] == 'uncertain'


def test_committed_response_recovers_without_repeating_inference(setup, monkeypatch):
    root, output, request, calls = setup
    original = a.ContinuationStore.prepare
    interrupted = False

    def stop_after_proposal(*args, **kwargs):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt('Stop after committed Qwen response')
        return original(*args, **kwargs)

    monkeypatch.setattr(a.ContinuationStore, 'prepare', stop_after_proposal)
    a.plan_loop(output, request)
    with pytest.raises(KeyboardInterrupt):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert len(calls) == Director.loads == 1 and Backend.calls == 0
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['reasoning_calls'] == 6 and len(calls) == Director.loads == 6
    assert Backend.calls == 4
    with SessionBudget(root) as budget:
        assert len([call for call in budget.summary()['calls'] if call['kind'] == 'reasoning']) == 6


def test_unregistered_future_execution_start_cannot_extend_wall_window(setup):
    _, output, request, _ = setup
    a.plan_loop(output, request)
    (output / 'execution_start.json').write_text(json.dumps({
        'started_at': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}))
    with pytest.raises(REJECTED, match='start|window|registered|future|artifact'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert Director.loads == Backend.calls == 0


def test_reasoning_load_crossing_deadline_does_not_reserve_or_invoke(setup, monkeypatch):
    root, output, request, calls = setup
    request['max_wall_seconds'] = 1
    base = datetime.now(timezone.utc)

    class Clock(datetime):
        offset = timedelta()

        @classmethod
        def now(cls, tz=None):
            value = base + cls.offset
            return value if tz is not None else value.replace(tzinfo=None)

    class SlowDirector(Director):
        def prepare(self):
            super().prepare()
            Clock.offset = timedelta(seconds=2)

    monkeypatch.setattr(a, 'datetime', Clock)
    a.plan_loop(output, request)
    try:
        result = a.run_loop(output, director_factory=SlowDirector, backend_factory=Backend)
    except BudgetExceeded:
        pass
    else:
        assert result['status'] == 'wall_limit'
        assert result['reasoning_calls'] == 0
    assert calls == [] and Backend.calls == 0
    with SessionBudget(root) as budget:
        assert budget.summary()['calls'] == []


def external_region(role, box):
    return {'id': role + '_external_detail', 'role': role, 'box': box,
            'origin': 'operator_annotation', 'reason': 'Frozen fixture control independent of the model map'}


@pytest.mark.parametrize('role,box,error', [
    ('required_edit', [0, 0, 8, 8], 'external required support'),
    ('protected', [20, 20, 24, 24], 'externally protected'),
])
def test_external_controls_cannot_be_omitted_by_model_map(setup, role, box, error):
    _, output, request, calls = setup
    request['requirements'] = [{'regions': [external_region(role, box)]}]
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['status'] == 'preflight_rejected'
    assert result['renderer_attempts'] == Backend.calls == 0
    assert result['reasoning_calls'] == len(calls) == request.get('max_rounds', 2)
    assert result['selected']['sha256'] == request['parent']['sha256']
    assert all(error in row['preflight_error'] for row in result['rounds'])
    assert all([e['id'] for e in row['proposal']['elements']] == ['center'] for row in result['rounds'])
    assert json.loads((output / 'plan.json').read_text())['request']['requirements'] == request['requirements']


def test_external_controls_allow_contained_edit_and_preserve_omitted_detail(setup):
    _, output, request, _ = setup
    request['requirements'] = [{'regions': [external_region('required_edit', [20, 20, 32, 32]),
                                          external_region('protected', [0, 0, 8, 8])]}]
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['renderer_attempts'] == Backend.calls == 4
    for row in result['rounds']:
        assert [e['id'] for e in row['proposal']['elements']] == ['center']
        for candidate in row['candidates']:
            with Image.open(candidate['path']) as image:
                assert image.crop((0, 0, 8, 8)).getextrema() == ((255, 255),) * 3


def test_unknown_external_region_role_fails_closed(setup):
    _, output, request, calls = setup
    request['requirements'] = [{'regions': [external_region('protectd', [20, 20, 24, 24])]}]
    with pytest.raises(ValueError, match='Unknown external region role'):
        a.plan_loop(output, request)
    assert not calls and Backend.calls == 0
    assert not (output / 'plan.json').exists()


@pytest.mark.parametrize('next_map', ['omit_without_reason', 'omit_with_reason', 'retain_unchanged'])
def test_protected_intention_persists_until_explicit_model_revisit(setup, monkeypatch, next_map):
    from test_autonomous import proposal
    _, output, request, calls = setup
    original = a._multimodal_json
    protected = {'id': 'quiet', 'role': 'Quiet corner', 'region': [0, 0, .125, .125],
                 'presence': 'present', 'evidence': 'Unchanged white corner', 'protect_pixels': True}

    def model(director, system, payload, images, validator, **kwargs):
        if system == a.PROPOSAL_SYSTEM:
            value = proposal()
            first = not payload['prior_rounds']
            if first or next_map == 'retain_unchanged':
                value['elements'].append(deepcopy(protected))
            elif next_map == 'omit_with_reason':
                value['revisit_reasons'] = {'quiet': 'Deliberately release the previous corner protection for future development'}
            result = {'raw_response': json.dumps(value), 'output': validator(value),
                      'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
            calls.append((system, payload, images))
            return result
        return original(director, system, payload, images, validator, **kwargs)

    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    first, second = result['rounds']
    assert second['parent'] == first['selected']
    first_continuation = a.ContinuationStore(output).get(first['continuation_id'])
    quiet = next(e for e in first_continuation['request']['intention_map']['elements'] if e['id'] == 'quiet')
    assert quiet['protect_pixels'] is True
    if next_map == 'omit_without_reason':
        assert result['status'] == second['failure_category'] == 'preflight_rejected'
        assert 'revisit reason' in second['preflight_error']
        assert result['renderer_attempts'] == Backend.calls == 2
        assert result['reasoning_calls'] == 4
        assert second['selected'] == first['selected']
    else:
        assert result['renderer_attempts'] == Backend.calls == 4
        assert result['reasoning_calls'] == 6
        second_map = a.ContinuationStore(output).get(second['continuation_id'])['request']['intention_map']
        assert second_map['previous_continuation_id'] == first['continuation_id']
        if next_map == 'omit_with_reason':
            assert second_map['revisit_reasons']['quiet'] == second['proposal']['revisit_reasons']['quiet']
            assert 'quiet' not in [e['id'] for e in second_map['elements']]
        else:
            assert next(e for e in second_map['elements'] if e['id'] == 'quiet') == quiet
    before = Backend.calls
    resumed = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert resumed['renderer_attempts'] == result['renderer_attempts']
    assert Backend.calls == before
