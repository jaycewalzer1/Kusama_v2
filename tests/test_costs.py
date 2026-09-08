"""Accounting invariants: partial estimates, immutable evidence and crash recovery."""
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from kusama.costs import CostJournal, estimate_local_cost, tracked_local_call, validate_policy


POLICY = {'currency': 'USD', 'average_power_watts': 100,
          'electricity_per_kwh': 0.20, 'hardware_per_hour': 1.50,
          'source': 'Synthetic test fixture, not the user\'s real tariff or power draw.'}


def test_unknown_rates_never_turn_local_api_zero_into_total_zero(tmp_path):
    with tracked_local_call(tmp_path, 'run', 'flux:0', 'FLUX') as cost:
        cost['runtime_seconds'] = 3600
        cost['measurements'] = {'sampled_max_bytes': {'process_rss_bytes': 1234}}
    with CostJournal(tmp_path) as journal:
        summary = journal.summary('run')
    usd = summary['currencies']['USD']
    assert usd['known_subtotal'] == 0 and usd['total'] is None
    assert usd['complete'] is False and usd['incomplete_calls'] == 1
    assert usd['components']['electricity']['unknown_calls'] == 1
    assert usd['components']['hardware']['unknown_calls'] == 1
    record = summary['records'][0]['finish']
    assert record['cost']['components']['provider_api']['status'] == 'actual'
    assert record['measurements']['sampled_max_bytes']['process_rss_bytes'] == 1234


def test_tariff_is_snapshotted_and_actual_is_distinct_from_estimated(tmp_path, monkeypatch):
    policy_path = tmp_path / 'policy.json'
    policy_path.write_text(json.dumps(POLICY))
    monkeypatch.setenv('KUSAMA_COST_POLICY', str(policy_path))
    with tracked_local_call(tmp_path, 'run', 'one', 'FLUX') as cost:
        policy_path.write_text(json.dumps({**POLICY, 'hardware_per_hour': 999}))
        cost['runtime_seconds'] = 3600
    with CostJournal(tmp_path) as journal:
        record = journal.call('run', 'one')
        summary = journal.summary('run')
    assert record['start']['policy']['hardware_per_hour'] == 1.5
    assert record['finish']['cost']['total'] == pytest.approx(1.52)
    assert record['finish']['cost']['components']['electricity']['status'] == 'estimated'
    assert record['finish']['cost']['components']['hardware']['source'] == POLICY['source']
    assert summary['currencies']['USD']['complete'] is True
    assert summary['currencies']['USD']['estimated_calls'] == 1


def test_partial_rates_known_runtime_and_currencies_are_not_summed_together(tmp_path):
    with CostJournal(tmp_path) as journal:
        journal.start('run', 'usd', 'FLUX', policy={**POLICY, 'hardware_per_hour': None})
        journal.finish('run', 'usd', 3600)
        journal.start('run', 'eur', 'Qwen', policy={**POLICY, 'currency': 'EUR'})
        journal.finish('run', 'eur', 3600)
        summary = journal.summary('run')
    assert summary['currencies']['USD']['known_subtotal'] == pytest.approx(.02)
    assert summary['currencies']['USD']['total'] is None
    assert summary['currencies']['EUR']['total'] == pytest.approx(1.52)


def test_cost_events_are_append_only_and_duplicate_operations_never_run(tmp_path):
    with CostJournal(tmp_path) as journal:
        journal.start('run', 'one', 'FLUX', policy=POLICY)
        journal.start('run', 'one', 'FLUX', policy=POLICY)
        journal.finish('run', 'one', 1)
        journal.finish('run', 'one', 1)
        assert journal.connection.execute('SELECT COUNT(*) FROM cost_events').fetchone()[0] == 2
        with pytest.raises(ValueError, match='different immutable'):
            journal.finish('run', 'one', 2)
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            journal.connection.execute("UPDATE cost_events SET created_at='changed'")
        with pytest.raises(sqlite3.IntegrityError, match='append-only'):
            journal.connection.execute('DELETE FROM cost_events')
    with pytest.raises(ValueError, match='already exists'):
        with tracked_local_call(tmp_path, 'run', 'one', 'FLUX'):
            pytest.fail('A duplicate identifier is not permission to execute again')


@pytest.mark.parametrize('error,outcome', [(RuntimeError('failure'), 'failed'), (KeyboardInterrupt(), 'interrupted')])
def test_failure_and_interruption_retain_consumed_runtime(tmp_path, error, outcome):
    with pytest.raises(type(error)):
        with tracked_local_call(tmp_path, 'run', 'one', 'FLUX', policy=POLICY) as cost:
            cost['runtime_seconds'] = 3600
            raise error
    with CostJournal(tmp_path) as journal:
        record = journal.call('run', 'one')
    assert record['finish']['outcome'] == outcome
    assert record['finish']['cost']['total'] == pytest.approx(1.52)


def test_power_loss_stays_unknown_then_can_be_reconciled_without_fabricated_runtime(tmp_path):
    code = '''
import os, sys
from pathlib import Path
from kusama.costs import tracked_local_call
with tracked_local_call(Path(sys.argv[1]), 'run', 'one', 'FLUX'):
    os._exit(77)
'''
    result = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, timeout=20)
    assert result.returncode == 77, result.stderr
    with CostJournal(tmp_path) as journal:
        assert journal.summary('run')['open_calls'] == 1
        journal.finish('run', 'one', outcome='uncertain', evidence='Executor died without a timing record.')
        result = journal.summary('run')
    assert result['open_calls'] == 0
    assert result['records'][0]['finish']['runtime_seconds'] is None
    assert result['currencies']['USD']['total'] is None


def test_missing_coverage_not_claimed_free_and_read_only_does_not_create_db(tmp_path):
    with pytest.raises(sqlite3.OperationalError):
        CostJournal(tmp_path, read_only=True)
    assert not (tmp_path / 'costs.sqlite3').exists()
    with CostJournal(tmp_path) as journal:
        assert journal.summary('unknown')['coverage'] == 'no_cost_records'
        assert journal.summary('unknown')['currencies'] == {}
    with CostJournal(tmp_path, read_only=True) as journal:
        assert journal.summary('unknown')['calls'] == 0


@pytest.mark.parametrize('policy', [[], {'currency': 'dollars'}, {'hardware_per_hour': -1, 'source': 'x'},
    {'hardware_per_hour': True, 'source': 'x'}, {'hardware_per_hour': math.inf, 'source': 'x'},
    {'electricity_per_kwh': .2}, {'unrecognized': 10}])
def test_invalid_or_unsourced_estimates_rejected(policy):
    with pytest.raises(ValueError):
        validate_policy(policy)


def test_an_explicit_zero_rate_can_be_complete_but_unknown_runtime_cannot():
    policy = {**POLICY, 'electricity_per_kwh': 0, 'hardware_per_hour': 0}
    assert estimate_local_cost(100, policy)['total'] == 0
    assert estimate_local_cost(None, policy)['total'] is None


def test_ledger_aggregates_descendants_once_without_changing_call_budgets(tmp_path):
    from kusama.ledger import Ledger
    with Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('parent', {}, 2, tmp_path / 'parent')
        ledger.create_job('child', {}, 1, tmp_path / 'child', parent_id='parent')
        ledger.reserve_call('child', 0)
        before = ledger.call_count('parent')
        assert ledger.cost_summary('parent')['coverage'] == 'no_cost_records'
        with CostJournal(tmp_path) as costs:
            costs.start('child', 'attempt:0', 'FLUX', policy=POLICY)
            costs.finish('child', 'attempt:0', 3600)
            costs.start('unrelated', 'attempt:0', 'FLUX', policy=POLICY)
            costs.finish('unrelated', 'attempt:0', 3600)
        total = ledger.cost_summary('parent')
        assert total['currencies']['USD']['total'] == pytest.approx(1.52)
        assert total['calls'] == 1
        assert total['included_run_ids'] == ['parent', 'child']
        assert ledger.cost_summary('parent', include_descendants=False)['coverage'] == 'no_cost_records'
        assert ledger.call_count('parent') == before == 1
