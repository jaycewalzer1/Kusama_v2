"""Offline safety tests for session reservations; no model execution."""
from datetime import timedelta
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kusama.ledger import BudgetExceeded, JobLocked, SpecMismatch
from kusama.resources import SessionBudget, OvernightConfig, load_config


@pytest.fixture
def config():
    return OvernightConfig.model_validate({**load_config().model_dump(), 'renderer_attempt_cap':2,'reasoning_token_cap':30})


@pytest.fixture(autouse=True)
def adequate_disk(monkeypatch):
    monkeypatch.setattr('kusama.resources.shutil.disk_usage',lambda p:SimpleNamespace(total=10**12,used=10**11,free=9*10**11))


def test_renderer_reservations_are_idempotent_and_failed_attempts_stay_charged(tmp_path,config):
    with SessionBudget(tmp_path,config) as budget:
        assert budget.reserve('a','renderer')
        assert not budget.reserve('a','renderer')
        budget.settle('a','failed')
        assert budget.reserve('b','renderer')
        with pytest.raises(BudgetExceeded,match='attempt cap'):budget.reserve('c','renderer')
        assert budget.summary()['renderer_attempts']==2


def test_uncertain_reasoning_reservation_survives_restart(tmp_path,config):
    with SessionBudget(tmp_path,config) as budget:
        assert budget.reserve('q','reasoning',max_tokens=20)
        budget.settle('q','uncertain')
    with SessionBudget(tmp_path,config) as budget:
        assert budget.summary()['reasoning_tokens_charged_or_reserved']==20
        assert not budget.reserve('q','reasoning',max_tokens=20)
        with pytest.raises(BudgetExceeded):budget.reserve('next','reasoning',max_tokens=11)
        budget.settle('q','completed',actual_tokens=8)
        assert budget.summary()['reasoning_tokens_remaining']==22
        with pytest.raises(SpecMismatch):budget.settle('q','completed',actual_tokens=3)


def test_reserved_resources_and_session_limits_are_frozen(tmp_path,config):
    with SessionBudget(tmp_path,config) as budget:
        budget.reserve('q','reasoning',max_tokens=10)
        with pytest.raises(SpecMismatch):budget.reserve('q','reasoning',max_tokens=11)
    changed=OvernightConfig.model_validate({**config.model_dump(),'renderer_attempt_cap':1})
    with pytest.raises(SpecMismatch):SessionBudget(tmp_path,changed)


def test_handoff_reserve_and_disk_floor_stop_new_dispatch(tmp_path,config,monkeypatch):
    import kusama.resources as resources
    with SessionBudget(tmp_path,config) as budget:
        current=resources.now()
        monkeypatch.setattr(resources,'now',lambda:current+timedelta(hours=7,minutes=20))
        with pytest.raises(BudgetExceeded,match='execution window'):budget.reserve('late','renderer')
        monkeypatch.setattr(resources,'now',lambda:current)
        monkeypatch.setattr(resources.shutil,'disk_usage',lambda p:SimpleNamespace(total=10**12,free=10**9))
        with pytest.raises(BudgetExceeded,match='free-space'):budget.reserve('disk','renderer')
        assert not budget.summary()['calls']


def test_workload_lease_and_append_only_events(tmp_path,config):
    with SessionBudget(tmp_path,config) as budget:
        with budget.workload_lock():
            with pytest.raises(JobLocked):
                with budget.workload_lock():pass
        with budget.workload_lock():pass
        with pytest.raises(sqlite3.IntegrityError,match='append-only'):
            budget.db.execute('DELETE FROM resource_events')


def test_measured_slow_edit_protects_handoff_even_without_caller_estimate(tmp_path, config, monkeypatch):
    from kusama.costs import CostJournal
    import kusama.resources as resources
    with SessionBudget(tmp_path, config) as budget:
        with CostJournal(tmp_path) as costs:
            costs.start('fixture', 'slow-edit', 'black-forest-labs/FLUX.2-klein-base-4B', scope='backend_request')
            costs.finish('fixture', 'slow-edit', runtime_seconds=600, outcome='succeeded')
        assert budget.estimated_seconds('renderer') == 750
        current = resources.now()
        monkeypatch.setattr(resources, 'now', lambda: current + timedelta(hours=7, minutes=6))
        with pytest.raises(BudgetExceeded, match='execution window'):
            budget.reserve('would-run-through-handoff', 'renderer')
        assert budget.summary()['renderer_attempts'] == 0


@pytest.mark.parametrize('field,value',[('renderer_attempt_cap',161),('reasoning_token_cap',2000001),('per_call_total_tokens',16385),('paid_api_usd_cap',1),('asset_byte_cap',3*1024**3),('schema_repair_cap',2),('active_gpu_workloads',2)])
def test_hard_config_caps(field,value):
    with pytest.raises(ValidationError):OvernightConfig.model_validate({**load_config().model_dump(),field:value})


def test_concurrent_workers_export_consistent_budget_snapshots(tmp_path,config):
    from concurrent.futures import ThreadPoolExecutor
    import json
    from threading import Barrier
    with SessionBudget(tmp_path,config):pass
    barrier=Barrier(2)
    def worker(n):
        with SessionBudget(tmp_path,config) as budget:
            barrier.wait(timeout=10)
            result=budget.reserve(f'parallel-{n}','renderer')
            budget.settle(f'parallel-{n}','completed')
            return result
    with ThreadPoolExecutor(2) as pool:assert list(pool.map(worker,[1,2]))==[True,True]
    saved=json.loads((tmp_path/'resource_summary.json').read_text())
    assert saved['renderer_attempts']==2
    assert {r['status'] for r in saved['calls']}=={'completed'}
