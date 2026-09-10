"""Larger-context admission and charge preservation; no model inference."""
import json
from types import SimpleNamespace

import pytest
import torch

from kusama.backend import BackendError
from kusama.director import DirectorConfig, _multimodal_json
from kusama.inspiration import model_call, preflight_model_load
from kusama.ledger import BudgetExceeded, Ledger, SpecMismatch
from kusama.panel_runs import StartRun, options
from kusama.resources import SessionBudget, load_config
from test_panel_runs import studio


@pytest.mark.parametrize('input_tokens', [14584, 14585])
def test_generation_respects_expanded_combined_boundary(tmp_path, input_tokens):
    calls = []
    class Inputs(dict):
        def to(self, device):
            return self
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return 'fixture prompt'
        def __call__(self, **kwargs):
            assert kwargs['truncation'] is False
            return Inputs(input_ids=torch.zeros((1, input_tokens), dtype=torch.long))
        def batch_decode(self, *args, **kwargs):
            return ['{"action":"stop"}']
    def generate(input_ids, max_new_tokens, **kwargs):
        calls.append(max_new_tokens)
        return torch.cat([input_ids, torch.ones((1, 1), dtype=torch.long)], dim=1)
    holder = SimpleNamespace(config=DirectorConfig(tmp_path, max_new_tokens=1800, max_input_tokens=16384),
        model=SimpleNamespace(generate=generate), processor=Processor(),
        execution_settings={'status': 'ready', 'device': 'cpu'})
    def invoke():
        return _multimodal_json(holder, 'policy', {}, [], lambda x: x,
            total_token_limit=16384, allow_text_only=True)
    if input_tokens == 14584:
        result = invoke()
        assert result['output'] == {'action': 'stop'} and calls == [1800]
        assert result['metadata']['input_token_limit'] == 14584
    else:
        with pytest.raises(BackendError, match='14585.*14584'):
            invoke()
        assert not calls
    assert DirectorConfig(tmp_path).max_input_tokens == 8192


def test_larger_reservation_replays_once_and_cannot_change_its_limit(tmp_path):
    directory = tmp_path / 'run'; directory.mkdir()
    calls = []
    def generate():
        calls.append(1)
        return {'output': {}, 'metadata': {'input_tokens': 12000, 'generated_tokens': 1000}}
    with SessionBudget(tmp_path, load_config()) as budget, Ledger(tmp_path / 'jobs.sqlite3') as ledger:
        ledger.create_job('fixture', {'fixture_only': True}, 1, directory)
        args = (ledger, 'fixture', directory, budget, 0, 'response.json', generate)
        first = model_call(*args, device='cpu', max_tokens=16384)
        assert model_call(*args, device='cpu', max_tokens=16384) == first
        with pytest.raises(SpecMismatch, match='reservation'):
            model_call(*args, device='cpu', max_tokens=8192)
        assert calls == [1]
        call = budget.summary()['calls'][0]
        assert call['token_reservation'] == 16384
        assert budget.summary()['reasoning_tokens_charged_or_reserved'] == 13000


@pytest.mark.parametrize('updates', [
    {'per_call_total_tokens': 8192}, {'reasoning_token_cap': 16383},
])
def test_insufficient_frozen_budget_stops_before_model_load(tmp_path, updates):
    cfg = load_config().model_copy(update=updates)
    with SessionBudget(tmp_path, cfg) as budget:
        with pytest.raises(BudgetExceeded):
            preflight_model_load(budget, 'reasoning', estimated_seconds=1, max_tokens=16384)
        assert not budget.summary()['calls']


def test_largest_panel_allowance_and_display_agree(studio):
    store, _ = studio
    req = StartRun(submission_id='maximum', question='Fixture', research_steps=48,
        directions=6, development_steps=6, development_research_steps=24)
    entry = store.start(req)
    frozen = json.loads((store.directory(entry['run_id']) / 'request.json').read_text())
    resource = frozen['resource_config']
    assert resource['reasoning_token_cap'] == 91 * 16384 == 1490944
    assert resource['per_call_total_tokens'] == options()['total_token_limit'] == 16384
    assert options()['reserved_output_tokens'] == 1800
