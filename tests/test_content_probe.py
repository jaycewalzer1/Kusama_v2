import json

import pytest

from kusama.content_probe import parse, plan


def test_refusal_is_preserved_as_invalid_not_repaired():
    with pytest.raises(ValueError):
        parse('I cannot assist with that request.')
    with pytest.raises(ValueError):
        parse('{"instruction":"one","instruction":"two","intended_effect":"x"}')
    with pytest.raises(ValueError):
        parse('{"instruction":"  ","intended_effect":"x"}')
    assert parse('{"instruction":"a painting","intended_effect":"unease"}')['instruction'] == 'a painting'


def test_planning_freezes_budget_without_execution(tmp_path):
    root = tmp_path/'probe'
    plan(root, 'configs/evaluation/content_probe_v1.json')
    spec = json.loads((root/'plan.json').read_text())
    assert spec['max_calls'] == 12
    assert spec['session_config']['renderer_attempt_cap'] == 8
    assert spec['session_config']['reasoning_token_cap'] == 32768
    assert spec['retry_cap'] == 0
    assert spec['render_profile'] == 'flux_base4b_study_512_v1'
    assert (spec['render_config']['width'], spec['render_config']['height']) == (512, 640)
    assert not (root/'resources.sqlite3').exists()
    assert not (root/'jobs.sqlite3').exists()
    with pytest.raises(ValueError, match='new directory'):
        plan(root, 'configs/evaluation/content_probe_v1.json')
