"""Proposal/execution-boundary fixtures; no real model inference or artistic labels."""
from pathlib import Path
import pytest
from pydantic import ValidationError

from kusama.inspiration import (ConnectionHypothesis,RenderAction,compile_connection,
    example,validate_connection,model_call)
from kusama.ledger import Ledger,SpecMismatch
from kusama.resources import SessionBudget,load_config


DOCS=[{'reference_id':'cleveland_1','region_id':None},{'reference_id':'met_2','region_id':'met_2_r1'}]


def proposal():
    value=example('c01',DOCS)
    value.update(title='Fixture relationship',transformation='Fixture role exchange')
    return value


def test_grounded_proposal_rejects_unsupplied_ids_and_duplicate_directions():
    value=proposal();assert validate_connection(value,'c01',DOCS)['id']=='c01'
    value['sources'][1]['region_id']='made_up_detail'
    with pytest.raises(ValueError,match='not supplied'):validate_connection(value,'c01',DOCS)
    with pytest.raises(ValueError,match='Duplicate'):validate_connection(proposal(),'c01',DOCS,[proposal()])


def test_malformed_controls_cannot_compile_to_execution():
    value=proposal();value['operation']='arbitrary_shell_command'
    with pytest.raises(ValidationError):compile_connection(value,run_id='run',seed=23,profile='fixture')
    value=proposal();value['mask_path']='/tmp/unsupported'
    with pytest.raises(ValidationError):ConnectionHypothesis.model_validate(value)
    value=proposal();value['composition']['parts'][1]['id']='p1'
    with pytest.raises(ValidationError):ConnectionHypothesis.model_validate(value)


def test_compilation_discloses_actual_weaker_edit_method():
    result=compile_connection(proposal(),run_id='run',seed=23,profile='fixture',parent_artifact_id='parent-real',target_region_id='parent-r1')
    assert result.kind=='edit' and result.scope=='part'
    assert result.execution_method=='flux_prompt_guided_edit' and result.preservation_guarantee=='none'
    assert result.reference_ids==['cleveland_1','met_2']
    assert result.parent_artifact_id=='parent-real'
    with pytest.raises(ValidationError):RenderAction.model_validate({**result.model_dump(),'strength':.4})
    with pytest.raises(ValidationError):RenderAction.model_validate({**result.model_dump(),'parent_artifact_id':None})


def test_saved_model_call_does_not_dispatch_twice(tmp_path):
    calls=[];directory=tmp_path/'run';directory.mkdir()
    with SessionBudget(tmp_path,load_config()) as budget, Ledger(tmp_path/'jobs.sqlite3') as ledger:
        ledger.create_job('fixture',{'fixture_only':True},2,directory)
        def fake_call():
            calls.append(1);return {'metadata':{'input_tokens':5,'generated_tokens':3},'output':{'fixture':True}}
        first=model_call(ledger,'fixture',directory,budget,0,'response.json',fake_call,device='cpu')
        second=model_call(ledger,'fixture',directory,budget,0,'response.json',fake_call,device='cpu')
        assert first==second and calls==[1]
        assert ledger.call_count('fixture')==1
        assert budget.summary()['reasoning_tokens_charged_or_reserved']==8


def test_interrupted_call_stays_uncertain_without_duplicate_dispatch(tmp_path):
    calls=[];directory=tmp_path/'run';directory.mkdir()
    with SessionBudget(tmp_path,load_config()) as budget, Ledger(tmp_path/'jobs.sqlite3') as ledger:
        ledger.create_job('fixture',{'fixture_only':True},2,directory)
        def interrupted():calls.append(1);raise KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):model_call(ledger,'fixture',directory,budget,0,'response.json',interrupted,device='cpu')
        with pytest.raises(RuntimeError,match='Uncertain'):model_call(ledger,'fixture',directory,budget,0,'response.json',interrupted,device='cpu')
        assert calls==[1] and ledger.call_count('fixture')==1
        assert budget.summary()['reasoning_tokens_charged_or_reserved']==8192
        assert ledger.get_attempt('fixture',0)['status']=='uncertain'
