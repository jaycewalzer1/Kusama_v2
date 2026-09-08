"""Offline fixtures for disclosed, lossless E0 recovery; no model execution."""
import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from kusama import e0_runner, e0_recovery
from kusama.artifacts import sha256, write_json
from kusama.director import DirectorOutputError
from kusama.evaluation import CONDITIONS
from kusama.ledger import Ledger, SpecMismatch, BudgetExceeded
from kusama.resources import SessionBudget
from test_e0_runner import setup, execute, FixtureBackend, FixtureDirector, FixtureIndex

ORIGINAL={'instruction':'Move the existing band toward the shell.', 'intended_effect':'Test a closer connection.',
    'preserve':'Shell ribbing, the surrounding field, and grayscale palette', 'uncertainty':'The connection may remain ambiguous.'}


def failed_response():
    try:e0_runner.ActionRequest.model_validate(ORIGINAL)
    except ValidationError as exc:error=str(exc)
    return {'raw_response':json.dumps(ORIGINAL), 'output':None, 'parse_error':error,
        'metadata':{'input_tokens':10,'generated_tokens':60},'fixture_only':True}


@pytest.fixture
def legacy(setup,monkeypatch):
    current=e0_runner._runtime_identity
    def old(*args,**kwargs):
        value=copy.deepcopy(current(*args,**kwargs));policy=value['execution_policy']
        for key in ('schema_repair_policy','schema_repair_system','throughput_estimator_version'):policy.pop(key)
        policy['scoped_code_sha256'].pop('e0_recovery.py');policy['scoped_code_sha256']['e0_runner.py']='0'*64
        return value
    primary=e0_runner._multimodal_json
    def fail_one(director,system,request,images,validate,**kwargs):
        if request['task']==e0_runner.ROLE_INSTRUCTIONS['ordinary_critique']:
            setup[3].append({'failed_schema':True})
            raise DirectorOutputError('fixture schema rejection',failed_response())
        return primary(director,system,request,images,validate,**kwargs)
    monkeypatch.setattr(e0_runner,'_runtime_identity',old)
    monkeypatch.setattr(e0_runner,'_multimodal_json',fail_one)
    assert execute(setup,brief_id='e0-d02')['briefs'][0]['generation_status']=='actions_incomplete'
    monkeypatch.setattr(e0_runner,'_runtime_identity',current)
    monkeypatch.setattr('kusama.retrieval.RetrievalIndex',FixtureIndex)
    repair_calls=[]
    def repair(director,system,request,images,validate,**kwargs):
        repair_calls.append(request)
        assert request['original_raw_response']==json.dumps(ORIGINAL)
        assert len(images)==3 and images[0][0]=='parent'
        value={**ORIGINAL,'preserve':[ORIGINAL['preserve']]}
        return {'output':validate(value),'raw_response':json.dumps(value),'metadata':{'input_tokens':10,'generated_tokens':60},'fixture_only':True}
    monkeypatch.setattr(e0_recovery,'_multimodal_json',repair)
    return setup,repair_calls


def prepare(legacy):
    setup,_=legacy
    return e0_recovery.prepare(setup[1],setup[2],root=setup[0],brief_id='e0-d02',condition='ordinary_critique')['authorization']


def recover(path):
    return e0_recovery.execute(path,backend_factory=FixtureBackend,director_factory=FixtureDirector)


def test_explicit_recovery_retains_original_evidence_and_replays_without_cost(legacy):
    setup,repair_calls=legacy;root,protocol,_,primary_calls=setup
    folder=protocol/'episodes'/'e0-d02'
    originals=[folder/'runtime_identity.json',*(folder/'actions').glob('action-*.json')]
    before={str(p):sha256(p) for p in originals}
    with Ledger(root/'jobs.sqlite3') as ledger:
        pid=json.loads((protocol/'manifest.json').read_text())['protocol_id']
        old_attempts=[ledger.get_attempt(f'{pid}_e0-d02_actions',i) for i in range(3)]
        old_specs={stage:ledger.get_job(f'{pid}_e0-d02_{stage}')['spec'] for stage in ('parent','actions')}
    counts=(len(FixtureBackend.calls),len(primary_calls),FixtureBackend.loads,FixtureDirector.loads)
    path=prepare(legacy)
    assert counts==(len(FixtureBackend.calls),len(primary_calls),FixtureBackend.loads,FixtureDirector.loads)
    assert not repair_calls
    assert recover(path)['generation_status']=='generation_complete'
    assert len(repair_calls)==1 and len(primary_calls)==3 and len(FixtureBackend.calls)==7
    assert before=={str(p):sha256(p) for p in originals}
    frozen=json.loads((folder/'frozen.json').read_text())
    repaired=frozen['action_requests']['ordinary_critique']
    assert repaired['preserve']==[ORIGINAL['preserve']]
    assert repaired['schema_repair']['primary_status_retained']=='failed'
    assert repaired['schema_repair']['authorization_sha256']==sha256(Path(path))
    with Ledger(root/'jobs.sqlite3') as ledger:
        assert [ledger.get_attempt(f'{pid}_e0-d02_actions',i) for i in range(3)]==old_attempts
        assert old_specs=={stage:ledger.get_job(f'{pid}_e0-d02_{stage}')['spec'] for stage in ('parent','actions')}
        assert all(ledger.get_job(f'{pid}_e0-d02_{stage}')['status']=='completed' for stage in ('parent','actions','edits'))
    counts=(FixtureBackend.loads,FixtureDirector.loads)
    assert recover(path)['edit_count']==6
    assert len(repair_calls)==1 and len(primary_calls)==3 and len(FixtureBackend.calls)==7
    assert counts==(FixtureBackend.loads,FixtureDirector.loads)
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts']==7
        assert len([row for row in budget.summary()['calls'] if row['kind']=='reasoning'])==4


def test_plain_resume_cannot_migrate_old_runtime_or_mutate_state(legacy):
    setup,_=legacy;p=setup[1]/'episodes'/'e0-d02'/'state.json';before=sha256(p)
    with pytest.raises(SpecMismatch,match='runtime changed'):execute(setup,brief_id='e0-d02')
    assert sha256(p)==before and len(FixtureBackend.calls)==1


@pytest.mark.parametrize('target',['authorization','raw','reference'])
def test_recovery_tamper_blocks_before_any_new_load_or_charge(legacy,target):
    setup,calls=legacy;path=Path(prepare(legacy));a=json.loads(path.read_text())
    if target=='authorization':path.write_text(path.read_text()+' ')
    elif target=='raw':Path(a['original_actions'][1]['path']).write_text('{}')
    else:Path(a['parent']['path']).write_bytes(b'corrupt fixture parent')
    loads=(FixtureBackend.loads,FixtureDirector.loads)
    with pytest.raises((ValueError,SpecMismatch)):recover(path)
    assert not calls and loads==(FixtureBackend.loads,FixtureDirector.loads)
    assert len(FixtureBackend.calls)==1


@pytest.mark.parametrize('field',['instruction','preserve'])
def test_artistic_change_in_repair_fails_without_children_and_never_rerolls(legacy,monkeypatch,field):
    setup,calls=legacy;path=prepare(legacy)
    def bad(director,system,request,images,validate,**kwargs):
        calls.append(request);value={**ORIGINAL,'preserve':[ORIGINAL['preserve']]}
        value[field]=['changed'] if field=='preserve' else 'Changed artistic action'
        try:validate(value)
        except ValueError as exc:
            raise DirectorOutputError(str(exc),{'raw_response':json.dumps(value),'output':None,'parse_error':str(exc),'fixture_only':True})
    monkeypatch.setattr(e0_recovery,'_multimodal_json',bad)
    assert recover(path)['generation_status']=='actions_incomplete'
    assert recover(path)['generation_status']=='actions_incomplete'
    assert len(calls)==1 and len(FixtureBackend.calls)==1
    assert not (setup[1]/'episodes'/'e0-d02'/'frozen.json').exists()


def test_uncertain_repair_never_dispatches_second_call(legacy,monkeypatch):
    setup,calls=legacy;path=prepare(legacy)
    def interrupt(*args,**kwargs):calls.append('interrupted');raise KeyboardInterrupt('fixture')
    monkeypatch.setattr(e0_recovery,'_multimodal_json',interrupt)
    assert recover(path)['generation_status']=='uncertain_requires_reconciliation'
    assert recover(path)['generation_status']=='uncertain_requires_reconciliation'
    assert len(calls)==1 and len(FixtureBackend.calls)==1


def test_repair_time_cap_precedes_model_load(legacy,monkeypatch):
    _,calls=legacy;path=prepare(legacy);loads=FixtureDirector.loads
    monkeypatch.setattr(e0_runner,'_preload_seconds',lambda *args:10**9)
    assert recover(path)['generation_status']=='queued_resource_limit'
    assert not calls and FixtureDirector.loads==loads


def test_no_new_authorization_after_child_work(legacy):
    setup,_=legacy;folder=setup[1]/'episodes'/'e0-d02';write_json(folder/'frozen.json',{'fixture':'unauthorized child freeze'})
    with pytest.raises(ValueError,match='after child'):prepare(legacy)


def test_schema_eligibility_and_losslessness_are_narrow():
    assert e0_recovery.eligible_source(failed_response())==ORIGINAL
    assert e0_recovery.eligible_source({**failed_response(),'inference_error':'failed'}) is None
    assert e0_recovery.eligible_source({**failed_response(),'raw_response':'not JSON'}) is None
    wrong={**ORIGINAL,'instruction':''}
    assert e0_recovery.eligible_source({**failed_response(),'raw_response':json.dumps(wrong)}) is None
    valid={**ORIGINAL,'preserve':[ORIGINAL['preserve']]}
    assert e0_recovery.validate_repair(valid,ORIGINAL)==valid
    with pytest.raises(ValueError):e0_recovery.validate_repair({**valid,'preserve':ORIGINAL['preserve']},ORIGINAL)


def test_stage_forecast_uses_slow_failed_attempts_and_all_repair_loads():
    row=lambda seconds,status:{'status':status,'backend':{'model_id':'real-recorded-model'},'measurements':{'runtime_seconds':seconds}}
    parents=[row(100,'completed'),row(200,'technical_failure')]
    edits=[row(180,'completed'),row(400,'technical_failure')]
    timing={'parent_render_seconds':max(map(e0_runner._measured_render_time,parents)),
        'edit_render_seconds':max(map(e0_runner._measured_render_time,edits)),
        'reasoning_seconds':12,'schema_repair_seconds':15,'model_load_seconds_per_brief':20,'schema_repair_model_load_seconds':10}
    assert e0_runner._stage_forecast(timing,8,48,2,24,24,8)==1.25*(8*200+48*400+2*400+24*12+24*15+8*20+24*10)


def test_new_episode_predeclares_one_schema_repair_and_replays_completed_children(setup,monkeypatch):
    primary=e0_runner._multimodal_json;repair_calls=[]
    def fail(director,system,request,images,validate,**kwargs):
        if request['task']==e0_runner.ROLE_INSTRUCTIONS['ordinary_critique']:
            setup[3].append({'failed_schema':True})
            raise DirectorOutputError('fixture',failed_response())
        return primary(director,system,request,images,validate,**kwargs)
    def repair(director,system,request,images,validate,**kwargs):
        repair_calls.append(request);value={**ORIGINAL,'preserve':[ORIGINAL['preserve']]}
        return {'output':validate(value),'raw_response':json.dumps(value),'fixture_only':True}
    monkeypatch.setattr(e0_runner,'_multimodal_json',fail)
    monkeypatch.setattr(e0_recovery,'_multimodal_json',repair)
    assert execute(setup,brief_id='e0-d01')['briefs'][0]['edit_count']==6
    loads=(FixtureBackend.loads,FixtureDirector.loads)
    assert execute(setup,brief_id='e0-d01')['briefs'][0]['edit_count']==6
    assert len(repair_calls)==1 and len(setup[3])==3 and len(FixtureBackend.calls)==7
    assert loads==(FixtureBackend.loads,FixtureDirector.loads)
    folder=setup[1]/'episodes'/'e0-d01'
    assert not (folder/'schema_repairs'/'ordinary_critique'/'authorization.json').exists()
    runtime=json.loads((folder/'runtime_identity.json').read_text())
    assert runtime['execution_policy']['schema_repair_policy']['max_calls_per_primary_action']==1


def test_runtime_recovery_rejects_model_profile_and_artistic_policy_changes(legacy,monkeypatch):
    setup,_=legacy;current=e0_runner._runtime_identity
    for section,key,value in [('render_config','width',128),('director_profile','max_new_tokens',1024),('execution_policy','action_system','changed')]:
        def changed(*args,section=section,key=key,value=value,**kwargs):
            result=copy.deepcopy(current(*args,**kwargs));result[section][key]=value;return result
        monkeypatch.setattr(e0_runner,'_runtime_identity',changed)
        with pytest.raises(SpecMismatch):prepare(legacy)
    assert len(FixtureBackend.calls)==1


def test_heldout_cannot_create_development_recovery(legacy):
    setup,_=legacy
    with pytest.raises(ValueError,match='development-only'):
        e0_recovery.prepare(setup[1],setup[2],root=setup[0],brief_id='e0-h01',condition='ordinary_critique')


def test_recovery_tokens_cap_precedes_new_load_and_charge(legacy,monkeypatch):
    setup,calls=legacy;path=prepare(legacy);loads=FixtureDirector.loads;summary=SessionBudget.summary
    def exhausted(self):return {**summary(self),'reasoning_tokens_remaining':8191}
    monkeypatch.setattr(SessionBudget,'summary',exhausted)
    assert recover(path)['generation_status']=='queued_resource_limit'
    assert not calls and FixtureDirector.loads==loads


def test_stage_freeze_includes_failed_renderer_and_reasoning_cost_receipts(setup,monkeypatch):
    """Fabricated timing receipts only; exercise aggregation without any model."""
    from kusama.costs import CostJournal
    root,protocol,_,_=setup
    execute(setup)
    pid=json.loads((protocol/'manifest.json').read_text())['protocol_id']
    # This test explicitly replaces fixture metadata with simulated live-shaped
    # timing records to exercise aggregation, never reporting real evidence.
    for bid in [f'e0-d{i+1:02}' for i in range(4)]:
        directory=protocol/'episodes'/bid
        for stage,seconds in [('parent',100),('edits',180)]:
            for p in (directory/stage).glob('attempt-*.json'):
                row=json.loads(p.read_text());row['backend']={'model_id':'SIMULATED_TEST_ONLY'}
                row['measurements']['runtime_seconds']=seconds;write_json(p,row)
        for p in (directory/'actions').glob('action-*.json'):
            row=json.loads(p.read_text());row.pop('fixture_only');row['measurements']['runtime_seconds']=12;write_json(p,row)
    directory=protocol/'episodes'/'e0-d01'
    for stage,seconds in [('parent',250),('edits',400)]:
        write_json(directory/stage/'attempt-99.json',{'status':'technical_failure','backend':{'model_id':'SIMULATED_TEST_ONLY'},'measurements':{'runtime_seconds':seconds}})
    repair=directory/'schema_repairs'/'ordinary_critique'/'response.json'
    repair.parent.mkdir(parents=True)
    write_json(repair,{'parse_error':'simulated failed repair timing aggregation','measurements':{},'SIMULATED_TEST_ONLY':True})
    repair_id=e0_recovery.repair_job_id(pid,'e0-d01','ordinary_critique')
    with CostJournal(root) as costs:
        costs.start(repair_id,'reasoning-0','Qwen_SIMULATED_TEST_ONLY',scope='inference')
        costs.finish(repair_id,'reasoning-0',runtime_seconds=45,outcome='failed')
        costs.start(f'{pid}_e0-d01_actions','slow-failed-simulated','Qwen_SIMULATED_TEST_ONLY',scope='inference')
        costs.finish(f'{pid}_e0-d01_actions','slow-failed-simulated',runtime_seconds=70,outcome='failed')
    # Separate tests verify real ledger rejection; this fixture isolates numerical aggregation.
    monkeypatch.setattr(e0_runner,'_verified_development_evidence',lambda *args:{str(p.resolve()):sha256(p) for p in protocol.rglob('*.json')})
    result=e0_runner.freeze_runtime(protocol,setup[2],root=root,retry_reserve=2)
    frozen=json.loads(Path(result['path']).read_text())
    assert frozen['parent_render_seconds']==250 and frozen['edit_render_seconds']==400
    assert frozen['schema_repair_seconds']==45 and frozen['reasoning_seconds']==70
    assert frozen['retry_reserve']==2
    assert len([row for row in frozen['throughput_evidence'] if row.get('attempt_status')=='technical_failure'])==2


def test_inconsistent_saved_raw_response_cannot_authorize_children(legacy,monkeypatch):
    setup,calls=legacy;path=prepare(legacy)
    def inconsistent(director,system,request,images,validate,**kwargs):
        calls.append(request)
        return {'raw_response':json.dumps(ORIGINAL),'output':validate({**ORIGINAL,'preserve':[ORIGINAL['preserve']]}),'fixture_only':True}
    monkeypatch.setattr(e0_recovery,'_multimodal_json',inconsistent)
    with pytest.raises(ValueError,match='exact original preservation'):recover(path)
    assert len(calls)==1 and len(FixtureBackend.calls)==1
    with pytest.raises(ValueError):recover(path)
    assert len(calls)==1 and not (setup[1]/'episodes'/'e0-d02'/'frozen.json').exists()


def test_interrupted_authorization_registration_resumes_without_model_work(legacy,monkeypatch):
    setup,calls=legacy;create=Ledger.create_job
    def interrupt(self,job_id,*args,**kwargs):
        if job_id.endswith('_schema_repair'):raise RuntimeError('fixture crash before job registration')
        return create(self,job_id,*args,**kwargs)
    monkeypatch.setattr(Ledger,'create_job',interrupt)
    with pytest.raises(RuntimeError,match='fixture crash'):prepare(legacy)
    p=setup[1]/'episodes'/'e0-d02'/'schema_repairs'/'ordinary_critique'/'authorization.json'
    before=sha256(p)
    monkeypatch.setattr(Ledger,'create_job',create)
    assert Path(prepare(legacy))==p and sha256(p)==before
    assert not calls and len(FixtureBackend.calls)==1
    assert recover(p)['edit_count']==6 and len(calls)==1


def test_throughput_freeze_rejects_timing_tamper_against_registered_job( setup):
    execute(setup)
    path=setup[1]/'episodes'/'e0-d01'/'edits'/'attempt-00.json'
    row=json.loads(path.read_text());row['measurements']['runtime_seconds']=0.01;write_json(path,row)
    with pytest.raises(SpecMismatch,match='artifact'):e0_runner.freeze_runtime(setup[1],setup[2],root=setup[0],retry_reserve=2)
    assert not (setup[1]/'runtime_freeze.json').exists()


def test_status_reconciliation_is_audited_and_preserves_attempt_and_charge_history(setup):
    execute(setup,brief_id='e0-d01')
    root,protocol,_,_=setup
    pid=json.loads((protocol/'manifest.json').read_text())['protocol_id']
    with Ledger(root/'jobs.sqlite3') as ledger:
        for stage in ('parent','actions','edits'):ledger.update_job(f'{pid}_e0-d01_{stage}',status='pending')
    result=e0_recovery.reconcile_status(protocol,root=root,brief_id='e0-d01')
    audit=Path(result['audits'][0]['path']);before=sha256(audit)
    assert result['model_calls']==0
    assert e0_recovery.reconcile_status(protocol,root=root,brief_id='e0-d01')['audits'][0]['sha256']==before
    row=json.loads(audit.read_text());assert all(change['previous_status']=='pending' for change in row['changes'])
    assert len(FixtureBackend.calls)==7 and len(setup[3])==3
    with Ledger(root/'jobs.sqlite3') as ledger:
        assert all(ledger.get_job(f'{pid}_e0-d01_{stage}')['status']=='completed' for stage in ('parent','actions','edits'))


def test_uncharged_repair_response_rejected_before_load(legacy):
    setup,calls=legacy;path=Path(prepare(legacy));write_json(path.parent/'response.json',{'raw_response':'untrusted uncharged fixture'})
    loads=FixtureDirector.loads
    with pytest.raises(ValueError,match='Uncharged'):recover(path)
    assert FixtureDirector.loads==loads and not calls


def test_invalid_charged_unregistered_repair_is_not_adopted_or_marked_completed(legacy):
    setup,calls=legacy;path=Path(prepare(legacy));a=json.loads(path.read_text());jid=a['repair_job_id']
    with SessionBudget(setup[0]) as budget,Ledger(setup[0]/'jobs.sqlite3') as ledger:
        budget.reserve(f'{jid}:reasoning:0','reasoning',max_tokens=8192,estimated_seconds=90)
        ledger.open_attempt(jid,0,payload={'kind':'reasoning'});ledger.reserve_call(jid,0)
        before=ledger.get_attempt(jid,0)
    write_json(path.parent/'response.json',{'raw_response':json.dumps(ORIGINAL),'output':{**ORIGINAL,'preserve':[ORIGINAL['preserve']]}})
    loads=FixtureDirector.loads
    with pytest.raises(ValueError):recover(path)
    with Ledger(setup[0]/'jobs.sqlite3') as ledger:
        assert ledger.get_attempt(jid,0)==before
        assert not any(row['path']=='response.json' for row in ledger.artifacts(jid))
    assert not calls and FixtureDirector.loads==loads


@pytest.mark.parametrize('stage,index,authority,status',[
    ('parent',0,'local','running'),('parent',0,'global','uncertain'),
    ('actions',0,'global','failed'),('actions',1,'global','completed'),
    ('actions',0,'local','uncertain'),('actions',1,'local','running')])
def test_original_local_and_global_status_must_be_settled_consistently(legacy,stage,index,authority,status):
    setup,calls=legacy;root,protocol,_,_=setup
    pid=json.loads((protocol/'manifest.json').read_text())['protocol_id'];jid=f'{pid}_e0-d02_{stage}'
    with Ledger(root/'jobs.sqlite3') as ledger,SessionBudget(root) as budget:
        if authority=='local':ledger.save_attempt(jid,index,status=status)
        else:
            # Inject a contradictory crash/corruption fixture; public settlement correctly forbids this mutation.
            budget.db.execute('UPDATE calls SET status=? WHERE key=?',(status,f'{jid}:{"render" if stage=="parent" else "reasoning"}:{index}'))
    with pytest.raises(SpecMismatch,match='explicit reconciliation'):prepare(legacy)
    assert not calls and len(FixtureBackend.calls)==1
