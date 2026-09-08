"""Proposal/execution-boundary fixtures; no real model inference or artistic labels."""
from pathlib import Path
import copy
from datetime import datetime, timezone
import json
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


def test_attempt_opened_before_local_reservation_cannot_recover_response(tmp_path):
    directory = tmp_path / 'run'; directory.mkdir()
    with SessionBudget(tmp_path, load_config()) as budget, Ledger(tmp_path/'jobs.sqlite3') as ledger:
        ledger.create_job('fixture', {'fixture_only': True}, 1, directory)
        budget.reserve('fixture:reasoning:0', 'reasoning', max_tokens=8192)
        ledger.open_attempt('fixture', 0)
        (directory/'response.json').write_text(json.dumps({'output': {'not_called': True}}))
        with pytest.raises(SpecMismatch, match='Uncharged'):
            model_call(ledger, 'fixture', directory, budget, 0, 'response.json', lambda: pytest.fail('No dispatch allowed'), device='cpu')
        assert ledger.call_count('fixture') == 0
        assert budget.summary()['calls'][0]['status'] == 'running'


def test_returned_failure_accounting_is_identical_on_replay(tmp_path):
    directory = tmp_path/'run'; directory.mkdir()
    with SessionBudget(tmp_path, load_config()) as budget, Ledger(tmp_path/'jobs.sqlite3') as ledger:
        ledger.create_job('fixture', {'fixture_only': True}, 1, directory)
        value = {'parse_error': 'Fixture schema failure', 'raw_response': '{}', 'metadata': {'input_tokens': 2, 'generated_tokens': 1}}
        first = model_call(ledger, 'fixture', directory, budget, 0, 'response.json', lambda: value, device='cpu')
        assert model_call(ledger, 'fixture', directory, budget, 0, 'response.json', lambda: pytest.fail('No replay dispatch'), device='cpu') == first
        assert budget.summary()['calls'][0]['status'] == 'failed'


def metadata_fixture(tmp_path, *, missing_render=False):
    from PIL import Image
    from kusama.artifacts import sha256, write_json
    from kusama.inspiration import wire_example, PROMPT_VERSION
    from kusama.resources import OvernightConfig
    root = tmp_path/'artifacts'; directory = root/'runs'/'fixture'
    (directory/'sources').mkdir(parents=True)
    config = OvernightConfig.model_validate({**load_config().model_dump(), 'device': 'cpu'})
    with SessionBudget(root, config) as budget:
        budget.reserve('fixture:reasoning:0', 'reasoning', max_tokens=8192)
        budget.settle('fixture:reasoning:0', 'failed', 3)
    docs = []
    for i in range(2):
        path = directory/'sources'/f'source{i}.png'
        Image.new('RGB', (64, 64), ('red', 'blue')[i]).save(path)
        docs.append({'id': f'source{i}', 'reference_id': f'source{i}', 'region_id': None,
                     'path': path.name, 'sha256': sha256(path),
                     'reference': {'title': 'Fixture source', 'url': 'https://example.invalid/fixture'}})
    value = wire_example('c01', docs)
    for field in ('id', 'title', 'question'):
        value.pop(field)
    for source in value['sources']:
        source.pop('level')
    if missing_render:
        value.pop('render_instruction')
    spec = {'kind': 'connection_inspiration', 'prompt_version': PROMPT_VERSION,
            'brief': 'Fixture external brief', 'directions': 1, 'source_documents': docs,
            'source_pairs': [['source0', 'source1']], 'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'whole', 'local_need': None, 'whole_context': None,
            'source_assignment_origin': 'fixture_operator_policy', 'shelf': {'selected': []},
            'config': config.model_dump()}
    with Ledger(root/'jobs.sqlite3') as ledger:
        ledger.create_job('fixture', spec, 2, directory)
        ledger.open_attempt('fixture', 0); ledger.reserve_call('fixture', 0)
        write_json(directory/'job.json', {'schema_version': 1, 'job_id': 'fixture', 'database': str(ledger.path)})
        write_json(directory/'proposal_c01.json', {'raw_response': json.dumps(value), 'parse_error': 'Fixture missing metadata'})
        ledger.register_artifact('fixture', 'proposal_c01.json', attempt_index=0, kind='response')
        ledger.save_attempt('fixture', 0, status='failed')
    write_json(directory/'manifest.json', {'connections': [], 'rejections': [{'id': 'c01', 'error': 'Fixture original rejection'}]})
    return root, directory, spec, value


def test_metadata_revalidation_only_adds_attributed_static_values(tmp_path):
    from kusama.artifacts import sha256
    from kusama.inspiration import revalidate_metadata, load_metadata_sidecar
    root, directory, spec, raw = metadata_fixture(tmp_path)
    original = sha256(directory/'proposal_c01.json')
    before = sha256(root/'resources.sqlite3')
    report = revalidate_metadata(directory, root)
    assert report['connections'] == 1
    sidecar = json.loads((directory/'c01_metadata_revalidation.json').read_text())
    assert sidecar['model_calls'] == 0
    assert sidecar['output']['render_instruction'] == raw['render_instruction']
    assert sidecar['output']['transformation'] == raw['transformation']
    assert [row['level'] for row in sidecar['output']['sources']] == [None, None]
    assert sidecar['output']['question'] == spec['brief']
    assert set(sidecar['additions']) == {'id', 'title', 'question', 'sources[0].level', 'sources[1].level'}
    assert sha256(directory/'proposal_c01.json') == original
    assert sha256(root/'resources.sqlite3') == before
    with Ledger(root/'jobs.sqlite3') as ledger:
        assert ledger.call_count('fixture') == 1
        assert load_metadata_sidecar(ledger, 'fixture', directory, spec, 'c01', spec['source_documents']) == sidecar
    assert revalidate_metadata(directory, root)['connections'] == 1


def test_metadata_revalidation_cannot_supply_missing_render_instruction(tmp_path):
    from kusama.inspiration import revalidate_metadata
    root, directory, _, _ = metadata_fixture(tmp_path, missing_render=True)
    assert revalidate_metadata(directory, root)['connections'] == 0
    assert not (directory/'c01_metadata_revalidation.json').exists()


def test_unregistered_sidecar_semantic_substitution_is_recomputed_and_rejected(tmp_path):
    from kusama.inspiration import _metadata_value, _sidecar_value, load_metadata_sidecar
    root, directory, spec, _ = metadata_fixture(tmp_path)
    original = directory/'proposal_c01.json'; result = json.loads(original.read_text())
    output, additions = _metadata_value(result, 'c01', spec['brief'], spec['source_documents'])
    sidecar = _sidecar_value(original, result, output, additions, {'fixture_only': 'no real model'})
    sidecar['output']['render_instruction'] = 'An entirely different substituted action.'
    (directory/'c01_metadata_revalidation.json').write_text(json.dumps(sidecar))
    with Ledger(root/'jobs.sqlite3') as ledger:
        with pytest.raises(SpecMismatch, match='original content'):
            load_metadata_sidecar(ledger, 'fixture', directory, spec, 'c01', spec['source_documents'])


def test_normal_resume_replays_metadata_without_model_loading_or_attribution_loss(tmp_path):
    from kusama.inspiration import revalidate_metadata, inspire
    root, directory, _, _ = metadata_fixture(tmp_path)
    revalidate_metadata(directory, root)
    with Ledger(root/'jobs.sqlite3') as ledger:
        ledger.create_job('fixture_render', {'fixture_only': True}, 1, directory/'generation')
        ledger.update_job('fixture_render', status='complete', payload={'state': {
            'images': [{'id': 'fixture_saved_child'}], 'status': 'needs_human_review'}})
    class NoLoadDirector:
        model = None
        def __init__(self, config): pass
        def prepare(self): pytest.fail('Metadata replay must not load a model')
        def close(self): pass
    assert inspire(root=root, resume=directory, director_factory=NoLoadDirector) == directory
    manifest = json.loads((directory/'manifest.json').read_text())
    assert len(manifest['connections']) == 1
    assert 'coordinator metadata' in manifest['connection_authorship']
    assert manifest['original_rejections'] == [{'id': 'c01', 'error': 'Fixture original rejection'}]
    assert manifest['images'] == [{'id': 'fixture_saved_child'}]
    assert manifest['generation_status'] == 'needs_human_review'


def test_renderer_attempt_before_reservation_cannot_adopt_an_image(tmp_path):
    from PIL import Image
    from test_episodes import source_run, FixtureBackend
    from kusama import episodes
    root, directory, run_id = source_run(tmp_path)
    original = Ledger.reserve_call
    def crash_before_local_charge(self, job_id, index):
        if job_id == run_id + '_render':
            raise KeyboardInterrupt('Fixture crash before local reservation')
        return original(self, job_id, index)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Ledger, 'reserve_call', crash_before_local_charge)
        with pytest.raises(KeyboardInterrupt):
            episodes.render(directory, root=root, backend_factory=FixtureBackend)
    path = directory/'generation'/'c01_s17.png'
    Image.new('RGB', (64, 64), 'green').save(path)
    with pytest.raises(SpecMismatch, match='Uncharged'):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    with Ledger(root/'jobs.sqlite3') as ledger:
        assert ledger.call_count(run_id+'_render') == 0


def test_optional_memory_is_cutoff_safe_bounded_and_replayed_exactly(tmp_path, monkeypatch):
    from PIL import Image
    from kusama import inspiration, ledger as ledger_module
    from kusama.artifacts import sha256
    from kusama.resources import OvernightConfig
    from test_memory import make_job, add_image
    root = tmp_path/'artifacts'
    real_now = ledger_module._now
    for episode, partition in [('prior_dev', 'development'), ('prior_sealed', 'sealed_test')]:
        ledger, prior_directory = make_job(root, monkeypatch, episode, partition)
        add_image(ledger, prior_directory, episode, color='red' if partition == 'development' else 'blue')
        ledger.close()
    monkeypatch.setattr(ledger_module, '_now', real_now)
    config = OvernightConfig.model_validate({**load_config().model_dump(), 'device': 'cpu'})
    with SessionBudget(root, config): pass
    index_path = tmp_path/'index'; index_path.mkdir(); (index_path/'manifest.json').write_text('{}')
    docs = []
    source_paths = {}
    for i in range(2):
        path = tmp_path/f'source{i}.png'; Image.new('RGB', (64, 64), ('red', 'blue')[i]).save(path)
        identifier = f'source{i}'; source_paths[identifier] = path
        docs.append({'id': identifier, 'reference_id': identifier, 'region_id': None,
                     'path': path.name, 'sha256': sha256(path),
                     'reference': {'title': 'Fixture source', 'url': 'https://example.invalid/fixture'}})
    class FixtureIndex:
        def __init__(self, path): self.by_id = {d['id']: d for d in docs}
        def query(self, *args, **kwargs): return {'selected': [{'id': d['id']} for d in docs]}
        def asset(self, identifier): return source_paths[identifier]
    class FixtureClip:
        def __init__(self, config): pass
        def text(self, value): return None, {'fixture_only': True}
        def close(self): pass
    class FixtureDirector:
        model = None
        loads = 0
        def __init__(self, config): pass
        def prepare(self): self.model = object(); type(self).loads += 1; return {'fixture_only': True}
        def close(self): self.model = None
    requests = []
    def fixture_call(director, system, request, images, validate, **kwargs):
        requests.append(copy.deepcopy(request))
        value = copy.deepcopy(request['response_example'])
        value['title'] = 'Fixture ' + value['id']
        value['transformation'] = 'Fixture transformation ' + value['id']
        return {'output': validate(value), 'metadata': {'input_tokens': 8, 'generated_tokens': 8}}
    monkeypatch.setattr(inspiration, 'RetrievalIndex', FixtureIndex)
    monkeypatch.setattr(inspiration, 'ClipEncoder', FixtureClip)
    monkeypatch.setattr(inspiration, '_multimodal_json', fixture_call)
    directory = inspiration.inspire(index_path, 'Enclosure support', root=root, run_id='next', use_memory=True, director_factory=FixtureDirector)
    assert len(requests) == 6
    excerpts = [request['prior_experiment_notes'] for request in requests]
    assert len(set(excerpts)) == 1 and len(excerpts[0]) <= 1800
    memory = json.loads((directory/'memory.json').read_text())
    snapshot = memory['candidate_snapshot']
    assert snapshot['selected'] and {event['episode_id'] for event in snapshot['selected']} == {'prior_dev'}
    assert all(event['available_at'] < snapshot['request']['decision_at'] for event in snapshot['selected'])
    original_memory = (directory/'memory.json').read_bytes()
    monkeypatch.setattr(inspiration, 'freeze_memory', lambda *args: pytest.fail('Resume cannot rebuild/query memory'))
    inspiration.inspire(root=root, resume=directory, director_factory=FixtureDirector)
    assert len(requests) == 6 and FixtureDirector.loads == 1
    assert (directory/'memory.json').read_bytes() == original_memory


def test_exhausted_renderer_budget_rejects_before_model_load(tmp_path):
    from test_episodes import source_run, FixtureBackend
    from kusama import episodes
    from kusama.ledger import BudgetExceeded
    root, directory, _ = source_run(tmp_path, cap=1)
    with SessionBudget(root) as budget:
        budget.reserve('fixture_already_charged', 'renderer')
        budget.settle('fixture_already_charged', 'completed')
    FixtureBackend.loads = 0
    with pytest.raises(BudgetExceeded, match='before model load'):
        episodes.render(directory, root=root, backend_factory=FixtureBackend)
    assert FixtureBackend.loads == 0
