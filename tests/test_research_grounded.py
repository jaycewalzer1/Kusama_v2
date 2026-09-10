"""V13 rules, distinct from retained legacy-protocol/recovery fixtures."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from PIL import Image
from kusama import research as r, research_rules as rules
from kusama.artifacts import sha256
from kusama.research_context import project_context

TEXT = 'The figures have exaggerated proportions and open space. The king depends on others.'

class Sources:
    def __init__(self, root): self.root = root; self.calls = []
    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        provider = kwargs.get('provider', 'local')
        return {'sources': [{'source_id': provider, 'candidate_id': provider, 'provider': provider,
                 'title': 'Figure with open space', 'description': TEXT, 'url': 'https://example.org/'+provider}]}
    def acquire(self, sid): return self.open(sid + '_canonical')
    def open(self, sid):
        self.root.mkdir(exist_ok=True)
        path = self.root / (sid + '.png'); Image.new('RGB', (32, 32), 'red').save(path)
        return {'source_id': sid, 'text': TEXT, 'claims': [{'id': sid+'_claim', 'claim': TEXT}],
                'images': [{'path': str(path), 'sha256': sha256(path)}], 'reference': {}, 'provenance': {}}


def hyp(ids=('artic', 'met')):
    return {'id': 'hypothesis_01', 'title': 'Experiment', 'question': 'Figure agency',
      'sources': [{'source_id': sid, 'role': 'Transfer geometry',
         'transferred_property': 'exaggerated proportions and open space',
         'transferred_property_evidence': {'quote': 'The figures have exaggerated proportions and open space.'},
         'discarded_property': 'The king depends on others',
         'discarded_property_evidence': {'quote': 'The king depends on others.'}} for sid in ids],
      'relationship': 'Vary figure scale', 'transformation': 'Assemble a new figure from fragments',
      'composition': 'A tall central figure', 'render_instruction': 'Paint exaggerated proportions and open space in red.',
      'formal_properties': ['exaggerated proportions', 'open space'], 'visible_test': 'Figure height exceeds half the canvas.',
      'visible_predicates': [{'id': 'height', 'subject': 'central figure', 'measure': 'height_fraction', 'operator': '>', 'value': .5}],
      'uncertainty': 'Visual transfer unverified'}


def analysis():
    return {'action': 'analyze_question', 'reason': 'Decompose inquiry', 'question_analysis': {
      'subject': 'figures', 'medium': 'collage', 'era': 'unknown', 'key_concepts': ['agency'], 'likely_artists_movements': ['unknown']},
      'queries': [{'query': 'figure', 'field': 'subject'}, {'query': 'collage', 'field': 'technique'}, {'query': 'portrait', 'field': 'object'}]}

@pytest.fixture
def session(tmp_path):
    spec = r.plan_research(tmp_path / 'run', {'id': 'rules', 'project_id': 'p', 'brief_id': 'b', 'problem': 'How can figures express agency?',
      'authorship': 'test operator', 'resource_root': str(tmp_path / 'budget'), 'max_steps': 48, 'fixture': True})
    spec.pop('draft_evidence_policy', None)  # This module exercises the retained V13 contract.
    return tmp_path / 'run', spec, [], Sources(tmp_path / 'sources')


def execute(session, value):
    root, spec, steps, backend = session
    value = {'reason': 'Fixture decision', **value}
    row = r._execute_action(root, spec, r._state(spec, steps), steps, r.ResearchActionEnvelope.model_validate(value), backend)
    steps.append(row)
    return row


def opened(session):
    execute(session, analysis())
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote', 'provider': 'met'})
    execute(session, {'action': 'assess_candidates', 'candidate_scores': [{'candidate_id': sid, 'score': .8, 'reason': 'Figure proportions address agency'} for sid in ('artic','met')]})
    for sid in ('artic', 'met'): execute(session, {'action': 'acquire', 'candidate_id': sid})


def observations(sid):
    return {'source_id': sid, 'image_index': 0, 'observations': ['A red diagonal stripe crosses the lower left corner.',
            'Three small circular holes pierce a horizontal base.'], 'supports': 'These observations support the proportions and negative spaces.'}


def test_queries_fanout_and_gate(session):
    execute(session, analysis())
    row = execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    assert len(session[3].calls) == 3
    assert all(c[0] == 'figure' for c in session[3].calls)
    assert all(h['matched_terms'] == ['figure'] for h in row['tool_result']['sources'])
    with pytest.raises(ValueError, match='(?s)relevance gate.*Next:',): execute(session, {'action':'acquire','candidate_id':'artic'})


def test_recall_pages_all_saved_candidates_without_repeating_search(session):
    execute(session, analysis())
    session[3].search = lambda query, **kwargs: {'sources': [
        {'candidate_id': kwargs['provider'] + str(i), 'source_id': kwargs['provider'] + str(i),
         'provider': kwargs['provider'], 'title': 'Complete retained title'} for i in range(8)]}
    execute(session, {'action': 'search', 'query': 'figure', 'mode': 'remote'})
    original = deepcopy(session[2][1])
    all_ids = []
    for offset in (0, 8, 16):
        row = execute(session, {'action': 'recall', 'recall_steps': [1], 'offset': offset, 'limit': 8})
        result = row['tool_result']['recalled'][0]['tool_result']
        all_ids.extend(s['candidate_id'] for s in result['sources'])
        assert result['sources_page']['total'] == 24
        payload = r._payload(session[1], r._state(session[1], session[2]), session[2])
        compact = payload['last_tool_result']['recalled'][0]['tool_result']
        assert compact['sources_page'] == result['sources_page']
    assert len(set(all_ids)) == 24 and session[2][1] == original


def test_alias_draft_attach_commit(session):
    opened(session)
    row = execute(session, {'action': 'draft', 'hypothesis': hyp()})
    assert row['executed_action']['hypothesis']['id'] == 'hypothesis_1'
    assert row['active_source_ids'] == ['artic_canonical', 'met_canonical']
    state = r._state(session[1], session[2])
    progress, _ = r._research_progress(session[1], state, session[2])
    assert progress['allowed_actions'] == ['commit','draft','reject','search','stop']
    assert all(not Path(i['path']).is_absolute() for s in row['opened_sources'] for i in s['images'])
    with pytest.raises(ValueError, match='unchanged since step'): execute(session, {'action':'draft','hypothesis':hyp()})
    with pytest.raises(ValueError, match='images already attached'): execute(session, {'action':'open','source_id':'artic'})
    with pytest.raises(ValueError, match='Reinspection requires'): execute(session, {'action':'commit','hypothesis':hyp()})
    row = execute(session, {'action':'commit','hypothesis':hyp(),'reinspection':[observations(sid) for sid in row['active_source_ids']]})
    assert row['committed_hypothesis']['id'] == 'hypothesis_1'


def test_grounding_fail_closed_and_visual_contract(session):
    opened(session)
    bad = hyp(); bad['sources'][0]['transferred_property'] = 'The king is independent of everyone'
    with pytest.raises(ValueError, match='entailment failed'): execute(session, {'action':'draft','hypothesis':bad})
    bad = hyp(); bad['sources'][0]['transferred_property_evidence'] = {'quote':'Invented source'}
    with pytest.raises(ValueError, match='grounding unverified'): execute(session, {'action':'draft','hypothesis':bad})
    bad=hyp();bad['visible_test']='The viewer should sense power'
    with pytest.raises(ValueError, match='Visible test'): execute(session, {'action':'draft','hypothesis':bad})
    bad=hyp();bad['render_instruction']='Paint a figure'
    with pytest.raises(ValueError, match='formal property'): execute(session, {'action':'draft','hypothesis':bad})
    with pytest.raises(ValueError, match='2 opened sources'): execute(session, {'action':'draft','hypothesis':hyp(('artic',))})


def test_caption_paraphrases_rejected(session):
    opened(session); execute(session, {'action':'draft','hypothesis':hyp()})
    data = [observations(sid) for sid in ('artic_canonical','met_canonical')]
    data[0]['observations'] = ['The figures have exaggerated proportions.', 'There is open space between figures.']
    with pytest.raises(ValueError, match='caption'): execute(session, {'action':'commit','hypothesis':hyp(),'reinspection':data})


def test_synthesis_only_pending_commit_or_stop(session):
    opened(session); execute(session, {'action':'draft','hypothesis':hyp()})
    session[1]['request']['max_steps'] = len(session[2]) + 3
    progress, _ = r._research_progress(session[1], r._state(session[1],session[2]),session[2])
    assert progress['allowed_actions'] == ['commit','stop']


def test_zero_result_requires_different_query(session):
    execute(session,analysis())
    session[3].search = lambda *a, **kw: {'sources':[]}
    row = execute(session,{'action':'search','query':'figure','mode':'remote'})
    assert 'different query' in row['tool_result']['message']
    with pytest.raises(ValueError,match='Repeated action|different query'): execute(session,{'action':'search','query':'figure','mode':'remote'})
    execute(session,{'action':'search','query':'collage','mode':'remote'})


def test_reason_loop_and_error_weights(session):
    with pytest.raises(ValueError,match='nonempty reason'): execute(session,{'action':'stop','reason':None})
    execute(session,analysis())
    execute(session,{'action':'note','note':'An observation'})
    with pytest.raises(ValueError,match='same args'): execute(session,{'action':'note','note':'An observation','reason':'Changed rationale'})
    session[2].append({'index':len(session[2]),'action':{'action':'note'},'tool_error':'invalid'})
    progress,_=r._research_progress(session[1],r._state(session[1],session[2]),session[2])
    assert progress['actions_without_progress']==3
    assert progress['last_no_progress_reason']=='invalid'


def test_projection_preserves_schema_titles_and_analysis(session):
    execute(session,analysis()); spec=session[1];state=r._state(spec,session[2])
    context={'system':r.SYSTEM,'payload':r._payload(spec,state,session[2]),'images':[]}
    for stage in range(7):
        projected=project_context(context,stage)
        assert projected['payload']['output_schema']==context['payload']['output_schema']
        assert projected['payload']['question_analysis']==state['analysis']


def test_predicate_comparisons():
    p=hyp()['visible_predicates']
    assert rules.evaluate_predicates(p,{'height':.6})['status']=='PASS'
    for value in (None,.5,'0.6',True,2):
        assert rules.evaluate_predicates(p,{'height':value})['status']=='FAIL'


def test_no_material_exit_after_bounded_reformulations(session):
    execute(session, analysis()); session[1]['request']['max_reformulations']=1
    session[3].search=lambda *a, **kw: {'sources': []}
    for q in ('figure','collage','portrait'): execute(session,{'action':'search','mode':'remote','query':q})
    rewrite={'action':'rewrite_queries','queries':[{'query':q,'field':'subject'} for q in ('torso','silhouette','fragment')]}
    execute(session,rewrite)
    for q in ('torso','silhouette','fragment'): execute(session,{'action':'search','mode':'remote','query':q})
    progress,_=r._research_progress(session[1],r._state(session[1],session[2]),session[2])
    assert progress['no_relevant_material'] and progress['stop_reason']==rules.NO_MATERIAL
    assert r._terminal_status(session[1],session[2])=='no_relevant_material'


def test_combined_render_contract_and_ranked_retention(session, monkeypatch):
    opened(session)
    session[1]['request']['render_prompt_contract']={'model_revision':'a'*40,'max_sequence_length':512}
    seen=[]
    def count(text, **kwargs): seen.append(text); return 513
    monkeypatch.setattr('kusama.backend.cached_prompt_token_count',count)
    with pytest.raises(ValueError,match='FLUX allows 512'): execute(session,{'action':'draft','hypothesis':hyp()})
    assert seen==[hyp()['composition']+'\n'+hyp()['render_instruction']]
    state=r._state(session[1],session[2]);state['scores']['cleveland']={'score':.95,'reason':'test','question':state['question']}
    _,candidates=r._research_progress(session[1],state,session[2])
    assert candidates[0]['candidate_id']=='cleveland'
    state['rejected'].add('cleveland')
    _,candidates=r._research_progress(session[1],state,session[2])
    assert not any(c['candidate_id']=='cleveland' for c in candidates)


def test_current_run_ledger_resume_render_and_judge(tmp_path, monkeypatch):
    from kusama.resources import SessionBudget, load_config
    from kusama import research_render, research_judge
    from kusama.config import Config
    from test_research import Director
    from test_research_render import Renderer
    resource=tmp_path/'budget'; output=tmp_path/'research'; render=tmp_path/'render'
    cfg=load_config().model_copy(update={'session_hours':24,'handoff_reserve_minutes':0,'device':'cpu'})
    with SessionBudget(resource,cfg): pass
    r.plan_research(output, {'id':'integration','project_id':'p','brief_id':'b','problem':'How can figures express agency?',
       'authorship':'test operator','resource_root':str(resource),'max_steps':24,'max_hypotheses':1,'fixture':True,'device':'cpu'})
    actions=[analysis(),{'action':'search','query':'figure','mode':'remote'},
      {'action':'assess_candidates','candidate_scores':[{'candidate_id':sid,'score':.8,'reason':'Useful proportions'} for sid in ('artic','met')]},
      {'action':'acquire','candidate_id':'artic'}, {'action':'inspect_images','reinspection':[observations('artic_canonical')]},
      {'action':'acquire','candidate_id':'met'}, {'action':'inspect_images','reinspection':[observations('met_canonical')]},
      {'action':'draft','hypothesis':hyp()},
      {'action':'commit','hypothesis':hyp(),'reinspection':[observations(sid) for sid in ('artic_canonical','met_canonical')]}]
    count=[]
    def model(director, system, payload, images, validator, **kwargs):
        value={'reason':'Fixture decision',**actions[len(count)]};count.append(payload)
        return {'output':validator(value),'raw_response':json.dumps(value),'metadata':{'input_tokens':100,'generated_tokens':100}}
    monkeypatch.setattr(r,'_multimodal_json',model)
    factory=lambda root, *a, **kw: Sources(root)
    report=r.run_research(output,director_factory=Director,sources_factory=factory)
    assert report['status']=='hypotheses_ready' and report['result']=='FAIL'
    assert report['acceptance_criteria']['two_relevant_opened_sources']
    assert report['acceptance_criteria']['zero_grounding_contradictions']
    assert not report['acceptance_criteria']['visible_test_passes_on_render']
    assert report['vision_probe']=='description compared to withheld caption'
    assert count[4]['source_context'][0]['text']=='[caption withheld for image probe]'
    assert not any(c.get('description')==TEXT for c in count[4]['candidate_memory'] if c['candidate_id']=='artic')
    assert r.run_research(output,director_factory=Director,sources_factory=factory)['hypotheses']==report['hypotheses']
    assert len(count)==9
    config=Config(output_dir=render,cache_dir=tmp_path/'cache',device='cpu',precision='fp32',local_files_only=True,width=64,height=64,steps=1)
    render_spec=research_render.plan_render(render,{'id':'v13','research_output':str(output),'hypothesis_ids':['hypothesis_1'],'seeds':[17],
       'resource_root':str(resource),'render_profile_id':'fixture','render_config':config.to_dict(),'selection_origin':'all_model_hypotheses','selection_reason':'Fixture'})
    assert render_spec['actions'][0]['prompt']==hyp()['composition']+'\n'+hyp()['render_instruction']
    rendered=research_render.run_render(render,backend_factory=Renderer)
    assert rendered['result']=='FAIL'  # No silent real judge for a fixture.
    def judge(director,system,payload,images,validator,**kwargs):
        value={'measurements':{'height':.7}}
        return {'output':validator(value),'raw_response':json.dumps(value),'metadata':{'input_tokens':100,'generated_tokens':20}}
    monkeypatch.setattr(research_judge,'_multimodal_json',judge)
    assessment=research_judge.run_judge(render,render_spec,rendered['images'],director_factory=Director)
    assert assessment['status']=='PASS'
    assert r.read_report(output)['result']=='PASS'
    (render/'assessment'/'response-000.json').write_text('{}')
    assert r.read_report(output)['result']=='FAIL'


@pytest.mark.parametrize('provider,field,expected',[('artic','artist','artist_title'),('artic','subject','subject_titles'),
    ('artic','technique','term_titles'),('artic','object','classification_titles'),('cleveland','artist','artists'),
    ('cleveland','object','type'),('cleveland','subject','q')])
def test_fielded_provider_queries(tmp_path,provider,field,expected):
    from kusama.research_sources import ResearchSources
    from test_research_source_audit import Transport
    from urllib.parse import urlsplit,parse_qs
    transport=Transport(provider)
    with_backend=ResearchSources(tmp_path,{'allow_network':True},opener=transport)
    try: with_backend.search('figure',mode='remote',provider=provider,limit=1,query_field=field)
    finally: with_backend.close()
    params=parse_qs(urlsplit(transport.urls[0]).query)
    if provider=='artic':
        d=json.loads(params['params'][0]); assert expected in d['query']['bool']['must'][0]['multi_match']['fields']
    else: assert params[expected]==['figure']


@pytest.mark.skipif(__import__('os').environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1', reason='Requires explicitly selected pinned local processor; no inference')
def test_current_six_image_prompt_admits_and_keeps_schema(session):
    from kusama.research_context import admit_context
    from kusama.director import DirectorConfig
    execute(session,analysis())
    spec=session[1];state=r._state(spec,session[2]);root=session[0]
    for sid in ('a','b','c','d','e','f'):
        state['opened'][sid]=r._snapshot_source(root,session[3].open(sid));state['active'].append(sid)
    state['draft']={'hypothesis':hyp(tuple(state['active'])),'step':1}
    context={'system':r.SYSTEM,'payload':r._payload(spec,state,session[2]),'images':r._images(spec,state)}
    projected,admission=admit_context(context,DirectorConfig(Path('.cache/huggingface').resolve(),max_new_tokens=1800,max_image_pixels=65536),base=root)
    assert admission['input_tokens']<=6392
    assert len(projected['images'])==6
    assert projected['payload']['output_schema']==context['payload']['output_schema']
    assert projected['payload']['pending_draft']==context['payload']['pending_draft']
