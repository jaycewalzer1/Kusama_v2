"""Tiny synthetic corpus/encoder fixtures; never evidence of real CLIP or source quality."""
import json
import numpy as np
import pytest

from kusama.artifacts import sha256
from kusama.corpus import CorpusStore
from kusama.resources import load_config
from kusama.retrieval import RetrievalIndex,build_index,normalized,audit
from test_corpus import reference


class FixtureEncoder:
    metadata={'model':'fixture-only','revision':load_config().clip_revision,'dimensions':512,'frozen':True}
    calls=0
    def prepare(self):return self.metadata
    def images(self,paths):
        self.calls+=1
        result=np.zeros((len(paths),512),dtype=np.float32)
        for n in range(len(paths)):result[n,n]=1
        return result
    def close(self):pass


@pytest.fixture
def tiny_index(tmp_path):
    store=CorpusStore(tmp_path/'corpus')
    for n,color in enumerate(['red','green','blue','yellow'],1):
        ref,regions=reference(store,n,color)
        family='same_source_family' if n in (1,2) else f'family_{n}'
        rights=ref.rights
        if n==4:rights=rights.model_copy(update={'image_display':rights.image_display.model_copy(update={'status':'excluded'})})
        ref=ref.model_copy(update={'title':f'Fixture enclosure {n}','source_family':family,'split_group':family,'rights':rights})
        regions=[r.model_copy(update={'rights':rights,'split_group':family}) for r in regions]
        store.insert(ref,regions,[])
    enc=FixtureEncoder()
    path=build_index(store.root,tmp_path/'indexes',config=load_config(),encoder=enc)
    return RetrievalIndex(path),store,enc


def test_rights_whole_part_and_family_dedup(tiny_index):
    index,store,enc=tiny_index
    assert len(index.documents)==6
    assert 'cleveland_4' not in index.by_id
    result=index.query('enclosure',limit=8,query_vector=np.eye(1,512,0)[0])
    assert result['visual_available'] and all(x['id'] in index.by_id for x in result['selected'])
    assert len(result['selected'])==2
    assert len({index.by_id[x['id']]['source_family'] for x in result['selected']})==2
    assert set(result['channels'])=={'lexical','visual','relationship'}
    assert result['human_usefulness'] is None


def test_source_holdout_removes_parent_and_all_descendant_crops(tiny_index):
    index,_,_=tiny_index
    result=index.query('enclosure',exclude_source_families=['same_source_family'])
    assert {x['reference_id'] for x in result['selected']}=={'cleveland_3'}
    assert all(index.by_id[identifier]['reference_id']=='cleveland_3' for ids in result['channels'].values() for identifier in ids)
    result=index.query('enclosure',allowed_reference_ids=['cleveland_1'])
    assert {x['reference_id'] for x in result['selected']}=={'cleveland_1'}


def test_part_queries_require_local_need_and_whole_context(tiny_index):
    index,_,_=tiny_index
    with pytest.raises(ValueError,match='local need'):index.query('vessel',scope='part')
    result=index.query('vessel',scope='part',local_need='opening in boundary',whole_context='Keep a coherent whole composition')
    assert result['selected'] and all(item['region_id'] for item in result['selected'])


def test_index_reuse_and_source_tampering(tiny_index,tmp_path):
    index,store,enc=tiny_index
    other=FixtureEncoder();path=build_index(store.root,tmp_path/'indexes',config=load_config(),encoder=other)
    assert path==index.directory and other.calls==0
    identifier=index.documents[0]['id'];asset=index.asset(identifier);asset.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen index'):index.asset(identifier)


def test_index_file_tampering_is_detected(tiny_index):
    index,_,_=tiny_index
    (index.directory/'documents.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):RetrievalIndex(index.directory)


def test_lexical_query_treats_source_style_text_as_data(tiny_index):
    index,_,_=tiny_index
    result=index.query('" OR 1=1; DROP TABLE documents; -- enclosure')
    assert result['selected']
    assert index.query('enclosure')['selected']


@pytest.mark.parametrize('bad',[np.zeros(512),np.ones(511),np.full(512,np.nan),np.ones((2,3,512))])
def test_descriptor_contract_rejects_invalid_vectors(bad):
    with pytest.raises(ValueError):normalized(bad)


def test_content_addressed_identity_cannot_be_rewritten_by_updating_one_file_hash(tiny_index):
    index,_,_=tiny_index
    path=index.directory/'documents.json';data=json.loads(path.read_text())
    data['documents'][0]['documented_text']='Changed future-outcome content'
    path.write_text(json.dumps(data))
    manifest_path=index.directory/'manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['files']['documents.json']=sha256(path);manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='identity changed'):RetrievalIndex(index.directory)


def test_audit_does_not_claim_complete_gate_for_empty_detail_shelves(tmp_path):
    store=CorpusStore(tmp_path/'corpus');ref,_=reference(store);store.insert(ref,[],[])
    path=build_index(store.root,tmp_path/'indexes',config=load_config(),encoder=FixtureEncoder())
    result=audit(RetrievalIndex(path),tmp_path/'audit.json')
    assert result['query_count']==12
    assert result['status']!='grounded_returns_verified'
    assert result['empty_query_ids']


def test_source_audit_is_frozen_in_new_index_identity_and_drops_parent_and_crops(tiny_index,tmp_path):
    index,store,_=tiny_index
    audit_path=store.root/'source_visual_audit.json'
    audit_path.write_text(json.dumps({'author':'fixture inspection','shortlist_exclusions':[{'reference_id':'cleveland_3','reason':'Fixture faint specimen'}]}))
    path=build_index(store.root,tmp_path/'indexes',config=load_config(),encoder=FixtureEncoder())
    newer=RetrievalIndex(path)
    assert newer.manifest['source_audit_excluded_count']==1
    assert all(doc['reference_id']!='cleveland_3' for doc in newer.documents)
    assert path!=index.directory and 'cleveland_3' in index.by_id
    audit_path.write_text(json.dumps({'author':'later inspection','shortlist_exclusions':[]}))
    newer.verify()
    assert newer.query('enclosure')['source_audit_exclusions'][0]['reference_id']=='cleveland_3'


def test_device_questions_and_proposed_effects_cannot_become_observed_edges():
    from kusama.retrieval import source_edges
    device={'id':'seed','reference_ids':['r1'],'observable_relationship':'Look for a frame that encloses; this is an inspection question.',
            'proposed_transfer':'Conceal the edges and repeat the shape','evidence_claim_ids':[]}
    assert source_edges([device],[])==[]


def test_typed_relation_channel_and_distant_selection_use_structural_evidence(tmp_path):
    from kusama.corpus import ContextClaim
    store=CorpusStore(tmp_path/'corpus');edges=[]
    families=['domain_a','domain_a','domain_b','domain_c','domain_b','domain_c']
    for n,(family,color) in enumerate(zip(families,['red','green','blue','yellow','orange','pink']),1):
        ref,_=reference(store,n,color)
        ref=ref.model_copy(update={'source_family':f'fixture_source_{n}','split_group':f'fixture_source_{n}','object_family':family})
        claim=ContextClaim(id=f'fixture_claim_{n}',reference_id=ref.id,claim='Fixture boundary surrounds a space',supporting_url=ref.url,
          passage_locator='Synthetic fixture only',status='machine_interpretation',author='fixture',rights=ref.rights.metadata)
        store.insert(ref,[],[claim])
        edges.append({'id':f'fixture_edge_{n}','reference_id':ref.id,'predicate':'encloses','statement':claim.claim,'evidence_ids':[claim.id],'origin':'operator_hypothesis'})
    path=build_index(store.root,tmp_path/'indexes',config=load_config(),encoder=FixtureEncoder(),relationship_edges=edges)
    index=RetrievalIndex(path)
    result=index.query('enclosure',scope='whole',limit=6,allocation={'relevant':4,'distant':2,'underrepresented':0})
    assert result['relationship_query']['brief_predicates']==['encloses']
    distant=[item for item in result['selected'] if item['allocation']=='distant']
    assert len(distant)==2 and result['shortages']['distant']==0
    assert result['distant_policy']['core_object_families']==['domain_a']
    assert all(item['relationship_matches'][0]['predicate']=='encloses' for item in distant)
    assert all(item['relationship_matches'][0]['validation']=='unvalidated' for item in distant)
    irrelevant=index.query('a poetic title with no structural cue',scope='whole')
    assert irrelevant['channels']['relationship']==[]
    context=index.query('enclosure',scope='whole',local_need='conceal the opening',whole_context='repeat the boundary')
    assert context['relationship_query']['local_predicates']==['conceals']
    assert context['relationship_query']['whole_predicates']==['encloses','repeats']


def test_new_predicates_require_explicit_source_grounded_edges():
    from kusama.retrieval import source_edges,predicate_matches,RelationshipEdge
    claim={'id':'claim_1','reference_id':'r1','status':'machine_interpretation','claim':'Fixture note'}
    edge={'id':'edge_1','reference_id':'r1','predicate':'interweaves','statement':'Proposed non-universal relation','evidence_ids':['claim_1'],'origin':'operator_hypothesis'}
    admitted=source_edges([], [claim], [edge])
    assert admitted[0]['predicate']=='interweaves'
    assert 'interweaves' in predicate_matches('a structure that interweaves',{'interweaves'})
    with pytest.raises(ValueError,match='resolve'):source_edges([], [claim], [{**edge,'reference_id':'r2'}])
    with pytest.raises(ValueError,match='documented'):source_edges([], [claim], [{**edge,'origin':'documented_source_statement'}])
    with pytest.raises(ValueError):RelationshipEdge.model_validate({**edge,'shell_command':'not permitted'})


def test_audit_visual_inputs_include_whole_context_for_detail_queries(tiny_index,tmp_path):
    index,_,_=tiny_index
    class TextFixture:
        def __init__(self):self.texts=[]
        def text(self,value):
            self.texts.append(value)
            return np.eye(1,512,0)[0],{'fixture_only':True,'admitted_text':value}
    encoder=TextFixture();result=audit(index,tmp_path/'audit_context.json',encoder=encoder)
    for row,admitted in zip(result['queries'],encoder.texts,strict=True):
        if row['scope']=='part':
            assert row['local_need'] in admitted and row['whole_context'] in admitted
            assert admitted.count(row['local_need'])==1


def test_query_uses_frozen_index_predicate_vocabulary(tiny_index,monkeypatch):
    from kusama.retrieval import PREDICATE_CUES
    index,_,_=tiny_index
    monkeypatch.setitem(PREDICATE_CUES,'encloses',['unrelated_new_policy'])
    assert 'encloses' in index.query('enclosure')['relationship_query']['brief_predicates']
