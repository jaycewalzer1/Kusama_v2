"""Fixture contracts; these tests do not prove live ingestion or source interpretation."""
import io
import json
from pathlib import Path
import urllib.error

from PIL import Image
import pytest
from pydantic import ValidationError

from kusama.corpus import (Asset, BoundedFetcher, CorpusReference, CorpusStore,
    Region, SourceConfig, digest, ingest_record, make_regions, normalize,
    prepare_assets, rights_for, verify_corpus)


def png(color='red'):
    out=io.BytesIO();Image.new('RGB',(180,240),color).save(out,'PNG');return out.getvalue()


def cleveland(identifier=1):
    return {'id':identifier,'title':'Fixture vessel','url':f'https://clevelandart.org/art/{identifier}',
      'share_license_status':'CC0','images':{'web':{'url':f'https://openaccess-cdn.clevelandart.org/{identifier}.jpg'}},
      'technique':'stone','type':'Vessel','culture':['Documented place'],'creation_date':'1700',
      'description':'The museum describes a carved vessel. This is a source statement.',
      'measurements':'10 cm','creditline':'Fixture credit','creators':[]}


def reference(store,identifier=1,color='red'):
    data=cleveland(identifier);raw=json.dumps(data).encode();rawpath=f'raw/{identifier}.json'
    store.asset_path(rawpath).parent.mkdir(parents=True,exist_ok=True);store.asset_path(rawpath).write_bytes(raw)
    fields,url=normalize('cleveland',data,{},rawpath,digest(raw),'2026-09-08T00:00:00Z','vessel')
    original,view,image=prepare_assets(store,png(color),url,fields['id'],SourceConfig())
    ref=CorpusReference(**fields,original=original,view=view)
    regions=make_regions(store,ref,image,SourceConfig())
    return ref,regions


def test_same_source_and_image_are_immutable_and_deduplicated(tmp_path):
    store=CorpusStore(tmp_path);ref,regions=reference(store)
    assert store.insert(ref,regions,[])
    assert not store.insert(ref,regions,[])
    other=ref.model_copy(update={'id':'cleveland_2','object_id':'2'})
    assert not store.insert(other,[],[])
    assert len(store.references())==1 and len(store.regions())==1
    assert store.get_reference(ref.id).original.sha256==ref.original.sha256
    assert verify_corpus(store)['passed']


@pytest.mark.parametrize('box',[(-.1,0,.8,.8),(0,0,1.1,1),(.5,0,.4,.8),(0,0,.01,.01),(0,0,float('nan'),1)])
def test_region_rejects_invalid_coordinates(tmp_path,box):
    store=CorpusStore(tmp_path);ref,regions=reference(store)
    with pytest.raises(ValidationError):Region.model_validate({**regions[0].model_dump(),'box':box})


def test_region_rights_parentage_and_split_inheritance(tmp_path):
    store=CorpusStore(tmp_path);ref,regions=reference(store)
    with pytest.raises(ValueError,match='inherit'):
        store.insert(ref,[regions[0].model_copy(update={'split_group':'held_out_other'})],[])
    assert len(store.references())==0
    for region in regions:
        assert region.rights==ref.rights and region.split_group==ref.split_group
        assert region.localization=='geometric_fallback'


def test_separate_description_rights_and_restricted_image_rules():
    rights=rights_for('artic',{'is_public_domain':True},'Art Institute attribution')
    assert rights.image_display.status=='allowed'
    assert rights.descriptive_text.status=='conditional'
    assert rights.descriptive_text.license=='CC-BY-4.0'
    assert rights.descriptive_text.attribution=='Art Institute attribution'
    assert rights_for('met',{'isPublicDomain':False},'x').image_display.status=='excluded'
    assert rights_for('smithsonian',{'_media_access':'Restricted','_metadata_access':'CC0'},'x').training.status=='excluded'


def test_response_cache_hash_validation_and_request_idempotence(tmp_path,monkeypatch):
    class Response:
        status=200;headers={}
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):return b'{"real_shape":"fixture"}'
    calls=[]
    def urlopen(request,timeout):calls.append(request.full_url);return Response()
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    monkeypatch.setattr('time.sleep',lambda n:None)
    store=CorpusStore(tmp_path);fetcher=BoundedFetcher(store,SourceConfig())
    first=fetcher.fetch('https://example.org/api');second=fetcher.fetch('https://example.org/api')
    assert first==second and len(calls)==1
    store.asset_path(first[1]).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='hash mismatch'):fetcher.fetch('https://example.org/api')
    assert len(calls)==1


def test_failed_request_attempts_persist_across_resume(tmp_path,monkeypatch):
    calls=[]
    def urlopen(*args,**kwargs):calls.append(1);raise TimeoutError('fixture interruption')
    monkeypatch.setattr('urllib.request.urlopen',urlopen);monkeypatch.setattr('time.sleep',lambda n:None)
    store=CorpusStore(tmp_path)
    for _ in range(2):
        with pytest.raises(RuntimeError):BoundedFetcher(store,SourceConfig()).fetch('https://example.org/api')
    assert len(calls)==3
    assert store.db.execute("SELECT COUNT(*) FROM events_v1 WHERE event='request_failed'").fetchone()[0]==3


def test_corrupt_image_is_not_accepted_and_success_reingest_preserves_counts(tmp_path):
    class Fetch:
        config=SourceConfig()
        data=b'not an image'
        def check_budget(self,*args):pass
        def fetch(self,*args,**kwargs):return self.data,'unused','unused','unused'
    store=CorpusStore(tmp_path);data=cleveland();raw=json.dumps(data).encode()
    rawpath='source.json';store.asset_path(rawpath).write_bytes(raw)
    fetcher=Fetch();args=(store,fetcher,'cleveland',data,{},(rawpath,digest(raw),'2026-09-08'),'fixture')
    assert not ingest_record(*args)
    assert not store.references()
    fetcher.data=png()
    assert ingest_record(*args)
    assert not ingest_record(*args)
    assert len(store.references())==1 and len(store.regions())==1
    assert len(store.claims())==2


def test_unknown_controls_and_paths_rejected(tmp_path):
    with pytest.raises(ValidationError):SourceConfig(max_gpu_calls=1)
    with pytest.raises(ValidationError):SourceConfig(target=301)
    with pytest.raises(ValueError):CorpusStore(tmp_path).asset_path('../../secret')
    store=CorpusStore(tmp_path);_,regions=reference(store)
    with pytest.raises(ValidationError):Region.model_validate({**regions[0].model_dump(),'id':'../../unsafe'})


def test_byte_budget_includes_existing_derivatives(tmp_path):
    (tmp_path/'derivatives.bin').write_bytes(b'x'*30000)
    fetcher=BoundedFetcher(CorpusStore(tmp_path),SourceConfig(max_bytes=1024))
    with pytest.raises(RuntimeError,match='byte budget'):fetcher.check_budget()


def test_changed_source_metadata_retains_original_and_conflict_evidence(tmp_path):
    store=CorpusStore(tmp_path);ref,regions=reference(store)
    store.insert(ref,regions,[])
    changed=ref.model_copy(update={'title':'Changed source title','raw_response_hash':'a'*64})
    assert not store.insert(changed,regions,[])
    assert store.get_reference(ref.id).title==ref.title
    event=store.db.execute("SELECT payload FROM events_v1 WHERE event='source_version_conflict'").fetchone()
    assert json.loads(event[0])['proposed_reference']['title']=='Changed source title'


def test_original_records_and_events_are_append_only(tmp_path):
    import sqlite3
    store=CorpusStore(tmp_path);ref,regions=reference(store);store.insert(ref,regions,[])
    for command in ['DELETE FROM references_v1','UPDATE references_v1 SET payload=payload','DELETE FROM events_v1','UPDATE events_v1 SET event=event','DELETE FROM regions_v1']:
        with pytest.raises(sqlite3.IntegrityError):
            with store.db:store.db.execute(command)
