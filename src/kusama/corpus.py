"""Bounded official collection ingestion; source facts are not artistic judgments.

The SQLite store is authoritative. JSONL files are atomic read/export snapshots.
All remote bytes are inert data; only adapter-controlled HTTP(S) requests execute.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from typing import Literal
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageCms, ImageDraw, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid')


class OperationRight(Contract):
    status: Literal['allowed', 'conditional', 'unknown', 'excluded']
    license: str
    evidence_url: str
    evidence_field: str
    attribution: str
    conditions: list[str] = Field(default_factory=list)


class Rights(Contract):
    metadata: OperationRight
    descriptive_text: OperationRight
    image_display: OperationRight
    derivatives: OperationRight
    training: OperationRight


class Asset(Contract):
    path: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    mime: str
    transforms: list[str] = Field(default_factory=list)
    source_url: str | None = None


class CorpusReference(Contract):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    provider: Literal['met', 'cleveland', 'artic', 'smithsonian']
    object_id: str
    title: str = Field(min_length=1)
    url: str = Field(pattern=r'^https?://')
    maker: str | None = None
    culture: str | None = None
    date: str | None = None
    medium: str | None = None
    dimensions: str | None = None
    source_family: str
    object_family: str
    non_art: bool = False
    classification_basis: str
    description: str = ''
    tags: list[str] = Field(default_factory=list)
    raw_response_path: str
    raw_response_hash: str = Field(pattern=r'^[0-9a-f]{64}$')
    raw_record_locator: str
    original: Asset
    view: Asset
    rights: Rights
    field_provenance: dict[str, str]
    retrieved_at: str
    split_group: str
    attribution: str
    discovery_query: str
    context_trust: Literal['untrusted_source_data'] = 'untrusted_source_data'


class Region(Contract):
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,160}$')
    reference_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,160}$')
    box: tuple[float, float, float, float]
    original_dimensions: tuple[int, int]
    coordinate_convention: Literal['normalized_xyxy_on_oriented_view'] = 'normalized_xyxy_on_oriented_view'
    crop: Asset
    role: str = 'geometric_detail'
    localization: Literal['geometric_fallback', 'supplied_box', 'validated_model_box'] = 'geometric_fallback'
    confidence: float | None = Field(default=None, ge=0, le=1)
    rights: Rights
    split_group: str
    full_context_path: str

    @model_validator(mode='after')
    def bounds(self):
        x0, y0, x1, y1 = self.box
        if not all(math.isfinite(x) for x in self.box) or not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError('Region must be finite normalized xyxy in [0,1]; no silent clamping')
        if (x1-x0)*(y1-y0) < .04:
            raise ValueError('Region minimum area is 4% of the full view')
        w,h = self.original_dimensions
        if min(w,h) < 1 or min((x1-x0)*w,(y1-y0)*h) < 32:
            raise ValueError('Region needs at least 32 pixels on each side')
        ratio=(x1-x0)*w/((y1-y0)*h)
        if not .1 <= ratio <= 10:
            raise ValueError('Region aspect ratio must be between 1:10 and 10:1')
        return self


class ContextClaim(Contract):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,160}$')
    reference_id: str | None = None
    claim: str
    supporting_url: str
    passage_locator: str
    source_date: str | None = None
    status: Literal['documented_source_statement', 'machine_interpretation']
    author: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    counterevidence: list[str] = Field(default_factory=list)
    rights: OperationRight
    source_response_hash: str | None = None
    trust: Literal['untrusted_source_data'] = 'untrusted_source_data'


class Device(Contract):
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,160}$')
    role: str
    observable_relationship: str
    reference_ids: list[str]
    region_ids: list[str]
    proposed_transfer: str
    contextual_dependencies: list[str]
    alternatives: list[str]
    uncertainty: str
    authorship: Literal['machine_authored_seed'] = 'machine_authored_seed'
    validation: Literal['unvalidated'] = 'unvalidated'
    evidence_claim_ids: list[str] = Field(default_factory=list)


class SourceConfig(Contract):
    schema_version: Literal[1] = 1
    root: str = 'artifacts/overnight/corpus'
    target: int = Field(default=200, ge=1, le=300)
    max_bytes: int = Field(default=2*1024**3, ge=1024, le=2*1024**3)
    max_asset_bytes: int = Field(default=12*1024**2, ge=1024, le=32*1024**2)
    max_request_attempts: int = Field(default=3, ge=1, le=3)
    long_edge: int = Field(default=1024, ge=256, le=2048)
    crops_per_reference: int = Field(default=2, ge=2, le=4)
    per_provider: dict[str, int] = Field(default_factory=lambda: {'cleveland':40,'artic':50,'met':60,'smithsonian':50})
    queries: list[str] = Field(default_factory=lambda: ['weave','vessel','mask','instrument','map','ornament','enclosure','shell','wing','repair','procession','landscape','architecture','crowd','empty space','repetition','root','textile','joint','tension','network','packaging'])
    smithsonian_units: list[str] = Field(default_factory=lambda: ['nasm','nmnhminsci','nmnhbotany','nmnhento','chndm'])
    query_limit: int = Field(default=6, ge=1, le=30)
    smithsonian_stream_bytes: int = Field(default=512*1024, ge=16384, le=2*1024**2)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def clean(value):
    if value is None:
        return None
    if isinstance(value,list):
        value='; '.join(str(v) for v in value if v)
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', str(value)))).strip() or None


def dump(value):
    return json.dumps(value.model_dump(mode='json') if isinstance(value,BaseModel) else value,ensure_ascii=False,sort_keys=True)


def atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


class CorpusStore:
    def __init__(self, root: Path | str):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'corpus.sqlite3', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS references_v1(id TEXT PRIMARY KEY, image_hash TEXT UNIQUE NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS regions_v1(id TEXT PRIMARY KEY, reference_id TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(reference_id,id));
        CREATE TABLE IF NOT EXISTS claims_v1(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS devices_v1(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS responses_v1(cache_key TEXT PRIMARY KEY, url TEXT NOT NULL,path TEXT NOT NULL,sha256 TEXT NOT NULL, retrieved_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events_v1(seq INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL, at TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cursors_v1(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TRIGGER IF NOT EXISTS corpus_events_no_update BEFORE UPDATE ON events_v1 BEGIN SELECT RAISE(ABORT, 'Corpus events are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS corpus_events_no_delete BEFORE DELETE ON events_v1 BEGIN SELECT RAISE(ABORT, 'Corpus events are append-only'); END;
        CREATE TRIGGER IF NOT EXISTS corpus_references_no_update BEFORE UPDATE ON references_v1 BEGIN SELECT RAISE(ABORT, 'Source references are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS corpus_references_no_delete BEFORE DELETE ON references_v1 BEGIN SELECT RAISE(ABORT, 'Source references are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS corpus_regions_no_update BEFORE UPDATE ON regions_v1 BEGIN SELECT RAISE(ABORT, 'Source regions are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS corpus_regions_no_delete BEFORE DELETE ON regions_v1 BEGIN SELECT RAISE(ABORT, 'Source regions are immutable'); END;
        ''')
        self.db.commit()

    def references(self):
        return [CorpusReference.model_validate_json(row[0]) for row in self.db.execute('SELECT payload FROM references_v1 ORDER BY id')]

    def regions(self, reference_id=None):
        rows=self.db.execute('SELECT payload FROM regions_v1 WHERE reference_id=? ORDER BY id',(reference_id,)) if reference_id else self.db.execute('SELECT payload FROM regions_v1 ORDER BY id')
        return [Region.model_validate_json(row[0]) for row in rows]

    def claims(self):
        return [ContextClaim.model_validate_json(row[0]) for row in self.db.execute('SELECT payload FROM claims_v1 ORDER BY id')]

    def devices(self):
        return [Device.model_validate_json(row[0]) for row in self.db.execute('SELECT payload FROM devices_v1 ORDER BY id')]

    def get_reference(self, identifier):
        row=self.db.execute('SELECT payload FROM references_v1 WHERE id=?',(identifier,)).fetchone()
        if not row: raise KeyError(identifier)
        return CorpusReference.model_validate_json(row[0])

    def get_region(self, identifier):
        row=self.db.execute('SELECT payload FROM regions_v1 WHERE id=?',(identifier,)).fetchone()
        if not row: raise KeyError(identifier)
        return Region.model_validate_json(row[0])

    def asset_path(self, relative):
        path=(self.root/relative).resolve()
        if not path.is_relative_to(self.root): raise ValueError('Asset path escapes corpus root')
        return path

    def event(self, event, **payload):
        with self.db:
            self.db.execute('INSERT INTO events_v1(event,at,payload) VALUES (?,?,?)',(event,utcnow(),dump(payload)))

    def cursor(self, key):
        row=self.db.execute('SELECT payload FROM cursors_v1 WHERE id=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_cursor(self,key,value):
        with self.db:self.db.execute('INSERT INTO cursors_v1 VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',(key,dump(value)))

    def insert(self, reference, regions, claims):
        existing=self.db.execute('SELECT id FROM references_v1 WHERE id=? OR image_hash=?',(reference.id,reference.original.sha256)).fetchone()
        if existing:
            if existing[0]==reference.id and self.get_reference(reference.id).raw_response_hash!=reference.raw_response_hash:
                self.event('source_version_conflict',id=reference.id,retained_reference=self.get_reference(reference.id).model_dump(mode='json'),proposed_reference=reference.model_dump(mode='json'),resolution='Existing immutable reference retained; new snapshot requires explicit versioned ID')
            else:
                self.event('duplicate_object' if existing[0]==reference.id else 'duplicate_image', id=reference.id, canonical_id=existing[0])
            return False
        for region in regions:
            if region.reference_id != reference.id or region.rights != reference.rights or region.split_group != reference.split_group:
                raise ValueError('Regions must inherit parent identity, rights and split grouping')
        with self.db:
            self.db.execute('INSERT INTO references_v1 VALUES (?,?,?)',(reference.id,reference.original.sha256,dump(reference)))
            for region in regions:self.db.execute('INSERT INTO regions_v1 VALUES (?,?,?)',(region.id,region.reference_id,dump(region)))
            for claim in claims:self.db.execute('INSERT OR IGNORE INTO claims_v1 VALUES (?,?)',(claim.id,dump(claim)))
        self.event('accepted',id=reference.id,provider=reference.provider)
        return True

    def export(self):
        for name,records in [('sources_manifest',self.references()),('regions',self.regions()),('context_claims',self.claims()),('devices',self.devices())]:
            atomic(self.root/f'{name}.jsonl',(''.join(dump(r)+'\n' for r in records)).encode())
        events=[{'seq':s,'event':e,'at':a,**json.loads(p)} for s,e,a,p in self.db.execute('SELECT * FROM events_v1 ORDER BY seq')]
        atomic(self.root/'events.jsonl',(''.join(dump(e)+'\n' for e in events)).encode())
        return self.report()

    def report(self):
        refs=self.references()
        events=Counter(dict(self.db.execute('SELECT event,COUNT(*) FROM events_v1 GROUP BY event')))
        report={'generated_at':utcnow(),'valid_references':len(refs),'distinct_image_hashes':len({r.original.sha256 for r in refs}),
          'providers':dict(Counter(r.provider for r in refs)),'object_families':dict(Counter(r.object_family for r in refs)),
          'non_art_references':sum(r.non_art for r in refs),'non_art_classification':'Machine rule over documented unit/object type; broad functional/non-fine-art category, not a curatorial value judgment.',
          'regions':len(self.regions()),'context_claims':len(self.claims()),'devices':len(self.devices()),
          'documented_places_or_cultures':dict(Counter(r.culture for r in refs if r.culture).most_common()),
          'documented_date_strings':dict(Counter(r.date for r in refs if r.date).most_common()),
          'context_statement_count_by_status':dict(Counter(c.status for c in self.claims())),
          'distinct_source_families':len({r.source_family for r in refs}),
          'devices_with_visual_examples':sum(bool(d.reference_ids) for d in self.devices()),
          'missing_fields':{k:sum(not getattr(r,k) for r in refs) for k in ['maker','culture','date','medium','dimensions','description']},
          'storage_bytes':sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()),'events':dict(events),
          'rights_rule_version':'official_cc0_images_v1_ccby_descriptions_separate',
          'artistic_validation':'unassessed','crop_localization':'geometric_fallback; no segmented-object claim',
          'source_data_trust':'untrusted; never executable instructions'}
        atomic(self.root/'corpus_report.json',(dump(report)+'\n').encode())
        return report


class BoundedFetcher:
    def __init__(self, store, config):
        self.store,self.config=store,config
        self.used=sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file())

    def check_budget(self,extra=0):
        # Include derivatives, SQLite and retained failed bytes, not only downloads.
        self.used=sum(p.stat().st_size for p in self.store.root.rglob('*') if p.is_file())
        if self.used+extra>self.config.max_bytes: raise RuntimeError('Corpus byte budget exhausted')

    def fetch(self,url,kind='metadata', byte_limit=None,range_bytes=None):
        parsed=urllib.parse.urlparse(url)
        if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Invalid remote source URL')
        cache_key=digest((url+f'|range={range_bytes}').encode())
        row=self.store.db.execute('SELECT path,sha256,retrieved_at FROM responses_v1 WHERE cache_key=?',(cache_key,)).fetchone()
        if row:
            raw=self.store.asset_path(row[0]).read_bytes()
            if digest(raw)!=row[1]: raise ValueError('Cached source hash mismatch')
            return raw,row[0],row[1],row[2]
        attempts=self.store.db.execute("SELECT COUNT(*) FROM events_v1 WHERE event='request_attempt' AND json_extract(payload,'$.cache_key')=?",(cache_key,)).fetchone()[0]
        limit=byte_limit or self.config.max_asset_bytes
        for attempt in range(attempts,self.config.max_request_attempts):
            self.check_budget(limit)
            self.store.event('request_attempt',cache_key=cache_key,url=url,attempt=attempt+1,kind=kind,range_bytes=range_bytes)
            raw=b''
            try:
                headers={'User-Agent':'KusamaResearch/0.1 (bounded public collection study)','Accept-Encoding':'identity'}
                if range_bytes: headers['Range']=f'bytes=0-{range_bytes-1}'
                request_url=urllib.parse.quote(url,safe=':/?=&%#')
                request=urllib.request.Request(request_url,headers=headers)
                with urllib.request.urlopen(request,timeout=25) as response:
                    raw=response.read(limit+1)
                    if len(raw)>limit: raise ValueError('Response exceeds configured byte limit')
                    if range_bytes and response.status!=206: raise ValueError('Provider did not honor bounded Range request')
                    response_headers={k:v for k,v in response.headers.items() if k.lower() in ('content-type','content-range','last-modified','etag','content-length')}
                h=digest(raw); path=f'raw/{kind}/{h}.bin';when=utcnow()
                atomic(self.store.root/path,raw)
                with self.store.db:self.store.db.execute('INSERT INTO responses_v1 VALUES (?,?,?,?,?)',(cache_key,url,path,h,when))
                self.store.event('request_completed',cache_key=cache_key,bytes=len(raw),sha256=h,response_headers=response_headers)
                time.sleep(.08)
                return raw,path,h,when
            except Exception as exc:
                failure_path=None
                if isinstance(exc,urllib.error.HTTPError):raw=exc.read(min(limit,65536))
                if raw:
                    failure_path=f'failures/{cache_key}_{attempt+1}.bin';atomic(self.store.root/failure_path,raw)
                self.store.event('request_failed',cache_key=cache_key,url=url,attempt=attempt+1,error=f'{type(exc).__name__}: {exc}',retained_bytes=failure_path)
                if isinstance(exc,urllib.error.HTTPError) and exc.code in (400,401,403,404):break
                if attempt+1 < self.config.max_request_attempts:time.sleep(min(2**attempt,4))
        raise RuntimeError(f'Bounded fetch failed: {url}')


POLICIES={
 'met':'https://www.metmuseum.org/policies/image-resources',
 'cleveland':'https://openaccess-api.clevelandart.org/',
 'artic':'https://api.artic.edu/docs/#artworks',
 'smithsonian':'https://www.si.edu/openaccess/faq',
}


def rights_for(provider, record, attribution):
    evidence=POLICIES[provider]
    image_field={'met':'isPublicDomain','cleveland':'share_license_status','artic':'is_public_domain','smithsonian':'content.descriptiveNonRepeating.online_media.media[].usage.access'}[provider]
    if provider=='met':eligible=record.get('isPublicDomain') is True
    elif provider=='cleveland':eligible=record.get('share_license_status')=='CC0'
    elif provider=='artic':eligible=record.get('is_public_domain') is True
    else:eligible=record.get('_media_access')=='CC0' and record.get('_metadata_access')=='CC0'
    base=OperationRight(status='allowed',license='CC0-1.0',evidence_url=evidence,evidence_field='official artwork-response data policy',attribution=attribution)
    image=base.model_copy(update={'status':'allowed' if eligible else 'excluded','evidence_field':image_field,'license':'CC0-1.0' if eligible else 'unverified'})
    text=base.model_copy(deep=True)
    if provider=='artic':
        text=OperationRight(status='conditional',license='CC-BY-4.0',evidence_url=evidence,evidence_field='info.license_text / description',attribution=attribution,conditions=['Retain Art Institute of Chicago attribution and source link; identify changes.'])
    return Rights(metadata=base,descriptive_text=text,image_display=image,derivatives=image.model_copy(deep=True),training=image.model_copy(update={'conditions':['This permission covers the admitted CC0 image bytes and CC0 metadata only. Descriptive text follows its separate license. No training launched by ingestion.']}))


def classify(title,medium,object_type,unit=''):
    # Originating communities and ritual meanings are never inferred from appearance.
    text=' '.join([title or '',medium or '',object_type or '']).lower()
    if unit.lower().startswith('nmnh'):
        return 'natural_form',True,'Documented Smithsonian natural-history unit '+unit
    if unit.lower()=='nasm':
        return 'scientific_mechanical_everyday',True,'Documented National Air and Space Museum object'
    if any(s in text for s in ['textile','weav','tapestry','cotton','silk','carpet','rug','embroid']):family='textile_pattern'
    elif any(s in text for s in ['architecture','architectural','chair','furniture','design','wallpaper']):family='design_architecture'
    elif any(s in text for s in ['painting','oil on','ink on','woodcut','lithograph','etching','print','drawing','watercolor']):family='painting_print'
    else:family='sculpture_functional_object'
    functional=any(s in (object_type or '').lower() for s in ['instrument','armor','armour','helmet','sword','tool','clock','chair','furniture','spoon','fork','knife','bottle','vessel','bowl','cup','ewer','lamp','basket'])
    if functional:family='scientific_mechanical_everyday' if any(s in text for s in ['instrument','clock','tool','armor','sword']) else family
    return family,functional,'Machine family rule over documented title/medium/object type; non-art flag requires documented functional object type'


def prepare_assets(store, raw, url, identifier, config):
    with Image.open(io.BytesIO(raw)) as im:
        if im.width*im.height>16_000_000:
            raise ValueError('Source exceeds 16-megapixel decoded-image ceiling')
        im.load()
        original_size=im.size
        mime=Image.MIME.get(im.format,'application/octet-stream')
        normalized=ImageOps.exif_transpose(im)
        transforms=['EXIF orientation applied']
        if im.info.get('icc_profile'):
            try:
                profile=ImageCms.ImageCmsProfile(io.BytesIO(im.info['icc_profile']))
                normalized=ImageCms.profileToProfile(normalized,profile,ImageCms.createProfile('sRGB'),outputMode='RGB')
                transforms.append('Embedded ICC profile converted to sRGB')
            except (ValueError, OSError, ImageCms.PyCMSError) as exc:
                transforms.append('ICC profile conversion failed; RGB conversion only: '+type(exc).__name__)
        else:transforms.append('No embedded ICC profile; assume sRGB, no colorimetric guarantee')
        normalized=normalized.convert('RGB')
        normalized.thumbnail((config.long_edge,config.long_edge),Image.Resampling.LANCZOS)
        transforms.append(f'RGB; long edge bounded at {config.long_edge}; JPEG quality 93 viewing derivative')
        h=digest(raw);original_path=f'images/original/{h}.bin'
        if not (store.root/original_path).exists():atomic(store.root/original_path,raw)
        view_path=f'images/view/{identifier}.jpg'
        buffer=io.BytesIO();normalized.save(buffer,'JPEG',quality=93,subsampling=0)
        atomic(store.root/view_path,buffer.getvalue())
        original=Asset(path=original_path,sha256=h,width=original_size[0],height=original_size[1],mime=mime,source_url=url,transforms=['Exact bytes of provider-returned bounded image derivative; not necessarily highest-resolution master'])
        view=Asset(path=view_path,sha256=digest(buffer.getvalue()),width=normalized.width,height=normalized.height,mime='image/jpeg',transforms=transforms,source_url=url)
        return original,view,normalized


def make_regions(store,reference,image,config):
    boxes=[(.05,.10,.65,.90),(.35,.10,.95,.90),(.15,.05,.85,.65),(.15,.35,.85,.95)]
    regions=[]; hashes=set()
    for n,box in enumerate(boxes[:config.crops_per_reference],1):
        # Geometric detail only: no claim that a crop isolates a semantic object.
        w,h=image.size
        pixels=tuple(round(v*(w if k%2==0 else h)) for k,v in enumerate(box))
        crop=image.crop(pixels);buf=io.BytesIO();crop.save(buf,'JPEG',quality=93,subsampling=0)
        hsh=digest(buf.getvalue())
        if hsh in hashes:continue
        hashes.add(hsh)
        identifier=f'{reference.id}_r{n}'
        path=f'images/crops/{identifier}.jpg'
        try:
            region=Region(id=identifier,reference_id=reference.id,box=box,original_dimensions=image.size,
             crop=Asset(path=path,sha256=hsh,width=crop.width,height=crop.height,mime='image/jpeg',transforms=[f'Geometric crop from oriented view; xyxy pixels {pixels}; JPEG quality 93']),
             rights=reference.rights,split_group=reference.split_group,full_context_path=reference.view.path)
        except ValueError as exc:
            store.event('crop_rejected',id=identifier,error=str(exc));continue
        atomic(store.root/path,buf.getvalue());regions.append(region)
    return regions


def normalize(provider,data,envelope,raw_path,raw_hash,when,query):
    d=data
    if provider=='cleveland':
        oid=str(d['id']); title=d['title'];url=d['url'];image_url=d.get('images',{}).get('web',{}).get('url')
        maker=clean([x.get('description','') for x in d.get('creators',[])])
        culture=clean(d.get('culture'));date=clean(d.get('creation_date'));medium=clean(d.get('technique'));dimensions=clean(d.get('measurements'))
        description=clean(d.get('description')) or '';kind=clean(d.get('type'));tags=[str(d.get('collection') or ''),str(d.get('department') or '')]
        credit=d.get('creditline') or '';family_group=d.get('cover_accession_number') or maker or f'{culture}:{title}'
        fields={'maker':'creators[].description','culture':'culture','date':'creation_date','medium':'technique','dimensions':'measurements','description':'description','url':'url','original':'images.web.url'}
    elif provider=='artic':
        oid=str(d['id']);title=d['title'];url=d.get('api_link') or f'https://api.artic.edu/api/v1/artworks/{oid}'
        base=envelope.get('config',{}).get('iiif_url');image_id=d.get('image_id')
        image_url=f'{base}/{image_id}/full/843,/0/default.jpg' if base and image_id else None
        maker=clean(d.get('artist_display'));culture=clean(d.get('place_of_origin'));date=clean(d.get('date_display'));medium=clean(d.get('medium_display'));dimensions=clean(d.get('dimensions'))
        description=clean(d.get('description')) or '';kind=clean(d.get('artwork_type_title'));tags=d.get('classification_titles') or [];credit=d.get('credit_line') or '';family_group=maker or f'{culture}:{title}'
        fields={'maker':'artist_display','culture':'place_of_origin (place, not inferred identity)','date':'date_display','medium':'medium_display','dimensions':'dimensions','description':'description; HTML removed (CC-BY attribution retained)','url':'api_link / official detail route','original':'config.iiif_url + image_id + documented IIIF syntax'}
    elif provider=='met':
        oid=str(d['objectID']);title=d['title'];url=d['objectURL'];image_url=d.get('primaryImageSmall') or d.get('primaryImage')
        maker=clean(d.get('artistDisplayName'));culture=clean(d.get('culture')) or clean(d.get('country'));date=clean(d.get('objectDate'));medium=clean(d.get('medium'));dimensions=clean(d.get('dimensions'));description='';kind=clean(d.get('objectName'));tags=[x['term'] for x in d.get('tags') or []];credit=d.get('creditLine') or '';family_group=maker or f'{culture}:{title}'
        fields={'maker':'artistDisplayName','culture':'culture; country fallback if culture absent','date':'objectDate','medium':'medium','dimensions':'dimensions','description':'not supplied by object API','url':'objectURL','original':'primaryImageSmall / primaryImage'}
    else:
        content=d['content'];desc=content['descriptiveNonRepeating'];freetext=content.get('freetext',{});indexed=content.get('indexedStructured',{})
        media=[m for m in desc.get('online_media',{}).get('media',[]) if m.get('type')=='Images' and m.get('usage',{}).get('access')=='CC0']
        selected=media[0] if media else {}
        d={**d,'_media_access':selected.get('usage',{}).get('access'),'_metadata_access':desc.get('metadata_usage',{}).get('access')}
        image_url=next((x['url'] for x in selected.get('resources',[]) if x.get('label')=='Screen Image'),selected.get('content'))
        oid=desc['record_ID'];title=desc.get('title',{}).get('content') or data.get('title');url=desc.get('record_link') or desc.get('guid')
        vals=lambda key:clean([x.get('content','') for x in freetext.get(key,[])])
        maker=clean([x.get('content','') for x in freetext.get('name',[]) if x.get('label','').lower() in ('artist','maker','manufacturer','designer')]);culture=vals('culture') or vals('place');date=vals('date')
        medium=clean([x.get('content','') for x in freetext.get('physicalDescription',[]) if x.get('label','').lower() in ('materials','medium','material')]);dimensions=clean([x.get('content','') for x in freetext.get('physicalDescription',[]) if x.get('label','').lower()=='dimensions'])
        description=clean([x.get('content','') for x in freetext.get('notes',[]) if x.get('label') not in ('Record Last Modified','Catalogue Status')]) or ''
        kind=vals('objectType');tags=[str(x) for x in indexed.get('topic',[])]+[str(x) for x in indexed.get('object_type',[])];credit=vals('creditLine') or '';family_group=maker or f'{desc.get("unit_code")}:{title}'
        fields={'maker':'content.freetext.name with explicit maker/artist/manufacturer/designer label','culture':'content.freetext.culture or documented place (not inferred identity)','date':'content.freetext.date','medium':'content.freetext.physicalDescription Materials/Medium','dimensions':'content.freetext.physicalDescription Dimensions','description':'content.freetext.notes, excluding record maintenance notes','url':'content.descriptiveNonRepeating.record_link or guid','original':'CC0 online_media.media[].resources Screen Image or content'}
    if not title or not url:raise ValueError('Missing canonical title or object URL')
    attribution=f'{title}. {maker or "Creator not supplied"}. {provider}; {credit}. {url}'
    rights=rights_for(provider,d,attribution)
    if not image_url or rights.image_display.status!='allowed':raise ValueError('No explicitly eligible CC0 image URL')
    identifier=f'{provider}_{oid}' if re.fullmatch(r'[A-Za-z0-9_-]{1,100}',oid) else f'{provider}_{digest(oid.encode())[:20]}'
    family,non_art,basis=classify(title,medium,kind,data.get('unitCode',''))
    source_family=f'{provider}:{clean(family_group)}'
    return dict(id=identifier,provider=provider,object_id=oid,title=clean(title),url=url,maker=maker,culture=culture,date=date,medium=medium,dimensions=dimensions,
      source_family=source_family,object_family=family,non_art=non_art,classification_basis=basis,description=description,tags=[t for t in tags if t],
      raw_response_path=raw_path,raw_response_hash=raw_hash,raw_record_locator=f'object_id={oid}',rights=rights,field_provenance={**fields,'title':'title / content.descriptiveNonRepeating.title','object_family':'machine classification over documented fields; not historical fact'},
      retrieved_at=when,split_group=source_family,attribution=attribution,discovery_query=query),image_url


def ingest_record(store,fetcher,provider,data,envelope,raw_info,query):
    try:
        fields,url=normalize(provider,data,envelope,*raw_info,query)
        if store.db.execute('SELECT 1 FROM references_v1 WHERE id=?',(fields['id'],)).fetchone():return False
        raw,_,_,_=fetcher.fetch(url,kind='image')
        existing=store.db.execute('SELECT id FROM references_v1 WHERE image_hash=?',(digest(raw),)).fetchone()
        if existing:
            store.event('duplicate_image',id=fields['id'],canonical_id=existing[0]);return False
        # Reserve a conservative bound for view plus up to four JPEG crops and original copy.
        fetcher.check_budget(len(raw)+12*1024**2)
        original,view,image=prepare_assets(store,raw,url,fields['id'],fetcher.config)
        reference=CorpusReference(**fields,original=original,view=view)
        regions=make_regions(store,reference,image,fetcher.config)
        claims=[]
        if reference.description:
            # Chunk at complete sentences where possible; all are source statements, not confirmed truth.
            passages=re.split(r'(?<=[.!?])\s+',reference.description)
            for n,passage in enumerate(passages):
                if not passage:continue
                claims.append(ContextClaim(id=f'{reference.id}_claim{n+1}',reference_id=reference.id,claim=passage,supporting_url=reference.url,
                 passage_locator=f'{reference.raw_record_locator} / {reference.field_provenance["description"]} / sentence {n+1}',status='documented_source_statement',author=provider,
                 rights=reference.rights.descriptive_text,source_response_hash=reference.raw_response_hash))
        accepted=store.insert(reference,regions,claims)
        if accepted:
            count=store.db.execute('SELECT COUNT(*) FROM references_v1').fetchone()[0]
            if count%10==0:
                store.export();print(dump({'progress':count,'provider':provider,'last_id':reference.id}),flush=True)
        return accepted
    except Exception as exc:
        store.event('record_rejected',provider=provider,object_id=str(data.get('id') or data.get('objectID') or ''),query=query,error=f'{type(exc).__name__}: {exc}')
        return False


def discover_collection(store,fetcher,provider,target):
    config=fetcher.config
    def count():return store.db.execute("SELECT COUNT(*) FROM references_v1 WHERE json_extract(payload,'$.provider')=?",(provider,)).fetchone()[0]
    if count()>=target:return
    for page in range(2):
      for query in config.queries:
        if count()>=target or len(store.references())>=config.target:return
        key=f'{provider}:{query}:{page}'; cursor=store.cursor(key) or {'done':False}
        if cursor['done']:continue
        try:
            params={'q':query,'limit':config.query_limit}
            if provider=='cleveland':
                params.update({'has_image':1,'skip':page*config.query_limit});url='https://openaccess-api.clevelandart.org/api/artworks/?cc0&'+urllib.parse.urlencode(params)
            elif provider=='artic':
                params.update({'page':page+1,'fields':'id,title,api_link,image_id,is_public_domain,artist_display,place_of_origin,date_display,medium_display,dimensions,description,artwork_type_title,classification_titles,credit_line','query[term][is_public_domain]':'true'})
                url='https://api.artic.edu/api/v1/artworks/search?'+urllib.parse.urlencode(params)
            else:
                params.update({'hasImages':'true','offset':page*config.query_limit});url='https://collectionapi.metmuseum.org/public/collection/v1.1/search?'+urllib.parse.urlencode(params)
            raw,path,hsh,when=fetcher.fetch(url,byte_limit=2*1024**2);envelope=json.loads(raw)
            rows=envelope.get('objectIDs') or [] if provider=='met' else envelope.get('data',[])
            for data in rows:
                if count()>=target or len(store.references())>=config.target:break
                if provider=='met':
                    try:
                        objectraw,objectpath,objecthash,objectwhen=fetcher.fetch(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{data}',byte_limit=2*1024**2)
                        ingest_record(store,fetcher,provider,json.loads(objectraw),{},(objectpath,objecthash,objectwhen),query)
                    except Exception as exc:store.event('discovery_object_failed',provider=provider,id=data,error=str(exc))
                else:ingest_record(store,fetcher,provider,data,envelope,(path,hsh,when),query)
            store.set_cursor(key,{'done':True,'url':url,'page':page})
        except Exception as exc:store.event('discovery_failed',provider=provider,query=query,error=str(exc))


def discover_smithsonian(store,fetcher,target):
    config=fetcher.config
    def count():return store.db.execute("SELECT COUNT(*) FROM references_v1 WHERE json_extract(payload,'$.provider')='smithsonian'").fetchone()[0]
    if count()>=target:return
    index_url='https://smithsonian-open-access.s3-us-west-2.amazonaws.com/metadata/edan/index.txt'
    try:unit_links=fetcher.fetch(index_url,byte_limit=64*1024)[0].decode().splitlines()
    except Exception as exc:
        store.event('discovery_failed',provider='smithsonian',error=str(exc));return
    start_count=count()
    per_unit=math.ceil(target/len(config.smithsonian_units))
    for unit in config.smithsonian_units:
        unit_link=next((u for u in unit_links if f'/{unit}/index.txt' in u),None)
        if not unit_link:continue
        try:streams=fetcher.fetch(unit_link,byte_limit=128*1024)[0].decode().splitlines()
        except Exception as exc:store.event('discovery_failed',provider='smithsonian',unit=unit,error=str(exc));continue
        unit_existing=sum(r.provider=='smithsonian' and r.discovery_query==unit for r in store.references())
        accepted=unit_existing
        for url in streams[:4]:
            if accepted>=per_unit or count()>=target or len(store.references())>=config.target:break
            key='smithsonian:'+url;cursor=store.cursor(key) or {'line':0,'done':False}
            if cursor['done']:continue
            try:
                raw,path,hsh,when=fetcher.fetch(url,byte_limit=config.smithsonian_stream_bytes,range_bytes=config.smithsonian_stream_bytes)
                lines=raw.splitlines(keepends=True)
                for n,line in enumerate(lines):
                    if n<cursor['line']:continue
                    if not line.endswith(b'\n'):break # Retain partial last bytes without pretending a full record.
                    if accepted>=per_unit or count()>=target or len(store.references())>=config.target:break
                    data=json.loads(line)
                    if ingest_record(store,fetcher,'smithsonian',data,{},(path,hsh,when),unit):accepted+=1
                    store.set_cursor(key,{'line':n+1,'done':False})
                else:store.set_cursor(key,{'line':len(lines),'done':True})
            except Exception as exc:store.event('discovery_failed',provider='smithsonian',unit=unit,url=url,error=str(exc))
    store.event('smithsonian_pass_completed',before=start_count,after=count())


DEVICE_SEEDS=[
 ('composition','supports','Transfer an arrangement that visibly supports another part; test stability against dependence.'),
 ('scale','contrasts','Vary one part relative to its surrounding field; compare monumentality, awkwardness and intimacy.'),
 ('material_transfer','substitutes','Give one depicted structure the documented material of this source; inspect what material logic survives.'),
 ('negative_space','encloses','Open a gap around a dense part and compare breathing room with separation.'),
 ('rhythm','repeats','Repeat a bounded unit with one interval changed; test continuity against interruption.'),
 ('symbolic_inversion','inverts','Reverse an explicitly proposed role; do not assume the original object has a universal symbolic meaning.'),
 ('framing','encloses','Borrow a boundary arrangement and test whether it holds together or restricts the surrounding work.'),
 ('concealment','conceals','Partially hide a joining point while preserving evidence of connection.'),
 ('fragmentation','fragments','Separate a continuous material into several pieces while retaining some shared alignment.'),
 ('repetition','repeats','Vary repeated marks without making their meaning equivalent to the source culture.'),
 ('absence','interrupts','Delete an expected continuation and compare release, incompleteness and deliberate restraint.'),
]


def seed_devices(store,target=40):
    refs=store.references()
    if not refs:return
    # Interleave institutions/families so seeds do not all attach to one famous artist.
    refs=sorted(refs,key=lambda r:(r.object_family,r.provider,r.id))
    chosen=refs[::max(1,len(refs)//target)][:target]
    for n,reference in enumerate(chosen):
        role,predicate,transfer=DEVICE_SEEDS[n%len(DEVICE_SEEDS)]
        regions=store.regions(reference.id)
        device=Device(id=f'device_seed_{n+1:02d}',role=role,
          observable_relationship=f'Inspection prompt for {reference.title}: look for a relationship that {predicate}. This seed does not assert that the relationship is present.',
          reference_ids=[reference.id],region_ids=[regions[n%len(regions)].id] if regions else [],proposed_transfer=transfer,
          contextual_dependencies=[f'Documented medium: {reference.medium or "not supplied"}.',f'Documented context: {reference.culture or "not supplied"}.','Interpret in the full source image and historical context; crop may remove relevant evidence.'],
          alternatives=['The proposed relationship may not be present in this example.','The transfer could read as playful, restrained, tense or neutral depending on the receiving work.'],
          uncertainty='Machine-authored unvalidated prompt grounded by a resolvable visual example, not an expert annotation or observed artistic effect.',
          evidence_claim_ids=[c.id for c in store.claims() if c.reference_id==reference.id][:2])
        with store.db:store.db.execute('INSERT OR IGNORE INTO devices_v1 VALUES (?,?)',(device.id,dump(device)))


def record_device_grounding(store, observations, inspection_path):
    """Append assistant visual observations while retaining every earlier seed payload.

    This is deliberately explicit: no unseen image receives an observation by template.
    The observations must come from an actual inspection; they are not human labels.
    """
    for device_id,observation in observations.items():
        current=next(d for d in store.devices() if d.id==device_id)
        ref=store.get_reference(current.reference_ids[0])
        claim_id=f'{device_id}_visual_{digest((ref.view.sha256+observation).encode())[:12]}'
        if claim_id in current.evidence_claim_ids:continue
        claim=ContextClaim(id=claim_id,reference_id=ref.id,claim=observation,supporting_url=ref.url,
          passage_locator=f'Assistant inspection of {inspection_path}; exact view hash {ref.view.sha256}; whole image, not semantic region localization',
          status='machine_interpretation',author='Codex visual inspection; not human review',rights=ref.rights.metadata,source_response_hash=ref.raw_response_hash)
        updated=current.model_copy(update={'observable_relationship':observation,'evidence_claim_ids':current.evidence_claim_ids+[claim_id],
          'uncertainty':'Assistant visually observed a cue in the saved source view. The transferable role and effects remain unvalidated hypotheses; geometric crops need separate semantic review.'})
        with store.db:
            store.db.execute('INSERT OR IGNORE INTO claims_v1 VALUES (?,?)',(claim.id,dump(claim)))
            store.db.execute('INSERT INTO events_v1(event,at,payload) VALUES (?,?,?)',('device_grounding_revision',utcnow(),dump({'before':current.model_dump(mode='json'),'after':updated.model_dump(mode='json'),'inspection_path':inspection_path})))
            store.db.execute('UPDATE devices_v1 SET payload=? WHERE id=?',(dump(updated),device_id))


READING_NOTES=[
 ('reading_mutu_art21','https://art21.org/artist/wangechi-mutu/','Artist profile, paragraphs 1–3',None,
  'Art21 describes Mutu combining magazine and illustration fragments with ink and paint, and working with Mylar, soil and found materials. These documented material transformations are one situated process reference, not a universal taste model.'),
 ('reading_mutu_process','https://art21.org/2021/07/21/weekly-watchlist-wangechi-mutu-returns-to-nature/','Weekly Watchlist, studio-process discussion','2021-07-21',
  'Art21’s studio account traces everyday wood, khanga patterns and soil into Mutu’s paintings and sculptures. This supports investigating material provenance; it does not establish permission to reuse her images or invent her private reasoning.'),
 ('reading_hoch_moma','https://www.moma.org/collection/works/33945','Object metadata and artist profile',None,
  'MoMA identifies Study for Man and Machine (1921) as watercolor, gouache and pencil on paper. Its linked profile discusses Höch’s recombination of mass-media images and text. The study itself must not be mislabeled as a photomontage.'),
 ('reading_surrealism_met','https://www.metmuseum.org/ja/exhibitions/surrealism-beyond-borders','Exhibition overview, international scope','2021-10-11',
  'The Met exhibition describes Surrealist exchange across 45 countries and nearly eight decades, including Asia, North Africa, the Caribbean and Latin America. Use this as a reason to widen inquiry; it does not document relationships between arbitrary corpus objects.')
]


def seed_readings(store):
    for identifier,url,locator,source_date,note in READING_NOTES:
        right=OperationRight(status='unknown',license='No broad reuse license verified for the linked page or its images',evidence_url=url,
          evidence_field='Reading link and short attributed assistant paraphrase only; full publication/image excluded',attribution=url,
          conditions=['No training use of linked publication or images inferred. Recheck rights before expanding excerpts.'])
        claim=ContextClaim(id=identifier,claim=note,supporting_url=url,passage_locator=locator,source_date=source_date,status='machine_interpretation',author='Codex short attributed research note, checked against official page 2026-09-08',rights=right)
        with store.db:store.db.execute('INSERT OR IGNORE INTO claims_v1 VALUES (?,?)',(claim.id,dump(claim)))


def contact_sheet(store):
    refs=store.references()
    if not refs:return None
    columns=8;cellw,cellh=190,208
    sheet=Image.new('RGB',(columns*cellw,math.ceil(len(refs)/columns)*cellh),'#f1f0ec');draw=ImageDraw.Draw(sheet)
    for n,r in enumerate(refs):
        x=(n%columns)*cellw;y=(n//columns)*cellh
        with Image.open(store.asset_path(r.view.path)) as image:
            image.thumbnail((180,165));sheet.paste(image,(x+(cellw-image.width)//2,y+(165-image.height)//2))
        draw.text((x+5,y+168),r.id[:27],fill='black')
        label=r.title.encode('ascii','replace').decode()[:29]
        draw.text((x+5,y+183),label,fill='black')
    path=store.root/'source_contact_sheet.jpg';sheet.save(path,quality=90)
    return path


def verify_corpus(store):
    failures=[];refs=store.references();regions=store.regions()
    for ref in refs:
        for a in [ref.original,ref.view]:
            path=store.asset_path(a.path)
            if not path.exists() or digest(path.read_bytes())!=a.sha256:failures.append(ref.id+':asset_hash')
            else:
                with Image.open(path) as im:
                    im.verify()
                    if im.size!=(a.width,a.height):failures.append(ref.id+':dimensions')
        raw=store.asset_path(ref.raw_response_path)
        if not raw.exists() or digest(raw.read_bytes())!=ref.raw_response_hash:failures.append(ref.id+':raw_hash')
    for region in regions:
        ref=store.get_reference(region.reference_id)
        if region.rights!=ref.rights or region.split_group!=ref.split_group:failures.append(region.id+':inheritance')
        path=store.asset_path(region.crop.path)
        if not path.exists() or digest(path.read_bytes())!=region.crop.sha256:failures.append(region.id+':crop_hash')
    result={'verified_at':utcnow(),'mode':'local bytes from live public APIs; no inference or human annotation','references_checked':len(refs),'regions_checked':len(regions),'failures':failures,'passed':not failures}
    atomic(store.root/'verification.json',(dump(result)+'\n').encode());return result


def ingest(config):
    store=CorpusStore(config.root);fetcher=BoundedFetcher(store,config)
    store.event('ingest_started',config=config.model_dump())
    config_bytes=(dump(config)+'\n').encode()
    atomic(store.root/f'configs/{digest(config_bytes)}.json',config_bytes)
    atomic(store.root/'effective_config.json',config_bytes)
    for provider,target in config.per_provider.items():
        if provider=='smithsonian':discover_smithsonian(store,fetcher,target)
        elif provider in ('met','cleveland','artic'):discover_collection(store,fetcher,provider,target)
        else:store.event('unsupported_provider',provider=provider)
    seed_devices(store);seed_readings(store);contact_sheet(store)
    report=store.export();verify_corpus(store)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['ingest','report','verify','schema'],nargs='?',default='ingest')
    parser.add_argument('--config',type=Path,default=Path('configs/starter_sources.yaml'))
    parser.add_argument('--root',type=Path)
    parser.add_argument('--target',type=int)
    args=parser.parse_args(argv)
    if args.command=='schema':
        print(dump({c.__name__:c.model_json_schema() for c in [CorpusReference,Region,ContextClaim,Device,SourceConfig]}));return
    # JSON is valid YAML 1.2: no undeclared YAML runtime dependency.
    config=SourceConfig.model_validate_json(args.config.read_bytes()) if args.config.exists() else SourceConfig()
    updates={}
    if args.root:updates['root']=str(args.root)
    if args.target:updates['target']=args.target
    config=SourceConfig.model_validate({**config.model_dump(),**updates})
    if args.command=='ingest':result=ingest(config)
    else:
        store=CorpusStore(config.root)
        result=verify_corpus(store) if args.command=='verify' else store.report()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
