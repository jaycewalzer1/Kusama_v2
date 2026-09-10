"""Small immutable whole/detail CLIP, lexical and proposed-relation retrieval.

Cosine similarity and rank fusion are retrieval diagnostics, not quality labels.
Indexes contain source records only; experiment memory has its own cutoff API.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Literal
import random
import re
import sqlite3
import time

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import sha256, write_json
from .resources import canonical, load_config, SessionBudget

CLIP_MODEL = 'openai/clip-vit-base-patch32'
INDEX_VERSION = 'clip512-rrf60-v3'


def normalized(values):
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim not in (1, 2) or matrix.shape[-1] != 512 or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 512-dimensional CLIP descriptors')
    lengths = np.linalg.norm(matrix, axis=-1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise ValueError('Zero vectors are not valid descriptors')
    return matrix / lengths


class ClipEncoder:
    def __init__(self, config=None, *, device='cpu'):
        self.config = config or load_config()
        self.device = device
        self.model = self.processor = None
        self.metadata = {'model': CLIP_MODEL, 'revision': self.config.clip_revision,
                         'dimensions': 512, 'normalized': True, 'device': device,
                         'role': 'retrieval descriptor; not artistic quality', 'frozen': True}

    def prepare(self):
        from transformers import CLIPModel, CLIPProcessor
        if self.model is not None:
            return self.metadata
        args = {'revision': self.config.clip_revision, 'cache_dir': self.config.cache_dir,
                'local_files_only': True, 'trust_remote_code': False}
        self.processor = CLIPProcessor.from_pretrained(CLIP_MODEL, **args)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL, **args).eval().to(self.device)
        self.model.requires_grad_(False)
        if self.model.config.projection_dim != 512:
            raise ValueError('Checkpoint projection is incompatible')
        self.metadata['packages'] = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'pillow')}
        self.metadata['processor'] = self.processor.to_dict() if hasattr(self.processor, 'to_dict') else {'class': type(self.processor).__name__}
        self.metadata['trainable_parameters'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return self.metadata

    @staticmethod
    def _features(output):
        # Transformers 5 returns a ModelOutput; older versions returned tensors.
        return output.pooler_output if hasattr(output, 'pooler_output') else output

    def images(self, paths, batch_size=8):
        import torch
        self.prepare()
        vectors = []
        for start in range(0, len(paths), batch_size):
            batch = []
            for path in paths[start:start + batch_size]:
                with Image.open(path) as source:
                    batch.append(ImageOps.exif_transpose(source).convert('RGB'))
            inputs = self.processor(images=batch, return_tensors='pt').to(self.device)
            with torch.inference_mode():
                output = self._features(self.model.get_image_features(**inputs))
            vectors.append(output.float().cpu().numpy())
        return normalized(np.concatenate(vectors)) if vectors else np.empty((0, 512), dtype=np.float32)

    def text(self, query):
        import torch
        self.prepare()
        tokens = self.processor.tokenizer(query, add_special_tokens=True, truncation=False)['input_ids']
        limit = self.model.config.text_config.max_position_embeddings
        inputs = self.processor.tokenizer([query], return_tensors='pt', padding=True, truncation=True, max_length=limit).to(self.device)
        with torch.inference_mode():
            output = self._features(self.model.get_text_features(**inputs))
        return normalized(output.float().cpu().numpy()[0]), {
            'raw_tokens': len(tokens), 'admitted_tokens': int(inputs['input_ids'].shape[-1]),
            'truncated': len(tokens) > limit, 'limit': limit,
            'admitted_text': self.processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True),
            'policy': 'CLIP uses this explicit truncated query; lexical search retains the complete query'}

    def close(self):
        self.model = self.processor = None
        gc.collect()
        if self.device == 'mps':
            import torch
            torch.mps.empty_cache()


def words(text):
    return re.findall(r"[\w'-]+", text.casefold(), flags=re.UNICODE)


def _allowed(right):
    if isinstance(right, dict):
        return right.get('status') == 'allowed'
    return False


# Explicit, inspectable query cues. These are a retrieval vocabulary, not universal
# meanings. Unknown predicates can be added through validated registered edges.
PREDICATE_CUES = {
    'supports': ['support', 'hold', 'load-bearing', 'rest on', 'resting on'],
    'encloses': ['enclos', 'surround', 'frame', 'border', 'confine', 'restrict', 'boundary', 'inside'],
    'interrupts': ['interrupt', 'gap', 'stop at', 'puncture', 'break'],
    'conceals': ['conceal', 'hide', 'hidden', 'obscur', 'cover'],
    'repeats': ['repeat', 'rhythm', 'multiple', 'several', 'paired', 'alternat', 'stack'],
    'substitutes': ['substitut', 'replace', 'instead'],
    'connects': ['connect', 'join', 'interlock', 'continu'],
    'fragments': ['fragment', 'separat', 'piece'],
    'contrasts': ['contrast', 'against', 'lighter', 'darker', 'denser', 'unequal'],
}


class RelationshipEdge(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,160}$')
    reference_id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,160}$')
    region_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,160}$')
    predicate: str = Field(pattern=r'^[a-z][a-z0-9_]{0,63}$')
    statement: str = Field(min_length=1,max_length=2000)
    evidence_ids: list[str] = Field(min_length=1,max_length=12)
    origin: Literal['machine_interpretation','operator_hypothesis','documented_source_statement']
    validation: Literal['unvalidated'] = 'unvalidated'
    scope_note: str = 'Whole source configuration; no segmented object or causal emotional effect established'


def predicate_matches(text, registered_predicates=(), *, cue_map=None):
    """Conservative lexical-to-predicate normalization, recorded in query evidence."""
    lowered=text.casefold()
    found={predicate for predicate,cues in (cue_map if cue_map is not None else PREDICATE_CUES).items()
           if any(re.search(r'\b'+re.escape(cue)+r'[a-z]*\b',lowered) for cue in cues)}
    found.update(predicate for predicate in registered_predicates
                 if re.search(r'\b'+re.escape(predicate.replace('_',' '))+r'\b',lowered))
    return found


def source_edges(devices, claims, registered=()):
    """Derive attributed edges only from saved source visual observations.

    Generic seed questions, proposed transfers, and predicted effects cannot
    become evidence that a relationship was observed in the source.
    """
    claims_by_id={c['id']:c for c in claims}
    devices_by_id={d['id']:d for d in devices}
    edges=[]
    for device in devices:
        evidence=[claims_by_id[c] for c in device.get('evidence_claim_ids',[]) if c in claims_by_id
                  and 'visual' in c and claims_by_id[c].get('status')=='machine_interpretation'
                  and claims_by_id[c].get('reference_id') in device.get('reference_ids',[])]
        for claim in evidence:
            for predicate in sorted(predicate_matches(claim['claim'])):
                identifier='edge_'+hashlib.sha256(canonical([device['id'],claim['id'],predicate]).encode()).hexdigest()[:24]
                edges.append(RelationshipEdge(id=identifier,reference_id=claim['reference_id'],predicate=predicate,
                  statement=claim['claim'],evidence_ids=[claim['id'],device['id']],origin='machine_interpretation').model_dump(mode='json'))
    for incoming in registered:
        edge=RelationshipEdge.model_validate(incoming)
        for identifier in edge.evidence_ids:
            claim=claims_by_id.get(identifier);device=devices_by_id.get(identifier)
            if not ((claim and claim.get('reference_id')==edge.reference_id) or
                    (device and edge.reference_id in device.get('reference_ids',[]))):
                raise ValueError('Registered relationship evidence must resolve to its source')
        if edge.origin=='documented_source_statement' and not any(
            claims_by_id.get(identifier,{}).get('status')=='documented_source_statement'
            and claims_by_id[identifier].get('claim')==edge.statement for identifier in edge.evidence_ids):
            raise ValueError('A documented-source edge requires the exact backing source claim; paraphrases remain interpretations')
        edges.append(edge.model_dump(mode='json'))
    if len({e['id'] for e in edges})!=len(edges):raise ValueError('Relationship edge IDs must be unique')
    return edges


def read_source_audit(corpus_root, reference_ids):
    path=corpus_root/'source_visual_audit.json'
    if not path.exists():return None,None
    raw=path.read_bytes()
    if len(raw)>1024**2:raise ValueError('Source visual audit is unreasonably large')
    audit=json.loads(raw)
    exclusions=audit.get('shortlist_exclusions',[])
    for row in exclusions:
        if not isinstance(row,dict) or row.get('reference_id') not in reference_ids or not isinstance(row.get('reason'),str):
            raise ValueError('Source audit exclusions must resolve to actual references and reasons')
    return {'sha256':hashlib.sha256(raw).hexdigest(),'file':'source_visual_audit.json',
            'shortlist_exclusions':exclusions,'authorship':audit.get('author')},raw


def build_index(corpus_root, output_root, *, config=None, device='cpu', encoder=None, relationship_edges=()):
    from .corpus import CorpusStore
    corpus_root, output_root = Path(corpus_root).resolve(), Path(output_root).resolve()
    config = config or load_config()
    store = CorpusStore(corpus_root)
    store.db.execute('BEGIN')
    try:
        references = [r.model_dump(mode='json') for r in store.references()]
        regions = [r.model_dump(mode='json') for r in store.regions()]
        devices = [r.model_dump(mode='json') for r in store.devices()]
        claims = [r.model_dump(mode='json') for r in store.claims()]
    finally:
        store.db.rollback()
        store.db.close()
    source_audit,audit_bytes=read_source_audit(corpus_root,{r['id'] for r in references})
    excluded={row['reference_id'] for row in (source_audit or {}).get('shortlist_exclusions',[])}
    accepted = {r['id']: r for r in references if _allowed(r['rights']['image_display']) and r['id'] not in excluded}
    edges=source_edges(devices,claims,relationship_edges)
    regions_by_id={r['id']:r for r in regions}
    for edge in edges:
        if edge['reference_id'] not in {r['id'] for r in references}:raise ValueError('Relationship edge source does not resolve')
        if edge['region_id'] and (edge['region_id'] not in regions_by_id or regions_by_id[edge['region_id']]['reference_id']!=edge['reference_id']):raise ValueError('Relationship region must belong to its source')
    documents = []
    for ref in sorted(accepted.values(), key=lambda r: r['id']):
        documented = ' '.join(str(ref.get(f) or '') for f in ('title', 'maker', 'culture', 'date', 'medium', 'description', 'object_family'))
        documented += ' ' + ' '.join(ref.get('tags', []))
        related = [d for d in devices if ref['id'] in d.get('reference_ids', []) or d.get('reference_id') == ref['id']]
        documents.append({'id': ref['id'], 'reference_id': ref['id'], 'region_id': None, 'scope': 'whole',
            'source_family': ref['source_family'], 'split_group': ref['split_group'],
            'object_family': ref['object_family'], 'non_art': ref['non_art'], 'path': ref['view']['path'],
            'sha256': ref['view']['sha256'], 'documented_text': documented,
            'interpretation_text': ' '.join(canonical(d) for d in related),
            'relationship_evidence': related, 'relationship_edges':[e for e in edges if e['reference_id']==ref['id'] and e['region_id'] is None], 'reference': ref})
    full_docs = {d['id']: d for d in documents}
    for region in sorted(regions, key=lambda r: r['id']):
        parent = full_docs.get(region['reference_id'])
        if parent and _allowed(region['rights']['derivatives']):
            documents.append({**parent, 'id': region['id'], 'region_id': region['id'], 'scope': 'part',
                              'path': region['crop']['path'], 'sha256': region['crop']['sha256'], 'region': region,
                              'relationship_edges':[e for e in edges if e['reference_id']==parent['reference_id'] and e['region_id'] in (None,region['id'])],
                              'region_relationship_caution':'Whole-source edges retain context; they are not proof that this geometric crop contains the relationship'})
    if not documents:
        raise ValueError('No eligible real source images; ingest before indexing')
    identity = {'version': INDEX_VERSION, 'clip_revision': config.clip_revision, 'documents': documents, 'claims': claims, 'source_visual_audit':source_audit, 'predicate_query_cues':PREDICATE_CUES}
    index_id = hashlib.sha256(canonical(identity).encode()).hexdigest()[:20]
    directory = output_root / index_id
    if (directory / 'manifest.json').exists():
        RetrievalIndex(directory).verify()
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    paths = [(corpus_root / doc['path']).resolve() for doc in documents]
    for doc, path in zip(documents, paths, strict=True):
        if not path.is_relative_to(corpus_root) or sha256(path) != doc['sha256']:
            raise ValueError(f"Source image integrity failure: {doc['id']}")
    encoder = encoder or ClipEncoder(config, device=device)
    started = time.monotonic()
    try:
        encoder.prepare()
        vectors = encoder.images(paths)
        if vectors.shape != (len(documents), 512):
            raise ValueError('Encoder returned incorrect image rows')
        with (directory / 'embeddings.npy.tmp').open('wb') as stream:
            np.save(stream, normalized(vectors), allow_pickle=False)
        (directory / 'embeddings.npy.tmp').replace(directory / 'embeddings.npy')
        write_json(directory / 'documents.json', {'documents': documents, 'claims': claims})
        if audit_bytes is not None:(directory/'source_visual_audit.json').write_bytes(audit_bytes)
        with closing(sqlite3.connect(directory / 'lexical.sqlite3')) as db, db:
            db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS documents USING fts5(id UNINDEXED, documented, interpretations)')
            db.execute('DELETE FROM documents')
            db.executemany('INSERT INTO documents VALUES(?,?,?)', [(d['id'], d['documented_text'], d['interpretation_text']) for d in documents])
        manifest = {'schema_version': 1, 'index_id': index_id, 'version': INDEX_VERSION,
            'corpus_root': str(corpus_root), 'created_at': datetime.now(timezone.utc).isoformat(),
            'whole_images': len(full_docs), 'detail_images': len(documents) - len(full_docs),
            'encoder': encoder.metadata, 'source_identity_sha256': hashlib.sha256(canonical(identity).encode()).hexdigest(),
            'files': {name: sha256(directory / name) for name in ('embeddings.npy', 'documents.json', 'lexical.sqlite3')},
            'runtime_seconds': time.monotonic() - started, 'input_pool': 'permitted reference corpus; no experimental or sealed outcomes',
            'rights_rule': 'Display-permitted whole views; details additionally require derivatives allowed',
            'source_visual_audit':source_audit,'source_audit_excluded_count':len(excluded),
            'relationship_policy':'Typed predicates from saved assistant visual observations or explicit registered source-grounded edges; always unvalidated',
            'relationship_edge_count':sum(len(d['relationship_edges']) for d in documents if d['scope']=='whole'),
            'predicate_query_cues':PREDICATE_CUES,
            'human_usefulness': None}
        if audit_bytes is not None:manifest['files']['source_visual_audit.json']=sha256(directory/'source_visual_audit.json')
        write_json(directory / 'manifest.json', manifest)
    finally:
        encoder.close()
    return directory


class RetrievalIndex:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.manifest = json.loads((self.directory / 'manifest.json').read_text())
        self.verify()
        self.documents = json.loads((self.directory / 'documents.json').read_text())['documents']
        self.by_id = {d['id']: d for d in self.documents}
        self.vectors = np.load(self.directory / 'embeddings.npy', allow_pickle=False)
        if len(self.by_id) != len(self.documents) or self.vectors.shape != (len(self.documents), 512):
            raise ValueError('Index ID or shape mismatch')
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError('Index has invalid descriptors')

    def verify(self):
        for name, digest in self.manifest['files'].items():
            path = (self.directory / name).resolve()
            if not path.is_relative_to(self.directory) or sha256(path) != digest:
                raise ValueError(f'Index hash mismatch: {name}')
        data = json.loads((self.directory / 'documents.json').read_text())
        identity = {'version': self.manifest['version'], 'clip_revision': self.manifest['encoder']['revision'],
                    'documents': data['documents'], 'claims': data['claims']}
        if self.manifest['version'] != 'clip512-rrf60-v1':identity['source_visual_audit']=self.manifest.get('source_visual_audit')
        if self.manifest['version'] == 'clip512-rrf60-v3':identity['predicate_query_cues']=self.manifest['predicate_query_cues']
        digest = hashlib.sha256(canonical(identity).encode()).hexdigest()
        if digest != self.manifest['source_identity_sha256'] or digest[:20] != self.manifest['index_id'] or self.directory.name != digest[:20]:
            raise ValueError('Content-addressed index identity changed')

    def asset(self, identifier):
        doc = self.by_id[identifier]
        root = Path(self.manifest['corpus_root']).resolve()
        path = (root / doc['path']).resolve()
        if not path.is_relative_to(root) or sha256(path) != doc['sha256']:
            raise ValueError('Source asset no longer matches frozen index')
        return path

    def query(self, text, *, scope='both', local_need=None, whole_context=None,
              query_vector=None, clip_admission=None, limit=48, seed=23,
              allocation=None, exclude_source_families=(), allowed_reference_ids=None, preferred_object_families=None):
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise ValueError('Supply a bounded nonempty query')
        if scope not in ('whole', 'part', 'both') or not 1 <= limit <= 48:
            raise ValueError('Invalid query scope or shelf limit')
        if scope == 'part' and (not local_need or not whole_context):
            raise ValueError('Part retrieval requires local need and whole-work context')
        full_query = '\n'.join(s for s in (text, local_need, whole_context) if s)
        eligible = {d['id'] for d in self.documents if (scope == 'both' or d['scope'] == scope)
            and d['source_family'] not in set(exclude_source_families)
            and (allowed_reference_ids is None or d['reference_id'] in set(allowed_reference_ids))}
        tokens = words(full_query)
        lexical = []
        if tokens:
            expression = ' OR '.join('"' + t.replace('"', '""') + '"' for t in dict.fromkeys(tokens))
            with closing(sqlite3.connect((self.directory / 'lexical.sqlite3').as_uri() + '?mode=ro', uri=True)) as db:
                lexical = [r[0] for r in db.execute('SELECT id FROM documents WHERE documents MATCH ? ORDER BY bm25(documents,0,1,0.35),id', (expression,)) if r[0] in eligible]
        visual, similarities = [], {}
        if query_vector is not None:
            scores = self.vectors @ normalized(query_vector)
            visual = [self.documents[i]['id'] for i in sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.documents[i]['id'])) if self.documents[i]['id'] in eligible]
            similarities = {self.documents[i]['id']: float(scores[i]) for i in range(len(scores)) if self.documents[i]['id'] in eligible}
        relation_scores,matched_edges = {},{}
        registered={edge['predicate'] for doc in self.documents for edge in doc.get('relationship_edges',[])}
        cue_map=self.manifest.get('predicate_query_cues',PREDICATE_CUES)
        whole_predicates=predicate_matches(whole_context or text,registered,cue_map=cue_map)
        brief_predicates=predicate_matches(text,registered,cue_map=cue_map)
        local_predicates=predicate_matches(local_need or '',registered,cue_map=cue_map)
        requested_predicates=whole_predicates|brief_predicates|local_predicates
        for identifier in eligible:
            doc=self.by_id[identifier]
            # Explicit typed edge matching; a word in an alternative or disclaimer
            # cannot masquerade as a source relation. Whole/part context stays distinct.
            matches=[edge for edge in doc.get('relationship_edges',[]) if edge['predicate'] in requested_predicates]
            predicates={edge['predicate'] for edge in matches}
            score=2*len(predicates & local_predicates)+len(predicates & (whole_predicates|brief_predicates))
            if score:
                relation_scores[identifier]=score
                matched_edges[identifier]=matches
        relational = sorted(relation_scores, key=lambda i: (-relation_scores[i], i))
        channels = {'lexical': lexical, 'visual': visual, 'relationship': relational}
        ranks = {name: {identifier: rank + 1 for rank, identifier in enumerate(ids)} for name, ids in channels.items()}
        fused = {identifier: sum(1 / (60 + rank[identifier]) for rank in ranks.values() if identifier in rank) for identifier in eligible}
        ordering = sorted(eligible, key=lambda i: (-fused[i], i))
        allocation = allocation or {'relevant': min(24, limit), 'distant': min(12, max(0, limit - 24)), 'underrepresented': max(0, limit - 36)}
        if set(allocation) != {'relevant', 'distant', 'underrepresented'} or any(type(n) is not int or n < 0 for n in allocation.values()):
            raise ValueError('Invalid shelf allocation')
        selected, families, selected_sources, shortages = [], set(), set(), {}

        def take(ids, category, count):
            used = 0
            for identifier in ids:
                doc = self.by_id[identifier]
                if len(selected) >= limit or used >= count:
                    break
                if doc['source_family'] in families or doc['reference_id'] in selected_sources:
                    continue
                selected.append({'id': identifier, 'reference_id': doc['reference_id'], 'region_id': doc['region_id'],
                    'allocation': category, 'rrf_score': fused[identifier],
                    'ranks': {name: values.get(identifier) for name, values in ranks.items()},
                    'cosine_similarity': similarities.get(identifier),
                    'relationship_matches':matched_edges.get(identifier,[])})
                families.add(doc['source_family']); selected_sources.add(doc['reference_id']); used += 1
            shortages[category] = max(0, count - used)

        take(ordering, 'relevant', allocation['relevant'])
        if preferred_object_families is not None:
            if not isinstance(preferred_object_families,(list,tuple)) or not preferred_object_families or any(not isinstance(f,str) or not f for f in preferred_object_families):raise ValueError('Preferred object families require nonempty family strings')
            main_families=set(preferred_object_families)
        else:
            # The first two relevant items define the core domain. Including all24
            # relevant families would often leave no meaningful distant candidates.
            main_families={self.by_id[x['id']]['object_family'] for x in selected[:2]}
        distant = [i for i in relational if self.by_id[i]['object_family'] not in main_families and matched_edges.get(i)]
        take(distant, 'distant', allocation['distant'])
        rng = random.Random(seed)
        sample = sorted(eligible); rng.shuffle(sample)
        family_counts = Counter(d['object_family'] for d in self.documents if d['scope'] == 'whole')
        sample.sort(key=lambda i: family_counts[self.by_id[i]['object_family']])
        take(sample, 'underrepresented', allocation['underrepresented'])
        missing = limit - len(selected)
        take(ordering, 'fallback_relevant', missing)
        return {'schema_version': 1, 'index_id': self.manifest['index_id'], 'query': text,
            'scope': scope, 'local_need': local_need, 'whole_context': whole_context, 'seed': seed,
            'channels': channels, 'selected': selected, 'allocation': allocation, 'shortages': shortages,
            'eligible_count': len(eligible), 'requested_count': limit,
            'fusion': 'reciprocal rank fusion, k=60; source-family and parent dedup before shelf output',
            'clip_admission': clip_admission, 'visual_available': query_vector is not None,
            'relationship_origin': 'explicit typed source relations; machine/operator interpretations remain unvalidated',
            'relationship_query':{'brief_predicates':sorted(brief_predicates),'whole_predicates':sorted(whole_predicates),'local_predicates':sorted(local_predicates)},
            'distant_policy':{'core_object_families':sorted(main_families),'rule':'Match an explicit requested predicate in a source outside the first two relevant items or declared preferred families'},
            'relationship_channel_available':any('relationship_edges' in d for d in self.documents),
            'source_audit_exclusions':(self.manifest.get('source_visual_audit') or {}).get('shortlist_exclusions',[]),
            'human_usefulness': None, 'exclusions': {'source_families': sorted(exclude_source_families),
                                                    'allowed_reference_ids': sorted(allowed_reference_ids) if allowed_reference_ids is not None else None}}


AUDIT_QUERIES = [
    ('whole', 'An enclosure that can support and confine, without assuming a single cultural meaning'),
    ('whole', 'Small figure and expansive empty landscape; scale relationship'),
    ('whole', 'Visible repair preserves a rupture; a join can remain conspicuous'),
    ('whole', 'A repeated shared structure separates its occupants'),
    ('part', 'A woven boundary with an interruption'),
    ('part', 'A material seam or fastener joining incompatible surfaces'),
    ('part', 'Negative space around a small presence'),
    ('part', 'An opening that allows passage while marking a limit'),
    ('whole', 'Documented material and date of a textile, without invented symbolism'),
    ('whole', 'A scientific instrument that supports measurement rather than resembles decoration'),
    ('whole', 'Natural branching structure with repeated relations, not a tree noun alone'),
    ('part', 'Decorative symmetry as a distractor for a load-bearing support relationship'),
]


def audit(index, output, *, encoder=None):
    rows = []
    for scope, query in AUDIT_QUERIES:
        local_need=query if scope == 'part' else None
        whole_context='Develop a collage with visibly interdependent parts; assess whole-work effects.' if scope == 'part' else None
        visual_query='\n'.join(dict.fromkeys(s for s in (query,local_need,whole_context) if s))
        vector, admission = encoder.text(visual_query) if encoder else (None, None)
        result = index.query(query, scope=scope,
            local_need=local_need,
            whole_context=whole_context,
            query_vector=vector, clip_admission=admission, limit=8,
            allocation={'relevant': 4, 'distant': 2, 'underrepresented': 2})
        for item in result['selected']:
            index.asset(item['id'])
        rows.append(result)
    empty = [i+1 for i,row in enumerate(rows) if not row['selected']]
    report = {'schema_version': 1, 'status': 'grounded_returns_verified' if not empty else 'grounded_but_incomplete', 'queries': rows,
              'empty_query_ids': empty,
              'query_count': len(rows), 'human_usefulness': None,
              'limits': 'ID/hash resolution is automated; contextual relevance and distractor rejection await human review.'}
    write_json(Path(output), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--corpus', type=Path, default=Path('artifacts/overnight/corpus'))
    build.add_argument('--output', type=Path, default=Path('artifacts/overnight/indexes'))
    build.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    build.add_argument('--resource-root', type=Path, default=Path('artifacts/overnight'))
    for command in ('query', 'audit'):
        p = sub.add_parser(command)
        p.add_argument('--index', type=Path, required=True)
        p.add_argument('--output', type=Path, required=True)
        if command == 'query':
            p.add_argument('--text', required=True)
            p.add_argument('--scope', choices=['whole', 'part', 'both'], default='both')
            p.add_argument('--local-need')
            p.add_argument('--whole-context')
            p.add_argument('--image', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'build':
        with SessionBudget(args.resource_root) as budget, budget.workload_lock():
            print(build_index(args.corpus, args.output, config=budget.config, device=args.device))
    else:
        index, encoder = RetrievalIndex(args.index), ClipEncoder()
        try:
            if args.command == 'audit':
                result = audit(index, args.output, encoder=encoder)
                print(json.dumps({'status': result['status'], 'queries': len(result['queries'])}))
            else:
                vector, admission = (encoder.images([args.image])[0], {'mode': 'image_to_image', 'sha256': sha256(args.image)}) if args.image else encoder.text('\n'.join(dict.fromkeys(s for s in (args.text,args.local_need,args.whole_context) if s)))
                result = index.query(args.text, scope=args.scope, local_need=args.local_need,
                    whole_context=args.whole_context, query_vector=vector, clip_admission=admission)
                write_json(args.output, result)
                print(json.dumps({'selected': len(result['selected']), 'output': str(args.output)}))
        finally:
            encoder.close()


if __name__ == '__main__':
    main()
