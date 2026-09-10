"""Seeded exposure, before model selection; never an artistic reward.

One draw per research session. The eligible pool, seed, source bytes and failures
are retained. A draw interrupted before completion cannot silently be repeated.
"""
from __future__ import annotations

import json
import random

from .artifacts import sha256
from .corpus import dump, normalize
from .evidence import immutable_json
from .ledger import BudgetExceeded, SpecMismatch


# Reused from the standing catalog's operator-authored breadth queries. These are
# an access policy, not a model's discovery or a representative sample of art.
BREADTH_QUERIES = (
    'African textile', 'Japanese ink painting', 'Mesoamerican vessel',
    'European preparatory drawing', 'Islamic calligraphy', 'Indian miniature painting',
    'Oceanic sculpture', 'Chinese landscape', 'American portrait', 'scientific instrument',
    'anatomical illustration', 'architecture plan', 'body armor', 'mourning',
    'ritual procession', 'repair', 'industrial design', 'photographic study',
)


def _pool(backend):
    """Eligible whole works only; deduplicate details and candidate aliases."""
    pool = {}
    if backend.index:
        backend.index.verify()
        allowed = backend._allowed_index_references()
        for doc in backend.index.documents:
            ref = doc['reference']
            if doc['scope'] != 'whole' or doc['reference_id'] not in allowed:
                continue
            if ref['provider'] not in backend.config.providers:
                continue
            pool[ref['id']] = {'source_id': doc['id'], 'reference_id': ref['id'],
                **{k: ref.get(k) for k in ('title', 'maker', 'culture', 'date', 'medium', 'description', 'provider', 'url')},
                'kind': 'image', 'acquired': True, 'origin': 'frozen_index'}
    for ref in backend.store.references():
        if backend._excluded(ref.id, ref.source_family) or ref.provider not in backend.config.providers:
            continue
        pool.setdefault(ref.id, {'source_id': backend._acquired_identifier(ref.id), 'reference_id': ref.id,
            **{k: getattr(ref, k) for k in ('title', 'maker', 'culture', 'date', 'medium', 'description', 'provider', 'url')},
            'kind': 'image', 'acquired': True, 'origin': 'session_acquisition'})
    for identifier, raw in backend.db.execute('SELECT id,payload FROM candidates ORDER BY id'):
        body = json.loads(raw)
        if body['kind'] != 'image' or body['provider'] not in backend.config.providers:
            continue
        fields, _ = normalize(body['provider'], body['data'], body['envelope'], *body['raw_info'], body['query'])
        if backend._excluded(identifier, fields['source_family'], fields['id']):
            continue
        pool.setdefault(fields['id'], {**backend._candidate_summary(identifier, body), 'reference_id': fields['id']})
    return [pool[key] for key in sorted(pool)]


def expose(backend, policy):
    """Return saved exposure or make one bounded, query-independent draw.

With no existing eligible works, sample one configured breadth query/provider
and draw from at most eight returned works. All network caps still apply.
"""
    intent_path = backend.root / 'chance_exposure_started.json'
    result_path = backend.root / 'chance_exposure.json'
    identity = {'source_identity': backend.identity['identity_sha256'], 'policy': policy}
    if result_path.exists():
        record = json.loads(result_path.read_text())
        if record['identity'] != identity:
            raise SpecMismatch('Frozen chance exposure policy changed')
        snapshot = record['result']
        path = backend.root / snapshot['snapshot_path']
        if not path.resolve().is_relative_to(backend.root) or sha256(path) != snapshot['snapshot_sha256']:
            raise SpecMismatch('Chance exposure snapshot missing or changed')
        saved = json.loads(path.read_text())
        if saved != {k: v for k, v in snapshot.items() if k not in ('snapshot_id', 'snapshot_path', 'snapshot_sha256')}:
            raise SpecMismatch('Chance exposure result differs from saved snapshot')
        return snapshot
    if intent_path.exists():
        if json.loads(intent_path.read_text())['identity'] != identity:
            raise SpecMismatch('Frozen chance exposure intent changed')
        raise SpecMismatch('Chance exposure is unsettled; inspect retained requests before recovery; no redraw')
    rng = random.Random(policy['seed'])
    pool = _pool(backend)
    queries = policy['breadth_queries']
    bootstrap = None
    if not pool and backend.config.allow_network and backend.config.providers and queries:
        bootstrap = {'query': rng.choice(queries), 'provider': rng.choice(backend.config.providers), 'limit': 8}
    immutable_json(intent_path, {'identity': identity, 'eligible_pool': pool, 'bootstrap': bootstrap})
    error = None
    search_snapshot = None
    try:
        if bootstrap:
            result = backend.search(mode='remote', **bootstrap)
            search_snapshot = {k: result[k] for k in ('snapshot_id', 'snapshot_path', 'snapshot_sha256')}
            # Search added only bounded, rights-checked candidates. Deduplicate
            # normalized work IDs, then freeze the actual selection population.
            pool = _pool(backend)
    except (BudgetExceeded, SpecMismatch):
        raise
    except Exception as exc:
        # No replacement provider/query and no retry to get a more pleasing draw.
        error = f'{type(exc).__name__}: {exc}'
        pool = []
    selected = rng.sample(pool, min(policy['count'], len(pool)))
    result = backend._snapshot('chance_exposure', {
        'seed': policy['seed'], 'requested_count': policy['count'],
        'eligible_pool': pool, 'eligible_count': len(pool), 'bootstrap': bootstrap,
        'search_snapshot': search_snapshot, 'error': error,
        'status': 'failed' if error else 'complete' if selected else 'empty',
        'sources': [{**s, 'exposure': 'chance', 'exposure_origin': 'operator_authored_random_sampling',
                     'relevance_score': None, 'score_origin': 'not_scored'} for s in selected],
        'policy': 'Uniform work sampling without replacement within the saved eligible pool. '
                  'An empty pool permits one random operator breadth query/provider. '
                  'Catalog access and search ranking bias the population; no semantic distance, novelty or quality is claimed. '
                  'Exposure does not require acquisition or use. No reward or human judgment.',
    })
    immutable_json(result_path, {'identity': identity, 'result': result})
    return result


def chance_candidates(state, steps):
    """Keep the small exposure shelf outside relevance-ranked context pruning."""
    found = {}
    for row in steps:
        if row.get('tool_error'):
            continue
        for source in row.get('tool_result', {}).get('sources', []):
            if source.get('exposure') != 'chance':
                continue
            sid = source['source_id']
            if sid in state['rejected'] or state['aliases'].get(sid, sid) in state['opened']:
                continue
            found.setdefault(sid, source)
    return list(found.values())
