"""Bounded, model-directed reference encounters with immutable decision evidence.

Search never downloads image candidates. Acquisition is an explicit second action.
The standing collection and encounter policies are operator scaffolding; retrieval
scores, exploratory samples and proposed contradictions are not artistic judgments.
"""
from __future__ import annotations

from collections import Counter
import argparse
from contextlib import closing
from html.parser import HTMLParser
import ipaddress
import json
from pathlib import Path
import random
import sqlite3
from typing import Literal
import urllib.error
import urllib.parse
import urllib.request

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import sha256
from .corpus import (CorpusStore, OperationRight, SourceConfig, atomic, digest, dump, ingest_record, normalize, utcnow)
from .retrieval import ClipEncoder, RetrievalIndex, words
from .resources import load_config


PROVIDER_HOSTS = {
    'met': ['collectionapi.metmuseum.org', 'images.metmuseum.org'],
    'cleveland': ['openaccess-api.clevelandart.org', 'openaccess-cdn.clevelandart.org'],
    'artic': ['api.artic.edu', 'www.artic.edu'],
}
MODES = ('text', 'visual', 'relationship', 'contextual', 'historical',
         'exploratory', 'contradiction', 'remote')


class ContextPolicy(BaseModel):
    """An operator-declared permission, restricted to an exact HTTPS host/prefix."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    host: str
    path_prefix: str = '/'
    rights: OperationRight

    @model_validator(mode='after')
    def validate_policy(self):
        if self.host != self.host.lower() or any(x in self.host for x in '/:@*'):
            raise ValueError('Context host must be an exact lowercase hostname')
        if not self.path_prefix.startswith('/') or '..' in self.path_prefix:
            raise ValueError('Context path prefix must be absolute and cannot traverse')
        if self.rights.status != 'allowed':
            raise ValueError('Context acquisition requires an explicit allowed text right')
        return self


class ContextSeed(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    url: str
    title: str = Field(min_length=1, max_length=500)
    author: str = Field(min_length=1, max_length=300)
    description: str = Field(default='', max_length=2000)


class ResearchSourceConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    schema_version: Literal[1] = 1
    allow_network: bool = False
    providers: list[Literal['met', 'artic', 'cleveland']] = Field(
        default_factory=lambda: ['met', 'artic', 'cleveland'], max_length=3)
    max_requests: int = Field(default=48, ge=0, le=500)
    max_network_bytes: int = Field(default=64 * 1024**2, ge=1024, le=2 * 1024**3)
    max_response_bytes: int = Field(default=2 * 1024**2, ge=1024, le=8 * 1024**2)
    max_image_bytes: int = Field(default=12 * 1024**2, ge=1024, le=32 * 1024**2)
    max_storage_bytes: int = Field(default=256 * 1024**2, ge=1024, le=2 * 1024**3)
    max_acquisitions: int = Field(default=12, ge=0, le=200)
    max_candidates: int = Field(default=300, ge=1, le=2000)
    max_text_chars: int = Field(default=20000, ge=256, le=80000)
    long_edge: int = Field(default=1024, ge=256, le=2048)
    enable_clip_text: bool = False
    clip_cache_dir: str = '.cache/huggingface'
    excluded_source_ids: list[str] = Field(default_factory=list, max_length=5000)
    excluded_source_families: list[str] = Field(default_factory=list, max_length=5000)
    catalog_path: str | None = None
    catalog_limit: int = Field(default=64, ge=1, le=300)
    context_policies: list[ContextPolicy] = Field(default_factory=list, max_length=40)
    context_seeds: list[ContextSeed] = Field(default_factory=list, max_length=100)
    coverage: dict[str, list[str]] = Field(default_factory=lambda: {
        'regions': [], 'periods': [], 'materials': [], 'queries': []})
    authorship: Literal['operator_supplied_policy'] = 'operator_supplied_policy'


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.links = [], []
        self.suppressed = 0

    def handle_starttag(self, tag, attrs):
        if tag in ('script', 'style', 'noscript'):
            self.suppressed += 1
        if not self.suppressed and tag == 'a':
            href = dict(attrs).get('href')
            if href and len(self.links) < 200:
                self.links.append(href)

    def handle_endtag(self, tag):
        if tag in ('script', 'style', 'noscript') and self.suppressed:
            self.suppressed -= 1

    def handle_data(self, data):
        if not self.suppressed:
            self.parts.append(data)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    # Each redirect is another explicit request: it cannot evade request/byte caps
    # or silently move to a host whose text rights were not supplied.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('Redirect rejected; retain failure and configure the canonical URL')


class SourceRequestFailed(RuntimeError):
    """A retained failed request; raising this never dispatches another request."""
    def __init__(self, payload):
        self.code = payload.get('http_status')
        super().__init__('Previous source request failed or is unsettled; no silent retry')


class ResearchSources:
    """Per-session source backend. All return values are JSON-serializable.

    ``search`` returns ``sources`` with stable ``source_id``/``candidate_id``;
    ``open`` returns ``images`` with absolute paths and hashes, exact attributed
    metadata, claims and observed links. ``acquire`` admits only seen candidates;
    ``follow`` admits only an observed link and an explicit context-right policy.
    ``summary`` exposes durable counts, coverage policy and immutable identity.

    Pass an encoder/opener only for an explicit runtime implementation or fixture.
    The default network transport never follows redirects or retries failures.
    """
    def __init__(self, root, config=None, index_path=None, *, encoder=None, opener=None, before_fetch=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = ResearchSourceConfig.model_validate(config or {})
        self.index = RetrievalIndex(index_path) if index_path else None
        self.encoder = encoder
        self.before_fetch = before_fetch
        self.opener = opener or urllib.request.build_opener(_NoRedirect()).open
        self.store = CorpusStore(self.root / 'corpus')
        self.db = sqlite3.connect(self.root / 'research_sources.sqlite3', timeout=30)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS candidates(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS text_sources(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS source_aliases(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS snapshots(id TEXT PRIMARY KEY, path TEXT NOT NULL, sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY, url TEXT NOT NULL, state TEXT NOT NULL,
                charged_bytes INTEGER NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS acquisitions(id TEXT PRIMARY KEY, state TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL,
                kind TEXT NOT NULL, payload TEXT NOT NULL);
        ''')
        for table in ('candidates', 'text_sources', 'source_aliases', 'snapshots', 'events'):
            for operation in ('UPDATE', 'DELETE'):
                self.db.execute(f'''CREATE TRIGGER IF NOT EXISTS {table}_no_{operation.lower()}
                    BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'Research evidence is immutable'); END''')
        self.db.commit()
        self.catalog_import = self._import_catalog()
        identity = {'schema_version': 1, 'config': self.config.model_dump(mode='json'),
                    'index_path': str(self.index.directory) if self.index else None,
                    'index_id': self.index.manifest['index_id'] if self.index else None,
                    'index_manifest_sha256': sha256(self.index.directory / 'manifest.json') if self.index else None,
                    'catalog_import_sha256': sha256(self.root / 'catalog_import.json') if self.catalog_import else None}
        identity['identity_sha256'] = digest(dump(identity).encode())
        path = self.root / 'inputs.json'
        if path.exists():
            if json.loads(path.read_bytes()) != identity:
                raise ValueError('Research source inputs changed; start a new session')
        else:
            atomic(path, dump(identity).encode())
        self.identity = identity
        for seed in self.config.context_seeds:
            self._context_policy(seed.url)
            self._candidate({'kind': 'context', **seed.model_dump(mode='json'),
                             'origin': 'operator_context_seed', 'query': ''})

    def close(self):
        if self.encoder and hasattr(self.encoder, 'close'):
            self.encoder.close()
        self.store.db.close()
        self.db.close()

    def _event(self, kind, payload):
        with self.db:
            self.db.execute('INSERT INTO events(at,kind,payload) VALUES (?,?,?)',
                            (utcnow(), kind, dump(payload)))

    def _snapshot(self, operation, payload):
        body = {'operation': operation, 'source_identity': self.identity['identity_sha256'],
                'context_trust': 'untrusted_source_data', **payload}
        raw = dump(body).encode()
        identifier = 'encounter_' + digest(raw)
        relative = f'snapshots/{identifier}.json'
        path = self.root / relative
        if path.exists() and path.read_bytes() != raw:
            raise ValueError('Research snapshot hash mismatch')
        if not path.exists():
            self.check_budget(len(raw))
            atomic(path, raw)
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO snapshots VALUES (?,?,?)',
                            (identifier, relative, digest(raw)))
        self._event(operation, {'snapshot_id': identifier})
        return {**body, 'snapshot_id': identifier, 'snapshot_path': relative, 'snapshot_sha256': digest(raw)}

    @staticmethod
    def _url(url):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                or parsed.port not in (None, 443) or parsed.fragment or '\\' in url):
            raise ValueError('Only canonical HTTPS source URLs without credentials/fragments are allowed')
        try:
            ipaddress.ip_address(parsed.hostname)
        except ValueError:
            pass
        else:
            raise ValueError('IP-address source URLs are not allowed')
        if parsed.hostname in ('localhost',) or parsed.hostname.endswith(('.local', '.localhost')):
            raise ValueError('Local source hosts are not allowed')
        return parsed

    def _context_policy(self, url):
        parsed = self._url(url)
        path = urllib.parse.unquote(parsed.path or '/')
        if '..' in path.split('/'):
            raise ValueError('Source path traversal rejected')
        for policy in self.config.context_policies:
            prefix = policy.path_prefix
            if parsed.hostname == policy.host and (path == prefix or path.startswith(prefix.rstrip('/') + '/')):
                return policy
        raise ValueError('No configured text rights for this source host/path')

    def _allowed_url(self, url, kind):
        parsed = self._url(url)
        if kind == 'context':
            self._context_policy(url)
        elif parsed.hostname not in {h for p in self.config.providers for h in PROVIDER_HOSTS[p]}:
            raise ValueError('Source URL host is not an enabled official collection endpoint')

    def _excluded(self, identifier, source_family=None, parent_id=None):
        return (identifier in self.config.excluded_source_ids or parent_id in self.config.excluded_source_ids
                or source_family in self.config.excluded_source_families)

    def _encoder(self):
        if self.encoder is None:
            config = load_config().model_copy(update={
                'clip_revision': self.index.manifest['encoder']['revision'],
                'cache_dir': self.config.clip_cache_dir})
            self.encoder = ClipEncoder(config=config, device='cpu')
        if self.encoder.metadata.get('revision') != self.index.manifest['encoder']['revision']:
            raise ValueError('Research encoder revision must match the frozen index')
        if self.encoder.metadata.get('device', 'cpu') != 'cpu':
            raise ValueError('Research descriptor encoding is CPU only')
        return self.encoder

    def _allowed_index_references(self):
        excluded = {d['reference_id'] for d in self.index.documents
                    if self._excluded(d['id'], d['source_family'], d['reference_id'])}
        return sorted({d['reference_id'] for d in self.index.documents} - excluded)

    def _acquired_identifier(self, identifier):
        row = self.db.execute("SELECT id FROM source_aliases WHERE json_extract(payload,'$.canonical_id')=?",
                              (identifier,)).fetchone()
        return row[0] if row else identifier

    def check_budget(self, extra=0):
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
        if used + extra > self.config.max_storage_bytes:
            raise RuntimeError('Research source storage byte budget exhausted')

    def fetch(self, url, kind='metadata', byte_limit=None, range_bytes=None):
        """Corpus-compatible fetch with durable reservations and no silent retry."""
        if range_bytes is not None:
            raise ValueError('Research source fetch does not support unbounded stream discovery')
        self._allowed_url(url, kind)
        key = digest(dump([url, kind]).encode())
        row = self.db.execute('SELECT state,payload FROM requests WHERE id=?', (key,)).fetchone()
        if row:
            if row[0] != 'complete':
                raise SourceRequestFailed(json.loads(row[1]))
            result = json.loads(row[1])
            raw = self.store.asset_path(result['path']).read_bytes()
            if digest(raw) != result['sha256']:
                raise ValueError('Cached research source hash mismatch')
            return raw, result['path'], result['sha256'], result['retrieved_at']
        if not self.config.allow_network:
            raise ValueError('Network acquisition is disabled by the frozen research source config')
        bound = min(byte_limit or (self.config.max_image_bytes if kind == 'image' else self.config.max_response_bytes),
                    self.config.max_image_bytes if kind == 'image' else self.config.max_response_bytes)
        self.check_budget(bound + 1)
        self.db.execute('BEGIN IMMEDIATE')
        try:
            count, charged = self.db.execute('SELECT COUNT(*), COALESCE(SUM(charged_bytes),0) FROM requests').fetchone()
            if count >= self.config.max_requests or charged + bound + 1 > self.config.max_network_bytes:
                raise RuntimeError('Research network request/byte budget exhausted')
            self.db.execute('INSERT INTO requests VALUES (?,?,?,?,?)',
                            (key, url, 'started', bound + 1, dump({'kind': kind, 'at': utcnow()})))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        raw = b''
        dispatched = False
        try:
            request = urllib.request.Request(url, headers={
                'User-Agent': 'KusamaResearch/0.2 (bounded official collection study)',
                'Accept-Encoding': 'identity'})
            if self.before_fetch is not None:
                self.before_fetch()
            dispatched = True
            with self.opener(request, timeout=25) as response:
                raw = response.read(bound + 1)
                if len(raw) > bound:
                    raise ValueError('Research source response exceeds byte limit')
                content_type = response.headers.get('Content-Type', '').split(';')[0].lower()
                if kind == 'context' and content_type not in ('text/html', 'text/plain', 'application/xhtml+xml'):
                    raise ValueError('Context source must be HTML or plain text; original bytes retained')
            hsh = digest(raw)
            relative = f'raw/research_{kind}/{hsh}.bin'
            atomic(self.store.root / relative, raw)
            result = {'path': relative, 'sha256': hsh, 'retrieved_at': utcnow(),
                      'content_type': content_type, 'url': url, 'bytes': len(raw)}
            with self.db:
                self.db.execute('UPDATE requests SET state=?,charged_bytes=?,payload=? WHERE id=?',
                                ('complete', len(raw), dump(result), key))
            self._event('request_complete', result)
            return raw, relative, hsh, result['retrieved_at']
        except Exception as exc:
            # Retain a full reservation if response byte count is uncertain.
            body_read_error = None
            if isinstance(exc, urllib.error.HTTPError) and not raw:
                try:
                    raw = exc.read(bound + 1)
                except Exception as read_exc:
                    body_read_error = f'{type(read_exc).__name__}: {read_exc}'
            failure = {'url': url, 'error': f'{type(exc).__name__}: {exc}', 'raw_path': None}
            failure['http_status'] = exc.code if isinstance(exc, urllib.error.HTTPError) else None
            failure['body_read_error'] = body_read_error
            failure['dispatched'] = dispatched
            if raw:
                failure['raw_path'] = f'failures/{key}.bin'
                atomic(self.store.root / failure['raw_path'], raw)
            with self.db:
                if not dispatched:
                    self.db.execute('UPDATE requests SET charged_bytes=0 WHERE id=?', (key,))
                self.db.execute('UPDATE requests SET state=?,payload=? WHERE id=?', ('failed', dump(failure), key))
            self._event('request_failed', failure)
            raise

    def _candidate(self, body):
        identifier = 'candidate_' + digest(dump(body).encode())[:32]
        if not self.db.execute('SELECT 1 FROM candidates WHERE id=?', (identifier,)).fetchone():
            if self.db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0] >= self.config.max_candidates:
                raise RuntimeError('Research candidate budget exhausted')
            with self.db:
                self.db.execute('INSERT INTO candidates VALUES (?,?)', (identifier, dump(body)))
        return identifier

    def _import_catalog(self):
        """Freeze admitted standing-catalog metadata without importing its images."""
        if not self.config.catalog_path:
            return None
        frozen_path = self.root / 'catalog_import.json'
        previous_inputs = self.root / 'inputs.json'
        if previous_inputs.exists():
            previous = json.loads(previous_inputs.read_bytes())
            if previous['config'] != self.config.model_dump(mode='json'):
                raise ValueError('Research source inputs changed; start a new session')
            if not frozen_path.exists():
                raise ValueError('Frozen catalog import is missing; do not requery growing catalog')
            if sha256(frozen_path) != previous.get('catalog_import_sha256'):
                raise ValueError('Frozen catalog import hash mismatch')
        if frozen_path.exists():
            snapshot = json.loads(frozen_path.read_bytes())
            if snapshot.get('snapshot_sha256') != digest(dump({
                    k: v for k, v in snapshot.items() if k != 'snapshot_sha256'}).encode()):
                raise ValueError('Frozen catalog import identity changed')
        else:
            catalog = Path(self.config.catalog_path).resolve()
            if catalog == self.root:
                raise ValueError('A research session cannot import its own catalog')
            state = read_source_summary(catalog)
            if state is None:
                raise ValueError('Standing catalog has no initialized source database')
            dbpath = catalog / 'research_sources.sqlite3'
            uri = dbpath.as_uri() + ('?mode=ro' if Path(str(dbpath) + '-wal').exists() else '?mode=ro&immutable=1')
            with closing(sqlite3.connect(uri, uri=True)) as db:
                rows = db.execute('SELECT id,payload FROM candidates ORDER BY id LIMIT ?',
                                  (self.config.catalog_limit,)).fetchall()
            admitted, exclusions, metadata = [], [], {}
            for identifier, raw in rows:
                body = json.loads(raw)
                # IDs derive from the exact saved candidate, not a new guess or
                # arbitrary external URL supplied by an imported record.
                if identifier != 'candidate_' + digest(dump(body).encode())[:32]:
                    raise ValueError('Standing catalog candidate identity changed')
                try:
                    family = body.get('source_family') or 'context:' + urllib.parse.urlsplit(body['url']).hostname
                    if self._excluded(identifier, family, body.get('reference_id')):
                        raise ValueError('Candidate excluded by frozen split/access policy')
                    if body['kind'] == 'image':
                        if body['provider'] not in self.config.providers:
                            raise ValueError('Candidate provider not enabled for this episode')
                        raw_path, expected, _ = body['raw_info']
                        original = (catalog / 'corpus' / raw_path).resolve()
                        if not original.is_relative_to((catalog / 'corpus').resolve()):
                            raise ValueError('Catalog metadata path escapes its source root')
                        if not original.is_file() or sha256(original) != expected:
                            raise ValueError('Standing catalog metadata hash mismatch')
                        if original.stat().st_size > self.config.max_response_bytes:
                            raise ValueError('Standing metadata exceeds episode response-byte limit')
                        fields, _ = normalize(body['provider'], body['data'], body['envelope'], *body['raw_info'], body['query'])
                        if fields['id'] != body['reference_id'] or self._excluded(fields['id'], fields['source_family']):
                            raise ValueError('Imported normalized candidate is excluded or inconsistent')
                        target = self.store.asset_path(raw_path)
                        self.check_budget(original.stat().st_size)
                        if target.exists() and sha256(target) != expected:
                            raise ValueError('Existing imported metadata differs')
                        if not target.exists():
                            atomic(target, original.read_bytes())
                        metadata[raw_path] = {'path': raw_path, 'sha256': expected}
                    elif body['kind'] == 'context':
                        self._context_policy(body['url'])
                    else:
                        raise ValueError('Unknown standing candidate kind')
                except (KeyError, TypeError, ValueError) as exc:
                    # Corruption is not merely a coverage exclusion.
                    if 'hash mismatch' in str(exc) or 'differs' in str(exc):
                        raise
                    exclusions.append({'candidate_id': identifier, 'reason': str(exc)})
                    continue
                admitted.append({'candidate_id': identifier, 'body': body,
                                 'candidate_sha256': digest(dump(body).encode())})
            snapshot = {'schema_version': 1, 'origin': 'operator_selected_standing_catalog',
                'catalog_path': str(catalog), 'catalog_identity': state['source_identity'],
                'catalog_inputs_sha256': sha256(catalog / 'inputs.json'),
                'scanned_candidates': len(rows), 'catalog_limit': self.config.catalog_limit,
                'candidates': admitted, 'metadata': list(metadata.values()), 'exclusions': exclusions,
                'policy': 'Immutable bounded metadata snapshot; no images or network fetched. '
                          'Model/operator acquisition remains a separate action under episode limits.'}
            snapshot['snapshot_sha256'] = digest(dump(snapshot).encode())
            atomic(frozen_path, dump(snapshot).encode())
        for asset in snapshot['metadata']:
            if sha256(self.store.asset_path(asset['path'])) != asset['sha256']:
                raise ValueError('Frozen imported metadata hash mismatch')
        incoming = snapshot['candidates']
        existing = {row[0] for row in self.db.execute('SELECT id FROM candidates')}
        if len(existing | {row['candidate_id'] for row in incoming}) > self.config.max_candidates:
            raise ValueError('Standing import exceeds episode candidate budget')
        with self.db:
            for row in incoming:
                body = row['body']
                if (digest(dump(body).encode()) != row['candidate_sha256'] or
                        row['candidate_id'] != 'candidate_' + digest(dump(body).encode())[:32]):
                    raise ValueError('Frozen catalog candidate hash mismatch')
                previous = self.db.execute('SELECT payload FROM candidates WHERE id=?', (row['candidate_id'],)).fetchone()
                if previous and previous[0] != dump(body):
                    raise ValueError('Imported candidate conflicts with existing evidence')
                self.db.execute('INSERT OR IGNORE INTO candidates VALUES (?,?)', (row['candidate_id'], dump(body)))
        return snapshot

    def _candidate_summary(self, identifier, row):
        imported = bool(self.catalog_import and any(r['candidate_id'] == identifier for r in self.catalog_import['candidates']))
        metadata = {}
        if row['kind'] == 'image':
            fields, _ = normalize(row['provider'], row['data'], row['envelope'], *row['raw_info'], row['query'])
            metadata = {key: fields[key] for key in ('maker', 'culture', 'date', 'medium', 'tags')}
        acquisition = self.db.execute('SELECT state,payload FROM acquisitions WHERE id=?', (identifier,)).fetchone()
        return {'candidate_id': identifier, 'source_id': identifier, 'kind': row['kind'],
                'title': row.get('title'), 'provider': row.get('provider'), 'url': row.get('url'),
                'description': row.get('description', ''), **metadata,
                'acquired': bool(acquisition and acquisition[0] == 'complete'),
                'acquired_source_id': json.loads(acquisition[1]).get('source_id') if acquisition and acquisition[0] == 'complete' else None,
                'acquisition_state': acquisition[0] if acquisition else 'not_acquired',
                'origin': row.get('origin', 'official_collection_metadata'),
                'catalog_import': {'snapshot_sha256': self.catalog_import['snapshot_sha256'],
                    'catalog_path': self.catalog_import['catalog_path'],
                    'catalog_identity': self.catalog_import['catalog_identity']} if imported else None,
                'metadata_sha256': (row.get('raw_info') or [None, None])[1]}

    def _remote(self, query, provider, limit, query_field=None):
        if provider not in self.config.providers:
            raise ValueError('Select an enabled museum provider')
        params = {'q': query, 'limit': limit}
        if provider == 'met':
            params['hasImages'] = 'true'
            params['offset'] = 0
            url = 'https://collectionapi.metmuseum.org/public/collection/v1.1/search?'
        elif provider == 'cleveland':
            params['cc0'] = '1'
            url = 'https://openaccess-api.clevelandart.org/api/artworks/?'
        else:
            params.update({'fields': 'id,title,api_link,image_id,is_public_domain,artist_display,place_of_origin,date_display,medium_display,dimensions,description,artwork_type_title,classification_titles,credit_line',
                           'query[term][is_public_domain]': 'true'})
            url = 'https://api.artic.edu/api/v1/artworks/search?'
        if query_field:
            if provider == 'artic':
                fields = {'artist': ['artist_title'], 'work': ['title'],
                    'subject': ['subject_titles', 'term_titles'], 'technique': ['term_titles', 'classification_titles'],
                    'object': ['classification_titles', 'term_titles']}[query_field]
                params = {'params': json.dumps({'limit': limit, 'fields': params['fields'].split(','),
                    'query': {'bool': {'filter': [{'term': {'is_public_domain': True}}],
                        'must': [{'multi_match': {'query': query, 'fields': fields}}]}}}, separators=(',', ':'))}
            elif provider == 'cleveland':
                # CMA documents q for keywords, not a keywords filter.
                field = {'artist': 'artists', 'object': 'type', 'work': 'title', 'technique': 'medium'}.get(query_field, 'q')
                params.pop('q', None); params[field] = query
        raw, path, hsh, when = self.fetch(url + urllib.parse.urlencode(params))
        envelope = json.loads(raw)
        key = 'objectIDs' if provider == 'met' else 'data'
        if not isinstance(envelope, dict) or key not in envelope:
            raise ValueError(f'Malformed {provider} search response: missing {key}; raw response retained')
        rows = envelope[key]
        if provider == 'met' and rows is None and envelope.get('total') == 0:
            rows = []
        if not isinstance(rows, list):
            raise ValueError(f'Malformed {provider} search response: {key} must be an array')
        rows = rows[:limit]
        result, exclusions = [], []
        for row in rows:
            info, record_envelope = (path, hsh, when), envelope
            object_id = str(row) if type(row) is int else str(row.get('id', '')) if isinstance(row, dict) else ''
            try:
                if provider == 'met':
                    if type(row) is not int or row <= 0:
                        raise ValueError('Malformed museum object identifier')
                    body, p, h, at = self.fetch(f'https://collectionapi.metmuseum.org/public/collection/v1/objects/{row}')
                    requested_id = row
                    row, info, record_envelope = json.loads(body), (p, h, at), {}
                    if not isinstance(row, dict) or row.get('objectID') != requested_id:
                        raise ValueError('Museum object response does not match requested ID')
                if not isinstance(row, dict):
                    raise ValueError('Museum record must be an object')
                fields, _ = normalize(provider, row, record_envelope, *info, query)
                if self._excluded(fields['id'], fields['source_family']):
                    raise ValueError('Source excluded by frozen split/access policy')
            except (urllib.error.HTTPError, SourceRequestFailed) as exc:
                if exc.code not in (404, 410):
                    raise
                exclusions.append({'object_id': object_id, 'reason': f'Object unavailable: HTTP {exc.code}; request failure retained'})
                continue
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                exclusions.append({'object_id': object_id, 'reason': f'{type(exc).__name__}: {exc}'})
                continue
            body = {'kind': 'image', 'provider': provider, 'title': fields['title'],
                    'url': fields['url'], 'description': fields['description'], 'query': query,
                    'data': row, 'envelope': record_envelope, 'raw_info': info,
                    'source_family': fields['source_family'],
                    'reference_id': fields['id'], 'rights': fields['rights'].model_dump(mode='json')}
            identifier = self._candidate(body)
            result.append(self._candidate_summary(identifier, body))
        return {'sources': result, 'exclusions': exclusions,
                'query_field': query_field, 'policy': 'Official metadata first; only explicit acquire actions download images'}

    def search(self, query, mode='text', limit=8, *, provider=None, source_id=None,
               seed=23, scope='both', local_need=None, whole_context=None, query_field=None):
        if not isinstance(query, str) or not query.strip() or len(query) > 12000:
            raise ValueError('Research query must be nonempty and at most 12000 characters')
        if mode not in MODES or type(limit) is not int or not 1 <= limit <= 24:
            raise ValueError('Invalid research search mode/limit')
        args = {'query': query, 'mode': mode, 'limit': limit, 'provider': provider,
                'source_id': source_id, 'seed': seed, 'scope': scope,
                'local_need': local_need, 'whole_context': whole_context}
        if mode == 'remote':
            return self._snapshot('search', {**args, **self._remote(query, provider, limit, query_field)})
        if provider is not None:
            raise ValueError('Provider is only valid for remote search')
        sources, evidence = [], None
        if self.index:
            self.index.verify()
            vector, admission = None, None
            if mode == 'visual':
                if source_id:
                    if self._excluded(source_id):
                        raise ValueError('Visual source excluded by frozen split/access policy')
                    if source_id in self.index.by_id:
                        doc = self.index.by_id[source_id]
                        if doc['reference_id'] not in self._allowed_index_references():
                            raise ValueError('Visual source excluded by frozen split/access policy')
                        self.index.asset(source_id)
                        vector = self.index.vectors[next(i for i, d in enumerate(self.index.documents) if d['id'] == source_id)]
                        admission = {'mode': 'image_to_image', 'source_id': source_id}
                    else:
                        viewed = self.open(source_id)
                        if not viewed['images']:
                            raise ValueError('Visual query requires an acquired image source')
                        encoder = self._encoder()
                        image = viewed['images'][-1]
                        vector = encoder.images([image['path']])[0]
                        admission = {'mode': 'acquired_image_to_frozen_index', 'source_id': source_id,
                                     'sha256': image['sha256'], 'encoder': self.encoder.metadata}
                elif self.config.enable_clip_text:
                    vector, admission = self._encoder().text(query)
                else:
                    raise ValueError('Visual search needs an indexed source_id or explicitly enabled CLIP text encoding')
            allocation = {'relevant': 0 if mode == 'exploratory' else limit,
                          'distant': 0, 'underrepresented': limit if mode == 'exploratory' else 0}
            evidence = self.index.query(query, scope=scope, local_need=local_need,
                whole_context=whole_context, query_vector=vector, clip_admission=admission,
                limit=limit, seed=seed, allocation=allocation,
                exclude_source_families=self.config.excluded_source_families,
                allowed_reference_ids=self._allowed_index_references())
            identifiers = [row['id'] for row in evidence['selected']]
            if mode in ('text', 'contextual', 'historical', 'contradiction'):
                # Do not disguise no-match fallback results as textual evidence.
                identifiers = evidence['channels']['lexical']
            elif mode == 'relationship':
                identifiers = evidence['channels']['relationship']
            elif mode == 'visual':
                identifiers = evidence['channels']['visual']
            seen_references = set()
            for identifier in identifiers:
                doc = self.index.by_id[identifier]
                if self._excluded(identifier, doc['source_family'], doc['reference_id']):
                    continue
                if doc['reference_id'] in seen_references:
                    continue
                seen_references.add(doc['reference_id'])
                sources.append({'source_id': identifier, 'reference_id': doc['reference_id'],
                    'title': doc['reference']['title'], 'kind': 'image', 'scope': doc['scope'],
                    'provider': doc['reference']['provider'], 'url': doc['reference']['url'],
                    'acquired': True, 'origin': 'frozen_index',
                    'relationship_edges': doc.get('relationship_edges', [])})
                if len(sources) >= limit:
                    break
        elif mode in ('visual', 'relationship'):
            raise ValueError('This search mode requires a frozen retrieval index')
        tokens = set(words(query))
        supplemental = []
        if mode not in ('visual', 'relationship'):
            for ref in self.store.references():
                if self._excluded(ref.id, ref.source_family):
                    continue
                body = ref.model_dump(mode='json')
                text = ' '.join(str(body.get(k) or '') for k in ('title', 'maker', 'culture', 'date', 'medium', 'description'))
                supplemental.append((len(tokens & set(words(text))), {'source_id': self._acquired_identifier(ref.id), 'title': ref.title,
                    'kind': 'image', 'provider': ref.provider, 'url': ref.url, 'acquired': True,
                    'origin': 'session_acquisition', 'visual_indexed': False}))
            for identifier, raw in self.db.execute('SELECT id,payload FROM text_sources ORDER BY id'):
                body = json.loads(raw)
                if self._excluded(identifier, body['source_family']):
                    continue
                supplemental.append((len(tokens & set(words(body['title'] + ' ' + body['text']))),
                    {'source_id': identifier, 'title': body['title'], 'kind': 'context',
                     'url': body['url'], 'acquired': True, 'origin': body['origin']}))
            for identifier, raw in self.db.execute('SELECT id,payload FROM candidates ORDER BY id'):
                if self.db.execute("SELECT 1 FROM acquisitions WHERE id=? AND state='complete'", (identifier,)).fetchone():
                    continue
                body = json.loads(raw)
                family = body.get('source_family') or 'context:' + urllib.parse.urlsplit(body['url']).hostname
                if self._excluded(identifier, family, body.get('reference_id')):
                    continue
                # The query that discovered a candidate is not evidence about it.
                text = body['title'] + ' ' + body.get('description', '')
                if body['kind'] == 'image':
                    fields, _ = normalize(body['provider'], body['data'], body['envelope'], *body['raw_info'], body['query'])
                    text += ' ' + ' '.join(str(fields.get(k) or '') for k in ('maker', 'culture', 'date', 'medium'))
                supplemental.append((len(tokens & set(words(text))), self._candidate_summary(identifier, body)))
            if mode == 'exploratory':
                random.Random(seed).shuffle(supplemental)
            else:
                supplemental = sorted((item for item in supplemental if item[0]), key=lambda x: (-x[0], x[1]['source_id']))
            sources = sources + [row for _, row in supplemental]
        unique = {row['source_id']: row for row in sources}
        sources = list(unique.values())[:limit]
        policy = {'text': 'lexical source matching', 'contextual': 'documented metadata and attributed text matching',
                  'historical': 'documented context/date/place text matching; not a learned historical representation',
                  'contradiction': 'model-authored alternative query; returned sources are leads, not established counterevidence',
                  'relationship': 'explicit typed edges; original claim attribution and unvalidated status retained',
                  'visual': 'frozen CLIP similarity; not compositional understanding or quality',
                  'exploratory': 'operator-authored seeded underrepresented-family encounters; not model-discovered diversity'}[mode]
        return self._snapshot('search', {**args, 'sources': sources, 'retrieval': evidence,
            'policy': policy, 'human_usefulness': None, 'known_facts_policy': 'Documented source statements are attributed claims; interpretations remain hypotheses'})

    def _links(self, urls):
        result = []
        for url in dict.fromkeys(urls):
            try:
                self._context_policy(url)
            except ValueError:
                eligible = False
            else:
                eligible = True
            result.append({'link_id': 'link_' + digest(url.encode())[:24], 'url': url,
                           'follow_allowed': eligible})
        return result

    def _reference_open(self, ref, store, source_id, region=None, claims=None):
        if self._excluded(source_id, ref.source_family, ref.id):
            raise ValueError('Source excluded by frozen split/access policy')
        images = []
        assets = [(ref.view, source_id if region is None else self._acquired_identifier(ref.id), 'whole')]
        if region:
            assets.append((region.crop, source_id, 'part'))
        for asset, identifier, scope in assets:
            path = store.asset_path(asset.path)
            if sha256(path) != asset.sha256:
                raise ValueError('Opened research asset hash mismatch')
            images.append({'path': str(path), 'sha256': asset.sha256, 'source_id': identifier, 'scope': scope})
        if sha256(store.asset_path(ref.original.path)) != ref.original.sha256:
            raise ValueError('Original research source hash mismatch')
        raw = store.asset_path(ref.raw_response_path)
        if sha256(raw) != ref.raw_response_hash:
            raise ValueError('Raw source metadata hash mismatch')
        if self.index and store.root == Path(self.index.manifest['corpus_root']).resolve():
            details = [{'source_id': d['id'], 'scope': 'part', 'localization': d['region']['localization']}
                       for d in self.index.documents if d['reference_id'] == ref.id and d['scope'] == 'part']
        else:
            details = [{'source_id': self._acquired_identifier(r.id), 'scope': 'part', 'localization': r.localization}
                       for r in store.regions(ref.id)]
        return {'source_id': source_id, 'canonical_reference_id': ref.id, 'kind': 'image', 'title': ref.title,
                'reference': ref.model_dump(mode='json'), 'images': images,
                'details': details,
                'text': ref.description, 'claims': claims if claims is not None else [
                    c.model_dump(mode='json') for c in store.claims() if c.reference_id == ref.id],
                'links': self._links([ref.url]),
                'provenance': {'raw_path': str(raw), 'raw_sha256': ref.raw_response_hash,
                    'original_path': str(store.asset_path(ref.original.path)), 'original_sha256': ref.original.sha256,
                    'attribution': ref.attribution, 'context_trust': 'untrusted_source_data'}}

    def open(self, source_id):
        if self._excluded(source_id):
            raise ValueError('Source excluded by frozen split/access policy')
        alias = self.db.execute('SELECT payload FROM source_aliases WHERE id=?', (source_id,)).fetchone()
        lookup_id = json.loads(alias[0])['canonical_id'] if alias else source_id
        if not alias and self.index and source_id in self.index.by_id:
            self.index.verify()
            doc = self.index.by_id[source_id]
            if doc['reference_id'] not in self._allowed_index_references():
                raise ValueError('Source excluded by frozen split/access policy')
            # All metadata comes from the immutable index, never a later corpus query.
            from .corpus import CorpusReference, Region
            ref = CorpusReference.model_validate(doc['reference'])
            original_store = CorpusStore(self.index.manifest['corpus_root'])
            try:
                region = Region.model_validate(doc['region']) if doc.get('region') else None
                frozen = json.loads((self.index.directory / 'documents.json').read_bytes())
                claims = [c for c in frozen['claims'] if c.get('reference_id') == ref.id]
                payload = self._reference_open(ref, original_store, source_id, region, claims)
                payload['relationship_edges'] = doc.get('relationship_edges', [])
            finally:
                original_store.db.close()
        else:
            row = self.db.execute('SELECT payload FROM text_sources WHERE id=?', (source_id,)).fetchone()
            if row:
                payload = json.loads(row[0])
                if self._excluded(source_id, payload['source_family']):
                    raise ValueError('Source excluded by frozen split/access policy')
                if sha256(self.store.asset_path(payload['provenance']['raw_path'])) != payload['provenance']['raw_sha256']:
                    raise ValueError('Context source raw hash mismatch')
                payload['images'] = []
            else:
                candidate = self.db.execute('SELECT payload FROM candidates WHERE id=?', (source_id,)).fetchone()
                if candidate:
                    body = json.loads(candidate[0])
                    family = body.get('source_family') or 'context:' + urllib.parse.urlsplit(body['url']).hostname
                    if self._excluded(source_id, family, body.get('reference_id')):
                        raise ValueError('Source excluded by frozen split/access policy')
                    payload = {**self._candidate_summary(source_id, body), 'images': [],
                               'text': body.get('description', ''), 'links': self._links([body['url']]),
                               'acquisition_required': True}
                else:
                    region = None
                    try:
                        ref = self.store.get_reference(lookup_id)
                    except KeyError:
                        region = self.store.get_region(lookup_id)
                        ref = self.store.get_reference(region.reference_id)
                    payload = self._reference_open(ref, self.store, source_id, region)
        return self._snapshot('open', payload)

    def _acquire_text(self, body):
        policy = self._context_policy(body['url'])
        raw, path, hsh, when = self.fetch(body['url'], kind='context')
        decoded = raw.decode('utf-8', errors='replace')
        parser = _TextParser()
        request_key = digest(dump([body['url'], 'context']).encode())
        info = json.loads(self.db.execute('SELECT payload FROM requests WHERE id=?', (request_key,)).fetchone()[0])
        if info['content_type'] == 'text/plain':
            text = ' '.join(decoded.split())
        else:
            parser.feed(decoded)
            text = ' '.join(' '.join(parser.parts).split())
        urls = []
        for href in parser.links:
            joined = urllib.parse.urldefrag(urllib.parse.urljoin(body['url'], href))[0]
            try:
                self._url(joined)
            except ValueError:
                continue
            urls.append(joined)
        identifier = 'context_' + digest(dump([body['url'], hsh]).encode())[:32]
        payload = {'source_id': identifier, 'kind': 'context', 'title': body['title'],
            'source_family': 'context:' + urllib.parse.urlsplit(body['url']).hostname,
            'url': body['url'], 'text': text[:self.config.max_text_chars],
            'text_truncated': len(text) > self.config.max_text_chars,
            'text_transform': 'HTML parsed as inert text; script/style omitted; whitespace collapsed; bounded UTF-8 replacement decoding',
            'origin': body.get('origin', 'model_selected_observed_link'),
            'author': body.get('author', urllib.parse.urlsplit(body['url']).hostname),
            'rights': policy.rights.model_dump(mode='json'), 'links': self._links(urls),
            'claims': [], 'statement_status': 'attributed_source_text_not_independently_verified',
            'provenance': {'raw_path': str(self.store.asset_path(path)), 'raw_sha256': hsh, 'retrieved_at': when,
                'url': body['url'], 'rights_authorship': 'operator_declared_policy',
                'parent_source_id': body.get('parent_source_id'), 'link_id': body.get('link_id')}}
        with self.db:
            self.db.execute('INSERT OR IGNORE INTO text_sources VALUES (?,?)', (identifier, dump(payload)))
        return identifier

    def acquire(self, candidate_id):
        row = self.db.execute('SELECT payload FROM candidates WHERE id=?', (candidate_id,)).fetchone()
        if not row:
            raise ValueError('Acquisition requires an observed candidate_id')
        body = json.loads(row[0])
        family = body.get('source_family') or 'context:' + urllib.parse.urlsplit(body['url']).hostname
        if self._excluded(candidate_id, family, body.get('reference_id')):
            raise ValueError('Source excluded by frozen split/access policy')
        previous = self.db.execute('SELECT state,payload FROM acquisitions WHERE id=?', (candidate_id,)).fetchone()
        if previous:
            if previous[0] != 'complete':
                raise RuntimeError('Previous acquisition failed or is unsettled; no silent retry')
            result = self.open(json.loads(previous[1])['source_id'])
            return self._snapshot('acquire', {'candidate_id': candidate_id, **{k: v for k, v in result.items() if k not in ('operation', 'snapshot_id', 'snapshot_path', 'snapshot_sha256')}})
        self.db.execute('BEGIN IMMEDIATE')
        try:
            if self.db.execute('SELECT COUNT(*) FROM acquisitions').fetchone()[0] >= self.config.max_acquisitions:
                raise RuntimeError('Research acquisition budget exhausted')
            self.db.execute('INSERT INTO acquisitions VALUES (?,?,?)', (candidate_id, 'started', dump({'at': utcnow()})))
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
        try:
            if body['kind'] == 'context':
                source_id = self._acquire_text(body)
            else:
                raw_path, hsh, _ = body['raw_info']
                if sha256(self.store.asset_path(raw_path)) != hsh:
                    raise ValueError('Candidate metadata hash mismatch')
                # Reuse existing rights, byte preservation, image checks, claims and crop contracts.
                backend = self
                class Adapter:
                    config = SourceConfig(long_edge=backend.config.long_edge, max_request_attempts=1)
                    def fetch(self, *args, **kwargs): return backend.fetch(*args, **kwargs)
                    def check_budget(self, extra=0): return backend.check_budget(extra)
                accepted = ingest_record(self.store, Adapter(), body['provider'], body['data'],
                    body['envelope'], body['raw_info'], body['query'])
                source_id = body['reference_id']
                if not accepted:
                    try:
                        self.store.get_reference(source_id)
                    except KeyError:
                        raise RuntimeError('Source acquisition rejected; failure retained in corpus ledger') from None
                reference = self.store.get_reference(source_id)
                fields, url = normalize(body['provider'], body['data'], body['envelope'], *body['raw_info'], body['query'])
                for field in ('title', 'maker', 'culture', 'date', 'medium', 'dimensions', 'description', 'tags', 'url'):
                    if getattr(reference, field) != fields[field]:
                        raise ValueError('Canonical source version conflict; original admitted reference retained')
                if reference.original.source_url != url:
                    raise ValueError('Canonical source image URL changed; original admitted reference retained')
                if reference.rights != fields['rights']:
                    raise ValueError('Canonical source rights changed; original admitted reference retained')
                # A canonical museum ID can already occur in the frozen index.
                # Give new admissions their own IDs so old encounters never open
                # a different version after a later acquisition.
                if self.index and source_id in self.index.by_id:
                    source_id = 'acquired_' + reference.id + '_' + reference.raw_response_hash[:12]
                    aliases = [(source_id, reference.id)]
                    aliases += [(source_id + '_' + r.id.removeprefix(reference.id + '_'), r.id)
                                for r in self.store.regions(reference.id)]
                    with self.db:
                        for identifier, canonical_id in aliases:
                            self.db.execute('INSERT OR IGNORE INTO source_aliases VALUES (?,?)',
                                (identifier, dump({'canonical_id': canonical_id, 'reference_id': reference.id})))
            with self.db:
                self.db.execute('UPDATE acquisitions SET state=?,payload=? WHERE id=?',
                    ('complete', dump({'source_id': source_id}), candidate_id))
            result = self.open(source_id)
            return self._snapshot('acquire', {'candidate_id': candidate_id, **{k: v for k, v in result.items() if k not in ('operation', 'snapshot_id', 'snapshot_path', 'snapshot_sha256')}})
        except Exception as exc:
            with self.db:
                self.db.execute('UPDATE acquisitions SET state=?,payload=? WHERE id=?',
                    ('failed', dump({'error': f'{type(exc).__name__}: {exc}'}), candidate_id))
            self._event('acquisition_failed', {'candidate_id': candidate_id, 'error': str(exc)})
            raise

    def follow(self, source_id, link_id):
        opened = self.open(source_id)
        link = next((row for row in opened.get('links', []) if row['link_id'] == link_id), None)
        if not link:
            raise ValueError('Follow requires an observed source link_id')
        self._context_policy(link['url'])
        body = {'kind': 'context', 'url': link['url'], 'title': link['url'],
                'origin': 'model_selected_observed_link', 'parent_source_id': source_id,
                'link_id': link_id, 'query': ''}
        result = self.acquire(self._candidate(body))
        return self._snapshot('follow', {'parent_source_id': source_id, 'link_id': link_id,
            **{k: v for k, v in result.items() if k not in ('operation', 'snapshot_id', 'snapshot_path', 'snapshot_sha256')}})

    def summary(self):
        requests, charged = self.db.execute('SELECT COUNT(*),COALESCE(SUM(charged_bytes),0) FROM requests').fetchone()
        states = dict(self.db.execute('SELECT state,COUNT(*) FROM requests GROUP BY state'))
        references = [d['reference'] for d in self.index.documents if d['scope'] == 'whole'] if self.index else []
        references += [r.model_dump(mode='json') for r in self.store.references()]
        references = [r for r in references if not self._excluded(r['id'], r['source_family'])]
        return {'source_identity': self.identity['identity_sha256'], 'index_id': self.identity['index_id'],
            'requests': requests, 'charged_network_bytes': charged, 'request_states': states,
            'acquisitions': self.db.execute('SELECT COUNT(*) FROM acquisitions').fetchone()[0],
            'acquisition_states': dict(self.db.execute('SELECT state,COUNT(*) FROM acquisitions GROUP BY state')),
            'candidates': self.db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
            'context_sources': self.db.execute('SELECT COUNT(*) FROM text_sources').fetchone()[0],
            'coverage_policy': self.config.coverage, 'policy_authorship': self.config.authorship,
            'coverage_observed': {key: dict(Counter(str(row.get(key) or 'not supplied') for row in references))
                                  for key in ('provider', 'culture', 'date', 'medium', 'object_family')},
            'coverage_status': 'descriptive counts; policy targets are not completeness or quality claims',
            'network_enabled': self.config.allow_network, 'frozen_models': True}

    def catalog_plan(self, *, query_limit=None, result_limit=4, providers=None):
        """Explicit operator-authored breadth queries; never an automatic download."""
        queries = self.config.coverage.get('queries', [])
        if query_limit is not None and (type(query_limit) is not int or query_limit < 1):
            raise ValueError('Catalog query limit must be positive')
        if type(result_limit) is not int or not 1 <= result_limit <= 24:
            raise ValueError('Catalog result limit must be 1..24')
        providers = providers or self.config.providers
        if not providers or any(p not in self.config.providers for p in providers) or len(set(providers)) != len(providers):
            raise ValueError('Catalog providers must be distinct enabled providers')
        selected = queries[:query_limit] if query_limit else queries
        if any(not q.strip() or len(q) > 12000 for q in selected):
            raise ValueError('Catalog queries must be bounded nonempty strings')
        searches = [{'query': q, 'provider': p, 'limit': result_limit}
                    for q in selected for p in providers]
        request_upper_bound = sum(1 + result_limit if x['provider'] == 'met' else 1 for x in searches)
        return {'authorship': 'operator_supplied_policy', 'searches': searches,
            'coverage_axes': self.config.coverage, 'metadata_request_upper_bound': request_upper_bound,
            'existing_usage': self.summary(), 'max_requests': self.config.max_requests,
            'max_network_bytes': self.config.max_network_bytes,
            'acquires_images': False, 'network_enabled': self.config.allow_network,
            'policy': 'Explicit standing-collection queries across official providers; limits may yield a partial catalog. '
                      'Configured regions, periods and materials describe intended coverage; catalog matches do not establish achieved coverage.'}

    def populate_catalog(self, *, query_limit=None, result_limit=4, providers=None):
        """Run the declared breadth searches, retaining successes and failures.

        Safe to resume: requests and candidates are reused, failed/uncertain
        requests never repeat. Selection/acquisition is a separate model/operator
        decision, sharing the same persistent source budget.
        """
        plan = self.catalog_plan(query_limit=query_limit, result_limit=result_limit, providers=providers)
        if not plan['searches']:
            raise ValueError('Catalog population requires explicit coverage queries')
        if not self.config.allow_network:
            raise ValueError('Catalog population requires allow_network in the frozen source config')
        results, candidates, failures = [], {}, []
        for search in plan['searches']:
            try:
                result = self.search(search['query'], mode='remote', provider=search['provider'], limit=search['limit'])
                results.append({'query': search['query'], 'provider': search['provider'],
                                'snapshot_id': result['snapshot_id'], 'candidate_count': len(result['sources'])})
                for candidate in result['sources']:
                    candidates[candidate['candidate_id']] = candidate
            except Exception as exc:
                failure = {**search, 'error': f'{type(exc).__name__}: {exc}'}
                self._event('catalog_search_failed', failure)
                failures.append(failure)
                # A failed deadline or exhausted global cap cannot be repaired by
                # dispatching the next provider; preserve the incomplete plan.
                state = self.summary()
                if isinstance(exc, TimeoutError) or 'budget exhausted' in str(exc) or state['requests'] >= self.config.max_requests:
                    break
        return self._snapshot('catalog', {'plan': plan, 'searches': results, 'sources': list(candidates.values()),
            'failures': failures, 'status': 'complete' if len(results) == len(plan['searches']) else 'partial',
            'unattempted_searches': max(0, len(plan['searches']) - len(results) - len(failures)),
            'usage': self.summary(), 'acquires_images': False})


def read_source_summary(root):
    """Read accounting for a planned/running/finished source session without writes.

    ``None`` means no initialized source accounting is available. Incomplete or
    changed frozen configuration is an integrity error, never zero usage. This
    function deliberately does not instantiate CorpusStore or ResearchSources.
    """
    root = Path(root).resolve()
    path = root / 'research_sources.sqlite3'
    if not path.is_file():
        return None
    inputs = root / 'inputs.json'
    if not inputs.is_file():
        raise ValueError('Research source accounting lacks frozen inputs')
    identity = json.loads(inputs.read_bytes())
    if identity.get('identity_sha256') != digest(dump({
            k: v for k, v in identity.items() if k != 'identity_sha256'}).encode()):
        raise ValueError('Research source frozen input identity changed')
    config = ResearchSourceConfig.model_validate(identity['config'])
    # A closed WAL database has no WAL sidecar. SQLite's ordinary read-only open
    # would recreate empty -wal/-shm files; immutable reads avoid that. A live
    # writer's existing WAL must still be read so its current accounting is seen.
    uri = path.as_uri() + ('?mode=ro' if Path(str(path) + '-wal').exists() else '?mode=ro&immutable=1')
    with closing(sqlite3.connect(uri, uri=True)) as db:
        requests, charged = db.execute('SELECT COUNT(*),COALESCE(SUM(charged_bytes),0) FROM requests').fetchone()
        return {'source_identity': identity['identity_sha256'], 'index_id': identity.get('index_id'),
            'requests': requests, 'charged_network_bytes': charged,
            'request_states': dict(db.execute('SELECT state,COUNT(*) FROM requests GROUP BY state')),
            'acquisitions': db.execute('SELECT COUNT(*) FROM acquisitions').fetchone()[0],
            'acquisition_states': dict(db.execute('SELECT state,COUNT(*) FROM acquisitions GROUP BY state')),
            'candidates': db.execute('SELECT COUNT(*) FROM candidates').fetchone()[0],
            'context_sources': db.execute('SELECT COUNT(*) FROM text_sources').fetchone()[0],
            'limits': {key: getattr(config, key) for key in ('max_requests', 'max_network_bytes',
                'max_response_bytes', 'max_image_bytes', 'max_storage_bytes', 'max_acquisitions', 'max_candidates')},
            'remaining_requests': max(0, config.max_requests - requests),
            'remaining_network_bytes': max(0, config.max_network_bytes - charged),
            'coverage_policy': config.coverage, 'policy_authorship': config.authorship,
            'coverage_status': 'Observed source-family coverage is not recomputed by this read-only accounting view',
            'network_enabled': config.allow_network, 'frozen_models': True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('plan', 'catalog', 'acquire', 'show'):
        command = commands.add_parser(name)
        command.add_argument('--root', type=Path, required=True)
        command.add_argument('--config', type=Path)
        command.add_argument('--index', type=Path)
        if name in ('plan', 'catalog'):
            command.add_argument('--query-limit', type=int)
            command.add_argument('--result-limit', type=int, default=4)
            command.add_argument('--provider', action='append', dest='providers')
        if name == 'acquire':
            command.add_argument('--candidate-id', required=True)
        if name == 'show':
            command.add_argument('--source-id')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_bytes()) if args.config else None
    backend = ResearchSources(args.root, config, index_path=args.index)
    try:
        if args.command in ('plan', 'catalog'):
            method = backend.catalog_plan if args.command == 'plan' else backend.populate_catalog
            result = method(query_limit=args.query_limit, result_limit=args.result_limit, providers=args.providers)
        elif args.command == 'acquire':
            result = backend.acquire(args.candidate_id)
        else:
            result = backend.open(args.source_id) if args.source_id else backend.summary()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
