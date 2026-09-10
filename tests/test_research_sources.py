"""Offline source encounters test provenance and controls, not artistic discovery."""
import json
from pathlib import Path
import sqlite3
import urllib.error
import urllib.parse

import pytest

from kusama.artifacts import sha256
from kusama.corpus import digest
from kusama.research_sources import ResearchSourceConfig, ResearchSources
from test_corpus import cleveland, png
from test_retrieval import tiny_index


class Response:
    def __init__(self, raw, content_type='application/json'):
        self.raw = raw
        self.headers = {'Content-Type': content_type}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, limit): return self.raw[:limit]


class CollectionTransport:
    def __init__(self): self.urls = []
    def __call__(self, request, timeout):
        self.urls.append(request.full_url)
        if request.full_url.startswith('https://openaccess-api.clevelandart.org/'):
            return Response(json.dumps({'data': [cleveland()]}).encode())
        if request.full_url.startswith('https://openaccess-cdn.clevelandart.org/'):
            return Response(png(), 'image/png')
        raise AssertionError(request.full_url)


def context_config(**kwargs):
    return {'allow_network': True, 'context_seeds': [
        {'url': 'https://history.example.edu/allowed/essay', 'title': 'Historical enclosure essay',
         'author': 'Fixture author', 'description': 'Contradictory readings of boundaries'}],
        'context_policies': [{'host': 'history.example.edu', 'path_prefix': '/allowed', 'rights': {
            'status': 'allowed', 'license': 'CC-BY-4.0', 'evidence_url': 'https://history.example.edu/license',
            'evidence_field': 'Fixture supplied license', 'attribution': 'Fixture author'}}], **kwargs}


def test_local_modes_snapshot_images_and_frozen_context(tiny_index, tmp_path):
    index, store, _ = tiny_index
    backend = ResearchSources(tmp_path / 'research', index_path=index.directory)
    result = backend.search('enclosure', limit=4)
    assert result['sources']
    assert result['source_identity'] == backend.identity['identity_sha256']
    assert sha256((backend.root / result['snapshot_path'])) == result['snapshot_sha256']
    opened = backend.open(result['sources'][0]['source_id'])
    assert opened['reference']['context_trust'] == 'untrusted_source_data'
    assert all(Path(x['path']).is_absolute() and sha256(Path(x['path'])) == x['sha256'] for x in opened['images'])
    detail = next(d for d in index.documents if d['scope'] == 'part')
    assert len(backend.open(detail['id'])['images']) == 2
    assert backend.search('xyzabcdefg')['sources'] == []
    assert backend.search('encloses', mode='relationship')['sources'] == []  # no invented edges
    for mode in ('contextual', 'historical', 'contradiction'):
        found = backend.search('enclosure', mode=mode)
        assert found['sources'] and found['human_usefulness'] is None
    assert backend.summary()['requests'] == 0


def test_visual_uses_frozen_vectors_and_exploration_is_attributed(tiny_index, tmp_path):
    index, _, _ = tiny_index
    backend = ResearchSources(tmp_path / 'research', index_path=index.directory)
    visual = backend.search('visual comparison', mode='visual', source_id='cleveland_1')
    assert visual['retrieval']['visual_available']
    assert visual['sources'][0]['source_id'] == 'cleveland_1'
    first = backend.search('encounter', mode='exploratory', seed=71)
    second = backend.search('encounter', mode='exploratory', seed=71)
    assert first['sources'] == second['sources']
    assert 'operator-authored' in first['policy']
    with pytest.raises(ValueError, match='CLIP'):
        backend.search('something', mode='visual')


def test_remote_search_is_metadata_first_and_acquire_preserves_originals(tmp_path):
    transport = CollectionTransport()
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport)
    result = backend.search('vessel', mode='remote', provider='cleveland')
    assert len(transport.urls) == 1 and not backend.store.references()
    candidate = result['sources'][0]['candidate_id']
    assert backend.open(candidate)['acquisition_required']
    acquired = backend.acquire(candidate)
    assert acquired['source_id'] == 'cleveland_1'
    assert len(transport.urls) == 2
    assert Path(acquired['provenance']['original_path']).read_bytes() == png()
    assert acquired['reference']['rights']['training']['status'] == 'allowed'
    assert acquired['claims'][0]['status'] == 'documented_source_statement'
    assert backend.search('vessel')['sources'][0]['source_id'] == 'cleveland_1'
    assert backend.acquire(candidate)['source_id'] == acquired['source_id']
    assert len(transport.urls) == 2
    assert backend.summary()['acquisitions'] == 1
    with pytest.raises(ValueError, match='observed candidate'):
        backend.acquire('https://unseen.example/object')


def test_resume_freezes_config_index_and_reuses_network_evidence(tmp_path):
    transport = CollectionTransport()
    config = {'allow_network': True}
    backend = ResearchSources(tmp_path, config, opener=transport)
    first = backend.search('vessel', mode='remote', provider='cleveland')
    backend.close()
    resumed = ResearchSources(tmp_path, config, opener=transport)
    second = resumed.search('vessel', mode='remote', provider='cleveland')
    assert first['snapshot_id'] == second['snapshot_id'] and len(transport.urls) == 1
    with pytest.raises(ValueError, match='inputs changed'):
        ResearchSources(tmp_path, {'allow_network': False}, opener=transport)
    resumed.close()


def test_request_caps_and_failures_survive_restarts(tmp_path):
    calls = []
    def failing(request, timeout):
        calls.append(request.full_url)
        raise TimeoutError('fixture uncertain request')
    config = {'allow_network': True, 'max_requests': 1}
    backend = ResearchSources(tmp_path, config, opener=failing)
    with pytest.raises(TimeoutError):
        backend.search('vessel', mode='remote', provider='cleveland')
    assert backend.summary()['charged_network_bytes'] == ResearchSourceConfig().max_response_bytes + 1
    backend.close()
    resumed = ResearchSources(tmp_path, config, opener=failing)
    with pytest.raises(RuntimeError, match='no silent retry'):
        resumed.search('vessel', mode='remote', provider='cleveland')
    with pytest.raises(RuntimeError, match='budget'):
        resumed.search('different', mode='remote', provider='cleveland')
    assert len(calls) == 1 and resumed.summary()['request_states'] == {'failed': 1}


def test_network_and_acquisition_caps_cannot_be_bypassed(tmp_path):
    with pytest.raises(ValueError, match='disabled'):
        ResearchSources(tmp_path / 'disabled').search('vessel', mode='remote', provider='cleveland')
    transport = CollectionTransport()
    backend = ResearchSources(tmp_path / 'limited', {'allow_network': True, 'max_acquisitions': 0}, opener=transport)
    candidate = backend.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    with pytest.raises(RuntimeError, match='acquisition budget'):
        backend.acquire(candidate)
    limited = ResearchSources(tmp_path / 'bytes', {'allow_network': True, 'max_network_bytes': 1024}, opener=transport)
    with pytest.raises(RuntimeError, match='byte budget'):
        limited.search('vessel', mode='remote', provider='cleveland')


def test_context_sources_preserve_bytes_and_observed_links_require_rights(tmp_path):
    raw = b'<html><script>do not execute me</script><h1>Enclosure</h1><p>A reading can be contested.</p><a href="/allowed/other">An alternative</a><a href="/private/secrets">No rights</a></html>'
    calls = []
    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(raw if request.full_url.endswith('essay') else b'Alternative source text.', 'text/html')
    backend = ResearchSources(tmp_path, context_config(), opener=opener)
    candidate = backend.search('historical', mode='historical')['sources'][0]['candidate_id']
    acquired = backend.acquire(candidate)
    assert acquired['kind'] == 'context' and acquired['images'] == []
    assert 'do not execute' not in acquired['text']
    assert backend.store.asset_path(acquired['provenance']['raw_path']).read_bytes() == raw
    assert acquired['statement_status'] == 'attributed_source_text_not_independently_verified'
    assert backend.search('contested', mode='contradiction')['sources'][0]['source_id'] == acquired['source_id']
    allowed, excluded = acquired['links']
    assert allowed['follow_allowed'] and not excluded['follow_allowed']
    followed = backend.follow(acquired['source_id'], allowed['link_id'])
    assert followed['provenance']['parent_source_id'] == acquired['source_id']
    with pytest.raises(ValueError, match='text rights'):
        backend.follow(acquired['source_id'], excluded['link_id'])
    with pytest.raises(ValueError, match='observed'):
        backend.follow(acquired['source_id'], 'invented_link')
    assert len(calls) == 2


def test_plain_context_keeps_literal_markup_and_bounds_excerpt(tmp_path):
    raw = ('<artifact> enclosure ' * 100).encode()
    backend = ResearchSources(tmp_path, context_config(max_text_chars=256),
                              opener=lambda *args, **kwargs: Response(raw, 'text/plain'))
    cid = backend.search('enclosure')['sources'][0]['candidate_id']
    result = backend.acquire(cid)
    assert result['text'].startswith('<artifact>') and result['text_truncated']
    assert len(result['text']) == 256
    assert digest(raw) == result['provenance']['raw_sha256']


def test_disallowed_hosts_and_rights_cannot_be_smuggled_through_provider_record(tmp_path):
    record = cleveland()
    record['images']['web']['url'] = 'https://127.0.0.1/admin'
    calls = []
    def opener(request, timeout):
        calls.append(request.full_url)
        return Response(json.dumps({'data': [record]}).encode())
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=opener)
    cid = backend.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    with pytest.raises(RuntimeError, match='rejected'):
        backend.acquire(cid)
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match='no silent retry'):
        backend.acquire(cid)
    with pytest.raises(ValueError, match='text rights'):
        ResearchSources(tmp_path / 'unsafe', context_config(context_policies=[]))


def test_excluded_image_records_do_not_become_candidates(tmp_path):
    record = cleveland()
    record['share_license_status'] = 'unknown'
    backend = ResearchSources(tmp_path, {'allow_network': True},
        opener=lambda *args, **kwargs: Response(json.dumps({'data': [record]}).encode()))
    result = backend.search('vessel', mode='remote', provider='cleveland')
    assert result['sources'] == [] and result['exclusions']


def test_raw_asset_snapshot_tampering_and_mutations_are_rejected(tmp_path):
    transport = CollectionTransport()
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport)
    found = backend.search('vessel', mode='remote', provider='cleveland')
    (backend.root / found['snapshot_path']).write_bytes(b'changed snapshot')
    with pytest.raises(ValueError, match='snapshot hash'):
        backend.search('vessel', mode='remote', provider='cleveland')
    cid = found['sources'][0]['candidate_id']
    acquired = backend.acquire(cid)
    Path(acquired['images'][0]['path']).write_bytes(b'changed image')
    with pytest.raises(ValueError, match='asset hash'):
        backend.open(acquired['source_id'])
    with pytest.raises(sqlite3.IntegrityError, match='immutable'):
        backend.db.execute('DELETE FROM candidates')


def test_met_only_fetches_bounded_metadata_records(tmp_path):
    calls = []
    def opener(request, timeout):
        calls.append(request.full_url)
        if '/search?' in request.full_url:
            return Response(json.dumps({'objectIDs': list(range(1, 100))}).encode())
        oid = int(request.full_url.rsplit('/', 1)[1])
        return Response(json.dumps({'objectID': oid, 'title': 'Fixture painting',
            'objectURL': f'https://www.metmuseum.org/art/collection/search/{oid}',
            'primaryImageSmall': f'https://images.metmuseum.org/{oid}.jpg',
            'isPublicDomain': True}).encode())
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=opener)
    result = backend.search('painting', mode='remote', provider='met', limit=2)
    assert len(result['sources']) == 2 and len(calls) == 3
    assert not backend.store.references()


def test_aic_record_uses_api_iiif_provenance(tmp_path):
    envelope = {'config': {'iiif_url': 'https://www.artic.edu/iiif/2'}, 'data': [{
        'id': 1, 'title': 'Fixture painting', 'image_id': 'fixture_image', 'is_public_domain': True,
        'api_link': 'https://api.artic.edu/api/v1/artworks/1', 'description': 'Museum context.'}]}
    def opener(request, timeout):
        return Response(png(), 'image/png') if '/iiif/' in request.full_url else Response(json.dumps(envelope).encode())
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=opener)
    cid = backend.search('painting', mode='remote', provider='artic')['sources'][0]['candidate_id']
    result = backend.acquire(cid)
    assert result['reference']['view']['source_url'] == 'https://www.artic.edu/iiif/2/fixture_image/full/843,/0/default.jpg'
    assert result['reference']['rights']['descriptive_text']['license'] == 'CC-BY-4.0'


def test_oversize_response_retained_and_counted_without_retry(tmp_path):
    raw = b'x' * 2048
    backend = ResearchSources(tmp_path, {'allow_network': True, 'max_response_bytes': 1024},
                              opener=lambda *args, **kwargs: Response(raw))
    with pytest.raises(ValueError, match='byte limit'):
        backend.search('vessel', mode='remote', provider='cleveland')
    assert backend.summary()['charged_network_bytes'] == 1025
    assert len(list((backend.store.root / 'failures').glob('*.bin'))) == 1
    assert next((backend.store.root / 'failures').glob('*.bin')).stat().st_size == 1025


def test_non_text_context_is_retained_but_never_opened_as_text(tmp_path):
    backend = ResearchSources(tmp_path, context_config(),
        opener=lambda *args, **kwargs: Response(b'%PDF-fixture', 'application/pdf'))
    cid = backend.search('historical')['sources'][0]['candidate_id']
    with pytest.raises(ValueError, match='HTML or plain text'):
        backend.acquire(cid)
    assert backend.summary()['context_sources'] == 0
    assert list((backend.store.root / 'failures').glob('*.bin'))


def test_source_exclusions_cover_search_open_visual_queries_and_descendants(tiny_index, tmp_path):
    index, _, _ = tiny_index
    config = {'excluded_source_families': ['same_source_family']}
    backend = ResearchSources(tmp_path / 'family', config, index_path=index.directory)
    result = backend.search('enclosure')
    assert {row['reference_id'] for row in result['sources']} == {'cleveland_3'}
    assert all(index.by_id[i]['reference_id'] == 'cleveland_3'
        for channel in result['retrieval']['channels'].values() for i in channel)
    with pytest.raises(ValueError, match='excluded'):
        backend.open('cleveland_1')
    with pytest.raises(ValueError, match='excluded'):
        backend.search('shape', mode='visual', source_id='cleveland_1')
    part = next(d for d in index.documents if d['reference_id'] == 'cleveland_3' and d['scope'] == 'part')
    partial = ResearchSources(tmp_path / 'part', {'excluded_source_ids': [part['id']]}, index_path=index.directory)
    assert all(row['reference_id'] != 'cleveland_3' for row in partial.search('enclosure')['sources'])
    with pytest.raises(ValueError, match='excluded'):
        partial.open('cleveland_3')


def test_remote_exclusion_prevents_candidate_and_acquisition(tmp_path):
    backend = ResearchSources(tmp_path, {'allow_network': True, 'excluded_source_ids': ['cleveland_1']}, opener=CollectionTransport())
    result = backend.search('vessel', mode='remote', provider='cleveland')
    assert not result['sources'] and 'excluded' in result['exclusions'][0]['reason']
    context = ResearchSources(tmp_path / 'context', context_config(
        excluded_source_families=['context:history.example.edu']))
    assert not context.search('historical')['sources']
    cid = context.db.execute('SELECT id FROM candidates').fetchone()[0]
    with pytest.raises(ValueError, match='excluded'):
        context.acquire(cid)
    with pytest.raises(ValueError, match='excluded'):
        context.open(cid)


def test_before_fetch_deadline_is_checked_for_each_uncached_request(tmp_path):
    transport = CollectionTransport()
    count = []
    def deadline():
        count.append(1)
        if len(count) > 1:
            raise TimeoutError('research deadline')
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=transport, before_fetch=deadline)
    cid = backend.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    backend.search('vessel', mode='remote', provider='cleveland')  # cached metadata; no request
    with pytest.raises(RuntimeError, match='rejected'):
        backend.acquire(cid)
    assert len(count) == 2 and len(transport.urls) == 1
    state = backend.db.execute("SELECT charged_bytes,payload FROM requests WHERE state='failed'").fetchone()
    assert state[0] == 0 and json.loads(state[1])['dispatched'] is False


def test_acquired_visual_query_checks_index_revision_and_cpu(tiny_index, tmp_path):
    import numpy as np
    index, _, _ = tiny_index
    transport = CollectionTransport()
    class Encoder:
        metadata = {'revision': index.manifest['encoder']['revision'], 'device': 'cpu'}
        def images(self, paths):
            assert len(paths) == 1 and Path(paths[0]).exists()
            return np.eye(1, 512)
        def close(self): pass
    backend = ResearchSources(tmp_path / 'research', {'allow_network': True}, index_path=index.directory,
                              opener=transport, encoder=Encoder())
    # Fixture source 1 already exists in the index, so admit a distinct ID.
    transport_record = cleveland(7)
    backend.opener = lambda request, timeout: (Response(png('purple'), 'image/png') if '.jpg' in request.full_url
        else Response(json.dumps({'data': [transport_record]}).encode()))
    cid = backend.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    source = backend.acquire(cid)['source_id']
    result = backend.search('shape', mode='visual', source_id=source)
    assert result['retrieval']['clip_admission']['mode'] == 'acquired_image_to_frozen_index'
    backend.encoder.metadata = {'revision': 'changed', 'device': 'cpu'}
    with pytest.raises(ValueError, match='revision'):
        backend.search('shape', mode='visual', source_id=source)
    backend.encoder.metadata = {'revision': index.manifest['encoder']['revision'], 'device': 'mps'}
    with pytest.raises(ValueError, match='CPU'):
        backend.search('shape', mode='visual', source_id=source)


def test_standing_catalog_executes_only_configured_bounded_metadata_searches(tmp_path):
    transport = CollectionTransport()
    config = {'allow_network': True, 'coverage': {'queries': ['vessel', 'textile'],
        'regions': ['documented region'], 'periods': ['documented period'], 'materials': ['stone']}}
    backend = ResearchSources(tmp_path, config, opener=transport)
    plan = backend.catalog_plan(providers=['cleveland'], result_limit=2)
    assert len(plan['searches']) == 2 and plan['metadata_request_upper_bound'] == 2
    assert not transport.urls
    result = backend.populate_catalog(providers=['cleveland'], result_limit=2)
    assert result['status'] == 'complete' and len(transport.urls) == 2
    assert not backend.store.references() and not result['acquires_images']
    repeated = backend.populate_catalog(providers=['cleveland'], result_limit=2)
    assert len(transport.urls) == 2 and repeated['sources'] == result['sources']
    assert result['plan']['authorship'] == 'operator_supplied_policy'


def test_standing_catalog_retains_partial_failure_under_shared_cap(tmp_path):
    backend = ResearchSources(tmp_path, {'allow_network': True, 'max_requests': 1,
        'coverage': {'queries': ['vessel', 'textile', 'painting']}}, opener=CollectionTransport())
    result = backend.populate_catalog(providers=['cleveland'])
    assert result['status'] == 'partial' and len(result['searches']) == 1
    assert len(result['failures']) == 1 and result['unattempted_searches'] == 1
    assert backend.summary()['requests'] == 1


def test_source_cli_plan_is_executable_offline(tmp_path, capsys):
    from kusama.research_sources import main
    config = tmp_path / 'source_config.json'
    config.write_text(json.dumps({'coverage': {'queries': ['portrait', 'instrument']}}))
    assert main(['plan', '--root', str(tmp_path / 'sources'), '--config', str(config), '--query-limit', '1']) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result['searches']) == 3 and result['existing_usage']['requests'] == 0
    assert result['network_enabled'] is False


def test_index_and_new_admission_with_same_canonical_id_remain_distinct(tiny_index, tmp_path):
    index, _, _ = tiny_index
    backend = ResearchSources(tmp_path / 'research', {'allow_network': True}, index_path=index.directory,
                              opener=CollectionTransport())
    old = backend.open('cleveland_1')
    cid = backend.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    acquired = backend.acquire(cid)
    assert acquired['source_id'].startswith('acquired_cleveland_1_')
    assert old['reference']['title'] != acquired['reference']['title']
    assert backend.open('cleveland_1')['reference'] == old['reference']
    assert backend.open(acquired['source_id'])['reference'] == acquired['reference']
    assert acquired['details']
    detail = backend.open(acquired['details'][0]['source_id'])
    assert len(detail['images']) == 2 and detail['images'][-1]['scope'] == 'part'
    assert detail['source_id'].startswith('acquired_')


def test_read_source_summary_preserves_durable_counts_without_writes(tmp_path):
    from kusama.research_sources import read_source_summary
    missing = tmp_path / 'missing'
    assert read_source_summary(missing) is None and not missing.exists()
    root = tmp_path / 'research'
    backend = ResearchSources(root, {'allow_network': True}, opener=CollectionTransport())
    cid = backend.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    backend.acquire(cid)
    expected = backend.summary()
    backend.close()
    before = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}
    summary = read_source_summary(root)
    after = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}
    assert before == after
    for key in ('source_identity', 'requests', 'charged_network_bytes', 'request_states',
                'acquisitions', 'candidates', 'context_sources', 'network_enabled'):
        assert summary[key] == expected[key]
    assert summary['acquisition_states'] == {'complete': 1}
    assert summary['remaining_requests'] == summary['limits']['max_requests'] - 2


def test_read_source_summary_distinguishes_missing_inputs_from_zero_usage(tmp_path):
    from kusama.research_sources import read_source_summary
    backend = ResearchSources(tmp_path)
    backend.close()
    (tmp_path / 'inputs.json').unlink()
    with pytest.raises(ValueError, match='lacks frozen inputs'):
        read_source_summary(tmp_path)


def test_read_source_summary_sees_live_wal_counts(tmp_path):
    from kusama.research_sources import read_source_summary
    backend = ResearchSources(tmp_path, {'allow_network': True}, opener=CollectionTransport())
    backend.search('vessel', mode='remote', provider='cleveland')
    before = {str(p.relative_to(tmp_path)): sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    summary = read_source_summary(tmp_path)
    after = {str(p.relative_to(tmp_path)): sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    assert summary['requests'] == 1 and summary['candidates'] == 1
    assert before.keys() == after.keys()
    # SQLite read-only live WAL access updates existing shared-memory read marks;
    # the authoritative DB/WAL and every evidence file must remain unchanged.
    assert {k: v for k, v in before.items() if not k.endswith('-shm')} == {
        k: v for k, v in after.items() if not k.endswith('-shm')}
    backend.close()


def test_standing_catalog_candidates_feed_new_episode_without_image_downloads(tmp_path):
    catalog_path = tmp_path / 'standing'
    transport = CollectionTransport()
    catalog = ResearchSources(catalog_path, {'allow_network': True}, opener=transport)
    candidate = catalog.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    catalog.close()
    episode_transport = CollectionTransport()
    config = {'allow_network': True, 'catalog_path': str(catalog_path), 'catalog_limit': 2}
    episode = ResearchSources(tmp_path / 'episode', config, opener=episode_transport)
    assert not episode_transport.urls and not episode.store.references()
    found = episode.search('vessel', mode='historical')['sources'][0]
    assert found['candidate_id'] == candidate and found['catalog_import']
    assert episode.open(candidate)['acquisition_required']
    admitted = episode.acquire(candidate)
    assert admitted['source_id'] == 'cleveland_1'
    assert len(episode_transport.urls) == 1 and 'openaccess-cdn' in episode_transport.urls[0]
    assert episode.summary()['requests'] == 1
    assert Path(admitted['provenance']['raw_path']).is_relative_to(episode.root)
    episode.close()


def test_episode_reuses_frozen_catalog_after_standing_growth_or_removal(tmp_path):
    catalog_path = tmp_path / 'standing'
    catalog = ResearchSources(catalog_path, {'allow_network': True}, opener=CollectionTransport())
    first = catalog.search('vessel', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    config = {'catalog_path': str(catalog_path)}
    episode = ResearchSources(tmp_path / 'episode', config)
    identity = episode.identity
    episode.close()
    second = catalog.search('textile', mode='remote', provider='cleveland')['sources'][0]['candidate_id']
    assert first != second
    catalog.close()
    catalog_path.rename(tmp_path / 'moved_standing')
    resumed = ResearchSources(tmp_path / 'episode', config)
    assert resumed.identity == identity
    assert resumed.summary()['candidates'] == 1
    assert resumed.search('vessel')['sources'][0]['candidate_id'] == first
    assert not resumed.search('textile')['sources']


def test_catalog_limits_exclusions_and_import_corruption_are_enforced(tmp_path):
    catalog = ResearchSources(tmp_path / 'standing', {'allow_network': True}, opener=CollectionTransport())
    catalog.search('vessel', mode='remote', provider='cleveland')
    catalog.search('textile', mode='remote', provider='cleveland')
    catalog.close()
    config = {'catalog_path': str(tmp_path / 'standing'), 'catalog_limit': 1}
    episode = ResearchSources(tmp_path / 'episode', config)
    assert episode.summary()['candidates'] == 1
    assert episode.catalog_import['scanned_candidates'] == 1
    episode.close()
    path = tmp_path / 'episode' / 'catalog_import.json'
    value = json.loads(path.read_bytes())
    value['candidates'][0]['body']['title'] = 'Rewritten history'
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='catalog import hash'):
        ResearchSources(tmp_path / 'episode', config)
    excluded = ResearchSources(tmp_path / 'excluded', {**config, 'excluded_source_ids': ['cleveland_1']})
    assert excluded.summary()['candidates'] == 0 and excluded.catalog_import['exclusions']


def test_frozen_catalog_metadata_bytes_are_revalidated_on_resume(tmp_path):
    catalog = ResearchSources(tmp_path / 'standing', {'allow_network': True}, opener=CollectionTransport())
    catalog.search('vessel', mode='remote', provider='cleveland')
    catalog.close()
    config = {'catalog_path': str(tmp_path / 'standing')}
    episode = ResearchSources(tmp_path / 'episode', config)
    metadata = episode.catalog_import['metadata'][0]
    path = episode.store.asset_path(metadata['path'])
    episode.close()
    path.write_bytes(b'changed source response')
    with pytest.raises(ValueError, match='metadata hash'):
        ResearchSources(tmp_path / 'episode', config)
