"""Research projections cannot manufacture charged decisions or inspected evidence."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from kusama import research as r
from kusama.ledger import Ledger, SpecMismatch
from kusama.artifacts import sha256
from test_research import setup, run, action, hypothesis, Sources, Director


def replace_saved_step(monkeypatch, *, index, transform):
    """Simulate a faulty projection/recovery before immutable registration."""
    original = r.immutable_json
    def altered(path, value):
        if path.name == f'step-{index:03}.json':
            value = deepcopy(value)
            transform(value)
        return original(path, value)
    monkeypatch.setattr(r, 'immutable_json', altered)


def test_uncharged_registered_commit_cannot_be_reported_or_exported(setup):
    output, _, request, _, _ = setup
    spec = r.plan_research(output, request)
    job_id = 'research_' + request['id']
    with Ledger(output / 'jobs.sqlite3') as ledger:
        ledger.create_job(job_id, spec, request['max_steps'], output)
        projection = {'index': 0, 'action': action('commit', hypothesis=hypothesis()),
            'committed_hypothesis': {**hypothesis(), 'source_snapshots': [], 'origin': 'model_proposal'},
            'seen_images': [], 'status': 'completed'}
        path = output / 'step-000.json'
        path.write_text(json.dumps(projection))
        ledger.register_artifact(job_id, path.name, kind='research_step')
        ledger.update_job(job_id, status='hypotheses_ready')
    with pytest.raises(SpecMismatch):
        r.read_report(output)
    with pytest.raises(SpecMismatch):
        r.export_hypotheses(output, output / 'invalid_export.json')
    assert not (output / 'invalid_export.json').exists()


def test_registered_step_indices_must_be_contiguous(setup, monkeypatch):
    _, _, _, _, actions = setup
    actions.append(action('stop'))
    replace_saved_step(monkeypatch, index=0, transform=lambda row: row.update(index=5))
    with pytest.raises(SpecMismatch, match='contiguous'):
        run(setup)


def test_projection_action_must_match_saved_model_response(setup, monkeypatch):
    _, _, _, _, actions = setup
    actions.append(action('note', note='An actual model note'))
    replace_saved_step(monkeypatch, index=0,
        transform=lambda row: row['action'].update(note='A supplied replacement idea'))
    with pytest.raises(SpecMismatch, match='model output'):
        run(setup)


def test_claimed_seen_images_must_match_saved_decision(setup, monkeypatch):
    _, _, _, _, actions = setup
    actions.append(action('stop'))
    replace_saved_step(monkeypatch, index=0,
        transform=lambda row: row.update(seen_images=[{'source_id': 'never_shown'}]))
    with pytest.raises(SpecMismatch, match='viewed evidence'):
        run(setup)


@pytest.mark.parametrize('field', ['title', 'render_instruction', 'source_snapshots'])
def test_committed_hypothesis_cannot_rewrite_output_or_inspected_sources(setup, monkeypatch, field):
    _, _, _, _, actions = setup
    actions.extend([action('search', query='enclosure'), action('open', source_id='s0'),
        action('draft', hypothesis=hypothesis()), action('commit', hypothesis=hypothesis())])
    replacement = [] if field == 'source_snapshots' else 'Post-hoc replacement'
    replace_saved_step(monkeypatch, index=3,
        transform=lambda row: row['committed_hypothesis'].update({field: replacement}))
    with pytest.raises(SpecMismatch, match='Committed hypothesis differs'):
        run(setup)


def test_saved_follow_requires_link_to_have_been_exposed_to_model(setup):
    output, _, request, calls, actions = setup
    followed = []
    class LinkedSources(Sources):
        def open(self, sid):
            value = super().open(sid)
            value['links'] = [{'link_id': f'l{i}', 'url': f'https://example.org/{i}'} for i in range(30)]
            return value
        def follow(self, source_id, link_id):
            followed.append(link_id)
            return self.open('followed_' + link_id)
    actions.extend([action('search', query='context'), action('open', source_id='s0'),
        action('follow', source_id='s0', link_id='l20'),
        action('read', source_id='s0', link_offset=12),
        action('follow', source_id='s0', link_id='l20'), action('stop')])
    r.plan_research(output, request)
    report = r.run_research(output, director_factory=Director, sources_factory=LinkedSources)
    assert 'Unknown source link' in report['steps'][2]['tool_error']
    assert followed == ['l20']
    assert 'l20' in {row['link_id'] for row in calls[4]['payload']['last_tool_result']['links']}
    assert report['steps'][4]['opened_sources'][0]['source_id'] == 'followed_l20'


def test_terminal_repeated_run_preserves_authoritative_source_usage(setup):
    output, _, _, calls, actions = setup
    actions.append(action('stop'))
    run(setup)
    first = r.read_report(output)
    count = len(calls)
    r.run_research(output, director_factory=Director, sources_factory=Sources)
    second = r.read_report(output)
    assert second['network'] == first['network']
    assert second['network']['requests'] == 0
    assert len(calls) == count


def test_interrupted_source_registration_cannot_hide_missing_original_bytes(setup, monkeypatch):
    output, _, request, _, actions = setup
    class ByteSources(Sources):
        def open(self, source_id):
            value = super().open(source_id)
            raw = self.root / 'raw_metadata.bin'
            raw.write_bytes(b'Exact fixture source metadata')
            value['provenance'].update(raw_path=str(raw), raw_sha256=sha256(raw))
            return value
    original = Ledger.register_artifact
    def interrupted(self, *args, **kwargs):
        if kwargs.get('kind') == 'research_source_bytes':
            raise RuntimeError('Fixture interruption during byte registration')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Ledger, 'register_artifact', interrupted)
    actions.extend([action('search', query='enclosure'), action('open', source_id='s0')])
    r.plan_research(output, request)
    with pytest.raises(RuntimeError, match='interruption'):
        r.run_research(output, director_factory=Director, sources_factory=ByteSources)
    row = json.loads((output / 'step-001.json').read_bytes())
    evidence = row['opened_sources'][0]['byte_evidence'][0]
    (output / evidence['path']).unlink()
    with pytest.raises(SpecMismatch, match='byte|source|artifact|evidence'):
        r.read_report(output)
