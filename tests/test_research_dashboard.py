"""Read-only discovery evidence and independent HTTP containment checks."""
import json
from pathlib import Path
import shutil
import subprocess

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.dashboard import create_app
from kusama.ledger import SpecMismatch


def client(root):
    return TestClient(create_app(root), base_url='http://127.0.0.1')


def saved_research(root, monkeypatch, relative='research/first'):
    """An HTTP-layer fixture; engine validation is tested separately."""
    from kusama import research

    directory = root / relative
    directory.mkdir(parents=True)
    image = directory / 'evidence' / 'source.png'
    image.parent.mkdir()
    Image.new('RGB', (32, 32), 'green').save(image)
    plan = {'kind': 'autonomous_research', 'request': {'id': 'first',
            'problem': 'Investigate the tension between enclosure and protection.',
            'authorship': 'operator'}, 'policy': {'source_pairing': 'model_selected'}}
    report = {'kind': 'autonomous_research', 'id': 'first', 'problem': plan['request']['problem'],
              'authorship': 'operator', 'status': 'completed', 'fixture': True,
              'reasoning_calls': 0, 'steps': [{'index': 0, 'action': {'kind': 'search',
              'query': 'enclosure', 'origin': 'fixture_actor'}, 'tool_result': {'ids': ['one']}}],
              'hypotheses': [{'id': 'h1', 'sources': [{'source_id': 'one', 'role': 'enclosure'}],
                             'origin': 'fixture_actor'}],
              'notebook': [{'status': 'rejected', 'reason': 'No visual relationship found.'}],
              'selected_sources': ['one'], 'source_history': [], 'human_labels': 0,
              'training_performed': False, 'evidence_images': [{'source_id': 'one', 'scope': 'whole',
              'path': str(image.relative_to(directory)), 'sha256': sha256(image),
              'rights': {'image_display': {'status': 'allowed'}}}]}
    (directory / 'plan.json').write_text(json.dumps(plan))
    (directory / 'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(research, 'read_report', lambda path: json.loads((path / 'report.json').read_text()))
    return directory, report


def save_report(directory, report):
    (directory / 'report.json').write_text(json.dumps(report))


def evidence_url(app):
    item = app.get('/api/research').json()['research'][0]
    response = app.get('/api/research/' + item['directory_id'])
    assert response.status_code == 200
    detail = response.json()
    return detail['evidence_images'][0]['url'], detail


def test_research_reads_empty_root_without_creating_files(tmp_path):
    root = tmp_path / 'absent'
    with client(root) as app:
        assert app.get('/api/research').json() == {'research': []}
        assert app.get('/api/research/unknown').status_code == 404
        assert app.post('/api/research', json={}).status_code == 405
    assert not root.exists()


def test_research_lists_nested_sessions_and_preserves_attribution(tmp_path, monkeypatch):
    directory, report = saved_research(tmp_path, monkeypatch, 'development/study/research/round-01')
    before = {str(path.relative_to(tmp_path)): path.read_bytes()
              for path in tmp_path.rglob('*') if path.is_file()}
    with client(tmp_path) as app:
        rows = app.get('/api/research').json()['research']
        assert len(rows) == 1
        assert rows[0]['directory'] == 'development/study/research/round-01'
        assert rows[0]['fixture'] and rows[0]['steps'] == 1 and rows[0]['hypotheses'] == 1
        url, detail = evidence_url(app)
        assert detail['frozen_plan']['policy']['source_pairing'] == 'model_selected'
        assert detail['steps'] == report['steps']
        assert detail['notebook'][0]['status'] == 'rejected'
        assert detail['human_labels'] == 0 and detail['training_performed'] is False
        assert app.get(url).content == (directory / 'evidence/source.png').read_bytes()
        assert app.post('/api/research/' + rows[0]['directory_id'] + '/execute', json={}).status_code == 404
    after = {str(path.relative_to(tmp_path)): path.read_bytes()
             for path in tmp_path.rglob('*') if path.is_file()}
    assert after == before
    assert not list(tmp_path.rglob('*.sqlite3'))


def test_research_image_hash_changes_are_conflicts(tmp_path, monkeypatch):
    directory, _ = saved_research(tmp_path, monkeypatch)
    with client(tmp_path) as app:
        url, _ = evidence_url(app)
        (directory / 'evidence/source.png').write_bytes(b'changed')
        assert app.get(url).status_code == 409


@pytest.mark.parametrize('escape', ['absolute', 'relative', 'symlink', 'directory_symlink'])
def test_research_images_never_escape_session(tmp_path, monkeypatch, escape):
    root = tmp_path / 'artifacts'
    directory, report = saved_research(root, monkeypatch)
    secret = tmp_path / 'secret.png'
    Image.new('RGB', (16, 16), 'red').save(secret)
    asset = report['evidence_images'][0]
    asset['sha256'] = sha256(secret)
    if escape == 'absolute':
        asset['path'] = str(secret)
    elif escape == 'relative':
        asset['path'] = '../../../secret.png'
    elif escape == 'symlink':
        (directory / 'linked.png').symlink_to(secret)
        asset['path'] = 'linked.png'
    else:
        (directory / 'linked').symlink_to(tmp_path, target_is_directory=True)
        asset['path'] = 'linked/secret.png'
    save_report(directory, report)
    with client(root) as app:
        url, _ = evidence_url(app)
        assert app.get(url).status_code == 404


def test_research_discovery_excludes_symlinked_and_unrelated_plans(tmp_path, monkeypatch):
    root = tmp_path / 'artifacts'
    saved_research(root, monkeypatch)
    outside = tmp_path / 'outside'; outside.mkdir()
    (outside / 'plan.json').write_text(json.dumps({'kind': 'autonomous_research'}))
    (root / 'linked').symlink_to(outside, target_is_directory=True)
    linked_file = root / 'linked_file'; linked_file.mkdir()
    (linked_file / 'plan.json').symlink_to(outside / 'plan.json')
    unrelated = root / 'unrelated'; unrelated.mkdir()
    (unrelated / 'plan.json').write_text(json.dumps({'kind': 'different_experiment'}))
    hidden = root / '.private'; hidden.mkdir()
    (hidden / 'plan.json').write_text(json.dumps({'kind': 'autonomous_research'}))
    with client(root) as app:
        assert len(app.get('/api/research').json()['research']) == 1
        assert app.get('/api/research/..%2Foutside').status_code == 404


@pytest.mark.parametrize('rights,status', [({}, 403), ({'image_display': {'status': 'excluded'}}, 403),
                                         ({'status': 'allowed', 'basis': 'operator_supplied_local_parent'}, 200)])
def test_research_images_enforce_saved_display_rights(tmp_path, monkeypatch, rights, status):
    directory, report = saved_research(tmp_path, monkeypatch)
    report['evidence_images'][0]['rights'] = rights
    save_report(directory, report)
    with client(tmp_path) as app:
        url, _ = evidence_url(app)
        assert app.get(url).status_code == status


def test_research_verification_failure_stays_visible_without_serving_evidence(tmp_path, monkeypatch):
    from kusama import research
    saved_research(tmp_path, monkeypatch)

    def invalid(path):
        raise SpecMismatch('Frozen plan changed')

    monkeypatch.setattr(research, 'read_report', invalid)
    with client(tmp_path) as app:
        item = app.get('/api/research').json()['research'][0]
        assert item['status'] == 'unreadable'
        assert 'Frozen plan changed' in item['error']
        assert app.get('/api/research/' + item['directory_id']).status_code == 409


def test_transcript_projection_marks_missing_and_symlinked_records(tmp_path):
    from kusama.run_inspection import research_transcripts
    directory = tmp_path / 'research'; directory.mkdir()
    outside = tmp_path / 'outside.json'; outside.write_text('{"system":"Outside text"}')
    (directory / 'decision-000.json').symlink_to(outside)
    result = research_transcripts(directory, {'steps': [{'index': 0}, {'index': 1}]})
    assert 'prompt' not in result[0]
    assert 'contained' in result[0]['prompt_error']
    assert 'response' not in result[0]
    assert 'unavailable' in result[1]['prompt_error']
    assert 'Outside text' not in json.dumps(result)


def test_research_cli_routes_to_bounded_engine(monkeypatch, capsys):
    from kusama import cli, research
    received = []
    monkeypatch.setattr(research, 'main', lambda args: received.append(args) or 7)
    assert cli.main(['research', 'show', '--output', 'a-session']) == 7
    assert received == [['show', '--output', 'a-session']]
    assert cli.main(['--help']) == 0
    assert 'research' in capsys.readouterr().out


def planned_session(tmp_path):
    from kusama.research import plan_research
    parent = tmp_path / 'parent.png'
    Image.new('RGB', (32, 32), 'blue').save(parent)
    root = tmp_path / 'artifacts'
    directory = root / 'research' / 'fixture'
    spec = plan_research(directory, {
        'id': 'fixture', 'problem': 'Investigate when protection becomes possession.',
        'authorship': 'operator fixture', 'project_id': 'project', 'brief_id': 'brief',
        'resource_root': tmp_path / 'resources', 'fixture': True,
        'device': 'cpu', 'precision': 'fp32',
        'parent': {'path': parent, 'sha256': sha256(parent)}, 'max_steps': 1,
    })
    return root, directory, spec


def test_real_planned_research_is_read_only_and_exposes_frozen_parent(tmp_path):
    root, directory, spec = planned_session(tmp_path)
    before = {str(path.relative_to(root)): path.read_bytes()
              for path in root.rglob('*') if path.is_file()}
    with client(root) as app:
        url, detail = evidence_url(app)
        assert detail['status'] == 'planned'
        assert detail['frozen_plan'] == spec
        assert detail['authorship']['problem'] == 'operator fixture'
        assert app.get(url).content == (tmp_path / 'parent.png').read_bytes()
        schemas = app.get('/api/contracts').json()
        assert {'ResearchRequest', 'ResearchAction', 'ResearchHypothesis', 'ResearchSourceConfig',
                'ResearchRenderRequest'} <= schemas.keys()
    assert before == {str(path.relative_to(root)): path.read_bytes()
                      for path in root.rglob('*') if path.is_file()}
    assert not (directory / 'jobs.sqlite3').exists()
    assert not (tmp_path / 'resources').exists()


def test_real_ledger_research_history_and_tamper_rejection(tmp_path, monkeypatch):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    from kusama import research
    from kusama.resources import SessionBudget, load_config
    root, directory, _ = planned_session(tmp_path)
    config = load_config().model_copy(update={'session_hours': 24, 'handoff_reserve_minutes': 0,
                                             'device': 'cpu', 'cache_dir': str(tmp_path / 'cache')})
    with SessionBudget(tmp_path / 'resources', config):
        pass

    class FixtureDirector:
        def __init__(self, config):
            self.config = config
        def prepare(self):
            pass
        def close(self):
            pass

    action = {'action': 'reject', 'reason': 'The intended relationship is unsupported.',
              'note': 'Keep this unsuccessful lead for later inspection.'}

    def fixture_response(director, system, payload, images, validator, **kwargs):
        return {'output': validator(action), 'raw_response': json.dumps(action),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}

    monkeypatch.setattr(research, '_multimodal_json', fixture_response)
    report = research.run_research(directory, director_factory=FixtureDirector)
    path = directory / 'step-000.json'
    step = json.loads(path.read_text())
    with client(root) as app:
        _, detail = evidence_url(app)
        assert detail['status'] == 'step_limit'
        assert detail['reasoning_calls'] == report['reasoning_calls'] == 1
        assert detail['steps'][0] == step
        transcript = detail['transcripts'][0]
        assert transcript['prompt'] == json.loads((directory / 'decision-000.json').read_text())
        assert transcript['response']['raw_response'] == json.dumps(action)
        assert detail['notebook'][0]['action'] == 'reject'
        assert detail['notebook'][0]['origin'] == 'fixture_actor'
        path.write_text(json.dumps(step | {'status': 'rewritten'}))
        assert app.get('/api/research/' + detail['directory_id']).status_code == 409
        assert app.get('/api/research').json()['research'][0]['status'] == 'unreadable'


def test_real_plan_identity_tamper_blocks_detail(tmp_path):
    root, directory, spec = planned_session(tmp_path)
    with client(root) as app:
        _, detail = evidence_url(app)
        (directory / 'plan.json').write_text(json.dumps(spec | {'max_reasoning_calls': 999}))
        assert app.get('/api/research/' + detail['directory_id']).status_code == 409


def test_research_rejects_symlinked_authority_files(tmp_path, monkeypatch):
    root = tmp_path / 'artifacts'
    directory, _ = saved_research(root, monkeypatch)
    outside = tmp_path / 'identity.json'; outside.write_text('{}')
    (directory / 'plan_identity.json').symlink_to(outside)
    with client(root) as app:
        item = app.get('/api/research').json()['research'][0]
        assert item['status'] == 'unreadable'
        assert app.get('/api/research/' + item['directory_id']).status_code == 409


def test_malformed_research_plan_remains_visible(tmp_path):
    directory = tmp_path / 'research' / 'broken'
    directory.mkdir(parents=True)
    (directory / 'plan.json').write_text(json.dumps({'kind': 'autonomous_research', 'request': []}))
    (directory / 'plan_identity.json').write_text(json.dumps({'plan_sha256': sha256(directory / 'plan.json')}))
    with client(tmp_path) as app:
        item = app.get('/api/research').json()['research'][0]
        assert item['status'] == 'unreadable'
        assert app.get('/api/research/' + item['directory_id']).status_code == 409


def test_research_sources_cli_routes_to_bounded_catalog(monkeypatch):
    from kusama import cli, research_sources
    received = []
    monkeypatch.setattr(research_sources, 'main', lambda args: received.append(args) or 3)
    assert cli.main(['research-sources', 'plan', '--root', 'standing-catalog']) == 3
    assert received == [['plan', '--root', 'standing-catalog']]


def test_research_frontend_lazy_evidence_is_read_only_and_uses_text_nodes():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable for the existing dashboard JavaScript checks')
    script = r'''
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.events = {}; this.value = ''; this.dataset = {}; this.open = false; }
  set textContent(value) { this.valueText = value; this.children = []; }
  get textContent() { return this.valueText || ''; }
  set innerHTML(_) { throw new Error('Research data must not become markup'); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.valueText = ''; this.children = [...nodes]; }
  addEventListener(name, callback) { (this.events[name] ||= []).push(callback); }
  querySelector() { return new Element(); }
}
const nodes = new Map();
const get = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
const calls = [];
const report = { id: 'first', status: 'model_stop', fixture: true, reasoning_calls: 0,
  problem: 'When does protection become possession?', authorship: { problem: 'operator fixture' },
  question: 'What does enclosure imply?', frozen_plan: { snapshots: { parent: { sha256: 'hash' } }, request: { prior_context: 'Observed parent' } },
  steps: [{ index: 0, action: { action: 'search', query: 'enclosure', reason: 'Investigate a relationship.' } },
          { index: 1, action: null, status: 'response_rejected' }],
  transcripts: [{ index: 0, prompt: { system: 'Exact saved system instruction', payload: { question: 'Saved decision-time question', candidates: [{ title: 'Candidate from admitted context' }] }, images: [] },
    response: { raw_response: '{"action":"search","query":"Original raw query","reason":"<img src=x onerror=alert(2)>"}', output: { reason: 'Normalized output should not replace original reply' } } },
    { index: 1, response: { raw_response: 'Malformed reply preserved verbatim {', parse_error: 'Invalid action response' } }],
  hypotheses: [{ id: 'h1', title: '<img src=x onerror=alert(1)>', relationship: 'Support and restriction', sources: [{ source_id: 'one', role: 'boundary' }] }],
  notebook: [{ action: 'reject', note: 'Weak lead' }], source_history: [],
  evidence_images: [{ url: '/api/research/hash/images/asset', source_id: 'one', scope: 'whole', sha256: 'hash', rights: { status: 'allowed' } }] };
const context = vm.createContext({ console, URL, location: { origin: 'http://127.0.0.1', pathname: '/', hash: '#research' }, window: { addEventListener() {} }, history: { replaceState() {} },
  document: { querySelectorAll() { return []; }, getElementById: get, createElement: tag => new Element(tag), createTextNode: value => String(value) },
  localStorage: { getItem() { return null; } },
  fetch: async (url, options) => { assert.ok(!options?.method || options.method === 'GET'); calls.push(url);
    return { ok: true, json: async () => url === '/api/research' ? { research: [{ directory_id: 'hash', id: 'first', status: 'model_stop', fixture: true }] } : report }; }
});
let source = fs.readFileSync('src/kusama/dashboard_static/dashboard.js', 'utf8');
source = source.replace(/^initialize\(\);$/m, '');
vm.runInContext(source, context);
const allText = node => typeof node === 'string' ? node : [node.textContent, ...node.children.map(allText)].join(' ');
(async () => {
  await vm.runInContext('refreshResearch()', context);
  assert.deepEqual(calls, ['/api/research'], 'Detailed evidence is fetched only when opened');
  const session = get('research-list').children[0]; session.open = true;
  await session.events.toggle[0]();
  const rendered = allText(session);
  for (const expected of ['Starting problem', 'Step 1', 'Step 2', 'fixture actor', 'Support and restriction',
    'Weak lead', 'Source history', 'Frozen problem', 'Final selected references', 'Observed parent', '<img src=x onerror=alert(1)>',
    'Exact saved system instruction', 'Saved decision-time question', 'Candidate from admitted context', 'Original raw query',
    '<img src=x onerror=alert(2)>', 'Malformed reply preserved verbatim {', 'rejected before execution', 'No saved prompt']) assert.ok(rendered.includes(expected), expected);
  // These values must be readable outside the optional raw JSON disclosures.
  const readableText = node => typeof node === 'string' ? node : node.tag === 'pre' ? '' : [node.textContent, ...node.children.map(readableText)].join(' ');
  assert.ok(readableText(session).includes('Saved decision-time question'));
  assert.ok(readableText(session).includes('Original raw query'));
  assert.ok(!readableText(session).includes('Normalized output should not replace original reply'));
  const imageHolder = get('image-transcript-check');
  context.imageHolder = imageHolder;
  vm.runInContext('renderImageTranscript(imageHolder, {saved_action: {prompt: "A precise saved FLUX prompt", source_ids: ["one"]}, seed: 0, size: [512, 640]})', context);
  assert.ok(readableText(imageHolder).includes('A precise saved FLUX prompt'));
  assert.ok(readableText(imageHolder).includes('Saved image shown above'));
  assert.deepEqual(calls, ['/api/research', '/api/research/hash']);
  await session.events.toggle[0]();
  assert.equal(calls.length, 2, 'Opening cached evidence does not launch another request');
})().catch(error => { console.error(error); process.exitCode = 1; });
'''
    subprocess.run([node, '-e', script], cwd=Path(__file__).resolve().parents[1], check=True,
                   capture_output=True, text=True)
