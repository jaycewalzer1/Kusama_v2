"""Research-to-render integration with explicit substitutes, never real inference."""
import json
from pathlib import Path
import sqlite3

from PIL import Image
import pytest

from kusama import research as research
from kusama import research_render as render
from kusama.artifacts import sha256
from kusama.config import Config
from kusama.ledger import BudgetExceeded, Ledger, SpecMismatch
from kusama.resources import SessionBudget, load_config
from kusama.reviews import ReviewStore


class ResearchDirector:
    def __init__(self, config):
        self.config = config
    def prepare(self):
        pass
    def close(self):
        pass


class SourceBackend:
    def __init__(self, root, config, index_path=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
    def search(self, *args, **kwargs):
        return {'sources': [{'source_id': f's{i}', 'title': f'Fixture source {i}'} for i in range(3)]}
    def open(self, identifier):
        path = self.root / f'{identifier}.png'
        if not path.exists():
            Image.new('RGB', (32, 32), ['red', 'green', 'blue'][int(identifier[-1])]).save(path)
        return {'source_id': identifier, 'images': [{'path': str(path), 'sha256': sha256(path)}],
                'text': 'Attributed fixture source', 'claims': [], 'links': [],
                'reference': {'rights': {'image_display': {'status': 'allowed'}}},
                'provenance': {'origin': 'fixture'}}
    def summary(self):
        return {'fixture': True, 'requests': 0}
    def close(self):
        pass


class Renderer:
    loads = 0
    calls = []
    fail_seed = None
    def __init__(self, config):
        self.config = config
    def prepare(self):
        type(self).loads += 1
    def prompt_token_count(self, prompt):
        return len(prompt.split())
    def generate(self, prompt, seed, parent_path=None):
        assert parent_path is None
        type(self).calls.append((prompt, seed))
        if seed == type(self).fail_seed:
            raise RuntimeError('Fixture renderer failure')
        return Image.new('RGB', (self.config.width, self.config.height), (seed % 255, 0, 120))
    def close(self):
        pass


def hypothesis(identifier, sources):
    return {'id': identifier, 'title': 'Fixture hypothesis ' + identifier,
        'question': 'How can a boundary support and constrain?',
        'sources': [{'source_id': sid, 'role': 'Study a boundary',
                     'transferred_property': 'An opening', 'discarded_property': 'Original subject'} for sid in sources],
        'relationship': 'Proposed relation ' + identifier, 'transformation': 'Proposed operation ' + identifier,
        'composition': 'An opening in an enclosing form', 'render_instruction': 'Paint an opening in a form ' + identifier,
        'visible_test': 'Inspect the boundary', 'uncertainty': 'Fixture only; no artistic judgment'}


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    root, research_output, output = tmp_path / 'session', tmp_path / 'research', tmp_path / 'render'
    config = Config(output_dir=output, cache_dir=tmp_path / 'cache', device='cpu', precision='fp32',
                    local_files_only=True, width=64, height=64, steps=1)
    cfg = load_config().model_copy(update={'session_hours': 24, 'handoff_reserve_minutes': 0, 'device': 'cpu',
        'cache_dir': str(tmp_path / 'cache'),
        'render_profiles': {**load_config().render_profiles,
                            'fixture64': {'width': 64, 'height': 64, 'steps': 1, 'precision': 'fp32'}}})
    with SessionBudget(root, cfg):
        pass
    h1, h2 = hypothesis('h1', ['s0']), hypothesis('h2', ['s0', 's1', 's2'])
    if getattr(request, 'param', False):
        h2['render_images'] = [{'source_id': s, 'image_index': 0, 'purpose': 'Borrow the boundary form'}
                               for s in ('s2', 's0')]
        h2['render_instruction'] = 'Paint an opening using the boundary of image 1 and color of image 2'
    actions = [{'action': 'search', 'query': 'boundary'}]
    actions += [{'action': 'open', 'source_id': f's{i}'} for i in range(3)]
    actions += [{'action': action, 'hypothesis': value} for value in (h1, h2) for action in ('draft', 'commit')]
    def model(director, system, payload, images, validator, **kwargs):
        value = {'reason': 'Fixture research decision', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(research, '_multimodal_json', model)
    research.plan_research(research_output, {'id': 'fixture', 'project_id': 'project', 'brief_id': 'brief',
        'problem': 'A consequential unresolved fixture problem', 'authorship': 'fixture operator',
        'resource_root': str(root), 'device': 'cpu', 'precision': 'fp32', 'max_steps': 8,
        'max_hypotheses': 2, 'fixture': True})
    result = research.run_research(research_output, director_factory=ResearchDirector, sources_factory=SourceBackend)
    assert len(result['hypotheses']) == 2
    request = {'id': 'fixture', 'research_output': str(research_output), 'hypothesis_ids': ['h1', 'h2'],
        'seeds': [17, 23, 41, 59], 'resource_root': str(root), 'render_profile_id': 'fixture64',
        'render_config': config.to_dict(), 'selection_origin': 'all_model_hypotheses',
        'selection_reason': 'Render every committed model hypothesis for human inspection'}
    Renderer.loads = 0; Renderer.calls = []; Renderer.fail_seed = None
    return root, research_output, output, request


def test_freezes_variable_reference_sets_and_saved_model_prompts(prepared):
    root, source, output, request = prepared
    spec = render.plan_render(output, request)
    assert [len(h['sources']) for h in spec['hypotheses']] == [1, 3]
    assert [a['source_ids'] for a in spec['actions']] == [['s0'], ['s0'], ['s0', 's1', 's2'], ['s0', 's1', 's2']]
    assert [a['seed'] for a in spec['actions']] == [17, 23, 41, 59]
    assert [a['prompt'] for a in spec['actions']] == [h['composition'] + '\n' + h['render_instruction'] for h in spec['hypotheses'] for _ in range(2)]
    for h in spec['hypotheses']:
        for reference in h['source_snapshots']:
            for image in reference['images']:
                assert Path(image['path']).is_relative_to(output)
                assert image['sha256'] == sha256(Path(image['path']))
    assert render.plan_render(output, request) == spec
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 0
    assert Renderer.loads == 0 and not Renderer.calls


def test_subset_needs_operator_attribution_and_uncommitted_ids_rejected(prepared):
    _, _, output, request = prepared
    request.update(hypothesis_ids=['h2'], seeds=[17])
    with pytest.raises(ValueError, match='Subset'):
        render.plan_render(output, request)
    assert not output.exists()
    request['selection_origin'] = 'operator_selection'
    assert render.plan_render(output, request)['request']['selection_origin'] == 'operator_selection'
    request['hypothesis_ids'] = ['invented']
    other = output.parent / 'other'
    request['render_config']['output_dir'] = str(other)
    with pytest.raises(ValueError, match='committed'):
        render.plan_render(other, request)


def test_planning_and_execution_do_not_create_an_unadmitted_resource_session(prepared):
    _, _, output, request = prepared
    missing = output.parent / 'unadmitted'
    request['resource_root'] = str(missing)
    render.plan_render(output, request)
    assert not missing.exists()
    with pytest.raises(ValueError, match='session'):
        render.run_render(output, backend_factory=Renderer)
    assert not missing.exists() and Renderer.loads == 0


def test_fixture_cannot_dispatch_real_flux(prepared):
    _, _, output, request = prepared
    render.plan_render(output, request)
    with pytest.raises(ValueError, match='Fixture'):
        render.run_render(output)


def test_profile_mismatch_stops_before_model_load(prepared, monkeypatch):
    _, _, output, request = prepared
    original = render.read_report
    monkeypatch.setattr(render, 'read_report', lambda path: {**original(path), 'fixture': False})
    request['render_profile_id'] = 'flux_base4b_1024_v1'
    render.plan_render(output, request)
    with pytest.raises(SpecMismatch, match='profile'):
        render.run_render(output, backend_factory=Renderer)
    assert Renderer.loads == 0 and not Renderer.calls


def test_matching_named_profile_runs_without_substitution(prepared, monkeypatch):
    _, _, output, request = prepared
    original = render.read_report
    monkeypatch.setattr(render, 'read_report', lambda path: {**original(path), 'fixture': False})
    request.update(hypothesis_ids=['h1'], seeds=[17], selection_origin='operator_selection')
    render.plan_render(output, request)
    report = render.run_render(output, backend_factory=Renderer)
    assert report['images'][0]['size'] == [64, 64]
    assert report['renderer_attempts'] == 1 and len(Renderer.calls) == 1


def test_exhausted_renderer_budget_prevents_loading(prepared):
    _, _, output, request = prepared
    root = output.parent / 'exhausted'
    cfg = load_config().model_copy(update={'session_hours': 24, 'handoff_reserve_minutes': 0,
        'device': 'cpu', 'renderer_attempt_cap': 0})
    with SessionBudget(root, cfg):
        pass
    request['resource_root'] = str(root)
    render.plan_render(output, request)
    with pytest.raises(BudgetExceeded, match='renderer cap'):
        render.run_render(output, backend_factory=Renderer)
    assert Renderer.loads == 0 and not Renderer.calls


def test_generates_once_at_exact_profile_and_registers_review_without_labels(prepared):
    root, _, output, request = prepared
    render.plan_render(output, request)
    report = render.run_render(output, backend_factory=Renderer)
    assert report['status'] == 'complete' and report['renderer_attempts'] == 4
    assert len(Renderer.calls) == 4
    assert all(im['size'] == [64, 64] and im['parent_sha256'] is None for im in report['images'])
    assert all(im['sha256'] == sha256(Path(im['path'])) for im in report['images'])
    assert report['human_quality'] is None and report['training_performed'] is False
    assert all(im['origin'] == 'render_fixture' for im in report['images'])
    review = ReviewStore(Path(report.get('review_root', output)))
    collection = review.collection(report['review_collection_id'])
    assert collection['presentation']['condition'] == 'statement_free'
    assert collection['pool_expected_count'] == 4
    assert [candidate['direction_id'] for candidate in collection['candidates']] == ['h1', 'h1', 'h2', 'h2']
    with sqlite3.connect(review.database) as db:
        assert db.execute('SELECT count(*) FROM review_votes').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM review_advancements').fetchone()[0] == 0
        assert db.execute('SELECT count(*) FROM review_pool_votes').fetchone()[0] == 0
    replay = render.run_render(output, backend_factory=Renderer)
    assert replay['images'] == report['images'] and len(Renderer.calls) == 4
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 4 and not budget.summary()['unsettled']


def test_failed_attempt_is_charged_and_never_retried(prepared):
    root, _, output, request = prepared
    Renderer.fail_seed = 23
    render.plan_render(output, request)
    report = render.run_render(output, backend_factory=Renderer)
    assert report['status'] == 'partial' and len(report['images']) == 3
    assert report['renderer_attempts'] == 4 and len(report['failures']) == 1
    render.run_render(output, backend_factory=Renderer)
    assert len(Renderer.calls) == 4
    with SessionBudget(root) as budget:
        renderer_calls = [call for call in budget.summary()['calls'] if call['kind'] == 'renderer']
        assert len(renderer_calls) == 4 and sum(c['status'] == 'failed' for c in renderer_calls) == 1


def test_uncertain_charged_attempt_cannot_be_repeated(prepared):
    root, _, output, request = prepared
    spec = render.plan_render(output, request)
    job_id = 'research_render_' + request['id']
    with Ledger(output / 'jobs.sqlite3') as ledger, SessionBudget(root) as budget:
        ledger.create_job(job_id, spec, len(spec['actions']), output)
        budget.reserve(job_id + ':renderer:0', 'renderer')
        ledger.open_attempt(job_id, 0)
        ledger.reserve_call(job_id, 0)
    with pytest.raises(RuntimeError, match='Uncertain'):
        render.run_render(output, backend_factory=Renderer)
    assert Renderer.loads == 0 and not Renderer.calls


def test_saved_output_reconciles_without_replacement_render(prepared, monkeypatch):
    _, _, output, request = prepared
    request.update(hypothesis_ids=['h1'], seeds=[17], selection_origin='operator_selection')
    render.plan_render(output, request)
    original = render.immutable_json
    def interrupt(path, value):
        if path.name == 'image-000.json':
            raise OSError('Fixture interruption after durable PNG')
        return original(path, value)
    monkeypatch.setattr(render, 'immutable_json', interrupt)
    with pytest.raises(RuntimeError, match='reconciliation'):
        render.run_render(output, backend_factory=Renderer)
    saved_hash = sha256(output / 'image-000.png')
    monkeypatch.setattr(render, 'immutable_json', original)
    report = render.run_render(output, backend_factory=Renderer)
    assert report['renderer_attempts'] == 1 and len(Renderer.calls) == 1
    assert report['images'][0]['sha256'] == saved_hash


def test_tampered_output_and_frozen_source_are_rejected(prepared):
    _, _, output, request = prepared
    spec = render.plan_render(output, request)
    report = render.run_render(output, backend_factory=Renderer)
    image = Path(report['images'][0]['path'])
    Image.new('RGB', (64, 64), 'yellow').save(image)
    with pytest.raises((ValueError, SpecMismatch)):
        render.run_render(output, backend_factory=Renderer)
    assert len(Renderer.calls) == 4
    source = Path(spec['hypotheses'][0]['source_snapshots'][0]['images'][0]['path'])
    source.write_bytes(b'changed')
    with pytest.raises((ValueError, SpecMismatch)):
        render.preflight_render(output)


def test_rendering_uses_frozen_copies_after_original_research_changes(prepared):
    _, _, output, request = prepared
    spec = render.plan_render(output, request)
    image = spec['hypotheses'][0]['source_snapshots'][0]['images'][0]
    original = Path(image['original_path'])
    original.write_bytes(b'later mutation of source research')
    report = render.run_render(output, backend_factory=Renderer)
    assert report['status'] == 'complete'
    assert sha256(Path(image['path'])) == image['sha256']


def test_dependency_identity_changes_reject_before_dispatch(prepared, monkeypatch):
    _, _, output, request = prepared
    spec = render.plan_render(output, request)
    assert {'backend.py', 'config.py', 'resources.py', 'ledger.py', 'uv.lock'} <= spec['code_sha256'].keys()
    monkeypatch.setattr(render, 'code_identity', lambda: {**spec['code_sha256'], 'backend.py': '0' * 64})
    with pytest.raises(SpecMismatch, match='implementation'):
        render.run_render(output, backend_factory=Renderer)
    assert Renderer.loads == 0 and not Renderer.calls


@pytest.mark.parametrize('prepared', [True], indirect=True)
def test_model_selected_images_reach_renderer_in_order_and_resume_without_recall(prepared):
    root, source, output, request = prepared
    captured = []
    class ImageRenderer(Renderer):
        def generate(self, prompt, seed, parent_path=None, *, reference_images=None):
            assert parent_path is None
            captured.append((seed, reference_images))
            if reference_images:
                assert [a['source_id'] for a in reference_images] == ['s2', 's0']
                assert [Image.open(a['path']).getpixel((0, 0)) for a in reference_images] == [(0, 0, 255), (255, 0, 0)]
                assert all(Path(a['path']).is_relative_to(output) for a in reference_images)
            return super().generate(prompt, seed)
    spec = render.plan_render(output, request)
    assert [a['conditioning']['mode'] for a in spec['actions']] == [
        'text_only', 'text_only', 'image_conditioned', 'image_conditioned']
    assert [i['input_index'] for i in spec['actions'][2]['conditioning']['images']] == [1, 2]
    result = render.run_render(output, backend_factory=ImageRenderer)
    assert result['status'] == 'complete' and len(captured) == 4
    assert result['images'][2]['conditioning'] == spec['actions'][2]['conditioning']
    assert result['images'][2]['parent_image_id'] is None
    again = render.run_render(output, backend_factory=ImageRenderer)
    assert again['images'] == result['images'] and len(captured) == 4


@pytest.mark.parametrize('prepared', [True], indirect=True)
def test_reference_tamper_stops_before_renderer_load(prepared):
    _, _, output, request = prepared
    spec = render.plan_render(output, request)
    Path(spec['actions'][2]['reference_images'][0]['path']).write_bytes(b'changed')
    with pytest.raises(SpecMismatch, match='evidence changed'):
        render.run_render(output, backend_factory=Renderer)
    assert Renderer.loads == 0 and not Renderer.calls
