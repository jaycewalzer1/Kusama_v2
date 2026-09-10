"""Real workflow and durable ledgers with explicitly scripted models, no inference."""
from copy import deepcopy
import json
from pathlib import Path
import os

from PIL import Image
import pytest

from kusama import research, research_render, study_development as studies
from kusama.artifacts import sha256
from kusama.config import Config
from kusama.ledger import BudgetExceeded, SpecMismatch
from kusama.panel_runs import StartRun
from kusama.resources import SessionBudget, load_config
from test_research_render import SourceBackend, ResearchDirector, hypothesis


def study(index):
    return hypothesis(f'h{index}', ['s0']) | {
        'composition': f'An enclosing form with {index} openings',
        'artistic_operation': f'Fixture spatial operation {index}',
        'study_reason': f'Find out whether relation {index} alters the enclosure',
        'distinct_from_previous': f'Investigates different spatial relation {index}'}


class Renderer:
    calls = []
    fail_seed = None

    def __init__(self, config): self.config = config
    def prepare(self): pass
    def close(self): pass
    def prompt_token_count(self, prompt): return len(prompt.split())
    def generate(self, prompt, seed, parent_path=None, *, reference_images=None):
        self.calls.append({'prompt': prompt, 'seed': seed,
                           'parent_hash': sha256(parent_path) if parent_path else None,
                           'references': reference_images or []})
        if seed == self.fail_seed: raise ValueError('Retained fixture failure')
        return Image.new('RGB', (64, 64), (seed, 100, 50))


@pytest.fixture
def prepared(tmp_path, monkeypatch, request):
    from research_legacy import use_legacy_protocol
    use_legacy_protocol(monkeypatch)
    parameters = getattr(request, 'param', (3, 1))
    count, variants = parameters[:2]
    cycle = parameters[2] if len(parameters) > 2 else None
    root, source, output = tmp_path / 'resources', tmp_path / 'research', tmp_path / 'render'
    cfg = load_config().model_copy(update={'device': 'cpu', 'session_hours': 24,
        'handoff_reserve_minutes': 0, 'renderer_attempt_cap': count * variants + (cycle['max_steps'] if cycle else 1),
        'reasoning_token_cap': (2 + 2 * count + count * variants + (2 * cycle['max_steps'] + 1
            + cycle.get('research_sessions', 1) * cycle.get('research_steps', 8) if cycle else 2)) * 8192})
    with SessionBudget(root, cfg): pass
    actions = [{'action': 'search', 'query': 'boundary'}, {'action': 'open', 'source_id': 's0'}]
    actions += [{'action': action, 'hypothesis': study(i)} for i in range(1, count + 1) for action in ('draft', 'commit')]
    def model(director, system, payload, images, validator, **kwargs):
        value = {'reason': 'Scripted fixture', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value)}
    monkeypatch.setattr(research, '_multimodal_json', model)
    research.plan_research(source, {'id': 'studies', 'problem': 'What can an enclosure do?',
        'project_id': 'project', 'brief_id': 'brief', 'authorship': 'fixture_operator',
        'resource_root': str(root), 'device': 'cpu', 'precision': 'fp32', 'fixture': True,
        'max_steps': len(actions), 'max_hypotheses': count, 'study_development': True, 'total_token_limit': 8192})
    result = research.run_research(source, director_factory=ResearchDirector, sources_factory=SourceBackend)
    assert len(result['hypotheses']) == count
    config = Config(output_dir=output, cache_dir=tmp_path / 'cache', local_files_only=True,
        device='cpu', precision='fp32', width=64, height=64, steps=1)
    request = {'id': 'studies', 'research_output': str(source), 'hypothesis_ids': [f'h{i}' for i in range(1, count+1)],
        'seeds': list(range(23, 23 + count * variants)), 'resource_root': str(root),
        'render_profile_id': 'fixture', 'render_config': config.to_dict(),
        'selection_origin': 'all_model_hypotheses', 'selection_reason': 'Inspect provisional studies',
        'register_review': False, 'study_development': True, 'development_cycle': cycle}
    spec = research_render.plan_render(output, request)
    Renderer.calls = []; Renderer.fail_seed = None
    return root, output, spec


def actor(monkeypatch, action='continue', *, invalid=False, stop=None):
    calls = []
    def model(director, system, payload, images, validator, **kwargs):
        calls.append({'system': system, 'payload': deepcopy(payload), 'images': list(images)})
        assert images and all(Path(path).is_file() for _, path in images)
        if system == studies.CHOOSE:
            ids = [ident for ident, _ in images]
            n = {'continue': 1, 'combine': 3, 'abandon': 0, 'reformulate': 0, 'stop': 1}[action]
            value = {'action': action, 'selected_image_ids': ids[:n],
                'reference_purposes': {ident: 'Transfer the visible interval from this study' for ident in ids[1:n]},
                'commitments': 'Keep the relation unresolved', 'what_changed': 'The spaces between forms matter more',
                'reason': 'Develop the spaces seen between forms',
                'evidence': [{'image_id': ident, 'observation': 'A visible interval'} for ident in ids]}
            if action in ('continue', 'combine', 'reformulate'):
                value.update(new_question='Can the intervals carry the work?', composition='Two openings divided by a narrow band',
                             render_instruction='Alter the central band; keep the two openings')
            if invalid: value['selected_image_ids'] = ['invented']
        else:
            value = {'visible_relationships': ['An interval separates the forms'], 'unexpected_effects': [],
                'explanation_gaps': ['Agency is not evident from the image'], 'uncertainty': 'Interpretation is uncertain',
                'possible_next_question': 'Could the interval become the subject?'}
        if stop: stop(len(calls))
        return {'output': validator(value), 'raw_response': json.dumps(value)}
    monkeypatch.setattr(studies, '_multimodal_json', model)
    return calls


@pytest.mark.parametrize('action', ['continue', 'combine', 'abandon', 'reformulate', 'stop'])
def test_studies_inspected_before_choice_and_one_optional_followup(prepared, monkeypatch, action):
    root, output, spec = prepared
    calls = actor(monkeypatch, action)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    development = result['study_development']
    assert result['status'] == development['status'] == 'complete'
    assert [call['system'] for call in calls[:4]] == [studies.OBSERVE] * 3 + [studies.CHOOSE]
    assert len(calls[3]['images']) == 3
    assert len(calls[3]['payload']['observations']) == 3
    assert len({h['artistic_operation'] for h in spec['hypotheses']}) == 3
    assert all(a['prompt'] == h['composition'] + '\n' + h['render_instruction']
               for a, h in zip(Renderer.calls, spec['hypotheses']))
    renders = action not in ('abandon', 'stop')
    assert len(Renderer.calls) == 3 + renders
    assert len(calls) == 4 + renders
    assert not list(output.rglob('reviews.sqlite3'))
    if renders:
        follow = json.loads((output / 'followup' / 'render_plan.json').read_text())
        actual = follow['actions'][0]
        assert actual['prompt'] == development['choice']['composition'] + '\n' + development['choice']['render_instruction']
        assert calls[-1]['system'] == studies.OBSERVE
        assert development['followup_observation']['explanation_gaps']
        if action != 'reformulate':
            assert Renderer.calls[-1]['parent_hash'] == result['images'][0]['sha256']
            assert actual['conditioning']['images'][0]['role'] == 'parent'
            child = development['followup']['images'][0]
            assert child['id'] != child['parent_image_id']
            assert child['parent_sha256'] == result['images'][0]['sha256']
        else:
            assert Renderer.calls[-1]['parent_hash'] is None
        if action == 'combine':
            assert [im['sha256'] for im in Renderer.calls[-1]['references']] == [im['sha256'] for im in result['images'][1:]]
    before = (len(calls), len(Renderer.calls))
    again = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert again == result and before == (len(calls), len(Renderer.calls)), 'Resume must not recharge or repeat work'
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 3 + renders
        assert budget.db.execute('SELECT count(*) FROM calls WHERE kind="reasoning"').fetchone()[0] == 12 + renders


@pytest.mark.parametrize('prepared', [(1, 2)], indirect=True)
def test_seed_variants_cannot_authorize_artistic_commitment(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'partial' and result['study_development']['choice'] is None
    assert len(calls) == 2 and all(c['system'] == studies.OBSERVE for c in calls)


def test_failed_image_blocks_selection_but_preserves_other_observations(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch); Renderer.fail_seed = 24
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'partial' and len(result['images']) == 2
    assert len(calls) == 2 and result['study_development']['choice'] is None
    assert (output / 'failure-001.json').exists()


def test_invalid_choice_is_charged_once_and_never_renders(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch, invalid=True)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'partial' and len(Renderer.calls) == 3 and len(calls) == 4
    again = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert again == result and len(calls) == 4
    assert (output / 'study_development' / 'response-003.json').exists()


def test_saved_image_tamper_stops_before_more_model_calls(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch, 'stop')
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    Image.new('RGB', (64, 64), 'white').save(result['images'][0]['path'])
    with pytest.raises((ValueError, SpecMismatch)):
        research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert len(calls) == 4


def test_fixture_without_observer_cannot_call_real_qwen(prepared):
    _, output, _ = prepared
    result = research_render.run_render(output, backend_factory=Renderer)
    assert result['status'] == 'partial'
    assert 'explicit substitute' in result['study_development']['reason']


def test_distinctness_and_default_panel_contract():
    with pytest.raises(ValueError, match='three distinct'):
        StartRun(submission_id='test', question='An unresolved question', directions=1)
    assert StartRun(submission_id='test', question='An unresolved question').directions == 3
    h = study(1)
    with pytest.raises(ValueError, match='study_reason'):
        research.validate_study(h | {'study_reason': ''})
    with pytest.raises(ValueError, match='Repeated artistic_operation'):
        research.validate_study(study(2) | {'artistic_operation': h['artistic_operation']}, [h])


def test_stop_after_inspections_prevents_choice_and_resume_reuses_evidence(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch, 'stop')
    def stop():
        if len(calls) == 3: raise BudgetExceeded('Stop requested before selection')
    with pytest.raises(BudgetExceeded, match='Stop requested'):
        research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector, before_dispatch=stop)
    assert len(calls) == 3 and len(Renderer.calls) == 3
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'complete' and len(calls) == 4 and len(Renderer.calls) == 3


def test_missing_charged_response_blocks_repeat_inference(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch, 'stop')
    research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    (output / 'study_development' / 'response-003.json').unlink()
    with pytest.raises((RuntimeError, ValueError, SpecMismatch, FileNotFoundError)):
        research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert len(calls) == 4 and len(Renderer.calls) == 3


def test_oversized_followup_prompt_is_retained_without_renderer_dispatch(prepared, monkeypatch):
    _, output, _ = prepared
    calls = actor(monkeypatch)
    original = studies.validate_choice
    def oversized(value, images, spec):
        value = {**value, 'composition': 'form ' * 330, 'render_instruction': 'space ' * 290}
        return original(value, images, spec)
    monkeypatch.setattr(studies, 'validate_choice', oversized)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'partial' and len(calls) == 4 and len(Renderer.calls) == 3
    response = json.loads((output / 'study_development' / 'response-003.json').read_text())
    assert 'exceeds FLUX capacity' in response['inference_error']


def test_followup_uses_frozen_context_even_if_original_research_changes(prepared, monkeypatch):
    _, output, spec = prepared
    def change_source(count):
        if count == 4:
            Path(spec['request']['research_output'], 'plan.json').write_text('{"later": "unrelated changes"}')
    actor(monkeypatch, stop=change_source)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'complete'
    follow = json.loads((output / 'followup' / 'render_plan.json').read_text())
    assert follow['operator_context'] == spec['operator_context']
    assert follow['research_plan_sha256'] == spec['research_plan_sha256']
    assert follow['director_config'] == spec['director_config']


@pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1',
                    reason='Optional cached-processor study prompt check; no model inference')
def test_actual_processor_admits_twelve_images_and_complete_choice_context(tmp_path):
    from kusama.director import DirectorConfig
    config = DirectorConfig(Path('.cache/huggingface').resolve(), max_new_tokens=1800, max_image_pixels=65536)
    images = []
    for index in range(12):
        path = tmp_path / f'image-{index:03}.png'
        Image.new('RGB', (512, 640), (index, 100, 30)).save(path)
        images.append({'id': path.stem, 'path': str(path), 'sha256': sha256(path)})
    payload = {'problem': 'How could a shared boundary shelter people while excluding others?',
        'observations': [{'image_id': image['id'], 'visible_relationships': ['Two shapes share a narrow interval.'],
            'unexpected_effects': [], 'explanation_gaps': ['The proposed action is not visible.'],
            'uncertainty': 'The role of the smaller shape is uncertain.',
            'possible_next_question': 'Can the shared interval become the subject?'} for image in images],
        'studies': [{'id': f'h{i}', 'artistic_operation': f'Fixture operation {i}',
                    'study_reason': 'Find out whether different spatial relations alter the enclosure.'} for i in range(6)],
        'output_schema': studies.StudyChoice.model_json_schema()}
    result = studies.admit_study_context(studies.CHOOSE, payload, images, config)
    assert result['input_tokens'] <= 6392
    assert [im['sha256'] for im in result['input_images']] == [im['sha256'] for im in images]
    assert 'explanation_gaps' in result['prompt']
