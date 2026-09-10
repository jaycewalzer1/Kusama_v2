"""Actual research/render/ledger orchestration with explicit fixture actors."""
from copy import deepcopy
import json
import os
from pathlib import Path

from PIL import Image
import pytest

from kusama import research, research_render, study_development as studies, study_cycle as cycle
from kusama.artifacts import sha256
from kusama.ledger import BudgetExceeded, SpecMismatch
from kusama.resources import SessionBudget
from test_study_development import prepared, Renderer, ResearchDirector
from test_research_render import SourceBackend, hypothesis


POLICY = {'max_steps': 3, 'research_steps': 8}
CURRENT_RESEARCH_POLICY = deepcopy(research.PROGRESS_POLICY)
CURRENT_RESEARCH_VERSION = research.VERSION


def cycle_actor(monkeypatch, actions, *, invalid=None):
    calls = []
    sequence = iter(actions)
    def model(director, system, payload, images, validator, **kwargs):
        calls.append({'system': system, 'payload': deepcopy(payload), 'images': list(images)})
        assert images and all(Path(path).is_file() for _, path in images)
        if system == cycle.CYCLE_SYSTEM:
            action, indices, *options = next(sequence)
            ids = [b['id'] for b in payload['branches']]
            selected = [ids[i] for i in indices]
            sources = [s['id'] for s in payload['source_images']][:1] if options and options[0] else []
            value = {'action': action, 'selected_image_ids': selected, 'source_image_ids': sources,
                'reference_purposes': {i: 'Transfer the observed spacing' for i in selected[1:] + sources},
                'commitments': 'Preserve unresolved relationships', 'what_changed': 'The gap became more consequential',
                'reason': 'Investigate the visible gap instead of assuming the proposal succeeded',
                'evidence': [{'image_id': i, 'observation': 'An interval interrupts the enclosing form'} for i in ids]}
            if action in ('research', 'continue', 'combine', 'reformulate'):
                value['new_question'] = 'How could the visible interval become the subject?'
            if action in ('continue', 'combine', 'reformulate'):
                value.update(composition='Two openings separated by a thin band',
                             render_instruction='Alter the central interval; preserve the two openings')
            if invalid:
                invalid(value, payload)
        else:
            value = {'visible_relationships': ['Two forms share an interval'], 'unexpected_effects': ['The gap dominates'],
                'explanation_gaps': ['Agency is not directly evident'], 'uncertainty': 'Interpretation remains uncertain',
                'possible_next_question': 'Could the interval become the subject?'}
        return {'output': validator(value), 'raw_response': json.dumps(value)}
    monkeypatch.setattr(studies, '_multimodal_json', model)
    return calls


def research_actor(monkeypatch, *, empty=False):
    h = hypothesis('new-reference', ['s1'])
    h['render_images'] = [{'source_id': 's1', 'image_index': 0, 'purpose': 'Study the visible opening'}]
    actions = ([{'action': 'stop', 'reason': 'No resolved finding within this bounded search'}] if empty else
        [{'action': 'search', 'query': 'interval'}, {'action': 'open', 'source_id': 's1'},
         {'action': 'draft', 'hypothesis': h}, {'action': 'commit', 'hypothesis': h}])
    calls = []
    def model(director, system, payload, images, validator, **kwargs):
        calls.append({'payload': deepcopy(payload), 'images': images})
        value = {'reason': 'Image-led fixture research', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value)}
    monkeypatch.setattr(research, '_multimodal_json', model)
    runner = research.run_research
    monkeypatch.setattr(research, 'run_research', lambda output, **kw: runner(output, sources_factory=SourceBackend, **kw))
    return calls


@pytest.mark.parametrize('prepared', [(3, 1, POLICY)], indirect=True)
def test_images_reopen_research_then_revisit_and_combine_saved_branches(prepared, monkeypatch):
    root, output, spec = prepared
    calls = cycle_actor(monkeypatch, [('research', [0]), ('continue', [0], True),
                                     ('combine', [1, 3], True), ('review', [4])])
    research_calls = research_actor(monkeypatch)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    development = result['study_development']
    assert result['status'] == 'complete', development
    assert development['human_review_requested'] and development['human_quality'] is None
    assert [r['choice']['action'] for r in development['cycles']] == ['research', 'continue', 'combine', 'review']
    assert len(development['branches']) == len({b['id'] for b in development['branches']}) == 5
    assert len(Renderer.calls) == 5
    assert Renderer.calls[-1]['parent_hash'] == result['images'][1]['sha256'], 'Revisit an older branch'
    assert len(Renderer.calls[-1]['references']) == 2
    assert Renderer.calls[-1]['references'][0]['sha256'] == development['branches'][3]['sha256']
    assert research_calls[0]['payload']['problem'] == spec['problem']
    assert research_calls[0]['payload']['current_question'] == development['cycles'][0]['choice']['new_question']
    assert 'Two forms share an interval' in research_calls[0]['payload']['development_context']
    assert research_calls[0]['images'], 'The actual selected parent must reach reopened research'
    next_choice = [c for c in calls if c['system'] == cycle.CYCLE_SYSTEM][1]
    assert next_choice['payload']['research_findings'][0]['findings']
    assert len(next_choice['images']) == 4, 'Three branches plus the researched image'
    final = [c for c in calls if c['system'] == cycle.CYCLE_SYSTEM][-1]
    assert final['payload']['allowed_actions'] == ['review', 'stop', 'abandon']
    assert not list(output.rglob('reviews.sqlite3'))
    before = len(calls), len(research_calls), len(Renderer.calls)
    again = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert again == result and before == (len(calls), len(research_calls), len(Renderer.calls))
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 5
        assert budget.db.execute('SELECT count(*) FROM calls WHERE kind="reasoning"').fetchone()[0] == 21


@pytest.mark.parametrize('prepared', [(3, 1, POLICY)], indirect=True)
def test_empty_research_returns_uncertainty_without_inventing_sources(prepared, monkeypatch):
    _, output, _ = prepared
    calls = cycle_actor(monkeypatch, [('research', [1]), ('reformulate', []), ('stop', [3])])
    research_actor(monkeypatch, empty=True)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    dev = result['study_development']
    assert result['status'] == 'complete' and len(Renderer.calls) == 4
    assert dev['research'][0]['sources'] == dev['research'][0]['findings'] == []
    assert Renderer.calls[-1]['parent_hash'] is None
    choice = [c for c in calls if c['system'] == cycle.CYCLE_SYSTEM][1]
    assert 'research' not in choice['payload']['allowed_actions'], 'No implicit second research allowance'


@pytest.mark.parametrize('prepared', [(3, 1, POLICY)], indirect=True)
def test_reopened_inquiry_uses_current_grounding_and_image_reinspection(prepared, monkeypatch):
    from test_research_grounded import Sources, analysis, hyp, observations
    _, output, spec = prepared
    monkeypatch.setattr(research, 'VERSION', CURRENT_RESEARCH_VERSION)
    monkeypatch.setattr(research, 'PROGRESS_POLICY', CURRENT_RESEARCH_POLICY)
    h = hyp(('artic',))
    h['single_source_justification'] = 'One inspected source isolates the spatial relationship for this fixture'
    h['render_images'] = [{'source_id': 'artic', 'image_index': 0, 'purpose': 'Study the visible interval'}]
    actions = [analysis(), {'action': 'search', 'query': 'figure', 'mode': 'remote'},
        {'action': 'assess_candidates', 'candidate_scores': [{'candidate_id': 'artic', 'score': .8,
            'reason': 'Visible geometry is relevant to the question'}]},
        {'action': 'acquire', 'candidate_id': 'artic'},
        {'action': 'inspect_images', 'reinspection': [observations('artic_canonical')]},
        {'action': 'draft', 'hypothesis': h},
        {'action': 'commit', 'hypothesis': h, 'reinspection': [observations('artic_canonical')]}]
    contexts = []
    def model(director, system, payload, images, validator, **kwargs):
        contexts.append(deepcopy(payload))
        value = {'reason': 'Current grounded fixture decision', **actions.pop(0)}
        return {'output': validator(value), 'raw_response': json.dumps(value)}
    monkeypatch.setattr(research, '_multimodal_json', model)
    runner = research.run_research
    monkeypatch.setattr(research, 'run_research', lambda path, **kw: runner(path,
        sources_factory=lambda root, *a, **kw: Sources(root), **kw))
    cycle_actor(monkeypatch, [('research', [1]), ('continue', [1], True), ('review', [3])])
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    dev = result['study_development']
    assert result['status'] == 'complete', dev
    assert dev['research'][0]['status'] == 'hypotheses_ready', dev['research'][0]
    assert len(contexts) == 7 and dev['research'][0]['reasoning_calls'] == 7
    assert all(c['problem'] == spec['problem'] for c in contexts)
    assert contexts[0]['current_question'] == dev['cycles'][0]['choice']['new_question']
    assert contexts[-1]['pending_draft']['hypothesis']['sources'][0]['source_id'] == 'artic_canonical'
    assert Renderer.calls[-1]['references'][0]['sha256'] == dev['research'][0]['sources'][0]['sha256']


@pytest.mark.parametrize('prepared', [(3, 1, {'max_steps': 1, 'research_sessions': 0})], indirect=True)
def test_terminal_budget_cannot_dispatch_another_render(prepared, monkeypatch):
    _, output, _ = prepared
    calls = cycle_actor(monkeypatch, [('continue', [0]), ('continue', [3])])
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'partial' and len(Renderer.calls) == 4
    assert 'rejected' in result['study_development']['reason']
    again = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert again == result and len(Renderer.calls) == 4 and len(calls) == 6


@pytest.mark.parametrize('prepared', [(3, 1, POLICY)], indirect=True)
def test_stop_after_render_replays_without_repeating_charged_work(prepared, monkeypatch):
    _, output, _ = prepared
    calls = cycle_actor(monkeypatch, [('continue', [0]), ('review', [3])])
    def stop():
        if len(Renderer.calls) == 4:
            raise BudgetExceeded('Stop after current render')
    with pytest.raises(BudgetExceeded):
        research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector, before_dispatch=stop)
    assert len(Renderer.calls) == len(calls) == 4
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert result['status'] == 'complete' and len(Renderer.calls) == 4 and len(calls) == 6


@pytest.mark.parametrize('prepared', [(3, 1, POLICY)], indirect=True)
@pytest.mark.parametrize('target', ['branch', 'research_response', 'choice_response'])
def test_tampered_or_missing_cycle_evidence_blocks_replay(prepared, monkeypatch, target):
    _, output, _ = prepared
    calls = cycle_actor(monkeypatch, [('research', [0]), ('continue', [0], True), ('stop', [3])])
    research_calls = research_actor(monkeypatch)
    result = research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    dev = result['study_development']
    if target == 'branch':
        Image.new('RGB', (64, 64), 'white').save(dev['branches'][-1]['path'])
    elif target == 'research_response':
        (Path(dev['research'][0]['directory']) / 'response-000.json').unlink()
    else:
        (output / 'study_development' / 'response-005.json').unlink()
    before = len(calls), len(research_calls), len(Renderer.calls)
    with pytest.raises((ValueError, RuntimeError, FileNotFoundError, SpecMismatch)):
        research_render.run_render(output, backend_factory=Renderer, director_factory=ResearchDirector)
    assert before == (len(calls), len(research_calls), len(Renderer.calls))


def test_invalid_cycle_caps_reject_before_planning():
    from kusama.panel_runs import StartRun
    with pytest.raises(ValueError, match='12-branch'):
        StartRun(submission_id='too-many', question='A question', directions=6, images_per_direction=2)
    with pytest.raises(ValueError, match='action budget'):
        cycle.DevelopmentCycle(max_steps=1, research_sessions=2)


@pytest.mark.skipif(os.environ.get('KUSAMA_PROCESSOR_ACCEPTANCE') != '1', reason='Cached processor only; no inference')
def test_cached_processor_admits_cycle_with_branches_and_sources(tmp_path):
    from kusama.director import DirectorConfig
    images = []
    for i in range(8):
        path = tmp_path / f'image-{i}.png'
        Image.new('RGB', (512, 640), (i, 100, 30)).save(path)
        images.append({'id': f'image-{i}', 'path': str(path), 'sha256': sha256(path)})
    observations = {b['id']: {'visible_relationships': ['Two forms share an opening.'],
        'unexpected_effects': [], 'explanation_gaps': ['Agency is uncertain.'], 'uncertainty': 'Interpretation uncertain.',
        'possible_next_question': 'Could the interval be the subject?'} for b in images[:6]}
    payload = {'problem': 'How can a shared boundary shelter people while excluding others?',
        'branches': cycle._branch_context(images[:6], observations), 'source_images': images[6:],
        'allowed_actions': ['continue', 'combine', 'reformulate', 'review', 'stop', 'abandon'],
        'remaining_limits': {'actions': 1, 'research_sessions': 0}, 'output_schema': cycle.CycleChoice.model_json_schema()}
    config = DirectorConfig(Path('.cache/huggingface').resolve(), max_new_tokens=1800, max_image_pixels=65536)
    result = studies.admit_study_context(cycle.CYCLE_SYSTEM, payload, images, config)
    assert result['input_tokens'] <= 6392
    assert len(result['input_images']) == 8
