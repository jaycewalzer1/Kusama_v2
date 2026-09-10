"""Substitute execution tests: no artistic evidence or real model calls."""
from datetime import datetime, timezone
import json

from PIL import Image
import pytest

from kusama import autonomous as a
from kusama.artifacts import sha256
from kusama.config import Config
from kusama.resources import SessionBudget, load_config


class Director:
    loads = 0
    def __init__(self, config):
        self.config = config
    def prepare(self):
        type(self).loads += 1
    def close(self):
        pass


class Backend:
    calls = 0
    def __init__(self, config):
        self.config = config
    def prepare(self, on_progress):
        on_progress({'device': 'cpu', 'test_double': True})
    def generate(self, prompt, seed, parent, **kwargs):
        self.on_inference_start()
        type(self).calls += 1
        return Image.new('RGB', (self.config.width, self.config.height), 'red')
    def close(self):
        pass


def proposal(**updates):
    return {'decision': 'edit', 'reason': 'Test the central relationship',
            'instruction': 'Make the center red', 'intended_change': 'Red center', 'target': 'center',
            'whole_work_plan': 'Retain the white field', 'preserve': ['outside'], 'method': 'masked',
            'region': [.25, .25, .75, .75], 'elements': [{'id': 'center', 'role': 'central form',
                'region': [.3, .3, .7, .7], 'presence': 'present', 'evidence': 'Visible central form'}],
            'target_ids': ['center'], **updates}


def test_coordinate_contract_separates_pixels_from_normalized_model_regions(tmp_path):
    parent = tmp_path / 'parent.png'
    Image.new('RGB', (512, 640), 'white').save(parent)
    requirements = [{'requirement_id': 'figure', 'regions': [
        {'id': 'whole_figure', 'role': 'required_edit', 'box': [155, 20, 350, 640]}]}]
    original = json.dumps(requirements)
    context = a.coordinate_context(requirements, parent)
    assert context['external_regions'][0]['normalized_box'] == [155/512, 20/640, 350/512, 1]
    assert json.dumps(requirements) == original
    valid = proposal(region=context['external_regions'][0]['normalized_box'])
    assert a.Proposal.model_validate(valid).region[-1] == 1
    with pytest.raises(ValueError):
        a.Proposal.model_validate(proposal(region=[155, 20, 350, 640]))
    schema = a.Proposal.model_json_schema()
    region = next(v for v in schema['properties']['region']['anyOf'] if v.get('type') == 'array')
    assert all(v['minimum'] == 0 and v['maximum'] == 1 for v in region['prefixItems'])
    assert all(v['maximum'] == 1 for v in schema['$defs']['Element']['properties']['region']['prefixItems'])


def test_preserves_all_five_constraints_from_live_brief_without_truncation():
    constraints = ['bat', 'wings', 'hands', 'overall_framing', 'top_ornament_detail']
    assert a.Proposal.model_validate(proposal(preserve=constraints)).preserve == constraints
    with pytest.raises(ValueError):
        a.Proposal.model_validate(proposal(preserve=[str(i) for i in range(9)]))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root, output = tmp_path / 'session', tmp_path / 'loop'
    parent = tmp_path / 'parent.png'
    Image.new('RGB', (64, 64), 'white').save(parent)
    config = Config(output_dir=output, cache_dir=tmp_path / 'cache', width=64, height=64,
                    steps=1, precision='fp32', device='cpu', local_files_only=True)
    request = {'id': 'fixture', 'brief_id': 'brief-1', 'project_id': 'project-1', 'authorship': 'fixture',
        'brief': 'Test the central relationship', 'inquiry': {'question': 'What changes?', 'stakes': 'Fixture',
        'position': 'Undetermined', 'viewing_conditions': 'Screen'},
        'parent': {'path': str(parent), 'sha256': sha256(parent)}, 'render_profile_id': 'fixture_64',
        'render_config': config.to_dict(), 'resource_root': str(root), 'fixture': True}
    cfg = load_config().model_copy(update={'session_hours': 24, 'handoff_reserve_minutes': 0,
        'device': 'cpu', 'cache_dir': str(tmp_path / 'cache')})
    with SessionBudget(root, cfg):
        pass
    calls = []
    def model(director, system, request, images, validator, **kwargs):
        calls.append((system, request, images))
        if system == a.PROPOSAL_SYSTEM:
            output = proposal()
        elif system == a.OBSERVATION_SYSTEM:
            output = {'observations': [{'candidate_id': i, 'fulfillment': 'uncertain',
                'visible_changes': ['Red center'], 'uncertainty': 'Fixture only'} for i in request['candidate_ids']]}
        else:
            output = {'selected_id': request['candidate_ids'][1], 'next_action': 'continue',
                'reason': 'Model fixture choice', 'local_assessment': 'Red', 'whole_assessment': 'White field',
                'preservation_assessment': 'Check passed'}
        return {'raw_response': json.dumps(output), 'output': validator(output),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(a, '_multimodal_json', model)
    Director.loads = 0; Backend.calls = 0
    return root, output, request, calls


def test_plan_does_not_start_resource_session(setup, tmp_path):
    _, output, request, _ = setup
    request['resource_root'] = str(tmp_path / 'unopened')
    spec = a.plan_loop(output, request)
    assert spec['max_reasoning_calls'] == 6 and spec['max_renderer_attempts'] == 4
    assert not (tmp_path / 'unopened').exists()
    assert a.plan_loop(output, request) == spec
    with pytest.raises(ValueError, match='admitted'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)


def test_real_models_cannot_run_fixture(setup):
    _, output, request, _ = setup
    a.plan_loop(output, request)
    with pytest.raises(ValueError, match='Fixture'):
        a.run_loop(output)


@pytest.mark.parametrize('fulfillment', ['fail', 'uncertain'])
def test_honest_no_visible_change_reaches_selection(setup, monkeypatch, fulfillment):
    _, output, request, calls = setup
    original = a._multimodal_json
    def no_change(director, system, payload, images, validator, **kwargs):
        if system == a.OBSERVATION_SYSTEM:
            value = {'observations': [{'candidate_id': candidate, 'fulfillment': fulfillment,
                'visible_changes': [], 'unintended_changes': [],
                'uncertainty': 'No requested new connection is discernible.'}
                for candidate in payload['candidate_ids']]}
        elif system == a.SELECTION_SYSTEM:
            value = {'selected_id': 'parent', 'next_action': 'stop',
                'reason': 'Retain the parent because the requested effect is not visible.',
                'local_assessment': 'No connection is discernible.', 'whole_assessment': 'No established gain.',
                'preservation_assessment': 'Outside pixels are preserved.'}
        else:
            return original(director, system, payload, images, validator, **kwargs)
        calls.append((system, payload, images))
        return {'raw_response': json.dumps(value), 'output': validator(value),
                'metadata': {'input_tokens': 100, 'generated_tokens': 100}}
    monkeypatch.setattr(a, '_multimodal_json', no_change)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['status'] == 'model_stop'
    assert result['reasoning_calls'] == 3 and result['renderer_attempts'] == 2
    assert result['selected']['sha256'] == request['parent']['sha256']
    assert all(row['visible_changes'] == [] and row['fulfillment'] == fulfillment
               for row in result['rounds'][0]['observations']['observations'])
    assert sum(system == a.SELECTION_SYSTEM for system, *_ in calls) == 1


def test_two_rounds_saved_parent_lineage_and_no_replay_calls(setup):
    root, output, request, calls = setup
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['renderer_attempts'] == 4 and result['reasoning_calls'] == 6
    assert result['rounds'][1]['parent'] == result['rounds'][0]['selected']
    assert len(calls) == 6 and Backend.calls == 4 and result['human_labels'] == 0
    assert any(label.endswith('_detail') for label, _ in calls[1][2])
    assert all(m['fulfillment'] == 'fail' for m in result['rounds'][1]['mechanical'].values())
    a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert len(calls) == 6 and Backend.calls == 4
    with SessionBudget(root) as budget:
        assert budget.summary()['renderer_attempts'] == 4
        assert not budget.summary()['unsettled']


def test_parent_retention_and_model_stop(setup, monkeypatch):
    _, output, request, _ = setup
    original = a._multimodal_json
    def model(director, system, payload, images, validator, **kwargs):
        if system == a.SELECTION_SYSTEM:
            value = {'selected_id': 'parent', 'next_action': 'stop', 'reason': 'Retain prior work',
                     'local_assessment': 'Uncertain', 'whole_assessment': 'Preserve', 'preservation_assessment': 'Pass'}
            return {'output': validator(value), 'raw_response': json.dumps(value)}
        return original(director, system, payload, images, validator, **kwargs)
    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['status'] == 'model_stop' and result['renderer_attempts'] == 2
    assert result['selected']['sha256'] == request['parent']['sha256']


def test_invalid_proposal_is_charged_and_never_repaired(setup, monkeypatch):
    root, output, request, _ = setup
    monkeypatch.setattr(a, '_multimodal_json', lambda *args, **kwargs: {
        'raw_response': '{}', 'output': None, 'parse_error': 'Incomplete artistic instructions'})
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['status'] == 'proposal_rejected' and Backend.calls == 0
    assert result['reasoning_calls'] == 1
    a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert Director.loads == 1


def test_preflight_failure_reaches_bounded_recovery_without_rendering(setup, monkeypatch):
    _, output, request, _ = setup
    original = a._multimodal_json
    def model(director, system, payload, images, validator, **kwargs):
        if system == a.PROPOSAL_SYSTEM and not payload['prior_rounds']:
            value = proposal(region=[.4, .4, .6, .6])
            return {'output': validator(value), 'raw_response': json.dumps(value)}
        return original(director, system, payload, images, validator, **kwargs)
    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['renderer_attempts'] == 2
    assert result['rounds'][0]['failure_category'] == 'preflight_rejected'
    assert 'entire' in result['rounds'][0]['preflight_error']


def test_invalid_candidate_id_cannot_select_unseen_image(setup, monkeypatch):
    _, output, request, _ = setup
    original = a._multimodal_json
    def model(director, system, payload, images, validator, **kwargs):
        if system == a.SELECTION_SYSTEM:
            value = {'selected_id': 'invented', 'next_action': 'continue', 'reason': 'Fixture',
                'local_assessment': 'Fixture', 'whole_assessment': 'Fixture', 'preservation_assessment': 'Fixture'}
            return {'output': validator(value), 'raw_response': json.dumps(value)}
        return original(director, system, payload, images, validator, **kwargs)
    monkeypatch.setattr(a, '_multimodal_json', model)
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    assert result['status'] == 'selection_rejected'
    assert result['selected']['sha256'] == request['parent']['sha256']


def test_changed_saved_candidate_refuses_resume(setup):
    _, output, request, _ = setup
    a.plan_loop(output, request)
    result = a.run_loop(output, director_factory=Director, backend_factory=Backend)
    Image.new('RGB', (64, 64), 'blue').save(result['rounds'][0]['candidates'][0]['path'])
    with pytest.raises(Exception, match='changed|hash|mismatch'):
        a.run_loop(output, director_factory=Director, backend_factory=Backend)


def test_memory_rejects_future_and_review_inputs(setup):
    _, output, request, _ = setup
    request.update(memory_enabled=True, memory_snapshot={'request': {'purpose': 'heldout'}, 'selected': []})
    with pytest.raises(ValueError, match='development'):
        a.plan_loop(output, request)


def test_pixel_measurement_does_not_claim_semantic_success(tmp_path):
    parent, child = tmp_path / 'a.png', tmp_path / 'b.png'
    Image.new('RGB', (64, 64), 'white').save(parent)
    Image.new('RGB', (64, 64), 'red').save(child)
    checks = a.mechanical_checks(parent, child, None)
    assert checks['fulfillment'] == 'uncertain' and checks['preservation'] == 'uncertain'
    assert a.mechanical_checks(parent, child, [.25, .25, .75, .75])['preservation'] == 'fail'


@pytest.mark.parametrize('method', ['whole', 'masked'])
def test_qwen_selected_reference_survives_continuation_and_controlled_execution(setup, tmp_path, monkeypatch, method):
    import sys
    from pathlib import Path
    from kusama.continuation import ContinuationStore
    _, output, request, calls = setup
    source = tmp_path / 'doll.png'
    Image.new('RGB', (48, 64), 'blue').save(source)
    source_hash = sha256(source)
    request.update(max_rounds=1, seeds=[23, 41], references=[{
        'source_id': 'doll', 'path': str(source), 'sha256': source_hash,
        'context': 'Fixture visual reference; no historical claim'}])
    original_proposal = proposal
    def image_proposal(**updates):
        return original_proposal(method=method, region=None if method == 'whole' else [.25, .25, .75, .75],
            render_images=[{'source_id': 'doll', 'image_index': 0, 'purpose': 'Borrow the forehead contour'}],
            instruction='Change the central shape in image 1 using the contour from image 2')
    monkeypatch.setattr(sys.modules[__name__], 'proposal', image_proposal)
    captured = []
    class ImageBackend(Backend):
        def generate(self, prompt, seed, parent, **kwargs):
            refs = kwargs['reference_images']
            assert len(refs) == 1 and refs[0]['sha256'] == source_hash
            assert Image.open(refs[0]['path']).getpixel((0, 0)) == (0, 0, 255)
            assert Path(refs[0]['path']).is_relative_to(output)
            assert Image.open(parent).getpixel((0, 0)) == (255, 255, 255)
            assert ('mask_path' in kwargs) == (method == 'masked')
            captured.append((prompt, seed, refs))
            return super().generate(prompt, seed, parent, **kwargs)
    a.plan_loop(output, request)
    # The original can change after planning; only preserved input bytes are used.
    Image.new('RGB', (48, 64), 'green').save(source)
    result = a.run_loop(output, director_factory=Director, backend_factory=ImageBackend)
    assert len(captured) == 2
    continuation = ContinuationStore(output).get('fixture_r00')
    assert continuation['request']['revision']['reference_images'][0]['sha256'] == source_hash
    assert all(c['sha256'] != source_hash for c in continuation['outputs'])
    manifests = [json.loads(p.read_text()) for p in output.glob('*controlled*/manifest.json')]
    # Studio uses hashed action job IDs, inspect the saved attempts directly.
    manifests += [json.loads(p.read_text()) for p in output.glob('studio_action_*/manifest.json')]
    assert len(manifests) == 2
    for manifest in manifests:
        inputs = manifest['attempts'][0]['conditioning']['images']
        assert [i['role'] for i in inputs] == ['parent', 'reference']
        assert inputs[1]['input_index'] == 2 and inputs[1]['sha256'] == source_hash
    a.run_loop(output, director_factory=Director, backend_factory=ImageBackend)
    assert len(captured) == 2
