import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from kusama.artifacts import sha256
from kusama.config import Config
from kusama.controlled import load_plan, preservation, run


@pytest.fixture
def plan(tmp_path):
    parent = Image.new('RGB', (128, 128), 'white')
    ImageDraw.Draw(parent).rectangle((52, 40, 67, 95), fill='blue')
    parent.save(tmp_path / 'parent.png')
    allowed = Image.new('L', parent.size)
    ImageDraw.Draw(allowed).rectangle((32, 24, 95, 111), fill=255)
    allowed.save(tmp_path / 'allowed.png')
    obj = Image.new('L', parent.size)
    ImageDraw.Draw(obj).rectangle((52, 40, 67, 95), fill=255)
    obj.save(tmp_path / 'object.png')
    data = dict(parent='parent.png', parent_sha256=sha256(tmp_path / 'parent.png'),
                instruction='Shrink the person', target='person', allowed_mask='allowed.png',
                method='auto', seed=23, max_calls=2, visual_checks=['Does the scene look natural?'])
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(data))
    return path, data


def update(plan, **kwargs):
    path, data = plan
    data.update(kwargs)
    path.write_text(json.dumps(data))
    return path


class FakeBackend:
    def __init__(self, config):
        self.config = config
    def prepare(self, on_progress=None):
        on_progress({'device': 'cpu', 'test_double': True})
    def generate(self, prompt, seed, parent_path, **kwargs):
        self.on_inference_start()
        return Image.new('RGB', (self.config.width, self.config.height), 'red')
    def close(self):
        pass


def execute(path):
    output = run(path, Config(output_dir=path.parent / 'runs'), FakeBackend)
    return output, json.loads((output / 'manifest.json').read_text())


def test_masked_preserves_exact_pixels_despite_model_drift(plan):
    output, manifest = execute(plan[0])
    attempt = manifest['attempts'][0]
    assert manifest['status'] == 'needs_review'
    assert attempt['raw_preservation']['outside_changed_pixels'] > 0
    assert attempt['preservation']['outside_changed_pixels'] == 0
    assert manifest['model_calls'] == 1
    assert json.loads((output / 'review.json').read_text())['status'] == 'pending'


def test_whole_drift_rejected_then_masked_fallback(plan):
    _, manifest = execute(update(plan, method='whole', fallbacks=['masked']))
    assert [a['status'] for a in manifest['attempts']] == ['rejected', 'needs_review']
    assert manifest['model_calls'] == 2


def test_budget_stops_fallback(plan):
    _, manifest = execute(update(plan, method='whole', fallbacks=['masked'], max_calls=1))
    assert manifest['status'] == 'budget_exhausted'
    assert len(manifest['attempts']) == 1


def test_geometry_keeps_anchor_and_original_color(plan):
    path = update(plan, object_mask='object.png', scale=0.5, anchor=[60, 96], repair_prompt='Empty background')
    output, manifest = execute(path)
    assert manifest['methods'] == ['geometric']
    attempt = manifest['attempts'][0]
    assert attempt['geometry']['transformed_core_bbox'] == [56, 68, 64, 96]
    assert attempt['preservation']['passed']
    result = Image.open(output / manifest['candidate'])
    assert result.getpixel((60, 80)) == (0, 0, 255)
    assert result.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize('changes', [dict(parent_sha256='wrong'), dict(max_calls=True),
    dict(method='unknown'), dict(fallbacks=['unknown']), dict(crop=[0, 0, 64, 64]),
    dict(scale=float('nan'), anchor=[60, 96]), dict(unrecognized=True),
    dict(object_mask='object.png', scale=3, anchor=[60, 96], repair_prompt='Remove object')])
def test_invalid_plans_rejected_before_inference(plan, changes):
    with pytest.raises(ValueError):
        load_plan(update(plan, **changes))


def test_soft_masks_rejected(plan):
    Image.new('L', (128, 128), 127).save(plan[0].parent / 'allowed.png')
    with pytest.raises(ValueError):
        load_plan(plan[0])


def test_wrong_size_raw_retained(plan):
    class WrongSize(FakeBackend):
        def generate(self, *args, **kwargs):
            self.on_inference_start()
            return Image.new('RGB', (64, 64))
    output = run(plan[0], Config(output_dir=plan[0].parent / 'runs'), WrongSize)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'rejected'
    assert (output / 'attempt_0_raw.png').exists()
    assert 'candidate' not in manifest


def test_preflight_failure_is_not_a_model_call(plan):
    class Failure(FakeBackend):
        def generate(self, *args, **kwargs):
            raise ValueError('prompt too long')
    output = run(plan[0], Config(output_dir=plan[0].parent / 'runs'), Failure)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['model_calls'] == 0
    assert manifest['attempts'][0]['status'] == 'failed'


def test_unchanged_output_is_not_an_edit(plan):
    class NoOp(FakeBackend):
        def generate(self, prompt, seed, parent_path, **kwargs):
            self.on_inference_start()
            return Image.open(parent_path).convert('RGB')
    output = run(plan[0], Config(output_dir=plan[0].parent / 'runs'), NoOp)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'rejected'
    assert manifest['attempts'][0]['preservation']['passed']
    assert not manifest['attempts'][0]['nonempty_edit']


def test_invocation_failure_consumes_budget(plan):
    class Failure(FakeBackend):
        def generate(self, *args, **kwargs):
            self.on_inference_start()
            raise RuntimeError('inference failure')
    path = update(plan, fallbacks=['masked'], max_calls=1)
    output = run(path, Config(output_dir=path.parent / 'runs'), Failure)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'budget_exhausted'
    assert manifest['model_calls'] == 1
    assert len(manifest['attempts']) == 1


def test_interruption_preserves_attempt_and_call_count(plan):
    class Interrupted(FakeBackend):
        def generate(self, *args, **kwargs):
            self.on_inference_start()
            raise KeyboardInterrupt('stopped')
    output = plan[0].parent / 'runs'
    with pytest.raises(KeyboardInterrupt):
        run(plan[0], Config(output_dir=output), Interrupted)
    manifest = json.loads(next(output.glob('*/manifest.json')).read_text())
    assert manifest['status'] == 'interrupted'
    assert manifest['attempts'][0]['status'] == 'interrupted'
    assert manifest['model_calls'] == 1


def test_tight_mask_preserves_background_inside_outer_permission(plan):
    path = update(plan, object_mask='object.png', repair_policy={'padding': 4, 'feather': 2})
    output, manifest = execute(path)
    result = Image.open(output / manifest['candidate'])
    assert result.getpixel((40, 30)) == (255, 255, 255)
    assert result.getpixel((55, 50)) == (255, 0, 0)
    assert manifest['attempts'][0]['effective_preservation']['passed']


def test_secondary_mask_cannot_expand_outer_permission(plan):
    Image.new('L', (128, 128), 255).save(plan[0].parent / 'secondary.png')
    with pytest.raises(ValueError, match='Secondary'):
        load_plan(update(plan, secondary_mask='secondary.png'))


def test_independent_failure_triggers_declared_fallback(plan, monkeypatch):
    import kusama.controlled as controlled
    outcomes = iter(['fail', 'pass'])
    monkeypatch.setattr(controlled, 'verify', lambda *args: ({'status': next(outcomes)}, None))
    output, manifest = execute(update(plan, fallbacks=['masked']))
    assert [a['status'] for a in manifest['attempts']] == ['rejected', 'needs_review']
    assert manifest['model_calls'] == 2
    evidence = json.loads((output / 'attempt_0_verification.json').read_text())
    assert evidence['image_sha256'] == sha256(output / 'attempt_0_result.png')


def test_observer_failure_is_uncertain_not_pass(plan, monkeypatch):
    import kusama.controlled as controlled
    def unavailable(*args):
        raise OSError('detector unavailable')
    monkeypatch.setattr(controlled, 'verify', unavailable)
    _, manifest = execute(plan[0])
    assert manifest['attempts'][0]['verification']['status'] == 'uncertain'
    assert 'detector unavailable' in manifest['attempts'][0]['verification']['error']


def test_whole_method_must_respect_tight_scope_even_inside_outer_mask(plan):
    class NearbyDrift(FakeBackend):
        def generate(self, prompt, seed, parent_path, **kwargs):
            self.on_inference_start()
            image = Image.open(parent_path).convert('RGB')
            image.putpixel((40, 30), (255, 0, 0))
            return image
    path = update(plan, method='whole', object_mask='object.png', repair_policy={'padding': 4, 'feather': 2})
    output = run(path, Config(output_dir=path.parent / 'runs'), NearbyDrift)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['attempts'][0]['preservation']['passed']
    assert not manifest['attempts'][0]['effective_preservation']['passed']
    assert manifest['status'] == 'rejected'
