"""Control integration fixtures verify geometry and accounting, never artistic quality."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from kusama.artifacts import sha256
from kusama.continuation import ContinuationRequest, request_snapshot
from kusama.studio import read_projects
from test_continuation import setup, Backend, forbidden
from test_intention_continuation import mapped


class CropBackend(Backend):
    calls = []

    def generate(self, prompt, seed, parent, **kwargs):
        self.on_inference_start()
        with Image.open(parent) as image, Image.open(kwargs['mask_path']) as mask:
            type(self).calls.append({'seed': seed, 'parent_sha256': sha256(parent),
                                    'input_size': image.size, 'mask_size': mask.size, 'prompt': prompt})
        return Image.new('RGB', (self.config.width, self.config.height), 'red' if seed == 23 else 'green')


def large_parent(setup):
    store, request, config = setup
    parent = Path(request['selection']['path'])
    image = Image.new('RGB', (128, 128), 'white')
    ImageDraw.Draw(image).rectangle((48, 40, 63, 79), fill='blue')
    image.save(parent)
    request['selection']['image_sha256'] = sha256(parent)
    request['revision']['crop'] = [32, 32, 96, 96]
    return store, request, replace(config, width=128, height=128)


def geometric(setup):
    store, request, config = large_parent(setup)
    path = store.root / 'operator_mask.png'
    mask = Image.new('L', (128, 128))
    ImageDraw.Draw(mask).rectangle((48, 40, 63, 79), fill=255)
    mask.save(path)
    request['revision'].update(method='geometric', geometric={
        'object_mask': str(path), 'object_mask_sha256': sha256(path), 'scale': .5,
        'anchor': [56, 80], 'repair_prompt': 'Continue the empty background',
        'repair_policy': {'padding': 4, 'feather': 2}})
    return store, request, config


def test_crop_uses_native_coordinates_same_parent_and_preserves_context(setup):
    store, request, config = large_parent(setup)
    CropBackend.calls = []
    prepared = store.prepare('crop', request, config)
    assert prepared['config']['width'] == prepared['config']['height'] == 128
    assert prepared['preflight']['inference_dimensions'] == [64, 64]
    assert prepared['preflight']['inference_crop'] == [32, 32, 96, 96]
    before = request_snapshot(request)
    result = store.execute('crop', CropBackend)
    assert result['request'] == before
    assert len(CropBackend.calls) == 2
    assert {call['seed'] for call in CropBackend.calls} == {23, 41}
    assert len({call['parent_sha256'] for call in CropBackend.calls}) == 1
    assert all(call['input_size'] == call['mask_size'] == (64, 64) for call in CropBackend.calls)
    assert {artifact['parent_artifact_id'] for artifact in result['outputs']} == {'root'}
    for artifact in result['outputs']:
        with Image.open(artifact['path']) as image:
            assert image.size == (128, 128)
            assert image.getpixel((0, 0)) == (255, 255, 255)
            assert image.getpixel((96, 96)) == (255, 255, 255)
        assert artifact['preservation']['outside_changed_pixels'] == 0
    assert store.execute('crop', forbidden)['model_calls'] == 2
    request['revision']['crop'] = [0, 0, 128, 128]
    with pytest.raises(ValueError, match='frozen'):
        store.prepare('crop', request, config)


@pytest.mark.parametrize('crop', [[0, 0, 64, 64], [32, 32, 80, 96], [32, 32, 97, 96],
                                  [-16, 0, 128, 128], [True, 0, 128, 128], [0, 0, 128.0, 128]])
def test_invalid_crops_never_register_work(setup, crop):
    store, request, config = large_parent(setup)
    request['revision']['crop'] = crop
    with pytest.raises(ValueError):
        store.prepare('bad', request, config)
    assert not (store.root / 'jobs.sqlite3').exists()


def test_whole_cannot_silently_ignore_crop(setup):
    _, request, _ = setup
    request['revision'].update(method='whole', region=None, crop=[0, 0, 64, 64])
    with pytest.raises(ValueError, match='isolation'):
        ContinuationRequest.model_validate(request)


def test_geometric_freezes_masks_and_reuses_after_source_removal(setup):
    store, request, config = geometric(setup)
    CropBackend.calls = []
    prepared = store.prepare('geometry', request, config)
    mask_hash = request['revision']['geometric']['object_mask_sha256']
    frozen = store.root / 'continuations/geometry/object_mask.png'
    assert sha256(frozen) == mask_hash
    assert prepared['preflight']['control_hashes'] == {'object_mask': mask_hash}
    Path(request['revision']['geometric']['object_mask']).unlink()
    Path(request['selection']['path']).unlink()
    result = store.execute('geometry', CropBackend)
    assert result['model_calls'] == len(CropBackend.calls) == 2
    assert all(call['prompt'] == 'Continue the empty background' for call in CropBackend.calls)
    for artifact in result['outputs']:
        with Image.open(artifact['path']) as image:
            assert image.getpixel((56, 70)) == (0, 0, 255)
            assert image.getpixel((0, 0)) == (255, 255, 255)
        assert artifact['preservation']['passed'] and artifact['effective_preservation']['passed']
    project = read_projects(store.root, prepared['project_id'])[0]
    assert {action['parent_sha256'] for action in project['actions'].values()} == {prepared['parent']['sha256']}
    assert store.execute('geometry', forbidden)['model_calls'] == 2
    frozen.write_bytes(b'altered mask')
    with pytest.raises(ValueError, match='hash changed'):
        store.execute('geometry', forbidden)


@pytest.mark.parametrize('problem', ['hash', 'outside', 'scale', 'anchor', 'secondary', 'padding'])
def test_invalid_geometry_rejected_before_job_registration(setup, problem):
    store, request, config = geometric(setup)
    geometry = request['revision']['geometric']
    if problem == 'hash': geometry['object_mask_sha256'] = '0' * 64
    if problem == 'outside': request['revision']['region'] = [.4, .4, .75, .75]
    if problem == 'scale': geometry['scale'] = 4
    if problem == 'anchor': geometry['anchor'] = [float('nan'), 80]
    if problem == 'secondary': geometry['secondary_mask'] = geometry['object_mask']
    if problem == 'padding': geometry['repair_policy'] = {'padding': 1, 'feather': 2}
    with pytest.raises(ValueError):
        store.prepare('bad', request, config)
    assert not (store.root / 'jobs.sqlite3').exists()


def test_target_full_coverage_and_required_support_are_preflight_requirements(setup):
    store, request, config = setup
    intentions = mapped(request)['intention_map']
    intentions['elements'][0]['region'] = [.25, .25, .8, .75]
    with pytest.raises(ValueError, match='entire declared attention target'):
        store.prepare('partial', request, config)
    intentions['elements'][0]['region'] = [.25, .25, .75, .75]
    intentions['elements'][1].update(protect_pixels=False)
    intentions['required_support_ids'] = ['figure']
    with pytest.raises(ValueError, match='entire required support'):
        store.prepare('support', request, config)
    assert not (store.root / 'jobs.sqlite3').exists()


@pytest.mark.parametrize('status', ['present', 'uncertain', None])
def test_existence_attribution_and_uncertainty_survive_freeze(setup, status):
    store, request, config = setup
    intentions = mapped(request)['intention_map']
    if status:
        intentions['elements'][0]['existence'] = {'status': status, 'author': 'fixture observer',
            'origin': 'model_observation', 'reason': 'The boundary is difficult to distinguish'}
    prepared = store.prepare('evidence', request, config)
    record = prepared['preflight']['target_existence'][0]
    assert record['status'] == (status or 'uncertain')
    assert record['evidence'] == intentions['elements'][0].get('existence')
    assert json.loads((store.root / 'continuations/evidence/preflight.json').read_text()) == prepared['preflight']
    assert store.execute('evidence', Backend)['preflight'] == prepared['preflight']


def test_absent_target_requires_add_operation(setup):
    store, request, config = setup
    intentions = mapped(request)['intention_map']
    intentions['elements'][0]['existence'] = {'status': 'absent', 'author': 'fixture operator',
        'origin': 'development_example', 'reason': 'Target absent from the parent'}
    with pytest.raises(ValueError, match='explicitly absent target'):
        store.prepare('absent', request, config)
    assert not (store.root / 'jobs.sqlite3').exists()
    intentions['operation'] = 'add'
    assert store.prepare('add', request, config)['preflight']['target_existence'][0]['status'] == 'absent'


def test_old_frozen_map_policy_is_not_rewritten_on_resume(setup, monkeypatch):
    """Emulate a pre-v2 writer with an overlapping, incompletely covered target."""
    import kusama.continuation as continuation
    store, request, config = setup
    mapped(request)['intention_map']['elements'][0]['region'] = [.25, .25, 1, 1]
    validate = continuation.validate_intention_edit
    with monkeypatch.context() as legacy:
        legacy.setattr(continuation, '_preflight', lambda *args: None)
        legacy.setattr(continuation, 'validate_intention_edit',
            lambda *args, **kwargs: validate(*args, full_coverage=False))
        prepared = store.prepare('legacy', request, config)
    assert 'preflight' not in prepared
    saved = (store.root / 'continuations/legacy/inputs.json').read_bytes()
    assert store.prepare('legacy', request, config) == prepared
    assert store.execute('legacy', Backend)['model_calls'] == 2
    assert (store.root / 'continuations/legacy/inputs.json').read_bytes() == saved
    new_request = deepcopy(request)
    with pytest.raises(ValueError, match='entire declared attention target'):
        store.prepare('new', new_request, config)


def test_request_snapshot_omits_new_defaults(setup):
    _, request, _ = setup
    snapshot = request_snapshot(mapped(request))
    assert 'crop' not in snapshot['revision'] and 'geometric' not in snapshot['revision']
    assert 'required_support_ids' not in snapshot['intention_map']
    assert all('existence' not in element for element in snapshot['intention_map']['elements'])


def test_before_dispatch_denial_prevents_loading_and_any_charge(setup):
    from kusama.ledger import BudgetExceeded
    from kusama.resources import SessionBudget, load_config
    store, request, config = setup
    with SessionBudget(store.root, load_config()):
        pass
    store.prepare('limited', request, config)

    def expired():
        raise BudgetExceeded('Frozen loop wall limit')

    result = store.execute('limited', forbidden, resource_root=store.root, before_dispatch=expired)
    assert result['model_calls'] == 0
    with SessionBudget(store.root) as budget:
        assert budget.summary()['renderer_attempts'] == 0


def test_before_dispatch_rechecks_after_load_before_reservation(setup):
    from kusama.ledger import BudgetExceeded
    from kusama.resources import SessionBudget, load_config
    store, request, config = setup
    with SessionBudget(store.root, load_config()):
        pass
    store.prepare('limited', request, config)
    expired = False
    loaded = []

    class SlowLoad(Backend):
        def prepare(self, on_progress):
            nonlocal expired
            super().prepare(on_progress)
            loaded.append(True)
            expired = True

    def gate():
        if expired:
            raise BudgetExceeded('Frozen loop wall limit')

    result = store.execute('limited', SlowLoad, resource_root=store.root, before_dispatch=gate)
    assert result['model_calls'] == 0 and len(loaded) == 1
    with SessionBudget(store.root) as budget:
        assert budget.summary()['renderer_attempts'] == 0
    assert Backend.calls == []
