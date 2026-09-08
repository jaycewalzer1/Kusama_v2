"""Director/editor handoff and recovery contracts; no real models are loaded."""
from copy import deepcopy
from dataclasses import replace
import json

from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.config import Config
from kusama.director import DirectorConfig
from kusama import directed
from kusama.costs import CostJournal


PROPOSAL = {
    'specific_change': 'Turn the white field red.',
    'edit_instruction': 'Turn the white field red. Preserve the image framing.',
    'expected_effect': 'The red field creates a more forceful visual presence.',
    'preservation_constraints': ['Preserve the image framing.'],
}


@pytest.fixture
def source(tmp_path):
    parent = tmp_path / 'source.png'
    Image.new('RGB', (64, 64), 'white').save(parent)
    return parent, Config(output_dir=tmp_path / 'runs', cache_dir=tmp_path / 'cache'), DirectorConfig(cache_dir=tmp_path / 'cache')


def factories(events):
    class Director:
        def __init__(self, config):
            self.config = config
            events.append(('director_created',))

        def prepare(self, on_progress):
            events.append(('director_prepared',))
            on_progress({'device': 'cpu', 'test_double': True})

        def propose(self, parent, intention):
            events.append(('director_called', sha256(parent), intention))
            return {'proposal': deepcopy(PROPOSAL), 'raw_response': json.dumps(PROPOSAL),
                    'input_image_sha256': sha256(parent), 'test_double': True}

        def close(self):
            events.append(('director_closed',))

    class Flux:
        def __init__(self, config):
            assert events[-1] == ('director_closed',), 'Qwen must close before FLUX is loaded'
            self.config = config
            events.append(('flux_created',))

        def prepare(self, on_progress):
            on_progress({'device': 'cpu', 'test_double': True})

        def generate(self, prompt, seed, parent, **kwargs):
            self.on_inference_start()
            events.append(('flux_called', prompt, seed, sha256(parent), kwargs))
            return Image.new('RGB', (self.config.width, self.config.height), 'red')

        def close(self):
            events.append(('flux_closed',))

    return Director, Flux


def forbidden_backend(config):
    pytest.fail('This operation must not construct an inference backend')


def make_packet(source, events=None):
    parent, config, director_config = source
    events = events if events is not None else []
    director, flux = factories(events)
    directory = directed.propose(parent, 'Make this field feel forceful', config,
                                 director_config, seed=41, director_factory=director)
    return directory, flux


def test_one_exact_director_instruction_and_parent_reach_flux_after_director_closes(source):
    events = []
    parent, config, _ = source
    directory, flux = make_packet(source, events)
    packet = json.loads((directory / 'manifest.json').read_text())
    assert packet['status'] == 'direction_ready'
    assert packet['director_calls'] == 1 and packet['model_calls'] == 0
    assert packet['human_labels'] == []
    assert not (directory / 'edit').exists()

    assert directed.execute(directory, config, backend_factory=flux) == directory
    calls = [event for event in events if event[0] == 'flux_called']
    assert calls == [('flux_called', PROPOSAL['edit_instruction'], 41, sha256(parent), {})]
    assert len([event for event in events if event[0] == 'director_called']) == 1
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_review'
    assert manifest['model_calls'] == manifest['director_calls'] == 1
    assert manifest['training_performed'] is False and manifest['frozen_weights'] is True
    assert manifest['human_labels'] == [] and manifest['visual_review'] == 'pending'
    review = json.loads((directory / 'edit' / 'review.json').read_text())
    assert review['human_labels'] == [] and review['status'] == 'pending'
    assert any(PROPOSAL['expected_effect'] in question for question in review['questions'])
    with Image.open(directory / 'comparison.png') as comparison:
        assert comparison.crop((0, 40, 64, 104)).getpixel((0, 0)) == (255, 255, 255)
        assert comparison.crop((64, 40, 128, 104)).getpixel((0, 0)) == (255, 0, 0)


def test_execute_replay_reuses_images_and_review_without_repeating_either_model(source):
    _, config, _ = source
    directory, flux = make_packet(source)
    directed.execute(directory, config, backend_factory=flux)
    paths = [directory / name for name in ('direction.json', 'parent.png', 'edit.png', 'comparison.png')]
    paths += [directory / 'edit' / 'review.json']
    before = {path: path.read_bytes() for path in paths}
    with CostJournal(directory, read_only=True) as costs:
        cost_before = costs.summary('edit')
    assert directed.execute(directory, backend_factory=forbidden_backend) == directory
    assert {path: path.read_bytes() for path in paths} == before
    with CostJournal(directory, read_only=True) as costs:
        assert costs.summary('edit') == cost_before
    assert cost_before['calls'] == 2  # Loading and the one edited-image request.
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['model_calls'] == manifest['director_calls'] == 1
    assert manifest['human_labels'] == []
    with pytest.raises(ValueError, match='frozen settings'):
        directed.execute(directory, replace(config, steps=51), backend_factory=forbidden_backend)


def test_malformed_director_response_retains_raw_and_never_produces_executable_packet(source):
    parent, config, director_config = source
    events = []
    director, _ = factories(events)
    raw = {'raw_response': 'This is not the required proposal JSON.',
           'input_image_sha256': sha256(parent), 'test_double': True}

    class MalformedDirector(director):
        def propose(self, parent, intention):
            error = ValueError('Malformed proposal')
            error.result = raw
            raise error

    with pytest.raises(ValueError, match='Malformed proposal'):
        directed.propose(parent, 'Make this field feel forceful', config, director_config,
                         director_factory=MalformedDirector)
    directory, = (path for path in config.output_dir.iterdir() if path.is_dir())
    assert json.loads((directory / 'direction.json').read_text()) == raw
    assert events[-1] == ('director_closed',)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'director_failed' and manifest['model_calls'] == 0
    assert not (directory / 'plan.json').exists()
    with pytest.raises(ValueError, match='complete saved Qwen direction packet'):
        directed.execute(directory, config, backend_factory=forbidden_backend)


@pytest.mark.parametrize('name', ['parent.png', 'direction.json', 'allowed_mask.png', 'plan.json'])
def test_changed_packet_file_is_refused_before_flux_construction(source, name):
    _, config, _ = source
    directory, _ = make_packet(source)
    path = directory / name
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='Saved direction packet changed'):
        directed.execute(directory, config, backend_factory=forbidden_backend)
    assert not (directory / 'edit').exists()


@pytest.mark.parametrize('missing', ['parent.png', 'direction.json', 'allowed_mask.png', 'plan.json'])
def test_incomplete_packet_hash_mapping_cannot_skip_file_verification(source, missing):
    _, config, _ = source
    directory, _ = make_packet(source)
    path = directory / 'manifest.json'
    manifest = json.loads(path.read_text())
    del manifest['packet_hashes'][missing]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete saved Qwen direction packet'):
        directed.execute(directory, config, backend_factory=forbidden_backend)
    assert not (directory / 'edit').exists()


def test_director_response_for_different_image_is_rejected_and_retained(source):
    parent, config, director_config = source
    events = []
    director, _ = factories(events)

    class WrongParentDirector(director):
        def propose(self, parent, intention):
            response = super().propose(parent, intention)
            response['input_image_sha256'] = '0' * 64
            return response

    with pytest.raises(ValueError, match='different parent'):
        directed.propose(parent, 'Make this field feel forceful', config, director_config,
                         director_factory=WrongParentDirector)
    directory, = (path for path in config.output_dir.iterdir() if path.is_dir())
    assert (directory / 'direction.json').exists()
    assert not (directory / 'plan.json').exists()
    assert events[-1] == ('director_closed',)


@pytest.mark.parametrize('seed', [-1, 2**63, True, 1.0])
def test_invalid_seed_is_rejected_without_creating_a_packet_or_loading_qwen(source, seed):
    parent, config, director_config = source
    with pytest.raises(ValueError, match='Seed must be an integer'):
        directed.propose(parent, 'Make this field feel forceful', config, director_config,
                         seed=seed, director_factory=forbidden_backend)
    assert not config.output_dir.exists()


def test_cleanup_failure_blocks_flux_until_originating_process_has_terminated(source, monkeypatch):
    parent, config, director_config = source
    events = []
    director, flux = factories(events)

    class CleanupFailure(director):
        def close(self):
            super().close()
            raise RuntimeError('Cannot release Qwen memory')

    with pytest.raises(RuntimeError, match='Cannot release Qwen memory'):
        directed.propose(parent, 'Make this field feel forceful', config, director_config,
                         director_factory=CleanupFailure)
    directory, = (path for path in config.output_dir.iterdir() if path.is_dir())
    assert (directory / 'direction.json').exists()
    manifest = json.loads((directory / 'manifest.json').read_text())
    origin = manifest['director_process']

    class ExistingOrigin:
        def __init__(self, pid):
            assert pid == origin['pid']

        def create_time(self):
            return origin['created_at']

    # Only the execution-time process lookup is replaced; proposing recorded
    # the real process identity before this mock was installed.
    monkeypatch.setattr(directed.psutil, 'Process', ExistingOrigin)
    with pytest.raises(ValueError, match='Director cleanup failed'):
        directed.execute(directory, config, backend_factory=forbidden_backend)
    assert not (directory / 'edit').exists()

    def terminated_origin(pid):
        assert pid == origin['pid']
        raise directed.psutil.NoSuchProcess(pid)

    monkeypatch.setattr(directed.psutil, 'Process', terminated_origin)
    assert directed.execute(directory, config, backend_factory=flux) == directory
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == 'needs_review'
    assert manifest['model_calls'] == manifest['director_calls'] == 1
    assert len([event for event in events if event[0] == 'director_called']) == 1


def test_interrupted_reserved_flux_call_remains_charged_in_outer_manifest_and_recovery(source):
    _, config, _ = source
    directory, flux = make_packet(source)

    class InterruptedFlux(flux):
        def generate(self, prompt, seed, parent, **kwargs):
            self.on_inference_start()
            raise KeyboardInterrupt('Interrupted after reservation, before saved output')

    with pytest.raises(KeyboardInterrupt, match='after reservation'):
        directed.execute(directory, config, backend_factory=InterruptedFlux)
    manifest = json.loads((directory / 'manifest.json').read_text())
    child = json.loads((directory / 'edit' / 'manifest.json').read_text())
    assert manifest['model_calls'] == child['model_calls'] == 1
    assert manifest['flux_manifest'] == 'edit/manifest.json'
    assert manifest['status'] == manifest['flux_status'] == child['status'] == 'interrupted'
    assert manifest['flux_runtime_seconds'] == child['runtime_seconds']
    assert not (directory / 'edit' / 'attempt_0_raw.png').exists()

    directed.execute(directory, backend_factory=forbidden_backend)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['status'] == manifest['flux_status'] == 'uncertain'
    assert manifest['model_calls'] == manifest['director_calls'] == 1
    assert manifest['human_labels'] == []
    assert not (directory / 'edit.png').exists()


def test_execute_cli_rejects_new_intention_before_accessing_packet(tmp_path):
    with pytest.raises(SystemExit) as error:
        directed.main(['--execute', str(tmp_path / 'absent'), '--intention', 'Different intention'])
    assert error.value.code == 2
