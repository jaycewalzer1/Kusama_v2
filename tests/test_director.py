"""Frozen director contracts, exercised without loading/downloading weights."""

from dataclasses import replace
import hashlib
import json
import sys
from types import SimpleNamespace
import weakref

from PIL import Image
import pytest
import torch

from kusama.backend import BackendError
from kusama.director import (
    DirectorConfig, DirectorOutputError, MODEL_ID, MODEL_REVISION,
    QwenDirector, parse_proposal, validate_proposal,
)


PROPOSAL = {
    "specific_change": "Reduce the foreground figure to one quarter of its current height.",
    "edit_instruction": "Make the foreground figure one quarter as tall, preserving the mountain silhouette, viewpoint, and light.",
    "expected_effect": "The smaller figure should make the landscape feel overwhelmingly vast.",
    "preservation_constraints": ["Mountain silhouette", "Viewpoint and light"],
}


class BatchStub(dict):
    def to(self, device):
        self.device = device
        return self


class ProcessorStub:
    def __init__(self, response=None):
        self.response = json.dumps(PROPOSAL) if response is None else response
        self.image_processor = SimpleNamespace()
        self.template_calls = []
        self.calls = []
        self.decode_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return messages[0]["content"] + "\n<image>\n" + messages[1]["content"][1]["text"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return BatchStub(
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.ones((1, 3), dtype=torch.int64),
            pixel_values=torch.zeros((4, 12)),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )

    def batch_decode(self, ids, **kwargs):
        self.decode_calls.append((ids, kwargs))
        return [self.response]


class ModelStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))
        self.calls = []
        self.inference_modes = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return torch.cat([kwargs["input_ids"], torch.tensor([[7, 8, 9, 10]])], dim=1)


@pytest.fixture
def ready_director(tmp_path):
    director = QwenDirector(DirectorConfig(cache_dir=tmp_path, device="cpu", precision="fp32"))
    director.model = ModelStub().eval().requires_grad_(False)
    director.processor = ProcessorStub()
    director.execution_settings = {"status": "ready", "device": "cpu", "precision": "fp32"}
    return director


def test_proposal_receives_image_and_intention_and_preserves_evidence(ready_director, tmp_path):
    parent = tmp_path / "parent.png"
    Image.new("RGBA", (256, 128), (123, 44, 66, 120)).save(parent)
    intention = "Make this landscape feel overwhelmingly vast."
    result = ready_director.propose(parent, intention)
    processor = ready_director.processor
    model = ready_director.model
    messages, template_kwargs = processor.template_calls[0]
    assert messages[1]["content"][0] == {"type": "image"}
    assert intention in messages[1]["content"][1]["text"]
    assert template_kwargs == {"tokenize": False, "add_generation_prompt": True}
    image = processor.calls[0]["images"][0]
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (123, 44, 66)
    assert processor.calls[0]["truncation"] is False
    assert "pixel_values" in model.calls[0]
    assert "image_grid_thw" in model.calls[0]
    assert model.calls[0]["do_sample"] is False
    assert model.calls[0]["max_new_tokens"] == 512
    assert model.inference_modes == [True]
    assert processor.decode_calls[0][0].tolist() == [[7, 8, 9, 10]]
    assert result["proposal"] == PROPOSAL
    assert result["raw_response"] == json.dumps(PROPOSAL)
    assert result["input_image_sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert result["metadata"]["input_tokens"] == 3
    assert result["metadata"]["generated_tokens"] == 4
    assert result["metadata"]["intention"] == intention
    assert intention in result["prompt"]
    json.dumps(result)  # Every result field can be persisted in a packet.


def test_large_input_is_bounded_without_rewriting_saved_parent(ready_director, tmp_path):
    parent = tmp_path / "large.png"
    Image.new("RGB", (2048, 1024), "red").save(parent)
    before = parent.read_bytes()
    result = ready_director.propose(parent, "Make it vast.")
    width, height = result["metadata"]["director_image_size"]
    assert width * height <= 1048576
    assert result["metadata"]["original_image_size"] == [2048, 1024]
    assert parent.read_bytes() == before


@pytest.mark.parametrize("invalid", [
    {}, {**PROPOSAL, "extra": "unexpected"},
    {**PROPOSAL, "specific_change": " "},
    {**PROPOSAL, "edit_instruction": 123},
    {**PROPOSAL, "edit_instruction": "x" * 2001},
    {**PROPOSAL, "expected_effect": []},
    {**PROPOSAL, "preservation_constraints": []},
    {**PROPOSAL, "preservation_constraints": [" "]},
    {**PROPOSAL, "preservation_constraints": ["x"] * 9},
    {**PROPOSAL, "preservation_constraints": ["x" * 301]},
    {**PROPOSAL, "preservation_constraints": "mountain"},
    [PROPOSAL],
])
def test_schema_rejects_empty_wrong_type_extra_or_unbounded_fields(invalid):
    with pytest.raises(ValueError):
        validate_proposal(invalid)


@pytest.mark.parametrize("raw", [
    "", "I suggest a smaller figure.", "```json\n" + json.dumps(PROPOSAL) + "\n```",
    json.dumps(PROPOSAL) + " trailing commentary", "x" * 16385,
    json.dumps(PROPOSAL)[:-1] + ', "specific_change": "duplicate"}',
])
def test_malformed_response_is_preserved_and_never_fabricated(ready_director, tmp_path, raw):
    parent = tmp_path / "parent.png"
    Image.new("RGB", (64, 64)).save(parent)
    ready_director.processor.response = raw
    with pytest.raises(DirectorOutputError) as failure:
        ready_director.propose(parent, "Make it vast.")
    error = failure.value
    assert error.raw_response == raw
    assert error.result["raw_response"] == raw
    assert error.result["proposal"] is None
    assert error.result["parse_error"]
    assert error.result["input_image_sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert ready_director.last_result is error.result
    assert len(ready_director.model.calls) == 1
    json.dumps(error.result)


def test_valid_parser_does_not_mutate_supplied_proposal():
    value = validate_proposal(PROPOSAL)
    value["preservation_constraints"].append("Only affects the copy")
    assert len(PROPOSAL["preservation_constraints"]) == 2
    assert parse_proposal("\n" + json.dumps(PROPOSAL) + "\n") == PROPOSAL


@pytest.mark.parametrize("input_kind", ["missing", "corrupt", "empty_intention", "long_intention"])
def test_bad_input_never_invokes_qwen(ready_director, tmp_path, input_kind):
    parent = tmp_path / "parent.png"
    intention = "Make it vast."
    if input_kind == "corrupt":
        parent.write_bytes(b"invalid image")
    elif input_kind != "missing":
        Image.new("RGB", (64, 64)).save(parent)
        intention = " " if input_kind == "empty_intention" else "x" * 4001
    with pytest.raises(BackendError):
        ready_director.propose(parent, intention)
    assert ready_director.model.calls == []


def test_missing_image_tensors_cannot_silently_run_text_only(ready_director, tmp_path):
    class TextOnlyProcessor(ProcessorStub):
        def __call__(self, **kwargs):
            return BatchStub(input_ids=torch.tensor([[1, 2, 3]]))
    ready_director.processor = TextOnlyProcessor()
    parent = tmp_path / "parent.png"
    Image.new("RGB", (64, 64)).save(parent)
    with pytest.raises(BackendError, match="did not include image tensors"):
        ready_director.propose(parent, "Make it vast.")
    assert ready_director.model.calls == []


def test_inference_failure_retains_prompt_and_parent_without_retry(ready_director, tmp_path):
    class FailingModel(ModelStub):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            raise RuntimeError("out of memory")
    ready_director.model = FailingModel()
    parent = tmp_path / "parent.png"
    Image.new("RGB", (64, 64)).save(parent)
    with pytest.raises(BackendError, match="out of memory") as failure:
        ready_director.propose(parent, "Make it vast.")
    result = failure.value.result
    assert "Make it vast." in result["prompt"]
    assert result["raw_response"] is None
    assert result["proposal"] is None
    assert result["input_image_sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert len(ready_director.model.calls) == 1
    json.dumps(result)


def test_large_token_input_is_rejected_before_inference(ready_director, tmp_path):
    class LongInputProcessor(ProcessorStub):
        def __call__(self, **kwargs):
            inputs = super().__call__(**kwargs)
            inputs["input_ids"] = torch.zeros((1, 8193), dtype=torch.int64)
            return inputs
    ready_director.processor = LongInputProcessor()
    parent = tmp_path / "parent.png"
    Image.new("RGB", (64, 64)).save(parent)
    with pytest.raises(BackendError, match="no truncation"):
        ready_director.propose(parent, "Make it vast.")
    assert ready_director.model.calls == []


def test_prepare_verifies_sha_and_freezes_all_parameters(monkeypatch, tmp_path):
    model = ModelStub()
    processor = ProcessorStub()
    loading = []

    class ModelLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loading.append(("model", model_id, kwargs))
            return model

    class ProcessorLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loading.append(("processor", model_id, kwargs))
            return processor

    class Hub:
        def model_info(self, model_id, revision):
            assert (model_id, revision) == (MODEL_ID, MODEL_REVISION)
            return SimpleNamespace(sha=MODEL_REVISION, gated=False)

    # Replace modules, since Transformers' lazy exports can overwrite attribute
    # patches during a later import. This guarantees no real loader is reachable.
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3VLForConditionalGeneration=ModelLoader, AutoProcessor=ProcessorLoader,
    ))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=Hub, hf_hub_download=lambda *args, **kwargs: None,
    ))
    director = QwenDirector(DirectorConfig(cache_dir=tmp_path, device="cpu", precision="fp32"))
    progress = []
    settings = director.prepare(progress.append)
    assert settings["resolved_revision"] == MODEL_REVISION
    assert settings["trainable_parameter_count"] == 0
    assert settings["training_enabled"] is False
    assert model.training is False
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert [entry["status"] for entry in progress] == ["resolving_revision", "loading_weights", "ready"]
    for component, model_id, kwargs in loading:
        assert model_id == MODEL_ID
        assert kwargs["revision"] == MODEL_REVISION
        assert kwargs["trust_remote_code"] is False
        if component == "processor":
            assert kwargs["use_fast"] is False
        else:
            assert kwargs["dtype"] == torch.float32
            assert kwargs["use_safetensors"] is True
    assert director.prepare() == settings
    assert len(loading) == 2


def test_prepare_rejects_wrong_remote_revision_before_loading(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40, gated=False)),
        hf_hub_download=lambda *args, **kwargs: None,
    ))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        Qwen3VLForConditionalGeneration=None, AutoProcessor=None,
    ))
    director = QwenDirector(DirectorConfig(cache_dir=tmp_path, device="cpu", precision="fp32"))
    progress = []
    with pytest.raises(BackendError, match="unexpected commit"):
        director.prepare(progress.append)
    assert director.model is None
    assert director.processor is None
    assert [entry["status"] for entry in progress] == ["resolving_revision", "load_failed"]


@pytest.mark.parametrize("changes", [
    {"model_revision": "main"}, {"device": "unknown"}, {"precision": "int8"},
    {"max_new_tokens": 0}, {"max_new_tokens": True}, {"max_new_tokens": 2049},
    {"max_image_pixels": 4095}, {"max_image_pixels": 4194305},
])
def test_invalid_configuration_rejected_before_model_loading(tmp_path, changes):
    config = replace(DirectorConfig(cache_dir=tmp_path), **changes)
    with pytest.raises(BackendError):
        QwenDirector(config).prepare()


def test_close_releases_model_and_processor_before_flux(ready_director, tmp_path):
    model_reference = weakref.ref(ready_director.model)
    processor_reference = weakref.ref(ready_director.processor)
    ready_director.close()
    assert ready_director.model is None
    assert ready_director.processor is None
    assert ready_director.execution_settings["status"] == "closed"
    assert model_reference() is None
    assert processor_reference() is None
    ready_director.close()  # Safe for failure cleanup and repeated close calls.
    with pytest.raises(BackendError, match="prepare"):
        ready_director.propose(tmp_path / "parent.png", "Make it vast.")


# Reference-aware generation and observation share the original frozen loader.
class ReferenceProcessorStub(ProcessorStub):
    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return json.dumps(messages, ensure_ascii=False)


def reference_packet(tmp_path):
    for name, color in [('first.png', 'red'), ('second.png', 'blue')]:
        Image.new('RGB', (64, 64), color).save(tmp_path / name)
    return {'schema_version': 1, 'id': 'test', 'title': 'History and continuity',
        'brief': 'Paint the feeling of interrupted cultural memory without claiming a culture is extinct.',
        'references': [{'id': name, 'title': name, 'url': 'https://example.org/' + name,
            'provenance': 'Source metadata.', 'permitted_uses': 'Test fixture.', 'source_family': 'Museum',
            'context': 'A sourced statement.', 'relationship_to_brief': 'A proposed pictorial analogy.',
            'image_path': path} for name, path in [('first', 'first.png'), ('second', 'second.png')]],
        'creative_constraints': ['Do not invent glyph meanings.']}


def direction_output():
    return {'directions': [{'id': 'd01', 'title': 'The gap', 'core_move': 'Paint a gap inside a continuous surface.',
        'flux_prompt': 'A painting of a self-possessed figure beside a gap in a layered blue surface.',
        'expected_effect': 'The interrupted surface should evoke absence within continuity.',
        'source_relationships': [{'source_id': 'first', 'relationship': 'Transform layering into visible interruption.'}]}]}


def test_reference_director_receives_multiple_images_context_and_exact_provenance(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    ready_director.processor = ReferenceProcessorStub(json.dumps(direction_output()))
    result = ready_director.propose_directions(packet, tmp_path, ['d01'])
    assert result['output'] == direction_output()
    assert len(ready_director.processor.calls[0]['images']) == 2
    assert len(result['metadata']['input_images']) == 2
    assert result['metadata']['input_images'][0]['sha256'] == hashlib.sha256((tmp_path / 'first.png').read_bytes()).hexdigest()
    assert result['metadata']['request']['packet']['references'][0]['url'] == 'https://example.org/first'
    assert result['metadata']['request']['requested_direction_count'] == 1
    assert result['metadata']['request']['requested_direction_ids'] == ['d01']
    assert 'Return exactly 1' in result['metadata']['request']['final_output_instruction']
    assert 'Stop immediately' in result['metadata']['request']['final_output_instruction']
    assert 'image_path' not in result['metadata']['request']['packet']['references'][0]
    assert packet['brief'] in result['prompt']
    assert ready_director.model.inference_modes == [True]
    assert ready_director.model.calls[0]['do_sample'] is False
    assert ready_director.processor.calls[0]['truncation'] is False


def test_reference_director_rejects_unknown_sources_and_preserves_raw_text(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    output = direction_output()
    output['directions'][0]['source_relationships'][0]['source_id'] = 'invented'
    ready_director.processor = ReferenceProcessorStub(json.dumps(output))
    with pytest.raises(DirectorOutputError) as error:
        ready_director.propose_directions(packet, tmp_path, ['d01'])
    assert error.value.result['raw_response'] == json.dumps(output)
    assert error.value.result['output'] is None
    assert len(ready_director.model.calls) == 1


def test_reference_director_rejects_duplicate_interventions_across_batches(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    output = direction_output()
    prior = [{**output['directions'][0], 'id': 'd00', 'title': 'A previous title'}]
    ready_director.processor = ReferenceProcessorStub(json.dumps(output))
    with pytest.raises(DirectorOutputError, match='Duplicate direction core_move'):
        ready_director.propose_directions(packet, tmp_path, ['d01'], prior=prior)


def test_reference_director_rejects_over_budget_before_model_inference(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    class LongReferenceProcessor(ReferenceProcessorStub):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result['input_ids'] = torch.zeros((1, 8193), dtype=torch.int64)
            return result
    ready_director.processor = LongReferenceProcessor()
    with pytest.raises(BackendError, match='no truncation') as error:
        ready_director.propose_directions(packet, tmp_path, ['d01'])
    assert error.value.result['metadata']['input_tokens'] == 8193
    assert ready_director.model.calls == []


def test_observer_separates_finished_quality_from_potential_and_binds_images(ready_director, tmp_path):
    reference_packet(tmp_path)
    observations = {'visible_observations': [
        {'candidate_id': 'a', 'observations': ['A red field is visible.']},
        {'candidate_id': 'b', 'observations': ['A blue field is visible.']}],
        'finished_quality': {'preference': 'candidate_a', 'reason': 'The surface is coherent.'},
        'development_potential': {'preference': 'candidate_b', 'reason': 'The open field could carry a strong intervention.'},
        'revision': {'candidate_id': 'b', 'instruction': 'Add one gap in the field; preserve the blue color.',
                     'expected_effect': 'The interruption might make absence legible.'},
        'uncertainty': 'The flat fields do not establish cultural meaning.'}
    ready_director.processor = ReferenceProcessorStub(json.dumps(observations))
    result = ready_director.observe_candidates('Paint interrupted memory.', direction_output()['directions'][0],
        [{'id': 'a', 'path': 'first.png'}, {'id': 'b', 'path': 'second.png'}], tmp_path)
    assert result['output'] == observations
    assert result['metadata']['request']['candidate_a'] == 'a'
    assert [i['id'] for i in result['metadata']['input_images']] == ['a', 'b']
    assert 'human' in result['metadata']['request']['human_authority'].lower()
    assert 'advancement' not in result['output']


def test_prior_context_uses_documented_compact_excerpts(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    ready_director.processor = ReferenceProcessorStub(json.dumps(direction_output()))
    prior = [{'id': 'd00', 'title': 'Prior intervention', 'core_move': 'A' * 800,
              'flux_prompt': 'A distinct previous painting.'}]
    result = ready_director.propose_directions(packet, tmp_path, ['d01'], prior=prior)
    request = result['metadata']['request']
    assert request['prior_directions'] == [{'id': 'd00', 'title': 'Prior intervention', 'core_move_excerpt': 'A' * 300}]
    assert len(prior[0]['core_move']) == 800  # Full original is not mutated.
    assert '300' in request['prior_context_policy']


def observer_output():
    return {'visible_observations': [
        {'candidate_id': 'a', 'observations': ['A red field is visible.']},
        {'candidate_id': 'b', 'observations': ['A blue field is visible.']}],
        'finished_quality': {'preference': 'candidate_a', 'reason': 'The surface is coherent.'},
        'development_potential': {'preference': 'candidate_b', 'reason': 'An open field leaves room for an intervention.'},
        'revision': {'candidate_id': 'b', 'instruction': 'Add one gap in the field while preserving its blue color.',
                     'expected_effect': 'The gap might make absence legible.'},
        'uncertainty': 'Cultural meaning cannot be established from these pixels alone.'}


def test_observer_accepts_real_qwen_prose_length_with_exact_bindings(ready_director, tmp_path):
    from pathlib import Path
    reference_packet(tmp_path)
    fixture = json.loads((Path(__file__).parent / 'fixtures/qwen_maya_single_direction_response.json').read_text())
    # A schema/length analogue using real 917-character Qwen prose. This is not
    # presented as an actual observation of the synthetic test images.
    prose = json.loads(fixture['raw_response'])['directions'][0]['core_move']
    assert len(prose) == 917
    output = observer_output()
    output['visible_observations'][0]['observations'][0] = prose
    output['finished_quality']['reason'] = prose
    output['development_potential']['reason'] = prose
    output['revision']['instruction'] = prose
    output['revision']['expected_effect'] = prose
    output['uncertainty'] = prose
    ready_director.processor = ReferenceProcessorStub(json.dumps(output))
    result = ready_director.observe_candidates('Study the painted surfaces.', direction_output()['directions'][0],
        [{'id': 'a', 'path': 'first.png'}, {'id': 'b', 'path': 'second.png'}], tmp_path)
    assert result['output'] == output
    instruction = result['metadata']['request']['final_output_instruction']
    assert '["a", "b"]' in instruction
    assert 'candidate_a, candidate_b, tie or unsure' in instruction
    assert 'Stop immediately' in instruction
    assert ready_director.model.inference_modes == [True]


@pytest.mark.parametrize('violation', ['wrong_id', 'wrong_order', 'wrong_preference', 'wrong_revision', 'extra_candidate', 'five_points', 'extra_key'])
def test_observer_retains_semantic_invariants(ready_director, tmp_path, violation):
    reference_packet(tmp_path)
    output = observer_output()
    if violation == 'wrong_id':
        output['visible_observations'][0]['candidate_id'] = 'unknown'
    elif violation == 'wrong_order':
        output['visible_observations'].reverse()
    elif violation == 'wrong_preference':
        output['finished_quality']['preference'] = 'a'  # Actual image ID is not a preference enum.
    elif violation == 'wrong_revision':
        output['revision']['candidate_id'] = 'unknown'
    elif violation == 'extra_candidate':
        output['visible_observations'].append({'candidate_id': 'c', 'observations': ['An extra image.']})
    elif violation == 'five_points':
        output['visible_observations'][0]['observations'] *= 5
    else:
        output['human_vote'] = 'A'
    raw = json.dumps(output)
    ready_director.processor = ReferenceProcessorStub(raw)
    with pytest.raises(DirectorOutputError) as error:
        ready_director.observe_candidates('Study the surfaces.', direction_output()['directions'][0],
            [{'id': 'a', 'path': 'first.png'}, {'id': 'b', 'path': 'second.png'}], tmp_path)
    assert error.value.raw_response == raw
    assert error.value.result['output'] is None
    assert len(ready_director.model.calls) == 1


def test_observer_global_raw_response_cap_remains_enforced(ready_director, tmp_path):
    reference_packet(tmp_path)
    output = observer_output()
    output['uncertainty'] = 'x' * 16385
    ready_director.processor = ReferenceProcessorStub(json.dumps(output))
    with pytest.raises(DirectorOutputError, match='bounded output contract'):
        ready_director.observe_candidates('Study the surfaces.', direction_output()['directions'][0],
            [{'id': 'a', 'path': 'first.png'}, {'id': 'b', 'path': 'second.png'}], tmp_path)


def test_composition_challenge_is_current_only_attributed_and_reduces_prior_anchoring(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    packet['composition_challenges'] = {'d01': 'Make absence legible through a panoramic empty landscape.',
                                        'd02': 'A distinct challenge that must not reach this request.'}
    prior = [{'id': 'd00', 'title': 'Old figure composition', 'core_move': 'Secret earlier body-and-veil description.',
              'flux_prompt': 'A previous painting prompt.'}]
    ready_director.processor = ReferenceProcessorStub(json.dumps(direction_output()))
    result = ready_director.propose_directions(packet, tmp_path, ['d01'], prior=prior)
    request = result['metadata']['request']
    assert request['assigned_composition_challenges'] == {'d01': packet['composition_challenges']['d01']}
    assert 'composition_challenges' not in request['packet']
    assert packet['composition_challenges']['d02'] not in result['prompt']
    assert prior[0]['core_move'] not in result['prompt']
    assert request['prior_directions'] == [{'id': 'd00', 'title': 'Old figure composition'}]
    assert request['composition_challenge_provenance']['origin'] == 'operator_authored_scaffold'
    assert list(request)[-2:] == ['assigned_composition_challenges', 'final_output_instruction']
    assert 'governing pictorial structure' in request['final_output_instruction']
    assert packet['composition_challenges']['d02']  # Complete source map is not modified.


def test_missing_current_challenge_fails_before_qwen_inference(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    packet['composition_challenges'] = {'d02': 'Only a second assignment.'}
    with pytest.raises(ValueError, match='missing requested direction IDs'):
        ready_director.propose_directions(packet, tmp_path, ['d01'])
    assert ready_director.model.calls == []


def test_scaffold_does_not_disable_full_prior_duplicate_validation(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    packet['composition_challenges'] = {'d01': 'Use a new spatial structure.'}
    output = direction_output()
    prior = [{**output['directions'][0], 'id': 'd00', 'title': 'Old different title'}]
    ready_director.processor = ReferenceProcessorStub(json.dumps(output))
    with pytest.raises(DirectorOutputError, match='Duplicate direction core_move'):
        ready_director.propose_directions(packet, tmp_path, ['d01'], prior=prior)
    assert 'core_move' not in ready_director.last_result['metadata']['request']['prior_directions'][0]


def test_legacy_empty_challenge_map_preserves_prompt_context(ready_director, tmp_path):
    packet = reference_packet(tmp_path)
    ready_director.processor = ReferenceProcessorStub(json.dumps(direction_output()))
    without = ready_director.propose_directions(packet, tmp_path, ['d01'])
    packet['composition_challenges'] = {}
    with_empty = ready_director.propose_directions(packet, tmp_path, ['d01'])
    assert without['prompt'] == with_empty['prompt']
    assert without['metadata']['request'] == with_empty['metadata']['request']


def test_observer_prompt_explicitly_models_two_separate_observation_entries(ready_director, tmp_path):
    from kusama.director import OBSERVER_SYSTEM_PROMPT
    reference_packet(tmp_path)
    ready_director.processor = ReferenceProcessorStub(json.dumps(observer_output()))
    result = ready_director.observe_candidates('Study both surfaces.', direction_output()['directions'][0],
        [{'id': 'a', 'path': 'first.png'}, {'id': 'b', 'path': 'second.png'}], tmp_path)
    structure = result['metadata']['request']['required_visible_observations_structure']
    assert len(structure) == 2 and [entry['candidate_id'] for entry in structure] == ['a', 'b']
    assert 'one or two brief observations of candidate_a' in OBSERVER_SYSTEM_PROMPT
    assert 'one or two brief observations of candidate_b' in OBSERVER_SYSTEM_PROMPT
