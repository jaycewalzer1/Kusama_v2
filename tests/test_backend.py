"""Backend-contract tests use small local doubles, never model downloads."""

from dataclasses import replace
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from kusama.backend import BackendError, FluxBackend, MODEL_ID, MODEL_REVISION
from kusama.config import Config


class TokenizerStub:
    def __init__(self):
        self.template_calls = []
        self.tokenizer_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return "user " + messages[0]["content"] + " end assistant"

    def __call__(self, text, **kwargs):
        self.tokenizer_calls.append((text, kwargs))
        return {"input_ids": list(range(len(text.split())))}


class PipelineStub:
    def __init__(self):
        self.tokenizer = TokenizerStub()
        self.calls = []
        self.inference_modes = []
        self.output = Image.new("RGB", (1024, 1024))

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.inference_modes.append(torch.is_inference_mode_enabled())
        return SimpleNamespace(images=[self.output])


@pytest.fixture
def ready_backend():
    backend = FluxBackend(Config(device="cpu", precision="fp32"))
    backend.pipeline = PipelineStub()
    backend.execution_settings = {"status": "ready", "device": "cpu", "precision": "fp32"}
    return backend


def test_generation_preserves_requested_model_call_and_seed(ready_backend):
    backend = ready_backend
    events = []
    backend.on_inference_start = lambda: events.append(("start", backend.last_pipeline_invoked))
    result = backend.generate("paint a mountain", seed=17)
    call = backend.pipeline.calls[0]
    generator = call.pop("generator")
    assert call == {
        "prompt": "paint a mountain", "height": 1024, "width": 1024,
        "num_inference_steps": 50, "guidance_scale": 4.0,
        "max_sequence_length": 512, "num_images_per_prompt": 1, "output_type": "pil",
    }
    assert generator.device == torch.device("cpu")
    assert generator.initial_seed() == 17
    assert events == [("start", False)]
    assert backend.last_pipeline_invoked is True
    assert backend.pipeline.inference_modes == [True]
    assert result is backend.pipeline.output


def test_edits_reload_saved_parent_and_convert_to_rgb(ready_backend, tmp_path):
    parent = tmp_path / "parent.png"
    Image.new("RGBA", (1024, 1024), (255, 0, 0, 123)).save(parent)
    ready_backend.generate("reduce person", seed=23, parent_path=parent)
    first = ready_backend.pipeline.calls[0]
    assert first["image"].mode == "RGB"
    assert first["image"].getpixel((0, 0)) == (255, 0, 0)
    assert "strength" not in first
    assert first["generator"].initial_seed() == 23

    Image.new("RGB", (1024, 1024), (0, 255, 0)).save(parent)
    ready_backend.generate("reduce person", seed=41, parent_path=parent)
    second = ready_backend.pipeline.calls[1]
    assert second["image"].getpixel((0, 0)) == (0, 255, 0)
    assert first["image"].getpixel((0, 0)) == (255, 0, 0)
    assert second["generator"].initial_seed() == 41


def test_prompt_limit_includes_template_and_rejects_without_invocation(ready_backend):
    backend = ready_backend
    backend.generate("short", seed=17)
    backend.config = replace(backend.config, max_sequence_length=4)
    callbacks = []
    backend.on_inference_start = lambda: callbacks.append("called")
    with pytest.raises(BackendError, match="requires 5 tokens.*no truncation"):
        backend.generate("too long", seed=23)
    assert backend.last_pipeline_invoked is False
    assert backend.last_prompt_token_count == 5
    assert callbacks == []
    assert len(backend.pipeline.calls) == 1
    messages, template_options = backend.pipeline.tokenizer.template_calls[-1]
    assert messages == [{"role": "user", "content": "too long"}]
    assert template_options == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert backend.pipeline.tokenizer.tokenizer_calls[-1][1] == {"truncation": False, "padding": False}


@pytest.mark.parametrize("parent_kind", ["missing", "corrupt", "wrong_size"])
def test_invalid_parent_does_not_spend_inference_budget(ready_backend, tmp_path, parent_kind):
    parent = tmp_path / "parent.png"
    if parent_kind == "corrupt":
        parent.write_bytes(b"not an image")
    elif parent_kind == "wrong_size":
        Image.new("RGB", (64, 64)).save(parent)
    callbacks = []
    ready_backend.on_inference_start = lambda: callbacks.append("called")
    with pytest.raises(BackendError):
        ready_backend.generate("edit person", seed=23, parent_path=parent)
    assert ready_backend.last_pipeline_invoked is False
    assert ready_backend.pipeline.calls == []
    assert callbacks == []


def test_budget_callback_can_prevent_pipeline_invocation(ready_backend):
    def no_budget():
        raise RuntimeError("No calls remaining")
    ready_backend.on_inference_start = no_budget
    with pytest.raises(BackendError, match="No calls remaining"):
        ready_backend.generate("mountain", seed=17)
    assert ready_backend.last_pipeline_invoked is False
    assert ready_backend.pipeline.calls == []


def test_actual_pipeline_failure_is_counted_as_invocation(ready_backend):
    class FailingPipeline(PipelineStub):
        def __call__(self, **kwargs):
            raise RuntimeError("MPS backend out of memory")
    ready_backend.pipeline = FailingPipeline()
    with pytest.raises(BackendError, match="Available memory was exhausted"):
        ready_backend.generate("mountain", seed=17)
    assert ready_backend.last_pipeline_invoked is True


def test_unexpected_output_dimensions_are_returned_for_preservation(ready_backend):
    ready_backend.pipeline.output = Image.new("RGB", (64, 64))
    result = ready_backend.generate("mountain", seed=17)
    assert result.size == (64, 64)


def test_prepare_pins_commit_and_freezes_every_model_component(monkeypatch, tmp_path):
    import diffusers
    import huggingface_hub

    pipeline = PipelineStub()
    pipeline.config = SimpleNamespace(is_distilled=False)
    pipeline.text_encoder = torch.nn.Linear(2, 2)
    pipeline.transformer = torch.nn.Linear(2, 2)
    pipeline.vae = torch.nn.Linear(2, 2)
    pipeline.components = {"text_encoder": pipeline.text_encoder,
                           "transformer": pipeline.transformer,
                           "vae": pipeline.vae, "tokenizer": pipeline.tokenizer}
    pipeline.scheduler = SimpleNamespace(config={"shift": 1.0})
    pipeline.to = lambda device: pipeline
    loading = []

    class Loader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            loading.append((model_id, kwargs))
            return pipeline

    class Hub:
        def model_info(self, model_id, revision):
            assert (model_id, revision) == (MODEL_ID, MODEL_REVISION)
            return SimpleNamespace(sha=MODEL_REVISION, gated=False)

    monkeypatch.setattr(diffusers, "Flux2KleinPipeline", Loader)
    monkeypatch.setattr(huggingface_hub, "HfApi", Hub)
    backend = FluxBackend(Config(device="cpu", precision="fp32", cache_dir=tmp_path))
    progress = []
    settings = backend.prepare(on_progress=progress.append)
    assert loading[0][0] == MODEL_ID
    assert loading[0][1]["revision"] == MODEL_REVISION
    assert loading[0][1]["dtype"] == torch.float32
    assert loading[0][1]["use_safetensors"] is True
    assert settings["resolved_revision"] == MODEL_REVISION
    assert settings["trainable_parameter_count"] == 0
    assert set(settings["frozen_modules"]) == {"text_encoder", "transformer", "vae"}
    for component in (pipeline.text_encoder, pipeline.transformer, pipeline.vae):
        assert component.training is False
        assert all(not parameter.requires_grad for parameter in component.parameters())
    assert [entry["status"] for entry in progress] == ["resolving_revision", "loading_weights", "ready"]
    assert progress[1]["resolved_revision"] == MODEL_REVISION


def test_prepare_reports_loading_failure_before_raising(monkeypatch, tmp_path):
    import huggingface_hub

    class UnavailableHub:
        def model_info(self, model_id, revision):
            raise RuntimeError("Hub unavailable")

    monkeypatch.setattr(huggingface_hub, "HfApi", UnavailableHub)
    backend = FluxBackend(Config(device="cpu", precision="fp32", cache_dir=tmp_path))
    progress = []
    with pytest.raises(BackendError, match="Hub unavailable"):
        backend.prepare(on_progress=progress.append)
    assert [entry["status"] for entry in progress] == ["resolving_revision", "load_failed"]
    assert backend.execution_settings["status"] == "load_failed"
