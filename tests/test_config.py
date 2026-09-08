"""Configuration precedence and reproducibility limits require no model weights."""

import pytest

from kusama.config import Config, EDIT_INSTRUCTION, MODEL_REVISION, PARENT_PROMPT, parse_args


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    import os
    for key in os.environ:
        if key.startswith("KUSAMA_") or key in {"HF_TOKEN", "PYTHON_DOTENV_DISABLED"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults_match_experiment_request(tmp_path):
    config, check = parse_args([])
    assert not check
    assert config.cache_dir == tmp_path / ".cache/huggingface"
    assert config.output_dir == tmp_path / "artifacts"
    assert (config.width, config.height, config.steps, config.guidance) == (1024, 1024, 50, 4.0)
    assert config.parent_seed == 17
    assert config.edit_seeds == (23, 41)
    assert config.prompt == PARENT_PROMPT
    assert config.edit_instruction == EDIT_INSTRUCTION
    assert config.model_revision == MODEL_REVISION
    assert config.precision == config.device == "auto"
    assert config.offload == "none"
    assert config.max_retries == 0


def test_every_documented_environment_setting_is_read(tmp_path, monkeypatch):
    values = {
        "KUSAMA_CACHE_DIR": "custom-cache", "KUSAMA_OUTPUT_DIR": "custom-output",
        "KUSAMA_DEVICE": "cuda", "KUSAMA_PRECISION": "fp16",
        "KUSAMA_OFFLOAD": "sequential", "KUSAMA_VAE_TILING": "true",
        "KUSAMA_MODEL_REVISION": "a" * 40, "KUSAMA_LOCAL_FILES_ONLY": "true",
        "HF_TOKEN": "synthetic-test-token",
    }
    (tmp_path / ".env").write_text("\n".join(f"{key}={value}" for key, value in values.items()))
    config, _ = parse_args([])
    assert config.cache_dir == tmp_path / "custom-cache"
    assert config.output_dir == tmp_path / "custom-output"
    assert config.device == "cuda"
    assert config.precision == "fp16"
    assert config.offload == "sequential"
    assert config.vae_tiling and config.local_files_only
    assert config.model_revision == "a" * 40
    import os
    assert os.getenv("HF_TOKEN") == "synthetic-test-token"
    assert "synthetic-test-token" not in repr(config.to_dict())


def test_cli_overrides_all_environment_config_and_process_env_overrides_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("KUSAMA_OUTPUT_DIR=from-dotenv\nKUSAMA_PRECISION=bf16\n")
    monkeypatch.setenv("KUSAMA_OUTPUT_DIR", "from-process")
    config, _ = parse_args([])
    assert config.output_dir == tmp_path / "from-process"
    for key, value in {
        "KUSAMA_CACHE_DIR": "env-cache", "KUSAMA_DEVICE": "cuda",
        "KUSAMA_OFFLOAD": "sequential", "KUSAMA_VAE_TILING": "true",
        "KUSAMA_MODEL_REVISION": "a" * 40, "KUSAMA_LOCAL_FILES_ONLY": "true",
    }.items():
        monkeypatch.setenv(key, value)
    config, check = parse_args([
        "--cache-dir", "cli-cache", "--output-dir", "cli-output",
        "--device", "cpu", "--precision", "fp32", "--offload", "none",
        "--no-vae-tiling", "--no-local-files-only", "--model-revision", "b" * 40,
        "--prompt", "Custom parent", "--edit-instruction", "Custom edit",
        "--parent-seed", "7", "--edit-seeds", "8", "9", "--width", "512",
        "--height", "768", "--steps", "12", "--guidance", "3.0",
        "--max-sequence-length", "256", "--parent-image", "input.png",
        "--max-retries", "2", "--note", "Local experiment", "--check-environment",
    ])
    assert check
    assert config == Config(
        cache_dir=tmp_path / "cli-cache", output_dir=tmp_path / "cli-output",
        device="cpu", precision="fp32", offload="none", vae_tiling=False,
        local_files_only=False, model_revision="b" * 40, prompt="Custom parent",
        edit_instruction="Custom edit", parent_seed=7, edit_seeds=(8, 9), width=512,
        height=768, steps=12, guidance=3.0, max_sequence_length=256,
        parent_image=tmp_path / "input.png", max_retries=2, note="Local experiment",
    )


def test_dotenv_is_not_loaded_from_parent_directories(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("KUSAMA_DEVICE=cpu\n")
    child = tmp_path / "child"
    child.mkdir()
    monkeypatch.chdir(child)
    config, _ = parse_args([])
    assert config.device == "auto"


@pytest.mark.parametrize("revision", ["main", "v1.0", "a" * 39, "a" * 41, "A" * 40, "z" * 40])
def test_moving_or_invalid_model_revisions_are_rejected(revision, capsys):
    with pytest.raises(SystemExit) as exc:
        parse_args(["--model-revision", revision])
    assert exc.value.code == 2
    assert "immutable" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [
    ["--edit-seeds", "23", "23"], ["--parent-seed", "-1"],
    ["--edit-seeds", "23", str(2**63)], ["--width", "63"],
    ["--height", "1000"], ["--steps", "0"], ["--guidance", "nan"],
    ["--guidance", "inf"], ["--max-sequence-length", "513"],
    ["--max-retries", "3"], ["--prompt", " "],
    ["--edit-instruction", " "], ["--device", "cpu", "--offload", "model"],
])
def test_invalid_execution_settings_fail_before_model_loading(arguments):
    with pytest.raises(SystemExit) as exc:
        parse_args(arguments)
    assert exc.value.code == 2


@pytest.mark.parametrize("name,value", [
    ("KUSAMA_DEVICE", "tpu"), ("KUSAMA_PRECISION", "int8"),
    ("KUSAMA_OFFLOAD", "automatic"), ("KUSAMA_VAE_TILING", "maybe"),
    ("KUSAMA_LOCAL_FILES_ONLY", "sometimes"), ("KUSAMA_MODEL_REVISION", "main"),
])
def test_invalid_environment_values_get_actionable_parser_errors(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit) as exc:
        parse_args([])
    assert exc.value.code == 2
