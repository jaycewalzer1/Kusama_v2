"""Frozen FLUX.2 Klein inference through the released Diffusers pipeline.

This module deliberately has no application-level lineage or artifact policy.
The caller owns saved images, manifests, timing, and failure records.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from functools import lru_cache
from typing import Any, Callable

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"


class BackendError(RuntimeError):
    """An actionable model, dependency, hardware, or inference failure."""


def count_prompt_tokens(tokenizer, prompt: str) -> int:
    if not isinstance(prompt, str) or not prompt.strip():
        raise BackendError('A non-empty text prompt is required.')
    formatted = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return len(tokenizer(formatted, truncation=False, padding=False)['input_ids'])


@lru_cache(maxsize=2)
def _cached_prompt_tokenizer(cache_dir: str, revision: str):
    from transformers import Qwen2TokenizerFast
    return Qwen2TokenizerFast.from_pretrained(MODEL_ID, subfolder='tokenizer', revision=revision,
        cache_dir=cache_dir, local_files_only=True)


def cached_prompt_token_count(prompt: str, *, cache_dir: str, revision: str) -> int:
    """Pinned, local tokenizer only; no FLUX weights or inference."""
    return count_prompt_tokens(_cached_prompt_tokenizer(cache_dir, revision), prompt)


def _run_json(command: list[str]) -> Any | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def _versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "huggingface-hub",
                 "safetensors", "tokenizers", "Pillow", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch() -> Any:
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise BackendError("PyTorch could not be imported. Run `uv sync --locked` in this project.") from exc
    return torch


def inspect_environment() -> dict[str, Any]:
    """Inspect the current process and available hardware without loading a model."""
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "os": platform.platform(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": _versions(),
        "cuda_available": False,
        "mps_available": False,
        "preferred_device": "cpu",
        "gpus": [],
    }
    try:
        import psutil
        ram = psutil.virtual_memory()
        result.update(system_total_bytes=ram.total, system_available_bytes=ram.available)
    except ImportError:
        result["memory_inspection_error"] = "psutil is unavailable; run uv sync --locked"

    if platform.system() == "Darwin":
        result["macos_version"] = platform.mac_ver()[0]
        displays = _run_json(["system_profiler", "SPDisplaysDataType", "-json"])
        if isinstance(displays, dict):
            result["gpus"] = [
                {"name": gpu.get("sppci_model", gpu.get("_name")),
                 "gpu_cores": gpu.get("sppci_cores"),
                 "metal_support": gpu.get("spdisplays_mtlgpufamilysupport"),
                 "driver": "Metal driver supplied by macOS " + platform.mac_ver()[0]}
                for gpu in displays.get("SPDisplaysDataType", [])
            ]
    if shutil.which("nvidia-smi"):
        try:
            nvidia = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10, check=True,
            )
            result["nvidia_smi"] = nvidia.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError) as exc:
            result["nvidia_smi_error"] = str(exc)
    try:
        torch = _torch()
        result.update(torch_version=torch.__version__, cuda_runtime_version=torch.version.cuda,
                      cuda_available=torch.cuda.is_available(),
                      mps_available=torch.backends.mps.is_available(),
                      mps_built=torch.backends.mps.is_built())
        if result["cuda_available"]:
            result["cuda_devices"] = [
                {"index": index, "name": torch.cuda.get_device_name(index),
                 "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                 "compute_capability": list(torch.cuda.get_device_capability(index))}
                for index in range(torch.cuda.device_count())
            ]
        if result["mps_available"]:
            result["mps_recommended_max_bytes"] = torch.mps.recommended_max_memory()
        probe_device = "cuda:0" if result["cuda_available"] else ("mps" if result["mps_available"] else "cpu")
        result["preferred_device"] = probe_device
        probe_dtype = torch.bfloat16 if probe_device != "cpu" else torch.float32
        try:
            probe = torch.ones((8, 8), device=probe_device, dtype=probe_dtype)
            probe_value = (probe @ probe).float().sum().item()
            result["arithmetic_probe"] = {
                "device": probe_device, "dtype": str(probe_dtype),
                "passed": probe_value == 512, "result": probe_value,
            }
            del probe
        except Exception as exc:
            result["arithmetic_probe"] = {"device": probe_device, "dtype": str(probe_dtype),
                                           "passed": False, "error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        from diffusers import Flux2KleinPipeline
        from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        result["pipeline_import"] = {
            "passed": True, "pipeline": Flux2KleinPipeline.__name__,
            "tokenizer": Qwen2TokenizerFast.__name__, "text_encoder": Qwen3ForCausalLM.__name__,
        }
    except Exception as exc:
        result["pipeline_import"] = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
    return result


def memory_snapshot(device: str) -> dict[str, Any]:
    """Return actual native counters; MPS has no native peak-memory counter."""
    result: dict[str, Any] = {"device": device}
    try:
        import psutil
        result["process_rss_bytes"] = psutil.Process().memory_info().rss
        memory = psutil.virtual_memory()
        result["system_available_bytes"] = memory.available
        result["system_total_bytes"] = memory.total
    except Exception as exc:
        result["host_memory_error"] = f"{type(exc).__name__}: {exc}"
    try:
        torch = _torch()
        kind = device.split(":", 1)[0]
        if kind == "cuda" and torch.cuda.is_available():
            result.update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        elif kind == "mps" and torch.backends.mps.is_available():
            result.update(
                mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                mps_recommended_max_bytes=torch.mps.recommended_max_memory(),
            )
    except Exception as exc:
        result["device_memory_error"] = f"{type(exc).__name__}: {exc}"
    return result


def synchronize(device: str) -> None:
    torch = _torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()


def reset_peak_memory(device: str) -> None:
    torch = _torch()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


class FluxBackend:
    """One frozen base-model pipeline, reused for independent inference calls."""

    def __init__(self, config: Any):
        self.config = config
        self.pipeline: Any | None = None
        self.execution_settings: dict[str, Any] = {}
        self.last_prompt_token_count: int | None = None
        self.last_pipeline_invoked = False
        self.on_inference_start: Callable[[], None] | None = None

    @staticmethod
    def _failure_hint(exc: Exception) -> str:
        message = str(exc).lower()
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status in (401, 403) or "gatedrepo" in type(exc).__name__.lower():
            return "Hugging Face denied access. Check HF_TOKEN (or `hf auth login`) and access to the requested repository/revision; the default 4B base repository is public. "
        if "out of memory" in message or "not enough memory" in message:
            return "Available memory was exhausted. Close memory-consuming applications; explicitly enable --offload model and --vae-tiling, or try --offload sequential with its latency cost. Record the change with --note. "
        return ""

    def _resolve_device_dtype(self, torch: Any) -> tuple[str, Any, str]:
        device = self.config.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        elif device == "cuda":
            device = "cuda:0"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise BackendError("CUDA was requested, but no CUDA device is available. Use --device mps on Apple Silicon or --device cpu.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise BackendError("MPS was requested, but PyTorch cannot use it. Use an arm64 Python and supported macOS, or select --device cpu.")
        precision = self.config.precision
        if precision == "auto":
            if device.startswith("cuda"):
                precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            else:
                precision = "bf16" if device == "mps" else "fp32"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        if device.startswith("cuda") and precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise BackendError("This CUDA device does not support BF16. Set --precision fp16 or fp32 explicitly.")
        try:
            # Probe arithmetic, not just a dtype allocation, before downloading weights.
            probe = torch.ones((8, 8), device=device, dtype=dtype)
            value = (probe @ probe).float().sum().item()
            if value != 512:
                raise RuntimeError(f"unexpected arithmetic result {value}")
            del probe
            synchronize(device)
        except Exception as exc:
            raise BackendError(f"{device} could not execute {precision} arithmetic. Choose a supported --precision or --device explicitly. Original error: {exc}") from exc
        return device, dtype, precision

    def prepare(self, on_progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        """Resolve a commit, publish metadata, then load and freeze its weights."""
        if self.pipeline is not None:
            return dict(self.execution_settings)
        torch = _torch()
        try:
            from diffusers import Flux2KleinPipeline
            from huggingface_hub import HfApi, hf_hub_download
        except (ImportError, OSError, RuntimeError) as exc:
            raise BackendError("FLUX.2 Klein dependencies could not be imported. Run `uv sync --locked`; this backend requires the pinned Diffusers and Transformers releases.") from exc

        revision = self.config.model_revision
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise BackendError("model_revision must be a full 40-character Hugging Face commit SHA to make this run reproducible.")
        if not 1 <= self.config.max_sequence_length <= 512:
            raise BackendError("max_sequence_length must be between 1 and 512 for this Klein text encoder.")
        device, dtype, precision = self._resolve_device_dtype(torch)
        if device == "cpu" and self.config.offload != "none":
            raise BackendError("CPU offloading requires an accelerator. For CPU execution, set --offload none.")
        self.execution_settings = {
            "model_id": MODEL_ID, "requested_revision": revision,
            "device": device, "precision": precision, "torch_dtype": str(dtype),
            "offload": self.config.offload, "vae_tiling": self.config.vae_tiling,
            "width": self.config.width, "height": self.config.height,
            "num_inference_steps": self.config.steps, "guidance_scale": self.config.guidance,
            "max_sequence_length": self.config.max_sequence_length,
            "num_images_per_prompt": 1, "generator_device": "cpu",
            "pipeline_class": "Flux2KleinPipeline", "local_files_only": self.config.local_files_only,
            "cache_dir": str(Path(self.config.cache_dir).expanduser().resolve()),
            "packages": _versions(), "status": "resolving_revision",
            "reproducibility_note": "Seeds repeat the initial CPU noise; exact pixels are not guaranteed across hardware or library versions.",
        }
        if on_progress:
            on_progress(dict(self.execution_settings))
        try:
            if self.config.local_files_only:
                cached_index = hf_hub_download(
                    MODEL_ID, "model_index.json", revision=revision,
                    cache_dir=self.config.cache_dir, local_files_only=True,
                )
                self.execution_settings["revision_verification"] = "pinned local cache"
                self.execution_settings["cached_model_index"] = str(cached_index)
            else:
                info = HfApi().model_info(MODEL_ID, revision=revision)
                if info.sha != revision:
                    raise BackendError(f"Hugging Face resolved {revision} to unexpected commit {info.sha}.")
                self.execution_settings["revision_verification"] = "Hugging Face model_info"
                self.execution_settings["model_gated"] = info.gated
            self.execution_settings.update(model_revision=revision, resolved_revision=revision, status="loading_weights")
            if on_progress:
                on_progress(dict(self.execution_settings))
            pipeline_class = Flux2KleinPipeline
            if getattr(self, "use_inpainting", False):
                from diffusers import Flux2KleinInpaintPipeline
                pipeline_class = Flux2KleinInpaintPipeline
            self.execution_settings["pipeline_class"] = pipeline_class.__name__
            pipeline = pipeline_class.from_pretrained(
                MODEL_ID, revision=revision, cache_dir=self.config.cache_dir,
                dtype=dtype, use_safetensors=True,
                local_files_only=self.config.local_files_only,
                low_cpu_mem_usage=True,
            )
            self.pipeline = pipeline
            if pipeline.config.is_distilled:
                raise BackendError("The loaded checkpoint is marked distilled; this milestone requires the undistilled base model.")
            frozen: dict[str, Any] = {}
            for name, component in pipeline.components.items():
                if isinstance(component, torch.nn.Module):
                    component.eval()
                    component.requires_grad_(False)
                    frozen[name] = {
                        "class": type(component).__name__,
                        "parameter_count": sum(parameter.numel() for parameter in component.parameters()),
                        "trainable_parameter_count": sum(parameter.numel() for parameter in component.parameters() if parameter.requires_grad),
                        "training": component.training,
                        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in component.parameters()}),
                    }
            if self.config.vae_tiling:
                pipeline.vae.enable_tiling()
            if self.config.offload == "model":
                pipeline.enable_model_cpu_offload(device=device)
            elif self.config.offload == "sequential":
                pipeline.enable_sequential_cpu_offload(device=device)
            elif self.config.offload == "none":
                pipeline.to(device)
            else:
                raise BackendError(f"Unknown offload mode: {self.config.offload}")
            self.execution_settings.update(
                status="ready", is_distilled=False, frozen_modules=frozen,
                trainable_parameter_count=sum(module["trainable_parameter_count"] for module in frozen.values()),
                tokenizer_class=type(pipeline.tokenizer).__name__,
                text_encoder_class=type(pipeline.text_encoder).__name__,
                text_encoder_out_layers=[9, 18, 27],
                tokenizer_policy={"chat_template": "user message, add_generation_prompt=True, enable_thinking=False",
                                  "truncation": "reject prompts exceeding max_sequence_length before inference",
                                  "padding": "max_length"},
                scheduler_class=type(pipeline.scheduler).__name__,
                scheduler_config=dict(pipeline.scheduler.config),
                attention_backend="Diffusers/PyTorch default scaled dot product attention",
            )
            if on_progress:
                on_progress(dict(self.execution_settings))
            return dict(self.execution_settings)
        except Exception as exc:
            self.execution_settings["status"] = "load_failed"
            if on_progress:
                on_progress(dict(self.execution_settings))
            if isinstance(exc, BackendError):
                raise
            raise BackendError(
                "FLUX.2 Klein loading failed. " + self._failure_hint(exc) +
                "Check available RAM/disk, network access to Hugging Face, "
                "and that the selected commit is cached when --local-files-only is set. "
                f"No model or precision fallback was attempted. Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def prompt_token_count(self, prompt: str) -> int:
        if self.pipeline is None:
            raise BackendError("Call prepare() before tokenizing prompts.")
        # These are the same template and special-token settings used by the
        # pipeline, but without its automatic truncation or padding.
        return count_prompt_tokens(self.pipeline.tokenizer, prompt)

    def generate(self, prompt: str, seed: int, parent_path: Path | None = None,
                 mask_path: Path | None = None, *, reference_images: list[dict] | None = None) -> Any:
        self.last_pipeline_invoked = False
        if self.pipeline is None or self.execution_settings.get("status") != "ready":
            raise BackendError("The FLUX backend is not ready; call prepare() first.")
        token_count = self.prompt_token_count(prompt)
        self.last_prompt_token_count = token_count
        if token_count > self.config.max_sequence_length:
            raise BackendError(f"Prompt requires {token_count} tokens including its chat template, exceeding the configured {self.config.max_sequence_length}. Shorten it; no truncation was performed.")
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise BackendError("Seed must be an integer between 0 and 2**63 - 1.")
        torch = _torch()
        arguments: dict[str, Any] = {
            "prompt": prompt, "height": self.config.height, "width": self.config.width,
            "num_inference_steps": self.config.steps, "guidance_scale": self.config.guidance,
            "max_sequence_length": self.config.max_sequence_length,
            "generator": torch.Generator(device="cpu").manual_seed(seed),
            "num_images_per_prompt": 1, "output_type": "pil",
        }
        if parent_path is not None:
            from PIL import Image
            try:
                with Image.open(parent_path) as saved_parent:
                    saved_parent.load()
                    parent = saved_parent.convert("RGB")
            except (OSError, ValueError) as exc:
                raise BackendError(f"Could not reload the saved parent image at {parent_path}: {exc}") from exc
            if parent.size != (self.config.width, self.config.height):
                raise BackendError(f"Saved parent has size {parent.size}; expected {(self.config.width, self.config.height)}. No resizing was performed.")
            arguments["image"] = parent
        if mask_path is not None:
            from PIL import Image
            if not getattr(self, "use_inpainting", False) or parent_path is None:
                raise BackendError("A mask requires the inpainting backend and a parent image.")
            with Image.open(mask_path) as saved_mask:
                mask = saved_mask.convert("L")
            if mask.size != (self.config.width, self.config.height):
                raise BackendError("Inpainting mask dimensions must match the parent.")
            arguments.update(mask_image=mask, strength=1.0)
        elif getattr(self, "use_inpainting", False):
            raise BackendError("The inpainting backend requires a mask.")
        if reference_images:
            from PIL import Image
            from .image_prompt import validate_assets
            try:
                assets = validate_assets(reference_images,
                    max_images=1 if getattr(self, 'use_inpainting', False) else 2)
                references = []
                for asset in assets:
                    with Image.open(asset['path']) as reference:
                        references.append(reference.convert('RGB'))
            except (ValueError, OSError) as exc:
                raise BackendError(f'Invalid image prompt reference: {exc}') from exc
            if getattr(self, 'use_inpainting', False):
                # The inpaint pipeline has a separate single-reference input;
                # a list there means a batch, not multiple reference slots.
                arguments['image_reference'] = references[0]
            else:
                arguments['image'] = ([arguments['image']] if 'image' in arguments else []) + references
        try:
            with torch.inference_mode():
                if self.on_inference_start is not None:
                    self.on_inference_start()
                self.last_pipeline_invoked = True
                result = self.pipeline(**arguments)
            if len(result.images) != 1:
                raise BackendError(f"Expected one output image, received {len(result.images)}.")
            image = result.images[0]
            return image
        except Exception as exc:
            if isinstance(exc, BackendError):
                raise
            raise BackendError(
                f"FLUX inference failed on {self.execution_settings['device']} with {self.execution_settings['precision']} "
                f"and {self.config.offload} offloading. " + self._failure_hint(exc) +
                "No retry or configuration change was performed. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

    def recover_after_failure(self) -> None:
        """Release temporary memory and reset model-offload hooks without a retry."""
        if self.pipeline is not None and self.config.offload == "model":
            self.pipeline.maybe_free_model_hooks()
        gc.collect()
        torch = _torch()
        device = self.execution_settings.get("device", "cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.startswith("mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def close(self) -> None:
        self.pipeline = None
        self.execution_settings["status"] = "closed"
        gc.collect()
        torch = _torch()
        device = self.execution_settings.get("device", "cpu")
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device.startswith("mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
