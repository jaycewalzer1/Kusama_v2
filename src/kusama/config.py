"""Small, explicit configuration surface. CLI values override .env and defaults."""

import argparse
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import re

from dotenv import load_dotenv

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
PARENT_PROMPT = (
    "A collage-style mountain landscape with a clearly visible solitary person "
    "standing near the mountain’s base. The person should be large enough that a "
    "subsequent reduction in their size will be easy to inspect."
)
EDIT_INSTRUCTION = (
    "Reduce the person to approximately one-third of their current height, keeping "
    "their feet anchored at the same ground location. Preserve the mountain’s shape, "
    "camera framing, colors, and lighting. The intended effect is to make the "
    "mountain feel much larger and more overwhelming."
)


@dataclass(frozen=True)
class Config:
    cache_dir: Path = Path(".cache/huggingface")
    output_dir: Path = Path("artifacts")
    device: str = "auto"
    precision: str = "auto"
    offload: str = "none"
    vae_tiling: bool = False
    model_revision: str = MODEL_REVISION
    local_files_only: bool = False
    prompt: str = PARENT_PROMPT
    edit_instruction: str = EDIT_INSTRUCTION
    parent_seed: int = 17
    edit_seeds: tuple[int, int] = (23, 41)
    width: int = 1024
    height: int = 1024
    steps: int = 50
    guidance: float = 4.0
    max_sequence_length: int = 512
    parent_image: Path | None = None
    max_retries: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {k: str(v) if isinstance(v, Path) else v for k, v in asdict(self).items()}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    if raw.lower() in {"1", "true", "yes"}:
        return True
    if raw.lower() in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false (received {raw!r}).")


def parse_args(argv: list[str] | None = None) -> tuple[Config, bool]:
    # Only the invocation directory's .env is used; do not search parent directories.
    load_dotenv(Path.cwd() / ".env", override=False)
    parser = argparse.ArgumentParser(description="Generate one parent and two independent FLUX.2 Klein base edits.")
    parser.add_argument("--cache-dir", type=Path, default=os.getenv("KUSAMA_CACHE_DIR", ".cache/huggingface"))
    parser.add_argument("--output-dir", type=Path, default=os.getenv("KUSAMA_OUTPUT_DIR", "artifacts"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=os.getenv("KUSAMA_DEVICE", "auto"))
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default=os.getenv("KUSAMA_PRECISION", "auto"))
    parser.add_argument("--offload", choices=["none", "model", "sequential"], default=os.getenv("KUSAMA_OFFLOAD", "none"))
    try:
        parser.add_argument("--vae-tiling", action=argparse.BooleanOptionalAction, default=env_bool("KUSAMA_VAE_TILING"))
        parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=env_bool("KUSAMA_LOCAL_FILES_ONLY"))
    except ValueError as exc:
        parser.error(str(exc))
    parser.add_argument("--model-revision", default=os.getenv("KUSAMA_MODEL_REVISION", MODEL_REVISION), help="Immutable 40-character Hugging Face commit SHA")
    parser.add_argument("--prompt", default=PARENT_PROMPT)
    parser.add_argument("--edit-instruction", default=EDIT_INSTRUCTION)
    parser.add_argument("--parent-seed", type=int, default=17)
    parser.add_argument("--edit-seeds", type=int, nargs=2, default=[23, 41], metavar=("SEED_A", "SEED_B"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--parent-image", type=Path, help="Reuse this image; never regenerate its parent")
    parser.add_argument("--max-retries", type=int, choices=range(3), default=0, help="Additional attempts per image call; default 0, maximum 2")
    parser.add_argument("--note", default="", help="Reason for configuration changes or other development context")
    parser.add_argument("--check-environment", action="store_true", help="Check imports, hardware and a small device operation; no weights or image calls")
    args = vars(parser.parse_args(argv))
    check = args.pop("check_environment")
    for key, choices in {"device": {"auto", "cuda", "mps", "cpu"}, "precision": {"auto", "bf16", "fp16", "fp32"}, "offload": {"none", "model", "sequential"}}.items():
        if args[key] not in choices:
            parser.error(f"Invalid {key}: {args[key]!r}; choose from {sorted(choices)}.")
    if not re.fullmatch(r"[0-9a-f]{40}", args["model_revision"]):
        parser.error("--model-revision / KUSAMA_MODEL_REVISION must be an immutable lowercase 40-character commit SHA, not main or a tag.")
    if any(args[k] < 64 or args[k] % 16 for k in ("width", "height")):
        parser.error("width and height must be at least 64 and multiples of 16; 1024 is the experiment default.")
    if args["steps"] < 1 or not 1 <= args["max_sequence_length"] <= 512:
        parser.error("steps must be positive and max-sequence-length must be 1..512.")
    if not math.isfinite(args["guidance"]) or args["guidance"] < 1:
        parser.error("guidance must be finite and at least 1.")
    if any(seed < 0 or seed >= 2**63 for seed in [args["parent_seed"], *args["edit_seeds"]]):
        parser.error("seeds must be integers in [0, 2**63).")
    if len(set(args["edit_seeds"])) != 2:
        parser.error("two distinct edit seeds are required so saved alternatives cannot overwrite each other.")
    if not args["prompt"].strip() or not args["edit_instruction"].strip():
        parser.error("prompt and edit-instruction must not be empty.")
    if args["device"] == "cpu" and args["offload"] != "none":
        parser.error("CPU execution requires --offload none.")
    for key in ("cache_dir", "output_dir", "parent_image"):
        if args[key] is not None:
            args[key] = Path(args[key]).expanduser().resolve()
    args["edit_seeds"] = tuple(args["edit_seeds"])
    return Config(**args), check


def adjustments(config: Config) -> list[dict]:
    baseline = Config()
    keys = ("model_revision", "width", "height", "steps", "guidance", "max_sequence_length", "offload", "vae_tiling", "parent_seed", "edit_seeds", "prompt", "edit_instruction")
    changes = [{"setting": key, "baseline": getattr(baseline, key), "requested": getattr(config, key), "reason": config.note or "Explicit CLI/environment override"} for key in keys if getattr(config, key) != getattr(baseline, key)]
    if config.precision not in ("auto", "bf16"):
        changes.append({"setting": "precision", "baseline": "bf16 where supported", "requested": config.precision, "reason": config.note or "Explicit CLI/environment override"})
    return changes
