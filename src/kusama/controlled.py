"""Bounded, plan-driven editing. Hard constraints are independent of visual review."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageChops

from .artifacts import sha256
from .backend import FluxBackend
from .config import parse_args
from .verification import validate_spec, verify, repair_metrics
from .experiment import handle_termination
from .ledger import LedgerError

METHODS = ("whole", "masked", "geometric")


def binary_mask(path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        if source.mode not in ("1", "L") or source.size != size:
            raise ValueError(f"Mask must be grayscale and match parent dimensions: {path}")
        mask = source.convert("L")
    if set(mask.get_flattened_data()) - {0, 255} or mask.getbbox() is None:
        raise ValueError("Masks must be nonempty and contain only 0 and 255.")
    return mask


def transformed(parent, mask, scale, anchor):
    """Uniform affine scale around a fixed point in full-image coordinates."""
    x, y = anchor
    inverse = (1 / scale, 0, x * (1 - 1 / scale),
               0, 1 / scale, y * (1 - 1 / scale))
    rgba = parent.convert("RGBA")
    rgba.putalpha(mask)
    # Premultiplied interpolation prevents color halos at the cutout edge.
    cutout = rgba.convert("RGBa").transform(parent.size, Image.Transform.AFFINE,
        inverse, Image.Resampling.BICUBIC).convert("RGBA")
    return cutout


def load_plan(path: Path):
    plan = json.loads(path.read_text())
    required = {"parent", "parent_sha256", "instruction", "target", "allowed_mask",
                "method", "seed", "max_calls", "visual_checks"}
    optional = {"object_mask", "scale", "anchor", "repair_prompt", "fallbacks", "crop", "author", "repair_policy", "secondary_mask", "verification"}
    if not isinstance(plan, dict) or required - plan.keys() or plan.keys() - required - optional:
        raise ValueError(f"Plan needs {sorted(required)}; optional fields: {sorted(optional)}")
    for field in ("parent", "instruction", "target", "allowed_mask"):
        if not isinstance(plan[field], str) or not plan[field].strip():
            raise ValueError(f"{field} must be a nonempty string")
    for field, high in (("seed", 2**63 - 1), ("max_calls", 6)):
        if type(plan[field]) is not int or not (0 if field == "seed" else 1) <= plan[field] <= high:
            raise ValueError(f"Invalid {field}")
    if not isinstance(plan["visual_checks"], list) or not plan["visual_checks"] or any(
            not isinstance(v, str) or not v.strip() for v in plan["visual_checks"]):
        raise ValueError("visual_checks must contain explicit review questions")
    parent_path = (path.parent / plan["parent"]).resolve()
    if sha256(parent_path) != plan["parent_sha256"]:
        raise ValueError("Parent SHA-256 does not match the approved plan")
    with Image.open(parent_path) as source:
        if source.mode != "RGB" or source.format != "PNG":
            raise ValueError("Use an RGB PNG parent; normalization must precede the plan")
        parent = source.copy()
    allowed = binary_mask(path.parent / plan["allowed_mask"], parent.size)
    obj = binary_mask(path.parent / plan["object_mask"], parent.size) if plan.get("object_mask") else None
    if plan["method"] not in (*METHODS, "auto"):
        raise ValueError("Unknown editing method")
    fallbacks = plan.get("fallbacks", [])
    if not isinstance(fallbacks, list) or len(fallbacks) > 5 or any(m not in METHODS for m in fallbacks):
        raise ValueError("fallbacks must contain at most five explicit method names")
    has_geometry = any(k in plan for k in ("scale", "anchor"))
    cutout = None
    if has_geometry or "geometric" in [plan["method"], *fallbacks]:
        scale, anchor = plan.get("scale"), plan.get("anchor")
        if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 4:
            raise ValueError("scale must be finite and between 0 (exclusive) and 4")
        if not isinstance(anchor, list) or len(anchor) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) for v in anchor):
            raise ValueError("anchor must be two finite image coordinates")
        if not (0 <= anchor[0] < parent.width and 0 <= anchor[1] < parent.height):
            raise ValueError("Anchor lies outside the image")
        if obj is None or not isinstance(plan.get("repair_prompt"), str) or not plan["repair_prompt"].strip():
            raise ValueError("Geometry requires object_mask and repair_prompt")
        bbox = obj.getbbox()
        mapped = [anchor[i % 2] + scale * (v - anchor[i % 2]) for i, v in enumerate(bbox)]
        if mapped[0] < 0 or mapped[1] < 0 or mapped[2] > parent.width or mapped[3] > parent.height:
            raise ValueError("Transformed object would be clipped")
        cutout = transformed(parent, obj, scale, anchor)
        if cutout.getchannel("A").getbbox() is None:
            raise ValueError("Scale would erase the object")
        for mask in (obj, cutout.getchannel("A")):
            if ImageChops.multiply(mask, ImageChops.invert(allowed)).getbbox():
                raise ValueError("Original and transformed object must fit inside allowed_mask, including interpolation edges")
    if obj is not None and ImageChops.multiply(obj, ImageChops.invert(allowed)).getbbox():
        raise ValueError("Object mask extends outside allowed_mask")
    policy = plan.get("repair_policy")
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) != {"padding", "feather"} or obj is None:
            raise ValueError("repair_policy requires object_mask and exactly padding/feather")
        if any(type(policy[k]) is not int for k in policy) or not 0 <= policy['feather'] <= policy['padding'] <= 16:
            raise ValueError("Repair settings require 0 <= feather <= padding <= 16, all integers")
    if plan.get("secondary_mask"):
        secondary = binary_mask(path.parent / plan['secondary_mask'], parent.size)
        if ImageChops.multiply(secondary, ImageChops.invert(allowed)).getbbox():
            raise ValueError("Secondary effects extend outside allowed_mask")
    if 'verification' in plan:
        validate_spec(plan['verification'])
        if plan['verification']['operation'] == 'resize' and cutout is None:
            raise ValueError("Resize verification requires scale, anchor and object mask")
        if plan['verification']['observer'] == 'color' and obj is None:
            raise ValueError("Color calibration requires object_mask")
    crop = plan.get("crop", [0, 0, *parent.size])
    if not isinstance(crop, list) or len(crop) != 4 or any(type(v) is not int for v in crop):
        raise ValueError("crop must be four integer coordinates")
    l, t, r, b = crop
    if not (0 <= l < r <= parent.width and 0 <= t < b <= parent.height):
        raise ValueError("Invalid crop bounds")
    if any(v < 64 or v % 16 for v in (r-l, b-t)):
        raise ValueError("Inference dimensions must be multiples of 16 and at least 64")
    ab = allowed.getbbox()
    if not (l <= ab[0] and t <= ab[1] and r >= ab[2] and b >= ab[3]):
        raise ValueError("crop must contain the entire authorized mask")
    method = plan["method"]
    if method == "auto":
        method = "geometric" if cutout is not None else (
            "whole" if allowed.getextrema() == (255, 255) else "masked")
    if "whole" in [method, *fallbacks] and any(v < 64 or v % 16 for v in parent.size):
        raise ValueError("Whole-image inference requires parent dimensions divisible by 16 and at least 64")
    return plan, parent, allowed, obj, cutout, crop, [method, *fallbacks]


def preservation(parent, result, allowed):
    if result.size != parent.size or result.mode != "RGB":
        return {"passed": False, "reason": "Output dimensions or color mode changed"}
    diff = ImageChops.difference(parent, result)
    channels = diff.split()
    magnitude = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    changed = magnitude.point(lambda x: 255 if x else 0)
    outside = ImageChops.multiply(changed, ImageChops.invert(allowed))
    count = outside.histogram()[255]
    return {"passed": count == 0, "outside_changed_pixels": count,
            "total_changed_pixels": changed.histogram()[255], "changed_bbox": changed.getbbox(),
            "protected_pixels": allowed.histogram()[0],
            "scope": "decoded RGB pixels; metadata and encoding are not compared"}


def run(plan_path=None, config=None, backend_factory=FluxBackend, call_guard=None, **kwargs):
    from .execution import run_job
    return run_job(plan_path, config, backend_factory, call_guard, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Execute a bounded, mask-constrained edit plan. Remaining flags match kusama-first-edit.")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--resume", type=Path, help="Resume a registered run directory with its frozen configuration")
    parser.add_argument("--job-id", help="Stable identifier for an idempotent new job")
    parser.add_argument("--validate-only", action="store_true")
    args, remaining = parser.parse_known_args(argv)
    if args.resume and (remaining or args.job_id or args.validate_only):
        parser.error("--resume uses saved settings; omit backend flags, --job-id and --validate-only")
    if not args.resume and args.plan is None:
        parser.error("Supply --plan for a new job or --resume for an existing run")
    config, check = parse_args(remaining) if not args.resume else (None, False)
    if check:
        parser.error("Use kusama-first-edit --check-environment for hardware diagnostics")
    try:
        if args.validate_only:
            *_, methods = load_plan(args.plan)
            print(json.dumps({"valid": True, "methods": methods}))
            return 0
        with handle_termination():
            path = run(args.plan, config, resume_dir=args.resume, job_id=args.job_id)
        print(path.resolve())
        status = json.loads((path / "manifest.json").read_text())["status"]
        return 0 if status == "needs_review" else 1
    except (ValueError, OSError, KeyError, TypeError, LedgerError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
