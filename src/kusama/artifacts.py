"""Durable output and provenance helpers, independent of the model."""

import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def save_image(image: Image.Image, path: Path) -> None:
    temporary = path.with_suffix(".tmp.png")
    image.save(temporary, format="PNG")
    with Image.open(temporary) as saved:
        saved.verify()
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path: Path) -> None:
    """Persist rename metadata as well as file contents on supported platforms."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def image_record(path: Path, *, image_id: str, parent: dict | None, seed: int | None, origin: str) -> dict:
    with Image.open(path) as image:
        image.load()
        size, mode = list(image.size), image.mode
    return {
        "id": image_id, "path": path.name, "sha256": sha256(path),
        "size": size, "mode": mode, "seed": seed, "origin": origin,
        "parent_image_id": parent["id"] if parent else None,
        "parent_sha256": parent["sha256"] if parent else None,
    }


def import_parent(source: Path, target: Path) -> dict:
    before_hash = sha256(source)
    with Image.open(source) as image:
        image.load()
        original = {"path": str(source), "sha256": before_hash, "size": list(image.size), "mode": image.mode}
        # Deterministic orientation and RGB conversion; no spatial rescaling here.
        normalized = ImageOps.exif_transpose(image).convert("RGB")
        save_image(normalized, target)
    if sha256(source) != before_hash:
        raise RuntimeError("Source parent changed while being copied; rerun with an immutable source file.")
    original["normalization"] = "EXIF orientation applied, RGB PNG copy; no resize"
    return original


def make_comparison(run_dir: Path, records: list[dict], width: int, height: int) -> dict:
    """Keep full resolution and use one common scale for every panel; pad, never crop."""
    sources = []
    for record in records:
        with Image.open(run_dir / record["path"]) as image:
            sources.append(image.convert("RGB"))
    panel_width = max([width] + [im.width for im in sources])
    panel_height = max([height] + [im.height for im in sources])
    header = 72
    canvas = Image.new("RGB", (panel_width * len(sources), panel_height + header), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=24)
    for i, (record, image) in enumerate(zip(records, sources, strict=True)):
        left = i * panel_width
        label = "Parent (existing image)" if record["seed"] is None else ("Parent" if record["parent_image_id"] is None else "Edit") + f" - seed {record['seed']}"
        draw.text((left + 24, 23), label, fill="#202020", font=font)
        canvas.paste(image, (left + (panel_width - image.width) // 2, header + (panel_height - image.height) // 2))
    target = run_dir / "comparison.png"
    save_image(canvas, target)
    return {"path": target.name, "sha256": sha256(target), "size": list(canvas.size), "image_ids": [r["id"] for r in records], "scale": "1 output pixel per panel pixel; centered padding, no crop", "partial": len(records) != 3}
