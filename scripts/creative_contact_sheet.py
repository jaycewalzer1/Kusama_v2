"""Make a labeled, uncropped presentation of saved creative candidates.

This does not generate art, change source PNGs, rank images, or cast votes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageOps

from kusama.artifacts import save_image, sha256, write_json
from kusama.ledger import Ledger


def build(directory: Path) -> Path:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    records = manifest["images"]
    if not records:
        raise ValueError("No saved painting candidates yet")
    count = len(records)
    target = directory / f"contact-sheet-{count:02d}.png"
    receipt = target.with_suffix(".json")
    if target.exists() or receipt.exists():
        raise ValueError(f"Presentation already exists; refusing to replace {target}")
    columns, thumb_width, thumb_height = 4, 256, 320
    gap, cell_height, header = 16, 385, 130
    width = gap + columns * (thumb_width + gap)
    height = header + math.ceil(count / columns) * cell_height + 42
    canvas = Image.new("RGB", (width, height), "#f2eee6")
    draw = ImageDraw.Draw(canvas)
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    def sized_font(size):
        return ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    title_font, font, small = (sized_font(size) for size in (26, 15, 13))
    for line_index, line in enumerate(textwrap.wrap(manifest["title"], width=70)[:2]):
        draw.text((gap, 16 + 31 * line_index), line, font=title_font, fill="#211d19")
    draw.text((gap, 86), f"{count} painting studies · two realizations per direction", font=font, fill="#514941")
    draw.text((gap, 109), "Same prompt within each pair. No ranking or human votes shown.", font=small, fill="#514941")
    sources = []
    for index, record in enumerate(records):
        path = (directory / record["path"]).resolve()
        if not path.is_relative_to(directory) or sha256(path) != record["sha256"]:
            raise ValueError(f"Candidate path/hash mismatch: {record['id']}")
        x = gap + (index % columns) * (thumb_width + gap)
        y = header + (index // columns) * cell_height
        with Image.open(path) as original:
            thumbnail = ImageOps.contain(original.convert("RGB"), (thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas.paste(thumbnail, (x + (thumb_width - thumbnail.width) // 2, y + (thumb_height - thumbnail.height) // 2))
        draw.text((x, y + thumb_height + 7), f"{record['id']} · seed {record['seed']}", font=font, fill="#211d19")
        title = record["label"].split(" / realization ")[0]
        for line_index, line in enumerate(textwrap.wrap(title, width=34)[:2]):
            draw.text((x, y + thumb_height + 29 + 16 * line_index), line, font=small, fill="#514941")
        sources.append({"id": record["id"], "path": record["path"], "sha256": record["sha256"]})
    size = f"{manifest['config']['width']} × {manifest['config']['height']}"
    draw.text((gap, height - 28), f"Frozen Qwen3-VL-8B-Instruct → frozen FLUX.2-klein-base-4B · {size} originals available in review", font=small, fill="#514941")
    save_image(canvas, target)
    write_json(receipt, {"schema_version": 1, "kind": "derived_contact_sheet", "path": target.name,
        "sha256": sha256(target), "size": list(canvas.size), "source_images": sources,
        "presentation": "Equal-size cells; proportionally reduced thumbnails; padding, no crop; no ranking.",
        "partial": count != manifest["direction_count"] * manifest["realizations"]})
    descriptor = json.loads((directory / "job.json").read_text())
    with Ledger(Path(descriptor["database"])) as ledger:
        for path in [target, receipt]:
            ledger.register_artifact(descriptor["job_id"], path.name, kind="derived_presentation")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(build(args.directory))
