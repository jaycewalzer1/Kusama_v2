"""Real PNG and JSON filesystem checks using small synthetic images."""

import hashlib
import json

from PIL import Image
import pytest

from kusama.artifacts import image_record, import_parent, make_comparison, save_image, write_json


def test_comparison_preserves_each_image_scale_and_edges_with_padding(tmp_path):
    records = []
    sizes = [(64, 64), (80, 64), (64, 80)]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for index, (size, color) in enumerate(zip(sizes, colors, strict=True)):
        path = tmp_path / f"image_{index}.png"
        save_image(Image.new("RGB", size, color), path)
        records.append(image_record(path, image_id=f"image_{index}", parent=records[0] if records else None, seed=17 + index, origin="test_fixture"))
    comparison = make_comparison(tmp_path, records, 64, 64)
    assert comparison["size"] == [240, 152]
    assert comparison["partial"] is False
    assert comparison["sha256"] == hashlib.sha256((tmp_path / comparison["path"]).read_bytes()).hexdigest()
    with Image.open(tmp_path / comparison["path"]) as image:
        image.load()
        for index, (size, color) in enumerate(zip(sizes, colors, strict=True)):
            left = index * 80 + (80 - size[0]) // 2
            top = 72 + (80 - size[1]) // 2
            assert image.crop((left, top, left + size[0], top + size[1])).tobytes() == Image.new("RGB", size, color).tobytes()


def test_parent_import_converts_mode_without_resizing_or_modifying_source(tmp_path):
    source, target = tmp_path / "source.png", tmp_path / "parent.png"
    Image.new("RGBA", (80, 64), (25, 50, 75, 255)).save(source)
    before = source.read_bytes()
    record = import_parent(source, target)
    assert source.read_bytes() == before
    assert record["sha256"] == hashlib.sha256(before).hexdigest()
    assert record["mode"] == "RGBA"
    assert record["size"] == [80, 64]
    with Image.open(target) as image:
        assert image.mode == "RGB"
        assert image.size == (80, 64)
        assert image.getpixel((79, 63)) == (25, 50, 75)


def test_json_failure_does_not_replace_last_persisted_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"status": "running", "images": ["parent.png"]})
    before = manifest.read_bytes()
    with pytest.raises(ValueError):
        write_json(manifest, {"bad_measurement": float("nan")})
    assert manifest.read_bytes() == before
    write_json(manifest, {"status": "completed", "prompt": "mountain’s base"})
    assert json.loads(manifest.read_text())["status"] == "completed"
    assert not manifest.with_suffix(".json.tmp").exists()
