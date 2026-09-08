"""Prepared-mask presentation uses saved evidence; no model execution."""
import html
from dataclasses import replace
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from kusama.artifacts import sha256
from kusama.dashboard import create_app
from kusama.reviews import ReviewConflict
from test_continuation import setup, Backend


def svg_element(body):
    return ET.fromstring(re.search(r'<svg\b.*?</svg>', body, re.S).group(0))


def file_hashes(root):
    return {str(path.relative_to(root)): sha256(path) for path in root.rglob('*') if path.is_file()}


def test_prepared_preview_is_exact_readonly_and_accessible(setup):
    store, request, config = setup
    request['revision']['instruction'] = 'Keep <supports> & "enclosure"\nInterrupt only the centre.'
    prepared = store.prepare('study', request, config)
    before = file_hashes(store.root)
    body = store.gallery_html('study')
    svg = svg_element(body)
    assert svg.attrib['viewBox'] == '0 0 64 64'
    assert svg.attrib['role'] == 'img'
    assert svg.attrib['aria-labelledby'] == 'edit-region-title edit-region-description'
    image = svg.find('{*}image')
    assert image.attrib['preserveAspectRatio'] == 'xMidYMid meet'
    assert (store.root / 'continuations/study' / image.attrib['href']).resolve() == Path(prepared['parent']['path'])
    rectangles = svg.findall('{*}rect')
    assert len(rectangles) == 2
    for rectangle in rectangles:
        assert {key: rectangle.attrib[key] for key in ('x', 'y', 'width', 'height')} == {
            'x': '16', 'y': '16', 'width': '32', 'height': '32'}
        assert rectangle.attrib['fill'] == 'none'
    assert '3,072 of 4,096 pixels protected' in body
    assert 'x [16, 48), y [16, 48)' in body
    exact_instruction = html.unescape(re.search(r'<pre class="continuation-instruction">(.*?)</pre>', body, re.S).group(1))
    assert exact_instruction == request['revision']['instruction']
    assert '<supports>' not in body
    assert file_hashes(store.root) == before
    assert store.get('study')['status'] == 'queued'
    assert store.get('study')['model_calls'] == 0 and Backend.calls == []


@pytest.mark.parametrize('method', ['whole', 'masked'])
def test_full_image_mask_warns_zero_protected_pixels(setup, method):
    store, request, config = setup
    request['revision'].update(method=method, region=None if method == 'whole' else [0, 0, 1, 1])
    store.prepare('study', request, config)
    body = store.gallery_html('study')
    assert '0 protected pixels — the whole image may change.' in body
    assert 'class="continuation-protection continuation-protection-warning"' in body
    assert svg_element(body).find('{*}rect').attrib['width'] == '64'


def test_non_square_parent_uses_prepared_pixel_rounding_and_responsive_viewbox(setup):
    store, request, config = setup
    Image.new('RGB', (512, 640), 'white').save(request['selection']['path'])
    request['selection']['image_sha256'] = sha256(Path(request['selection']['path']))
    config = replace(config, width=512, height=640)
    request['revision']['region'] = [.35, .30, .65, .70]
    store.prepare('study', request, config)
    body = store.gallery_html('study')
    svg = svg_element(body)
    assert svg.attrib['viewBox'] == '0 0 512 640'
    rectangle = svg.find('{*}rect')
    assert {key: rectangle.attrib[key] for key in ('x', 'y', 'width', 'height')} == {
        'x': '179', 'y': '192', 'width': '153', 'height': '256'}
    assert '288,512 of 327,680 pixels protected' in body
    assert 'height' not in svg.attrib and 'width' not in svg.attrib
    assert 'width:100%;height:auto' in body


def test_local_export_and_web_preview_preserve_parent_bytes_before_execution(setup):
    store, request, config = setup
    prepared = store.prepare('study', request, config)
    local = store.export_gallery('study')
    assert svg_element(local.read_text()).find('{*}image').attrib['href'].endswith('/parent.png')
    client = TestClient(create_app(store.root), base_url='http://localhost')
    response = client.get('/api/continuations/study/gallery')
    assert response.status_code == 200
    assert '<link rel="stylesheet" href="/static/dashboard.css">' in response.text
    image_url = svg_element(response.text).find('{*}image').attrib['href']
    assert image_url == '/api/projects/cont_study/artifacts/root'
    assert client.get(image_url).content == Path(prepared['parent']['path']).read_bytes()
    assert '3,072 of 4,096 pixels protected' in response.text
    assert store.get('study')['model_calls'] == 0 and Backend.calls == []


def test_changed_prepared_mask_is_rejected_instead_of_showing_false_region(setup):
    store, request, config = setup
    store.prepare('study', request, config)
    Image.new('L', (64, 64), 255).save(store.root / 'continuations/study/allowed_mask.png')
    with pytest.raises(ReviewConflict, match='prepared mask hash changed'):
        store.gallery_html('study')
