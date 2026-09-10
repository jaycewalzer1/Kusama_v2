"""Image selection validation uses local fixtures only."""
import pytest
from PIL import Image

from kusama.artifacts import sha256
from kusama.image_prompt import resolve_selections
from kusama.research import ResearchHypothesis


@pytest.fixture
def sources(tmp_path):
    path = tmp_path / 'whole.png'; detail = tmp_path / 'detail.png'
    Image.new('RGB', (32, 48), 'red').save(path)
    Image.new('RGB', (48, 32), 'blue').save(detail)
    return [{'source_id': 'doll', 'images': [
        {'path': str(p), 'sha256': sha256(p)} for p in (path, detail)]},
        {'source_id': 'essay', 'images': []}]


def test_whole_or_detail_is_an_explicit_choice(sources):
    selection = [{'source_id': 'doll', 'image_index': 1, 'purpose': 'Transfer forehead contour'}]
    result = resolve_selections(selection, sources)
    assert result[0]['sha256'] == sources[0]['images'][1]['sha256']
    assert resolve_selections([], sources) == []


@pytest.mark.parametrize('selection', [
    {'source_id': 'unknown', 'image_index': 0},
    {'source_id': 'essay', 'image_index': 0},
    {'source_id': 'doll', 'image_index': 2},
    {'source_id': 'doll', 'image_index': -1},
    {'source_id': 'doll', 'image_index': True},
    {'source_id': 'doll', 'image_index': 0, 'path': '/arbitrary/model/path'},
])
def test_unavailable_or_invented_image_selection_is_rejected(sources, selection):
    with pytest.raises(ValueError):
        resolve_selections([{**selection, 'purpose': 'Borrow contour'}], sources)


def test_source_derivative_restriction_is_respected(sources):
    sources[0]['reference'] = {'rights': {'derivatives': {'status': 'denied'}}}
    with pytest.raises(ValueError, match='policy'):
        resolve_selections([{'source_id': 'doll', 'purpose': 'Borrow contour'}], sources)
