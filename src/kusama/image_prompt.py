"""Explicit model-selected image conditioning; source evidence is never a parent."""
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import sha256

MAX_REFERENCE_IMAGES = 2
PREPROCESSING = ('Saved bytes decoded as RGB; pinned FLUX processor bounds each reference to '
                 '1024*1024 pixels and crops to multiples of 16. Output profile is unchanged.')


class ImagePromptSelection(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    source_id: str = Field(min_length=1, max_length=180)
    image_index: int = Field(default=0, ge=0, strict=True)
    purpose: str = Field(min_length=1, max_length=500,
                         description='What visible property to transfer and why; put the visual instruction in the FLUX prompt too.')


class ImagePromptAsset(ImagePromptSelection):
    path: Path
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')


def validate_assets(assets, *, base=None, max_images=MAX_REFERENCE_IMAGES):
    if not isinstance(assets, list) or len(assets) > max_images:
        raise ValueError(f'Image prompt accepts at most {max_images} reference images')
    result, seen = [], set()
    for value in assets:
        asset = ImagePromptAsset.model_validate(value).model_dump(mode='json')
        key = (asset['source_id'], asset['image_index'])
        if key in seen:
            raise ValueError('Duplicate image prompt selection')
        seen.add(key)
        path = (Path(base) / asset['path'] if base else Path(asset['path'])).resolve()
        if not path.is_file() or sha256(path) != asset['sha256']:
            raise ValueError('Image prompt reference is missing or its hash changed')
        with Image.open(path) as image:
            image.load()
            if min(image.size) < 16:
                raise ValueError('Image prompt reference dimensions must be at least 16 pixels')
        result.append({**asset, 'path': str(path)})
    return result


def resolve_selections(selections, sources, *, max_images=MAX_REFERENCE_IMAGES, base=None):
    """Resolve only supplied, inspected views, never model-provided filesystem paths."""
    by_id = {s['source_id']: s for s in sources}
    assets = []
    for value in selections:
        selected = ImagePromptSelection.model_validate(value).model_dump(mode='json')
        source = by_id.get(selected['source_id'])
        if source is None:
            raise ValueError('Image prompt source must belong to the selected, opened references')
        images = source.get('images', [])
        if selected['image_index'] >= len(images):
            raise ValueError('Image prompt image_index must identify an opened image, not text or metadata')
        rights = source.get('reference', {}).get('rights', {}).get('derivatives', source.get('render_rights'))
        if rights and rights.get('status') != 'allowed':
            raise ValueError('Source policy does not permit image-conditioned derivatives')
        image = images[selected['image_index']]
        assets.append({**selected, 'path': image['path'], 'sha256': image['sha256']})
    return validate_assets(assets, max_images=max_images, base=base)


def image_label(image):
    label = image['source_id'] + (':' + image['scope'] if image.get('scope') in ('whole', 'part') else '')
    return label + (f" [image_index={image['image_index']}]" if 'image_index' in image else '')


def conditioning_record(assets, *, parent=None):
    """One-based FLUX image order, distinct from zero-based source view indices."""
    images = [] if parent is None else [{'input_index': 1, 'role': 'parent', **parent}]
    images += [{'input_index': i + 1 + len(images), 'role': 'reference', **asset}
               for i, asset in enumerate(assets)]
    return {'mode': 'image_conditioned' if assets else ('parent_edit' if parent else 'text_only'),
            'images': images, 'preprocessing': PREPROCESSING}
