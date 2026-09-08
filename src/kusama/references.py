"""Small, sourced reference packets consumed at inference with frozen weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .artifacts import sha256, write_json
from .execution import _copy_snapshot_file


class ReferenceItem(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    title: str = Field(min_length=1, max_length=500)
    url: str = Field(pattern=r'^https?://', max_length=2000)
    provenance: str = Field(min_length=1, max_length=1500)
    permitted_uses: str = Field(min_length=1, max_length=1500)
    source_family: str = Field(min_length=1, max_length=300)
    context: str = Field(min_length=1, max_length=1800)
    relationship_to_brief: str = Field(min_length=1, max_length=1000)
    image_path: str | None = Field(default=None, min_length=1, max_length=2000)
    maker: str | None = Field(default=None, max_length=500)
    date: str | None = Field(default=None, max_length=300)
    medium: str | None = Field(default=None, max_length=500)


class ReferencePacket(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    schema_version: Literal[1] = 1
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    title: str = Field(min_length=1, max_length=500)
    brief: str = Field(min_length=1, max_length=5000)
    references: list[ReferenceItem] = Field(min_length=1, max_length=12)
    creative_constraints: list[str] = Field(default_factory=list, max_length=20)
    composition_challenges: dict[str, str] = Field(default_factory=dict, max_length=12)

    @model_validator(mode='after')
    def check_packet(self):
        if len({ref.id for ref in self.references}) != len(self.references):
            raise ValueError('Reference IDs must be unique')
        if not 1 <= sum(bool(ref.image_path) for ref in self.references) <= 6:
            raise ValueError('Supply one to six reference images, plus optional text-only sources')
        if any(not s.strip() or len(s) > 800 for s in self.creative_constraints):
            raise ValueError('Each creative constraint requires 1–800 characters')
        for identifier, challenge in self.composition_challenges.items():
            if not re.fullmatch(r'd(?:0[1-9]|1[0-2])', identifier):
                raise ValueError('Composition challenge IDs must be d01–d12')
            if not challenge.strip() or len(challenge) > 800:
                raise ValueError('Each composition challenge requires 1–800 characters')
        return self


def load_packet(path: Path) -> tuple[dict, dict]:
    """Validate local assets; return normalized content and source byte hashes."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    packet = ReferencePacket.model_validate_json(raw).model_dump()
    # Preserve the legacy normalized shape when no scaffolding is supplied.
    if not packet['composition_challenges']:
        packet.pop('composition_challenges')
    hashes = {'packet': hashlib.sha256(raw).hexdigest(), 'images': {}}
    for ref in packet['references']:
        if ref['image_path']:
            asset = (path.parent / ref['image_path']).resolve()
            digest = sha256(asset)
            with Image.open(asset) as image:
                image.verify()
            if sha256(asset) != digest:
                raise ValueError(f'Reference image changed during validation: {asset}')
            hashes['images'][ref['id']] = digest
    return packet, hashes


def snapshot_packet(source: Path, directory: Path, expected_hashes: dict) -> dict:
    """Preserve exact source bytes and normalize only the packet's asset paths."""
    source, directory = Path(source).resolve(), Path(directory).resolve()
    packet, hashes = load_packet(source)
    if hashes != expected_hashes:
        raise ValueError('Reference source changed before snapshot; create a new experiment')
    directory.mkdir(parents=True, exist_ok=True)
    _copy_snapshot_file(source, directory / 'source_packet.json', hashes['packet'])
    assets = directory / 'references'
    assets.mkdir(exist_ok=True)
    for ref in packet['references']:
        if ref['image_path']:
            asset = (source.parent / ref['image_path']).resolve()
            # Keep source encoding; the vision processor handles PNG/JPEG/etc.
            target = assets / f"{ref['id']}{asset.suffix.lower()}"
            _copy_snapshot_file(asset, target, hashes['images'][ref['id']])
            ref['image_path'] = str(target.relative_to(directory))
    target = directory / 'packet.json'
    if target.exists():
        if json.loads(target.read_text()) != packet:
            raise ValueError('Saved reference packet differs from the frozen source')
    else:
        write_json(target, packet)
    return packet


def prompt_context(packet: dict) -> dict:
    """Context includes citations and declared analogies, never local machine paths."""
    # The complete operator map remains in packet.json. Only the current
    # assignment is separately supplied to Qwen, so other challenges cannot
    # become examples that anchor the present composition.
    return {**{key: value for key, value in packet.items() if key != 'composition_challenges'}, 'references': [
        {**{key: value for key, value in ref.items() if key != 'image_path'},
         'image_supplied': bool(ref.get('image_path'))}
        for ref in packet['references']
    ]}


def selected_challenges(packet: dict, expected_ids: list[str]) -> dict[str, str]:
    challenges = packet.get('composition_challenges') or {}
    if not challenges:
        return {}
    missing = set(expected_ids) - set(challenges)
    if missing:
        raise ValueError(f'Composition challenges missing requested direction IDs: {sorted(missing)}')
    return {identifier: challenges[identifier] for identifier in expected_ids}
