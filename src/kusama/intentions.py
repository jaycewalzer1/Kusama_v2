"""Attributed artistic intentions, distinct from render controls and outcomes."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetExistence(BaseModel):
    """Attributed observation of the saved parent, never inferred from a box."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    status: Literal['present', 'absent', 'uncertain']
    author: str = Field(min_length=1, max_length=200)
    origin: Literal['operator', 'model_observation', 'development_example']
    reason: str = Field(min_length=1, max_length=4000)


class IntentionElement(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    id: str = Field(pattern=r'^[A-Za-z0-9_-]{1,64}$')
    role: str = Field(min_length=1, max_length=2000)
    region: tuple[float, float, float, float]
    subtlety: str = Field(min_length=1, max_length=1000)
    relationships: list[str] = Field(default_factory=list, max_length=32)
    must_survive: str = Field(min_length=1, max_length=2000)
    status: Literal['unresolved', 'resolved', 'leave_alone'] = 'unresolved'
    protect_pixels: bool = False
    existence: TargetExistence | None = None

    @model_validator(mode='after')
    def valid_region(self):
        l, t, r, b = self.region
        if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
            raise ValueError('Intention region must have ordered normalized bounds')
        return self


class IntentionMap(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    schema_version: Literal[1] = 1
    parent_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    author: str = Field(min_length=1, max_length=200)
    origin: Literal['operator', 'model_proposal', 'development_example']
    whole: str = Field(min_length=1, max_length=4000)
    elements: list[IntentionElement] = Field(min_length=1, max_length=32)
    target_ids: list[str] = Field(min_length=1, max_length=32)
    required_support_ids: list[str] = Field(default_factory=list, max_length=32)
    operation: Literal['add', 'remove', 'exaggerate', 'conceal', 'move', 'simplify', 'revise']
    attention_reason: str = Field(min_length=1, max_length=4000)
    expected_reading: str = Field(min_length=1, max_length=4000)
    previous_continuation_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]{1,40}$')
    revisit_reasons: dict[str, str] = Field(default_factory=dict, max_length=32)

    @model_validator(mode='after')
    def known_elements(self):
        ids = [element.id for element in self.elements]
        if len(ids) != len(set(ids)) or len(self.target_ids) != len(set(self.target_ids)):
            raise ValueError('Intention and target IDs must be unique')
        if set(self.target_ids) - set(ids):
            raise ValueError('Every attention target must resolve to an intention element')
        if (len(self.required_support_ids) != len(set(self.required_support_ids))
                or set(self.required_support_ids) - set(ids)):
            raise ValueError('Required edit supports must be unique known intention elements')
        if any(set(element.relationships) - set(ids) for element in self.elements):
            raise ValueError('Every intention relationship must resolve to an element')
        if self.revisit_reasons and not self.previous_continuation_id:
            raise ValueError('Revisiting requires a previous continuation')
        if any(not reason.strip() for reason in self.revisit_reasons.values()):
            raise ValueError('Revisiting a decision requires a nonblank reason')
        return self


def pixel_bounds(region, size):
    l, t, r, b = region
    bounds = (int(l * size[0]), int(t * size[1]), int(r * size[0]), int(b * size[1]))
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise ValueError('Region is empty after mapping to parent pixels')
    return bounds


def overlaps(a, b):
    return max(a[0], b[0]) < min(a[2], b[2]) and max(a[1], b[1]) < min(a[3], b[3])


def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]


def validate_intention_edit(intentions, revision, parent_sha256, size, *, full_coverage=True):
    if intentions.parent_sha256 != parent_sha256:
        raise ValueError('Intention map must describe the exact selected parent hash')
    edit = pixel_bounds(revision.region or (0, 0, 1, 1), size)
    for element in intentions.elements:
        bounds = pixel_bounds(element.region, size)
        if element.protect_pixels and overlaps(bounds, edit):
            raise ValueError('Edit overlaps pixel-protected intention: ' + element.id)
        if element.id in intentions.target_ids:
            if element.status != 'unresolved':
                raise ValueError('Attention targets must be explicitly reopened as unresolved')
            if not overlaps(bounds, edit):
                raise ValueError('Edit does not intersect attention target: ' + element.id)
            if full_coverage and not contains(edit, bounds):
                raise ValueError('Edit must contain the entire declared attention target: ' + element.id)
            if (full_coverage and intentions.operation != 'add' and element.existence
                    and element.existence.status == 'absent'):
                raise ValueError('Existing-object operation has an explicitly absent target: ' + element.id)
        if full_coverage and element.id in intentions.required_support_ids:
            if not contains(edit, bounds):
                raise ValueError('Edit must contain the entire required support: ' + element.id)
            if element.existence and element.existence.status == 'absent':
                raise ValueError('Required support is explicitly absent: ' + element.id)
