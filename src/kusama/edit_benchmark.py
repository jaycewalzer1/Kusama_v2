"""Development-only edit evidence. Offline, hash-bound and deliberately conservative.

No function in this module loads a model, renders an image or writes a review.
Semantic claims that have no independent measurement remain uncertain.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .verification import assess_boxes, valid_box, DETECTOR_ID, DETECTOR_REVISION

SCHEMA_VERSION = 1
Status = Literal['pass', 'fail', 'uncertain']
ID = r'^[A-Za-z0-9_-]{1,120}$'
SHA = r'^[a-f0-9]{64}$'


class Record(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Asset(Record):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA)


class Region(Record):
    id: str = Field(pattern=ID)
    box: tuple[int, int, int, int]
    role: Literal['required_edit', 'protected']
    origin: Literal['assistant_annotation', 'operator_annotation', 'model_proposal']
    reason: str = Field(min_length=5)

    @model_validator(mode='after')
    def geometry(self):
        if not valid_box(self.box) or min(self.box) < 0:
            raise ValueError('Region must be a nonnegative nonempty half-open pixel rectangle')
        return self


class CachedBoxes(Record):
    observation: Asset
    parent_binding: Asset
    scale: float = Field(default=1.0, gt=0, le=10)
    anchor: tuple[float, float] = (0.0, 0.0)
    height_tolerance: float = Field(default=.2, ge=0, le=1)
    width_tolerance: float = Field(default=.2, ge=0, le=1)
    position_tolerance: float = Field(default=6, ge=0, le=100)


class ModelClaim(Record):
    source: Asset
    json_pointer: str = Field(pattern=r'^/')
    assertion: Literal['success', 'failure', 'uncertain']
    parent_sha256_pointer: str = Field(pattern=r'^/')
    child_sha256_pointer: str = Field(pattern=r'^/')
    author: str = Field(min_length=1)
    interpretation_origin: Literal['assistant_annotation', 'structured_model_field']
    # Interpretation is never itself an independent outcome label.


class EditCase(Record):
    case_id: str = Field(pattern=ID)
    brief_id: str = Field(pattern=ID)
    operation: Literal['connect', 'remove', 'resize', 'preserve_detail', 'change_relationship']
    instruction: str = Field(min_length=10)
    parent: Asset
    child: Asset | None = None
    allowed_mask: Asset | None = None
    regions: list[Region] = Field(default_factory=list)
    cached_boxes: CachedBoxes | None = None
    model_claim: ModelClaim | None = None
    historical_notes: list[Asset] = Field(default_factory=list)
    evidence_kind: Literal['historical_diagnostic', 'prospective', 'fixture']
    method: Literal['whole', 'masked', 'geometric', 'unexecuted']
    render_profile: str = Field(min_length=1)
    authorship: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistency(self):
        if self.cached_boxes and (not self.child or self.operation not in ('resize', 'remove')):
            raise ValueError('Cached box verification needs a child and resize/remove task')
        if self.model_claim and not self.child:
            raise ValueError('Model outcome claim needs a child')
        if len({r.id for r in self.regions}) != len(self.regions):
            raise ValueError('Region IDs must be unique')
        if self.operation == 'preserve_detail' and not any(r.role == 'protected' for r in self.regions):
            raise ValueError('Detail preservation requires an explicit protected region')
        return self


class EditSuite(Record):
    schema_version: Literal[1] = 1
    suite_id: str = Field(pattern=ID)
    split: Literal['development'] = 'development'
    description: str
    cases: list[EditCase] = Field(min_length=1, max_length=100)
    attribution: str

    @model_validator(mode='after')
    def consistency(self):
        if len({c.case_id for c in self.cases}) != len(self.cases):
            raise ValueError('Case IDs must be unique')
        kinds = {c.evidence_kind for c in self.cases}
        if 'fixture' in kinds and len(kinds) != 1:
            raise ValueError('Fixture outcomes must remain in a separate suite')
        return self


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked(asset: Asset, root: Path) -> Path:
    path = (root / asset.path).resolve()
    if digest(path) != asset.sha256:
        raise ValueError(f'Artifact hash mismatch: {asset.path}')
    return path


def load_suite(path: Path | str) -> EditSuite:
    return EditSuite.model_validate_json(Path(path).read_text())


def _image(asset: Asset, root: Path, mode='RGB') -> Image.Image:
    with Image.open(checked(asset, root)) as image:
        image.load()
        return image.convert(mode)


def _outcome(status: Status, reason: str, **measurements):
    return {'status': status, 'reason': reason, **measurements}


def preflight(case: EditCase | dict, root: Path = Path('.')) -> dict:
    """Check input integrity and declared editing scope; never assert target semantics."""
    case = case if isinstance(case, EditCase) else EditCase.model_validate(case)
    root = Path(root)
    errors, scope, warnings = [], [], []
    try:
        parent = _image(case.parent, root)
        width, height = parent.size
        allowed = None
        if case.allowed_mask:
            mask = _image(case.allowed_mask, root, 'L')
            if mask.size != parent.size:
                raise ValueError('Allowed mask dimensions do not match parent')
            allowed = np.asarray(mask) > 0
        for region in case.regions:
            x0, y0, x1, y1 = region.box
            if x1 > width or y1 > height:
                errors.append(f'Region {region.id} extends outside parent')
                continue
            if region.role == 'required_edit':
                fraction = float(allowed[y0:y1, x0:x1].mean()) if allowed is not None else 1.0
                scope.append({'region_id': region.id, 'editable_fraction': fraction,
                              'origin': region.origin, 'box': list(region.box)})
                if fraction < 1:
                    errors.append(f'Required edit region {region.id} is not fully editable')
        required = [r for r in case.regions if r.role == 'required_edit']
        if case.operation in ('connect', 'change_relationship') and len(required) < 2:
            warnings.append('Relationship task needs both endpoint regions before execution')
        if case.operation in ('resize', 'remove') and not required:
            warnings.append('Target region has not been localized before execution')
        for protected in [r for r in case.regions if r.role == 'protected']:
            for editable in required:
                a, b = protected.box, editable.box
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    errors.append(f'Required edit {editable.id} overlaps protected region {protected.id}')
    except (OSError, ValueError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
    return {'status': 'fail' if errors else 'uncertain' if warnings else 'pass',
            'errors': errors, 'warnings': warnings, 'required_region_coverage': scope,
            'legacy_any_overlap_status': ('pass' if any(r['editable_fraction'] > 0 for r in scope) else 'fail' if scope else 'uncertain'),
            'scope': 'Input integrity and annotation coverage only; target existence is separate'}


def _cached_measurement(case: EditCase, root: Path):
    spec = case.cached_boxes
    if not spec:
        return (_outcome('uncertain', 'No independent parent target observation'),
                _outcome('uncertain', 'No independent semantic output measurement'))
    saved = json.loads(checked(spec.observation, root).read_text())
    binding = json.loads(checked(spec.parent_binding, root).read_text())
    if binding.get('parent_sha256') != case.parent.sha256 or saved.get('image_sha256') != case.child.sha256:
        raise ValueError('Cached measurement is not bound to this parent and child')
    observer = saved.get('observer', {})
    if observer.get('model_id') != DETECTOR_ID or observer.get('revision') != DETECTOR_REVISION:
        raise ValueError('Cached detector provenance does not match pinned independent observer')
    measured = assess_boxes(saved.get('before_detections', []), saved.get('after_detections', []),
                            {'operation': case.operation, 'height_tolerance': spec.height_tolerance,
                             'width_tolerance': spec.width_tolerance,
                             'position_tolerance': spec.position_tolerance}, spec.scale, spec.anchor)
    before = measured.get('before_detections', [])
    unique = len(before) == 1 and before[0].get('score', 0) >= .3 and not measured.get('invalid_detections')
    existence = _outcome('pass' if unique else 'uncertain',
                         'One strong cached target detection' if unique else 'Target observation unavailable or ambiguous',
                         observer=observer, detections=before)
    measured.update(observer=observer, measurement_version='cached_boxes_revalidation_width_v1',
                    original_saved_status=saved.get('status'),
                    limitation='Frozen detector boxes are estimates, not human labels; nondetection never proves removal')
    return existence, measured


def _pointer(value, pointer):
    for segment in pointer.lstrip('/').split('/'):
        key = segment.replace('~1', '/').replace('~0', '~')
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def measure_case(case: EditCase | dict, root: Path = Path('.')) -> dict:
    """Measure saved pixels and replay pinned detector evidence; retain every error."""
    case = case if isinstance(case, EditCase) else EditCase.model_validate(case)
    root = Path(root)
    result = {'schema_version': SCHEMA_VERSION, 'case_id': case.case_id, 'brief_id': case.brief_id,
              'operation': case.operation, 'evidence_kind': case.evidence_kind,
              'input': case.model_dump(mode='json'), 'preflight': preflight(case, root),
              'execution': 'saved_child' if case.child else 'unexecuted',
              'target_existence': _outcome('uncertain', 'No independent target measurement'),
              'fulfillment': _outcome('uncertain', 'No child or independent semantic measurement'),
              'preservation': _outcome('uncertain', 'No preservation measurement'),
              'model_claim': None, 'false_success': 'not_assessed', 'errors': [],
              'artistic_quality': 'not_measured', 'inference_calls': 0}
    try:
        parent = _image(case.parent, root)
        for note in case.historical_notes:
            checked(note, root)
        if not case.child:
            result['status'] = 'uncertain'
            return result
        child = _image(case.child, root)
        if child.size != parent.size:
            raise ValueError('Child dimensions do not match parent')
        a, b = np.asarray(parent), np.asarray(child)
        changed = np.any(a != b, axis=2)
        protected = np.zeros(changed.shape, dtype=bool)
        protection_declared = bool(case.allowed_mask or any(r.role == 'protected' for r in case.regions))
        if case.allowed_mask:
            mask = _image(case.allowed_mask, root, 'L')
            if mask.size != parent.size:
                raise ValueError('Allowed mask dimensions do not match parent')
            protected |= np.asarray(mask) == 0
        detail_rows = []
        for region in case.regions:
            if region.role == 'protected':
                x0, y0, x1, y1 = region.box
                if x1 > parent.width or y1 > parent.height:
                    raise ValueError(f'Protected region {region.id} extends outside parent')
                protected[y0:y1, x0:x1] = True
                detail_rows.append({'region_id': region.id, 'changed_pixels': int(changed[y0:y1, x0:x1].sum()),
                                    'pixels': (x1-x0)*(y1-y0), 'origin': region.origin})
        count = int((protected & changed).sum())
        result['preservation'] = _outcome('fail' if count else 'pass' if protection_declared and protected.any() else 'uncertain',
            'Exact decoded RGB comparison; semantic preservation is not implied',
            protected_pixels=int(protected.sum()), changed_protected_pixels=count,
            total_changed_pixels=int(changed.sum()), details=detail_rows)
        result['target_existence'], result['fulfillment'] = _cached_measurement(case, root)
        if case.operation == 'preserve_detail':
            # The target of this narrow task is the stated region's actual pixels.
            result['target_existence'] = _outcome('pass', 'Nonempty protected pixel region is present; semantic identity unverified')
            result['fulfillment'] = _outcome('fail' if any(r['changed_pixels'] for r in detail_rows) else 'pass',
                                             'Exact preservation of declared detail pixels', details=detail_rows)
        if case.model_claim:
            claim = case.model_claim
            source = json.loads(checked(claim.source, root).read_text())
            if (_pointer(source, claim.parent_sha256_pointer) != case.parent.sha256
                    or _pointer(source, claim.child_sha256_pointer) != case.child.sha256):
                raise ValueError('Model claim is not bound to this parent and child')
            raw = _pointer(source, claim.json_pointer)
            result['model_claim'] = {**claim.model_dump(mode='json'), 'raw_value': raw}
            if claim.assertion == 'success':
                status = result['fulfillment']['status']
                result['false_success'] = 'confirmed' if status == 'fail' else 'unsupported' if status == 'uncertain' else 'not_detected'
            else:
                result['false_success'] = 'no_success_claim'
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        result['errors'].append(f'{type(exc).__name__}: {exc}')
        # Integrity failures must not leave apparently usable partial success labels.
        result['fulfillment'] = _outcome('uncertain', 'Evidence integrity or measurement error')
        result['false_success'] = 'not_assessed'
    states = [result[k]['status'] for k in ('target_existence', 'fulfillment', 'preservation')]
    result['status'] = ('uncertain' if result['errors'] else 'fail' if 'fail' in states
                        else 'pass' if all(s == 'pass' for s in states) else 'uncertain')
    return result


def summarize(results: list[dict]) -> dict:
    briefs = sorted({r['brief_id'] for r in results})
    return {'cases': len(results), 'briefs': len(briefs), 'brief_ids': briefs,
            'experimental_unit': 'brief; seeds and repeated operations on one parent are correlated',
            'status_counts': dict(Counter(r['status'] for r in results)),
            'fulfillment_counts': dict(Counter(r['fulfillment']['status'] for r in results)),
            'preflight_counts': dict(Counter(r['preflight']['status'] for r in results)),
            'false_success_counts': dict(Counter(r['false_success'] for r in results)),
            'scope_rule_comparison': {
                'baseline': 'counterfactual guard accepting any overlap with declared required regions; not the historical runtime policy',
                'current': 'every declared required pixel must be editable',
                'additional_scope_rejections': sum(r['preflight']['legacy_any_overlap_status'] == 'pass'
                                                   and r['preflight']['status'] == 'fail'
                                                   and any(0 < c['editable_fraction'] < 1 for c in r['preflight']['required_region_coverage']) for r in results),
                'heldout': False, 'artistic_improvement': None},
            'executed_cases': sum(r['execution'] == 'saved_child' for r in results),
            'measurement_errors': sum(bool(r['errors']) for r in results),
            'inference_calls': 0, 'quality_or_preference_claim': None}


def _write_new(path: Path, data):
    with path.open('x') as stream:
        stream.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def evaluate_suite(suite_path: Path | str, output_dir: Path | str, root: Path = Path('.')) -> Path:
    """Write a fresh report directory. Existing outputs and source artifacts are immutable."""
    suite_path, directory, root = Path(suite_path), Path(output_dir), Path(root)
    suite = load_suite(suite_path)
    directory.mkdir(parents=True, exist_ok=False)
    snapshot = suite.model_dump(mode='json')
    _write_new(directory / 'suite.json', snapshot)
    manifest = {'schema_version': 1, 'suite_id': suite.suite_id, 'split': suite.split,
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source_suite_sha256': digest(suite_path), 'snapshot_sha256': digest(directory / 'suite.json'),
                'code_sha256': digest(Path(__file__)),
                'measurement_code_sha256': {name: digest(Path(__file__).with_name(name))
                                           for name in ('edit_benchmark.py', 'verification.py')},
                'packages': {name: version(name) for name in ('numpy', 'Pillow', 'pydantic')},
                'root': str(root.resolve()), 'inference_calls': 0,
                'postprocessing': 'New offline sidecars; historical original measurements are not replaced'}
    _write_new(directory / 'manifest.json', manifest)
    results = []
    for case in suite.cases:
        result = measure_case(case, root)
        _write_new(directory / f'{case.case_id}.json', result)
        results.append(result)
    summary = summarize(results)
    _write_new(directory / 'summary.json', summary)
    lines = [f'# {suite.suite_id}', '', suite.description, '',
             f"{len(results)} cases across {summary['briefs']} brief(s). No inference calls.", '',
             '| Case | Execution | Scope preflight | Target | Fulfillment | Preservation | Success claim check |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in results:
        lines.append(f"| [{row['case_id']}]({row['case_id']}.json) | {row['execution']} | {row['preflight']['status']} | "
                     f"{row['target_existence']['status']} | {row['fulfillment']['status']} | {row['preservation']['status']} | {row['false_success']} |")
    lines += ['', '## Inspect original artifacts', '',
              '| Case | Saved parent | Saved child | Declared instruction |',
              '| --- | --- | --- | --- |']
    for case in suite.cases:
        parent_path = (root / case.parent.path).resolve()
        child_link = f'[child](<{(root / case.child.path).resolve()}>)' if case.child else 'unexecuted'
        instruction = case.instruction.replace('|', '\\|').replace('\n', ' ')
        lines.append(f'| {case.case_id} | [parent](<{parent_path}>) | {child_link} | {instruction} |')
    lines += ['', 'Scope checks test declared geometry. Assistant annotations and model claims remain attributed; neither supplies independent semantic truth.',
              'A failed preflight is a newly measured planning defect, not proof that a semantic output is absent. Unsupported success claims are distinct from confirmed false successes.',
              'Cached detector boxes are revalidated with both height and width checks; original historical statuses remain untouched.',
              'Cases are development diagnostics selected after observing failures; this is not a general success-rate estimate. Artistic quality, preference and readiness are not measured.', '',
              '## Retained measurement errors', '']
    lines.extend(f"- {r['case_id']}: {e}" for r in results for e in r['errors'])
    (directory / 'report.md').write_text('\n'.join(lines) + '\n')
    _write_new(directory / 'artifacts.sha256.json', {p.name: digest(p) for p in sorted(directory.iterdir()) if p.is_file()})
    return directory


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate', 'evaluate'])
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            suite = load_suite(args.suite)
            rows = {c.case_id: preflight(c, args.root) for c in suite.cases}
            print(json.dumps(rows, indent=2))
            return 1 if any(v['status'] == 'fail' for v in rows.values()) else 0
        if args.out is None:
            parser.error('evaluate requires --out pointing to a new directory')
        directory = evaluate_suite(args.suite, args.out, args.root)
        print(directory.resolve())
        summary = json.loads((directory / 'summary.json').read_text())
        return 1 if summary['measurement_errors'] else 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
