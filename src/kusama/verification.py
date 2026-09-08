"""Independent observations of saved pixels, never inferred from edit construction."""
from __future__ import annotations

import math
from numbers import Real
import time
import numpy as np
from PIL import Image, ImageDraw

DETECTOR_ID = 'IDEA-Research/grounding-dino-tiny'
DETECTOR_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'


def validate_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError('verification must be an object')
    allowed = {'observer', 'label', 'operation', 'height_tolerance', 'width_tolerance', 'position_tolerance',
               'source_color', 'target_color', 'background_rgb'}
    if spec.keys() - allowed or spec.get('observer') not in ('grounding_dino', 'color'):
        raise ValueError('Unknown verification observer or fields')
    if spec.get('operation') not in ('resize', 'remove', 'recolor'):
        raise ValueError('verification.operation must be resize, remove or recolor')
    for key, default, upper in [('height_tolerance', 0.2, 1),
                                ('width_tolerance', spec.get('height_tolerance', 0.2), 1),
                                ('position_tolerance', 6, 100)]:
        value = spec.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError(f'Invalid {key}')
    if spec['observer'] == 'grounding_dino':
        if not isinstance(spec.get('label'), str) or not spec['label'].strip():
            raise ValueError('Detector requires a target label')
        if spec['operation'] == 'recolor':
            raise ValueError('Bounding-box detection cannot verify color')
    else:
        for key in ['source_color'] + (['target_color'] if spec['operation'] == 'recolor' else []):
            if spec.get(key) not in ('red', 'green', 'blue'):
                raise ValueError('Color calibration supports red, green and blue')
        if spec['operation'] == 'remove':
            rgb = spec.get('background_rgb')
            if not isinstance(rgb, list) or len(rgb) != 3 or any(type(v) is not int or not 0 <= v <= 255 for v in rgb):
                raise ValueError('Calibrated removal requires the known background_rgb')


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area / union if union else 0


def nms(boxes):
    kept = []
    for item in sorted(boxes, key=lambda x: -x['score']):
        if all(iou(item['box'], previous['box']) < 0.5 for previous in kept):
            kept.append(item)
    return kept


class GroundingObserver:
    """Frozen CPU detector, loaded after the editing backend is released."""
    def __init__(self, cache_dir, local_files_only=False):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        kwargs = dict(revision=DETECTOR_REVISION, cache_dir=cache_dir, local_files_only=local_files_only)
        self.processor = AutoProcessor.from_pretrained(DETECTOR_ID, use_fast=False, **kwargs)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(DETECTOR_ID, **kwargs).eval().requires_grad_(False)
        self.provenance = {'model_id': DETECTOR_ID, 'revision': DETECTOR_REVISION,
                           'device': 'cpu', 'dtype': 'float32', 'strong_threshold': 0.30,
                           'weak_threshold': 0.15, 'nms_iou': 0.5}

    def detect(self, image, label, crop):
        inputs = self.processor(images=image.crop(crop), text=label.rstrip('.') + '.', return_tensors='pt')
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.15, text_threshold=0.2,
            target_sizes=[(crop[3]-crop[1], crop[2]-crop[0])])[0]
        boxes = []
        for box, score in zip(result['boxes'].tolist(), result['scores'].tolist()):
            boxes.append({'box': [box[0]+crop[0], box[1]+crop[1], box[2]+crop[0], box[3]+crop[1]], 'score': score})
        return nms(boxes)


def color_mask(image, color):
    hsv = np.asarray(image.convert('HSV'))
    hue = hsv[:, :, 0].astype(float) / 255
    center = {'red': 0, 'green': 1/3, 'blue': 2/3}[color]
    distance = np.minimum(abs(hue-center), 1-abs(hue-center))
    return (distance < 0.09) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 55)


def components(mask, offset=(0, 0)):
    seen = mask.copy()
    boxes = []
    for y, x in zip(*np.nonzero(mask)):
        if not seen[y, x]:
            continue
        stack = [(int(x), int(y))]; seen[y, x] = False
        count = 0; left = right = int(x); top = bottom = int(y)
        while stack:
            xx, yy = stack.pop(); count += 1
            left = min(left, xx); right = max(right, xx); top = min(top, yy); bottom = max(bottom, yy)
            for nx, ny in ((xx-1, yy), (xx+1, yy), (xx, yy-1), (xx, yy+1)):
                if 0 <= ny < seen.shape[0] and 0 <= nx < seen.shape[1] and seen[ny, nx]:
                    seen[ny, nx] = False; stack.append((nx, ny))
        if count >= 8:
            boxes.append({'box': [left+offset[0], top+offset[1], right+1+offset[0], bottom+1+offset[1]],
                          'score': 1.0, 'pixels': count})
    return boxes


def valid_box(bounds):
    """Reject malformed observations before arithmetic or preview drawing."""
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return False
    if any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in bounds):
        return False
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    return math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0


def assess_boxes(before, after, spec, scale=None, anchor=None):
    """Evaluate measured boxes; weak/no detections remain uncertain, not absence proof."""
    invalid = []
    observations = {}
    for stage, detections in [('before', before), ('after', after)]:
        observations[stage] = []
        for index, detection in enumerate(detections):
            score = detection.get('score') if isinstance(detection, dict) else None
            if (not isinstance(detection, dict) or not valid_box(detection.get('box'))
                    or isinstance(score, bool) or not isinstance(score, Real)
                    or not math.isfinite(score) or not 0 <= score <= 1):
                # String evidence keeps NaN/Infinity and malformed values out of JSON numbers.
                invalid.append({'stage': stage, 'index': index, 'observation': repr(detection)})
            else:
                observations[stage].append(detection)
    before, after = observations['before'], observations['after']
    strong_before = [b for b in before if b['score'] >= 0.3]
    strong_after = [b for b in after if b['score'] >= 0.3]
    result = {'status': 'uncertain', 'before_detections': before, 'after_detections': after,
              'duplicate_or_remnant': 'uncertain'}
    if invalid:
        return result | {'reason': 'Invalid observed box or confidence', 'invalid_detections': invalid}
    if len(strong_before) != 1:
        return result | {'reason': 'Reference target not uniquely observed'}
    if len(strong_after) > 1:
        return result | {'status': 'fail', 'duplicate_or_remnant': 'detected', 'reason': 'Multiple target detections'}
    if spec['operation'] == 'remove':
        return result | {'status': 'fail' if strong_after else 'uncertain',
                         'reason': 'Target remains' if strong_after else 'Nondetection does not prove removal'}
    if len(strong_after) != 1 or len(after) > 1:
        return result | {'reason': 'Output target missing, weak, or ambiguous'}
    if (isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale <= 0
            or not isinstance(anchor, (list, tuple)) or len(anchor) != 2
            or any(isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) for v in anchor)):
        return result | {'reason': 'Invalid requested scale or anchor'}
    a, b = strong_before[0]['box'], strong_after[0]['box']
    expected = [anchor[i % 2] + scale * (v-anchor[i % 2]) for i, v in enumerate(a)]
    measured_ratio = (b[3]-b[1]) / (a[3]-a[1])
    measured_width_ratio = (b[2]-b[0]) / (a[2]-a[0])
    position_error = math.hypot((b[0]+b[2]-expected[0]-expected[2])/2, b[3]-expected[3])
    height_error = abs(measured_ratio-scale) / scale
    width_error = abs(measured_width_ratio-scale) / scale
    if not all(math.isfinite(v) for v in [*expected, measured_ratio, measured_width_ratio,
                                         position_error, height_error, width_error]):
        return result | {'reason': 'Observed geometry cannot be measured with finite precision'}
    height_tolerance = spec.get('height_tolerance', .2)
    width_tolerance = spec.get('width_tolerance', height_tolerance)
    position_tolerance = spec.get('position_tolerance', 6)
    passed = (height_error <= height_tolerance and width_error <= width_tolerance
              and position_error <= position_tolerance)
    return result | {'status': 'pass' if passed else 'fail', 'duplicate_or_remnant': 'not_detected',
                     'measured_height_ratio': measured_ratio, 'relative_height_error': height_error,
                     'measured_width_ratio': measured_width_ratio, 'relative_width_error': width_error,
                     'tolerances': {'height': height_tolerance, 'width': width_tolerance,
                                    'position_px': position_tolerance},
                     'bottom_center_error_px': position_error, 'expected_box': expected,
                     'reason': 'Measured geometry within tolerances' if passed else 'Measured geometry outside tolerances'}


def verify(parent, result, obj, plan, config, observer=None):
    spec = plan.get('verification')
    if spec is None:
        return {'status': 'not_requested', 'artistic_review': 'pending'}, None
    started = time.monotonic()
    crop = plan.get('crop', [0, 0, *parent.size])
    if spec['observer'] == 'grounding_dino':
        observer = observer or GroundingObserver(config.cache_dir, config.local_files_only)
        before = observer.detect(parent, spec['label'], crop)
        after = observer.detect(result, spec['label'], crop)
        report = assess_boxes(before, after, spec, plan.get('scale'), plan.get('anchor'))
        report['observer'] = observer.provenance
        report['limitation'] = 'Learned box estimates can miss small objects or fragments; no semantic absence or artistic guarantee'
    else:
        source = color_mask(parent, spec['source_color'])
        if obj is None or np.count_nonzero(source & (np.asarray(obj) > 0)) < 8:
            return {'status': 'uncertain', 'reason': 'Reference color not supported by object annotation'}, None
        region = np.zeros(source.shape, dtype=bool)
        region[crop[1]:crop[3], crop[0]:crop[2]] = True
        source &= region
        target = color_mask(result, spec.get('target_color', spec['source_color'])) & region
        before, after = components(source), components(target)
        if spec['operation'] == 'resize':
            report = assess_boxes(before, after, spec, plan['scale'], plan['anchor'])
        else:
            pixels = np.asarray(obj) > 0
            remaining = float(np.count_nonzero(color_mask(result, spec['source_color']) & pixels) / pixels.sum())
            report = {'before_detections': before, 'after_detections': after, 'source_color_remaining_fraction': remaining}
            if spec['operation'] == 'recolor':
                coverage = float(np.count_nonzero(target & pixels) / pixels.sum())
                box_iou = iou(before[0]['box'], after[0]['box']) if len(before) == len(after) == 1 else 0
                passed = coverage >= .8 and remaining <= .05 and box_iou >= .8
                report.update(target_color_coverage=coverage, measured_box_iou=box_iou,
                              duplicate_or_remnant='detected' if len(after) > 1 else 'not_detected')
            else:
                error = float(np.abs(np.asarray(result).astype(float)[pixels] - spec['background_rgb']).mean())
                passed = not after and error <= 15
                report.update(background_mae=error, duplicate_or_remnant='detected' if after else 'not_detected')
            report['status'] = 'pass' if passed else 'fail'
        report['observer'] = {'method': 'independent HSV connected components', 'hue_radius': .09,
                              'min_saturation': 90, 'min_value': 55, 'min_component_pixels': 8}
        report['limitation'] = 'Calibration scenes only; color absence cannot establish general semantic removal'
    report.update(runtime_seconds=time.monotonic()-started, artistic_review='pending')
    overlay = result.copy(); draw = ImageDraw.Draw(overlay)
    for box in report.get('after_detections', []):
        if valid_box(box.get('box')):
            draw.rectangle(box['box'], outline='red', width=2)
    return report, overlay


def repair_metrics(parent, result, repair):
    """Descriptive local drift and boundary discontinuity; never an artistic score."""
    mask = np.asarray(repair) > 0
    a, b = np.asarray(parent).astype(float), np.asarray(result).astype(float)
    changed = np.any(a != b, axis=2)
    edges = []
    for axis in (0, 1):
        pair = np.diff(mask.astype(int), axis=axis) != 0
        delta = np.abs(np.diff(b, axis=axis)).mean(axis=2) - np.abs(np.diff(a, axis=axis)).mean(axis=2)
        edges.extend(delta[pair].tolist())
    return {'repair_pixels': int(mask.sum()), 'changed_repair_pixels': int((mask & changed).sum()),
            'repair_rgb_mae': float(np.abs(a-b)[mask].mean()) if mask.any() else 0,
            'mean_boundary_gradient_increase': float(np.mean(edges)) if edges else 0,
            'limitation': 'Boundary gradient change can reflect legitimate content changes; visual seam review is still required'}
