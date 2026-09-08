import json
from PIL import Image, ImageDraw
import pytest
from kusama.verification import assess_boxes, validate_spec, verify
from kusama.config import Config


def box(bounds, score=.9):
    return {'box': bounds, 'score': score}


def test_independent_geometry_pass_and_displacement_failure():
    before=[box([20,20,40,80])]
    spec={'operation':'resize','position_tolerance':3}
    assert assess_boxes(before,[box([25,50,35,80])],spec,.5,[30,80])['status']=='pass'
    r=assess_boxes(before,[box([35,50,45,80])],spec,.5,[30,80])
    assert r['status']=='fail'
    assert r['bottom_center_error_px']==10


def test_wrong_scale_and_duplicate_remnants_fail():
    before=[box([20,20,40,80])]; spec={'operation':'resize'}
    assert assess_boxes(before,before,spec,.5,[30,80])['status']=='fail'
    r=assess_boxes(before,[box([25,50,35,80]),box([50,20,70,80])],spec,.5,[30,80])
    assert r['status']=='fail' and r['duplicate_or_remnant']=='detected'


@pytest.mark.parametrize('bounds,width_ratio', [([5, 50, 55, 80], 2.5), ([29, 50, 31, 80], .1)])
def test_correct_height_and_position_cannot_hide_distorted_width(bounds, width_ratio):
    report = assess_boxes([box([20, 20, 40, 80])], [box(bounds)], {'operation': 'resize'}, .5, [30, 80])
    assert report['status'] == 'fail'
    assert report['relative_height_error'] == report['bottom_center_error_px'] == 0
    assert report['measured_width_ratio'] == width_ratio
    assert report['relative_width_error'] == pytest.approx(abs(width_ratio-.5)/.5)


def test_width_tolerance_defaults_to_height_tolerance_and_can_be_explicit():
    before = [box([20, 20, 40, 80])]
    after = [box([24.6, 50, 35.4, 80])]
    spec = {'operation': 'resize', 'height_tolerance': .05}
    report = assess_boxes(before, after, spec, .5, [30, 80])
    assert report['status'] == 'fail'
    assert report['tolerances']['width'] == .05
    report = assess_boxes(before, after, spec | {'width_tolerance': .1}, .5, [30, 80])
    assert report['status'] == 'pass'
    assert report['relative_width_error'] == pytest.approx(.08)


@pytest.mark.parametrize('stage', ['before', 'after'])
@pytest.mark.parametrize('invalid', [
    box([20, 20, 20, 80]), box([20, 80, 40, 20]), box([20, 20, 40, 20]),
    box([float('nan'), 20, 40, 80]), box([20, 20, float('inf'), 80]),
    box([20, 20, 40]), box([20, 20, 40, 80], float('nan')), {'box': [20, 20, 40, 80]},
    box([-1e308, 20, 1e308, 80]), None])
def test_invalid_observations_are_serializable_uncertainty(stage, invalid):
    before, after = [box([20, 20, 40, 80])], [box([25, 50, 35, 80])]
    if stage == 'before':
        before = [invalid]
    else:
        after = [invalid]
    report = assess_boxes(before, after, {'operation': 'resize'}, .5, [30, 80])
    assert report['status'] == 'uncertain'
    assert report['invalid_detections'][0]['stage'] == stage
    json.dumps(report, allow_nan=False)


def test_invalid_detector_box_does_not_break_verification_overlay():
    class Observer:
        provenance = {'method': 'test double'}
        def __init__(self):
            self.calls = 0
        def detect(self, *args):
            self.calls += 1
            return [box([20, 20, 40, 80])] if self.calls == 1 else [box([30, 50, float('nan'), 80])]
    parent = Image.new('RGB', (100, 100), 'white')
    plan = {'verification': {'observer': 'grounding_dino', 'operation': 'resize', 'label': 'person'},
            'scale': .5, 'anchor': [30, 80]}
    report, overlay = verify(parent, parent, None, plan, Config(), Observer())
    assert report['status'] == 'uncertain'
    json.dumps(report, allow_nan=False)
    assert overlay.tobytes() == parent.tobytes()


@pytest.mark.parametrize('after',[[],[box([25,50,35,80],.2)]])
def test_missing_and_weak_detections_are_uncertain(after):
    assert assess_boxes([box([20,20,40,80])],after,{'operation':'resize'},.5,[30,80])['status']=='uncertain'


def test_detector_nondetection_cannot_prove_removal():
    assert assess_boxes([box([20,20,40,80])],[],{'operation':'remove'})['status']=='uncertain'


def test_color_recolor_geometry_and_wrong_color():
    parent=Image.new('RGB',(64,64),'white');ImageDraw.Draw(parent).rectangle((20,20,40,40),fill='red')
    mask=Image.new('L',parent.size);ImageDraw.Draw(mask).rectangle((20,20,40,40),fill=255)
    plan={'verification':{'observer':'color','operation':'recolor','source_color':'red','target_color':'blue'}}
    r,_=verify(parent,parent,mask,plan,Config());assert r['status']=='fail'
    result=parent.copy();ImageDraw.Draw(result).rectangle((20,20,40,40),fill='blue')
    r,_=verify(parent,result,mask,plan,Config());assert r['status']=='pass'
    ImageDraw.Draw(result).rectangle((45,45,55,55),fill='blue')
    r,_=verify(parent,result,mask,plan,Config());assert r['status']=='fail'


def test_removal_requires_known_background_not_only_absent_color():
    parent=Image.new('RGB',(64,64),'white');ImageDraw.Draw(parent).ellipse((20,20,40,40),fill='red')
    mask=Image.new('L',parent.size);ImageDraw.Draw(mask).ellipse((20,20,40,40),fill=255)
    plan={'verification':{'observer':'color','operation':'remove','source_color':'red','background_rgb':[255,255,255]}}
    result=Image.new('RGB',parent.size,'white')
    assert verify(parent,result,mask,plan,Config())[0]['status']=='pass'
    ImageDraw.Draw(result).ellipse((20,20,40,40),fill='black')
    assert verify(parent,result,mask,plan,Config())[0]['status']=='fail'


@pytest.mark.parametrize('spec',[{'observer':'magic'}, {'observer':'grounding_dino','operation':'recolor','label':'person'},
    {'observer':'color','operation':'remove','source_color':'red'},
    {'observer':'color','operation':'resize','source_color':'red','height_tolerance':float('nan')},
    *[{'observer':'color','operation':'resize','source_color':'red','width_tolerance': value}
      for value in [float('nan'), float('inf'), -.1, 1.1, True, '0.2']]])
def test_invalid_verification_specs(spec):
    with pytest.raises(ValueError):validate_spec(spec)


def test_explicit_width_tolerance_spec_is_supported():
    validate_spec({'observer': 'grounding_dino', 'operation': 'resize', 'label': 'person', 'width_tolerance': .1})
