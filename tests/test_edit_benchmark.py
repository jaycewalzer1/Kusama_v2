"""Temporary fixtures verify contracts, not artistic or real inference performance."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
from pydantic import ValidationError

from kusama.edit_benchmark import (Asset, EditCase, EditSuite, digest, evaluate_suite,
                                    load_suite, main, measure_case, preflight, summarize)
from kusama.verification import DETECTOR_ID, DETECTOR_REVISION


def asset(path):
    return {'path': str(path), 'sha256': digest(path)}


def save(path, value):
    path.write_text(json.dumps(value))
    return asset(path)


@pytest.fixture
def example(tmp_path):
    pixels = np.zeros((24, 24, 3), dtype=np.uint8)
    pixels[5:15, 5:10] = 255
    parent = tmp_path / 'parent.png'
    Image.fromarray(pixels).save(parent)
    pixels[7:10, 7:10] = 128
    child = tmp_path / 'child.png'
    Image.fromarray(pixels).save(child)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:16, 4:16] = 255
    allowed = tmp_path / 'allowed.png'
    Image.fromarray(mask).save(allowed)
    return dict(case_id='case', brief_id='brief', operation='resize', instruction='Reduce the target uniformly by one half.',
                parent=asset(parent), child=asset(child), allowed_mask=asset(allowed),
                regions=[dict(id='target', box=[5,5,10,15], role='required_edit', origin='operator_annotation', reason='Fixture rectangle target.')],
                evidence_kind='fixture', method='masked', render_profile='fixture_only', authorship='test fixture')


def detector(example, tmp_path, before=None, after=None):
    source = save(tmp_path / 'detector.json', {'image_sha256': example['child']['sha256'], 'status': 'pass',
        'before_detections': before if before is not None else [{'box':[5,5,10,15], 'score':.9}],
        'after_detections': after if after is not None else [{'box':[6.25,10,8.75,15], 'score':.9}],
        'observer': {'model_id': DETECTOR_ID, 'revision': DETECTOR_REVISION}})
    binding = save(tmp_path/'binding.json', {'parent_sha256':example['parent']['sha256']})
    example['cached_boxes'] = dict(observation=source,parent_binding=binding,scale=.5,anchor=[7.5,15])


def claim(example, tmp_path, status='success'):
    source = save(tmp_path/'claim.json', {'parent':example['parent']['sha256'], 'child':example['child']['sha256'], 'claim':'The target was resized.'})
    example['model_claim'] = dict(source=source,json_pointer='/claim',parent_sha256_pointer='/parent',child_sha256_pointer='/child',assertion=status,author='fixture Qwen substitute',interpretation_origin='assistant_annotation')


def test_exact_preservation_does_not_prove_fulfillment(example):
    result = measure_case(example)
    assert result['preservation']['status'] == 'pass'
    assert result['preservation']['total_changed_pixels'] == 9
    assert result['fulfillment']['status'] == 'uncertain'
    assert result['status'] == 'uncertain'
    assert result['artistic_quality'] == 'not_measured'


def test_required_support_rejects_partial_mask_coverage(example):
    example['regions'][0]['box'] = [2,5,10,15]
    result = preflight(example)
    assert result['status'] == 'fail'
    assert result['required_region_coverage'][0]['editable_fraction'] == .75
    assert result['legacy_any_overlap_status'] == 'pass'


def test_relationship_requires_localized_endpoints(example):
    example['operation']='connect'
    assert preflight(example)['status']=='uncertain'


def test_protected_edit_conflict_is_not_hidden(example):
    example['regions'].append(dict(id='detail',box=[6,7,9,12],role='protected',origin='operator_annotation',reason='Fixture protected detail.'))
    assert preflight(example)['status']=='fail'
    assert measure_case(example)['preservation']['status']=='fail'


def test_non_detection_is_not_removal_success(example,tmp_path):
    example['operation']='remove'
    detector(example,tmp_path,after=[])
    claim(example,tmp_path)
    result=measure_case(example)
    assert result['target_existence']['status']=='pass'
    assert result['fulfillment']['status']=='uncertain'
    assert result['false_success']=='unsupported'


def test_old_passing_resize_revalidated_with_width(example,tmp_path):
    detector(example,tmp_path,after=[{'box':[5,10,10,15],'score':.9}])
    claim(example,tmp_path)
    result=measure_case(example)
    assert result['fulfillment']['original_saved_status']=='pass'
    assert result['fulfillment']['status']=='fail'
    assert result['false_success']=='confirmed'
    assert result['fulfillment']['measured_width_ratio']==1


def test_cached_detector_pass_is_bounded_to_measurement(example,tmp_path):
    detector(example,tmp_path)
    result=measure_case(example)
    assert result['status']=='pass'
    assert result['inference_calls']==0
    assert result['artistic_quality']=='not_measured'


@pytest.mark.parametrize('field', ['parent_sha256','image_sha256'])
def test_cached_detector_binding_mismatch_retained(example,tmp_path,field):
    detector(example,tmp_path)
    part='parent_binding' if field=='parent_sha256' else 'observation'
    path=Path(example['cached_boxes'][part]['path'])
    d=json.loads(path.read_text());d[field]='0'*64
    example['cached_boxes'][part]=save(path,d)
    result=measure_case(example)
    assert result['errors'] and result['fulfillment']['status']=='uncertain'
    assert result['false_success']=='not_assessed'


def test_claim_for_different_child_not_admitted(example,tmp_path):
    detector(example,tmp_path)
    claim(example,tmp_path)
    path=Path(example['model_claim']['source']['path'])
    d=json.loads(path.read_text());d['child']='1'*64
    example['model_claim']['source']=save(path,d)
    result=measure_case(example)
    assert result['status']=='uncertain'
    assert 'not bound' in result['errors'][0]


def test_missing_image_is_retained_not_excluded(example):
    Path(example['child']['path']).unlink()
    result=measure_case(example)
    assert result['errors'] and result['status']=='uncertain'


def test_changed_parent_bytes_are_rejected(example):
    Image.new('RGB',(24,24),'red').save(example['parent']['path'])
    result=measure_case(example)
    assert 'hash mismatch' in result['errors'][0]
    assert result['preflight']['status']=='fail'


def test_unexecuted_task_remains_unexecuted(example):
    example.pop('child')
    result=measure_case(example)
    assert result['execution']=='unexecuted'
    assert result['fulfillment']['status']=='uncertain'


def test_detail_pixels_are_independent_narrow_check(example):
    example['operation']='preserve_detail'
    example['regions']=[dict(id='detail',box=[0,0,4,4],role='protected',origin='operator_annotation',reason='Exact fixture detail pixels.')]
    result=measure_case(example)
    assert result['fulfillment']['status']=='pass'
    example['regions'][0]['box']=[7,7,10,10]
    assert measure_case(example)['fulfillment']['status']=='fail'


def test_suite_development_only_and_fixture_isolation(example):
    base=dict(suite_id='suite',description='fixture',cases=[example],attribution='fixture')
    with pytest.raises(ValidationError):
        EditSuite.model_validate(base|{'split':'heldout'})
    second=deepcopy(example);second.update(case_id='real',evidence_kind='historical_diagnostic')
    with pytest.raises(ValidationError):
        EditSuite.model_validate(base|{'cases':[example,second]})


def test_report_is_immutable_and_keeps_all_failures(example,tmp_path):
    other=deepcopy(example);other['case_id']='missing';other['child']['path']=str(tmp_path/'no-child.png')
    suite=tmp_path/'suite.json'
    save(suite,dict(suite_id='suite',description='fixture',cases=[example,other],attribution='fixture'))
    out=evaluate_suite(suite,tmp_path/'output')
    summary=json.loads((out/'summary.json').read_text())
    assert summary['cases']==2 and summary['briefs']==1 and summary['measurement_errors']==1
    assert (out/'missing.json').exists()
    checks=json.loads((out/'artifacts.sha256.json').read_text())
    assert all(digest(out/name)==expected for name,expected in checks.items())
    with pytest.raises(FileExistsError):
        evaluate_suite(suite,out)


def test_actual_saved_suite_manifest_covers_five_prospective_operations():
    suite=load_suite('configs/edit_benchmark/development_v1.json')
    assert len({c.operation for c in suite.cases})==5
    assert all(c.child is None and c.evidence_kind=='prospective' for c in suite.cases)
    cached=load_suite('configs/edit_benchmark/saved_failures_v1.json')
    assert {c.brief_id for c in cached.cases}=={'mountain_person','care_control'}
    assert all(c.child and c.evidence_kind=='historical_diagnostic' for c in cached.cases)


def test_cli_evaluate_reports_errors_through_exit_status(example,tmp_path):
    Path(example['child']['path']).unlink()
    suite=tmp_path/'suite.json';save(suite,dict(suite_id='suite',description='fixture',cases=[example],attribution='fixture'))
    assert main(['evaluate','--suite',str(suite),'--out',str(tmp_path/'out')])==1
