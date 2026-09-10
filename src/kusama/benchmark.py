"""Small explicit experiment suites with a durable global inference budget."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from .artifacts import save_image
from .config import parse_args
from .controlled import load_plan, run
from .experiment import handle_termination
from .ledger import LedgerError


def load_suite(path):
    suite = json.loads(path.read_text())
    if not isinstance(suite, dict) or set(suite) != {'name', 'max_calls', 'cases'}:
        raise ValueError('Suite requires exactly name, max_calls and cases')
    if type(suite['max_calls']) is not int or not 1 <= suite['max_calls'] <= 20:
        raise ValueError('Suite max_calls must be 1 through 20')
    if not isinstance(suite['name'], str) or not isinstance(suite['cases'], list) or not suite['cases']:
        raise ValueError('Suite needs a name and nonempty cases')
    seen = set()
    for case in suite['cases']:
        if not isinstance(case, dict) or set(case) != {'id', 'plan'} or not isinstance(case['id'], str) or not isinstance(case['plan'], str):
            raise ValueError('Each case requires string id and plan')
        if not case['id'] or case['id'] in seen:
            raise ValueError('Case ids must be unique and nonempty')
        seen.add(case['id'])
        load_plan(path.parent / case['plan'])
    return suite


def reporting_error(report, stage, exc, case_id=None):
    error = {'stage': stage, 'error': f'{type(exc).__name__}: {exc}'}
    if case_id is not None:
        error['case'] = case_id
    report.setdefault('reporting_errors', []).append(error)
    return error['error']


def summarize(directory, report):
    rows = []
    for case in report['cases']:
        row = {'case': case['id'], 'status': case['status'], 'run': case.get('run', ''),
               'calls': 0, 'seconds': 0, 'verification': 'unavailable', 'protected_changed': '',
               'edit_pixels': '', 'height_ratio': '', 'width_ratio': '', 'relative_width_error': '',
               'position_error_px': '', 'boundary_gradient_increase': '', 'attempt_index': '',
               'attempt_status': '', 'attempt_error': '', 'report_error': '',
               'effective_protected_changed': '', 'visual_review': 'pending', 'model_repair_assessment': 'unreviewed'}
        if case.get('run') and case['status'] != 'skipped_budget':
            try:
                manifest = json.loads((Path(case['run']) / 'manifest.json').read_text())
                seconds = manifest.get('runtime_seconds')
                row.update(calls=manifest.get('model_calls', 0),
                           seconds=round(seconds, 2) if isinstance(seconds, (int, float)) else '')
                row['visual_review'] = manifest.get('visual_review', 'pending')
                attempts = manifest.get('attempts', [])
                if attempts:
                    # A failed latest attempt must not borrow a prior attempt's passing check.
                    attempt = attempts[-1]
                    verification = attempt.get('verification') or {}
                    preservation = attempt.get('preservation') or {}
                    row.update(attempt_index=len(attempts)-1, attempt_status=attempt.get('status', 'unavailable'),
                               attempt_error=attempt.get('error', ''),
                               verification=verification.get('status', 'unavailable'),
                               protected_changed=preservation.get('outside_changed_pixels', ''),
                               edit_pixels=preservation.get('total_changed_pixels', ''),
                               height_ratio=verification.get('measured_height_ratio', ''),
                               width_ratio=verification.get('measured_width_ratio', ''),
                               relative_width_error=verification.get('relative_width_error', ''),
                               position_error_px=verification.get('bottom_center_error_px', ''),
                               effective_protected_changed=(attempt.get('effective_preservation') or {}).get('outside_changed_pixels', ''),
                               boundary_gradient_increase=(attempt.get('repair_metrics') or {}).get('mean_boundary_gradient_increase', ''))
                    image_hash = verification.get('image_sha256')
                    observations = [o for o in manifest.get('model_observations', [])
                                    if image_hash and o.get('image_sha256') == image_hash]
                    if observations:
                        row['model_repair_assessment'] = observations[-1].get('repair_defect', 'unreviewed')
            except Exception as exc:
                row['report_error'] = reporting_error(report, 'summary_case', exc, case['id'])
        rows.append(row)
    if rows:
        with (directory/'results.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report['summary'] = {'cases': len(rows), 'verification_counts': {status: sum(r['verification'] == status for r in rows)
                            for status in ('pass', 'fail', 'uncertain', 'not_requested', 'unavailable')},
                         'artistic_acceptance': 'not_automated'}
    lines = [f"# {report['name']}", '', f"Status: {report['status']}. Model calls: {report['model_calls']}/{report['max_calls']}.", '',
             '| Case | Case / last attempt status | Independent check | Protected pixels changed | Calls | Seconds | Model visual assessment |', '| --- | --- | --- | ---: | ---: | ---: | --- |']
    for r in rows:
        lines.append(f"| {r['case']} | {r['status']} / {r['attempt_status']} | {r['verification']} | {r['protected_changed']} | {r['calls']} | {r['seconds']} | {r['model_repair_assessment']} |")
    lines += ['', 'These are development cases, not statistical evidence of general performance. A passing independent check applies only to its stated observer and tolerance. Artistic/repair review remains separate.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    return rows


def contact_sheet(directory, report):
    panels = []
    for case in report['cases']:
        if not case.get('run') or case.get('status') == 'skipped_budget':
            continue
        try:
            p = Path(case['run']); manifest = json.loads((p/'manifest.json').read_text())
            indices = [i for i in range(len(manifest.get('attempts', []))) if (p/f'attempt_{i}_result.png').exists()]
            if not indices:
                continue
            i = indices[-1]
            plan = json.loads((p/'plan.json').read_text())
            crop = plan.get('crop')
            images = []
            for filename in ('parent.png', f'attempt_{i}_result.png', f'attempt_{i}_detections.png'):
                target = p/filename
                if not target.exists():
                    target = p/f'attempt_{i}_result.png'
                with Image.open(target) as image:
                    image = image.convert('RGB')
                    if crop:
                        image = image.crop(crop)
                    image.thumbnail((320, 320))
                    images.append(image.copy())
            panels.append((f"{case['id']} (attempt {i})", images))
        except Exception as exc:
            reporting_error(report, 'contact_sheet_case', exc, case['id'])
    if not panels:
        return
    sheet = Image.new('RGB', (960, 360*len(panels)), 'white')
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default(size=16)
    for n, (title, images) in enumerate(panels):
        for col, (label, image) in enumerate(zip(('Original crop', 'Result crop', 'Observed boxes'), images)):
            draw.text((col*320+5, n*360+6), f'{title}: {label}', fill='black', font=font)
            sheet.paste(image, (col*320, n*360+35))
    save_image(sheet, directory/'comparison.png')


def run_suite(path=None, config=None, runner=run, **kwargs):
    from .suite_execution import run_suite_job
    return run_suite_job(path, config, runner, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run an explicit controlled-edit benchmark suite')
    parser.add_argument('--suite', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--job-id')
    parser.add_argument('--validate-only', action='store_true')
    args, rest = parser.parse_known_args(argv)
    if args.resume and (rest or args.suite or args.job_id or args.validate_only):
        parser.error('--resume uses saved inputs/settings; omit other flags')
    if not args.resume and not args.suite:
        parser.error('Supply --suite or --resume')
    config, check = parse_args(rest) if not args.resume else (None, False)
    try:
        if check:
            raise ValueError('Use kusama-first-edit --check-environment')
        if args.validate_only:
            print(json.dumps(load_suite(args.suite))); return 0
        with handle_termination():
            directory = run_suite(args.suite, config, resume_dir=args.resume, job_id=args.job_id)
        print(directory.resolve())
        report = json.loads((directory/'benchmark.json').read_text())
        return 0 if report['status'] == 'completed' and not report.get('reporting_errors') else 1
    except (ValueError, OSError, TypeError, KeyError, LedgerError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
