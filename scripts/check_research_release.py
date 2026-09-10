"""Run the research release gate; skipped live/processor checks cannot pass.

Consumes a completed, separately executed live panel run. Does not start model
inference or retry runs. The ordinary fixture suite alone is not this gate.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-run', type=Path, required=True)
    parser.add_argument('--dashboard', required=True)
    parser.add_argument('--recorded-incident', type=Path, required=True)
    parser.add_argument('--recorded-hypothesis-incident', type=Path, required=True)
    parser.add_argument('--recorded-context', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    for directory in (args.live_run, args.recorded_incident, args.recorded_hypothesis_incident):
        if not directory.is_dir():
            parser.error(f'Existing run directory required: {directory}')
    if not args.recorded_context.is_file():
        parser.error('Existing saved context required')
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in
           (args.live_run, args.recorded_incident, args.recorded_hypothesis_incident, args.recorded_context.parent)):
        parser.error('Gate output must be outside original run directories')
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, KUSAMA_PROCESSOR_ACCEPTANCE='1',
        KUSAMA_RECORDED_RESEARCH=str(args.recorded_incident.resolve()),
        KUSAMA_RECORDED_HYPOTHESIS=str(args.recorded_hypothesis_incident.resolve()),
        KUSAMA_RECORDED_CONTEXT=str(args.recorded_context.resolve()),
        KUSAMA_LIVE_RESEARCH=str(args.live_run.resolve()), KUSAMA_LIVE_DASHBOARD=args.dashboard)
    xml_path = args.output.resolve() / 'pytest.xml'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', f'--junitxml={xml_path}'], cwd=repo, env=env)
    suites = list(ET.parse(xml_path).getroot().iter('testsuite')) if xml_path.exists() else []
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests','failures','errors','skipped')}
    passed = result.returncode == 0 and counts['tests'] > 0 and not any(counts[k] for k in ('failures','errors','skipped'))
    summary = {'status': 'passed' if passed else 'failed', 'pytest_exit_code': result.returncode,
        **counts, 'live_run': str(args.live_run.resolve()), 'dashboard': args.dashboard,
        'generation_dispatched_by_gate': False,
        'scope': 'Software, cached processor, archived incident replay and current unmocked panel-process evidence.',
        'artistic_quality': 'Requires human assessment; not certified by this gate.'}
    (args.output / 'result.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
