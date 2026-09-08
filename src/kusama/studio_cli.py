"""Small explicit controls for persistent studio projects."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import parse_args
from .experiment import handle_termination
from .studio import Studio, read_projects


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts'))
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('create', 'inquire', 'reference', 'start', 'propose', 'step', 'run',
                 'select', 'decide', 'observe', 'stop', 'finish', 'show', 'list'):
        sub = commands.add_parser(name)
        if name != 'list':
            sub.add_argument('--project', required=True)
        if name in ('start', 'propose', 'step', 'run', 'select', 'stop', 'finish'):
            sub.add_argument('--episode', required=True)
        if name in ('create', 'inquire', 'select', 'decide', 'observe', 'stop', 'finish'):
            sub.add_argument('--author', required=True)
        if name in ('inquire', 'select', 'decide', 'stop', 'finish'):
            sub.add_argument('--reason', required=True)
        if name in ('create', 'inquire'):
            sub.add_argument('--inquiry', type=Path, required=True)
        if name == 'create':
            sub.add_argument('--name', required=True)
            sub.add_argument('--parent', type=Path, required=True)
        if name in ('reference', 'propose'):
            sub.add_argument('--file', type=Path, required=True)
        if name == 'start':
            sub.add_argument('--max-calls', type=int, required=True)
            sub.add_argument('--max-actions', type=int, default=12)
            sub.add_argument('--max-rounds', type=int, default=3)
        if name in ('select', 'decide', 'observe'):
            sub.add_argument('--artifact', required=True)
        if name == 'decide':
            sub.add_argument('--decision', choices=['retain', 'reject'], required=True)
        if name == 'observe':
            sub.add_argument('--note', required=True)
            sub.add_argument('--origin', choices=['operator', 'model_observation', 'development_example'], required=True)
    args, backend_args = parser.parse_known_args(argv)
    if backend_args and args.command != 'start':
        parser.error('Backend settings are accepted only by start; later commands use frozen settings')
    try:
        if args.command in ('list', 'show'):
            result = read_projects(args.artifacts, getattr(args, 'project', None))
            if args.command == 'show':
                if not result:
                    raise ValueError('Unknown project')
                result = result[0]
        else:
            with Studio(args.artifacts) as studio, handle_termination():
                p = args.project
                c = args.command
                if c == 'create':
                    result = studio.create(p, args.name, json.loads(args.inquiry.read_text()), args.parent, args.author)
                elif c == 'inquire':
                    result = studio.revise(p, json.loads(args.inquiry.read_text()), args.reason, args.author)
                elif c == 'reference':
                    result = studio.add_reference(p, json.loads(args.file.read_text()))
                elif c == 'start':
                    config, check = parse_args(backend_args)
                    if check:
                        raise ValueError('Use kusama-edit for environment checks')
                    result = studio.start(p, args.episode, replace(config, output_dir=args.artifacts.resolve()),
                                          args.max_calls, args.max_actions, args.max_rounds)
                elif c == 'propose':
                    result = studio.propose(p, args.episode, args.file)
                elif c in ('step', 'run'):
                    result = getattr(studio, c)(p, args.episode)
                elif c == 'select':
                    result = studio.select(p, args.episode, args.artifact, args.reason, args.author)
                elif c == 'decide':
                    result = studio.decide(p, args.artifact, args.decision, args.reason, args.author)
                elif c == 'observe':
                    result = studio.observe(p, args.artifact, args.note, args.author, args.origin)
                else:
                    result = studio.close_episode(p, args.episode, args.reason, args.author, finish=c == 'finish')
        print(json.dumps(result, indent=2, default=str))
        return 0
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f'kusama-studio: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
