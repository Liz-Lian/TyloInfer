"""Run offline reference generation or compare already persisted results."""

import argparse
import json
import sys
from pathlib import Path

from .case import Case
from .compare import compare


def main(argv=None):
    """Run the requested offline tool; comparison never imports Torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='Create new fixed inputs and reference results')
    generate.add_argument(
        'kind',
        choices=('rms', 'basic', 'rope', 'conv', 'gelu', 'tq', 'flash', 'mx', 'gdn', 'nemotron'),
    )
    generate.add_argument('output', type=Path)
    generate.add_argument(
        '--suite',
        help='Required for TQ, Flash-Next, MX, GDN and Nemotron generators',
    )
    generate.add_argument('--device', default='cuda')
    rerun = commands.add_parser(
        'reference', help='Recompute a reference from existing fixed inputs'
    )
    rerun.add_argument('case', type=Path)
    rerun.add_argument(
        'output', type=Path, help='A new result directory; existing files are never replaced'
    )
    rerun.add_argument('--device', default='cuda', help='CUDA device for Torch references')
    check = commands.add_parser('compare', help='Validate and compare fixed result artifacts')
    check.add_argument('case', type=Path)
    check.add_argument('--reference', required=True, type=Path)
    check.add_argument('--actual', type=Path)
    check.add_argument(
        '--report', type=Path, help='Create a report file without replacing an old report'
    )
    args = parser.parse_args(argv)
    try:
        if args.command == 'generate':
            if (args.kind in {'tq', 'flash', 'mx', 'gdn', 'nemotron'}) != (args.suite is not None):
                parser.error('--suite is required only for TQ, Flash-Next, MX, GDN and Nemotron')
            if args.kind == 'tq':
                from .tq_ref import generate

                paths = generate(args.output, args.suite, args.device)
            elif args.kind == 'flash':
                from .flash_ref import generate

                paths = generate(args.output, args.suite)
            elif args.kind == 'mx':
                from .mx_ref import generate

                paths = generate(args.output, args.suite, args.device)
            elif args.kind == 'gdn':
                from .gdn_ref import generate

                paths = generate(args.output, args.suite, args.device)
            elif args.kind == 'nemotron':
                if args.suite == 'backbone':
                    from .nemotron_text_ref import generate
                elif args.suite == 'vision':
                    from .nemotron_vision_ref import generate
                else:
                    parser.error('Nemotron suite must be backbone or vision')
                paths = generate(args.output, args.suite)
            else:
                from .torch_ref import generate

                paths = generate(args.output, args.kind, args.device)
            print(json.dumps({'status': 'generated', 'cases': [str(path) for path in paths]}))
            return 0
        if args.command == 'reference':
            case = Case.load(args.case)
            generator = case.data['source'].get('generator')
            if generator == 'tests/numerics/torch_ref.py':
                from .torch_ref import reference

                path = reference(case, args.output, args.device)
            elif generator == 'tests/numerics/tq_ref.py':
                from .tq_ref import reference

                path = reference(case, args.output, args.device)
            elif generator == 'tests/numerics/flash_ref.py':
                from .flash_ref import reference

                path = reference(case, args.output)
            elif generator == 'tests/numerics/mx_ref.py':
                from .mx_ref import reference

                path = reference(case, args.output, args.device)
            elif generator == 'tests/numerics/gdn_ref.py':
                from .gdn_ref import reference

                path = reference(case, args.output, args.device)
            elif generator == 'tests/numerics/nemotron_text_ref.py':
                from .nemotron_text_ref import reference

                path = reference(case, args.output)
            elif generator == 'tests/numerics/nemotron_vision_ref.py':
                from .nemotron_vision_ref import reference

                path = reference(case, args.output)
            else:
                raise ValueError(f'Unsupported reference generator: {generator!r}')
            print(
                json.dumps({'status': 'generated', 'case_id': case.data['id'], 'result': str(path)})
            )
            return 0
        report = compare(args.case, args.reference, args.actual)
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.report:
            with args.report.open('x', encoding='utf-8', newline='\n') as stream:
                stream.write(text)
        print(text, end='')
        return {'pass': 0, 'fail': 1, 'not_executed': 2}[report['status']]
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'Numerical tool failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
