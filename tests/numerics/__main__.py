"""Run offline reference generation or compare already persisted results."""

import argparse
import json
import sys
from pathlib import Path

from .compare import compare


def main(argv=None):
    """Run the requested offline tool; comparison never imports Torch."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    generate = commands.add_parser('generate', help='Create new fixed inputs and reference results')
    generate.add_argument('kind', choices=('rms', 'basic', 'tq'))
    generate.add_argument('output', type=Path)
    generate.add_argument(
        '--suite',
        choices=(
            'gemv',
            'row_scale',
            'float_prefill',
            'q8',
            'q8_parallel',
            'ffn',
            'embedding',
            'routed',
        ),
        help='TQ suite, required when kind=tq',
    )
    generate.add_argument('--device', default='cuda')
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
            if (args.kind == 'tq') != (args.suite is not None):
                parser.error('--suite is required only for TQ generation')
            if args.kind == 'tq':
                from .tq_ref import generate

                paths = generate(args.output, args.suite, args.device)
            else:
                from .torch_ref import generate

                paths = generate(args.output, args.kind, args.device)
            print(json.dumps({'status': 'generated', 'cases': [str(path) for path in paths]}))
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
