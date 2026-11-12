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
        choices=(
            'rms',
            'basic',
            'rope',
            'conv',
            'gelu',
            'tq',
            'flash',
            'mx',
            'gdn',
            'nemotron',
            'attn',
            'sample',
            'mfq',
            'zero',
            'nepq',
            'sq4',
            'sq8',
        ),
    )
    generate.add_argument('output', type=Path)
    generate.add_argument(
        '--suite',
        help='Required for grouped reference generators',
    )
    generate.add_argument('--device', default='cuda')
    generate.add_argument(
        '--source', type=Path, help='Explicit MFQ checkout used only for reference generation'
    )
    generate.add_argument('--experts', type=int, default=1, help='MFQ fixture expert count')
    generate.add_argument(
        '--capture', type=Path, help='Native MTP diagnostic record for nemotron/mtp-target'
    )
    rerun = commands.add_parser(
        'reference', help='Recompute a reference from existing fixed inputs'
    )
    rerun.add_argument('case', type=Path)
    rerun.add_argument(
        'output', type=Path, help='A new result directory; existing files are never replaced'
    )
    rerun.add_argument('--device', default='cuda', help='CUDA device for Torch references')
    rerun.add_argument('--source', type=Path, help='The original MFQ source checkout for MFQ cases')
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
            grouped = args.kind in {
                'tq',
                'flash',
                'mx',
                'gdn',
                'nemotron',
                'attn',
                'sample',
                'zero',
                'nepq',
                'sq4',
                'sq8',
            }
            if grouped != (args.suite is not None):
                parser.error('--suite is required only for grouped reference generators')
            if (args.kind in {'mfq', 'zero', 'nepq', 'sq4', 'sq8'}) != (args.source is not None):
                parser.error('--source is required only for MFQ-based reference generation')
            if args.kind != 'mfq' and args.experts != 1:
                parser.error('--experts is only available for MFQ generation')
            if (args.kind == 'nemotron' and args.suite == 'mtp-target') != (
                args.capture is not None
            ):
                parser.error('--capture is required only for nemotron/mtp-target')
            if args.kind == 'mfq':
                from .mfq_ref import generate

                paths = generate(args.output, args.source, args.experts)
            elif args.kind == 'zero':
                from .zero_ref import generate

                paths = generate(args.output, args.source, args.suite, args.device)
            elif args.kind == 'nepq':
                from .nepq_ref import generate

                paths = generate(args.output, args.source, args.suite, args.device)
            elif args.kind == 'sq4':
                from .sq4_ref import generate

                paths = generate(args.output, args.source, args.suite, args.device)
            elif args.kind == 'sq8':
                from .sq8_ref import generate

                paths = generate(args.output, args.source, args.suite, args.device)
            elif args.kind == 'tq':
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
                if args.suite == 'mtp-target':
                    from .nemotron_text_ref import generate_target

                    paths = [generate_target(args.output, args.capture)]
                elif args.suite in {'backbone', 'mtp'}:
                    from .nemotron_text_ref import generate

                    paths = generate(args.output, args.suite)
                elif args.suite in {'vision', 'prompt'}:
                    from .nemotron_vision_ref import generate

                    paths = generate(args.output, args.suite)
                else:
                    parser.error(
                        'Nemotron suite must be backbone, mtp, mtp-target, vision or prompt'
                    )
            elif args.kind == 'attn':
                from .attn_ref import generate

                paths = generate(args.output, args.suite, args.device)
            elif args.kind == 'sample':
                from .sample_ref import generate

                paths = generate(args.output, args.suite, args.device)
            else:
                from .torch_ref import generate

                paths = generate(args.output, args.kind, args.device)
            print(json.dumps({'status': 'generated', 'cases': [str(path) for path in paths]}))
            return 0
        if args.command == 'reference':
            case = Case.load(args.case)
            generator = case.data['source'].get('generator')
            external = generator in {
                f'tests/numerics/{name}_ref.py' for name in ('mfq', 'zero', 'nepq', 'sq4', 'sq8')
            }
            if external != (args.source is not None):
                parser.error('--source is required only for an MFQ-based reference case')
            if generator == 'tests/numerics/mfq_ref.py':
                from .mfq_ref import reference

                path = reference(case, args.output, args.source)
            elif generator == 'tests/numerics/zero_ref.py':
                from .zero_ref import reference

                path = reference(case, args.output, args.source, args.device)
            elif generator == 'tests/numerics/nepq_ref.py':
                from .nepq_ref import reference

                path = reference(case, args.output, args.source, args.device)
            elif generator == 'tests/numerics/sq4_ref.py':
                from .sq4_ref import reference

                path = reference(case, args.output, args.source, args.device)
            elif generator == 'tests/numerics/sq8_ref.py':
                from .sq8_ref import reference

                path = reference(case, args.output, args.source, args.device)
            elif generator == 'tests/numerics/torch_ref.py':
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
            elif generator == 'tests/numerics/attn_ref.py':
                from .attn_ref import reference

                path = reference(case, args.output, args.device)
            elif generator == 'tests/numerics/sample_ref.py':
                from .sample_ref import reference

                path = reference(case, args.output, args.device)
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
    except (
        ValueError,
        OSError,
        ImportError,
        RuntimeError,
        KeyError,
        TypeError,
        IndexError,
    ) as error:
        print(f'Numerical tool failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
