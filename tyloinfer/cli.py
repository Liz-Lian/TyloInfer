"""TyloInfer command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from tyloinfer._version import __version__


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"command {args.command!r} is not implemented", file=sys.stderr)
    return 2


def _voice_runtime_check(_args: argparse.Namespace) -> int:
    import onnxruntime  # noqa: F401
    import s3tokenizer  # noqa: F401
    import scipy  # noqa: F401
    import soundfile  # noqa: F401
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
    from hyperpyyaml import load_hyperpyyaml  # noqa: F401
    from stepaudio2.flashcosyvoice.modules.hifigan import HiFTGenerator  # noqa: F401
    from stepaudio2.token2wav import _setup_cosyvoice2_alias  # noqa: F401

    print(json.dumps({"voice_output_runtime": "ready"}))
    return 0


def _mlx_runtime_check(_args: argparse.Namespace) -> int:
    """Touch the bundled MLX Metal runtime without loading a model."""
    import mlx.core as mx

    value = mx.array([1.0], dtype=mx.float32)
    mx.eval(value)
    print(json.dumps({"mlx_metal_runtime": "ready"}))
    return 0


def _hub_download(args: argparse.Namespace) -> int:
    """Download a hub snapshot from the unified CLI used by packaged Studio builds."""
    if args.provider == "huggingface":
        from huggingface_hub import snapshot_download

        output = snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=args.include or None,
            ignore_patterns=args.exclude or None,
            max_workers=args.max_workers,
            token=os.environ.get("HF_TOKEN") or None,
        )
    else:
        from modelscope_hub import HubApi

        api = HubApi(
            token=os.environ.get("MODELSCOPE_API_TOKEN")
            or os.environ.get("MODELSCOPE_TOKEN")
            or None
        )
        output = api.download_repo(
            args.repo_id,
            args.repo_type,
            revision=args.revision,
            local_dir=args.local_dir,
            allow_patterns=args.include or None,
            ignore_patterns=args.exclude or None,
            max_workers=args.max_workers,
        )
    print(json.dumps({"event": "hub_download_complete", "path": str(output)}))
    return 0


def _wt2_reference(args: argparse.Namespace) -> int:
    from tyloinfer.evaluation.reference import generate_reference
    backend = ('metal' if sys.platform == 'darwin' else 'cuda') if args.backend == 'auto' else args.backend
    device = args.device or ('mps' if backend == 'metal' else 'cuda:0')
    if device.split(':', 1)[0] != ('mps' if backend == 'metal' else 'cuda'):
        raise ValueError('WT2 device must match the selected backend')
    generate_reference(args.model, args.dataset, args.output, context_size=args.context_size,
        chunks=args.chunks, device=device, layerwise=not args.resident, runtime=args.runtime or None)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    from tyloinfer.commands.build import add_parser as add_build_parser
    from tyloinfer.commands.serve import add_parser as add_serve_parser

    parser = argparse.ArgumentParser(prog="tyloinfer", description="TQ inference and serving.")
    parser.add_argument("-V", "--version", action="version", version=f"tyloinfer {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    add_build_parser(sub)
    add_serve_parser(sub)
    reference = sub.add_parser('wt2-reference', help='generate official WT2 reference logits from original HF weights')
    reference.add_argument('--model', required=True)
    reference.add_argument('--dataset', required=True)
    reference.add_argument('--output', required=True)
    reference.add_argument('--context-size', type=int, default=512)
    reference.add_argument('--chunks', type=int, default=8)
    reference.add_argument('--backend', choices=('auto', 'metal', 'cuda'), default='auto')
    reference.add_argument('--device', default='')
    reference.add_argument('--resident', action='store_true')
    reference.add_argument('--runtime', default='', help='native CUDA runtime executable')
    reference.set_defaults(_impl=_wt2_reference)
    sub.add_parser(
        "voice-runtime-check",
        help="verify optional MiniCPM-o voice output dependencies",
    ).set_defaults(_impl=_voice_runtime_check)
    sub.add_parser("_mlx-runtime-check", help=argparse.SUPPRESS).set_defaults(
        _impl=_mlx_runtime_check
    )
    hub_download = sub.add_parser("_hub-download", help=argparse.SUPPRESS)
    hub_download.add_argument(
        "--provider", required=True, choices=("huggingface", "modelscope")
    )
    hub_download.add_argument("--repo-id", required=True)
    hub_download.add_argument("--repo-type", default="model")
    hub_download.add_argument("--revision", required=True)
    hub_download.add_argument("--local-dir", type=Path, required=True)
    hub_download.add_argument("--max-workers", type=int, default=8)
    hub_download.add_argument("--include", action="append", default=[])
    hub_download.add_argument("--exclude", action="append", default=[])
    hub_download.set_defaults(_impl=_hub_download)
    sub.add_parser("inspect", help="inspect a TQ checkpoint").set_defaults(_impl=_not_implemented)
    return parser


def main(argv: list[str] | None = None) -> int:
    if bool(getattr(sys, "frozen", False)):
        import multiprocessing

        multiprocessing.freeze_support()
    arguments = sys.argv[1:] if argv is None else argv
    if arguments == ["_official-scoring-probe"]:
        print(4)
        return 0
    if arguments == ["_official-scoring-worker"]:
        from tyloinfer.server.services.official_scoring_worker import main as score_main

        score_main()
        return 0
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args._impl(args))


if __name__ == "__main__":
    raise SystemExit(main())
