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




def _build_parser() -> argparse.ArgumentParser:
    from tyloinfer.commands.build import add_parser as add_build_parser
    from tyloinfer.commands.serve import add_parser as add_serve_parser

    parser = argparse.ArgumentParser(prog="tyloinfer", description="TQ inference and serving.")
    parser.add_argument("-V", "--version", action="version", version=f"tyloinfer {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    from tyloinfer.commands.fit_residual import add_parser as add_residual_parser
    add_residual_parser(sub)
    add_build_parser(sub)
    add_serve_parser(sub)
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args._impl(args))


if __name__ == "__main__":
    raise SystemExit(main())
