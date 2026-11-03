"""Fit and persist independent per-neuron residuals from existing checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path


def add_parser(sub):
    parser = sub.add_parser(
        "fit-residual", help="fit per-neuron GGUF TQK/TQP residuals without changing base weights"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--tensor", help="stored GGUF tensor name")
    parser.add_argument("--reference-tensor", help="reference tensor name, defaults to --tensor")
    parser.add_argument("--runtime-tensor", help="canonical runtime name for the residual manifest")
    parser.add_argument(
        "--format",
        default="TQK6",
        choices=("TQK4", "TQK5", "TQK6", "TQK7", "TQ2_T", "TQP5", "TQP6", "TQP7"),
    )
    parser.add_argument(
        "--row-formats", type=Path, help="JSON list of per-neuron formats; null disables a row"
    )
    parser.add_argument(
        "--importance", type=Path, help="NumPy column or per-neuron importance weights"
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, help="create or update a .row-residual.json model manifest"
    )
    parser.set_defaults(_impl=run)


def run(args):
    import numpy as np

    from tyloinfer.formats.weight_rows import open_weight_rows
    from tyloinfer.runtime.gguf_residual import fit_residual

    if args.output.resolve() in (args.base.resolve(), args.reference.resolve()):
        raise ValueError("residual output must not overwrite a source checkpoint")
    base = open_weight_rows(args.base, args.tensor)
    reference = open_weight_rows(args.reference, args.reference_tensor or args.tensor)
    choices = json.loads(args.row_formats.read_text()) if args.row_formats else args.format
    importance = np.load(args.importance, allow_pickle=False) if args.importance else None
    # Validate the manifest before starting any potentially expensive fit.
    document = None
    if args.manifest:
        if args.base.suffix.lower() != ".gguf" or not args.tensor:
            raise ValueError("runtime manifests require a named GGUF base tensor")
        if args.manifest.resolve() in (
            args.base.resolve(),
            args.reference.resolve(),
            args.output.resolve(),
        ):
            raise ValueError("manifest and weight paths must be distinct")
        document = {
            "format": "tyloinfer-row-residual-v1",
            "base": os.path.relpath(args.base.resolve(), args.manifest.parent.resolve()).replace(
                os.sep, "/"
            ),
            "tensors": {},
        }
        if args.manifest.exists():
            existing = json.loads(args.manifest.read_text())
            if (
                existing.get("format") != document["format"]
                or existing.get("base") != document["base"]
            ):
                raise ValueError("existing residual manifest refers to a different base")
            document = existing
        name = args.runtime_tensor or args.tensor
        document["tensors"][name] = os.path.relpath(
            args.output.resolve(), args.manifest.parent.resolve()
        ).replace(os.sep, "/")
    residual = fit_residual(reference, base, choices, importance=importance, workers=args.workers)
    residual.save(args.output)
    if document is not None:
        from tyloinfer.formats.gguf_residual import atomic_write

        atomic_write(args.manifest, (json.dumps(document, indent=2) + "\n").encode())
    print(
        json.dumps(
            {
                "rows": len(residual.rows),
                "corrected_rows": sum(name is not None for name in residual.formats),
                "bytes": residual.nbytes,
                "output": str(args.output),
            }
        )
    )
    return 0
