"""TQ CUDA kernel sources and native operator bindings."""

from __future__ import annotations

from pathlib import Path


def tq_source_directory() -> Path:
    package = Path(__file__).resolve().parent
    source = package.parents[2] / "csrc/engine/backends/cuda/kernels"
    return source if (source / "tq_ops.cu").is_file() else package
