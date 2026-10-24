"""MFQ file headers and tensor-level metadata.

An MFQ file consists of one global header and several tensor records. The header stores the magic number,
version, and model-structure summary. Each tensor record stores its precision specification (NINT variant,
group size, and so on) and the location of its compressed weight blob.

See :mod:`tyloinfer.formats.io` for the binary layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TYLOINFER_MAGIC = b"MFQ1"
# TQ checkpoints use the same MFQ1 container and shard metadata.
TYLOINFER_SUFFIXES = (".tq", ".mfq")
TENSOR_DTYPES = frozenset({'TQ', 'BF16', 'F16', 'F32', 'F8_E4M3', 'I32', 'I64', 'MXFP4', 'MXFP8', 'BLOB'})


@dataclass
class FileHeader:
    """Global MFQ file header."""

    magic: bytes = TYLOINFER_MAGIC
    version: int = 1
    model_arch: str = ""
    num_tensors: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TensorRecord:
    """Location and precision metadata for one tensor in the file."""

    name: str
    dtype: str          # Canonical public dtype, e.g. "NINT", "MFE", "BF16"
    shape: tuple[int, ...]
    offset: int         # Byte offset of the weight blob in the file
    nbytes: int         # Size of the weight blob in bytes
    groupsize: int = 0
    hadamard: bool = False
