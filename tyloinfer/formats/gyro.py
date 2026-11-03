"""GGUF Gyro row encodings; the normative codebook is from AgentionAI.

Copyright 2026 AgentionAI (Laurent Zuijdwijk), Apache-2.0.
See ``tyloinfer/_gyro/NOTICE`` for attribution and the upstream reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

FORMATS = {
    "TQ2_T": (144, 8, 32),
    "TQK6": (145, 6, 32),
    "TQK7": (146, 7, 32),
    "TQP5": (147, 5, 64),
    "TQP6": (148, 6, 64),
    "TQP7": (149, 7, 64),
    "TQK5": (150, 5, 32),
    "TQK4": (151, 4, 32),
}
FORMAT_NAMES = {value[0]: name for name, value in FORMATS.items()}


def format_name(value: str | int) -> str:
    if isinstance(value, (int, np.integer)):
        value = FORMAT_NAMES.get(int(value), "")
    if not isinstance(value, str) or value not in FORMATS:
        raise ValueError(f"unsupported Gyro format: {value!r}")
    return str(value)


def row_bytes(format: str | int, columns: int) -> int:
    _, bits, steps = FORMATS[format_name(format)]
    if not isinstance(columns, (int, np.integer)) or isinstance(columns, bool) or columns <= 0:
        raise ValueError("row width must be positive")
    return ((columns + 127) // 128) * (2 + steps * bits // 8)


@lru_cache(maxsize=1)
def codebook() -> np.ndarray:
    value = np.frombuffer(files("tyloinfer._gyro").joinpath("codebook_f16.bin").read_bytes(), "<f2")
    if value.size != 4096:
        raise ValueError("invalid Gyro codebook")
    return value.reshape(2048, 2)


def decode_row(packed: bytes, format: str | int, columns: int) -> np.ndarray:
    name = format_name(format)
    _, bits, steps = FORMATS[name]
    if len(packed) != row_bytes(name, columns):
        raise ValueError("Gyro row byte count differs")
    stream_bytes = steps * bits // 8
    blocks = np.frombuffer(packed, np.uint8).reshape(-1, stream_bytes + 2)
    scales = blocks[:, :2].copy().view("<f2").astype(np.float32)
    stream = blocks[:, 2:]
    if bits == 8:
        states = (np.roll(stream.astype(np.uint32), 1, axis=1) << 8) | stream
    else:
        position = (steps - 1 - np.arange(steps)) * bits
        byte = position // 8
        word = sum(
            stream[:, (byte + i) % stream_bytes].astype(np.uint32) << (8 * i) for i in range(3)
        )
        states = (word >> (position % 8)) & 65535
    hashed = (states.astype(np.uint32) & np.uint32(32767)) * np.uint32(0x9E3779B1)
    indices = [hashed >> 21]
    if steps == 32:
        indices.append((hashed >> 10) & 2047)
    values = np.concatenate([codebook()[index].astype(np.float32) for index in indices], axis=-1)
    return (values.reshape(len(blocks), 128) * scales).reshape(-1)[:columns].copy()


@dataclass(frozen=True)
class GyroTensor:
    """One GGUF-compatible format per tensor; each output neuron owns its row."""

    format: str
    columns: int
    rows: tuple[bytes, ...]

    def __post_init__(self):
        object.__setattr__(self, "format", format_name(self.format))
        object.__setattr__(self, "rows", tuple(bytes(row) for row in self.rows))
        if not self.rows or any(
            len(row) != row_bytes(self.format, self.columns) for row in self.rows
        ):
            raise ValueError("invalid Gyro tensor rows")

    @property
    def shape(self):
        return len(self.rows), self.columns

    def read_rows(self, begin: int, end: int) -> np.ndarray:
        if not 0 <= begin < end <= len(self.rows):
            raise IndexError("Gyro row range is out of bounds")
        return np.stack(
            [decode_row(row, self.format, self.columns) for row in self.rows[begin:end]]
        )

    def dequantize(self) -> np.ndarray:
        return self.read_rows(0, len(self.rows))

    def gguf_bytes(self) -> bytes:
        if self.columns % 128:
            raise ValueError("standard GGUF Gyro tensors require complete 128-value blocks")
        return b"".join(self.rows)
