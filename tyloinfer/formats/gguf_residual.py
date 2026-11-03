"""Independent, heterogeneous output-neuron residuals for immutable weights."""

from __future__ import annotations

import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .gyro import FORMAT_NAMES, FORMATS, decode_row, format_name, row_bytes

_HEADER = struct.Struct("<8sIIQQQQ")
_ROW = struct.Struct("<QII")
MAGIC = b"TQRES001"


def atomic_write(path: str | Path, value: bytes):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class RowResidual:
    """Missing rows have no payload; present rows retain standard Gyro blocks."""

    columns: int
    formats: tuple[str | None, ...]
    rows: tuple[bytes, ...]

    def __post_init__(self):
        if (
            not isinstance(self.columns, (int, np.integer))
            or isinstance(self.columns, bool)
            or self.columns <= 0
            or not self.rows
            or len(self.formats) != len(self.rows)
        ):
            raise ValueError("invalid residual geometry")
        formats = tuple(None if item is None else format_name(item) for item in self.formats)
        rows = tuple(bytes(item) for item in self.rows)
        for name, row in zip(formats, rows):
            if len(row) != (row_bytes(name, self.columns) if name else 0):
                raise ValueError("residual row byte count differs")
        object.__setattr__(self, "formats", formats)
        object.__setattr__(self, "rows", rows)

    @classmethod
    def empty(cls, rows: int, columns: int):
        return cls(columns, (None,) * rows, (b"",) * rows)

    @property
    def shape(self):
        return len(self.rows), self.columns

    @property
    def nbytes(self):
        return _HEADER.size + len(self.rows) * _ROW.size + sum(map(len, self.rows))

    def read_rows(self, begin: int, end: int) -> np.ndarray:
        if not 0 <= begin < end <= len(self.rows):
            raise IndexError("residual row range is out of bounds")
        output = np.zeros((end - begin, self.columns), dtype=np.float32)
        for i in range(begin, end):
            if self.formats[i]:
                output[i - begin] = decode_row(self.rows[i], self.formats[i], self.columns)
        return output

    def dequantize(self) -> np.ndarray:
        return self.read_rows(0, len(self.rows))

    def project(self, input: np.ndarray) -> np.ndarray:
        input = np.asarray(input, dtype=np.float32)
        if input.ndim < 1 or input.shape[-1] != self.columns:
            raise ValueError("residual projection input width differs")
        output = np.zeros((*input.shape[:-1], len(self.rows)), dtype=np.float32)
        for row, name in enumerate(self.formats):
            if name:
                output[..., row] = input @ decode_row(self.rows[row], name, self.columns)
        return output

    def replace(self, row_ids, replacement: RowResidual):
        ids = np.asarray(row_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(map(int, ids))) != len(ids):
            raise ValueError("row IDs must be unique integers")
        if len(ids) != len(replacement.rows) or self.columns != replacement.columns:
            raise ValueError("replacement geometry differs")
        if np.any(ids < 0) or np.any(ids >= len(self.rows)):
            raise IndexError("replacement row is out of bounds")
        formats, rows = list(self.formats), list(self.rows)
        for i, row in enumerate(ids):
            formats[row], rows[row] = replacement.formats[i], replacement.rows[i]
        return RowResidual(self.columns, tuple(formats), tuple(rows))

    def to_bytes(self) -> bytes:
        descriptors, payload, offset = [], [], 0
        for name, row in zip(self.formats, self.rows):
            descriptors.append(_ROW.pack(offset, FORMATS[name][0] if name else 0, len(row)))
            payload.append(row)
            offset += len(row)
        return (
            _HEADER.pack(
                MAGIC, 1, 0, len(self.rows), self.columns, (self.columns + 127) // 128 * 128, offset
            )
            + b"".join(descriptors)
            + b"".join(payload)
        )

    @classmethod
    def from_bytes(cls, value: bytes):
        if len(value) < _HEADER.size:
            raise ValueError("truncated residual header")
        magic, version, flags, count, columns, padded, size = _HEADER.unpack_from(value)
        if (
            magic != MAGIC
            or version != 1
            or flags
            or count < 1
            or columns < 1
            or padded != (columns + 127) // 128 * 128
            or count > (len(value) - _HEADER.size) // _ROW.size
        ):
            raise ValueError("invalid residual header")
        start = _HEADER.size + count * _ROW.size
        if size != len(value) - start:
            raise ValueError("residual payload length differs")
        rows, formats, cursor = [], [], 0
        for index in range(count):
            offset, kind, length = _ROW.unpack_from(value, _HEADER.size + index * _ROW.size)
            name = FORMAT_NAMES.get(kind)
            if (
                offset != cursor
                or (kind and name is None)
                or length != (row_bytes(name, columns) if name else 0)
                or length > size - cursor
            ):
                raise ValueError("invalid residual row descriptor")
            formats.append(name)
            rows.append(value[start + cursor : start + cursor + length])
            cursor += length
        if cursor != size:
            raise ValueError("unreferenced residual payload")
        return cls(columns, tuple(formats), tuple(rows))

    def save(self, path: str | Path):
        atomic_write(path, self.to_bytes())

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_bytes(Path(path).read_bytes())
