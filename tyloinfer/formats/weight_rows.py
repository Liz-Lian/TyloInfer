"""Bounded source-row reads for quantization and residual fitting."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from tyloinfer._gyro.reference import GGUF, TYPES, block_bytes, hadamard_unrotate


class GgufWeightRows:
    def __init__(self, path: str | Path, tensor: str):
        path = Path(path)
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name, re.IGNORECASE)
        paths = (
            [path]
            if match is None
            else [
                path.with_name(f"{match[1]}-{i:05d}-of-{int(match[3]):05d}.gguf")
                for i in range(1, int(match[3]) + 1)
            ]
        )
        self._owners = []
        selected = []
        for item in paths:
            reader = GGUF(item)
            self._owners.append(reader)
            if tensor in reader.tensors:
                selected.append(reader)
        if len(selected) != 1:
            raise ValueError(f"expected exactly one GGUF tensor named {tensor!r}")
        self._reader, self.name = selected[0], tensor
        dimensions, self.type, offset = self._reader.tensors[tensor]
        if len(dimensions) < 2 or min(dimensions) < 1:
            raise ValueError("residual fitting requires matrix or expert-bank rows")
        self.tensor_shape = tuple(reversed(dimensions))
        self.shape = math.prod(dimensions[1:]), dimensions[0]
        if self.type in TYPES:
            block, size = 128, block_bytes(self.type)
        else:
            from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

            block, size = GGML_QUANT_SIZES[GGMLQuantizationType(self.type)]
        if self.shape[1] % block:
            raise ValueError("GGUF source row contains an incomplete block")
        self.row_bytes = self.shape[1] // block * size
        self._offset = self._reader.data_offset + offset
        if self._offset + self.shape[0] * self.row_bytes > self._reader._mm.size:
            raise ValueError("truncated GGUF weight tensor")

    def read_rows(self, begin: int, end: int) -> np.ndarray:
        if not 0 <= begin < end <= self.shape[0]:
            raise IndexError("GGUF source row range is out of bounds")
        raw = self._reader._mm[
            self._offset + begin * self.row_bytes : self._offset + end * self.row_bytes
        ]
        if self.type in TYPES:
            from .gyro import decode_row

            value = np.stack(
                [
                    decode_row(row.tobytes(), self.type, self.shape[1])
                    for row in raw.reshape(end - begin, self.row_bytes)
                ]
            )
        else:
            from gguf.constants import GGMLQuantizationType
            from gguf.quants import dequantize

            value = dequantize(raw, GGMLQuantizationType(self.type)).reshape(
                end - begin, self.shape[1]
            )
        rotation = self._reader.hadamard
        if rotation is not None and rotation.is_rotated(self.name):
            if rotation.gdn_v_grouped and ".ssm_out." in self.name:
                raise ValueError(
                    "grouped GDN source requires its feature permutation before residual fitting"
                )
            value = hadamard_unrotate(value, rotation.block_size, rotation.signs_for(self.shape[1]))
        return np.ascontiguousarray(value, dtype=np.float32)


def open_weight_rows(path: str | Path, tensor: str | None = None):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.ndim < 2 or min(value.shape) < 1 or value.dtype.kind != "f":
            raise ValueError("expected a floating-point weight matrix")
        return value.reshape(-1, value.shape[-1])
    if not tensor:
        raise ValueError("a tensor name is required for a GGUF source")
    return GgufWeightRows(path, tensor)
