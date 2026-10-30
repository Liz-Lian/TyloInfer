from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TensorInspection:
    shape: tuple[int, ...]
    family: str
    row_bytes: np.ndarray | None = None
    granularity: str = 'tensor'


def _shape(blob: memoryview, offset: int, rank: int) -> tuple[int, ...]:
    if not 1 <= rank <= 16:
        raise ValueError('invalid tensor rank')
    shape = struct.unpack_from(f'<{rank}q', blob, offset)
    if any(value <= 0 for value in shape):
        raise ValueError('invalid tensor dimensions')
    return shape


def _tq(blob: memoryview, rows: bool, logical_shape: tuple[int, ...] | None) -> TensorInspection:
    if len(blob) < 64:
        raise ValueError('truncated TQ header')
    magic, version, outputs, width, columns, gs, packed = struct.unpack_from('<8s6I', blob)
    if magic == b'RTQEXP01':
        count, outputs, width, stride = outputs, width, columns, gs
        if (version != 1 or not count or not outputs or not width or stride != 16 or packed
                or count > (len(blob) - 64) // 16 or any(blob[32:64])):
            raise ValueError('invalid TQ expert container')
        shape = (count, outputs, width)
        if logical_shape and logical_shape != shape:
            raise ValueError('TQ expert shape differs from metadata')
        end = 64 + count * 16
        storage = []
        for index in range(count):
            offset, size = struct.unpack_from('<QQ', blob, 64 + index * 16)
            if offset != end or size < 64 or offset > len(blob) or size > len(blob) - offset:
                raise ValueError('invalid TQ expert span')
            child = blob[offset:offset + size]
            try:
                if bytes(child[:8]) not in (b'RTRELL05', b'RTRELL06'):
                    raise ValueError('nested TQ expert containers are not supported')
                item = _tq(child, rows, (outputs, width))
                if rows:
                    storage.append(item.row_bytes + (16 + 64 / count) / outputs)
            finally:
                child.release()
            end = offset + size
        if end != len(blob):
            raise ValueError('TQ expert payload length differs')
        return TensorInspection(shape, 'TQ', np.concatenate(storage) if rows else None, 'row')
    compact = magic == b'RTRELL06' and version == 6
    if (not (compact or (magic == b'RTRELL05' and version == 5))
            or not outputs or not width or width % 32
            or (compact and (not columns or width % 256 == 0))
            or (not compact and width % 256)
            or columns not in (0, 1) or any(blob[32:64]) or packed % 4
            or (not columns and (gs < 16 or gs > 2**20 or gs & (gs - 1)))):
        raise ValueError('invalid TQ row geometry')
    groups = 0 if columns else math.ceil(width / gs)
    scales = width * 2 if columns else (outputs * groups + 1) // 2
    if 64 + outputs * 8 + scales + packed != len(blob):
        raise ValueError('TQ payload length differs')
    shape = logical_shape or (outputs, width)
    if len(shape) not in (2, 3) or math.prod(shape[:-1]) != outputs or not 0 < shape[-1] <= width:
        raise ValueError('TQ logical shape differs')
    storage = None
    if rows:
        descriptors = np.frombuffer(blob, '<u8', outputs, 64)
        bits = ((descriptors >> 32) & 7) + 1
        words = width * bits // 32
        offsets = descriptors & np.uint64(0x7ffffffe if compact else 0xffffffff)
        seed = (((descriptors >> 35) & 31) | (((descriptors >> 56) & 255) << 5)
                | ((descriptors & 1) << 13) | (((descriptors >> 31) & 1) << 14))
        if ((not compact and np.any(descriptors & np.uint64(0xff0000f800000000)))
                or (compact and (np.any(words & 1) or int(words.sum()) >= 2**31
                                 or np.any(seed >= (1 << (16 - bits)))))
                or np.any(((descriptors >> 40) & 65535) > 0x7bff)
                or not np.array_equal(offsets, np.r_[np.uint64(0), np.cumsum(words[:-1])])
                or int(words.sum()) * 4 != packed):
            raise ValueError('invalid TQ row descriptors')
        storage = words.astype(np.float64) * 4 + 8 + (64 + scales) / outputs
    return TensorInspection(shape, 'TQ', storage, 'row')


def inspect_tensor(dtype: str, blob: memoryview, *, rows: bool = False,
                   logical_shape: tuple[int, ...] | None = None) -> TensorInspection:
    if dtype == 'TQ':
        return _tq(blob, rows, logical_shape)
    if dtype in {'MXFP4', 'MXFP8'}:
        magic, version, kind, reserved, outputs, width = struct.unpack_from('<4sBBHQQ', blob)
        if magic != b'MXT1' or version != 1 or reserved or min(outputs, width) <= 0:
            raise ValueError('invalid MX geometry')
        shape = logical_shape or (outputs, width)
        if math.prod(shape) != outputs * width:
            raise ValueError('MX logical shape differs')
        return TensorInspection(shape, dtype)
    if dtype in {'BF16', 'F16', 'F32', 'F8_E4M3', 'I32', 'I64'}:
        rank = struct.unpack_from('<I', blob)[0]
        shape = _shape(blob, 4, rank) if rank else ()
        size = {'BF16': 2, 'F16': 2, 'F32': 4, 'F8_E4M3': 1, 'I32': 4, 'I64': 8}[dtype]
        if 4 + 8 * rank + math.prod(shape) * size != len(blob):
            raise ValueError('dense shape does not match payload')
        return TensorInspection(shape, dtype)
    from tyloinfer.formats.compat import LEGACY_DTYPES, canonical_dtype
    if canonical_dtype(dtype) in LEGACY_DTYPES:
        from tyloinfer.formats.mfq.metadata import inspect_tensor as inspect_mfq
        result = inspect_mfq(dtype, blob, rows=rows)
        if logical_shape and logical_shape != result.shape:
            raise ValueError('MFQ logical shape differs')
        return result
    raise ValueError(f'unsupported TyloInfer tensor dtype: {dtype}')
