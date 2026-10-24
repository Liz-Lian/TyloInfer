from __future__ import annotations

import math
import struct

import numpy as np

from tyloinfer.formats.compat import canonical_dtype
from tyloinfer.formats.io import unpack_bits


from tyloinfer.formats.inspection import TensorInspection


def _shape(blob: memoryview, offset: int, rank: int) -> tuple[int, ...]:
    if not 1 <= rank <= 16:
        raise ValueError("invalid tensor rank")
    shape = struct.unpack_from(f"<{rank}q", blob, offset)
    if any(value <= 0 for value in shape):
        raise ValueError("invalid tensor dimensions")
    return shape


def _selectors(blob: memoryview, offset: int, count: int, bits: int) -> np.ndarray:
    if count > 16_777_216 or offset + (count * bits + 7) // 8 > len(blob):
        raise ValueError("invalid or oversized row selector table")
    return unpack_bits(blob, offset, count, bits)[0].astype(np.int64)


def _row_storage(cost: np.ndarray, size: int) -> np.ndarray:
    remainder = size - float(cost.sum())
    if remainder < -1e-6:
        raise ValueError("row storage exceeds tensor payload")
    return cost + max(0.0, remainder) / len(cost)


def inspect_tensor(dtype: str, blob: memoryview, *, rows: bool = False) -> TensorInspection:
    dtype = canonical_dtype(dtype)
    if dtype in {"MFE", "MFED"}:
        magic, experts, outputs, width, pools = struct.unpack_from('<4sIIII', blob)
        if magic not in {b'MFE1', b'NIM1', b'NIM2', b'MFD1', b'NID2'} or min(experts, outputs, width, pools) <= 0 or pools > experts:
            raise ValueError('invalid MFE header')
        cost = np.full((experts, outputs), np.nan) if rows else None
        owners = set()
        offset = 20
        for _ in range(pools):
            if magic == b'NIM1':
                count, payload = struct.unpack_from('<IQ', blob, offset)
                offset += 12
                label, label_bytes, runtime = 'NINT', 0, 0
            else:
                count, label_bytes, payload, runtime = struct.unpack_from('<IIQQ', blob, offset)
                offset += 24
                if not 1 <= label_bytes <= 32:
                    raise ValueError('invalid MFE cohort dtype')
            if not 0 < count <= experts or offset + count * 4 + label_bytes + runtime + payload > len(blob):
                raise ValueError('invalid MFE cohort span')
            ids = struct.unpack_from(f'<{count}i', blob, offset)
            offset += count * 4
            if any(i < 0 or i >= experts or i in owners for i in ids) or len(set(ids)) != count:
                raise ValueError('invalid MFE expert IDs')
            owners.update(ids)
            if label_bytes:
                label = bytes(blob[offset:offset + label_bytes]).decode('ascii')
            offset += label_bytes + runtime
            if canonical_dtype(label) in {'MFE', 'MFED'}:
                raise ValueError('nested MFE containers are not supported')
            if rows:
                child = blob[offset:offset + payload]
                try:
                    from tyloinfer.formats.inspection import inspect_tensor as inspect_child
                    item = inspect_child(label, child, rows=True)
                    if math.prod(item.shape) != count * outputs * width:
                        raise ValueError('MFE cohort shape differs')
                    storage = item.row_bytes if item.row_bytes is not None else np.full(count * outputs, payload / (count * outputs))
                    cost[list(ids)] = storage.reshape(count, outputs)
                finally:
                    child.release()
            offset += payload
        if offset != len(blob) or len(owners) != experts:
            raise ValueError('incomplete MFE tensor')
        return TensorInspection((experts, outputs, width), dtype,
            _row_storage(cost.reshape(-1), len(blob)) if rows else None, 'expert')
    if dtype == "NINT":
        raw_q, k, gs, axis, width = struct.unpack_from("<BBiii", blob)
        rank = struct.unpack_from("<I", blob, 14)[0]
        shape = _shape(blob, 18, rank)
        offset = 18 + rank * 8
        outputs, groups = struct.unpack_from("<II", blob, offset)
        if not 0 <= axis < rank or shape[axis] != outputs or gs <= 0 or groups != math.ceil(width / gs):
            raise ValueError("invalid NINT geometry")
        q = raw_q & 127
        if not 1 <= q <= 8 or not 1 <= k <= 8 or math.prod(shape) != outputs * width:
            raise ValueError("invalid NINT precision or shape")
        row_bytes = None
        if rows:
            if raw_q & 128:
                offset += 8 + outputs * 4
                ks = _selectors(blob, offset, outputs, 2) + k - 1
                if np.any(ks < 1) or np.any(ks > 8):
                    raise ValueError("invalid NINT subgroup selector")
                offset += (outputs * 2 + 7) // 8
                offset += sum(2 * ((int(np.count_nonzero(ks == value)) * groups * value + 7) // 8)
                              for value in range(1, 9))
                qs = _selectors(blob, offset, outputs, 3) + 1
                row_bytes = _row_storage((32 + 5 + 2 * groups * ks + groups * gs * qs) / 8, len(blob))
            else:
                row_bytes = np.full(outputs, len(blob) / outputs)
        return TensorInspection(shape, "NINTv2" if raw_q & 128 else f"NINT{q}", row_bytes,
                                "row" if raw_q & 128 and rows else "cohort")
    if dtype in {"NVQ", "NPQ"}:
        magic, profile, _sub_bits, _gs, _axis, _width, rank = struct.unpack_from("<4sBBHiiI", blob)
        shape = _shape(blob, 20, rank)
        families = {b"NQ1L": "NVQ1-L", b"NQ1S": "NVQ1-S", b"NPQL": "NPQ0-L", b"NPQS": "NPQ0-S"}
        family = families.get(magic)
        if magic in {b"NVQ1", b"NIQ1"}:
            codebook = profile & 31
            if profile & 32:
                family = {1: "NVQ2J", 2: "NVQ3J", 3: "NVQ3J-512", 4: "NVQ2J-L", 5: "NVQ2J-XL", 6: "NVQ3J-L"}.get(codebook)
            else:
                family = {1: "NVQ2", 2: "NVQ3"}.get(codebook)
        if family is None:
            raise ValueError("unknown NVQ/NPQ profile")
        return TensorInspection(shape, family, granularity="cohort")
    if dtype == "NINT8-0":
        magic, _axis, _width, rank = struct.unpack_from("<4siiI", blob)
        if magic != b"NI80":
            raise ValueError("invalid NINT8-0 magic")
        return TensorInspection(_shape(blob, 16, rank), dtype, granularity="cohort")
    if dtype == "NEPQ":
        magic, version, profile, _groups, _flags, experts, outputs, width, _banks, _rotation, _seed = struct.unpack_from("<4sBBBBIIIIIQ", blob)
        if magic != b"NEP1" or version != 1:
            raise ValueError("unsupported NEPQ header")
        shape = (experts, outputs, width) if experts > 1 else (outputs, width)
        if any(value <= 0 for value in shape):
            raise ValueError("invalid NEPQ geometry")
        labels = {0: "NEPQ0-S", 1: "NEPQ0-L", 2: "NEPQ1-S", 3: "NEPQ1-L", 4: "NEPQ0-A", 5: "NEPQ1-A"}
        if profile not in labels:
            raise ValueError("unknown NEPQ profile")
        return TensorInspection(shape, labels[profile], granularity="cohort")
    if dtype in {"MXFP4", "MXFP8"}:
        magic, _version, _kind, _reserved, outputs, width = struct.unpack_from("<4sBBHQQ", blob)
        if magic != b"MXT1" or min(outputs, width) <= 0:
            raise ValueError("invalid MX geometry")
        return TensorInspection((outputs, width), dtype, granularity="cohort")
    if dtype == "MXFP4-SQ":
        magic, _version, _base, _reserved, outputs, width = struct.unpack_from("<4sBBHQQ", blob)
        if min(outputs, width) <= 0 or magic not in {b"SQV2", b"SQ2\0", b"SQ3\0", b"SQ31"}:
            raise ValueError("unsupported MXFP4-SQ header")
        row_bytes = None
        if rows and magic == b"SQV2":
            qs = _selectors(blob, 24, outputs, 2) + 1
            cost = (width * qs + 2) / 8 + np.where(qs == 4, width / 32, width / 256 + 7)
            row_bytes = _row_storage(cost, len(blob))
        return TensorInspection((outputs, width), dtype, row_bytes, "row" if row_bytes is not None else "cohort")
    if dtype in {"MXFP8-SQ", "FP8-128SQ"}:
        magic, _version, _scale, _flags, _reserved, _br, _bc, outputs, width, _sr, _sc = struct.unpack_from("<4sBBBBHHQQQQ", blob)
        if magic not in {b"M8SQ", b"F8SQ"} or min(outputs, width) <= 0:
            raise ValueError("invalid FP8-SQ geometry")
        row_bytes = None
        if rows:
            qs = _selectors(blob, 44, outputs, 3) + 1
            row_bytes = _row_storage(np.ceil(width * qs / 8) + 3 / 8, len(blob))
        return TensorInspection((outputs, width), dtype, row_bytes, "row" if rows else "cohort")
    if dtype in {"BF16", "F16", "F32", "F8_E4M3", "I32", "I64"}:
        rank = struct.unpack_from("<I", blob)[0]
        shape = _shape(blob, 4, rank) if rank else ()
        size = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I32": 4, "I64": 8}[dtype]
        if 4 + 8 * rank + math.prod(shape) * size != len(blob):
            raise ValueError("dense shape does not match payload")
        return TensorInspection(shape, dtype, granularity="cohort")
    raise ValueError(f"unsupported tensor dtype: {dtype}")
