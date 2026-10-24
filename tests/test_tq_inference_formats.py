import struct

import numpy as np
import pytest

from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.inspection import inspect_tensor
from tyloinfer.formats.io import load, open_mmap, save
from tyloinfer.formats.tq import TQTensor, TQExpertTensor


def tensor(bits, columns=True, width=512, salt=1):
    bits = np.asarray(bits, dtype=np.uint64)
    words = width * bits // 32
    descriptors = np.r_[np.uint64(0), np.cumsum(words[:-1])]
    descriptors |= (bits - 1) << np.uint64(32)
    descriptors |= np.uint64(0x3c00) << np.uint64(40)
    scales = np.full(width if columns else (len(bits) * (width // 16) + 1) // 2,
        salt if columns else 0x89, dtype=np.float16 if columns else np.uint8)
    packed = np.arange(int(words.sum()) * 4, dtype=np.uint8)
    return TQTensor(len(bits), width, 16, columns, descriptors, scales, packed)


@pytest.mark.parametrize('columns', [False, True])
def test_expert_container_preserves_rows_scales_and_variable_sizes(tmp_path, columns):
    value = TQExpertTensor((tensor([1, 2, 3], columns), tensor([6, 7, 8], columns, salt=2)), 511)
    raw = value.to_bytes()
    restored = TQExpertTensor.from_bytes(raw)
    assert restored.to_bytes() == raw and restored.shape == (2, 3, 511)
    with memoryview(raw) as view:
        result = inspect_tensor('TQ', view, rows=True)
    assert result.shape == restored.shape and result.granularity == 'row'
    assert result.row_bytes.sum() == pytest.approx(len(raw))
    assert result.row_bytes[:3].sum() < result.row_bytes[3:].sum()
    path = tmp_path / 'model.tq'
    save(path, FileHeader(version=2, model_arch='qwen4_exp_text'), {'model.block.0.mlp.experts.gate.weight': value})
    with open_mmap(path) as store:
        assert isinstance(store['model.block.0.mlp.experts.gate.weight'], TQExpertTensor)


@pytest.mark.parametrize('columns', [False, True])
def test_flat_row_bank_uses_logical_unpadded_shape(columns):
    value = tensor([1, 3, 6, 8], columns)
    with memoryview(value.to_bytes()) as view:
        result = inspect_tensor('TQ', view, rows=True, logical_shape=(2, 2, 509))
    assert result.shape == (2, 2, 509)
    assert result.row_bytes.sum() == pytest.approx(value.nbytes)


@pytest.mark.parametrize('dtype', ['NINT', 'NVQ', 'MFE', 'MXFP4-SQ', 'MXFP8-SQ', 'FP8-128SQ'])
def test_old_formats_are_rejected_at_the_container_boundary(tmp_path, dtype):
    path = tmp_path / 'old.tq'
    save(path, FileHeader(model_arch='test'), {'weight': tensor([4])})
    data = path.read_bytes()
    old = struct.pack('<I', 2) + b'TQ'
    new = struct.pack('<I', len(dtype)) + dtype.encode()
    path.write_bytes(data.replace(old, new, 1))
    with pytest.raises(ValueError, match='does not support tensor format'):
        load(path)


def test_tq_directory_rejects_overlaps_truncation_and_shape_mismatch():
    value = TQExpertTensor((tensor([2]), tensor([5], salt=2)), 511)
    raw = value.to_bytes()
    mutations = [raw[:-1], raw[:64] + struct.pack('<Q', 0) + raw[72:], raw + b'x']
    for bad in mutations:
        with pytest.raises(ValueError):
            TQExpertTensor.from_bytes(bad)
        with memoryview(bad) as view, pytest.raises(ValueError):
            inspect_tensor('TQ', view, rows=True)
