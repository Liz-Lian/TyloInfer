import struct
import re
from pathlib import Path

import numpy as np
import pytest

from tyloinfer.formats.compat import canonical_dtype
from tyloinfer.formats.header import TENSOR_DTYPES, TYLOINFER_SUFFIXES
from tyloinfer.formats.inspection import inspect_tensor
from tyloinfer.formats.io import open_mmap, unpack_tensor_payload


ROOT = Path(__file__).resolve().parents[1]


def nint_metadata(bits, rows=2, width=32):
    header = struct.pack('<BBiiiI2qII', bits, 4, 32, 0, width, 2, rows, width, rows, 1)
    return header + bytes(rows * width * bits // 8 + rows * 6)


def test_legacy_metadata_does_not_restore_python_weight_decoding(tmp_path):
    blob = nint_metadata(5)
    string = lambda value: struct.pack('<I', len(value)) + value
    path = tmp_path / 'old.mfq'
    path.write_bytes(b'MFQ1' + struct.pack('<I', 1) + string(b'llama') + struct.pack('<I', 1)
        + string(b'weight') + string(b'NINT5') + struct.pack('<Q', len(blob)) + blob)
    with open_mmap(path) as store:
        assert store.records['weight'].dtype == 'NINT'
        with store.blob_view(store.records['weight']) as view:
            metadata = inspect_tensor('NINT5', view, rows=True)
            assert metadata.shape == (2, 32)
            assert metadata.row_bytes.sum() == len(blob)
    assert '.mfq' in TYLOINFER_SUFFIXES
    assert 'NINT' not in TENSOR_DTYPES
    with pytest.raises(ValueError, match='native compatibility runtime'):
        unpack_tensor_payload('NINT', blob)


def test_legacy_expert_metadata_keeps_each_cohorts_precision():
    pools = []
    for expert, bits in enumerate((3, 6)):
        payload = nint_metadata(bits)
        pools.append(struct.pack('<IIQQi', 1, 4, len(payload), 0, expert) + b'NINT' + payload)
    blob = struct.pack('<4sIIII', b'MFE1', 2, 2, 32, 2) + b''.join(pools)
    with memoryview(blob) as view:
        metadata = inspect_tensor('MFE', view, rows=True)
    assert metadata.shape == (2, 2, 32)
    assert metadata.row_bytes.sum() == len(blob)
    assert np.all(metadata.row_bytes[:2] < metadata.row_bytes[2:])
    bad = blob[:48] + struct.pack('<i', 3) + blob[52:]
    with memoryview(bad) as view, pytest.raises((ValueError, struct.error)):
        inspect_tensor('MFE', view, rows=True)


@pytest.mark.parametrize('alias,expected', [('NINTv2', 'NINT'), ('NVQ3J-L', 'NVQ'),
    ('NINTM', 'MFE'), ('MXFP4-SQ3', 'MXFP4-SQ'), ('TQ', 'TQ')])
def test_legacy_dtype_aliases_are_read_only_metadata(alias, expected):
    assert canonical_dtype(alias) == expected


def test_ordinary_build_is_explicit_and_compatibility_is_opt_in():
    cmake = (ROOT / 'csrc/CMakeLists.txt').read_text()
    assert 'option(TYLOINFER_ENABLE_MFQ_COMPAT' in cmake
    assert '"Include legacy MFQ inference formats" OFF)' in cmake
    for backend in ('cpu', 'cuda', 'hip', 'metal'):
        source = (ROOT / f'csrc/backends/{backend}/CMakeLists.txt').read_text()
        assert 'if(TYLOINFER_ENABLE_MFQ_COMPAT)' in source
        assert 'ops/common/' in source
        assert 'ops/mfq/' in source
    assert '-DTYLOINFER_ENABLE_MFQ_COMPAT=OFF' in (ROOT / 'tyloinfer/commands/build.py').read_text()


def test_moved_native_operators_keep_resolvable_relative_includes():
    for path in (ROOT / 'csrc/backends').rglob('*'):
        if path.suffix not in {'.h', '.cpp', '.cu', '.hip', '.mm'}:
            continue
        for include in re.findall(r'^#include\s+"(\.\.[^"]+)"', path.read_text(), re.M):
            assert (path.parent / include).is_file(), (path, include)
