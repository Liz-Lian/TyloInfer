"""Persist independent CPU Hadamard observations for CUDA grid boundaries.

The source tests have no standalone Hadamard tolerance. Both directions are
observations until an independent operator-level comparison rule is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np

from .case import Case, CaseWriter, save_tensor, write_result


_ROWS = (32, 65535, 65536, 131328)
_BLOCK = 128
_CHUNK = 1024
_SOURCE = Path(__file__).resolve().parents[1] / 'test_nepq_kernels_cuda.py'


def _hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _inputs(rows: int) -> tuple[np.ndarray, np.ndarray]:
    value = np.empty((rows, _BLOCK), dtype=np.float16)
    columns = np.arange(_BLOCK, dtype=np.int32)
    for start in range(0, rows, _CHUNK):
        positions = np.arange(start, min(rows, start + _CHUNK), dtype=np.int32)
        value[start:start + len(positions)] = (
            (positions[:, None] * 13 + columns[None, :] * 7) % 31 - 15
        ).astype(np.float16)
    signs = np.where((columns * 11 + columns // 3) % 2, -1, 1).astype(np.int8)
    return value, signs


def reference(case: Case, root: Path) -> Path:
    """Calculate a saved case in F32 chunks and persist its F16 CPU reference."""
    config = case.data['config']
    if (case.data['op'] != 'nepq_hadamard' or config.get('block_size') != _BLOCK
            or config.get('direction') not in ('input', 'adjoint')
            or config.get('rows') not in _ROWS):
        raise ValueError('Unsupported Hadamard case')
    if case.data['source'].get('sha256') != _hash(_SOURCE):
        raise ValueError('Hadamard source test changed since case creation')
    value, signs = case.tensor('input'), case.tensor('signs')
    if (value.dtype != np.float16 or value.shape != (config['rows'], _BLOCK)
            or signs.dtype != np.int8 or signs.shape != (_BLOCK,)
            or not np.isin(signs, (-1, 1)).all()):
        raise ValueError('Invalid Hadamard input or signs')
    output = np.empty(value.shape, dtype=np.float16)
    for start in range(0, value.shape[0], _CHUNK):
        end = min(start + _CHUNK, value.shape[0])
        blocks = value[start:end].astype(np.float32)
        if config['direction'] == 'input':
            blocks *= signs
        stride = 1
        while stride < _BLOCK:
            paired = blocks.reshape(-1, 2, stride)
            first = paired[:, 0].copy()
            second = paired[:, 1].copy()
            paired[:, 0] = first + second
            paired[:, 1] = first - second
            stride *= 2
        blocks *= np.float32(1.0 / math.sqrt(_BLOCK))
        if config['direction'] == 'adjoint':
            blocks *= signs
        output[start:end] = blocks.astype(np.float16)
    root = Path(root)
    spec = save_tensor(root, 'output', output)
    return write_result(case, root, {'output': spec}, {
        'framework': 'NumPy',
        'numpy_version': np.__version__,
        'device': 'cpu',
        'arithmetic': 'F32 butterflies, F32 normalization, final F16 rounding',
        'independence': 'CPU formula does not invoke native kernels',
        'source_sha256': _hash(_SOURCE),
        'generator_sha256': _hash(Path(__file__).resolve()),
    })


def generate(root: Path) -> list[Path]:
    """Create eight fixed cases and their independent CPU reference outputs."""
    root = Path(root)
    source = {
        'file': 'tests/test_nepq_kernels_cuda.py',
        'sha256': _hash(_SOURCE),
        'generator': 'tests/numerics/hadamard_ref.py',
        'generator_sha256': _hash(Path(__file__).resolve()),
        'formula': '_fwht_input F32 butterflies; adjoint moves signs after transform',
        'criterion': 'no standalone Hadamard atol/rtol; observation only',
        'related_forward': 'rotated GEMV output relative L2 < 0.012, not an operator tolerance',
        'related_backward': 'full input-gradient atol=rtol=0.006, not an operator tolerance',
    }
    paths = []
    for rows in _ROWS:
        for direction in ('input', 'adjoint'):
            writer = CaseWriter(root, f'nepq_hadamard_{direction}_{rows}', 'nepq_hadamard',
                                {'block_size': _BLOCK, 'direction': direction, 'rows': rows}, source)
            value, signs = _inputs(rows)
            writer.input('input', value, dtype='float16')
            writer.input('signs', signs, dtype='int8')
            del value, signs
            writer.output('output', (rows, _BLOCK), 'float16', {'kind': 'observe'})
            case = writer.finish()
            reference(case, case.root / 'reference')
            paths.append(case.root / 'case.json')
    return paths


def main() -> None:
    """Parse the destination and write the ordered fixed reference cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    for path in generate(args.out):
        print(path)


if __name__ == '__main__':
    main()
