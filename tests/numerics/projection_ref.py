"""Independent CPU references for the native synthetic NINT projection gate."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
from pathlib import Path

import numpy as np

from .case import Case, CaseWriter, save_tensor, write_result

_TEST = 'csrc/engine/backends/cuda/tests/float_projection_test.cpp'


def _fixture(rows, width, group, zero):
    """Build the original packed mixed-width or signed-Q8 synthetic buffers."""
    groups = (width + group - 1) // group
    padded = groups * group
    ids = np.arange(rows, dtype=np.int64)
    bits = (1 + ids % 8).astype(np.uint8)
    lengths = padded * (np.full(rows, 8, np.int64) if zero else bits.astype(np.int64))
    offsets = np.concatenate((np.array([0], np.int64), np.cumsum(lengths[:-1], dtype=np.int64)))
    packed = np.zeros((int(lengths.sum()) + 7) // 8 + 8, dtype=np.uint8)
    cols = np.arange(padded, dtype=np.int64)
    for row in range(rows):
        q = ((row * 17 + cols * 3) % (255 if zero else (1 << int(bits[row])))).astype(np.uint8)
        if zero:
            packed[row * padded : (row + 1) * padded] = q
        else:
            stream = np.unpackbits(q[:, None], axis=-1, bitorder='little')[:, : int(bits[row])]
            data = np.packbits(stream.reshape(-1), bitorder='little')
            start = int(offsets[row]) // 8
            packed[start : start + data.size] = data
    result = {'q_packed': packed}
    metadata = ids[:, None] * groups + np.arange(groups, dtype=np.int64)[None, :]
    if zero:
        values = (1 + metadata % 13).astype(np.float32) * np.float32(0.003)
        result['zero_scale'] = values.astype(np.float16)
    else:
        g = np.arange(groups, dtype=np.int64)[None, :]
        result.update(
            row_q_bits=bits,
            row_q_bit_offsets=offsets,
            sub_scale=(1 + (ids[:, None] + g * 7) % 24).astype(np.uint8),
            sub_min=((ids[:, None] + g) % 4).astype(np.uint8),
            neuron_scale=(1 + ids % 13).astype(np.float32) * np.float32(0.0001),
            neuron_min=(1 + ids % 7).astype(np.float32) * np.float32(0.00001),
        )
    return result


def _input(batch, width):
    """Evaluate the source C++ float sin expression with the platform sinf."""
    name = ctypes.util.find_library('m') or ctypes.util.find_library('msvcrt')
    if not name:
        raise RuntimeError('A C math library with sinf is required for the source input')
    lib = ctypes.CDLL(name)
    sinf = lib.sinf
    sinf.argtypes, sinf.restype = [ctypes.c_float], ctypes.c_float
    result = np.empty(batch * width, dtype=np.float32)
    for i in range(result.size):
        argument = np.float32(i) * np.float32(0.031)
        result[i] = np.float32(sinf(float(argument))) * np.float32(0.1)
    return result.reshape(batch, width), name


def _dense(case):
    """Decode fixed row streams with the source's float32 arithmetic order."""
    config = case.data['config']
    rows, width, group = config['outputs'], config['width'], config['group_size']
    groups = (width + group - 1) // group
    packed = case.tensor('q_packed')
    if packed.size < 8 or np.any(packed[-8:] != 0):
        raise ValueError('NINT packed fixture requires eight zero guard bytes')
    dense = np.empty((rows, width), dtype=np.float32)
    if config['zero']:
        codes = packed[: rows * groups * group].view(np.int8).reshape(rows, groups * group)
        scales = case.tensor('zero_scale').astype(np.float32)
        for row in range(rows):
            dense[row] = (
                codes[row, :width].astype(np.float32) * np.repeat(scales[row], group)[:width]
            )
        return dense
    bits, offsets = case.tensor('row_q_bits'), case.tensor('row_q_bit_offsets')
    sub_scale, sub_min = case.tensor('sub_scale'), case.tensor('sub_min')
    anchor, minimum = case.tensor('neuron_scale'), case.tensor('neuron_min')
    columns = np.arange(width, dtype=np.int64)
    for row in range(rows):
        bitpos = offsets[row] + columns * int(bits[row])
        byte, shift = bitpos // 8, bitpos % 8
        words = packed[byte].astype(np.uint16) | (packed[byte + 1].astype(np.uint16) << 8)
        codes = ((words >> shift) & ((1 << int(bits[row])) - 1)).astype(np.float32)
        scale = anchor[row] * sub_scale[row].astype(np.float32)
        lower = minimum[row] * sub_min[row].astype(np.float32)
        dense[row] = np.repeat(scale, group)[:width] * codes - np.repeat(lower, group)[:width]
    return dense


def reference(case: Case, root: Path) -> Path:
    """Recompute split and unsplit expected outputs from fixed CPU buffers."""
    if (
        case.data['op'] != 'nint_synthetic_projection'
        or case.data['source'].get('generator') != 'tests/numerics/projection_ref.py'
    ):
        raise ValueError('Unsupported synthetic projection case')
    dense = _dense(case)
    values = case.tensor('input')
    result = np.zeros((values.shape[0], dense.shape[0]), dtype=np.float64)
    for k in range(dense.shape[1]):
        result += values[:, k, None].astype(np.float64) * dense[None, :, k].astype(np.float64)
    expected = result.astype(np.float32)
    target = Path(root)
    outputs = {name: save_tensor(target, name, expected) for name in ('split', 'unsplit')}
    return write_result(
        case,
        target,
        outputs,
        {
            'framework': 'NumPy',
            'numpy_version': np.__version__,
            'device': 'cpu',
            'arithmetic': 'float32 scalar decode, sequential FP64 dot product, float32 output',
            'independence': 'standalone CPU bit decoding and arithmetic, no native projection kernel',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


def generate(root: Path) -> list[Path]:
    """Freeze the 18 original synthetic projection input and weight cases."""
    geometries = [(17, 77, group, batch, False) for group in (32, 48, 64) for batch in (1, 3, 11)]
    geometries += [(17, 77, 32, batch, True) for batch in (1, 3, 11)]
    geometries += [
        (rows, width, 64, batch, False)
        for rows, width in ((320, 10240), (10240, 320), (4, 10240))
        for batch in (1, 2)
    ]
    test = Path(__file__).parents[2] / _TEST
    paths = []
    for rows, width, group, batch, zero in geometries:
        buffers = _fixture(rows, width, group, zero)
        inputs, library = _input(batch, width)
        name = f'nint_synthetic_n{rows}_k{width}_g{group}_m{batch}_z{int(zero)}'
        writer = CaseWriter(
            root,
            name,
            'nint_synthetic_projection',
            {
                'outputs': rows,
                'width': width,
                'group_size': group,
                'batch': batch,
                'zero': zero,
                'input_precision': 'float32',
                'input_sinf_library': library,
                'input_formula': 'sinf(float32(index)*0.031f)*0.1f',
                'native_checks': ['split_projection', 'unsplit_projection'],
            },
            {
                'generator': 'tests/numerics/projection_ref.py',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'test': _TEST + '::synthetic',
                'test_sha256': hashlib.sha256(test.read_bytes()).hexdigest(),
            },
        )
        for field, value in buffers.items():
            writer.input(field, value)
        writer.input('input', inputs)
        for output in ('split', 'unsplit'):
            writer.output(
                output, (batch, rows), 'float32', {'kind': 'close', 'atol': 1e-5, 'rtol': 1e-4}
            )
        case = writer.finish()
        reference(case, case.root / 'reference')
        paths.append(case.root / 'case.json')
    return paths
