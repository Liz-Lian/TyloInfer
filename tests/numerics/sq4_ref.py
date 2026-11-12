"""Freeze MXFP4-SQ wire fixtures and independently calculate references."""

import hashlib
import math
import re
import struct
from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .mfq_source import file_hash, load_fixture, open_source
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings

_FIXTURE = 'bench/cuda_mxfp4_sq_fixtures.py'
_PALETTE = 'cpp_runtime/backends/metal/ops/mlx_mxfp4_sq.h'
_MAGNITUDES = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)


def _palette(source, bits):
    """Extract the shared fixed palette table from its recorded source bytes."""
    match = re.search(
        rb'kMxfp4Sq' + str(bits).encode() + rb'PaletteNibbles\s*\{(.*?)\};', source, re.S
    )
    if match is None:
        raise ValueError('Missing SQ palette')
    values = np.array([int(value) for value in re.findall(rb'\d+', match.group(1))], dtype=np.uint8)
    if values.size != 32 * (1 << bits) or np.any(values > 15):
        raise ValueError('Invalid SQ palette')
    return values.reshape(32, 1 << bits)


def _bits(blob, offset, count, width):
    """Read a little-endian packed integer stream and return its next offset."""
    length = (count * width + 7) // 8
    data = np.frombuffer(blob[offset : offset + length], dtype=np.uint8)
    if data.size != length:
        raise ValueError('Truncated SQ stream')
    stream = np.unpackbits(data, bitorder='little')[: count * width].reshape(count, width)
    values = (stream.astype(np.uint16) << np.arange(width, dtype=np.uint16)).sum(axis=1)
    return values.astype(np.uint8), offset + length


def _decode(blob, tables):
    """Decode the standalone fixture wire format without native decoder code."""
    if len(blob) < 24:
        raise ValueError('Truncated SQ header')
    magic, version, base, reserved, rows, width = struct.unpack_from('<4sBBHQQ', blob)
    if reserved or base > 251 or not rows or not width or width % 32:
        raise ValueError('Invalid SQ geometry')
    adaptive = magic == b'SQV2' and version == 2
    if adaptive:
        raw_q, offset = _bits(blob, 24, rows, 2)
        widths = raw_q.astype(np.int64) + 1
        offset = 24 + ((offset - 24 + 3) // 4 * 4)
    elif magic in (b'SQ2\0', b'SQ3\0') and version == 1:
        widths = np.full(rows, int(chr(magic[2])), dtype=np.int64)
        offset = 24
    else:
        raise ValueError('Unsupported SQ fixture format')
    symbols = []
    for bits in widths:
        row, offset = _bits(blob, offset, width, int(bits))
        symbols.append(row)
    blocks = width // 32
    sq_rows = int(np.count_nonzero(widths != 4))
    native_rows = int(rows - sq_rows)
    selectors, offset = _bits(blob, offset, sq_rows * blocks, 1)
    scales, offset = _bits(blob, offset, sq_rows * 8, 2)
    palettes, offset = _bits(blob, offset, sq_rows * 8, 5)
    native = np.frombuffer(blob[offset:], dtype=np.uint8)
    if native.size != native_rows * blocks:
        raise ValueError('Invalid SQ payload length')
    result = np.empty((rows, width), dtype=np.float32)
    sq_row = native_row = 0
    for row, bits in enumerate(widths):
        for block in range(blocks):
            start = block * 32
            values = symbols[row][start : start + 32]
            if bits == 4:
                exponent = int(native[native_row * blocks + block])
                nibbles = values
            else:
                if bits == 1:
                    low = 0
                    for lane, value in enumerate(values):
                        low ^= int(value) << (lane & 1)
                else:
                    low = int(np.bitwise_xor.reduce(values))
                tag = (low & 3) | (int(selectors[sq_row * blocks + block]) << 2)
                state = sq_row * 8 + tag
                exponent = base + int(scales[state])
                nibbles = tables[int(bits)][int(palettes[state]), values]
            scale = math.ldexp(1.0, exponent - 127)
            numbers = _MAGNITUDES[nibbles & 7] * scale
            numbers = np.copysign(numbers, np.where(nibbles & 8, -1.0, 1.0))
            with np.errstate(over='ignore'):
                result[row, start : start + 32] = numbers.astype(np.float32)
        if bits == 4:
            native_row += 1
        else:
            sq_row += 1
    return result


def reference(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Recompute observations from saved SQ bytes and the saved shared tables."""
    device = resolve_device(device)
    _, identity = open_source(source_root)
    source = case.data['source']
    if (
        case.data['op'] not in {'sq4_decode', 'sq4_forward', 'sq4_backward'}
        or source.get('generator') != 'tests/numerics/sq4_ref.py'
    ):
        raise ValueError('Unsupported SQ reference case')
    if identity != source['mfq_source']:
        raise ValueError('MFQ source differs from the case')
    _, fixture_identity = load_fixture(source_root, _FIXTURE)
    if (
        fixture_identity != source['fixture_sha256']
        or file_hash(Path(source_root) / _PALETTE) != source['palette_sha256']
    ):
        raise ValueError('Fixture or shared palette source changed')
    palette_source = case.asset('palette_source')
    if hashlib.sha256(palette_source).hexdigest() != source['palette_sha256']:
        raise ValueError('Saved palette source differs from its recorded identity')
    tables = {bits: _palette(palette_source, bits) for bits in (1, 2, 3)}
    for bits in tables:
        if not np.array_equal(tables[bits].reshape(-1), case.tensor(f'palette{bits}')):
            raise ValueError('Saved palette table differs from its source')
    dense = _decode(case.asset('weight'), tables)
    config = case.data['config']
    if list(dense.shape) != config['weight_shape']:
        raise ValueError('SQ weight geometry differs from the case')
    with scope(config['compute']):
        if case.data['op'] == 'sq4_decode':
            with np.errstate(over='ignore'):
                outputs = {'f32': dense, 'f16': dense.astype(np.float16)}
            arithmetic = 'Independent scalar wire decode; float16 nearest rounding'
        else:
            values = torch.from_numpy(case.tensor('reference_input')).double()
            weight = torch.from_numpy(dense).double()
            result = values @ (weight.T if case.data['op'] == 'sq4_forward' else weight)
            outputs = {'output': result.numpy()}
            arithmetic = 'Torch CPU float64 matmul using source test reference input'
        specs = {name: save_tensor(root, name, value) for name, value in outputs.items()}
        provenance = execution(device, arithmetic)
        provenance.update(
            generator_sha256=file_hash(__file__),
            mfq_source=identity,
            fixture_sha256=fixture_identity,
            palette_sha256=source['palette_sha256'],
        )
        return write_result(case, root, specs, provenance)


def _geometries():
    """Enumerate the 37 original native fixture geometries."""
    cases = [(bits, n, k, 120) for bits in (2, 3) for n in (1, 7, 33, 128) for k in (32, 96, 256)]
    cases += [(bits, 128, 256, base) for bits in (2, 3) for base in (0, 1, 251)]
    cases += [(0, n, k, 120) for n, k in ((4, 32), (7, 96), (33, 96), (128, 256))]
    cases += [(0, 128, 256, base) for base in (0, 1, 251)]
    return cases


def _make_case(
    root,
    source_root,
    device,
    fixture,
    identity,
    fixture_identity,
    palettes,
    geometry,
    suite,
    rows=None,
    dtype=None,
    external=False,
):
    """Persist one native or external test observation and its exact fixture."""
    bits, n, k, base = geometry
    blob, original_dense = (
        fixture.adaptive_fixture(n, k, base) if bits == 0 else fixture.fixture(bits, n, k, base)
    )
    tables = {bits_: _palette(palettes, bits_) for bits_ in (1, 2, 3)}
    dense = _decode(blob, tables)
    if dense.astype('<f4', copy=False).tobytes() != original_dense:
        raise ValueError('Independent SQ decoder disagrees with original fixture')
    label = f'sq{bits or "v2"}_n{n}_k{k}_b{base}'
    if suite != 'decode':
        label += f'_{suite}_m{rows}_{dtype}' + ('_external' if external else '_native')
    native_test = 'csrc/engine/backends/cuda/tests/mfq_mxfp4_sq_test.cu'
    external_test = 'tests/test_mxfp4_sq_cuda.py::test_packed_backward_input'
    source_path = (
        Path(source_root) / external_test.split('::')[0]
        if external
        else Path(__file__).parents[2] / native_test
    )
    category = (
        'bitwise_decode_reference'
        if suite == 'decode'
        else 'legacy_ideal_input_baseline'
        if suite == 'backward' and dtype == 'float16' and not external
        else 'same_input_independent_reference'
    )
    writer = CaseWriter(
        root,
        label,
        f'sq4_{suite}',
        {
            'bits': bits,
            'weight_shape': [n, k],
            'base': base,
            'rows': rows,
            'input_dtype': dtype,
            'compute': settings(),
            'test_origin': 'external_python' if external else 'tyloinfer_native',
            'category': category,
            'reference_input': 'pre-rounding source values'
            if not external
            else 'actual rounded gradient',
            'native_checks': ['decode_fp32_fp16_bitwise']
            if suite == 'decode'
            else ['shape', 'dtype'],
        },
        {
            'generator': 'tests/numerics/sq4_ref.py',
            'generator_sha256': file_hash(__file__),
            'test': external_test if external else native_test,
            'test_sha256': file_hash(source_path),
            'mfq_source': identity,
            'fixture_sha256': fixture_identity,
            'palette_path': _PALETTE,
            'palette_sha256': file_hash(Path(source_root) / _PALETTE),
            'independence': (
                'Standalone Python wire formula with shared Metal palette data; '
                'legacy ideal-input backward uses pre-float16 values'
                if category == 'legacy_ideal_input_baseline'
                else 'Standalone Python wire formula with shared Metal palette data; '
                'reference uses the actual input values'
            ),
        },
    )
    writer.asset('weight', blob, encoding='MXFP4-SQ')
    writer.asset('palette_source', palettes, encoding='source-text')
    for width_, table in tables.items():
        writer.input(f'palette{width_}', table.reshape(-1))
    if suite == 'decode':
        writer.output('f32', [n, k], 'float32', {'kind': 'bits'}, allow_infinity=True)
        writer.output('f16', [n, k], 'float16', {'kind': 'bits'}, allow_infinity=True)
    else:
        input_width = k if suite == 'forward' else n
        torch_dtype = torch.float16 if dtype == 'float16' else torch.float32
        if external:
            seed = 260930 + rows + (100 if bits == 3 else 0)
            torch.manual_seed(seed)
            native = torch.randn(rows, input_width, device=device, dtype=torch_dtype)
            values = native.float().cpu()
            writer.data['config']['input_seed'] = seed
            writer.data['config']['seed_origin'] = 'chosen; source was unseeded'
        else:
            index = torch.arange(rows * input_width, dtype=torch.int64)
            values = (
                ((index * 17 + 11) % 65 - 32).float() / 32
                if suite == 'forward'
                else ((index * 13 + 7) % 49 - 24).float() / 24
            ).reshape(rows, input_width)
            native = values.to(device=device, dtype=torch_dtype)
        writer.input('native_input', native.detach().contiguous().cpu().numpy())
        writer.input('reference_input', values.contiguous().numpy())
        writer.output(
            'output',
            [rows, n if suite == 'forward' else k],
            'float64',
            {'kind': 'close', 'atol': 0.02, 'rtol': 0.006},
            actual_dtype=dtype,
        )
    case = writer.finish()
    reference(case, case.root / 'reference', source_root, device)
    return case.root / 'case.json'


def generate(
    root: Path, source_root: Path, suite: str = 'decode', device: str = 'cuda'
) -> list[Path]:
    """Freeze SQ decode, forward or backward source cases and references."""
    if suite not in {'decode', 'forward', 'backward'}:
        raise ValueError(f'Unknown SQ4 suite: {suite}')
    device = resolve_device(device)
    _, identity = open_source(source_root)
    fixture, fixture_identity = load_fixture(source_root, _FIXTURE)
    palettes = (Path(source_root) / _PALETTE).read_bytes()
    results = []
    for geometry in _geometries():
        if suite == 'decode':
            results.append(
                _make_case(
                    Path(root),
                    source_root,
                    device,
                    fixture,
                    identity,
                    fixture_identity,
                    palettes,
                    geometry,
                    suite,
                )
            )
        elif geometry[3] == 120:
            for rows in (0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 32, 65):
                for dtype in ('float16', 'float32'):
                    results.append(
                        _make_case(
                            Path(root),
                            source_root,
                            device,
                            fixture,
                            identity,
                            fixture_identity,
                            palettes,
                            geometry,
                            suite,
                            rows,
                            dtype,
                        )
                    )
    if suite == 'backward':
        for bits in (2, 3):
            for rows in (1, 2, 3, 4, 5, 8, 16):
                for dtype in ('float16', 'float32'):
                    results.append(
                        _make_case(
                            Path(root),
                            source_root,
                            device,
                            fixture,
                            identity,
                            fixture_identity,
                            palettes,
                            (bits, 33, 96, 120),
                            suite,
                            rows,
                            dtype,
                            external=True,
                        )
                    )
    return results
