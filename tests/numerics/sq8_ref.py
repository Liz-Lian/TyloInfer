"""Fixed MXFP8-SQ and FP8-128SQ references from the original CUDA cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, load_tensor, save_tensor, write_result
from .mfq_source import file_hash, open_source
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings


def _fixture(dtype, modules):
    """Build the original mixed SQ1--SQ8 payload using CPU quantization."""
    quant = modules['mfq.quantize.fp8_sq']
    rng = np.random.default_rng(20260912)
    source = rng.choice(quant.E4M3_LEGAL_CODES, size=(8, 128)).astype(np.uint8)
    q = np.arange(1, 9, dtype=np.uint8)
    if dtype == 'MXFP8-SQ':
        scales = np.full((8, 4), 127, dtype=np.uint8)
        tensor = quant.quantize_mxfp8_sq(source, scales, block_shape=(1, 32), row_q_bits=q)
    else:
        scales = np.ones((1, 1), dtype=np.float16)
        tensor = quant.quantize_fp8_128_sq(source, scales, row_q_bits=q)
    return tensor, source, scales


def _weight(case, modules):
    """Decode the persisted unit-scale payload to the original float32 matrix."""
    fmt = modules['mfq.formats.fp8_sq']
    quant = modules['mfq.quantize.fp8_sq']
    payload = case.asset('weight')
    tensor = fmt.unpack_fp8_sq(case.data['config']['format'], payload)
    layout = fmt.parse_fp8_sq_layout(tensor.dtype, payload)
    if (
        layout.shape != (8, 128)
        or layout.row_q_bits != tuple(range(1, 9))
        or payload[layout.palettes_offset : layout.symbols_offset] != case.asset('palettes')
    ):
        raise ValueError('FP8-SQ payload or saved codebook differs from the original fixture')
    scales = case.tensor('scales')
    expected_scale = (
        np.full((8, 4), 127, np.uint8)
        if tensor.dtype == 'MXFP8-SQ'
        else np.ones((1, 1), np.float16)
    )
    if (
        not np.array_equal(scales, expected_scale)
        or fmt.fp8_sq_scale_bytes(tensor) != scales.tobytes()
    ):
        raise ValueError('FP8-SQ fixture requires its saved unit scales')
    return quant.E4M3_VALUES[fmt.decode_fp8_sq_codes(tensor)].astype(np.float32)


def reference(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Recalculate one original reference from fixed payload and fixed inputs."""
    if case.data['source'].get('generator') != 'tests/numerics/sq8_ref.py' or case.data[
        'op'
    ] not in ('sq8_decode', 'sq8_forward', 'sq8_backward'):
        raise ValueError('Unsupported FP8-SQ reference case')
    modules, identity = open_source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('MFQ source commit or module hashes differ from the case')
    device = resolve_device(device)
    with scope(case.data['config']['compute']):
        dense = _weight(case, modules)
        op = case.data['op']
        if op == 'sq8_decode':
            dtype = np.float32 if case.data['config']['actual_dtype'] == 'float32' else np.float16
            value = dense.astype(dtype)
            arithmetic = 'CPU E4M3 lookup float32, cast to source decode output dtype'
        else:
            # The original reference computes CPU float64 matmul from pre-cast float32 inputs.
            inputs = torch.from_numpy(case.tensor('reference_input')).double()
            weight = torch.from_numpy(dense).double()
            value = (inputs @ (weight.T if op == 'sq8_forward' else weight)).numpy()
            arithmetic = (
                'CPU Torch float64 matmul from source float32 reference input and decoded weight'
            )
        target = Path(root)
        spec = save_tensor(target, 'output', value)
        metadata = execution(device, arithmetic)
        metadata.update(
            reference_device='cpu',
            mfq_source=identity,
            generator_sha256=file_hash(__file__),
            independence='external CPU scalar decoding and Torch formula, no native SQ kernel',
        )
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during reference calculation')
        return write_result(case, target, {'output': spec}, metadata)


def _case(
    root, source_root, modules, identity, device, dtype, suite, actual_dtype, rows, supplement=False
):
    """Persist one original fixture and call the shared reference path."""
    tensor, source, scales = _fixture(dtype, modules)
    fmt = modules['mfq.formats.fp8_sq']
    layout = fmt.parse_fp8_sq_layout(dtype, tensor.payload)
    test = {
        'decode': 'test_cuda_decode_matches_native_codepoints',
        'forward': 'test_cuda_dense_dispatch_covers_decode_and_prefill',
        'backward': 'test_cuda_backward_input',
    }[suite]
    category = (
        'same_input_independent_supplement'
        if supplement
        else 'legacy_ideal_input_baseline'
        if suite != 'decode' and actual_dtype == 'float16'
        else 'same_input_original'
    )
    config = {
        'format': dtype,
        'shape': [8, 128],
        'row_q_bits': list(layout.row_q_bits),
        'block_shape': list(layout.block_shape),
        'scale_shape': list(layout.scale_shape),
        'scale_dtype': layout.scale_dtype,
        'palettes_offset': layout.palettes_offset,
        'symbols_offset': layout.symbols_offset,
        'scales_offset': layout.scales_offset,
        'row_symbol_byte_offsets': list(layout.row_symbol_byte_offsets),
        'actual_dtype': actual_dtype,
        'category': category,
        'compute': settings(),
        'comparison_limit': (
            'legacy expected uses pre-rounding float32 input; not same-input kernel correctness'
            if category == 'legacy_ideal_input_baseline'
            else 'same-input supplement inherits original atol=0.04, rtol=0.006; no new validation evidence'
            if supplement
            else 'original source comparison'
        ),
    }
    name = f'{dtype.lower().replace("-", "_")}_{suite}_{actual_dtype}'
    if suite == 'forward':
        name += f'_m{rows}'
    if supplement:
        name += '_same_input'
    test_file = Path(source_root) / 'tests/test_fp8_sq_cuda.py'
    writer = CaseWriter(
        root,
        name,
        f'sq8_{suite}',
        config,
        {
            'generator': 'tests/numerics/sq8_ref.py',
            'generator_sha256': file_hash(__file__),
            'test': 'tests/test_fp8_sq_cuda.py::' + test,
            'test_sha256': file_hash(test_file),
            'fixture_seed': 20260912,
            'mfq_source': identity,
        },
    )
    writer.asset('weight', tensor.payload, encoding=dtype)
    writer.asset(
        'palettes', tensor.payload[layout.palettes_offset : layout.symbols_offset], encoding='E4M3'
    )
    writer.input('encoded_source', source)
    writer.input('scales', scales)
    if suite == 'decode':
        writer.output('output', (8, 128), actual_dtype, {'kind': 'bits'})
    else:
        shape = (rows, 128) if suite == 'forward' else (3, 8)
        count = shape[0] * shape[1]
        values = torch.linspace(-1.0, 1.0, max(1, count), dtype=torch.float32)[:count].reshape(
            shape
        )
        native_spec = writer.input('native_input', values.to(getattr(torch, actual_dtype)).numpy())
        reference_input = (
            load_tensor(writer.root, native_spec).astype(np.float32)
            if supplement
            else values.numpy()
        )
        writer.input('reference_input', reference_input)
        out_shape = (rows, 8) if suite == 'forward' else (3, 128)
        writer.output(
            'output',
            out_shape,
            'float64',
            {'kind': 'close', 'atol': 0.04, 'rtol': 0.006},
            actual_dtype=actual_dtype,
        )
    case = writer.finish()
    reference(case, case.root / 'reference', source_root, device)
    return case.root / 'case.json'


def generate(
    root: Path, source_root: Path, suite: str = 'decode', device: str = 'cuda'
) -> list[Path]:
    """Generate one original FP8-SQ suite and return fixed case manifests."""
    if suite not in ('decode', 'forward', 'backward'):
        raise ValueError(f'Unknown FP8-SQ suite: {suite!r}')
    modules, identity = open_source(source_root)
    device = resolve_device(device)
    paths = []
    for dtype in ('MXFP8-SQ', 'FP8-128SQ'):
        for actual_dtype in ('float16', 'float32'):
            sizes = (0, 1, 2, 3, 4, 5, 6, 7, 16, 65) if suite == 'forward' else (None,)
            for rows in sizes:
                paths.append(
                    _case(
                        root,
                        source_root,
                        modules,
                        identity,
                        device,
                        dtype,
                        suite,
                        actual_dtype,
                        rows,
                    )
                )
                if suite != 'decode' and actual_dtype == 'float16':
                    paths.append(
                        _case(
                            root,
                            source_root,
                            modules,
                            identity,
                            device,
                            dtype,
                            suite,
                            actual_dtype,
                            rows,
                            supplement=True,
                        )
                    )
    return paths
