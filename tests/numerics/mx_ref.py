"""Persist independent references for the original CUDA MX format cases."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from tyloinfer.formats.mx import MxTensor, pack_mx, unpack_mx

from .case import Case, CaseWriter, save_tensor, write_result
from .torch_env import device as _device
from .torch_env import execution, scope, settings


def _array(value):
    """Copy a Torch tensor to contiguous host storage at its existing precision."""
    return value.detach().contiguous().cpu().numpy()


def _fixture(dtype, shape, seed):
    """Reproduce the original MX value and scale factories for one shape."""
    rng = np.random.default_rng(seed)
    rows, width = shape
    if dtype == 'MXFP4':
        values = rng.integers(0, 256, (rows, width // 2), dtype=np.uint8)
        scales = rng.integers(124, 130, (rows, width // 32), dtype=np.uint8)
    elif dtype == 'MXFP8':
        values = rng.integers(0, 255, (rows, width), dtype=np.uint8)
        values[(values & 127) == 127] = 126
        scales = rng.integers(124, 130, ((rows + 127) // 128, width // 128), dtype=np.uint8)
    else:
        raise ValueError(f'Unsupported MX dtype: {dtype}')
    return MxTensor(dtype, shape, values, scales)


def _dense(tensor):
    """Decode the fixed bytes with the original test's independent NumPy formulas."""
    rows, width = tensor.shape
    if tensor.dtype == 'MXFP4':
        packed = np.stack((tensor.values & 15, tensor.values >> 4), axis=-1).reshape(rows, width)
        magnitude = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
        dense = magnitude[packed & 7]
        dense = np.where((packed & 8) == 0, dense, -dense)
        dense *= np.repeat(np.exp2(tensor.scales.astype(np.int16) - 127), 32, axis=1)
        return dense
    unsigned = tensor.values.astype(np.uint16)
    exponent = (unsigned >> 3) & 15
    mantissa = unsigned & 7
    subnormal = np.ldexp(mantissa.astype(np.float32) / 8.0, -6)
    normal = np.ldexp(1.0 + mantissa.astype(np.float32) / 8.0, exponent.astype(np.int16) - 7)
    dense = np.where((unsigned & 128) == 0, normal, -normal)
    dense = np.where(exponent == 0, np.sign(dense) * subnormal, dense)
    scales = np.repeat(np.repeat(tensor.scales, 128, axis=0), 128, axis=1)
    dense *= np.exp2(scales[:rows, :width].astype(np.int16) - 127)
    return dense


def _provenance(device):
    """Describe the arithmetic environment used by the independent reference."""
    return {
        **execution(device, 'float32'),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def reference(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Recalculate MX outputs from a validated case without sampling new inputs."""
    with scope(case.data['config']['compute']):
        return _reference(case, root, device)


def _reference(case, root, device):
    """Compute one MX reference with the caller's selected matmul settings."""
    device = _device(device)
    config = case.data['config']
    op = case.data['op']
    if case.data['source'].get('generator') != 'tests/numerics/mx_ref.py':
        raise ValueError('Unsupported MX reference source')
    if op not in {'mx_forward', 'mx_backward', 'mx_embedding'}:
        raise ValueError('Unsupported MX reference operation')
    tensor = unpack_mx(config['dtype'], case.asset('weight'))
    if list(tensor.shape) != config['weight_shape']:
        raise ValueError('MX payload does not match case shape')
    dense = _dense(tensor)
    weight = torch.as_tensor(dense, device=device)
    if op == 'mx_forward':
        source = torch.from_numpy(case.tensor('source')).to(device)
        outputs = {'output': source.float() @ weight.T}
        if 'dequantized' in case.data['outputs']:
            outputs['dequantized'] = weight
    elif op == 'mx_backward':
        gradient = torch.from_numpy(case.tensor('output_gradient')).to(device)
        expected = gradient.float() @ weight
        outputs = {name: expected for name in case.data['outputs']}
    else:
        ids = case.tensor('ids')
        expected = np.zeros((*ids.shape, tensor.shape[1]), dtype=np.float16)
        valid = (ids >= 0) & (ids < tensor.shape[0])
        expected[valid] = dense[ids[valid]].astype(np.float16)
        outputs = {'output': torch.from_numpy(expected).to(device)}
    specs = {name: save_tensor(root, name, _array(value)) for name, value in outputs.items()}
    return write_result(case, root, specs, _provenance(device))


def _make_case(
    root,
    device,
    name,
    test,
    op,
    dtype,
    shape,
    weight_seed,
    rows,
    seed,
    output_kind='native',
    autograd=False,
    graph=False,
):
    """Freeze one original scenario and generate its reference from saved inputs."""
    tensor = _fixture(dtype, shape, weight_seed)
    config = {
        'dtype': dtype,
        'weight_shape': list(shape),
        'activation_rows': rows,
        'weight_seed': weight_seed,
        'input_seed': seed,
        'output_kind': output_kind,
        'execution_checks': ['cuda_graph_capture_and_replay'] if graph else [],
        'compute': settings(),
    }
    test_path = Path(__file__).resolve().parents[1] / 'test_format_completion_cuda.py'
    source = {
        'generator': 'tests/numerics/mx_ref.py',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test': f'tests/test_format_completion_cuda.py::{test}',
        'test_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest(),
        'formula': 'NumPy packed MX decode followed by Torch float32 matrix multiplication',
        'seed_note': 'Input seed freezes an originally unseeded case'
        if seed >= 10000
        else 'Original input seed',
    }
    writer = CaseWriter(root, name, op, config, source)
    writer.asset('weight', pack_mx(tensor), encoding='MXT1')
    torch.manual_seed(seed)
    if op == 'mx_forward':
        value = torch.randn(rows, shape[1], device=device, dtype=torch.float16)
        writer.input('source', _array(value))
        tolerance = 0.02 if dtype == 'MXFP4' else 0.03
        writer.output(
            'output',
            [rows, shape[0]],
            'float32',
            {'kind': 'close', 'atol': tolerance, 'rtol': 0.006},
            actual_dtype='float32' if output_kind == 'float32' else 'float16',
        )
        if output_kind == 'native':
            writer.output(
                'dequantized', shape, 'float32', {'kind': 'exact'}, actual_dtype='float16'
            )
    elif op == 'mx_backward':
        gradient = torch.randn(rows, shape[0], device=device, dtype=torch.float16)
        if graph:
            writer.input('initial_output_gradient', _array(gradient))
            gradient.add_(0.25)
        writer.input('output_gradient', _array(gradient))
        if autograd:
            value = torch.randn(rows, shape[1], device=device, dtype=torch.float16)
            writer.input('source', _array(value))
        rule = {'kind': 'close', 'atol': 0.03, 'rtol': 0.006}
        writer.output('input_gradient', [rows, shape[1]], 'float32', rule, actual_dtype='float16')
        if autograd:
            writer.output(
                'autograd_input_gradient', [rows, shape[1]], 'float32', rule, actual_dtype='float16'
            )
    else:
        writer.input('ids', np.asarray([[0, shape[0] - 1], [2, -1]], dtype=np.int64))
        writer.output('output', [2, 2, shape[1]], 'float16', {'kind': 'exact'})
    case = writer.finish()
    reference(case, case.root / 'reference', device)
    return case.root / 'case.json'


def generate(root: Path, suite: str = 'forward', device: str = 'cuda') -> list[Path]:
    """Freeze the original MX forward or input-backward cases and references."""
    if suite not in {'forward', 'backward'}:
        raise ValueError(f'Unsupported MX suite: {suite}')
    device = _device(device)
    cases = []
    for dtype in ('MXFP4', 'MXFP8'):
        shape = (11, 96) if dtype == 'MXFP4' else (11, 128)
        weight_seed = 604 if dtype == 'MXFP4' else 608
        if suite == 'forward':
            for rows in (1, 16, 65):
                test = (
                    'test_mxfp4_native_matmul_matches_packed_reference'
                    if dtype == 'MXFP4'
                    else 'test_mxfp8_matmul_matches_packed_reference'
                )
                name = f'{dtype.lower()}_forward_{rows}'
                cases.append(
                    _make_case(
                        root,
                        device,
                        name,
                        test,
                        'mx_forward',
                        dtype,
                        shape,
                        weight_seed,
                        rows,
                        10000 + weight_seed + rows,
                    )
                )
            cases.append(
                _make_case(
                    root,
                    device,
                    f'{dtype.lower()}_embedding',
                    'test_mx_embedding_decodes_only_selected_rows',
                    'mx_embedding',
                    dtype,
                    shape,
                    weight_seed,
                    2,
                    10000 + weight_seed,
                )
            )
            if dtype == 'MXFP8':
                for rows in (16, 65):
                    cases.append(
                        _make_case(
                            root,
                            device,
                            f'mxfp8_fp32_forward_{rows}',
                            'test_mxfp8_native_fp32_output_matches_packed_reference',
                            'mx_forward',
                            dtype,
                            shape,
                            weight_seed,
                            rows,
                            12000 + rows,
                            output_kind='float32',
                        )
                    )
        else:
            for rows in (1, 2, 3, 4, 5, 7, 8, 16):
                seed = 1600 + rows + (100 if dtype == 'MXFP8' else 0)
                cases.append(
                    _make_case(
                        root,
                        device,
                        f'{dtype.lower()}_backward_{rows}',
                        'test_mx_packed_backward_and_autograd_match_dequant',
                        'mx_backward',
                        dtype,
                        shape,
                        weight_seed,
                        rows,
                        seed,
                        autograd=True,
                    )
                )
    if suite == 'backward':
        for shape in ((257, 256), (257, 1024)):
            for rows in (5, 8):
                cases.append(
                    _make_case(
                        root,
                        device,
                        f'mxfp8_tiled_{shape[1]}_{rows}',
                        'test_mxfp8_tiled_backward_matches_packed_reference',
                        'mx_backward',
                        'MXFP8',
                        shape,
                        1608 + shape[1],
                        rows,
                        13000 + shape[1] + rows,
                    )
                )
        for shape in ((127, 128), (129, 256), (385, 384), (769, 512), (1024, 512)):
            cases.append(
                _make_case(
                    root,
                    device,
                    f'mxfp8_single_{shape[0]}_{shape[1]}',
                    'test_mxfp8_single_row_backward_matches_packed_reference',
                    'mx_backward',
                    'MXFP8',
                    shape,
                    3608 + shape[0],
                    1,
                    14000 + shape[0],
                )
            )
        cases.append(
            _make_case(
                root,
                device,
                'mxfp8_graph_replay',
                'test_mxfp8_wide_backward_cuda_graph_replays',
                'mx_backward',
                'MXFP8',
                (257, 1024),
                2608,
                8,
                15000,
                graph=True,
            )
        )
    return cases
