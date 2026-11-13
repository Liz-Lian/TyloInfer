"""Freeze NINT8-0 tests and independent CPU-decoded references."""

from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .mfq_source import file_hash, open_source
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings


def _array(value):
    """Copy a Torch tensor to contiguous host storage without changing dtype."""
    return value.detach().contiguous().cpu().numpy()


def reference(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Recompute NINT8-0 reference observations from the fixed packed weight."""
    device = resolve_device(device)
    source = case.data['source']
    if (
        case.data['op'] not in {'nint8_zero_forward', 'nint8_zero_backward'}
        or source.get('generator') != 'tests/numerics/zero_ref.py'
    ):
        raise ValueError('Unsupported NINT8-0 reference case')
    modules, identity = open_source(source_root)
    if identity != source['mfq_source']:
        raise ValueError('MFQ source identity differs from the case')
    module = modules['mfq.formats.nint8_zero']
    tensor = module.unpack_nint8_zero(case.asset('weight'))
    if tensor.shape != (13, 96) or tensor.axis != 0:
        raise ValueError('NINT8-0 weight geometry differs from the case')
    with scope(case.data['config']['compute']):
        if case.data['op'] == 'nint8_zero_forward':
            x = torch.from_numpy(case.tensor('input')).to(device).float()
            if case.data['config']['input_quantization'] == 'per_32_q8':
                grouped = x.reshape(x.shape[0], 3, 32)
                scale = grouped.abs().amax(dim=-1) / 127.0
                scale = torch.where(scale > 0, scale, torch.ones_like(scale))
                quantized = torch.round(grouped / scale[..., None]).clamp(-127, 127)
                q = torch.as_tensor(tensor.q, device=device).float()
                weight_scale = torch.as_tensor(tensor.scale, device=device).float()
                result = torch.einsum('mgk,ngk,mg,ng->mn', quantized, q, scale, weight_scale)
            else:
                dense = torch.as_tensor(module.dequantize_nint8_zero(tensor), device=device)
                result = x @ dense.to(torch.float16).float().T
            outputs = {'output': result.half().float()}
            arithmetic = (
                'Original forward formula, result rounded to float16 and expanded to float32'
            )
        else:
            grad = torch.from_numpy(case.tensor('output_gradient')).to(device).float()
            dense = torch.as_tensor(module.dequantize_nint8_zero(tensor), device=device)
            expected = grad @ dense.to(torch.float16).float()
            outputs = {'input_gradient': expected, 'autograd_input_gradient': expected}
            arithmetic = (
                'Independent NumPy decode rounded to float16, Torch float32 backward matmul'
            )
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during reference computation')
        specs = {name: save_tensor(root, name, _array(value)) for name, value in outputs.items()}
        provenance = execution(device, arithmetic)
        provenance.update(generator_sha256=file_hash(__file__), mfq_source=identity)
        return write_result(case, root, specs, provenance)


def candidate(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Run the local NINT8-zero wrapper on the verified packed weight and inputs."""
    source, config, op = case.data['source'], case.data['config'], case.data['op']
    if (op not in {'nint8_zero_forward', 'nint8_zero_backward'}
            or source.get('generator') != 'tests/numerics/zero_ref.py'
            or case.data.get('actual_role') != 'candidate'):
        raise ValueError('Unsupported NINT8-0 candidate case')
    from . import zero_cuda as implementation
    from tyloinfer.kernels.cuda._ext import ext

    modules, identity = open_source(source_root)
    if identity != source['mfq_source']:
        raise ValueError('MFQ source identity differs from the case')
    tensor = modules['mfq.formats.nint8_zero'].unpack_nint8_zero(case.asset('weight'))
    if tensor.shape != (13, 96) or tensor.axis != 0 or tensor.neuron_len != 96 or config['weight_shape'] != [13, 96]:
        raise ValueError('NINT8-0 candidate weight geometry differs')
    rows = config['rows']
    if rows not in ((1, 16, 65) if op == 'nint8_zero_forward' else (3,)):
        raise ValueError('Unsupported NINT8-0 candidate rows')
    device = resolve_device(device)
    with scope(config['compute']), torch.cuda.device(device):
        weight = {
            'q': torch.from_numpy(np.ascontiguousarray(tensor.q).view(np.uint8)).to(device),
            'scale': torch.from_numpy(np.ascontiguousarray(tensor.scale)).to(device),
            'out': 13, 'ng': 3, 'neuron_len': 96,
        }
        x = torch.from_numpy(case.tensor('input')).to(device)
        if x.dtype != torch.float16 or x.shape != (rows, 96):
            raise ValueError('NINT8-0 candidate input differs from the original case')
        if op == 'nint8_zero_forward':
            outputs = {'output': implementation.nint8_zero_matmul(weight, x)}
            entry = 'nint8_zero_matmul'
        else:
            gradient = torch.from_numpy(case.tensor('output_gradient')).to(device)
            if gradient.dtype != torch.float16 or gradient.shape != (rows, 13):
                raise ValueError('NINT8-0 candidate gradient differs from the original case')
            direct = implementation.nint8_zero_backward_input(weight, gradient)
            x.requires_grad_(True)
            (implementation.nint8_zero_matmul(weight, x) * gradient).sum().backward()
            outputs = {'input_gradient': direct, 'autograd_input_gradient': x.grad}
            entry = 'nint8_zero_backward_input+nint8_zero_matmul autograd'
        specs = {name: save_tensor(root, name, _array(value)) for name, value in outputs.items()}
        module = ext()
        provenance = execution(device, entry)
        provenance.update(category='python_candidate', native_worker_executed=False,
                          mfq_source=identity, wrapper_sha256=file_hash(implementation.__file__),
                          jit_module=str(Path(module.__file__).resolve()),
                          jit_sha256=file_hash(module.__file__), generator_sha256=file_hash(__file__))
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during candidate execution')
        return write_result(case, root, specs, provenance, role='candidate')


def generate(
    root: Path, source_root: Path, suite: str = 'forward', device: str = 'cuda'
) -> list[Path]:
    """Freeze the original NINT8-0 input cases and produce fixed references."""
    if suite not in {'forward', 'backward'}:
        raise ValueError(f'Unknown NINT8-0 suite: {suite}')
    device = resolve_device(device)
    modules, identity = open_source(source_root)
    module = modules['mfq.formats.nint8_zero']
    test_path = Path(__file__).parents[1] / 'test_format_completion_cuda.py'
    paths = []
    for rows in (1, 16, 65) if suite == 'forward' else (3,):
        seed = 608 if suite == 'forward' else 1608
        rng = np.random.default_rng(seed)
        tensor = module.Nint8ZeroTensor(
            shape=(13, 96),
            axis=0,
            scale=rng.uniform(0.001, 0.02, (13, 3)).astype(np.float16),
            q=rng.integers(-127, 128, (13, 3, 32)).astype(np.int8),
            neuron_len=96,
        )
        test = (
            'test_nint8_zero_native_matmul_matches_packed_reference'
            if suite == 'forward'
            else 'test_nint8_zero_packed_backward_and_autograd_match_dequant'
        )
        torch_seed = 608 + rows if suite == 'forward' else 16083
        name = f'nint8_zero_{suite}_{rows}'
        category = (
            'original_independent_input_q8_formula'
            if suite == 'forward' and rows <= 64
            else 'new_independent_cpu_decode_supplement'
        )
        writer = CaseWriter(
            Path(root),
            name,
            f'nint8_zero_{suite}',
            {
                'rows': rows,
                'weight_shape': [13, 96],
                'weight_seed': seed,
                'input_seed': torch_seed,
                'input_seed_origin': 'original'
                if suite == 'forward'
                else 'chosen; source was unseeded',
                'input_quantization': 'per_32_q8' if suite == 'forward' and rows <= 64 else 'none',
                'category': category,
                'preserved_native_check': 'same native dequantization baseline'
                if category.startswith('new_')
                else None,
                'compute': settings(),
            },
            {
                'generator': 'tests/numerics/zero_ref.py',
                'generator_sha256': file_hash(__file__),
                'test': f'tests/test_format_completion_cuda.py::{test}',
                'test_sha256': file_hash(test_path),
                'mfq_source': identity,
                'decoder': 'mfq.formats.nint8_zero.dequantize_nint8_zero',
                'independence': 'NumPy CPU decoder does not invoke the native CUDA kernel',
            },
            actual_role='candidate',
        )
        writer.asset('weight', module.pack_nint8_zero(tensor), encoding='NINT8-0')
        torch.manual_seed(torch_seed)
        rule = {'kind': 'close', 'atol': 0.02, 'rtol': 0.004}
        if suite == 'forward':
            value = torch.randn(rows, 96, device=device, dtype=torch.float16)
            writer.input('input', _array(value))
            writer.output('output', [rows, 13], 'float32', rule, actual_dtype='float16')
        else:
            gradient = torch.randn(3, 13, device=device, dtype=torch.float16)
            value = torch.randn(3, 96, device=device, dtype=torch.float16)
            writer.input('output_gradient', _array(gradient))
            writer.input('input', _array(value))
            for output in ('input_gradient', 'autograd_input_gradient'):
                writer.output(output, [3, 96], 'float32', rule, actual_dtype='float16')
        case = writer.finish()
        reference(case, case.root / 'reference', source_root, device)
        candidate(case, case.root / 'candidate', source_root, device)
        paths.append(case.root / 'case.json')
    return paths
