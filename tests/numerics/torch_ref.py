"""Generate fixed independent PyTorch references for existing CUDA cases."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from .case import CaseWriter, save_tensor, write_result


def _array(value):
    """Copy a tensor to a contiguous host array without changing its dtype."""
    return value.detach().contiguous().cpu().numpy()


def _execution(device, accumulation):
    """Describe the framework and settings used to calculate the reference."""
    return {
        'framework': 'PyTorch',
        'torch_version': str(torch.__version__),
        'numpy_version': np.__version__,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device),
        'device_capability': list(torch.cuda.get_device_capability(device)),
        'cuda_version': torch.version.cuda,
        'accumulation': accumulation,
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
    }


def _generate_case(root, device, name, op, seed, tensors, config, formula, tolerance):
    """Save generated inputs, reload them, and calculate one fixed reference."""
    torch.manual_seed(seed)
    writer = CaseWriter(
        root,
        name,
        op,
        config,
        {
            'generator': 'tests/numerics/torch_ref.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'test': f'tests/test_kernels_cuda.py::{name}',
            'formula': formula,
            'seed': seed,
        },
    )
    for tensor_name, shape, dtype in tensors:
        writer.input(tensor_name, _array(torch.randn(*shape, device=device, dtype=dtype)))

    shape = tensors[0][1]
    dtype = tensors[0][2]
    dtype_name = 'float16' if dtype == torch.float16 else 'float32'
    writer.output(
        'output',
        shape,
        dtype_name,
        {
            'kind': 'close',
            'atol': tolerance,
            'rtol': tolerance,
        },
    )
    case = writer.finish()
    inputs = {
        tensor_name: torch.from_numpy(case.tensor(tensor_name)).to(device)
        for tensor_name, _, _ in tensors
    }
    if op == 'rms_norm':
        x, w = inputs['x'], inputs['w']
        var = x.pow(2).mean(-1, keepdim=True)
        output = (x * torch.rsqrt(var + config['eps'])) * w
        accumulation = 'float32'
    elif op == 'l2_norm':
        output = functional.normalize(inputs['x'], dim=-1, eps=config['eps'])
        accumulation = 'float32'
    elif op == 'acc':
        output = inputs['a'] + inputs['b']
        accumulation = dtype_name
    elif op == 'silu_mul':
        gate, up = inputs['gate'], inputs['up']
        if dtype == torch.float16:
            output = (functional.silu(gate.float()) * up.float()).to(torch.float16)
            accumulation = 'float32, output rounded to float16'
        else:
            output = functional.silu(gate) * up
            accumulation = 'float32'
    else:
        raise ValueError(f'Unsupported reference operation: {op}')

    reference_root = case.root / 'reference'
    reference_root.mkdir(parents=True, exist_ok=True)
    spec = save_tensor(reference_root, 'output', _array(output))
    write_result(case, reference_root, {'output': spec}, _execution(device, accumulation))
    return case.root / 'case.json'


def generate(root: Path, suite: str = 'rms', device: str = 'cuda') -> list[Path]:
    """Generate the requested cases and return their case metadata paths.

    The RMS suite contains the original RMSNorm case; basic contains the
    independent L2, accumulation, and SiLU multiplication cases. Inputs are
    persisted and reloaded before reference calculation.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('These reference cases require a CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if suite == 'rms':
        cases = [
            (
                'test_rms_norm_cuda',
                'rms_norm',
                0,
                [('x', (8, 128), torch.float32), ('w', (128,), torch.float32)],
                {'eps': 1e-6},
                '(x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w',
                1e-5,
            ),
        ]
    elif suite == 'basic':
        cases = [
            (
                'test_l2_norm_cuda',
                'l2_norm',
                1,
                [('x', (4, 64), torch.float32)],
                {'eps': 1e-5, 'dim': -1},
                'functional.normalize(x, dim=-1, eps=1e-5)',
                1e-5,
            ),
            (
                'test_acc_cuda',
                'acc',
                2,
                [('a', (7, 32), torch.float32), ('b', (7, 32), torch.float32)],
                {},
                'a + b',
                1e-6,
            ),
            (
                'test_acc_cuda_f16_preserves_dtype',
                'acc',
                202,
                [('a', (7, 32), torch.float16), ('b', (7, 32), torch.float16)],
                {},
                'a + b',
                0,
            ),
            (
                'test_silu_mul_cuda_f32',
                'silu_mul',
                21,
                [('gate', (5, 37), torch.float32), ('up', (5, 37), torch.float32)],
                {},
                'functional.silu(gate) * up',
                1e-6,
            ),
            (
                'test_silu_mul_cuda_f16',
                'silu_mul',
                22,
                [('gate', (6, 41), torch.float16), ('up', (6, 41), torch.float16)],
                {},
                '(functional.silu(gate.float()) * up.float()).to(torch.float16)',
                1e-3,
            ),
        ]
    else:
        raise ValueError(f'Unknown reference suite: {suite}')
    return [_generate_case(Path(root), device, *case) for case in cases]
