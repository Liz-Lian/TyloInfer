"""Generate independent references for existing Gated DeltaNet cases."""

import hashlib
from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, save_tensor, write_result


def _numpy_recurrence(inputs):
    """Apply the original NumPy FP64 recurrence to the rounded inputs."""
    q, k, v, g, beta, state = (
        inputs[name].astype(np.float64) for name in ('q', 'k', 'v', 'g', 'beta', 'state')
    )
    _, _, tokens, width = q.shape
    out = np.zeros(q.shape)
    for t in range(tokens):
        qt, kt, vt, bt = q[:, :, t], k[:, :, t], v[:, :, t], beta[:, :, t]
        if g.ndim == 4:
            state = state * np.exp(g[:, :, t, :])[..., None]
        else:
            state = state * np.exp(g[:, :, t])[:, :, None, None]
        stk = (state.transpose(0, 1, 3, 2) @ kt[..., None])[..., 0]
        delta = (vt - stk) * bt[..., None]
        state = state + kt[..., None] * delta[..., None, :]
        out[:, :, t] = (state.transpose(0, 1, 3, 2) @ qt[..., None])[..., 0] * (width**-0.5)
    return out.astype(np.float32), state.astype(np.float32)


def _torch_recurrence(inputs):
    """Apply the original Torch FP32 recurrence without calling project kernels."""
    q, k, v, g, beta, state = (inputs[name] for name in ('q', 'k', 'v', 'g', 'beta', 'state'))
    batch, heads, tokens, width = q.shape
    state = state.clone()
    outputs = []
    for t in range(tokens):
        qt, kt, vt, bt = q[:, :, t], k[:, :, t], v[:, :, t], beta[:, :, t]
        if g.dim() == 4:
            state = state * torch.exp(g[:, :, t, :]).unsqueeze(-1)
        else:
            state = state * torch.exp(g[:, :, t]).view(batch, heads, 1, 1)
        stk = (state.transpose(-1, -2) @ kt.unsqueeze(-1)).squeeze(-1)
        delta = (vt - stk) * bt.unsqueeze(-1)
        state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((state.transpose(-1, -2) @ qt.unsqueeze(-1)).squeeze(-1) * (width**-0.5))
    return torch.stack(outputs, dim=2), state


def _device(value, kind):
    """Require the original case's CPU or CUDA execution device."""
    device = torch.device(value)
    if device.type != kind:
        raise ValueError(f'This case requires a {kind} device')
    if kind == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


def reference(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Recompute supported references from the fixed case tensors."""
    source = case.data['source']
    name = source.get('test')
    cpu_test = 'tests/test_kernels_ops.py::test_gdn_matches_reference'
    cuda_tests = {
        'tests/test_kernels_cuda.py::test_gdn_cuda_scalar_gate_matches_ref',
        'tests/test_kernels_cuda.py::test_gdn_cuda_kda_matches_ref',
        'tests/test_kernels_cuda.py::test_gdn_cuda_decode_column_path_matches_ref',
        'tests/test_kernels_cuda.py::test_gdn_cuda_d128_shared_mem',
    }
    if (
        case.data['op'] != 'gated_delta_net'
        or source.get('generator') != 'tests/numerics/gdn_ref.py'
        or name not in cuda_tests | {cpu_test}
    ):
        raise ValueError('Unsupported GDN reference source')
    device = _device(device, 'cpu' if name == cpu_test else 'cuda')
    arrays = {key: case.tensor(key) for key in ('q', 'k', 'v', 'g', 'beta', 'state')}
    if name == cpu_test:
        output, state = _numpy_recurrence(arrays)
        accumulation = 'NumPy float64 recurrence, output rounded to float32'
    else:
        values = {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}
        result, final_state = _torch_recurrence(values)
        output, state = (
            value.detach().contiguous().cpu().numpy() for value in (result, final_state)
        )
        accumulation = 'Torch float32 recurrence'
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {'output': save_tensor(root, 'output', output)}
    if 'state' in case.data['outputs']:
        outputs['state'] = save_tensor(root, 'state', state)
    execution = {
        'framework': 'NumPy' if name == cpu_test else 'PyTorch',
        'numpy_version': np.__version__,
        'torch_version': str(torch.__version__),
        'device': str(device),
        'cuda_version': torch.version.cuda,
        'accumulation': accumulation,
        'comparison_note': (
            'Float32 NumPy and Torch GPU reductions can differ in rounding'
            if name != cpu_test
            else 'Original NumPy FP64 recurrence'
        ),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    if device.type == 'cuda':
        execution.update(
            device_name=torch.cuda.get_device_name(device),
            device_capability=list(torch.cuda.get_device_capability(device)),
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
        )
    return write_result(case, root, outputs, execution)


def generate(root: Path, suite: str = 'cuda', device: str = 'cuda') -> list[Path]:
    """Save the original CPU or CUDA case inputs and generate their references."""
    if suite == 'cpu':
        device = _device(device, 'cpu')
        cases = [('test_gdn_matches_reference', 2, (2, 3, 5, 4), False, False)]
        source_file = 'tests/test_kernels_ops.py'
    elif suite == 'cuda':
        device = _device(device, 'cuda')
        cases = [
            ('test_gdn_cuda_scalar_gate_matches_ref', 0, (2, 4, 32, 32), False, False),
            ('test_gdn_cuda_kda_matches_ref', 0, (1, 2, 16, 32), True, False),
            ('test_gdn_cuda_decode_column_path_matches_ref', 33, (1, 8, 1, 64), False, True),
            ('test_gdn_cuda_d128_shared_mem', 0, (1, 2, 16, 128), False, False),
        ]
        source_file = 'tests/test_kernels_cuda.py'
    else:
        raise ValueError(f'Unknown GDN reference suite: {suite}')
    paths = []
    for name, seed, shape, kda, initial_state in cases:
        batch, heads, tokens, width = shape
        torch.manual_seed(seed)
        q, k, v = (torch.randn(*shape, device=device) for _ in range(3))
        if suite == 'cpu':
            g = torch.randn(batch, heads, tokens, device=device)
            beta = torch.sigmoid(torch.randn(batch, heads, tokens, device=device))
        else:
            q, k, v = q * 0.1, k * 0.1, v * 0.1
            gate_shape = shape if kda else shape[:3]
            g = -(torch.rand(*gate_shape, device=device) * 3 + 0.5)
            beta = torch.sigmoid(torch.randn(batch, heads, tokens, device=device)) * 0.5
        state_shape = (batch, heads, width, width)
        state = (
            torch.randn(*state_shape, device=device) * 0.01
            if initial_state
            else torch.zeros(*state_shape, device=device)
        )
        config = {
            'seed': seed,
            'gate': 'per_dimension' if kda else 'scalar',
            'initial_state': 'random' if initial_state else 'zero',
            'cuda_column_path': initial_state,
            'scale': width**-0.5,
        }
        writer = CaseWriter(
            root,
            name,
            'gated_delta_net',
            config,
            {
                'generator': 'tests/numerics/gdn_ref.py',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'test': f'{source_file}::{name}',
                'formula': 'NumPy float64 recurrence'
                if suite == 'cpu'
                else 'Torch float32 recurrence',
            },
        )
        for key, value in {'q': q, 'k': k, 'v': v, 'g': g, 'beta': beta, 'state': state}.items():
            writer.input(key, value.detach().contiguous().cpu().numpy())
        if suite == 'cpu':
            output_rule = state_rule = {'kind': 'close', 'atol': 1e-4, 'rtol': 1e-3}
        else:
            output_rule = {
                'kind': 'relative_l2',
                'limit': 1e-5,
                'strict': True,
                'precision': 'float32',
            }
            state_rule = {'kind': 'max_abs', 'limit': 1e-5, 'strict': True, 'precision': 'float32'}
        writer.output('output', shape, 'float32', output_rule)
        if suite == 'cpu' or name in {
            'test_gdn_cuda_scalar_gate_matches_ref',
            'test_gdn_cuda_decode_column_path_matches_ref',
        }:
            writer.output('state', state_shape, 'float32', state_rule)
        case = writer.finish()
        reference(case, case.root / 'reference', device)
        paths.append(case.root / 'case.json')
    return paths
