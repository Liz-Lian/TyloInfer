"""Fixed NINT input-gradient references from the external CUDA test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .mfq_source import file_hash, open_source
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings

_FIELDS = ('q', 'neuron_scale', 'neuron_min', 'sub_scale', 'sub_min', 'row_sub_bits', 'row_q_bits')
_RUNTIME = ('q_packed', 'row_q_bits', 'row_q_bit_offsets', 'sub_scale', 'sub_min',
            'neuron_scale', 'neuron_min')


def _rebuild(case, modules):
    """Reconstruct the original in-memory NINT tensor without a lossy roundtrip."""
    nint = modules['mfq.formats.nint']
    fields = {name: case.tensor(name) for name in _FIELDS}
    return nint.NintTensor(
        spec=nint.NintSpec(4, 24, 6), shape=(31, 113), axis=0, neuron_len=113, **fields
    )


def reference(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Recalculate direct and autograd input-gradient expectations from fixed inputs."""
    if (
        case.data['op'] != 'nint_backward_input'
        or case.data['source'].get('generator') != 'tests/numerics/nint_ref.py'
    ):
        raise ValueError('Unsupported NINT backward case')
    modules, identity = open_source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('MFQ source differs from the fixed case')
    tensor = _rebuild(case, modules)
    deployed = modules['mfq.kernels.torch_backend'].nint_deploy_arrays(tensor)
    if any(
        not np.array_equal(value, case.tensor('runtime_' + name))
        for name, value in deployed.items()
    ):
        raise ValueError('Saved NINT runtime arrays differ from the source components')
    device = resolve_device(device)
    with scope(case.data['config']['compute']):
        dense = modules['mfq.quantize.nint_quant'].dequantize(tensor)
        weight = torch.as_tensor(np.ascontiguousarray(dense), device=device, dtype=torch.float16)
        gradient = torch.as_tensor(case.tensor('gradient'), device=device, dtype=torch.float16)
        expected = (gradient @ weight).detach().cpu().numpy()
        target = Path(root)
        outputs = {
            name: save_tensor(target, name, expected)
            for name in ('input_gradient', 'autograd_input_gradient')
        }
        metadata = execution(
            device, 'CPU NumPy NINT decode, FP16 weight and CUDA FP16 Torch matmul'
        )
        metadata.update(
            generator_sha256=file_hash(__file__),
            mfq_source=identity,
            independence='external NumPy dequantize and Torch matrix multiplication; shared format history',
        )
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during reference calculation')
        return write_result(case, target, outputs, metadata)


def candidate(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Run the local NINT wrapper and autograd on verified fixed runtime buffers."""
    if (case.data['op'] != 'nint_backward_input'
            or case.data['source'].get('generator') != 'tests/numerics/nint_ref.py'
            or case.data.get('actual_role') != 'candidate'):
        raise ValueError('Unsupported NINT candidate case')
    from . import nint_cuda as implementation
    from tyloinfer.kernels.cuda._ext import ext

    modules, identity = open_source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('MFQ source differs from the fixed case')
    tensor = _rebuild(case, modules)
    deployed = modules['mfq.kernels.torch_backend'].nint_deploy_arrays(tensor)
    if set(deployed) != set(_RUNTIME) or any(
        not np.array_equal(deployed[name], case.tensor('runtime_' + name)) for name in _RUNTIME
    ):
        raise ValueError('Saved NINT runtime arrays differ from the source components')
    config = case.data['config']
    rows = config['rows']
    if rows not in (1, 6, 9, 32) or config['weight_shape'] != [31, 113] or config['spec'] != [4, 24, 6] or config['axis'] != 0:
        raise ValueError('NINT candidate geometry differs from the original case')
    expected_path = 'direct_packed_backward' if rows <= 8 else 'decode_matmul_fallback'
    if config['external_wrapper_path'] != expected_path:
        raise ValueError('NINT candidate dispatch description differs from the case')
    device = resolve_device(device)
    with scope(config['compute']), torch.cuda.device(device):
        weight = {name: torch.from_numpy(case.tensor('runtime_' + name)).to(device) for name in _RUNTIME}
        weight.update(out=31, ng=int(weight['sub_scale'].shape[1]), gs=24, neuron_len=113)
        gradient = torch.from_numpy(case.tensor('gradient')).to(device)
        source = torch.from_numpy(case.tensor('source')).to(device)
        if (gradient.dtype != torch.float16 or gradient.shape != (rows, 31)
                or source.dtype != torch.float16 or source.shape != (rows, 113)):
            raise ValueError('NINT candidate inputs differ from the original case')
        direct = implementation.nint_backward_input(weight, gradient)
        source.requires_grad_(True)
        (implementation.nint_matmul(weight, source) * gradient).sum().backward()
        outputs = {'input_gradient': direct, 'autograd_input_gradient': source.grad}
        specs = {name: save_tensor(root, name, value.detach().contiguous().cpu().numpy())
                 for name, value in outputs.items()}
        module = ext()
        provenance = execution(device, 'Local NINT wrapper direct call and autograd')
        provenance.update(category='python_candidate', native_worker_executed=False,
                          backward_path=expected_path, mfq_source=identity,
                          wrapper_sha256=file_hash(implementation.__file__),
                          jit_module=str(Path(module.__file__).resolve()),
                          jit_sha256=file_hash(module.__file__), generator_sha256=file_hash(__file__))
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during candidate execution')
        return write_result(case, root, specs, provenance, role='candidate')


def generate(root: Path, source_root: Path, device: str = 'cuda') -> list[Path]:
    """Freeze all four original NINT backward and autograd input-gradient cases."""
    modules, identity = open_source(source_root)
    device = resolve_device(device)
    nint, quant = modules['mfq.formats.nint'], modules['mfq.quantize.nint_quant']
    test = Path(source_root) / 'tests/test_kernels_cuda.py'
    paths = []
    for rows in (1, 6, 9, 32):
        seed = 2600 + rows
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed)
        source_weight = rng.randn(31, 113).astype(np.float32) * np.float32(0.04)
        q_bits = np.resize(np.arange(1, 9, dtype=np.uint8), 31)
        sub_bits = np.resize(np.asarray([5, 6, 7, 8], dtype=np.uint8), 31)
        tensor = quant.quantize(
            source_weight, nint.NintSpec(4, 24, 6), axis=0, row_q_bits=q_bits, row_sub_bits=sub_bits
        )
        deployed = modules['mfq.kernels.torch_backend'].nint_deploy_arrays(tensor)
        gradient = torch.randn(rows, 31, device=device, dtype=torch.float16)
        source = torch.randn(rows, 113, device=device, dtype=torch.float16)
        writer = CaseWriter(
            root,
            f'nint_backward_m{rows}',
            'nint_backward_input',
            {
                'rows': rows,
                'weight_shape': [31, 113],
                'spec': [4, 24, 6],
                'axis': 0,
                'seed': seed,
                'compute': settings(),
                'external_wrapper_path': 'direct_packed_backward'
                if rows <= 8
                else 'decode_matmul_fallback',
                'behavior_observations': {
                    'wrapper_and_autograd': ['input_gradient', 'autograd_input_gradient'],
                },
                'reference_scope': 'Independent expected gradient; candidate records packed or fallback wrapper execution',
            },
            {
                'generator': 'tests/numerics/nint_ref.py',
                'generator_sha256': file_hash(__file__),
                'test': 'tests/test_kernels_cuda.py::test_nint_backward_and_autograd_match_canonical_decode',
                'test_origin': 'external_mfq_checkout',
                'test_sha256': file_hash(test),
                'mfq_source': identity,
            },
            actual_role='candidate',
        )
        writer.input('source_weight', source_weight)
        for name in _FIELDS:
            writer.input(name, getattr(tensor, name))
        for name, value in deployed.items():
            writer.input('runtime_' + name, value)
        writer.input('gradient', gradient.cpu().numpy())
        writer.input('source', source.cpu().numpy())
        for name in ('input_gradient', 'autograd_input_gradient'):
            writer.output(
                name, (rows, 113), 'float16', {'kind': 'close', 'atol': 0.002, 'rtol': 0.003}
            )
        case = writer.finish()
        reference(case, case.root / 'reference', source_root, device)
        candidate(case, case.root / 'candidate', source_root, device)
        paths.append(case.root / 'case.json')
    return paths
