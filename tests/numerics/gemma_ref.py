"""Freeze Gemma4 materialized CUDA paths as execution-consistency baselines."""

import hashlib
from pathlib import Path

import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings

_ROOT = Path(__file__).resolve().parents[2]
_FILES = (
    'tyloinfer/kernels/cuda/_ext.py',
    'tyloinfer/kernels/cuda/mfq_cuda.cpp',
    'csrc/engine/backends/cuda/kernels/norm.cu',
    'csrc/engine/backends/cuda/kernels/acc.cu',
    'csrc/engine/backends/cuda/kernels/reduce.cuh',
    'csrc/engine/backends/cuda/native/env.h',
    'csrc/engine/backends/cuda/native/tensor/backend.h',
)


def _hash(path):
    """Identify one reference source or the loaded extension binary."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _sources():
    """Pin the JIT compilation sources and the relevant shared headers."""
    from tyloinfer.kernels.cuda import _ext

    files = set(_FILES)
    files.update(Path(path).resolve().relative_to(_ROOT).as_posix() for path in _ext._SOURCES)
    return {name: _hash(_ROOT / name) for name in sorted(files)}


def _array(tensor):
    """Copy a completed tensor to contiguous host storage at its dtype."""
    return tensor.detach().contiguous().cpu().numpy()


def reference(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Recompute the original shared-kernel path from saved inputs."""
    device = resolve_device(device)
    config = case.data['config']
    source = case.data['source']
    if (
        case.data['op'] not in {'gemma_pre_norms', 'gemma_ffn_merge'}
        or source.get('generator') != 'tests/numerics/gemma_ref.py'
        or source.get('kernel_sources_sha256') != _sources()
    ):
        raise ValueError('Unsupported Gemma case or changed shared kernel sources')
    values = {name: torch.from_numpy(case.tensor(name)).to(device) for name in case.data['inputs']}
    with scope(config['compute']):
        from tyloinfer.kernels.cuda import _ext

        jit_sources = {
            str(Path(path).resolve().relative_to(_ROOT)): _hash(path) for path in _ext._SOURCES
        }
        kernel = _ext.ext()
        binary_hash = _hash(kernel.__file__)
        if source.get('jit_binary_sha256', binary_hash) != binary_hash:
            raise ValueError('Loaded JIT module differs from the fixed case')
        eps = config['eps']
        if case.data['op'] == 'gemma_pre_norms':
            attn_post = kernel.rms_norm_f16_cuda(values['attn'], values['weight0'], eps, 0.0)
            residual = kernel.acc_cuda(values['residual'], attn_post)
            dense = kernel.rms_norm_f16_cuda(residual, values['weight1'], eps, 0.0)
            router = kernel.rms_norm_offset_cuda(
                residual.float().contiguous(), values['weight2'], eps, 0.0
            )
            moe = kernel.rms_norm_f16_cuda(residual, values['weight3'], eps, 0.0)
            outputs = {'residual': residual, 'dense': dense, 'router': router, 'moe': moe}
        else:
            dense_post = kernel.rms_norm_f16_cuda(values['dense'], values['weight0'], eps, 0.0)
            moe_post = kernel.rms_norm_f16_cuda(values['moe'], values['weight1'], eps, 0.0)
            combined = (dense_post + moe_post).contiguous()
            post = kernel.rms_norm_f16_cuda(combined, values['weight2'], eps, 0.0)
            outputs = {
                'output': (
                    kernel.acc_cuda(values['residual'], post) * values['layer_scale']
                ).contiguous()
            }
        if _hash(kernel.__file__) != binary_hash or source['kernel_sources_sha256'] != _sources() or any(
            _hash(_ROOT / path) != digest for path, digest in jit_sources.items()
        ):
            raise ValueError('Shared kernel sources changed during computation')
        specs = {name: save_tensor(root, name, _array(value)) for name, value in outputs.items()}
        provenance = execution(
            device, 'Original materialized shared CUDA kernels and Torch operations'
        )
        provenance.update(
            category=config['category'],
            independence='Shared CUDA kernels; this is not an independent numerical reference',
            kernel_sources_sha256=source['kernel_sources_sha256'],
            jit_sources_sha256=jit_sources,
            jit_binary_sha256=binary_hash,
            binary_build_identity='Loaded binary recorded; correspondence to source not verified',
            generator_sha256=_hash(__file__),
        )
        return write_result(case, root, specs, provenance)


def candidate(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Run the original fused path using the case's recorded JIT module."""
    device = resolve_device(device)
    config = case.data['config']
    source = case.data['source']
    if (
        case.data['op'] not in {'gemma_pre_norms', 'gemma_ffn_merge'}
        or source.get('generator') != 'tests/numerics/gemma_ref.py'
        or case.data.get('actual_role') != 'candidate'
        or source.get('kernel_sources_sha256') != _sources()
        or not source.get('jit_binary_sha256')
    ):
        raise ValueError('Unsupported Gemma candidate or changed shared kernel sources')
    values = {name: torch.from_numpy(case.tensor(name)).to(device) for name in case.data['inputs']}
    with scope(config['compute']):
        from tyloinfer.kernels.cuda import _ext

        kernel = _ext.ext()
        binary_hash = _hash(kernel.__file__)
        if binary_hash != source['jit_binary_sha256']:
            raise ValueError('Loaded JIT module differs from the fixed case')
        eps = config['eps']
        if case.data['op'] == 'gemma_pre_norms':
            result = kernel.gemma4_attn_residual_pre_norms_f16_cuda(
                values['residual'], values['attn'], *(values[f'weight{i}'] for i in range(4)), eps
            )
            outputs = dict(zip(('residual', 'dense', 'router', 'moe'), result, strict=True))
        else:
            outputs = {'output': kernel.gemma4_ffn_merge_f16_cuda(
                values['dense'], values['moe'], values['residual'],
                *(values[f'weight{i}'] for i in range(3)), values['layer_scale'], eps
            )}
        if source['kernel_sources_sha256'] != _sources() or _hash(kernel.__file__) != binary_hash:
            raise ValueError('JIT module or shared kernel sources changed during computation')
        specs = {name: save_tensor(root, name, _array(value)) for name, value in outputs.items()}
        provenance = execution(device, 'Original fused CUDA kernel from the same JIT module')
        provenance.update(
            category=config['category'],
            independence='Shared CUDA kernels; this is not an independent numerical reference',
            kernel_sources_sha256=source['kernel_sources_sha256'],
            jit_binary_sha256=binary_hash,
            binary_build_identity='Same loaded module binary as the fixed materialized reference',
            generator_sha256=_hash(__file__),
        )
        return write_result(case, root, specs, provenance, role='candidate')


def generate(root: Path, suite: str = 'norm', device: str = 'cuda') -> list[Path]:
    """Freeze the original two Gemma materialized-path cases for each row count."""
    if suite not in {'norm', 'merge'}:
        raise ValueError(f'Unknown Gemma reference suite: {suite}')
    device = resolve_device(device)
    with scope(settings()):
        from tyloinfer.kernels.cuda import _ext

        binary_hash = _hash(_ext.ext().__file__)
    paths = []
    for rows in (1, 16):
        norm = suite == 'norm'
        seed = (203 if norm else 223) + rows
        test = (
            'test_gemma4_fused_pre_norms_match_materialized_path'
            if norm
            else 'test_gemma4_fused_ffn_merge_matches_materialized_path'
        )
        torch.manual_seed(seed)
        shape = (rows, 2816)
        if norm:
            residual = torch.randn(*shape, device=device, dtype=torch.float16)
            values = {'residual': residual, 'attn': torch.randn_like(residual)}
        else:
            dense = torch.randn(*shape, device=device, dtype=torch.float16)
            values = {
                'dense': dense,
                'moe': torch.randn_like(dense),
                'residual': torch.randn_like(dense),
            }
        for index in range(4 if norm else 3):
            values[f'weight{index}'] = torch.randn(2816, device=device, dtype=torch.float32)
        if not norm:
            values['layer_scale'] = torch.tensor([0.9375], device=device, dtype=torch.float16)
        writer = CaseWriter(
            Path(root),
            f'gemma_{suite}_{rows}',
            'gemma_pre_norms' if norm else 'gemma_ffn_merge',
            {
                'rows': rows,
                'width': 2816,
                'eps': 1e-6,
                'seed': seed,
                'category': 'jit_same_build_execution_consistency',
                'compute': settings(),
            },
            {
                'generator': 'tests/numerics/gemma_ref.py',
                'generator_sha256': _hash(__file__),
                'test': f'tests/test_kernels_cuda.py::{test}',
                'test_sha256': _hash(_ROOT / 'tests/test_kernels_cuda.py'),
                'kernel_sources_sha256': _sources(),
                'jit_binary_sha256': binary_hash,
                'independence': 'Shared CUDA kernels; execution consistency only',
            },
            actual_role='candidate',
        )
        for name, value in values.items():
            writer.input(name, _array(value))
        names = ('residual', 'dense', 'router', 'moe') if norm else ('output',)
        for name in names:
            writer.output(
                name,
                shape,
                'float32' if name == 'router' else 'float16',
                {'kind': 'close', 'atol': 0, 'rtol': 0},
            )
        case = writer.finish()
        reference(case, case.root / 'reference', device)
        paths.append(case.root / 'case.json')
    return paths
