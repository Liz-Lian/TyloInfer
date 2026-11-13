"""Fixed NEPQ input-gradient references using external NumPy decoding."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .case import Case, CaseWriter, save_tensor, write_result
from .mfq_source import file_hash, open_source
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings


def _tables(spec, modules):
    """Rebuild the source fixture's two packed table banks."""
    nepq = modules['mfq.formats.nepq']
    if spec is nepq.NEPQ0_S:
        pack = modules['mfq.formats.npq0_s'].pack_npq0_s_tables
        scale = np.asarray([0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        first = np.arange(4 * 8 * 4, dtype=np.int16).reshape(4, 8, 4)
        second = np.flip(first, axis=1).copy()
    elif spec is nepq.NEPQ0_L:
        pack = modules['mfq.formats.npq0_l'].pack_npq0_l_tables
        scale = np.linspace(0.125, 1.0, 8, dtype=np.float32)
        first = np.arange(8 * 8 * 4, dtype=np.int16).reshape(8, 8, 4)
        second = np.arange(8 * 16 * 4, dtype=np.int16).reshape(8, 16, 4)
    elif spec is nepq.NEPQ1_S:
        nvq = modules['mfq.formats.nvq1_s']
        return np.stack(
            [
                np.frombuffer(
                    nvq.pack_nvq1_s_banked_codebook(np.roll(nvq.NVQ1_S_SYNTHETIC_BANKS, i, axis=1)),
                    dtype=np.uint8,
                )
                for i in range(2)
            ]
        )
    else:
        nvq = modules['mfq.formats.nvq1_l']
        return np.stack(
            [
                np.frombuffer(
                    nvq.pack_ternary_codebook(np.roll(nvq.IQ1S_TERNARY_2048, i, axis=0)),
                    dtype=np.uint8,
                )
                for i in range(2)
            ]
        )
    return np.stack(
        [
            np.frombuffer(
                pack(
                    scale,
                    ((first + i) % 17 - 8).astype(np.int8),
                    ((second + 2 * i) % 19 - 9).astype(np.int8),
                ),
                dtype=np.uint8,
            )
            for i in range(2)
        ]
    )


def _base(spec, modules):
    """Reconstruct the original unrounded NEPQ fixture from its random seed."""
    nepq = modules['mfq.formats.nepq']
    rng = np.random.default_rng(20260723 + spec.profile_id)
    shape = (2, 3, 104)
    groups, vectors = math.ceil(shape[2] / 24), shape[2] // 8
    supergroups = math.ceil(groups / 4)
    selected = (np.arange(shape[0] * shape[1]) % 2).reshape(shape[:2])
    return nepq.NepqTensor(
        spec=spec,
        shape=shape,
        neuron_scale=rng.uniform(0.01, 0.2, size=shape[:2]).astype(np.float32),
        state=rng.integers(0, 1 << spec.state_bits, size=shape[:2] + (groups,), dtype=np.uint8),
        indices=rng.integers(0, 1 << spec.index_bits, size=shape[:2] + (vectors,), dtype=np.uint16),
        aux=(
            rng.integers(0, 2, size=shape[:2] + (groups,), dtype=np.uint8)
            if spec.aux_bits
            else None
        ),
        bank_ids=np.repeat(selected[:, :, None], supergroups, axis=2).astype(np.uint8),
        table_payloads=_tables(spec, modules),
        rotation_block=8,
        rotation_seed=18601311049,
    )


def _fixture(spec, modules):
    """Recreate original dynamic base banks or pre-rounded sparse residual weights."""
    nepq = modules['mfq.formats.nepq']
    if not spec.is_residual:
        tensor = _base(spec, modules)
        for group in range(tensor.bank_ids.shape[-1]):
            tensor.bank_ids[:, :, group] = (
                np.arange(tensor.n_experts * tensor.out_per_expert).reshape(tensor.shape[:2])
                + group
            ) % tensor.bank_count
        return tensor
    base = nepq.NEPQ0_S if spec is nepq.NEPQ0_A else nepq.NEPQ1_S
    tensor = nepq.unpack_nepq(nepq.pack_nepq(_base(base, modules)))
    tensor.spec = spec
    blocks = tensor.residual_blocks_per_row
    tensor.residual_codebook = np.zeros((1024, 8), dtype=np.float16)
    tensor.residual_codebook[1] = np.linspace(-0.25, 0.25, 8, dtype=np.float16)
    tensor.residual_first = np.full(
        tensor.shape[:2] + (blocks,), 1 << spec.residual_position_bits, dtype=np.uint16
    )
    if spec is nepq.NEPQ1_A:
        mask = np.zeros(tensor.shape[:2] + (blocks,), dtype=np.uint8)
        mask.reshape(-1)[::2] = 1
        record = np.uint16(1 | (1 << spec.residual_position_bits))
        tensor.residual_second_mask = mask
        tensor.residual_second_records = np.full(np.count_nonzero(mask), record, dtype=np.uint16)
        tensor.residual_padding_nbytes = 7
    return tensor


_FIELDS = (
    'neuron_scale',
    'state',
    'indices',
    'aux',
    'bank_ids',
    'table_payloads',
    'residual_codebook',
    'residual_first',
    'residual_second_mask',
    'residual_second_records',
)


def _rebuild(case, modules):
    """Reconstruct the exact tensor from separately persisted components."""
    nepq = modules['mfq.formats.nepq']
    config = case.data['config']
    values = {name: case.tensor(name) if name in case.data['inputs'] else None for name in _FIELDS}
    return nepq.NepqTensor(
        spec=nepq.nepq_spec(config['profile']),
        shape=tuple(config['shape']),
        rotation_block=config['rotation_block'],
        rotation_seed=config['rotation_seed'],
        residual_padding_nbytes=config['residual_padding_nbytes'],
        **values,
    )


def _runtime(tensor, modules):
    """Build the original packed GPU buffers from the fixed source components."""
    nepq = modules['mfq.formats.nepq']
    ng, _, nsuper, _, rows = nepq.validate_nepq(tensor)
    base = nepq.nepq_base_spec(tensor.spec)
    tables = []
    for payload in tensor.table_payloads:
        raw = np.ascontiguousarray(payload, dtype=np.uint8).tobytes()
        if base is nepq.NEPQ1_S:
            table = modules['mfq.formats.nvq1_s'].unpack_nvq1_s_banked_codebook(raw)
        elif base is nepq.NEPQ1_L:
            table = modules['mfq.formats.nvq1_l'].unpack_ternary_codebook(raw)
        else:
            table = np.frombuffer(raw, dtype=np.int8)
        tables.append(np.asarray(table, dtype=np.int8).reshape(-1))
    table_pool = np.ascontiguousarray(np.stack(tables), dtype=np.int8)
    if table_pool.shape[1] != tensor.spec.runtime_table_bytes:
        raise ValueError('NEPQ runtime table width differs from source profile')
    aux = np.empty(0, dtype=np.uint8) if tensor.aux is None else np.asarray(tensor.aux)
    result = {
        'indices_packed': np.frombuffer(nepq._pack_bits(tensor.indices, tensor.spec.index_bits), dtype=np.uint8).copy(),
        'aux_packed': np.frombuffer(nepq._pack_bits(aux, tensor.spec.aux_bits), dtype=np.uint8).copy(),
        'state_packed': np.frombuffer(nepq._pack_bits(tensor.state, tensor.spec.state_bits), dtype=np.uint8).copy(),
        'neuron_scale': np.ascontiguousarray(tensor.neuron_scale, dtype=np.float32).reshape(-1),
        'table_pool': table_pool,
        'bank_ids': np.ascontiguousarray(tensor.bank_ids, dtype=np.uint8).reshape(rows, nsuper),
        'rotation_signs': nepq.rotation_signs(tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed),
    }
    if tensor.spec.is_residual:
        blocks = tensor.residual_blocks_per_row
        first = np.asarray(tensor.residual_first, dtype=np.int16).reshape(rows, blocks)
        second = np.full((rows, blocks), -1, dtype=np.int16)
        if tensor.spec.residual_second:
            mask = np.asarray(tensor.residual_second_mask, dtype=np.uint8).reshape(-1)
            records = np.asarray(tensor.residual_second_records, dtype=np.int16).reshape(-1)
            second.reshape(-1)[np.flatnonzero(mask)] = records
        result.update(
            residual_codebook=np.ascontiguousarray(tensor.residual_codebook, dtype=np.float16),
            residual_first=np.ascontiguousarray(first),
            residual_second=second,
        )
    return result, ng


def _verified(case, modules):
    """Verify the saved payload, rotation and deployed buffers against source components."""
    tensor = _rebuild(case, modules)
    nepq = modules['mfq.formats.nepq']
    if nepq.pack_nepq(tensor) != case.asset('packed'):
        raise ValueError('Packed NEPQ payload differs from the saved components')
    runtime, ng = _runtime(tensor, modules)
    for name, expected in runtime.items():
        value = case.tensor('runtime_' + name)
        if value.dtype != expected.dtype or value.shape != expected.shape or not np.array_equal(value, expected):
            raise ValueError(f'NEPQ runtime buffer differs from source components: {name}')
    if ng != case.data['config']['ng'] or not np.array_equal(case.tensor('signs'), runtime['rotation_signs']):
        raise ValueError('NEPQ rotation or group count differs from the case')
    return tensor


def reference(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Recompute a fixed input-gradient case with independent CPU weight decoding."""
    if (
        case.data['op'] != 'nepq_backward_input'
        or case.data['source'].get('generator') != 'tests/numerics/nepq_ref.py'
    ):
        raise ValueError('Unsupported NEPQ reference case')
    modules, identity = open_source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('MFQ source commit or module hashes differ from the case')
    device = resolve_device(device)
    tensor = _verified(case, modules) if 'runtime_indices_packed' in case.data['inputs'] else _rebuild(case, modules)
    config = case.data['config']
    signs = case.tensor('signs')
    nepq = modules['mfq.formats.nepq']
    if nepq.pack_nepq(tensor) != case.asset('packed') or not np.array_equal(
        signs, nepq.rotation_signs(tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed)
    ):
        raise ValueError('Saved NEPQ payload or rotation differs from the components')
    with scope(config['compute']):
        dense = torch.as_tensor(
            nepq.dequantize_nepq(tensor).reshape(-1, tensor.neuron_len),
            device=device,
            dtype=torch.float16,
        )
        gradient = torch.as_tensor(case.tensor('gradient'), device=device, dtype=torch.float16)
        diagonal = torch.as_tensor(signs, device=device, dtype=torch.float16)
        stored = gradient @ dense
        blocks = stored.reshape(-1, tensor.rotation_block).clone()
        stride = 1
        while stride < tensor.rotation_block:
            paired = blocks.reshape(-1, 2, stride)
            first, second = paired[:, 0].clone(), paired[:, 1].clone()
            paired[:, 0], paired[:, 1] = first + second, first - second
            stride *= 2
        scale = (
            np.float32(1.0 / np.sqrt(tensor.rotation_block))
            if tensor.spec.is_residual
            else 1.0 / math.sqrt(tensor.rotation_block)
        )
        expected = (blocks.reshape_as(stored) * scale * diagonal).to(torch.float16)
        target = Path(root)
        value = expected.detach().cpu().numpy()
        specs = {
            name: save_tensor(target, name, value)
            for name in ('input_gradient', 'autograd_input_gradient')
        }
        if 'cache_before_absent' in case.data['outputs']:
            for name in ('cache_before_absent', 'cache_after_expected'):
                specs[name] = save_tensor(target, name, np.asarray(True, dtype=np.bool_))
        metadata = execution(device, 'float16 matmul output and per-stage float16 FWHT arithmetic')
        metadata.update(
            generator_sha256=file_hash(__file__),
            mfq_source=identity,
            independence='external NumPy CPU decode and Torch arithmetic; shared format history',
        )
        if open_source(source_root)[1] != identity:
            raise ValueError('MFQ source changed during reference calculation')
        return write_result(case, target, specs, metadata)


def candidate(case: Case, root: Path, source_root: Path, device: str = 'cuda') -> Path:
    """Run the local NEPQ packed backward and autograd on saved runtime buffers."""
    if (
        case.data['op'] != 'nepq_backward_input'
        or case.data['source'].get('generator') != 'tests/numerics/nepq_ref.py'
        or case.data.get('actual_role') != 'candidate'
    ):
        raise ValueError('Unsupported NEPQ candidate case')
    modules, identity = open_source(source_root)
    if identity != case.data['source']['mfq_source']:
        raise ValueError('NEPQ source identity differs from the case')
    tensor = _verified(case, modules)
    config = case.data['config']
    if (
        config['format'] != tensor.spec.base_format_id
        or config['sub_bits'] != tensor.spec.state_bits
        or config['gs'] != tensor.spec.groupsize
    ):
        raise ValueError('NEPQ runtime geometry differs from the source profile')
    from . import nepq_cuda as adapter
    from tyloinfer.kernels.cuda._ext import ext

    device = resolve_device(device)
    names = (
        'indices_packed', 'aux_packed', 'state_packed', 'neuron_scale',
        'table_pool', 'bank_ids', 'rotation_signs',
    )
    if tensor.spec.is_residual:
        names += ('residual_codebook', 'residual_first', 'residual_second')
    with scope(config['compute']), torch.cuda.device(device):
        weight = {name: torch.from_numpy(case.tensor('runtime_' + name)).to(device) for name in names}
        weight.update(
            rotation_block=tensor.rotation_block, format=config['format'],
            sub_bits=config['sub_bits'], gs=config['gs'], ng=config['ng'],
            rows=tensor.n_experts * tensor.out_per_expert,
            n_experts=tensor.n_experts, out_per_expert=tensor.out_per_expert,
            neuron_len=tensor.neuron_len, shape=tuple(tensor.shape), device=device,
        )
        if tensor.spec.is_residual:
            weight.update(
                residual_position_bits=tensor.spec.residual_position_bits,
                residual_block_vectors=tensor.spec.residual_block_vectors,
            )
        gradient = torch.from_numpy(case.tensor('gradient')).to(device)
        source = torch.from_numpy(case.tensor('source')).to(device)
        if (
            gradient.dtype != torch.float16 or source.dtype != torch.float16
            or gradient.shape != (config['m'], weight['rows'])
            or source.shape != (config['m'], tensor.neuron_len)
        ):
            raise ValueError('NEPQ candidate inputs differ from the case')
        before = '_residual_transpose' not in weight
        direct = adapter.nepq_backward_input(weight, gradient)
        outputs = {'input_gradient': direct}
        if tensor.spec.is_residual:
            cached = weight.get('_residual_transpose')
            if config['m'] <= 8:
                after = (
                    isinstance(cached, tuple) and len(cached) == 3
                    and cached[0].numel() == tensor.neuron_len // 8 + 1
                    and cached[1].numel() == cached[2].numel()
                )
            else:
                after = cached is None
            outputs['cache_before_absent'] = torch.tensor(before, dtype=torch.bool)
            outputs['cache_after_expected'] = torch.tensor(after, dtype=torch.bool)
        source.requires_grad_(True)
        (adapter.nepq_matmul(weight, source).reshape(config['m'], -1) * gradient).sum().backward()
        outputs['autograd_input_gradient'] = source.grad
        specs = {
            name: save_tensor(root, name, value.detach().contiguous().cpu().numpy())
            for name, value in outputs.items()
        }
        module = ext()
        metadata = execution(device, 'NEPQ Python adapter and shared CUDA kernels')
        metadata.update(
            category='python_candidate', entrypoint='nepq_backward_input+nepq_matmul autograd',
            backward_path='residual_dense_fallback' if tensor.spec.is_residual and config['m'] > 8
            else 'packed_backward',
            generator_sha256=file_hash(__file__), candidate_source_sha256=file_hash(adapter.__file__),
            jit_module=str(Path(module.__file__).resolve()), jit_sha256=file_hash(module.__file__),
            mfq_source=identity,
        )
        if open_source(source_root)[1] != identity:
            raise ValueError('NEPQ source changed during candidate execution')
        return write_result(case, root, specs, metadata, role='candidate')


def generate(
    root: Path, source_root: Path, suite: str = 'base', device: str = 'cuda'
) -> list[Path]:
    """Freeze original NEPQ backward cases with explicitly seeded actual gradients.

    ``base`` covers four S/L profiles; ``residual`` covers the two A profiles.
    The seed for the original unseeded Torch inputs is newly chosen and recorded.
    """
    if suite not in ('base', 'residual'):
        raise ValueError(f'Unknown NEPQ suite: {suite!r}')
    modules, identity = open_source(source_root)
    nepq = modules['mfq.formats.nepq']
    profiles = (
        (nepq.NEPQ0_S, nepq.NEPQ0_L, nepq.NEPQ1_S, nepq.NEPQ1_L)
        if suite == 'base'
        else (nepq.NEPQ0_A, nepq.NEPQ1_A)
    )
    sizes = (1, 2, 4, 5, 7, 8, 16) if suite == 'base' else (1, 2, 3, 4, 5, 7, 8, 16)
    device = resolve_device(device)
    source_root = Path(source_root)
    original = Path(__file__).parents[1] / (
        'test_nepq_kernels_cuda.py' if suite == 'base' else 'test_nepq_a_kernels_cuda.py'
    )
    fixtures = [source_root / 'tests/test_formats/test_nepq.py']
    if suite == 'residual':
        fixtures.append(source_root / 'tests/test_formats/test_nepq_a.py')
    paths = []
    for profile in profiles:
        for m in sizes:
            tensor = _fixture(profile, modules)
            runtime, ng = _runtime(tensor, modules)
            seed = 20261112 + profile.profile_id * 100 + m
            torch.manual_seed(seed)
            gradient = torch.randn(
                m, tensor.n_experts * tensor.out_per_expert, device=device, dtype=torch.float16
            )
            source = torch.randn(m, tensor.neuron_len, device=device, dtype=torch.float16)
            signs = nepq.rotation_signs(
                tensor.neuron_len, tensor.rotation_block, tensor.rotation_seed
            )
            config = {
                'profile': profile.label,
                'shape': list(tensor.shape),
                'm': m,
                'ng': ng,
                'gs': profile.groupsize,
                'format': profile.base_format_id,
                'sub_bits': profile.state_bits,
                'rotation_block': tensor.rotation_block,
                'rotation_seed': tensor.rotation_seed,
                'residual_padding_nbytes': tensor.residual_padding_nbytes,
                'compute': settings(),
                'packed_asset': 'serialized FP16 anchors; reference uses saved original components',
                'native_weight': 'components',
                'native_weight_detail': (
                    'base components retain original FP32 anchors'
                    if suite == 'base'
                    else 'components retain original base FP16 pack/unpack roundtrip'
                ),
                'behavior_observations': {
                    'cache_before_absent': 'original cache absent before direct backward',
                    'cache_after_expected': 'original sparse cache shape or dense fallback absence',
                } if suite == 'residual' else {},
            }
            provenance = {
                'generator': 'tests/numerics/nepq_ref.py',
                'generator_sha256': file_hash(__file__),
                'test': original.name,
                'test_sha256': file_hash(original),
                'fixture_sources': {
                    p.relative_to(source_root).as_posix(): file_hash(p) for p in fixtures
                },
                'mfq_source': identity,
                'gradient_seed': seed,
                'gradient_seed_note': 'new fixed seed; the original test did not seed Torch',
                'independence': 'external NumPy decode and Torch formula, no native NEPQ operation',
            }
            name = f'nepq_{profile.label.lower().replace("-", "_")}_m{m}'
            writer = CaseWriter(root, name, 'nepq_backward_input', config, provenance, actual_role='candidate')
            for field in _FIELDS:
                value = getattr(tensor, field)
                if value is not None:
                    writer.input(field, value)
            writer.input('signs', signs)
            for field, value in runtime.items():
                writer.input('runtime_' + field, value)
            writer.input('gradient', gradient.detach().cpu().numpy())
            writer.input('source', source.detach().cpu().numpy())
            writer.asset('packed', nepq.pack_nepq(tensor), encoding='NEPQ')
            tolerance = 0.008 if suite == 'residual' else 0.006
            for output in ('input_gradient', 'autograd_input_gradient'):
                writer.output(
                    output,
                    (m, tensor.neuron_len),
                    'float16',
                    {'kind': 'close', 'atol': tolerance, 'rtol': tolerance},
                )
            if suite == 'residual':
                for output in ('cache_before_absent', 'cache_after_expected'):
                    writer.output(output, (), 'bool', {'kind': 'exact'})
            case = writer.finish()
            reference(case, case.root / 'reference', source_root, device)
            candidate(case, case.root / 'candidate', source_root, device)
            paths.append(case.root / 'case.json')
    return paths
