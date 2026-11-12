"""Generate fixed TQ cases using CPU decoding and independent Torch arithmetic.

The suites preserve the cases in tests/test_tq_cuda.py. Native graph replay,
guard regions and workspace checks are recorded as separate execution checks;
this module does not execute them.
"""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from tyloinfer.formats.tq import TQTensor

from .case import CaseWriter, save_tensor, write_result

_DTYPES = {'f16': torch.float16, 'bf16': torch.bfloat16, 'f32': torch.float32}
_NAMES = {'f16': 'float16', 'bf16': 'bfloat16', 'f32': 'float32'}
_SOURCE = {
    'generator': 'tests/numerics/tq_ref.py',
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'file': 'tests/test_tq_cuda.py',
    'weight_decode': 'tyloinfer.formats.tq.TQTensor.dequantize',
}


def _fixture(rows, width, columns=False, seed=7, groupsize=512):
    """Build the original mixed-precision packed fixture."""
    rng = np.random.default_rng(seed)
    stored = (width + 255) // 256 * 256
    bits = (rng.permutation(np.arange(rows)) % 8 + 1).astype(np.uint64)
    desc = np.r_[np.uint64(0), np.cumsum(stored * bits[:-1] // 32)]
    desc |= (bits - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(0.015).view(np.uint16)) << np.uint64(40)
    n = (rows * ((stored + groupsize - 1) // groupsize) + 1) // 2
    scales = np.ones(stored, np.float16) if columns else rng.integers(0, 256, n, dtype=np.uint8)
    data = rng.integers(0, 256, int((stored * bits // 8).sum()), dtype=np.uint8)
    return TQTensor(rows, stored, groupsize, columns, desc, scales, data)


def _numpy(tensor):
    """Represent BF16 exactly as expanded float32 for NumPy storage."""
    tensor = tensor.detach().cpu()
    return (tensor.float() if tensor.dtype == torch.bfloat16 else tensor).numpy()


def _input(writer, name, tensor, dtype=None, layout=None):
    """Store a concrete tensor, including its original logical layout."""
    writer.input(name, _numpy(tensor), dtype=dtype, layout=layout or 'contiguous')


def _tensor(case, name, device, dtype=None):
    """Read a persisted input back into Torch on the requested device."""
    return torch.tensor(case.tensor(name), device=device, dtype=dtype)


def _weight(case, name, width, device):
    """Decode a persisted payload through the established CPU reference."""
    return torch.tensor(
        TQTensor.from_bytes(case.asset(name)).dequantize(width), device=device
    ).float()


def _rms(limit):
    """Describe the original strict relative RMS comparison."""
    return {'kind': 'rms', 'limit': limit, 'strict': True, 'precision': 'float32', 'floor': 1e-20}


def _writer(root, name, op, config, test):
    """Create a case with explicit execution checks and original source."""
    return CaseWriter(root, name, op, config, {**_SOURCE, 'test': test})


def _result(case, outputs, device):
    """Save float32 reference outputs and execution provenance."""
    target = case.root / 'reference'
    specs = {name: save_tensor(target, name, _numpy(value)) for name, value in outputs.items()}
    execution = {
        'framework': 'torch',
        'torch_version': str(torch.__version__),
        'cuda_version': torch.version.cuda,
        'numpy_version': np.__version__,
        'device': str(device),
        'device_name': torch.cuda.get_device_name(device),
        'device_capability': list(torch.cuda.get_device_capability(device)),
        'allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'accumulation': 'Torch float32 matmul/einsum and reduction with recorded backend settings',
        'independence': 'CPU TQTensor.dequantize and Torch arithmetic are independent of native CUDA TQ kernels; CPU decoding is shared production code',
    }
    write_result(case, target, specs, execution)
    return case.root / 'case.json'


def _asset(writer, name, tensor):
    """Persist the exact encoded weight payload."""
    writer.asset(name, tensor.to_bytes(), encoding='RTRELL05')


def _random(shape, dtype, device, scale=1.0):
    """Create an input with the same dtype and arithmetic as the source test."""
    return torch.randn(shape, device=device, dtype=dtype) * scale


def _output(writer, name, shape, rule, actual_dtype):
    """Declare a float32 reference observation and its original comparison."""
    writer.output(name, list(shape), 'float32', rule, actual_dtype=actual_dtype)


def _gemv(root, device, boundary=False):
    """Generate GEMV and subgroup boundary cases."""
    results = []
    geometries = (
        [(24, 273, False, gs) for gs in (16, 32, 256, 512)]
        if boundary
        else [
            (19, 257, False, 512),
            (7, 513, False, 512),
            (1, 37, True, 512),
            (273, 273, False, 512),
            (2048, 4096, False, 512),
            (4096, 2048, True, 512),
        ]
    )
    for (rows, width, columns, gs), (label, dtype) in itertools.product(
        geometries, _DTYPES.items()
    ):
        name = (
            f'row_{rows}_{width}_g{gs}_{label}'
            if boundary
            else f'gemv_{rows}_{width}_c{int(columns)}_{label}'
        )
        limit = (
            (0.004 if label == 'bf16' else 0.001)
            if boundary
            else (0.001 if label == 'f16' else 0.01)
        )
        config = {
            'width': width,
            'actual_dtype': _NAMES[label],
            'groupsize': gs,
            'columns': columns,
            'native_checks': ['guard_42'] if not boundary else [],
            'warps_rows_per_block': [[0, 0], [1, 4], [2, 2], [4, 1], [8, 1], [16, 1], [32, 1]]
            if not boundary
            else [[0, 0]],
        }
        w = _writer(
            root, name, 'tq_gemv', config, 'test_row_scale_boundaries' if boundary else 'test_gemv'
        )
        _asset(w, 'weight', _fixture(rows, width, columns, groupsize=gs))
        x = _random((width + 1,), dtype, device)[1:]
        _input(w, 'x', x, _NAMES[label], {'strides': [1], 'offset': 1})
        if not boundary:
            _input(w, 'bias', torch.linspace(-0.02, 0.02, rows, device=device))
        _output(w, 'output', (rows,), _rms(limit), _NAMES[label])
        case = w.finish()
        value = _weight(case, 'weight', width, device) @ _tensor(case, 'x', device, dtype).float()
        if not boundary:
            value = value + _tensor(case, 'bias', device)
        results.append(_result(case, {'output': value}, device))
    return results


def _prefill(root, device):
    """Generate original grouped float prefill cases."""
    results = []
    for (label, dtype), (groups, columns) in itertools.product(
        [('bf16', torch.bfloat16), ('f32', torch.float32)], [(1, False), (3, True)]
    ):
        width, tokens, rows = 273, 257, groups * 19
        w = _writer(
            root,
            f'prefill_g{groups}_{label}',
            'tq_float_prefill',
            {
                'width': width,
                'groups': groups,
                'tokens': tokens,
                'actual_dtype': _NAMES[label],
                'workspace_bytes': 128 << 10,
                'native_checks': ['guard_42', 'graph_replay'],
            },
            'test_float_prefill_exact_source_precision_and_bounded_workspace',
        )
        _asset(w, 'weight', _fixture(rows, width, columns, seed=94, groupsize=16))
        _input(w, 'x', _random((tokens, groups, width), dtype, device, 0.2), _NAMES[label])
        _input(w, 'bias', torch.linspace(-0.01, 0.01, rows, device=device))
        _output(
            w, 'output', (tokens, rows), _rms(0.004 if label == 'bf16' else 3e-5), _NAMES[label]
        )
        case = w.finish()
        weights = _weight(case, 'weight', width, device).reshape(groups, 19, width)
        x = _tensor(case, 'x', device, dtype).float()
        value = torch.einsum('tgk,grk->tgr', x, weights).reshape(tokens, rows) + _tensor(
            case, 'bias', device
        )
        results.append(_result(case, {'output': value}, device))
    return results


def _q8(case, asset, x, width, device, groups=1):
    """Apply the original per-16 input quantization and float matrix formula."""
    tensor = TQTensor.from_bytes(case.asset(asset))
    matrix = torch.tensor(tensor.dequantize(width), device=device).float()
    values = x.float()
    if tensor.columns:
        scales = torch.tensor(tensor.scales[:width].astype(np.float32), device=device)
        values = values * scales
        matrix = torch.where(scales != 0, matrix / scales, 0)
    padded = functional.pad(values, (0, -width % 16)).reshape(*values.shape[:-1], -1, 16)
    maximum = padded.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127, 1)
    decoded = ((padded / scale).round().clamp(-127, 127) * scale).flatten(-2)[..., :width]
    if groups == 1:
        return decoded @ matrix.T
    return torch.einsum('tgk,grk->tgr', decoded, matrix.reshape(groups, -1, width)).flatten(-2)


def _q8_cases(root, device, parallel=False):
    """Generate Q8 single and parallel projection parameter combinations."""
    results = []
    dtypes = [('f16', torch.float16), ('f32', torch.float32)]
    variants = (
        list(itertools.product((0, 2), (1, 11), (False, True)))
        if parallel
        else [(1, False), (3, True)]
    )
    for (label, dtype), tokens, variant in itertools.product(dtypes, (1, 2, 3, 4), variants):
        width = 273
        if parallel:
            joined, copies, column_scaled = variant
            tensors = [
                _fixture(rows, width, columns=column_scaled and i == 1, groupsize=16, seed=71 + i)
                for i, rows in enumerate((1, 19, 65) * copies)
            ]
            rows, groups = sum(t.rows for t in tensors), 1
            name = f'q8p_t{tokens}_j{joined}_n{copies}_c{int(column_scaled)}_{label}'
            slots = len(tensors) + 1 if column_scaled else 1
        else:
            groups, columns = variant
            rows, joined, slots = groups * 19, 0, groups
            tensors = [_fixture(rows, width, columns, groupsize=16)]
            name = f'q8_t{tokens}_g{groups}_{label}'
        workspace = slots * tokens * ((width + 511) // 512 * 512) * 3 // 2
        checks = ['guard_42', 'graph_changed_input', 'workspace_minus_one_rejected']
        if not parallel:
            checks.append('graph_zero_input')
        w = _writer(
            root,
            name,
            'tq_q8_parallel' if parallel else 'tq_q8',
            {
                'width': width,
                'groups': groups,
                'tokens': tokens,
                'actual_dtype': _NAMES[label],
                'joined': joined,
                'joined_columns': sum(t.rows for t in tensors[:joined]),
                'projection_rows': [t.rows for t in tensors],
                'workspace_slots': slots,
                'workspace_bytes': workspace,
                'native_checks': checks,
            },
            'test_q8_parallel_mixed_scales_and_changed_graph'
            if parallel
            else 'test_q8_projection_reference_guards_and_changed_graph',
        )
        for i, tensor in enumerate(tensors):
            _asset(w, f'weight{i}', tensor)
        x = (
            _random((tokens, width), dtype, device, 0.2)
            if parallel
            else _random((tokens * groups * width + 1,), dtype, device)[1:].reshape(
                tokens, groups, width
            )
        )
        layout = None if parallel else {'strides': [groups * width, width, 1], 'offset': 1}
        _input(w, 'x', x, _NAMES[label], layout)
        _input(w, 'x_changed', x * -0.75, _NAMES[label], layout)
        if not parallel:
            _input(w, 'x_zero', torch.zeros_like(x), _NAMES[label], layout)
            _input(w, 'bias', torch.linspace(-0.01, 0.01, rows, device=device))
        shape = (tokens * rows,) if parallel else (tokens, rows)
        for observation in ('output', 'changed') if parallel else ('output', 'changed', 'zero'):
            _output(w, observation, shape, _rms(0.001), _NAMES[label])
        case = w.finish()
        outputs = {}
        names = [('output', 'x'), ('changed', 'x_changed')]
        if not parallel:
            names.append(('zero', 'x_zero'))
        for out, inp in names:
            value = _tensor(case, inp, device, dtype)
            parts = [
                _q8(case, f'weight{i}', value, width, device, groups) for i in range(len(tensors))
            ]
            if parallel:
                if joined:
                    parts = [torch.cat(parts[:joined], -1), *parts[joined:]]
                outputs[out] = torch.cat([part.ravel() for part in parts])
            else:
                outputs[out] = parts[0].reshape(tokens, rows) + _tensor(case, 'bias', device)
        results.append(_result(case, outputs, device))
    return results


def _ffn(root, device):
    """Generate the source SwiGLU and FFN observations."""
    results = []
    for (width, hidden, output), (label, dtype) in itertools.product(
        [(257, 19, 37), (4096, 2048, 4096)], _DTYPES.items()
    ):
        w = _writer(
            root,
            f'ffn_{width}_{hidden}_{output}_{label}',
            'tq_ffn',
            {
                'width': width,
                'hidden': hidden,
                'actual_dtype': _NAMES[label],
                'native_checks': ['graph_replay', 'stream_reuse'],
            },
            'test_ffn_graph',
        )
        for name, tensor in [
            ('gate', _fixture(hidden, width, False, 11)),
            ('up', _fixture(hidden, width, True, 22)),
            ('down', _fixture(output, hidden, True, 33)),
        ]:
            _asset(w, name, tensor)
        _input(w, 'x', _random((width,), dtype, device, 0.2), _NAMES[label])
        _output(w, 'workspace', (hidden,), _rms(0.01), _NAMES[label])
        _output(w, 'output', (output,), _rms(0.015), _NAMES[label])
        case = w.finish()
        x = _tensor(case, 'x', device, dtype).float()
        gate = (_weight(case, 'gate', width, device) @ x).to(dtype).float()
        up = (_weight(case, 'up', width, device) @ x).to(dtype).float()
        hidden_value = (up * functional.silu(gate)).to(dtype)
        output_value = _weight(case, 'down', hidden, device) @ hidden_value.float()
        results.append(
            _result(case, {'workspace': hidden_value.float(), 'output': output_value}, device)
        )
    return results


def _embedding(root, device):
    """Generate exact mixed-table row selection observations."""
    results = []
    width = 273
    for label, dtype in _DTYPES.items():
        w = _writer(
            root,
            f'embedding_{label}',
            'tq_embedding',
            {
                'width': width,
                'actual_dtype': _NAMES[label],
                'native_checks': ['guard_42', 'graph_changed_table_views'],
            },
            'test_embedding_row_views_replay_mixed_tables',
        )
        for i in range(3):
            _asset(
                w,
                f'weight{i}',
                _fixture(1, width, columns=i % 2 == 1, groupsize=16 << i, seed=1842 + i),
            )
        _input(w, 'ids', torch.tensor([0, 2, 0], dtype=torch.int64))
        _input(w, 'ids_changed', torch.tensor([2, 1, 2], dtype=torch.int64))
        _output(w, 'output', (3 * width,), {'kind': 'exact'}, _NAMES[label])
        _output(w, 'changed', (3 * width,), {'kind': 'exact'}, _NAMES[label])
        case = w.finish()
        outputs = {}
        for name, ids_name in [('output', 'ids'), ('changed', 'ids_changed')]:
            ids = case.tensor(ids_name).tolist()
            values = [_weight(case, f'weight{i}', width, device).reshape(-1) for i in ids]
            outputs[name] = torch.cat(values).to(dtype).float()
        results.append(_result(case, outputs, device))
    return results


def _routed(root, device):
    """Generate expert selection, SwiGLU and weighted reduction references."""
    results = []
    width, hidden, output, routes = 273, 19, 33, 3
    for label, dtype in _DTYPES.items():
        w = _writer(
            root,
            f'routed_{label}',
            'tq_routed_ffn',
            {
                'width': width,
                'hidden': hidden,
                'output': output,
                'routes': routes,
                'limit': 0.1,
                'actual_dtype': _NAMES[label],
                'native_checks': ['guard_42', 'graph_changed_expert_ids'],
            },
            'test_routed_view_tables_reuse_graph_with_new_expert_ids',
        )
        for role, rows, k in [
            ('gate', hidden, width),
            ('up', hidden, width),
            ('down', output, hidden),
        ]:
            for i in range(3):
                _asset(
                    w, f'{role}{i}', _fixture(rows, k, columns=i == 1, seed=131 + i, groupsize=16)
                )
        _input(w, 'x', _random((width,), dtype, device, 0.2), _NAMES[label])
        _input(w, 'ids', torch.tensor([1, 0, 1], dtype=torch.int64))
        _input(w, 'ids_changed', torch.tensor([2, 1, 2], dtype=torch.int64))
        _input(w, 'weights', torch.tensor([0.2, 0.3, 0.5], dtype=torch.float32))
        for name in ('output', 'changed'):
            _output(w, name, (output,), _rms(0.015 if label == 'bf16' else 0.004), _NAMES[label])
        case = w.finish()
        x = _tensor(case, 'x', device, dtype).float()
        weights = _tensor(case, 'weights', device)
        outputs = {}
        for name, ids_name in [('output', 'ids'), ('changed', 'ids_changed')]:
            values = []
            for expert in case.tensor(ids_name).tolist():
                gate = (
                    (_weight(case, f'gate{expert}', width, device) @ x)
                    .to(dtype)
                    .float()
                    .clamp_max(0.1)
                )
                up = (
                    (_weight(case, f'up{expert}', width, device) @ x)
                    .to(dtype)
                    .float()
                    .clamp(-0.1, 0.1)
                )
                h = (functional.silu(gate) * up).to(dtype).float()
                values.append(
                    (_weight(case, f'down{expert}', hidden, device) @ h).to(dtype).float()
                )
            outputs[name] = (torch.stack(values) * weights[:, None]).sum(0).to(dtype).float()
        results.append(_result(case, outputs, device))
    return results


def generate(root: Path, suite: str = 'gemv', device: str = 'cuda') -> list[Path]:
    """Generate one TQ reference suite from fixed serialized inputs and weights.

    Suites are ``gemv``, ``row_scale``, ``float_prefill``, ``q8``,
    ``q8_parallel``, ``ffn``, ``embedding`` and ``routed``. The returned paths
    identify case manifests; running a suite does not run native
    kernels or its recorded graph, guard, and workspace checks.
    """
    suites = (
        'gemv',
        'row_scale',
        'float_prefill',
        'q8',
        'q8_parallel',
        'ffn',
        'embedding',
        'routed',
    )
    if suite not in suites:
        raise ValueError(f'Unknown TQ suite {suite!r}; choose one of {", ".join(suites)}')
    if torch.device(device).type != 'cuda':
        raise ValueError('TQ reference migration requires a CUDA device')
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if suite in ('gemv', 'ffn'):
            torch.backends.cuda.matmul.allow_tf32 = False
        if suite == 'gemv':
            return _gemv(root, device)
        if suite == 'row_scale':
            return _gemv(root, device, boundary=True)
        if suite == 'float_prefill':
            return _prefill(root, device)
        if suite == 'q8':
            return _q8_cases(root, device)
        if suite == 'q8_parallel':
            return _q8_cases(root, device, parallel=True)
        if suite == 'ffn':
            return _ffn(root, device)
        if suite == 'embedding':
            return _embedding(root, device)
        return _routed(root, device)
    finally:
        if suite in ('gemv', 'ffn'):
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
