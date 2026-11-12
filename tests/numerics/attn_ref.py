"""Independent Torch attention and KV references for existing CUDA tests."""

import hashlib
from pathlib import Path

import torch
from torch.nn import functional

from .case import Case, CaseWriter, save_tensor, write_result
from .torch_env import device as resolve_device
from .torch_env import execution, scope, settings


def _array(value):
    """Copy a tensor to host without changing its dtype."""
    return value.detach().contiguous().cpu().numpy()


def _mask(q, k, window=None):
    """Build the source causal-offset or sliding-window attention mask."""
    tq, tk = q.size(2), k.size(2)
    result = torch.full((tq, tk), float('-inf'), device=q.device, dtype=q.dtype)
    for row in range(tq):
        end = row + tk - tq + 1
        start = 0 if window is None else max(0, end - window)
        result[row, start:end] = 0
    return result.view(1, 1, tq, tk)


def reference(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Recompute supported references exclusively from persisted inputs."""
    device = resolve_device(device)
    with scope(case.data['config']['compute']):
        return _reference(case, root, device)


def _reference(case, root, device):
    """Calculate saved observations under the recorded Torch settings."""
    config, op = case.data['config'], case.data['op']
    if case.data['source'].get('generator') != 'tests/numerics/attn_ref.py':
        raise ValueError('Unsupported attention reference source')
    values = {name: torch.from_numpy(case.tensor(name)).to(device) for name in case.data['inputs']}
    if op == 'kv_write':
        key, value = values['initial_k'].clone(), values['initial_v'].clone()
        pos = values['positions']
        if pos.dim() == 1:
            key[:, :, pos.cpu().tolist(), :] = values['k']
            value[:, :, pos.cpu().tolist(), :] = values['v']
        else:
            for batch in range(pos.shape[0]):
                for token in range(pos.shape[1]):
                    p = int(pos[batch, token].item())
                    key[batch, :, p, :] = values['k'][batch, :, token, :]
                    value[batch, :, p, :] = values['v'][batch, :, token, :]
        outputs = {'k': key, 'v': value}
    elif op == 'kv_grow':
        outputs = {'k': values['k'].to(torch.float16), 'v': values['v'].to(torch.float16)}
    elif op == 'attention':
        q = values['q']
        if config.get('append'):
            k = torch.cat([values[name] for name in config['append_k']], dim=2)
            v = torch.cat([values[name] for name in config['append_v']], dim=2)
        else:
            k, v = values['k'], values['v']
        repeat = q.size(1) // k.size(1)
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
        outputs = {
            'output': functional.scaled_dot_product_attention(
                q, k, v, attn_mask=values.get('mask'), is_causal=config['reference_is_causal']
            )
        }
    else:
        raise ValueError(f'Unsupported operation: {op}')
    root = Path(root)
    specs = {name: save_tensor(root, name, _array(value)) for name, value in outputs.items()}
    provenance = execution(device, 'Torch SDPA automatic backend dispatch or tensor assignment')
    provenance.update(
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sdpa_backend='automatic; available backends are not fixed',
        flash_sdp_enabled=torch.backends.cuda.flash_sdp_enabled(),
        math_sdp_enabled=torch.backends.cuda.math_sdp_enabled(),
        mem_efficient_sdp_enabled=torch.backends.cuda.mem_efficient_sdp_enabled(),
    )
    return write_result(case, root, specs, provenance)


def _case(root, device, name, op, config, values, outputs):
    """Publish a fixed case and calculate its reference by reading it back."""
    source_path = Path(__file__).parents[1] / 'test_kernels_cuda.py'
    writer = CaseWriter(
        root,
        name,
        op,
        {**config, 'compute': settings()},
        {
            'generator': 'tests/numerics/attn_ref.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'test': f'tests/test_kernels_cuda.py::{name}',
            'test_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
            'formula': 'Torch SDPA' if op == 'attention' else 'Torch tensor assignment or cast',
        },
    )
    for key, value in values.items():
        writer.input(key, _array(value))
    for key, shape, dtype, rule in outputs:
        writer.output(key, shape, dtype, rule)
    case = writer.finish()
    reference(case, case.root / 'reference', device)
    return case.root / 'case.json'


def _rule(tolerance):
    """Represent the original element-wise tolerance."""
    return {'kind': 'close', 'atol': tolerance, 'rtol': tolerance}


def _attention_cases(root, device):
    """Freeze the original direct and circular-cache attention examples."""
    cases = [
        ('test_attention_cuda_causal', 5, (2, 4, 8, 16), (2, 4, 8, 16), 'f32', 'causal', None),
        ('test_attention_cuda_gqa', 6, (1, 8, 6, 16), (1, 2, 6, 16), 'f32', 'none', None),
        (
            'test_attention_cuda_decode_unequal',
            7,
            (1, 4, 2, 16),
            (1, 2, 4, 16),
            'f32',
            'offset',
            None,
        ),
        (
            'test_attention_cuda_splitk_decode_matches_sdpa',
            71,
            (1, 8, 1, 64),
            (1, 2, 1024, 64),
            'f16',
            'none',
            None,
        ),
        (
            'test_attention_swa_gqa_unequal_lengths_matches_sdpa',
            72,
            (1, 8, 7, 64),
            (1, 2, 19, 64),
            'f16',
            'window',
            5,
        ),
        (
            'test_attention_swa_circular_cache_wrap_matches_sdpa',
            73,
            (1, 8, 3, 64),
            (1, 2, 19, 64),
            'f16',
            'ring',
            5,
        ),
        (
            'test_attention_swa_circular_cache_splitk_matches_sdpa',
            76,
            (1, 8, 3, 64),
            (1, 2, 900, 64),
            'f16',
            'ring',
            512,
        ),
    ]
    paths = []
    for name, seed, qshape, kshape, kind, mode, window in cases:
        dtype = torch.float16 if kind == 'f16' else torch.float32
        torch.manual_seed(seed)
        if mode != 'ring':
            q = torch.randn(*qshape, device=device, dtype=dtype)
        k = torch.randn(*kshape, device=device, dtype=dtype)
        v = torch.randn_like(k)
        if mode == 'ring':
            q = torch.randn(*qshape, device=device, dtype=dtype)
        values = {'q': q, 'k': k, 'v': v}
        config = {
            'seed': seed,
            'reference_is_causal': mode == 'causal',
            'native_causal': mode in {'causal', 'offset'} or seed == 71,
            'window': window,
            'mode': mode,
        }
        if mode in {'offset', 'window', 'ring'}:
            values['mask'] = _mask(q, k, window)
        if mode == 'ring':
            capacity = 8 if seed == 73 else 768
            values['seq_len'] = torch.tensor([kshape[2]], dtype=torch.int64, device=device)
            values['initial_k'] = torch.zeros(
                kshape[0], kshape[1], capacity, kshape[3], dtype=dtype, device=device
            )
            values['initial_v'] = torch.zeros_like(values['initial_k'])
            config.update(
                capacity=capacity,
                initial_state='chosen_zero_original_empty_fully_overwritten',
                ring_writes=[[0, 8], [8, 16], [16, 19]] if seed == 73 else [[0, 900]],
            )
        tolerance = 1e-3 if kind == 'f16' else 2e-4
        paths.append(
            _case(
                root,
                device,
                name,
                'attention',
                config,
                values,
                [('output', qshape, 'float16' if kind == 'f16' else 'float32', _rule(tolerance))],
            )
        )
    return paths


def _kv_cases(root, device):
    """Freeze indexed writes, append-attention and cache growth examples."""
    paths = []
    for name, seed, shape, maximum, positions in [
        ('test_kv_cache_write_matches_assignment', 23, (2, 3, 4, 8), 11, [1, 3, 7, 9]),
        ('test_kv_cache_write_batch_positions', 24, (2, 2, 3, 8), 9, [[0, 2, 4], [1, 3, 5]]),
    ]:
        torch.manual_seed(seed)
        k = torch.randn(*shape, device=device)
        v = torch.randn(*shape, device=device)
        cache_shape = (shape[0], shape[1], maximum, shape[3])
        values = {
            'k': k,
            'v': v,
            'positions': torch.tensor(positions, dtype=torch.int64, device=device),
            'initial_k': torch.zeros(*cache_shape, device=device),
            'initial_v': torch.zeros(*cache_shape, device=device),
        }
        paths.append(
            _case(
                root,
                device,
                name,
                'kv_write',
                {'seed': seed},
                values,
                [('k', cache_shape, 'float32', _rule(0)), ('v', cache_shape, 'float32', _rule(0))],
            )
        )
    for seed, parts in [(25, (3, 2)), (251, (5,))]:
        name = (
            'test_kv_cache_append_attention_matches_sdpa'
            if seed == 25
            else 'test_kv_cache_default_f16_and_attention_matches_sdpa'
        )
        dtype = torch.float32 if seed == 25 else torch.float16
        torch.manual_seed(seed)
        values = {}
        keys, vals = [], []
        for i, length in enumerate(parts):
            keys.append(f'k{i}')
            vals.append(f'v{i}')
            values[keys[-1]] = torch.randn(1, 2, length, 16, device=device, dtype=dtype)
            values[vals[-1]] = torch.randn(1, 2, length, 16, device=device, dtype=dtype)
        q = torch.randn(1, 4, 2, 16, device=device, dtype=dtype)
        values['q'] = q
        values['mask'] = _mask(q, torch.cat([values[key] for key in keys], dim=2))
        values['initial_k'] = torch.zeros(1, 2, 8, 16, device=device, dtype=dtype)
        values['initial_v'] = torch.zeros_like(values['initial_k'])
        config = {
            'seed': seed,
            'append': True,
            'append_k': keys,
            'append_v': vals,
            'capacity': 8,
            'initial_valid_length': 0,
            'native_causal': True,
            'initial_state': 'chosen_zero_original_empty_unread',
            'reference_is_causal': False,
        }
        paths.append(
            _case(
                root,
                device,
                name,
                'attention',
                config,
                values,
                [
                    (
                        'output',
                        q.shape,
                        'float32' if seed == 25 else 'float16',
                        _rule(2e-4 if seed == 25 else 1e-3),
                    )
                ],
            )
        )
    torch.manual_seed(350)
    shape = (1, 2, 5, 8)
    values = {'k': torch.randn(*shape, device=device), 'v': torch.randn(*shape, device=device)}
    values['initial_k'] = torch.zeros(1, 2, 2, 8, device=device, dtype=torch.float16)
    values['initial_v'] = torch.zeros_like(values['initial_k'])
    paths.append(
        _case(
            root,
            device,
            'test_kv_cache_grows_without_full_prealloc',
            'kv_grow',
            {
                'input_seed': 350,
                'seed_origin': 'chosen; source was unseeded',
                'initial_capacity': 2,
                'expected_capacity': 8,
                'max_seq': 1024,
                'initial_valid_length': 0,
                'initial_state': 'chosen_zero_original_empty_unread',
                'native_checks': ['capacity_and_shape'],
            },
            values,
            [('k', shape, 'float16', _rule(0)), ('v', shape, 'float16', _rule(0))],
        )
    )
    return paths


def generate(root: Path, suite: str = 'attention', device: str = 'cuda') -> list[Path]:
    """Persist the requested original attention or KV cases and their references."""
    device = resolve_device(device)
    if suite == 'attention':
        return _attention_cases(Path(root), device)
    if suite == 'kv':
        return _kv_cases(Path(root), device)
    raise ValueError(f'Unknown attention suite: {suite}')
