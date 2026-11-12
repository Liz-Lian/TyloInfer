"""Generate fixed independent PyTorch references for existing CUDA cases."""

import hashlib
from pathlib import Path

import torch
from torch.nn import functional

from .case import Case, CaseWriter, save_tensor, write_result
from .torch_env import execution as describe
from .torch_env import scope, settings


def _array(value):
    """Copy a tensor to a contiguous host array without changing its dtype."""
    return value.detach().contiguous().cpu().numpy()


def _execution(device, accumulation):
    """Describe the framework and settings used to calculate the reference."""
    return {
        **describe(device, accumulation),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
    }


def _generate_case(root, device, name, op, seed, tensors, config, formula, tolerance):
    """Save generated inputs, reload them, and calculate one fixed reference."""
    torch.manual_seed(seed)
    writer = CaseWriter(
        root,
        name,
        op,
        {**config, 'compute': settings()},
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
    reference(case, case.root / 'reference', device)
    return case.root / 'case.json'


def _rope(x, pos, rotary_dim, sections):
    """Apply the original split-half Torch RoPE reference formula."""
    tokens, width = x.shape[-2:]
    rotary_dim = width if rotary_dim is None else rotary_dim
    half = rotary_dim // 2
    out = x.clone()
    freqs = 1e6 ** (
        -torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float32) / rotary_dim
    )
    if sections is None:
        pos_used = pos.float().view(1, tokens).expand(half, tokens)
    else:
        axes = torch.empty(half, device=x.device, dtype=torch.long)
        s0, s1, s2 = sections
        axes[:s0] = 0
        axes[s0 : s0 + s1] = 1
        axes[s0 + s1 : s0 + s1 + s2] = 2
        pos2 = pos.float()
        if pos2.dim() == 1:
            pos2 = pos2.view(1, tokens).expand(3, tokens)
        pos_used = pos2[axes]
    angles = pos_used.t() * freqs.view(1, half)
    shape = (1,) * (x.dim() - 2) + (tokens, half)
    cos, sin = angles.cos().view(shape), angles.sin().view(shape)
    x0, x1 = x[..., :half], x[..., half:rotary_dim]
    out[..., :half] = x0 * cos - x1 * sin
    out[..., half:rotary_dim] = x1 * cos + x0 * sin
    return out


def _extra_case(root, device, name, source, op, config, values, formula, tolerance, tail=None):
    """Persist inputs and publish a reference computed from reloaded values."""
    writer = CaseWriter(
        root,
        name,
        op,
        {**config, 'compute': settings()},
        {
            'generator': 'tests/numerics/torch_ref.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'test': f'tests/test_kernels_cuda.py::{source}',
            'formula': formula,
        },
    )
    for key, value in values.items():
        writer.input(key, _array(value))
    key = 'x' if op == 'rope' else 'gate' if op == 'gelu_mul' else 'conv_input'
    shape = list(values[key].shape)
    if op == 'ssm_conv_silu':
        shape[1] = config['n_tokens']
    dtype = values[key].dtype
    dtype_name = 'float16' if dtype == torch.float16 else 'float32'
    writer.output(
        'output', shape, dtype_name, {'kind': 'close', 'atol': tolerance, 'rtol': tolerance}
    )
    if tail is not None:
        writer.output('tail', [*shape[:-1], shape[-1] - tail], dtype_name, {'kind': 'exact'})
    case = writer.finish()
    reference(case, case.root / 'reference', device)
    return case.root / 'case.json'


def reference(case: Case, root: Path, device: str = 'cuda') -> Path:
    """Recalculate a supported reference from its validated, fixed inputs."""
    with scope(case.data['config']['compute']):
        return _reference(case, root, device)


def _reference(case, root, device):
    """Evaluate one supported formula with fixed inputs and selected Torch settings."""
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('These reference cases require a CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    op, config = case.data['op'], case.data['config']
    sources = {
        'rms_norm': {'test_rms_norm_cuda'},
        'l2_norm': {'test_l2_norm_cuda'},
        'acc': {'test_acc_cuda', 'test_acc_cuda_f16_preserves_dtype'},
        'silu_mul': {'test_silu_mul_cuda_f32', 'test_silu_mul_cuda_f16'},
        'rope': {
            'test_rope_cuda_pos0_identity',
            'test_rope_cuda_matches_ref',
            'test_rope_cuda_table_matches_ref',
            'test_rope_cuda_partial_matches_ref',
            'test_rope_cuda_mrope_sections_matches_ref',
        },
        'ssm_conv_silu': {
            'test_ssm_conv_silu_cuda_c1k_matches_torch',
            'test_ssm_conv_silu_cuda_kc_matches_torch',
            'test_ssm_conv_silu_cuda_ck_matches_torch',
            'test_ssm_conv_silu_cuda_bias_matches_torch',
        },
        'gelu_mul': {
            'test_gelu_mul_matches_tanh_approximation',
            'test_gelu_mul_saturates_f16_overflow_without_nonfinite_values',
        },
    }
    source = case.data['source']
    if source.get('generator') != 'tests/numerics/torch_ref.py' or source.get('test') not in {
        f'tests/test_kernels_cuda.py::{name}' for name in sources.get(op, set())
    }:
        raise ValueError('Unsupported reference source or operation')
    if op == 'gelu_mul' and bool(config.get('saturation')) != (
        source['test'].endswith('test_gelu_mul_saturates_f16_overflow_without_nonfinite_values')
    ):
        raise ValueError('GELU saturation configuration does not match its source')
    if op == 'rope' and bool(config.get('identity')) != (
        source['test'].endswith('test_rope_cuda_pos0_identity')
    ):
        raise ValueError('RoPE identity configuration does not match its source')
    inputs = {key: torch.from_numpy(case.tensor(key)).to(device) for key in case.data['inputs']}
    if op == 'rms_norm':
        x, w = inputs['x'], inputs['w']
        var = x.pow(2).mean(-1, keepdim=True)
        output = (x * torch.rsqrt(var + config['eps'])) * w
    elif op == 'l2_norm':
        output = functional.normalize(inputs['x'], dim=-1, eps=config['eps'])
    elif op == 'acc':
        output = inputs['a'] + inputs['b']
    elif op == 'silu_mul':
        gate, up = inputs['gate'], inputs['up']
        output = (
            (functional.silu(gate.float()) * up.float()).to(torch.float16)
            if gate.dtype == torch.float16
            else functional.silu(gate) * up
        )
    elif op == 'rope':
        x, pos = inputs['x'], inputs['pos']
        output = (
            x if config.get('identity') else _rope(x, pos, config['rotary_dim'], config['sections'])
        )
    elif op == 'ssm_conv_silu':
        x, weight = inputs['conv_input'], inputs['weight']
        if weight.dim() == 3:
            kernel, w = weight.shape[-1], weight[:, 0, :]
        elif weight.shape[0] == x.shape[-1]:
            kernel, w = weight.shape[1], weight
        else:
            kernel, w = weight.shape[0], weight.t()
        windows = x[:, : config['n_tokens'] + kernel - 1].unfold(1, kernel, 1)
        summed = (windows * w.view(1, 1, w.size(0), kernel)).sum(dim=-1)
        if 'bias' in inputs:
            summed = summed + inputs['bias'].view(1, 1, w.size(0))
        output = functional.silu(summed)
    elif config.get('saturation'):
        gate = torch.tensor([[200.0, 200.0, -200.0]], device=device, dtype=torch.float16)
        up = torch.tensor([[400.0, -400.0, 400.0]], device=device, dtype=torch.float16)
        if not torch.equal(inputs['gate'], gate) or not torch.equal(inputs['up'], up):
            raise ValueError('Manual saturation baseline requires its original fixed inputs')
        output = torch.tensor([[65504.0, -65504.0, -0.0]], device=device, dtype=torch.float16)
    else:
        output = functional.gelu(inputs['gate'], approximate='tanh') * inputs['up']
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {'output': save_tensor(root, 'output', _array(output))}
    if 'tail' in case.data['outputs']:
        outputs['tail'] = save_tensor(
            root, 'tail', _array(inputs['x'][..., config['rotary_dim'] :])
        )
    accumulation = (
        'float32, output rounded to float16'
        if op == 'silu_mul' and output.dtype == torch.float16
        else str(output.dtype).removeprefix('torch.')
    )
    execution = _execution(device, accumulation)
    execution['generator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return write_result(case, root, outputs, execution)


def _extra_suite(root, device, suite):
    """Generate the original RoPE, convolution, or GELU case group."""
    paths = []
    if suite == 'rope':
        cases = [
            ('test_rope_cuda_pos0_identity', 3, (2, 3, 4, 8), 'zero', None, None, None),
            ('test_rope_cuda_matches_ref', 4, (2, 3, 6, 16), 'float', None, None, None),
            ('test_rope_cuda_table_matches_ref', 41, (2, 3, 6, 16), 'table', None, None, 64),
            ('test_rope_cuda_partial_matches_ref', 31, (2, 3, 4, 10), 'partial', 6, None, None),
            (
                'test_rope_cuda_mrope_sections_matches_ref',
                32,
                (1, 2, 5, 12),
                'sections',
                8,
                (1, 2, 1),
                None,
            ),
        ]
        for name, seed, shape, mode, rotary_dim, sections, table_len in cases:
            torch.manual_seed(seed)
            x = torch.randn(*shape, device=device)
            tokens = shape[-2]
            if mode == 'zero':
                pos = torch.zeros(tokens, device=device)
            elif mode == 'float':
                pos = torch.arange(float(tokens), device=device)
            elif mode == 'table':
                pos = torch.arange(tokens, device=device, dtype=torch.int64) + 5
            elif mode == 'partial':
                pos = torch.arange(float(tokens), device=device) + 2
            else:
                base = torch.arange(tokens, device=device)
                pos = torch.stack([base, base + 10, base + 20], dim=0)
            config = {
                'seed': seed,
                'base': 1e6,
                'rotary_dim': rotary_dim,
                'sections': sections,
                'table_len': table_len,
                'identity': mode == 'zero',
            }
            paths.append(
                _extra_case(
                    root,
                    device,
                    name,
                    name,
                    'rope',
                    config,
                    {'x': x, 'pos': pos},
                    'identity' if mode == 'zero' else '_rope_ref split-half rotation',
                    1e-5,
                    rotary_dim,
                )
            )
    elif suite == 'conv':
        cases = [
            ('test_ssm_conv_silu_cuda_c1k_matches_torch', 29, 2, 7, 65, 4, 'c1k', False),
            ('test_ssm_conv_silu_cuda_kc_matches_torch', 30, 1, 9, 33, 5, 'kc', False),
            ('test_ssm_conv_silu_cuda_ck_matches_torch', 32, 1, 9, 33, 5, 'ck', False),
            ('test_ssm_conv_silu_cuda_bias_matches_torch', 31, 2, 6, 17, 4, 'c1k', True),
        ]
        for name, seed, batch, tokens, channels, kernel, layout, bias in cases:
            torch.manual_seed(seed)
            values = {
                'conv_input': torch.randn(batch, tokens + kernel - 1, channels, device=device)
            }
            weight_shape = {
                'c1k': (channels, 1, kernel),
                'kc': (kernel, channels),
                'ck': (channels, kernel),
            }[layout]
            values['weight'] = torch.randn(*weight_shape, device=device)
            if bias:
                values['bias'] = torch.randn(channels, device=device)
            config = {'seed': seed, 'n_tokens': tokens, 'weight_layout': layout, 'bias': bias}
            paths.append(
                _extra_case(
                    root,
                    device,
                    name,
                    name,
                    'ssm_conv_silu',
                    config,
                    values,
                    'functional.silu((windows * weight).sum(-1) + optional bias)',
                    1e-5,
                )
            )
    else:
        source = 'test_gelu_mul_matches_tanh_approximation'
        for dtype, suffix, tolerance in [
            (torch.float16, 'f16', 1e-3),
            (torch.float32, 'f32', 2e-6),
        ]:
            torch.manual_seed(74)
            gate = torch.randn(33, 71, device=device, dtype=dtype)
            values = {'gate': gate, 'up': torch.randn_like(gate)}
            paths.append(
                _extra_case(
                    root,
                    device,
                    f'{source}_{suffix}',
                    source,
                    'gelu_mul',
                    {'seed': 74, 'approximate': 'tanh'},
                    values,
                    "functional.gelu(gate, approximate='tanh') * up",
                    tolerance,
                )
            )
        name = 'test_gelu_mul_saturates_f16_overflow_without_nonfinite_values'
        values = {
            'gate': torch.tensor([[200.0, 200.0, -200.0]], device=device, dtype=torch.float16),
            'up': torch.tensor([[400.0, -400.0, 400.0]], device=device, dtype=torch.float16),
        }
        paths.append(
            _extra_case(
                root,
                device,
                name,
                name,
                'gelu_mul',
                {'saturation': True},
                values,
                'original manually specified finite saturation baseline',
                0,
            )
        )
    return paths


def generate(root: Path, suite: str = 'rms', device: str = 'cuda') -> list[Path]:
    """Generate the requested cases and return their case metadata paths.

    Suites cover RMSNorm, basic operators, RoPE, convolution, and GELU.
    Inputs are persisted and reloaded before reference calculation.
    """
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('These reference cases require a CUDA device')
    if device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    if suite in {'rope', 'conv', 'gelu'}:
        return _extra_suite(Path(root), device, suite)
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
