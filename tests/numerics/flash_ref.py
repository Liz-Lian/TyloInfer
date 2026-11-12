"""Fixed Flash-Next references from the source tests' NumPy equations."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .case import Case, CaseWriter, save_tensor, write_result


def _sigmoid(x):
    """Evaluate the source sigmoid formula."""
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x, axis=-1):
    """Evaluate the stable source softmax formula."""
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _rounded(x, dtype):
    """Match Torch's nearest-even float16 and bfloat16 input conversion."""
    if dtype == 'float16':
        return x.astype(np.float16).astype(np.float32)
    if dtype == 'bfloat16':
        bits = np.asarray(x, dtype=np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(
            np.float32
        )
    return x


def _layout(shape, noncontiguous):
    """Describe the source transpose-contiguous-transpose logical strides."""
    if not noncontiguous or len(shape) < 2:
        return 'contiguous'
    strides = [0] * len(shape)
    strides[-2], strides[-1] = 1, shape[-2]
    stride = shape[-2] * shape[-1]
    for i in range(len(shape) - 3, -1, -1):
        strides[i], stride = stride, stride * shape[i]
    return {'strides': strides, 'offset': 0}


def _start(root, name, op, config, inputs, outputs, dtype='float32', noncontiguous=False):
    """Persist source inputs and publish a case before calculating outputs."""
    source = Path(__file__).parents[1] / 'test_cuda_flash_next.py'
    config = {
        **config,
        'compute': {
            'framework': 'NumPy',
            'device': 'cpu',
            'input_rounding': dtype,
            'accumulation': 'source NumPy float32 formulas',
        },
    }
    writer = CaseWriter(
        root,
        name,
        op,
        config,
        {
            'file': 'tests/test_cuda_flash_next.py',
            'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'generator': 'tests/numerics/flash_ref.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'formula': 'existing independent NumPy reference',
        },
    )
    for key, value in inputs.items():
        array = np.asarray(value)
        logical = 'int64' if np.issubdtype(array.dtype, np.integer) else dtype
        if logical == 'float16':
            array = array.astype(np.float16)
        selected = noncontiguous is True or (
            isinstance(noncontiguous, tuple) and key in noncontiguous
        )
        writer.input(key, array, dtype=logical, layout=_layout(array.shape, selected))
    for key, shape, rule in outputs:
        writer.output(key, shape, 'float32', rule, actual_dtype='float32')
    return writer.finish()


def _save(case, root, outputs):
    """Persist the calculated float32 observations and CPU provenance."""
    specs = {
        key: save_tensor(root, key, np.asarray(value, dtype=np.float32))
        for key, value in outputs.items()
    }
    write_result(
        case,
        root,
        specs,
        {
            'framework': 'NumPy',
            'numpy_version': np.__version__,
            'device': 'cpu',
            'arithmetic': 'source NumPy float32 formulas and reduction order; inputs rounded before computation',
            'independence': 'NumPy formulas do not call project native operators or runners',
        },
    )
    return root / 'result.json'


def _close(atol, rtol):
    """Describe the source allclose assertion."""
    return {'kind': 'close', 'atol': atol, 'rtol': rtol}


def _norm(root):
    """Generate grouped norm and gated residual observations."""
    rng = np.random.default_rng(53)
    c, h, rank = 4, 8, 5
    x = rng.normal(size=(1, 3, c * h)).astype(np.float32)
    norm = rng.normal(scale=0.1, size=c * h).astype(np.float32)
    down = rng.normal(scale=0.1, size=(rank, c * h)).astype(np.float32)
    up = rng.normal(scale=0.1, size=(c * h, rank)).astype(np.float32)
    inject = rng.normal(scale=0.1, size=(c, c * h)).astype(np.float32)
    branch = rng.normal(size=(1, 3, h)).astype(np.float32)
    close = _close(2e-4, 2e-4)
    case = _start(
        root,
        'flash_norm',
        'grouped_norm_gated_residual',
        {'group_size': h, 'hc_count': c, 'optional_inject_none': True},
        dict(x=x, norm=norm, down=down, up=up, inject=inject, branch=branch),
        [
            ('normalized', x.shape, close),
            ('pre', branch.shape, close),
            ('residual', x.shape, {'kind': 'exact'}),
            ('gates', (1, 3, c), close),
            ('no_inject_pre', branch.shape, close),
            ('post', x.shape, close),
        ],
    )
    reference(case, case.root / 'reference')
    return [case.root / 'case.json']


def _mhc(root):
    """Generate all original Sinkhorn iteration cases."""
    results = []
    for iterations in (1, 5, 20):
        rng = np.random.default_rng(54)
        c, h = 4, 8
        mix = (2 + c) * c
        x = rng.normal(size=(1, 2, c, h)).astype(np.float32)
        function = rng.normal(scale=0.08, size=(mix, c * h)).astype(np.float32)
        base = rng.normal(scale=0.05, size=mix).astype(np.float32)
        scale = np.asarray([0.8, 0.9, 1.1], np.float32)
        branch = rng.normal(size=(1, 2, h)).astype(np.float32)
        case = _start(
            root,
            f'flash_mhc_{iterations}',
            'mhc_sinkhorn',
            {'sinkhorn_iterations': iterations, 'post_inputs_contiguous': True},
            dict(x=x, function=function, base=base, scale=scale, branch=branch),
            [
                ('post_gate', (1, 2, c), _close(7e-4, 7e-4)),
                ('comb', (1, 2, c, c), _close(7e-4, 7e-4)),
                ('pre', (1, 2, h), _close(7e-4, 7e-4)),
                ('post', x.shape, _close(8e-4, 8e-4)),
            ],
            noncontiguous=('x', 'function'),
        )
        reference(case, case.root / 'reference')
        results.append(case.root / 'case.json')
    return results


def _kda(root):
    """Generate both lower-bound forget gate cases."""
    results = []
    for bound in (-5.0, -2.0):
        rng = np.random.default_rng(55)
        x = rng.normal(size=(2, 5, 12)).astype(np.float32)
        fa = rng.normal(scale=0.2, size=(4, 12)).astype(np.float32)
        fb = rng.normal(scale=0.2, size=(12, 4)).astype(np.float32)
        bias = rng.normal(scale=0.1, size=12).astype(np.float32)
        log = rng.normal(scale=0.1, size=3).astype(np.float32)
        case = _start(
            root,
            f'flash_kda_{int(-bound)}',
            'kda_forget_gate',
            {'lower_bound': bound, 'num_heads': 3, 'head_dim': 4, 'native_checks': ['bounds']},
            dict(x=x, fa=fa, fb=fb, bias=bias, log=log),
            [('output', (2, 5, 3, 4), _close(2e-3, 1e-3))],
            noncontiguous=True,
        )
        reference(case, case.root / 'reference')
        results.append(case.root / 'case.json')
    return results


def _scores(root):
    """Generate QSA, weighted K-pool scores and complete-group states."""
    results = []
    for dtype in ('float32', 'float16', 'bfloat16'):
        rng = np.random.default_rng(56)
        q = _rounded(rng.normal(size=(2, 3, 4, 8)).astype(np.float32), dtype)
        k = _rounded(rng.normal(size=(2, 5, 8)).astype(np.float32), dtype)
        weights = _rounded(rng.normal(size=(2, 3, 4)).astype(np.float32), dtype)
        for op in ('qsa', 'kpool_scores'):
            inputs = {'q': q, 'k': k}
            if op == 'kpool_scores':
                inputs['weights'] = weights
            case = _start(
                root,
                f'flash_{op}_{dtype}',
                op,
                {},
                inputs,
                [('output', (2, 3, 5), _close(4e-3, 2e-3))],
                dtype=dtype,
                noncontiguous=op == 'qsa',
            )
            reference(case, case.root / 'reference')
            results.append(case.root / 'case.json')
    for tokens in (0, 3, 10):
        rng = np.random.default_rng(56)
        keys = rng.normal(size=(2, tokens, 8)).astype(np.float32)
        gates = rng.normal(size=keys.shape).astype(np.float32)
        ape = rng.normal(scale=0.1, size=(4, 8)).astype(np.float32)
        complete = tokens // 4
        case = _start(
            root,
            f'flash_kpool_states_{tokens}',
            'kpool_states',
            {'pool_size': 4},
            dict(keys=keys, gates=gates, ape=ape),
            [('output', (2, complete, 8), _close(3e-3, 2e-3) if complete else {'kind': 'exact'})],
        )
        reference(case, case.root / 'reference')
        results.append(case.root / 'case.json')
    return results


def _attention(q, k, v, offset=None, indices=None, scale=None):
    """Evaluate the independent source attention reference, including invalid indices."""
    b, heads, tokens, d = q.shape
    k = np.repeat(k, heads // k.shape[1], axis=1)
    v = np.repeat(v, heads // v.shape[1], axis=1)
    if indices is not None:
        k, v = k.astype(np.float16).astype(np.float32), v.astype(np.float16).astype(np.float32)
    out = np.zeros((b, tokens, heads, d), np.float32)
    for batch in range(b):
        for t in range(tokens):
            selected = (
                np.arange(offset + t + 1)
                if indices is None
                else indices[batch, t][(indices[batch, t] >= 0) & (indices[batch, t] < k.shape[2])]
            )
            if selected.size == 0:
                continue
            scores = np.einsum('hd,hkd->hk', q[batch, :, t], k[batch][:, selected])
            scores *= 1 / math.sqrt(d) if scale is None else scale
            out[batch, t] = np.einsum('hk,hkd->hd', _softmax(scores), v[batch][:, selected])
    return out


def _attention_cases(root):
    """Generate dense and sparse cases, including published width combinations."""
    results = []
    for mla in (False, True):
        rng = np.random.default_rng(57 if mla else 58)
        heads, kv = (3, 1) if mla else (4, 2)
        q = rng.normal(scale=0.2, size=(1, heads, 2, 128)).astype(np.float32)
        k = rng.normal(scale=0.2, size=(1, kv, 5, 128)).astype(np.float32)
        v = k if mla else rng.normal(scale=0.2, size=k.shape).astype(np.float32)
        indices = np.asarray([[[0, 2, 3, -1], [1, 2, 4, -1]]], np.int64)
        scale = 1 / math.sqrt(64) if mla else None
        for sparse in (False, True):
            inputs = {'q': q, 'k': k[:, 0] if mla else k}
            if not mla:
                inputs['v'] = v
            if sparse:
                inputs['indices'] = indices
            case = _start(
                root,
                f'flash_attn_{"mla" if mla else "gqa"}_{"sparse" if sparse else "dense"}',
                'attention',
                {
                    'mla': mla,
                    'sparse': sparse,
                    'query_offset': None if sparse else 3,
                    'scale': scale,
                },
                inputs,
                [
                    (
                        'output',
                        (1, 2, heads, 128),
                        _close(5e-3, 5e-3) if sparse else _close(4e-3, 4e-3),
                    )
                ],
            )
            reference(case, case.root / 'reference')
            results.append(case.root / 'case.json')
    geometries = [(False, 256, 24, 2), (True, 512, 64, 1), (False, 31, 4, 2), (True, 1025, 2, 1)]
    for mla, width, heads, kv in geometries:
        for dtype in ('float32', 'float16', 'bfloat16'):
            rng = np.random.default_rng(5801)
            q = _rounded(rng.normal(scale=0.1, size=(2, heads, 2, width)).astype(np.float32), dtype)
            k = _rounded(rng.normal(scale=0.1, size=(2, kv, 8, width)).astype(np.float32), dtype)
            v = (
                k
                if mla
                else _rounded(rng.normal(scale=0.1, size=k.shape).astype(np.float32), dtype)
            )
            indices = np.asarray(
                [[[0, 2, 4, 6], [3, 3, -1, 8]], [[-1, -1, 9, -2], [7, 0, 2, 4]]], np.int64
            )
            scale = 1 / math.sqrt(256) if mla else None
            inputs = {'q': q, 'k': k[:, 0] if mla else k, 'indices': indices}
            if not mla:
                inputs['v'] = v
            case = _start(
                root,
                f'flash_sparse_{width}_{heads}_{kv}_{dtype}',
                'attention',
                {
                    'mla': mla,
                    'sparse': True,
                    'scale': scale,
                    'native_checks': ['empty_selection_zero'],
                },
                inputs,
                [('output', (2, 2, heads, width), _close(6e-3, 6e-3))],
                dtype=dtype,
                noncontiguous=True,
            )
            reference(case, case.root / 'reference')
            results.append(case.root / 'case.json')
    return results


def _ple(root):
    """Generate full convolution references and retain chunk checks as consistency checks."""
    results = []
    for dilation, kernel in ((3, 4), (1, 1)):
        rng = np.random.default_rng(59)
        x = rng.normal(size=(2, 7, 11)).astype(np.float32)
        weight = rng.normal(scale=0.2, size=(11, 1, kernel)).astype(np.float32)
        case = _start(
            root,
            f'flash_ple_d{dilation}_k{kernel}',
            'ple_dilated_conv_silu',
            {
                'dilation': dilation,
                'kernel': kernel,
                'first_tokens': 3,
                'native_checks': ['chunk_concatenation_matches_full', 'state_last_equals_full'],
                'chunk_comparison': _close(1e-5, 1e-5),
                'state_comparison': {'kind': 'exact'},
                'state_reference': 'source checks equality only; no independent state values',
            },
            {'x': x, 'first_input': x[:, :3], 'second_input': x[:, 3:], 'weight': weight},
            [
                ('full', (2, 7, 11), _close(1e-5, 1e-5)),
            ],
        )
        reference(case, case.root / 'reference')
        results.append(case.root / 'case.json')
    return results


def _reference_norm(case):
    """Calculate grouped normalization and gated residual outputs."""
    x, norm, down, up, inject, branch = (
        case.tensor(k) for k in ('x', 'norm', 'down', 'up', 'inject', 'branch')
    )
    c, h = case.data['config']['hc_count'], case.data['config']['group_size']
    grouped = x.reshape(*x.shape[:-1], c, h)
    normalized = (grouped / np.sqrt(np.mean(grouped**2, axis=-1, keepdims=True) + 1e-6)).reshape(
        x.shape
    ) * (1 + norm)
    low = normalized @ down.T / c
    low = low * _sigmoid(low)
    mixing = _sigmoid(low @ up.T).reshape(grouped.shape)
    pre = (mixing * normalized.reshape(grouped.shape)).mean(axis=-2)
    gates = 2 * _sigmoid(normalized @ inject.T / c)
    post = x + (branch[..., None, :] * gates[..., :, None]).reshape(x.shape)
    return dict(
        normalized=normalized, pre=pre, residual=x, gates=gates, no_inject_pre=pre, post=post
    )


def _reference_mhc(case):
    """Calculate Sinkhorn pre- and post-mixing observations."""
    x, function, base, scale, branch = (
        case.tensor(k) for k in ('x', 'function', 'base', 'scale', 'branch')
    )
    c = x.shape[-2]
    flat = x.reshape(*x.shape[:-2], -1)
    flat = flat / np.sqrt(np.mean(flat**2, axis=-1, keepdims=True) + 1e-5)
    pre, post, comb = np.split(flat @ function.T, [c, 2 * c], axis=-1)
    pre = _sigmoid(pre * scale[0] + base[:c]) + 1e-6
    post = 2 * _sigmoid(post * scale[1] + base[c : 2 * c])
    comb = (
        _softmax(comb.reshape(*x.shape[:-2], c, c) * scale[2] + base[2 * c :].reshape(c, c)) + 1e-6
    )
    comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
    for _ in range(case.data['config']['sinkhorn_iterations'] - 1):
        comb /= comb.sum(axis=-1, keepdims=True) + 1e-6
        comb /= comb.sum(axis=-2, keepdims=True) + 1e-6
    value = (pre[..., None] * x).sum(axis=-2)
    final = post[..., None] * branch[..., None, :] + np.matmul(comb.swapaxes(-1, -2), x)
    return dict(post_gate=post, comb=comb, pre=value, post=final)


def _reference_scores(case):
    """Calculate QSA or weighted K-pool score observations."""
    q, k = case.tensor('q').astype(np.float32), case.tensor('k').astype(np.float32)
    dots = np.maximum(np.einsum('bthd,bpd->bthp', q, k), 0)
    if case.data['op'] == 'qsa':
        return {'output': dots.sum(axis=2) / math.sqrt(8)}
    weights = case.tensor('weights').astype(np.float32)
    return {'output': (dots / math.sqrt(8) * (weights / math.sqrt(4))[..., None]).sum(axis=2)}


def _reference_ple(case):
    """Calculate the independent full output; chunk replay is checked natively."""
    x, weight = case.tensor('x'), case.tensor('weight')
    chunks = np.concatenate([case.tensor('first_input'), case.tensor('second_input')], axis=1)
    if not np.array_equal(x, chunks):
        raise ValueError('PLE chunks must contain the same fixed input as full execution')
    config = case.data['config']
    dilation, kernel = config['dilation'], config['kernel']
    length = (kernel - 1) * dilation
    combined = np.concatenate([np.zeros((x.shape[0], length, x.shape[2]), np.float32), x], axis=1)
    acc = np.zeros_like(x)
    for tap in range(kernel):
        acc += combined[:, tap * dilation : tap * dilation + x.shape[1]] * weight[:, 0, tap]
    return {'full': acc * _sigmoid(acc)}


def reference(case: Case, root: Path) -> Path:
    """Recompute and persist a Flash-Next reference using only fixed case inputs."""
    op, config = case.data['op'], case.data['config']
    if op == 'grouped_norm_gated_residual':
        outputs = _reference_norm(case)
    elif op == 'mhc_sinkhorn':
        outputs = _reference_mhc(case)
    elif op == 'kda_forget_gate':
        x, fa, fb, bias, log = (case.tensor(k) for k in ('x', 'fa', 'fb', 'bias', 'log'))
        gate = ((x @ fa.T) @ fb.T + bias).reshape(
            *x.shape[:-1], config['num_heads'], config['head_dim']
        )
        outputs = {
            'output': config['lower_bound'] * _sigmoid(np.exp(log)[None, None, :, None] * gate)
        }
    elif op in ('qsa', 'kpool_scores'):
        outputs = _reference_scores(case)
    elif op == 'kpool_states':
        keys, gates, ape = (case.tensor(k) for k in ('keys', 'gates', 'ape'))
        size = config['pool_size']
        complete = keys.shape[1] // size
        if complete:
            shape = (keys.shape[0], complete, size, keys.shape[2])
            value = (
                _softmax(gates[:, : complete * size].reshape(shape) + ape[None, None], axis=2)
                * keys[:, : complete * size].reshape(shape)
            ).sum(axis=2)
        else:
            value = np.empty((keys.shape[0], 0, keys.shape[2]), np.float32)
        outputs = {'output': value}
    elif op == 'attention':
        q, k = case.tensor('q').astype(np.float32), case.tensor('k').astype(np.float32)
        if config['mla']:
            k, v = k[:, None], k[:, None]
        else:
            v = case.tensor('v').astype(np.float32)
        indices = case.tensor('indices') if config['sparse'] else None
        outputs = {
            'output': _attention(
                q, k, v, offset=config.get('query_offset'), indices=indices, scale=config['scale']
            )
        }
    elif op == 'ple_dilated_conv_silu':
        outputs = _reference_ple(case)
    else:
        raise ValueError(f'Unsupported Flash-Next operation: {op!r}')
    return _save(case, Path(root), outputs)


def generate(root: Path, suite: str = 'norm') -> list[Path]:
    """Generate a selected Flash-Next NumPy suite and return case manifests.

    Supported suites are ``norm``, ``mhc``, ``kda``, ``scores``, ``attention``
    and ``ple``. Graph, stream, error and state equality checks require separate
    native execution and are not performed by this generator.
    """
    if suite == 'norm':
        return _norm(root)
    if suite == 'mhc':
        return _mhc(root)
    if suite == 'kda':
        return _kda(root)
    if suite == 'scores':
        return _scores(root)
    if suite == 'attention':
        return _attention_cases(root)
    if suite == 'ple':
        return _ple(root)
    raise ValueError(f'Unknown Flash-Next suite: {suite!r}')
