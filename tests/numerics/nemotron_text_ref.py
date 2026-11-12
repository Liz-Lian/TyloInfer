"""Fixed NumPy references for the synthetic Nemotron-H text backbone."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .case import Case, CaseWriter, save_tensor, write_result

HIDDEN = 64
MAMBA_HEADS = 4
MAMBA_HEAD_DIM = 8
MAMBA_GROUPS = 2
SSM_STATE = 32
CONV_KERNEL = 4
ATTENTION_HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
EXPERTS = 3
TOP_K = 2
MOE_INTERMEDIATE = 8
MOE_LATENT = 16
SHARED_INTERMEDIATE = 12
DENSE_INTERMEDIATE = 32
VOCAB = 32

LAYER_MIXERS = ('mamba', 'moe', 'attention', 'moe', 'mamba', 'attention')
MTP_MIXERS = ('attention', 'moe')


def _sigmoid(x):
    """Calculate the source _sigmoid."""
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    """Calculate the stable source _softplus."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _half(x):
    """Round a float32 array through the runtime's float16 operand boundary."""

    return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)


def _rms_norm(x, weight, eps):
    """Apply the source float32 RMS normalization."""
    x = np.asarray(x, np.float32)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def _config_dict(mixers, mtp=False):
    """Construct the source synthetic backbone configuration."""
    llm = dict(
        model_type='nemotron_h',
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=DENSE_INTERMEDIATE,
        num_attention_heads=ATTENTION_HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attention_bias=False,
        max_position_embeddings=32,
        layer_norm_epsilon=1e-5,
        layers_block_type=list(mixers),
        mamba_num_heads=MAMBA_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        ssm_state_size=SSM_STATE,
        n_groups=MAMBA_GROUPS,
        conv_kernel=CONV_KERNEL,
        use_conv_bias=True,
        mamba_hidden_act='silu',
        mamba_proj_bias=False,
        mamba_ssm_cache_dtype='float32',
        chunk_size=4,
        time_step_min=1e-3,
        time_step_max=1e-1,
        time_step_floor=1e-4,
        time_step_limit=[0.0, {'__float__': 'Infinity'}],
        n_routed_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=MOE_INTERMEDIATE,
        moe_shared_expert_intermediate_size=SHARED_INTERMEDIATE,
        moe_latent_size=MOE_LATENT,
        moe_shared_expert_overlap=False,
        mlp_hidden_act='relu2',
        norm_topk_prob=True,
        routed_scaling_factor=5.0,
        n_shared_experts=1,
        use_bias=False,
        mlp_bias=False,
        num_nextn_predict_layers=0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        tie_word_embeddings=False,
        eos_token_id=11,
    )
    if mtp:
        llm['num_nextn_predict_layers'] = 1
        llm['mtp_layers_block_type'] = list(MTP_MIXERS)
        llm['max_position_embeddings'] = 64
    return dict(
        model_type='nemotron_h_omni',
        llm_config=llm,
        tie_word_embeddings=False,
        eos_token_id=11,
    )


def _rand(rng, shape, scale=0.08):
    """Draw the original float32 synthetic weights in order."""
    return rng.normal(scale=scale, size=shape).astype(np.float32)


def _mixer_weights(rng, prefix, kind):
    """Build one pre-normed mixer block under a canonical block prefix."""

    weights = {prefix + '.attention.norm.weight': np.ones(HIDDEN, np.float32)}
    inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    conv_width = inner + 2 * MAMBA_GROUPS * SSM_STATE
    projection = inner + conv_width + MAMBA_HEADS
    if kind == 'mamba':
        name = prefix + '.linear_attention.'
        weights[name + 'in_projection.weight'] = _rand(rng, (projection, HIDDEN))
        weights[name + 'conv.weight'] = _rand(rng, (conv_width, 1, CONV_KERNEL), 0.05)
        weights[name + 'conv.bias'] = _rand(rng, (conv_width,), 0.02)
        weights[name + 'a'] = _rand(rng, (MAMBA_HEADS,), 0.05)
        weights[name + 'd'] = _rand(rng, (MAMBA_HEADS,), 0.05)
        weights[name + 'dt_bias'] = _rand(rng, (MAMBA_HEADS,), 0.05)
        weights[name + 'norm.weight'] = np.ones(inner, np.float32)
        weights[name + 'output.weight'] = _rand(rng, (HIDDEN, inner))
    elif kind == 'attention':
        name = prefix + '.attention.'
        weights[name + 'query.weight'] = _rand(rng, (HIDDEN, HIDDEN))
        weights[name + 'key.weight'] = _rand(rng, (KV_HEADS * HEAD_DIM, HIDDEN))
        weights[name + 'value.weight'] = _rand(rng, (KV_HEADS * HEAD_DIM, HIDDEN))
        weights[name + 'output.weight'] = _rand(rng, (HIDDEN, HIDDEN))
    elif kind == 'moe':
        name = prefix + '.mlp.'
        weights[name + 'router.weight'] = _rand(rng, (EXPERTS, HIDDEN))
        weights[name + 'router.bias'] = _rand(rng, (EXPERTS,), 0.02)
        weights[name + 'latent.down.weight'] = _rand(rng, (MOE_LATENT, HIDDEN))
        weights[name + 'latent.up.weight'] = _rand(rng, (HIDDEN, MOE_LATENT))
        weights[name + 'experts.up.weight'] = _rand(rng, (EXPERTS, MOE_INTERMEDIATE, MOE_LATENT))
        weights[name + 'experts.down.weight'] = _rand(rng, (EXPERTS, MOE_LATENT, MOE_INTERMEDIATE))
        weights[name + 'shared_expert.up.weight'] = _rand(rng, (SHARED_INTERMEDIATE, HIDDEN))
        weights[name + 'shared_expert.down.weight'] = _rand(rng, (HIDDEN, SHARED_INTERMEDIATE))
    elif kind == 'mlp':
        name = prefix + '.mlp.'
        weights[name + 'up.weight'] = _rand(rng, (DENSE_INTERMEDIATE, HIDDEN))
        weights[name + 'down.weight'] = _rand(rng, (HIDDEN, DENSE_INTERMEDIATE))
    else:
        raise AssertionError(kind)
    return weights


def fixture(mixers=('mamba', 'moe'), mtp=False):
    """Return the original synthetic configuration and ordered float32 weights."""
    from tyloinfer.architectures.nemotron_h import NemotronHConfig

    config = _config_dict(mixers, mtp)
    # The fixture must satisfy the same contract as a converted checkpoint.
    NemotronHConfig.from_hf_config(config)
    rng = np.random.default_rng(5306)

    weights = {
        'model.token_embedding.weight': _rand(rng, (VOCAB, HIDDEN), 0.2),
        'model.output.weight': _rand(rng, (VOCAB, HIDDEN), 0.1),
        'model.output_norm.weight': np.ones(HIDDEN, np.float32),
    }
    for layer, kind in enumerate(mixers):
        weights.update(_mixer_weights(rng, f'model.block.{layer}', kind))
    if mtp:
        weights['model.output_norm.weight'] = np.linspace(0.5, 1.5, HIDDEN).astype(np.float32)
        weights['predictor.embedding_norm.weight'] = np.ones(HIDDEN, np.float32)
        weights['predictor.hidden_norm.weight'] = np.ones(HIDDEN, np.float32)
        weights['predictor.output_norm.weight'] = np.linspace(0.6, 1.4, HIDDEN).astype(np.float32)
        weights['predictor.fusion.weight'] = _rand(rng, (HIDDEN, 2 * HIDDEN))
        for layer, kind in enumerate(MTP_MIXERS):
            weights.update(_mixer_weights(rng, f'predictor.block.{layer}', kind))
    return config, weights


def _mamba_reference(x, weights, prefix, contract, eps, initial_state):
    """Evaluate the original Mamba convolution, scan and gated projection."""
    batch, tokens, _ = x.shape
    inner = MAMBA_HEADS * MAMBA_HEAD_DIM
    state_width = SSM_STATE
    groups = MAMBA_GROUPS
    conv_width = inner + 2 * groups * state_width
    projection = x @ weights[prefix + 'in_projection.weight'].T
    gate = projection[..., :inner]
    conv_input = projection[..., inner : inner + conv_width]
    dt = projection[..., inner + conv_width :]
    # The convolution starts from an empty state and is causal: tap j of the
    # [C,1,K] kernel weights the input lag K-1-j.
    convolved = np.zeros((batch, tokens, conv_width), np.float32)
    for tap in range(CONV_KERNEL):
        lag = CONV_KERNEL - 1 - tap
        shifted = np.zeros_like(convolved)
        if lag < tokens:
            shifted[:, lag:] = conv_input[:, : tokens - lag]
        convolved += shifted * weights[prefix + 'conv.weight'][:, 0, tap]
    convolved = convolved + weights[prefix + 'conv.bias']
    convolved = convolved * _sigmoid(convolved)
    scan_input = convolved[..., :inner].reshape(batch, tokens, MAMBA_HEADS, MAMBA_HEAD_DIM)
    b = convolved[..., inner : inner + groups * state_width].reshape(
        batch, tokens, groups, state_width
    )
    c = convolved[..., inner + groups * state_width :].reshape(batch, tokens, groups, state_width)
    repeat = MAMBA_HEADS // groups
    b = np.repeat(b, repeat, axis=2)
    c = np.repeat(c, repeat, axis=2)
    a = -np.exp(weights[prefix + 'a'].astype(np.float32))
    lo, hi = contract.time_step_limit
    step = np.clip(_softplus(dt + weights[prefix + 'dt_bias']), lo, hi)
    skip = weights[prefix + 'd'].astype(np.float32).reshape(1, 1, MAMBA_HEADS, 1)
    state = initial_state.copy()
    scanned = np.zeros((batch, tokens, MAMBA_HEADS, MAMBA_HEAD_DIM), np.float32)
    for token in range(tokens):
        decay = np.exp(step[:, token, :, None, None] * a[None, :, None, None])
        state = (
            state * decay
            + (step[:, token, :, None, None] * b[:, token, :, None, :])
            * scan_input[:, token, :, :, None]
        )
        scanned[:, token] = (
            np.sum(state * c[:, token, :, None, :], axis=-1) + skip[:, 0] * scan_input[:, token]
        )
    gated = scanned.reshape(batch * tokens, inner) * (gate * _sigmoid(gate)).reshape(
        batch * tokens, inner
    )
    grouped = gated.reshape(batch * tokens, groups, inner // groups)
    normalized = grouped / np.sqrt(np.mean(grouped * grouped, axis=-1, keepdims=True) + eps)
    normalized = normalized.reshape(batch * tokens, inner) * weights[prefix + 'norm.weight']
    return (normalized @ weights[prefix + 'output.weight'].T).reshape(x.shape)


def _attention_reference(x, weights, prefix, cache, offset):
    """Apply float16 attention boundaries and the causal cache formula."""
    batch, tokens, _ = x.shape
    q = (
        _half(x @ weights[prefix + 'query.weight'].T)
        .reshape(batch, tokens, ATTENTION_HEADS, HEAD_DIM)
        .transpose(0, 2, 1, 3)
    )
    k = (
        _half(x @ weights[prefix + 'key.weight'].T)
        .reshape(batch, tokens, KV_HEADS, HEAD_DIM)
        .transpose(0, 2, 1, 3)
    )
    v = (
        _half(x @ weights[prefix + 'value.weight'].T)
        .reshape(batch, tokens, KV_HEADS, HEAD_DIM)
        .transpose(0, 2, 1, 3)
    )
    cache[0][:, :, offset : offset + tokens] = k
    cache[1][:, :, offset : offset + tokens] = v
    visible = offset + tokens
    keys = cache[0][:, :, :visible]
    values = cache[1][:, :, :visible]
    repeat = ATTENTION_HEADS // KV_HEADS
    keys = np.repeat(keys, repeat, axis=1)
    values = np.repeat(values, repeat, axis=1)
    scores = np.einsum('bhqd,bhkd->bhqk', q, keys) / math.sqrt(HEAD_DIM)
    queries = np.arange(offset, offset + tokens)
    allowed = np.arange(visible)[None, :] <= queries[:, None]
    scores = np.where(allowed[None, None], scores, -np.inf)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    attended = _half(np.einsum('bhqk,bhkd->bhqd', probabilities, values))
    attended = attended.transpose(0, 2, 1, 3).reshape(batch, tokens, -1)
    return attended @ weights[prefix + 'output.weight'].T


def _moe_reference(x, weights, prefix, contract):
    """Calculate stable expert routing and the shared feed-forward branch."""
    # The shared FFN casts its activation to float16 once at its entry.
    flat = _half(x).reshape(-1, HIDDEN)
    logits = flat @ weights[prefix + 'router.weight'].T
    scores = _sigmoid(logits)
    choice = scores + weights[prefix + 'router.bias']
    ids = np.argsort(-choice, axis=-1, kind='stable')[:, :TOP_K]
    selected = np.take_along_axis(scores, ids, axis=-1)
    selected = selected / np.maximum(selected.sum(axis=-1, keepdims=True), 1e-20)
    selected = selected * contract.routed_scaling_factor
    latent = flat @ weights[prefix + 'latent.down.weight'].T
    up = np.einsum('troi,ti->tro', weights[prefix + 'experts.up.weight'][ids], latent)
    hidden = np.maximum(up, 0.0) ** 2
    down = np.einsum('troi,tri->tro', weights[prefix + 'experts.down.weight'][ids], hidden)
    routed = (down * selected[..., None]).sum(axis=1)
    routed = routed @ weights[prefix + 'latent.up.weight'].T
    shared = np.maximum(flat @ weights[prefix + 'shared_expert.up.weight'].T, 0.0) ** 2
    shared = _half(shared @ weights[prefix + 'shared_expert.down.weight'].T)
    return (routed + shared).reshape(x.shape)


def _dense_mlp_reference(x, weights, prefix):
    """Calculate the dense squared-ReLU feed-forward branch."""
    flat = _half(x).reshape(-1, HIDDEN)
    hidden = np.maximum(flat @ weights[prefix + 'up.weight'].T, 0.0) ** 2
    return (hidden @ weights[prefix + 'down.weight'].T).reshape(x.shape)


def _model_reference(config, weights, ids, cache, state):
    """Evaluate the synthetic backbone with its recorded zero initial state."""
    from tyloinfer.architectures.nemotron_h import NemotronHConfig

    contract = NemotronHConfig.from_hf_config(config)
    hidden = weights['model.token_embedding.weight'][ids]
    offset = 0
    for layer, kind in enumerate(contract.mixer_types):
        prefix = f'model.block.{layer}.'
        normalized = _rms_norm(
            hidden, weights[prefix + 'attention.norm.weight'], contract.norm_epsilon
        )
        if kind == 'linear_attention':
            branch = _mamba_reference(
                normalized,
                weights,
                prefix + 'linear_attention.',
                contract,
                contract.norm_epsilon,
                state,
            )
        elif kind == 'full_attention':
            branch = _attention_reference(normalized, weights, prefix + 'attention.', cache, offset)
        elif kind == 'moe':
            branch = _moe_reference(normalized, weights, prefix + 'mlp.', contract)
        else:
            branch = _dense_mlp_reference(normalized, weights, prefix + 'mlp.')
        hidden = hidden + branch
    finalized = _rms_norm(hidden, weights['model.output_norm.weight'], contract.norm_epsilon)
    logits = finalized @ weights['model.output.weight'].T
    return finalized, logits


def _mtp_head(config, weights, previous, ids, cache, offset):
    """Calculate one MTP pass with its explicit predictor attention history."""
    from tyloinfer.architectures.nemotron_h import NemotronHConfig

    contract = NemotronHConfig.from_hf_config(config)
    eps = contract.norm_epsilon
    embedded = weights['model.token_embedding.weight'][ids]
    e = _rms_norm(embedded, weights['predictor.embedding_norm.weight'], eps)
    h = _rms_norm(previous, weights['predictor.hidden_norm.weight'], eps)
    hidden = np.concatenate([e, h], axis=-1) @ weights['predictor.fusion.weight'].T
    for layer, kind in enumerate(contract.mtp_mixer_types):
        prefix = f'predictor.block.{layer}.'
        normalized = _rms_norm(hidden, weights[prefix + 'attention.norm.weight'], eps)
        if kind == 'full_attention':
            branch = _attention_reference(normalized, weights, prefix + 'attention.', cache, offset)
        elif kind == 'moe':
            branch = _moe_reference(normalized, weights, prefix + 'mlp.', contract)
        else:
            branch = _dense_mlp_reference(normalized, weights, prefix + 'mlp.')
        hidden = hidden + branch
    finalized = _rms_norm(hidden, weights['predictor.output_norm.weight'], eps)
    return finalized, finalized @ weights['model.output.weight'].T


def reference(case: Case, root: Path) -> Path:
    """Recompute backbone or MTP observations from fixed case inputs."""
    op = case.data['op']
    if (
        op
        not in (
            'nemotron_text_backbone',
            'nemotron_mtp_head',
            'nemotron_mtp_chain',
            'nemotron_mtp_target',
        )
        or case.data['source'].get('generator') != 'tests/numerics/nemotron_text_ref.py'
    ):
        raise ValueError('Unsupported Nemotron text reference case')
    ids = case.tensor('ids')
    cache = [case.tensor('key_cache'), case.tensor('value_cache')]
    if any(np.any(value != 0) for value in cache):
        raise ValueError('The migrated reference requires zero initial KV state')
    weights = {
        name: case.tensor(name)
        for name in case.data['inputs']
        if name.startswith(('model.', 'predictor.'))
    }
    config = case.data['config']['model']
    if op == 'nemotron_text_backbone':
        state, conv = case.tensor('mamba_state'), case.tensor('conv_left_context')
        if np.any(state != 0) or np.any(conv != 0):
            raise ValueError('The migrated backbone reference requires zero initial state')
        hidden, logits = _model_reference(config, weights, ids, cache, state)
        values = {'hidden': hidden, 'logits': logits}
    elif op == 'nemotron_mtp_head':
        hidden, logits = _mtp_head(config, weights, case.tensor('previous'), ids, cache, 0)
        values = {'head': hidden, 'logits': logits}
    elif op == 'nemotron_mtp_chain':
        hidden = case.tensor('previous')[:, :1]
        values = {}
        for index, name in enumerate(('chain_first', 'chain_second')):
            hidden, _ = _mtp_head(config, weights, hidden, ids[:, index : index + 1], cache, index)
            values[name] = hidden
    else:
        hidden, _ = _mtp_head(
            config, weights, case.tensor('target_hidden')[:, :6], ids[:, 1:], cache, 0
        )
        values = {'head_from_target': hidden}
    root = Path(root)
    outputs = {name: save_tensor(root, name, value) for name, value in values.items()}
    execution = {
        'framework': 'NumPy',
        'numpy_version': np.__version__,
        'device': 'cpu',
        'arithmetic': 'source float32 graph with float16 attention and MoE boundaries',
        'independence': 'NumPy model formulas without native kernels or runtime',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return write_result(case, root, outputs, execution)


def _mtp_case(root, kind, ids, hidden, capture=None):
    """Freeze MTP inputs, source weights and the empty predictor KV cache."""
    config, weights = fixture(LAYER_MIXERS, mtp=True)
    source = Path(__file__).parents[1] / 'test_cuda_nemotron_runtime.py'
    diagnostics = Path(__file__).parents[2] / 'csrc/diagnostics/runtime_checks.h'
    tests = {
        'head': 'test_native_mtp_head_matches_reference',
        'chain': 'test_native_mtp_depth_chain_matches_reference',
        'target': 'test_native_mtp_target_hidden_is_post_output_norm',
    }
    provenance = {
        'generator': 'tests/numerics/nemotron_text_ref.py',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'file': 'tests/test_cuda_nemotron_runtime.py',
        'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'diagnostics': 'csrc/diagnostics/runtime_checks.h',
        'diagnostics_sha256': hashlib.sha256(diagnostics.read_bytes()).hexdigest(),
        'test': tests[kind],
        'seed': 5306,
    }
    checks = ['chunked', 'rollback', 'commit'] if kind == 'head' else []
    if kind == 'target':
        provenance['capture_sha256'] = hashlib.sha256(capture).hexdigest()
        checks = ['target_hidden_equals_target_normalized', 'pre_norm_negative_check']
    elif kind == 'chain':
        checks = ['pre_final_norm_chain_negative_check']
    writer = CaseWriter(
        root,
        f'nemotron_mtp_{kind}',
        f'nemotron_mtp_{kind}',
        {
            'model': config,
            'compute': {'device': 'cpu', 'weights': 'float32', 'attention_cache': 'float16'},
            'native_checks': checks,
            'input_origin': 'native capture' if kind == 'target' else 'diagnostic fixed formula',
        },
        provenance,
    )
    for name, value in weights.items():
        writer.input(name, value)
    writer.input('ids', ids)
    writer.input('target_hidden' if kind == 'target' else 'previous', hidden)
    for name in ('key_cache', 'value_cache'):
        writer.input(name, np.zeros((1, KV_HEADS, 32, HEAD_DIM), np.float16))
    if capture is not None:
        writer.asset('capture', capture, encoding='nemotron_h_mtp_check JSON')
    rule = {'kind': 'close', 'atol': 2e-3, 'rtol': 2e-3}
    if kind == 'head':
        writer.output('head', (1, 7, HIDDEN), 'float32', rule)
        writer.output('logits', (1, 7, VOCAB), 'float32', rule)
    elif kind == 'chain':
        for name in ('chain_first', 'chain_second'):
            writer.output(name, (1, 1, HIDDEN), 'float32', rule)
    else:
        writer.output('head_from_target', (1, 6, HIDDEN), 'float32', rule)
    case = writer.finish()
    reference(case, case.root / 'reference')
    return case.root / 'case.json'


def generate_target(root: Path, capture: Path) -> Path:
    """Freeze the actual target hidden and IDs from one MTP diagnostic capture.

    The captured target hidden is an input; native consistency observations are
    recorded separately and never treated as independent expected values.
    """
    raw = Path(capture).read_bytes()
    text = raw.decode('utf-8')
    prefix = 'nemotron_h_mtp_check '
    lines = [line[len(prefix) :] for line in text.splitlines() if line.startswith(prefix)]
    if len(lines) > 1:
        raise ValueError('Expected exactly one MTP diagnostic record')
    data = json.loads(lines[0] if lines else text)
    if not isinstance(data, dict) or data.get('architecture') != 'nemotron_h':
        raise ValueError('Capture is not a Nemotron-H MTP diagnostic')
    ids_entry, target_entry = data['ids'], data['target_hidden']
    if ids_entry['shape'] != [1, 7] or target_entry['shape'] != [1, 7, HIDDEN]:
        raise ValueError('Unexpected MTP capture input shape')
    ids_raw = np.asarray(ids_entry['data'], dtype=np.float64)
    target = np.asarray(target_entry['data'], dtype=np.float32)
    if ids_raw.shape != (7,) or not np.array_equal(ids_raw, np.arange(1, 8)):
        raise ValueError('MTP capture IDs do not match the diagnostic')
    if target.shape != (7 * HIDDEN,) or not np.isfinite(target).all():
        raise ValueError('MTP capture target hidden is invalid')
    return _mtp_case(
        root, 'target', ids_raw.astype(np.int64).reshape(1, 7), target.reshape(1, 7, HIDDEN), raw
    )


def generate(root: Path, suite: str = 'backbone') -> list[Path]:
    """Create the original backbone or fixed-input MTP reference cases."""
    if suite == 'mtp':
        ids = np.arange(1, 8, dtype=np.int64)[None]
        previous = (np.arange(7 * HIDDEN, dtype=np.int64) % 31).astype(np.float32)
        previous = np.multiply(previous, np.float32(0.01))
        previous = np.subtract(previous, np.float32(0.15)).reshape(1, 7, HIDDEN)
        return [_mtp_case(root, kind, ids, previous) for kind in ('head', 'chain')]
    if suite != 'backbone':
        raise ValueError(f'Unknown Nemotron text suite: {suite!r}')
    source = Path(__file__).parents[1] / 'test_cuda_nemotron_runtime.py'
    cases = [LAYER_MIXERS, ('mamba', 'mamba'), ('attention', 'moe'), ('moe', 'mlp', 'attention')]
    results = []
    for index, mixers in enumerate(cases):
        config, weights = fixture(mixers)
        writer = CaseWriter(
            root,
            f'nemotron_text_{index}',
            'nemotron_text_backbone',
            {
                'model': config,
                'mixers': list(mixers),
                'offset': 0,
                'compute': {
                    'device': 'cpu',
                    'weights': 'float32',
                    'attention_cache': 'float16',
                    'mamba_state': 'float32',
                    'conv_left_context': 'logical zero padding, not a native state layout',
                },
                'native_checks': ['decode', 'chunked', 'restore', 'rollback', 'batch', 'reset'],
            },
            {
                'generator': 'tests/numerics/nemotron_text_ref.py',
                'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'file': 'tests/test_cuda_nemotron_runtime.py',
                'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'test': 'test_native_complete_graph',
                'seed': 5306,
            },
        )
        for name, value in weights.items():
            writer.input(name, value)
        writer.input('ids', np.arange(1, 8, dtype=np.int64)[None])
        for name in ('key_cache', 'value_cache'):
            writer.input(name, np.zeros((1, KV_HEADS, 32, HEAD_DIM), np.float16))
        writer.input(
            'mamba_state', np.zeros((1, MAMBA_HEADS, MAMBA_HEAD_DIM, SSM_STATE), np.float32)
        )
        conv_width = MAMBA_HEADS * MAMBA_HEAD_DIM + 2 * MAMBA_GROUPS * SSM_STATE
        writer.input('conv_left_context', np.zeros((1, CONV_KERNEL - 1, conv_width), np.float32))
        rule = {'kind': 'close', 'atol': 2e-3, 'rtol': 2e-3}
        writer.output('hidden', (1, 7, HIDDEN), 'float32', rule)
        writer.output('logits', (1, 7, VOCAB), 'float32', rule)
        case = writer.finish()
        reference(case, case.root / 'reference')
        results.append(case.root / 'case.json')
    return results
