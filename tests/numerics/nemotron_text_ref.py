"""Fixed NumPy references for the synthetic Nemotron-H text backbone."""

from __future__ import annotations

import hashlib
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


def _config_dict(mixers):
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


def fixture(mixers=('mamba', 'moe')):
    """Return the original synthetic configuration and ordered float32 weights."""
    from tyloinfer.architectures.nemotron_h import NemotronHConfig

    config = _config_dict(mixers)
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


def reference(case: Case, root: Path) -> Path:
    """Recompute backbone hidden states and logits from a fixed synthetic case."""
    if (
        case.data['op'] != 'nemotron_text_backbone'
        or case.data['source'].get('generator') != 'tests/numerics/nemotron_text_ref.py'
    ):
        raise ValueError('Unsupported Nemotron text reference case')
    ids = case.tensor('ids')
    cache = [case.tensor('key_cache'), case.tensor('value_cache')]
    state = case.tensor('mamba_state')
    conv = case.tensor('conv_left_context')
    if any(np.any(value != 0) for value in [*cache, state, conv]):
        raise ValueError('The migrated backbone reference requires zero initial state')
    weights = {name: case.tensor(name) for name in case.data['inputs'] if name.startswith('model.')}
    hidden, logits = _model_reference(case.data['config']['model'], weights, ids, cache, state)
    root = Path(root)
    outputs = {
        'hidden': save_tensor(root, 'hidden', hidden),
        'logits': save_tensor(root, 'logits', logits),
    }
    execution = {
        'framework': 'NumPy',
        'numpy_version': np.__version__,
        'device': 'cpu',
        'arithmetic': 'source float32 graph with float16 attention and MoE boundaries',
        'independence': 'NumPy model formulas without native kernels or runtime',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return write_result(case, root, outputs, execution)


def generate(root: Path, suite: str = 'backbone') -> list[Path]:
    """Create the four original backbone cases and their NumPy references.

    MTP requires captured native inputs and is intentionally not generated.
    """
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
