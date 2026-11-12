"""Build the existing TQ Qwen4 batching fixture for native graph diagnostics.

The original fixture uses a 1024-token context. This new consumer case uses
2048 so a prompt and the original 1024-token generation diagnostic both fit.
It is an execution-consistency fixture, not an independent model reference.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from bench.tq_smoke_fixture import matrix
from tyloinfer.architectures.flash_next import Qwen4ExpConfig
from tyloinfer.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import load, save
from tyloinfer.formats.tq import TQExpertTensor, TQTensor


def _base(interval, silu_gate, ple):
    """Reproduce the original Qwen fixture's configuration and random weights."""
    rng = np.random.default_rng(3805)

    def rand(shape, scale=0.05):
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    h, streams, d, heads, kv, ih = 16, 2, 32, 2, 1, 8
    kinds = ['full_attention' if (i + 1) % interval == 0 else 'linear_attention' for i in range(2)]
    ple_layers = (
        [i + 1 for i, kind in enumerate(kinds) if kind == 'linear_attention'] if ple else []
    )
    c = dict(
        model_type='qwen4_exp_text',
        vocab_size=32,
        hidden_size=h,
        num_hidden_layers=2,
        max_position_embeddings=32,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=d,
        partial_rotary_factor=0.25,
        rope_parameters=dict(rope_theta=1e7, mrope_section=[2, 1, 1], mrope_interleaved=silu_gate),
        rms_norm_eps=1e-6,
        layer_types=kinds,
        full_attention_interval=interval,
        hc_count=streams,
        hc_lowrank=4,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_key_head_dim=d,
        linear_value_head_dim=d,
        linear_conv_kernel_dim=4,
        num_experts=3,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        norm_topk_prob=silu_gate,
        output_gate_type='silu' if silu_gate else 'sigmoid',
        hidden_act='silu',
        attention_bias=False,
        indexer_n_heads=ih,
        indexer_kv_heads=1,
        indexer_head_dim=d,
        indexer_budget=4,
        indexer_compress_ratio=2,
        ple_layer_ids=ple_layers,
        ple_embed_dim=h,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=8,
        split_ngram_parts=4,
        make_ngram_vocab_size_divisible_by=1,
        seed=3805,
        mtp_num_hidden_layers=0,
        tie_word_embeddings=False,
        eos_token_id=3,
    )
    outer = dict(model_type='qwen4_exp', text_config=c)
    Qwen4ExpConfig.from_hf_config(outer)
    weights = {
        'model.token_embedding.weight': rand((32, h), 0.2),
        'model.output.weight': rand((32, h)),
    }

    def gr(prefix, combine=True):
        weights[prefix + '.norm.weight'] = rand((streams * h,), 0.02)
        weights[prefix + '.down.weight'] = rand((4, streams * h))
        weights[prefix + '.up.weight'] = rand((streams * h, 4))
        if combine:
            weights[prefix.removesuffix('.pre') + '.post.inject.weight'] = rand(
                (streams, streams * h)
            )

    gr('model.mhc.pre', False)
    for i, kind in enumerate(kinds):
        prefix = f'model.block.{i}'
        gr(prefix + '.attention.mhc.pre')
        gr(prefix + '.mlp.mhc.pre')
        mlp = prefix + '.mlp'
        weights[mlp + '.experts.gate_up.weight'] = rand((3, 32, h))
        weights[mlp + '.experts.down.weight'] = rand((3, h, 16))
        weights[mlp + '.router.weight'] = rand((3, h))
        for key, shape in {
            'gate': (16, h),
            'up': (16, h),
            'down': (h, 16),
            'router': (1, h),
        }.items():
            weights[mlp + f'.shared_expert.{key}.weight'] = rand(shape)
        if kind == 'linear_attention':
            attn = prefix + '.linear_attention'
            for key, shape in {
                'qkv': (4 * d, h),
                'gate': (2 * d, h),
                'alpha': (2, h),
                'beta': (2, h),
                'output': (h, 2 * d),
            }.items():
                weights[attn + f'.{key}.weight'] = rand(shape)
            conv = np.zeros((4 * d, 1, 4), np.float32)
            conv[:, 0, -1], conv[:, 0, -2] = 0.8, 0.1
            weights[attn + '.conv.weight'] = conv
            weights[attn + '.dt_bias'] = rand((2,))
            weights[attn + '.a'] = rand((2,))
            weights[attn + '.norm.weight'] = np.ones(d, np.float32)
        else:
            attn = prefix + '.attention'
            for key, shape in {
                'query': (heads * 2 * d, h),
                'key': (kv * d, h),
                'value': (kv * d, h),
                'output': (h, heads * d),
                'indexer.query_key': ((ih + 1) * d, h),
            }.items():
                weights[attn + f'.{key}.weight'] = rand(shape)
            for key in ['query_norm', 'key_norm', 'indexer.query_norm', 'indexer.key_norm']:
                weights[attn + f'.{key}.weight'] = rand((d,), 0.02)
        if i + 1 in ple_layers:
            pos = prefix + '.position_embedding'
            for shard in range(4):
                weights[pos + f'.ngram.shard.{shard}.weight'] = rand((8, h // 4), 0.2)
            weights[pos + '.ngram.layer_multipliers'] = np.array(
                [-7046029254386353131, 6364136223846793005, -4658895280553007687], np.int64
            )
            weights[pos + '.ngram.head_offsets'] = np.array([0, 8, 16, 24], np.int64)
            weights[pos + '.ngram.head_vocab_sizes'] = np.array([5, 7, 5, 7], np.int64)
            weights[pos + '.key.weight'] = rand((streams * h, h))
            weights[pos + '.value.weight'] = rand((h, h))
            for key in ['key_norm', 'query_norm', 'conv_norm']:
                weights[pos + f'.{key}.weight'] = rand((streams * h,), 0.02)
            weights[pos + '.conv.weight'] = rand((3, streams * h)).T[:, None, :].copy()
    return outer, weights


def _predictor():
    """Reproduce the original one-layer Qwen predictor fixture."""
    rng = np.random.default_rng(3811)

    def rand(shape):
        return rng.normal(scale=0.05, size=shape).astype(np.float32)

    config, weights = _base(2, True, True)
    _, head = _base(1, True, False)
    h, streams = config['text_config']['hidden_size'], config['text_config']['hc_count']
    config['text_config']['mtp_num_hidden_layers'] = 1
    weights['predictor.embedding_norm.weight'] = rand((h,))
    weights['predictor.hidden_norm.weight'] = rand((streams * h,))
    weights['predictor.fusion.embedding.weight'] = rand((h, h))
    weights['predictor.fusion.hidden.weight'] = rand((h, h))
    for name, value in head.items():
        if name.startswith('model.mhc.pre.'):
            weights[name.replace('model.', 'predictor.', 1)] = value.copy()
    for name, value in head.items():
        if name.startswith('model.block.0.'):
            weights[name.replace('model.', 'predictor.', 1)] = value.copy()
    return config, weights


def _experts(weights):
    """Apply the existing seven-expert TQ fixture generator in original order."""
    result = dict(weights)
    rng = np.random.default_rng(2342)
    for name, value in list(weights.items()):
        if '.experts.' not in name:
            continue
        parts = [(name, value)]
        if name.endswith('.gate_up.weight'):
            gate, up = np.split(value, 2, axis=1)
            parts = [
                (name.replace('gate_up.weight', role + '.weight'), part)
                for role, part in (('gate', gate), ('up', up))
            ]
            del result[name]
        for key, part in parts:
            _, rows, width = part.shape
            if key.endswith('.down.weight'):
                width = 128
            else:
                rows = 128
            result[key] = TQExpertTensor(
                tuple(matrix(rows, width, rng, True) for _ in range(7)), width
            )
    return result


def _independent(config, weights):
    """Split the original independent attention projection tensors."""
    result = dict(weights)
    c = config['text_config']
    kw = c['linear_num_key_heads'] * c['linear_key_head_dim']
    heads, width = c['num_attention_heads'], c['head_dim']
    for name, value in weights.items():
        if name.endswith('.linear_attention.qkv.weight'):
            prefix = name.removesuffix('.qkv.weight')
            for key, value_part in zip(
                ('query', 'key', 'value'), np.split(value, [kw, 2 * kw], axis=0), strict=True
            ):
                result[prefix + f'.{key}.weight'] = value_part.copy()
            del result[name]
        elif name.endswith('.attention.query.weight'):
            pair = value.reshape(heads, 2 * width, -1)
            result[name] = pair[:, :width].reshape(heads * width, -1).copy()
            result[name.removesuffix('.query.weight') + '.gate.weight'] = (
                pair[:, width:].reshape(heads * width, -1).copy()
            )
    return result


def _identity(path):
    """Return a file's content hash and byte size."""
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': sha}


def generate(out):
    """Create the fixed model, prompt, and provenance in a new directory."""
    out = Path(out).resolve()
    out.mkdir(parents=False, exist_ok=False)
    config, weights = _predictor()
    c = config['text_config']
    c.update(
        max_position_embeddings=2048,
        num_attention_heads=24,
        num_key_value_heads=2,
        num_experts=7,
        head_dim=256,
        indexer_n_heads=4,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=64,
        rope_parameters={'rope_theta': 1e7},
        eos_token_id=31,
        moe_intermediate_size=128,
    )
    rng = np.random.default_rng(3817)
    dimension = 128
    c.update(linear_key_head_dim=dimension, linear_value_head_dim=dimension)
    prefix = 'model.block.0.linear_attention'
    for name, shape in {
        'qkv': (4 * dimension, 16),
        'gate': (2 * dimension, 16),
        'output': (16, 2 * dimension),
    }.items():
        weights[f'{prefix}.{name}.weight'] = rng.normal(0, 0.05, shape).astype(np.float32)
    convolution = np.zeros((4 * dimension, 1, 4), np.float32)
    convolution[:, 0, -1], convolution[:, 0, -2] = 0.8, 0.1
    weights[prefix + '.conv.weight'] = convolution
    weights[prefix + '.norm.weight'] = np.ones(dimension, np.float32)
    weights['model.token_embedding.weight'] = rng.uniform(0.1, 0.4, (32, 16)).astype(np.float32)
    ple_rng = np.random.default_rng(59731)
    for name, value in list(weights.items()):
        if '.position_embedding.ngram.shard.' in name:
            weights[name] = matrix(value.shape[0], value.shape[1], ple_rng, False)
    weights['model.block.0.position_embedding.ngram.weight_scale'] = np.array([1.0], np.float32)
    for prefix in ('model.block.1.attention', 'predictor.block.0.attention'):
        for name, shape in {
            'query': (24 * 2 * 256, 16),
            'key': (2 * 256, 16),
            'value': (2 * 256, 16),
            'output': (16, 24 * 256),
            'indexer.query_key': (5 * 128, 16),
        }.items():
            weights[f'{prefix}.{name}.weight'] = rng.normal(0, 0.05, shape).astype(np.float32)
        weights[f'{prefix}.indexer.query_key.weight'] = rng.uniform(
            0.01, 0.09, (5 * 128, 16)
        ).astype(np.float32)
        for name, width in {
            'query_norm': 256,
            'key_norm': 256,
            'indexer.query_norm': 128,
            'indexer.key_norm': 128,
        }.items():
            weights[f'{prefix}.{name}.weight'] = rng.normal(0, 0.02, width).astype(np.float32)
    weights = _experts(weights)
    for name, value in list(weights.items()):
        if name.endswith('.mlp.router.weight'):
            weights[name] = rng.normal(0, 0.05, (7, value.shape[-1])).astype(np.float32)
    weights = _independent(config, weights)
    Qwen4ExpConfig.from_hf_config(config)
    graph = dict(
        schema_version=1,
        architecture='qwen4_exp',
        canonical_naming=dict(
            namespace='mfq.tensor', version=1, component_roots=['model', 'predictor']
        ),
        topology=dict(text_layers=2, predictor_layers=1),
        graph=dict(kind='causal_lm', backbone='qwen4_exp'),
        components=[
            dict(kind='text', tensor_root='model', implementation='qwen4_exp', policy='decoder'),
            dict(
                kind='predictor',
                tensor_root='predictor',
                implementation='next_token_prediction',
                policy='optional',
            ),
        ],
        capabilities=['text', 'mtp'],
    )
    model = out / 'model.tq'
    ple_names = [name for name in weights if '.position_embedding.ngram.shard.' in name]
    logical_shapes = {name: [8, 4] for name in ple_names}
    save(
        model,
        FileHeader(version=2, model_arch='qwen4_exp', extra={'tq_shapes': logical_shapes}),
        {
            **weights,
            MODEL_CONFIG_ASSET: json.dumps(config).encode(),
            MODEL_GRAPH_ASSET: json.dumps(graph).encode(),
        },
    )
    header, decoded = load(model)
    if header.extra.get('tq_shapes') != logical_shapes:
        raise ValueError('TQ logical row metadata roundtrip failed')
    expert_names = [name for name in weights if '.experts.' in name]
    if len(ple_names) != 4 or not all(isinstance(decoded[name], TQTensor) for name in ple_names):
        raise ValueError('TQ PLE roundtrip failed')
    if len(expert_names) != 9 or not all(
        isinstance(decoded[name], TQExpertTensor) for name in expert_names
    ):
        raise ValueError('TQ expert roundtrip failed')
    prompt = [1, 2, 3, 4, 5]
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    (out / 'prompt.txt').write_text(','.join(map(str, prompt)) + '\n', encoding='utf-8')
    root = Path(__file__).resolve().parents[2]
    sources = {
        name: _identity(root / name)
        for name in (
            'tests/native_flash_next_fixtures.py',
            'tests/test_metal_continuous_batching.py',
            'bench/tq_smoke_fixture.py',
            'tests/numerics/qwen4_fixture.py',
        )
    }
    manifest = {
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'category': 'existing synthetic fixture adapted for native graph diagnostics; not independent numerical reference',
        'source_functions': [
            'qwen_model_fixture',
            'mtp_fixture',
            'independent_qwen_attention',
            'synthetic_tq_experts',
            'batch_model dimension=128',
            'matrix',
        ],
        'source_files': sources,
        'seeds': {'base': 3805, 'predictor': 3811, 'model': 3817, 'ple': 59731, 'experts': 2342},
        'original_context': 1024,
        'context': 2048,
        'dimension': 128,
        'prompt_ids': prompt,
        'metadata_change': 'tq_shapes records PLE logical width 4 separately from padded storage width 256',
        'format': 'MFQ v2 TQ expert and TQ PLE tables, dense model tensors',
        'native_execution': 'not_run',
        'ple_tables': ple_names,
        'expert_tables': expert_names,
        'files': {
            name: _identity(out / name) for name in ('model.tq', 'config.json', 'prompt.txt')
        },
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return out


def main():
    """Parse the output location and write one immutable fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(generate(args.out))


if __name__ == '__main__':
    main()
