"""Create a fixed synthetic DeepSeek V4 text model for native execution checks.

This one-layer model is not a pretrained quality reference and does not include
vision or prediction. Compressed attention is optional and remains unverified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from tyloinfer.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import save
from tyloinfer.formats.tq import TQExpertTensor, TQTensor


def _identity(path: Path) -> dict:
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def _config(context: int, ratio: int) -> dict:
    return {
        'model_type': 'deepseek_v4',
        'vocab_size': 32,
        'hidden_size': 4096,
        'num_hidden_layers': 1,
        'num_attention_heads': 64,
        'num_key_value_heads': 1,
        'head_dim': 512,
        'max_position_embeddings': context,
        'rope_theta': 10000.0,
        'partial_rotary_factor': 0.125,
        'qk_rope_head_dim': 64,
        'rms_norm_eps': 1e-6,
        'compress_ratios': [ratio],
        'compress_rope_theta': 10000.0,
        'num_hash_layers': 0,
        'num_experts': 256,
        'num_experts_per_tok': 6,
        'moe_intermediate_size': 2048,
        'n_shared_experts': 1,
        'shared_expert_intermediate_size': 2048,
        'q_lora_rank': 1024,
        'index_head_dim': 128,
        'index_n_heads': 64,
        'index_topk': 512,
        'o_groups': 8,
        'o_lora_rank': 1024,
        'hc_mult': 4,
        'hc_sinkhorn_iters': 20,
        'hc_eps': 1e-6,
        'scoring_func': 'sqrtsoftplus',
        'norm_topk_prob': True,
        'routed_scaling_factor': 1.0,
        'swiglu_limit': 10.0,
        'tie_word_embeddings': False,
        'eos_token_id': 31,
    }


def _graph() -> dict:
    return {
        'schema_version': 1,
        'architecture': 'deepseek_v4',
        'canonical_naming': {
            'namespace': 'mfq.tensor', 'version': 1, 'component_roots': ['model'],
        },
        'topology': {'text_layers': 1, 'vision_layers': 0, 'predictor_layers': 0},
        'graph': {'kind': 'causal_lm', 'backbone': 'deepseek_v4'},
        'components': [
            {'kind': 'text', 'tensor_root': 'model', 'implementation': 'deepseek_v4',
             'policy': 'decoder'},
        ],
        'capabilities': ['text'],
    }


def _random(rng: np.random.Generator, shape: tuple[int, ...], scale: float,
            dtype: type = np.float32) -> np.ndarray:
    # Generate bounded F32 chunks before storing in the requested precision.
    value = np.empty(shape, dtype=dtype)
    flat = value.reshape(-1)
    for start in range(0, flat.size, 1 << 20):
        part = flat[start:start + (1 << 20)]
        sample = rng.random(part.size, dtype=np.float32)
        sample *= 2.0 * scale
        sample -= scale
        part[:] = sample
    return value


def _expert(rng: np.random.Generator, rows: int, width: int) -> TQTensor:
    # Follow bench/tq_smoke_fixture.py's fixed-book, column-scale payload.
    words = width // 32
    descriptors = np.arange(rows, dtype=np.uint64) * np.uint64(words)
    descriptors |= np.uint64(np.float16(0.002).view(np.uint16)) << np.uint64(40)
    scales = rng.uniform(0.8, 1.2, width).astype(np.float16)
    packed = rng.integers(0, 256, rows * words * 4, dtype=np.uint8)
    return TQTensor(rows, width, 16, True, descriptors, scales, packed)


def _bank(rng: np.random.Generator, rows: int, width: int) -> TQExpertTensor:
    return TQExpertTensor(tuple(_expert(rng, rows, width) for _ in range(256)), width)


def _weights(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    p = 'model.block.0.'
    weights = {
        'model.token_embedding.weight': _random(rng, (32, 4096), 0.05, np.float16),
        'model.output.weight': _random(rng, (32, 4096), 0.01, np.float16),
        'model.output_norm.weight': np.ones(4096, dtype=np.float32),
        'model.mhc.output.function': _random(rng, (4, 16384), 0.001),
        'model.mhc.output.scale': np.ones(1, dtype=np.float32),
        'model.mhc.output.base': np.full(4, 0.01, dtype=np.float32),
        p + 'attention.norm.weight': np.ones(4096, dtype=np.float32),
        p + 'mlp.norm.weight': np.ones(4096, dtype=np.float32),
        p + 'attention.query_a_norm.weight': np.ones(1024, dtype=np.float32),
        p + 'attention.key_value_a_norm.weight': np.ones(512, dtype=np.float32),
        p + 'attention.sink': np.full(64, -1.0, dtype=np.float32),
    }
    shapes = {
        'attention.query_a.weight': (1024, 4096),
        'attention.query_b.weight': (32768, 1024),
        'attention.key_value_a.weight': (512, 4096),
        'attention.output_a.weight': (8192, 4096),
        'attention.output_b.weight': (4096, 8192),
        'mlp.router.weight': (256, 4096),
        'mlp.shared_expert.gate.weight': (2048, 4096),
        'mlp.shared_expert.up.weight': (2048, 4096),
        'mlp.shared_expert.down.weight': (4096, 2048),
    }
    for name, shape in shapes.items():
        dtype = np.float32 if name == 'mlp.router.weight' else np.float16
        weights[p + name] = _random(rng, shape, 0.01, dtype)
    for block in ('attention', 'mlp'):
        prefix = p + block + '.mhc.pre.'
        weights[prefix + 'function'] = _random(rng, (24, 16384), 0.001)
        weights[prefix + 'scale'] = np.full(3, 0.1, dtype=np.float32)
        weights[prefix + 'base'] = np.full(24, 0.01, dtype=np.float32)
    for role, rows, width in (('gate', 2048, 4096), ('up', 2048, 4096),
                              ('down', 4096, 2048)):
        weights[p + f'mlp.experts.{role}.weight'] = _bank(rng, rows, width)
    return weights


def _compressed(weights: dict, seed: int) -> None:
    rng = np.random.default_rng(seed)
    p = 'model.block.0.attention.'
    shapes = {
        'compressor.key_value.weight': (1024, 4096),
        'compressor.gate.weight': (1024, 4096),
        'indexer.compressor.key_value.weight': (256, 4096),
        'indexer.compressor.gate.weight': (256, 4096),
        'indexer.query.weight': (8192, 1024),
        'indexer.score.weight': (64, 4096),
    }
    for name, shape in shapes.items():
        dtype = np.float16 if name == 'indexer.query.weight' else np.float32
        weights[p + name] = _random(rng, shape, 0.01, dtype)
    weights[p + 'compressor.position'] = np.full((4, 1024), 0.01, dtype=np.float32)
    weights[p + 'compressor.norm.weight'] = np.ones(512, dtype=np.float32)
    weights[p + 'indexer.compressor.position'] = np.full((4, 256), 0.01, dtype=np.float32)
    weights[p + 'indexer.compressor.norm.weight'] = np.ones(128, dtype=np.float32)


def generate(out: Path, context: int = 256, seed: int = 4107, ratio: int = 4) -> Path:
    """Write a fixed synthetic V4 model, prompt, and provenance to a new directory."""
    if context not in (128, 256, 4096):
        raise ValueError('context must be 128, 256, or 4096')
    if ratio not in (0, 4):
        raise ValueError('ratio must be 0 or 4')
    out = Path(out).resolve()
    out.mkdir(parents=False, exist_ok=False)
    config = _config(context, ratio)
    graph = _graph()
    weights = _weights(seed)
    if ratio == 4:
        _compressed(weights, seed + 1)
    tensors = {
        name: {'shape': list(value.shape), 'dtype': 'TQ' if isinstance(value, TQExpertTensor)
               else {'float16': 'F16', 'float32': 'F32'}[value.dtype.name]}
        for name, value in weights.items()
    }
    model = out / 'model.tq'
    save(model, FileHeader(version=2, model_arch='deepseek_v4'), {
        **weights,
        MODEL_CONFIG_ASSET: json.dumps(config, sort_keys=True).encode('utf-8'),
        MODEL_GRAPH_ASSET: json.dumps(graph, sort_keys=True).encode('utf-8'),
    })
    prompt = [1, 2, 3, 4, 5, 6, 7, 8]
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    (out / 'prompt.txt').write_text(','.join(map(str, prompt)) + '\n', encoding='utf-8')
    manifest = {
        'category': 'synthetic native execution and state consistency; not pretrained quality',
        'scope': 'one synthetic text layer with local attention, HC, and routed/shared experts',
        'compressed_attention': {
            'ratio': ratio,
            'status': 'not_run',
            'minimum_tokens_for_window': 4 if ratio == 4 else None,
            'minimum_tokens_for_query_score_top512': 2052 if ratio == 4 else None,
        },
        'excluded': ['vision', 'MTP', 'pretrained quality'],
        'native_execution': 'not_run',
        'seed': seed,
        'compressed_seed': seed + 1 if ratio == 4 else None,
        'context': context,
        'prompt_ids': prompt,
        'config': config,
        'tensors': tensors,
        'expert_encoding': 'RTRELL05 fixed-book, one bit per weight, column scales',
        'hc_precision': 'native HC pre/post require F16 activations; HC parameters and mixes use F32',
        'generator': _identity(Path(__file__).resolve()),
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
        'files': {name: _identity(out / name) for name in ('model.tq', 'config.json', 'prompt.txt')},
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return out


def main() -> None:
    """Parse fixture options and write the model without overwriting an existing directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--context', type=int, choices=(128, 256, 4096), default=256)
    parser.add_argument('--ratio', type=int, choices=(0, 4), default=4)
    parser.add_argument('--seed', type=int, default=4107)
    args = parser.parse_args()
    print(generate(args.out, args.context, args.seed, args.ratio))


if __name__ == '__main__':
    main()
