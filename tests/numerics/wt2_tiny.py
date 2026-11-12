"""Freeze and cross-check the original tiny MLX WT2 quality scenarios."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np

from .case import Case, CaseWriter, save_tensor, write_result
from .wt2_ref import _file, _framework, _implementation

_FAMILIES = ('qwen3_5', 'qwen3_5_moe', 'gemma4_text', 'gemma4_moe', 'gemma4_shared_ple')


def _config(family):
    """Return the original tiny model configuration for a supported family."""
    if family not in _FAMILIES:
        raise ValueError('Unsupported tiny WT2 family')
    qwen = family.startswith('qwen')
    config = {
        'model_type': family if qwen else 'gemma4_text',
        'hidden_size': 32,
        'vocab_size': 67,
        'intermediate_size': 64,
        'num_hidden_layers': 2,
        'num_attention_heads': 2,
        'num_key_value_heads': 1,
        'head_dim': 16,
        'max_position_embeddings': 64,
        'tie_word_embeddings': False,
        'rms_norm_eps': 1e-6,
    }
    if qwen:
        config.update(
            full_attention_interval=2,
            linear_num_value_heads=2,
            linear_num_key_heads=1,
            linear_key_head_dim=32,
            linear_value_head_dim=32,
            linear_conv_kernel_dim=4,
            rope_parameters={
                'rope_theta': 10000.0,
                'partial_rotary_factor': 0.5,
                'mrope_section': [1, 1, 2],
            },
        )
        if family.endswith('_moe'):
            config.update(
                num_experts=4,
                num_experts_per_tok=2,
                moe_intermediate_size=64,
                shared_expert_intermediate_size=32,
            )
    else:
        config.update(
            global_head_dim=16,
            num_global_key_value_heads=1,
            num_kv_shared_layers=0,
            hidden_size_per_layer_input=0,
            sliding_window=16,
            layer_types=['sliding_attention', 'full_attention'],
            enable_moe_block=False,
            use_double_wide_mlp=False,
            final_logit_softcapping=30.0,
        )
        if family.endswith('_moe'):
            config.update(
                enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=32
            )
        if family.endswith('_shared_ple'):
            config.update(
                num_hidden_layers=4,
                num_kv_shared_layers=2,
                hidden_size_per_layer_input=8,
                vocab_size_per_layer_input=67,
                layer_types=['sliding_attention', 'full_attention'] * 2,
            )
    return config


def _model(config, family):
    """Construct the original upstream MLX model class for this family."""
    from mlx_lm.models import gemma4_text, qwen3_5

    if family.startswith('qwen'):
        return qwen3_5.TextModel(qwen3_5.TextModelArgs.from_dict(config))
    return gemma4_text.Model(gemma4_text.ModelArgs.from_dict(config))


def _write_weights(path, values):
    """Write original F32 or BF16 raw tensors in the source test's safetensors form."""
    header, blobs, offset = {}, [], 0
    for name, (dtype, value) in values.items():
        value = np.ascontiguousarray(value)
        raw = value.tobytes()
        header[name] = {
            'dtype': dtype,
            'shape': list(value.shape),
            'data_offsets': [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header).encode()
    with Path(path).open('xb') as stream:
        stream.write(struct.pack('<Q', len(encoded)))
        stream.write(encoded)
        for raw in blobs:
            stream.write(raw)


def _read_weights(path):
    """Restore the upstream parameter bits without converting BF16 through F32."""
    import mlx.core as mx

    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError('Truncated tiny upstream checkpoint')
    size = struct.unpack_from('<Q', data)[0]
    if size > len(data) - 8:
        raise ValueError('Truncated tiny upstream tensor header')
    header = json.loads(data[8 : 8 + size])
    body = memoryview(data)[8 + size :]
    values = []
    for name, spec in header.items():
        dtype = spec['dtype']
        if dtype not in ('F32', 'BF16'):
            raise ValueError('Unsupported tiny upstream parameter precision')
        start, end = spec['data_offsets']
        shape = spec['shape']
        storage = np.dtype('<f4' if dtype == 'F32' else '<u2')
        if (
            start < 0
            or end < start
            or end > len(body)
            or end - start != math.prod(shape) * storage.itemsize
        ):
            raise ValueError('Invalid tiny upstream parameter range')
        array = mx.array(np.frombuffer(body[start:end], dtype=storage).reshape(shape).copy())
        if dtype == 'BF16':
            array = array.view(mx.bfloat16)
        values.append((name, array))
    return values


def _verify(case):
    """Validate fixed model assets and the complete MLX implementation identity."""
    verified = Case.load(case.root / 'case.json')
    if verified.root != case.root.resolve() or verified.data != case.data:
        raise ValueError('Tiny WT2 case changed since it was loaded')
    if (
        case.data['op'] != 'wt2_tiny'
        or case.data['source'].get('generator') != 'tests/numerics/wt2_tiny.py'
        or case.data.get('actual_role') != 'candidate'
    ):
        raise ValueError('Unsupported tiny WT2 case')
    config = case.data['config']
    if json.loads(case.asset('config')) != _config(config['family']):
        raise ValueError('Tiny model configuration differs from the source scenario')
    for name in ('upstream', 'checkpoint'):
        case.asset(name)
    checkpoint = case.asset_path('checkpoint')
    configuration = case.asset_path('config')
    if (
        configuration.parent != checkpoint.parent
        or configuration.name != 'config.json'
        or checkpoint.name != 'model.safetensors'
    ):
        raise ValueError('Tiny checkpoint assets must retain standard HF paths')
    identity = _implementation()
    if identity != case.data['source']['implementation']:
        raise ValueError('MLX implementation differs from the fixed case')
    tokens = case.tensor('tokens')
    if (
        tokens.shape != (2, 32)
        or tokens.dtype != np.dtype('<i4')
        or np.any(tokens < 0)
        or np.any(tokens >= 67)
    ):
        raise ValueError('Tiny WT2 fixed tokens have invalid geometry')
    return checkpoint.parent, tokens, identity


def reference(case: Case, root: Path) -> Path:
    """Run the saved original upstream model to obtain target log-probabilities."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    _, tokens, identity = _verify(case)
    config = case.data['config']
    model = _model(_config(config['family']), config['family'])
    values = _read_weights(case.asset_path('upstream'))
    expected = dict(tree_flatten(model.parameters()))
    stored = dict(values)
    dtype = getattr(mx, config['dtype'])
    if (
        len(stored) != len(values)
        or set(stored) != set(expected)
        or any(
            tuple(stored[name].shape) != tuple(expected[name].shape) or stored[name].dtype != dtype
            for name in expected
        )
    ):
        raise ValueError('Tiny upstream parameter names, shapes or dtypes differ from the model')
    model.load_weights(values, strict=True)
    loaded = dict(tree_flatten(model.parameters()))
    if set(loaded) != set(stored) or any(
        tuple(loaded[name].shape) != tuple(stored[name].shape) or loaded[name].dtype != dtype
        for name in stored
    ):
        raise ValueError('Tiny upstream model did not retain the fixed parameters')
    model.eval()
    mx.eval(model.parameters())
    logits = np.concatenate(
        [np.asarray(model(mx.array(ids[None]))[:, 16:31].astype(mx.float32)) for ids in tokens]
    )
    logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
    target = np.take_along_axis(logp, tokens[:, 17:, None], -1).squeeze(-1)
    outputs = {
        'target_logp': target,
        'quality_equal': np.asarray(True, dtype=np.bool_),
        'finite_nll': np.asarray(True, dtype=np.bool_),
        'modes_executed': np.asarray(True, dtype=np.bool_),
    }
    if config['fault_injection']:
        outputs['fallback_equal'] = np.asarray(True, dtype=np.bool_)
    if _verify(case)[2] != identity:
        raise ValueError('Tiny model or MLX implementation changed during upstream evaluation')
    specs = {name: save_tensor(root, name, value) for name, value in outputs.items()}
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'mlx_lm upstream model',
            'device': 'mlx',
            'category': 'upstream_execution_consistency',
            'independence': 'upstream model quality expectation; no native execution',
            'implementation': identity,
            'generator_sha256': _file(__file__)['sha256'],
            'fixed_assets': case.data['assets'],
        },
    )


def _quality(source, tokens, root, name, layerwise):
    """Run one production MLX quality path and retain its original quality file."""
    from tyloinfer.evaluation.reference import ReferenceWriter

    module = _framework()
    scratch = root / (name + '_scratch')
    scratch.mkdir()
    path = root / (name + '.logits')
    with ReferenceWriter(path, tokens, 67) as writer:
        nll, metadata = module.generate(source, tokens, writer, scratch, layerwise=layerwise)
        targets = np.array(writer.targets)
        writer.commit()
    return path, targets, nll, metadata


def candidate(case: Case, root: Path) -> Path:
    """Run resident, layerwise, and the original controlled fallback quality paths."""
    from tyloinfer.evaluation.source import ReferenceSource

    directory, tokens, identity = _verify(case)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    source = ReferenceSource(directory)
    layer, targets, nll_layer, layer_info = _quality(source, tokens, root, 'layerwise', True)
    resident, _, nll_resident, resident_info = _quality(source, tokens, root, 'resident', False)
    outputs = {
        'target_logp': targets,
        'quality_equal': np.asarray(layer.read_bytes() == resident.read_bytes(), dtype=np.bool_),
        'finite_nll': np.asarray(
            np.isfinite(nll_layer)
            and np.isfinite(nll_resident)
            and layer_info.get('runtime') == resident_info.get('runtime') == 'mlx',
            dtype=np.bool_,
        ),
        'modes_executed': np.asarray(
            layer_info.get('layerwise') is True and resident_info.get('layerwise') is False,
            dtype=np.bool_,
        ),
    }
    evidence = {
        'layerwise': layer_info,
        'resident': resident_info,
        'quality_files': {'layerwise': _file(layer), 'resident': _file(resident)},
    }
    if case.data['config']['fault_injection']:
        module = _framework()
        original = module.SourceWeights.layer_weights
        failures = []

        def pressure(self, index, gemma):
            """Inject the original one-time resident layer allocation failure."""
            if index == 1 and not failures:
                failures.append(index)
                raise RuntimeError('allocation failed: out of memory')
            return original(self, index, gemma)

        module.SourceWeights.layer_weights = pressure
        try:
            fallback, _, nll, info = _quality(source, tokens, root, 'fallback', False)
        finally:
            module.SourceWeights.layer_weights = original
        outputs['fallback_equal'] = np.asarray(
            failures == [1]
            and info.get('layerwise') is True
            and info.get('runtime') == 'mlx'
            and np.isfinite(nll)
            and fallback.read_bytes() == layer.read_bytes(),
            dtype=np.bool_,
        )
        evidence['fallback'] = {'events': failures, 'metadata': info, 'file': _file(fallback)}
    if _verify(case)[2] != identity:
        raise ValueError('Tiny model or MLX implementation changed during candidate execution')
    specs = {name: save_tensor(root, name, value) for name, value in outputs.items()}
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'TyloInfer MLX quality path',
            'device': 'mlx',
            'category': 'python_candidate',
            'native_executed': False,
            'independence': 'upstream execution consistency, not native',
            'implementation': identity,
            'execution': evidence,
            'generator_sha256': _file(__file__)['sha256'],
            'fixed_assets': case.data['assets'],
        },
        role='candidate',
    )


def generate(root: Path) -> list[Path]:
    """Freeze and execute the ten original tiny MLX quality scenarios."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    paths = []
    for family in _FAMILIES:
        for dtype in ('float32', 'bfloat16'):
            config = _config(family)
            model = _model(config, family)
            mx.random.seed(123)
            original, exported = {}, {}
            for name, value in tree_flatten(model.parameters()):
                if value.ndim >= 2:
                    value = mx.random.normal(value.shape) * 0.05
                value = value.astype(getattr(mx, dtype))
                model.load_weights([(name, value)], strict=False)
                saved = (
                    np.asarray(value.view(mx.uint16)) if dtype == 'bfloat16' else np.asarray(value)
                )
                original[name] = ('BF16' if dtype == 'bfloat16' else 'F32', saved)
                if family.startswith('qwen'):
                    if name.endswith(
                        (
                            'input_layernorm.weight',
                            'post_attention_layernorm.weight',
                            'q_norm.weight',
                            'k_norm.weight',
                            'model.norm.weight',
                        )
                    ):
                        value = value - 1
                    if name.endswith('conv1d.weight'):
                        value = value.transpose(0, 2, 1)
                saved = (
                    np.asarray(value.view(mx.uint16)) if dtype == 'bfloat16' else np.asarray(value)
                )
                exported[name] = ('BF16' if dtype == 'bfloat16' else 'F32', saved)
            for index in range(config['num_hidden_layers']):
                base = f'model.layers.{index}.' + (
                    'mlp.switch_mlp' if family.startswith('qwen') else 'experts.switch_glu'
                )
                if base + '.gate_proj.weight' in exported:
                    gate_type, gate = exported.pop(base + '.gate_proj.weight')
                    _, up = exported.pop(base + '.up_proj.weight')
                    head = base.replace('switch_mlp', 'experts').removesuffix('.switch_glu')
                    exported[head + '.gate_up_proj'] = (
                        gate_type,
                        np.concatenate((gate, up), axis=-2),
                    )
                    exported[head + '.down_proj'] = exported.pop(base + '.down_proj.weight')
            model.eval()
            mx.eval(model.parameters())
            identity = _implementation()
            writer = CaseWriter(
                root,
                f'{family}_{dtype}',
                'wt2_tiny',
                {
                    'family': family,
                    'dtype': dtype,
                    'context': 32,
                    'chunks': 2,
                    'vocab': 67,
                    'fault_injection': family == 'qwen3_5' and dtype == 'float32',
                    'category': 'upstream_execution_consistency',
                    'native_limitation': 'Gemma shared KV and per-layer input are unsupported'
                    if family == 'gemma4_shared_ple'
                    else None,
                },
                {
                    'generator': 'tests/numerics/wt2_tiny.py',
                    'generator_sha256': _file(__file__)['sha256'],
                    'test': 'tests/test_native_wt2_reference.py::test_mlx_layerwise_and_resident_match_original_precision_graph',
                    'test_sha256': _file(
                        Path(__file__).parents[1] / 'test_native_wt2_reference.py'
                    )['sha256'],
                    'implementation': identity,
                },
                actual_role='candidate',
            )
            checkpoint = writer.root / 'checkpoint'
            checkpoint.mkdir()
            config_path = checkpoint / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            hf_path = checkpoint / 'model.safetensors'
            upstream_path = writer.root / 'upstream.safetensors'
            _write_weights(hf_path, exported)
            _write_weights(upstream_path, original)
            writer.asset('config', config_path, encoding='json')
            writer.asset('checkpoint', hf_path, encoding='safetensors')
            writer.asset('upstream', upstream_path, encoding='safetensors')
            writer.input(
                'tokens', np.random.default_rng(42).integers(0, 67, (2, 32), dtype=np.int32)
            )
            writer.output(
                'target_logp',
                [2, 15],
                'float32',
                {'kind': 'close', 'atol': 0.005 if dtype == 'bfloat16' else 5e-5, 'rtol': 1e-7},
            )
            writer.output('quality_equal', [], 'bool', {'kind': 'exact'})
            writer.output('finite_nll', [], 'bool', {'kind': 'exact'})
            writer.output('modes_executed', [], 'bool', {'kind': 'exact'})
            if family == 'qwen3_5' and dtype == 'float32':
                writer.output('fallback_equal', [], 'bool', {'kind': 'exact'})
            if _implementation() != identity:
                raise ValueError('MLX implementation changed while freezing tiny model')
            case = writer.finish()
            reference(case, case.root / 'reference')
            candidate(case, case.root / 'candidate')
            paths.append(case.root / 'case.json')
    return paths
