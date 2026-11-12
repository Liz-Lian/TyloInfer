"""Freeze independent HF RADIO versus local Python reference crosschecks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import sys
import tempfile
import types
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

from . import nemotron_vision_ref as vision
from .case import Case, CaseWriter, save_tensor, write_result

_PACKAGE = 'tyloinfer_hf_radio_reference'
_HF_ROOT = None
_HF_SOURCES = None


def _hash(path):
    """Hash a source or checkpoint file without reading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_path(root, value):
    """Validate a relative checkpoint path while preserving symlink targets."""
    if not isinstance(value, (str, Path)):
        raise ValueError('Checkpoint path must be a relative path')
    relative = Path(value)
    if not relative.parts or relative.anchor or '..' in relative.parts:
        raise ValueError('Checkpoint path must stay inside its directory')
    path = Path(os.path.abspath(root / relative))
    if not path.is_relative_to(root):
        raise ValueError('Checkpoint path escapes its directory')
    return path


def _source_files(root):
    """Identify all Python sources available to the checkpoint package."""
    result = {}
    for path in sorted(root.rglob('*.py')):
        logical = _logical_path(root, path.relative_to(root))
        result[logical.relative_to(root).as_posix()] = _hash(logical)
    if 'modeling_radio.py' not in result:
        raise ValueError('HF checkpoint is missing modeling_radio.py')
    return result


def _module_files(root):
    """Verify the actual checkpoint package imports and identify their files."""
    result = {}
    for name, module in tuple(sys.modules.items()):
        if name == _PACKAGE:
            paths = list(getattr(module, '__path__', ()))
            if len(paths) != 1 or Path(paths[0]).resolve() != root:
                raise ValueError('HF package was loaded from another checkpoint')
        elif name.startswith(_PACKAGE + '.'):
            filename = getattr(module, '__file__', None)
            if not filename:
                raise ValueError(f'HF module has no verifiable source: {name}')
            path = Path(os.path.abspath(filename))
            if not path.is_relative_to(root) or path.suffix != '.py':
                raise ValueError(f'HF module is outside the checkpoint: {name}')
            result[name] = {'path': path.relative_to(root).as_posix(), 'sha256': _hash(path)}
    return dict(sorted(result.items()))


@contextmanager
def _hf(root):
    """Load checkpoint code without external cache writes or stale bytecode reads."""
    global _HF_ROOT, _HF_SOURCES
    root = Path(root).resolve(strict=True)
    sources = _source_files(root)
    loaded = _module_files(root)
    if _HF_ROOT is None and (_PACKAGE in sys.modules or loaded):
        raise ValueError('HF package was already loaded outside this adapter')
    if _HF_ROOT is not None and (root != _HF_ROOT or sources != _HF_SOURCES):
        raise ValueError('HF checkpoint source directory or files changed')
    old_bytecode, old_prefix = sys.dont_write_bytecode, sys.pycache_prefix
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(Path(tempfile.gettempdir()) / f'hf-radio-{uuid.uuid4().hex}')
    try:
        if _HF_ROOT is None:
            package = types.ModuleType(_PACKAGE)
            package.__path__ = [str(root)]
            sys.modules[_PACKAGE] = package
        module = importlib.import_module(f'{_PACKAGE}.modeling_radio')
        _module_files(root)
        if _source_files(root) != sources:
            raise ValueError('HF source changed during import')
        _HF_ROOT, _HF_SOURCES = root, sources
        yield module, sources
        if _source_files(root) != sources:
            raise ValueError('HF source changed during computation')
        _module_files(root)
    finally:
        sys.dont_write_bytecode, sys.pycache_prefix = old_bytecode, old_prefix


def _versions():
    """Record the installed libraries used by both RADIO reference paths."""
    return {
        'torch': str(torch.__version__),
        'numpy': np.__version__,
        'transformers': importlib.metadata.version('transformers'),
        'safetensors': importlib.metadata.version('safetensors'),
    }


def _config(module, values):
    """Construct the exact RadioConfig used by the original HF checks."""
    keys = (
        'hidden_size',
        'num_hidden_layers',
        'num_attention_heads',
        'mlp_ratio',
        'num_channels',
        'patch_size',
        'max_img_size',
        'num_cls_tokens',
        'num_registers',
        'layer_norm_eps',
        'hidden_act',
        'qkv_bias',
        'use_swiglu_ffn',
        'layerscale_value',
    )
    return module.RadioConfig(
        **{key: values[key] for key in keys}, summary_idxs=[0, 1], drop_path_rate=0.0
    )


def _weight(case, name):
    """Read one fixed tensor as a CPU float64 Torch value."""
    return torch.tensor(case.tensor(name), dtype=torch.float64)


def _block_state(case, prefix, destination):
    """Map canonical combined-QKV block weights to the upstream RADIO names."""
    weight = _weight(case, prefix + 'attention.qkv.weight')
    bias = _weight(case, prefix + 'attention.qkv.bias')
    width = weight.shape[1]
    result = {}
    for index, label in enumerate(('query', 'key', 'value')):
        result[destination + f'attention.attention.{label}.weight'] = weight[
            index * width : (index + 1) * width
        ]
        result[destination + f'attention.attention.{label}.bias'] = bias[
            index * width : (index + 1) * width
        ]
    for local, remote in (
        ('attention.output.weight', 'attention.output.dense.weight'),
        ('attention.output.bias', 'attention.output.dense.bias'),
        ('norm1.weight', 'norm1.weight'),
        ('norm1.bias', 'norm1.bias'),
        ('norm2.weight', 'norm2.weight'),
        ('norm2.bias', 'norm2.bias'),
        ('mlp.up.weight', 'mlp.fc1.weight'),
        ('mlp.up.bias', 'mlp.fc1.bias'),
        ('mlp.down.weight', 'mlp.fc2.weight'),
        ('mlp.down.bias', 'mlp.fc2.bias'),
    ):
        result[destination + remote] = _weight(case, prefix + local)
    return result


def _hf_result(case, module):
    """Execute the original HF model or layer against persisted inputs only."""
    config = _config(module, case.data['config']['vision_config'])
    if case.data['op'] == 'radio_features':
        model = module.RadioModel(config).to(torch.float64)
        model.make_preprocessor_external()
        state = {
            'embeddings.patch_projection.weight': _weight(case, 'vision.patch_embedding.weight'),
            'embeddings.position_embedding': _weight(case, 'vision.position_embedding.weight'),
            'embeddings.cls_register_token': _weight(case, 'vision.cls_register_token.weight'),
        }
        for index in range(2):
            state.update(_block_state(case, f'vision.block.{index}.', f'encoder.layer.{index}.'))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(
            not any(item in key for item in ('input_conditioner', 'layer_scale', 'summary_idxs'))
            for key in missing
        ):
            raise ValueError(f'HF RADIO state mismatch: missing={missing}, unexpected={unexpected}')
        model.eval()
        pixels = _weight(case, 'pixels')
        pieces, offset = [], 0
        with torch.no_grad():
            for rows, columns in case.data['config']['image_grids']:
                count = rows * columns
                image = pixels[offset : offset + count].reshape(rows, columns, 3, 2, 2)
                image = image.permute(2, 0, 3, 1, 4).reshape(1, 3, rows * 2, columns * 2)
                pieces.append(model(pixel_values=image).features[0])
                offset += count
        return torch.cat(pieces, dim=0).contiguous().numpy()
    layer = module.RadioLayer(config).to(torch.float64)
    missing, unexpected = layer.load_state_dict(
        _block_state(case, 'vision.block.0.', ''), strict=False
    )
    if unexpected or any('layer_scale' not in key for key in missing):
        raise ValueError(f'HF RADIO layer mismatch: missing={missing}, unexpected={unexpected}')
    layer.eval()
    with torch.no_grad():
        return layer(_weight(case, 'hidden')).detach().contiguous().numpy()


def reference(case: Case, root: Path, hf_root: Path) -> Path:
    """Recompute the upstream HF expected output from a fixed crosscheck case."""
    if (
        case.data['source'].get('generator') != 'tests/numerics/hf_ref.py'
        or case.data['op'] not in {'radio_features', 'radio_layer'}
        or case.data.get('actual_role') != 'candidate'
    ):
        raise ValueError('Unsupported HF RADIO crosscheck')
    with _hf(hf_root) as (module, sources):
        if sources != case.data['source']['hf_sources_sha256']:
            raise ValueError('HF source identity differs from the fixed case')
        loaded = _module_files(Path(hf_root).resolve())
        if any(
            loaded.get(name) != item
            for name, item in case.data['source']['hf_initial_modules'].items()
        ):
            raise ValueError('HF loaded module identity differs from the fixed case')
        versions = _versions()
        if versions != case.data['source']['versions']:
            raise ValueError('HF reference library versions differ from the fixed case')
        output = _hf_result(case, module)
        closure = _module_files(Path(hf_root).resolve())
        if _source_files(Path(hf_root).resolve()) != sources:
            raise ValueError('HF sources changed during reference calculation')
        spec = save_tensor(root, 'output', output)
        return write_result(
            case,
            root,
            {'output': spec},
            {
                'framework': 'Hugging Face RADIO',
                'device': 'cpu',
                'arithmetic': 'float64',
                'category': 'python_reference_crosscheck',
                'native_executed': False,
                'versions': versions,
                'hf_modules': closure,
                'generator_sha256': _hash(__file__),
            },
        )


def candidate(case: Case, root: Path) -> Path:
    """Recompute the local Python candidate from the fixed inputs, without HF output."""
    if (
        case.data['source'].get('generator') != 'tests/numerics/hf_ref.py'
        or case.data.get('actual_role') != 'candidate'
    ):
        raise ValueError('Unsupported HF RADIO crosscheck')
    outputs = vision.compute(case)
    specs = {name: save_tensor(root, name, value) for name, value in outputs.items()}
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'TyloInfer Python RADIO reference',
            'device': 'cpu',
            'arithmetic': 'float64',
            'category': 'python_reference_crosscheck',
            'native_executed': False,
            'torch_version': str(torch.__version__),
            'numpy_version': np.__version__,
            'generator_sha256': _hash(__file__),
            'candidate_source_sha256': _hash(vision.__file__),
        },
        role='candidate',
    )


def _layer_inputs(root):
    """Read and identify only the twelve original real RADIO layer tensors."""
    from safetensors import safe_open

    index_path = root / 'model.safetensors.index.json'
    index_hash = _hash(index_path)
    index = json.loads(index_path.read_text(encoding='utf-8'))['weight_map']
    base = 'vision_model.radio_model.model.blocks.0.'
    names = {
        'attention.qkv.weight': 'attn.qkv.weight',
        'attention.qkv.bias': 'attn.qkv.bias',
        'attention.output.weight': 'attn.proj.weight',
        'attention.output.bias': 'attn.proj.bias',
        'norm1.weight': 'norm1.weight',
        'norm1.bias': 'norm1.bias',
        'norm2.weight': 'norm2.weight',
        'norm2.bias': 'norm2.bias',
        'mlp.up.weight': 'mlp.fc1.weight',
        'mlp.up.bias': 'mlp.fc1.bias',
        'mlp.down.weight': 'mlp.fc2.weight',
        'mlp.down.bias': 'mlp.fc2.bias',
    }
    values, records, shard_hashes, shards = {}, {}, {}, {}
    with ExitStack() as stack:
        for canonical, suffix in names.items():
            name = base + suffix
            shard_name = index[name]
            path = _logical_path(root, shard_name)
            if shard_name not in shards:
                shard_hashes[shard_name] = _hash(path)
                shards[shard_name] = stack.enter_context(
                    safe_open(str(path), framework='pt', device='cpu')
                )
            tensor = shards[shard_name].get_tensor(name).contiguous()
            raw = tensor.view(torch.uint8).numpy().tobytes()
            records[canonical] = {
                'name': name,
                'shard': shard_name,
                'dtype': str(tensor.dtype),
                'shape': list(tensor.shape),
                'sha256': hashlib.sha256(raw).hexdigest(),
            }
            values['vision.block.0.' + canonical] = tensor.to(torch.float64).numpy()
    if values['vision.block.0.norm1.weight'].shape != (1280,) or values[
        'vision.block.0.attention.qkv.weight'
    ].shape != (3840, 1280):
        raise ValueError('Real RADIO layer geometry differs from the original test')
    for shard, digest in shard_hashes.items():
        if _hash(_logical_path(root, shard)) != digest:
            raise ValueError('Checkpoint shard changed while reading tensors')
    if _hash(index_path) != index_hash:
        raise ValueError('Checkpoint index changed while reading tensors')
    return values, {'index_sha256': index_hash, 'shards_sha256': shard_hashes, 'tensors': records}


def generate(root: Path, hf_root: Path, suite: str = 'synthetic') -> list[Path]:
    """Freeze one original RADIO crosscheck and export HF and Python results."""
    if suite not in {'synthetic', 'layer'}:
        raise ValueError(f'Unknown HF RADIO suite: {suite}')
    hf_root = Path(hf_root).resolve(strict=True)
    with _hf(hf_root) as (_, sources):
        versions = _versions()
        if suite == 'synthetic':
            config, weights = vision.fixture()
            config['image_grids'] = [[4, 6], [6, 2]]
            pixels = np.asarray(
                [((i * 37 + 11) % 101) / 64.0 for i in range(36 * 12)], dtype=np.float64
            ).reshape(36, 12)
            values = {**weights, 'pixels': pixels}
            op, shape, test, checkpoint = (
                'radio_features',
                (36, 64),
                'test_reference_matches_hf_radio_model',
                None,
            )
        else:
            values, checkpoint = _layer_inputs(hf_root)
            generator = torch.Generator(device='cpu').manual_seed(7)
            values['hidden'] = torch.randn(
                1, 31, 1280, dtype=torch.float64, generator=generator
            ).numpy()
            config = {
                'heads': 16,
                'vision_config': {
                    'hidden_size': 1280,
                    'num_hidden_layers': 1,
                    'num_attention_heads': 16,
                    'mlp_ratio': 4.0,
                    'num_channels': 3,
                    'patch_size': 16,
                    'max_img_size': 2048,
                    'num_cls_tokens': 3,
                    'num_registers': 7,
                    'layer_norm_eps': 1e-6,
                    'hidden_act': 'gelu',
                    'qkv_bias': True,
                    'use_swiglu_ffn': False,
                    'layerscale_value': 1.0,
                },
            }
            op, shape, test = (
                'radio_layer',
                (1, 31, 1280),
                'test_reference_matches_one_real_encoder_layer',
            )
        source_path = Path(__file__).parents[1] / 'test_cuda_nemotron_vision.py'
        writer = CaseWriter(
            root,
            f'hf_radio_{suite}',
            op,
            {
                **config,
                'category': 'python_reference_crosscheck',
                'native_executed': False,
            },
            {
                'generator': 'tests/numerics/hf_ref.py',
                'generator_sha256': _hash(__file__),
                'test': f'tests/test_cuda_nemotron_vision.py::{test}',
                'test_sha256': _hash(source_path),
                'hf_sources_sha256': sources,
                'hf_initial_modules': _module_files(hf_root),
                'versions': versions,
                'checkpoint': checkpoint,
                'category': 'python_reference_crosscheck',
                'native_executed': False,
            },
            actual_role='candidate',
        )
        for name, value in values.items():
            writer.input(name, value)
        writer.output('output', shape, 'float64', {'kind': 'close', 'atol': 2e-4, 'rtol': 2e-4})
        case = writer.finish()
    reference(case, case.root / 'reference', hf_root)
    candidate(case, case.root / 'candidate')
    return [case.root / 'case.json']
