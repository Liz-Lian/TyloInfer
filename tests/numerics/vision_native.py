"""Execute fixed synthetic RADIO cases through the native vision diagnostic."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from tyloinfer.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import load, save

from .case import Case, save_tensor, write_result


def _identity(path):
    """Identify the bytes of a retained runtime or fixture artifact."""
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def _model(case, path):
    """Write the original synthetic container using the case's fixed weights."""
    config = {
        name: value
        for name, value in case.data['config'].items()
        if name not in {'image_grids', 'native_checks', 'other_checks'}
    }
    expected = {
        'model.token_embedding.weight': (32, 64),
        'model.output.weight': (32, 64),
        'model.output_norm.weight': (64,),
        'model.block.0.attention.norm.weight': (64,),
        'model.block.0.linear_attention.in_projection.weight': (196, 64),
        'model.block.0.linear_attention.conv.weight': (160, 1, 4),
        'model.block.0.linear_attention.conv.bias': (160,),
        'model.block.0.linear_attention.a': (4,),
        'model.block.0.linear_attention.d': (4,),
        'model.block.0.linear_attention.dt_bias': (4,),
        'model.block.0.linear_attention.norm.weight': (32,),
        'model.block.0.linear_attention.output.weight': (64, 32),
        'model.block.1.attention.norm.weight': (64,),
        'model.block.1.mlp.router.weight': (3, 64),
        'model.block.1.mlp.router.bias': (3,),
        'model.block.1.mlp.latent.down.weight': (16, 64),
        'model.block.1.mlp.latent.up.weight': (64, 16),
        'model.block.1.mlp.experts.up.weight': (3, 8, 16),
        'model.block.1.mlp.experts.down.weight': (3, 16, 8),
        'model.block.1.mlp.shared_expert.up.weight': (12, 64),
        'model.block.1.mlp.shared_expert.down.weight': (64, 12),
        'vision.patch_embedding.weight': (64, 12),
        'vision.position_embedding.weight': (1, 64, 64),
        'vision.cls_register_token.weight': (4, 64),
        'vision.projector.norm.weight': (64,),
        'vision.projector.norm.bias': (64,),
        'vision.projector.mlp.norm.weight': (256,),
        'vision.projector.mlp.up.weight': (128, 256),
        'vision.projector.mlp.down.weight': (64, 128),
    }
    block = {
        'norm1.weight': (64,),
        'norm1.bias': (64,),
        'norm2.weight': (64,),
        'norm2.bias': (64,),
        'attention.qkv.weight': (192, 64),
        'attention.qkv.bias': (192,),
        'attention.output.weight': (64, 64),
        'attention.output.bias': (64,),
        'mlp.up.weight': (256, 64),
        'mlp.up.bias': (256,),
        'mlp.down.weight': (64, 256),
        'mlp.down.bias': (64,),
    }
    for layer in range(2):
        expected.update({f'vision.block.{layer}.{name}': shape for name, shape in block.items()})
    names = {name for name in case.data['inputs'] if name.startswith(('model.', 'vision.'))}
    if names != set(expected):
        raise ValueError('Fixed RADIO weight names differ from the original fixture')
    saved = {name: case.tensor(name) for name in expected}
    if any(
        saved[name].dtype != np.float32
        or saved[name].shape != shape
        or not np.isfinite(saved[name]).all()
        for name, shape in expected.items()
    ):
        raise ValueError('Fixed RADIO weight shape or dtype differs')
    if config.get('model_type') != 'nemotron_h_omni' or config.get('llm_config', {}).get(
        'layers_block_type'
    ) != ['mamba', 'moe']:
        raise ValueError('Fixed RADIO text configuration differs')
    component = {
        'kind': 'vision',
        'tensor_root': 'vision',
        'implementation': 'radio',
        'policy': 'optional',
        'input_contract': 'radio_cpe.v1',
    }
    graph = {
        'schema_version': 1,
        'architecture': 'nemotron_h_omni',
        'canonical_naming': {
            'namespace': 'tyloinfer.tensor',
            'version': 1,
            'component_roots': ['model', 'vision'],
        },
        'topology': {
            'text_layers': len(config['llm_config']['layers_block_type']),
            'vision_layers': 1,
        },
        'graph': {'kind': 'causal_lm', 'backbone': 'nemotron_h'},
        'components': [
            {
                'kind': 'text',
                'tensor_root': 'model',
                'implementation': 'nemotron_h',
                'policy': 'decoder',
            },
            component,
        ],
        'capabilities': ['text'],
    }
    records = {
        **saved,
        MODEL_CONFIG_ASSET: json.dumps(config).encode(),
        MODEL_GRAPH_ASSET: json.dumps(graph).encode(),
    }
    if path.exists():
        raise FileExistsError('RADIO fixture already exists')
    save(path, FileHeader(version=2, model_arch='nemotron_h_omni'), records)
    header, loaded = load(path)
    if header.version != 2 or header.model_arch != 'nemotron_h_omni' or set(loaded) != set(records):
        raise ValueError('RADIO fixture container identity differs')
    if (
        loaded[MODEL_CONFIG_ASSET] != records[MODEL_CONFIG_ASSET]
        or loaded[MODEL_GRAPH_ASSET] != records[MODEL_GRAPH_ASSET]
    ):
        raise ValueError('RADIO fixture configuration or graph differs')
    if any(
        loaded[name].dtype != value.dtype
        or loaded[name].shape != value.shape
        or not np.array_equal(loaded[name], value)
        for name, value in saved.items()
    ):
        raise ValueError('RADIO fixture weights changed during serialization')
    return config


def _tensor(raw, name, shape, dtype=np.float32):
    """Read finite numeric native tensor values with the exact declared geometry."""
    item = raw[name]
    if item['shape'] != list(shape):
        raise ValueError(f'RADIO native {name} shape differs')
    data = item['data']
    if (
        not isinstance(data, list)
        or len(data) != int(np.prod(shape))
        or any(type(value) not in (int, float) or not np.isfinite(value) for value in data)
    ):
        raise ValueError(f'RADIO native {name} payload length differs')
    if dtype == np.int64 and any(int(value) != value for value in data):
        raise ValueError(f'RADIO native {name} must contain integer positions')
    value = np.asarray(data, dtype=dtype)
    if not np.isfinite(value).all():
        raise ValueError(f'RADIO native {name} contains nonfinite values')
    return value.reshape(shape)


def run(case: Case, root: Path, executable: Path, device: int = 0) -> Path:
    """Run the native RADIO diagnostic and export the fixed prompt or projector output."""
    if (
        case.data['op'] not in {'radio_prompt', 'radio_projector'}
        or case.data['source'].get('generator') != 'tests/numerics/nemotron_vision_ref.py'
        or case.data.get('actual_role', 'native') != 'native'
        or type(device) is not int
        or device < 0
    ):
        raise ValueError('Unsupported native RADIO case')
    verified = Case.load(case.root / 'case.json')
    if verified.data != case.data or verified.root != case.root.resolve():
        raise ValueError('RADIO case changed since it was loaded')
    executable = Path(executable).resolve(strict=True)
    if not executable.is_file():
        raise ValueError('RADIO diagnostics executable must be a file')
    binary_identity = _identity(executable)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    model_path = root / 'model.tq'
    config = _model(case, model_path)
    model_identity = _identity(model_path)
    environment = os.environ.copy()
    visible = environment.get('CUDA_VISIBLE_DEVICES')
    if visible:
        devices = visible.split(',')
        if device >= len(devices):
            raise ValueError('RADIO CUDA device is not visible')
        environment['CUDA_VISIBLE_DEVICES'] = devices[device]
    else:
        environment['CUDA_VISIBLE_DEVICES'] = str(device)
    stdout_path, stderr_path = root / 'diagnostics.stdout', root / 'diagnostics.stderr'
    command = [
        str(executable),
        '--model',
        str(model_path),
        '--ctx-size',
        '32',
        '--check-nemotron-h-vision',
    ]
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.run(
            command, stdout=stdout, stderr=stderr, env=environment, check=False
        )
    if process.returncode:
        raise RuntimeError(f'RADIO diagnostics exited with status {process.returncode}')
    lines = [
        line.removeprefix('nemotron_h_vision_check ')
        for line in stdout_path.read_text(encoding='utf-8').splitlines()
        if line.startswith('nemotron_h_vision_check ')
    ]
    if len(lines) != 1:
        raise ValueError('RADIO diagnostic must emit exactly one observation')
    raw = json.loads(lines[0])
    grids = [[4, 6], [6, 2]]
    ids = np.asarray([1, 2, *([18] * 6), 3, *([18] * 3), 3], dtype=np.int64)
    positions = np.flatnonzero(ids == 18)
    pixels = case.tensor('pixels')
    actual_pixels = np.asarray(raw['pixels'], dtype=np.float32)
    expected_pixels = pixels.astype(np.float32).reshape(-1)
    vision = config['vision_config']
    if (
        raw['architecture'] != 'nemotron_h'
        or raw['image_grids'] != grids
        or raw['ids'] != ids.tolist()
        or not np.array_equal(actual_pixels, expected_pixels)
        or pixels.shape != (36, 12)
        or case.data['config']['image_grids'] != grids
        or raw['patch_width'] != 12
        or raw['hidden_size'] != 64
        or raw['downsample_factor'] != 2
        or raw['position_side'] != 8
        or raw['summary_tokens'] != 4
        or raw['heads'] != vision['num_attention_heads']
        or raw['depth'] != vision['num_hidden_layers']
        or raw['image_token_id'] != 18
        or raw['layer_norm_eps'] != vision['layer_norm_eps']
        or raw['projector_norm_eps'] != 1e-5
        or type(raw['placeholder_mismatch_rejected']) is not bool
    ):
        raise ValueError('RADIO diagnostic inputs or configuration differ from the case')
    if case.data['op'] == 'radio_prompt' and (
        not np.array_equal(case.tensor('token_ids'), ids)
        or not np.array_equal(case.tensor('image_positions'), positions)
        or not np.array_equal(
            case.tensor('vision_grid'), np.asarray([[1, 4, 6], [1, 6, 2]], dtype=np.int32)
        )
        or not np.array_equal(case.tensor('vision_types'), np.ones(2, dtype=np.int32))
    ):
        raise ValueError('RADIO fixed prompt inputs differ from the diagnostic')
    embeddings = _tensor(raw, 'embeddings', (1, 13, 64))
    native_positions = _tensor(raw, 'positions', (1, 13), dtype=np.int64)
    if not np.array_equal(native_positions, np.arange(13, dtype=np.int64).reshape(1, 13)):
        raise ValueError('RADIO native positions differ from the fixed prompt')
    outputs = (
        {'embeddings': embeddings, 'positions': native_positions}
        if case.data['op'] == 'radio_prompt'
        else {'output': embeddings[0, positions]}
    )
    specs = {name: save_tensor(root, name, value) for name, value in outputs.items()}
    checks = None
    if case.data['op'] == 'radio_prompt':
        checks = {
            'native_0': {
                'status': 'pass' if raw['placeholder_mismatch_rejected'] else 'fail',
                'evidence': {'placeholder_mismatch_rejected': raw['placeholder_mismatch_rejected']},
            }
        }
    if (
        Case.load(case.root / 'case.json').data != verified.data
        or _identity(executable) != binary_identity
    ):
        raise ValueError('RADIO case or diagnostics executable changed during execution')
    if _identity(model_path) != model_identity:
        raise ValueError('RADIO fixture changed during execution')
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'TyloInfer native CUDA',
            'device': device,
            'cuda_visible_devices': environment['CUDA_VISIBLE_DEVICES'],
            'entrypoint': 'run_nemotron_h_vision_check',
            'compute': 'RADIO layer dtype follows F32 saved position embeddings; projector RMS uses F32',
            'pixels': 'fixed F64 or F32 values converted to diagnostic F32 input',
            'fixture': {'file': model_path.name, **model_identity},
            'binary': {'file': str(executable), **binary_identity},
            'stdout': {'file': stdout_path.name, **_identity(stdout_path)},
            'stderr': {'file': stderr_path.name, **_identity(stderr_path)},
            'generator_sha256': _identity(__file__)['sha256'],
        },
        role='native',
        checks=checks,
    )
