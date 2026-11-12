"""Freeze the synthetic RADIO CPU reference from the CUDA vision tests."""

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional

from .case import Case, CaseWriter, save_tensor, write_result
from .nemotron_text_ref import fixture as text_fixture

HIDDEN = 64
HEADS = 4
DEPTH = 2
PATCH_SIZE = 2
POSITION_SIDE = 8
CLS_TOKENS = 3
REGISTERS = 1
PROJECTOR_INTERMEDIATE = 128
LAYER_NORM_EPS = 1e-6
PROJECTOR_EPS = 1e-5
GRIDS = ((4, 6), (6, 2))


def _fixture():
    """Reproduce the source configuration and float32 vision weights in RNG order."""
    config, text_weights = text_fixture(('mamba', 'moe'))
    config.update(
        vision_config={
            'model_type': 'radio',
            'hidden_size': HIDDEN,
            'num_hidden_layers': DEPTH,
            'num_attention_heads': HEADS,
            'mlp_ratio': 4.0,
            'num_channels': 3,
            'patch_size': PATCH_SIZE,
            'max_img_size': PATCH_SIZE * POSITION_SIDE,
            'num_cls_tokens': CLS_TOKENS,
            'num_registers': REGISTERS,
            'layer_norm_eps': LAYER_NORM_EPS,
            'hidden_act': 'gelu',
            'qkv_bias': True,
            'use_swiglu_ffn': False,
            'layerscale_value': 1.0,
        },
        vit_hidden_size=HIDDEN,
        projector_hidden_size=PROJECTOR_INTERMEDIATE,
        downsample_ratio=0.5,
        ps_version='v2',
        img_context_token_id=18,
    )
    rng = np.random.default_rng(20260119)

    def rand(shape, scale=0.08):
        """Draw the next original float32 synthetic weight."""
        return rng.normal(scale=scale, size=shape).astype(np.float32)

    weights = {'model.token_embedding.weight': text_weights['model.token_embedding.weight']}
    weights['vision.patch_embedding.weight'] = rand((HIDDEN, 12))
    weights['vision.position_embedding.weight'] = rand((1, POSITION_SIDE**2, HIDDEN), 0.05)
    weights['vision.cls_register_token.weight'] = rand((CLS_TOKENS + REGISTERS, HIDDEN), 0.05)
    for layer in range(DEPTH):
        prefix = f'vision.block.{layer}.'
        weights[prefix + 'norm1.weight'] = np.linspace(0.8, 1.2, HIDDEN).astype(np.float32)
        weights[prefix + 'norm1.bias'] = rand((HIDDEN,), 0.02)
        weights[prefix + 'norm2.weight'] = np.linspace(1.1, 0.9, HIDDEN).astype(np.float32)
        weights[prefix + 'norm2.bias'] = rand((HIDDEN,), 0.02)
        weights[prefix + 'attention.qkv.weight'] = rand((3 * HIDDEN, HIDDEN))
        weights[prefix + 'attention.qkv.bias'] = rand((3 * HIDDEN,), 0.02)
        weights[prefix + 'attention.output.weight'] = rand((HIDDEN, HIDDEN))
        weights[prefix + 'attention.output.bias'] = rand((HIDDEN,), 0.02)
        weights[prefix + 'mlp.up.weight'] = rand((4 * HIDDEN, HIDDEN))
        weights[prefix + 'mlp.up.bias'] = rand((4 * HIDDEN,), 0.02)
        weights[prefix + 'mlp.down.weight'] = rand((HIDDEN, 4 * HIDDEN))
        weights[prefix + 'mlp.down.bias'] = rand((HIDDEN,), 0.02)
    weights['vision.projector.norm.weight'] = np.linspace(0.7, 1.3, HIDDEN).astype(np.float32)
    weights['vision.projector.norm.bias'] = rand((HIDDEN,), 0.05)
    weights['vision.projector.mlp.norm.weight'] = np.linspace(0.9, 1.1, 4 * HIDDEN).astype(
        np.float32
    )
    weights['vision.projector.mlp.up.weight'] = rand((PROJECTOR_INTERMEDIATE, 4 * HIDDEN))
    weights['vision.projector.mlp.down.weight'] = rand((HIDDEN, PROJECTOR_INTERMEDIATE))
    return config, weights


def _layer_norm(x, weight, bias, eps):
    """Apply the source's explicit population-variance layer normalization."""
    mean = x.mean(-1, keepdim=True)
    variance = (x - mean).pow(2).mean(-1, keepdim=True)
    return (x - mean) / torch.sqrt(variance + eps) * weight + bias


def _block(hidden, weights, prefix):
    """Apply one original float64 pre-LayerNorm RADIO attention and MLP block."""
    batch, tokens, width = hidden.shape
    head_dim = width // HEADS
    x = _layer_norm(
        hidden, weights[prefix + 'norm1.weight'], weights[prefix + 'norm1.bias'], LAYER_NORM_EPS
    )
    qkv = x @ weights[prefix + 'attention.qkv.weight'].T + weights[prefix + 'attention.qkv.bias']
    q, k, v = qkv.reshape(batch, tokens, 3, HEADS, head_dim).permute(2, 0, 3, 1, 4)
    scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / math.sqrt(head_dim)
    attended = torch.einsum('bhqk,bhkd->bhqd', torch.softmax(scores, dim=-1), v)
    attended = attended.transpose(1, 2).reshape(batch, tokens, width)
    attended = attended @ weights[prefix + 'attention.output.weight'].T
    attended = attended + weights[prefix + 'attention.output.bias']
    hidden = hidden + attended
    x = _layer_norm(
        hidden, weights[prefix + 'norm2.weight'], weights[prefix + 'norm2.bias'], LAYER_NORM_EPS
    )
    x = x @ weights[prefix + 'mlp.up.weight'].T + weights[prefix + 'mlp.up.bias']
    x = 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    x = x @ weights[prefix + 'mlp.down.weight'].T + weights[prefix + 'mlp.down.bias']
    return hidden + x


def _features(pixels, weights, grids):
    """Compute the original cropped-position RADIO features in image order."""
    table = weights['vision.position_embedding.weight'][0]
    prefix = weights['vision.cls_register_token.weight']
    features = []
    offset = 0
    for rows, columns in grids:
        square = max(rows, columns)
        position = table.reshape(POSITION_SIDE, POSITION_SIDE, HIDDEN).permute(2, 0, 1).unsqueeze(0)
        position = functional.interpolate(
            position, size=(square, square), mode='bilinear', align_corners=False
        )
        position = position[..., :rows, :columns].flatten(2).transpose(1, 2)[0]
        count = rows * columns
        hidden = pixels[offset : offset + count] @ weights['vision.patch_embedding.weight'].T
        hidden = torch.cat([prefix.unsqueeze(0), (hidden + position).unsqueeze(0)], dim=1)
        for layer in range(DEPTH):
            hidden = _block(hidden, weights, f'vision.block.{layer}.')
        features.append(hidden[0, CLS_TOKENS + REGISTERS :])
        offset += count
    return torch.cat(features, dim=0)


def _project(features, weights, grids):
    """Apply the source v2 pixel shuffle and the independent projector formula."""
    normalized = _layer_norm(
        features,
        weights['vision.projector.norm.weight'],
        weights['vision.projector.norm.bias'],
        LAYER_NORM_EPS,
    )
    pieces = []
    start = 0
    for rows, columns in grids:
        patch = normalized[start : start + rows * columns].reshape(1, rows, columns, HIDDEN)
        patch = patch.reshape(1, rows, columns // 2, 2 * HIDDEN).permute(0, 2, 1, 3).contiguous()
        patch = patch.reshape(1, columns // 2, rows // 2, 4 * HIDDEN)
        pieces.append(patch.permute(0, 2, 1, 3).contiguous().reshape(-1, 4 * HIDDEN))
        start += rows * columns
    shuffled = torch.cat(pieces, dim=0)
    variance = shuffled.pow(2).mean(-1, keepdim=True)
    shuffled = shuffled / torch.sqrt(variance + PROJECTOR_EPS)
    shuffled = shuffled * weights['vision.projector.mlp.norm.weight']
    hidden = torch.relu(shuffled @ weights['vision.projector.mlp.up.weight'].T) ** 2
    return hidden @ weights['vision.projector.mlp.down.weight'].T


def reference(case: Case, root: Path) -> Path:
    """Recalculate one CPU float64 RADIO result from fixed pixels and weights."""
    if case.data['op'] not in {'radio_features', 'radio_projector'}:
        raise ValueError('Unsupported RADIO reference operation')
    if case.data['source'].get('generator') != 'tests/numerics/nemotron_vision_ref.py':
        raise ValueError('Unsupported RADIO reference source')
    grids = case.data['config']['image_grids']
    if grids != [list(grid) for grid in GRIDS]:
        raise ValueError('Unsupported RADIO image grids')
    weights = {
        name: torch.tensor(case.tensor(name), dtype=torch.float64)
        for name in case.data['inputs']
        if name.startswith('vision.')
    }
    pixels = torch.tensor(case.tensor('pixels'), dtype=torch.float64)
    features = _features(pixels, weights, grids)
    output = features if case.data['op'] == 'radio_features' else _project(features, weights, grids)
    spec = save_tensor(root, 'output', output.detach().contiguous().numpy())
    return write_result(
        case,
        root,
        {'output': spec},
        {
            'framework': 'PyTorch',
            'torch_version': str(torch.__version__),
            'numpy_version': np.__version__,
            'device': 'cpu',
            'arithmetic': 'float64',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


def generate(root: Path, suite: str = 'vision') -> list[Path]:
    """Freeze the two-image RADIO tower and projector cases from the source test."""
    if suite != 'vision':
        raise ValueError(f'Unsupported RADIO suite: {suite}')
    config, weights = _fixture()
    config['image_grids'] = [list(grid) for grid in GRIDS]
    config['unmigrated'] = [
        'native placeholder IDs and positions',
        'dynamic MTP hidden states',
        'real HF checkpoint and one real encoder layer',
    ]
    count = sum(rows * columns for rows, columns in GRIDS)
    pixels = np.asarray(
        [((index * 37 + 11) % 101) / 64.0 for index in range(count * 12)], dtype=np.float64
    ).reshape(count, 12)
    test_path = Path(__file__).resolve().parents[1] / 'test_cuda_nemotron_vision.py'
    source = {
        'generator': 'tests/numerics/nemotron_vision_ref.py',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_sha256': hashlib.sha256(test_path.read_bytes()).hexdigest(),
        'vision_seed': 20260119,
        'pixels': '((index * 37 + 11) % 101) / 64',
    }
    paths = []
    for op, test, shape, actual_dtype in (
        ('radio_projector', 'test_native_radio_matches_reference', (9, HIDDEN), 'float32'),
        ('radio_features', 'test_reference_matches_hf_radio_model', (count, HIDDEN), 'float64'),
    ):
        writer = CaseWriter(
            root, op, op, config, {**source, 'test': f'tests/test_cuda_nemotron_vision.py::{test}'}
        )
        writer.input('pixels', pixels)
        for name, weight in weights.items():
            writer.input(name, weight)
        writer.output(
            'output',
            shape,
            'float64',
            {'kind': 'close', 'atol': 2e-4, 'rtol': 2e-4},
            actual_dtype=actual_dtype,
        )
        case = writer.finish()
        reference(case, case.root / 'reference')
        paths.append(case.root / 'case.json')
    return paths
