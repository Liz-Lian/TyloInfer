"""Freeze original HF/WT2 inputs and lossless MLX model observations."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from tyloinfer.evaluation.reference import ReferenceWriter, tokenize_windows
from tyloinfer.evaluation.source import ReferenceSource

from .case import Case, CaseWriter, save_tensor, write_result


def _file(path):
    """Identify the exact bytes used by a source file or external checkpoint."""
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def _model(source):
    """Hash checkpoint inputs without copying or narrowing their tensors."""
    files = set(source.files)
    index = source.path / 'model.safetensors.index.json'
    if index.is_file():
        files.add(index)
    result = {}
    for path in sorted(files):
        logical = Path(os.path.abspath(path))
        if not logical.is_relative_to(source.path):
            raise ValueError('Checkpoint file path escapes the explicit HF directory')
        result[logical.relative_to(source.path).as_posix()] = _file(logical)
    return result


def _framework():
    """Load the existing independent model computation only in the reference process."""
    module = importlib.import_module('tyloinfer.evaluation.reference_mlx')
    for name in ('base', 'gemma4_text', 'qwen3_5'):
        importlib.import_module(f'mlx_lm.models.{name}')
    return module


def _implementation():
    """Pin complete package code so lazy imports cannot escape case identity."""
    files = {}
    for name in ('mlx', 'mlx_lm'):
        package = importlib.import_module(name)
        roots = sorted(Path(path) for path in package.__path__)
        for index, root in enumerate(roots):
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in {
                    '.py',
                    '.so',
                    '.pyd',
                    '.dll',
                    '.dylib',
                    '.json',
                }:
                    files[f'{name}/{index}/{path.relative_to(root).as_posix()}'] = _file(path)
    for name in ('reference', 'reference_mlx', 'source'):
        module = importlib.import_module(f'tyloinfer.evaluation.{name}')
        files[f'tyloinfer/evaluation/{name}.py'] = _file(module.__file__)
    module = importlib.import_module('tyloinfer.formats.hf_source')
    files['tyloinfer/formats/hf_source.py'] = _file(module.__file__)
    return {
        'versions': {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'numpy')},
        'files': files,
    }


def _rules(value):
    """Require explicit observation criteria instead of borrowing quality tolerances."""
    if not isinstance(value, dict) or set(value) != {'embedding', 'layer', 'logits', 'target_logp'}:
        raise ValueError('Rules must specify embedding, layer, logits and target_logp')
    for name, item in value.items():
        if (
            not isinstance(item, dict)
            or set(item) != {'rule', 'source'}
            or not isinstance(item['rule'], dict)
            or not isinstance(item['source'], str)
            or not item['source'].strip()
        ):
            raise ValueError(f'An explicit rule and its source are required for {name}')
    return value


def _geometry(source, tokens):
    """Validate the original non-strided windows without retokenizing fixed inputs."""
    if not source.summary()['reference_supported']:
        raise ValueError('WT2 references support original Qwen3.5 and Gemma4 HF models')
    vocab = int(source.text.get('vocab_size', source.config.get('vocab_size', 0)))
    if (
        tokens.dtype != np.dtype('<i4')
        or tokens.ndim != 2
        or tokens.shape[0] < 1
        or tokens.shape[1] < 32
        or tokens.shape[1] > int(source.text['max_position_embeddings'])
        or vocab < 1
        or np.any(tokens < 0)
        or np.any(tokens >= vocab)
    ):
        raise ValueError('Invalid fixed WT2 token windows')
    return vocab, int(source.text['hidden_size']), int(source.text['num_hidden_layers'])


class _Writer:
    """Export exact selected logits while delegating the original quality calculation."""

    def __init__(self, case, root, quality):
        self.case, self.root, self.quality = case, root, quality
        self.start, self.count = quality.start, quality.count
        self.vocab, self.chunks = quality.vocab, quality.chunks
        self.outputs = {}

    def logits(self, chunk, offset, values):
        """Persist one original head block and its exact target log-probability."""
        nll = self.quality.logits(chunk, offset, values)
        for label, array in (
            ('logits', np.asarray(values, dtype=np.float32)),
            ('target_logp', np.array(self.quality.targets[chunk, offset : offset + len(values)])),
        ):
            name = f'{label}_{chunk}_{offset}'
            if name in self.outputs:
                raise ValueError(f'Duplicate WT2 observation: {name}')
            contract = self.case.data['outputs'].get(name)
            if contract is None or list(array.shape) != contract['shape']:
                raise ValueError(f'Unexpected WT2 observation: {name}')
            self.outputs[name] = save_tensor(self.root, name, array)
        return nll


def reference(case: Case, root: Path, hf_root: Path) -> Path:
    """Run the original MLX graph on fixed tokens and export complete observations."""
    if (
        case.data['source'].get('generator') != 'tests/numerics/wt2_ref.py'
        or case.data['op'] != 'wt2_model'
    ):
        raise ValueError('Unsupported WT2 reference case')
    source = ReferenceSource(hf_root)
    tokens = case.tensor('tokens')
    vocab, width, layers = _geometry(source, tokens)
    identity = _model(source)
    if identity != case.data['source']['model_files']:
        raise ValueError('HF checkpoint bytes differ from the fixed WT2 case')
    config = case.data['config']
    if (config['vocab'], config['hidden_size'], config['layers']) != (vocab, width, layers):
        raise ValueError('WT2 model geometry differs from the fixed case')
    expected_positions = np.broadcast_to(np.arange(tokens.shape[1], dtype=np.int32), tokens.shape)
    if not np.array_equal(case.tensor('positions'), expected_positions):
        raise ValueError('WT2 positions must restart at zero for each chunk')
    module = _framework()
    before = _implementation()
    expected = case.data['source']['implementation']
    if before != expected:
        raise ValueError('WT2 reference implementation differs from the fixed case')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='wt2-', dir=root) as temporary:
        scratch = Path(temporary)
        capture = scratch / 'capture'
        capture.mkdir()
        with ReferenceWriter(root / 'quality.logits', tokens, vocab) as quality:
            writer = _Writer(case, root, quality)
            nll, execution = module.generate(
                source, tokens, writer, scratch, layerwise=config['layerwise'], capture=capture
            )
            for chunk in range(tokens.shape[0]):
                for label, filename in [(f'embedding_{chunk}', f'embedding-{chunk}.npy')] + [
                    (f'layer_{layer}_{chunk}', f'layer-{layer}-{chunk}.npy')
                    for layer in range(layers)
                ]:
                    value = np.load(capture / filename, allow_pickle=False)
                    if value.shape != (tokens.shape[1], width) or value.dtype != np.float32:
                        raise ValueError(f'Invalid WT2 hidden observation: {label}')
                    writer.outputs[label] = save_tensor(root, label, value)
            after = _implementation()
            if _model(source) != identity or after != before:
                raise ValueError('WT2 weights or reference implementation changed during execution')
            quality.commit()
    return write_result(
        case,
        root,
        writer.outputs,
        {
            **execution,
            'framework': 'MLX upstream model',
            'category': 'original_hf_model_reference',
            'independence': 'mlx_lm model graph; no TyloInfer engine or CUDA kernel execution',
            'implementation': after,
            'generator_sha256': _file(__file__)['sha256'],
            'quality': {
                'file': 'quality.logits',
                **_file(root / 'quality.logits'),
                'magic': '_logit3_',
                'distribution': 'lossy_uint16_log_probability',
                'target_logp': 'float32',
                'nll': nll,
            },
        },
    )


def generate(
    root: Path,
    hf_root: Path,
    dataset: Path,
    rules: Path,
    *,
    context: int = 512,
    chunks: int = 8,
    layerwise: bool = True,
) -> list[Path]:
    """Freeze actual original-HF WT2 inputs and run the independent MLX reference.

    Rules are explicit case inputs: original quality thresholds do not define
    permissible errors for newly exported hidden states or logits.
    """
    from tyloinfer.server.services.evaluation_datasets import WT2_TEXT_SHA256

    rules_bytes = Path(rules).read_bytes()
    criteria = _rules(json.loads(rules_bytes))
    source = ReferenceSource(hf_root)
    if type(context) is not int or context < 32 or type(chunks) is not int or chunks < 1:
        raise ValueError('Invalid WT2 context or window count')
    dataset = Path(dataset)
    dataset_identity = _file(dataset)
    if dataset_identity['sha256'] != WT2_TEXT_SHA256:
        raise ValueError('WT2 generation requires the official mfq-wt2-text-v1 corpus')
    identity = _model(source)
    vocab = int(source.text.get('vocab_size', source.config.get('vocab_size', 0)))
    tokens, tokenizer = tokenize_windows(
        source.path, dataset.read_text(encoding='utf-8'), context, chunks, vocab
    )
    vocab, width, layers = _geometry(source, tokens)
    if _model(source) != identity or _file(dataset) != dataset_identity:
        raise ValueError('HF model or WT2 corpus changed while preparing inputs')
    _framework()
    writer = CaseWriter(
        root,
        'wt2_reference',
        'wt2_model',
        {
            'architecture': source.architecture,
            'vocab': vocab,
            'hidden_size': width,
            'layers': layers,
            'layerwise': layerwise,
            'rules': criteria,
            'layer_types': source.text.get('layer_types'),
            'precision': source.summary()['source_precisions'],
            'protocol': 'llama.cpp-non-strided-second-half-v1',
            'attention_mask': 'original model causal/sliding mask with fresh per-chunk state',
            'initial_state': 'fresh for each layer/chunk; Gemma shared KV follows original producer',
            'hidden_observation': 'F32 disk values before reloading at original source execution dtype',
            'head_observation': 'F32 selected logits after original final norm and softcap',
            'cuda_limitations': 'existing CUDA reference rejects Gemma shared KV and per-layer embeddings',
        },
        {
            'generator': 'tests/numerics/wt2_ref.py',
            'generator_sha256': _file(__file__)['sha256'],
            'model_files': identity,
            'tokenizer': tokenizer,
            'dataset': dataset_identity,
            'implementation': _implementation(),
            'category': 'original_hf_model_reference',
            'source_tests': {
                name: _file(Path(__file__).parents[1] / name)['sha256']
                for name in ('test_native_wt2_reference.py', 'test_wt2_reference.py')
            },
            'quality_checks': [
                'original resident/layerwise equality and memory-pressure fallback',
                'original model-family and source-precision parameter combinations',
            ],
        },
    )
    writer.input('tokens', tokens)
    writer.input('positions', np.broadcast_to(np.arange(context, dtype=np.int32), tokens.shape))
    writer.asset('rules', rules_bytes, encoding='json')
    count = context - (context // 2 + 1)
    for chunk in range(chunks):
        writer.output(
            f'embedding_{chunk}', (context, width), 'float32', criteria['embedding']['rule']
        )
        for layer in range(layers):
            writer.output(
                f'layer_{layer}_{chunk}', (context, width), 'float32', criteria['layer']['rule']
            )
        for offset in range(0, count, 32):
            rows = min(32, count - offset)
            writer.output(
                f'logits_{chunk}_{offset}', (rows, vocab), 'float32', criteria['logits']['rule']
            )
            writer.output(
                f'target_logp_{chunk}_{offset}', (rows,), 'float32', criteria['target_logp']['rule']
            )
    case = writer.finish()
    reference(case, case.root / 'reference', hf_root)
    return [case.root / 'case.json']
