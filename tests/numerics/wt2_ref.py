"""Freeze original HF/WT2 inputs and lossless MLX model observations."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import struct
import subprocess
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
    model_config = _file(source.path / 'config.json')
    if identity != case.data['source']['model_files'] or (
        'model_config' in case.data['source']
        and model_config != case.data['source']['model_config']
    ):
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
            if (
                _model(source) != identity
                or _file(source.path / 'config.json') != model_config
                or after != before
            ):
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


def native(
    case: Case, root: Path, hf_root: Path, executable: Path, resident: bool = False, device: int = 0
) -> Path:
    """Run native WT2 layers and publish fixed-input, lossless observations."""
    if (
        case.data['op'] != 'wt2_model'
        or case.data['source'].get('generator') != 'tests/numerics/wt2_ref.py'
        or case.data.get('actual_role', 'native') != 'native'
        or type(resident) is not bool
        or type(device) is not int
        or device < 0
    ):
        raise ValueError('Unsupported native WT2 case or execution options')
    verified = Case.load(case.root / 'case.json')
    if verified.data != case.data or verified.root != case.root.resolve():
        raise ValueError('WT2 case changed since it was loaded')
    source = ReferenceSource(hf_root)
    tokens = case.tensor('tokens')
    vocab, width, layers = _geometry(source, tokens)
    config = case.data['config']
    if (config['vocab'], config['hidden_size'], config['layers']) != (vocab, width, layers) or (
        config['architecture'] != source.architecture
    ):
        raise ValueError('WT2 native model geometry differs from the case')
    positions = np.broadcast_to(np.arange(tokens.shape[1], dtype=np.int32), tokens.shape)
    if not np.array_equal(case.tensor('positions'), positions):
        raise ValueError('WT2 native positions differ from the fixed case')
    identity = _model(source)
    model_config = _file(source.path / 'config.json')
    if identity != case.data['source']['model_files'] or (
        'model_config' in case.data['source']
        and model_config != case.data['source']['model_config']
    ):
        raise ValueError('WT2 native model bytes differ from the fixed case')
    executable = Path(executable).resolve(strict=True)
    if not executable.is_file():
        raise ValueError('WT2 native runtime must be a file')
    runtime_identity = _file(executable)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    environment = os.environ.copy()
    visible = environment.get('CUDA_VISIBLE_DEVICES')
    if visible:
        devices = visible.split(',')
        if device >= len(devices):
            raise ValueError('WT2 native CUDA device is not visible')
        environment['CUDA_VISIBLE_DEVICES'] = devices[device]
    else:
        environment['CUDA_VISIBLE_DEVICES'] = str(device)
    scratch = root / 'raw'
    scratch.mkdir(exist_ok=False)
    input_path, logits_path, capture = (
        scratch / 'tokens.bin',
        scratch / 'logits.f32',
        scratch / 'capture',
    )
    with input_path.open('xb') as stream:
        stream.write(struct.pack('<II', *tokens.shape))
        stream.write(tokens.astype('<i4', copy=False).tobytes())
    command = [
        str(executable),
        '--wt2-reference',
        '--model',
        str(source.path),
        '--ctx-size',
        str(tokens.shape[1]),
        '--reference-input',
        str(input_path),
        '--reference-output',
        str(logits_path),
        '--reference-capture',
        str(capture),
    ]
    if resident:
        command.append('--resident')
    stdout_path, stderr_path = root / 'runtime.stdout', root / 'runtime.stderr'
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
        process = subprocess.run(
            command, stdout=stdout, stderr=stderr, env=environment, check=False
        )
    if process.returncode:
        raise RuntimeError(f'WT2 native runtime failed with exit code {process.returncode}')
    metadata = json.loads((capture / 'capture.json').read_text(encoding='utf-8'))
    chunks, context = tokens.shape
    expected = {
        'chunks': chunks,
        'context': context,
        'width': width,
        'layers': layers,
        'vocab': vocab,
        'hidden_dtype': 'float32',
        'logits_dtype': 'float32',
        'activation_restore_dtype': 'float16',
        'head_input_dtype': 'float16',
        'requested_resident': resident,
        'primary_device': 0,
    }
    if any(metadata.get(name) != value for name, value in expected.items()) or (
        type(metadata.get('resident')) is not bool
        or type(metadata.get('oom_fallback')) is not bool
        or metadata['resident'] != (resident and not metadata['oom_fallback'])
        or (metadata['oom_fallback'] and not resident)
    ):
        raise ValueError('WT2 native capture metadata differs from the case')
    count = context - (context // 2 + 1)
    expected_size = chunks * count * vocab * 4
    if logits_path.stat().st_size != expected_size:
        raise ValueError('WT2 native logits have an incorrect byte count')
    with ReferenceWriter(root / 'quality.logits', tokens, vocab) as quality:
        writer = _Writer(case, root, quality)
        with logits_path.open('rb') as stream:
            for chunk in range(chunks):
                for offset in range(0, count, 32):
                    rows = min(32, count - offset)
                    raw = stream.read(rows * vocab * 4)
                    if len(raw) != rows * vocab * 4:
                        raise ValueError('WT2 native logits are incomplete')
                    writer.logits(
                        chunk, offset, np.frombuffer(raw, dtype='<f4').reshape(rows, vocab)
                    )
            if stream.read(1):
                raise ValueError('WT2 native logits contain extra bytes')
        for chunk in range(chunks):
            for label, filename in [(f'embedding_{chunk}', f'embedding-{chunk}.bin')] + [
                (f'layer_{layer}_{chunk}', f'layer-{layer}-{chunk}.bin') for layer in range(layers)
            ]:
                raw = (capture / filename).read_bytes()
                if len(raw) != context * width * 4:
                    raise ValueError(f'Invalid WT2 native hidden observation: {label}')
                writer.outputs[label] = save_tensor(
                    root, label, np.frombuffer(raw, dtype='<f4').reshape(context, width)
                )
        if _model(source) != identity or _file(source.path / 'config.json') != model_config:
            raise ValueError('WT2 native model changed during execution')
        if _file(executable) != runtime_identity:
            raise ValueError('WT2 native runtime changed during execution')
        if Case.load(case.root / 'case.json').data != verified.data:
            raise ValueError('WT2 case changed during native execution')
        quality.commit()
    return write_result(
        case,
        root,
        writer.outputs,
        {
            'framework': 'TyloInfer native CUDA',
            'device': device,
            'cuda_visible_devices': environment['CUDA_VISIBLE_DEVICES'],
            'capture': metadata,
            'model_files': identity,
            'model_config': model_config,
            'runtime': {'file': str(executable), **runtime_identity},
            'stdout': {'file': stdout_path.name, **_file(stdout_path)},
            'stderr': {'file': stderr_path.name, **_file(stderr_path)},
            'generator_sha256': _file(__file__)['sha256'],
            'quality': {
                'file': 'quality.logits',
                **_file(root / 'quality.logits'),
                'magic': '_logit3_',
                'distribution': 'lossy_uint16_log_probability',
            },
        },
        role='native',
    )


def freeze(
    root: Path,
    hf_root: Path,
    tokens: np.ndarray,
    input_source: dict,
    rules: Path | None = None,
    *,
    layerwise: bool = True,
) -> Case:
    """Freeze fixed tokens and model identity using the WT2 observation contract."""
    if not isinstance(input_source, dict) or not input_source:
        raise ValueError('Fixed input provenance is required')
    rules_bytes = Path(rules).read_bytes() if rules is not None else None
    criteria = (
        _rules(json.loads(rules_bytes))
        if rules_bytes is not None
        else {
            name: {'rule': {'kind': 'observe'}, 'source': 'no original numeric criterion'}
            for name in ('embedding', 'layer', 'logits', 'target_logp')
        }
    )
    source = ReferenceSource(hf_root)
    identity = _model(source)
    model_config = _file(source.path / 'config.json')
    vocab, width, layers = _geometry(source, tokens)
    chunks, context = tokens.shape
    _framework()
    provenance = {
        'generator': 'tests/numerics/wt2_ref.py',
        'generator_sha256': _file(__file__)['sha256'],
        'model_files': identity,
        'model_config': model_config,
        'input_source': input_source,
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
    }
    if input_source.get('kind') == 'official_wt2':
        provenance['dataset'] = input_source['dataset']
        provenance['tokenizer'] = input_source['tokenizer']
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
        provenance,
    )
    writer.input('tokens', tokens)
    writer.input('positions', np.broadcast_to(np.arange(context, dtype=np.int32), tokens.shape))
    if rules_bytes is not None:
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
    if _model(source) != identity or _file(source.path / 'config.json') != model_config:
        raise ValueError('HF model changed while freezing inputs')
    return writer.finish()


def generate(
    root: Path,
    hf_root: Path,
    dataset: Path,
    rules: Path | None = None,
    *,
    context: int = 512,
    chunks: int = 8,
    layerwise: bool = True,
) -> list[Path]:
    """Freeze official WT2 windows and run the independent MLX reference."""
    from tyloinfer.server.services.evaluation_datasets import WT2_TEXT_SHA256

    source = ReferenceSource(hf_root)
    if type(context) is not int or context < 32 or type(chunks) is not int or chunks < 1:
        raise ValueError('Invalid WT2 context or window count')
    dataset = Path(dataset)
    dataset_identity = _file(dataset)
    if dataset_identity['sha256'] != WT2_TEXT_SHA256:
        raise ValueError('WT2 generation requires the official mfq-wt2-text-v1 corpus')
    identity = _model(source)
    model_config = _file(source.path / 'config.json')
    vocab = int(source.text.get('vocab_size', source.config.get('vocab_size', 0)))
    tokens, tokenizer = tokenize_windows(
        source.path, dataset.read_text(encoding='utf-8'), context, chunks, vocab
    )
    _geometry(source, tokens)
    if (
        _model(source) != identity
        or _file(source.path / 'config.json') != model_config
        or _file(dataset) != dataset_identity
    ):
        raise ValueError('HF model or WT2 corpus changed while preparing inputs')
    case = freeze(
        root,
        hf_root,
        tokens,
        {'kind': 'official_wt2', 'dataset': dataset_identity, 'tokenizer': tokenizer},
        rules,
        layerwise=layerwise,
    )
    if (
        _model(source) != identity
        or _file(source.path / 'config.json') != model_config
        or _file(dataset) != dataset_identity
    ):
        raise ValueError('HF model or WT2 corpus changed while freezing inputs')
    reference(case, case.root / 'reference', hf_root)
    return [case.root / 'case.json']
