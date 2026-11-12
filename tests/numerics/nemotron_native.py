"""Run fixed Nemotron text cases through the native diagnostics executable."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from tyloinfer.architectures.tensor_schema import CANONICAL_TENSOR_NAMESPACE

from .case import Case, save_tensor, write_result
from .compare import _evaluate


def _hash(path: Path) -> str:
    """Return the SHA256 identity of one file."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def _fixture(path: Path, config: dict, weights: dict, predictor: bool) -> None:
    """Write the original Nemotron synthetic model graph and fixed weights."""
    from tyloinfer.formats.assets import MODEL_CONFIG_ASSET, MODEL_GRAPH_ASSET
    from tyloinfer.formats.header import FileHeader
    from tyloinfer.formats.io import save

    components = [
        dict(kind='text', tensor_root='model', implementation='nemotron_h', policy='decoder')
    ]
    roots, capabilities = ['model'], ['text']
    topology = dict(text_layers=len(config['llm_config']['layers_block_type']))
    if predictor:
        components.append(
            dict(
                kind='predictor',
                tensor_root='predictor',
                implementation='nemotron_h_mtp',
                policy='optional',
            )
        )
        roots.append('predictor')
        capabilities.append('speculative_prediction')
        topology['predictor_layers'] = len(
            config['llm_config'].get('mtp_layers_block_type') or ('attention',)
        )
    graph = dict(
        schema_version=1,
        architecture='nemotron_h_omni',
        canonical_naming=dict(namespace=CANONICAL_TENSOR_NAMESPACE, version=1, component_roots=roots),
        topology=topology,
        graph=dict(kind='causal_lm', backbone='nemotron_h'),
        components=components,
        capabilities=capabilities,
    )
    save(
        path,
        FileHeader(version=2, model_arch='nemotron_h_omni'),
        {
            **weights,
            MODEL_CONFIG_ASSET: json.dumps(config).encode(),
            MODEL_GRAPH_ASSET: json.dumps(graph).encode(),
        },
    )


def _tensor(record: dict, name: str) -> np.ndarray:
    """Read a finite float32 tensor and its explicit shape from diagnostic JSON."""
    item = record[name]
    if not isinstance(item, dict) or not isinstance(item.get('shape'), list):
        raise ValueError(f'Invalid diagnostic tensor: {name}')
    shape = tuple(item['shape'])
    if any(type(part) is not int or part < 0 for part in shape):
        raise ValueError(f'Invalid diagnostic tensor shape: {name}')
    data = item.get('data')
    if not isinstance(data, list) or any(type(value) not in (int, float) for value in data):
        raise ValueError(f'Invalid flat diagnostic tensor data: {name}')
    array = np.asarray(data, dtype=np.float32)
    if array.size != int(np.prod(shape, dtype=np.int64)) or not np.isfinite(array).all():
        raise ValueError(f'Invalid diagnostic tensor data: {name}')
    return np.ascontiguousarray(array.reshape(shape))


def _check(
    left: np.ndarray, right: np.ndarray, *, exact: bool = False, tolerance: float = 2e-3
) -> dict:
    """Compare two actual observations at the source test's tolerance."""
    if left.shape != right.shape:
        raise ValueError('Diagnostic comparison shapes differ')
    rule = {'kind': 'exact'} if exact else {'kind': 'close', 'atol': tolerance, 'rtol': tolerance}
    comparison = _evaluate(left, right, {'rule': rule})
    return {'status': comparison['status'], 'evidence': comparison}


def _negative(case: Case, actual: np.ndarray) -> dict:
    """Compare the actual head against the original independently computed wrong input."""
    from .nemotron_text_ref import _mtp_head

    weights = {
        name: case.tensor(name)
        for name in case.data['inputs']
        if name.startswith(('model.', 'predictor.'))
    }
    config, ids = case.data['config']['model'], case.tensor('ids')
    cache = [case.tensor('key_cache'), case.tensor('value_cache')]
    if case.data['op'] == 'nemotron_mtp_target':
        previous = case.tensor('target_hidden')[:, :6] / weights['model.output_norm.weight']
        wrong, _ = _mtp_head(config, weights, previous, ids[:, 1:], cache, 0)
    else:
        _, _, previous = _mtp_head(
            config, weights, case.tensor('previous')[:, :1], ids[:, :1], cache, 0, return_pre=True
        )
        wrong, _ = _mtp_head(config, weights, previous, ids[:, 1:2], cache, 1)
    comparison = _check(actual, wrong)
    return {
        'status': 'pass' if comparison['status'] == 'fail' else 'fail',
        'evidence': {'category': 'independent_numpy_negative_reference', 'comparison': comparison},
    }


def _observations(case: Case, record: dict) -> tuple[dict, dict]:
    """Map native observations and evaluate the independent negative checks."""
    op = case.data['op']
    checks = {}
    if op == 'nemotron_text_backbone':
        full = _tensor(record, 'full')
        values = {
            'hidden': _tensor(record, 'full_hidden'),
            'logits': full,
            'full_tail': full[:, 5:].copy(),
        }
        for name in (
            'decode',
            'chunked',
            'restored',
            'continued',
            'rejected',
            'rejected_reference',
            'committed',
            'committed_reference',
            'reset',
        ):
            values[name] = _tensor(record, name)
        batch = _tensor(record, 'batch')
        if batch.shape != (2, *full.shape[1:]):
            raise ValueError('Diagnostic batch shape differs from the source test')
        values['batch_0'], values['batch_1'] = batch[:1].copy(), batch[1:2].copy()
    elif op == 'nemotron_mtp_head':
        head = _tensor(record, 'head')
        values = {
            'head': head,
            'logits': _tensor(record, 'logits'),
            'head_tail': head[:, 4:7].copy(),
        }
        for name in ('chunked', 'replayed', 'rollback', 'committed'):
            values[name] = _tensor(record, name)
    elif op == 'nemotron_mtp_chain':
        values = {name: _tensor(record, name) for name in ('chain_first', 'chain_second')}
        checks['native_0'] = _negative(case, values['chain_second'])
    elif op == 'nemotron_mtp_target':
        target = _tensor(record, 'target_hidden')
        if not np.array_equal(target, case.tensor('target_hidden')):
            raise ValueError('Actual target hidden differs from the captured fixed input')
        values = {
            'head_from_target': _tensor(record, 'head_from_target'),
            'actual_target': target,
            'target_normalized': _tensor(record, 'target_normalized'),
        }
        checks['native_1'] = _negative(case, values['head_from_target'])
    else:
        raise ValueError('Unsupported Nemotron native operation')
    return values, checks


def run(case: Case, root: Path, executable: Path) -> Path:
    """Execute fixed Nemotron weights with native diagnostics and publish actual outputs."""
    if not isinstance(case, Case):
        raise TypeError('case must be a Case')
    case_path = case.root / 'case.json'
    verified = Case.load(case_path)
    if verified.data != case.data or verified.root != case.root.resolve():
        raise ValueError('Case changed before native execution')
    op = case.data['op']
    if case.data['source'].get('generator') != 'tests/numerics/nemotron_text_ref.py' or op not in {
        'nemotron_text_backbone',
        'nemotron_mtp_head',
        'nemotron_mtp_chain',
        'nemotron_mtp_target',
    }:
        raise ValueError('Unsupported Nemotron native case')
    ids = case.tensor('ids')
    if (
        ids.dtype != np.int64
        or ids.shape != (1, 7)
        or not np.array_equal(ids, np.arange(1, 8, dtype=np.int64)[None])
    ):
        raise ValueError('Diagnostic fixed token IDs differ from the case')
    from . import nemotron_text_ref as text_ref

    state_shapes = {
        'key_cache': (1, text_ref.KV_HEADS, 32, text_ref.HEAD_DIM),
        'value_cache': (1, text_ref.KV_HEADS, 32, text_ref.HEAD_DIM),
        'mamba_state': (1, text_ref.MAMBA_HEADS, text_ref.MAMBA_HEAD_DIM, text_ref.SSM_STATE),
        'conv_left_context': (
            1,
            text_ref.CONV_KERNEL - 1,
            text_ref.MAMBA_HEADS * text_ref.MAMBA_HEAD_DIM
            + 2 * text_ref.MAMBA_GROUPS * text_ref.SSM_STATE,
        ),
    }
    required_states = (
        ('key_cache', 'value_cache') if op != 'nemotron_text_backbone' else tuple(state_shapes)
    )
    for name in required_states:
        state = case.tensor(name)
        expected_dtype = np.float16 if name in ('key_cache', 'value_cache') else np.float32
        if state.shape != state_shapes[name] or state.dtype != expected_dtype or np.any(state != 0):
            raise ValueError('Diagnostic requires the original zero initial state')
    if op in {'nemotron_mtp_head', 'nemotron_mtp_chain'}:
        previous = case.tensor('previous')
        index = (np.arange(previous.size, dtype=np.int64) % 31).astype(np.float32)
        expected = np.subtract(np.multiply(index, np.float32(0.01)), np.float32(0.15)).reshape(
            previous.shape
        )
        if (
            previous.dtype != np.float32
            or previous.shape != (1, 7, text_ref.HIDDEN)
            or not np.array_equal(previous, expected)
        ):
            raise ValueError('Diagnostic previous hidden differs from the case')
    if op == 'nemotron_mtp_target':
        capture = case.asset('capture')
        if hashlib.sha256(capture).hexdigest() != case.data['source']['capture_sha256']:
            raise ValueError('MTP capture identity differs from the case')
        content = capture.decode('utf-8')
        prefix = 'nemotron_h_mtp_check '
        lines = [line[len(prefix) :] for line in content.splitlines() if line.startswith(prefix)]
        if len(lines) > 1:
            raise ValueError('Expected exactly one captured diagnostic record')
        captured = json.loads(lines[0] if lines else content)
        if (
            captured.get('architecture') != 'nemotron_h'
            or not np.array_equal(_tensor(captured, 'ids'), ids)
            or not np.array_equal(_tensor(captured, 'target_hidden'), case.tensor('target_hidden'))
        ):
            raise ValueError('MTP capture tensors differ from the case')
    weights = {
        name: case.tensor(name)
        for name in case.data['inputs']
        if name.startswith(('model.', 'predictor.'))
    }
    if any(value.dtype != np.float32 for value in weights.values()):
        raise ValueError('Nemotron fixture requires float32 weights')
    executable = Path(executable).resolve(strict=True)
    if not executable.is_file():
        raise ValueError('Diagnostics executable must be a file')
    binary_hash = _hash(executable)
    case_hash = _hash(case_path)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    fixture = root / 'model.tq'
    predictor = op != 'nemotron_text_backbone'
    _fixture(fixture, case.data['config']['model'], weights, predictor)
    fixture_hash = _hash(fixture)
    command = [
        str(executable),
        '--model',
        str(fixture),
        '--ctx-size',
        '64' if predictor else '32',
        '--check-nemotron-h-mtp' if predictor else '--check-nemotron-h',
    ]
    stdout, stderr = root / 'diagnostics.stdout.txt', root / 'diagnostics.stderr.txt'
    try:
        process = subprocess.run(command, capture_output=True, timeout=300 if predictor else 180)
    except (OSError, subprocess.TimeoutExpired) as error:
        stdout.write_bytes(getattr(error, 'stdout', None) or b'')
        stderr.write_bytes(getattr(error, 'stderr', None) or str(error).encode())
        raise RuntimeError(f'Nemotron diagnostic failed to execute; evidence: {root}') from error
    stdout.write_bytes(process.stdout)
    stderr.write_bytes(process.stderr)
    if process.returncode:
        raise RuntimeError(f'Nemotron diagnostic exited {process.returncode}; evidence: {root}')
    if (
        _hash(executable) != binary_hash
        or _hash(fixture) != fixture_hash
        or _hash(case_path) != case_hash
    ):
        raise ValueError(f'Executable, fixture or case changed during execution; evidence: {root}')
    if Case.load(case_path).data != verified.data:
        raise ValueError(f'Fixed inputs changed during execution; evidence: {root}')
    prefix = 'nemotron_h_mtp_check ' if predictor else 'nemotron_h_check '
    lines = [
        line[len(prefix) :]
        for line in process.stdout.decode('utf-8').splitlines()
        if line.startswith(prefix)
    ]
    if len(lines) != 1:
        raise ValueError(f'Expected one diagnostic record; evidence: {root}')
    record = json.loads(lines[0])
    if record.get('architecture') != 'nemotron_h':
        raise ValueError('Unexpected diagnostic architecture')
    if predictor:
        actual_ids = _tensor(record, 'ids')
        if actual_ids.shape != ids.shape or not np.array_equal(actual_ids, ids):
            raise ValueError('Diagnostic token IDs differ from the case')
        if op in {'nemotron_mtp_head', 'nemotron_mtp_chain'} and not np.array_equal(
            _tensor(record, 'previous'), case.tensor('previous')
        ):
            raise ValueError('Diagnostic previous hidden differs from the case')
    outputs, checks = _observations(case, record)
    specs = {name: save_tensor(root, name, value) for name, value in outputs.items()}
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'native Nemotron diagnostics',
            'category': 'native_model_execution',
            'executable': str(executable),
            'executable_sha256': binary_hash,
            'fixture': fixture.name,
            'fixture_sha256': fixture_hash,
            'case_sha256': case_hash,
            'stdout_sha256': _hash(stdout),
            'stderr_sha256': _hash(stderr),
            'generator_sha256': _hash(Path(__file__)),
            'command': command,
            'negative_checks': 'original NumPy wrong-input formula compared with native observations',
        },
        role='native',
        checks=checks,
    )
