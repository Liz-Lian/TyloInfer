"""Freeze TQ format-boundary fixtures and their structural expectations."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import numpy as np

from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.inspection import inspect_tensor
from tyloinfer.formats.io import load, open_mmap, save
from tyloinfer.formats.tq import TQExpertTensor, TQTensor

from .case import Case, CaseWriter, save_tensor, write_result

_SOURCE = 'tests/test_tq_inference_formats.py'
_ENCODING_SOURCES = (
    'tyloinfer/formats/tq.py',
    'tyloinfer/formats/io.py',
    'tyloinfer/formats/header.py',
    'tyloinfer/formats/inspection.py',
)
_BOOLEAN_OUTPUTS = {
    'tq_format_expert': (
        'roundtrip_equal',
        'mmap_type',
        'is_row_granularity',
        'row_groups_increase',
    ),
    'tq_format_flat': (),
    'tq_format_legacy': ('load_rejected',),
    'tq_format_directory': ('from_bytes_rejected', 'inspect_rejected'),
}


def _row_tensor(bits, columns, salt=1):
    """Construct the original fixed TQ tensor without using a format loader."""
    width, group = 512, 16
    offsets, words = [], 0
    for bit in bits:
        offsets.append(words | ((bit - 1) << 32) | (0x3C00 << 40))
        words += width * bit // 32
    descriptors = np.asarray(offsets, dtype='<u8')
    scales = (
        np.full(width, salt, dtype='<f2')
        if columns
        else np.full((len(bits) * (width // group) + 1) // 2, 0x89, dtype=np.uint8)
    )
    packed = np.arange(words * 4, dtype=np.uint8)
    return TQTensor(len(bits), width, group, columns, descriptors, scales, packed)


def _container(writer, tensor, name, arch, version):
    """Save the original one-record container using the production writer."""
    path = writer.root / 'model.tq'
    save(path, FileHeader(version=version, model_arch=arch), {name: tensor})
    return path


def _expected_bytes(bits, columns):
    """Calculate row payload bytes from the fixture's fixed geometry."""
    scales = 512 * 2 if columns else (len(bits) * 32 + 1) // 2
    return 64 + len(bits) * 8 + scales + sum(512 * bit // 8 for bit in bits)


def _check(kind, **fields):
    """Describe an unexecuted format consumer check."""
    return {'kind': kind, **fields}


def _sources():
    """Identify the actual loaded production modules and their source bytes."""
    project = Path(__file__).resolve().parents[2]
    result = {}
    for name in _ENCODING_SOURCES:
        path = (project / name).resolve(strict=True)
        module = sys.modules.get(name.removesuffix('.py').replace('/', '.'))
        if module is None or Path(module.__file__).resolve() != path:
            raise ValueError(f'TQ production module source differs: {name}')
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _writer(root, name, kind, config, checks, test):
    """Create a case with source identity and explicit execution obligations."""
    path = Path(__file__).parents[1] / 'test_tq_inference_formats.py'
    source = {
        'generator': 'tests/numerics/tq_format_ref.py',
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'file': _SOURCE,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'encoding_sources_sha256': _sources(),
        'test': test,
        'category': 'format_and_loading_behavior',
    }
    return CaseWriter(
        root,
        name,
        'tq_format_' + kind,
        {
            **config,
            'actual_checks': checks,
            'asset_bytes_semantics': 'identity metadata only; not evidence of consumer acceptance or rejection',
        },
        source,
        actual_role='candidate',
    )


def reference(case: Case, root: Path) -> Path:
    """Recompute structural metadata from fixed assets and fixture geometry.

    Rejection, roundtrip and type checks require an external consumer; no
    inspected or loaded result is used as an expectation.
    """
    if case.data['source'].get('generator') != 'tests/numerics/tq_format_ref.py' or case.data[
        'op'
    ] not in ('tq_format_expert', 'tq_format_flat', 'tq_format_legacy', 'tq_format_directory'):
        raise ValueError('Unsupported TQ format case')
    config = case.data['config']
    payload = case.asset('payload')
    if hashlib.sha256(payload).hexdigest() != config['payload_sha256']:
        raise ValueError('Fixed payload differs from the case configuration')
    if case.data['op'] == 'tq_format_legacy':
        valid = case.asset('valid_container')
        dtype = config['dtype'].encode()
        expected = valid.replace(
            struct.pack('<I', 2) + b'TQ', struct.pack('<I', len(dtype)) + dtype, 1
        )
        if payload != expected:
            raise ValueError('Saved legacy label mutation differs from its source fixture')
    if case.data['op'] == 'tq_format_directory':
        valid = case.asset('valid_payload')
        mutation = config['mutation']
        if mutation == 'truncated':
            expected = valid[:-1]
        elif mutation == 'offset_zero':
            expected = valid[:64] + struct.pack('<Q', 0) + valid[72:]
        elif mutation == 'appended':
            expected = valid + b'x'
        else:
            raise ValueError('Unsupported expert directory mutation')
        if payload != expected:
            raise ValueError('Saved directory mutation differs from its source fixture')
    output = {'asset_bytes': np.asarray(len(payload), dtype=np.int64)}
    for name in _BOOLEAN_OUTPUTS[case.data['op']]:
        output[name] = np.asarray(True, dtype=np.bool_)
    if case.data['op'] in ('tq_format_expert', 'tq_format_flat'):
        columns = config['columns']
        if case.data['op'] == 'tq_format_expert':
            size = (
                64
                + 16 * len(config['bits'])
                + sum(_expected_bytes(bits, columns) for bits in config['bits'])
            )
        else:
            size = _expected_bytes(config['bits'], columns)
        if len(payload) != size:
            raise ValueError('Fixed TQ payload size differs from its structural geometry')
        output['shape'] = np.asarray(config['logical_shape'], dtype=np.int64)
        output['row_bytes_sum'] = np.asarray(size, dtype=np.float64)
    target = Path(root)
    specs = {name: save_tensor(target, name, value) for name, value in output.items()}
    return write_result(
        case,
        target,
        specs,
        {
            'framework': 'Python and NumPy',
            'numpy_version': np.__version__,
            'device': 'cpu',
            'arithmetic': 'fixed fixture geometry and serialized asset byte length; length is identity metadata only',
            'category': 'format_and_loading_behavior',
            'independence': 'manual fixture structure; no production parser or loader is used',
            'behavioral_expectations': 'declared source test contracts, not consumer execution',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


def candidate(case: Case, root: Path) -> Path:
    """Execute the original Python TQ container and inspection behaviors."""
    source, config, op = case.data['source'], case.data['config'], case.data['op']
    if (
        source.get('generator') != 'tests/numerics/tq_format_ref.py'
        or case.data.get('actual_role') != 'candidate'
        or op not in _BOOLEAN_OUTPUTS
    ):
        raise ValueError('Unsupported TQ format candidate')
    identity = _sources()
    if identity != source['encoding_sources_sha256']:
        raise ValueError('TQ production sources differ from the fixed case')
    payload = case.asset('payload')
    if hashlib.sha256(payload).hexdigest() != config['payload_sha256']:
        raise ValueError('Fixed TQ payload differs from its case')
    output = {'asset_bytes': np.asarray(len(payload), dtype=np.int64)}
    if op in ('tq_format_expert', 'tq_format_flat'):
        if op == 'tq_format_expert':
            restored = TQExpertTensor.from_bytes(payload)
            output['roundtrip_equal'] = np.asarray(
                restored.to_bytes() == payload
                and tuple(restored.shape) == tuple(config['logical_shape']),
                dtype=np.bool_,
            )
            shape = None
        else:
            shape = tuple(config['logical_shape'])
        with memoryview(payload) as view:
            inspected = inspect_tensor('TQ', view, rows=True, logical_shape=shape)
        output['shape'] = np.asarray(inspected.shape, dtype=np.int64)
        output['row_bytes_sum'] = np.asarray(inspected.row_bytes.sum(), dtype=np.float64)
        if op == 'tq_format_expert':
            output['is_row_granularity'] = np.asarray(
                inspected.granularity == 'row', dtype=np.bool_
            )
            output['row_groups_increase'] = np.asarray(
                inspected.row_bytes[:3].sum() < inspected.row_bytes[3:].sum(), dtype=np.bool_
            )
            with open_mmap(case.asset_path('container')) as store:
                output['mmap_type'] = np.asarray(
                    isinstance(store['model.block.0.mlp.experts.gate.weight'], TQExpertTensor),
                    dtype=np.bool_,
                )
            case.asset('container')
    elif op == 'tq_format_legacy':
        rejected = False
        path = case.asset_path('payload')
        try:
            load(path)
        except ValueError as error:
            if 'does not support tensor format' not in str(error):
                raise
            rejected = True
        output['load_rejected'] = np.asarray(rejected, dtype=np.bool_)
    else:
        rejected = False
        try:
            TQExpertTensor.from_bytes(payload)
        except ValueError:
            rejected = True
        output['from_bytes_rejected'] = np.asarray(rejected, dtype=np.bool_)
        rejected = False
        try:
            with memoryview(payload) as view:
                inspect_tensor('TQ', view, rows=True)
        except ValueError:
            rejected = True
        output['inspect_rejected'] = np.asarray(rejected, dtype=np.bool_)
    if _sources() != identity:
        raise ValueError('TQ production sources changed during candidate execution')
    case.asset('payload')
    specs = {name: save_tensor(root, name, value) for name, value in output.items()}
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'Python and NumPy',
            'numpy_version': np.__version__,
            'device': 'cpu',
            'category': 'python_candidate',
            'native_executed': False,
            'entrypoint': op,
            'encoding_sources_sha256': identity,
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        role='candidate',
    )


def _publish(writer):
    """Persist the case and calculate its fixed structural metadata."""
    writer.output('asset_bytes', (), 'int64', {'kind': 'exact'})
    for name in _BOOLEAN_OUTPUTS[writer.data['op']]:
        writer.output(name, (), 'bool', {'kind': 'exact'})
    case = writer.finish()
    reference(case, case.root / 'reference')
    candidate(case, case.root / 'candidate')
    return case.root / 'case.json'


def _valid(root, columns, expert):
    """Freeze one original expert or flattened row-bank fixture."""
    if expert:
        bits = [[1, 2, 3], [6, 7, 8]]
        tensor = TQExpertTensor(
            (_row_tensor(bits[0], columns), _row_tensor(bits[1], columns, salt=2)), 511
        )
        payload = tensor.to_bytes()
        shape = [2, 3, 511]
        checks = [
            _check(
                'from_bytes_roundtrip',
                asset='payload',
                byte_equal=True,
                type='TQExpertTensor',
                observation='roundtrip_equal',
            ),
            _check(
                'inspect_tensor',
                asset='payload',
                rows=True,
                granularity='row',
                observations=[
                    'shape',
                    'row_bytes_sum',
                    'is_row_granularity',
                    'row_groups_increase',
                ],
                first_three_sum_less_than_last_three=True,
            ),
            _check(
                'open_mmap_type',
                asset='container',
                tensor='model.block.0.mlp.experts.gate.weight',
                expected_type='TQExpertTensor',
                observation='mmap_type',
            ),
        ]
        test = 'test_expert_container_preserves_rows_scales_and_variable_sizes'
    else:
        bits = [1, 3, 6, 8]
        tensor = _row_tensor(bits, columns)
        payload = tensor.to_bytes()
        shape = [2, 2, 509]
        checks = [
            _check(
                'inspect_tensor',
                asset='payload',
                rows=True,
                logical_shape=shape,
                observations=['shape', 'row_bytes_sum'],
            )
        ]
        test = 'test_flat_row_bank_uses_logical_unpadded_shape'
    kind = 'expert' if expert else 'flat'
    writer = _writer(
        root,
        f'tq_format_{kind}_c{int(columns)}',
        kind,
        {
            'bits': bits,
            'columns': columns,
            'logical_shape': shape,
            'stored_width': 512,
            'payload_sha256': hashlib.sha256(payload).hexdigest(),
            'construction': 'TQTensor/TQExpertTensor.to_bytes',
        },
        checks,
        test,
    )
    writer.asset('payload', payload, encoding='RTQEXP01' if expert else 'RTRELL05')
    if expert:
        writer.asset(
            'container',
            _container(
                writer, tensor, 'model.block.0.mlp.experts.gate.weight', 'qwen4_exp_text', 2
            ),
            encoding='MFQ1',
        )
    writer.output('shape', (3,), 'int64', {'kind': 'exact'})
    writer.output('row_bytes_sum', (), 'float64', {'kind': 'close', 'atol': 1e-12, 'rtol': 1e-6})
    return _publish(writer)


def _legacy(root, dtype):
    """Freeze one original old-format directory-label mutation."""
    writer = _writer(
        root,
        f'tq_format_legacy_{dtype.replace("-", "_")}',
        'legacy',
        {'dtype': dtype},
        [
            _check(
                'load_reject',
                asset='payload',
                exception='ValueError',
                message_contains='does not support tensor format',
                observation='load_rejected',
            )
        ],
        'test_old_formats_are_rejected_at_the_container_boundary',
    )
    valid_path = _container(writer, _row_tensor([4], True), 'weight', 'test', 1)
    valid = valid_path.read_bytes()
    old = struct.pack('<I', 2) + b'TQ'
    new = struct.pack('<I', len(dtype)) + dtype.encode()
    malformed = valid.replace(old, new, 1)
    writer.data['config']['payload_sha256'] = hashlib.sha256(malformed).hexdigest()
    writer.data['config']['construction'] = 'io.save then replace first TQ dtype label'
    writer.asset('valid_container', valid_path, encoding='MFQ1')
    writer.asset('payload', malformed, encoding='MFQ1-mutated')
    return _publish(writer)


def _directory(root, mutation):
    """Freeze one original expert-directory corruption."""
    valid = TQExpertTensor((_row_tensor([2], True), _row_tensor([5], True, salt=2)), 511).to_bytes()
    if mutation == 'truncated':
        payload = valid[:-1]
    elif mutation == 'offset_zero':
        payload = valid[:64] + struct.pack('<Q', 0) + valid[72:]
    else:
        payload = valid + b'x'
    checks = [
        _check(
            'expert_from_bytes_reject',
            asset='payload',
            exception='ValueError',
            observation='from_bytes_rejected',
        ),
        _check(
            'inspect_tensor_reject',
            asset='payload',
            rows=True,
            exception='ValueError',
            observation='inspect_rejected',
        ),
    ]
    writer = _writer(
        root,
        f'tq_format_directory_{mutation}',
        'directory',
        {
            'mutation': mutation,
            'payload_sha256': hashlib.sha256(payload).hexdigest(),
            'construction': 'TQExpertTensor.to_bytes then mutate',
        },
        checks,
        'test_tq_directory_rejects_overlaps_truncation_and_shape_mismatch',
    )
    writer.asset('valid_payload', valid, encoding='RTQEXP01')
    writer.asset('payload', payload, encoding='RTQEXP01-mutated')
    return _publish(writer)


def generate(root: Path) -> list[Path]:
    """Freeze all original TQ format-boundary cases without loading an engine."""
    results = [
        _valid(root, columns, expert) for expert in (True, False) for columns in (False, True)
    ]
    results.extend(
        _legacy(root, dtype)
        for dtype in ('NINT', 'NVQ', 'MFE', 'MXFP4-SQ', 'MXFP8-SQ', 'FP8-128SQ')
    )
    results.extend(
        _directory(root, mutation) for mutation in ('truncated', 'offset_zero', 'appended')
    )
    return results
