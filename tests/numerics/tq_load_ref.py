"""Freeze independent native TQ container and row-loader observations."""

import hashlib
import struct
from pathlib import Path

import numpy as np

from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import save
from tyloinfer.formats.tq import TQExpertTensor

from .case import Case, CaseWriter, save_tensor, write_result
from .tq_format_ref import _row_tensor, _sources

_NAME = 'weight'


def reference(case: Case, root: Path) -> Path:
    """Publish declared loading expectations for a verified fixed container."""
    if case.data['source'].get('generator') != 'tests/numerics/tq_load_ref.py':
        raise ValueError('Unsupported TQ loading case')
    identity = _sources()
    if identity != case.data['source']['encoding_sources_sha256']:
        raise ValueError('TQ fixture sources differ from the case')
    op, config = case.data['op'], case.data['config']
    if op == 'tq_load_valid':
        output = {
            'shape': np.asarray(config['logical_shape'], dtype=np.int64),
            'payload_equal': np.asarray(True, dtype=np.bool_),
            'rows_equal': np.asarray(True, dtype=np.bool_),
            'projection_loaded': np.asarray(True, dtype=np.bool_),
        }
    elif op in ('tq_load_legacy', 'tq_load_directory'):
        output = {'rejected': np.asarray(True, dtype=np.bool_)}
    else:
        raise ValueError('Unsupported TQ loading operation')
    specs = {name: save_tensor(root, name, value) for name, value in output.items()}
    if _sources() != identity:
        raise ValueError('TQ fixture sources changed during reference calculation')
    return write_result(
        case,
        root,
        specs,
        {
            'framework': 'declared loading behavior',
            'device': 'cpu',
            'independence': 'declared structural expectations, not native execution',
            'rejection_scope': 'native TQ-specific consumer; compatible MFQ source acceptance is recorded separately',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )


def _case(root, label, kind, config, tensor, payload, projection, mutate=None):
    """Persist one container and its fixed payload before publishing its case."""
    identity = _sources()
    writer = CaseWriter(
        root,
        label,
        'tq_load_' + kind,
        config,
        {
            'generator': 'tests/numerics/tq_load_ref.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fixture_generator_sha256': hashlib.sha256(
                Path(__file__).with_name('tq_format_ref.py').read_bytes()
            ).hexdigest(),
            'test': 'tests/test_tq_inference_formats.py',
            'test_sha256': hashlib.sha256(
                (Path(__file__).parents[1] / 'test_tq_inference_formats.py').read_bytes()
            ).hexdigest(),
            'encoding_sources_sha256': identity,
            'category': 'native_tq_source_and_loading',
        },
    )
    model = writer.root / 'model.tq'
    tensors = {_NAME: tensor}
    default_shape = (
        list(tensor.shape) if isinstance(tensor, TQExpertTensor) else [tensor.rows, tensor.width]
    )
    shapes = {_NAME: config.get('logical_shape', default_shape)}
    if projection is not None:
        tensors['projection'] = projection
        shapes['projection'] = [projection.rows, config['logical_shape'][-1]]
    save(model, FileHeader(version=2, model_arch='test', extra={'tq_shapes': shapes}), tensors)
    if mutate is not None:
        raw = model.read_bytes()
        if kind == 'legacy':
            raw = raw.replace(
                struct.pack('<I', 2) + b'TQ', struct.pack('<I', len(mutate)) + mutate.encode(), 1
            )
        else:
            original = tensor.to_bytes()
            if not raw.endswith(original) or raw[
                -len(original) - 8 : -len(original)
            ] != struct.pack('<Q', len(original)):
                raise ValueError('Unexpected single-record TQ container layout')
            raw = raw[: -len(original) - 8] + struct.pack('<Q', len(payload)) + payload
        model.write_bytes(raw)
    writer.asset('model', model, encoding='MFQ1')
    writer.asset('payload', payload, encoding='TQ')
    if kind == 'valid':
        parts = tensor.experts if isinstance(tensor, TQExpertTensor) else (tensor,)
        for index, part in enumerate(parts):
            writer.asset(f'part{index}', part.to_bytes(), encoding='RTRELL05')
        writer.output('shape', (len(config['logical_shape']),), 'int64', {'kind': 'exact'})
        for name in ('payload_equal', 'rows_equal', 'projection_loaded'):
            writer.output(name, (), 'bool', {'kind': 'exact'})
    else:
        writer.output('rejected', (), 'bool', {'kind': 'exact'})
    if _sources() != identity:
        raise ValueError('TQ sources changed while preparing the case')
    case = writer.finish()
    reference(case, case.root / 'reference')
    return case.root / 'case.json'


def generate(root: Path) -> list[Path]:
    """Freeze native TQ valid, legacy-label and corrupt-directory cases."""
    root = Path(root)
    paths = []
    for expert in (True, False):
        for columns in (False, True):
            first = _row_tensor([1, 2, 3] if expert else [1, 3, 6, 8], columns)
            tensor = (
                TQExpertTensor((first, _row_tensor([6, 7, 8], columns, salt=2)), 511)
                if expert
                else first
            )
            shape = [2, 3, 511] if expert else [2, 2, 509]
            paths.append(
                _case(
                    root,
                    f'tq_load_{"expert" if expert else "flat"}_c{int(columns)}',
                    'valid',
                    {'logical_shape': shape, 'tensor': _NAME},
                    tensor,
                    tensor.to_bytes(),
                    first,
                )
            )
    for dtype in ('NINT', 'NVQ', 'MFE', 'MXFP4-SQ', 'MXFP8-SQ', 'FP8-128SQ'):
        tensor = _row_tensor([4], True)
        paths.append(
            _case(
                root,
                'tq_load_legacy_' + dtype.replace('-', '_'),
                'legacy',
                {'dtype': dtype, 'tensor': _NAME},
                tensor,
                tensor.to_bytes(),
                None,
                dtype,
            )
        )
    tensor = TQExpertTensor((_row_tensor([2], True), _row_tensor([5], True, salt=2)), 511)
    valid = tensor.to_bytes()
    for name, payload in (
        ('truncated', valid[:-1]),
        ('offset_zero', valid[:64] + struct.pack('<Q', 0) + valid[72:]),
        ('appended', valid + b'x'),
    ):
        paths.append(
            _case(
                root,
                'tq_load_directory_' + name,
                'directory',
                {'mutation': name, 'tensor': _NAME, 'logical_shape': [2, 1, 511]},
                tensor,
                payload,
                None,
                name,
            )
        )
    return paths
