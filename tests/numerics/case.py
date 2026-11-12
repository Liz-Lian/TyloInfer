"""Persist numerical cases and results without importing a reference framework."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_FORMAT = 'tyloinfer.numerics'
_DTYPES = {
    'float16': '<f2',
    'bfloat16': '<f4',
    'float32': '<f4',
    'float64': '<f8',
    'int8': 'i1',
    'uint8': 'u1',
    'int16': '<i2',
    'uint16': '<u2',
    'int32': '<i4',
    'uint32': '<u4',
    'int64': '<i8',
    'uint64': '<u8',
    'bool': '?',
}


def _name(value):
    """Require a single portable artifact name."""
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*', value):
        raise ValueError(f'Invalid artifact name: {value!r}')
    return value


def _json(value):
    """Encode a stable identity without non-finite JSON numbers."""
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _read_json(path):
    """Read JSON while rejecting duplicate keys and non-standard numbers."""

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f'Duplicate metadata key: {key}')
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f'Invalid JSON number: {value}')

    return json.loads(
        Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant
    )


def _write_json(path, data):
    """Create metadata once; never replace a fixed case or baseline."""
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n'
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)


def _path(root, value):
    """Resolve a recorded file inside its artifact directory."""
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('Artifact path must be relative')
    root = Path(root).resolve()
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Artifact path escapes its directory')
    return path


def _shape(value):
    """Validate a logical shape, including scalar and empty tensors."""
    if not isinstance(value, list) or any(type(n) is not int or n < 0 for n in value):
        raise ValueError('Invalid tensor shape')
    return tuple(value)


def _layout(value, shape):
    """Validate the original element strides and offset, independent of file encoding."""
    if value == 'contiguous':
        return
    if not isinstance(value, dict) or set(value) != {'strides', 'offset'}:
        raise ValueError('Invalid logical layout')
    if (
        type(value['offset']) is not int
        or value['offset'] < 0
        or not isinstance(value['strides'], list)
        or len(value['strides']) != len(shape)
        or any(type(n) is not int or n < 0 for n in value['strides'])
    ):
        raise ValueError('Invalid logical strides or offset')


def _blob(root, spec):
    """Read a complete file only after checking its length and content identity."""
    if not isinstance(spec, dict) or type(spec.get('nbytes')) is not int or spec['nbytes'] < 0:
        raise ValueError('Invalid artifact byte count')
    digest = spec.get('sha256')
    if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
        raise ValueError('Invalid artifact SHA-256')
    path = _path(root, spec['file'])
    if path.stat().st_size != spec['nbytes']:
        raise ValueError(f'Incomplete artifact: {spec["file"]}')
    data = path.read_bytes()
    if len(data) != spec['nbytes'] or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Artifact content changed: {spec["file"]}')
    return data


def _save_blob(root, name, data, encoding):
    """Create a content-addressed description for one immutable binary file."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    filename = _name(name) + '.bin'
    with (root / filename).open('xb') as stream:
        stream.write(data)
    return {
        'file': filename,
        'encoding': encoding,
        'nbytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


def save_tensor(root, name, value, dtype=None, layout=None):
    """Save logical values in contiguous little-endian form without loss of precision.

    BF16 is expanded exactly to F32 and retains its logical dtype. Other logical
    dtypes must match the supplied array; implicit narrowing is rejected.
    """
    value = np.asarray(value)
    logical = dtype or value.dtype.name
    if logical not in _DTYPES:
        raise ValueError(f'Unsupported tensor dtype: {logical}')
    storage = np.dtype(_DTYPES[logical])
    if value.dtype.newbyteorder('<') != storage:
        raise ValueError(f'{logical} must be supplied as {storage.name}, without implicit casts')
    value = np.array(value, dtype=storage, order='C', copy=True)
    if logical == 'bfloat16' and np.any(value.view('<u4') & np.uint32(0xFFFF)):
        raise ValueError('Expanded BF16 contains values not representable in BF16')
    original_layout = layout or 'contiguous'
    _layout(original_layout, value.shape)
    spec = _save_blob(root, name, value.tobytes(order='C'), storage.str)
    spec.update(shape=list(value.shape), dtype=logical, layout=original_layout, order='C')
    return spec


def load_tensor(root, spec):
    """Read verified logical values without importing Torch or changing precision."""
    shape = _shape(spec['shape'])
    logical = spec['dtype']
    if logical not in _DTYPES or spec.get('encoding') != np.dtype(_DTYPES[logical]).str:
        raise ValueError('Tensor dtype and binary encoding differ')
    _layout(spec['layout'], shape)
    if spec.get('order') != 'C':
        raise ValueError('Tensor files must use contiguous logical order')
    dtype = np.dtype(_DTYPES[logical])
    if math.prod(shape) * dtype.itemsize != spec['nbytes']:
        raise ValueError('Tensor shape and byte count differ')
    value = np.frombuffer(_blob(root, spec), dtype=dtype).reshape(shape).copy()
    if logical == 'bfloat16' and np.any(value.view('<u4') & np.uint32(0xFFFF)):
        raise ValueError('Invalid expanded BF16 values')
    return value


def _rule(rule):
    """Validate original numerical criteria without inventing new tolerances."""
    if not isinstance(rule, dict):
        raise ValueError('Comparison rule must be an object')
    kind = rule.get('kind')
    if kind in {'exact', 'bits'}:
        return
    fields = (
        ('atol', 'rtol') if kind == 'close' else ('limit', 'floor') if kind == 'rms' else ('limit',)
    )
    if kind not in {'close', 'rms', 'relative_l2', 'max_abs'}:
        raise ValueError(f'Unsupported comparison rule: {kind}')
    for name in fields:
        value = rule.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid comparison parameter: {name}')
    if kind != 'close' and (rule.get('strict') is not True or rule.get('precision') != 'float32'):
        raise ValueError('Unsupported aggregate comparison semantics')
    if kind == 'rms' and rule['floor'] == 0:
        raise ValueError('RMS requires its original positive denominator floor')


@dataclass(frozen=True)
class Case:
    """A fixed case whose identity includes configuration and all input identities."""

    root: Path
    data: dict

    @classmethod
    def load(cls, path):
        """Validate metadata and all inputs before exposing a fixed case."""
        path = Path(path)
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError('Case metadata must be an object')
        identity = {key: value for key, value in data.items() if key != 'id'}
        if (
            data.get('format') != _FORMAT
            or data.get('version') != 1
            or data.get('kind') != 'case'
            or data.get('id') != hashlib.sha256(_json(identity)).hexdigest()
        ):
            raise ValueError('Case format or identity does not match')
        _name(data['name'])
        if not isinstance(data['op'], str) or not data['op']:
            raise ValueError('Case operation is required')
        for field in ('config', 'source', 'inputs', 'assets', 'outputs'):
            if not isinstance(data.get(field), dict):
                raise ValueError(f'Case {field} must be an object')
        if not data['source'] or not data['outputs']:
            raise ValueError('Case requires a reference source and output observations')
        case = cls(path.resolve().parent, data)
        for name in data['inputs']:
            _name(name)
            case.tensor(name)
        for name in data['assets']:
            _name(name)
            case.asset(name)
        for name, contract in data['outputs'].items():
            _name(name)
            _shape(contract['shape'])
            if (
                contract['dtype'] not in _DTYPES
                or contract.get('actual_dtype', contract['dtype']) not in _DTYPES
                or contract.get('layout') != 'contiguous'
                or type(contract.get('allow_infinity', False)) is not bool
            ):
                raise ValueError(f'Invalid output contract: {name}')
            _rule(contract['rule'])
            if (
                contract['rule']['kind'] == 'bits'
                and contract.get('actual_dtype', contract['dtype']) != contract['dtype']
            ):
                raise ValueError('Bit-exact observations require matching logical dtypes')
        return case

    def tensor(self, name):
        """Read a named input after checking its stored contents."""
        return load_tensor(self.root, self.data['inputs'][name])

    def asset(self, name):
        """Read fixed packed weights or another opaque case asset."""
        return _blob(self.root, self.data['assets'][name])


class CaseWriter:
    """Build one new fixed case, refusing to overwrite any existing case directory."""

    def __init__(self, root, name, op, config, source):
        self.root = Path(root) / _name(name)
        self.root.mkdir(parents=True, exist_ok=False)
        self.data = {
            'format': _FORMAT,
            'version': 1,
            'kind': 'case',
            'name': name,
            'op': op,
            'config': config,
            'source': source,
            'inputs': {},
            'assets': {},
            'outputs': {},
        }
        self.finished = False

    def _add(self, field, name, spec):
        """Require unique entries and reject mutation after case publication."""
        if self.finished or name in self.data[field]:
            raise ValueError('Case is already published or observation name is duplicated')
        self.data[field][_name(name)] = spec

    def input(self, name, value, dtype=None, layout=None):
        """Persist an actual input with its original logical layout."""
        if self.finished or name in self.data['inputs']:
            raise ValueError('Input already exists or case is published')
        spec = save_tensor(self.root / 'inputs', name, value, dtype, layout)
        spec['file'] = 'inputs/' + spec['file']
        self._add('inputs', name, spec)
        return spec

    def asset(self, name, value, encoding='binary'):
        """Store packed bytes or register an existing file inside this case directory."""
        if self.finished or name in self.data['assets']:
            raise ValueError('Asset already exists or case is published')
        if isinstance(value, Path):
            path = value.resolve()
            relative = path.relative_to(self.root.resolve()).as_posix()
            data = path.read_bytes()
            spec = {
                'file': relative,
                'encoding': encoding,
                'nbytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(),
            }
        else:
            spec = _save_blob(self.root / 'assets', name, bytes(value), encoding)
            spec['file'] = 'assets/' + spec['file']
        self._add('assets', name, spec)
        return spec

    def output(
        self, name, shape, dtype, rule, layout='contiguous', actual_dtype=None, allow_infinity=False
    ):
        """Declare a result observation and the original native/reference precisions."""
        spec = {
            'shape': list(shape),
            'dtype': dtype,
            'layout': layout,
            'rule': rule,
            'allow_infinity': allow_infinity,
        }
        if actual_dtype is not None:
            spec['actual_dtype'] = actual_dtype
        self._add('outputs', name, spec)

    def finish(self):
        """Publish metadata only after inputs are stored; return a validated case."""
        if self.finished:
            raise ValueError('Case is already published')
        self.data['id'] = hashlib.sha256(_json(self.data)).hexdigest()
        path = self.root / 'case.json'
        _write_json(path, self.data)
        self.finished = True
        return Case.load(path)


def _validate_result(case, root, result, role):
    """Validate complete observations against the case before numeric comparison."""
    if (
        role not in {'reference', 'native'}
        or not isinstance(result, dict)
        or result.get('format') != _FORMAT
        or result.get('version') != 1
        or result.get('kind') != 'result'
        or result.get('role') != role
        or result.get('case_id') != case.data['id']
    ):
        raise ValueError('Result format, role or case identity does not match')
    if not isinstance(result.get('execution'), dict) or not result['execution']:
        raise ValueError('Result execution provenance is required')
    outputs = result.get('outputs')
    if not isinstance(outputs, dict) or set(outputs) != set(case.data['outputs']):
        raise ValueError('Result observations do not match the case')
    for name, spec in outputs.items():
        contract = case.data['outputs'][name]
        dtype = (
            contract['dtype']
            if role == 'reference'
            else contract.get('actual_dtype', contract['dtype'])
        )
        if (
            spec['shape'] != contract['shape']
            or spec['dtype'] != dtype
            or spec['layout'] != contract['layout']
        ):
            raise ValueError(f'Result shape, dtype or layout does not match: {name}')
        load_tensor(root, spec)


def write_result(case, root, outputs, execution, role='reference'):
    """Publish a validated reference or native result without replacing a baseline."""
    result = {
        'format': _FORMAT,
        'version': 1,
        'kind': 'result',
        'role': role,
        'case_id': case.data['id'],
        'execution': execution,
        'outputs': outputs,
    }
    _validate_result(case, root, result, role)
    path = Path(root) / 'result.json'
    _write_json(path, result)
    return path


def read_result(case, path, role):
    """Read and verify a result's identity, observations and binary payloads."""
    path = Path(path)
    result = _read_json(path)
    _validate_result(case, path.resolve().parent, result, role)
    return result
