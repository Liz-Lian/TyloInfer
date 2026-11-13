"""Compare two persisted prefill traces without imposing numerical thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


_DTYPES = {
    'float16': '<f2', 'float32': '<f4', 'float64': '<f8',
    'bfloat16': '<u2', 'int8': 'i1', 'uint8': 'u1', 'int16': '<i2',
    'int32': '<i4', 'int64': '<i8', 'bool': '?',
}
_FLOATS = {'float16', 'float32', 'float64', 'bfloat16'}
_INTEGERS = {'int8', 'uint8', 'int16', 'int32', 'int64'}


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(root: Path) -> tuple[dict, dict]:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f'Trace is not a directory: {root}')
    metadata_path = root / 'trace.json'
    if metadata_path.resolve(strict=True).parent != root:
        raise ValueError('Trace metadata must remain inside its directory')
    metadata_bytes = metadata_path.read_bytes()
    meta = json.loads(metadata_bytes)
    if not isinstance(meta, dict) or meta.get('format') != 'tyloinfer.prefill-trace.v1':
        raise ValueError('Unsupported trace format')
    for key in ('tokens', 'chunk', 'repeat', 'layer'):
        if type(meta.get(key)) is not int or meta[key] < (1 if key in ('tokens', 'chunk') else 0):
            raise ValueError(f'Invalid trace {key}')
    if not isinstance(meta.get('stages'), list):
        raise ValueError('Trace stages must be a list')
    stages = {}
    for stage in meta['stages']:
        if not isinstance(stage, dict):
            raise ValueError('Invalid stage metadata')
        name, filename = stage.get('name'), stage.get('file')
        shape, dtype, nbytes = stage.get('shape'), stage.get('dtype'), stage.get('nbytes')
        if not isinstance(name, str) or not name or name in stages:
            raise ValueError(f'Invalid or duplicate stage name: {name!r}')
        if (not isinstance(filename, str) or not filename or '\\' in filename
                or ':' in filename):
            raise ValueError(f'Invalid stage path: {filename!r}')
        relative = Path(filename)
        if relative.is_absolute() or any(part in ('.', '..') for part in relative.parts):
            raise ValueError(f'Invalid stage path: {filename!r}')
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Stage path escapes trace directory: {filename!r}')
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(dtype, str) or dtype not in _DTYPES
                or type(nbytes) is not int or nbytes < 0):
            raise ValueError(f'Invalid stage shape, dtype, or size: {name}')
        expected = math.prod(shape) * np.dtype(_DTYPES[dtype]).itemsize
        if nbytes != expected or path.stat().st_size != expected:
            raise ValueError(f'Stage byte count mismatch: {name}')
        data = path.read_bytes()
        if len(data) != expected:
            raise ValueError(f'Stage changed while reading: {name}')
        if dtype == 'bool' and any(byte > 1 for byte in data):
            raise ValueError(f'Invalid boolean encoding: {name}')
        stages[name] = {'metadata': stage, 'data': data, 'sha256': _hash(data)}
    if 'ids' not in stages:
        raise ValueError('Trace is missing ids')
    ids = stages['ids']['metadata']
    if ids['dtype'] != 'int64' or ids['shape'] != [1, meta['tokens']]:
        raise ValueError('Trace ids must be int64 [1,tokens]')
    return {'path': str(root), 'sha256': _hash(metadata_bytes),
            'tokens': meta['tokens'], 'chunk': meta['chunk'],
            'repeat': meta['repeat'], 'layer': meta['layer']}, stages


def _values(data: bytes, dtype: str) -> np.ndarray:
    values = np.frombuffer(data, dtype=_DTYPES[dtype])
    if dtype == 'bfloat16':
        return (values.astype(np.uint32) << 16).view(np.float32)
    return values


def _compare(first: dict, second: dict) -> dict:
    left, right = first['metadata'], second['metadata']
    result = {'first_sha256': first['sha256'], 'second_sha256': second['sha256'],
              'first_shape': left['shape'], 'second_shape': right['shape'],
              'first_dtype': left['dtype'], 'second_dtype': right['dtype'],
              'compatible': left['shape'] == right['shape'] and left['dtype'] == right['dtype']}
    if not result['compatible']:
        return result
    a, b = first['data'], second['data']
    width = np.dtype(_DTYPES[left['dtype']]).itemsize
    bits_a = np.frombuffer(a, dtype=np.uint8).reshape(-1, width)
    bits_b = np.frombuffer(b, dtype=np.uint8).reshape(-1, width)
    result['byte_exact'] = a == b
    result['different_elements'] = int(np.count_nonzero(np.any(bits_a != bits_b, axis=1)))
    dtype = left['dtype']
    x, y = _values(a, dtype), _values(b, dtype)
    if dtype in _FLOATS:
        finite_x, finite_y = np.isfinite(x), np.isfinite(y)
        nonfinite = ~finite_x
        same_mask = np.array_equal(finite_x, finite_y)
        result['nonfinite_mask_equal'] = bool(same_mask)
        result['nonfinite_kind_equal'] = bool(
            same_mask and np.array_equal(np.isnan(x[nonfinite]), np.isnan(y[nonfinite])))
        result['nonfinite_sign_equal'] = bool(
            same_mask and np.array_equal(np.signbit(x[nonfinite]), np.signbit(y[nonfinite])))
        common = finite_x & finite_y
        result['finite_count'] = int(np.count_nonzero(common))
        if np.any(common):
            xf, yf = x[common].astype(np.float64), y[common].astype(np.float64)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                delta = yf - xf
                scale = float(np.sum(xf * xf))
                maximum = float(np.max(np.abs(delta)))
                relative = float(np.sqrt(np.sum(delta * delta) / scale)) if scale else float('nan')
            result['finite_max_abs'] = maximum if math.isfinite(maximum) else None
            result['finite_relative_rms'] = relative if math.isfinite(relative) else None
        else:
            result['finite_max_abs'] = None
            result['finite_relative_rms'] = None
    if dtype in _INTEGERS:
        maximum = 0
        squared_error = 0
        squared_reference = 0
        for av, bv in zip(x, y):
            av, bv = int(av), int(bv)
            delta = bv - av
            maximum = max(maximum, abs(delta))
            squared_error += delta * delta
            squared_reference += av * av
        result['finite_max_abs'] = maximum if x.size else None
        result['finite_relative_rms'] = (
            math.sqrt(squared_error / squared_reference) if squared_reference else None)
        if left['name'].endswith('selected'):
            result['sorted_unique_ids_equal'] = bool(np.array_equal(np.unique(x), np.unique(y)))
    return result


def compare(first: Path, second: Path) -> dict:
    """Validate two traces and return threshold-free stage observations."""
    left, left_stages = _read(Path(first))
    right, right_stages = _read(Path(second))
    if left['tokens'] != right['tokens'] or left['layer'] != right['layer']:
        raise ValueError('Trace tokens or layer differ')
    if left_stages['ids']['data'] != right_stages['ids']['data']:
        raise ValueError('Trace input IDs differ')
    left_names, right_names = set(left_stages), set(right_stages)
    names = sorted(left_names | right_names)
    stages = {}
    for name in names:
        if name in left_stages and name in right_stages:
            stages[name] = _compare(left_stages[name], right_stages[name])
        else:
            item = left_stages.get(name) or right_stages[name]
            stages[name] = {'first_sha256': item['sha256'] if name in left_stages else None,
                            'second_sha256': item['sha256'] if name in right_stages else None}
    order = list(left_stages) + [name for name in right_stages if name not in left_stages]
    first_difference = None
    for name in order:
        if name not in left_stages or name not in right_stages:
            first_difference = {'name': name, 'reason': 'missing'}
        elif not stages[name]['compatible']:
            first_difference = {'name': name, 'reason': 'incompatible'}
        elif not stages[name]['byte_exact']:
            first_difference = {'name': name, 'reason': 'bytes'}
        if first_difference is not None:
            break
    first_byte_difference = (first_difference['name']
                             if first_difference and first_difference['reason'] == 'bytes' else None)
    return {'status': 'observe', 'coverage': 'observed_stages_only',
            'completeness': 'unverified', 'producer_sha256': _hash(Path(__file__).read_bytes()),
            'first': left, 'second': right,
            'stage_names_match': left_names == right_names,
            'first_only': sorted(left_names - right_names),
            'second_only': sorted(right_names - left_names),
            'first_difference': first_difference,
            'first_byte_difference': first_byte_difference, 'stages': stages}


def main() -> None:
    """Write one exclusive JSON report for two existing trace directories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.first, args.second)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
