"""Execute fixed numerical cases through the native worker and publish its observations."""

from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from .case import _DTYPES, Case, _read_json, save_tensor, write_result

_STDERR_LIMIT = 65536


def _outputs(case, raw):
    """Validate worker metadata and return the exact logical output arrays."""
    if raw.is_symlink() or not raw.is_dir() or raw.resolve().parent != raw.parent.resolve():
        raise ValueError('Worker evidence directory must be inside the result directory')
    if (raw / 'raw.json').resolve().parent != raw.resolve():
        raise ValueError('Worker manifest escapes the evidence directory')
    manifest = _read_json(raw / 'raw.json')
    if not isinstance(manifest, dict):
        raise ValueError('Worker manifest must be an object')
    outputs = manifest.get('outputs')
    contracts = {**case.data['outputs'], **case.data.get('auxiliary', {})}
    if not isinstance(outputs, dict) or set(outputs) != set(contracts):
        raise ValueError('Worker observations do not match the case')
    execution = manifest.get('execution')
    if not isinstance(execution, dict) or not execution:
        raise ValueError('Worker execution provenance is required')
    checks = manifest.get('checks', {})
    if not isinstance(checks, dict):
        raise ValueError('Worker checks must be an object')
    descriptors = case.data['config'].get('native_checks', [])
    names = {f'native_{index}' for index in range(len(descriptors))}
    central = {
        f'native_{index}' for index, descriptor in enumerate(descriptors)
        if isinstance(descriptor, dict) and descriptor.get('kind') == 'compare_observations'
    }
    if set(checks) - names:
        raise ValueError('Worker reported an unknown native check')
    if set(checks) & central:
        raise ValueError('Worker cannot claim a centrally evaluated observation comparison')
    for name, check in checks.items():
        if not isinstance(check, dict) or set(check) - {'status', 'evidence'}:
            raise ValueError(f'Invalid native behavior check: {name}')
        status = check.get('status')
        if status not in {'pass', 'fail', 'not_executed'}:
            raise ValueError(f'Invalid native behavior status: {name}')
        if (status in {'pass', 'fail'} or 'evidence' in check) and (
            not isinstance(check.get('evidence'), dict) or not check['evidence']
        ):
            raise ValueError(f'Native behavior check requires evidence: {name}')
    checks = {
        name: checks.get(name, {'status': 'not_executed'}) for name in sorted(names - central)
    }

    values = {}
    used = {(raw / 'raw.json').resolve()}
    for name, contract in contracts.items():
        spec = outputs[name]
        if not isinstance(spec, dict):
            raise ValueError(f'Invalid worker observation: {name}')
        dtype = contract.get('actual_dtype', contract['dtype'])
        encoding = np.dtype(_DTYPES[dtype])
        if (
            spec.get('shape') != contract['shape']
            or not isinstance(spec.get('shape'), list)
            or any(type(part) is not int for part in spec['shape'])
            or spec.get('dtype') != dtype
            or spec.get('encoding') != encoding.str
            or spec.get('layout') != 'contiguous'
            or spec.get('order') != 'C'
            or type(spec.get('nbytes')) is not int
            or spec['nbytes'] != math.prod(contract['shape']) * encoding.itemsize
        ):
            raise ValueError(f'Worker observation contract differs: {name}')
        filename = spec.get('file')
        if not isinstance(filename, str) or not filename or Path(filename).is_absolute():
            raise ValueError(f'Invalid worker output path: {name}')
        path = (raw / filename).resolve()
        if not path.is_relative_to(raw.resolve()) or path == raw.resolve() or path in used:
            raise ValueError(f'Escaping or duplicate worker output path: {name}')
        used.add(path)
        if path.stat().st_size != spec['nbytes']:
            raise ValueError(f'Worker output size differs: {name}')
        data = path.read_bytes()
        if len(data) != spec['nbytes']:
            raise ValueError(f'Worker output changed while reading: {name}')
        if dtype == 'bool' and any(byte > 1 for byte in data):
            raise ValueError(f'Invalid boolean encoding: {name}')
        values[name] = np.frombuffer(data, dtype=encoding).reshape(contract['shape']).copy()
    return values, execution, checks


def run(case: Case, root: Path, executable: Path) -> Path:
    """Run a verified case once and publish the native result in a new directory.

    The worker owns the raw evidence directory. Check evidence is structurally
    validated but does not independently prove that the worker executed it.
    """
    if not isinstance(case, Case):
        raise TypeError('case must be a Case')
    case_path = case.root / 'case.json'
    verified = Case.load(case_path)
    if verified.data != case.data or verified.root != case.root.resolve():
        raise ValueError('Case changed since it was loaded')
    if verified.data.get('actual_role', 'native') != 'native':
        raise ValueError('Native worker cannot produce a candidate result')
    executable = Path(executable).resolve(strict=True)
    if not executable.is_file():
        raise ValueError('Worker executable must be a file')
    worker_hash = hashlib.sha256(executable.read_bytes()).hexdigest()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    root = root.resolve()
    raw = root / 'raw'
    stderr_path = root / 'worker.stderr.txt'
    try:
        process = subprocess.Popen(
            [str(executable), str(case_path.resolve()), str(raw)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as error:
        stderr_path.write_text(str(error), encoding='utf-8')
        raise RuntimeError(f'Native worker could not start; evidence: {root}') from error
    retained = bytearray()
    try:
        assert process.stderr is not None
        with process.stderr:
            while chunk := process.stderr.read(8192):
                retained.extend(chunk[: max(0, _STDERR_LIMIT - len(retained))])
        status = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        stderr_path.write_bytes(retained)
    if status:
        raise RuntimeError(f'Native worker exited with status {status}; evidence: {root}')
    if Case.load(case_path).data != verified.data:
        raise ValueError(f'Case changed during native execution; evidence: {root}')
    if hashlib.sha256(executable.read_bytes()).hexdigest() != worker_hash:
        raise ValueError(f'Worker changed during native execution; evidence: {root}')
    values, execution, checks = _outputs(verified, raw)
    execution['worker'] = {'file': str(executable), 'sha256': worker_hash}
    outputs = {}
    for name, value in values.items():
        contract = {**verified.data['outputs'], **verified.data.get('auxiliary', {})}[name]
        outputs[name] = save_tensor(
            root, name, value, dtype=contract.get('actual_dtype', contract['dtype'])
        )
    return write_result(verified, root, outputs, execution, role='native', checks=checks)


def main(argv=None):
    """Run the native worker from the command line and print the result path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--exe', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run(Case.load(args.case), args.output, args.exe)
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        print(f'native execution failed: {error}', file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
