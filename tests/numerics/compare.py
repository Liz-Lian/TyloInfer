"""Compare persisted native outputs against fixed numerical references."""

import math
from fractions import Fraction
from pathlib import Path

import numpy as np

from .case import Case, load_tensor, read_result


def _number(value):
    """Convert a scalar to a JSON-safe diagnostic value."""
    value = value.item() if isinstance(value, np.generic) else value
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')
    return value


def _differences(actual, reference, matching_inf):
    """Compute absolute differences without overflowing integer arithmetic."""
    if actual.dtype.kind in 'biu' and reference.dtype.kind in 'biu':
        return np.asarray(np.abs(actual.astype(object) - reference.astype(object)), dtype=object)
    with np.errstate(over='ignore', invalid='ignore'):
        differences = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    return np.where(matching_inf, 0.0, differences)


def _maximum(differences, actual, reference):
    """Find the largest error and its corresponding values, including nonfinite errors."""
    if not differences.size:
        return None, None, None, None
    # An invalid difference is diagnostic evidence rather than a numeric maximum.
    if differences.dtype != object:
        invalid = np.isnan(differences)
        flat = int(np.flatnonzero(invalid)[0]) if invalid.any() else int(np.argmax(differences))
    else:
        flat = int(np.argmax(differences))
    index = np.unravel_index(flat, differences.shape)
    return (
        _number(differences[index]),
        [int(part) for part in index],
        _number(actual[index]),
        _number(reference[index]),
    )


def _close(actual, reference, differences, valid, matching_inf, rule, allow_infinity):
    """Apply absolute and relative tolerances without accepting overflow as equality."""
    with np.errstate(over='ignore', invalid='ignore'):
        threshold = rule['atol'] + rule['rtol'] * np.abs(reference.astype(np.float64))
        passed = np.asarray(
            valid & ((differences <= threshold) | (matching_inf & allow_infinity))
        ).copy()
    if differences.dtype != object:
        overflow = (
            valid
            & np.isfinite(actual)
            & np.isfinite(reference)
            & (~np.isfinite(differences) | ~np.isfinite(threshold))
        )
        for index in zip(*np.nonzero(np.atleast_1d(overflow)), strict=True):
            index = () if actual.ndim == 0 else index
            a = Fraction(_number(actual[index]))
            b = Fraction(_number(reference[index]))
            limit = Fraction(rule['atol']) + Fraction(rule['rtol']) * abs(b)
            passed[index] = abs(a - b) <= limit
    return passed


def _evaluate(actual, reference, contract):
    """Evaluate one output and return its rule result and diagnostics."""
    rule = contract['rule']
    allow_infinity = contract.get('allow_infinity', False)
    if not isinstance(allow_infinity, bool):
        raise ValueError('allow_infinity must be a boolean')
    if actual.dtype.kind not in 'biuf' or reference.dtype.kind not in 'biuf':
        raise ValueError('Comparison requires real numeric tensors')
    actual_finite = np.isfinite(actual)
    reference_finite = np.isfinite(reference)
    matching_inf = np.isinf(actual) & np.isinf(reference) & (actual == reference)
    valid = (actual_finite & reference_finite) | (matching_inf & allow_infinity)
    differences = _differences(actual, reference, matching_inf & allow_infinity)
    maximum, index, actual_value, reference_value = _maximum(differences, actual, reference)
    result = {
        'status': 'fail',
        'rule': rule,
        'failed_elements': None,
        'max_absolute_error': maximum,
        'max_error_index': index,
        'actual_value': actual_value,
        'reference_value': reference_value,
        'reason': None,
    }
    kind = rule['kind']
    if kind in {'rms', 'relative_l2'}:
        result['elementwise_threshold'] = 'not established'
        result['reduction_semantics'] = (
            'Float32 NumPy reduction follows the original formula but may differ '
            'from Torch reduction rounding. Allowed matching infinities are replaced '
            'with zero in both arrays and remain included in the mean.'
        )
        result[kind] = None
        if not valid.all():
            result['reason'] = 'NaN or disallowed/mismatched infinity'
            return result
        if not actual.size:
            result['reason'] = 'Relative error is undefined for an empty output'
            return result
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            a = np.where(matching_inf, 0, actual).astype(np.float32)
            b = np.where(matching_inf, 0, reference).astype(np.float32)
            if kind == 'rms':
                numerator = np.mean(np.square(a - b), dtype=np.float32)
                denominator = np.maximum(
                    np.mean(np.square(b), dtype=np.float32), np.float32(rule['floor'])
                )
                error = np.sqrt(numerator / denominator)
            else:
                error = np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel())
        result[kind] = _number(error)
        if not np.isfinite(error) or not error < rule['limit']:
            result['reason'] = 'Relative error does not satisfy the original strict limit'
        else:
            result['status'] = 'pass'
        return result
    if kind == 'max_abs':
        if not actual.size:
            result['reason'] = 'Maximum error is undefined for an empty output'
            return result
        with np.errstate(over='ignore', invalid='ignore'):
            delta = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
        passed = valid & ((delta < rule['limit']) | (matching_inf & allow_infinity))
    elif kind == 'bits':
        if actual.dtype != reference.dtype:
            raise ValueError('Bit-exact comparison requires matching storage dtypes')
        shape = (*actual.shape, actual.dtype.itemsize)
        a = actual.reshape(-1).view(np.uint8).reshape(shape)
        b = reference.reshape(-1).view(np.uint8).reshape(shape)
        passed = valid & np.all(a == b, axis=-1)
    elif kind == 'exact':
        if actual.dtype.kind in 'biu' or reference.dtype.kind in 'biu':
            equal = actual.astype(object) == reference.astype(object)
        else:
            equal = actual == reference
        passed = valid & equal
    else:
        passed = _close(actual, reference, differences, valid, matching_inf, rule, allow_infinity)
    result['failed_elements'] = int(np.count_nonzero(~passed))
    if result['failed_elements']:
        flat = int(np.flatnonzero(~passed)[0])
        result['first_failure_index'] = [int(part) for part in np.unravel_index(flat, actual.shape)]
        result['reason'] = (
            'NaN or disallowed/mismatched infinity'
            if not valid.all()
            else 'Values do not satisfy the comparison rule'
        )
    else:
        result['status'] = 'pass'
    return result


def compare(case_path, reference_path, actual_path):
    """Validate persisted results and compare every output against its original rule.

    Return a JSON-safe report; absent native execution is explicitly reported as
    not_executed, while invalid or incomplete inputs produce a failure report.
    """
    report = {'status': 'fail', 'case_id': None, 'case_name': None, 'outputs': {}, 'reason': None}
    try:
        case = Case.load(case_path)
        report['case_id'] = case.data['id']
        report['case_name'] = case.data.get('name', case.data['id'])
        if reference_path is None:
            raise ValueError('A reference result is required')
        reference = read_result(case, reference_path, 'reference')
        if actual_path is None:
            report['status'] = 'not_executed'
            report['reason'] = 'Native result was not provided'
            return report
        actual = read_result(case, actual_path, 'native')
        reference_root = Path(reference_path).parent
        actual_root = Path(actual_path).parent
        for name, contract in case.data['outputs'].items():
            ref = load_tensor(reference_root, reference['outputs'][name])
            got = load_tensor(actual_root, actual['outputs'][name])
            if tuple(contract['shape']) != ref.shape or got.shape != ref.shape:
                raise ValueError(f'Output {name} has an incorrect shape')
            if (
                contract.get('layout') != 'contiguous'
                or not ref.flags.c_contiguous
                or not got.flags.c_contiguous
            ):
                raise ValueError(f'Output {name} has an incorrect layout')
            report['outputs'][name] = _evaluate(got, ref, contract)
        if not report['outputs']:
            raise ValueError('Case has no outputs')
        if all(output['status'] == 'pass' for output in report['outputs'].values()):
            report['status'] = 'pass'
        else:
            report['reason'] = 'One or more outputs failed comparison'
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        report['reason'] = str(error)
    return report
