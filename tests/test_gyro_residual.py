from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from tyloinfer import _gyro_fit
from tyloinfer.formats.gguf_residual import RowResidual
from tyloinfer.formats.gyro import FORMATS, codebook, decode_row
from tyloinfer.runtime.gguf_residual import OnlineResidual, fit_residual


def test_normative_codebook():
    assert hashlib.sha256(codebook().tobytes()).hexdigest() == (
        "ee481f151d8947ce8b08817fdf89d0a56d647b6435ceb3df0f079ea748e0e588"
    )


@pytest.mark.parametrize("name", FORMATS)
def test_ordinary_and_residual_rows(name):
    reference = np.random.default_rng(4).normal(size=(2, 128)).astype(np.float32)
    base = np.rint(reference * 2) / 2
    frozen = base.copy()
    residual = fit_residual(reference, base, [name, None], workers=2)
    correction = residual.dequantize()
    assert np.sum((reference[0] - base[0] - correction[0]) ** 2) < np.sum(
        (reference[0] - base[0]) ** 2
    )
    np.testing.assert_array_equal(correction[1], 0)
    np.testing.assert_array_equal(base, frozen)


def test_heterogeneous_neurons_tail_and_projection(tmp_path):
    reference = np.random.default_rng(12).normal(size=(4, 137)).astype(np.float32)
    base = np.rint(reference * 4) / 4
    choices = ["TQK4", "TQP7", None, "TQK7"]
    importance = np.linspace(0, 2, 137, dtype=np.float32)
    residual = fit_residual(reference, base, choices, importance=importance, workers=2)
    assert residual.formats == tuple(choices)
    for row in (0, 1, 3):
        one = fit_residual(
            reference[row : row + 1],
            base[row : row + 1],
            choices[row],
            importance=importance,
            workers=1,
        )
        assert residual.rows[row] == one.rows[0]
    path = tmp_path / "weights.residual"
    residual.save(path)
    restored = RowResidual.load(path)
    assert restored.to_bytes() == residual.to_bytes()
    assert restored.nbytes == path.stat().st_size
    input = np.random.default_rng(13).normal(size=(2, 3, 137)).astype(np.float32)
    np.testing.assert_allclose(
        restored.project(input), input @ restored.dequantize().T, atol=1e-5, rtol=1e-5
    )


def test_independent_rows_and_readers():
    value = np.random.default_rng(3).normal(size=(3, 128)).astype(np.float32)

    class Reader:
        shape = value.shape

        def __init__(self, data):
            self.data, self.requests = data, []

        def read_rows(self, begin, end):
            self.requests.append((begin, end))
            assert end - begin == 1
            return self.data[begin:end]

    base = Reader(np.rint(value * 2) / 2)
    reference = Reader(value)
    result = fit_residual(reference, base, ["TQK4", "TQP5", "TQK7"])
    assert sorted(base.requests) == [(0, 1), (1, 2), (2, 3)]
    assert len(reference.requests) == 3
    permutation = [2, 0, 1]
    permuted = fit_residual(
        value[permutation], base.data[permutation], [result.formats[i] for i in permutation]
    )
    assert permuted.rows == tuple(result.rows[i] for i in permutation)


def test_online_update_is_atomic_and_base_is_immutable():
    reference = np.random.default_rng(5).normal(size=(3, 128)).astype(np.float32)
    base = np.rint(reference * 3) / 3
    online = OnlineResidual(base)
    assert not online.base.flags.writeable
    online.update_rows([2], reference[2:], "TQP5")
    previous = online.snapshot
    assert previous.formats[:2] == (None, None)
    assert online.generation == 1
    with pytest.raises(ValueError):
        online.update_rows([0], np.full((1, 128), np.nan, dtype=np.float32))
    assert online.snapshot is previous
    assert online.generation == 1
    input = np.ones((2, 128), dtype=np.float32)
    np.testing.assert_allclose(
        online.project(input), input @ (base + previous.dequantize()).T, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_array_equal(base, online.base)


def test_zero_and_zero_importance_need_no_payload():
    x = np.ones((2, 129), dtype=np.float32)
    assert fit_residual(x, x).formats == (None, None)
    assert fit_residual(x, np.zeros_like(x), importance=np.zeros(129)).formats == (None, None)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b[:20],
        lambda b: b + b"x",
        lambda b: b[:8] + struct.pack("<I", 2) + b[12:],
        lambda b: b[:56] + struct.pack("<Q", 1) + b[64:],
        lambda b: b[:64] + struct.pack("<I", 1) + b[68:],
    ],
)
def test_invalid_payload_rejected(mutate):
    payload = RowResidual.empty(2, 129).to_bytes()
    with pytest.raises(ValueError):
        RowResidual.from_bytes(mutate(payload))


def test_validation_before_native_search():
    x = np.ones((2, 128), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_residual(x, x, ["TQK4"])
    with pytest.raises(ValueError):
        fit_residual(x, x, importance=-1)
    with pytest.raises(ValueError):
        _gyro_fit.fit_residual(x[0], x[0, :2], 145)
    with pytest.raises(ValueError):
        _gyro_fit.fit_residual(x[0], x[0], 0)
    with pytest.raises(ValueError):
        decode_row(b"", "TQK4", 128)


def test_inference_does_not_expose_ordinary_fitter():
    assert not hasattr(_gyro_fit, "encode_row")


def test_residual_native_requires_a_base():
    x = np.ones(128, dtype=np.float32)
    with pytest.raises(ValueError, match="base"):
        _gyro_fit.fit_residual(x, None, 145)


def test_residual_rejects_invalid_geometry_and_types():
    with pytest.raises(ValueError):
        RowResidual.empty(2, 1.5)
    x = np.ones((1, 128), dtype=np.float32)
    with pytest.raises(ValueError):
        fit_residual(x, x, formats=[{}])
