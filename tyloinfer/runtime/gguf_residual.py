"""Per-output-neuron residual fitting; never rewrites or refits the base."""

from __future__ import annotations

from threading import Lock

import numpy as np

from tyloinfer._gyro.parallel import map_rows
from tyloinfer.formats.gguf_residual import RowResidual
from tyloinfer.formats.gyro import FORMATS, decode_row, format_name


def _row(source, index: int, columns: int) -> np.ndarray:
    value = (
        source.read_rows(index, index + 1)[0]
        if hasattr(source, "read_rows")
        else np.asarray(source)[index]
    )
    value = np.ascontiguousarray(value, dtype=np.float32)
    if value.shape != (columns,) or not np.isfinite(value).all():
        raise ValueError("expected a finite weight row of the declared width")
    return value


def fit_residual(reference, base, formats="TQK6", *, importance=None, workers=None) -> RowResidual:
    """Fit each neuron independently, reading at most one source row per worker.

    ``reference`` and ``base`` accept arrays or row readers exposing ``shape``
    and ``read_rows(begin,end)``. Format choices are a scalar or one per row;
    None disables a row. Corrections that fail to lower weighted SSE are omitted.
    """
    from tyloinfer import _gyro_fit

    shape = tuple(base.shape)
    if len(shape) != 2 or min(shape) < 1 or tuple(reference.shape) != shape:
        raise ValueError("reference and base must have the same nonempty matrix shape")
    choices = (
        [formats] * shape[0]
        if isinstance(formats, (str, int)) or formats is None
        else list(formats)
    )
    if len(choices) != shape[0]:
        raise ValueError("one residual format is required per output neuron")
    choices = [None if item is None else format_name(item) for item in choices]
    im = (
        None
        if importance is None
        else np.broadcast_to(np.asarray(importance, dtype=np.float32), shape)
    )
    if im is not None and (not np.isfinite(im).all() or (im < 0).any()):
        raise ValueError("importance must be finite and nonnegative")

    def fit(index):
        name = choices[index]
        if name is None:
            return None, b""
        target, original = _row(reference, index, shape[1]), _row(base, index, shape[1])
        error = target.astype(np.float64) - original
        weights = 1.0 if im is None else im[index].astype(np.float64)
        before = np.sum(error * error * weights)
        if before == 0:
            return None, b""
        packed = _gyro_fit.fit_residual(
            target,
            original,
            FORMATS[name][0],
            None if im is None else np.ascontiguousarray(im[index]),
        )
        corrected = original.astype(np.float64) + decode_row(packed, name, shape[1])
        after = np.sum((target - corrected) ** 2 * weights)
        return (name, packed) if after < before else (None, b"")

    fitted = list(map_rows(fit, shape[0], workers))
    return RowResidual(
        shape[1], tuple(item[0] for item in fitted), tuple(item[1] for item in fitted)
    )


class OnlineResidual:
    """Atomically publish fitted row corrections while retaining the base.

    Array bases are copied once and made immutable. Source readers must expose
    immutable checkpoint rows. A failed fit leaves the previous snapshot intact.
    """

    def __init__(self, base, residual: RowResidual | None = None):
        if hasattr(base, "read_rows"):
            self.base = base
        else:
            self.base = np.array(base, dtype=np.float32, copy=True)
            self.base.flags.writeable = False
        if len(self.base.shape) != 2 or min(self.base.shape) < 1:
            raise ValueError("expected a nonempty base matrix")
        self._snapshot = residual if residual is not None else RowResidual.empty(*self.base.shape)
        if self._snapshot.shape != tuple(self.base.shape):
            raise ValueError("initial residual geometry differs")
        self._lock = Lock()
        self.generation = 0

    @property
    def snapshot(self):
        with self._lock:
            return self._snapshot

    def fit(self, reference, formats="TQK6", *, importance=None, workers=None):
        fitted = fit_residual(reference, self.base, formats, importance=importance, workers=workers)
        with self._lock:
            self._snapshot = fitted
            self.generation += 1
        return fitted

    def update_rows(self, row_ids, reference, formats="TQK6", *, importance=None, workers=None):
        ids = np.asarray(row_ids)
        if (
            ids.ndim != 1
            or ids.dtype.kind not in "iu"
            or not len(ids)
            or len(set(map(int, ids))) != len(ids)
        ):
            raise ValueError("row IDs must be unique integers")
        if np.any(ids < 0) or np.any(ids >= self.base.shape[0]):
            raise IndexError("residual row is out of bounds")
        original = np.stack([_row(self.base, int(index), self.base.shape[1]) for index in ids])
        fitted = fit_residual(reference, original, formats, importance=importance, workers=workers)
        with self._lock:
            self._snapshot = self._snapshot.replace(ids, fitted)
            self.generation += 1
            return self._snapshot

    def project(self, input):
        input = np.asarray(input, dtype=np.float32)
        snapshot = self.snapshot
        if hasattr(self.base, "read_rows"):
            output = np.stack(
                [input @ _row(self.base, i, self.base.shape[1]) for i in range(self.base.shape[0])],
                axis=-1,
            )
        else:
            output = input @ self.base.T
        return output + snapshot.project(input)
