"""Bound the number of queued row fits independently of model size."""

import os
from concurrent.futures import ThreadPoolExecutor


def map_rows(function, count, workers=None):
    if workers is not None and (not isinstance(workers, int) or workers < 1):
        raise ValueError("workers must be a positive integer")
    window = max(1, workers or os.cpu_count() or 1) * 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, count, window):
            yield from pool.map(function, range(start, min(count, start + window)))
