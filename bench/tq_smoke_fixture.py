"""Small synthetic TQ MoE checkpoint for native integration checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tyloinfer.formats.header import FileHeader
from tyloinfer.formats.io import save
from tyloinfer.formats.tq import TQTensor, TQExpertTensor


def matrix(rows: int, width: int, rng: np.random.Generator, columns: bool) -> TQTensor:
    stored = (width + 255) // 256 * 256
    bits = rng.integers(1, 9, rows, dtype=np.uint64)
    words = bits * (stored // 32)
    desc = np.r_[np.uint64(0), np.cumsum(words[:-1])]
    desc |= (bits - 1) << np.uint64(32)
    desc |= np.uint64(np.float16(.01).view(np.uint16)) << np.uint64(40)
    scales = (rng.uniform(.5, 1.5, stored).astype(np.float16) if columns else
        rng.integers(0, 256, (rows * (stored // 16) + 1) // 2, dtype=np.uint8))
    return TQTensor(rows, stored, 16, columns, desc, scales,
        rng.integers(0, 256, int(words.sum()) * 4, dtype=np.uint8))


def generate(path: Path) -> None:
    rng = np.random.default_rng(2341)
    tensors = {}
    for role, rows, width in (('gate', 128, 256), ('up', 128, 256), ('down', 256, 128)):
        tensors[f'unit.mlp.experts.{role}.weight'] = TQExpertTensor(
            tuple(matrix(rows, width, rng, role != 'up') for _ in range(8)), width)
        tensors[f'unit.mlp.shared_expert.{role}.weight'] = rng.normal(0, .01, (rows, width)).astype(np.float16)
    tensors['unit.mlp.router.weight'] = rng.normal(0, .01, (8, 256)).astype(np.float16)
    tensors['unit.mlp.router.bias'] = np.zeros(8, dtype=np.float32)
    save(path, FileHeader(version=2, model_arch='synthetic-tq-moe'), tensors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    generate(args.output)
