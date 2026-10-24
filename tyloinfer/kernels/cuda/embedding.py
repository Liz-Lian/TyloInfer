"""Embedding lookup kernels."""

from __future__ import annotations

import torch

from tyloinfer.kernels.cuda._ext import ext


def embedding(weight: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather rows from ``weight [vocab,D]`` for int64 token ids."""

    return ext().embedding_lookup_cuda(
        weight.contiguous(),
        token_ids.contiguous().to(device=weight.device, dtype=torch.int64),
    )


def nint_embedding(g: dict, token_ids: torch.Tensor) -> torch.Tensor:
    """Gather and dequantize selected NINT embedding rows."""

    if g.get("row_q_bits") is None or g.get("row_q_bit_offsets") is None:
        raise ValueError(
            "NINT embedding requires canonical per-neuron row metadata"
        )
    return ext().nint_embedding_cuda(
        g["q_packed"],
        g["row_q_bits"],
        g["row_q_bit_offsets"],
        g["sub_scale"],
        g["sub_min"],
        g["neuron_scale"],
        g["neuron_min"],
        token_ids.contiguous().to(
            device=g["q_packed"].device, dtype=torch.int64
        ),
        int(g["neuron_len"]),
        int(g["gs"]),
    )
