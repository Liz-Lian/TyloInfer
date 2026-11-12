"""Torch adapter for canonical packed NINT runtime buffers.

Adapted from TyloQuant mfq/kernels/cuda/nint_matmul.py (bee8015e).
"""

from __future__ import annotations

import torch

from tyloinfer.kernels.cuda._ext import ext


def _workspace(weight: dict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse quantized activation buffers for this device and geometry."""
    rows = int(x.shape[0])
    padded_width = int(weight['ng']) * int(weight['gs'])
    groups = int(weight['ng'])
    key = (str(x.device), rows, padded_width)
    workspaces = weight.setdefault('_workspace', {})
    cached = workspaces.get(key)
    if cached is not None and all(value.device == x.device for value in cached):
        return cached
    cached = (
        torch.empty((rows, padded_width), device=x.device, dtype=torch.int8),
        torch.empty((rows, groups), device=x.device, dtype=torch.float32),
    )
    workspaces[key] = cached
    return cached


def _prepare(weight: dict, x: torch.Tensor) -> torch.Tensor:
    """Flatten, convert, and pad activations as required by the packed weight."""
    required = ('row_q_bits', 'row_q_bit_offsets')
    if any(weight.get(name) is None for name in required):
        raise ValueError('NINT runtime input has incomplete row metadata')
    source = x.reshape(-1, x.shape[-1]).contiguous().to(torch.float16)
    neuron_len = int(weight['neuron_len'])
    if source.shape[1] > neuron_len:
        raise ValueError(f'x last dim {source.shape[1]} exceeds NINT width {neuron_len}')
    if source.shape[1] < neuron_len:
        source = torch.nn.functional.pad(source, (0, neuron_len - int(source.shape[1])))
    return source


def _nint_matmul_forward(weight: dict, x: torch.Tensor) -> torch.Tensor:
    """Dispatch the original small-row packed or decoded matmul path."""
    source = _prepare(weight, x)
    rows = int(source.shape[0])
    if rows <= 8:
        qx, xscale = _workspace(weight, source)
        return ext().nint_matmul_ws_cuda(
            weight['q_packed'], weight['row_q_bits'], weight['row_q_bit_offsets'],
            weight['sub_scale'], weight['sub_min'], weight['neuron_scale'],
            weight['neuron_min'], source, int(weight['gs']), qx, xscale,
        )
    dense = ext().nint_decode_cuda(
        weight['q_packed'], weight['row_q_bits'], weight['row_q_bit_offsets'],
        weight['sub_scale'], weight['sub_min'], weight['neuron_scale'],
        weight['neuron_min'], int(weight['neuron_len']), int(weight['gs']),
    )
    return source @ dense.T


def nint_backward_input(weight: dict, output_gradient: torch.Tensor) -> torch.Tensor:
    """Compute the input gradient using packed NINT buffers and the original dispatch."""
    gradient = output_gradient.reshape(-1, output_gradient.shape[-1]).contiguous().to(torch.float16)
    if int(gradient.shape[0]) <= 8:
        return ext().nint_backward_input_cuda(
            weight['q_packed'], weight['row_q_bits'], weight['row_q_bit_offsets'],
            weight['sub_scale'], weight['sub_min'], weight['neuron_scale'],
            weight['neuron_min'], gradient, int(weight['neuron_len']), int(weight['gs']),
        )
    dense = ext().nint_decode_cuda(
        weight['q_packed'], weight['row_q_bits'], weight['row_q_bit_offsets'],
        weight['sub_scale'], weight['sub_min'], weight['neuron_scale'],
        weight['neuron_min'], int(weight['neuron_len']), int(weight['gs']),
    )
    return gradient @ dense


class _NintMatmulAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: dict) -> torch.Tensor:
        """Retain the weight and input metadata for the matching backward call."""
        ctx.weight = weight
        ctx.input_width = int(x.shape[-1])
        ctx.input_dtype = x.dtype
        return _nint_matmul_forward(weight, x)

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        """Trim input padding and restore the original gradient dtype."""
        gradient = nint_backward_input(ctx.weight, output_gradient)
        return gradient[:, : ctx.input_width].to(ctx.input_dtype), None


def nint_matmul(weight: dict, x: torch.Tensor) -> torch.Tensor:
    """Apply canonical GPU NINT buffers, with an input gradient when requested."""
    if not torch.is_grad_enabled() or not x.requires_grad:
        return _nint_matmul_forward(weight, x)
    return _NintMatmulAutograd.apply(x, weight)


__all__ = ['nint_backward_input', 'nint_matmul']
