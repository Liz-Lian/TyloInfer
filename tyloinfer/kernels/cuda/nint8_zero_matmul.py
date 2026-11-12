"""Torch adapter for packed NINT8-zero runtime buffers.

Adapted from TyloQuant mfq/kernels/cuda/nint8_zero_matmul.py (c4adacd7).
"""

from __future__ import annotations

import torch

from tyloinfer.kernels.cuda._ext import ext


def _nint8_zero_matmul_forward(weight: dict[str, object], x: torch.Tensor) -> torch.Tensor:
    """Dispatch the original small-row GEMV or packed MMQ projection."""
    q = weight['q']
    if not isinstance(q, torch.Tensor):
        raise TypeError('NINT8-0 q must be a torch tensor')
    x = torch.as_tensor(x, device=q.device, dtype=torch.float16)
    width = int(weight['neuron_len'])
    if x.shape[-1] != width:
        raise ValueError(f'NINT8-0 input width {x.shape[-1]} does not match {width}')
    original = x.shape
    flat = x.reshape(-1, width).contiguous()
    rows = int(flat.shape[0])
    ng = int(weight['ng'])
    if rows <= 8:
        qx = torch.empty((rows, ng * 32), device=q.device, dtype=torch.int8)
        xscale = torch.empty((rows, ng), device=q.device, dtype=torch.float32)
        output = ext().nint8_zero_gemv_ws_cuda(weight['q'], weight['scale'], flat, qx, xscale)
    else:
        output = ext().nint8_zero_mmq_f16_packed_cuda(weight['q'], weight['scale'], flat, width)
    return output.reshape(*original[:-1], int(weight['out']))


def nint8_zero_backward_input(
    weight: dict[str, object], output_gradient: torch.Tensor,
) -> torch.Tensor:
    """Compute the input gradient directly from packed NINT8-zero buffers."""
    gradient = output_gradient.reshape(-1, output_gradient.shape[-1]).contiguous()
    return ext().nint8_zero_backward_input_cuda(
        weight['q'], weight['scale'], gradient, int(weight['neuron_len']),
    )


class _Nint8ZeroMatmulAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: dict[str, object]) -> torch.Tensor:
        """Retain the weight and input metadata for the matching backward call."""
        ctx.weight = weight
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        return _nint8_zero_matmul_forward(weight, x)

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        """Restore the input gradient's original shape and dtype."""
        gradient = nint8_zero_backward_input(ctx.weight, output_gradient)
        return gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None


def nint8_zero_matmul(weight: dict[str, object], x: torch.Tensor) -> torch.Tensor:
    """Apply GPU NINT8-zero buffers, with an input gradient when requested."""
    if not torch.is_grad_enabled() or not x.requires_grad:
        return _nint8_zero_matmul_forward(weight, x)
    return _Nint8ZeroMatmulAutograd.apply(x, weight)


__all__ = ['nint8_zero_backward_input', 'nint8_zero_matmul']
