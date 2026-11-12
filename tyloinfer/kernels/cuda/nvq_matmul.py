"""Torch adapter for NVQ and NPQ runtime GPU buffers.

Adapted from TyloQuant mfq/kernels/cuda/nvq_matmul.py (0417c912).
"""

from __future__ import annotations

import torch

from tyloinfer.kernels.cuda._ext import ext


def _workspace(g: dict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the original quantized activation workspace."""
    m = int(x.shape[0])
    k_pad = int(g["ng"]) * int(g["gs"])
    key = (str(x.device), m, k_pad)
    cache = g.setdefault("_workspace", {})
    value = cache.get(key)
    if value is not None and value[0].device == x.device:
        return value
    qx = torch.empty((m, k_pad), device=x.device, dtype=torch.int8)
    xscale = torch.empty((m, int(g["ng"])), device=x.device, dtype=torch.float32)
    cache[key] = (qx, xscale)
    return qx, xscale


def _kernel_args(g: dict) -> tuple:
    """Return packed buffers in the kernel argument order."""
    return (
        g["indices_packed"],
        g["aux_packed"],
        g["sub_scale_packed"],
        g["neuron_scale"],
        g["codebook"],
    )


def nvq_dequantize(g: dict) -> torch.Tensor:
    """Decode one compact NVQ matrix to fp16 on the current CUDA stream."""

    return ext().nvq_dequant_cuda(
        *_kernel_args(g),
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
    )


def nvq_mmq(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run the gs24 int8 Tensor Core candidate for M=4..64."""

    x = x.contiguous().to(torch.float16)
    if x.dim() != 2 or not 4 <= x.shape[0] <= 64 or x.shape[1] != int(g["neuron_len"]):
        raise ValueError("NVQ MMQ input must be [M, neuron_len] with M in [4,64]")
    qx, xscale = _workspace(g, x)
    return ext().nvq_mmq_ws_cuda(
        *_kernel_args(g),
        x,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
        qx,
        xscale,
    )


def nvq_gemm_f16(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run online NVQ decode with FP16 Tensor Core GEMM for M=16..256."""

    x = x.contiguous().to(torch.float16)
    if (
        x.dim() != 2
        or not 16 <= x.shape[0] <= 256
        or x.shape[1] != int(g["neuron_len"])
    ):
        raise ValueError(
            "NVQ FP16 GEMM input must be [M, neuron_len] with M in [16,256]"
        )
    return ext().nvq_gemm_f16_cuda(
        *_kernel_args(g),
        x,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
    )


def _select_matmul_path(g: dict, m: int) -> str:
    """Choose the original geometry-dependent projection path."""
    storage_format = int(g.get("storage_format", g["format"]))
    out = int(g["out"])
    neuron_len = int(g["neuron_len"])
    e8_family = storage_format in {2, 5, 7, 8, 9, 13, 14}

    if storage_format == 8 and 14 <= m <= 16 and neuron_len >= 2 * out:
        return "dequant_gemm"

    if storage_format == 9 and m >= 13:
        wide_output = out * 8 >= neuron_len * 21
        wide_mmq = out >= 4096 and (out >= 2 * neuron_len or neuron_len >= 2 * out)
        if m <= 15:
            if wide_mmq:
                return "mmq"
            return "gemv" if out >= 1024 else "dequant_gemm"
        if m == 16:
            if out >= 4096:
                return "mmq"
            return "gemv" if out >= 1024 else "dequant_gemm"
        if m <= 31:
            if wide_output:
                return "online_f16"
            if out >= 4096 and out >= neuron_len:
                return "mmq"
            return "dequant_gemm"
        if m == 32:
            return "mmq" if out >= 4096 else "dequant_gemm"
        if m <= 47:
            return "online_f16" if wide_output else "dequant_gemm"
        if m == 48:
            return "mmq" if wide_output else "dequant_gemm"
        if m <= 63:
            return "online_f16" if wide_output else "dequant_gemm"
        if m == 64:
            if wide_output:
                return "online_f16"
            return "mmq" if out >= 8192 else "dequant_gemm"
        return "dequant_gemm"

    if m <= 13:
        return "gemv"
    if m == 14:
        return "gemv" if out >= 2048 else "dequant_gemm"
    if m == 15:
        if e8_family and out >= 8192:
            return "mmq"
        if e8_family and out >= 2048:
            return "gemv"
        return "dequant_gemm"
    if m == 16:
        if e8_family and out >= 6144:
            return "mmq"
        if e8_family and out >= 2048:
            return "gemv"
        if (
            storage_format in {3, 10, 11, 12, 15}
            and out >= 4096
            and neuron_len >= 8192
        ):
            return "mmq"
        return "dequant_gemm"

    wide_expansion = e8_family and out >= 3 * neuron_len
    if wide_expansion and (17 <= m <= 31 or 33 <= m <= 47):
        return "online_f16"
    if m == 32 and e8_family and out >= 8192:
        return "mmq"
    if m == 48 and e8_family and out >= 12288:
        return "mmq"
    if m == 64 and storage_format in {7, 9} and out >= 8192:
        return "mmq"
    return "dequant_gemm"


def _nvq_matmul_forward(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Compute ``x @ W.T`` from a compact NPQ/NVQ matrix."""

    original = tuple(x.shape)
    if not original:
        raise ValueError("NVQ matmul input must have at least one dimension")
    x2 = x.reshape(-1, original[-1]).contiguous().to(torch.float16)
    neuron_len = int(g["neuron_len"])
    if x2.shape[1] > neuron_len:
        raise ValueError(f"x last dim {x2.shape[1]} exceeds NVQ neuron_len {neuron_len}")
    if x2.shape[1] < neuron_len:
        x2 = torch.nn.functional.pad(x2, (0, neuron_len - x2.shape[1]))

    m = int(x2.shape[0])
    path = _select_matmul_path(g, m)
    if path == "gemv":
        qx, xscale = _workspace(g, x2)
        y = ext().nvq_gemv_ws_cuda(
            *_kernel_args(g),
            x2,
            neuron_len,
            int(g["gs"]),
            int(g["sub_bits"]),
            int(g["format"]),
            int(g["sign_mode"]),
            qx,
            xscale,
        )
    elif path == "mmq":
        y = nvq_mmq(g, x2)
    elif path == "online_f16":
        y = nvq_gemm_f16(g, x2)
    else:
        weight = nvq_dequantize(g)
        y = ext().nint_cublas_gemm_nt_f16acc_cuda(x2, weight)
    return y.reshape(*original[:-1], int(g["out"]))


def nvq_backward_input(g: dict, output_gradient: torch.Tensor) -> torch.Tensor:
    """Compute ``dX = dY @ W`` directly from packed NVQ/NPQ storage."""

    gradient = output_gradient.reshape(-1, output_gradient.shape[-1]).contiguous()
    return ext().nvq_backward_input_cuda(
        *_kernel_args(g),
        gradient,
        int(g["neuron_len"]),
        int(g["gs"]),
        int(g["sub_bits"]),
        int(g["format"]),
        int(g["sign_mode"]),
    )


class _NvqMatmulAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, g: dict) -> torch.Tensor:
        """Retain the packed weight and input metadata for the matching backward call."""
        ctx.g = g
        ctx.input_shape = tuple(x.shape)
        ctx.input_width = int(x.shape[-1])
        ctx.input_dtype = x.dtype
        return _nvq_matmul_forward(g, x)

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        """Trim padding and restore the input gradient's original shape and dtype."""
        gradient = nvq_backward_input(ctx.g, output_gradient)
        gradient = gradient[:, : ctx.input_width]
        return gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None


def nvq_matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Compute a packed NVQ/NPQ projection with a packed input gradient."""

    if not torch.is_grad_enabled() or not x.requires_grad:
        return _nvq_matmul_forward(g, x)
    return _NvqMatmulAutograd.apply(x, g)


__all__ = ['nvq_backward_input', 'nvq_matmul']
