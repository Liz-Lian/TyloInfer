"""Torch NEPQ runtime-buffer adapter with the original packed backward dispatch.

Adapted from TyloQuant eea6bf9867494ff8f3384bb4932ee2b2b88bdadc:
``mfq/kernels/cuda/nepq_matmul.py`` SHA256 4c72d4b885e247ff66559585aee8314e3ed659f838506149fc5ae2742f7e0e0a;
residual transpose from ``mfq/formats/nepq.py`` SHA256 f22c12da84437cafe856e1df7093537431ce65d5d9d829b038f53e3b4d697354.
The input dictionary contains already constructed runtime buffers; its workspace
and residual transpose entries are populated lazily by this adapter.
"""

from __future__ import annotations

import numpy as np
import torch

from tyloinfer.kernels.cuda._ext import ext


def _args(g: dict) -> tuple:
    """Return the packed base projection buffers in kernel order."""
    return (g['indices_packed'], g['aux_packed'], g['state_packed'],
            g['neuron_scale'], g['table_pool'], g['bank_ids'])


def _transpose(first, second, vectors, block_vectors, position_bits):
    """Index the source residual records by input vector in stable order."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 2 or second.shape != first.shape:
        raise ValueError('NEPQ residual streams must be matching rank-2 arrays')
    vectors, block_vectors, position_bits = int(vectors), int(block_vectors), int(position_bits)
    if vectors < 0 or block_vectors <= 0 or position_bits <= 0:
        raise ValueError('invalid NEPQ residual transpose geometry')
    blocks = int(first.shape[1])
    vector_parts, row_parts, dictionary_parts = [], [], []
    position_mask = (1 << position_bits) - 1
    for records in (first, second):
        flat = np.asarray(records, dtype=np.int32).reshape(-1)
        locations = np.flatnonzero(flat >= 0)
        if not locations.size:
            continue
        selected = flat[locations]
        rows = locations // blocks
        block_ids = locations - rows * blocks
        vector_ids = block_ids * block_vectors + (selected & position_mask)
        dictionary_ids = selected >> position_bits
        valid = (vector_ids < vectors) & (dictionary_ids < 1024)
        vector_parts.append(vector_ids[valid])
        row_parts.append(rows[valid])
        dictionary_parts.append(dictionary_ids[valid])
    if not vector_parts:
        return (np.zeros(vectors + 1, dtype=np.uint32), np.zeros(1, dtype=np.uint32),
                np.zeros(1, dtype=np.uint16))
    vector_ids = np.concatenate(vector_parts)
    rows = np.concatenate(row_parts)
    dictionary_ids = np.concatenate(dictionary_parts)
    order = np.argsort(vector_ids, kind='stable')
    vector_ids = vector_ids[order]
    counts = np.bincount(vector_ids, minlength=vectors)
    offsets = np.empty(vectors + 1, dtype=np.uint32)
    offsets[0] = 0
    np.cumsum(counts, dtype=np.uint32, out=offsets[1:])
    return (offsets, np.ascontiguousarray(rows[order], dtype=np.uint32),
            np.ascontiguousarray(dictionary_ids[order], dtype=np.uint16))


def _residual_transpose(g: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build and cache the original host-transposed residual index on the device."""
    cached = g.get('_residual_transpose')
    if cached is not None:
        return cached
    first = g['residual_first'].detach().cpu().numpy()
    second = g['residual_second'].detach().cpu().numpy()
    offsets, rows, dictionary = _transpose(first, second, int(g['neuron_len']) // 8,
                                            int(g['residual_block_vectors']),
                                            int(g['residual_position_bits']))
    cached = (
        torch.as_tensor(offsets.astype(np.int32, copy=False), device=g['device'], dtype=torch.int32).contiguous(),
        torch.as_tensor(rows.astype(np.int32, copy=False), device=g['device'], dtype=torch.int32).contiguous(),
        torch.as_tensor(dictionary.astype(np.int16, copy=False), device=g['device'], dtype=torch.int16).contiguous(),
    )
    g['_residual_transpose'] = cached
    return cached


def _workspace(g: dict, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reuse the original Q8 workspace for this device and batch geometry."""
    m = int(x.shape[0])
    padded = int(g['ng']) * int(g['gs'])
    key = (str(x.device), m, padded)
    cache = g.setdefault('_workspace', {})
    value = cache.get(key)
    if value is not None and value[0].device == x.device:
        return value
    value = (torch.empty((m, padded), device=x.device, dtype=torch.int8),
             torch.empty((m, int(g['ng'])), device=x.device, dtype=torch.float32))
    cache[key] = value
    return value


def _prepare(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Convert and rotate activations in the original order."""
    value = x.contiguous().to(device=g['device'], dtype=torch.float16)
    if value.dim() != 2 or value.shape[1] != int(g['neuron_len']):
        raise ValueError('NEPQ input must be rank-2 [M,K]')
    block = int(g['rotation_block'])
    if block:
        value = ext().nepq_hadamard_input_cuda(value, g['rotation_signs'], block)
    return value


def _dequant(g: dict) -> torch.Tensor:
    """Decode the base and optional sparse residual to stored-coordinate FP16."""
    value = ext().nepq_dequant_cuda(*_args(g), int(g['neuron_len']),
                                   int(g['sub_bits']), int(g['format']))
    if 'residual_codebook' in g:
        value = ext().nepq_sparse_residual_dequant_cuda(
            g['residual_codebook'], g['residual_first'], g['residual_second'],
            int(g['residual_position_bits']), int(g['residual_block_vectors']),
            value.reshape(int(g['rows']), int(g['neuron_len'])),
        )
    return value.reshape(g['shape'])


def _forward(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run the source row-count dispatch and optional residual projection."""
    original = tuple(x.shape)
    if not original:
        raise ValueError('NEPQ matmul input must have at least one dimension')
    x2 = x.reshape(-1, original[-1])
    m = int(x2.shape[0])
    value = _prepare(g, x2)
    if m <= 16:
        if not 1 <= m <= 16:
            raise ValueError('NEPQ GEMV requires M in [1,16]')
        qx, xscale = _workspace(g, value)
        output = ext().nepq_gemv_ws_cuda(*_args(g), value, int(g['neuron_len']),
                                         int(g['sub_bits']), int(g['format']), qx, xscale)
    elif m <= 64:
        qx, xscale = _workspace(g, value)
        output = ext().nepq_mmq_ws_cuda(*_args(g), value, int(g['neuron_len']),
                                        int(g['sub_bits']), int(g['format']), qx, xscale)
    elif m <= 256:
        output = ext().nepq_gemm_f16_cuda(*_args(g), value, int(g['neuron_len']),
                                          int(g['sub_bits']), int(g['format']))
    else:
        weight = _dequant(g).reshape(int(g['rows']), -1)
        output = ext().nint_cublas_gemm_nt_f16acc_cuda(value, weight)
        return output.reshape(*original[:-1], int(g['n_experts']), int(g['out_per_expert']))
    if 'residual_codebook' in g:
        output = ext().nepq_sparse_residual_matmul_cuda(
            g['residual_codebook'], g['residual_first'], g['residual_second'], value,
            int(g['residual_position_bits']), int(g['residual_block_vectors']), output,
        )
    return output.reshape(*original[:-1], int(g['n_experts']), int(g['out_per_expert']))


def nepq_backward_input(g: dict, output_gradient: torch.Tensor) -> torch.Tensor:
    """Compute the input gradient from runtime buffers using the source dispatch."""
    gradient = output_gradient.reshape(-1, int(g['rows'])).contiguous().to(torch.float16)
    residual = 'residual_codebook' in g
    split = residual and gradient.shape[0] <= 8
    fused = split and int(g['rotation_block']) == 8
    if residual and not split:
        weight = _dequant(g).reshape(int(g['rows']), int(g['neuron_len']))
        result = torch.mm(gradient, weight)
    else:
        result = ext().nepq_backward_input_cuda(*_args(g), gradient, int(g['neuron_len']),
                                                 int(g['sub_bits']), int(g['format']))
        if split:
            offsets, rows, dictionary = _residual_transpose(g)
            result = ext().nepq_sparse_residual_backward_transpose_cuda(
                g['residual_codebook'], offsets, rows, dictionary, gradient,
                g['rotation_signs'], fused, result,
            )
    block = int(g['rotation_block'])
    if block and not fused:
        result = ext().nepq_hadamard_adjoint_cuda(result, g['rotation_signs'], block)
    return result


class _NepqMatmulAutograd(torch.autograd.Function):
    """Connect the original packed input backward to the real forward path."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, g: dict) -> torch.Tensor:
        """Record input geometry and run the packed forward dispatch."""
        ctx.g = g
        ctx.input_shape = tuple(x.shape)
        ctx.input_dtype = x.dtype
        return _forward(g, x)

    @staticmethod
    def backward(ctx, output_gradient: torch.Tensor):
        """Apply the packed input gradient and restore the original input shape."""
        gradient = nepq_backward_input(ctx.g, output_gradient)
        return gradient.reshape(ctx.input_shape).to(ctx.input_dtype), None


def nepq_matmul(g: dict, x: torch.Tensor) -> torch.Tensor:
    """Run NEPQ expert projections, attaching packed input autograd when needed."""
    if not torch.is_grad_enabled() or not x.requires_grad:
        return _forward(g, x)
    return _NepqMatmulAutograd.apply(x, g)
