#pragma once

#include "engine/weights/sq.h"
#include "engine/backends/cuda/kernels/fp8_sq.h"

mfq_tensor_backend::Tensor sq_matmul_cpu(
    const Fp8SqWeight& weight,
    mfq_tensor_backend::Tensor input);

struct Fp8SqLinear {
    Fp8SqWeight weight;

    mfq_tensor_backend::Tensor forward(
            mfq_tensor_backend::Tensor x) const {
        if (!weight.blob.is_cuda()) return sq_matmul_cpu(weight, x);
        auto shape = x.sizes().vec();
        const auto original_dtype = x.scalar_type();
        auto source = x.reshape({-1, x.size(-1)});
        if (source.scalar_type() != mfq_tensor_backend::kFloat16 &&
                source.scalar_type() != mfq_tensor_backend::kFloat32) {
            source = source.to(mfq_tensor_backend::kFloat16).contiguous();
        } else {
            source = source.contiguous();
        }
        mfq_tensor_backend::Tensor output;
        if (weight.dtype == "MXFP8-SQ") {
            output = mxfp8_sq_matmul_cuda(
                weight.blob, weight.row_q,
                weight.row_symbol_byte_offsets, source,
                weight.out, weight.neuron_len,
                weight.block_rows, weight.block_columns,
                weight.scale_rows, weight.scale_columns,
                weight.palettes_offset, weight.symbols_offset,
                weight.scales_offset);
        } else if (weight.dtype == "FP8-128SQ") {
            output = fp8_128_sq_matmul_cuda(
                weight.blob, weight.row_q,
                weight.row_symbol_byte_offsets, source,
                weight.out, weight.neuron_len, weight.scale_kind,
                weight.palettes_offset, weight.symbols_offset,
                weight.scales_offset);
        } else {
            throw std::runtime_error("unsupported FP8-SQ linear dtype");
        }
        if (original_dtype == mfq_tensor_backend::kBFloat16) {
            output = output.to(original_dtype).contiguous();
        }
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }
};

mfq_tensor_backend::Tensor dequant_fp8_sq(
    const Fp8SqWeight& weight, bool fp32);
