#pragma once

#include "engine/weights/sq.h"
#include "engine/backends/cuda/kernels/mxfp4_sq.h"

mfq_tensor_backend::Tensor sq_matmul_cpu(
    const Mxfp4SqWeight& weight,
    mfq_tensor_backend::Tensor input);

mfq_tensor_backend::Tensor dequant_mxfp4_sq_cpu(
    const Mxfp4SqWeight& weight);

struct Mxfp4SqLinear {
    Mxfp4SqWeight weight;

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
        auto output = mxfp4_sq_matmul_cuda(
            weight.blob,
            weight.row_q,
            weight.row_symbol_byte_offsets,
            weight.row_auxiliary,
            source,
            weight.bits,
            weight.out,
            weight.neuron_len,
            weight.matrix_scale_base,
            weight.q_sum,
            weight.sq4_rows);
        if (original_dtype == mfq_tensor_backend::kBFloat16) {
            output = output.to(original_dtype).contiguous();
        }
        shape.back() = output.size(-1);
        return output.reshape(shape);
    }
};
