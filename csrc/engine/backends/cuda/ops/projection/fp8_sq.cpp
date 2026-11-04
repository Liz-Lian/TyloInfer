#include "fp8_sq.h"

#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/fp8_sq_blob.h"

namespace tyloi::cuda {

::tyloi::Tensor sq_matmul(const Fp8SqWeight& weight, ::tyloi::Tensor x) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.blob.is_cuda()) return sq_matmul_cpu(weight, x);
    auto shape = x.sizes().vec();
    const auto original_dtype = x.scalar_type();
    auto source = x.reshape({-1, x.size(-1)});
    if (source.scalar_type() != ::tyloi::kFloat16 &&
            source.scalar_type() != ::tyloi::kFloat32) {
        source = source.to(::tyloi::kFloat16).contiguous();
    } else {
        source = source.contiguous();
    }
    ::tyloi::Tensor output;
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
    if (original_dtype == ::tyloi::kBFloat16) {
        output = output.to(original_dtype).contiguous();
    }
    shape.back() = output.size(-1);
    return output.reshape(shape);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

} // namespace tyloi::cuda

::tyloi::Tensor dequant_fp8_sq(
        const Fp8SqWeight & weight,
        bool fp32) {
    if (!weight.blob.is_cuda()) {
        auto output = ::tyloi::cuda::empty(
            {weight.out, weight.neuron_len},
            weight.blob.options().dtype(::tyloi::kFloat32));
        const auto* blob = weight.blob.data_ptr<uint8_t>();
        const auto layout = tyloi::fp8sq::parse(
            weight.dtype, blob, weight.blob.numel());
        mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
            for (auto row = begin; row < end; ++row) {
                for (int64_t column = 0;
                        column < weight.neuron_len; ++column) {
                    output.data_ptr<float>()[row * weight.neuron_len + column] =
                        tyloi::fp8sq::decode_cpu(
                            blob, layout,
                            weight.row_q.data_ptr<uint8_t>()[row],
                            weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                            row, column);
                }
            }
        });
        return output.to(fp32
            ? ::tyloi::kFloat32
            : ::tyloi::kFloat16);
    }
    if (weight.dtype == "MXFP8-SQ") {
        return mxfp8_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.out, weight.neuron_len,
            weight.block_rows, weight.block_columns,
            weight.scale_rows, weight.scale_columns,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, fp32);
    }
    if (weight.dtype == "FP8-128SQ") {
        return fp8_128_sq_dequant_cuda(
            weight.blob, weight.row_q, weight.row_symbol_byte_offsets,
            weight.out, weight.neuron_len, weight.scale_kind,
            weight.palettes_offset, weight.symbols_offset,
            weight.scales_offset, fp32);
    }
    throw std::runtime_error("unsupported FP8-SQ dequant dtype");
}
