#include "mxfp4_sq.h"
#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/mxfp4_sq_decode.h"

namespace tyloi::cuda {

::tyloi::Tensor sq_matmul(const Mxfp4SqWeight& weight, ::tyloi::Tensor x) {
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

::tyloi::Tensor dequant_mxfp4_sq_cpu(
        const Mxfp4SqWeight& weight) {
    auto output = ::tyloi::cuda::empty(
        {weight.out, weight.neuron_len},
        weight.blob.options().dtype(::tyloi::kFloat32));
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = tyloi::sq::parse(blob, weight.blob.numel());
    mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
        for (auto row = begin; row < end; ++row) {
            tyloi::sq::decode_cpu_row(
                blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                weight.row_auxiliary.data_ptr<int32_t>()[row],
                output.data_ptr<float>() + row * weight.neuron_len);
        }
    });
    return output;
}
