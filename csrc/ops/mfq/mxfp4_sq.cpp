#include "mxfp4_sq.h"
#include "formats/mxfp4_sq_decode.h"

mfq_tensor_backend::Tensor dequant_mxfp4_sq_cpu(
        const Mxfp4SqWeight& weight) {
    auto output = mfq_tensor_backend::empty(
        {weight.out, weight.neuron_len},
        weight.blob.options().dtype(mfq_tensor_backend::kFloat32));
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::sq::parse(blob, weight.blob.numel());
    mfq_parallel_for(0, weight.out, 1, [&](int64_t begin, int64_t end) {
        for (auto row = begin; row < end; ++row) {
            mfq::sq::decode_cpu_row(
                blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                weight.row_auxiliary.data_ptr<int32_t>()[row],
                output.data_ptr<float>() + row * weight.neuron_len);
        }
    });
    return output;
}
