static mfq_tensor_backend::Tensor dequant_nint_dense_f32(const NintWeight & w) {
    if (w.q8_zero) {
        return nint8_zero_dequant_f32_cuda(
            w.q_packed, w.q8_zero_scale, w.neuron_len).contiguous();
    }
    return nint_decode_f32_cuda(
        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
        w.neuron_len, w.gs).contiguous();
}

static mfq_tensor_backend::Tensor dequant_mxfp4_sq_cpu(
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
