template <typename Decode>
static mfq_tensor_backend::Tensor sq_matmul_cpu_impl(
        mfq_tensor_backend::Tensor input,
        int64_t outputs,
        int64_t width,
        Decode decode) {
    TYLOINFER_RUNTIME_CHECK(
        !input.is_cuda() && input.size(-1) == width,
        "SQ CPU activation device or width mismatch");
    const auto dtype = input.scalar_type();
    auto shape = input.sizes().vec();
    input = input.to(mfq_tensor_backend::kFloat32)
        .reshape({-1, width}).contiguous();
    const auto rows = input.size(0);
    auto result = mfq_tensor_backend::empty({rows, outputs}, input.options());
    const auto* source = input.data_ptr<float>();
    auto* destination = result.data_ptr<float>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> weights(width);
        for (auto output = begin; output < end; ++output) {
            decode(output, weights.data());
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0;
                for (int64_t column = 0; column < width; ++column) {
                    sum = std::fma(
                        source[row * width + column], weights[column], sum);
                }
                destination[row * outputs + output] = sum;
            }
        }
    });
    shape.back() = outputs;
    return result.to(dtype).reshape(shape);
}

mfq_tensor_backend::Tensor sq_matmul_cpu(
        const Mxfp4SqWeight& weight,
        mfq_tensor_backend::Tensor input) {
    TYLOINFER_RUNTIME_CHECK(!weight.blob.is_cuda(), "SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::sq::parse(blob, weight.blob.numel());
    return sq_matmul_cpu_impl(
        std::move(input), weight.out, weight.neuron_len,
        [&](int64_t row, float* output) {
            mfq::sq::decode_cpu_row(
                blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                weight.row_auxiliary.data_ptr<int32_t>()[row], output);
        });
}

mfq_tensor_backend::Tensor sq_matmul_cpu(
        const Fp8SqWeight& weight,
        mfq_tensor_backend::Tensor input) {
    TYLOINFER_RUNTIME_CHECK(!weight.blob.is_cuda(), "SQ CPU weights must be on CPU");
    const auto* blob = weight.blob.data_ptr<uint8_t>();
    const auto layout = mfq::fp8sq::parse(
        weight.dtype, blob, weight.blob.numel());
    return sq_matmul_cpu_impl(
        std::move(input), weight.out, weight.neuron_len,
        [&](int64_t row, float* output) {
            for (int64_t column = 0; column < weight.neuron_len; ++column) {
                output[column] = mfq::fp8sq::decode_cpu(
                    blob, layout, weight.row_q.data_ptr<uint8_t>()[row],
                    weight.row_symbol_byte_offsets.data_ptr<int32_t>()[row],
                    row, column);
            }
        });
}

// Execution policy stays here; the format operators only receive a profiler.
mfq_tensor_backend::Tensor run_nint_linear(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        InputGateMode mode) {
    if (!x.is_cuda() || !w.q8_zero || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nint_matmul_input_mul(profiler, w, x, gate.value(), static_cast<int>(mode))
            : nint_matmul(profiler, w, x);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    if (gate.has_value()) {
        TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu, "NINT input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(mfq_tensor_backend::kFloat16);
        TYLOINFER_RUNTIME_CHECK(x.sizes() == g.sizes(), "NINT x and gate shapes must match");
        x = mode == InputGateMode::Sigmoid ? x * mfq_tensor_backend::sigmoid(g) : x * mfq_tensor_backend::silu(g);
    }
    x = pad_last(x, w.neuron_len);
    const int64_t rows = x.size(0);
    TYLOINFER_RUNTIME_CHECK(rows > 0, "KLD NINT8-0 MMQ requires activation rows");
    if (rows < 16) {
        x = mfq_tensor_backend::cat({x, mfq_tensor_backend::zeros(
            {16 - rows, x.size(1)}, x.options())}, 0).contiguous();
    }
    x = kl_mmq.prepare_activation(x);
    ++kl_mmq.dense_calls;
    auto result = profiler.measure("kld_mmq.nint8_zero.fp16", [&]() {
        return nint8_zero_mmq_f16_packed_cuda(
            w.q_packed, w.q8_zero_scale, x, w.neuron_len);
    });
    return rows < 16 ? result.narrow(0, 0, rows).contiguous() : result;
}

mfq_tensor_backend::Tensor run_nvq_linear(
        CudaProfiler& profiler,
        KlMmqState& kl_mmq,
        const NvqWeight& w,
        mfq_tensor_backend::Tensor x,
        MfqOptional<mfq_tensor_backend::Tensor> gate,
        InputGateMode mode) {
    if (!x.is_cuda() || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nvq_matmul_input_mul(profiler, w, x, gate.value(), static_cast<int>(mode))
            : nvq_matmul(profiler, w, x);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    if (gate.has_value()) {
        TYLOINFER_RUNTIME_CHECK(mode == InputGateMode::Sigmoid || mode == InputGateMode::Silu, "NVQ input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(mfq_tensor_backend::kFloat16);
        TYLOINFER_RUNTIME_CHECK(x.sizes() == g.sizes(), "NVQ x and gate shapes must match");
        x = mode == InputGateMode::Sigmoid ? x * mfq_tensor_backend::sigmoid(g) : x * mfq_tensor_backend::silu(g);
    }
    x = pad_last(x, w.neuron_len);
    TYLOINFER_RUNTIME_CHECK(x.size(0) >= 16, "KLD common NVQ MMQ requires at least 16 activation rows");
    x = kl_mmq.prepare_activation(x);
    ++kl_mmq.dense_calls;
    return profiler.measure("kld_mmq.nvq.fp16", [&]() {
        return nvq_gemm_f16_cuda(
            w.indices_packed, w.aux_packed, w.sub_scale_packed,
            w.neuron_scale, w.codebook, x, w.neuron_len, w.gs,
            w.sub_bits, w.kernel_format, w.sign_mode);
    });
}
