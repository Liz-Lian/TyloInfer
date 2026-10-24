static NintWeight cat_weights(const std::vector<NintWeight> & ws) {
    if (ws.empty()) throw std::runtime_error("empty NINT group");
    const auto & a = ws[0];
    std::vector<mfq_tensor_backend::Tensor> qp, rqb, rqoff, q8s, ss, sm, ns, nm;
    int64_t out = 0;
    int64_t q_bit_base = 0;
    bool aligned_q8 = !a.q8_zero;
    int uniform_qbits=a.uniform_qbits;
    for (const auto & w : ws) {
        if(w.uniform_qbits!=uniform_qbits)uniform_qbits=0;
        if (w.ng != a.ng || w.gs != a.gs ||
            w.neuron_len != a.neuron_len || w.q8_zero != a.q8_zero) {
            throw std::runtime_error("cannot group NINT tensors with different input layout");
        }
        qp.push_back(w.q_packed);
        if (!w.q8_zero) {
            aligned_q8 = aligned_q8 && w.aligned_q8 && (q_bit_base & 31) == 0;
            rqb.push_back(w.row_q_bits);
            rqoff.push_back(w.row_q_bit_offsets + q_bit_base);
            q_bit_base += w.q_packed.numel() * 8;
        }
        if (w.q8_zero) {
            q8s.push_back(w.q8_zero_scale);
            out += w.out;
            continue;
        }
        ss.push_back(w.sub_scale);
        sm.push_back(w.sub_min);
        ns.push_back(w.neuron_scale);
        nm.push_back(w.neuron_min);
        out += w.out;
    }
    NintWeight g;
    g.out = out;
    g.ng = a.ng;
    g.gs = a.gs;
    g.bits = a.bits;
    g.neuron_len = a.neuron_len;
    g.q8_zero = a.q8_zero;
    g.aligned_q8 = aligned_q8;
    g.uniform_qbits=uniform_qbits;
    g.shape = a.shape;
    g.shape[0] = out;
    g.q_packed = mfq_tensor_backend::cat(qp, 0).contiguous();
    if (!a.q8_zero) {
        g.row_q_bits = mfq_tensor_backend::cat(rqb, 0).contiguous();
        g.row_q_bit_offsets = mfq_tensor_backend::cat(rqoff, 0).contiguous();
    }
    if (a.q8_zero) {
        g.q8_zero_scale = mfq_tensor_backend::cat(q8s, 0).contiguous();
        return g;
    }
    g.sub_scale = mfq_tensor_backend::cat(ss, 0).contiguous();
    g.sub_min = mfq_tensor_backend::cat(sm, 0).contiguous();
    g.neuron_scale = mfq_tensor_backend::cat(ns, 0).contiguous();
    g.neuron_min = mfq_tensor_backend::cat(nm, 0).contiguous();
    return g;
}

static NintLinearGroup make_linear_group(const std::vector<NintWeight> & ws) {
    NintLinearGroup g;
    for (const auto & w : ws) g.outs.push_back(w.out);
    bool same = !ws.empty();
    for (size_t i = 1; i < ws.size(); ++i) {
        if (ws[i].ng != ws[0].ng || ws[i].gs != ws[0].gs ||
            ws[i].neuron_len != ws[0].neuron_len ||
            ws[i].q8_zero != ws[0].q8_zero) {
            same = false;
            break;
        }
    }
    if (same) {
        g.w = cat_weights(ws);
        if (g.w.q8_zero) {
            g.projection_w.reserve(ws.size());
            int64_t offset = 0;
            for (const auto & source : ws) {
                NintWeight projection = g.w;
                projection.out = source.out;
                projection.shape = source.shape;
                projection.q_packed =
                    g.w.q_packed.narrow(0, offset, source.out);
                projection.q8_zero_scale =
                    g.w.q8_zero_scale.narrow(0, offset, source.out);
                projection.workspaces.clear();
                g.projection_w.push_back(std::move(projection));
                offset += source.out;
            }
            TYLOINFER_RUNTIME_CHECK(offset == g.w.out,
                        "Q8 projection views do not cover grouped output");
        }
    } else {
        std::vector<NintWeight> cur;
        std::vector<int64_t> cur_outs;
        auto flush = [&]() {
            if (cur.empty()) return;
            g.split_w.push_back(cur.size() == 1 ? cur[0] : cat_weights(cur));
            g.split_outs.push_back(cur_outs);
            cur.clear();
            cur_outs.clear();
        };
        for (const auto & w : ws) {
            bool append = !cur.empty() &&
                w.ng == cur[0].ng && w.gs == cur[0].gs &&
                w.neuron_len == cur[0].neuron_len &&
                w.q8_zero == cur[0].q8_zero;
            if (!append) flush();
            cur.push_back(w);
            cur_outs.push_back(w.out);
        }
        flush();
    }
    return g;
}

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
        int mode) {
    if (!x.is_cuda() || !w.q8_zero || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nint_matmul_input_mul(profiler, w, x, gate.value(), mode)
            : nint_matmul(profiler, w, x);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    if (gate.has_value()) {
        TYLOINFER_RUNTIME_CHECK(mode == 1 || mode == 2, "NINT input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(mfq_tensor_backend::kFloat16);
        TYLOINFER_RUNTIME_CHECK(x.sizes() == g.sizes(), "NINT x and gate shapes must match");
        x = mode == 1 ? x * mfq_tensor_backend::sigmoid(g) : x * mfq_tensor_backend::silu(g);
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
        int mode) {
    if (!x.is_cuda() || kl_mmq.mode == KlMmqMode::Default) {
        return gate.has_value()
            ? nvq_matmul_input_mul(profiler, w, x, gate.value(), mode)
            : nvq_matmul(profiler, w, x);
    }
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    if (gate.has_value()) {
        TYLOINFER_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ input gate mode must be sigmoid or SiLU");
        auto g = gate.value().contiguous().to(mfq_tensor_backend::kFloat16);
        TYLOINFER_RUNTIME_CHECK(x.sizes() == g.sizes(), "NVQ x and gate shapes must match");
        x = mode == 1 ? x * mfq_tensor_backend::sigmoid(g) : x * mfq_tensor_backend::silu(g);
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
