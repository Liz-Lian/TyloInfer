std::vector<mfq_tensor_backend::Tensor> NintLinearGroup::forward(
            CudaProfiler& profiler,
            KlMmqState& kl_mmq,
            mfq::cuda::Branches& branches,
            const CudaExecutionConfig& config,
            bool serial_branches,
            mfq_tensor_backend::Tensor x) const {
    auto shape = x.sizes().vec();
    std::vector<mfq_tensor_backend::Tensor> parts;
    auto xf = x.reshape({-1, x.size(-1)});
    if (w.q8_zero && xf.size(0) > 64 &&
            projection_w.size() == outs.size()) {
        parts.reserve(projection_w.size());
        for (const auto & projection : projection_w) {
            parts.push_back(run_nint_linear(
                profiler, kl_mmq, projection, xf));
        }
    } else if (!split_w.empty()) {
        parts.reserve(outs.size());
        std::vector<int> shared_groups;
        const auto quant_group_setting=mfq::cuda::runtime_options::nint_group_quantize();
        if((!quant_group_setting || *quant_group_setting) && serial_branches &&
                kl_mmq.mode==KlMmqMode::Default && xf.is_contiguous() &&
                xf.size(0)>=1 && xf.size(0)<=6 &&
                (xf.scalar_type()==mfq_tensor_backend::kFloat16 || xf.scalar_type()==mfq_tensor_backend::kFloat32)) {
            bool compatible=true;
            for(const auto& weight:split_w) {
                compatible=compatible && !weight.q8_zero && weight.neuron_len==xf.size(1) &&
                    (weight.gs==24 || weight.gs==28 || weight.gs==48);
                if(std::find(shared_groups.begin(),shared_groups.end(),weight.gs)==shared_groups.end())
                    shared_groups.push_back(weight.gs);
            }
            if(!compatible || shared_groups.size()<2)shared_groups.clear();
        }
        NintInputQuantizationScope prepared(xf,shared_groups);
        std::vector<mfq_tensor_backend::Tensor> grouped_outputs;
        const bool parallel =
            decode_branch_parallel_enabled(
                config, serial_branches, xf.size(0)) &&
            branches.run(
                split_w.size(),
                [&](size_t index) {
                    return run_nint_linear(
                        profiler, kl_mmq, split_w[index], xf);
                },
                grouped_outputs);
        for (size_t i = 0; i < split_w.size(); ++i) {
            auto y = parallel
                ? grouped_outputs[i]
                : run_nint_linear(
                      profiler, kl_mmq, split_w[i], xf);
            auto ys = y.split_with_sizes(split_outs[i], -1);
            for (auto & p : ys) parts.push_back(p);
        }
    } else {
        auto y = run_nint_linear(
            profiler, kl_mmq,
            w, x.reshape({-1, x.size(-1)}));
        parts = y.split_with_sizes(outs, -1);
    }
    for (auto & p : parts) {
        auto s = shape;
        s.back() = p.size(-1);
        p = p.reshape(s);
    }
    return parts;
}

mfq_tensor_backend::Tensor NintLinearGroup::forward_swiglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
    if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("NINT SwiGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_swiglu(
        profiler, w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}

mfq_tensor_backend::Tensor NintLinearGroup::forward_geglu(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
    if (!split_w.empty() || outs.size() != 2 || outs[0] != outs[1]) {
        throw std::runtime_error("NINT GeGLU fusion requires one packed [gate, up] group");
    }
    auto shape = x.sizes().vec();
    auto y = nint_matmul_geglu(
        profiler, w, x.reshape({-1, x.size(-1)}));
    shape.back() = y.size(-1);
    return y.reshape(shape);
}
