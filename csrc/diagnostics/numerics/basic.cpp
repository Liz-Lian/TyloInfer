#include "basic.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/native/env.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Native RoPE adapter entry points are implemented by ops/rope/wrappers.cpp.
::tyloi::DenseTensor rope_cuda(::tyloi::DenseTensor, ::tyloi::DenseTensor, double);
::tyloi::DenseTensor rope_ext_cuda(::tyloi::DenseTensor, ::tyloi::DenseTensor, double, std::int64_t,
                                   ::tyloi::DenseTensor);

namespace tyloi::diagnostics::numerics {
namespace {
/** Execute the original RoPE wrapper's table or direct selection. */
tyloi::DenseTensor rope(TensorWork& work, const Json& config, Json& details) {
    auto x = work.input("x");
    auto pos = work.input("pos");
    const auto width = x.size(-1);
    const auto rotary =
        config.at("rotary_dim").is_null() ? width : config.at("rotary_dim").get<std::int64_t>();
    const auto base = config.at("base").get<double>();
    const auto cpu = tyloi::TensorOptions().dtype(tyloi::kInt64).device(tyloi::Device(tyloi::kCPU));
    tyloi::DenseTensor sections;
    if (config.at("sections").is_null())
        sections = work.keep(tyloi::cuda::empty({0}, cpu));
    else {
        const auto values = config.at("sections").get<std::vector<std::int64_t>>();
        sections = work.keep(tyloi::cuda::tensor(values, cpu));
    }
    const auto* setting = tyloi::cuda::env_lookup("TYLOINFER_ROPE_TABLE");
    const bool table = !config.at("table_len").is_null() || !setting || std::string(setting) != "0";
    details["rope_path"] = table ? "table" : "direct";
    if (!table) {
        auto converted = work.keep(pos.to(tyloi::kFloat32));
        auto positions = work.keep(converted.contiguous());
        if (config.at("rotary_dim").is_null() && config.at("sections").is_null())
            return work.keep(rope_cuda(x, positions, base));
        return work.keep(rope_ext_cuda(x, positions, base, rotary, sections));
    }
    const auto length =
        config.at("table_len").is_null()
            ? std::max<std::int64_t>(16, static_cast<std::int64_t>(pos.max().item<double>()) + 1)
            : config.at("table_len").get<std::int64_t>();
    details["rope_table_len"] = length;
    auto converted = work.keep(pos.to(tyloi::kInt64));
    auto positions = work.keep(converted.contiguous());
    const auto options = tyloi::TensorOptions().dtype(tyloi::kFloat32).device(work.device());
    auto steps = work.keep(tyloi::cuda::arange(0, rotary, 2, options));
    auto negative = work.keep(-steps);
    auto exponent = work.keep(negative / double(rotary));
    auto bases = work.keep(tyloi::cuda::full({rotary / 2}, base, options));
    auto freq = work.keep(tyloi::cuda::pow(bases, exponent));
    auto times = work.keep(tyloi::cuda::arange(length, options));
    auto angles = work.keep(times.unsqueeze(1) * freq.unsqueeze(0));
    auto cosine_values = work.keep(tyloi::cuda::cos(angles));
    auto sine_values = work.keep(tyloi::cuda::sin(angles));
    auto cosine = work.keep(cosine_values.contiguous());
    auto sine = work.keep(sine_values.contiguous());
    return work.keep(rope_table_cuda(x, positions, cosine, sine, rotary, sections));
}
} // namespace

bool run_basic(Case& value) {
    const auto& data = value.data();
    const auto op = data.at("op").get<std::string>();
    if (op != "rms_norm" && op != "l2_norm" && op != "acc" && op != "silu_mul" && op != "rope" &&
        op != "ssm_conv_silu" && op != "gelu_mul")
        return false;
    TensorWork work(value);
    const auto& config = data.at("config");
    Json details = Json::object();
    tyloi::DenseTensor output;
    if (op == "rms_norm") {
        auto x = work.input("x");
        auto weight = work.input("w");
        output = work.keep(rms_norm_cuda(x, weight, config.at("eps").get<double>()));
    } else if (op == "l2_norm") {
        output = work.keep(l2_norm_cuda(work.input("x"), config.at("eps").get<double>()));
    } else if (op == "acc") {
        auto a = work.input("a");
        auto b = work.input("b");
        output = work.keep(acc_cuda(a, b));
    } else if (op == "silu_mul" || op == "gelu_mul") {
        auto gate = work.input("gate");
        auto up = work.input("up");
        output = work.keep(op == "silu_mul" ? silu_mul_cuda(gate, up) : gelu_mul_cuda(gate, up));
    } else if (op == "ssm_conv_silu") {
        auto input = work.input("conv_input");
        auto weight = work.input("weight");
        tyloi::DenseTensor bias;
        if (data.at("inputs").contains("bias"))
            bias = work.input("bias");
        else
            bias = work.keep(tyloi::cuda::empty(
                {0}, tyloi::TensorOptions().dtype(tyloi::kFloat32).device(work.device())));
        output = work.keep(
            ssm_conv_silu_cuda(input, weight, bias, config.at("n_tokens").get<std::int64_t>()));
    } else {
        output = rope(work, config, details);
    }
    work.output("output", output);
    if (data.at("outputs").contains("tail")) {
        const auto rotary = config.at("rotary_dim").get<std::int64_t>();
        auto tail = work.keep(output.narrow(-1, rotary, output.size(-1) - rotary));
        work.output("tail", work.keep(tail.contiguous()));
    }
    work.finish(op, std::move(details));
    return true;
}
} // namespace tyloi::diagnostics::numerics
