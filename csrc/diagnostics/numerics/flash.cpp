#include "flash.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/glm5_next.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
using tyloi::DenseTensor;

void observe(TensorWork &work, const std::string &name, DenseTensor value) {
    value = work.keep(std::move(value));
    auto converted = work.keep(value.to(tyloi::kFloat32));
    work.output(name, work.keep(converted.contiguous()));
}

DenseTensor host_float(TensorWork &work, DenseTensor value) {
    return work.keep(value.to(tyloi::kCPU).to(tyloi::kFloat32).contiguous());
}

void check(Json &checks, int index, bool passed, Json evidence) {
    checks["native_" + std::to_string(index)] = {{"status", passed ? "pass" : "fail"},
                                                 {"evidence", std::move(evidence)}};
}

void norm(Case &value, TensorWork &work, Json &checks) {
    const auto &config = value.data().at("config");
    const auto width = config.at("group_size").get<std::int64_t>();
    const auto count = config.at("hc_count").get<std::int64_t>();
    auto x = work.input("x");
    auto weights = work.input("norm");
    auto down = work.input("down");
    auto up = work.input("up");
    auto inject = work.input("inject");
    auto branch = work.input("branch");
    auto post_gates = work.input("post_gates");
    observe(work, "normalized", mfq_qwen4_exp::grouped_rms_norm(x, weights, width, 1e-6));
    auto pre = mfq_qwen4_exp::gated_residual_pre(x, weights, down, up, inject, width, count, 1e-6);
    for (auto &item : pre)
        work.keep(item);
    if (pre.size() != 3)
        throw std::runtime_error("invalid gated residual result");
    observe(work, "pre", pre[0]);
    observe(work, "residual", pre[1]);
    observe(work, "gates", pre[2]);
    auto absent =
        mfq_qwen4_exp::gated_residual_pre(x, weights, down, up, std::nullopt, width, count, 1e-6);
    for (auto &item : absent)
        work.keep(item);
    if (absent.size() != 3)
        throw std::runtime_error("invalid optional gated residual result");
    observe(work, "no_inject_pre", absent[0]);
    const bool no_injection = !absent[2].defined();
    check(checks, 0, no_injection, {{"injection_defined", !no_injection}});
    observe(work, "post", mfq_qwen4_exp::gated_residual_post(branch, x, post_gates, count));
}

void mhc(Case &value, TensorWork &work) {
    const auto iterations = value.data().at("config").at("sinkhorn_iterations").get<std::int64_t>();
    auto x = work.input("x");
    auto function = work.input("function");
    auto base = work.input("base");
    auto scale = work.input("scale");
    auto branch = work.input("branch");
    auto post = work.input("post_input");
    auto comb = work.input("comb_input");
    auto result = mfq_glm5_next::mhc_pre(x, function, base, scale, iterations, 1e-6, 1e-5);
    for (auto &item : result)
        work.keep(item);
    if (result.size() != 3)
        throw std::runtime_error("invalid mHC result");
    observe(work, "post_gate", result[0]);
    observe(work, "comb", result[1]);
    observe(work, "pre", result[2]);
    auto post_residual = work.keep(x.contiguous());
    observe(work, "post", mfq_glm5_next::mhc_post(branch, post_residual, post, comb));
}

void kda(Case &value, TensorWork &work, Json &checks) {
    const auto &config = value.data().at("config");
    auto x = work.input("x");
    auto fa = work.input("fa");
    auto fb = work.input("fb");
    auto bias = work.input("bias");
    auto log = work.input("log");
    const auto lower = config.at("lower_bound").get<double>();
    auto result = work.keep(mfq_glm5_next::kda_forget_gate(
        x, fa, fb, bias, log, config.at("num_heads").get<std::int64_t>(),
        config.at("head_dim").get<std::int64_t>(), lower));
    auto host = host_float(work, result);
    const auto *values = host.data_ptr<float>();
    bool bounded = true;
    for (std::int64_t i = 0; i < host.numel(); ++i)
        bounded = bounded && values[i] >= lower && values[i] <= 0;
    check(checks, 0, bounded,
          {{"lower_bound", lower}, {"upper_bound", 0}, {"elements", host.numel()}});
    observe(work, "output", result);
}

void scores(const std::string &op, Case &value, TensorWork &work) {
    if (op == "kpool_states") {
        auto keys = work.input("keys");
        auto gates = work.input("gates");
        auto ape = work.input("ape");
        observe(
            work, "output",
            mfq_glm5_next::kpool_states(
                keys, gates, ape, value.data().at("config").at("pool_size").get<std::int64_t>()));
        return;
    }
    auto q = work.input("q");
    auto k = work.input("k");
    if (op == "qsa")
        observe(work, "output", mfq_qwen4_exp::block_scores(q, k));
    else {
        auto weights = work.input("weights");
        observe(work, "output", mfq_glm5_next::kpool_scores(q, k, weights));
    }
}

void attention(Case &value, TensorWork &work, Json &checks) {
    const auto &config = value.data().at("config");
    const bool mla = config.at("mla").get<bool>();
    const bool sparse = config.at("sparse").get<bool>();
    auto q = work.input("q");
    auto k = work.input("k");
    DenseTensor output;
    if (mla) {
        std::optional<double> scale;
        if (!config.at("scale").is_null())
            scale = config.at("scale").get<double>();
        if (sparse)
            output = mfq_glm5_next::sparse_mla_attention(q, k, work.input("indices"), scale);
        else
            output = mfq_glm5_next::dense_mla_attention(
                q, k, config.at("query_offset").get<std::int64_t>(), scale);
    } else {
        auto v = work.input("v");
        if (sparse)
            output = mfq_qwen4_exp::sparse_gqa_attention(q, k, v, work.input("indices"));
        else
            output = mfq_qwen4_exp::dense_gqa_attention(
                q, k, v, config.at("query_offset").get<std::int64_t>());
    }
    output = work.keep(std::move(output));
    if (config.contains("native_checks")) {
        auto host = host_float(work, output);
        if (host.dim() != 4 || host.size(0) != 2 || host.size(1) != 2)
            throw std::runtime_error("invalid empty-selection observation shape");
        const auto span = host.size(2) * host.size(3);
        const auto *values = host.data_ptr<float>() + 2 * span;
        bool zero = true;
        for (std::int64_t i = 0; i < span; ++i)
            zero = zero && values[i] == 0.0f;
        check(checks, 0, zero, {{"batch", 1}, {"token", 0}, {"elements", span}});
    }
    observe(work, "output", output);
}

void ple(Case &value, TensorWork &work) {
    auto x = work.input("x");
    auto first_input = work.input("first_input");
    auto second_input = work.input("second_input");
    auto weight = work.input("weight");
    const auto dilation = value.data().at("config").at("dilation").get<std::int64_t>();
    auto full = mfq_qwen4_exp::ple_dilated_conv_silu(x, weight, std::nullopt, dilation);
    for (auto &item : full)
        work.keep(item);
    auto first = mfq_qwen4_exp::ple_dilated_conv_silu(first_input, weight, std::nullopt, dilation);
    for (auto &item : first)
        work.keep(item);
    if (full.size() != 2 || first.size() != 2)
        throw std::runtime_error("invalid PLE result");
    auto second = mfq_qwen4_exp::ple_dilated_conv_silu(second_input, weight, first[1], dilation);
    for (auto &item : second)
        work.keep(item);
    if (second.size() != 2)
        throw std::runtime_error("invalid PLE chunk result");
    auto joined = work.keep(tyloi::cuda::cat({first[0], second[0]}, 1));
    observe(work, "full", full[0]);
    observe(work, "joined", joined);
    observe(work, "full_state", full[1]);
    observe(work, "last_state", second[1]);
}
} // namespace

bool run_flash(Case &value) {
    const auto op = value.data().at("op").get<std::string>();
    if (op != "grouped_norm_gated_residual" && op != "mhc_sinkhorn" && op != "kda_forget_gate" &&
        op != "qsa" && op != "kpool_scores" && op != "kpool_states" && op != "attention" &&
        op != "ple_dilated_conv_silu")
        return false;
    if (value.data().at("source").at("generator") != "tests/numerics/flash_ref.py")
        throw std::invalid_argument("unsupported Flash-Next case source");
    TensorWork work(value);
    Json checks = Json::object();
    if (op == "grouped_norm_gated_residual")
        norm(value, work, checks);
    else if (op == "mhc_sinkhorn")
        mhc(value, work);
    else if (op == "kda_forget_gate")
        kda(value, work, checks);
    else if (op == "qsa" || op == "kpool_scores" || op == "kpool_states")
        scores(op, value, work);
    else if (op == "attention")
        attention(value, work, checks);
    else
        ple(value, work);
    work.finish(op, Json::object(), std::move(checks));
    return true;
}
} // namespace tyloi::diagnostics::numerics
