#include "engine/ops/ple.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/kernels/ple_fused.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
struct Ple::State::Impl {
    std::vector<std::vector<Tensor>> chunk_owners, graph_owners;
};
struct Ple::Impl {
    PleWeights weights;
    std::int64_t hidden, streams, dilation;
    double eps;
    Device target;
    bool fused;
    Impl(PleWeights value, std::int64_t h, std::int64_t s, std::int64_t d, double e, Device device)
        : weights(std::move(value)), hidden(h), streams(s), dilation(d), eps(e), target(device),
          fused([] { const auto value = cuda::options::ple_native_fusion(); return !value || *value; }()) {}
};
Ple::State::State() : conv(DenseTensor{}), saved_conv(DenseTensor{}), impl_(std::make_unique<Impl>()) {}
Ple::State::~State() = default;
Ple::State::State(State&&) noexcept = default;
Ple::State& Ple::State::operator=(State&&) noexcept = default;
Ple::Ple(PleWeights weights, std::int64_t hidden, std::int64_t streams, std::int64_t dilation, double eps, Device target)
    : impl_(std::make_unique<Impl>(std::move(weights), hidden, streams, dilation, eps, target)) {}
Ple::~Ple() = default;
Ple::Ple(Ple&&) noexcept = default;
Ple& Ple::operator=(Ple&&) noexcept = default;
void Ple::reset(State& state, bool retain_storage) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    if (retain_storage && state.conv.defined()) state.conv.dense().zero_();
    else state.conv = {};
    commit(state);
    state.impl_->chunk_owners.clear();
    state.impl_->graph_owners.clear();
}
void Ple::commit(State& state) const {
    state.saved = false;
    state.saved_conv = {};
    state.saved_prefix_conv.clear();
    state.saved_prefix_position = -1;
}
void Ple::rollback(State& state, std::int64_t keep) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    TYLOINFER_RUNTIME_CHECK(state.saved, "Qwen4 PLE has no speculative checkpoint");
    if (keep >= 0 && state.saved_prefix_position >= 0) {
        const auto index = keep - state.saved_prefix_position;
        TYLOINFER_RUNTIME_CHECK(index >= 0 && index < std::int64_t(state.saved_prefix_conv.size()),
            "Qwen4 PLE prefix checkpoint is out of range");
        state.saved_conv = state.saved_prefix_conv[std::size_t(index)];
    }
    if (state.conv.defined()) state.conv.dense().copy_(state.saved_conv.dense());
    else state.conv = state.saved_conv;
    commit(state);
}
void Ple::checkpoint(State& state) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    TYLOINFER_RUNTIME_CHECK(!state.saved && state.conv.defined(), "PLE checkpoint requires committed convolution state");
    std::optional<MfqCudaGuard> guard;
    if (state.conv.device().is_cuda()) guard.emplace(state.conv.device());
    Tensor saved{DenseTensor{}};
    saved.dense() = state.conv.dense().clone();
    state.saved_conv = std::move(saved);
    state.saved = true;
}
void Ple::bind_graph_checkpoint(State& state, const std::vector<Tensor>& conv, std::int64_t position) const {
    TYLOINFER_RUNTIME_CHECK(!state.saved, "resolve previous PLE checkpoint");
    TYLOINFER_RUNTIME_CHECK(!conv.empty(), "invalid PLE prefix checkpoints");
    std::vector<Tensor> prefixes;
    prefixes.reserve(conv.size());
    for (const auto& value : conv) prefixes.emplace_back(value.dense());
    state.saved_conv = prefixes.front();
    state.saved_prefix_conv = std::move(prefixes);
    state.saved_prefix_position = position;
    state.saved = true;
}
void Ple::restore_session(State& state, const Tensor& convolution, bool retain_storage) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    TYLOINFER_RUNTIME_CHECK(convolution.defined(), "missing Qwen4 PLE session state");
    if (retain_storage) {
        TYLOINFER_RUNTIME_CHECK(state.conv.defined() && state.conv.dense().sizes() == convolution.dense().sizes(),
            "Qwen4 PLE graph session geometry changed");
        commit(state);
        state.conv.dense().copy_(convolution.dense());
    } else {
        reset(state);
        Tensor restored{DenseTensor{}};
        restored.dense() = convolution.dense().clone();
        state.conv = std::move(restored);
    }
}
const Tensor& Ple::conv_state(const State& state) const { return state.conv; }
std::vector<Tensor*> Ple::graph_state(State& state) const { return {&state.conv}; }
std::vector<Tensor> Ple::storage(const State& state) const {
    std::vector<Tensor> result;
    const auto add = [&](const Tensor& value) { if (value.defined()) result.emplace_back(value.dense()); };
    add(state.conv); add(state.saved_conv);
    for (const auto& value : state.saved_prefix_conv) add(value);
    for (const auto& group : state.impl_->chunk_owners) for (const auto& value : group) add(value);
    for (const auto& group : state.impl_->graph_owners) for (const auto& value : group) add(value);
    return result;
}
std::size_t Ple::growth_bytes(const State& state, std::int64_t batch) const {
    TYLOINFER_RUNTIME_CHECK(batch > 0, "invalid Qwen4 PLE batch");
    if (state.conv.defined()) return 0;
    const auto& weight = impl_->weights.conv;
    TYLOINFER_RUNTIME_CHECK(weight.defined() && weight.dim() >= 2 && weight.size(-1) > 0 &&
        impl_->hidden > 0 && impl_->streams > 0 && impl_->dilation > 0, "invalid Qwen4 PLE geometry");
    std::size_t bytes = sizeof(float);
    for (auto value : {batch, weight.size(-1) - 1, impl_->dilation, impl_->hidden, impl_->streams}) {
        if (value && bytes > std::numeric_limits<std::size_t>::max() / std::size_t(value))
            throw std::overflow_error("Qwen4 PLE allocation size overflow");
        bytes *= std::size_t(value);
    }
    return bytes;
}
Tensor Ple::forward_chunk(State& state, const Tensor& input, const Tensor& embeddings, bool cache,
        std::vector<Tensor>* checkpoints, std::size_t checkpoint_offset) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    Tensor result{DenseTensor{}}, next{DenseTensor{}};
    std::vector<Tensor> owners;
    owners.reserve(4 + (checkpoints ? checkpoints->size() : 0));
    owners.emplace_back(input.dense()); owners.emplace_back(embeddings.dense()); owners.push_back(result); owners.push_back(next);
    if (checkpoints) for (const auto& checkpoint : *checkpoints) owners.emplace_back(checkpoint.dense());
    state.impl_->chunk_owners.push_back(std::move(owners));
    if (!state.conv.representation<DenseTensor>()) state.conv = Tensor{DenseTensor{}};
    const auto& x = input.dense();
    const auto& rows = embeddings.dense();
    const auto& w = impl_->weights;
    const auto b = x.size(0), t = x.size(1);
    auto conv = state.conv.defined() ? state.conv.dense() : DenseTensor{};
    const auto save_prefixes = [&](const DenseTensor& normalized) {
        if (!checkpoints) return;
        TYLOINFER_RUNTIME_CHECK(cache && conv.defined(), "PLE prefix history is unavailable");
        const auto lookback = conv.size(1);
        for (std::int64_t count = 1; count < t; ++count) {
            auto& saved = checkpoints->at(checkpoint_offset + std::size_t(count - 1)).dense();
            if (count >= lookback) saved.copy_(normalized.narrow(1, count - lookback, lookback));
            else {
                saved.narrow(1, 0, lookback - count).copy_(conv.narrow(1, count, lookback - count));
                saved.narrow(1, lookback - count, count).copy_(normalized.narrow(1, 0, count));
            }
        }
    };
    if (impl_->fused && x.is_contiguous() && t <= 32 && impl_->hidden > 128 && impl_->hidden <= 8192 &&
        (x.scalar_type() == ::tyloi::kFloat16 || x.scalar_type() == ::tyloi::kFloat32)) {
        auto key = mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(rows, w.key).dense(), w.key_norm.dense(), impl_->hidden, impl_->eps);
        auto query = mfq_qwen4_exp::grouped_rms_norm(x, w.query_norm.dense(), impl_->hidden, impl_->eps);
        auto value = ::tyloi::ops::project(rows, w.value).dense();
        auto prepared = mfq_qwen4_exp::ple_gate_norm_cuda(key, query, value,
            w.conv_norm.dense().to(::tyloi::kFloat32).contiguous(), impl_->hidden, impl_->streams, impl_->eps);
        save_prefixes(prepared[1]);
        auto output = mfq_qwen4_exp::ple_conv_add_cuda(prepared[1], w.conv.dense(),
            cache && conv.defined() ? std::optional<DenseTensor>(conv) : std::nullopt,
            impl_->dilation, prepared[0], x.scalar_type() == ::tyloi::kFloat16);
        result.dense() = output[0];
        next.dense() = output[1];
        if (cache) { if (conv.defined()) conv.copy_(next.dense()); else state.conv.dense() = next.dense(); }
        state.impl_->chunk_owners.pop_back();
        return result;
    }
    auto key = mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(rows, w.key).dense(), w.key_norm.dense(),
        impl_->hidden, impl_->eps).reshape({b, t, impl_->streams, impl_->hidden});
    auto query = mfq_qwen4_exp::grouped_rms_norm(x, w.query_norm.dense(), impl_->hidden, impl_->eps)
        .reshape({b, t, impl_->streams, impl_->hidden});
    auto score = (key.to(::tyloi::kFloat32) * query.to(::tyloi::kFloat32)).sum(-1) / std::sqrt(double(impl_->hidden));
    auto sign = (score > 0).to(::tyloi::kFloat32) - (score < 0).to(::tyloi::kFloat32);
    auto root = sign * ::tyloi::cuda::clamp_min(score.abs(), 1e-6).sqrt();
    auto gated = (::tyloi::cuda::sigmoid(root).unsqueeze(-1) *
        ::tyloi::ops::project(rows, w.value).dense().to(::tyloi::kFloat32).unsqueeze(-2))
        .reshape({b, t, impl_->streams * impl_->hidden});
    auto normalized = mfq_qwen4_exp::grouped_rms_norm(gated, w.conv_norm.dense(), impl_->hidden, impl_->eps);
    save_prefixes(normalized);
    auto convolved = mfq_qwen4_exp::ple_dilated_conv_silu(normalized, w.conv.dense(),
        cache && conv.defined() ? std::optional<DenseTensor>(conv) : std::nullopt, impl_->dilation);
    next.dense() = convolved[1];
    result.dense() = (gated + convolved[0]).to(x.scalar_type());
    if (cache) { if (conv.defined()) conv.copy_(next.dense()); else state.conv.dense() = next.dense(); }
    state.impl_->chunk_owners.pop_back();
    return result;
}
Tensor Ple::forward_graph(State& state, const Tensor& input, const Tensor& embeddings, std::int64_t confirmed,
        std::vector<Tensor>& checkpoints) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    Tensor result{DenseTensor{}}, prefix{DenseTensor{}}, suffix{DenseTensor{}};
    std::vector<Tensor> owners;
    owners.reserve(5 + checkpoints.size());
    owners.emplace_back(input.dense()); owners.emplace_back(embeddings.dense()); owners.push_back(result);
    owners.push_back(prefix); owners.push_back(suffix);
    for (const auto& checkpoint : checkpoints) owners.emplace_back(checkpoint.dense());
    state.impl_->graph_owners.push_back(std::move(owners));
    const auto& x = input.dense();
    const auto& rows = embeddings.dense();
    if (!confirmed) {
        result.dense() = forward_chunk(state, input, embeddings, true).dense();
        state.impl_->graph_owners.pop_back();
        return result;
    }
    TYLOINFER_RUNTIME_CHECK(confirmed > 0 && confirmed < x.size(1) &&
        checkpoints.size() >= std::size_t(x.size(1) - confirmed), "invalid PLE graph checkpoints");
    Tensor x_prefix{x.narrow(1, 0, confirmed)}, row_prefix{rows.narrow(1, 0, confirmed)};
    Tensor x_suffix{x.narrow(1, confirmed, x.size(1) - confirmed)}, row_suffix{rows.narrow(1, confirmed, x.size(1) - confirmed)};
    prefix.dense() = forward_chunk(state, x_prefix, row_prefix, true).dense();
    checkpoints.front().dense().copy_(state.conv.dense());
    suffix.dense() = forward_chunk(state, x_suffix, row_suffix, true, &checkpoints, 1).dense();
    result.dense() = ::tyloi::cuda::cat({prefix.dense(), suffix.dense()}, 1);
    state.impl_->graph_owners.pop_back();
    return result;
}
} // namespace tyloi::ops
#endif
