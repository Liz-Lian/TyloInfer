#include "engine/ops/gdn.h"

#include "engine/backends/cuda/ops/attention/gdn_core.h"
#include "engine/backends/cuda/executor/executor.h"

#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
namespace {
Tensor hold(DenseTensor value) {
    return Tensor(std::move(value));
}
std::vector<DenseTensor> views(const std::vector<Tensor>& values) {
    std::vector<DenseTensor> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        if (value.defined() && !value.representation<DenseTensor>())
            throw std::invalid_argument("GDN activation must be dense");
        result.push_back(value.representation<DenseTensor>() ? value.dense() : DenseTensor{});
    }
    return result;
}
void publish(std::vector<Tensor>& owners, const std::vector<DenseTensor>& values) {
    std::vector<Tensor> result;
    result.reserve(values.size());
    for (const auto& value : values) result.push_back(hold(value));
    owners = std::move(result);
}
DenseTensor optional_view(const Tensor& value) {
    if (value.defined() && !value.representation<DenseTensor>())
        throw std::invalid_argument("GDN activation must be dense");
    return value.representation<DenseTensor>() ? value.dense() : DenseTensor{};
}
} // namespace

struct Gdn::Impl {
    Device target;
    detail::GdnCore core;

    Impl(GdnWeights weights, std::int64_t key_heads, std::int64_t value_heads,
         std::int64_t width, std::int64_t kernel, double eps, bool silu_gate, Device device)
        : target(device), core(std::make_shared<const GdnWeights>(std::move(weights)), key_heads, value_heads,
               width, kernel, eps, silu_gate, true, cuda::Executor::current().mode().transposed,
               cuda::Executor::current().options().gdn_fused_output, true,
               cuda::Executor::current().options().gdn_fused_preparation,
               cuda::Executor::current().options().gdn_fused_core) {}
};

Gdn::State::State()
    : conv(DenseTensor{}), recurrent(DenseTensor{}), saved_conv(DenseTensor{}), saved_state(DenseTensor{}),
      impl_(std::make_unique<Impl>()) {}
Gdn::State::~State() = default;
Gdn::State::State(State&&) noexcept = default;
Gdn::State& Gdn::State::operator=(State&&) noexcept = default;

Gdn::Gdn(GdnWeights weights, std::int64_t key_heads, std::int64_t value_heads,
         std::int64_t width, std::int64_t kernel, double eps, bool silu_gate, Device target)
    : impl_(std::make_unique<Impl>(std::move(weights), key_heads, value_heads, width, kernel, eps, silu_gate, target)) {}
Gdn::~Gdn() = default;
Gdn::Gdn(Gdn&&) noexcept = default;
Gdn& Gdn::operator=(Gdn&&) noexcept = default;

Gdn::ProjectionScope::ProjectionScope(std::int64_t rows)
    : previous_(detail::gdn_projection_rows) { detail::gdn_projection_rows = rows; }
Gdn::ProjectionScope::~ProjectionScope() { detail::gdn_projection_rows = previous_; }

void Gdn::reset(State& state, bool retain_storage) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    impl_->core.reset(state, retain_storage);
}
void Gdn::commit(State& state) const { impl_->core.commit(state); }
void Gdn::rollback(State& state, std::int64_t keep) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    impl_->core.rollback(state, keep);
}
void Gdn::initialize_request_cache(State& state, bool allocate_counters) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    impl_->core.initialize_request_cache(state, allocate_counters);
}
std::size_t Gdn::growth_bytes(const State& state) const { return impl_->core.growth_bytes(state); }
std::size_t Gdn::request_cache_growth_bytes(const State& state) const {
    return impl_->core.request_cache_growth_bytes(state);
}
void Gdn::restore_session(State& state, const Tensor& convolution, const Tensor& recurrent,
                          bool retain_storage) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    impl_->core.restore_session(state, convolution.dense(), recurrent.dense(), retain_storage);
}
Tensor Gdn::recurrent_state(const State& state) const { return hold(impl_->core.recurrent_state(state)); }
std::vector<Tensor*> Gdn::graph_state(State& state) const { return impl_->core.graph_state(state); }
std::vector<Tensor> Gdn::storage(const State& state) const {
    std::vector<Tensor> result;
    const auto add = [&](const Tensor& value) {
        if (value.defined()) result.push_back(hold(value.dense()));
    };
    add(state.conv);
    add(state.recurrent);
    if (state.impl_) {
        add(state.impl_->core_workspace);
        for (const auto& value : state.impl_->failed_conv) if (value.defined()) result.push_back(hold(value));
        for (const auto& value : state.impl_->failed_state) if (value.defined()) result.push_back(hold(value));
        if (state.impl_->failed_output.defined()) result.push_back(hold(state.impl_->failed_output));
    }
    add(state.saved_conv);
    add(state.saved_state);
    for (const auto& value : state.saved_prefix_conv) add(value);
    for (const auto& value : state.saved_prefix_state) add(value);
    return result;
}
std::vector<Tensor> Gdn::project_inputs(const Tensor& input) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    std::vector<Tensor> result;
    result.reserve(4);
    for (int i=0; i<4; ++i) result.emplace_back(DenseTensor{});
    auto values=impl_->core.project_inputs(input.dense());
    if (values.size()!=result.size()) throw std::logic_error("GDN projection count changed");
    for (std::size_t i=0; i<values.size(); ++i) result[i].dense()=std::move(values[i]);
    return result;
}
Tensor Gdn::project_output(const Tensor& input) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.project_output(input.dense());
    return result;
}
Tensor Gdn::prepare_independent_batch(const std::vector<std::pair<const Gdn*, State*>>& rows) {
    std::optional<MfqCudaGuard> guard;
    if (!rows.empty() && rows.front().first && rows.front().first->impl_->target.is_cuda())
        guard.emplace(rows.front().first->impl_->target);
    Tensor result{DenseTensor{}};
    std::vector<std::pair<const detail::GdnCore*, State*>> cores;
    cores.reserve(rows.size());
    for (const auto& [gdn, state] : rows) cores.emplace_back(&gdn->impl_->core, state);
    result.dense()=detail::GdnCore::prepare_independent_batch(cores);
    return result;
}
Tensor Gdn::forward_independent_batch(State& state, const Tensor& input,
        const std::vector<Tensor>& projections, const Tensor& addresses) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.forward_independent_batch(state, input.dense(), views(projections), optional_view(addresses));
    return result;
}
Tensor Gdn::forward_graph(State& state, const Tensor& input, std::int64_t confirmed,
        std::vector<Tensor>& checkpoint_conv, std::vector<Tensor>& checkpoint_state) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    auto conv = views(checkpoint_conv);
    auto recurrent = views(checkpoint_state);
    DenseTensor result;
    try {
        result = impl_->core.forward_graph(state, input.dense(), confirmed, conv, recurrent);
        publish(checkpoint_conv, conv);
        publish(checkpoint_state, recurrent);
        return hold(result);
    } catch (...) {
        state.impl_->failed_conv = std::move(conv);
        state.impl_->failed_state = std::move(recurrent);
        state.impl_->failed_output = std::move(result);
        throw;
    }
}
void Gdn::bind_graph_checkpoint(State& state, const std::vector<Tensor>& conv,
        const std::vector<Tensor>& recurrent, std::int64_t position) const {
    impl_->core.bind_graph_checkpoint(state, views(conv), views(recurrent), position);
}
Tensor Gdn::forward(State& state, const Tensor& input, bool cache, std::int64_t confirmed,
                    std::int64_t cache_position) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.forward(state, input.dense(), cache, confirmed, cache_position);
    return result;
}
Tensor Gdn::forward_chunk(State& state, const Tensor& input, bool cache, std::vector<Tensor> projections,
        bool project_output, Tensor checkpoint_conv, Tensor checkpoint_state, bool* snapshot_written) const {
    std::optional<MfqCudaGuard> guard;
    if (impl_->target.is_cuda()) guard.emplace(impl_->target);
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.forward_chunk(state, input.dense(), cache, views(projections), project_output,
        optional_view(checkpoint_conv), optional_view(checkpoint_state), snapshot_written);
    return result;
}
} // namespace tyloi::ops
