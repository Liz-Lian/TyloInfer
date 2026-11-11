#include "engine/ops/qsa.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/attention/qsa_core.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::ops {
namespace {
Tensor hold(DenseTensor value) { return Tensor(std::move(value)); }
std::vector<DenseTensor> views(const std::vector<Tensor>& values) {
    std::vector<DenseTensor> result;
    result.reserve(values.size());
    for (const auto& value : values) {
        if (value.defined() && !value.representation<DenseTensor>())
            throw std::invalid_argument("QSA activation must be dense");
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
} // namespace

struct Qsa::Impl {
    detail::QsaCore core;
    Impl(QsaWeights weights, QsaConfig config)
        : core(std::make_shared<const QsaWeights>(std::move(weights)), config,
               std::make_shared<cuda::RotaryEmbedding>(config.rotary, config.maximum, config.rope_base,
                   config.sections, config.interleaved, cuda::options::rotary_fusion().value_or(true), config.yarn),
               true, true) {}
};
struct Qsa::State::Impl {
    detail::QsaCore::State cache;
    // Retain failed trace bridge allocations until the request has drained and resets.
    std::vector<DenseTensor> failed_trace;
    DenseTensor failed_output;
    explicit Impl(const detail::QsaCore& model) : cache(model) {}
};

Qsa::Qsa(QsaWeights weights, QsaConfig config)
    : impl_(std::make_unique<Impl>(std::move(weights), std::move(config))) {}
Qsa::~Qsa() = default;
Qsa::Qsa(Qsa&&) noexcept = default;
Qsa& Qsa::operator=(Qsa&&) noexcept = default;
Qsa::State::State(const Qsa& model) : impl_(std::make_unique<Impl>(model.impl_->core)) {}
Qsa::State::~State() = default;
Qsa::State::State(State&&) noexcept = default;
Qsa::State& Qsa::State::operator=(State&&) noexcept = default;

std::int64_t Qsa::position(const State& state) const { return impl_->core.position(state.impl_->cache); }
bool Qsa::managed_memory(const State& state) const { return impl_->core.managed_memory(state.impl_->cache); }
std::int64_t Qsa::graph_capacity(const State& state) const { return impl_->core.graph_capacity(state.impl_->cache); }
std::vector<Tensor*> Qsa::graph_state(State& state) const { return impl_->core.graph_state(state.impl_->cache); }
void Qsa::activate_memory(State& state) const { impl_->core.activate_memory(state.impl_->cache); }
std::size_t Qsa::growth_bytes(const State& state, std::int64_t tokens) const {
    return impl_->core.growth_bytes(state.impl_->cache, tokens);
}
std::size_t Qsa::allocation_bytes(const State& state, std::int64_t batch, std::int64_t tokens, bool reset) const {
    return impl_->core.allocation_bytes(state.impl_->cache, batch, tokens, reset);
}
std::size_t Qsa::exit_growth(const State& state, std::int64_t tokens) const {
    return impl_->core.exit_growth(state.impl_->cache, tokens);
}
std::size_t Qsa::graph_growth_bytes(const State& state, std::int64_t capacity) const {
    return impl_->core.graph_growth_bytes(state.impl_->cache, capacity);
}
bool Qsa::supports_session(const State& state) const { return impl_->core.supports_session(state.impl_->cache); }
void Qsa::validate_session(const State& state, const Snapshot& snapshot, std::int64_t count, Device device) const {
    impl_->core.validate_session(state.impl_->cache, snapshot, count, device);
}
std::size_t Qsa::fresh_session_growth(const State& state, std::int64_t count) const {
    return impl_->core.fresh_session_growth(state.impl_->cache, count);
}
void Qsa::stage_session(State& state, const Snapshot& snapshot, std::int64_t count) const {
    impl_->core.stage_session(state.impl_->cache, snapshot, count);
}
std::size_t Qsa::session_bytes(const State& state, std::int64_t count) const {
    return impl_->core.session_bytes(state.impl_->cache, count);
}
void Qsa::capture_session(const State& state, Snapshot& snapshot, std::int64_t count, std::size_t& bytes) const {
    impl_->core.capture_session(state.impl_->cache, snapshot, count, bytes);
}
void Qsa::restore_session(State& state, const Snapshot& snapshot, std::int64_t count) const {
    impl_->core.restore_session(state.impl_->cache, snapshot, count);
}
std::vector<Tensor> Qsa::resident_cache_tensors(const State& state) const {
    return impl_->core.resident_cache_tensors(state.impl_->cache);
}
std::vector<Tensor> Qsa::storage(const State& state) const {
    std::vector<Tensor> result;
    const auto add = [&](const Tensor& value) {
        if (value.defined()) result.push_back(hold(value.dense()));
    };
    const auto& cache = state.impl_->cache;
    add(cache.keys.storage()); add(cache.values.storage()); add(cache.index.storage());
    add(cache.graph_keys); add(cache.graph_values); add(cache.graph_pooled); add(cache.graph_index);
    if (cache.memory) {
        add(cache.memory->keys); add(cache.memory->values); add(cache.memory->index); add(cache.memory->pooled);
        const auto& hot = cache.memory->hot;
        add(hot.keys); add(hot.values); add(hot.mapping); add(hot.slot_ids); add(hot.requested);
        add(hot.epochs); add(hot.queue); add(hot.control); add(hot.cold_keys); add(hot.cold_values);
        add(hot.cold_mapping); add(hot.free_cold); add(hot.free_gpu); add(hot.cold_control);
    }
    for (const auto& value : state.impl_->failed_trace) if (value.defined()) result.push_back(hold(value));
    if (state.impl_->failed_output.defined()) result.push_back(hold(state.impl_->failed_output));
    return result;
}
void Qsa::leave_graph(State& state) const { impl_->core.leave_graph(state.impl_->cache); }
void Qsa::reset(State& state) const {
    impl_->core.reset(state.impl_->cache);
    state.impl_->failed_trace.clear();
    state.impl_->failed_output = {};
}
void Qsa::prepare_graph(State& state, const Tensor& positions, std::int64_t capacity) const {
    if (positions.defined() && !positions.representation<DenseTensor>())
        throw std::invalid_argument("QSA positions must be dense");
    impl_->core.prepare_graph(state.impl_->cache,
        positions.representation<DenseTensor>() ? positions.dense() : DenseTensor{}, capacity);
}
void Qsa::advance_graph(State& state, std::int64_t tokens) const { impl_->core.advance_graph(state.impl_->cache, tokens); }
void Qsa::truncate(State& state, std::int64_t keep) const { impl_->core.truncate(state.impl_->cache, keep); }
std::vector<Tensor> Qsa::project_inputs(const Tensor& input) const {
    std::vector<Tensor> result;
    result.reserve(4);
    for (int i=0; i<4; ++i) result.emplace_back(DenseTensor{});
    auto values = impl_->core.project_inputs(input.dense());
    if (values.size()!=result.size()) throw std::logic_error("QSA projection count changed");
    for (std::size_t i=0; i<values.size(); ++i) result[i].dense()=std::move(values[i]);
    return result;
}
Tensor Qsa::project_output(const Tensor& input) const {
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.project_output(input.dense());
    return result;
}
Tensor Qsa::forward(State& state, const Tensor& hidden, const Tensor& positions, const Tensor& full_positions,
        bool cache, std::vector<Tensor>* selection_trace, std::vector<Tensor> projections,
        bool project_output, bool history_only) const {
    auto trace = selection_trace ? views(*selection_trace) : std::vector<DenseTensor>{};
    DenseTensor result;
    try {
        result = impl_->core.forward(state.impl_->cache, hidden.dense(), positions.dense(), full_positions.dense(),
            cache, selection_trace ? &trace : nullptr, views(projections), project_output, history_only);
        if (selection_trace) publish(*selection_trace, trace);
        return hold(result);
    } catch (...) {
        state.impl_->failed_trace = std::move(trace);
        state.impl_->failed_output = std::move(result);
        throw;
    }
}
Tensor Qsa::forward_graph(State& state, const Tensor& hidden, const Tensor& positions, const Tensor& cache_positions,
        std::vector<Tensor> projections, bool project_output, bool history_only) const {
    Tensor result{DenseTensor{}};
    result.dense()=impl_->core.forward_graph(state.impl_->cache, hidden.dense(), positions.dense(), cache_positions.dense(),
        views(projections), project_output, history_only);
    return result;
}
struct Qsa::Graph::Impl {
    std::vector<Request> requests;
    std::vector<Row> rows;
    bool independent, admitted = false, clear = false, hidden = false, laid_out = false;
    std::int64_t tokens = 0;
    Impl(std::vector<Request> values, bool batch)
        : requests(std::move(values)), rows(requests.size()), independent(batch) {}
};
Qsa::Graph::Graph(std::vector<Request> requests, bool independent)
    : impl_(std::make_unique<Impl>(std::move(requests), independent)) {}
Qsa::Graph::~Graph() = default;
Qsa::Graph::Graph(Graph&&) noexcept = default;
Qsa::Graph& Qsa::Graph::operator=(Graph&&) noexcept = default;
bool Qsa::Graph::eligible() const {
    return cuda::KvMemoryManager::graph_supported(impl_->independent) &&
        (!impl_->independent || mfq_cuda_graph_capture_supported());
}
bool Qsa::Graph::reusable() const {
    if (impl_->requests.size() != 1 || impl_->requests.front().layers.empty()) return false;
    const auto& layers = impl_->requests.front().layers;
    return std::all_of(layers.begin(), layers.end(), [](const auto& layer) {
        return layer.first->graph_capacity(*layer.second) > 0;
    });
}
bool Qsa::Graph::warmup_supported() const {
    if (!eligible()) return false;
    for (const auto& request : impl_->requests)
        for (const auto& layer : request.layers)
            if (!layer.first->graph_capacity(*layer.second)) return false;
    return true;
}
void Qsa::Graph::admit(std::int64_t tokens, std::int64_t previous_capacity, bool hidden) {
    auto& plan = *impl_;
    if (plan.admitted) throw std::logic_error("QSA graph plan was already admitted");
    if (plan.independent) {
        if (tokens != 1) throw std::invalid_argument("independent QSA graph requires one token");
        std::int64_t required = 0;
        for (const auto& request : plan.requests) required = std::max(required, request.position + 1);
        cuda::KvMemoryManager::ensure(required);
    } else {
        if (plan.requests.size() != 1) throw std::invalid_argument("single QSA graph requires one request");
        const auto& request = plan.requests.front();
        TYLOINFER_RUNTIME_CHECK(tokens > 0 && request.position >= 0 && request.maximum > 0 &&
            request.position <= request.maximum && tokens <= request.maximum - request.position,
            "Qwen4 graph exceeds context capacity");
        const auto required = request.position + tokens;
        auto capacity = previous_capacity;
        if (!request.layers.empty()) {
            const auto& first = request.layers.front();
            if (first.first->graph_capacity(*first.second)) {
                cuda::KvMemoryManager::ensure(required);
                capacity = first.first->graph_capacity(*first.second);
            }
        }
        plan.rows.front().capacity = capacity;
        plan.clear = required > capacity;
    }
    plan.tokens = tokens;
    plan.hidden = hidden;
    plan.admitted = true;
    plan.laid_out = !plan.independent && !plan.clear;
}
bool Qsa::Graph::clear_required() const {
    if (!impl_->admitted) throw std::logic_error("QSA graph plan is not admitted");
    return impl_->clear;
}
void Qsa::Graph::layout() {
    auto& plan = *impl_;
    if (!plan.admitted) throw std::logic_error("QSA graph plan is not admitted");
    if (!plan.independent) {
        const auto& request = plan.requests.front();
        const auto required = request.position + plan.tokens;
        const auto planned = std::max(required, request.reserve);
        TYLOINFER_RUNTIME_CHECK(planned <= request.maximum, "Qwen4 graph reserve exceeds context capacity");
        auto capacity = std::min<std::int64_t>(request.maximum,
            std::max<std::int64_t>(plan.hidden ? 2048 : 256, plan.rows.front().capacity));
        while (capacity < planned)
            capacity = std::min(request.maximum, capacity > request.maximum / 2 ? request.maximum : capacity * 2);
        plan.rows.front().capacity = capacity;
        plan.laid_out = true;
        if (const auto* trace = cuda::env_lookup("TYLOINFER_RUNTIME_TRACE_CUDA_GRAPH"); trace && trace[0] == '1')
            std::cerr << "runtime_qwen4_graph_capacity position=" << request.position
                      << " required=" << required << " planned=" << planned << " capacity=" << capacity << '\n';
        return;
    }
    const auto batch = static_cast<std::int64_t>(plan.requests.size());
    for (std::size_t i = 0; i < plan.requests.size(); ++i) {
        const auto& request = plan.requests[i];
        const auto required = request.position + 1;
        TYLOINFER_RUNTIME_CHECK(required <= request.maximum, "Flash-Next batch graph exceeds max_context");
        auto capacity = std::min<std::int64_t>(batch > 8 ? cuda::options::continuous_batch_kv_initial_capacity() : 256,
            request.maximum);
        if (batch > 16 && capacity < required) {
            const auto chunks = 1 + (required - 1) / 128;
            capacity = chunks > request.maximum / 128 ? request.maximum : chunks * 128;
        } else while (capacity < required)
            capacity = std::min(request.maximum, capacity > request.maximum / 2 ? request.maximum : capacity * 2);
        for (const auto& layer : request.layers)
            if (layer.first->graph_capacity(*layer.second)) capacity = layer.first->graph_capacity(*layer.second);
        auto& row = plan.rows[i];
        row.capacity = capacity;
        row.growth = 0;
        for (const auto& layer : request.layers) row.growth += layer.first->graph_growth_bytes(*layer.second, capacity);
    }
    plan.laid_out = true;
}
const std::vector<Qsa::Graph::Row>& Qsa::Graph::rows() const {
    if (!impl_->admitted) throw std::logic_error("QSA graph plan is not admitted");
    return impl_->rows;
}
void Qsa::Graph::prepare(std::size_t request_index, std::size_t qsa_index, const Tensor& positions) const {
    if (!impl_->admitted || !impl_->laid_out) throw std::logic_error("QSA graph layout is not ready");
    const auto& request = impl_->requests.at(request_index);
    const auto& layer = request.layers.at(qsa_index);
    layer.first->prepare_graph(*layer.second, positions, impl_->rows.at(request_index).capacity);
}
void Qsa::Graph::prepare(std::size_t request_index, const Tensor& positions) const {
    const auto& request = impl_->requests.at(request_index);
    for (std::size_t i = 0; i < request.layers.size(); ++i) prepare(request_index, i, positions);
}

} // namespace tyloi::ops
#endif
