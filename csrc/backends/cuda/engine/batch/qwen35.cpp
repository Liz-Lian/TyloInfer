#include "engine/batch/qwen35.h"

#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/options.h"
#include "engine/executor/ffn.h"
#include "native/error.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mfq::cuda::qwen35 {

using Tensor = mfq_tensor_backend::Tensor;

Batch::~Batch() {
    if (state_)
        detach_paged(*state_);
    state_.reset();
    executor_.reset();
}

static std::shared_ptr<const Model>
require_weights(std::shared_ptr<const Model> weights) {
    if (!weights)
        throw std::invalid_argument("continuous batching requires Qwen35 model weights");
    return weights;
}

Batch::Batch(std::shared_ptr<const Model> weights, CudaExecutionContext &execution,
             std::size_t slots, bool paged)
    : weights_(require_weights(std::move(weights))), execution_(&execution),
      executor_(execution.make_executor()),
      state_(std::make_unique<State>(*weights_, execution.config.gdn_transposed_state)) {
    if (slots == 0 || slots > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("continuous batching max sequences must be positive int32");
    const auto reason = incompatibility(execution);
    if (!reason.empty())
        throw std::runtime_error(reason);
    slots_.resize(slots);
    state_->reset(static_cast<std::int64_t>(slots));
    if (paged) {
        paged_slots_.resize(slots);
        std::vector<PagedKv::Layout> layouts;
        const bool preallocate = runtime_options::kv_vram_preallocated();
        for (const auto &layer : weights_->layers) {
            std::visit(
                [&](const auto &value) {
                    using T = std::decay_t<decltype(value)>;
                    if constexpr (std::is_same_v<T, Full>) {
                        layouts.push_back({layer.device,
                            value.attention.kv_heads > 0 ? value.attention.kv_heads
                                                         : weights_->metadata.num_key_value_heads,
                            value.attention.attention_head_dim > 0
                                ? value.attention.attention_head_dim
                                : weights_->metadata.head_dim,
                            value.attention.kv_quantization});
                    }
                },
                layer.weights);
        }
        paged_kv_ = std::make_unique<PagedKv>(layouts, weights_->metadata.max_position_embeddings,
                                               static_cast<int32_t>(slots), preallocate);
        detach_paged(*state_);
    }
}

bool Batch::paged_kv_enabled() const noexcept { return static_cast<bool>(paged_kv_); }

PagedKv::Stats Batch::paged_kv_stats() const noexcept {
    return paged_kv_ ? paged_kv_->stats() : PagedKv::Stats{};
}

void Batch::prepare_prefill(RequestState &request, std::int64_t offset, std::int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(request.slot_ < 0 && offset >= 0 && offset < tokens,
                            "continuous batching prefill state is invalid");
    if (!request.state_) {
        TYLOINFER_RUNTIME_CHECK(offset == 0, "continuous batching lost prefill state");
        auto executor = execution_->make_executor();
        auto state = std::make_unique<State>(*weights_, execution_->config.gdn_transposed_state);
        state->reset(1);
        if (paged_kv_) {
            detach_paged(*state);
            paged_kv_->ensure_tokens(request.pages_, tokens);
        }
        request.executor_ = std::move(executor);
        request.state_ = std::move(state);
    }
    if (paged_kv_)
        bind_paged(*request.state_, {&request.pages_});
}

void Batch::activate(RequestState &request) {
    TYLOINFER_RUNTIME_CHECK(request.slot_ < 0 && request.state_ && request.executor_,
                            "continuous batching request has no prefill state");
    const auto free = std::find(slots_.begin(), slots_.end(), nullptr);
    TYLOINFER_RUNTIME_CHECK(free != slots_.end(), "continuous batching has no free stable slot");
    const auto slot = static_cast<int32_t>(free - slots_.begin());
    if (state_->batch != static_cast<std::int64_t>(slots_.size())) {
        TYLOINFER_RUNTIME_CHECK(
            std::none_of(slots_.begin(), slots_.end(), [](auto *owner) { return owner != nullptr; }),
            "continuous batching aggregate state has active slots at the wrong batch size");
        state_->reset(static_cast<std::int64_t>(slots_.size()));
    }
    if (paged_kv_)
        bind_paged_slots();
    copy_to_slot(*state_, *request.state_, slot);
    bool pages_swapped = false;
    try {
        if (paged_kv_) {
            auto &target = paged_slots_[slot];
            paged_kv_->release(target);
            std::swap(target, request.pages_);
            pages_swapped = true;
        }
        *free = &request;
        request.slot_ = slot;
        if (paged_kv_)
            bind_paged_slots();
        request.state_.reset();
        request.executor_.reset();
    } catch (...) {
        if (request.slot_ == slot) {
            *free = nullptr;
            request.slot_ = -1;
        }
        if (pages_swapped) {
            detach_paged(*state_);
            detach_paged(*request.state_);
            std::swap(paged_slots_[slot], request.pages_);
        }
        throw;
    }
}

void Batch::resume_decode(std::int64_t cache_position) {
    state_->cache_pos = cache_position;
    const bool active = std::any_of(slots_.begin(), slots_.end(), [](auto *owner) {
        return owner != nullptr;
    });
    if (paged_kv_ && active)
        bind_paged_slots();
    else if (paged_kv_) {
        state_->reset(1);
        detach_paged(*state_);
        for (auto &sequence : paged_slots_)
            paged_kv_->release(sequence);
    } else if (!active) {
        state_->reset(1);
    }
    page_table_dirty_ = false;
}

void Batch::release(RequestState &request) {
    if (request.slot_ >= 0) {
        const auto slot = static_cast<std::size_t>(request.slot_);
        TYLOINFER_RUNTIME_CHECK(slot < slots_.size() && slots_[slot] == &request,
                                "continuous batching stable slot ownership changed");
        if (request.state_)
            detach_paged(*request.state_);
        if (paged_kv_)
            paged_kv_->release(paged_slots_[slot]);
        slots_[slot] = nullptr;
        request.slot_ = -1;
        ++slot_releases_;
    }
    if (request.state_)
        detach_paged(*request.state_);
    if (paged_kv_)
        paged_kv_->release(request.pages_);
    request.state_.reset();
    request.executor_.reset();
}

void Batch::discard_prefill(RequestState &request) {
    mfq_cuda_synchronize();
    release(request);
}

void Batch::finish_retire(std::int64_t cache_position) {
    if (std::none_of(slots_.begin(), slots_.end(), [](auto *owner) { return owner != nullptr; })) {
        state_->reset(1);
        if (paged_kv_) {
            detach_paged(*state_);
            for (auto &sequence : paged_slots_)
                paged_kv_->release(sequence);
        }
    } else if (paged_kv_) {
        state_->cache_pos = cache_position;
        bind_paged_slots();
    } else {
        state_->cache_pos = cache_position;
    }
    page_table_dirty_ = false;
}

void Batch::ensure_decode_tokens(const RequestState &request, std::int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(request.slot_ >= 0 &&
                                static_cast<std::size_t>(request.slot_) < slots_.size() &&
                                slots_[request.slot_] == &request,
                            "continuous batching request lost its stable slot");
    if (paged_kv_)
        page_table_dirty_ =
            paged_kv_->ensure_tokens(paged_slots_[request.slot_], tokens) || page_table_dirty_;
}

void Batch::prepare_decode(std::int64_t cache_position) {
    if (page_table_dirty_)
        bind_paged_slots();
    state_->cache_pos = cache_position;
}

void Batch::finish_decode(std::int64_t cache_position) { state_->cache_pos = cache_position; }

void Batch::recover(const std::vector<RequestState *> &requests) {
    try {
        mfq_cuda_synchronize();
    } catch (...) {
    }
    for (auto *request : requests)
        release(*request);
    finish_retire(0);
}

void Batch::bind_paged(State &state, const std::vector<const PagedKv::Sequence *> &sequences) {
    TYLOINFER_RUNTIME_CHECK(paged_kv_, "continuous batching paged KV state is unavailable");
    paged_kv_->update(sequences);
    std::size_t state_index = 0;
    std::size_t paged_layer = 0;
    for (const auto &layer : weights_->layers) {
        std::visit(
            [&](const auto &value) {
                using T = std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T, Full>) {
                    MfqCudaGuard guard(layer.device);
                    auto *cache = std::get_if<KVCache>(&state.layers[state_index]);
                    TYLOINFER_RUNTIME_CHECK(cache, "continuous batching full state layout changed");
                    *cache = paged_kv_->view(paged_layer++, static_cast<int64_t>(sequences.size()));
                }
                ++state_index;
            },
            layer.weights);
    }
}

void Batch::detach_paged(State &state) noexcept {
    if (!paged_kv_)
        return;
    for (auto &layer_state : state.layers)
        if (auto *cache = std::get_if<KVCache>(&layer_state))
            *cache = KVCache();
}

void Batch::bind_paged_slots() {
    if (!paged_kv_)
        return;
    std::vector<const PagedKv::Sequence *> sequences;
    sequences.reserve(paged_slots_.size());
    for (auto &sequence : paged_slots_) {
        paged_kv_->ensure_tokens(sequence, 1);
        sequences.push_back(&sequence);
    }
    bind_paged(*state_, sequences);
    page_table_dirty_ = false;
}

bool Batch::has_moe() const {
    for (const auto &layer : weights_->layers) {
        const bool result = std::visit(
            [](const auto &value) { return value.ffn.is_moe; }, layer.weights);
        if (result)
            return true;
    }
    return false;
}

bool Batch::has_cached_moe() const { return has_cached_moe(*weights_); }

bool Batch::has_cached_moe(const Model &model) {
    for (const auto &layer : model.layers) {
        const bool result = std::visit(
            [](const auto &value) { return mfq::cuda::uses_expert_cache(value.ffn); },
            layer.weights);
        if (result)
            return true;
    }
    return false;
}

std::string Batch::incompatibility(const CudaExecutionContext &execution) const {
    if (weights_->layers.empty())
        return "continuous batching requires at least one model block";
    if (execution.dense_cpu_layer_count != 0 || !execution.dsv4_cpu_offload_layers.empty())
        return "continuous batching requires GPU-resident model blocks";
    for (const auto &layer : weights_->layers) {
        if (layer.cpu)
            return "continuous batching cannot use CPU-offloaded blocks";
        const auto reason = std::visit(
            [&](const auto &value) -> std::string {
                if constexpr (std::is_same_v<std::decay_t<decltype(value)>, Full>) {
                    if (value.attention.sliding)
                        return "continuous batching requires non-sliding Qwen attention";
                }
                if (mfq::cuda::uses_expert_cache(value.ffn) &&
                    value.ffn.moe_top_k > execution.moe_cache_registration_min_slots)
                    return "continuous batching requires one cached slot per routed expert";
                return {};
            },
            layer.weights);
        if (!reason.empty())
            return reason;
    }
    return {};
}

void Batch::copy_to_slot(State &target, const State &source, std::int64_t slot) const {
    TYLOINFER_RUNTIME_CHECK(source.batch == 1 && slot >= 0 && slot < target.batch &&
                                target.layers.size() == source.layers.size() &&
                                source.speculative_start < 0,
                            "continuous batching received an invalid stable slot");
    for (std::size_t index = 0; index < target.layers.size(); ++index) {
        MfqCudaGuard guard(weights_->layers[index].device);
        auto &destination = target.layers[index];
        const auto &origin = source.layers[index];
        TYLOINFER_RUNTIME_CHECK(destination.index() == origin.index(),
                                "continuous batching stable slot layout changed");
        if (auto *cache = std::get_if<KVCache>(&destination)) {
            const auto &from = std::get<KVCache>(origin);
            TYLOINFER_RUNTIME_CHECK(from.defined() && from.batch_size() == 1,
                                    "continuous batching KV state is unavailable");
            TYLOINFER_RUNTIME_CHECK(cache->is_paged() == from.is_paged(),
                                    "continuous batching stable slot KV layout changed");
            if (!cache->is_paged()) {
                TYLOINFER_RUNTIME_CHECK(from.k.dim() == 4 && from.v.dim() == 4 &&
                                            from.k.size(0) == 1 && from.v.size(0) == 1 &&
                                            !from.ring &&
                                            session_kv_geometry_matches(from.k, from.v),
                                        "continuous batching KV cache geometry changed");
                if (cache->defined())
                    TYLOINFER_RUNTIME_CHECK(
                        cache->batch_size() == target.batch && !cache->ring &&
                            cache->k.dim() == 4 && cache->v.dim() == 4 &&
                            session_kv_geometry_matches(cache->k, cache->v) &&
                            cache->k.size(1) == from.k.size(1) &&
                            cache->v.size(1) == from.v.size(1) &&
                            cache->k.size(2) == from.k.size(2) &&
                            cache->v.size(2) == from.v.size(2) &&
                            cache->k.size(3) == from.k.size(3) &&
                            cache->v.size(3) == from.v.size(3),
                        "continuous batching stable slot KV geometry changed");
                if (!cache->defined()) {
                    *cache = from;
                    auto key_shape = from.k.sizes().vec();
                    auto value_shape = from.v.sizes().vec();
                    key_shape[0] = target.batch;
                    value_shape[0] = target.batch;
                    cache->k = mfq_tensor_backend::zeros(key_shape, from.k.options());
                    cache->v = mfq_tensor_backend::zeros(value_shape, from.v.options());
                }
                cache->k.narrow(0, slot, 1).copy_(from.k);
                cache->v.narrow(0, slot, 1).copy_(from.v);
            }
        } else {
            auto &state = std::get<LinearState>(destination);
            const auto &from = std::get<LinearState>(origin);
            TYLOINFER_RUNTIME_CHECK(
                from.mixer.defined() && !from.mixer.pending() &&
                    from.mixer.view().convolution.size(0) == 1 &&
                    from.mixer.view().recurrent.size(0) == 1,
                "continuous batching recurrent state is unavailable");
            const auto &source = from.mixer.view();
            auto &target_view = state.mixer.view();
            if (!target_view.defined()) {
                auto convolution_shape = source.convolution.sizes().vec();
                auto recurrent_shape = source.recurrent.sizes().vec();
                convolution_shape[0] = target.batch;
                recurrent_shape[0] = target.batch;
                target_view.convolution = mfq_tensor_backend::zeros(
                    convolution_shape, source.convolution.options());
                target_view.recurrent = mfq_tensor_backend::zeros(
                    recurrent_shape, source.recurrent.options());
                target_view.transposed = source.transposed;
            }
            TYLOINFER_RUNTIME_CHECK(target_view.transposed == source.transposed,
                                    "continuous batching recurrent layout changed");
            target_view.convolution.narrow(0, slot, 1).copy_(source.convolution);
            target_view.recurrent.narrow(0, slot, 1).copy_(source.recurrent);
        }
    }
}

Tensor Batch::logits_from_last_hidden(Tensor hidden) {
    auto last = hidden.index({mfq_tensor_backend::indexing::Slice(), -1,
                              mfq_tensor_backend::indexing::Slice()})
                    .to(mfq_tensor_backend::kFloat16)
                    .contiguous();
    return qwen35::logits(*weights_, *executor_, std::move(last), true);
}

std::vector<const void *> Batch::decode_state_addresses() {
    std::vector<const void *> addresses;
    const auto storage = state_->storage();
    addresses.reserve(storage.size());
    for (const auto &tensor : storage)
        addresses.push_back(tensor.data_ptr());
    return addresses;
}

BatchState Batch::capture_recurrent_slots(const std::vector<std::int32_t> &slots) const {
    BatchState state;
    state.batch = static_cast<int64_t>(slots.size());
    if (slots.empty())
        return state;
    state.layers.resize(state_->layers.size());
    for (std::size_t index = 0; index < state_->layers.size(); ++index) {
        const auto *linear = std::get_if<LinearState>(&state_->layers[index]);
        if (!linear)
            continue;
        MfqCudaGuard guard(weights_->layers[index].device);
        std::vector<Tensor> conv, recurrent;
        for (auto slot : slots) {
            conv.push_back(linear->mixer.view().convolution.narrow(0, slot, 1));
            recurrent.push_back(linear->mixer.view().recurrent.narrow(0, slot, 1));
        }
        state.layers[index] = Recurrent::State{
            mfq_tensor_backend::cat(conv, 0).clone(),
            mfq_tensor_backend::cat(recurrent, 0).clone(),
            linear->mixer.view().transposed};
    }
    return state;
}

void Batch::restore_recurrent_slots(const std::vector<std::int32_t> &slots,
                                    const BatchState &state) {
    if (slots.empty())
        return;
    TYLOINFER_RUNTIME_CHECK(state.batch == static_cast<int64_t>(slots.size()) &&
                                state.layers.size() == state_->layers.size(),
                            "continuous batching recurrent snapshot layout changed");
    for (std::size_t index = 0; index < state_->layers.size(); ++index) {
        auto *linear = std::get_if<LinearState>(&state_->layers[index]);
        if (!linear)
            continue;
        MfqCudaGuard guard(weights_->layers[index].device);
        const auto *saved = std::get_if<Recurrent::State>(&state.layers[index]);
        TYLOINFER_RUNTIME_CHECK(saved && saved->defined() &&
                                    saved->convolution.size(0) == state.batch &&
                                    saved->recurrent.size(0) == state.batch &&
                                    saved->transposed == linear->mixer.view().transposed,
                                "continuous batching recurrent snapshot changed");
    }
    for (std::size_t index = 0; index < state_->layers.size(); ++index) {
        auto *linear = std::get_if<LinearState>(&state_->layers[index]);
        if (!linear)
            continue;
        MfqCudaGuard guard(weights_->layers[index].device);
        const auto &saved = std::get<Recurrent::State>(state.layers[index]);
        for (std::size_t row = 0; row < slots.size(); ++row) {
            linear->mixer.view().convolution.narrow(0, slots[row], 1)
                .copy_(saved.convolution.narrow(0, row, 1));
            linear->mixer.view().recurrent.narrow(0, slots[row], 1)
                .copy_(saved.recurrent.narrow(0, row, 1));
        }
    }
}

} // namespace mfq::cuda::qwen35
