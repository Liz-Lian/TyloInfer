#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/models/qwen35/batch.h"
#include "engine/ops.h"

#include "engine/weights/moe/experts.h"
#include "components/error.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi::cuda::qwen35 {

using DenseTensor = ::tyloi::DenseTensor;

template <class Ffn>
static bool has_cached_experts(const Ffn& ffn) {
    if (!ffn.is_moe)
        return false;
    if (ffn.moe_split_gate_up)
        return ::tyloi::weights::cached(ffn.moe_gate) ||
            ::tyloi::weights::cached(ffn.moe_up) ||
            ::tyloi::weights::cached(ffn.moe_down);
    return ::tyloi::weights::cached(ffn.moe_gate_up) ||
        ::tyloi::weights::cached(ffn.moe_down);
}

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

static std::unique_ptr<Executor>
require_executor(std::unique_ptr<Executor> executor) {
    if (!executor)
        throw std::invalid_argument("continuous batching requires an executor");
    return executor;
}

Batch::Batch(std::shared_ptr<const Model> weights, std::unique_ptr<Executor> executor,
             const bool &transposed, Options options)
    : weights_(require_weights(std::move(weights))), transposed_(transposed),
      executor_(require_executor(std::move(executor))),
      state_(std::make_unique<State>(*weights_, transposed_)) {
    if (options.slots == 0 ||
        options.slots > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("continuous batching max sequences must be positive int32");
    const auto reason = incompatibility(options.cpu_offload, options.cached_slots);
    if (!reason.empty())
        throw std::runtime_error(reason);
    slots_.resize(options.slots);
    state_->reset(static_cast<std::int64_t>(options.slots));
    if (options.paged) {
        paged_slots_.resize(options.slots);
        std::vector<PagedKv::Layout> layouts;
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
                                               static_cast<int32_t>(options.slots),
                                               options.preallocate);
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
        auto executor = executor_->fork();
        auto state = std::make_unique<State>(*weights_, transposed_);
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
                    auto *cache = std::get_if<tyloi::state::KvView>(&state.layers[state_index]);
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
        if (auto *cache = std::get_if<tyloi::state::KvView>(&layer_state))
            *cache = {};
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
            [](const auto &value) { return has_cached_experts(value.ffn); },
            layer.weights);
        if (result)
            return true;
    }
    return false;
}

std::string Batch::incompatibility(bool cpu_offload, int cached_slots) const {
    if (weights_->layers.empty())
        return "continuous batching requires at least one model block";
    if (cpu_offload)
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
                if (has_cached_experts(value.ffn) &&
                    value.ffn.moe_top_k > cached_slots)
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
        if (auto *cache = std::get_if<tyloi::state::KvView>(&destination)) {
            const auto &from = std::get<tyloi::state::KvView>(origin);
            TYLOINFER_RUNTIME_CHECK(from.defined() && from.batch == 1,
                                    "continuous batching KV state is unavailable");
            TYLOINFER_RUNTIME_CHECK(cache->paged() == from.paged(),
                                    "continuous batching stable slot KV layout changed");
            if (!cache->paged()) {
                const auto& from_key = from.physical_key;
                const auto& from_value = from.physical_value;
                TYLOINFER_RUNTIME_CHECK(from_key.dim() == 4 && from_value.dim() == 4 &&
                                            from_key.size(0) == 1 && from_value.size(0) == 1 &&
                                            !from.ring &&
                                            session_kv_geometry_matches(from_key, from_value),
                                        "continuous batching KV cache geometry changed");
                if (cache->defined())
                    TYLOINFER_RUNTIME_CHECK(
                        cache->batch == target.batch && !cache->ring &&
                            cache->physical_key.dim() == 4 && cache->physical_value.dim() == 4 &&
                            session_kv_geometry_matches(cache->physical_key, cache->physical_value) &&
                            cache->physical_key.size(1) == from_key.size(1) &&
                            cache->physical_value.size(1) == from_value.size(1) &&
                            cache->physical_key.size(2) == from_key.size(2) &&
                            cache->physical_value.size(2) == from_value.size(2) &&
                            cache->physical_key.size(3) == from_key.size(3) &&
                            cache->physical_value.size(3) == from_value.size(3),
                        "continuous batching stable slot KV geometry changed");
                if (!cache->defined()) {
                    *cache = from;
                    std::vector<int64_t> key_shape(from_key.dim()), value_shape(from_value.dim());
                    for (int64_t axis = 0; axis < from_key.dim(); ++axis) key_shape[axis] = from_key.size(axis);
                    for (int64_t axis = 0; axis < from_value.dim(); ++axis) value_shape[axis] = from_value.size(axis);
                    key_shape[0] = target.batch;
                    value_shape[0] = target.batch;
                    cache->physical_key = tyloi::ops::zeros(key_shape, from_key.options());
                    cache->physical_value = tyloi::ops::zeros(value_shape, from_value.options());
                    cache->batch = target.batch;
                }
                auto key_row = tyloi::ops::slice(cache->physical_key, 0, slot, slot + 1);
                auto value_row = tyloi::ops::slice(cache->physical_value, 0, slot, slot + 1);
                tyloi::ops::copy_into(key_row, from_key);
                tyloi::ops::copy_into(value_row, from_value);
            }
        } else {
            auto &state = std::get<LinearState>(destination);
            const auto &from = std::get<LinearState>(origin);
#ifdef TYLOINFER_NATIVE_CUDA
            const auto source = from.attention.view();
            TYLOINFER_RUNTIME_CHECK(from.attention.defined() && !from.attention.pending() &&
                source.convolution.size(0)==1 && source.recurrent.size(0)==1,
                "continuous batching recurrent state is unavailable");
            if (!state.attention.defined()) state.attention.initialize(source,target.batch);
            auto target_view = state.attention.view();
#else
            TYLOINFER_RUNTIME_CHECK(
                from.mixer.defined() && !from.mixer.pending() &&
                    from.mixer.view().convolution.size(0) == 1 &&
                    from.mixer.view().recurrent.size(0) == 1,
                "continuous batching recurrent state is unavailable");
            const auto &source = from.mixer.view();
            auto &target_view = state.mixer.view();
            if (!target_view.defined()) {
                auto convolution_shape = source.convolution.dense().sizes().vec();
                auto recurrent_shape = source.recurrent.dense().sizes().vec();
                convolution_shape[0] = target.batch;
                recurrent_shape[0] = target.batch;
                target_view.convolution = ::tyloi::ops::zeros(
                    convolution_shape, source.convolution.options());
                target_view.recurrent = ::tyloi::ops::zeros(
                    recurrent_shape, source.recurrent.options());
                target_view.transposed = source.transposed;
            }
#endif
            TYLOINFER_RUNTIME_CHECK(target_view.transposed == source.transposed,
                                    "continuous batching recurrent layout changed");
            auto conv = ::tyloi::ops::slice(target_view.convolution, 0, slot, slot + 1);
            auto recurrent = ::tyloi::ops::slice(target_view.recurrent, 0, slot, slot + 1);
            ::tyloi::ops::copy_into(conv, source.convolution);
            ::tyloi::ops::copy_into(recurrent, source.recurrent);
        }
    }
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Batch::logits_from_last_hidden(::tyloi::Tensor hidden, const Executor::Mode& mode) {
    auto last = tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::select(hidden, 1, -1), ::tyloi::kFloat16));
#else
DenseTensor Batch::logits_from_last_hidden(DenseTensor hidden, const Executor::Mode& mode) {
    auto last = hidden.index({::tyloi::indexing::Slice(), -1,
                              ::tyloi::indexing::Slice()})
                    .to(::tyloi::kFloat16)
                    .contiguous();
#endif
    return qwen35::logits(*weights_, *executor_, std::move(last), true, mode);
}

std::vector<const void *> Batch::decode_state_addresses() {
    std::vector<const void *> addresses;
    const auto storage = state_->storage();
    addresses.reserve(storage.size());
    for (const auto &tensor : storage)
        addresses.push_back(tensor.dense().data_ptr());
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
        std::vector<::tyloi::Tensor> conv, recurrent;
#ifdef TYLOINFER_NATIVE_CUDA
        const auto view=linear->attention.view();
#else
        const auto& view=linear->mixer.view();
#endif
        for (auto slot : slots) {
            conv.push_back(::tyloi::ops::slice(view.convolution, 0, slot, slot + 1));
            recurrent.push_back(::tyloi::ops::slice(view.recurrent, 0, slot, slot + 1));
        }
        state.layers[index] = Recurrent::State{
            ::tyloi::ops::clone(::tyloi::ops::cat(conv, 0)),
            ::tyloi::ops::clone(::tyloi::ops::cat(recurrent, 0)),
            view.transposed};
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
#ifdef TYLOINFER_NATIVE_CUDA
        const auto view=linear->attention.view();
#else
        const auto& view=linear->mixer.view();
#endif
        TYLOINFER_RUNTIME_CHECK(saved && saved->defined() &&
                                    saved->convolution.size(0) == state.batch &&
                                    saved->recurrent.size(0) == state.batch &&
                                    saved->transposed == view.transposed,
                                "continuous batching recurrent snapshot changed");
    }
    for (std::size_t index = 0; index < state_->layers.size(); ++index) {
        auto *linear = std::get_if<LinearState>(&state_->layers[index]);
        if (!linear)
            continue;
        MfqCudaGuard guard(weights_->layers[index].device);
        const auto &saved = std::get<Recurrent::State>(state.layers[index]);
#ifdef TYLOINFER_NATIVE_CUDA
        const auto view=linear->attention.view();
#else
        const auto& view=linear->mixer.view();
#endif
        for (std::size_t row = 0; row < slots.size(); ++row) {
            auto conv = ::tyloi::ops::slice(view.convolution, 0, slots[row], slots[row] + 1);
            auto recurrent = ::tyloi::ops::slice(view.recurrent, 0, slots[row], slots[row] + 1);
            ::tyloi::ops::copy_into(conv, ::tyloi::ops::slice(saved.convolution, 0, row, row + 1));
            ::tyloi::ops::copy_into(recurrent, ::tyloi::ops::slice(saved.recurrent, 0, row, row + 1));
        }
    }
}

} // namespace tyloi::cuda::qwen35

#endif // !TYLOINFER_NATIVE_CUDA

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/batch.h"
#include "engine/state/paged_kv.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/ops.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tyloi::cuda::qwen35 {
namespace {
using Growth = tyloi::models::BatchGrowth;
void add(Growth& out, Device device, std::size_t bytes) {
    if (!device.is_cuda() || !bytes) return;
    auto found=std::find_if(out.begin(),out.end(),[&](const auto& value){return value.first==device;});
    if (found==out.end()) out.emplace_back(device,bytes);
    else { if (found->second>std::numeric_limits<std::size_t>::max()-bytes) throw std::overflow_error("Qwen batch growth overflow"); found->second+=bytes; }
}
void merge(Growth& out,const Growth& more) { for (const auto& [device,bytes]:more) add(out,device,bytes); }
std::size_t mul(std::size_t a,std::size_t b) {
    if (b && a>std::numeric_limits<std::size_t>::max()/b) throw std::overflow_error("Qwen batch growth overflow");
    return a*b;
}
bool cached(const ::tyloi::ops::FfnWeights& f) {
    return f.is_moe && (f.moe_split_gate_up ?
        (tyloi::weights::cached(f.moe_gate) || tyloi::weights::cached(f.moe_up) || tyloi::weights::cached(f.moe_down)) :
        (tyloi::weights::cached(f.moe_gate_up) || tyloi::weights::cached(f.moe_down)));
}
class Batch final : public tyloi::models::Batch {
    struct Request final : tyloi::models::BatchState {
        Batch* owner;
        State state;
        PagedKv::Sequence pages;
        std::int64_t prompt;
        int slot=-1;
        std::int64_t active_position=0;
        bool initialized=false;
        Request(Batch& batch,std::int64_t tokens) : owner(&batch),state(batch.model_,batch.model_.transposed),prompt(tokens) {}
        ~Request() override {
            if (!owner) return;
            try { owner->release(*this); } catch (...) {}
            std::erase(owner->requests_,this);
        }
        std::int64_t position() const override { return slot>=0 ? active_position : state.position(); }
        std::size_t resident_bytes() const override {
            std::map<const void*,std::size_t> seen;
            for (const auto& t:state.storage()) if (t.defined() && t.device().is_cuda()) {
                const auto& d=t.dense(); const auto& storage=d.storage();
                if (storage) { const void* id=storage->owner ? storage->owner.get() : storage->base; if (id) seen[id]=std::max(seen[id],storage->bytes); }
            }
            std::size_t result=0; for (const auto& [_,bytes]:seen) { if (result>std::numeric_limits<std::size_t>::max()-bytes) throw std::overflow_error("Qwen batch resident overflow"); result+=bytes; } return result;
        }
        Growth initial_growth_bytes() const override {
            Growth result;
            if (owner->paged_) { const std::array requests{std::pair<const PagedKv::Sequence*,int64_t>{&pages,prompt}}; merge(result,owner->paged_->growth(requests)); }
            return result;
        }
        void initialize() override {
            if (initialized) throw std::logic_error("Qwen batch request already initialized");
            state.reset(1);
            if (owner->paged_) { owner->paged_->ensure_tokens(pages,prompt); owner->bind(state,{&pages}); }
            initialized=true;
        }
        Growth prefill_growth_bytes(std::int64_t tokens) const override {
            if (!initialized) throw std::logic_error("Qwen batch request is not initialized");
            if (slot>=0) return {};
            tyloi::models::ForwardOptions options; options.primary=owner->model_.primary_device;
            return state.growth(1,tokens,options,0);
        }
    };
    struct Graph {
        std::unique_ptr<DecodeWindow> window;
        std::int64_t length=0;
        std::vector<const void*> addresses;
        Tensor ids, positions, lengths, output;
        std::vector<Tensor> checkpoint;
        bool valid=false;
        explicit Graph(std::int64_t n,std::vector<const void*> a) : window(DecodeWindow::batch()),length(n),addresses(std::move(a)) {}
    };
    struct GraphPlan final : Plan {
        const Batch* owner;
        std::int64_t length;
        std::vector<const void*> addresses;
        std::shared_ptr<Graph> graph;
        GraphPlan(const Batch& value,std::int64_t n,std::vector<const void*> a,std::shared_ptr<Graph> g)
            : owner(&value),length(n),addresses(std::move(a)),graph(std::move(g)) { rebuild=!graph; }
    };
public:
    Batch(Model& model,const tyloi::models::BatchConfig& config) : model_(model),config_(config),state_(model,model.transposed) {
        if (!config.slots || config.slots>std::size_t(std::numeric_limits<int32_t>::max())) throw std::invalid_argument("continuous batching max sequences must be positive int32");
        if (model.layers.empty()) throw std::runtime_error("continuous batching requires at least one model block");
        if (config.cpu_offload) throw std::runtime_error("continuous batching requires GPU-resident model blocks");
        for (const auto& layer:model.layers) {
            if (layer.cpu) throw std::runtime_error("continuous batching cannot use CPU-offloaded blocks");
            std::visit([&](const auto& value) {
                using T=std::decay_t<decltype(value)>;
                if constexpr (std::is_same_v<T,Full>) {
                    if (value.attention.sliding) throw std::runtime_error("continuous batching requires non-sliding Qwen attention");
                    layouts_.push_back({layer.device,value.attention.kv_heads>0?value.attention.kv_heads:model.metadata.num_key_value_heads,
                        value.attention.attention_head_dim>0?value.attention.attention_head_dim:model.metadata.head_dim,value.attention.kv_quantization});
                }
                moe_ = moe_ || value.ffn.is_moe;
                const bool hit=cached(value.ffn); cached_=cached_ || hit;
                if (hit && value.ffn.moe_top_k>config.cached_slots) throw std::runtime_error("continuous batching requires one cached slot per routed expert");
            },layer.weights);
        }
        slots_.resize(config.slots,nullptr);
        if (config.paged) pages_.resize(config.slots);
    }
    ~Batch() override {
        try { clear(); } catch (...) {}
        for (auto* request:requests_) { request->slot=-1; request->owner=nullptr; }
    }
    Growth initial_growth_bytes() const override {
        return config_.paged ? PagedKv::initial_growth(layouts_,model_.metadata.max_position_embeddings,int32_t(config_.slots),config_.preallocate) : Growth{};
    }
    void initialize() override {
        if (initialized_) throw std::logic_error("Qwen batch already initialized");
        state_.reset(std::int64_t(slots_.size()));
        if (config_.paged) {
            paged_=std::make_unique<PagedKv>(layouts_,model_.metadata.max_position_embeddings,int32_t(slots_.size()),config_.preallocate);
            paged_->initialize();
        }
        initialized_=true;
    }
    std::unique_ptr<tyloi::models::BatchState> make_state(std::int64_t prompt) override {
        if (!initialized_ || prompt<=0 || prompt>model_.metadata.max_position_embeddings) throw std::invalid_argument("invalid Qwen batch prompt");
        auto request=std::make_unique<Request>(*this,prompt);
        requests_.push_back(request.get());
        return request;
    }
    std::size_t prefill_row_bytes() const override { return mul(mul(std::size_t(model_.metadata.hidden_size),std::size_t(model_.metadata.hc_mult)),sizeof(float)*8); }
    Growth activate_growth(const tyloi::models::BatchState& value) const override {
        const auto& request=checked(value);
        if (!request.initialized || request.slot>=0) throw std::logic_error("invalid Qwen batch activation");
        Growth result;
        for (std::size_t i=0;i<state_.layers.size();++i) {
            if (const auto* linear=std::get_if<LinearState>(&request.state.layers[i])) {
                const auto& target=std::get<LinearState>(state_.layers[i]);
                if (state_.batch!=std::int64_t(slots_.size()) || !target.attention.defined()) {
                    const auto source=linear->attention.view();
                    if (source.convolution.defined()) add(result,source.convolution.device(),mul(source.convolution.nbytes(),slots_.size()));
                    if (source.recurrent.defined()) add(result,source.recurrent.device(),mul(source.recurrent.nbytes(),slots_.size()));
                }
            } else if (!paged_) {
                const auto& source=std::get<tyloi::state::KvView>(request.state.layers[i]);
                const auto& target=std::get<tyloi::state::KvView>(state_.layers[i]);
                if (state_.batch!=std::int64_t(slots_.size()) || !target.defined()) {
                    if (source.physical_key.defined()) add(result,source.physical_key.device(),mul(source.physical_key.nbytes(),slots_.size()));
                    if (source.physical_value.defined()) add(result,source.physical_value.device(),mul(source.physical_value.nbytes(),slots_.size()));
                }
            }
        }
        if (paged_) {
            std::vector<std::pair<const PagedKv::Sequence*,int64_t>> needs;
            for (const auto& page:pages_) needs.emplace_back(&page,1);
            merge(result,paged_->growth(needs));
        }
        return result;
    }
    void activate(tyloi::models::BatchState& value) override {
        auto& request=checked(value);
        if (!request.initialized || request.slot>=0) throw std::logic_error("invalid Qwen batch activation");
        auto free=std::find(slots_.begin(),slots_.end(),nullptr);
        if (free==slots_.end()) throw std::runtime_error("continuous batching has no free stable slot");
        const auto slot=std::size_t(free-slots_.begin());
        if (state_.batch!=std::int64_t(slots_.size())) {
            if (active()) throw std::logic_error("continuous batching aggregate state has active slots at the wrong batch size");
            state_.reset(std::int64_t(slots_.size()));
        }
        if (paged_) bind_slots();
        copy_slot(request,slot);
        bool swapped=false;
        try {
            if (paged_) { paged_->release(pages_[slot]); std::swap(pages_[slot],request.pages); swapped=true; }
            request.active_position=request.state.position();
            slots_[slot]=&request; request.slot=int(slot);
            if (paged_) bind_slots();
            request.state=State(model_,model_.transposed);
        } catch (...) {
            if (request.slot==int(slot)) { slots_[slot]=nullptr; request.slot=-1; }
            if (swapped) { detach(state_); detach(request.state); std::swap(pages_[slot],request.pages); }
            throw;
        }
    }
    void release(tyloi::models::BatchState& value) override {
        auto& request=checked(value);
        if (request.slot>=0) {
            const auto slot=std::size_t(request.slot);
            if (slot>=slots_.size() || slots_[slot]!=&request) throw std::logic_error("continuous batching stable slot ownership changed");
            if (paged_) paged_->release(pages_[slot]);
            slots_[slot]=nullptr; request.slot=-1; ++releases_;
        }
        if (paged_) paged_->release(request.pages);
    }
    void resume(std::int64_t position) override { state_.cache_pos=position; if (paged_ && active()) bind_slots(); }
    void retire(std::int64_t position) override {
        state_.cache_pos=position;
        if (!active()) { clear(); state_.reset(1); if (paged_) { detach(state_); for (auto& p:pages_) paged_->release(p); } }
        else if (paged_) bind_slots();
    }
    void recover() override { clear(); paused_.clear(); for (auto* r:slots_) if (r) { r->slot=-1; } std::fill(slots_.begin(),slots_.end(),nullptr); if (paged_) for (auto& p:pages_) paged_->release(p); state_.reset(1); }
    Growth decode_growth(std::span<const tyloi::models::BatchRow> rows) const override {
        Growth result;
        add(result,Device(kCUDA,model_.primary_device),mul(slots_.size(),sizeof(int64_t)*3));
        if (paged_) { std::vector<std::pair<const PagedKv::Sequence*,int64_t>> needs;
            for (std::size_t i=0;i<slots_.size();++i) {
                int64_t count=1;
                for (const auto& row:rows) { const auto& r=checked(*row.state); if (r.slot==int(i)) count=std::max<int64_t>(count,row.position+1); }
                needs.emplace_back(&pages_[i],count);
            }
            merge(result,paged_->growth(needs));
        }
        std::size_t paused=0; for (const auto& row:rows) if (!row.eligible) ++paused;
        if (paused) for (const auto& layer:state_.layers) if (const auto* linear=std::get_if<LinearState>(&layer)) {
            const auto v=linear->attention.view();
            if (v.convolution.defined()) add(result,v.convolution.device(),mul(v.convolution.nbytes()/slots_.size(),paused));
            if (v.recurrent.defined()) add(result,v.recurrent.device(),mul(v.recurrent.nbytes()/slots_.size(),paused));
        }
        return result;
    }
    void prepare_decode(std::span<const tyloi::models::BatchRow> rows) override {
        int64_t position=0;
        for (const auto& row:rows) { auto& r=checked(*row.state); validate_slot(r); position=std::max(position,row.position); if (paged_) paged_->ensure_tokens(pages_[r.slot],row.position+1); }
        if (paged_) bind_slots(); state_.cache_pos=position;
    }
    tyloi::StepSequence<std::vector<Tensor>> prefill(std::vector<tyloi::models::BatchState*> states,std::vector<Tensor> ids) override {
        if (states.size()!=1 || ids.size()!=1) throw std::invalid_argument("Qwen batch prefill requires one request");
        auto& request=checked(*states[0]);
        if (!request.initialized || request.slot>=0) throw std::logic_error("invalid Qwen batch prefill state");
        if (paged_) bind(request.state,{&request.pages});
        tyloi::models::Input input; input.ids=ids[0];
        if (request.state.position()>0 && ids[0].size(1)==1)
            input.lengths=tyloi::ops::full(std::array{int64_t(1)},request.state.position()+1,Device(kCUDA,model_.primary_device));
        tyloi::models::ForwardOptions options; options.primary=model_.primary_device;
        auto hidden=qwen35::forward(model_,request.state,std::move(input),options);
        co_yield std::vector<Tensor>{std::move(hidden)};
    }
    Tensor decode(std::span<const tyloi::models::BatchRow> rows,Tensor,const Tensor&) override {
        ::tyloi::ops::FfnScope policy(::tyloi::ops::FfnScope::Role::batch,cached_,int64_t(slots_.size()),false);
        auto metadata=inputs(rows); ++metadata_batches_; save_paused(rows);
        try { auto output=compute(metadata.ids,metadata.positions,metadata.lengths,0,0,false); restore_paused(paused_); paused_.clear(); advance(rows); return select(output,rows); }
        catch (...) { auto error=std::current_exception(); try { clear(); } catch (...) {} std::rethrow_exception(error); }
    }
    tyloi::StepSequence<std::unique_ptr<Plan>> plan(std::span<const tyloi::models::BatchRow> rows,bool greedy) override {
        if (!config_.graph || !config_.greedy || !greedy || cached_ || slots_.size()<2 || !mfq_cuda_graph_capture_supported() || rows.empty()) co_return;
        int64_t requested=0, remaining=std::numeric_limits<int64_t>::max();
        for (const auto& row:rows) { validate_slot(checked(*row.state)); if (!row.greedy) co_return; requested=std::max(requested,row.position+row.remaining); remaining=std::min(remaining,row.remaining); }
        const auto length=decode_graph_bucket(requested,model_.metadata.max_position_embeddings);
        auto addresses=state_addresses();
        std::shared_ptr<Graph> found;
        for (auto& graph:graphs_) if (graph->valid && graph->length==length && graph->addresses==addresses) { found=graph; break; }
        if (!found && remaining<config_.minimum) co_return;
        co_yield tyloi::GraphPhase::prepare;
        if (!found) co_yield tyloi::GraphPhase::ready;
        auto result=std::make_unique<GraphPlan>(*this,length,std::move(addresses),std::move(found));
        if (result->rebuild) {
            const Device device(kCUDA,model_.primary_device);
            add(result->growth,device,mul(slots_.size(),sizeof(int64_t)*3));
            for (auto* tensor:state_.graph_state()) add(result->growth,tensor->device(),tensor->nbytes());
            add(result->growth,device,mul(slots_.size(),sizeof(int64_t)));
        }
        std::unique_ptr<Plan> output=std::move(result); co_yield std::move(output);
    }
    std::optional<Tensor> capture(std::span<const tyloi::models::BatchRow> rows,const Tensor&,const Tensor&,bool greedy,const Plan& raw) override {
        const auto* plan=dynamic_cast<const GraphPlan*>(&raw);
        if (!plan || plan->owner!=this || !greedy || plan->addresses!=state_addresses()) throw std::invalid_argument("Qwen batch graph plan changed");
        auto graph=plan->graph;
        if (!graph) {
            if (graphs_.size()>=8) clear();
            graph=std::make_shared<Graph>(plan->length,plan->addresses);
            graphs_.push_back(graph);
        }
        auto scope=graph->window->enter();
        if (!plan->graph) {
            graph->window->prepare();
            const Device device(kCUDA,model_.primary_device);
            graph->ids=tyloi::ops::zeros({int64_t(slots_.size()),1},TensorOptions().device(device).dtype(kInt64));
            graph->positions=tyloi::ops::zeros({int64_t(slots_.size()),1},TensorOptions().device(device).dtype(kInt64));
            graph->lengths=tyloi::ops::zeros({int64_t(slots_.size())},TensorOptions().device(device).dtype(kInt64));
        }
        auto metadata=inputs(rows); ++metadata_batches_;
        tyloi::ops::copy_into(graph->ids,metadata.ids); tyloi::ops::copy_into(graph->positions,metadata.positions); tyloi::ops::copy_into(graph->lengths,metadata.lengths);
        save_paused(rows);
        ::tyloi::ops::FfnScope policy(::tyloi::ops::FfnScope::Role::batch,cached_,int64_t(slots_.size()),true);
        if (!plan->graph) {
            auto members=state_.graph_state(); graph->checkpoint.reserve(members.size());
            for (auto* member:members) {
                graph->checkpoint.push_back(empty_like(*member));
                tyloi::ops::copy_into(graph->checkpoint.back(),*member);
            }
            auto restore=[&] { for (std::size_t i=0;i<members.size();++i) tyloi::ops::copy_into(*members[i],graph->checkpoint[i]); };
            auto driver=graph->window->capture();
            try { while (driver->next()) { graph->output=compute(graph->ids,graph->positions,graph->lengths,plan->length,tyloi::ops::attention_parts(plan->length,Device(kCUDA,model_.primary_device)),true);
                    driver->submitted(); restore(); if (driver->warmup()) graph->output={}; driver->restored(); } }
            catch (...) { auto error=driver->fail(std::current_exception()); restore(); std::rethrow_exception(error); }
            graph->addresses=state_addresses(); graph->valid=true; graph->checkpoint.clear(); ++captures_;
        }
        try { graph->window->run(); restore_paused(paused_); paused_.clear(); advance(rows); ++replays_; return select(graph->output,rows); }
        catch (...) { auto error=std::current_exception(); try { clear(); } catch (...) {} std::rethrow_exception(error); }
    }
    void clear() override {
        for (auto& graph:graphs_) { auto scope=graph->window->enter(false); graph->window->clear(); }
        graphs_.clear();
    }
    tyloi::engine::Metrics metrics() const override {
        const auto p=paged_?paged_->stats():PagedKv::Stats{};
        return {{"continuous_batching_compactions",0.0},{"continuous_batching_packed_metadata_batches",double(metadata_batches_)},
            {"continuous_batching_stable_slot_releases",double(releases_)},{"continuous_batching_cuda_graph_captures",double(captures_)},
            {"continuous_batching_cuda_graph_replays",double(replays_)},{"continuous_batching_moe",double(moe_)},
            {"continuous_batching_moe_cached_row_serial",double(cached_)},{"continuous_batching_paged_kv",double(bool(paged_))},
            {"paged_kv_page_size",double(p.page_size)},{"paged_kv_live_pages",double(p.live_pages)},
            {"paged_kv_peak_live_pages",double(p.peak_live_pages)},{"paged_kv_capacity_pages",double(p.capacity_pages)},
            {"paged_kv_reserved_bytes",double(p.reserved_bytes)},{"paged_kv_page_allocations",double(p.allocations)},
            {"paged_kv_page_reuses",double(p.reuses)},{"paged_kv_page_releases",double(p.releases)},{"paged_kv_table_updates",double(p.table_updates)}};
    }
private:
    struct Metadata { Tensor ids,positions,lengths; };
    struct Paused { int slot; std::vector<std::pair<Tensor,Tensor>> values; };
    Request& checked(tyloi::models::BatchState& v) const { auto* p=dynamic_cast<Request*>(&v); if (!p || p->owner!=this) throw std::invalid_argument("Qwen batch request belongs to another batch"); return *p; }
    const Request& checked(const tyloi::models::BatchState& v) const { return checked(const_cast<tyloi::models::BatchState&>(v)); }
    void validate_slot(const Request& r) const { if (r.slot<0 || std::size_t(r.slot)>=slots_.size() || slots_[r.slot]!=&r) throw std::logic_error("continuous batching request lost its stable slot"); }
    bool active() const { return std::any_of(slots_.begin(),slots_.end(),[](auto* v){return v;}); }
    void detach(State& state) { for (auto& layer:state.layers) if (auto* kv=std::get_if<tyloi::state::KvView>(&layer)) *kv={}; }
    void bind(State& state,const std::vector<const PagedKv::Sequence*>& pages) {
        paged_->update(pages); std::size_t page=0;
        for (auto& layer:state.layers) if (auto* kv=std::get_if<tyloi::state::KvView>(&layer)) *kv=paged_->view(page++,pages.size());
    }
    void bind_slots() { std::vector<const PagedKv::Sequence*> rows; for (auto& page:pages_) { paged_->ensure_tokens(page,1); rows.push_back(&page); } bind(state_,rows); }
    void copy_slot(Request& request,std::size_t slot) {
        if (request.state.batch!=1 || request.state.speculative_start>=0) throw std::logic_error("invalid Qwen batch source");
        for (std::size_t i=0;i<state_.layers.size();++i) {
            if (auto* target=std::get_if<LinearState>(&state_.layers[i])) {
                const auto& source=std::get<LinearState>(request.state.layers[i]); auto v=source.attention.view();
                if (!source.attention.defined() || source.attention.pending()) throw std::logic_error("invalid Qwen recurrent source");
                if (!target->attention.defined()) target->attention.initialize(v,slots_.size()); auto t=target->attention.view();
                if (v.transposed!=t.transposed) throw std::logic_error("continuous batching recurrent layout changed");
                tyloi::ops::copy_into(tyloi::ops::slice(t.convolution,0,slot,slot+1),v.convolution);
                tyloi::ops::copy_into(tyloi::ops::slice(t.recurrent,0,slot,slot+1),v.recurrent);
            } else if (!paged_) {
                auto& target=std::get<tyloi::state::KvView>(state_.layers[i]); const auto& source=std::get<tyloi::state::KvView>(request.state.layers[i]);
                if (!source.defined()) throw std::logic_error("invalid Qwen KV source");
                if (!target.defined()) { target=source; std::vector<int64_t> k,v; for(int j=0;j<source.physical_key.dim();++j) k.push_back(source.physical_key.size(j)); for(int j=0;j<source.physical_value.dim();++j) v.push_back(source.physical_value.size(j)); k[0]=v[0]=slots_.size(); target.physical_key=tyloi::ops::zeros(k,source.physical_key.options()); target.physical_value=tyloi::ops::zeros(v,source.physical_value.options()); target.batch=slots_.size(); }
                tyloi::ops::copy_into(tyloi::ops::slice(target.physical_key,0,slot,slot+1),source.physical_key);
                tyloi::ops::copy_into(tyloi::ops::slice(target.physical_value,0,slot,slot+1),source.physical_value);
            }
        }
    }
    Metadata inputs(std::span<const tyloi::models::BatchRow> rows) const {
        std::vector<int64_t> ids(slots_.size()),pos(slots_.size()),lengths(slots_.size(),1);
        for (const auto& row:rows) { const auto& request=checked(*row.state); validate_slot(request); ids[request.slot]=row.token; pos[request.slot]=row.position; lengths[request.slot]=row.position+1; }
        const Device device(kCUDA,model_.primary_device);
        return {tyloi::ops::reshape(tyloi::ops::tensor(ids,device),{int64_t(slots_.size()),1}),
            tyloi::ops::reshape(tyloi::ops::tensor(pos,device),{int64_t(slots_.size()),1}),tyloi::ops::tensor(lengths,device)};
    }
    static Tensor empty_like(const Tensor& source) {
        std::vector<int64_t> shape;
        for (int64_t axis=0;axis<source.dim();++axis) shape.push_back(source.size(axis));
        return tyloi::ops::empty(shape,source.options());
    }
    void save_paused(std::span<const tyloi::models::BatchRow> rows) {
        paused_.clear();
        paused_.reserve(rows.size());
        for (const auto& row:rows) if (!row.eligible) {
            paused_.push_back(Paused{checked(*row.state).slot,{}});
            auto& p=paused_.back();
            p.values.reserve(state_.layers.size());
            for (const auto& layer:state_.layers) if (const auto* linear=std::get_if<LinearState>(&layer)) {
                const auto v=linear->attention.view();
                auto conv=tyloi::ops::slice(v.convolution,0,p.slot,p.slot+1);
                auto recurrent=tyloi::ops::slice(v.recurrent,0,p.slot,p.slot+1);
                p.values.emplace_back(Tensor{},Tensor{});
                auto& targets=p.values.back();
                targets.first=empty_like(conv);
                targets.second=empty_like(recurrent);
                tyloi::ops::copy_into(targets.first,conv);
                tyloi::ops::copy_into(targets.second,recurrent);
            }
        }
    }
    void restore_paused(const std::vector<Paused>& rows) {
        for (const auto& row:rows) { std::size_t i=0; for (auto& layer:state_.layers) if (auto* linear=std::get_if<LinearState>(&layer)) {
            const auto v=linear->attention.view(); tyloi::ops::copy_into(tyloi::ops::slice(v.convolution,0,row.slot,row.slot+1),row.values[i].first);
            tyloi::ops::copy_into(tyloi::ops::slice(v.recurrent,0,row.slot,row.slot+1),row.values[i++].second);
        }}
    }
    Tensor compute(const Tensor& ids,const Tensor& pos,const Tensor& lengths,int64_t kv,int64_t parts,bool greedy) {
        tyloi::models::Input input; input.ids=ids; input.positions=pos; input.lengths=lengths; input.cache_positions=pos; input.kv_length=kv; input.attention_parts=parts;
        tyloi::models::ForwardOptions options; options.primary=model_.primary_device;
        auto hidden=qwen35::forward(model_,state_,std::move(input),options);
        auto last=tyloi::ops::contiguous(tyloi::ops::cast(tyloi::ops::select(hidden,1,-1),kFloat16));
        auto output=qwen35::logits(model_,last,true);
        if (greedy) output=tyloi::ops::sample_greedy(tyloi::ops::reshape(output,{int64_t(slots_.size()),-1}));
        return output;
    }
    void advance(std::span<const tyloi::models::BatchRow> rows) {
        state_.cache_pos=0;
        for (const auto& row:rows) {
            state_.cache_pos=std::max(state_.cache_pos,row.position+1);
            if (row.eligible) checked(*row.state).active_position=row.position+1;
        }
    }
    Tensor select(const Tensor& output,std::span<const tyloi::models::BatchRow> rows) const {
        std::vector<int64_t> indexes; for (const auto& row:rows) if (row.eligible) indexes.push_back(checked(*row.state).slot);
        return tyloi::ops::index_select(output,0,tyloi::ops::tensor(indexes,output.device()));
    }
    std::vector<const void*> state_addresses() const { std::vector<const void*> values; for (const auto& t:state_.storage()) if (t.defined()) values.push_back(t.dense().data_ptr()); return values; }
    Model& model_; tyloi::models::BatchConfig config_; State state_;
    std::vector<PagedKv::Layout> layouts_; std::vector<Request*> requests_,slots_; std::vector<PagedKv::Sequence> pages_; std::unique_ptr<PagedKv> paged_;
    std::vector<std::shared_ptr<Graph>> graphs_;
    std::vector<Paused> paused_;
    bool initialized_=false,moe_=false,cached_=false;
    int64_t releases_=0,captures_=0,replays_=0,metadata_batches_=0;
};
}
std::unique_ptr<tyloi::models::Batch> Model::batch(const tyloi::models::BatchConfig& config,tyloi::models::State* serial) {
    if (serial) { auto* state=dynamic_cast<State*>(serial); if (!state || !state->belongs_to(*this)) throw std::invalid_argument("Qwen batch serial state belongs to another model");
        if (config.preallocate) for (auto& layer:state->layers) if (auto* kv=std::get_if<tyloi::state::KvView>(&layer)) *kv={}; }
    return std::make_unique<Batch>(*this,config);
}
}
#endif // TYLOINFER_NATIVE_CUDA
