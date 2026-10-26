#include "engine/batch/qwen35.h"
#include "engine/executor/ffn.h"

#include "backends/cuda/engine/executor/execution.h"
#include "models/qwen35/linear_attention.h"
#include "models/qwen35/ops.h"
#include "engine/executor/options.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::cuda::qwen35 {

using Tensor = mfq_tensor_backend::Tensor;
using LinearBlock = LinearAttentionBlock;

Batch::~Batch() { detach_paged(); }

Batch::Batch(Qwen35CausalLm &model, std::size_t slots, bool paged)
    : model_(model) {
    if (slots == 0 || slots > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("continuous batching max sequences must be positive int32");
    const auto reason = incompatibility(*model.execution);
    if (!reason.empty())
        throw std::runtime_error(reason);
    slots_.resize(slots);
    if (paged) {
        paged_slots_.resize(slots);
        std::vector<PagedKv::Layout> layouts;
        const bool preallocate = runtime_options::kv_vram_preallocated();
        for (const auto& block : model_.blocks) {
            if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
                layouts.push_back({block->cuda_device,
                    full->kv_heads > 0 ? full->kv_heads : model_.num_key_value_heads(),
                    full->attention_head_dim > 0 ? full->attention_head_dim : model_.head_dim(),
                    full->kv_quantization});
                if (preallocate) full->cache = KVCache();
            }
        }
        paged_kv_ = std::make_unique<PagedKv>(layouts, model_.max_position_embeddings(),
            static_cast<int32_t>(slots), preallocate);
    }
}

bool Batch::paged_kv_enabled() const noexcept {
    return static_cast<bool>(paged_kv_);
}

PagedKv::Stats Batch::paged_kv_stats() const noexcept {
    return paged_kv_ ? paged_kv_->stats() : PagedKv::Stats{};
}

void Batch::suspend_decode() {
    TYLOINFER_RUNTIME_CHECK(!suspended_, "continuous batching decode state is already suspended");
    if (std::any_of(slots_.begin(), slots_.end(), [](auto *owner) { return owner != nullptr; }))
        suspended_ = take(static_cast<int64_t>(slots_.size()));
}

void Batch::prepare_prefill(RequestState &request, int64_t offset, int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(request.slot_ < 0 && offset >= 0 && offset < tokens,
                      "continuous batching prefill state is invalid");
    if (request.prefill_) {
        restore(*request.prefill_, offset);
        request.prefill_.reset();
    } else {
        TYLOINFER_RUNTIME_CHECK(offset == 0, "continuous batching lost prefill state");
        model_.reset(1);
        if (paged_kv_)
            paged_kv_->ensure_tokens(request.pages_, tokens);
    }
    if (paged_kv_)
        bind_paged({&request.pages_});
}

void Batch::pause_prefill(RequestState &request) {
    request.prefill_ = take(1);
}

void Batch::activate(RequestState &request) {
    TYLOINFER_RUNTIME_CHECK(request.slot_ < 0, "continuous batching request already has a slot");
    const auto free = std::find(slots_.begin(), slots_.end(), nullptr);
    TYLOINFER_RUNTIME_CHECK(free != slots_.end(), "continuous batching has no free stable slot");
    const auto slot = static_cast<int32_t>(free - slots_.begin());
    auto admitted = take(1);
    if (!suspended_)
        suspended_ = make_slot_state(admitted, slots_.size());
    copy_to_slot(*suspended_, admitted, slot);
    if (paged_kv_) {
        auto &target = paged_slots_[slot];
        paged_kv_->release(target);
        std::swap(target, request.pages_);
    }
    *free = &request;
    request.slot_ = slot;
}

void Batch::resume_decode(int64_t cache_position) {
    if (suspended_) {
        restore(*suspended_, cache_position);
        bind_paged_slots();
        suspended_.reset();
    } else {
        clear_model();
    }
}

void Batch::release(RequestState &request) {
    if (request.slot_ >= 0) {
        const auto slot = static_cast<std::size_t>(request.slot_);
        TYLOINFER_RUNTIME_CHECK(slot < slots_.size() && slots_[slot] == &request,
                          "continuous batching stable slot ownership changed");
        if (paged_kv_)
            paged_kv_->release(paged_slots_[slot]);
        slots_[slot] = nullptr;
        request.slot_ = -1;
        ++slot_releases_;
    }
    if (paged_kv_)
        paged_kv_->release(request.pages_);
    request.prefill_.reset();
}

void Batch::discard_prefill(RequestState &request) {
    mfq_cuda_synchronize();
    release(request);
    clear_model();
}

void Batch::clear_model() {
    model_.reset(1);
    detach_paged();
}

void Batch::finish_retire(int64_t cache_position) {
    if (std::none_of(slots_.begin(), slots_.end(), [](auto *owner) { return owner != nullptr; })) {
        clear_model();
        if (paged_kv_)
            for (auto &sequence : paged_slots_)
                paged_kv_->release(sequence);
    } else {
        model_.cache_pos = cache_position;
        bind_paged_slots();
    }
}

void Batch::ensure_decode_tokens(const RequestState &request, int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(request.slot_ >= 0 &&
                          static_cast<std::size_t>(request.slot_) < slots_.size() &&
                          slots_[request.slot_] == &request,
                      "continuous batching request lost its stable slot");
    if (paged_kv_)
        page_table_dirty_ =
            paged_kv_->ensure_tokens(paged_slots_[request.slot_], tokens) || page_table_dirty_;
}

void Batch::prepare_decode(int64_t cache_position) {
    if (page_table_dirty_)
        bind_paged_slots();
    model_.cache_pos = cache_position;
}

void Batch::finish_decode(int64_t cache_position) {
    model_.cache_pos = cache_position;
}

void Batch::recover(const std::vector<RequestState *> &requests) {
    try {
        mfq_cuda_synchronize();
    } catch (...) {
    }
    for (auto *request : requests)
        release(*request);
    suspended_.reset();
    finish_retire(0);
}

void Batch::bind_paged(const std::vector<const PagedKv::Sequence*>& sequences) {
    paged_kv_->update(sequences);
    size_t layer = 0;
    for (const auto& block : model_.blocks) {
        if (auto* full = dynamic_cast<FullBlock*>(block.get()))
            full->cache = paged_kv_->view(layer++, static_cast<int64_t>(sequences.size()));
    }
}

void Batch::detach_paged() noexcept {
    if (!paged_kv_) return;
    for (const auto& block : model_.blocks) {
        if (auto* full = dynamic_cast<FullBlock*>(block.get()))
            full->cache = KVCache();
    }
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
    bind_paged(sequences);
    page_table_dirty_ = false;
}

bool Batch::has_moe() const {
    for (const auto &block : model_.blocks) {
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->ffn.is_moe)
                return true;
        } else if (const auto *linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (linear->ffn.is_moe)
                return true;
        }
    }
    return false;
}

bool Batch::has_cached_moe() const { return has_cached_moe(model_); }

bool Batch::has_cached_moe(const Qwen35CausalLm &model) {
    for (const auto &block : model.blocks) {
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            if (mfq::cuda::uses_expert_cache(full->ffn))
                return true;
        } else if (const auto *linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (mfq::cuda::uses_expert_cache(linear->ffn))
                return true;
        }
    }
    return false;
}

std::string Batch::incompatibility(const CudaExecutionContext &execution) const {
    if (model_.blocks.empty()) {
        return "continuous batching requires at least one model block";
    }
    if (execution.dense_cpu_layer_count != 0 || !execution.dsv4_cpu_offload_layers.empty()) {
        return "continuous batching requires GPU-resident model blocks";
    }
    for (const auto &block : model_.blocks) {
        if (block->cpu_offloaded) {
            return "continuous batching cannot use CPU-offloaded blocks";
        }
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            if (full->sliding) {
                return "continuous batching requires non-sliding Qwen attention";
            }
            if (mfq::cuda::uses_expert_cache(full->ffn) &&
                full->ffn.moe_top_k > execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        if (const auto *linear = dynamic_cast<const LinearBlock *>(block.get())) {
            if (mfq::cuda::uses_expert_cache(linear->ffn) &&
                linear->ffn.moe_top_k > execution.moe_cache_registration_min_slots) {
                return "continuous batching requires one cached slot per routed expert";
            }
            continue;
        }
        return "continuous batching encountered an unsupported Qwen block";
    }
    return {};
}

BatchState Batch::take(int64_t batch) {
    TYLOINFER_RUNTIME_CHECK(model_.speculative_start < 0,
        "continuous batching cannot detach speculative state");
    BatchState state;
    state.batch = batch;
    state.layers.reserve(model_.blocks.size());
    for (const auto& block : model_.blocks) {
        if (const auto* full = dynamic_cast<const FullBlock*>(block.get())) {
            const auto& cache = full->cache;
            TYLOINFER_RUNTIME_CHECK(cache.defined() && cache.batch_size() == batch &&
                cache.is_paged() == bool(paged_kv_),
                "continuous batching KV state is unavailable");
            TYLOINFER_RUNTIME_CHECK(cache.is_paged() ||
                (cache.k.dim() == 4 && session_kv_geometry_matches(cache.k, cache.v)),
                "continuous batching full-attention state is unavailable");
            state.layers.emplace_back(cache);
        } else if (const auto* linear = dynamic_cast<const LinearBlock*>(block.get())) {
            const auto& mixer = linear->mixer;
            TYLOINFER_RUNTIME_CHECK(mixer.defined() && !mixer.speculative_pending &&
                mixer.convolution.size(0) == batch && mixer.recurrent.size(0) == batch,
                "continuous batching recurrent state is unavailable");
            state.layers.emplace_back(BatchState::Recurrent{
                mixer.convolution, mixer.recurrent, linear->transposed_gdn_state});
        } else {
            throw std::runtime_error("continuous batching encountered an unsupported block state");
        }
    }
    // Detach only after every layer has supplied a valid snapshot.
    for (const auto& block : model_.blocks) {
        MfqCudaGuard guard(block->cuda_device);
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            full->cache = KVCache();
        } else {
            auto& linear = *static_cast<LinearBlock*>(block.get());
            linear.mixer.convolution = {};
            linear.mixer.recurrent = {};
            linear.clear_speculative();
        }
    }
    model_.cache_pos = 0;
    return state;
}

void Batch::restore(const BatchState& state, int64_t cache_position) {
    TYLOINFER_RUNTIME_CHECK(state.batch > 0 && state.layers.size() == model_.blocks.size(),
        "continuous batching state layout changed");
    for (size_t index = 0; index < model_.blocks.size(); ++index) {
        const auto& block = model_.blocks[index];
        const auto& saved = state.layers[index];
        if (dynamic_cast<const FullBlock*>(block.get())) {
            const auto* cache = std::get_if<KVCache>(&saved);
            TYLOINFER_RUNTIME_CHECK(cache && cache->defined() &&
                cache->is_paged() == bool(paged_kv_) && cache->batch_size() == state.batch,
                "continuous batching full-attention state changed");
            TYLOINFER_RUNTIME_CHECK(cache->is_paged() ||
                (!cache->ring && cache->k.dim() == 4 &&
                 session_kv_geometry_matches(cache->k, cache->v)),
                "continuous batching KV cache geometry changed");
        } else if (dynamic_cast<const LinearBlock*>(block.get())) {
            const auto* recurrent = std::get_if<BatchState::Recurrent>(&saved);
            TYLOINFER_RUNTIME_CHECK(recurrent && recurrent->convolution.defined() &&
                recurrent->recurrent.defined() && recurrent->convolution.size(0) == state.batch &&
                recurrent->recurrent.size(0) == state.batch,
                "continuous batching recurrent state changed");
        } else {
            throw std::runtime_error("continuous batching restore encountered an unsupported block");
        }
    }
    for (size_t index = 0; index < model_.blocks.size(); ++index) {
        auto& block = model_.blocks[index];
        const auto& saved = state.layers[index];
        MfqCudaGuard guard(block->cuda_device);
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            full->cache = std::get<KVCache>(saved);
        } else {
            auto& linear = *static_cast<LinearBlock*>(block.get());
            const auto& recurrent = std::get<BatchState::Recurrent>(saved);
            linear.mixer.convolution = recurrent.convolution;
            linear.mixer.recurrent = recurrent.recurrent;
            linear.transposed_gdn_state = recurrent.transposed;
            linear.clear_speculative();
        }
    }
    model_.cache_pos = cache_position;
    model_.speculative_start = -1;
    model_.speculative_confirmed = 0;
}

BatchState Batch::make_slot_state(const BatchState& source, int64_t slots) const {
    TYLOINFER_RUNTIME_CHECK(source.batch == 1 && slots > 0,
        "continuous batching slot state requires one source row");
    const auto expand = [slots](const Tensor& value) {
        auto shape = value.sizes().vec();
        shape[0] = slots;
        return mfq_tensor_backend::zeros(shape, value.options());
    };
    BatchState result;
    result.batch = slots;
    result.layers.reserve(source.layers.size());
    for (const auto& saved : source.layers) {
        if (const auto* cache = std::get_if<KVCache>(&saved)) {
            auto target = *cache;
            target.paged_batch = slots;
            if (!cache->is_paged()) {
                target.k = expand(cache->k);
                target.v = expand(cache->v);
            }
            result.layers.emplace_back(std::move(target));
        } else {
            const auto& recurrent = std::get<BatchState::Recurrent>(saved);
            result.layers.emplace_back(BatchState::Recurrent{
                expand(recurrent.convolution), expand(recurrent.recurrent), recurrent.transposed});
        }
    }
    return result;
}

void Batch::copy_to_slot(BatchState& slots, const BatchState& source, int64_t slot) const {
    TYLOINFER_RUNTIME_CHECK(source.batch == 1 && slot >= 0 && slot < slots.batch &&
        slots.layers.size() == source.layers.size(),
        "continuous batching received an invalid stable slot");
    for (size_t index = 0; index < slots.layers.size(); ++index) {
        auto& target = slots.layers[index];
        const auto& saved = source.layers[index];
        TYLOINFER_RUNTIME_CHECK(target.index() == saved.index(),
            "continuous batching stable slot layout changed");
        if (auto* cache = std::get_if<KVCache>(&target)) {
            const auto& from = std::get<KVCache>(saved);
            TYLOINFER_RUNTIME_CHECK(cache->is_paged() == from.is_paged(),
                "continuous batching stable slot KV layout changed");
            if (!cache->is_paged()) {
                cache->k.narrow(0, slot, 1).copy_(from.k);
                cache->v.narrow(0, slot, 1).copy_(from.v);
            }
        } else {
            auto& recurrent = std::get<BatchState::Recurrent>(target);
            const auto& from = std::get<BatchState::Recurrent>(saved);
            TYLOINFER_RUNTIME_CHECK(recurrent.transposed == from.transposed,
                "continuous batching recurrent layout changed");
            recurrent.convolution.narrow(0, slot, 1).copy_(from.convolution);
            recurrent.recurrent.narrow(0, slot, 1).copy_(from.recurrent);
        }
    }
}

Tensor Batch::logits_from_last_hidden(Tensor hidden) {
    auto last = hidden.index({Slice(), -1, Slice()}).to(mfq_tensor_backend::kFloat16).contiguous();
    return model_.apply_final_logit_softcap(model_.execution->projector().run(model_.lm_head, last));
}

std::vector<const void *> Batch::decode_state_addresses() {
    std::vector<const void *> addresses;
    addresses.reserve(2 * model_.blocks.size());
    for (auto &block : model_.blocks) {
        if (auto *full = dynamic_cast<FullBlock *>(block.get())) {
            if (full->cache.is_paged()) {
                addresses.push_back(full->cache.k_chunk_ptrs.data_ptr());
                addresses.push_back(full->cache.v_chunk_ptrs.data_ptr());
                addresses.push_back(full->cache.page_table.data_ptr());
            } else {
                addresses.push_back(full->cache.k.data_ptr());
                addresses.push_back(full->cache.v.data_ptr());
            }
        } else if (auto *linear = dynamic_cast<LinearBlock *>(block.get())) {
            addresses.push_back(linear->mixer.convolution.data_ptr());
            addresses.push_back(linear->mixer.recurrent.data_ptr());
        }
    }
    return addresses;
}

BatchState Batch::capture_recurrent_slots(const std::vector<std::int32_t>& slots) const {
    BatchState state;
    state.batch = static_cast<int64_t>(slots.size());
    if (slots.empty()) return state;
    state.layers.resize(model_.blocks.size());
    for (size_t index = 0; index < model_.blocks.size(); ++index) {
        const auto* linear = dynamic_cast<const LinearBlock*>(model_.blocks[index].get());
        if (!linear) continue;
        MfqCudaGuard guard(linear->cuda_device);
        std::vector<Tensor> conv, gdn;
        for (auto slot : slots) {
            conv.push_back(linear->mixer.convolution.narrow(0, slot, 1));
            gdn.push_back(linear->mixer.recurrent.narrow(0, slot, 1));
        }
        state.layers[index] = BatchState::Recurrent{
            mfq_tensor_backend::cat(conv, 0).clone(), mfq_tensor_backend::cat(gdn, 0).clone(),
            linear->transposed_gdn_state};
    }
    return state;
}

void Batch::restore_recurrent_slots(const std::vector<std::int32_t>& slots,
                                    const BatchState& state) {
    if (slots.empty()) return;
    TYLOINFER_RUNTIME_CHECK(state.batch == static_cast<int64_t>(slots.size()) &&
        state.layers.size() == model_.blocks.size(),
        "continuous batching recurrent snapshot layout changed");
    for (size_t index = 0; index < model_.blocks.size(); ++index) {
        auto* linear = dynamic_cast<LinearBlock*>(model_.blocks[index].get());
        if (!linear) continue;
        const auto* saved = std::get_if<BatchState::Recurrent>(&state.layers[index]);
        TYLOINFER_RUNTIME_CHECK(saved && saved->convolution.defined() && saved->recurrent.defined() &&
            saved->convolution.size(0) == state.batch && saved->recurrent.size(0) == state.batch &&
            saved->transposed == linear->transposed_gdn_state,
            "continuous batching recurrent snapshot changed");
    }
    for (size_t index = 0; index < model_.blocks.size(); ++index) {
        auto* linear = dynamic_cast<LinearBlock*>(model_.blocks[index].get());
        if (!linear) continue;
        MfqCudaGuard guard(linear->cuda_device);
        const auto& saved = std::get<BatchState::Recurrent>(state.layers[index]);
        for (size_t row = 0; row < slots.size(); ++row) {
            linear->mixer.convolution.narrow(0, slots[row], 1)
                .copy_(saved.convolution.narrow(0, row, 1));
            linear->mixer.recurrent.narrow(0, slots[row], 1)
                .copy_(saved.recurrent.narrow(0, row, 1));
        }
    }
}

} // namespace mfq::cuda::qwen35
