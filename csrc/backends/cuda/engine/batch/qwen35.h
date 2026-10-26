#pragma once

#include "engine/state/paged_kv.h"
#include "engine/state/recurrent.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

struct CudaExecutionContext;

namespace mfq::models::qwen35 {
template <class Backend> struct CausalLm;
}

namespace mfq::cuda {
struct Qwen35Model;
template <typename Model> struct CudaCausalOps;
using Qwen35CausalLm = mfq::models::qwen35::CausalLm<CudaCausalOps<Qwen35Model>>;
} // namespace mfq::cuda

namespace mfq::cuda::qwen35 {

struct BatchState {
    std::int64_t batch = 0;
    std::vector<std::variant<KVCache, Recurrent::State>> layers;
};

// A request owns its native cache state; the batch alone binds it to slots.
// Requests have stable addresses while admitted and release before retirement.
class RequestState {
  public:
    RequestState() = default;
    RequestState(const RequestState &) = delete;
    RequestState &operator=(const RequestState &) = delete;
    std::int32_t slot() const noexcept { return slot_; }

  private:
    friend class Batch;
    std::optional<BatchState> prefill_;
    PagedKv::Sequence pages_;
    std::int32_t slot_ = -1;
};

class Batch {
  public:
    Batch(Qwen35CausalLm &model, std::size_t slots, bool paged);
    Batch(const Batch &) = delete;
    Batch &operator=(const Batch &) = delete;
    ~Batch();

    bool has_moe() const;
    bool has_cached_moe() const;
    static bool has_cached_moe(const Qwen35CausalLm &model);
    std::string incompatibility(const CudaExecutionContext &execution) const;
    bool paged_kv_enabled() const noexcept;
    PagedKv::Stats paged_kv_stats() const noexcept;
    std::int64_t slot_releases() const noexcept { return slot_releases_; }

    void suspend_decode();
    void prepare_prefill(RequestState &request, std::int64_t offset, std::int64_t tokens);
    void pause_prefill(RequestState &request);
    void activate(RequestState &request);
    void resume_decode(std::int64_t cache_position);
    void discard_prefill(RequestState &request);
    void release(RequestState &request);
    void finish_retire(std::int64_t cache_position);
    void ensure_decode_tokens(const RequestState &request, std::int64_t tokens);
    void prepare_decode(std::int64_t cache_position);
    void finish_decode(std::int64_t cache_position);
    void recover(const std::vector<RequestState *> &requests);

    mfq_tensor_backend::Tensor logits_from_last_hidden(mfq_tensor_backend::Tensor hidden);
    std::vector<const void *> decode_state_addresses();
    BatchState capture_recurrent_slots(const std::vector<std::int32_t> &slots) const;
    void restore_recurrent_slots(const std::vector<std::int32_t> &slots,
                                 const BatchState &state);

  private:
    BatchState take(std::int64_t batch);
    void restore(const BatchState &state, std::int64_t cache_position);
    BatchState make_slot_state(const BatchState &source, std::int64_t slots) const;
    void copy_to_slot(BatchState &slots, const BatchState &source, std::int64_t slot) const;
    void bind_paged(const std::vector<const PagedKv::Sequence*>& sequences);
    void bind_paged_slots();
    void detach_paged() noexcept;
    void clear_model();

    Qwen35CausalLm &model_;
    std::vector<RequestState *> slots_;
    std::optional<BatchState> suspended_;
    std::int64_t slot_releases_ = 0;
    bool page_table_dirty_ = false;
    std::unique_ptr<PagedKv> paged_kv_;
    std::vector<PagedKv::Sequence> paged_slots_;
};

} // namespace mfq::cuda::qwen35
