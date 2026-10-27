#pragma once

#include "models/qwen35/forward.h"
#include "engine/state/paged_kv.h"

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

struct CudaExecutionContext;

namespace mfq::cuda::qwen35 {

// A copyable snapshot is only needed for paused recurrent decode rows. KV state
// remains owned by the aggregate state and is copied into its stable slot in place.
struct BatchState {
    std::int64_t batch = 0;
    std::vector<std::variant<KVCache, Recurrent::State>> layers;
};

// A request owns its prefill state and executor. The batch only copies state
// into a stable decode slot; immutable weights are shared by every request.
class RequestState {
  public:
    RequestState() = default;
    RequestState(const RequestState &) = delete;
    RequestState &operator=(const RequestState &) = delete;
    std::int32_t slot() const noexcept { return slot_; }
    qwen35::State &state() noexcept { return *state_; }
    const qwen35::State &state() const noexcept { return *state_; }
    Executor &executor() noexcept { return *executor_; }
    const Executor &executor() const noexcept { return *executor_; }

  private:
    friend class Batch;
    std::unique_ptr<Executor> executor_;
    std::unique_ptr<qwen35::State> state_;
    PagedKv::Sequence pages_;
    std::int32_t slot_ = -1;
};

class Batch {
  public:
    Batch(std::shared_ptr<const Model> weights, CudaExecutionContext &execution,
          std::size_t slots, bool paged);
    Batch(const Batch &) = delete;
    Batch &operator=(const Batch &) = delete;
    ~Batch();

    bool has_moe() const;
    bool has_cached_moe() const;
    static bool has_cached_moe(const Model &model);
    std::string incompatibility(const CudaExecutionContext &execution) const;
    bool paged_kv_enabled() const noexcept;
    PagedKv::Stats paged_kv_stats() const noexcept;
    std::int64_t slot_releases() const noexcept { return slot_releases_; }

    void prepare_prefill(RequestState &request, std::int64_t offset, std::int64_t tokens);
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

    qwen35::State &state() noexcept { return *state_; }
    const qwen35::State &state() const noexcept { return *state_; }
    Executor &executor() noexcept { return *executor_; }
    const Executor &executor() const noexcept { return *executor_; }

  private:
    void copy_to_slot(qwen35::State &target, const qwen35::State &source,
                      std::int64_t slot) const;
    void bind_paged(qwen35::State &state,
                    const std::vector<const PagedKv::Sequence *> &sequences);
    void bind_paged_slots();
    void detach_paged(qwen35::State &state) noexcept;

    std::shared_ptr<const Model> weights_;
    CudaExecutionContext *execution_ = nullptr;
    std::unique_ptr<Executor> executor_;
    std::unique_ptr<qwen35::State> state_;
    std::vector<RequestState *> slots_;
    std::int64_t slot_releases_ = 0;
    bool page_table_dirty_ = false;
    std::unique_ptr<PagedKv> paged_kv_;
    std::vector<PagedKv::Sequence> paged_slots_;
};

} // namespace mfq::cuda::qwen35
