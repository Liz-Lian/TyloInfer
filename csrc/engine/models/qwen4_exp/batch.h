#pragma once

#include "engine/models/batch.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "engine/models/qwen4_exp/ops.h"
#include "engine/config.h"
#include "engine/generation/batch.h"
#include "engine/backends/cuda/executor/sampling.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::engine { class Budget; }
namespace tyloi::cuda::qwen4_exp {

#ifndef TYLOINFER_NATIVE_CUDA
struct BatchRequest : tyloi::engine::BatchRequest {
    using tyloi::engine::BatchRequest::BatchRequest;

    std::optional<Sampler> sampler;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor counts;
#else
    ::tyloi::DenseTensor counts;
#endif
    ::tyloi::DenseTensor prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    std::unique_ptr<Qwen4BatchRequestState> cache;
    int64_t decode_row = -1;
    double packed_prefill_ms = 0;
};

class BatchOps {
  public:
    using DenseTensor = ::tyloi::DenseTensor;
    using Request = BatchRequest;
    using State = tyloi::engine::BatchState<Request>;
    using Sample = tyloi::models::BatchSample;
    struct Decoded {
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor logits, greedy_tokens;
#else
        DenseTensor logits, greedy_tokens;
#endif
    };

    BatchOps(qwen4_exp::Model&, CudaExecutionContext&, tyloi::engine::Budget&,
             tyloi::engine::ContinuousBatchConfig);
    ~BatchOps();
    int64_t vocab_size() const;
    int64_t max_context() const;
    tyloi::engine::Metrics metrics() const;
    void suspend_decode();
    tyloi::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&,
                                            tyloi::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request>&, tyloi::engine::PrefillChunk);
    std::size_t prefill_batch_limit() const;
    int64_t prefill_batch_token_budget() const;
    std::vector<tyloi::engine::PrefillBatchResult<Sample>> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>, tyloi::engine::PrefillChunk>>&);
    void activate(const std::shared_ptr<Request>&, const Sample&);
    void resume_decode(int64_t);
    void discard_prefill(const std::shared_ptr<Request>&);
    void retire(const std::vector<std::shared_ptr<Request>>&, int64_t);
    Decoded decode(State&);
    Sample sample(const std::shared_ptr<Request>&, const Decoded&);
    void accept(const std::shared_ptr<Request>&, const Sample&);
    void recover(State&);
    static constexpr bool isolate_prefill_failures = true;
    static constexpr bool isolate_context_growth_failures = true;

  private:
    void initialize_prefill(const std::shared_ptr<Request>&, tyloi::engine::PrefillChunk);
#ifdef TYLOINFER_NATIVE_CUDA
    Sample finish_prefill(const std::shared_ptr<Request>&, ::tyloi::Tensor next);
#else
    Sample finish_prefill(const std::shared_ptr<Request>&, DenseTensor next);
#endif

    qwen4_exp::Model& model_;
    CudaExecutionContext& execution_;
    tyloi::engine::Budget& budget_; // The owner keeps the budget alive until batch destruction.
    tyloi::engine::ContinuousBatchConfig config_;
    std::vector<std::weak_ptr<Request>> requests_;
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
    int64_t decode_batches_ = 0, decode_tokens_ = 0, greedy_batches_ = 0;
    int64_t packed_prefill_batches_ = 0, packed_prefill_requests_ = 0,
            packed_prefill_tokens_ = 0;
};
#endif

} // namespace tyloi::cuda::qwen4_exp

namespace tyloi::cuda {

#ifndef TYLOINFER_NATIVE_CUDA
std::unique_ptr<tyloi::engine::Batch> make_batch(
    qwen4_exp::Model& model, CudaExecutionContext& execution, tyloi::engine::Budget& budget,
    const tyloi::engine::ContinuousBatchConfig& config, std::int64_t chunk_size);
#endif

} // namespace tyloi::cuda
