#pragma once

#include "backends/cuda/engine/batch/batch.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/models/qwen4_exp/ops.h"
#include "cuda_runtime_config.h"
#include "engine/batch/batch.h"
#include "engine/executor/sampler.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::qwen4_exp {

struct BatchRequest : mfq::engine::BatchRequest {
    using mfq::engine::BatchRequest::BatchRequest;

    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor counts, prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    std::unique_ptr<Qwen4BatchRequestState> cache;
    int64_t decode_row = -1;
    double packed_prefill_ms = 0;
};

class BatchOps {
  public:
    using Tensor = mfq_tensor_backend::Tensor;
    using Request = BatchRequest;
    using State = mfq::engine::BatchState<Request>;
    using Sample = mfq::cuda::BatchSample;
    struct Decoded {
        Tensor logits, greedy_tokens;
    };

    BatchOps(Qwen4CausalLm&, CudaExecutionContext&, CudaContinuousBatchConfig);
    ~BatchOps();
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    void suspend_decode();
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&,
                                            mfq::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    std::size_t prefill_batch_limit() const;
    int64_t prefill_batch_token_budget() const;
    std::vector<mfq::engine::PrefillBatchResult<Sample>> prefill_batch(
        const std::vector<std::pair<std::shared_ptr<Request>, mfq::engine::PrefillChunk>>&);
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
    void initialize_prefill(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    Sample finish_prefill(const std::shared_ptr<Request>&, Tensor next);

    Qwen4CausalLm& model_;
    CudaExecutionContext& execution_;
    CudaContinuousBatchConfig config_;
    std::vector<std::weak_ptr<Request>> requests_;
    std::unique_ptr<Qwen4BatchDecodeGraph> graph_;
    int64_t decode_batches_ = 0, decode_tokens_ = 0, greedy_batches_ = 0;
    int64_t packed_prefill_batches_ = 0, packed_prefill_requests_ = 0,
            packed_prefill_tokens_ = 0;
};

} // namespace mfq::cuda::qwen4_exp
