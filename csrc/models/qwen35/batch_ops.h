#pragma once

#include "backends/cuda/engine/batch/batch.h"
#include "cuda_runtime_config.h"
#include "engine/batch/batch.h"
#include "engine/executor/sampler.h"
#include "models/qwen35/batch.h"
#include "native/cuda/profile.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mfq::cuda::qwen35 {

bool batch_graph_enabled(const Model&, const CudaContinuousBatchConfig&);

struct BatchRequest : mfq::engine::BatchRequest {
    using mfq::engine::BatchRequest::BatchRequest;

    std::optional<Sampler> sampler;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor prefill_ids;
    std::vector<std::unique_ptr<internal::PrefillCudaTimer>> prefill_timers;
    RequestState cache;
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

    // Calls run inside the owner's CUDA device/context scope.
    BatchOps(std::shared_ptr<const Model>, std::unique_ptr<Executor>, Collectives&,
             Executor::Mode, Batch::Options, std::shared_ptr<MoeExpertCache>,
             CudaContinuousBatchConfig);
    ~BatchOps();
    BatchOps(const BatchOps&) = delete;
    BatchOps& operator=(const BatchOps&) = delete;
    int64_t vocab_size() const;
    int64_t max_context() const;
    mfq::engine::Metrics metrics() const;
    void suspend_decode();
    mfq::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&,
                                            mfq::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request>&, mfq::engine::PrefillChunk);
    void activate(const std::shared_ptr<Request>&, const Sample&);
    void resume_decode(int64_t cache_position);
    void discard_prefill(const std::shared_ptr<Request>&);
    void retire(const std::vector<std::shared_ptr<Request>>&, int64_t cache_position);
    Decoded decode(State&);
    Sample sample(const std::shared_ptr<Request>&, const Decoded&);
    void accept(const std::shared_ptr<Request>&, const Sample&);
    void recover(State&);

  private:
    void initialize_sampling(Request&, const Tensor& prompt_ids);
    void ensure_decode_metadata_buffers(int primary);
    void invalidate_decode_graph();

    std::shared_ptr<const Model> model_;
    Collectives& collectives_;
    const Executor::Mode mode_;
    Batch state_;
    std::shared_ptr<MoeExpertCache> moe_cache_;
    const CudaContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    int64_t decode_batches_{0};
    int64_t decode_tokens_{0};
    int64_t batched_greedy_batches_{0};
    int64_t packed_metadata_batches_{0};
    int64_t cuda_graph_captures_{0};
    int64_t cuda_graph_replays_{0};
    Tensor decode_metadata_host_;
    Tensor decode_metadata_cuda_;
    std::vector<std::unique_ptr<mfq::cuda::BatchGraph>> decode_graphs_;
};

} // namespace mfq::cuda::qwen35
