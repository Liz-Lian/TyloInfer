#pragma once

#include "engine/models/batch.h"
#include "engine/backends/cuda/executor/batch_graph.h"
#include "engine/config.h"
#include "engine/generation/batch.h"
#include "engine/backends/cuda/executor/sampling.h"
#include "engine/models/qwen35/batch.h"
#include "engine/backends/cuda/native/profile.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::engine { class Budget; }
namespace tyloi::cuda::qwen35 {

bool batch_graph_enabled(const Model&, const tyloi::engine::ContinuousBatchConfig&);

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
    RequestState cache;
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

    // Calls run inside the owner's CUDA device/context scope.
    BatchOps(std::shared_ptr<const Model>, std::unique_ptr<Executor>, Collectives&,
             Executor::Mode, Batch::Options, tyloi::engine::Budget&,
             tyloi::engine::ContinuousBatchConfig);
    ~BatchOps();
    BatchOps(const BatchOps&) = delete;
    BatchOps& operator=(const BatchOps&) = delete;
    int64_t vocab_size() const;
    int64_t max_context() const;
    tyloi::engine::Metrics metrics() const;
    void suspend_decode();
    tyloi::engine::PrefillChunk prefill_chunk(const std::shared_ptr<Request>&,
                                            tyloi::engine::PrefillChunk);
    Sample prefill(const std::shared_ptr<Request>&, tyloi::engine::PrefillChunk);
    void activate(const std::shared_ptr<Request>&, const Sample&);
    void resume_decode(int64_t cache_position);
    void discard_prefill(const std::shared_ptr<Request>&);
    void retire(const std::vector<std::shared_ptr<Request>>&, int64_t cache_position);
    Decoded decode(State&);
    Sample sample(const std::shared_ptr<Request>&, const Decoded&);
    void accept(const std::shared_ptr<Request>&, const Sample&);
    void recover(State&);

  private:
    void initialize_sampling(Request&, const DenseTensor& prompt_ids);
    void ensure_decode_metadata_buffers(int primary);
    void invalidate_decode_graph();

    std::shared_ptr<const Model> model_;
    Collectives& collectives_;
    const Executor::Mode mode_;
    Batch state_;
    tyloi::engine::Budget& budget_; // The owner keeps the budget alive until batch destruction.
    const tyloi::engine::ContinuousBatchConfig config_;
    int32_t max_sequences_ = 0;
    bool moe_enabled_ = false;
    bool cached_moe_enabled_ = false;
    int64_t decode_batches_{0};
    int64_t decode_tokens_{0};
    int64_t batched_greedy_batches_{0};
    int64_t packed_metadata_batches_{0};
    int64_t cuda_graph_captures_{0};
    int64_t cuda_graph_replays_{0};
    DenseTensor decode_metadata_host_;
    DenseTensor decode_metadata_cuda_;
    std::vector<std::unique_ptr<tyloi::cuda::BatchGraph>> decode_graphs_;
};

} // namespace tyloi::cuda::qwen35
