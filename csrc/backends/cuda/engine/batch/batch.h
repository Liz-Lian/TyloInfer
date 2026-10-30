#pragma once

#include "backends/cuda/engine/executor/decode_graph.h"
#include "engine/batch/batch.h"
#include "engine/requests/result.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

class MoeExpertCache;

namespace mfq::cuda {

class Collectives;

struct BatchSample {
    mfq_tensor_backend::Tensor next;
    int64_t token = 0;
    MfqPrefillTiming timing;
};

mfq::engine::PrefillChunk budget_batch_prefill(
        std::int64_t hidden_size, std::int64_t hc_mult, int device,
        const std::shared_ptr<MoeExpertCache>&,
        mfq::engine::PrefillChunk, std::size_t growth = 0);

class BatchCacheScope {
  public:
    BatchCacheScope(bool& serial, bool enabled);
    ~BatchCacheScope();

    BatchCacheScope(const BatchCacheScope&) = delete;
    BatchCacheScope& operator=(const BatchCacheScope&) = delete;

  private:
    bool& serial_;
    bool previous_ = false;
};

struct BatchGraph {
    decltype(mfq_get_stream_from_pool(false)) stream;
    std::vector<MfqCudaStream> compute_streams;
    std::unique_ptr<MfqCudaGraph> graph;
    mfq_tensor_backend::Tensor static_next;
    std::vector<const void*> state_addresses;
    int64_t batch = 0;
    int64_t planned_len = 0;
    bool valid = false;

    BatchGraph();

    void ensure_compute_streams(const Collectives& collectives);

    std::vector<MfqCudaStream>
    participant_streams(const mfq::cuda::Collectives& collectives) const;

    bool matches(int64_t candidate_batch, int64_t candidate_len,
                 const std::vector<const void*>& candidate_addresses) const;

    void set_key(int64_t candidate_batch, int64_t candidate_len,
                 std::vector<const void*> candidate_addresses);

    void invalidate();
};

} // namespace mfq::cuda
