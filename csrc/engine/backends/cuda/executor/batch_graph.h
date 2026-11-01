#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace mfq::cuda {

class Collectives;

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
