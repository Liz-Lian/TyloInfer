#include "engine/backends/cuda/executor/batch_graph.h"

#include "engine/backends/cuda/executor/decode_graph.h"

#include <utility>

namespace tyloi::cuda {

BatchGraph::BatchGraph() : stream(mfq_get_stream_from_pool(false)) {}

void BatchGraph::ensure_compute_streams(const Collectives& collectives) {
    if (compute_streams.empty()) {
        compute_streams = make_cuda_graph_compute_streams(stream, collectives);
    }
}

std::vector<MfqCudaStream>
BatchGraph::participant_streams(const tyloi::cuda::Collectives& collectives) const {
    return cuda_graph_participant_streams(compute_streams, collectives);
}

bool BatchGraph::matches(int64_t candidate_batch, int64_t candidate_len,
                         const std::vector<const void*>& candidate_addresses) const {
    return valid && batch == candidate_batch && planned_len == candidate_len &&
           state_addresses == candidate_addresses;
}

void BatchGraph::set_key(int64_t candidate_batch, int64_t candidate_len,
                         std::vector<const void*> candidate_addresses) {
    batch = candidate_batch;
    planned_len = candidate_len;
    state_addresses = std::move(candidate_addresses);
    valid = true;
}

void BatchGraph::invalidate() {
    if (graph)
        graph->reset();
    graph.reset();
    static_next = ::tyloi::DenseTensor();
    state_addresses.clear();
    valid = false;
}

} // namespace tyloi::cuda
