#include "backends/cuda/engine/batch/batch.h"

#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "engine/batch/prefill_budget.h"
#include "native/error.h"

#include <utility>

namespace mfq::cuda {

mfq::engine::PrefillChunk budget_batch_prefill(
        std::int64_t hidden_size, std::int64_t hc_mult, CudaExecutionContext& execution,
        mfq::engine::PrefillChunk chunk, std::size_t growth) {
    const auto context=default_context(execution.layer_placement.primary_device());
    const auto transfer=moe_prefill_workspace_budget(execution.stores.moe_expert_cache,false).transfer_bytes;
    const auto fixed=mfq::engine::prefill_budget_add((std::size_t(1)<<30)+growth,transfer);
    const auto available=context->available_memory()+context->workspace_available();
    const auto row_bytes=std::size_t(hidden_size)*hc_mult*sizeof(float)*8;
    const auto selected=mfq::engine::select_prefill_chunk(chunk.count,chunk.count,available,fixed,row_bytes);
    if(!selected)throw mfq::engine::InferenceInputError(
        mfq::engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for batched prefill; reduce concurrent requests");
    chunk.count=selected;
    return chunk;
}

BatchCacheScope::BatchCacheScope(bool& serial, bool enabled)
    : serial_(serial), previous_(serial) {
    serial_ = enabled;
}

BatchCacheScope::~BatchCacheScope() { serial_ = previous_; }

BatchGraph::BatchGraph() : stream(mfq_get_stream_from_pool(false)) {}

void BatchGraph::ensure_compute_streams(const ParallelConfig& parallel) {
    if (compute_streams.empty()) {
        compute_streams = make_cuda_graph_compute_streams(stream, parallel);
    }
}

std::vector<MfqCudaStream>
BatchGraph::participant_streams(const mfq::cuda::Collectives& collectives) const {
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
    static_next = mfq_tensor_backend::Tensor();
    state_addresses.clear();
    valid = false;
}

} // namespace mfq::cuda
