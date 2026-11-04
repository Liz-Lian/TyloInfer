#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/requests/sampling.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::cuda { class Collectives; }

std::vector<MfqCudaStream> make_cuda_graph_compute_streams(
    const MfqCudaStream& primary_stream,
    const tyloi::cuda::Collectives& collectives);
std::vector<MfqCudaStream> cuda_graph_participant_streams(
    const std::vector<MfqCudaStream>& compute_streams,
    const tyloi::cuda::Collectives& collectives);
std::vector<std::unique_ptr<MfqCudaStreamGuard>>
activate_cuda_graph_compute_streams(
    const std::vector<MfqCudaStream>& compute_streams);

struct DecodeGraphBranchScope {
    bool& setting;
    bool previous;

    explicit DecodeGraphBranchScope(bool& value)
        : setting(value), previous(value) {
        setting = true;
    }
    ~DecodeGraphBranchScope() {
        setting = previous;
    }
};

struct DecodeGraphCache {
    decltype(mfq_get_stream_from_pool(false)) stream;
    std::vector<MfqCudaStream> compute_streams;
    std::vector<::tyloi::DenseTensor> storage;
    std::unique_ptr<MfqCudaGraph> graph;
    ::tyloi::DenseTensor static_input;
    ::tyloi::DenseTensor static_pos;
    ::tyloi::DenseTensor static_len;
    ::tyloi::DenseTensor static_step;
    ::tyloi::DenseTensor generated;
    ::tyloi::DenseTensor random;
    ::tyloi::DenseTensor counts;
    ::tyloi::DenseTensor static_next;
    int64_t generated_capacity = 0;
    int64_t planned_len = 0;
    bool greedy = false;
    double temperature = 0.0;
    int32_t top_k = 0;
    double top_p = 1.0;
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    double repetition_penalty = 1.0;
    uint64_t captures = 0;
    uint64_t reuses = 0;
    bool valid = false;

    explicit DecodeGraphCache(int64_t context_capacity);
    void ensure_compute_streams(const tyloi::cuda::Collectives& collectives);
    std::vector<MfqCudaStream> graph_participant_streams(
        const tyloi::cuda::Collectives& collectives) const;
    bool matches(int64_t candidate_len, const MfqSamplingParams& sampling,
                 bool candidate_greedy,
                 const std::vector<::tyloi::DenseTensor>& candidate_storage) const;
    void ensure_storage(int64_t vocab_size);
    void initialize_inputs(const ::tyloi::DenseTensor& pending, int64_t position,
                           const ::tyloi::DenseTensor& random_host);
    void upload_random(const ::tyloi::DenseTensor& random_host);
    void invalidate();
    void set_key(int64_t candidate_len, const MfqSamplingParams& sampling,
                 bool candidate_greedy,
                 const std::vector<::tyloi::DenseTensor>& candidate_storage);
};

template <typename StateTensor, typename PositionState, typename Warmup>
void prepare_decode_graph_memory(
        const std::vector<StateTensor*>& states,
        const PositionState& position_state,
        bool double_warmup,
        MfqCudaGraph& graph,
        Warmup&& warmup,
        const std::vector<MfqCudaStream>& participant_streams = {}) {
    using DenseTensor = ::tyloi::DenseTensor;
    struct SavedRecurrentState {
        StateTensor* target;
        DenseTensor value;
        const void* address;
    };
    // Snapshot before entering the private graph allocator. These copies are
    // temporary and must not become retained allocations in the captured pool.
    std::vector<SavedRecurrentState> saved;
    saved.reserve(states.size());
    for (auto* state : states) {
        saved.push_back({state, DenseTensor(*state).clone(), state->data_ptr()});
    }
    const int64_t saved_position = position_state.position();
    auto restore = [&]() {
        for (const auto& state : saved) {
            TYLOINFER_RUNTIME_CHECK(state.target->data_ptr() == state.address,
                "decode warmup changed recurrent storage addresses");
            DenseTensor(*state.target).copy_(state.value);
        }
        TYLOINFER_RUNTIME_CHECK(position_state.position() == saved_position,
            "static decode warmup changed cache position");
        if (participant_streams.empty()) {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(
                mfq_get_current_cuda_stream().stream()));
            return;
        }
        for (const auto& participant : participant_streams) {
            MfqCudaGuard guard(participant.device_index());
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(
                participant.stream()));
        }
    };
    mfq_prepare_cuda_graph_memory(graph, participant_streams);
    if (!saved.empty() || double_warmup) {
        // The first pass may initialize persistent CUDA/NCCL workspace state.
        // A second pass then records the complete set of reusable temporaries
        // needed by capture after those persistent allocations exist.
        for (int pass = 0; pass < 2; ++pass) {
            try {
                // Replay overwrites this same, still-unconfirmed KV position.
                warmup();
                restore();
            } catch (...) {
                restore();
                throw;
            }
        }
    }
}

int64_t decode_graph_bucket(int64_t planned_len, int64_t context_capacity);
int64_t decode_graph_attention_parts(int64_t planned_len, int64_t max_parts);
