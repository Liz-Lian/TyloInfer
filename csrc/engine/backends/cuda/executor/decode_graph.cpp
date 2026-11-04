#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/native/collectives.h"

#include <algorithm>

namespace {

bool same_tensor_layout(const ::tyloi::Tensor& expected,
                        const ::tyloi::Tensor& candidate) {
    if (expected.defined() != candidate.defined()) return false;
    if (!expected.defined()) return true;
    if (expected.data_ptr() != candidate.data_ptr() ||
        expected.device() != candidate.device() ||
        expected.scalar_type() != candidate.scalar_type() ||
        expected.dim() != candidate.dim())
        return false;
    for (std::int64_t dimension = 0; dimension < expected.dim(); ++dimension) {
        if (expected.size(dimension) != candidate.size(dimension) ||
            expected.stride(dimension) != candidate.stride(dimension))
            return false;
    }
    return true;
}

bool same_storage(const std::vector<::tyloi::Tensor>& expected,
                  const std::vector<::tyloi::Tensor>& candidate) {
    if (expected.size() != candidate.size()) return false;
    for (std::size_t index = 0; index < expected.size(); ++index)
        if (!same_tensor_layout(expected[index], candidate[index])) return false;
    return true;
}

} // namespace

std::vector<MfqCudaStream> make_cuda_graph_compute_streams(
        const MfqCudaStream& primary_stream,
        const tyloi::cuda::Collectives& collectives) {
    if (collectives.streams().empty()) return {primary_stream};
    std::vector<MfqCudaStream> streams;
    streams.reserve(collectives.streams().size());
    for (const auto& participant : collectives.streams()) {
        const int device = participant.device_index();
        streams.push_back(
            device == primary_stream.device_index()
                ? primary_stream
                : mfq_get_stream_from_pool(false, device));
    }
    return streams;
}

std::vector<MfqCudaStream> cuda_graph_participant_streams(
        const std::vector<MfqCudaStream>& compute_streams,
        const tyloi::cuda::Collectives& collectives) {
    auto participants = compute_streams;
    participants.insert(
        participants.end(),
        collectives.streams().begin(), collectives.streams().end());
    return participants;
}

std::vector<std::unique_ptr<MfqCudaStreamGuard>>
activate_cuda_graph_compute_streams(
        const std::vector<MfqCudaStream>& compute_streams) {
    std::vector<std::unique_ptr<MfqCudaStreamGuard>> guards;
    guards.reserve(compute_streams.size());
    for (const auto& stream : compute_streams) {
        guards.push_back(
            std::make_unique<MfqCudaStreamGuard>(stream));
    }
    return guards;
}

DecodeGraphCache::DecodeGraphCache(int64_t context_capacity)
    : stream(mfq_get_stream_from_pool(false)),
      generated_capacity(std::max<int64_t>(context_capacity, 2048)) {}

void DecodeGraphCache::ensure_compute_streams(
        const tyloi::cuda::Collectives& collectives) {
    if (!compute_streams.empty()) return;
    compute_streams = make_cuda_graph_compute_streams(stream, collectives);
}

std::vector<MfqCudaStream> DecodeGraphCache::graph_participant_streams(
        const tyloi::cuda::Collectives& collectives) const {
    return cuda_graph_participant_streams(compute_streams, collectives);
}

bool DecodeGraphCache::matches(
        int64_t candidate_len, const MfqSamplingParams & sampling,
        bool candidate_greedy,
        const std::vector<::tyloi::Tensor>& candidate_storage) const {
    return valid && planned_len == candidate_len && greedy == candidate_greedy &&
           same_storage(storage, candidate_storage) &&
           (candidate_greedy ||
            (temperature == sampling.temperature && top_k == sampling.top_k &&
             top_p == sampling.top_p)) &&
           presence_penalty == sampling.presence_penalty &&
           frequency_penalty == sampling.frequency_penalty &&
           repetition_penalty == sampling.repetition_penalty;
}

void DecodeGraphCache::ensure_storage(int64_t vocab_size) {
    auto i64 = ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::kCUDA);
    if (!static_input.defined()) static_input = ::tyloi::cuda::empty({1, 1}, i64);
    if (!static_pos.defined()) static_pos = ::tyloi::cuda::empty({1}, i64);
    if (!static_len.defined()) static_len = ::tyloi::cuda::empty({1}, i64);
    if (!static_step.defined()) static_step = ::tyloi::cuda::empty({1}, i64);
    if (!generated.defined()) generated = ::tyloi::cuda::empty({generated_capacity}, i64);
    if (!random.defined()) {
        random = ::tyloi::cuda::empty(
            {1}, ::tyloi::TensorOptions().dtype(::tyloi::kFloat32).device(::tyloi::kCUDA));
    }
    if (!counts.defined()) {
        counts = ::tyloi::cuda::empty(
            {vocab_size}, ::tyloi::TensorOptions().dtype(::tyloi::kInt32).device(::tyloi::kCUDA));
    }
}

void DecodeGraphCache::initialize_inputs(
        const ::tyloi::Tensor& pending, int64_t position,
        const ::tyloi::Tensor& random_host) {
    cudaStream_t graph_stream = stream.stream();
    std::int64_t length = position + 1;
    std::int64_t step = 1;
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_input.data_ptr<int64_t>(),
        pending.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(generated.data_ptr<int64_t>(),
        pending.data_ptr<int64_t>(),
        sizeof(int64_t),
        cudaMemcpyDeviceToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_pos.data_ptr<int64_t>(),
        &position,
        sizeof(int64_t),
        cudaMemcpyHostToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_len.data_ptr<int64_t>(),
        &length,
        sizeof(int64_t),
        cudaMemcpyHostToDevice,
        graph_stream));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_step.data_ptr<int64_t>(),
        &step,
        sizeof(int64_t),
        cudaMemcpyHostToDevice,
        graph_stream));
    *random_host.data_ptr<float>() = 0.5f;
    upload_random(random_host);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream));
}

void DecodeGraphCache::upload_random(const ::tyloi::Tensor& random_host) {
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(random.data_ptr<float>(),
        random_host.data_ptr<float>(),
        sizeof(float),
        cudaMemcpyHostToDevice,
        stream.stream()));
}

void DecodeGraphCache::invalidate() {
    if (graph) graph->reset();
    graph.reset();
    storage.clear();
    static_next = ::tyloi::Tensor();
    valid = false;
}

void DecodeGraphCache::set_key(
        int64_t candidate_len, const MfqSamplingParams & sampling,
        bool candidate_greedy,
        const std::vector<::tyloi::Tensor>& candidate_storage) {
    planned_len = candidate_len;
    greedy = candidate_greedy;
    temperature = sampling.temperature;
    top_k = sampling.top_k;
    top_p = sampling.top_p;
    presence_penalty = sampling.presence_penalty;
    frequency_penalty = sampling.frequency_penalty;
    repetition_penalty = sampling.repetition_penalty;
    storage = candidate_storage;
    valid = true;
}

int64_t decode_graph_bucket(int64_t planned_len, int64_t context_capacity) {
    const int64_t quantum = planned_len <= 4096 ? 512 :
        (planned_len <= 16384 ? 1024 : 2048);
    const int64_t rounded = ((planned_len + quantum - 1) / quantum) * quantum;
    return std::min<int64_t>(rounded, context_capacity);
}

int64_t decode_graph_attention_parts(
        int64_t planned_len, int64_t max_parts) {
    const int64_t parts = planned_len >= 192
        ? (planned_len + 127) / 128 : 1;
    return std::min(parts, max_parts);
}
