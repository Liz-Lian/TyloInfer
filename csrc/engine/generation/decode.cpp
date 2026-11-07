#include "engine/generation/decode.h"

#ifdef TYLOINFER_NATIVE_CUDA
#include <algorithm>
#include <utility>

namespace tyloi::engine {
std::int64_t decode_bucket(std::int64_t planned_len, std::int64_t context_capacity) {
    const std::int64_t quantum = planned_len <= 4096 ? 512 : (planned_len <= 16384 ? 1024 : 2048);
    return std::min(((planned_len + quantum - 1) / quantum) * quantum, context_capacity);
}

Decode::Decode(std::int64_t capacity, Device target, std::unique_ptr<Executor::Graph> execution_graph)
    : device(target), graph(std::move(execution_graph)), generated_capacity(std::max<std::int64_t>(capacity, 2048)) {}
void Decode::ensure_storage(std::int64_t vocabulary) {
    const auto i64 = TensorOptions().dtype(kInt64).device(device);
    if (!static_input) static_input = ops::empty({1, 1}, i64);
    if (!static_pos) static_pos = ops::empty({1}, i64);
    if (!static_len) static_len = ops::empty({1}, i64);
    if (!static_step) static_step = ops::empty({1}, i64);
    if (!generated) generated = ops::empty({generated_capacity}, i64);
    if (!random) random = ops::empty({1}, TensorOptions().dtype(kFloat32).device(device));
    if (!counts) counts = ops::empty({vocabulary}, TensorOptions().dtype(kInt32).device(device));
}
void Decode::initialize_inputs(const Tensor& pending, std::int64_t position, Tensor& random_host) {
    ops::copy_into(*static_input, *ops::reshape(pending, {1, 1}));
    ops::copy_into(*ops::slice(*generated, 0, 0, 1), *ops::reshape(pending, {1}));
    ops::fill(*static_pos, position);
    ops::fill(*static_len, position + 1);
    ops::fill(*static_step, 1);
    ops::fill(random_host, 0.5);
    upload_random(random_host);
    graph->wait();
}
void Decode::upload_random(const Tensor& random_host) { ops::copy_into(*random, random_host); }
void Decode::capture_next(std::shared_ptr<Tensor> next) {
    static_next = std::move(next);
    ops::decode_commit(*static_next, *generated, *static_step, *static_input, *static_pos, *static_len);
}
bool Decode::matches(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy,
                     std::span<const std::shared_ptr<const Tensor>> owners) const {
    return valid_ && planned_len == planned && greedy_ == greedy &&
        (greedy || (temperature_ == sampling.temperature && top_k_ == sampling.top_k && top_p_ == sampling.top_p)) &&
        presence_ == sampling.presence_penalty && frequency_ == sampling.frequency_penalty &&
        repetition_ == sampling.repetition_penalty && graph->matches(owners);
}
void Decode::set_key(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy) {
    planned_len = planned; greedy_ = greedy;
    temperature_ = sampling.temperature; top_k_ = sampling.top_k; top_p_ = sampling.top_p;
    presence_ = sampling.presence_penalty; frequency_ = sampling.frequency_penalty;
    repetition_ = sampling.repetition_penalty; valid_ = true;
}
void Decode::invalidate() {
    if (graph) graph->invalidate();
    static_next.reset(); valid_ = false;
}
} // namespace tyloi::engine
#endif
