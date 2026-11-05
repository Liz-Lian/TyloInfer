#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/executor.h"
#include "engine/ops.h"
#include "engine/requests/sampling.h"

namespace tyloi::engine {

std::int64_t decode_bucket(std::int64_t planned_len, std::int64_t context_capacity);

class Decode {
public:
    Decode(std::int64_t capacity, Device device, std::unique_ptr<Executor::Graph> graph);
    void ensure_storage(std::int64_t vocabulary);
    void initialize_inputs(const Tensor& pending, std::int64_t position, Tensor& random_host);
    void upload_random(const Tensor& random_host);
    void capture_next(std::shared_ptr<Tensor> next);
    bool matches(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy,
                 std::span<const std::shared_ptr<const Tensor>> owners) const;
    void set_key(std::int64_t planned, const MfqSamplingParams& sampling, bool greedy);
    void invalidate();

    Device device;
    std::shared_ptr<Tensor> static_input, static_pos, static_len, static_step;
    std::shared_ptr<Tensor> generated, random, counts, static_next;
    std::unique_ptr<Executor::Graph> graph;
    std::int64_t generated_capacity;
    std::uint64_t captures = 0, reuses = 0;
private:
    std::int64_t planned_len = 0;
    double temperature_ = 0, top_p_ = 1, presence_ = 0, frequency_ = 0, repetition_ = 1;
    std::int32_t top_k_ = 0;
    bool greedy_ = false, valid_ = false;
};
} // namespace tyloi::engine
#endif
