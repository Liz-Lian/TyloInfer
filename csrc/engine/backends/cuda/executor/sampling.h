#pragma once

#include "engine/generation/sampling.h"
#include "engine/generation/mtp/policy.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/requests/sampling.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace tyloi::cuda {

class SamplingOps {
public:
    using DenseTensor = ::tyloi::DenseTensor;
    using Params = ::MfqSamplingParams;

    SamplingOps()
        : random_host_(::tyloi::cuda::empty({1}, ::tyloi::TensorOptions()
              .dtype(::tyloi::kFloat32).device(::tyloi::kCPU)
              .pinned_memory(true))),
          random_cuda_(::tyloi::cuda::empty({1}, ::tyloi::TensorOptions()
              .dtype(::tyloi::kFloat32).device(::tyloi::kCUDA))) {}

    const DenseTensor& random_host() const noexcept { return random_host_; }

    static DenseTensor token_counts(const DenseTensor& ids, std::int64_t vocabulary, DenseTensor storage = {}) {
        if (!storage.defined())
            storage = ::tyloi::cuda::empty({vocabulary}, ids.options().dtype(::tyloi::kInt32));
        storage.zero_();
        sample_token_counts_add_cuda(storage, ids);
        return storage;
    }

    DenseTensor sample_greedy(DenseTensor logits) {
        return ::sample_greedy_cuda(std::move(logits));
    }

    DenseTensor sample_stochastic(
            DenseTensor logits, float random, const Params& params) {
        *random_host_.data_ptr<float>() = random;
        random_cuda_.copy_(random_host_, true);
        return sample_stochastic(
            std::move(logits), random_cuda_, params);
    }

    DenseTensor sample_stochastic(
            DenseTensor logits, DenseTensor random_cuda, const Params& params) {
        if (params.top_k > 0) {
            return ::sample_top_k_top_p_cuda(
                std::move(logits), std::move(random_cuda),
                params.temperature, params.top_k, params.top_p);
        }
        return ::sample_softmax_cuda(
            std::move(logits), std::move(random_cuda), params.temperature);
    }

    DenseTensor apply_penalties(
            DenseTensor logits, const DenseTensor& counts, const Params& params) {
        return ::sample_apply_penalties_cuda(
            std::move(logits), counts, params.presence_penalty,
            params.frequency_penalty, params.repetition_penalty);
    }

    static DenseTensor constrain(
            const DenseTensor& logits, const MfqTokenConstraintPtr& token_constraint) {
        auto masked = logits
            .to(::tyloi::kCPU, ::tyloi::kFloat32)
            .contiguous();
        token_constraint->apply(
            masked.data_ptr<float>(),
            static_cast<std::size_t>(masked.numel()));
        return masked;
    }

private:

    DenseTensor random_host_;
    DenseTensor random_cuda_;
};

using Sampler = tyloi::engine::Sampler;

} // namespace tyloi::cuda
