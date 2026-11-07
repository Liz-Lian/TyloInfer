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

    static std::vector<float> probabilities(DenseTensor logits, const Params& parameters) {
        auto host = logits.to(::tyloi::kFloat32).cpu().contiguous();
        return tyloi::engine::mtp::distribution(
            std::span<const float>(host.template data_ptr<float>(), host.numel()),
            parameters.temperature, parameters.top_k, parameters.top_p);
    }

    static std::vector<tyloi::engine::mtp::CompactDistribution>
    compact_radix_rows(DenseTensor logits, const Params& parameters) {
        const auto host = sample_top_k_compact_cuda(logits, parameters.top_k).cpu().contiguous();
        return compact_rows_from_host(host, parameters);
    }

    static std::vector<tyloi::engine::mtp::CompactDistribution>
    compact_rows_from_host(const DenseTensor& host, const Params& parameters) {
        const auto* packets = host.template data_ptr<int64_t>();
        std::array<float, 64> values;
        std::array<int64_t, 64> indices;
        std::vector<tyloi::engine::mtp::CompactDistribution> result;
        result.reserve(std::size_t(host.size(0)));
        for (int64_t row = 0; row < host.size(0); ++row) {
            for (int i = 0; i < parameters.top_k; ++i) {
                const auto packet = uint64_t(packets[row * host.size(1) + i]);
                const uint32_t raw = uint32_t(packet >> 32);
                std::memcpy(&values[i], &raw, sizeof(raw));
                indices[i] = uint32_t(packet);
            }
            result.push_back(tyloi::engine::mtp::compact_distribution_from_topk(values.data(), indices.data(),
                int(parameters.top_k), parameters.temperature, parameters.top_p));
        }
        return result;
    }

    static tyloi::engine::mtp::CompactDistribution
    compact_probabilities(DenseTensor logits, const Params& parameters) {
        if(compact_radix_enabled(logits))return std::move(compact_radix_rows(logits,parameters).front());
        auto selected =
            ::tyloi::cuda::topk(logits, parameters.top_k, -1, true, true);
        auto values =
            std::get<0>(selected)
                .to(::tyloi::kCPU, ::tyloi::kFloat32)
                .contiguous()
                .reshape({-1});
        auto indices = std::get<1>(selected)
                           .to(::tyloi::kCPU, ::tyloi::kInt64)
                           .contiguous()
                           .reshape({-1});
        return tyloi::engine::mtp::compact_distribution_from_topk(
            values.template data_ptr<float>(), indices.template data_ptr<int64_t>(),
            static_cast<int>(values.numel()), parameters.temperature,
            parameters.top_p);
    }

    static std::vector<tyloi::engine::mtp::CompactDistribution>
    compact_probability_rows(DenseTensor logits, const Params& parameters) {
        if(compact_radix_enabled(logits))return compact_radix_rows(logits,parameters);
        auto selected =
            ::tyloi::cuda::topk(logits, parameters.top_k, -1, true, true);
        auto values =
            std::get<0>(selected)
                .to(::tyloi::kCPU, ::tyloi::kFloat32)
                .contiguous();
        auto indices = std::get<1>(selected)
                           .to(::tyloi::kCPU, ::tyloi::kInt64)
                           .contiguous();
        const int64_t rows = values.size(0);
        const int64_t columns = values.size(1);
        const float *value_data = values.template data_ptr<float>();
        const int64_t *index_data = indices.template data_ptr<int64_t>();
        std::vector<tyloi::engine::mtp::CompactDistribution> result;
        result.reserve(static_cast<size_t>(rows));
        for (int64_t row = 0; row < rows; ++row) {
            result.push_back(tyloi::engine::mtp::compact_distribution_from_topk(
                value_data + row * columns, index_data + row * columns,
                static_cast<int>(columns), parameters.temperature, parameters.top_p));
        }
        return result;
    }

private:
    static bool compact_radix_enabled(const DenseTensor& logits) {
        const auto* option = tyloi::cuda::env_lookup("TYLOINFER_MTP_RADIX_TOPK");
        return option && option[0] == '1' && logits.size(0) <= 32;
    }

    DenseTensor random_host_;
    DenseTensor random_cuda_;
};

#ifdef TYLOINFER_NATIVE_CUDA
using Sampler = tyloi::engine::Sampler;
#else
using Sampler = tyloi::engine::Sampler<SamplingOps>;
#endif

} // namespace tyloi::cuda
