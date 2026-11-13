#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda {

inline ::tyloi::DenseTensor vision_non_causal_attention(
        ::tyloi::DenseTensor query, ::tyloi::DenseTensor key,
        ::tyloi::DenseTensor value,
        MfqOptional<::tyloi::DenseTensor> additive_mask, double scale) {
    auto scores = ::tyloi::cuda::matmul(query, key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = ::tyloi::cuda::softmax(scores, -1, ::tyloi::kFloat32)
                             .to(value.scalar_type());
    return ::tyloi::cuda::matmul(probabilities, value).contiguous();
}

} // namespace tyloi::cuda

namespace tyloi::cuda::vision {

using DenseTensor = ::tyloi::DenseTensor;

inline DenseTensor apply_axial_rope(
    const DenseTensor& input,
    const std::vector<std::int32_t>& positions,
    double theta) {
    if (input.dim() != 3 || input.size(0) <= 0 ||
        input.size(2) <= 0 || input.size(2) % 4 != 0 ||
        positions.size() != static_cast<std::size_t>(input.size(0) * 2)) {
        throw std::runtime_error("vision axial RoPE geometry mismatch");
    }
    const std::int64_t tokens = input.size(0);
    const std::int64_t head_dim = input.size(2);
    const std::int64_t half = head_dim / 2;
    const std::int64_t frequencies = half / 2;
    std::vector<float> cosine(static_cast<std::size_t>(tokens * head_dim));
    std::vector<float> sine(cosine.size());
    for (std::int64_t token = 0; token < tokens; ++token) {
        for (int axis = 0; axis < 2; ++axis) {
            const float position = static_cast<float>(
                positions[static_cast<std::size_t>(token * 2 + axis)]);
            for (std::int64_t index = 0; index < frequencies; ++index) {
                const float inverse = std::pow(
                    static_cast<float>(theta),
                    -static_cast<float>(2 * index) /
                        static_cast<float>(half));
                const float angle = position * inverse;
                const std::int64_t column = axis * frequencies + index;
                for (int repeat = 0; repeat < 2; ++repeat) {
                    const auto offset = static_cast<std::size_t>(
                        token * head_dim + repeat * half + column);
                    cosine[offset] = std::cos(angle);
                    sine[offset] = std::sin(angle);
                }
            }
        }
    }
    const auto cpu = ::tyloi::TensorOptions()
                         .dtype(::tyloi::kFloat32)
                         .device(::tyloi::kCPU);
    auto cos = ::tyloi::cuda::tensor(cosine, cpu)
                   .reshape({tokens, 1, head_dim})
                   .to(input.device());
    auto sin = ::tyloi::cuda::tensor(sine, cpu)
                   .reshape({tokens, 1, head_dim})
                   .to(input.device());
    auto source = input.to(::tyloi::kFloat32);
    auto first = source.narrow(-1, 0, half);
    auto second = source.narrow(-1, half, half);
    auto rotated = ::tyloi::cuda::cat({-second, first}, -1);
    return (source * cos + rotated * sin)
        .to(input.scalar_type())
        .contiguous();
}

inline DenseTensor segmented_attention(
    const DenseTensor& query,
    const DenseTensor& key,
    const DenseTensor& value,
    const std::vector<std::int32_t>& lengths) {
    std::vector<DenseTensor> outputs;
    outputs.reserve(lengths.size());
    std::int64_t offset = 0;
    for (const std::int32_t length : lengths) {
        if (length <= 0 || offset + length > query.size(0)) {
            throw std::runtime_error("vision attention segment is invalid");
        }
        const auto segment = [offset, length](const DenseTensor& input) {
            return input.narrow(0, offset, length)
                .transpose(0, 1)
                .unsqueeze(0)
                .contiguous();
        };
        auto attended = attention_cuda(
            segment(query).to(::tyloi::kFloat16),
            segment(key).to(::tyloi::kFloat16),
            segment(value).to(::tyloi::kFloat16),
            1.0 / std::sqrt(static_cast<double>(query.size(2))),
            false);
        outputs.push_back(
            attended.index({0}).transpose(0, 1).contiguous());
        offset += length;
    }
    if (outputs.empty() || offset != query.size(0)) {
        throw std::runtime_error(
            "vision attention segments do not cover every patch");
    }
    return outputs.size() == 1
        ? outputs.front()
        : ::tyloi::cuda::cat(outputs, 0);
}

} // namespace tyloi::cuda::vision

namespace tyloi::cuda::ops {
inline tyloi::DenseTensor attention(tyloi::DenseTensor query, tyloi::DenseTensor key, tyloi::DenseTensor value,
                             const std::optional<tyloi::DenseTensor>& mask, double scale) {
    std::optional<MfqCudaGuard> guard;
    if (query.is_cuda()) guard.emplace(query.device().index);
    return vision_non_causal_attention(query, key, value, mask, scale);
}
inline tyloi::DenseTensor axial_rope(tyloi::DenseTensor input, const std::vector<std::int32_t>& positions, double theta) {
    MfqCudaGuard guard(input.device().index);
    return vision::apply_axial_rope(input, positions, theta);
}
inline tyloi::DenseTensor segmented_attention(tyloi::DenseTensor query, tyloi::DenseTensor key, tyloi::DenseTensor value,
                                       const std::vector<std::int32_t>& lengths) {
    MfqCudaGuard guard(query.device().index);
    return vision::segmented_attention(query, key,
                                       value, lengths);
}
}
