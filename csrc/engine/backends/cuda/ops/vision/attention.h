#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {

inline mfq_tensor_backend::Tensor vision_non_causal_attention(
        mfq_tensor_backend::Tensor query, mfq_tensor_backend::Tensor key,
        mfq_tensor_backend::Tensor value,
        MfqOptional<mfq_tensor_backend::Tensor> additive_mask, double scale) {
    auto scores = mfq_tensor_backend::matmul(query, key.transpose(-2, -1)) * scale;
    if (additive_mask.has_value()) {
        scores = scores + additive_mask.value().to(scores.scalar_type());
    }
    auto probabilities = mfq_tensor_backend::softmax(scores, -1, mfq_tensor_backend::kFloat32)
                             .to(value.scalar_type());
    return mfq_tensor_backend::matmul(probabilities, value).contiguous();
}

} // namespace mfq::cuda

namespace mfq::cuda::vision {

using Tensor = mfq_tensor_backend::Tensor;

inline Tensor apply_axial_rope(
    const Tensor& input,
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
    const auto cpu = mfq_tensor_backend::TensorOptions()
                         .dtype(mfq_tensor_backend::kFloat32)
                         .device(mfq_tensor_backend::kCPU);
    auto cos = mfq_tensor_backend::tensor(cosine, cpu)
                   .reshape({tokens, 1, head_dim})
                   .to(input.device());
    auto sin = mfq_tensor_backend::tensor(sine, cpu)
                   .reshape({tokens, 1, head_dim})
                   .to(input.device());
    auto source = input.to(mfq_tensor_backend::kFloat32);
    auto first = source.narrow(-1, 0, half);
    auto second = source.narrow(-1, half, half);
    auto rotated = mfq_tensor_backend::cat({-second, first}, -1);
    return (source * cos + rotated * sin)
        .to(input.scalar_type())
        .contiguous();
}

inline Tensor segmented_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::vector<std::int32_t>& lengths) {
    std::vector<Tensor> outputs;
    outputs.reserve(lengths.size());
    std::int64_t offset = 0;
    for (const std::int32_t length : lengths) {
        if (length <= 0 || offset + length > query.size(0)) {
            throw std::runtime_error("vision attention segment is invalid");
        }
        const auto segment = [offset, length](const Tensor& input) {
            return input.narrow(0, offset, length)
                .transpose(0, 1)
                .unsqueeze(0)
                .contiguous();
        };
        auto attended = attention_cuda(
            segment(query).to(mfq_tensor_backend::kFloat16),
            segment(key).to(mfq_tensor_backend::kFloat16),
            segment(value).to(mfq_tensor_backend::kFloat16),
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
        : mfq_tensor_backend::cat(outputs, 0);
}

} // namespace mfq::cuda::vision
