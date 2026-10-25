#pragma once

#include "quant_linear.h"
#include "backends/cuda/engine/executor/execution.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "engine/models/common/grid_vision_model.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda::vision {

using Tensor = mfq_tensor_backend::Tensor;

struct Affine {
    QuantLinear linear;
    Tensor bias;

    Tensor operator()(CudaExecutionContext& execution, Tensor input) const {
        auto output = execution.projector().run(linear, std::move(input));
        return bias.defined()
            ? output + bias.to(output.scalar_type())
            : output;
    }
};

struct RmsNorm {
    Tensor weight;
    double epsilon = 1e-6;

    template <class Visitor>
    void visit_tensors(Visitor visit) {
        visit(weight);
    }

    Tensor operator()(const Tensor& input) const {
        if (input.dim() == 0 || input.size(-1) != weight.size(0)) {
            throw std::runtime_error("vision RMSNorm input width mismatch");
        }
        const auto shape = input.sizes().vec();
        auto normalized = rms_norm_cuda(
            input.reshape({-1, shape.back()}).to(mfq_tensor_backend::kFloat32),
            weight,
            epsilon);
        return normalized.reshape(shape)
            .to(input.scalar_type())
            .contiguous();
    }
};

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

struct VisionAttention {
    Affine qkv;
    Affine output;
    std::int64_t hidden = 0;
    std::int64_t heads = 0;
    std::int64_t head_dim = 0;
    double theta = 10'000.0;

    Tensor operator()(
        CudaExecutionContext& execution,
        const Tensor& input,
        const std::vector<std::int32_t>& positions,
        const std::vector<std::int32_t>& segment_lengths) const {
        const std::int64_t tokens = input.size(0);
        return mfq::models::grid_vision::attention(
            input,
            [&](Tensor value) { return qkv(execution, std::move(value)); },
            [&](Tensor value) {
                auto projected = value.reshape({tokens, 3, heads, head_dim});
                return std::array<Tensor, 3>{
                    projected.select(1, 0),
                    projected.select(1, 1),
                    projected.select(1, 2),
                };
            },
            [&](Tensor value) {
                return apply_axial_rope(value, positions, theta);
            },
            [&](Tensor q, Tensor k, Tensor v) {
                return segmented_attention(q, k, v, segment_lengths);
            },
            [&](Tensor value) {
                return output(
                    execution,
                    value.reshape({tokens, hidden}));
            });
    }
};

inline Tensor gelu_tanh(const Tensor& input) {
    return mfq_tensor_backend::gelu(input, "tanh");
}

} // namespace mfq::cuda::vision
