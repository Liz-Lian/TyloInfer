#pragma once

#include "engine/backends/cuda/ops/projection/projection.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace mfq::cuda {

struct BiasLinear {
    QuantLinear weight;
    mfq_tensor_backend::Tensor bias;

    // Slice a dense fused projection into contiguous output rows.
    BiasLinear rows(int64_t begin, int64_t count) const {
        if (!weight.is_dense()) {
            throw std::runtime_error("fused vision projection must stay dense to be split");
        }
        if (begin < 0 || count <= 0 || begin + count > weight.dense.size(0)) {
            throw std::runtime_error("fused vision projection row range is invalid");
        }
        BiasLinear result;
        result.weight = weight;
        result.weight.dense = weight.dense.narrow(0, begin, count).contiguous();
        if (bias.defined()) {
            result.bias = bias.narrow(0, begin, count).contiguous();
        }
        return result;
    }

    mfq_tensor_backend::Tensor forward(const Projector& projector,
                                       mfq_tensor_backend::Tensor input) const {
        const auto output_dtype = input.scalar_type();
        if (weight.is_dense()) {
            const auto dense_dtype = weight.dense.scalar_type();
            std::optional<mfq_tensor_backend::Tensor> dense_bias = std::nullopt;
            if (bias.defined()) {
                dense_bias = bias.to(dense_dtype);
            }
            return mfq_linear(input.to(dense_dtype), weight.dense, dense_bias)
                .to(output_dtype)
                .contiguous();
        }
        auto output = projector.run(weight, input);
        if (bias.defined()) {
            output = output + bias.to(output.scalar_type());
        }
        return output.to(output_dtype).contiguous();
    }
};

} // namespace mfq::cuda

namespace mfq::cuda::vision {

using Tensor = mfq_tensor_backend::Tensor;

struct Affine {
    QuantLinear linear;
    Tensor bias;

    Tensor operator()(const Projector& projector, Tensor input) const {
        auto output = projector.run(linear, std::move(input));
        return bias.defined()
            ? output + bias.to(output.scalar_type())
            : output;
    }
};

} // namespace mfq::cuda::vision
