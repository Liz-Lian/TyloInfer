#pragma once

#include "engine/tensor.h"

namespace mfq::cuda::ops {

mfq::Tensor softcap(mfq::Tensor input, double cap);
mfq::Tensor scale(mfq::Tensor input, double factor);
mfq::Tensor copy(mfq::Tensor input, Device target);
mfq::Tensor cast(mfq::Tensor input, ScalarType type);
mfq::Tensor contiguous(mfq::Tensor input);
mfq::Tensor clone(mfq::Tensor input);
mfq::Tensor reshape(mfq::Tensor input, std::span<const std::int64_t> shape);
mfq::Tensor arange(std::int64_t first, std::int64_t last, Device device);
mfq::Tensor full(std::span<const std::int64_t> shape, std::int64_t value, Device device);
mfq::Tensor add(mfq::Tensor input, std::int64_t offset);
mfq::Tensor add(mfq::Tensor left, mfq::Tensor right);
mfq::Tensor multiply(mfq::Tensor left, mfq::Tensor right);
mfq::Tensor matmul(mfq::Tensor left, mfq::Tensor right);
mfq::Tensor transpose(mfq::Tensor input, std::int64_t first, std::int64_t second);
mfq::Tensor add_residual(mfq::Tensor residual, mfq::Tensor value);

} // namespace mfq::cuda::ops
