#pragma once

#include "engine/tensor.h"

#include <array>
#include <cstdint>

namespace tyloi::ops {

/** Build a resident position span and append it to optional request history. */
std::array<Tensor, 2> position_span(std::int64_t begin, std::int64_t count, std::int64_t offset,
    const Tensor& history, const TensorOptions& options);

/** Normalize supplied or generated positions to [axes,batch,tokens] and append resident history. */
std::array<Tensor, 2> position_axes(std::int64_t begin, std::int64_t count, std::int64_t batch,
    std::int64_t axes, const Tensor& supplied, const Tensor& history, const TensorOptions& options);

} // namespace tyloi::ops
