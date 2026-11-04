#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

std::uint32_t read_u32_from(
    const std::vector<std::uint8_t>& bytes, std::size_t& offset);

namespace tyloi::cuda::quant_format {

::tyloi::DenseTensor cpu_u8_tensor(
    const std::vector<uint8_t>& values,
    std::initializer_list<int64_t> shape);
::tyloi::DenseTensor cpu_i64_tensor(
    const std::vector<int64_t>& values,
    std::initializer_list<int64_t> shape);
::tyloi::DenseTensor cpu_i32_tensor(
    const std::vector<int32_t>& values,
    std::initializer_list<int64_t> shape);
::tyloi::DenseTensor cpu_f16_tensor(
    const std::vector<uint16_t>& values,
    std::initializer_list<int64_t> shape);
::tyloi::DenseTensor cpu_f16_to_f32_tensor(
    const std::vector<uint16_t>& values, int64_t count);
::tyloi::DenseTensor cpu_i8_tensor(
    const std::vector<int8_t>& values,
    std::initializer_list<int64_t> shape);

} // namespace tyloi::cuda::quant_format
