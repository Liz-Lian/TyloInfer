#pragma once

#include "shape.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::native_ops_detail {

__device__ inline std::int64_t tensor_offset(const TensorView& view, std::int64_t linear) {
    std::int64_t offset = 0;
    for (std::size_t reverse = view.rank; reverse > 0; --reverse) {
        const auto dimension = reverse - 1;
        const auto coordinate = linear % view.sizes[dimension];
        linear /= view.sizes[dimension];
        offset += coordinate * view.strides[dimension];
    }
    return offset;
}

template <typename Value>
__device__ double load_number(const Value* values, std::int64_t index) {
    if constexpr (std::is_same_v<Value, __half>) {
        return static_cast<double>(__half2float(values[index]));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return static_cast<double>(__bfloat162float(values[index]));
    } else {
        return static_cast<double>(values[index]);
    }
}

template <typename Value>
__device__ Value store_number(double value) {
    if constexpr (std::is_same_v<Value, __half>) {
        return __float2half_rn(static_cast<float>(value));
    } else if constexpr (std::is_same_v<Value, __nv_bfloat16>) {
        return __float2bfloat16_rn(static_cast<float>(value));
    } else {
        return static_cast<Value>(value);
    }
}

template <typename Function>
void dispatch_numeric(ScalarType type, Function&& function) {
    switch (type) {
        case kBool: function.template operator()<bool>(); break;
        case kUInt8: function.template operator()<std::uint8_t>(); break;
        case kInt8: function.template operator()<std::int8_t>(); break;
        case kInt16: function.template operator()<std::int16_t>(); break;
        case kInt32: function.template operator()<std::int32_t>(); break;
        case kInt64: function.template operator()<std::int64_t>(); break;
        case kFloat16: function.template operator()<__half>(); break;
        case kBFloat16: function.template operator()<__nv_bfloat16>(); break;
        case kFloat32: function.template operator()<float>(); break;
        case kFloat64: function.template operator()<double>(); break;
        case kFloat8E4M3FN:
            throw std::invalid_argument("generic native op does not accept FP8 storage");
    }
}

inline std::pair<int, int> launch_geometry(std::int64_t elements) {
    constexpr int threads = 256;
    const int blocks = static_cast<int>(std::min<std::int64_t>(
        4096, std::max<std::int64_t>(1, (elements + threads - 1) / threads)));
    return {blocks, threads};
}


extern std::atomic<std::int64_t> native_random_seed;

}  // namespace tyloi::cuda::native_ops_detail
