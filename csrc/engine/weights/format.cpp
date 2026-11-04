#include "format.h"

#include <cstring>

uint32_t read_u32_from(const std::vector<uint8_t> & b, size_t & off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, sizeof(v));
    off += sizeof(v);
    return v;
}

namespace tyloi::cuda::quant_format {

::tyloi::Tensor cpu_u8_tensor(const std::vector<uint8_t> & v, std::initializer_list<int64_t> shape) {
    return ::tyloi::cuda::from_blob((void *)v.data(), shape, ::tyloi::TensorOptions().dtype(::tyloi::kUInt8)).clone();
}

::tyloi::Tensor cpu_i64_tensor(
        const std::vector<int64_t> & v,
        std::initializer_list<int64_t> shape) {
    return ::tyloi::cuda::from_blob(
        (void *)v.data(), shape,
        ::tyloi::TensorOptions().dtype(::tyloi::kInt64)).clone();
}

::tyloi::Tensor cpu_i32_tensor(
        const std::vector<int32_t> & v,
        std::initializer_list<int64_t> shape) {
    return ::tyloi::cuda::from_blob(
        (void *)v.data(), shape,
        ::tyloi::TensorOptions().dtype(::tyloi::kInt32)).clone();
}

::tyloi::Tensor cpu_f16_tensor(
        const std::vector<uint16_t> & v,
        std::initializer_list<int64_t> shape) {
    return ::tyloi::cuda::from_blob(
        (void *)v.data(), shape,
        ::tyloi::TensorOptions().dtype(::tyloi::kFloat16)).clone();
}

::tyloi::Tensor cpu_f16_to_f32_tensor(const std::vector<uint16_t> & v, int64_t n) {
    auto h = ::tyloi::cuda::from_blob((void *)v.data(), {n}, ::tyloi::TensorOptions().dtype(::tyloi::kFloat16));
    return h.to(::tyloi::kFloat32).contiguous();
}
::tyloi::Tensor cpu_i8_tensor(const std::vector<int8_t> & v, std::initializer_list<int64_t> shape) {
    return ::tyloi::cuda::from_blob((void *)v.data(), shape, ::tyloi::TensorOptions().dtype(::tyloi::kInt8)).clone();
}

} // namespace tyloi::cuda::quant_format
