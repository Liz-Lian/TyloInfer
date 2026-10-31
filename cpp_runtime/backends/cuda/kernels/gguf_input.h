#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::gguf {
struct RouteGroup;
enum class InputType { Float32, Float16, BFloat16 };
// Consume the original activation view once. The optional mapped FP32 copy
// supplies CPU experts; GPU experts consume the identical native Q8_1 packet.
void prepare_moe_input(const void* input, InputType type, std::int64_t tokens,
    std::int64_t columns, std::int64_t row_stride, std::int64_t column_stride,
    void* packed, float* mapped_cpu, cudaStream_t stream);
// A warp keeps four adjacent values per lane, then writes native Q8 directly.
// Blocks above the register tile return false for the shared-memory transform.
bool try_prepare_rotated_moe_input(const void* input, InputType type, std::int64_t tokens,
    std::int64_t columns, std::int64_t row_stride, std::int64_t column_stride,
    const std::int8_t* signs, int block, void* packed, float* mapped_cpu, cudaStream_t stream);
bool try_quantize_rotated_moe_hidden(const float* gate, const float* up,
    std::int64_t rows, std::int64_t columns, bool half, float limit,
    const std::int8_t* signs, int block, void* packed, cudaStream_t stream);
void materialize_moe_hidden(const float* gate, const float* up, std::int64_t rows,
    std::int64_t columns, bool half, float limit, void* output, cudaStream_t stream);
// One slot per thread permits a cooperative address/group snapshot. Larger
// route sets retain the separate kernels when this exceeds the kernel limit.
bool try_prepare_resident_routes(const std::uint64_t* mapped, std::uint64_t* device,
    const std::int64_t* ids, std::int64_t slots, std::int64_t experts, int group_capacity,
    RouteGroup* groups, const std::uint32_t* admission_ready, std::uint64_t* start_clock,
    cudaStream_t stream);
}
