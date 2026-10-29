#pragma once

#include <mlx/mlx.h>
#include <array>
#include <span>
#include <vector>

namespace mfq::metal {
class MfqContainer;

struct MlxNativeGgufWeight {
    std::array<mlx::core::array, 3> planes;
    int format, width, outputs, experts = 1, padded_outputs = 0;
    int storage_outputs() const noexcept { return padded_outputs ? padded_outputs : outputs; }
};

struct MlxNativeGgufMoe {
    std::array<MlxNativeGgufWeight, 3> routed, shared;
    mlx::core::array router, scalar_gate, slots;
    int top_k;
};

MlxNativeGgufWeight mlx_native_gguf_load(const MfqContainer& model, const std::string& name);
std::array<mlx::core::array, 2> mlx_native_gguf_hyper(const mlx::core::array& streams,
    const mlx::core::array& norm, const MlxNativeGgufWeight& down, const MlxNativeGgufWeight& up,
    const mlx::core::array* inject);
MlxNativeGgufWeight mlx_native_gguf_float(const mlx::core::array& weight);
mlx::core::array mlx_native_gguf_moe(const MlxNativeGgufMoe& weights, const mlx::core::array& input);
bool mlx_native_gguf_supported(int ggml_type, int rows, int width) noexcept;
MlxNativeGgufWeight mlx_native_gguf_pack(const mlx::core::array& packed,
    int ggml_type, int rows, int width, int row_bytes);
MlxNativeGgufWeight mlx_native_affine_pack(const mlx::core::array& values,
    const mlx::core::array& scales, const mlx::core::array& biases, int bits, int group_size);
bool mlx_native_affine_supported(int bits, int group_size, int outputs, int width) noexcept;
std::vector<mlx::core::array> mlx_native_gguf_project(
    std::span<const MlxNativeGgufWeight> weights, const mlx::core::array& input);
mlx::core::array mlx_native_gguf_swiglu(const MlxNativeGgufWeight& gate,
    const MlxNativeGgufWeight& up, const mlx::core::array& input);

}
