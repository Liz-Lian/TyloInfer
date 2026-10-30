#pragma once
#include "mlx_gguf.h"
#include "mlx_gguf_routes.h"

namespace mfq::metal {
class MlxGgufStagedPrefill {
public:
    static bool supports(const MlxGgufWeight& weight) noexcept;
    static bool supports_grouped(const MlxGgufWeight& weight) noexcept;
    static bool prefers_weighted_down(const MlxGgufWeight& weight) noexcept;
    static const std::string& scalar_decoder_header();
    static const std::string& decoder_header();
    static void prepare(MlxGgufWeight& weight);
    static mlx::core::array dequantize(const MlxGgufWeight& weight, mlx::core::Dtype dtype);
    static mlx::core::array embedding(const MlxGgufWeight& weight, const mlx::core::array& ids, mlx::core::Dtype dtype);
    static mlx::core::array project(const MlxGgufWeight& weight, const mlx::core::array& input, int groups = 1);
    static std::vector<mlx::core::array> project_many(std::span<const MlxGgufWeight> weights,
        const mlx::core::array& input);
    static std::optional<std::vector<mlx::core::array>> project_split_many(
        std::span<const MlxGgufWeight> weights, const mlx::core::array& input);
    static mlx::core::array dense(const MlxGgufWeight& weight, const mlx::core::array& input);
    static mlx::core::array dense_small(const MlxGgufWeight& weight, const mlx::core::array& input);
    static bool prefers_small_matrix(const MlxGgufWeight& weight) noexcept;
    static mlx::core::array gate_up(const MlxGgufWeight& gate, const MlxGgufWeight& up,
        const mlx::core::array& input, const MlxGgufGroupedRoutes& routes, mlx::core::Dtype dtype,
        const MlxGgufWeight* shared_gate = nullptr, const MlxGgufWeight* shared_up = nullptr);
    static mlx::core::array down(const MlxGgufWeight& weight, const mlx::core::array& input,
        const MlxGgufGroupedRoutes& routes, mlx::core::Dtype dtype, const MlxGgufWeight* shared = nullptr);
    static mlx::core::array weighted_down(const MlxGgufWeight& weight, const mlx::core::array& input,
        const mlx::core::array& ids, const mlx::core::array& probabilities, mlx::core::Dtype dtype);
};
} // namespace mfq::metal
