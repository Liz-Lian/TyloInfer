#pragma once

#include <mlx/mlx.h>
#include <optional>

namespace mfq::metal {
std::optional<mlx::core::array> mlx_staged_attention(
    const mlx::core::array& query, const mlx::core::array& key,
    const mlx::core::array& value, bool causal, float scale,
    const std::optional<mlx::core::array>& mask);
}
