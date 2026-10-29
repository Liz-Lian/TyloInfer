#pragma once
#include <mlx/mlx.h>
#include <array>
#include <optional>

namespace mfq::metal {
struct MlxExpertRouteTiles {
    mlx::core::array order, bounds, tile_offsets;
    int tiles;
    int tile_rows = 16;
    mutable std::optional<mlx::core::array> affine_sorted_ids;
    mutable std::array<std::optional<std::pair<mlx::core::array, mlx::core::array>>, 3> affine_tiles;
};
MlxExpertRouteTiles mlx_expert_route_tiles(const mlx::core::array& ids, int experts, int tile_rows = 16);
} // namespace mfq::metal
