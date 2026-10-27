#pragma once
#include <mlx/mlx.h>

namespace mfq::metal {
struct MlxExpertRouteTiles {
    mlx::core::array order, bounds, tile_offsets;
    int tiles;
    int tile_rows = 16;
};
MlxExpertRouteTiles mlx_expert_route_tiles(const mlx::core::array& ids, int experts, int tile_rows = 16);
} // namespace mfq::metal
