#pragma once
#include <mlx/mlx.h>

namespace mfq::metal {
struct MlxGgufGroupedRoutes {
    mlx::core::array descriptors, count, tile_offsets, grouped_routes, route_rows;
    int tiles, tile_rows;
    int maximum_live_rows = 0;
};
MlxGgufGroupedRoutes gguf_group_routes(const mlx::core::array& ids, int experts, int tile_rows, bool shared = false);
}
