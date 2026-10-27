#include "mlx_expert_route_tiles.h"
#include <mlx/fast.h>
#include <climits>
#include <stdexcept>

namespace mfq::metal {
using namespace mlx::core;
MlxExpertRouteTiles mlx_expert_route_tiles(const array& ids, int experts, int tile_rows) {
    if (ids.ndim() != 2 || experts < 1 || tile_rows < 1 || tile_rows > 256 || ids.size() == 0 || ids.size() > INT_MAX ||
        (ids.dtype() != int32 && ids.dtype() != uint32) || ids.size() / tile_rows + std::size_t(experts) > INT_MAX)
        throw std::invalid_argument("invalid expert route tile geometry");
    static const auto kernel = fast::metal_kernel("tinfer_expert_route_bounds",
        {"sorted", "params"}, {"bounds", "tiles"}, R"METAL(
        const uint expert = thread_position_in_grid.x;
        if (expert >= params[1]) return;
        const uint count = params[0];
        uint low = 0u, high = count;
        while (low < high) {
            const uint mid = low + (high - low) / 2u;
            if (sorted[mid] < int(expert)) low = mid + 1u; else high = mid;
        }
        const uint first = low;
        high = count;
        while (low < high) {
            const uint mid = low + (high - low) / 2u;
            if (sorted[mid] <= int(expert)) low = mid + 1u; else high = mid;
        }
        bounds[expert * 2u] = int(first);
        bounds[expert * 2u + 1u] = int(low);
        tiles[expert] = int((low - first + params[2] - 1u) / params[2]);
        )METAL");
    auto flat = contiguous(reshape(astype(ids, int32), {-1}));
    auto order = astype(argsort(flat), int32);
    auto ranges = kernel({take(flat, order), array({uint32_t(ids.size()), uint32_t(experts), uint32_t(tile_rows)})},
        {{experts, 2}, {experts}}, {int32, int32}, {((experts + 127) / 128) * 128, 1, 1},
        {128, 1, 1}, {}, std::nullopt, false, {});
    auto offsets = concatenate({zeros({1}, int32), cumsum(ranges[1])});
    return {std::move(order), std::move(ranges[0]), std::move(offsets),
        int(ids.size() / tile_rows + std::size_t(experts)), tile_rows};
}
} // namespace mfq::metal
