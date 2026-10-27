#include "mlx_gguf_routes.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace mfq::metal {
using namespace mlx::core;
MlxGgufGroupedRoutes gguf_group_routes(const array& ids, int experts, int tile_rows, bool shared) {
    if (ids.ndim() != 2 || ids.size() == 0 || ids.size() > INT_MAX || experts < 1 || experts > 512 ||
        (ids.dtype() != int32 && ids.dtype() != uint32) || tile_rows < 1 || tile_rows > 32)
        throw std::invalid_argument("invalid GGUF grouped route geometry");
    if (shared && ids.size() > std::size_t(INT_MAX - ids.shape(0)))
        throw std::overflow_error("GGUF shared routes are too large");
    const auto tiles = std::min(ids.size(), ids.size() / tile_rows + std::size_t(experts)) +
        (shared ? (ids.shape(0) + tile_rows - 1) / tile_rows : 0);
    if (tiles > INT_MAX / tile_rows) throw std::overflow_error("GGUF grouped routes are too large");
    static const auto kernel = fast::metal_kernel("tinfer_gguf_group_routes",
        {"ids", "params"}, {"descriptors", "count", "offsets", "grouped", "positions"}, R"METAL(
uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
uint routes = params[0], experts = params[1], rows = params[2], selected = params[3], shared = params[4];
threadgroup atomic_uint counts[SLOTS], cursors[SLOTS];
threadgroup uint starts[SLOTS], totals[SLOTS / 32];
atomic_store_explicit(&counts[tid], 0u, memory_order_relaxed);
atomic_store_explicit(&cursors[tid], 0u, memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint route = tid; route < routes; route += SLOTS) {
    int e = int(ids[route]);
    if (e >= 0 && uint(e) < experts) atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint live = atomic_load_explicit(&counts[tid], memory_order_relaxed);
uint tiles = (live + rows - 1u) / rows, before = simd_prefix_exclusive_sum(tiles);
if (lane == 31u) totals[sg] = before + tiles;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0u; i < sg; ++i) before += totals[i];
starts[tid] = before;
if (tid < experts) offsets[tid] = int(before);
for (uint tile = 0u; tile < tiles; ++tile) {
    descriptors[(before + tile) * 2u] = int(tid);
    descriptors[(before + tile) * 2u + 1u] = int(min(rows, live - tile * rows));
}
for (uint pad = live; pad < tiles * rows; ++pad) grouped[before * rows + pad] = -1;
if (tid == SLOTS - 1u) {
    uint routed_tiles = before + tiles, tokens = routes / selected;
    uint shared_tiles = shared ? (tokens + rows - 1u) / rows : 0u;
    count[0] = int(routed_tiles + shared_tiles);
    offsets[experts] = int(routed_tiles);
    if (shared) offsets[experts + 1u] = int(routed_tiles + shared_tiles);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint route = tid; route < routes; route += SLOTS) {
    uint original = shared ? route / selected * (selected + 1u) + route % selected : route;
    int e = int(ids[route]);
    if (e < 0 || uint(e) >= experts) { positions[original] = -1; continue; }
    uint slot = atomic_fetch_add_explicit(&cursors[e], 1u, memory_order_relaxed);
    uint row = starts[e] * rows + slot;
    grouped[row] = int(original);
    positions[original] = int(row);
}
if (shared) {
    uint base = 0u, tokens = routes / selected;
    for (uint i = 0u; i < SLOTS / 32; ++i) base += totals[i];
    uint shared_tiles = (tokens + rows - 1u) / rows;
    for (uint tile = tid; tile < shared_tiles; tile += SLOTS) {
        descriptors[(base + tile) * 2u] = int(experts);
        descriptors[(base + tile) * 2u + 1u] = int(min(rows, tokens - tile * rows));
    }
    for (uint token = tid; token < shared_tiles * rows; token += SLOTS) {
        uint row = base * rows + token;
        int original = token < tokens ? int(token * (selected + 1u) + selected) : -1;
        grouped[row] = original;
        if (original >= 0) positions[original] = int(row);
    }
}
)METAL");
    const int threads = experts <= 256 ? 256 : 512;
    auto result = kernel({contiguous(reshape(ids, {-1})), array({uint32_t(ids.size()), uint32_t(experts), uint32_t(tile_rows),
        uint32_t(ids.shape(1)), uint32_t(shared)})},
        {{int(tiles), 2}, {1}, {experts + 1 + int(shared)}, {int(tiles) * tile_rows}, {int(ids.size()) + (shared ? ids.shape(0) : 0)}},
        {int32, int32, int32, int32, int32}, {threads, 1, 1}, {threads, 1, 1},
        {{"SLOTS", threads}}, std::nullopt, false, {});
    return {std::move(result[0]), std::move(result[1]), std::move(result[2]), std::move(result[3]),
        std::move(result[4]), int(tiles), tile_rows};
}
}
