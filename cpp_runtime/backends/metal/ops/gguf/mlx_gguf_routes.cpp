#include "mlx_gguf_routes.h"
#include "mlx_cached_kernel.h"
#include <algorithm>
#include <climits>
#include <stdexcept>

namespace mfq::metal {
using namespace mlx::core;
MlxGgufGroupedRoutes gguf_group_routes(const array& ids, int experts, int tile_rows, bool shared, bool compact) {
    if (ids.ndim() != 2 || ids.size() == 0 || ids.size() > INT_MAX || experts < 1 || experts > 512 ||
        (ids.dtype() != int32 && ids.dtype() != uint32) || tile_rows < 1 || (tile_rows > 32 && tile_rows != 48))
        throw std::invalid_argument("invalid GGUF grouped route geometry");
    if (compact && tile_rows != 8 && tile_rows != 32 && tile_rows != 48)
        throw std::invalid_argument("invalid compact GGUF route tile");
    if (shared && ids.size() > std::size_t(INT_MAX - ids.shape(0)))
        throw std::overflow_error("GGUF shared routes are too large");
    const auto tiles = std::min(ids.size(), ids.size() / tile_rows + std::size_t(experts)) +
        (shared ? (ids.shape(0) + tile_rows - 1) / tile_rows : 0);
    if (tiles > INT_MAX / tile_rows) throw std::overflow_error("GGUF grouped routes are too large");
    const auto tail_padding = tile_rows == 8 ? 7u : 15u;
    const auto capacity = compact ? std::min(tiles * tile_rows, ids.size() +
        std::min(ids.size(), std::size_t(experts)) * tail_padding +
        (shared ? std::size_t(ids.shape(0)) + tail_padding : 0u)) : tiles * tile_rows;
    static const auto kernel = cached_metal_kernel("tinfer_gguf_group_routes",
        {"ids", "params"}, {"descriptors", "count", "offsets", "grouped", "positions"}, R"METAL(
uint tid = thread_index_in_threadgroup, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
uint routes = params[0], experts = params[1], rows = params[2], selected = params[3], shared = params[4];
threadgroup atomic_uint counts[SLOTS], cursors[SLOTS];
threadgroup uint starts[SLOTS], totals[SLOTS / 32];
threadgroup uint row_starts[COMPACT ? SLOTS : 1], row_totals[COMPACT ? SLOTS / 32 : 1];
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
uint reserved = tiles * rows, row_before = 0u;
if constexpr (COMPACT) {
    uint tail = tiles ? live - (tiles - 1u) * rows : 0u;
    uint padded_tail = tail == 0u ? 0u : rows == 8u || tail <= 8u ? 8u : tail <= 16u ? 16u : tail <= 32u ? 32u : 48u;
    reserved = tiles ? (tiles - 1u) * rows + padded_tail : 0u;
    row_before = simd_prefix_exclusive_sum(reserved);
    if (lane == 31u) row_totals[sg] = row_before + reserved;
}
if (lane == 31u) totals[sg] = before + tiles;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = 0u; i < sg; ++i) before += totals[i];
if constexpr (COMPACT) {
    for (uint i = 0u; i < sg; ++i) row_before += row_totals[i];
    row_starts[tid] = row_before;
} else row_before = before * rows;
starts[tid] = before;
if (tid < experts) offsets[tid] = int(before);
for (uint tile = 0u; tile < tiles; ++tile) {
    descriptors[(before + tile) * (COMPACT ? 3u : 2u)] = int(tid);
    descriptors[(before + tile) * (COMPACT ? 3u : 2u) + 1u] = int(min(rows, live - tile * rows));
    if constexpr (COMPACT) descriptors[(before + tile) * 3u + 2u] = int(row_before + tile * rows);
}
for (uint pad = live; pad < reserved; ++pad) grouped[row_before + pad] = -1;
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
    uint row = (COMPACT ? row_starts[e] : starts[e] * rows) + slot;
    grouped[row] = int(original);
    positions[original] = int(row);
}
if (shared) {
    uint base = 0u, row_base = 0u, tokens = routes / selected;
    for (uint i = 0u; i < SLOTS / 32; ++i) base += totals[i];
    if constexpr (COMPACT) for (uint i = 0u; i < SLOTS / 32; ++i) row_base += row_totals[i];
    else row_base = base * rows;
    uint shared_tiles = (tokens + rows - 1u) / rows;
    for (uint tile = tid; tile < shared_tiles; tile += SLOTS) {
        descriptors[(base + tile) * (COMPACT ? 3u : 2u)] = int(experts);
        descriptors[(base + tile) * (COMPACT ? 3u : 2u) + 1u] = int(min(rows, tokens - tile * rows));
        if constexpr (COMPACT) descriptors[(base + tile) * 3u + 2u] = int(row_base + tile * rows);
    }
    uint shared_rows = shared_tiles * rows;
    if constexpr (COMPACT) {
        uint tail = tokens - (shared_tiles - 1u) * rows;
        shared_rows = (shared_tiles - 1u) * rows + (rows == 8u || tail <= 8u ? 8u : tail <= 16u ? 16u : tail <= 32u ? 32u : 48u);
    }
    for (uint token = tid; token < shared_rows; token += SLOTS) {
        uint row = row_base + token;
        int original = token < tokens ? int(token * (selected + 1u) + selected) : -1;
        grouped[row] = original;
        if (original >= 0) positions[original] = int(row);
    }
}
)METAL");
    const int threads = experts <= 256 ? 256 : 512;
    auto result = kernel({contiguous(reshape(ids, {-1})), array({uint32_t(ids.size()), uint32_t(experts), uint32_t(tile_rows),
        uint32_t(ids.shape(1)), uint32_t(shared)})},
        {{int(tiles), compact ? 3 : 2}, {1}, {experts + 1 + int(shared)}, {int(capacity)}, {int(ids.size()) + (shared ? ids.shape(0) : 0)}},
        {int32, int32, int32, int32, int32}, {threads, 1, 1}, {threads, 1, 1},
        {{"SLOTS", threads}, {"COMPACT", compact}}, std::nullopt, false, {});
    return {std::move(result[0]), std::move(result[1]), std::move(result[2]), std::move(result[3]),
        std::move(result[4]), int(tiles), tile_rows, 0, int(capacity)};
}
}
