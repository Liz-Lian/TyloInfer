#include "mlx_sparse_attention.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;
#include "../../kernels/common/mfq_sparse_block_nax.inc"

const fast::CustomKernelFunction& independent_rows() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_independent_rows",
        {"q", "k", "v", "sel", "counts", "params", "scale"}, {"out"},
        std::string("if (counts[uint(threadgroup_position_in_grid.x) / 8u] <= uint(params[3])) return;\n") +
            kSparseBlockNaxSource,
        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
        "#define UNROLL _Pragma(\"clang loop unroll(full)\")\n#define TYLOINFER_QSA_PREFILL 0\n",
        false, false, options);
    return kernel;
}

const fast::CustomKernelFunction& pack_queries() {
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_pack_queries",
        {"q", "params"}, {"packed"}, R"METAL(
uint chunk = thread_position_in_grid.x;
uint rows = params[0], tiles = (rows + 7u) / 8u;
if (chunk >= 2u * tiles * 96u * 32u) return;
uint column = (chunk % 32u) * 8u, fused = (chunk / 32u) % 96u;
uint tile = (chunk / (32u * 96u)) % tiles, kv = chunk / (32u * 96u * tiles);
uint row = tile * 8u + fused / 12u, head = kv * 12u + fused % 12u;
for (uint i = 0u; i < 8u; ++i)
    packed[ulong(chunk) * 8u + i] = row < rows ? q[ulong(head) * q_strides[1] + ulong(row) * q_strides[2] + (column + i) * q_strides[3]] : T(0);
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& mark_blocks() {
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_mark_blocks",
        {"selected", "params"}, {"mask"}, R"METAL(
uint tile = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
uint rows = params[0], offset = params[2], blocks = params[3];
for (uint at = tid; at < 8u * 513u; at += 256u) {
    uint row = tile * 8u + at / 513u, slot = at % 513u;
    if (row >= rows) continue;
    uint visible = offset + row + 1u, complete = visible / 4u;
    int block = -1;
    if (slot < min(512u, complete)) block = selected[ulong(row) * 512u + slot];
    else if (slot == 512u && visible % 4u) block = int(complete);
    if (block >= 0 && uint(block) < blocks)
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint*>(mask) + ulong(tile) * blocks + uint(block),
            1u << (row % 8u), memory_order_relaxed);
}
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& list_blocks() {
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_list_blocks",
        {"mask", "params"}, {"lists", "counts"}, R"METAL(
uint tile = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
uint blocks = params[0], stride = params[1], per = (blocks + 255u) / 256u;
uint begin = min(tid * per, blocks), end = min(begin + per, blocks), count = 0u;
for (uint block = begin; block < end; ++block) count += mask[ulong(tile) * blocks + block] != 0u;
uint inclusive = simd_prefix_inclusive_sum(count);
threadgroup uint totals[8];
if (lane == 31u) totals[sg] = inclusive;
threadgroup_barrier(mem_flags::mem_threadgroup);
uint before = inclusive - count;
for (uint i = 0u; i < sg; ++i) before += totals[i];
for (uint block = begin; block < end; ++block) {
    uint bits = mask[ulong(tile) * blocks + block];
    if (bits) lists[ulong(tile) * stride + before++] = (block << 8u) | bits;
}
if (tid == 255u) counts[tile] = before;
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& dense_blocks() {
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_dense_blocks",
        {"params"}, {"lists", "counts"}, R"METAL(
uint tile = threadgroup_position_in_grid.x, tid = thread_index_in_threadgroup;
uint rows = params[0], offset = params[1], stride = params[2];
uint count = (offset + min(rows, tile * 8u + 8u) + 3u) / 4u;
for (uint block = tid; block < count; block += 256u)
    lists[ulong(tile) * stride + block] = (block << 8u) | 255u;
if (tid == 0u) counts[tile] = count;
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& attend_tiles() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_attention_tiles",
        {"queries", "k", "v", "lists", "counts", "params", "scale"}, {"partials", "statistics"}, R"METAL(
using namespace mpp::tensor_ops;
constexpr ushort M = 96, N = 16, D = 256;
uint kv = threadgroup_position_in_grid.x, tile = threadgroup_position_in_grid.y;
uint tid = thread_index_in_threadgroup, rows = params[0], keys = params[1], offset = params[2], stride = params[3];
const float factor = scale[0];
uint tiles = (rows + 7u) / 8u, row0 = tile * 8u, active = min(8u, rows - row0);
uint count = counts[tile];
if constexpr (FALLBACK) if (count > params[4]) return;
threadgroup uint entries[4];
alignas(16) threadgroup T stage[N * D];
alignas(16) threadgroup float scores[M * N];
alignas(16) threadgroup T probabilities[M * N];
threadgroup float maxima[M], denominators[M], previous_scale[M];
threadgroup atomic_uint rescale;
if (tid < M) { maxima[tid] = -INFINITY; denominators[tid] = 0.f; }
auto qt = tensor(const_cast<device T*>(queries) + (ulong(kv) * tiles + tile) * M * D,
    dextents<int, 2>{D, M}, array<int, 2>{1, D});
auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
tensor<threadgroup T, dextents<int, 2>, tensor_inline> kt(stage, dextents<int, 2>{D, N}, array<int, 2>{1, D});
auto q0 = qt.template slice<D, M>(0, 0);
auto p0 = pt.template slice<N, M>(0, 0);
auto k0 = kt.template slice<D, N>(0, 0);
constexpr auto qk_desc = matmul2d_descriptor(M, N, D, false, true, false, matmul2d_descriptor::mode::multiply);
constexpr auto pv_desc = matmul2d_descriptor(M, D, N, false, false, true, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<qk_desc, execution_simdgroups<8>> qk;
matmul2d<pv_desc, execution_simdgroups<8>> pv;
auto running = pv.template get_destination_cooperative_tensor<decltype(p0), decltype(k0), float>();
const bool full = uint(running.get_capacity()) * 256u == uint(M) * D;
#pragma clang loop unroll(full)
for (ushort i = 0; i < running.get_capacity(); ++i) if (full || running.is_valid_element(i)) running[i] = 0.f;
for (uint step = 0u; step < count; step += 4u) {
    if (tid < 4u) entries[tid] = step + tid < count ? lists[ulong(tile) * stride + step + tid] : 0xffffffffu;
    if (tid == 0u) atomic_store_explicit(&rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint chunk = tid; chunk < N * D / 8u; chunk += 256u) {
        uint token = chunk / 32u, column = (chunk % 32u) * 8u, entry = entries[token / 4u];
        uint position = (entry >> 8u) * 4u + token % 4u;
        uint4 data = 0u;
        if (entry != 0xffffffffu && position < keys)
            data = *reinterpret_cast<device const uint4*>(k + ulong(kv) * k_strides[1] + ulong(position) * k_strides[2] + column);
        *reinterpret_cast<threadgroup uint4*>(stage + token * D + column) = data;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    auto products = qk.template get_destination_cooperative_tensor<decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, products);
    products.store(st.template slice<N, M>(0, 0));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint chunk = tid; chunk < N * D / 8u; chunk += 256u) {
        uint token = chunk / 32u, column = (chunk % 32u) * 8u, entry = entries[token / 4u];
        uint position = (entry >> 8u) * 4u + token % 4u;
        uint4 data = 0u;
        if (entry != 0xffffffffu && position < keys)
            data = *reinterpret_cast<device const uint4*>(v + ulong(kv) * v_strides[1] + ulong(position) * v_strides[2] + column);
        *reinterpret_cast<threadgroup uint4*>(stage + token * D + column) = data;
    }
    if (tid < M * 2u) {
        uint fused = tid / 2u, column = (tid % 2u) * 8u;
        uint row = min(fused / 12u, active - 1u), causal = offset + row0 + row + 1u;
        float values[8], maximum = -INFINITY;
        uint allowed = 0u;
#pragma clang loop unroll(full)
        for (uint block = 0u; block < 2u; ++block) {
            const uint entry = entries[column / 4u + block], position = (entry >> 8u) * 4u;
            const uint end = min(causal, keys);
            if (entry != 0xffffffffu && ((entry >> row) & 1u) && position < end)
                allowed |= ((1u << min(4u, end - position)) - 1u) << (block * 4u);
        }
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
            const float score = scores[fused * N + column + i];
            values[i] = (allowed >> i) & 1u ? score * factor : -INFINITY;
            maximum = max(maximum, values[i]);
        }
        maximum = max(maximum, simd_shuffle_xor(maximum, 1u));
        float old_maximum = maxima[fused], next_maximum = max(old_maximum, maximum), total = 0.f;
#pragma clang loop unroll(full)
        for (uint i = 0u; i < 8u; ++i) {
            float probability = (allowed >> i) & 1u ? fast::exp(values[i] - next_maximum) : 0.f;
            probabilities[fused * N + column + i] = T(probability);
            total += probability;
        }
        total += simd_shuffle_xor(total, 1u);
        if (column == 0u) {
            float alpha = next_maximum == -INFINITY || next_maximum == old_maximum ? 1.f : fast::exp(old_maximum - next_maximum);
            previous_scale[fused] = alpha;
            denominators[fused] = denominators[fused] * alpha + total;
            maxima[fused] = next_maximum;
            if (alpha != 1.f) atomic_store_explicit(&rescale, 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(&rescale, memory_order_relaxed)) {
#pragma clang loop unroll(full)
        for (ushort i = 0; i < running.get_capacity(); ++i) if (full || running.is_valid_element(i)) {
            auto at = running.get_multidimensional_index(i);
            running[i] *= previous_scale[at[1]];
        }
    }
    pv.run(p0, k0, running);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const ulong base = (ulong(kv) * tiles + tile) * M;
auto target = tensor(partials + base * D, dextents<int, 2>{D, M}, array<int, 2>{1, D});
running.store(target.template slice<D, M>(0, 0));
if (tid < M) {
    statistics[(base + tid) * 2u] = maxima[tid];
    statistics[(base + tid) * 2u + 1u] = denominators[tid];
}
)METAL", "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n", false, false, options);
    return kernel;
}

const fast::CustomKernelFunction& normalize_tiles() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = fast::metal_kernel("tinfer_sparse_prefill_normalize_tiles",
        {"partials", "statistics", "params", "counts", "fallback"}, {"out"}, R"METAL(
uint chunk = thread_position_in_grid.x, rows = params[0], tiles = (rows + 7u) / 8u;
if (chunk >= rows * 24u * 32u) return;
uint row = chunk / (24u * 32u), head = (chunk / 32u) % 24u, column = (chunk % 32u) * 8u;
if constexpr (FALLBACK) if (counts[row / 8u] > params[4]) {
    for (uint i = 0u; i < 8u; ++i) out[ulong(chunk) * 8u + i] = T(fallback[ulong(chunk) * 8u + i]);
    return;
}
ulong fused = (ulong(head / 12u) * tiles + row / 8u) * 96u + (row % 8u) * 12u + head % 12u;
float denominator = statistics[fused * 2u + 1u];
float inverse = denominator > 0.f ? 1.f / denominator : 0.f;
for (uint i = 0u; i < 8u; ++i)
    out[ulong(chunk) * 8u + i] = T(partials[fused * 256u + column + i] * inverse);
)METAL", "", true, false, options);
    return kernel;
}
} // namespace

array mlx_sparse_block_gqa_prefill_tiles(const array& query, const array& key, const array& value,
    const std::optional<array>& selected, int query_offset, float scale, Dtype output_dtype) {
    if (query.ndim() != 4 || key.ndim() != 4 || value.shape() != key.shape() ||
        query.shape(0) != 1 || key.shape(0) != 1 || query.shape(1) != 24 || key.shape(1) != 2 ||
        query.shape(3) != 256 || key.shape(3) != 256 || query.shape(2) < 1 ||
        (output_dtype != float16 && output_dtype != bfloat16 && output_dtype != float32) ||
        (query.dtype() != float16 && query.dtype() != bfloat16) || key.dtype() != query.dtype() ||
        value.dtype() != query.dtype() || key.strides(3) != 1 || value.strides(3) != 1 ||
        key.offset() % 16 || value.offset() % 16 || key.strides(1) % 8 || key.strides(2) % 8 ||
        value.strides(1) % 8 || value.strides(2) % 8 || !std::isfinite(scale) ||
        (selected && (selected->shape() != Shape{1, query.shape(2), 512} || selected->dtype() != int32)) ||
        (!selected && std::int64_t(query_offset) + query.shape(2) > 2048) ||
        query_offset < 0 || query_offset > key.shape(2) - query.shape(2))
        throw std::invalid_argument("unsupported tiled sparse GQA prefill geometry");
    const int rows = query.shape(2), keys = key.shape(2), tiles = (rows - 1) / 8 + 1;
    const int blocks = (keys - 1) / 4 + 1, stride = std::min(blocks, 8 * 513);
    constexpr int maximum_shared_blocks = 512 * 16 / 12;
    const bool fallback_enabled = selected && blocks > maximum_shared_blocks;
    if (std::size_t(tiles) * blocks > INT_MAX || std::size_t(tiles) * 2 * 96 * 256 > INT_MAX)
        throw std::overflow_error("tiled sparse GQA prefill workspace is too large");
    const auto params = array({uint32_t(rows), uint32_t(keys), uint32_t(query_offset), uint32_t(blocks),
        uint32_t(maximum_shared_blocks)});
    auto packed = pack_queries()({query, params}, {{2, tiles * 96, 256}}, {query.dtype()},
        {2 * tiles * 96 * 32, 1, 1}, {256, 1, 1}, {{"T", query.dtype()}}, std::nullopt, false, {}).front();
    auto listed = [&] {
        if (!selected) return dense_blocks()({array({uint32_t(rows), uint32_t(query_offset), uint32_t(stride)})},
            {{tiles, stride}, {tiles}}, {uint32, uint32}, {tiles * 256, 1, 1}, {256, 1, 1}, {}, std::nullopt, false, {});
        auto mask = mark_blocks()({contiguous(*selected), params}, {{tiles, blocks}}, {uint32},
            {tiles * 256, 1, 1}, {256, 1, 1}, {}, 0.f, false, {}).front();
        return list_blocks()({mask, array({uint32_t(blocks), uint32_t(stride)})},
            {{tiles, stride}, {tiles}}, {uint32, uint32}, {tiles * 256, 1, 1}, {256, 1, 1}, {}, std::nullopt, false, {});
    }();
    auto fallback = fallback_enabled ? independent_rows()({query, key, value, contiguous(*selected), listed[1],
        array({query_offset, keys, rows, maximum_shared_blocks}), array({scale})},
        {{1, rows, 24, 256}}, {query.dtype()}, {rows * 64, 2, 1}, {64, 1, 1},
        {{"T", query.dtype()}, {"GQA", 12}, {"TOPK", 512}}, std::nullopt, false, {}).front()
        : zeros({1}, query.dtype());
    auto attended = attend_tiles()({packed, key, value, listed[0], listed[1],
        array({uint32_t(rows), uint32_t(keys), uint32_t(query_offset), uint32_t(stride),
            uint32_t(maximum_shared_blocks)}), array({scale})},
        {{2, tiles, 96, 256}, {2, tiles, 96, 2}}, {float32, float32}, {2 * 256, tiles, 1}, {256, 1, 1},
        {{"T", query.dtype()}, {"FALLBACK", fallback_enabled}}, std::nullopt, false, {});
    return normalize_tiles()({attended[0], attended[1], params, listed[1], fallback},
        {{1, rows, 24, 256}}, {output_dtype}, {rows * 24 * 32, 1, 1}, {256, 1, 1},
        {{"T", output_dtype}, {"FALLBACK", fallback_enabled}}, std::nullopt, false, {}).front();
}
} // namespace mfq::metal
