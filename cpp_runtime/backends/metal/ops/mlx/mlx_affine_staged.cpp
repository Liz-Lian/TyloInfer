#include "mlx_affine_staged.h"
#include "mlx_cached_kernel.h"
#include "mlx_affine.h"
#include "mlx_affine_indirect.h"
#include "mlx_affine_nax.h"
#include "mlx_indirect_weight_residency.h"
#include "mlx_moe_ops.h"
#include "mlx_platform.h"

#include <algorithm>
#include <climits>

namespace mfq::metal {
namespace {
using namespace mlx::core;

struct Geometry {
    int rows = 64, depth = 64, tile_x = 0, padding = 0;
    bool double_buffer = false;
};

Geometry geometry(int rows, int experts, int width, int output) {
    if (width % 64 || output % 64) return {};
    const double mean = double(rows) / experts;
    if (mean < 36 || (width < 1024 && mean < 120)) return {64, 64, 0, 0, true};
    if (width < 1024) return {96, 128, 32, 0, false};
    if (mean < 48) return {64, 64, 32, 0, true};
    if (mean < 96) return {96, 64, 32, 0, true};
    return {128, 128, 32, 8192, false};
}

const fast::CustomKernelFunction& tile_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_affine_segment_tiles",
        {"ids", "params"}, {"tiles", "count"}, R"METAL(
        threadgroup uint first[MAXE], end[MAXE], sums[32];
        const uint tid = thread_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint rows = params[0], experts = params[1], capacity = params[2];
        for (uint e = tid; e < experts; e += 1024u) first[e] = end[e] = 0u;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid * 4u; i < rows; i += 4096u) {
            uint v[6];
            v[0] = i ? ids[i - 1u] : 0xffffffffu;
            for (uint j = 0; j < 5u; ++j) v[j + 1u] = i + j < rows ? ids[i + j] : 0xffffffffu;
            for (uint j = 0; j < 4u && i + j < rows; ++j) {
                const uint e = v[j + 1u];
                if (e < experts) {
                    if (v[j] != e) first[e] = i + j;
                    if (v[j + 2u] != e) end[e] = i + j + 1u;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint running = 0u;
        for (uint base = 0; base < experts; base += 1024u) {
            const uint e = base + tid;
            const uint start = e < experts ? first[e] : 0u;
            const uint live = e < experts ? end[e] - start : 0u;
            const uint n = (live + BM - 1u) / BM;
            const uint local = simd_prefix_exclusive_sum(n);
            const uint simd_total = simd_sum(n);
            if (!lane) sums[sg] = simd_total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint prefix = 0u, total = 0u;
            for (uint s = 0; s < 32u; ++s) {
                total += sums[s];
                if (s < sg) prefix += sums[s];
            }
            const uint off = running + prefix + local;
            for (uint t = 0; t < n && off + t < capacity; ++t) {
                const uint row = start + t * BM;
                reinterpret_cast<device uint4*>(tiles)[off + t] = uint4(row, e, min(uint(BM), start + live - row), 0u);
            }
            running += total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!tid) count[0] = min(running, capacity);
        )METAL");
    return kernel;
}

const std::string& staged_header() {
    static const std::string header = std::string(kMlxAffineNax) + R"METAL(
    namespace tinfer_affine {
    using namespace mlx::steel;

    template <typename T>
    struct Fields {
        const device uint* values;
        const device T* scales;
        const device T* biases;
        uint expert;
        bool valid;
    };

    template <bool INDIRECT, typename T, typename P, typename S, typename B>
    METAL_FUNC Fields<T> fields(P packed, S scales, B biases, uint expert) {
        if constexpr (INDIRECT) {
            if (!packed[expert] || !scales[expert] || !biases[expert])
                return {nullptr, nullptr, nullptr, 0u, false};
            return {reinterpret_cast<const device uint*>(packed[expert]),
                reinterpret_cast<const device T*>(scales[expert]),
                reinterpret_cast<const device T*>(biases[expert]), 0u, true};
        } else {
            return {packed, scales, biases, expert, true};
        }
    }

    template <typename T, int Q, int GS, int BM, int BK, bool PAIR>
    struct Loader {
        static constant constexpr int THREADS = BM * 2;
        static constant constexpr int LOADERS = BM == 96 ? 128 : THREADS;
        static constant constexpr int VALUES = 64 * BK / LOADERS;
        static constant constexpr int WORDS = VALUES * Q / 32;
        static constant constexpr int GROUP_VALUES = VALUES < GS ? VALUES : GS;
        static constant constexpr int GROUPS = VALUES / GROUP_VALUES;
        static constant constexpr int STRIDE = BK + 16 / sizeof(T);
        const device uint* packed;
        const device T* scales;
        const device T* biases;
        const int row, column, width;
        uint words[WORDS];
        float scale[GROUPS], bias[GROUPS];

        METAL_FUNC Loader(const device uint* w, const device T* s, const device T* b,
            int k, int expert, int outputs, int n0, uint tid) thread
            : packed(w), scales(s), biases(b), row(tid / (BK / VALUES)),
              column((tid % (BK / VALUES)) * VALUES), width(k) {
            const int output_row = PAIR ? n0 / 2 + ((row / 16) & 1) * (outputs / 2)
                + (row / 32) * 16 + row % 16 : n0 + row;
            const ulong wr = ulong(expert) * outputs + output_row;
            packed += wr * (k / 32 * Q) + column / 32 * Q;
            scales += wr * (k / GS);
            biases += wr * (k / GS);
        }

        template <bool FULL_N, bool FULL_K>
        METAL_FUNC void fetch(int k0, int valid, bool live) thread {
            #pragma clang loop unroll(full)
            for (int i = 0; i < WORDS; ++i)
                words[i] = (FULL_N || live) && (FULL_K || column + i * 32 / Q < valid)
                    ? packed[k0 / 32 * Q + i] : 0u;
            #pragma clang loop unroll(full)
            for (int i = 0; i < GROUPS; ++i) {
                const int g = (k0 + column + i * GROUP_VALUES) / GS;
                scale[i] = (FULL_N || live) && (FULL_K || column + i * GROUP_VALUES < valid) ? float(scales[g]) : 0.f;
                bias[i] = (FULL_N || live) && (FULL_K || column + i * GROUP_VALUES < valid) ? float(biases[g]) : 0.f;
            }
        }

        METAL_FUNC void store(threadgroup T* dst) const thread {
            #pragma clang loop unroll(full)
            for (int i = 0; i < VALUES; i += 4) {
                vec<T, 4> values;
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) {
                    const int bit = (i + j) * Q, word = bit / 32, shift = bit % 32;
                    uint code = words[word] >> shift;
                    if (shift + Q > 32) code |= words[word + 1] << (32 - shift);
                    values[j] = T(scale[(i + j) / GROUP_VALUES] * float(code & ((1u << Q) - 1u))
                        + bias[(i + j) / GROUP_VALUES]);
                }
                *reinterpret_cast<threadgroup vec<T, 4>*>(dst + row * STRIDE + column + i) = values;
            }
        }
    };
    }
    )METAL";
    return header;
}

const fast::CustomKernelFunction& projection_kernel(bool indirect = false) {
    static const CompileOptions options;
    static const std::string source = R"METAL(
        using namespace mlx::steel;
        using W = tinfer_affine::Loader<T, Q, GS, BM, BK, PAIR>;
        constexpr int STRIDE = W::STRIDE;
        constexpr int TILE = 64 * STRIDE;
        threadgroup T weights[(DB ? 2 : 1) * TILE + PAD / sizeof(T)];
        const uint outputs = params[0], width = params[1];
        uint tile, slab;
        if constexpr (GX) {
            const uint columns = (outputs + 63u) / 64u;
            tile = (threadgroup_position_in_grid.y / columns) * GX + threadgroup_position_in_grid.x;
            slab = threadgroup_position_in_grid.y % columns;
        } else {
            tile = threadgroup_position_in_grid.y;
            slab = threadgroup_position_in_grid.x;
        }
        if (tile >= count[0]) return;
        const uint4 desc = reinterpret_cast<const device uint4*>(tiles)[tile];
        const auto fields = tinfer_affine::fields<INDIRECT, T>(packed, scales, biases, desc.y);
        if (!fields.valid) return;
        const uint tid = thread_index_in_threadgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint n0 = slab * 64u;
        const uint m0 = (sg / 2u) * 32u;
        const uint nsg = (sg % 2u) * 32u;
        const int live = min(32, int(desc.z) - int(m0));
        const short fragments = live > 0 ? short((live + 15) / 16) : 0;
        const short2 coord = BaseNAXFrag::get_coord();
        uint offsets[2][2];
        #pragma clang loop unroll(full)
        for (short r = 0; r < 2; ++r) {
            #pragma clang loop unroll(full)
            for (short h = 0; h < 2; ++h) {
                const uint row = desc.x + min(m0 + uint(r * 16 + h * 8 + coord.y), desc.z - 1u);
                const uint source = params[3] ? row : order[row] / params[2];
                offsets[r][h] = source * width + uint(coord.x);
            }
        }
        NAXTile<float, 2, 2> accum;
        accum.clear();
        W loader(fields.values, fields.scales, fields.biases, width, fields.expert, outputs, n0, tid);
        const bool loads = tid < uint(W::LOADERS);
        if constexpr (DB) {
            if (loads) {
                loader.template fetch<AN, true>(0, BK, n0 + uint(loader.row) < outputs);
                loader.store(weights);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        for (uint k0 = 0; k0 < width; k0 += BK) {
            const uint valid = min(uint(BK), width - k0);
            const bool more = k0 + BK < width;
            if constexpr (DB) {
                if (more && loads) loader.template fetch<AN, true>(k0 + BK, BK, n0 + uint(loader.row) < outputs);
            } else {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (loads) {
                    loader.template fetch<AN, AK>(k0, valid, n0 + uint(loader.row) < outputs);
                    loader.store(weights);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            const threadgroup T* ws = weights + nsg * STRIDE + (DB ? ((k0 / BK) & 1u) * TILE : 0u);
            #pragma clang loop unroll(disable)
            for (uint kk = 0; kk < valid; kk += 32u) {
                NAXTile<T, 2, 2> b;
                b.template load<T, STRIDE, 1>(ws + kk);
                #pragma clang loop unroll(full)
                for (short r = 0; r < 2; ++r) if (r < fragments) {
                    NAXTile<T, 1, 2> a;
                    #pragma clang loop unroll(full)
                    for (short h = 0; h < 2; ++h) {
                        const device T* xp = x + offsets[r][h] + k0 + kk;
                        const vec<T, 4> left = *reinterpret_cast<const device vec<T, 4>*>(xp);
                        const vec<T, 4> right = *reinterpret_cast<const device vec<T, 4>*>(xp + 16);
                        #pragma clang loop unroll(full)
                        for (short j = 0; j < 4; ++j) {
                            a.frag_at(0, 0)[h * 4 + j] = left[j];
                            a.frag_at(0, 1)[h * 4 + j] = right[j];
                        }
                    }
                    #pragma clang loop unroll(full)
                    for (short k = 0; k < 2; ++k)
                        BaseNAXFrag::mma(accum.frag_at(r, 0), accum.frag_at(r, 1),
                            a.frag_at(0, k), metal::bool_constant<false>{},
                            b.frag_at(0, k), b.frag_at(1, k), metal::bool_constant<true>{});
                }
            }
            if constexpr (DB) {
                if (more && loads) loader.store(weights + (((k0 / BK) + 1u) & 1u) * TILE);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        if constexpr (PAIR) {
            if (live > 0) {
                #pragma clang loop unroll(full)
                for (short r = 0; r < 2; ++r) {
                    #pragma clang loop unroll(full)
                    for (short h = 0; h < 2; ++h) {
                        const uint row = uint(r * 16 + h * 8 + coord.y);
                        if (row < uint(live)) {
                            vec<Y, 4> values;
                            #pragma clang loop unroll(full)
                            for (short j = 0; j < 4; ++j) {
                                const float gate = float(T(accum.frag_at(r, 0)[h * 4 + j]));
                                const float up = float(T(accum.frag_at(r, 1)[h * 4 + j]));
                                values[j] = Y((gate / (1.f + exp(-gate))) * up);
                            }
                            *reinterpret_cast<device vec<Y, 4>*>(y + ulong(desc.x + m0 + row) * (outputs / 2u)
                                + (n0 + nsg) / 2u + uint(coord.x)) = values;
                        }
                    }
                }
            }
        } else if (live > 0) {
            auto dest = y + ulong(desc.x + m0) * outputs + n0 + nsg;
            const short columns = short(min(32, max(0, int(outputs) - int(n0 + nsg))));
            accum.store_safe(dest, outputs, short2(columns, live));
        }
        )METAL";
    static const std::vector<std::string> inputs{"x", "packed", "scales", "biases", "order", "tiles", "count", "params"};
    static const auto direct = cached_metal_kernel("tinfer_affine_segmented_nax", inputs, {"y"},
        source, staged_header(), true, false, options);
    static const auto indexed = indirect_weight_metal_kernel("tinfer_affine_segmented_nax_indirect", inputs, {"y"},
        source, staged_header(), true, false, options);
    return indirect ? indexed : direct;
}
}

std::optional<mlx::core::array> MlxAffineStaged::project(const MlxAffineWeight& weight,
    const mlx::core::array& input, const MlxAffineRoutePlan& plan,
    bool shared_input, bool input_in_route_order, bool swiglu) {
    using namespace mlx::core;
    const int width = weight.width_, output = weight.output_size();
    if (!plan.order_ || plan.tokens_ <= 6 || plan.experts_ > 2048 ||
        width % 32 || output % 32 || (swiglu && output % 64) ||
        (weight.group_size_ != 32 && weight.group_size_ != 64 && weight.group_size_ != 128) ||
        (weight.scales_.dtype() != float16 && weight.scales_.dtype() != bfloat16) ||
        input.size() >= std::uint64_t(1) << 32 || mlx_metal_nax_disabled() ||
        !mlx_apple_chip_starts_with("Apple M5")) return std::nullopt;
    const auto total = std::int64_t(plan.tokens_) * plan.routes_;
    if (total > INT_MAX) return std::nullopt;
    const auto config = geometry(int(total), plan.experts_, width, output);
    const int capacity = int(std::min(total, (total + config.rows - 1) / config.rows + plan.experts_ - 1));
    const int slot = config.rows / 32 - 2;
    if (!plan.tiles_[slot]) {
        int max_experts = 1;
        while (max_experts < plan.experts_) max_experts *= 2;
        auto result = tile_kernel()({plan.ids_, array({int(total), plan.experts_, capacity}, int32)},
            {{capacity, 4}, {1}}, {uint32, uint32}, {1024, 1, 1}, {1024, 1, 1},
            {{"BM", config.rows}, {"MAXE", max_experts}}, std::nullopt, false, {});
        plan.tiles_[slot].emplace(result[0], result[1]);
    }
    const auto& [tiles, count] = *plan.tiles_[slot];
    auto x = reshape(astype(input, weight.scales_.dtype()), {-1, width});
    const auto params = array({output, width, shared_input ? plan.routes_ : 1,
        !shared_input && input_in_route_order ? 1 : 0}, int32);
    const int columns = (output + 63) / 64;
    const int threads = config.rows * 2;
    const std::tuple<int, int, int> grid = config.tile_x
        ? std::tuple{config.tile_x * threads, ((capacity + config.tile_x - 1) / config.tile_x) * columns, 1}
        : std::tuple{columns * threads, capacity, 1};
    const auto result_dtype = swiglu ? input.dtype() : weight.scales_.dtype();
    auto result = projection_kernel()({x, weight.values_, weight.scales_, weight.biases_, *plan.order_, tiles, count, params},
        {{int(total), 1, swiglu ? output / 2 : output}}, {result_dtype}, grid, {threads, 1, 1},
        {{"T", weight.scales_.dtype()}, {"Q", weight.bits_}, {"GS", weight.group_size_},
         {"BM", config.rows}, {"BK", config.depth}, {"DB", config.double_buffer},
         {"GX", config.tile_x}, {"PAD", config.padding}, {"AN", output % 64 == 0},
         {"AK", width % config.depth == 0}, {"PAIR", swiglu}, {"Y", result_dtype}, {"INDIRECT", false}}, std::nullopt, false, {})[0];
    return astype(result, input.dtype());
}

std::optional<mlx::core::array> MlxAffineStaged::project_indirect(const MlxAffineIndirectWeight& weight,
    const mlx::core::array& input, const mlx::core::array& ids,
    bool shared_input, const MlxExpertRouteTiles& plan) {
    using namespace mlx::core;
    const int width = weight.input_size(), output = weight.output_size(), experts = weight.expert_count();
    if (ids.shape(0) <= MlxAffineIndirectWeight::kVectorRows || experts > 2048 || width % 32 || output % 32 ||
        (weight.group_size_ != 32 && weight.group_size_ != 64 && weight.group_size_ != 128) ||
        weight.dtype_ != input.dtype() || (input.dtype() != float16 && input.dtype() != bfloat16) ||
        input.size() >= std::uint64_t(1) << 32 || mlx_metal_nax_disabled() ||
        !mlx_apple_chip_starts_with("Apple M5")) return std::nullopt;
    const auto config = geometry(int(ids.size()), experts, width, output);
    const int capacity = int(std::min(ids.size(), (ids.size() + config.rows - 1) / config.rows + experts - 1));
    if (plan.order.size() != ids.size() || plan.bounds.shape() != Shape{experts, 2})
        throw std::invalid_argument("MLX indirect route plan differs from projection geometry");
    const auto order = view(plan.order, uint32);
    const int slot = config.rows / 32 - 2;
    if (!plan.affine_tiles[slot]) {
        if (!plan.affine_sorted_ids) plan.affine_sorted_ids = take(reshape(view(ids, uint32), {-1}), order, 0);
        int max_experts = 1;
        while (max_experts < experts) max_experts *= 2;
        auto tiles = tile_kernel()({*plan.affine_sorted_ids, array({int(ids.size()), experts, capacity}, int32)},
            {{capacity, 4}, {1}}, {uint32, uint32}, {1024, 1, 1}, {1024, 1, 1},
            {{"BM", config.rows}, {"MAXE", max_experts}}, std::nullopt, false, {});
        plan.affine_tiles[slot].emplace(tiles[0], tiles[1]);
    }
    const auto& [tiles, count] = *plan.affine_tiles[slot];
    const int columns = (output + 63) / 64, threads = config.rows * 2;
    const std::tuple<int, int, int> grid = config.tile_x
        ? std::tuple{config.tile_x * threads, ((capacity + config.tile_x - 1) / config.tile_x) * columns, 1}
        : std::tuple{columns * threads, capacity, 1};
    auto result = projection_kernel(true)({reshape(input, {-1, width}), weight.tables_[0],
        weight.tables_[1], weight.tables_[2], order, tiles, count,
        array({output, width, shared_input ? ids.shape(1) : 1, 0}, int32)},
        {{int(ids.size()), output}}, {input.dtype()}, grid, {threads, 1, 1},
        {{"T", input.dtype()}, {"Y", input.dtype()}, {"Q", weight.bits_}, {"GS", weight.group_size_},
         {"BM", config.rows}, {"BK", config.depth}, {"DB", config.double_buffer},
         {"GX", config.tile_x}, {"PAD", config.padding}, {"AN", output % 64 == 0},
         {"AK", width % config.depth == 0}, {"PAIR", false}, {"INDIRECT", true}}, 0.f, false, {})[0];
    return reshape(moe_restore_routed_rows(result, plan.order, ids),
        {ids.shape(0), ids.shape(1), output});
}
}
