#include "mlx_affine_indirect.h"
#include "mlx_affine_staged.h"
#include "mlx_buffer_view.h"
#include "mlx_indirect_weight_residency.h"
#include "mlx_platform.h"
#include "mlx_moe_ops.h"
#include "mlx_affine_nax.h"
#include "formats/mlx_affine.h"
#include "../../kernels/mlx/mlx_affine_decode.h"

#include <climits>

namespace mfq::metal {
namespace {
using namespace mlx::core;

const fast::CustomKernelFunction& projection_kernel() {
    static const CompileOptions options(static_cast<int>(MathMode::Fast));
    static const auto kernel = indirect_weight_metal_kernel("tinfer_affine_indirect_qmv",
        {"values", "scales", "biases", "x", "ids", "params"}, {"y"}, R"METAL(
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u + lane / 8u;
        const uint route = threadgroup_position_in_grid.y;
        const uint width = params[0], outputs = params[1];
        if (row >= outputs) return;
        const int expert = ids[route];
        float result = 0.0f;
        if (expert >= 0 && uint(expert) < params[2] && values[expert] && scales[expert] && biases[expert]) {
            const auto v = reinterpret_cast<device const uint*>(values[expert]);
            const auto s = reinterpret_cast<device const S*>(scales[expert]);
            const auto b = reinterpret_cast<device const S*>(biases[expert]);
            const uint input_row = params[4] ? route / params[3] : route;
            const auto source = x + ulong(input_row) * width;
            for (uint group = 0; group < width / GS; ++group) {
                float dot_value = 0.0f, input_sum = 0.0f;
                for (uint local = (lane & 7u) * 4u; local < uint(GS); local += 32u) {
                    const uint column = group * GS + local;
                    const float4 activation(source[column], source[column + 1u], source[column + 2u], source[column + 3u]);
                    const float4 code = float4(tinfer_affine_codes4<Q>(v, row, column, width));
                    dot_value += dot(activation, code);
                    input_sum += activation.x + activation.y + activation.z + activation.w;
                }
                const uint index = row * (width / GS) + group;
                result = fma(float(s[index]), dot_value, fma(float(b[index]), input_sum, result));
            }
        }
        result += simd_shuffle_down(result, 4u);
        result += simd_shuffle_down(result, 2u);
        result += simd_shuffle_down(result, 1u);
        if ((lane & 7u) == 0u) y[ulong(route) * outputs + row] = T(result);
        )METAL", detail::kMlxAffineDecode, true, false, options);
    return kernel;
}

const fast::CustomKernelFunction& gate_up_kernel() {
    static const CompileOptions options(static_cast<int>(MathMode::Fast));
    static const auto kernel = indirect_weight_metal_kernel("tinfer_affine_indirect_gate_up",
        {"gv", "gs", "gb", "uv", "us", "ub", "x", "ids", "params"}, {"y"}, R"METAL(
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x * 16u + simdgroup_index_in_threadgroup * 4u + lane / 8u;
        const uint route = threadgroup_position_in_grid.y;
        const uint width = params[0], outputs = params[1];
        if (row >= outputs) return;
        const int expert = ids[route];
        float gate = 0.0f, up = 0.0f;
        if (expert >= 0 && uint(expert) < params[2] && gv[expert] && gs[expert] && gb[expert] &&
            uv[expert] && us[expert] && ub[expert]) {
            const auto gate_values = reinterpret_cast<device const uint*>(gv[expert]);
            const auto gate_scales = reinterpret_cast<device const S*>(gs[expert]);
            const auto gate_biases = reinterpret_cast<device const S*>(gb[expert]);
            const auto up_values = reinterpret_cast<device const uint*>(uv[expert]);
            const auto up_scales = reinterpret_cast<device const S*>(us[expert]);
            const auto up_biases = reinterpret_cast<device const S*>(ub[expert]);
            const auto source = x + ulong(route / params[3]) * width;
            for (uint group = 0; group < width / GS; ++group) {
                float gate_dot = 0.0f, up_dot = 0.0f, input_sum = 0.0f;
                for (uint local = (lane & 7u) * 4u; local < uint(GS); local += 32u) {
                    const uint column = group * GS + local;
                    const float4 activation(source[column], source[column + 1u], source[column + 2u], source[column + 3u]);
                    const float4 gate_code = float4(tinfer_affine_codes4<Q>(gate_values, row, column, width));
                    const float4 up_code = float4(tinfer_affine_codes4<Q>(up_values, row, column, width));
                    gate_dot += dot(activation, gate_code); up_dot += dot(activation, up_code);
                    input_sum += activation.x + activation.y + activation.z + activation.w;
                }
                const uint index = row * (width / GS) + group;
                gate = fma(float(gate_scales[index]), gate_dot, fma(float(gate_biases[index]), input_sum, gate));
                up = fma(float(up_scales[index]), up_dot, fma(float(up_biases[index]), input_sum, up));
            }
        }
        gate += simd_shuffle_down(gate, 4u); up += simd_shuffle_down(up, 4u);
        gate += simd_shuffle_down(gate, 2u); up += simd_shuffle_down(up, 2u);
        gate += simd_shuffle_down(gate, 1u); up += simd_shuffle_down(up, 1u);
        if ((lane & 7u) == 0u) {
            const float g = float(T(gate)), u = float(T(up));
            y[ulong(route) * outputs + row] = T((g / (1.0f + exp(-g))) * u);
        }
        )METAL", detail::kMlxAffineDecode, true, false, options);
    return kernel;
}

const fast::CustomKernelFunction& prefill_kernel(bool nax) {
    static const CompileOptions options(static_cast<int>(MathMode::Fast));
    static const std::string setup = R"METAL(
        const uint block = threadgroup_position_in_grid.y;
        const uint width = params[0], outputs = params[1], experts = params[2];
        if (block >= uint(offsets[experts])) return;
        uint low = 0, high = experts;
        while (low < high) {
            const uint mid = low + (high - low) / 2;
            if (uint(offsets[mid + 1u]) <= block) low = mid + 1u; else high = mid;
        }
        const uint expert = low;
        if (!values[expert] || !scales[expert] || !biases[expert]) return;
        const auto v = reinterpret_cast<device const uint*>(values[expert]);
        const auto s = reinterpret_cast<device const S*>(scales[expert]);
        const auto b = reinterpret_cast<device const S*>(biases[expert]);
        const uint first = uint(bounds[expert * 2u]);
        const uint rows = uint(bounds[expert * 2u + 1u]) - first;
        const uint m0 = (block - uint(offsets[expert])) * 16u;
    )METAL";
    static const auto ax = indirect_weight_metal_kernel("tinfer_affine_indirect_nax",
        {"values", "scales", "biases", "x", "order", "bounds", "offsets", "params"}, {"y"}, setup + R"METAL(
        const short2 coordinate = mlx::steel::BaseNAXFrag::get_coord();
        const bool upper = (thread_index_in_simdgroup & 1u) != 0u;
        const uint n0 = threadgroup_position_in_grid.x * 128u + simdgroup_index_in_threadgroup * 32u;
        mlx::steel::NAXTile<float, 1, 2> accumulator;
        accumulator.clear();
        for (uint k0 = 0; k0 < width; k0 += 16u) {
            mlx::steel::NAXTile<T, 1, 1> activation;
            mlx::steel::NAXTile<T, 2, 1> weight;
            for (uint r = 0; r < 2u; ++r) {
                const uint row = m0 + uint(coordinate.y) + r * 8u;
                const uint route = row < rows ? uint(order[first + row]) : 0u;
                const uint input_row = params[4] ? route / params[3] : route;
                const uint k = k0 + uint(coordinate.x);
                for (uint j = 0; j < 4u; ++j)
                    activation.frag_at(0, 0)[r * 4u + j] = row < rows ? x[ulong(input_row) * width + k + j] : T(0);
            }
            for (uint tile = 0; tile < 2u; ++tile) {
                const uint row = n0 + tile * 16u + uint(coordinate.y) + (upper ? 8u : 0u);
                const uint k = k0 + (uint(coordinate.x) & ~7u);
                vec<T, 4> first_values(T(0)), second_values(T(0));
                if (row < outputs) for (uint j = 0; j < 4u; ++j) {
                    first_values[j] = T(tinfer_affine_value<Q, GS>(v, s, b, row, k + j, width));
                    second_values[j] = T(tinfer_affine_value<Q, GS>(v, s, b, row, k + 4u + j, width));
                }
                const float4 peer_first = simd_shuffle_xor(float4(first_values), 1u);
                const float4 peer_second = simd_shuffle_xor(float4(second_values), 1u);
                const float4 left = upper ? peer_second : float4(first_values);
                const float4 right = upper ? float4(second_values) : peer_first;
                for (uint j = 0; j < 4u; ++j) {
                    weight.frag_at(tile, 0)[j] = T(left[j]); weight.frag_at(tile, 0)[4u + j] = T(right[j]);
                }
            }
            mlx::steel::tile_matmad_nax(accumulator, activation, metal::bool_constant<false>{}, weight, metal::bool_constant<true>{});
        }
        for (uint tile = 0; tile < 2u; ++tile) for (uint r = 0; r < 2u; ++r) {
            const uint row = m0 + uint(coordinate.y) + r * 8u;
            const uint column = n0 + tile * 16u + uint(coordinate.x);
            if (row < rows) for (uint j = 0; j < 4u; ++j) if (column + j < outputs)
                y[ulong(order[first + row]) * outputs + column + j] = T(accumulator.frag_at(0, tile)[r * 4u + j]);
        }
        )METAL", std::string(kMlxAffineNax) + detail::kMlxAffineDecode, true, false, options);
    static const auto simd = indirect_weight_metal_kernel("tinfer_affine_indirect_simd_mm",
        {"values", "scales", "biases", "x", "order", "bounds", "offsets", "params"}, {"y"}, setup + R"METAL(
        threadgroup half a[16 * 32], weight[32 * 32];
        threadgroup float output[16 * 32];
        const uint tid = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
        const uint n0 = threadgroup_position_in_grid.x * 32u;
        simdgroup_float8x8 c0(0.f), c1(0.f);
        for (uint k0 = 0; k0 < width; k0 += 32u) {
            for (uint index = tid; index < 16u * 32u; index += 128u) {
                const uint row = m0 + index / 32u, k = k0 + index % 32u;
                const uint route = row < rows ? uint(order[first + row]) : 0u;
                const uint input_row = params[4] ? route / params[3] : route;
                a[index] = row < rows ? half(x[ulong(input_row) * width + k]) : half(0);
            }
            for (uint index = tid; index < 32u * 32u; index += 128u) {
                const uint row = n0 + index % 32u, k = k0 + index / 32u;
                weight[index] = row < outputs ? half(tinfer_affine_value<Q, GS>(v, s, b, row, k, width)) : half(0);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint k = 0; k < 32u; k += 8u) {
                simdgroup_half8x8 aa, b0, b1;
                simdgroup_load(aa, a + sg / 2u * 8u * 32u + k, 32);
                simdgroup_load(b0, weight + k * 32u + sg % 2u * 16u, 32);
                simdgroup_load(b1, weight + k * 32u + sg % 2u * 16u + 8u, 32);
                simdgroup_multiply_accumulate(c0, aa, b0, c0); simdgroup_multiply_accumulate(c1, aa, b1, c1);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        simdgroup_store(c0, output + sg / 2u * 8u * 32u + sg % 2u * 16u, 32);
        simdgroup_store(c1, output + sg / 2u * 8u * 32u + sg % 2u * 16u + 8u, 32);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint index = tid; index < 16u * 32u; index += 128u) {
            const uint row = m0 + index / 32u, column = n0 + index % 32u;
            if (row < rows && column < outputs) y[ulong(order[first + row]) * outputs + column] = T(output[index]);
        }
        )METAL", std::string("#include <metal_simdgroup_matrix>\n") + detail::kMlxAffineDecode, true, false, options);
    return nax ? ax : simd;
}
}

MlxAffineIndirectWeight::MlxAffineIndirectWeight(Shape shape, int bits, int group, Dtype dtype,
    std::array<array, 3> tables) : shape_(std::move(shape)), bits_(bits), group_size_(group), dtype_(dtype), tables_(std::move(tables)) {}

MlxAffineIndirectWeight MlxAffineIndirectWeight::from_address_tables(Shape shape, int bits, int group,
    Dtype dtype, std::array<array, 3> tables, std::shared_ptr<MlxWeightResidency> residency) {
    mfq::MlxAffineLayout layout;
    layout.shape.assign(shape.begin(), shape.end()); layout.bits = bits; layout.group_size = group;
    layout.scale_dtype = dtype == bfloat16 ? "BF16" : dtype == float16 ? "F16" : dtype == float32 ? "F32" : "";
    layout.validate();
    if (shape.size() != 3 || !residency) throw std::invalid_argument("indirect MLX projection requires experts and residency");
    for (auto& table : tables) {
        if (table.dtype() != uint64 || table.shape() != Shape{shape[0]} || !table.flags().row_contiguous)
            throw std::invalid_argument("invalid indirect MLX expert address table");
        table.eval();
        struct Lifetime { array table; std::shared_ptr<MlxWeightResidency> residency; };
        auto owner = std::make_shared<Lifetime>(Lifetime{table, residency});
        table = mlx_buffer_view(table.buffer(), table.shape(), uint64, owner);
        register_indirect_weight_pool(table, residency);
    }
    return MlxAffineIndirectWeight(std::move(shape), bits, group, dtype, std::move(tables));
}

array MlxAffineIndirectWeight::routed_matmul(const array& input, const array& ids, bool shared,
    const MlxExpertRouteTiles* plan) const {
    if (input.ndim() < 1 || input.shape(-1) != input_size() ||
        (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32) ||
        ids.ndim() != 2 || (ids.dtype() != int32 && ids.dtype() != uint32) || ids.shape(1) < 1 ||
        input.size() / input_size() != std::size_t(ids.shape(0)) * (shared ? 1 : ids.shape(1)) || ids.size() > INT_MAX)
        throw std::invalid_argument("indirect MLX projection inputs disagree");
    const Shape shape{ids.shape(0), ids.shape(1), output_size()};
    if (!ids.size()) return zeros(shape, input.dtype());
    const array params({uint32_t(input_size()), uint32_t(output_size()), uint32_t(expert_count()),
        uint32_t(ids.shape(1)), uint32_t(shared)});
    if (ids.shape(0) <= kVectorRows) return projection_kernel()(
        {tables_[0], tables_[1], tables_[2], input, astype(ids, int32), params},
        {shape}, {input.dtype()}, {((output_size() + 15) / 16) * 128, int(ids.size()), 1}, {128, 1, 1},
        {{"T", input.dtype()}, {"S", dtype_}, {"Q", bits_}, {"GS", group_size_}}, std::nullopt, false, {}).front();
    std::optional<MlxExpertRouteTiles> owned;
    if (!plan) { owned = mlx_expert_route_tiles(ids, expert_count()); plan = &*owned; }
    if (auto result = MlxAffineStaged::project_indirect(*this, input, ids, shared, *plan)) return *result;
    const bool nax = input.dtype() != float32 && !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
    const int tile_width = nax ? 128 : 32;
    return prefill_kernel(nax)({tables_[0], tables_[1], tables_[2], input, plan->order, plan->bounds, plan->tile_offsets, params},
        {shape}, {input.dtype()}, {((output_size() + tile_width - 1) / tile_width) * 128, plan->tiles, 1}, {128, 1, 1},
        {{"T", input.dtype()}, {"S", dtype_}, {"Q", bits_}, {"GS", group_size_}}, 0.f, false, {}).front();
}

array MlxAffineIndirectWeight::routed_swiglu(const MlxAffineIndirectWeight& up, const array& input,
    const array& ids, const MlxExpertRouteTiles* plan) const {
    if (shape_ != up.shape_) throw std::invalid_argument("indirect MLX Gate/Up geometry differs");
    if (input.ndim() < 1 || input.shape(-1) != input_size() ||
        (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32) ||
        ids.ndim() != 2 || (ids.dtype() != int32 && ids.dtype() != uint32) || ids.shape(1) < 1 ||
        input.size() / input_size() != std::size_t(ids.shape(0)) || ids.size() > INT_MAX)
        throw std::invalid_argument("indirect MLX Gate/Up inputs disagree");
    std::optional<MlxExpertRouteTiles> owned;
    if (ids.shape(0) > kVectorRows && !plan) { owned = mlx_expert_route_tiles(ids, expert_count()); plan = &*owned; }
    if (ids.shape(0) > kVectorRows || bits_ != up.bits_ || group_size_ != up.group_size_ || dtype_ != up.dtype_)
        return moe_swiglu_pair(routed_matmul(input, ids, true, plan), up.routed_matmul(input, ids, true, plan));
    const Shape shape{ids.shape(0), ids.shape(1), output_size()};
    if (!ids.size()) return zeros(shape, input.dtype());
    const array params({uint32_t(input_size()), uint32_t(output_size()), uint32_t(expert_count()), uint32_t(ids.shape(1))});
    return gate_up_kernel()({tables_[0], tables_[1], tables_[2], up.tables_[0], up.tables_[1], up.tables_[2],
        input, astype(ids, int32), params}, {shape}, {input.dtype()},
        {((output_size() + 15) / 16) * 128, int(ids.size()), 1}, {128, 1, 1},
        {{"T", input.dtype()}, {"S", dtype_}, {"Q", bits_}, {"GS", group_size_}}, std::nullopt, false, {}).front();
}
} // namespace mfq::metal
