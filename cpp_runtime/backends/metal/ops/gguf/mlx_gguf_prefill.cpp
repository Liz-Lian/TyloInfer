#include "mlx_gguf_prefill.h"
#include "mlx_gguf_staged.h"
#include "mlx_gguf_rotation.h"
#include "mlx_moe_ops.h"
#include "gguf_embedded.h"
#include "mlx_indirect_weight_residency.h"
#include "mlx_platform.h"

#include <climits>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;

const std::string& header() {
    static const std::string source = MlxGgufStagedPrefill::decoder_header() + R"METAL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
template<bool Indirect, typename P>
device const uchar* expert_data(P data, uint e, uint n, uint stride) {
    if constexpr (Indirect) return reinterpret_cast<device const uchar*>(data[e]);
    else return reinterpret_cast<device const uchar*>(data) + ulong(e) * n * stride;
}
half4 signed_grid(uint grid, uint signs, float scale, uint shift) {
    auto values = float4(as_type<uchar4>(grid)) * scale;
    return half4(select(values, -values, (uint4(signs) & (uint4(1u, 2u, 4u, 8u) << shift)) != 0u));
}
template<int F, class Lookup>
void load_pair(device const uchar* row, uint k, Lookup lookup, thread half4& low, thread half4& high) {
    const uint p = k % 256u, group = p / 32u, sub = (p % 32u) / 8u, part = (p % 8u) / 4u;
    if constexpr (F == 16 || F == 18) {
        constexpr uint stride = F == 16 ? 66u : 98u;
        auto block = row + ulong(k / 256u) * stride;
        const auto payload = block + 2u;
        const uint auxiliary = gguf_read_u32(payload + (F == 16 ? group * 8u + 4u : 64u + group * 4u));
        const float scale = float(*reinterpret_cast<device const half*>(block)) *
            (.5f + float(auxiliary >> 28u)) * (F == 16 ? .25f : .5f);
        uint a, b;
        if constexpr (F == 16) {
            a = lookup.word(payload[group * 8u + sub], part);
            b = lookup.word(payload[group * 8u + sub + 2u], part);
        } else {
            a = lookup.word(payload[group * 8u + sub * 2u + part], 0u);
            b = lookup.word(payload[group * 8u + (sub + 2u) * 2u + part], 0u);
        }
        low = signed_grid(a, lookup.sign((auxiliary >> (7u * sub)) & 127u), scale, part * 4u);
        high = signed_grid(b, lookup.sign((auxiliary >> (7u * (sub + 2u))) & 127u), scale, part * 4u);
    } else if constexpr (F == 17 || F == 22) {
        constexpr uint stride = F == 17 ? 74u : 82u;
        auto block = row + ulong(k / 256u) * stride;
        const float scale = float(*reinterpret_cast<device const half*>(block));
        const uint scales = block[(F == 17 ? 66u : 74u) + group];
        uint a, b, sa, sb;
        if constexpr (F == 17) {
            const auto codes = reinterpret_cast<device const ushort*>(block + 2u) + group * 4u;
            const uint ca = codes[sub], cb = codes[sub + 2u];
            a = ca & 511u; b = cb & 511u; sa = lookup.sign(ca >> 9u); sb = lookup.sign(cb >> 9u);
        } else {
            const uint bits = block[66u + group];
            a = uint(block[2u + group * 4u + sub]) | (((bits >> (2u * sub)) & 3u) << 8u);
            b = uint(block[2u + group * 4u + sub + 2u]) | (((bits >> (2u * (sub + 2u))) & 3u) << 8u);
            sa = block[34u + group * 4u + sub]; sb = block[34u + group * 4u + sub + 2u];
        }
        low = signed_grid(lookup.word(a, part), sa, scale * (.5f + float(scales & 15u)) * .25f, part * 4u);
        high = signed_grid(lookup.word(b, part), sb, scale * (.5f + float(scales >> 4u)) * .25f, part * 4u);
    } else if constexpr (F == 21) {
        auto block = reinterpret_cast<device const block_iq3_s*>(row) + k / 256u;
        const uint bits = block->qh[group], scales = block->scales[group / 2u];
        const float scale = float(block->d) * (1u + 2u * ((scales >> (4u * (group % 2u))) & 15u));
        const uint first = sub * 2u + part, second = first + 4u;
        const uint a = uint(block->qs[group * 8u + first]) | (((bits >> first) & 1u) << 8u);
        const uint b = uint(block->qs[group * 8u + second]) | (((bits >> second) & 1u) << 8u);
        low = signed_grid(lookup.word(a, 0u), block->signs[group * 4u + sub], scale, part * 4u);
        high = signed_grid(lookup.word(b, 0u), block->signs[group * 4u + sub + 2u], scale, part * 4u);
    } else if constexpr (F == 12 || F == 13) {
        device const uchar* scales;
        device const uchar* codes;
        float d, minimum;
        if constexpr (F == 12) {
            auto block = reinterpret_cast<device const block_q4_K*>(row) + k / 256u;
            scales = block->scales; codes = block->qs; d = float(block->d); minimum = float(block->dmin);
        } else {
            auto block = reinterpret_cast<device const block_q5_K*>(row) + k / 256u;
            scales = block->scales; codes = block->qs; d = float(block->d); minimum = float(block->dmin);
        }
        const uint sc = group < 4u ? scales[group] & 63u
            : (scales[group + 4u] & 15u) | ((scales[group - 4u] >> 6u) << 4u);
        const uint bias = group < 4u ? scales[group + 4u] & 63u
            : (scales[group + 4u] >> 4u) | ((scales[group] >> 6u) << 4u);
        const float scale = d * float(sc), offset = minimum * float(bias);
        for (uint pair = 0; pair < 2u; ++pair) {
            const uint col = p + pair * 16u;
            const uint raw = gguf_read_u32(codes + col / 64u * 32u + col % 32u);
            uint4 values = uint4(as_type<uchar4>((raw >> (4u * ((col % 64u) / 32u))) & 0x0f0f0f0fu));
            if constexpr (F == 13) {
                auto block = reinterpret_cast<device const block_q5_K*>(row) + k / 256u;
                const uint extra = gguf_read_u32(block->qh + col % 32u);
                values |= uint4(as_type<uchar4>((extra >> group) & 0x01010101u)) << 4u;
            }
            const half4 value = half4(float4(values) * scale - offset);
            if (pair) high = value; else low = value;
        }
    } else if constexpr (F == 20 || F == 23) {
        device const uchar* codes;
        float scale;
        if constexpr (F == 20) {
            auto block = reinterpret_cast<device const block_iq4_nl*>(row) + k / 32u;
            codes = block->qs; scale = float(block->d);
        } else {
            auto block = reinterpret_cast<device const block_iq4_xs*>(row) + k / 256u;
            codes = block->qs + group * 16u;
            scale = float(block->d) * float(int(((block->scales_l[group / 2u] >> (4u * (group % 2u))) & 15u)
                | (((uint(block->scales_h) >> (2u * group)) & 3u) << 4u)) - 32);
        }
        const auto packed = as_type<uchar4>(gguf_read_u32(codes + k % 16u));
        float4 a, b;
        for (uint j = 0; j < 4u; ++j) { a[j] = kvalues_iq4nl[packed[j] & 15u]; b[j] = kvalues_iq4nl[packed[j] >> 4u]; }
        low = half4(a * scale); high = half4(b * scale);
    } else {
        low = half4(gguf_values4<F>(row, k, lookup));
        high = half4(gguf_values4<F>(row, k + 16u, lookup));
    }
}
template<int F, class Right, class Lookup>
void fill_right(thread Right& right, device const uchar* matrix, uint first_n, uint k0,
    uint n, uint k, uint stride, Lookup lookup) {
    const ushort middle = right.get_capacity() / 2u;
#pragma clang loop unroll(full)
    for (ushort i = 0; i < middle; i += 4u) {
        const auto low_index = right.get_multidimensional_index(i);
        const auto high_index = right.get_multidimensional_index(i + middle);
        const uint row = first_n + low_index[1];
        const uint first_k = k0 + low_index[0], second_k = k0 + high_index[0];
        half4 low(0.h), high(0.h);
        if (matrix && row < n) {
            const auto source = matrix + ulong(row) * stride;
            if (second_k < k) load_pair<F>(source, first_k, lookup, low, high);
            else if (first_k < k) low = half4(gguf_values4<F>(source, first_k, lookup));
        }
        for (ushort j = 0; j < 4u; ++j) {
            right[i + j] = low[j]; right[i + middle + j] = high[j];
        }
    }
}
)METAL";
    return source;
}

const fast::CustomKernelFunction& mapping_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_positions",
        {"ids", "order", "bounds", "offsets", "params"}, {"positions", "slots"}, R"METAL(
uint at = thread_position_in_grid.x;
if (at >= params[0]) return;
uint original = uint(order[at]);
int expert = ids[original];
if (expert < 0 || uint(expert) >= params[1]) return;
uint position = uint(offsets[expert]) * params[2] + at - uint(bounds[expert * 2]);
positions[original] = int(position);
slots[position] = int(original);
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& dense_kernel() {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_prefill_dense_gemm",
        {"x", "weight", "params"}, {"y"}, R"METAL(
uint k = params[0], n = params[1], m = params[2];
tensor<device half, dextents<int, 2>, tensor_inline> a(
    const_cast<device half*>(x), dextents<int, 2>(k, m));
tensor<device half, dextents<int, 2>, tensor_inline> b(
    const_cast<device half*>(weight), dextents<int, 2>(k, n));
tensor<device float, dextents<int, 2>, tensor_inline> c(y, dextents<int, 2>(n, m));
constexpr auto description = matmul2d_descriptor(64, 64, dynamic_length_v<int>, false, true, false);
matmul2d<description, execution_simdgroups<4>> operation;
auto left = a.slice(0, int(threadgroup_position_in_grid.y) * 64);
auto right = b.slice(0, int(threadgroup_position_in_grid.x) * 64);
auto destination = c.slice(int(threadgroup_position_in_grid.x) * 64, int(threadgroup_position_in_grid.y) * 64);
operation.run(left, right, destination);
)METAL", "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace mpp::tensor_ops;\n");
    return kernel;
}

const fast::CustomKernelFunction& gather_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_gather",
        {"x", "slots", "offsets", "params", "descriptors"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.x;
uint k = params[0], routes = params[1], tile_rows = params[3];
if (tile >= uint(offsets[params[2]])) return;
uint column = threadgroup_position_in_grid.y * 256u + (thread_index_in_threadgroup % 32u) * 8u;
if (column >= k) return;
uint live = COMPACT ? uint(descriptors[tile * 3u + 1u]) : tile_rows;
uint capacity = COMPACT ? (tile_rows == 8u || live <= 8u ? 8u : live <= 16u ? 16u : live <= 32u ? 32u : 48u) : tile_rows;
for (uint local = thread_index_in_threadgroup / 32u; local < capacity; local += 8u) {
    uint row = (COMPACT ? uint(descriptors[tile * 3u + 2u]) : tile * tile_rows) + local;
    int original = slots[row];
    half4 low(0.h), high(0.h);
    if (original >= 0) {
        ulong source = ulong(uint(original) / routes) * k + column;
        low = half4(half(x[source]), half(x[source + 1u]), half(x[source + 2u]), half(x[source + 3u]));
        high = half4(half(x[source + 4u]), half(x[source + 5u]), half(x[source + 6u]), half(x[source + 7u]));
    }
    *reinterpret_cast<device half4*>(y + ulong(row) * k + column) = low;
    *reinterpret_cast<device half4*>(y + ulong(row) * k + column + 4u) = high;
}
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& rotated_gather_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_rotated_gate_up_gather",
        {"x", "slots", "offsets", "params", "gate_signs", "up_signs", "descriptors"}, {"g", "u"}, R"METAL(
uint tile = threadgroup_position_in_grid.x;
if (tile >= uint(offsets[params[2]])) return;
uint lane = thread_index_in_simdgroup;
uint col = threadgroup_position_in_grid.y * 128u + lane * 4u;
uint live = COMPACT ? uint(descriptors[tile * 3u + 1u]) : params[3];
uint capacity = COMPACT ? (params[3] == 8u || live <= 8u ? 8u : live <= 16u ? 16u : live <= 32u ? 32u : 48u) : params[3];
for (uint local = simdgroup_index_in_threadgroup; local < capacity; local += 8u) {
    uint row = (COMPACT ? uint(descriptors[tile * 3u + 2u]) : tile * params[3]) + local;
    int original = slots[row];
    float4 gate(0.f), up(0.f);
    if (original >= 0) {
        ulong at = ulong(uint(original) / params[1]) * params[0] + col;
        gate = up = float4(x[at], x[at + 1u], x[at + 2u], x[at + 3u]);
        if constexpr (GR) {
            if constexpr (GS) gate *= float4(gate_signs[col], gate_signs[col + 1u], gate_signs[col + 2u], gate_signs[col + 3u]);
            gate = gguf_gather_rotate(gate, lane);
        }
        if constexpr (UR) {
            if constexpr (US) up *= float4(up_signs[col], up_signs[col + 1u], up_signs[col + 2u], up_signs[col + 3u]);
            up = gguf_gather_rotate(up, lane);
        }
    }
    const ulong at = ulong(row) * params[0] + col;
    *reinterpret_cast<device half4*>(g + at) = half4(half(S(gate.x)), half(S(gate.y)), half(S(gate.z)), half(S(gate.w)));
    *reinterpret_cast<device half4*>(u + at) = half4(half(S(up.x)), half(S(up.y)), half(S(up.z)), half(S(up.w)));
}
)METAL", R"METAL(
float4 gguf_gather_rotate(float4 value, uint lane) {
    value = float4(value.x + value.y, value.x - value.y, value.z + value.w, value.z - value.w);
    value = float4(value.x + value.z, value.y + value.w, value.x - value.z, value.y - value.w);
    for (ushort mask = 1u; mask < 32u; mask *= 2u) {
        float4 other(simd_shuffle_xor(value.x, mask), simd_shuffle_xor(value.y, mask),
            simd_shuffle_xor(value.z, mask), simd_shuffle_xor(value.w, mask));
        value = lane & mask ? other - value : value + other;
    }
    return value * 0.08838834764831845f;
}
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& restore_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_grouped_restore",
        {"x", "positions", "params"}, {"y"}, R"METAL(
uint at = thread_position_in_grid.x;
uint n = params[0];
if (at >= params[1] * (n / V)) return;
int row = positions[at / (n / V)];
if constexpr (V == 4) {
    const auto value = row < 0 ? vec<T, 4>(T(0))
        : *reinterpret_cast<device const vec<T, 4>*>(x + ulong(row) * n + (at % (n / V)) * V);
    *reinterpret_cast<device vec<T, 4>*>(y + ulong(at) * V) = value;
} else {
    y[at] = row < 0 ? T(0) : T(x[ulong(row) * n + at % n]);
}
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& gate_up_kernel() {
    static const auto kernel = indirect_weight_metal_kernel("tinfer_gguf_prefill_register_gate_up",
        {"gate", "up", "x", "bounds", "offsets", "params", "x_up"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.y, experts = params[3] & 0x7fffffffu;
if (tile >= uint(offsets[experts])) return;
uint e = 0u, high = experts;
while (e < high) {
    uint middle = e + (high - e) / 2u;
    if (uint(offsets[middle + 1u]) <= tile) e = middle + 1u; else high = middle;
}
uint k = params[0], n = params[1], padded = params[2], row0 = tile * 16u;
uint n0 = (threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup) * 32u;
constexpr bool SAME_TABLE = FG == FU || (PIPELINED && FG >= 144 && FG <= 151 && FU >= 144 && FU <= 151);
threadgroup uint gate_table[PIPELINED ? PlaneLayout<FG>::table_words : GgufConstantLookup<FG>::storage_size];
threadgroup uint up_table[SAME_TABLE ? 1u : (PIPELINED ? PlaneLayout<FU>::table_words : GgufConstantLookup<FU>::storage_size)];
if constexpr (PIPELINED) {
    load_staged_lookup<FG>(gate_table, thread_index_in_threadgroup, 128u);
    if constexpr (!SAME_TABLE) load_staged_lookup<FU>(up_table, thread_index_in_threadgroup, 128u);
} else {
    gguf_load_lookup<FG>(gate_table, thread_index_in_threadgroup, 128u);
    if constexpr (!SAME_TABLE) gguf_load_lookup<FU>(up_table, thread_index_in_threadgroup, 128u);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
auto g = expert_data<IG>(gate, e, n, params[4]);
auto u = expert_data<IU>(up, e, n, params[5]);
constexpr auto description = matmul2d_descriptor(16, 32, 32, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> activations(
    const_cast<device half*>(x), dextents<int, 2>(k, padded));
tensor<device half, dextents<int, 2>, tensor_inline> up_activations(
    const_cast<device half*>(SEPARATE_INPUT ? x_up : x), dextents<int, 2>(k, padded));
auto right_g = operation.template get_right_input_cooperative_tensor<half, half, float>();
auto right_u = operation.template get_right_input_cooperative_tensor<half, half, float>();
auto input = activations.template slice<32, 16>(0, row0);
auto sum_g = operation.template get_destination_cooperative_tensor<decltype(input), decltype(right_g), float>();
auto sum_u = operation.template get_destination_cooperative_tensor<decltype(input), decltype(right_u), float>();
for (ushort i = 0; i < sum_g.get_capacity(); ++i) { sum_g[i] = 0.f; sum_u[i] = 0.f; }
if constexpr (PIPELINED) {
    if (n0 >= n) return;
    threadgroup half stages[8192];
    auto scratch = stages + simdgroup_index_in_threadgroup * 2048u;
    if (g) staged_dot<FG, 16, false>(x, g, params, params, n0 + thread_index_in_simdgroup, k, padded, row0,
        scratch, gate_table, sum_g);
    if (u) staged_dot<FU, 16, false>(SEPARATE_INPUT ? x_up : x, u, params, params, n0 + thread_index_in_simdgroup, k, padded, row0,
        scratch, SAME_TABLE ? gate_table : up_table, sum_u);
} else {
    for (uint start = 0; start < k; start += 32u) {
        fill_right<FG>(right_g, g, n0, start, n, k, params[4], GgufThreadgroupLookup<FG>{gate_table});
        fill_right<FU>(right_u, u, n0, start, n, k, params[5], GgufThreadgroupLookup<FU>{FG == FU ? gate_table : up_table});
        input = activations.template slice<32, 16>(start, row0);
        operation.run(input, right_g, sum_g);
        if constexpr (SEPARATE_INPUT) {
            auto up_input = up_activations.template slice<32, 16>(start, row0);
            operation.run(up_input, right_u, sum_u);
        } else operation.run(input, right_u, sum_u);
    }
}
for (ushort i = 0; i < sum_g.get_capacity(); ++i) if (sum_g.is_valid_element(i)) {
    auto index = sum_g.get_multidimensional_index(i);
    uint column = n0 + index[0], row = row0 + index[1];
    if (column < n) {
        float a = float(T(sum_g[i])), b = float(T(sum_u[i]));
        if (params[3] >> 31u) { const float value = a; a = b; b = value; }
        const float limit = as_type<float>(params[6]);
        if constexpr (CLIPPED) { a = min(a, limit); b = clamp(b, -limit, limit); }
        y[ulong(row) * n + column] = T((a / (1.f + exp(-a))) * b);
    }
}
)METAL", header(), true, false);
    return kernel;
}

const fast::CustomKernelFunction& down_kernel() {
    static const auto kernel = indirect_weight_metal_kernel("tinfer_gguf_prefill_register_down",
        {"packed", "x", "offsets", "params"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.y, experts = params[3];
if (tile >= uint(offsets[experts])) return;
uint e = 0u, high = experts;
while (e < high) {
    uint middle = e + (high - e) / 2u;
    if (uint(offsets[middle + 1u]) <= tile) e = middle + 1u; else high = middle;
}
uint k = params[0], n = params[1], row0 = tile * 16u;
uint n0 = (threadgroup_position_in_grid.x * 2u + simdgroup_index_in_threadgroup) * 64u;
threadgroup uint table[PIPELINED ? PlaneLayout<F>::table_words : GgufConstantLookup<F>::storage_size];
if constexpr (PIPELINED) load_staged_lookup<F>(table, thread_index_in_threadgroup, 64u);
else gguf_load_lookup<F>(table, thread_index_in_threadgroup, 64u);
threadgroup_barrier(mem_flags::mem_threadgroup);
auto matrix = expert_data<INDIRECT>(packed, e, n, params[4]);
if constexpr (PIPELINED) {
    if (n0 >= n || !matrix) return;
    threadgroup half stages[4096];
    constexpr auto description = matmul2d_descriptor(16, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<description, execution_simdgroup> operation;
    tensor<device half, dextents<int, 2>, tensor_inline> activations(
        const_cast<device half*>(x), dextents<int, 2>(k, params[2]));
    auto input = activations.template slice<32, 16>(0, row0);
    auto right = operation.template get_right_input_cooperative_tensor<half, half, float>();
    for (uint chunk = 0u; chunk < 2u && n0 + chunk * 32u < n; ++chunk) {
        auto sum = operation.template get_destination_cooperative_tensor<decltype(input), decltype(right), float>();
        for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
        staged_dot<F, 16, false>(x, matrix, params, params, n0 + chunk * 32u + thread_index_in_simdgroup,
            k, params[2], row0, stages + simdgroup_index_in_threadgroup * 2048u, table, sum);
        for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
            auto index = sum.get_multidimensional_index(i);
            y[ulong(row0 + index[1]) * n + n0 + chunk * 32u + index[0]] = T(sum[i]);
        }
    }
    return;
}
constexpr auto description = matmul2d_descriptor(16, 64, 32, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> activations(
    const_cast<device half*>(x), dextents<int, 2>(k, params[2]));
auto right = operation.template get_right_input_cooperative_tensor<half, half, float>();
auto input = activations.template slice<32, 16>(0, row0);
auto sum = operation.template get_destination_cooperative_tensor<decltype(input), decltype(right), float>();
for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
for (uint start = 0; start < k; start += 32u) {
    fill_right<F>(right, matrix, n0, start, n, k, params[4], GgufThreadgroupLookup<F>{table});
    input = activations.template slice<32, 16>(start, row0);
    operation.run(input, right, sum);
}
for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
    auto index = sum.get_multidimensional_index(i);
    uint column = n0 + index[0];
    if (column < n) y[ulong(row0 + index[1]) * n + column] = T(sum[i]);
}
)METAL", header(), true, false);
    return kernel;
}

const fast::CustomKernelFunction& reduce_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_reduce",
        {"x", "positions", "weights", "params"}, {"y"}, R"METAL(
uint index = thread_position_in_grid.x;
uint n = params[0], tokens = params[1], count = params[2];
if (index >= tokens * n) return;
uint token = index / n, column = index % n;
float sum = 0.f;
for (uint r = 0; r < count; ++r) {
    uint original = token * count + r;
    int position = positions[original];
    if (position >= 0) sum += float(x[ulong(position) * n + column]) * float(weights[original]);
}
y[index] = T(sum);
)METAL");
    return kernel;
}
} // namespace

array gguf_dense_prefill(const array& input, const array& weight) {
    const int k = input.shape(-1), n = weight.shape(0);
    const auto m = input.size() / k;
    if (weight.ndim() != 2 || weight.shape(1) != k || weight.dtype() != float16 || m > INT_MAX)
        throw std::invalid_argument("invalid GGUF dense prefill geometry");
    auto shape = input.shape();
    shape.back() = n;
    auto output = dense_kernel()({contiguous(astype(input, float16)), weight,
        array({uint32_t(k), uint32_t(n), uint32_t(m)})}, {shape}, {float32},
        {((n + 63) / 64) * 128, int((m + 63) / 64), 1}, {128, 1, 1},
        {}, std::nullopt, false, {}).front();
    return astype(output, input.dtype());
}

bool gguf_expert_prefill_available(const array& input) {
    return input.ndim() >= 2 && input.shape(-1) > 0 && input.size() / input.shape(-1) > 6 &&
        !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
}

bool gguf_short_sparse_routes(const array& ids, int experts) {
    return ids.ndim() == 2 && ids.shape(0) > 0 && ids.shape(0) <= 8 && ids.size() <= std::size_t(experts);
}

int gguf_prefill_tile_rows(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const MlxGgufWeight& down, const array& ids) {
    if (gate.expert_count() > 512 || !MlxGgufStagedPrefill::supports_grouped(gate) ||
        !MlxGgufStagedPrefill::supports_grouped(up) || !MlxGgufStagedPrefill::supports_grouped(down)) return 16;
    if (ids.shape(0) <= 6 || ids.size() <= std::size_t(gate.expert_count())) return 8;
    const auto gyro = [](const MlxGgufWeight& weight) {
        return weight.format() >= 144 && weight.format() <= 151 && weight.has_staged_planes();
    };
    return (gyro(gate) || gyro(up) || gyro(down)) &&
        ids.size() >= std::size_t(gate.expert_count()) * 48 ? 48 : 32;
}

bool gguf_compact_ffn_available(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const MlxGgufWeight& down, const array& ids) {
    const auto raw_k = [](const MlxGgufWeight& weight) { return weight.format() == 10 || weight.format() == 11; };
    const auto raw_iq2 = [](const MlxGgufWeight& weight) { return weight.format() == 16 || weight.format() == 17 || weight.format() == 22; };
    const auto q8 = [](const MlxGgufWeight& weight) { return weight.format() == 8 || weight.format() == 15; };
    return ids.ndim() == 2 && ids.size() && ids.shape(0) > 0 && ids.shape(0) <= 6 &&
        !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5") &&
        !gate.has_mixed_formats() && !up.has_mixed_formats() && !down.has_mixed_formats() &&
        (gate.has_staged_planes() || up.has_staged_planes() || (raw_k(gate) && raw_k(up)) ||
         (ids.shape(0) > 1 && ((raw_k(gate) && raw_iq2(up)) || (raw_k(up) && raw_iq2(gate)) ||
                             (q8(gate) && q8(up))))) &&
        gguf_prefill_tile_rows(gate, up, down, ids) == 8;
}

array gguf_routed_ffn(const std::array<MlxGgufWeight, 3>& w, const array& input,
    const array& ids, const array& probabilities, float limit, bool unique_per_token) {
    if (ids.ndim() != 2 || w[0].input_size() != w[1].input_size() || w[0].output_size() != w[1].output_size() ||
        w[0].expert_count() != w[1].expert_count() || w[0].expert_count() != w[2].expert_count() ||
        w[2].input_size() != w[0].output_size() || w[2].output_size() != w[0].input_size())
        throw std::invalid_argument("GGUF FFN projections and routes disagree");
    const bool compact_decode = gguf_compact_ffn_available(w[0], w[1], w[2], ids);
    if ((gguf_expert_prefill_available(input) || compact_decode) &&
        !w[0].has_mixed_formats() && !w[1].has_mixed_formats() && !w[2].has_mixed_formats()) {
        const int tile = gguf_prefill_tile_rows(w[0], w[1], w[2], ids);
        if (tile != 16 || (!w[0].has_staged_planes() && !w[1].has_staged_planes() && !w[2].has_staged_planes())) {
            const auto plan = unique_per_token ? MlxGgufPrefillPlan::from_topk(input, ids, w[0].expert_count(), tile)
                : MlxGgufPrefillPlan(input, ids, w[0].expert_count(), tile);
            return plan.down(w[2], plan.gate_up(w[0], w[1], nullptr, nullptr, limit), probabilities);
        }
    }
    std::optional<MlxGgufRoutePlan> plan;
    if (ids.shape(0) > 6) plan = gguf_route_plan(ids, w[0].expert_count(),
        w[0].has_mixed_formats() && w[1].has_mixed_formats() && w[2].has_mixed_formats());
    const auto* routes = plan ? &*plan : nullptr;
    return w[2].routed_weighted_matmul(w[0].routed_swiglu(w[1], input, ids, routes, limit), ids, probabilities, routes);
}

MlxGgufPrefillPlan::MlxGgufPrefillPlan(const array& input, const array& ids, int experts, int tile_rows,
    bool shared_input, bool include_shared)
    : MlxGgufPrefillPlan(input, ids, experts, tile_rows, shared_input, include_shared, false) {}

MlxGgufPrefillPlan MlxGgufPrefillPlan::from_topk(const array& input, const array& ids,
    int experts, int tile_rows, bool include_shared) {
    return MlxGgufPrefillPlan(input, ids, experts, tile_rows, true, include_shared, true);
}

MlxGgufPrefillPlan::MlxGgufPrefillPlan(const array& input, const array& ids, int experts, int tile_rows,
    bool shared_input, bool include_shared, bool unique_per_token)
    : source_(input), ids_(ids), positions_(ids), input_(input), dtype_(input.dtype()),
      tokens_(ids.ndim() == 2 ? ids.shape(0) : 0), selected_(ids.ndim() == 2 ? ids.shape(1) : 0), padded_rows_(0) {
    if (input.ndim() < 2 || input.shape(-1) <= 0 || mlx_metal_nax_disabled() ||
        !mlx_apple_chip_starts_with("Apple M5") || ids.ndim() != 2 || !ids.size() ||
        input.size() / input.shape(-1) != std::size_t(tokens_) * (shared_input ? 1 : selected_) ||
        (include_shared && (!shared_input || tile_rows == 16)) ||
        (tile_rows != 8 && tile_rows != 16 && tile_rows != 32 && tile_rows != 48))
        throw std::invalid_argument("invalid GGUF expert prefill geometry");
    selected_ += int(include_shared);
    if (tokens_ == 1 && tile_rows == 8) {
        if (experts < 1 || experts > 512 || (ids.dtype() != int32 && ids.dtype() != uint32))
            throw std::invalid_argument("invalid GGUF single-token route geometry");
        std::vector<int> positions(selected_);
        for (int i = 0; i < selected_; ++i) positions[i] = i;
        positions_ = array(positions.begin(), Shape{selected_}, int32);
        grouped_.emplace(MlxGgufGroupedRoutes{contiguous(reshape(astype(ids, int32), {-1})),
            array({selected_}, int32), ids, ids, positions_, selected_, 1});
        padded_rows_ = selected_;
        input_ = reshape(contiguous(astype(input, float16)),
            {int(input.size() / input.shape(-1)), input.shape(-1)});
        return;
    }
    array slots = ids, offsets = ids;
    int tiles;
    if (tile_rows != 16) {
        grouped_.emplace(gguf_group_routes(ids, experts, tile_rows, include_shared, true));
        if (unique_per_token) grouped_->maximum_live_rows = std::min(tile_rows, tokens_);
        tiles = grouped_->tiles;
        padded_rows_ = grouped_->row_capacity;
        positions_ = grouped_->route_rows;
        slots = grouped_->grouped_routes;
        offsets = grouped_->tile_offsets;
    } else {
        routes_.emplace(mlx_expert_route_tiles(ids, experts, tile_rows));
        if (routes_->tiles > INT_MAX / tile_rows) throw std::overflow_error("GGUF prefill routes are too large");
        tiles = routes_->tiles;
        padded_rows_ = tiles * tile_rows;
        auto maps = mapping_kernel()({contiguous(reshape(astype(ids, int32), {-1})), routes_->order,
            routes_->bounds, routes_->tile_offsets, array({uint32_t(ids.size()), uint32_t(experts), uint32_t(tile_rows)})},
            {{int(ids.size())}, {padded_rows_}}, {int32, int32}, {int((ids.size() + 127) / 128 * 128), 1, 1},
            {128, 1, 1}, {}, -1.f, false, {});
        positions_ = std::move(maps[0]);
        slots = std::move(maps[1]);
        offsets = routes_->tile_offsets;
    }
    const auto width = input.shape(-1);
    const auto elements = std::size_t(padded_rows_) * width;
    if (elements > INT_MAX - 127) throw std::overflow_error("GGUF prefill activation extent is too large");
    gather_.emplace(Gather{slots, offsets,
        array({uint32_t(width), uint32_t(shared_input ? selected_ : 1), uint32_t(experts + int(include_shared)), uint32_t(tile_rows)}),
        grouped_ ? grouped_->descriptors : offsets, tiles, bool(grouped_)});
    input_.reset();
}

const array& MlxGgufPrefillPlan::grouped_input() const {
    if (!input_) input_.emplace(gather_input(source_));
    return *input_;
}

array MlxGgufPrefillPlan::gather_input(const array& input) const {
    const int width = input.shape(-1);
    if (!gather_) return reshape(contiguous(astype(input, float16)), {int(input.size() / width), width});
    return gather_kernel()({contiguous(input), gather_->slots, gather_->offsets, gather_->params, gather_->descriptors},
        {{padded_rows_, width}}, {float16}, {gather_->tiles * 256, (width + 255) / 256, 1}, {256, 1, 1},
        {{"COMPACT", gather_->compact}}, std::nullopt, false, {}).front();
}

array MlxGgufPrefillPlan::restore(const array& grouped) const {
    if (grouped.ndim() != 2 || grouped.shape(0) != padded_rows_)
        throw std::invalid_argument("GGUF grouped projection output geometry mismatch");
    if (tokens_ == 1 && grouped_ && grouped_->tile_rows == 1)
        return reshape(grouped, {tokens_, selected_, grouped.shape(1)});
    const auto elements = std::size_t(tokens_) * selected_ * grouped.shape(1);
    if (elements > INT_MAX - 127) throw std::overflow_error("GGUF restored projection extent is too large");
    const int vector = grouped.shape(1) % 4 == 0 && grouped.dtype() == dtype_ &&
        (dtype_ == float16 || dtype_ == float32) ? 4 : 1;
    return restore_kernel()({grouped, positions_, array({uint32_t(grouped.shape(1)), uint32_t(tokens_ * selected_)})},
        {{tokens_, selected_, grouped.shape(1)}}, {dtype_}, {int((elements / vector + 127) / 128 * 128), 1, 1},
        {128, 1, 1}, {{"T", dtype_}, {"V", vector}}, std::nullopt, false, {}).front();
}

array MlxGgufPrefillPlan::project(const MlxGgufWeight& weight) const {
    if (weight.has_hadamard()) {
        auto plain = weight;
        plain.rotation_.reset();
        auto transformed = *this;
        transformed.input_ = gather_input(weight.rotate_input(source_));
        return transformed.project(plain);
    }
    if (!grouped_ || !MlxGgufStagedPrefill::supports_grouped(weight) || weight.input_size() != source_.shape(-1))
        throw std::invalid_argument("GGUF grouped projection weight geometry mismatch");
    return restore(MlxGgufStagedPrefill::down(weight, grouped_input(), *grouped_, dtype_));
}

array MlxGgufPrefillPlan::gate_up(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const MlxGgufWeight* shared_gate, const MlxGgufWeight* shared_up, float limit) const {
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid GGUF SwiGLU limit");
    if (grouped_ && grouped_->tile_rows == 8 && tokens_ > 1 && tokens_ <= 8 && source_.dtype() == float16 &&
        source_.size() / source_.shape(-1) == std::size_t(tokens_) &&
        MlxGgufStagedPrefill::supports_grouped(gate) && MlxGgufStagedPrefill::supports_grouped(up) &&
        !shared_gate && !shared_up) {
        if (gate.shape_.size() != 3 || gate.shape_ != up.shape_ || gate.input_size() != source_.shape(-1))
            throw std::invalid_argument("GGUF source Gate/Up weights disagree");
        const auto source = reshape(contiguous(source_), {tokens_, source_.shape(-1)});
        const auto first = gate.has_hadamard() ? gate.rotate_input(source) : source;
        const bool same = gate.same_rotation(up);
        const auto second = same ? first : up.has_hadamard() ? up.rotate_input(source) : source;
        auto gate_plain = gate, up_plain = up;
        gate_plain.rotation_.reset(); up_plain.rotation_.reset();
        return MlxGgufStagedPrefill::gate_up(gate_plain, up_plain, first, *grouped_, dtype_,
            shared_gate, shared_up, same ? nullptr : &second, limit, selected_);
    }
    if (dtype_ == float32 && bool(gate.planes_) != bool(up.planes_) && !shared_gate && !shared_up && ids_.shape(1) == selected_) {
        const auto g = gate.routed_matmul(source_, ids_, true), u = up.routed_matmul(source_, ids_, true);
        const auto values = limit > 0.f ? moe_limited_swiglu_pair(g, u, limit) : moe_swiglu_pair(g, u);
        const auto flat = reshape(values, {tokens_ * selected_, gate.output_size()});
        if (!gather_) return flat;
        const auto valid = logical_and(gather_->slots >= array(0, int32), gather_->slots < array(tokens_ * selected_, int32));
        const auto rows = minimum(maximum(gather_->slots, array(0, int32)), array(tokens_ * selected_ - 1, int32));
        return where(expand_dims(valid, -1), take(flat, rows, 0), array(0.f));
    }
    if (gate.has_hadamard() || up.has_hadamard()) {
        if (shared_gate || shared_up)
            throw std::invalid_argument("GGUF rotated prefill cannot merge an untransformed shared expert");
        auto gate_plain = gate, up_plain = up;
        gate_plain.rotation_.reset(); up_plain.rotation_.reset();
        if (!gate.same_rotation(up)) {
            auto transformed = *this;
            const int width = source_.shape(-1);
            if (gather_ && width % 128 == 0 &&
                (!gate.rotation_ || gate.rotation_->block == 128) && (!up.rotation_ || up.rotation_->block == 128)) {
                static const auto one = array({1.f}, float32);
                auto gathered = rotated_gather_kernel()({contiguous(source_), gather_->slots, gather_->offsets, gather_->params,
                    gate.rotation_ && !gate.rotation_->pattern.empty() ? gate.rotation_->signs : one,
                    up.rotation_ && !up.rotation_->pattern.empty() ? up.rotation_->signs : one, gather_->descriptors},
                    {{padded_rows_, width}, {padded_rows_, width}}, {float16, float16}, {gather_->tiles * 256, width / 128, 1}, {256, 1, 1},
                    {{"S", source_.dtype()}, {"COMPACT", gather_->compact}, {"GR", bool(gate.rotation_)}, {"UR", bool(up.rotation_)},
                        {"GS", gate.rotation_ && !gate.rotation_->pattern.empty()}, {"US", up.rotation_ && !up.rotation_->pattern.empty()}},
                    std::nullopt, false, {});
                transformed.input_ = std::move(gathered[0]);
                return transformed.gate_up_inputs(gate_plain, up_plain, &gathered[1], nullptr, nullptr, limit);
            }
            transformed.input_ = gate.has_hadamard() ? gather_input(gate.rotate_input(source_)) : grouped_input();
            const auto up_input = up.has_hadamard() ? gather_input(up.rotate_input(source_)) : grouped_input();
            return transformed.gate_up_inputs(gate_plain, up_plain, &up_input, nullptr, nullptr, limit);
        }
        auto transformed = *this;
        transformed.input_ = gather_input(gate.rotate_input(source_));
        return transformed.gate_up(gate_plain, up_plain, nullptr, nullptr, limit);
    }
    return gate_up_inputs(gate, up, nullptr, shared_gate, shared_up, limit);
}

array MlxGgufPrefillPlan::gate_up_inputs(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const array* up_input, const MlxGgufWeight* shared_gate, const MlxGgufWeight* shared_up, float limit) const {
    if (gate.shape_.size() != 3 || gate.shape_ != up.shape_ || gate.input_size() != source_.shape(-1))
        throw std::invalid_argument("GGUF prefill Gate/Up weights disagree");
    if (!grouped_ && (gate.planes_ || up.planes_))
        throw std::invalid_argument("GGUF compressed planes require a grouped route plan");
    const auto& input = grouped_input();
    if (grouped_) {
        if (!MlxGgufStagedPrefill::supports_grouped(gate) || !MlxGgufStagedPrefill::supports_grouped(up))
            throw std::invalid_argument("GGUF staged Gate/Up format is unsupported");
        if (!shared_gate && !shared_up && grouped_->tile_rows >= 32 && (up_input || gate.format() != up.format())) {
            const auto g = MlxGgufStagedPrefill::down(gate, input, *grouped_, dtype_);
            return MlxGgufStagedPrefill::down(up, up_input ? *up_input : input, *grouped_, dtype_, nullptr, &g, limit);
        }
        return MlxGgufStagedPrefill::gate_up(gate, up, input, *grouped_, dtype_, shared_gate, shared_up, up_input, limit);
    }
    const int n = gate.output_size();
    const bool reverse = gguf_gate_up_reversed(gate.format_, up.format_);
    const auto& first = reverse ? up : gate;
    const auto& second = reverse ? gate : up;
    const auto& second_input = up_input ? *up_input : input;
    return gate_up_kernel()({first.bytes_, second.bytes_, reverse ? second_input : input, routes_->bounds, routes_->tile_offsets,
        array({uint32_t(gate.input_size()), uint32_t(n), uint32_t(padded_rows_), uint32_t(gate.expert_count()) | (uint32_t(reverse) << 31u),
            uint32_t(first.row_bytes_), uint32_t(second.row_bytes_), std::bit_cast<uint32_t>(limit)}), reverse ? input : second_input}, {{padded_rows_, n}}, {dtype_},
        {((n + 127) / 128) * 128, routes_->tiles, 1}, {128, 1, 1},
        {{"FG", first.format_}, {"FU", second.format_}, {"T", dtype_}, {"IG", first.indirect_}, {"IU", second.indirect_},
            {"SEPARATE_INPUT", up_input != nullptr}, {"CLIPPED", limit > 0.f},
            {"PIPELINED", n % 32 == 0 && MlxGgufStagedPrefill::supports_grouped_format(gate.format_) &&
                MlxGgufStagedPrefill::supports_grouped_format(up.format_) &&
                ((gate.format_ >= 144 && gate.format_ <= 151) || (up.format_ >= 144 && up.format_ <= 151))}},
        0.f, false, {}).front();
}

array MlxGgufPrefillPlan::down(const MlxGgufWeight& weight, const array& hidden,
    const array& probabilities, bool float_accumulation, const MlxGgufWeight* shared) const {
    if (dtype_ == float32 && !weight.planes_ && !shared && !float_accumulation && ids_.shape(1) == selected_)
        return weight.routed_weighted_matmul(restore(hidden), ids_, probabilities);
    if (weight.has_hadamard()) {
        if (shared) throw std::invalid_argument("GGUF rotated prefill cannot merge an untransformed shared expert");
        auto plain = weight;
        plain.rotation_.reset();
        return down(plain, weight.rotate_input(hidden), probabilities, float_accumulation);
    }
    const int n = weight.output_size();
    if (weight.shape_.size() != 3 || hidden.shape() != Shape{padded_rows_, weight.input_size()} ||
        probabilities.shape() != Shape{tokens_, selected_}) throw std::invalid_argument("GGUF prefill Down inputs disagree");
    if (!grouped_ && weight.planes_)
        throw std::invalid_argument("GGUF compressed planes require a grouped route plan");
    if (grouped_ && grouped_->tile_rows == 1 && !weight.planes_ && !shared && !float_accumulation &&
        grouped_->descriptors.size() == std::size_t(selected_))
        return weight.routed_weighted_matmul(reshape(hidden, {1, selected_, weight.input_size()}),
            reshape(grouped_->descriptors, {1, selected_}), probabilities);
    if (grouped_ && grouped_->tile_rows == 1 && !shared && !float_accumulation &&
        MlxGgufStagedPrefill::prefers_weighted_down(weight) && grouped_->descriptors.size() == std::size_t(selected_))
        return MlxGgufStagedPrefill::weighted_down(weight, hidden,
            reshape(grouped_->descriptors, {1, selected_}), probabilities, dtype_);
    const auto regular_down = [&] { return down_kernel()({weight.bytes_, astype(hidden, float16), routes_->tile_offsets,
        array({uint32_t(weight.input_size()), uint32_t(n), uint32_t(padded_rows_), uint32_t(weight.expert_count()),
            uint32_t(weight.row_bytes_)})}, {{padded_rows_, n}}, {dtype_},
        {((n + 127) / 128) * 64, routes_->tiles, 1}, {64, 1, 1},
        {{"F", weight.format_}, {"T", dtype_}, {"INDIRECT", weight.indirect_},
            {"PIPELINED", n % 32 == 0 && weight.format_ >= 144 && weight.format_ <= 151}}, 0.f, false, {}).front(); };
    if (grouped_ && !MlxGgufStagedPrefill::supports_grouped(weight))
        throw std::invalid_argument("GGUF staged Down format is unsupported");
    auto pairs = grouped_ ? MlxGgufStagedPrefill::down(weight, hidden, *grouped_, dtype_, shared) : regular_down();
    const auto elements = std::size_t(tokens_) * n;
    if (elements > INT_MAX - 127) throw std::overflow_error("GGUF prefill reduction extent is too large");
    return reduce_kernel()({pairs, positions_, contiguous(probabilities),
        array({uint32_t(n), uint32_t(tokens_), uint32_t(selected_)})}, {{tokens_, n}},
        {float_accumulation ? float32 : dtype_}, {int((elements + 127) / 128 * 128), 1, 1}, {128, 1, 1},
        {{"T", float_accumulation ? float32 : dtype_}}, std::nullopt, false, {}).front();
}
} // namespace mfq::metal
