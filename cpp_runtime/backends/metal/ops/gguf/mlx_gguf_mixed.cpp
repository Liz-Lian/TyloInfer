#include "mlx_gguf.h"
#include "mlx_gguf_staged.h"
#include "gguf_embedded.h"
#include "mlx_buffer_view.h"
#include "mlx_indirect_weight_residency.h"
#include "mlx_platform.h"
#include "mlx_moe_ops.h"
#include "../../platform/metal_device_info.h"
#include "ggml.h"

#include <algorithm>
#include <bit>
#include <climits>
#include <cctype>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;
constexpr int types[]{2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 42, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 39};

int format_index(const std::string& dtype) {
    for (std::size_t j = 0; j < std::size(types); ++j) {
        auto name = "GGUF_" + std::string(ggml_type_name(static_cast<ggml_type>(types[j])));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        if (name == dtype) return int(j);
    }
    throw std::invalid_argument("unsupported GGUF mixed expert format");
}

const std::string& mixed_header(bool staged = false) {
    const auto make = [](bool staged) {
        std::string result = staged ? MlxGgufStagedPrefill::scalar_decoder_header() : kGgufDecode;
        result += R"METAL(
template<int F, typename X>
float mixed_dot(device const uchar* row, X x, uint K, uint lane) {
    float sum = 0.f;
    if constexpr (F == 143) {
        for (uint k = 0u; k < K; k += 128u) sum += gguf_ptq_dot128<8>(row, x, k, lane);
    } else if constexpr (F == 42) {
        for (uint k = lane * 16u; k < K; k += 128u) sum += gguf_q2_dot<16>(row, x, k);
    } else {
        for (uint k = lane * 4u; k < K; k += 32u)
            sum += dot(gguf_values4<F>(row, k), float4(x[k], x[k + 1u], x[k + 2u], x[k + 3u]));
    }
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 1u);
    return sum;
}
template<int MASK, typename X>
float mixed_dispatch(uint format, device const uchar* row, X x, uint K, uint lane) {
    if (!row) return 0.f;
    switch (format) {
)METAL";
        for (std::size_t i = 0; i < std::size(types); ++i)
            result += "case " + std::to_string(types[i]) + ": if constexpr (MASK & " + std::to_string(1u << i) +
                ") return mixed_dot<" + std::to_string(types[i]) + ">(row, x, K, lane); break;\n";
        result += R"METAL(} return 0.f; }
template<bool TABLE, typename P>
device const uchar* mixed_matrix(P addresses, uint expert, uint N, uint stride) {
    if constexpr (TABLE) return reinterpret_cast<device const uchar*>(addresses[expert]);
    else return reinterpret_cast<device const uchar*>(addresses) + ulong(expert) * N * stride;
}
)METAL";
        if (staged) result += R"METAL(
template<int F, typename X, class Auxiliary, class Scales>
float mixed_staged_dot(device const uchar* p0, Auxiliary p1, Scales meta,
    X x, uint row, uint K, uint lane, threadgroup uint* table) {
    float sum = 0.f;
    const uint groups = K / 32u;
    for (uint g = lane; g < groups; g += 8u) {
        GroupPayload payload;
        fetch_group<F>(payload, p0, p1, meta, row, groups, g, true);
        decode_group_chunks<F, float>(payload, staged_lookup<F>(table), g,
            [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
                uint k = g * 32u + c * 4u;
                sum += dot(low, float4(x[k], x[k + 1u], x[k + 2u], x[k + 3u]));
                sum += dot(high, float4(x[k + 16u], x[k + 17u], x[k + 18u], x[k + 19u]));
            });
    }
    sum += simd_shuffle_xor(sum, 4u);
    sum += simd_shuffle_xor(sum, 2u);
    sum += simd_shuffle_xor(sum, 1u);
    return sum;
}
)METAL";
        return result;
    };
    static const auto source = make(false);
    if (!staged) return source;
    static const auto planes = make(true);
    return planes;
}

const fast::CustomKernelFunction& project_kernel() {
    static const auto kernel = indirect_weight_metal_kernel("tinfer_gguf_mixed_project",
        {"addresses", "layout", "x", "ids", "params"}, {"y"}, R"METAL(
uint row = thread_position_in_grid.x / 8u, lane = thread_index_in_simdgroup % 8u;
uint route = thread_position_in_grid.y;
uint K = params[0], N = params[1], E = params[2], R = params[3];
if (row >= N) return;
int expert = ids[route];
float sum = 0.f;
if (expert >= 0 && uint(expert) < E && addresses[expert]) {
    auto matrix = reinterpret_cast<device const uchar*>(addresses[expert]);
    auto source = x + ulong(SHARED ? route / R : route) * K;
    sum = mixed_dispatch<MASK>(layout[expert * 2u], matrix + ulong(row) * layout[expert * 2u + 1u], source, K, lane);
}
if (lane == 0u) y[ulong(route) * N + row] = T(sum);
)METAL", mixed_header());
    return kernel;
}

const fast::CustomKernelFunction& gate_up_kernel(bool staged) {
    const auto make = [](bool staged) {
    std::vector<std::string> names{"gate", "gate_layout", "up", "up_layout", "x", "ids", "params"};
    if (staged) names.insert(names.end(), {"gate_aux", "gate_meta", "up_aux", "up_meta"});
    std::string source = staged ? R"METAL(
threadgroup uint gate_table[PlaneLayout<GF>::table_words], up_table[PlaneLayout<UF>::table_words];
if constexpr (GSTAGED) load_staged_lookup<GF>(gate_table, thread_index_in_threadgroup, 128u);
if constexpr (USTAGED) load_staged_lookup<UF>(up_table, thread_index_in_threadgroup, 128u);
threadgroup_barrier(mem_flags::mem_threadgroup);
)METAL" : "";
    source += R"METAL(
uint row = thread_position_in_grid.x / 8u, lane = thread_index_in_simdgroup % 8u;
uint route = thread_position_in_grid.y;
uint K = params[0], N = params[1], E = params[2], R = params[3];
if (row >= N) return;
int expert = ids[route];
float g = 0.f, u = 0.f;
if (expert >= 0 && uint(expert) < E) {
    auto source = x + ulong(route / R) * K;
    auto matrix_g = mixed_matrix<GMIXED || GINDIRECT>(gate, uint(expert), N, params[5]);
    auto matrix_u = mixed_matrix<UMIXED || UINDIRECT>(up, uint(expert), N, params[6]);
    if (matrix_g) {
)METAL";
    if (staged) source += R"METAL(
        if constexpr (GSTAGED) g = mixed_staged_dot<GF>(gate, gate_aux, gate_meta, source, uint(expert) * N + row, K, lane, gate_table);
        else
)METAL";
    source += R"METAL(
        if constexpr (GMIXED) g = mixed_dispatch<GMASK>(gate_layout[expert * 2u],
            matrix_g + ulong(row) * gate_layout[expert * 2u + 1u], source, K, lane);
        else g = mixed_dot<GF>(matrix_g + ulong(row) * params[5], source, K, lane);
    }
    if (matrix_u) {
)METAL";
    if (staged) source += R"METAL(
        if constexpr (USTAGED) u = mixed_staged_dot<UF>(up, up_aux, up_meta, source, uint(expert) * N + row, K, lane, up_table);
        else
)METAL";
    source += R"METAL(
        if constexpr (UMIXED) u = mixed_dispatch<UMASK>(up_layout[expert * 2u],
            matrix_u + ulong(row) * up_layout[expert * 2u + 1u], source, K, lane);
        else u = mixed_dot<UF>(matrix_u + ulong(row) * params[6], source, K, lane);
    }
}
g = float(T(g)); u = float(T(u));
if (params[4]) { float limit = as_type<float>(params[4]); g = min(g, limit); u = clamp(u, -limit, limit); }
if (lane == 0u) y[ulong(route) * N + row] = T(g / (1.f + exp(-g)) * u);
)METAL";
    return indirect_weight_metal_kernel(staged ? "tinfer_gguf_mixed_gate_up_staged" : "tinfer_gguf_mixed_gate_up",
        names, {"y"}, source, mixed_header(staged));
    };
    static const auto raw = make(false);
    if (!staged) return raw;
    static const auto planes = make(true);
    return planes;
}

const fast::CustomKernelFunction& prefill_gather_kernel() {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_mixed_prefill_gather",
        {"x", "order", "bounds", "offsets", "params"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.x;
if (tile >= uint(offsets[params[2]])) return;
uint lo = 0u, hi = params[2];
while (lo < hi) {
    uint mid = lo + (hi - lo) / 2u;
    if (uint(offsets[mid + 1u]) <= tile) lo = mid + 1u; else hi = mid;
}
uint begin = uint(bounds[lo * 2u]), rows = uint(bounds[lo * 2u + 1u]) - begin;
uint m0 = (tile - uint(offsets[lo])) * BM, K = params[0];
uint column = threadgroup_position_in_grid.y * 256u + (thread_index_in_threadgroup % 32u) * 8u;
if (column >= K) return;
for (uint row = thread_index_in_threadgroup / 32u; row < BM; row += 8u) {
    half4 low(0.h), high(0.h);
    if (m0 + row < rows) {
        uint route = uint(order[begin + m0 + row]);
        ulong source = ulong(SHARED ? route / params[3] : route) * K + column;
        low = half4(half(x[source]), half(x[source + 1u]), half(x[source + 2u]), half(x[source + 3u]));
        high = half4(half(x[source + 4u]), half(x[source + 5u]), half(x[source + 6u]), half(x[source + 7u]));
    }
    auto destination = y + ulong(tile * BM + row) * K + column;
    *reinterpret_cast<device half4*>(destination) = low;
    *reinterpret_cast<device half4*>(destination + 4u) = high;
}
)METAL");
    return kernel;
}

bool gather_prefill(int input, int output, int experts, std::size_t routes, const MlxGgufRoutePlan& plan) {
    return routes >= std::size_t(experts) * plan.tile_rows * (output > input && plan.tile_rows < 64 ? 2 : 1);
}

bool simd_prefill(int input, int output, int experts, std::size_t routes) {
    return routes >= std::size_t(experts) * (output > input ? 32 : 16) &&
        (mlx_metal_nax_disabled() || !mlx_apple_chip_starts_with("Apple M5"));
}

array gather_prefill_input(const array& input, int experts, int routes, bool shared, const MlxGgufRoutePlan& plan) {
    const int width = input.shape(-1);
    if (std::size_t(plan.tiles) * plan.tile_rows > std::size_t(INT_MAX) / width)
        throw std::overflow_error("GGUF mixed prefill activation extent is too large");
    return prefill_gather_kernel()({contiguous(input), plan.order, plan.bounds, plan.tile_offsets,
        array({uint32_t(width), uint32_t(0), uint32_t(experts), uint32_t(routes)})},
        {{plan.tiles * plan.tile_rows, width}}, {float16},
        {plan.tiles * 256, (width + 255) / 256, 1}, {256, 1, 1},
        {{"SHARED", shared}, {"BM", plan.tile_rows}}, 0.f, false, {}).front();
}

const fast::CustomKernelFunction& prefill_kernel(bool grouped_input) {
    static const auto make = [](bool grouped) {
        const std::string header = MlxGgufStagedPrefill::scalar_decoder_header() + kGgufNax + R"METAL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
template<int F, int BM, typename T>
void mixed_register_matmul(device const uchar* matrix, device const half* x, device T* y,
    device const int* order, uint begin, uint K, uint N, uint stride,
    uint rows, uint m0, uint n0, uint padded, uint tile) {
    constexpr auto description = matmul2d_descriptor(BM, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<description, execution_simdgroup> operation;
    tensor<device half, dextents<int, 2>, tensor_inline> activations(
        const_cast<device half*>(x), dextents<int, 2>(K, padded));
    auto left = activations.template slice<32, BM>(0, tile * BM);
    auto right = operation.template get_right_input_cooperative_tensor<half, half, float>();
    auto sum = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
    for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
    for (uint start = 0; start < K; start += 32u) {
        for (ushort i = 0; i < right.get_capacity(); i += 4u) {
            const auto index = right.get_multidimensional_index(i);
            const uint row = n0 + index[1], k = start + index[0];
            half4 values(0.h);
            if (row < N && k < K) values = half4(gguf_values4<F>(matrix + ulong(row) * stride, k));
            for (ushort j = 0; j < 4u; ++j) right[i + j] = values[j];
        }
        left = activations.template slice<32, BM>(start, tile * BM);
        operation.run(left, right, sum);
    }
    for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
        const auto index = sum.get_multidimensional_index(i);
        const uint row = m0 + index[1], column = n0 + index[0];
        if (row < rows && column < N) y[ulong(uint(order[begin + row])) * N + column] = T(sum[i]);
    }
}
template<int F, int BM, typename T>
void mixed_mpp_matmul(device const uchar* matrix, device const half* x, device T* y,
    device const int* order, uint begin, uint K, uint N, uint stride,
    uint rows, uint m0, uint n0, uint padded, uint tile,
    threadgroup half* stages, uint lane) {
    if constexpr (!gguf_group_format<F>() || F == 20 || F == 23) {
        mixed_register_matmul<F, BM>(matrix, x, y, order, begin, K, N, stride, rows, m0, n0, padded, tile);
    } else {
        constexpr auto description = matmul2d_descriptor(BM, 32, 32, false, true, false,
            matmul2d_descriptor::mode::multiply_accumulate);
        matmul2d<description, execution_simdgroup> operation;
        tensor<device half, dextents<int, 2>, tensor_inline> activations(
            const_cast<device half*>(x), dextents<int, 2>(K, padded));
        tensor<threadgroup half, dextents<int, 2>, tensor_inline> weights0(stages, dextents<int, 2>(32, 32));
        tensor<threadgroup half, dextents<int, 2>, tensor_inline> weights1(stages + 1024u, dextents<int, 2>(32, 32));
        auto right0 = weights0.template slice<32, 32>(0, 0);
        auto right1 = weights1.template slice<32, 32>(0, 0);
        auto left = activations.template slice<32, BM>(0, tile * BM);
        auto sum = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right0), float>();
        for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
        const uint groups = K / 32u, row = n0 + lane;
        GroupPayload payload;
        if (row < N) fetch_raw_group<F>(payload, matrix, row, groups, 0u, true);
        for (uint g = 0u; g < groups; ++g) {
            auto destination = stages + (g % 2u) * 1024u + lane * 32u;
            if (row < N) decode_group<F>(payload, destination, GgufConstantLookup<F>{}, g);
            else for (uint j = 0u; j < 32u; ++j) destination[j] = 0.h;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            if (g + 1u < groups && row < N) fetch_raw_group<F>(payload, matrix, row, groups, g + 1u,
                (g + 1u) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups);
            left = activations.template slice<32, BM>(g * 32u, tile * BM);
            if (g % 2u) operation.run(left, right1, sum); else operation.run(left, right0, sum);
        }
        for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
            const auto index = sum.get_multidimensional_index(i);
            const uint output_row = m0 + index[1], column = n0 + index[0];
            if (output_row < rows && column < N) y[ulong(uint(order[begin + output_row])) * N + column] = T(sum[i]);
        }
    }
}
template<int F, int BM, bool SHARED, typename T>
void mixed_sparse_rows(device const uchar* matrix, device const T* x, device T* y,
    device const int* order, uint begin, uint K, uint N, uint stride, uint rows, uint first, uint base_n,
    uint routes, uint thread_index, uint group) {
    const uint lane = thread_index % 8u, column = thread_index / 8u;
    for (uint j = 0u; j < 8u; ++j) {
        uint n = base_n + group * 32u + j * 4u + column;
        if (n >= N) continue;
        auto weight = matrix + ulong(n) * stride;
        for (uint r = first; r < min(rows, first + uint(BM)); ++r) {
            uint route = uint(order[begin + r]);
            auto source = x + ulong(SHARED ? route / routes : route) * K;
            float sum = 0.f;
            for (uint k = lane * 4u; k < K; k += 32u)
                sum += dot(float4(half4(gguf_values4<F>(weight, k))),
                    float4(half4(half(source[k]), half(source[k + 1u]), half(source[k + 2u]), half(source[k + 3u]))));
            sum += simd_shuffle_xor(sum, 4u);
            sum += simd_shuffle_xor(sum, 2u);
            sum += simd_shuffle_xor(sum, 1u);
            if (!lane) y[ulong(route) * N + n] = T(sum);
        }
    }
}
)METAL";
        std::string source = R"METAL(
uint tile = threadgroup_position_in_grid.y;
if (tile >= uint(offsets[params[2]])) return;
uint lo = 0u, hi = params[2];
while (lo < hi) {
    uint mid = lo + (hi - lo) / 2u;
    if (uint(offsets[mid + 1u]) <= tile) lo = mid + 1u; else hi = mid;
}
uint expert = lo;
if (!addresses[expert]) return;
uint begin = uint(bounds[expert * 2u]);
uint rows = uint(bounds[expert * 2u + 1u]) - begin;
uint m0 = (tile - uint(offsets[expert])) * BM;
uint n0 = threadgroup_position_in_grid.x * 128u + simdgroup_index_in_threadgroup * 32u;
uint K = params[0], N = params[1], stride = layout[expert * 2u + 1u];
auto matrix = reinterpret_cast<device const uchar*>(addresses[expert]);
auto input_row = [&](uint row) { uint route = uint(order[begin + row]); return SHARED ? route / params[3] : route; };
auto output_row = [&](uint row) { return uint(order[begin + row]); };
switch (layout[expert * 2u]) {
)METAL";
        if (grouped) source.insert(0, "threadgroup half stages[8192];\n");
        for (std::size_t i = 0; i < std::size(types); ++i) {
            source += "case " + std::to_string(types[i]) + ": if constexpr (MASK & " + std::to_string(1u << i) + ") ";
            if (!grouped) source += "{ if (SPARSE && rows <= 2u) mixed_sparse_rows<" + std::to_string(types[i]) +
                ", BM, SHARED>(matrix, x, y, order, begin, K, N, stride, rows, m0, threadgroup_position_in_grid.x * 128u, params[3], "
                "thread_index_in_simdgroup, simdgroup_index_in_threadgroup); else ";
            source += grouped ? "mixed_mpp_matmul<" + std::to_string(types[i]) +
                ", BM>(matrix, x, y, order, begin, K, N, stride, rows, m0, n0, params[4], tile, "
                "stages + simdgroup_index_in_threadgroup * 2048u, thread_index_in_simdgroup); break;\n"
                : "gguf_matmul<" + std::to_string(types[i]) +
                ", BM>(matrix, x, y, K, N, stride, rows, m0, n0, input_row, output_row); } break;\n";
        }
        source += "}\n";
        return indirect_weight_metal_kernel(grouped ? "tinfer_gguf_mixed_prefill_grouped" : "tinfer_gguf_mixed_prefill",
            {"addresses", "layout", "x", "order", "bounds", "offsets", "params"}, {"y"}, source,
            header, true, false);
    };
    static const auto gathered = make(true), scattered = make(false);
    return grouped_input ? gathered : scattered;
}

const fast::CustomKernelFunction& simd_prefill_kernel() {
    static const auto kernel = [] {
        const std::string header = std::string("#include <metal_simdgroup_matrix>\n") + kGgufDecode + R"METAL(
template<int F, bool SHARED, typename T>
void mixed_simd_matmul(device const uchar* matrix, device const T* x, device T* y,
    device const int* order, uint first, uint end, uint K, uint N, uint stride, uint R,
    uint base_n, uint t, uint sg, threadgroup half* a_tile, threadgroup half* b_tile,
    threadgroup float* result) {
    simdgroup_float8x8 c0(0.f), c1(0.f);
    for (uint base_k = 0u; base_k < K; base_k += 32u) {
        for (uint i = t; i < 16u * 32u; i += 128u) {
            uint route_index = first + i / 32u, k = base_k + i % 32u;
            uint route = route_index < end ? uint(order[route_index]) : 0u;
            uint row = SHARED ? route / R : route;
            a_tile[i] = route_index < end ? half(x[ulong(row) * K + k]) : half(0);
        }
        for (uint i = t; i < 32u * 8u; i += 128u) {
            uint n = base_n + i / 8u, k = base_k + (i % 8u) * 4u;
            half4 values(0.h);
            if (n < N) values = half4(gguf_values4<F>(matrix + ulong(n) * stride, k));
            for (uint j = 0u; j < 4u; ++j) b_tile[((i % 8u) * 4u + j) * 32u + i / 8u] = values[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0u; k < 32u; k += 8u) {
            simdgroup_half8x8 a, b0, b1;
            simdgroup_load(a, a_tile + (sg / 2u) * 8u * 32u + k, 32);
            simdgroup_load(b0, b_tile + k * 32u + (sg % 2u) * 16u, 32);
            simdgroup_load(b1, b_tile + k * 32u + (sg % 2u) * 16u + 8u, 32);
            simdgroup_multiply_accumulate(c0, a, b0, c0);
            simdgroup_multiply_accumulate(c1, a, b1, c1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(c0, result + (sg / 2u) * 8u * 32u + (sg % 2u) * 16u, 32);
    simdgroup_store(c1, result + (sg / 2u) * 8u * 32u + (sg % 2u) * 16u + 8u, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < 16u * 32u; i += 128u) {
        uint route_index = first + i / 32u, n = base_n + i % 32u;
        if (route_index < end && n < N) y[ulong(uint(order[route_index])) * N + n] = T(result[i]);
    }
}
)METAL";
        std::string source = R"METAL(
uint tile = threadgroup_position_in_grid.y;
if (tile >= uint(offsets[params[2]])) return;
uint lo = 0u, hi = params[2];
while (lo < hi) {
    uint mid = lo + (hi - lo) / 2u;
    if (uint(offsets[mid + 1u]) <= tile) lo = mid + 1u; else hi = mid;
}
uint expert = lo;
if (!addresses[expert]) return;
uint start = uint(bounds[expert * 2u]) + (tile - uint(offsets[expert])) * BM;
uint first = start + threadgroup_position_in_grid.z * 16u;
uint end = min(uint(bounds[expert * 2u + 1u]), start + uint(BM));
if (first >= end) return;
threadgroup half a_tile[16 * 32], b_tile[32 * 32];
threadgroup float result[16 * 32];
auto matrix = reinterpret_cast<device const uchar*>(addresses[expert]);
switch (layout[expert * 2u]) {
)METAL";
        for (std::size_t i = 0; i < std::size(types); ++i)
            source += "case " + std::to_string(types[i]) + ": if constexpr (MASK & " + std::to_string(1u << i) +
                ") mixed_simd_matmul<" + std::to_string(types[i]) + ", SHARED>(matrix, x, y, order, first, end, "
                "params[0], params[1], layout[expert * 2u + 1u], params[3], threadgroup_position_in_grid.x * 32u, "
                "thread_index_in_threadgroup, simdgroup_index_in_threadgroup, a_tile, b_tile, result); break;\n";
        source += "}\n";
        return indirect_weight_metal_kernel("tinfer_gguf_mixed_prefill_simd",
            {"addresses", "layout", "x", "order", "bounds", "offsets", "params"}, {"y"}, source, header);
    }();
    return kernel;
}

const fast::CustomKernelFunction& weighted_kernel() {
    static const auto kernel = indirect_weight_metal_kernel("tinfer_gguf_mixed_down_reduce",
        {"addresses", "layout", "x", "ids", "probabilities", "params"}, {"y"}, R"METAL(
uint lane = thread_index_in_simdgroup % 8u;
uint column = thread_index_in_simdgroup / 8u;
uint group = simdgroup_index_in_threadgroup;
uint row = threadgroup_position_in_grid.x * 4u + column;
uint token = threadgroup_position_in_grid.y;
uint K = params[0], N = params[1], E = params[2], R = params[3];
float sum = 0.f;
for (uint route = group; route < R; route += 8u) {
    int expert = ids[token * R + route];
    if (row < N && expert >= 0 && uint(expert) < E) {
        auto matrix = mixed_matrix<MIXED || INDIRECT>(addresses, uint(expert), N, params[4]);
        if (matrix) {
            auto input = x + ulong(token * R + route) * K;
            float value;
            if constexpr (MIXED) value = mixed_dispatch<MASK>(layout[expert * 2u],
                matrix + ulong(row) * layout[expert * 2u + 1u], input, K, lane);
            else value = mixed_dot<F>(matrix + ulong(row) * params[4], input, K, lane);
            sum += float(T(value)) * probabilities[token * R + route];
        }
    }
}
threadgroup float partial[32];
if (lane == 0u) partial[group * 4u + column] = sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u && lane == 0u && row < N) {
    float value = 0.f;
    for (uint g = 0u; g < 8u; ++g) value += partial[g * 4u + column];
    y[ulong(token) * N + row] = T(value);
}
)METAL", mixed_header());
    return kernel;
}
}

MlxGgufWeight MlxGgufWeight::from_mixed_experts(Shape shape, const std::vector<std::int32_t>& ids,
    const std::vector<std::string>& dtypes, const std::vector<array>& packed,
    std::shared_ptr<MlxWeightResidency> residency) {
    if (shape.size() != 3 || std::any_of(shape.begin(), shape.end(), [](int d) { return d <= 0; }) ||
        ids.empty() || ids.size() != packed.size() || ids.size() != dtypes.size())
        throw std::invalid_argument("invalid GGUF mixed expert table");
    std::vector<std::uint64_t> addresses(shape[0], 0);
    std::vector<std::uint32_t> layout(std::size_t(shape[0]) * 2, 0);
    std::size_t allocated = 0;
    int mask = 0;
    eval(packed);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto stride = gguf_packed_row_bytes(dtypes[i], shape[2]);
        const auto& value = packed[i];
        if (ids[i] < 0 || ids[i] >= shape[0] || addresses[ids[i]] || value.dtype() != uint8 ||
            stride > INT_MAX || value.nbytes() != std::size_t(shape[1]) * stride || !value.flags().row_contiguous)
            throw std::invalid_argument("invalid GGUF mixed expert allocation");
        const auto index = format_index(dtypes[i]);
        const auto* buffer = static_cast<const MTL::Buffer*>(value.buffer().ptr());
        addresses[ids[i]] = buffer->gpuAddress() + value.offset();
        layout[ids[i] * 2] = types[index]; layout[ids[i] * 2 + 1] = std::uint32_t(stride);
        allocated += buffer->allocatedSize(); mask |= 1 << index;
    }
    struct Lifetime {
        std::vector<array> experts;
        array table;
        std::shared_ptr<MlxWeightResidency> residency;
    };
    auto lifetime = std::make_shared<Lifetime>(Lifetime{packed, array(addresses.begin(), Shape{shape[0]}, uint64),
        residency ? std::move(residency) : MlxWeightResidency::retain_arrays(packed, allocated, true)});
    auto table = mlx_buffer_view(lifetime->table.buffer(), lifetime->table.shape(), uint64, lifetime);
    register_indirect_weight_pool(table, lifetime->residency);
    MlxGgufWeight result(std::move(table), std::move(shape), -1, 0, true);
    result.mixed_layout_ = array(layout.begin(), Shape{result.expert_count(), 2}, uint32);
    result.mixed_mask_ = mask;
    for (const auto& value : packed) result.mixed_payload_bytes_ += value.nbytes();
    return result;
}

MlxGgufWeight MlxGgufWeight::from_mixed_address_table(Shape shape, const std::vector<std::string>& dtypes,
    array table, std::shared_ptr<MlxWeightResidency> residency) {
    if (shape.size() != 3 || std::any_of(shape.begin(), shape.end(), [](int d) { return d <= 0; }) ||
        dtypes.size() != std::size_t(shape[0]) || table.dtype() != uint64 || table.shape() != Shape{shape[0]} ||
        !table.flags().row_contiguous || !residency)
        throw std::invalid_argument("invalid GGUF mixed pooled expert table");
    std::vector<std::uint32_t> layout(std::size_t(shape[0]) * 2);
    int mask = 0;
    for (int e = 0; e < shape[0]; ++e) {
        const auto stride = gguf_packed_row_bytes(dtypes[e], shape[2]);
        if (stride > INT_MAX) throw std::overflow_error("GGUF mixed row extent is too large");
        const auto index = format_index(dtypes[e]);
        layout[e * 2] = types[index]; layout[e * 2 + 1] = std::uint32_t(stride); mask |= 1 << index;
    }
    table.eval();
    struct Lifetime { array table; std::shared_ptr<MlxWeightResidency> residency; };
    auto owner = std::make_shared<Lifetime>(Lifetime{table, std::move(residency)});
    auto handle = mlx_buffer_view(table.buffer(), table.shape(), uint64, owner);
    register_indirect_weight_pool(handle, owner->residency);
    MlxGgufWeight result(std::move(handle), std::move(shape), -1, 0, true);
    result.mixed_layout_ = array(layout.begin(), Shape{result.expert_count(), 2}, uint32);
    result.mixed_mask_ = mask;
    return result;
}

array MlxGgufWeight::mixed_matmul(const array& input, const array& ids, bool shared_input,
    const MlxGgufRoutePlan* plan, const array* gathered_input) const {
    const int tokens = ids.shape(0), routes = ids.shape(1), n = output_size();
    const Shape shape{tokens, routes, n};
    const auto params = array({uint32_t(input_size()), uint32_t(n), uint32_t(expert_count()), uint32_t(routes)});
    if (tokens > 8 && simd_prefill(input_size(), n, expert_count(), ids.size())) {
        std::optional<MlxGgufRoutePlan> owned;
        if (!plan) { owned = gguf_route_plan(ids, expert_count(), true); plan = &*owned; }
        return simd_prefill_kernel()({bytes_, *mixed_layout_, contiguous(input), plan->order, plan->bounds, plan->tile_offsets, params},
            {shape}, {input.dtype()}, {((n + 31) / 32) * 128, plan->tiles, (plan->tile_rows + 15) / 16}, {128, 1, 1},
            {{"MASK", mixed_mask_}, {"T", input.dtype()}, {"SHARED", shared_input}, {"BM", plan->tile_rows}}, 0.f, false, {}).front();
    }
    if (tokens > 8 && !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5")) {
        std::optional<MlxGgufRoutePlan> owned;
        if (!plan) { owned = gguf_route_plan(ids, expert_count(), true); plan = &*owned; }
        if (!gather_prefill(input_size(), n, expert_count(), ids.size(), *plan))
            return prefill_kernel(false)({bytes_, *mixed_layout_, contiguous(input), plan->order, plan->bounds, plan->tile_offsets, params},
                {shape}, {input.dtype()}, {((n + 127) / 128) * 128, plan->tiles, 1}, {128, 1, 1},
                {{"MASK", mixed_mask_}, {"T", input.dtype()}, {"SHARED", shared_input}, {"BM", plan->tile_rows},
                    {"SPARSE", ids.size() <= std::size_t(expert_count()) * 2}}, 0.f, false, {}).front();
        const auto gathered = gathered_input ? *gathered_input
            : gather_prefill_input(input, expert_count(), routes, shared_input, *plan);
        return prefill_kernel(true)({bytes_, *mixed_layout_, gathered, plan->order, plan->bounds, plan->tile_offsets,
            array({uint32_t(input_size()), uint32_t(n), uint32_t(expert_count()), uint32_t(routes), uint32_t(plan->tiles * plan->tile_rows)})},
            {shape}, {input.dtype()}, {((n + 127) / 128) * 128, plan->tiles, 1}, {128, 1, 1},
            {{"MASK", mixed_mask_}, {"T", input.dtype()}, {"SHARED", false}, {"BM", plan->tile_rows}}, 0.f, false, {}).front();
    }
    return project_kernel()({bytes_, *mixed_layout_, contiguous(input), contiguous(astype(ids, int32)), params},
        {shape}, {input.dtype()}, {((n + 15) / 16) * 128, int(ids.size()), 1}, {128, 1, 1},
        {{"MASK", mixed_mask_}, {"T", input.dtype()}, {"SHARED", shared_input}}, std::nullopt, false, {}).front();
}

array MlxGgufWeight::mixed_swiglu(const MlxGgufWeight& up, const array& input, const array& ids,
    float limit, const MlxGgufRoutePlan* plan) const {
    if (ids.shape(0) > 8) {
        std::optional<MlxGgufRoutePlan> owned;
        std::optional<array> gathered;
        const bool nax = !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
        if (nax || simd_prefill(input_size(), output_size(), expert_count(), ids.size())) {
            if (!plan) { owned = gguf_route_plan(ids, expert_count(), true); plan = &*owned; }
            if (nax && gather_prefill(input_size(), output_size(), expert_count(), ids.size(), *plan))
                gathered = gather_prefill_input(input, expert_count(), ids.shape(1), true, *plan);
        }
        const auto* prepared = gathered ? &*gathered : nullptr;
        const auto g = mixed_matmul(input, ids, true, plan, prepared);
        const auto u = up.mixed_matmul(input, ids, true, plan, prepared);
        return limit > 0.f ? moe_limited_swiglu_pair(g, u, limit) : moe_swiglu_pair(g, u);
    }
    const int n = output_size();
    const bool staged = planes_.has_value() || up.planes_.has_value();
    const auto params = array({uint32_t(input_size()), uint32_t(n), uint32_t(expert_count()), uint32_t(ids.shape(1)),
        std::bit_cast<uint32_t>(limit), planes_ ? 0u : uint32_t(row_bytes_), up.planes_ ? 0u : uint32_t(up.row_bytes_)});
    std::vector<array> inputs{bytes_, mixed_layout_ ? *mixed_layout_ : params,
        up.bytes_, up.mixed_layout_ ? *up.mixed_layout_ : params, contiguous(input), contiguous(astype(ids, int32)),
        params};
    if (staged) inputs.insert(inputs.end(), {planes_ ? (*planes_)[0] : params, planes_ ? (*planes_)[1] : params,
        up.planes_ ? (*up.planes_)[0] : params, up.planes_ ? (*up.planes_)[1] : params});
    return gate_up_kernel(staged)(inputs,
        {{ids.shape(0), ids.shape(1), n}}, {input.dtype()}, {((n + 15) / 16) * 128, int(ids.size()), 1}, {128, 1, 1},
        {{"GMASK", mixed_mask_}, {"UMASK", up.mixed_mask_}, {"T", input.dtype()},
         {"GMIXED", bool(mixed_layout_)}, {"UMIXED", bool(up.mixed_layout_)},
         {"GSTAGED", bool(planes_)}, {"USTAGED", bool(up.planes_)},
         {"GINDIRECT", indirect_}, {"UINDIRECT", up.indirect_},
         {"GF", mixed_layout_ ? 0 : format_}, {"UF", up.mixed_layout_ ? 0 : up.format_}}, std::nullopt, false, {}).front();
}

array MlxGgufWeight::routed_weighted_matmul(const array& input, const array& ids,
    const array& probabilities, const MlxGgufRoutePlan* plan) const {
    if (shape_.size() != 3 || ids.ndim() != 2 || probabilities.shape() != ids.shape() ||
        (ids.dtype() != int32 && ids.dtype() != uint32) || input.ndim() == 0 || input.shape(-1) != input_size() ||
        input.size() / input_size() != ids.size() ||
        (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("GGUF weighted routed projection geometry mismatch");
    if (ids.shape(0) == 0) return zeros({0, output_size()}, input.dtype());
    if (rotation_) { auto plain = *this; plain.rotation_.reset(); return plain.routed_weighted_matmul(rotate_input(input), ids, probabilities, plan); }
    if (ids.shape(0) == 1 && ids.shape(1) > 0 && MlxGgufStagedPrefill::prefers_weighted_down(*this))
        return MlxGgufStagedPrefill::weighted_down(*this, input, ids, probabilities, input.dtype());
    if (planes_ || ids.shape(0) > 8 || (!mixed_layout_ && (format_ < 147 || format_ > 149 || ids.shape(0) > 6)))
        return moe_weighted_reduce(routed_matmul(input, ids, false, plan), probabilities);
    const int n = output_size();
    const auto params = array({uint32_t(input_size()), uint32_t(n), uint32_t(expert_count()), uint32_t(ids.shape(1)),
        uint32_t(row_bytes_)});
    return weighted_kernel()({bytes_, mixed_layout_ ? *mixed_layout_ : params, contiguous(input), contiguous(astype(ids, int32)),
        contiguous(astype(probabilities, float32)),
        params},
        {{ids.shape(0), n}}, {input.dtype()}, {((n + 3) / 4) * 256, ids.shape(0), 1}, {256, 1, 1},
        {{"MASK", mixed_mask_}, {"T", input.dtype()}, {"F", mixed_layout_ ? 0 : format_}, {"MIXED", bool(mixed_layout_)},
            {"INDIRECT", indirect_}}, std::nullopt, false, {}).front();
}
}
