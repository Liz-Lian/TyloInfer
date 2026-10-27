#include "mlx_gguf_prefill.h"
#include "mlx_gguf_staged.h"
#include "gguf_embedded.h"
#include "mlx_indirect_weight_residency.h"
#include "mlx_platform.h"

#include <climits>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;

const std::string& header() {
    static const std::string source = std::string(kGgufDecode) + R"METAL(
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
    static const auto kernel = fast::metal_kernel("tinfer_gguf_prefill_positions",
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
    static const auto kernel = fast::metal_kernel("tinfer_gguf_prefill_gather",
        {"x", "slots", "offsets", "params"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.x;
uint k = params[0], routes = params[1], tile_rows = params[3];
if (tile >= uint(offsets[params[2]])) return;
uint column = threadgroup_position_in_grid.y * 256u + (thread_index_in_threadgroup % 32u) * 8u;
if (column >= k) return;
for (uint local = thread_index_in_threadgroup / 32u; local < tile_rows; local += 8u) {
    uint row = tile * tile_rows + local;
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

const fast::CustomKernelFunction& restore_kernel() {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_grouped_restore",
        {"x", "positions", "params"}, {"y"}, R"METAL(
uint at = thread_position_in_grid.x;
uint n = params[0];
if (at >= params[1] * n) return;
int row = positions[at / n];
y[at] = row < 0 ? T(0) : T(x[ulong(row) * n + at % n]);
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& gate_up_kernel() {
    static const auto kernel = indirect_weight_metal_kernel("tinfer_gguf_prefill_register_gate_up",
        {"gate", "up", "x", "bounds", "offsets", "params"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.y, experts = params[3];
if (tile >= uint(offsets[experts])) return;
uint e = 0u, high = experts;
while (e < high) {
    uint middle = e + (high - e) / 2u;
    if (uint(offsets[middle + 1u]) <= tile) e = middle + 1u; else high = middle;
}
uint k = params[0], n = params[1], padded = params[2], row0 = tile * 16u;
uint n0 = (threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup) * 32u;
threadgroup uint gate_table[GgufConstantLookup<FG>::storage_size];
threadgroup uint up_table[FG == FU ? 1u : GgufConstantLookup<FU>::storage_size];
gguf_load_lookup<FG>(gate_table, thread_index_in_threadgroup, 128u);
if constexpr (FG != FU) gguf_load_lookup<FU>(up_table, thread_index_in_threadgroup, 128u);
threadgroup_barrier(mem_flags::mem_threadgroup);
auto g = expert_data<IG>(gate, e, n, params[4]);
auto u = expert_data<IU>(up, e, n, params[5]);
constexpr auto description = matmul2d_descriptor(16, 32, 32, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> activations(
    const_cast<device half*>(x), dextents<int, 2>(k, padded));
auto right_g = operation.template get_right_input_cooperative_tensor<half, half, float>();
auto right_u = operation.template get_right_input_cooperative_tensor<half, half, float>();
auto input = activations.template slice<32, 16>(0, row0);
auto sum_g = operation.template get_destination_cooperative_tensor<decltype(input), decltype(right_g), float>();
auto sum_u = operation.template get_destination_cooperative_tensor<decltype(input), decltype(right_u), float>();
for (ushort i = 0; i < sum_g.get_capacity(); ++i) { sum_g[i] = 0.f; sum_u[i] = 0.f; }
for (uint start = 0; start < k; start += 32u) {
    fill_right<FG>(right_g, g, n0, start, n, k, params[4], GgufThreadgroupLookup<FG>{gate_table});
    fill_right<FU>(right_u, u, n0, start, n, k, params[5], GgufThreadgroupLookup<FU>{FG == FU ? gate_table : up_table});
    input = activations.template slice<32, 16>(start, row0);
    operation.run(input, right_g, sum_g);
    operation.run(input, right_u, sum_u);
}
for (ushort i = 0; i < sum_g.get_capacity(); ++i) if (sum_g.is_valid_element(i)) {
    auto index = sum_g.get_multidimensional_index(i);
    uint column = n0 + index[0], row = row0 + index[1];
    if (column < n) {
        float a = float(T(sum_g[i])), b = float(T(sum_u[i]));
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
threadgroup uint table[GgufConstantLookup<F>::storage_size];
gguf_load_lookup<F>(table, thread_index_in_threadgroup, 64u);
threadgroup_barrier(mem_flags::mem_threadgroup);
auto matrix = expert_data<INDIRECT>(packed, e, n, params[4]);
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
    static const auto kernel = fast::metal_kernel("tinfer_gguf_prefill_reduce",
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

int gguf_prefill_tile_rows(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const MlxGgufWeight& down, const array& ids) {
    return gate.expert_count() <= 512 && MlxGgufStagedPrefill::supports(gate) &&
        MlxGgufStagedPrefill::supports(up) && MlxGgufStagedPrefill::supports(down)
        ? (ids.size() <= std::size_t(gate.expert_count()) ? 8 : 32) : 16;
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
    : positions_(ids), input_(input), dtype_(input.dtype()),
      tokens_(ids.ndim() == 2 ? ids.shape(0) : 0), selected_(ids.ndim() == 2 ? ids.shape(1) : 0), padded_rows_(0) {
    if (input.ndim() < 2 || input.shape(-1) <= 0 || mlx_metal_nax_disabled() ||
        !mlx_apple_chip_starts_with("Apple M5") || ids.ndim() != 2 || !ids.size() ||
        input.size() / input.shape(-1) != std::size_t(tokens_) * (shared_input ? 1 : selected_) ||
        (include_shared && (!shared_input || tile_rows == 16)) ||
        (tile_rows != 8 && tile_rows != 16 && tile_rows != 32))
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
        grouped_.emplace(gguf_group_routes(ids, experts, tile_rows, include_shared));
        if (unique_per_token) grouped_->maximum_live_rows = std::min(tile_rows, tokens_);
        tiles = grouped_->tiles;
        padded_rows_ = tiles * tile_rows;
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
    input_ = gather_kernel()({contiguous(input), slots, offsets,
        array({uint32_t(width), uint32_t(shared_input ? selected_ : 1), uint32_t(experts + int(include_shared)), uint32_t(tile_rows)})},
        {{padded_rows_, width}}, {float16}, {tiles * 256, (width + 255) / 256, 1}, {256, 1, 1},
        {}, std::nullopt, false, {}).front();
}

array MlxGgufPrefillPlan::restore(const array& grouped) const {
    if (grouped.ndim() != 2 || grouped.shape(0) != padded_rows_)
        throw std::invalid_argument("GGUF grouped projection output geometry mismatch");
    const auto elements = std::size_t(tokens_) * selected_ * grouped.shape(1);
    if (elements > INT_MAX - 127) throw std::overflow_error("GGUF restored projection extent is too large");
    return restore_kernel()({grouped, positions_, array({uint32_t(grouped.shape(1)), uint32_t(tokens_ * selected_)})},
        {{tokens_, selected_, grouped.shape(1)}}, {dtype_}, {int((elements + 127) / 128 * 128), 1, 1},
        {128, 1, 1}, {{"T", dtype_}}, std::nullopt, false, {}).front();
}

array MlxGgufPrefillPlan::project(const MlxGgufWeight& weight) const {
    if (!grouped_ || !MlxGgufStagedPrefill::supports(weight) || weight.input_size() != input_.shape(1))
        throw std::invalid_argument("GGUF grouped projection weight geometry mismatch");
    return restore(MlxGgufStagedPrefill::down(weight, input_, *grouped_, dtype_));
}

array MlxGgufPrefillPlan::gate_up(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const MlxGgufWeight* shared_gate, const MlxGgufWeight* shared_up) const {
    if (gate.shape_.size() != 3 || gate.shape_ != up.shape_ || gate.input_size() != input_.shape(1))
        throw std::invalid_argument("GGUF prefill Gate/Up weights disagree");
    if (!grouped_ && (gate.planes_ || up.planes_))
        throw std::invalid_argument("GGUF compressed planes require a grouped route plan");
    if (grouped_) {
        if (!MlxGgufStagedPrefill::supports(gate) || !MlxGgufStagedPrefill::supports(up))
            throw std::invalid_argument("GGUF staged Gate/Up format is unsupported");
        return MlxGgufStagedPrefill::gate_up(gate, up, input_, *grouped_, dtype_, shared_gate, shared_up);
    }
    const int n = gate.output_size();
    return gate_up_kernel()({gate.bytes_, up.bytes_, input_, routes_->bounds, routes_->tile_offsets,
        array({uint32_t(gate.input_size()), uint32_t(n), uint32_t(padded_rows_), uint32_t(gate.expert_count()),
            uint32_t(gate.row_bytes_), uint32_t(up.row_bytes_)})}, {{padded_rows_, n}}, {dtype_},
        {((n + 127) / 128) * 128, routes_->tiles, 1}, {128, 1, 1},
        {{"FG", gate.format_}, {"FU", up.format_}, {"T", dtype_}, {"IG", gate.indirect_}, {"IU", up.indirect_}},
        0.f, false, {}).front();
}

array MlxGgufPrefillPlan::down(const MlxGgufWeight& weight, const array& hidden,
    const array& probabilities, bool float_accumulation, const MlxGgufWeight* shared) const {
    const int n = weight.output_size();
    if (weight.shape_.size() != 3 || hidden.shape() != Shape{padded_rows_, weight.input_size()} ||
        probabilities.shape() != Shape{tokens_, selected_}) throw std::invalid_argument("GGUF prefill Down inputs disagree");
    if (!grouped_ && weight.planes_)
        throw std::invalid_argument("GGUF compressed planes require a grouped route plan");
    const auto regular_down = [&] { return down_kernel()({weight.bytes_, astype(hidden, float16), routes_->tile_offsets,
        array({uint32_t(weight.input_size()), uint32_t(n), uint32_t(padded_rows_), uint32_t(weight.expert_count()),
            uint32_t(weight.row_bytes_)})}, {{padded_rows_, n}}, {dtype_},
        {((n + 127) / 128) * 64, routes_->tiles, 1}, {64, 1, 1},
        {{"F", weight.format_}, {"T", dtype_}, {"INDIRECT", weight.indirect_}}, 0.f, false, {}).front(); };
    if (grouped_ && !MlxGgufStagedPrefill::supports(weight))
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
