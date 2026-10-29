#include "mlx_affine.h"

#include <mlx/fast.h>

namespace mfq::metal {
namespace {
using namespace mlx::core;

constexpr const char* kDot = R"METAL(
template <int Q, int K, typename W>
inline uint affine_code(W weight, uint row, uint column) {
    const uint bit = column * Q;
    const uint word = row * uint(K / 32 * Q) + bit / 32u;
    const uint shift = bit & 31u;
    uint value = weight[word] >> shift;
    if (shift + Q > 32u) value |= weight[word + 1u] << (32u - shift);
    return value & ((1u << Q) - 1u);
}
template <int Q, int GS, int K, int LANES, typename W, typename S, typename B, typename X>
inline float affine_dot(W weight, S scales, B biases, X input, uint row, uint lane,
    uint begin = 0u, uint end = K) {
    float value = 0.0f;
    constexpr uint GROUP_STEP = LANES * 4 > GS ? uint(LANES * 4 / GS) : 1u;
    for (uint group = begin / GS + lane * 4u / GS; group < end / GS; group += GROUP_STEP) {
        float dot_value = 0.0f, input_sum = 0.0f;
        for (uint local = lane * 4u % GS; local < uint(GS); local += uint(LANES * 4)) {
            const uint column = group * GS + local;
            const float4 x(input[column], input[column + 1u], input[column + 2u], input[column + 3u]);
            const float4 q(affine_code<Q, K>(weight, row, column),
                affine_code<Q, K>(weight, row, column + 1u), affine_code<Q, K>(weight, row, column + 2u),
                affine_code<Q, K>(weight, row, column + 3u));
            dot_value += dot(x, q);
            input_sum += x.x + x.y + x.z + x.w;
        }
        const uint index = row * uint(K / GS) + group;
        value = fma(float(scales[index]), dot_value, fma(float(biases[index]), input_sum, value));
    }
    if (LANES >= 32) value += simd_shuffle_down(value, 16u);
    if (LANES >= 16) value += simd_shuffle_down(value, 8u);
    value += simd_shuffle_down(value, 4u);
    value += simd_shuffle_down(value, 2u);
    value += simd_shuffle_down(value, 1u);
    return value;
}
)METAL";

const fast::CustomKernelFunction& down_kernel() {
    static const CompileOptions options(static_cast<int>(MathMode::Fast));
    static const auto kernel = fast::metal_kernel("tinfer_affine_hc_down_gate",
        {"input", "down", "down_scales", "down_biases", "injection", "inject_scales", "inject_biases"},
        {"parts"}, R"METAL(
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
        const uint split = threadgroup_position_in_grid.y, token = threadgroup_position_in_grid.z;
        const uint begin = split * uint(WIDTH / SPLITS), end = begin + uint(WIDTH / SPLITS);
        if (row < uint(LOW)) {
            const float value = affine_dot<DQ, DGS, WIDTH, 32>(down, down_scales, down_biases,
                input + token * WIDTH, row, lane, begin, end);
            if (lane == 0u) parts[(token * SPLITS + split) * (LOW + HC) + row] = value;
        } else if (HAS_INJECT && row < uint(LOW + HC)) {
            const uint index = row - LOW;
            const float value = affine_dot<IQ, IGS, WIDTH, 32>(injection, inject_scales, inject_biases,
                input + token * WIDTH, index, lane, begin, end);
            if (lane == 0u) parts[(token * SPLITS + split) * (LOW + HC) + row] = value;
        }
        )METAL", kDot, true, false, options);
    return kernel;
}

const fast::CustomKernelFunction& mix_kernel() {
    static const CompileOptions options(static_cast<int>(MathMode::Fast));
    static const auto kernel = fast::metal_kernel("tinfer_affine_hc_up_mix",
        {"parts", "up", "scales", "biases", "normalized"}, {"output", "injection"}, R"METAL(
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x * 16u
            + simdgroup_index_in_threadgroup * 4u + lane / 8u;
        const uint token = threadgroup_position_in_grid.y;
        threadgroup T low[LOW];
        for (uint i = thread_index_in_threadgroup; i < uint(LOW); i += 128u) {
            float value = 0.f;
            for (uint split = 0; split < uint(SPLITS); ++split)
                value += parts[(token * SPLITS + split) * (LOW + HC) + i];
            const T projected = T(value), scaled = T(float(projected) / float(HC));
            const T gate = T(1.f / (1.f + exp(-float(scaled))));
            low[i] = T(float(scaled) * float(gate));
        }
        if (HAS_INJECT && threadgroup_position_in_grid.x == 0u && thread_index_in_threadgroup < uint(HC)) {
            const uint i = thread_index_in_threadgroup;
            float value = 0.f;
            for (uint split = 0; split < uint(SPLITS); ++split)
                value += parts[(token * SPLITS + split) * (LOW + HC) + LOW + i];
            const T projected = T(value), scaled = T(float(projected) / float(HC));
            const T gate = T(1.f / (1.f + exp(-float(scaled))));
            injection[token * HC + i] = T(2.f * float(gate));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (row >= uint(HIDDEN)) return;
        float sum = 0.0f;
        for (uint stream = 0; stream < uint(HC); ++stream) {
            const uint column = stream * HIDDEN + row;
            const float value = affine_dot<Q, GS, LOW, 8>(up, scales, biases,
                low, column, lane & 7u);
            if ((lane & 7u) == 0u) {
                const T projected = T(value);
                const T gate = T(1.0f / (1.0f + exp(-float(projected))));
                const T product = T(float(gate) * float(normalized[token * HIDDEN * HC + column]));
                sum += float(product);
            }
        }
        if ((lane & 7u) == 0u) output[token * HIDDEN + row] = T(sum / float(HC));
        )METAL", kDot, true, false, options);
    return kernel;
}
}

std::optional<std::vector<array>> MlxAffineWeight::gated_residual(const array& normalized,
    const MlxAffineWeight& up, const MlxAffineWeight* injection, int streams) const {
    const auto dtype = normalized.dtype();
    if (values_.ndim() != 2 || up.values_.ndim() != 2 || normalized.ndim() < 2 ||
        (dtype != float16 && dtype != bfloat16) || dtype != scales_.dtype() || dtype != up.scales_.dtype() ||
        streams < 2 || streams > 8 || width_ % streams || normalized.shape(-1) != width_ ||
        up.input_size() != output_size() || up.output_size() != width_ || output_size() > 1024 ||
        (injection && (injection->values_.ndim() != 2 || injection->width_ != width_ ||
            injection->output_size() != streams || injection->scales_.dtype() != dtype))) return std::nullopt;
    const int rows = int(normalized.size() / width_);
    if (rows < 1 || rows > 6) return std::nullopt;
    const auto& inject = injection ? *injection : *this;
    const int splits = rows == 1 && width_ >= 4096 &&
        width_ / streams % group_size_ == 0 && width_ / streams % inject.group_size_ == 0 ? streams : 1;
    auto source = reshape(normalized, {rows, width_});
    auto first = down_kernel()({source, values_, scales_, biases_, inject.values_, inject.scales_, inject.biases_},
        {{rows, splits, output_size() + streams}}, {float32},
        {((output_size() + (injection ? streams : 0) + 3) / 4) * 128, splits, rows}, {128, 1, 1},
        {{"T", dtype}, {"WIDTH", width_}, {"LOW", output_size()}, {"HC", streams},
         {"DQ", bits_}, {"DGS", group_size_}, {"IQ", inject.bits_}, {"IGS", inject.group_size_},
         {"HAS_INJECT", int(injection != nullptr)}, {"SPLITS", splits}}, std::nullopt, false, {});
    auto shape = normalized.shape(); shape.back() /= streams;
    auto gate_shape = normalized.shape(); gate_shape.back() = streams;
    auto mixed = mix_kernel()({first[0], up.values_, up.scales_, up.biases_, source}, {shape, gate_shape}, {dtype, dtype},
        {((width_ / streams + 15) / 16) * 128, rows, 1}, {128, 1, 1},
        {{"T", dtype}, {"Q", up.bits_}, {"GS", up.group_size_}, {"LOW", output_size()},
         {"HIDDEN", width_ / streams}, {"HC", streams}, {"SPLITS", splits},
         {"HAS_INJECT", int(injection != nullptr)}}, std::nullopt, false, {});
    std::vector<array> result{std::move(mixed[0])};
    if (injection) {
        result.push_back(std::move(mixed[1]));
    }
    return result;
}
} // namespace mfq::metal
