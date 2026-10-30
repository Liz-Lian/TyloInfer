#include "mlx_gguf.h"
#include "mlx_cached_kernel.h"
#include "mlx_tensor.h"

namespace mfq::metal {
namespace {
using namespace mlx::core;
constexpr const char* kDot = R"METAL(
struct HcQ8Block { half scale; char codes[32]; };
template<int LANES, typename X>
inline float q8_hc_dot(device const uchar* packed, X input, uint row, uint lane, uint width) {
    const uint blocks = width / 32u;
    const auto weights = reinterpret_cast<device const HcQ8Block*>(packed) + row * blocks;
    float result = 0.0f;
    if constexpr (LANES == 4 || LANES == 8) {
        for (uint block = lane; block < blocks; block += uint(LANES)) {
            float sum = 0.0f;
            for (uint local = 0; local < 32u; local += 4u) {
                const uint column = block * 32u + local;
                const float4 x(input[column], input[column + 1u], input[column + 2u], input[column + 3u]);
                const float4 code(weights[block].codes[local], weights[block].codes[local + 1u],
                    weights[block].codes[local + 2u], weights[block].codes[local + 3u]);
                sum += dot(x, code);
            }
            result += float(weights[block].scale) * sum;
        }
    } else for (uint block = lane * 4u / 32u; block < blocks; block += uint(LANES * 4 / 32)) {
        float sum = 0.0f;
        for (uint local = lane * 4u % 32u; local < 32u; local += uint(LANES * 4)) {
            const uint column = block * 32u + local;
            const float4 x(input[column], input[column + 1u], input[column + 2u], input[column + 3u]);
            const float4 code(weights[block].codes[local], weights[block].codes[local + 1u],
                weights[block].codes[local + 2u], weights[block].codes[local + 3u]);
            sum += dot(x, code);
        }
        result += float(weights[block].scale) * sum;
    }
    if (LANES == 32) return simd_sum(result);
    if (LANES >= 8) result += simd_shuffle_down(result, 4u);
    result += simd_shuffle_down(result, 2u);
    result += simd_shuffle_down(result, 1u);
    return result;
}
)METAL";

const fast::CustomKernelFunction& single_down_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = cached_metal_kernel("tinfer_gguf_hc_down_block",
        {"input", "down", "params"}, {"low"}, R"METAL(
        const uint width = params[0], low_width = params[1];
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x * 16u + thread_index_in_threadgroup / 8u;
        if (row < low_width) {
            const float value = q8_hc_dot<8>(down, input, row, lane & 7u, width);
            if ((lane & 7u) == 0u) {
                const T projected = T(value);
                const T scaled = T(float(projected) / float(HC));
                const T tail = T(1) / (T(1) + metal::precise::exp(metal::abs(scaled)));
                const T sigmoid = scaled < T(0) ? tail : T(1) - tail;
                low[row] = T(float(scaled) * float(sigmoid));
            }
        }
        )METAL", kDot, true, false, options);
    return kernel;
}

const fast::CustomKernelFunction& down_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = fast::metal_kernel("tinfer_gguf_hc_down_gate",
        {"input", "down", "injection", "params"}, {"low", "gate"}, R"METAL(
        const uint width = params[0], low_width = params[1];
        const uint lane = thread_index_in_simdgroup;
        const uint row = threadgroup_position_in_grid.x * 4u + simdgroup_index_in_threadgroup;
        const uint token = threadgroup_position_in_grid.y;
        if (row < low_width) {
            const float value = q8_hc_dot<32>(down, input + token * width, row, lane, width);
            if (!lane) {
                const T projected = T(value);
                const T scaled = T(float(projected) / float(HC));
                const T sigmoid = T(1.0f / (1.0f + exp(-float(scaled))));
                low[token * low_width + row] = T(float(scaled) * float(sigmoid));
            }
        } else if (HAS_INJECT && row < low_width + HC) {
            const uint index = row - low_width;
            float sum = 0.0f;
            for (uint column = lane; column < width; column += 32u)
                sum += float(I(input[token * width + column])) * float(injection[index * width + column]);
            sum = simd_sum(sum);
            if (!lane) {
                const I projected = I(sum);
                const G scaled = G(float(projected) / float(HC));
                const G sigmoid = G(1.0f / (1.0f + exp(-float(scaled))));
                gate[token * HC + index] = G(2.0f * float(sigmoid));
            }
        }
        )METAL", kDot, true, false, options);
    return kernel;
}

const fast::CustomKernelFunction& mix_kernel() {
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    static const auto kernel = fast::metal_kernel("tinfer_gguf_hc_up_mix",
        {"low", "up", "normalized", "params"}, {"output"}, R"METAL(
        const uint low_width = params[1], hidden = params[0] / HC;
        const uint lane = thread_index_in_simdgroup;
        const uint local_row = lane / 4u;
        const uint row = threadgroup_position_in_grid.x * 8u + local_row;
        const uint stream = simdgroup_index_in_threadgroup;
        const uint token = threadgroup_position_in_grid.y;
        threadgroup float products[HC * 8];
        float product = 0.0f;
        if (row < hidden) {
            const uint column = stream * hidden + row;
            const float value = q8_hc_dot<4>(up, low + token * low_width, column, lane & 3u, low_width);
            if ((lane & 3u) == 0u) {
                const T projected = T(value);
                T gate;
                if constexpr (PRECISE) {
                    const T tail = T(1) / (T(1) + metal::precise::exp(metal::abs(projected)));
                    gate = projected < T(0) ? tail : T(1) - tail;
                } else gate = T(1.0f / (1.0f + exp(-float(projected))));
                product = float(T(float(gate) * float(normalized[token * params[0] + column])));
            }
        }
        if ((lane & 3u) == 0u) products[stream * 8u + local_row] = product;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint index = thread_index_in_threadgroup;
        if (index < 8u && threadgroup_position_in_grid.x * 8u + index < hidden) {
            float sum = 0.0f;
            for (uint i = 0; i < uint(HC); ++i) sum += products[i * 8u + index];
            output[token * hidden + threadgroup_position_in_grid.x * 8u + index] = T(sum / float(HC));
        }
        )METAL", kDot, true, false, options);
    return kernel;
}
}

std::optional<std::vector<array>> MlxGgufWeight::gated_residual(const array& normalized,
    const MlxGgufWeight& up, const array* injection, int streams) const {
    const auto dtype = normalized.dtype();
    if (rotation_ || up.rotation_) return std::nullopt;
    if (format_ != 8 || up.format_ != 8 || indirect_ || up.indirect_ || shape_.size() != 2 || up.shape_.size() != 2 ||
        normalized.ndim() < 2 || (dtype != float16 && dtype != bfloat16 && dtype != float32) ||
        streams < 2 || streams > 8 || input_size() % streams || normalized.shape(-1) != input_size() ||
        up.input_size() != output_size() || up.output_size() != input_size() || output_size() > 1024 ||
        (injection && (injection->shape() != Shape{streams, input_size()} ||
            (injection->dtype() != float16 && injection->dtype() != bfloat16 && injection->dtype() != float32))))
        return std::nullopt;
    const int rows = int(normalized.size() / input_size());
    if (rows < 1 || rows > 8) return std::nullopt;
    const auto source = reshape(normalized, {rows, input_size()});
    const auto inject = injection ? contiguous(*injection) : zeros({1}, dtype);
    const auto gate_dtype = promote_types(dtype, inject.dtype());
    const array params({uint32_t(input_size()), uint32_t(output_size())});
    const bool independent_injection = rows == 1 && dtype == float32;
    auto first = independent_injection
        ? single_down_kernel()({source, bytes_, params}, {{rows, output_size()}}, {dtype},
            {((output_size() + 15) / 16) * 128, 1, 1}, {128, 1, 1},
            {{"T", dtype}, {"HC", streams}}, std::nullopt, false, {})
        : down_kernel()({source, bytes_, inject, params}, {{rows, output_size()}, {rows, streams}},
        {dtype, gate_dtype}, {((output_size() + (injection ? streams : 0) + 3) / 4) * 128, rows, 1}, {128, 1, 1},
        {{"T", dtype}, {"I", inject.dtype()}, {"G", gate_dtype}, {"HC", streams}, {"HAS_INJECT", int(injection != nullptr)}},
        std::nullopt, false, {});
    auto shape = normalized.shape(); shape.back() /= streams;
    const int threads = 32 * streams;
    auto mixed = mix_kernel()({first[0], up.bytes_, source, params}, {shape}, {dtype},
        {((input_size() / streams + 7) / 8) * threads, rows, 1}, {threads, 1, 1},
        {{"T", dtype}, {"HC", streams}, {"PRECISE", int(independent_injection)}}, std::nullopt, false, {}).front();
    std::vector<array> result{std::move(mixed)};
    if (injection) {
        auto gate_shape = normalized.shape(); gate_shape.back() = streams;
        if (independent_injection) {
            auto projected = MlxLinear(*injection)(normalized);
            result.push_back(array(2.f, dtype) * sigmoid(projected / array(float(streams), dtype)));
        } else result.push_back(reshape(first[1], gate_shape));
    }
    return result;
}
} // namespace mfq::metal
