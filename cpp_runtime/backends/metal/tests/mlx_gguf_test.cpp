#include "mlx_gguf.h"
#include "mlx_gguf_rows.h"
#include "mlx_gguf_prefill.h"
#include "mlx_gguf_staged.h"
#include "mlx_platform.h"
#include "native/mlx_native_gguf.h"
#include "mlx_kernel_prepare.h"
#include "metal_device_info.h"
#include "mlx_tensor.h"
#include "mlx_affine.h"
#include "mlx_moe_ops.h"
#include "mfq_container.h"
#include "qwen4_ops.h"
#include "ggml.h"
#include "ggml-quants.h"
#include "gguf.h"
#include <mlx/primitives.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <sys/mman.h>
#include <unordered_set>

namespace {
using namespace mlx::core;
using namespace mfq::metal;

void compare(const array& actual, const array& expected, float tolerance, const std::string& label) {
    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    const auto scale = max(abs(astype(expected, float32))).item<float>();
    if (!std::isfinite(error) || error > tolerance * std::max(1.f, scale))
        throw std::runtime_error(label + " error=" + std::to_string(error) + " scale=" + std::to_string(scale));
}

void check_native_views(const std::vector<array>& outputs) {
    std::vector<std::pair<array, array>> views;
    std::unordered_set<const Primitive*> visited;
    const auto visit = [&](const auto& self, const array& value) -> void {
        if (!value.has_primitive() || !visited.insert(value.primitive_ptr().get()).second) return;
        if (std::string_view(value.primitive().name()) == "MlxRowMajorView")
            views.emplace_back(value.inputs().front(), value);
        for (const auto& input : value.inputs()) self(self, input);
    };
    for (const auto& output : outputs) visit(visit, output);
    if (views.empty()) throw std::runtime_error("native input normalization was not exercised");
    eval(outputs);
    for (const auto& [source, view] : views) {
        if (!view.flags().row_contiguous) throw std::runtime_error("native input normalization kept strided rows");
        if (source.flags().row_contiguous &&
            (source.buffer().ptr() != view.buffer().ptr() || source.offset() != view.offset()))
            throw std::runtime_error("native input normalization copied a contiguous view");
    }
}

array reference_mm(const array& input, const array& weight) {
    const auto x = contiguous(astype(input, float32));
    const auto w = contiguous(astype(weight, float32));
    eval(x, w);
    const int m = x.shape(0), k = x.shape(1), n = w.shape(0);
    std::vector<float> values(m * n);
    for (int i = 0; i < m; ++i) for (int j = 0; j < n; ++j) {
        double sum = 0;
        for (int c = 0; c < k; ++c) sum += double(x.data<float>()[i * k + c]) * w.data<float>()[j * k + c];
        values[i * n + j] = float(sum);
    }
    return array(values.begin(), Shape{m, n});
}

std::vector<std::uint8_t> packed(ggml_type type, int k, int rows) {
    std::vector<std::uint8_t> bytes(ggml_row_size(type, k) * rows);
    std::mt19937 rng(711 + int(type));
    for (auto& byte : bytes) byte = std::uint8_t(rng());
    const auto block = ggml_type_size(type);
    for (std::size_t offset = 0; offset < bytes.size(); offset += block) {
        const auto scale = ggml_fp32_to_fp16(type == GGML_TYPE_PTQ1_0 ? .03125f : .00025f);
        std::size_t d = type == GGML_TYPE_PTQ1_0 ? 26 : type == GGML_TYPE_Q2_K ? 80 : type == GGML_TYPE_Q3_K ? 108 : type == GGML_TYPE_Q6_K ? 208 : 0;
        if (type == GGML_TYPE_MXFP4) bytes[offset] = uint8_t(114 + offset / block % 5);
        else if (type == GGML_TYPE_Q8_K) {
            float value = .002f;
            std::memcpy(bytes.data() + offset, &value, 4);
        } else std::memcpy(bytes.data() + offset + d, &scale, 2);
        if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
            std::memcpy(bytes.data() + offset + 2, &scale, 2);
        if (type == GGML_TYPE_Q2_K) std::memcpy(bytes.data() + offset + 82, &scale, 2);
    }
    return bytes;
}

array reference_routed_mm(const array& input, const array& ids, const array& weight, bool shared) {
    const auto x = contiguous(astype(input, float32));
    const auto w = contiguous(astype(weight, float32));
    const auto routes = contiguous(astype(ids, int32));
    eval(x, w, routes);
    const int m = ids.shape(0), r = ids.shape(1), n = w.shape(1), k = w.shape(2);
    std::vector<float> output(m * r * n);
    for (int i = 0; i < m * r; ++i) {
        const auto e = routes.data<int32_t>()[i];
        if (e < 0 || e >= w.shape(0)) continue;
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            for (int c = 0; c < k; ++c)
                sum += double(x.data<float>()[(shared ? i / r : i) * k + c]) * w.data<float>()[(e * n + j) * k + c];
            output[i * n + j] = float(sum);
        }
    }
    return array(output.begin(), Shape{m, r, n});
}

void check_group_decoders() {
    const auto kernel = fast::metal_kernel("tinfer_gguf_group_decoder_test", {"packed", "params"}, {"y"}, R"METAL(
uint row = thread_position_in_grid.x, groups = params[0];
if (row >= params[1]) return;
GroupPayload payload;
for (uint g = 0u; g < groups; ++g) {
    fetch_raw_group<F>(payload, packed, row, groups, g,
        g == 0u || g / PlaneLayout<F>::meta_groups != (g - 1u) / PlaneLayout<F>::meta_groups);
    decode_group_chunks<F, half>(payload, GgufConstantLookup<F>{}, g,
        [&](uint c, half4 low, half4 high) {
            for (uint i = 0u; i < 4u; ++i) {
                y[ulong(row) * groups * 32u + g * 32u + c * 4u + i] = float(low[i]);
                y[ulong(row) * groups * 32u + g * 32u + 16u + c * 4u + i] = float(high[i]);
            }
        });
}
)METAL", MlxGgufStagedPrefill::scalar_decoder_header());
    for (const auto type : {GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K,
        GGML_TYPE_TQ2_T, GGML_TYPE_TQK4, GGML_TYPE_TQK5, GGML_TYPE_TQK6, GGML_TYPE_TQK7,
        GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7}) {
        constexpr int K = 512, N = 35;
        auto bytes = packed(type, K, N);
        const bool gyro = int(type) >= 144 && int(type) <= 151;
        if (gyro) {
            constexpr float scales[]{0.f, 0x1p-24f, -0x1p-24f, 0x1p-14f, -0x1p-14f,
                .00025f, -.00025f, .03125f, -.03125f, .5f, 1.f, 32.f, 512.f};
            const auto stride = ggml_type_size(type);
            for (std::size_t at = 0; at < bytes.size(); at += stride) {
                const auto scale = ggml_fp32_to_fp16(scales[(at / stride) % std::size(scales)]);
                std::memcpy(bytes.data() + at, &scale, sizeof(scale));
            }
        }
        std::vector<float> values(K * N);
        ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), values.size());
        const auto actual = kernel({array(bytes.begin(), Shape{int(bytes.size())}, uint8),
            array({uint32_t(K / 32), uint32_t(N)})}, {{N, K}}, {float32}, {64, 1, 1}, {64, 1, 1},
            {{"F", int(type)}}, std::nullopt, false, {}).front();
        compare(actual, astype(array(values.begin(), Shape{N, K}), float16), gyro ? 0.f : 1e-6f,
            std::string(ggml_type_name(type)) + " grouped decoder");
    }
}

void check_prefill_restore() {
    constexpr int E = 3, M = 9, R = 4, K = 128;
    if (!gguf_expert_prefill_available(zeros({M, K}, float16))) return;
    const auto ids = reshape(remainder(arange(M * R, int32), array(E + 2, int32)) - 1, {M, R});
    const auto routes = gguf_group_routes(ids, E, 32);
    const auto experts = reshape(broadcast_to(slice(routes.descriptors,
        {0, 0}, {routes.tiles, 1}), {routes.tiles, 32}), {-1, 1});
    for (const auto dtype : {float16, bfloat16, float32}) {
        const MlxGgufPrefillPlan plan(zeros({M, K}, dtype), ids, E, 32);
        for (const int width : {7, 128, 129}) for (const auto source_type : {float16, float32}) {
            const int rows = routes.tiles * 32, extent = rows * width;
            const auto values_by_expert = sin(astype(experts, float32) * .3f + arange(width, float32) * .013f);
            const auto backing = concatenate({zeros({1}, source_type), reshape(astype(values_by_expert, source_type), {-1})});
            eval(backing);
            const auto values = reshape(slice(backing, {1}, {extent + 1}), {rows, width});
            const auto expected_values = astype(sin(expand_dims(astype(ids, float32), -1) * .3f +
                arange(width, float32) * .013f), source_type);
            const auto expected = astype(where(expand_dims((ids >= array(0, int32)) & (ids < array(E, int32)), -1),
                expected_values, zeros({M, R, width}, source_type)), dtype);
            compare(plan.restore(values), expected, 0.f, "GGUF prefill restore width/dtype/offset/invalid routes width=" +
                std::to_string(width) + " source=" + std::to_string(int(source_type.val())) +
                " destination=" + std::to_string(int(dtype.val())));
        }
    }
}

void check_sparse_short_routes() {
    constexpr int E = 64, N = 16, K = 256, R = 4;
    if (!gguf_short_sparse_routes(zeros({8, R}, int32), E) ||
        gguf_short_sparse_routes(zeros({9, R}, int32), E) ||
        gguf_short_sparse_routes(zeros({8, R}, int32), R) ||
        gguf_short_sparse_routes(zeros({0, R}, int32), E))
        throw std::runtime_error("short sparse GGUF route boundary changed");
    for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS, GGML_TYPE_TQ2_T, GGML_TYPE_TQK4,
        GGML_TYPE_TQK5, GGML_TYPE_TQK6, GGML_TYPE_TQK7, GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7}) {
        const auto bytes = packed(type, K, E * N);
        auto format = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : format) c = std::toupper(static_cast<unsigned char>(c));
        const auto weight = MlxGgufWeight::from_bytes(format, bytes, {E, N, K});
        std::vector<float> decoded(E * N * K);
        ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data(), decoded.size());
        const array gold(decoded.begin(), Shape{E, N, K});
        for (int M : {7, 8}) for (auto dt : {float16, bfloat16, float32}) for (bool repeated : {false, true}) {
            std::vector<int> selected(M * R);
            for (int i = 0; i < M * R; ++i) selected[i] = repeated ? i % R : (i * 17) % E;
            selected[0] = -1; selected.back() = E;
            const array ids(selected.begin(), Shape{M, R}, int32);
            const auto input = astype(reshape(sin(arange(M * K, float32) * .017f), {M, K}), dt);
            const auto expected = astype(reference_routed_mm(input, ids, gold, true), dt);
            const auto projected = weight.routed_matmul(input, ids, true);
            compare(projected, expected, dt == bfloat16 ? .015f : .003f, "sparse short routed projection");
            compare(weight.routed_swiglu(weight, input, ids, nullptr, .7f),
                moe_limited_swiglu_pair(expected, expected, .7f), dt == bfloat16 ? .025f : .004f,
                "sparse short fused Gate/Up");
            const auto separate = broadcast_to(expand_dims(input, 1), {M, R, K});
            const auto probabilities = full({M, R}, .25f);
            compare(weight.routed_weighted_matmul(separate, ids, probabilities),
                moe_weighted_reduce(expected, probabilities), dt == bfloat16 ? .015f : .003f,
                "sparse short weighted Down");
        }
    }
}

void check_sparse_mixed_prefill() {
    constexpr int E = 64, N = 35, K = 256, R = 4;
    const std::array types{GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K};
    std::vector<int> experts;
    std::vector<std::string> formats;
    std::vector<array> storage;
    std::vector<float> decoded(E * N * K);
    for (int e = 0; e < E; ++e) {
        const auto type = types[e % types.size()];
        const auto bytes = packed(type, K, N);
        auto format = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : format) c = std::toupper(static_cast<unsigned char>(c));
        experts.push_back(e); formats.push_back(format);
        storage.emplace_back(bytes.begin(), Shape{int(bytes.size())}, uint8);
        ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data() + e * N * K, N * K);
    }
    const auto weight = MlxGgufWeight::from_mixed_experts({E, N, K}, experts, formats, storage);
    const array gold(decoded.begin(), Shape{E, N, K});
    for (int M : {9, 17}) for (auto dt : {float16, bfloat16, float32}) for (bool shared : {true, false}) {
        std::vector<int> selected(M * R);
        for (int i = 0; i < M * R; ++i) selected[i] = i % E;
        selected[0] = -1; selected.back() = E;
        const array ids(selected.begin(), Shape{M, R}, int32);
        const auto input = astype(reshape(sin(arange(M * (shared ? 1 : R) * K, float32) * .017f),
            shared ? Shape{M, K} : Shape{M, R, K}), dt);
        const auto expected = astype(reference_routed_mm(reshape(astype(input, float16), {-1, K}), ids,
            astype(gold, float16), shared), dt);
        compare(weight.routed_matmul(input, ids, shared), expected, dt == bfloat16 ? .015f : .003f,
            "sparse mixed prefill projection");
        if (shared) compare(weight.routed_swiglu(weight, input, ids, nullptr, .7f),
            moe_limited_swiglu_pair(expected, expected, .7f), dt == bfloat16 ? .025f : .004f,
            "sparse mixed prefill Gate/Up");
        else compare(weight.routed_weighted_matmul(input, ids, full({M, R}, .25f)),
            moe_weighted_reduce(expected, full({M, R}, .25f)), dt == bfloat16 ? .015f : .003f,
            "sparse mixed prefill weighted Down");
    }
}

void check_wide_mixed_prefill() {
    constexpr int E = 4, N = 512, K = 256, M = 65, R = 4;
    const std::array types{GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K};
    std::vector<int> experts;
    std::vector<std::string> formats;
    std::vector<array> storage;
    std::vector<float> decoded(E * N * K);
    for (int e = 0; e < E; ++e) {
        const auto bytes = packed(types[e], K, N);
        auto format = "GGUF_" + std::string(ggml_type_name(types[e]));
        for (auto& c : format) c = std::toupper(static_cast<unsigned char>(c));
        experts.push_back(e); formats.push_back(format);
        storage.emplace_back(bytes.begin(), Shape{int(bytes.size())}, uint8);
        ggml_get_type_traits(types[e])->to_float(bytes.data(), decoded.data() + e * N * K, N * K);
    }
    const auto weight = MlxGgufWeight::from_mixed_experts({E, N, K}, experts, formats, storage);
    const array gold(decoded.begin(), Shape{E, N, K});
    auto ids = reshape(remainder(arange(M * R, int32), array(E, int32)), {M, R});
    const auto plan = gguf_route_plan(ids, E, true);
    if (!mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5") && plan.tile_rows != 64)
        throw std::runtime_error("wide mixed prefill dispatch was not selected");
    for (auto dt : {float16, bfloat16, float32}) for (bool shared : {true, false}) {
        const auto input = astype(reshape(sin(arange(M * (shared ? 1 : R) * K, float32) * .017f),
            shared ? Shape{M, K} : Shape{M, R, K}), dt);
        const auto expected = astype(reference_routed_mm(reshape(astype(input, float16), {-1, K}), ids,
            astype(gold, float16), shared), dt);
        compare(weight.routed_matmul(input, ids, shared, &plan), expected, dt == bfloat16 ? .015f : .003f,
            "wide mixed prefill projection");
        if (shared) compare(weight.routed_swiglu(weight, input, ids, &plan, .7f),
            moe_limited_swiglu_pair(expected, expected, .7f), dt == bfloat16 ? .025f : .004f,
            "wide mixed prefill Gate/Up");
        else compare(weight.routed_weighted_matmul(input, ids, full({M, R}, .25f), &plan),
            moe_weighted_reduce(expected, full({M, R}, .25f)), dt == bfloat16 ? .015f : .003f,
            "wide mixed prefill weighted Down");
    }
}

void check(ggml_type type) {
    constexpr int K = 512, N = 35, E = 3;
    const auto bytes = packed(type, K, N * E);
    std::vector<float> decoded(K * N * E);
    if (type == GGML_TYPE_Q8_K)
        dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(bytes.data()), decoded.data(), decoded.size());
    else ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data(), decoded.size());
    const auto dtype = "GGUF_" + std::string(ggml_type_name(type));
    auto upper = dtype;
    for (auto& c : upper) c = std::toupper(static_cast<unsigned char>(c));
    if (gguf_expert_prefill_available(zeros({65, K}, float16))) {
        const auto matrix_bytes = packed(type, K, 64);
        const auto matrix = MlxGgufWeight::from_bytes(upper, matrix_bytes, {64, K});
        std::vector<float> matrix_values(64 * K);
        if (type == GGML_TYPE_Q8_K)
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(matrix_bytes.data()), matrix_values.data(), matrix_values.size());
        else ggml_get_type_traits(type)->to_float(matrix_bytes.data(), matrix_values.data(), matrix_values.size());
        compare(matrix.dequantize(float32), array(matrix_values.begin(), Shape{64, K}), 2e-6f, upper + " compressed planes");
        if (mlx_native_gguf_supported(type, 64, K)) {
            const auto native = mlx_native_gguf_pack(array(matrix_bytes.begin(), Shape{int(matrix_bytes.size())}, uint8),
                type, 64, K, int(ggml_row_size(type, K)));
            for (const int count : {1, 2, 3, 4, 8, 65}) {
                const auto input = astype(reshape(sin(arange(count * K, float32) * .013f), {count, K}), bfloat16);
                const auto expected = reference_mm(input, matrix.dequantize(float32));
                const auto actual = mlx_native_gguf_project(std::span(&native, 1), input)[0];
                const auto rms = sqrt(mean(square(astype(actual, float32) - expected))).item<float>();
                const auto scale = sqrt(mean(square(expected))).item<float>();
                if (!std::isfinite(rms) || rms > .02f * scale + 1e-6f)
                    throw std::runtime_error(upper + " native projection relative RMS=" + std::to_string(rms / scale));
            }
        }
        for (const int count : {1, 2, 6, 7, 8, 19, 64, 65, 128}) {
            const auto rows = reshape(sin(arange(count * K, float32) * .013f), {count, K});
            const bool direct = count <= 6 || (count <= 8 && matrix.has_staged_planes());
            compare(matrix.matmul(rows), reference_mm(direct ? rows : astype(rows, float16),
                matrix.dequantize(direct ? float32 : float16)),
                2e-4f, upper + " compressed dense prefill M=" + std::to_string(count));
            if ((count == 7 || count == 8) && matrix.has_staged_planes()) compare(MlxGgufStagedPrefill::dense_small(matrix, rows),
                reference_mm(astype(rows, float16), matrix.dequantize(float16)), 2e-4f,
                upper + " compact matrix tile M=" + std::to_string(count));
        }
        constexpr int intermediate = 256;
        auto gate_bytes = packed(type, K, intermediate * E);
        if (type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4)
            for (std::size_t b = 0; b < gate_bytes.size(); b += ggml_type_size(type)) {
                const auto scale = ggml_fp32_to_fp16(.015625f * (1 + b / ggml_type_size(type) % 7));
                std::memcpy(gate_bytes.data() + b, &scale, sizeof(scale));
            }
        const auto gate = MlxGgufWeight::from_bytes(upper, gate_bytes, {E, intermediate, K});
        const auto down = MlxGgufWeight::from_bytes(upper, packed(type, intermediate, K * E), {E, K, intermediate});
        if (type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4) {
            const auto& encoded = gate_bytes;
            std::vector<float> values(E * intermediate * K);
            ggml_get_type_traits(type)->to_float(encoded.data(), values.data(), values.size());
            if (!gate.has_staged_planes() || gate.packed_nbytes() != encoded.size() + 1)
                throw std::runtime_error(upper + " compressed planes changed the packed payload size");
            compare(gate.dequantize(float32), array(values.begin(), Shape{E, intermediate, K}), 2e-6f,
                upper + " grouped compressed plane CPU decode");
            const auto rounded_weight = astype(array(values.begin(), Shape{E, intermediate, K}), float16);
            for (int count : {1, 2, 6, 8, 65}) {
                const auto x = astype(reshape(sin(arange(count * K, float32) * .017f), {count, K}), float16);
                const auto selected = reshape(remainder(arange(count * E, int32), array(E, int32)), {count, E});
                if (count == 65 && gguf_prefill_tile_rows(gate, gate, down, selected) != 48)
                    throw std::runtime_error(upper + " populated Gyro routes did not select the wider tile");
                compare(gate.routed_matmul(x, selected, true),
                    reference_routed_mm(x, selected, array(values.begin(), Shape{E, intermediate, K}), true),
                    .003f, upper + " compressed expert projection M=" + std::to_string(count));
                const auto probability = full({count, E}, 1.f / E);
                compare(gate.routed_weighted_matmul(broadcast_to(expand_dims(x, 1), {count, E, K}), selected, probability),
                    moe_weighted_reduce(reference_routed_mm(x, selected,
                        array(values.begin(), Shape{E, intermediate, K}), true), probability),
                    .003f, upper + " compressed expert weighted projection M=" + std::to_string(count));
                if (count == 1) {
                    const MlxGgufPrefillPlan plan(x, selected, E, 8);
                    const auto grouped = plan.gate_up(gate, gate), restored = plan.restore(grouped);
                    eval(grouped, restored);
                    if (grouped.buffer().ptr() != restored.buffer().ptr() || grouped.offset() != restored.offset())
                        throw std::runtime_error(upper + " single-token restore copied identity routes");
                    const auto reduced = plan.down(down, grouped, probability);
                    if (MlxGgufStagedPrefill::prefers_weighted_down(down) && (!reduced.has_primitive() ||
                        mlx_kernel_preparation_key(reduced.primitive()).find("tinfer_gguf_planes_down_reduce") == std::string::npos))
                        throw std::runtime_error(upper + " single-token FFN skipped the fused Down kernel");
                }
                if (count == 2 || count == 6 || count == 65) for (const auto dt : {float16, bfloat16, float32}) {
                    const auto typed_input = astype(x, dt);
                    const auto linear = astype(reference_routed_mm(astype(typed_input, float16), selected, rounded_weight, true), dt);
                    const auto expected = moe_swiglu_pair(linear, linear);
                    const auto actual = gate.routed_swiglu(gate, typed_input, selected);
                    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
                    const auto peak = max(abs(astype(expected, float32))).item<float>();
                    if (!std::isfinite(error) || error > (dt == bfloat16 ? .02f : .002f) * peak + 1e-6f)
                        throw std::runtime_error(upper + " compressed Gate/Up CPU oracle M=" + std::to_string(count) +
                            " error=" + std::to_string(error) + " peak=" + std::to_string(peak));
                }
            }
            for (const auto dt : {float16, bfloat16, float32}) for (int routes : {1, 2, 3, 10, 17}) {
                const auto x = astype(reshape(sin(arange(routes * K, float32) * .017f), {1, routes, K}), dt);
                const auto ids = reshape(remainder(arange(routes, int32) + 1, array(E + 2, int32)) - 1, {1, routes});
                const auto probability = reshape((arange(routes, float32) + 1.f) / float(routes * routes), {1, routes});
                const auto projected = astype(reference_routed_mm(astype(x, float16), ids, rounded_weight, false), dt);
                const auto expected = moe_weighted_reduce(projected, probability);
                const MlxGgufPrefillPlan plan(x, ids, E, 8, false);
                for (const auto actual : {gate.routed_weighted_matmul(x, ids, probability),
                        plan.down(gate, reshape(x, {routes, K}), probability)}) {
                    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
                    const auto peak = max(abs(astype(expected, float32))).item<float>();
                    if (actual.dtype() != dt || !std::isfinite(error) || error > (dt == bfloat16 ? .02f : .002f) * peak + 1e-6f)
                        throw std::runtime_error(upper + " fused weighted CPU oracle failed");
                }
            }
        }
        auto ids = reshape(remainder(arange(65 * 3, int32), array(5, int32)) - 1, {65, 3});
        const auto probabilities = full({65, 3}, .25f, float32);
        for (const auto dt : {float16, bfloat16, float32}) {
            if (dt != float16 && type != GGML_TYPE_IQ3_S && type != GGML_TYPE_Q4_K) continue;
            auto input = astype(reshape(sin(arange(65 * K, float32) * .017f), {65, K}), dt);
            const MlxGgufPrefillPlan plan(input, ids, E, gguf_prefill_tile_rows(gate, gate, down, ids));
            const auto expected = moe_weighted_reduce(down.routed_matmul(gate.routed_swiglu(gate, input, ids), ids, false), probabilities);
            compare(plan.down(down, plan.gate_up(gate, gate), probabilities), expected,
                dt == bfloat16 ? .02f : .003f, upper + " complete register prefill FFN");
            if (gguf_prefill_tile_rows(gate, gate, down, ids) == 32) {
                const MlxGgufPrefillPlan staged(input, ids, E, 32);
                compare(staged.down(down, staged.gate_up(gate, gate), probabilities), expected,
                    dt == bfloat16 ? .02f : .003f, upper + " packed staged prefill FFN");
            }
        }
    }
    const auto experts = MlxGgufWeight::from_bytes(upper, bytes, Shape{E, N, K});
    const array gold(decoded.begin(), Shape{E, N, K});
    for (const int m : {1, 2, 6, 7, 19, 65}) {
        const auto source = reshape(cos(arange(m * K, float32) * .013f), {m, K});
        std::vector<int> ids(m * 3);
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = int(i % 5) - 1;
        const array routes(ids.begin(), {m, 3}, int32);
        for (const auto dtype : {float16, bfloat16, float32}) {
            const auto input = astype(source, dtype);
            compare(experts.routed_swiglu(experts, input, routes),
                moe_swiglu_pair(experts.routed_matmul(input, routes, true),
                    experts.routed_matmul(input, routes, true)), dtype == bfloat16 ? .01f : .002f,
                    upper + " fused Gate/Up M=" + std::to_string(m) + " dtype=" +
                        (dtype == bfloat16 ? "bf16" : dtype == float16 ? "f16" : "f32"));
        }
    }
    compare(experts.dequantize(float32), gold, 2e-6f, upper + " decode");
    compare(experts.dequantize(float16), astype(gold, float16), 0.f, upper + " half decode");
    compare(experts.dequantize(bfloat16), astype(gold, bfloat16), 0.f, upper + " bfloat decode");
    MlxMappedGgufRows rows(bytes, upper, E * N, K);
    MlxGgufRowBatch selected;
    for (const auto row : {E * N - 1, 0, 3, 3}) rows.append_row(row, selected);
    compare(selected.decode(), astype(take(reshape(gold, {E * N, K}),
        array({E * N - 1, 0, 3, 3}, int32), 0), float16), 0.f, upper + " selected rows");
    if (selected.source_bytes_read() != 4 * ggml_row_size(type, K))
        throw std::runtime_error("GGUF selected row reads expanded the backing table");
    MlxGgufRowBatch merged;
    merged.append_batch(MlxGgufRowBatch{});
    merged.append_batch(selected); merged.append_batch(selected);
    compare(merged.decode(), concatenate({selected.decode(), selected.decode()}, 0), 0.f, upper + " selected row batch order");
    if (merged.source_bytes_read() != 8 * ggml_row_size(type, K))
        throw std::runtime_error("GGUF merged row payloads differ");
    bool bad_row = false;
    try { rows.append_row(E * N, selected); } catch (const std::out_of_range&) { bad_row = true; }
    if (!bad_row) throw std::runtime_error("GGUF selected row bounds were ignored");
    const auto weight = MlxGgufWeight::from_bytes(upper,
        std::span(bytes.data(), std::size_t(ggml_row_size(type, K)) * N), Shape{N, K});
    const auto dense = reshape(slice(gold, {0, 0, 0}, {1, N, K}), {N, K});
    MlxLinear linear(weight);
    MlxEmbedding embedding(weight);
    MlxProjectionBatch gate_up(std::vector<const MlxLinear*>{&linear, &linear});
    for (const int m : {1, 2, 3, 4, 5, 6, 7, 8, 65}) for (const auto dt : {float16, bfloat16, float32}) {
        const auto x = astype(reshape(sin(arange(m * 5 * K, float32) * .031f), {m, 5, K}), dt);
        std::vector<array> expected;
        for (int g = 0; g < 5; ++g) {
            auto source = reshape(slice(x, {0, g, 0}, {m, g + 1, K}), {m, K});
            auto selected = slice(dense, {g * 7, 0}, {(g + 1) * 7, K});
            if (m > 6) { source = astype(source, float16); selected = astype(selected, float16); }
            expected.push_back(astype(reference_mm(source, selected), dt));
        }
        compare(linear.grouped_row_matmul(x, 5), stack(expected, 1), dt == bfloat16 ? .01f : .002f,
            upper + " diagonal grouped O projection M=" + std::to_string(m));
    }
    if (linear(zeros({0, K})).size() || embedding(zeros({0}, int32)).size())
        throw std::runtime_error("empty GGUF projection changed shape");
    const array ids({N - 1, 0, -1, N, 3}, int32);
    auto expected_rows = concatenate({take(dense, array({N - 1, 0}, int32), 0), zeros({2, K}), take(dense, array({3}, int32), 0)}, 0);
    compare(embedding(ids, float32), expected_rows, 2e-6f, upper + " embedding");
    for (const int M : {1, 2, 6, 7, 19, 65}) {
        auto x = reshape(sin(arange(M * K, float32) * .073f), Shape{M, K});
        const auto result = linear(x);
        auto expected = reference_mm(x, dense);
        if (M > 6) expected = reference_mm(astype(x, float16), astype(dense, float16));
        compare(result, expected, 2e-4f, upper + " M=" + std::to_string(M));
        compare(embedding.project(x), result, 0.f, upper + " tied head");
        compare(gate_up.swiglu(x), moe_swiglu_pair(result, result), 2e-4f, upper + " dense fused Gate/Up");
        compare(gate_up.swiglu(x, .25f), moe_limited_swiglu_pair(result, result, .25f), 0.f, upper + " clipped dense fallback");
        if (M == 65) for (const auto dtype : {float16, bfloat16, float32}) {
            const auto input = astype(x, dtype);
            const auto actual = linear(input);
            if (actual.dtype() != dtype) throw std::runtime_error("GGUF dense prefill output dtype changed");
            compare(actual, astype(reference_mm(astype(input, float16), astype(dense, float16)), dtype),
                dtype == bfloat16 ? .01f : .002f, upper + " dense prefill decode-once dispatch");
        }
    }
    const array routes({2, 0, 2, 1, 0, -1}, Shape{2, 3}, int32);
    auto x = reshape(cos(arange(2 * K, float32) * .03f), {2, K});
    const auto out = experts.routed_matmul(x, routes, true);
    std::vector<array> expected;
    const int chosen[]{2, 0, 2, 1, 0, -1};
    for (int r = 0; r < 6; ++r)
        expected.push_back(chosen[r] < 0 ? zeros({1, N}) : reference_mm(take(x, array({r / 3}, int32), 0), take(gold, chosen[r], 0)));
    compare(out, reshape(concatenate(expected, 0), {2, 3, N}), 2e-4f, upper + " routed");
    const auto streamed = [&] {
        std::vector<array> allocations;
        const auto stride = ggml_row_size(type, K);
        for (const int expert : {2, 0}) allocations.emplace_back(bytes.begin() + expert * N * stride,
            Shape{N, int(stride)}, uint8);
        return MlxGgufWeight::from_experts(upper, {E, N, K}, {2, 0}, allocations);
    }();
    expected[3] = zeros({1, N});
    compare(streamed.routed_matmul(x, routes, true), reshape(concatenate(expected, 0), {2, 3, N}),
        2e-4f, upper + " indirect resident experts");
    compare(streamed.routed_swiglu(experts, x, routes),
        moe_swiglu_pair(streamed.routed_matmul(x, routes, true), experts.routed_matmul(x, routes, true)),
        .002f, upper + " mixed direct/indirect Gate/Up");
    compare(streamed.routed_swiglu(experts, x, routes, nullptr, .7f),
        moe_limited_swiglu_pair(streamed.routed_matmul(x, routes, true), experts.routed_matmul(x, routes, true), .7f),
        .002f, upper + " clipped direct/indirect Gate/Up");
    auto separate = reshape(cos(arange(6 * K, float32) * .019f), {2, 3, K});
    expected.clear();
    const auto flat = reshape(separate, {6, K});
    for (int r = 0; r < 6; ++r)
        expected.push_back(chosen[r] < 0 ? zeros({1, N}) : reference_mm(take(flat, array({r}, int32), 0), take(gold, chosen[r], 0)));
    compare(experts.routed_matmul(separate, routes, false), reshape(concatenate(expected, 0), {2, 3, N}), 2e-4f, upper + " down");
    for (const int m : {1, 2, 3, 4, 5, 6, 7, 8}) for (const auto dt : {float16, bfloat16, float32}) {
        const auto hidden = astype(reshape(cos(arange(m * 3 * K, float32) * .019f), {m, 3, K}), dt);
        const auto ids = reshape(remainder(arange(m * 3, int32), array(5, int32)) - 1, {m, 3});
        const auto p = reshape(sin(arange(m * 3, float32) * .1f), {m, 3});
        compare(experts.routed_weighted_matmul(hidden, ids, p), moe_weighted_reduce(experts.routed_matmul(hidden, ids, false), p),
            dt == bfloat16 ? .01f : .002f, upper + " fused Down/reduce M=" + std::to_string(m));
    }
    for (const int M : {7, 19, 65}) {
        std::vector<int> chosen(M * 3);
        for (int i = 0; i < M * 3; ++i) chosen[i] = i % 5 - 1;
        const array route_ids(chosen.begin(), {M, 3}, int32);
        const auto input = reshape(sin(arange(M * K, float32) * .017f), {M, K});
        const auto plan = gguf_route_plan(route_ids, E);
        for (const bool shared : {true, false}) {
            const auto source = shared ? input : reshape(sin(arange(M * 3 * K, float32) * .017f), {M * 3, K});
            std::vector<array> reference, partial;
            for (int i = 0; i < M * 3; ++i) {
                const int expert = chosen[i];
                auto value = expert < 0 || expert >= E ? zeros({1, N}) : reference_mm(
                    astype(take(source, array({shared ? i / 3 : i}, int32), 0), float16),
                    astype(take(gold, expert, 0), float16));
                reference.push_back(value);
                partial.push_back(expert == 1 ? zeros({1, N}) : value);
            }
            compare(experts.routed_matmul(source, route_ids, shared, &plan),
                reshape(concatenate(reference, 0), {M, 3, N}), 2e-4f, upper + " routed prefill");
            compare(streamed.routed_matmul(source, route_ids, shared, &plan),
                reshape(concatenate(partial, 0), {M, 3, N}), 2e-4f, upper + " indirect routed prefill");
        }
    }
    if (type == GGML_TYPE_Q4_K) for (auto dt : {float16, bfloat16}) {
        compare(linear(astype(x, dt)), matmul(astype(x, float32), transpose(dense)), .02f, upper + " output dtype");
        if (linear(astype(x, dt)).dtype() != dt) throw std::runtime_error("GGUF output dtype changed");
    }
    auto strided = transpose(reshape(sin(arange(3 * K, float32) * .012f), {K, 3}));
    compare(linear(strided), reference_mm(strided, dense), 2e-4f, upper + " strided input");
    std::cout << upper << " decode/embedding/M=1,2,6/prefill/routed passed\n";
}
void check_heterogeneous_file() {
    char pattern[] = "/tmp/tinfer-gguf-mixed.XXXXXX";
    const auto* directory = mkdtemp(pattern);
    if (!directory) throw std::runtime_error("cannot create GGUF mixed fixture directory");
    struct Cleanup {
        std::filesystem::path file, directory;
        ~Cleanup() { std::error_code error; std::filesystem::remove(file, error); std::filesystem::remove(directory, error); }
    } cleanup{std::filesystem::path(directory) / "mixed.gguf", directory};
    auto* context = ggml_init({4 << 20, nullptr, false});
    auto* file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "test");
    constexpr int E = 4, N = 64, K = 512;
    const std::array<ggml_type, E> types{GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K};
    std::array<std::vector<array>, 3> gold;
    for (int p = 0; p < 3; ++p) {
        const auto role = std::array{"gate", "up", "down"}[p];
        for (const auto type : {GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K}) {
            const auto count = std::count(types.begin(), types.end(), type);
            auto* tensor = ggml_new_tensor_3d(context, type, K, N, count);
            auto suffix = std::string(ggml_type_name(type));
            for (auto& c : suffix) c = std::tolower(static_cast<unsigned char>(c));
            const auto name = std::string("blk.0.ffn_") + role + "_exps." + suffix + ".weight";
            ggml_set_name(tensor, name.c_str());
            const auto bytes = packed(type, K, N * count);
            std::memcpy(tensor->data, bytes.data(), bytes.size());
            gguf_add_tensor(file, tensor);
            std::vector<float> decoded(N * K * count);
            ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data(), decoded.size());
            int slot = 0;
            if (gold[p].empty()) gold[p].assign(E, zeros({N, K}));
            for (int e = 0; e < E; ++e) if (types[e] == type) {
                gold[p][e] = array(decoded.begin() + slot * N * K, Shape{N, K}); ++slot;
            }
        }
    }
    for (const auto* role : {"gu", "down"}) {
        auto* map = ggml_new_tensor_1d(context, GGML_TYPE_I32, E);
        const auto name = std::string("blk.0.ffn_exps_") + role + "_type";
        ggml_set_name(map, name.c_str());
        for (int e = 0; e < E; ++e) static_cast<int32_t*>(map->data)[e] = types[e];
        gguf_add_tensor(file, map);
    }
    const auto written = gguf_write_to_file(file, cleanup.file.string().c_str(), false);
    gguf_free(file); ggml_free(context);
    if (!written) throw std::runtime_error("cannot write GGUF mixed fixture");
    const MfqContainer model(cleanup.file);
    const auto gate = MlxGgufWeight::load(model, "model.block.0.mlp.experts.gate.weight");
    const auto up = MlxGgufWeight::load(model, "model.block.0.mlp.experts.up.weight");
    const auto down = MlxGgufWeight::load(model, "model.block.0.mlp.experts.down.weight");
    for (const int m : {1, 8, 65}) {
        const auto input = astype(reshape(sin(arange(m * K, float32) * .011f), {m, K}), float16);
        const auto ids = broadcast_to(arange(E, int32), {m, E});
        std::array<std::vector<array>, 3> expected;
        for (int p = 0; p < 3; ++p) for (int e = 0; e < E; ++e)
            expected[p].push_back(astype(reference_mm(input, gold[p][e]), float16));
        compare(gate.routed_matmul(input, ids, true), stack(expected[0], 1), .002f, "mixed GGUF file projection");
        compare(gate.routed_swiglu(up, input, ids), moe_swiglu_pair(stack(expected[0], 1), stack(expected[1], 1)),
            .003f, "mixed GGUF file fused Gate/Up");
        const auto probabilities = full({m, E}, .25f);
        compare(down.routed_weighted_matmul(broadcast_to(expand_dims(input, 1), {m, E, K}), ids, probabilities),
            moe_weighted_reduce(stack(expected[2], 1), probabilities), .003f, "mixed GGUF file fused Down");
    }
}

void check_heterogeneous_experts() {
    check_heterogeneous_file();
    constexpr int E = 5, N = 64, K = 512;
    const std::array types{GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q3_K, GGML_TYPE_MXFP4};
    std::array<std::vector<array>, 2> gold;
    const auto make = [&](int offset) {
        std::vector<std::string> dtypes;
        std::vector<array> storage;
        for (int e = 0; e < E - 1; ++e) {
            const auto type = types[(e + offset) % types.size()];
            auto dtype = "GGUF_" + std::string(ggml_type_name(type));
            for (auto& c : dtype) c = std::toupper(static_cast<unsigned char>(c));
            auto bytes = packed(type, K, N);
            std::vector<float> values(N * K);
            ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), values.size());
            gold[offset].emplace_back(values.begin(), Shape{N, K});
            dtypes.push_back(std::move(dtype));
            storage.emplace_back(bytes.begin(), Shape{int(bytes.size())}, uint8);
        }
        gold[offset].push_back(zeros({N, K}));
        return MlxGgufWeight::from_mixed_experts({E, N, K}, {0, 1, 2, 3}, dtypes, storage);
    };
    const auto gate = make(0), up = make(1);
    std::vector<MlxGgufWeight> uniform;
    for (const auto type : {GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4}) {
        auto dtype = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : dtype) c = std::toupper(static_cast<unsigned char>(c));
        const auto bytes = packed(type, K, N * E);
        uniform.push_back(MlxGgufWeight::from_bytes(dtype, bytes, {E, N, K}));
        std::vector<array> allocations;
        const auto stride = ggml_row_size(type, K);
        for (int expert : {0, 2, 4}) allocations.emplace_back(bytes.begin() + expert * N * stride,
            Shape{N, int(stride)}, uint8);
        uniform.push_back(MlxGgufWeight::from_experts(dtype, {E, N, K}, {0, 2, 4}, allocations));
    }
    if (!gate.has_mixed_formats() || gate.has_staged_planes())
        throw std::runtime_error("heterogeneous GGUF bank lost its compressed expert layout");
    for (const int m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 33, 65}) {
        std::vector<int> selected(m * 5);
        for (std::size_t i = 0; i < selected.size(); ++i) selected[i] = int(i % 7) - 1;
        const array ids(selected.begin(), {m, 5}, int32);
        for (const auto dt : {float16, bfloat16, float32}) {
            const auto input = astype(reshape(sin(arange(m * K, float32) * .017f), {m, K}), dt);
            const auto values = astype(input, float32);
            eval(values);
            std::array<std::vector<float>, 2> outputs{std::vector<float>(m * 5 * N), std::vector<float>(m * 5 * N)};
            for (int part = 0; part < 2; ++part) for (int e = 0; e < E - 1; ++e) {
                const auto w = astype(gold[part][e], float32); eval(w);
                for (int r = 0; r < m * 5; ++r) if (selected[r] == e)
                    for (int n = 0; n < N; ++n) {
                        double sum = 0;
                        for (int k = 0; k < K; ++k)
                            sum += double(values.data<float>()[(r / 5) * K + k]) * w.data<float>()[n * K + k];
                        outputs[part][r * N + n] = float(sum);
                    }
            }
            const array expected_gate(outputs[0].begin(), Shape{m, 5, N});
            const array expected_up(outputs[1].begin(), Shape{m, 5, N});
            compare(gate.routed_matmul(input, ids, true), astype(expected_gate, dt), dt == bfloat16 ? .01f : .002f,
                "heterogeneous GGUF routed projection M=" + std::to_string(m));
            compare(gate.routed_swiglu(up, input, ids), moe_swiglu_pair(astype(expected_gate, dt), astype(expected_up, dt)),
                dt == bfloat16 ? .02f : .003f, "heterogeneous GGUF fused Gate/Up M=" + std::to_string(m));
            compare(gate.routed_swiglu(up, input, ids, nullptr, .7f),
                moe_limited_swiglu_pair(astype(expected_gate, dt), astype(expected_up, dt), .7f),
                dt == bfloat16 ? .02f : .003f, "heterogeneous GGUF clipped Gate/Up M=" + std::to_string(m));
            for (const auto& fixed : uniform) {
                const auto projected = fixed.routed_matmul(input, ids, true);
                compare(gate.routed_swiglu(fixed, input, ids),
                    moe_swiglu_pair(gate.routed_matmul(input, ids, true), projected),
                    dt == bfloat16 ? .02f : .003f, "heterogeneous Gate / uniform Up M=" + std::to_string(m));
                compare(fixed.routed_swiglu(up, input, ids, nullptr, .7f),
                    moe_limited_swiglu_pair(projected, up.routed_matmul(input, ids, true), .7f),
                    dt == bfloat16 ? .02f : .003f, "uniform Gate / heterogeneous Up M=" + std::to_string(m));
            }
            const auto separate = broadcast_to(expand_dims(input, 1), {m, 5, K});
            compare(gate.routed_matmul(separate, ids, false), astype(expected_gate, dt), dt == bfloat16 ? .01f : .002f,
                "heterogeneous GGUF separate routed inputs M=" + std::to_string(m));
            const auto probabilities = reshape(cos(arange(m * 5, float32) * .1f), {m, 5});
            compare(gate.routed_weighted_matmul(separate, ids, probabilities),
                moe_weighted_reduce(astype(expected_gate, dt), probabilities), dt == bfloat16 ? .015f : .003f,
                "heterogeneous GGUF fused Down/reduce M=" + std::to_string(m));
            if (m == 65 && mlx_metal_nax_disabled()) {
                const auto backing = concatenate({input, input}, -1);
                const auto strided = slice(backing, {0, 0}, {m, K});
                const auto unsigned_ids = astype(ids, uint32);
                for (const int tile : {7, 16, 31, 32}) {
                    const auto plan = mlx_expert_route_tiles(unsigned_ids, E, tile);
                    compare(gate.routed_matmul(strided, unsigned_ids, true, &plan), astype(expected_gate, dt),
                        dt == bfloat16 ? .01f : .002f, "heterogeneous SIMD prefill external tile=" + std::to_string(tile));
                    compare(gate.routed_swiglu(up, strided, unsigned_ids, &plan, .7f),
                        moe_limited_swiglu_pair(astype(expected_gate, dt), astype(expected_up, dt), .7f),
                        dt == bfloat16 ? .02f : .003f, "heterogeneous SIMD clipped Gate/Up external plan");
                    compare(gate.routed_weighted_matmul(separate, unsigned_ids, probabilities, &plan),
                        moe_weighted_reduce(astype(expected_gate, dt), probabilities), dt == bfloat16 ? .015f : .003f,
                        "heterogeneous SIMD Down/reduce external plan");
                }
            }
        }
    }
    bool rejected = false;
    try {
        MlxGgufWeight::from_mixed_experts({E, N, K}, {0, 0}, {"GGUF_PTQ1_0", "GGUF_PTQ1_0"},
            {zeros({N * K / 128 * 28}, uint8), zeros({N * K / 128 * 28}, uint8)});
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) throw std::runtime_error("duplicate mixed GGUF expert accepted");
    const std::vector<ggml_type> all_types{GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K, GGML_TYPE_Q8_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_0, GGML_TYPE_PTQ1_0, GGML_TYPE_MXFP4,
        GGML_TYPE_TQ2_T, GGML_TYPE_TQK6, GGML_TYPE_TQK7, GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7,
        GGML_TYPE_TQK5, GGML_TYPE_TQK4};
    std::vector<std::string> names;
    std::vector<array> storage, matrices;
    std::vector<int32_t> experts;
    for (const auto type : all_types) {
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        const auto bytes = packed(type, K, 7);
        std::vector<float> decoded(7 * K);
        if (type == GGML_TYPE_Q8_K)
            dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(bytes.data()), decoded.data(), decoded.size());
        else ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data(), decoded.size());
        experts.push_back(int(experts.size())); names.push_back(std::move(name));
        storage.emplace_back(bytes.begin(), Shape{int(bytes.size())}, uint8);
        matrices.emplace_back(decoded.begin(), Shape{7, K});
    }
    const auto all = MlxGgufWeight::from_mixed_experts({int(all_types.size()), 7, K}, experts, names, storage);
    for (const int m : {1, 8, 65}) {
        const auto input = astype(reshape(sin(arange(m * K, float32) * .02f), {m, K}), float16);
        const auto ids = broadcast_to(array(experts.begin(), Shape{int(experts.size())}, int32), {m, int(experts.size())});
        std::vector<array> expected;
        for (const auto& matrix : matrices) expected.push_back(astype(reference_mm(input, matrix), float16));
        compare(all.routed_matmul(input, ids, true), stack(expected, 1), .003f,
            "all supported heterogeneous GGUF formats with partial output tiles");
    }
    {
        constexpr int width = 256, output = 513;
        std::vector<std::string> formats;
        std::vector<array> data, decoded;
        for (const auto type : types) {
            auto name = "GGUF_" + std::string(ggml_type_name(type));
            for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
            const auto bytes = packed(type, width, output);
            std::vector<float> values(output * width);
            ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), values.size());
            formats.push_back(std::move(name));
            data.emplace_back(bytes.begin(), Shape{int(bytes.size())}, uint8);
            decoded.emplace_back(values.begin(), Shape{output, width});
        }
        const auto expanded = MlxGgufWeight::from_mixed_experts({4, output, width}, {0, 1, 2, 3}, formats, data);
        for (const int m : {9, 65}) {
            const auto input = astype(reshape(sin(arange(m * width, float32) * .017f), {m, width}), float16);
            const auto ids = broadcast_to(array({0, 1, 2, 3}, int32), {m, 4});
            compare(expanded.routed_matmul(input, ids, true),
                astype(reference_routed_mm(input, ids, stack(decoded), true), float16), .003f,
                "heterogeneous expanding prefill and partial output tile M=" + std::to_string(m));
        }
    }
    std::cout << "heterogeneous compressed experts M=1..8/prefill/GateUp/lifetime passed\n";
}

void check_gated_residual() {
    const auto down_bytes = packed(GGML_TYPE_Q8_0, 512, 128);
    const auto up_bytes = packed(GGML_TYPE_Q8_0, 128, 512);
    const auto down = MlxGgufWeight::from_bytes("GGUF_Q8_0", down_bytes, {128, 512});
    const auto up = MlxGgufWeight::from_bytes("GGUF_Q8_0", up_bytes, {512, 128});
    for (const auto dtype : {float16, bfloat16, float32}) for (const int streams : {2, 4}) {
        const auto injection = reshape(sin(arange(streams * 512, float32) * .017f) * .01f, {streams, 512});
        for (const int rows : {1, 2, 6, 7, 8}) {
            const auto input = astype(reshape(cos(arange(rows * 512, float32) * .013f), {1, rows, 512}), dtype);
            auto low = down.matmul(input) / array(float(streams), dtype);
            low = low * sigmoid(low);
            auto expected = mean(reshape(sigmoid(up.matmul(low)) * input, {1, rows, streams, 512 / streams}), -2);
            const auto injected = reshape(reference_mm(reshape(input, {rows, 512}), injection), {1, rows, streams});
            auto expected_gate = array(2.f, dtype) * sigmoid(injected / array(float(streams), dtype));
            for (const bool gate : {false, true}) {
                const auto output = down.gated_residual(input, up, gate ? &injection : nullptr, streams);
                if (!output || (*output)[0].dtype() != dtype || output->size() != std::size_t(gate ? 2 : 1))
                    throw std::runtime_error("GGUF mHC lost output shape or precision");
                compare((*output)[0], expected, dtype == bfloat16 ? .008f : dtype == float16 ? .001f : 5e-5f,
                    "GGUF fused Q8 mHC mix");
                if (gate) compare((*output)[1], expected_gate, 5e-5f, "GGUF fused dense mHC gate");
            }
            if (rows == 1) {
                const auto norm = zeros({512}, float32);
                const auto normalized = qwen4_grouped_rms_norm(input, norm, 512 / streams);
                auto projected = down.matmul(normalized) / array(float(streams), dtype);
                projected = projected * sigmoid(projected);
                const auto expected_branch = mean(reshape(sigmoid(up.matmul(projected)) * normalized,
                    {1, 1, streams, 512 / streams}), -2);
                const auto actual = qwen4_gated_residual_pre(input, norm, MlxLinear(down), MlxLinear(up),
                    MlxLinear(injection), 512 / streams, streams);
                compare(actual.branch, expected_branch, dtype == bfloat16 ? .008f : .0001f, "GGUF single-token mHC dispatch");
            }
        }
    }
    if (down.gated_residual(zeros({1, 9, 512}), up, nullptr, 4))
        throw std::runtime_error("GGUF mHC selected an unsupported prefill shape");
    for (const auto dimensions : {std::array{32, 32, 8}, std::array{10240, 320, 4}, std::array{4096, 32, 8}})
      for (const int rows : {1, 2, 6, 7, 8}) {
        const auto width = dimensions[0], rank = dimensions[1], streams = dimensions[2];
        const auto low_weight = MlxGgufWeight::from_bytes("GGUF_Q8_0", packed(GGML_TYPE_Q8_0, width, rank), {rank, width});
        const auto high_weight = MlxGgufWeight::from_bytes("GGUF_Q8_0", packed(GGML_TYPE_Q8_0, rank, width), {width, rank});
        const auto input = slice(reshape(cos(arange(2 * rows * width, float32) * .013f), {1, rows, 2 * width}),
            {0, 0, 0}, {1, rows, 2 * width}, {1, 1, 2});
        auto low = low_weight.matmul(input) / array(float(streams));
        low = low * sigmoid(low);
        const auto expected = mean(reshape(sigmoid(high_weight.matmul(low)) * input,
            {1, rows, streams, width / streams}), -2);
        const auto plain = low_weight.gated_residual(input, high_weight, nullptr, streams);
        if (!plain || plain->size() != 1) throw std::runtime_error("GGUF single-token mHC without injection failed");
        compare((*plain)[0], expected, 5e-5f, "GGUF strided single-token mHC mix");
        for (const auto dtype : {float16, bfloat16, float32}) {
            const auto injection = astype(reshape(sin(arange(streams * width, float32) * .017f) * .01f,
                {streams, width}), dtype);
            const auto projected = MlxLinear(injection)(input);
            const auto reference = rows == 1 ? projected : astype(reshape(reference_mm(
                astype(reshape(input, {rows, width}), dtype), astype(injection, float32)),
                {1, rows, streams}), projected.dtype());
            const auto expected_gate = array(2.f) * sigmoid(reference / array(float(streams)));
            const auto output = low_weight.gated_residual(input, high_weight, &injection, streams);
            if (!output || output->size() != 2) throw std::runtime_error("GGUF single-token mHC injection failed");
            compare((*output)[0], expected, 5e-5f, "GGUF real-shape single-token mHC mix");
            compare((*output)[1], expected_gate, rows == 1 ? 0.f : dtype == bfloat16 ? .008f :
                dtype == float16 ? .001f : 5e-5f, "GGUF independent injection precision rows=" +
                std::to_string(rows) + " width=" + std::to_string(width) + " dtype=" + std::to_string(int(dtype.val())));
        }
    }
}
void check_mixed_formats() {
    constexpr int K = 512, N = 37, E = 2;
    const auto weight = [&](ggml_type type) {
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        return MlxGgufWeight::from_bytes(name, packed(type, K, N * E), {E, N, K});
    };
    for (const auto& pair : std::vector<std::pair<ggml_type, ggml_type>>{
            {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS}, {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S},
            {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S}, {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ2_XXS},
            {GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XS}, {GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS}}) {
        const auto gate = weight(pair.first), up = weight(pair.second);
        for (int rows : {1, 6}) for (const auto dtype : {float16, float32}) {
            const auto input = astype(reshape(cos(arange(rows * K, float32) * .031f), {rows, K}), dtype);
            std::vector<int> ids(rows * 3);
            for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = int(i % 4) - 1;
            const array routes(ids.begin(), {rows, 3});
            compare(gate.routed_swiglu(up, input, routes),
                moe_swiglu_pair(gate.routed_matmul(input, routes, true), up.routed_matmul(input, routes, true)),
                dtype == float16 ? .002f : .0001f, "heterogeneous GGUF Gate/Up");
        }
    }
}
void check_mixed_prefill_formats() {
    constexpr int K = 512, N = 256, E = 3, M = 65;
    const auto weight = [&](ggml_type type, int output, int input) {
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        return MlxGgufWeight::from_bytes(name, packed(type, input, output * E), {E, output, input});
    };
    const auto ids = reshape(remainder(arange(M * 3, int32), array(5, int32)) - 1, {M, 3});
    const auto probabilities = full({M, 3}, .25f, float32);
    const auto down = weight(GGML_TYPE_IQ4_NL, K, N);
    for (const auto& pair : std::vector<std::pair<ggml_type, ggml_type>>{
            {GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_XXS}, {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_XS},
            {GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_K}, {GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS}}) {
        const auto gate = weight(pair.first, N, K), up = weight(pair.second, N, K);
        for (const auto dtype : {float16, bfloat16, float32}) {
            const auto input = astype(reshape(cos(arange(M * K, float32) * .031f), {M, K}), dtype);
            const auto expected = moe_weighted_reduce(down.routed_matmul(gate.routed_swiglu(up, input, ids), ids, false), probabilities);
            const int tile = gguf_prefill_tile_rows(gate, up, down, ids);
            const MlxGgufPrefillPlan plan(input, ids, E, tile);
            const auto hidden = plan.gate_up(gate, up);
            compare(plan.down(down, hidden, probabilities), expected,
                dtype == bfloat16 ? .02f : .003f, "heterogeneous GGUF complete prefill FFN");
            const auto pairs = down.routed_matmul(plan.restore(hidden), ids, false);
            const auto float_expected = sum(astype(pairs, float32) * expand_dims(probabilities, -1), -2);
            const auto accumulated = plan.down(down, hidden, probabilities, true);
            if (accumulated.dtype() != float32) throw std::runtime_error("GGUF prefill lost float accumulation");
            compare(accumulated, float_expected, dtype == bfloat16 ? .02f : .003f,
                "GGUF prefill float accumulation");
        }
    }
}
void check_projection_batch() {
    constexpr int K = 512;
    const std::vector<MlxGgufWeight> weights{
        MlxGgufWeight::from_bytes("GGUF_Q4_K", packed(GGML_TYPE_Q4_K, K, 64), {64, K}),
        MlxGgufWeight::from_bytes("GGUF_Q6_K", packed(GGML_TYPE_Q6_K, K, 128), {128, K}),
        MlxGgufWeight::from_bytes("GGUF_IQ4_XS", packed(GGML_TYPE_IQ4_XS, K, 64), {64, K}),
        MlxGgufWeight::from_bytes("GGUF_Q5_K", packed(GGML_TYPE_Q5_K, K, 64), {64, K}),
        MlxGgufWeight::from_bytes("GGUF_Q5_1", packed(GGML_TYPE_Q5_1, K, 35), {35, K})};
    std::vector<MlxLinear> linears;
    std::vector<const MlxLinear*> references;
    for (const auto& weight : weights) linears.emplace_back(weight);
    for (const auto& linear : linears) references.push_back(&linear);
    const MlxProjectionBatch batch(std::move(references));
    for (const int rows : {1, 2, 3, 4, 5, 6, 7, 8, 64, 65, 128}) for (const auto dtype : {float16, bfloat16, float32}) {
        const auto input = astype(transpose(reshape(sin(arange(K * rows, float32) * .013f), {K, rows})), dtype);
        const auto actual = batch(input);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            if (actual[index].dtype() != dtype || actual[index].shape() != Shape{rows, weights[index].output_size()})
                throw std::runtime_error("GGUF batched projection changed shape or output dtype");
            compare(actual[index], weights[index].matmul(input), 0.f, "GGUF heterogeneous projection batch");
        }
        compare(weights[0].swiglu(weights[2], input),
            moe_swiglu_pair(weights[0].matmul(input), weights[2].matmul(input)), rows <= 6 ? .002f : 0.f,
            "GGUF heterogeneous dense Gate/Up");
    }
    for (const int count : {2, 3}) {
        std::vector<const MlxLinear*> partial;
        for (int i = 0; i < count; ++i) partial.push_back(&linears[i]);
        const MlxProjectionBatch group(std::move(partial));
        const auto input = reshape(sin(arange(3 * K, float32) * .017f), {3, K});
        const auto output = group(input);
        for (int i = 0; i < count; ++i)
            compare(output[i], weights[i].matmul(input), 0.f, "GGUF partial projection batch");
    }
    const MlxLinear wide(MlxGgufWeight::from_bytes("GGUF_Q5_K",
        packed(GGML_TYPE_Q5_K, K, 16384), {16384, K}));
    const MlxProjectionBatch mixed_lanes({&linears.front(), &wide});
    const auto input = reshape(cos(arange(2 * K, float32) * .019f), {2, K});
    const auto output = mixed_lanes(input);
    compare(output[0], linears.front()(input), 0.f, "GGUF narrow projection segment");
    compare(output[1], wide(input), 0.f, "GGUF wide projection segment");
}

void check_compact_expansion() {
    constexpr int K = 256;
    if (!gguf_expert_prefill_available(zeros({8, K}, float32))) return;
    const std::vector<MlxGgufWeight> weights{
        MlxGgufWeight::from_bytes("GGUF_Q4_K", packed(GGML_TYPE_Q4_K, K, 512), {512, K}),
        MlxGgufWeight::from_bytes("GGUF_IQ4_XS", packed(GGML_TYPE_IQ4_XS, K, 256), {256, K}),
        MlxGgufWeight::from_bytes("GGUF_IQ3_S", packed(GGML_TYPE_IQ3_S, K, 128), {128, K}),
        MlxGgufWeight::from_bytes("GGUF_Q6_K", packed(GGML_TYPE_Q6_K, K, 512), {512, K})};
    for (auto dtype : {float16, bfloat16, float32}) for (int rows : {7, 8}) {
        const auto input = astype(reshape(sin(arange(rows * K, float32) * .017f), {rows, K}), dtype);
        auto grouped = MlxGgufStagedPrefill::project_many(weights, input);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto separate = weights[index].matmul(input);
            compare(grouped[index], separate, 0.f, "GGUF expansion/contraction grouped dispatch");
            const bool nax = weights[index].output_size() >= K;
            auto expected = reference_mm(nax ? astype(input, float16) : input,
                weights[index].dequantize(nax ? float16 : float32));
            compare(grouped[index], astype(expected, dtype), dtype == bfloat16 ? .01f : .002f,
                "GGUF compact expansion independent reference");
        }
    }
}
void check_compact_contraction() {
    constexpr int K = 5120, N = 2048;
    if (!gguf_expert_prefill_available(zeros({8, K}, float32))) return;
    for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ3_S, GGML_TYPE_Q6_K}) {
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& ch : name) ch = std::toupper(static_cast<unsigned char>(ch));
        const auto weight = MlxGgufWeight::from_bytes(name, packed(type, K, N), {N, K});
        if (!MlxGgufStagedPrefill::prefers_small_matrix(weight))
            throw std::runtime_error("large GGUF contraction did not select compact split-K");
        const auto dense = weight.dequantize(float16);
        for (auto dtype : {float16, bfloat16, float32}) for (int rows : {7, 8}) {
            const auto input = astype(reshape(sin(arange(rows * K, float32) * .017f) * .1f, {rows, K}), dtype);
            auto expected = astype(reference_mm(astype(input, float16), dense), dtype);
            compare(weight.matmul(input), expected, dtype == bfloat16 ? .01f : .002f,
                "GGUF compact contraction split-K independent reference");
        }
    }
}
void check_split_projection() {
    constexpr int K = 2048;
    if (!gguf_expert_prefill_available(zeros({65, K}, float16)) || metal_apple_gpu_core_count() == 0) return;
    const std::vector<MlxGgufWeight> weights{
        MlxGgufWeight::from_bytes("GGUF_Q4_K", packed(GGML_TYPE_Q4_K, K, 64), {64, K}),
        MlxGgufWeight::from_bytes("GGUF_Q6_K", packed(GGML_TYPE_Q6_K, K, 128), {128, K}),
        MlxGgufWeight::from_bytes("GGUF_IQ4_XS", packed(GGML_TYPE_IQ4_XS, K, 2560), {2560, K}),
        MlxGgufWeight::from_bytes("GGUF_IQ3_S", packed(GGML_TYPE_IQ3_S, K, 64), {64, K})};
    const auto pair_up = MlxGgufWeight::from_bytes("GGUF_IQ4_XS", packed(GGML_TYPE_IQ4_XS, K, 128), {128, K});
    for (const auto dtype : {float16, bfloat16, float32}) for (const int rows : {1, 2, 3, 4, 5, 6, 7, 8}) {
        const auto input = astype(reshape(sin(arange(rows * K, float32) * .013f), {rows, K}), dtype);
        compare(weights[1].swiglu(pair_up, input),
            moe_swiglu_pair(weights[1].matmul(input), pair_up.matmul(input)), 0.f,
            "GGUF split dense Gate/Up dispatch");
        const auto actual = MlxGgufStagedPrefill::project_split_many(weights, input);
        if (!actual) throw std::runtime_error("GGUF split projection did not activate");
        const auto compiled = compile([&](const std::vector<array>& values) {
            return *MlxGgufStagedPrefill::project_split_many(weights, values.front());
        });
        const auto other = input + array(.25f, dtype);
        auto expected_first = *MlxGgufStagedPrefill::project_split_many(weights, input);
        eval(expected_first);
        auto expected_second = *MlxGgufStagedPrefill::project_split_many(weights, other);
        eval(expected_second);
        const auto first = compiled({input}), second = compiled({other});
        auto together = first;
        together.insert(together.end(), second.begin(), second.end());
        eval(together);
        for (std::size_t i = 0; i < weights.size(); ++i) {
            compare(first[i], expected_first[i], 0.f, "GGUF simultaneous compiled first workspace");
            compare(second[i], expected_second[i], 0.f, "GGUF simultaneous compiled second workspace");
        }
        for (std::size_t i = 0; i < weights.size(); ++i) {
            compare((*actual)[i], astype(reference_mm(input, weights[i].dequantize(float32)), dtype),
                dtype == bfloat16 ? .02f : dtype == float16 ? .003f : 2e-4f, "GGUF split independent reference");
            for (int repeat = 0; repeat < 16; ++repeat) {
                const auto source = input + array(float(repeat + 1) * .001f, dtype);
                const auto current = MlxGgufStagedPrefill::project_split_many(weights, source);
                compare((*current)[i], weights[i].matmul(source), 0.f, "GGUF split cross-threadgroup publication");
                compare(compiled({source})[i], (*current)[i], 0.f, "GGUF split compiled workspace replay");
            }
        }
    }
    const auto long_k = MlxGgufWeight::from_bytes("GGUF_Q6_K",
        packed(GGML_TYPE_Q6_K, 6144, 64), {64, 6144});
    const auto x = reshape(cos(arange(6144, float32) * .019f), {1, 6144});
    compare(long_k.matmul(x), reference_mm(x, long_k.dequantize(float32)), 2e-4f,
        "GGUF eight-way split and metadata boundaries");
    const auto wide = MlxGgufWeight::from_bytes("GGUF_Q4_K",
        packed(GGML_TYPE_Q4_K, K, 16384), {16384, K});
    const std::array mixed{weights.front(), wide};
    const auto source = reshape(cos(arange(K, float32) * .021f), {1, K});
    const auto result = MlxGgufStagedPrefill::project_split_many(mixed, source);
    if (!result) throw std::runtime_error("GGUF mixed split plan did not activate");
    compare((*result)[0], mixed[0].matmul(source), 0.f, "GGUF partitioned mixed segment");
    compare((*result)[1], mixed[1].matmul(source), 0.f, "GGUF unpartitioned mixed segment");
    std::cout << "GGUF single/mixed split publication and independent references passed\n";
}

void check_grouped_shared() {
    for (const int experts : {1, 255, 256, 511, 512}) for (const int tile : {8, 32}) {
        constexpr int rows = 65, selected = 3;
        const auto ids = reshape(remainder(arange(rows * selected, int32), array(experts + 2, int32)) - 1,
            {rows, selected});
        const auto grouped = gguf_group_routes(ids, experts, tile, true);
        eval(ids, grouped.descriptors, grouped.count, grouped.grouped_routes, grouped.route_rows);
        const int count = grouped.count.item<int32_t>();
        int live_rows = 0;
        for (int at = 0; at < count; ++at) {
            const int e = grouped.descriptors.data<int32_t>()[at * 2];
            const int live = grouped.descriptors.data<int32_t>()[at * 2 + 1];
            if (e < 0 || e > experts || live < 1 || live > tile)
                throw std::runtime_error("GGUF grouped descriptor bounds");
            for (int local = 0; local < tile; ++local) {
                const int position = at * tile + local;
                const int original = grouped.grouped_routes.data<int32_t>()[position];
                if (local >= live) {
                    if (original != -1) throw std::runtime_error("GGUF grouped padding was not cleared");
                    continue;
                }
                if (original < 0 || original >= rows * (selected + 1) ||
                    grouped.route_rows.data<int32_t>()[original] != position)
                    throw std::runtime_error("GGUF grouped reverse mapping");
                const int row = original / (selected + 1), rank = original % (selected + 1);
                const int expected = rank == selected ? experts : ids.data<int32_t>()[row * selected + rank];
                if (e != expected) throw std::runtime_error("GGUF grouped expert association");
                ++live_rows;
            }
        }
        int expected_rows = rows;
        for (int i = 0; i < rows * selected; ++i) {
            const int e = ids.data<int32_t>()[i];
            expected_rows += e >= 0 && e < experts;
        }
        if (live_rows != expected_rows) throw std::runtime_error("GGUF grouped route coverage");
    }
    constexpr int E = 8, K = 256, N = 64;
    {
        const array ids({2, 0, 1}, {1, 3}, int32);
        const auto input = ones({1, K}, float16);
        const auto grouped = reshape(arange(6, float32), {3, 2});
        const MlxGgufPrefillPlan direct(input, ids, E, 8);
        compare(direct.restore(grouped), reshape(astype(grouped, float16), {1, 3, 2}),
            0.f, "GGUF direct restore preserves output dtype");
        const auto strided = transpose(reshape(arange(6, float32), {2, 3}));
        compare(direct.restore(strided), reshape(astype(strided, float16), {1, 3, 2}),
            0.f, "GGUF direct restore preserves strided values");
    }
    auto oversized = MlxGgufWeight::from_bytes("GGUF_Q4_K",
        packed(GGML_TYPE_Q4_K, K, 513 * N), {513, N, K});
    oversized.prepare_grouped();
    if (oversized.has_staged_planes())
        throw std::runtime_error("GGUF converted an expert bank beyond grouped route capacity");
    const auto make = [&](ggml_type type, int output, int input, bool shared) {
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        auto weight = MlxGgufWeight::from_bytes(name, packed(type, input, output * (shared ? 1 : E)),
            shared ? Shape{output, input} : Shape{E, output, input});
        if (shared) weight.prepare_grouped();
        return weight;
    };
    const auto gate = make(GGML_TYPE_IQ3_S, N, K, false), up = make(GGML_TYPE_IQ2_XS, N, K, false);
    const auto down = make(GGML_TYPE_IQ4_NL, K, N, false);
    const auto sg = make(GGML_TYPE_IQ4_XS, N, K, true), su = make(GGML_TYPE_Q4_K, N, K, true);
    const auto sd = make(GGML_TYPE_IQ4_NL, K, N, true);
    for (bool shared : {false, true}) for (bool broadcast : {false, true}) {
        const array ids({2, 2, -1, E, E + 1, 4}, {1, 6}, int32);
        const auto input = astype(reshape(sin(arange((broadcast ? 1 : 6) * K, float32) * .017f),
            {broadcast ? 1 : 6, K}), float16);
        const MlxGgufPrefillPlan plan(input, ids, E, 8, broadcast, false);
        const auto projection = plan.project(gate);
        compare(projection, gate.routed_matmul(input, ids, broadcast), 0.f, "GGUF direct single-token projection");
        compare(gate.routed_matmul(reshape(input, {1, broadcast ? 1 : 6, K}), ids, broadcast),
            projection, 0.f, "GGUF direct single-token rank-three input");
        const auto dense = gate.dequantize(float16);
        eval(ids);
        std::vector<array> expected;
        for (int route = 0; route < 6; ++route) {
            const int e = ids.data<int32_t>()[route];
            if (e < 0 || e >= E) { expected.push_back(zeros({1, N}, float16)); continue; }
            const auto x = slice(input, {broadcast ? 0 : route, 0}, {broadcast ? 1 : route + 1, K});
            const auto weight = reshape(slice(dense, {e, 0, 0}, {e + 1, N, K}), {N, K});
            expected.push_back(astype(reference_mm(x, weight), float16));
        }
        compare(reshape(projection, {-1, N}), concatenate(expected, 0), .003f, "GGUF direct route bounds/reference");
        if (broadcast) {
            const MlxGgufPrefillPlan ffn(input, ids, E, 8, true, shared);
            const auto probability = full({1, 6 + int(shared)}, .125f, float32);
            const auto hidden = ffn.gate_up(gate, up, shared ? &sg : nullptr, shared ? &su : nullptr);
            if (hidden.shape() != Shape{6 + int(shared), N})
                throw std::runtime_error("GGUF direct decode retained padded rows");
            const auto output = ffn.down(down, hidden, probability, false, shared ? &sd : nullptr);
            const auto compiled = compile([&](const std::vector<array>& args) {
                const MlxGgufPrefillPlan current(args[0], args[1], E, 8, true, shared);
                return std::vector<array>{current.down(down,
                    current.gate_up(gate, up, shared ? &sg : nullptr, shared ? &su : nullptr),
                    probability, false, shared ? &sd : nullptr)};
            });
            compare(compiled({input, ids}).front(), output, 0.f, "GGUF direct decode compiled replay");
        }
    }
    for (const int m : {2, 3, 6}) for (const auto dtype : {float16, bfloat16, float32})
        for (bool shared : {false, true}) for (int pattern = 0; pattern < 3; ++pattern) {
        const auto input = astype(reshape(sin(arange(m * K, float32) * .017f), {m, K}), dtype);
        std::vector<int> selected(m * 3);
        for (int row = 0; row < m; ++row) for (int route = 0; route < 3; ++route)
            selected[row * 3 + route] = (route + row * (pattern == 0 ? 3 : pattern == 1 ? 0 : 1)) % E;
        if (pattern == 0) { selected[0] = -1; selected.back() = E; }
        const array ids(selected.begin(), Shape{m, 3}, int32);
        const MlxGgufPrefillPlan generic(input, ids, E, 8, true, shared);
        const auto topk = MlxGgufPrefillPlan::from_topk(input, ids, E, 8, shared);
        const auto generic_hidden = generic.gate_up(gate, up, shared ? &sg : nullptr, shared ? &su : nullptr);
        const auto topk_hidden = topk.gate_up(gate, up, shared ? &sg : nullptr, shared ? &su : nullptr);
        compare(topk.restore(topk_hidden), generic.restore(generic_hidden), 0.f,
            "GGUF Top-K live-row bound preserves heterogeneous Gate/Up");
        const auto probability = full({m, 3 + int(shared)}, .25f, float32);
        const auto expected = generic.down(down, generic_hidden, probability, false, shared ? &sd : nullptr);
        compare(topk.down(down, topk_hidden, probability, false, shared ? &sd : nullptr), expected,
            0.f, "GGUF Top-K live-row bound preserves heterogeneous Down/shared expert");
        if (m == 2 && pattern == 1) {
            const auto compiled = compile([&](const std::vector<array>& args) {
                const auto plan = MlxGgufPrefillPlan::from_topk(args[0], args[1], E, 8, shared);
                return std::vector<array>{plan.down(down,
                    plan.gate_up(gate, up, shared ? &sg : nullptr, shared ? &su : nullptr),
                    probability, false, shared ? &sd : nullptr)};
            });
            compare(compiled({input, ids}).front(), expected, 0.f, "GGUF Top-K bounded compiled replay");
        }
    }
    compare(sg.embedding(array({0, 63, -1, 64}, int32), float32),
        concatenate({take(sg.dequantize(float32), array({0, 63}, int32), 0), zeros({2, K})}, 0), 0.f,
        "compressed plane embedding");
    for (const int m : {1, 2, 3, 6, 65}) {
        const auto input = astype(reshape(sin(arange(m * K, float32) * .03f), {m, K}), float16);
        const auto ids = reshape(remainder(arange(m * 3, int32), array(E + 2, int32)) - 1, {m, 3});
        const auto probabilities = full({m, 4}, .25f, float32);
        const MlxGgufPrefillPlan plan(input, ids, E, m <= 6 ? 8 : 32, true, true);
        std::vector<array> outputs;
        eval(ids);
        for (int row = 0; row < m; ++row) {
            auto x = slice(input, {row, 0}, {row + 1, K});
            auto sum = zeros({1, K}, float32);
            for (int route = 0; route < 4; ++route) {
                const int e = route == 3 ? 0 : ids.data<int32_t>()[row * 3 + route];
                if (e < 0 || e >= E) continue;
                const auto w = [&](const MlxGgufWeight& a, const MlxGgufWeight& b) {
                    return route == 3 ? b.dequantize(float16) : reshape(slice(a.dequantize(float16),
                        {e, 0, 0}, {e + 1, a.output_size(), a.input_size()}), {a.output_size(), a.input_size()});
                };
                auto g = astype(reference_mm(x, w(gate, sg)), float16);
                auto u = astype(reference_mm(x, w(up, su)), float16);
                auto hidden = moe_swiglu_pair(g, u);
                sum = sum + astype(astype(reference_mm(hidden, w(down, sd)), float16), float32) * .25f;
            }
            outputs.push_back(astype(sum, float16));
        }
        compare(plan.down(down, plan.gate_up(gate, up, &sg, &su), probabilities, false, &sd),
            concatenate(outputs, 0), .003f, "compressed routed/shared M=" + std::to_string(m));
    }
    std::cout << "compressed routed/shared planes, column/live-row tiles passed\n";
}
void check_native_affine_pairs() {
    constexpr int experts = 8, top_k = 2;
    for (const auto [width, middle] : std::array{std::pair{256, 512}, std::pair{512, 256}})
        for (const int group : {32, 64, 128}) for (const int bits : {2, 3, 4, 5, 6, 8}) {
        const auto fields = [&](int outputs, int columns, float phase) {
            return quantize(astype(reshape(sin(arange(experts * outputs * columns, float32) * .017f + phase)
                * .02f, {experts, outputs, columns}), bfloat16), group, bits, "affine");
        };
        const auto gate = fields(middle, width, 0.f), down = fields(width, middle, 1.f);
        const auto weight = [&](const std::vector<array>& q, bool shared) {
            std::vector<array> values;
            for (const auto& value : q) values.push_back(shared ? take(value, array(0, int32), 0) : value);
            return MlxAffineWeight::from_arrays(values[0], values[1], values[2], bits, group);
        };
        const auto native = [&](const MlxAffineWeight& value) {
            auto prepared = value; prepared.prepare_native();
            if (!prepared.native_weight_ref()) throw std::runtime_error("paired affine fixture did not prepare");
            return *prepared.native_weight_ref();
        };
        const auto g = weight(gate, false), d = weight(down, false);
        const auto sg = weight(gate, true), sd = weight(down, true);
        const auto gn = native(g), dn = native(d), sgn = native(sg), sdn = native(sd);
        const auto router = reshape(cos(arange(experts * width, float32) * .019f), {experts, width}) * .03f;
        const auto scalar = reshape(sin(arange(width, float32) * .011f), {1, width}) * .02f;
        const auto input = astype(reshape(cos(arange(width, float32) * .013f), {1, width}), bfloat16);
        const MlxNativeGgufMoe moe{{gn, gn, dn}, {sgn, sgn, sdn}, router, scalar, arange(experts, uint32), top_k};
        const auto scores = matmul(astype(input, float32), transpose(router));
        const auto ids = astype(slice(argsort(-scores, -1), {0, 0}, {1, top_k}), int32);
        const auto probabilities = softmax(take_along_axis(scores, ids, -1), -1);
        const auto routed_gate = g.routed_matmul(input, ids, true);
        const auto routed = d.routed_matmul(astype(moe_swiglu_pair(routed_gate, routed_gate), bfloat16), ids, false);
        const auto shared_gate = sg.matmul(input);
        const auto shared = sd.matmul(astype(moe_swiglu_pair(shared_gate, shared_gate), bfloat16));
        const auto expected = astype(sum(astype(routed, float32) * expand_dims(probabilities, -1), 1) +
            astype(shared, float32) * sigmoid(matmul(astype(input, float32), transpose(scalar))), bfloat16);
        const auto actual = mlx_native_gguf_moe(moe, input);
        const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
        const auto peak = max(abs(astype(expected, float32))).item<float>();
        if (!std::isfinite(error) || error > .02f * peak + 1e-6f)
            throw std::runtime_error("paired affine MoE failed Q=" + std::to_string(bits) +
                " GS=" + std::to_string(group) + " K=" + std::to_string(width));
    }
}
void check_native_grid_gate_up(int width = 1024, int inner = 256) {
    constexpr int experts = 16;
    MlxKernelPreparation preparation(1, 1);
    const std::array formats{GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S};
    for (std::size_t format = 0; format < formats.size(); ++format) {
        std::vector<MlxNativeGgufWeight> native;
        std::vector<MlxGgufWeight> reference;
        for (const bool routed : {true, false}) for (int part = 0; part < 3; ++part) {
            const auto type = part == 2 ? (inner % 256 ? GGML_TYPE_Q2_0 : GGML_TYPE_Q4_K) : routed ? formats[(format + part) % formats.size()]
                : part == 0 ? GGML_TYPE_Q5_K : GGML_TYPE_Q6_K;
            const int k = part == 2 ? inner : width, n = part == 2 ? width : inner;
            const int rows = n * (routed ? experts : 1);
            const auto bytes = packed(type, k, rows);
            auto weight = mlx_native_gguf_pack(array(bytes.begin(), Shape{int(bytes.size())}, uint8),
                type, rows, k, int(ggml_row_size(type, k)));
            weight.outputs = n; weight.experts = routed ? experts : 1;
            native.push_back(std::move(weight));
            auto name = "GGUF_" + std::string(ggml_type_name(type));
            for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
            reference.push_back(MlxGgufWeight::from_bytes(name, bytes, routed ? Shape{experts, n, k} : Shape{n, k}));
        }
        const auto router = reshape(sin(arange(experts * width, float32) * .019f), {experts, width}) * .03f;
        const auto scalar = reshape(cos(arange(width, float32) * .017f), {1, width}) * .02f;
        const auto slots = remainder(arange(experts, uint32) + array(1u), array(uint32_t(experts)));
        const auto row = astype(reshape(cos(arange(width, float32) * .023f), {1, width}), bfloat16);
        for (const auto& input : {row, concatenate({row, row}, 0), concatenate({row, -row}, 0),
                concatenate({row, -row, row}, 0), concatenate({row, row, -row, -row}, 0)}) {
            const int count = input.shape(0);
            for (const auto dtype : {float32, bfloat16}) for (const int selected : {1, 2, experts - 1, experts}) {
                const auto current_router = astype(router, dtype);
                const auto scores = matmul(astype(input, float32), transpose(astype(current_router, float32)));
                const auto ids = astype(slice(argsort(-scores, -1), {0, 0}, {count, selected}), int32);
                const auto physical = astype(take(slots, ids, 0), int32);
                const auto probability = softmax(take_along_axis(scores, ids, -1), -1);
                const auto gate = reference[0].routed_matmul(input, physical, true);
                const auto up = reference[1].routed_matmul(input, physical, true);
                const auto routed = reference[2].routed_matmul(astype(moe_swiglu_pair(gate, up), bfloat16), physical, false);
                const auto shared = reference[5].matmul(astype(moe_swiglu_pair(reference[3].matmul(input), reference[4].matmul(input)), bfloat16));
                const auto expected = astype(sum(astype(routed, float32) * expand_dims(probability, -1), 1) +
                    astype(shared, float32) * sigmoid(matmul(astype(input, float32), transpose(scalar))), bfloat16);
                const MlxNativeGgufMoe weights{{native[0], native[1], native[2]}, {native[3], native[4], native[5]},
                    current_router, scalar, slots, selected};
                const auto actual = mlx_native_gguf_moe(weights, input);
                if (count == 1 && dtype == float32 && selected == 1) {
                    const auto previous = preparation.size();
                    preparation.collect([&] { preparation.add(actual); });
                    if (format && preparation.size() == previous)
                        throw std::runtime_error("native Gate/Up preparation merges different format specializations");
                }
                compare(actual, expected, .025f,
                    "native grid split-K Gate/Up with mixed formats and compact slots");
            }
        }
    }
    preparation.finish();
}

void check_native_stack() {
    check_native_affine_pairs();
    check_native_grid_gate_up();
    check_native_grid_gate_up(2560, 640);
    constexpr int K = 256, N = 256, E = 8, top_k = 2;
    for (const auto [width, outputs] : std::array{std::pair{512, 768}, std::pair{4096, 64}, std::pair{6144, 2560}}) {
        const auto bytes = packed(GGML_TYPE_IQ4_XS, width, outputs);
        const auto native = mlx_native_gguf_pack(array(bytes.begin(), Shape{int(bytes.size())}, uint8),
            GGML_TYPE_IQ4_XS, outputs, width, int(ggml_row_size(GGML_TYPE_IQ4_XS, width)));
        const auto reference = MlxGgufWeight::from_bytes("GGUF_IQ4_XS", bytes, {outputs, width}).dequantize(float32);
        for (const int count : {1, 2, 3, 4}) for (const auto dtype : {float16, bfloat16, float32}) {
            const auto backing = astype(reshape(sin(arange(5 * width, float32) * .013f), {5, width}), dtype);
            const auto input = slice(backing, {1, 0}, {1 + count, width});
            compare(mlx_native_gguf_project(std::span(&native, 1), input)[0],
                reference_mm(astype(input, bfloat16), reference), .01f, "native IQ4_XS factored projection");
        }
    }
    {
        const auto source = astype(reshape(cos(arange(160 * 2560, float32) * .017f), {160, 2560}) * .05f, bfloat16);
        const auto q = quantize(source, 64, 4, "affine");
        const auto native = mlx_native_affine_pack(q[0], q[1], q[2], 4, 64);
        const auto reference = MlxAffineWeight::from_arrays(q[0], q[1], q[2], 4, 64);
        const auto dense_reference = reference.dequantize(float16);
        for (const int count : {1, 8, 32, 65}) {
            const auto input = astype(reshape(sin(arange(count * 2560, float32) * .013f), {count, 2560}), bfloat16);
            const auto expected = astype(reference_mm(input, dense_reference), bfloat16);
            compare(mlx_native_gguf_project(std::span(&native, 1), input)[0], expected, .01f,
                "native affine projection M=" + std::to_string(count));
            compare(mlx_native_gguf_swiglu(native, native, input),
                astype(moe_swiglu_pair(expected, expected), bfloat16), .025f,
                "native affine Gate/Up M=" + std::to_string(count));
        }
        for (const auto dtype : {float16, float32}) {
            auto backing = astype(reshape(cos(arange(3 * 2560, float32) * .013f), {3, 2560}), dtype);
            const auto input = slice(backing, {1, 0}, {2, 2560});
            const auto actual = mlx_native_gguf_project(std::span(&native, 1), input)[0];
            if (actual.dtype() != dtype) throw std::runtime_error("native single-row projection changed output dtype");
            compare(actual, reference_mm(astype(input, bfloat16), dense_reference), .01f,
                "native single-row split projection with offset input");
        }
    }
    const std::array types{GGML_TYPE_IQ3_XXS, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K};
    std::vector<MlxNativeGgufWeight> routed, shared;
    std::vector<MlxGgufWeight> routed_reference, shared_reference;
    for (const bool experts : {true, false}) for (const auto type : types) {
        const int rows = N * (experts ? E : 1);
        const auto bytes = packed(type, K, rows);
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        auto weight = mlx_native_gguf_pack(array(bytes.begin(), Shape{int(bytes.size())}, uint8),
            type, rows, K, int(ggml_row_size(type, K)));
        weight.outputs = N; weight.experts = experts ? E : 1;
        (experts ? routed : shared).push_back(std::move(weight));
        (experts ? routed_reference : shared_reference).push_back(
            MlxGgufWeight::from_bytes(name, bytes, experts ? Shape{E, N, K} : Shape{N, K}));
    }
    const auto router = reshape(sin(arange(E * K, float32) * .019f), {E, K}) * .03f;
    const auto scalar = reshape(cos(arange(K, float32) * .017f), {1, K}) * .02f;
    MlxNativeGgufMoe moe{{routed[0], routed[1], routed[2]}, {shared[0], shared[1], shared[2]},
        router, scalar, arange(E, uint32), top_k};
    const auto reference_moe = [&](const MlxNativeGgufMoe& weights, const array& input) {
        const int count = int(input.size() / K);
        const auto scores = matmul(astype(input, float32), transpose(weights.router));
        const auto ids = astype(slice(argsort(-scores, -1), {0, 0}, {count, weights.top_k}), int32);
        const auto probabilities = softmax(take_along_axis(scores, ids, -1), -1);
        const auto slots = astype(take(weights.slots, ids, 0), int32);
        const auto gate = routed_reference[0].routed_matmul(input, slots, true);
        const auto up = routed_reference[1].routed_matmul(input, slots, true);
        const auto intermediate = astype(moe_swiglu_pair(gate, up), bfloat16);
        const auto routed_out = routed_reference[2].routed_matmul(intermediate, slots, false);
        const auto sg = shared_reference[0].matmul(input), su = shared_reference[1].matmul(input);
        const auto shared_out = shared_reference[2].matmul(astype(moe_swiglu_pair(sg, su), bfloat16));
        return astype(sum(astype(routed_out, float32) * expand_dims(probabilities, -1), 1) +
            astype(shared_out, float32) * sigmoid(matmul(astype(input, float32), transpose(weights.scalar_gate))), bfloat16);
    };
    for (const auto dtype : {float32, bfloat16}) {
        auto current = moe;
        current.router = astype(router, dtype);
        for (const int count : {1, 2, 3, 4, 6}) {
            const auto input = astype(reshape(cos(arange(count * K, float32) * .023f), {count, K}), bfloat16);
            for (const int selected : {1, top_k, E}) {
                current.top_k = selected;
                compare(mlx_native_gguf_moe(current, input), reference_moe(current, input), .025f,
                    "native single-row Down/combine and floating router precision");
            }
        }
    }
    {
        auto wide = moe;
        wide.router = tile(router, {4, 1});
        wide.slots = remainder(arange(E * 4, uint32), array(uint32_t(E)));
        for (auto& weight : wide.routed) {
            weight.experts *= 4;
            for (auto& plane : weight.planes)
                if (plane.size()) plane = concatenate({plane, plane, plane, plane}, 0);
        }
        const auto input = astype(reshape(cos(arange(K, float32) * .029f), {1, K}), bfloat16);
        for (const int selected : {15, 31, 32}) {
            wide.top_k = selected;
            compare(mlx_native_gguf_moe(wide, input), reference_moe(wide, input), .025f,
                "native Down/combine route-count boundary and compact slot aliases");
        }
    }
    for (const int count : {1, 2, 3, 4, 5, 6, 7, 8, 17, 32, 65}) {
        const auto input = astype(reshape(sin(arange(count * K, float32) * .013f), {count, K}), bfloat16);
        const auto expected = reference_moe(moe, input);
        compare(mlx_native_gguf_moe(moe, input), expected, .025f,
            "native grouped MoE mixed formats M=" + std::to_string(count));
        const auto actual = mlx_native_gguf_project(shared, input);
        for (int i = 0; i < 3; ++i)
            compare(actual[i], shared_reference[i].matmul(input), .015f,
                "native mixed projections M=" + std::to_string(count));
        compare(mlx_native_gguf_swiglu(shared[0], shared[1], input),
            astype(moe_swiglu_pair(shared_reference[0].matmul(input), shared_reference[1].matmul(input)), bfloat16), .025f,
            "native mixed Gate/Up M=" + std::to_string(count));
    }
    for (const int count : {4096, 4161}) {
        const auto input = astype(reshape(sin(arange(count * K, float32) * .013f), {count, K}), bfloat16);
        compare(mlx_native_gguf_moe(moe, input), reference_moe(moe, input), .025f,
            "native grouped MoE prefill boundary M=" + std::to_string(count));
    }
    for (const int count : {2051, 4096, 4161}) for (const auto dtype : {bfloat16, float16, float32}) {
        const auto input = astype(reshape(sin(arange(count * K, float32) * .013f), {count, K}), dtype);
        const auto actual = mlx_native_gguf_project(shared, input);
        eval(actual);
        for (std::size_t i = 0; i < shared.size(); ++i) {
            compare(actual[i], shared_reference[i].matmul(input), .02f, "native chunked mixed projections");
            if (dtype == bfloat16 && actual[i].buffer().ptr() != actual.front().buffer().ptr())
                throw std::runtime_error("native prefill projections copied chunk outputs");
        }
        const std::array matching{shared[0], shared[0], shared[0]};
        const auto grouped = mlx_native_gguf_project(matching, input);
        for (const auto& value : grouped)
            compare(value, shared_reference[0].matmul(input), .02f, "native chunked matching projections");
    }
    const auto initial_stream = default_stream(Device::gpu);
    std::vector<array> pending, expected;
    for (const auto stream : {initial_stream, new_stream(Device::gpu), initial_stream}) {
        set_default_stream(stream);
        for (int iteration = 0; iteration < 8; ++iteration) {
            const int count = std::array{1, 2, 3, 4, 8, 17}[iteration % 6];
            const auto input = astype(reshape(sin(arange(count * K, float32) * .013f + iteration * .4f),
                {count, K}), bfloat16);
            auto current = moe;
            if (iteration % 2) { current.router = -router; current.scalar_gate = -scalar; }
            if (iteration % 3) current.slots = astype(E - 1 - arange(E, int32), uint32);
            pending.push_back(mlx_native_gguf_moe(current, input));
            expected.push_back(reference_moe(current, input));
            pending.push_back(mlx_native_gguf_swiglu(shared[0], shared[1], input));
            expected.push_back(astype(moe_swiglu_pair(shared_reference[0].matmul(input),
                shared_reference[1].matmul(input)), bfloat16));
            const auto projections = mlx_native_gguf_project(shared, input);
            pending.insert(pending.end(), projections.begin(), projections.end());
            for (const auto& weight : shared_reference) expected.push_back(weight.matmul(input));
        }
    }
    for (int iteration = 0; iteration < 5; ++iteration) {
        set_default_stream(new_stream(Device::gpu));
        const auto input = astype(reshape(cos(arange(K, float32) * .013f + iteration * .4f), {1, K}), bfloat16);
        pending.push_back(mlx_native_gguf_moe(moe, input));
        expected.push_back(reference_moe(moe, input));
    }
    set_default_stream(initial_stream);
    eval(pending);
    for (std::size_t i = 0; i < pending.size(); ++i)
        compare(pending[i], expected[i], .025f, "native deferred mixed-plan/stream reuse " + std::to_string(i));
    for (const bool strided : {false, true}) {
        auto backing = astype(reshape(sin(arange(130 * K, float32) * .017f), {130, K}), bfloat16);
        auto input = strided ? transpose(reshape(backing, {K, 130})) : slice(backing, {65, 0}, {130, K});
        auto outputs = mlx_native_gguf_project(shared, input);
        outputs.push_back(mlx_native_gguf_moe(moe, input));
        check_native_views(outputs);
        for (std::size_t i = 0; i < shared.size(); ++i)
            compare(outputs[i], shared_reference[i].matmul(input), .015f, "native offset/strided projection");
        compare(outputs.back(), reference_moe(moe, input), .025f, "native offset/strided MoE");
    }
    for (const auto [width, intermediate] : std::array{std::pair{256, 512}, std::pair{K, N}, std::pair{512, 256}})
        for (const int group : {32, 64, 128}) for (const int shift : {0, 3}) {
        std::vector<MlxNativeGgufWeight> affine_routed, affine_shared;
        std::vector<MlxAffineWeight> affine_routed_reference, affine_shared_reference;
        for (const bool experts : {true, false}) for (int i = 0; i < 3; ++i) {
            const int columns = i == 2 ? intermediate : width, outputs = i == 2 ? width : intermediate;
            const auto shape = experts ? Shape{E, outputs, columns} : Shape{outputs, columns};
            const auto fields = quantize(astype(reshape(sin(arange((experts ? E : 1) * outputs * columns, float32) * .017f + i)
                * .03f, shape), bfloat16), group, std::array{2, 3, 4, 5, 6, 8}[i + shift], "affine");
            const int bits = std::array{2, 3, 4, 5, 6, 8}[i + shift];
            auto packed = mlx_native_affine_pack(fields[0], fields[1], fields[2], bits, group);
            if (packed.outputs != outputs || packed.experts != (experts ? E : 1))
                throw std::runtime_error("native affine expert dimension lost during packing");
            (experts ? affine_routed : affine_shared).push_back(std::move(packed));
            (experts ? affine_routed_reference : affine_shared_reference).push_back(
                MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, group));
        }
        const auto affine_router = reshape(sin(arange(E * width, float32) * .019f), {E, width}) * .03f;
        const auto affine_scalar = reshape(cos(arange(width, float32) * .017f), {1, width}) * .02f;
        const MlxNativeGgufMoe affine_moe{{affine_routed[0], affine_routed[1], affine_routed[2]},
            {affine_shared[0], affine_shared[1], affine_shared[2]}, affine_router, affine_scalar, arange(E, uint32), top_k};
        for (int projection = 0; projection < 3; ++projection) {
            auto weight = affine_routed[projection];
            weight.outputs *= weight.experts; weight.experts = 1;
            const auto x = astype(reshape(cos(arange(2 * weight.width, float32) * .013f),
                {2, weight.width}), bfloat16);
            const auto dense = reshape(affine_routed_reference[projection].dequantize(float32),
                {weight.outputs, weight.width});
            const auto expected = astype(reference_mm(x, dense), bfloat16);
            compare(mlx_native_gguf_project(std::span(&weight, 1), x)[0], expected, .01f,
                "native affine two-row CPU projection " + std::to_string(projection) +
                " K=" + std::to_string(weight.width) + " GS=" + std::to_string(group) +
                " shift=" + std::to_string(shift));
        }
        for (const int rows : {1, 6, 65}) {
            const auto input = astype(reshape(cos(arange(rows * width, float32) * .013f), {rows, width}), bfloat16);
            const auto scores = matmul(astype(input, float32), transpose(affine_router));
            const auto ids = astype(slice(argsort(-scores, -1), {0, 0}, {rows, top_k}), int32);
            const auto probability = softmax(take_along_axis(scores, ids, -1), -1);
            const auto routed_reference = [&](int p, const array& x, bool shared) {
                return rows <= 6 ? affine_routed_reference[p].routed_matmul(x, ids, shared)
                    : astype(reference_routed_mm(x, ids, affine_routed_reference[p].dequantize(float16), shared), bfloat16);
            };
            const auto shared_reference = [&](int p, const array& x) {
                return rows <= 6 ? affine_shared_reference[p].matmul(x)
                    : astype(reference_mm(x, affine_shared_reference[p].dequantize(float16)), bfloat16);
            };
            const auto gate = routed_reference(0, input, true);
            const auto up = routed_reference(1, input, true);
            const auto routed_out = routed_reference(2, astype(moe_swiglu_pair(gate, up), bfloat16), false);
            const auto shared_hidden = astype(moe_swiglu_pair(
                shared_reference(0, input), shared_reference(1, input)), bfloat16);
            const auto shared_out = shared_reference(2, shared_hidden);
            const auto expected = astype(sum(astype(routed_out, float32) * expand_dims(probability, -1), 1) +
                astype(shared_out, float32) * sigmoid(matmul(astype(input, float32), transpose(affine_scalar))), bfloat16);
            compare(mlx_native_gguf_moe(affine_moe, input), expected, .025f,
                "native affine complete mixed-format MoE K=" + std::to_string(width) + " N=" +
                std::to_string(intermediate) + " GS=" + std::to_string(group) + " shift=" +
                std::to_string(shift) + " M=" + std::to_string(rows));
        }
    }
    std::cout << "native GGUF repack, grouped projections, mixed Gate/Up and complete MoE passed\n";
}

void check_native_hyper() {
    constexpr int hidden = 2560, width = hidden * 4, rank = 320;
    const auto down = astype(reshape(sin(arange(rank * width, float32) * .017f) * .001f, {rank, width}), bfloat16);
    const auto up = astype(reshape(cos(arange(width * rank, float32) * .013f) * .01f, {width, rank}), bfloat16);
    const auto inject = reshape(sin(arange(4 * width, float32) * .019f) * .01f, {4, width});
    const auto norm = reshape(cos(arange(width, float32) * .013f) * .02f, {width});
    for (const int rows : {1, 2, 3, 6, 8, 33, 64, 65, 2051, 4096, 4161}) {
        const auto input = reshape(sin(arange(rows * width, float32) * .013f), {1, rows, width});
        for (const bool injection : {false, true}) {
            const auto actual = mlx_native_gguf_hyper(input, norm + array(1.f), mlx_native_gguf_float(down),
                mlx_native_gguf_float(up), injection ? &inject : nullptr);
            check_native_views({actual[0], actual[1]});
            const auto expected = qwen4_gated_residual_pre(input, norm, down, up,
                injection ? std::optional<array>(inject) : std::nullopt, hidden, 4, 1e-6f);
            compare(actual[0], expected.branch, .02f, "native hyper branch");
            if (injection) compare(reshape(actual[1], {1, rows, 4}), *expected.injection, .001f, "native hyper injection");
            {
                const auto packed_input = astype(input, bfloat16);
                const auto direct = mlx_native_gguf_hyper(packed_input, norm + array(1.f), mlx_native_gguf_float(down),
                    mlx_native_gguf_float(up), injection ? &inject : nullptr);
                const auto promoted = mlx_native_gguf_hyper(astype(packed_input, float32), norm + array(1.f),
                    mlx_native_gguf_float(down), mlx_native_gguf_float(up), injection ? &inject : nullptr);
                compare(direct[0], promoted[0], 0.f, "native BF16 hyper input preserves branch bits");
                if (injection) compare(direct[1], promoted[1], 0.f, "native BF16 hyper input preserves injection bits");
            }
        }
    }
    const auto down_q = quantize(down, 64, 4, "affine"), up_q = quantize(up, 64, 4, "affine");
    const auto native_down = mlx_native_affine_pack(down_q[0], down_q[1], down_q[2], 4, 64);
    const auto native_up = mlx_native_affine_pack(up_q[0], up_q[1], up_q[2], 4, 64);
    const auto down_reference = astype(dequantize(down_q[0], down_q[1], down_q[2], 64, 4, "affine"), bfloat16);
    const auto up_reference = astype(dequantize(up_q[0], up_q[1], up_q[2], 64, 4, "affine"), bfloat16);
    std::vector<array> branches, expected_branches;
    for (const int rows : {1, 7, 8, 17}) {
        const auto input = reshape(sin(arange(rows * width, float32) * .013f), {1, rows, width});
        branches.push_back(mlx_native_gguf_hyper(input, norm + array(1.f), native_down, native_up, &inject)[0]);
        expected_branches.push_back(qwen4_gated_residual_pre(input, norm, down_reference, up_reference,
            inject, hidden, 4, 1e-6f).branch);
    }
    eval(branches);
    for (std::size_t i = 0; i < branches.size(); ++i)
        compare(branches[i], expected_branches[i], .03f, "native hyper prepared affine input");
    std::cout << "native hyper-connection norm, projections, activation and mix passed\n";
}

void check_row_preparation() {
    for (const auto& [format, width] : std::vector<std::pair<std::string, int>>{
            {"GGUF_IQ4_NL", 160}, {"GGUF_Q8_0", 160}, {"GGUF_IQ3_XXS", 256}, {"GGUF_Q4_K", 256}}) {
        const auto stride = gguf_packed_row_bytes(format, width);
        const auto bytes = 17 * stride;
        void* mapping = mmap(nullptr, bytes, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("cannot allocate unreadable GGUF row fixture");
        struct Mapping { void* value; std::size_t bytes; ~Mapping() { munmap(value, bytes); } } owner{mapping, bytes};
        {
            MlxKernelPreparation preparation(8192, 4096);
            const auto backing = std::span(static_cast<const std::uint8_t*>(mapping), bytes);
            MlxMappedGgufRows prepared(backing, format, 17, width);
            const auto count = preparation.size();
            if (!count) throw std::runtime_error("GGUF row decode did not register load-time preparation");
            MlxMappedGgufRows repeated(backing, format, 17, width);
            if (preparation.size() != count) throw std::runtime_error("GGUF row decode preparation was not deduplicated");
            preparation.finish();
        }
        const std::vector<std::uint8_t> packed(bytes, 0);
        MlxMappedGgufRows rows(packed, format, 17, width);
        for (const int count : {1, 2, 6, 17}) {
            MlxGgufRowBatch batch;
            for (int row = 0; row < count; ++row) rows.append_row(row, batch);
            compare(batch.decode(), zeros({count, width}, float16), 0.f, format + " prepared row decode");
        }
    }
    std::cout << "GGUF row preparation avoids source reads, deduplicates and preserves decode\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        mlx::core::set_default_device(mlx::core::Device::gpu);
        auto* ctx = ggml_init({1024 * 1024, nullptr, true});
        ggml_free(ctx);
        if (argc > 1) {
            if (std::string_view(argv[1]) == "prefill-transfers") {
                check_prefill_restore();
                if (gguf_expert_prefill_available(zeros({65, 512}, float16))) check_mixed_prefill_formats();
                return 0;
            }
            if (std::string_view(argv[1]) == "native-stack") { check_native_stack(); return 0; }
            if (std::string_view(argv[1]) == "native-hyper") { check_native_hyper(); return 0; }
            if (std::string_view(argv[1]) == "rows") { check_row_preparation(); return 0; }
            if (std::string_view(argv[1]) == "mhc") { check_gated_residual(); return 0; }
            if (std::string_view(argv[1]) == "grouped") { check_grouped_shared(); return 0; }
            if (std::string_view(argv[1]) == "projections") {
                check_projection_batch(); check_compact_expansion(); check_compact_contraction(); return 0;
            }
            if (std::string_view(argv[1]) == "splits") { check_split_projection(); return 0; }
            if (std::string_view(argv[1]) == "heterogeneous") { check_heterogeneous_experts(); return 0; }
            if (std::string_view(argv[1]) == "gyro") {
                check_group_decoders();
                check_prefill_restore();
                check_sparse_short_routes();
                check_sparse_mixed_prefill();
                check_wide_mixed_prefill();
                for (const auto type : {GGML_TYPE_TQ2_T, GGML_TYPE_TQK4, GGML_TYPE_TQK5, GGML_TYPE_TQK6,
                    GGML_TYPE_TQK7, GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7}) check(type);
                return 0;
            }
            for (int index = 0; index < GGML_TYPE_COUNT; ++index) {
                const auto type = static_cast<ggml_type>(index);
                const auto* name = ggml_type_name(type);
                if (name && std::string_view(name) == argv[1]) { check(type); return 0; }
            }
            throw std::invalid_argument("unknown GGUF test format");
        }
        check_sparse_short_routes();
        check_group_decoders();
        check_prefill_restore();
        check_sparse_mixed_prefill();
        check_wide_mixed_prefill();
        check_row_preparation();
        check_heterogeneous_experts();
        check_gated_residual();
        check_mixed_formats();
        check_projection_batch();
        check_compact_expansion();
        check_compact_contraction();
        if (gguf_expert_prefill_available(zeros({65, 512}, float16))) check_mixed_prefill_formats();
        for (auto type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
            GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
            GGML_TYPE_Q6_K, GGML_TYPE_Q8_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS,
            GGML_TYPE_Q2_0, GGML_TYPE_PTQ1_0, GGML_TYPE_MXFP4, GGML_TYPE_TQ2_T, GGML_TYPE_TQK4, GGML_TYPE_TQK5,
            GGML_TYPE_TQK6, GGML_TYPE_TQK7, GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7}) check(type);
        bool rejected = false;
        try { MlxGgufWeight::from_bytes("GGUF_UNKNOWN", {}, {1, 256}); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("unsupported GGUF format was accepted");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
