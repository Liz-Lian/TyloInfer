#include "mlx_gguf.h"
#include "mlx_gguf_rows.h"
#include "mlx_gguf_prefill.h"
#include "mlx_gguf_staged.h"
#include "mlx_kernel_prepare.h"
#include "metal_device_info.h"
#include "mlx_tensor.h"
#include "mlx_moe_ops.h"
#include "qwen4_ops.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/mman.h>

namespace {
using namespace mlx::core;
using namespace mfq::metal;

void compare(const array& actual, const array& expected, float tolerance, const std::string& label) {
    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    const auto scale = max(abs(astype(expected, float32))).item<float>();
    if (!std::isfinite(error) || error > tolerance * std::max(1.f, scale))
        throw std::runtime_error(label + " error=" + std::to_string(error) + " scale=" + std::to_string(scale));
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
        const auto scale = ggml_fp32_to_fp16(.00025f);
        std::size_t d = type == GGML_TYPE_Q2_K ? 80 : type == GGML_TYPE_Q3_K ? 108 : type == GGML_TYPE_Q6_K ? 208 : 0;
        if (type == GGML_TYPE_Q8_K) {
            float value = .002f;
            std::memcpy(bytes.data() + offset, &value, 4);
        } else std::memcpy(bytes.data() + offset + d, &scale, 2);
        if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1 || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)
            std::memcpy(bytes.data() + offset + 2, &scale, 2);
        if (type == GGML_TYPE_Q2_K) std::memcpy(bytes.data() + offset + 82, &scale, 2);
    }
    return bytes;
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
        const auto gate = MlxGgufWeight::from_bytes(upper, packed(type, K, intermediate * E), {E, intermediate, K});
        const auto down = MlxGgufWeight::from_bytes(upper, packed(type, intermediate, K * E), {E, K, intermediate});
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
    auto separate = reshape(cos(arange(6 * K, float32) * .019f), {2, 3, K});
    expected.clear();
    const auto flat = reshape(separate, {6, K});
    for (int r = 0; r < 6; ++r)
        expected.push_back(chosen[r] < 0 ? zeros({1, N}) : reference_mm(take(flat, array({r}, int32), 0), take(gold, chosen[r], 0)));
    compare(experts.routed_matmul(separate, routes, false), reshape(concatenate(expected, 0), {2, 3, N}), 2e-4f, upper + " down");
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
    for (const auto dimensions : {std::array{32, 32, 8}, std::array{10240, 320, 4}}) {
        const auto width = dimensions[0], rank = dimensions[1], streams = dimensions[2];
        const auto low_weight = MlxGgufWeight::from_bytes("GGUF_Q8_0", packed(GGML_TYPE_Q8_0, width, rank), {rank, width});
        const auto high_weight = MlxGgufWeight::from_bytes("GGUF_Q8_0", packed(GGML_TYPE_Q8_0, rank, width), {width, rank});
        const auto input = slice(reshape(cos(arange(2 * width, float32) * .013f), {1, 1, 2 * width}),
            {0, 0, 0}, {1, 1, 2 * width}, {1, 1, 2});
        auto low = low_weight.matmul(input) / array(float(streams));
        low = low * sigmoid(low);
        const auto expected = mean(reshape(sigmoid(high_weight.matmul(low)) * input,
            {1, 1, streams, width / streams}), -2);
        const auto plain = low_weight.gated_residual(input, high_weight, nullptr, streams);
        if (!plain || plain->size() != 1) throw std::runtime_error("GGUF single-token mHC without injection failed");
        compare((*plain)[0], expected, 5e-5f, "GGUF strided single-token mHC mix");
        for (const auto dtype : {float16, bfloat16, float32}) {
            const auto injection = astype(reshape(sin(arange(streams * width, float32) * .017f) * .01f,
                {streams, width}), dtype);
            const auto projected = MlxLinear(injection)(input);
            const auto expected_gate = array(2.f) * sigmoid(projected / array(float(streams)));
            const auto output = low_weight.gated_residual(input, high_weight, &injection, streams);
            if (!output || output->size() != 2) throw std::runtime_error("GGUF single-token mHC injection failed");
            compare((*output)[0], expected, 5e-5f, "GGUF real-shape single-token mHC mix");
            compare((*output)[1], expected_gate, 0.f, "GGUF independent injection precision");
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
            compare(plan.down(down, plan.gate_up(gate, up), probabilities), expected,
                dtype == bfloat16 ? .02f : .003f, "heterogeneous GGUF complete prefill FFN");
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
            if (std::string_view(argv[1]) == "rows") { check_row_preparation(); return 0; }
            if (std::string_view(argv[1]) == "mhc") { check_gated_residual(); return 0; }
            if (std::string_view(argv[1]) == "grouped") { check_grouped_shared(); return 0; }
            if (std::string_view(argv[1]) == "projections") {
                check_projection_batch(); check_compact_expansion(); check_compact_contraction(); return 0;
            }
            if (std::string_view(argv[1]) == "splits") { check_split_projection(); return 0; }
            for (int index = 0; index < GGML_TYPE_COUNT; ++index) {
                const auto type = static_cast<ggml_type>(index);
                const auto* name = ggml_type_name(type);
                if (name && std::string_view(name) == argv[1]) { check(type); return 0; }
            }
            throw std::invalid_argument("unknown GGUF test format");
        }
        check_row_preparation();
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
            GGML_TYPE_Q2_0}) check(type);
        bool rejected = false;
        try { MlxGgufWeight::from_bytes("GGUF_UNKNOWN", {}, {1, 256}); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("unsupported GGUF format was accepted");
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
