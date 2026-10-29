#include "mlx_affine.h"
#include "mlx_tensor.h"
#include "mlx_affine_rows.h"
#include "mlx_float_rows.h"
#include "mlx_affine_indirect.h"
#include "mlx_moe_ops.h"
#include "mlx_weight_residency.h"
#include "mlx_platform.h"
#include "formats/mlx_affine.h"
#include <mlx/io.h>

#include <cmath>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using namespace mlx::core;
using namespace mfq::metal;
void compare(const array& a, const array& b, float tolerance, const std::string& name) {
    const auto error = max(abs(astype(a, float32) - astype(b, float32))).item<float>();
    const auto scale = max(abs(astype(b, float32))).item<float>();
    if (!std::isfinite(error) || error > tolerance * std::max(1.f, scale))
        throw std::runtime_error(name + " error=" + std::to_string(error) + " scale=" + std::to_string(scale));
}
void check(int bits, int group, Dtype dtype) {
    const auto dense = astype(reshape(sin(arange(3 * 32 * 256, float32) * .031f) * .1f, {3, 32, 256}), dtype);
    const auto fields = quantize(dense, group, bits, "affine");
    const auto experts = MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, group);
    const auto weight = MlxAffineWeight::from_arrays(take(fields[0], 0, 0), take(fields[1], 0, 0),
        take(fields[2], 0, 0), bits, group);
    const auto full = experts.dequantize(float32);
    const auto joined = MlxAffineWeight::join_rows(experts, experts);
    if (!joined) throw std::runtime_error("compatible MLX projection rows could not be joined");
    compare(joined->dequantize(float32), concatenate({full, full}, -2), 0.f,
        "MLX joined projections changed raw quantized rows");
    MlxLinear linear(weight); MlxEmbedding embedding(weight);
    MlxProjectionBatch batch({&linear, &linear, &linear, &linear});
    if (embedding.preferred_activation_dtype() != dtype)
        throw std::runtime_error("MLX model activation precision lost its scale dtype");
    for (const int m : {1, 2, 6, 19, 128}) {
        const auto x = reshape(cos(arange(m * 256, float32) * .017f), {m, 256});
        const auto expected = matmul(astype(astype(x, dtype), float32), transpose(astype(take(full, 0, 0), float32)));
        compare(linear(x), expected, dtype == bfloat16 && m >= 128 ? .02f : .012f,
            "MLX ordinary projection M=" + std::to_string(m));
        compare(linear(x), astype(quantized_matmul(astype(x, dtype), take(fields[0], 0, 0),
            take(fields[1], 0, 0), take(fields[2], 0, 0), true, group, bits, "affine"), float32),
            linear.affine_weight_ref()->native_weight_ref() ? .02f : 0.f, "MLX native quantized projection agreement");
        compare(embedding.project(x), linear(x), linear.affine_weight_ref()->native_weight_ref() ? .02f : 0.f,
            "MLX tied LM head");
        const auto projected = batch(x);
        if (projected.size() != 4) throw std::runtime_error("MLX projection batch lost an output");
        for (const auto& value : projected)
            compare(value, linear(x), .02f, "MLX vector/matrix projection batch dispatch");
    }
    const array ids({31, 0, -1, 32, 7}, int32);
    const auto rows = concatenate({take(take(full, 0, 0), array({31, 0}, int32), 0), zeros({2, 256}, dtype),
        take(take(full, 0, 0), array({7}, int32), 0)}, 0);
    compare(embedding(ids, dtype), astype(rows, dtype), 0.f, "MLX selected embeddings");
    const array routes({2, 0, 2, 1, -1, 3}, {2, 3}, int32);
    const auto x = reshape(cos(arange(2 * 256, float32) * .03f), {2, 256});
    const int selected[]{2, 0, 2, 1, -1, 3};
    std::vector<array> expected;
    for (int i = 0; i < 6; ++i) expected.push_back(selected[i] < 0 || selected[i] >= 3 ? zeros({1, 32}) :
        matmul(astype(astype(take(x, array({i / 3}, int32), 0), dtype), float32),
            transpose(astype(take(full, selected[i], 0), float32))));
    compare(experts.routed_matmul(x, routes, true), reshape(concatenate(expected, 0), {2, 3, 32}),
        .012f, "MLX shared routed projection");
    compare(joined->routed_matmul(x, routes, true), concatenate({experts.routed_matmul(x, routes, true),
        experts.routed_matmul(x, routes, true)}, -1), .012f, "MLX joined routed projection");
    const auto separate = reshape(cos(arange(6 * 256, float32) * .013f), {2, 3, 256});
    expected.clear();
    for (int i = 0; i < 6; ++i) expected.push_back(selected[i] < 0 || selected[i] >= 3 ? zeros({1, 32}) :
        matmul(astype(astype(take(reshape(separate, {6, 256}), array({i}, int32), 0), dtype), float32),
            transpose(astype(take(full, selected[i], 0), float32))));
    compare(experts.routed_matmul(separate, routes, false), reshape(concatenate(expected, 0), {2, 3, 32}),
        .012f, "MLX per-route projection");
    for (const bool shared : {true, false}) {
        constexpr int m = 128, routes_per_token = 3;
        const auto ids = reshape(remainder(arange(m * routes_per_token, int32), array(5, int32)) - 1,
            {m, routes_per_token});
        const auto safe = astype(clip(ids, array(0, int32), array(2, int32)), uint32);
        const auto input = reshape(cos(arange(m * (shared ? 1 : routes_per_token) * 256, float32) * .017f), {-1, 1, 256});
        auto lhs = reshape(arange(m * routes_per_token, uint32), ids.shape());
        if (shared) lhs = floor_divide(lhs, array(routes_per_token, uint32));
        auto gold = gather_mm(astype(astype(input, dtype), float32), swapaxes(full, -1, -2), lhs, safe);
        gold = where(expand_dims(logical_and(ids >= 0, ids < 3), -1), reshape(gold, {m, routes_per_token, 32}),
            zeros({m, routes_per_token, 32}));
        compare(experts.routed_matmul(input, ids, shared), gold, .02f, "MLX sorted prefill route restore");
        const MlxAffineRoutePlan plan(ids, 3);
        compare(plan.restore(experts.routed_matmul(input, plan, shared)), gold, .02f,
            "MLX reused route plan");
        if (!shared) {
            const auto order = argsort(reshape(safe, {-1}));
            compare(plan.restore(experts.routed_matmul(take(input, order, 0), plan, false, true)),
                gold, .02f, "MLX intermediate kept in expert order");
        }
    }
    for (const int tokens : {1, 6, 32}) for (const int routes : {1, 3}) {
        const auto ids = reshape(remainder(arange(tokens * routes, int32), array(3, int32)), {tokens, routes});
        const auto input = reshape(cos(arange(tokens * 256, float32) * .017f), {tokens, 256});
        const auto plan = MlxAffineRoutePlan::from_topk(ids, 3);
        compare(plan.restore(experts.routed_matmul(input, plan, true)),
            experts.routed_matmul(input, ids, true), 0.f, "MLX valid top-k route plan");
    }
    std::cout << "MLX affine bits=" << bits << " group=" << group << " projections/embeddings/routes PASS\n";
}
void check_rows() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tinfer-affine-rows-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{path};
    std::filesystem::create_directory(path);
    auto fields = quantize(astype(reshape(sin(arange(128 * 160, float32) * .017f), {128, 160}), bfloat16), 32, 4);
    eval(fields);
    std::ofstream(path / "config.json") << R"({"model_type":"qwen3_5","quantization":{"bits":4,"group_size":32,"mode":"affine"}})";
    save_safetensors((path / "model.safetensors").string(),
        {{"table.weight", fields[0]}, {"table.scales", fields[1]}, {"table.biases", fields[2]}});
    MfqContainer model(path);
    MlxMappedAffineRows table(model, "table.weight"); MlxAffineRowBatch batch;
    const auto identities = table.file_identities();
    if (identities[0] == MfqMappedBytes::FileIdentity{} || identities[0] != identities[1] || identities[0] != identities[2])
        throw std::runtime_error("MLX row fields lost their shared file identity");
    const auto linked = path / "linked";
    std::filesystem::create_directory(linked);
    std::filesystem::copy_file(path / "config.json", linked / "config.json");
    std::filesystem::create_hard_link(path / "model.safetensors", linked / "model.safetensors");
    const MfqContainer linked_model(linked);
    if (MlxMappedAffineRows(linked_model, "table.weight").file_identities() != identities)
        throw std::runtime_error("hard-linked MLX rows were treated as independent I/O domains");
    const auto copied = path / "copied";
    std::filesystem::create_directory(copied);
    std::filesystem::copy_file(path / "config.json", copied / "config.json");
    std::filesystem::copy_file(path / "model.safetensors", copied / "model.safetensors");
    const MfqContainer copied_model(copied);
    if (MlxMappedAffineRows(copied_model, "table.weight").file_identities()[0] == identities[0])
        throw std::runtime_error("independent MLX files share one I/O domain");
    if (table.source_bytes() != fields[0].nbytes() + fields[1].nbytes() + fields[2].nbytes())
        throw std::runtime_error("MLX mapped table telemetry lost packed payloads");
    for (const auto row : {127, 0, 13, 13}) table.append_row(row, batch);
    const auto weight = MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], 4, 32);
    compare(batch.decode(), weight.embedding(array({127, 0, 13, 13}, int32)), 0.f, "MLX mapped selected rows");
    compare(batch.decode(bfloat16), weight.embedding(array({127, 0, 13, 13}, int32), bfloat16),
        0.f, "MLX native BF16 selected rows");
    if (batch.source_bytes_read() != 400) throw std::runtime_error("MLX selected rows expanded or read extra payloads");
    MlxAffineRowBatch merged;
    merged.append_batch(MlxAffineRowBatch{});
    merged.append_batch(batch); merged.append_batch(batch);
    compare(merged.decode(), concatenate({batch.decode(), batch.decode()}, 0), 0.f, "MLX selected row batch order");
    if (merged.source_bytes_read() != 800) throw std::runtime_error("MLX merged row payloads differ");
    bool rejected = false;
    try { table.append_row(128, batch); } catch (const std::out_of_range&) { rejected = true; }
    if (!rejected) throw std::runtime_error("MLX selected row bounds were ignored");
    compare(MlxAffineWeight::load(model, "table.weight").dequantize(float32), weight.dequantize(float32),
        0.f, "MLX composite loader field boundaries");
}

void check_float_rows() {
    for (const auto dtype : {float16, bfloat16, float32}) {
        const std::string name = dtype == float16 ? "F16" : dtype == bfloat16 ? "BF16" : "F32";
        std::vector<float> data{0.f, .25f, -1.f, 2.f, 3.5f, -.1f, .1f, 5.f, -7.f, 11.f, 13.f, -17.f, 19.f, 23.f, 29.f};
        if (dtype != float16) data[7] = 1e30f;
        auto source = astype(array(data.begin(), Shape{3, 5}), dtype); source.eval();
        std::vector<std::uint8_t> bytes(20 + source.nbytes());
        const std::uint32_t ndim = 2;
        const std::int64_t rows = 3, width = 5;
        std::memcpy(bytes.data(), &ndim, 4);
        std::memcpy(bytes.data() + 4, &rows, 8);
        std::memcpy(bytes.data() + 12, &width, 8);
        std::memcpy(bytes.data() + 20, source.data<std::uint8_t>(), source.nbytes());
        MlxMappedFloatRows table(bytes, name);
        if (table.rows() != rows || table.width() != width)
            throw std::runtime_error("floating row geometry lost");
        MlxFloatRowBatch selected, tail;
        table.append_row(2, selected, .5f); table.append_row(0, selected, .5f);
        table.append_row(2, tail, .5f); selected.append_batch(tail);
        if (selected.source_bytes_read() != 3 * width * dtype.size())
            throw std::runtime_error("floating row accounting differs");
        const array ids({2, 0, 2}, int32);
        const auto expected = astype(take(source, ids, 0), float32) * .5f;
        for (const auto output : {float32, bfloat16})
            compare(selected.decode(output), astype(expected, output), 0.f, "floating selected rows " + name);
        if (dtype == float16)
            compare(selected.decode(float16), astype(expected, float16), 0.f, "F16 selected rows");
        for (const auto row : {-1, 3}) {
            bool rejected = false;
            try { table.append_row(row, selected); } catch (const std::out_of_range&) { rejected = true; }
            if (!rejected) throw std::runtime_error("floating row bounds accepted");
        }
        bool rejected = false;
        try { selected.append_batch(selected); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("floating row self append accepted");
        bytes.pop_back(); rejected = false;
        try { MlxMappedFloatRows invalid(bytes, name); } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("truncated floating row accepted");
    }
}

void check_gated_residual() {
    for (const int width : {512, 4096}) for (const auto dtype : {float16, bfloat16}) for (const int bits : {2, 3, 4, 5, 6, 8})
        for (const int group : {32, 64, 128}) {
        const auto make = [&](int output, int input, int precision, float phase) {
            const auto fields = quantize(astype(reshape(sin(arange(output * input, float32) * .017f + phase)
                * .02f, {output, input}), dtype), group, precision, "affine");
            return MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], precision, group);
        };
        const auto down = make(128, width, bits, .1f), up = make(width, 128, bits, .3f);
        const auto inject = make(4, width, bits == 8 ? 4 : 8, .7f);
        for (const int rows : {1, 2, 6}) {
            const auto input = astype(reshape(cos(arange(rows * width, float32) * .013f), {1, rows, width}), dtype);
            auto low = down.matmul(input) / array(4.f, dtype);
            low = low * sigmoid(low);
            const auto mixing = sigmoid(up.matmul(low));
            const auto expected = mean(reshape(mixing * input, {1, rows, 4, width / 4}), -2);
            const auto expected_gate = array(2.f, dtype) * sigmoid(inject.matmul(input) / array(4.f, dtype));
            for (const bool injection : {true, false}) {
                const auto actual = down.gated_residual(input, up, injection ? &inject : nullptr, 4);
                if (!actual || actual->size() != std::size_t(injection ? 2 : 1) || (*actual)[0].dtype() != dtype)
                    throw std::runtime_error("MLX fused gated residual did not preserve shape/precision");
                compare((*actual)[0], expected, dtype == bfloat16 ? .015f : .002f, "MLX fused affine mHC mix");
                if (injection) compare((*actual)[1], expected_gate, dtype == bfloat16 ? .015f : .002f,
                    "MLX fused affine injection gate");
            }
        }
        if (down.gated_residual(zeros({1, 7, width}, dtype), up, &inject, 4))
            throw std::runtime_error("MLX fused mHC selected a prefill shape");
    }
}

void check_indirect() {
    for (const auto dtype : {float16, bfloat16}) for (const int bits : {2, 3, 4, 5, 6, 8})
        for (const int group : {32, 64, 128}) {
        const auto fields = quantize(astype(reshape(sin(arange(3 * 37 * 256, float32) * .017f) * .1f,
            {3, 37, 256}), dtype), group, bits, "affine");
        eval(fields);
        const auto direct = MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, group);
        const auto residency = std::shared_ptr<MlxWeightResidency>(MlxWeightResidency::retain_arrays(fields,
            fields[0].buffer_size() + fields[1].buffer_size() + fields[2].buffer_size(), true));
        std::vector<array> tables;
        for (const auto& field : fields) {
            const auto base = static_cast<const MTL::Buffer*>(field.buffer().ptr())->gpuAddress() + field.offset();
            const std::vector<std::uint64_t> addresses{base, 0, base + field.nbytes() / 3 * 2};
            tables.emplace_back(addresses.begin(), Shape{3}, uint64);
        }
        const auto weight = MlxAffineIndirectWeight::from_address_tables({3, 37, 256}, bits, group, dtype,
            {tables[0], tables[1], tables[2]}, residency);
        for (const int rows : {1, 6, 19}) for (const bool shared : {true, false}) {
            const auto ids = reshape(remainder(arange(rows * 3, int32), array(5, int32)) - 1, {rows, 3});
            const auto x = astype(reshape(cos(arange(rows * (shared ? 1 : 3) * 256, float32) * .013f),
                {rows * (shared ? 1 : 3), 256}), dtype);
            auto expected = direct.routed_matmul(x, ids, shared);
            expected = where(expand_dims(ids != 1, -1), expected, zeros(expected.shape(), dtype));
            compare(weight.routed_matmul(x, ids, shared), expected, dtype == bfloat16 ? .025f : .003f,
                "MLX indirect native affine projection");
            if (shared) {
                const auto projected = weight.routed_matmul(x, ids, true);
                compare(weight.routed_swiglu(weight, x, ids), moe_swiglu_pair(projected, projected),
                    dtype == bfloat16 ? .005f : .001f, "MLX indirect fused Gate/Up");
            }
        }
    }
}

void check_staged() {
    struct Case { int tokens, width, outputs; };
    constexpr int experts = 7, routes = 3;
    for (const auto dtype : {float16, bfloat16}) for (const int bits : {2, 3, 4, 5, 6, 8})
        for (const int group : {32, 64, 128})
        for (const auto test : {Case{25, 256, 64}, Case{96, 1280, 64}, Case{128, 1280, 64},
            Case{256, 1280, 64}, Case{320, 256, 64}, Case{63, 288, 96}}) {
        if (test.width % group) continue;
        const auto fields = quantize(astype(reshape(sin(arange(experts * test.outputs * test.width, float32)
            * .019f) * .03f, {experts, test.outputs, test.width}), dtype), group, bits, "affine");
        const auto weight = MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, group);
        const auto ids = reshape(remainder(arange(test.tokens * routes, int32) * 5 + 1,
            array(experts, int32)), {test.tokens, routes});
        const auto order = astype(argsort(reshape(ids, {-1})), uint32);
        const auto sorted_ids = astype(take(reshape(ids, {-1}), order, 0), uint32);
        const MlxAffineRoutePlan plan(ids, experts);
        for (const bool shared : {true, false}) {
            const auto x = astype(reshape(cos(arange(test.tokens * (shared ? 1 : routes) * test.width, float32)
                * .013f), {-1, test.width}), dtype);
            const auto sorted_x = take(x, shared ? floor_divide(order, array(routes, uint32)) : order, 0);
            const auto expected = gather_qmm(reshape(sorted_x, {-1, 1, test.width}), fields[0], fields[1],
                fields[2], std::nullopt, sorted_ids, true, group, bits, "affine", std::nullopt, true);
            const auto actual = weight.routed_matmul(x, plan, shared);
            compare(actual, expected, test.width % 64 ? .012f : 0.f,
                "MLX staged NAX agreement bits=" + std::to_string(bits) + " tokens=" + std::to_string(test.tokens));
            if (!shared) compare(weight.routed_matmul(sorted_x, plan, false, true), actual,
                0.f, "MLX staged sorted intermediate");
            const auto full = weight.dequantize(dtype);
            const auto dense = gather_mm(reshape(astype(sorted_x, float32), {-1, 1, test.width}),
                swapaxes(astype(full, float32), -1, -2), std::nullopt, sorted_ids);
            compare(actual, dense, dtype == bfloat16 ? .015f : .002f, "MLX staged independent dense reference");
            if (shared) compare(weight.routed_swiglu(x, plan), moe_swiglu_split(actual),
                dtype == bfloat16 ? .008f : .001f, "MLX staged Gate/Up activation epilogue");
        }
        if (group == 64 && test.tokens == 128) {
            const auto compiled = compile([&](const std::vector<array>& values) {
                const MlxAffineRoutePlan current(values[1], experts);
                return std::vector<array>{current.restore(weight.routed_matmul(values[0], current, true)),
                    current.restore(weight.routed_swiglu(values[0], current))};
            });
            for (int repeat = 0; repeat < 4; ++repeat) {
                const auto input = astype(reshape(sin(arange(test.tokens * test.width, float32) * .021f
                    + repeat * .03f), {test.tokens, test.width}), repeat % 2 ? float32 : dtype);
                const auto current_ids = remainder(ids + repeat, array(experts, int32));
                const MlxAffineRoutePlan current(current_ids, experts);
                const auto result = compiled({input, current_ids});
                compare(result[0], current.restore(weight.routed_matmul(input, current, true)), 0.f,
                    "MLX staged compiled route replay");
                compare(result[1], current.restore(weight.routed_swiglu(input, current)), 0.f,
                    "MLX staged compiled Gate/Up replay");
                compare(result[1], current.restore(moe_swiglu_split(weight.routed_matmul(input, current, true))),
                    dtype == bfloat16 ? .008f : .001f, "MLX staged activation output dtype");
            }
        }
    }
    std::cout << "MLX staged geometries, row maps and K/N tails PASS\n";
}

void check_indirect_staged(bool ragged_only = false) {
    constexpr int experts = 7, routes = 3;
    for (const int outputs : {64, 96})
        for (const auto dtype : {float16, bfloat16}) for (const int bits : {2, 3, 4, 5, 6, 8})
        for (const int width : {256, 1280}) {
        if (ragged_only && outputs == 64) continue;
        const auto fields = quantize(astype(reshape(sin(arange(experts * outputs * width, float32) * .017f)
            * .04f, {experts, outputs, width}), dtype), 64, bits, "affine");
        eval(fields);
        const auto direct = MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, 64);
        const auto residency = std::shared_ptr<MlxWeightResidency>(MlxWeightResidency::retain_arrays(fields,
            fields[0].buffer_size() + fields[1].buffer_size() + fields[2].buffer_size(), true));
        std::vector<array> tables;
        for (const auto& field : fields) {
            const auto base = static_cast<const MTL::Buffer*>(field.buffer().ptr())->gpuAddress() + field.offset();
            std::vector<std::uint64_t> addresses(experts);
            for (int e = 0; e < experts; ++e) addresses[e] = e == 2 ? 0 : base + e * (field.nbytes() / experts);
            tables.emplace_back(addresses.begin(), Shape{experts}, uint64);
        }
        const auto weight = MlxAffineIndirectWeight::from_address_tables({experts, outputs, width}, bits, 64,
            dtype, {tables[0], tables[1], tables[2]}, residency);
        for (const int tokens : {25, 128}) for (const bool shared : {false, true}) {
            const auto ids = reshape(remainder(arange(tokens * routes, int32) * 5,
                array(experts + 2, int32)) - 1, {tokens, routes});
            const auto input = astype(reshape(cos(arange(tokens * (shared ? 1 : routes) * width, float32) * .023f),
                {-1, width}), dtype);
            const auto expected = direct.routed_matmul(input, ids, shared);
            const auto gold = where(expand_dims(ids != 2, -1), expected, zeros(expected.shape(), dtype));
            const auto plan = mlx_expert_route_tiles(ids, experts);
            compare(weight.routed_matmul(input, ids, shared, &plan), gold,
                !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5")
                    ? 0.f : dtype == bfloat16 ? .02f : .003f,
                "MLX staged indirect bounds, missing pool and route restore");
        }
    }
    std::cout << "MLX staged indirect shared/per-route inputs and absent experts PASS\n";
}
}
int main(int argc, char** argv) {
    try {
        set_default_device(Device::gpu);
        if (argc == 2 && std::string(argv[1]) == "staged") { check_staged(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "indirect-staged") { check_indirect_staged(); return 0; }
        if (argc == 2 && std::string(argv[1]) == "indirect-staged-ragged") { check_indirect_staged(true); return 0; }
        for (const int bits : {2, 3, 4, 5, 6, 8}) for (const int group : {32, 64, 128})
            for (const auto dtype : {float16, bfloat16}) check(bits, group, dtype);
        check_rows();
        check_float_rows();
        check_gated_residual();
        check_indirect();
        check_staged();
        check_indirect_staged();
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
