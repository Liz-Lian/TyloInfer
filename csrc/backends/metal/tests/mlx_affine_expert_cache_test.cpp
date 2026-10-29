#include "mlx_affine_expert_cache.h"
#include "mlx_affine.h"
#include "mlx_affine_indirect.h"
#include "mlx_moe_ops.h"
#include "mlx_platform.h"
#include "mlx_weight_residency.h"
#include "../kernels/mlx/mlx_affine_decode.h"
#include "mfq_container.h"
#include <mlx/io.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <unistd.h>

namespace {
using namespace mfq::metal;
using namespace mlx::core;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void compare(const array& actual, const array& expected) {
    require(actual.dtype() == expected.dtype() && actual.shape() == expected.shape(), "MLX SSD output shape/precision mismatch");
    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    const auto scale = max(abs(astype(expected, float32))).item<float>();
    require(std::isfinite(error) && error <= .03f * std::max(.01f, scale), "MLX SSD FFN numerical disagreement");
}

MlxAffineIndirectWeight bind_fields(const std::vector<array>& fields, const Shape& shape, int bits, int group) {
    auto residency = std::shared_ptr<MlxWeightResidency>(MlxWeightResidency::retain_arrays(fields,
        fields[0].buffer_size() + fields[1].buffer_size() + fields[2].buffer_size(), true));
    std::vector<array> tables;
    for (const auto& field : fields) {
        const auto base = static_cast<const MTL::Buffer*>(field.buffer().ptr())->gpuAddress() + field.offset();
        std::vector<std::uint64_t> addresses;
        for (int expert = 0; expert < shape[0]; ++expert)
            addresses.push_back(base + expert * (field.nbytes() / shape[0]));
        tables.emplace_back(addresses.begin(), Shape{shape[0]}, uint64);
    }
    return MlxAffineIndirectWeight::from_address_tables(shape, bits, group, fields[1].dtype(),
        {tables[0], tables[1], tables[2]}, residency);
}

void vector_codes() {
    const auto kernel = fast::metal_kernel("tinfer_affine_codes4_test", {"packed", "params"}, {"output"}, R"METAL(
        const uint at = thread_position_in_grid.x;
        const uint width = params[0], rows = params[1];
        if (at >= rows * width / 4u) return;
        reinterpret_cast<device uint4*>(output)[at] = tinfer_affine_codes4<Q>(packed,
            at * 4u / width, at * 4u % width, width);
        )METAL", detail::kMlxAffineDecode);
    for (int bits = 1; bits <= 8; ++bits) for (int width : {32, 64, 96, 128, 640, 2560}) {
        const int stride = width / 32 * bits;
        std::vector<std::uint32_t> words(3 * stride), expected(3 * width);
        for (std::size_t i = 0; i < words.size(); ++i) words[i] = 0x9e3779b9u * (i + 1u) + 0x7f4a7c15u;
        for (int row = 0; row < 3; ++row) for (int column = 0; column < width; ++column) {
            const int bit = column * bits, shift = bit % 32, index = row * stride + bit / 32;
            std::uint32_t value = words[index] >> shift;
            if (shift + bits > 32) value |= words[index + 1] << (32 - shift);
            expected[row * width + column] = value & ((1u << bits) - 1u);
        }
        auto output = kernel({array(words.begin(), Shape{int(words.size())}, uint32),
            array({std::uint32_t(width), 3u})}, {{3, width}}, {uint32}, {3 * width / 4, 1, 1}, {128, 1, 1},
            {{"Q", bits}}, std::nullopt, false, {}).front();
        output.eval();
        require(std::equal(expected.begin(), expected.end(), output.data<std::uint32_t>()),
            "MLX vector code extraction differs at a word or row boundary");
    }
}

void vector_projections() {
    const auto source = astype(reshape(sin(arange(4 * 64 * 256, float32) * .013f) * .02f,
        {4, 64, 256}), bfloat16);
    const auto input = astype(reshape(cos(arange(3 * 256, float32) * .017f), {3, 256}), bfloat16);
    const array ids({0, 3, 1, 2, 3, 0}, {3, 2}, int32);
    for (int bits : {2, 3, 4, 5, 6, 8}) for (int group : {32, 64, 128}) {
        auto fields = quantize(source, group, bits, "affine"); eval(fields);
        const auto direct = MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, group);
        const auto indirect = bind_fields(fields, {4, 64, 256}, bits, group);
        const auto expected = direct.routed_matmul(input, ids, true);
        compare(indirect.routed_matmul(input, ids, true), expected);
        compare(indirect.routed_swiglu(indirect, input, ids), moe_swiglu_pair(expected, expected));
    }
}

void route_plan_reuse() {
    if (mlx_metal_nax_disabled() || !mlx_apple_chip_starts_with("Apple M5")) return;
    std::vector<MlxAffineWeight> direct;
    std::vector<MlxAffineIndirectWeight> indirect;
    for (int projection = 0; projection < 3; ++projection) {
        const int width = projection == 2 ? 128 : 1024, output = projection == 2 ? 1024 : 128;
        const int bits = std::array{3, 2, 4}[projection];
        auto source = astype(reshape(sin(arange(4 * output * width, float32) * .013f + projection)
            * .02f, {4, output, width}), bfloat16);
        auto fields = quantize(source, 64, bits, "affine"); eval(fields);
        direct.push_back(MlxAffineWeight::from_arrays(fields[0], fields[1], fields[2], bits, 64));
        indirect.push_back(bind_fields(fields, {4, output, width}, bits, 64));
    }
    const auto input = astype(reshape(cos(arange(257 * 1024, float32) * .01f), {257, 1024}), bfloat16);
    const auto ids = reshape(remainder(arange(257 * 2, int32), array(4, int32)), {257, 2});
    const auto plan = mlx_expert_route_tiles(ids, 4);
    const auto hidden = indirect[0].routed_swiglu(indirect[1], input, ids, &plan);
    require(bool(plan.affine_sorted_ids) && bool(plan.affine_tiles[2]), "MLX Gate/Up did not share its 128-row plan");
    const auto sorted_id = plan.affine_sorted_ids->id(), tile_id = plan.affine_tiles[2]->first.id();
    const auto result = indirect[2].routed_matmul(hidden, ids, false, &plan);
    require(bool(plan.affine_tiles[1]) && plan.affine_tiles[2]->first.id() == tile_id &&
        plan.affine_sorted_ids->id() == sorted_id, "MLX Down did not retain its distinct 96-row plan");
    const auto expected_hidden = moe_swiglu_pair(direct[0].routed_matmul(input, ids, true),
        direct[1].routed_matmul(input, ids, true));
    compare(result, direct[2].routed_matmul(expected_hidden, ids, false));
    compare(indirect[0].routed_swiglu(indirect[1], input, ids, &plan), hidden);
    require(plan.affine_tiles[2]->first.id() == tile_id, "MLX reused routes rebuilt the staged tile table");
}
}

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tinfer-mlx-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{path};
    try {
        set_default_device(Device::gpu); std::filesystem::create_directory(path);
        vector_codes();
        vector_projections();
        route_plan_reuse();
        nlohmann::json quantization{{"bits", 4}, {"group_size", 64}, {"mode", "affine"}};
        std::unordered_map<std::string, array> tensors;
        std::size_t payload = 0, one_expert = 0;
        const auto page = std::size_t(getpagesize());
        for (int layer = 0; layer < 6; ++layer) for (int projection = 0; projection < 3; ++projection) {
            const auto module = "model.block." + std::to_string(layer) + ".mlp.experts." +
                std::array{"gate", "up", "down"}[projection];
            const int input = projection == 2 ? 128 : 256, output = projection == 2 ? 256 : 128;
            const int bits = std::array{3, 2, 4}[projection], group = projection == 1 ? 32 : 64;
            const auto values = astype(reshape(sin(arange(4 * output * input, float32) * .013f + layer * .3f + projection)
                * .02f, {4, output, input}), bfloat16);
            const auto fields = quantize(values, group, bits, "affine"); eval(fields);
            quantization[module] = {{"bits", bits}, {"group_size", group}, {"mode", "affine"}};
            for (int field = 0; field < 3; ++field) {
                tensors.emplace(module + std::array{".weight", ".scales", ".biases"}[field], fields[field]);
                if (layer < 2) payload += fields[field].nbytes();
                if (!layer) one_expert += (fields[field].nbytes() / 4 + page - 1) / page * page;
            }
        }
        std::ofstream(path / "config.json") << nlohmann::json{{"model_type", "qwen3_5"}, {"quantization", quantization}};
        save_safetensors((path / "model.safetensors").string(), tensors);
        MfqContainer model(path);
        MlxAffineExpertCache cache(model, {"model.block.0", "model.block.1"}, one_expert * 2, 4);
        require(cache.backing_bytes() == payload, "MLX SSD accounting included virtual headers or expanded weights");
        for (int layer = 0; layer < 2; ++layer) {
            const auto prefix = "model.block." + std::to_string(layer) + ".mlp.experts.";
            const auto gate = MlxAffineWeight::load(model, prefix + "gate.weight");
            const auto up = MlxAffineWeight::load(model, prefix + "up.weight");
            const auto down = MlxAffineWeight::load(model, prefix + "down.weight");
            cache.validate_layer(layer, 4, 256, 128);
            for (const int rows : {1, 2, 6, 7, 8, 33}) {
                const auto input = astype(reshape(cos(arange(rows * 256, float32) * .01f), {rows, 256}), bfloat16);
                const auto ids = reshape(remainder(arange(rows * 2, int32), array(4, int32)), {rows, 2});
                const auto weights = full({rows, 2}, .5f, float32);
                const auto intermediate = moe_swiglu_pair(gate.routed_matmul(input, ids, true), up.routed_matmul(input, ids, true));
                const auto expected = moe_weighted_reduce(down.routed_matmul(intermediate, ids, false), weights);
                const auto parallel = cos(input);
                const auto misses = cache.stats().misses;
                compare(cache.apply_ffn(layer, input, ids, weights, &parallel), expected);
                if (cache.stats().misses > misses)
                    require(parallel.is_available(), "MLX pipeline did not submit independent shared work");
                compare(cache.apply_ffn(layer, input, ids, weights), expected);
                if (rows == 2 || rows == 7 || rows == 33) {
                    const auto strided_input = slice(reshape(stack({input, input}, -1), {rows, 512}),
                        {0, 0}, {rows, 512}, {1, 2});
                    const auto strided_ids = slice(reshape(stack({ids, ids}, -1), {rows, 4}),
                        {0, 0}, {rows, 4}, {1, 2});
                    eval(strided_input, strided_ids);
                    require(!strided_input.flags().row_contiguous && !strided_ids.flags().row_contiguous,
                        "MLX strided expert fixture became contiguous");
                    compare(cache.apply_ffn(layer, strided_input, strided_ids, weights), expected);
                }
                if (rows == 8) {
                    const auto parent = concatenate({input, input}, 0);
                    const auto alias = slice(parent, {rows, 0}, {rows * 2, 256});
                    compare(cache.apply_ffn(layer, alias, ids, weights), expected);
                }
                require(cache.resident_bytes() <= cache.limit_bytes() && cache.stats().cached_payload_bytes <= cache.resident_bytes(),
                    "MLX SSD exceeded its physical budget");
                if (rows == 2) cache.set_limit(one_expert);
                if (rows == 6) { cache.clear(); cache.set_limit(one_expert * 4); }
                if (rows == 7) cache.set_limit(one_expert);
            }
        }
        {
            const std::vector<std::string> prefixes{"model.block.0", "model.block.1", "model.block.2",
                "model.block.3", "model.block.4", "model.block.5"};
            MlxAffineExpertCache retained(model, prefixes, one_expert * 14, 4);
            const auto many = astype(reshape(cos(arange(33 * 256, float32) * .01f), {33, 256}), bfloat16);
            const auto many_ids = reshape(remainder(arange(33 * 2, int32), array(4, int32)), {33, 2});
            const auto many_weights = full({33, 2}, .5f);
            for (std::size_t layer = 0; layer < prefixes.size(); ++layer)
                retained.apply_ffn(layer, many, many_ids, many_weights).eval();
            const auto before_decode = retained.stats();
            const auto input = slice(many, {32, 0}, {33, 256});
            const array ids({1, 1}, {1, 2}, int32);
            const auto weights = full({1, 2}, .5f);
            const auto gate = MlxAffineWeight::load(model, "model.block.0.mlp.experts.gate.weight");
            const auto up = MlxAffineWeight::load(model, "model.block.0.mlp.experts.up.weight");
            const auto down = MlxAffineWeight::load(model, "model.block.0.mlp.experts.down.weight");
            const auto hidden = moe_swiglu_pair(gate.routed_matmul(input, ids, true), up.routed_matmul(input, ids, true));
            compare(retained.apply_ffn(0, input, ids, weights),
                moe_weighted_reduce(down.routed_matmul(hidden, ids, false), weights));
            require(retained.stats().bytes_read == before_decode.bytes_read,
                "MLX prefill did not retain the early layer's latest expert");
            require(retained.resident_bytes() <= retained.limit_bytes(), "MLX prefill-tail exceeded budget");
            for (std::size_t layer = 0; layer < prefixes.size(); ++layer)
                retained.apply_ffn(layer, many, many_ids, many_weights).eval();
            retained.set_limit(one_expert);
            compare(retained.apply_ffn(0, input, ids, weights),
                moe_weighted_reduce(down.routed_matmul(hidden, ids, false), weights));
            retained.clear();
            require(retained.resident_bytes() == 0, "MLX prefill-tail clear retained memory");
        }
        require(cache.stats().hits > 0 && cache.stats().evictions > 0 && cache.stats().bytes_read > 0,
            "MLX SSD did not exercise reuse/eviction/read paths");
        require(cache.stats().overlapped_read_batches > 0, "MLX pipeline did not overlap its raw segment reads");
        bool invalid = false;
        try { cache.apply_ffn(0, zeros({1, 256}, bfloat16), array({0, 4}, {1, 2}, int32), full({1, 2}, .5f)); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "MLX SSD accepted an out-of-range expert");
        cache.clear();
        require(cache.resident_bytes() == 0 && cache.stats().resident_experts == 0, "MLX SSD clear retained experts");
        std::cout << "MLX SSD mixed formats/GS/readv/reuse/LRU/shrink/partial batches PASS\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
