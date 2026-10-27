#include "mlx_affine_expert_cache.h"
#include "mlx_affine.h"
#include "mlx_moe_ops.h"
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
}

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tinfer-mlx-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    struct Cleanup { std::filesystem::path path; ~Cleanup() { std::filesystem::remove_all(path); } } cleanup{path};
    try {
        set_default_device(Device::gpu); std::filesystem::create_directory(path);
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
            for (const int rows : {1, 2, 6, 33}) {
                const auto input = astype(reshape(cos(arange(rows * 256, float32) * .01f), {rows, 256}), bfloat16);
                const auto ids = reshape(remainder(arange(rows * 2, int32), array(4, int32)), {rows, 2});
                const auto weights = full({rows, 2}, .5f, float32);
                const auto intermediate = moe_swiglu_pair(gate.routed_matmul(input, ids, true), up.routed_matmul(input, ids, true));
                const auto expected = moe_weighted_reduce(down.routed_matmul(intermediate, ids, false), weights);
                const auto parallel = cos(input);
                compare(cache.apply_ffn(layer, input, ids, weights, &parallel), expected);
                require(parallel.is_available(), "MLX pipeline did not submit independent shared work");
                compare(cache.apply_ffn(layer, input, ids, weights), expected);
                require(cache.resident_bytes() <= cache.limit_bytes() && cache.stats().cached_payload_bytes <= cache.resident_bytes(),
                    "MLX SSD exceeded its physical budget");
                if (rows == 2) cache.set_limit(one_expert);
                if (rows == 6) { cache.clear(); cache.set_limit(one_expert * 4); }
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
