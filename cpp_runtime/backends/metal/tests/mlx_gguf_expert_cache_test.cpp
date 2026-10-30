#include "mlx_gguf_expert_cache.h"
#include "mfq_container.h"
#include "mlx_moe_ops.h"
#include "mlx_deepseek_v41_moe.h"
#include "mlx_kernel_prepare.h"
#include "gguf.h"
#include "ggml-quants.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unistd.h>

namespace {
using namespace mfq::metal;
using namespace mlx::core;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void check(const array& actual, const array& expected) {
    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    const auto scale = max(abs(astype(expected, float32))).item<float>();
    require(std::isfinite(error) && error <= .002f * std::max(1.f, scale), "GGUF SSD FFN changed numerics");
}
}

int main() {
    const auto path = std::filesystem::temp_directory_path() /
        ("tinfer-gguf-cache-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        set_default_device(Device::gpu);
        auto* context = ggml_init({8 << 20, nullptr, false});
        auto* gguf = gguf_init_empty();
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u32(gguf, "qwen35.embedding_length", 256);
        gguf_set_val_u32(gguf, "qwen35.attention.head_count", 2);
        gguf_set_val_u32(gguf, "qwen35.block_count", 6);
        std::vector<float> source(4 * 256 * 256);
        for (std::size_t i = 0; i < source.size(); ++i) source[i] = .02f * std::sin(i * .013f);
        for (int layer = 0; layer < 6; ++layer) for (const auto* projection : {"gate", "up", "down"}) {
            const auto type = !layer || std::string(projection) == "up" ? GGML_TYPE_Q8_0 :
                std::string(projection) == "gate" ? GGML_TYPE_Q4_0 : GGML_TYPE_Q5_1;
            auto* tensor = ggml_new_tensor_3d(context, type, 256, 256, 4);
            const auto name = std::string("blk.") + std::to_string(layer) + ".ffn_" + projection + "_exps.weight";
            ggml_set_name(tensor, name.c_str());
            if (type == GGML_TYPE_Q8_0) quantize_row_q8_0_ref(source.data(), static_cast<block_q8_0*>(tensor->data), source.size());
            else if (type == GGML_TYPE_Q4_0) quantize_row_q4_0_ref(source.data(), static_cast<block_q4_0*>(tensor->data), source.size());
            else quantize_row_q5_1_ref(source.data(), static_cast<block_q5_1*>(tensor->data), source.size());
            gguf_add_tensor(gguf, tensor);
        }
        for (int layer : {6, 7}) {
            const std::vector<int32_t> gu{143, 42, 39, 11}, down{42, 39, 11, 143};
            const auto root = "blk." + std::to_string(layer) + ".";
            for (const auto* role : {"gate", "up", "down"}) {
                const auto& types = std::string_view(role) == "down" ? down : gu;
                for (const auto type : {GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K}) {
                    const auto count = std::count(types.begin(), types.end(), int32_t(type));
                    auto* tensor = ggml_new_tensor_3d(context, type, 256, 256, count);
                    auto suffix = std::string(ggml_type_name(type));
                    for (auto& c : suffix) c = std::tolower(static_cast<unsigned char>(c));
                    const auto name = root + "ffn_" + role + "_exps." + suffix + ".weight";
                    ggml_set_name(tensor, name.c_str());
                    auto* bytes = static_cast<uint8_t*>(tensor->data);
                    for (std::size_t i = 0; i < ggml_nbytes(tensor); ++i) bytes[i] = uint8_t(i * 17 + 6);
                    const auto scale = ggml_fp32_to_fp16(.01f);
                    const auto offset = type == GGML_TYPE_PTQ1_0 ? 26 : type == GGML_TYPE_Q3_K ? 108 : 0;
                    for (std::size_t b = 0; b < ggml_nbytes(tensor); b += ggml_type_size(type)) {
                        if (type == GGML_TYPE_MXFP4) bytes[b] = 117;
                        else std::memcpy(bytes + b + offset, &scale, 2);
                    }
                    gguf_add_tensor(gguf, tensor);
                }
            }
            for (const auto* kind : {"gu", "down"}) {
                auto* map = ggml_new_tensor_1d(context, GGML_TYPE_I32, 4);
                const auto name = root + "ffn_exps_" + kind + "_type";
                ggml_set_name(map, name.c_str());
                const auto& values = std::string_view(kind) == "gu" ? gu : down;
                std::memcpy(map->data, values.data(), 16); gguf_add_tensor(gguf, map);
            }
        }
        for (const auto* role : {"gate", "up", "down", "router", "bias"}) {
            const bool router = std::string_view(role) == "router";
            const bool bias = std::string_view(role) == "bias";
            auto* tensor = bias ? ggml_new_tensor_1d(context, GGML_TYPE_F32, 4)
                : ggml_new_tensor_2d(context, GGML_TYPE_F32, 256, router ? 4 : 256);
            const auto name = bias ? std::string("blk.6.exp_probs_b.bias") : router ? std::string("blk.6.ffn_gate_inp.weight")
                : std::string("blk.6.ffn_") + role + "_shexp.weight";
            ggml_set_name(tensor, name.c_str());
            auto* values = static_cast<float*>(tensor->data);
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) values[i] = .02f * std::sin(i * .019f + (bias ? .7f : .1f));
            gguf_add_tensor(gguf, tensor);
        }
        require(gguf_write_to_file(gguf, path.string().c_str(), false), "cannot write GGUF cache fixture");
        gguf_free(gguf); ggml_free(context);
        MfqContainer model(path);
        auto gate = MlxGgufWeight::load(model, "model.block.0.mlp.experts.gate.weight");
        auto up = MlxGgufWeight::load(model, "model.block.0.mlp.experts.up.weight");
        auto down = MlxGgufWeight::load(model, "model.block.0.mlp.experts.down.weight");
        const auto page = std::size_t(getpagesize());
        const auto extent = gguf_packed_row_bytes("GGUF_Q8_0", 256) * 256;
        const auto one_expert = 3 * ((extent + page - 1) / page * page);
        bool undersized = false;
        try { MlxGgufExpertCache invalid(model, {"model.block.0"}, one_expert - 1, 4); }
        catch (const std::invalid_argument&) { undersized = true; }
        require(undersized, "GGUF cache accepted a budget below one expert");
        MlxGgufExpertCache cache(model, {"model.block.0"}, one_expert * 2, 4);
        cache.validate_layer(0, 4, 256, 256);
        const auto x = astype(reshape(sin(arange(2 * 256, float32) * .01f), {2, 256}), float16);
        const array ids({0, 1, 1, 0}, {2, 2}, int32);
        const auto weights = full({2, 2}, .5f);
        const auto reference = [&](const array& routes) {
            auto hidden = moe_swiglu_pair(gate.routed_matmul(x, routes, true), up.routed_matmul(x, routes, true));
            return moe_weighted_reduce(down.routed_matmul(hidden, routes, false), weights);
        };
        const auto parallel = cos(x);
        check(cache.apply_ffn(0, x, ids, weights, &parallel), reference(ids));
        require(parallel.is_available(), "GGUF pipeline did not submit independent shared work");
        require(cache.stats().overlapped_read_batches > 0, "GGUF pipeline did not overlap Gate/Up with Down reads");
        require(cache.stats().misses == 2 && cache.stats().bytes_read == 6 * extent, "GGUF cache read unexpected extents");
        require(cache.stats().read_calls == 3, "GGUF cache did not combine adjacent expert reads");
        check(cache.apply_ffn(0, x, ids, weights), reference(ids));
        require(cache.stats().hits == 2 && cache.stats().misses == 2, "GGUF cache failed to reuse repeated routes");
        const auto original_limit = cache.limit_bytes(), original_resident = cache.resident_bytes();
        undersized = false;
        try { cache.set_limit(one_expert - 1); } catch (const std::invalid_argument&) { undersized = true; }
        require(undersized && cache.limit_bytes() == original_limit && cache.resident_bytes() == original_resident,
            "invalid GGUF shrink changed the existing budget or entries");
        const array only_two({2, 2, 2, 2}, {2, 2}, int32);
        const array only_zero({0, 0, 0, 0}, {2, 2}, int32);
        check(cache.apply_ffn(0, x, only_two, weights), reference(only_two));
        const auto after_insert = cache.stats();
        check(cache.apply_ffn(0, x, only_zero, weights), reference(only_zero));
        require(cache.stats().misses == after_insert.misses && cache.stats().hits == after_insert.hits + 1,
            "GGUF LRU retained numeric expert order instead of the last token's accesses");
        require(cache.resident_bytes() <= cache.limit_bytes(), "GGUF cache exceeded budget");
        require(cache.set_limit(one_expert) > 0 && cache.resident_bytes() <= one_expert, "GGUF cache shrink did not evict");
        const array distinct({0, 1, 2, 3}, {2, 2}, int32);
        check(cache.apply_ffn(0, x, distinct, weights), reference(distinct));
        require(cache.resident_bytes() <= one_expert && cache.stats().evictions >= 3, "GGUF bounded route batches exceeded budget");
        auto delayed = cache.apply_ffn(0, x, ids, weights);
        cache.clear();
        check(delayed, reference(ids));
        delayed = cache.apply_ffn(0, x, ids, weights);
        auto independent = cache.apply_ffn(0, x, only_two, weights);
        check(delayed, reference(ids));
        check(independent, reference(only_two));
        delayed = cache.apply_ffn(0, x, ids, weights);
        cache.set_limit(one_expert * 2);
        check(delayed, reference(ids));
        bool invalid = false;
        try { cache.apply_ffn(0, x, array({0, 1, 2, 4}, {2, 2}, int32), weights); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "GGUF cache accepted an invalid expert");
        cache.clear();
        require(cache.resident_bytes() == 0 && cache.stats().resident_experts == 0, "GGUF cache clear retained entries");
        cache.set_limit(one_expert * 4);
        const auto before_gaps = cache.stats();
        const array gaps({0, 2, 2, 0}, {2, 2}, int32);
        check(cache.apply_ffn(0, x, gaps, weights), reference(gaps));
        require(cache.stats().read_calls - before_gaps.read_calls == 6 &&
            cache.stats().bytes_read - before_gaps.bytes_read == 6 * extent,
            "GGUF cache read a gap between selected experts");
        const auto gate1 = MlxGgufWeight::load(model, "model.block.1.mlp.experts.gate.weight");
        const auto up1 = MlxGgufWeight::load(model, "model.block.1.mlp.experts.up.weight");
        const auto down1 = MlxGgufWeight::load(model, "model.block.1.mlp.experts.down.weight");
        MlxGgufExpertCache multi(model, {"model.block.0", "model.block.1"}, one_expert * 2, 4);
        for (int round = 0; round < 8; ++round) {
            if (round == 2) multi.set_limit(one_expert);
            if (round == 4) multi.set_limit(one_expert * 4);
            if (round == 6) multi.clear();
            check(multi.apply_ffn(0, x, gaps, weights), reference(gaps));
            const auto hidden = moe_swiglu_pair(gate1.routed_matmul(x, ids, true), up1.routed_matmul(x, ids, true));
            check(multi.apply_ffn(1, x, ids, weights), moe_weighted_reduce(down1.routed_matmul(hidden, ids, false), weights));
            require(multi.resident_bytes() <= multi.limit_bytes() &&
                multi.stats().cached_payload_bytes.value() <= multi.resident_bytes(), "GGUF arena exceeded its physical budget");
        }
        MlxGgufExpertCache ahead(model, {"model.block.0", "model.block.1"}, one_expert * 8, 4);
        const auto many = astype(reshape(cos(arange(33 * 256, float32) * .01f), {33, 256}), float16);
        const auto many_ids = reshape(remainder(arange(33 * 2, int32), array(4, int32)), {33, 2});
        const auto many_weights = full({33, 2}, .5f);
        const auto expected = [&](const auto& g, const auto& u, const auto& d) {
            auto hidden = moe_swiglu_pair(g.routed_matmul(many, many_ids, true), u.routed_matmul(many, many_ids, true));
            return moe_weighted_reduce(d.routed_matmul(hidden, many_ids, false), many_weights);
        };
        check(ahead.apply_ffn(0, many, many_ids, many_weights), expected(gate, up, down));
        require(ahead.stats().overlapped_read_submissions == 1, "GGUF prefill did not start next-layer reads");
        check(ahead.apply_ffn(1, many, many_ids, many_weights), expected(gate1, up1, down1));
        require(ahead.stats().prefill_expert_reads == 4 && ahead.resident_bytes() <= ahead.limit_bytes(),
            "GGUF prefetch did not account for its bounded resident experts");
        ahead.clear();
        auto outstanding = ahead.apply_ffn(0, many, many_ids, many_weights);
        ahead.set_limit(one_expert);
        check(outstanding, expected(gate, up, down));
        require(ahead.resident_bytes() <= one_expert, "GGUF shrink retained an active prefetch reservation");
        ahead.set_limit(one_expert * 8); ahead.clear();
        outstanding = ahead.apply_ffn(0, many, many_ids, many_weights);
        ahead.clear();
        check(outstanding, expected(gate, up, down));
        require(ahead.resident_bytes() == 0, "GGUF clear retained an active prefetch reservation");
        const auto zeros_ids = zeros({33, 2}, int32);
        auto sparse_hidden = moe_swiglu_pair(gate1.routed_matmul(many, zeros_ids, true), up1.routed_matmul(many, zeros_ids, true));
        check(ahead.apply_ffn(0, many, many_ids, many_weights), expected(gate, up, down));
        check(ahead.apply_ffn(1, many, zeros_ids, many_weights),
            moe_weighted_reduce(down1.routed_matmul(sparse_hidden, zeros_ids, false), many_weights));
        require(ahead.stats().resident_experts == 5, "GGUF cache retained unused prefetched experts");
        MlxGgufExpertCache bounded_ahead(model, {"model.block.0", "model.block.1"}, one_expert * 4, 4);
        check(bounded_ahead.apply_ffn(0, many, many_ids, many_weights), expected(gate, up, down));
        require(bounded_ahead.stats().overlapped_read_submissions == 0 && bounded_ahead.stats().evictions == 0,
            "GGUF prefetch evicted useful experts to make its own reservation");
        {
            MlxGgufExpertCache abandoned(model, {"model.block.0", "model.block.1"}, one_expert * 8, 4);
            check(abandoned.apply_ffn(0, many, many_ids, many_weights), expected(gate, up, down));
        }
        {
            const std::vector<std::string> prefixes{"model.block.0", "model.block.1", "model.block.2",
                "model.block.3", "model.block.4", "model.block.5"};
            MlxGgufExpertCache retained(model, prefixes, one_expert * 14, 4);
            for (std::size_t layer = 0; layer < prefixes.size(); ++layer)
                retained.apply_ffn(layer, many, many_ids, many_weights).eval();
            const auto before_decode = retained.stats();
            const auto token_input = slice(many, {32, 0}, {33, 256});
            const array tail_ids({1, 1}, {1, 2}, int32);
            const auto token_weights = full({1, 2}, .5f);
            auto hidden = moe_swiglu_pair(gate.routed_matmul(token_input, tail_ids, true),
                up.routed_matmul(token_input, tail_ids, true));
            check(retained.apply_ffn(0, token_input, tail_ids, token_weights),
                moe_weighted_reduce(down.routed_matmul(hidden, tail_ids, false), token_weights));
            require(retained.stats().bytes_read == before_decode.bytes_read,
                "prefill did not retain the early layer's latest expert");
            require(retained.resident_bytes() <= retained.limit_bytes(), "prefill-tail exceeded budget");
            for (std::size_t layer = 0; layer < prefixes.size(); ++layer)
                retained.apply_ffn(layer, many, many_ids, many_weights).eval();
            retained.set_limit(one_expert);
            check(retained.apply_ffn(0, x, ids, weights), reference(ids));
            retained.clear();
            require(retained.resident_bytes() == 0, "prefill-tail clear retained memory");
        }
        {
            const auto g = MlxGgufWeight::load(model, "model.block.6.mlp.experts.gate.weight");
            const auto u = MlxGgufWeight::load(model, "model.block.6.mlp.experts.up.weight");
            const auto d = MlxGgufWeight::load(model, "model.block.6.mlp.experts.down.weight");
            const auto layouts = nlohmann::json::parse(model.header().extra_json.at("gguf_mixed_experts"));
            std::size_t maximum = 0, backing = 0;
            for (int e = 0; e < 4; ++e) {
                std::size_t bytes = 0;
                for (const auto* role : {"gate", "up", "down"}) {
                    const auto extent = layouts.at(std::string("model.block.6.mlp.experts.") + role + ".weight")[e]["nbytes"].get<std::size_t>();
                    bytes += (extent + page - 1) / page * page; backing += extent;
                }
                maximum = std::max(maximum, bytes);
            }
            MlxGgufExpertCache mixed(model, {"model.block.6", "model.block.7"}, maximum * 8, 4);
            require(mixed.backing_bytes() == backing * 2, "mixed GGUF backing bytes used a fixed expert stride");
            {
                MlxKernelPreparation preparation(128, 65);
                mixed.prepare_layer(0, 2, .7f);
                mixed.prepare_layer(1, 2, .7f);
                require(preparation.size() > 0, "fresh mixed GGUF cache did not prepare its kernels");
                preparation.finish();
                require(preparation.unsupported().empty(), "fresh mixed GGUF cache preparation was incomplete");
                require(mixed.resident_bytes() == 0 && mixed.stats().bytes_read == 0,
                    "fresh mixed GGUF preparation allocated or read expert weights");
            }
            for (const int rows : {1, 8, 65}) for (const int capacity : {1, 2, 8}) for (const float limit : {0.f, .7f}) {
                mixed.clear(); mixed.set_limit(maximum * capacity);
                const auto input = astype(reshape(cos(arange(rows * 256, float32) * .017f), {rows, 256}), float16);
                const auto routes = reshape(remainder(arange(rows * 4, int32), array(4, int32)), {rows, 4});
                const auto p = full({rows, 4}, .25f);
                const auto hidden = limit > 0.f
                    ? moe_limited_swiglu_pair(g.routed_matmul(input, routes, true), u.routed_matmul(input, routes, true), limit)
                    : g.routed_swiglu(u, input, routes);
                const auto expected = moe_weighted_reduce(d.routed_matmul(hidden, routes, false), p);
                check(mixed.apply_ffn(0, input, routes, p, nullptr, limit), expected);
                require(mixed.resident_bytes() <= mixed.limit_bytes(), "mixed expert cache exceeded its byte budget");
                if (capacity == 8) {
                    require(mixed.stats().cached_payload_bytes.value() == backing,
                        "mixed GGUF resident payload was counted at the largest expert precision");
                    check(mixed.apply_ffn(1, input, routes, p, nullptr, limit), expected);
                    require(mixed.stats().cached_payload_bytes.value() == backing * 2,
                        "mixed GGUF prefetched payload accounting differs from stored bytes");
                    if (rows == 65) require(mixed.stats().prefill_expert_reads >= 4,
                        "mixed expert cache did not consume its variable-size prefetch");
                }
            }
            mixed.clear();
            require(mixed.resident_bytes() == 0, "mixed GGUF cache retained stale reservations");
            {
                const auto reads = mixed.stats().bytes_read;
                MlxKernelPreparation preparation(4096, 4096);
                mixed.prepare_layer(0, 2, .7f);
                const auto jobs = preparation.size();
                mixed.prepare_layer(0, 2, .7f);
                require(jobs > 0 && preparation.size() == jobs, "mixed GGUF preparation duplicated its kernel families");
                preparation.collect([&] {
                    for (const auto rows : preparation.row_buckets()) {
                        const auto input = zeros({rows, 256}, float16);
                        const auto ids = zeros({rows, 2}, int32);
                        std::optional<MlxGgufRoutePlan> plan;
                        if (rows > 6) plan = gguf_route_plan(ids, 4);
                        const auto hidden = g.routed_swiglu(u, input, ids, plan ? &*plan : nullptr, .7f);
                        const auto pairs = d.routed_matmul(hidden, ids, false, plan ? &*plan : nullptr);
                        preparation.add(moe_weighted_reduce(astype(pairs, float32), zeros({rows, 2}, float32)));
                    }
                });
                require(preparation.size() == jobs, "partial GGUF cache reduction was not prepared at load time");
                preparation.finish();
                require(preparation.unsupported().empty(), "mixed GGUF prefill kernels were not prepared at load time");
                require(mixed.resident_bytes() == 0 && mixed.stats().bytes_read == reads,
                    "mixed GGUF preparation read expert weights");
            }
            DeepseekV41Config config;
            config.hidden = 256; config.moe_inter = 256; config.n_experts = 4; config.top_k = 2;
            config.norm_topk_prob = true; config.routed_scaling = 1.2; config.swiglu_limit = .7;
            const auto root = std::string("model.block.6.mlp");
            auto resident = [&] {
                MlxKernelPreparation preparation(128, 65);
                auto loaded = MlxDeepseekV41Moe::load(model, config, root);
                const auto jobs = preparation.size();
                preparation.collect([&] {
                    for (const auto rows : preparation.row_buckets()) {
                        const auto input = zeros({rows, 256}, float16);
                        const auto ids = zeros({rows, 2}, int32);
                        std::optional<MlxGgufRoutePlan> plan;
                        if (rows > 8) plan = gguf_route_plan(ids, 4);
                        const auto hidden = g.routed_swiglu(u, input, ids, plan ? &*plan : nullptr, .7f);
                        preparation.add(d.routed_weighted_matmul(hidden, ids, zeros({rows, 2}, float32),
                            plan ? &*plan : nullptr));
                    }
                });
                require(jobs > 0 && preparation.size() == jobs,
                    "resident mixed GGUF experts were not prepared at load time");
                preparation.finish();
                return loaded;
            }();
            auto streamed_cache = std::make_shared<MlxGgufExpertCache>(model,
                std::vector<std::string>{"model.block.6"}, maximum, 4);
            auto streamed = MlxDeepseekV41Moe::load(model, config, root, false,
                nullptr, nullptr, 0, nullptr, streamed_cache);
            const auto router = MlxLinear::load(model, root + ".router.weight");
            const auto shared_g = MlxLinear::load(model, root + ".shared_expert.gate.weight");
            const auto shared_u = MlxLinear::load(model, root + ".shared_expert.up.weight");
            const auto shared_d = MlxLinear::load(model, root + ".shared_expert.down.weight");
            const auto bias = load_dense_array(model.record(root + ".router.bias").dtype,
                model.map_record(root + ".router.bias").view());
            for (const int rows : {1, 2, 8, 65}) {
                const auto input = astype(reshape(cos(arange(rows * 256, float32) * .017f), {rows, 256}), float16);
                const auto routes = moe_topk(router(input), 2, false, true, true, false, bias, std::nullopt, 1e-20f, 1.2f);
                const auto hidden = moe_limited_swiglu_pair(g.routed_matmul(input, routes.ids, true),
                    u.routed_matmul(input, routes.ids, true), .7f);
                const auto expected = moe_weighted_reduce(d.routed_matmul(hidden, routes.ids, false), routes.weights);
                const auto expected_shared = shared_d(moe_limited_swiglu_pair(shared_g(input), shared_u(input), .7f));
                for (auto* moe : {&resident, &streamed}) {
                    auto output = moe->forward(input);
                    check(output.routed, expected);
                    check(output.shared, expected_shared);
                }
                require(streamed_cache->resident_bytes() <= maximum, "DeepSeek mixed GGUF cache exceeded budget");
            }
            resident.validate_gguf_offload(streamed_cache);
            resident.use_gguf_offload(streamed_cache);
            require(resident.resident_routed_bytes() == 0, "DeepSeek GGUF offload retained its resident bank");
            check(resident.forward(x).routed, streamed.forward(x).routed);
        }
        const auto truncated_offset = model.record("model.block.1.mlp.experts.gate.weight").offset;
        require(truncated_offset > 0, "GGUF failure fixture has no physical next-layer offset");
        std::filesystem::resize_file(path, truncated_offset);
        MlxGgufExpertCache failed_ahead(model, {"model.block.0", "model.block.1"}, one_expert * 8, 4);
        for (int attempt = 0; attempt < 2; ++attempt) {
            check(failed_ahead.apply_ffn(0, many, many_ids, many_weights), expected(gate, up, down));
            bool read_failed = false;
            try { failed_ahead.apply_ffn(1, many, many_ids, many_weights); }
            catch (const std::runtime_error&) { read_failed = true; }
            require(read_failed && failed_ahead.stats().failures == std::uint64_t(attempt + 1),
                "GGUF prefetch swallowed a failed read or reused its failed reservation");
            failed_ahead.clear();
            require(failed_ahead.resident_bytes() == 0 && failed_ahead.stats().resident_experts == 0,
                "GGUF failed prefetch retained pool reservations");
        }
        std::filesystem::remove(path);
        std::cout << "GGUF SSD cache numerics/reuse/LRU/shrink/bounded batches passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::cerr << error.what() << '\n'; return 1;
    }
}
