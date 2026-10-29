#include "mlx_gguf_expert_cache.h"
#include "mfq_container.h"
#include "mlx_moe_ops.h"
#include "gguf.h"
#include "ggml-quants.h"

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
        auto* context = ggml_init({4 << 20, nullptr, false});
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
