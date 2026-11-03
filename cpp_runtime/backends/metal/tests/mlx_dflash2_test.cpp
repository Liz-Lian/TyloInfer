#include "../models/dflash2/mlx_dflash2.h"
#include "mlx_kernel_prepare.h"
#include "qwen4_ops.h"
#include "mlx_linear_attention.h"
#include "mlx_dflash2_serving.h"
#include "mlx_qwen35_causal_lm.h"
#include <mlx/io.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace mlx::core;
using namespace mfq::metal;

int main() {
    const auto directory = std::filesystem::temp_directory_path() / ("tinfer-dflash2-test-" + std::to_string(getpid()));
    try {
        set_default_device(Device::gpu);
        MfqSamplingParams sampling;
        sampling.temperature = 0;
        if (!mlx_dflash2_sampling_supported(sampling, {})) throw std::runtime_error("greedy DFlash2 unavailable");
        sampling.temperature = 1;
        if (!mlx_dflash2_sampling_supported(sampling, {})) throw std::runtime_error("stochastic DFlash2 unavailable");
        sampling.top_k = 1;
        if (!mlx_dflash2_sampling_supported(sampling, {})) throw std::runtime_error("top-k one DFlash2 unavailable");
        sampling.presence_penalty = .1;
        if (mlx_dflash2_sampling_supported(sampling, {})) throw std::runtime_error("penalty DFlash2 accepted");
        sampling.presence_penalty = 0; sampling.enable_mtp = false;
        if (mlx_dflash2_sampling_supported(sampling, {})) throw std::runtime_error("disabled DFlash2 accepted");
        std::filesystem::create_directories(directory);
        nlohmann::json config{{"model_type", "qwen3"}, {"hidden_size", 64}, {"vocab_size", 1024}, {"num_hidden_layers", 1},
            {"intermediate_size", 64}, {"num_attention_heads", 1}, {"num_key_value_heads", 1}, {"head_dim", 64},
            {"sliding_window", 16}, {"rms_norm_eps", 1e-6}, {"rope_parameters", {{"rope_theta", 10000}}}};
        config["dflash_config"] = {{"block_size", 8}, {"conv_group_size", 16}, {"conv_kernel_size", 2},
            {"mask_token_id", 1023}, {"selector_rank", 32}, {"selector_top_k", 16}, {"target_layer_ids", {0}}};
        std::ofstream(directory / "config.json") << config;
        std::unordered_map<std::string, array> weights;
        const auto dense = [&](const std::string& name, Shape shape, float value = 0.f) {
            weights.emplace(name, full(shape, value, bfloat16));
        };
        dense("fc.weight", {64, 64}); dense("hidden_norm.weight", {64}, 1);
        dense("norm.weight", {64}, 1);
        dense("candidate_selector.hidden_projection.weight", {32, 64}, .01f);
        dense("candidate_selector.predecessor_codebook", {1024, 32}, 1);
        std::vector<float> successor(1024 * 32);
        std::fill_n(successor.begin() + 777 * 32, 32, 1.f);
        weights.emplace("candidate_selector.successor_codebook", astype(array(successor.begin(), {1024, 32}), bfloat16));
        const std::string p = "layers.0.";
        for (const auto* name : {"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"})
            dense(p + name + ".weight", {64}, 1);
        for (const auto* name : {"attention_conv", "mlp_conv"}) {
            dense(p + name + ".base_kernel", {2, 2, 64});
            dense(p + name + ".kernel_projection.weight", {16, 64});
        }
        for (const auto* name : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) dense(p + name + ".weight", {64, 64});
        weights.at("fc.weight") = eye(64, bfloat16);
        weights.at(p + "self_attn.k_proj.weight") = eye(64, bfloat16);
        weights.at(p + "self_attn.v_proj.weight") = eye(64, bfloat16);
        save_safetensors((directory / "model.safetensors").string(), weights);
        std::vector<float> head(8 * 1024, -1000.f);
        for (int row = 0; row < 8; ++row) {
            for (int rank = 0; rank < 15; ++rank) head[row * 1024 + 700 + rank] = 15.f - rank;
            head[row * 1024 + 777] = 0.f;
        }
        const auto logits = array(head.begin(), {1, 8, 1024});
        auto draft = MlxDFlash2::load(directory);
        {
            MlxKernelPreparation preparation(128, 32);
            draft.prepare([&](const array&) { return logits; });
            preparation.finish();
        }
        MlxDFlash2Context context;
        draft.reset(context);
        draft.append_context(context, ones({1, 17, 64}, bfloat16), 0);
        const auto original = context;
        auto original_keys = view(*original.layers[0].keys, uint16);
        original_keys.eval();
        const std::vector<std::uint16_t> saved_keys(original_keys.data<std::uint16_t>(),
            original_keys.data<std::uint16_t>() + original_keys.size());
        draft.append_context(context, ones({1, 3, 64}, bfloat16), 17);
        if (context.position != 20 || original.position != 17 || context.bytes() != 6144)
            throw std::runtime_error("draft context accounting or snapshot changed");
        auto updated_keys = view(*context.layers[0].keys, uint16);
        updated_keys.eval();
        if (!std::equal(saved_keys.begin(), saved_keys.end(), original_keys.data<std::uint16_t>()) ||
            all(original_keys == updated_keys).item<bool>())
            throw std::runtime_error("asynchronous draft ring update changed a snapshot or failed to wrap");
        auto selected_hidden = zeros({1}, float32);
        auto proposal = draft.propose(context, 7, ones({1, 8, 64}, bfloat16), [&](const array&) { return logits; },
            [&](std::string_view name, const array& value) { if (name == "selector_hidden") selected_hidden = value; });
        eval(proposal.tokens, proposal.confidence, selected_hidden);
        auto h = astype(selected_hidden, float32); h.eval();
        double edge = 0;
        for (int i = 0; i < 32; ++i) edge += h.data<float>()[32 + i];
        double mass = 1;
        for (int i = 0; i < 15; ++i) mass += std::exp(15. - i - edge);
        for (int i = 0; i < 7; ++i)
            if (proposal.tokens.data<std::int32_t>()[i] != 777 ||
                std::abs(proposal.confidence.data<float>()[i] - 1. / mass) > 1e-5)
                throw std::runtime_error("joint selector differs from CPU reference");
        auto invalid = context; invalid.layers[0].keys.reset();
        bool rejected = false;
        try { draft.propose(invalid, 7, ones({1, 8, 64}, bfloat16), [&](const array&) { return logits; }); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("invalid draft cache was accepted");
        rejected = false;
        try { draft.append_context(context, ones({1, 1, 64}, bfloat16), 19); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected || context.position != 20) throw std::runtime_error("draft accepted a discontinuous append");
        {
            Qwen35Config geometry;
            geometry.model_type = "qwen3_5"; geometry.text_model_type = "qwen3_5_text";
            geometry.vocab_size = 1024; geometry.hidden_size = 64; geometry.intermediate_size = 64;
            geometry.num_hidden_layers = 1; geometry.num_attention_heads = 1; geometry.num_key_value_heads = 1;
            geometry.max_position_embeddings = 128; geometry.head_dim = 64; geometry.rotary_dim = 64;
            geometry.norm_weight_offset = 0; geometry.layer_types = {"full_attention"};
            const MlxRmsNorm norm(ones({64}, float32), 1e-6f);
            const auto linear = [&] { return MlxLinear(zeros({64, 64}, float32)); };
            const auto make_target = [&](bool reject = false) {
                std::vector<MlxQwen35Layer> blocks;
                blocks.emplace_back(MlxQwen35FullAttentionBlock(geometry, norm, linear(), linear(), linear(), linear(),
                    std::nullopt, std::nullopt, norm, MlxQwen35DenseSwiGlu(linear(), linear(), linear())));
                auto output = zeros({1024, 64}, float32);
                output = slice_update(output, ones({1, 64}, float32), {777, 0}, {778, 64});
                if (reject) output = slice_update(output, full({1, 64}, 1.0625f, float32), {88, 0}, {89, 64});
                return MlxQwen35CausalLm(geometry, Qwen35Embedding(MlxEmbedding(ones({1024, 64}, float32))),
                    std::move(blocks), norm, Qwen35Linear(MlxLinear(output)), float16);
            };
            auto target = make_target();
            MlxSamplingParams greedy;
            greedy.temperature = 0;
            const std::vector<std::int64_t> prompt{1, 2, 3};
            for (int depth = 1; depth <= 7; ++depth) {
                std::vector<std::int64_t> tokens;
                MlxDFlash2GenerationStats stats;
                const auto count = draft.generate(target, prompt, greedy, 16,
                    [&](auto token) { tokens.push_back(token); return true; }, {}, 32, std::nullopt, &stats, depth);
                if (count != 16 || tokens != std::vector<std::int64_t>(16, 777) || !stats.cycles ||
                    !stats.accepted || stats.accepted != stats.verified)
                    throw std::runtime_error("Qwen3.5 DFlash2 full generation or depth differs");
                if (target.take_hidden_capture()) throw std::runtime_error("DFlash2 left hidden capture active");
            }
            std::vector<std::int64_t> cancelled;
            if (draft.generate(target, prompt, greedy, 16, [&](auto token) {
                    cancelled.push_back(token); return cancelled.size() < 3;
                }) != 3 || cancelled != std::vector<std::int64_t>(3, 777) || target.cache_position() != 6)
                throw std::runtime_error("Qwen3.5 DFlash2 ignored callback cancellation");
            const int position = target.cache_position();
            if (draft.generate(target, prompt, greedy, 0) != 0 || target.cache_position() != position)
                throw std::runtime_error("zero-token DFlash2 changed target state");
            bool invalid = false;
            try { draft.generate(target, {1024}, greedy, 1); } catch (const std::invalid_argument&) { invalid = true; }
            if (!invalid || target.cache_position() != position)
                throw std::runtime_error("invalid DFlash2 prompt changed target state");
            invalid = false;
            int emitted = 0;
            try { draft.generate(target, prompt, greedy, 16, [&](auto) {
                if (++emitted == 3) throw std::runtime_error("cancel");
                return true;
            }); }
            catch (const std::runtime_error&) { invalid = true; }
            if (!invalid || emitted != 3 || target.cache_position() != 6 || target.take_hidden_capture())
                throw std::runtime_error("DFlash2 callback exception leaked capture or unaccepted state");
            if (draft.generate(target, prompt, greedy, 2) != 2)
                throw std::runtime_error("DFlash2 target failed to recover after callback exception");
            if (draft.generate(target, prompt, greedy, 16, {}, {}, 32, 777) != 0)
                throw std::runtime_error("DFlash2 emitted the requested stop token");
            if (draft.generate(target, prompt, greedy, 200) != 126 || target.cache_position() != 128)
                throw std::runtime_error("DFlash2 generation crossed the target context limit");
            auto rejecting_target = make_target(true);
            std::vector<std::int64_t> rejected_tokens;
            MlxDFlash2GenerationStats rejected_stats;
            if (draft.generate(rejecting_target, prompt, greedy, 16, [&](auto token) {
                    rejected_tokens.push_back(token); return true;
                }, {}, 32, std::nullopt, &rejected_stats) != 16 ||
                rejected_tokens != std::vector<std::int64_t>(16, 88) || !rejected_stats.cycles ||
                rejected_stats.accepted != 0 || rejecting_target.cache_position() != 18)
                throw std::runtime_error("DFlash2 rejection changed teacher tokens or retained invalid state");
        }
        for (auto dtype : {float16, bfloat16, float32}) {
            auto q = astype(reshape(sin(arange(8 * 512, float32) * .01f), {1, 8, 512}), dtype);
            auto k = astype(reshape(cos(arange(8 * 128, float32) * .02f), {1, 8, 128}), dtype);
            auto iq = astype(reshape(sin(arange(8 * 192, float32) * .03f), {1, 8, 192}), dtype);
            const auto norm = ones({64}, float32), positions = arange(4096, 4104, int32);
            auto combined = qwen4_qsa_decode_prologue(q, k, iq, norm, norm, norm, positions,
                4, 2, 2, 64, 64, 32, 10000.f, 1e-6f);
            for (int row = 0; row < 8; ++row) {
                auto single = qwen4_qsa_decode_prologue(slice(q, {0, row, 0}, {1, row + 1, 512}),
                    slice(k, {0, row, 0}, {1, row + 1, 128}), slice(iq, {0, row, 0}, {1, row + 1, 192}),
                    norm, norm, norm, slice(positions, {row}, {row + 1}), 4, 2, 2, 64, 64, 32, 10000.f, 1e-6f);
                const std::array<std::pair<array, array>, 4> values{{
                    {slice(combined.query, {0, 0, row, 0}, {1, 4, row + 1, 64}), single.query},
                    {slice(combined.key, {0, 0, row, 0}, {1, 2, row + 1, 64}), single.key},
                    {slice(combined.output_gate, {0, row, 0}, {1, row + 1, 256}), single.output_gate},
                    {slice(combined.index_query, {0, row, 0, 0}, {1, row + 1, 2, 64}), single.index_query}}};
                for (const auto& [actual, expected] : values)
                    if (max(abs(astype(actual, float32) - astype(expected, float32))).item<float>() != 0.f)
                        throw std::runtime_error("QSA M8 prologue differs from independent M1 rows");
            }
        }
        {
            constexpr int dimension = 128, channels = 4 * dimension;
            const auto qk = reshape(sin(arange(8 * 2 * dimension, float32) * .017f) * .05f, {1, 8, 2 * dimension});
            const auto values = reshape(cos(arange(8 * 2 * dimension, float32) * .023f) * .03f, {1, 8, 2 * dimension});
            const auto conv = reshape(sin(arange(3 * channels, float32) * .013f) * .01f, {1, 3, channels});
            const auto state = reshape(cos(arange(2 * dimension * dimension, float32) * .01f) * .001f,
                {1, 2, dimension, dimension});
            const auto weight = full({channels, 1, 4}, .1f, float32);
            const auto gate = full({1, 2, 8}, -.1f, float32), beta = full({1, 2, 8}, .6f, float32);
            const MlxGatedDeltaSpeculativeState transaction{conv, state, qk, values, gate, beta, 19, 1, 1, 8};
            const auto replay = compile_gated_delta_speculative_replay(weight, 1, 2, dimension, dimension,
                std::nullopt, 1e-6f, true);
            for (int accepted = 0; accepted < 8; ++accepted) {
                auto actual = replay(transaction, accepted);
                auto prepared = linear_conv_qkv(conv, slice(qk, {0, 0, 0}, {1, accepted + 1, 2 * dimension}),
                    slice(values, {0, 0, 0}, {1, accepted + 1, 2 * dimension}), weight, 1, 2, dimension, dimension,
                    std::nullopt, 1e-6f);
                auto expected = gated_delta_net(prepared.query, prepared.key, prepared.value,
                    slice(gate, {0, 0, 0}, {1, 2, accepted + 1}), slice(beta, {0, 0, 0}, {1, 2, accepted + 1}), state, true);
                if (actual.position != 20 + accepted ||
                    max(abs(actual.convolution_state - prepared.state)).item<float>() != 0.f ||
                    max(abs(actual.recurrent_state - expected.state)).item<float>() > 2e-4f)
                    throw std::runtime_error("GDN M8 prefix replay differs from independent recurrence");
            }
        }
        std::filesystem::remove_all(directory);
        std::cout << "DFlash2 sampling policy, loader, preparation, ring wrap, context validation and joint selector passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n'; return 1;
    }
}
