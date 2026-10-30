#include "mlx_deepseek_v41_causal_lm.h"
#include "gguf.h"
#include "ggml-quants.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
using namespace mfq::metal;
using namespace mlx::core;
using nlohmann::json;

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

json config() {
    return {{"model_type", "deepseek_v41"}, {"eos_token_id", 1}, {"image_token_id", 255},
        {"quantization_config", {{"quant_method", "fp8"}, {"weight_block_size", {32, 32}},
            {"scale_fmt", "ue8m0"}, {"expert_dtype", "fp4"}}},
        {"text_config", {
            {"model_type", "deepseek_v41_text"}, {"vocab_size", 256}, {"hidden_size", 256},
            {"moe_intermediate_size", 256}, {"num_hidden_layers", 2}, {"num_attention_heads", 2},
            {"num_key_value_heads", 1}, {"head_dim", 512}, {"qk_rope_head_dim", 64},
            {"q_lora_rank", 256}, {"o_lora_rank", 128}, {"o_groups", 2}, {"hidden_act", "silu"},
            {"rms_norm_eps", 1e-6}, {"max_position_embeddings", 256}, {"rope_theta", 10000},
            {"rope_scaling", {{"rope_type", "yarn"}, {"factor", 1}, {"original_max_position_embeddings", 256}}},
            {"n_routed_experts", 4}, {"n_shared_experts", 1}, {"num_experts_per_tok", 2},
            {"scoring_func", "sqrtsoftplus"}, {"topk_method", "noaux_tc"}, {"norm_topk_prob", true},
            {"routed_scaling_factor", 1.5}, {"swiglu_limit", .7}, {"sliding_window", 16},
            {"compress_ratios", {2, 1, 0}}, {"compress_rope_theta", 160000},
            {"kv_source_layer_ids", {0, 1}}, {"index_source_layer_ids", {0, 1}},
            {"index_n_heads", 2}, {"index_head_dim", 128}, {"index_topk", 16},
            {"candidate_source_layer_id", 1}, {"candidate_topk_blocks", 16}, {"candidate_block_size", 8},
            {"hc_mult", 4}, {"hc_sinkhorn_iters", 20}, {"hc_eps", 1e-6},
            {"engram_layer_ids", {0}}, {"engram_num_embeddings", {8}}, {"engram_max_ngram_size", 2},
            {"engram_vocab_size", 3}, {"engram_n_heads", 2}, {"engram_head_dim", 256},
            {"engram_pad_token_id", 2}, {"engram_compressed_vocab_size", 2},
            {"num_nextn_predict_layers", 1}, {"dspark_block_size", 5}, {"dspark_noise_token_id", 0},
            {"dspark_target_layer_ids", {1}}, {"dspark_markov_rank", 16},
            {"dspark_n_routed_experts", 4}, {"dspark_num_experts_per_tok", 2}}},
        {"vision_config", {{"model_type", "deepseek_v41_vision"}, {"num_hidden_layers", 1},
            {"hidden_size", 8}, {"num_attention_heads", 1}, {"intermediate_size", 16}, {"patch_size", 2},
            {"downsample_ratio", 1}, {"max_image_tokens", 16}, {"min_pixels", 16}}}};
}

void write_fixture(const std::filesystem::path& path, const std::filesystem::path& table_path) {
    auto* context = ggml_init({32 << 20, nullptr, false});
    auto* file = gguf_init_empty();
    auto* table = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "deepseek41");
    gguf_set_val_str(file, "deepseek41.config", config().dump().c_str());
    gguf_set_val_str(file, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    const uint32_t layers[]{0}, rows[]{8}, primes[]{3, 5};
    std::vector<uint32_t> tokens(256);
    for (int i = 0; i < 256; ++i) tokens[i] = i % 2;
    const uint64_t multipliers[]{3, 5};
    gguf_set_arr_data(file, "deepseek41.engram.layer_ids", GGUF_TYPE_UINT32, layers, 1);
    gguf_set_arr_data(file, "deepseek41.engram.rows", GGUF_TYPE_UINT32, rows, 1);
    gguf_set_arr_data(file, "deepseek41.engram.primes", GGUF_TYPE_UINT32, primes, 2);
    gguf_set_arr_data(file, "deepseek41.engram.token_map", GGUF_TYPE_UINT32, tokens.data(), tokens.size());
    gguf_set_arr_data(file, "deepseek41.engram.multipliers", GGUF_TYPE_UINT64, multipliers, 2);
    gguf_set_val_u32(file, "deepseek41.engram.compressed_vocab_size", 2);
    gguf_set_val_u32(file, "deepseek41.engram.pad_id", 0);
    const auto add = [&](const std::string& name, int n, int k, ggml_type type, bool norm = false) {
        auto* t = k ? ggml_new_tensor_2d(context, type, k, n) : ggml_new_tensor_1d(context, type, n);
        ggml_set_name(t, name.c_str());
        std::vector<float> values(ggml_nelements(t));
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = norm ? 1.f : .02f * std::sin(i * .017f + name.size());
        if (type == GGML_TYPE_F32) std::memcpy(t->data, values.data(), values.size() * 4);
        else if (type == GGML_TYPE_F16) for (std::size_t i = 0; i < values.size(); ++i)
            static_cast<uint16_t*>(t->data)[i] = ggml_fp32_to_fp16(values[i]);
        else quantize_row_q8_0_ref(values.data(), static_cast<block_q8_0*>(t->data), values.size());
        gguf_add_tensor(file, t);
    };
    add("token_embd.weight", 256, 256, GGML_TYPE_F16);
    add("output.weight", 256, 256, GGML_TYPE_Q8_0);
    add("output_norm.weight", 256, 0, GGML_TYPE_F32, true);
    for (int layer = 0; layer < 2; ++layer) {
        const auto root = "blk." + std::to_string(layer) + ".";
        for (const auto* site : {"attn", "ffn"}) {
            add(root + site + "_norm.weight", 256, 0, GGML_TYPE_F32, true);
            add(root + "hc_" + site + "_fn.weight", 24, 1024, GGML_TYPE_F16);
            add(root + "hc_" + site + "_base.weight", 24, 0, GGML_TYPE_F32);
            add(root + "hc_" + site + "_scale.weight", 3, 0, GGML_TYPE_F32);
        }
        add(root + "attn_sinks.weight", 2, 0, GGML_TYPE_F32);
        add(root + "attn_q_a.weight", 256, 256, GGML_TYPE_Q8_0);
        add(root + "attn_q_b.weight", 1024, 256, GGML_TYPE_Q8_0);
        add(root + "attn_q_a_norm.weight", 256, 0, GGML_TYPE_F32, true);
        add(root + "attn_kv.weight", 512, 256, GGML_TYPE_Q8_0);
        add(root + "attn_kv_a_norm.weight", 512, 0, GGML_TYPE_F32, true);
        add(root + "attn_output_a.weight", 256, 512, GGML_TYPE_Q8_0);
        add(root + "attn_output_b.weight", 256, 256, GGML_TYPE_Q8_0);
        add(root + "attn_compressor_kv.weight", 512, 256, GGML_TYPE_F16);
        add(root + "attn_compressor_norm.weight", 512, 0, GGML_TYPE_F32, true);
        if (!layer) add(root + "attn_compressor_gate.weight", 512, 256, GGML_TYPE_F16);
        add(root + "indexer.attn_q_b.weight", 256, 256, GGML_TYPE_F16);
        add(root + "indexer.attn_k.weight", 128, 512, GGML_TYPE_F16);
        add(root + "indexer.k_norm.weight", 128, 0, GGML_TYPE_F32, true);
        add(root + "indexer.proj.weight", 2, 256, GGML_TYPE_F16);
        add(root + "ffn_gate_inp.weight", 4, 256, GGML_TYPE_F32);
        add(root + "exp_probs_b.bias", 4, 0, GGML_TYPE_F32);
        add(root + "exp_probs_b_vl.bias", 4, 0, GGML_TYPE_F32);
        const std::vector<int32_t> gu{143, 42, 39, 11}, down{42, 39, 11, 143};
        for (const auto* role : {"gate", "up", "down"}) {
            add(root + "ffn_" + role + "_shexp.weight", 256, 256, GGML_TYPE_Q8_0);
            const auto& types = std::string_view(role) == "down" ? down : gu;
            for (const auto type : {GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_MXFP4, GGML_TYPE_Q3_K}) {
                const auto count = std::count(types.begin(), types.end(), int32_t(type));
                auto* t = ggml_new_tensor_3d(context, type, 256, 256, count);
                auto suffix = std::string(ggml_type_name(type));
                for (auto& c : suffix) c = std::tolower(static_cast<unsigned char>(c));
                ggml_set_name(t, (root + "ffn_" + role + "_exps." + suffix + ".weight").c_str());
                auto* bytes = static_cast<uint8_t*>(t->data);
                for (std::size_t i = 0; i < ggml_nbytes(t); ++i) bytes[i] = uint8_t(i * 17 + 6);
                const auto scale = ggml_fp32_to_fp16(.001f);
                const auto at = type == GGML_TYPE_PTQ1_0 ? 26 : type == GGML_TYPE_Q3_K ? 108 : 0;
                for (std::size_t b = 0; b < ggml_nbytes(t); b += ggml_type_size(type)) {
                    if (type == GGML_TYPE_MXFP4) bytes[b] = 117;
                    else std::memcpy(bytes + b + at, &scale, 2);
                }
                gguf_add_tensor(file, t);
            }
        }
        for (const auto* role : {"gu", "down"}) {
            auto* t = ggml_new_tensor_1d(context, GGML_TYPE_I32, 4);
            ggml_set_name(t, (root + "ffn_exps_" + role + "_type").c_str());
            const auto& types = std::string_view(role) == "gu" ? gu : down;
            std::memcpy(t->data, types.data(), 16); gguf_add_tensor(file, t);
        }
    }
    add("blk.0.engram_q_norm.weight", 4, 256, GGML_TYPE_F32, true);
    add("blk.0.engram_k_norm.weight", 4, 256, GGML_TYPE_F32, true);
    add("blk.0.engram_kv.weight", 1280, 512, GGML_TYPE_F16);
    auto* engram = ggml_new_tensor_2d(context, GGML_TYPE_I8, 264, 8);
    ggml_set_name(engram, "blk.0.engram_embd.weight");
    for (int row = 0; row < 8; ++row) {
        auto* bytes = static_cast<uint8_t*>(engram->data) + row * 264;
        std::memset(bytes, row % 2 ? 0x40 : 0x38, 256); std::memset(bytes + 256, 123, 8);
    }
    gguf_add_tensor(table, engram);
    for (auto* shard : {file, table}) gguf_set_val_u16(shard, "split.count", 2);
    gguf_set_val_u16(file, "split.no", 0); gguf_set_val_u16(table, "split.no", 1);
    const auto count = gguf_get_n_tensors(file) + gguf_get_n_tensors(table);
    for (auto* shard : {file, table}) gguf_set_val_i32(shard, "split.tensors.count", int32_t(count));
    const bool ok = gguf_write_to_file(file, path.string().c_str(), false) &&
        gguf_write_to_file(table, table_path.string().c_str(), false);
    gguf_set_val_u16(file, "split.count", 1);
    gguf_set_val_i32(file, "split.tensors.count", int32_t(gguf_get_n_tensors(file)));
    require(gguf_write_to_file(file, (path.parent_path() / "external.gguf").string().c_str(), false),
        "DeepSeek external Engram fixture write failed");
    gguf_free(file); gguf_free(table); ggml_free(context);
    require(ok, "DeepSeek GGUF model fixture write failed");
    const auto directory = path.parent_path() / "external-tables";
    std::filesystem::create_directory(directory);
    const json header{{"layers.0.engram.embed.weight", {{"dtype", "F8_E4M3"}, {"shape", {8, 256}}, {"data_offsets", {0, 2048}}}},
        {"layers.0.engram.embed.scale", {{"dtype", "F8_E8M0"}, {"shape", {8, 8}}, {"data_offsets", {2048, 2112}}}}};
    auto encoded = header.dump();
    encoded.append((8 - encoded.size() % 8) % 8, ' ');
    std::ofstream output(directory / "engram.safetensors", std::ios::binary);
    const uint64_t length = encoded.size();
    for (unsigned i = 0; i < 8; ++i) output.put(char(length >> (8 * i)));
    output.write(encoded.data(), encoded.size());
    for (int row = 0; row < 8; ++row) for (int col = 0; col < 256; ++col) output.put(row % 2 ? 0x40 : 0x38);
    for (int i = 0; i < 64; ++i) output.put(123);
    output.close();
    std::ofstream index(directory / "model.safetensors.index.json");
    index << json{{"weight_map", {{"layers.0.engram.embed.weight", "engram.safetensors"},
        {"layers.0.engram.embed.scale", "engram.safetensors"}, {"layers.0.ffn.gate.weight", "not-downloaded.safetensors"}}}};
}

void close(array a, array b) {
    require(a.shape() == b.shape(), "DeepSeek GGUF output shapes differ");
    const auto error = max(abs(astype(a, float32) - astype(b, float32))).item<float>();
    const auto scale = max(abs(astype(b, float32))).item<float>();
    require(std::isfinite(error) && error <= .002f * std::max(1.f, scale), "DeepSeek GGUF resident/SSD output differs");
}
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / ("tinfer-xq-" +
        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        set_default_device(Device::gpu);
        std::filesystem::create_directories(root);
        const auto path = root / "model-00001-of-00002.gguf", table = root / "model-00002-of-00002.gguf";
        write_fixture(path, table);
        const MfqContainer source(path);
        require(source.source_paths().size() == 2, "DeepSeek GGUF Engram shard was not discovered");
        auto resident = MlxDeepseekV41CausalLm::load(source, 128);
        auto streamed = MlxDeepseekV41CausalLm::load(source, 128, 256 << 10);
        const MfqContainer external_source(root / "external.gguf", {}, {root / "external-tables"});
        require(external_source.record("model.block.0.associative_memory.embedding.weight").dtype == "MXFP8" &&
            external_source.source_paths().size() == 2, "external Engram was not kept in its native FP8 layout");
        auto external = MlxDeepseekV41CausalLm::load(external_source, 128, 256 << 10);
        bool rejected = false;
        try { MfqContainer duplicate(path, {}, {root / "external-tables"}); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "external Engram replaced an embedded table");
        std::ofstream(root / "external-tables" / "config.json") << config();
        std::ifstream original(root / "external.gguf", std::ios::binary);
        const std::vector<char> bytes((std::istreambuf_iterator<char>(original)), std::istreambuf_iterator<char>());
        const std::size_t boundaries[]{0, 37, bytes.size() / 2 + 3, bytes.size()};
        std::filesystem::path selected;
        for (int part = 1; part <= 3; ++part) {
            selected = root / ("external.gguf.part-0" + std::to_string(part) + "-of-03");
            std::ofstream output(selected, std::ios::binary);
            output.write(bytes.data() + boundaries[part - 1], boundaries[part] - boundaries[part - 1]);
        }
        const MfqContainer configured_source(selected, {}, {root / "external-tables"});
        require(configured_source.source_paths().size() == 4, "raw XQ parts or external Engram path lost");
        auto configured = MlxDeepseekV41CausalLm::load(configured_source, 128, 256 << 10);
        auto multipart_resident = MlxDeepseekV41CausalLm::load(configured_source, 128);
        require(!resident.supports_multimodal() && !resident.supports_mtp(), "text-only GGUF advertises missing towers");
        const auto ids = arange(2, 18, int32);
        const auto first = resident.forward(ids);
        close(first, streamed.forward(ids));
        close(first, external.forward(ids));
        close(first, configured.forward(ids));
        close(first, multipart_resident.forward(ids));
        require(streamed.expert_cache_limit_bytes() == 256 << 10 && streamed.ssd_expert_cache_stats()->bytes_read > 0,
            "DeepSeek GGUF bypassed the expert cache");
        const std::vector<int64_t> tokens{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
        const auto snapshot = streamed.capture_text_session_state(tokens);
        const auto next = array({18}, int32);
        const auto expected = resident.forward(next, true);
        close(streamed.forward(next, true), expected);
        close(external.forward(next, true), expected);
        close(configured.forward(next, true), expected);
        close(multipart_resident.forward(next, true), expected);
        require(external.engram_ssd_stats()->rows_loaded > 0, "external Engram bypassed native row reads");
        streamed.restore_text_session_state(snapshot);
        close(streamed.forward(next, true), expected);
        require(streamed.engram_ssd_stats()->rows_loaded > 0, "DeepSeek GGUF did not stream Engram rows");
        const auto released = resident.set_expert_cache_limit(256 << 10);
        require(resident.resident_full_expert_bytes() == 0 && resident.expert_cache_limit_bytes() == 256 << 10,
            "DeepSeek GGUF dynamic budget retained resident experts");
        close(resident.forward(array({19}, int32), true), streamed.forward(array({19}, int32), true));
        close(resident.prefill(ids, 8), streamed.prefill(ids, 8));
        close(resident.decode(next), streamed.decode(next));
        const auto long_ids = arange(2, 67, int32);
        const auto long_expected = multipart_resident.prefill(long_ids, 65);
        close(long_expected, streamed.prefill(long_ids, 65));
        close(long_expected, configured.prefill(long_ids, 65));
        const auto continuation = array({67}, int32);
        const auto continued = multipart_resident.decode(continuation);
        close(continued, streamed.decode(continuation));
        close(continued, configured.decode(continuation));
        (void)released;
        std::filesystem::remove_all(root);
        std::cout << "DeepSeek XQ full graph, resident/SSD, Engram and session restore passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::filesystem::remove_all(root);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
