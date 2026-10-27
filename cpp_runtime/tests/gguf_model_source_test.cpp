#include "tyloinfer/gguf_model_source.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

void qwen4_source(const std::filesystem::path& directory) {
    auto* tensors = ggml_init({2 << 20, nullptr, false});
    auto* main = gguf_init_empty();
    gguf_set_val_str(main, "general.architecture", "qwen4exp");
    const char* tags[]{"model", "test"};
    gguf_set_arr_str(main, "general.tags", tags, 2);
    for (const auto& pair : std::vector<std::pair<const char*, uint32_t>>{
            {"embedding_length", 256}, {"block_count", 4}, {"context_length", 32768},
            {"attention.head_count", 2}, {"attention.head_count_kv", 1}, {"attention.key_length", 128},
            {"rope.dimension_count", 64}, {"full_attention_interval", 4},
            {"ssm.conv_kernel", 4}, {"ssm.state_size", 128}, {"ssm.group_count", 1}, {"ssm.time_step_rank", 2},
            {"expert_count", 4}, {"expert_used_count", 2}, {"expert_feed_forward_length", 128},
            {"expert_shared_feed_forward_length", 128}, {"hyper_connection.count", 2},
            {"hyper_connection.low_rank", 64}, {"attention.indexer.head_count", 1},
            {"attention.indexer.key_length", 128}, {"attention.indexer.top_k", 256},
            {"ple.ngram_size", 2}, {"ple.heads_per_ngram", 2}, {"ple.conv_kernel", 4}, {"ple.eos_token_id", 1}}) {
        const auto name = std::string("qwen4exp.") + pair.first;
        gguf_set_val_u32(main, name.c_str(), pair.second);
    }
    gguf_set_val_f32(main, "qwen4exp.rope.freq_base", 10000.f);
    const int32_t layers[]{0}, ratios[]{0, 0, 0, 4}, sections[]{16, 16, 0, 0};
    const int64_t multipliers[]{3, 5}, offsets[]{0, 2}, vocabs[]{2, 2};
    gguf_set_arr_data(main, "qwen4exp.ple.layers", GGUF_TYPE_INT32, layers, 1);
    gguf_set_arr_data(main, "qwen4exp.attention.compress_ratios", GGUF_TYPE_INT32, ratios, 4);
    gguf_set_arr_data(main, "qwen4exp.rope.dimension_sections", GGUF_TYPE_INT32, sections, 4);
    gguf_set_arr_data(main, "qwen4exp.ple.layer_multipliers", GGUF_TYPE_INT64, multipliers, 2);
    gguf_set_arr_data(main, "qwen4exp.ple.head_offsets", GGUF_TYPE_INT64, offsets, 2);
    gguf_set_arr_data(main, "qwen4exp.ple.head_vocab_sizes", GGUF_TYPE_INT64, vocabs, 2);
    const auto add = [&](const char* name, int elements, float value) {
        auto* t = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, elements);
        ggml_set_name(t, name);
        for (int i = 0; i < elements; ++i) static_cast<float*>(t->data)[i] = value;
        gguf_add_tensor(main, t);
    };
    add("output_hc_norm.weight", 512, 1.5f);
    add("blk.0.ssm_norm.weight", 128, 1.5f);
    add("blk.0.ssm_a", 2, -2.f);
    add("blk.0.ffn_gate_inp_shexp.weight", 256, .25f);
    auto* down = ggml_new_tensor_3d(tensors, GGML_TYPE_Q2_0, 64, 3, 2);
    ggml_set_name(down, "blk.0.ffn_down_exps.weight");
    std::memset(down->data, 0xe4, ggml_nbytes(down));
    for (std::size_t offset = 0; offset < ggml_nbytes(down); offset += 18) {
        const auto scale = ggml_fp32_to_fp16(.5f);
        std::memcpy(static_cast<uint8_t*>(down->data) + offset, &scale, 2);
    }
    gguf_add_tensor(main, down);
    auto* embedding = ggml_new_tensor_2d(tensors, GGML_TYPE_Q4_K, 256, 3);
    ggml_set_name(embedding, "token_embd.weight");
    std::memset(embedding->data, 0, ggml_nbytes(embedding));
    gguf_add_tensor(main, embedding);
    auto* qkv = ggml_new_tensor_2d(tensors, GGML_TYPE_Q8_0, 256, 512);
    ggml_set_name(qkv, "blk.0.attn_qkv.weight");
    std::memset(qkv->data, 0, ggml_nbytes(qkv));
    gguf_add_tensor(main, qkv);
    const auto main_path = directory / "qwen4.gguf";
    require(gguf_write_to_file(main, main_path.string().c_str(), false), "Qwen4 main write failed");
    auto* companion = gguf_init_empty();
    auto* ple = ggml_new_tensor_2d(tensors, GGML_TYPE_IQ4_NL, 128, 4);
    ggml_set_name(ple, "per_layer_token_embd.weight");
    std::memset(ple->data, 17, ggml_nbytes(ple));
    gguf_add_tensor(companion, ple);
    const auto table_path = directory / "ple.gguf";
    require(gguf_write_to_file(companion, table_path.string().c_str(), false), "Qwen4 PLE write failed");
    gguf_free(main); gguf_free(companion); ggml_free(tensors);
    const mfq::GgufModelSource source(main_path, {table_path});
    const auto* packed_down = source.find_tensor("model.block.0.mlp.experts.down.weight");
    require(packed_down && packed_down->dtype == "GGUF_Q2_0" && packed_down->nbytes == 108 &&
        *packed_down->shape == std::vector<int64_t>({2, 3, 64}), "GGUF Q2_0 expert metadata wrong");
    const auto packed_bytes = source.read(packed_down->name);
    std::vector<float> decoded(384);
    ggml_get_type_traits(GGML_TYPE_Q2_0)->to_float(packed_bytes.data(), decoded.data(), decoded.size());
    for (std::size_t i = 0; i < decoded.size(); ++i)
        require(decoded[i] == .5f * (int(i % 4) - 1), "GGUF Q2_0 code order changed");
    const auto cfg = nlohmann::json::parse(source.model_config_json());
    require(cfg["model_type"] == "qwen4_exp_text" && cfg["hc_count"] == 2 &&
        cfg["indexer_compress_ratio"] == 4 && cfg["ple_layer_ids"] == nlohmann::json({1}), "Qwen4 config translation differs");
    const auto read_float = [&](const char* name) {
        const auto bytes = source.read(name); float value;
        std::memcpy(&value, bytes.data() + 12, 4); return value;
    };
    require(read_float("model.mhc.pre.norm.weight") == .5f, "GGUF GR norm offset was applied twice");
    require(read_float("model.block.0.linear_attention.norm.weight") == 1.5f, "GGUF GDN output norm offset changed");
    require(std::abs(read_float("model.block.0.linear_attention.a") - std::log(2.f)) < 1e-6f, "GGUF GDN decay was exponentiated twice");
    require(*source.find_tensor("model.block.0.mlp.shared_expert.router.weight")->shape == std::vector<int64_t>({1, 256}), "GGUF shared gate lost its row dimension");
    const auto* table = source.find_tensor("model.block.0.position_embedding.ngram.shard.0.weight");
    require(table && *table->shape == std::vector<int64_t>({4, 128}), "GGUF external PLE missing");
    const auto range = source.tensor_file_range(table->name);
    require(range && range->path == table_path && range->nbytes == 288, "GGUF PLE cannot be mapped without copying");
    const auto bytes = source.read("model.block.0.position_embedding.ngram.head_vocab_sizes");
    int64_t vocab = 0; std::memcpy(&vocab, bytes.data() + 12, 8);
    require(vocab == 2, "GGUF PLE hash metadata missing");
}
}

int main() {
    const auto directory = std::filesystem::temp_directory_path() /
        ("tinfer-gguf-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(directory);
    try {
        auto* tensors = ggml_init({1 << 20, nullptr, false});
        auto* gguf = gguf_init_empty();
        gguf_set_val_str(gguf, "general.architecture", "qwen35");
        gguf_set_val_u32(gguf, "qwen35.embedding_length", 256);
        gguf_set_val_u32(gguf, "qwen35.feed_forward_length", 512);
        gguf_set_val_u32(gguf, "qwen35.attention.head_count", 2);
        gguf_set_val_u32(gguf, "qwen35.attention.head_count_kv", 1);
        gguf_set_val_u32(gguf, "qwen35.block_count", 1);
        gguf_set_val_u32(gguf, "qwen35.context_length", 32768);
        gguf_set_val_str(gguf, "tokenizer.ggml.model", "gpt2");
        const char* tokens[]{"a", "b", "c"};
        gguf_set_arr_str(gguf, "tokenizer.ggml.tokens", tokens, 3);
        auto* embedding = ggml_new_tensor_2d(tensors, GGML_TYPE_Q4_K, 256, 3);
        ggml_set_name(embedding, "token_embd.weight");
        std::memset(embedding->data, 17, ggml_nbytes(embedding));
        gguf_add_tensor(gguf, embedding);
        auto* norm = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 256);
        ggml_set_name(norm, "output_norm.weight");
        for (int i = 0; i < 256; ++i) static_cast<float*>(norm->data)[i] = float(i);
        gguf_add_tensor(gguf, norm);
        const auto path = directory / "model.gguf";
        require(gguf_write_to_file(gguf, path.string().c_str(), false), "GGUF write failed");
        gguf_set_val_u16(gguf, "split.count", 2);
        gguf_set_val_u16(gguf, "split.no", 0);
        gguf_set_val_i32(gguf, "split.tensors.count", 3);
        const auto split_path = directory / "model-00001-of-00002.gguf";
        require(gguf_write_to_file(gguf, split_path.string().c_str(), false), "GGUF shard write failed");
        bool missing = false;
        try { mfq::GgufModelSource invalid(split_path); } catch (const std::runtime_error&) { missing = true; }
        require(missing, "incomplete GGUF shard set accepted");
        auto* second = gguf_init_empty();
        gguf_set_val_u16(second, "split.count", 2);
        gguf_set_val_u16(second, "split.no", 1);
        auto* extra = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 256);
        ggml_set_name(extra, "blk.0.attn_norm.weight");
        std::memset(extra->data, 0, ggml_nbytes(extra));
        gguf_add_tensor(second, extra);
        const auto second_path = directory / "model-00002-of-00002.gguf";
        require(gguf_write_to_file(second, second_path.string().c_str(), false), "GGUF second shard write failed");
        const mfq::GgufModelSource split(second_path);
        require(split.source_paths().size() == 2 && split.tensors().size() == 3 &&
            split.find_tensor("model.block.0.attention.norm.weight"), "GGUF shard discovery failed");
        gguf_set_val_u16(second, "split.no", 0);
        require(gguf_write_to_file(second, second_path.string().c_str(), false), "GGUF invalid shard write failed");
        bool mismatched = false;
        try { mfq::GgufModelSource invalid(split_path); } catch (const std::runtime_error&) { mismatched = true; }
        require(mismatched, "GGUF shard number mismatch accepted");
        gguf_free(second);
        gguf_free(gguf); ggml_free(tensors);
        auto source = mfq::open_model_source(path);
        require(source->architecture() == "qwen35", "architecture lost");
        const auto* table = source->find_tensor("model.token_embedding.weight");
        require(table && table->dtype == "GGUF_Q4_K" && *table->shape == std::vector<int64_t>({3, 256}), "GGUF quantized metadata wrong");
        const auto data = source->read(table->name);
        require(data.size() == 3 * 144 && data.front() == std::byte{17}, "packed GGUF data changed");
        const auto dense = source->read("model.output_norm.weight");
        require(dense.size() == 12 + 256 * 4 && dense[0] == std::byte{1}, "dense prefix missing");
        float end = 0;
        std::memcpy(&end, dense.data() + dense.size() - 4, 4);
        require(end == 255, "dense GGUF data changed");
        const auto config = nlohmann::json::parse(source->model_config_json());
        require(config["vocab_size"] == 3 && config["tie_word_embeddings"] == true, "GGUF config translation wrong");
        auto tokenizer = source->read_asset(mfq::kTokenizerGgufAsset);
        auto* tok = gguf_init_from_buffer(tokenizer.data(), tokenizer.size(), {true, nullptr});
        require(tok && gguf_get_n_tensors(tok) == 0 && gguf_find_key(tok, "tokenizer.ggml.tokens") >= 0, "GGUF tokenizer extraction wrong");
        gguf_free(tok);
        bool bounded = false;
        try { source->read(table->name, table->nbytes, 1); } catch (const std::out_of_range&) { bounded = true; }
        require(bounded, "out-of-range GGUF read accepted");
        const auto reader = source->tensor_reader(table->name);
        source.reset();
        std::byte value{};
        reader(0, &value, 1);
        require(value == std::byte{17}, "retained GGUF reader expired");
        std::filesystem::resize_file(path, 80);
        bool truncated = false;
        try { mfq::GgufModelSource invalid(path); } catch (const std::runtime_error&) { truncated = true; }
        require(truncated, "truncated GGUF accepted");
        qwen4_source(directory);
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n'; return 1;
    }
}
