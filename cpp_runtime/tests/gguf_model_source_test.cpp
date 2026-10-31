#include "tyloinfer/gguf_model_source.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <map>
#include <sstream>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct MirrorSetting {
    std::string previous;bool present=false;
    explicit MirrorSetting(const std::filesystem::path& directory) {
        if(const auto* value=std::getenv("TYLOINFER_MODEL_MIRROR_DIRS")){present=true;previous=value;}
        set(directory.string().c_str());
    }
    static void set(const char* value) {
#if defined(_WIN32)
        _putenv_s("TYLOINFER_MODEL_MIRROR_DIRS",value?value:"");
#else
        if(value)setenv("TYLOINFER_MODEL_MIRROR_DIRS",value,1);else unsetenv("TYLOINFER_MODEL_MIRROR_DIRS");
#endif
    }
    ~MirrorSetting(){set(present?previous.c_str():nullptr);}
};


void raw_part_source(const std::filesystem::path& path, const std::vector<std::filesystem::path>& companions = {},
        std::size_t first_cut = 37, int count = 4, int width = 2) {
    std::ifstream input(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const mfq::GgufModelSource complete(path, companions);
    std::vector<std::size_t> cuts{0, first_cut, bytes.size() / 2 + 3, bytes.size() - 7, bytes.size()};
    if (count != 4) {
        cuts = {0, first_cut};
        for (int i = 2; i <= count; ++i)
            cuts.push_back(first_cut + (bytes.size() - first_cut) * (i - 1) / (count - 1));
    }
    std::vector<std::filesystem::path> parts;
    for (std::size_t i = 1; i < cuts.size(); ++i) {
        std::ostringstream name;
        name << path.filename().string() << ".part-" << std::setfill('0') << std::setw(width) << i << "-of-" << std::setw(width) << count;
        parts.push_back(path.parent_path() / name.str());
        std::ofstream output(parts.back(), std::ios::binary);
        output.write(bytes.data() + cuts[i - 1], cuts[i] - cuts[i - 1]);
    }
    require(!mfq::is_gguf_file(path.parent_path() / "bad.gguf.part-00-of-04"), "invalid raw part number was recognized");
    for (const auto& selected : {parts.front(), parts.back()}) {
        require(mfq::is_gguf_file(selected), "noninitial raw GGUF part not recognized");
        const std::shared_ptr<const mfq::ModelSource> source = companions.empty() ? mfq::open_model_source(selected)
            : std::make_shared<mfq::GgufModelSource>(selected, companions);
        require(source->architecture() == complete.architecture() && source->metadata() == complete.metadata() &&
            source->tensors().size() == complete.tensors().size(), "raw parts changed model metadata");
        require(source->source_paths().size() == parts.size() + companions.size(), "raw parts source paths missing");
        for (const auto& tensor : complete.tensors()) {
            const auto expected = complete.read(tensor.name);
            require(source->read(tensor.name) == expected, "raw part tensor bytes changed");
            const auto original_range = complete.tensor_file_range(tensor.name);
            if (original_range && original_range->path == path)
                require(!source->tensor_file_range(tensor.name), "raw part tensor exposed a false contiguous file range");
            std::vector<std::byte> actual(expected.size()), tail(13);
            source->tensor_batch_reader(tensor.name)({{uint64_t(expected.size() - 13), tail.data(), tail.size()},
                {0, actual.data(), actual.size()}, {tensor.nbytes, nullptr, 0}});
            require(actual == expected && std::equal(tail.begin(), tail.end(), expected.end() - 13), "raw part batch read changed bytes");
        }
        for (const auto& asset : complete.assets())
            require(source->read_asset(asset) == complete.read_asset(asset), "raw parts changed tokenizer or assets");
    }
    mfq::ModelSource::TensorReader retained;
    std::vector<std::byte> expected;
    {
        const mfq::GgufModelSource source(parts.back(), companions);
        const auto name = source.tensors().front().name;
        expected = source.read(name);
        retained = source.tensor_reader(name);
    }
    std::vector<std::byte> actual(expected.size());
    retained(0, actual.data(), actual.size());
    require(actual == expected, "retained raw part reader expired");
    const auto hidden = parts[1].string() + ".missing";
    std::filesystem::rename(parts[1], hidden);
    retained(0, actual.data(), actual.size());
    require(actual == expected, "retained raw part reader changed after rename");
    bool missing = false;
    try { mfq::GgufModelSource invalid(parts.front(), companions); } catch (const std::runtime_error&) { missing = true; }
    require(missing, "incomplete raw GGUF parts accepted");
    std::filesystem::rename(hidden, parts[1]);
    std::filesystem::resize_file(parts[2], 1);
    bool truncated = false;
    try { mfq::GgufModelSource invalid(parts.front(), companions); } catch (const std::runtime_error&) { truncated = true; }
    require(truncated, "truncated raw GGUF payload accepted");
}

void gyro_layout() {
    static const uint16_t codebook[]{
#include "../third_party/agention-gyro/codebook_f16.inc"
    };
    for (const auto type : {GGML_TYPE_TQ2_T, GGML_TYPE_TQK4, GGML_TYPE_TQK5, GGML_TYPE_TQK6,
            GGML_TYPE_TQK7, GGML_TYPE_TQP5, GGML_TYPE_TQP6, GGML_TYPE_TQP7}) {
        const int bits = type == GGML_TYPE_TQ2_T ? 8 : type == GGML_TYPE_TQK4 ? 4 :
            type == GGML_TYPE_TQK5 || type == GGML_TYPE_TQP5 ? 5 : type == GGML_TYPE_TQK6 || type == GGML_TYPE_TQP6 ? 6 : 7;
        const int steps = type >= GGML_TYPE_TQP5 && type <= GGML_TYPE_TQP7 ? 64 : 32;
        const int length = steps * bits / 8;
        require(ggml_type_size(type) == std::size_t(length + 2) && ggml_blck_size(type) == 128,
            "Gyro block layout differs from its specification");
        std::vector<uint8_t> bytes((length + 2) * 2);
        std::vector<float> expected(256), actual(256);
        for (int b = 0; b < 2; ++b) {
            const float scale = b ? -2.f : .5f;
            const auto half = ggml_fp32_to_fp16(scale);
            auto* block = bytes.data() + b * (length + 2);
            std::memcpy(block, &half, 2);
            for (int j = 0; j < length; ++j) block[2 + j] = uint8_t(j * 37 + b * 19 + 3);
            for (int t = 0; t < steps; ++t) {
                uint32_t state = 0;
                for (int bit = 0; bit < 16; ++bit) {
                    const int at = type == GGML_TYPE_TQ2_T ? ((t * 8 + bit) % (length * 8))
                        : (((steps - 1 - t) * bits + bit) % (length * 8));
                    const int index = type == GGML_TYPE_TQ2_T ?
                        (bit < 8 ? 2 + t : 2 + (t + 31) % 32) : 2 + at / 8;
                    state |= uint32_t((block[index] >> (at % 8)) & 1) << bit;
                }
                const uint32_t hash = (state & 32767u) * uint32_t(0x9e3779b1);
                for (int j = 0; j < 128 / steps; ++j) {
                    const auto index = j < 2 ? hash >> 21 : (hash >> 10) & 2047;
                    expected[b * 128 + t * (128 / steps) + j] = scale * ggml_fp16_to_fp32(codebook[index * 2 + j % 2]);
                }
            }
        }
        ggml_get_type_traits(type)->to_float(bytes.data(), actual.data(), actual.size());
        require(actual == expected, "Gyro circular stream or codebook indexing differs");
        require(ggml_validate_row_data(type, bytes.data(), bytes.size()), "valid Gyro payload rejected");
    }
    std::vector<uint8_t> example{0, 0x3c, 7, 0x18, 0x29, 0x3a, 0x4b, 0x5c, 0x6d, 0x7e,
        0x8f, 0xa0, 0xb1, 0xc2, 0xd3, 0xe4, 0xf5, 6};
    std::vector<float> decoded(128);
    ggml_get_type_traits(GGML_TYPE_TQK4)->to_float(example.data(), decoded.data(), decoded.size());
    require(decoded[0] == .98681640625f && decoded[1] == .455322265625f && decoded[2] == -.2119140625f &&
        decoded[3] == .65380859375f, "Gyro normative TQK4 worked example differs");
}

void ptq1_layout() {
    std::vector<uint8_t> q2(36, 0xe4);
    const auto q2_scale = ggml_fp32_to_fp16(.5f);
    std::memcpy(q2.data(), &q2_scale, 2);
    std::memcpy(q2.data() + 18, &q2_scale, 2);
    require(ggml_validate_row_data(GGML_TYPE_Q2_0, q2.data(), q2.size()), "Q2_0 valid payload rejected");
    require(!ggml_validate_row_data(GGML_TYPE_Q2_0, q2.data(), q2.size() - 1), "Q2_0 partial block accepted");
    const uint16_t nan_scale = 0x7e00;
    std::memcpy(q2.data() + 18, &nan_scale, 2);
    require(!ggml_validate_row_data(GGML_TYPE_Q2_0, q2.data(), q2.size()), "Q2_0 NaN scale accepted");
    constexpr int blocks = 243;
    std::vector<uint8_t> bytes(blocks * 28);
    std::vector<float> expected(blocks * 128), decoded(expected.size());
    for (int b = 0; b < blocks; ++b) {
        const float scale = b % 3 == 0 ? 0.f : b % 3 == 1 ? .5f : -1.5f;
        const auto half = ggml_fp32_to_fp16(scale);
        std::memcpy(bytes.data() + b * 28 + 26, &half, 2);
        for (int byte = 0; byte < 26; ++byte) {
            int code = byte < 24 ? (b + byte * 17) % 243 : ((b + byte * 17) % 81) * 3;
            bytes[b * 28 + byte] = uint8_t((code * 256 + 242) / 243);
            const int width = byte < 16 ? 16 : byte < 24 ? 8 : 2;
            const int base = byte < 16 ? byte : byte < 24 ? 80 + byte - 16 : 120 + byte - 24;
            for (int digit = 4; digit >= 0; --digit) {
                const int trit = code % 3;
                code /= 3;
                if (byte < 24 || digit < 4)
                    expected[b * 128 + base + digit * width] = scale * (trit - 1);
            }
        }
    }
    ggml_get_type_traits(GGML_TYPE_PTQ1_0)->to_float(bytes.data(), decoded.data(), decoded.size());
    require(decoded == expected, "PTQ1_0 ternary packing differs from canonical codes");
    require(ggml_validate_row_data(GGML_TYPE_PTQ1_0, bytes.data(), bytes.size()), "PTQ1_0 valid payload rejected");
    require(!ggml_validate_row_data(GGML_TYPE_PTQ1_0, bytes.data(), bytes.size() - 1), "PTQ1_0 partial block accepted");
    const uint16_t invalid = 0x7c00;
    std::memcpy(bytes.data() + 26, &invalid, 2);
    require(!ggml_validate_row_data(GGML_TYPE_PTQ1_0, bytes.data(), bytes.size()), "PTQ1_0 infinite scale accepted");
    require(!ggml_validate_row_data(static_cast<ggml_type>(142), bytes.data(), 28), "undefined GGUF type accepted");
}

void deepseek_source(const std::filesystem::path& directory) {
    using nlohmann::json;
    auto* tensors = ggml_init({1 << 20, nullptr, false});
    auto* file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "deepseek41");
    const json config{{"model_type", "deepseek_v41"}, {"text_config", {
        {"model_type", "deepseek_v41_text"}, {"hidden_size", 256}, {"vocab_size", 3},
        {"num_hidden_layers", 2}, {"engram_layer_ids", {1}}, {"engram_num_embeddings", {8}},
        {"engram_max_ngram_size", 2}, {"engram_n_heads", 2}, {"engram_head_dim", 256},
        {"engram_compressed_vocab_size", 2}, {"engram_pad_token_id", 0}}}};
    gguf_set_val_str(file, "deepseek41.config", config.dump().c_str());
    gguf_set_val_str(file, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    const uint32_t layers[]{1}, rows[]{8}, primes[]{3, 5}, tokens[]{0, 1, 0};
    const uint64_t multipliers[]{3, 5};
    gguf_set_arr_data(file, "deepseek41.engram.layer_ids", GGUF_TYPE_UINT32, layers, 1);
    gguf_set_arr_data(file, "deepseek41.engram.rows", GGUF_TYPE_UINT32, rows, 1);
    gguf_set_arr_data(file, "deepseek41.engram.primes", GGUF_TYPE_UINT32, primes, 2);
    gguf_set_arr_data(file, "deepseek41.engram.token_map", GGUF_TYPE_UINT32, tokens, 3);
    gguf_set_arr_data(file, "deepseek41.engram.multipliers", GGUF_TYPE_UINT64, multipliers, 2);
    gguf_set_val_u32(file, "deepseek41.engram.compressed_vocab_size", 2);
    gguf_set_val_u32(file, "deepseek41.engram.pad_id", 0);
    auto* embedding = ggml_new_tensor_2d(tensors, GGML_TYPE_F16, 256, 3);
    ggml_set_name(embedding, "token_embd.weight");
    std::memset(embedding->data, 0, ggml_nbytes(embedding));
    gguf_add_tensor(file, embedding);
    auto* engram = ggml_new_tensor_2d(tensors, GGML_TYPE_I8, 264, 8);
    ggml_set_name(engram, "blk.1.engram_embd.weight");
    for (std::size_t i = 0; i < ggml_nbytes(engram); ++i) static_cast<uint8_t*>(engram->data)[i] = uint8_t(i);
    gguf_add_tensor(file, engram);
    for (const auto& [stored, canonical] : std::map<std::string, std::string>{
        {"hc_attn_fn.weight", "attention.mhc.pre.function"}, {"hc_ffn_base.weight", "mlp.mhc.pre.base"},
        {"attn_q_a.weight", "attention.query_a.weight"}, {"attn_kv.weight", "attention.key_value.weight"},
        {"attn_compressor_kv.weight", "attention.compressor.key_value.weight"},
        {"indexer.attn_q_b.weight", "attention.indexer.query.weight"},
        {"indexer.attn_k.weight", "attention.indexer.key.weight"}, {"indexer.proj.weight", "attention.indexer.score.weight"},
        {"engram_q_norm.weight", "associative_memory.query.weight"},
        {"engram_kv.weight", "associative_memory.projection.weight"},
        {"exp_probs_b.bias", "mlp.router.bias"}, {"exp_probs_b_vl.bias", "mlp.router.vision_bias"}}) {
        auto* tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 32, 2);
        ggml_set_name(tensor, ("blk.1." + stored).c_str());
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
        gguf_add_tensor(file, tensor);
    }
    const auto path = directory / "deepseek.gguf";
    require(gguf_write_to_file(file, path.string().c_str(), false), "DeepSeek fixture write failed");
    const mfq::GgufModelSource source(path);
    require(json::parse(source.model_config_json()) == config, "DeepSeek embedded config changed");
    const auto name = "model.block.1.associative_memory.embedding.weight";
    const auto* table = source.find_tensor(name);
    require(table && table->dtype == "GGUF_ENGRAM_MXFP8" && table->nbytes == 8 * 264 &&
        *table->shape == std::vector<int64_t>({8, 256}), "DeepSeek Engram wire layout changed");
    require(source.tensor_file_range(name).has_value() &&
        source.read(name, 263, 3) == std::vector<std::byte>({std::byte{7}, std::byte{8}, std::byte{9}}),
        "DeepSeek Engram row boundary or file mapping changed");
    for (const auto* suffix : {"attention.mhc.pre.function", "mlp.mhc.pre.base", "attention.query_a.weight",
        "attention.key_value.weight", "attention.compressor.key_value.weight", "attention.indexer.query.weight",
        "attention.indexer.key.weight", "attention.indexer.score.weight", "associative_memory.query.weight",
        "associative_memory.projection.weight", "mlp.router.bias", "mlp.router.vision_bias"})
        require(source.find_tensor("model.block.1." + std::string(suffix)), "DeepSeek tensor alias missing");
    const auto hash = source.read_asset("__mfq_asset__/deepseek-v41-engram-v1.bin");
    require(hash.size() == 104 && std::memcmp(hash.data(), "MFQENGR1", 8) == 0,
        "DeepSeek Engram hash asset layout differs");
    const auto read_u64 = [&](std::size_t at) { uint64_t v; std::memcpy(&v, hash.data() + at, 8); return v; };
    require(read_u64(36) == 8 && read_u64(44) == 3 && read_u64(52) == 5 &&
        read_u64(60) == 0 && read_u64(68) == 3 && read_u64(76) == 3 && read_u64(84) == 5,
        "DeepSeek Engram bucket offsets or multipliers differ");
    const auto rejects = [&] {
        require(gguf_write_to_file(file, path.string().c_str(), false), "invalid DeepSeek fixture write failed");
        bool rejected = false;
        try { mfq::GgufModelSource invalid(path); } catch (const std::exception&) { rejected = true; }
        require(rejected, "invalid DeepSeek GGUF accepted");
    };
    gguf_set_val_u32(file, "deepseek41.engram.pad_id", 1); rejects();
    gguf_set_val_u32(file, "deepseek41.engram.pad_id", 0);
    const uint32_t invalid_primes[]{3, 7};
    gguf_set_arr_data(file, "deepseek41.engram.primes", GGUF_TYPE_UINT32, invalid_primes, 2); rejects();
    gguf_set_arr_data(file, "deepseek41.engram.primes", GGUF_TYPE_UINT32, primes, 2);
    gguf_set_val_str(file, "deepseek41.engram.encoding", "unknown"); rejects();
    gguf_set_val_str(file, "deepseek41.engram.encoding", "e4m3_e8m0_32_row264");
    auto invalid_config = config;
    invalid_config["text_config"]["hidden_size"] = 128;
    gguf_set_val_str(file, "deepseek41.config", invalid_config.dump().c_str()); rejects();
    gguf_free(file); ggml_free(tensors);
}

void qwen4_source(const std::filesystem::path& directory) {
    gyro_layout();
    ptq1_layout();
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
    auto* gate = ggml_new_tensor_3d(tensors, GGML_TYPE_PTQ1_0, 256, 3, 2);
    ggml_set_name(gate, "blk.0.ffn_gate_exps.weight");
    std::memset(gate->data, 0, ggml_nbytes(gate));
    gguf_add_tensor(main, gate);
    auto* embedding = ggml_new_tensor_2d(tensors, GGML_TYPE_Q4_K, 256, 3);
    ggml_set_name(embedding, "token_embd.weight");
    std::memset(embedding->data, 0, ggml_nbytes(embedding));
    gguf_add_tensor(main, embedding);
    auto* qkv = ggml_new_tensor_2d(tensors, GGML_TYPE_Q8_0, 256, 512);
    ggml_set_name(qkv, "blk.0.attn_qkv.weight");
    std::memset(qkv->data, 0, ggml_nbytes(qkv));
    gguf_add_tensor(main, qkv);
    const std::vector<int32_t> gu_types{143, 42, 143, 11}, down_types{42, 11, 143, 42};
    const std::vector<int32_t> uniform_gu(4, 143);
    std::map<std::string, std::vector<std::byte>> mixed_payloads;
    for (const int layer : {1, 2}) {
    for (const auto* role : {"gate", "up", "down"}) {
        const auto& types = std::string_view(role) == "down" ? down_types : layer == 1 ? gu_types : uniform_gu;
        auto& payload = mixed_payloads[std::to_string(layer) + role];
        for (const auto type : {GGML_TYPE_PTQ1_0, GGML_TYPE_Q2_0, GGML_TYPE_Q3_K}) {
            const auto count = std::count(types.begin(), types.end(), int32_t(type));
            if (!count) continue;
            auto* t = ggml_new_tensor_3d(tensors, type, 256, 3, count);
            auto suffix = std::string(ggml_type_name(type));
            std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });
            const auto name = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps" +
                (count == 4 ? std::string{} : "." + suffix) + ".weight";
            ggml_set_name(t, name.c_str());
            int slot = 0;
            const auto bytes = ggml_row_size(type, 256) * 3;
            for (int e = 0; e < 4; ++e) if (types[e] == type)
                std::memset(static_cast<uint8_t*>(t->data) + slot++ * bytes, e + 17, bytes);
            gguf_add_tensor(main, t);
        }
        for (int e = 0; e < 4; ++e) {
            const auto bytes = ggml_row_size(static_cast<ggml_type>(types[e]), 256) * 3;
            payload.insert(payload.end(), bytes, std::byte(e + 17));
        }
    }
    for (const auto* kind : {"gu", "down"}) {
        if (layer == 2 && std::string_view(kind) == "gu") continue;
        auto* t = ggml_new_tensor_1d(tensors, GGML_TYPE_I32, 4);
        const auto name = "blk." + std::to_string(layer) + ".ffn_exps_" + kind + "_type";
        ggml_set_name(t, name.c_str());
        const auto& values = std::string_view(kind) == "gu" ? gu_types : down_types;
        std::memcpy(t->data, values.data(), 16);
        gguf_add_tensor(main, t);
    }
    }
    const auto main_path = directory / "qwen4.gguf";
    require(gguf_write_to_file(main, main_path.string().c_str(), false), "Qwen4 main write failed");
    auto* type_map = ggml_get_tensor(tensors, "blk.1.ffn_exps_gu_type");
    for (const int32_t invalid : {142, 11}) {
        static_cast<int32_t*>(type_map->data)[0] = invalid;
        const auto invalid_path = directory / ("mixed-invalid-" + std::to_string(invalid) + ".gguf");
        require(gguf_write_to_file(main, invalid_path.string().c_str(), false), "invalid mixed fixture write failed");
        bool rejected = false;
        try { mfq::GgufModelSource invalid_source(invalid_path); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "invalid mixed expert type or slot count accepted");
    }
    static_cast<int32_t*>(type_map->data)[0] = gu_types[0];
    auto* companion = gguf_init_empty();
    auto* ple = ggml_new_tensor_2d(tensors, GGML_TYPE_IQ4_NL, 128, 4);
    ggml_set_name(ple, "per_layer_token_embd.weight");
    std::memset(ple->data, 17, ggml_nbytes(ple));
    gguf_add_tensor(companion, ple);
    const auto table_path = directory / "ple.gguf";
    require(gguf_write_to_file(companion, table_path.string().c_str(), false), "Qwen4 PLE write failed");
    gguf_free(main); gguf_free(companion); ggml_free(tensors);
    raw_part_source(main_path, {table_path});
    const mfq::GgufModelSource source(main_path, {table_path});
    for (const auto* role : {"gate", "up", "down"}) {
        const auto name = std::string("model.block.1.mlp.experts.") + role + ".weight";
        const auto* tensor = source.find_tensor(name);
        require(tensor && tensor->dtype == "GGUF_MIXED" &&
            *tensor->shape == std::vector<int64_t>({4, 3, 256}), "mixed expert bank was not normalized");
        const auto& expected = mixed_payloads.at(std::string("1") + role);
        require(source.read(name) == expected, "mixed expert bank did not preserve expert order");
        require(!source.tensor_file_range(name), "mixed expert bank exposed a false contiguous file range");
        std::vector<std::byte> partial(97), tail(17);
        source.tensor_batch_reader(name)({{499, partial.data(), partial.size()}, {expected.size() - 17, tail.data(), tail.size()}});
        require(std::equal(partial.begin(), partial.end(), expected.begin() + 499) &&
            std::equal(tail.begin(), tail.end(), expected.end() - 17), "mixed expert cross-format range reads differ");
        const auto batch = source.tensor_batch_reader(name);
        std::vector<std::byte> reordered(320), duplicate(111), overlap(100), wanted(100);
        batch({{expected.size() - 83, reordered.data(), 83}, {0, reordered.data() + 83, 111},
            {91, reordered.data() + 194, 126}, {0, duplicate.data(), 111}, {expected.size(), nullptr, 0}});
        require(std::equal(reordered.begin(), reordered.begin() + 83, expected.end() - 83) &&
            std::equal(reordered.begin() + 83, reordered.begin() + 194, expected.begin()) &&
            std::equal(reordered.begin() + 194, reordered.end(), expected.begin() + 91) &&
            std::equal(duplicate.begin(), duplicate.end(), expected.begin()), "mixed expert reordered or duplicate reads differ");
        std::copy_n(expected.begin(), wanted.size(), wanted.begin());
        std::copy_n(expected.end() - 40, 40, wanted.begin() + 13);
        batch({{0, overlap.data(), overlap.size()}, {expected.size() - 40, overlap.data() + 13, 40}});
        require(overlap == wanted, "mixed expert overlapping destinations lost caller order");
        std::fill(overlap.begin(), overlap.end(), std::byte{99});
        bool invalid = false;
        try { batch({{0, overlap.data(), overlap.size()}, {expected.size(), overlap.data(), 1}}); }
        catch (const std::out_of_range&) { invalid = true; }
        require(invalid && overlap == std::vector<std::byte>(100, std::byte{99}), "invalid mixed batch changed a destination");
        invalid = false;
        try { source.tensor_reader(name)(0, nullptr, 1); }
        catch (const std::invalid_argument&) { invalid = true; }
        require(invalid, "mixed scalar reader accepted a null destination");
        mfq::ModelSource::TensorReader retained;
        mfq::ModelSource::TensorBatchReader retained_batch;
        {
            const mfq::GgufModelSource temporary(main_path, {table_path});
            retained = temporary.tensor_reader(name);
            retained_batch = temporary.tensor_batch_reader(name);
        }
        retained(expected.size(), nullptr, 0);
        retained(91, duplicate.data(), duplicate.size());
        retained_batch({{expected.size() - overlap.size(), overlap.data(), overlap.size()}});
        require(std::equal(duplicate.begin(), duplicate.end(), expected.begin() + 91) &&
            std::equal(overlap.begin(), overlap.end(), expected.end() - overlap.size()), "retained mixed reader expired");
        const auto layouts = nlohmann::json::parse(source.metadata().at("gguf_mixed_experts"));
        require(layouts.at(name).size() == 4 && layouts.at(name)[2]["dtype"] == "GGUF_PTQ1_0",
            "mixed expert precision map was lost");
    }
    const auto* only_down = source.find_tensor("model.block.2.mlp.experts.down.weight");
    require(only_down && only_down->dtype == "GGUF_MIXED" &&
        source.read(only_down->name) == mixed_payloads.at("2down"), "Down-only mixed expert map was not normalized");
    require(source.find_tensor("model.block.2.mlp.experts.gate.weight")->dtype == "GGUF_PTQ1_0",
        "uniform Gate bank was changed by a mixed Down map");
    const auto* packed_down = source.find_tensor("model.block.0.mlp.experts.down.weight");
    require(packed_down && packed_down->dtype == "GGUF_Q2_0" && packed_down->nbytes == 108 &&
        *packed_down->shape == std::vector<int64_t>({2, 3, 64}), "GGUF Q2_0 expert metadata wrong");
    const auto packed_bytes = source.read(packed_down->name);
    const auto* packed_gate = source.find_tensor("model.block.0.mlp.experts.gate.weight");
    require(packed_gate && packed_gate->dtype == "GGUF_PTQ1_0" && packed_gate->nbytes == 336 &&
        *packed_gate->shape == std::vector<int64_t>({2, 3, 256}), "GGUF PTQ1_0 expert metadata wrong");
    require(source.read(packed_gate->name) == std::vector<std::byte>(336), "GGUF PTQ1_0 payload changed");
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

    auto* predictor_tensors = ggml_init({3 << 20, nullptr, false});
    auto* predictor = gguf_init_empty();
    gguf_set_val_str(predictor, "general.architecture", "qwen4exp-mtp");
    const auto add_predictor = [&](const char* name, ggml_type type, std::vector<int64_t> shape) {
        std::reverse(shape.begin(), shape.end());
        auto* tensor = ggml_new_tensor(predictor_tensors, type, static_cast<int>(shape.size()), shape.data());
        ggml_set_name(tensor, name);
        std::memset(tensor->data, 0, ggml_nbytes(tensor));
        if (type == GGML_TYPE_F32)
            for (int64_t i = 0; i < ggml_nelements(tensor); ++i) static_cast<float*>(tensor->data)[i] = .25f;
        gguf_add_tensor(predictor, tensor);
    };
    add_predictor("mtp.pre_fc_norm_embedding.weight", GGML_TYPE_F32, {256});
    add_predictor("mtp.pre_fc_norm_hidden.weight", GGML_TYPE_F32, {512});
    add_predictor("mtp.fc_embedding.weight", GGML_TYPE_BF16, {256, 256});
    add_predictor("mtp.fc_hidden.weight", GGML_TYPE_BF16, {256, 256});
    add_predictor("mtp.layers.0.self_attn.q_proj.weight", GGML_TYPE_BF16, {512, 256});
    add_predictor("mtp.layers.0.mlp.gate.weight", GGML_TYPE_BF16, {4, 256});
    add_predictor("mtp.layers.0.mlp.experts.gate_up_proj", GGML_TYPE_Q2_0, {4, 256, 256});
    add_predictor("mtp.layers.0.mlp.experts.down_proj", GGML_TYPE_Q2_0, {4, 256, 128});
    const auto predictor_path = directory / "predictor.gguf";
    require(gguf_write_to_file(predictor, predictor_path.string().c_str(), false), "GGUF predictor write failed");
    const mfq::GgufModelSource with_predictor(main_path, {table_path, predictor_path});
    const auto predictor_config = nlohmann::json::parse(with_predictor.model_config_json());
    require(predictor_config["mtp_num_hidden_layers"] == 1, "GGUF companion predictor was not declared");
    require(with_predictor.find_tensor("predictor.block.0.mlp.experts.gate_up.weight") &&
        *with_predictor.find_tensor("predictor.block.0.mlp.experts.gate_up.weight")->shape ==
            std::vector<int64_t>({4, 256, 256}), "GGUF fused predictor experts were not normalized");
    const auto predictor_norm = with_predictor.read("predictor.embedding_norm.weight");
    float original_norm = 0;
    std::memcpy(&original_norm, predictor_norm.data() + 12, 4);
    require(original_norm == .25f, "raw MTP companion norm was converted as a folded main-model norm");
    require(with_predictor.read(packed_down->name) == packed_bytes, "MTP companion changed backbone weights");
    {
        auto* canonical_tensors = ggml_init({3 << 20, nullptr, false});
        auto* canonical = gguf_init_empty();
        gguf_set_val_str(canonical, "general.architecture", "qwen4exp-mtp");
        const auto add = [&](const char* name, ggml_type type, std::vector<int64_t> shape) {
            std::reverse(shape.begin(), shape.end());
            auto* tensor = ggml_new_tensor(canonical_tensors, type, int(shape.size()), shape.data());
            ggml_set_name(tensor, name);
            std::memset(tensor->data, 0, ggml_nbytes(tensor));
            if (type == GGML_TYPE_F32)
                for (int64_t i = 0; i < ggml_nelements(tensor); ++i) static_cast<float*>(tensor->data)[i] = 1.25f;
            gguf_add_tensor(canonical, tensor);
        };
        add("predictor.embedding_norm.weight", GGML_TYPE_F32, {256});
        add("predictor.hidden_norm.weight", GGML_TYPE_F32, {512});
        add("predictor.fusion.embedding.weight", GGML_TYPE_BF16, {256, 256});
        add("predictor.fusion.hidden.weight", GGML_TYPE_BF16, {256, 256});
        add("predictor.block.0.attention.query.weight", GGML_TYPE_BF16, {512, 256});
        add("predictor.block.0.mlp.router.weight", GGML_TYPE_F32, {4, 256});
        add("predictor.block.0.mlp.experts.gate.weight", GGML_TYPE_Q2_0, {4, 128, 256});
        add("predictor.block.0.mlp.experts.up.weight", GGML_TYPE_Q2_0, {4, 128, 256});
        add("predictor.block.0.mlp.experts.down.weight", GGML_TYPE_Q2_0, {4, 256, 128});
        add("predictor.output.weight", GGML_TYPE_Q6_K, {32, 256});
        add("predictor.output.token_ids", GGML_TYPE_I64, {32});
        const auto canonical_path = directory / "canonical-predictor.gguf";
        require(gguf_write_to_file(canonical, canonical_path.string().c_str(), false), "canonical MTP write failed");
        const mfq::GgufModelSource canonical_source(main_path, {table_path, canonical_path});
        const auto canonical_config = nlohmann::json::parse(canonical_source.model_config_json());
        require(canonical_config["mtp_num_hidden_layers"] == 1, "canonical MTP layer count differs");
        require(canonical_source.find_tensor("predictor.block.0.mlp.experts.gate.weight") &&
            canonical_source.find_tensor("predictor.block.0.mlp.experts.up.weight") &&
            canonical_source.find_tensor("predictor.output.token_ids"), "canonical MTP tensors were discarded");
        const auto norm = canonical_source.read("predictor.embedding_norm.weight");
        float delta = 0; std::memcpy(&delta, norm.data() + 12, 4);
        require(delta == .25f, "canonical MTP norm no longer uses the backbone convention");
        gguf_free(canonical); ggml_free(canonical_tensors);
    }
    add_predictor("mtp.unsupported.weight", GGML_TYPE_F32, {1});
    require(gguf_write_to_file(predictor, predictor_path.string().c_str(), false), "GGUF invalid predictor write failed");
    bool unsupported_predictor = false;
    try { mfq::GgufModelSource invalid(main_path, {table_path, predictor_path}); }
    catch (const std::runtime_error&) { unsupported_predictor = true; }
    require(unsupported_predictor, "GGUF predictor silently ignored unsupported tensors");
    gguf_free(predictor); ggml_free(predictor_tensors);
    const auto mixed_reader = source.tensor_reader("model.block.1.mlp.experts.gate.weight");
    const auto mixed_batch = source.tensor_batch_reader("model.block.1.mlp.experts.gate.weight");
    std::filesystem::resize_file(main_path, 0);
    std::byte target{};
    bool truncated = false;
    try { mixed_reader(0, &target, 1); } catch (const std::runtime_error&) { truncated = true; }
    require(truncated, "mixed scalar reader accepted a truncated backing file");
    truncated = false;
    try { mixed_batch({{0, &target, 1}}); } catch (const std::runtime_error&) { truncated = true; }
    require(truncated, "mixed batch reader accepted a truncated backing file");
}
}

namespace {
void compact_formats(const std::filesystem::path& directory) {
    auto* tensors = ggml_init({2 << 20, nullptr, false});
    auto* header = gguf_init_empty();
    gguf_set_val_str(header, "general.architecture", "qwen35");
    gguf_set_val_u32(header, "qwen35.embedding_length", 128);
    gguf_set_val_u32(header, "qwen35.block_count", 1);
    gguf_set_val_u32(header, "qwen35.attention.head_count", 1);
    gguf_set_val_u32(header, "qwen35.attention.head_count_kv", 1);
    gguf_set_val_u32(header, "qwen35.context_length", 32768);
    gguf_set_val_f32(header, "qwen35.rope.freq_base", 10000.f);
    gguf_set_val_u32(header, "agention.trellis.state_bits", 15);
    std::vector<std::vector<std::byte>> payloads;
    for (int type = GGML_TYPE_PTQ1_0; type <= GGML_TYPE_TQK4; ++type) {
        auto* matrix = ggml_new_tensor_2d(tensors, ggml_type(type), 128, 7);
        const auto name = "compact." + std::to_string(type) + ".weight";
        ggml_set_name(matrix, name.c_str());
        auto* bytes = static_cast<std::byte*>(matrix->data);
        for (std::size_t i = 0; i < ggml_nbytes(matrix); ++i) bytes[i] = std::byte((i * 37 + type) % 256);
        payloads.emplace_back(bytes, bytes + ggml_nbytes(matrix));
        gguf_add_tensor(header, matrix);
    }
    const auto path = directory / "compact-formats.gguf";
    require(gguf_write_to_file(header, path.string().c_str(), false), "compact GGUF write failed");
    {
        mfq::GgufModelSource source(path);
        for (int type = GGML_TYPE_PTQ1_0; type <= GGML_TYPE_TQK4; ++type) {
            const auto name = "compact." + std::to_string(type) + ".weight";
            const auto* tensor = source.find_tensor(name);
            auto dtype = std::string("GGUF_") + ggml_type_name(ggml_type(type));
            std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) { return std::toupper(c); });
            require(tensor && tensor->dtype == dtype && *tensor->shape == std::vector<int64_t>({7, 128}),
                "compact GGUF type/shape changed");
            require(source.read(name) == payloads[type - GGML_TYPE_PTQ1_0], "compact GGUF changed original bytes");
        }
    }
    gguf_set_val_u32(header, "prism.hadamard.version", 2);
    gguf_set_val_u32(header, "prism.hadamard.block_size", 128);
    gguf_set_val_str(header, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    gguf_set_val_str(header, "prism.hadamard.axis", "input-last-dimension");
    gguf_set_val_str(header, "prism.hadamard.sign_mode", "explicit");
    const int64_t widths[]{128};
    std::array<int8_t, 128> signs{};
    for (int i = 0; i < 128; ++i) signs[i] = (i * 17 + i / 7) % 3 ? -1 : 1;
    gguf_set_arr_data(header, "prism.hadamard.sign_widths", GGUF_TYPE_INT64, widths, 1);
    gguf_set_arr_data(header, "prism.hadamard.sign_values", GGUF_TYPE_INT8, signs.data(), signs.size());
    std::vector<std::string> names;
    for (int type = GGML_TYPE_PTQ1_0; type <= GGML_TYPE_TQK4; ++type)
        names.push_back("compact." + std::to_string(type) + ".weight");
    std::vector<const char*> name_pointers;
    for (const auto& name : names) name_pointers.push_back(name.c_str());
    gguf_set_arr_str(header, "prism.hadamard.weight_names", name_pointers.data(), name_pointers.size());
    const auto rotated = directory / "compact-hadamard.gguf";
    require(gguf_write_to_file(header, rotated.string().c_str(), false), "Hadamard metadata write failed");
    {
        mfq::GgufModelSource source(rotated);
        const auto transforms = nlohmann::json::parse(source.metadata().at("gguf_hadamard"));
        require(transforms.size() == names.size() && source.metadata().at("gguf_requires_hadamard") == "true",
            "Hadamard metadata/capability requirement missing");
        const auto* parsed=source.hadamard_transforms();
        require(parsed && parsed==source.hadamard_transforms() && parsed->size()==names.size(),
            "Hadamard source did not retain one immutable parsed map");
        for (const auto& name : names) require(transforms.at(name).at("block_size") == 128 &&
            transforms.at(name).at("signs").get<std::vector<int8_t>>() == std::vector<int8_t>(signs.begin(), signs.end()),
            "Hadamard signs or block size changed");
        for(const auto& name:names)require(parsed->at(name).block_size==128 &&
            parsed->at(name).signs==transforms.at(name).at("signs").get<std::vector<int8_t>>(),
            "Typed Hadamard map differs from normalized metadata");
    }
    signs[3] = 0;
    gguf_set_arr_data(header, "prism.hadamard.sign_values", GGUF_TYPE_INT8, signs.data(), signs.size());
    const auto bad_sign = directory / "compact-invalid-sign.gguf";
    require(gguf_write_to_file(header, bad_sign.string().c_str(), false), "invalid-sign fixture write failed");
    bool sign_rejected = false;
    try { mfq::GgufModelSource source(bad_sign); }
    catch (const std::runtime_error& error) { sign_rejected = std::string(error.what()).find("sign value") != std::string::npos; }
    require(sign_rejected, "Hadamard accepted a non-sign value");
    gguf_set_val_u32(header, "prism.hadamard.version", 1);
    gguf_set_val_str(header, "prism.hadamard.sign_mode", "identity");
    const auto identity = directory / "compact-hadamard-identity.gguf";
    require(gguf_write_to_file(header, identity.string().c_str(), false), "identity-Hadamard fixture write failed");
    {
        mfq::GgufModelSource source(identity);
        const auto transforms = nlohmann::json::parse(source.metadata().at("gguf_hadamard"));
        for (const auto& name : names) require(transforms.at(name).at("signs").empty(), "Identity Hadamard acquired signs");
        for(const auto& name:names)require(source.hadamard_transforms()->at(name).signs.empty(),
            "Typed identity Hadamard acquired signs");
    }
    gguf_set_val_u32(header, "agention.trellis.state_bits", 14);
    const auto invalid = directory / "compact-invalid-state.gguf";
    require(gguf_write_to_file(header, invalid.string().c_str(), false), "invalid-state fixture write failed");
    bool rejected = false;
    try { mfq::GgufModelSource source(invalid); }
    catch (const std::runtime_error& error) { rejected = std::string(error.what()).find("state_bits=15") != std::string::npos; }
    require(rejected, "unsupported Gyro state width accepted");
    gguf_free(header); ggml_free(tensors);
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
        raw_part_source(path);
        raw_part_source(path, {}, 2);
        raw_part_source(path, {}, 2, 16, 0);
        raw_part_source(path, {}, 2, 16, 2);
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
        {
            const auto mirror=directory/"mirror";std::filesystem::create_directory(mirror);
            std::filesystem::copy_file(path,mirror/path.filename());
            MirrorSetting setting(mirror);
            mfq::GgufModelSource mirrored(path);
            const auto read=mirrored.tensor_reader(table->name);
            const auto batch=mirrored.tensor_batch_reader(table->name);
            std::array<std::byte,19> first{},last{};
            read(0,first.data(),first.size());batch({{13,last.data(),last.size()}});
            require(std::equal(first.begin(),first.end(),data.begin()) &&
                std::equal(last.begin(),last.end(),data.begin()+13),"GGUF mirror integration changed original quantized bytes");
        }
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
        compact_formats(directory);
        deepseek_source(directory);
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(directory);
        std::cerr << error.what() << '\n'; return 1;
    }
}
