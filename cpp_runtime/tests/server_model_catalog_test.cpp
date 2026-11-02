// ModelCatalog discovery for the file formats TyloInfer actually loads.
#include "models/catalog.h"

#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;
using mfq::server::ModelCatalog;

namespace {

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

template <typename T> void write_scalar(std::ofstream& stream, T value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void write_name(std::ofstream& stream, const std::string& value) {
    write_scalar(stream, static_cast<std::uint32_t>(value.size()));
    stream.write(value.data(), static_cast<std::streamsize>(value.size()));
}

// A minimal MFQ1 container; .tq and .mfq share this layout.
void write_container(const fs::path& path) {
    const std::string config = R"({"model_type":"qwen3_5","max_position_embeddings":8192})";
    std::ofstream stream(path, std::ios::binary);
    stream.write("MFQ1", 4);
    write_scalar<std::uint32_t>(stream, 2);
    write_name(stream, "qwen3_5");
    write_scalar<std::uint32_t>(stream, 0);
    write_scalar<std::uint32_t>(stream, 2);
    write_name(stream, "model.token_embedding.weight");
    write_name(stream, "BF16");
    write_scalar<std::uint64_t>(stream, 14);
    write_name(stream, "__mfq_asset__/model_config.json");
    write_name(stream, "BLOB");
    write_scalar<std::uint64_t>(stream, config.size());
    write_scalar<std::uint32_t>(stream, 1);
    write_scalar<std::int64_t>(stream, 1);
    write_scalar<std::uint16_t>(stream, 0);
    stream << config;
    require(stream.good(), "container fixture write failed");
}

void write_gguf(const fs::path& path, const char* tensor, int split_no = 0, int split_count = 1) {
    // gguf_add_tensor keeps pointers into the ggml buffer, so both stay alive until the write.
    auto* tensors = ggml_init({1 << 20, nullptr, false});
    require(tensors != nullptr, "ggml init failed");
    auto* gguf = gguf_init_empty();
    gguf_set_val_str(gguf, "general.architecture", "qwen35");
    gguf_set_val_u32(gguf, "qwen35.embedding_length", 256);
    gguf_set_val_u32(gguf, "qwen35.feed_forward_length", 512);
    gguf_set_val_u32(gguf, "qwen35.attention.head_count", 2);
    gguf_set_val_u32(gguf, "qwen35.attention.head_count_kv", 1);
    gguf_set_val_u32(gguf, "qwen35.block_count", 1);
    if (split_count > 1) {
        gguf_set_val_u16(gguf, "split.count", std::uint16_t(split_count));
        gguf_set_val_u16(gguf, "split.no", std::uint16_t(split_no));
    }
    auto* norm = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 256);
    ggml_set_name(norm, tensor);
    std::memset(norm->data, 3, ggml_nbytes(norm));
    gguf_add_tensor(gguf, norm);
    require(gguf_write_to_file(gguf, path.string().c_str(), false), "GGUF fixture write failed");
    gguf_free(gguf);
    ggml_free(tensors);
}

const ModelCatalog::Artifact* find(const std::vector<ModelCatalog::Artifact>& artifacts, const std::string& name) {
    const auto found = std::find_if(artifacts.begin(), artifacts.end(),
        [&](const auto& item) { return item.name == name; });
    return found == artifacts.end() ? nullptr : &*found;
}

} // namespace

int main() {
    const auto root = fs::temp_directory_path() /
        ("tinfer-catalog-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto models = root / "models";
    fs::create_directories(models);
    int failures = 0;
    try {
        write_container(models / "alpha.tq");
        write_gguf(models / "beta.gguf", "output_norm.weight");
        write_gguf(models / "ple-table.gguf", "per_layer_token_embd.weight");
        write_gguf(models / "split-00001-of-00002.gguf", "output_norm.weight", 0, 2);

        ModelCatalog catalog({models});
        auto artifacts = catalog.list(true);
        const auto* alpha = find(artifacts, "alpha");
        require(alpha && alpha->format == "tq" && alpha->complete && alpha->shard_count == 1,
            "TQ artifact discovery failed");
        const auto* beta = find(artifacts, "beta");
        require(beta && beta->format == "gguf" && beta->complete && beta->architecture == "qwen35",
            "GGUF artifact discovery failed");
        require(find(artifacts, "ple-table") == nullptr,
            "a GGUF that only holds PLE tables is listed as a standalone model");
        const auto* split = find(artifacts, "split");
        require(split && split->format == "gguf" && !split->complete && split->missing_shards == 1,
            "incomplete GGUF shard set is not reported");

        write_gguf(models / "split-00002-of-00002.gguf", "blk.0.attn_norm.weight", 1, 2);
        artifacts = catalog.list(true);
        split = find(artifacts, "split");
        require(split && split->complete && split->shard_count == 2 && split->missing_shards == 0,
            "complete GGUF shard set is not reported");

        const auto external = root / "downloaded";
        fs::create_directories(external);
        write_container(external / "gamma.tq");
        const auto registered = catalog.register_directory(ModelCatalog::DirectorySource{{}, external});
        require(registered.size() == 1 && registered.front().name == "gamma",
            "directory registration returned the wrong artifacts");
        require(fs::is_regular_file(models / ".tq-files.json"), "registration index was not written");
        // The index keeps an out-of-root artifact visible across later root-only scans.
        artifacts = catalog.list(true);
        const auto* gamma = find(artifacts, "gamma");
        require(gamma && gamma->format == "tq" && gamma->complete, "registered artifact was not kept");
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "model catalog test failed: " << error.what() << '\n';
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    return failures == 0 ? 0 : 1;
}
