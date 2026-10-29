#include "source/hf_model_source.h"
#include "formats/mlx_affine.h"
#include "source/mlx_tensor_names.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void u64(std::ostream& out, std::uint64_t value) {
    for (unsigned i = 0; i < 8; ++i) out.put(char(value >> (i * 8)));
}
}

int main() {
    const auto root = std::filesystem::temp_directory_path() /
        ("tinfer-mlx-affine-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try {
        std::filesystem::create_directory(root);
        std::ofstream(root / "config.json") << R"({"model_type":"qwen3_5","quantization":{"bits":2,"group_size":64,"mode":"affine","probe":{"bits":5,"group_size":32,"mode":"affine"}}})";
        const nlohmann::json header{
            {"probe.weight", {{"dtype", "U32"}, {"shape", {2, 3, 20}}, {"data_offsets", {0, 480}}}},
            {"probe.scales", {{"dtype", "BF16"}, {"shape", {2, 3, 4}}, {"data_offsets", {480, 528}}}},
            {"probe.biases", {{"dtype", "BF16"}, {"shape", {2, 3, 4}}, {"data_offsets", {528, 576}}}}};
        std::vector<std::byte> payload(576);
        for (std::size_t i = 0; i < payload.size(); ++i) payload[i] = std::byte(i * 17u + 3u);
        {
            std::ofstream file(root / "model.safetensors", std::ios::binary);
            const auto text = header.dump(); u64(file, text.size()); file << text;
            file.write(reinterpret_cast<const char*>(payload.data()), payload.size());
        }
        mfq::HfModelSource source(root);
        const auto* tensor = source.find_tensor("probe.weight");
        require(tensor && tensor->dtype == "MLX-AFFINE" && tensor->shape ==
            std::vector<std::int64_t>{2, 3, 128}, "MLX logical shape/format was not preserved");
        require(!source.find_tensor("probe.scales") && !source.find_tensor("probe.biases"),
            "MLX auxiliary arrays remained separate model tensors");
        const auto bytes = source.read("probe.weight");
        const auto layout = mfq::MlxAffineLayout::parse(bytes.data(), bytes.size());
        require(layout.bits == 5 && layout.group_size == 32 && layout.values_bytes == 480 &&
            layout.scales_bytes == 48 && layout.total_bytes() == bytes.size(), "MLX module override was ignored");
        require(std::equal(bytes.begin() + layout.prefix_bytes, bytes.end(), payload.begin()),
            "MLX affine loader unpacked or reordered source payloads");
        std::vector<std::byte> tail(80);
        source.read_range_into("probe.weight", layout.scales_offset() + 16, tail.data(), tail.size());
        require(std::equal(tail.begin(), tail.end(), payload.begin() + 496), "MLX segmented scale/bias read changed bytes");
        const auto mapped = source.tensor_file_range_at("probe.weight", layout.scales_offset(), layout.scales_bytes);
        require(mapped && mapped->nbytes == 48 && mapped->path.filename() == "model.safetensors",
            "MLX scale range did not retain its original file location");
        std::ifstream original(mapped->path, std::ios::binary); original.seekg(mapped->offset);
        std::vector<std::byte> scale_bytes(mapped->nbytes);
        original.read(reinterpret_cast<char*>(scale_bytes.data()), scale_bytes.size());
        require(std::equal(scale_bytes.begin(), scale_bytes.end(), payload.begin() + 480), "MLX mapped scale offsets changed");
        require(!source.tensor_file_range_at("probe.weight", layout.prefix_bytes - 1, 2),
            "MLX inline/source boundary was exposed as one raw file range");
        require(mfq::mlx_hf_tensor_name("language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shards.12.weight") ==
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_12.weight", "MLX PLE names were not normalized");
        require(mfq::mlx_hf_tensor_shape("model.block.0.linear_attention.conv.weight", {10240, 4, 1}) ==
            std::vector<std::int64_t>{10240, 1, 4}, "MLX depthwise convolution singleton axis was not normalized");
        auto invalid = layout; invalid.group_size = 48;
        bool rejected = false;
        try { invalid.validate(); } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "MLX invalid group size was accepted");
        rejected = false;
        try { mfq::MlxAffineLayout::parse(bytes.data(), layout.prefix_bytes - 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "MLX truncated metadata was accepted");
        std::filesystem::remove_all(root);
        std::cout << "MLX affine source overrides/logical shape/bit-exact payload/range reads PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n'; return 1;
    }
}
