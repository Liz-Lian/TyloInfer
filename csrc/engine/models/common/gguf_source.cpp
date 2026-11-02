#include "engine/models/common/gguf_source.h"

#include "engine/models/common/gguf.h"
#include "engine/models/common/legacy_names.h"
#include "engine/weights/gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <utility>

namespace mfq::models {
namespace {
using json = nlohmann::json;
} // namespace

struct GgufSource::Impl {
    std::vector<std::filesystem::path> paths;
    std::string arch;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, gguf::Record> records;
    std::unordered_map<std::string, std::vector<std::byte>> asset_data;
    std::vector<std::string> assets;
};

GgufSource::GgufSource(const std::filesystem::path& path,
                       std::vector<std::filesystem::path> companions)
    : impl_(std::make_shared<Impl>()) {
    const auto split_count = gguf::splits(path);
    impl_->paths = gguf::paths(path, split_count);
    const auto model_shards = impl_->paths.size();
    for (const auto& companion : companions) {
        const auto absolute = std::filesystem::absolute(companion);
        if (std::find(impl_->paths.begin(), impl_->paths.end(), absolute) != impl_->paths.end())
            throw std::runtime_error("GGUF companion duplicates a model shard");
        impl_->paths.push_back(absolute);
    }

    json meta = json::object(), shapes = json::object();
    for (std::size_t shard = 0; shard < impl_->paths.size(); ++shard) {
        gguf::Shard source(impl_->paths[shard], shard < model_shards ? split_count : 1, shard);
        if (shard == 0) {
            meta = source.metadata();
            if (!meta.contains("general.architecture")) throw std::runtime_error("GGUF has no architecture");
            impl_->arch = meta["general.architecture"].get<std::string>();
            impl_->asset_data[kTokenizerGgufAsset] = source.tokenizer();
        }
        std::unordered_map<std::string, std::string> predictor_names;
        const bool raw_predictor = shard >= model_shards &&
            source.scalar("general.architecture") == "qwen4exp-mtp";
        if (raw_predictor) {
            const auto names = source.names();
            if (impl_->arch != "qwen4exp")
                throw std::runtime_error("Qwen4 MTP companion requires a Qwen4 backbone");
            const auto aliases = legacy_names("qwen4_exp_text",
                json{{"model_type", "qwen4_exp_text"},
                     {"num_hidden_layers", meta.at("qwen4exp.block_count")}}.dump(), names);
            for (const auto& [canonical, stored] : aliases.canonical_to_stored)
                if (canonical.rfind("predictor.", 0) == 0) predictor_names.emplace(stored, canonical);
            if (predictor_names.size() != names.size())
                throw std::runtime_error("GGUF MTP companion contains unsupported predictor tensors");
        }
        for (std::int64_t i = 0; i < source.count(); ++i) {
            const auto raw = source.tensor(i);
            auto shape = raw.shape;
            auto name = raw.name;
            if (raw_predictor) name = predictor_names.at(raw.name);
            else name = gguf_name(raw.name);
            if (impl_->arch == "qwen4exp") {
                if (name == "per_layer_token_embd.weight") {
                    const auto layers = meta.at("qwen4exp.ple.layers").get<std::vector<std::int64_t>>();
                    if (layers.size() != 1) throw std::runtime_error("GGUF PLE table requires one declared layer");
                    name = "model.block." + std::to_string(layers.front()) +
                        ".position_embedding.ngram.shard.0.weight";
                }
                if (name.find(".mlp.shared_expert.router.weight") != std::string::npos && shape.size() == 1)
                    shape.insert(shape.begin(), 1);
            }
            if (impl_->records.count(name)) throw std::runtime_error("duplicate GGUF tensor: " + name);
            auto record = gguf::prepare(raw, name, shape);
            const bool norm_delta = impl_->arch == "qwen4exp" && !raw_predictor &&
                name.find("norm.weight") != std::string::npos &&
                name.find(".linear_attention.norm.weight") == std::string::npos;
            const bool decay_log = impl_->arch == "qwen4exp" &&
                name.find(".linear_attention.a") != std::string::npos &&
                name.size() >= 2 && name.substr(name.size() - 2) == ".a";
            if (norm_delta || decay_log) gguf::normalize(record, raw, decay_log);
            shapes[name] = shape;
            impl_->tensors.push_back(record.tensor);
            impl_->records.emplace(name, std::move(record));
        }
    }
    if (impl_->arch == "qwen4exp") {
        const auto blocks = meta.at("qwen4exp.block_count").get<std::int64_t>();
        for (std::int64_t block = 0; block < blocks; ++block) {
            const auto root = "model.block." + std::to_string(block) + ".attention.indexer.";
            const auto q = impl_->records.find(root + "query.weight");
            const auto k = impl_->records.find(root + "key.weight");
            if (q == impl_->records.end() && k == impl_->records.end()) continue;
            if (q == impl_->records.end() || k == impl_->records.end())
                throw std::runtime_error("GGUF indexer query/key pair is incomplete");
            const auto name = root + "query_key.weight";
            auto joined = gguf::join(name, q->second, k->second);
            impl_->records.erase(root + "query.weight");
            impl_->records.erase(root + "key.weight");
            impl_->tensors.erase(std::remove_if(impl_->tensors.begin(), impl_->tensors.end(),
                [&](const TensorMetadata& tensor) {
                    return tensor.name == root + "query.weight" || tensor.name == root + "key.weight";
                }), impl_->tensors.end());
            shapes.erase(root + "query.weight");
            shapes.erase(root + "key.weight");
            shapes[name] = joined.tensor.shape.value();
            impl_->tensors.push_back(joined.tensor);
            impl_->records.emplace(name, std::move(joined));
        }
        const auto layers = meta.at("qwen4exp.ple.layers").get<std::vector<std::int64_t>>();
        for (const auto layer : layers) {
            const auto root = "model.block." + std::to_string(layer) + ".position_embedding.ngram.";
            for (const auto* key : {"layer_multipliers", "head_offsets", "head_vocab_sizes"}) {
                const auto values = meta.at(std::string("qwen4exp.ple.") + key).get<std::vector<std::int64_t>>();
                const auto name = root + key;
                auto record = gguf::integers(name, values);
                shapes[name] = record.tensor.shape.value();
                impl_->tensors.push_back(record.tensor);
                impl_->records.emplace(name, std::move(record));
            }
        }
    }
    const auto config = gguf_config(meta, shapes, impl_->arch).dump();
    impl_->asset_data[std::string(kModelConfigAsset)] = std::vector<std::byte>(
        reinterpret_cast<const std::byte*>(config.data()),
        reinterpret_cast<const std::byte*>(config.data() + config.size()));
    for (const auto& [name, data] : impl_->asset_data) impl_->assets.push_back(name);
    impl_->metadata["source_format"] = "\"gguf\"";
    impl_->metadata["tensor_shapes"] = shapes.dump();
    impl_->metadata["gguf_metadata"] = meta.dump();
}

GgufSource::~GgufSource() = default;
const std::vector<std::filesystem::path>& GgufSource::source_paths() const noexcept { return impl_->paths; }
std::string_view GgufSource::architecture() const noexcept { return impl_->arch; }
const std::unordered_map<std::string, std::string>& GgufSource::metadata() const noexcept { return impl_->metadata; }
const std::vector<TensorMetadata>& GgufSource::tensors() const noexcept { return impl_->tensors; }
const TensorMetadata* GgufSource::find_tensor(std::string_view name) const noexcept {
    const auto it = impl_->records.find(std::string(name));
    return it == impl_->records.end() ? nullptr : &it->second.tensor;
}
GgufSource::TensorReader GgufSource::tensor_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    return [record = it->second](std::uint64_t offset, std::byte* destination, std::size_t size) {
        record.read(offset, destination, size);
    };
}
std::optional<TensorFileRange> GgufSource::tensor_file_range(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) return std::nullopt;
    return it->second.file_range();
}
GgufSource::TensorBatchReader GgufSource::tensor_batch_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    return [record = it->second](const std::vector<ReadSpan>& spans) {
        record.read(spans);
    };
}
void GgufSource::read_range_into(std::string_view name, std::uint64_t offset,
                                 std::byte* destination, std::size_t size) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    it->second.read(offset, destination, size);
}
const std::vector<std::string>& GgufSource::assets() const noexcept { return impl_->assets; }
bool GgufSource::has_asset(std::string_view name) const noexcept {
    return impl_->asset_data.count(std::string(name)) != 0;
}
std::vector<std::byte> GgufSource::read_asset(std::string_view name) const {
    return impl_->asset_data.at(std::string(name));
}
std::optional<ModelGraph> GgufSource::model_graph() const { return std::nullopt; }

} // namespace mfq::models
