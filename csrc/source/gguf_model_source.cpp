#include "gguf_model_source.h"
#include "engine/models/common/gguf.h"
#include "engine/models/common/legacy_names.h"
#include "engine/weights/encoding/dense_blob.h"
#include "engine/weights/source/gguf.h"

#include "components/ggml/include/gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>

namespace mfq {
namespace {
using json = nlohmann::json;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Ggml = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

} // namespace

struct GgufModelSource::Impl {
    std::vector<std::filesystem::path> paths;
    std::string arch;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, gguf::Record> records;
    std::unordered_map<std::string, std::vector<std::byte>> asset_data;
    std::vector<std::string> assets;
};

GgufModelSource::GgufModelSource(const std::filesystem::path& path,
        std::vector<std::filesystem::path> companions) : impl_(std::make_shared<Impl>()) {
    const auto load = [](const std::filesystem::path& file, ggml_context** tensors) {
        Gguf ctx(gguf_init_from_file(file.string().c_str(), {true, tensors}), gguf_free);
        if (!ctx) throw std::runtime_error("invalid GGUF file: " + file.filename().string());
        return ctx;
    };
    auto first = load(path, nullptr);
    const auto split_key = gguf_find_key(first.get(), "split.count");
    const auto splits = split_key < 0 ? 1 : gguf::scalar(first.get(), split_key).get<int>();
    impl_->paths = gguf::paths(path, splits);
    const auto model_shards = impl_->paths.size();
    for (const auto& companion : companions) {
        const auto absolute = std::filesystem::absolute(companion);
        if (std::find(impl_->paths.begin(), impl_->paths.end(), absolute) != impl_->paths.end())
            throw std::runtime_error("GGUF companion duplicates a model shard");
        impl_->paths.push_back(absolute);
    }

    json meta = json::object(), shapes = json::object();
    for (std::size_t shard = 0; shard < impl_->paths.size(); ++shard) {
        ggml_context* raw = nullptr;
        auto ctx = load(impl_->paths[shard], &raw);
        Ggml tensors(raw, ggml_free);
        if (gguf_get_n_tensors(ctx.get()) > (1 << 20))
            throw std::runtime_error("GGUF has too many tensor records");
        if (splits > 1 && shard < model_shards) {
            const auto split_no = gguf_find_key(ctx.get(), "split.no");
            const auto split_count = gguf_find_key(ctx.get(), "split.count");
            if (split_no < 0 || split_count < 0 || gguf::scalar(ctx.get(), split_no) != shard || gguf::scalar(ctx.get(), split_count) != splits)
                throw std::runtime_error("inconsistent GGUF shard metadata");
        }
        const auto file_size = std::filesystem::file_size(impl_->paths[shard]);
        auto file = std::make_shared<FileRangeReader>(impl_->paths[shard], model_file_read_mode());
        if (shard == 0) {
            for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
                const std::string name = gguf_get_key(ctx.get(), i);
                if (name.rfind("tokenizer.", 0) == 0 && gguf_get_kv_type(ctx.get(), i) == GGUF_TYPE_ARRAY) continue;
                const auto value = gguf::value(ctx.get(), i);
                if (!value.is_null()) meta[name] = value;
            }
            if (!meta.contains("general.architecture")) throw std::runtime_error("GGUF has no architecture");
            impl_->arch = meta["general.architecture"].get<std::string>();
            Gguf tokenizer(gguf_init_empty(), gguf_free);
            gguf_set_kv(tokenizer.get(), ctx.get());
            auto& asset = impl_->asset_data[kTokenizerGgufAsset];
            asset.resize(gguf_get_meta_size(tokenizer.get()));
            gguf_get_meta_data(tokenizer.get(), asset.data());
        }
        std::unordered_map<std::string, std::string> predictor_names;
        const auto architecture_key = gguf_find_key(ctx.get(), "general.architecture");
        const bool raw_predictor = shard >= model_shards && architecture_key >= 0 &&
            gguf::scalar(ctx.get(), architecture_key) == "qwen4exp-mtp";
        if (raw_predictor) {
            if (impl_->arch != "qwen4exp")
                throw std::runtime_error("Qwen4 MTP companion requires a Qwen4 backbone");
            std::vector<std::string> names;
            for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i)
                names.emplace_back(gguf_get_tensor_name(ctx.get(), i));
            const auto aliases = make_legacy_tensor_aliases("qwen4_exp_text",
                json{{"model_type", "qwen4_exp_text"}, {"num_hidden_layers", meta.at("qwen4exp.block_count")}}.dump(), names);
            for (const auto& [canonical, stored] : aliases.canonical_to_stored)
                if (canonical.rfind("predictor.", 0) == 0) predictor_names.emplace(stored, canonical);
            if (predictor_names.size() != names.size())
                throw std::runtime_error("GGUF MTP companion contains unsupported predictor tensors");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
            const auto* original = gguf_get_tensor_name(ctx.get(), i);
            const auto* tensor = ggml_get_tensor(tensors.get(), original);
            if (!tensor) throw std::runtime_error("GGUF tensor shape is missing");
            const auto type = gguf_get_tensor_type(ctx.get(), i);
            std::vector<int64_t> shape;
            for (int d = ggml_n_dims(tensor) - 1; d >= 0; --d) shape.push_back(tensor->ne[d]);
            auto name = raw_predictor ? predictor_names.at(original) : models::gguf_name(original);
            if (impl_->arch == "qwen4exp") {
                if (name == "per_layer_token_embd.weight") {
                    const auto layers = meta.at("qwen4exp.ple.layers").get<std::vector<int64_t>>();
                    if (layers.size() != 1) throw std::runtime_error("GGUF PLE table requires one declared layer");
                    name = "model.block." + std::to_string(layers.front()) + ".position_embedding.ngram.shard.0.weight";
                }
                if (name.find(".mlp.shared_expert.router.weight") != std::string::npos && shape.size() == 1)
                    shape.insert(shape.begin(), 1);
            }
            if (impl_->records.count(name)) throw std::runtime_error("duplicate GGUF tensor: " + name);
            gguf::Record record;
            record.file = file;
            record.path = impl_->paths[shard];
            const auto data_offset = gguf_get_data_offset(ctx.get());
            const auto tensor_offset = gguf_get_tensor_offset(ctx.get(), i);
            if (tensor_offset > std::numeric_limits<uint64_t>::max() - data_offset)
                throw std::runtime_error("GGUF tensor offset overflow: " + name);
            record.offset = data_offset + tensor_offset;
            auto size = gguf_get_tensor_size(ctx.get(), i);
            if (record.offset > file_size || size > file_size - record.offset)
                throw std::runtime_error("truncated GGUF tensor: " + name);
            std::string dtype;
            if (type == GGML_TYPE_F32) dtype = "F32";
            else if (type == GGML_TYPE_F16) dtype = "F16";
            else if (type == GGML_TYPE_BF16) dtype = "BF16";
            else if (type == GGML_TYPE_I32) dtype = "I32";
            else if (type == GGML_TYPE_I64) dtype = "I64";
            else if (ggml_is_quantized(type)) dtype = "GGUF_" + std::string(ggml_type_name(type));
            else throw std::runtime_error("unsupported GGUF tensor type: " + std::string(ggml_type_name(type)));
            std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) { return std::toupper(c); });
            if (!ggml_is_quantized(type)) record.prefix = dense::prefix(shape);
            record.tensor = {name, dtype, dtype, shape, size + record.prefix.size()};
            const bool norm_delta = impl_->arch == "qwen4exp" && !raw_predictor && name.find("norm.weight") != std::string::npos &&
                name.find(".linear_attention.norm.weight") == std::string::npos;
            const bool decay_log = impl_->arch == "qwen4exp" && name.find(".linear_attention.a") != std::string::npos &&
                name.size() >= 2 && name.substr(name.size() - 2) == ".a";
            if (norm_delta || decay_log) {
                if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)
                    throw std::runtime_error("GGUF normalization requires floating-point tensors: " + name);
                std::vector<std::byte> raw(size);
                file->read(record.offset, raw.data(), raw.size());
                const auto elements = ggml_nelements(tensor);
                record.inline_bytes = dense::prefix(shape);
                const auto begin = record.inline_bytes.size();
                record.inline_bytes.resize(begin + elements * sizeof(float));
                for (int64_t j = 0; j < elements; ++j) {
                    float value = 0;
                    if (type == GGML_TYPE_F32) std::memcpy(&value, raw.data() + j * 4, 4);
                    else {
                        uint16_t bits; std::memcpy(&bits, raw.data() + j * 2, 2);
                        if (type == GGML_TYPE_F16) value = ggml_fp16_to_fp32(bits);
                        else { const uint32_t expanded = uint32_t(bits) << 16; std::memcpy(&value, &expanded, 4); }
                    }
                    if (decay_log) {
                        if (!std::isfinite(value) || value >= 0) throw std::runtime_error("invalid GGUF GDN decay: " + name);
                        value = std::log(-value);
                    } else value -= 1.f;
                    std::memcpy(record.inline_bytes.data() + begin + j * 4, &value, 4);
                }
                record.tensor.dtype = record.tensor.stored_dtype = "F32";
                record.tensor.nbytes = record.inline_bytes.size();
            }
            shapes[name] = shape;
            impl_->tensors.push_back(record.tensor);
            impl_->records.emplace(name, std::move(record));
        }
    }
    if (impl_->arch == "qwen4exp") {
        const auto blocks = meta.at("qwen4exp.block_count").get<int64_t>();
        for (int64_t block = 0; block < blocks; ++block) {
            const auto root = "model.block." + std::to_string(block) + ".attention.indexer.";
            const auto q = impl_->records.find(root + "query.weight");
            const auto k = impl_->records.find(root + "key.weight");
            if (q == impl_->records.end() && k == impl_->records.end()) continue;
            if (q == impl_->records.end() || k == impl_->records.end()) throw std::runtime_error("GGUF indexer query/key pair is incomplete");
            const auto& qt = q->second.tensor;
            const auto& kt = k->second.tensor;
            if (!qt.shape || !kt.shape || qt.shape->size() != 2 || kt.shape->size() != 2 ||
                qt.shape->back() != kt.shape->back() || qt.dtype != kt.dtype ||
                (qt.dtype != "F32" && qt.dtype != "F16" && qt.dtype != "BF16"))
                throw std::runtime_error("GGUF indexer query/key layouts cannot be joined without changing precision");
            const std::vector<int64_t> shape{qt.shape->front() + kt.shape->front(), qt.shape->back()};
            const auto name = root + "query_key.weight";
            gguf::Record joined;
            joined.inline_bytes = dense::prefix(shape);
            for (const auto* record : {&q->second, &k->second}) {
                const auto bytes = read(record->tensor.name);
                joined.inline_bytes.insert(joined.inline_bytes.end(), bytes.begin() + record->prefix.size(), bytes.end());
            }
            joined.tensor = {name, qt.dtype, qt.stored_dtype, shape, joined.inline_bytes.size()};
            impl_->records.erase(root + "query.weight");
            impl_->records.erase(root + "key.weight");
            impl_->tensors.erase(std::remove_if(impl_->tensors.begin(), impl_->tensors.end(), [&](const TensorMetadata& tensor) {
                return tensor.name == root + "query.weight" || tensor.name == root + "key.weight";
            }), impl_->tensors.end());
            shapes.erase(root + "query.weight"); shapes.erase(root + "key.weight");
            shapes[name] = shape;
            impl_->tensors.push_back(joined.tensor);
            impl_->records.emplace(name, std::move(joined));
        }
        const auto layers = meta.at("qwen4exp.ple.layers").get<std::vector<int64_t>>();
        for (const auto layer : layers) {
            const auto root = "model.block." + std::to_string(layer) + ".position_embedding.ngram.";
            for (const auto* key : {"layer_multipliers", "head_offsets", "head_vocab_sizes"}) {
                const auto values = meta.at(std::string("qwen4exp.ple.") + key).get<std::vector<int64_t>>();
                gguf::Record record;
                const auto name = root + key;
                std::vector<int64_t> shape{static_cast<int64_t>(values.size())};
                record.inline_bytes = dense::prefix(shape);
                const auto start = record.inline_bytes.size();
                record.inline_bytes.resize(start + values.size() * sizeof(int64_t));
                std::memcpy(record.inline_bytes.data() + start, values.data(), values.size() * sizeof(int64_t));
                record.tensor = {name, "I64", "I64", shape, record.inline_bytes.size()};
                shapes[name] = shape;
                impl_->tensors.push_back(record.tensor);
                impl_->records.emplace(name, std::move(record));
            }
        }
    }
    const auto config = models::gguf_config(meta, shapes, impl_->arch).dump();
    impl_->asset_data[std::string(kModelConfigAsset)] = std::vector<std::byte>(
        reinterpret_cast<const std::byte*>(config.data()), reinterpret_cast<const std::byte*>(config.data() + config.size()));
    for (const auto& [name, data] : impl_->asset_data) impl_->assets.push_back(name);
    impl_->metadata["source_format"] = "\"gguf\"";
    impl_->metadata["tensor_shapes"] = shapes.dump();
    impl_->metadata["gguf_metadata"] = meta.dump();
}

GgufModelSource::~GgufModelSource() = default;
const std::vector<std::filesystem::path>& GgufModelSource::source_paths() const noexcept { return impl_->paths; }
std::string_view GgufModelSource::architecture() const noexcept { return impl_->arch; }
const std::unordered_map<std::string, std::string>& GgufModelSource::metadata() const noexcept { return impl_->metadata; }
const std::vector<TensorMetadata>& GgufModelSource::tensors() const noexcept { return impl_->tensors; }
const TensorMetadata* GgufModelSource::find_tensor(std::string_view name) const noexcept {
    const auto it = impl_->records.find(std::string(name));
    return it == impl_->records.end() ? nullptr : &it->second.tensor;
}
ModelSource::TensorReader GgufModelSource::tensor_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    return [record = it->second](uint64_t offset, std::byte* destination, std::size_t size) {
        record.read(offset, destination, size);
    };
}
std::optional<TensorFileRange> GgufModelSource::tensor_file_range(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) return std::nullopt;
    const auto& record = it->second;
    return record.file_range();
}
ModelSource::TensorBatchReader GgufModelSource::tensor_batch_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    return [record = it->second](const std::vector<ReadSpan>& spans) {
        record.read(spans);
    };
}
void GgufModelSource::read_range_into(std::string_view name, uint64_t offset, std::byte* destination, std::size_t size) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    const auto& record = it->second;
    record.read(offset, destination, size);
}
const std::vector<std::string>& GgufModelSource::assets() const noexcept { return impl_->assets; }
bool GgufModelSource::has_asset(std::string_view name) const noexcept { return impl_->asset_data.count(std::string(name)) != 0; }
std::vector<std::byte> GgufModelSource::read_asset(std::string_view name) const { return impl_->asset_data.at(std::string(name)); }
std::optional<ModelGraph> GgufModelSource::model_graph() const { return std::nullopt; }

} // namespace mfq
