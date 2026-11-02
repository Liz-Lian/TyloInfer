#include "hf_safetensors_source.h"

#include "engine/weights/source/json.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace mfq {
namespace {

using json = nlohmann::json;

constexpr std::string_view kAssetPrefix = "__mfq_asset__/";

} // namespace

struct HfSafetensorsSource::Impl {
    Impl(std::filesystem::path requested_root, std::string source_prefix)
        : files(std::move(requested_root), std::move(source_prefix)) {}

    Safetensors files;
    std::string architecture;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, std::size_t> tensor_indices;
    std::unordered_map<std::string, std::string> exposed_to_source;
    std::unordered_map<std::string, std::filesystem::path> asset_paths;
    std::unordered_map<std::string, std::vector<std::byte>> inline_assets;
    std::vector<std::string> assets;
};

HfSafetensorsSource::HfSafetensorsSource(
    std::filesystem::path requested_root,
    std::unordered_map<std::string, std::string> canonical_to_source,
    std::string source_prefix)
    : impl_(std::make_unique<Impl>(
          std::move(requested_root), std::move(source_prefix))) {

    if (canonical_to_source.empty()) {
        for (const auto& [name, unused] : impl_->files.records()) {
            static_cast<void>(unused);
            canonical_to_source.emplace(name, name);
        }
    }
    std::vector<std::pair<std::string, std::string>> aliases(
        canonical_to_source.begin(), canonical_to_source.end());
    std::sort(aliases.begin(), aliases.end());
    std::unordered_set<std::string> claimed_sources;
    for (const auto& [canonical, source] : aliases) {
        const auto found = impl_->files.records().find(source);
        if (canonical.empty() || source.empty() ||
            found == impl_->files.records().end()) {
            throw std::runtime_error("invalid HF canonical tensor mapping: " +
                                     canonical);
        }
        if (!claimed_sources.insert(source).second ||
            !impl_->exposed_to_source.emplace(canonical, source).second) {
            throw std::runtime_error("ambiguous HF canonical tensor mapping: " +
                                     canonical);
        }
        auto metadata = found->second.metadata;
        metadata.name = canonical;
        impl_->tensor_indices.emplace(canonical, impl_->tensors.size());
        impl_->tensors.push_back(std::move(metadata));
    }

    const auto add_asset = [&](std::string name, const std::filesystem::path& path) {
        std::error_code asset_error;
        const auto resolved = std::filesystem::canonical(path, asset_error);
        if (asset_error || !std::filesystem::is_regular_file(resolved, asset_error) ||
            asset_error || std::filesystem::file_size(resolved, asset_error) == 0 ||
            asset_error) {
            throw std::runtime_error("invalid native HF runtime asset: " +
                                     path.string());
        }
        if (impl_->inline_assets.find(name) != impl_->inline_assets.end() ||
            !impl_->asset_paths.emplace(name, resolved).second) {
            throw std::runtime_error("duplicate native HF runtime asset: " +
                                     path.string());
        }
        impl_->assets.push_back(std::move(name));
    };
    const auto add_inline_asset = [&](std::string name, std::string payload) {
        std::vector<std::byte> bytes(payload.size());
        if (!payload.empty()) {
            std::memcpy(bytes.data(), payload.data(), payload.size());
        }
        if (impl_->asset_paths.find(name) != impl_->asset_paths.end() ||
            !impl_->inline_assets.emplace(name, std::move(bytes)).second) {
            throw std::runtime_error("duplicate native HF inline asset: " + name);
        }
        impl_->assets.push_back(std::move(name));
    };

    const auto config = impl_->files.root() / "config.json";
    if (std::filesystem::is_regular_file(config)) {
        auto parsed = source::parse_json(source::read_text(config), config);
        const auto inference_config =
            impl_->files.root() / "inference" / "config.json";
        if (std::filesystem::is_regular_file(inference_config)) {
            const auto supplemental = source::parse_json(
                source::read_text(inference_config), inference_config);
            if (!parsed.is_object() || !supplemental.is_object()) {
                throw std::runtime_error(
                    "HF model configuration files must contain JSON objects");
            }
            auto* target = &parsed;
            if (const auto text = parsed.find("text_config");
                text != parsed.end() && text->is_object()) {
                target = &*text;
            }
            const auto* source = &supplemental;
            if (const auto text = supplemental.find("text_config");
                text != supplemental.end() && text->is_object()) {
                source = &*text;
            }
            for (const auto& [key, value] : source->items()) {
                if (!target->contains(key)) (*target)[key] = value;
            }
        }
        const auto model_type = parsed.find("model_type");
        if (model_type == parsed.end() || !model_type->is_string() ||
            model_type->get_ref<const std::string&>().empty()) {
            throw std::runtime_error("HF config.json has no model_type");
        }
        impl_->architecture = model_type->get<std::string>() + "-hf-full-mfq";
        impl_->metadata.emplace("source.format", "hf-safetensors");
        impl_->metadata.emplace("source.precision", "native");
        add_inline_asset(std::string(kModelConfigAsset), parsed.dump());
    }
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 4>
        sidecars{{
            {"tokenizer.json", "hf/tokenizer.json"},
            {"tokenizer_config.json", "hf/tokenizer_config.json"},
            {"chat_template.jinja", "hf/chat_template.jinja"},
            {"generation_config.json", "hf/generation_config.json"},
        }};
    for (const auto& [filename, asset_name] : sidecars) {
        const auto path = impl_->files.root() / filename;
        if (std::filesystem::is_regular_file(path)) {
            add_asset(std::string(kAssetPrefix) + std::string(asset_name), path);
        }
    }
    const auto generation_path = impl_->files.root() / "generation_config.json";
    if (std::filesystem::is_regular_file(generation_path)) {
        const auto generation = source::parse_json(
            source::read_text(generation_path), generation_path);
        if (!generation.is_object()) {
            throw std::runtime_error(
                "HF generation_config.json must be an object");
        }
        static constexpr std::array<
            std::pair<std::string_view, std::string_view>, 7> aliases{{
            {"max_new_tokens", "max_tokens"},
            {"temperature", "temperature"},
            {"top_k", "top_k"},
            {"top_p", "top_p"},
            {"presence_penalty", "presence_penalty"},
            {"frequency_penalty", "frequency_penalty"},
            {"repetition_penalty", "repetition_penalty"},
        }};
        json chat = json::object();
        for (const auto& [source, target] : aliases) {
            const auto found = generation.find(std::string(source));
            if (found != generation.end() && !found->is_null()) {
                chat[std::string(target)] = *found;
            }
        }
        if (!chat.empty()) {
            impl_->metadata.emplace(
                "runtime.sampling.v1",
                json({
                    {"schema", "mfq.runtime.sampling"},
                    {"version", 1},
                    {"chat", std::move(chat)},
                    {"provenance", {{"source", "hf:generation_config.json"}}},
                }).dump());
        }
    }
    const auto add_directory = [&](const std::filesystem::path& directory) {
        std::error_code directory_error;
        if (!std::filesystem::is_directory(directory, directory_error) ||
            directory_error) return;
        const auto resolved = std::filesystem::canonical(directory, directory_error);
        if (directory_error) {
            throw std::runtime_error("cannot resolve native HF runtime asset directory: " +
                                     directory.string());
        }
        for (std::filesystem::recursive_directory_iterator iterator(
                 resolved, directory_error), end;
             !directory_error && iterator != end;
             iterator.increment(directory_error)) {
            if (!iterator->is_regular_file(directory_error) || directory_error) continue;
            const auto relative = std::filesystem::relative(
                iterator->path(), resolved, directory_error);
            if (directory_error || relative.empty() || relative.is_absolute() ||
                *relative.begin() == "..") {
                throw std::runtime_error("invalid native HF runtime asset path: " +
                                         iterator->path().string());
            }
            add_asset(std::string(kAssetPrefix) + relative.generic_string(),
                      iterator->path());
        }
        if (directory_error) {
            throw std::runtime_error("cannot enumerate native HF runtime assets: " +
                                     resolved.string());
        }
    };
    add_directory(impl_->files.root() / ".mfq-assets");
    if (const auto* configured = std::getenv("TYLOINFER_RUNTIME_ASSET_DIRECTORY");
        configured != nullptr && *configured != '\0') {
        add_directory(configured);
    }
    std::sort(impl_->assets.begin(), impl_->assets.end());
}

HfSafetensorsSource::~HfSafetensorsSource() = default;
HfSafetensorsSource::HfSafetensorsSource(HfSafetensorsSource&&) noexcept = default;
HfSafetensorsSource& HfSafetensorsSource::operator=(
    HfSafetensorsSource&&) noexcept = default;

const std::filesystem::path& HfSafetensorsSource::root() const noexcept {
    return impl_->files.root();
}
const std::vector<std::filesystem::path>&
HfSafetensorsSource::source_paths() const noexcept {
    return impl_->files.paths();
}
const Safetensors& HfSafetensorsSource::files() const noexcept {
    return impl_->files;
}
void HfSafetensorsSource::drop_file_cache() const noexcept {
    impl_->files.drop_file_cache();
}
std::string_view HfSafetensorsSource::architecture() const noexcept {
    return impl_->architecture;
}
const std::unordered_map<std::string, std::string>&
HfSafetensorsSource::metadata() const noexcept {
    return impl_->metadata;
}
const std::vector<TensorMetadata>& HfSafetensorsSource::tensors() const noexcept {
    return impl_->tensors;
}

const TensorMetadata* HfSafetensorsSource::find_tensor(
    std::string_view name) const noexcept {
    const auto found = impl_->tensor_indices.find(std::string(name));
    return found == impl_->tensor_indices.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

const HfTensorLocation& HfSafetensorsSource::location(
    std::string_view name) const {
    const auto alias = impl_->exposed_to_source.find(std::string(name));
    if (alias == impl_->exposed_to_source.end()) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    return impl_->files.records().at(alias->second).location;
}

void HfSafetensorsSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    const auto alias = impl_->exposed_to_source.find(std::string(name));
    if (alias == impl_->exposed_to_source.end()) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& source = impl_->files.records().at(alias->second);
    if (relative_offset > source.metadata.nbytes ||
        size > source.metadata.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    impl_->files.read(source.location.shard,
                      source.location.offset + relative_offset,
                      destination, size);
}

const std::vector<std::string>& HfSafetensorsSource::assets() const noexcept {
    return impl_->assets;
}

bool HfSafetensorsSource::has_asset(std::string_view name) const noexcept {
    const auto key = std::string(name);
    return impl_->inline_assets.find(key) != impl_->inline_assets.end() ||
        impl_->asset_paths.find(key) != impl_->asset_paths.end();
}

std::vector<std::byte> HfSafetensorsSource::read_asset(
    std::string_view name) const {
    const auto key = std::string(name);
    if (const auto inline_asset = impl_->inline_assets.find(key);
        inline_asset != impl_->inline_assets.end()) {
        return inline_asset->second;
    }
    const auto found = impl_->asset_paths.find(key);
    if (found == impl_->asset_paths.end()) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    return source::read_file(found->second);
}

std::optional<ModelGraph> HfSafetensorsSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq
