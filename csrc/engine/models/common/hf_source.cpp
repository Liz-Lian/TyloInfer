#include "hf_source.h"

#include "engine/weights/source/assets.h"
#include "engine/weights/source/json.h"
#include "engine/weights/source/safetensors.h"
#include "engine/weights/source/read.h"

#include "engine/models/common/hf.h"
#include "engine/models/common/mlx_names.h"
#include "engine/weights/hf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <stdexcept>

namespace mfq::models {
namespace {

using json = nlohmann::json;

constexpr std::string_view kAssetPrefix = "__mfq_asset__/";

constexpr std::uint32_t kMaxRecords = std::uint32_t{1} << 20;

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

struct HfSource::Impl {
    explicit Impl(std::filesystem::path root, std::string source_prefix)
        : files(std::move(root), std::move(source_prefix)) {}

    Safetensors files;
    source::Assets asset_data;
    std::vector<std::string> asset_names;
    std::string arch;
    std::unordered_map<std::string, std::string> metadata;
    TensorLayout layout;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, std::size_t> tensor_indices;
    std::unordered_map<std::string, hf::Record> records;
};

HfSource::HfSource(std::filesystem::path root, std::string source_prefix)
    : impl_(std::make_unique<Impl>(std::move(root), source_prefix)) {
    std::vector<std::string> raw_names;
    raw_names.reserve(impl_->files.records().size());
    for (const auto& [name, unused] : impl_->files.records()) {
        static_cast<void>(unused);
        raw_names.push_back(name);
    }
    std::sort(raw_names.begin(), raw_names.end());
    for (const auto& name : raw_names) {
        if (name.empty()) {
            throw std::runtime_error("invalid HF canonical tensor mapping: " + name);
        }
    }

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
        impl_->arch = model_type->get<std::string>() + "-hf-full-mfq";
        impl_->metadata.emplace("source.format", "hf-safetensors");
        impl_->metadata.emplace("source.precision", "native");
        impl_->asset_data.add(std::string(kModelConfigAsset), parsed.dump());
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
            impl_->asset_data.add(std::string(kAssetPrefix) + std::string(asset_name), path);
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
    impl_->asset_data.add_directory(impl_->files.root() / ".mfq-assets");
    if (const auto* configured = std::getenv("TYLOINFER_RUNTIME_ASSET_DIRECTORY");
        configured != nullptr && *configured != '\0') {
        impl_->asset_data.add_directory(configured);
    }
    impl_->asset_names = impl_->asset_data.names();
    const bool has_explicit_source_map = has_asset(kHfSourceMapAsset);
    std::optional<std::string> source_map;
    std::optional<std::string> config_text;
    if (has_explicit_source_map) {
        source_map = read_asset_text(*this, kHfSourceMapAsset);
    } else if (has_asset(kModelConfigAsset)) {
        config_text = read_asset_text(*this, kModelConfigAsset);
    }
    auto mapping = hf_names(impl_->arch, raw_names, source_map, config_text);
    impl_->layout = mapping.layout;
    auto& canonical_to_source = mapping.canonical_to_stored;
    if (!source_prefix.empty()) {
        for (auto it = canonical_to_source.begin(); it != canonical_to_source.end();) {
            if (it->second.rfind(source_prefix, 0) != 0) it = canonical_to_source.erase(it);
            else ++it;
        }
    }
    if (canonical_to_source.empty()) {
        for (const auto& name : raw_names) {
            canonical_to_source.emplace(name, name);
        }
    }
    std::unordered_map<std::string, std::string> source_to_canonical;
    for (const auto& [canonical, source] : canonical_to_source) {
        if (canonical.empty() || source.empty() ||
            impl_->files.find(source) == nullptr) {
            throw std::runtime_error("invalid HF canonical tensor mapping: " +
                                     canonical);
        }
        const auto [found, inserted] =
            source_to_canonical.emplace(source, canonical);
        if (!inserted && found->second != canonical) {
            throw std::runtime_error(
                "native HF source tensor has multiple canonical owners: " + source);
        }
    }

    const auto scale_candidates = [&](
            const std::string& source_name,
            const std::string& canonical_name) {
        std::vector<std::string> candidates;
        if (ends_with(canonical_name, ".weight")) {
            const auto base = canonical_name.substr(0, canonical_name.size() - 7);
            for (const auto suffix : {".weight_scale_inv", ".weight_scale",
                                      ".weight_scale_2", ".scale"}) {
                const auto found = canonical_to_source.find(base + suffix);
                if (found != canonical_to_source.end()) {
                    candidates.push_back(found->second);
                }
            }
        }
        if (!has_explicit_source_map && ends_with(source_name, ".weight")) {
            const auto base = source_name.substr(0, source_name.size() - 7);
            candidates.push_back(source_name + "_scale_inv");
            candidates.push_back(source_name + "_scale");
            candidates.push_back(base + ".weight_scale_inv");
            candidates.push_back(base + ".scale");
            candidates.push_back(base + ".weight_scale");
            candidates.push_back(base + ".weight_scale_2");
        }
        return candidates;
    };
    std::unordered_set<std::string> consumed_scales;
    json mlx_quantization;
    if (has_asset(kModelConfigAsset)) {
        const auto config = read_asset(kModelConfigAsset);
        const auto parsed = json::parse(reinterpret_cast<const char*>(config.data()),
            reinterpret_cast<const char*>(config.data() + config.size()));
        mlx_quantization = parsed.value("quantization", json{});
    }
    std::vector<std::pair<std::string, std::string>> aliases(
        canonical_to_source.begin(), canonical_to_source.end());
    std::sort(aliases.begin(), aliases.end());
    for (const auto& [canonical, source_name] : aliases) {
        const auto* values = impl_->files.find(source_name);
        if (values == nullptr || values->dtype == "F8_E8M0") continue;
        const auto dense_shape = values->shape
            ? mlx_hf_tensor_shape(canonical, *values->shape)
            : std::vector<std::int64_t>{};
        auto record = hf::prepare(
            impl_->files, *values, canonical, dense_shape,
            scale_candidates(source_name, canonical), mlx_quantization);
        for (const auto& segment : record.segments) {
            if (!segment.source_name.empty() &&
                segment.source_name != source_name) {
                consumed_scales.insert(segment.source_name);
            }
        }
        if (!impl_->records.emplace(canonical, std::move(record)).second) {
            throw std::runtime_error("duplicate canonical HF tensor: " + canonical);
        }
    }
    for (const auto& scale : consumed_scales) {
        const auto canonical = source_to_canonical.find(scale);
        if (canonical != source_to_canonical.end()) {
            impl_->records.erase(canonical->second);
        }
    }
    for (const auto& name : raw_names) {
        const auto& tensor = impl_->files.records().at(name).metadata;
        if (tensor.dtype == "F8_E8M0" &&
            source_to_canonical.find(tensor.name) != source_to_canonical.end() &&
            consumed_scales.find(tensor.name) == consumed_scales.end()) {
            throw std::runtime_error("orphan E8M0 scale tensor: " + tensor.name);
        }
    }

    struct Projection {
        std::map<std::size_t, const hf::Record*> experts;
    };
    std::map<std::string, Projection> projections;
    const auto collect = [&](const hf::Record& record) {
        const std::string_view canonical = record.metadata.name;
        constexpr std::string_view marker = ".mlp.experts.";
        const auto marker_offset = canonical.find(marker);
        if (marker_offset == std::string_view::npos) return;
        const auto expert_begin = marker_offset + marker.size();
        const auto expert_end = canonical.find('.', expert_begin);
        if (expert_end == std::string_view::npos || expert_end == expert_begin) return;
        std::size_t expert = 0;
        const auto id = canonical.substr(expert_begin, expert_end - expert_begin);
        const auto parsed = std::from_chars(
            id.data(), id.data() + id.size(), expert);
        if (parsed.ec != std::errc{} || parsed.ptr != id.data() + id.size()) return;
        const auto projection_end = canonical.find('.', expert_end + 1);
        if (projection_end == std::string_view::npos ||
            canonical.substr(projection_end) != ".weight") return;
        const auto projection = canonical.substr(
            expert_end + 1, projection_end - expert_end - 1);
        if (projection != "gate" && projection != "up" && projection != "down") {
            return;
        }
        const auto aggregate = std::string(canonical.substr(0, marker_offset)) +
            std::string(marker) + std::string(projection) + ".weight";
        projections[aggregate].experts.emplace(expert, &record);
    };
    for (const auto& [unused, record] : impl_->records) {
        static_cast<void>(unused);
        collect(record);
    }

    for (const auto& [name, projection] : projections) {
        if (impl_->records.find(name) != impl_->records.end() ||
            projection.experts.empty()) continue;
        impl_->records.emplace(name, hf::experts(name, projection.experts));
    }

    if (impl_->records.size() > kMaxRecords) {
        throw std::runtime_error("HF canonical tensor count exceeds the record limit");
    }
    std::vector<std::string> names;
    names.reserve(impl_->records.size());
    for (const auto& [name, unused] : impl_->records) {
        static_cast<void>(unused);
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    for (const auto& name : names) {
        impl_->tensor_indices.emplace(name, impl_->tensors.size());
        impl_->tensors.push_back(impl_->records.at(name).metadata);
    }
}

HfSource::~HfSource() = default;
HfSource::HfSource(HfSource&&) noexcept = default;
HfSource& HfSource::operator=(HfSource&&) noexcept = default;

const std::vector<std::filesystem::path>& HfSource::source_paths() const noexcept {
    return impl_->files.paths();
}
void HfSource::drop_file_cache() const noexcept {
    impl_->files.drop_file_cache();
}
std::string_view HfSource::architecture() const noexcept {
    return impl_->arch;
}
const std::unordered_map<std::string, std::string>&
HfSource::metadata() const noexcept {
    return impl_->metadata;
}
const std::vector<TensorMetadata>& HfSource::tensors() const noexcept {
    return impl_->tensors;
}
const TensorMetadata* HfSource::find_tensor(
    std::string_view name) const noexcept {
    const auto found = impl_->tensor_indices.find(std::string(name));
    return found == impl_->tensor_indices.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

TensorLayout HfSource::tensor_layout() const noexcept {
    return impl_->layout;
}

std::optional<TensorFileRange> HfSource::tensor_file_range_at(std::string_view name,
    std::uint64_t offset, std::size_t size) const {
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end()) throw std::out_of_range("HF tensor not found: " + std::string(name));
    return found->second.file_range(impl_->files, offset, size);
}

void HfSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    const auto found = impl_->records.find(std::string(name));
    if (found == impl_->records.end()) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    found->second.read(
        impl_->files, relative_offset, destination, size);
}

const std::vector<std::string>& HfSource::assets() const noexcept {
    return impl_->asset_names;
}
bool HfSource::has_asset(std::string_view name) const noexcept {
    return impl_->asset_data.has(name);
}
std::vector<std::byte> HfSource::read_asset(std::string_view name) const {
    return impl_->asset_data.read(name);
}
std::optional<ModelGraph> HfSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq::models
