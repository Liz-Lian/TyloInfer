#include "hf_source.h"

#include "engine/weights/source/assets.h"
#include "engine/weights/source/json.h"
#include "engine/weights/source/safetensors.h"
#include "source/read.h"

#include "engine/weights/encoding/dense_blob.h"
#include "engine/weights/encoding/fp8_sq_blob.h"
#include "engine/weights/encoding/mx_blob.h"
#include "engine/weights/encoding/quant/mfe_blob.h"
#include "engine/models/common/hf.h"
#include "engine/weights/source/hf.h"
#include "engine/weights/mlx.h"
#include "engine/models/common/mlx_names.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <charconv>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace mfq::models {
namespace {

using json = nlohmann::json;

constexpr std::string_view kAssetPrefix = "__mfq_asset__/";

constexpr std::uint32_t kMaxRecords = std::uint32_t{1} << 20;

bool ends_with(std::string_view value, std::string_view suffix) {
    return value.size() >= suffix.size() &&
        value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(what) + " byte size overflow");
    }
    return left + right;
}

std::uint64_t checked_product(
    const std::vector<std::int64_t>& shape,
    std::uint64_t item_size,
    const std::string& name) {
    auto result = item_size;
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint64_t>::max() / result) {
            throw std::runtime_error("invalid Safetensors tensor shape: " + name);
        }
        result *= static_cast<std::uint64_t>(dimension);
    }
    return result;
}

std::uint64_t dense_item_size(std::string_view dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "I64") return 8;
    return 0;
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
    const auto find_scale = [&](const std::string& source_name,
                                const std::string& canonical_name,
                                const auto& accepted) {
        for (const auto& candidate :
             scale_candidates(source_name, canonical_name)) {
            const auto* tensor = impl_->files.find(candidate);
            if (tensor != nullptr && accepted(tensor->dtype)) {
                return candidate;
            }
        }
        return std::string{};
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
        if (!values->shape) {
            throw std::runtime_error("HF tensor shape is missing: " + source_name);
        }

        hf::Record record;
        record.metadata.name = canonical;
        record.metadata.stored_dtype = values->dtype;
        record.metadata.shape = values->shape;
        const auto item_size = dense_item_size(values->dtype);
        if (values->dtype == "U32") {
            const auto binding = mlx_affine_binding(
                impl_->files, mlx_quantization, *values);
            const auto& layout = binding.layout;
            record.metadata.dtype = "MLX-AFFINE";
            record.metadata.shape = layout.shape;
            record.segments.push_back({0, binding.prefix, {}, 0, 0});
            record.segments.push_back({layout.prefix_bytes, {}, source_name, 0, layout.values_bytes});
            record.segments.push_back({layout.scales_offset(), {}, binding.scales, 0, layout.scales_bytes});
            record.segments.push_back({layout.biases_offset(), {}, binding.biases, 0, layout.scales_bytes});
            record.metadata.nbytes = layout.total_bytes();
            consumed_scales.insert(binding.scales); consumed_scales.insert(binding.biases);
        } else if (item_size != 0) {
            if (checked_product(*values->shape, item_size, source_name) !=
                values->nbytes) {
                throw std::runtime_error(
                    "dense Safetensors byte size mismatch: " + source_name);
            }
            record.metadata.dtype = values->dtype;
            record.metadata.shape = mlx_hf_tensor_shape(canonical, *values->shape);
            auto prefix = dense::prefix(*record.metadata.shape);
            record.segments.push_back({0, std::move(prefix), {}, 0, 0});
            const auto values_offset = record.segments.front().inline_bytes.size();
            record.segments.push_back({
                values_offset, {}, source_name, 0, values->nbytes});
            record.metadata.nbytes = checked_add(
                values_offset, values->nbytes, "dense tensor");
        } else if (values->dtype == "I8" || values->dtype == "F8_E4M3" ||
                   values->dtype == "F8_E4M3FN") {
            if (checked_product(*values->shape, 1, source_name) !=
                values->nbytes) {
                throw std::runtime_error(
                    "native FP8/MX values byte size mismatch: " + source_name);
            }
            auto logical_shape = *values->shape;
            std::string scale_name;
            if (values->dtype != "I8") {
                scale_name = find_scale(
                    source_name, canonical, [](std::string_view dtype) {
                        return dtype == "BF16" || dtype == "F16" ||
                            dtype == "F32";
                    });
            }
            if (!scale_name.empty()) {
                record.metadata.dtype = "FP8-128SQ";
            } else {
                scale_name = find_scale(
                    source_name, canonical, [](std::string_view dtype) {
                        return dtype == "F8_E8M0";
                    });
                if (scale_name.empty()) {
                    throw std::runtime_error(
                        "native FP8/MX tensor has no supported scale binding: " +
                        source_name);
                }
                if (values->dtype == "I8") {
                    if (logical_shape.size() != 2 || logical_shape[1] >
                            std::numeric_limits<std::int64_t>::max() / 2) {
                        throw std::runtime_error(
                            "invalid native MXFP4 tensor shape: " + source_name);
                    }
                    logical_shape[1] *= 2;
                    record.metadata.dtype = "MXFP4";
                } else {
                    record.metadata.dtype = "MXFP8";
                }
            }
            record.metadata.stored_dtype = record.metadata.dtype;
            record.metadata.shape = logical_shape;
            const auto* scales = impl_->files.find(scale_name);
            if (scales == nullptr || !scales->shape) {
                throw std::runtime_error("native FP8/MX scale shape is missing: " +
                                         scale_name);
            }
            const auto scale_item_size = dense_item_size(scales->dtype);
            const bool block_fp8 = record.metadata.dtype == "FP8-128SQ";
            const auto expected_scale_bytes = block_fp8
                ? checked_product(*scales->shape, scale_item_size, scale_name)
                : checked_product(*scales->shape, 1, scale_name);
            if ((block_fp8 && scale_item_size == 0) ||
                expected_scale_bytes != scales->nbytes) {
                throw std::runtime_error(
                    "native FP8/MX scale byte size mismatch: " + scale_name);
            }
            auto prefix = record.metadata.dtype == "FP8-128SQ"
                ? fp8sq::block128_prefix(
                    logical_shape, scales->dtype, *scales->shape, source_name)
                : mx::prefix(record.metadata.dtype, logical_shape,
                             *values->shape, *scales->shape, source_name);
            const auto values_offset = prefix.size();
            const auto scales_offset = checked_add(
                values_offset, values->nbytes, "native FP8/MX tensor");
            record.segments.push_back({0, std::move(prefix), {}, 0, 0});
            record.segments.push_back({
                values_offset, {}, source_name, 0, values->nbytes});
            record.segments.push_back({
                scales_offset, {}, scale_name, 0, scales->nbytes});
            record.metadata.nbytes = checked_add(
                scales_offset, scales->nbytes, "native FP8/MX tensor");
            consumed_scales.insert(scale_name);
        } else {
            throw std::runtime_error("unsupported full-precision HF dtype " +
                                     values->dtype + ": " + source_name);
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
        std::map<std::size_t, std::string> experts;
    };
    std::map<std::string, Projection> projections;
    const auto collect = [&](std::string_view canonical) {
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
        projections[aggregate].experts.emplace(expert, std::string(canonical));
    };
    for (const auto& [name, unused] : impl_->records) {
        static_cast<void>(unused);
        collect(name);
    }

    for (const auto& [name, projection] : projections) {
        if (impl_->records.find(name) != impl_->records.end() ||
            projection.experts.empty()) continue;
        const auto expert_count = projection.experts.rbegin()->first + 1;
        if (projection.experts.size() != expert_count) {
            throw std::runtime_error(
                "native HF expert projection has missing expert IDs: " + name);
        }
        hf::Record aggregate;
        aggregate.metadata.name = name;
        aggregate.metadata.dtype = "MFE";
        aggregate.metadata.stored_dtype = "MFE";
        std::string dtype;
        std::vector<std::int64_t> shape;
        std::uint64_t logical_offset = 0;
        const auto append_inline = [&](std::vector<std::byte> bytes) {
            const auto count = bytes.size();
            aggregate.segments.push_back({
                logical_offset, std::move(bytes), {}, 0, count});
            logical_offset = checked_add(
                logical_offset, count, "virtual MFE metadata");
        };
        const auto append_source_segment = [&](const hf::Segment& segment) {
            if (!segment.inline_bytes.empty()) {
                append_inline(segment.inline_bytes);
            } else {
                aggregate.segments.push_back({
                    logical_offset, {}, segment.source_name,
                    segment.source_offset, segment.nbytes});
                logical_offset = checked_add(
                    logical_offset, segment.nbytes, "virtual MFE tensor");
            }
        };

        for (const auto& [expert, tensor_name] : projection.experts) {
            const auto& record = impl_->records.at(tensor_name);
            if (!record.metadata.shape || record.metadata.shape->size() != 2 ||
                record.segments.empty()) {
                throw std::runtime_error(
                    "native HF expert must be rank two: " + tensor_name);
            }
            if (dtype.empty()) {
                dtype = record.metadata.dtype;
                shape = *record.metadata.shape;
                if (shape[0] <= 0 || shape[1] <= 0 ||
                    shape[0] > std::numeric_limits<std::uint32_t>::max() ||
                    shape[1] > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                        "native HF expert geometry exceeds MFE: " + tensor_name);
                }
                const auto output = static_cast<std::uint32_t>(shape[0]);
                const auto input = static_cast<std::uint32_t>(shape[1]);
                append_inline(mfe::header(
                    static_cast<std::uint32_t>(expert_count), output, input,
                    static_cast<std::uint32_t>(expert_count)));
            } else if (dtype != record.metadata.dtype ||
                       shape != *record.metadata.shape) {
                throw std::runtime_error(
                    "native HF expert projection has mixed geometry: " + name);
            }

            auto pool = mfe::expert_prefix(
                static_cast<std::int32_t>(expert), dtype, record.metadata.nbytes);
            pool.insert(pool.end(), record.segments.front().inline_bytes.begin(),
                        record.segments.front().inline_bytes.end());
            append_inline(std::move(pool));
            for (std::size_t index = 1; index < record.segments.size(); ++index) {
                append_source_segment(record.segments[index]);
            }
        }
        aggregate.metadata.nbytes = logical_offset;
        impl_->records.emplace(name, std::move(aggregate));
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
