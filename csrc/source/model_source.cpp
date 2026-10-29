#include "model_source.h"
#include "read.h"

#include "hf_model_source.h"
#include "gguf_model_source.h"
#include "mfq_model_source.h"
#include "tq_fixture_source.h"
#include "compat/mfq_legacy_model_graph.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>

namespace mfq {

std::shared_ptr<const ModelSource> open_model_source(
        const std::filesystem::path& path) {
    for (const char* environment :
         {"TYLOINFER_TENSOR_OVERLAY", "TYLOINFER_EXPERT_OVERLAY"}) {
        const char* overlay = std::getenv(environment);
        if (overlay != nullptr && overlay[0] != '\0') {
            throw std::runtime_error(
                std::string(environment) +
                " is unsupported by the canonical ModelSource path");
        }
    }

    const auto filename=path.filename().u8string();
    const std::string fixture_suffix=".tq-fixture.json";
    if(filename.size()>=fixture_suffix.size() &&
       filename.compare(filename.size()-fixture_suffix.size(),fixture_suffix.size(),fixture_suffix)==0)
        return open_tq_fixture_source(path);
    std::error_code error;
    if (std::filesystem::is_directory(path, error) && !error) {
        return std::make_shared<HfModelSource>(path);
    }
    if (is_gguf_file(path)) return std::make_shared<GgufModelSource>(path);
    return std::make_shared<MfqModelSource>(path);
}

std::string ModelSource::model_config_json(const std::string& external_path) const {
    if (!external_path.empty()) {
        std::ifstream input(external_path, std::ios::binary);
        if (!input) throw std::runtime_error("cannot open config: " + external_path);
        return {std::istreambuf_iterator<char>(input), {}};
    }
    if (!has_asset(kModelConfigAsset)) {
        throw std::runtime_error("model source has no model config");
    }
    const auto bytes = read_asset(kModelConfigAsset);
    return {
        reinterpret_cast<const char*>(bytes.data()),
        bytes.size(),
    };
}

ModelGraph ModelSource::resolved_model_graph() const {
    if (auto graph = model_graph()) return *graph;
    const auto config = model_config_json();
    return synthesize_legacy_model_graph(
        architecture(), config,
        [&](std::string_view name) { return find_tensor(name) != nullptr; });
}

const mfq::TensorMetadata& require_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto* tensor = source.find_tensor(name);
    if (tensor == nullptr) {
        throw std::runtime_error("missing tensor: " + std::string(name));
    }
    return *tensor;
}

bool has_tensor(
        const mfq::ModelSource& source,
        std::string_view name) noexcept {
    return source.find_tensor(name) != nullptr;
}

bool has_tensor_prefix(
        const mfq::ModelSource& source,
        std::string_view prefix) {
    for (const auto& tensor : source.tensors()) {
        if (tensor.name == prefix ||
            (tensor.name.size() > prefix.size() &&
             tensor.name.compare(0, prefix.size(), prefix) == 0 &&
             tensor.name[prefix.size()] == '.')) {
            return true;
        }
    }
    return false;
}

std::vector<std::uint8_t> read_tensor(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto& tensor = require_tensor(source, name);
    if (tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model tensor is too large to read");
    }
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(tensor.nbytes));
    source.read_range_into(
        name, 0, reinterpret_cast<std::byte*>(result.data()), result.size());
    return result;
}

std::vector<std::uint8_t> read_asset(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    std::vector<std::uint8_t> result(bytes.size());
    if (!bytes.empty()) {
        std::memcpy(result.data(), bytes.data(), bytes.size());
    }
    return result;
}

std::string read_asset_text(
        const mfq::ModelSource& source,
        std::string_view name) {
    const auto bytes = source.read_asset(name);
    return {
        reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

} // namespace mfq
