#include "hf.h"

#include "mlx_names.h"
#include "source/model_source.h"
#include "nlohmann/json.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mfq::models {

using json = nlohmann::json;

MfqLegacyTensorAliases hf_names(const ModelSource& source) {
    if (source.has_asset(kHfSourceMapAsset)) {
        const auto bytes = source.read_asset(kHfSourceMapAsset);
        json payload;
        try {
            payload = json::parse(
                reinterpret_cast<const char*>(bytes.data()),
                reinterpret_cast<const char*>(bytes.data() + bytes.size()));
        } catch (const json::exception& error) {
            throw std::runtime_error(
                std::string("invalid native HF source map: ") + error.what());
        }
        const auto aliases = payload.find("canonical_to_source");
        if (!payload.is_object() ||
            payload.value("schema", std::string{}) != "mfq.hf-source-map" ||
            payload.value("version", 0) != 1 || aliases == payload.end() ||
            !aliases->is_object()) {
            throw std::runtime_error("unsupported native HF source map contract");
        }
        MfqLegacyTensorAliases result;
        for (const auto& [canonical, raw_source] : aliases->items()) {
            if (canonical.empty() || !raw_source.is_string() ||
                raw_source.get_ref<const std::string&>().empty() ||
                !result.canonical_to_stored.emplace(
                    canonical, raw_source.get<std::string>()).second) {
                throw std::runtime_error(
                    "native HF source map has an invalid entry");
            }
        }
        return result;
    }

    if (!source.has_asset(kModelConfigAsset)) return {};
    const auto config = source.read_asset(kModelConfigAsset);
    std::vector<std::string> names;
    names.reserve(source.tensors().size());
    std::unordered_map<std::string, std::string> normalized_to_source;
    for (const auto& tensor : source.tensors()) {
        auto normalized = mlx_hf_tensor_name(tensor.name);
        if (!normalized_to_source.emplace(normalized, tensor.name).second)
            throw std::invalid_argument("duplicate normalized HF tensor: " + normalized);
        names.push_back(std::move(normalized));
    }
    auto compatibility = make_legacy_tensor_aliases(
        source.architecture(),
        std::string_view(reinterpret_cast<const char*>(config.data()), config.size()),
        names);
    for (auto& [canonical, stored] : compatibility.canonical_to_stored)
        stored = normalized_to_source.at(stored);
    std::unordered_set<std::string> claimed;
    for (const auto& [canonical, stored] :
         compatibility.canonical_to_stored) {
        static_cast<void>(canonical);
        claimed.insert(stored);
    }
    for (const auto& tensor : source.tensors()) {
        if (claimed.find(tensor.name) == claimed.end()) {
            compatibility.canonical_to_stored.emplace(tensor.name, tensor.name);
        }
    }
    return compatibility;
}

} // namespace mfq::models
