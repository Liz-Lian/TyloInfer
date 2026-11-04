#include "hf.h"

#include "mlx_names.h"
#include "nlohmann/json.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tyloi::models {

using json = nlohmann::json;

Names hf_names(
    std::string_view architecture,
    const std::vector<std::string>& stored_names,
    std::optional<std::string_view> source_map,
    std::optional<std::string_view> config) {
    if (source_map) {
        json payload;
        try {
            payload = json::parse(
                source_map->data(), source_map->data() + source_map->size());
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
        Names result;
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

    if (!config) return {};
    std::vector<std::string> names;
    names.reserve(stored_names.size());
    std::unordered_map<std::string, std::string> normalized_to_source;
    for (const auto& stored : stored_names) {
        auto normalized = mfq::mlx_hf_tensor_name(stored);
        if (!normalized_to_source.emplace(normalized, stored).second)
            throw std::invalid_argument("duplicate normalized HF tensor: " + normalized);
        names.push_back(std::move(normalized));
    }
    auto compatibility = legacy_names(architecture, *config, names);
    for (auto& [canonical, stored] : compatibility.canonical_to_stored)
        stored = normalized_to_source.at(stored);
    std::unordered_set<std::string> claimed;
    for (const auto& [canonical, stored] :
         compatibility.canonical_to_stored) {
        static_cast<void>(canonical);
        claimed.insert(stored);
    }
    for (const auto& stored : stored_names) {
        if (claimed.find(stored) == claimed.end()) {
            compatibility.canonical_to_stored.emplace(stored, stored);
        }
    }
    return compatibility;
}

} // namespace tyloi::models
