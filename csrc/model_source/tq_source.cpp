#include "model_source/tq_source.h"

#include "nlohmann/json.hpp"

#include <climits>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_set>

namespace mfq {
namespace {

std::vector<std::string> row_parts(
        const ModelSource& source, std::string_view name) {
    const auto tensor_name = std::string(name);
    for (const auto* key : {"tq_row_shards", "tq.segments"}) {
        const auto entry = source.metadata().find(key);
        if (entry == source.metadata().end()) continue;
        const auto tables = nlohmann::json::parse(entry->second);
        if (!tables.is_object()) {
            throw std::invalid_argument("invalid TQ row-shard manifest");
        }
        if (!tables.contains(tensor_name)) continue;
        if (std::string_view(key) == "tq_row_shards" && source.find_tensor(name)) {
            throw std::invalid_argument("ambiguous TQ row table: " + tensor_name);
        }
        const auto names = tables.at(tensor_name).get<std::vector<std::string>>();
        if (names.empty()) {
            throw std::invalid_argument("empty TQ row table: " + tensor_name);
        }
        return names;
    }
    return {tensor_name};
}

std::uint64_t expected_rows(const std::vector<std::int64_t>& shape,
                            std::string_view name) {
    std::uint64_t rows = 1;
    for (std::size_t index = 0; index + 1 < shape.size(); ++index) {
        const auto dimension = static_cast<std::uint64_t>(shape[index]);
        if (rows > UINT64_MAX / dimension) {
            throw std::overflow_error("TQ logical row count exceeds bounds: " +
                                      std::string(name));
        }
        rows *= dimension;
    }
    return rows;
}

} // namespace

std::vector<std::int64_t> tq_shape(
        const ModelSource& source, std::string_view name) {
    const auto* record = source.find_tensor(name);
    const auto tensor_name = std::string(name);
    if (record && record->dtype != "TQ") {
        throw std::invalid_argument("expected TQ tensor: " + std::string(name));
    }
    std::vector<std::int64_t> shape;
    bool found = false;
    for (const auto* key : {"tq_shapes", "tensor_shapes"}) {
        const auto metadata = source.metadata().find(key);
        if (metadata == source.metadata().end()) continue;
        const auto shapes = nlohmann::json::parse(metadata->second);
        if (!shapes.is_object()) {
            throw std::invalid_argument("invalid TQ shape metadata");
        }
        if (shapes.contains(tensor_name)) {
            shape = shapes.at(tensor_name).get<std::vector<std::int64_t>>();
            found = true;
            break;
        }
    }
    if (!found && record && record->shape) shape = *record->shape;
    if (shape.empty() && record) {
        return mfq::tq_tensor_shape(
            record->nbytes, source.tensor_reader(name));
    }
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::invalid_argument(
            "TQ logical tensor shape must have rank 2 or 3: " + std::string(name));
    }
    std::uint64_t count = 1;
    for (const auto dimension : shape) {
        if (dimension < 1 || count > UINT64_MAX / std::uint64_t(dimension)) {
            throw std::invalid_argument(
                "invalid TQ logical tensor shape: " + std::string(name));
        }
        count *= std::uint64_t(dimension);
    }
    return shape;
}

std::shared_ptr<TqTable> tq_table(
        const ModelSource& source, std::string_view name) {
    const auto shape = tq_shape(source, name);
    const auto columns = shape.back();
    const auto names = row_parts(source, name);
    std::unordered_set<std::string> unique;
    std::vector<std::shared_ptr<TqRows>> segments;
    std::uint64_t storage_bytes = 0;
    for (const auto& part : names) {
        const auto* record = source.find_tensor(part);
        if (!unique.insert(part).second || !record || record->dtype != "TQ") {
            throw std::invalid_argument("invalid TQ row segment: " + part);
        }
        auto physical = mfq::tq_row_segments(
            record->nbytes, source.tensor_reader(part), columns);
        if (storage_bytes > UINT64_MAX - record->nbytes) {
            throw std::overflow_error("TQ row table size exceeds bounds");
        }
        storage_bytes += record->nbytes;
        for (auto& rows : physical) segments.push_back(std::move(rows));
    }
    return std::make_shared<TqTable>(
        std::move(segments), columns, expected_rows(shape, name), storage_bytes);
}

bool is_tq_table(
        const ModelSource& source, std::string_view name) {
    const auto* record = source.find_tensor(name);
    const auto tensor_name = std::string(name);
    for (const auto* key : {"tq_row_shards", "tq.segments"}) {
        const auto entry = source.metadata().find(key);
        if (entry == source.metadata().end()) continue;
        const auto tables = nlohmann::json::parse(entry->second);
        if (!tables.is_object()) {
            throw std::invalid_argument("invalid TQ row-shard manifest");
        }
        if (tables.contains(tensor_name)) {
            if (std::string_view(key) == "tq_row_shards" && record) {
                throw std::invalid_argument(
                    "ambiguous TQ row table: " + std::string(name));
            }
            return true;
        }
    }
    return record && record->dtype == "TQ";
}

std::array<std::int64_t, 2> tq_projection_shape(
        const ModelSource& source, std::string_view name) {
    const auto shape = tq_shape(source, name);
    if (shape.size() != 2) {
        throw std::invalid_argument(
            "ordinary TQ projection must be a matrix: " + std::string(name));
    }
    const auto* record = source.find_tensor(name);
    if (!record) {
        throw std::invalid_argument("missing TQ projection: " + std::string(name));
    }
    (void)mfq::tq_tensor_shape(
        record->nbytes, source.tensor_reader(name), shape);
    return {shape[0], shape[1]};
}

} // namespace mfq
