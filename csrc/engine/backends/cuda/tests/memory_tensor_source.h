#pragma once

#include "engine/models/source.h"

#include <cstring>

struct MemoryTensorSource final : tyloi::ModelSource {
    std::vector<uint8_t> bytes;
    std::vector<tyloi::TensorMetadata> records{{"weight", "F32", "F32", std::nullopt, 0}};
    std::vector<std::filesystem::path> paths;
    std::unordered_map<std::string, std::string> meta;
    std::vector<std::string> asset_names;
    const std::vector<std::filesystem::path>& source_paths() const noexcept override { return paths; }
    std::string_view architecture() const noexcept override { return {}; }
    const std::unordered_map<std::string, std::string>& metadata() const noexcept override { return meta; }
    const std::vector<tyloi::TensorMetadata>& tensors() const noexcept override { return records; }
    const tyloi::TensorMetadata* find_tensor(std::string_view name) const noexcept override {
        for(const auto& record:records)if(record.name==name)return &record;
        return nullptr;
    }
    void read_range_into(std::string_view, uint64_t offset, std::byte* output, size_t size) const override {
        if (offset > bytes.size() || size > bytes.size() - offset)
            throw std::out_of_range("memory fixture range");
        if (size) std::memcpy(output, bytes.data() + offset, size);
    }
    const std::vector<std::string>& assets() const noexcept override { return asset_names; }
    bool has_asset(std::string_view) const noexcept override { return false; }
    std::vector<std::byte> read_asset(std::string_view) const override { throw std::runtime_error("no assets"); }
    std::optional<tyloi::ModelGraph> model_graph() const override { return std::nullopt; }
};
