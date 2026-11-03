#pragma once
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace mfq {
class FileRangeReader;
struct ResidualRow {
    std::uint64_t offset = 0;
    std::uint32_t type = 0, bytes = 0;
};
// The index is small and immutable; compressed row data stays in its file.
class RowResidualFile {
    std::shared_ptr<FileRangeReader> reader_;
    std::uint64_t columns_=0, payload_offset_=0, payload_bytes_=0;
    std::vector<ResidualRow> rows_;
public:
    explicit RowResidualFile(const std::filesystem::path&);
    std::uint64_t columns() const noexcept { return columns_; }
    std::uint64_t row_count() const noexcept { return rows_.size(); }
    std::uint64_t payload_bytes() const noexcept { return payload_bytes_; }
    const std::vector<ResidualRow>& rows() const noexcept { return rows_; }
    void read_payload(std::uint64_t offset, void* output, std::size_t bytes) const;
    std::vector<std::uint8_t> read_row(std::uint64_t index) const;
};
class ModelSource;
std::shared_ptr<const ModelSource> open_residual_model_source(const std::filesystem::path&);
}
