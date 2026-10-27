#pragma once

#include "model_source.h"

namespace mfq {

bool is_gguf_file(const std::filesystem::path& path);

class GgufModelSource final : public ModelSource {
public:
    explicit GgufModelSource(const std::filesystem::path& path,
        std::vector<std::filesystem::path> companions = {});
    ~GgufModelSource() override;
    const std::vector<std::filesystem::path>& source_paths() const noexcept override;
    std::string_view architecture() const noexcept override;
    const std::unordered_map<std::string, std::string>& metadata() const noexcept override;
    const std::vector<TensorMetadata>& tensors() const noexcept override;
    const TensorMetadata* find_tensor(std::string_view name) const noexcept override;
    void read_range_into(std::string_view name, std::uint64_t offset,
                         std::byte* destination, std::size_t size) const override;
    TensorReader tensor_reader(std::string_view name) const override;
    TensorBatchReader tensor_batch_reader(std::string_view name) const override;
    std::optional<TensorFileRange> tensor_file_range(std::string_view name) const override;
    bool supports_parallel_tensor_reads() const noexcept override { return true; }
    const std::vector<std::string>& assets() const noexcept override;
    bool has_asset(std::string_view name) const noexcept override;
    std::vector<std::byte> read_asset(std::string_view name) const override;
    std::optional<ModelGraph> model_graph() const override;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

} // namespace mfq
