#pragma once

#include "formats/model_graph.h"
#include "components/io/file_range_reader.h"
#include "components/io/read_span.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mfq {

inline constexpr const char* kTokenizerGgufAsset = "__mfq_asset__/tokenizer.gguf";

inline constexpr std::string_view kModelConfigAsset =
    "__mfq_asset__/model_config.json";

struct TensorMetadata {
    std::string name;
    std::string dtype;
    std::string stored_dtype;
    std::optional<std::vector<std::int64_t>> shape;
    std::uint64_t nbytes = 0;
};

// The small set of legacy numeric-layout facts still needed by model loaders.
// Name aliases remain private to each source implementation.
struct TensorLayout {
    double norm_weight_offset = 1.0;
    bool linear_attention_a_is_log = true;
    bool qwen_gdn_gguf_layout = false;
};

struct TensorFileRange {
    std::filesystem::path path;
    std::uint64_t offset = 0;
    std::uint64_t nbytes = 0;
};

// A capability refusal, distinct from invalid metadata or a failed read.
class TensorReaderUnsupported : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Backend-neutral, read-only model input. Implementations normalize storage
// names before tensors cross this boundary; CUDA and Metal only consume the
// canonical names exposed here.
class ModelSource {
public:
    using TensorBatchReader=std::function<void(const std::vector<ReadSpan>&)>;
    using TensorReader = std::function<void(
        std::uint64_t, std::byte*, std::size_t)>;
    virtual ~ModelSource() = default;

    virtual const std::vector<std::filesystem::path>& source_paths()
        const noexcept = 0;
    virtual std::string_view architecture() const noexcept = 0;
    virtual const std::unordered_map<std::string, std::string>& metadata()
        const noexcept = 0;
    virtual const std::vector<TensorMetadata>& tensors() const noexcept = 0;
    virtual const TensorMetadata* find_tensor(
        std::string_view name) const noexcept = 0;
    virtual TensorLayout tensor_layout() const noexcept { return {}; }
    virtual void read_range_into(
        std::string_view name,
        std::uint64_t relative_offset,
        std::byte* destination,
        std::size_t size) const = 0;
    virtual void drop_file_cache() const noexcept {}
    virtual FileReadStats file_read_stats() const noexcept { return {}; }
    // True when independent tensor reads can overlap without a shared seek
    // lock. Unknown callbacks must keep their serial access order.
    virtual bool supports_parallel_tensor_reads() const noexcept { return false; }
    virtual std::optional<TensorFileRange> tensor_file_range(std::string_view) const { return std::nullopt; }
    virtual std::optional<TensorFileRange> tensor_file_range_at(std::string_view name,
        std::uint64_t offset, std::size_t size) const {
        auto range = tensor_file_range(name);
        if (!range) return std::nullopt;
        if (offset > range->nbytes || size > range->nbytes - offset)
            throw std::out_of_range("tensor file range is out of bounds");
        range->offset += offset; range->nbytes = size;
        return range;
    }

    // An independently owned, bounded range reader that survives this source.
    // Packed row consumers must not fall back to reading the complete tensor.
    virtual TensorReader tensor_reader(std::string_view) const {
        throw TensorReaderUnsupported("model source has no retained tensor range reader");
    }
    virtual TensorBatchReader tensor_batch_reader(std::string_view) const { return {}; }

    // An optional retained, bounded, concurrently readable mapping.
    // An empty reader means this source does not offer mapped access.
    virtual TensorReader mapped_tensor_reader(std::string_view) const { return {}; }
    // Optional mapped range batch: prefetch only the requested pages before
    // copying. An empty callback leaves the ordinary row reader available.
    virtual TensorBatchReader mapped_tensor_batch_reader(std::string_view) const { return {}; }

    virtual const std::vector<std::string>& assets() const noexcept = 0;
    virtual bool has_asset(std::string_view name) const noexcept = 0;
    virtual std::vector<std::byte> read_asset(std::string_view name) const = 0;
    // The stored graph may be absent in legacy artifacts. Runtime callers
    // should use resolved_model_graph(), which applies compatibility here at
    // the source boundary.
    virtual std::optional<ModelGraph> model_graph() const = 0;

    std::string model_config_json(const std::string& external_path = {}) const;
    ModelGraph resolved_model_graph() const;

    std::vector<std::byte> read(
        std::string_view name,
        std::uint64_t relative_offset = 0,
        std::optional<std::size_t> size = std::nullopt) const {
        const auto* tensor = find_tensor(name);
        if (tensor == nullptr) {
            throw std::runtime_error("model tensor not found: " + std::string(name));
        }
        if (relative_offset > tensor->nbytes) {
            throw std::out_of_range("model tensor byte range is out of bounds: " +
                                    std::string(name));
        }
        const auto available = tensor->nbytes - relative_offset;
        if (!size && available > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("model tensor is too large to read");
        }
        const auto requested = size ? *size : static_cast<std::size_t>(available);
        if (requested > available) {
            throw std::out_of_range("model tensor byte range is out of bounds: " +
                                    std::string(name));
        }
        std::vector<std::byte> result(requested);
        read_range_into(name, relative_offset, result.data(), result.size());
        return result;
    }
};

std::shared_ptr<const ModelSource> open_model_source(
    const std::filesystem::path& path);

} // namespace mfq
