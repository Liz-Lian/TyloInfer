#include "mfq.h"
#include "engine/weights/encoding/tensor_dtype.h"
#include "engine/weights/encoding/legacy.h"
#include "nlohmann/json.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace mfq {
namespace {

constexpr std::uint64_t kMaxStringBytes = std::uint64_t{64} << 20;
constexpr std::uint32_t kMaxMetadataEntries = std::uint32_t{1} << 16;
constexpr std::uint32_t kMaxRecordEntries = std::uint32_t{1} << 20;
constexpr std::string_view kAssetPrefix = "__mfq_asset__/";
constexpr std::string_view kTinferAssetPrefix = "__tinfer_asset__/";
std::string canonical_asset_name(std::string name) {
    if (name.compare(0, kTinferAssetPrefix.size(), kTinferAssetPrefix) == 0)
        name.replace(0, kTinferAssetPrefix.size(), kAssetPrefix);
    return name;
}

std::uint64_t usable_file_size(const std::filesystem::path& path) {
    std::error_code error;
    const auto value = std::filesystem::file_size(path, error);
    if (error || value > std::numeric_limits<std::uint64_t>::max() ||
        value > static_cast<std::uintmax_t>(
                    std::numeric_limits<std::streamoff>::max())) {
        throw std::runtime_error("cannot determine usable MFQ file size: " +
                                 path.string());
    }
    return static_cast<std::uint64_t>(value);
}

class Input {
public:
    Input(std::istream& stream, std::uint64_t size, const std::filesystem::path& path)
        : stream_(stream), size_(size), path_(path) {}

    std::uint64_t position() const noexcept { return position_; }
    std::uint64_t remaining() const noexcept { return size_ - position_; }

    template <typename T>
    T scalar(const char* what) {
        T value{};
        read(reinterpret_cast<char*>(&value), sizeof(value), what);
        return value;
    }

    std::string string(const char* what) {
        const auto size = scalar<std::uint32_t>("string length");
        if (size > kMaxStringBytes) {
            throw std::runtime_error(std::string("MFQ ") + what +
                                     " exceeds the supported string length: " +
                                     path_.string());
        }
        std::string result(size, '\0');
        if (size != 0) read(result.data(), size, what);
        return result;
    }

private:
    void read(char* destination, std::uint64_t size, const char* what) {
        if (size > remaining() ||
            size > static_cast<std::uint64_t>(
                       std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error(std::string("unexpected EOF reading ") +
                                     what + ": " + path_.string());
        }
        stream_.read(destination, static_cast<std::streamsize>(size));
        if (!stream_) {
            throw std::runtime_error(std::string("failed reading MFQ ") + what +
                                     ": " + path_.string());
        }
        position_ += size;
    }

    std::istream& stream_;
    std::uint64_t size_;
    const std::filesystem::path& path_;
    std::uint64_t position_ = 0;
};

void validate_count(
    std::uint32_t count,
    std::uint32_t limit,
    std::uint64_t minimum_bytes,
    std::uint64_t remaining,
    const char* what,
    const std::filesystem::path& path) {
    if (count > limit || count > remaining / minimum_bytes) {
        throw std::runtime_error(std::string("invalid MFQ ") + what + ": " +
                                 path.string());
    }
}

} // namespace

std::filesystem::path mfq_path(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error("cannot open MFQ file: " + path.string());
    }
    auto result = std::filesystem::weakly_canonical(path, error);
    if (error) {
        error.clear();
        result = std::filesystem::absolute(path, error).lexically_normal();
    }
    if (error || !result.is_absolute()) {
        throw std::runtime_error("cannot resolve MFQ file path: " + path.string());
    }
    return result;
}

MfqFile read_mfq(const std::filesystem::path& requested) {
    const auto path = mfq_path(requested);
    const auto size = usable_file_size(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MFQ file: " + path.string());
    Input input(stream, size, path);

    char magic[4]{};
    for (char& value : magic) value = input.scalar<char>("MFQ magic");
    if (std::memcmp(magic, "MFQ1", sizeof(magic)) != 0) {
        throw std::runtime_error("bad MFQ magic: " + path.string());
    }

    MfqFile result;
    result.header.version = input.scalar<std::uint32_t>("MFQ version");
    if (result.header.version == 0 || result.header.version > 2) {
        throw std::runtime_error("unsupported MFQ version " +
                                 std::to_string(result.header.version) + ": " +
                                 path.string());
    }
    result.header.architecture = input.string("architecture");
    if (result.header.version >= 2) {
        const auto count = input.scalar<std::uint32_t>("metadata count");
        validate_count(count, kMaxMetadataEntries, 2 * sizeof(std::uint32_t),
                       input.remaining(), "metadata count", path);
        for (std::uint32_t index = 0; index < count; ++index) {
            auto key = input.string("metadata key");
            auto value = input.string("metadata value");
            if (!result.header.metadata.emplace(std::move(key), std::move(value)).second) {
                throw std::runtime_error("duplicate MFQ metadata key: " + path.string());
            }
        }
        const auto assets = result.header.metadata.find("runtime_assets");
        if (assets != result.header.metadata.end() && assets->second.find(kTinferAssetPrefix) != std::string::npos) {
            auto manifest = nlohmann::json::parse(assets->second);
            if (manifest.is_object() && manifest.contains("assets") && manifest["assets"].is_object())
                for (auto& asset : manifest["assets"].items())
                    if (asset.value().is_object() && asset.value().contains("record") && asset.value()["record"].is_string())
                        asset.value()["record"] = canonical_asset_name(asset.value()["record"].get<std::string>());
            assets->second = manifest.dump();
        }
    }

    result.header.record_count = input.scalar<std::uint32_t>("record count");
    validate_count(result.header.record_count, kMaxRecordEntries,
                   2 * sizeof(std::uint32_t) + sizeof(std::uint64_t),
                   input.remaining(), "record count", path);
    result.records.reserve(result.header.record_count);
    for (std::uint32_t index = 0; index < result.header.record_count; ++index) {
        MfqStoredRecord record;
        record.tensor.name = canonical_asset_name(input.string("record name"));
        record.tensor.stored_dtype = input.string("record dtype");
        const auto dtype =
#if TYLOINFER_ENABLE_MFQ_COMPAT
            canonical_format_dtype(record.tensor.stored_dtype);
#else
            std::string_view(record.tensor.stored_dtype);
#endif
        if (!supported_tensor_dtype(dtype)) {
            throw std::runtime_error("TyloInfer does not support tensor format "
                + record.tensor.stored_dtype + ": " + record.tensor.name);
        }
        record.tensor.dtype = std::string(dtype);
        record.tensor.nbytes = input.scalar<std::uint64_t>("record length");
        constexpr std::string_view legacy_asset_prefix = "__tinfer_asset__/";
        if (record.tensor.name.compare(0, legacy_asset_prefix.size(), legacy_asset_prefix) == 0)
            record.tensor.name = std::string(kAssetPrefix) + record.tensor.name.substr(legacy_asset_prefix.size());
        record.asset = record.tensor.name.compare(
                           0, kAssetPrefix.size(), kAssetPrefix) == 0;
        record.source_path = path;
        result.records.push_back(std::move(record));
    }
    auto offset = input.position();
    for (auto& record : result.records) {
        record.offset = offset;
        if (record.tensor.nbytes > size - offset) {
            throw std::runtime_error(
                "MFQ file length does not match record table: " + path.string());
        }
        offset += record.tensor.nbytes;
    }
    if (offset != size) {
        throw std::runtime_error(
            "MFQ file length does not match record table: " + path.string());
    }
    return result;
}

} // namespace mfq
