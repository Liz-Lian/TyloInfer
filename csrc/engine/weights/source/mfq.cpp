#include "mfq.h"
#include "engine/weights/encoding/tensor_dtype.h"
#include "engine/weights/encoding/legacy.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>

namespace mfq::archive {
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

std::filesystem::path resolve(const std::filesystem::path& path) {
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

File read(const std::filesystem::path& requested) {
    const auto path = resolve(requested);
    const auto size = usable_file_size(path);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open MFQ file: " + path.string());
    Input input(stream, size, path);

    char magic[4]{};
    for (char& value : magic) value = input.scalar<char>("MFQ magic");
    if (std::memcmp(magic, "MFQ1", sizeof(magic)) != 0) {
        throw std::runtime_error("bad MFQ magic: " + path.string());
    }

    File result;
    result.path = path;
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
        Record record;
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

std::uint64_t integer(
    const Header& header,
    const std::string& key,
    std::uint64_t fallback) {
    const auto found = header.metadata.find(key);
    if (found == header.metadata.end()) return fallback;
    std::size_t parsed = 0;
    std::uint64_t result = 0;
    try {
        result = std::stoull(found->second, &parsed, 10);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " +
                                 found->second);
    }
    while (parsed < found->second.size() &&
           std::isspace(static_cast<unsigned char>(found->second[parsed]))) {
        ++parsed;
    }
    if (parsed != found->second.size()) {
        throw std::runtime_error("invalid MFQ integer metadata " + key + ": " +
                                 found->second);
    }
    return result;
}

Shards shards(
    const std::filesystem::path& path,
    std::uint64_t split_no,
    std::uint64_t split_count) {
    static const std::regex pattern(shard_pattern, std::regex::icase);
    std::smatch match;
    const auto filename = path.filename().string();
    const bool root_entry = !std::regex_match(filename, match, pattern);
    auto suffix = root_entry ? path.extension().string() : match[4].str();
    std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });
    if (root_entry && (split_no != 0 || (suffix != ".tq" && suffix != ".mfq"))) {
        throw std::runtime_error(
            "sharded MFQ path lacks -00001-of-00000 suffix: " + path.string());
    }
    if (!root_entry && (std::stoull(match[2].str()) != split_no + 1 ||
        std::stoull(match[3].str()) != split_count)) {
        throw std::runtime_error("MFQ shard filename/metadata mismatch: " +
                                 path.string());
    }
    const auto stem = root_entry ? path.stem().string() : match[1].str();
    const auto root = root_entry ? path : path.parent_path() / (stem + path.extension().string());
    const auto same_suffix = [&](const std::ssub_match& candidate) {
        auto value = candidate.str();
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
        return value == suffix;
    };
    Shards result;
    result.entry = path;
    result.stem = stem;
    result.suffix = suffix;
    result.paths.resize(static_cast<std::size_t>(split_count));
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        std::ostringstream name;
        name << stem << '-' << std::setfill('0') << std::setw(5)
             << index << "-of-" << std::setw(5) << split_count << suffix;
        result.paths[index - 1] = path.parent_path() / name.str();
    }
    if (root_entry) {
        if (std::filesystem::is_regular_file(result.paths[0]) &&
            !std::filesystem::equivalent(result.paths[0], root))
            throw std::runtime_error("duplicate MFQ shard index 1: " + path.string());
        result.paths[0] = root;
    } else if (!std::filesystem::is_regular_file(result.paths[0]) &&
               std::filesystem::is_regular_file(root)) {
        result.paths[0] = root;
    }
    if (std::all_of(result.paths.begin(), result.paths.end(),
            [](const auto& candidate) { return std::filesystem::is_regular_file(candidate); })) {
        for (auto& candidate : result.paths) candidate = resolve(candidate);
        result.complete = true;
        return result;
    }
    result.found.assign(result.paths.size(), false);
    result.found[0] = result.paths[0] == root;
    for (const auto& entry : std::filesystem::directory_iterator(path.parent_path())) {
        std::smatch candidate;
        const auto candidate_name = entry.path().filename().string();
        if (!entry.is_regular_file() || !std::regex_match(candidate_name, candidate, pattern) ||
            candidate[1].str() != stem || !same_suffix(candidate[4]) ||
            std::stoull(candidate[3].str()) != split_count) {
            continue;
        }
        const auto index = std::stoull(candidate[2].str());
        if (index < 1 || index > split_count) continue;
        if (result.found[index - 1] &&
            !std::filesystem::equivalent(result.paths[index - 1], entry.path())) {
            throw std::runtime_error("duplicate MFQ shard index " + std::to_string(index) + ": " + path.string());
        }
        result.paths[index - 1] = entry.path();
        result.found[index - 1] = true;
    }
    return result;
}

std::vector<std::filesystem::path> resolve(Shards parts) {
    if (parts.complete) return std::move(parts.paths);
    for (auto& path : parts.paths) path = resolve(path);
    return std::move(parts.paths);
}

Merge::Merge(const Header& initial, std::uint64_t split_count)
    : initial_version_(initial.version),
      initial_architecture_(initial.architecture),
      split_count_(split_count),
      expected_records_(std::numeric_limits<std::uint64_t>::max()),
      expected_tensors_(std::numeric_limits<std::uint64_t>::max()) {}

void Merge::add(File parsed, std::uint64_t index) {
    if (split_count_ > 1) {
        if (parsed.header.version != initial_version_ ||
            parsed.header.architecture != initial_architecture_ ||
            integer(parsed.header, "split.no", split_count_) != index ||
            integer(parsed.header, "split.count", 0) != split_count_) {
            throw std::runtime_error("MFQ shard metadata mismatch: " +
                                     parsed.path.string());
        }
        if (index == 0) output_.header = parsed.header;
        const auto records = integer(
            parsed.header, "split.records.count",
            std::numeric_limits<std::uint64_t>::max());
        const auto tensors = integer(
            parsed.header, "split.tensors.count",
            std::numeric_limits<std::uint64_t>::max());
        if (expected_records_ == std::numeric_limits<std::uint64_t>::max()) {
            expected_records_ = records;
        } else if (records != std::numeric_limits<std::uint64_t>::max() &&
                   records != expected_records_) {
            throw std::runtime_error("MFQ shard record count mismatch");
        }
        if (expected_tensors_ == std::numeric_limits<std::uint64_t>::max()) {
            expected_tensors_ = tensors;
        } else if (tensors != std::numeric_limits<std::uint64_t>::max() &&
                   tensors != expected_tensors_) {
            throw std::runtime_error("MFQ shard tensor count mismatch");
        }
        actual_records_ += parsed.header.record_count;
        for (const auto& record : parsed.records) {
            if (!record.asset) ++actual_tensors_;
        }
    } else {
        output_.header = parsed.header;
    }
    for (auto& record : parsed.records) {
        if (!output_.names.emplace(record.tensor.name, output_.records.size()).second) {
            throw std::runtime_error("duplicate MFQ tensor record: " +
                                     record.source_path.string());
        }
        output_.records.push_back(std::move(record));
    }
}

Index Merge::finish() {
    if (split_count_ > 1) {
        if (expected_records_ != std::numeric_limits<std::uint64_t>::max() &&
            actual_records_ != expected_records_) {
            throw std::runtime_error("MFQ shard record total mismatch");
        }
        if (expected_tensors_ != std::numeric_limits<std::uint64_t>::max() &&
            actual_tensors_ != expected_tensors_) {
            throw std::runtime_error("MFQ shard tensor total mismatch");
        }
    }
    return std::move(output_);
}

} // namespace mfq::archive
