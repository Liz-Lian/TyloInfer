#include "mfq_model_source.h"
#include "engine/weights/source/buffered.h"
#include "engine/weights/source/mapped.h"
#include "components/io/async_range_reader.h"

#include "engine/models/common/legacy_names.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <limits>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <system_error>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cerrno>
#endif

namespace mfq {
namespace {

std::uint64_t metadata_uint(
    const MfqSourceHeader& header,
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

std::vector<std::filesystem::path> shard_paths(
    const std::filesystem::path& path,
    std::uint64_t split_no,
    std::uint64_t split_count) {
    static const std::regex pattern(
        R"(^(.*)-([0-9]{5})-of-([0-9]{5})(\.mfq|\.tq)$)", std::regex::icase);
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
    std::vector<std::filesystem::path> result(static_cast<std::size_t>(split_count));
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        std::ostringstream name;
        name << stem << '-' << std::setfill('0') << std::setw(5)
             << index << "-of-" << std::setw(5) << split_count << suffix;
        result[index - 1] = path.parent_path() / name.str();
    }
    if (root_entry) {
        if (std::filesystem::is_regular_file(result[0]) && !std::filesystem::equivalent(result[0], root))
            throw std::runtime_error("duplicate MFQ shard index 1: " + path.string());
        result[0] = root;
    } else if (!std::filesystem::is_regular_file(result[0]) && std::filesystem::is_regular_file(root)) {
        result[0] = root;
    }
    if (std::all_of(result.begin(), result.end(),
            [](const auto& candidate) { return std::filesystem::is_regular_file(candidate); })) {
        for (auto& candidate : result) candidate = mfq_path(candidate);
        return result;
    }
    std::vector<bool> found(result.size(), false);
    found[0] = result[0] == root;
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
        if (found[index - 1] && !std::filesystem::equivalent(result[index - 1], entry.path())) {
            throw std::runtime_error("duplicate MFQ shard index " + std::to_string(index) + ": " + path.string());
        }
        result[index - 1] = entry.path();
        found[index - 1] = true;
    }
    // Shared Engram shards keep their original indices, but live once beside
    // the precision directories. Never substitute another model or split set.
    static const std::regex engram_pattern(R"(^(.*)-Engram-.+$)", std::regex::icase);
    std::unordered_map<std::uint64_t, std::filesystem::path> shared;
    std::vector<std::filesystem::path> directories{
        path.parent_path(), path.parent_path() / "Engram",
        path.parent_path().parent_path() / "Engram"};
    std::sort(directories.begin(), directories.end());
    directories.erase(std::unique(directories.begin(), directories.end()), directories.end());
    for (const auto& directory : directories) {
        if (!std::filesystem::is_directory(directory)) continue;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            std::smatch candidate, family;
            const auto name = entry.path().filename().string();
            if (!entry.is_regular_file() || !std::regex_match(name, candidate, pattern)) continue;
            const auto shared_stem = candidate[1].str();
            if (!std::regex_match(shared_stem, family, engram_pattern) ||
                stem.compare(0, family[1].length() + 1, family[1].str() + "-") != 0 ||
                std::stoull(candidate[3].str()) != split_count || !same_suffix(candidate[4])) continue;
            const auto index = std::stoull(candidate[2].str());
            if (index < 1 || index > split_count || found[index - 1]) continue;
            if (!shared.emplace(index, entry.path()).second)
                throw std::runtime_error("ambiguous shared Engram shard " + std::to_string(index));
        }
    }
    for (const auto& [index, candidate] : shared) result[index - 1] = candidate;
    for (std::uint64_t index = 1; index <= split_count; ++index) {
        result[index - 1] = mfq_path(result[index - 1]);
    }
    return result;
}

} // namespace

struct MfqModelSource::Impl {
    FileReadMode read_mode = FileReadMode::Buffered;
    std::unordered_map<std::filesystem::path, std::shared_ptr<FileRangeReader>> direct_readers;
    mutable std::mutex pipeline_reader_mutex;
    mutable std::unordered_map<std::filesystem::path,std::shared_ptr<AsyncRangeReader>> pipeline_readers;
    MfqSourceHeader header;
    std::vector<std::filesystem::path> source_paths;
    std::vector<MfqStoredRecord> records;
    std::vector<TensorMetadata> tensors;
    std::vector<std::string> assets;
    MfqLegacyTensorAliases legacy_tensor_compatibility;
    std::unordered_map<std::string, std::size_t> records_by_name;
    std::unordered_map<std::string, std::size_t> tensors_by_name;
};

MfqModelSource::MfqModelSource(std::filesystem::path requested, FileReadMode mode)
    : impl_(std::make_unique<Impl>()) {
    impl_->read_mode = mode;
    const auto initial_path = mfq_path(requested);
    auto initial = read_mfq(initial_path);
    const auto split_no = metadata_uint(initial.header, "split.no", 0);
    const auto split_count = metadata_uint(initial.header, "split.count", 1);
    if (split_count == 0 || split_count > 99999 || split_no >= split_count) {
        throw std::runtime_error("invalid MFQ split metadata: " +
                                 initial_path.string());
    }

    auto append = [&](MfqFile parsed) {
        for (auto& record : parsed.records) {
            if (!impl_->records_by_name.emplace(
                    record.tensor.name, impl_->records.size()).second) {
                throw std::runtime_error("duplicate MFQ tensor record: " +
                                         record.source_path.string());
            }
            impl_->records.push_back(std::move(record));
        }
    };

    if (split_count == 1) {
        impl_->header = initial.header;
        impl_->source_paths.push_back(initial_path);
        append(std::move(initial));
    } else {
        // A numbered entry may be a symlink to an unnumbered downloaded file.
        // Discover siblings from the requested name before resolving each file.
        const auto paths = shard_paths(std::filesystem::absolute(requested).lexically_normal(), split_no, split_count);
        std::uint64_t actual_records = 0;
        std::uint64_t actual_tensors = 0;
        auto expected_records = std::numeric_limits<std::uint64_t>::max();
        auto expected_tensors = std::numeric_limits<std::uint64_t>::max();
        for (std::uint64_t index = 0; index < split_count; ++index) {
            auto current = read_mfq(paths[static_cast<std::size_t>(index)]);
            static const std::regex shared_name(R"(^.*-Engram-.+-[0-9]{5}-of-[0-9]{5}\.(mfq|tq)$)", std::regex::icase);
            if (std::regex_match(paths[index].filename().string(), shared_name)) {
                const auto table = [](const MfqStoredRecord& record) {
                    std::string_view name = record.tensor.name;
                    constexpr std::string_view marker = ".tq_row_shard.";
                    const auto shard = name.rfind(marker);
                    if (shard != std::string_view::npos && shard + marker.size() < name.size()
                        && std::all_of(name.begin() + shard + marker.size(), name.end(), [](char c) { return c >= '0' && c <= '9'; }))
                        name = name.substr(0, shard);
                    for (const std::string_view suffix : {".associative_memory.embedding.weight", ".engram.embed.weight"})
                        if (name.size() >= suffix.size() &&
                            name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) return true;
                    return false;
                };
                if (current.records.empty() || !std::all_of(current.records.begin(), current.records.end(), table))
                    throw std::runtime_error("shared Engram shard contains non-table records: " + paths[index].string());
            }
            if (current.header.version != initial.header.version ||
                current.header.architecture != initial.header.architecture ||
                metadata_uint(current.header, "split.no", split_count) != index ||
                metadata_uint(current.header, "split.count", 0) != split_count) {
                throw std::runtime_error("MFQ shard metadata mismatch: " +
                                         paths[static_cast<std::size_t>(index)].string());
            }
            if (index == 0) impl_->header = current.header;
            const auto records = metadata_uint(
                current.header, "split.records.count",
                std::numeric_limits<std::uint64_t>::max());
            const auto tensors = metadata_uint(
                current.header, "split.tensors.count",
                std::numeric_limits<std::uint64_t>::max());
            if (expected_records == std::numeric_limits<std::uint64_t>::max()) {
                expected_records = records;
            } else if (records != std::numeric_limits<std::uint64_t>::max() &&
                       records != expected_records) {
                throw std::runtime_error("MFQ shard record count mismatch");
            }
            if (expected_tensors == std::numeric_limits<std::uint64_t>::max()) {
                expected_tensors = tensors;
            } else if (tensors != std::numeric_limits<std::uint64_t>::max() &&
                       tensors != expected_tensors) {
                throw std::runtime_error("MFQ shard tensor count mismatch");
            }
            actual_records += current.header.record_count;
            for (const auto& record : current.records) {
                if (!record.asset) ++actual_tensors;
            }
            append(std::move(current));
        }
        if (expected_records != std::numeric_limits<std::uint64_t>::max() &&
            actual_records != expected_records) {
            throw std::runtime_error("MFQ shard record total mismatch");
        }
        if (expected_tensors != std::numeric_limits<std::uint64_t>::max() &&
            actual_tensors != expected_tensors) {
            throw std::runtime_error("MFQ shard tensor total mismatch");
        }
        impl_->source_paths = paths;
    }

    if (mode == FileReadMode::Direct) {
        for (const auto& path : impl_->source_paths)
            impl_->direct_readers.emplace(path, std::make_shared<FileRangeReader>(path, mode));
    }
    std::vector<std::string> stored_names;
    for (const auto& record : impl_->records) {
        if (record.asset) {
            impl_->assets.push_back(record.tensor.name);
        } else {
            stored_names.push_back(record.tensor.name);
        }
    }
    std::sort(impl_->assets.begin(), impl_->assets.end());

    std::unordered_map<std::string, std::string> stored_to_canonical;
    if (!has_asset(kModelGraphAsset) && has_asset(kModelConfigAsset)) {
        const auto config = read_asset(kModelConfigAsset);
        impl_->legacy_tensor_compatibility = make_legacy_tensor_aliases(
            architecture(),
            std::string_view(
                reinterpret_cast<const char*>(config.data()), config.size()),
            stored_names);
        for (const auto& [canonical, stored] :
             impl_->legacy_tensor_compatibility.canonical_to_stored) {
            const auto record = impl_->records_by_name.find(stored);
            if (record == impl_->records_by_name.end() ||
                    impl_->records[record->second].asset ||
                    impl_->records_by_name.find(canonical) !=
                        impl_->records_by_name.end()) {
                throw std::runtime_error(
                    "invalid legacy MFQ tensor alias: " + canonical);
            }
            const auto [found, inserted] =
                stored_to_canonical.emplace(stored, canonical);
            if (!inserted && found->second != canonical) {
                throw std::runtime_error(
                    "legacy MFQ tensor has multiple canonical names: " +
                    stored);
            }
        }
    }

    for (const auto& record : impl_->records) {
        if (record.asset) continue;
        auto tensor = record.tensor;
        if (const auto found = stored_to_canonical.find(tensor.name);
            found != stored_to_canonical.end()) {
            tensor.name = found->second;
        }
        if (!impl_->tensors_by_name.emplace(
                tensor.name, impl_->tensors.size()).second) {
            throw std::runtime_error(
                "legacy MFQ tensor aliases collide at canonical name: " +
                tensor.name);
        }
        impl_->tensors.push_back(std::move(tensor));
    }
}

MfqModelSource::~MfqModelSource() = default;
MfqModelSource::MfqModelSource(MfqModelSource&&) noexcept = default;
MfqModelSource& MfqModelSource::operator=(MfqModelSource&&) noexcept = default;

const MfqSourceHeader& MfqModelSource::header() const noexcept { return impl_->header; }
const std::vector<std::filesystem::path>& MfqModelSource::source_paths() const noexcept {
    return impl_->source_paths;
}
const std::vector<MfqStoredRecord>& MfqModelSource::records() const noexcept {
    return impl_->records;
}
std::string_view MfqModelSource::architecture() const noexcept {
    return impl_->header.architecture;
}
const std::unordered_map<std::string, std::string>&
MfqModelSource::metadata() const noexcept {
    return impl_->header.metadata;
}
const std::vector<TensorMetadata>& MfqModelSource::tensors() const noexcept {
    return impl_->tensors;
}

const TensorMetadata* MfqModelSource::find_tensor(std::string_view name) const noexcept {
    const auto found = impl_->tensors_by_name.find(std::string(name));
    return found == impl_->tensors_by_name.end()
        ? nullptr
        : &impl_->tensors[found->second];
}

TensorLayout MfqModelSource::tensor_layout() const noexcept {
    return impl_->legacy_tensor_compatibility.layout;
}

void MfqModelSource::read_range_into(
    std::string_view name,
    std::uint64_t relative_offset,
    std::byte* destination,
    std::size_t size) const {
    auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end()) {
        const auto alias = impl_->legacy_tensor_compatibility
            .canonical_to_stored.find(std::string(name));
        if (alias != impl_->legacy_tensor_compatibility
                .canonical_to_stored.end()) {
            found = impl_->records_by_name.find(alias->second);
        }
    }
    if (found == impl_->records_by_name.end() ||
            impl_->records[found->second].asset) {
        throw std::runtime_error("model tensor not found: " + std::string(name));
    }
    const auto& record = impl_->records[found->second];
    if (relative_offset > record.tensor.nbytes ||
        size > record.tensor.nbytes - relative_offset) {
        throw std::out_of_range("model tensor byte range is out of bounds: " +
                                std::string(name));
    }
    if (relative_offset > std::numeric_limits<std::uint64_t>::max() - record.offset) {
        throw std::overflow_error("model tensor byte offset overflow");
    }
    if (impl_->read_mode == FileReadMode::Direct)
        impl_->direct_readers.at(record.source_path)->read(record.offset + relative_offset, destination, size);
    else mfq::read_buffered(record.source_path, record.offset + relative_offset, destination, size);
}

ModelSource::TensorReader MfqModelSource::tensor_reader(std::string_view name) const {
    auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end()) {
        const auto alias = impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if (alias != impl_->legacy_tensor_compatibility.canonical_to_stored.end())
            found = impl_->records_by_name.find(alias->second);
    }
    if (found == impl_->records_by_name.end() || impl_->records[found->second].asset)
        throw std::runtime_error("model tensor not found: " + std::string(name));
    if (impl_->read_mode == FileReadMode::Direct) {
        const auto record = impl_->records[found->second];
        auto reader = impl_->direct_readers.at(record.source_path);
        return [record, reader](std::uint64_t offset, std::byte* destination, std::size_t size) {
            if (offset > record.tensor.nbytes || size > record.tensor.nbytes - offset)
                throw std::out_of_range("model tensor row range is out of bounds");
            reader->read(record.offset + offset, destination, size);
        };
    }
    const auto& record = impl_->records[found->second];
    return mfq::buffered_reader(record.source_path, record.offset, record.tensor.nbytes);
}

ModelSource::TensorReader MfqModelSource::mapped_tensor_reader(std::string_view name) const {
    auto found=impl_->records_by_name.find(std::string(name));
    if(found==impl_->records_by_name.end()) {
        const auto alias=impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if(alias!=impl_->legacy_tensor_compatibility.canonical_to_stored.end())found=impl_->records_by_name.find(alias->second);
    }
    if(found==impl_->records_by_name.end() || impl_->records[found->second].asset)
        throw std::runtime_error("mapped model tensor not found: "+std::string(name));
    const auto& record=impl_->records[found->second];
    return mfq::mapped_reader(record.source_path,record.offset,record.tensor.nbytes);
}

bool MfqModelSource::supports_parallel_tensor_reads() const noexcept {
    return impl_->read_mode == FileReadMode::Direct;
}
ModelSource::TensorBatchReader MfqModelSource::mapped_tensor_batch_reader(std::string_view name) const {
    auto found=impl_->records_by_name.find(std::string(name));
    if(found==impl_->records_by_name.end()) {
        const auto alias=impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if(alias!=impl_->legacy_tensor_compatibility.canonical_to_stored.end())found=impl_->records_by_name.find(alias->second);
    }
    if(found==impl_->records_by_name.end() || impl_->records[found->second].asset)
        throw std::runtime_error("mapped model tensor not found: "+std::string(name));
    const auto& record=impl_->records[found->second];
    return mfq::mapped_batch_reader(record.source_path,record.offset,record.tensor.nbytes);
}
ModelSource::TensorBatchReader MfqModelSource::tensor_batch_reader(std::string_view name) const {
    auto found=impl_->records_by_name.find(std::string(name));
    if(found==impl_->records_by_name.end()) {
        const auto alias=impl_->legacy_tensor_compatibility.canonical_to_stored.find(std::string(name));
        if(alias!=impl_->legacy_tensor_compatibility.canonical_to_stored.end())found=impl_->records_by_name.find(alias->second);
    }
    if(found==impl_->records_by_name.end() || impl_->records[found->second].asset)throw std::runtime_error("model tensor not found: "+std::string(name));
    const auto record=impl_->records[found->second];
    std::shared_ptr<AsyncRangeReader> reader;
    {
        std::lock_guard lock(impl_->pipeline_reader_mutex);
        auto& slot=impl_->pipeline_readers[record.source_path];
        if(!slot)slot=std::make_shared<AsyncRangeReader>(record.source_path,impl_->read_mode);
        reader=slot;
    }
    return [record,reader](const std::vector<ReadSpan>& spans) {
        std::vector<ReadSpan> absolute;absolute.reserve(spans.size());
        for(auto span:spans) {
            if(span.offset>record.tensor.nbytes || span.size>record.tensor.nbytes-span.offset)
                throw std::out_of_range("batched tensor read exceeds canonical record");
            span.offset+=record.offset;absolute.push_back(span);
        }
        reader->read(absolute);
    };
}

FileReadStats MfqModelSource::file_read_stats() const noexcept {
    FileReadStats result;
    result.mode = impl_->read_mode;
    for (const auto& [path, reader] : impl_->direct_readers) {
        const auto value = reader->stats();
        result.files += value.files; result.calls += value.calls;
        result.logical_bytes += value.logical_bytes; result.physical_bytes += value.physical_bytes;
        result.errors += value.errors; result.staging_bytes += value.staging_bytes;
        result.staging_peak_bytes += value.staging_peak_bytes; result.read_nanoseconds += value.read_nanoseconds;
        result.staging_cached_bytes+=value.staging_cached_bytes;result.staging_allocations+=value.staging_allocations;
    }
    {
        std::lock_guard lock(impl_->pipeline_reader_mutex);
        for(const auto& [path,reader]:impl_->pipeline_readers) {
            const auto value=reader->stats();result.calls+=value.calls;result.logical_bytes+=value.logical_bytes;
            result.physical_bytes+=value.physical_bytes;result.errors+=value.errors;
            result.staging_bytes+=value.staging_bytes;result.staging_peak_bytes+=value.staging_peak_bytes;
            result.read_nanoseconds+=value.read_nanoseconds;
            result.staging_cached_bytes+=value.staging_cached_bytes;result.staging_allocations+=value.staging_allocations;
        }
    }
    return result;
}

void MfqModelSource::drop_file_cache() const noexcept {
#if !defined(_WIN32) && defined(POSIX_FADV_DONTNEED)
    for (const auto& path : impl_->source_paths) {
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) continue;
        static_cast<void>(::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED));
        static_cast<void>(::close(fd));
    }
#endif
}

const std::vector<std::string>& MfqModelSource::assets() const noexcept {
    return impl_->assets;
}

bool MfqModelSource::has_asset(std::string_view name) const noexcept {
    const auto found = impl_->records_by_name.find(std::string(name));
    return found != impl_->records_by_name.end() &&
           impl_->records[found->second].asset;
}

std::vector<std::byte> MfqModelSource::read_asset(std::string_view name) const {
    const auto found = impl_->records_by_name.find(std::string(name));
    if (found == impl_->records_by_name.end() ||
        !impl_->records[found->second].asset) {
        throw std::runtime_error("model asset not found: " + std::string(name));
    }
    const auto& record = impl_->records[found->second];
    if (record.tensor.nbytes > std::numeric_limits<std::size_t>::max()) {
        throw std::overflow_error("model asset is too large to read");
    }
    std::vector<std::byte> result(static_cast<std::size_t>(record.tensor.nbytes));
    if (impl_->read_mode == FileReadMode::Direct)
        impl_->direct_readers.at(record.source_path)->read(record.offset, result.data(), result.size());
    else mfq::read_buffered(record.source_path, record.offset, result.data(), result.size());
    return result;
}

std::optional<ModelGraph> MfqModelSource::model_graph() const {
    if (!has_asset(kModelGraphAsset)) return std::nullopt;
    const auto bytes = read_asset(kModelGraphAsset);
    return ModelGraph::from_json(std::string_view(
        reinterpret_cast<const char*>(bytes.data()), bytes.size()));
}

} // namespace mfq
