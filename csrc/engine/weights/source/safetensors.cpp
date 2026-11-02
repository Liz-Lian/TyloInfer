#include "engine/weights/source/safetensors.h"

#include "engine/weights/source/json.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <system_error>
#include <utility>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mfq {
namespace {

using json = nlohmann::json;

constexpr std::uint64_t kMaxSafetensorsHeaderBytes = 100'000'000;

std::uint64_t little_u64(const std::array<unsigned char, 8>& bytes) {
    std::uint64_t result = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        result |= static_cast<std::uint64_t>(bytes[index]) << (index * 8);
    }
    return result;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("Safetensors byte offset overflow");
    }
    return left + right;
}

std::vector<std::int64_t> parse_shape(
    const json& value,
    const std::string& name) {
    if (!value.is_array()) {
        throw std::runtime_error("Safetensors shape is not an array: " + name);
    }
    std::vector<std::int64_t> result;
    result.reserve(value.size());
    for (const auto& dimension : value) {
        if (!dimension.is_number_integer()) {
            throw std::runtime_error(
                "Safetensors shape has a non-integer dimension: " + name);
        }
        const auto size = dimension.get<std::int64_t>();
        if (size < 0) {
            throw std::runtime_error(
                "Safetensors shape has a negative dimension: " + name);
        }
        result.push_back(size);
    }
    return result;
}

bool inside(const std::filesystem::path& root, const std::filesystem::path& path) {
    auto parent = root.begin();
    auto child = path.begin();
    for (; parent != root.end(); ++parent, ++child) {
        if (child == path.end() || *child != *parent) return false;
    }
    return true;
}

std::filesystem::path shard_path(
    const std::filesystem::path& root,
    const std::string& name) {
    std::error_code error;
    const auto path = std::filesystem::weakly_canonical(root / name, error);
    if (error || !inside(root, path) ||
        !std::filesystem::is_regular_file(path, error) || error) {
        throw std::runtime_error("invalid Safetensors shard path: " + name);
    }
    return path;
}

} // namespace

struct Safetensors::Impl {
    struct ShardReader {
        ShardReader(std::filesystem::path source, std::uint64_t source_size)
            : path(std::move(source)), size(source_size) {}

        ~ShardReader() {
#if !defined(_WIN32)
            if (descriptor >= 0) ::close(descriptor);
#endif
        }

        void read(
            std::uint64_t offset,
            std::byte* destination,
            std::size_t count) const {
            if (count == 0) return;
            if (offset > size || count > size - offset) {
                throw std::out_of_range(
                    "Safetensors byte range is outside shard " + path.string());
            }
#if defined(_WIN32)
            if (offset > static_cast<std::uint64_t>(
                             std::numeric_limits<std::streamoff>::max()) ||
                count > static_cast<std::size_t>(
                            std::numeric_limits<std::streamsize>::max())) {
                throw std::runtime_error("Safetensors byte range is too large");
            }
            std::lock_guard<std::mutex> lock(mutex);
            if (!stream.is_open()) stream.open(path, std::ios::binary);
            stream.clear();
            stream.seekg(static_cast<std::streamoff>(offset));
            stream.read(reinterpret_cast<char*>(destination),
                        static_cast<std::streamsize>(count));
            if (!stream) {
                throw std::runtime_error(
                    "unexpected EOF while reading " + path.string());
            }
#else
            const auto max_offset = static_cast<std::uint64_t>(
                std::numeric_limits<off_t>::max());
            if (offset > max_offset || count > max_offset - offset) {
                throw std::runtime_error("Safetensors byte range is too large");
            }
            const auto file = open();
            std::size_t done = 0;
            while (done < count) {
                constexpr std::size_t kMaxReadBytes = std::size_t{32} << 20;
                const auto request = std::min(count - done, kMaxReadBytes);
                const auto result = ::pread(
                    file,
                    destination + done,
                    request,
                    static_cast<off_t>(offset + done));
                if (result < 0) {
                    if (errno == EINTR) continue;
                    throw std::system_error(
                        errno, std::generic_category(),
                        "pread " + path.string());
                }
                if (result == 0) {
                    throw std::runtime_error(
                        "unexpected EOF while reading " + path.string());
                }
                done += static_cast<std::size_t>(result);
            }
#endif
        }

#if !defined(_WIN32)
        int open() const {
            std::lock_guard<std::mutex> lock(mutex);
            if (descriptor < 0) {
                descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
                if (descriptor < 0) {
                    throw std::system_error(
                        errno, std::generic_category(),
                        "open " + path.string());
                }
#if defined(__APPLE__) && defined(F_NOCACHE)
                // Raw-HF tensors are copied into final runtime buffers or
                // consumed through bounded row/expert stores. Retaining the
                // same payload in the macOS file cache creates a transient
                // second model copy and can compress live GPU allocations.
                if (::fcntl(descriptor, F_NOCACHE, 1) != 0) {
                    const auto error = errno;
                    ::close(descriptor);
                    descriptor = -1;
                    throw std::system_error(
                        error,
                        std::generic_category(),
                        "F_NOCACHE " + path.string());
                }
#endif
            }
            return descriptor;
        }
#endif

        std::filesystem::path path;
        std::uint64_t size;
        mutable std::mutex mutex;
#if defined(_WIN32)
        mutable std::ifstream stream;
#else
        mutable int descriptor = -1;
#endif
    };

    std::filesystem::path root;
    std::vector<std::filesystem::path> source_paths;
    std::vector<std::unique_ptr<ShardReader>> readers;
    std::unordered_map<std::string, SafetensorsRecord> physical;
};

Safetensors::Safetensors(
    std::filesystem::path requested_root,
    std::string source_prefix)
    : impl_(std::make_unique<Impl>()) {
    std::error_code error;
    impl_->root = std::filesystem::canonical(std::move(requested_root), error);
    if (error || !std::filesystem::is_directory(impl_->root, error) || error) {
        throw std::runtime_error("cannot open HF model directory: " +
                                 impl_->root.string());
    }

    const auto index_path = impl_->root / "model.safetensors.index.json";
    std::unordered_map<std::string, std::vector<std::string>> names_by_shard;
    std::vector<std::string> shard_names;
    if (std::filesystem::is_regular_file(index_path)) {
        const auto index = source::parse_json(
            source::read_text(index_path), index_path);
        const auto weight_map = index.find("weight_map");
        if (weight_map == index.end() || !weight_map->is_object() ||
            weight_map->empty()) {
            throw std::runtime_error(
                "Safetensors index requires a non-empty object weight_map: " +
                index_path.string());
        }
        for (const auto& [name, shard] : weight_map->items()) {
            if (!source_prefix.empty() && name.rfind(source_prefix, 0) != 0) {
                continue;
            }
            if (!shard.is_string()) {
                throw std::runtime_error(
                    "Safetensors weight_map value is not a string: " + name);
            }
            names_by_shard[shard.get<std::string>()].push_back(name);
        }
        for (const auto& [name, unused] : names_by_shard) {
            static_cast<void>(unused);
            shard_names.push_back(name);
        }
    } else {
        for (const auto& entry : std::filesystem::directory_iterator(impl_->root)) {
            if (entry.is_regular_file() &&
                entry.path().extension() == ".safetensors") {
                shard_names.push_back(entry.path().filename().string());
            }
        }
        if (shard_names.empty()) {
            throw std::runtime_error("no Safetensors weights found under " +
                                     impl_->root.string());
        }
    }
    std::sort(shard_names.begin(), shard_names.end());

    for (const auto& shard_name : shard_names) {
        const auto path = shard_path(impl_->root, shard_name);
        const auto size = std::filesystem::file_size(path);
        if (size < 8) {
            throw std::runtime_error(
                "cannot read Safetensors header size: " + path.string());
        }
        std::ifstream stream(path, std::ios::binary);
        std::array<unsigned char, 8> raw_header_size{};
        stream.read(reinterpret_cast<char*>(raw_header_size.data()), 8);
        if (!stream) {
            throw std::runtime_error(
                "cannot read Safetensors header size: " + path.string());
        }
        const auto header_size = little_u64(raw_header_size);
        if (header_size > size - 8 ||
            header_size > kMaxSafetensorsHeaderBytes ||
            header_size > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max())) {
            throw std::runtime_error("invalid Safetensors header size: " +
                                     path.string());
        }
        std::string header(static_cast<std::size_t>(header_size), '\0');
        stream.read(header.data(), static_cast<std::streamsize>(header.size()));
        if (!stream && !header.empty()) {
            throw std::runtime_error("cannot read Safetensors header: " +
                                     path.string());
        }
        const auto metadata = source::parse_json(header, path);
        const auto shard_index = impl_->source_paths.size();
        impl_->source_paths.push_back(path);
        impl_->readers.push_back(
            std::make_unique<Impl::ShardReader>(path, size));
        const auto payload_base = checked_add(8, header_size);

        std::vector<std::string> names;
        const auto indexed = names_by_shard.find(shard_name);
        if (indexed != names_by_shard.end()) {
            names = indexed->second;
        } else {
            for (const auto& [name, unused] : metadata.items()) {
                static_cast<void>(unused);
                if (name != "__metadata__") names.push_back(name);
            }
        }
        for (const auto& name : names) {
            if (!source_prefix.empty() && name.rfind(source_prefix, 0) != 0) {
                continue;
            }
            const auto found = metadata.find(name);
            if (found == metadata.end() || !found->is_object()) {
                throw std::runtime_error(
                    "Safetensors index references a missing tensor: " + name);
            }
            const auto dtype = found->find("dtype");
            const auto shape = found->find("shape");
            const auto offsets = found->find("data_offsets");
            if (dtype == found->end() || !dtype->is_string() ||
                shape == found->end() || offsets == found->end() ||
                !offsets->is_array() || offsets->size() != 2 ||
                !(*offsets)[0].is_number_unsigned() ||
                !(*offsets)[1].is_number_unsigned()) {
                throw std::runtime_error(
                    "invalid Safetensors tensor metadata: " + name);
            }
            const auto begin = (*offsets)[0].get<std::uint64_t>();
            const auto end = (*offsets)[1].get<std::uint64_t>();
            if (end < begin || checked_add(payload_base, end) > size) {
                throw std::runtime_error(
                    "Safetensors tensor range is outside its shard: " + name);
            }
            SafetensorsRecord tensor;
            tensor.metadata.name = name;
            tensor.metadata.dtype = dtype->get<std::string>();
            tensor.metadata.stored_dtype = tensor.metadata.dtype;
            tensor.metadata.shape = parse_shape(*shape, name);
            tensor.metadata.nbytes = end - begin;
            tensor.location = {shard_index, checked_add(payload_base, begin)};
            if (!impl_->physical.emplace(name, std::move(tensor)).second) {
                throw std::runtime_error("duplicate Safetensors tensor: " + name);
            }
        }
    }
}

Safetensors::~Safetensors() = default;
Safetensors::Safetensors(Safetensors&&) noexcept = default;
Safetensors& Safetensors::operator=(Safetensors&&) noexcept = default;

const std::filesystem::path& Safetensors::root() const noexcept {
    return impl_->root;
}

const std::vector<std::filesystem::path>& Safetensors::paths() const noexcept {
    return impl_->source_paths;
}

const std::unordered_map<std::string, SafetensorsRecord>&
Safetensors::records() const noexcept {
    return impl_->physical;
}

const TensorMetadata* Safetensors::find(std::string_view name) const noexcept {
    const auto found = impl_->physical.find(std::string(name));
    return found == impl_->physical.end() ? nullptr : &found->second.metadata;
}

void Safetensors::read(
    std::size_t shard,
    std::uint64_t offset,
    std::byte* destination,
    std::size_t size) const {
    if (shard >= impl_->source_paths.size()) {
        throw std::out_of_range("Safetensors shard index out of range");
    }
    impl_->readers[shard]->read(offset, destination, size);
}

void Safetensors::drop_file_cache() const noexcept {
#if !defined(_WIN32) && defined(POSIX_FADV_DONTNEED)
    for (const auto& reader : impl_->readers) {
        try {
            const auto descriptor = reader->open();
            static_cast<void>(::posix_fadvise(
                descriptor,
                0,
                static_cast<off_t>(reader->size),
                POSIX_FADV_DONTNEED));
        } catch (...) {
        }
    }
#endif
}

} // namespace mfq
