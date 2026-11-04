#include "engine/weights/source/gguf.h"

#include "components/ggml/include/gguf.h"
#include "components/io/file_range_reader.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdexcept>
#include <sstream>
#include <utility>

namespace tyloi {

bool is_gguf_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    char magic[4]{};
    return file.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

namespace gguf {

using json = nlohmann::json;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Ggml = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

void Record::read(std::uint64_t relative_offset, std::byte* destination, std::size_t size) const {
    if (relative_offset > tensor.nbytes || size > tensor.nbytes - relative_offset)
        throw std::out_of_range("GGUF tensor read exceeds its byte range");
    if (!inline_bytes.empty()) {
        if (size) std::memcpy(destination, inline_bytes.data() + relative_offset, size);
        return;
    }
    if (relative_offset < prefix.size()) {
        const auto count = std::min<std::size_t>(size, prefix.size() - relative_offset);
        std::memcpy(destination, prefix.data() + relative_offset, count);
        relative_offset += count; destination += count; size -= count;
    }
    if (size) file->read(offset + relative_offset - prefix.size(), destination, size);
}

void Record::read(const std::vector<ReadSpan>& spans) const {
    for (const auto& span : spans) {
        if (span.offset > tensor.nbytes || span.size > tensor.nbytes - span.offset)
            throw std::out_of_range("GGUF tensor read exceeds its byte range");
        if (span.size && !span.destination) throw std::invalid_argument("GGUF tensor destination is null");
    }
    if (!prefix.empty() || !inline_bytes.empty()) {
        for (const auto& span : spans) read(span.offset, span.destination, span.size);
        return;
    }
    std::vector<ReadSpan> destinations;
    for (std::size_t begin = 0; begin < spans.size();) {
        destinations.clear();
        auto end = begin;
        auto next = spans[begin].offset;
        while (end < spans.size() && spans[end].offset == next) {
            destinations.push_back({offset + spans[end].offset, spans[end].destination, spans[end].size});
            next += spans[end++].size;
        }
        file->read_contiguous(destinations);
        begin = end;
    }
}

std::optional<TensorFileRange> Record::file_range() const {
    if (!prefix.empty() || !inline_bytes.empty()) return std::nullopt;
    return TensorFileRange{path, offset, tensor.nbytes};
}

namespace {

json scalar_value(const gguf_context* ctx, int64_t id) {
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx, id);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(ctx, id);
        case GGUF_TYPE_STRING: return gguf_get_val_str(ctx, id);
        default: return nullptr;
    }
}

json metadata_value(const gguf_context* ctx, int64_t id) {
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return scalar_value(ctx, id);
    const auto count = gguf_get_arr_n(ctx, id);
    const auto* data = gguf_get_arr_type(ctx, id) == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, id);
    json result = json::array();
    for (std::size_t i = 0; i < count; ++i) {
        switch (gguf_get_arr_type(ctx, id)) {
            case GGUF_TYPE_UINT8: result.push_back(static_cast<const uint8_t*>(data)[i]); break;
            case GGUF_TYPE_INT8: result.push_back(static_cast<const int8_t*>(data)[i]); break;
            case GGUF_TYPE_UINT16: result.push_back(static_cast<const uint16_t*>(data)[i]); break;
            case GGUF_TYPE_INT16: result.push_back(static_cast<const int16_t*>(data)[i]); break;
            case GGUF_TYPE_UINT32: result.push_back(static_cast<const uint32_t*>(data)[i]); break;
            case GGUF_TYPE_INT32: result.push_back(static_cast<const int32_t*>(data)[i]); break;
            case GGUF_TYPE_UINT64: result.push_back(static_cast<const uint64_t*>(data)[i]); break;
            case GGUF_TYPE_INT64: result.push_back(static_cast<const int64_t*>(data)[i]); break;
            case GGUF_TYPE_FLOAT32: result.push_back(static_cast<const float*>(data)[i]); break;
            case GGUF_TYPE_FLOAT64: result.push_back(static_cast<const double*>(data)[i]); break;
            case GGUF_TYPE_BOOL: result.push_back(static_cast<const uint8_t*>(data)[i] != 0); break;
            case GGUF_TYPE_STRING: result.push_back(gguf_get_arr_str(ctx, id, i)); break;
            default: throw std::runtime_error("unsupported GGUF metadata array");
        }
    }
    return result;
}

} // namespace

struct Shard::Impl {
    Gguf context;
    Ggml tensors;
    std::shared_ptr<FileRangeReader> file;
    std::filesystem::path path;
    std::uint64_t file_size;

    Impl(Gguf context, Ggml tensors, std::shared_ptr<FileRangeReader> file,
         std::filesystem::path path, std::uint64_t file_size)
        : context(std::move(context)), tensors(std::move(tensors)),
          file(std::move(file)), path(std::move(path)), file_size(file_size) {}
};

Shard::Shard(const std::filesystem::path& path, int split_count,
             std::size_t split_no) {
    ggml_context* raw = nullptr;
    Gguf context(gguf_init_from_file(path.string().c_str(), {true, &raw}), gguf_free);
    Ggml tensors(raw, ggml_free);
    if (!context) throw std::runtime_error("invalid GGUF file: " + path.filename().string());
    if (gguf_get_n_tensors(context.get()) > (1 << 20))
        throw std::runtime_error("GGUF has too many tensor records");
    if (split_count > 1) {
        const auto split_no_key = gguf_find_key(context.get(), "split.no");
        const auto split_count_key = gguf_find_key(context.get(), "split.count");
        if (split_no_key < 0 || split_count_key < 0 ||
            scalar_value(context.get(), split_no_key) != split_no ||
            scalar_value(context.get(), split_count_key) != split_count)
            throw std::runtime_error("inconsistent GGUF shard metadata");
    }
    const auto file_size = std::filesystem::file_size(path);
    auto file_reader = std::make_shared<FileRangeReader>(path, model_file_read_mode());
    impl_ = std::make_unique<Impl>(std::move(context), std::move(tensors),
                                   std::move(file_reader), path, file_size);
}

Shard::~Shard() = default;

std::int64_t Shard::count() const {
    return gguf_get_n_tensors(impl_->context.get());
}

json Shard::scalar(std::string_view key) const {
    const auto id = gguf_find_key(impl_->context.get(), std::string(key).c_str());
    return id < 0 ? json(nullptr) : scalar_value(impl_->context.get(), id);
}

json Shard::metadata() const {
    json result = json::object();
    for (std::int64_t i = 0; i < gguf_get_n_kv(impl_->context.get()); ++i) {
        const std::string name = gguf_get_key(impl_->context.get(), i);
        if (name.rfind("tokenizer.", 0) == 0 &&
            gguf_get_kv_type(impl_->context.get(), i) == GGUF_TYPE_ARRAY)
            continue;
        const auto value = metadata_value(impl_->context.get(), i);
        if (!value.is_null()) result[name] = value;
    }
    return result;
}

std::vector<std::byte> Shard::tokenizer() const {
    Gguf tokenizer(gguf_init_empty(), gguf_free);
    gguf_set_kv(tokenizer.get(), impl_->context.get());
    std::vector<std::byte> result(gguf_get_meta_size(tokenizer.get()));
    gguf_get_meta_data(tokenizer.get(), result.data());
    return result;
}

std::vector<std::string> Shard::names() const {
    std::vector<std::string> result;
    result.reserve(gguf_get_n_tensors(impl_->context.get()));
    for (std::int64_t i = 0; i < gguf_get_n_tensors(impl_->context.get()); ++i)
        result.emplace_back(gguf_get_tensor_name(impl_->context.get(), i));
    return result;
}

ShardTensorInfo Shard::tensor(std::int64_t index) const {
    const auto* name = gguf_get_tensor_name(impl_->context.get(), index);
    const auto* tensor = ggml_get_tensor(impl_->tensors.get(), name);
    if (!tensor) throw std::runtime_error("GGUF tensor shape is missing");
    std::vector<std::int64_t> shape;
    for (int d = ggml_n_dims(tensor) - 1; d >= 0; --d) shape.push_back(tensor->ne[d]);
    return {name, std::move(shape), static_cast<int>(gguf_get_tensor_type(impl_->context.get(), index)),
            ggml_nelements(tensor), gguf_get_data_offset(impl_->context.get()),
            gguf_get_tensor_offset(impl_->context.get(), index),
            gguf_get_tensor_size(impl_->context.get(), index), impl_->file_size,
            impl_->path, impl_->file};
}

int splits(const std::filesystem::path& path) {
    Gguf context(gguf_init_from_file(path.string().c_str(), {true, nullptr}), gguf_free);
    if (!context) throw std::runtime_error("invalid GGUF file: " + path.filename().string());
    const auto key = gguf_find_key(context.get(), "split.count");
    return key < 0 ? 1 : scalar_value(context.get(), key).get<int>();
}

std::vector<std::filesystem::path> paths(const std::filesystem::path& path, int splits) {
    if (splits < 1 || splits > 99999) throw std::runtime_error("invalid GGUF split count");
    std::vector<std::filesystem::path> result;
    if (splits > 1) {
        const auto filename = path.filename().string();
        const auto marker = filename.rfind("-of-");
        if (marker == std::string::npos || marker < 6 || filename.substr(filename.size() - 5) != ".gguf")
            throw std::runtime_error("split GGUF requires a -00001-of-000NN.gguf filename");
        const auto stem = filename.substr(0, marker - 6);
        for (int i = 1; i <= splits; ++i) {
            std::ostringstream name;
            name << stem << '-' << std::setfill('0') << std::setw(5) << i << "-of-" << std::setw(5) << splits << ".gguf";
            const auto shard = path.parent_path() / name.str();
            if (!std::filesystem::is_regular_file(shard)) throw std::runtime_error("missing GGUF shard: " + name.str());
            result.push_back(std::filesystem::absolute(shard));
        }
    } else result.push_back(std::filesystem::absolute(path));
    return result;
}

} // namespace gguf
} // namespace tyloi
