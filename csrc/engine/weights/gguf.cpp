#include "engine/weights/gguf.h"

#include "components/ggml/include/ggml.h"
#include "components/io/file_range_reader.h"
#include "engine/weights/encoding/dense_blob.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tyloi::gguf {
namespace {

std::vector<std::byte> read_record(const Record& record) {
    if (record.tensor.nbytes > std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("model tensor is too large to read");
    std::vector<std::byte> result(static_cast<std::size_t>(record.tensor.nbytes));
    record.read(0, result.data(), result.size());
    return result;
}

} // namespace

Record prepare(const ShardTensorInfo& tensor, std::string name,
               std::vector<std::int64_t> shape) {
    Record record;
    record.file = tensor.file;
    record.path = tensor.path;
    if (tensor.tensor_offset > std::numeric_limits<std::uint64_t>::max() - tensor.data_offset)
        throw std::runtime_error("GGUF tensor offset overflow: " + name);
    record.offset = tensor.data_offset + tensor.tensor_offset;
    if (record.offset > tensor.file_size || tensor.nbytes > tensor.file_size - record.offset)
        throw std::runtime_error("truncated GGUF tensor: " + name);
    const auto type = static_cast<ggml_type>(tensor.type);
    std::string dtype;
    if (type == GGML_TYPE_F32) dtype = "F32";
    else if (type == GGML_TYPE_F16) dtype = "F16";
    else if (type == GGML_TYPE_BF16) dtype = "BF16";
    else if (type == GGML_TYPE_I32) dtype = "I32";
    else if (type == GGML_TYPE_I64) dtype = "I64";
    else if (ggml_is_quantized(type)) dtype = "GGUF_" + std::string(ggml_type_name(type));
    else throw std::runtime_error("unsupported GGUF tensor type: " + std::string(ggml_type_name(type)));
    std::transform(dtype.begin(), dtype.end(), dtype.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    if (!ggml_is_quantized(type)) record.prefix = dense::prefix(shape);
    record.tensor = {std::move(name), dtype, dtype, std::move(shape),
                     tensor.nbytes + record.prefix.size()};
    return record;
}

void normalize(Record& record, const ShardTensorInfo& tensor, bool decay_log) {
    const auto type = static_cast<ggml_type>(tensor.type);
    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)
        throw std::runtime_error("GGUF normalization requires floating-point tensors: " + record.tensor.name);
    std::vector<std::byte> raw(static_cast<std::size_t>(tensor.nbytes));
    tensor.file->read(record.offset, raw.data(), raw.size());
    record.inline_bytes = dense::prefix(*record.tensor.shape);
    const auto begin = record.inline_bytes.size();
    record.inline_bytes.resize(begin + tensor.elements * sizeof(float));
    for (std::int64_t j = 0; j < tensor.elements; ++j) {
        float value = 0;
        if (type == GGML_TYPE_F32) std::memcpy(&value, raw.data() + j * 4, 4);
        else {
            std::uint16_t bits;
            std::memcpy(&bits, raw.data() + j * 2, 2);
            if (type == GGML_TYPE_F16) value = ggml_fp16_to_fp32(bits);
            else {
                const std::uint32_t expanded = std::uint32_t(bits) << 16;
                std::memcpy(&value, &expanded, 4);
            }
        }
        if (decay_log) {
            if (!std::isfinite(value) || value >= 0)
                throw std::runtime_error("invalid GGUF GDN decay: " + record.tensor.name);
            value = std::log(-value);
        } else value -= 1.f;
        std::memcpy(record.inline_bytes.data() + begin + j * 4, &value, 4);
    }
    record.tensor.dtype = record.tensor.stored_dtype = "F32";
    record.tensor.nbytes = record.inline_bytes.size();
}

Record join(std::string name, const Record& query, const Record& key) {
    const auto& query_tensor = query.tensor;
    const auto& key_tensor = key.tensor;
    if (!query_tensor.shape || !key_tensor.shape || query_tensor.shape->size() != 2 ||
        key_tensor.shape->size() != 2 || query_tensor.shape->back() != key_tensor.shape->back() ||
        query_tensor.dtype != key_tensor.dtype ||
        (query_tensor.dtype != "F32" && query_tensor.dtype != "F16" && query_tensor.dtype != "BF16"))
        throw std::runtime_error("GGUF indexer query/key layouts cannot be joined without changing precision");
    const std::vector<std::int64_t> shape{
        query_tensor.shape->front() + key_tensor.shape->front(), query_tensor.shape->back()};
    Record result;
    result.inline_bytes = dense::prefix(shape);
    for (const auto* record : {&query, &key}) {
        const auto bytes = read_record(*record);
        result.inline_bytes.insert(result.inline_bytes.end(),
                                   bytes.begin() + record->prefix.size(), bytes.end());
    }
    result.tensor = {std::move(name), query_tensor.dtype, query_tensor.stored_dtype,
                     shape, result.inline_bytes.size()};
    return result;
}

Record integers(std::string name, const std::vector<std::int64_t>& values) {
    Record record;
    const std::vector<std::int64_t> shape{static_cast<std::int64_t>(values.size())};
    record.inline_bytes = dense::prefix(shape);
    const auto start = record.inline_bytes.size();
    record.inline_bytes.resize(start + values.size() * sizeof(std::int64_t));
    std::memcpy(record.inline_bytes.data() + start, values.data(), values.size() * sizeof(std::int64_t));
    record.tensor = {std::move(name), "I64", "I64", shape, record.inline_bytes.size()};
    return record;
}

} // namespace tyloi::gguf
