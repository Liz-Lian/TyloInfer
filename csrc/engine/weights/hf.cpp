#include "hf.h"

#include "encoding/dense_blob.h"
#include "encoding/fp8_sq_blob.h"
#include "encoding/mx_blob.h"
#include "encoding/quant/mfe_blob.h"
#include "mlx.h"
#include "source/safetensors.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace mfq::hf {
namespace {

std::uint64_t checked_add(
    std::uint64_t left,
    std::uint64_t right,
    std::string_view what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error(std::string(what) + " byte size overflow");
    }
    return left + right;
}

std::uint64_t checked_product(
    const std::vector<std::int64_t>& shape,
    std::uint64_t item_size,
    const std::string& name) {
    auto result = item_size;
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            static_cast<std::uint64_t>(dimension) >
                std::numeric_limits<std::uint64_t>::max() / result) {
            throw std::runtime_error("invalid Safetensors tensor shape: " + name);
        }
        result *= static_cast<std::uint64_t>(dimension);
    }
    return result;
}

std::uint64_t dense_item_size(std::string_view dtype) {
    if (dtype == "BF16" || dtype == "F16") return 2;
    if (dtype == "F32" || dtype == "I32") return 4;
    if (dtype == "I64") return 8;
    return 0;
}

std::string find_scale(
    const Safetensors& files,
    const std::vector<std::string>& names,
    bool block_fp8) {
    for (const auto& name : names) {
        const auto* scale = files.find(name);
        if (scale != nullptr &&
            ((block_fp8 && (scale->dtype == "BF16" || scale->dtype == "F16" ||
                            scale->dtype == "F32")) ||
             (!block_fp8 && scale->dtype == "F8_E8M0"))) {
            return name;
        }
    }
    return {};
}

void append(
    Record& aggregate,
    std::uint64_t& logical_offset,
    std::vector<std::byte> bytes) {
    const auto count = bytes.size();
    aggregate.segments.push_back({logical_offset, std::move(bytes), {}, 0, count});
    logical_offset = checked_add(
        logical_offset, count, "virtual MFE metadata");
}

void append(
    Record& aggregate,
    std::uint64_t& logical_offset,
    const Segment& segment) {
    if (!segment.inline_bytes.empty()) {
        append(aggregate, logical_offset, segment.inline_bytes);
    } else {
        aggregate.segments.push_back({
            logical_offset, {}, segment.source_name,
            segment.source_offset, segment.nbytes});
        logical_offset = checked_add(
            logical_offset, segment.nbytes, "virtual MFE tensor");
    }
}

} // namespace

Record prepare(
    const Safetensors& files,
    const TensorMetadata& values,
    std::string name,
    const std::vector<std::int64_t>& dense_shape,
    const std::vector<std::string>& scale_names,
    const nlohmann::json& quantization) {
    if (!values.shape) {
        throw std::runtime_error("HF tensor shape is missing: " + values.name);
    }

    Record record;
    record.metadata.name = std::move(name);
    record.metadata.stored_dtype = values.dtype;
    record.metadata.shape = values.shape;
    const auto item_size = dense_item_size(values.dtype);
    if (values.dtype == "U32") {
        const auto binding = mlx_affine_binding(files, quantization, values);
        const auto& layout = binding.layout;
        record.metadata.dtype = "MLX-AFFINE";
        record.metadata.shape = layout.shape;
        record.segments.push_back({0, binding.prefix, {}, 0, 0});
        record.segments.push_back({layout.prefix_bytes, {}, values.name, 0, layout.values_bytes});
        record.segments.push_back({layout.scales_offset(), {}, binding.scales, 0, layout.scales_bytes});
        record.segments.push_back({layout.biases_offset(), {}, binding.biases, 0, layout.scales_bytes});
        record.metadata.nbytes = layout.total_bytes();
    } else if (item_size != 0) {
        if (checked_product(*values.shape, item_size, values.name) !=
            values.nbytes) {
            throw std::runtime_error(
                "dense Safetensors byte size mismatch: " + values.name);
        }
        record.metadata.dtype = values.dtype;
        record.metadata.shape = dense_shape;
        auto prefix = dense::prefix(*record.metadata.shape);
        record.segments.push_back({0, std::move(prefix), {}, 0, 0});
        const auto values_offset = record.segments.front().inline_bytes.size();
        record.segments.push_back({
            values_offset, {}, values.name, 0, values.nbytes});
        record.metadata.nbytes = checked_add(
            values_offset, values.nbytes, "dense tensor");
    } else if (values.dtype == "I8" || values.dtype == "F8_E4M3" ||
               values.dtype == "F8_E4M3FN") {
        if (checked_product(*values.shape, 1, values.name) != values.nbytes) {
            throw std::runtime_error(
                "native FP8/MX values byte size mismatch: " + values.name);
        }
        auto logical_shape = *values.shape;
        std::string scale_name;
        if (values.dtype != "I8") scale_name = find_scale(files, scale_names, true);
        if (!scale_name.empty()) {
            record.metadata.dtype = "FP8-128SQ";
        } else {
            scale_name = find_scale(files, scale_names, false);
            if (scale_name.empty()) {
                throw std::runtime_error(
                    "native FP8/MX tensor has no supported scale binding: " +
                    values.name);
            }
            if (values.dtype == "I8") {
                if (logical_shape.size() != 2 || logical_shape[1] >
                        std::numeric_limits<std::int64_t>::max() / 2) {
                    throw std::runtime_error(
                        "invalid native MXFP4 tensor shape: " + values.name);
                }
                logical_shape[1] *= 2;
                record.metadata.dtype = "MXFP4";
            } else {
                record.metadata.dtype = "MXFP8";
            }
        }
        record.metadata.stored_dtype = record.metadata.dtype;
        record.metadata.shape = logical_shape;
        const auto* scales = files.find(scale_name);
        if (scales == nullptr || !scales->shape) {
            throw std::runtime_error("native FP8/MX scale shape is missing: " +
                                     scale_name);
        }
        const auto scale_item_size = dense_item_size(scales->dtype);
        const bool block_fp8 = record.metadata.dtype == "FP8-128SQ";
        const auto expected_scale_bytes = block_fp8
            ? checked_product(*scales->shape, scale_item_size, scale_name)
            : checked_product(*scales->shape, 1, scale_name);
        if ((block_fp8 && scale_item_size == 0) ||
            expected_scale_bytes != scales->nbytes) {
            throw std::runtime_error(
                "native FP8/MX scale byte size mismatch: " + scale_name);
        }
        auto prefix = record.metadata.dtype == "FP8-128SQ"
            ? fp8sq::block128_prefix(
                logical_shape, scales->dtype, *scales->shape, values.name)
            : mx::prefix(record.metadata.dtype, logical_shape,
                         *values.shape, *scales->shape, values.name);
        const auto values_offset = prefix.size();
        const auto scales_offset = checked_add(
            values_offset, values.nbytes, "native FP8/MX tensor");
        record.segments.push_back({0, std::move(prefix), {}, 0, 0});
        record.segments.push_back({
            values_offset, {}, values.name, 0, values.nbytes});
        record.segments.push_back({
            scales_offset, {}, scale_name, 0, scales->nbytes});
        record.metadata.nbytes = checked_add(
            scales_offset, scales->nbytes, "native FP8/MX tensor");
    } else {
        throw std::runtime_error("unsupported full-precision HF dtype " +
                                 values.dtype + ": " + values.name);
    }
    return record;
}

Record experts(std::string name, const std::map<std::size_t, const Record*>& records) {
    const auto expert_count = records.rbegin()->first + 1;
    if (records.size() != expert_count) {
        throw std::runtime_error(
            "native HF expert projection has missing expert IDs: " + name);
    }

    Record aggregate;
    aggregate.metadata.name = std::move(name);
    aggregate.metadata.dtype = "MFE";
    aggregate.metadata.stored_dtype = "MFE";
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::uint64_t logical_offset = 0;
    for (const auto& [expert, record] : records) {
        const auto& tensor_name = record->metadata.name;
        if (!record->metadata.shape || record->metadata.shape->size() != 2 ||
            record->segments.empty()) {
            throw std::runtime_error(
                "native HF expert must be rank two: " + tensor_name);
        }
        if (dtype.empty()) {
            dtype = record->metadata.dtype;
            shape = *record->metadata.shape;
            if (shape[0] <= 0 || shape[1] <= 0 ||
                shape[0] > std::numeric_limits<std::uint32_t>::max() ||
                shape[1] > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error(
                    "native HF expert geometry exceeds MFE: " + tensor_name);
            }
            const auto output = static_cast<std::uint32_t>(shape[0]);
            const auto input = static_cast<std::uint32_t>(shape[1]);
            append(aggregate, logical_offset, mfe::header(
                static_cast<std::uint32_t>(expert_count), output, input,
                static_cast<std::uint32_t>(expert_count)));
        } else if (dtype != record->metadata.dtype ||
                   shape != *record->metadata.shape) {
            throw std::runtime_error(
                "native HF expert projection has mixed geometry: " +
                aggregate.metadata.name);
        }

        auto pool = mfe::expert_prefix(
            static_cast<std::int32_t>(expert), dtype, record->metadata.nbytes);
        pool.insert(pool.end(), record->segments.front().inline_bytes.begin(),
                    record->segments.front().inline_bytes.end());
        append(aggregate, logical_offset, std::move(pool));
        for (std::size_t index = 1; index < record->segments.size(); ++index) {
            append(aggregate, logical_offset, record->segments[index]);
        }
    }
    aggregate.metadata.nbytes = logical_offset;
    return aggregate;
}

} // namespace mfq::hf
