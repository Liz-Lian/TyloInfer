#include "engine/weights/quant.h"

#include "ops/common/include/format.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "formats/mxfp4_sq_decode.h"
#endif

#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace mfq::cuda::quant_format;

mfq::mxfp4::Data select_mxfp4_cpu_rows(
        const mfq::mxfp4::Data & source,
        const std::vector<int64_t> & rows) {
    mfq::mxfp4::Data result;
    result.out = static_cast<int64_t>(rows.size());
    result.neuron_len = source.neuron_len;
    const size_t value_row = static_cast<size_t>(source.neuron_len / 2);
    const size_t scale_row = static_cast<size_t>(source.neuron_len / 32);
    result.values.resize(rows.size() * value_row);
    result.scales.resize(rows.size() * scale_row);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("MXFP4 selected row is out of range");
        }
        std::memcpy(
            result.values.data() + destination * value_row,
            source.values.data() + static_cast<size_t>(source_row) * value_row,
            value_row);
        std::memcpy(
            result.scales.data() + destination * scale_row,
            source.scales.data() + static_cast<size_t>(source_row) * scale_row,
            scale_row);
    }
    return result;
}

mfq::mxfp4::Data slice_mxfp4_cpu(
        const mfq::mxfp4::Data & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid MXFP4 output shard");
        }
        std::vector<int64_t> rows(static_cast<size_t>(end - begin));
        std::iota(rows.begin(), rows.end(), begin);
        return select_mxfp4_cpu_rows(source, rows);
    }
    if (axis != TensorParallelAxis::Input || begin < 0 || begin >= end ||
            end > source.neuron_len || begin % 32 != 0 || end % 32 != 0) {
        throw std::runtime_error(
            "MXFP4 input shards must preserve 32-value scale blocks");
    }
    mfq::mxfp4::Data result;
    result.out = source.out;
    result.neuron_len = end - begin;
    const size_t source_value_row =
        static_cast<size_t>(source.neuron_len / 2);
    const size_t destination_value_row =
        static_cast<size_t>(result.neuron_len / 2);
    const size_t source_scale_row =
        static_cast<size_t>(source.neuron_len / 32);
    const size_t destination_scale_row =
        static_cast<size_t>(result.neuron_len / 32);
    result.values.resize(static_cast<size_t>(result.out) * destination_value_row);
    result.scales.resize(static_cast<size_t>(result.out) * destination_scale_row);
    for (int64_t row = 0; row < result.out; ++row) {
        std::memcpy(
            result.values.data() + static_cast<size_t>(row) * destination_value_row,
            source.values.data() + static_cast<size_t>(row) * source_value_row +
                static_cast<size_t>(begin / 2),
            destination_value_row);
        std::memcpy(
            result.scales.data() + static_cast<size_t>(row) * destination_scale_row,
            source.scales.data() + static_cast<size_t>(row) * source_scale_row +
                static_cast<size_t>(begin / 32),
            destination_scale_row);
    }
    return result;
}

mfq::mxfp8::Data slice_mxfp8_cpu(
        const mfq::mxfp8::Data & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis != TensorParallelAxis::Output &&
            axis != TensorParallelAxis::Input) {
        throw std::runtime_error("MXFP8 slicing requires output or input axis");
    }
    const int64_t extent = axis == TensorParallelAxis::Output
        ? source.out : source.neuron_len;
    const int block=axis==TensorParallelAxis::Output?source.block_rows:source.block_columns;
    if (begin < 0 || begin >= end || end > extent ||
            begin % block != 0 || (end != extent && end % block != 0)) {
        throw std::runtime_error(
            "MXFP8 tensor-parallel shards must preserve scale blocks");
    }
    mfq::mxfp8::Data result;result.block_rows=source.block_rows;result.block_columns=source.block_columns;
    if (axis == TensorParallelAxis::Output) {
        result.out = end - begin;
        result.neuron_len = source.neuron_len;
        const size_t value_begin =
            static_cast<size_t>(begin * source.neuron_len);
        const size_t value_end =
            static_cast<size_t>(end * source.neuron_len);
        result.values.assign(
            source.values.begin() + static_cast<ptrdiff_t>(value_begin),
            source.values.begin() + static_cast<ptrdiff_t>(value_end));
        const int64_t scale_columns = source.neuron_len / source.block_columns;
        const size_t scale_begin =
            static_cast<size_t>((begin / source.block_rows) * scale_columns);
        const size_t scale_end = static_cast<size_t>(
            ((end + source.block_rows-1) / source.block_rows) * scale_columns);
        result.scales.assign(
            source.scales.begin() + static_cast<ptrdiff_t>(scale_begin),
            source.scales.begin() + static_cast<ptrdiff_t>(scale_end));
        return result;
    }

    result.out = source.out;
    result.neuron_len = end - begin;
    result.values.resize(
        static_cast<size_t>(result.out * result.neuron_len));
    for (int64_t row = 0; row < source.out; ++row) {
        std::memcpy(
            result.values.data() +
                static_cast<size_t>(row * result.neuron_len),
            source.values.data() +
                static_cast<size_t>(row * source.neuron_len + begin),
            static_cast<size_t>(result.neuron_len));
    }
    const int64_t source_scale_columns = source.neuron_len / source.block_columns;
    const int64_t result_scale_columns = result.neuron_len / source.block_columns;
    const int64_t scale_rows = (source.out + source.block_rows-1) / source.block_rows;
    result.scales.resize(
        static_cast<size_t>(scale_rows * result_scale_columns));
    for (int64_t row = 0; row < scale_rows; ++row) {
        std::memcpy(
            result.scales.data() +
                static_cast<size_t>(row * result_scale_columns),
            source.scales.data() + static_cast<size_t>(
                row * source_scale_columns + begin / source.block_columns),
            static_cast<size_t>(result_scale_columns));
    }
    return result;
}

Mxfp8Weight to_device_mxfp8(
        const mfq::mxfp8::Data & source,
        bool cuda,
        int device) {
    Mxfp8Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values,
        {source.out, source.neuron_len});
    result.scales = cpu_u8_tensor(
        source.scales,
        {(source.out + source.block_rows-1) / source.block_rows, source.neuron_len / source.block_columns});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

Mxfp4Weight to_device_mxfp4(
        const mfq::mxfp4::Data & source, bool cuda, int device) {
    Mxfp4Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values, {source.out, source.neuron_len / 2});
    result.scales = cpu_u8_tensor(
        source.scales, {source.out, source.neuron_len / 32});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
Mxfp4SqWeight to_device_mxfp4_sq(
        const std::vector<uint8_t> & payload,
        bool cuda,
        int device) {
    const auto layout = mfq::sq::parse(payload.data(), payload.size());
    const auto rows = mfq::sq::row_metadata(payload.data(), layout);
    Mxfp4SqWeight result;
    result.bits = layout.bits;
    result.out = layout.outputs;
    result.neuron_len = layout.width;
    result.matrix_scale_base = layout.base;
    result.sq4_rows = layout.sq4_rows;
    result.format_version = layout.version;
    result.aggregate_bpw = static_cast<double>(layout.bytes - 24) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<size_t, 4> q_counts{};
    std::vector<int32_t> symbol_offsets(rows.symbol_byte_offsets.size());
    std::vector<int32_t> auxiliary_rows(rows.auxiliary_rows.size());
    for (size_t row = 0; row < rows.q.size(); ++row) {
        const int q = rows.q[row];
        result.q_sum += q;
        ++q_counts[static_cast<size_t>(q - 1)];
        if (rows.symbol_byte_offsets[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
                rows.auxiliary_rows[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "MXFP4-SQ runtime row metadata exceeds int32 limits");
        }
        symbol_offsets[row] =
            static_cast<int32_t>(rows.symbol_byte_offsets[row]);
        auxiliary_rows[row] =
            static_cast<int32_t>(rows.auxiliary_rows[row]);
    }
    for (const auto count : q_counts) {
        if (count == 0) continue;
        const double probability =
            static_cast<double>(count) / layout.outputs;
        result.distribution_entropy -= probability * std::log2(probability);
    }
    result.blob = cpu_u8_tensor(
        payload,
        {static_cast<int64_t>(payload.size())});
    result.row_q = cpu_u8_tensor(rows.q, {layout.outputs});
    result.row_symbol_byte_offsets = cpu_i32_tensor(
        symbol_offsets, {layout.outputs});
    result.row_auxiliary = cpu_i32_tensor(
        auxiliary_rows, {layout.outputs});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        result.blob = result.blob.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
        result.row_q = result.row_q.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
        result.row_symbol_byte_offsets =
            result.row_symbol_byte_offsets.to(
                mfq_tensor_backend::Device(
                    mfq_tensor_backend::kCUDA,
                    target_device),
                false,
                false).contiguous();
        result.row_auxiliary = result.row_auxiliary.to(
            mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                target_device),
            false,
            false).contiguous();
    }
    return result;
}

Fp8SqWeight to_device_fp8_sq(
        std::string_view dtype,
        const std::vector<uint8_t> & payload,
        bool cuda,
        int device) {
    const auto layout = mfq::fp8sq::parse(
        dtype, payload.data(), payload.size());
    const auto rows = mfq::fp8sq::row_metadata(payload.data(), layout);
    Fp8SqWeight result;
    result.dtype = std::string(dtype);
    result.out = layout.outputs;
    result.neuron_len = layout.width;
    result.block_rows = layout.block_rows;
    result.block_columns = layout.block_columns;
    result.scale_rows = layout.scale_rows;
    result.scale_columns = layout.scale_columns;
    result.scale_kind = static_cast<int64_t>(layout.scale_kind);
    result.palettes_offset = static_cast<int64_t>(layout.palettes);
    result.symbols_offset = static_cast<int64_t>(layout.symbols);
    result.scales_offset = static_cast<int64_t>(layout.scales);
    result.format_version = layout.version;
    result.aggregate_bpw = static_cast<double>(
        layout.bytes - mfq::fp8sq::kHeaderBytes) * 8.0 /
        (static_cast<double>(layout.outputs) * layout.width);
    std::array<size_t, 8> q_counts{};
    std::vector<int32_t> symbol_offsets(rows.symbol_byte_offsets.size());
    for (size_t row = 0; row < rows.q.size(); ++row) {
        const int q = rows.q[row];
        ++q_counts[static_cast<size_t>(q - 1)];
        if (rows.symbol_byte_offsets[row] >
                static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error(
                "FP8-SQ runtime row metadata exceeds int32 limits");
        }
        symbol_offsets[row] =
            static_cast<int32_t>(rows.symbol_byte_offsets[row]);
    }
    for (const auto count : q_counts) {
        if (count == 0) continue;
        const double probability =
            static_cast<double>(count) / layout.outputs;
        result.distribution_entropy -= probability * std::log2(probability);
    }
    result.blob = cpu_u8_tensor(
        payload, {static_cast<int64_t>(payload.size())});
    result.row_q = cpu_u8_tensor(rows.q, {layout.outputs});
    result.row_symbol_byte_offsets = cpu_i32_tensor(
        symbol_offsets, {layout.outputs});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, target_device);
        result.blob = result.blob.to(target, false, false).contiguous();
        result.row_q = result.row_q.to(target, false, false).contiguous();
        result.row_symbol_byte_offsets =
            result.row_symbol_byte_offsets.to(
                target, false, false).contiguous();
    }
    return result;
}
#endif
