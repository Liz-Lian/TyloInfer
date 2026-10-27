#include "engine/weights/quant.h"
#include "ops/common/include/format.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace mfq::cuda::quant_format;

static uint64_t nint_row_payload_bits(
        int ng, int gs, int bits) {
    if (ng <= 0 || gs <= 0 || bits < 1 || bits > 8 ||
            static_cast<uint64_t>(ng) >
                std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(gs) /
                    static_cast<uint64_t>(bits)) {
        throw std::runtime_error("invalid NINT row bit geometry");
    }
    return static_cast<uint64_t>(ng) *
        static_cast<uint64_t>(gs) * static_cast<uint64_t>(bits);
}

static void copy_nint_packed_bits(
        const std::vector<uint8_t> & source,
        uint64_t source_bit,
        std::vector<uint8_t> & destination,
        uint64_t destination_bit,
        uint64_t bit_count) {
    const uint64_t source_capacity =
        static_cast<uint64_t>(source.size()) * 8u;
    const uint64_t destination_capacity =
        static_cast<uint64_t>(destination.size()) * 8u;
    if (source_bit > source_capacity ||
            bit_count > source_capacity - source_bit ||
            destination_bit > destination_capacity ||
            bit_count > destination_capacity - destination_bit) {
        throw std::runtime_error("NINT packed bit range is out of bounds");
    }
    if (((source_bit | destination_bit | bit_count) & 7u) == 0u) {
        std::memcpy(
            destination.data() + static_cast<size_t>(destination_bit >> 3),
            source.data() + static_cast<size_t>(source_bit >> 3),
            static_cast<size_t>(bit_count >> 3));
        return;
    }
    while (bit_count != 0) {
        const int source_shift = static_cast<int>(source_bit & 7u);
        const int destination_shift = static_cast<int>(destination_bit & 7u);
        const int chunk = static_cast<int>(std::min<uint64_t>(
            bit_count,
            static_cast<uint64_t>(std::min(
                8 - source_shift, 8 - destination_shift))));
        const uint8_t mask = static_cast<uint8_t>((1u << chunk) - 1u);
        const uint8_t value = static_cast<uint8_t>(
            (source[static_cast<size_t>(source_bit >> 3)] >> source_shift) &
            mask);
        destination[static_cast<size_t>(destination_bit >> 3)] |=
            static_cast<uint8_t>(value << destination_shift);
        source_bit += static_cast<uint64_t>(chunk);
        destination_bit += static_cast<uint64_t>(chunk);
        bit_count -= static_cast<uint64_t>(chunk);
    }
}

static void require_canonical_nint_cpu(const mfq::nint::Data & source) {
    if (source.out <= 0 ||
            source.row_q_bits.size() != static_cast<size_t>(source.out) ||
            source.row_q_bit_offsets.size() != static_cast<size_t>(source.out) ||
            source.row_sub_bits.size() != static_cast<size_t>(source.out) ||
            source.q_packed.size() < 8) {
        throw std::runtime_error("NINT storage is missing canonical row metadata");
    }
    const uint64_t payload_capacity =
        static_cast<uint64_t>(source.q_packed.size() - 8) * 8u;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const int64_t signed_offset =
            source.row_q_bit_offsets[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        if (signed_offset < 0 ||
                static_cast<uint64_t>(signed_offset) > payload_capacity ||
                row_bits > payload_capacity -
                    static_cast<uint64_t>(signed_offset)) {
            throw std::runtime_error("invalid canonical NINT row metadata");
        }
    }
}

static mfq::nint::Data repack_nint_cpu_rows(
        const mfq::nint::Data & source,
        const std::vector<int64_t> & rows) {
    require_canonical_nint_cpu(source);
    if (rows.empty()) {
        throw std::runtime_error("cannot create an empty NINT row selection");
    }
    uint64_t total_bits = 0;
    for (int64_t row : rows) {
        if (row < 0 || row >= source.out) {
            throw std::runtime_error("NINT selected row is out of range");
        }
        const uint64_t row_bits = nint_row_payload_bits(
            source.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT selected row payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT selected row payload is too large");
    }
    mfq::nint::Data result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bits.resize(rows.size());
    result.row_q_bit_offsets.resize(rows.size());
    result.row_sub_bits.resize(rows.size());
    result.sub_scale.resize(rows.size() * static_cast<size_t>(source.ng));
    result.sub_min.resize(rows.size() * static_cast<size_t>(source.ng));
    result.neuron_scale_h.resize(rows.size());
    result.neuron_min_h.resize(rows.size());
    uint64_t destination_bit = 0;
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const size_t source_row = static_cast<size_t>(rows[destination]);
        const int bits = source.row_q_bits[source_row];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        result.row_q_bits[destination] = static_cast<uint8_t>(bits);
        result.row_sub_bits[destination] = source.row_sub_bits[source_row];
        result.row_q_bit_offsets[destination] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed,
            static_cast<uint64_t>(source.row_q_bit_offsets[source_row]),
            result.q_packed, destination_bit, row_bits);
        const size_t source_group = source_row * static_cast<size_t>(source.ng);
        const size_t destination_group =
            destination * static_cast<size_t>(source.ng);
        std::copy_n(
            source.sub_scale.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_scale.begin() + static_cast<ptrdiff_t>(destination_group));
        std::copy_n(
            source.sub_min.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_min.begin() + static_cast<ptrdiff_t>(destination_group));
        result.neuron_scale_h[destination] = source.neuron_scale_h[source_row];
        result.neuron_min_h[destination] = source.neuron_min_h[source_row];
        destination_bit += row_bits;
    }
    mfq::nint::describe(result);
    return result;
}

mfq::nint::Data slice_nint_cpu_output(
        const mfq::nint::Data & source, int64_t begin, int64_t end) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT output shard");
    }
    std::vector<int64_t> rows(static_cast<size_t>(end - begin));
    std::iota(rows.begin(), rows.end(), begin);
    return repack_nint_cpu_rows(source, rows);
}

mfq::nint::Data slice_nint_cpu_input_groups(
        const mfq::nint::Data & source, int64_t begin, int64_t end) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT input shard");
    }
    require_canonical_nint_cpu(source);
    mfq::nint::Data result = source;
    result.ng = static_cast<int>(end - begin);
    const int64_t element_begin = begin * source.gs;
    const int64_t element_end =
        std::min<int64_t>(end * source.gs, source.neuron_len);
    result.neuron_len = static_cast<int>(element_end - element_begin);
    result.shape[1] = result.neuron_len;
    uint64_t total_bits = 0;
    for (int row = 0; row < source.out; ++row) {
        const uint64_t row_bits = nint_row_payload_bits(
            result.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT input shard payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT input shard payload is too large");
    }
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bit_offsets.resize(static_cast<size_t>(source.out));
    result.sub_scale.resize(
        static_cast<size_t>(source.out) * result.ng);
    result.sub_min.resize(
        static_cast<size_t>(source.out) * result.ng);
    uint64_t destination_bit = 0;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(result.ng, source.gs, bits);
        const uint64_t source_bit = static_cast<uint64_t>(
            source.row_q_bit_offsets[static_cast<size_t>(row)]) +
            static_cast<uint64_t>(begin) *
                static_cast<uint64_t>(source.gs) *
                static_cast<uint64_t>(bits);
        result.row_q_bit_offsets[static_cast<size_t>(row)] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed, source_bit,
            result.q_packed, destination_bit, row_bits);
        destination_bit += row_bits;
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.sub_scale.data() + destination_group,
            source.sub_scale.data() + source_group,
            static_cast<size_t>(result.ng));
        std::memcpy(
            result.sub_min.data() + destination_group,
            source.sub_min.data() + source_group,
            static_cast<size_t>(result.ng));
    }
    mfq::nint::describe(result);
    return result;
}

mfq::nint8::Data slice_nint8_zero_cpu_output(
        const mfq::nint8::Data & source, int64_t begin, int64_t end) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT8-0 output shard");
    }
    mfq::nint8::Data result = source;
    result.out = static_cast<int>(end - begin);
    result.shape[0] = result.out;
    const size_t first_group =
        static_cast<size_t>(begin) * source.ng;
    const size_t group_count =
        static_cast<size_t>(result.out) * source.ng;
    result.q.assign(
        source.q.begin() +
            static_cast<ptrdiff_t>(first_group * 32),
        source.q.begin() +
            static_cast<ptrdiff_t>((first_group + group_count) * 32));
    result.scale_h.assign(
        source.scale_h.begin() + static_cast<ptrdiff_t>(first_group),
        source.scale_h.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    return result;
}

mfq::nint8::Data slice_nint8_zero_cpu_input_groups(
        const mfq::nint8::Data & source, int64_t begin, int64_t end) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT8-0 input shard");
    }
    mfq::nint8::Data result = source;
    result.ng = static_cast<int>(end - begin);
    result.neuron_len = result.ng * 32;
    result.shape[1] = result.neuron_len;
    result.q.resize(
        static_cast<size_t>(source.out) * result.ng * 32);
    result.scale_h.resize(
        static_cast<size_t>(source.out) * result.ng);
    for (int row = 0; row < source.out; ++row) {
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.q.data() + destination_group * 32,
            source.q.data() + source_group * 32,
            static_cast<size_t>(result.ng) * 32);
        std::memcpy(
            result.scale_h.data() + destination_group,
            source.scale_h.data() + source_group,
            static_cast<size_t>(result.ng) * sizeof(uint16_t));
    }
    return result;
}
NintWeight to_device_nint(
        const mfq::nint::Data & c, bool cuda, int device) {
    if (c.bits < 1 || c.bits > 8) throw std::runtime_error("unsupported NINT bits");
    require_canonical_nint_cpu(c);
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = c.gs;
    w.bits = c.bits;
    w.neuron_len = c.neuron_len;
    w.format_version = c.format_version;
    w.aggregate_bpw = c.aggregate_bpw;
    w.distribution_entropy = c.distribution_entropy;
    w.shape = c.shape;
    w.aligned_q8 = c.gs % 4 == 0;
    w.uniform_qbits=c.row_q_bits.empty()?0:c.row_q_bits.front();
    for (size_t row = 0; row < c.row_q_bits.size(); ++row) {
        if(c.row_q_bits[row]!=w.uniform_qbits)w.uniform_qbits=0;
        w.aligned_q8 = w.aligned_q8 && c.row_q_bits[row] == 8 &&
            (c.row_q_bit_offsets[row] & 31) == 0;
    }
    w.q_packed = cpu_u8_tensor(
        c.q_packed, {static_cast<int64_t>(c.q_packed.size())});
    w.row_q_bits = cpu_u8_tensor(c.row_q_bits, {c.out});
    w.row_q_bit_offsets = cpu_i64_tensor(c.row_q_bit_offsets, {c.out});
    w.sub_scale = cpu_u8_tensor(c.sub_scale, {c.out, c.ng});
    w.sub_min = cpu_u8_tensor(c.sub_min, {c.out, c.ng});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    w.neuron_min = cpu_f16_to_f32_tensor(c.neuron_min_h, c.out);
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, target_device);
        w.q_packed = w.q_packed.to(target).contiguous();
        w.row_q_bits = w.row_q_bits.to(target).contiguous();
        w.row_q_bit_offsets = w.row_q_bit_offsets.to(target).contiguous();
        w.sub_scale = w.sub_scale.to(target).contiguous();
        w.sub_min = w.sub_min.to(target).contiguous();
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.neuron_min = w.neuron_min.to(target).contiguous();
    }
    return w;
}


NintWeight to_device_mfe_nint(
        const mfq::nint::Data & source,
        int local_experts,
        int out_per_expert,
        bool cuda) {
    require_canonical_nint_cpu(source);
    if (local_experts <= 0 || out_per_expert <= 0 ||
            source.out != local_experts * out_per_expert) {
        throw std::runtime_error("invalid MFE NINT expert geometry");
    }
    uint64_t maximum_payload_bits = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t payload_bits = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const uint64_t row_bits = nint_row_payload_bits(
                source.ng, source.gs, source.row_q_bits[row]);
            if (row_bits > std::numeric_limits<uint64_t>::max() - payload_bits) {
                throw std::runtime_error("MFE NINT expert payload is too large");
            }
            payload_bits += row_bits;
        }
        maximum_payload_bits = std::max(maximum_payload_bits, payload_bits);
    }
    const uint64_t stride_u64 = (maximum_payload_bits + 7u) / 8u + 8u;
    if (stride_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            static_cast<uint64_t>(local_experts) >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                    stride_u64) {
        throw std::runtime_error("MFE NINT expert payload is too large");
    }
    const size_t stride = static_cast<size_t>(stride_u64);
    std::vector<uint8_t> expert_stream(
        static_cast<size_t>(local_experts) * stride, 0);
    std::vector<int64_t> relative_offsets(static_cast<size_t>(source.out));
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t destination_bit = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const int bits = source.row_q_bits[row];
            const uint64_t row_bits =
                nint_row_payload_bits(source.ng, source.gs, bits);
            relative_offsets[row] = static_cast<int64_t>(destination_bit);
            copy_nint_packed_bits(
                source.q_packed,
                static_cast<uint64_t>(source.row_q_bit_offsets[row]),
                expert_stream,
                static_cast<uint64_t>(expert) * stride_u64 * 8u +
                    destination_bit,
                row_bits);
            destination_bit += row_bits;
        }
    }
    NintWeight result = to_device_nint(source, false);
    result.aligned_q8 = false;
    result.q_expert_stride = static_cast<int64_t>(stride);
    result.q_packed = cpu_u8_tensor(
        expert_stream, {local_experts, static_cast<int64_t>(stride)});
    result.row_q_bit_offsets = cpu_i64_tensor(
        relative_offsets, {source.out});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        result.q_packed = result.q_packed.to(target).contiguous();
        result.row_q_bits = result.row_q_bits.to(target).contiguous();
        result.row_q_bit_offsets =
            result.row_q_bit_offsets.to(target).contiguous();
        result.sub_scale = result.sub_scale.to(target).contiguous();
        result.sub_min = result.sub_min.to(target).contiguous();
        result.neuron_scale = result.neuron_scale.to(target).contiguous();
        result.neuron_min = result.neuron_min.to(target).contiguous();
    }
    return result;
}

NintWeight to_device_nint8_zero(
        const mfq::nint8::Data & c, bool cuda, int device) {
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = 32;
    w.bits = 8;
    w.neuron_len = c.neuron_len;
    w.q8_zero = true;
    w.shape = c.shape;
    w.q_packed = cpu_u8_tensor(c.q, {c.out, c.ng, 32});
    w.q8_zero_scale = cpu_f16_tensor(c.scale_h, {c.out, c.ng});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, target_device);
        w.q_packed = w.q_packed.to(target).contiguous();
        w.q8_zero_scale = w.q8_zero_scale.to(target).contiguous();
    }
    return w;
}


mfq::nint::Data select_nint_cpu_rows(
        const mfq::nint::Data & source,
        const std::vector<int64_t> & rows) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    return repack_nint_cpu_rows(source, rows);
}

mfq::nint8::Data select_nint8_zero_cpu_rows(
        const mfq::nint8::Data & source,
        const std::vector<int64_t> & rows) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (rows.empty()) {
        throw std::runtime_error(
            "cannot create an empty NINT8-0 MoE shard");
    }
    mfq::nint8::Data result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    const size_t q_row_bytes =
        static_cast<size_t>(source.ng) * 32;
    result.q.resize(rows.size() * q_row_bytes);
    result.scale_h.resize(
        rows.size() * source.ng);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NINT8-0 MoE shard row is out of range");
        }
        std::memcpy(
            result.q.data() +
                destination * q_row_bytes,
            source.q.data() +
                static_cast<size_t>(source_row) *
                    q_row_bytes,
            q_row_bytes);
        std::memcpy(
            result.scale_h.data() +
                destination * source.ng,
            source.scale_h.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng) *
                sizeof(uint16_t));
    }
    return result;
}

