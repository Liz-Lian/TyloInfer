#include "nint_blob.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq {
namespace {

template<class T>
T read(const std::vector<uint8_t>& blob, size_t& offset) {
    if (offset > blob.size() || sizeof(T) > blob.size() - offset)
        throw std::runtime_error("truncated NINT metadata");
    T result;
    std::memcpy(&result, blob.data() + offset, sizeof(result));
    offset += sizeof(result);
    return result;
}

std::vector<uint8_t> unpack_bits(const std::vector<uint8_t> & blob, size_t & off, size_t count, int bits) {
    if (bits < 1 || bits > 8 ||
            count > (std::numeric_limits<size_t>::max() - 7) /
                static_cast<size_t>(bits)) {
        throw std::runtime_error("invalid packed-bit stream dimensions");
    }
    const size_t required =
        (count * static_cast<size_t>(bits) + 7) / 8;
    if (off > blob.size() || required > blob.size() - off) {
        throw std::runtime_error("truncated packed-bit stream");
    }
    std::vector<uint8_t> out(count);
    if (bits == 8) {
        std::copy(blob.begin() + (ptrdiff_t)off, blob.begin() + (ptrdiff_t)(off + count), out.begin());
        off += count;
        return out;
    }
    if (bits == 4) {
        size_t nbytes = (count + 1) / 2;
        for (size_t i = 0; i < count; ++i) {
            uint8_t p = blob[off + i / 2];
            out[i] = (i & 1) ? (p >> 4) : (p & 0x0f);
        }
        off += nbytes;
        return out;
    }
    size_t nbytes = (count * (size_t)bits + 7) / 8;
    const uint32_t mask=(1u<<bits)-1;
    for (size_t i = 0; i < count; ++i) {
        const size_t bit=i*static_cast<size_t>(bits),byte=bit/8;
        uint32_t word=blob[off+byte];
        if(byte+1<nbytes)word|=uint32_t(blob[off+byte+1])<<8;
        out[i]=static_cast<uint8_t>((word>>(bit&7))&mask);
    }
    off += nbytes;
    return out;
}

} // namespace

namespace nint {

void describe(Data & t) {
    if (t.out <= 0 || t.neuron_len <= 0 || t.ng <= 0 || t.gs <= 0 ||
            t.row_q_bits.size() != static_cast<size_t>(t.out) ||
            t.row_sub_bits.size() != static_cast<size_t>(t.out)) {
        throw std::runtime_error("cannot describe incomplete NINT row metadata");
    }
    constexpr int q_selector_bits = 3;
    constexpr int k_selector_bits = 2;
    double encoded_bits = static_cast<double>(t.out) *
        (32.0 + q_selector_bits + k_selector_bits);
    std::array<size_t, 64> joint_counts{};
    const double padded_values_per_row =
        static_cast<double>(t.ng) * t.gs;
    for (int row = 0; row < t.out; ++row) {
        const int q_bits = t.row_q_bits[static_cast<size_t>(row)];
        const int k_bits = t.row_sub_bits[static_cast<size_t>(row)];
        if (q_bits < 1 || q_bits > 8 || k_bits < 1 || k_bits > 8) {
            throw std::runtime_error("invalid NINT descriptor row metadata");
        }
        encoded_bits += padded_values_per_row * q_bits +
            2.0 * t.ng * k_bits;
        ++joint_counts[static_cast<size_t>((q_bits - 1) * 8 + (k_bits - 1))];
    }
    t.aggregate_bpw = encoded_bits /
        (static_cast<double>(t.out) * t.neuron_len);
    t.distribution_entropy = 0.0;
    for (const size_t count : joint_counts) {
        if (count == 0) continue;
        const double probability = static_cast<double>(count) / t.out;
        t.distribution_entropy -= probability * std::log2(probability);
    }
}

Data unpack(const std::vector<uint8_t> & blob) {
    constexpr size_t fixed_header_nbytes = 2 + 3 * sizeof(int32_t) + sizeof(uint32_t);
    if (blob.size() < fixed_header_nbytes) {
        throw std::runtime_error("truncated NINT header");
    }
    Data t;
    size_t off = 0;
    const int raw_bits = blob[off++];
    const bool is_nint_v2 = (raw_bits & 0x80) != 0;
    t.bits = raw_bits & 0x7f;
    t.sub_bits = blob[off++];
    t.gs = read<int32_t>(blob, off);
    t.axis = read<int32_t>(blob, off);
    t.neuron_len = read<int32_t>(blob, off);
    uint32_t ndim = read<uint32_t>(blob, off);
    if (t.bits < 1 || t.bits > 8 || t.sub_bits < 1 || t.sub_bits > 8 ||
            t.gs <= 0 || t.axis < 0 || t.neuron_len <= 0 ||
            ndim == 0 || t.axis >= static_cast<int>(ndim) ||
            ndim > (blob.size() - off) / sizeof(int64_t)) {
        throw std::runtime_error("invalid NINT header");
    }
    const size_t shape_and_counts =
        static_cast<size_t>(ndim) * sizeof(int64_t) + 2 * sizeof(uint32_t);
    if (shape_and_counts > blob.size() - off) {
        throw std::runtime_error("truncated NINT dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read<int64_t>(blob, off);
    t.out = (int)read<uint32_t>(blob, off);
    t.ng = (int)read<uint32_t>(blob, off);
    int64_t flattened_neuron_len = 1;
    for (uint32_t index = 0; index < ndim; ++index) {
        if (t.shape[index] <= 0 ||
                (static_cast<int>(index) != t.axis &&
                 flattened_neuron_len >
                    std::numeric_limits<int64_t>::max() / t.shape[index])) {
            throw std::runtime_error("invalid NINT tensor shape");
        }
        if (static_cast<int>(index) != t.axis) {
            flattened_neuron_len *= t.shape[index];
        }
    }
    const int64_t expected_groups =
        (static_cast<int64_t>(t.neuron_len) + t.gs - 1) / t.gs;
    const bool routed_shape = ndim == 3 && t.axis == 0 &&
        t.shape[0] <= std::numeric_limits<int>::max() &&
        t.shape[1] <= std::numeric_limits<int>::max() &&
        t.shape[0] * t.shape[1] == t.out && t.shape[2] == t.neuron_len;
    if (t.out <= 0 || t.ng <= 0 ||
            (!routed_shape && (t.shape[static_cast<size_t>(t.axis)] != t.out ||
             flattened_neuron_len != t.neuron_len)) ||
            expected_groups != t.ng) {
        throw std::runtime_error("inconsistent NINT tensor dimensions");
    }
    const size_t anchor_nbytes = static_cast<size_t>(t.out) * 2;
    if (anchor_nbytes > (blob.size() - off) / 2) {
        throw std::runtime_error("truncated NINT neuron metadata");
    }
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    t.neuron_min_h.resize(t.out);
    std::memcpy(t.neuron_min_h.data(), blob.data() + off, (size_t)t.out * 2);
    off += (size_t)t.out * 2;
    for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
        if ((t.neuron_scale_h[row] & 0x7c00u) == 0x7c00u ||
                (t.neuron_min_h[row] & 0x7c00u) == 0x7c00u) {
            throw std::runtime_error("NINT neuron metadata must be finite");
        }
    }
    if (static_cast<size_t>(t.out) >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(t.ng)) {
        throw std::runtime_error("NINT metadata size overflow");
    }
    size_t sub_count = static_cast<size_t>(t.out) * static_cast<size_t>(t.ng);
    if (sub_count >
            std::numeric_limits<size_t>::max() / static_cast<size_t>(t.gs)) {
        throw std::runtime_error("NINT value size overflow");
    }
    size_t q_count = sub_count * static_cast<size_t>(t.gs);
    t.row_sub_bits.assign(
        static_cast<size_t>(t.out),
        static_cast<uint8_t>(t.sub_bits));
    if (is_nint_v2) {
        constexpr int selector_bits = 2;
        const auto selectors = unpack_bits(
            blob, off, static_cast<size_t>(t.out), selector_bits);
        for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
            const int row_bits = t.sub_bits - 1 + selectors[row];
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT v2 subgroup width");
            }
            t.row_sub_bits[row] = static_cast<uint8_t>(row_bits);
        }
        t.sub_scale.resize(sub_count);
        t.sub_min.resize(sub_count);
        for (int selector = 0; selector < (1 << selector_bits); ++selector) {
            const int row_bits = t.sub_bits - 1 + selector;
            const size_t selected_rows = static_cast<size_t>(std::count(
                selectors.begin(), selectors.end(), static_cast<uint8_t>(selector)));
            if (selected_rows == 0) continue;
            if (row_bits < 1 || row_bits > 8) {
                throw std::runtime_error("invalid NINT v2 subgroup width");
            }
            const size_t selected_values = selected_rows * static_cast<size_t>(t.ng);
            const auto scales = unpack_bits(blob, off, selected_values, row_bits);
            const auto minima = unpack_bits(blob, off, selected_values, row_bits);
            size_t local_row = 0;
            for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
                if (selectors[row] != static_cast<uint8_t>(selector)) continue;
                const size_t source = local_row * static_cast<size_t>(t.ng);
                const size_t destination = row * static_cast<size_t>(t.ng);
                std::copy_n(
                    scales.begin() + static_cast<ptrdiff_t>(source),
                    t.ng,
                    t.sub_scale.begin() + static_cast<ptrdiff_t>(destination));
                std::copy_n(
                    minima.begin() + static_cast<ptrdiff_t>(source),
                    t.ng,
                    t.sub_min.begin() + static_cast<ptrdiff_t>(destination));
                ++local_row;
            }
        }
    } else {
        t.sub_scale = unpack_bits(blob, off, sub_count, t.sub_bits);
        t.sub_min = unpack_bits(blob, off, sub_count, t.sub_bits);
    }
    if (is_nint_v2) {
        constexpr int selector_bits = 3;
        const auto selectors = unpack_bits(
            blob, off, static_cast<size_t>(t.out), selector_bits);
        t.row_q_bits.resize(static_cast<size_t>(t.out));
        t.row_q_bit_offsets.resize(static_cast<size_t>(t.out));
        const size_t values_per_row =
            static_cast<size_t>(t.ng) * static_cast<size_t>(t.gs);
        for (int selector = 0; selector < (1 << selector_bits); ++selector) {
            const int row_bits = selector + 1;
            const size_t selected_rows = static_cast<size_t>(std::count(
                selectors.begin(), selectors.end(), static_cast<uint8_t>(selector)));
            if (selected_rows == 0) continue;
            if (values_per_row > std::numeric_limits<size_t>::max() / selected_rows) {
                throw std::runtime_error("NINT mixed-q value count overflow");
            }
            const size_t selected_values = selected_rows * values_per_row;
            if (selected_values >
                    (std::numeric_limits<size_t>::max() - 7) /
                        static_cast<size_t>(row_bits)) {
                throw std::runtime_error("NINT mixed-q packed size overflow");
            }
            const size_t stream_nbytes =
                (selected_values * static_cast<size_t>(row_bits) + 7) / 8;
            if (off > blob.size() || stream_nbytes > blob.size() - off) {
                throw std::runtime_error("truncated NINT mixed-q value stream");
            }
            const uint64_t cohort_start_bits =
                static_cast<uint64_t>(t.q_packed.size()) * 8u;
            t.q_packed.insert(
                t.q_packed.end(),
                blob.begin() + static_cast<ptrdiff_t>(off),
                blob.begin() + static_cast<ptrdiff_t>(off + stream_nbytes));
            off += stream_nbytes;
            size_t local_row = 0;
            const uint64_t packed_row_bits =
                static_cast<uint64_t>(values_per_row) *
                static_cast<uint64_t>(row_bits);
            for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
                if (selectors[row] != static_cast<uint8_t>(selector)) continue;
                const uint64_t bit_offset = cohort_start_bits +
                    static_cast<uint64_t>(local_row) * packed_row_bits;
                if (bit_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw std::runtime_error("NINT mixed-q row offset overflow");
                }
                t.row_q_bits[row] = static_cast<uint8_t>(row_bits);
                t.row_q_bit_offsets[row] = static_cast<int64_t>(bit_offset);
                ++local_row;
            }
        }
        t.q_packed.insert(t.q_packed.end(), 8, 0);
    } else {
        if (q_count >
                (std::numeric_limits<size_t>::max() - 7) /
                    static_cast<size_t>(t.bits)) {
            throw std::runtime_error("NINT packed value size overflow");
        }
        const size_t compact_q_nbytes =
            (q_count * static_cast<size_t>(t.bits) + 7) / 8;
        if (off > blob.size() || compact_q_nbytes > blob.size() - off) {
            throw std::runtime_error("truncated NINT packed values");
        }
        t.q_packed.assign(
            blob.begin() + static_cast<ptrdiff_t>(off),
            blob.begin() + static_cast<ptrdiff_t>(off + compact_q_nbytes));
        off += compact_q_nbytes;
        t.row_q_bits.assign(static_cast<size_t>(t.out),
                            static_cast<uint8_t>(t.bits));
        t.row_q_bit_offsets.resize(static_cast<size_t>(t.out));
        const uint64_t row_bits =
            static_cast<uint64_t>(t.ng) * static_cast<uint64_t>(t.gs) *
            static_cast<uint64_t>(t.bits);
        for (size_t row = 0; row < static_cast<size_t>(t.out); ++row) {
            const uint64_t bit_offset = static_cast<uint64_t>(row) * row_bits;
            if (bit_offset >
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                throw std::runtime_error("NINT row offset overflow");
            }
            t.row_q_bit_offsets[row] = static_cast<int64_t>(bit_offset);
        }
        t.q_packed.insert(t.q_packed.end(), 8, 0);
    }
    if (off != blob.size()) {
        throw std::runtime_error("invalid NINT trailing bytes");
    }
    describe(t);
    return t;
}

} // namespace nint

namespace nint8 {

Data unpack(const std::vector<uint8_t> & blob) {
    constexpr size_t block_bytes = 34;
    if (blob.size() < 24 || std::memcmp(blob.data(), "NI80", 4) != 0) {
        throw std::runtime_error("invalid NINT8-0 header");
    }
    Data t;
    size_t off = 4;
    t.axis = read<int32_t>(blob, off);
    t.neuron_len = read<int32_t>(blob, off);
    uint32_t ndim = read<uint32_t>(blob, off);
    if (ndim == 0 || t.axis < 0 || t.axis >= static_cast<int>(ndim) ||
        t.neuron_len <= 0 || t.neuron_len % 32 != 0 ||
        ndim > (blob.size() - off - 2 * sizeof(uint32_t)) / sizeof(int64_t)) {
        throw std::runtime_error("invalid NINT8-0 dimensions");
    }
    t.shape.resize(ndim);
    for (uint32_t index = 0; index < ndim; ++index) {
        t.shape[index] = read<int64_t>(blob, off);
    }
    t.out = static_cast<int>(read<uint32_t>(blob, off));
    t.ng = static_cast<int>(read<uint32_t>(blob, off));
    if (t.out <= 0 || t.ng != t.neuron_len / 32 ||
        t.shape[static_cast<size_t>(t.axis)] != t.out) {
        throw std::runtime_error("NINT8-0 shape/header mismatch");
    }
    int64_t elements = 1;
    for (int64_t value : t.shape) {
        if (value <= 0 || elements > INT64_MAX / value) {
            throw std::runtime_error("invalid NINT8-0 logical shape");
        }
        elements *= value;
    }
    if (elements != static_cast<int64_t>(t.out) * t.neuron_len) {
        throw std::runtime_error("NINT8-0 logical element count mismatch");
    }
    const size_t blocks = static_cast<size_t>(t.out) * t.ng;
    if (off > blob.size() ||
            blocks > (blob.size() - off) / block_bytes ||
            blob.size() - off != blocks * block_bytes) {
        throw std::runtime_error("invalid NINT8-0 block payload length");
    }
    t.q.resize(blocks * 32);
    t.scale_h.resize(blocks);
    for (size_t block = 0; block < blocks; ++block) {
        const uint8_t * source = blob.data() + off + block * block_bytes;
        std::memcpy(&t.scale_h[block], source, sizeof(uint16_t));
        const uint16_t scale = t.scale_h[block];
        if ((scale & 0x7c00u) == 0x7c00u || (scale & 0x8000u) != 0u) {
            throw std::runtime_error(
                "NINT8-0 scales must be finite and non-negative");
        }
        std::memcpy(t.q.data() + block * 32, source + 2, 32);
    }
    return t;
}

} // namespace nint8
} // namespace mfq
