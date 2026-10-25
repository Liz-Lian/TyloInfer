#include "vq_blob.h"
#include "formats/generated/nvq_codebooks.generated.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mfq {
namespace {

template<class T>
T read(const std::vector<uint8_t>& blob, size_t& offset) {
    if (offset > blob.size() || sizeof(T) > blob.size() - offset)
        throw std::runtime_error("truncated VQ metadata");
    T result;
    std::memcpy(&result, blob.data() + offset, sizeof(result));
    offset += sizeof(result);
    return result;
}

size_t packed_bytes(size_t count, int bits) {
    if (bits == 0) return 0;
    if (bits < 0 || count > (std::numeric_limits<size_t>::max() - 7) / static_cast<size_t>(bits))
        throw std::runtime_error("VQ packed stream is too large");
    return (count * static_cast<size_t>(bits) + 7) / 8;
}

std::vector<int8_t> expand_codebook(
    int format, const uint16_t * packed, size_t count) {
    const int dims = format == 3 ? 4 : 8;
    const int expected = format == 1 ? 2048 : (format == 8 ? 1024 : 256);
    if ((int)count != expected) throw std::runtime_error("NVQ codebook entry count mismatch");
    const int digit_bits = format == 3 ? 3 : 2;
    std::vector<int8_t> codebook(count * (size_t)dims);
    for (size_t row = 0; row < count; ++row) {
        uint16_t word = packed[row];
        for (int i = 0; i < dims; ++i) {
            int digit = (word >> (digit_bits * i)) & ((1 << digit_bits) - 1);
            const bool ternary = format == 1 || format == 8;
            int value = ternary ? digit - 1 : 2 * digit + 1;
            if ((ternary && (value < -1 || value > 1)) ||
                (!ternary && (value < 1 || value > (format == 2 ? 7 : 15)))) {
                throw std::runtime_error("invalid NVQ codebook digit");
            }
            codebook[row * (size_t)dims + i] = (int8_t)value;
        }
    }
    return codebook;
}

std::vector<int8_t> expand_npq0(
    const std::vector<uint8_t> & packed) {
    constexpr size_t kMetadataBytes = 64;
    constexpr size_t kStates = 4;
    constexpr size_t kEntries = 8;
    constexpr size_t kSubvector = 4;
    constexpr size_t kPackedBytes = kMetadataBytes + 2 * kStates * kEntries * kSubvector;
    constexpr size_t kRuntimeBytes = kMetadataBytes + kStates * 64 * 8;
    if (packed.size() != kPackedBytes) {
        throw std::runtime_error("NPQ0-S packed table size mismatch");
    }
    std::vector<int8_t> runtime(kRuntimeBytes);
    std::memcpy(runtime.data(), packed.data(), kMetadataBytes);
    const int8_t * first = reinterpret_cast<const int8_t *>(packed.data() + kMetadataBytes);
    const int8_t * second = first + kStates * kEntries * kSubvector;
    for (size_t state = 0; state < kStates; ++state) {
        for (size_t second_index = 0; second_index < kEntries; ++second_index) {
            for (size_t first_index = 0; first_index < kEntries; ++first_index) {
                const size_t index = first_index | (second_index << 3);
                int8_t * destination = runtime.data() + kMetadataBytes
                    + (state * 64 + index) * 8;
                std::memcpy(
                    destination,
                    first + (state * kEntries + first_index) * kSubvector,
                    kSubvector);
                std::memcpy(
                    destination + kSubvector,
                    second + (state * kEntries + second_index) * kSubvector,
                    kSubvector);
            }
        }
    }
    return runtime;
}

std::vector<uint16_t> read_codebook(
    const std::vector<uint8_t> & blob, size_t & off, size_t count) {
    size_t nbytes = count * sizeof(uint16_t);
    if (off > blob.size() || nbytes > blob.size() - off) throw std::runtime_error("truncated NVQ custom codebook");
    std::vector<uint16_t> words(count);
    std::memcpy(words.data(), blob.data() + off, nbytes);
    off += nbytes;
    return words;
}

std::vector<uint8_t> take(
    const std::vector<uint8_t> & blob, size_t & off, size_t count, const char * label) {
    if (off > blob.size() || count > blob.size() - off) throw std::runtime_error(std::string("truncated NVQ ") + label);
    std::vector<uint8_t> result(count);
    std::memcpy(result.data(), blob.data() + off, count);
    off += count;
    return result;
}

void store_bits(
        std::vector<uint8_t> & destination,
        size_t bit,
        int bits,
        uint32_t value) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    const int bytes = (shift + bits + 7) >> 3;
    if (bits <= 0 || bits > 24 || byte + bytes > destination.size() ||
            value >= (1u << bits)) {
        throw std::runtime_error("invalid compact bitstream write");
    }
    const uint32_t shifted = value << shift;
    for (int index = 0; index < bytes; ++index) {
        destination[byte + index] |= static_cast<uint8_t>(
            shifted >> (index * 8));
    }
}

std::vector<uint16_t> unpack_bits(
        const std::vector<uint8_t> & blob,
        size_t & off,
        size_t count,
        int bits,
        const char * label) {
    const size_t nbytes = packed_bytes(count, bits);
    if (off > blob.size() || nbytes > blob.size() - off) {
        throw std::runtime_error(std::string("truncated ") + label);
    }
    std::vector<uint16_t> result(count, 0);
    for (size_t index = 0; index < count; ++index) {
        const size_t first_bit = index * static_cast<size_t>(bits);
        uint16_t value = 0;
        for (int bit = 0; bit < bits; ++bit) {
            const size_t source_bit = first_bit + static_cast<size_t>(bit);
            value |= static_cast<uint16_t>(
                ((blob[off + source_bit / 8] >> (source_bit & 7)) & 1u)
                << bit);
        }
        result[index] = value;
    }
    off += nbytes;
    return result;
}

void set_profile(nepq::Data & value, int profile) {
    if (profile == 0) {
        value.profile = 0;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
    } else if (profile == 1) {
        value.profile = 1;
        value.format = 7;
        value.state_bits = 3;
        value.index_bits = 7;
        value.aux_bits = 0;
        value.table_bytes = 832;
        value.runtime_table_bytes = 832;
    } else if (profile == 2) {
        value.profile = 2;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
    } else if (profile == 3) {
        value.profile = 3;
        value.format = 1;
        value.state_bits = 3;
        value.index_bits = 11;
        value.aux_bits = 1;
        value.table_bytes = 4096;
        value.runtime_table_bytes = 2048 * 8;
    } else if (profile == 4) {
        value.profile = 4;
        value.format = 9;
        value.state_bits = 2;
        value.index_bits = 6;
        value.aux_bits = 0;
        value.table_bytes = 320;
        value.runtime_table_bytes = 320;
        value.residual = true;
        value.residual_position_bits = 5;
        value.residual_record_bits = 15;
        value.residual_block_vectors = 24;
    } else if (profile == 5) {
        value.profile = 5;
        value.format = 8;
        value.state_bits = 4;
        value.index_bits = 9;
        value.aux_bits = 1;
        value.table_bytes = 2048;
        value.runtime_table_bytes = 1024 * 8;
        value.residual = true;
        value.residual_second = true;
        value.residual_position_bits = 4;
        value.residual_record_bits = 14;
        value.residual_block_vectors = 16;
    } else {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
}

std::vector<int8_t> expand_table(
        const uint8_t * source, const nepq::Data & value) {
    std::vector<uint8_t> packed(
        source, source + static_cast<ptrdiff_t>(value.table_bytes));
    if (value.profile == 0 || value.profile == 4) {
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-S table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    if (value.profile == 1) {
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (packed[(size_t)i] != expected[i]) {
                throw std::runtime_error("unsupported NEPQ0-L table profile");
            }
        }
        std::vector<int8_t> result(value.table_bytes);
        std::memcpy(result.data(), packed.data(), packed.size());
        return result;
    }
    std::vector<uint16_t> words((size_t)value.table_bytes / 2);
    std::memcpy(words.data(), packed.data(), packed.size());
    return expand_codebook(
        (value.profile == 2 || value.profile == 5) ? 8 : 1,
        words.data(), words.size());
}

} // namespace

namespace nvq {

Data unpack(const std::vector<uint8_t> & blob, const std::string & dtype) {
    if (blob.size() < 20) throw std::runtime_error("truncated NVQ header");
    Data t;
    const bool nvq_family = dtype == "NVQ";
    const bool npq_family = dtype == "NPQ";
    const bool nvq_magic =
        (std::memcmp(blob.data(), "NVQ1", 4) == 0) ||
        (std::memcmp(blob.data(), "NIQ1", 4) == 0);
    const bool nvq1_l_magic = std::memcmp(blob.data(), "NQ1L", 4) == 0;
    const bool nvq1_s_magic = std::memcmp(blob.data(), "NQ1S", 4) == 0;
    const bool npq0_l_magic = std::memcmp(blob.data(), "NPQL", 4) == 0;
    const bool npq0_s_magic = std::memcmp(blob.data(), "NPQS", 4) == 0;
    if ((nvq_family && !(nvq_magic || nvq1_l_magic || nvq1_s_magic)) ||
        (npq_family && !(npq0_l_magic || npq0_s_magic)) ||
        (!nvq_family && !npq_family)) {
        throw std::runtime_error("compact VQ family/payload mismatch: " + dtype);
    }
    size_t off = 4;
    const uint8_t profile = blob[off++];
    if (nvq1_l_magic) {
        t.format = 1;
    } else if (nvq1_s_magic) {
        t.format = 8;
    } else if (npq0_l_magic) {
        t.format = 7;
    } else if (npq0_s_magic) {
        t.format = 9;
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        const int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        if (jsc) {
            t.format = codebook_id == 1 ? 5 :
                (codebook_id == 2 ? 10 :
                 (codebook_id == 3 ? 12 :
                  (codebook_id == 4 ? 13 :
                   (codebook_id == 5 ? 14 :
                    (codebook_id == 6 ? 15 : 0)))));
        } else {
            t.format = codebook_id == 1 ? 2 :
                (codebook_id == 2 ? 3 : 0);
        }
        if (t.format == 0) {
            throw std::runtime_error("unsupported NVQ payload profile");
        }
    }
    t.sub_bits = blob[off++];
    t.gs = (int)read<uint16_t>(blob, off);
    t.axis = read<int32_t>(blob, off);
    t.neuron_len = read<int32_t>(blob, off);
    uint32_t ndim = read<uint32_t>(blob, off);
    if (ndim == 0 || ndim > 8) throw std::runtime_error("invalid NVQ ndim");
    t.shape.resize(ndim);
    for (uint32_t i = 0; i < ndim; ++i) t.shape[i] = read<int64_t>(blob, off);
    t.out = (int)read<uint32_t>(blob, off);
    if (t.gs != 24 || t.sub_bits < 1 || t.sub_bits > 8) {
        throw std::runtime_error("C++ NVQ runtime requires gs24 and sub_bits in [1,8]");
    }
    if (t.out <= 0 || t.neuron_len <= 0 || t.axis != 0 || t.shape.size() != 2 ||
        t.shape[0] != t.out || t.shape[1] != t.neuron_len) {
        throw std::runtime_error("C++ NVQ runtime requires row-major rank-2 axis=0 weights");
    }

    bool custom = false;
    bool group64_storage = false;
    if (t.format == 7) {
        if (profile != 1 || t.sub_bits != 3) {
            throw std::runtime_error("unsupported NPQ0-L profile");
        }
    } else if (t.format == 1) {
        if (profile != 1 && profile != 2) throw std::runtime_error("unsupported NVQ1-L profile");
        custom = profile == 2;
    } else if (t.format == 8) {
        if (profile != 1 || t.sub_bits != 4) {
            throw std::runtime_error("unsupported NVQ1-S profile");
        }
        custom = true;
    } else if (t.format == 9) {
        if (profile != 2 || t.sub_bits != 2) {
            throw std::runtime_error("unsupported NPQ0-S profile");
        }
    } else {
        constexpr uint8_t kIndexParity = 0x80;
        constexpr uint8_t kCustom = 0x40;
        constexpr uint8_t kJsc = 0x20;
        const bool jsc = (profile & kJsc) != 0;
        int codebook_id = profile & ~(kIndexParity | kCustom | kJsc);
        int expected_id =
            t.format == 13 ? 4 :
            (t.format == 14 ? 5 :
             (t.format == 15 ? 6 :
              ((t.format == 2 || t.format == 5)
               ? 1
               : (t.format == 12 ? 3 : 2))));
        if (codebook_id != expected_id) throw std::runtime_error("NVQ payload profile/codebook mismatch");
        t.sign_mode = (profile & kIndexParity) ? 1 : 0;
        if (t.sign_mode && t.format != 2) throw std::runtime_error("NVQ index parity requires NVQ2");
        custom = (profile & kCustom) != 0;
        if (t.format == 5 || t.format == 10 || t.format == 12 ||
            t.format == 13 || t.format == 14 || t.format == 15) {
            if (!jsc || custom || t.sign_mode || t.sub_bits != 4) {
                throw std::runtime_error("invalid NVQ-JSC profile flags");
            }
        } else if (jsc) {
            throw std::runtime_error("NVQ-JSC flag requires a JSC payload profile");
        }
    }

    const size_t codebook_count = t.format == 1 ? 2048 : (t.format == 8 ? 1024 : 256);
    if (t.format == 7) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 832;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-L product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {1, 8, 3, 4, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-L table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-L reserved table bytes");
        }
        for (int state = 0; state < 8; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-L scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 24; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-L reserved table bytes");
            }
        }
        auto metadata = take(blob, off, kTableBytes, "NPQ0-L product tables");
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (t.format == 9) {
        constexpr size_t kMetadataBytes = 64;
        constexpr size_t kTableBytes = 320;
        if (off + kTableBytes > blob.size()) {
            throw std::runtime_error("truncated NPQ0-S product tables");
        }
        const uint8_t * header = blob.data() + off;
        const uint8_t expected[6] = {2, 4, 3, 3, 24, 8};
        for (int i = 0; i < 6; ++i) {
            if (header[i] != expected[i]) {
                throw std::runtime_error("unsupported NPQ0-S table profile");
            }
        }
        if (header[6] != 0 || header[7] != 0) {
            throw std::runtime_error("invalid NPQ0-S reserved table bytes");
        }
        for (int state = 0; state < 4; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 8 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NPQ0-S scale LUT must be finite and non-negative");
            }
        }
        for (size_t i = 16; i < kMetadataBytes; ++i) {
            if (header[i] != 0) {
                throw std::runtime_error("invalid NPQ0-S reserved table bytes");
            }
        }
        auto metadata = take(blob, off, kTableBytes, "NPQ0-S product tables");
        t.codebook = expand_npq0(metadata);
    } else if (t.format == 5 || t.format == 10 || t.format == 12 ||
               t.format == 13 || t.format == 14 || t.format == 15) {
        constexpr size_t kHeaderBytes = 64;
        const int vector_size = nvq::vector_size(t.format);
        const size_t entries =
            t.format == 12 ? 512 :
            ((t.format == 13 || t.format == 15) ? 1024 :
             (t.format == 14 ? 4096 : 256));
        const size_t kBankBytes = entries * (size_t)vector_size;
        if (off + kHeaderBytes > blob.size()) {
            throw std::runtime_error("truncated NVQ2J metadata header");
        }
        const uint8_t * header = blob.data() + off;
        const int banks = header[1];
        const int metadata_version = header[0];
        if ((metadata_version != 1 && metadata_version != 2) ||
                (banks != 1 && banks != 2 && banks != 4) || header[2] != 16) {
            throw std::runtime_error("unsupported NVQ2J metadata dimensions");
        }
        if (metadata_version == 1) {
            if (header[52] != 0) {
                throw std::runtime_error("invalid NVQ-JSC v1 storage layout");
            }
        } else {
            if (t.format != 14 || header[52] != 1) {
                throw std::runtime_error(
                    "NVQ-JSC group64 storage requires NVQ2J-XL");
            }
            group64_storage = true;
        }
        if (header[3] != 0 && header[3] != 1) {
            throw std::runtime_error("invalid NVQ-JSC state mode");
        }
        for (int state = 0; state < 16; ++state) {
            uint16_t alpha_h;
            std::memcpy(&alpha_h, header + 4 + state * 2, sizeof(alpha_h));
            if ((alpha_h & 0x8000u) || (alpha_h & 0x7c00u) == 0x7c00u) {
                throw std::runtime_error("NVQ-JSC scale LUT must be finite and non-negative");
            }
            if (header[36 + state] >= banks) {
                throw std::runtime_error("NVQ-JSC state references a missing bank");
            }
            if (header[3] == 1) {
                const uint8_t expected_bank = (uint8_t)(state % banks);
                const int rank = state / banks;
                const float expected_scale = banks == 1
                    ? (float)state
                    : (vector_size == 4
                        ? (float)(rank + 1)
                        : 15.0f * (float)(rank + 1) / (float)(16 / banks));
                // Analytic scales are zero or exactly representable positive normal FP16 values.
                uint32_t scale_bits;
                std::memcpy(&scale_bits, &expected_scale, sizeof(scale_bits));
                const uint16_t expected_alpha_h = expected_scale == 0 ? 0 :
                    static_cast<uint16_t>((((scale_bits >> 23) - 112u) << 10) |
                                          ((scale_bits >> 13) & 0x3ffu));
                if (header[36 + state] != expected_bank || alpha_h != expected_alpha_h) {
                    throw std::runtime_error("invalid analytic NVQ-JSC state tables");
                }
            }
        }
        const size_t reserved_begin = metadata_version == 1 ? 52 : 53;
        for (size_t i = reserved_begin; i < kHeaderBytes; ++i) {
            if (header[i] != 0) throw std::runtime_error("invalid NVQ-JSC reserved metadata bytes");
        }
        const size_t metadata_bytes = kHeaderBytes + (size_t)banks * kBankBytes;
        auto metadata = take(blob, off, metadata_bytes, "JSC metadata");
        for (int bank = 0; bank < banks; ++bank) {
            const size_t base = kHeaderBytes + (size_t)bank * kBankBytes;
            for (size_t entry = 0; entry < entries; ++entry) {
                bool nonzero = false;
                for (int dim = 0; dim < vector_size; ++dim) {
                    const uint8_t value =
                        metadata[base + (size_t)entry * vector_size + dim];
                    if (value > 127) throw std::runtime_error("NVQ-JSC codebook value exceeds int8");
                    nonzero = nonzero || value != 0;
                }
                if (!nonzero) throw std::runtime_error("NVQ-JSC codeword must not be all zero");
            }
        }
        t.codebook.resize(metadata.size());
        std::memcpy(t.codebook.data(), metadata.data(), metadata.size());
    } else if (custom) {
        auto words = read_codebook(blob, off, codebook_count);
        t.codebook = expand_codebook(t.format, words.data(), words.size());
    } else if (t.format == 1) {
        t.codebook = expand_codebook(
            1, mfq::nvq_codebooks::kNvq1LCodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq1LCodebookPacked) / sizeof(uint16_t));
    } else if (t.format == 2) {
        t.codebook = expand_codebook(
            2, mfq::nvq_codebooks::kNvq2CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq2CodebookPacked) / sizeof(uint16_t));
    } else {
        t.codebook = expand_codebook(
            3, mfq::nvq_codebooks::kNvq3CodebookPacked,
            sizeof(mfq::nvq_codebooks::kNvq3CodebookPacked) / sizeof(uint16_t));
    }

    t.ng = 1 + (t.neuron_len - 1) / t.gs;
    const int vector_size = nvq::vector_size(t.format);
    t.nvec = 1 + (t.neuron_len - 1) / vector_size;
    t.nsign = 1 + (t.neuron_len - 1) / 8;
    size_t anchor_bytes = (size_t)t.out * 2;
    if (off + anchor_bytes > blob.size()) throw std::runtime_error("truncated NVQ neuron anchors");
    t.neuron_scale_h.resize(t.out);
    std::memcpy(t.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    for (uint16_t anchor : t.neuron_scale_h) {
        if ((anchor & 0x7c00u) == 0x7c00u || (anchor & 0x8000u) != 0u) {
            throw std::runtime_error(
                "NVQ neuron anchors must be finite and non-negative");
        }
    }
    off += anchor_bytes;
    size_t sub_bytes = ((size_t)t.out * t.ng * t.sub_bits + 7) / 8;
    const size_t index_bits = (size_t)nvq::index_bits(t.format);
    size_t index_bytes = ((size_t)t.out * t.nvec * index_bits + 7) / 8;
    const bool delta_format = t.format == 1 || t.format == 8;
    const bool no_aux_format = t.format == 7 || t.format == 9;
    size_t aux_count = delta_format ? (size_t)t.out * t.ng :
        (no_aux_format ? 0 : (size_t)t.out * t.nsign);
    size_t aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    size_t aux_bytes = (aux_count * aux_bits + 7) / 8;
    if (group64_storage) {
        const size_t record_count = static_cast<size_t>(t.out) * t.ng;
        const auto records = take(
            blob, off, record_count * 8, "group64 stream");
        t.sub_scale_packed.assign(sub_bytes, 0);
        t.indices_packed.assign(index_bytes, 0);
        t.aux_packed.assign(aux_bytes, 0);
        for (size_t linear = 0; linear < record_count; ++linear) {
            uint64_t packed = 0;
            std::memcpy(&packed, records.data() + linear * 8, sizeof(packed));
            const uint32_t state = static_cast<uint32_t>(packed >> 60);
            store_bits(
                t.sub_scale_packed, linear * t.sub_bits, t.sub_bits, state);
            const size_t row = linear / static_cast<size_t>(t.ng);
            const size_t group = linear - row * static_cast<size_t>(t.ng);
            for (int local = 0; local < 3; ++local) {
                const uint32_t segment = static_cast<uint32_t>(
                    (packed >> (local * 20)) & 0xfffffu);
                const uint32_t index = segment & 0xfffu;
                const uint32_t mask8 = segment >> 12;
                uint32_t parity = mask8 & 0x7fu;
                parity ^= parity >> 4;
                parity ^= parity >> 2;
                parity ^= parity >> 1;
                if ((mask8 >> 7) != (parity & 1u)) {
                    throw std::runtime_error(
                        "invalid NVQ-JSC group64 parity bit");
                }
                const size_t local_vector = group * 3 + local;
                if (local_vector >= static_cast<size_t>(t.nvec)) {
                    if (segment != 0) {
                        throw std::runtime_error(
                            "NVQ-JSC group64 padding must be zero");
                    }
                    continue;
                }
                const size_t vector = row * static_cast<size_t>(t.nvec) +
                    local_vector;
                store_bits(
                    t.indices_packed, vector * index_bits,
                    static_cast<int>(index_bits), index);
                store_bits(
                    t.aux_packed, vector * 7, 7, mask8 & 0x7fu);
            }
        }
    } else {
        t.sub_scale_packed = take(blob, off, sub_bytes, "sub-scale stream");
        t.indices_packed = take(blob, off, index_bytes, "index stream");
        t.aux_packed = take(blob, off, aux_bytes, "aux stream");
    }
    if (off != blob.size()) throw std::runtime_error("invalid NVQ blob tail");
    return t;
}

} // namespace nvq

namespace nepq {

Data unpack(
        const std::vector<uint8_t> & blob,
        const std::string & dtype,
        const std::vector<uint8_t> & runtime_payload) {
    if (blob.size() < 36 || std::memcmp(blob.data(), "NEP1", 4) != 0) {
        throw std::runtime_error("invalid NEPQ cohort header");
    }
    if (dtype != "NEPQ") {
        throw std::runtime_error("unsupported NEPQ cohort dtype: " + dtype);
    }
    Data value;
    size_t off = 4;
    const uint8_t version = blob[off++];
    const uint8_t profile = blob[off++];
    set_profile(value, profile);
    const uint8_t groups_per_supergroup = blob[off++];
    const uint8_t flags = blob[off++];
    value.n_experts = (int)read<uint32_t>(blob, off);
    value.out_per_expert = (int)read<uint32_t>(blob, off);
    value.neuron_len = (int)read<uint32_t>(blob, off);
    value.bank_count = (int)read<uint32_t>(blob, off);
    const auto rotation_block = read<uint32_t>(blob, off);
    if (rotation_block > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("invalid NEPQ Hadamard block");
    }
    value.rotation_block = static_cast<int>(rotation_block);
    value.rotation_seed = read<uint64_t>(blob, off);
    if (version != 1 || profile != value.profile ||
        groups_per_supergroup != 4 || (flags & ~1u) != 0 ||
        ((flags & 1u) != 0) != (value.rotation_block != 0)) {
        throw std::runtime_error("unsupported NEPQ cohort profile");
    }
    if (value.n_experts <= 0 || value.out_per_expert <= 0 ||
        value.neuron_len <= 0 || value.neuron_len % 8 != 0 ||
        value.bank_count <= 0 || value.bank_count > 256 ||
        value.n_experts >
            std::numeric_limits<int>::max() / value.out_per_expert) {
        throw std::runtime_error("invalid NEPQ cohort dimensions");
    }
    if (value.rotation_block != 0 &&
        ((value.rotation_block & (value.rotation_block - 1)) != 0 ||
         value.neuron_len % value.rotation_block != 0)) {
        throw std::runtime_error("invalid NEPQ Hadamard block");
    }
    if (value.residual && value.rotation_block == 0) {
        throw std::runtime_error("NEPQ-A requires a Hadamard rotation");
    }
    value.ng = 1 + (value.neuron_len - 1) / 24;
    value.nvec = value.neuron_len / 8;
    value.nsuper = (value.ng + 3) / 4;
    const int rows = value.n_experts * value.out_per_expert;

    const size_t all_table_bytes =
        (size_t)value.bank_count * value.table_bytes;
    if (off + all_table_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ table pool");
    }
    value.table_pool.reserve(
        (size_t)value.bank_count * value.runtime_table_bytes);
    value.grouped_table_pool.reserve(
        (size_t)value.bank_count *
        ((value.profile == 0 || value.profile == 4)
            ? 2112 : value.runtime_table_bytes));
    for (int bank = 0; bank < value.bank_count; ++bank) {
        const uint8_t * table = blob.data() + off + (size_t)bank * value.table_bytes;
        auto runtime = expand_table(table, value);
        value.table_pool.insert(
            value.table_pool.end(), runtime.begin(), runtime.end());
        if (value.profile == 0 || value.profile == 4) {
            std::vector<uint8_t> compact(
                table, table + static_cast<ptrdiff_t>(value.table_bytes));
            auto grouped = expand_npq0(compact);
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), grouped.begin(), grouped.end());
        } else {
            value.grouped_table_pool.insert(
                value.grouped_table_pool.end(), runtime.begin(), runtime.end());
        }
    }
    off += all_table_bytes;
    const size_t anchor_bytes = (size_t)rows * 2;
    if (off + anchor_bytes > blob.size()) {
        throw std::runtime_error("truncated NEPQ neuron anchors");
    }
    value.neuron_scale_h.resize((size_t)rows);
    std::memcpy(value.neuron_scale_h.data(), blob.data() + off, anchor_bytes);
    for (uint16_t raw : value.neuron_scale_h) {
        if ((raw & 0x7c00u) == 0x7c00u ||
                ((raw & 0x8000u) != 0 && (raw & 0x7fffu) != 0)) {
            throw std::runtime_error(
                "NEPQ neuron anchors must be finite and non-negative");
        }
    }
    off += anchor_bytes;
    value.state_packed = take(
        blob, off, packed_bytes((size_t)rows * value.ng, value.state_bits),
        "NEPQ state stream");
    value.indices_packed = take(
        blob, off, packed_bytes((size_t)rows * value.nvec, value.index_bits),
        "NEPQ index stream");
    value.aux_packed = take(
        blob, off, packed_bytes((size_t)rows * value.ng, value.aux_bits),
        "NEPQ aux stream");
    value.bank_ids = take(
        blob, off, (size_t)rows * value.nsuper, "NEPQ bank selectors");
    if (value.residual) {
        const size_t header = off;
        if (header + 64 > blob.size() ||
            std::memcmp(blob.data() + header, "NRA1", 4) != 0) {
            throw std::runtime_error("invalid NEPQ-A residual header");
        }
        off += 4;
        const uint8_t residual_version = blob[off++];
        const uint8_t record_bits = blob[off++];
        const uint8_t position_bits = blob[off++];
        const uint8_t residual_flags = blob[off++];
        const uint32_t dictionary_entries = read<uint32_t>(blob, off);
        const uint32_t block_count = read<uint32_t>(blob, off);
        const uint32_t second_count = read<uint32_t>(blob, off);
        const uint32_t padding_nbytes = read<uint32_t>(blob, off);
        const uint64_t reserved = read<uint64_t>(blob, off);
        value.residual_blocks_per_row =
            (value.nvec + value.residual_block_vectors - 1) /
            value.residual_block_vectors;
        const size_t expected_blocks =
            static_cast<size_t>(rows) * value.residual_blocks_per_row;
        const uint8_t expected_flags = value.residual_second ? 1 : 0;
        if (residual_version != 1 ||
            record_bits != value.residual_record_bits ||
            position_bits != value.residual_position_bits ||
            residual_flags != expected_flags ||
            dictionary_entries != 1024 ||
            block_count != expected_blocks ||
            reserved != 0 ||
            (!value.residual_second && second_count != 0)) {
            throw std::runtime_error("unsupported NEPQ-A residual profile");
        }
        off = header + 64;
        const size_t dictionary_values = 1024 * 8;
        const size_t dictionary_nbytes = dictionary_values * sizeof(uint16_t);
        if (off + dictionary_nbytes > blob.size()) {
            throw std::runtime_error("truncated NEPQ-A residual dictionary");
        }
        value.residual_codebook_h.resize(dictionary_values);
        std::memcpy(
            value.residual_codebook_h.data(), blob.data() + off,
            dictionary_nbytes);
        for (size_t row = 0; row < 1024; ++row) {
            for (size_t column = 0; column < 8; ++column) {
                const uint16_t raw = value.residual_codebook_h[row * 8 + column];
                if ((raw & 0x7c00u) == 0x7c00u) {
                    throw std::runtime_error(
                        "NEPQ-A residual dictionary must be finite");
                }
            }
        }
        off += dictionary_nbytes;
        auto first = unpack_bits(
            blob, off, expected_blocks, value.residual_record_bits,
            "NEPQ-A first residual stream");
        value.residual_first.assign(first.begin(), first.end());
        value.residual_second_dense.assign(expected_blocks, -1);
        if (value.residual_second) {
            auto mask = unpack_bits(
                blob, off, expected_blocks, 1,
                "NEPQ-A second residual bitmap");
            auto second = unpack_bits(
                blob, off, second_count, value.residual_record_bits,
                "NEPQ-A second residual stream");
            size_t compact = 0;
            for (size_t block = 0; block < expected_blocks; ++block) {
                if (mask[block] != 0) {
                    if (compact >= second.size()) {
                        throw std::runtime_error(
                            "NEPQ-A second residual count mismatch");
                    }
                    value.residual_second_dense[block] = second[compact++];
                }
            }
            if (compact != second.size()) {
                throw std::runtime_error(
                    "NEPQ-A second residual count mismatch");
            }
        }
        auto validate_record = [&](uint16_t record, size_t block) {
            const uint32_t position_mask =
                (1u << value.residual_position_bits) - 1u;
            const uint32_t position = record & position_mask;
            const uint32_t dictionary_id =
                record >> value.residual_position_bits;
            const size_t block_in_row =
                block % value.residual_blocks_per_row;
            const int available = std::min(
                value.residual_block_vectors,
                value.nvec - static_cast<int>(block_in_row) *
                    value.residual_block_vectors);
            if (dictionary_id >= 1024 || position >=
                    static_cast<uint32_t>(available)) {
                throw std::runtime_error(
                    "NEPQ-A residual record is out of range");
            }
        };
        for (size_t block = 0; block < expected_blocks; ++block) {
            validate_record(value.residual_first[block], block);
            if (value.residual_second_dense[block] >= 0) {
                validate_record(
                    static_cast<uint16_t>(
                        value.residual_second_dense[block]),
                    block);
            }
        }
        if (padding_nbytes > blob.size() - off) {
            throw std::runtime_error("truncated NEPQ-A residual padding");
        }
        const size_t padding_end = off + padding_nbytes;
        for (; off < padding_end; ++off) {
            if (blob[off] != 0) {
                throw std::runtime_error(
                    "NEPQ-A residual padding must be zero");
            }
        }
        const uint32_t position_mask =
            (1u << value.residual_position_bits) - 1u;
        for (size_t block = 0; block < expected_blocks; ++block) {
            const int block_in_row =
                static_cast<int>(block % value.residual_blocks_per_row);
            const int available = std::min(
                value.residual_block_vectors,
                value.nvec - block_in_row * value.residual_block_vectors);
            for (int record : {
                     value.residual_first[block],
                     value.residual_second_dense[block]}) {
                if (record < 0) continue;
                const int position = record & position_mask;
                const int dictionary_id =
                    record >> value.residual_position_bits;
                if (position >= available || dictionary_id >= 1024) {
                    throw std::runtime_error(
                        "invalid NEPQ-A residual record");
                }
            }
        }
    }
    if (off != blob.size()) throw std::runtime_error("invalid NEPQ cohort tail");
    for (uint8_t bank : value.bank_ids) {
        if ((int)bank >= value.bank_count) {
            throw std::runtime_error("NEPQ selector references a missing bank");
        }
    }

    if (value.rotation_block == 0) {
        if (!runtime_payload.empty()) {
            throw std::runtime_error("unexpected NEPQ rotation metadata");
        }
    } else {
        if (runtime_payload.size() != 20 + (size_t)value.neuron_len ||
            std::memcmp(runtime_payload.data(), "HSG1", 4) != 0) {
            throw std::runtime_error("missing NEPQ rotation sign vector");
        }
        size_t runtime_off = 4;
        uint32_t width = read<uint32_t>(runtime_payload, runtime_off);
        uint32_t block = read<uint32_t>(runtime_payload, runtime_off);
        uint64_t seed = read<uint64_t>(runtime_payload, runtime_off);
        if (width != static_cast<uint32_t>(value.neuron_len) ||
            block != static_cast<uint32_t>(value.rotation_block) || seed != value.rotation_seed) {
            throw std::runtime_error("NEPQ rotation metadata mismatch");
        }
        value.rotation_signs.resize((size_t)value.neuron_len);
        std::memcpy(
            value.rotation_signs.data(), runtime_payload.data() + runtime_off,
            (size_t)value.neuron_len);
        for (int8_t sign : value.rotation_signs) {
            if (sign != -1 && sign != 1) {
                throw std::runtime_error("invalid NEPQ rotation sign");
            }
        }
    }
    return value;
}

} // namespace nepq
} // namespace mfq
