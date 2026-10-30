#include "engine/weights/quant.h"
#include "backends/cuda/engine/executor/options.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "ops/common/include/format.h"
#include "parallel.h"
#include "nvq_group.h"
#include "nvq_dense_pack.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace mfq::cuda::quant_format;

bool prepare_nvq_dense_groups(NvqWeight& w) {
    if(w.dense_groups)return true;
    if(!w.indices_packed.is_cpu() || w.gs!=24 || w.sub_bits!=4 ||
            w.out<=0 || w.out>std::numeric_limits<int>::max() ||
            w.neuron_len<=0 || w.neuron_len>std::numeric_limits<int>::max() ||
            !mfq::cpu::nvq_dense_rows_dot_kernel(int(w.kernel_format)))return false;
    const int rows=int(w.out),pairs=int((w.neuron_len+7)/8),groups=int(w.ng);
    const bool d4=w.kernel_format==10 || w.kernel_format==11 || w.kernel_format==12 || w.kernel_format==15;
    if(d4 && (w.neuron_len+3)/4!=2*pairs)return false;
    // Every expert in an arena starts at a byte boundary. No per-expert
    // padding is inserted; unsupported odd shapes keep canonical storage.
    const int bits=w.kernel_format==12?9:(w.kernel_format==13 || w.kernel_format==15)?10:w.kernel_format==14?12:8;
    if((int64_t(rows)*(int64_t(pairs)*((d4?2:1)*bits+7)+int64_t(groups)*4))&7)return false;
    std::vector<uint8_t> compact;
#define PACK(B,P) mfq::cpu::dense::pack<B,P>(w.indices_packed.data_ptr<uint8_t>(),w.indices_packed.numel(), \
        w.aux_packed.data_ptr<uint8_t>(),w.aux_packed.numel(),w.sub_scale_packed.data_ptr<uint8_t>(), \
        w.sub_scale_packed.numel(),rows,pairs,groups)
    switch(w.kernel_format) {
        case 5:compact=PACK(8,1);break;case 13:compact=PACK(10,1);break;case 14:compact=PACK(12,1);break;
        case 10:case 11:compact=PACK(8,2);break;case 12:compact=PACK(9,2);break;case 15:compact=PACK(10,2);break;
        default:return false;
    }
#undef PACK
    if(compact.size()>size_t(w.indices_packed.numel()+w.aux_packed.numel()+w.sub_scale_packed.numel()))
        throw std::runtime_error("compact NVQ unexpectedly expanded weights");
    w.indices_packed=mfq_tensor_backend::tensor(compact);
    w.aux_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.dense_groups=true;
    return true;
}

void prepare_nvq_integer_codebook(NvqWeight& w) {
    const auto enabled=mfq::cuda::runtime_options::nvq1_integer_delta();
    if(!enabled || !*enabled || w.integer_codebook.defined() ||
       (w.kernel_format!=1 && w.kernel_format!=8) || !w.codebook.is_cuda())return;
    const auto canonical=w.codebook.to(mfq_tensor_backend::kCPU).contiguous();
    const bool small=w.kernel_format==8;
    const int bank_bytes=small?512*8:2048*8;
    if(canonical.numel()!=int64_t(small?2:1)*bank_bytes)
        throw std::invalid_argument("NVQ1 integer decode book has invalid canonical size");
    std::vector<int8_t> digits(2*bank_bytes);
    const auto* source=canonical.data_ptr<int8_t>();
    for(int negative=0;negative<2;++negative)for(int i=0;i<bank_bytes;++i) {
        const int v=source[(small?negative*bank_bytes:0)+i];
        if(v<-1 || v>1)throw std::invalid_argument("NVQ1 integer decode digits require -1/0/1");
        digits[negative*bank_bytes+i]=int8_t((small?32:8)*v+(negative?-1:1)*(small?5:1));
    }
    w.integer_codebook=mfq_tensor_backend::tensor(digits).to(w.codebook.device()).contiguous();
}
void prepare_nvq_decode_records(NvqWeight& w) {
    const auto enabled=mfq::cuda::runtime_options::nvq1_group_records();
    if(!enabled || !*enabled || w.decode_records.defined() ||
       (w.kernel_format!=1 && w.kernel_format!=8) || !w.indices_packed.is_cuda() || w.gs!=24 ||
       w.sub_bits!=(w.kernel_format==8?4:3))return;
    if(w.aux_packed.numel()==0){w.decode_records=w.indices_packed;return;}
    const auto indices=w.indices_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto auxiliary=w.aux_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto states=w.sub_scale_packed.to(mfq_tensor_backend::kCPU).contiguous();
    const auto bits=[](const mfq_tensor_backend::Tensor& data,int64_t bit,int width) {
        uint32_t value=0;
        for(int i=0;i<width;++i)if((bit+i)/8<data.numel())
            value|=uint32_t((data.data_ptr<uint8_t>()[(bit+i)/8]>>((bit+i)&7))&1u)<<i;
        return value;
    };
    const bool small=w.kernel_format==8;
    const int index_bits=small?9:11,record_bytes=small?4:5,nvec=int((w.neuron_len+7)/8);
    std::vector<uint8_t> records(size_t(w.out)*w.ng*record_bytes);
    for(int64_t row=0;row<w.out;++row)for(int group=0;group<w.ng;++group) {
        const int64_t linear=row*w.ng+group;
        uint64_t record=0;
        for(int segment=0;segment<3;++segment)if(group*3+segment<nvec)
            record|=uint64_t(bits(indices,(row*nvec+group*3+segment)*index_bits,index_bits))<<(segment*index_bits);
        record|=uint64_t(bits(states,linear*w.sub_bits,int(w.sub_bits)))<<(3*index_bits);
        record|=uint64_t(bits(auxiliary,linear,1))<<(3*index_bits+w.sub_bits);
        std::memcpy(records.data()+linear*record_bytes,&record,record_bytes);
    }
    w.decode_records=mfq_tensor_backend::tensor(records).to(w.indices_packed.device()).contiguous();
    w.indices_packed=w.decode_records;
    w.aux_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
    w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
}

static bool nvq_delta_format(int format) {
    return format == 1 || format == 8;
}

static bool nvq_no_aux_format(int format) {
    return format == 7 || format == 9;
}

static void copy_packed_bits(
        const std::vector<uint8_t> & source,
        size_t source_bit,
        std::vector<uint8_t> & destination,
        size_t destination_bit,
        size_t bit_count) {
    if (bit_count == 0) return;
    if (source_bit + bit_count >
            source.size() * 8 ||
        destination_bit + bit_count >
            destination.size() * 8) {
        throw std::runtime_error(
            "packed tensor-parallel bit copy is out of bounds");
    }
    if ((source_bit & 7u) == 0 &&
        (destination_bit & 7u) == 0) {
        const size_t full_bytes =
            bit_count / 8;
        if (full_bytes != 0) {
            std::memcpy(
                destination.data() +
                    destination_bit / 8,
                source.data() +
                    source_bit / 8,
                full_bytes);
            source_bit += full_bytes * 8;
            destination_bit += full_bytes * 8;
            bit_count -= full_bytes * 8;
        }
    }
    while (bit_count >= 8) {
        const size_t source_byte =
            source_bit >> 3;
        const int source_shift =
            static_cast<int>(source_bit & 7u);
        uint16_t source_word =
            source[source_byte];
        if (source_shift != 0 &&
            source_byte + 1 < source.size()) {
            source_word |=
                static_cast<uint16_t>(
                    source[source_byte + 1])
                << 8;
        }
        const uint8_t value =
            static_cast<uint8_t>(
                source_word >> source_shift);
        const size_t destination_byte =
            destination_bit >> 3;
        const int destination_shift =
            static_cast<int>(
                destination_bit & 7u);
        destination[destination_byte] |=
            static_cast<uint8_t>(
                value << destination_shift);
        if (destination_shift != 0 &&
            destination_byte + 1 <
                destination.size()) {
            destination[destination_byte + 1] |=
                static_cast<uint8_t>(
                    value >>
                    (8 - destination_shift));
        }
        source_bit += 8;
        destination_bit += 8;
        bit_count -= 8;
    }
    for (size_t bit = 0; bit < bit_count; ++bit) {
        if ((source[source_bit >> 3] >>
             (source_bit & 7u)) & 1u) {
            destination[destination_bit >> 3] |=
                static_cast<uint8_t>(
                    1u << (destination_bit & 7u));
        }
        ++source_bit;
        ++destination_bit;
    }
}

static std::vector<uint8_t> slice_packed_rows(
        const std::vector<uint8_t> & source,
        int64_t source_rows,
        int64_t source_items_per_row,
        int64_t row_begin,
        int64_t row_end,
        int64_t item_begin,
        int64_t item_count,
        int bits) {
    if (bits == 0) return {};
    if (source_rows <= 0 || source_items_per_row <= 0 ||
        row_begin < 0 || row_begin >= row_end ||
        row_end > source_rows || item_begin < 0 ||
        item_count <= 0 ||
        item_begin + item_count > source_items_per_row) {
        throw std::runtime_error("invalid packed tensor-parallel slice");
    }
    const int64_t destination_rows = row_end - row_begin;
    const size_t destination_items =
        static_cast<size_t>(destination_rows) * item_count;
    std::vector<uint8_t> destination(
        (destination_items * static_cast<size_t>(bits) + 7) / 8,
        0);
    for (int64_t row = 0; row < destination_rows; ++row) {
        const size_t source_item =
            static_cast<size_t>(row + row_begin) *
                source_items_per_row +
            static_cast<size_t>(item_begin);
        const size_t destination_item =
            static_cast<size_t>(row) * item_count;
        copy_packed_bits(
            source,
            source_item * static_cast<size_t>(bits),
            destination,
            destination_item * static_cast<size_t>(bits),
            static_cast<size_t>(item_count) *
                static_cast<size_t>(bits));
    }
    return destination;
}

mfq::nvq::Data slice_nvq_cpu(
        const mfq::nvq::Data & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    mfq::require_parallel_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NVQ");
    if (axis != TensorParallelAxis::Output &&
        axis != TensorParallelAxis::Input) {
        throw std::runtime_error("NVQ shard requires an output or input axis");
    }
    mfq::nvq::Data result = source;
    int64_t row_begin = 0;
    int64_t row_end = source.out;
    int64_t group_begin = 0;
    int64_t group_count = source.ng;
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid NVQ output shard");
        }
        row_begin = begin;
        row_end = end;
        result.out = static_cast<int>(end - begin);
        result.shape[0] = result.out;
    } else {
        if (begin < 0 || begin >= end || end > source.ng) {
            throw std::runtime_error("invalid NVQ input shard");
        }
        group_begin = begin;
        group_count = end - begin;
        result.ng = static_cast<int>(group_count);
        const int64_t element_begin = begin * source.gs;
        const int64_t element_end =
            std::min<int64_t>(end * source.gs, source.neuron_len);
        result.neuron_len =
            static_cast<int>(element_end - element_begin);
        result.shape[1] = result.neuron_len;
        const int vector_size = mfq::nvq::vector_size(source.format);
        if (element_begin % vector_size != 0 ||
            element_begin % 8 != 0) {
            throw std::runtime_error(
                "NVQ input shard does not preserve vector boundaries");
        }
        result.nvec =
            (result.neuron_len + vector_size - 1) / vector_size;
        result.nsign = (result.neuron_len + 7) / 8;
    }

    result.neuron_scale_h.assign(
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_begin),
        source.neuron_scale_h.begin() +
            static_cast<ptrdiff_t>(row_end));
    result.sub_scale_packed = slice_packed_rows(
        source.sub_scale_packed,
        source.out, source.ng,
        row_begin, row_end,
        group_begin, group_count,
        source.sub_bits);

    const int vector_size = mfq::nvq::vector_size(source.format);
    const int64_t element_begin = group_begin * source.gs;
    const int64_t vector_begin = element_begin / vector_size;
    const int64_t vector_count =
        axis == TensorParallelAxis::Output
        ? source.nvec
        : result.nvec;
    result.indices_packed = slice_packed_rows(
        source.indices_packed,
        source.out, source.nvec,
        row_begin, row_end,
        vector_begin, vector_count,
        mfq::nvq::index_bits(source.format));

    if (nvq_delta_format(source.format)) {
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.ng,
            row_begin, row_end,
            group_begin, group_count,
            1);
    } else if (nvq_no_aux_format(source.format)) {
        result.aux_packed.clear();
    } else {
        const int64_t sign_begin = element_begin / 8;
        const int64_t sign_count =
            axis == TensorParallelAxis::Output
            ? source.nsign
            : result.nsign;
        result.aux_packed = slice_packed_rows(
            source.aux_packed,
            source.out, source.nsign,
            row_begin, row_end,
            sign_begin, sign_count,
            7);
    }
    return result;
}
static std::vector<uint8_t> repack_nvq2_exec_metadata(const mfq::nvq::Data & c) {
    if ((c.format != 2 && c.format != 5) || c.nvec != c.nsign) {
        throw std::runtime_error("NVQ2 execution metadata requires one sign record per vector");
    }
    const size_t count = (size_t)c.out * c.nvec;
    if (c.indices_packed.size() != count) {
        throw std::runtime_error("NVQ2 index stream length mismatch during execution repack");
    }
    std::vector<uint8_t> metadata(count * 2);
    for (size_t linear = 0; linear < count; ++linear) {
        const size_t bit = linear * 7;
        const size_t byte = bit >> 3;
        const int shift = (int)(bit & 7);
        uint16_t word = c.aux_packed[byte];
        if (byte + 1 < c.aux_packed.size()) {
            word |= (uint16_t)c.aux_packed[byte + 1] << 8;
        }
        const uint8_t index = c.indices_packed[linear];
        const uint8_t mask7 = (uint8_t)((word >> shift) & 0x7f);
        uint8_t parity = mask7;
        parity ^= parity >> 4;
        parity ^= parity >> 2;
        parity ^= parity >> 1;
        const uint8_t last = (parity & 1u) ^
            (c.sign_mode ? ((index >> 7) & 1u) : 0u);
        metadata[2 * linear] = c.indices_packed[linear];
        metadata[2 * linear + 1] = mask7 | (last << 7);
    }
    return metadata;
}

static uint32_t load_compact_bits_cpu(
        const std::vector<uint8_t> & data, size_t bit, int bits) {
    const size_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = byte < data.size() ? data[byte] : 0;
    if (byte + 1 < data.size()) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < data.size()) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

static void put_compact_bits96(
        uint32_t (&words)[3], int bit, int bits, uint32_t value) {
    const int word = bit >> 5;
    const int shift = bit & 31;
    words[word] |= value << shift;
    if (shift + bits > 32) words[word + 1] |= value >> (32 - shift);
}

static uint32_t compact_parity7(uint32_t value) {
    value &= 0x7fu;
    value ^= value >> 4;
    value ^= value >> 2;
    value ^= value >> 1;
    return value & 1u;
}

// NVQ2J-XL needs exactly 64 bits per gs24 group (three 12-bit indices, three
// 8-bit sign masks, and one 4-bit state). NVQ3J-L stores three indices in
// a 96-bit record per gs24 group (six 10-bit indices, three 8-bit sign masks,
// and one 4-bit state).
// This replaces the two independently bit-packed streams on CUDA; it does not
// keep a second execution copy of the model weights.
static std::vector<uint8_t> repack_extended_jsc_group_metadata(
        const mfq::nvq::Data & c,
        const std::array<uint8_t, 16> * state_remap = nullptr,
        const std::array<std::array<uint16_t, 4096>, 4> * index_remap = nullptr) {
    if (c.format != 14 && c.format != 15) {
        throw std::runtime_error("extended JSC group metadata requires NVQ2J-XL or NVQ3J-L");
    }
    const int index_bits = c.format == 14 ? 12 : 10;
    const int vectors_per_group = c.format == 14 ? 3 : 6;
    const size_t record_bytes = c.format == 14 ? 8 : 12;
    const size_t group_count = static_cast<size_t>(c.out) * c.ng;
    std::vector<uint8_t> metadata(group_count * record_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(group_count), 4096, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / c.ng;
            const int group = static_cast<int>(linear - row * c.ng);
            const size_t vector_base = static_cast<size_t>(row) * c.nvec +
                static_cast<size_t>(group) * vectors_per_group;
            const size_t sign_base = static_cast<size_t>(row) * c.nsign +
                static_cast<size_t>(group) * 3;
            const uint32_t source_state = load_compact_bits_cpu(
                c.sub_scale_packed, static_cast<size_t>(linear) * c.sub_bits,
                c.sub_bits);
            const uint32_t state = state_remap == nullptr
                ? source_state
                : (*state_remap)[source_state];
            uint8_t * destination = metadata.data() +
                static_cast<size_t>(linear) * record_bytes;
            if (c.format == 14) {
                uint64_t packed = 0;
                for (int local = 0; local < 3; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t source_index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    const uint32_t source_bank = static_cast<uint8_t>(
                        c.codebook[36 + source_state]);
                    const uint32_t index = index_remap == nullptr
                        ? source_index
                        : (*index_remap)[source_bank][source_index];
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    const uint32_t segment = index | (mask8 << index_bits);
                    packed |= static_cast<uint64_t>(segment) << (local * 20);
                }
                packed |= static_cast<uint64_t>(state) << 60;
                std::memcpy(destination, &packed, sizeof(packed));
            } else {
                uint32_t words[3] = {0, 0, 0};
                for (int local = 0; local < 6; ++local) {
                    const size_t vector = vector_base + local;
                    const uint32_t index = vector < static_cast<size_t>((row + 1) * c.nvec)
                        ? load_compact_bits_cpu(c.indices_packed, vector * index_bits, index_bits)
                        : 0;
                    put_compact_bits96(words, local * index_bits, index_bits, index);
                }
                for (int local = 0; local < 3; ++local) {
                    const size_t sign = sign_base + local;
                    const uint32_t mask7 = sign < static_cast<size_t>((row + 1) * c.nsign)
                        ? load_compact_bits_cpu(c.aux_packed, sign * 7, 7)
                        : 0;
                    const uint32_t mask8 = mask7 | (compact_parity7(mask7) << 7);
                    put_compact_bits96(words, 60 + local * 8, 8, mask8);
                }
                put_compact_bits96(words, 84, 4, state);
                std::memcpy(destination, words, sizeof(words));
            }
        }
    });
    return metadata;
}





static std::vector<int8_t> reorder_extended_e8_codebook_for_stage3(
        const mfq::nvq::Data & c,
        std::array<uint8_t, 16> & state_remap,
        std::array<std::array<uint16_t, 4096>, 4> & index_remap) {
    constexpr size_t metadata_bytes = 64;
    constexpr size_t bank_bytes = 4096 * 8;
    constexpr int bank_count = 4;
    if (c.format != 14 ||
        c.codebook.size() != metadata_bytes + bank_count * bank_bytes ||
        static_cast<uint8_t>(c.codebook[1]) != bank_count) {
        throw std::runtime_error("extended E8 runtime layout requires NVQ2J-XL");
    }

    uint64_t counts[bank_count] = {0, 0, 0, 0};
    std::vector<uint64_t> index_counts(bank_count * 4096, 0);
    const size_t groups = static_cast<size_t>(c.out) * c.ng;
    for (size_t linear = 0; linear < groups; ++linear) {
        const uint32_t state = load_compact_bits_cpu(
            c.sub_scale_packed, linear * c.sub_bits, c.sub_bits);
        const uint32_t bank = static_cast<uint8_t>(c.codebook[36 + state]);
        if (bank >= bank_count) {
            throw std::runtime_error("NVQ2J-XL state references an invalid bank");
        }
        ++counts[bank];
        const size_t row = linear / c.ng;
        const size_t group = linear - row * c.ng;
        const size_t vector_base = row * c.nvec + group * 3;
        const size_t vector_end = (row + 1) * c.nvec;
        for (int local = 0; local < 3; ++local) {
            const size_t vector = vector_base + local;
            if (vector >= vector_end) break;
            const uint32_t index = load_compact_bits_cpu(
                c.indices_packed, vector * 12, 12);
            ++index_counts[bank * 4096 + index];
        }
    }
    std::array<int, bank_count> order = {0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int left, int right) {
        return counts[left] > counts[right];
    });
    std::vector<int8_t> reordered = c.codebook;
    int runtime_bank[bank_count] = {0, 0, 0, 0};
    for (int destination = 0; destination < bank_count; ++destination) {
        const int source = order[destination];
        runtime_bank[source] = destination;
        std::array<int, 4096> index_order;
        std::iota(index_order.begin(), index_order.end(), 0);
        std::stable_sort(index_order.begin(), index_order.end(),
            [&](int left, int right) {
                return index_counts[source * 4096 + left] >
                    index_counts[source * 4096 + right];
            });
        for (int destination_index = 0;
             destination_index < 4096;
             ++destination_index) {
            const int source_index = index_order[destination_index];
            index_remap[source][source_index] =
                static_cast<uint16_t>(destination_index);
            std::memcpy(
                reordered.data() + metadata_bytes + destination * bank_bytes +
                    destination_index * 8,
                c.codebook.data() + metadata_bytes + source * bank_bytes +
                    source_index * 8,
                8);
        }
    }
    int bank_state_count[bank_count] = {0, 0, 0, 0};
    for (int source_state = 0; source_state < 16; ++source_state) {
        const uint8_t source_bank = static_cast<uint8_t>(
            c.codebook[36 + source_state]);
        const int rank_in_bank = bank_state_count[source_bank]++;
        if (rank_in_bank >= 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank has more than four scale states");
        }
        const int destination_bank = runtime_bank[source_bank];
        const int destination_state = rank_in_bank * bank_count + destination_bank;
        state_remap[source_state] = static_cast<uint8_t>(destination_state);
        std::memcpy(
            reordered.data() + 4 + destination_state * sizeof(uint16_t),
            c.codebook.data() + 4 + source_state * sizeof(uint16_t),
            sizeof(uint16_t));
        reordered[36 + destination_state] = static_cast<int8_t>(destination_bank);
    }
    for (int bank = 0; bank < bank_count; ++bank) {
        if (bank_state_count[bank] != 4) {
            throw std::runtime_error(
                "NVQ2J-XL runtime bank must have four scale states");
        }
    }
    return reordered;
}

static std::vector<uint8_t> remap_extended_e8_sub_scale_states(
        const mfq::nvq::Data & c,
        const std::array<uint8_t, 16> & state_remap) {
    if (c.format != 14 || c.sub_bits != 4) {
        throw std::runtime_error(
            "extended E8 scale-state remap requires 4-bit NVQ2J-XL states");
    }
    const size_t state_count = static_cast<size_t>(c.out) * c.ng;
    const size_t expected_bytes = (state_count + 1) / 2;
    if (c.sub_scale_packed.size() != expected_bytes) {
        throw std::runtime_error("NVQ2J-XL scale-state stream length mismatch");
    }
    std::vector<uint8_t> remapped(expected_bytes, 0);
    mfq_parallel_for(0, static_cast<int64_t>(expected_bytes), 4096,
        [&](int64_t begin, int64_t end) {
            for (int64_t byte = begin; byte < end; ++byte) {
                const uint8_t source = c.sub_scale_packed[byte];
                const size_t low_state = static_cast<size_t>(byte) * 2;
                const uint8_t low = state_remap[source & 0x0fu];
                const uint8_t high = low_state + 1 < state_count
                    ? state_remap[source >> 4]
                    : 0;
                remapped[byte] = low | static_cast<uint8_t>(high << 4);
            }
        });
    return remapped;
}

NvqWeight to_device_nvq(
        const mfq::nvq::Data & c,
        bool cuda,
        bool nvq2, bool extended, int device) {
    NvqWeight w;
    std::array<uint8_t, 16> e8_state_remap = {};
    std::unique_ptr<std::array<std::array<uint16_t,4096>,4>> e8_index_remap;
    std::vector<int8_t> e8_runtime_codebook;
    std::vector<uint8_t> e8_runtime_sub_scale;
    const bool use_extended_e8 =
        cuda && c.format == 14 &&
        extended;
    if (use_extended_e8) {
        e8_index_remap=std::make_unique<std::array<std::array<uint16_t,4096>,4>>();
        e8_runtime_codebook = reorder_extended_e8_codebook_for_stage3(
            c, e8_state_remap, *e8_index_remap);
        e8_runtime_sub_scale = remap_extended_e8_sub_scale_states(
            c, e8_state_remap);
    }
    w.format = c.format;
    w.kernel_format = c.format;
    w.sign_mode = c.sign_mode;
    w.sub_bits = c.sub_bits;
    w.gs = c.gs;
    w.out = c.out;
    w.ng = c.ng;
    w.neuron_len = c.neuron_len;
    w.shape = c.shape;
    if (cuda && (c.format == 14 || c.format == 15) &&
        extended) {
        auto metadata = repack_extended_jsc_group_metadata(
            c,
            use_extended_e8 ? &e8_state_remap : nullptr,
            use_extended_e8 ? e8_index_remap.get() : nullptr);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 14
            ? kNvq2JscXlGroupExecKernelFormat
            : kNvq3JscLGroupExecKernelFormat;
    } else if (cuda && (c.format == 2 || c.format == 5) &&
               nvq2) {
        auto metadata = repack_nvq2_exec_metadata(c);
        w.indices_packed = cpu_u8_tensor(
            metadata, {(int64_t)metadata.size()});
        w.aux_packed = mfq_tensor_backend::empty(
            {0}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kUInt8));
        w.kernel_format = c.format == 5 ? 6 : 4;
    } else if (
        c.format == 10 && c.codebook.size() >= 64 &&
        c.codebook[3] == 1 && c.codebook[1] == 2) {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
        w.kernel_format = 11;
    } else {
        w.indices_packed = cpu_u8_tensor(
            c.indices_packed, {(int64_t)c.indices_packed.size()});
        w.aux_packed = cpu_u8_tensor(
            c.aux_packed, {(int64_t)c.aux_packed.size()});
    }
    w.sub_scale_packed = use_extended_e8
        ? cpu_u8_tensor(
            e8_runtime_sub_scale, {(int64_t)e8_runtime_sub_scale.size()})
        : cpu_u8_tensor(
            c.sub_scale_packed, {(int64_t)c.sub_scale_packed.size()});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    if (
        c.format == 5 || c.format == 7 || c.format == 9 ||
        c.format == 10 || c.format == 12 ||
        c.format == 13 || c.format == 14 || c.format == 15) {
        if (use_extended_e8) {
            w.codebook = cpu_i8_tensor(
                e8_runtime_codebook, {(int64_t)e8_runtime_codebook.size()});
        } else {
            w.codebook = cpu_i8_tensor(
                c.codebook, {(int64_t)c.codebook.size()});
        }
    } else {
        const int dims = mfq::nvq::vector_size(c.format);
        const int entries = c.format == 1 ? 2048 : (c.format == 8 ? 1024 : 256);
        w.codebook = cpu_i8_tensor(c.codebook, {entries, dims});
    }
    // Retain the immutable map before upload. Decode and graph capture must
    // never copy codebook metadata back from the device.
    if ((w.kernel_format==5 || w.kernel_format==13 || w.kernel_format==14 ||
         w.kernel_format==10 || w.kernel_format==12 || w.kernel_format==15) &&
        w.codebook.numel()>=64) {
        const auto* metadata=w.codebook.data_ptr<int8_t>();
        for(unsigned state=0;state<16;++state)
            w.bank_map|=uint32_t(uint8_t(metadata[36+state])&3u)<<(2*state);
        if((w.kernel_format==12 || w.kernel_format==15) && metadata[1]==2 && metadata[3]==1) {
            static std::atomic<uint32_t> reported{0};const uint32_t bit=1u<<w.kernel_format;
            if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                "mfe_d4_analytic_scale_metadata scope=first_validated_table format=%d banks=2 mode=1\n",
                int(w.kernel_format));
        }
    }
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, target_device);
        w.indices_packed = w.indices_packed.to(target).contiguous();
        w.aux_packed = w.aux_packed.to(target).contiguous();
        w.sub_scale_packed =
            w.sub_scale_packed.to(target).contiguous();
        if(w.kernel_format==16 || w.kernel_format==17)
            w.sub_scale_packed=mfq_tensor_backend::empty({0},w.indices_packed.options());
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.codebook = w.codebook.to(target).contiguous();
        prepare_nvq_decode_records(w);
        {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            mfq::cuda::ResidentAllocationScope resident;
#endif
            (void)w.workspace(1);
        }
    }
    return w;
}

static size_t packed_nbytes(size_t count, int bits) {
    return bits == 0 ? 0 : (count * (size_t)bits + 7) / 8;
}

NepqWeight to_device_nepq(const mfq::nepq::Data & cpu, bool cuda) {
    NepqWeight value;
    value.format = cpu.format;
    value.state_bits = cpu.state_bits;
    value.n_experts = cpu.n_experts;
    value.out_per_expert = cpu.out_per_expert;
    value.neuron_len = cpu.neuron_len;
    value.ng = cpu.ng;
    value.rotation_block = cpu.rotation_block;
    value.rotation_seed = cpu.rotation_seed;
    value.residual = cpu.residual;
    value.residual_position_bits = cpu.residual_position_bits;
    value.residual_block_vectors = cpu.residual_block_vectors;
    value.indices_packed = cpu_u8_tensor(
        cpu.indices_packed, {(int64_t)cpu.indices_packed.size()});
    value.aux_packed = cpu_u8_tensor(
        cpu.aux_packed, {(int64_t)cpu.aux_packed.size()});
    value.state_packed = cpu_u8_tensor(
        cpu.state_packed, {(int64_t)cpu.state_packed.size()});
    value.neuron_scale = cpu_f16_to_f32_tensor(
        cpu.neuron_scale_h, cpu.n_experts * cpu.out_per_expert);
    value.table_pool = cpu_i8_tensor(
        cpu.table_pool, {cpu.bank_count, cpu.runtime_table_bytes});
    const int grouped_stride =
        (cpu.profile == 0 || cpu.profile == 4)
        ? 2112 : cpu.runtime_table_bytes;
    value.grouped_table_pool = cpu_i8_tensor(
        cpu.grouped_table_pool, {cpu.bank_count, grouped_stride});
    value.bank_ids = cpu_u8_tensor(
        cpu.bank_ids, {
            (int64_t)cpu.n_experts * cpu.out_per_expert, cpu.nsuper});
    value.rotation_signs = cpu_i8_tensor(
        cpu.rotation_signs, {(int64_t)cpu.rotation_signs.size()});
    if (cpu.residual) {
        value.residual_codebook = cpu_f16_tensor(
            cpu.residual_codebook_h, {1024, 8});
        value.residual_first = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_first.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
        value.residual_second = mfq_tensor_backend::from_blob(
            (void *)cpu.residual_second_dense.data(),
            {cpu.n_experts * cpu.out_per_expert,
             cpu.residual_blocks_per_row},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt16)).clone();
    }
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device());
        value.indices_packed =
            value.indices_packed.to(target).contiguous();
        value.aux_packed =
            value.aux_packed.to(target).contiguous();
        value.state_packed =
            value.state_packed.to(target).contiguous();
        value.neuron_scale =
            value.neuron_scale.to(target).contiguous();
        value.table_pool =
            value.table_pool.to(target).contiguous();
        value.grouped_table_pool =
            value.grouped_table_pool.to(target).contiguous();
        value.bank_ids =
            value.bank_ids.to(target).contiguous();
        value.rotation_signs =
            value.rotation_signs.to(target).contiguous();
        if (cpu.residual) {
            value.residual_codebook =
                value.residual_codebook.to(target).contiguous();
            value.residual_first =
                value.residual_first.to(target).contiguous();
            value.residual_second =
                value.residual_second.to(target).contiguous();
        }
    }
    return value;
}

mfq::nvq::Data select_nvq_cpu_rows(
        const mfq::nvq::Data & source,
        const std::vector<int64_t> & rows) {
    if (source.shape.size() != 2 ||
        source.axis != 0 ||
        source.shape[0] != source.out ||
        rows.empty()) {
        throw std::runtime_error(
            "invalid NVQ MoE output shard source");
    }
    mfq::nvq::Data result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.neuron_scale_h.resize(rows.size());
    const int index_bits =
        mfq::nvq::index_bits(source.format);
    const bool delta =
        nvq_delta_format(source.format);
    const bool no_aux =
        nvq_no_aux_format(source.format);
    const int aux_items =
        delta ? source.ng :
        (no_aux ? 0 : source.nsign);
    const int aux_bits =
        delta ? 1 : (no_aux ? 0 : 7);
    result.sub_scale_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.sub_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(aux_items),
            aux_bits),
        0);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NVQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.sub_scale_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.sub_bits,
            result.sub_scale_packed,
            destination * source.ng *
                source.sub_bits,
            static_cast<size_t>(source.ng) *
                source.sub_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * index_bits,
            result.indices_packed,
            destination * source.nvec *
                index_bits,
            static_cast<size_t>(source.nvec) *
                index_bits);
        if (aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    aux_items * aux_bits,
                result.aux_packed,
                destination * aux_items *
                    aux_bits,
                static_cast<size_t>(aux_items) *
                    aux_bits);
        }
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
    }
    return result;
}
mfq::nepq::Data select_nepq_cpu_rows(
        const mfq::nepq::Data & source,
        const std::vector<int64_t> & rows,
        int output_per_expert,
        int selected_experts) {
    const int destination_experts = selected_experts < 0
        ? source.n_experts
        : selected_experts;
    const int source_rows =
        source.n_experts *
        source.out_per_expert;
    if (rows.empty() ||
        output_per_expert <= 0 ||
        destination_experts <= 0 ||
        static_cast<int>(rows.size()) !=
            destination_experts *
                output_per_expert) {
        throw std::runtime_error(
            "invalid NEPQ MoE output shard");
    }
    mfq::nepq::Data result = source;
    result.n_experts = destination_experts;
    result.out_per_expert = output_per_expert;
    result.neuron_scale_h.resize(rows.size());
    result.state_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.state_bits),
        0);
    result.indices_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.nvec),
            source.index_bits),
        0);
    result.aux_packed.assign(
        packed_nbytes(
            rows.size() *
                static_cast<size_t>(source.ng),
            source.aux_bits),
        0);
    result.bank_ids.resize(
        rows.size() * source.nsuper);
    if (source.residual) {
        result.residual_first.resize(
            rows.size() * source.residual_blocks_per_row);
        result.residual_second_dense.resize(
            rows.size() * source.residual_blocks_per_row);
    }
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source_rows) {
            throw std::runtime_error(
                "NEPQ MoE shard row is out of range");
        }
        copy_packed_bits(
            source.state_packed,
            static_cast<size_t>(source_row) *
                source.ng * source.state_bits,
            result.state_packed,
            destination * source.ng *
                source.state_bits,
            static_cast<size_t>(source.ng) *
                source.state_bits);
        copy_packed_bits(
            source.indices_packed,
            static_cast<size_t>(source_row) *
                source.nvec * source.index_bits,
            result.indices_packed,
            destination * source.nvec *
                source.index_bits,
            static_cast<size_t>(source.nvec) *
                source.index_bits);
        if (source.aux_bits != 0) {
            copy_packed_bits(
                source.aux_packed,
                static_cast<size_t>(source_row) *
                    source.ng * source.aux_bits,
                result.aux_packed,
                destination * source.ng *
                    source.aux_bits,
                static_cast<size_t>(source.ng) *
                    source.aux_bits);
        }
        std::memcpy(
            result.bank_ids.data() +
                destination * source.nsuper,
            source.bank_ids.data() +
                static_cast<size_t>(source_row) *
                    source.nsuper,
            static_cast<size_t>(source.nsuper));
        result.neuron_scale_h[destination] =
            source.neuron_scale_h[
                static_cast<size_t>(source_row)];
        if (source.residual) {
            std::memcpy(
                result.residual_first.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_first.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
            std::memcpy(
                result.residual_second_dense.data()
                    + destination * source.residual_blocks_per_row,
                source.residual_second_dense.data()
                    + static_cast<size_t>(source_row)
                        * source.residual_blocks_per_row,
                static_cast<size_t>(source.residual_blocks_per_row)
                    * sizeof(int16_t));
        }
    }
    return result;
}
