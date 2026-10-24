#include "tyloinfer/tq_rows.h"
#include "tyloinfer/format_features.h"
#include "tyloinfer/tensor_dtype.h"

#include <cassert>
#include <iostream>

namespace {
using Bytes = std::vector<std::uint8_t>;

Bytes tensor(bool columns, std::uint32_t salt, unsigned width = 256, unsigned gs = 16) {
    constexpr unsigned rows = 3;
    const unsigned bits[]{1, 3, 8};
    const auto scale_bytes = columns ? width * 2 : (rows * ((width + gs - 1) / gs) + 1) / 2;
    const auto packed_bytes = width * (bits[0] + bits[1] + bits[2]) / 8;
    Bytes result(64 + rows * 8 + scale_bytes + packed_bytes, 0);
    std::memcpy(result.data(), "RTRELL05", 8);
    for (auto value : std::vector<std::pair<unsigned, unsigned>>{
            {8, 5}, {12, rows}, {16, width}, {20, unsigned(columns)}, {24, gs}, {28, packed_bytes}})
        mfq::TqLayout::integer(result.data() + value.first, value.second, 4);
    unsigned offset = 0;
    for (unsigned row = 0; row < rows; ++row) {
        mfq::TqLayout::integer(result.data() + 64 + row * 8,
            offset | (std::uint64_t(bits[row] - 1) << 32) | (std::uint64_t(0x3c00) << 40), 8);
        offset += width * bits[row] / 32;
    }
    auto* scale = result.data() + 64 + rows * 8;
    if (columns) for (unsigned col = 0; col < width; ++col)
        mfq::TqLayout::integer(scale + col * 2, 0x3800 + salt, 2);
    else std::fill(scale, scale + scale_bytes, 0x89);
    for (unsigned i = 0; i < packed_bytes; ++i) scale[scale_bytes + i] = std::uint8_t(i + salt);
    return result;
}

Bytes bank(bool columns) {
    auto a = tensor(columns, 1), b = tensor(columns, 2);
    Bytes result(96 + a.size() + b.size());
    std::memcpy(result.data(), "RTQEXP01", 8);
    for (auto value : std::vector<std::pair<unsigned, unsigned>>{{8, 1}, {12, 2}, {16, 3}, {20, 251}, {24, 16}})
        mfq::TqLayout::integer(result.data() + value.first, value.second, 4);
    mfq::TqLayout::integer(result.data() + 64, 96, 8);
    mfq::TqLayout::integer(result.data() + 72, a.size(), 8);
    mfq::TqLayout::integer(result.data() + 80, 96 + a.size(), 8);
    mfq::TqLayout::integer(result.data() + 88, b.size(), 8);
    std::copy(a.begin(), a.end(), result.begin() + 96);
    std::copy(b.begin(), b.end(), result.begin() + 96 + a.size());
    return result;
}

mfq::ModelSource::TensorReader reader(const Bytes& value, std::size_t& read_bytes) {
    return [&value, &read_bytes](std::uint64_t offset, std::byte* target, std::size_t size) {
        if (offset > value.size() || size > value.size() - offset) throw std::out_of_range("test read");
        read_bytes += size;
        std::memcpy(target, value.data() + offset, size);
    };
}

void check(bool columns) {
    const auto value = bank(columns);
    std::size_t bytes = 0;
    auto parts = mfq::tq_row_segments(value.size(), reader(value, bytes), 251);
    assert(parts.size() == 2 && bytes == 64 + 32 + 2 * 64);
    assert(parts[0]->layout().rows == 3 && parts[1]->layout().width == 256);
    assert((mfq::tq_tensor_shape(value.size(), reader(value, bytes)) ==
        std::vector<std::int64_t>{2, 3, 251}));
    const auto flat = tensor(columns, 3);
    assert((mfq::tq_tensor_shape(flat.size(), reader(flat, bytes), {1, 3, 251}) ==
        std::vector<std::int64_t>{1, 3, 251}));
    for (const auto& shape : std::vector<std::vector<std::int64_t>>{
            {3, 251}, {2, 2, 251}, {2, 3, 250}, {2, 3, 0}, {2, -3, 251}, {2, 3, 1, 251}}) {
        bool rejected = false;
        try { (void)mfq::tq_tensor_shape(value.size(), reader(value, bytes), shape); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    for (unsigned expert = 0; expert < 2; ++expert) {
        const auto expected = tensor(columns, expert + 1);
        assert(parts[expert]->slice_rows(0, 3) == expected);
        auto partial = parts[expert]->slice_rows(1, 3);
        mfq::TqLayout layout(partial.data(), partial.size());
        assert(layout.rows == 2 && mfq::TqLayout::integer(partial.data() + 64, 4) == 0);
        assert(mfq::TqLayout::integer(partial.data() + 72, 4) == 24);
        const mfq::TqLayout original(expected.data(), expected.size());
        assert(std::equal(partial.begin() + layout.packed, partial.end(), expected.begin() + original.packed + 32));
    }
    for (int mutation = 0; mutation < 7; ++mutation) {
        auto bad = value;
        if (mutation == 0) bad[8] = 2;
        if (mutation == 1) bad[24] = 8;
        if (mutation == 2) bad[32] = 1;
        if (mutation == 3) bad[64] = 95;
        if (mutation == 4) bad.pop_back();
        if (mutation == 5) bad[96 + 12] = 4;
        if (mutation == 6) bad[20] = 250;
        bool refused = false;
        try { (void)mfq::tq_row_segments(bad.size(), reader(bad, bytes), 251); }
        catch (const std::exception&) { refused = true; }
        assert(refused);
    }
}

void check_columns(bool columns, unsigned gs) {
    auto value = tensor(columns, 3, 1024, gs);
    const mfq::TqLayout original(value.data(), value.size());
    if (columns) {
        for (unsigned col = 0; col < original.width; ++col)
            mfq::TqLayout::integer(value.data() + original.scales + col * 2, 0x3800 + col, 2);
    } else {
        for (std::size_t group = 0; group < original.rows * original.groups; ++group) {
            auto& byte = value[original.scales + group / 2];
            const unsigned shift = (group & 1) * 4;
            byte = (byte & ~(15u << shift)) | (((group * 7 + group / 3) & 15) << shift);
        }
    }
    std::size_t read_bytes = 0;
    const mfq::TqRows source(value.size(), reader(value, read_bytes));
    for (const auto [begin, end] : {std::pair<unsigned, unsigned>{0, 256}, {256, 768}, {768, 1024}, {0, 1024}}) {
        const auto part = source.slice_columns(begin, end);
        const mfq::TqLayout layout(part.data(), part.size());
        assert(layout.rows == original.rows && layout.width == end - begin && layout.columns == columns);
        std::uint64_t packed_end = 0;
        for (unsigned row = 0; row < layout.rows; ++row) {
            const auto descriptor = mfq::TqLayout::integer(value.data() + 64 + row * 8, 8);
            const auto sliced = mfq::TqLayout::integer(part.data() + 64 + row * 8, 8);
            const auto bits = ((descriptor >> 32) & 7) + 1;
            assert((descriptor >> 32) == (sliced >> 32) && std::uint32_t(sliced) * 4 == packed_end);
            const auto bytes = (end - begin) * bits / 8;
            assert(std::equal(part.begin() + layout.packed + packed_end,
                part.begin() + layout.packed + packed_end + bytes,
                value.begin() + original.packed + std::uint32_t(descriptor) * 4 + begin * bits / 8));
            packed_end += bytes;
            for (unsigned col = 0; col < layout.width; ++col) {
                if (columns) {
                    assert(mfq::TqLayout::integer(part.data() + layout.scales + col * 2, 2) ==
                        mfq::TqLayout::integer(value.data() + original.scales + (begin + col) * 2, 2));
                } else {
                    const auto from = row * original.groups + (begin + col) / original.gs;
                    const auto to = row * layout.groups + col / layout.gs;
                    assert(((part[layout.scales + to / 2] >> ((to & 1) * 4)) & 15) ==
                        ((value[original.scales + from / 2] >> ((from & 1) * 4)) & 15));
                }
            }
        }
        assert(packed_end == layout.packed_bytes);
    }
    for (const auto [begin, end] : {std::pair<unsigned, unsigned>{1, 256}, {0, 257}, {256, 256}, {0, 1280}}) {
        bool rejected = false;
        try { (void)source.slice_columns(begin, end); }
        catch (const std::out_of_range&) { rejected = true; }
        assert(rejected);
    }
    value[64] = 1;
    bool rejected = false;
    try { (void)source.slice_columns(256, 512); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
}

int main() {
    check(false);
    check(true);
    for (unsigned gs : {16, 128, 256, 512, 1024, 2048}) {
        check_columns(false, gs);
        check_columns(true, gs);
    }
    assert(mfq::supported_tensor_dtype("TQ") && mfq::supported_tensor_dtype("MXFP8"));
    for (const auto* dtype : {"NINT", "NVQ", "MFE", "FP8-128SQ", "MXFP4-SQ"})
        assert(mfq::supported_tensor_dtype(dtype) == mfq::mfq_compat_enabled);
    std::cout << "TQ expert spans, independent scales, row reads and admission: ok\n";
}
