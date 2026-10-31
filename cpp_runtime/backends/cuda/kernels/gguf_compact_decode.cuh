#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace mfq::cuda::gguf::compact_detail {
template<int Format> struct Layout {
    static_assert(Format >= 143 && Format <= 151);
    static constexpr int bits = Format == 143 ? 0 : Format == 144 ? 8 :
        (Format == 145 || Format == 148) ? 6 : (Format == 146 || Format == 149) ? 7 : Format == 151 ? 4 : 5;
    static constexpr int steps = (Format >= 147 && Format <= 149) ? 64 : 32;
    static constexpr int bytes = bits ? 2 + bits * steps / 8 : 28;
};
// Normative Agention codebook, with its license and notice in third_party.
// Read-only global memory permits divergent lane indices without constant-
// memory broadcast serialization. This is 8 KiB, not an expanded state table.
static __device__ __align__(4) const std::uint16_t gyro_book[4096] = {
#include "../../../third_party/agention-gyro/codebook_f16.inc"
};
static __device__ __forceinline__ float scale_at(const std::uint8_t* at) {
    __half_raw value;
    value.x = std::uint16_t(at[0]) | (std::uint16_t(at[1]) << 8);
    return __half2float(__half(value));
}
static __device__ __forceinline__ std::uint32_t ternary_four(const std::uint8_t* q, int lane) {
    // PTQ1_0 stores digit planes over 16-byte, 8-byte and 2-byte tiles.
    // The last two lanes each span two planes of the final two bytes.
    constexpr std::uint32_t mask = 0x00ff00ffu, low = 0x00030003u;
    std::uint32_t codes;
    if (lane < 30) {
        const int digit = lane < 20 ? lane / 4 : (lane - 20) / 2;
        const int byte = lane < 20 ? (lane % 4) * 4 : 16 + ((lane - 20) % 2) * 4;
        const std::uint32_t power = (0x511b090301ULL >> (digit * 8)) & 255u;
        const auto word = std::uint32_t(q[byte]) | (std::uint32_t(q[byte + 1]) << 8) |
            (std::uint32_t(q[byte + 2]) << 16) | (std::uint32_t(q[byte + 3]) << 24);
        // Two independent 16-bit lanes: 255*81 < 2^16. Masking reproduces
        // the normative uint8 wrap before *3, whose result is also <2^16.
        const auto a = (((((word & mask) * power) & mask) * 3) >> 8) & low;
        const auto b = ((((((word >> 8) & mask) * power) & mask) * 3) >> 8) & low;
        codes = a | (b << 8);
    } else {
        const std::uint32_t power = (lane & 1) ? 9u : 1u;
        const auto pair = std::uint32_t(q[24]) | (std::uint32_t(q[25]) << 16);
        const auto a = (((((pair * power) & mask) * 3) >> 8) & low);
        const auto b = (((((pair * (power * 3)) & mask) * 3) >> 8) & low);
        codes = (a & 255u) | ((a >> 8) & 0xff00u) | ((b << 16) & 0xff0000u) | ((b << 8) & 0xff000000u);
    }
    return __vsub4(codes, 0x01010101u);
}
static __device__ __forceinline__ float2 point(std::uint32_t index) {
    union { std::uint32_t bits; __half2 value; } entry;
    entry.bits = __ldg(reinterpret_cast<const std::uint32_t*>(gyro_book) + index);
    return __half22float2(entry.value);
}

template<int Bits, int Steps>
__device__ __forceinline__ float4 decode_four(const std::uint8_t* q, int lane) {
    if constexpr (!Bits) {
        const auto codes = ternary_four(q, lane);
        return make_float4(std::int8_t(codes), std::int8_t(codes >> 8),
            std::int8_t(codes >> 16), std::int8_t(codes >> 24));
    } else {
        constexpr int Bytes = Bits * Steps / 8;
        std::uint32_t hash0, hash1;
        if constexpr (Steps == 64) {
            // Adjacent pair states share one circular byte window.
            const int offset = (62 - 2 * lane) * Bits, byte = offset / 8, shift = offset % 8;
            std::uint32_t word = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) word |= std::uint32_t(q[(byte + j) % Bytes]) << (j * 8);
            hash0 = ((word >> (shift + Bits)) & 32767u) * 0x9e3779b1u;
            hash1 = ((word >> shift) & 32767u) * 0x9e3779b1u;
        } else {
            std::uint32_t state;
            if constexpr (Bits == 8) state = (std::uint32_t(q[(lane + 31) % 32]) << 8) | q[lane];
            else {
                const int offset = (31 - lane) * Bits, byte = offset / 8;
                const auto word = std::uint32_t(q[byte]) | (std::uint32_t(q[(byte + 1) % Bytes]) << 8) |
                    (std::uint32_t(q[(byte + 2) % Bytes]) << 16);
                state = word >> (offset % 8);
            }
            hash0 = (state & 32767u) * 0x9e3779b1u; hash1 = hash0;
        }
        const auto a = point(hash0 >> 21);
        const auto b = point(Steps == 64 ? hash1 >> 21 : (hash1 >> 10) & 2047u);
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

}
