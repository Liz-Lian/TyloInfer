#pragma once
#include "gguf_compact_decode.cuh"
// GGUF scalar decoder shared in design with the native Metal backend.
// Layouts and lookup tables are supplied by the bundled ggml-common.h.

__device__ __forceinline__ std::uint32_t gguf_sign_mask(std::uint32_t index) {
    return index | ((__popc(index) & 1u) << 7u);
}

template<int F> __device__ __forceinline__ float gguf_value(const std::uint8_t* row, std::uint64_t k) {
    std::uint32_t j = k % 32u, p = k % 256u;
    if constexpr (F >= 143 && F <= 151) {
        using L = mfq::cuda::gguf::compact_detail::Layout<F>;
        const auto* q = row + (k / 128) * L::bytes;
        const auto w = mfq::cuda::gguf::compact_detail::decode_four<L::bits, L::steps>(q + (L::bits ? 2 : 0), int(k % 128) / 4);
        const float d = mfq::cuda::gguf::compact_detail::scale_at(q + (L::bits ? 0 : 26));
        const auto part = k % 4;
        return __fmul_rn(part == 0 ? w.x : part == 1 ? w.y : part == 2 ? w.z : w.w, d);
    } else if constexpr (F == 42) {
        auto block = row + std::uint64_t(k / 64u) * 18u;
        const float scale = float(*reinterpret_cast<const half*>(block));
        const std::uint32_t code = (block[2u + (k % 64u) / 4u] >> (2u * (k % 4u))) & 3u;
        return scale * (int(code) - 1);
    } else if constexpr (F == 2 || F == 3) {
        if constexpr (F == 2) {
            auto b = reinterpret_cast<const block_q4_0*>(row) + k / 32u;
            return float(b->d) * (int((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) - 8);
        } else {
            auto b = reinterpret_cast<const block_q4_1*>(row) + k / 32u;
            return float(b->data.d) * ((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) + float(b->data.m);
        }
    } else if constexpr (F == 6 || F == 7) {
        if constexpr (F == 6) {
            auto b = reinterpret_cast<const block_q5_0*>(row) + k / 32u;
            std::uint32_t high = (b->qh[j / 8u] >> (j % 8u)) & 1u;
            return float(b->d) * (int(((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) | (high << 4u)) - 16);
        } else {
            auto b = reinterpret_cast<const block_q5_1*>(row) + k / 32u;
            std::uint32_t high = (b->qh[j / 8u] >> (j % 8u)) & 1u;
            return float(b->data.d) * (((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) | (high << 4u)) + float(b->data.m);
        }
    } else if constexpr (F == 8) {
        auto b = reinterpret_cast<const block_q8_0*>(row) + k / 32u;
        return float(b->d) * float(b->qs[j]);
    } else if constexpr (F == 10) {
        auto b = reinterpret_cast<const block_q2_K*>(row) + k / 256u;
        std::uint32_t scale = b->scales[p / 16u];
        std::uint32_t q = (b->qs[(p / 128u) * 32u + p % 32u] >> (2u * ((p % 128u) / 32u))) & 3u;
        return float(b->data.d) * (scale & 15u) * q - float(b->data.dmin) * (scale >> 4u);
    } else if constexpr (F == 11) {
        auto b = reinterpret_cast<const block_q3_K*>(row) + k / 256u;
        std::uint32_t g = p / 16u;
        std::uint32_t lo = (b->scales[g % 8u] >> (4u * (g / 8u))) & 15u;
        std::uint32_t hi = (b->scales[8u + g % 4u] >> (2u * (g / 4u))) & 3u;
        std::uint32_t q = (b->qs[(p / 128u) * 32u + p % 32u] >> (2u * ((p % 128u) / 32u))) & 3u;
        int value = int(q) - ((b->hmask[p % 32u] & (1u << (p / 32u))) ? 0 : 4);
        return float(b->d) * (int(lo | (hi << 4u)) - 32) * value;
    } else if constexpr (F == 12 || F == 13) {
        const std::uint8_t* scales;
        float d, minimum;
        std::uint32_t q;
        if constexpr (F == 12) {
            auto b = reinterpret_cast<const block_q4_K*>(row) + k / 256u;
            d = float(b->data.d); minimum = float(b->data.dmin); scales = b->scales;
            q = (b->qs[(p / 64u) * 32u + p % 32u] >> (4u * ((p % 64u) / 32u))) & 15u;
        } else {
            auto b = reinterpret_cast<const block_q5_K*>(row) + k / 256u;
            d = float(b->data.d); minimum = float(b->data.dmin); scales = b->scales;
            q = ((b->qs[(p / 64u) * 32u + p % 32u] >> (4u * ((p % 64u) / 32u))) & 15u)
                | (((b->qh[p % 32u] >> (p / 32u)) & 1u) << 4u);
        }
        std::uint32_t g = p / 32u;
        std::uint32_t s = g < 4u ? scales[g] & 63u : (scales[g + 4u] & 15u) | ((scales[g - 4u] >> 6u) << 4u);
        std::uint32_t m = g < 4u ? scales[g + 4u] & 63u : (scales[g + 4u] >> 4u) | ((scales[g] >> 6u) << 4u);
        return d * s * q - minimum * m;
    } else if constexpr (F == 14) {
        auto b = reinterpret_cast<const block_q6_K*>(row) + k / 256u;
        std::uint32_t n = p % 128u, l = n % 32u, quarter = n / 32u;
        std::uint32_t low = b->ql[(p / 128u) * 64u + (quarter % 2u) * 32u + l];
        std::uint32_t high = b->qh[(p / 128u) * 32u + l];
        int q = int(((low >> (4u * (quarter / 2u))) & 15u) | (((high >> (2u * quarter)) & 3u) << 4u)) - 32;
        return float(b->d) * float(b->scales[p / 16u]) * q;
    } else if constexpr (F == 15) {
        auto b = reinterpret_cast<const block_q8_K*>(row) + k / 256u;
        return b->d * float(b->qs[p]);
    } else if constexpr (F == 16) {
        auto b = reinterpret_cast<const block_iq2_xxs*>(row) + k / 256u;
        std::uint32_t group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        auto raw = reinterpret_cast<const std::uint8_t*>(b->qs) + group * 8u;
        std::uint32_t aux = std::uint32_t(raw[4]) | (std::uint32_t(raw[5]) << 8u) | (std::uint32_t(raw[6]) << 16u) | (std::uint32_t(raw[7]) << 24u);
        std::uint32_t signs = gguf_sign_mask((aux >> (7u * sub)) & 127u);
        std::uint32_t value = std::uint32_t((iq2xxs_grid[raw[sub]] >> (8u * c)) & 255ul);
        return float(b->d) * (.5f + float(aux >> 28u)) * .25f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 17) {
        auto b = reinterpret_cast<const block_iq2_xs*>(row) + k / 256u;
        std::uint32_t group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        std::uint32_t code = b->qs[group * 4u + sub], signs = gguf_sign_mask(code >> 9u);
        std::uint32_t value = std::uint32_t((iq2xs_grid[code & 511u] >> (8u * c)) & 255ul);
        std::uint32_t scale = (b->scales[group] >> (4u * (sub / 2u))) & 15u;
        return float(b->d) * (.5f + scale) * .25f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 22) {
        auto b = reinterpret_cast<const block_iq2_s*>(row) + k / 256u;
        std::uint32_t group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        std::uint32_t code = b->qs[group * 4u + sub] | (((std::uint32_t(b->qh[group]) >> (2u * sub)) & 3u) << 8u);
        std::uint32_t signs = b->qs[32u + group * 4u + sub];
        std::uint32_t value = std::uint32_t((iq2s_grid[code] >> (8u * c)) & 255ul);
        std::uint32_t scale = (b->scales[group] >> (4u * (sub / 2u))) & 15u;
        return float(b->d) * (.5f + scale) * .25f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 18) {
        auto b = reinterpret_cast<const block_iq3_xxs*>(row) + k / 256u;
        std::uint32_t group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        auto raw = b->qs + 64u + group * 4u;
        std::uint32_t aux = std::uint32_t(raw[0]) | (std::uint32_t(raw[1]) << 8u) | (std::uint32_t(raw[2]) << 16u) | (std::uint32_t(raw[3]) << 24u);
        std::uint32_t signs = gguf_sign_mask((aux >> (7u * sub)) & 127u);
        std::uint32_t grid = iq3xxs_grid[b->qs[group * 8u + sub * 2u + c / 4u]];
        std::uint32_t value = (grid >> (8u * (c % 4u))) & 255u;
        return float(b->d) * (.5f + float(aux >> 28u)) * .5f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 21) {
        auto b = reinterpret_cast<const block_iq3_s*>(row) + k / 256u;
        std::uint32_t group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        std::uint32_t code = b->qs[group * 8u + sub * 2u + c / 4u]
            | (((std::uint32_t(b->qh[group]) >> (sub * 2u + c / 4u)) & 1u) << 8u);
        std::uint32_t signs = b->signs[group * 4u + sub];
        std::uint32_t value = (iq3s_grid[code] >> (8u * (c % 4u))) & 255u;
        std::uint32_t scale = (b->scales[group / 2u] >> (4u * (group % 2u))) & 15u;
        return float(b->d) * (1u + 2u * scale) * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 20) {
        auto b = reinterpret_cast<const block_iq4_nl*>(row) + k / 32u;
        return float(b->d) * float(kvalues_iq4nl[(b->qs[j % 16u] >> (4u * (j / 16u))) & 15u]);
    } else if constexpr (F == 23) {
        auto b = reinterpret_cast<const block_iq4_xs*>(row) + k / 256u;
        std::uint32_t group = p / 32u;
        int scale = int(((b->scales_l[group / 2u] >> (4u * (group % 2u))) & 15u)
            | (((std::uint32_t(b->scales_h) >> (2u * group)) & 3u) << 4u)) - 32;
        std::uint32_t code = (b->qs[group * 16u + j % 16u] >> (4u * (j / 16u))) & 15u;
        return float(b->d) * scale * float(kvalues_iq4nl[code]);
    }
    return 0.f;
}

// Share a codebook lookup, sign word and scale across eight adjacent values,
// as in Strata's dq_iq* kernels (MIT, third_party/strata.LICENSE). A group
// begins on an eight-value boundary and cannot cross a GGUF block.
template<int F> __device__ __forceinline__ void gguf_values8(
        const std::uint8_t* row, std::uint64_t k, float (&values)[8]) {
    if constexpr(F>=143 && F<=151) {
        using L=mfq::cuda::gguf::compact_detail::Layout<F>;
        const auto* block=row+(k/128)*L::bytes;
        const auto* states=block+(L::bits?2:0);
        const auto lane=int(k%128)/4;
        const float scale=mfq::cuda::gguf::compact_detail::scale_at(block+(L::bits?0:26));
        // Eight-value groups never cross a 128-value block. Decode each
        // four-value state once, sharing its window, hash, codebook and scale.
        const auto a=mfq::cuda::gguf::compact_detail::decode_four<L::bits,L::steps>(states,lane);
        const auto b=mfq::cuda::gguf::compact_detail::decode_four<L::bits,L::steps>(states,lane+1);
        values[0]=__fmul_rn(a.x,scale);values[1]=__fmul_rn(a.y,scale);
        values[2]=__fmul_rn(a.z,scale);values[3]=__fmul_rn(a.w,scale);
        values[4]=__fmul_rn(b.x,scale);values[5]=__fmul_rn(b.y,scale);
        values[6]=__fmul_rn(b.z,scale);values[7]=__fmul_rn(b.w,scale);
    } else if constexpr(F==16 || F==17 || F==22 || F==18 || F==21) {
        const auto p=std::uint32_t(k%256), group=p/32, sub=p%32/8;
        float scale;
        std::uint32_t signs;
        std::uint64_t grid;
        if constexpr(F==16) {
            const auto* b=reinterpret_cast<const block_iq2_xxs*>(row)+k/256;
            const auto* raw=reinterpret_cast<const std::uint8_t*>(b->qs)+group*8;
            const auto aux=std::uint32_t(raw[4])|(std::uint32_t(raw[5])<<8)|
                (std::uint32_t(raw[6])<<16)|(std::uint32_t(raw[7])<<24);
            signs=gguf_sign_mask((aux>>(7*sub))&127);
            grid=iq2xxs_grid[raw[sub]];
            scale=float(b->d)*(.5f+float(aux>>28))*.25f;
        } else if constexpr(F==17) {
            const auto* b=reinterpret_cast<const block_iq2_xs*>(row)+k/256;
            const auto code=b->qs[group*4+sub];
            signs=gguf_sign_mask(code>>9);
            grid=iq2xs_grid[code&511];
            scale=float(b->d)*(.5f+float((b->scales[group]>>(4*(sub/2)))&15))*.25f;
        } else if constexpr(F==22) {
            const auto* b=reinterpret_cast<const block_iq2_s*>(row)+k/256;
            const auto code=b->qs[group*4+sub]|(((std::uint32_t(b->qh[group])>>(2*sub))&3)<<8);
            signs=b->qs[32+group*4+sub];
            grid=iq2s_grid[code];
            scale=float(b->d)*(.5f+float((b->scales[group]>>(4*(sub/2)))&15))*.25f;
        } else if constexpr(F==18) {
            const auto* b=reinterpret_cast<const block_iq3_xxs*>(row)+k/256;
            const auto* raw=b->qs+64+group*4;
            const auto aux=std::uint32_t(raw[0])|(std::uint32_t(raw[1])<<8)|
                (std::uint32_t(raw[2])<<16)|(std::uint32_t(raw[3])<<24);
            signs=gguf_sign_mask((aux>>(7*sub))&127);
            grid=std::uint64_t(iq3xxs_grid[b->qs[group*8+sub*2]])|
                (std::uint64_t(iq3xxs_grid[b->qs[group*8+sub*2+1]])<<32);
            scale=float(b->d)*(.5f+float(aux>>28))*.5f;
        } else {
            const auto* b=reinterpret_cast<const block_iq3_s*>(row)+k/256;
            const auto high=std::uint32_t(b->qh[group]);
            grid=std::uint64_t(iq3s_grid[b->qs[group*8+sub*2]|(((high>>(sub*2))&1)<<8)])|
                (std::uint64_t(iq3s_grid[b->qs[group*8+sub*2+1]|(((high>>(sub*2+1))&1)<<8)])<<32);
            signs=b->signs[group*4+sub];
            scale=float(b->d)*(1u+2u*((b->scales[group/2]>>(4*(group%2)))&15));
        }
        #pragma unroll
        for(int j=0;j<8;++j)
            values[j]=scale*float((grid>>(8*j))&255)*((signs&(1u<<j))?-1.f:1.f);
    } else {
        #pragma unroll
        for(int j=0;j<8;++j)values[j]=gguf_value<F>(row,k+j);
    }
}

template<typename Output> __device__ __forceinline__ void gguf_store8(
        Output* output, const float (&values)[8]) {
    struct __align__(16) Vector { Output values[8]; } converted;
    #pragma unroll
    for(int j=0;j<8;++j)converted.values[j]=Output(values[j]);
    #pragma unroll
    for(int j=0;j<int(sizeof(converted)/sizeof(uint4));++j)
        reinterpret_cast<uint4*>(output)[j]=reinterpret_cast<const uint4*>(&converted)[j];
}
