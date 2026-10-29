METAL_FUNC uint gguf_sign_mask(uint index) {
    return index | ((popcount(index) & 1u) << 7u);
}

template<int F> METAL_FUNC float gguf_value(device const uchar* row, uint k) {
    uint j = k % 32u, p = k % 256u;
    if constexpr (F == 42) {
        auto block = row + ulong(k / 64u) * 18u;
        const float scale = float(*reinterpret_cast<device const half*>(block));
        const uint code = (block[2u + (k % 64u) / 4u] >> (2u * (k % 4u))) & 3u;
        return scale * (int(code) - 1);
    } else if constexpr (F == 2 || F == 3) {
        if constexpr (F == 2) {
            auto b = reinterpret_cast<device const block_q4_0*>(row) + k / 32u;
            return float(b->d) * (int((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) - 8);
        } else {
            auto b = reinterpret_cast<device const block_q4_1*>(row) + k / 32u;
            return float(b->d) * ((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) + float(b->m);
        }
    } else if constexpr (F == 6 || F == 7) {
        if constexpr (F == 6) {
            auto b = reinterpret_cast<device const block_q5_0*>(row) + k / 32u;
            uint high = (b->qh[j / 8u] >> (j % 8u)) & 1u;
            return float(b->d) * (int(((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) | (high << 4u)) - 16);
        } else {
            auto b = reinterpret_cast<device const block_q5_1*>(row) + k / 32u;
            uint high = (b->qh[j / 8u] >> (j % 8u)) & 1u;
            return float(b->d) * (((b->qs[j % 16u] >> (4u * (j / 16u))) & 15u) | (high << 4u)) + float(b->m);
        }
    } else if constexpr (F == 8) {
        auto b = reinterpret_cast<device const block_q8_0*>(row) + k / 32u;
        return float(b->d) * float(b->qs[j]);
    } else if constexpr (F == 10) {
        auto b = reinterpret_cast<device const block_q2_K*>(row) + k / 256u;
        uint scale = b->scales[p / 16u];
        uint q = (b->qs[(p / 128u) * 32u + p % 32u] >> (2u * ((p % 128u) / 32u))) & 3u;
        return float(b->d) * (scale & 15u) * q - float(b->dmin) * (scale >> 4u);
    } else if constexpr (F == 11) {
        auto b = reinterpret_cast<device const block_q3_K*>(row) + k / 256u;
        uint g = p / 16u;
        uint lo = (b->scales[g % 8u] >> (4u * (g / 8u))) & 15u;
        uint hi = (b->scales[8u + g % 4u] >> (2u * (g / 4u))) & 3u;
        uint q = (b->qs[(p / 128u) * 32u + p % 32u] >> (2u * ((p % 128u) / 32u))) & 3u;
        int value = int(q) - ((b->hmask[p % 32u] & (1u << (p / 32u))) ? 0 : 4);
        return float(b->d) * (int(lo | (hi << 4u)) - 32) * value;
    } else if constexpr (F == 12 || F == 13) {
        device const uchar* scales;
        float d, minimum;
        uint q;
        if constexpr (F == 12) {
            auto b = reinterpret_cast<device const block_q4_K*>(row) + k / 256u;
            d = float(b->d); minimum = float(b->dmin); scales = b->scales;
            q = (b->qs[(p / 64u) * 32u + p % 32u] >> (4u * ((p % 64u) / 32u))) & 15u;
        } else {
            auto b = reinterpret_cast<device const block_q5_K*>(row) + k / 256u;
            d = float(b->d); minimum = float(b->dmin); scales = b->scales;
            q = ((b->qs[(p / 64u) * 32u + p % 32u] >> (4u * ((p % 64u) / 32u))) & 15u)
                | (((b->qh[p % 32u] >> (p / 32u)) & 1u) << 4u);
        }
        uint g = p / 32u;
        uint s = g < 4u ? scales[g] & 63u : (scales[g + 4u] & 15u) | ((scales[g - 4u] >> 6u) << 4u);
        uint m = g < 4u ? scales[g + 4u] & 63u : (scales[g + 4u] >> 4u) | ((scales[g] >> 6u) << 4u);
        return d * s * q - minimum * m;
    } else if constexpr (F == 14) {
        auto b = reinterpret_cast<device const block_q6_K*>(row) + k / 256u;
        uint n = p % 128u, l = n % 32u, quarter = n / 32u;
        uint low = b->ql[(p / 128u) * 64u + (quarter % 2u) * 32u + l];
        uint high = b->qh[(p / 128u) * 32u + l];
        int q = int(((low >> (4u * (quarter / 2u))) & 15u) | (((high >> (2u * quarter)) & 3u) << 4u)) - 32;
        return float(b->d) * float(b->scales[p / 16u]) * q;
    } else if constexpr (F == 15) {
        auto b = reinterpret_cast<device const block_q8_K*>(row) + k / 256u;
        return b->d * float(b->qs[p]);
    } else if constexpr (F == 16) {
        auto b = reinterpret_cast<device const block_iq2_xxs*>(row) + k / 256u;
        uint group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        auto raw = reinterpret_cast<device const uchar*>(b->qs) + group * 8u;
        uint aux = uint(raw[4]) | (uint(raw[5]) << 8u) | (uint(raw[6]) << 16u) | (uint(raw[7]) << 24u);
        uint signs = gguf_sign_mask((aux >> (7u * sub)) & 127u);
        uint value = uint((iq2xxs_grid[raw[sub]] >> (8u * c)) & 255ul);
        return float(b->d) * (.5f + float(aux >> 28u)) * .25f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 17) {
        auto b = reinterpret_cast<device const block_iq2_xs*>(row) + k / 256u;
        uint group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        uint code = b->qs[group * 4u + sub], signs = gguf_sign_mask(code >> 9u);
        uint value = uint((iq2xs_grid[code & 511u] >> (8u * c)) & 255ul);
        uint scale = (b->scales[group] >> (4u * (sub / 2u))) & 15u;
        return float(b->d) * (.5f + scale) * .25f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 22) {
        auto b = reinterpret_cast<device const block_iq2_s*>(row) + k / 256u;
        uint group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        uint code = b->qs[group * 4u + sub] | (((uint(b->qh[group]) >> (2u * sub)) & 3u) << 8u);
        uint signs = b->qs[32u + group * 4u + sub];
        uint value = uint((iq2s_grid[code] >> (8u * c)) & 255ul);
        uint scale = (b->scales[group] >> (4u * (sub / 2u))) & 15u;
        return float(b->d) * (.5f + scale) * .25f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 18) {
        auto b = reinterpret_cast<device const block_iq3_xxs*>(row) + k / 256u;
        uint group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        auto raw = b->qs + 64u + group * 4u;
        uint aux = uint(raw[0]) | (uint(raw[1]) << 8u) | (uint(raw[2]) << 16u) | (uint(raw[3]) << 24u);
        uint signs = gguf_sign_mask((aux >> (7u * sub)) & 127u);
        uint grid = iq3xxs_grid[b->qs[group * 8u + sub * 2u + c / 4u]];
        uint value = (grid >> (8u * (c % 4u))) & 255u;
        return float(b->d) * (.5f + float(aux >> 28u)) * .5f * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 21) {
        auto b = reinterpret_cast<device const block_iq3_s*>(row) + k / 256u;
        uint group = p / 32u, sub = (p % 32u) / 8u, c = p % 8u;
        uint code = b->qs[group * 8u + sub * 2u + c / 4u]
            | (((uint(b->qh[group]) >> (sub * 2u + c / 4u)) & 1u) << 8u);
        uint signs = b->signs[group * 4u + sub];
        uint value = (iq3s_grid[code] >> (8u * (c % 4u))) & 255u;
        uint scale = (b->scales[group / 2u] >> (4u * (group % 2u))) & 15u;
        return float(b->d) * (1u + 2u * scale) * value * ((signs & (1u << c)) ? -1.f : 1.f);
    } else if constexpr (F == 20) {
        auto b = reinterpret_cast<device const block_iq4_nl*>(row) + k / 32u;
        return float(b->d) * float(kvalues_iq4nl[(b->qs[j % 16u] >> (4u * (j / 16u))) & 15u]);
    } else if constexpr (F == 23) {
        auto b = reinterpret_cast<device const block_iq4_xs*>(row) + k / 256u;
        uint group = p / 32u;
        int scale = int(((b->scales_l[group / 2u] >> (4u * (group % 2u))) & 15u)
            | (((uint(b->scales_h) >> (2u * group)) & 3u) << 4u)) - 32;
        uint code = (b->qs[group * 16u + j % 16u] >> (4u * (j / 16u))) & 15u;
        return float(b->d) * scale * float(kvalues_iq4nl[code]);
    }
    return 0.f;
}

template<int F> struct GgufConstantLookup {
    enum {
        entry_words = F == 18 || F == 21 ? 1 : 2,
        grid_words = F == 16 ? 512 : F == 17 ? 1024 : F == 22 ? 2048 : F == 18 ? 256 : F == 21 ? 512 : 0,
        words = grid_words,
        storage_size = words ? words : 1,
    };
    METAL_FUNC uint word(uint index, uint part) const {
        if constexpr (F == 16) return reinterpret_cast<constant uint*>(iq2xxs_grid)[index * 2u + part];
        if constexpr (F == 17) return reinterpret_cast<constant uint*>(iq2xs_grid)[index * 2u + part];
        if constexpr (F == 22) return reinterpret_cast<constant uint*>(iq2s_grid)[index * 2u + part];
        if constexpr (F == 18) return iq3xxs_grid[index];
        if constexpr (F == 21) return iq3s_grid[index];
        return 0u;
    }
    METAL_FUNC uint sign(uint index) const { return gguf_sign_mask(index); }
};

template<int F> struct GgufThreadgroupLookup {
    threadgroup const uint* table;
    METAL_FUNC uint word(uint index, uint part) const {
        return table[index * GgufConstantLookup<F>::entry_words + part];
    }
    METAL_FUNC uint sign(uint index) const { return gguf_sign_mask(index); }
};

template<int F> METAL_FUNC void gguf_load_lookup(threadgroup uint* table, uint local, uint threads) {
    const GgufConstantLookup<F> source;
    for (uint index = local; index < GgufConstantLookup<F>::grid_words; index += threads)
        table[index] = source.word(index / GgufConstantLookup<F>::entry_words, index % GgufConstantLookup<F>::entry_words);
}

template<int F, typename Lookup> METAL_FUNC float4 gguf_values4(device const uchar* row, uint k, Lookup lookup) {
    if constexpr (F == 42) {
        auto block = row + ulong(k / 64u) * 18u;
        const uint codes = block[2u + (k % 64u) / 4u];
        const float scale = float(*reinterpret_cast<device const half*>(block));
        return (float4((uint4(codes) >> uint4(0u, 2u, 4u, 6u)) & 3u) - 1.f) * scale;
    } else if constexpr (F == 8) {
        auto block = reinterpret_cast<device const block_q8_0*>(row) + k / 32u;
        const uint col = k % 32u;
        return float4(block->qs[col], block->qs[col + 1u], block->qs[col + 2u], block->qs[col + 3u]) * float(block->d);
    } else if constexpr (F == 16 || F == 17 || F == 22) {
        const uint p = k % 256u, group = p / 32u, sub = (p % 32u) / 8u;
        uint codes;
        uint signs;
        float scale;
        if constexpr (F == 16) {
            auto b = reinterpret_cast<device const block_iq2_xxs*>(row) + k / 256u;
            auto raw = reinterpret_cast<device const uchar*>(b->qs) + group * 8u;
            const uint aux = uint(raw[4]) | (uint(raw[5]) << 8u) | (uint(raw[6]) << 16u) | (uint(raw[7]) << 24u);
            signs = lookup.sign((aux >> (7u * sub)) & 127u);
            codes = lookup.word(raw[sub], (p % 8u) / 4u);
            scale = float(b->d) * (.5f + float(aux >> 28u)) * .25f;
        } else if constexpr (F == 17) {
            auto b = reinterpret_cast<device const block_iq2_xs*>(row) + k / 256u;
            const uint code = b->qs[group * 4u + sub];
            signs = lookup.sign(code >> 9u);
            codes = lookup.word(code & 511u, (p % 8u) / 4u);
            scale = float(b->d) * (.5f + float((b->scales[group] >> (4u * (sub / 2u))) & 15u)) * .25f;
        } else {
            auto b = reinterpret_cast<device const block_iq2_s*>(row) + k / 256u;
            const uint code = b->qs[group * 4u + sub] | (((uint(b->qh[group]) >> (2u * sub)) & 3u) << 8u);
            signs = b->qs[32u + group * 4u + sub];
            codes = lookup.word(code, (p % 8u) / 4u);
            scale = float(b->d) * (.5f + float((b->scales[group] >> (4u * (sub / 2u))) & 15u)) * .25f;
        }
        const float4 values = float4(as_type<uchar4>(codes));
        const float4 sign = select(float4(1.f), float4(-1.f), (uint4(signs) & (uint4(1u, 2u, 4u, 8u) << (p % 8u))) != 0u);
        return (scale * values) * sign;
    } else if constexpr (F == 18 || F == 21) {
        const uint p = k % 256u, group = p / 32u, sub = (p % 32u) / 8u;
        uint grid, signs;
        float scale;
        if constexpr (F == 18) {
            auto b = reinterpret_cast<device const block_iq3_xxs*>(row) + k / 256u;
            auto raw = b->qs + 64u + group * 4u;
            const uint aux = uint(raw[0]) | (uint(raw[1]) << 8u) | (uint(raw[2]) << 16u) | (uint(raw[3]) << 24u);
            signs = lookup.sign((aux >> (7u * sub)) & 127u);
            grid = lookup.word(b->qs[group * 8u + sub * 2u + (p % 8u) / 4u], 0u);
            scale = float(b->d) * (.5f + float(aux >> 28u)) * .5f;
        } else {
            auto b = reinterpret_cast<device const block_iq3_s*>(row) + k / 256u;
            const uint index = sub * 2u + (p % 8u) / 4u;
            const uint code = b->qs[group * 8u + index] | (((uint(b->qh[group]) >> index) & 1u) << 8u);
            signs = b->signs[group * 4u + sub];
            grid = lookup.word(code, 0u);
            scale = float(b->d) * (1u + 2u * ((b->scales[group / 2u] >> (4u * (group % 2u))) & 15u));
        }
        const float4 values = float4(as_type<uchar4>(grid));
        const float4 sign = select(float4(1.f), float4(-1.f), (uint4(signs) & (uint4(1u, 2u, 4u, 8u) << (p % 8u))) != 0u);
        return (scale * values) * sign;
    } else if constexpr (F == 12 || F == 13) {
        const uint col = k % 256u, group = col / 32u;
        device const uchar* scales;
        device const uchar* codes;
        float d, minimum;
        uint4 high(0u);
        if constexpr (F == 12) {
            auto block = reinterpret_cast<device const block_q4_K*>(row) + k / 256u;
            scales = block->scales; codes = block->qs;
            d = float(block->d); minimum = float(block->dmin);
        } else {
            auto block = reinterpret_cast<device const block_q5_K*>(row) + k / 256u;
            scales = block->scales; codes = block->qs;
            d = float(block->d); minimum = float(block->dmin);
            auto h = block->qh + col % 32u;
            high = (uint4(h[0], h[1], h[2], h[3]) >> (col / 32u)) & 1u;
        }
        const uint scale = group < 4u ? scales[group] & 63u
            : (scales[group + 4u] & 15u) | ((scales[group - 4u] >> 6u) << 4u);
        const uint bias = group < 4u ? scales[group + 4u] & 63u
            : (scales[group + 4u] >> 4u) | ((scales[group] >> 6u) << 4u);
        auto q = codes + col / 64u * 32u + col % 32u;
        const uint4 values = ((uint4(q[0], q[1], q[2], q[3]) >> (4u * ((col % 64u) / 32u))) & 15u) | (high << 4u);
        return float4(values) * (d * float(scale)) - minimum * float(bias);
    } else {
        return float4(gguf_value<F>(row, k), gguf_value<F>(row, k + 1u),
            gguf_value<F>(row, k + 2u), gguf_value<F>(row, k + 3u));
    }
}

template<int F> METAL_FUNC float4 gguf_values4(device const uchar* row, uint k) {
    return gguf_values4<F>(row, k, GgufConstantLookup<F>{});
}

METAL_FUNC uint gguf_read_u32(device const uchar* bytes) {
    const auto words = reinterpret_cast<device const ushort*>(bytes);
    return uint(words[0]) | (uint(words[1]) << 16u);
}

template<int F, typename X>
METAL_FUNC float gguf_iq4_dot32(device const uchar* row, X input, uint k) {
    device const uchar* codes;
    float scale;
    if constexpr (F == 20) {
        const auto block = reinterpret_cast<device const block_iq4_nl*>(row) + k / 32u;
        codes = block->qs;
        scale = float(block->d);
    } else {
        const auto block = reinterpret_cast<device const block_iq4_xs*>(row) + k / 256u;
        const uint group = (k % 256u) / 32u;
        codes = block->qs + group * 16u;
        const int local_scale = int(((block->scales_l[group / 2u] >> (4u * (group % 2u))) & 15u)
            | (((uint(block->scales_h) >> (2u * group)) & 3u) << 4u)) - 32;
        scale = float(block->d) * float(local_scale);
    }
    float sum = 0.f;
    for (uint local = 0u; local < 16u; local += 4u) {
        const uint4 packed(codes[local], codes[local + 1u], codes[local + 2u], codes[local + 3u]);
        const uint4 lower = packed & 15u, upper = packed >> 4u;
        const float4 low(kvalues_iq4nl[lower.x], kvalues_iq4nl[lower.y], kvalues_iq4nl[lower.z], kvalues_iq4nl[lower.w]);
        const float4 high(kvalues_iq4nl[upper.x], kvalues_iq4nl[upper.y], kvalues_iq4nl[upper.z], kvalues_iq4nl[upper.w]);
        const uint first = k + local, second = first + 16u;
        sum += dot(low, float4(input[first], input[first + 1u], input[first + 2u], input[first + 3u]));
        sum += dot(high, float4(input[second], input[second + 1u], input[second + 2u], input[second + 3u]));
    }
    return scale * sum;
}

template<int F, typename X, typename Lookup>
METAL_FUNC float gguf_iq_dot16(device const uchar* row, X input, uint k, Lookup lookup) {
    constexpr uint block_bytes = F == 16 ? 66u : F == 17 ? 74u : F == 22 ? 82u : F == 18 ? 98u : 110u;
    const auto block = row + ulong(k / 256u) * block_bytes;
    const uint group = (k % 256u) / 32u, first = (k % 32u) / 8u;
    uint auxiliary = 0u;
    float scale = float(*reinterpret_cast<device const half*>(block));
    if constexpr (F == 16 || F == 18) {
        auxiliary = gguf_read_u32(block + (F == 16 ? 6u + group * 8u : 66u + group * 4u));
        scale *= (.5f + float(auxiliary >> 28u)) * (F == 16 ? .25f : .5f);
    } else if constexpr (F == 17 || F == 22) {
        scale *= (.5f + float((block[(F == 17 ? 66u : 74u) + group] >> (4u * (first / 2u))) & 15u)) * .25f;
    } else {
        scale *= float(1u + 2u * ((block[106u + group / 2u] >> (4u * (group % 2u))) & 15u));
    }
    float sum = 0.f;
    for (uint section = 0u; section < 2u; ++section) {
        const uint sub = first + section;
        uint signs, indices[2];
        if constexpr (F == 16) {
            indices[0] = indices[1] = block[2u + group * 8u + sub];
            signs = lookup.sign((auxiliary >> (7u * sub)) & 127u);
        } else if constexpr (F == 17) {
            const uint code = reinterpret_cast<device const ushort*>(block + 2u)[group * 4u + sub];
            indices[0] = indices[1] = code & 511u;
            signs = lookup.sign(code >> 9u);
        } else if constexpr (F == 22) {
            indices[0] = indices[1] = uint(block[2u + group * 4u + sub])
                | (((uint(block[66u + group]) >> (2u * sub)) & 3u) << 8u);
            signs = block[34u + group * 4u + sub];
        } else if constexpr (F == 18) {
            indices[0] = block[2u + group * 8u + sub * 2u];
            indices[1] = block[3u + group * 8u + sub * 2u];
            signs = lookup.sign((auxiliary >> (7u * sub)) & 127u);
        } else {
            const uint high = block[66u + group];
            indices[0] = uint(block[2u + group * 8u + sub * 2u]) | (((high >> (sub * 2u)) & 1u) << 8u);
            indices[1] = uint(block[3u + group * 8u + sub * 2u]) | (((high >> (sub * 2u + 1u)) & 1u) << 8u);
            signs = block[74u + group * 4u + sub];
        }
        for (uint part = 0u; part < 2u; ++part) {
            const uint word = lookup.word(indices[part], F == 18 || F == 21 ? 0u : part);
            const float4 values = float4(as_type<uchar4>(word));
            const float4 signed_values = select(values, -values,
                (uint4(signs >> (part * 4u)) & uint4(1u, 2u, 4u, 8u)) != 0u);
            const uint offset = k + section * 8u + part * 4u;
            sum += dot(signed_values, float4(input[offset], input[offset + 1u], input[offset + 2u], input[offset + 3u]));
        }
    }
    return scale * sum;
}

template<typename X> METAL_FUNC float gguf_q6_dot16(device const uchar* row, X input, uint k) {
    const auto block = reinterpret_cast<device const block_q6_K*>(row) + k / 256u;
    const uint p = k % 256u, quarter = (p % 128u) / 32u;
    auto low = block->ql + (p / 128u) * 64u + (quarter % 2u) * 32u + p % 32u;
    auto high = block->qh + (p / 128u) * 32u + p % 32u;
    float sum = 0.f;
    for (uint local = 0u; local < 16u; local += 4u) {
        const uint4 lo(low[local], low[local + 1u], low[local + 2u], low[local + 3u]);
        const uint4 hi(high[local], high[local + 1u], high[local + 2u], high[local + 3u]);
        const uint4 q = ((lo >> (4u * (quarter / 2u))) & 15u) | (((hi >> (2u * quarter)) & 3u) << 4u);
        const float4 x(input[k + local], input[k + local + 1u], input[k + local + 2u], input[k + local + 3u]);
        sum += dot(float4(q) - 32.f, x);
    }
    return (float(block->d) * float(block->scales[p / 16u])) * sum;
}

template<int F, typename X> METAL_FUNC float gguf_dot32(device const uchar* row, X input, uint k) {
    const uint group = (k % 256u) / 32u;
    device const uchar* scales;
    device const uchar* codes;
    device const uchar* high = nullptr;
    float scale, minimum;
    if constexpr (F == 12) {
        auto block = reinterpret_cast<device const block_q4_K*>(row) + k / 256u;
        scales = block->scales; codes = block->qs;
        scale = float(block->d); minimum = float(block->dmin);
    } else {
        auto block = reinterpret_cast<device const block_q5_K*>(row) + k / 256u;
        scales = block->scales; codes = block->qs; high = block->qh;
        scale = float(block->d); minimum = float(block->dmin);
    }
    const uint s = group < 4u ? scales[group] & 63u
        : (scales[group + 4u] & 15u) | ((scales[group - 4u] >> 6u) << 4u);
    const uint m = group < 4u ? scales[group + 4u] & 63u
        : (scales[group + 4u] >> 4u) | ((scales[group] >> 6u) << 4u);
    auto source = codes + group / 2u * 32u;
    float dot_value = 0.f, sum = 0.f;
    for (uint local = 0u; local < 32u; local += 4u) {
        uint4 q = (uint4(source[local], source[local + 1u], source[local + 2u], source[local + 3u])
            >> (4u * (group % 2u))) & 15u;
        if constexpr (F == 13)
            q |= ((uint4(high[local], high[local + 1u], high[local + 2u], high[local + 3u]) >> group) & 1u) << 4u;
        const float4 x(input[k + local], input[k + local + 1u], input[k + local + 2u], input[k + local + 3u]);
        dot_value += dot(float4(q), x);
        sum += x.x + x.y + x.z + x.w;
    }
    return (scale * float(s)) * dot_value - (minimum * float(m)) * sum;
}
