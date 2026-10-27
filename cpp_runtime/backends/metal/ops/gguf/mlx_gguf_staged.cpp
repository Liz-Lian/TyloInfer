#include "mlx_gguf_staged.h"
#include "mlx_gguf_prefill.h"
#include "mlx_cached_kernel.h"
#include "gguf_embedded.h"
#include "../../platform/metal_device_info.h"

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;

struct Layout { int payload, auxiliary, meta_groups, meta_halfs; };
Layout layout(int format) {
    switch (format) {
        case 12: return {16, 0, 8, 8};
        case 13: return {16, 4, 8, 8};
        case 14: return {16, 8, 8, 10};
        case 16: return {8, 0, 8, 1};
        case 17: return {8, 1, 8, 1};
        case 18: return {8, 4, 8, 1};
        case 21: return {16, 0, 8, 1};
        case 22: return {8, 2, 8, 1};
        case 20: return {16, 0, 1, 1};
        case 23: return {16, 0, 8, 4};
        case 42: return {8, 0, 2, 1};
        default: return {};
    }
}

const std::string& header() {
    static const auto source = std::string("#pragma clang fp reassociate(off)\n") + kGgufDecode + R"METAL(
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
template<int F> struct PlaneLayout {
    enum { payload = F == 16 || F == 17 || F == 18 || F == 22 || F == 42 ? 8 : 16,
        auxiliary = F == 14 ? 8 : F == 13 || F == 18 ? 4 : F == 17 ? 1 : F == 22 ? 2 : 0,
        meta_groups = F == 20 ? 1 : F == 42 ? 2 : 8,
        meta_halfs = F == 12 || F == 13 ? 8 : F == 14 ? 10 : F == 23 ? 4 : 1,
        table_words = F == 20 || F == 23 ? 256 : 1 };
};
template<int F>
void load_staged_lookup(threadgroup uint* table, uint local, uint threads) {
    if constexpr (F == 20 || F == 23) {
        for (uint i = local; i < 256u; i += threads)
            table[i] = as_type<uint>(half2(half(kvalues_iq4nl[i & 15u]), half(kvalues_iq4nl[i >> 4u])));
    }
}
template<int F>
auto staged_lookup(threadgroup uint* table) {
    if constexpr (F == 20 || F == 23) return GgufThreadgroupLookup<F>{table};
    else return GgufConstantLookup<F>{};
}
struct GroupPayload { uint4 a; uint b, extra; ushort scale; uint4 header; };
uint iq4_pair_indices(uint q) {
    const uint a = (q & 15u) | ((q >> 4u) & 240u), b = ((q >> 16u) & 15u) | ((q >> 20u) & 240u);
    const uint c = ((q >> 4u) & 15u) | ((q >> 8u) & 240u), d = ((q >> 20u) & 15u) | ((q >> 24u) & 240u);
    return a | (b << 8u) | (c << 16u) | (d << 24u);
}
template<int F>
void fetch_raw_group(thread GroupPayload& value, device const uchar* p0,
    uint row, uint groups, uint group, bool metadata) {
    constexpr uint block_bytes = F == 12 ? 144u : F == 13 ? 176u : F == 14 ? 210u : F == 23 ? 136u
        : F == 16 ? 66u : F == 17 ? 74u : F == 18 ? 98u : F == 21 ? 110u : 82u;
    const uint stride = F == 20 ? groups * 18u : F == 42 ? groups / 2u * 18u : groups / 8u * block_bytes;
    auto source = p0 + ulong(row) * stride;
    auto block = source + ulong(group / 8u) * block_bytes;
    const uint g = group % 8u;
    if constexpr (F == 16 || F == 17 || F == 18) {
        value.a.x = gguf_read_u32(block + 2u + g * 8u);
        value.a.y = gguf_read_u32(block + 6u + g * 8u);
        if constexpr (F == 17) value.b = block[66u + g];
        if constexpr (F == 18) value.b = gguf_read_u32(block + 66u + g * 4u);
    } else if constexpr (F == 22) {
        value.a.x = gguf_read_u32(block + 2u + g * 4u);
        value.a.y = gguf_read_u32(block + 34u + g * 4u);
        value.b = uint(block[66u + g]) | (uint(block[74u + g]) << 8u);
    } else if constexpr (F == 21) {
        const uint scale = (block[106u + g / 2u] >> (4u * (g % 2u))) & 15u;
        const uint bits = block[66u + g];
        for (uint c = 0u; c < 4u; ++c) {
            const uint signs = ((block[74u + g * 4u + c / 2u] >> (4u * (c % 2u))) & 15u)
                | (((block[76u + g * 4u + c / 2u] >> (4u * (c % 2u))) & 15u) << 4u);
            value.a[c] = uint(block[2u + g * 8u + c]) | (uint(block[6u + g * 8u + c]) << 8u)
                | (signs << 16u) | (((bits >> c) & 1u) << 24u) | (((bits >> (c + 4u)) & 1u) << 25u) | (scale << 26u);
        }
    } else if constexpr (F == 12 || F == 13 || F == 14) {
        device const uchar* codes;
        uint shift;
        if constexpr (F == 14) { codes = block + (g / 4u) * 64u + (g % 2u) * 32u; shift = 4u * ((g % 4u) / 2u); }
        else { codes = block + (F == 12 ? 16u : 48u) + (g / 2u) * 32u; shift = 4u * (g % 2u); }
        for (uint i = 0u; i < 4u; ++i) value.a[i] = ((gguf_read_u32(codes + i * 4u) >> shift) & 0x0f0f0f0fu)
            | (((gguf_read_u32(codes + 16u + i * 4u) >> shift) & 0x0f0f0f0fu) << 4u);
        if constexpr (F == 13) {
            value.b = 0u;
            for (uint i = 0u; i < 8u; ++i) {
                const uint q = (gguf_read_u32(block + 16u + i * 4u) >> g) & 0x01010101u;
                value.b |= ((q & 1u) | ((q >> 7u) & 2u) | ((q >> 14u) & 4u) | ((q >> 21u) & 8u)) << (i * 4u);
            }
        } else if constexpr (F == 14) {
            value.b = value.extra = 0u;
            for (uint i = 0u; i < 8u; ++i) {
                const uint q = (gguf_read_u32(block + 128u + (g / 4u) * 32u + i * 4u) >> (2u * (g % 4u))) & 0x03030303u;
                const uint bits = (q & 3u) | ((q >> 6u) & 12u) | ((q >> 12u) & 48u) | ((q >> 18u) & 192u);
                if (i < 4u) value.b |= bits << (i * 8u); else value.extra |= bits << ((i - 4u) * 8u);
            }
        }
    } else if constexpr (F == 23) {
        for (uint i = 0u; i < 4u; ++i) value.a[i] = iq4_pair_indices(gguf_read_u32(block + 8u + g * 16u + i * 4u));
    } else if constexpr (F == 20) {
        block = source + ulong(group) * 18u;
        for (uint i = 0u; i < 4u; ++i) value.a[i] = iq4_pair_indices(gguf_read_u32(block + 2u + i * 4u));
    } else if constexpr (F == 42) {
        block = source + ulong(group / 2u) * 18u;
        value.a.x = gguf_read_u32(block + 2u + (group % 2u) * 8u);
        value.a.y = gguf_read_u32(block + 6u + (group % 2u) * 8u);
    }
    if (metadata) {
        const auto meta = block + (F == 14 ? 192u : 0u);
        value.scale = *reinterpret_cast<device const ushort*>(meta + (F == 14 ? 16u : 0u));
        if constexpr (PlaneLayout<F>::meta_halfs > 1) {
#pragma clang loop unroll(full)
            for (uint i = 0; i < (F == 23 ? 2u : 4u); ++i)
                value.header[i] = gguf_read_u32(meta + i * 4u);
        }
    }
}
template<int F, class Auxiliary, class Scales>
void fetch_group(thread GroupPayload& value, device const uchar* p0, Auxiliary p1,
    Scales scales, uint row, uint groups, uint group, bool metadata) {
    const ulong at = (ulong(row / 64u) * groups + group) * 64u + row % 64u;
    const auto words = reinterpret_cast<device const uint*>(p0 + at * PlaneLayout<F>::payload);
    if constexpr (PlaneLayout<F>::payload == 16) value.a = *reinterpret_cast<device const uint4*>(words);
    else value.a.xy = *reinterpret_cast<device const uint2*>(words);
    if constexpr (PlaneLayout<F>::auxiliary == 1) value.b = p1[at];
    if constexpr (PlaneLayout<F>::auxiliary == 2) value.b = uint(p1[at * 2u]) | (uint(p1[at * 2u + 1u]) << 8u);
    if constexpr (PlaneLayout<F>::auxiliary >= 4) value.b = uint(p1[at * PlaneLayout<F>::auxiliary])
        | (uint(p1[at * PlaneLayout<F>::auxiliary + 1u]) << 8u)
        | (uint(p1[at * PlaneLayout<F>::auxiliary + 2u]) << 16u) | (uint(p1[at * PlaneLayout<F>::auxiliary + 3u]) << 24u);
    if constexpr (PlaneLayout<F>::auxiliary == 8) value.extra = uint(p1[at * 8u + 4u]) | (uint(p1[at * 8u + 5u]) << 8u)
        | (uint(p1[at * 8u + 6u]) << 16u) | (uint(p1[at * 8u + 7u]) << 24u);
    if (metadata) {
        const ulong offset = ((ulong(row / 64u) * (groups / PlaneLayout<F>::meta_groups)
            + group / PlaneLayout<F>::meta_groups) * 64u + row % 64u) * PlaneLayout<F>::meta_halfs;
        value.scale = scales[offset + (F == 14 ? 8u : 0u)];
        if constexpr (PlaneLayout<F>::meta_halfs > 1) {
#pragma clang loop unroll(full)
            for (uint i = 0; i < (F == 23 ? 2u : 4u); ++i)
                value.header[i] = uint(scales[offset + i * 2u]) | (uint(scales[offset + i * 2u + 1u]) << 16u);
        }
    }
}
template<typename O = half>
vec<O, 4> plane_signed(uint word, uint signs, float scale, uint shift) {
    auto values = float4(as_type<uchar4>(word)) * scale;
    return vec<O, 4>(select(values, -values, (uint4(signs) & (uint4(1u, 2u, 4u, 8u) << shift)) != 0u));
}
template<int Zero, typename O = half>
vec<O, 4> plane_linear(uint4 codes, float scale, float bias = 0.f) {
    const uint2 pairs(codes.x | (codes.y << 16u), codes.z | (codes.w << 16u));
    const half4 values = half4(as_type<half2>(pairs.x | 0x64006400u), as_type<half2>(pairs.y | 0x64006400u))
        - half4(half(1024 + Zero));
    if constexpr (Zero != 0) return vec<O, 4>(float4(values) * scale);
    else return vec<O, 4>(fma(float4(values), float4(scale), float4(bias)));
}
template<int F, typename O, class Lookup, class Sink>
void decode_group_chunks(GroupPayload value, Lookup lookup, uint group, Sink sink) {
    const float d = float(as_type<half>(value.scale));
#pragma clang loop unroll(full)
    for (uint c = 0; c < 4u; ++c) {
        vec<O, 4> low, high;
        if constexpr (F == 16 || F == 18) {
            const uint aux = F == 16 ? value.a.y : value.b;
            const uint sub = c / 2u, shift = (c % 2u) * 4u;
            uint a, b;
            if constexpr (F == 16) {
                a = lookup.word((value.a.x >> (8u * sub)) & 255u, c % 2u);
                b = lookup.word((value.a.x >> (8u * (sub + 2u))) & 255u, c % 2u);
            } else {
                a = lookup.word((value.a.x >> (8u * c)) & 255u, 0u);
                b = lookup.word((value.a.y >> (8u * c)) & 255u, 0u);
            }
            const float scale = d * (.5f + float(aux >> 28u)) * (F == 16 ? .25f : .5f);
            low = plane_signed<O>(a, lookup.sign((aux >> (7u * sub)) & 127u), scale, shift);
            high = plane_signed<O>(b, lookup.sign((aux >> (7u * (sub + 2u))) & 127u), scale, shift);
        } else if constexpr (F == 17 || F == 22) {
            const uint sub = c / 2u, shift = (c % 2u) * 4u;
            uint a, b, sa, sb, scales;
            if constexpr (F == 17) {
                const uint ca = (value.a.x >> (16u * sub)) & 65535u;
                const uint cb = (value.a.y >> (16u * sub)) & 65535u;
                a = ca & 511u; b = cb & 511u; sa = lookup.sign(ca >> 9u); sb = lookup.sign(cb >> 9u);
                scales = value.b;
            } else {
                a = ((value.a.x >> (8u * sub)) & 255u) | (((value.b >> (2u * sub)) & 3u) << 8u);
                b = ((value.a.x >> (8u * (sub + 2u))) & 255u) | (((value.b >> (2u * (sub + 2u))) & 3u) << 8u);
                sa = (value.a.y >> (8u * sub)) & 255u; sb = (value.a.y >> (8u * (sub + 2u))) & 255u;
                scales = value.b >> 8u;
            }
            low = plane_signed<O>(lookup.word(a, c % 2u), sa, d * (.5f + float(scales & 15u)) * .25f, shift);
            high = plane_signed<O>(lookup.word(b, c % 2u), sb, d * (.5f + float(scales >> 4u)) * .25f, shift);
        } else if constexpr (F == 21) {
            const uint q = value.a[c];
            const float scale = d * (1u + 2u * ((q >> 26u) & 15u));
            low = plane_signed<O>(lookup.word((q & 255u) | ((q >> 16u) & 256u), 0u), q >> 16u, scale, 0u);
            high = plane_signed<O>(lookup.word(((q >> 8u) & 255u) | ((q >> 17u) & 256u), 0u), q >> 16u, scale, 4u);
        } else if constexpr (F == 20 || F == 23) {
            const auto indices = as_type<uchar4>(value.a[c]);
            const half4 a(as_type<half2>(lookup.table[indices.x]), as_type<half2>(lookup.table[indices.y]));
            const half4 b(as_type<half2>(lookup.table[indices.z]), as_type<half2>(lookup.table[indices.w]));
            float scale = d;
            if constexpr (F == 23) scale *= float(int(((value.header.y >> (4u * (group % 8u))) & 15u)
                | (((value.header.x >> (16u + 2u * (group % 8u))) & 3u) << 4u)) - 32);
            if constexpr (F == 20 && sizeof(O) == 2) { low = a * half(scale); high = b * half(scale); }
            else { low = vec<O, 4>(float4(a) * scale); high = vec<O, 4>(float4(b) * scale); }
        } else if constexpr (F == 12 || F == 13 || F == 14) {
            const auto bytes = as_type<uchar4>(value.a[c]);
            uint4 a = uint4(bytes) & 15u, b = uint4(bytes) >> 4u;
            const uint g = group % 8u;
            if constexpr (F == 12 || F == 13) {
                if constexpr (F == 13) {
                    a |= ((uint4(value.b) >> (uint4(0u, 1u, 2u, 3u) + 4u * c)) & 1u) << 4u;
                    b |= ((uint4(value.b) >> (uint4(16u, 17u, 18u, 19u) + 4u * c)) & 1u) << 4u;
                }
                const uint sh = 8u * (g % 4u), extra = value.header.w >> sh;
                const uint sc = g < 4u ? (value.header.y >> sh) & 63u
                    : (extra & 15u) | (((value.header.y >> (sh + 6u)) & 3u) << 4u);
                const uint bias = g < 4u ? (value.header.z >> sh) & 63u
                    : ((extra >> 4u) & 15u) | (((value.header.z >> (sh + 6u)) & 3u) << 4u);
                const float scale = d * float(sc);
                const float offset = float(as_type<half>(ushort(value.header.x >> 16u))) * float(bias);
                low = plane_linear<0, O>(a, scale, -offset); high = plane_linear<0, O>(b, scale, -offset);
            } else {
                a |= ((uint4(value.b) >> (uint4(0u, 2u, 4u, 6u) + 8u * c)) & 3u) << 4u;
                b |= ((uint4(value.extra) >> (uint4(0u, 2u, 4u, 6u) + 8u * c)) & 3u) << 4u;
                const uint scales = g < 2u ? value.header.x : g < 4u ? value.header.y : g < 6u ? value.header.z : value.header.w;
                const uint shift = 16u * (g % 2u);
                low = plane_linear<32, O>(a, d * float(as_type<char>(uchar(scales >> shift))));
                high = plane_linear<32, O>(b, d * float(as_type<char>(uchar(scales >> (shift + 8u)))));
            }
        } else if constexpr (F == 42) {
            const uint a = (value.a.x >> (8u * c)) & 255u, b = (value.a.y >> (8u * c)) & 255u;
            low = plane_linear<1, O>((uint4(a) >> uint4(0u, 2u, 4u, 6u)) & 3u, d);
            high = plane_linear<1, O>((uint4(b) >> uint4(0u, 2u, 4u, 6u)) & 3u, d);
        }
        sink(c, low, high);
    }
}
template<int F, class Lookup>
void decode_group(GroupPayload value, threadgroup half* output, Lookup lookup, uint group) {
    decode_group_chunks<F, half>(value, lookup, group, [&](uint c, half4 low, half4 high) __attribute__((always_inline)) {
        *reinterpret_cast<threadgroup half4*>(output + c * 4u) = low;
        *reinterpret_cast<threadgroup half4*>(output + 16u + c * 4u) = high;
    });
}
template<int F, int Rows, bool Packed, class Auxiliary, class Scales, class Acc>
void staged_dot(device const half* x, device const uchar* p0, Auxiliary p1, Scales meta,
    uint row, uint k, uint padded, uint first_token,
    threadgroup half* stages, threadgroup uint* table, thread Acc& sum) {
    constexpr auto description = matmul2d_descriptor(Rows, 32, 32, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<description, execution_simdgroup> operation;
    tensor<device half, dextents<int, 2>, tensor_inline> input(
        const_cast<device half*>(x) + ulong(first_token) * k, dextents<int, 2>(k, Rows));
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> b0(stages, dextents<int, 2>(32, 32));
    tensor<threadgroup half, dextents<int, 2>, tensor_inline> b1(stages + 1024, dextents<int, 2>(32, 32));
    auto right0 = b0.template slice<32, 32>(0, 0), right1 = b1.template slice<32, 32>(0, 0);
    const uint groups = k / 32u;
    GroupPayload payload;
    if constexpr (Packed) fetch_group<F>(payload, p0, p1, meta, row, groups, 0u, true);
    else fetch_raw_group<F>(payload, p0, row, groups, 0u, true);
    for (uint g = 0; g < groups; ++g) {
        decode_group<F>(payload, stages + (g % 2u) * 1024u + uint(__metal_get_thread_index_in_simdgroup(ushort())) * 32u,
            staged_lookup<F>(table), g);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (g + 1u < groups) {
            const bool metadata = (g + 1u) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups;
            if constexpr (Packed) fetch_group<F>(payload, p0, p1, meta, row, groups, g + 1u, metadata);
            else fetch_raw_group<F>(payload, p0, row, groups, g + 1u, metadata);
        }
        auto left = input.template slice<32, Rows>(g * 32u, 0);
        if (g % 2u) operation.run(left, right1, sum); else operation.run(left, right0, sum);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
template<int F, bool Packed, int L = 1, class Auxiliary, class Scales>
inline void column_dot(device const half* x, uint rows, uint k, device const uchar* p0,
    Auxiliary p1, Scales meta, uint column, threadgroup uint* table, thread float (&sums)[2]) {
    const uint groups = k / 32u;
    const uint part = uint(__metal_get_thread_index_in_simdgroup(ushort())) % L;
    GroupPayload payload;
    if (part < groups) {
        if constexpr (Packed) fetch_group<F>(payload, p0, p1, meta, column, groups, part, true);
        else fetch_raw_group<F>(payload, p0, column, groups, part, true);
    }
    sums[0] = sums[1] = 0.f;
    for (uint g = part; g < groups; g += L) {
        half4 values[8];
        decode_group_chunks<F, half>(payload, staged_lookup<F>(table), g,
            [&](uint c, half4 low, half4 high) __attribute__((always_inline)) { values[c] = low; values[c + 4u] = high; });
        if (g + L < groups) {
            const bool metadata = (g + L) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups;
            if constexpr (Packed) fetch_group<F>(payload, p0, p1, meta, column, groups, g + L, metadata);
            else fetch_raw_group<F>(payload, p0, column, groups, g + L, metadata);
        }
#pragma clang loop unroll(full)
        for (uint r = 0u; r < 2u; ++r) {
            if (r >= rows) break;
            auto input = reinterpret_cast<device const uint4*>(x + ulong(r) * k + g * 32u);
            float sum = 0.f;
#pragma clang loop unroll(full)
            for (uint i = 0u; i < 8u; ++i) {
                const uint4 raw = input[i / 2u];
                const float4 a = float4(as_type<half4>(i % 2u ? raw.zw : raw.xy)), b = float4(values[i]);
                sum = sum + b.x * a.x;
                sum = sum + b.y * a.y;
                sum = sum + b.z * a.z;
                sum = sum + b.w * a.w;
            }
            sums[r] += sum;
        }
    }
#pragma clang loop unroll(full)
    for (uint offset = L / 2u; offset > 0u; offset /= 2u) {
        sums[0] += simd_shuffle_down(sums[0], offset);
        sums[1] += simd_shuffle_down(sums[1], offset);
    }
}
template<int FG, int FU, int FGS, int FUS, bool PG, bool PU, int L, typename T,
    class GA, class GM, class UA, class UM, class SGA, class SGM, class SUA, class SUM>
__attribute__((noinline)) void gate_up_columns(device const half* x, uint live, uint k, uint n, uint e,
    uint n0, uint lane, uint sg, bool shared, bool same_table,
    device const uchar* gate, GA gate_aux, GM gate_meta, device const uchar* up, UA up_aux, UM up_meta,
    device const uchar* shared_gate, SGA shared_gate_aux, SGM shared_gate_meta,
    device const uchar* shared_up, SUA shared_up_aux, SUM shared_up_meta,
    threadgroup uint* gate_table, threadgroup uint* up_table, threadgroup T* gates, device T* y) {
    float sums[2];
    const uint column = (shared ? 0u : e * n) + n0 + lane / L;
    if (shared) {
        if (sg == 0u) column_dot<FGS, true, L>(x, live, k, shared_gate, shared_gate_aux, shared_gate_meta, column, gate_table, sums);
        else column_dot<FUS, true, L>(x, live, k, shared_up, shared_up_aux, shared_up_meta, column, same_table ? gate_table : up_table, sums);
    } else {
        if (sg == 0u) column_dot<FG, PG, L>(x, live, k, gate, gate_aux, gate_meta, column, gate_table, sums);
        else column_dot<FU, PU, L>(x, live, k, up, up_aux, up_meta, column, same_table ? gate_table : up_table, sums);
    }
    if (sg == 0u && lane % L == 0u) for (uint r = 0u; r < live; ++r) gates[r * 32u + lane / L] = T(sums[r]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 1u && lane % L == 0u) for (uint r = 0u; r < live; ++r) {
        const float g = float(gates[r * 32u + lane / L]), u = float(T(sums[r]));
        y[ulong(r) * n + n0 + lane / L] = T((g / (1.f + exp(-g))) * u);
    }
}
)METAL";
    return source;
}

const fast::CustomKernelFunction& pack_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_pack_planes",
        {"packed", "anchor", "params"}, {"payload", "auxiliary", "metadata"}, R"METAL(
uint at = thread_position_in_grid.x, groups = params[0] / 32u, rows = params[1];
if (at >= groups * rows) return;
uint group = at % groups, row = at / groups;
auto source = packed + ulong(row) * params[2];
ulong position = (ulong(row / 64u) * groups + group) * 64u + row % 64u;
auto output = reinterpret_cast<device uint*>(payload + position * PlaneLayout<F>::payload);
auto block = source + ulong(group / 8u) * (F == 12 ? 144u : F == 13 ? 176u : F == 14 ? 210u : F == 23 ? 136u
    : F == 16 ? 66u : F == 17 ? 74u : F == 18 ? 98u : F == 21 ? 110u : 82u);
uint g = group % 8u;
if constexpr (F == 16 || F == 17) {
    output[0] = gguf_read_u32(block + 2u + g * 8u); output[1] = gguf_read_u32(block + 6u + g * 8u);
    if constexpr (F == 17) auxiliary[position] = block[66u + g];
} else if constexpr (F == 18) {
    output[0] = gguf_read_u32(block + 2u + g * 8u); output[1] = gguf_read_u32(block + 6u + g * 8u);
    reinterpret_cast<device uint*>(auxiliary)[position] = gguf_read_u32(block + 66u + g * 4u);
} else if constexpr (F == 22) {
    output[0] = gguf_read_u32(block + 2u + g * 4u); output[1] = gguf_read_u32(block + 34u + g * 4u);
    reinterpret_cast<device ushort*>(auxiliary)[position] = ushort(block[66u + g]) | (ushort(block[74u + g]) << 8u);
} else if constexpr (F == 21) {
    const uint scales = (block[106u + g / 2u] >> (4u * (g % 2u))) & 15u;
    const uint bits = block[66u + g];
    for (uint c = 0; c < 4u; ++c) {
        const uint signs = ((block[74u + g * 4u + c / 2u] >> (4u * (c % 2u))) & 15u)
            | (((block[76u + g * 4u + c / 2u] >> (4u * (c % 2u))) & 15u) << 4u);
        output[c] = uint(block[2u + g * 8u + c]) | (uint(block[6u + g * 8u + c]) << 8u)
            | (signs << 16u) | (((bits >> c) & 1u) << 24u) | (((bits >> (c + 4u)) & 1u) << 25u) | (scales << 26u);
    }
} else if constexpr (F == 12 || F == 13 || F == 14) {
    device const uchar* codes;
    uint shift;
    if constexpr (F == 14) { codes = block + (g / 4u) * 64u + (g % 2u) * 32u; shift = 4u * ((g % 4u) / 2u); }
    else { codes = block + (F == 12 ? 16u : 48u) + (g / 2u) * 32u; shift = 4u * (g % 2u); }
    for (uint i = 0; i < 4u; ++i) output[i] = ((gguf_read_u32(codes + i * 4u) >> shift) & 0x0f0f0f0fu)
        | (((gguf_read_u32(codes + 16u + i * 4u) >> shift) & 0x0f0f0f0fu) << 4u);
    if constexpr (F == 13) {
        uint bits = 0u;
        for (uint i = 0; i < 8u; ++i) {
            const uint q = (gguf_read_u32(block + 16u + i * 4u) >> g) & 0x01010101u;
            bits |= ((q & 1u) | ((q >> 7u) & 2u) | ((q >> 14u) & 4u) | ((q >> 21u) & 8u)) << (i * 4u);
        }
        reinterpret_cast<device uint*>(auxiliary)[position] = bits;
    } else if constexpr (F == 14) {
        for (uint i = 0; i < 8u; ++i) {
            const uint q = (gguf_read_u32(block + 128u + (g / 4u) * 32u + i * 4u) >> (2u * (g % 4u))) & 0x03030303u;
            auxiliary[position * 8u + i] = uchar((q & 3u) | ((q >> 6u) & 12u) | ((q >> 12u) & 48u) | ((q >> 18u) & 192u));
        }
    }
} else if constexpr (F == 23) {
    for (uint i = 0; i < 4u; ++i) output[i] = iq4_pair_indices(gguf_read_u32(block + 8u + g * 16u + i * 4u));
} else if constexpr (F == 20) {
    block = source + ulong(group) * 18u;
    for (uint c = 0; c < 4u; ++c) output[c] = iq4_pair_indices(gguf_read_u32(block + 2u + c * 4u));
} else if constexpr (F == 42) {
    block = source + ulong(group / 2u) * 18u;
    output[0] = gguf_read_u32(block + 2u + (group % 2u) * 8u);
    output[1] = gguf_read_u32(block + 6u + (group % 2u) * 8u);
}
if (group % PlaneLayout<F>::meta_groups == 0u) {
    const ulong offset = ((ulong(row / 64u) * (groups / PlaneLayout<F>::meta_groups)
        + group / PlaneLayout<F>::meta_groups) * 64u + row % 64u) * PlaneLayout<F>::meta_halfs;
    for (uint i = 0; i < PlaneLayout<F>::meta_halfs; ++i)
        metadata[offset + i] = F == 14 && i == 9u ? ushort(0u)
            : *reinterpret_cast<device const ushort*>(block + (F == 14 ? 192u : 0u) + i * 2u);
}
)METAL", header());
    return kernel;
}

std::vector<array> pack(const array& bytes, const array& anchor, int format, int rows, int k, int stride) {
    const auto fields = layout(format);
    const auto groups = std::size_t(rows) * (k / 32);
    const auto payload = groups * fields.payload, auxiliary = groups * fields.auxiliary;
    const auto metadata = groups / fields.meta_groups * fields.meta_halfs;
    if (groups > INT_MAX - 127 || payload > INT_MAX || auxiliary > INT_MAX || metadata > INT_MAX)
        throw std::overflow_error("GGUF prefill plane extent is too large");
    return pack_kernel()({bytes, anchor, array({uint32_t(k), uint32_t(rows), uint32_t(stride)})},
        {{int(payload)}, {int(std::max<std::size_t>(1, auxiliary))}, {int(metadata)}},
        {uint8, uint8, uint16}, {int((groups + 127) / 128 * 128), 1, 1}, {128, 1, 1},
        {{"F", format}}, std::nullopt, false, {});
}

const fast::CustomKernelFunction& dequantize_planes_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_planes_dequantize",
        {"p0", "p1", "meta", "params"}, {"out"}, R"METAL(
uint at = thread_position_in_grid.x, k = params[0], rows = params[1], groups = k / 32u;
threadgroup uint table[PlaneLayout<F>::table_words];
load_staged_lookup<F>(table, thread_index_in_threadgroup, 128u);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (at >= rows * groups) return;
uint row = at / groups, group = at % groups;
GroupPayload payload;
fetch_group<F>(payload, p0, p1, meta, row, groups, group, true);
decode_group_chunks<F, float>(payload, staged_lookup<F>(table), group,
    [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
        for (uint i = 0u; i < 4u; ++i) {
            out[ulong(at) * 32u + c * 4u + i] = T(low[i]);
            out[ulong(at) * 32u + 16u + c * 4u + i] = T(high[i]);
        }
    });
)METAL", header());
    return kernel;
}

const fast::CustomKernelFunction& embedding_planes_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_planes_embedding",
        {"p0", "p1", "meta", "ids", "params"}, {"out"}, R"METAL(
uint at = thread_position_in_grid.x, groups = params[0] / 32u;
threadgroup uint table[PlaneLayout<F>::table_words];
load_staged_lookup<F>(table, thread_index_in_threadgroup, 128u);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (at >= params[2] * groups) return;
int row = ids[at / groups];
uint group = at % groups;
GroupPayload payload;
bool valid = row >= 0 && uint(row) < params[1];
fetch_group<F>(payload, p0, p1, meta, valid ? uint(row) : 0u, groups, group, true);
decode_group_chunks<F, float>(payload, staged_lookup<F>(table), group,
    [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
        for (uint i = 0u; i < 4u; ++i) {
            out[ulong(at) * 32u + c * 4u + i] = valid ? T(low[i]) : T(0);
            out[ulong(at) * 32u + 16u + c * 4u + i] = valid ? T(high[i]) : T(0);
        }
    });
)METAL", header());
    return kernel;
}

const fast::CustomKernelFunction& gate_up_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_staged_gate_up",
        {"gate", "up", "x", "descriptors", "count", "params", "gate_aux", "gate_meta", "up_aux", "up_meta",
            "shared_gate", "shared_up", "shared_gate_aux", "shared_gate_meta", "shared_up_aux", "shared_up_meta"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.y;
if (tile >= uint(count[0])) return;
uint e = MAX_ROWS == 1 ? (tile >= params[4] ? params[3] : uint(descriptors[tile])) : uint(descriptors[tile * 2u]);
bool shared = HAS_SHARED && (MAX_ROWS == 1 ? tile >= params[4] : e == params[3]);
uint k = params[0], n = params[1], row0 = tile * MAX_ROWS, n0 = threadgroup_position_in_grid.x * (32u / COLUMN_LANES);
uint sg = simdgroup_index_in_threadgroup;
if constexpr (MAX_ROWS == 1) if (e >= params[3] && !shared) {
    const uint lane = thread_index_in_simdgroup;
    if (sg == 1u && lane % COLUMN_LANES == 0u) y[ulong(tile) * n + n0 + lane / COLUMN_LANES] = T(0);
    return;
}
threadgroup half stages[MAX_ROWS == 1 ? 32 : COLUMN_ONLY ? 128 : 4096];
threadgroup uint gate_table[PlaneLayout<FG>::table_words > PlaneLayout<FGS>::table_words ? PlaneLayout<FG>::table_words : PlaneLayout<FGS>::table_words];
threadgroup uint up_table[FG == FU && FGS == FUS ? 1u : (PlaneLayout<FU>::table_words > PlaneLayout<FUS>::table_words ? PlaneLayout<FU>::table_words : PlaneLayout<FUS>::table_words)];
bool same_table = shared ? FGS == FUS : FG == FU;
if (shared) {
    load_staged_lookup<FGS>(gate_table, thread_index_in_threadgroup, 64u);
    if (!same_table) load_staged_lookup<FUS>(up_table, thread_index_in_threadgroup, 64u);
} else {
    load_staged_lookup<FG>(gate_table, thread_index_in_threadgroup, 64u);
    if (!same_table) load_staged_lookup<FU>(up_table, thread_index_in_threadgroup, 64u);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
uint live = MAX_ROWS == 1 ? 1u : uint(descriptors[tile * 2u + 1u]);
if (MAX_ROWS == 1 || (live <= 2u && (tile == 0u || descriptors[(tile - 1u) * 2u] != int(e)))) {
    gate_up_columns<FG, FU, FGS, FUS, PG, PU, COLUMN_LANES>(x + (MAX_ROWS == 1 ? 0ul : ulong(row0) * k), live, k, n, e, n0,
        thread_index_in_simdgroup, sg, shared, same_table, gate, gate_aux, gate_meta, up, up_aux, up_meta,
        shared_gate, shared_gate_aux, shared_gate_meta, shared_up, shared_up_aux, shared_up_meta,
        gate_table, up_table, reinterpret_cast<threadgroup T*>(stages), y + ulong(row0) * n);
    return;
}
if constexpr (MAX_ROWS > 1 && !COLUMN_ONLY) {
if (threadgroup_position_in_grid.x % COLUMN_LANES != 0u) return;
auto run = [&](auto rows) __attribute__((always_inline)) {
constexpr ushort R = decltype(rows)::value;
constexpr auto description = matmul2d_descriptor(R, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> a(const_cast<device half*>(x) + ulong(row0) * k, dextents<int, 2>(k, R));
tensor<threadgroup half, dextents<int, 2>, tensor_inline> b(stages + sg * 2048u, dextents<int, 2>(32, 32));
auto left = a.template slice<32, R>(0, 0);
auto right = b.template slice<32, 32>(0, 0);
auto sum = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
const uint row = (shared ? 0u : e * n) + n0 + thread_index_in_simdgroup;
if (shared) {
    if (sg == 0u) staged_dot<FGS, R, true>(x, shared_gate, shared_gate_aux, shared_gate_meta, row, k, params[2], row0, stages, gate_table, sum);
    else staged_dot<FUS, R, true>(x, shared_up, shared_up_aux, shared_up_meta, row, k, params[2], row0, stages + 2048,
        same_table ? gate_table : up_table, sum);
} else {
    if (sg == 0u) staged_dot<FG, R, PG>(x, gate, gate_aux, gate_meta, row, k, params[2], row0, stages, gate_table, sum);
    else staged_dot<FU, R, PU>(x, up, up_aux, up_meta, row, k, params[2], row0, stages + 2048,
        same_table ? gate_table : up_table, sum);
}
auto gate_values = reinterpret_cast<threadgroup T*>(stages);
if (sg == 0u) for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
    auto at = sum.get_multidimensional_index(i); gate_values[at[1] * 32u + at[0]] = T(sum[i]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 1u) for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
    auto at = sum.get_multidimensional_index(i);
    float g = float(gate_values[at[1] * 32u + at[0]]), u = float(T(sum[i]));
    y[ulong(row0 + at[1]) * n + n0 + at[0]] = T((g / (1.f + exp(-g))) * u);
}
};
if constexpr (MAX_ROWS == 8) run(integral_constant<ushort, 8>{});
else {
    if (live <= 8u) run(integral_constant<ushort, 8>{});
    else if (live <= 16u) run(integral_constant<ushort, 16>{});
    else run(integral_constant<ushort, MAX_ROWS>{});
}
}
)METAL", header());
    return kernel;
}

const fast::CustomKernelFunction& projection_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_planes_projection",
        {"p0", "p1", "meta", "x", "params"}, {"out"}, R"METAL(
const uint lane = thread_index_in_simdgroup, part = lane % L;
const uint column = (threadgroup_position_in_grid.x * 2u + simdgroup_index_in_threadgroup) * (32u / L) + lane / L;
const uint k = params[0], n = params[1], m = params[2], groups = k / 32u;
threadgroup uint table[PlaneLayout<F>::table_words];
load_staged_lookup<F>(table, thread_index_in_threadgroup, 64u);
threadgroup_barrier(mem_flags::mem_threadgroup);
if (column >= n) return;
float sums[MR] = {};
GroupPayload payload;
if (part < groups) fetch_group<F>(payload, p0, p1, meta, column, groups, part, true);
for (uint g = part; g < groups; g += L) {
    decode_group_chunks<F, float>(payload, staged_lookup<F>(table), g,
        [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
#pragma clang loop unroll(full)
            for (uint row = 0u; row < uint(MR); ++row) {
                if (row >= m) break;
                auto input = x + ulong(row) * k + g * 32u + c * 4u;
                const float4 a(input[0], input[1], input[2], input[3]);
                const float4 b(input[16], input[17], input[18], input[19]);
                sums[row] += dot(low, a) + dot(high, b);
            }
        });
    if (g + L < groups) fetch_group<F>(payload, p0, p1, meta, column, groups, g + L,
        (g + L) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups);
}
#pragma clang loop unroll(full)
for (uint row = 0u; row < uint(MR); ++row) {
    if (row >= m) break;
    float sum = sums[row];
#pragma clang loop unroll(full)
    for (uint offset = L / 2u; offset > 0u; offset /= 2u) sum += simd_shuffle_down(sum, offset);
    if (part == 0u) out[ulong(row) * n + column] = T(sum);
}
)METAL", header());
    return kernel;
}

const fast::CustomKernelFunction& projection_many_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_planes_projection_many",
        {"p00", "p01", "m0", "p10", "p11", "m1", "p20", "p21", "m2", "p30", "p31", "m3", "x", "params"},
        {"out0", "out1", "out2", "out3"}, R"METAL(
uint tile = threadgroup_position_in_grid.x;
const uint t0 = params[2] / (64u / L0), t1 = params[3] / (64u / L1), t2 = params[4] / (64u / L2);
threadgroup uint table[256];
const uint tid = thread_position_in_threadgroup.x, lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
if (tile < t0) {
    load_staged_lookup<F0>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    planes_project<F0, L0, MR>(p00, p01, m0, x, out0, table, params[0], params[2], params[1], tile, lane, sg);
} else if ((tile -= t0) < t1) {
    load_staged_lookup<F1>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    planes_project<F1, L1, MR>(p10, p11, m1, x, out1, table, params[0], params[3], params[1], tile, lane, sg);
} else if ((tile -= t1) < t2) {
    load_staged_lookup<F2>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    planes_project<F2, L2, MR>(p20, p21, m2, x, out2, table, params[0], params[4], params[1], tile, lane, sg);
} else {
    tile -= t2;
    load_staged_lookup<F3>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    planes_project<F3, L3, MR>(p30, p31, m3, x, out3, table, params[0], params[5], params[1], tile, lane, sg);
}
)METAL", header() + R"METAL(
template<int F, int L, int MR, typename T, class Auxiliary, class Scales>
void planes_project(device const uchar* p0, Auxiliary p1, Scales meta,
    device const T* x, device T* out, threadgroup uint* table, uint k, uint n, uint m, uint tile, uint lane, uint sg) {
    const uint part = lane % L, column = (tile * 2u + sg) * (32u / L) + lane / L, groups = k / 32u;
    float sums[MR] = {};
    GroupPayload payload;
    if (part < groups) fetch_group<F>(payload, p0, p1, meta, column, groups, part, true);
    for (uint g = part; g < groups; g += L) {
        decode_group_chunks<F, float>(payload, staged_lookup<F>(table), g,
            [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
#pragma clang loop unroll(full)
                for (uint row = 0u; row < uint(MR); ++row) {
                    if (row >= m) break;
                    auto input = x + ulong(row) * k + g * 32u + c * 4u;
                    const float4 a(input[0], input[1], input[2], input[3]);
                    const float4 b(input[16], input[17], input[18], input[19]);
                    sums[row] += dot(low, a) + dot(high, b);
                }
            });
        if (g + L < groups) fetch_group<F>(payload, p0, p1, meta, column, groups, g + L,
            (g + L) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups);
    }
#pragma clang loop unroll(full)
    for (uint row = 0u; row < uint(MR); ++row) {
        if (row >= m) break;
        float sum = sums[row];
#pragma clang loop unroll(full)
        for (uint offset = L / 2u; offset > 0u; offset /= 2u) sum += simd_shuffle_down(sum, offset);
        if (part == 0u) out[ulong(row) * n + column] = T(sum);
    }
}
)METAL");
    return kernel;
}

const fast::CustomKernelFunction& down_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_staged_down",
        {"packed", "x", "descriptors", "count", "params", "aux", "meta", "shared_packed", "shared_aux", "shared_meta"}, {"y"}, R"METAL(
uint tile = threadgroup_position_in_grid.y;
if (tile >= uint(count[0])) return;
uint e = MAX_ROWS == 1 ? (tile >= params[4] ? params[3] : uint(descriptors[tile])) : uint(descriptors[tile * 2u]);
bool shared = HAS_SHARED && (MAX_ROWS == 1 ? tile >= params[4] : e == params[3]);
uint k = params[0], n = params[1], row0 = tile * MAX_ROWS;
uint input_row = BROADCAST ? 0u : row0;
uint sg = simdgroup_index_in_threadgroup, n0 = (threadgroup_position_in_grid.x * 2u + sg) * 32u;
if constexpr (MAX_ROWS == 1) if (e >= params[3] && !shared) {
    y[ulong(tile) * n + n0 + thread_index_in_simdgroup] = T(0);
    return;
}
threadgroup half stages[MAX_ROWS == 1 || COLUMN_ONLY ? 1 : 4096];
threadgroup uint table[PlaneLayout<F>::table_words > PlaneLayout<FS>::table_words ? PlaneLayout<F>::table_words : PlaneLayout<FS>::table_words];
if (shared) load_staged_lookup<FS>(table, thread_index_in_threadgroup, 64u);
else load_staged_lookup<F>(table, thread_index_in_threadgroup, 64u);
threadgroup_barrier(mem_flags::mem_threadgroup);
uint live = MAX_ROWS == 1 ? 1u : uint(descriptors[tile * 2u + 1u]);
if (MAX_ROWS == 1 || (live <= 2u && (tile == 0u || descriptors[(tile - 1u) * 2u] != int(e)))) {
    float sums[2];
    if (shared) column_dot<FS, true>(x + ulong(input_row) * k, live, k, shared_packed, shared_aux, shared_meta,
        n0 + thread_index_in_simdgroup, table, sums);
    else column_dot<F, PACKED>(x + ulong(input_row) * k, live, k, packed, aux, meta,
        e * n + n0 + thread_index_in_simdgroup, table, sums);
    for (uint r = 0u; r < live; ++r)
        y[ulong(row0 + r) * n + n0 + thread_index_in_simdgroup] = T(sums[r]);
    return;
}
if constexpr (MAX_ROWS > 1 && !COLUMN_ONLY) {
auto run = [&](auto rows) __attribute__((always_inline)) {
constexpr ushort R = decltype(rows)::value;
constexpr auto description = matmul2d_descriptor(R, 32, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> a(const_cast<device half*>(x) + ulong(row0) * k, dextents<int, 2>(k, R));
tensor<threadgroup half, dextents<int, 2>, tensor_inline> b(stages + sg * 2048u, dextents<int, 2>(32, 32));
auto left = a.template slice<32, R>(0, 0);
auto right = b.template slice<32, 32>(0, 0);
auto sum = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right), float>();
for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
if (shared) staged_dot<FS, R, true>(x, shared_packed, shared_aux, shared_meta, n0 + thread_index_in_simdgroup, k, params[2], row0,
    stages + sg * 2048u, table, sum);
else staged_dot<F, R, PACKED>(x, packed, aux, meta, e * n + n0 + thread_index_in_simdgroup, k, params[2], row0,
    stages + sg * 2048u, table, sum);
for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
    auto at = sum.get_multidimensional_index(i); y[ulong(row0 + at[1]) * n + n0 + at[0]] = T(sum[i]);
}
};
if constexpr (MAX_ROWS == 8) run(integral_constant<ushort, 8>{});
else {
    if (live <= 8u) run(integral_constant<ushort, 8>{});
    else if (live <= 16u) run(integral_constant<ushort, 16>{});
    else run(integral_constant<ushort, MAX_ROWS>{});
}
}
)METAL", header());
    return kernel;
}

int compact_splits(int n, int k) {
    if (n >= k) return 1;
    if (k < 4096 || std::int64_t(n) * k < 8 * 1024 * 1024) return 0;
    const int cores = metal_apple_gpu_core_count();
    if (cores <= 0) return 0;
    int splits = 1;
    while (splits < 8 && (k / 32) % (2 * splits) == 0 &&
        std::size_t(n / 64) * splits < std::size_t(cores) * 6 && k / (2 * splits) >= 512)
        splits *= 2;
    return splits > 1 ? splits : 0;
}

const fast::CustomKernelFunction& compact_split_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_compact_matrix_split",
        {"p0", "p1", "meta", "x", "params"}, {"partial"}, R"METAL(
uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
uint k = params[0], n = params[1], splits = params[2], split = threadgroup_position_in_grid.y;
uint n0 = (threadgroup_position_in_grid.x * 2u + sg) * 32u;
uint groups = k / 32u, per = groups / splits, begin = split * per, end = begin + per;
threadgroup half stages[4096];
threadgroup uint table[PlaneLayout<F>::table_words];
load_staged_lookup<F>(table, thread_index_in_threadgroup, 64u);
threadgroup_barrier(mem_flags::mem_threadgroup);
auto storage = stages + sg * 2048u;
constexpr auto description = matmul2d_descriptor(8, 32, 32, false, true, false,
    matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> input(const_cast<device half*>(x), dextents<int, 2>(k, 8));
tensor<threadgroup half, dextents<int, 2>, tensor_inline> b0(storage, dextents<int, 2>(32, 32));
tensor<threadgroup half, dextents<int, 2>, tensor_inline> b1(storage + 1024, dextents<int, 2>(32, 32));
auto right0 = b0.template slice<32, 32>(0, 0), right1 = b1.template slice<32, 32>(0, 0);
auto left = input.template slice<32, 8>(begin * 32u, 0);
auto accum = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right0), float>();
for (ushort i = 0; i < accum.get_capacity(); ++i) accum[i] = 0.f;
GroupPayload payload;
fetch_group<F>(payload, p0, p1, meta, n0 + lane, groups, begin, true);
for (uint group = begin; group < end; ++group) {
    decode_group<F>(payload, storage + ((group - begin) % 2u) * 1024u + lane * 32u, staged_lookup<F>(table), group);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (group + 1u < end) fetch_group<F>(payload, p0, p1, meta, n0 + lane, groups, group + 1u,
        (group + 1u) / PlaneLayout<F>::meta_groups != group / PlaneLayout<F>::meta_groups);
    left = input.template slice<32, 8>(group * 32u, 0);
    if ((group - begin) % 2u) operation.run(left, right1, accum); else operation.run(left, right0, accum);
}
for (ushort i = 0; i < accum.get_capacity(); ++i) if (accum.is_valid_element(i)) {
    auto at = accum.get_multidimensional_index(i);
    partial[(ulong(split) * 8u + at[1]) * n + n0 + at[0]] = accum[i];
}
)METAL", header());
    return kernel;
}

const fast::CustomKernelFunction& dense_kernel() {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_prefill_staged_dense",
        {"p0", "p1", "meta", "x", "params"}, {"y"}, R"METAL(
uint k = params[0], n = params[1], m = params[2];
uint n0 = threadgroup_position_in_grid.x * 64u;
uint sg = simdgroup_index_in_threadgroup, row0 = threadgroup_position_in_grid.y * 128u + sg * 32u;
uint local = thread_index_in_threadgroup, column = local % 64u, part = local / 64u;
threadgroup half stages[8192];
threadgroup uint table[PlaneLayout<F>::table_words];
load_staged_lookup<F>(table, local, 128u);
threadgroup_barrier(mem_flags::mem_threadgroup);
constexpr auto description = matmul2d_descriptor(32, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<description, execution_simdgroup> operation;
tensor<device half, dextents<int, 2>, tensor_inline> a(const_cast<device half*>(x), dextents<int, 2>(k, m));
tensor<threadgroup half, dextents<int, 2>, tensor_inline> b0(stages, dextents<int, 2>(64, 64));
tensor<threadgroup half, dextents<int, 2>, tensor_inline> b1(stages + 4096u, dextents<int, 2>(64, 64));
auto left = a.template slice<64, 32>(0, row0);
auto right0 = b0.template slice<64, 64>(0, 0);
auto right1 = b1.template slice<64, 64>(0, 0);
auto sum = operation.template get_destination_cooperative_tensor<decltype(left), decltype(right0), float>();
#pragma clang loop unroll(full)
for (ushort i = 0; i < sum.get_capacity(); ++i) sum[i] = 0.f;
GroupPayload payload;
fetch_group<F>(payload, p0, p1, meta, n0 + column, k / 32u, part, true);
for (uint step = 0; step < k / 64u; ++step) {
    const uint group = step * 2u + part;
    decode_group<F>(payload, stages + (step % 2u) * 4096u + column * 64u + part * 32u,
        staged_lookup<F>(table), group);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (group + 2u < k / 32u) fetch_group<F>(payload, p0, p1, meta, n0 + column, k / 32u, group + 2u,
        (group + 2u) / PlaneLayout<F>::meta_groups != group / PlaneLayout<F>::meta_groups);
    if (row0 < m) {
        left = a.template slice<64, 32>(step * 64u, row0);
        if (step % 2u) operation.run(left, right1, sum); else operation.run(left, right0, sum);
    }
}
if (row0 < m) {
    if constexpr (!TAIL) {
        auto output = tensor(y, dextents<int, 2>{int(n), int(m)}, array<int, 2>{1, int(n)});
        sum.store(output.template slice<64, 32>(n0, row0));
    } else {
#pragma clang loop unroll(full)
        for (ushort i = 0; i < sum.get_capacity(); ++i) if (sum.is_valid_element(i)) {
            auto at = sum.get_multidimensional_index(i);
            if (row0 + at[1] < m) y[ulong(row0 + at[1]) * n + n0 + at[0]] = sum[i];
        }
    }
}
)METAL", header());
    return kernel;
}
} // namespace

const std::string& MlxGgufStagedPrefill::decoder_header() { return header(); }

bool MlxGgufStagedPrefill::supports(const MlxGgufWeight& weight) noexcept {
    return !weight.indirect_ && (weight.shape_.size() == 2 || weight.shape_.size() == 3) && weight.output_size() % 64 == 0 &&
        weight.expert_count() <= 512 && weight.expert_count() <= INT_MAX / weight.output_size() &&
        layout(weight.format_).payload != 0;
}

void MlxGgufStagedPrefill::prepare(MlxGgufWeight& weight) {
    if (weight.planes_ || !supports(weight)) return;
    const int rows = weight.expert_count() * weight.output_size();
    auto planes = pack(weight.bytes_, zeros({1}, float16), weight.format_, rows,
        weight.input_size(), weight.row_bytes_);
    eval(planes);
    weight.planes_.emplace(std::array<array, 2>{std::move(planes[1]), std::move(planes[2])});
    weight.bytes_ = std::move(planes[0]);
}

array MlxGgufStagedPrefill::dequantize(const MlxGgufWeight& weight, Dtype dtype) {
    if (!weight.planes_) throw std::invalid_argument("GGUF weight has no compressed planes");
    const auto groups = std::size_t(weight.expert_count()) * weight.output_size() * weight.input_size() / 32;
    if (groups > INT_MAX - 127) throw std::overflow_error("GGUF dequantization grid is too large");
    return dequantize_planes_kernel()({weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1],
        array({uint32_t(weight.input_size()), uint32_t(weight.expert_count() * weight.output_size())})},
        {weight.shape_}, {dtype}, {int((groups + 127) / 128 * 128), 1, 1}, {128, 1, 1},
        {{"F", weight.format_}, {"T", dtype}}, std::nullopt, false, {}).front();
}

array MlxGgufStagedPrefill::embedding(const MlxGgufWeight& weight, const array& ids, Dtype dtype) {
    if (!weight.planes_ || weight.shape_.size() != 2)
        throw std::invalid_argument("GGUF embedding requires a matrix with compressed planes");
    const auto groups = ids.size() * (weight.input_size() / 32);
    if (groups > INT_MAX - 127) throw std::overflow_error("GGUF embedding grid is too large");
    auto shape = ids.shape(); shape.push_back(weight.input_size());
    return embedding_planes_kernel()({weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1],
        contiguous(astype(ids, int32)), array({uint32_t(weight.input_size()), uint32_t(weight.output_size()), uint32_t(ids.size())})},
        {shape}, {dtype}, {int((groups + 127) / 128 * 128), 1, 1}, {128, 1, 1},
        {{"F", weight.format_}, {"T", dtype}}, std::nullopt, false, {}).front();
}

array MlxGgufStagedPrefill::project(const MlxGgufWeight& weight, const array& input) {
    const auto m = input.size() / weight.input_size();
    if (!weight.planes_ || weight.shape_.size() != 2 || m < 1 || m > 8)
        throw std::invalid_argument("GGUF compressed projection geometry mismatch");
    if (auto output = project_split_many(std::span(&weight, 1), input)) return std::move(output->front());
    const int lanes = weight.output_size() >= 16384 ? 1 : 4;
    auto shape = input.shape(); shape.back() = weight.output_size();
    return projection_kernel()({weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1], contiguous(input),
        array({uint32_t(weight.input_size()), uint32_t(weight.output_size()), uint32_t(m)})},
        {shape}, {input.dtype()}, {weight.output_size() / (64 / lanes) * 64, 1, 1}, {64, 1, 1},
        {{"F", weight.format_}, {"T", input.dtype()}, {"L", lanes}, {"MR", m <= 6 ? 6 : 8}}, std::nullopt, false, {}).front();
}

std::vector<array> MlxGgufStagedPrefill::project_many(std::span<const MlxGgufWeight> weights, const array& input) {
    if (weights.size() < 2 || weights.size() > 4 || input.ndim() == 0 ||
        input.shape(-1) != weights.front().input_size())
        throw std::invalid_argument("GGUF projection batch geometry mismatch");
    const int k = input.shape(-1);
    const auto rows = input.size() / k;
    if (rows < 1 || rows > 8) throw std::invalid_argument("GGUF projection batch requires one through eight rows");
    if (rows >= 7 && k % 64 == 0 && gguf_expert_prefill_available(input) &&
        std::any_of(weights.begin(), weights.end(), [](const auto& weight) { return prefers_small_matrix(weight); })) {
        for (const auto& weight : weights)
            if (!weight.planes_ || weight.shape_.size() != 2 || weight.input_size() != k)
                throw std::invalid_argument("GGUF projection batch requires matching native matrices");
        std::vector<array> outputs;
        for (std::size_t index = 0; index < weights.size();) {
            if (prefers_small_matrix(weights[index])) {
                outputs.push_back(dense_small(weights[index], input));
                ++index;
            } else {
                auto end = index + 1;
                while (end < weights.size() && !prefers_small_matrix(weights[end])) ++end;
                if (end - index == 1) outputs.push_back(project(weights[index], input));
                else {
                    auto group = project_many(weights.subspan(index, end - index), input);
                    outputs.insert(outputs.end(), group.begin(), group.end());
                }
                index = end;
            }
        }
        return outputs;
    }
    if (auto output = project_split_many(weights, input)) return std::move(*output);
    std::vector<array> inputs;
    std::vector<Shape> shapes;
    std::vector<Dtype> dtypes(4, input.dtype());
    std::vector<std::pair<std::string, fast::TemplateArg>> arguments;
    arguments.emplace_back("MR", rows <= 6 ? 6 : 8);
    std::vector<uint32_t> params{uint32_t(k), uint32_t(rows)};
    std::size_t tiles = 0;
    for (int i = 0; i < 4; ++i) {
        const auto& weight = weights[std::min(std::size_t(i), weights.size() - 1)];
        if (!weight.planes_ || weight.shape_.size() != 2 || weight.input_size() != k)
            throw std::invalid_argument("GGUF projection batch requires matching native matrices");
        inputs.insert(inputs.end(), {weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1]});
        const int n = i < weights.size() ? weight.output_size() : 0;
        const int lanes = weight.output_size() >= 16384 ? 1 : 4;
        auto shape = input.shape(); shape.back() = n;
        shapes.push_back(n ? std::move(shape) : Shape{1});
        params.push_back(uint32_t(n));
        tiles += n / (64 / lanes);
        arguments.emplace_back("F" + std::to_string(i), weight.format_);
        arguments.emplace_back("L" + std::to_string(i), lanes);
    }
    if (tiles > INT_MAX / 64) throw std::overflow_error("GGUF projection batch grid is too large");
    inputs.push_back(contiguous(input));
    inputs.push_back(array(params.begin(), Shape{6}));
    auto outputs = projection_many_kernel()(inputs, shapes, dtypes, {int(tiles * 64), 1, 1}, {64, 1, 1},
        arguments, std::nullopt, false, {});
    outputs.erase(outputs.begin() + weights.size(), outputs.end());
    return outputs;
}

array MlxGgufStagedPrefill::dense(const MlxGgufWeight& weight, const array& input) {
    const int k = weight.input_size(), n = weight.output_size();
    const int m = int(input.size() / k);
    auto planes = weight.planes_ ? std::vector<array>{weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1]}
        : pack(weight.bytes_, input, weight.format_, n, k, weight.row_bytes_);
    auto shape = input.shape(); shape.back() = n;
    auto result = dense_kernel()({planes[0], planes[1], planes[2], contiguous(astype(input, float16)),
        array({uint32_t(k), uint32_t(n), uint32_t(m)})}, {shape}, {float32},
        {n / 64 * 128, (m + 127) / 128, 1}, {128, 1, 1}, {{"F", weight.format_}, {"TAIL", m % 32 != 0}},
        std::nullopt, false, {}).front();
    return astype(result, input.dtype());
}

array MlxGgufStagedPrefill::dense_small(const MlxGgufWeight& weight, const array& input) {
    const int k = weight.input_size(), n = weight.output_size();
    const auto rows = input.size() / k;
    if (!weight.planes_ || weight.shape_.size() != 2 || input.ndim() < 2 || input.shape(-1) != k ||
        rows < 1 || rows > 8 || k % 64 ||
        (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("GGUF compact matrix tile geometry mismatch");
    auto x = reshape(contiguous(astype(input, float16)), {int(rows), k});
    if (rows < 8) x = concatenate({x, zeros({8 - int(rows), k}, float16)}, 0);
    const int splits = compact_splits(n, k);
    if (splits > 1) {
        auto partial = compact_split_kernel()({weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1], x,
            array({uint32_t(k), uint32_t(n), uint32_t(splits)})}, {{splits, 8, n}}, {float32},
            {n, splits, 1}, {64, 1, 1}, {{"F", weight.format_}}, std::nullopt, false, {})[0];
        auto output = astype(sum(partial, 0), input.dtype());
        if (rows < 8) output = slice(output, {0, 0}, {int(rows), n});
        auto shape = input.shape(); shape.back() = n;
        return reshape(output, shape);
    }
    static const array zero({0}, int32), one({1}, int32);
    const MlxGgufGroupedRoutes routes{array({0, int(rows)}, int32), one, zero, zero, zero, 1, 8, int(rows)};
    auto output = down(weight, x, routes, input.dtype());
    if (rows < 8) output = slice(output, {0, 0}, {int(rows), n});
    auto shape = input.shape(); shape.back() = n;
    return reshape(output, shape);
}

bool MlxGgufStagedPrefill::prefers_small_matrix(const MlxGgufWeight& weight) noexcept {
    return weight.planes_ && weight.shape_.size() == 2 && weight.input_size() % 64 == 0 &&
        compact_splits(weight.output_size(), weight.input_size()) > 0;
}

array MlxGgufStagedPrefill::gate_up(const MlxGgufWeight& gate, const MlxGgufWeight& up,
    const array& input, const MlxGgufGroupedRoutes& routes, Dtype dtype,
    const MlxGgufWeight* shared_gate, const MlxGgufWeight* shared_up) {
    const int n = gate.output_size();
    const int column_lanes = routes.tile_rows <= 8 ? 4 : 1;
    const bool shared = shared_gate && shared_up;
    if (bool(shared_gate) != bool(shared_up) || (shared &&
        (!shared_gate->planes_ || !shared_up->planes_ || shared_gate->shape_ != Shape{n, gate.input_size()} ||
         shared_up->shape_ != shared_gate->shape_))) throw std::invalid_argument("GGUF shared Gate/Up geometry mismatch");
    if (!shared) { shared_gate = &gate; shared_up = &up; }
    const auto empty_meta = array(uint16_t(0), uint16);
    const auto aux = [](const MlxGgufWeight& w) { return w.planes_ ? (*w.planes_)[0] : w.bytes_; };
    const auto meta = [&](const MlxGgufWeight& w) { return w.planes_ ? (*w.planes_)[1] : empty_meta; };
    return gate_up_kernel()({gate.bytes_, up.bytes_, input, routes.descriptors, routes.count,
        array({uint32_t(gate.input_size()), uint32_t(n), uint32_t(input.shape(0)), uint32_t(gate.expert_count()),
            uint32_t(routes.tile_rows == 1 ? routes.descriptors.size() : 0)}),
        aux(gate), meta(gate), aux(up), meta(up), shared_gate->bytes_, shared_up->bytes_,
        aux(*shared_gate), meta(*shared_gate), aux(*shared_up), meta(*shared_up)},
        {{routes.tiles * routes.tile_rows, n}}, {dtype}, {n / (32 / column_lanes) * 64, routes.tiles, 1}, {64, 1, 1},
        {{"FG", gate.format_}, {"FU", up.format_}, {"T", dtype}, {"PG", gate.planes_.has_value()},
            {"PU", up.planes_.has_value()}, {"MAX_ROWS", routes.tile_rows}, {"HAS_SHARED", shared},
            {"FGS", shared_gate->format_}, {"FUS", shared_up->format_}, {"COLUMN_LANES", column_lanes},
            {"COLUMN_ONLY", routes.maximum_live_rows > 0 && routes.maximum_live_rows <= 2}}, std::nullopt, false, {}).front();
}

array MlxGgufStagedPrefill::down(const MlxGgufWeight& weight, const array& input,
    const MlxGgufGroupedRoutes& routes, Dtype dtype, const MlxGgufWeight* shared) {
    const int n = weight.output_size();
    const bool has_shared = shared != nullptr;
    if (shared && (!shared->planes_ || shared->shape_ != Shape{n, weight.input_size()}))
        throw std::invalid_argument("GGUF shared Down geometry mismatch");
    if (!shared) shared = &weight;
    const auto empty_meta = array(uint16_t(0), uint16);
    return down_kernel()({weight.bytes_, contiguous(astype(input, float16)), routes.descriptors, routes.count,
        array({uint32_t(weight.input_size()), uint32_t(n), uint32_t(input.shape(0)), uint32_t(weight.expert_count()),
            uint32_t(routes.tile_rows == 1 ? routes.descriptors.size() : 0)}),
        weight.planes_ ? (*weight.planes_)[0] : weight.bytes_, weight.planes_ ? (*weight.planes_)[1] : empty_meta,
        shared->bytes_, shared->planes_ ? (*shared->planes_)[0] : shared->bytes_, shared->planes_ ? (*shared->planes_)[1] : empty_meta},
        {{routes.tiles * routes.tile_rows, n}}, {dtype}, {n, routes.tiles, 1}, {64, 1, 1},
        {{"F", weight.format_}, {"T", dtype}, {"PACKED", weight.planes_.has_value()}, {"MAX_ROWS", routes.tile_rows},
            {"HAS_SHARED", has_shared}, {"FS", shared->format_},
            {"COLUMN_ONLY", routes.maximum_live_rows > 0 && routes.maximum_live_rows <= 2},
            {"BROADCAST", routes.tile_rows == 1 && input.shape(0) == 1}},
        std::nullopt, false, {}).front();
}
} // namespace mfq::metal
