template <class F>
inline float native_affine_vector_dot(device const bfloat* x, device uchar* w0, device uchar* w1,
    device uchar* meta, uint column, uint width, uint lane, uint lanes) {
    const uint groups = width / 32, units = groups / F::MetaGroups;
    const uint tile = column / QUANT_TILE_ROWS, row = column % QUANT_TILE_ROWS;
    float result = 0.0f;
    for (uint g = lane; g < groups; g += lanes) {
        const ulong index = (ulong(tile) * groups + g) * QUANT_TILE_ROWS + row;
        const ulong coefficient = (ulong(tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS + row;
        auto p0 = w0 + index * F::P0, p1 = w1 + index * F::P1;
        const auto scale = F::coef(F::loadMeta(meta + coefficient * F::MetaBytes), 0);
        float dot = 0.0f, sum = 0.0f;
        for (ushort c = 0; c < 4; ++c) {
            const uint4 q = F::codes(F::loadChunk(p0, p1, c));
            const uint offsets[4] = {4u * c, 4u * c + 2u, 16u + 4u * c, 18u + 4u * c};
            for (uint j = 0; j < 4; ++j) {
                const float a = float(x[g * 32 + offsets[j]]), b = float(x[g * 32 + offsets[j] + 1]);
                dot += a * float(q[j] & 0xFFFFu) + b * float(q[j] >> 16);
                sum += a + b;
            }
        }
        result += scale.s.x * dot + scale.m.x * sum;
    }
    if (lanes == 8) result += simd_shuffle_down(result, 4);
    result += simd_shuffle_down(result, 2);
    result += simd_shuffle_down(result, 1);
    return result;
}

inline float native_affine_vector_any(uint format, device const bfloat* x, device uchar* w0, device uchar* w1,
    device uchar* meta, uint column, uint width, uint lane, uint lanes) {
    switch (format) {
#define NATIVE_AFFINE_VECTOR(B, GS) case GGUF_FMT_AF##B##G##GS: \
    return native_affine_vector_dot<FmtAF##B##G##GS>(x, w0, w1, meta, column, width, lane, lanes);
#define NATIVE_AFFINE_VECTOR_GROUPS(B) NATIVE_AFFINE_VECTOR(B, 32) NATIVE_AFFINE_VECTOR(B, 64) NATIVE_AFFINE_VECTOR(B, 128)
        NATIVE_AFFINE_VECTOR_GROUPS(2) NATIVE_AFFINE_VECTOR_GROUPS(3) NATIVE_AFFINE_VECTOR_GROUPS(4)
        NATIVE_AFFINE_VECTOR_GROUPS(5) NATIVE_AFFINE_VECTOR_GROUPS(6) NATIVE_AFFINE_VECTOR_GROUPS(8)
#undef NATIVE_AFFINE_VECTOR_GROUPS
#undef NATIVE_AFFINE_VECTOR
    }
    return 0.0f;
}

template <class F>
inline float2 native_affine_vector_pair_dot(device const bfloat* x, device uchar* g0, device uchar* g1,
    device uchar* gm, device uchar* u0, device uchar* u1, device uchar* um,
    uint column, uint width, uint lane, uint lanes) {
    const uint groups = width / 32, units = groups / F::MetaGroups;
    const uint tile = column / QUANT_TILE_ROWS, row = column % QUANT_TILE_ROWS;
    float2 result = 0.0f;
    for (uint g = lane; g < groups; g += lanes) {
        const ulong index = (ulong(tile) * groups + g) * QUANT_TILE_ROWS + row;
        const ulong coefficient = (ulong(tile) * units + g / F::MetaGroups) * QUANT_TILE_ROWS + row;
        const auto gs = F::coef(F::loadMeta(gm + coefficient * F::MetaBytes), 0);
        const auto us = F::coef(F::loadMeta(um + coefficient * F::MetaBytes), 0);
        float2 dot = 0.0f;
        float sum = 0.0f;
        for (ushort c = 0; c < 4; ++c) {
            const uint4 gq = F::codes(F::loadChunk(g0 + index * F::P0, g1 + index * F::P1, c));
            const uint4 uq = F::codes(F::loadChunk(u0 + index * F::P0, u1 + index * F::P1, c));
            const uint offsets[4] = {4u * c, 4u * c + 2u, 16u + 4u * c, 18u + 4u * c};
            for (uint j = 0; j < 4; ++j) {
                const float a = float(x[g * 32 + offsets[j]]), b = float(x[g * 32 + offsets[j] + 1]);
                dot.x += a * float(gq[j] & 0xFFFFu) + b * float(gq[j] >> 16);
                dot.y += a * float(uq[j] & 0xFFFFu) + b * float(uq[j] >> 16);
                sum += a + b;
            }
        }
        result += float2(gs.s.x, us.s.x) * dot + float2(gs.m.x, us.m.x) * sum;
    }
    if (lanes == 8) result += simd_shuffle_down(result, 4);
    result += simd_shuffle_down(result, 2);
    result += simd_shuffle_down(result, 1);
    return result;
}

inline float2 native_affine_vector_pair_any(uint format, device const bfloat* x, device uchar* g0,
    device uchar* g1, device uchar* gm, device uchar* u0, device uchar* u1, device uchar* um,
    uint column, uint width, uint lane, uint lanes) {
    switch (format) {
#define NATIVE_AFFINE_PAIR(B, GS) case GGUF_FMT_AF##B##G##GS: \
    return native_affine_vector_pair_dot<FmtAF##B##G##GS>(x, g0, g1, gm, u0, u1, um, column, width, lane, lanes);
#define NATIVE_AFFINE_PAIR_GROUPS(B) NATIVE_AFFINE_PAIR(B, 32) NATIVE_AFFINE_PAIR(B, 64) NATIVE_AFFINE_PAIR(B, 128)
        NATIVE_AFFINE_PAIR_GROUPS(2) NATIVE_AFFINE_PAIR_GROUPS(3) NATIVE_AFFINE_PAIR_GROUPS(4)
        NATIVE_AFFINE_PAIR_GROUPS(5) NATIVE_AFFINE_PAIR_GROUPS(6) NATIVE_AFFINE_PAIR_GROUPS(8)
#undef NATIVE_AFFINE_PAIR_GROUPS
#undef NATIVE_AFFINE_PAIR
    }
    return 0.0f;
}

kernel void moe_affine_vector_gate_up(device const bfloat* x [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* g0 [[buffer(3)]], device uchar* g1 [[buffer(4)]],
    device uchar* gm [[buffer(5)]], device uchar* sg0 [[buffer(6)]], device uchar* sg1 [[buffer(7)]],
    device uchar* sgm [[buffer(8)]], device uchar* u0 [[buffer(9)]], device uchar* u1 [[buffer(10)]],
    device uchar* um [[buffer(11)]], device uchar* su0 [[buffer(12)]], device uchar* su1 [[buffer(13)]],
    device uchar* sumeta [[buffer(14)]], device bfloat* output [[buffer(15)]],
    constant MoeGgufGateUpParams& p [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lanes = p.input_size > p.output_size ? 8u : 4u;
    const uint column = group.x * (256u / lanes) + tid / lanes, lane = tid % lanes;
    if (column >= p.output_size) return;
    const uint expert = ids[group.y];
    const bool shared = expert == p.experts;
    const uint row = shared ? column : slots[expert] * p.output_size + column;
    const uint gate_format = shared ? p.gate_shared_format : p.gate_routed_format;
    const uint up_format = shared ? p.up_shared_format : p.up_routed_format;
    if (gate_format == up_format) {
        const float2 value = native_affine_vector_pair_any(gate_format, x,
            shared ? sg0 : g0, shared ? sg1 : g1, shared ? sgm : gm,
            shared ? su0 : u0, shared ? su1 : u1, shared ? sumeta : um,
            row, p.input_size, lane, lanes);
        if (!lane) output[ulong(group.y) * p.output_size + column] =
            bfloat(splash_silu(float(bfloat(value.x))) * float(bfloat(value.y)));
        return;
    }
    const float gate = native_affine_vector_any(gate_format,
        x, shared ? sg0 : g0, shared ? sg1 : g1, shared ? sgm : gm, row, p.input_size, lane, lanes);
    const float up = native_affine_vector_any(up_format,
        x, shared ? su0 : u0, shared ? su1 : u1, shared ? sumeta : um, row, p.input_size, lane, lanes);
    if (!lane) output[ulong(group.y) * p.output_size + column] = bfloat(splash_silu(float(bfloat(gate))) * float(bfloat(up)));
}

kernel void moe_affine_vector_down(device const bfloat* x [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* w0 [[buffer(3)]], device uchar* w1 [[buffer(4)]],
    device uchar* wm [[buffer(5)]], device uchar* s0 [[buffer(6)]], device uchar* s1 [[buffer(7)]],
    device uchar* sm [[buffer(8)]], device bfloat* output [[buffer(9)]],
    constant MoeGgufExpertParams& p [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const uint lanes = p.input_size > p.output_size ? 8u : 4u;
    const uint column = group.x * (256u / lanes) + tid / lanes, lane = tid % lanes;
    if (column >= p.output_size) return;
    const uint expert = ids[group.y];
    const bool shared = expert == p.experts;
    const uint row = shared ? column : slots[expert] * p.output_size + column;
    const float value = native_affine_vector_any(shared ? p.shared_format : p.routed_format,
        x + ulong(group.y) * p.input_size, shared ? s0 : w0, shared ? s1 : w1, shared ? sm : wm,
        row, p.input_size, lane, lanes);
    if (!lane) output[ulong(group.y) * p.output_size + column] = bfloat(value);
}

kernel void moe_single_reduce(device const bfloat* values [[buffer(0)]], device const float* weights [[buffer(1)]],
    device bfloat* output [[buffer(2)]], constant MoeCombineParams& p [[buffer(3)]], uint column [[thread_position_in_grid]]) {
    if (column >= p.hidden_size) return;
    float sum = 0.0f;
    for (uint route = 0; route < p.routes_per_row; ++route)
        sum += float(values[ulong(route) * p.hidden_size + column]) * weights[route];
    output[column] = bfloat(sum);
}
