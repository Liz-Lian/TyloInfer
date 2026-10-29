constant uint tinfer_gu_gr [[function_constant(0)]];
constant uint tinfer_gu_gs [[function_constant(1)]];
constant uint tinfer_gu_ur [[function_constant(2)]];
constant uint tinfer_gu_us [[function_constant(3)]];

inline uint4 moe_native_gu_formats(constant MoeGgufGateUpParams& p) {
    return uint4(is_function_constant_defined(tinfer_gu_gr) ? tinfer_gu_gr : p.gate_routed_format,
        is_function_constant_defined(tinfer_gu_gs) ? tinfer_gu_gs : p.gate_shared_format,
        is_function_constant_defined(tinfer_gu_ur) ? tinfer_gu_ur : p.up_routed_format,
        is_function_constant_defined(tinfer_gu_us) ? tinfer_gu_us : p.up_shared_format);
}

kernel void moe_native_single_gate_up(device bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* g0 [[buffer(3)]], device uchar* g1 [[buffer(4)]],
    device uchar* gm [[buffer(5)]], device uchar* sg0 [[buffer(6)]], device uchar* sg1 [[buffer(7)]],
    device uchar* sgm [[buffer(8)]], device uchar* u0 [[buffer(9)]], device uchar* u1 [[buffer(10)]],
    device uchar* um [[buffer(11)]], device uchar* su0 [[buffer(12)]], device uchar* su1 [[buffer(13)]],
    device uchar* sumeta [[buffer(14)]], device bfloat* output [[buffer(15)]],
    constant MoeGgufGateUpParams& p [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint4 formats = moe_native_gu_formats(p);
    const auto gs = moe_gguf_segment(ids[group.y], p.experts, p.input_size, p.output_size,
        formats.x, formats.y, slots, g0, g1, gm, sg0, sg1, sgm);
    const auto us = moe_gguf_segment(ids[group.y], p.experts, p.input_size, p.output_size,
        formats.z, formats.w, slots, u0, u1, um, su0, su1, sumeta);
    threadgroup half2 tables[2 * kQuantPairTableEntries];
    threadgroup bfloat gate[MOE_GGUF_COLUMN_ROWS * GGUF_STAGED_COLUMNS];
    bool initialized = false;
    quant_format_switch(gs.format, [&](auto f) {
        initialized |= quant_pair_entries<decltype(f)>(tables, tid, 64);
    });
    quant_format_switch(us.format, [&](auto f) {
        initialized |= quant_pair_entries<decltype(f)>(tables + kQuantPairTableEntries, tid, 64);
    });
    if (initialized) threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool up = simdgroup == 1;
    moe_gate_up_column(input, 1, p.input_size, p.output_size, up ? us : gs,
        group.x * GGUF_STAGED_COLUMNS, lane, tables + (up ? kQuantPairTableEntries : 0), up,
        gate, output, ulong(group.y) * p.output_size);
}

kernel void moe_native_single_gate_up_split(device bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* g0 [[buffer(3)]], device uchar* g1 [[buffer(4)]],
    device uchar* gm [[buffer(5)]], device uchar* sg0 [[buffer(6)]], device uchar* sg1 [[buffer(7)]],
    device uchar* sgm [[buffer(8)]], device uchar* u0 [[buffer(9)]], device uchar* u1 [[buffer(10)]],
    device uchar* um [[buffer(11)]], device uchar* su0 [[buffer(12)]], device uchar* su1 [[buffer(13)]],
    device uchar* sumeta [[buffer(14)]], device bfloat* output [[buffer(15)]],
    constant MoeGgufGateUpParams& p [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint4 formats = moe_native_gu_formats(p);
    const auto gs = moe_gguf_segment(ids[group.y], p.experts, p.input_size, p.output_size,
        formats.x, formats.y, slots, g0, g1, gm, sg0, sg1, sgm);
    const auto us = moe_gguf_segment(ids[group.y], p.experts, p.input_size, p.output_size,
        formats.z, formats.w, slots, u0, u1, um, su0, su1, sumeta);
    threadgroup half2 tables[2 * kQuantPairTableEntries];
    threadgroup bfloat gate[16];
    bool initialized = false;
    quant_format_switch(gs.format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(tables, tid, 64); });
    quant_format_switch(us.format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(tables + kQuantPairTableEntries, tid, 64); });
    if (initialized) threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint split = lane / 16;
    const bool up = simdgroup == 1;
    const auto weight = up ? us : gs;
    const uint per = p.input_size / 64;
    float sums[1];
    quant_format_switch(weight.format, [&](auto f) {
        column_dot<decltype(f), 1>(input, 1, p.input_size, weight.w0, weight.w1, weight.meta,
            group.x * 16 + (lane & 15), split * per, (split + 1) * per,
            tables + (up ? kQuantPairTableEntries : 0), sums);
    });
    const float value = sums[0] + simd_shuffle_xor(sums[0], 16);
    if (!split && !up) gate[lane] = bfloat(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!split && up) output[ulong(group.y) * p.output_size + group.x * 16 + lane] =
        bfloat(float(bfloat(value)) * splash_silu(float(gate[lane])));
}

kernel void moe_native_single_down(device const bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* w0 [[buffer(3)]], device uchar* w1 [[buffer(4)]],
    device uchar* wm [[buffer(5)]], device uchar* s0 [[buffer(6)]], device uchar* s1 [[buffer(7)]],
    device uchar* sm [[buffer(8)]], device bfloat* output [[buffer(9)]],
    constant MoeGgufExpertParams& p [[buffer(10)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const auto weight = moe_gguf_segment(ids[group.y], p, slots, w0, w1, wm, s0, s1, sm);
    threadgroup half2 table[kQuantPairTableEntries];
    float sums[1];
    const uint column = group.x * GGUF_TILE_COLUMNS + tid;
    column_dot_any<1>(weight.format, input + ulong(group.y) * p.input_size, 1, p.input_size,
        weight.w0, weight.w1, weight.meta, column, 0, p.input_size / 32, table, tid, 64, sums);
    output[ulong(group.y) * p.output_size + column] = bfloat(sums[0]);
}

struct MoeNativeSingleDownParams {
    MoeGgufExpertParams projection;
    uint routes;
};

kernel void moe_native_single_down_combine(device const bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* w0 [[buffer(3)]], device uchar* w1 [[buffer(4)]],
    device uchar* wm [[buffer(5)]], device uchar* s0 [[buffer(6)]], device uchar* s1 [[buffer(7)]],
    device uchar* sm [[buffer(8)]], device bfloat* output [[buffer(9)]],
    device const float* probabilities [[buffer(10)]], constant MoeNativeSingleDownParams& config [[buffer(11)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint route [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constant auto& p = config.projection;
    const uint routes = config.routes;
    threadgroup half2 table[2 * kQuantPairTableEntries];
    threadgroup bfloat values[32 * 32];
    bool initialized = false;
    quant_format_switch(p.routed_format, [&](auto f) {
        initialized |= quant_pair_entries<decltype(f)>(table, tid, routes * 32);
    });
    quant_format_switch(p.shared_format, [&](auto f) {
        initialized |= quant_pair_entries<decltype(f)>(table + kQuantPairTableEntries, tid, routes * 32);
    });
    if (initialized) threadgroup_barrier(mem_flags::mem_threadgroup);
    const auto weight = moe_gguf_segment(ids[route], p, slots, w0, w1, wm, s0, s1, sm);
    float sums[1];
    quant_format_switch(weight.format, [&](auto f) {
        column_dot<decltype(f), 1>(input + ulong(route) * p.input_size, 1, p.input_size,
            weight.w0, weight.w1, weight.meta, group * 32 + lane, 0, p.input_size / 32,
            table + (ids[route] == p.experts ? kQuantPairTableEntries : 0), sums);
    });
    values[route * 32 + lane] = bfloat(sums[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!route) {
        float sum = 0.f;
        for (uint i = 0; i < routes; ++i) sum += float(values[i * 32 + lane]) * probabilities[i];
        output[group * 32 + lane] = bfloat(sum);
    }
}
