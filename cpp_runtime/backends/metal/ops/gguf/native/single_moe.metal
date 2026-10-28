kernel void moe_native_single_gate_up(device bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* g0 [[buffer(3)]], device uchar* g1 [[buffer(4)]],
    device uchar* gm [[buffer(5)]], device uchar* sg0 [[buffer(6)]], device uchar* sg1 [[buffer(7)]],
    device uchar* sgm [[buffer(8)]], device uchar* u0 [[buffer(9)]], device uchar* u1 [[buffer(10)]],
    device uchar* um [[buffer(11)]], device uchar* su0 [[buffer(12)]], device uchar* su1 [[buffer(13)]],
    device uchar* sumeta [[buffer(14)]], device bfloat* output [[buffer(15)]],
    constant MoeGgufGateUpParams& p [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const auto gs = moe_gguf_segment(ids[group.y], p.experts, p.input_size, p.output_size,
        p.gate_routed_format, p.gate_shared_format, slots, g0, g1, gm, sg0, sg1, sgm);
    const auto us = moe_gguf_segment(ids[group.y], p.experts, p.input_size, p.output_size,
        p.up_routed_format, p.up_shared_format, slots, u0, u1, um, su0, su1, sumeta);
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
