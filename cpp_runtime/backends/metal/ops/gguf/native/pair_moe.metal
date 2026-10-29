struct MoeNativeSingleDownParams {
    MoeGgufExpertParams projection;
    uint routes;
};

struct MoeNativePairGateUpParams {
    MoeGgufGateUpParams projection;
    uint routes;
};

inline uint moe_native_pair_peer(device const uint* ids, uint index, uint routes) {
    const uint begin = index < routes ? routes : 0;
    const uint expert = ids[index];
    for (uint route = 0; route < routes; ++route)
        if (ids[begin + route] == expert) return begin + route;
    return 2 * routes;
}

__attribute__((noinline)) float2 moe_native_pair_dot(MoeGgufSegment weight, device const bfloat* input,
    uint rows, uint width, uint column, threadgroup half2* table) {
    float sums[2];
    quant_format_switch(weight.format, [&](auto f) {
        column_dot<decltype(f), 2>(input, rows, width, weight.w0, weight.w1, weight.meta,
            column, 0, width / 32, table, sums);
    });
    return float2(sums[0], sums[1]);
}

__attribute__((noinline)) float2 moe_native_pair_split_dot(MoeGgufSegment weight, device const bfloat* input,
    uint rows, uint width, uint column, uint split, threadgroup half2* table) {
    float sums[2];
    const uint per = width / 64;
    quant_format_switch(weight.format, [&](auto f) {
        column_dot<decltype(f), 2>(input, rows, width, weight.w0, weight.w1, weight.meta,
            column, split * per, (split + 1) * per, table, sums);
    });
    return float2(sums[0], sums[1]);
}

template <bool SplitK>
kernel void moe_native_pair_gate_up(device bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* g0 [[buffer(3)]], device uchar* g1 [[buffer(4)]],
    device uchar* gm [[buffer(5)]], device uchar* sg0 [[buffer(6)]], device uchar* sg1 [[buffer(7)]],
    device uchar* sgm [[buffer(8)]], device uchar* u0 [[buffer(9)]], device uchar* u1 [[buffer(10)]],
    device uchar* um [[buffer(11)]], device uchar* su0 [[buffer(12)]], device uchar* su1 [[buffer(13)]],
    device uchar* sumeta [[buffer(14)]], device bfloat* output [[buffer(15)]],
    constant MoeNativePairGateUpParams& config [[buffer(16)]], uint3 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constant auto& p = config.projection;
    const uint index = group.y, row_offset = group.z * 2, token = index / config.routes;
    ids += row_offset * config.routes;
    const uint peer = moe_native_pair_peer(ids, index, config.routes);
    const bool paired = peer < 2 * config.routes;
    if (token && paired) return;
    const auto gs = moe_gguf_segment(ids[index], p.experts, p.input_size, p.output_size,
        p.gate_routed_format, p.gate_shared_format, slots, g0, g1, gm, sg0, sg1, sgm);
    const auto us = moe_gguf_segment(ids[index], p.experts, p.input_size, p.output_size,
        p.up_routed_format, p.up_shared_format, slots, u0, u1, um, su0, su1, sumeta);
    threadgroup half2 tables[2 * kQuantPairTableEntries];
    threadgroup bfloat gate[2 * 32];
    bool initialized = false;
    quant_format_switch(gs.format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(tables, tid, 64); });
    quant_format_switch(us.format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(tables + kQuantPairTableEntries, tid, 64); });
    if (initialized) threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool up = simdgroup == 1;
    const auto weight = up ? us : gs;
    const uint rows = paired ? 2 : 1;
    constexpr uint columns = SplitK ? 16 : 32;
    float2 sums;
    if constexpr (SplitK) {
        const float2 partial = moe_native_pair_split_dot(weight, input + ulong(row_offset + token) * p.input_size,
            rows, p.input_size, group.x * columns + (lane & 15), lane / 16,
            tables + (up ? kQuantPairTableEntries : 0));
        sums = partial + float2(simd_shuffle_xor(partial.x, 16), simd_shuffle_xor(partial.y, 16));
    } else {
        sums = moe_native_pair_dot(weight, input + ulong(row_offset + token) * p.input_size,
            rows, p.input_size, group.x * columns + lane, tables + (up ? kQuantPairTableEntries : 0));
    }
    if (!up && lane < columns) for (uint row = 0; row < rows; ++row) gate[row * columns + lane] = bfloat(sums[row]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (up && lane < columns) for (uint row = 0; row < rows; ++row)
        output[(ulong(row_offset * config.routes + index) * 2 + row) * p.output_size + group.x * columns + lane] =
            bfloat(float(bfloat(sums[row])) * splash_silu(float(gate[row * columns + lane])));
}

template [[host_name("moe_native_pair_gate_up")]] kernel decltype(moe_native_pair_gate_up<false>) moe_native_pair_gate_up<false>;
template [[host_name("moe_native_pair_gate_up_split")]] kernel decltype(moe_native_pair_gate_up<true>) moe_native_pair_gate_up<true>;

kernel void moe_native_pair_down_combine(device const bfloat* input [[buffer(0)]], device const uint* ids [[buffer(1)]],
    device const uint* slots [[buffer(2)]], device uchar* w0 [[buffer(3)]], device uchar* w1 [[buffer(4)]],
    device uchar* wm [[buffer(5)]], device uchar* s0 [[buffer(6)]], device uchar* s1 [[buffer(7)]],
    device uchar* sm [[buffer(8)]], device bfloat* output [[buffer(9)]],
    device const float* probabilities [[buffer(10)]], constant MoeNativeSingleDownParams& config [[buffer(11)]],
    uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    constant auto& p = config.projection;
    const uint index = tid / 32, lane = tid % 32, routes = config.routes;
    const uint token = index / routes, rank = index % routes;
    const uint row_offset = group.y * 2;
    ids += row_offset * routes;
    probabilities += row_offset * routes;
    const uint peer = moe_native_pair_peer(ids, index, routes);
    const bool paired = peer < 2 * routes;
    threadgroup half2 table[2 * kQuantPairTableEntries];
    threadgroup bfloat values[2 * 16 * 32];
    bool initialized = false;
    quant_format_switch(p.routed_format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(table, tid, routes * 64); });
    quant_format_switch(p.shared_format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(table + kQuantPairTableEntries, tid, routes * 64); });
    if (initialized) threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!token || !paired) {
        const auto weight = moe_gguf_segment(ids[index], p, slots, w0, w1, wm, s0, s1, sm);
        const uint rows = paired ? 2 : 1;
        const float2 sums = moe_native_pair_dot(weight, input + ulong(row_offset * routes + index) * 2 * p.input_size,
            rows, p.input_size, group.x * 32 + lane,
            table + (ids[index] == p.experts ? kQuantPairTableEntries : 0));
        values[(token * routes + rank) * 32 + lane] = bfloat(sums[0]);
        if (paired) values[peer * 32 + lane] = bfloat(sums[1]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (index < 2) {
        float sum = 0.f;
        for (uint route = 0; route < routes; ++route)
            sum += float(values[(index * routes + route) * 32 + lane]) * probabilities[index * routes + route];
        output[ulong(row_offset + index) * p.output_size + group.x * 32 + lane] = bfloat(sum);
    }
}
