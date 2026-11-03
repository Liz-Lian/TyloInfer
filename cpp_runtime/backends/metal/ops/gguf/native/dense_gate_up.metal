struct NativeDenseGateUpParams {
    uint input_size, output_size, gate_format, up_format, round_fp16;
};

template <ushort Splits, bool GridOnly>
kernel void dense_gguf_gate_up(device const bfloat* input [[buffer(0)]],
    device uchar* g0 [[buffer(1)]], device uchar* g1 [[buffer(2)]], device uchar* gm [[buffer(3)]],
    device uchar* u0 [[buffer(4)]], device uchar* u1 [[buffer(5)]], device uchar* um [[buffer(6)]],
    device bfloat* output [[buffer(7)]], constant NativeDenseGateUpParams& p [[buffer(8)]],
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    constexpr uint columns = 32 / Splits;
    threadgroup half2 tables[GridOnly ? 1 : 2 * kQuantPairTableEntries];
    threadgroup bfloat gate[columns];
    if constexpr (!GridOnly) {
        bool initialized = false;
        quant_format_switch(p.gate_format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(tables, tid, 64); });
        quant_format_switch(p.up_format, [&](auto f) { initialized |= quant_pair_entries<decltype(f)>(tables + kQuantPairTableEntries, tid, 64); });
        if (initialized) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bool up = simdgroup == 1;
    const uint split = lane / columns, per = p.input_size / (32 * Splits);
    auto table = tables;
    if constexpr (!GridOnly) table += up ? kQuantPairTableEntries : 0;
    float sums[1] = {0.f};
    quant_format_switch(up ? p.up_format : p.gate_format, [&](auto f) {
        if constexpr (!GridOnly || decltype(f)::Kind == QuantGrid)
            column_dot<decltype(f), 1>(input, 1, p.input_size, up ? u0 : g0, up ? u1 : g1, up ? um : gm,
                group * columns + lane % columns, split * per, (split + 1) * per, table, sums);
    });
    float value = sums[0];
    for (uint offset = columns; offset < 32; offset *= 2) value += simd_shuffle_xor(value, offset);
    if (!split && !up) gate[lane] = bfloat(value);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!split && up) {
        const bfloat product = bfloat(float(bfloat(value)) * splash_silu(float(gate[lane])));
        output[group * columns + lane] = p.round_fp16 ? bfloat(half(product)) : product;
    }
}

#define TINFER_DENSE_GU(S, G, suffix) \
template [[host_name("dense_gguf_gate_up" suffix)]] kernel decltype(dense_gguf_gate_up<S, G>) dense_gguf_gate_up<S, G>;
TINFER_DENSE_GU(1, false, "")
TINFER_DENSE_GU(2, false, "_split")
TINFER_DENSE_GU(4, false, "_split4")
TINFER_DENSE_GU(1, true, "_grid")
TINFER_DENSE_GU(2, true, "_grid_split")
TINFER_DENSE_GU(4, true, "_grid_split4")
#undef TINFER_DENSE_GU
