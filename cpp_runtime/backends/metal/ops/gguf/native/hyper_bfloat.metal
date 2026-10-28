kernel void hyper_connection_norm_bfloat_loaded(device const bfloat* input [[buffer(0)]],
    device const float* norm [[buffer(1)]], device const float* inject [[buffer(2)]],
    device bfloat* normalized [[buffer(3)]], device float* weights [[buffer(4)]],
    constant HyperConnectionParams& p [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    constexpr uint width = kStreams * kNormHidden;
    const auto source = input + ulong(row) * width;
    const auto output = normalized + ulong(row) * width;
    threadgroup float scratch[kSimdgroups * kStreams];
    float values[kStreams][kNormValues];
    for (uint stream = 0; stream < kStreams; ++stream)
        for (uint i = 0; i < kNormValues; ++i)
            values[stream][i] = float(source[stream * kNormHidden + tid + i * kRowThreads]);
    float squares[kStreams] = {0.f, 0.f, 0.f, 0.f};
    for (uint stream = 0; stream < kStreams; ++stream)
        for (uint i = 0; i < kNormValues; ++i)
            squares[stream] += values[stream][i] * values[stream][i];
    reduce_streams(squares, kStreams, scratch, lane, sg);
    float inverse[kStreams];
    for (uint stream = 0; stream < kStreams; ++stream)
        inverse[stream] = rsqrt(squares[stream] / kNormHidden + kRmsEpsilon);
    float projection[kStreams] = {0.f, 0.f, 0.f, 0.f};
    for (uint stream = 0; stream < kStreams; ++stream) {
        float coefficients[kNormValues];
        for (uint i = 0; i < kNormValues; ++i)
            coefficients[i] = norm[stream * kNormHidden + tid + i * kRowThreads];
        for (uint i = 0; i < kNormValues; ++i) {
            const uint column = stream * kNormHidden + tid + i * kRowThreads;
            const float value = values[stream][i] * inverse[stream] * coefficients[i];
            output[column] = bfloat(value);
            if (p.inject)
                for (uint j = 0; j < kStreams; ++j)
                    projection[j] += inject[ulong(j) * width + column] * value;
        }
    }
    if (!p.inject) return;
    reduce_streams(projection, kStreams, scratch, lane, sg);
    if (tid < kStreams)
        weights[row * kStreams + tid] = 2.f * splash_sigmoid(projection[tid] / kStreams);
}

kernel void hyper_connection_norm_bfloat_quarters(device const bfloat* input [[buffer(0)]],
    device const float* norm [[buffer(1)]], device const float* inject [[buffer(2)]],
    device bfloat* normalized [[buffer(3)]], device float* weights [[buffer(4)]],
    constant HyperConnectionParams& p [[buffer(5)]], uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    if (row >= p.rows) return;
    constexpr uint width = kStreams * kNormHidden;
    const uint quarter = thread_id / kRowThreads, tid = thread_id % kRowThreads;
    const auto source = input + ulong(row) * width;
    const auto output = normalized + ulong(row) * width;
    threadgroup float scratch[kStreams * kSimdgroups];
    threadgroup float inverse[kStreams];
    float values[kNormValues];
    for (uint i = 0; i < kNormValues; ++i)
        values[i] = float(source[quarter * kNormHidden + tid + i * kRowThreads]);
    float squares = 0.f;
    for (uint i = 0; i < kNormValues; ++i) squares += values[i] * values[i];
    squares = simd_sum(squares);
    if (!lane) scratch[sg] = squares;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) {
        float total = 0.f;
        for (uint group = 0; group < kSimdgroups; ++group)
            total += scratch[quarter * kSimdgroups + group];
        inverse[quarter] = rsqrt(total / kNormHidden + kRmsEpsilon);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float projection = 0.f;
    for (uint stream = 0; stream < kStreams; ++stream) {
        const float scale = inverse[stream];
        float xs[kNormValues], ns[kNormValues], ws[kNormValues];
        for (uint i = 0; i < kNormValues; ++i) {
            const uint column = stream * kNormHidden + tid + i * kRowThreads;
            xs[i] = float(source[column]);
            ns[i] = norm[column];
            ws[i] = inject[ulong(quarter) * width + column];
        }
        for (uint i = 0; i < kNormValues; ++i) {
            const uint column = stream * kNormHidden + tid + i * kRowThreads;
            const float value = xs[i] * scale * ns[i];
            if (stream == quarter) output[column] = bfloat(value);
            projection += ws[i] * value;
        }
    }
    projection = simd_sum(projection);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!lane) scratch[sg] = projection;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid) return;
    float total = 0.f;
    for (uint group = 0; group < kSimdgroups; ++group)
        total += scratch[quarter * kSimdgroups + group];
    weights[row * kStreams + quarter] = 2.f * splash_sigmoid(total / kStreams);
}
