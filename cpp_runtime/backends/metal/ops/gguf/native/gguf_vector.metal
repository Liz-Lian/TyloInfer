inline float native_iq4_vector_dot(device const bfloat* input, device uchar* w0, device uchar* meta,
    uint column, uint width, uint begin, uint end) {
    const uint groups = width / 32, tile = column / QUANT_TILE_ROWS, row = column % QUANT_TILE_ROWS;
    const auto values = w0 + (ulong(tile) * groups * QUANT_TILE_ROWS + row) * FmtIQ4XS::P0;
    const auto scales = meta + (ulong(tile) * (groups / FmtIQ4XS::MetaGroups) * QUANT_TILE_ROWS + row) * FmtIQ4XS::MetaBytes;
    float result = 0.0f;
    for (uint group = begin; group < end; ++group) {
        const auto payload = FmtIQ4XS::load(values + ulong(group) * QUANT_TILE_ROWS * FmtIQ4XS::P0, w0);
        const auto m = FmtIQ4XS::loadMeta(scales + ulong(group / FmtIQ4XS::MetaGroups) * QUANT_TILE_ROWS * FmtIQ4XS::MetaBytes);
        float sum = 0.0f;
        for (ushort chunk = 0; chunk < 4; ++chunk) {
            const uchar4 codes = as_type<uchar4>(FmtIQ4XS::indices(FmtIQ4XS::chunk(payload, chunk)));
            const float4 low = float4(*reinterpret_cast<device const bfloat4*>(input + group * 32 + chunk * 4));
            const float4 high = float4(*reinterpret_cast<device const bfloat4*>(input + group * 32 + 16 + chunk * 4));
            sum += low.x * float(FmtIQ4XS::value(codes.x & 15));
            sum += low.y * float(FmtIQ4XS::value(codes.x >> 4));
            sum += low.z * float(FmtIQ4XS::value(codes.y & 15));
            sum += low.w * float(FmtIQ4XS::value(codes.y >> 4));
            sum += high.x * float(FmtIQ4XS::value(codes.z & 15));
            sum += high.y * float(FmtIQ4XS::value(codes.z >> 4));
            sum += high.z * float(FmtIQ4XS::value(codes.w & 15));
            sum += high.w * float(FmtIQ4XS::value(codes.w >> 4));
        }
        result += sum * FmtIQ4XS::coef(m, ushort(group % FmtIQ4XS::MetaGroups)).s.x;
    }
    return result;
}

template <class T>
inline void native_gguf_vector(device const bfloat* input, device uchar* a0, device uchar* a1,
    device uchar* am, device uchar* b0, device uchar* b1, device uchar* bm, device uchar* c0,
    device uchar* c1, device uchar* cm, device T* output,
    constant GgufDecodeFusedParams& p, uint group, uint tid, threadgroup half2* table) {
    const uint end0 = p.cols[0] / 64, end1 = end0 + p.cols[1] / 64;
    const uint segment = group < end0 ? 0 : group < end1 ? 1 : 2;
    const uint tile = group - (segment == 0 ? 0 : segment == 1 ? end0 : end1);
    const uint column = tile * 64 + tid;
    const uint groups = p.input_size / 32;
    float sums[1];
    if (p.fmt[segment] == GGUF_FMT_IQ4XS && p.input_size >= 4096 && p.input_size > p.cols[segment])
        sums[0] = native_iq4_vector_dot(input, segment == 0 ? a0 : segment == 1 ? b0 : c0,
            segment == 0 ? am : segment == 1 ? bm : cm, column, p.input_size, 0, groups);
    else
        column_dot_any<1>(p.fmt[segment], input, 1, p.input_size,
            segment == 0 ? a0 : segment == 1 ? b0 : c0,
            segment == 0 ? a1 : segment == 1 ? b1 : c1,
            segment == 0 ? am : segment == 1 ? bm : cm,
            column, 0, groups, table, tid, 64, sums);
    const uint index = p.offset[segment] + column;
    output[index] = T(sums[0]);
}

#define NATIVE_GGUF_VECTOR(T, suffix) \
kernel void moe_native_vector_##suffix(device const bfloat* input [[buffer(0)]], \
    device uchar* a0 [[buffer(1)]], device uchar* a1 [[buffer(2)]], device uchar* am [[buffer(3)]], \
    device uchar* b0 [[buffer(4)]], device uchar* b1 [[buffer(5)]], device uchar* bm [[buffer(6)]], \
    device uchar* c0 [[buffer(7)]], device uchar* c1 [[buffer(8)]], device uchar* cm [[buffer(9)]], \
    device T* output [[buffer(10)]], \
    constant GgufDecodeFusedParams& p [[buffer(11)]], uint group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half2 table[kQuantPairTableEntries]; \
    native_gguf_vector(input, a0, a1, am, b0, b1, bm, c0, c1, cm, output, p, group, tid, table); \
}
NATIVE_GGUF_VECTOR(bfloat, bf16)
NATIVE_GGUF_VECTOR(float, f32)
#undef NATIVE_GGUF_VECTOR

template <class T, ushort Rows = 1>
inline void native_gguf_vector_combine(device const bfloat* input, device uchar* a0, device uchar* a1,
    device uchar* am, device uchar* b0, device uchar* b1, device uchar* bm, device uchar* c0,
    device uchar* c1, device uchar* cm, device T* output, constant GgufDecodeFusedParams& p,
    uint group, uint tid, uint split, uint lane, threadgroup half2* table, threadgroup float* partials, uint rows) {
    const uint end0 = p.cols[0] / 32, end1 = end0 + p.cols[1] / 32;
    const uint segment = group < end0 ? 0 : group < end1 ? 1 : 2;
    const uint tile = group - (segment == 0 ? 0 : segment == 1 ? end0 : end1);
    const uint column = tile * 32 + lane;
    const uint per = p.input_size / 32 / p.splits;
    auto w0 = segment == 0 ? a0 : segment == 1 ? b0 : c0;
    auto w1 = segment == 0 ? a1 : segment == 1 ? b1 : c1;
    auto meta = segment == 0 ? am : segment == 1 ? bm : cm;
    float sums[Rows];
    if (Rows == 1 && p.fmt[segment] == GGUF_FMT_IQ4XS && p.input_size >= 4096 && p.input_size > p.cols[segment])
        sums[0] = native_iq4_vector_dot(input, w0, meta, column, p.input_size, split * per, (split + 1) * per);
    else column_dot_any<Rows>(p.fmt[segment], input, rows, p.input_size, w0, w1, meta, column,
        split * per, (split + 1) * per, table, tid, p.splits * 32, sums);
    for (ushort row = 0; row < Rows && row < rows; ++row)
        partials[(row * p.splits + split) * 32 + lane] = sums[row];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!split) for (ushort row = 0; row < Rows && row < rows; ++row) {
        float value = 0.f;
        for (uint i = 0; i < p.splits; ++i) value += partials[(row * p.splits + i) * 32 + lane];
        output[ulong(row) * p.out_stride + p.offset[segment] + column] = T(value);
    }
}

#define NATIVE_GGUF_VECTOR_COMBINE(T, suffix, Rows) \
kernel void moe_native_vector_combine_##suffix(device const bfloat* input [[buffer(0)]], \
    device uchar* a0 [[buffer(1)]], device uchar* a1 [[buffer(2)]], device uchar* am [[buffer(3)]], \
    device uchar* b0 [[buffer(4)]], device uchar* b1 [[buffer(5)]], device uchar* bm [[buffer(6)]], \
    device uchar* c0 [[buffer(7)]], device uchar* c1 [[buffer(8)]], device uchar* cm [[buffer(9)]], \
    device T* output [[buffer(10)]], constant GgufDecodeFusedParams& p [[buffer(11)]], \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint split [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half2 table[kQuantPairTableEntries]; \
    threadgroup float partials[Rows * 8 * 32]; \
    native_gguf_vector_combine<T, Rows>(input, a0, a1, am, b0, b1, bm, c0, c1, cm, output, p, \
        group, tid, split, lane, table, partials, Rows); \
}
NATIVE_GGUF_VECTOR_COMBINE(bfloat, bf16, 1)
NATIVE_GGUF_VECTOR_COMBINE(float, f32, 1)
NATIVE_GGUF_VECTOR_COMBINE(bfloat, pair_bf16, 2)
NATIVE_GGUF_VECTOR_COMBINE(float, pair_f32, 2)
#undef NATIVE_GGUF_VECTOR_COMBINE

struct NativeGgufVectorParams {
    GgufDecodeFusedParams projection;
    uint rows;
};

#define NATIVE_GGUF_VECTOR_SMALL(T, suffix) \
kernel void moe_native_vector_small_##suffix(device const bfloat* input [[buffer(0)]], \
    device uchar* a0 [[buffer(1)]], device uchar* a1 [[buffer(2)]], device uchar* am [[buffer(3)]], \
    device uchar* b0 [[buffer(4)]], device uchar* b1 [[buffer(5)]], device uchar* bm [[buffer(6)]], \
    device uchar* c0 [[buffer(7)]], device uchar* c1 [[buffer(8)]], device uchar* cm [[buffer(9)]], \
    device T* output [[buffer(10)]], constant NativeGgufVectorParams& p [[buffer(11)]], \
    uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint split [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    threadgroup half2 table[kQuantPairTableEntries]; \
    threadgroup float partials[4 * 8 * 32]; \
    native_gguf_vector_combine<T, 4>(input, a0, a1, am, b0, b1, bm, c0, c1, cm, output, p.projection, \
        group, tid, split, lane, table, partials, p.rows); \
}
NATIVE_GGUF_VECTOR_SMALL(bfloat, bf16)
NATIVE_GGUF_VECTOR_SMALL(float, f32)
#undef NATIVE_GGUF_VECTOR_SMALL
