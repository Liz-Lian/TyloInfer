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
    device uchar* c1, device uchar* cm, device T* output, device float* partials,
    constant GgufDecodeFusedParams& p, uint2 group, uint tid, threadgroup half2* table) {
    const uint end0 = p.cols[0] / 64, end1 = end0 + p.cols[1] / 64;
    const uint segment = group.x < end0 ? 0 : group.x < end1 ? 1 : 2;
    const uint tile = group.x - (segment == 0 ? 0 : segment == 1 ? end0 : end1);
    const uint column = tile * 64 + tid;
    const uint groups = p.input_size / 32, per = groups / p.splits;
    float sums[1];
    if (p.fmt[segment] == GGUF_FMT_IQ4XS && p.input_size >= 4096 && p.input_size > p.cols[segment])
        sums[0] = native_iq4_vector_dot(input, segment == 0 ? a0 : segment == 1 ? b0 : c0,
            segment == 0 ? am : segment == 1 ? bm : cm, column, p.input_size, group.y * per, (group.y + 1) * per);
    else
        column_dot_any<1>(p.fmt[segment], input, 1, p.input_size,
            segment == 0 ? a0 : segment == 1 ? b0 : c0,
            segment == 0 ? a1 : segment == 1 ? b1 : c1,
            segment == 0 ? am : segment == 1 ? bm : cm,
            column, group.y * per, (group.y + 1) * per, table, tid, 64, sums);
    const uint index = p.offset[segment] + column;
    if (p.splits == 1) output[index] = T(sums[0]);
    else partials[ulong(group.y) * p.out_stride + index] = sums[0];
}

#define NATIVE_GGUF_VECTOR(T, suffix) \
kernel void moe_native_vector_##suffix(device const bfloat* input [[buffer(0)]], \
    device uchar* a0 [[buffer(1)]], device uchar* a1 [[buffer(2)]], device uchar* am [[buffer(3)]], \
    device uchar* b0 [[buffer(4)]], device uchar* b1 [[buffer(5)]], device uchar* bm [[buffer(6)]], \
    device uchar* c0 [[buffer(7)]], device uchar* c1 [[buffer(8)]], device uchar* cm [[buffer(9)]], \
    device T* output [[buffer(10)]], device float* partials [[buffer(11)]], \
    constant GgufDecodeFusedParams& p [[buffer(12)]], uint2 group [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]]) { \
    threadgroup half2 table[kQuantPairTableEntries]; \
    native_gguf_vector(input, a0, a1, am, b0, b1, bm, c0, c1, cm, output, partials, p, group, tid, table); \
} \
kernel void moe_native_vector_reduce_##suffix(device const float* partials [[buffer(0)]], \
    device T* output [[buffer(1)]], constant uint2& p [[buffer(2)]], uint index [[thread_position_in_grid]]) { \
    if (index >= p.x) return; \
    float value = 0.0f; \
    for (uint split = 0; split < p.y; ++split) value += partials[ulong(split) * p.x + index]; \
    output[index] = T(value); \
}
NATIVE_GGUF_VECTOR(bfloat, bf16)
NATIVE_GGUF_VECTOR(float, f32)
#undef NATIVE_GGUF_VECTOR
