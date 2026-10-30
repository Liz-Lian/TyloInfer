#ifndef TYLOINFER_GYRO_DECODE_H
#define TYLOINFER_GYRO_DECODE_H

static const uint16_t ggml_gyro_codebook[4096] = {
#include "../../../third_party/agention-gyro/codebook_f16.inc"
};

static void dequantize_gyro(const void * src, float * dst, int64_t count, unsigned bits, unsigned steps) {
    GGML_ASSERT(count % 128 == 0);
    const unsigned stream_bytes = steps * bits / 8, values = 128 / steps;
    const uint8_t * block = src;
    for (int64_t base = 0; base < count; base += 128, block += stream_bytes + 2) {
        ggml_fp16_t scale_bits;
        memcpy(&scale_bits, block, sizeof(scale_bits));
        const float scale = GGML_FP16_TO_FP32(scale_bits);
        const uint8_t * stream = block + 2;
        for (unsigned step = 0; step < steps; ++step) {
            uint32_t state;
            if (bits == 8 && steps == 32) {
                state = ((uint32_t) stream[(step + 31) % 32] << 8) | stream[step];
            } else {
                const unsigned offset = (steps - 1 - step) * bits, byte = offset / 8;
                const uint32_t window = stream[byte] | ((uint32_t) stream[(byte + 1) % stream_bytes] << 8) |
                    ((uint32_t) stream[(byte + 2) % stream_bytes] << 16);
                state = (window >> (offset % 8)) & 65535;
            }
            const uint32_t hash = (state & 32767) * UINT32_C(0x9e3779b1);
            for (unsigned j = 0; j < values; ++j) {
                const unsigned index = j < 2 ? hash >> 21 : (hash >> 10) & 2047;
                dst[base + step * values + j] = scale * GGML_FP16_TO_FP32(ggml_gyro_codebook[index * 2 + j % 2]);
            }
        }
    }
}

#define GGML_GYRO_DECODER(name, bits, steps) \
    static void dequantize_row_##name(const void * src, float * dst, int64_t count) { \
        dequantize_gyro(src, dst, count, bits, steps); \
    }
GGML_GYRO_DECODER(tq2_t, 8, 32)
GGML_GYRO_DECODER(tqk6, 6, 32)
GGML_GYRO_DECODER(tqk7, 7, 32)
GGML_GYRO_DECODER(tqp5, 5, 64)
GGML_GYRO_DECODER(tqp6, 6, 64)
GGML_GYRO_DECODER(tqp7, 7, 64)
GGML_GYRO_DECODER(tqk5, 5, 32)
GGML_GYRO_DECODER(tqk4, 4, 32)
#undef GGML_GYRO_DECODER

#endif
