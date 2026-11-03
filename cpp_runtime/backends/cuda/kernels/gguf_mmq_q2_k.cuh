// Q2_K tile math derived from llama.cpp MMQ (MIT; ../../../components/ggml/LICENSE).
#pragma once
#include "mmq.cuh"

template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void native_q2_fold_load(const char *__restrict__ x, const char* x_gate, int *__restrict__ x_tile, const int kbx0,
                                                           const int i_max, const int stride) {
#if defined(TURING_MMA_AVAILABLE)

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(type, J, fallback, GGML_PREC_Q8) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(type, J, fallback, GGML_PREC_Q8);
    constexpr int sram_stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    int *x_qs = (int *)x_tile;
    half2 *x_dm = (half2 *)(x_qs + 2 * MMQ_TILE_NE_K);
#else
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    int *x_qs = (int *)x_tile;
    half2 *x_dm = (half2 *)(x_qs + txs.qs);
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)

    constexpr int threads_per_row = MMQ_ITER_K / (4 * QR2_K);
    constexpr int nrows = ggml_cuda_get_physical_warp_size() / threads_per_row;
    const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
    for (int i0 = 0; i0 < I; i0 += nrows * nwarps) {
        int i = i0 + threadIdx.y * nrows + threadIdx.x / threads_per_row;

        if (fallback) {
            i = min(i, i_max);
        }

        const block_q2_K *bxi = (const block_q2_K *)x + kbx0 + i * stride;

        const int x_ql_0 = get_int_b2(bxi->qs, kqsx);

#pragma unroll
        for (int l = 0; l < QR2_K; ++l) {
            const int k = (kqsx / 8) * 32 + l * 8 + kqsx % 8;

            const int x_qs_k = ((x_ql_0 >> (2 * l)) & 0x03030303) * (bxi->scales[k / 4] & 15);

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
            x_qs[i * sram_stride + k] = x_qs_k;
#else
            x_qs[i * (2 * MMQ_TILE_NE_K + 1) + k] = x_qs_k;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        }

        const int sc_m = bxi->scales[kqsx];
#ifdef FAST_FP16_AVAILABLE
        const half2 x_dm_ik = __hmul2(bxi->dm, make_half2(1, sc_m >> 4));
#else
        const float2 bxi_dmf = __half22float2(bxi->dm);
        const half2 x_dm_ik = make_half2(bxi_dmf.x, bxi_dmf.y * (sc_m >> 4));
#endif // FAST_FP16_AVAILABLE

#if defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
        x_dm[i * sram_stride + kqsx] = x_dm_ik;
#else
        x_dm[i * (MMQ_TILE_NE_K + 1) + kqsx] = x_dm_ik;
#endif // defined(AMD_MFMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    }

#else
    ggml_cuda_mmq_load_tiles_q2_K<type, J, fallback>(x, x_gate, x_tile, kbx0, i_max, stride);
#endif
}

// Q2_K q*scale <=45, so folding its four-bit scale into the INT8 code is
// exact. Wider MMA accumulation then applies the common block scale once.
// The offset path retains the imported Q8 partial-sum semantics.
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void
native_q2_fold_dot(const int *__restrict__ x, const int *__restrict__ y, float *__restrict__ sum, int k00) {
#if defined(TURING_MMA_AVAILABLE)
    using A32 = tile<16, 8, int>;
    using B32 = tile<8, 8, int>;
    using C = tile<16, 8, int>;
    using A16 = tile<16, 4, int>;
    using B16 = tile<8, 4, int>;
    constexpr int stride = ggml_cuda_mmq_get_sram_stride(type, J, fallback, GGML_PREC_Q8);
    constexpr int rows = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback, GGML_PREC_Q8), ntx = rows / C::I;
    y += (threadIdx.y % ntx) * (C::J * MMQ_TILE_Y_K);
    const auto *xdm = reinterpret_cast<const half2 *>(x + 2 * MMQ_TILE_NE_K);
    const auto *yds = reinterpret_cast<const half2 *>(y);
    const auto *yq = y + 4;
    const int i0 = (threadIdx.y / ntx) * rows;
#pragma unroll
    for (int chunk = 0; chunk < MMQ_TILE_NE_K; chunk += 16) {
        A32 a[ntx][2];
        float d[ntx][C::ne / 2];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int piece = 0; piece < 2; ++piece)
                load_ldmatrix(a[n][piece], x + (i0 + n * A32::I) * stride + k00 + chunk + 8 * piece, stride);
#pragma unroll
            for (int l = 0; l < C::ne / 2; ++l) {
                const int i = i0 + n * A32::I + C::get_i(2 * l);
                d[n][l] = __low2float(xdm[i * stride]);
            }
        }
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * C::J) {
            B32 b[2];
            float db[C::ne / 2];
            load_generic(b[0], yq + j0 * MMQ_TILE_Y_K + chunk, MMQ_TILE_Y_K);
            load_generic(b[1], yq + j0 * MMQ_TILE_Y_K + chunk + 8, MMQ_TILE_Y_K);
#pragma unroll
            for (int l = 0; l < C::ne / 2; ++l) {
                const int j = j0 + C::get_j(l);
                const auto scales = __half22float2(yds[j * MMQ_TILE_Y_K]);
                db[l] = chunk < MMQ_TILE_NE_K / 2 ? scales.x : scales.y;
            }
#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                C c;
                mma(c, a[n][0], b[0]);
                mma(c, a[n][1], b[1]);
#pragma unroll
                for (int l = 0; l < C::ne; ++l)
                    sum[(j0 / C::J + n) * C::ne + l] += float(c.x[l]) * d[n][l / 2] * db[l % 2];
            }
// The first six 16-element activation sums are stored directly.
#pragma unroll
            for (int sub = 0; sub < 16 && chunk + sub < MMQ_TILE_NE_K * 3 / 4; sub += 4) {
                float sb[C::ne / 2];
#pragma unroll
                for (int l = 0; l < C::ne / 2; ++l) {
                    const int j = j0 + C::get_j(l);
                    const auto s = __half22float2(yds[j * MMQ_TILE_Y_K + 1 + (chunk + sub) / QI8_1]);
                    sb[l] = ((chunk + sub) / 4) % 2 ? s.y : s.x;
                }
#pragma unroll
                for (int n = 0; n < ntx; ++n) {
#pragma unroll
                    for (int l = 0; l < C::ne; ++l) {
                        const int i = i0 + n * A32::I + C::get_i(l);
                        sum[(j0 / C::J + n) * C::ne + l] -=
                            __high2float(xdm[i * stride + (k00 + chunk + sub) / 4]) * sb[l % 2];
                    }
                }
            }
            if (chunk + 16 > MMQ_TILE_NE_K * 3 / 4) {
                A16 one;
                for (int l = 0; l < A16::ne; ++l)
                    one.x[l] = 0x01010101;
                const auto *last = reinterpret_cast<const B16 *>(&b[1]);
#pragma unroll
                for (int sub = 0; sub < 2; ++sub) {
                    C offset;
                    mma(offset, one, last[sub]);
#pragma unroll
                    for (int n = 0; n < ntx; ++n) {
#pragma unroll
                        for (int l = 0; l < C::ne; ++l) {
                            const int i = i0 + n * A32::I + C::get_i(l);
                            sum[(j0 / C::J + n) * C::ne + l] -=
                                float(offset.x[l]) *
                                __high2float(xdm[i * stride + (k00 + chunk + 8 + 4 * sub) / 4]) * db[l % 2];
                        }
                    }
                }
            }
        }
    }
#else
    ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma<type, J, fallback>(x, y, sum, k00);
#endif
}

// Specialize only the tile math. Both imported schedulers keep their original
// traversal, guards and fixup; the upstream files and fingerprint are intact.
// m16n8k32 INT8 MMA is native starting at SM80.
#if defined(TURING_MMA_AVAILABLE) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define NATIVE_Q2_FOLD_TILE(J, F)                                                                            \
    template <>                                                                                              \
    constexpr __device__ ggml_cuda_mmq_util_funcs ggml_cuda_mmq_get_util_funcs<GGML_TYPE_Q2_K, J, F, GGML_PREC_Q8, false>() {     \
        if constexpr (ggml_cuda_mmq_get_config(GGML_TYPE_Q2_K, J, F, GGML_PREC_Q8).type == GGML_TYPE_COUNT)                \
            return {0, nullptr, nullptr, nullptr};                                                           \
        else                                                                                                 \
            return {-1, native_q2_fold_load<GGML_TYPE_Q2_K, J, F>, native_q2_fold_dot<GGML_TYPE_Q2_K, J, F>, \
                    ggml_cuda_mmq_write_back_mma<GGML_TYPE_Q2_K, J, F, GGML_PREC_Q8, false>};                                     \
    }
#define NATIVE_Q2_FOLD_BOTH(J) NATIVE_Q2_FOLD_TILE(J, false) NATIVE_Q2_FOLD_TILE(J, true)
NATIVE_Q2_FOLD_BOTH(8)
NATIVE_Q2_FOLD_BOTH(16) NATIVE_Q2_FOLD_BOTH(24) NATIVE_Q2_FOLD_BOTH(32) NATIVE_Q2_FOLD_BOTH(40)
    NATIVE_Q2_FOLD_BOTH(48) NATIVE_Q2_FOLD_BOTH(56) NATIVE_Q2_FOLD_BOTH(64) NATIVE_Q2_FOLD_BOTH(72)
        NATIVE_Q2_FOLD_BOTH(80) NATIVE_Q2_FOLD_BOTH(88) NATIVE_Q2_FOLD_BOTH(96) NATIVE_Q2_FOLD_BOTH(104)
            NATIVE_Q2_FOLD_BOTH(112) NATIVE_Q2_FOLD_BOTH(120) NATIVE_Q2_FOLD_BOTH(128)
#undef NATIVE_Q2_FOLD_BOTH
#undef NATIVE_Q2_FOLD_TILE
#endif
