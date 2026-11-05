// Adapted from Strata src/kernels/cuda/dflash2.cu, Copyright (c) 2026
// Niko1221 and contributors, MIT: ../../../third_party/strata.LICENSE.
// Signed-byte expansion is shared by all activation columns. The Q8
// activation, F32 partial sums, XOR tree and BF16 output rounding are retained.
#include "dflash2.h"
#include "kernel_dispatch_trace.h"
#include "warp_sums.cuh"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdexcept>
namespace mfq::cuda::dflash2 {
namespace {
int padk(int k) {
    if (k < 1 || k % 32 || k > INT_MAX - 511)
        throw std::invalid_argument("Invalid Q4 K");
    return (k + 511) / 512 * 512;
}
int padn(int n) {
    if (n < 1 || n > INT_MAX - 31)
        throw std::invalid_argument("Invalid Q4 N");
    return (n + 31) / 32 * 32;
}
void checked() {
    auto e = cudaGetLastError();
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
template <bool Reverse>
__global__ void retile_q4(const unsigned char *src, unsigned char *dst, int n, int k, int pn,
                          int pk) {
    const int64_t b = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int nb = k / 32, np = pk / 32;
    if (b >= int64_t(Reverse ? n : pn) * (Reverse ? nb : np))
        return;
    const int64_t row = b / (Reverse ? nb : np);
    const int blk = b % (Reverse ? nb : np), r = row % 8, st = blk / 16, bi = blk % 16;
    const auto scale = row / 8 * 8 * np + int64_t(st) * 128 + r * 16 + bi;
    const auto code = row / 8 * 8 * (pk / 2) + int64_t(st) * 2048 + r * 256 + bi * 16;
    if constexpr (Reverse) {
        const auto *ds = src + int64_t(pn) * pk / 2;
        dst[b * 18] = ds[scale * 2];
        dst[b * 18 + 1] = ds[scale * 2 + 1];
        for (int half = 0; half < 2; ++half)
            for (int j = 0; j < 8; ++j) {
                const auto v = src[code + half * 8 + j];
                const int i = (j / 4) * 8 + j % 4;
                if (!half) {
                    dst[b * 18 + 2 + i] = v & 15;
                    dst[b * 18 + 2 + i + 4] = v >> 4;
                } else {
                    dst[b * 18 + 2 + i] |= (v & 15) << 4;
                    dst[b * 18 + 2 + i + 4] |= v & 240;
                }
            }
    } else {
        auto *ds = dst + int64_t(pn) * pk / 2;
        const bool valid = row < n && blk < nb;
        const auto off = (row * nb + blk) * 18;
        ds[scale * 2] = valid ? src[off] : 0;
        ds[scale * 2 + 1] = valid ? src[off + 1] : 0;
        for (int half = 0; half < 2; ++half)
            for (int j = 0; j < 8; ++j) {
                const int i = (j / 4) * 8 + j % 4;
                const unsigned a = valid ? (src[off + 2 + i] >> (4 * half)) & 15 : 8;
                const unsigned c = valid ? (src[off + 2 + i + 4] >> (4 * half)) & 15 : 8;
                dst[code + half * 8 + j] = a | (c << 4);
            }
    }
}
struct Q4Segment {
    const int8_t *q;
    const half *d;
    float *y;
    int n, row0;
};
struct Q4Segments {
    Q4Segment s[3];
    int n;
};
__device__ __forceinline__ int dot4q4(int a, int b, int c) {
#if __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    for (int j = 0; j < 4; ++j)
        c += int(static_cast<signed char>(unsigned(a) >> (j * 8))) *
             int(static_cast<signed char>(unsigned(b) >> (j * 8)));
    return c;
#endif
}
__device__ __forceinline__ uint2 load_nibbles(const uint2 *p) {
    uint2 v;
    asm("ld.global.cg.v2.u32 {%0,%1},[%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
    return v;
}
__device__ __forceinline__ int center(unsigned q) { return int((q + 0x78787878u) ^ 0x80808080u); }
template <int Tokens, int Rows>
__device__ __forceinline__ void project_q4_body(Q4Segments segs, const int8_t *xq, const float *xd,
                                                int pk, int ncols, int group, int first) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, row_g = (group * 4 + warp) * Rows;
    Q4Segment s = segs.s[0];
    if (segs.n > 1 && row_g >= segs.s[1].row0)
        s = segs.s[1];
    if (segs.n > 2 && row_g >= segs.s[2].row0)
        s = segs.s[2];
    const int row0 = row_g - s.row0, nb = pk / 32, steps = pk / 512;
    const auto *wq =
        reinterpret_cast<const uint2 *>(s.q + size_t(row0 / 8) * 8 * (pk / 2) + (row0 % 8) * 256) +
        lane;
    const auto *wd = s.d + size_t(row0 / 8) * 8 * nb + (row0 % 8) * 16 + (lane >> 1);
    const auto *xv = reinterpret_cast<const int4 *>(xq + size_t(first) * pk) + lane;
    xd += size_t(first) * nb;
    float sums[Rows][Tokens]{};
#pragma unroll 2
    for (int st = 0; st < steps; ++st) {
        int4 w[Rows];
        float dw[Rows];
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
            const auto q = load_nibbles(wq + size_t(st) * 256 + r * 32);
            w[r] = make_int4(center(q.x & 0x0f0f0f0f), center((q.x >> 4) & 0x0f0f0f0f),
                             center(q.y & 0x0f0f0f0f), center((q.y >> 4) & 0x0f0f0f0f));
            dw[r] = __half2float(wd[size_t(st) * 128 + r * 16]);
        }
#pragma unroll
        for (int t = 0; t < Tokens; ++t) {
            const auto a = xv[size_t(t) * (pk / 16) + st * 32];
            const float scale = xd[size_t(t) * nb + st * 16 + (lane >> 1)];
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
                int v = dot4q4(w[r].x, a.x, 0);
                v = dot4q4(w[r].y, a.y, v);
                v = dot4q4(w[r].z, a.z, v);
                v = dot4q4(w[r].w, a.w, v);
                sums[r][t] = fmaf(dw[r] * scale, float(v), sums[r][t]);
            }
        }
    }
    // Share the XOR tree across row/token outputs. The FP32 addition order
    // is unchanged; successive levels retain half the outputs per lane.
    constexpr int Outputs = Rows * Tokens;
    static_assert(Outputs <= 32 && (Outputs & (Outputs - 1)) == 0);
    float values[Outputs];
#pragma unroll
    for (int i = 0; i < Outputs; ++i)
        values[i] = sums[i / Tokens][i % Tokens];
    const float value = warp_sums::reduce<Outputs>(values, lane);
    const int index = warp_sums::index<Outputs>(lane), r = index / Tokens, t = index % Tokens;
    if (lane == warp_sums::owner<Outputs>(index) && t < ncols && row0 + r < s.n)
        s.y[size_t(first + t) * s.n + row0 + r] = __bfloat162float(__float2bfloat16_rn(value));
}
template <int Tokens, int Rows>
__global__ __launch_bounds__(128) void project_q4(Q4Segments segs, const int8_t *codes,
                                                  const float *scales, int pk, int tokens) {
    project_q4_body<Tokens, Rows>(segs, codes, scales, pk, tokens, blockIdx.x, 0);
}
template <int Rows>
__global__ __launch_bounds__(128) void project_q4_many(Q4Segments segs, const int8_t *codes,
                                                       const float *scales, int pk, int tokens,
                                                       int groups) {
    const int tiles = (tokens + 7) / 8;
    for (int64_t work = blockIdx.x; work < int64_t(groups) * tiles; work += gridDim.x) {
        const int first = int(work % tiles) * 8;
        project_q4_body<8, Rows>(segs, codes, scales, pk, min(8, tokens - first), int(work / tiles),
                                 first);
    }
}
} // namespace
void q4_tile(const void *src, void *dst, int n, int k, cudaStream_t stream) {
    if (!src || !dst || src == dst)
        throw std::invalid_argument("Q4 tile requires distinct input and output storage");
    const int pk = padk(k), pn = padn(n);
    const auto count = int64_t(pn) * (pk / 32);
    retile_q4<false><<<unsigned((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const unsigned char *>(src), static_cast<unsigned char *>(dst), n, k, pn, pk);
    checked();
}
void q4_untile(const void *src, void *dst, int n, int k, cudaStream_t stream) {
    if (!src || !dst || src == dst)
        throw std::invalid_argument("Q4 tile requires distinct input and output storage");
    const int pk = padk(k), pn = padn(n);
    const auto count = int64_t(n) * (k / 32);
    retile_q4<true><<<unsigned((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const unsigned char *>(src), static_cast<unsigned char *>(dst), n, k, pn, pk);
    checked();
}
void q4_gemm(const Q8TiledMatrix *matrices, int count, const void *codes, const float *scales,
             int tokens, int k, cudaStream_t stream) {
    const int pk = padk(k);
    if (count < 1 || count > 3 || tokens < 1 || tokens > INT_MAX - 7 || !matrices || !codes ||
        !scales)
        throw std::invalid_argument("Invalid Q4 projection");
    Q4Segments segs{};
    segs.n = count;
    int rows = 0;
    for (int i = 0; i < count; ++i) {
        const auto &m = matrices[i];
        if (!m.weight || !m.output || m.padded_rows != padn(m.rows) ||
            rows > INT_MAX - m.padded_rows)
            throw std::invalid_argument("Q4 row padding differs");
        const auto *q = static_cast<const int8_t *>(m.weight);
        segs.s[i] = {q, reinterpret_cast<const half *>(q + int64_t(m.padded_rows) * pk / 2),
                     m.output, m.rows, rows};
        rows += m.padded_rows;
    }
    const auto launch = [&]<int R>() {
        if (tokens > 8) {
            const int groups = rows / (4 * R);
            const auto blocks = std::min<int64_t>(int64_t(groups) * ((tokens + 7) / 8), 65535);
            project_q4_many<R><<<unsigned(blocks), 128, 0, stream>>>(
                segs, static_cast<const int8_t *>(codes), scales, pk, tokens, groups);
        } else {
            const auto run = [&]<int T>() {
                constexpr int N = T > 4 ? R : (T > 2 ? 4 : 8);
                const auto function = project_q4<T, N>;
                function<<<rows / (4 * N), 128, 0, stream>>>(
                    segs, static_cast<const int8_t *>(codes), scales, pk, tokens);
                if (dispatch_trace::enabled())
                    dispatch_trace::record(
                        stream, "dflash2_q4_projection", reinterpret_cast<const void *>(function),
                        "M=" + std::to_string(tokens) + " K=" + std::to_string(k) +
                            " rows_per_warp=" + std::to_string(N) + " signed_weight_reuse=1" +
                            " butterfly_outputs=" + std::to_string(N * T));
            };
            if (tokens == 1)
                run.template operator()<1>();
            else if (tokens == 2)
                run.template operator()<2>();
            else if (tokens <= 4)
                run.template operator()<4>();
            else
                run.template operator()<8>();
        }
    };
    launch.template operator()<4>();
    checked();
}
} // namespace mfq::cuda::dflash2
