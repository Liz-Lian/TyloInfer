
// Full / Grouped-Query Attention (ggml fattn.cu).
//
//   out[b,hq,tq,:] = softmax_tq( (q . k^T) * scale ) . v
//
// One block per query vector (b, hq, tq); blockDim = warp-rounded to >= D. q row in shared
// mem; per key s, online softmax (flash-v1 single-query): running max m, norm-sum l,
// accumulator O[tid]. One block_sum reduces the q . k_s dot and returns it to all threads.
// causal mask follows torch SDPA for Tq != Tk: query tq sees key s iff s <= tq + (Tk - T).
// q/k/v may be fp16 or fp32; dot/softmax accumulation stays fp32.

#include <cuda_runtime.h>
#include "attention_raw.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cfloat>
#include <climits>
#include <cstdlib>
#include <utility>

#include "reduce.cuh"

namespace wmma = nvcuda::wmma;

// Qwen3.5 full-attention prefill: FP16 GQA, head_dim=256, causal self-attention.
// QK is recomputed in the second pass so the full T x T score matrix is never materialized.
__global__ void attention_flash256_f16_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ out,
    int Hq, int Hk, int T, float scale)
{
    constexpr int D = 256;
    constexpr int QT = 128;
    constexpr int KT = 16;

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q0 = blockIdx.x * QT;
    const int bhq = blockIdx.y;
    const int hq = bhq % Hq;
    const int b = bhq / Hq;
    const int hk = hq / (Hq / Hk);

    extern __shared__ __align__(16) unsigned char smem[];
    half* q_s = reinterpret_cast<half*>(smem);
    half* kv_s = q_s + QT * D;
    float* scores = reinterpret_cast<float*>(kv_s + KT * D);
    half* probs = reinterpret_cast<half*>(scores + QT * KT);
    float* row_max = reinterpret_cast<float*>(probs + QT * KT);
    float* row_sum = row_max + QT;

    const size_t q_base = (size_t)bhq * T * D;
    const size_t kv_base = ((size_t)b * Hk + hk) * T * D;
    for (int i = tid; i < QT * D; i += blockDim.x) {
        const int qr = i / D;
        const int d = i - qr * D;
        q_s[i] = q0 + qr < T ? q[q_base + (size_t)(q0 + qr) * D + d] : __float2half(0.0f);
    }
    if (tid < QT) {
        row_max[tid] = -FLT_MAX;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int k0 = 0; k0 < T; k0 += KT) {
        for (int i = tid; i < KT * D; i += blockDim.x) {
            const int kr = i / D;
            const int d = i - kr * D;
            kv_s[i] = k0 + kr < T ? k[kv_base + (size_t)(k0 + kr) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        if (warp < 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            const int qr = warp;
            for (int d0 = 0; d0 < D; d0 += 16) {
                wmma::load_matrix_sync(a_frag, q_s + qr * 16 * D + d0, D);
                wmma::load_matrix_sync(b_frag, kv_s + d0, D);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + qr * 16 * KT, c_frag, KT, wmma::mem_row_major);
        }
        __syncthreads();

        if (warp < 8) {
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            const int r = warp * 16 + rr;
            const int qg = q0 + r;
            if (qg >= T) continue;
            const int kg = k0 + lane;
            float s = lane < KT && kg < T && kg <= qg ? scores[r * KT + lane] * scale : -FLT_MAX;
            const float tile_max = warp_max(s);
            const float old_max = row_max[r];
            const float new_max = fmaxf(old_max, tile_max);
            float add = s == -FLT_MAX ? 0.0f : expf(s - new_max);
            add = warp_sum(add);
            if (lane == 0) {
                const float old_scaled = old_max == -FLT_MAX ? 0.0f : row_sum[r] * expf(old_max - new_max);
                row_max[r] = new_max;
                row_sum[r] = old_scaled + add;
            }
        }
        }
        __syncthreads();
    }

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[8];
    #pragma unroll
    for (int i = 0; i < 8; ++i) wmma::fill_fragment(o_frag[i], 0.0f);

    for (int k0 = 0; k0 < T; k0 += KT) {
        for (int i = tid; i < KT * D; i += blockDim.x) {
            const int kr = i / D;
            const int d = i - kr * D;
            kv_s[i] = k0 + kr < T ? k[kv_base + (size_t)(k0 + kr) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        if (warp < 8) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
            wmma::fill_fragment(c_frag, 0.0f);
            const int qr = warp;
            for (int d0 = 0; d0 < D; d0 += 16) {
                wmma::load_matrix_sync(a_frag, q_s + qr * 16 * D + d0, D);
                wmma::load_matrix_sync(b_frag, kv_s + d0, D);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            wmma::store_matrix_sync(scores + qr * 16 * KT, c_frag, KT, wmma::mem_row_major);
        }
        __syncthreads();

        for (int i = tid; i < QT * KT; i += blockDim.x) {
            const int r = i / KT;
            const int c = i - r * KT;
            const int qg = q0 + r;
            const int kg = k0 + c;
            float p = 0.0f;
            if (qg < T && kg < T && kg <= qg && row_sum[r] > 0.0f) {
                p = expf(scores[i] * scale - row_max[r]) / row_sum[r];
            }
            probs[i] = __float2half_rn(p);
        }
        __syncthreads();

        for (int i = tid; i < D * KT; i += blockDim.x) {
            const int d = i / KT;
            const int kr = i - d * KT;
            kv_s[i] = k0 + kr < T ? v[kv_base + (size_t)(k0 + kr) * D + d] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> v_frag;
        const int qr = warp >> 1;
        const int parity = warp & 1;
        for (int kk = 0; kk < KT; kk += 16) {
            wmma::load_matrix_sync(p_frag, probs + qr * 16 * KT + kk, KT);
            #pragma unroll
            for (int fi = 0; fi < 8; ++fi) {
                const int ni = parity + fi * 2;
                wmma::load_matrix_sync(v_frag, kv_s + ni * 16 * KT + kk, KT);
                wmma::mma_sync(o_frag[fi], p_frag, v_frag, o_frag[fi]);
            }
        }
        __syncthreads();
    }

    float* out_s = reinterpret_cast<float*>(smem) + warp * 256;
    const int qr = warp >> 1;
    const int parity = warp & 1;
    #pragma unroll
    for (int fi = 0; fi < 8; ++fi) {
        const int ni = parity + fi * 2;
        wmma::store_matrix_sync(out_s, o_frag[fi], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = e / 16;
            const int c = e - r * 16;
            if (q0 + qr * 16 + r < T) {
                out[q_base + (size_t)(q0 + qr * 16 + r) * D + ni * 16 + c] = __float2half_rn(out_s[e]);
            }
        }
        __syncwarp();
    }
}



template <int BD, typename scalar_t>
__global__ void attention_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    scalar_t* __restrict__ out,
    int B, int Hq, int Hk, int T, int Tk, int D, int rep, float scale, int causal, int window)
{
    constexpr int NW = BD / 32;
    int qv = blockIdx.x;
    int tq = qv % T;
    int rem = qv / T;
    int hq = rem % Hq;
    int b = rem / Hq;
    int hk = hq / rep;
    int tid = threadIdx.x;

    extern __shared__ float buf[];
    float* q_s = buf;        // [D]

    if (tid < D) {
        q_s[tid] = (float)q[((size_t)b * Hq + hq) * T * D + (size_t)tq * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    const int offset = Tk - T;
    const int end = causal ? min(Tk, tq + offset + 1) : Tk;
    const int start = window > 0 ? max(0, end - window) : 0;

    for (int s = start; s < end; ++s) {
        float kd = (tid < D) ? (float)k[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        float dot = (tid < D) ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);

        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float fac = expf(m - m_new);
        float p = expf(score - m_new);
        float vd = (tid < D) ? (float)v[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }

    if (tid < D) {
        float lo = (l > 0.0f) ? l : 1.0f;
        out[((size_t)b * Hq + hq) * T * D + (size_t)tq * D + tid] = (scalar_t)(O / lo);
    }
}

template <int BD, typename scalar_t>
__global__ void attention_split_part_kernel(
    const scalar_t* __restrict__ q, const scalar_t* __restrict__ k, const scalar_t* __restrict__ v,
    float* __restrict__ partial_o, float* __restrict__ partial_m, float* __restrict__ partial_l,
    int B, int Hq, int Hk, int T, int Tk, int D, int rep, int parts,
    float scale, int causal, int window)
{
    constexpr int NW = BD / 32;
    int part = blockIdx.x % parts;
    int qv = blockIdx.x / parts;
    int tq = qv % T;
    int rem = qv / T;
    int hq = rem % Hq;
    int b = rem / Hq;
    int hk = hq / rep;
    int tid = threadIdx.x;
    const int offset = Tk - T;
    const int visible_end = causal ? min(Tk, tq + offset + 1) : Tk;
    const int visible_start = window > 0 ? max(0, visible_end - window) : 0;
    const int visible = visible_end - visible_start;
    const int start = visible_start + (int)(((int64_t)visible * part) / parts);
    const int end = visible_start + (int)(((int64_t)visible * (part + 1)) / parts);

    extern __shared__ float q_s[];
    if (tid < D) {
        q_s[tid] = (float)q[((size_t)b * Hq + hq) * T * D + (size_t)tq * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    int seen = 0;

    for (int s = start; s < end; ++s) {
        float kd = (tid < D) ? (float)k[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        float dot = (tid < D) ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float fac = expf(m - m_new);
        float p = expf(score - m_new);
        float vd = (tid < D) ? (float)v[((size_t)b * Hk + hk) * Tk * D + (size_t)s * D + tid] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
        seen = 1;
    }

    size_t po = ((size_t)qv * parts + part) * D;
    if (tid < D) {
        partial_o[po + tid] = seen ? O : 0.0f;
    }
    if (tid == 0) {
        partial_m[(size_t)qv * parts + part] = seen ? m : -1e30f;
        partial_l[(size_t)qv * parts + part] = seen ? l : 0.0f;
    }
}

template <int BD, typename scalar_t>
__global__ void attention_split_reduce_kernel(
    const float* __restrict__ partial_o, const float* __restrict__ partial_m, const float* __restrict__ partial_l,
    scalar_t* __restrict__ out, int total, int parts, int D)
{
    int qv = blockIdx.x;
    int tid = threadIdx.x;
    if (qv >= total) {
        return;
    }
    float m = -1e30f;
    for (int p = 0; p < parts; ++p) {
        m = fmaxf(m, partial_m[(size_t)qv * parts + p]);
    }
    float l = 0.0f;
    float O = 0.0f;
    for (int p = 0; p < parts; ++p) {
        float lp = partial_l[(size_t)qv * parts + p];
        float w = lp > 0.0f ? expf(partial_m[(size_t)qv * parts + p] - m) : 0.0f;
        l += w * lp;
        if (tid < D) {
            O += w * partial_o[((size_t)qv * parts + p) * D + tid];
        }
    }
    if (tid < D) {
        float lo = (l > 0.0f) ? l : 1.0f;
        out[(size_t)qv * D + tid] = (scalar_t)(O / lo);
    }
}







template <int BD, typename scalar_t>
__global__ void attention_cache_decode_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    scalar_t* __restrict__ out,
    int B, int Hq, int Hk, int max_seq, int D, int rep, float scale,
    int maximum_cache_position)
{
    constexpr int NW = BD / 32;
    int qv = blockIdx.x;
    int hq = qv % Hq;
    int b = qv / Hq;
    int hk = hq / rep;
    int tid = threadIdx.x;
    int Tk = (int)seq_len[b];
    if (Tk < 0) {
        Tk = 0;
    }
    if (Tk > max_seq) {
        Tk = max_seq;
    }
    if (maximum_cache_position >= 0 &&
            Tk - 1 > maximum_cache_position) {
        if (tid < D) {
            out[((size_t)b * Hq + hq) * D + tid] = (scalar_t)0.0f;
        }
        return;
    }

    extern __shared__ float q_s[];
    if (tid < D) {
        q_s[tid] = (float)q[((size_t)b * Hq + hq) * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;

    for (int s = 0; s < Tk; ++s) {
        float kd = (tid < D) ? (float)k[((size_t)b * Hk + hk) * max_seq * D + (size_t)s * D + tid] : 0.0f;
        float dot = (tid < D) ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        float score = dot * scale;
        float m_new = fmaxf(m, score);
        float fac = expf(m - m_new);
        float p = expf(score - m_new);
        float vd = (tid < D) ? (float)v[((size_t)b * Hk + hk) * max_seq * D + (size_t)s * D + tid] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }

    if (tid < D) {
        float lo = (l > 0.0f) ? l : 1.0f;
        out[((size_t)b * Hq + hq) * D + tid] = (scalar_t)(O / lo);
    }
}



__device__ __forceinline__ int attention_decode_active_parts(
    int token_count, int launch_parts, int dynamic_parts)
{
    if (!dynamic_parts) {
        return launch_parts;
    }
    const int cache_position = token_count > 0 ? token_count - 1 : 0;
    if (cache_position < 192) {
        return 1;
    }
    const int active = (cache_position + 127) / 128;
    return active < launch_parts ? active : launch_parts;
}

template <int BD, typename scalar_t>
__global__ void attention_cache_decode_split_part_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int B, int Hq, int Hk, int max_seq, int D, int rep,
    int parts, int workspace_parts, float scale, int dynamic_parts)
{
    constexpr int NW = BD / 32;
    const int part = blockIdx.x % parts;
    const int qv = blockIdx.x / parts;
    const int hq = qv % Hq;
    const int b = qv / Hq;
    const int hk = hq / rep;
    const int tid = threadIdx.x;
    int Tk = (int)seq_len[b];
    Tk = Tk < 0 ? 0 : (Tk > max_seq ? max_seq : Tk);
    const int active_parts = attention_decode_active_parts(
        Tk, parts, dynamic_parts);
    const size_t stat_idx = (size_t)qv * workspace_parts + part;
    if (part >= active_parts) {
        if (tid < D) {
            partial_o[stat_idx * D + tid] = 0.0f;
        }
        if (tid == 0) {
            partial_m[stat_idx] = -1e30f;
            partial_l[stat_idx] = 0.0f;
        }
        return;
    }
    const int start = (int)(((int64_t)Tk * part) / active_parts);
    const int end = (int)(((int64_t)Tk * (part + 1)) / active_parts);

    extern __shared__ float q_s[];
    if (tid < D) {
        q_s[tid] = (float)q[(size_t)qv * D + tid];
    }
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    for (int s = start; s < end; ++s) {
        const size_t kv_idx = ((size_t)b * Hk + hk) * max_seq * D + (size_t)s * D + tid;
        const float kd = tid < D ? (float)k[kv_idx] : 0.0f;
        float dot = tid < D ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float fac = expf(m - m_new);
        const float p = expf(score - m_new);
        const float vd = tid < D ? (float)v[kv_idx] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }

    if (tid < D) {
        partial_o[stat_idx * D + tid] = O;
    }
    if (tid == 0) {
        partial_m[stat_idx] = start < end ? m : -1e30f;
        partial_l[stat_idx] = start < end ? l : 0.0f;
    }
}

template <typename scalar_t>
__global__ void attention_cache_decode_split_gqa4_d128_part_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int B, int Hq, int Hk, int max_seq,
    int parts, int workspace_parts, float scale, int dynamic_parts)
{
    constexpr int D = 128;
    constexpr int REP = 4;
    constexpr int NW = D / 32;
    const int part = blockIdx.x % parts;
    const int kv = blockIdx.x / parts;
    const int hk = kv % Hk;
    const int b = kv / Hk;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int hq0 = hk * REP;
    int Tk = (int)seq_len[b];
    Tk = Tk < 0 ? 0 : (Tk > max_seq ? max_seq : Tk);
    const int active_parts = attention_decode_active_parts(
        Tk, parts, dynamic_parts);
    if (part >= active_parts) {
        #pragma unroll
        for (int r = 0; r < REP; ++r) {
            const int qv_idx = b * Hq + hq0 + r;
            const size_t stat_idx =
                (size_t)qv_idx * workspace_parts + part;
            partial_o[stat_idx * D + tid] = 0.0f;
            if (tid == 0) {
                partial_m[stat_idx] = -1e30f;
                partial_l[stat_idx] = 0.0f;
            }
        }
        return;
    }
    const int start = (int)(((int64_t)Tk * part) / active_parts);
    const int end = (int)(((int64_t)Tk * (part + 1)) / active_parts);

    float qv[REP];
    float m[REP];
    float l[REP];
    float O[REP];
    #pragma unroll
    for (int r = 0; r < REP; ++r) {
        qv[r] = (float)q[((size_t)b * Hq + hq0 + r) * D + tid];
        m[r] = -1e30f;
        l[r] = 0.0f;
        O[r] = 0.0f;
    }

    __shared__ float warp_dot[REP][NW];
    __shared__ float score[REP];
    const size_t kv_base = ((size_t)b * Hk + hk) * max_seq * D;
    for (int s = start; s < end; ++s) {
        const size_t idx = kv_base + (size_t)s * D + tid;
        const float kd = (float)k[idx];
        const float vd = (float)v[idx];
        #pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float dot = warp_sum(qv[r] * kd);
            if (lane == 0) {
                warp_dot[r][warp] = dot;
            }
        }
        __syncthreads();
        if (warp == 0) {
            #pragma unroll
            for (int r = 0; r < REP; ++r) {
                float dot = lane < NW ? warp_dot[r][lane] : 0.0f;
                dot = warp_sum(dot);
                if (lane == 0) {
                    score[r] = dot * scale;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < REP; ++r) {
            const float m_new = fmaxf(m[r], score[r]);
            const float fac = expf(m[r] - m_new);
            const float p = expf(score[r] - m_new);
            O[r] = O[r] * fac + p * vd;
            l[r] = l[r] * fac + p;
            m[r] = m_new;
        }
    }

    #pragma unroll
    for (int r = 0; r < REP; ++r) {
        const int qv_idx = b * Hq + hq0 + r;
        const size_t stat_idx = (size_t)qv_idx * workspace_parts + part;
        partial_o[stat_idx * D + tid] = O[r];
        if (tid == 0) {
            partial_m[stat_idx] = start < end ? m[r] : -1e30f;
            partial_l[stat_idx] = start < end ? l[r] : 0.0f;
        }
    }
}

template <int BD, typename scalar_t>
__global__ void attention_cache_decode_split_reduce_kernel(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_m,
    const float* __restrict__ partial_l,
    const int64_t* __restrict__ seq_len,
    scalar_t* __restrict__ out,
    int total, int Hq, int parts, int workspace_parts,
    int D, int dynamic_parts)
{
    const int qv = blockIdx.x;
    const int tid = threadIdx.x;
    if (qv >= total) {
        return;
    }

    const int b = qv / Hq;
    const int token_count = (int)seq_len[b];
    const int active_parts = attention_decode_active_parts(
        token_count, parts, dynamic_parts);
    float m = -1e30f;
    for (int p = 0; p < active_parts; ++p) {
        m = fmaxf(m, partial_m[(size_t)qv * workspace_parts + p]);
    }
    float l = 0.0f;
    float O = 0.0f;
    for (int p = 0; p < active_parts; ++p) {
        const size_t stat_idx = (size_t)qv * workspace_parts + p;
        const float lp = partial_l[stat_idx];
        const float w = lp > 0.0f ? expf(partial_m[stat_idx] - m) : 0.0f;
        l += w * lp;
        if (tid < D) {
            O += w * partial_o[stat_idx * D + tid];
        }
    }
    if (tid < D) {
        out[(size_t)qv * D + tid] = (scalar_t)(O / (l > 0.0f ? l : 1.0f));
    }
}









template <int BD, typename scalar_t>
__global__ void attention_cache_swa_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    scalar_t* __restrict__ out,
    int Hq, int Hk, int T, int capacity, int D, int rep,
    int window, float scale)
{
    constexpr int NW = BD / 32;
    const int qv = blockIdx.x;
    const int tq = qv % T;
    const int rem = qv / T;
    const int hq = rem % Hq;
    const int b = rem / Hq;
    const int hk = hq / rep;
    const int tid = threadIdx.x;
    const int64_t length = seq_len[b] > 0 ? seq_len[b] : 0;
    const int64_t qpos = length - T + tq;
    const int64_t end = qpos + 1 < length ? qpos + 1 : length;
    const int64_t start = end > window ? end - window : 0;

    extern __shared__ float q_s[];
    if (tid < D) q_s[tid] = (float)q[(size_t)qv * D + tid];
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    for (int64_t pos = start; pos < end; ++pos) {
        const int slot = (int)(pos % capacity);
        const size_t kv_idx = ((size_t)b * Hk + hk) * capacity * D + (size_t)slot * D + tid;
        const float kd = tid < D ? (float)k[kv_idx] : 0.0f;
        float dot = tid < D ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float fac = expf(m - m_new);
        const float p = expf(score - m_new);
        const float vd = tid < D ? (float)v[kv_idx] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }
    if (tid < D) out[(size_t)qv * D + tid] = (scalar_t)(O / (l > 0.0f ? l : 1.0f));
}

template <int BD, typename scalar_t>
__global__ void attention_cache_swa_split_part_kernel(
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int64_t* __restrict__ seq_len,
    float* __restrict__ partial_o,
    float* __restrict__ partial_m,
    float* __restrict__ partial_l,
    int Hq, int Hk, int T, int capacity, int D, int rep,
    int window, int parts, float scale)
{
    constexpr int NW = BD / 32;
    const int part = blockIdx.x % parts;
    const int qv = blockIdx.x / parts;
    const int tq = qv % T;
    const int rem = qv / T;
    const int hq = rem % Hq;
    const int b = rem / Hq;
    const int hk = hq / rep;
    const int tid = threadIdx.x;
    const int64_t length = seq_len[b] > 0 ? seq_len[b] : 0;
    const int64_t qpos = length - T + tq;
    const int64_t visible_end = qpos + 1 < length ? qpos + 1 : length;
    const int64_t visible_start = visible_end > window ? visible_end - window : 0;
    const int64_t visible = visible_end - visible_start;
    const int64_t start = visible_start + visible * part / parts;
    const int64_t end = visible_start + visible * (part + 1) / parts;

    extern __shared__ float q_s[];
    if (tid < D) q_s[tid] = (float)q[(size_t)qv * D + tid];
    __syncthreads();

    float m = -1e30f;
    float l = 0.0f;
    float O = 0.0f;
    for (int64_t pos = start; pos < end; ++pos) {
        const int slot = (int)(pos % capacity);
        const size_t kv_idx = ((size_t)b * Hk + hk) * capacity * D + (size_t)slot * D + tid;
        const float kd = tid < D ? (float)k[kv_idx] : 0.0f;
        float dot = tid < D ? q_s[tid] * kd : 0.0f;
        dot = block_sum<NW>(dot);
        const float score = dot * scale;
        const float m_new = fmaxf(m, score);
        const float fac = expf(m - m_new);
        const float p = expf(score - m_new);
        const float vd = tid < D ? (float)v[kv_idx] : 0.0f;
        O = O * fac + p * vd;
        l = l * fac + p;
        m = m_new;
    }
    const size_t stat = (size_t)qv * parts + part;
    if (tid < D) partial_o[stat * D + tid] = O;
    if (tid == 0) {
        partial_m[stat] = start < end ? m : -1e30f;
        partial_l[stat] = start < end ? l : 0.0f;
    }
}







namespace tyloi::cuda::attention_raw {
void flash256(const Args& a,cudaStream_t stream) {
    const dim3 grid((a.tokens+127)/128,a.batch*a.heads);
    constexpr int shared=96*1024;
    cudaFuncSetAttribute(attention_flash256_f16_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared);
    attention_flash256_f16_kernel<<<grid,512,shared,stream>>>(static_cast<const half*>(a.q),static_cast<const half*>(a.k),static_cast<const half*>(a.v),static_cast<half*>(a.out),a.heads,a.kv_heads,a.tokens,a.scale);
}
template<int BD,class T> void full_t(const Args& a,cudaStream_t s) {
    const int total=a.batch*a.heads*a.tokens, rep=a.heads/a.kv_heads, shared=a.width*sizeof(float);
    auto q=static_cast<const T*>(a.q);auto k=static_cast<const T*>(a.k);auto v=static_cast<const T*>(a.v);auto out=static_cast<T*>(a.out);
    if(a.parts>1) {
        attention_split_part_kernel<BD,T><<<total*a.parts,BD,shared,s>>>(q,k,v,a.partial_o,a.partial_m,a.partial_l,a.batch,a.heads,a.kv_heads,a.tokens,a.keys,a.width,rep,a.parts,a.scale,int(a.causal),a.window);
        attention_split_reduce_kernel<BD,T><<<total,BD,0,s>>>(a.partial_o,a.partial_m,a.partial_l,out,total,a.parts,a.width);
    } else attention_kernel<BD,T><<<total,BD,shared,s>>>(q,k,v,out,a.batch,a.heads,a.kv_heads,a.tokens,a.keys,a.width,rep,a.scale,int(a.causal),a.window);
}
template<int BD,class T> void decode_t(const Args& a,cudaStream_t s) {
    const int total=a.batch*a.heads,rep=a.heads/a.kv_heads,shared=a.width*sizeof(float);
    attention_cache_decode_kernel<BD,T><<<total,BD,shared,s>>>(static_cast<const T*>(a.q),static_cast<const T*>(a.k),static_cast<const T*>(a.v),a.lengths,static_cast<T*>(a.out),a.batch,a.heads,a.kv_heads,a.keys,a.width,rep,a.scale,a.maximum_cache_position);
}
template<int BD,class T> void split_t(const Args& a,cudaStream_t s) {
    const int total=a.batch*a.heads,rep=a.heads/a.kv_heads,shared=a.width*sizeof(float);
    auto q=static_cast<const T*>(a.q);auto k=static_cast<const T*>(a.k);auto v=static_cast<const T*>(a.v);
    if(a.width==128 && rep==4)
        attention_cache_decode_split_gqa4_d128_part_kernel<T><<<a.batch*a.kv_heads*a.parts,128,0,s>>>(q,k,v,a.lengths,a.partial_o,a.partial_m,a.partial_l,a.batch,a.heads,a.kv_heads,a.keys,a.parts,a.workspace_parts,a.scale,int(a.dynamic_parts));
    else attention_cache_decode_split_part_kernel<BD,T><<<total*a.parts,BD,shared,s>>>(q,k,v,a.lengths,a.partial_o,a.partial_m,a.partial_l,a.batch,a.heads,a.kv_heads,a.keys,a.width,rep,a.parts,a.workspace_parts,a.scale,int(a.dynamic_parts));
    attention_cache_decode_split_reduce_kernel<BD,T><<<total,BD,0,s>>>(a.partial_o,a.partial_m,a.partial_l,a.lengths,static_cast<T*>(a.out),total,a.heads,a.parts,a.workspace_parts,a.width,int(a.dynamic_parts));
}
template<int BD,class T> void swa_t(const Args& a,cudaStream_t s) {
    const int total=a.batch*a.heads*a.tokens,rep=a.heads/a.kv_heads,shared=a.width*sizeof(float);
    auto q=static_cast<const T*>(a.q);auto k=static_cast<const T*>(a.k);auto v=static_cast<const T*>(a.v);auto out=static_cast<T*>(a.out);
    if(a.parts==1) attention_cache_swa_kernel<BD,T><<<total,BD,shared,s>>>(q,k,v,a.lengths,out,a.heads,a.kv_heads,a.tokens,a.keys,a.width,rep,a.window,a.scale);
    else {
        attention_cache_swa_split_part_kernel<BD,T><<<total*a.parts,BD,shared,s>>>(q,k,v,a.lengths,a.partial_o,a.partial_m,a.partial_l,a.heads,a.kv_heads,a.tokens,a.keys,a.width,rep,a.window,a.parts,a.scale);
        attention_split_reduce_kernel<BD,T><<<total,BD,0,s>>>(a.partial_o,a.partial_m,a.partial_l,out,total,a.parts,a.width);
    }
}
#define ATTENTION_DISPATCH(F) \
    if(a.type==NumericType::f16) { if(a.width<=64)F<64,half>(a,s);else if(a.width<=128)F<128,half>(a,s);else if(a.width<=256)F<256,half>(a,s);else F<512,half>(a,s); } \
    else if(a.type==NumericType::bf16) { if(a.width<=64)F<64,__nv_bfloat16>(a,s);else if(a.width<=128)F<128,__nv_bfloat16>(a,s);else if(a.width<=256)F<256,__nv_bfloat16>(a,s);else F<512,__nv_bfloat16>(a,s); } \
    else { if(a.width<=64)F<64,float>(a,s);else if(a.width<=128)F<128,float>(a,s);else if(a.width<=256)F<256,float>(a,s);else F<512,float>(a,s); }
void full(const Args& a,cudaStream_t s) { ATTENTION_DISPATCH(full_t) }
void decode(const Args& a,cudaStream_t s) { ATTENTION_DISPATCH(decode_t) }
void decode_split(const Args& a,cudaStream_t s) { ATTENTION_DISPATCH(split_t) }
void cache_swa(const Args& a,cudaStream_t s) { ATTENTION_DISPATCH(swa_t) }
#undef ATTENTION_DISPATCH
}
