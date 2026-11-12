#pragma once

#ifndef TYLOINFER_FATTN_KERNEL_ONLY
#define TYLOINFER_FATTN_KERNEL_ONLY
#define TYLOINFER_SPARSE_ATTN_UNDEF_KERNEL_ONLY
#endif
#include "mfq_fattn_mma_f16.cuh"
#ifdef TYLOINFER_SPARSE_ATTN_UNDEF_KERNEL_ONLY
#undef TYLOINFER_SPARSE_ATTN_UNDEF_KERNEL_ONLY
#undef TYLOINFER_FATTN_KERNEL_ONLY
#endif

#include "sparse_mma_raw.h"


#include <algorithm>
#include <cstdint>
#include <limits>

namespace mfq_sparse_attn_mma {



struct RuntimeInput {
    const void* positions = nullptr;
    int64_t query_batch_stride = 0;
    int64_t query_head_stride = 0;
    int64_t query_token_stride = 0;
    int64_t query_column_stride = 0;
    int causal_columns = 0;
    bool half_query = false;
    bool wide_positions = false;
    bool direct_query = false;
    tyloi::cuda::sparse_mma_raw::Hot hot{};
};

inline uint3 sparse_fastdiv_values(uint32_t divisor) {
    uint32_t shift = 0;
    while (shift < 32 && (uint32_t{1} << shift) < divisor) {
        ++shift;
    }
    const uint32_t multiplier = static_cast<uint32_t>(
        (uint64_t{1} << 32) * ((uint64_t{1} << shift) - divisor) /
        divisor + 1);
    return make_uint3(multiplier, shift, divisor);
}

// Shared selected-token attention kernel. Model-specific indexers own the
// selection order and optional validity mask; this kernel owns KV gathering,
// MMA, online softmax and stream-K reduction.
template<int DKQ, int DV, int ncols1, int ncols2, bool V_is_K_view,
         bool needs_fixup, bool is_fixup, bool use_hot>
static __device__ __forceinline__ void process_sparse_attention_tile(
    const void * __restrict__ q,
    const half * __restrict__ k,
    const half * __restrict__ v,
    const int * __restrict__ indices,
    const half * __restrict__ mask,
    const float * __restrict__ sinks,
    float * __restrict__ out,
    float2 * __restrict__ meta,
    float scale,
    int M,
    int heads,
    int kv_heads,
    int max_seq,
    int selected,
    uint3 ne01,
    int64_t work,
    int64_t work_per_sequence,
    int iter_k,
    int iter_j,
    int iter_z_gqa,
    int kb0_start,
    int kb0_stop,
    RuntimeInput input) {
    constexpr int ncols = ncols1 * ncols2;
    constexpr int nthreads =
        ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols);
    constexpr int nwarps = nthreads / 32;
    constexpr bool use_logit_softcap = false;
    constexpr bool use_indirect = true;

    const int gqa_ratio = heads / kv_heads;
    const int sequence = static_cast<int>(work / work_per_sequence);
    const int64_t sequence_work = work -
        static_cast<int64_t>(sequence) * work_per_sequence;
    const int z_kv = static_cast<int>(sequence_work /
        (static_cast<int64_t>(iter_k) * iter_j * iter_z_gqa));
    const int64_t head_work = sequence_work -
        static_cast<int64_t>(z_kv) * iter_k * iter_j * iter_z_gqa;
    const int zt_gqa = static_cast<int>(head_work /
        (static_cast<int64_t>(iter_k) * iter_j));
    const int64_t query_work = head_work -
        static_cast<int64_t>(zt_gqa) * iter_k * iter_j;
    const int jt = static_cast<int>(query_work / iter_k);
    const int zt_q = z_kv * gqa_ratio + zt_gqa * ncols2;

    const float2 * q_f2 = input.direct_query ? nullptr : reinterpret_cast<const float2 *>(q) +
        (static_cast<int64_t>(sequence) * heads * M +
         static_cast<int64_t>(zt_q) * M) * (DKQ / 2);
    const int64_t physical_seq=use_hot && input.hot.cold_mapping?input.hot.cold_cells:max_seq;
    const half2 * k_h2 = reinterpret_cast<const half2 *>(k) +
        (static_cast<int64_t>(sequence) * kv_heads * physical_seq +
         static_cast<int64_t>(z_kv) * physical_seq) * (DKQ / 2);
    const half2 * v_h2 = V_is_K_view
        ? k_h2
        : reinterpret_cast<const half2 *>(v) +
            (static_cast<int64_t>(sequence) * kv_heads * physical_seq +
             static_cast<int64_t>(z_kv) * physical_seq) * (DV / 2);
    float2 * dst = reinterpret_cast<float2 *>(out) +
        (static_cast<int64_t>(sequence) * M * heads + zt_q) * (DV / 2);
    const int * row_indices = indices ? indices +
        (static_cast<int64_t>(sequence) * M + jt) * selected : nullptr;
    const half * sequence_mask = mask == nullptr ? nullptr :
        mask + static_cast<int64_t>(sequence) * M * selected;
    const float * tile_sinks = sinks == nullptr ? nullptr : sinks + zt_q;
    mfq_fattn_runtime_input tile_input{};
    if (input.direct_query) {
        const int64_t query_offset = int64_t(sequence)*input.query_batch_stride +
            int64_t(zt_q)*input.query_head_stride;
        tile_input.query = static_cast<const char*>(q) + query_offset*(input.half_query ? sizeof(half) : sizeof(float));
        tile_input.query_token_stride = input.query_token_stride;
        tile_input.query_head_stride = input.query_head_stride;
        tile_input.query_column_stride = input.query_column_stride;
        tile_input.half_query = input.half_query;
    }
    if (input.positions) {
        int64_t end = input.wide_positions ? static_cast<const int64_t*>(input.positions)[jt]
            : static_cast<const int32_t*>(input.positions)[jt];
        end = end < -1 ? -1 : end;
        end = end >= max_seq ? max_seq - 1 : end;
        end = end >= input.causal_columns ? input.causal_columns - 1 : end;
        tile_input.causal_query_offset = static_cast<int>(end) - jt;
        tile_input.causal_row_count = static_cast<int>(end + 1);
    }

    if constexpr(use_hot) {
        tile_input.hot_keys=reinterpret_cast<const half2*>(input.hot.keys)+(int64_t(sequence)*kv_heads+z_kv)*input.hot.cells*(DKQ/2);
        tile_input.hot_values=reinterpret_cast<const half2*>(input.hot.values)+(int64_t(sequence)*kv_heads+z_kv)*input.hot.cells*(DV/2);
        tile_input.hot_mapping=input.hot.mapping;
        tile_input.cold_mapping=input.hot.cold_mapping;
    }
    flash_attn_ext_f16_process_tile<
        DKQ, DV, ncols1, ncols2, nwarps,
        use_logit_softcap, V_is_K_view, needs_fixup, is_fixup,
        use_indirect, use_hot>(
        q_f2, k_h2, v_h2, sequence_mask, tile_sinks, dst, meta,
        scale, 1.0f, 0.0f, ne01, heads, gqa_ratio, selected,
        DKQ / 2, M * (DKQ / 2), DKQ / 2,
        V_is_K_view ? DKQ / 2 : DV / 2,
        sequence_mask == nullptr ? 0 : selected,
        jt, zt_gqa, kb0_start, kb0_stop, row_indices,
        input.positions != nullptr, 0, tile_input);
}

template<int DKQ, int DV, int ncols1, int ncols2, bool V_is_K_view, bool use_hot = false>
__launch_bounds__(
    ggml_cuda_fattn_mma_get_nthreads(DKQ, DV, ncols1 * ncols2),
    ggml_cuda_fattn_mma_get_occupancy(DKQ, DV, ncols1 * ncols2))
__global__ void sparse_attention_kernel(
    const void * __restrict__ q,
    const half * __restrict__ k,
    const half * __restrict__ v,
    const int * __restrict__ indices,
    const half * __restrict__ mask,
    const float * __restrict__ sinks,
    float * __restrict__ out,
    float2 * __restrict__ meta,
    float scale,
    int B,
    int M,
    int heads,
    int kv_heads,
    int max_seq,
    int selected,
    uint3 ne01,
    RuntimeInput input) {
#if defined(FLASH_ATTN_AVAILABLE) && defined(TURING_MMA_AVAILABLE)
    static_assert(ncols1 == 1,
        "selected-token tiles require one index row per query tile");
    static_assert(DKQ % 2 == 0 && DV % 2 == 0, "attention widths must be even");
    static_assert(!V_is_K_view || DV <= DKQ, "V view exceeds K width");

    constexpr int ncols = ncols1 * ncols2;
    constexpr int nbatch_fa =
        ggml_cuda_fattn_mma_get_nbatch_fa(DKQ, DV, ncols);
    const int gqa_ratio = heads / kv_heads;
    // The positive inputs and divisors make each quotient no larger than its
    // original int input; widen before adding the rounding term.
    const int iter_k = static_cast<int>((static_cast<int64_t>(selected) + nbatch_fa - 1) / nbatch_fa);
    const int iter_j = static_cast<int>((static_cast<int64_t>(M) + ncols1 - 1) / ncols1);
    const int iter_z_gqa = static_cast<int>((static_cast<int64_t>(gqa_ratio) + ncols2 - 1) / ncols2);
    const int64_t work_per_sequence =
        static_cast<int64_t>(iter_k) * iter_j * iter_z_gqa * kv_heads;
    const int64_t total_work = work_per_sequence * B;
    // Split the product so block index times total work cannot overflow int64.
    const int64_t quotient = total_work / gridDim.x;
    const int64_t remainder = total_work % gridDim.x;
    const int64_t start_block = blockIdx.x;
    const int64_t end_block = start_block + 1;
    int64_t kbc = start_block * quotient + start_block * remainder / gridDim.x;
    const int64_t kbc_stop = end_block * quotient + end_block * remainder / gridDim.x;

    int kb0_start = static_cast<int>(kbc % iter_k);
    int kb0_stop = static_cast<int>(min(static_cast<int64_t>(iter_k),
        static_cast<int64_t>(kb0_start) + kbc_stop - kbc));

    while (kbc < kbc_stop && kb0_stop == iter_k) {
        if (kb0_start == 0) {
            process_sparse_attention_tile<
                DKQ, DV, ncols1, ncols2, V_is_K_view, false, false, use_hot>(
                q, k, v, indices, mask, sinks, out, meta, scale,
                M, heads, kv_heads, max_seq, selected, ne01, kbc,
                work_per_sequence, iter_k, iter_j, iter_z_gqa,
                kb0_start, kb0_stop, input);
        } else {
            process_sparse_attention_tile<
                DKQ, DV, ncols1, ncols2, V_is_K_view, true, false, use_hot>(
                q, k, v, indices, mask, sinks, out, meta, scale,
                M, heads, kv_heads, max_seq, selected, ne01, kbc,
                work_per_sequence, iter_k, iter_j, iter_z_gqa,
                kb0_start, kb0_stop, input);
        }
        kbc += iter_k;
        kbc -= kbc % iter_k;
        kb0_start = 0;
        kb0_stop = static_cast<int>(min(static_cast<int64_t>(iter_k), kbc_stop - kbc));
    }

    if (kbc < kbc_stop) {
        process_sparse_attention_tile<
            DKQ, DV, ncols1, ncols2, V_is_K_view, false, true, use_hot>(
            q, k, v, indices, mask, sinks, out, meta, scale,
            M, heads, kv_heads, max_seq, selected, ne01, kbc,
            work_per_sequence, iter_k, iter_j, iter_z_gqa,
            kb0_start, kb0_stop, input);
    }
#else
    (void)q; (void)k; (void)v; (void)indices; (void)mask; (void)sinks;
    (void)out; (void)meta; (void)scale; (void)B; (void)M; (void)heads;
    (void)kv_heads; (void)max_seq; (void)selected; (void)ne01; (void)input;
#endif
}


} // namespace mfq_sparse_attn_mma
