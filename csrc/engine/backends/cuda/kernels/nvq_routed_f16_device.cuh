#pragma once
constexpr int kNvqMoeF16PoolTileN = 64;
constexpr int kNvqMoeF16HeteroTileN = 128;
constexpr int kNvqMoeF16AllFormats = 0;
constexpr int kNvqMoeF16StandardFormats = 1;
constexpr int kNvqMoeF16ExtendedFormats = 2;
constexpr int kNvqMoeF16LegacyFormats = 3;

template <int GROUPS_PER_CHUNK>
struct NvqMoeF16KLayout {
    static constexpr int kTileK = kGroupSize * GROUPS_PER_CHUNK;
    static_assert(kTileK % 16 == 0);
    static constexpr int kStrideK = kTileK + 8;
};

template <int BM, int BN, int GROUPS_PER_CHUNK>
struct __align__(16) NvqMoeF16SharedStorage {
    static constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    static constexpr int kOperandBytes =
        (BN + BM) * kStrideK * sizeof(__half);
    static constexpr int kOutputBytes = BM >= 128
        ? (BM / 64) * BN * 16 * sizeof(float)
        : (BN / 16) * BM * 16 * sizeof(float);
    static constexpr int kBytes =
        kOperandBytes > kOutputBytes ? kOperandBytes : kOutputBytes;
    uint8_t bytes[kBytes];
    int32_t pair_rows[BM];
    int32_t source_rows[BM];
};

static_assert(sizeof(NvqMoeF16SharedStorage<64, 128, 4>) <= 48 * 1024);
static_assert(sizeof(NvqMoeF16SharedStorage<128, 128, 2>) <= 48 * 1024);
static_assert(sizeof(NvqMoeF16SharedStorage<128, 64, 4>) <= 48 * 1024);

template <bool ASYNC_COPY, int BM, int GROUPS_PER_CHUNK,
          int THREAD_WARPS>
__device__ __forceinline__ void nvq_moe_load_activation_tile(
    const __half * x,
    const int32_t * source_rows,
    __half * activation_tile,
    int k_base,
    int K,
    int tid) {
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    constexpr int kTileK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kTileK;
    constexpr int kThreads = THREAD_WARPS * 32;
    constexpr int kVectorWidth = 8;
    static_assert(kTileK % kVectorWidth == 0);
    static_assert(kThreads % BM == 0);
    constexpr int kVectorsPerRow = kTileK / kVectorWidth;
    constexpr int kLanesPerRow = kThreads / BM;
    const int m_local = tid / kLanesPerRow;
    const int vector_lane = tid - m_local * kLanesPerRow;
    const int source_row = source_rows[m_local];
#pragma unroll
    for (int vector_local = vector_lane;
         vector_local < kVectorsPerRow;
         vector_local += kLanesPerRow) {
        const int k_local = vector_local * kVectorWidth;
        const int k = k_base + k_local;
        __half * destination =
            activation_tile + m_local * kStrideK + k_local;
        if (source_row < 0 || k >= K) {
            *reinterpret_cast<int4 *>(destination) =
                make_int4(0, 0, 0, 0);
        } else if ((K & 7) == 0 && k + 7 < K) {
            const __half * source =
                x + static_cast<int64_t>(source_row) * K + k;
            if constexpr (ASYNC_COPY) {
                tyloi::cuda_detail::copy_16_async(destination, source);
            } else {
                *reinterpret_cast<int4 *>(destination) =
                    *reinterpret_cast<const int4 *>(source);
            }
        } else {
#pragma unroll
            for (int element = 0; element < kVectorWidth; ++element) {
                destination[element] = k + element < K
                    ? x[static_cast<int64_t>(source_row) * K + k + element]
                    : __float2half(0.0f);
            }
        }
    }
}

template <int FORMAT, int BM, int BN, int GROUPS_PER_CHUNK,
          int THREAD_WARPS, bool ASYNC_ACTIVATION = false>
__device__ __forceinline__ void nvq_moe_grouped_f16_task(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    const int32_t * ids_dst,
    __half * output,
    __half * weight_tile,
    __half * activation_tile,
    float * output_tile,
    int32_t * pair_rows,
    int32_t * source_rows,
    int routes,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input,
    int local_expert,
    int first,
    int last,
    int n0) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    static_assert(BM == 16 || BM == 32 || BM == 64 || BM == 128 || BM == 256);
    static_assert(BN == 64 || BN == 128);
    static_assert(GROUPS_PER_CHUNK == 2 || GROUPS_PER_CHUNK == 4);
    constexpr int kMFragments = BM / 16;
    constexpr int kNFragments = BN / 16;
    constexpr int kWarps = THREAD_WARPS;
    constexpr int kRowTeams = BM >= 128 ? BM / 64 : 1;
    constexpr int kComputeWarps = kRowTeams * kNFragments;
    static_assert(kWarps >= kComputeWarps);
    constexpr int kAccumulatorFragments = kMFragments / kRowTeams;
    constexpr int kThreads = kWarps * 32;
    constexpr int kTileK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kTileK;
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    if (tid < BM) {
        const int compact = first + tid;
        const int pair_index = compact < last ? ids_dst[compact] : -1;
        pair_rows[tid] = pair_index;
        source_rows[tid] = pair_index < 0
            ? -1 : (routed_input ? pair_index : pair_index / routes);
    }
    __syncthreads();
    FragmentC accumulators[kAccumulatorFragments];
#pragma unroll
    for (int fragment = 0;
         fragment < kAccumulatorFragments;
         ++fragment) {
        wmma::fill_fragment(accumulators[fragment], 0.0f);
    }
    const int chunks =
        (ng + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * GROUPS_PER_CHUNK;
        const int k_base = group_base * kGroupSize;
        if constexpr (ASYNC_ACTIVATION) {
            nvq_moe_load_activation_tile<
                true, BM, GROUPS_PER_CHUNK, THREAD_WARPS>(
                x, source_rows, activation_tile, k_base, K, tid);
            tyloi::cuda_detail::copy_async_commit();
        }
        constexpr int kWeightStates = BN * GROUPS_PER_CHUNK;
        // BM256 doubles the activation readers. Keep both warp teams useful
        // during dequantization by sharing the three vectors of each group.
        constexpr int kDecodeLanes = BM == 256 ? 2 : 1;
        constexpr int kWeightThreads = kThreads / kDecodeLanes;
        const int segment_lane = BM == 256 ? tid / kWeightThreads : 0;
#pragma unroll
        for (int weight_index = BM == 256 ? tid % kWeightThreads : tid;
             weight_index < kWeightStates;
            weight_index += kWeightThreads) {
            const int row_local =
                weight_index / GROUPS_PER_CHUNK;
            const int group_local = weight_index -
                row_local * GROUPS_PER_CHUNK;
            const int local_row = n0 + row_local;
            const int row = local_expert * out_per_expert + local_row;
            const int group = group_base + group_local;
            const bool valid_weight =
                local_row < out_per_expert && group < ng;
            DenseNvqGroup dense_record;
            const bool dense=dense_nvq_format(FORMAT) && aux_nbytes==0;
            uint32_t state = 0;
            float scale = 0.0f;
            uint32_t packed_signs = 0;
            uint64_t group_exec64 = 0;
            uint32_t group_exec96[3] = {0, 0, 0};
            int group_delta = 1;
            const int8_t * bank = codebook;
            if (valid_weight) {
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                if constexpr(dense_nvq_format(FORMAT)) {
                    if(dense) {
                        dense_record=load_dense_nvq_group<FORMAT>(indices,indices_nbytes,row,group,nsign,ng);
                        state=dense_record.state();
                    }else state=load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                        sub_linear,sub_bits,ng);
                }else state=load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                scale = format_scale<FORMAT>(
                    neuron_scale[row], state, codebook);
                if constexpr (is_e8_format(FORMAT) || is_d4_format(FORMAT)) {
                    bank = active_codebook<FORMAT>(codebook, state);
                }
                if constexpr (FORMAT == kNvq2JscXLGroupExec) {
                    group_exec64 = load_group_exec64(
                        indices, row, group, ng);
                } else if constexpr (FORMAT == kNvq3JscLGroupExec) {
                    load_group_exec96_words(
                        indices, row, group, ng, group_exec96);
                } else if constexpr (
                    FORMAT == kNvq2 || FORMAT == kNvq2Jsc ||
                    FORMAT == kNvq2JscL || FORMAT == kNvq2JscXL ||
                    FORMAT == kNvq3 || FORMAT == kNvq3Jsc ||
                    FORMAT == kNvq3Jsc2 || FORMAT == kNvq3Jsc512 ||
                    FORMAT == kNvq3JscL) {
                    const int64_t sign_linear =
                        static_cast<int64_t>(row) * nsign + group * 3;
                    if(!dense)packed_signs = load_packed_sign_group3(
                        aux, sign_linear * 7, aux_nbytes);
                } else if constexpr (
                    FORMAT == kNvq1S || FORMAT == kNvq1L) {
                    group_delta = load_packed_bits(
                        aux, sub_linear, 1, aux_nbytes) ? -1 : 1;
                    if constexpr (FORMAT == kNvq1S) {
                        bank = codebook +
                            (group_delta < 0 ? kNvq1SBankBytes : 0);
                    }
                }
            }
#pragma unroll
            for (int segment_local = 0;
                 segment_local < kChunksPerGroup / 2;
                 ++segment_local) {
                if constexpr (BM == 256) {
                    if (segment_local % kDecodeLanes != segment_lane) continue;
                }
                int2 packed = make_int2(0, 0);
                if constexpr(dense_nvq_format(FORMAT))if(valid_weight && dense && group*3+segment_local<nsign)
                    packed=decode_dense_nvq_vector<FORMAT>(dense_record,bank,segment_local,sign_mode);
                if (valid_weight && !dense) {
                    const auto decoded = load_nvq_group_vec8<FORMAT>(
                        indices, indices_nbytes, aux, aux_nbytes,
                        codebook, bank, row, group, segment_local,
                        ng, nvec, nsign, sign_mode, state,
                        packed_signs, group_exec64, group_exec96,
                        group_delta);
                    packed = decoded.values;
                    if constexpr (FORMAT == kNvq1S) {
                        packed.x = nvq1_s_scale_delta4(
                            packed.x, decoded.delta);
                        packed.y = nvq1_s_scale_delta4(
                            packed.y, decoded.delta);
                    } else if constexpr (FORMAT == kNvq1L) {
                        packed.x = nvq1_l_scale_delta4(
                            packed.x, decoded.delta);
                        packed.y = nvq1_l_scale_delta4(
                            packed.y, decoded.delta);
                    }
                }
#pragma unroll
                for (int pair = 0; pair < 4; ++pair) {
                    const int word = pair < 2 ? packed.x : packed.y;
                    const int shift = (pair & 1) * 16;
                    const int value0 = static_cast<int>(
                        static_cast<int8_t>((word >> shift) & 0xff));
                    const int value1 = static_cast<int>(
                        static_cast<int8_t>(
                            (word >> (shift + 8)) & 0xff));
                    *reinterpret_cast<__half2 *>(
                        weight_tile + row_local * kStrideK +
                            group_local * kGroupSize +
                            segment_local * 8 + pair * 2) =
                        __halves2half2(
                            __float2half(
                                scale * static_cast<float>(value0)),
                            __float2half(
                                scale * static_cast<float>(value1)));
                }
            }
        }

        if constexpr (ASYNC_ACTIVATION) {
            tyloi::cuda_detail::copy_async_wait();
        } else {
            nvq_moe_load_activation_tile<
                false, BM, GROUPS_PER_CHUNK, THREAD_WARPS>(
                x, source_rows, activation_tile, k_base, K, tid);
        }
        __syncthreads();

        if (warp < kComputeWarps) {
            const int warp_n = BM >= 128 ? warp % kNFragments : warp;
            const int m_fragment_base = BM >= 128
                ? warp / kNFragments : 0;
#pragma unroll
            for (int k_local = 0;
                 k_local < kTileK;
                 k_local += 16) {
                if (BM < 64 || k_base + k_local < K) {
                    FragmentB weight_fragment;
                    wmma::load_matrix_sync(
                        weight_fragment,
                        weight_tile + warp_n * 16 * kStrideK + k_local,
                        kStrideK);
#pragma unroll
                    for (int fragment = 0;
                         fragment < kAccumulatorFragments;
                         ++fragment) {
                        const int m_fragment = m_fragment_base +
                            fragment * kRowTeams;
                        if constexpr (BM == 256) {
                            // Uniform within the warp; empty fragments have no
                            // output rows, and need no Tensor Core work.
                            if (m_fragment * 16 >= last - first) continue;
                        }
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            activation_tile +
                                m_fragment * 16 * kStrideK + k_local,
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[fragment], activation_fragment,
                            weight_fragment, accumulators[fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

    if constexpr (BM >= 128) {
        if (warp < kComputeWarps) {
            const int warp_n = warp % kNFragments;
            const int m_fragment_base = warp / kNFragments;
#pragma unroll
            for (int fragment = 0;
                 fragment < kAccumulatorFragments;
                 ++fragment) {
                const int m_fragment =
                    m_fragment_base + fragment * kRowTeams;
                wmma::store_matrix_sync(
                    output_tile + warp * 16 * 16,
                    accumulators[fragment],
                    16, wmma::mem_row_major);
                __syncwarp();
                for (int element = lane; element < 16 * 16; element += 32) {
                    const int m_local = m_fragment * 16 + element / 16;
                    const int n_local = element % 16;
                    const int local_row = n0 + warp_n * 16 + n_local;
                    const int pair_index = pair_rows[m_local];
                    if (pair_index >= 0 && local_row < out_per_expert) {
                        output[
                            static_cast<int64_t>(pair_index) * out_per_expert +
                            local_row] = __float2half(
                                output_tile[warp * 16 * 16 + element]);
                    }
                }
                __syncwarp();
            }
        }
    } else {
        if (warp < kNFragments) {
#pragma unroll
            for (int m_fragment = 0;
                 m_fragment < kMFragments;
                 ++m_fragment) {
                wmma::store_matrix_sync(
                    output_tile +
                        (warp * BM + m_fragment * 16) * 16,
                    accumulators[m_fragment],
                    16, wmma::mem_row_major);
            }
        }
        __syncthreads();
        if (warp < kNFragments) {
            for (int element = lane; element < BM * 16; element += 32) {
                const int m_local = element / 16;
                const int n_local = element % 16;
                const int local_row = n0 + warp * 16 + n_local;
                const int pair_index = pair_rows[m_local];
                if (pair_index >= 0 && local_row < out_per_expert) {
                    output[
                        static_cast<int64_t>(pair_index) * out_per_expert +
                        local_row] = __float2half(
                            output_tile[
                                (warp * BM + m_local) * 16 + n_local]);
                }
            }
        }
    }
    __syncthreads();
}

template <int FORMAT, int BM = 16>
__global__ void __launch_bounds__(256, 1) nvq_moe_grouped_f16_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const __half * x,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    int max_tiles,
    bool routed_input) {
    constexpr int BN = kNvqMoeF16PoolTileN;
    constexpr int GROUPS_PER_CHUNK = 4;
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    __shared__ NvqMoeF16SharedStorage<
        BM, BN, GROUPS_PER_CHUNK> shared;
    auto * weight_tile = reinterpret_cast<__half *>(shared.bytes);
    auto * activation_tile = weight_tile + BN * kStrideK;
    auto * output_tile = reinterpret_cast<float *>(shared.bytes);
    const int ntiles_n =
        (out_per_expert + BN - 1) / BN;
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(
            task - static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % (BM / 8) != 0) continue;
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) continue;
        const int first = expert_bounds[expert] + local_fine_tile * 8;
        const int last = min(
            first + BM, expert_bounds[expert + 1]);
        nvq_moe_grouped_f16_task<
            FORMAT, BM, BN, GROUPS_PER_CHUNK,
            (BM == 128 ? 16 : 8)>(
            indices, indices_nbytes, aux, aux_nbytes,
            sub_scale, sub_scale_nbytes, neuron_scale, codebook,
            x, ids_dst, output, weight_tile,
            activation_tile, output_tile, shared.pair_rows,
            shared.source_rows, routes,
            out_per_expert, K, ng, nvec, nsign, sub_bits, sign_mode,
            routed_input, local_expert, first, last,
            ntile * BN);
    }
}

template <int BM, int BN, int ROUTE_TILE_M, int GROUPS_PER_CHUNK,
          int FORMAT_GROUP, bool ASYNC_ACTIVATION = false>
__global__ void __launch_bounds__(
    (BM >= 128 ? BM / 64 : 1) * (BN / 16) * 32,
    BM == 128 && BN == 64 && GROUPS_PER_CHUNK == 2 &&
        FORMAT_GROUP != kNvqMoeF16AllFormats ? 3 :
    BM < 256 && (BN == 64 ||
        (FORMAT_GROUP != kNvqMoeF16AllFormats && GROUPS_PER_CHUNK == 2)
        ) ? 2 : 1)
nvq_moe_grouped_hetero_f16_kernel(
    const int64_t * weight_ptrs,
    const int64_t * weight_sizes,
    const int32_t * pool_params,
    const int32_t * expert_pool,
    const int32_t * expert_local,
    const __half * x,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_count,
    int out_per_expert,
    int K,
    int max_tiles,
    bool routed_input) {
    static_assert(BM % ROUTE_TILE_M == 0);
    constexpr int kStrideK =
        NvqMoeF16KLayout<GROUPS_PER_CHUNK>::kStrideK;
    extern __shared__ __align__(16) uint8_t shared_bytes[];
    auto & shared = *reinterpret_cast<NvqMoeF16SharedStorage<
        BM, BN, GROUPS_PER_CHUNK> *>(shared_bytes);
    auto * weight_tile = reinterpret_cast<__half *>(shared.bytes);
    auto * activation_tile = weight_tile + BN * kStrideK;
    auto * output_tile = reinterpret_cast<float *>(shared.bytes);
    const int ntiles_n =
        (out_per_expert + BN - 1) / BN;
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(
            task - static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % (BM / ROUTE_TILE_M) != 0) continue;
        const int pool = expert_pool[expert];
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(pool) >=
            static_cast<unsigned int>(pool_count) || local_expert < 0) {
            continue;
        }
        const int64_t * pointers = weight_ptrs + static_cast<size_t>(pool) * 5;
        const int64_t * sizes = weight_sizes + static_cast<size_t>(pool) * 3;
        const int32_t * params = pool_params + static_cast<size_t>(pool) * 7;
        const int pool_experts = params[0];
        if (local_expert >= pool_experts) continue;
        const int first =
            expert_bounds[expert] + local_fine_tile * ROUTE_TILE_M;
        const int last = min(
            first + BM, expert_bounds[expert + 1]);
        const uint8_t * indices = reinterpret_cast<const uint8_t *>(
            static_cast<uintptr_t>(pointers[0]));
        const uint8_t * aux = reinterpret_cast<const uint8_t *>(
            static_cast<uintptr_t>(pointers[1]));
        const uint8_t * sub_scale = reinterpret_cast<const uint8_t *>(
            static_cast<uintptr_t>(pointers[2]));
        const float * neuron_scale = reinterpret_cast<const float *>(
            static_cast<uintptr_t>(pointers[3]));
        const int8_t * codebook = reinterpret_cast<const int8_t *>(
            static_cast<uintptr_t>(pointers[4]));
        const int ng = params[1];
        const int nvec = params[2];
        const int nsign = params[3];
        const int sub_bits = params[4];
        const int sign_mode = params[5];
        const int format = params[6];

#define NVQ_MOE_HETERO_F16_CASE(FORMAT_VALUE)                                  \
        case FORMAT_VALUE:                                                      \
            nvq_moe_grouped_f16_task<                                           \
                FORMAT_VALUE, BM, BN, GROUPS_PER_CHUNK,                         \
                (BM >= 128 ? BM / 64 : 1) * (BN / 16), ASYNC_ACTIVATION>(        \
                indices, sizes[0], aux, sizes[1], sub_scale, sizes[2],          \
                neuron_scale, codebook, x, ids_dst, output,                     \
                weight_tile, activation_tile, output_tile,                     \
                shared.pair_rows, shared.source_rows, routes,                   \
                out_per_expert, K, ng, nvec,                                    \
                nsign, sub_bits, sign_mode, routed_input, local_expert, first,  \
                last, ntile * BN);                                              \
            break
        if constexpr (FORMAT_GROUP == kNvqMoeF16StandardFormats) {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq2);
                NVQ_MOE_HETERO_F16_CASE(kNvq3);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Exec);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscExec);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc2);
            }
        } else if constexpr (
            FORMAT_GROUP == kNvqMoeF16ExtendedFormats) {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc512);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXL);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXLGroupExec);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscLGroupExec);
            }
        } else if constexpr (
            FORMAT_GROUP == kNvqMoeF16LegacyFormats) {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq1L);
                NVQ_MOE_HETERO_F16_CASE(kNpq0L);
                NVQ_MOE_HETERO_F16_CASE(kNvq1S);
                NVQ_MOE_HETERO_F16_CASE(kNpq0S);
            }
        } else {
            switch (format) {
                NVQ_MOE_HETERO_F16_CASE(kNvq1L);
                NVQ_MOE_HETERO_F16_CASE(kNvq2);
                NVQ_MOE_HETERO_F16_CASE(kNvq3);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Exec);
                NVQ_MOE_HETERO_F16_CASE(kNvq2Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscExec);
                NVQ_MOE_HETERO_F16_CASE(kNpq0L);
                NVQ_MOE_HETERO_F16_CASE(kNvq1S);
                NVQ_MOE_HETERO_F16_CASE(kNpq0S);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc2);
                NVQ_MOE_HETERO_F16_CASE(kNvq3Jsc512);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXL);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscL);
                NVQ_MOE_HETERO_F16_CASE(kNvq2JscXLGroupExec);
                NVQ_MOE_HETERO_F16_CASE(kNvq3JscLGroupExec);
            }
        }
#undef NVQ_MOE_HETERO_F16_CASE
    }
}

template <int FORMAT>
__global__ void __launch_bounds__(256, 1) nepq_moe_grouped_f16_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const __half * x,
    const int32_t * expert_local,
    const int32_t * ids_dst,
    const int32_t * expert_bounds,
    const int32_t * tile_bounds,
    const int32_t * tile_experts,
    __half * output,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int max_tiles,
    bool routed_input) {
    constexpr int kTileM = 16;
    constexpr int kTileN = 64;
    constexpr int kGroupsPerChunk = kNepqGroupsPerSupergroup;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kNFragments = kTileN / 16;
    constexpr int kFineTilesPerTask = kTileM / 8;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int ntiles_n = (out_per_expert + kTileN - 1) / kTileN;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * ntiles_n;
    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][16][16];

    using FragmentA = wmma::fragment<wmma::matrix_a, 16, 16, 16,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, 16, 16, 16,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int fine_tile = static_cast<int>(task / ntiles_n);
        const int ntile = static_cast<int>(task -
            static_cast<int64_t>(fine_tile) * ntiles_n);
        if (fine_tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[fine_tile];
        const int local_fine_tile = fine_tile - tile_bounds[expert];
        if (local_fine_tile % kFineTilesPerTask != 0) continue;
        const int local_expert = expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) continue;
        const int first = expert_bounds[expert] + local_fine_tile * 8;
        const int last = min(first + kTileM, expert_bounds[expert + 1]);
        const int n0 = ntile * kTileN;

        FragmentC accumulator;
        wmma::fill_fragment(accumulator, 0.0f);
        const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
        for (int chunk = 0; chunk < chunks; ++chunk) {
            const int group_base = chunk * kGroupsPerChunk;
            const int k_base = group_base * kGroupSize;
            const int row_local = tid / kGroupsPerChunk;
            const int group_local = tid - row_local * kGroupsPerChunk;
            const int local_row = n0 + row_local;
            const int row = local_expert * out_per_expert + local_row;
            const int group = group_base + group_local;
            const bool valid_weight =
                local_row < out_per_expert && group < ng;
            uint32_t state = 0;
            float scale = 0.0f;
            const int8_t * table = table_pool;
            if (valid_weight) {
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                table = nepq_active_table(
                    table_pool, bank_ids, row, group,
                    nsuper, table_stride);
                scale = format_scale<FORMAT>(
                    neuron_scale[row], state, table);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                const int packed = valid_weight
                    ? decode_nepq_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, table,
                          row, group, quartet, nvec, nsign, ng,
                          0, state)
                    : 0;
#pragma unroll
                for (int pair = 0; pair < 2; ++pair) {
                    const int shift = pair * 16;
                    const int value0 = static_cast<int>(
                        static_cast<int8_t>((packed >> shift) & 0xff));
                    const int value1 = static_cast<int>(
                        static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                    *reinterpret_cast<__half2 *>(
                        &weight_tile[row_local]
                            [group_local * kGroupSize + quartet * 4 + pair * 2]) =
                        __halves2half2(
                            __float2half(scale * static_cast<float>(value0)),
                            __float2half(scale * static_cast<float>(value1)));
                }
            }

            constexpr int kActivationPairs = kTileM * (kTileK / 2);
            for (int index = tid; index < kActivationPairs; index += 256) {
                const int m_local = index / (kTileK / 2);
                const int k_pair = index - m_local * (kTileK / 2);
                const int compact = first + m_local;
                const int k = k_base + k_pair * 2;
                __half2 values = __float2half2_rn(0.0f);
                if (compact < last) {
                    const int pair_index = ids_dst[compact];
                    const int source_row = routed_input
                        ? pair_index : pair_index / routes;
                    if (k + 1 < K) {
                        values = *reinterpret_cast<const __half2 *>(
                            x + static_cast<int64_t>(source_row) * K + k);
                    } else if (k < K) {
                        values = __halves2half2(
                            x[static_cast<int64_t>(source_row) * K + k],
                            __float2half(0.0f));
                    }
                }
                *reinterpret_cast<__half2 *>(
                    &activation_tile[m_local][k_pair * 2]) = values;
            }
            __syncthreads();

            if (warp < kNFragments) {
#pragma unroll
                for (int k_local = 0; k_local < kTileK; k_local += 16) {
                    FragmentA activation_fragment;
                    FragmentB weight_fragment;
                    wmma::load_matrix_sync(
                        activation_fragment, &activation_tile[0][k_local],
                        kStrideK);
                    wmma::load_matrix_sync(
                        weight_fragment, &weight_tile[warp * 16][k_local],
                        kStrideK);
                    wmma::mma_sync(
                        accumulator, activation_fragment,
                        weight_fragment, accumulator);
                }
            }
            __syncthreads();
        }

        if (warp < kNFragments) {
            wmma::store_matrix_sync(
                &output_tile[warp][0][0], accumulator,
                16, wmma::mem_row_major);
        }
        __syncthreads();
        if (warp < kNFragments) {
            for (int element = lane; element < 256; element += 32) {
                const int m_local = element / 16;
                const int n_local = element - m_local * 16;
                const int compact = first + m_local;
                const int local_row = n0 + warp * 16 + n_local;
                if (compact < last && local_row < out_per_expert) {
                    const int pair_index = ids_dst[compact];
                    output[static_cast<int64_t>(pair_index) * out_per_expert +
                           local_row] =
                        __float2half(output_tile[warp][m_local][n_local]);
                }
            }
        }
        __syncthreads();
    }
}
