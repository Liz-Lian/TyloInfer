#pragma once
template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(NWARPS * 32) nepq_moe_mmvq_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    const int32_t * expert_local,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_experts,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    bool routed_input) {
    const int pair = blockIdx.y;
    const int row0 = blockIdx.x * ROWS_PER_BLOCK;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local == nullptr ? expert : expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
        acc[row_local] = 0.0f;
    }
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int8_t * activation =
            qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
        const float activation_scale =
            xscale[static_cast<int64_t>(source_row) * ng + group];
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            const int local_row = row0 + row_local;
            if (local_row >= out_per_expert) continue;
            const int row = local_expert * out_per_expert + local_row;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            const auto weight = load_nepq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                row, segment, group, ng, nvec, nsign, 0, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, table);
            acc[row_local] = fmaf(
                weight_scale * activation_scale,
                dot_nvq_vec8<FORMAT>(weight, activation),
                acc[row_local]);
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[row_local] += __shfl_xor_sync(0xffffffff, acc[row_local], offset);
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            partial[row_local][warp] = acc[row_local];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < NWARPS ? partial[row_local][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_xor_sync(0xffffffff, value, offset);
            }
            const int local_row = row0 + row_local;
            if (lane == 0 && local_row < out_per_expert) {
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(value);
            }
        }
    }
}

template <int FORMAT, int TILE_M>
__global__ void __launch_bounds__(128) nepq_moe_grouped_tile_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    const int8_t * qx,
    const float * xscale,
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
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int max_tiles,
    int row_tiles,
    bool routed_input) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < max_tasks; task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        if (tile >= tile_bounds[global_experts]) continue;
        const int expert = tile_experts[tile];
        const int local_expert =
            expert_local == nullptr ? expert : expert_local[expert];
        if (static_cast<unsigned int>(local_expert) >=
            static_cast<unsigned int>(pool_experts)) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int local_row = row_tile * 4 + warp;
        if (local_row >= out_per_expert) continue;
        const int row = local_expert * out_per_expert + local_row;

        float acc[TILE_M];
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) acc[item] = 0.0f;
        for (int segment = lane; segment < nsign; segment += 32) {
            const int group = segment / 3;
            const int segment_in_group = segment - group * 3;
            const int k = group * kGroupSize + segment_in_group * 8;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            const int8_t * table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, table,
                row, segment, group, ng, nvec, nsign, 0, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, table);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                const int compact = first + item;
                if (compact >= last) continue;
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int8_t * activation =
                    qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
                acc[item] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(source_row) * ng + group],
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[item]);
            }
        }
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[item] += __shfl_xor_sync(0xffffffff, acc[item], offset);
            }
            if (lane == 0 && first + item < last) {
                const int pair = ids_dst[first + item];
                output[static_cast<int64_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[item]);
            }
        }
    }
}
