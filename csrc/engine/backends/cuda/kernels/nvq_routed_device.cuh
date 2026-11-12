#pragma once
template<int Format>
__device__ __forceinline__ float nvq_dense_routed_group_dot(
        const NvqDeviceWeight& w,int row,const int8_t* input,const float* scales) {
    float sum=0.f;
    for(int group=threadIdx.x;group<w.ng;group+=32) {
        const auto record=load_dense_nvq_group<Format>(w.indices,w.indices_nbytes,row,group,w.nsign,w.ng);
        const auto state=record.state();
        const auto* bank=active_codebook<Format>(w.codebook,state);
        int dot=0;
#pragma unroll
        for(int segment=0;segment<3;++segment)if(group*3+segment<w.nsign) {
            const auto values=decode_dense_nvq_vector<Format>(record,bank,segment,w.sign_mode);
            const auto x=*reinterpret_cast<const int2*>(input+group*24+segment*8);
            dot=__dp4a(values.y,x.y,__dp4a(values.x,x.x,dot));
        }
        sum=fmaf(format_scale<Format>(w.neuron_scale[row],state,w.codebook)*scales[group],float(dot),sum);
    }
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    return sum;
}

template<int Format>
__device__ __forceinline__ float nvq_routed_group_dot(
        const NvqDeviceWeight& w,int row,const int8_t* input,const float* scales) {
    if constexpr(dense_nvq_format(Format))if(w.aux_nbytes==0)
        return nvq_dense_routed_group_dot<Format>(w,row,input,scales);
    float sum=0.f;
    for(int group=threadIdx.x;group<w.ng;group+=32) {
        const int64_t state_index=int64_t(row)*w.ng+group;
        const uint32_t state=load_nvq_state<Format>(w.indices,w.indices_nbytes,w.sub_scale,
            w.sub_scale_nbytes,state_index,w.sub_bits,w.ng);
        const float scale=(Format==kNvq1L || Format==kNvq1S) ? w.neuron_scale[row]*float(state) :
            format_scale<Format>(w.neuron_scale[row],state,w.codebook);
        const int8_t* bank=active_codebook<Format>(w.codebook,state);
        uint64_t packed=0;uint32_t signs=0,exec96[3]{};int delta=0;
        if constexpr(Format==kNvq1L || Format==kNvq1S) {
            if(w.aux_nbytes==0)packed=load_nvq1_record64<Format>(w.indices,w.indices_nbytes,row,group,w.ng);
            const int negative=w.aux_nbytes==0?int((packed>>(Format==kNvq1S?31:36))&1u):
                load_packed_bits(w.aux,state_index,1,w.aux_nbytes);
            delta=negative?-1:1;
            if constexpr(Format==kNvq1S)bank=w.codebook+negative*kNvq1SBankBytes;
        }else if constexpr(Format==kNvq2JscXLGroupExec) {
            packed=load_group_exec64(w.indices,row,group,w.ng);
        }else if constexpr(Format==kNvq3JscLGroupExec) {
            load_group_exec96_words(w.indices,row,group,w.ng,exec96);
        }else if constexpr(Format!=kNvq2Exec && Format!=kNvq2JscExec && Format!=kNpq0L && Format!=kNpq0S) {
            const int64_t sign_bit=(int64_t(row)*w.nsign+group*3)*7;
            signs=w.ng>32 ? load_packed_sign_group3_words(w.aux,sign_bit,w.aux_nbytes) :
                load_packed_sign_group3(w.aux,sign_bit,w.aux_nbytes);
        }
        if constexpr(Format!=kNvq2Exec && Format!=kNvq2JscExec && Format!=kNpq0L && Format!=kNpq0S &&
                     Format!=kNvq2JscXLGroupExec && Format!=kNvq3JscLGroupExec) {
            constexpr int bits=format_index_bits(Format),vectors=is_d4_format(Format)?6:3;
            if(!((Format==kNvq1L || Format==kNvq1S) && w.aux_nbytes==0))
                packed=load_packed_group_window<bits*vectors>(w.indices,(int64_t(row)*w.nvec+group*vectors)*bits,w.indices_nbytes);
        }
        int dot=0,activation_sum=0;
#pragma unroll
        for(int segment=0;segment<3;++segment) {
            const auto weight=load_nvq_group_vec8<Format,true>(w.indices,w.indices_nbytes,w.aux,w.aux_nbytes,
                w.codebook,bank,row,group,segment,w.ng,w.nvec,w.nsign,w.sign_mode,state,signs,packed,exec96,delta);
            const int2 activation=*reinterpret_cast<const int2*>(input+group*24+segment*8);
            if(weight.valid) {
                dot=__dp4a(weight.values.y,activation.y,__dp4a(weight.values.x,activation.x,dot));
                if constexpr(Format==kNvq1L || Format==kNvq1S)
                    activation_sum=__dp4a(0x01010101,activation.y,__dp4a(0x01010101,activation.x,activation_sum));
            }
        }
        float value=float(dot);
        if constexpr(Format==kNvq1L)value+=0.125f*float(delta*activation_sum);
        if constexpr(Format==kNvq1S)value+=0.15625f*float(delta*activation_sum);
        sum=fmaf(scale*scales[group],value,sum);
    }
#pragma unroll
    for(int offset=16;offset>0;offset>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,offset);
    return sum;
}

template<int Format>
__global__ void __launch_bounds__(128) nvq_moe_groups_kernel(
        NvqDeviceWeight w,const int8_t* input,const float* scales,const int32_t* ids,
        const int32_t* local_map,half* output,int experts,int local_experts,int rows,int routes,bool down) {
    const int pair=blockIdx.y,row=blockIdx.x*4+threadIdx.y;
    if(row>=rows)return;
    const int expert=ids[pair];
    if(unsigned(expert)>=unsigned(experts))return;
    const int local=local_map[expert];if(unsigned(local)>=unsigned(local_experts))return;
    const int source=down?pair:pair/routes;
    const float value=nvq_routed_group_dot<Format>(w,local*rows+row,input+int64_t(source)*w.ng*24,scales+int64_t(source)*w.ng);
    if(threadIdx.x==0)output[int64_t(pair)*rows+row]=__float2half(value);
}

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE>
__device__ __forceinline__ void nvq_moe_mmvq_task(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int8_t * qx,
    const float * xscale,
    __half * output,
    float * partial,
    int pair,
    int routes,
    int out_per_expert,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode,
    bool routed_input,
    int source_row,
    int local_expert,
    int row0) {
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;

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
        const unsigned int active_mask = __activemask();
        const int group_leader = lane - segment_in_group;
        const int source_lane = group_leader >= 0 ? group_leader : lane;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int8_t * activation =
            qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
        float activation_scale;
        if constexpr (
            SHARE_GROUP_STATE &&
            (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec)) {
            const float loaded_scale = lane == source_lane
                ? xscale[static_cast<int64_t>(source_row) * ng + group]
                : 0.0f;
            activation_scale = __shfl_sync(
                active_mask, loaded_scale, source_lane);
        } else {
            activation_scale =
                xscale[static_cast<int64_t>(source_row) * ng + group];
        }
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            const int local_row = row0 + row_local;
            if (local_row >= out_per_expert) continue;
            const int row = local_expert * out_per_expert + local_row;
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            uint32_t sub;
            float weight_scale;
            if constexpr (
                SHARE_GROUP_STATE &&
                (FORMAT == kNvq2Exec || FORMAT == kNvq2JscExec)) {
                const uint32_t loaded_sub = lane == source_lane
                    ? load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng)
                    : 0;
                sub = __shfl_sync(active_mask, loaded_sub, source_lane);
                const float loaded_weight_scale = lane == source_lane
                    ? format_scale<FORMAT>(
                        neuron_scale[row], loaded_sub, codebook)
                    : 0.0f;
                weight_scale = __shfl_sync(
                    active_mask, loaded_weight_scale, source_lane);
            } else {
                sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                    ? neuron_scale[row] * static_cast<float>(sub)
                    : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
            }
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
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
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            partial[row_local * NWARPS + warp] = acc[row_local];
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < NWARPS
                ? partial[row_local * NWARPS + lane]
                : 0.0f;
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

template <int FORMAT, int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE>
__global__ void __launch_bounds__(NWARPS * 32) nvq_moe_mmvq_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
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
    int sub_bits,
    int sign_mode,
    bool routed_input) {
    const int pair = blockIdx.y;
    if (pair >= pairs) return;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];
    nvq_moe_mmvq_task<
        FORMAT, NWARPS, ROWS_PER_BLOCK, SHARE_GROUP_STATE>(
        indices, indices_nbytes, aux, aux_nbytes,
        sub_scale, sub_scale_nbytes, neuron_scale, codebook,
        qx, xscale, output, &partial[0][0], pair, routes,
        out_per_expert, ng, nvec, nsign, sub_bits, sign_mode,
        routed_input, routed_input ? pair : pair / routes,
        local_expert, blockIdx.x * ROWS_PER_BLOCK);
}

template <int FORMAT, int PHYSICAL_WARPS, int LOGICAL_WARPS, int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(PHYSICAL_WARPS * 32)
nvq_moe_mmvq_exact_reduction_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
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
    int sub_bits,
    int sign_mode,
    bool routed_input) {
    static_assert(LOGICAL_WARPS % PHYSICAL_WARPS == 0);
    constexpr int kAccumulators = LOGICAL_WARPS / PHYSICAL_WARPS;
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
    const int local_expert = expert_local[expert];
    if (static_cast<unsigned int>(local_expert) >=
        static_cast<unsigned int>(pool_experts)) {
        return;
    }
    const int source_row = routed_input ? pair : pair / routes;

    float acc[ROWS_PER_BLOCK][kAccumulators];
#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
            acc[row_local][accumulator] = 0.0f;
        }
    }
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
        const int logical_warp = warp + accumulator * PHYSICAL_WARPS;
        for (int segment = logical_warp * 32 + lane;
             segment < nsign;
             segment += LOGICAL_WARPS * 32) {
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
                const int64_t sub_linear =
                    static_cast<int64_t>(row) * ng + group;
                const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                const auto weight = load_nvq_vec8<FORMAT>(
                    indices, indices_nbytes, aux, aux_nbytes, codebook,
                    row, segment, group, ng, nvec, nsign, sign_mode, sub);
                const float weight_scale =
                    (FORMAT == kNvq1L || FORMAT == kNvq1S)
                    ? neuron_scale[row] * static_cast<float>(sub)
                    : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
                acc[row_local][accumulator] = fmaf(
                    weight_scale * activation_scale,
                    dot_nvq_vec8<FORMAT>(weight, activation),
                    acc[row_local][accumulator]);
            }
        }
    }

#pragma unroll
    for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
        for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[row_local][accumulator] += __shfl_xor_sync(
                    0xffffffff, acc[row_local][accumulator], offset);
            }
        }
    }
    __shared__ float partial[ROWS_PER_BLOCK][LOGICAL_WARPS];
    if (lane == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
#pragma unroll
            for (int accumulator = 0; accumulator < kAccumulators; ++accumulator) {
                const int logical_warp = warp + accumulator * PHYSICAL_WARPS;
                partial[row_local][logical_warp] =
                    acc[row_local][accumulator];
            }
        }
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int row_local = 0; row_local < ROWS_PER_BLOCK; ++row_local) {
            float value = lane < LOGICAL_WARPS
                ? partial[row_local][lane]
                : 0.0f;
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
__global__ void __launch_bounds__(128) nvq_moe_grouped_tile_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
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
    int sub_bits,
    int sign_mode,
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
        const int local_expert = expert_local[expert];
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
            const auto weight = load_nvq_vec8<FORMAT>(
                indices, indices_nbytes, aux, aux_nbytes, codebook,
                row, segment, group, ng, nvec, nsign, sign_mode, sub);
            const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
                ? neuron_scale[row] * static_cast<float>(sub)
                : format_scale<FORMAT>(neuron_scale[row], sub, codebook);
#pragma unroll
            for (int item = 0; item < TILE_M; ++item) {
                const int compact = first + item;
                if (compact >= last) continue;
                const int pair = ids_dst[compact];
                const int source_row = routed_input ? pair : pair / routes;
                const int8_t * activation =
                    qx + (static_cast<int64_t>(source_row) * ng * kGroupSize + k);
                acc[item] = fmaf(
                    weight_scale *
                        xscale[static_cast<int64_t>(source_row) * ng + group],
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

__global__ void __launch_bounds__(128) nvq_moe_groups_hetero_kernel(
        const int64_t* pointers,const int64_t* sizes,const int32_t* params,
        const int32_t* expert_pool,const int32_t* expert_local,const int32_t* ids,
        const int8_t* input,const float* scales,half* output,const int32_t* active,
        int experts,int pools,int rows,int routes,int pairs,bool down) {
    const int pair=blockIdx.y,row=blockIdx.x*4+threadIdx.y;
    if(row>=rows)return;
    const int expert=ids[pair];if(unsigned(expert)>=unsigned(experts))return;
    const int pool=active?active[1+pair]:expert_pool[expert];
    const int local=active?active[1+pairs+pair]:expert_local[expert];
    if(unsigned(pool)>=unsigned(pools) || local<0)return;
    const auto* p=pointers+int64_t(pool)*5;const auto* s=sizes+int64_t(pool)*3;
    const auto* g=params+int64_t(pool)*7;if(local>=g[0])return;
    const NvqDeviceWeight w{reinterpret_cast<const uint8_t*>(p[0]),s[0],reinterpret_cast<const uint8_t*>(p[1]),s[1],
        reinterpret_cast<const uint8_t*>(p[2]),s[2],reinterpret_cast<const float*>(p[3]),reinterpret_cast<const int8_t*>(p[4]),
        0,g[0]*rows,g[1],g[2],g[3],g[4],g[5]};
    const int source=down?pair:pair/routes;float value=0.f;
#define TYLOINFER_NVQ_ROUTED_GROUP_CASE(F) case F: value=nvq_routed_group_dot<F>(w,local*rows+row, \
        input+int64_t(source)*w.ng*24,scales+int64_t(source)*w.ng);break
    switch(g[6]) {
        TYLOINFER_NVQ_ROUTED_GROUP_CASE(1);TYLOINFER_NVQ_ROUTED_GROUP_CASE(2);TYLOINFER_NVQ_ROUTED_GROUP_CASE(3);
        TYLOINFER_NVQ_ROUTED_GROUP_CASE(4);TYLOINFER_NVQ_ROUTED_GROUP_CASE(5);TYLOINFER_NVQ_ROUTED_GROUP_CASE(6);
        TYLOINFER_NVQ_ROUTED_GROUP_CASE(7);TYLOINFER_NVQ_ROUTED_GROUP_CASE(8);TYLOINFER_NVQ_ROUTED_GROUP_CASE(9);
        TYLOINFER_NVQ_ROUTED_GROUP_CASE(10);TYLOINFER_NVQ_ROUTED_GROUP_CASE(11);TYLOINFER_NVQ_ROUTED_GROUP_CASE(12);
        TYLOINFER_NVQ_ROUTED_GROUP_CASE(13);TYLOINFER_NVQ_ROUTED_GROUP_CASE(14);TYLOINFER_NVQ_ROUTED_GROUP_CASE(15);
        TYLOINFER_NVQ_ROUTED_GROUP_CASE(16);TYLOINFER_NVQ_ROUTED_GROUP_CASE(17);
    }
#undef TYLOINFER_NVQ_ROUTED_GROUP_CASE
    if(threadIdx.x==0)output[int64_t(pair)*rows+row]=__float2half(value);
}

template <int NWARPS, int ROWS_PER_BLOCK, bool SHARE_GROUP_STATE,bool ActiveRows=false>
__global__ void __launch_bounds__(NWARPS * 32)
nvq_moe_mmvq_hetero_kernel(
    const int64_t * weight_ptrs,
    const int64_t * weight_sizes,
    const int32_t * pool_params,
    const int32_t * expert_pool,
    const int32_t * expert_local,
    const int8_t * qx,
    const float * xscale,
    const int32_t * ids,
    __half * output,
    int pairs,
    int routes,
    int global_experts,
    int pool_count,
    int out_per_expert,
    bool routed_input,const int32_t* active_plan=nullptr) {
    const int row_tiles=(out_per_expert-1)/ROWS_PER_BLOCK+1;
    const int64_t total=ActiveRows?int64_t(active_plan[0])*row_tiles:1;
    for(int64_t task=ActiveRows?blockIdx.x:0;task<total;task+=ActiveRows?gridDim.x:1) {
    const int pair = ActiveRows?active_plan[1+2*pairs+task/row_tiles]:int(blockIdx.y);
    const int row0=ActiveRows?int(task%row_tiles)*ROWS_PER_BLOCK:int(blockIdx.x)*ROWS_PER_BLOCK;
    if (pair >= pairs) continue;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(global_experts)) {
        continue;
    }
    const int pool = ActiveRows?active_plan[1+pair]:expert_pool[expert];
    const int local_expert = ActiveRows?active_plan[1+pairs+pair]:expert_local[expert];
    if (static_cast<unsigned int>(pool) >=
            static_cast<unsigned int>(pool_count) || local_expert < 0) {
        continue;
    }
    const int64_t * pointers =
        weight_ptrs + static_cast<size_t>(pool) * 5;
    const int64_t * sizes =
        weight_sizes + static_cast<size_t>(pool) * 3;
    const int32_t * params =
        pool_params + static_cast<size_t>(pool) * 7;
    if (local_expert >= params[0]) continue;
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
    __shared__ float partial[ROWS_PER_BLOCK][NWARPS];

#define NVQ_MOE_HETERO_MMVQ_CASE(FORMAT_VALUE)                                 \
    case FORMAT_VALUE:                                                          \
        nvq_moe_mmvq_task<                                                      \
            FORMAT_VALUE, NWARPS, ROWS_PER_BLOCK, SHARE_GROUP_STATE>(           \
            indices, sizes[0], aux, sizes[1], sub_scale, sizes[2],              \
            neuron_scale, codebook, qx, xscale, output, &partial[0][0],         \
            pair, routes, out_per_expert, ng, nvec, nsign, sub_bits,           \
            sign_mode, routed_input, routed_input ? pair : pair / routes,       \
            local_expert, row0);                                              \
        break
    switch (format) {
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq1L);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2Exec);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2Jsc);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscExec);
        NVQ_MOE_HETERO_MMVQ_CASE(kNpq0L);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq1S);
        NVQ_MOE_HETERO_MMVQ_CASE(kNpq0S);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3Jsc);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3Jsc2);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3Jsc512);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscL);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscXL);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3JscL);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq2JscXLGroupExec);
        NVQ_MOE_HETERO_MMVQ_CASE(kNvq3JscLGroupExec);
    }
#undef NVQ_MOE_HETERO_MMVQ_CASE
    if constexpr(ActiveRows)__syncthreads();
    }
}
