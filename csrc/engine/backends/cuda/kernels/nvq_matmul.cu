#include "env.h"
// Compact NPQ/NVQ inference kernels.
//
// The persistent GPU representation keeps every variable-width stream packed:
//   NVQ1-L: 11-bit ternary codebook indices, 1-bit group delta signs
//   NVQ2: 8-bit E8 indices, 7-bit even-parity signs per 8 weights
//   NVQ3: 8-bit D4 indices per 4 weights, 7-bit even-parity signs per 8 weights
// All profiles share one fp32 neuron anchor and one packed relative scale per
// gs=24 group. Decode/small-batch matmul quantizes activations per group and
// performs signed int8 dot products with __dp4a or integer MMA. The medium-M
// path expands only one 64x96 fp16 weight tile at a time for Tensor Core GEMM.

#include <cuda_runtime.h>
#include "backend.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <map>
#include <type_traits>

#include "packed_nvq.cuh"
#include "nvq_format.cuh"
#include "nvq_dense_group.cuh"
#include "nvq_routed_reuse.cuh"
#include "async_copy.cuh"

using namespace nvcuda;

namespace {

using namespace tyloi::cuda::packed_nvq;
using tyloi::cuda::nvq_format::launch_by_format;
using tyloi::cuda::nvq_format::launch_nepq_by_format;


#include "nvq_projection_device.cuh"

// Integer Tensor Core path for M=4..64. Each gs24 quantization group is
// represented as one transient K32 tile in shared memory. The MMA dot is
// rescaled per group before accumulation, preserving NVQ's group scales.

// Online-dequant FP16 Tensor Core path for M=16..256. Four gs24 groups form
// one K=96 chunk. A block expands a 64-row weight tile into shared memory and
// reuses it across 16..128 activation rows without materializing full weights.


template <int FORMAT, int NWARPS, int MAX_M>
__global__ void __launch_bounds__(NWARPS * 32, 1) nepq_gemv_batch_vec8_kernel(
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
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    const int row = blockIdx.x;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= N) return;

    float acc[MAX_M];
#pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
    for (int segment = warp * 32 + lane;
         segment < nsign;
         segment += NWARPS * 32) {
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const int8_t * table = nepq_active_table(
            table_pool, bank_ids, row, group, nsuper, table_stride);
        const auto weight = load_nepq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, table,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float weight_scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, table);
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                const float dot = dot_nvq_vec8<FORMAT>(
                    weight, qx + (static_cast<int64_t>(m) * ng * kGroupSize + k));
                acc[m] = fmaf(
                    weight_scale * xscale[static_cast<int64_t>(m) * ng + group],
                    dot,
                    acc[m]);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
        }
    }
    __shared__ float partial[MAX_M][NWARPS];
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) partial[m][warp] = acc[m];
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            acc[m] = lane < NWARPS ? partial[m][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc[m] += __shfl_xor_sync(0xffffffff, acc[m], offset);
            }
            if (lane == 0 && m < M) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[m]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nepq_mmq_mma24_kernel(
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
    __half * output,
    int M,
    int N,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    constexpr int kChunkGroups = 8;
    constexpr int kGroupInts = 8;
    constexpr int kStride = kChunkGroups * kGroupInts + 4;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kRowsPerWarp = 8;
    constexpr int kWarps = 8;
    constexpr int kTileN = kRowsPerWarp * kWarps;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;

    __shared__ int activation[kTileM][kStride];
    __shared__ int weight[kWarps][kRowsPerWarp][kStride];
    __shared__ __half weight_scale[kChunkGroups][kTileN];
    __shared__ float activation_scale[kChunkGroups][kTileM];

    float acc[MTILES][4];
#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) acc[mt][fragment] = 0.0f;
    }

    for (int group_base = 0; group_base < ng; group_base += kChunkGroups) {
        const int active_groups = min(kChunkGroups, ng - group_base);
        for (int index = tid; index < kTileM * kChunkGroups * kGroupInts;
             index += kWarps * 32) {
            const int m_local = index / (kChunkGroups * kGroupInts);
            const int rest = index - m_local * kChunkGroups * kGroupInts;
            const int group_local = rest / kGroupInts;
            const int quartet = rest - group_local * kGroupInts;
            const int m = m0 + m_local;
            int value = 0;
            if (group_local < active_groups && quartet < kChunksPerGroup &&
                (!NEED_CHECK || m < M)) {
                const int group = group_base + group_local;
                value = load_i8x4(
                    qx + (static_cast<int64_t>(m) * ng + group) * kGroupSize
                    + quartet * 4);
            }
            activation[m_local][group_local * kGroupInts + quartet] = value;
        }

        constexpr int kWeightTasks = kRowsPerWarp * kChunkGroups;
        for (int task = lane; task < kWeightTasks; task += 32) {
            const int row_local = task / kChunkGroups;
            const int group_local = task - row_local * kChunkGroups;
            const int row = n0 + warp * kRowsPerWarp + row_local;
            const int group = group_base + group_local;
            const bool valid = group_local < active_groups && (!NEED_CHECK || row < N);
            uint32_t sub = 0;
            const int8_t * table = table_pool;
            if (valid) {
                const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
                sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
                table = nepq_active_table(
                    table_pool, bank_ids, row, group, nsuper, table_stride);
            }
#pragma unroll
            for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
                weight[warp][row_local][group_local * kGroupInts + quartet] = valid
                    ? decode_nepq_chunk4<FORMAT>(
                          indices, indices_nbytes, aux, aux_nbytes, table,
                          row, group, quartet, nvec, nsign, ng, sign_mode, sub)
                    : 0;
            }
            weight[warp][row_local][group_local * kGroupInts + 6] = 0;
            weight[warp][row_local][group_local * kGroupInts + 7] = 0;
            const float scale = valid
                ? format_scale<FORMAT>(neuron_scale[row], sub, table)
                : 0.0f;
            weight_scale[group_local][warp * kRowsPerWarp + row_local] = __float2half(scale);
        }

        for (int index = tid; index < kTileM * kChunkGroups; index += kWarps * 32) {
            const int m_local = index / kChunkGroups;
            const int group_local = index - m_local * kChunkGroups;
            const int m = m0 + m_local;
            activation_scale[group_local][m_local] =
                group_local < active_groups && (!NEED_CHECK || m < M)
                ? xscale[static_cast<int64_t>(m) * ng + group_base + group_local]
                : 0.0f;
        }
        __syncthreads();

#pragma unroll
        for (int group_local = 0; group_local < kChunkGroups; ++group_local) {
            int a[MTILES][4];
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                load_mma_a_m16n8k32(
                    a[mt],
                    &activation[mt * 16 + lane % 16]
                               [group_local * kGroupInts + (lane / 16) * 4]);
            }
            int b[2];
            load_mma_b_m16n8k32(
                b,
                &weight[warp][lane % 8]
                       [group_local * kGroupInts + (((lane / 8) * 4) & 7)]);
#pragma unroll
            for (int mt = 0; mt < MTILES; ++mt) {
                int dot[4] = {0, 0, 0, 0};
                mma_m16n8k32_s8(dot, a[mt], b);
#pragma unroll
                for (int fragment = 0; fragment < 4; ++fragment) {
                    const int m_local = mt * 16 + mma168_i(fragment);
                    const int row_local = warp * kRowsPerWarp + mma168_j(fragment);
                    const int m = m0 + m_local;
                    const int row = n0 + row_local;
                    if (!NEED_CHECK || (m < M && row < N)) {
                        const float scale = activation_scale[group_local][m_local]
                            * __half2float(weight_scale[group_local][row_local]);
                        acc[mt][fragment] = fmaf(
                            scale, static_cast<float>(dot[fragment]), acc[mt][fragment]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mt = 0; mt < MTILES; ++mt) {
#pragma unroll
        for (int fragment = 0; fragment < 4; ++fragment) {
            const int m = m0 + mt * 16 + mma168_i(fragment);
            const int row = n0 + warp * kRowsPerWarp + mma168_j(fragment);
            if (!NEED_CHECK || (m < M && row < N)) {
                output[static_cast<int64_t>(m) * N + row] = __float2half(acc[mt][fragment]);
            }
        }
    }
}

template <int FORMAT, int MTILES, bool NEED_CHECK>
__global__ void __launch_bounds__(256) nepq_gemm_f16_gs24_kernel(
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
    __half * output,
    int M,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    constexpr int kGroupsPerChunk = kNepqGroupsPerSupergroup;
    constexpr int kTileK = kGroupSize * kGroupsPerChunk;
    constexpr int kStrideK = kTileK + 8;
    constexpr int kTileM = 16 * MTILES;
    constexpr int kTileN = 64;
    constexpr int kWarps = 8;
    constexpr int kWmmaM = 16;
    constexpr int kWmmaN = 16;
    constexpr int kWmmaK = 16;
    constexpr int kNFragments = kTileN / kWmmaN;
    constexpr int kAccumulatorsPerWarp = (MTILES + 1) / 2;

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int m0 = blockIdx.y * kTileM;
    const int n0 = blockIdx.x * kTileN;
    const int warp_m0 = warp / kNFragments;
    const int warp_n = warp % kNFragments;

    __shared__ __half weight_tile[kTileN][kStrideK];
    __shared__ __half activation_tile[kTileM][kStrideK];
    __shared__ float output_tile[kNFragments][kWmmaM][kWmmaN];

    using FragmentA = wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::row_major>;
    using FragmentB = wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                                     __half, wmma::col_major>;
    using FragmentC = wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>;
    FragmentC accumulators[kAccumulatorsPerWarp];
#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
        wmma::fill_fragment(accumulators[accumulator], 0.0f);
    }

    const int chunks = (ng + kGroupsPerChunk - 1) / kGroupsPerChunk;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int group_base = chunk * kGroupsPerChunk;
        const int k_base = group_base * kGroupSize;
        const int row_local = tid / kGroupsPerChunk;
        const int group_local = tid - row_local * kGroupsPerChunk;
        const int row = n0 + row_local;
        const int group = group_base + group_local;
        const bool valid_weight = group < ng && (!NEED_CHECK || row < N);
        uint32_t state = 0;
        float scale = 0.0f;
        const int8_t * table = table_pool;
        if (valid_weight) {
            const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
            state = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
            table = nepq_active_table(
                table_pool, bank_ids, row, group, nsuper, table_stride);
            scale = format_scale<FORMAT>(neuron_scale[row], state, table);
        }
#pragma unroll
        for (int quartet = 0; quartet < kChunksPerGroup; ++quartet) {
            const int packed = valid_weight
                ? decode_nepq_chunk4<FORMAT>(
                      indices, indices_nbytes, aux, aux_nbytes, table,
                      row, group, quartet, nvec, nsign, ng, sign_mode, state)
                : 0;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int shift = pair * 16;
                float value0 = static_cast<float>(
                    static_cast<int8_t>((packed >> shift) & 0xff));
                float value1 = static_cast<float>(
                    static_cast<int8_t>((packed >> (shift + 8)) & 0xff));
                const __half2 values = __halves2half2(
                    __float2half(scale * value0), __float2half(scale * value1));
                *reinterpret_cast<__half2 *>(
                    &weight_tile[row_local]
                                [group_local * kGroupSize + quartet * 4 + pair * 2]) = values;
            }
        }

        constexpr int kActivationPairs = kTileM * (kTileK / 2);
        for (int index = tid; index < kActivationPairs; index += kWarps * 32) {
            const int m_local = index / (kTileK / 2);
            const int pair = index - m_local * (kTileK / 2);
            const int m = m0 + m_local;
            const int k = k_base + pair * 2;
            __half2 values = __float2half2_rn(0.0f);
            if ((!NEED_CHECK || m < M) && k + 1 < K && (K & 1) == 0) {
                values = *reinterpret_cast<const __half2 *>(
                    x + static_cast<int64_t>(m) * K + k);
            } else if ((!NEED_CHECK || m < M) && k < K) {
                const __half value0 = x[static_cast<int64_t>(m) * K + k];
                const __half value1 = k + 1 < K
                    ? x[static_cast<int64_t>(m) * K + k + 1]
                    : __float2half(0.0f);
                values = __halves2half2(value0, value1);
            }
            *reinterpret_cast<__half2 *>(
                &activation_tile[m_local][pair * 2]) = values;
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int k_local = 0; k_local < kTileK; k_local += kWmmaK) {
            if (warp_active) {
                FragmentB weight_fragment;
                wmma::load_matrix_sync(
                    weight_fragment, &weight_tile[warp_n * kWmmaN][k_local], kStrideK);
#pragma unroll
                for (int accumulator = 0;
                     accumulator < kAccumulatorsPerWarp;
                     ++accumulator) {
                    const int m_fragment = warp_m0 + accumulator * 2;
                    if (m_fragment < MTILES) {
                        FragmentA activation_fragment;
                        wmma::load_matrix_sync(
                            activation_fragment,
                            &activation_tile[m_fragment * kWmmaM][k_local],
                            kStrideK);
                        wmma::mma_sync(
                            accumulators[accumulator], activation_fragment,
                            weight_fragment, accumulators[accumulator]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int accumulator = 0; accumulator < kAccumulatorsPerWarp; ++accumulator) {
#pragma unroll
        for (int warp_row = 0; warp_row < 2; ++warp_row) {
            const int m_fragment = warp_m0 + accumulator * 2;
            const bool owns_fragment = warp_m0 == warp_row && m_fragment < MTILES;
            if (owns_fragment) {
                wmma::store_matrix_sync(
                    &output_tile[warp_n][0][0], accumulators[accumulator],
                    kWmmaN, wmma::mem_row_major);
            }
            __syncthreads();
            if (owns_fragment) {
                const int output_m0 = m0 + m_fragment * kWmmaM;
                const int output_n0 = n0 + warp_n * kWmmaN;
#pragma unroll
                for (int element = lane; element < kWmmaM * kWmmaN; element += 32) {
                    const int local_m = element / kWmmaN;
                    const int local_n = element - local_m * kWmmaN;
                    const int m = output_m0 + local_m;
                    const int n = output_n0 + local_n;
                    if (!NEED_CHECK || (m < M && n < N)) {
                        output[static_cast<int64_t>(m) * N + n] =
                            __float2half(output_tile[warp_n][local_m][local_n]);
                    }
                }
            }
            __syncthreads();
        }
    }
}

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

template<int Warps>
int nvq_active_resident_blocks() {
    int device=0;
    TYLOINFER_RUNTIME_CHECK(cudaGetDevice(&device)==cudaSuccess,"NVQ active device query failed");
    static thread_local std::map<int,int> cached;
    if(const auto found=cached.find(device);found!=cached.end())return found->second;
    cudaDeviceProp properties{};int active=0;
    TYLOINFER_RUNTIME_CHECK(cudaGetDeviceProperties(&properties,device)==cudaSuccess,
        "NVQ active device properties unavailable");
    TYLOINFER_RUNTIME_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        nvq_moe_mmvq_hetero_kernel<Warps,2,false,true>,Warps*32,0)==cudaSuccess && active>0,
        "NVQ active kernel occupancy unavailable");
    const int blocks=active*properties.multiProcessorCount;
    cached.emplace(device,blocks);return blocks;
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

void check_nepq_common(
    const ::tyloi::DenseTensor & indices,
    const ::tyloi::DenseTensor & aux,
    const ::tyloi::DenseTensor & sub_scale,
    const ::tyloi::DenseTensor & neuron_scale,
    const ::tyloi::DenseTensor & table_pool,
    const ::tyloi::DenseTensor & bank_ids,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    TYLOINFER_RUNTIME_CHECK(
        format == kNvq1L || format == kNpq0L ||
        format == kNvq1S || format == kNpq0S,
        "NEPQ base format must be NVQ1-L/NPQ0-L/NVQ1-S/NPQ0-S");
    TYLOINFER_RUNTIME_CHECK(neuron_len > 0 && neuron_len % 8 == 0,
                "NEPQ neuron_len must be a positive multiple of 8");
    const int expected_sub_bits = format == kNvq1S ? 4 :
        (format == kNpq0S ? 2 : 3);
    TYLOINFER_RUNTIME_CHECK(sub_bits == expected_sub_bits, "NEPQ state width mismatch");
    for (const auto * tensor : {&indices, &aux, &sub_scale, &bank_ids}) {
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->scalar_type() == ::tyloi::kUInt8 &&
                    tensor->is_contiguous(),
                    "NEPQ packed streams must be CUDA contiguous uint8");
    }
    TYLOINFER_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
                neuron_scale.is_contiguous() && neuron_scale.dim() == 1,
                "NEPQ neuron_scale must be CUDA contiguous float32 rank-1");
    TYLOINFER_RUNTIME_CHECK(table_pool.is_cuda() && table_pool.scalar_type() == ::tyloi::kInt8 &&
                table_pool.is_contiguous() && table_pool.dim() == 2 &&
                table_pool.size(0) >= 1 && table_pool.size(0) <= 256,
                "NEPQ table pool must be CUDA contiguous int8 [1..256,stride]");
    const int table_stride = format == kNvq1L ? 2048 * 8 :
        (format == kNvq1S ? 1024 * 8 :
         (format == kNpq0L ? kNpq0LTableBytes : kNepq0SCompactTableBytes));
    TYLOINFER_RUNTIME_CHECK(table_pool.size(1) == table_stride, "NEPQ table stride mismatch");
    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + kGroupSize - 1) / kGroupSize;
    const int64_t nvec = neuron_len / 8;
    const int64_t nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    TYLOINFER_RUNTIME_CHECK(bank_ids.dim() == 2 && bank_ids.size(0) == N &&
                bank_ids.size(1) == nsuper,
                "NEPQ bank selectors must have shape [rows,ceil(groups/4)]");
    const int index_bits = format == kNvq1L ? 11 :
        (format == kNvq1S ? 9 : (format == kNpq0L ? 7 : 6));
    const int64_t index_bytes = (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const int64_t aux_bytes = delta_format ? (N * ng + 7) / 8 : 0;
    const int64_t sub_bytes = (N * ng * sub_bits + 7) / 8;
    TYLOINFER_RUNTIME_CHECK(indices.numel() == index_bytes, "NEPQ index stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(aux.numel() == aux_bytes, "NEPQ aux stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(sub_scale.numel() == sub_bytes, "NEPQ state stream length mismatch");
}

void check_common(
    const ::tyloi::DenseTensor & indices,
    const ::tyloi::DenseTensor & aux,
    const ::tyloi::DenseTensor & sub_scale,
    const ::tyloi::DenseTensor & neuron_scale,
    const ::tyloi::DenseTensor & codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,bool allow_dense=false) {
    TYLOINFER_RUNTIME_CHECK(format >= kNvq1L && format <= kNvq3JscLGroupExec,
                "NVQ format must be in [1,17]");
    TYLOINFER_RUNTIME_CHECK(gs == kGroupSize, "NVQ CUDA kernels currently require gs=24");
    TYLOINFER_RUNTIME_CHECK(sub_bits >= 1 && sub_bits <= 8, "NVQ sub_bits must be in [1,8]");
    TYLOINFER_RUNTIME_CHECK(neuron_len > 0, "NVQ neuron_len must be positive");
    TYLOINFER_RUNTIME_CHECK(sign_mode == 0 ||
                (sign_mode == 1 && (format == kNvq2 || format == kNvq2Exec)),
                "index-parity sign mode is valid only for NVQ2");
    for (const auto * tensor : {&indices, &aux, &sub_scale}) {
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->scalar_type() == ::tyloi::kUInt8 && tensor->is_contiguous(),
                    "NVQ packed streams must be CUDA contiguous uint8");
    }
    TYLOINFER_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
                neuron_scale.is_contiguous(), "NVQ neuron_scale must be CUDA contiguous float32");
    TYLOINFER_RUNTIME_CHECK(codebook.is_cuda() && codebook.scalar_type() == ::tyloi::kInt8 && codebook.is_contiguous(),
                "NVQ codebook must be CUDA contiguous int8");
    if (format == kNvq1L) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 2048 && codebook.size(1) == 8,
                    "NVQ1-L codebook must have shape [2048,8]");
    } else if (format == kNvq1S) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 1024 && codebook.size(1) == 8,
                    "NVQ1-S codebook must have shape [1024,8]");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ1-S requires a 4-bit scale and no sign mode");
    } else if (format == kNvq2 || format == kNvq2Exec) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 8,
                    "NVQ2 codebook must have shape [256,8]");
    } else if (format == kNvq2Jsc || format == kNvq2JscExec) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * kJscE8CodebookBytes),
                    "NVQ2J codebook metadata has an invalid size");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq2JscL || format == kNvq2JscXL ||
        format == kNvq2JscXLGroupExec) {
        const int e8_bank_bytes = format == kNvq2JscL
            ? kJscE8Codebook1024Bytes
            : kJscE8Codebook4096Bytes;
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * e8_bank_bytes),
                    "extended NVQ2J codebook metadata has an invalid size");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "extended NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq3Jsc || format == kNvq3Jsc2 ||
        format == kNvq3Jsc512 || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) {
        const int d4_bank_bytes = format == kNvq3Jsc512
            ? kJscD4Codebook512Bytes
            : (format == kNvq3JscL || format == kNvq3JscLGroupExec
               ? kJscD4Codebook1024Bytes
               : kJscD4CodebookBytes);
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * d4_bank_bytes),
                    "NVQ3J codebook metadata has an invalid size");
        if (format == kNvq3Jsc2) {
            TYLOINFER_RUNTIME_CHECK(
                codebook.numel() == kJscMetadataBytes + 2 * kJscD4CodebookBytes,
                "NVQ3J analytic-2 requires exactly two D4 banks");
        }
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ3J requires a 4-bit state and even-parity signs");
    } else if (format == kNpq0L) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0LTableBytes,
                    "NPQ0-L codebook metadata must contain 832 bytes");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 3 && sign_mode == 0,
                    "NPQ0-L requires a 3-bit state and no sign stream");
    } else if (format == kNpq0S) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0STableBytes,
                    "NPQ0-S expanded PQ3+3 runtime LUT must contain 2112 bytes");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 2 && sign_mode == 0,
                    "NPQ0-S requires a 2-bit state and no sign stream");
    } else {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 4,
                    "NVQ3 codebook must have shape [256,4]");
    }

    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + gs - 1) / gs;
    const int64_t nvec = (neuron_len + (is_d4_format(format) ? 3 : 7)) /
                         (is_d4_format(format) ? 4 : 8);
    const int64_t nsign = (neuron_len + 7) / 8;
    const bool exec_layout = format == kNvq2Exec || format == kNvq2JscExec;
    const bool group_exec_layout =
        format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec;
    const int index_bits = format_index_bits(format);
    const bool dense=allow_dense && dense_nvq_format(int(format)) && aux.numel()==0 && sub_scale.numel()==0;
    if(dense) {
        TYLOINFER_RUNTIME_CHECK(sub_bits==4 && (!is_d4_format(int(format)) || nvec==2*nsign),
            "compact NVQ requires complete D4 index pairs and 4-bit state");
        const int64_t bytes=(N*(nvec*index_bits+nsign*7+ng*4)+7)/8;
        TYLOINFER_RUNTIME_CHECK(indices.numel()==bytes,"compact NVQ stream length mismatch");
        return;
    }
    const bool nvq1_records=(format==kNvq1L || format==kNvq1S) && aux.numel()==0;
    if(nvq1_records)TYLOINFER_RUNTIME_CHECK(sub_bits==(format==kNvq1S?4:3),"NVQ1 record state width mismatch");
    const int64_t index_bytes = nvq1_records ? N*ng*(format==kNvq1S?4:5) : group_exec_layout
        ? N * ng * (format == kNvq2JscXLGroupExec ? 8 : 12)
        : (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const bool no_aux_format = format == kNpq0L || format == kNpq0S;
    const int aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    const int64_t aux_count = delta_format ? N * ng :
        (no_aux_format ? 0 : N * nsign);
    const int64_t aux_bytes = exec_layout || group_exec_layout || nvq1_records
        ? 0
        : (aux_count * aux_bits + 7) / 8;
    const int64_t sub_bytes = nvq1_records || (group_exec_layout && sub_scale.numel()==0)?
        0:(N * ng * sub_bits + 7) / 8;
    TYLOINFER_RUNTIME_CHECK(indices.numel() == index_bytes, "NVQ index stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(aux.numel() == aux_bytes, "NVQ aux stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(sub_scale.numel() == sub_bytes, "NVQ sub-scale stream length mismatch");
}

#include "nvq_projection_launch.cuh"

NvqDeviceWeight make_device_weight(
    const ::tyloi::DenseTensor & indices,
    const ::tyloi::DenseTensor & aux,
    const ::tyloi::DenseTensor & sub_scale,
    const ::tyloi::DenseTensor & neuron_scale,
    const ::tyloi::DenseTensor & codebook,
    int neuron_len,
    int gs,
    int sub_bits,
    int format,
    int sign_mode) {
    const int ng = (neuron_len + gs - 1) / gs;
    return NvqDeviceWeight{
        indices.data_ptr<uint8_t>(), indices.numel(),
        aux.data_ptr<uint8_t>(), aux.numel(),
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),
        static_cast<int>(codebook.numel()),
        static_cast<int>(neuron_scale.numel()), ng,
        (neuron_len + (is_d4_format(format) ? 3 : 7)) /
            (is_d4_format(format) ? 4 : 8),
        (neuron_len + 7) / 8, sub_bits, sign_mode};
}

template <int FORMAT, int NWARPS, int MAX_M, bool FAST_NVQ2_S4 = false>
void launch_multi2_vec8(
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const ::tyloi::DenseTensor & qx,
    const ::tyloi::DenseTensor & xscale,
    ::tyloi::DenseTensor & output,
    int M,
    cudaStream_t stream) {
    nvq_gemv_multi2_vec8_kernel<FORMAT, NWARPS, MAX_M, FAST_NVQ2_S4>
        <<<first.N + second.N, dim3(32, NWARPS), 0, stream>>>(
            first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), M);
}

void launch_selected_multi2_vec8(
    int format,
    const NvqDeviceWeight & first,
    const NvqDeviceWeight & second,
    const ::tyloi::DenseTensor & qx,
    const ::tyloi::DenseTensor & xscale,
    ::tyloi::DenseTensor & output,
    int M,
    cudaStream_t stream) {
#define NVQ_MULTI2(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M>(first, second, qx, xscale, output, M, stream)
#define NVQ_MULTI2_FAST(F, NW, MAX_M) \
    launch_multi2_vec8<F, NW, MAX_M, true>(first, second, qx, xscale, output, M, stream)
    const bool fast_nvq2_s4 =
        (format == kNvq2 || format == kNvq2Exec ||
         format == kNvq2Jsc || format == kNvq2JscExec) &&
        first.sub_bits == 4 && second.sub_bits == 4;
    if (M == 1) {
        switch (format) {
            case kNvq1L: NVQ_MULTI2(kNvq1L, 4, 1); break;
            case kNvq2:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2, 4, 1);
                else NVQ_MULTI2(kNvq2, 4, 1);
                break;
            case kNvq3: NVQ_MULTI2(kNvq3, 4, 1); break;
            case kNvq2Exec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Exec, 4, 1);
                else NVQ_MULTI2(kNvq2Exec, 4, 1);
                break;
            case kNvq2Jsc:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2Jsc, 4, 1);
                else NVQ_MULTI2(kNvq2Jsc, 4, 1);
                break;
            case kNvq2JscExec:
                if (fast_nvq2_s4) NVQ_MULTI2_FAST(kNvq2JscExec, 4, 1);
                else NVQ_MULTI2(kNvq2JscExec, 4, 1);
                break;
            case kNpq0L: NVQ_MULTI2(kNpq0L, 4, 1); break;
            case kNvq1S: NVQ_MULTI2(kNvq1S, 4, 1); break;
            case kNpq0S: NVQ_MULTI2(kNpq0S, 4, 1); break;
            case kNvq3Jsc: NVQ_MULTI2(kNvq3Jsc, 4, 1); break;
            case kNvq3Jsc2: NVQ_MULTI2(kNvq3Jsc2, 4, 1); break;
            case kNvq3Jsc512: NVQ_MULTI2(kNvq3Jsc512, 4, 1); break;
            case kNvq2JscL: NVQ_MULTI2(kNvq2JscL, 4, 1); break;
            case kNvq2JscXL: NVQ_MULTI2(kNvq2JscXL, 4, 1); break;
            case kNvq3JscL: NVQ_MULTI2(kNvq3JscL, 4, 1); break;
            case kNvq2JscXLGroupExec:
                launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
                    first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), stream);
                break;
            case kNvq3JscLGroupExec:
                launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
                    first, second, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                    reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), stream);
                break;
        }
    } else if (M <= 4) {
        if (format == kNvq1L || format == kNvq1S) {
            if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 4);
            else NVQ_MULTI2(kNvq1L, 8, 4);
        } else if (is_e8_format(format)) {
            if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 4);
            else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 4);
            else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 4);
            else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 4);
            else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 4);
            else if (format == kNvq2JscXLGroupExec) {
                NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq2, 4, 4);
        } else if (format == kNpq0L || format == kNpq0S) {
            if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 4);
            else NVQ_MULTI2(kNpq0L, 4, 4);
        } else if (M <= 2) {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 2, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 2, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 2, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 2, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 2, 4);
            }
            else NVQ_MULTI2(kNvq3, 2, 4);
        } else {
            if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 4, 4);
            else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 4, 4);
            else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 4, 4);
            else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 4, 4);
            else if (format == kNvq3JscLGroupExec) {
                NVQ_MULTI2(kNvq3JscLGroupExec, 4, 4);
            }
            else NVQ_MULTI2(kNvq3, 4, 4);
        }
    } else if (format == kNvq1L || format == kNvq1S) {
        if (format == kNvq1S) NVQ_MULTI2(kNvq1S, 8, 8);
        else NVQ_MULTI2(kNvq1L, 8, 8);
    } else if (is_e8_format(format)) {
        if (format == kNvq2Exec) NVQ_MULTI2(kNvq2Exec, 4, 8);
        else if (format == kNvq2Jsc) NVQ_MULTI2(kNvq2Jsc, 4, 8);
        else if (format == kNvq2JscExec) NVQ_MULTI2(kNvq2JscExec, 4, 8);
        else if (format == kNvq2JscL) NVQ_MULTI2(kNvq2JscL, 4, 8);
        else if (format == kNvq2JscXL) NVQ_MULTI2(kNvq2JscXL, 4, 8);
        else if (format == kNvq2JscXLGroupExec) {
            NVQ_MULTI2(kNvq2JscXLGroupExec, 4, 8);
        }
        else NVQ_MULTI2(kNvq2, 4, 8);
    } else if (format == kNpq0L || format == kNpq0S) {
        if (format == kNpq0S) NVQ_MULTI2(kNpq0S, 4, 8);
        else NVQ_MULTI2(kNpq0L, 4, 8);
    } else {
        if (format == kNvq3Jsc2) NVQ_MULTI2(kNvq3Jsc2, 8, 8);
        else if (format == kNvq3Jsc) NVQ_MULTI2(kNvq3Jsc, 8, 8);
        else if (format == kNvq3Jsc512) NVQ_MULTI2(kNvq3Jsc512, 8, 8);
        else if (format == kNvq3JscL) NVQ_MULTI2(kNvq3JscL, 8, 8);
        else if (format == kNvq3JscLGroupExec) {
            NVQ_MULTI2(kNvq3JscLGroupExec, 8, 8);
        }
        else NVQ_MULTI2(kNvq3, 8, 8);
    }
#undef NVQ_MULTI2
#undef NVQ_MULTI2_FAST
}

}  // namespace


::tyloi::DenseTensor nepq_gemv_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K && M >= 1 && M <= 16,
                "NEPQ GEMV input must be [M,K] with M in [1,16]");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);

#define NEPQ_GEMV_LAUNCH(NWARPS_VALUE, MAX_M_VALUE)                               \
    nepq_gemv_batch_vec8_kernel<F, NWARPS_VALUE, MAX_M_VALUE><<<                 \
        N, dim3(32, NWARPS_VALUE), 0, stream>>>(                                  \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),            \
        bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                      \
        xscale.data_ptr<float>(),                                                 \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                  \
        M, N, ng, nvec, nsign, nsuper, table_stride,                              \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        const int nwarps = (format == kNvq1S && N <= 2048) ? 8 :
            ((format == kNpq0S && N <= 1024) ? 2 : 4);
        if (M == 1) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 1);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 1);
            else NEPQ_GEMV_LAUNCH(4, 1);
        } else if (M <= 2) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 2);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 2);
            else NEPQ_GEMV_LAUNCH(4, 2);
        } else if (M <= 4) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 4);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 4);
            else NEPQ_GEMV_LAUNCH(4, 4);
        } else if (M <= 8) {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 8);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 8);
            else NEPQ_GEMV_LAUNCH(4, 8);
        } else {
            if (nwarps == 8) NEPQ_GEMV_LAUNCH(8, 16);
            else if (nwarps == 2) NEPQ_GEMV_LAUNCH(2, 16);
            else NEPQ_GEMV_LAUNCH(4, 16);
        }
    });
#undef NEPQ_GEMV_LAUNCH
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ GEMV kernel launch failed");
    return output;
}

::tyloi::DenseTensor nepq_mmq_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K && M >= 4 && M <= 64,
                "NEPQ MMQ input must be [M,K] with M in [4,64]");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), M, K, ng);
    const dim3 block(32, 8);
#define NEPQ_MMQ_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                           \
    nepq_mmq_mma24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<                  \
        dim3((N + 63) / 64, 1), block, 0, stream>>>(                             \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),           \
        bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                     \
        xscale.data_ptr<float>(),                                                \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                 \
        M, N, ng, nvec, nsign, nsuper, table_stride,                             \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0) NEPQ_MMQ_LAUNCH(1, false);
            else NEPQ_MMQ_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0) NEPQ_MMQ_LAUNCH(2, false);
            else NEPQ_MMQ_LAUNCH(2, true);
        } else if (M <= 48) {
            if (M == 48 && N % 64 == 0) NEPQ_MMQ_LAUNCH(3, false);
            else NEPQ_MMQ_LAUNCH(3, true);
        } else {
            if (M == 64 && N % 64 == 0) NEPQ_MMQ_LAUNCH(4, false);
            else NEPQ_MMQ_LAUNCH(4, true);
        }
    });
#undef NEPQ_MMQ_LAUNCH
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ MMQ kernel launch failed");
    return output;
}

::tyloi::DenseTensor nepq_gemm_f16_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K && M >= 16 && M <= 256,
                "NEPQ online GEMM input must be [M,K] with M in [16,256]");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    const dim3 block(32, 8);
#define NEPQ_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE)                          \
    nepq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<<              \
        dim3((N + 63) / 64, (M + 16 * MTILES_VALUE - 1) / (16 * MTILES_VALUE)),  \
        block, 0, stream>>>(                                                       \
        indices.data_ptr<uint8_t>(), indices.numel(),                             \
        aux.data_ptr<uint8_t>(), aux.numel(),                                     \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                         \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),            \
        bank_ids.data_ptr<uint8_t>(),                                             \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                 \
        reinterpret_cast<__half *>(output.data_ptr<mfq_half>()),                  \
        M, N, K, ng, nvec, nsign, nsuper, table_stride,                           \
        static_cast<int>(sub_bits), 0)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        if (M <= 16) {
            if (M == 16 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(1, false);
            else NEPQ_GEMM_LAUNCH(1, true);
        } else if (M <= 32) {
            if (M == 32 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(2, false);
            else NEPQ_GEMM_LAUNCH(2, true);
        } else if (M <= 64) {
            if (M == 64 && N % 64 == 0 && ng % 4 == 0) NEPQ_GEMM_LAUNCH(4, false);
            else NEPQ_GEMM_LAUNCH(4, true);
        } else {
            if (M % 128 == 0 && N % 64 == 0 && ng % 4 == 0) {
                NEPQ_GEMM_LAUNCH(8, false);
            } else {
                NEPQ_GEMM_LAUNCH(8, true);
            }
        }
    });
#undef NEPQ_GEMM_LAUNCH
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NEPQ online GEMM kernel launch failed");
    return output;
}

::tyloi::DenseTensor nepq_moe_grouped_matmul_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids,
    ::tyloi::DenseTensor grouped_table_pool,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = format == kNpq0S
        ? kNpq0STableBytes
        : static_cast<int>(table_pool.size(1));
    TYLOINFER_RUNTIME_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == ::tyloi::kInt8 &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    TYLOINFER_RUNTIME_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ expert count must be in [1,4096]");
    TYLOINFER_RUNTIME_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == n_experts * out_per_expert,
                "NEPQ expert shape does not match the flattened row count");
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == ::tyloi::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TYLOINFER_RUNTIME_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    TYLOINFER_RUNTIME_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    TYLOINFER_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == ::tyloi::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == ::tyloi::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == ::tyloi::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int grouped_table_stride =
        static_cast<int>(grouped_table_pool.size(1));
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
        qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);

    if (tokens <= 8) {
        constexpr int rows_per_block = 2;
        const int row_blocks = (static_cast<int>(out_per_expert) + rows_per_block - 1) /
            rows_per_block;
#define NEPQ_MOE_SMALL_LAUNCH(NWARPS_VALUE)                                      \
        nepq_moe_mmvq_kernel<F, NWARPS_VALUE, rows_per_block><<<                 \
            dim3(row_blocks, pairs), dim3(32, NWARPS_VALUE), 0, stream>>>(       \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),       \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                   \
            nullptr,                                                             \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(n_experts),                                         \
            static_cast<int>(out_per_expert), ng, nvec, nsign, nsuper,           \
            table_stride, static_cast<int>(sub_bits), routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (K >= 4096) NEPQ_MOE_SMALL_LAUNCH(8);
            else NEPQ_MOE_SMALL_LAUNCH(4);
        });
#undef NEPQ_MOE_SMALL_LAUNCH
    } else {
        TYLOINFER_RUNTIME_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == ::tyloi::kInt32 && ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        TYLOINFER_RUNTIME_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == ::tyloi::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        TYLOINFER_RUNTIME_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == ::tyloi::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        TYLOINFER_RUNTIME_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == ::tyloi::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(std::min<int64_t>(max_tasks, 4096));
#define NEPQ_MOE_GROUPED_LAUNCH()                                                \
        nepq_moe_grouped_tile_kernel<F, route_tile><<<                           \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), grouped_table_pool.data_ptr<int8_t>(), \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), nullptr, ids_dst.data_ptr<int32_t>(),      \
            expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),  \
            tile_experts.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(n_experts),                                         \
            static_cast<int>(out_per_expert), ng, nvec, nsign, nsuper,           \
            grouped_table_stride, static_cast<int>(sub_bits), max_tiles,         \
            row_tiles,                                                           \
            routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NEPQ_MOE_GROUPED_LAUNCH();
        });
#undef NEPQ_MOE_GROUPED_LAUNCH
    }
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ routed matmul kernel launch failed");
    return out;
}

// Route-compacted online-dequant FP16 Tensor Core path for mixed NVQ pools.
// Each task reuses one 64/128-column weight tile across 16 to 256 routed rows.
// Keep the format switch outside the inner K loop so one launch can serve
// every NVQ cohort without a runtime branch per decoded value.
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

::tyloi::DenseTensor nvq_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor codebook, ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    ::tyloi::DenseTensor out, ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode,true);
    TYLOINFER_RUNTIME_CHECK(gs == kGroupSize, "NVQ routed FP16 requires gs24");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == ::tyloi::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NVQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    const int max_tiles = (pairs + 7) / 8 + static_cast<int>(n_experts);
    const int ntiles_n = (static_cast<int>(out_per_expert) + 63) / 64;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(max_tiles) * ntiles_n, 4096));
    const dim3 threads(32, 8);
    const cudaStream_t stream = mfq_current_cuda_stream();
#define NVQ_MOE_F16_LAUNCH()                                                     \
    nvq_moe_grouped_f16_kernel<F><<<blocks, threads, 0, stream>>>(               \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),             \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                \
        expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),           \
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),      \
        tile_experts.data_ptr<int32_t>(),                                        \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                    \
        routes, static_cast<int>(n_experts), static_cast<int>(pool_experts),     \
        static_cast<int>(out_per_expert), K, ng, nvec, nsign,                   \
        static_cast<int>(sub_bits), static_cast<int>(sign_mode),                 \
        max_tiles, x.dim() == 3)
    launch_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        NVQ_MOE_F16_LAUNCH();
    });
#undef NVQ_MOE_F16_LAUNCH
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

::tyloi::DenseTensor nvq_moe_grouped_matmul_hetero_f16_cuda(
    ::tyloi::DenseTensor weight_ptrs,
    ::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params,
    ::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local,
    ::tyloi::DenseTensor x,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t route_tile_m,
    ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts,
    int64_t format_group,
    bool masked_experts) {
    TYLOINFER_RUNTIME_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "NVQ heterogeneous expert count must be in [1,4096]");
    TYLOINFER_RUNTIME_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX &&
            neuron_len > 0 && neuron_len <= INT_MAX,
        "NVQ heterogeneous dimensions must fit int32");
    TYLOINFER_RUNTIME_CHECK(
        route_tile_m == 8 || route_tile_m == 64 || route_tile_m == 128,
        "NVQ heterogeneous route tile must be 8, 64, or 128");
    TYLOINFER_RUNTIME_CHECK(
        format_group >= kNvqMoeF16AllFormats &&
            format_group <= kNvqMoeF16LegacyFormats,
        "NVQ heterogeneous FP16 format group is invalid");
    TYLOINFER_RUNTIME_CHECK(
        weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
            weight_ptrs.scalar_type() == ::tyloi::kInt64 &&
            weight_ptrs.dim() == 2 && weight_ptrs.size(0) > 0 &&
            weight_ptrs.size(1) == 5,
        "NVQ heterogeneous weight pointers must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TYLOINFER_RUNTIME_CHECK(
        weight_sizes.is_cuda() && weight_sizes.is_contiguous() &&
            weight_sizes.scalar_type() == ::tyloi::kInt64 &&
            weight_sizes.dim() == 2 && weight_sizes.size(0) == pools &&
            weight_sizes.size(1) == 3,
        "NVQ heterogeneous weight sizes must be CUDA int64 [pools,3]");
    TYLOINFER_RUNTIME_CHECK(
        pool_params.is_cuda() && pool_params.is_contiguous() &&
            pool_params.scalar_type() == ::tyloi::kInt32 &&
            pool_params.dim() == 2 && pool_params.size(0) == pools &&
            pool_params.size(1) == 7,
        "NVQ heterogeneous pool parameters must be CUDA int32 [pools,7]");
    TYLOINFER_RUNTIME_CHECK(
        expert_pool.is_cuda() && expert_pool.is_contiguous() &&
            expert_pool.scalar_type() == ::tyloi::kInt32 &&
            expert_pool.numel() == n_experts,
        "NVQ heterogeneous expert-pool map must be CUDA int32 [experts]");
    TYLOINFER_RUNTIME_CHECK(
        expert_local.is_cuda() && expert_local.is_contiguous() &&
            expert_local.scalar_type() == ::tyloi::kInt32 &&
            expert_local.numel() == n_experts,
        "NVQ heterogeneous expert-local map must be CUDA int32 [experts]");
    TYLOINFER_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == ::tyloi::kFloat16 &&
            (x.dim() == 2 || x.dim() == 3) &&
            x.size(-1) == neuron_len,
        "NVQ heterogeneous FP16 input shape mismatch");
    TYLOINFER_RUNTIME_CHECK(
        out.is_cuda() && out.is_contiguous() &&
            out.scalar_type() == ::tyloi::kFloat16 &&
            out.dim() == 3 && out.size(2) == out_per_expert,
        "NVQ heterogeneous output must be CUDA FP16 [tokens,routes,out]");
    const int tokens = static_cast<int>(out.size(0));
    const int routes = static_cast<int>(out.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TYLOINFER_RUNTIME_CHECK(
        tokens > 8 && routes > 0 &&
            ((!routed_input && x.size(0) == tokens) ||
             (routed_input && x.size(0) == tokens && x.size(1) == routes)),
        "NVQ heterogeneous input leading dimensions do not match output");
    TYLOINFER_RUNTIME_CHECK(
        ids_dst.is_cuda() && ids_dst.is_contiguous() &&
            ids_dst.scalar_type() == ::tyloi::kInt32 &&
            ids_dst.numel() >= pairs,
        "NVQ heterogeneous compact route map is missing");
    TYLOINFER_RUNTIME_CHECK(
        expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
            expert_bounds.scalar_type() == ::tyloi::kInt32 &&
            expert_bounds.numel() >= n_experts + 1,
        "NVQ heterogeneous expert bounds are missing");
    TYLOINFER_RUNTIME_CHECK(
        tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
            tile_bounds.scalar_type() == ::tyloi::kInt32 &&
            tile_bounds.numel() >= n_experts + 1,
        "NVQ heterogeneous tile bounds are missing");
    TYLOINFER_RUNTIME_CHECK(
        tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == ::tyloi::kInt32 &&
            tile_experts.numel() >= pairs,
        "NVQ heterogeneous tile expert map is missing");
    for (const auto * tensor : {
             &weight_sizes, &pool_params, &expert_pool, &expert_local,
             &x, &out, &ids_dst, &expert_bounds, &tile_bounds,
             &tile_experts}) {
        TYLOINFER_RUNTIME_CHECK(
            tensor->device() == weight_ptrs.device(),
            "NVQ heterogeneous tensors must share one CUDA device");
    }

    const int tile_m = static_cast<int>(route_tile_m);
    const int rows_per_expert = std::max(
        1, (pairs + static_cast<int>(n_experts) - 1) /
               static_cast<int>(n_experts));
    const int fine_bm = rows_per_expert <= 16
        ? 16
        : (rows_per_expert <= 32 ? 32 : 64);
    const int max_tiles =
        (pairs + tile_m - 1) / tile_m + static_cast<int>(n_experts);
    const int base_block_cap = pairs >= 32768 ? 8192 : 4096;
    const int64_t route_tiles =
        (pairs + tile_m - 1) / tile_m;
    const int narrow_ntiles_n =
        (static_cast<int>(out_per_expert) + 63) / 64;
    // Use two smaller resident CTAs only while each useful tile gets its own
    // CTA; longer routes retain the wider tile's activation reuse.
    const bool use_narrow_tile = tile_m == 128 &&
        route_tiles * narrow_ntiles_n <= base_block_cap;
    const int tile_n = use_narrow_tile
        ? 64 : kNvqMoeF16HeteroTileN;
    const int ntiles_n =
        (static_cast<int>(out_per_expert) + tile_n - 1) / tile_n;
    const int64_t useful_tiles =
        (pairs + tile_m - 1) / tile_m;
    const int64_t useful_tasks = useful_tiles * ntiles_n;
    int block_cap = std::max(
        base_block_cap,
        static_cast<int>(std::min<int64_t>(
            INT_MAX, (useful_tasks + 3) / 4)));
    const int64_t max_tasks =
        static_cast<int64_t>(max_tiles) * ntiles_n;
    if (max_tasks > block_cap) {
        const int average_route_tiles =
            (rows_per_expert + tile_m - 1) / tile_m;
        const int task_period = average_route_tiles * ntiles_n;
        // Rotate CTAs between expert phases when the grid stride would
        // otherwise repeat the same valid/empty task pattern.  For a fully
        // covered multi-pool tensor, one extra CTA breaks a repeated pool
        // phase without changing single-pool launches.
        if (task_period > 1 && block_cap % task_period == 0) {
            if (masked_experts && tile_m != 8 && task_period >= 64) {
                block_cap += task_period;
            } else if (tile_m == 8 && base_block_cap == 4096 &&
                       rows_per_expert > 16) {
                block_cap += std::min(16, std::max(1, task_period / 4));
            } else if (pools > 1 && task_period >= 256) {
                ++block_cap;
            }
        }
    }
    const int blocks = static_cast<int>(std::max<int64_t>(
        1, std::min<int64_t>(max_tasks, block_cap)));
#define NVQ_MOE_HETERO_F16_LAUNCH(                                             \
    BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE, FORMAT_GROUP_VALUE)          \
    do {                                                                         \
        constexpr int kSharedBytes = sizeof(NvqMoeF16SharedStorage<             \
            BM_VALUE, BN_VALUE, GROUP_VALUE>);                                  \
        if constexpr (kSharedBytes > 48 * 1024) {                               \
            static const cudaError_t attribute_status = cudaFuncSetAttribute(   \
                nvq_moe_grouped_hetero_f16_kernel<                              \
                    BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE,               \
                    FORMAT_GROUP_VALUE, true>,                                  \
                cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);     \
            TYLOINFER_RUNTIME_CHECK(                                                   \
                attribute_status == cudaSuccess,                                \
                "failed to opt in to the NVQ MoE shared-memory size");          \
        }                                                                        \
        nvq_moe_grouped_hetero_f16_kernel<                                      \
            BM_VALUE, BN_VALUE, ROUTE_VALUE, GROUP_VALUE,                       \
            FORMAT_GROUP_VALUE, true><<<                                        \
            blocks, dim3(32, (BM_VALUE >= 128 ? BM_VALUE / 64 : 1) *            \
                (BN_VALUE / 16)), kSharedBytes,                                 \
            mfq_current_cuda_stream()>>>(                                       \
        weight_ptrs.data_ptr<int64_t>(), weight_sizes.data_ptr<int64_t>(),      \
        pool_params.data_ptr<int32_t>(), expert_pool.data_ptr<int32_t>(),       \
        expert_local.data_ptr<int32_t>(),                                      \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),              \
        ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),         \
        tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),      \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()), routes,          \
        static_cast<int>(n_experts), pools,                                    \
        static_cast<int>(out_per_expert), static_cast<int>(neuron_len),        \
        max_tiles, routed_input);                                                \
    } while (false)
    // A bounded format switch fits the wide G2 task in 64 registers, so two
    // BM128 CTAs can reside together.  Cross-group mixtures keep the full
    // switch, G4 path, and its wider K reuse.
    const char* wide_rows_option = tyloi::cuda::env_lookup("TYLOINFER_NVQ_PREFILL_M256");
    const bool wide_rows = wide_rows_option && std::atoi(wide_rows_option) != 0;
    const char* narrow_g2_option = tyloi::cuda::env_lookup("TYLOINFER_NVQ_PREFILL_NARROW_G2");
    const bool narrow_g2 = narrow_g2_option && std::atoi(narrow_g2_option) != 0;
    const char* cohorts_option = tyloi::cuda::env_lookup("TYLOINFER_NVQ_PREFILL_COHORTS");
    const bool cohorts = !cohorts_option || std::atoi(cohorts_option) != 0;
    // Keep the M/N geometry and Tensor Core sum order. A shorter K tile
    // lowers decode liveness and shared storage, allowing three narrow CTAs.
    if (cohorts && tile_m == 128 && format_group == kNvqMoeF16AllFormats) {
        // Disjoint format switches write disjoint expert rows. Every task
        // retains the canonical Tensor Core K sequence, with no new maps
        // or weight buffers. Keep G4 for narrow tiles and the legacy decoder.
        if (use_narrow_tile) {
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16StandardFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16ExtendedFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,64,128,4,kNvqMoeF16LegacyFormats);
        } else {
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,2,kNvqMoeF16StandardFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,2,kNvqMoeF16ExtendedFormats);
            NVQ_MOE_HETERO_F16_LAUNCH(128,128,128,4,kNvqMoeF16LegacyFormats);
        }
    } else if (narrow_g2 && tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 2, kNvqMoeF16StandardFormats);
    } else if (narrow_g2 && tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16ExtendedFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 2, kNvqMoeF16ExtendedFormats);
    } else if (wide_rows && tile_m == 128 && use_narrow_tile && rows_per_expert >= 128 &&
            out_per_expert <= neuron_len) {
        if (format_group == kNvqMoeF16StandardFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16StandardFormats);
        } else if (format_group == kNvqMoeF16ExtendedFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16ExtendedFormats);
        } else if (format_group == kNvqMoeF16LegacyFormats) {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16LegacyFormats);
        } else {
            NVQ_MOE_HETERO_F16_LAUNCH(
                256, 64, 128, 4, kNvqMoeF16AllFormats);
        }
    } else if (tile_m == 128 && use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 4, kNvqMoeF16StandardFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16StandardFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16StandardFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16ExtendedFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16ExtendedFormats);
    } else if (tile_m == 128 && !use_narrow_tile &&
            format_group == kNvqMoeF16LegacyFormats) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 2, kNvqMoeF16LegacyFormats);
    } else if (tile_m == 128 && use_narrow_tile) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 64, 128, 4, kNvqMoeF16AllFormats);
    } else if (tile_m == 128) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 128, 4, kNvqMoeF16AllFormats);
    } else if (tile_m == 64) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            128, 128, 64, 4, kNvqMoeF16AllFormats);
    } else if (fine_bm == 16) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            16, 128, 8, 4, kNvqMoeF16AllFormats);
    } else if (fine_bm == 32) {
        NVQ_MOE_HETERO_F16_LAUNCH(
            32, 128, 8, 4, kNvqMoeF16AllFormats);
    } else {
        NVQ_MOE_HETERO_F16_LAUNCH(
            64, 128, 8, 4, kNvqMoeF16AllFormats);
    }
#undef NVQ_MOE_HETERO_F16_LAUNCH
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

static ::tyloi::DenseTensor nvq_moe_grouped_matmul_hetero_impl(
    ::tyloi::DenseTensor weight_ptrs,
    ::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params,
    ::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    bool input_quantized,
    ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,::tyloi::DenseTensor active_plan) {
    TYLOINFER_RUNTIME_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "NVQ heterogeneous expert count must be in [1,4096]");
    TYLOINFER_RUNTIME_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX &&
            neuron_len > 0 && neuron_len <= INT_MAX,
        "NVQ heterogeneous dimensions must fit int32");
    TYLOINFER_RUNTIME_CHECK(
        weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
            weight_ptrs.scalar_type() == ::tyloi::kInt64 &&
            weight_ptrs.dim() == 2 && weight_ptrs.size(0) > 0 &&
            weight_ptrs.size(1) == 5,
        "NVQ heterogeneous weight pointers must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TYLOINFER_RUNTIME_CHECK(
        weight_sizes.is_cuda() && weight_sizes.is_contiguous() &&
            weight_sizes.scalar_type() == ::tyloi::kInt64 &&
            weight_sizes.dim() == 2 && weight_sizes.size(0) == pools &&
            weight_sizes.size(1) == 3,
        "NVQ heterogeneous weight sizes must be CUDA int64 [pools,3]");
    TYLOINFER_RUNTIME_CHECK(
        pool_params.is_cuda() && pool_params.is_contiguous() &&
            pool_params.scalar_type() == ::tyloi::kInt32 &&
            pool_params.dim() == 2 && pool_params.size(0) == pools &&
            pool_params.size(1) == 7,
        "NVQ heterogeneous pool parameters must be CUDA int32 [pools,7]");
    TYLOINFER_RUNTIME_CHECK(
        expert_pool.is_cuda() && expert_pool.is_contiguous() &&
            expert_pool.scalar_type() == ::tyloi::kInt32 &&
            expert_pool.numel() == n_experts &&
            expert_local.is_cuda() && expert_local.is_contiguous() &&
            expert_local.scalar_type() == ::tyloi::kInt32 &&
            expert_local.numel() == n_experts,
        "NVQ heterogeneous expert maps must be CUDA int32 [experts]");
    TYLOINFER_RUNTIME_CHECK(
        ids.is_cuda() && ids.is_contiguous() &&
            ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2,
        "NVQ heterogeneous route IDs must be CUDA int32 [tokens,routes]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    TYLOINFER_RUNTIME_CHECK(
        tokens > 0 && tokens <= 8 && routes > 0,
        "NVQ heterogeneous MMVQ requires one to eight tokens");
    TYLOINFER_RUNTIME_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == ::tyloi::kFloat16 &&
            (x.dim() == 2 || x.dim() == 3) &&
            (input_quantized || x.size(-1) == neuron_len),
        "NVQ heterogeneous routed input shape mismatch");
    const bool routed_input = x.dim() == 3;
    TYLOINFER_RUNTIME_CHECK(
        (!routed_input && x.size(0) == tokens) ||
            (routed_input && x.size(0) == tokens && x.size(1) == routes),
        "NVQ heterogeneous input leading dimensions do not match routes");
    TYLOINFER_RUNTIME_CHECK(
        out.is_cuda() && out.is_contiguous() &&
            out.scalar_type() == ::tyloi::kFloat16 &&
            out.dim() == 3 && out.size(0) == tokens &&
            out.size(1) == routes && out.size(2) == out_per_expert,
        "NVQ heterogeneous output must be CUDA FP16 [tokens,routes,out]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    TYLOINFER_RUNTIME_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
            qx.scalar_type() == ::tyloi::kInt8 && qx.dim() == 2 &&
            qx.size(0) >= input_rows && qx.size(1) >= K_pad,
        "NVQ heterogeneous qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
            xscale.scalar_type() == ::tyloi::kFloat32 &&
            xscale.dim() == 2 && xscale.size(0) >= input_rows &&
            xscale.size(1) >= ng,
        "NVQ heterogeneous xscale workspace mismatch");
    for (const auto * tensor : {
             &weight_sizes, &pool_params, &expert_pool, &expert_local,
             &x, &ids, &out, &qx, &xscale}) {
        TYLOINFER_RUNTIME_CHECK(
            tensor->device() == weight_ptrs.device(),
            "NVQ heterogeneous tensors must share one CUDA device");
    }

    if(active_plan.defined())TYLOINFER_RUNTIME_CHECK(active_plan.is_cuda() && active_plan.is_contiguous() &&
        active_plan.scalar_type()==::tyloi::kInt32 && active_plan.device()==x.device() &&
        active_plan.numel()==1+int64_t(pairs)*3,"NVQ active route plan geometry/device mismatch");
    const cudaStream_t stream = mfq_current_cuda_stream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }
#define NVQ_MOE_HETERO_MMVQ_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                   \
    nvq_moe_mmvq_hetero_kernel<                                                \
        NWARPS_VALUE, ROWS_VALUE, false><<<                                    \
        dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) / ROWS_VALUE, pairs), \
        dim3(32, NWARPS_VALUE), 0, stream>>>(                                  \
        weight_ptrs.data_ptr<int64_t>(), weight_sizes.data_ptr<int64_t>(),     \
        pool_params.data_ptr<int32_t>(), expert_pool.data_ptr<int32_t>(),      \
        expert_local.data_ptr<int32_t>(), qx.data_ptr<int8_t>(),               \
        xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                     \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()), pairs, routes,   \
        static_cast<int>(n_experts), pools, static_cast<int>(out_per_expert),  \
        routed_input)
    const char* grouped=tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_GROUP_DOT");
    if(ng<=128 && out_per_expert<=4096 && (!grouped || grouped[0]!='0')) {
        tyloi::cuda::NvqRoutedReuseParams reuse;
        reuse.pointers=weight_ptrs.data_ptr<int64_t>();reuse.sizes=weight_sizes.data_ptr<int64_t>();
        reuse.params=pool_params.data_ptr<int32_t>();reuse.expert_pool=expert_pool.data_ptr<int32_t>();
        reuse.expert_local=expert_local.data_ptr<int32_t>();reuse.ids=ids.data_ptr<int32_t>();
        reuse.active=active_plan.defined()?active_plan.data_ptr<int32_t>():nullptr;
        reuse.input=qx.data_ptr<int8_t>();reuse.scales=xscale.data_ptr<float>();
        reuse.output=reinterpret_cast<half*>(out.data_ptr<mfq_half>());
        reuse.experts=int(n_experts);reuse.pools=pools;reuse.rows=int(out_per_expert);
        reuse.routes=routes;reuse.pairs=pairs;reuse.down=routed_input;
        if(!tyloi::cuda::nvq_launch_routed_reuse(reuse,stream))
            nvq_moe_groups_hetero_kernel<<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
            weight_ptrs.data_ptr<int64_t>(),weight_sizes.data_ptr<int64_t>(),pool_params.data_ptr<int32_t>(),
            expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),
            qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),reinterpret_cast<half*>(out.data_ptr<mfq_half>()),
            active_plan.defined()?active_plan.data_ptr<int32_t>():nullptr,
            int(n_experts),pools,int(out_per_expert),routes,pairs,routed_input);
    }else if(active_plan.defined()) {
        const auto launch=[&](auto warp_tag) {
            constexpr int warps=decltype(warp_tag)::value;
            const int blocks=int(std::min<int64_t>(nvq_active_resident_blocks<warps>(),
                ((out_per_expert-1)/2+1)*pairs));
            nvq_moe_mmvq_hetero_kernel<warps,2,false,true><<<blocks,dim3(32,warps),0,stream>>>(
                weight_ptrs.data_ptr<int64_t>(),weight_sizes.data_ptr<int64_t>(),pool_params.data_ptr<int32_t>(),
                expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),qx.data_ptr<int8_t>(),
                xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),reinterpret_cast<__half*>(out.data_ptr<mfq_half>()),
                pairs,routes,static_cast<int>(n_experts),pools,static_cast<int>(out_per_expert),routed_input,
                active_plan.data_ptr<int32_t>());
        };
        if(K>=4096)launch(std::integral_constant<int,8>{});else launch(std::integral_constant<int,4>{});
    } else if (K >= 4096) NVQ_MOE_HETERO_MMVQ_LAUNCH(8,2);
    else {
        // Keep each row's four-warp accumulation and reduction unchanged.
        // The opt-in sweep varies only the number of rows sharing a CTA.
        const auto* requested=tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_ROW_TILE");
        const int rows=tokens==1 && requested ? std::atoi(requested) : 2;
        switch(rows) {
            case 1: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,1);break;
            case 4: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,4);break;
            case 8: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,8);break;
            case 16: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,16);break;
            case 32: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,32);break;
            default: NVQ_MOE_HETERO_MMVQ_LAUNCH(4,2);break;
        }
    }
#undef NVQ_MOE_HETERO_MMVQ_LAUNCH
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

::tyloi::DenseTensor nvq_moe_grouped_matmul_hetero_ws_cuda(
    ::tyloi::DenseTensor weight_ptrs,::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params,::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local,::tyloi::DenseTensor x,::tyloi::DenseTensor ids,
    int64_t n_experts,int64_t out_per_expert,int64_t neuron_len,bool input_quantized,
    ::tyloi::DenseTensor out,::tyloi::DenseTensor qx,::tyloi::DenseTensor xscale) {
    return nvq_moe_grouped_matmul_hetero_impl(weight_ptrs,weight_sizes,pool_params,expert_pool,expert_local,
        x,ids,n_experts,out_per_expert,neuron_len,input_quantized,out,qx,xscale,{});
}
::tyloi::DenseTensor nvq_moe_grouped_matmul_active_ws_cuda(
    ::tyloi::DenseTensor weight_ptrs,::tyloi::DenseTensor weight_sizes,
    ::tyloi::DenseTensor pool_params,::tyloi::DenseTensor expert_pool,
    ::tyloi::DenseTensor expert_local,::tyloi::DenseTensor x,::tyloi::DenseTensor ids,
    int64_t n_experts,int64_t out_per_expert,int64_t neuron_len,bool input_quantized,
    ::tyloi::DenseTensor out,::tyloi::DenseTensor qx,::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor active_plan) {
    TYLOINFER_RUNTIME_CHECK(active_plan.defined(),"NVQ active route plan required");
    return nvq_moe_grouped_matmul_hetero_impl(weight_ptrs,weight_sizes,pool_params,expert_pool,expert_local,
        x,ids,n_experts,out_per_expert,neuron_len,input_quantized,out,qx,xscale,active_plan);
}

::tyloi::DenseTensor nepq_moe_grouped_matmul_pool_f16_cuda(
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux, ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale, ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids, ::tyloi::DenseTensor x, ::tyloi::DenseTensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    ::tyloi::DenseTensor out, ::tyloi::DenseTensor ids_dst, ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds, ::tyloi::DenseTensor tile_experts) {
    check_nepq_common(indices, aux, sub_scale, neuron_scale, table_pool,
                      bank_ids, neuron_len, sub_bits, format);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == ::tyloi::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NEPQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int max_tiles = (pairs + 7) / 8 + static_cast<int>(n_experts);
    const int ntiles_n = (static_cast<int>(out_per_expert) + 63) / 64;
    const int blocks = static_cast<int>(std::min<int64_t>(
        static_cast<int64_t>(max_tiles) * ntiles_n, 4096));
    const dim3 threads(32, 8);
    const cudaStream_t stream = mfq_current_cuda_stream();
#define NEPQ_MOE_F16_LAUNCH()                                                    \
    nepq_moe_grouped_f16_kernel<F><<<blocks, threads, 0, stream>>>(              \
        indices.data_ptr<uint8_t>(), indices.numel(),                            \
        aux.data_ptr<uint8_t>(), aux.numel(),                                    \
        sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                        \
        neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),           \
        bank_ids.data_ptr<uint8_t>(),                                            \
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),                \
        expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),           \
        expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),      \
        tile_experts.data_ptr<int32_t>(),                                        \
        reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                    \
        routes, static_cast<int>(n_experts), static_cast<int>(pool_experts),     \
        static_cast<int>(out_per_expert), K, ng, nvec, nsign, nsuper,           \
        table_stride, static_cast<int>(sub_bits), max_tiles, x.dim() == 3)
    launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        NEPQ_MOE_F16_LAUNCH();
    });
#undef NEPQ_MOE_F16_LAUNCH
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

::tyloi::DenseTensor nvq_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids,
    ::tyloi::DenseTensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    bool input_quantized,
    ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts) {
    check_common(
        indices, aux, sub_scale, neuron_scale, codebook,
        neuron_len, gs, sub_bits, format, sign_mode,true);
    TYLOINFER_RUNTIME_CHECK(n_experts > 0 && n_experts <= 4096,
                "NVQ global expert count must be in [1,4096]");
    TYLOINFER_RUNTIME_CHECK(
        pool_experts > 0 &&
        pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "NVQ pool storage expert count must fit int32");
    TYLOINFER_RUNTIME_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NVQ pool shape does not match the flattened row count");
    TYLOINFER_RUNTIME_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == ::tyloi::kInt32 &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NVQ expert-local map must be CUDA contiguous int32");
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2,
                "NVQ route IDs must be CUDA contiguous int32 [T,R]");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == ::tyloi::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) &&
                (input_quantized || x.size(-1) == neuron_len),
                "NVQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TYLOINFER_RUNTIME_CHECK(tokens > 0 && routes > 0, "NVQ route dimensions must be nonzero");
    TYLOINFER_RUNTIME_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NVQ input leading dimensions do not match routes");
    TYLOINFER_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == ::tyloi::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NVQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int input_rows = routed_input ? pairs : tokens;
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == ::tyloi::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NVQ routed qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == ::tyloi::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NVQ routed xscale workspace mismatch");
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    cudaStream_t stream = mfq_current_cuda_stream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }

    if(dense_nvq_format(int(format)) && aux.numel()==0) {
        const NvqDeviceWeight w{indices.data_ptr<uint8_t>(),indices.numel(),nullptr,0,nullptr,0,
            neuron_scale.data_ptr<float>(),codebook.data_ptr<int8_t>(),int(codebook.numel()),
            int(pool_experts*out_per_expert),ng,nvec,nsign,int(sub_bits),int(sign_mode)};
        launch_by_format(int(format),[&](auto tag) {
            constexpr int F=decltype(tag)::value;
            if constexpr(dense_nvq_format(F))
                nvq_moe_groups_kernel<F><<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
                    w,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),
                    expert_local.data_ptr<int32_t>(),reinterpret_cast<half*>(out.data_ptr<mfq_half>()),
                    int(n_experts),int(pool_experts),int(out_per_expert),routes,routed_input);
        });
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();return out;
    }

    if (tokens <= 8) {
        const char * rows_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_ROWS_PER_BLOCK");
        int rows_per_block = rows_env == nullptr ? 2 : std::atoi(rows_env);
        if (rows_per_block != 1 && rows_per_block != 2 &&
                rows_per_block != 4 && rows_per_block != 8) {
            rows_per_block = 2;
        }
        const char * warps_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_WARPS");
        int warps_override = warps_env == nullptr ? 0 : std::atoi(warps_env);
        if (warps_override != 1 && warps_override != 2 &&
                warps_override != 4 && warps_override != 8) {
            warps_override = 0;
        }
        const char * exact_reduction_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_EXACT_REDUCTION");
        const bool exact_reduction =
            exact_reduction_env != nullptr &&
            std::atoi(exact_reduction_env) != 0;
        const char * exact_physical_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_EXACT_PHYSICAL_WARPS");
        const int exact_physical_warps =
            exact_physical_env != nullptr && std::atoi(exact_physical_env) == 4
            ? 4
            : 2;
        const char * share_group_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_SHARE_GROUP_STATE");
        const bool share_group_state =
            share_group_env != nullptr && std::atoi(share_group_env) != 0;
#define NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE)              \
        nvq_moe_mmvq_kernel<F, NWARPS_VALUE, ROWS_VALUE, SHARE_VALUE><<<         \
            dim3((static_cast<int>(out_per_expert) + ROWS_VALUE - 1) /          \
                     ROWS_VALUE,                                                 \
                 pairs),                                                         \
            dim3(32, NWARPS_VALUE), 0, stream>>>(                                \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),           \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
#define NVQ_MOE_SELECTED_LAUNCH(NWARPS_VALUE, ROWS_VALUE)                        \
        do {                                                                      \
            if constexpr (F == kNvq2Exec || F == kNvq2JscExec) {                 \
                if (share_group_state) {                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, true);         \
                } else {                                                          \
                    NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);        \
                }                                                                 \
            } else {                                                              \
                NVQ_MOE_SMALL_LAUNCH(NWARPS_VALUE, ROWS_VALUE, false);            \
            }                                                                     \
        } while (0)
#define NVQ_MOE_EXACT_LAUNCH(PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE)          \
        nvq_moe_mmvq_exact_reduction_kernel<                                    \
            F, PHYSICAL_WARPS_VALUE, LOGICAL_WARPS_VALUE, 2><<<                 \
            dim3((static_cast<int>(out_per_expert) + 1) / 2, pairs),            \
            dim3(32, PHYSICAL_WARPS_VALUE), 0, stream>>>(                        \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            ids.data_ptr<int32_t>(), expert_local.data_ptr<int32_t>(),           \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            const char* grouped=tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_GROUP_DOT");
            if(!exact_reduction && ng<=128 && out_per_expert<=4096 && (!grouped || grouped[0]!='0')) {
                const NvqDeviceWeight w{indices.data_ptr<uint8_t>(),indices.numel(),aux.data_ptr<uint8_t>(),aux.numel(),
                    sub_scale.data_ptr<uint8_t>(),sub_scale.numel(),neuron_scale.data_ptr<float>(),codebook.data_ptr<int8_t>(),
                    int(codebook.numel()),int(pool_experts*out_per_expert),ng,nvec,nsign,int(sub_bits),int(sign_mode)};
                tyloi::cuda::NvqRoutedReuseParams reuse;
                reuse.weight=w;reuse.format=F;reuse.expert_local=expert_local.data_ptr<int32_t>();
                reuse.ids=ids.data_ptr<int32_t>();reuse.input=qx.data_ptr<int8_t>();reuse.scales=xscale.data_ptr<float>();
                reuse.output=reinterpret_cast<half*>(out.data_ptr<mfq_half>());
                reuse.experts=int(n_experts);reuse.local_experts=int(pool_experts);reuse.rows=int(out_per_expert);
                reuse.routes=routes;reuse.pairs=pairs;reuse.down=routed_input;
                if(!tyloi::cuda::nvq_launch_routed_reuse(reuse,stream))
                    nvq_moe_groups_kernel<F><<<dim3((out_per_expert+3)/4,pairs),dim3(32,4),0,stream>>>(
                    w,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),
                    reinterpret_cast<half*>(out.data_ptr<mfq_half>()),int(n_experts),int(pool_experts),int(out_per_expert),routes,routed_input);
                return;
            }
            if (exact_reduction) {
                if (K >= 4096) {
                    if (exact_physical_warps == 4) {
                        NVQ_MOE_EXACT_LAUNCH(4, 8);
                    } else {
                        NVQ_MOE_EXACT_LAUNCH(2, 8);
                    }
                } else {
                    NVQ_MOE_EXACT_LAUNCH(2, 4);
                }
                return;
            }
            if (warps_override == 1) {
                NVQ_MOE_SELECTED_LAUNCH(1, 2);
                return;
            }
            if (warps_override == 2) {
                if (rows_per_block == 4) {
                    NVQ_MOE_SELECTED_LAUNCH(2, 4);
                } else {
                    NVQ_MOE_SELECTED_LAUNCH(2, 2);
                }
                return;
            }
            if (warps_override == 4) {
                NVQ_MOE_SELECTED_LAUNCH(4, 2);
                return;
            }
            if (warps_override == 8) {
                NVQ_MOE_SELECTED_LAUNCH(8, 2);
                return;
            }
            if (K >= 4096) {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(8, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(8, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(8, 8);
                else NVQ_MOE_SELECTED_LAUNCH(8, 2);
            } else {
                if (rows_per_block == 1) NVQ_MOE_SELECTED_LAUNCH(4, 1);
                else if (rows_per_block == 4) NVQ_MOE_SELECTED_LAUNCH(4, 4);
                else if (rows_per_block == 8) NVQ_MOE_SELECTED_LAUNCH(4, 8);
                else NVQ_MOE_SELECTED_LAUNCH(4, 2);
            }
        });
#undef NVQ_MOE_EXACT_LAUNCH
#undef NVQ_MOE_SELECTED_LAUNCH
#undef NVQ_MOE_SMALL_LAUNCH
    } else {
        TYLOINFER_RUNTIME_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == ::tyloi::kInt32 &&
                    ids_dst.numel() >= pairs,
                    "NVQ compact route map is missing");
        TYLOINFER_RUNTIME_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == ::tyloi::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NVQ expert bounds are missing");
        TYLOINFER_RUNTIME_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == ::tyloi::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NVQ tile bounds are missing");
        TYLOINFER_RUNTIME_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == ::tyloi::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NVQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NVQ_MOE_GROUPED_LAUNCH()                                                 \
        nvq_moe_grouped_tile_kernel<F, route_tile><<<                            \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), codebook.data_ptr<int8_t>(),         \
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),                     \
            expert_local.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(),       \
            expert_bounds.data_ptr<int32_t>(), tile_bounds.data_ptr<int32_t>(),  \
            tile_experts.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, static_cast<int>(sub_bits),                         \
            static_cast<int>(sign_mode), max_tiles, row_tiles, routed_input)
        launch_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NVQ_MOE_GROUPED_LAUNCH();
        });
#undef NVQ_MOE_GROUPED_LAUNCH
    }
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NVQ routed pool matmul kernel launch failed");
    return out;
}

::tyloi::DenseTensor nepq_moe_grouped_matmul_pool_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor table_pool,
    ::tyloi::DenseTensor bank_ids,
    ::tyloi::DenseTensor grouped_table_pool,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor ids,
    ::tyloi::DenseTensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    bool input_quantized,
    ::tyloi::DenseTensor out,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor ids_dst,
    ::tyloi::DenseTensor expert_bounds,
    ::tyloi::DenseTensor tile_bounds,
    ::tyloi::DenseTensor tile_experts) {
    check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = format == kNpq0S
        ? kNpq0STableBytes
        : static_cast<int>(table_pool.size(1));
    TYLOINFER_RUNTIME_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == ::tyloi::kInt8 &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    TYLOINFER_RUNTIME_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ global expert count must be in [1,4096]");
    TYLOINFER_RUNTIME_CHECK(pool_experts > 0 &&
                pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NEPQ pool shape does not match the flattened row count");
    TYLOINFER_RUNTIME_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == ::tyloi::kInt32 &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NEPQ expert-local map must be CUDA contiguous int32");
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == ::tyloi::kInt32 && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == ::tyloi::kFloat16 &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TYLOINFER_RUNTIME_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    TYLOINFER_RUNTIME_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    TYLOINFER_RUNTIME_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == ::tyloi::kFloat16 && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int K_pad = ng * kGroupSize;
    const int input_rows = routed_input ? pairs : tokens;
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == ::tyloi::kInt8 && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == ::tyloi::kFloat32 && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    const int grouped_table_stride =
        static_cast<int>(grouped_table_pool.size(1));
    cudaStream_t stream = mfq_current_cuda_stream();
    if (!input_quantized) {
        nvq_quantize_x_gs24_kernel<<<dim3(input_rows, ng), 32, 0, stream>>>(
            reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()),
            qx.data_ptr<int8_t>(), xscale.data_ptr<float>(), input_rows, K, ng);
    }

    if (tokens <= 8) {
        constexpr int rows_per_block = 2;
        const int row_blocks =
            (static_cast<int>(out_per_expert) + rows_per_block - 1) /
            rows_per_block;
#define NEPQ_MOE_POOL_SMALL_LAUNCH(NWARPS_VALUE)                                 \
        nepq_moe_mmvq_kernel<F, NWARPS_VALUE, rows_per_block><<<                 \
            dim3(row_blocks, pairs), dim3(32, NWARPS_VALUE), 0, stream>>>(       \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), table_pool.data_ptr<int8_t>(),       \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), ids.data_ptr<int32_t>(),                   \
            expert_local.data_ptr<int32_t>(),                                    \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            pairs, routes, static_cast<int>(n_experts),                          \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, nsuper, table_stride,                               \
            static_cast<int>(sub_bits), routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            if (K >= 4096) NEPQ_MOE_POOL_SMALL_LAUNCH(8);
            else NEPQ_MOE_POOL_SMALL_LAUNCH(4);
        });
#undef NEPQ_MOE_POOL_SMALL_LAUNCH
    } else {
        TYLOINFER_RUNTIME_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == ::tyloi::kInt32 &&
                    ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        TYLOINFER_RUNTIME_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == ::tyloi::kInt32 &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        TYLOINFER_RUNTIME_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == ::tyloi::kInt32 &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        TYLOINFER_RUNTIME_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == ::tyloi::kInt32 &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        constexpr int route_tile = 8;
        const int row_tiles = (static_cast<int>(out_per_expert) + 3) / 4;
        const int max_tiles = (pairs + route_tile - 1) / route_tile +
            static_cast<int>(n_experts);
        const int64_t max_tasks = static_cast<int64_t>(max_tiles) * row_tiles;
        const int blocks = static_cast<int>(
            std::min<int64_t>(max_tasks, 4096));
#define NEPQ_MOE_POOL_GROUPED_LAUNCH()                                           \
        nepq_moe_grouped_tile_kernel<F, route_tile><<<                           \
            blocks, dim3(32, 4), 0, stream>>>(                                   \
            indices.data_ptr<uint8_t>(), indices.numel(),                        \
            aux.data_ptr<uint8_t>(), aux.numel(),                                \
            sub_scale.data_ptr<uint8_t>(), sub_scale.numel(),                    \
            neuron_scale.data_ptr<float>(), grouped_table_pool.data_ptr<int8_t>(), \
            bank_ids.data_ptr<uint8_t>(), qx.data_ptr<int8_t>(),                 \
            xscale.data_ptr<float>(), expert_local.data_ptr<int32_t>(),          \
            ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),      \
            tile_bounds.data_ptr<int32_t>(), tile_experts.data_ptr<int32_t>(),   \
            reinterpret_cast<__half *>(out.data_ptr<mfq_half>()),                \
            routes, static_cast<int>(n_experts),                                 \
            static_cast<int>(pool_experts), static_cast<int>(out_per_expert),    \
            ng, nvec, nsign, nsuper, grouped_table_stride,                       \
            static_cast<int>(sub_bits), max_tiles, row_tiles, routed_input)
        launch_nepq_by_format(static_cast<int>(format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            NEPQ_MOE_POOL_GROUPED_LAUNCH();
        });
#undef NEPQ_MOE_POOL_GROUPED_LAUNCH
    }
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NEPQ routed pool matmul kernel launch failed");
    return out;
}










::tyloi::DenseTensor nvq_gemv_multi2_ws_cuda(
    ::tyloi::DenseTensor first_indices,
    ::tyloi::DenseTensor first_aux,
    ::tyloi::DenseTensor first_sub_scale,
    ::tyloi::DenseTensor first_neuron_scale,
    ::tyloi::DenseTensor first_codebook,
    ::tyloi::DenseTensor second_indices,
    ::tyloi::DenseTensor second_aux,
    ::tyloi::DenseTensor second_sub_scale,
    ::tyloi::DenseTensor second_neuron_scale,
    ::tyloi::DenseTensor second_codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t first_sub_bits,
    int64_t first_format,
    int64_t first_sign_mode,
    int64_t second_sub_bits,
    int64_t second_format,
    int64_t second_sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    check_common(first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
                 neuron_len, gs, first_sub_bits, first_format, first_sign_mode);
    check_common(second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
                 neuron_len, gs, second_sub_bits, second_format, second_sign_mode);
    TYLOINFER_RUNTIME_CHECK(first_format == second_format,
                "NVQ multi-projection requires the same NVQ format");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ multi-projection x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(M >= 1 && M <= 8, "NVQ multi-projection supports M in [1,8]");
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ multi-projection x width mismatch");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NVQ multi-projection qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NVQ multi-projection xscale workspace mismatch");
    const auto first = make_device_weight(
        first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
        K, static_cast<int>(gs), static_cast<int>(first_sub_bits),
        static_cast<int>(first_format), static_cast<int>(first_sign_mode));
    const auto second = make_device_weight(
        second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
        K, static_cast<int>(gs), static_cast<int>(second_sub_bits),
        static_cast<int>(second_format), static_cast<int>(second_sign_mode));
    auto output = ::tyloi::cuda::empty(
        {M, first.N + second.N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(M, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), M, K, ng);
    launch_selected_multi2_vec8(
        static_cast<int>(first_format), first, second, qx, xscale, output, M, stream);
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ multi-projection kernel launch failed");
    return output;
}

::tyloi::DenseTensor nvq_gemv_swiglu_ws_cuda(
    ::tyloi::DenseTensor gate_indices,
    ::tyloi::DenseTensor gate_aux,
    ::tyloi::DenseTensor gate_sub_scale,
    ::tyloi::DenseTensor gate_neuron_scale,
    ::tyloi::DenseTensor gate_codebook,
    ::tyloi::DenseTensor up_indices,
    ::tyloi::DenseTensor up_aux,
    ::tyloi::DenseTensor up_sub_scale,
    ::tyloi::DenseTensor up_neuron_scale,
    ::tyloi::DenseTensor up_codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TYLOINFER_RUNTIME_CHECK(gate_format == up_format, "NVQ SwiGLU requires matching NVQ formats");
    TYLOINFER_RUNTIME_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ SwiGLU gate/up output widths must match");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1,
                "NVQ SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ SwiGLU x width mismatch");
    TYLOINFER_RUNTIME_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ SwiGLU workspace mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), static_cast<int>(gate_sub_bits),
        static_cast<int>(gate_format), static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), static_cast<int>(up_sub_bits),
        static_cast<int>(up_format), static_cast<int>(up_sign_mode));
    auto output = ::tyloi::cuda::empty({1, gate.N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq2JscXLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq3JscLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
                reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
        });
    }
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ SwiGLU kernel launch failed");
    return output;
}

::tyloi::DenseTensor nvq2_gemv_swiglu_vec4_ordered_ws_cuda(
    ::tyloi::DenseTensor gate_indices,
    ::tyloi::DenseTensor gate_aux,
    ::tyloi::DenseTensor gate_sub_scale,
    ::tyloi::DenseTensor gate_neuron_scale,
    ::tyloi::DenseTensor gate_codebook,
    ::tyloi::DenseTensor up_indices,
    ::tyloi::DenseTensor up_aux,
    ::tyloi::DenseTensor up_sub_scale,
    ::tyloi::DenseTensor up_neuron_scale,
    ::tyloi::DenseTensor up_codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TYLOINFER_RUNTIME_CHECK(gate_format == kNvq2 && up_format == kNvq2 &&
                gate_sub_bits == 4 && up_sub_bits == 4,
                "NVQ2 ordered vec4 SwiGLU requires NVQ2 sub_bits=4 weights");
    TYLOINFER_RUNTIME_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ2 ordered vec4 SwiGLU gate/up output widths must match");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ2 ordered vec4 SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int nsign = (K + 7) / 8;
    TYLOINFER_RUNTIME_CHECK(nsign % 4 == 0,
                "NVQ2 ordered vec4 SwiGLU requires a multiple of four 8-value vectors");
    TYLOINFER_RUNTIME_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ2 ordered vec4 SwiGLU workspace mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), 4, kNvq2, static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), 4, kNvq2, static_cast<int>(up_sign_mode));
    auto output = ::tyloi::cuda::empty({1, gate.N}, x.options());
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), qx.data_ptr<int8_t>(),
        xscale.data_ptr<float>(), 1, K, ng);
    const size_t shared_bytes = static_cast<size_t>(nsign) * sizeof(uint32_t);
    nvq2_gemv_swiglu_vec4_ordered_kernel<4>
        <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
            gate, up, qx.data_ptr<int8_t>(), xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(output.data_ptr<mfq_half>()), nullptr);
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess,
                "NVQ2 ordered vec4 SwiGLU kernel launch failed");
    return output;
}

void nvq_ffn_swiglu_quant_ws_cuda(
    ::tyloi::DenseTensor gate_indices,
    ::tyloi::DenseTensor gate_aux,
    ::tyloi::DenseTensor gate_sub_scale,
    ::tyloi::DenseTensor gate_neuron_scale,
    ::tyloi::DenseTensor gate_codebook,
    ::tyloi::DenseTensor up_indices,
    ::tyloi::DenseTensor up_aux,
    ::tyloi::DenseTensor up_sub_scale,
    ::tyloi::DenseTensor up_neuron_scale,
    ::tyloi::DenseTensor up_codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    int64_t down_gs,
    ::tyloi::DenseTensor input_qx,
    ::tyloi::DenseTensor input_xscale,
    ::tyloi::DenseTensor output_qx,
    ::tyloi::DenseTensor output_xscale,
    ::tyloi::DenseTensor swiglu_scratch) {
    check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TYLOINFER_RUNTIME_CHECK(gate_format == up_format, "NVQ fused FFN requires matching gate/up formats");
    TYLOINFER_RUNTIME_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ fused FFN gate/up widths must match");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ fused FFN x must be CUDA contiguous fp16 [1,K]");
    TYLOINFER_RUNTIME_CHECK(down_gs == 24 || down_gs == 28 || down_gs == 32,
                "NVQ fused FFN down gs must be 24, 28, or 32");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int N = static_cast<int>(gate_neuron_scale.numel());
    const int down_ng = (N + static_cast<int>(down_gs) - 1) / static_cast<int>(down_gs);
    const int down_k_pad = down_ng * static_cast<int>(down_gs);
    TYLOINFER_RUNTIME_CHECK(input_qx.is_cuda() && input_qx.scalar_type() == ::tyloi::kInt8 &&
                input_qx.is_contiguous() && input_qx.numel() >= K_pad,
                "NVQ fused FFN input qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(input_xscale.is_cuda() && input_xscale.scalar_type() == ::tyloi::kFloat32 &&
                input_xscale.is_contiguous() && input_xscale.numel() >= ng,
                "NVQ fused FFN input xscale workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(output_qx.is_cuda() && output_qx.scalar_type() == ::tyloi::kInt8 &&
                output_qx.is_contiguous() && output_qx.numel() >= down_k_pad,
                "NVQ fused FFN output qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(output_xscale.is_cuda() && output_xscale.scalar_type() == ::tyloi::kFloat32 &&
                output_xscale.is_contiguous() && output_xscale.numel() >= down_ng,
                "NVQ fused FFN output xscale workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(swiglu_scratch.is_cuda() && swiglu_scratch.scalar_type() == ::tyloi::kFloat32 &&
                swiglu_scratch.is_contiguous() && swiglu_scratch.numel() >= N,
                "NVQ fused FFN SwiGLU scratch mismatch");
    const auto gate = make_device_weight(
        gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
        K, static_cast<int>(gs), static_cast<int>(gate_sub_bits),
        static_cast<int>(gate_format), static_cast<int>(gate_sign_mode));
    const auto up = make_device_weight(
        up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
        K, static_cast<int>(gs), static_cast<int>(up_sub_bits),
        static_cast<int>(up_format), static_cast<int>(up_sign_mode));
    cudaStream_t stream = mfq_current_cuda_stream();
    nvq_quantize_x_gs24_kernel<<<dim3(1, ng), 32, 0, stream>>>(
        reinterpret_cast<const __half *>(x.data_ptr<mfq_half>()), input_qx.data_ptr<int8_t>(),
        input_xscale.data_ptr<float>(), 1, K, ng);
    bool aligned_independent_rows = false;
    const char * vec4_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_SWIGLU_VEC4");
    if (vec4_env == nullptr) vec4_env = tyloi::cuda::env_lookup("TYLOINFER_NIQ_SWIGLU_VEC4");
    const bool use_vec4 = vec4_env == nullptr || vec4_env[0] != '0';
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
            gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(swiglu_scratch.data_ptr<float>()), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
            gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
            reinterpret_cast<__half *>(swiglu_scratch.data_ptr<float>()), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (use_vec4 && gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4 &&
        gate.nsign % 4 == 0 && up.nsign % 4 == 0) {
        const size_t shared_bytes = static_cast<size_t>(gate.nsign) * sizeof(uint32_t);
        nvq2_gemv_swiglu_vec4_ordered_kernel<4, true>
            <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4, true><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, input_qx.data_ptr<int8_t>(), input_xscale.data_ptr<float>(),
                nullptr, swiglu_scratch.data_ptr<float>());
        });
    }
    switch (static_cast<int>(down_gs)) {
        case 24:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<24><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<24><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
        case 28:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<28><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<28><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
        case 32:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<32><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(swiglu_scratch.data_ptr<float>()),
                    output_qx.data_ptr<int8_t>(), output_xscale.data_ptr<float>(),
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<32><<<down_ng, 32, 0, stream>>>(
                    swiglu_scratch.data_ptr<float>(), output_qx.data_ptr<int8_t>(),
                    output_xscale.data_ptr<float>(), N, down_k_pad);
            }
            break;
    }
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError() == cudaSuccess, "NVQ fused FFN kernel launch failed");
}





void nvq_check_native(const ::tyloi::DenseTensor& indices, const ::tyloi::DenseTensor& aux,
    const ::tyloi::DenseTensor& state, const ::tyloi::DenseTensor& scale,
    const ::tyloi::DenseTensor& table, int64_t width, int64_t gs, int64_t bits,
    int64_t format, int64_t sign) {
    check_common(indices, aux, state, scale, table, width, gs, bits, format, sign);
}
void nepq_check_native(const ::tyloi::DenseTensor& indices, const ::tyloi::DenseTensor& aux,
    const ::tyloi::DenseTensor& state, const ::tyloi::DenseTensor& scale,
    const ::tyloi::DenseTensor& table, const ::tyloi::DenseTensor& banks,
    int64_t width, int64_t bits, int64_t format) {
    check_nepq_common(indices, aux, state, scale, table, banks, width, bits, format);
}
