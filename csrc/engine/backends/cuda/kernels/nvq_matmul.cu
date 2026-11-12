#include "nvq_raw.h"
#include "env.h"
#include <stdexcept>
#include <cstdlib>
#include <climits>
#include <map>
#include "packed_nvq.cuh"
#include "nvq_dense_group.cuh"
#include "nvq_routed_reuse.cuh"
#include "nvq_format.cuh"
#include "async_copy.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <cstdint>
using namespace nvcuda;
namespace {
using namespace tyloi::cuda::packed_nvq;
#include "nvq_projection_device.cuh"
#include "nepq_routed_device.cuh"
#include "nvq_routed_device.cuh"
template<int Warps>
int nvq_active_resident_blocks() {
    int device=0;
    if(cudaGetDevice(&device)!=cudaSuccess) throw std::runtime_error("NVQ active device query failed");
    static thread_local std::map<int,int> cached;
    if(const auto found=cached.find(device);found!=cached.end())return found->second;
    cudaDeviceProp properties{};int active=0;
    if(cudaGetDeviceProperties(&properties,device)!=cudaSuccess) throw std::runtime_error("NVQ active device properties unavailable");
    if(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        nvq_moe_mmvq_hetero_kernel<Warps,2,false,true>,Warps*32,0)!=cudaSuccess || active<=0)
        throw std::runtime_error("NVQ active kernel occupancy unavailable");
    const int blocks=active*properties.multiProcessorCount;
    cached.emplace(device,blocks);return blocks;
}

#include "nvq_routed_f16_device.cuh"
#include "nvq_projection_launch.cuh"
#include "nvq_multi_launch.cuh"
using tyloi::cuda::nvq_raw::Projection;
using tyloi::cuda::nvq_format::launch_by_format;
#include "nvq_vec_launch.cuh"
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


}
namespace tyloi::cuda::nvq_raw {
using namespace tyloi::cuda::packed_nvq;
using tyloi::cuda::nvq_format::launch_by_format;
using tyloi::cuda::nvq_format::launch_nepq_by_format;
void gemm(const Weight& w, const void* input, void* output, int rows, cudaStream_t stream) {
    const int M=rows,N=w.rows,K=w.width,ng=(K+w.group_size-1)/w.group_size;
    const int nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(K+7)/8;
    const dim3 block(32,8);
#define NVQ_F16_GEMM_LAUNCH(MTILES_VALUE, NEED_CHECK_VALUE) \
    nvq_gemm_f16_gs24_kernel<F, MTILES_VALUE, NEED_CHECK_VALUE><<< \
        dim3((N+63)/64,(M+16*MTILES_VALUE-1)/(16*MTILES_VALUE)),block,0,stream>>>( \
        w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table, \
        static_cast<const __half*>(input),static_cast<__half*>(output), \
        M,N,K,ng,nvec,nsign,w.state_bits,w.sign_mode)
    launch_by_format(w.format, [&](auto tag) {
        constexpr int F=decltype(tag)::value;
        if (M<=16) { if(M==16 && N%64==0) NVQ_F16_GEMM_LAUNCH(1,false); else NVQ_F16_GEMM_LAUNCH(1,true); }
        else if (M<=32) { if(M==32 && N%64==0) NVQ_F16_GEMM_LAUNCH(2,false); else NVQ_F16_GEMM_LAUNCH(2,true); }
        else if (M<=48) { if(M==48 && N%64==0) NVQ_F16_GEMM_LAUNCH(3,false); else NVQ_F16_GEMM_LAUNCH(3,true); }
        else if (M<=64) { if(M==64 && N%64==0) NVQ_F16_GEMM_LAUNCH(4,false); else NVQ_F16_GEMM_LAUNCH(4,true); }
        else if (M<=80) { if(M==80 && N%64==0) NVQ_F16_GEMM_LAUNCH(5,false); else NVQ_F16_GEMM_LAUNCH(5,true); }
        else if (M<=96) { if(M==96 && N%64==0) NVQ_F16_GEMM_LAUNCH(6,false); else NVQ_F16_GEMM_LAUNCH(6,true); }
        else if (M<=112) { if(M==112 && N%64==0) NVQ_F16_GEMM_LAUNCH(7,false); else NVQ_F16_GEMM_LAUNCH(7,true); }
        else { if(M%128==0 && N%64==0) NVQ_F16_GEMM_LAUNCH(8,false); else NVQ_F16_GEMM_LAUNCH(8,true); }
    });
#undef NVQ_F16_GEMM_LAUNCH
}

static void quantize_input(const Projection& a, bool gated, cudaStream_t stream) {
    const int M=a.rows,K=a.weight.width,ng=(K+a.weight.group_size-1)/a.weight.group_size;
    if(gated && a.gate_mode==1) nvq_quantize_x_gate_gs24_kernel<1><<<dim3(M,ng),32,0,stream>>>(
        static_cast<const __half*>(a.input),static_cast<const __half*>(a.gate),a.quantized,a.scales,M,K,ng);
    else if(gated) nvq_quantize_x_gate_gs24_kernel<2><<<dim3(M,ng),32,0,stream>>>(
        static_cast<const __half*>(a.input),static_cast<const __half*>(a.gate),a.quantized,a.scales,M,K,ng);
    else nvq_quantize_x_gs24_kernel<<<dim3(M,ng),32,0,stream>>>(
        static_cast<const __half*>(a.input),a.quantized,a.scales,M,K,ng);
}
void gemv(const Projection& a, bool quantize, bool gate, int method, cudaStream_t stream) {
    const auto& w=a.weight;
    const int M=a.rows,N=w.rows,K=w.width,ng=(K+w.group_size-1)/w.group_size;
    const int nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(K+7)/8;
    if(quantize) quantize_input(a,gate,stream);
    if(method==1) {
        switch(a.warps) {
        case 1:launch_m1_vec8_by_format<1>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        case 2:launch_m1_vec8_by_format<2>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        case 4:launch_m1_vec8_by_format<4>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        case 8:launch_m1_vec8_by_format<8>(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);break;
        }
    } else if(method==2) {
#define NVQ_RAW_BATCH(NW) \
    if(M<=2) launch_batch_vec8_by_format<NW,2>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else if(M<=4) launch_batch_vec8_by_format<NW,4>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else if(M<=8) launch_batch_vec8_by_format<NW,8>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else if(M<=12) launch_batch_vec8_by_format<NW,12>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream); \
    else launch_batch_vec8_by_format<NW,16>(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream)
        switch(a.warps) { case 2: NVQ_RAW_BATCH(2); break; case 4: NVQ_RAW_BATCH(4); break; case 8: NVQ_RAW_BATCH(8); break; }
#undef NVQ_RAW_BATCH
    } else if(M==1) launch_selected_m1_vec8(w.format,a,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);
    else launch_selected_batch_vec8(w.format,a,M,N,ng,nvec,nsign,w.state_bits,w.sign_mode,stream);
}
bool group_exec(int format) { return format==kNvq2JscXLGroupExec || format==kNvq3JscLGroupExec; }
void gemv_residual(const Projection& a, cudaStream_t stream) {
    const auto& w=a.weight;
    const int ng=(w.width+w.group_size-1)/w.group_size;
    const int nvec=(w.width+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(w.width+7)/8;
    const NvqDeviceWeight weight{w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,
        w.scale,w.table,int(w.table_bytes),w.rows,ng,nvec,nsign,w.state_bits,w.sign_mode};
    if(w.format==kNvq2JscXLGroupExec) launch_aligned_group_m1<kNvq2JscXLGroupExec,true>(
        weight,a.quantized,a.scales,static_cast<__half*>(a.output),stream,static_cast<const __half*>(a.residual));
    else launch_aligned_group_m1<kNvq3JscLGroupExec,true>(
        weight,a.quantized,a.scales,static_cast<__half*>(a.output),stream,static_cast<const __half*>(a.residual));
}
void mmq(const Projection& a, bool gated, cudaStream_t stream) {
    const auto& w=a.weight;
    const int M=a.rows,N=w.rows,K=w.width,ng=(K+w.group_size-1)/w.group_size;
    const int nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(K+7)/8;
    quantize_input(a,gated,stream);
#define NVQ_RAW_MMQ(TILES,CHECK) \
    nvq_mmq_mma24_kernel<F,TILES,CHECK><<<dim3((N+63)/64,1),dim3(32,8),0,stream>>>( \
        w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table, \
        a.quantized,a.scales,static_cast<__half*>(a.output),M,N,ng,nvec,nsign,w.state_bits,w.sign_mode)
    launch_by_format(w.format,[&](auto tag) {
        constexpr int F=decltype(tag)::value;
        if(M<=16) { if(M==16 && N%64==0) NVQ_RAW_MMQ(1,false); else NVQ_RAW_MMQ(1,true); }
        else if(M<=32) { if(M==32 && N%64==0) NVQ_RAW_MMQ(2,false); else NVQ_RAW_MMQ(2,true); }
        else if(M<=48) { if(M==48 && N%64==0) NVQ_RAW_MMQ(3,false); else NVQ_RAW_MMQ(3,true); }
        else { if(M==64 && N%64==0) NVQ_RAW_MMQ(4,false); else NVQ_RAW_MMQ(4,true); }
    });
#undef NVQ_RAW_MMQ
}
void nepq_gemv(const Projection& a, cudaStream_t stream) {
    const auto& w=a.weight; const int M=a.rows,N=w.rows,K=w.width,ng=(K+23)/24,nvec=K/8,nsign=(K+7)/8;
    const int nsuper=(ng+kNepqGroupsPerSupergroup-1)/kNepqGroupsPerSupergroup;
    quantize_input(a,false,stream);
#define NEPQ_RAW_GEMV(W,R) nepq_gemv_batch_vec8_kernel<F,W,R><<<N,dim3(32,W),0,stream>>>( \
    w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,w.banks, \
    a.quantized,a.scales,static_cast<__half*>(a.output),M,N,ng,nvec,nsign,nsuper,w.table_stride,w.state_bits,0)
#define NEPQ_RAW_WARPS(R) if(nwarps==8) NEPQ_RAW_GEMV(8,R); else if(nwarps==2) NEPQ_RAW_GEMV(2,R); else NEPQ_RAW_GEMV(4,R)
    launch_nepq_by_format(w.format,[&](auto tag) {
        constexpr int F=decltype(tag)::value;
        const int nwarps=(w.format==kNvq1S && N<=2048)?8:((w.format==kNpq0S && N<=1024)?2:4);
        if(M==1) { NEPQ_RAW_WARPS(1); } else if(M<=2) { NEPQ_RAW_WARPS(2); }
        else if(M<=4) { NEPQ_RAW_WARPS(4); } else if(M<=8) { NEPQ_RAW_WARPS(8); }
        else { NEPQ_RAW_WARPS(16); }
    });
#undef NEPQ_RAW_WARPS
#undef NEPQ_RAW_GEMV
}
void nepq_mmq(const Projection& a, cudaStream_t stream) {
    const auto& w=a.weight; const int M=a.rows,N=w.rows,K=w.width,ng=(K+23)/24,nvec=K/8,nsign=(K+7)/8;
    const int nsuper=(ng+kNepqGroupsPerSupergroup-1)/kNepqGroupsPerSupergroup;
    quantize_input(a,false,stream);
#define NEPQ_RAW_MMQ(T,C) nepq_mmq_mma24_kernel<F,T,C><<<dim3((N+63)/64,1),dim3(32,8),0,stream>>>( \
    w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,w.banks, \
    a.quantized,a.scales,static_cast<__half*>(a.output),M,N,ng,nvec,nsign,nsuper,w.table_stride,w.state_bits,0)
    launch_nepq_by_format(w.format,[&](auto tag) { constexpr int F=decltype(tag)::value;
        if(M<=16) { if(M==16 && N%64==0) NEPQ_RAW_MMQ(1,false); else NEPQ_RAW_MMQ(1,true); }
        else if(M<=32) { if(M==32 && N%64==0) NEPQ_RAW_MMQ(2,false); else NEPQ_RAW_MMQ(2,true); }
        else if(M<=48) { if(M==48 && N%64==0) NEPQ_RAW_MMQ(3,false); else NEPQ_RAW_MMQ(3,true); }
        else { if(M==64 && N%64==0) NEPQ_RAW_MMQ(4,false); else NEPQ_RAW_MMQ(4,true); }
    });
#undef NEPQ_RAW_MMQ
}
void nepq_gemm(const Projection& a, cudaStream_t stream) {
    const auto& w=a.weight; const int M=a.rows,N=w.rows,K=w.width,ng=(K+23)/24,nvec=K/8,nsign=(K+7)/8;
    const int nsuper=(ng+kNepqGroupsPerSupergroup-1)/kNepqGroupsPerSupergroup;
#define NEPQ_RAW_GEMM(T,C) nepq_gemm_f16_gs24_kernel<F,T,C><<<dim3((N+63)/64,(M+16*T-1)/(16*T)),dim3(32,8),0,stream>>>( \
    w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,w.banks, \
    static_cast<const __half*>(a.input),static_cast<__half*>(a.output),M,N,K,ng,nvec,nsign,nsuper,w.table_stride,w.state_bits,0)
    launch_nepq_by_format(w.format,[&](auto tag) { constexpr int F=decltype(tag)::value;
        if(M<=16) { if(M==16 && N%64==0 && ng%4==0) NEPQ_RAW_GEMM(1,false); else NEPQ_RAW_GEMM(1,true); }
        else if(M<=32) { if(M==32 && N%64==0 && ng%4==0) NEPQ_RAW_GEMM(2,false); else NEPQ_RAW_GEMM(2,true); }
        else if(M<=64) { if(M==64 && N%64==0 && ng%4==0) NEPQ_RAW_GEMM(4,false); else NEPQ_RAW_GEMM(4,true); }
        else { if(M%128==0 && N%64==0 && ng%4==0) NEPQ_RAW_GEMM(8,false); else NEPQ_RAW_GEMM(8,true); }
    });
#undef NEPQ_RAW_GEMM
}void quantize(const Projection& a, cudaStream_t stream) { quantize_input(a,false,stream); }
int nepq_grouped_stride(int format, int stride) { return format==kNpq0S ? kNpq0STableBytes : stride; }
void nepq_grouped(const Grouped& a, cudaStream_t stream) {
    const auto& p=a.projection;const auto& w=p.weight;
    const int K=w.width,ng=(K+23)/24,nvec=K/8,nsign=(K+7)/8;
    const int nsuper=(ng+kNepqGroupsPerSupergroup-1)/kNepqGroupsPerSupergroup;
    const int pairs=a.tokens*a.routes;
    if(a.tokens<=8) {
        const int row_blocks=(a.output_rows+1)/2;
#define NEPQ_GROUP_SMALL(W) nepq_moe_mmvq_kernel<F,W,2><<<dim3(row_blocks,pairs),dim3(32,W),0,stream>>>( \
    w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,w.banks, \
    p.quantized,p.scales,a.ids,a.expert_local,static_cast<__half*>(p.output),pairs,a.routes,a.experts,a.pool_experts, \
    a.output_rows,ng,nvec,nsign,nsuper,w.table_stride,w.state_bits,a.routed_input)
        launch_nepq_by_format(w.format,[&](auto tag){constexpr int F=decltype(tag)::value;
            if(K>=4096) NEPQ_GROUP_SMALL(8);else NEPQ_GROUP_SMALL(4);});
#undef NEPQ_GROUP_SMALL
    } else {
        const int row_tiles=(a.output_rows+3)/4,max_tiles=(pairs+7)/8+a.experts;
        const int blocks=static_cast<int>(std::min<int64_t>(static_cast<int64_t>(max_tiles)*row_tiles,4096));
        launch_nepq_by_format(w.format,[&](auto tag){constexpr int F=decltype(tag)::value;
            nepq_moe_grouped_tile_kernel<F,8><<<blocks,dim3(32,4),0,stream>>>(
                w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,a.grouped_table,w.banks,
                p.quantized,p.scales,a.expert_local,a.ids_dst,a.expert_bounds,a.tile_bounds,a.tile_experts,
                static_cast<__half*>(p.output),a.routes,a.experts,a.pool_experts,a.output_rows,ng,nvec,nsign,nsuper,
                a.grouped_stride,w.state_bits,max_tiles,row_tiles,a.routed_input);
        });
    }
}void nepq_grouped_f16(const Grouped& a, cudaStream_t stream) {
    const auto& p=a.projection;const auto& w=p.weight;
    const int K=w.width,ng=(K+23)/24,nvec=K/8,nsign=(K+7)/8;
    const int nsuper=(ng+kNepqGroupsPerSupergroup-1)/kNepqGroupsPerSupergroup;
    const int pairs=a.tokens*a.routes,max_tiles=(pairs+7)/8+a.experts;
    const int ntiles_n=(a.output_rows+63)/64;
    const int blocks=static_cast<int>(std::min<int64_t>(static_cast<int64_t>(max_tiles)*ntiles_n,4096));
    launch_nepq_by_format(w.format,[&](auto tag){constexpr int F=decltype(tag)::value;
        nepq_moe_grouped_f16_kernel<F><<<blocks,dim3(32,8),0,stream>>>(
            w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,w.banks,
            static_cast<const __half*>(p.input),a.expert_local,a.ids_dst,a.expert_bounds,a.tile_bounds,a.tile_experts,
            static_cast<__half*>(p.output),a.routes,a.experts,a.pool_experts,a.output_rows,K,ng,nvec,nsign,nsuper,
            w.table_stride,w.state_bits,max_tiles,a.routed_input);
    });
}void nvq_grouped_f16(const Grouped& a, cudaStream_t stream) {
    const auto& p=a.projection;const auto& w=p.weight;
    const int K=w.width,ng=(K+23)/24,nvec=(K+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8),nsign=(K+7)/8;
    const int pairs=a.tokens*a.routes,max_tiles=(pairs+7)/8+a.experts;
    const int ntiles_n=(a.output_rows+63)/64;
    const int blocks=static_cast<int>(std::min<int64_t>(static_cast<int64_t>(max_tiles)*ntiles_n,4096));
    launch_by_format(w.format,[&](auto tag){constexpr int F=decltype(tag)::value;
        nvq_moe_grouped_f16_kernel<F><<<blocks,dim3(32,8),0,stream>>>(
            w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,
            static_cast<const __half*>(p.input),a.expert_local,a.ids_dst,a.expert_bounds,a.tile_bounds,a.tile_experts,
            static_cast<__half*>(p.output),a.routes,a.experts,a.pool_experts,a.output_rows,K,ng,nvec,nsign,
            w.state_bits,w.sign_mode,max_tiles,a.routed_input);
    });
}
bool dense_format(int format) { return dense_nvq_format(format); }
#include "nvq_pool_launch.cuh"
bool hetero_format(std::int64_t group) { return group>=kNvqMoeF16AllFormats && group<=kNvqMoeF16LegacyFormats; }
#include "nvq_hetero_f16_launch.cuh"
#include "nvq_hetero_quant_launch.cuh"
static NvqDeviceWeight device_weight(const Weight& w) {
    const int ng=(w.width+w.group_size-1)/w.group_size;
    return {w.indices,w.indices_bytes,w.aux,w.aux_bytes,w.state,w.state_bytes,w.scale,w.table,int(w.table_bytes),
        w.rows,ng,(w.width+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8),
        (w.width+7)/8,w.state_bits,w.sign_mode};
}
void multi2(const Pair& a, cudaStream_t stream) {
    const auto& p=a.first;
    const auto first=device_weight(p.weight),second=device_weight(a.second);
    quantize_input(p,false,stream);
    launch_selected_multi2_vec8(p.weight.format,first,second,p.quantized,p.scales,
        static_cast<__half*>(p.output),p.rows,stream);
}
bool nvq2_format(std::int64_t format) { return format==kNvq2; }
void swiglu(const Pair& a, bool ordered, cudaStream_t stream) {
    const auto& p=a.first; const auto gate=device_weight(p.weight),up=device_weight(a.second);
    const int gate_format=p.weight.format,gate_sub_bits=p.weight.state_bits,up_sub_bits=a.second.state_bits;
    const int nsign=gate.nsign;
    quantize_input(p,false,stream);
    if(ordered) {
    const size_t shared_bytes = static_cast<size_t>(nsign) * sizeof(uint32_t);
    nvq2_gemv_swiglu_vec4_ordered_kernel<4>
        <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
            gate, up, p.quantized, p.scales,
            static_cast<__half *>(p.output), nullptr);
        return;
    }
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq2JscXLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        aligned_group_swiglu_pair_kernel<kNvq3JscLGroupExec, 4>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, false, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                static_cast<__half *>(p.output), nullptr);
        });
    }
}
void ffn(const Ffn& a, cudaStream_t stream) {
    const auto& p=a.pair.first; const auto gate=device_weight(p.weight),up=device_weight(a.pair.second);
    const int gate_format=p.weight.format,gate_sub_bits=p.weight.state_bits,up_sub_bits=a.pair.second.state_bits;
    const int down_gs=a.down_group,N=gate.N,down_ng=(N+down_gs-1)/down_gs,down_k_pad=down_ng*down_gs;
    quantize_input(p,false,stream);
    bool aligned_independent_rows = false;
    const char * vec4_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_SWIGLU_VEC4");
    if (vec4_env == nullptr) vec4_env = tyloi::cuda::env_lookup("TYLOINFER_NIQ_SWIGLU_VEC4");
    const bool use_vec4 = vec4_env == nullptr || vec4_env[0] != '0';
    if (gate_format == kNvq2JscXLGroupExec &&
        gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq2JscXLGroupExec>(
            gate, up, p.quantized, p.scales,
            reinterpret_cast<__half *>(a.scratch), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq3JscLGroupExec &&
               gate_sub_bits == 4 && up_sub_bits == 4) {
        launch_aligned_group_multi2_m1<kNvq3JscLGroupExec>(
            gate, up, p.quantized, p.scales,
            reinterpret_cast<__half *>(a.scratch), stream);
        aligned_independent_rows = true;
    } else if (gate_format == kNvq2Exec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Exec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                nullptr, a.scratch);
    } else if (gate_format == kNvq2JscExec && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2JscExec, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                nullptr, a.scratch);
    } else if (gate_format == kNvq2Jsc && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2Jsc, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                nullptr, a.scratch);
    } else if (use_vec4 && gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4 &&
        gate.nsign % 4 == 0 && up.nsign % 4 == 0) {
        const size_t shared_bytes = static_cast<size_t>(gate.nsign) * sizeof(uint32_t);
        nvq2_gemv_swiglu_vec4_ordered_kernel<4, true>
            <<<gate.N, dim3(32, 4), shared_bytes, stream>>>(
                gate, up, p.quantized, p.scales,
                nullptr, a.scratch);
    } else if (gate_format == kNvq2 && gate_sub_bits == 4 && up_sub_bits == 4) {
        nvq_gemv_swiglu_pair_kernel<kNvq2, 4, true, true>
            <<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                nullptr, a.scratch);
    } else {
        launch_by_format(static_cast<int>(gate_format), [&](auto tag) {
            constexpr int F = decltype(tag)::value;
            nvq_gemv_swiglu_pair_kernel<F, 4, true><<<gate.N, dim3(32, 4), 0, stream>>>(
                gate, up, p.quantized, p.scales,
                nullptr, a.scratch);
        });
    }
    switch (static_cast<int>(down_gs)) {
        case 24:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<24><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(a.scratch),
                    a.output_codes, a.output_scales,
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<24><<<down_ng, 32, 0, stream>>>(
                    a.scratch, a.output_codes,
                    a.output_scales, N, down_k_pad);
            }
            break;
        case 28:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<28><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(a.scratch),
                    a.output_codes, a.output_scales,
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<28><<<down_ng, 32, 0, stream>>>(
                    a.scratch, a.output_codes,
                    a.output_scales, N, down_k_pad);
            }
            break;
        case 32:
            if (aligned_independent_rows) {
                nvq_swiglu_quantize_f16_pair_kernel<32><<<down_ng, 32, 0, stream>>>(
                    reinterpret_cast<const __half *>(a.scratch),
                    a.output_codes, a.output_scales,
                    N, down_k_pad);
            } else {
                nvq_quantize_f32_kernel<32><<<down_ng, 32, 0, stream>>>(
                    a.scratch, a.output_codes,
                    a.output_scales, N, down_k_pad);
            }
            break;
    }
}
}
