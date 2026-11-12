#include "nvq_raw.h"
#include "packed_nvq.cuh"
#include "nvq_format.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace {
using namespace tyloi::cuda::packed_nvq;
using tyloi::cuda::nvq_format::launch_by_format;
using tyloi::cuda::nvq_format::launch_nepq_by_format;
template <int FORMAT>
__global__ void nvq_dequant_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    __half * weight,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int segment = blockIdx.x * blockDim.x + threadIdx.x;
    if (segment >= (K + 7) / 8) return;
    for (int row = blockIdx.y; row < N; row += gridDim.y) {
        const int k0 = segment * 8;
        const int group = k0 / kGroupSize;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const float scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        auto decoded = load_nvq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        if constexpr (FORMAT == kNvq1L) {
            decoded.values.x = nvq1_l_scale_delta4(decoded.values.x, decoded.delta);
            decoded.values.y = nvq1_l_scale_delta4(decoded.values.y, decoded.delta);
        } else if constexpr (FORMAT == kNvq1S) {
            decoded.values.x = nvq1_s_scale_delta4(decoded.values.x, decoded.delta);
            decoded.values.y = nvq1_s_scale_delta4(decoded.values.y, decoded.delta);
        }
        half values[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int packed = i < 4 ? decoded.values.x : decoded.values.y;
            const int value = int(int8_t((packed >> (8 * (i & 3))) & 0xff));
            values[i] = __float2half(scale * float(value));
        }
        if ((K & 7) == 0) {
            reinterpret_cast<uint4*>(weight + int64_t(row) * K)[segment] = make_uint4(
                (uint32_t(__half_as_ushort(values[0])) | (uint32_t(__half_as_ushort(values[1])) << 16)),
                (uint32_t(__half_as_ushort(values[2])) | (uint32_t(__half_as_ushort(values[3])) << 16)),
                (uint32_t(__half_as_ushort(values[4])) | (uint32_t(__half_as_ushort(values[5])) << 16)),
                (uint32_t(__half_as_ushort(values[6])) | (uint32_t(__half_as_ushort(values[7])) << 16)));
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i)
                if (k0 + i < K) weight[int64_t(row) * K + k0 + i] = values[i];
        }
    }
}

template <int FORMAT>
__global__ void nvq_embedding_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * codebook,
    const int64_t * token_ids,
    __half * output,
    int tokens,
    int vocab,
    int K,
    int ng,
    int nvec,
    int nsign,
    int sub_bits,
    int sign_mode) {
    const int segments = (K + 7) / 8;
    const int64_t total = static_cast<int64_t>(tokens) * segments;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int token = static_cast<int>(linear / segments);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(token) * segments);
        const int row = static_cast<int>(token_ids[token]);
        if (row < 0 || row >= vocab) continue;
        const int k0 = segment * 8;
        const int group = k0 / kGroupSize;
        const int chunk0 = (k0 - group * kGroupSize) / 4;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const float scale = format_scale<FORMAT>(neuron_scale[row], sub, codebook);
        const int lo = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0, nvec, nsign, ng, sign_mode, sub);
        const int hi = decode_chunk4<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, codebook,
            row, group, chunk0 + 1, nvec, nsign, ng, sign_mode, sub);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k < K) {
                const int packed = i < 4 ? lo : hi;
                const int byte = (packed >> (8 * (i & 3))) & 0xff;
                const int value = static_cast<int>(static_cast<int8_t>(byte));
                output[static_cast<int64_t>(token) * K + k] =
                    __float2half(scale * static_cast<float>(value));
            }
        }
    }
}

template <int FORMAT>
__global__ void nepq_dequant_kernel(
    const uint8_t * indices,
    int64_t indices_nbytes,
    const uint8_t * aux,
    int64_t aux_nbytes,
    const uint8_t * sub_scale,
    int64_t sub_scale_nbytes,
    const float * neuron_scale,
    const int8_t * table_pool,
    const uint8_t * bank_ids,
    __half * weight,
    int N,
    int K,
    int ng,
    int nvec,
    int nsign,
    int nsuper,
    int table_stride,
    int sub_bits,
    int sign_mode) {
    const int64_t total = static_cast<int64_t>(N) * nsign;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(linear / nsign);
        const int segment = static_cast<int>(linear - static_cast<int64_t>(row) * nsign);
        const int group = segment / 3;
        const int segment_in_group = segment - group * 3;
        const int k0 = group * kGroupSize + segment_in_group * 8;
        const int64_t sub_linear = static_cast<int64_t>(row) * ng + group;
        const uint32_t sub = load_nvq_state<FORMAT>(indices,indices_nbytes,sub_scale,sub_scale_nbytes,
                    sub_linear,sub_bits,ng);
        const int8_t * table = nepq_active_table(
            table_pool, bank_ids, row, group, nsuper, table_stride);
        const auto values = load_nepq_vec8<FORMAT>(
            indices, indices_nbytes, aux, aux_nbytes, table,
            row, segment, group, ng, nvec, nsign, sign_mode, sub);
        const float scale = (FORMAT == kNvq1L || FORMAT == kNvq1S)
            ? neuron_scale[row] * static_cast<float>(sub)
            : format_scale<FORMAT>(neuron_scale[row], sub, table);
        const int2 packed = values.values;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int k = k0 + i;
            if (k >= K) continue;
            const int word = i < 4 ? packed.x : packed.y;
            float value = static_cast<float>(
                static_cast<int8_t>((word >> (8 * (i & 3))) & 0xff));
            if constexpr (FORMAT == kNvq1L) {
                value += 0.125f * static_cast<float>(values.delta);
            }
            if constexpr (FORMAT == kNvq1S) {
                value += 0.15625f * static_cast<float>(values.delta);
            }
            weight[static_cast<int64_t>(row) * K + k] = __float2half(scale * value);
        }
    }
}

} // namespace

namespace tyloi::cuda::nvq_raw {
using namespace tyloi::cuda::packed_nvq;
using tyloi::cuda::nvq_format::launch_by_format;
using tyloi::cuda::nvq_format::launch_nepq_by_format;
void decode(const Weight& w, void* output, cudaStream_t stream) {
    const int ng = (w.width + w.group_size - 1) / w.group_size;
    const int nvec = (w.width + (is_d4_format(w.format) ? 3 : 7)) / (is_d4_format(w.format) ? 4 : 8);
    const int nsign = (w.width + 7) / 8;
    const dim3 grid((nsign + 127) / 128, std::min(w.rows, 65535));
    launch_by_format(w.format, [&](auto tag) {
        constexpr int F = decltype(tag)::value;
        nvq_dequant_kernel<F><<<grid, 128, 0, stream>>>(w.indices,w.indices_bytes,w.aux,w.aux_bytes,
            w.state,w.state_bytes,w.scale,w.table,static_cast<__half*>(output),
            w.rows,w.width,ng,nvec,nsign,w.state_bits,w.sign_mode);
    });
}
void decode_nepq(const Weight& w, void* output, cudaStream_t stream) {
    const int ng=(w.width+kGroupSize-1)/kGroupSize;
    const int nvec=w.width/8, nsign=(w.width+7)/8;
    const int nsuper=(ng+kNepqGroupsPerSupergroup-1)/kNepqGroupsPerSupergroup;
    const int grid=static_cast<int>(std::min<int64_t>((static_cast<int64_t>(w.rows)*nsign+255)/256,65535));
    launch_nepq_by_format(w.format, [&](auto tag) {
        constexpr int F=decltype(tag)::value;
        nepq_dequant_kernel<F><<<grid,256,0,stream>>>(w.indices,w.indices_bytes,w.aux,w.aux_bytes,
            w.state,w.state_bytes,w.scale,w.table,w.banks,static_cast<__half*>(output),
            w.rows,w.width,ng,nvec,nsign,nsuper,w.table_stride,w.state_bits,0);
    });
}
void embedding(const Weight& w, const int64_t* ids, int tokens, void* output, cudaStream_t stream) {
    const int ng=(w.width+w.group_size-1)/w.group_size;
    const int nvec=(w.width+(is_d4_format(w.format)?3:7))/(is_d4_format(w.format)?4:8);
    const int nsign=(w.width+7)/8;
    const int64_t total=static_cast<int64_t>(tokens)*nsign;
    const int grid=static_cast<int>(std::min<int64_t>((total+255)/256,65535));
    launch_by_format(w.format, [&](auto tag) {
        constexpr int F=decltype(tag)::value;
        nvq_embedding_kernel<F><<<grid,256,0,stream>>>(w.indices,w.indices_bytes,w.aux,w.aux_bytes,
            w.state,w.state_bytes,w.scale,w.table,ids,static_cast<__half*>(output),tokens,w.rows,
            w.width,ng,nvec,nsign,w.state_bits,w.sign_mode);
    });
}
}
