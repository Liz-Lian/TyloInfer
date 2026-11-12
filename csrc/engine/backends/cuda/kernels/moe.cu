
// GPU-resident MoE routing and legacy symmetric NINT8-0 kernels.
//
// The execution layout follows llama.cpp's CUDA mul_mat_id path:
//   * route ids are token-major [tokens, routes]
//   * large batches are compacted by expert into ids_dst/expert_bounds
//   * results are scattered back to token-major [tokens, routes, out]
//
// MFQ can avoid duplicating the activation during compaction. Shared expert
// inputs use [tokens, K], while routed/down-projection inputs use
// [tokens, routes, K]. ids_dst identifies either source row directly.
// Canonical NINT routed projection reuses nint_matmul_kernel from
// nint_matmul.cu; q/k metadata does not create an MFE-specific compute kernel.

#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "moe_cache_transfer.h"
#include "moe_raw.h"
#include "moe_copy_bytes.cuh"
#include "glu.cuh"




namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 4;
constexpr int kRouteTile = 8;

__global__ void moe_cache_scatter_kernel(
        const std::uint8_t * staging,
        std::int64_t descriptor_offset,
        int transfer_count) {
    const int transfer = static_cast<int>(blockIdx.x);
    if (transfer >= transfer_count) return;
    const auto * descriptors =
        reinterpret_cast<const tyloi::MoeCacheScatterDescriptor *>(
            staging + descriptor_offset);
    const auto item = descriptors[transfer];
    auto * destination = reinterpret_cast<std::uint8_t *>(
        static_cast<std::uintptr_t>(item.destination));
    const auto * source = staging + item.source_offset;
    const std::uint64_t nbytes = item.nbytes;
    if ((((reinterpret_cast<std::uintptr_t>(destination) |
            reinterpret_cast<std::uintptr_t>(source) |
            static_cast<std::uintptr_t>(nbytes)) & 15u) == 0u)) {
        auto * output = reinterpret_cast<uint4 *>(destination);
        const auto * input = reinterpret_cast<const uint4 *>(source);
        const std::uint64_t count = nbytes / sizeof(uint4);
        for (std::uint64_t index = threadIdx.x;
             index < count;
             index += blockDim.x) {
            output[index] = input[index];
        }
        return;
    }
    for (std::uint64_t index = threadIdx.x;
         index < nbytes;
         index += blockDim.x) {
        destination[index] = source[index];
    }
}

__global__ void moe_cache_mapped_gather_kernel(
        const tyloi::MoeCacheMappedCopyDescriptor * descriptors,
        int transfer_count) {
    const int transfer = static_cast<int>(blockIdx.y);
    if (transfer >= transfer_count) return;
    const auto item = descriptors[transfer];
    auto * destination = reinterpret_cast<std::uint8_t *>(
        static_cast<std::uintptr_t>(item.destination));
    const auto * source = reinterpret_cast<const std::uint8_t *>(
        static_cast<std::uintptr_t>(item.source));
    const std::uint64_t nbytes = item.nbytes;
    const std::uint64_t worker =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t workers =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    mfq_moe_copy_bytes(destination, source, nbytes, worker, workers);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

template <typename scalar_t>
__device__ __forceinline__ float load_float(const scalar_t * ptr, int index) {
    return static_cast<float>(ptr[index]);
}

template <>
__device__ __forceinline__ float load_float<__half>(const __half * ptr, int index) {
    return __half2float(*reinterpret_cast<const __half *>(ptr + index));
}

template <>
__device__ __forceinline__ float load_float<__nv_bfloat16>(const __nv_bfloat16 * ptr, int index) {
    return __bfloat162float(static_cast<__nv_bfloat16>(ptr[index]));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t store_float(float value) {
    return static_cast<scalar_t>(value);
}

template <>
__device__ __forceinline__ __half store_float<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 store_float<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float moe_sqrt_softplus(float value) {
    const float softplus = value > 20.0f ? value : log1pf(expf(value));
    return sqrtf(softplus);
}

#include "moe_topk512.cuh"

template <typename scalar_t>
__global__ void __launch_bounds__(128) moe_topk_kernel(
        const scalar_t * __restrict__ logits,
        const float * __restrict__ bias,
        int32_t * __restrict__ ids,
        float * __restrict__ weights,
        int rows,
        int experts,int bias_stride,
        int top_k,
        bool use_sigmoid,
        bool use_sqrt_softplus,
        bool normalize,
        bool delayed_softmax,
        float norm_floor,
        float scale) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const scalar_t * row_logits = logits + static_cast<size_t>(row) * experts;
    if(bias)bias+=static_cast<size_t>(row)*bias_stride;
    float softmax_max = -CUDART_INF_F;
    if (!use_sigmoid && !use_sqrt_softplus && !delayed_softmax) {
        for (int expert = lane; expert < experts; expert += kWarpSize) {
            float value = load_float(row_logits, expert);
            value = isnan(value) ? -FLT_MAX : value;
            softmax_max = fmaxf(softmax_max, value);
        }
        softmax_max = warp_max(softmax_max);
    }

    float softmax_sum = 1.0f;
    if (!use_sigmoid && !use_sqrt_softplus && !delayed_softmax) {
        softmax_sum = 0.0f;
        for (int expert = lane; expert < experts; expert += kWarpSize) {
            float value = load_float(row_logits, expert);
            value = isnan(value) ? -FLT_MAX : value;
            softmax_sum += expf(value - softmax_max);
        }
        softmax_sum = warp_sum(softmax_sum);
    }

    // The generic fallback also serves Nemotron's top-22/512 routing.
    int selected[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        selected[i] = -1;
    }

    for (int rank = 0; rank < top_k; ++rank) {
        float best_score = -CUDART_INF_F;
        float best_weight = -CUDART_INF_F;
        int best_expert = INT_MAX;

        for (int expert = lane; expert < experts; expert += kWarpSize) {
            bool already_selected = false;
#pragma unroll
            for (int previous = 0; previous < 32; ++previous) {
                if (previous < rank && selected[previous] == expert) {
                    already_selected = true;
                }
            }
            if (already_selected) {
                continue;
            }

            float raw = load_float(row_logits, expert);
            raw = isnan(raw) ? -FLT_MAX : raw;
            float weight;
            if (delayed_softmax) {
                weight = raw;
            } else if (use_sigmoid) {
                weight = 1.0f / (1.0f + expf(-raw));
            } else if (use_sqrt_softplus) {
                weight = moe_sqrt_softplus(raw);
            } else {
                weight = expf(raw - softmax_max) / softmax_sum;
            }
            const float score = weight + (bias == nullptr ? 0.0f : bias[expert]);
            if (score > best_score || (score == best_score && expert < best_expert)) {
                best_score = score;
                best_weight = weight;
                best_expert = expert;
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
            const float other_weight = __shfl_down_sync(0xffffffffu, best_weight, offset);
            const int other_expert = __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (other_score > best_score ||
                    (other_score == best_score && other_expert < best_expert)) {
                best_score = other_score;
                best_weight = other_weight;
                best_expert = other_expert;
            }
        }
        best_weight = __shfl_sync(0xffffffffu, best_weight, 0);
        best_expert = __shfl_sync(0xffffffffu, best_expert, 0);
        selected[rank] = best_expert;
        if (lane == 0) {
            ids[static_cast<size_t>(row) * top_k + rank] = best_expert;
            weights[static_cast<size_t>(row) * top_k + rank] = best_weight;
        }
    }

    __syncwarp();
    float value = lane < top_k
        ? weights[static_cast<size_t>(row) * top_k + lane]
        : 0.0f;
    if (delayed_softmax) {
        float selected_max = lane < top_k ? value : -CUDART_INF_F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            selected_max = fmaxf(
                selected_max,
                __shfl_down_sync(0xffffffffu, selected_max, offset));
        }
        selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
        if (lane < top_k) {
            value = expf(value - selected_max);
        } else {
            value = 0.0f;
        }
        const float denom = warp_sum(value);
        if (lane < top_k) {
            value /= denom;
        }
    } else if (normalize) {
        float denom = warp_sum(value);
        denom = fmaxf(denom, norm_floor);
        if (lane < top_k) {
            value /= denom;
        }
    }
    if (lane < top_k) {
        weights[static_cast<size_t>(row) * top_k + lane] = value * scale;
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(128) moe_topk_cached_kernel(
        const scalar_t * __restrict__ logits,
        const float * __restrict__ bias,
        int32_t * __restrict__ ids,
        float * __restrict__ weights,
        int rows,
        int experts,int bias_stride,
        int top_k,
        bool use_sigmoid,
        bool use_sqrt_softplus,
        bool normalize,
        bool delayed_softmax,
        float norm_floor,
        float scale) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const scalar_t * row_logits = logits + static_cast<size_t>(row) * experts;
    if(bias)bias+=static_cast<size_t>(row)*bias_stride;
    float softmax_max = -CUDART_INF_F;
    if (!use_sigmoid && !use_sqrt_softplus && !delayed_softmax) {
        for (int expert = lane; expert < experts; expert += kWarpSize) {
            float value = load_float(row_logits, expert);
            value = isnan(value) ? -FLT_MAX : value;
            softmax_max = fmaxf(softmax_max, value);
        }
        softmax_max = warp_max(softmax_max);
    }

    float softmax_sum = 1.0f;
    if (!use_sigmoid && !use_sqrt_softplus && !delayed_softmax) {
        softmax_sum = 0.0f;
        for (int expert = lane; expert < experts; expert += kWarpSize) {
            float value = load_float(row_logits, expert);
            value = isnan(value) ? -FLT_MAX : value;
            softmax_sum += expf(value - softmax_max);
        }
        softmax_sum = warp_sum(softmax_sum);
    }

    // Every lane owns at most sixteen experts. Constant slot indexes keep
    // their original activated weights in registers; the mask removes chosen
    // experts without a dynamically indexed selected-ID array.
    float cached_weights[16];
    unsigned int selected_mask = 0;
#pragma unroll
    for (int slot = 0; slot < 16; ++slot) {
        const int expert = lane + slot * kWarpSize;
        float weight = 0.0f;
        if (expert < experts) {
            float raw = load_float(row_logits, expert);
            raw = isnan(raw) ? -FLT_MAX : raw;
            if (delayed_softmax) {
                weight = raw;
            } else if (use_sigmoid) {
                weight = 1.0f / (1.0f + expf(-raw));
            } else if (use_sqrt_softplus) {
                weight = moe_sqrt_softplus(raw);
            } else {
                weight = expf(raw - softmax_max) / softmax_sum;
            }
        }
        cached_weights[slot] = weight;
    }

    for (int rank = 0; rank < top_k; ++rank) {
        float best_score = -CUDART_INF_F;
        float best_weight = -CUDART_INF_F;
        int best_expert = INT_MAX;

#pragma unroll
        for (int slot = 0; slot < 16; ++slot) {
            const int expert = lane + slot * kWarpSize;
            if (expert >= experts || (selected_mask & (1u << slot))) {
                continue;
            }
            const float weight = cached_weights[slot];
            const float score = weight + (bias == nullptr ? 0.0f : bias[expert]);
            if (score > best_score || (score == best_score && expert < best_expert)) {
                best_score = score;
                best_weight = weight;
                best_expert = expert;
            }
        }

#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_score = __shfl_down_sync(0xffffffffu, best_score, offset);
            const float other_weight = __shfl_down_sync(0xffffffffu, best_weight, offset);
            const int other_expert = __shfl_down_sync(0xffffffffu, best_expert, offset);
            if (other_score > best_score ||
                    (other_score == best_score && other_expert < best_expert)) {
                best_score = other_score;
                best_weight = other_weight;
                best_expert = other_expert;
            }
        }
        best_weight = __shfl_sync(0xffffffffu, best_weight, 0);
        best_expert = __shfl_sync(0xffffffffu, best_expert, 0);
        if (best_expert >= 0 && best_expert < experts && (best_expert & 31) == lane) {
            selected_mask |= 1u << (best_expert >> 5);
        }
        if (lane == 0) {
            ids[static_cast<size_t>(row) * top_k + rank] = best_expert;
            weights[static_cast<size_t>(row) * top_k + rank] = best_weight;
        }
    }

    __syncwarp();
    float value = lane < top_k
        ? weights[static_cast<size_t>(row) * top_k + lane]
        : 0.0f;
    if (delayed_softmax) {
        float selected_max = lane < top_k ? value : -CUDART_INF_F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            selected_max = fmaxf(
                selected_max,
                __shfl_down_sync(0xffffffffu, selected_max, offset));
        }
        selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
        if (lane < top_k) {
            value = expf(value - selected_max);
        } else {
            value = 0.0f;
        }
        const float denom = warp_sum(value);
        if (lane < top_k) {
            value /= denom;
        }
    } else if (normalize) {
        float denom = warp_sum(value);
        denom = fmaxf(denom, norm_floor);
        if (lane < top_k) {
            value /= denom;
        }
    }
    if (lane < top_k) {
        weights[static_cast<size_t>(row) * top_k + lane] = value * scale;
    }
}


__device__ __forceinline__ bool moe_score_before(
        float lhs_score, int lhs_expert, float rhs_score, int rhs_expert) {
    return lhs_score > rhs_score ||
        (lhs_score == rhs_score && lhs_expert < rhs_expert);
}

__global__ void __launch_bounds__(32) moe_topk_1x128_8_kernel(
        const float * __restrict__ logits,
        int32_t * __restrict__ ids,
        float * __restrict__ weights) {
    const int lane = threadIdx.x;
    float values[4];
#pragma unroll
    for (int item = 0; item < 4; ++item) {
        float value = logits[lane + item * 32];
        values[item] = isnan(value) ? -FLT_MAX : value;
    }

    float selected = 0.0f;
#pragma unroll
    for (int rank = 0; rank < 8; ++rank) {
        float best = values[0];
        int best_expert = lane;
#pragma unroll
        for (int item = 1; item < 4; ++item) {
            const int expert = lane + item * 32;
            if (moe_score_before(values[item], expert, best, best_expert)) {
                best = values[item];
                best_expert = expert;
            }
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask);
            const int other_expert = __shfl_xor_sync(0xffffffffu, best_expert, mask);
            if (moe_score_before(other, other_expert, best, best_expert)) {
                best = other;
                best_expert = other_expert;
            }
        }
        if (lane == (best_expert & 31)) {
            values[best_expert >> 5] = -CUDART_INF_F;
        }
        if (lane == rank) {
            selected = best;
            ids[rank] = best_expert;
        }
    }

    float selected_max = lane < 8 ? selected : -CUDART_INF_F;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        selected_max = fmaxf(
            selected_max,
            __shfl_down_sync(0xffffffffu, selected_max, offset));
    }
    selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
    const float value = lane < 8 ? expf(selected - selected_max) : 0.0f;
    const float denom = warp_sum(value);
    if (lane < 8) weights[lane] = value / denom;
}

__global__ void __launch_bounds__(256) moe_topk_1x256_8_kernel(
        const float * __restrict__ logits,
        int32_t * __restrict__ ids,
        float * __restrict__ weights) {
    __shared__ float scores[256];
    __shared__ int experts[256];
    const int tid = threadIdx.x;
    float score = logits[tid];
    score = isnan(score) ? -FLT_MAX : score;
    scores[tid] = score;
    experts[tid] = tid;
    __syncthreads();

#pragma unroll
    for (int width = 2; width <= 256; width <<= 1) {
#pragma unroll
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            if (other > tid) {
                const float lhs_score = scores[tid];
                const int lhs_expert = experts[tid];
                const float rhs_score = scores[other];
                const int rhs_expert = experts[other];
                const bool descending = (tid & width) == 0;
                const bool lhs_before = moe_score_before(
                    lhs_score, lhs_expert, rhs_score, rhs_expert);
                const bool do_swap = descending ? !lhs_before : lhs_before;
                if (do_swap) {
                    scores[tid] = rhs_score;
                    experts[tid] = rhs_expert;
                    scores[other] = lhs_score;
                    experts[other] = lhs_expert;
                }
            }
            __syncthreads();
        }
    }

    if (tid < 32) {
        float value = tid < 8 ? scores[tid] : 0.0f;
        float selected_max = tid < 8 ? value : -CUDART_INF_F;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            selected_max = fmaxf(
                selected_max,
                __shfl_down_sync(0xffffffffu, selected_max, offset));
        }
        selected_max = __shfl_sync(0xffffffffu, selected_max, 0);
        value = tid < 8 ? expf(value - selected_max) : 0.0f;
        const float denom = warp_sum(value);
        if (tid < 8) {
            ids[tid] = experts[tid];
            weights[tid] = value / denom;
        }
    }
}

__global__ void __launch_bounds__(256) moe_topk_1x256_6_sqrtsoftplus_kernel(
        const float * __restrict__ logits,
        const float * __restrict__ bias,
        int32_t * __restrict__ ids,
        float * __restrict__ weights,
        float norm_floor,
        float scale) {
    __shared__ float scores[256];
    __shared__ float route_weights[256];
    __shared__ int experts[256];
    const int tid = threadIdx.x;
    float raw = logits[tid];
    raw = isnan(raw) ? -FLT_MAX : raw;
    const float weight = moe_sqrt_softplus(raw);
    route_weights[tid] = weight;
    scores[tid] = weight + (bias == nullptr ? 0.0f : bias[tid]);
    experts[tid] = tid;
    __syncthreads();

#pragma unroll
    for (int width = 2; width <= 256; width <<= 1) {
#pragma unroll
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            if (other > tid) {
                const float lhs_score = scores[tid];
                const int lhs_expert = experts[tid];
                const float rhs_score = scores[other];
                const int rhs_expert = experts[other];
                const bool descending = (tid & width) == 0;
                const bool lhs_before = moe_score_before(
                    lhs_score, lhs_expert, rhs_score, rhs_expert);
                const bool do_swap = descending ? !lhs_before : lhs_before;
                if (do_swap) {
                    scores[tid] = rhs_score;
                    experts[tid] = rhs_expert;
                    scores[other] = lhs_score;
                    experts[other] = lhs_expert;
                }
            }
            __syncthreads();
        }
    }

    if (tid < 32) {
        const int expert = tid < 6 ? experts[tid] : 0;
        float value = tid < 6 ? route_weights[expert] : 0.0f;
        float denom = warp_sum(value);
        denom = fmaxf(denom, norm_floor);
        if (tid < 6) {
            ids[tid] = expert;
            weights[tid] = value / denom * scale;
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(128) moe_sqrtsoftplus_weights_kernel(
        const scalar_t * __restrict__ logits,
        const int32_t * __restrict__ ids,
        float * __restrict__ weights,
        int rows,
        int experts,
        int top_k,
        float norm_floor,
        float scale) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows) {
        return;
    }

    float value = 0.0f;
    if (lane < top_k) {
        const int expert = ids[static_cast<size_t>(row) * top_k + lane];
        if (static_cast<unsigned int>(expert) < static_cast<unsigned int>(experts)) {
            float raw = load_float(logits + static_cast<size_t>(row) * experts, expert);
            raw = isnan(raw) ? -FLT_MAX : raw;
            value = moe_sqrt_softplus(raw);
        }
    }
    float denom = warp_sum(value);
    denom = fmaxf(denom, norm_floor);
    if (lane < top_k) {
        weights[static_cast<size_t>(row) * top_k + lane] = value / denom * scale;
    }
}

__global__ void count_experts_kernel(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ counts,
        int pairs,
        int experts) {
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
            pair < pairs;
            pair += blockDim.x * gridDim.x) {
        const int expert = ids[pair];
        if (static_cast<unsigned int>(expert) < static_cast<unsigned int>(experts)) {
            atomicAdd(counts + expert, 1);
        }
    }
}

__global__ void scan_expert_counts_kernel(
        const int32_t * __restrict__ counts,
        int32_t * __restrict__ cursors,
        int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ tile_bounds,
        int32_t * __restrict__ secondary_tile_bounds,
        int32_t * __restrict__ tertiary_tile_bounds,
        int experts,
        int tile_m,
        int secondary_tile_m,
        int tertiary_tile_m) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    int pair_offset = 0;
    int tile_offset = 0;
    int secondary_tile_offset = 0;
    int tertiary_tile_offset = 0;
    expert_bounds[0] = 0;
    tile_bounds[0] = 0;
    if (secondary_tile_bounds != nullptr) {
        secondary_tile_bounds[0] = 0;
    }
    if (tertiary_tile_bounds != nullptr) {
        tertiary_tile_bounds[0] = 0;
    }
    for (int expert = 0; expert < experts; ++expert) {
        const int count = counts[expert];
        const int tiles = (count + tile_m - 1) / tile_m;
        cursors[expert] = 0;
        pair_offset += count;
        tile_offset += tiles;
        expert_bounds[expert + 1] = pair_offset;
        tile_bounds[expert + 1] = tile_offset;
        if (secondary_tile_bounds != nullptr) {
            secondary_tile_offset +=
                (count + secondary_tile_m - 1) / secondary_tile_m;
            secondary_tile_bounds[expert + 1] = secondary_tile_offset;
        }
        if (tertiary_tile_bounds != nullptr) {
            tertiary_tile_offset +=
                (count + tertiary_tile_m - 1) / tertiary_tile_m;
            tertiary_tile_bounds[expert + 1] = tertiary_tile_offset;
        }
    }
}

__global__ void fill_tile_experts_kernel(
        const int32_t * __restrict__ tile_bounds,
        int32_t * __restrict__ tile_experts,
        const int32_t * __restrict__ secondary_tile_bounds,
        int32_t * __restrict__ secondary_tile_experts,
        const int32_t * __restrict__ tertiary_tile_bounds,
        int32_t * __restrict__ tertiary_tile_experts,
        int experts) {
    for (int expert = blockIdx.x * blockDim.x + threadIdx.x;
            expert < experts;
            expert += blockDim.x * gridDim.x) {
        for (int tile = tile_bounds[expert]; tile < tile_bounds[expert + 1]; ++tile) {
            tile_experts[tile] = expert;
        }
        if (secondary_tile_bounds != nullptr) {
            for (int tile = secondary_tile_bounds[expert];
                    tile < secondary_tile_bounds[expert + 1]; ++tile) {
                secondary_tile_experts[tile] = expert;
            }
        }
        if (tertiary_tile_bounds != nullptr) {
            for (int tile = tertiary_tile_bounds[expert];
                    tile < tertiary_tile_bounds[expert + 1]; ++tile) {
                tertiary_tile_experts[tile] = expert;
            }
        }
    }
}

__global__ void scatter_routes_kernel(
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_bounds,
        int32_t * __restrict__ cursors,
        int32_t * __restrict__ ids_dst,
        int pairs,
        int experts) {
    for (int pair = blockIdx.x * blockDim.x + threadIdx.x;
            pair < pairs;
            pair += blockDim.x * gridDim.x) {
        const int expert = ids[pair];
        if (static_cast<unsigned int>(expert) >= static_cast<unsigned int>(experts)) {
            continue;
        }
        const int compact = expert_bounds[expert] + atomicAdd(cursors + expert, 1);
        ids_dst[compact] = pair;
    }
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
__device__ __forceinline__ void quantize_moe_input_group(
        const __half * __restrict__ x,
        int8_t * __restrict__ qx,
        float * __restrict__ xscale,
        int rows,
        int k_real,
        int k_pad,
        int gs,int row,int group,int groups) {
    const int tid = threadIdx.x;
    const int base = group * gs;
    const bool real = tid < gs && base + tid < k_real;
    const float value = real
        ? __half2float(x[static_cast<size_t>(row) * k_real + base + tid])
        : 0.0f;

    float max_value = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
    }
    __shared__ float warp_maxima[2];
    if ((tid & 31) == 0) {
        warp_maxima[tid >> 5] = max_value;
    }
    __syncthreads();
    if (tid < 32) {
        const int active_warps = (blockDim.x + 31) / 32;
        max_value = tid < active_warps ? warp_maxima[tid] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_value = fmaxf(max_value, __shfl_down_sync(0xffffffffu, max_value, offset));
        }
    }
    __shared__ float group_scale;
    if (tid == 0) {
        group_scale = max_value > 0.0f ? max_value / 127.0f : 1.0f;
        xscale[static_cast<size_t>(row) * groups + group] = group_scale;
    }
    __syncthreads();
    if (tid < gs) {
        int quant = 0;
        if (real) {
            quant = static_cast<int>(roundf(value / group_scale));
            quant = max(-127, min(127, quant));
        }
        qx[static_cast<size_t>(row) * k_pad + base + tid] = static_cast<int8_t>(quant);
    }
}

__global__ void __launch_bounds__(64) quantize_moe_input_kernel(const __half* x,
        int8_t* qx,float* xscale,int rows,int k_real,int k_pad,int gs) {
    quantize_moe_input_group(x,qx,xscale,rows,k_real,k_pad,gs,blockIdx.x,blockIdx.y,gridDim.y);
}

__global__ void __launch_bounds__(64) quantize_shared_moe_input_kernel(const __half* x,
        const int64_t* descriptors,int rows,int width) {
    const int64_t* item=descriptors;
    int group=blockIdx.x;
    while(group>=item[2]) {group-=int(item[2]);item+=4;}
    const int groups=int(item[2]),gs=int(item[3]);
    quantize_moe_input_group(x,reinterpret_cast<int8_t*>(item[0]),reinterpret_cast<float*>(item[1]),
        rows,width,groups*gs,gs,blockIdx.y,group,groups);
}

// Prepare only routed slots; the original row arrays remain the weight layout.
// The first occurrence of a slot owns all its row writes, including aliases.
__global__ void __launch_bounds__(64) quantize_shared_nint_rows_kernel(const __half* x,
        const int64_t* descriptors,int rows,int width,int total_groups,bool quantize,
        const int64_t* offsets,const float* scales,const float* minima,uint4* metadata,
        const int32_t* expert_local,const int32_t* ids,int pairs,int experts,
        int output_rows,int local_experts) {
    const int blocks_per_pair=(output_rows+63)/64;
    const int pair=int(blockIdx.x)/blocks_per_pair;
    const int row=(int(blockIdx.x)%blocks_per_pair)*64+int(threadIdx.x);
    int local=-1;
    if((threadIdx.x&31)==0 && pair<pairs) {
        const int expert=ids[pair];
        if(unsigned(expert)<unsigned(experts)) {
            local=expert_local[expert];
            if(unsigned(local)>=unsigned(local_experts))local=-1;
            if(local>=0)for(int prior=0;prior<pair;++prior) {
                const int earlier=ids[prior];
                if(unsigned(earlier)<unsigned(experts) && expert_local[earlier]==local) {
                    local=-1;break;
                }
            }
        }
    }
    local=__shfl_sync(0xffffffffu,local,0);
    if(local>=0 && row<output_rows) {
        const int neuron=local*output_rows+row;
        const uint64_t bit=uint64_t(offsets[neuron]);
        metadata[neuron]=make_uint4(uint32_t(bit),uint32_t(bit>>32),
            __float_as_uint(scales[neuron]),__float_as_uint(minima[neuron]));
    }
    if(quantize && int64_t(blockIdx.x)<int64_t(rows)*total_groups) {
        const int source=int(blockIdx.x)/total_groups;
        int group=int(blockIdx.x)%total_groups;
        const int64_t* item=descriptors;
        while(group>=item[2]) {group-=int(item[2]);item+=4;}
        const int groups=int(item[2]),gs=int(item[3]);
        quantize_moe_input_group(x,reinterpret_cast<int8_t*>(item[0]),reinterpret_cast<float*>(item[1]),
            rows,width,groups*gs,gs,source,group,groups);
    }
}

__device__ __forceinline__ int load_i8x4(const int8_t * values) {
    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(values);
    return static_cast<int>(bytes[0]) |
        (static_cast<int>(bytes[1]) << 8) |
        (static_cast<int>(bytes[2]) << 16) |
        (static_cast<int>(bytes[3]) << 24);
}

template <int ITEMS_PER_BLOCK>
__global__ void nint8_zero_moe_mmvq_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int tokens,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_pad,
        bool routed_input) {
    constexpr int gs = 32;
    constexpr int rows_per_warp = 2;
    constexpr int chunks = gs / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    const int token =
        static_cast<int>(blockIdx.z) * ITEMS_PER_BLOCK +
        static_cast<int>(threadIdx.y);
    const int route = static_cast<int>(blockIdx.y);
    const int lane = threadIdx.x;
    const int row0 = static_cast<int>(blockIdx.x) * rows_per_warp;
    if (token >= tokens || route >= routes) {
        return;
    }
    const int pair = token * routes + route;
    const int expert = ids[pair];
    if (static_cast<unsigned int>(expert) >=
        static_cast<unsigned int>(experts)) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int source_row = routed_input ? pair : token;

    float acc[rows_per_warp] = {0.0f, 0.0f};
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;

    for (int group_base = 0; group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const int k = group * gs + group_offset;
        const int packed_x = load_i8x4(
            qx + static_cast<size_t>(source_row) * k_pad + k);
        const float activation_scale =
            xscale[static_cast<size_t>(source_row) * groups + group];

#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int local_row = row0 + r;
            if (local_row >= out_per_expert) {
                continue;
            }
            const int weight_row =
                local_expert * out_per_expert + local_row;
            const size_t meta =
                static_cast<size_t>(weight_row) * groups + group;
            const int packed_weight = load_i8x4(
                reinterpret_cast<const int8_t *>(
                    q + meta * gs + group_offset));
            const int dot = __dp4a(packed_weight, packed_x, 0);
            acc[r] += activation_scale *
                __half2float(scale[meta]) * static_cast<float>(dot);
        }
    }

#pragma unroll
    for (int r = 0; r < rows_per_warp; ++r) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const int local_row = row0 + r;
            if (local_row < out_per_expert) {
                out[static_cast<size_t>(pair) * out_per_expert + local_row] =
                    __float2half(acc[r]);
            }
        }
    }
}

__device__ __forceinline__ int find_expert_tile(
        const int32_t * tile_bounds,
        int experts,
        int tile) {
    int low = 0;
    int high = experts;
    while (low < high) {
        const int middle = (low + high) >> 1;
        if (tile_bounds[middle + 1] <= tile) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

template <int TILE_M>
__device__ __forceinline__ void nint8_zero_moe_grouped_tile_profile(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ out,
        int lane,
        int row,
        int first,
        int last,
        int local_expert,
        int routes,
        int out_per_expert,
        int groups,
        bool routed_input) {
    constexpr int gs = 32;
    constexpr int chunks = gs / 4;
    constexpr int groups_per_warp = kWarpSize / chunks;
    float acc[TILE_M];
#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
        acc[item] = 0.0f;
    }
    const int relative_group = lane / chunks;
    const int chunk = lane - relative_group * chunks;
    const int group_offset = chunk * 4;
    const bool active_lane = relative_group < groups_per_warp;
    const int weight_row = local_expert * out_per_expert + row;
    const int k_pad = groups * gs;

    for (int group_base = 0; group_base < groups;
         group_base += groups_per_warp) {
        const int group = group_base + relative_group;
        if (!active_lane || group >= groups) {
            continue;
        }
        const size_t meta =
            static_cast<size_t>(weight_row) * groups + group;
        const int packed_weight = load_i8x4(
            reinterpret_cast<const int8_t *>(
                q + meta * gs + group_offset));
        const float weight_scale = __half2float(scale[meta]);
        const int k = group * gs + group_offset;

#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact >= last) {
                continue;
            }
            const int pair = ids_dst[compact];
            const int source_row = routed_input ? pair : pair / routes;
            const int packed_x = load_i8x4(
                qx + static_cast<size_t>(source_row) * k_pad + k);
            const int dot = __dp4a(packed_weight, packed_x, 0);
            const float activation_scale =
                xscale[static_cast<size_t>(source_row) * groups + group];
            acc[item] +=
                activation_scale * weight_scale * static_cast<float>(dot);
        }
    }

#pragma unroll
    for (int item = 0; item < TILE_M; ++item) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc[item] += __shfl_xor_sync(0xffffffffu, acc[item], offset);
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int item = 0; item < TILE_M; ++item) {
            const int compact = first + item;
            if (compact < last) {
                const int pair = ids_dst[compact];
                out[static_cast<size_t>(pair) * out_per_expert + row] =
                    __float2half(acc[item]);
            }
        }
    }
}

template <int TILE_M>
__global__ void __launch_bounds__(128) nint8_zero_moe_grouped_tile_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int max_tiles,
        bool routed_input) {
    const int row_tiles =
        (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int linear_block = static_cast<int>(blockIdx.x);
    const int row_tile = linear_block % row_tiles;
    const int tile = linear_block / row_tiles;
    if (tile >= max_tiles || tile >= tile_bounds[experts]) {
        return;
    }
    const int expert = find_expert_tile(tile_bounds, experts, tile);
    if (expert >= experts) {
        return;
    }
    const int local_expert = expert_local[expert];
    if (local_expert < 0) {
        return;
    }
    const int local_tile = tile - tile_bounds[expert];
    const int first = expert_bounds[expert] + local_tile * TILE_M;
    const int last = min(first + TILE_M, expert_bounds[expert + 1]);
    const int row = row_tile * kRowsPerBlock + threadIdx.y;
    if (row >= out_per_expert) {
        return;
    }
    nint8_zero_moe_grouped_tile_profile<TILE_M>(
        q, scale, qx, xscale, ids_dst, out, threadIdx.x, row, first, last,
        local_expert, routes, out_per_expert, groups, routed_input);
}

template <int TILE_M>
__global__ void __launch_bounds__(128)
nint8_zero_moe_grouped_tile_persistent_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int8_t * __restrict__ qx,
        const float * __restrict__ xscale,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        const int32_t * __restrict__ expert_local,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        bool routed_input) {
    const int row_tiles =
        (out_per_expert + kRowsPerBlock - 1) / kRowsPerBlock;
    const int total_tiles = tile_bounds[experts];
    const int64_t total_tasks =
        static_cast<int64_t>(total_tiles) * row_tiles;
    for (int64_t task = blockIdx.x; task < total_tasks;
         task += gridDim.x) {
        const int row_tile = static_cast<int>(task % row_tiles);
        const int tile = static_cast<int>(task / row_tiles);
        const int expert = tile_experts[tile];
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int local_tile = tile - tile_bounds[expert];
        const int first = expert_bounds[expert] + local_tile * TILE_M;
        const int last = min(first + TILE_M, expert_bounds[expert + 1]);
        const int row = row_tile * kRowsPerBlock + threadIdx.y;
        if (row >= out_per_expert) {
            continue;
        }
        nint8_zero_moe_grouped_tile_profile<TILE_M>(
            q, scale, qx, xscale, ids_dst, out, threadIdx.x, row, first,
            last, local_expert, routes, out_per_expert, groups,
            routed_input);
    }
}

constexpr int kMoeMmaBn = 64;
constexpr int kMoeMmaMaxBkStride = 120;

template <int GROUPS_PER_CHUNK, int BM>
__device__ __forceinline__ void nint8_zero_moe_mma_profile(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const __half * __restrict__ x,
        const int32_t * __restrict__ ids_dst,
        __half * __restrict__ out,
        __half (*W_s)[kMoeMmaMaxBkStride],
        __half (*X_s)[kMoeMmaMaxBkStride],
        float (*C_s)[16][16],
        int first,
        int last,
        int n0,
        int local_expert,
        int routes,
        int out_per_expert,
        int weight_out_stride,
        int weight_row_offset,
        int groups,
        int k_real,
        bool routed_input) {
    constexpr int GS = 32;
    constexpr int BK = GS * GROUPS_PER_CHUNK;
    constexpr int MTILES = BM / 16;
    constexpr int NFRAGS = kMoeMmaBn / 16;
    constexpr int ACCS_PER_WARP = (MTILES + 1) / 2;
    static_assert(BK % 16 == 0 && BK <= kMoeMmaMaxBkStride);
    static_assert(BM == 16 || BM == 32 || BM == 64);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid = warp * 32 + lane;
    const int warp_m0 = warp / NFRAGS;
    const int warp_n = warp % NFRAGS;

    using FragA = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                         __half, nvcuda::wmma::row_major>;
    using FragB = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                                         __half, nvcuda::wmma::col_major>;
    using FragC = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>;
    FragC acc[ACCS_PER_WARP];
#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        nvcuda::wmma::fill_fragment(acc[a], 0.0f);
    }

    const int chunks = (groups + GROUPS_PER_CHUNK - 1) / GROUPS_PER_CHUNK;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int gbase = chunk * GROUPS_PER_CHUNK;
        const int kb = gbase * GS;
        const int weight_tasks = kMoeMmaBn * GROUPS_PER_CHUNK;
        for (int task = tid; task < weight_tasks; task += 256) {
            const int nn = task / GROUPS_PER_CHUNK;
            const int gl = task - nn * GROUPS_PER_CHUNK;
            const int gn = n0 + nn;
            const int group = gbase + gl;
            const bool valid = gn < out_per_expert && group < groups;
            const int8_t * qg = nullptr;
            float d = 0.0f;
            if (valid) {
                const int weight_row =
                    local_expert * weight_out_stride + weight_row_offset + gn;
                const size_t meta = static_cast<size_t>(weight_row) * groups + group;
                qg = reinterpret_cast<const int8_t *>(q + meta * GS);
                d = __half2float(scale[meta]);
            }
#pragma unroll
            for (int i = 0; i < GS; ++i) {
                const float value = valid
                    ? d * static_cast<float>(qg[i])
                    : 0.0f;
                W_s[nn][gl * GS + i] = __float2half_rn(value);
            }
        }

        constexpr int kActivationVectorWidth = 8;
        static_assert(BK % kActivationVectorWidth == 0);
        constexpr int kActivationVectorsPerRow =
            BK / kActivationVectorWidth;
        constexpr int kActivationVectors =
            BM * kActivationVectorsPerRow;
        for (int index = tid; index < kActivationVectors; index += 256) {
            const int mm = index / kActivationVectorsPerRow;
            const int vector_local =
                index - mm * kActivationVectorsPerRow;
            const int k_local = vector_local * kActivationVectorWidth;
            const int compact = first + mm;
            const int k = kb + k_local;
            __half * destination = &X_s[mm][k_local];
            int source_row = -1;
            if (compact < last) {
                const int pair = ids_dst[compact];
                source_row = routed_input ? pair : pair / routes;
            }
            if (source_row < 0 || k >= k_real) {
                *reinterpret_cast<int4 *>(destination) =
                    make_int4(0, 0, 0, 0);
            } else if ((k_real & 7) == 0 && k + 7 < k_real) {
                *reinterpret_cast<int4 *>(destination) =
                    *reinterpret_cast<const int4 *>(
                        x + static_cast<size_t>(source_row) * k_real + k);
            } else {
#pragma unroll
                for (int element = 0;
                     element < kActivationVectorWidth;
                     ++element) {
                    destination[element] = k + element < k_real
                        ? x[static_cast<size_t>(source_row) * k_real +
                            k + element]
                        : __float2half_rn(0.0f);
                }
            }
        }
        __syncthreads();

        const bool warp_active = warp_m0 < MTILES;
#pragma unroll
        for (int ks = 0; ks < BK; ks += 16) {
            if (warp_active) {
                FragB bfrag;
                nvcuda::wmma::load_matrix_sync(
                    bfrag, &W_s[warp_n * 16][ks], kMoeMmaMaxBkStride);
#pragma unroll
                for (int a = 0; a < ACCS_PER_WARP; ++a) {
                    const int mi = warp_m0 + a * 2;
                    if (mi < MTILES) {
                        FragA afrag;
                        nvcuda::wmma::load_matrix_sync(
                            afrag, &X_s[mi * 16][ks], kMoeMmaMaxBkStride);
                        nvcuda::wmma::mma_sync(acc[a], afrag, bfrag, acc[a]);
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int a = 0; a < ACCS_PER_WARP; ++a) {
        const int mi = warp_m0 + a * 2;
        const bool owns = mi < MTILES;
        if (owns) {
            nvcuda::wmma::store_matrix_sync(
                &C_s[warp][0][0], acc[a], 16, nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        if (owns) {
            const int compact0 = first + mi * 16;
            const int gn0 = n0 + warp_n * 16;
#pragma unroll
            for (int element = lane; element < 16 * 16; element += 32) {
                const int r = element / 16;
                const int c = element - r * 16;
                const int compact = compact0 + r;
                const int gn = gn0 + c;
                if (compact < last && gn < out_per_expert) {
                    const int pair = ids_dst[compact];
                    out[static_cast<size_t>(pair) * out_per_expert + gn] =
                        __float2half_rn(C_s[warp][r][c]);
                }
            }
        }
        __syncthreads();
    }
}

template <int BM, bool COARSE_TILES = false>
__global__ void __launch_bounds__(256, 1) nint8_zero_moe_mma_kernel(
        const uint8_t * __restrict__ q,
        const __half * __restrict__ scale,
        const int32_t * __restrict__ expert_local,
        const __half * __restrict__ x,
        const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds,
        const int32_t * __restrict__ tile_bounds,
        const int32_t * __restrict__ tile_experts,
        __half * __restrict__ out,
        int routes,
        int experts,
        int out_per_expert,
        int groups,
        int k_real,
        bool routed_input) {
    constexpr int fine_tiles_per_mma = BM / kRouteTile;
    constexpr int groups_per_chunk = 3;
    __shared__ __half W_s[kMoeMmaBn][kMoeMmaMaxBkStride];
    __shared__ __half X_s[BM][kMoeMmaMaxBkStride];
    __shared__ float C_s[8][16][16];

    const int ntiles_n = (out_per_expert + kMoeMmaBn - 1) / kMoeMmaBn;
    const int total_tiles = tile_bounds[experts];
    const int64_t total_tasks = static_cast<int64_t>(total_tiles) * ntiles_n;
    for (int64_t task = blockIdx.x; task < total_tasks; task += gridDim.x) {
        const int tile = static_cast<int>(task / ntiles_n);
        const int ntile =
            static_cast<int>(task - static_cast<int64_t>(tile) * ntiles_n);
        const int expert = tile_experts[tile];
        const int local_tile = tile - tile_bounds[expert];
        if constexpr (!COARSE_TILES) {
            if (local_tile % fine_tiles_per_mma != 0) {
                continue;
            }
        }
        const int local_expert = expert_local[expert];
        if (local_expert < 0) {
            continue;
        }
        const int first = expert_bounds[expert] + local_tile *
            (COARSE_TILES ? BM : kRouteTile);
        const int last = min(first + BM, expert_bounds[expert + 1]);
        const int n0 = ntile * kMoeMmaBn;
        nint8_zero_moe_mma_profile<groups_per_chunk, BM>(
            q, scale, x, ids_dst, out, W_s, X_s, C_s, first, last, n0,
            local_expert, routes, out_per_expert, out_per_expert, 0,
            groups, k_real, routed_input);
    }
}

#endif

template <typename scalar_t>
__global__ void moe_weighted_reduce_kernel(
        const scalar_t * __restrict__ pair_output,
        const float * __restrict__ weights,
        scalar_t * __restrict__ output,
        int tokens,
        int routes,
        int width) {
    const size_t total = static_cast<size_t>(tokens) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total;
            index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int token = static_cast<int>(index / width);
        const int column = static_cast<int>(index - static_cast<size_t>(token) * width);
        float value = 0.0f;
        for (int route = 0; route < routes; ++route) {
            const size_t pair = static_cast<size_t>(token) * routes + route;
            value += weights[pair] * load_float(pair_output, static_cast<int>(pair * width + column));
        }
        output[index] = store_float<scalar_t>(value);
    }
}

template <bool GELU>
__global__ void moe_glu_split_kernel(
        const __half * __restrict__ gate_up,
        __half * __restrict__ output,
        int rows,
        int width) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(rows) * width;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / width);
    const int column = static_cast<int>(index - static_cast<int64_t>(row) * width);
    const int64_t base = static_cast<int64_t>(row) * (2 * width);
    const float gate = __half2float(gate_up[base + column]);
    const float up = __half2float(gate_up[base + width + column]);
    output[index] = __float2half_rn(mfq_glu<GELU>(gate, up));
}

__global__ void moe_sigmoid_table_kernel(__half* output) {
    const unsigned bits=blockIdx.x*blockDim.x+threadIdx.x;
    if(bits>=65536)return;
    const float value=__half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
    output[bits]=__float2half_rn(1.0f/(1.0f+::expf(-value)));
    output[65536+bits]=__float2half_rn(static_cast<float>(1.0/(1.0+::exp(-static_cast<double>(value)))));
}

__global__ void moe_swiglu_rounded_kernel(const __half* gate,const __half* up,
        const __half* sigmoid_table,__half* output,int64_t count) {
    const int64_t index=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count)return;
    const __half raw_gate=gate[index];
    const float value=__half2float(raw_gate);
    const __half rounded_sigmoid=sigmoid_table[__half_as_ushort(raw_gate)];
    const __half activated=__float2half_rn(value*__half2float(rounded_sigmoid));
    output[index]=__float2half_rn(__half2float(activated)*__half2float(up[index]));
}

__global__ void moe_add_shared_gate_kernel(
        const __half * __restrict__ routed,
        const __half * __restrict__ shared,
        const float * __restrict__ gate_logits,
        __half * __restrict__ output,
        int rows,
        int width) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(rows) * width;
    if (index >= total) {
        return;
    }
    const int row = static_cast<int>(index / width);
    const float gate = 1.0f / (1.0f + expf(-gate_logits[row]));
    const float value = __half2float(routed[index]) + gate * __half2float(shared[index]);
    output[index] = __float2half_rn(value);
}

__global__ void moe_weighted_reduce_shared_gate_kernel(
        const __half * __restrict__ pair_output,
        const float * __restrict__ weights,
        const __half * __restrict__ shared,
        const float * __restrict__ gate_logits,
        __half * __restrict__ output,
        int tokens,
        int routes,
        int width) {
    const size_t total = static_cast<size_t>(tokens) * width;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
            index < total;
            index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const int token = static_cast<int>(index / width);
        const int column = static_cast<int>(index - static_cast<size_t>(token) * width);
        float routed = 0.0f;
        for (int route = 0; route < routes; ++route) {
            const size_t pair = static_cast<size_t>(token) * routes + route;
            routed += weights[pair] * __half2float(pair_output[pair * width + column]);
        }
        routed = __half2float(__float2half_rn(routed));
        const float gate = 1.0f / (1.0f + expf(-gate_logits[token]));
        output[index] = __float2half_rn(
            routed + gate * __half2float(shared[index]));
    }
}





#if TYLOINFER_ENABLE_MFQ_COMPAT






#endif

} // namespace

void tyloi::moe_cache_scatter_cuda(
        const std::uint8_t * staging,
        std::int64_t descriptor_offset,
        int transfer_count,
        cudaStream_t stream) {
    if (!staging) throw std::runtime_error("MoE cache staging pointer is null");
    if (descriptor_offset < 0) throw std::runtime_error("MoE cache descriptor offset must be non-negative");
    if (transfer_count <= 0) throw std::runtime_error("MoE cache scatter requires at least one transfer");
    moe_cache_scatter_kernel<<<transfer_count, 256, 0, stream>>>(
        staging, descriptor_offset, transfer_count);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("MoE cache scatter: ") + cudaGetErrorString(error));
}

void tyloi::moe_cache_mapped_gather_cuda(
        const MoeCacheMappedCopyDescriptor * descriptors,
        int transfer_count,
        int blocks_per_transfer,
        cudaStream_t stream) {
    if (!descriptors) throw std::runtime_error("MoE cache mapped-copy descriptor pointer is null");
    if (transfer_count <= 0) throw std::runtime_error("MoE cache mapped gather requires at least one transfer");
    if (blocks_per_transfer < 1 || blocks_per_transfer > 128)
        throw std::runtime_error("MoE cache mapped gather blocks must be in [1, 128]");
    const dim3 grid(
        static_cast<unsigned int>(blocks_per_transfer),
        static_cast<unsigned int>(transfer_count));
    moe_cache_mapped_gather_kernel<<<grid, 256, 0, stream>>>(
        descriptors, transfer_count);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("MoE cache gather: ") + cudaGetErrorString(error));
}









#if TYLOINFER_ENABLE_MFQ_COMPAT








#endif











__global__ void moe_apply_expert_scale_kernel(
        float * __restrict__ weights,
        const int32_t * __restrict__ ids,
        const float * __restrict__ scales,
        int total) {
    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += blockDim.x * gridDim.x) {
        weights[index] *= scales[ids[index]];
    }
}







namespace tyloi::cuda::moe_raw {
namespace {
template<class T> void general_topk(const Router& a, cudaStream_t stream) {
    const dim3 block(32, 4);
    const int grid = (a.rows + 3) / 4;
    if (a.kernel == RouterKernel::fixed_512) {
        topk512_specialized_kernel<T, true, true><<<a.rows, 32, 0, stream>>>(
            static_cast<const T*>(a.logits), a.ids, a.weights, a.norm_floor, a.scale);
    } else if (a.kernel == RouterKernel::cached) {
        moe_topk_cached_kernel<T><<<grid, block, 0, stream>>>(
            static_cast<const T*>(a.logits), a.bias, a.ids, a.weights, a.rows, a.experts,
            a.bias_stride, a.top_k, a.sigmoid, a.sqrt_softplus, a.normalize, a.delayed_softmax,
            a.norm_floor, a.scale);
    } else {
        moe_topk_kernel<T><<<grid, block, 0, stream>>>(
            static_cast<const T*>(a.logits), a.bias, a.ids, a.weights, a.rows, a.experts,
            a.bias_stride, a.top_k, a.sigmoid, a.sqrt_softplus, a.normalize, a.delayed_softmax,
            a.norm_floor, a.scale);
    }
}
}

void topk(const Router& a, cudaStream_t stream) {
    if (a.kernel == RouterKernel::fixed_128)
        moe_topk_1x128_8_kernel<<<1, 32, 0, stream>>>(static_cast<const float*>(a.logits), a.ids, a.weights);
    else if (a.kernel == RouterKernel::fixed_256)
        moe_topk_1x256_8_kernel<<<1, 256, 0, stream>>>(static_cast<const float*>(a.logits), a.ids, a.weights);
    else if (a.kernel == RouterKernel::fixed_sqrt_256)
        moe_topk_1x256_6_sqrtsoftplus_kernel<<<1, 256, 0, stream>>>(
            static_cast<const float*>(a.logits), a.bias, a.ids, a.weights, a.norm_floor, a.scale);
    else if (a.type == NumericType::f16) general_topk<__half>(a, stream);
    else general_topk<float>(a, stream);
}

void sqrt_weights(const void* logits, const int32_t* ids, float* weights,
    int rows, int experts, int top_k, float floor, float scale, NumericType type, cudaStream_t stream) {
    const dim3 block(32, 4);
    const int grid = (rows + 3) / 4;
    if (type == NumericType::f16)
        moe_sqrtsoftplus_weights_kernel<__half><<<grid, block, 0, stream>>>(
            static_cast<const __half*>(logits), ids, weights, rows, experts, top_k, floor, scale);
    else
        moe_sqrtsoftplus_weights_kernel<float><<<grid, block, 0, stream>>>(
            static_cast<const float*>(logits), ids, weights, rows, experts, top_k, floor, scale);
}

void build_map(const RouteMap& a, cudaStream_t stream) {
    const auto error = cudaMemsetAsync(a.counts, 0, a.experts * sizeof(int32_t), stream);
    if (error != cudaSuccess) throw std::runtime_error(std::string("MoE map memset: ") + cudaGetErrorString(error));
    const int grid = std::min((a.pairs + 255) / 256, 65535);
    count_experts_kernel<<<grid, 256, 0, stream>>>(a.ids, a.counts, a.pairs, a.experts);
    scan_expert_counts_kernel<<<1, 1, 0, stream>>>(a.counts, a.cursors, a.expert_bounds,
        a.tile_bounds, a.secondary_bounds, a.tertiary_bounds, a.experts, a.tile,
        a.secondary_tile, a.tertiary_tile);
    fill_tile_experts_kernel<<<(a.experts + 255) / 256, 256, 0, stream>>>(a.tile_bounds,
        a.tile_experts, a.secondary_bounds, a.secondary_experts, a.tertiary_bounds,
        a.tertiary_experts, a.experts);
    scatter_routes_kernel<<<grid, 256, 0, stream>>>(a.ids, a.expert_bounds, a.cursors,
        a.ids_dst, a.pairs, a.experts);
}

void reduce(const void* input, const float* weights, void* output, int tokens,
    int routes, int width, NumericType type, cudaStream_t stream) {
    const int grid = static_cast<int>((static_cast<int64_t>(tokens) * width + 255) / 256);
    if (type == NumericType::f16)
        moe_weighted_reduce_kernel<__half><<<grid,256,0,stream>>>(static_cast<const __half*>(input), weights, static_cast<__half*>(output), tokens, routes, width);
    else if (type == NumericType::bf16)
        moe_weighted_reduce_kernel<__nv_bfloat16><<<grid,256,0,stream>>>(static_cast<const __nv_bfloat16*>(input), weights, static_cast<__nv_bfloat16*>(output), tokens, routes, width);
    else
        moe_weighted_reduce_kernel<float><<<grid,256,0,stream>>>(static_cast<const float*>(input), weights, static_cast<float*>(output), tokens, routes, width);
}
void glu(const void* input, void* output, int rows, int width, bool gelu, cudaStream_t stream) {
    const int grid = static_cast<int>((static_cast<int64_t>(rows) * width + 255) / 256);
    if (gelu) moe_glu_split_kernel<true><<<grid,256,0,stream>>>(static_cast<const __half*>(input), static_cast<__half*>(output), rows, width);
    else moe_glu_split_kernel<false><<<grid,256,0,stream>>>(static_cast<const __half*>(input), static_cast<__half*>(output), rows, width);
}
void sigmoid_table(void* output, cudaStream_t stream) {
    moe_sigmoid_table_kernel<<<256,256,0,stream>>>(static_cast<__half*>(output));
}
void rounded(const void* gate, const void* up, const void* table, void* output,
    int64_t count, bool double_table, cudaStream_t stream) {
    const auto* selected = static_cast<const __half*>(table) + (double_table ? 65536 : 0);
    moe_swiglu_rounded_kernel<<<static_cast<unsigned>((count+255)/256),256,0,stream>>>(
        static_cast<const __half*>(gate), static_cast<const __half*>(up), selected, static_cast<__half*>(output), count);
}
void expert_scale(float* weights, const int32_t* ids, const float* scales, int count, cudaStream_t stream) {
    moe_apply_expert_scale_kernel<<<static_cast<int>((int64_t(count)+255)/256),256,0,stream>>>(weights, ids, scales, count);
}
void add_shared(const void* routed, const void* shared, const float* gate, void* output,
    int rows, int width, cudaStream_t stream) {
    const int grid = static_cast<int>((static_cast<int64_t>(rows) * width + 255) / 256);
    moe_add_shared_gate_kernel<<<grid,256,0,stream>>>(static_cast<const __half*>(routed), static_cast<const __half*>(shared), gate, static_cast<__half*>(output), rows, width);
}
void reduce_shared(const void* pair, const float* weights, const void* shared, const float* gate,
    void* output, int tokens, int routes, int width, cudaStream_t stream) {
    const int grid = static_cast<int>((static_cast<int64_t>(tokens) * width + 255) / 256);
    moe_weighted_reduce_shared_gate_kernel<<<grid,256,0,stream>>>(static_cast<const __half*>(pair), weights, static_cast<const __half*>(shared), gate, static_cast<__half*>(output), tokens, routes, width);
}
#if TYLOINFER_ENABLE_MFQ_COMPAT
void quantize_shared(const void* input, const int64_t* descriptors, int rows, int width, int groups, cudaStream_t stream) {
    quantize_shared_moe_input_kernel<<<dim3(unsigned(groups),unsigned(rows)),64,0,stream>>>(
        static_cast<const __half*>(input), descriptors, rows, width);
}
void quantize_nint_rows(const void* input, const int64_t* descriptors, int rows, int width,
    int groups, bool quantize_input, int64_t* offsets, float* scales, float* minima,
    void* metadata, const int32_t* local, const int32_t* ids, int pairs, int experts,
    int output_rows, int local_experts, int blocks, cudaStream_t stream) {
    quantize_shared_nint_rows_kernel<<<unsigned(blocks),64,0,stream>>>(
        static_cast<const __half*>(input), descriptors, rows, width, groups, quantize_input,
        offsets, scales, minima, static_cast<uint4*>(metadata), local, ids, pairs, experts,
        output_rows, local_experts);
}
void quantize(const void* input, int8_t* qx, float* scales, int rows, int width,
    int padded_width, int groups, int group_size, cudaStream_t stream) {
    quantize_moe_input_kernel<<<dim3(rows,groups),group_size<=32?32:64,0,stream>>>(
        static_cast<const __half*>(input), qx, scales, rows, width, padded_width, group_size);
}
template<int Items> void direct_t(const Nint8& a,cudaStream_t stream) {
    nint8_zero_moe_mmvq_kernel<Items><<<dim3((a.output_rows+1)/2,a.routes,(a.tokens+Items-1)/Items),dim3(32,Items),0,stream>>>(
        a.q,static_cast<const __half*>(a.scale),a.qx,a.xscale,a.ids,a.local,static_cast<__half*>(a.output),
        a.tokens,a.routes,a.experts,a.output_rows,a.groups,a.width,a.routed_input);
}
void nint8_direct(const Nint8& a,int items,cudaStream_t stream) {
    switch(items) { case 1:direct_t<1>(a,stream);break;case 2:direct_t<2>(a,stream);break;
        case 4:direct_t<4>(a,stream);break;case 8:direct_t<8>(a,stream);break;
        case 16:direct_t<16>(a,stream);break;case 32:direct_t<32>(a,stream);break; }
}
void nint8_persistent(const Nint8& a,int blocks,cudaStream_t stream) {
    nint8_zero_moe_grouped_tile_persistent_kernel<kRouteTile><<<blocks,dim3(32,4),0,stream>>>(
        a.q,static_cast<const __half*>(a.scale),a.qx,a.xscale,a.ids_dst,a.expert_bounds,a.tile_bounds,
        a.tile_experts,a.local,static_cast<__half*>(a.output),a.routes,a.experts,a.output_rows,a.groups,a.routed_input);
}
void nint8_grouped(const Nint8& a,int blocks,int max_tiles,cudaStream_t stream) {
    nint8_zero_moe_grouped_tile_kernel<kRouteTile><<<blocks,dim3(32,4),0,stream>>>(
        a.q,static_cast<const __half*>(a.scale),a.qx,a.xscale,a.ids_dst,a.expert_bounds,a.tile_bounds,
        a.local,static_cast<__half*>(a.output),a.routes,a.experts,a.output_rows,a.groups,max_tiles,a.routed_input);
}
template<int Rows,bool Coarse> void mma_t(const Nint8& a,int blocks,cudaStream_t stream) {
    nint8_zero_moe_mma_kernel<Rows,Coarse><<<blocks,dim3(32,8),0,stream>>>(
        a.q,static_cast<const __half*>(a.scale),a.local,static_cast<const __half*>(a.input),
        a.ids_dst,a.expert_bounds,a.tile_bounds,a.tile_experts,static_cast<__half*>(a.output),
        a.routes,a.experts,a.output_rows,a.groups,a.width,a.routed_input);
}
void nint8_mma(const Nint8& a,int blocks,int rows,bool coarse,cudaStream_t stream) {
    if(rows==16) { if(coarse)mma_t<16,true>(a,blocks,stream);else mma_t<16,false>(a,blocks,stream); }
    else if(rows==32) { if(coarse)mma_t<32,true>(a,blocks,stream);else mma_t<32,false>(a,blocks,stream); }
    else { if(coarse)mma_t<64,true>(a,blocks,stream);else mma_t<64,false>(a,blocks,stream); }
}
#endif
}
