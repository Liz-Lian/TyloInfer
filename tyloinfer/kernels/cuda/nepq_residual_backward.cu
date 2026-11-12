#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void nepq_validate_residual(const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, int64_t, int64_t, int64_t);

namespace {
constexpr int kWarpSize = 32;

__global__ void nepq_sparse_residual_backward_input_kernel(
    const __half * dictionary,
    const int16_t * first,
    const int16_t * second,
    const __half * output_gradient,
    __half * input_gradient,
    int input_rows,
    int rows,
    int width,
    int blocks_per_row,
    int position_bits,
    int block_vectors) {
    extern __shared__ float warp_gradients[];
    const int input_row = static_cast<int>(blockIdx.y);
    const int residual_block = static_cast<int>(blockIdx.x);
    const int values_per_block = block_vectors * 8;
    const int warp_count = static_cast<int>(blockDim.x) / kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int row_group = lane / 8;
    const int component = lane & 7;
    for (int index = static_cast<int>(threadIdx.x);
         index < warp_count * values_per_block;
         index += static_cast<int>(blockDim.x)) {
        warp_gradients[index] = 0.0f;
    }
    __syncthreads();

    const int position_mask = (1 << position_bits) - 1;
    for (int row_base = warp * 4;
         row_base < rows;
         row_base += warp_count * 4) {
        const int row = row_base + row_group;
        float gradient = 0.0f;
        int first_record = -1;
        int second_record = -1;
        if (component == 0 && row < rows) {
            gradient = __half2float(output_gradient[
                static_cast<int64_t>(input_row) * rows + row]);
            const int64_t record_index =
                static_cast<int64_t>(row) * blocks_per_row + residual_block;
            first_record = first[record_index];
            second_record = second[record_index];
        }
        const int source_lane = row_group * 8;
        gradient = __shfl_sync(0xffffffffu, gradient, source_lane);
        first_record = __shfl_sync(0xffffffffu, first_record, source_lane);
        second_record = __shfl_sync(0xffffffffu, second_record, source_lane);
        const int records[2] = {first_record, second_record};
#pragma unroll
        for (int stream = 0; stream < 2; ++stream) {
            const int record = records[stream];
            if (record < 0) continue;
            const int position = record & position_mask;
            const int dictionary_id = record >> position_bits;
            atomicAdd(
                warp_gradients + warp * values_per_block +
                    position * 8 + component,
                gradient * __half2float(dictionary[
                    static_cast<int64_t>(dictionary_id) * 8 + component]));
        }
    }
    __syncthreads();

    for (int index = static_cast<int>(threadIdx.x);
         index < values_per_block;
         index += static_cast<int>(blockDim.x)) {
        const int column = residual_block * values_per_block + index;
        if (column < width) {
            float value = 0.0f;
            for (int source_warp = 0; source_warp < warp_count; ++source_warp) {
                value += warp_gradients[
                    source_warp * values_per_block + index];
            }
            const int64_t logical =
                static_cast<int64_t>(input_row) * width + column;
            input_gradient[logical] = __float2half(
                __half2float(input_gradient[logical]) + value);
        }
    }
}

__global__ void nepq_sparse_residual_backward_transpose_kernel(
    const __half * dictionary,
    const int32_t * transpose_offsets,
    const int32_t * transpose_rows,
    const int16_t * transpose_dictionary,
    const __half * output_gradient,
    const int8_t * rotation_signs,
    __half * input_gradient,
    int input_rows,
    int rows,
    int width,
    int vectors,
    bool fuse_rotation) {
    const int64_t thread =
        static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t group = thread / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int component = static_cast<int>(thread & 7);
    const int record_split = lane / 8;
    const int64_t total_groups = static_cast<int64_t>(input_rows) * vectors;
    if (group >= total_groups) return;

    const int input_row = static_cast<int>(group / vectors);
    const int vector = static_cast<int>(group -
        static_cast<int64_t>(input_row) * vectors);
    const unsigned active = __activemask();
    int begin = lane == 0 ? transpose_offsets[vector] : 0;
    int end = lane == 0 ? transpose_offsets[vector + 1] : 0;
    begin = __shfl_sync(active, begin, 0);
    end = __shfl_sync(active, end, 0);

    float value = 0.0f;
    for (int record = begin + record_split;
         record < end;
         record += 4) {
        const unsigned record_active = __activemask();
        const int source_lane = record_split * 8;
        int output = component == 0 ? transpose_rows[record] : 0;
        int dictionary_id = component == 0
            ? static_cast<int>(transpose_dictionary[record])
            : 0;
        output = __shfl_sync(record_active, output, source_lane);
        dictionary_id = __shfl_sync(
            record_active, dictionary_id, source_lane);
        float gradient = component == 0
            ? __half2float(output_gradient[
                static_cast<int64_t>(input_row) * rows + output])
            : 0.0f;
        gradient = __shfl_sync(record_active, gradient, source_lane);
        value = fmaf(
            gradient,
            __half2float(dictionary[
                static_cast<int64_t>(dictionary_id) * 8 + component]),
            value);
    }

    const unsigned reduce_active = __activemask();
    const float split1 = __shfl_sync(
        reduce_active, value, component + 8);
    const float split2 = __shfl_sync(
        reduce_active, value, component + 16);
    const float split3 = __shfl_sync(
        reduce_active, value, component + 24);
    value += split1 + split2 + split3;
    if (record_split != 0) return;

    const int column = vector * 8 + component;
    const int64_t logical =
        static_cast<int64_t>(input_row) * width + column;
    value += __half2float(input_gradient[logical]);
    if (fuse_rotation) {
        const unsigned rotate_active = __activemask();
#pragma unroll
        for (int stride = 1; stride < 8; stride <<= 1) {
            const float other = __shfl_xor_sync(
                rotate_active, value, stride, 8);
            value = (component & stride) ? other - value : value + other;
        }
        value *= 0.3535533905932738f *
            static_cast<float>(rotation_signs[column]);
    }
    input_gradient[logical] = __float2half(value);
}

}  // namespace

/** Accumulate the sparse residual input gradient into the supplied tensor. */
torch::Tensor nepq_sparse_residual_backward_input_cuda(
    torch::Tensor dictionary,
    torch::Tensor first,
    torch::Tensor second,
    torch::Tensor output_gradient,
    int64_t position_bits,
    int64_t block_vectors,
    torch::Tensor input_gradient) {
    TORCH_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.scalar_type() == at::kHalf &&
        output_gradient.is_contiguous() && output_gradient.dim() == 2 &&
        output_gradient.size(1) == first.size(0),
        "NEPQ-A backward output gradient must be CUDA fp16 [M,rows]");
    TORCH_CHECK(
        input_gradient.is_cuda() &&
        input_gradient.scalar_type() == at::kHalf &&
        input_gradient.is_contiguous() && input_gradient.dim() == 2 &&
        input_gradient.size(0) == output_gradient.size(0),
        "NEPQ-A backward input gradient must be CUDA fp16 [M,K]");
    nepq_validate_residual(
        dictionary, first, second, input_gradient.size(1),
        position_bits, block_vectors);
    TORCH_CHECK(
        dictionary.device() == output_gradient.device() &&
        output_gradient.device() == input_gradient.device(),
        "NEPQ-A backward tensors must share one CUDA device");
    c10::cuda::CUDAGuard guard(input_gradient.device());
    if (input_gradient.numel() == 0) {
        return input_gradient;
    }
    const int threads = output_gradient.size(0) <= 8 ? 1024 : 256;
    const dim3 blocks(
        static_cast<unsigned>(first.size(1)),
        static_cast<unsigned>(output_gradient.size(0)));
    const size_t shared_bytes =
        static_cast<size_t>(threads / kWarpSize) * block_vectors * 8 * sizeof(float);
    nepq_sparse_residual_backward_input_kernel<<<
        blocks, threads, shared_bytes, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        first.data_ptr<int16_t>(), second.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<at::Half>()),
        reinterpret_cast<__half *>(input_gradient.data_ptr<at::Half>()),
        static_cast<int>(output_gradient.size(0)),
        static_cast<int>(first.size(0)),
        static_cast<int>(input_gradient.size(1)),
        static_cast<int>(first.size(1)),
        static_cast<int>(position_bits),
        static_cast<int>(block_vectors));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return input_gradient;
}


/** Accumulate a transposed sparse residual gradient, optionally rotating it. */
torch::Tensor nepq_sparse_residual_backward_transpose_cuda(
    torch::Tensor dictionary,
    torch::Tensor transpose_offsets,
    torch::Tensor transpose_rows,
    torch::Tensor transpose_dictionary,
    torch::Tensor output_gradient,
    torch::Tensor rotation_signs,
    bool fuse_rotation,
    torch::Tensor input_gradient) {
    TORCH_CHECK(
        dictionary.is_cuda() &&
        dictionary.scalar_type() == at::kHalf &&
        dictionary.is_contiguous() && dictionary.dim() == 2 &&
        dictionary.size(0) == 1024 && dictionary.size(1) == 8,
        "NEPQ-A transpose dictionary must be CUDA fp16 [1024,8]");
    TORCH_CHECK(
        transpose_offsets.is_cuda() &&
        transpose_offsets.scalar_type() == at::kInt &&
        transpose_offsets.is_contiguous() && transpose_offsets.dim() == 1,
        "NEPQ-A transpose offsets must be CUDA contiguous int32");
    TORCH_CHECK(
        transpose_rows.is_cuda() &&
        transpose_rows.scalar_type() == at::kInt &&
        transpose_rows.is_contiguous() && transpose_rows.dim() == 1 &&
        transpose_dictionary.is_cuda() &&
        transpose_dictionary.scalar_type() == at::kShort &&
        transpose_dictionary.is_contiguous() &&
        transpose_dictionary.dim() == 1 &&
        transpose_dictionary.numel() == transpose_rows.numel(),
        "NEPQ-A transpose records must be matching int32/int16 vectors");
    TORCH_CHECK(
        output_gradient.is_cuda() &&
        output_gradient.scalar_type() == at::kHalf &&
        output_gradient.is_contiguous() && output_gradient.dim() == 2,
        "NEPQ-A transpose gradient must be CUDA contiguous fp16 [M,rows]");
    TORCH_CHECK(
        input_gradient.is_cuda() &&
        input_gradient.scalar_type() == at::kHalf &&
        input_gradient.is_contiguous() && input_gradient.dim() == 2 &&
        input_gradient.size(0) == output_gradient.size(0) &&
        input_gradient.size(1) % 8 == 0,
        "NEPQ-A transpose output must be CUDA contiguous fp16 [M,K]");
    const int64_t vectors = input_gradient.size(1) / 8;
    TORCH_CHECK(
        transpose_offsets.numel() == vectors + 1,
        "NEPQ-A transpose offset count mismatch");
    TORCH_CHECK(
        rotation_signs.is_cuda() &&
        rotation_signs.scalar_type() == at::kChar &&
        rotation_signs.is_contiguous() && rotation_signs.dim() == 1 &&
        (!fuse_rotation || rotation_signs.numel() == input_gradient.size(1)),
        "NEPQ-A fused rotation signs must match K");
    TORCH_CHECK(
        dictionary.device() == transpose_offsets.device() &&
        dictionary.device() == transpose_rows.device() &&
        dictionary.device() == transpose_dictionary.device() &&
        dictionary.device() == output_gradient.device() &&
        dictionary.device() == rotation_signs.device() &&
        dictionary.device() == input_gradient.device(),
        "NEPQ-A transpose tensors must share one CUDA device");
    if (input_gradient.numel() == 0) return input_gradient;

    c10::cuda::CUDAGuard guard(input_gradient.device());
    constexpr int threads = 256;
    const int64_t total_threads =
        output_gradient.size(0) * vectors * kWarpSize;
    const int blocks = static_cast<int>(
        (total_threads + threads - 1) / threads);
    nepq_sparse_residual_backward_transpose_kernel<<<
        blocks, threads, 0, at::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<const __half *>(dictionary.data_ptr<at::Half>()),
        transpose_offsets.data_ptr<int32_t>(),
        transpose_rows.data_ptr<int32_t>(),
        transpose_dictionary.data_ptr<int16_t>(),
        reinterpret_cast<const __half *>(
            output_gradient.data_ptr<at::Half>()),
        rotation_signs.data_ptr<int8_t>(),
        reinterpret_cast<__half *>(input_gradient.data_ptr<at::Half>()),
        static_cast<int>(output_gradient.size(0)),
        static_cast<int>(output_gradient.size(1)),
        static_cast<int>(input_gradient.size(1)),
        static_cast<int>(vectors),
        fuse_rotation);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return input_gradient;
}
