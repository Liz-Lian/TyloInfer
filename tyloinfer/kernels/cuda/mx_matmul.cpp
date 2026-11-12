#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "mx_raw.h"
#include "mx_check.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <type_traits>
namespace raw = ::tyloi::cuda::mx_raw;
namespace {
using mfq_jit_mx::mxfp8_column_shift;
using mfq_jit_mx::mxfp8_row_shift;
using mfq_jit_mx::validate_mxfp8;
using mfq_jit_mx::validate_mxfp4_dense;
using mfq_jit_mx::validate_mxfp4_moe;
template <int Tile, typename Output>
void launch_mxfp8_small_m(const torch::Tensor& values,const torch::Tensor& scales,
    const torch::Tensor& x,torch::Tensor& y,int groups,int outputs_per_group,cudaStream_t stream) {
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();
    a.input=x.data_ptr();a.output=y.data_ptr();a.rows=int(x.size(0));a.outputs=int(values.size(0));a.width=int(values.size(1));
    a.groups=groups;a.outputs_per_group=outputs_per_group;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);
    a.output_float=std::is_same_v<Output,float>;raw::small8(a,stream);
}
} // namespace
torch::Tensor mxfp8_dequant_cuda(
        torch::Tensor values,
        torch::Tensor scales) {
    validate_mxfp8(values, scales);
    auto dense = torch::empty(
        values.sizes(), values.options().dtype(at::kHalf));
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.output=dense.data_ptr();a.outputs=int(values.size(0));a.width=int(values.size(1));a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);
    raw::dequant8(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

torch::Tensor mxfp8_embedding_lookup_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor token_ids) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == at::kLong &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP8 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1));
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(shape, values.options().dtype(at::kHalf));
    const int64_t total=int64_t(count)*width;
    if(total>0) {
        raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.ids64=token_ids.data_ptr<int64_t>();a.output=output.data_ptr();a.rows=count;a.outputs=vocab;a.width=width;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);
        raw::embed8(a,at::cuda::getCurrentCUDAStream().stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor mxfp8_small_m_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                    x.is_contiguous() && x.dim() == 2,
                "MXFP8 activation must be contiguous CUDA fp16 rank-2");
    TORCH_CHECK(x.size(1) == values.size(1),
                "MXFP8 activation width mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 small-M kernel supports M in [1, 8]");
    auto y = torch::empty(
        {x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#define TYLOINFER_LAUNCH_MXFP8(TILE) \
    launch_mxfp8_small_m<TILE, __half>( \
        values, scales, x, y, 1, int(values.size(0)), stream)
    if (x.size(0) == 1) {
        TYLOINFER_LAUNCH_MXFP8(1);
    } else if (x.size(0) <= 2) {
        TYLOINFER_LAUNCH_MXFP8(2);
    } else if (x.size(0) <= 4) {
        TYLOINFER_LAUNCH_MXFP8(4);
    } else {
        TYLOINFER_LAUNCH_MXFP8(8);
    }
#undef TYLOINFER_LAUNCH_MXFP8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp8_matmul_f16_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor input) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 && input.size(1) == values.size(1) &&
                    input.size(0) <= std::numeric_limits<int>::max(),
                "MXFP8 activation must be non-empty contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = torch::empty({rows, outputs}, input.options());
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=input.data_ptr();a.output=result.data_ptr();a.rows=rows;a.outputs=outputs;a.width=width;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);a.outputs_per_group=outputs;
    raw::gemm8(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor mxfp8_small_m_f32_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output activation geometry mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 FP32-output small-M kernel supports M in [1, 8]");
    auto y = torch::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(at::kFloat));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#define TYLOINFER_LAUNCH_MXFP8_F32(TILE) \
    launch_mxfp8_small_m<TILE, float>( \
        values, scales, x, y, 1, int(values.size(0)), stream)
    if (x.size(0) == 1) {
        TYLOINFER_LAUNCH_MXFP8_F32(1);
    } else if (x.size(0) <= 2) {
        TYLOINFER_LAUNCH_MXFP8_F32(2);
    } else if (x.size(0) <= 4) {
        TYLOINFER_LAUNCH_MXFP8_F32(4);
    } else {
        TYLOINFER_LAUNCH_MXFP8_F32(8);
    }
#undef TYLOINFER_LAUNCH_MXFP8_F32
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp8_gemm_f32_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output GEMM activation geometry mismatch");
    const int M = int(x.size(0));
    const int N = int(values.size(0));
    auto y = torch::empty(
        {M, N}, x.options().dtype(at::kFloat));
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=x.data_ptr();a.output=y.data_ptr();a.rows=M;a.outputs=N;a.width=int(x.size(1));a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);a.outputs_per_group=N;a.output_float=true;
    raw::gemm8(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

namespace {
template <typename Output>
torch::Tensor mxfp8_groupwise_gemm(
        torch::Tensor values,torch::Tensor scales,
        torch::Tensor input,int64_t groups) {
    validate_mxfp8(values,scales);
    TORCH_CHECK(input.is_cuda() && input.scalar_type()==at::kHalf &&
        input.is_contiguous() && input.dim()==3 && groups>0 && input.size(1)==groups &&
        values.size(0)%groups==0 && input.size(2)==values.size(1) && input.size(0)>0 &&
        input.size(0)<=std::numeric_limits<int>::max() && values.size(0)<=std::numeric_limits<int>::max() &&
        values.size(1)<=std::numeric_limits<int>::max() && input.device()==values.device() &&
        scales.device()==values.device(),"invalid grouped MXFP8 GEMM geometry");
    const int rows=int(input.size(0)),outputs=int(values.size(0)),width=int(values.size(1));
    const int per_group=int(values.size(0)/groups);
    auto options=input.options();
    if constexpr(std::is_same_v<Output,float>)options=options.dtype(at::kFloat);
    auto result=torch::empty({rows,outputs},options);
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=input.data_ptr();a.output=result.data_ptr();a.rows=rows;a.outputs=outputs;a.width=width;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);a.groups=int(groups);a.outputs_per_group=per_group;a.output_float=std::is_same_v<Output,float>;
    raw::gemm8(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    static std::atomic_flag recorded=ATOMIC_FLAG_INIT;
    if(!recorded.test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr,"mxfp8_grouped_gemm scope=first_launch enabled=1 decode=shared_table groups=%lld rows=%d width=%d outputs=%d tile_m=%d threads=%d\n",
            static_cast<long long>(groups),rows,width,outputs,raw::prefill_rows,raw::prefill_threads);
    return result;
}
}

torch::Tensor mxfp8_groupwise_gemm_f16_cuda(
        torch::Tensor values,torch::Tensor scales,
        torch::Tensor input,int64_t groups) {
    return mxfp8_groupwise_gemm<__half>(values,scales,input,groups);
}
torch::Tensor mxfp8_groupwise_gemm_f32_cuda(
        torch::Tensor values,torch::Tensor scales,
        torch::Tensor input,int64_t groups) {
    return mxfp8_groupwise_gemm<float>(values,scales,input,groups);
}

torch::Tensor mxfp8_groupwise_small_m_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                    x.is_contiguous() && x.dim() == 3,
                "MXFP8 groupwise activation must be contiguous CUDA fp16 rank-3");
    TORCH_CHECK(groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise geometry mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise small-M kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    TORCH_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = torch::empty({x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#define TYLOINFER_LAUNCH_GROUPED_MXFP8(TILE) \
    launch_mxfp8_small_m<TILE, __half>( \
        values, scales, x, y, int(groups), int(outputs_per_group), stream)
    if (x.size(0) == 1) {
        TYLOINFER_LAUNCH_GROUPED_MXFP8(1);
    } else if (x.size(0) <= 2) {
        TYLOINFER_LAUNCH_GROUPED_MXFP8(2);
    } else if (x.size(0) <= 4) {
        TYLOINFER_LAUNCH_GROUPED_MXFP8(4);
    } else {
        TYLOINFER_LAUNCH_GROUPED_MXFP8(8);
    }
#undef TYLOINFER_LAUNCH_GROUPED_MXFP8
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp8_groupwise_small_m_f32_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                    x.is_contiguous() && x.dim() == 3 &&
                    groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise FP32-output geometry mismatch");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise FP32-output kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    TORCH_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = torch::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(at::kFloat));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
#define TYLOINFER_LAUNCH_GROUPED_MXFP8_F32(TILE) \
    launch_mxfp8_small_m<TILE, float>( \
        values, scales, x, y, int(groups), int(outputs_per_group), stream)
    if (x.size(0) == 1) {
        TYLOINFER_LAUNCH_GROUPED_MXFP8_F32(1);
    } else if (x.size(0) <= 2) {
        TYLOINFER_LAUNCH_GROUPED_MXFP8_F32(2);
    } else if (x.size(0) <= 4) {
        TYLOINFER_LAUNCH_GROUPED_MXFP8_F32(4);
    } else {
        TYLOINFER_LAUNCH_GROUPED_MXFP8_F32(8);
    }
#undef TYLOINFER_LAUNCH_GROUPED_MXFP8_F32
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

torch::Tensor mxfp4_dequant_cuda(
        torch::Tensor values,
        torch::Tensor scales) {
    validate_mxfp4_dense(values, scales);
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto dense = torch::empty(
        {outputs, width}, values.options().dtype(at::kHalf));
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.output=dense.data_ptr();a.outputs=outputs;a.width=width;
    raw::dequant4(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

torch::Tensor mxfp4_embedding_lookup_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor token_ids) {
    validate_mxfp4_dense(values, scales);
    TORCH_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == at::kLong &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP4 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = torch::empty(shape, values.options().dtype(at::kHalf));
    const int64_t total=int64_t(count)*width;
    if(total>0) {
        raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.ids64=token_ids.data_ptr<int64_t>();a.output=output.data_ptr();a.rows=count;a.outputs=vocab;a.width=width;
        raw::embed4(a,at::cuda::getCurrentCUDAStream().stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

torch::Tensor mxfp4_matmul_f16_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor input) {
    validate_mxfp4_dense(values, scales);
    TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kHalf &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 &&
                    input.size(0) <= std::numeric_limits<int>::max() &&
                    input.size(1) == values.size(1) * 2,
                "MXFP4 activation must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = torch::empty({rows, outputs}, input.options());
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=input.data_ptr();a.output=result.data_ptr();a.rows=rows;a.outputs=outputs;a.width=width;
    raw::gemm4(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
        torch::Tensor values,
        torch::Tensor scales,
        torch::Tensor input,
        torch::Tensor ids,
        torch::Tensor expert_local,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len,
        torch::Tensor output,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts) {
    validate_mxfp4_moe(
        values, scales, input, ids, expert_local, output,
        global_experts, pool_experts, out_per_expert, neuron_len);
    const int tokens=static_cast<int>(ids.size(0)),routes=static_cast<int>(ids.size(1));
    const int pairs=tokens*routes;
    raw::Args a{};
    a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();
    a.input=input.data_ptr();a.output=output.data_ptr();a.ids=ids.data_ptr<std::int32_t>();
    a.expert_local=expert_local.data_ptr<std::int32_t>();a.tokens=tokens;a.routes=routes;
    a.global_experts=static_cast<int>(global_experts);a.pool_experts=static_cast<int>(pool_experts);
    a.out_per_expert=static_cast<int>(out_per_expert);a.width=static_cast<int>(neuron_len);
    a.routed_input=input.dim()==3;a.compact=ids_dst.numel()==ids.numel();
    if(a.compact) {
        TORCH_CHECK(expert_bounds.is_cuda() && tile_bounds.is_cuda() && tile_experts.is_cuda() &&
            expert_bounds.scalar_type()==at::kInt && tile_bounds.scalar_type()==at::kInt &&
            tile_experts.scalar_type()==at::kInt && expert_bounds.is_contiguous() &&
            tile_bounds.is_contiguous() && tile_experts.is_contiguous() &&
            expert_bounds.numel()>=global_experts+1 && tile_bounds.numel()>=global_experts+1,
            "MXFP4 compact route map is invalid");
        a.tensor_core=pairs>=8*global_experts;
        if(a.tensor_core) {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                "mxfp4_moe_prefill_dispatch path=packed_tensor_core tokens=%d routes=%d experts=%lld pool=%lld width=%lld outputs=%lld\n",
                tokens,routes,static_cast<long long>(global_experts),static_cast<long long>(pool_experts),
                static_cast<long long>(neuron_len),static_cast<long long>(out_per_expert));
        }
        a.ids_dst=ids_dst.data_ptr<std::int32_t>();a.expert_bounds=expert_bounds.data_ptr<std::int32_t>();
        a.tile_bounds=tile_bounds.data_ptr<std::int32_t>();a.tile_experts=tile_experts.data_ptr<std::int32_t>();
        a.max_tiles=(pairs+7)/8+static_cast<int>(global_experts);
    }
    raw::routed4(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
