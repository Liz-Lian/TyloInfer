#include "backend.h"
#include "mx_raw.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <type_traits>
namespace raw = ::tyloi::cuda::mx_raw;
namespace {
int mxfp8_column_shift(const ::tyloi::DenseTensor& values,const ::tyloi::DenseTensor& scales) {
    return values.size(1)/scales.size(1)==128?7:5;
}
int mxfp8_row_shift(const ::tyloi::DenseTensor& values,const ::tyloi::DenseTensor& scales) {
    return scales.size(0)==values.size(0)?0:mxfp8_column_shift(values,scales);
}

template <int Tile, typename Output>
void launch_mxfp8_small_m(const ::tyloi::DenseTensor& values,const ::tyloi::DenseTensor& scales,
    const ::tyloi::DenseTensor& x,::tyloi::DenseTensor& y,int groups,int outputs_per_group,cudaStream_t stream) {
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();
    a.input=x.data_ptr();a.output=y.data_ptr();a.rows=int(x.size(0));a.outputs=int(values.size(0));a.width=int(values.size(1));
    a.groups=groups;a.outputs_per_group=outputs_per_group;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);
    a.output_float=std::is_same_v<Output,float>;raw::small8(a,stream);
}
void validate_mxfp8(
        const ::tyloi::DenseTensor & values,
        const ::tyloi::DenseTensor & scales) {
    TYLOINFER_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP8 values and scales must be CUDA tensors");
    TYLOINFER_RUNTIME_CHECK(values.scalar_type() == ::tyloi::kUInt8 &&
                    scales.scalar_type() == ::tyloi::kUInt8,
                "MXFP8 values and scales must be uint8");
    TYLOINFER_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP8 values and scales must be contiguous");
    TYLOINFER_RUNTIME_CHECK(values.dim() == 2 && scales.dim() == 2,
                "MXFP8 values and scales must be rank-2");
    TYLOINFER_RUNTIME_CHECK(values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() &&
                    values.size(1) % 32 == 0,
                "MXFP8 input width must be divisible by 32");
    const bool block128=values.size(1)%128==0 && scales.size(0)==(values.size(0)+127)/128 && scales.size(1)==values.size(1)/128;
    const bool block32=scales.size(0)==(values.size(0)+31)/32 && scales.size(1)==values.size(1)/32;
    const bool row32=scales.size(0)==values.size(0) && scales.size(1)==values.size(1)/32;
    TYLOINFER_RUNTIME_CHECK(block128 || block32 || row32,"MXFP8 scale geometry mismatch");
}

void validate_mxfp4_dense(
        const ::tyloi::DenseTensor & values,
        const ::tyloi::DenseTensor & scales) {
    TYLOINFER_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda(),
                "MXFP4 values and scales must be CUDA tensors");
    TYLOINFER_RUNTIME_CHECK(values.scalar_type() == ::tyloi::kUInt8 &&
                    scales.scalar_type() == ::tyloi::kUInt8,
                "MXFP4 values and scales must be uint8");
    TYLOINFER_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous(),
                "MXFP4 values and scales must be contiguous");
    TYLOINFER_RUNTIME_CHECK(values.dim() == 2 && scales.dim() == 2 &&
                    values.size(0) > 0 && values.size(1) > 0 &&
                    values.size(0) <= std::numeric_limits<int>::max() &&
                    values.size(1) <= std::numeric_limits<int>::max() / 2 &&
                    values.size(1) % 16 == 0 &&
                    scales.size(0) == values.size(0) &&
                    scales.size(1) == values.size(1) / 16,
                "MXFP4 weight geometry mismatch");
}

void validate_mxfp4_moe(
        const ::tyloi::DenseTensor & values,
        const ::tyloi::DenseTensor & scales,
        const ::tyloi::DenseTensor & input,
        const ::tyloi::DenseTensor & ids,
        const ::tyloi::DenseTensor & expert_local,
        const ::tyloi::DenseTensor & output,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len) {
    TYLOINFER_RUNTIME_CHECK(values.is_cuda() && scales.is_cuda() && input.is_cuda() &&
                    ids.is_cuda() && expert_local.is_cuda() && output.is_cuda(),
                "MXFP4 routed tensors must be CUDA tensors");
    TYLOINFER_RUNTIME_CHECK(values.scalar_type() == ::tyloi::kUInt8 &&
                    scales.scalar_type() == ::tyloi::kUInt8 &&
                    input.scalar_type() == ::tyloi::kFloat16 &&
                    ids.scalar_type() == ::tyloi::kInt32 &&
                    expert_local.scalar_type() == ::tyloi::kInt32 &&
                    output.scalar_type() == ::tyloi::kFloat16,
                "MXFP4 routed tensor dtypes are invalid");
    TYLOINFER_RUNTIME_CHECK(values.is_contiguous() && scales.is_contiguous() &&
                    input.is_contiguous() && ids.is_contiguous() &&
                    expert_local.is_contiguous() && output.is_contiguous(),
                "MXFP4 routed tensors must be contiguous");
    TYLOINFER_RUNTIME_CHECK(neuron_len > 0 && neuron_len % 32 == 0 &&
                    values.dim() == 2 &&
                    values.size(0) == pool_experts * out_per_expert &&
                    values.size(1) == neuron_len / 2 &&
                    scales.sizes() == ::tyloi::IntArrayRef(
                        {pool_experts * out_per_expert, neuron_len / 32}),
                "MXFP4 routed weight geometry mismatch");
    TYLOINFER_RUNTIME_CHECK((input.dim() == 2 || input.dim() == 3) &&
                    input.size(-1) == neuron_len && ids.dim() == 2 &&
                    input.size(0) == ids.size(0) &&
                    output.sizes() == ::tyloi::IntArrayRef(
                        {ids.size(0), ids.size(1), out_per_expert}) &&
                    expert_local.numel() == global_experts,
                "MXFP4 routed activation/route geometry mismatch");
}

} // namespace
::tyloi::DenseTensor mxfp8_dequant_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales) {
    validate_mxfp8(values, scales);
    auto dense = ::tyloi::cuda::empty(
        values.sizes(), values.options().dtype(::tyloi::kFloat16));
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.output=dense.data_ptr();a.outputs=int(values.size(0));a.width=int(values.size(1));a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);
    raw::dequant8(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

::tyloi::DenseTensor mxfp8_embedding_lookup_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor token_ids) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == ::tyloi::kInt64 &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP8 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1));
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = ::tyloi::cuda::empty(shape, values.options().dtype(::tyloi::kFloat16));
    const int64_t total=int64_t(count)*width;
    if(total>0) {
        raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.ids64=token_ids.data_ptr<int64_t>();a.output=output.data_ptr();a.rows=count;a.outputs=vocab;a.width=width;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);
        raw::embed8(a,mfq_current_cuda_stream());
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

::tyloi::DenseTensor mxfp8_small_m_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor x) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2,
                "MXFP8 activation must be contiguous CUDA fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(x.size(1) == values.size(1),
                "MXFP8 activation width mismatch");
    TYLOINFER_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 small-M kernel supports M in [1, 8]");
    auto y = ::tyloi::cuda::empty(
        {x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

::tyloi::DenseTensor mxfp8_matmul_f16_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor input) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 && input.size(1) == values.size(1) &&
                    input.size(0) <= std::numeric_limits<int>::max(),
                "MXFP8 activation must be non-empty contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = ::tyloi::cuda::empty({rows, outputs}, input.options());
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=input.data_ptr();a.output=result.data_ptr();a.rows=rows;a.outputs=outputs;a.width=width;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);a.outputs_per_group=outputs;
    raw::gemm8(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

::tyloi::DenseTensor mxfp8_small_m_f32_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor x) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output activation geometry mismatch");
    TYLOINFER_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 FP32-output small-M kernel supports M in [1, 8]");
    auto y = ::tyloi::cuda::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(::tyloi::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

::tyloi::DenseTensor mxfp8_gemm_f32_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor x) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                    x.is_contiguous() && x.dim() == 2 &&
                    x.size(1) == values.size(1),
                "MXFP8 FP32-output GEMM activation geometry mismatch");
    const int M = int(x.size(0));
    const int N = int(values.size(0));
    auto y = ::tyloi::cuda::empty(
        {M, N}, x.options().dtype(::tyloi::kFloat32));
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=x.data_ptr();a.output=y.data_ptr();a.rows=M;a.outputs=N;a.width=int(x.size(1));a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);a.outputs_per_group=N;a.output_float=true;
    raw::gemm8(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

namespace {
template <typename Output>
::tyloi::DenseTensor mxfp8_groupwise_gemm(
        ::tyloi::DenseTensor values,::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor input,int64_t groups) {
    validate_mxfp8(values,scales);
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.scalar_type()==::tyloi::kFloat16 &&
        input.is_contiguous() && input.dim()==3 && groups>0 && input.size(1)==groups &&
        values.size(0)%groups==0 && input.size(2)==values.size(1) && input.size(0)>0 &&
        input.size(0)<=std::numeric_limits<int>::max() && values.size(0)<=std::numeric_limits<int>::max() &&
        values.size(1)<=std::numeric_limits<int>::max() && input.device()==values.device() &&
        scales.device()==values.device(),"invalid grouped MXFP8 GEMM geometry");
    const int rows=int(input.size(0)),outputs=int(values.size(0)),width=int(values.size(1));
    const int per_group=int(values.size(0)/groups);
    auto options=input.options();
    if constexpr(std::is_same_v<Output,float>)options=options.dtype(::tyloi::kFloat32);
    auto result=::tyloi::cuda::empty({rows,outputs},options);
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=input.data_ptr();a.output=result.data_ptr();a.rows=rows;a.outputs=outputs;a.width=width;a.row_shift=mxfp8_row_shift(values,scales);a.column_shift=mxfp8_column_shift(values,scales);a.groups=int(groups);a.outputs_per_group=per_group;a.output_float=std::is_same_v<Output,float>;
    raw::gemm8(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    static std::atomic_flag recorded=ATOMIC_FLAG_INIT;
    if(!recorded.test_and_set(std::memory_order_relaxed))
        std::fprintf(stderr,"mxfp8_grouped_gemm scope=first_launch enabled=1 decode=shared_table groups=%lld rows=%d width=%d outputs=%d tile_m=%d threads=%d\n",
            static_cast<long long>(groups),rows,width,outputs,raw::prefill_rows,raw::prefill_threads);
    return result;
}
}

::tyloi::DenseTensor mxfp8_groupwise_gemm_f16_cuda(
        ::tyloi::DenseTensor values,::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor input,int64_t groups) {
    return mxfp8_groupwise_gemm<__half>(values,scales,input,groups);
}
::tyloi::DenseTensor mxfp8_groupwise_gemm_f32_cuda(
        ::tyloi::DenseTensor values,::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor input,int64_t groups) {
    return mxfp8_groupwise_gemm<float>(values,scales,input,groups);
}

::tyloi::DenseTensor mxfp8_groupwise_small_m_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3,
                "MXFP8 groupwise activation must be contiguous CUDA fp16 rank-3");
    TYLOINFER_RUNTIME_CHECK(groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise geometry mismatch");
    TYLOINFER_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise small-M kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    TYLOINFER_RUNTIME_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = ::tyloi::cuda::empty({x.size(0), values.size(0)}, x.options());
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

::tyloi::DenseTensor mxfp8_groupwise_small_m_f32_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor x,
        int64_t groups) {
    validate_mxfp8(values, scales);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 &&
                    x.is_contiguous() && x.dim() == 3 &&
                    groups > 0 && x.size(1) == groups &&
                    values.size(0) % groups == 0 &&
                    x.size(2) == values.size(1),
                "MXFP8 groupwise FP32-output geometry mismatch");
    TYLOINFER_RUNTIME_CHECK(x.size(0) >= 1 && x.size(0) <= 8,
                "MXFP8 groupwise FP32-output kernel supports M in [1, 8]");
    const int64_t outputs_per_group = values.size(0) / groups;
    TYLOINFER_RUNTIME_CHECK(outputs_per_group % 32 == 0,
                "MXFP8 groupwise outputs per group must be divisible by 32");
    auto y = ::tyloi::cuda::empty(
        {x.size(0), values.size(0)},
        x.options().dtype(::tyloi::kFloat32));
    const cudaStream_t stream = mfq_current_cuda_stream();
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
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

::tyloi::DenseTensor mxfp4_dequant_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales) {
    validate_mxfp4_dense(values, scales);
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto dense = ::tyloi::cuda::empty(
        {outputs, width}, values.options().dtype(::tyloi::kFloat16));
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.output=dense.data_ptr();a.outputs=outputs;a.width=width;
    raw::dequant4(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return dense;
}

::tyloi::DenseTensor mxfp4_embedding_lookup_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor token_ids) {
    validate_mxfp4_dense(values, scales);
    TYLOINFER_RUNTIME_CHECK(token_ids.is_cuda() && token_ids.is_contiguous() &&
                    token_ids.scalar_type() == ::tyloi::kInt64 &&
                    token_ids.get_device() == values.get_device() &&
                    token_ids.numel() <= std::numeric_limits<int>::max(),
                "MXFP4 embedding ids must be contiguous CUDA int64 on the weight device");
    const int count = static_cast<int>(token_ids.numel());
    const int vocab = static_cast<int>(values.size(0));
    const int width = static_cast<int>(values.size(1) * 2);
    auto shape = token_ids.sizes().vec();
    shape.push_back(width);
    auto output = ::tyloi::cuda::empty(shape, values.options().dtype(::tyloi::kFloat16));
    const int64_t total=int64_t(count)*width;
    if(total>0) {
        raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.ids64=token_ids.data_ptr<int64_t>();a.output=output.data_ptr();a.rows=count;a.outputs=vocab;a.width=width;
        raw::embed4(a,mfq_current_cuda_stream());
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return output;
}

::tyloi::DenseTensor mxfp4_matmul_f16_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor input) {
    validate_mxfp4_dense(values, scales);
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.scalar_type() == ::tyloi::kFloat16 &&
                    input.is_contiguous() && input.dim() == 2 &&
                    input.size(0) > 0 &&
                    input.size(0) <= std::numeric_limits<int>::max() &&
                    input.size(1) == values.size(1) * 2,
                "MXFP4 activation must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int outputs = static_cast<int>(values.size(0));
    const int width = static_cast<int>(input.size(1));
    auto result = ::tyloi::cuda::empty({rows, outputs}, input.options());
    raw::Args a{};a.values=values.data_ptr<std::uint8_t>();a.scales=scales.data_ptr<std::uint8_t>();a.input=input.data_ptr();a.output=result.data_ptr();a.rows=rows;a.outputs=outputs;a.width=width;
    raw::gemm4(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

::tyloi::DenseTensor mxfp4_moe_grouped_matmul_pool_f16_cuda(
        ::tyloi::DenseTensor values,
        ::tyloi::DenseTensor scales,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor ids,
        ::tyloi::DenseTensor expert_local,
        int64_t global_experts,
        int64_t pool_experts,
        int64_t out_per_expert,
        int64_t neuron_len,
        ::tyloi::DenseTensor output,
        ::tyloi::DenseTensor ids_dst,
        ::tyloi::DenseTensor expert_bounds,
        ::tyloi::DenseTensor tile_bounds,
        ::tyloi::DenseTensor tile_experts) {
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
        TYLOINFER_RUNTIME_CHECK(expert_bounds.is_cuda() && tile_bounds.is_cuda() && tile_experts.is_cuda() &&
            expert_bounds.scalar_type()==::tyloi::kInt32 && tile_bounds.scalar_type()==::tyloi::kInt32 &&
            tile_experts.scalar_type()==::tyloi::kInt32 && expert_bounds.is_contiguous() &&
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
    raw::routed4(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
