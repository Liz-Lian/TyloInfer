#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "rope_raw.h"
namespace raw = ::tyloi::cuda::rope_raw;
torch::Tensor rope_ext_cuda(torch::Tensor,torch::Tensor,double,int64_t,torch::Tensor);
torch::Tensor rope_cuda(torch::Tensor x, torch::Tensor pos, double base)
{
    return rope_ext_cuda(
        x, pos, base, x.size(-1),
        torch::empty({0}, torch::TensorOptions().dtype(at::kLong).device(torch::kCPU)));
}

torch::Tensor rope_ext_cuda(torch::Tensor x, torch::Tensor pos, double base, int64_t rotary_dim, torch::Tensor sections)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "rope: x must be cuda contiguous f32");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == at::kFloat,
                "rope: pos must be cuda contiguous f32");
    int T = (int)x.size(-2);
    int D = (int)x.size(-1);
    int RD = (int)rotary_dim;
    TORCH_CHECK(RD > 0 && RD <= D && RD % 2 == 0, "rope: rotary_dim must be positive even and <= D");
    TORCH_CHECK(pos.dim() == 1 || pos.dim() == 2, "rope: pos must be [T] or [A,T]");
    int pos_axes = pos.dim() == 1 ? 1 : (int)pos.size(0);
    TORCH_CHECK(pos.size(-1) == T, "rope: pos last dim must match T");
    int s0 = 0, s1 = 0, s2 = 0;
    if (sections.numel() > 0) {
        TORCH_CHECK(!sections.is_cuda() && sections.is_contiguous() && sections.scalar_type() == at::kLong,
                    "rope: sections must be CPU contiguous int64");
        TORCH_CHECK(sections.numel() == 3, "rope: sections must have 3 entries");
        const int64_t* sp = sections.data_ptr<int64_t>();
        s0 = (int)sp[0];
        s1 = (int)sp[1];
        s2 = (int)sp[2];
        TORCH_CHECK(s0 + s1 + s2 == RD / 2, "rope: sections must sum to rotary_dim/2");
    }
    int MT = (int)(x.numel() / ((size_t)T * D));
    const int pos_batches = sections.numel() == 0 && pos.dim() == 2
        ? (int)pos.size(0) : 0;
    TORCH_CHECK(pos_batches == 0 || MT % pos_batches == 0,
                "rope: batch positions do not match x leading dimensions");
    auto out = torch::empty_like(x);
    raw::rope(x.data_ptr<float>(),pos.data_ptr<float>(),out.data_ptr<float>(),MT*T,T,D,RD,pos_axes,pos_batches,s0,s1,s2,float(base),at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor rope_table_cuda(torch::Tensor x, torch::Tensor pos, torch::Tensor cos, torch::Tensor sin,
                              int64_t rotary_dim, torch::Tensor sections)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "rope_table: x must be cuda contiguous f32");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == at::kLong,
                "rope_table: pos must be cuda contiguous int64");
    TORCH_CHECK(cos.is_cuda() && cos.is_contiguous() && cos.scalar_type() == at::kFloat,
                "rope_table: cos must be cuda contiguous f32");
    TORCH_CHECK(sin.is_cuda() && sin.is_contiguous() && sin.scalar_type() == at::kFloat,
                "rope_table: sin must be cuda contiguous f32");
    int T = (int)x.size(-2);
    int D = (int)x.size(-1);
    int RD = (int)rotary_dim;
    int half = RD / 2;
    TORCH_CHECK(RD > 0 && RD <= D && RD % 2 == 0, "rope_table: rotary_dim must be positive even and <= D");
    TORCH_CHECK(pos.dim() == 1 || pos.dim() == 2, "rope_table: pos must be [T] or [A,T]");
    int pos_axes = pos.dim() == 1 ? 1 : (int)pos.size(0);
    TORCH_CHECK(pos.size(-1) == T, "rope_table: pos last dim must match T");
    TORCH_CHECK(cos.dim() == 2 && sin.dim() == 2 && cos.sizes() == sin.sizes(),
                "rope_table: cos/sin must be [table_len, rotary_dim/2]");
    TORCH_CHECK(cos.size(1) == half, "rope_table: cos/sin width mismatch");
    int table_len = (int)cos.size(0);
    int s0 = 0, s1 = 0, s2 = 0;
    if (sections.numel() > 0) {
        TORCH_CHECK(!sections.is_cuda() && sections.is_contiguous() && sections.scalar_type() == at::kLong,
                    "rope_table: sections must be CPU contiguous int64");
        TORCH_CHECK(sections.numel() == 3, "rope_table: sections must have 3 entries");
        const int64_t* sp = sections.data_ptr<int64_t>();
        s0 = (int)sp[0];
        s1 = (int)sp[1];
        s2 = (int)sp[2];
        TORCH_CHECK(s0 + s1 + s2 == half, "rope_table: sections must sum to rotary_dim/2");
    }
    int MT = (int)(x.numel() / ((size_t)T * D));
    const int pos_batches = sections.numel() == 0 && pos.dim() == 2
        ? (int)pos.size(0) : 0;
    TORCH_CHECK(pos_batches == 0 || MT % pos_batches == 0,
                "rope_table: batch positions do not match x leading dimensions");
    auto out = torch::empty_like(x);
    raw::table(x.data_ptr<float>(),pos.data_ptr<int64_t>(),cos.data_ptr<float>(),sin.data_ptr<float>(),out.data_ptr<float>(),MT*T,T,D,RD,table_len,pos_axes,pos_batches,s0,s1,s2,at::cuda::getCurrentCUDAStream().stream());
    return out;
}
