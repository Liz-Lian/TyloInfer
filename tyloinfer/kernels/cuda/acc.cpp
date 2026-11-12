#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "acc_raw.h"
#include <stdexcept>
#include <vector>
namespace {
tyloi::cuda::NumericType raw_type(at::ScalarType type) {
    if (type == at::kFloat) return tyloi::cuda::NumericType::f32;
    if (type == at::kHalf) return tyloi::cuda::NumericType::f16;
    if (type == at::kBFloat16) return tyloi::cuda::NumericType::bf16;
    throw std::invalid_argument("unsupported residual dtype");
}
}

torch::Tensor acc_cuda(torch::Tensor a, torch::Tensor b)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "acc: a must be cuda contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(), "acc: b must be cuda contiguous");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "acc: a/b dtype mismatch");
    TORCH_CHECK(
        a.scalar_type() == at::kHalf ||
            a.scalar_type() == at::kFloat ||
            a.scalar_type() == at::kBFloat16,
        "acc: dtype must be f16, f32, or bf16");
    TORCH_CHECK(a.sizes() == b.sizes(), "acc: a/b shape mismatch");
    int n = (int)a.numel();
    c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty_like(a);
    tyloi::cuda::acc_raw::add(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, raw_type(a.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return out;
}

std::vector<torch::Tensor> acc_rms_norm_cuda(torch::Tensor a, torch::Tensor b,
                                             torch::Tensor weight, double eps,
                                             double weight_offset)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm: a must be cuda contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm: b must be cuda contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat,
                "acc_rms_norm: weight must be cuda contiguous f32");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "acc_rms_norm: a/b dtype mismatch");
    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kFloat,
                "acc_rms_norm: dtype must be f16 or f32");
    TORCH_CHECK(a.sizes() == b.sizes(), "acc_rms_norm: a/b shape mismatch");
    TORCH_CHECK(a.dim() >= 1, "acc_rms_norm: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    TORCH_CHECK(weight.numel() == D, "acc_rms_norm: weight length mismatch");

    c10::cuda::CUDAGuard guard(a.device());
    auto sum = torch::empty_like(a);
    auto norm = torch::empty(a.sizes(), a.options().dtype(at::kFloat));
    tyloi::cuda::acc_raw::rms(a.data_ptr(), b.data_ptr(), weight.data_ptr<float>(), sum.data_ptr(), norm.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(a.scalar_type()), tyloi::cuda::NumericType::f32, at::cuda::getCurrentCUDAStream().stream());
    return {sum, norm};
}

std::vector<torch::Tensor> acc_rms_norm_f16_cuda(torch::Tensor a, torch::Tensor b,
                                                 torch::Tensor weight, double eps,
                                                 double weight_offset)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm_f16: a must be cuda contiguous");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm_f16: b must be cuda contiguous");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat,
                "acc_rms_norm_f16: weight must be cuda contiguous f32");
    TORCH_CHECK(a.scalar_type() == at::kHalf && b.scalar_type() == at::kHalf,
                "acc_rms_norm_f16: a/b must be f16");
    TORCH_CHECK(a.sizes() == b.sizes(), "acc_rms_norm_f16: a/b shape mismatch");
    TORCH_CHECK(a.dim() >= 1, "acc_rms_norm_f16: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    TORCH_CHECK(weight.numel() == D, "acc_rms_norm_f16: weight length mismatch");

    c10::cuda::CUDAGuard guard(a.device());
    auto sum = torch::empty_like(a);
    auto norm = torch::empty_like(a);
    tyloi::cuda::acc_raw::rms(a.data_ptr(), b.data_ptr(), weight.data_ptr<float>(), sum.data_ptr(), norm.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(a.scalar_type()), tyloi::cuda::NumericType::f16, at::cuda::getCurrentCUDAStream().stream());
    return {sum, norm};
}

std::vector<torch::Tensor> acc_rms_norm_bf16_cuda(
    torch::Tensor a, torch::Tensor b,
    torch::Tensor weight, double eps, double weight_offset)
{
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() &&
                    a.scalar_type() == at::kBFloat16,
                "acc_rms_norm_bf16: a must be cuda contiguous bf16");
    TORCH_CHECK(b.is_cuda() && b.is_contiguous() &&
                    b.scalar_type() == at::kBFloat16,
                "acc_rms_norm_bf16: b must be cuda contiguous bf16");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == at::kFloat,
                "acc_rms_norm_bf16: weight must be cuda contiguous f32");
    TORCH_CHECK(a.sizes() == b.sizes() && a.dim() >= 1,
                "acc_rms_norm_bf16: a/b shape mismatch");
    const int D = (int)a.size(-1);
    const int N = (int)(a.numel() / D);
    TORCH_CHECK(D > 0 && weight.numel() == D,
                "acc_rms_norm_bf16: weight length mismatch");
    c10::cuda::CUDAGuard guard(a.device());
    auto sum = torch::empty_like(a);
    auto norm = torch::empty_like(a);
    tyloi::cuda::acc_raw::rms(a.data_ptr(), b.data_ptr(), weight.data_ptr<float>(), sum.data_ptr(), norm.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(a.scalar_type()), tyloi::cuda::NumericType::bf16, at::cuda::getCurrentCUDAStream().stream());
    return {sum, norm};
}

std::vector<torch::Tensor> gemma4_attn_residual_pre_norms_f16_cuda(
    torch::Tensor residual, torch::Tensor attn,
    torch::Tensor attn_post_weight, torch::Tensor dense_pre_weight,
    torch::Tensor router_weight, torch::Tensor moe_pre_weight, double eps)
{
    TORCH_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                    residual.scalar_type() == at::kHalf,
                "gemma4 fused pre norms: residual must be cuda contiguous f16");
    TORCH_CHECK(attn.is_cuda() && attn.is_contiguous() &&
                    attn.scalar_type() == at::kHalf,
                "gemma4 fused pre norms: attention output must be cuda contiguous f16");
    TORCH_CHECK(residual.sizes() == attn.sizes(),
                "gemma4 fused pre norms: activation shape mismatch");
    TORCH_CHECK(residual.dim() >= 1,
                "gemma4 fused pre norms: activation must have at least one dimension");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / D);
    for (const auto& weight : {attn_post_weight, dense_pre_weight, router_weight, moe_pre_weight}) {
        TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == at::kFloat && weight.numel() == D,
                    "gemma4 fused pre norms: weights must be cuda contiguous f32[D]");
    }

    c10::cuda::CUDAGuard guard(residual.device());
    auto residual_out = torch::empty_like(residual);
    auto dense_out = torch::empty_like(residual);
    auto router_out = torch::empty(residual.sizes(), residual.options().dtype(at::kFloat));
    auto moe_out = torch::empty_like(residual);
    tyloi::cuda::acc_raw::gemma_pre(residual.data_ptr(), attn.data_ptr(), attn_post_weight.data_ptr<float>(), dense_pre_weight.data_ptr<float>(), router_weight.data_ptr<float>(), moe_pre_weight.data_ptr<float>(), residual_out.data_ptr(), dense_out.data_ptr(), router_out.data_ptr<float>(), moe_out.data_ptr(), N, D, float(eps), at::cuda::getCurrentCUDAStream().stream());
    return {residual_out, dense_out, router_out, moe_out};
}

torch::Tensor gemma4_ffn_merge_f16_cuda(
    torch::Tensor dense, torch::Tensor moe, torch::Tensor residual,
    torch::Tensor dense_post_weight, torch::Tensor moe_post_weight,
    torch::Tensor final_post_weight, torch::Tensor layer_scale, double eps)
{
    for (const auto& value : {dense, moe, residual}) {
        TORCH_CHECK(value.is_cuda() && value.is_contiguous() &&
                        value.scalar_type() == at::kHalf,
                    "gemma4 fused FFN merge: activations must be cuda contiguous f16");
    }
    TORCH_CHECK(dense.sizes() == moe.sizes() && dense.sizes() == residual.sizes(),
                "gemma4 fused FFN merge: activation shape mismatch");
    TORCH_CHECK(dense.dim() >= 1,
                "gemma4 fused FFN merge: activation must have at least one dimension");
    const int D = (int)dense.size(-1);
    const int N = (int)(dense.numel() / D);
    for (const auto& weight : {dense_post_weight, moe_post_weight, final_post_weight}) {
        TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == at::kFloat && weight.numel() == D,
                    "gemma4 fused FFN merge: weights must be cuda contiguous f32[D]");
    }
    TORCH_CHECK(layer_scale.is_cuda() && layer_scale.is_contiguous() &&
                    layer_scale.scalar_type() == at::kHalf && layer_scale.numel() == 1,
                "gemma4 fused FFN merge: layer scale must be cuda contiguous f16[1]");

    c10::cuda::CUDAGuard guard(dense.device());
    auto out = torch::empty_like(dense);
    tyloi::cuda::acc_raw::gemma_merge(dense.data_ptr(), moe.data_ptr(), residual.data_ptr(), dense_post_weight.data_ptr<float>(), moe_post_weight.data_ptr<float>(), final_post_weight.data_ptr<float>(), layer_scale.data_ptr(), out.data_ptr(), N, D, float(eps), at::cuda::getCurrentCUDAStream().stream());
    return out;
}

void decode_graph_commit_cuda(torch::Tensor next, torch::Tensor generated, torch::Tensor step,
                              torch::Tensor input, torch::Tensor pos, torch::Tensor len)
{
    TORCH_CHECK(next.is_cuda() && next.is_contiguous() && next.scalar_type() == at::kLong,
                "decode_graph_commit: next must be cuda int64 contiguous");
    TORCH_CHECK(generated.is_cuda() && generated.is_contiguous() && generated.scalar_type() == at::kLong,
                "decode_graph_commit: generated must be cuda int64 contiguous");
    TORCH_CHECK(step.is_cuda() && step.is_contiguous() && step.scalar_type() == at::kLong && step.numel() == 1,
                "decode_graph_commit: step must be cuda int64[1]");
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == at::kLong && input.numel() == 1,
                "decode_graph_commit: input must be cuda int64[1]");
    TORCH_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == at::kLong && pos.numel() == 1,
                "decode_graph_commit: pos must be cuda int64[1]");
    TORCH_CHECK(len.is_cuda() && len.is_contiguous() && len.scalar_type() == at::kLong && len.numel() == 1,
                "decode_graph_commit: len must be cuda int64[1]");
    c10::cuda::CUDAGuard guard(next.device());
    tyloi::cuda::acc_raw::commit(next.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step.data_ptr<int64_t>(), input.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), len.data_ptr<int64_t>(), at::cuda::getCurrentCUDAStream().stream());
}
