#include "backend.h"
#include "acc_raw.h"
#include <stdexcept>
#include <vector>

namespace {
tyloi::cuda::NumericType raw_type(tyloi::ScalarType type) {
    if (type == tyloi::kFloat32) return tyloi::cuda::NumericType::f32;
    if (type == tyloi::kFloat16) return tyloi::cuda::NumericType::f16;
    if (type == tyloi::kBFloat16) return tyloi::cuda::NumericType::bf16;
    throw std::invalid_argument("unsupported residual dtype");
}
}

::tyloi::DenseTensor acc_cuda(::tyloi::DenseTensor a, ::tyloi::DenseTensor b)
{
    TYLOINFER_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous(), "acc: a must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous(), "acc: b must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(a.scalar_type() == b.scalar_type(), "acc: a/b dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(
        a.scalar_type() == ::tyloi::kFloat16 ||
            a.scalar_type() == ::tyloi::kFloat32 ||
            a.scalar_type() == ::tyloi::kBFloat16,
        "acc: dtype must be f16, f32, or bf16");
    TYLOINFER_RUNTIME_CHECK(a.sizes() == b.sizes(), "acc: a/b shape mismatch");
    int n = (int)a.numel();
    auto out = ::tyloi::cuda::empty_like(a);
    tyloi::cuda::acc_raw::add(a.data_ptr(), b.data_ptr(), out.data_ptr(), n, raw_type(a.scalar_type()), mfq_current_cuda_stream());
    return out;
}

std::vector<::tyloi::DenseTensor> acc_rms_norm_cuda(::tyloi::DenseTensor a, ::tyloi::DenseTensor b,
                                             ::tyloi::DenseTensor weight, double eps,
                                             double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm: a must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm: b must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "acc_rms_norm: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(a.scalar_type() == b.scalar_type(), "acc_rms_norm: a/b dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(a.scalar_type() == ::tyloi::kFloat16 || a.scalar_type() == ::tyloi::kFloat32,
                "acc_rms_norm: dtype must be f16 or f32");
    TYLOINFER_RUNTIME_CHECK(a.sizes() == b.sizes(), "acc_rms_norm: a/b shape mismatch");
    TYLOINFER_RUNTIME_CHECK(a.dim() >= 1, "acc_rms_norm: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    TYLOINFER_RUNTIME_CHECK(weight.numel() == D, "acc_rms_norm: weight length mismatch");

    auto sum = ::tyloi::cuda::empty_like(a);
    auto norm = ::tyloi::cuda::empty(a.sizes(), a.options().dtype(::tyloi::kFloat32));
    tyloi::cuda::acc_raw::rms(a.data_ptr(), b.data_ptr(), weight.data_ptr<float>(), sum.data_ptr(), norm.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(a.scalar_type()), tyloi::cuda::NumericType::f32, mfq_current_cuda_stream());
    return {sum, norm};
}

std::vector<::tyloi::DenseTensor> acc_rms_norm_f16_cuda(::tyloi::DenseTensor a, ::tyloi::DenseTensor b,
                                                 ::tyloi::DenseTensor weight, double eps,
                                                 double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous(), "acc_rms_norm_f16: a must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous(), "acc_rms_norm_f16: b must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == ::tyloi::kFloat32,
                "acc_rms_norm_f16: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(a.scalar_type() == ::tyloi::kFloat16 && b.scalar_type() == ::tyloi::kFloat16,
                "acc_rms_norm_f16: a/b must be f16");
    TYLOINFER_RUNTIME_CHECK(a.sizes() == b.sizes(), "acc_rms_norm_f16: a/b shape mismatch");
    TYLOINFER_RUNTIME_CHECK(a.dim() >= 1, "acc_rms_norm_f16: a must have at least one dim");
    int D = (int)a.size(-1);
    int N = (int)(a.numel() / D);
    TYLOINFER_RUNTIME_CHECK(weight.numel() == D, "acc_rms_norm_f16: weight length mismatch");

    auto sum = ::tyloi::cuda::empty_like(a);
    auto norm = ::tyloi::cuda::empty_like(a);
    tyloi::cuda::acc_raw::rms(a.data_ptr(), b.data_ptr(), weight.data_ptr<float>(), sum.data_ptr(), norm.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(a.scalar_type()), tyloi::cuda::NumericType::f16, mfq_current_cuda_stream());
    return {sum, norm};
}

std::vector<::tyloi::DenseTensor> acc_rms_norm_bf16_cuda(
    ::tyloi::DenseTensor a, ::tyloi::DenseTensor b,
    ::tyloi::DenseTensor weight, double eps, double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(a.is_cuda() && a.is_contiguous() &&
                    a.scalar_type() == ::tyloi::kBFloat16,
                "acc_rms_norm_bf16: a must be cuda contiguous bf16");
    TYLOINFER_RUNTIME_CHECK(b.is_cuda() && b.is_contiguous() &&
                    b.scalar_type() == ::tyloi::kBFloat16,
                "acc_rms_norm_bf16: b must be cuda contiguous bf16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == ::tyloi::kFloat32,
                "acc_rms_norm_bf16: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(a.sizes() == b.sizes() && a.dim() >= 1,
                "acc_rms_norm_bf16: a/b shape mismatch");
    const int D = (int)a.size(-1);
    const int N = (int)(a.numel() / D);
    TYLOINFER_RUNTIME_CHECK(D > 0 && weight.numel() == D,
                "acc_rms_norm_bf16: weight length mismatch");
    auto sum = ::tyloi::cuda::empty_like(a);
    auto norm = ::tyloi::cuda::empty_like(a);
    tyloi::cuda::acc_raw::rms(a.data_ptr(), b.data_ptr(), weight.data_ptr<float>(), sum.data_ptr(), norm.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(a.scalar_type()), tyloi::cuda::NumericType::bf16, mfq_current_cuda_stream());
    return {sum, norm};
}

std::vector<::tyloi::DenseTensor> gemma4_attn_residual_pre_norms_f16_cuda(
    ::tyloi::DenseTensor residual, ::tyloi::DenseTensor attn,
    ::tyloi::DenseTensor attn_post_weight, ::tyloi::DenseTensor dense_pre_weight,
    ::tyloi::DenseTensor router_weight, ::tyloi::DenseTensor moe_pre_weight, double eps)
{
    TYLOINFER_RUNTIME_CHECK(residual.is_cuda() && residual.is_contiguous() &&
                    residual.scalar_type() == ::tyloi::kFloat16,
                "gemma4 fused pre norms: residual must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(attn.is_cuda() && attn.is_contiguous() &&
                    attn.scalar_type() == ::tyloi::kFloat16,
                "gemma4 fused pre norms: attention output must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(residual.sizes() == attn.sizes(),
                "gemma4 fused pre norms: activation shape mismatch");
    TYLOINFER_RUNTIME_CHECK(residual.dim() >= 1,
                "gemma4 fused pre norms: activation must have at least one dimension");
    const int D = (int)residual.size(-1);
    const int N = (int)(residual.numel() / D);
    for (const auto& weight : {attn_post_weight, dense_pre_weight, router_weight, moe_pre_weight}) {
        TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == ::tyloi::kFloat32 && weight.numel() == D,
                    "gemma4 fused pre norms: weights must be cuda contiguous f32[D]");
    }

    auto residual_out = ::tyloi::cuda::empty_like(residual);
    auto dense_out = ::tyloi::cuda::empty_like(residual);
    auto router_out = ::tyloi::cuda::empty(residual.sizes(), residual.options().dtype(::tyloi::kFloat32));
    auto moe_out = ::tyloi::cuda::empty_like(residual);
    tyloi::cuda::acc_raw::gemma_pre(residual.data_ptr(), attn.data_ptr(), attn_post_weight.data_ptr<float>(), dense_pre_weight.data_ptr<float>(), router_weight.data_ptr<float>(), moe_pre_weight.data_ptr<float>(), residual_out.data_ptr(), dense_out.data_ptr(), router_out.data_ptr<float>(), moe_out.data_ptr(), N, D, float(eps), mfq_current_cuda_stream());
    return {residual_out, dense_out, router_out, moe_out};
}

::tyloi::DenseTensor gemma4_ffn_merge_f16_cuda(
    ::tyloi::DenseTensor dense, ::tyloi::DenseTensor moe, ::tyloi::DenseTensor residual,
    ::tyloi::DenseTensor dense_post_weight, ::tyloi::DenseTensor moe_post_weight,
    ::tyloi::DenseTensor final_post_weight, ::tyloi::DenseTensor layer_scale, double eps)
{
    for (const auto& value : {dense, moe, residual}) {
        TYLOINFER_RUNTIME_CHECK(value.is_cuda() && value.is_contiguous() &&
                        value.scalar_type() == ::tyloi::kFloat16,
                    "gemma4 fused FFN merge: activations must be cuda contiguous f16");
    }
    TYLOINFER_RUNTIME_CHECK(dense.sizes() == moe.sizes() && dense.sizes() == residual.sizes(),
                "gemma4 fused FFN merge: activation shape mismatch");
    TYLOINFER_RUNTIME_CHECK(dense.dim() >= 1,
                "gemma4 fused FFN merge: activation must have at least one dimension");
    const int D = (int)dense.size(-1);
    const int N = (int)(dense.numel() / D);
    for (const auto& weight : {dense_post_weight, moe_post_weight, final_post_weight}) {
        TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                        weight.scalar_type() == ::tyloi::kFloat32 && weight.numel() == D,
                    "gemma4 fused FFN merge: weights must be cuda contiguous f32[D]");
    }
    TYLOINFER_RUNTIME_CHECK(layer_scale.is_cuda() && layer_scale.is_contiguous() &&
                    layer_scale.scalar_type() == ::tyloi::kFloat16 && layer_scale.numel() == 1,
                "gemma4 fused FFN merge: layer scale must be cuda contiguous f16[1]");

    auto out = ::tyloi::cuda::empty_like(dense);
    tyloi::cuda::acc_raw::gemma_merge(dense.data_ptr(), moe.data_ptr(), residual.data_ptr(), dense_post_weight.data_ptr<float>(), moe_post_weight.data_ptr<float>(), final_post_weight.data_ptr<float>(), layer_scale.data_ptr(), out.data_ptr(), N, D, float(eps), mfq_current_cuda_stream());
    return out;
}

void decode_graph_commit_cuda(::tyloi::DenseTensor next, ::tyloi::DenseTensor generated, ::tyloi::DenseTensor step,
                              ::tyloi::DenseTensor input, ::tyloi::DenseTensor pos, ::tyloi::DenseTensor len)
{
    TYLOINFER_RUNTIME_CHECK(next.is_cuda() && next.is_contiguous() && next.scalar_type() == ::tyloi::kInt64,
                "decode_graph_commit: next must be cuda int64 contiguous");
    TYLOINFER_RUNTIME_CHECK(generated.is_cuda() && generated.is_contiguous() && generated.scalar_type() == ::tyloi::kInt64,
                "decode_graph_commit: generated must be cuda int64 contiguous");
    TYLOINFER_RUNTIME_CHECK(step.is_cuda() && step.is_contiguous() && step.scalar_type() == ::tyloi::kInt64 && step.numel() == 1,
                "decode_graph_commit: step must be cuda int64[1]");
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type() == ::tyloi::kInt64 && input.numel() == 1,
                "decode_graph_commit: input must be cuda int64[1]");
    TYLOINFER_RUNTIME_CHECK(pos.is_cuda() && pos.is_contiguous() && pos.scalar_type() == ::tyloi::kInt64 && pos.numel() == 1,
                "decode_graph_commit: pos must be cuda int64[1]");
    TYLOINFER_RUNTIME_CHECK(len.is_cuda() && len.is_contiguous() && len.scalar_type() == ::tyloi::kInt64 && len.numel() == 1,
                "decode_graph_commit: len must be cuda int64[1]");
    tyloi::cuda::acc_raw::commit(next.data_ptr<int64_t>(), generated.data_ptr<int64_t>(), step.data_ptr<int64_t>(), input.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), len.data_ptr<int64_t>(), mfq_current_cuda_stream());
}
