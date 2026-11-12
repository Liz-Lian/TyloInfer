#include "backend.h"
#include "env.h"
#include "norm_raw.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
constexpr int NORM_BD = 256;
tyloi::cuda::NumericType raw_type(tyloi::ScalarType type) {
    if (type == tyloi::kFloat32) return tyloi::cuda::NumericType::f32;
    if (type == tyloi::kFloat16) return tyloi::cuda::NumericType::f16;
    if (type == tyloi::kBFloat16) return tyloi::cuda::NumericType::bf16;
    throw std::invalid_argument("unsupported norm dtype");
}
}

::tyloi::DenseTensor rms_norm_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, double eps)
{
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == ::tyloi::kFloat32,
                "rms_norm: x must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.scalar_type() == ::tyloi::kFloat32, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = ::tyloi::cuda::empty_like(x);
    tyloi::cuda::norm_raw::rms(x.data_ptr(), weight.data_ptr<float>(), out.data_ptr(), N, D, float(eps), 0.0f, raw_type(x.scalar_type()), mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor rms_norm_offset_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, double eps, double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == ::tyloi::kFloat32,
                "rms_norm_offset: x must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.scalar_type() == ::tyloi::kFloat32, "weight must be cuda f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = ::tyloi::cuda::empty_like(x);
    tyloi::cuda::norm_raw::rms(x.data_ptr(), weight.data_ptr<float>(), out.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(x.scalar_type()), mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor rms_norm_f16_cuda(::tyloi::DenseTensor x, ::tyloi::DenseTensor weight, double eps,
                                double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == ::tyloi::kFloat16,
                "rms_norm_f16: x must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == ::tyloi::kFloat32,
                "rms_norm_f16: weight must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    TYLOINFER_RUNTIME_CHECK(weight.numel() == D, "rms_norm_f16: weight length mismatch");
    auto out = ::tyloi::cuda::empty_like(x);
    tyloi::cuda::norm_raw::rms(x.data_ptr(), weight.data_ptr<float>(), out.data_ptr(), N, D, float(eps), float(weight_offset), raw_type(x.scalar_type()), mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor gdn_rms_norm_gate_cuda(::tyloi::DenseTensor attended,
    ::tyloi::DenseTensor gate,::tyloi::DenseTensor weight,double eps,
    bool silu_gate,bool output_half) {
    TYLOINFER_RUNTIME_CHECK(attended.is_cuda() && attended.is_contiguous() && attended.dim()==4 &&
        attended.scalar_type()==::tyloi::kFloat32 && attended.size(0)>0 && attended.size(1)>0 &&
        attended.size(2)>0 && attended.size(3)>0 && std::isfinite(eps) && eps>0,
        "gdn_rms_norm_gate: expected nonempty contiguous CUDA f32 [batch,heads,tokens,width]");
    const auto batch=attended.size(0),heads=attended.size(1),tokens=attended.size(2),width=attended.size(3);
    const auto rows=attended.numel()/width;
    TYLOINFER_RUNTIME_CHECK(heads<=std::numeric_limits<int>::max() && tokens<=std::numeric_limits<int>::max() &&
        width<=std::numeric_limits<int>::max() && rows<=std::numeric_limits<int>::max(),
        "gdn_rms_norm_gate: geometry exceeds CUDA launch bounds");
    TYLOINFER_RUNTIME_CHECK(gate.is_cuda() && gate.device()==attended.device() &&
        gate.dim()==3 && gate.size(0)==batch && gate.size(1)==tokens && gate.size(2)==heads*width &&
        gate.stride(2)==1 && gate.stride(1)>=heads*width &&
        (batch==1 || gate.stride(0)>=tokens*gate.stride(1)) &&
        (gate.scalar_type()==::tyloi::kFloat16 || gate.scalar_type()==::tyloi::kFloat32),
        "gdn_rms_norm_gate: expected row-strided CUDA f16/f32 [batch,tokens,heads*width] gate");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.device()==attended.device() && weight.is_contiguous() &&
        weight.scalar_type()==::tyloi::kFloat32 && weight.dim()==1 && weight.numel()==width,
        "gdn_rms_norm_gate: expected CUDA f32 width-vector weight");
    MfqCudaGuard guard(attended.device());
    auto output=::tyloi::cuda::empty({batch,tokens,heads*width},attended.options().dtype(output_half?::tyloi::kFloat16: ::tyloi::kFloat32));
    // Match the arithmetic selected after the old mandatory contiguous copy.
    const bool double_sigmoid=gate.scalar_type()==::tyloi::kFloat32 && tokens>1 && heads>1;
    const auto* warp_setting=tyloi::cuda::env_lookup("TYLOINFER_GDN_NORM_WARP_REDUCTION");
    const bool warp_tail=warp_setting?warp_setting[0]=='1':width==128 && tokens>=2 && tokens<=8;
    tyloi::cuda::norm_raw::gdn_gate(attended.data_ptr<float>(), gate.data_ptr(), weight.data_ptr<float>(), output.data_ptr(), int(rows), int(heads), int(tokens), int(width), eps, silu_gate, double_sigmoid, warp_tail, gate.stride(0), gate.stride(1), raw_type(gate.scalar_type()), output_half ? tyloi::cuda::NumericType::f16 : tyloi::cuda::NumericType::f32, mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

::tyloi::DenseTensor grouped_gated_rms_norm_cuda(
    ::tyloi::DenseTensor x, ::tyloi::DenseTensor gate,
    ::tyloi::DenseTensor weight, int64_t group_size, double eps)
{
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() > 0 &&
        gate.is_cuda() && gate.is_contiguous() && gate.device() == x.device() &&
        gate.sizes() == x.sizes(),
        "grouped_gated_rms_norm: x and gate must be matching contiguous CUDA tensors");
    TYLOINFER_RUNTIME_CHECK((x.scalar_type() == ::tyloi::kFloat32 || x.scalar_type() == ::tyloi::kFloat16 ||
        x.scalar_type() == ::tyloi::kBFloat16) &&
        (gate.scalar_type() == ::tyloi::kFloat32 || gate.scalar_type() == ::tyloi::kFloat16 ||
        gate.scalar_type() == ::tyloi::kBFloat16),
        "grouped_gated_rms_norm: x and gate dtypes must be f32, f16 or bf16");
    const int64_t total_width = x.size(-1);
    TYLOINFER_RUNTIME_CHECK(group_size > 0 && total_width % group_size == 0,
        "grouped_gated_rms_norm: group_size must divide the last dimension");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
        weight.device() == x.device() && weight.scalar_type() == ::tyloi::kFloat32 &&
        weight.dim() == 1 && weight.numel() == total_width,
        "grouped_gated_rms_norm: weight must be contiguous CUDA f32 of the input width");
    auto output = ::tyloi::cuda::empty_like(x);
    const int64_t blocks = x.numel() / group_size;
    if (blocks == 0) return output;
    tyloi::cuda::norm_raw::grouped_gated(x.data_ptr(), gate.data_ptr(), weight.data_ptr<float>(), output.data_ptr(), blocks, int(group_size), int(total_width/group_size), float(eps), raw_type(x.scalar_type()), raw_type(gate.scalar_type()), mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

::tyloi::DenseTensor grouped_rms_norm_cuda(::tyloi::DenseTensor x,
    ::tyloi::DenseTensor weight, int64_t group, double eps, double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() > 0 && group > 0 &&
        x.size(-1) > 0 && x.size(-1) % group == 0 &&
        (x.scalar_type() == ::tyloi::kFloat16 || x.scalar_type() == ::tyloi::kFloat32),
        "grouped_rms_norm: input must be contiguous cuda f16/f32 with complete groups");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.device() == x.device() && weight.is_contiguous() &&
        weight.scalar_type() == ::tyloi::kFloat32 && weight.dim() == 1 && weight.numel() == x.size(-1),
        "grouped_rms_norm: weight must be contiguous cuda f32 with input width");
    const int rows = static_cast<int>(x.numel() / group);
    const int width = static_cast<int>(group), weight_rows = static_cast<int>(x.size(-1) / group);
    auto output = ::tyloi::cuda::empty_like(x);
    if (rows == 0) return output;
    tyloi::cuda::norm_raw::grouped(x.data_ptr(), weight.data_ptr<float>(), output.data_ptr(), rows, width, weight_rows, float(eps), float(weight_offset), raw_type(x.scalar_type()), mfq_current_cuda_stream());
    return output;
}

::tyloi::DenseTensor qwen_rms_norm_bf16_cuda(
    ::tyloi::DenseTensor input, ::tyloi::DenseTensor weight, double eps,
    double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() &&
                    input.scalar_type() == ::tyloi::kBFloat16,
                "qwen_rms_norm_bf16: input must be cuda contiguous bf16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() &&
                    weight.scalar_type() == ::tyloi::kFloat32,
                "qwen_rms_norm_bf16: weight must be cuda contiguous f32");
    const int D = (int)input.size(-1);
    const int64_t count = input.numel();
    TYLOINFER_RUNTIME_CHECK(D > 0 && weight.numel() == D,
                "qwen_rms_norm_bf16: weight length mismatch");
    const int N = (int)(count / D);
    auto stream = mfq_current_cuda_stream();
    auto output = ::tyloi::cuda::empty_like(input);
    tyloi::cuda::norm_raw::qwen_bf16(input.data_ptr(), weight.data_ptr<float>(), output.data_ptr(), N, D, float(eps), float(weight_offset), stream);
    return output;
}

std::vector<::tyloi::DenseTensor> qwen_rms_norm_pair_bf16_cuda(
    ::tyloi::DenseTensor first, ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor first_weight, ::tyloi::DenseTensor second_weight,
    double eps, double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(first.is_cuda() && first.is_contiguous() &&
                    first.scalar_type() == ::tyloi::kBFloat16 &&
                    second.is_cuda() && second.is_contiguous() &&
                    second.scalar_type() == ::tyloi::kBFloat16,
                "paired Qwen RMSNorm inputs must be cuda contiguous bf16");
    TYLOINFER_RUNTIME_CHECK(first_weight.is_cuda() && first_weight.is_contiguous() &&
                    first_weight.scalar_type() == ::tyloi::kFloat32 &&
                    second_weight.is_cuda() && second_weight.is_contiguous() &&
                    second_weight.scalar_type() == ::tyloi::kFloat32,
                "paired Qwen RMSNorm weights must be cuda contiguous f32");
    const int D = (int)first.size(-1);
    TYLOINFER_RUNTIME_CHECK(D > 0 && second.size(-1) == D &&
                    first_weight.numel() == D && second_weight.numel() == D,
                "paired Qwen RMSNorm widths must match");
    const int64_t first_count = first.numel();
    const int64_t total_count = first_count + second.numel();
    const int64_t total_rows = total_count / D;
    auto squared = ::tyloi::cuda::empty(
        {total_rows, D}, first.options().dtype(::tyloi::kFloat32));
    const int blocks = (int)std::min<int64_t>(
        (total_count + NORM_BD - 1) / NORM_BD, 65535);
    auto stream = mfq_current_cuda_stream();
    tyloi::cuda::norm_raw::qwen_pair_square(first.data_ptr(), second.data_ptr(), squared.data_ptr<float>(), first_count, total_count, blocks, stream);
    auto mean = squared.mean(-1, true);
    auto first_output = ::tyloi::cuda::empty_like(first);
    auto second_output = ::tyloi::cuda::empty_like(second);
    tyloi::cuda::norm_raw::qwen_pair_finish(first.data_ptr(), second.data_ptr(), mean.data_ptr<float>(), first_weight.data_ptr<float>(), second_weight.data_ptr<float>(), first_output.data_ptr(), second_output.data_ptr(), first_count, total_count, D, blocks, float(eps), float(weight_offset), stream);
    return {first_output, second_output};
}

::tyloi::DenseTensor minicpm_qk_norm_rope_cache_write_bf16_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor q_weight, ::tyloi::DenseTensor k_weight,
    ::tyloi::DenseTensor rope_pos, ::tyloi::DenseTensor write_pos,
    ::tyloi::DenseTensor cos, ::tyloi::DenseTensor sin,
    ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    double eps, double weight_offset)
{
    const auto bf16 = ::tyloi::kBFloat16;
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() &&
                    q.is_contiguous() && k.is_contiguous() && v.is_contiguous() &&
                    q.scalar_type() == bf16 && k.scalar_type() == bf16 &&
                    v.scalar_type() == bf16,
                "minicpm_qk_norm_rope_kv: q/k/v must be cuda contiguous bf16");
    TYLOINFER_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4 &&
                    q.size(0) == k.size(0) && q.size(1) == 32 &&
                    k.size(1) == 8 && q.size(2) == 1 && k.size(2) == 1 &&
                    q.size(3) == 128 && k.size(3) == 128 && k.sizes() == v.sizes(),
                "minicpm_qk_norm_rope_kv: expected Bx32x1x128 Q and Bx8x1x128 K/V");
    TYLOINFER_RUNTIME_CHECK(q_weight.is_cuda() && k_weight.is_cuda() &&
                    q_weight.is_contiguous() && k_weight.is_contiguous() &&
                    q_weight.scalar_type() == ::tyloi::kFloat32 &&
                    k_weight.scalar_type() == ::tyloi::kFloat32 &&
                    q_weight.numel() == 128 && k_weight.numel() == 128,
                "minicpm_qk_norm_rope_kv: norm weights must be cuda contiguous f32[128]");
    TYLOINFER_RUNTIME_CHECK(k_cache.is_cuda() && v_cache.is_cuda() &&
                    k_cache.is_contiguous() && v_cache.is_contiguous() &&
                    k_cache.scalar_type() == bf16 && v_cache.scalar_type() == bf16 &&
                    k_cache.sizes() == v_cache.sizes() && k_cache.dim() == 4 &&
                    k_cache.size(0) == k.size(0) && k_cache.size(1) == 8 &&
                    k_cache.size(3) == 128,
                "minicpm_qk_norm_rope_kv: cache shape mismatch");
    TYLOINFER_RUNTIME_CHECK(rope_pos.is_cuda() && write_pos.is_cuda() &&
                    rope_pos.is_contiguous() && write_pos.is_contiguous() &&
                    rope_pos.scalar_type() == ::tyloi::kInt64 &&
                    write_pos.scalar_type() == ::tyloi::kInt64 &&
                    rope_pos.numel() == 1 && write_pos.numel() == 1,
                "minicpm_qk_norm_rope_kv: positions must be cuda contiguous int64[1]");
    TYLOINFER_RUNTIME_CHECK(cos.is_cuda() && sin.is_cuda() && cos.is_contiguous() &&
                    sin.is_contiguous() && cos.scalar_type() == ::tyloi::kFloat32 &&
                    sin.scalar_type() == ::tyloi::kFloat32 && cos.sizes() == sin.sizes() &&
                    cos.dim() == 2 && cos.size(1) == 64,
                "minicpm_qk_norm_rope_kv: cos/sin must be matching f32 tables");

    auto stream = mfq_current_cuda_stream();
    auto q_out = ::tyloi::cuda::empty_like(q);
    const int B = (int)q.size(0);
    tyloi::cuda::norm_raw::minicpm_qk(q.data_ptr(), k.data_ptr(), v.data_ptr(), q_weight.data_ptr<float>(), k_weight.data_ptr<float>(), rope_pos.data_ptr<int64_t>(), write_pos.data_ptr<int64_t>(), cos.data_ptr<float>(), sin.data_ptr<float>(), q_out.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(), B, int(cos.size(0)), int(k_cache.size(2)), float(eps), float(weight_offset), stream);
    return q_out;
}

std::vector<::tyloi::DenseTensor> rms_norm_pair_f16_f32_offset_cuda(
    ::tyloi::DenseTensor first,
    ::tyloi::DenseTensor second,
    ::tyloi::DenseTensor first_weight,
    ::tyloi::DenseTensor second_weight,
    double eps,
    double weight_offset)
{
    TYLOINFER_RUNTIME_CHECK(
        first.is_cuda() && first.is_contiguous() &&
        first.scalar_type() == ::tyloi::kFloat16,
        "paired RMSNorm first input must be CUDA contiguous fp16");
    TYLOINFER_RUNTIME_CHECK(
        second.is_cuda() && second.is_contiguous() &&
        second.scalar_type() == ::tyloi::kFloat16,
        "paired RMSNorm second input must be CUDA contiguous fp16");
    TYLOINFER_RUNTIME_CHECK(
        first_weight.is_cuda() && first_weight.is_contiguous() &&
        first_weight.scalar_type() == ::tyloi::kFloat32,
        "paired RMSNorm first weight must be CUDA contiguous fp32");
    TYLOINFER_RUNTIME_CHECK(
        second_weight.is_cuda() && second_weight.is_contiguous() &&
        second_weight.scalar_type() == ::tyloi::kFloat32,
        "paired RMSNorm second weight must be CUDA contiguous fp32");
    const int D = (int)first.size(-1);
    TYLOINFER_RUNTIME_CHECK(second.size(-1) == D, "paired RMSNorm widths must match");
    TYLOINFER_RUNTIME_CHECK(
        first_weight.numel() == D && second_weight.numel() == D,
        "paired RMSNorm weight lengths must match the input width");
    const int first_rows = (int)(first.numel() / D);
    const int second_rows = (int)(second.numel() / D);
    auto output_options = first.options().dtype(::tyloi::kFloat32);
    auto first_out = ::tyloi::cuda::empty(first.sizes(), output_options);
    auto second_out = ::tyloi::cuda::empty(second.sizes(), output_options);
    tyloi::cuda::norm_raw::rms_pair_f16_f32(first.data_ptr(), second.data_ptr(), first_weight.data_ptr<float>(), second_weight.data_ptr<float>(), first_out.data_ptr<float>(), second_out.data_ptr<float>(), first_rows, second_rows, D, float(eps), float(weight_offset), mfq_current_cuda_stream());
    return {first_out, second_out};
}

::tyloi::DenseTensor l2_norm_cuda(::tyloi::DenseTensor x, double eps)
{
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == ::tyloi::kFloat32,
                "l2_norm: x must be cuda contiguous f32");
    int D = (int)x.size(-1);
    int N = (int)(x.numel() / D);
    auto out = ::tyloi::cuda::empty_like(x);
    tyloi::cuda::norm_raw::l2(x.data_ptr<float>(), out.data_ptr<float>(), N, D, float(eps), mfq_current_cuda_stream());
    return out;
}
