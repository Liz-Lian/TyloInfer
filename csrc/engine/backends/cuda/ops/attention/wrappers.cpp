#include "backend.h"
#include "env.h"
#include "attention_raw.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>

static tyloi::cuda::NumericType attention_type(const ::tyloi::DenseTensor& q) {
    if(q.scalar_type()==::tyloi::kFloat16) return tyloi::cuda::NumericType::f16;
    if(q.scalar_type()==::tyloi::kBFloat16) return tyloi::cuda::NumericType::bf16;
    return tyloi::cuda::NumericType::f32;
}
::tyloi::DenseTensor attention_flash256_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v, double scale)
{
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == ::tyloi::kFloat16,
                "attention_flash256: q must be contiguous CUDA fp16");
    TYLOINFER_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == ::tyloi::kFloat16,
                "attention_flash256: k must be contiguous CUDA fp16");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == ::tyloi::kFloat16,
                "attention_flash256: v must be contiguous CUDA fp16");
    TYLOINFER_RUNTIME_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "attention_flash256: q/k/v must be rank 4");
    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int Hk = (int)k.size(1);
    TYLOINFER_RUNTIME_CHECK(q.size(3) == 256 && k.size(3) == 256 && v.size(3) == 256,
                "attention_flash256: head_dim must be 256");
    TYLOINFER_RUNTIME_CHECK(k.size(0) == B && v.size(0) == B && k.size(2) == T && v.size(2) == T && v.size(1) == Hk,
                "attention_flash256: only self-attention with matching k/v shapes is supported");
    TYLOINFER_RUNTIME_CHECK(Hq % Hk == 0, "attention_flash256: Hq must be divisible by Hkv");
    auto out = ::tyloi::cuda::empty_like(q);
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k.data_ptr(); a.v=v.data_ptr(); a.out=out.data_ptr();
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.tokens=T; a.scale=float(scale);
    tyloi::cuda::attention_raw::flash256(a,mfq_current_cuda_stream());
    return out;
}

static ::tyloi::DenseTensor attention_impl_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
                                         double scale, bool causal, int window)
{
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention: q must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous(), "attention: k must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous(), "attention: v must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "attention: q/k/v dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == ::tyloi::kFloat16 || q.scalar_type() == ::tyloi::kFloat32,
                "attention: dtype must be f16 or f32");
    int B = (int)q.size(0), Hq = (int)q.size(1), T = (int)q.size(2), D = (int)q.size(3);
    int Hk = (int)k.size(1), Tk = (int)k.size(2);
    TYLOINFER_RUNTIME_CHECK(D == (int)k.size(3) && D == (int)v.size(3), "q/k/v last dim must match");
    TYLOINFER_RUNTIME_CHECK(Hq % Hk == 0, "GQA requires H_q % H_kv == 0");
    auto out = ::tyloi::cuda::empty_like(q);
    int total = B * Hq * T;
    int parts = 1;
    const int visible_keys = window > 0 ? std::min(Tk, window) : Tk;
    const char* split_env = tyloi::cuda::env_lookup("TYLOINFER_ATTENTION_SPLITK");
    if (total < 128 && visible_keys >= 512 && !(split_env && split_env[0] == '0')) {
        parts = (visible_keys + 255) / 256;
        parts = parts < 2 ? 1 : parts;
        parts = parts > 16 ? 16 : parts;
    }
    TYLOINFER_RUNTIME_CHECK(D<=512, "attention: D>512 unsupported, got ", D);
    ::tyloi::DenseTensor po,pm,pl;
    if(parts>1) {
        po=::tyloi::cuda::empty({total,parts,D},q.options().dtype(::tyloi::kFloat32));
        pm=::tyloi::cuda::empty({total,parts},q.options().dtype(::tyloi::kFloat32));
        pl=::tyloi::cuda::empty({total,parts},q.options().dtype(::tyloi::kFloat32));
    }
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k.data_ptr(); a.v=v.data_ptr(); a.out=out.data_ptr();
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.tokens=T; a.keys=Tk; a.width=D;
    a.scale=float(scale); a.causal=causal; a.window=window; a.parts=parts; a.type=attention_type(q);
    a.partial_o=po.defined()?po.data_ptr<float>():nullptr; a.partial_m=pm.defined()?pm.data_ptr<float>():nullptr; a.partial_l=pl.defined()?pl.data_ptr<float>():nullptr;
    tyloi::cuda::attention_raw::full(a, mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor attention_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
                             double scale, bool causal)
{
    return attention_impl_cuda(q, k, v, scale, causal, 0);
}

::tyloi::DenseTensor attention_swa_cuda(::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
                                 double scale, int64_t window)
{
    TYLOINFER_RUNTIME_CHECK(window > 0 && window <= INT_MAX, "attention_swa: window must be in [1, INT_MAX]");
    return attention_impl_cuda(q, k, v, scale, true, (int)window);
}

static ::tyloi::DenseTensor attention_cache_decode_impl_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache,
    ::tyloi::DenseTensor v_cache, ::tyloi::DenseTensor seq_len,
    double scale, int maximum_cache_position)
{
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_decode: q must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "attention_cache_decode: k_cache must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "attention_cache_decode: v_cache must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == ::tyloi::kInt64,
                "attention_cache_decode: seq_len must be cuda int64[B]");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_decode: q/k/v dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == ::tyloi::kFloat16 || q.scalar_type() == ::tyloi::kBFloat16 ||
                q.scalar_type() == ::tyloi::kFloat32,
                "attention_cache_decode: dtype must be f16, bf16, or f32");
    TYLOINFER_RUNTIME_CHECK(q.dim() == 4 && q.size(2) == 1, "attention_cache_decode: q must be [B,Hq,1,D]");
    TYLOINFER_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "attention_cache_decode: caches must be [B,Hk,max_seq,D]");
    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int D = (int)q.size(3);
    int Hk = (int)k_cache.size(1);
    int max_seq = (int)k_cache.size(2);
    TYLOINFER_RUNTIME_CHECK(seq_len.numel() == B,
                "attention_cache_decode: seq_len must contain one value per batch row");
    TYLOINFER_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_decode: cache shape mismatch");
    TYLOINFER_RUNTIME_CHECK(Hq % Hk == 0, "attention_cache_decode: GQA requires Hq % Hk == 0");
    auto out = ::tyloi::cuda::empty_like(q);
    TYLOINFER_RUNTIME_CHECK(D<=512, "attention_cache_decode: D>512 unsupported, got ", D);
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k_cache.data_ptr(); a.v=v_cache.data_ptr(); a.out=out.data_ptr(); a.lengths=seq_len.data_ptr<int64_t>();
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.keys=max_seq; a.width=D; a.scale=float(scale); a.maximum_cache_position=maximum_cache_position; a.type=attention_type(q);
    tyloi::cuda::attention_raw::decode(a,mfq_current_cuda_stream());
    return out;
}

static ::tyloi::DenseTensor attention_cache_decode_split_impl_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale,
    ::tyloi::DenseTensor partial_o, ::tyloi::DenseTensor partial_m, ::tyloi::DenseTensor partial_l,
    int64_t parts, bool dynamic_parts)
{
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_decode_split: q must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "attention_cache_decode_split: k_cache must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "attention_cache_decode_split: v_cache must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == ::tyloi::kInt64,
                "attention_cache_decode_split: seq_len must be cuda int64[B]");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_decode_split: q/k/v dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == ::tyloi::kFloat16 || q.scalar_type() == ::tyloi::kBFloat16 ||
                q.scalar_type() == ::tyloi::kFloat32,
                "attention_cache_decode_split: dtype must be f16, bf16, or f32");
    TYLOINFER_RUNTIME_CHECK(q.dim() == 4 && q.size(2) == 1, "attention_cache_decode_split: q must be [B,Hq,1,D]");
    TYLOINFER_RUNTIME_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "attention_cache_decode_split: caches must be [B,Hk,max_seq,D]");
    TYLOINFER_RUNTIME_CHECK(partial_o.is_cuda() && partial_o.is_contiguous() && partial_o.scalar_type() == ::tyloi::kFloat32,
                "attention_cache_decode_split: partial_o must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(partial_m.is_cuda() && partial_m.is_contiguous() && partial_m.scalar_type() == ::tyloi::kFloat32 &&
                partial_l.is_cuda() && partial_l.is_contiguous() && partial_l.scalar_type() == ::tyloi::kFloat32,
                "attention_cache_decode_split: partial_m/l must be contiguous CUDA f32");

    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int D = (int)q.size(3);
    const int Hk = (int)k_cache.size(1);
    const int max_seq = (int)k_cache.size(2);
    const int total = B * Hq;
    TYLOINFER_RUNTIME_CHECK(seq_len.numel() == B,
                "attention_cache_decode_split: seq_len must contain one value per batch row");
    TYLOINFER_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_decode_split: cache shape mismatch");
    TYLOINFER_RUNTIME_CHECK(Hq % Hk == 0, "attention_cache_decode_split: GQA requires Hq % Hk == 0");
    TYLOINFER_RUNTIME_CHECK(partial_o.dim() == 3 && partial_o.size(0) == total && partial_o.size(2) == D,
                "attention_cache_decode_split: partial_o shape mismatch");
    const int workspace_parts = (int)partial_o.size(1);
    TYLOINFER_RUNTIME_CHECK(parts >= 2 && parts <= workspace_parts, "attention_cache_decode_split: invalid parts");
    TYLOINFER_RUNTIME_CHECK(partial_m.sizes() == ::tyloi::IntArrayRef({total, workspace_parts}) &&
                partial_l.sizes() == partial_m.sizes(), "attention_cache_decode_split: partial_m/l shape mismatch");

    auto out = ::tyloi::cuda::empty_like(q);
    TYLOINFER_RUNTIME_CHECK(D<=512, "attention_cache_decode_split: D>512 unsupported, got ", D);
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k_cache.data_ptr(); a.v=v_cache.data_ptr(); a.out=out.data_ptr(); a.lengths=seq_len.data_ptr<int64_t>();
    a.partial_o=partial_o.data_ptr<float>(); a.partial_m=partial_m.data_ptr<float>(); a.partial_l=partial_l.data_ptr<float>();
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.keys=max_seq; a.width=D; a.scale=float(scale); a.parts=int(parts); a.workspace_parts=workspace_parts; a.dynamic_parts=dynamic_parts; a.type=attention_type(q);
    tyloi::cuda::attention_raw::decode_split(a,mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor attention_cache_decode_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache,
    ::tyloi::DenseTensor v_cache, ::tyloi::DenseTensor seq_len,
    double scale)
{
    return attention_cache_decode_impl_cuda(
        std::move(q), std::move(k_cache), std::move(v_cache),
        std::move(seq_len), scale, -1);
}

::tyloi::DenseTensor attention_cache_decode_split_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache,
    ::tyloi::DenseTensor v_cache, ::tyloi::DenseTensor seq_len,
    double scale, ::tyloi::DenseTensor partial_o,
    ::tyloi::DenseTensor partial_m,
    ::tyloi::DenseTensor partial_l, int64_t parts)
{
    return attention_cache_decode_split_impl_cuda(
        std::move(q), std::move(k_cache), std::move(v_cache),
        std::move(seq_len), scale, std::move(partial_o),
        std::move(partial_m), std::move(partial_l), parts, false);
}

::tyloi::DenseTensor attention_cache_decode_dynamic_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache,
    ::tyloi::DenseTensor v_cache, ::tyloi::DenseTensor seq_len,
    double scale, ::tyloi::DenseTensor partial_o,
    ::tyloi::DenseTensor partial_m,
    ::tyloi::DenseTensor partial_l, int64_t max_parts)
{
    return attention_cache_decode_split_impl_cuda(
        std::move(q), std::move(k_cache), std::move(v_cache),
        std::move(seq_len), scale, std::move(partial_o),
        std::move(partial_m), std::move(partial_l), max_parts, true);
}

static ::tyloi::DenseTensor attention_cache_swa_impl_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t window, int64_t planned_length)
{
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_swa: q must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "attention_cache_swa: caches must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == ::tyloi::kInt64,
                "attention_cache_swa: seq_len must be cuda int64[B]");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_swa: q/k/v dtype mismatch");
    TYLOINFER_RUNTIME_CHECK(q.scalar_type() == ::tyloi::kFloat16 || q.scalar_type() == ::tyloi::kFloat32,
                "attention_cache_swa: dtype must be f16 or f32");
    TYLOINFER_RUNTIME_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4 &&
                k_cache.sizes() == v_cache.sizes(),
                "attention_cache_swa: expected q[B,Hq,T,D] and cache[B,Hk,capacity,D]");
    TYLOINFER_RUNTIME_CHECK(window > 0 && window <= INT_MAX, "attention_cache_swa: invalid window");

    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k_cache.size(1);
    const int capacity = (int)k_cache.size(2);
    TYLOINFER_RUNTIME_CHECK(seq_len.numel() == B,
                "attention_cache_swa: seq_len must contain one value per batch row");
    TYLOINFER_RUNTIME_CHECK(T > 0 && capacity >= window, "attention_cache_swa: cache capacity must cover the window");
    TYLOINFER_RUNTIME_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_swa: cache shape mismatch");
    TYLOINFER_RUNTIME_CHECK(Hq % Hk == 0, "attention_cache_swa: GQA requires Hq % Hk == 0");

    auto out = ::tyloi::cuda::empty_like(q);
    const int total = B * Hq * T;
    const int visible_keys = planned_length > 0 ? std::min<int64_t>(window, planned_length) : static_cast<int>(window);
    int parts = total < 128 && visible_keys >= 512 ? (visible_keys + 255) / 256 : 1;
    parts = std::max(1, std::min(parts, 16));
    TYLOINFER_RUNTIME_CHECK(D<=512, "attention_cache_swa: D>512 unsupported, got ", D);
    ::tyloi::DenseTensor po,pm,pl;
    if(parts>1) {
        po=::tyloi::cuda::empty({total,parts,D},q.options().dtype(::tyloi::kFloat32));
        pm=::tyloi::cuda::empty({total,parts},q.options().dtype(::tyloi::kFloat32));
        pl=::tyloi::cuda::empty({total,parts},q.options().dtype(::tyloi::kFloat32));
    }
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k_cache.data_ptr(); a.v=v_cache.data_ptr(); a.out=out.data_ptr(); a.lengths=seq_len.data_ptr<int64_t>();
    a.partial_o=po.defined()?po.data_ptr<float>():nullptr; a.partial_m=pm.defined()?pm.data_ptr<float>():nullptr; a.partial_l=pl.defined()?pl.data_ptr<float>():nullptr;
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.tokens=T; a.keys=capacity; a.width=D; a.scale=float(scale); a.parts=parts; a.window=int(window); a.type=attention_type(q);
    tyloi::cuda::attention_raw::cache_swa(a,mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor attention_cache_swa_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t window)
{
    return attention_cache_swa_impl_cuda(
        q, k_cache, v_cache, seq_len, scale, window, 0);
}

::tyloi::DenseTensor attention_cache_swa_planned_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k_cache, ::tyloi::DenseTensor v_cache,
    ::tyloi::DenseTensor seq_len, double scale, int64_t window, int64_t planned_length)
{
    TYLOINFER_RUNTIME_CHECK(planned_length > 0 && planned_length <= INT_MAX,
        "attention_cache_swa_planned: planned length must be in [1, INT_MAX]");
    return attention_cache_swa_impl_cuda(
        q, k_cache, v_cache, seq_len, scale, window, planned_length);
}
