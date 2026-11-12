#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "env.h"
#include "attention_raw.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>

static tyloi::cuda::NumericType attention_type(const torch::Tensor& q) {
    if(q.scalar_type()==at::kHalf) return tyloi::cuda::NumericType::f16;
    if(q.scalar_type()==at::kBFloat16) return tyloi::cuda::NumericType::bf16;
    return tyloi::cuda::NumericType::f32;
}
static torch::Tensor attention_impl_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                         double scale, bool causal, int window)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "attention: q must be cuda contiguous");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous(), "attention: k must be cuda contiguous");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous(), "attention: v must be cuda contiguous");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "attention: q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat,
                "attention: dtype must be f16 or f32");
    int B = (int)q.size(0), Hq = (int)q.size(1), T = (int)q.size(2), D = (int)q.size(3);
    int Hk = (int)k.size(1), Tk = (int)k.size(2);
    TORCH_CHECK(D == (int)k.size(3) && D == (int)v.size(3), "q/k/v last dim must match");
    TORCH_CHECK(Hq % Hk == 0, "GQA requires H_q % H_kv == 0");
    auto out = torch::empty_like(q);
    int total = B * Hq * T;
    int parts = 1;
    const int visible_keys = window > 0 ? std::min(Tk, window) : Tk;
    const char* split_env = tyloi::cuda::env_lookup("TYLOINFER_ATTENTION_SPLITK");
    if (total < 128 && visible_keys >= 512 && !(split_env && split_env[0] == '0')) {
        parts = (visible_keys + 255) / 256;
        parts = parts < 2 ? 1 : parts;
        parts = parts > 16 ? 16 : parts;
    }
    TORCH_CHECK(D<=512, "attention: D>512 unsupported, got ", D);
    torch::Tensor po,pm,pl;
    if(parts>1) {
        po=torch::empty({total,parts,D},q.options().dtype(at::kFloat));
        pm=torch::empty({total,parts},q.options().dtype(at::kFloat));
        pl=torch::empty({total,parts},q.options().dtype(at::kFloat));
    }
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k.data_ptr(); a.v=v.data_ptr(); a.out=out.data_ptr();
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.tokens=T; a.keys=Tk; a.width=D;
    a.scale=float(scale); a.causal=causal; a.window=window; a.parts=parts; a.type=attention_type(q);
    a.partial_o=po.defined()?po.data_ptr<float>():nullptr; a.partial_m=pm.defined()?pm.data_ptr<float>():nullptr; a.partial_l=pl.defined()?pl.data_ptr<float>():nullptr;
    tyloi::cuda::attention_raw::full(a, at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor attention_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                             double scale, bool causal)
{
    return attention_impl_cuda(q, k, v, scale, causal, 0);
}

torch::Tensor attention_swa_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                 double scale, int64_t window)
{
    TORCH_CHECK(window > 0 && window <= INT_MAX, "attention_swa: window must be in [1, INT_MAX]");
    return attention_impl_cuda(q, k, v, scale, true, (int)window);
}

static torch::Tensor attention_cache_decode_impl_cuda(
    torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor seq_len,
    double scale, int maximum_cache_position)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_decode: q must be cuda contiguous");
    TORCH_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "attention_cache_decode: k_cache must be cuda contiguous");
    TORCH_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "attention_cache_decode: v_cache must be cuda contiguous");
    TORCH_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == at::kLong,
                "attention_cache_decode: seq_len must be cuda int64[B]");
    TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_decode: q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16 ||
                q.scalar_type() == at::kFloat,
                "attention_cache_decode: dtype must be f16, bf16, or f32");
    TORCH_CHECK(q.dim() == 4 && q.size(2) == 1, "attention_cache_decode: q must be [B,Hq,1,D]");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "attention_cache_decode: caches must be [B,Hk,max_seq,D]");
    int B = (int)q.size(0);
    int Hq = (int)q.size(1);
    int D = (int)q.size(3);
    int Hk = (int)k_cache.size(1);
    int max_seq = (int)k_cache.size(2);
    TORCH_CHECK(seq_len.numel() == B,
                "attention_cache_decode: seq_len must contain one value per batch row");
    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_decode: cache shape mismatch");
    TORCH_CHECK(Hq % Hk == 0, "attention_cache_decode: GQA requires Hq % Hk == 0");
    auto out = torch::empty_like(q);
    TORCH_CHECK(D<=512, "attention_cache_decode: D>512 unsupported, got ", D);
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k_cache.data_ptr(); a.v=v_cache.data_ptr(); a.out=out.data_ptr(); a.lengths=seq_len.data_ptr<int64_t>();
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.keys=max_seq; a.width=D; a.scale=float(scale); a.maximum_cache_position=maximum_cache_position; a.type=attention_type(q);
    tyloi::cuda::attention_raw::decode(a,at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor attention_cache_decode_cuda(
    torch::Tensor q, torch::Tensor k_cache,
    torch::Tensor v_cache, torch::Tensor seq_len,
    double scale)
{
    return attention_cache_decode_impl_cuda(
        std::move(q), std::move(k_cache), std::move(v_cache),
        std::move(seq_len), scale, -1);
}

static torch::Tensor attention_cache_swa_impl_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t window, int64_t planned_length)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "attention_cache_swa: q must be cuda contiguous");
    TORCH_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "attention_cache_swa: caches must be cuda contiguous");
    TORCH_CHECK(seq_len.is_cuda() && seq_len.is_contiguous() &&
                seq_len.scalar_type() == at::kLong,
                "attention_cache_swa: seq_len must be cuda int64[B]");
    TORCH_CHECK(q.scalar_type() == k_cache.scalar_type() && q.scalar_type() == v_cache.scalar_type(),
                "attention_cache_swa: q/k/v dtype mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kFloat,
                "attention_cache_swa: dtype must be f16 or f32");
    TORCH_CHECK(q.dim() == 4 && k_cache.dim() == 4 && v_cache.dim() == 4 &&
                k_cache.sizes() == v_cache.sizes(),
                "attention_cache_swa: expected q[B,Hq,T,D] and cache[B,Hk,capacity,D]");
    TORCH_CHECK(window > 0 && window <= INT_MAX, "attention_cache_swa: invalid window");

    const int B = (int)q.size(0);
    const int Hq = (int)q.size(1);
    const int T = (int)q.size(2);
    const int D = (int)q.size(3);
    const int Hk = (int)k_cache.size(1);
    const int capacity = (int)k_cache.size(2);
    TORCH_CHECK(seq_len.numel() == B,
                "attention_cache_swa: seq_len must contain one value per batch row");
    TORCH_CHECK(T > 0 && capacity >= window, "attention_cache_swa: cache capacity must cover the window");
    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(3) == D, "attention_cache_swa: cache shape mismatch");
    TORCH_CHECK(Hq % Hk == 0, "attention_cache_swa: GQA requires Hq % Hk == 0");

    auto out = torch::empty_like(q);
    const int total = B * Hq * T;
    const int visible_keys = planned_length > 0 ? std::min<int64_t>(window, planned_length) : static_cast<int>(window);
    int parts = total < 128 && visible_keys >= 512 ? (visible_keys + 255) / 256 : 1;
    parts = std::max(1, std::min(parts, 16));
    TORCH_CHECK(D<=512, "attention_cache_swa: D>512 unsupported, got ", D);
    torch::Tensor po,pm,pl;
    if(parts>1) {
        po=torch::empty({total,parts,D},q.options().dtype(at::kFloat));
        pm=torch::empty({total,parts},q.options().dtype(at::kFloat));
        pl=torch::empty({total,parts},q.options().dtype(at::kFloat));
    }
    tyloi::cuda::attention_raw::Args a;
    a.q=q.data_ptr(); a.k=k_cache.data_ptr(); a.v=v_cache.data_ptr(); a.out=out.data_ptr(); a.lengths=seq_len.data_ptr<int64_t>();
    a.partial_o=po.defined()?po.data_ptr<float>():nullptr; a.partial_m=pm.defined()?pm.data_ptr<float>():nullptr; a.partial_l=pl.defined()?pl.data_ptr<float>():nullptr;
    a.batch=B; a.heads=Hq; a.kv_heads=Hk; a.tokens=T; a.keys=capacity; a.width=D; a.scale=float(scale); a.parts=parts; a.window=int(window); a.type=attention_type(q);
    tyloi::cuda::attention_raw::cache_swa(a,at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor attention_cache_swa_cuda(
    torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor seq_len, double scale, int64_t window)
{
    return attention_cache_swa_impl_cuda(
        q, k_cache, v_cache, seq_len, scale, window, 0);
}
