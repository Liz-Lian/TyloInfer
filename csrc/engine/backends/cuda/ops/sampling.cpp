#include "backend.h"
#include "env.h"
#include "sampling_raw.h"
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <utility>

constexpr int SAMPLE_BD = 256;
constexpr int SAMPLE_GREEDY_CHUNK = 4096;
constexpr int SAMPLE_MAX_TOP_K = 1024;
constexpr int SAMPLE_RADIX_MAX_TOP_K = 64;

static tyloi::cuda::NumericType sample_type(const ::tyloi::DenseTensor& x) {
    return x.scalar_type() == ::tyloi::kFloat16 ? tyloi::cuda::NumericType::f16 :
        x.scalar_type() == ::tyloi::kBFloat16 ? tyloi::cuda::NumericType::bf16 : tyloi::cuda::NumericType::f32;
}
::tyloi::DenseTensor sample_greedy_cuda(::tyloi::DenseTensor logits)
{
    TYLOINFER_RUNTIME_CHECK(logits.is_cuda() && logits.is_contiguous(), "sample_greedy: logits must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(logits.dim() == 2, "sample_greedy: logits must be [B,V]");
    TYLOINFER_RUNTIME_CHECK(
        logits.scalar_type() == ::tyloi::kFloat32 ||
        logits.scalar_type() == ::tyloi::kFloat16 ||
        logits.scalar_type() == ::tyloi::kBFloat16,
        "sample_greedy: logits dtype must be f32, f16, or bf16");
    int B = (int)logits.size(0);
    int V = (int)logits.size(1);
    auto out = ::tyloi::cuda::empty({B}, logits.options().dtype(::tyloi::kInt64));
    ::tyloi::DenseTensor values, indices;
    if (V >= 16384 && B <= 65535) {
        const int chunks = (V + SAMPLE_GREEDY_CHUNK - 1) / SAMPLE_GREEDY_CHUNK;
        values = ::tyloi::cuda::empty({B, chunks}, logits.options().dtype(::tyloi::kFloat32));
        indices = ::tyloi::cuda::empty({B, chunks}, logits.options().dtype(::tyloi::kInt64));
    }
    tyloi::cuda::sampling_raw::greedy(logits.data_ptr(), out.data_ptr<int64_t>(),
        values.defined() ? values.data_ptr<float>() : nullptr,
        indices.defined() ? indices.data_ptr<int64_t>() : nullptr, B, V, sample_type(logits), mfq_current_cuda_stream());
    return out;
}

::tyloi::DenseTensor sample_softmax_cuda(::tyloi::DenseTensor logits, ::tyloi::DenseTensor random, double temperature)
{
    TYLOINFER_RUNTIME_CHECK(logits.is_cuda() && logits.is_contiguous(), "sample_softmax: logits must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(random.is_cuda() && random.is_contiguous() && random.scalar_type() == ::tyloi::kFloat32,
                "sample_softmax: random must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(logits.dim() == 2, "sample_softmax: logits must be [B,V]");
    TYLOINFER_RUNTIME_CHECK(
        logits.scalar_type() == ::tyloi::kFloat32 ||
        logits.scalar_type() == ::tyloi::kFloat16 ||
        logits.scalar_type() == ::tyloi::kBFloat16,
        "sample_softmax: logits dtype must be f32, f16, or bf16");
    int B = (int)logits.size(0);
    int V = (int)logits.size(1);
    TYLOINFER_RUNTIME_CHECK(random.numel() == B, "sample_softmax: random length must match B");
    TYLOINFER_RUNTIME_CHECK(temperature > 0.0, "sample_softmax: temperature must be > 0");
    auto out = ::tyloi::cuda::empty({B}, logits.options().dtype(::tyloi::kInt64));
    tyloi::cuda::sampling_raw::softmax(logits.data_ptr(), random.data_ptr<float>(), out.data_ptr<int64_t>(),
        B, V, float(temperature), sample_type(logits), mfq_current_cuda_stream());
    return out;
}

template<bool Conditional>
static ::tyloi::DenseTensor sample_top_k_compact_impl(
    ::tyloi::DenseTensor logits,int64_t top_k,const volatile int32_t* enabled) {
    TYLOINFER_RUNTIME_CHECK(logits.is_cuda() && logits.is_contiguous() && logits.dim()==2,
        "sample_top_k_compact: logits must be contiguous CUDA [B,V]");
    TYLOINFER_RUNTIME_CHECK(logits.scalar_type()==::tyloi::kFloat32 ||
        logits.scalar_type()==::tyloi::kFloat16 || logits.scalar_type()==::tyloi::kBFloat16,
        "sample_top_k_compact: unsupported logit dtype");
    const int B=int(logits.size(0)),V=int(logits.size(1));
    TYLOINFER_RUNTIME_CHECK(B>0 && B<=32 && logits.size(1)<=INT_MAX && top_k>0 && top_k<=V && top_k<=64,
        "sample_top_k_compact: unsupported rows, vocab or top-k");
    const int items=V<=8192?1:V<=65536?4:8;
    const int blocks=(V+SAMPLE_BD*items-1)/(SAMPLE_BD*items);
    // Explicit types keep these non-dependent for nvcc's MSVC-mode front end.
    ::tyloi::DenseTensor out=::tyloi::cuda::empty({B,top_k},logits.options().dtype(::tyloi::kInt64));
    ::tyloi::DenseTensor a=::tyloi::cuda::empty({int64_t(B)*blocks*top_k},out.options());
    ::tyloi::DenseTensor b=::tyloi::cuda::empty(a.sizes(),a.options());
    tyloi::cuda::sampling_raw::compact(logits.data_ptr(), out.data_ptr<int64_t>(), a.data_ptr<int64_t>(),
        b.data_ptr<int64_t>(), B, V, int(top_k), Conditional ? enabled : nullptr, sample_type(logits), mfq_current_cuda_stream());
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return out;
}

::tyloi::DenseTensor sample_top_k_compact_cuda(::tyloi::DenseTensor logits,int64_t top_k) {
    return sample_top_k_compact_impl<false>(std::move(logits),top_k,nullptr);
}
::tyloi::DenseTensor sample_top_k_compact_if_cuda(
        ::tyloi::DenseTensor logits,int64_t top_k,const volatile int32_t* enabled) {
    TYLOINFER_RUNTIME_CHECK(enabled,"sample_top_k_compact_if: null replay flag");
    return sample_top_k_compact_impl<true>(std::move(logits),top_k,enabled);
}

::tyloi::DenseTensor sample_top_k_top_p_cuda(
    ::tyloi::DenseTensor logits,
    ::tyloi::DenseTensor random,
    double temperature,
    int64_t top_k,
    double top_p)
{
    TYLOINFER_RUNTIME_CHECK(logits.is_cuda() && logits.is_contiguous(), "sample_top_k_top_p: logits must be cuda contiguous");
    TYLOINFER_RUNTIME_CHECK(random.is_cuda() && random.is_contiguous() && random.scalar_type() == ::tyloi::kFloat32,
                "sample_top_k_top_p: random must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(logits.dim() == 2, "sample_top_k_top_p: logits must be [B,V]");
    TYLOINFER_RUNTIME_CHECK(
        logits.scalar_type() == ::tyloi::kFloat32 ||
        logits.scalar_type() == ::tyloi::kFloat16 ||
        logits.scalar_type() == ::tyloi::kBFloat16,
        "sample_top_k_top_p: logits dtype must be f32, f16, or bf16");
    int B = (int)logits.size(0);
    int V = (int)logits.size(1);
    TYLOINFER_RUNTIME_CHECK(random.numel() == B, "sample_top_k_top_p: random length must match B");
    TYLOINFER_RUNTIME_CHECK(temperature > 0.0, "sample_top_k_top_p: temperature must be > 0");
    TYLOINFER_RUNTIME_CHECK(top_k > 0 && top_k <= V && top_k <= SAMPLE_MAX_TOP_K,
                "sample_top_k_top_p: top_k must be in [1, min(V,1024)]");
    auto out = ::tyloi::cuda::empty({B}, logits.options().dtype(::tyloi::kInt64));

    const char* radix_env = tyloi::cuda::env_lookup("TYLOINFER_SAMPLE_RADIX_TOPK");
    const bool radix = B == 1 && top_k <= SAMPLE_RADIX_MAX_TOP_K && (!radix_env || radix_env[0] != '0');
    const char* key_env = radix ? tyloi::cuda::env_lookup("TYLOINFER_SAMPLE_RADIX_KEYS") : nullptr;
    const bool keys = radix && (!key_env || key_env[0] != '0');
    const char* item_env = keys ? tyloi::cuda::env_lookup("TYLOINFER_SAMPLE_RADIX_ITEMS") : nullptr;
    const int requested = item_env ? std::atoi(item_env) : (V <= 8192 ? 1 : V <= 65536 ? 4 : 8);
    const int items = requested == 8 ? 8 : requested == 1 ? 1 : 4;
    const char* bitonic_env = keys ? tyloi::cuda::env_lookup("TYLOINFER_SAMPLE_KEY_BITONIC") : nullptr;
    const bool bitonic = !bitonic_env || bitonic_env[0] != '0';
    ::tyloi::DenseTensor a, b, c, d;
    if (radix) {
        const int blocks = keys ? (V + SAMPLE_BD * items - 1) / (SAMPLE_BD * items) : (V + SAMPLE_BD - 1) / SAMPLE_BD;
        const int64_t capacity = int64_t(blocks) * top_k;
        a = ::tyloi::cuda::empty({capacity}, logits.options().dtype(keys ? ::tyloi::kInt64 : ::tyloi::kFloat32));
        b = ::tyloi::cuda::empty({capacity}, logits.options().dtype(keys ? ::tyloi::kInt64 : ::tyloi::kInt32));
        if (!keys) {
            c = ::tyloi::cuda::empty({capacity}, a.options());
            d = ::tyloi::cuda::empty({capacity}, b.options());
        }
    }
    tyloi::cuda::sampling_raw::topk(logits.data_ptr(), random.data_ptr<float>(), out.data_ptr<int64_t>(),
        a.defined() ? a.data_ptr() : nullptr, b.defined() ? b.data_ptr() : nullptr,
        c.defined() ? c.data_ptr() : nullptr, d.defined() ? d.data_ptr() : nullptr,
        B, V, int(top_k), float(temperature), float(top_p), radix, keys, items, bitonic, sample_type(logits), mfq_current_cuda_stream());
    return out;
}

void sample_token_counts_add_cuda(::tyloi::DenseTensor counts, ::tyloi::DenseTensor tokens)
{
    TYLOINFER_RUNTIME_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == ::tyloi::kInt32,
                "sample_token_counts_add: counts must be contiguous CUDA int32");
    TYLOINFER_RUNTIME_CHECK(tokens.is_cuda() && tokens.is_contiguous() && tokens.scalar_type() == ::tyloi::kInt64,
                "sample_token_counts_add: tokens must be contiguous CUDA int64");
    TYLOINFER_RUNTIME_CHECK(counts.dim() == 1, "sample_token_counts_add: counts must be [V]");
    int64_t n = tokens.numel();
    if (n == 0) return;
    tyloi::cuda::sampling_raw::counts(counts.data_ptr<int32_t>(), tokens.data_ptr<int64_t>(), n,
        int(counts.numel()), mfq_current_cuda_stream());
}

::tyloi::DenseTensor sample_apply_penalties_cuda(
    ::tyloi::DenseTensor logits,
    ::tyloi::DenseTensor counts,
    double presence_penalty,
    double frequency_penalty,
    double repetition_penalty)
{
    TYLOINFER_RUNTIME_CHECK(logits.is_cuda() && logits.is_contiguous(),
                "sample_apply_penalties: logits must be contiguous CUDA");
    TYLOINFER_RUNTIME_CHECK(logits.dim() == 2 && logits.size(0) == 1,
                "sample_apply_penalties: logits must be [1,V]");
    TYLOINFER_RUNTIME_CHECK(
        logits.scalar_type() == ::tyloi::kFloat32 ||
        logits.scalar_type() == ::tyloi::kFloat16 ||
        logits.scalar_type() == ::tyloi::kBFloat16,
        "sample_apply_penalties: logits dtype must be f32, f16, or bf16");
    TYLOINFER_RUNTIME_CHECK(counts.is_cuda() && counts.is_contiguous() && counts.scalar_type() == ::tyloi::kInt32,
                "sample_apply_penalties: counts must be contiguous CUDA int32");
    TYLOINFER_RUNTIME_CHECK(counts.dim() == 1 && counts.numel() == logits.size(1),
                "sample_apply_penalties: counts must be [V]");
    int vocab_size = (int)logits.size(1);
    tyloi::cuda::sampling_raw::penalties(logits.data_ptr(), counts.data_ptr<int32_t>(), vocab_size,
        float(presence_penalty), float(frequency_penalty), float(repetition_penalty), sample_type(logits), mfq_current_cuda_stream());
    return logits;
}
