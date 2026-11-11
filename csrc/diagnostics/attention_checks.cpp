#include "model_checks.h"

#include "engine/backends/cuda/kernels/deepseek_v4_attention.h"
#include "engine/backends/cuda/kernels/deepseek_v4_hc.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"
#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int run_attention_decode_check(int length, int reps, int D, bool sliding, int window) {
    if (length < 1 || length > 262144) {
        throw std::runtime_error("--check-attention-decode must be in [1, 262144]");
    }
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 16;
    constexpr int max_parts = 256;
    if (D != 256 && D != 512) {
        throw std::runtime_error("--check-attention-head-dim must be 256 or 512");
    }
    if (sliding && D != 256) {
        throw std::runtime_error("SWA decode check currently requires head_dim 256");
    }
    const int Hk = sliding ? 8 : 2;
    const int kv_tile = D == 512 ? 32 : 64;
    const int visible_len = sliding ? std::min(length, window) : length;
    const int max_seq = sliding ? window : (length + kv_tile - 1) / kv_tile * kv_tile;
    const int parts = std::min(max_parts, std::max(1, (length + 127) / 128));
    auto cuda = tyloi::TensorOptions().device(tyloi::kCUDA);
    tyloi::cuda::manual_seed(20260720);
    auto q = tyloi::cuda::randn({B, Hq, 1, D}, cuda.dtype(tyloi::kFloat32));
    auto k = tyloi::cuda::randn({B, Hk, max_seq, D}, cuda.dtype(tyloi::kFloat16));
    auto v = tyloi::cuda::randn({B, Hk, max_seq, D}, cuda.dtype(tyloi::kFloat16));
    auto seq_len = tyloi::cuda::tensor({length}, cuda.dtype(tyloi::kInt64));
    auto partial_o = tyloi::cuda::empty({B * Hq, max_parts, D}, cuda.dtype(tyloi::kFloat32));
    auto partial_m = tyloi::cuda::empty({B * Hq, max_parts}, cuda.dtype(tyloi::kFloat32));
    auto partial_l = tyloi::cuda::empty({B * Hq, max_parts}, cuda.dtype(tyloi::kFloat32));
    auto qh = q.to(tyloi::kFloat16);
    const int mask_stride = (visible_len + kv_tile - 1) / kv_tile * kv_tile;
    const int ntiles_kv = (visible_len + kv_tile - 1) / kv_tile;
    const int64_t max_blocks = B * Hk * ntiles_kv;
    const int64_t meta_float2 = max_blocks * 8 * (2 + D / 2);
    auto mask = tyloi::cuda::empty({B, mask_stride}, cuda.dtype(tyloi::kFloat16));
    auto kv_max = tyloi::cuda::empty({B}, cuda.dtype(tyloi::kInt32));
    auto meta = tyloi::cuda::empty({2 * meta_float2}, cuda.dtype(tyloi::kFloat32));
    const double scale = 1.0 / std::sqrt((double)D);
    auto run_ref = [&]() {
        if (sliding) {
            return attention_cache_swa_planned_cuda(
                qh, k, v, seq_len, scale, window, visible_len);
        }
        return parts > 1
            ? attention_cache_decode_split_cuda(
                  qh, k, v, seq_len, scale, partial_o, partial_m, partial_l, parts)
            : attention_cache_decode_cuda(qh, k, v, seq_len, scale);
    };
    auto run_test = [&]() {
        if (sliding) {
            return mfq_attention_mma256_swa_decode_cuda(
                q, k, v, seq_len, scale, visible_len, mask, kv_max, meta);
        }
        return D == 512
            ? mfq_attention_mma512_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta)
            : mfq_attention_mma256_decode_cuda(
                q, k, v, seq_len, scale, length, mask, kv_max, meta);
    };
    tyloi::DenseTensor ref, test;
    for (int i = 0; i < 10; ++i) {
        ref = run_ref();
        test = run_test();
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        cudaEventRecord(start, stream);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    ref = run_ref().to(tyloi::kFloat32);
    test = run_test().permute({0, 2, 1, 3}).contiguous();
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    auto diff = (test - ref).abs();
    std::cout << "attention_decode_check mode=" << (sliding ? "swa" : "full")
              << " head_dim=" << D << " length=" << length
              << " parts=" << parts << " ref_ms=" << ref_ms
              << " test_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";
    std::cout << "attention_decode_rel="
              << ((test - ref).norm() / ref.norm()).item<float>() << "\n";
    std::cout << "attention_decode_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "attention_decode_max_abs=" << diff.max().item<float>() << "\n";
    std::cout << "attention_decode_test_finite="
              << (tyloi::cuda::isfinite(test).all().item<bool>() ? 1 : 0) << "\n";
    return 0;
}

int run_gemma4_swa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    constexpr int B = 1;
    constexpr int Hq = 32;
    constexpr int Hk = 16;
    constexpr int D = 256;
    const double scale = 1.0 / std::sqrt((double)D);
    auto cuda = tyloi::TensorOptions().device(tyloi::kCUDA);
    tyloi::cuda::manual_seed(20260721);

    struct Shape {
        int tokens;
        int window;
    };
    const Shape shapes[] = {
        {1, 1}, {17, 5}, {33, 17}, {73, 32}, {256, 128}, {1024, 1024},
    };
    double worst_rel = 0.0;
    double worst_mean_abs = 0.0;
    double worst_max_abs = 0.0;
    for (const auto shape : shapes) {
        auto q = tyloi::cuda::randn({B, Hq, shape.tokens, D}, cuda.dtype(tyloi::kFloat32));
        auto k = tyloi::cuda::randn({B, Hk, shape.tokens, D}, cuda.dtype(tyloi::kFloat16));
        auto v = tyloi::cuda::randn({B, Hk, shape.tokens, D}, cuda.dtype(tyloi::kFloat16));
        auto ref = attention_swa_cuda(
            q, k.to(tyloi::kFloat32), v.to(tyloi::kFloat32), scale, shape.window);
        auto test = mfq_attention_mma256_swa_cuda(q, k, v, scale, shape.window)
            .permute({0, 2, 1, 3}).contiguous();
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const double mean_abs = diff.mean().item<double>();
        const double max_abs = diff.max().item<double>();
        const bool finite = tyloi::cuda::isfinite(test).all().item<bool>();
        worst_rel = std::max(worst_rel, rel);
        worst_mean_abs = std::max(worst_mean_abs, mean_abs);
        worst_max_abs = std::max(worst_max_abs, max_abs);
        std::cout << "gemma4_swa_check tokens=" << shape.tokens
                  << " window=" << shape.window
                  << " rel=" << rel
                  << " mean_abs=" << mean_abs
                  << " max_abs=" << max_abs
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 SWA FlashAttention numerical check failed");
        }
    }

    struct FullAttentionShape {
        int tokens;
        int kv_heads;
    };
    for (const auto shape : {
            FullAttentionShape{33, 4}, FullAttentionShape{256, 4},
            FullAttentionShape{65, 2}, FullAttentionShape{256, 2}}) {
        constexpr int full_hq = 16;
        const int tokens = shape.tokens;
        const int full_hk = shape.kv_heads;
        auto q = tyloi::cuda::randn({B, full_hq, tokens, D}, cuda.dtype(tyloi::kFloat32));
        auto k = tyloi::cuda::randn({B, full_hk, tokens, D}, cuda.dtype(tyloi::kFloat16));
        auto v = tyloi::cuda::randn({B, full_hk, tokens, D}, cuda.dtype(tyloi::kFloat16));
        auto ref = attention_cuda(
            q, k.to(tyloi::kFloat32), v.to(tyloi::kFloat32), scale, true);
        auto test = mfq_attention_mma256_cuda(q, k, v, scale)
            .permute({0, 2, 1, 3}).contiguous();
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = tyloi::cuda::isfinite(test).all().item<bool>();
        std::cout << "flash256_full_check tokens=" << tokens
                  << " gqa=" << full_hq / full_hk
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {65, 256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 128;
        const double full_scale = 1.0 / std::sqrt((double)full_d);
        auto q = tyloi::cuda::randn(
            {B, full_hq, tokens, full_d},
            cuda.dtype(tyloi::kFloat32));
        auto k = tyloi::cuda::randn(
            {B, full_hk, tokens, full_d},
            cuda.dtype(tyloi::kFloat16));
        auto v = tyloi::cuda::randn(
            {B, full_hk, tokens, full_d},
            cuda.dtype(tyloi::kFloat16));
        auto ref = attention_cuda(
            q, k.to(tyloi::kFloat32),
            v.to(tyloi::kFloat32), full_scale, true);
        auto test = mfq_attention_mma128_cuda(q, k, v, full_scale)
            .permute({0, 2, 1, 3}).contiguous();
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = tyloi::cuda::isfinite(test).all().item<bool>();
        std::cout << "flash128_full_check tokens=" << tokens
                  << " gqa=" << full_hq / full_hk
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error(
                "D128 full FlashAttention numerical regression failed");
        }
    }

    for (const int tokens : {256}) {
        constexpr int full_hq = 16;
        constexpr int full_hk = 2;
        constexpr int full_d = 512;
        auto q = tyloi::cuda::randn({B, full_hq, tokens, full_d}, cuda.dtype(tyloi::kFloat32));
        auto k = tyloi::cuda::randn({B, full_hk, tokens, full_d}, cuda.dtype(tyloi::kFloat16));
        auto v = tyloi::cuda::randn({B, full_hk, tokens, full_d}, cuda.dtype(tyloi::kFloat16));
        auto ref = attention_cuda(
            q, k.to(tyloi::kFloat32), v.to(tyloi::kFloat32), 1.0, true);
        auto test = mfq_attention_mma512_cuda(q, k, v, 1.0)
            .permute({0, 2, 1, 3}).contiguous();
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto diff = (test - ref).abs();
        const double rel = ((test - ref).norm() / ref.norm()).item<double>();
        const bool finite = tyloi::cuda::isfinite(test).all().item<bool>();
        std::cout << "gemma4_flash512_check tokens=" << tokens
                  << " rel=" << rel
                  << " mean_abs=" << diff.mean().item<double>()
                  << " max_abs=" << diff.max().item<double>()
                  << " finite=" << (finite ? 1 : 0) << "\n";
        if (!finite || rel > 0.02) {
            throw std::runtime_error("Gemma4 D512 FlashAttention numerical check failed");
        }
    }

    constexpr int bench_tokens = 256;
    constexpr int bench_window = 128;
    auto q = tyloi::cuda::randn({B, Hq, bench_tokens, D}, cuda.dtype(tyloi::kFloat32));
    auto k = tyloi::cuda::randn({B, Hk, bench_tokens, D}, cuda.dtype(tyloi::kFloat16));
    auto v = tyloi::cuda::randn({B, Hk, bench_tokens, D}, cuda.dtype(tyloi::kFloat16));
    auto kf = k.to(tyloi::kFloat32);
    auto vf = v.to(tyloi::kFloat32);
    auto run_ref = [&]() { return attention_swa_cuda(q, kf, vf, scale, bench_window); };
    auto run_test = [&]() {
        return mfq_attention_mma256_swa_cuda(q, k, v, scale, bench_window);
    };
    for (int i = 0; i < 10; ++i) {
        run_ref();
        run_test();
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&start));
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(stop, stream));
        TYLOINFER_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(start));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    const float ref_ms = time_ms(run_ref);
    const float test_ms = time_ms(run_test);
    std::cout << "gemma4_swa_bench tokens=" << bench_tokens
              << " window=" << bench_window
              << " generic_ms=" << ref_ms
              << " mma_attention_ms=" << test_ms
              << " speedup=" << ref_ms / test_ms << "\n";

    std::cout << "gemma4_swa_worst_rel=" << worst_rel
              << " worst_mean_abs=" << worst_mean_abs
              << " worst_max_abs=" << worst_max_abs << "\n";
    return 0;
}

int run_glm_dsa_check(int reps) {
    if (reps < 1) throw std::runtime_error("--check-attention-reps must be positive");
    tyloi::cuda::NoGradGuard no_grad;
    auto cuda = tyloi::TensorOptions().device(tyloi::kCUDA);
    tyloi::cuda::manual_seed(20260722);

    {
        constexpr int B = 2, H = 3, T = 5, D = 128, RD = 64;
        auto x = tyloi::cuda::randn({B, H, T, D}, cuda.dtype(tyloi::kFloat32));
        auto pos = tyloi::cuda::arange(7, 7 + T, cuda.dtype(tyloi::kInt64));
        auto freq = tyloi::cuda::pow(
            tyloi::cuda::full({RD / 2}, 8000000.0, cuda.dtype(tyloi::kFloat32)),
            -tyloi::cuda::arange(0, RD, 2, cuda.dtype(tyloi::kFloat32)) / (double)RD);
        auto angle = tyloi::cuda::arange(32, cuda.dtype(tyloi::kFloat32)).unsqueeze(1) * freq.unsqueeze(0);
        auto cos = tyloi::cuda::cos(angle).contiguous();
        auto sin = tyloi::cuda::sin(angle).contiguous();
        auto test = glm_interleaved_rope_cuda(x, pos, cos, sin, RD);
        auto paired = x.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(0, RD)})
            .reshape({B, H, T, RD / 2, 2});
        auto c = cos.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto s = sin.index_select(0, pos).reshape({1, 1, T, RD / 2});
        auto rotated = tyloi::cuda::stack({
            paired.select(-1, 0) * c - paired.select(-1, 1) * s,
            paired.select(-1, 1) * c + paired.select(-1, 0) * s,
        }, -1).reshape({B, H, T, RD});
        auto ref = tyloi::cuda::cat({
            rotated, x.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(RD, D)})}, -1);
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_rope_max_abs=" << max_abs << "\n";
        if (max_abs > 2e-6) throw std::runtime_error("GLM interleaved RoPE numerical check failed");
    }

    {
        constexpr int ROWS = 17, D = 128;
        auto x = tyloi::cuda::randn({ROWS, D}, cuda.dtype(tyloi::kFloat16));
        auto weight = tyloi::cuda::randn({D}, cuda.dtype(tyloi::kFloat32));
        auto bias = tyloi::cuda::randn({D}, cuda.dtype(tyloi::kFloat32));
        auto test = glm_dsa_indexer_layer_norm_cuda(x, weight, bias, 1e-5);
        auto ref = tyloi::cuda::layer_norm(
            x.to(tyloi::kFloat32), {D}, weight, bias, 1e-5).to(tyloi::kFloat16);
        const double max_abs = (test.to(tyloi::kFloat32) - ref.to(tyloi::kFloat32))
            .abs().max().item<double>();
        std::cout << "glm_indexer_layer_norm_max_abs=" << max_abs << "\n";
        if (max_abs > 0.004) {
            throw std::runtime_error("GLM indexer LayerNorm numerical check failed");
        }
    }

    {
        constexpr int ROWS = 3, EXPERTS = 256, TOPK = 8;
        auto logits = tyloi::cuda::randn({ROWS, EXPERTS}, cuda.dtype(tyloi::kFloat32));
        auto bias = tyloi::cuda::randn({EXPERTS}, cuda.dtype(tyloi::kFloat32));
        auto selected = moe_topk_cuda(
            logits, TOPK, true, false, true, false, bias, 1e-20, 2.5);
        auto sigmoid = tyloi::cuda::sigmoid(logits);
        auto expected_topk = tyloi::cuda::topk(
            sigmoid + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = sigmoid.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 2.5;
        const bool ids_equal = selected[0].to(tyloi::kInt64).equal(expected_ids);
        const double max_abs = (selected[1] - expected_weights)
            .abs().max().item<double>();
        std::cout << "glm_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6) {
            throw std::runtime_error("GLM MoE routing numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 3, K = 2112, H = 32, D = 128;
        auto q = tyloi::cuda::randn({B, M, H, D}, cuda.dtype(tyloi::kFloat16));
        auto k = tyloi::cuda::randn({B, K, D}, cuda.dtype(tyloi::kFloat16));
        auto weights = tyloi::cuda::randn({B, M, H}, cuda.dtype(tyloi::kFloat32));
        const int offset = K - M;
        auto test = glm_dsa_indexer_scores_cuda(q, k, weights, offset, K);
        auto heads = tyloi::cuda::einsum(
            "bmhd,bkd->bmhk", {q.to(tyloi::kFloat32), k.to(tyloi::kFloat32)}) /
            std::sqrt(128.0);
        auto ref = (tyloi::cuda::relu(heads) * weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        auto key_pos = tyloi::cuda::arange(K, cuda.dtype(tyloi::kInt64)).reshape({1, 1, K});
        auto query_pos = tyloi::cuda::arange(offset, offset + M, cuda.dtype(tyloi::kInt64)).reshape({1, M, 1});
        ref = ref.masked_fill(key_pos > query_pos, -std::numeric_limits<float>::infinity());
        auto finite = tyloi::cuda::isfinite(ref);
        auto diff = tyloi::cuda::where(finite, (test - ref).abs(), tyloi::cuda::zeros_like(ref));
        const double rel = tyloi::cuda::where(finite, test - ref, tyloi::cuda::zeros_like(ref)).norm().item<double>() /
            std::max(tyloi::cuda::where(finite, ref, tyloi::cuda::zeros_like(ref)).norm().item<double>(), 1e-30);
        const double max_abs = diff.max().item<double>();
        std::cout << "glm_indexer_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!tyloi::cuda::isneginf(test.masked_select(~finite)).all().item<bool>() || rel > 0.003) {
            throw std::runtime_error("GLM indexer score numerical check failed");
        }
    }

    {
        constexpr int B = 1, M = 1, VISIBLE = 2112, PLANNED = 2176;
        constexpr int H = 32, D = 128;
        auto q = tyloi::cuda::randn({B, M, H, D}, cuda.dtype(tyloi::kFloat16));
        auto k = tyloi::cuda::randn({B, PLANNED, D}, cuda.dtype(tyloi::kFloat16));
        auto weights = tyloi::cuda::randn({B, M, H}, cuda.dtype(tyloi::kFloat32));
        auto seq_len = tyloi::cuda::tensor({VISIBLE}, cuda.dtype(tyloi::kInt64));
        auto test = glm_dsa_indexer_scores_decode_cuda(
            q, k, weights, seq_len, PLANNED);
        auto ref = (tyloi::cuda::relu(tyloi::cuda::einsum(
            "bmhd,bkd->bmhk",
            {q.to(tyloi::kFloat32), k.to(tyloi::kFloat32)}) / std::sqrt(128.0)) *
            weights.unsqueeze(-1)).sum(2) / std::sqrt(32.0);
        ref.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(VISIBLE, PLANNED)})
            .fill_(-std::numeric_limits<float>::infinity());
        auto finite = tyloi::cuda::isfinite(ref);
        const double rel = tyloi::cuda::where(
            finite, test - ref, tyloi::cuda::zeros_like(ref)).norm().item<double>() /
            std::max(tyloi::cuda::where(
                finite, ref, tyloi::cuda::zeros_like(ref)).norm().item<double>(), 1e-30);
        const bool future_inf = tyloi::cuda::isneginf(
            test.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(VISIBLE, PLANNED)})).all().item<bool>();
        std::cout << "glm_indexer_decode_rel=" << rel
                  << " future_inf=" << (future_inf ? 1 : 0) << "\n";
        if (rel > 0.003 || !future_inf) {
            throw std::runtime_error("GLM decode indexer numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, T = 33, DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = tyloi::cuda::randn({B, H, T, DQ}, cuda.dtype(tyloi::kFloat32));
        auto kv = tyloi::cuda::randn({B, 1, T, DQ}, cuda.dtype(tyloi::kFloat16));
        auto kv_cache = tyloi::cuda::zeros({B, 1, 64, DQ}, cuda.dtype(tyloi::kFloat16));
        kv_cache.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(0, T), tyloi::indexing::Slice()}).copy_(kv);
        auto mask = tyloi::cuda::empty({T, 64}, cuda.dtype(tyloi::kFloat16));
        auto kv_max = tyloi::cuda::empty({B * ((T + 3) / 4)}, cuda.dtype(tyloi::kInt32));
        auto meta = tyloi::cuda::empty({8 * 1024 * 1024}, cuda.dtype(tyloi::kFloat32));
        auto test = attention_glm_mla576_cached_cuda(
            q, kv_cache, T, mask, kv_max, meta, scale);
        auto q_ref = q.to(tyloi::kFloat16).to(tyloi::kFloat32);
        auto k_ref = kv.to(tyloi::kFloat32).expand({B, H, T, DQ});
        auto scores = tyloi::cuda::matmul(q_ref, k_ref.transpose(-1, -2)) * scale;
        auto causal = tyloi::cuda::ones({T, T}, cuda.dtype(tyloi::kBool)).tril();
        scores = scores.masked_fill(~causal, -std::numeric_limits<float>::infinity());
        auto ref = tyloi::cuda::matmul(
            tyloi::cuda::softmax(scores, -1),
            k_ref.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
        const double max_abs = (test - ref).abs().max().item<double>();
        std::cout << "glm_dense_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
        if (!tyloi::cuda::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense MLA numerical check failed");
        }
    }

    {
        constexpr int B = 1, H = 64, VISIBLE = 33, PLANNED = 64;
        constexpr int DQ = 576, DV = 512;
        const double scale = 1.0 / std::sqrt(256.0);
        auto q = tyloi::cuda::randn({B, H, 1, DQ}, cuda.dtype(tyloi::kFloat32));
        auto kv = tyloi::cuda::randn({B, 1, PLANNED, DQ}, cuda.dtype(tyloi::kFloat16));
        auto seq_len = tyloi::cuda::tensor({VISIBLE}, cuda.dtype(tyloi::kInt64));
        auto mask = tyloi::cuda::empty({B, PLANNED}, cuda.dtype(tyloi::kFloat16));
        auto kv_max = tyloi::cuda::empty({B}, cuda.dtype(tyloi::kInt32));
        auto meta = tyloi::cuda::empty({8 * 1024 * 1024}, cuda.dtype(tyloi::kFloat32));
        auto test = attention_glm_mla576_decode_cuda(
            q, kv, seq_len, scale, PLANNED, mask, kv_max, meta);
        auto selected = kv.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(0, VISIBLE), tyloi::indexing::Slice()})
            .to(tyloi::kFloat32).expand({B, H, VISIBLE, DQ});
        auto ref = tyloi::cuda::matmul(
            tyloi::cuda::softmax(tyloi::cuda::matmul(q, selected.transpose(-1, -2)) * scale, -1),
            selected.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(), tyloi::indexing::Slice(0, DV)}))
            .permute({0, 2, 1, 3}).contiguous();
        const double rel = (test - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "glm_dense_decode_rel=" << rel << "\n";
        if (!tyloi::cuda::isfinite(test).all().item<bool>() || rel > 0.02) {
            throw std::runtime_error("GLM dense decode numerical check failed");
        }
    }

    constexpr int B = 1, H = 64, M = 3, K = 2112, TOPK = 2048, DQ = 576, DV = 512;
    const double scale = 1.0 / std::sqrt(256.0);
    auto q = tyloi::cuda::randn({B, H, M, DQ}, cuda.dtype(tyloi::kFloat32));
    auto kv = tyloi::cuda::randn({B, K, DQ}, cuda.dtype(tyloi::kFloat16));
    std::vector<tyloi::DenseTensor> rows;
    rows.reserve(M);
    for (int row = 0; row < M; ++row) {
        rows.push_back(tyloi::cuda::randperm(K - M + row + 1, cuda.dtype(tyloi::kInt64))
            .narrow(0, 0, TOPK).to(tyloi::kInt32));
    }
    auto indices = tyloi::cuda::stack(rows, 0).unsqueeze(0).contiguous();
    auto meta = tyloi::cuda::empty({8 * 1024 * 1024}, cuda.dtype(tyloi::kFloat32));
    auto run_sparse = [&]() {
        return attention_glm_mla_sparse_cuda(q, kv, indices, meta, scale);
    };
    auto test = run_sparse();
    std::vector<tyloi::DenseTensor> refs;
    refs.reserve(M);
    auto q_ref = q.to(tyloi::kFloat16).to(tyloi::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = indices.index({0, row}).to(tyloi::kInt64);
        auto selected = kv.index_select(1, idx).index({0}).to(tyloi::kFloat32);
        auto score = tyloi::cuda::matmul(q_ref.index({0, tyloi::indexing::Slice(), row}), selected.transpose(0, 1)) * scale;
        refs.push_back(tyloi::cuda::matmul(
            tyloi::cuda::softmax(score, -1), selected.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(0, DV)})));
    }
    auto ref = tyloi::cuda::stack(refs, 0).unsqueeze(0);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    const double rel = (test - ref).norm().item<double>() / ref.norm().item<double>();
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "glm_sparse_mla_rel=" << rel << " max_abs=" << max_abs << "\n";
    if (!tyloi::cuda::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error("GLM sparse MLA numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&start));
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(stop, stream));
        TYLOINFER_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(start));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    for (int i = 0; i < 5; ++i) run_sparse();
    const float sparse_ms = time_ms(run_sparse);
    std::cout << "glm_sparse_mla_m=" << M << " k=" << K
              << " topk=" << TOPK << " cuda_ms=" << sparse_ms << "\n";
    return 0;
}

int run_dsv4_hc_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    tyloi::cuda::NoGradGuard no_grad;
    auto cuda = tyloi::TensorOptions().device(tyloi::kCUDA);
    constexpr int64_t hc = 4;
    constexpr int64_t hidden = 4096;
    constexpr int64_t mix_width = 24;
    constexpr double eps = 1e-6;

    auto x_sequence = tyloi::cuda::arange(
        hc * hidden, cuda.dtype(tyloi::kFloat32));
    auto x = (
        (x_sequence.remainder(251) - 125.0) / 64.0 +
        0.125 * tyloi::cuda::sin(x_sequence * 0.015625))
        .to(tyloi::kFloat16)
        .reshape({1, 1, hc, hidden})
        .contiguous();
    auto function_sequence = tyloi::cuda::arange(
        mix_width * hc * hidden, cuda.dtype(tyloi::kFloat32));
    auto function = (
        0.003 * tyloi::cuda::sin(function_sequence * 0.001953125) +
        0.001 * tyloi::cuda::cos(function_sequence * 0.0078125))
        .reshape({mix_width, hc * hidden})
        .contiguous();
    auto scale = (
        0.7 + 0.2 * tyloi::cuda::arange(3, cuda.dtype(tyloi::kFloat32)))
        .contiguous();
    auto base = (
        0.1 * tyloi::cuda::sin(
            tyloi::cuda::arange(
                mix_width, cuda.dtype(tyloi::kFloat32)) * 0.3125))
        .contiguous();
    auto flat = x.flatten(2).to(tyloi::kFloat32);
    auto inverse_rms = tyloi::cuda::rsqrt(
        flat.square().mean(-1, true) + eps);
    auto mixes = (
        tyloi::cuda::matmul(flat, function.transpose(0, 1)) * inverse_rms)
        .contiguous();

    auto reference_split = dsv4_hc_split_sinkhorn(
        mixes, scale, base, hc, 20, eps);
    auto reference_reduced = (
        reference_split.at(0).unsqueeze(-1) *
        flat.reshape({1, 1, hc, hidden}))
        .sum(2).to(tyloi::kFloat16).contiguous();
    auto candidate = dsv4_hc_pre_cuda(
        x, mixes, scale, base, 20, eps);

    auto direct = x.index(
        {tyloi::indexing::Slice(), tyloi::indexing::Slice(), 0, tyloi::indexing::Slice()}).contiguous();
    auto reference_post = (
        reference_split.at(1).unsqueeze(-1) *
            direct.unsqueeze(-2) +
        (reference_split.at(2).unsqueeze(-1) *
            x.to(tyloi::kFloat32).unsqueeze(-2)).sum(2))
        .to(tyloi::kFloat16).contiguous();
    auto candidate_post = dsv4_hc_post_cuda(
        direct, x, candidate.at(1), candidate.at(2));
    cudaStreamSynchronize(::tyloi::cuda::current_stream());

    auto compare = [](const char * name,
                      tyloi::DenseTensor reference,
                      tyloi::DenseTensor value) {
        auto reference_f32 = reference.to(tyloi::kFloat32);
        auto value_f32 = value.to(tyloi::kFloat32);
        auto diff = value_f32 - reference_f32;
        const double norm = reference_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "dsv4_hc_ab tensor=" << name
                  << " equal=" << (value.equal(reference) ? 1 : 0)
                  << " differing="
                  << value.ne(reference).sum().item<int64_t>()
                  << " rel_l2="
                  << (norm == 0.0
                      ? 0.0
                      : diff.norm().item<double>() / norm)
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    };
    compare("reduced", reference_reduced, candidate.at(0));
    compare("post", reference_split.at(1), candidate.at(1));
    compare("combination", reference_split.at(2), candidate.at(2));
    compare("hc_post", reference_post, candidate_post);

    auto time_ms = [&](auto && fn) {
        for (int warmup = 0; warmup < 10; ++warmup) fn();
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        cudaEvent_t begin, end;
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
        auto stream = mfq_get_current_cuda_stream().stream();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin, stream));
        for (int iteration = 0; iteration < reps; ++iteration) fn();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end, stream));
        TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed = 0.0f;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
        return elapsed / reps;
    };
    const float reference_pre_ms = time_ms([&]() {
        auto split = dsv4_hc_split_sinkhorn(
            mixes, scale, base, hc, 20, eps);
        return (
            split.at(0).unsqueeze(-1) *
            flat.reshape({1, 1, hc, hidden}))
            .sum(2).to(tyloi::kFloat16).contiguous();
    });
    const float candidate_pre_ms = time_ms([&]() {
        return dsv4_hc_pre_cuda(
            x, mixes, scale, base, 20, eps);
    });
    const float reference_post_ms = time_ms([&]() {
        return (
            reference_split.at(1).unsqueeze(-1) *
                direct.unsqueeze(-2) +
            (reference_split.at(2).unsqueeze(-1) *
                x.to(tyloi::kFloat32).unsqueeze(-2)).sum(2))
            .to(tyloi::kFloat16).contiguous();
    });
    const float candidate_post_ms = time_ms([&]() {
        return dsv4_hc_post_cuda(
            direct, x, candidate.at(1), candidate.at(2));
    });
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_hc_timing"
              << " reference_pre_ms=" << reference_pre_ms
              << " candidate_pre_ms=" << candidate_pre_ms
              << " pre_speedup=" << reference_pre_ms / candidate_pre_ms
              << " reference_post_ms=" << reference_post_ms
              << " candidate_post_ms=" << candidate_post_ms
              << " post_speedup=" << reference_post_ms / candidate_post_ms
              << "\n";
    return 0;
}

int run_dsv4_attention_check(int reps) {
    if (reps < 1) {
        throw std::runtime_error("--check-attention-reps must be positive");
    }
    tyloi::cuda::NoGradGuard no_grad;
    auto cuda = tyloi::TensorOptions().device(tyloi::kCUDA);
    tyloi::cuda::manual_seed(20260723);

    {
        constexpr int EXPERTS = 256, TOPK = 6;
        auto logits = tyloi::cuda::randn({1, EXPERTS}, cuda.dtype(tyloi::kFloat32)) * 3.0;
        auto bias = tyloi::cuda::randn({EXPERTS}, cuda.dtype(tyloi::kFloat32)) * 0.05;
        auto selected = moe_topk_cuda(
            logits, TOPK, false, true, true, false, bias, 1e-20, 1.5);
        auto transformed = tyloi::cuda::sqrt(tyloi::cuda::where(
            logits > 20.0, logits, tyloi::cuda::log1p(tyloi::cuda::exp(logits))));
        auto expected_topk = tyloi::cuda::topk(
            transformed + bias.unsqueeze(0), TOPK, 1, true, true);
        auto expected_ids = std::get<1>(expected_topk);
        auto expected_weights = transformed.gather(1, expected_ids);
        expected_weights = expected_weights /
            expected_weights.sum(1, true).clamp_min(1e-20) * 1.5;
        const bool ids_equal =
            selected.at(0).to(tyloi::kInt64).equal(expected_ids);
        const double max_abs = (selected.at(1) - expected_weights)
            .abs().max().item<double>();

        auto hash_ids = tyloi::cuda::randint(
            0, EXPERTS, {7, TOPK}, cuda.dtype(tyloi::kInt32));
        auto hash_logits =
            tyloi::cuda::randn({7, EXPERTS}, cuda.dtype(tyloi::kFloat32)) * 3.0;
        auto hash_weights = moe_sqrtsoftplus_weights_cuda(
            hash_logits, hash_ids, 1e-20, 1.5);
        auto hash_transformed = tyloi::cuda::sqrt(tyloi::cuda::where(
            hash_logits > 20.0, hash_logits,
            tyloi::cuda::log1p(tyloi::cuda::exp(hash_logits))));
        auto expected_hash = hash_transformed.gather(
            1, hash_ids.to(tyloi::kInt64));
        expected_hash = expected_hash /
            expected_hash.sum(1, true).clamp_min(1e-20) * 1.5;
        const double hash_max_abs = (hash_weights - expected_hash)
            .abs().max().item<double>();
        std::cout << "dsv4_moe_route_ids_equal=" << (ids_equal ? 1 : 0)
                  << " max_abs=" << max_abs
                  << " hash_max_abs=" << hash_max_abs << "\n";
        if (!ids_equal || max_abs > 2e-6 || hash_max_abs > 2e-6) {
            throw std::runtime_error(
                "DeepSeek V4 MoE routing numerical check failed");
        }
    }

    constexpr int B = 1;
    constexpr int D = 512;
    constexpr int RD = 64;
    constexpr double eps = 1e-6;
    auto norm = tyloi::cuda::randn({D}, cuda.dtype(tyloi::kFloat32));

    {
        constexpr int W = 2, R = 128;
        auto kv = tyloi::cuda::randn({B, W, R, D}, cuda.dtype(tyloi::kFloat16));
        auto gate = tyloi::cuda::randn({B, W, R, D}, cuda.dtype(tyloi::kFloat16));
        auto ape = tyloi::cuda::randn({R, D}, cuda.dtype(tyloi::kFloat32));
        auto empty = tyloi::cuda::empty({0}, cuda.dtype(tyloi::kFloat16));
        auto positions = tyloi::cuda::arange(W, cuda.dtype(tyloi::kInt64)).reshape({B, W});
        auto cos = tyloi::cuda::ones({W + 1, RD / 2}, cuda.dtype(tyloi::kFloat32));
        auto sin = tyloi::cuda::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, empty, empty, positions, cos, sin,
            R, false, 1, eps);
        auto score = gate.to(tyloi::kFloat32) +
            ape.reshape({1, 1, R, D});
        auto pooled = (kv.to(tyloi::kFloat32) *
            tyloi::cuda::softmax(score, 2)).sum(2);
        auto ref = pooled * tyloi::cuda::rsqrt(
            pooled.square().mean(-1, true) + eps) * norm;
        ref = ref.to(tyloi::kBFloat16).to(tyloi::kFloat32);
        auto ref_nope = ref.slice(-1, 0, D - RD)
            .reshape({-1, (D - RD) / 64, 64});
        auto fp8_scale = ref_nope.abs().amax(-1, true)
            .clamp_min(1e-4) / 448.0;
        ref_nope = (ref_nope / fp8_scale)
            .clamp(-448.0, 448.0)
            .to(mfq_float8_e4m3fn)
            .to(tyloi::kFloat32) * fp8_scale;
        ref.slice(-1, 0, D - RD).copy_(
            ref_nope.reshape({B, W, D - RD}));
        ref = ref.to(tyloi::kBFloat16)
            .to(tyloi::kFloat16).to(tyloi::kFloat32);
        const double rel =
            (test.to(tyloi::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_hca_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-128 compressor numerical check failed");
        }
    }

    {
        constexpr int W = 3, R = 4, OD = 2 * D;
        auto kv = tyloi::cuda::randn({B, W, R, OD}, cuda.dtype(tyloi::kFloat16));
        auto gate = tyloi::cuda::randn({B, W, R, OD}, cuda.dtype(tyloi::kFloat16));
        auto prev_kv = tyloi::cuda::randn({B, R, D}, cuda.dtype(tyloi::kFloat16));
        auto prev_gate = tyloi::cuda::randn({B, R, D}, cuda.dtype(tyloi::kFloat16));
        auto ape = tyloi::cuda::randn({R, OD}, cuda.dtype(tyloi::kFloat32));
        auto positions = tyloi::cuda::arange(W, cuda.dtype(tyloi::kInt64)).reshape({B, W});
        auto cos = tyloi::cuda::ones({W + 1, RD / 2}, cuda.dtype(tyloi::kFloat32));
        auto sin = tyloi::cuda::zeros_like(cos);
        auto test = dsv4_compress_cuda(
            kv, gate, ape, norm, prev_kv, prev_gate, positions,
            cos, sin, R, true, 0, eps);
        std::vector<tyloi::DenseTensor> ref_rows;
        for (int w = 0; w < W; ++w) {
            auto left_kv = w == 0
                ? prev_kv
                : kv.index({tyloi::indexing::Slice(), w - 1, tyloi::indexing::Slice(), tyloi::indexing::Slice(0, D)});
            auto left_gate = w == 0
                ? prev_gate
                : gate.index({tyloi::indexing::Slice(), w - 1, tyloi::indexing::Slice(), tyloi::indexing::Slice(0, D)});
            auto right_kv =
                kv.index({tyloi::indexing::Slice(), w, tyloi::indexing::Slice(), tyloi::indexing::Slice(D, OD)});
            auto right_gate =
                gate.index({tyloi::indexing::Slice(), w, tyloi::indexing::Slice(), tyloi::indexing::Slice(D, OD)});
            auto values = tyloi::cuda::cat({left_kv, right_kv}, 1)
                .to(tyloi::kFloat32);
            auto score = tyloi::cuda::cat({
                left_gate.to(tyloi::kFloat32) +
                    ape.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(0, D)}).unsqueeze(0),
                right_gate.to(tyloi::kFloat32) +
                    ape.index({tyloi::indexing::Slice(), tyloi::indexing::Slice(D, OD)}).unsqueeze(0)}, 1);
            auto pooled = (values * tyloi::cuda::softmax(score, 1)).sum(1);
            ref_rows.push_back(
                pooled * tyloi::cuda::rsqrt(
                    pooled.square().mean(-1, true) + eps) * norm);
        }
        auto ref = tyloi::cuda::stack(ref_rows, 1)
            .to(tyloi::kFloat16).to(tyloi::kFloat32);
        const double rel =
            (test.to(tyloi::kFloat32) - ref).norm().item<double>() /
            std::max(ref.norm().item<double>(), 1e-30);
        std::cout << "dsv4_csa_overlap_compressor_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 ratio-4 overlap compressor numerical check failed");
        }
    }

    {
        constexpr int T = 12, R = 4, W = T / R, OD = 2 * D;
        auto kv = tyloi::cuda::randn({B, T, OD}, cuda.dtype(tyloi::kFloat32));
        auto gate = tyloi::cuda::randn({B, T, OD}, cuda.dtype(tyloi::kFloat32));
        auto ape = tyloi::cuda::randn({R, OD}, cuda.dtype(tyloi::kFloat32));
        auto empty = tyloi::cuda::empty({0}, cuda.dtype(tyloi::kFloat16));
        auto positions = tyloi::cuda::arange(W, cuda.dtype(tyloi::kInt64)).reshape({B, W});
        auto cos = tyloi::cuda::ones({W + 1, RD / 2}, cuda.dtype(tyloi::kFloat32));
        auto sin = tyloi::cuda::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, norm, empty, empty, positions, cos, sin,
            R, true, 1, eps);
        auto state_kv = tyloi::cuda::zeros(
            {B, R, OD}, cuda.dtype(tyloi::kFloat32));
        auto state_gate = tyloi::cuda::zeros_like(state_kv);
        auto prev_kv = tyloi::cuda::zeros(
            {B, R, D}, cuda.dtype(tyloi::kFloat32));
        auto prev_gate = tyloi::cuda::zeros_like(prev_kv);
        auto pool = tyloi::cuda::zeros(
            {B, W, D}, cuda.dtype(tyloi::kFloat16));
        auto seq_len = tyloi::cuda::zeros({B}, cuda.dtype(tyloi::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, norm, state_kv, state_gate, prev_kv, prev_gate,
                pool, seq_len, cos, sin, R, true, 1, eps);
        }
        const double rel =
            (pool.to(tyloi::kFloat32) - batch.to(tyloi::kFloat32))
                .norm().item<double>() /
            std::max(batch.to(tyloi::kFloat32).norm().item<double>(), 1e-30);
        std::cout << "dsv4_decode_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 decode compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 12, R = 4, W = T / R;
        constexpr int OD = 2 * ID;
        auto index_norm = tyloi::cuda::randn(
            {ID}, cuda.dtype(tyloi::kFloat32));
        auto kv = tyloi::cuda::randn(
            {B, T, OD}, cuda.dtype(tyloi::kFloat32));
        auto gate = tyloi::cuda::randn(
            {B, T, OD}, cuda.dtype(tyloi::kFloat32));
        auto ape = tyloi::cuda::randn(
            {R, OD}, cuda.dtype(tyloi::kFloat32));
        auto empty = tyloi::cuda::empty({0}, cuda.dtype(tyloi::kFloat16));
        auto positions = tyloi::cuda::arange(
            W, cuda.dtype(tyloi::kInt64)).reshape({B, W});
        auto cos = tyloi::cuda::ones(
            {W + 1, RD / 2}, cuda.dtype(tyloi::kFloat32));
        auto sin = tyloi::cuda::zeros_like(cos);
        auto batch = dsv4_compress_cuda(
            kv.reshape({B, W, R, OD}).contiguous(),
            gate.reshape({B, W, R, OD}).contiguous(),
            ape, index_norm, empty, empty, positions, cos, sin,
            R, true, 2, eps);
        auto state_kv = tyloi::cuda::zeros(
            {B, R, OD}, cuda.dtype(tyloi::kFloat32));
        auto state_gate = tyloi::cuda::zeros_like(state_kv);
        auto prev_kv = tyloi::cuda::zeros(
            {B, R, ID}, cuda.dtype(tyloi::kFloat32));
        auto prev_gate = tyloi::cuda::zeros_like(prev_kv);
        auto pool = tyloi::cuda::zeros(
            {B, W, ID}, cuda.dtype(tyloi::kFloat16));
        auto seq_len = tyloi::cuda::zeros(
            {B}, cuda.dtype(tyloi::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            dsv4_decode_pool_update_cuda(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                ape, index_norm, state_kv, state_gate,
                prev_kv, prev_gate, pool, seq_len, cos, sin,
                R, true, 2, eps);
        }
        const double rel =
            (pool.to(tyloi::kFloat32) - batch.to(tyloi::kFloat32))
                .norm().item<double>() /
            std::max(
                batch.to(tyloi::kFloat32).norm().item<double>(),
                1e-30);
        std::cout << "dsv4_indexer_pool_state_rel=" << rel << "\n";
        if (rel > 0.002) {
            throw std::runtime_error(
                "DeepSeek V4 indexer compressor state check failed");
        }
    }

    {
        constexpr int ID = 128, T = 14, R = 4, OD = 2 * ID;
        auto index_norm = tyloi::cuda::randn(
            {ID}, cuda.dtype(tyloi::kFloat32));
        auto kv = tyloi::cuda::randn(
            {B, T, OD}, cuda.dtype(tyloi::kFloat32));
        auto gate = tyloi::cuda::randn(
            {B, T, OD}, cuda.dtype(tyloi::kFloat32));
        Dsv4RopeTable rope;
        rope.cos = tyloi::cuda::ones(
            {T + 1, RD / 2}, cuda.dtype(tyloi::kFloat32));
        rope.sin = tyloi::cuda::zeros_like(rope.cos);
        rope.negative_sin = -rope.sin;

        Dsv4PoolState batched;
        batched.ratio = R;
        batched.head_dim = ID;
        batched.overlap = true;
        batched.cache_quant_mode = 2;
        batched.ape = tyloi::cuda::randn(
            {R, OD}, cuda.dtype(tyloi::kFloat32));
        batched.norm = index_norm;
        batched.reset(B, T);

        Dsv4PoolState decoded;
        decoded.ratio = batched.ratio;
        decoded.head_dim = batched.head_dim;
        decoded.overlap = batched.overlap;
        decoded.cache_quant_mode = batched.cache_quant_mode;
        decoded.ape = batched.ape;
        decoded.norm = batched.norm;
        decoded.reset(B, T);

        const int64_t windows = batched.prefill(kv, gate, rope);
        auto seq_len = tyloi::cuda::zeros(
            {B}, cuda.dtype(tyloi::kInt64));
        for (int t = 0; t < T; ++t) {
            seq_len.fill_(t + 1);
            decoded.update(
                kv.narrow(1, t, 1).contiguous(),
                gate.narrow(1, t, 1).contiguous(),
                seq_len, rope);
        }
        const auto pool_width = T / R;
        auto reference_pool = decoded.pool.narrow(1, 0, pool_width);
        auto candidate_pool = batched.pool.narrow(1, 0, pool_width);
        const double pool_rel =
            (candidate_pool.to(tyloi::kFloat32) -
             reference_pool.to(tyloi::kFloat32)).norm().item<double>() /
            std::max(
                reference_pool.to(tyloi::kFloat32).norm().item<double>(),
                1e-30);
        const double state_rel =
            (batched.state_kv - decoded.state_kv)
                .norm().item<double>() /
            std::max(decoded.state_kv.norm().item<double>(), 1e-30);
        const double gate_rel =
            (batched.state_gate - decoded.state_gate)
                .norm().item<double>() /
            std::max(decoded.state_gate.norm().item<double>(), 1e-30);
        const double previous_rel =
            (batched.previous_kv - decoded.previous_kv)
                .norm().item<double>() /
            std::max(decoded.previous_kv.norm().item<double>(), 1e-30);
        const double previous_gate_rel =
            (batched.previous_gate - decoded.previous_gate)
                .norm().item<double>() /
            std::max(decoded.previous_gate.norm().item<double>(), 1e-30);
        std::cout << "dsv4_pool_prefill_windows=" << windows
                  << " pool_rel=" << pool_rel
                  << " state_rel=" << state_rel
                  << " gate_rel=" << gate_rel
                  << " previous_rel=" << previous_rel
                  << " previous_gate_rel=" << previous_gate_rel << "\n";
        if (windows != pool_width || pool_rel > 0.002 ||
            state_rel > 1e-7 || gate_rel > 1e-7 ||
            previous_rel > 1e-7 || previous_gate_rel > 1e-7) {
            throw std::runtime_error(
                "DeepSeek V4 batched compressor prefill state check failed");
        }
    }

    {
        constexpr int ROWS = 9, WIDTH = 128;
        auto input = tyloi::cuda::randn(
            {ROWS, WIDTH}, cuda.dtype(tyloi::kFloat16)) * 2.0;
        auto test = dsv4_fp4_sim_cuda(input.contiguous());
        auto grouped = input.to(tyloi::kFloat32)
            .reshape({ROWS, WIDTH / 32, 32});
        auto scale = tyloi::cuda::exp2(tyloi::cuda::ceil(tyloi::cuda::log2(
            grouped.abs().amax(-1, true)
                .clamp_min(6.0 * std::ldexp(1.0, -126)) / 6.0)));
        auto normalized = (grouped / scale).clamp(-6.0, 6.0);
        auto magnitude = normalized.abs();
        auto quantized = tyloi::cuda::where(
            magnitude <= 0.25, tyloi::cuda::zeros_like(magnitude),
            tyloi::cuda::where(
                magnitude < 0.75, tyloi::cuda::full_like(magnitude, 0.5),
                tyloi::cuda::where(
                    magnitude <= 1.25, tyloi::cuda::ones_like(magnitude),
                    tyloi::cuda::where(
                        magnitude < 1.75,
                        tyloi::cuda::full_like(magnitude, 1.5),
                        tyloi::cuda::where(
                            magnitude <= 2.5,
                            tyloi::cuda::full_like(magnitude, 2.0),
                            tyloi::cuda::where(
                                magnitude < 3.5,
                                tyloi::cuda::full_like(magnitude, 3.0),
                                tyloi::cuda::where(
                                    magnitude <= 5.0,
                                    tyloi::cuda::full_like(magnitude, 4.0),
                                    tyloi::cuda::full_like(
                                        magnitude, 6.0))))))));
        quantized = tyloi::cuda::where(
            normalized < 0, -quantized, quantized);
        auto reference = (quantized * scale)
            .reshape({ROWS, WIDTH}).to(tyloi::kFloat16);
        const double max_abs = (test - reference)
            .abs().max().item<double>();
        std::cout << "dsv4_fp4_sim_max_abs=" << max_abs << "\n";
        if (max_abs != 0.0) {
            throw std::runtime_error(
                "DeepSeek V4 FP4 activation simulation check failed");
        }
    }

    {
        constexpr int M = 3, K = 768, H = 64, ID = 128;
        auto q = tyloi::cuda::randn({B, M, H, ID}, cuda.dtype(tyloi::kFloat16));
        auto k = tyloi::cuda::randn({B, K, ID}, cuda.dtype(tyloi::kFloat16));
        auto weights = tyloi::cuda::randn({B, M, H}, cuda.dtype(tyloi::kFloat16));
        auto test = dsv4_indexer_scores_cuda(q, k, weights, 4096, 4);
        auto dot = tyloi::cuda::einsum(
            "bmhd,bkd->bmhk",
            {q.to(tyloi::kFloat32), k.to(tyloi::kFloat32)});
        auto ref = (tyloi::cuda::relu(dot) *
            weights.to(tyloi::kFloat32).unsqueeze(-1)).sum(2) /
            std::sqrt(static_cast<double>(H * ID));
        ref = ref.to(tyloi::kFloat16);
        const double rel =
            (test.to(tyloi::kFloat32) - ref.to(tyloi::kFloat32))
                .norm().item<double>() /
            std::max(ref.to(tyloi::kFloat32).norm().item<double>(), 1e-30);
        auto selected = dsv4_topk512_cuda(test);
        auto expected_topk = tyloi::cuda::topk(test, 512, -1, true, false);
        auto expected_scores = std::get<0>(expected_topk);
        auto expected = std::get<1>(expected_topk);
        auto selected_i64 = selected.to(tyloi::kInt64);
        auto selected_sorted = std::get<0>(
            tyloi::cuda::sort(selected_i64, -1));
        auto expected_sorted = std::get<0>(
            tyloi::cuda::sort(expected.to(tyloi::kInt64), -1));
        const bool topk_id_equal = selected_sorted.equal(expected_sorted);
        const bool topk_ids_valid =
            (selected_i64 >= 0).all().item<bool>() &&
            (selected_i64 < test.size(-1)).all().item<bool>();
        const bool topk_ids_unique =
            selected_sorted.slice(-1, 1, selected_sorted.size(-1))
                .ne(selected_sorted.slice(-1, 0, selected_sorted.size(-1) - 1))
                .all().item<bool>();
        bool topk_scores_equal = false;
        if (topk_ids_valid) {
            auto selected_scores = test.gather(-1, selected_i64);
            auto selected_score_sorted = std::get<0>(
                tyloi::cuda::sort(selected_scores, -1));
            auto expected_score_sorted = std::get<0>(
                tyloi::cuda::sort(expected_scores, -1));
            topk_scores_equal =
                selected_score_sorted.equal(expected_score_sorted);
        }
        std::cout << "dsv4_indexer_rel=" << rel
                  << " topk_id_set_equal=" << (topk_id_equal ? 1 : 0)
                  << " topk_score_multiset_equal="
                  << (topk_scores_equal ? 1 : 0)
                  << " topk_ids_valid=" << (topk_ids_valid ? 1 : 0)
                  << " topk_ids_unique=" << (topk_ids_unique ? 1 : 0)
                  << "\n";
        if (rel > 0.004 || !topk_scores_equal ||
            !topk_ids_valid || !topk_ids_unique) {
            throw std::runtime_error(
                "DeepSeek V4 indexer/top-k numerical check failed");
        }
    }

    constexpr int H = 64;
    constexpr int M = 3;
    constexpr int HISTORY = 127;
    constexpr int POOL = 800;
    constexpr int TOPK = 512;
    constexpr int WINDOW = 128;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    auto q = tyloi::cuda::randn({B, H, M, D}, cuda.dtype(tyloi::kFloat32));
    auto raw = tyloi::cuda::randn({B, HISTORY + M, D}, cuda.dtype(tyloi::kFloat16));
    auto pooled = tyloi::cuda::randn({B, POOL, D}, cuda.dtype(tyloi::kFloat16));
    auto kv = tyloi::cuda::cat({raw, pooled}, 1).contiguous();
    std::vector<tyloi::DenseTensor> topk_rows;
    for (int row = 0; row < M; ++row) {
        topk_rows.push_back(
            tyloi::cuda::randperm(POOL, cuda.dtype(tyloi::kInt64))
                .narrow(0, 0, TOPK).to(tyloi::kInt32));
    }
    auto topk = tyloi::cuda::stack(topk_rows, 0).unsqueeze(0).contiguous();
    auto plan = dsv4_build_prefill_plan_cuda(
        topk, 4096, HISTORY, POOL, 4, WINDOW);
    {
        auto seq_len = tyloi::cuda::tensor({4097}, cuda.dtype(tyloi::kInt64));
        auto decode_plan = dsv4_build_decode_plan_cuda(
            topk.narrow(1, 0, 1).contiguous(),
            seq_len, POOL, 4, WINDOW);
        auto expected_local =
            tyloi::cuda::arange(4097 - WINDOW, 4097, cuda.dtype(tyloi::kInt64))
                .remainder(WINDOW);
        const bool local_equal = decode_plan[0]
            .index({0, 0, tyloi::indexing::Slice(0, WINDOW)})
            .to(tyloi::kInt64).equal(expected_local);
        const bool pooled_equal = decode_plan[0]
            .index({0, 0, tyloi::indexing::Slice(WINDOW, WINDOW + TOPK)})
            .equal(topk.index({0, 0}) + WINDOW);
        const bool mask_clear =
            (decode_plan[1] == 0).all().item<bool>();
        std::cout << "dsv4_decode_plan="
                  << (local_equal && pooled_equal && mask_clear ? 1 : 0)
                  << "\n";
        if (!local_equal || !pooled_equal || !mask_clear) {
            throw std::runtime_error(
                "DeepSeek V4 decode cache plan check failed");
        }
    }
    auto sinks = tyloi::cuda::randn({H}, cuda.dtype(tyloi::kFloat32));
    auto meta = tyloi::cuda::empty(
        {8 * 1024 * 1024}, cuda.dtype(tyloi::kFloat32));
    auto run_sparse = [&]() {
        return attention_dsv4_sparse_cuda(
            q, kv, plan[0], plan[1], sinks, meta, scale);
    };
    auto test = run_sparse();
    std::vector<tyloi::DenseTensor> ref_rows;
    auto q_ref = q.to(tyloi::kFloat16).to(tyloi::kFloat32);
    for (int row = 0; row < M; ++row) {
        auto idx = plan[0].index({0, row}).to(tyloi::kInt64);
        auto selected = kv.index_select(1, idx).index({0})
            .to(tyloi::kFloat32);
        auto score = tyloi::cuda::matmul(
            q_ref.index({0, tyloi::indexing::Slice(), row}),
            selected.transpose(0, 1)) * scale;
        score = score + plan[1].index({0, row})
            .to(tyloi::kFloat32).unsqueeze(0);
        auto logits = tyloi::cuda::cat({score, sinks.unsqueeze(1)}, 1);
        auto probabilities = tyloi::cuda::softmax(logits, -1)
            .index({tyloi::indexing::Slice(), tyloi::indexing::Slice(0, score.size(1))});
        ref_rows.push_back(tyloi::cuda::matmul(probabilities, selected));
    }
    auto ref = tyloi::cuda::stack(ref_rows, 0).unsqueeze(0);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    const double rel = (test - ref).norm().item<double>() /
        std::max(ref.norm().item<double>(), 1e-30);
    const double max_abs = (test - ref).abs().max().item<double>();
    std::cout << "dsv4_sparse_attention_rel=" << rel
              << " max_abs=" << max_abs << "\n";
    if (!tyloi::cuda::isfinite(test).all().item<bool>() || rel > 0.02) {
        throw std::runtime_error(
            "DeepSeek V4 sparse attention numerical check failed");
    }

    auto time_ms = [&](auto && fn) {
        cudaEvent_t start, stop;
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&start));
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&stop));
        auto stream = mfq_get_current_cuda_stream().stream();
        for (int i = 0; i < 5; ++i) fn();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < reps; ++i) fn();
        TYLOINFER_CUDA_CHECK(cudaEventRecord(stop, stream));
        TYLOINFER_CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(start));
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(stop));
        return elapsed / reps;
    };
    std::cout << "dsv4_sparse_attention_m=" << M
              << " selected=" << plan[0].size(2)
              << " cuda_ms=" << time_ms(run_sparse) << "\n";
    return 0;
}
