#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_context.h"
#include "runtime/decode_window.h"
#include "runtime/graph_pool_lease.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tb = mfq_tensor_backend;

struct Errors {
    double reference = 0;
    double fp64 = 0;
};

static void verify(const tb::Tensor& output, const tb::Tensor& query,
                   const tb::Tensor& keys, const tb::Tensor& values, Errors& errors) {
    const auto batch = query.size(0), heads = query.size(1), tokens = query.size(2);
    const auto width = query.size(3), kv_heads = keys.size(1), ratio = heads / kv_heads;
    const double scale = 1.0 / std::sqrt(double(width));
    auto expected = mfq_scaled_dot_product_attention(
        query.to(tb::kFloat16), keys, values, std::nullopt, 0.0, true, scale, true)
        .transpose(1, 2).contiguous().to(tb::kCPU, tb::kFloat32);
    auto actual = output.to(tb::kCPU, tb::kFloat32).contiguous();
    if (actual.sizes() != expected.sizes()) throw std::runtime_error("prefill shape differs");
    for (int64_t i = 0; i < actual.numel(); ++i) {
        const auto error = std::abs(double(actual.data_ptr<float>()[i]) - expected.data_ptr<float>()[i]);
        if (!std::isfinite(error)) throw std::runtime_error("nonfinite prefill output");
        errors.reference = std::max(errors.reference, error);
    }
    if (errors.reference > 1.5e-3) throw std::runtime_error("prefill differs from the attention reference");

    // Independent double-precision softmax for early, middle and final queries.
    auto q = query.to(tb::kCPU), k = keys.to(tb::kCPU, tb::kFloat32);
    auto v = values.to(tb::kCPU, tb::kFloat32);
    for (int64_t b = 0; b < batch; ++b) for (int64_t h = 0; h < heads; ++h) {
        for (int64_t t : {int64_t(0), tokens / 2, tokens - 1}) {
            std::vector<double> scores(t + 1);
            double maximum = -INFINITY;
            for (int64_t j = 0; j <= t; ++j) {
                double score = 0;
                for (int64_t d = 0; d < width; ++d)
                    score += double(q.data_ptr<float>()[((b * heads + h) * tokens + t) * width + d]) *
                        k.data_ptr<float>()[((b * kv_heads + h / ratio) * tokens + j) * width + d];
                scores[j] = score * scale;
                maximum = std::max(maximum, scores[j]);
            }
            double denominator = 0;
            for (auto& score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (int64_t d : {int64_t(0), int64_t(17), int64_t(63), int64_t(127), width - 1}) {
                double value = 0;
                for (int64_t j = 0; j <= t; ++j)
                    value += scores[j] * v.data_ptr<float>()[((b * kv_heads + h / ratio) * tokens + j) * width + d];
                value /= denominator;
                const auto actual_value = actual.data_ptr<float>()[((b * tokens + t) * heads + h) * width + d];
                errors.fp64 = std::max(errors.fp64, std::abs(double(actual_value) - value));
            }
        }
    }
    if (errors.fp64 > 5e-4) throw std::runtime_error("prefill differs from the FP64 reference");
}

int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    const auto device = mfq_current_cuda_device();
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess)
        throw std::runtime_error("cannot query CUDA capabilities");
    // The shared 32-column FP16 MMA body requires Volta instructions.
    if (properties.major < 7) return 77;
    const auto stream = mfq_current_cuda_stream();
    auto pool = mfq::cuda::GraphPoolLease::acquire(mfq::cuda::default_context(device), stream);
    tb::NoGradGuard guard;
    const auto cpu = tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    Errors errors;
    int cases = 0;
    const auto run = [&](int batch, int kv_heads, int ratio, int tokens, bool dynamic) {
        constexpr int width = 256;
        const auto heads = kv_heads * ratio;
        auto hq = tb::empty({batch, heads, tokens, width}, cpu);
        auto hk = tb::empty({batch, kv_heads, tokens, width}, cpu), hv = tb::empty_like(hk);
        for (int64_t i = 0; i < hq.numel(); ++i) hq.data_ptr<float>()[i] = .7f * std::sin(float(i) * .0317f);
        for (int64_t i = 0; i < hk.numel(); ++i) {
            hk.data_ptr<float>()[i] = .5f * std::sin(float(i) * .0119f);
            hv.data_ptr<float>()[i] = .3f * std::cos(float(i) * .0071f);
        }
        auto q = hq.to(tb::kCUDA), k = hk.to(tb::kCUDA, tb::kFloat16), v = hv.to(tb::kCUDA, tb::kFloat16);
        tb::Tensor output;
        const auto body = [&] { output = mfq_attention_mma256_cuda(q, k, v, 1.0 / 16); };
        const auto check = [&] { verify(output, q, k, v, errors); ++cases; };
        body(); check();
        if (dynamic) {
            mfq::cuda::DecodeWindow graph(stream);
            graph.capture(body, [] {}, [&] { output = {}; });
            for (float factor : {.25f, -.5f, 0.f}) { q.copy_(q * factor); graph.run(); check(); }
        }
    };
    for (int ratio : {1, 2, 3, 6, 12})
        for (int tokens : {1, 7, 32, 63, 64, 65, 129}) run(1, 1, ratio, tokens, false);
    run(2, 2, 6, 97, true);
    run(1, 2, 3, 257, false);
    std::cout << "General GQA prefill PASS cases=" << cases << " reference_error="
              << errors.reference << " fp64_error=" << errors.fp64 << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
