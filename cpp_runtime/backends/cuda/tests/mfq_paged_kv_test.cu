#include "mfq_cuda_paged_kv.h"
#include "mfq_native_tensor.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using mfq::cuda::Device;
using mfq::cuda::DeviceType;
using mfq::cuda::Tensor;
using mfq::cuda::TensorOptions;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

std::vector<float> float_values(const Tensor & tensor) {
    auto host = tensor.contiguous().to(mfq::cuda::kFloat32)
        .to(mfq::cuda::kCPU);
    return std::vector<float>(
        host.data_ptr<float>(), host.data_ptr<float>() + host.numel());
}

void check_kernels(int D,int ratio,int Page,int PagesPerChunk,mfq::cuda::ScalarType dtype) {
    constexpr int B = 2;
    const int Hq = 2*ratio;
    constexpr int Hk = 2;
    const int T = 2*Page-1;
    constexpr int LogicalPages = 4;
    const Device gpu{DeviceType::cuda, 0};

    std::vector<float> k_host(B * Hk * T * D);
    std::vector<float> v_host(k_host.size());
    std::vector<float> q_host(B * Hq * D);
    for (size_t index = 0; index < k_host.size(); ++index) {
        k_host[index] = static_cast<float>(
            static_cast<int>((index * 17 + 3) % 97) - 48) / 83.0f;
        v_host[index] = static_cast<float>(
            static_cast<int>((index * 29 + 5) % 89) - 44) / 71.0f;
    }
    for (size_t index = 0; index < q_host.size(); ++index) {
        q_host[index] = static_cast<float>(
            static_cast<int>((index * 11 + 7) % 79) - 39) / 67.0f;
    }
    auto k = mfq::cuda::tensor<float>(k_host).reshape({B, Hk, T, D})
        .to(gpu).to(dtype).contiguous();
    auto v = mfq::cuda::tensor<float>(v_host).reshape({B, Hk, T, D})
        .to(gpu).to(dtype).contiguous();
    auto q = mfq::cuda::tensor<float>(q_host).reshape({B, Hq, 1, D})
        .to(gpu).to(dtype).contiguous();
    auto k_chunk_0 = mfq::cuda::empty(
        {PagesPerChunk, Hk, Page, D},
        TensorOptions().device(gpu).dtype(dtype));
    auto v_chunk_0 = mfq::cuda::empty_like(k_chunk_0);
    auto k_chunk_1 = mfq::cuda::empty_like(k_chunk_0);
    auto v_chunk_1 = mfq::cuda::empty_like(k_chunk_0);
    const auto k_pointer_0 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(k_chunk_0.data_ptr()));
    const auto v_pointer_0 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(v_chunk_0.data_ptr()));
    const auto k_pointer_1 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(k_chunk_1.data_ptr()));
    const auto v_pointer_1 = static_cast<std::int64_t>(
        reinterpret_cast<std::intptr_t>(v_chunk_1.data_ptr()));
    auto k_chunks = mfq::cuda::tensor<std::int64_t>({
        k_pointer_0, k_pointer_1}).to(gpu);
    auto v_chunks = mfq::cuda::tensor<std::int64_t>({
        v_pointer_0, v_pointer_1}).to(gpu);
    // Logical pages deliberately map to non-contiguous physical pages and
    // cross the chunk boundary at physical page 4.
    auto page_table = mfq::cuda::tensor<std::int32_t>({
        2,PagesPerChunk,-1,-1,PagesPerChunk+1,3,-1,-1,
    }).reshape({B,LogicalPages}).to(gpu).contiguous();
    std::vector<int64_t> position_values(B*T);
    for(int b=0;b<B;++b)for(int t=0;t<T;++t)position_values[b*T+t]=t;
    auto positions=mfq::cuda::tensor<int64_t>(position_values).reshape({B,T}).to(gpu).contiguous();
    paged_kv_cache_write_cuda(
        k_chunks, v_chunks, page_table,
        k.narrow(2, 0, 3).contiguous(), v.narrow(2, 0, 3).contiguous(),
        positions.narrow(1, 0, 3).contiguous(),
        Page, PagesPerChunk);
    auto first_k = mfq::cuda::empty({B, Hk, 3, D},
        TensorOptions().device(gpu).dtype(dtype));
    auto first_v = mfq::cuda::empty_like(first_k);
    paged_kv_cache_gather_cuda(k_chunks, v_chunks, page_table,
        first_k, first_v, Page, PagesPerChunk);
    require(float_values(first_k) == float_values(k.narrow(2, 0, 3)),
        "Paged first-chunk key gather differs from source");
    require(float_values(first_v) == float_values(v.narrow(2, 0, 3)),
        "Paged first-chunk value gather differs from source");
    paged_kv_cache_write_cuda(k_chunks, v_chunks, page_table,
        k.narrow(2, 3, T-3).contiguous(), v.narrow(2, 3, T-3).contiguous(),
        positions.narrow(1, 3, T-3).contiguous(), Page, PagesPerChunk);
    auto gathered_k = mfq::cuda::empty_like(k);
    auto gathered_v = mfq::cuda::empty_like(v);
    paged_kv_cache_gather_cuda(k_chunks, v_chunks, page_table,
        gathered_k, gathered_v, Page, PagesPerChunk);
    require(float_values(gathered_k) == float_values(k),
        "Paged complete-prefix key gather differs from source");
    require(float_values(gathered_v) == float_values(v),
        "Paged complete-prefix value gather differs from source");

    auto lengths = mfq::cuda::tensor<std::int64_t>({T,T-2}).to(gpu);
    auto partial_o = mfq::cuda::empty(
        {B * Hq, 4, D},
        TensorOptions().device(gpu).dtype(mfq::cuda::kFloat32));
    auto partial_m = mfq::cuda::empty(
        {B * Hq, 4},
        TensorOptions().device(gpu).dtype(mfq::cuda::kFloat32));
    auto partial_l = mfq::cuda::empty_like(partial_m);
    constexpr double Scale = 0.0625;
    auto direct = attention_paged_cache_decode_cuda(
        q, k_chunks, v_chunks, page_table, lengths, Scale,
        Page, PagesPerChunk, Hk,
        partial_o, partial_m, partial_l, 1, false);
    auto split = attention_paged_cache_decode_cuda(
        q, k_chunks, v_chunks, page_table, lengths, Scale,
        Page, PagesPerChunk, Hk,
        partial_o, partial_m, partial_l, 3, false);
    auto dynamic = attention_paged_cache_decode_cuda(
        q, k_chunks, v_chunks, page_table, lengths, Scale,
        Page, PagesPerChunk, Hk,
        partial_o, partial_m, partial_l, 4, true);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceSynchronize());

    const auto rounded_k = float_values(k);
    const auto rounded_v = float_values(v);
    const auto rounded_q = float_values(q);
    const auto direct_values = float_values(direct);
    const auto split_values = float_values(split);
    const auto dynamic_values = float_values(dynamic);
    for (int batch = 0; batch < B; ++batch) {
        const int token_count = batch == 0 ? T : T-2;
        for (int query_head = 0; query_head < Hq; ++query_head) {
            const int kv_head = query_head / (Hq / Hk);
            std::vector<double> scores(token_count);
            double maximum = -1e300;
            for (int token = 0; token < token_count; ++token) {
                double dot = 0.0;
                for (int dim = 0; dim < D; ++dim) {
                    const auto q_index =
                        (static_cast<size_t>(batch) * Hq + query_head) * D + dim;
                    const auto kv_index =
                        ((static_cast<size_t>(batch) * Hk + kv_head) * T +
                         token) * D + dim;
                    dot += static_cast<double>(rounded_q[q_index]) *
                        rounded_k[kv_index];
                }
                scores[token] = dot * Scale;
                maximum = std::max(maximum, scores[token]);
            }
            double denominator = 0.0;
            for (const auto score : scores) {
                denominator += std::exp(score - maximum);
            }
            for (int dim = 0; dim < D; ++dim) {
                double expected = 0.0;
                for (int token = 0; token < token_count; ++token) {
                    const auto kv_index =
                        ((static_cast<size_t>(batch) * Hk + kv_head) * T +
                         token) * D + dim;
                    expected += std::exp(scores[token] - maximum) *
                        rounded_v[kv_index];
                }
                expected /= denominator;
                const auto output_index =
                    (static_cast<size_t>(batch) * Hq + query_head) * D + dim;
                const auto error = std::abs(
                    direct_values[output_index] - expected);
                if (!std::isfinite(direct_values[output_index]) ||
                        error > 2.5e-3 ||
                        std::abs(split_values[output_index] -
                            direct_values[output_index]) > 2.5e-3 ||
                        std::abs(dynamic_values[output_index] -
                            direct_values[output_index]) > 2.5e-3) {
                    throw std::runtime_error(
                        "Paged KV attention numerical mismatch at " +
                        std::to_string(output_index));
                }
            }
        }
    }
    lengths.fill_(0);
    auto empty=attention_paged_cache_decode_cuda(q,k_chunks,v_chunks,page_table,lengths,Scale,
        Page,PagesPerChunk,Hk,partial_o,partial_m,partial_l,3,false);
    for(float value:float_values(empty))require(value==0,"Empty paged attention must be zero");
    lengths.fill_(1);
    auto first=attention_paged_cache_decode_cuda(q,k_chunks,v_chunks,page_table,lengths,Scale,
        Page,PagesPerChunk,Hk,partial_o,partial_m,partial_l,4,true);
    auto first_values=float_values(first);
    for(int b=0;b<B;++b)for(int h=0;h<Hq;++h)for(int d=0;d<D;++d)
        require(std::abs(first_values[(b*Hq+h)*D+d]-rounded_v[((b*Hk+h/ratio)*T)*D+d])<1e-6,
            "Single-token paged attention differs from its value");

}

} // namespace

int main() try {
    std::cout.setf(std::ios::unitbuf);
    int devices = 0;
    const auto status = cudaGetDeviceCount(&devices);
    if (status != cudaSuccess || devices == 0) {
        std::cerr << "paged_kv_test SKIP: " << cudaGetErrorString(status)
                  << " devices=" << devices << '\n';
        (void)cudaGetLastError();
        return 77;
    }
    for(auto dtype:{mfq::cuda::kFloat16,mfq::cuda::kBFloat16,mfq::cuda::kFloat32}) {
        for(int width:{63,64,65,128,256,512})for(int ratio:{1,3,6})
            check_kernels(width,ratio,4,4,dtype);
        check_kernels(256,4,4,4,dtype);
        check_kernels(256,4,16,64,dtype);
        check_kernels(256,6,16,64,dtype);
        check_kernels(256,6,100,4,dtype);
    }
    std::cout << "paged_kv_test PASS 66 geometries, scalar/vector dimensions, arbitrary GQA, FP16/BF16/FP32, empty/single/split/dynamic\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
