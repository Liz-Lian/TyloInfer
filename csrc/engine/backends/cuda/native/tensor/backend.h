#pragma once
#include "env.h"
#include "engine/tensor.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <optional>
#include <atomic>
#include <algorithm>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>
#include <utility>


#include "engine/backends/cpu/native/host_parallel.h"
#include "engine/backends/cuda/native/device.h"
#include "context.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cuda/native/graph.h"
#include "tensor.h"

using mfq_half = __half;
using mfq_bfloat16 = __nv_bfloat16;
using MfqBackendError = ::tyloi::cuda::Error;
inline constexpr auto mfq_float8_e4m3fn = ::tyloi::kFloat8E4M3FN;
inline constexpr auto mfq_dispatch_half = ::tyloi::ScalarType::float16;
inline constexpr auto mfq_dispatch_bfloat16 = ::tyloi::ScalarType::bfloat16;

inline cudaStream_t mfq_current_cuda_stream() {
    return ::tyloi::cuda::current_stream().stream();
}

inline cublasHandle_t mfq_current_cublas_handle() {
    int device = 0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    auto context = ::tyloi::cuda::default_context(device);
    context->blas().set_stream(::tyloi::cuda::current_stream(device).stream());
    return context->blas().get();
}

inline void mfq_cuda_synchronize() {
    int device = 0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(
        ::tyloi::cuda::current_stream(device).stream()));
}

template <typename... Values>
[[noreturn]] inline void mfq_runtime_check_failed(Values&&... values) {
    std::ostringstream message;
    (message << ... << std::forward<Values>(values));
    throw std::runtime_error(message.str());
}

#define TYLOINFER_RUNTIME_CHECK(condition, ...) \
    do { \
        if (!(condition)) { \
            ::mfq_runtime_check_failed(__VA_ARGS__); \
        } \
    } while (false)

#define TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK() \
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError())

#define TYLOINFER_CUDA_CHECK(expression) TYLOINFER_NATIVE_CUDA_CHECK(expression)

#define TYLOINFER_DISPATCH_FLOATING_TYPES_AND_HALF(TYPE, NAME, ...) \
    ([&]() { \
        switch (TYPE) { \
            case ::tyloi::ScalarType::float64: { using scalar_t = double; return (__VA_ARGS__)(); } \
            case ::tyloi::ScalarType::float32: { using scalar_t = float; return (__VA_ARGS__)(); } \
            case ::tyloi::ScalarType::float16: { using scalar_t = __half; return (__VA_ARGS__)(); } \
            default: ::mfq_runtime_check_failed(NAME, ": unsupported floating dtype"); \
        } \
    })()

#define TYLOINFER_DISPATCH_FLOATING_TYPES_AND2(TYPE1, TYPE2, TYPE, NAME, ...) \
    ([&]() { \
        switch (TYPE) { \
            case ::tyloi::ScalarType::float64: { using scalar_t = double; return (__VA_ARGS__)(); } \
            case ::tyloi::ScalarType::float32: { using scalar_t = float; return (__VA_ARGS__)(); } \
            case ::tyloi::ScalarType::float16: { using scalar_t = __half; return (__VA_ARGS__)(); } \
            case ::tyloi::ScalarType::bfloat16: { using scalar_t = __nv_bfloat16; return (__VA_ARGS__)(); } \
            default: ::mfq_runtime_check_failed(NAME, ": unsupported floating dtype"); \
        } \
    })()

class MfqCudaGuard final {
public:
    explicit MfqCudaGuard(int device) : guard_(device) {}
    explicit MfqCudaGuard(const ::tyloi::Device& device) : guard_(device.index) {}

private:
    ::tyloi::cuda::DeviceGuard guard_;
};

using MfqCudaStream = ::tyloi::cuda::StreamHandle;
using MfqCudaStreamGuard = ::tyloi::cuda::StreamGuard;
using MfqCudaGraph = ::tyloi::cuda::Graph;

inline void mfq_prepare_cuda_graph_memory(MfqCudaGraph& graph) {
    graph.prepare_memory();
}

inline void mfq_prepare_cuda_graph_memory(
        MfqCudaGraph& graph,
        const std::vector<MfqCudaStream>& participant_streams) {
    graph.prepare_memory(participant_streams);
}

inline MfqCudaStream mfq_get_current_cuda_stream(int device = -1) {
    return ::tyloi::cuda::current_stream(device);
}

inline MfqCudaStream mfq_get_stream_from_pool(
    bool high_priority = false,
    int device = -1) {
    return ::tyloi::cuda::stream_from_pool(high_priority, device);
}

inline int mfq_current_cuda_device() {
    int device = 0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

inline void mfq_cuda_empty_cache() {
    ::tyloi::cuda::default_context(mfq_current_cuda_device())->trim();
}

inline bool mfq_cuda_graph_capture_supported() {
    return ::tyloi::cuda::default_context(
        mfq_current_cuda_device())->supports_async_allocations();
}

inline void mfq_cuda_record_stream(
    const ::tyloi::TensorImpl& tensor,
    const MfqCudaStream& stream) {
    tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()));
}

inline void mfq_cuda_record_stream(
    const ::tyloi::Tensor& tensor,
    const MfqCudaStream& stream) {
    tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()));
}

struct MfqCudaMemoryStats {
    std::uint64_t allocated_bytes = 0;
    std::uint64_t active_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t inactive_split_bytes = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t peak_allocated_bytes = 0;
    std::uint64_t peak_reserved_bytes = 0;
    std::uint64_t allocations = 0;
    std::uint64_t segments = 0;
    std::uint64_t retries = 0;
    std::uint64_t ooms = 0;
};

inline MfqCudaMemoryStats mfq_cuda_memory_stats(int device) {
    ::tyloi::cuda::DeviceGuard guard(device);
    const auto context=::tyloi::cuda::default_context(device);
    const auto owned=context->memory_stats();
    MfqCudaMemoryStats stats;
    stats.allocated_bytes = owned.allocated;
    stats.active_bytes = stats.allocated_bytes;
    stats.reserved_bytes = context->resident_memory_usage();
    stats.requested_bytes = stats.allocated_bytes;
    stats.peak_allocated_bytes = owned.peak;
    stats.peak_reserved_bytes = std::max(stats.reserved_bytes,stats.peak_allocated_bytes);
    return stats;
}

inline std::atomic<int> mfq_host_thread_count{
    static_cast<int>(std::max(1u, std::thread::hardware_concurrency()))};

inline int mfq_get_num_threads() {
    return mfq_host_thread_count.load(std::memory_order_relaxed);
}

inline void mfq_set_num_threads(int count) {
    mfq_host_thread_count.store(std::max(1, count), std::memory_order_relaxed);
}

template <typename Function>
void mfq_parallel_for(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain,
    Function&& function) {
    ::tyloi::host_parallel_for(begin, end, grain, mfq_get_num_threads(),
        std::forward<Function>(function));
}

inline void mfq_disable_tf32_cublas() {
    auto context = ::tyloi::cuda::default_context(mfq_current_cuda_device());
    context->blas().set_stream(mfq_current_cuda_stream());
    TYLOINFER_NATIVE_CUDA_CHECK(cublasSetMathMode(
        context->blas().get(), CUBLAS_DEFAULT_MATH));
}

inline void mfq_cuda_manual_seed_all(std::uint64_t seed) {
    ::tyloi::cuda::manual_seed(static_cast<std::int64_t>(seed));
}


template <typename Value>
using MfqOptional = std::optional<Value>;

inline constexpr std::nullopt_t mfq_nullopt = std::nullopt;

inline ::tyloi::DenseTensor mfq_linear(
    const ::tyloi::DenseTensor& input,
    const ::tyloi::DenseTensor& weight,
    const std::optional<::tyloi::DenseTensor>& bias = std::nullopt) {
    return ::tyloi::cuda::linear(input, weight, bias);
}

inline ::tyloi::DenseTensor mfq_linear(
    const ::tyloi::DenseTensor& input,
    const ::tyloi::DenseTensor& weight,
    const ::tyloi::DenseTensor& bias) {
    return mfq_linear(
        input, weight, std::optional<::tyloi::DenseTensor>(bias));
}

inline ::tyloi::DenseTensor mfq_scaled_dot_product_attention(
    const ::tyloi::DenseTensor& query,
    const ::tyloi::DenseTensor& key,
    const ::tyloi::DenseTensor& value,
    const std::optional<::tyloi::DenseTensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention) {
    return ::tyloi::cuda::scaled_dot_product_attention(
        query, key, value, mask, dropout, causal, scale,
        enable_grouped_query_attention);
}
