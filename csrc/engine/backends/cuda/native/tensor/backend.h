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

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME

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

inline void mfq_debug_dump_cuda_graph(MfqCudaGraph&) {}

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

#else

#include <ATen/Parallel.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using mfq_half = ::at::Half;
using mfq_bfloat16 = ::at::BFloat16;
using MfqBackendError = ::c10::Error;
inline constexpr auto mfq_float8_e4m3fn = ::at::kFloat8_e4m3fn;
inline constexpr auto mfq_dispatch_half = ::at::ScalarType::Half;
inline constexpr auto mfq_dispatch_bfloat16 = ::at::ScalarType::BFloat16;

inline cudaStream_t mfq_current_cuda_stream() {
    return ::at::cuda::getCurrentCUDAStream();
}

inline cublasHandle_t mfq_current_cublas_handle() {
    return ::at::cuda::getCurrentCUDABlasHandle();
}

inline void mfq_cuda_synchronize() {
    ::torch::cuda::synchronize();
}

#define TYLOINFER_RUNTIME_CHECK TORCH_CHECK
#define TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK C10_CUDA_KERNEL_LAUNCH_CHECK
#define TYLOINFER_CUDA_CHECK C10_CUDA_CHECK
#define TYLOINFER_DISPATCH_FLOATING_TYPES_AND_HALF AT_DISPATCH_FLOATING_TYPES_AND_HALF
#define TYLOINFER_DISPATCH_FLOATING_TYPES_AND2 AT_DISPATCH_FLOATING_TYPES_AND2
using MfqCudaGuard = ::c10::cuda::CUDAGuard;
using MfqCudaStream = ::at::cuda::CUDAStream;
using MfqCudaStreamGuard = ::c10::cuda::CUDAStreamGuard;
using MfqCudaGraph = ::at::cuda::CUDAGraph;

inline void mfq_prepare_cuda_graph_memory(MfqCudaGraph& graph) {
    const char* debug_path = tyloi::cuda::env_lookup("TYLOINFER_TORCH_CUDA_GRAPH_DUMP");
    if (debug_path != nullptr && debug_path[0] != '\0') {
        graph.enable_debug_mode();
    }
}

inline void mfq_prepare_cuda_graph_memory(
        MfqCudaGraph& graph,
        const std::vector<MfqCudaStream>&) {
    mfq_prepare_cuda_graph_memory(graph);
}

inline void mfq_debug_dump_cuda_graph(MfqCudaGraph& graph) {
    const char* debug_path = tyloi::cuda::env_lookup("TYLOINFER_TORCH_CUDA_GRAPH_DUMP");
    if (debug_path != nullptr && debug_path[0] != '\0') {
        TYLOINFER_CUDA_CHECK(cudaGraphDebugDotPrint(
            graph.raw_cuda_graph(), debug_path, cudaGraphDebugDotFlagsVerbose));
    }
}

inline MfqCudaStream mfq_get_current_cuda_stream(int device = -1) {
    return device < 0
        ? ::at::cuda::getCurrentCUDAStream()
        : ::at::cuda::getCurrentCUDAStream(device);
}

inline MfqCudaStream mfq_get_stream_from_pool(
    bool high_priority = false,
    int device = -1) {
    return device < 0
        ? ::at::cuda::getStreamFromPool(high_priority)
        : ::at::cuda::getStreamFromPool(high_priority, device);
}

inline int mfq_current_cuda_device() {
    return ::c10::cuda::current_device();
}

inline void mfq_cuda_empty_cache() {
    ::c10::cuda::CUDACachingAllocator::emptyCache();
}

inline bool mfq_cuda_graph_capture_supported() { return true; }

inline void mfq_cuda_record_stream(
    const ::torch::Tensor& tensor,
    const MfqCudaStream& stream) {
    ::c10::cuda::CUDACachingAllocator::recordStream(
        tensor.storage().data_ptr(), stream);
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
    const auto source = ::c10::cuda::CUDACachingAllocator::getDeviceStats(device);
    constexpr auto aggregate = static_cast<std::size_t>(
        ::c10::CachingDeviceAllocator::StatType::AGGREGATE);
    return {
        static_cast<std::uint64_t>(source.allocated_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.active_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.reserved_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.inactive_split_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.requested_bytes[aggregate].current),
        static_cast<std::uint64_t>(source.allocated_bytes[aggregate].peak),
        static_cast<std::uint64_t>(source.reserved_bytes[aggregate].peak),
        static_cast<std::uint64_t>(source.allocation[aggregate].current),
        static_cast<std::uint64_t>(source.segment[aggregate].current),
        static_cast<std::uint64_t>(source.num_alloc_retries),
        static_cast<std::uint64_t>(source.num_ooms),
    };
}

inline int mfq_get_num_threads() { return ::at::get_num_threads(); }
inline void mfq_set_num_threads(int count) { ::at::set_num_threads(count); }

template <typename Function>
void mfq_parallel_for(
    std::int64_t begin,
    std::int64_t end,
    std::int64_t grain,
    Function&& function) {
    ::at::parallel_for(begin, end, grain, std::forward<Function>(function));
}

inline void mfq_disable_tf32_cublas() {
    ::at::globalContext().setAllowTF32CuBLAS(false);
}


inline void mfq_cuda_manual_seed_all(std::uint64_t seed) {
    ::torch::cuda::manual_seed_all(seed);
}

#endif

template <typename Value>
using MfqOptional = std::optional<Value>;

inline constexpr std::nullopt_t mfq_nullopt = std::nullopt;

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
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
#else
inline ::at::Tensor mfq_linear(
    const ::at::Tensor& input,
    const ::at::Tensor& weight,
    const std::optional<::at::Tensor>& bias = std::nullopt) {
    return ::at::linear(input, weight, bias);
}

inline ::at::Tensor mfq_linear(
    const ::at::Tensor& input,
    const ::at::Tensor& weight,
    const ::at::Tensor& bias) {
    return mfq_linear(input, weight, std::optional<::at::Tensor>(bias));
}

inline ::at::Tensor mfq_scaled_dot_product_attention(
    const ::at::Tensor& query,
    const ::at::Tensor& key,
    const ::at::Tensor& value,
    const std::optional<::at::Tensor>& mask,
    double dropout,
    bool causal,
    const std::optional<double>& scale,
    bool enable_grouped_query_attention) {
    return ::at::scaled_dot_product_attention(
        query, key, value, mask, dropout, causal, scale,
        enable_grouped_query_attention);
}
#endif
