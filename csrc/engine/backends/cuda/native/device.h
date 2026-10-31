#pragma once

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <stdexcept>

namespace mfq::cuda {

class Error final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

void check(cudaError_t status, const char* expression, const char* file, int line);
void check(cublasStatus_t status, const char* expression, const char* file, int line);

#define TYLOINFER_NATIVE_CUDA_CHECK(expression) \
    ::mfq::cuda::check((expression), #expression, __FILE__, __LINE__)

class DeviceGuard final {
public:
    explicit DeviceGuard(int device);
    ~DeviceGuard() noexcept;

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int previous_ = 0;
    bool restore_ = false;
};

namespace internal {

template <typename Function>
void on_device_noexcept(int device, Function&& function) noexcept {
    int previous = 0;
    if (cudaGetDevice(&previous) != cudaSuccess) return;
    const bool restore = previous != device;
    if (restore && cudaSetDevice(device) != cudaSuccess) return;
    try {
        function();
    } catch (...) {
        // Resource destruction must never terminate inference during unwinding.
    }
    if (restore) (void)cudaSetDevice(previous);
}

}  // namespace internal

}  // namespace mfq::cuda
