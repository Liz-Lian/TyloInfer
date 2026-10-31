#include "device.h"

#include <sstream>
#include <string>

namespace mfq::cuda {
namespace {

std::string location_message(
    const char* category,
    const char* detail,
    const char* expression,
    const char* file,
    int line) {
    std::ostringstream message;
    message << category << " error: " << detail << " (" << expression << ") at "
            << file << ':' << line;
    return message.str();
}

const char* cublas_status_name(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "unknown cuBLAS status";
}

}  // namespace

void check(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw Error(location_message(
            "CUDA", cudaGetErrorString(status), expression, file, line));
    }
}

void check(cublasStatus_t status, const char* expression, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw Error(location_message(
            "cuBLAS", cublas_status_name(status), expression, file, line));
    }
}

DeviceGuard::DeviceGuard(int device) {
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&previous_));
    if (previous_ != device) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaSetDevice(device));
        restore_ = true;
    }
}

DeviceGuard::~DeviceGuard() noexcept {
    if (restore_) {
        (void)cudaSetDevice(previous_);
    }
}

}  // namespace mfq::cuda
