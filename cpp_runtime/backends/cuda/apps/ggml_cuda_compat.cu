#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime_api.h>
#include "common.cuh"

// MFQ's custom kernels use the integrated GGML CUDA helper headers without
// linking the complete ggml-cuda backend. Provide the host symbols they need.
void ggml_cuda_error(
    const char * statement,
    const char * function,
    const char * file,
    int line,
    const char * message) {
    std::fprintf(
        stderr,
        "CUDA error: %s\n  statement: %s\n  function: %s\n  at %s:%d\n",
        message != nullptr ? message : "unknown",
        statement != nullptr ? statement : "unknown",
        function != nullptr ? function : "unknown",
        file != nullptr ? file : "unknown",
        line);
    std::abort();
}

int ggml_cuda_get_device() {
    int device = 0;
    const cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        ggml_cuda_error(
            "cudaGetDevice(&device)",
            __func__,
            __FILE__,
            __LINE__,
            cudaGetErrorString(status));
    }
    return device;
}

void ggml_cuda_set_device(int device) { CUDA_CHECK(cudaSetDevice(device)); }

const ggml_cuda_device_info& ggml_cuda_info() {
    // Matches Strata's MMQ host adapter; query the actual device capabilities.
    static const ggml_cuda_device_info info=[] {
        ggml_cuda_device_info value{};
        int count=0;CUDA_CHECK(cudaGetDeviceCount(&count));
        value.device_count=value.physical_device_count=std::min(count,GGML_CUDA_MAX_DEVICES);
        for(int id=0;id<value.device_count;++id) {
            auto attribute=[&](cudaDeviceAttr which) {int result=0;CUDA_CHECK(cudaDeviceGetAttribute(&result,which,id));return result;};
            auto& d=value.devices[id];
            d.cc=100*attribute(cudaDevAttrComputeCapabilityMajor)+10*attribute(cudaDevAttrComputeCapabilityMinor);
            d.smpbo=attribute(cudaDevAttrMaxSharedMemoryPerBlockOptin);
            d.smpb=attribute(cudaDevAttrMaxSharedMemoryPerBlock);
            d.integrated=attribute(cudaDevAttrIntegrated)!=0;
            d.supports_cooperative_launch=attribute(cudaDevAttrCooperativeLaunch)!=0;
            d.nsm=attribute(cudaDevAttrMultiProcessorCount);d.warp_size=attribute(cudaDevAttrWarpSize);
            d.physical_device=id;d.physical_share_count=1;d.virtual_index=0;
            cudaDeviceProp properties{};CUDA_CHECK(cudaGetDeviceProperties(&properties,id));
            d.total_vram=properties.totalGlobalMem;
        }
        return value;
    }();
    return info;
}
