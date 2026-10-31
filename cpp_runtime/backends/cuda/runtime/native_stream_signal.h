#pragma once
#include "mfq_cuda_context.h"
#include <cuda.h>

namespace mfq::cuda {
// A stream memory operation does not launch a completion kernel behind a
// graph's mapped-memory waiter. Registered host flags require their device
// alias; managed memory is not accepted by this driver API.
// https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html
class NativeStreamSignal {
    using Function=decltype(&cuStreamWriteValue32);
    static Function entry() {
#if CUDA_VERSION >= 12000 && CUDART_VERSION >= 12000
        static const auto function=[]() -> Function {
            void* raw=nullptr;
            cudaDriverEntryPointQueryResult query=cudaDriverEntryPointSymbolNotFound;
#if CUDART_VERSION >= 12050
            const auto status=cudaGetDriverEntryPointByVersion("cuStreamWriteValue32",&raw,12000,cudaEnableDefault,&query);
#else
            const auto status=cudaGetDriverEntryPoint("cuStreamWriteValue32",&raw,cudaEnableDefault,&query);
#endif
            if(status==cudaErrorNotSupported){cudaGetLastError();return nullptr;}
            TYLOINFER_NATIVE_CUDA_CHECK(status);
            return query==cudaDriverEntryPointSuccess?reinterpret_cast<Function>(raw):nullptr;
        }();
        return function;
#else
        return nullptr;
#endif
    }
public:
    static bool submit(std::uint32_t* mapped,cudaStream_t stream) {
        const auto write=entry();
        if(!write || !mapped)return false;
        const auto result=write(reinterpret_cast<CUstream>(stream),reinterpret_cast<CUdeviceptr>(mapped),1,0);
        if(result==CUDA_ERROR_NOT_SUPPORTED)return false;
        if(result!=CUDA_SUCCESS)throw std::runtime_error("CUDA stream completion signal failed: "+std::to_string(result));
        return true;
    }
};
}
