#include <cuda_runtime.h>
#include "gguf_stream_plan.cuh"
#include "kernel_dispatch_trace.h"
#include <stdexcept>
namespace mfq::cuda::gguf {
namespace {
__global__ void snapshot_stream_plan_kernel(StreamPlanSnapshot p) {
    __shared__ stream_plan_detail::Header header;
    if(!threadIdx.x && p.ready) {
        while(!*reinterpret_cast<const volatile std::uint32_t*>(p.ready)) {
#if __CUDA_ARCH__ >= 700
            __nanosleep(64);
#endif
        }
        __threadfence_system();
    }
    __syncthreads();
    // Warp zero reads plan metadata while other warps fetch the selected
    // expert pointers. Their common publication gate is observed only once.
    if(threadIdx.x>=32 && p.mapped_addresses) {
        for(auto slot=std::int64_t(threadIdx.x-32);slot<p.selected_slots;slot+=blockDim.x-32) {
            const auto expert=p.ids[slot];bool first=expert>=0 && expert<p.experts;
            for(std::int64_t i=0;first && i<slot;++i)first=p.ids[i]!=expert;
            if(first)for(int role=0;role<3;++role)
                p.addresses[role*p.experts+expert]=p.mapped_addresses[role*p.experts+expert];
        }
    }
    stream_plan_detail::read_header(p,header);
    __syncthreads();
    stream_plan_detail::copy_payload(p,header);
}
}
void snapshot_stream_plan(const StreamPlanSnapshot& p,cudaStream_t stream) {
    snapshot_stream_plan_kernel<<<1,256,0,stream>>>(p);
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_stream_plan_snapshot",
        reinterpret_cast<const void*>(snapshot_stream_plan_kernel),"wait_and_metadata=1 address_snapshot="+
        std::to_string(p.mapped_addresses!=nullptr)+" launches=1");
}
}
