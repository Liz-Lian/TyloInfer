#include <cuda_runtime.h>
#include "gguf_stream_plan.cuh"
#include "kernel_dispatch_trace.h"
#include "system_ready.cuh"
#include <stdexcept>
namespace mfq::cuda::gguf {
namespace {
__device__ __forceinline__ void wait_ready(const std::uint32_t* ready) {
    mfq::cuda::detail::wait_system_ready(ready);
}
template<bool ResidentBypass>
__global__ void snapshot_stream_plan_kernel(StreamPlanSnapshot p,ResidentStreamPlan resident) {
    __shared__ stream_plan_detail::Header header;
    if constexpr(ResidentBypass) {
        if(!threadIdx.x)wait_ready(resident.ready);
        __syncthreads();
        bool missing=!resident.addresses || !p.addresses || !p.ids || p.experts<=0 ||
            p.selected_slots<=0 || p.cpu_slots || p.device_slots || p.slots ||
            (p.cancelled && *reinterpret_cast<const volatile std::uint32_t*>(p.cancelled));
        if(!missing)for(auto slot=std::int64_t(threadIdx.x);slot<p.selected_slots;slot+=blockDim.x) {
            const auto expert=p.ids[slot];
            if(expert<0 || expert>=p.experts){missing=true;break;}
            const auto* addresses=reinterpret_cast<const volatile std::uint64_t*>(resident.addresses);
            for(int role=0;role<3;++role)if(!addresses[role*p.experts+expert])missing=true;
        }
        if(!__syncthreads_or(missing)) {
            // All three projections are already owned by the resident pass.
            // Publish an empty streamed pass without reading a CPU packet
            // which may still describe the preceding replay.
            if(!threadIdx.x) {
                auto* output=static_cast<std::uint64_t*>(p.snapshot);
                for(int word=0;word<4;++word)output[word]=0;
                if(p.device_groups)p.device_groups[0].count=0;
            }
            for(auto slot=std::int64_t(threadIdx.x);slot<p.selected_slots;slot+=blockDim.x) {
                const auto expert=p.ids[slot];bool first=true;
                for(std::int64_t previous=0;first && previous<slot;++previous)first=p.ids[previous]!=expert;
                if(first)for(int role=0;role<3;++role)p.addresses[role*p.experts+expert]=0;
            }
            __syncthreads();
            if(!threadIdx.x) {
                __threadfence_system();
                if(resident.gate_up_done)*reinterpret_cast<volatile std::uint32_t*>(resident.gate_up_done)=1;
                if(resident.down_done)*reinterpret_cast<volatile std::uint32_t*>(resident.down_done)=1;
                __threadfence_system();
            }
            return;
        }
    }
    if(!threadIdx.x)wait_ready(p.ready);
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
    snapshot_stream_plan_kernel<false><<<1,256,0,stream>>>(p,{});
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_stream_plan_snapshot",
        reinterpret_cast<const void*>(snapshot_stream_plan_kernel<false>),"wait_and_metadata=1 address_snapshot="+
        std::to_string(p.mapped_addresses!=nullptr)+" launches=1");
}
void snapshot_stream_plan(const StreamPlanSnapshot& p,const ResidentStreamPlan& resident,cudaStream_t stream) {
    snapshot_stream_plan_kernel<true><<<1,256,0,stream>>>(p,resident);
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
    if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_stream_plan_snapshot",
        reinterpret_cast<const void*>(snapshot_stream_plan_kernel<true>),"resident_only_bypass=1 wait_on_miss=1 launches=1");
}
}
