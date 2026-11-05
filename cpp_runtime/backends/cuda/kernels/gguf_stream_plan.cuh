#pragma once
#include "gguf.h"
namespace mfq::cuda::gguf::stream_plan_detail {
struct Header {std::uint64_t count;std::uint32_t abort;int group_count;};
__device__ __forceinline__ void read_header(const StreamPlanSnapshot& p,Header& state) {
    if(threadIdx.x)return;
    const auto* source=static_cast<const std::uint64_t*>(p.mapped);
    auto* destination=static_cast<std::uint64_t*>(p.snapshot);
    state.count=source[0];const bool empty=state.count==~std::uint64_t(0);
    if(empty)state.count=0;
    state.abort=*reinterpret_cast<const volatile std::uint32_t*>(p.cancelled) || state.count>p.capacity;
    destination[0]=state.abort?0:state.count;destination[1]=source[1];
    destination[2]=state.abort?1u:empty?2u:0u;destination[3]=state.abort || empty?0:source[3];
    state.group_count=p.host_groups?p.host_groups[0].count:0;
    if(state.group_count<0 || state.group_count>p.group_capacity)state.group_count=0;
    if(p.device_groups)p.device_groups[0].count=state.abort?0:state.group_count;
}
// The caller synchronizes after read_header. Keep legacy and fused snapshots
// identical for cancellation, empty plans, payload bounds and CPU fallback.
__device__ __forceinline__ void copy_payload(const StreamPlanSnapshot& p,const Header& state) {
    for(auto i=std::int64_t(threadIdx.x);i<p.slots;i+=blockDim.x)
        p.device_slots[i]=state.abort?0:reinterpret_cast<const volatile std::int32_t*>(p.cpu_slots)[i];
    if(state.abort)return;
    if(p.host_groups && p.device_groups) {
        const auto* input=reinterpret_cast<const volatile std::uint32_t*>(p.host_groups+1);
        auto* output=reinterpret_cast<std::uint32_t*>(p.device_groups+1);
        for(auto i=std::int64_t(threadIdx.x);i<std::int64_t(state.group_count)*sizeof(RouteGroup)/4;i+=blockDim.x)
            output[i]=input[i];
    }
    const auto* source=static_cast<const std::uint64_t*>(p.mapped);
    auto* destination=static_cast<std::uint64_t*>(p.snapshot);
    for(auto word=std::uint64_t(threadIdx.x);word<state.count*4;word+=blockDim.x)
        destination[4+word]=source[4+word];
}
}
