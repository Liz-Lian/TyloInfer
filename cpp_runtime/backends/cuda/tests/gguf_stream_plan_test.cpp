#include "../kernels/gguf.h"
#include "../runtime/moe_pipeline.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace gg=mfq::cuda::gguf;
void check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Device {
    T* data=nullptr;
    std::size_t count;
    explicit Device(std::size_t count):count(count) {check(cudaMalloc(&data,count*sizeof(T)));}
    ~Device(){cudaFree(data);}
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
template<class T> struct Mapped {
    T* host=nullptr;
    T* device=nullptr;
    std::size_t count;
    explicit Mapped(std::size_t count):count(count) {
        check(cudaHostAlloc(&host,count*sizeof(T),cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&device,host,0));
    }
    ~Mapped(){cudaFreeHost(host);}
    Mapped(const Mapped&)=delete;
    Mapped& operator=(const Mapped&)=delete;
};
struct Stream {
    cudaStream_t value=nullptr;
    Stream(){check(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking));}
    ~Stream(){cudaStreamDestroy(value);}
};
struct Graph {
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}
};
template<class T> std::vector<T> read(const Device<T>& source,cudaStream_t stream) {
    std::vector<T> result(source.count);
    check(cudaMemcpyAsync(result.data(),source.data,result.size()*sizeof(T),cudaMemcpyDeviceToHost,stream));
    check(cudaStreamSynchronize(stream));return result;
}
template<class T> void equal(const Device<T>& a,const Device<T>& b,cudaStream_t stream) {
    const auto x=read(a,stream),y=read(b,stream);
    if(x.size()!=y.size() || std::memcmp(x.data(),y.data(),x.size()*sizeof(T)))
        throw std::runtime_error("Fused stream plan changed snapshot bytes");
}
template<class T> void initialize(Device<T>& value,cudaStream_t stream) {
    check(cudaMemsetAsync(value.data,0xa5,value.count*sizeof(T),stream));
}

void run(int slots,int variant,bool cpu,bool grouped,bool addressed,int& cases,int& delayed) {
    const int experts=slots<40?17:512,capacity=std::max(1,slots*3);
    const int groups=std::min(slots,13),words=4+capacity*4;
    const std::uint64_t count=variant==0?0:variant==2?std::uint64_t(capacity+1):
        variant==3?~std::uint64_t(0):std::uint64_t(std::min(capacity,3));
    Stream stream;
    Mapped<std::uint64_t> source(words),addresses(3*experts);
    Mapped<std::uint32_t> flags(2);
    Mapped<std::int32_t> mask(std::max(slots,1));
    Mapped<gg::RouteGroup> route(groups+1);
    Device<std::int64_t> ids(std::max(slots,1));
    Device<std::uint64_t> old(words),fresh(words),old_addresses(3*experts),fresh_addresses(3*experts);
    Device<std::int32_t> old_mask(std::max(slots,1)),fresh_mask(std::max(slots,1));
    Device<gg::RouteGroup> old_groups(groups+1),fresh_groups(groups+1);
    std::vector<std::int64_t> selected(std::max(slots,1));
    for(int i=0;i<slots;++i)selected[i]=i%7==0?-1:i%11==0?experts:(i*13+i/3)%experts;
    check(cudaMemcpyAsync(ids.data,selected.data(),selected.size()*sizeof(std::int64_t),cudaMemcpyHostToDevice,stream.value));
    const auto fill=[&](int revision) {
        for(int i=0;i<words;++i)source.host[i]=std::uint64_t(i)*0x123456789ULL+17;
        source.host[0]=count;source.host[1]=987+revision;source.host[3]=456;
        for(int i=0;i<3*experts;++i)addresses.host[i]=0x123400000000ULL+std::uint64_t(i+revision)*0x100;
        flags.host[1]=variant==4;
        for(int i=0;i<std::max(slots,1);++i)mask.host[i]=i%3==0;
        std::memset(route.host,0x3b,route.count*sizeof(gg::RouteGroup));
        route.host[0].count=variant==2?slots+1:variant==5?-1:groups;
    };
    fill(0);flags.host[0]=1;
    initialize(old,stream.value);initialize(fresh,stream.value);
    initialize(old_addresses,stream.value);initialize(fresh_addresses,stream.value);
    initialize(old_mask,stream.value);initialize(fresh_mask,stream.value);
    initialize(old_groups,stream.value);initialize(fresh_groups,stream.value);
    const auto launch=[&](bool fused) {
        if(!fused) {
            mfq::cuda::wait_mapped_flag(flags.device,stream.value);
            if(addressed)gg::snapshot_routed_addresses(addresses.device,old_addresses.data,ids.data,slots,experts,stream.value);
            gg::snapshot_direct_plan(source.device,flags.device+1,old.data,capacity,
                cpu?mask.device:nullptr,cpu?old_mask.data:nullptr,cpu?slots:0,
                grouped?route.device:nullptr,variant==5?nullptr:old_groups.data,slots,stream.value);
            return;
        }
        gg::StreamPlanSnapshot p;
        p.ready=variant==5?nullptr:flags.device;p.mapped=source.device;p.cancelled=flags.device+1;
        p.snapshot=fresh.data;p.capacity=capacity;
        p.cpu_slots=cpu?mask.device:nullptr;p.device_slots=cpu?fresh_mask.data:nullptr;p.slots=cpu?slots:0;
        p.host_groups=grouped?route.device:nullptr;p.device_groups=variant==5?nullptr:fresh_groups.data;
        p.group_capacity=slots;p.mapped_addresses=addressed?addresses.device:nullptr;
        p.addresses=addressed?fresh_addresses.data:nullptr;p.ids=ids.data;p.selected_slots=slots;p.experts=experts;
        gg::snapshot_stream_plan(p,stream.value);
    };
    const auto verify=[&] {
        equal(old,fresh,stream.value);equal(old_addresses,fresh_addresses,stream.value);
        equal(old_mask,fresh_mask,stream.value);equal(old_groups,fresh_groups,stream.value);
        const auto header=read(fresh,stream.value);
        const bool empty=count==~std::uint64_t(0),abort=variant==4 || (!empty && count>std::uint64_t(capacity));
        if(header[0]!=(abort || empty?0:count) || header[1]!=source.host[1] ||
           header[2]!=(abort?1u:empty?2u:0u) || header[3]!=(abort || empty?0:source.host[3]))
            throw std::runtime_error("Stream-plan header differs from independent CPU reference");
        auto expected=std::vector<std::uint64_t>(3*experts,0xa5a5a5a5a5a5a5a5ULL);
        if(addressed)for(int i=0;i<slots;++i)if(selected[i]>=0 && selected[i]<experts)
            for(int role=0;role<3;++role)expected[role*experts+selected[i]]=addresses.host[role*experts+selected[i]];
        if(read(fresh_addresses,stream.value)!=expected)
            throw std::runtime_error("Stream-plan selected pointers differ from independent CPU reference");
        ++cases;
    };
    launch(false);launch(true);verify();
    if(slots!=40 || variant!=1 || !cpu || !grouped || !addressed)return;
    Graph graph[2];
    for(int mode=0;mode<2;++mode) {
        check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));launch(mode!=0);
        check(cudaStreamEndCapture(stream.value,&graph[mode].graph));
        check(cudaGraphInstantiate(&graph[mode].exec,graph[mode].graph,0));
    }
    for(int revision=1;revision<=3;++revision) {
        for(auto& captured:graph) {
            flags.host[0]=0;check(cudaGraphLaunch(captured.exec,stream.value));
            std::thread writer([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));fill(revision);
                std::atomic_thread_fence(std::memory_order_seq_cst);
                *reinterpret_cast<volatile std::uint32_t*>(flags.host)=1;
            });
            const auto status=cudaStreamSynchronize(stream.value);writer.join();check(status);++delayed;
        }
        verify();
    }
}
void resident_bypass(int tokens,int& cases) {
    const int slots=tokens*7,experts=19,capacity=slots*3,words=4+capacity*4;
    Stream stream;Mapped<std::uint64_t> source(words),resident(3*experts),mapped(3*experts);
    Mapped<std::uint32_t> flags(5);Mapped<std::int32_t> cpu(slots);Mapped<gg::RouteGroup> groups(slots+1);
    Device<std::uint64_t> output(words),addresses(3*experts);Device<std::int64_t> ids(slots);
    Device<std::int32_t> device_cpu(slots);Device<gg::RouteGroup> device_groups(slots+1);
    gg::StreamPlanSnapshot p;p.ready=flags.device;p.cancelled=flags.device+1;p.mapped=source.device;
    p.snapshot=output.data;p.capacity=capacity;p.host_groups=groups.device;p.device_groups=device_groups.data;
    p.group_capacity=slots;p.mapped_addresses=mapped.device;p.addresses=addresses.data;
    p.ids=ids.data;p.selected_slots=slots;p.experts=experts;
    gg::ResidentStreamPlan r;r.addresses=resident.device;r.ready=flags.device+2;
    r.gate_up_done=flags.device+3;r.down_done=flags.device+4;
    Graph graphs[3];
    for(int kind=0;kind<3;++kind) {
        p.cpu_slots=kind==1?cpu.device:nullptr;p.device_slots=kind==1?device_cpu.data:nullptr;p.slots=kind==1?slots:0;
        r.addresses=kind==2?nullptr:resident.device;
        check(cudaStreamBeginCapture(stream.value,cudaStreamCaptureModeThreadLocal));
        gg::snapshot_stream_plan(p,r,stream.value);check(cudaStreamEndCapture(stream.value,&graphs[kind].graph));
        check(cudaGraphInstantiate(&graphs[kind].exec,graphs[kind].graph,0));
    }
    // Retained graphs must switch between all hits, misses in each projection,
    // cancellation and a late residency-table publication on every replay.
    for(int variant=0;variant<10;++variant) {
        const bool fast=variant==0 || variant==2 || variant==7;
        std::vector<std::int64_t> selected(slots);
        for(int i=0;i<slots;++i)selected[i]=(i*13+i/3+variant)%experts;
        if(variant==8)selected[0]=-1;
        for(int i=0;i<words;++i)source.host[i]=std::uint64_t(i)*113+17;
        source.host[0]=2;source.host[1]=513;source.host[3]=255;
        std::memset(groups.host,0x53,groups.count*sizeof(gg::RouteGroup));groups.host[0].count=1;
        for(int i=0;i<slots;++i)cpu.host[i]=i%3==0;
        const auto publish_resident=[&] {
            for(int i=0;i<3*experts;++i)resident.host[i]=0x123400000000ULL+std::uint64_t(i)*0x100;
            if(variant==1 || variant==3 || variant==4)
                resident.host[(variant==1?0:variant==3?1:2)*experts+selected[0]]=0;
        };
        publish_resident();
        for(int i=0;i<3*experts;++i)mapped.host[i]=fast?0:0x223400000000ULL+std::uint64_t(i)*0x100;
        flags.host[0]=0;flags.host[1]=variant==5;flags.host[2]=variant==2?0:1;flags.host[3]=flags.host[4]=0;
        if(variant==2)std::fill(resident.host,resident.host+resident.count,0);
        check(cudaMemcpyAsync(ids.data,selected.data(),selected.size()*sizeof(std::int64_t),cudaMemcpyHostToDevice,stream.value));
        initialize(output,stream.value);initialize(addresses,stream.value);initialize(device_groups,stream.value);initialize(device_cpu,stream.value);
        check(cudaStreamSynchronize(stream.value));
        auto& graph=graphs[variant==6?1:variant==9?2:0];
        for(int replay=0;replay<2;++replay) {
            flags.host[0]=0;flags.host[3]=flags.host[4]=0;
            if(variant==2){flags.host[2]=0;std::fill(resident.host,resident.host+resident.count,0);}
            check(cudaGraphLaunch(graph.exec,stream.value));
            std::thread publisher;
            if(!fast || variant==2)publisher=std::thread([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if(variant==2){publish_resident();std::atomic_thread_fence(std::memory_order_seq_cst);flags.host[2]=1;}
                else {std::atomic_thread_fence(std::memory_order_seq_cst);flags.host[0]=1;}
            });
            const auto started=std::chrono::steady_clock::now();auto status=cudaStreamQuery(stream.value);
            while(status==cudaErrorNotReady && std::chrono::steady_clock::now()-started<std::chrono::seconds(2)) {
                std::this_thread::yield();status=cudaStreamQuery(stream.value);
            }
            const bool timeout=status==cudaErrorNotReady;
            if(timeout){flags.host[0]=flags.host[2]=1;status=cudaStreamSynchronize(stream.value);}
            if(publisher.joinable())publisher.join();check(status);
            if(timeout)throw std::runtime_error("Resident-only snapshot waited for an unpublished CPU plan");
            const auto header=read(output,stream.value);
            const bool abort=variant==5;
            if(header[0]!=(fast || abort?0:2) || header[1]!=(fast?0:513) ||
                    header[2]!=(abort?1:0) || header[3]!=(fast || abort?0:255))
                throw std::runtime_error("Resident snapshot header differs from independent reference");
            for(int word=4;word<words;++word)if(header[word]!=(!fast && !abort && word<12?
                    source.host[word]:0xa5a5a5a5a5a5a5a5ULL))
                throw std::runtime_error("Resident snapshot overwrote a descriptor payload boundary");
            auto expected=std::vector<std::uint64_t>(3*experts,0xa5a5a5a5a5a5a5a5ULL);
            for(auto expert:selected)if(expert>=0 && expert<experts)for(int role=0;role<3;++role)
                expected[role*experts+expert]=fast?0:mapped.host[role*experts+expert];
            if(read(addresses,stream.value)!=expected)throw std::runtime_error("Resident snapshot changed selected pointer bounds");
            if(read(device_groups,stream.value)[0].count!=(fast || abort?0:1))
                throw std::runtime_error("Resident snapshot lost its empty streamed group count");
            if(fast && (flags.host[0] || flags.host[3]!=1 || flags.host[4]!=1))
                throw std::runtime_error("Resident snapshot failed to publish empty DMA completion independently");
            if(!fast && (flags.host[3] || flags.host[4]))throw std::runtime_error("A real miss bypassed the DMA producer");
            if(variant==6) {
                const auto mask=read(device_cpu,stream.value);
                for(int i=0;i<slots;++i)if(mask[i]!=cpu.host[i])throw std::runtime_error("CPU expert slots were bypassed");
            }
            ++cases;
        }
    }
}
}

int main() {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) {
        std::cout<<"SKIP: CUDA device unavailable\n";return 77;
    }
    try {
        int device=0,can_map=0;check(cudaGetDevice(&device));
        check(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,device));
        if(!can_map){std::cout<<"SKIP: mapped host memory unavailable\n";return 77;}
        int cases=0,delayed=0;
        for(int slots:{0,1,4,40,257})for(int variant:{0,1,2,3,4,5})
        for(bool cpu:{false,true})for(bool grouped:{false,true})for(bool addressed:{false,true})
            run(slots,variant,cpu,grouped,addressed,cases,delayed);
        int resident_cases=0;
        for(int tokens=1;tokens<=8;++tokens)resident_bypass(tokens,resident_cases);
        std::cout<<"Stream-plan snapshots: "<<cases<<" exact/reference cases, "<<delayed
            <<" delayed host-publication graph replays, "<<resident_cases<<" resident/miss/CPU/cancel Graph cases PASS\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
