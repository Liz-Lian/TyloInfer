#include "backends/cuda/engine/executor/tq_moe.h"
#include "native/cuda/profile.h"
#include "engine/executor/tq.h"
#include "backends/cuda/engine/executor/decode_window.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/weights/tq/cache.h"
#include "ops/tq/include/detail.h"
#include "ops/tq/include/moe.h"
#include "context.h"
#include "env.h"
#include "kernels/cuda/moe/pipeline.h"
#include "ops/common/include/moe.h"
#include "quant_linear.h"
#include "weight_memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace tb = mfq_tensor_backend;

bool tq_graph_supports_tokens(std::int64_t tokens)noexcept {
#if CUDART_VERSION >= 11060
    return tokens>0;
#else
    return tokens==1;
#endif
}

namespace {

struct TqGraphLease {std::array<std::vector<TqWeight>,3> weights;void clear(){for(auto& w:weights)w.clear();}};
const char tq_graph_lease_identity=0;
struct TqMoeGraphStage {
    std::array<std::shared_ptr<TqRoutedWeight>,3> banks;
    std::shared_ptr<TqGraphLease> lease;
    MfqCudaStream stream;
    std::unique_ptr<mfq::cuda::HostBuffer> host_ids;
    std::array<tb::Tensor,3> views;
    tb::Tensor input,ids,weights,hidden,pairs,result,plan,route_snapshot;
    tb::ScalarType route_dtype;
    void* mapped_routes=nullptr;
    std::vector<int64_t> widened_routes;
    int routes,type,tokens;
    std::size_t slots=0,route_bytes=0,plan_bytes=0;
    bool wide=false;
    struct Plane {std::size_t offset=0;unsigned capacity=0,count=0;int tile=0,kind=0;};
    std::array<Plane,13> planes{};
    struct KernelChunk {int plane;std::size_t first,segment;unsigned capacity;cudaGraphNode_t node;};
    std::vector<KernelChunk> kernels;
    TqMoeGraphStage(TqExec& exec,const TqMoeBanks& b,const tb::Tensor& source,int count,tb::ScalarType route_type,
                   std::shared_ptr<TqGraphLease> l)
        :banks{b.gate,b.up,b.down},lease(std::move(l)),stream(mfq_get_current_cuda_stream(source.get_device())),
         route_dtype(route_type),
         routes(count),type(tq_detail::dtype(source.scalar_type())),tokens(int(source.size(0))) {
        slots=std::size_t(tokens)*routes;
        if(slots>INT_MAX)throw std::overflow_error("TQ graph route count exceeds plan limits");
        route_bytes=slots*sizeof(int64_t);
        if(tokens>1) {
            wide=type==0 && exec.get(source.get_device())->indexed_mma();
            for(int i=0;i<int(planes.size());++i) {
                auto& plane=planes[i];
                // GU1..4, GU8, Down1..4, Down5..8, in execution order.
                plane.kind=i<4?0:i==4?1:i<9?2:3;
                plane.tile=i<4?i+1:i==4?8:i-4;
                plane.offset=plan_bytes;
                // Every job consumes tile distinct routing slots. This bound
                // reserves less metadata than giving every tile all slots.
                plane.capacity=(plane.kind==1 || plane.kind==3) && !wide?0:unsigned(slots/plane.tile);
                const auto bytes=plane.kind==1?sizeof(TrellisIndexedWidePairJob):
                    plane.kind==3?sizeof(TrellisIndexedWideProjectionJob):sizeof(TrellisIndexedProjectionJob);
                if(plane.capacity>(std::size_t(INT64_MAX)-route_bytes-plan_bytes)/bytes)
                    throw std::overflow_error("TQ graph plan size exceeds tensor limits");
                plan_bytes+=plane.capacity*bytes;
            }
            plan=tb::empty({static_cast<int64_t>(plan_bytes)},source.options().dtype(tb::kUInt8));
            kernels.reserve(planes.size()*((slots+65534)/65535));
        }
        int can_map=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,source.get_device()));
        const auto* mapped_option=mfq::cuda::env_lookup("TYLOINFER_TQ_GRAPH_MAPPED_ROUTES");
        const bool integral=route_dtype==tb::kInt32 || route_dtype==tb::kInt64;
        const auto words=slots*(route_dtype==tb::kInt64?2:1);
        const bool map=can_map && integral && words<=INT_MAX && (!mapped_option || mapped_option[0]!='0');
        // The first cache line is the publication flag. Int64 IDs are copied
        // as two int32 words, preserving their full range without narrowing.
        host_ids=std::make_unique<mfq::cuda::HostBuffer>(64+route_bytes+plan_bytes,map);
        if(map)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_routes,host_ids->data(),0));
        if(mapped_routes) {
            route_snapshot=tb::empty({tokens,count},source.options().dtype(route_dtype));
            if(route_dtype==tb::kInt32)widened_routes.resize(slots);
        }
        input=tb::empty_like(source);
        ids=tb::empty({tokens,count},source.options().dtype(tb::kInt64));
        weights=tb::empty({tokens,count},source.options().dtype(tb::kFloat32));
        hidden=tb::empty({int64_t(slots),b.up->out},source.options());
        pairs=tb::empty({tokens,count,b.down->out},source.options());
        result=tb::empty({tokens,b.down->out},source.options());
        if(tokens==1)for(int role=0;role<3;++role)views[role]=banks[role]->cache->views(
            banks[role]->source_id,nullptr,{},source.get_device());
        mfq::cuda::charge_tensor_host_bytes(host_ids->size());
    }
    void reset_routes() {
        if(!mapped_routes)return;
        *static_cast<volatile uint32_t*>(host_ids->data())=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    void* route_data()const {return static_cast<uint8_t*>(host_ids->data())+64;}
    void publish_routes(const tb::Tensor& current,bool views,const int64_t* device_ids) {
        if(!mapped_routes) {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(route_data(),device_ids,route_bytes,cudaMemcpyDeviceToHost,stream.stream()));
            return;
        }
        reset_routes();
        const auto* source=&current;
        if(!views || !current.is_contiguous()){route_snapshot.copy_(current);source=&route_snapshot;}
        mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const int32_t*>(source->data_ptr()),nullptr,
            reinterpret_cast<int32_t*>(static_cast<uint8_t*>(mapped_routes)+64),
            static_cast<uint32_t*>(mapped_routes),0,int(slots*(route_dtype==tb::kInt64?2:1)),stream.stream());
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    ~TqMoeGraphStage() {
        // The D2H destination belongs to this stage, rather than the tensor
        // allocator. A window destroys its graphs before releasing the stage.
        int previous=0;const bool restore=cudaGetDevice(&previous)==cudaSuccess;
        (void)cudaSetDevice(stream.device());(void)cudaStreamSynchronize(stream.stream());
        mfq::cuda::tensor_host_bytes.fetch_sub(host_ids->size());
        if(restore)(void)cudaSetDevice(previous);
    }
    void load(mfq::cuda::DecodeWindow* window=nullptr) {
        const MfqCudaStreamGuard guard(stream);
        const auto* selected=static_cast<const int64_t*>(route_data());
        if(mapped_routes && route_dtype==tb::kInt32) {
            std::copy_n(static_cast<const int32_t*>(route_data()),slots,widened_routes.data());
            selected=widened_routes.data();
        }
        for(std::size_t i=0;i<slots;++i)if(selected[i]<0 || selected[i]>=banks[0]->experts)
            throw std::out_of_range("TQ graph expert ID is out of bounds");
        std::vector<int64_t> unique;
        std::vector<std::vector<int64_t>> grouped;
        if(tokens==1)unique.assign(selected,selected+routes);
        else {
            std::unordered_map<int64_t,std::size_t> positions;
            for(std::size_t i=0;i<slots;++i) {
                const auto [entry,inserted]=positions.emplace(selected[i],unique.size());
                if(inserted){unique.push_back(selected[i]);grouped.emplace_back();}
                grouped[entry->second].push_back(int64_t(i));
            }
        }
        // The preceding GPU segment has consumed the prior layer's images.
        // Keep one live layer across all TQ sources, independently of cache
        // ownership, so graph capture cannot pin every layer's experts.
        lease->clear();
        for(int role=0;role<3;++role) {
            const auto& bank=*banks[role];
            lease->weights[role]=bank.cache->get(bank.source_id,unique.data(),unique.size(),stream.device());
            if(tokens==1) {
                const auto table=bank.cache->views(bank.source_id,unique.data(),
                    lease->weights[role],stream.device());
                if(table.data_ptr()!=views[role].data_ptr())throw std::logic_error("TQ graph view table address changed");
            }
        }
        if(tokens==1)return;
        const auto jobs=tq_ops::make_indexed_jobs(lease->weights[0],lease->weights[1],lease->weights[2],
            grouped,0,routes,banks[0]->input,banks[1]->out,banks[2]->out,wide);
        auto* packet=static_cast<std::uint8_t*>(route_data())+route_bytes;
        std::memset(packet,0,plan_bytes);
        for(int i=0;i<int(planes.size());++i) {
            auto& plane=planes[i];
            const auto fill=[&](const auto& jobs) {
                if(jobs.size()>plane.capacity)throw std::overflow_error("TQ graph job count exceeds prepared plan");
                plane.count=unsigned(jobs.size());
                if(!jobs.empty())std::memcpy(packet+plane.offset,jobs.data(),jobs.size()*sizeof(jobs[0]));
            };
            if(plane.kind==0)fill(jobs.gate_up[plane.tile-1]);
            else if(plane.kind==1)fill(jobs.wide_gate_up);
            else if(plane.kind==2)fill(jobs.down[plane.tile-1]);
            else fill(jobs.wide[plane.tile-5]);
        }
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(plan.data_ptr(),packet,plan_bytes,cudaMemcpyHostToDevice,stream.stream()));
        if(window)for(const auto& kernel:kernels) {
            const auto count=kernel.first>=planes[kernel.plane].count?0:
                unsigned(std::min<std::size_t>(kernel.capacity,planes[kernel.plane].count-kernel.first));
            window->set_kernel_rows(kernel.segment,kernel.node,count);
        }
        if(window && wide && mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")) {
            unsigned mask=0,jobs=0;
            for(int i=9;i<int(planes.size());++i)if(planes[i].count){mask|=1u<<planes[i].tile;jobs+=planes[i].count;}
            if(planes[4].count)mask|=1u<<16;
            static std::atomic<unsigned> seen{0};
            if(mask & ~seen.fetch_or(mask,std::memory_order_relaxed))
                std::fprintf(stderr,"tq_indexed_dispatch mode=graph dtype=%d tokens=%d wide_jobs=%u gu8_jobs=%u reuse_mask=%u plan_bytes=%zu\n",
                    type,tokens,jobs,planes[4].count,mask,plan_bytes);
        }
    }
    void indexed(float limit,mfq::cuda::DecodeWindow& window,bool recording,const void* input_values) {
#if CUDART_VERSION >= 11060
        if(recording)kernels.clear();
        for(int i=0;i<int(planes.size());++i) {
            const auto& plane=planes[i];
            for(std::size_t first=0;first<plane.capacity;first+=65535) {
                const auto capacity=unsigned(std::min<std::size_t>(65535,plane.capacity-first));
                const auto active=first>=plane.count?0:
                    unsigned(std::min<std::size_t>(capacity,plane.count-first));
                const auto count=recording?capacity:std::max(1u,active);
                const auto* address=plan.data_ptr<std::uint8_t>()+plane.offset;
                const auto* jobs=reinterpret_cast<const TrellisIndexedProjectionJob*>(address)+first;
                if(plane.kind==0)tq_detail::status(mfq_tq_indexed_swiglu(jobs,input_values,hidden.data_ptr(),int(count),
                    int(banks[1]->out),type,plane.tile,limit,stream.stream()));
                else if(plane.kind==1)tq_detail::status(mfq_tq_indexed_wide_swiglu(
                    reinterpret_cast<const TrellisIndexedWidePairJob*>(address)+first,
                    input_values,hidden.data_ptr(),int(count),int(banks[1]->out),type,plane.tile,limit,stream.stream()));
                else if(plane.kind==2)tq_detail::status(mfq_tq_indexed(jobs,hidden.data_ptr(),pairs.data_ptr(),int(count),
                    int(banks[2]->out),type,plane.tile,stream.stream()));
                else tq_detail::status(mfq_tq_indexed_wide(reinterpret_cast<const TrellisIndexedWideProjectionJob*>(address)+first,
                    hidden.data_ptr(),pairs.data_ptr(),int(count),int(banks[2]->out),type,plane.tile,stream.stream()));
                if(recording) {
                    cudaStreamCaptureStatus capture;const cudaGraphNode_t* dependencies=nullptr;std::size_t n=0;
                    // CUDA 13 folded the versioned capture queries into the unversioned API
                    // and added an edge-data output before the dependency count.
#if CUDART_VERSION >= 13000
                    TYLOINFER_CUDA_CHECK(cudaStreamGetCaptureInfo(stream.stream(),&capture,nullptr,nullptr,&dependencies,nullptr,&n));
#else
                    TYLOINFER_CUDA_CHECK(cudaStreamGetCaptureInfo_v2(stream.stream(),&capture,nullptr,nullptr,&dependencies,&n));
#endif
                    if(capture!=cudaStreamCaptureStatusActive || n!=1)throw std::logic_error("TQ kernel capture has no unique node");
                    kernels.push_back({i,first,window.segments()-1,capacity,dependencies[0]});
                }
            }
            if(i==4)window.mark_current("gate_up_end");
        }
#else
        (void)limit;(void)window;(void)recording;
        throw std::runtime_error("multi-token TQ graph plan updates require CUDA11.6 or newer");
#endif
    }
};
tb::Tensor tq_graph_forward(TqExec& exec,const TqMoeBanks& banks,const tb::Tensor& source,const tb::Tensor& ids,
        const tb::Tensor& weights,float limit,mfq::cuda::DecodeWindow& window) {
    window.mark_current("expert_begin");
    auto lease=window.workspace<TqGraphLease>(&tq_graph_lease_identity,[]{return std::make_shared<TqGraphLease>();});
    auto stage=window.workspace<TqMoeGraphStage>(&banks,[&] {
        return std::make_shared<TqMoeGraphStage>(exec,banks,source,int(ids.size(1)),ids.scalar_type(),lease);
    });
    if(stage->routes!=ids.size(1) || stage->input.sizes()!=source.sizes() || stage->input.scalar_type()!=source.scalar_type() ||
       stage->route_dtype!=ids.scalar_type())
        throw std::invalid_argument("TQ graph stage geometry changed");
    const bool views=mfq::cuda::runtime_options::window_input_views() && source.is_contiguous() &&
        ids.is_contiguous() && weights.is_contiguous() && weights.scalar_type()==tb::kFloat32;
    if(mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")) {
        const unsigned mode=(stage->mapped_routes?(stage->route_dtype==tb::kInt32?1u:2u):4u)|(views?8u:0u);
        static std::atomic<unsigned> seen{0};const unsigned bit=1u<<mode;
        if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))
            std::fprintf(stderr,"tq_graph_handoff mapped_routes=%d route_bits=%d input_views=%d\n",
                int(stage->mapped_routes!=nullptr),stage->route_dtype==tb::kInt32?32:64,int(views));
    }
    if(!views){stage->input.copy_(source);stage->weights.copy_(weights);}
    const int64_t* device_ids=nullptr;
    if(views && ids.scalar_type()==tb::kInt64)device_ids=ids.data_ptr<int64_t>();
    else if(stage->tokens==1 || !stage->mapped_routes || !views) {
        stage->ids.copy_(ids);device_ids=stage->ids.data_ptr<int64_t>();
    }
    stage->publish_routes(ids,views,device_ids);
    window.mark_current("cpu_wait_begin");
    cudaStreamCaptureStatus capture;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stage->stream.stream(),&capture));
    if(capture==cudaStreamCaptureStatusNone) {
        // DecodeWindow also calls the body during preparation/warmup. Execute
        // real transfers there; host_barrier itself only records a callback.
        if(stage->mapped_routes)mfq::cuda::wait_route_publication(static_cast<const uint32_t*>(stage->host_ids->data()),stage->stream.stream());
        else TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stage->stream.stream()));
        stage->load();
    }
    window.enroll({stage.get(),[stage]{stage->reset_routes();},[stage,&window] {
        if(!stage->mapped_routes)return;
        mfq::cuda::wait_route_publication(static_cast<const uint32_t*>(stage->host_ids->data()),stage->stream.stream());
        stage->load(&window);
    },[lease]{lease->clear();},[]{},[lease]{lease->clear();},{}});
    window.host_barrier([stage,&window]{if(!stage->mapped_routes)stage->load(&window);});
    window.mark_current("hot_begin");
    const auto* input_values=views?source.data_ptr():stage->input.data_ptr();
    if(stage->tokens>1)stage->indexed(limit,window,capture==cudaStreamCaptureStatusActive,input_values);
    else {
    tq_detail::status(mfq_tq_routed_swiglu(reinterpret_cast<const TrellisProjectionView*>(stage->views[0].data_ptr()),
        reinterpret_cast<const TrellisProjectionView*>(stage->views[1].data_ptr()),device_ids,
        input_values,stage->hidden.data_ptr(),stage->routes,int(banks.up->out),stage->type,limit,stage->stream.stream()));
    window.mark_current("gate_up_end");
    tq_detail::status(mfq_tq_routed_down(reinterpret_cast<const TrellisProjectionView*>(stage->views[2].data_ptr()),
        device_ids,stage->hidden.data_ptr(),stage->pairs.data_ptr(),stage->routes,
        int(banks.down->out),int(banks.up->out),stage->type,stage->stream.stream()));
    }
    window.mark_current("down_end");
    tq_detail::status(mfq_tq_weighted_reduce(stage->pairs.data_ptr(),views?weights.data_ptr<float>():stage->weights.data_ptr<float>(),stage->result.data_ptr(),
        stage->tokens,stage->routes,int(banks.down->out),stage->type,stage->stream.stream()));
    window.mark_current("hot_end");
    return stage->result;
}
}

tb::Tensor TqMoeBanks::forward(TqExec& exec,tb::Tensor input_value,tb::Tensor ids,
        tb::Tensor weights,float limit,CudaProfiler* profiler,std::int64_t expert_begin,std::int64_t expert_end)const {
    const auto phase=[&](const char* name,auto&& operation) {
        return profiler?profiler->measure(name,operation):operation();
    };
    if(!gate || !up || !down || !gate->cache || !up->cache || !down->cache ||
       gate->cache!=up->cache || gate->cache!=down->cache ||
       gate->experts!=up->experts || up->experts!=down->experts ||
       gate->input!=up->input || gate->out!=up->out || down->input!=up->out ||
       ids.dim()!=2 || ids.size(1)<1 || ids.size(1)>INT_MAX || input_value.dim()!=2 ||
       !input_value.is_cuda() || weights.device()!=input_value.device() ||
       input_value.size(0)!=ids.size(0) || input_value.size(1)!=up->input ||
       weights.sizes().vec()!=ids.sizes().vec() || !std::isfinite(limit) || limit<0)
        throw std::invalid_argument("invalid TQ three-bank MoE geometry");
    if(expert_end==-1)expert_end=gate->experts;
    if(expert_begin<0 || expert_end<expert_begin || expert_end>gate->experts)
        throw std::invalid_argument("invalid TQ expert ownership range");
    const bool partitioned=expert_begin!=0 || expert_end!=gate->experts;
    const auto tokens=ids.size(0),routes=ids.size(1),hidden=up->out,width=down->out;
    if(tokens>INT_MAX || hidden>INT_MAX || width>INT_MAX || up->input>INT_MAX)
        throw std::overflow_error("TQ MoE activation dimensions exceed CUDA limits");
    const auto source=input_value.contiguous();
    const MfqCudaGuard guard(source.get_device());
    if(auto* window=mfq::cuda::DecodeWindow::recording();!partitioned && window && tq_graph_supports_tokens(tokens) && ids.device()==source.device())
        return tq_graph_forward(exec,*this,source,ids,weights,limit,*window);
    auto result=tb::empty({tokens,width},source.options());
    if(!tokens)return result;
    const auto type=tq_detail::dtype(source.scalar_type());
    const auto coefficients=weights.to(tb::kFloat32).contiguous();
    // Exactly one route readback for all three banks, independent of the
    // number of selected tokens and their expert reuse pattern.
    // Router output is normally int32. Widen this tiny ID list on CPU after
    // the readback, avoiding a GPU allocation/conversion before every wait.
    const auto host=phase("tq.moe.routes",[&] {return exec.get(source.get_device())->routes(ids);});
    const auto* selected=host.data();
    std::vector<std::int64_t> unique;
    std::vector<std::vector<std::int64_t>> slots;
    if(static_cast<std::uint64_t>(up->experts)<=host.size()) {
        // Dense counters use no more memory than the route readback. Count
        // first to reserve each expert's slots once, then reuse counters as
        // indices. Sparse expert domains retain bounded hash lookup below.
        std::vector<std::size_t> positions(static_cast<std::size_t>(up->experts),0);
        for(const auto expert:host) {
            if(expert<0 || expert>=up->experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
            if(expert<expert_begin || expert>=expert_end)continue;
            if(!positions[expert]++)unique.push_back(expert);
        }
        slots.resize(unique.size());
        for(std::size_t i=0;i<unique.size();++i) {
            auto& count=positions[unique[i]];slots[i].reserve(count);count=i;
        }
        for(std::size_t i=0;i<host.size();++i)
            if(selected[i]>=expert_begin && selected[i]<expert_end)slots[positions[selected[i]]].push_back(i);
    }else {
        std::unordered_map<std::int64_t,std::size_t> positions;
        positions.reserve(host.size());
        for(std::size_t i=0;i<host.size();++i) {
            const auto expert=selected[i];
            if(expert<0 || expert>=up->experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
            if(expert<expert_begin || expert>=expert_end)continue;
            const auto [position,inserted]=positions.emplace(expert,unique.size());
            if(inserted){unique.push_back(expert);slots.emplace_back();}
            slots[position->second].push_back(i);
        }
    }
    if(unique.empty())return tb::zeros({tokens,width},source.options());
    // Choose GEMM by each expert's actual token count. A long prompt can
    // still have many sparsely selected experts that belong in indexed jobs.
    std::size_t long_groups=0;
    for(std::size_t i=0;i<unique.size();++i)if(slots[i].size()>=std::size_t(tq_detail::gemm_min_tokens(source.scalar_type()))) {
        std::swap(unique[i],unique[long_groups]);std::swap(slots[i],slots[long_groups]);++long_groups;
    }
    const auto resources=exec.get(source.get_device());
    auto* blas=&resources->blas();
    std::size_t prefill_groups=0;
    if(type==0 && width==up->input && hidden<=INT_MAX/2) {
        // Batch up to two 128-token reuse tiles to amortize expert dispatch.
        // Larger groups retain GEMM, which decodes each weight tile once.
        for(std::size_t i=0;i<long_groups;++i)if(slots[i].size()<=256) {
            std::swap(unique[i],unique[prefill_groups]);std::swap(slots[i],slots[prefill_groups]);++prefill_groups;
        }
    }
    auto pairs=partitioned?tb::zeros({tokens,routes,width},source.options()):tb::empty({tokens,routes,width},source.options());
    auto flat_pairs=pairs.reshape({tokens*routes,width});
    if(prefill_groups) {
        // Similar token counts share a compute tile. A frequently selected
        // expert must not force padding every sparse expert to its capacity.
        std::vector<std::pair<int64_t,std::vector<int64_t>>> sorted;
        sorted.reserve(prefill_groups);
        for(std::size_t i=0;i<prefill_groups;++i)sorted.emplace_back(unique[i],std::move(slots[i]));
        std::stable_sort(sorted.begin(),sorted.end(),[](const auto& a,const auto& b){return a.second.size()>b.second.size();});
        std::size_t selected_slots=0;
        for(std::size_t i=0;i<prefill_groups;++i) {
            unique[i]=sorted[i].first;slots[i]=std::move(sorted[i].second);selected_slots+=slots[i].size();
        }
        const auto stream=mfq_get_current_cuda_stream(source.get_device());
        resources->retain(stream);
        std::size_t gemm_groups=0;
        while(gemm_groups<prefill_groups && (!resources->prefill_mma() || slots[gemm_groups].size()>128))++gemm_groups;
        const auto job_bytes=(prefill_groups*sizeof(TrellisPrefillMoeJob)+15)&~std::size_t(15);
        TqPrefillPlanLease lease{resources->prefill_plan(job_bytes+selected_slots*sizeof(int64_t)),stream.stream()};
        auto* host_bytes=static_cast<std::uint8_t*>(lease.plan->host.data_ptr());
        auto* device_bytes=static_cast<std::uint8_t*>(lease.plan->device.data_ptr());
        auto* host_slots=reinterpret_cast<int64_t*>(host_bytes+job_bytes);
        std::size_t first=0;
        for(std::size_t i=0;i<prefill_groups;++i) {
            std::memcpy(host_slots+first,slots[i].data(),slots[i].size()*sizeof(int64_t));first+=slots[i].size();
        }
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_bytes+job_bytes,host_slots,selected_slots*sizeof(int64_t),
            cudaMemcpyHostToDevice,stream.stream()));
        const auto free_bytes=[&] {
            return phase("tq.moe.capacity",[&] {
                std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));return available;
            });
        };
        const auto available=free_bytes();
        const auto max_width=std::uint64_t(std::max(up->input,hidden));
        const auto output_width=std::uint64_t(std::max(up->input,2*hidden));
        const auto max_capacity=(slots[0].size()+15)&~std::size_t(15);
        const auto minimum_activations=max_capacity*(gemm_groups?2*max_width+4*output_width:4*max_width+2*hidden)+32;
        std::uint64_t useful_activations=gemm_groups?
            max_capacity*(2*max_width+4*output_width)*std::min<std::size_t>(16,gemm_groups)+32:0;
        if(gemm_groups<prefill_groups) {
            const auto short_capacity=(slots[gemm_groups].size()+15)&~std::size_t(15);
            const auto useful=short_capacity*(4*max_width+2*hidden)*(prefill_groups-gemm_groups)+32;
            useful_activations=std::max<std::uint64_t>(useful_activations,
                std::min<std::uint64_t>(std::uint64_t(16)<<20,useful));
        }
        const auto activation_budget=std::max<std::uint64_t>(minimum_activations,
            std::min<std::uint64_t>({std::uint64_t(gemm_groups?64:16)<<20,available/16,useful_activations}));
        auto activations=tb::empty({static_cast<int64_t>(activation_budget)},source.options().dtype(tb::kUInt8));
        tb::Tensor decoded;
        cublasHandle_t blas=nullptr;
        if(gemm_groups) {
            const auto useful=4*std::uint64_t(up->input)*hidden*std::min<std::size_t>(16,gemm_groups);
            const auto bytes=std::max<std::uint64_t>(max_width*2,
                std::min<std::uint64_t>({std::uint64_t(128)<<20,available/16,useful}));
            decoded=tb::empty({static_cast<int64_t>(bytes)},source.options().dtype(tb::kUInt8));
            blas=resources->get();
        }
        const auto image_bound=[](const TqRoutedWeight& w) {
            return std::uint64_t(w.out)*(((std::uint64_t(w.input)+255)&~std::uint64_t(255))*2+16)+512;
        };
        const auto image_per_expert=image_bound(*gate)+image_bound(*up)+image_bound(*down);
        // Image leases and compute scratch have independent live budgets.
        // Account for allocated scratch before admitting compressed misses.
        const auto image_batch=std::min<std::size_t>(prefill_groups,
            std::max<std::uint64_t>(1,(free_bytes()/2)/image_per_expert));
        if(mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE"))std::fprintf(stderr,
            "tq_prefill_moe jobs=%zu gemm_jobs=%zu scratch=%lld decoded=%lld image_batch=%zu\n",
            prefill_groups,gemm_groups,static_cast<long long>(activations.numel()),
            static_cast<long long>(decoded.defined()?decoded.numel():0),image_batch);
        first=0;
        for(std::size_t begin=0;begin<prefill_groups;begin+=image_batch) {
            const auto count=std::min(image_batch,prefill_groups-begin);
            const auto banks=phase("tq.moe.weights",[&] {
                return std::array<std::vector<TqWeight>,3>{
                    gate->cache->get(gate->source_id,unique.data()+begin,count,source.get_device()),
                    up->cache->get(up->source_id,unique.data()+begin,count,source.get_device()),
                    down->cache->get(down->source_id,unique.data()+begin,count,source.get_device())};
            });
            for(std::size_t i=0;i<count;++i) {
                const auto index=begin+i;
                const TrellisPrefillMoeJob job{banks[0][i].view,banks[1][i].view,banks[2][i].view,
                    static_cast<int64_t>(first),static_cast<int>(slots[index].size())};
                std::memcpy(host_bytes+index*sizeof(job),&job,sizeof(job));first+=slots[index].size();
            }
            const auto offset=begin*sizeof(TrellisPrefillMoeJob);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_bytes+offset,host_bytes+offset,count*sizeof(TrellisPrefillMoeJob),
                cudaMemcpyHostToDevice,stream.stream()));
            for(std::size_t tile=begin;tile<begin+count;) {
                const bool gemm=tile<gemm_groups;
                const auto capacity=(slots[tile].size()+15)&~std::size_t(15);
                const auto per_expert=capacity*(gemm?2*max_width+4*output_width:4*max_width+2*hidden);
                const auto act_limit=gemm?activation_budget:std::min<std::uint64_t>(activation_budget,std::uint64_t(16)<<20);
                auto tile_count=std::min<std::size_t>({begin+count-tile,65535,
                    std::max<std::uint64_t>(1,(act_limit-std::min<std::uint64_t>(32,act_limit))/per_expert)});
                if(gemm) {
                    // Bound independent GEMMs while keeping full weight tiles
                    // when possible; larger matrices still use row tiling.
                    const auto full_tile=4*std::uint64_t(up->input)*hidden;
                    tile_count=std::min<std::size_t>({tile_count,gemm_groups-tile,16,
                        std::max<std::uint64_t>(1,decoded.nbytes()/full_tile)});
                }
                const auto* plan=reinterpret_cast<const TrellisPrefillMoeJob*>(device_bytes+tile*sizeof(TrellisPrefillMoeJob));
                const auto* device_slots=reinterpret_cast<const int64_t*>(device_bytes+job_bytes);
                if(gemm)phase("tq.moe.prefill_gemm",[&] {tq_detail::status(mfq_tq_prefill_moe_gemm(plan,device_slots,
                    source.data_ptr(),pairs.data_ptr(),static_cast<int>(tile_count),static_cast<int>(capacity),
                    static_cast<int>(up->input),static_cast<int>(hidden),static_cast<int>(routes),limit,
                    decoded.data_ptr(),decoded.nbytes(),activations.data_ptr(),activations.nbytes(),blas,stream.stream()));return 0;});
                else phase("tq.moe.prefill",[&] {tq_detail::status(mfq_tq_prefill_moe(plan,device_slots,
                    source.data_ptr(),pairs.data_ptr(),static_cast<int>(tile_count),static_cast<int>(capacity),
                    static_cast<int>(up->input),static_cast<int>(hidden),static_cast<int>(routes),limit,
                    activations.data_ptr(),activations.nbytes(),stream.stream()));return 0;});
                tile+=tile_count;
            }
        }
    }
    if(prefill_groups<long_groups) {
        // Long prefill gathers one expert's rows and uses the ordinary
        // multi-token/GEMM kernels. Only one compressed expert and bounded
        // decoded tiles need to be live, regardless of the prompt length.
        for(std::size_t index=prefill_groups;index<long_groups;++index) {
            const auto expert=unique[index];
            const auto g=gate->cache->get(gate->source_id,&expert,1,source.get_device());
            const auto u=up->cache->get(up->source_id,&expert,1,source.get_device());
            const auto d=down->cache->get(down->source_id,&expert,1,source.get_device());
            tq_ops::run_long_expert(g[0],u[0],d[0],slots[index],routes,
                source,flat_pairs,blas,limit);
        }
    }
    if(long_groups<unique.size()) {
        auto activated=tb::empty({tokens,routes,hidden},source.options());
        // The live batch is bounded by current free VRAM, not a device or
        // model whitelist. FP16-sized images conservatively bound every
        // supported 1..8-bit compressed row plus metadata and scale data.
        const auto image_bound=[](const TqRoutedWeight& w) {
            return std::uint64_t(w.out)*(((std::uint64_t(w.input)+255)&~std::uint64_t(255))*2+16)+512;
        };
        const auto per_expert=image_bound(*gate)+image_bound(*up)+image_bound(*down);
        for(std::size_t begin=long_groups;begin<unique.size();) {
            const auto free=phase("tq.moe.capacity",[&] {
                std::size_t available=0,total=0;TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));return available;
            });
            const auto count=std::min<std::size_t>(std::min<std::size_t>(65535,unique.size()-begin),
                std::max<std::uint64_t>(1,(free/2)/per_expert));
            const auto banks=phase("tq.moe.weights",[&] {
                return std::array<std::vector<TqWeight>,3>{
                    gate->cache->get(gate->source_id,unique.data()+begin,count,source.get_device()),
                    up->cache->get(up->source_id,unique.data()+begin,count,source.get_device()),
                    down->cache->get(down->source_id,unique.data()+begin,count,source.get_device())};
            });
            const auto& g=banks[0];const auto& u=banks[1];const auto& d=banks[2];
            if(tokens==1 && unique.size()==static_cast<std::size_t>(routes)) {
                // Preserve the tuned single-vector geometry. Each selected
                // expert gets one slot, with no multi-token dummy work.
                const auto tables=phase("tq.moe.views",[&] {
                    return std::array<tb::Tensor,3>{
                        gate->cache->views(gate->source_id,unique.data()+begin,g,source.get_device()),
                        up->cache->views(up->source_id,unique.data()+begin,u,source.get_device()),
                        down->cache->views(down->source_id,unique.data()+begin,d,source.get_device())};
                });
                const auto& gp=tables[0];const auto& upp=tables[1];const auto& dp=tables[2];
                const auto route_ids=ids.to(source.device()).to(tb::kInt64).contiguous();
                const auto element=source.element_size();
                auto* h=static_cast<std::uint8_t*>(activated.data_ptr())+begin*hidden*element;
                auto* p=static_cast<std::uint8_t*>(pairs.data_ptr())+begin*width*element;
                phase("tq.moe.gu",[&] {tq_detail::status(mfq_tq_routed_swiglu(reinterpret_cast<const TrellisProjectionView*>(gp.data_ptr()),
                    reinterpret_cast<const TrellisProjectionView*>(upp.data_ptr()),route_ids.data_ptr<int64_t>()+begin,
                    source.data_ptr(),h,static_cast<int>(count),static_cast<int>(hidden),type,limit,mfq_current_cuda_stream()));return 0;});
                phase("tq.moe.down",[&] {tq_detail::status(mfq_tq_routed_down(reinterpret_cast<const TrellisProjectionView*>(dp.data_ptr()),
                    route_ids.data_ptr<int64_t>()+begin,h,p,static_cast<int>(count),static_cast<int>(width),
                    static_cast<int>(hidden),type,mfq_current_cuda_stream()));return 0;});
            }else {
                tq_ops::run_indexed(g,u,d,slots,begin,routes,source,activated,pairs,
                    limit,type==0 && resources->indexed_mma());
            }
            begin+=count;
        }
    }
    phase("tq.moe.reduce",[&] {tq_detail::status(mfq_tq_weighted_reduce(pairs.data_ptr(),coefficients.data_ptr<float>(),result.data_ptr(),
        static_cast<int>(tokens),static_cast<int>(routes),static_cast<int>(width),type,mfq_current_cuda_stream()));return 0;});
    return result;
}
