#include "kernels/qsa_ram.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

namespace tq=tyloinfer::qsa;
void check(cudaError_t e) {if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;
    explicit Device(std::size_t n){check(cudaMalloc(&p,std::max<std::size_t>(1,n)*sizeof(T)));}
    ~Device(){cudaFree(p);}
    Device(const Device&)=delete;
    void put(const std::vector<T>& v){check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
};
struct Pinned {
    std::uint8_t* host=nullptr;std::uint8_t* mapped=nullptr;
    explicit Pinned(std::size_t bytes) {
        check(cudaHostAlloc(&host,bytes,cudaHostAllocMapped));
        check(cudaHostGetDevicePointer(&mapped,host,0));
    }
    ~Pinned(){cudaFreeHost(host);}
};
struct Graph {
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
    Graph(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);cudaStreamDestroy(stream);}
};

void run(int queries,int count,int capacity,bool unaligned,int cache_sets,int& cases) {
    constexpr int images=31,image_stride=512;
    constexpr std::size_t slot=512;
    const int selections=queries*count;
    const auto arena_bytes=std::size_t(selections)*slot;
    Pinned payload(images*image_stride);
    std::vector<tq::RamBlockView> source(capacity);
    std::vector<int> ids(images);
    for(int i=0;i<images;++i) {
        const int id=i*(capacity-1)/(images-1);ids[i]=id;
        auto& b=source[id];
        b.key_bytes=17+(i*37)%181;b.value_bytes=3+(i*53)%189;
        b.image.key=payload.mapped+i*image_stride+(unaligned?1:0);
        b.image.value=payload.mapped+i*image_stride+256+(unaligned?3:0);
        b.image.key_bits=(i%3==0?0:i%3==1?254:4);
        b.image.value_bits=i%4==0?tq::tq_precision_flags(255,12):b.image.key_bits;
    }
    Device<tq::RamBlockView> sources(capacity);sources.put(source);
    Device<tq::SelectedBlock> selected(selections);
    Device<std::uint32_t> first(capacity);
    Device<tq::TqBlockView> result(selections);
    Device<std::uint8_t> arena(arena_bytes+32);
    Device<std::uint64_t> copied(1);
    Device<std::uint64_t> hit_blocks(1),frame(1);
    Device<tq::RamCacheSlot> slots(cache_sets*tq::ram_cache_ways);
    Device<std::uint32_t> locks(cache_sets);
    Device<std::uint8_t> cached(std::size_t(cache_sets)*tq::ram_cache_ways*slot);
    tq::RamStageCache cache{slots.p,locks.p,frame.p,cached.p,std::size_t(cache_sets)*tq::ram_cache_ways*slot,slot,cache_sets};
    std::vector<tq::SelectedBlock> selection(selections);
    std::vector<tq::TqBlockView> actual(selections);
    std::vector<std::uint8_t> staging(arena_bytes+32);
    std::vector<std::uint8_t> cache_image(std::size_t(cache_sets)*tq::ram_cache_ways*slot);
    Graph graph;
    if(cache_sets)check(cudaError_t(tq::tq_reset_ram_stage_cache(cache,graph.stream)));
    const auto launch=[&] {
        if(cache_sets)check(cudaError_t(tq::tq_stage_ram_selected_cached(sources.p,capacity,selected.p,
            queries,count,first.p,result.p,arena.p+16,arena_bytes,slot,cache,copied.p,hit_blocks.p,graph.stream)));
        else check(cudaError_t(tq::tq_stage_ram_selected(sources.p,capacity,selected.p,
            queries,count,first.p,result.p,arena.p+16,arena_bytes,slot,copied.p,graph.stream)));
    };
    // Populate the per-device resource query before graph capture.
    for(auto& s:selection)s={-1,0};selected.put(selection);launch();
    check(cudaStreamSynchronize(graph.stream));
    check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));launch();
    check(cudaStreamEndCapture(graph.stream,&graph.graph));
    check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
    for(int replay=0;replay<3;++replay) {
        for(std::size_t i=0;i<images*image_stride;++i)
            payload.host[i]=std::uint8_t(i*37+i/53+11*replay);
        for(int id:ids)source[id].generation=replay+1;
        sources.put(source);
        std::set<int> unique;
        for(int i=0;i<selections;++i) {
            auto& s=selection[i];s.id=ids[(i*13+i/count+replay*7)%images];
            s.token_mask=std::uint32_t((i+replay)%16);
            if(i%17==0)s.id=-1;
            if(i%97==0)s.id=capacity; // invalid IDs must not index the table.
            if(s.id>=0 && s.id<capacity && s.token_mask)unique.insert(s.id);
        }
        // These CPU writes happen after the preceding graph has retired;
        // the next upload and graph launch publish both payload and IDs.
        selected.put(selection);
        check(cudaMemsetAsync(arena.p,0x5a,arena_bytes+32,graph.stream));
        check(cudaGraphLaunch(graph.exec,graph.stream));
        check(cudaStreamSynchronize(graph.stream));
        check(cudaMemcpy(actual.data(),result.p,selections*sizeof(actual[0]),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(staging.data(),arena.p,staging.size(),cudaMemcpyDeviceToHost));
        if(cache_sets)check(cudaMemcpy(cache_image.data(),cached.p,cache_image.size(),cudaMemcpyDeviceToHost));
        std::uint64_t transferred=0,expected=0;
        check(cudaMemcpy(&transferred,copied.p,sizeof(transferred),cudaMemcpyDeviceToHost));
        for(int id:unique)expected+=source[id].key_bytes+source[id].value_bytes;
        if(transferred!=expected)throw std::runtime_error("RAM staging copied a repeated block or missed an image");
        for(int i=0;i<16;++i)
            if(staging[i]!=0x5a || staging[staging.size()-16+i]!=0x5a)
                throw std::runtime_error("RAM staging overwrote an arena guard");
        for(int i=0;i<selections;++i) {
            const auto s=selection[i];const auto view=actual[i];
            if(s.id<0 || s.id>=capacity || !s.token_mask) {
                if(view.key || view.value)throw std::runtime_error("Empty selection exposed an image");
                continue;
            }
            const auto& original=source[s.id];
            if(view.key_bits!=original.image.key_bits || view.value_bits!=original.image.value_bits)
                throw std::runtime_error("RAM staging changed a precision descriptor");
            const auto compare=[&](const std::uint8_t* gpu,const std::uint8_t* mapped,std::size_t bytes) {
                const auto offset=std::uintptr_t(gpu)-std::uintptr_t(arena.p);
                const auto cache_offset=std::uintptr_t(gpu)-std::uintptr_t(cached.p);
                const auto host_offset=std::uintptr_t(mapped)-std::uintptr_t(payload.mapped);
                if(cache_sets && cache_offset<=cache_image.size() && bytes<=cache_image.size()-cache_offset) {
                    if(std::memcmp(cache_image.data()+cache_offset,payload.host+host_offset,bytes))
                        throw std::runtime_error("LRU payload is stale after a generation change");
                    return;
                }
                if(offset<16 || offset>arena_bytes+16 || bytes>arena_bytes+16-offset ||
                    std::memcmp(staging.data()+offset,payload.host+host_offset,bytes))
                    throw std::runtime_error("RAM staging differs from its immutable source image");
            };
            compare(view.key,original.image.key,original.key_bytes);
            compare(view.value,original.image.value,original.value_bytes);
        }
        ++cases;
        if(cache_sets) {
            // Reuse the same graph after all source versions and selections
            // remain unchanged. A fitting set must make no second RAM read.
            check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
            std::uint64_t hits=0;check(cudaMemcpy(&hits,hit_blocks.p,8,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(&transferred,copied.p,8,cudaMemcpyDeviceToHost));
            std::vector<int> occupancy(cache_sets);for(int id:unique)++occupancy[tq::ram_cache_set(unsigned(id),unsigned(cache_sets))];
            if(*std::max_element(occupancy.begin(),occupancy.end())<=tq::ram_cache_ways &&
                (transferred || hits!=unique.size()))throw std::runtime_error("Hot LRU still reads unchanged RAM images");
            if(hits>unique.size() || transferred>expected)throw std::runtime_error("LRU accounting exceeds the requested union");
            ++cases;
        }
    }
    if(tq::tq_stage_ram_selected(sources.p,capacity,selected.p,INT_MAX,2,
            first.p,result.p,arena.p,arena_bytes,slot,nullptr,graph.stream)!=cudaErrorInvalidValue ||
       tq::tq_stage_ram_selected(sources.p,capacity,selected.p,queries,count,
            first.p,result.p,arena.p,arena_bytes,slot-1,nullptr,graph.stream)!=cudaErrorInvalidValue ||
       tq::tq_stage_ram_selected(sources.p,capacity,selected.p,queries,count,
            first.p,result.p,arena.p,arena_bytes-1,slot,nullptr,graph.stream)!=cudaErrorInvalidValue)
        throw std::runtime_error("Invalid RAM staging geometry was accepted");
}

void publication(int& cases) {
    constexpr int capacity=16,count=3,slot=512,cache_slot=256;
    Pinned payload(capacity*slot);
    std::vector<tq::TqBlockView> records(capacity);
    for(int id=0;id<capacity;++id)records[id]={payload.mapped+id*slot,payload.mapped+id*slot+256,
        unsigned(id%2?254:0),unsigned(id%2?254:0)};
    Device<tq::TqBlockView> live(capacity),output(count);live.put(records);
    Device<tq::RamBlockView> source(capacity),patch(1);
    Device<std::int32_t> ids(1);ids.put({5});
    Device<std::int64_t> position(1);
    Device<tq::SelectedBlock> selected(count);
    Device<std::uint32_t> first(capacity),locks(1);
    Device<std::uint64_t> copied(1),hits(1),frame(1);
    Device<tq::RamCacheSlot> slots(tq::ram_cache_ways);
    Device<std::uint8_t> arena(count*slot),cached(tq::ram_cache_ways*cache_slot);
    const tq::RamStageCache cache{slots.p,locks.p,frame.p,cached.p,tq::ram_cache_ways*cache_slot,cache_slot,1};
    Graph graph;
    check(cudaMemset(source.p,0,capacity*sizeof(tq::RamBlockView)));
    check(cudaError_t(tq::tq_reset_ram_stage_cache(cache,graph.stream)));
    const auto stage=[&] {
        check(cudaError_t(tq::tq_stage_ram_selected_cached(source.p,capacity,selected.p,1,count,
            first.p,output.p,arena.p,count*slot,slot,cache,copied.p,hits.p,graph.stream)));
    };
    // Prime the device capability query before capturing a changing position.
    selected.put({{-1,0},{-1,0},{-1,0}});stage();check(cudaStreamSynchronize(graph.stream));
    check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));
    check(cudaError_t(tq::tq_publish_ram_append(source.p,capacity,live.p,position.p,0,8,256,128,graph.stream)));
    stage();check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
    for(int replay=0;replay<4;++replay) {
        const int start=3+4*replay;
        position.put({start});
        selected.put({{start/4,15},{start/4+1,15},{start/4+2,7}});
        for(int at=0;at<capacity*slot;++at)payload.host[at]=std::uint8_t(at*13+replay*29);
        check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
        std::vector<tq::TqBlockView> actual(count);check(cudaMemcpy(actual.data(),output.p,count*sizeof(actual[0]),cudaMemcpyDeviceToHost));
        for(int at=0;at<count;++at) {
            const int id=start/4+at,bytes=id%2?128:256;
            std::vector<std::uint8_t> k(bytes),v(bytes);
            check(cudaMemcpy(k.data(),actual[at].key,bytes,cudaMemcpyDeviceToHost));
            check(cudaMemcpy(v.data(),actual[at].value,bytes,cudaMemcpyDeviceToHost));
            if(std::memcmp(k.data(),payload.host+id*slot,bytes) || std::memcmp(v.data(),payload.host+id*slot+256,bytes))
                throw std::runtime_error("Live append reused stale cached data or an incorrect source extent");
        }
        std::uint64_t hit=1;check(cudaMemcpy(&hit,hits.p,8,cudaMemcpyDeviceToHost));
        if(hit)throw std::runtime_error("Live device append entered immutable RAM LRU");
        ++cases;
    }
    // A resident packed cache hit remains readable after RAM is released to
    // SSD. Only a miss requires materializing the RAM source again.
    tq::RamBlockView image{records[5],128,128,37};patch.put({image});
    check(cudaError_t(tq::tq_publish_ram_sources(source.p,capacity,ids.p,patch.p,1,graph.stream)));
    selected.put({{5,15},{5,7},{-1,0}});stage();check(cudaStreamSynchronize(graph.stream));
    image.image.key=image.image.value=nullptr;patch.put({image});
    check(cudaError_t(tq::tq_publish_ram_sources(source.p,capacity,ids.p,patch.p,1,graph.stream)));
    stage();check(cudaStreamSynchronize(graph.stream));
    std::uint64_t read=1,hit=0;check(cudaMemcpy(&read,copied.p,8,cudaMemcpyDeviceToHost));check(cudaMemcpy(&hit,hits.p,8,cudaMemcpyDeviceToHost));
    if(read || hit!=1)throw std::runtime_error("Packed cache hit requested an evicted RAM source");
    ++cases;
}

void selected_hit_protection(int& cases) {
    constexpr int capacity=16,slot=256;
    Pinned payload(capacity*slot);
    for(int i=0;i<capacity*slot;++i)payload.host[i]=std::uint8_t(i*7+11);
    std::vector<tq::RamBlockView> records(capacity);
    for(int i=0;i<capacity;++i)records[i]={{payload.mapped+i*slot,payload.mapped+i*slot+128,4,4},128,128,1};
    Device<tq::RamBlockView> source(capacity);source.put(records);
    Device<tq::SelectedBlock> selected(capacity);
    Device<std::uint32_t> first(capacity),locks(1);
    Device<std::uint64_t> copied(1),hits(1),frame(1);
    Device<tq::TqBlockView> output(capacity);
    Device<std::uint8_t> arena(capacity*slot),cached(tq::ram_cache_ways*slot);
    Device<tq::RamCacheSlot> slots(tq::ram_cache_ways);
    Device<std::int32_t> missing(capacity);Device<int> missing_count(1);
    tq::RamStageCache cache{slots.p,locks.p,frame.p,cached.p,tq::ram_cache_ways*slot,slot,1};
    Graph graph;check(cudaError_t(tq::tq_reset_ram_stage_cache(cache,graph.stream)));
    std::vector<tq::SelectedBlock> selection(capacity);
    for(int i=0;i<capacity;++i)selection[i]={i,i<tq::ram_cache_ways?15u:0u};selected.put(selection);
    const auto stage=[&]{check(cudaError_t(tq::tq_stage_ram_selected_cached(source.p,capacity,selected.p,1,capacity,
        first.p,output.p,arena.p,capacity*slot,slot,cache,copied.p,hits.p,graph.stream)));};
    stage();check(cudaStreamSynchronize(graph.stream));
    for(int i=0;i<tq::ram_cache_ways;++i)records[i].image.key=records[i].image.value=nullptr;
    source.put(records);for(int i=0;i<capacity;++i)selection[i].token_mask=15;selected.put(selection);
    check(cudaError_t(tq::tq_collect_ram_missing(source.p,capacity,selected.p,1,capacity,first.p,missing.p,missing_count.p,cache,graph.stream)));
    int count=-1;check(cudaStreamSynchronize(graph.stream));check(cudaMemcpy(&count,missing_count.p,4,cudaMemcpyDeviceToHost));
    if(count)throw std::runtime_error("Selected packed hits requested an unnecessary SSD read");
    // Eight incoming misses must use transient slots rather than evicting a
    // selected hit that still needs its only readable copy in the GPU cache.
    stage();check(cudaStreamSynchronize(graph.stream));std::uint64_t actual_hits=0,bytes=0;
    check(cudaMemcpy(&actual_hits,hits.p,8,cudaMemcpyDeviceToHost));check(cudaMemcpy(&bytes,copied.p,8,cudaMemcpyDeviceToHost));
    if(actual_hits!=tq::ram_cache_ways || bytes!=(capacity-tq::ram_cache_ways)*slot)
        throw std::runtime_error("Staging displaced a selected cache hit with a RAM miss");
    ++records[0].generation;source.put(records);
    check(cudaError_t(tq::tq_collect_ram_missing(source.p,capacity,selected.p,1,capacity,first.p,missing.p,missing_count.p,cache,graph.stream)));
    check(cudaStreamSynchronize(graph.stream));check(cudaMemcpy(&count,missing_count.p,4,cudaMemcpyDeviceToHost));
    int id=-1;check(cudaMemcpy(&id,missing.p,4,cudaMemcpyDeviceToHost));
    if(count!=1 || id!=0)throw std::runtime_error("SSD fetch did not reject a stale packed generation");
    ++cases;
}

void split_staging(int queries,int cache_sets,bool unaligned,int& cases) {
    constexpr int capacity=37,count=32,slot=256;
    const int total=queries*count;
    Pinned payload(capacity*slot);
    std::vector<std::uint8_t> expected(capacity*slot);
    for(std::size_t i=0;i<expected.size();++i)expected[i]=std::uint8_t(i*19+i/13);
    std::memcpy(payload.host,expected.data(),expected.size());
    std::vector<tq::RamBlockView> full(capacity),initial,hydrated;
    for(int id=0;id<capacity;++id)full[id]={{payload.mapped+id*slot+(unaligned?1:0),payload.mapped+id*slot+128+(unaligned?3:0),
        std::uint32_t(id%3==0?0:id%3==1?254:4),std::uint32_t(id%3==0?0:id%3==1?254:4)},
        unsigned(17+id%61),unsigned(23+id%79),1};
    Device<tq::RamBlockView> sources(capacity),patch(capacity);sources.put(full);
    Device<tq::SelectedBlock> selections(total);std::vector<tq::SelectedBlock> selected(total,{-1,0});
    Device<std::uint32_t> first(capacity),locks(cache_sets);
    Device<tq::TqBlockView> output(total);
    Device<std::uint8_t> arena(total*slot),cached(cache_sets*tq::ram_cache_ways*slot);
    Device<tq::RamCacheSlot> slots(cache_sets*tq::ram_cache_ways);
    Device<std::uint64_t> copied(1),hits(1),frame(1);
    tq::RamStageCache cache{slots.p,locks.p,frame.p,cached.p,std::size_t(cache_sets)*tq::ram_cache_ways*slot,slot,cache_sets};
    Graph graph;
    if(cache_sets)check(cudaError_t(tq::tq_reset_ram_stage_cache(cache,graph.stream)));
    for(int id=0;id<tq::ram_cache_ways;++id)selected[id]={id,15};selections.put(selected);
    if(cache_sets)check(cudaError_t(tq::tq_stage_ram_selected_cached(sources.p,capacity,selections.p,queries,count,
        first.p,output.p,arena.p,total*slot,slot,cache,copied.p,hits.p,graph.stream)));
    initial=hydrated=full;
    // Include unavailable packed hits, a stale packed generation and true
    // SSD misses. Completed resident views are made unreadable at the source
    // before the second step, which must retain their first copy unchanged.
    ++initial[0].generation;++hydrated[0].generation;
    for(int id=0;id<capacity;++id) {
        const bool missing=id<tq::ram_cache_ways || id%3==0;
        if(missing)initial[id].image.key=initial[id].image.value=nullptr;
        else hydrated[id].image.key=hydrated[id].image.value=nullptr;
    }
    check(cudaStreamSynchronize(graph.stream));sources.put(initial);patch.put(hydrated);
    for(int at=0;at<total;++at)selected[at]={(at%17==0?-1:(at*7+at/count)%capacity),unsigned(at%13==0?0:1u<<(at%4))};
    selections.put(selected);
    const auto step=[&](tq::RamStageStep phase){check(cudaError_t(tq::tq_stage_ram_selected_step(sources.p,capacity,selections.p,queries,count,
        first.p,output.p,arena.p,total*slot,slot,cache,phase,copied.p,hits.p,graph.stream)));};
    // Both phases and source publication also work inside a captured graph.
    check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));
    step(tq::RamStageStep::Available);
    check(cudaMemcpyAsync(sources.p,patch.p,capacity*sizeof(tq::RamBlockView),cudaMemcpyDeviceToDevice,graph.stream));
    step(tq::RamStageStep::Hydrated);
    check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
    check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
    std::vector<tq::TqBlockView> views(total);std::vector<std::uint8_t> transient(total*slot),hot(cache_sets*tq::ram_cache_ways*slot);
    check(cudaMemcpy(views.data(),output.p,views.size()*sizeof(views[0]),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(transient.data(),arena.p,transient.size(),cudaMemcpyDeviceToHost));
    if(!hot.empty())check(cudaMemcpy(hot.data(),cached.p,hot.size(),cudaMemcpyDeviceToHost));
    const auto address=[&](const std::uint8_t* p,std::size_t bytes) {
        const auto x=std::uintptr_t(p),a=std::uintptr_t(arena.p),c=std::uintptr_t(cached.p);
        if(x>=a && x-a<=transient.size() && bytes<=transient.size()-(x-a))return transient.data()+(x-a);
        if(x>=c && x-c<=hot.size() && bytes<=hot.size()-(x-c))return hot.data()+(x-c);
        throw std::runtime_error("Split RAM staging produced an invalid payload address");
    };
    std::set<int> unique;std::uint64_t expected_bytes=0,expected_hits=0;
    for(int at=0;at<total;++at) {
        const auto id=selected[at].id;const auto& view=views[at];
        if(id<0 || !selected[at].token_mask) {
            if(view.key || view.value)throw std::runtime_error("Split staging retained an invalid selection");continue;
        }
        const auto& src=full[id];
        if(view.key_bits!=src.image.key_bits || view.value_bits!=src.image.value_bits ||
                std::memcmp(address(view.key,src.key_bytes),expected.data()+id*slot+(unaligned?1:0),src.key_bytes) ||
                std::memcmp(address(view.value,src.value_bytes),expected.data()+id*slot+128+(unaligned?3:0),src.value_bytes))
            throw std::runtime_error("Split SSD hydration changed an original packed image");
        if(unique.insert(id).second) {
            if(cache_sets && id>0 && id<tq::ram_cache_ways)++expected_hits;
            else expected_bytes+=src.key_bytes+src.value_bytes;
        }
    }
    std::uint64_t bytes=0,actual_hits=0,actual_frame=0;
    check(cudaMemcpy(&bytes,copied.p,8,cudaMemcpyDeviceToHost));check(cudaMemcpy(&actual_hits,hits.p,8,cudaMemcpyDeviceToHost));
    if(cache_sets)check(cudaMemcpy(&actual_frame,frame.p,8,cudaMemcpyDeviceToHost));
    if(bytes!=expected_bytes || actual_hits!=expected_hits || (cache_sets && actual_frame!=2))
        throw std::runtime_error("Split staging copied an image twice or advanced the LRU twice");
    ++cases;
}

int main()try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::puts("SKIP: CUDA unavailable");return 77;}
    int device=0,mapped=0;check(cudaGetDevice(&device));check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));
    if(!mapped){std::puts("SKIP: mapped host memory unavailable");return 77;}
    int cases=0;
    for(int capacity:{64,65536})for(int count:{1,33,513})for(int queries:{1,2,3,4,5,6,7,8})
        for(bool unaligned:{false,true})for(int cache_sets:{0,1,64})run(queries,count,capacity,unaligned,cache_sets,cases);
    publication(cases);
    selected_hit_protection(cases);
    for(int queries:{1,2,3,4,5,6,7,8})for(int sets:{0,1,7})for(bool unaligned:{false,true})
        split_staging(queries,sets,unaligned,cases);
    std::printf("QSA RAM union/LRU byte parity, bounds, duplicates, masks, generations and graph replay PASS cases=%d\n",cases);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
