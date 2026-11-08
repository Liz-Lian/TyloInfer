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

void mixed_tier_lru(int queries,int& cases) {
    constexpr int capacity=16,count=8;constexpr std::size_t stride=64;
    Pinned payload(capacity*stride);
    std::vector<tq::RamBlockView> records(capacity);
    for(int id=0;id<capacity;++id) {
        std::memset(payload.host+id*stride,id+7,stride);
        records[id]={{payload.mapped+id*stride,payload.mapped+id*stride+16,4,4},16,16,1};
    }
    Device<tq::RamBlockView> sources(capacity);sources.put(records);
    Device<tq::SelectedBlock> selected(queries*count);
    Device<std::uint32_t> first(capacity),locks(1);
    Device<tq::TqBlockView> output(queries*count);
    Device<std::uint8_t> arena(queries*count*stride),cached(count*stride);
    Device<tq::RamCacheSlot> slots(count);
    Device<std::uint64_t> frame(1),copied(1),hits(1);
    Device<int> missing(queries*count),missing_count(1);
    tq::RamStageCache cache{slots.p,locks.p,frame.p,cached.p,count*stride,stride,1};
    Graph graph;check(cudaError_t(tq::tq_reset_ram_stage_cache(cache,graph.stream)));
    std::vector<tq::SelectedBlock> ids(queries*count);
    for(int at=0;at<queries*count;++at)ids[at]={at%count,15};selected.put(ids);
    check(cudaError_t(tq::tq_stage_ram_selected_cached(sources.p,capacity,selected.p,queries,count,first.p,output.p,
        arena.p,queries*count*stride,stride,cache,copied.p,hits.p,graph.stream)));
    check(cudaStreamSynchronize(graph.stream));
    for(int id=0;id<count;++id)records[id].image.key=records[id].image.value=nullptr;
    sources.put(records);
    // Replacing the complete working set makes the previous SSD images cold.
    // The same bounded LRU also admits frequently reused resident RAM images.
    for(auto& id:ids)id.id+=count;selected.put(ids);
    const auto stage=[&] {check(cudaError_t(tq::tq_stage_ram_selected_step(sources.p,capacity,selected.p,queries,count,
        first.p,output.p,arena.p,queries*count*stride,stride,cache,tq::RamStageStep::Available,copied.p,hits.p,graph.stream)));};
    stage();check(cudaStreamSynchronize(graph.stream));
    for(auto& id:ids)id.id-=count;selected.put(ids);
    check(cudaError_t(tq::tq_collect_ram_missing(sources.p,capacity,selected.p,queries,count,first.p,
        missing.p,missing_count.p,cache,graph.stream)));
    check(cudaStreamSynchronize(graph.stream));int misses=-1;
    check(cudaMemcpy(&misses,missing_count.p,sizeof(misses),cudaMemcpyDeviceToHost));
    if(misses!=count)throw std::runtime_error("Mixed-tier LRU did not replace the cold working set");
    for(int id=0;id<count;++id) {
        records[id].image.key=payload.mapped+id*stride;records[id].image.value=payload.mapped+id*stride+16;
    }
    sources.put(records);
    stage();check(cudaStreamSynchronize(graph.stream));
    std::vector<tq::TqBlockView> views(queries*count);
    check(cudaMemcpy(views.data(),output.p,views.size()*sizeof(views[0]),cudaMemcpyDeviceToHost));
    for(int at=0;at<queries*count;++at)for(auto* pointer:{views[at].key,views[at].value}) {
        std::uint8_t bytes[16]{};check(cudaMemcpy(bytes,pointer,16,cudaMemcpyDeviceToHost));
        for(auto value:bytes)if(value!=std::uint8_t(at%count+7))throw std::runtime_error("Restored mixed-tier LRU bytes changed");
    }
    ++cases;
}

void mixed_slot_groups(int queries,int& cases) {
    constexpr int capacity=32,count=8,base=64,stride=1024;
    Pinned payload(capacity*stride);
    std::vector<tq::RamBlockView> records(capacity);
    const auto replace=[&](int id,int bytes,unsigned generation) {
        std::memset(payload.host+id*stride,id+generation*17,stride);
        records[id]={{payload.mapped+id*stride,payload.mapped+id*stride+512,unsigned(id%3==0?0:id%3==1?254:4),4},
            unsigned(bytes),unsigned(bytes),generation};
    };
    replace(0,112,1);replace(1,48,1);replace(2,16,1);replace(3,16,1);
    replace(4,16,1);replace(5,496,1);replace(6,240,1);
    Device<tq::RamBlockView> source(capacity);source.put(records);
    Device<tq::SelectedBlock> selected(queries*count);Device<tq::TqBlockView> output(queries*count);
    Device<std::uint32_t> first(capacity),locks(1);Device<tq::RamCacheSlot> slots(8);
    Device<std::uint8_t> arena(queries*count*stride),cached(8*base);
    Device<std::uint64_t> frame(1),copied(1),hits(1);Device<int> missing(queries*count),missing_count(1);
    tq::RamStageCache cache{slots.p,locks.p,frame.p,cached.p,8*base,base,1};Graph graph;
    std::vector<tq::SelectedBlock> selections(queries*count,{-1,0});
    const auto select=[&](const std::vector<int>& ids) {
        std::fill(selections.begin(),selections.end(),tq::SelectedBlock{-1,0});
        for(int q=0;q<queries;++q)for(std::size_t at=0;at<ids.size();++at)selections[q*count+at]={ids[at],unsigned(q%2?7:15)};
        selected.put(selections);
    };
    const auto stage=[&]{check(cudaError_t(tq::tq_stage_ram_selected_cached(source.p,capacity,selected.p,queries,count,
        first.p,output.p,arena.p,queries*count*stride,stride,cache,copied.p,hits.p,graph.stream)));};
    const auto verify=[&] {
        std::vector<tq::TqBlockView> views(queries*count);check(cudaMemcpy(views.data(),output.p,views.size()*sizeof(views[0]),cudaMemcpyDeviceToHost));
        for(int at=0;at<queries*count;++at)if(selections[at].token_mask) {
            const auto id=selections[at].id;const auto& original=records[id];
            for(auto* p:{views[at].key,views[at].value}) {
                std::vector<std::uint8_t> bytes(original.key_bytes);check(cudaMemcpy(bytes.data(),p,bytes.size(),cudaMemcpyDeviceToHost));
                for(auto b:bytes)if(b!=std::uint8_t(id+original.generation*17))throw std::runtime_error("Mixed-size LRU changed opaque bytes");
            }
        }
    };
    const auto counter=[&](const auto& value){std::uint64_t result=0;check(cudaMemcpy(&result,value.p,8,cudaMemcpyDeviceToHost));return result;};
    check(cudaError_t(tq::tq_reset_ram_stage_cache(cache,graph.stream)));select({});stage();check(cudaStreamSynchronize(graph.stream));
    check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));stage();check(cudaStreamEndCapture(graph.stream,&graph.graph));
    check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
    const auto replay=[&]{check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));verify();};
    select({0,1,2,3,0,1});replay();
    if(counter(copied)!=384)throw std::runtime_error("Mixed-size LRU copied a repeated image");
    for(int id=0;id<4;++id)records[id].image.key=records[id].image.value=nullptr;source.put(records);replay();
    if(counter(copied) || counter(hits)!=4)throw std::runtime_error("Mixed-size LRU failed a fitting warm working set");
    replace(0,112,2);source.put(records);replay();
    if(counter(copied)!=224 || counter(hits)!=3)throw std::runtime_error("Mixed-size LRU reused a stale group generation");
    records[0].image.key=records[0].image.value=nullptr;source.put(records);
    select({0,1,2,3,4});replay();
    if(counter(copied)!=32 || counter(hits)!=4)throw std::runtime_error("Mixed-size LRU displaced a protected group");
    std::vector<tq::RamCacheSlot> before(8),after(8);check(cudaMemcpy(before.data(),slots.p,8*sizeof(before[0]),cudaMemcpyDeviceToHost));
    select({4});replay();check(cudaMemcpy(after.data(),slots.p,8*sizeof(after[0]),cudaMemcpyDeviceToHost));
    for(int way=0;way<8;++way)if(before[way].id>=0 && after[way].id!=before[way].id) {
        const auto victim=before[way].id;for(const auto& slot:after)if(slot.id==victim)throw std::runtime_error("Mixed-size eviction retained a partial owner");
    }
    check(cudaMemcpy(before.data(),slots.p,8*sizeof(before[0]),cudaMemcpyDeviceToHost));
    select({5});replay();check(cudaMemcpy(after.data(),slots.p,8*sizeof(after[0]),cudaMemcpyDeviceToHost));
    if(counter(copied)!=992)throw std::runtime_error("Oversized image did not use transient staging");
    for(int way=0;way<8;++way)if(after[way].id!=before[way].id || after[way].generation!=before[way].generation)
        throw std::runtime_error("Oversized image replaced a bounded cache allocation");
    select({6});replay();replay();if(counter(copied) || counter(hits)!=1)throw std::runtime_error("Whole-set image did not reuse eight ways");
    ++cases;
}

void contiguous_publication(int& cases) {
    for(int count:{0,1,3,513})for(int bytes:{17,2048})for(unsigned tag:{0u,254u,tq::tq_precision_flags(255,12)}) {
        Device<std::uint8_t> planes(std::size_t(2)*std::max(1,count)*bytes+16);
        Device<tq::TqBlockView> output(count+2);
        Graph graph;
        tq::TqBlockView base{planes.p+1,planes.p+1+std::max(1,count)*bytes,tag,tag^1u};
        const auto launch=[&] {check(cudaError_t(tq::tq_publish_contiguous_views(base,output.p+1,bytes,count,graph.stream)));};
        check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));launch();
        check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
        for(int replay=0;replay<2;++replay) {
            check(cudaMemsetAsync(output.p,0xA5,(count+2)*sizeof(tq::TqBlockView),graph.stream));
            check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
            std::vector<tq::TqBlockView> actual(count+2);
            check(cudaMemcpy(actual.data(),output.p,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost));
            for(int i=0;i<count;++i) {
                const auto& v=actual[i+1];
                if(v.key!=base.key+std::size_t(i)*bytes || v.value!=base.value+std::size_t(i)*bytes ||
                    v.key_bits!=base.key_bits || v.value_bits!=base.value_bits)
                    throw std::runtime_error("Contiguous image publication changed plane offsets or flags");
            }
            for(int i:{0,count+1})for(auto byte:std::vector<std::uint8_t>(
                    reinterpret_cast<std::uint8_t*>(&actual[i]),reinterpret_cast<std::uint8_t*>(&actual[i])+sizeof(actual[i])))
                if(byte!=0xA5)throw std::runtime_error("Contiguous image publication exceeded output bounds");
        }
        if(tq::tq_publish_contiguous_views(base,output.p,bytes,-1,graph.stream)!=int(cudaErrorInvalidValue) ||
            tq::tq_publish_contiguous_views(base,output.p,0,count,graph.stream)!=int(cudaErrorInvalidValue))
            throw std::runtime_error("Contiguous image publication accepted invalid extents");
        ++cases;
    }
}

void mapped_record_publication(int& cases) {
    for(int count:{0,1,31,257})for(bool mapped:{false,true})for(bool update_live:{false,true}) {
        const int capacity=2*count+17;
        Pinned host_ids(std::max(1,count)*sizeof(std::int32_t));
        Pinned host_records(std::max(1,count)*sizeof(tq::RamBlockView));
        auto* ids=reinterpret_cast<std::int32_t*>(host_ids.host);
        auto* records=reinterpret_cast<tq::RamBlockView*>(host_records.host);
        Device<std::int32_t> device_ids(count);Device<tq::RamBlockView> device_records(count),output(capacity);
        Device<tq::TqBlockView> live(capacity);Device<std::uint8_t> payload(32*std::max(1,count));
        Graph graph;
        const auto launch=[&] {
            check(cudaError_t(tq::tq_publish_ram_sources(output.p,capacity,
                mapped?reinterpret_cast<const std::int32_t*>(host_ids.mapped):device_ids.p,
                mapped?reinterpret_cast<const tq::RamBlockView*>(host_records.mapped):device_records.p,
                count,graph.stream,update_live?live.p:nullptr)));
        };
        check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));launch();
        check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
        for(int replay=0;replay<3;++replay) {
            std::vector<tq::RamBlockView> expected(capacity);
            std::memset(expected.data(),0,expected.size()*sizeof(expected[0]));
            for(int i=0;i<count;++i) {
                ids[i]=(13*i+7*replay)%capacity;records[i]={};
                auto& r=records[i];r.generation=(std::uint64_t(replay+1)<<40)+i;
                r.key_bytes=17+i%11;r.value_bytes=3+i%7;
                r.image={i%3?payload.p+32*i:nullptr,i%3?payload.p+32*i+16:nullptr,
                    unsigned(i%2?254:0),tq::tq_precision_flags(255,12+replay)};
                expected[ids[i]]=r;
            }
            if(!mapped && count) {
                device_ids.put(std::vector<std::int32_t>(ids,ids+count));
                device_records.put(std::vector<tq::RamBlockView>(records,records+count));
            }
            check(cudaMemsetAsync(output.p,0,capacity*sizeof(tq::RamBlockView),graph.stream));
            check(cudaMemsetAsync(live.p,0,capacity*sizeof(tq::TqBlockView),graph.stream));
            check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
            std::vector<tq::RamBlockView> actual(capacity);std::vector<tq::TqBlockView> actual_live(capacity);
            check(cudaMemcpy(actual.data(),output.p,actual.size()*sizeof(actual[0]),cudaMemcpyDeviceToHost));
            check(cudaMemcpy(actual_live.data(),live.p,actual_live.size()*sizeof(actual_live[0]),cudaMemcpyDeviceToHost));
            for(int id=0;id<capacity;++id) {
                const auto& a=actual[id];const auto& b=expected[id];
                if(a.generation!=b.generation || a.key_bytes!=b.key_bytes || a.value_bytes!=b.value_bytes ||
                        std::memcmp(&a.image,&b.image,sizeof(a.image)))
                    throw std::runtime_error("Mapped/device RAM descriptor publication lost changed IDs, versions or bounds");
                const tq::TqBlockView none{nullptr,nullptr,0,0};const auto& image=update_live?b.image:none;
                if(std::memcmp(&actual_live[id],&image,sizeof(image)))
                    throw std::runtime_error("RAM descriptor publication changed optional live views");
            }
        }
        ++cases;
    }
}

void changing_cache_layout(int queries,int& cases) {
    constexpr int capacity=128,count=11,maximum_sets=512;
    constexpr std::size_t bytes=maximum_sets*tq::ram_cache_ways*64;
    Pinned payload(capacity*512);Device<std::uint8_t> cached(bytes),arena(queries*count*256);
    Device<tq::RamCacheSlot> slots(maximum_sets*tq::ram_cache_ways);
    Device<std::uint32_t> locks(maximum_sets),first(capacity);
    Device<std::uint64_t> frame(1),copied(1),hits(1);
    Device<tq::RamBlockView> source(capacity);Device<tq::SelectedBlock> selected(queries*count);
    Device<tq::TqBlockView> output(queries*count);Device<tq::RamStageCache> descriptor(1);
    tq::RamStageCache envelope{slots.p,locks.p,frame.p,cached.p,bytes,64,maximum_sets};
    std::vector<int> ids;std::set<unsigned> used;
    for(int id=0;id<capacity && ids.size()<count;++id)
        if(used.insert(tq::ram_cache_set(id,128)).second)ids.push_back(id);
    std::vector<tq::SelectedBlock> selection(queries*count);
    for(int q=0;q<queries;++q)for(int i=0;i<count;++i)selection[q*count+i]={ids[(i+q)%count],unsigned(1+(i+q)%15)};
    selected.put(selection);Graph graph;
    // Populate the resource query before capture. Only the descriptor, not
    // this graph or its allocation envelope, changes between replays.
    auto layout=envelope;layout.slot_bytes=256;layout.sets=int(bytes/256/tq::ram_cache_ways);
    descriptor.put({layout});
    check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));
    check(cudaError_t(tq::tq_stage_ram_selected_cached(source.p,capacity,selected.p,queries,count,
        first.p,output.p,arena.p,queries*count*256,256,envelope,copied.p,hits.p,graph.stream,descriptor.p)));
    check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
    int version=0;
    for(int stride:{256,128,64,256,64}) {
        ++version;layout=envelope;layout.slot_bytes=stride;layout.sets=int(bytes/stride/tq::ram_cache_ways);
        check(cudaError_t(tq::tq_reset_ram_stage_cache(layout,graph.stream)));descriptor.put({layout});
        std::vector<tq::RamBlockView> records(capacity);std::uint64_t expected_bytes=0;
        for(int i=0;i<count;++i) {
            const int id=ids[i];auto& r=records[id];r.generation=version;
            r.key_bytes=i%3==0?17:i%3==1?65:129;r.value_bytes=i%3==0?31:i%3==1?47:97;
            r.image={payload.mapped+id*512+1,payload.mapped+id*512+259,unsigned(i%3?254:0),4};
            for(int j=0;j<512;++j)payload.host[id*512+j]=std::uint8_t(j*13+id+version*29);
            expected_bytes+=r.key_bytes+r.value_bytes;
        }
        for(bool warm:{false,true}) {
            source.put(records);check(cudaGraphLaunch(graph.exec,graph.stream));check(cudaStreamSynchronize(graph.stream));
            std::uint64_t moved=0,found=0;check(cudaMemcpy(&moved,copied.p,8,cudaMemcpyDeviceToHost));check(cudaMemcpy(&found,hits.p,8,cudaMemcpyDeviceToHost));
            if(moved!=(warm?0:expected_bytes) || found!=(warm?count:0))
                throw std::runtime_error("Retained RAM Graph used a stale cache slot size or set count");
            std::vector<tq::TqBlockView> views(queries*count);check(cudaMemcpy(views.data(),output.p,views.size()*sizeof(views[0]),cudaMemcpyDeviceToHost));
            for(int at=0;at<queries*count;++at) {
                const auto id=selection[at].id;const auto& r=records[id];
                std::vector<std::uint8_t> key(r.key_bytes),value(r.value_bytes);
                check(cudaMemcpy(key.data(),views[at].key,key.size(),cudaMemcpyDeviceToHost));
                check(cudaMemcpy(value.data(),views[at].value,value.size(),cudaMemcpyDeviceToHost));
                if(std::memcmp(key.data(),payload.host+id*512+1,key.size()) ||
                        std::memcmp(value.data(),payload.host+id*512+259,value.size()))
                    throw std::runtime_error("Dynamic cache layout changed an opaque image during Graph replay");
            }
            for(int id:ids)records[id].image.key=records[id].image.value=nullptr;
        }
    }
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
    for(int queries:{1,2,3,4,5,6,7,8})mixed_tier_lru(queries,cases);
    for(int queries:{1,2,3,4,5,6,7,8})mixed_slot_groups(queries,cases);
    contiguous_publication(cases);
    mapped_record_publication(cases);
    for(int queries:{1,2,3,4,5,6,7,8})changing_cache_layout(queries,cases);
    std::printf("QSA RAM union/LRU byte parity, bounds, duplicates, masks, generations and graph replay PASS cases=%d\n",cases);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
