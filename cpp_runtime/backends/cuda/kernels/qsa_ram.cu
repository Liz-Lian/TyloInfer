#include "qsa_ram.h"
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>

namespace tyloinfer::qsa {
namespace {
__global__ void publish_sources(RamBlockView* destination,const std::int32_t* ids,
                               const RamBlockView* source,int count,int capacity,TqBlockView* live) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=count)return;
    const int id=ids[at];if(id<0 || id>=capacity){asm volatile("trap;");return;}
    destination[id]=source[at];if(live)live[id]=source[at].image;
}
__global__ void publish_append(RamBlockView* destination,const TqBlockView* live,
        const std::int64_t* position,std::int64_t host_position,int tokens,
        unsigned fp16_bytes,unsigned affine8_bytes,int capacity) {
    const auto start=position?*position:host_position;
    const auto id=start/4+std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(id>=(start+tokens+3)/4)return;
    if(start<0 || id<0 || id>=capacity){asm volatile("trap;");return;}
    const auto image=live[id];
    const unsigned k=image.key_bits&255u,v=image.value_bits&255u;
    if((k!=0 && k!=254) || (v!=0 && v!=254)){asm volatile("trap;");return;}
    destination[id]={image,k?affine8_bytes:fp16_bytes,v?affine8_bytes:fp16_bytes,ram_uncached_generation};
}
__global__ void mark_first_use(const SelectedBlock* selected,std::uint32_t* first,
                              int count,int capacity,std::uint64_t* frame=nullptr) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(!at && frame)atomicAdd(reinterpret_cast<unsigned long long*>(frame),1ULL);
    if(at>=count)return;
    const auto s=selected[at];
    if(s.id>=0 && s.id<capacity && s.token_mask)atomicMin(first+s.id,unsigned(at));
}

template<bool Warp=false>
__device__ void copy_image(std::uint8_t* out,const std::uint8_t* in,unsigned bytes) {
    const unsigned thread=Warp?threadIdx.x&31:threadIdx.x,threads=Warp?32:blockDim.x;
    // Archives can have only byte alignment. Aligned images use coalesced
    // vector reads; the remaining bytes never read beyond the source image.
    if(((reinterpret_cast<std::uintptr_t>(out)|reinterpret_cast<std::uintptr_t>(in))&15u)==0) {
        for(std::size_t at=thread*16;at+16<=bytes;at+=threads*16)
            *reinterpret_cast<uint4*>(out+at)=__ldg(reinterpret_cast<const uint4*>(in+at));
        const std::size_t tail=bytes&~15u;
        for(std::size_t at=tail+thread;at<bytes;at+=threads)out[at]=in[at];
    }else {
        for(std::size_t at=thread;at<bytes;at+=threads)out[at]=in[at];
    }
}

__global__ void reset_cache_slots(RamCacheSlot* slots,int count) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<count)slots[at]=RamCacheSlot{};
}

__global__ void initialize_staging(std::uint32_t* first,int capacity,RamStageCache cache,
        std::uint64_t* copied,std::uint64_t* hits) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at<capacity)first[at]=UINT_MAX;
    if(at<cache.sets)cache.locks[at]=0;
    if(!at) {
        if(cache.frame)++*cache.frame;
        if(copied)*copied=0;
        if(hits)*hits=0;
    }
}

template<bool Resume=false> __global__ void bucket_cache_sets(const SelectedBlock* selected,std::uint32_t* first,
        TqBlockView* output,int count,int capacity,RamStageCache cache) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=count)return;
    const auto s=selected[at];
    if(s.id<0 || s.id>=capacity || !s.token_mask)return;
    // Any selection position can represent a logical block. Claim it once,
    // then publish its set link in this same kernel; duplicate masks remain
    // per query and are applied by attention after snapshot resolution.
    if constexpr(Resume) {
        if(first[s.id]!=unsigned(at) || output[at].key)return;
    } else if(atomicCAS(first+s.id,UINT_MAX,unsigned(at))!=UINT_MAX)return;
    const auto set=ram_cache_set(unsigned(s.id),unsigned(cache.sets));
    const auto next=atomicExch(cache.locks+set,unsigned(at)+1);
    // The list exists only until the following kernel replaces each unique
    // output with its actual image. No consumer observes these tagged links.
    output[at].key=reinterpret_cast<const std::uint8_t*>(std::uintptr_t(next));
}

__device__ void find_cache_slot(const RamBlockView* source,const std::uint32_t* first,
        int capacity,const RamStageCache& cache,int set,int selected_id,std::uint64_t selected_generation,
        std::uint64_t frame,int& slot,int& hit) {
    const int lane=threadIdx.x&31,base=set*ram_cache_ways;
    int id=-1;unsigned long long generation=0,last=0;
    if(lane<ram_cache_ways) {
        id=__ldcg(&cache.slots[base+lane].id);
        generation=__ldcg(reinterpret_cast<const unsigned long long*>(&cache.slots[base+lane].generation));
        last=__ldcg(reinterpret_cast<const unsigned long long*>(&cache.slots[base+lane].last_frame));
    }
    const unsigned matching=__ballot_sync(0xffffffff,lane<ram_cache_ways && id==selected_id && generation==selected_generation);
    hit=matching!=0;
    if(hit){slot=base+__ffs(matching)-1;return;}
    const bool protected_hit=lane<ram_cache_ways && id>=0 && id<capacity && first[id]!=UINT_MAX && source[id].generation==generation;
    const bool eligible=lane<ram_cache_ways && !protected_hit && last!=frame;
    const unsigned empty=__ballot_sync(0xffffffff,eligible && id<0);
    // Match serial LRU ties: the last empty way wins; occupied ways prefer
    // the oldest frame, then the first way. Check all ways in parallel.
    if(empty){slot=base+31-__clz(empty);return;}
    unsigned way=eligible && id>=0 && last!=~0ULL?unsigned(lane):UINT_MAX;
    auto oldest=way==UINT_MAX?~0ULL:last;
    #pragma unroll
    for(int offset=ram_cache_ways/2;offset;offset/=2) {
        const auto other=__shfl_down_sync(0xffffffff,oldest,offset);
        const auto other_way=__shfl_down_sync(0xffffffff,way,offset);
        if(other<oldest || (other==oldest && other_way<way)){oldest=other;way=other_way;}
    }
    way=__shfl_sync(0xffffffff,way,0);slot=way==UINT_MAX?-1:base+int(way);
}

template<bool AllowMissing=false> __global__ void stage_cached(const RamBlockView* source,const SelectedBlock* selected,
        const std::uint32_t* first,TqBlockView* output,std::uint8_t* arena,
        std::size_t stride,int count,int capacity,RamStageCache cache,
        std::uint64_t* copied,std::uint64_t* hits) {
    const auto frame=*cache.frame;
    __shared__ RamBlockView record;
    __shared__ TqBlockView destination;
    __shared__ int slot,hit;
    __shared__ unsigned next;
    // One complete CTA owns each set. Its warps cooperate on the payload,
    // preserving enough outstanding reads even for a small packed cache.
    // Set ownership keeps the LRU deterministic and needs no inter-CTA lock.
    for(std::int64_t set=blockIdx.x;set<cache.sets;set+=gridDim.x) {
        auto link=cache.locks[set];
        while(link) {
            const auto at=std::int64_t(link)-1;
            const auto s=selected[at];
            if(threadIdx.x<32) {
                const auto local_record=source[s.id];
                const auto extent=(std::size_t(local_record.key_bytes)+15u)&~std::size_t(15u);
                if(extent+local_record.value_bytes>stride){asm volatile("trap;");return;}
                int chosen=-1,found=0;
                if(local_record.generation!=ram_uncached_generation && extent+local_record.value_bytes<=cache.slot_bytes)
                    find_cache_slot(source,first,capacity,cache,int(set),s.id,local_record.generation,frame,chosen,found);
                if(!threadIdx.x) {
                    next=unsigned(reinterpret_cast<std::uintptr_t>(output[at].key));
                    record=local_record;slot=chosen;hit=found;
                    if(!hit && (!record.image.key || !record.image.value)) {
                        if constexpr(AllowMissing)destination={};
                        else {asm volatile("trap;");return;}
                    } else {
                        auto* key=slot<0?arena+std::size_t(at)*stride:cache.payload+std::size_t(slot)*cache.slot_bytes;
                        destination={key,key+extent,record.image.key_bits,record.image.value_bits};
                    }
                }
            }
            __syncthreads();
            const auto following=next;
            if(destination.key && !hit) {
                copy_image(const_cast<std::uint8_t*>(destination.key),record.image.key,record.key_bytes);
                copy_image(const_cast<std::uint8_t*>(destination.value),record.image.value,record.value_bytes);
                __syncthreads();
            }
            if(!threadIdx.x) {
                output[at]=destination;
                if(destination.key) {
                    if(slot>=0)cache.slots[slot]={s.id,record.generation,frame};
                    if(hit && hits)atomicAdd(reinterpret_cast<unsigned long long*>(hits),1ULL);
                    if(!hit && copied)atomicAdd(reinterpret_cast<unsigned long long*>(copied),
                        static_cast<unsigned long long>(record.key_bytes)+record.value_bytes);
                }
            }
            // Capture the next link before any warp overwrites shared state.
            __syncthreads();link=following;
        }
    }
}

__global__ void collect_missing(const RamBlockView* source,const SelectedBlock* selected,
        const std::uint32_t* first,std::int32_t* missing,int* missing_count,
        int total,int capacity,RamStageCache cache) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=total)return;
    const auto s=selected[at];if(s.id<0 || s.id>=capacity || !s.token_mask || first[s.id]!=unsigned(at))return;
    const auto record=source[s.id];if(record.image.key && record.image.value)return;
    if(cache.sets && record.generation!=ram_uncached_generation) {
        const int base=int(ram_cache_set(unsigned(s.id),unsigned(cache.sets)))*ram_cache_ways;
        for(int way=0;way<ram_cache_ways;++way)
            if(cache.slots[base+way].id==s.id && cache.slots[base+way].generation==record.generation)return;
    }
    missing[atomicAdd(missing_count,1)]=s.id;
}

__global__ void resolve_cached_views(const SelectedBlock* selected,const std::uint32_t* first,
                                    TqBlockView* output,int total,int capacity) {
    const auto at=std::int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(at>=total)return;
    const auto s=selected[at];
    if(s.id<0 || s.id>=capacity || !s.token_mask)output[at]=TqBlockView{};
    else if(first[s.id]!=unsigned(at))output[at]=output[first[s.id]];
}

bool valid_cache(const RamStageCache& c) {
    return c.sets>0 && c.sets<=INT_MAX/ram_cache_ways && c.slots && c.locks && c.frame && c.payload &&
        c.slot_bytes && !(c.slot_bytes&15u) && !(reinterpret_cast<std::uintptr_t>(c.payload)&15u) &&
        c.slot_bytes<=std::numeric_limits<std::size_t>::max()/std::size_t(c.sets)/ram_cache_ways &&
        c.payload_bytes>=c.slot_bytes*std::size_t(c.sets)*ram_cache_ways;
}

template<int Step=0> __global__ void stage_selected(const RamBlockView* source,const SelectedBlock* selected,
        const std::uint32_t* first,TqBlockView* output,std::uint8_t* arena,
        std::size_t stride,int count,int capacity,std::uint64_t* bytes) {
    for(std::int64_t at=blockIdx.x;at<count;at+=gridDim.x) {
        const auto s=selected[at];
        if(s.id<0 || s.id>=capacity || !s.token_mask) {
            if(!threadIdx.x)output[at]=TqBlockView{};
            continue;
        }
        if constexpr(Step==2)if(output[at].key)continue;
        const auto record=source[s.id];
        const std::size_t key_extent=(std::size_t(record.key_bytes)+15u)&~std::size_t(15u);
        if constexpr(Step==1)if(!record.image.key || !record.image.value) {
            if(!threadIdx.x)output[at]={};continue;
        }
        if(std::uint64_t(key_extent)+record.value_bytes>stride || !record.image.key || !record.image.value) {
            // Publishing an incomplete resident image violates the same
            // invariant as tq_snapshot_blocks; never silently use zero KV.
            asm volatile("trap;");return;
        }
        const unsigned slot=first[s.id];
        auto* key=arena+std::size_t(slot)*stride;
        auto* value=key+key_extent;
        if(!threadIdx.x)output[at]={key,value,record.image.key_bits,record.image.value_bits};
        if(slot!=unsigned(at))continue;
        copy_image(key,record.image.key,record.key_bytes);
        copy_image(value,record.image.value,record.value_bytes);
        if(!threadIdx.x && bytes)
            atomicAdd(reinterpret_cast<unsigned long long*>(bytes),
                static_cast<unsigned long long>(record.key_bytes)+record.value_bytes);
    }
}

int staging_ctas(int count,int* result) {
    // Bound outstanding PCIe reads using the actual GPU's resources.
    // Cache the query on each worker; a device change refreshes it.
    static thread_local int cached_device=-1,cached_sms=0;
    int device=0;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    if(device!=cached_device) {
        error=cudaDeviceGetAttribute(&cached_sms,cudaDevAttrMultiProcessorCount,device);
        if(error!=cudaSuccess)return error;
        cached_device=device;
    }
    *result=int(std::min<std::int64_t>(count,4LL*cached_sms));return cudaSuccess;
}
}

int tq_stage_ram_selected(const RamBlockView* sources,int capacity,
        const SelectedBlock* selected,int queries,int count,
        std::uint32_t* first,TqBlockView* output,void* arena,
        std::size_t arena_bytes,std::size_t stride,
        std::uint64_t* copied_bytes,cudaStream_t stream) {
    if(queries<0 || count<0 || capacity<0)return cudaErrorInvalidValue;
    if(!queries || !count)return cudaSuccess;
    if(!sources || !selected || !first || !output || !arena || !capacity ||
            !stride || (stride&15u) || (reinterpret_cast<std::uintptr_t>(arena)&15u) ||
            queries>INT_MAX/count)return cudaErrorInvalidValue;
    const int total=queries*count;
    if(stride>std::numeric_limits<std::size_t>::max()/std::size_t(total) ||
            arena_bytes<std::size_t(total)*stride)return cudaErrorInvalidValue;
    int ctas=0;auto error=staging_ctas(total,&ctas);if(error!=cudaSuccess)return error;
    error=cudaMemsetAsync(first,0xff,std::size_t(capacity)*sizeof(*first),stream);
    if(error!=cudaSuccess)return error;
    if(copied_bytes) {
        error=cudaMemsetAsync(copied_bytes,0,sizeof(*copied_bytes),stream);
        if(error!=cudaSuccess)return error;
    }
    mark_first_use<<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,total,capacity);
    error=cudaGetLastError();if(error!=cudaSuccess)return error;
    stage_selected<><<<ctas,128,0,stream>>>(sources,selected,first,output,
        static_cast<std::uint8_t*>(arena),stride,total,capacity,copied_bytes);
    return cudaGetLastError();
}

int tq_publish_ram_sources(RamBlockView* sources,int capacity,const std::int32_t* ids,
        const RamBlockView* images,int count,cudaStream_t stream,TqBlockView* live) {
    if(count<0 || capacity<0)return cudaErrorInvalidValue;if(!count)return cudaSuccess;
    if(!sources || !ids || !images || !capacity)return cudaErrorInvalidValue;
    publish_sources<<<unsigned((std::int64_t(count)+255)/256),256,0,stream>>>(sources,ids,images,count,capacity,live);
    return cudaGetLastError();
}
int tq_publish_ram_append(RamBlockView* sources,int capacity,const TqBlockView* live,
        const std::int64_t* position,std::int64_t host_position,int tokens,
        std::uint32_t fp16_bytes,std::uint32_t affine8_bytes,cudaStream_t stream) {
    if(!sources || !live || capacity<=0 || tokens<=0 || !fp16_bytes || !affine8_bytes ||
        (!position && (host_position<0 || host_position+tokens>std::int64_t(capacity)*4)))return cudaErrorInvalidValue;
    const auto maximum_blocks=(std::int64_t(tokens)+6)/4;
    publish_append<<<unsigned((maximum_blocks+127)/128),128,0,stream>>>(sources,live,position,host_position,tokens,
        fp16_bytes,affine8_bytes,capacity);
    return cudaGetLastError();
}

int tq_reset_ram_stage_cache(const RamStageCache& cache,cudaStream_t stream) {
    if(!valid_cache(cache))return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(cache.locks,0,std::size_t(cache.sets)*4,stream);
    if(error!=cudaSuccess)return error;
    error=cudaMemsetAsync(cache.frame,0,sizeof(*cache.frame),stream);if(error!=cudaSuccess)return error;
    const auto slots=std::int64_t(cache.sets)*ram_cache_ways;
    reset_cache_slots<<<unsigned((slots+255)/256),256,0,stream>>>(cache.slots,int(slots));
    return cudaGetLastError();
}

int tq_collect_ram_missing(const RamBlockView* sources,int capacity,const SelectedBlock* selected,
        int queries,int count,std::uint32_t* first,std::int32_t* missing,int* missing_count,
        const RamStageCache& cache,cudaStream_t stream) {
    if(capacity<=0 || queries<0 || count<0 || (count && queries>INT_MAX/count) ||
        !sources || !first || !missing || !missing_count || (!selected && queries && count) ||
        (cache.sets && !valid_cache(cache)))return cudaErrorInvalidValue;
    auto error=cudaMemsetAsync(missing_count,0,sizeof(*missing_count),stream);if(error!=cudaSuccess)return error;
    if(!queries || !count)return cudaSuccess;
    const int total=queries*count;
    error=cudaMemsetAsync(first,0xff,std::size_t(capacity)*sizeof(*first),stream);if(error!=cudaSuccess)return error;
    mark_first_use<<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,total,capacity);
    error=cudaGetLastError();if(error!=cudaSuccess)return error;
    collect_missing<<<unsigned((std::int64_t(total)+127)/128),128,0,stream>>>(sources,selected,first,missing,missing_count,total,capacity,cache);
    return cudaGetLastError();
}

int tq_stage_ram_selected_cached(const RamBlockView* sources,int capacity,
        const SelectedBlock* selected,int queries,int count,std::uint32_t* first,
        TqBlockView* output,void* arena,std::size_t arena_bytes,std::size_t stride,
        const RamStageCache& cache,std::uint64_t* copied,std::uint64_t* hits,cudaStream_t stream) {
    if(queries<0 || count<0 || capacity<0)return cudaErrorInvalidValue;
    if(!queries || !count)return cudaSuccess;
    if(!valid_cache(cache) || !stride || (stride&15u) || !sources || !capacity ||
        !selected || !first || !output || !arena || (reinterpret_cast<std::uintptr_t>(arena)&15u) ||
        queries>INT_MAX/count)return cudaErrorInvalidValue;
    const int total=queries*count;
    if(stride>std::numeric_limits<std::size_t>::max()/std::size_t(total) ||
        arena_bytes<std::size_t(total)*stride)return cudaErrorInvalidValue;
    int ctas=0;auto error=staging_ctas(cache.sets,&ctas);if(error!=cudaSuccess)return error;
    initialize_staging<<<unsigned((std::int64_t(std::max(capacity,cache.sets))+255)/256),256,0,stream>>>(first,capacity,cache,copied,hits);
    error=cudaGetLastError();if(error!=cudaSuccess)return error;
    bucket_cache_sets<><<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,output,total,capacity,cache);
    error=cudaGetLastError();if(error!=cudaSuccess)return error;
    stage_cached<><<<ctas,512,0,stream>>>(sources,selected,first,output,
        static_cast<std::uint8_t*>(arena),stride,total,capacity,cache,copied,hits);
    error=cudaGetLastError();if(error!=cudaSuccess)return error;
    resolve_cached_views<<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,output,total,capacity);
    return cudaGetLastError();
}

int tq_stage_ram_selected_step(const RamBlockView* sources,int capacity,
        const SelectedBlock* selected,int queries,int count,std::uint32_t* first,
        TqBlockView* output,void* arena,std::size_t arena_bytes,std::size_t stride,
        const RamStageCache& cache,RamStageStep step,std::uint64_t* copied,std::uint64_t* hits,cudaStream_t stream) {
    if(queries<0 || count<0 || capacity<0 || (step!=RamStageStep::Available && step!=RamStageStep::Hydrated))return cudaErrorInvalidValue;
    if(!queries || !count)return cudaSuccess;
    if(!sources || !selected || !first || !output || !arena || !capacity || !stride ||
            (stride&15u) || (reinterpret_cast<std::uintptr_t>(arena)&15u) || queries>INT_MAX/count ||
            (cache.sets && !valid_cache(cache)))return cudaErrorInvalidValue;
    const int total=queries*count;
    if(stride>std::numeric_limits<std::size_t>::max()/std::size_t(total) || arena_bytes<std::size_t(total)*stride)return cudaErrorInvalidValue;
    const bool resume=step==RamStageStep::Hydrated;
    int ctas=0;auto error=staging_ctas(cache.sets?cache.sets:total,&ctas);
    if(error!=cudaSuccess)return error;
    if(!resume) {
        initialize_staging<<<unsigned((std::int64_t(std::max(capacity,cache.sets))+255)/256),256,0,stream>>>(first,capacity,cache,copied,hits);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
    } else if(cache.sets) {
        error=cudaMemsetAsync(cache.locks,0,std::size_t(cache.sets)*sizeof(*cache.locks),stream);
        if(error!=cudaSuccess)return error;
    }
    if(cache.sets) {
        if(resume)bucket_cache_sets<true><<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,output,total,capacity,cache);
        else bucket_cache_sets<false><<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,output,total,capacity,cache);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
        if(resume)stage_cached<false><<<ctas,512,0,stream>>>(sources,selected,first,output,static_cast<std::uint8_t*>(arena),stride,total,capacity,cache,copied,hits);
        else stage_cached<true><<<ctas,512,0,stream>>>(sources,selected,first,output,static_cast<std::uint8_t*>(arena),stride,total,capacity,cache,copied,hits);
        error=cudaGetLastError();if(error!=cudaSuccess)return error;
        resolve_cached_views<<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,output,total,capacity);
    } else {
        if(!resume) {
            mark_first_use<<<unsigned((std::int64_t(total)+255)/256),256,0,stream>>>(selected,first,total,capacity);
            error=cudaGetLastError();if(error!=cudaSuccess)return error;
        }
        if(resume)stage_selected<2><<<ctas,128,0,stream>>>(sources,selected,first,output,static_cast<std::uint8_t*>(arena),stride,total,capacity,copied);
        else stage_selected<1><<<ctas,128,0,stream>>>(sources,selected,first,output,static_cast<std::uint8_t*>(arena),stride,total,capacity,copied);
    }
    return cudaGetLastError();
}
} // namespace tyloinfer::qsa
