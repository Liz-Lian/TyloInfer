#include "qsa_ram_cache.h"
#include "kernels/qsa_ram.h"
#include "mfq_cuda_context.h"
#include "decode_window.h"
#include "tyloinfer/host_memory.h"
#include "tyloinfer/host_parallel.h"
#include "tyloinfer/qsa_tq_cpu.h"
#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <tuple>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace tq=tyloinfer::qsa;
namespace {
using RamReclaimKey=std::pair<int,cudaStream_t>;
struct RamReclaimRegistration;
struct RamReclaimRegistry {
    std::mutex mutex;
    std::map<RamReclaimKey,std::vector<std::weak_ptr<RamReclaimRegistration>>> entries;
};
std::shared_ptr<RamReclaimRegistry> ram_reclaim_registry() {
    static auto registry=std::make_shared<RamReclaimRegistry>();return registry;
}
struct RamReclaimRegistration {
    RamReclaimKey key;
    QsaRamReclaimer reclaim;
    std::weak_ptr<RamReclaimRegistry> registry;
    ~RamReclaimRegistration() {
        if(auto owner=registry.lock()) {
            std::lock_guard<std::mutex> lock(owner->mutex);
            const auto it=owner->entries.find(key);
            if(it==owner->entries.end())return;
            auto& values=it->second;
            values.erase(std::remove_if(values.begin(),values.end(),
                [](const auto& value){return value.expired();}),values.end());
            if(values.empty())owner->entries.erase(it);
        }
    }
};
void check(int code){TYLOINFER_CUDA_CHECK(cudaError_t(code));}
std::size_t setting(const char* name,std::size_t fallback=0) {
    const auto* value=std::getenv(name);if(!value || !*value)return fallback;
    std::size_t used=0;const auto result=std::stoull(value,&used);
    if(value[0]=='-' || used!=std::strlen(value))throw std::invalid_argument(std::string("Invalid ")+name);
    return std::size_t(result);
}
std::size_t tq4_bytes(int rows){return ((2*std::size_t(rows)+(rows+1)/2+(3*rows+7)/8+3)&~std::size_t(3))+std::size_t(rows)*128;}
tq::MicroblockImage copy(const tq::RamKvImage& image) {
    tq::MicroblockImage result;result.key.assign(image.key,image.key+image.key_bytes);
    result.value.assign(image.value,image.value+image.value_bytes);
    result.key_bits=image.key_bits;result.value_bits=image.value_bits;
    result.key_state_bits=image.key_state_bits;result.value_state_bits=image.value_state_bits;
    return result;
}
struct RamAccount {std::atomic<std::size_t> physical{0},logical{0};};
struct RamForegroundCodec {
    std::atomic<bool>& flag;bool previous;
    explicit RamForegroundCodec(std::atomic<bool>& value):flag(value),previous(value.exchange(true)){}
    ~RamForegroundCodec(){flag.store(previous);}
};
struct RamPinnedPool;
void trim_pinned_pool(const std::shared_ptr<RamPinnedPool>& pool);
struct RamClaims {
    std::vector<std::uint8_t>& marks;
    std::vector<std::size_t> ids;
    bool active=true;
    RamClaims(std::vector<std::uint8_t>& storage,std::size_t capacity,std::size_t count):marks(storage) {
        if(marks.size()<capacity)marks.resize(capacity);ids.reserve(count);
    }
    void add(std::size_t id) {
        if(!marks.at(id)){ids.push_back(id);marks[id]=1;}
    }
    void clear()noexcept {
        if(active){for(auto id:ids)marks[id]=0;active=false;}
    }
    ~RamClaims(){clear();}
};
struct RamRetirement {
    cudaEvent_t done=nullptr;
    std::vector<tq::RamKvChange> owners;
    tb::Tensor descriptors,ids,host_descriptors,host_ids;
};
struct RamWorker {
    int device;cudaStream_t stream;
    std::mutex mutex;std::condition_variable changed;
    std::deque<std::pair<std::uint64_t,std::packaged_task<void()>>> jobs;
    std::vector<std::weak_ptr<RamAccount>> accounts;
    std::shared_ptr<RamAccount> pool_account=std::make_shared<RamAccount>();
    std::shared_ptr<RamPinnedPool> pinned_pool;
    mfq::HostParallelPool cpu_codecs;
    // Cache calls are serialized on their owning model thread. Collect all
    // layers' finished publications instead of retaining a previous layer's
    // obsolete pinned images until that layer runs again.
    std::mutex retired_mutex;
    std::vector<RamRetirement> retired;
    std::vector<RamRetirement> publications;
    // Model-thread scratch shared by its layers. Clear only selected IDs,
    // keeping union construction independent of the total context length.
    std::vector<std::uint8_t> claims;
    std::uint64_t next_job=0;
    std::atomic<std::uint64_t> running_job{0},urgent_through{0};
    bool stop=false;std::thread worker;
    explicit RamWorker(int device,cudaStream_t stream):device(device),stream(stream),
        cpu_codecs([device](std::size_t){check(cudaSetDevice(device));}),worker([this]{
        for(;;) {
            std::pair<std::uint64_t,std::packaged_task<void()>> task;
            {std::unique_lock<std::mutex> lock(mutex);changed.wait(lock,[&]{return stop || !jobs.empty();});
                if(jobs.empty())return;task=std::move(jobs.front());jobs.pop_front();}
            running_job.store(task.first);task.second();running_job.store(0);
        }
    }){}
    ~RamWorker(){
        {std::lock_guard<std::mutex> lock(mutex);stop=true;changed.notify_all();}worker.join();
        for(auto& p:retired)if(p.done){(void)cudaEventSynchronize(p.done);(void)cudaEventDestroy(p.done);}
    }
    std::size_t collect() {
        std::vector<RamRetirement> ready;
        {
            std::lock_guard<std::mutex> lock(retired_mutex);
            for(auto it=retired.begin();it!=retired.end();) {
                const auto status=cudaEventQuery(it->done);
                if(status==cudaSuccess) {
                    ready.push_back(std::move(*it));it=retired.erase(it);
                }else {if(status!=cudaErrorNotReady)check(status);++it;}
            }
        }
        // Releasing a slab can enter an allocator. Keep this outside the
        // retirement lock so model publication and codec work can proceed.
        const auto completed=ready.size();
        for(auto& p:ready) {
            check(cudaEventDestroy(p.done));p.done=nullptr;
            if(p.host_ids.defined()) {
                p.owners.clear();publications.push_back(std::move(p));
            }
        }
        ready.clear();trim_pinned_pool(pinned_pool);
        return completed;
    }
    RamRetirement publication(std::size_t count,const tb::TensorOptions& options) {
        auto best=publications.end();
        for(auto it=publications.begin();it!=publications.end();++it)
            if(std::size_t(it->host_ids.numel())>=count &&
                    (best==publications.end() || it->host_ids.numel()<best->host_ids.numel()))best=it;
        if(best!=publications.end()) {
            auto result=std::move(*best);*best=std::move(publications.back());publications.pop_back();
            return result;
        }
        std::size_t capacity=1;while(capacity<count)capacity*=2;
        RamRetirement result;
        result.ids=tb::empty({int64_t(capacity)},options.dtype(tb::kInt32));
        result.descriptors=tb::empty({int64_t(capacity*sizeof(tq::RamBlockView))},options.dtype(tb::kUInt8));
        result.host_ids=tb::empty({int64_t(capacity)},options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
        result.host_descriptors=tb::empty({int64_t(capacity*sizeof(tq::RamBlockView))},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        pool_account->physical.fetch_add(result.host_ids.nbytes()+result.host_descriptors.nbytes());
        return result;
    }
    std::size_t release_readers() {
        std::vector<std::vector<tq::RamKvChange>> ready;
        {
            std::lock_guard<std::mutex> lock(retired_mutex);
            for(auto& p:retired) {
                if(p.owners.empty())continue;
                const auto status=cudaEventQuery(p.done);
                if(status==cudaSuccess)ready.push_back(std::move(p.owners));
                else if(status!=cudaErrorNotReady)check(status);
            }
        }
        // Background work may return image slots, but must not cudaFreeHost
        // transfer descriptors or release model-stream GPU allocations. Such
        // frees can wait for a host-served graph that the model is submitting.
        // Keep the empty retirement records for the model thread's collect().
        return ready.size();
    }
    void retire(RamRetirement pending) {
        std::lock_guard<std::mutex> lock(retired_mutex);
        retired.push_back(std::move(pending));
    }
    std::future<void> submit(std::function<void()> work,std::uint64_t& sequence) {
        std::packaged_task<void()> task(std::move(work));auto result=task.get_future();
        {std::lock_guard<std::mutex> lock(mutex);
            if(next_job==std::numeric_limits<std::uint64_t>::max())throw std::overflow_error("QSA RAM worker sequence exhausted");
            sequence=++next_job;jobs.emplace_back(sequence,std::move(task));changed.notify_one();}return result;
    }
    void prioritize(std::uint64_t through) {
        auto previous=urgent_through.load();
        while(previous<through && !urgent_through.compare_exchange_weak(previous,through)){}
    }
    std::size_t budget() {
        std::size_t physical=pool_account->physical.load(),logical=0,count=0;
        {std::lock_guard<std::mutex> lock(mutex);
            for(auto it=accounts.begin();it!=accounts.end();)if(auto a=it->lock()) {
                logical+=a->logical.load();++count;++it;
            }else it=accounts.erase(it);}
        auto free=mfq::sample_host_memory().available_host_bytes();
        const auto shortage=tq::ram_kv_reclaim_bytes(free,physical,logical);
        // Called on the model thread, outside store/pool locks and graph
        // capture. Release disposable prefix RAM before quantizing live KV.
        if(shortage && reclaim_qsa_ram(device,stream,shortage)) {
            free=mfq::sample_host_memory().available_host_bytes();
            physical=pool_account->physical.load();
        }
        const auto total=tq::ram_kv_budget(free,physical);
        const auto overhead=physical>logical?physical-logical:0;
        return count && total>overhead?(total-overhead)/count:0;
    }
};
// One background encoder per serialized model execution stream. Different
// streams never share staging memory or contend through a single codec job.
std::shared_ptr<RamWorker> worker_for(int device,cudaStream_t stream) {
    static thread_local std::map<std::pair<int,cudaStream_t>,std::weak_ptr<RamWorker>> workers;
    auto& weak=workers[{device,stream}];auto worker=weak.lock();
    if(!worker){worker=std::make_shared<RamWorker>(device,stream);weak=worker;}return worker;
}
struct RamSlab {
    tb::Tensor host;
    std::uint8_t* mapped=nullptr;
    std::size_t stride=0;
    std::vector<unsigned> free;
    unsigned occupied=0;
};
struct RamPinnedPool:std::enable_shared_from_this<RamPinnedPool> {
    std::mutex mutex;
    // Only slabs with free slots are indexed. Full slabs remain owned by
    // their images, so allocation does not scan the entire context.
    std::map<std::size_t,std::vector<std::shared_ptr<RamSlab>>> available;
    std::map<std::size_t,std::size_t> cached_empty_slots;
    std::map<std::size_t,std::size_t> total_slabs;
    tb::TensorOptions options;std::shared_ptr<RamAccount> account;
    struct Allocation {
        std::shared_ptr<RamPinnedPool> pool;std::shared_ptr<RamSlab> slab;unsigned slot=0;
        ~Allocation() {
            if(!slab)return;
            std::lock_guard<std::mutex> lock(pool->mutex);
            auto& list=pool->available[slab->stride];
            if(slab->free.empty())list.push_back(slab);
            slab->free.push_back(slot);--slab->occupied;
            if(!slab->occupied) {
                auto& cached=pool->cached_empty_slots[slab->stride];
                // Slot returns also run on codec threads. Leave physical
                // pinned-page release to trim() on the model thread.
                cached+=slab->free.size();
            }
            // Drop this allocation's last slab reference before making an
            // empty slab visible to the model thread's physical trimming.
            slab.reset();
        }
    };
    tq::RamKvImagePtr allocate(const tq::MicroblockImage& image) {
        const auto key_extent=(image.key.size()+15)&~std::size_t(15);
        const auto stride=(key_extent+image.value.size()+15)&~std::size_t(15);
        auto owner=std::make_shared<Allocation>();owner->pool=shared_from_this();
        auto result=std::make_shared<tq::RamKvImage>();result->owner=owner;
        std::shared_ptr<RamSlab> slab;unsigned slot=0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto& list=available[stride];
            cached_empty_slots.try_emplace(stride,0);
            if(!list.empty()) {
                slab=list.back();
                if(!slab->occupied)cached_empty_slots[stride]-=slab->free.size();
            }
            if(!slab) {
                // Slot returns run in a shared_ptr destructor. Reserve its
                // free-slab index here so returning a full slab cannot allocate.
                auto& total=total_slabs[stride];
                if(list.capacity()<=total)list.reserve(std::max(total+1,list.capacity()*2));
                slab=std::make_shared<RamSlab>();slab->stride=stride;
                const auto slots=std::max<std::size_t>(1,std::min<std::size_t>(64,(1ULL<<20)/stride));
                slab->host=tb::empty({int64_t(slots*stride)},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
                void* mapped=nullptr;check(cudaHostGetDevicePointer(&mapped,slab->host.data_ptr(),0));slab->mapped=static_cast<std::uint8_t*>(mapped);
                slab->free.reserve(slots);for(unsigned i=0;i<slots;++i)slab->free.push_back(unsigned(slots-1-i));
                list.push_back(slab);++total;account->physical.fetch_add(slots*stride);
            }
            slot=slab->free.back();slab->free.pop_back();++slab->occupied;
            if(slab->free.empty())list.pop_back();
        }
        owner->slab=slab;owner->slot=slot;
        auto* key=slab->host.data_ptr<std::uint8_t>()+slot*stride;
        std::memcpy(key,image.key.data(),image.key.size());std::memcpy(key+key_extent,image.value.data(),image.value.size());
        result->key=key;result->value=key+key_extent;result->key_bytes=image.key.size();result->value_bytes=image.value.size();
        result->key_bits=image.key_bits;result->value_bits=image.value_bits;
        result->key_state_bits=image.key_state_bits;result->value_state_bits=image.value_state_bits;
        return result;
    }
    void trim() {
        std::vector<std::shared_ptr<RamSlab>> unused;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for(auto& [stride,cached]:cached_empty_slots) {
                if(cached<=2048/4)continue;
                auto& list=available.at(stride);
                for(auto it=list.begin();it!=list.end() && cached>2048/4;) {
                    const auto& slab=*it;
                    if(slab->occupied){++it;continue;}
                    auto& total=total_slabs.at(stride);
                    unused.push_back(slab);
                    cached-=slab->free.size();--total;
                    account->physical.fetch_sub(std::size_t(slab->host.numel()));
                    it=list.erase(it);
                }
            }
        }
    }
};
void trim_pinned_pool(const std::shared_ptr<RamPinnedPool>& pool) {
    if(!pool)return;
    cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
    if(capture!=cudaStreamCaptureStatusNone)return;
    const auto status=cudaStreamQuery(mfq_current_cuda_stream());
    if(status==cudaSuccess)pool->trim();
    else if(status!=cudaErrorNotReady)check(status);
}
struct StageArena {
    StageArena()=default;
    StageArena(const StageArena&)=delete;
    StageArena& operator=(const StageArena&)=delete;
    tb::Tensor payload,first,views,missing,host_missing,host_selected;
    std::shared_ptr<RamAccount> pinned_account;
    std::size_t stride=0,pinned_bytes=0;
    ~StageArena(){if(pinned_bytes)pinned_account->physical.fetch_sub(pinned_bytes);}
};
std::shared_ptr<StageArena> arena_for(int device,cudaStream_t stream,int capacity,int count,std::size_t stride,const tb::TensorOptions& options,
        const std::shared_ptr<RamAccount>& pinned_account) {
    using Key=std::tuple<int,cudaStream_t,int,int,std::size_t>;
    static thread_local std::map<Key,std::weak_ptr<StageArena>> arenas;
    auto& weak=arenas[{device,stream,capacity,count,stride}];auto arena=weak.lock();
    if(!arena) {
        arena=std::make_shared<StageArena>();arena->stride=stride;arena->pinned_account=pinned_account;
        arena->payload=tb::empty({int64_t(8ULL*count*stride)},options.dtype(tb::kUInt8));
        arena->first=tb::empty({capacity},options.dtype(tb::kInt32));
        arena->missing=tb::empty({1+int64_t(8)*count},options.dtype(tb::kInt32));
        arena->host_missing=tb::empty({1+int64_t(8)*count},options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
        arena->host_selected=tb::empty({int64_t(8)*count*int64_t(sizeof(tq::SelectedBlock))},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        arena->views=tb::empty({int64_t(8ULL*count*sizeof(tq::TqBlockView))},options.dtype(tb::kUInt8));weak=arena;
        arena->pinned_bytes=arena->host_missing.nbytes()+arena->host_selected.nbytes();
        pinned_account->physical.fetch_add(arena->pinned_bytes);
    }
    return arena;
}
}
std::shared_ptr<void> register_qsa_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim) {
    if(!reclaim)return {};
    const auto registry=ram_reclaim_registry();
    auto registration=std::make_shared<RamReclaimRegistration>();
    registration->key={device,stream};registration->reclaim=std::move(reclaim);registration->registry=registry;
    {std::lock_guard<std::mutex> lock(registry->mutex);registry->entries[registration->key].push_back(registration);}
    return registration;
}
std::size_t reclaim_qsa_ram(int device,cudaStream_t stream,std::size_t bytes) {
    if(!bytes)return 0;
    const auto registry=ram_reclaim_registry();
    std::vector<std::shared_ptr<RamReclaimRegistration>> callbacks;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        const auto it=registry->entries.find({device,stream});
        if(it==registry->entries.end())return 0;
        // Reserve before locking weak entries: an allocation failure must
        // not destroy their last owner while the registry mutex is held.
        callbacks.reserve(it->second.size());
        for(const auto& weak:it->second)if(auto owner=weak.lock())callbacks.push_back(std::move(owner));
    }
    std::size_t released=0;
    for(const auto& callback:callbacks) {
        const auto n=callback->reclaim(bytes-released);
        released+=std::min(n,bytes-released);
        if(released==bytes)break;
    }
    return released;
}
struct QsaRamCache::Impl {
    int capacity,rows,count,bits,device=0;std::size_t fp16_bytes,affine_bytes,cache_stride;
    bool ssd_enabled=false;
    bool reported_ssd_fetch=false;
    std::size_t sampled_budget=0;
    bool budget_sampled=false;
    std::atomic<bool> foreground_codec{false};
    tb::TensorOptions options;
    Encode encode;
    std::shared_ptr<RamWorker> worker;
    std::shared_ptr<RamAccount> account=std::make_shared<RamAccount>();
    std::shared_ptr<RamPinnedPool> pool;
    std::unique_ptr<tq::RamKvStore> store;
    std::future<void> maintenance;
    std::uint64_t maintenance_sequence=0;
    tb::Tensor sources,live,cache_slots,cache_locks,cache_frame,cache_payload;
    tq::RamStageCache gpu_cache;
    std::shared_ptr<StageArena> arena;
    std::vector<tq::RamBlockView> host_sources;
    std::mutex queue_mutex;std::vector<tq::RamKvChange> queued;
    void queue(const std::vector<tq::RamKvChange>& changes) {
        std::lock_guard<std::mutex> lock(queue_mutex);const auto needed=queued.size()+changes.size();
        if(needed>queued.capacity())queued.reserve(std::max(needed,queued.capacity()+queued.capacity()/2));
        queued.insert(queued.end(),changes.begin(),changes.end());
    }
    using Pending=RamRetirement;
    Impl(int c,int r,int n,int b,const tb::TensorOptions& o,const tb::Tensor& logical,bool ssd,const std::string& directory,Encode e)
        :capacity(c),rows(r),count(n),bits(b),fp16_bytes(std::size_t(rows)*512),affine_bytes(((3*rows+3)&~3)+std::size_t(rows)*256),options(o),encode(std::move(e)),live(logical),host_sources(c) {
        if(c<=0 || r<=0 || n<=0 || (b!=16 && b!=8 && b!=4))throw std::invalid_argument("Invalid QSA RAM cache geometry");
        ssd_enabled=ssd;
        check(cudaGetDevice(&device));worker=worker_for(device,mfq_current_cuda_stream());
        if(!worker->pinned_pool) {
            worker->pinned_pool=std::make_shared<RamPinnedPool>();worker->pinned_pool->options=options;
            worker->pinned_pool->account=worker->pool_account;
        }
        pool=worker->pinned_pool;
        {std::lock_guard<std::mutex> lock(worker->mutex);worker->accounts.push_back(account);}
        tq::RamKvStoreConfig config;config.pressure.canonical_tier=b==16?tq::RamKvTier::Fp16:b==8?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
        config.pressure.ssd_enabled=ssd;
        if(ssd) {
            mfq::cache::QsaKvStoreConfig disk;disk.directory=directory;disk.buffer_bytes=disk.budget_bytes=1ULL<<20;
            config.session_ssd=std::make_shared<mfq::cache::QsaKvStore>(disk);
        }
        store=std::make_unique<tq::RamKvStore>(config,[this](const auto& image){return pool->allocate(image);},
            [this](const auto& images,auto tier){return convert(images,tier);},
            [this](const auto& changes){queue(changes);});
        sources=tb::zeros({capacity,int64_t(sizeof(tq::RamBlockView))},options.dtype(tb::kUInt8));
        const auto tq4=tq4_bytes(rows);
        cache_stride=((b==16?2*fp16_bytes:b==8?2*affine_bytes:2*tq4)+31)&~std::size_t(15);
        const auto cache_bytes=setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES");
        const auto sets=std::min<std::size_t>(INT_MAX/tq::ram_cache_ways,cache_bytes/cache_stride/tq::ram_cache_ways);
        if(sets) {
            cache_slots=tb::empty({int64_t(sets*tq::ram_cache_ways*sizeof(tq::RamCacheSlot))},options.dtype(tb::kUInt8));
            cache_locks=tb::empty({int64_t(sets)},options.dtype(tb::kInt32));cache_frame=tb::empty({8},options.dtype(tb::kUInt8));
            cache_payload=tb::empty({int64_t(sets*tq::ram_cache_ways*cache_stride)},options.dtype(tb::kUInt8));
            gpu_cache={cache_slots.data_ptr<tq::RamCacheSlot>(),cache_locks.data_ptr<std::uint32_t>(),cache_frame.data_ptr<std::uint64_t>(),
                cache_payload.data_ptr<std::uint8_t>(),std::size_t(cache_payload.numel()),cache_stride,int(sets)};
            check(tq::tq_reset_ram_stage_cache(gpu_cache,mfq_current_cuda_stream()));
        }
    }
    ~Impl() {
        try {wait();}catch(...){}
        (void)cudaStreamSynchronize(mfq_current_cuda_stream());
        try {worker->collect();}catch(...){}
    }
    std::vector<tq::RamKvImagePtr> convert(const std::vector<tq::RamKvImagePtr>& input,tq::RamKvTier tier) {
        check(cudaSetDevice(device));std::vector<tq::RamKvImagePtr> result(input.size());
        if(tier==tq::RamKvTier::Affine8) {
            const auto running=worker->running_job.load();
            const bool urgent=foreground_codec.load() || (running && running<=worker->urgent_through.load());
            const auto cpus=std::max(1u,std::thread::hardware_concurrency());
            const int threads=int(std::min<std::size_t>(input.size(),urgent?cpus:std::max(1u,cpus/2)));
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                static thread_local unsigned reported=0;const unsigned kind=urgent?1u:2u;
                if(!(reported&kind)){reported|=kind;
                    std::fprintf(stderr,"native_qsa_ram_affine actual_match=1 threads=%d urgent=%d shared_pinned_pool=1\n",threads,int(urgent));}
            }
            worker->cpu_codecs.run(0,std::int64_t(input.size()),1,threads,[&](auto first,auto last){check(cudaSetDevice(device));for(auto i=first;i<last;++i) {
                tq::CpuTqBlockFitter fitter(copy(*input[i]),rows,1);result[i]=pool->allocate(fitter.image({8,8}));
            }});
        }else {
            std::vector<tq::MicroblockImage> source;source.reserve(input.size());for(const auto& image:input)source.push_back(copy(*image));
            const auto images=encode(source,std::vector<tq::Precision>(input.size(),{4,4}));
            if(images.size()!=input.size())throw std::runtime_error("QSA RAM GPU codec changed block count");
            for(std::size_t i=0;i<images.size();++i)result[i]=pool->allocate(images[i]);
        }
        return result;
    }
    std::size_t budget()const {const auto explicit_bytes=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");return explicit_bytes?explicit_bytes:worker->budget();}
    void wait(){
        if(!maintenance.valid())return;
        if(maintenance.wait_for(std::chrono::seconds(0))!=std::future_status::ready)
            worker->prioritize(maintenance_sequence);
        maintenance.get();
    }
    tq::RamBlockView record(const tq::RamKvChange& change) {
        auto result=host_sources.at(change.id);result.generation=change.generation;
        const auto& image=change.image;
        if(!image){result.image.key=result.image.value=nullptr;return result;}
        const auto allocation=std::static_pointer_cast<const RamPinnedPool::Allocation>(image->owner);
        const auto offset=std::uintptr_t(image->key)-std::uintptr_t(allocation->slab->host.data_ptr());
        result.image={allocation->slab->mapped+offset,allocation->slab->mapped+offset+(image->value-image->key),
            tq::tq_precision_flags(image->key_bits,image->key_state_bits),tq::tq_precision_flags(image->value_bits,image->value_state_bits)};
        result.key_bytes=std::uint32_t(image->key_bytes);result.value_bytes=std::uint32_t(image->value_bytes);return result;
    }
    void drain() {
        if(maintenance.valid() && maintenance.wait_for(std::chrono::seconds(0))==std::future_status::ready)maintenance.get();
        worker->collect();
        Pending p;
        {std::lock_guard<std::mutex> lock(queue_mutex);p.owners.swap(queued);}
        if(p.owners.empty())return;
        // Several precision transitions can replace an ID. Retain only its
        // final descriptor, without allocating a tree node for every block.
        RamClaims latest(worker->claims,capacity,p.owners.size());
        for(const auto& change:p.owners){auto rec=record(change);host_sources[change.id]=rec;latest.add(change.id);}
        latest.clear();const auto n=latest.ids.size();
        auto publication=worker->publication(n,options);publication.owners=std::move(p.owners);p=std::move(publication);
        auto* ids=p.host_ids.data_ptr<std::int32_t>();auto* descriptors=p.host_descriptors.data_ptr<tq::RamBlockView>();
        std::size_t at=0;for(auto id:latest.ids){ids[at]=std::int32_t(id);descriptors[at++]=host_sources[id];}
        check(cudaMemcpyAsync(p.ids.data_ptr(),p.host_ids.data_ptr(),n*sizeof(std::int32_t),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(p.descriptors.data_ptr(),p.host_descriptors.data_ptr(),n*sizeof(tq::RamBlockView),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        check(tq::tq_publish_ram_sources(sources.data_ptr<tq::RamBlockView>(),capacity,p.ids.data_ptr<std::int32_t>(),p.descriptors.data_ptr<tq::RamBlockView>(),int(n),mfq_current_cuda_stream(),live.data_ptr<tq::TqBlockView>()));
        check(cudaEventCreateWithFlags(&p.done,cudaEventDisableTiming));check(cudaEventRecord(p.done,mfq_current_cuda_stream()));worker->retire(std::move(p));
        account->logical.store(store->stats().ram_bytes);
    }
};
QsaRamCache::QsaRamCache(int c,int r,int n,int bits,const tb::TensorOptions& options,const Tensor& live,bool ssd,const std::string& directory,Encode encode)
    :impl_(std::make_unique<Impl>(c,r,n,bits,options,live,ssd,directory,std::move(encode))){}
QsaRamCache::~QsaRamCache()=default;
void QsaRamCache::prepare() {
    auto& s=*impl_;if(!s.arena)s.arena=arena_for(s.device,mfq_current_cuda_stream(),s.capacity,s.count,2*s.fp16_bytes,s.options,s.worker->pool_account);
    s.drain();
}
void QsaRamCache::append(std::vector<tq::MicroblockImage> images,std::uint64_t epoch,bool fresh) {
    auto& s=*impl_;
    if(s.bits==4 && !fresh) {
        RamForegroundCodec foreground(s.foreground_codec);
        images=s.encode(images,std::vector<tq::Precision>(images.size(),{4,4}));
    }
    const auto tq4=tq4_bytes(s.rows);
    const auto tier=s.bits==16?tq::RamKvTier::Fp16:s.bits==8 || fresh?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
    for(const auto& image:images)s.store->append_at_tier(s.pool->allocate(image),{2*s.fp16_bytes,2*s.affine_bytes,2*tq4},tier,epoch);
    s.account->logical.store(s.store->stats().ram_bytes);s.drain();
    const auto usage=s.store->stats();
    if(!s.budget_sampled || usage.ram_bytes>s.sampled_budget || (s.bits==4 && usage.tokens[1]>=2048))
        observe({},epoch);
}
void QsaRamCache::drain(){impl_->drain();}
QsaRamCache::Tensor QsaRamCache::stage(const Tensor& selected,int queries,int count,const Tensor& live,
        const Tensor& position,std::int64_t host_position,int appended) {
    auto& s=*impl_;if(queries<1 || queries>8 || count!=s.count)throw std::invalid_argument("QSA RAM staging requires its prepared query window");
    if(!s.arena)prepare();
    std::vector<tq::RamKvChange> leased,temporary;
    struct CopyGuard {
        cudaStream_t stream;bool pending=false;
        // SSD errors can unwind after the first copy was submitted. Its RAM
        // leases must outlive that finite copy even when no attention runs.
        ~CopyGuard(){if(pending)(void)cudaStreamSynchronize(stream);}
    } copy_guard{mfq_current_cuda_stream()};
    bool split_copy=false;
    const auto publish_append=[&] {
        if(appended>0)check(tq::tq_publish_ram_append(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,live.data_ptr<tq::TqBlockView>(),
            position.defined()?position.data_ptr<int64_t>():nullptr,host_position,appended,unsigned(s.fp16_bytes),unsigned(s.affine_bytes),mfq_current_cuda_stream()));
    };
    const auto stage_step=[&](tq::RamStageStep step) {
        auto& a=*s.arena;
        check(tq::tq_stage_ram_selected_step(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
            a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,step,nullptr,nullptr,mfq_current_cuda_stream()));
    };
    if(s.store->stats().tokens[3]) {
        cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&status));
        if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("SSD KV fetch requires an eager graph boundary");
        // Finish this layer's outstanding replacements before collecting its
        // actual SSD misses. The new selected-hit protection keeps packed
        // hits readable even while incoming misses exceed cache capacity.
        s.wait();s.drain();
        auto& a=*s.arena;
        check(tq::tq_collect_ram_missing(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
            selected.data_ptr<tq::SelectedBlock>(),queries,count,a.first.data_ptr<std::uint32_t>(),
            a.missing.data_ptr<std::int32_t>()+1,a.missing.data_ptr<int>(),s.gpu_cache,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(a.host_missing.data_ptr(),a.missing.data_ptr(),std::size_t(1+queries*count)*sizeof(int),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(a.host_selected.data_ptr(),selected.data_ptr(),std::size_t(queries)*count*sizeof(tq::SelectedBlock),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        const auto* blocks=a.host_selected.data_ptr<tq::SelectedBlock>();
        // The copy above also retires the descriptor upload. Its publication
        // owner is no longer a reader lease and must not prevent admission.
        s.drain();
        const auto stored=s.store->stats().blocks;
        const auto* missing=a.host_missing.data_ptr<std::int32_t>();
        if(missing[0]<0 || missing[0]>queries*count)throw std::runtime_error("Invalid QSA RAM miss count");
        for(int at=0;at<missing[0];++at)if(std::size_t(missing[at+1])<stored){split_copy=true;break;}
        if(split_copy) {
            // This layer's producer was joined above. Its store retains the
            // immutable sources while the selected leases are assembled;
            // admission cannot start until all those leases are acquired.
            publish_append();copy_guard.pending=true;stage_step(tq::RamStageStep::Available);
            check_stream_progress(mfq_current_cuda_stream());
        }
        RamClaims ids(s.worker->claims,s.capacity,std::size_t(queries)*count);
        for(int i=0;i<queries*count;++i)if(blocks[i].id>=0 && std::size_t(blocks[i].id)<stored && blocks[i].token_mask)ids.add(std::size_t(blocks[i].id));
        ids.clear();
        // Pin all current RAM selections before admitting any SSD miss, so
        // admission cannot evict another block used by this same window.
        leased=s.store->lease(ids.ids,std::uint64_t(host_position));
        const auto admission_budget=s.budget();
        RamForegroundCodec foreground(s.foreground_codec);
        for(int at=0;at<missing[0];++at) {
            const auto id=std::size_t(missing[at+1]);if(id>=stored || s.store->resident(id))continue;
            auto image=s.store->read(id,admission_budget);
            tq::RamKvChange change{id,s.store->generation(id),image,{},tq::RamKvTier::Tq4};leased.push_back(change);
            if(!s.store->resident(id))temporary.push_back(change);
        }
        if(missing[0] && !s.reported_ssd_fetch && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            s.reported_ssd_fetch=true;const auto stats=s.store->stats();
            std::fprintf(stderr,"qsa_ram_ssd_fetch actual_match=1 misses=%d read_bytes=%llu promotions=%llu packed_hits_skip_ssd=1\n",
                missing[0],static_cast<unsigned long long>(stats.ssd_read_bytes),static_cast<unsigned long long>(stats.promoted_blocks));
        }
        if(!temporary.empty()) {
            s.queue(temporary);
        }
        s.drain();
        if(split_copy && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
            if(!(reported&mask)){reported|=mask;std::fprintf(stderr,
                "native_qsa_ram_ssd_overlap actual_match=1 queries=%d misses=%d cache_bytes=%zu resident_copy_before_read=1 copy_before_lease=1 pinned_selection=1\n",queries,missing[0],s.gpu_cache.payload_bytes);}
        }
    }
    // New GPU appends are not immutable RAM records yet. They bypass the
    // persistent LRU until their archive has completed and been published.
    if(!split_copy)publish_append();
    auto& a=*s.arena;
    if(split_copy)stage_step(tq::RamStageStep::Hydrated);
    else if(s.gpu_cache.sets)check(tq::tq_stage_ram_selected_cached(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
        a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,nullptr,nullptr,mfq_current_cuda_stream()));
    else check(tq::tq_stage_ram_selected(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
        a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,nullptr,mfq_current_cuda_stream()));
    if(!temporary.empty()) {
        for(auto& change:temporary){change.previous=std::move(change.image);change.tier=tq::RamKvTier::Ssd;}
        s.queue(temporary);
        s.drain();
    }
    if(!leased.empty()) {
        Impl::Pending pending;pending.owners=std::move(leased);
        check(cudaEventCreateWithFlags(&pending.done,cudaEventDisableTiming));check(cudaEventRecord(pending.done,mfq_current_cuda_stream()));
        s.worker->retire(std::move(pending));
    }
    copy_guard.pending=false;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
        if(!(reported&mask)){reported|=mask;std::fprintf(stderr,"native_qsa_ram_stage actual_match=1 queries=%d bits=%d union_copy=1 cache_bytes=%zu shared_stream_arena=1\n",queries,s.bits,s.gpu_cache.payload_bytes);}
    }
    return a.views;
}
void QsaRamCache::observe(const std::vector<double>& hits,std::uint64_t epoch) {
    auto& s=*impl_;s.drain();std::vector<std::size_t> touched;
    const auto blocks=s.store->stats().blocks;for(std::size_t id=0;id<std::min(blocks,hits.size());++id)if(hits[id]>0)touched.push_back(id);
    s.store->touch(touched,epoch);
    if(s.maintenance.valid()) {
        if(s.maintenance.wait_for(std::chrono::seconds(0))!=std::future_status::ready)return;
        s.maintenance.get();
    }
    const auto physical_budget=s.budget();s.sampled_budget=physical_budget;s.budget_sampled=true;
    auto target=physical_budget;const auto usage=s.store->stats();
    const bool pressure=usage.ram_bytes>physical_budget;
    if(!pressure && s.bits==4 && usage.tokens[1]>=2048) {
        // Keep a partial batch of fresh Affine8 images off the serial decode
        // path. Fit only complete2048-token batches on the background worker.
        const auto saving=512*(2*s.affine_bytes-2*tq4_bytes(s.rows));
        target=usage.ram_bytes-saving;
    }
    if(usage.ram_bytes<=target)return;
    s.maintenance=s.worker->submit([&s,target,physical_budget,pressure]{
        check(cudaSetDevice(s.device));s.worker->release_readers();
        auto result=s.store->maintain(target,pressure);
        // A completed GPU publication must stop pinning its old images even
        // when the model has moved on to other layers. Retry only after
        // releasing completed leases; long-lived readers remain untouched.
        while(result.needs_readers && s.worker->release_readers())
            result=s.store->maintain(target,pressure);
        // Fresh blocks can arrive during optional format conversion. Only
        // the actual RAM limit permits SSD eviction or a capacity error.
        if(!pressure && s.store->stats().ram_bytes>physical_budget)result=s.store->maintain(physical_budget);
        const auto stats=s.store->stats();s.account->logical.store(stats.ram_bytes);
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            std::fprintf(stderr,"qsa_ram_pressure actual_match=1 budget_bytes=%zu ram_bytes=%zu pooled_pinned_bytes=%zu fp16_tokens=%llu affine8_tokens=%llu tq4_tokens=%llu ssd_tokens=%llu quantized=%llu spilled=%llu\n",
                physical_budget,stats.ram_bytes,s.pool->account->physical.load(),static_cast<unsigned long long>(stats.tokens[0]),static_cast<unsigned long long>(stats.tokens[1]),
                static_cast<unsigned long long>(stats.tokens[2]),static_cast<unsigned long long>(stats.tokens[3]),
                static_cast<unsigned long long>(stats.quantized_blocks),static_cast<unsigned long long>(stats.spilled_blocks));
        }
        if(stats.ram_bytes>physical_budget && !result.needs_readers)throw std::runtime_error(s.ssd_enabled?
            "QSA RAM cannot hold the protected high-precision KV tiers; reduce the context":
            "QSA RAM KV exhausted; enable session SSD KV or reduce the context");
    },s.maintenance_sequence);
}
void QsaRamCache::wait(){
    auto& s=*impl_;s.wait();s.drain();
    if(s.store->stats().ram_bytes>s.budget()) {
        // An early background plan can encounter descriptor-publication
        // leases from this same append. Retire those readers at an explicit
        // wait boundary and retry pressure management before returning.
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));s.drain();
        observe({},0);s.wait();s.drain();
    }
}
void QsaRamCache::truncate(std::size_t blocks){impl_->wait();impl_->store->truncate(blocks);impl_->drain();}
bool QsaRamCache::graph_safe()const{return (!impl_->ssd_enabled || !impl_->maintenance.valid()) && impl_->store->stats().tokens[3]==0;}
bool QsaRamCache::encoding_urgent()const{
    if(impl_->foreground_codec.load())return true;
    const auto& worker=*impl_->worker;const auto job=worker.running_job.load();
    return job && job<=worker.urgent_through.load();
}
std::shared_ptr<tq::RamKvSnapshot> QsaRamCache::snapshot(std::size_t blocks) {
    wait();return impl_->store->snapshot(blocks);
}
void QsaRamCache::restore(const tq::RamKvSnapshot& state,bool borrow_prefix) {
    auto& s=*impl_;s.wait();check(cudaStreamSynchronize(mfq_current_cuda_stream()));s.drain();
    if(s.gpu_cache.sets)check(tq::tq_reset_ram_stage_cache(s.gpu_cache,mfq_current_cuda_stream()));
    RamForegroundCodec foreground(s.foreground_codec);
    const auto written=s.store->stats().ssd_written_bytes;
    s.store->restore(state,s.budget(),borrow_prefix);s.account->logical.store(s.store->stats().ram_bytes);s.drain();
    if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
        const auto stats=s.store->stats();
        std::fprintf(stderr,"qsa_ram_restore actual_match=1 borrow_prefix=%d ram_bytes=%zu session_ssd_written_bytes=%llu\n",
            int(borrow_prefix),stats.ram_bytes,static_cast<unsigned long long>(stats.ssd_written_bytes-written));
    }
}
tq::MicroblockImage QsaRamCache::reconstruction_image(std::size_t id){
    auto& s=*impl_;RamForegroundCodec foreground(s.foreground_codec);return copy(*s.store->read(id,s.budget(),false));
}
tq::AdaptiveCacheStats QsaRamCache::stats()const {
    const auto a=impl_->store->stats();tq::AdaptiveCacheStats result;result.blocks=a.blocks;result.resident_bytes=a.ram_bytes;
    result.blocks_per_tier[5]=std::size_t(a.tokens[0]+a.tokens[1])/4;result.blocks_per_tier[0]=std::size_t(a.tokens[2])/4;
    result.demotions=a.quantized_blocks;result.promotions=a.promoted_blocks;result.canonical_read_bytes=a.ssd_read_bytes;
    return result;
}
} // namespace mfq::cuda
