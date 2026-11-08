#include "qsa_ram_cache.h"
#include "kernels/qsa_ram.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_resident_allocations.h"
#include "decode_window.h"
#include "moe_pipeline.h"
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
    unsigned kind=0;
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
struct RamAccount {
    std::atomic<std::size_t> physical{0},logical{0},target{0},restore_target{0};
    std::mutex owner_mutex;
    QsaRamCache* owner=nullptr;
    bool can_spill=false;
};
struct PrefillTransfer {
    std::shared_ptr<RamAccount> account;
    std::size_t charge=0,packet_bytes=0;
    std::array<tb::Tensor,2> packets,records;
    std::array<cudaEvent_t,2> done{};
    ~PrefillTransfer() {
        for(auto event:done)if(event) {
            (void)cudaEventSynchronize(event);(void)cudaEventDestroy(event);
        }
        if(charge)account->physical.fetch_sub(charge);
    }
};
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
    const tq::RamBlockView* mapped_descriptors=nullptr;
    const std::int32_t* mapped_ids=nullptr;
    std::shared_ptr<void> temporary;
};
struct RamWorker {
    int device;cudaStream_t stream;
    std::mutex mutex;std::condition_variable changed;
    std::deque<std::pair<std::uint64_t,std::packaged_task<void()>>> jobs;
    std::vector<std::weak_ptr<RamAccount>> accounts;
    std::shared_ptr<RamAccount> pool_account=std::make_shared<RamAccount>();
    std::atomic<std::size_t> workspace_bytes{0};
    std::shared_ptr<RamPinnedPool> pinned_pool;
    std::shared_ptr<PrefillTransfer> prefill_transfer;
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
        try{reserve_qsa_ram_budget(device,stream,0);}catch(...){}
        {std::lock_guard<std::mutex> lock(mutex);stop=true;changed.notify_all();}worker.join();
        for(auto& p:retired)if(p.done){(void)cudaEventSynchronize(p.done);(void)cudaEventDestroy(p.done);}
        for(auto& p:publications)if(p.done)(void)cudaEventDestroy(p.done);
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
            if(p.host_ids.defined()) {
                // Its event has completed. Re-record it only after this whole
                // publication is acquired again, together with its buffers.
                p.owners.clear();publications.push_back(std::move(p));
            }else {check(cudaEventDestroy(p.done));p.done=nullptr;}
        }
        ready.clear();trim_pinned_pool(pinned_pool);
        return completed;
    }
    RamRetirement publication(std::size_t count,const tb::TensorOptions& options,bool mapped) {
        auto best=publications.end();
        for(auto it=publications.begin();it!=publications.end();++it)
            if(std::size_t(it->host_ids.numel())>=count &&
                    (best==publications.end() || it->host_ids.numel()<best->host_ids.numel()))best=it;
        if(best!=publications.end()) {
            auto result=std::move(*best);*best=std::move(publications.back());publications.pop_back();
            return result;
        }
        std::size_t capacity=1;while(capacity<count)capacity*=2;
        ResidentAllocationScope resident;
        RamRetirement result;
        result.host_ids=tb::empty({int64_t(capacity)},options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
        result.host_descriptors=tb::empty({int64_t(capacity*sizeof(tq::RamBlockView))},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        if(mapped) {
            void* pointer=nullptr;check(cudaHostGetDevicePointer(&pointer,result.host_ids.data_ptr(),0));
            result.mapped_ids=static_cast<const std::int32_t*>(pointer);
            check(cudaHostGetDevicePointer(&pointer,result.host_descriptors.data_ptr(),0));
            result.mapped_descriptors=static_cast<const tq::RamBlockView*>(pointer);
        }else {
            result.ids=tb::empty({int64_t(capacity)},options.dtype(tb::kInt32));
            result.descriptors=tb::empty({int64_t(capacity*sizeof(tq::RamBlockView))},options.dtype(tb::kUInt8));
        }
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
    std::size_t reclaim_capacity(std::size_t per_cache) {
        const auto physical=pool_account->physical.load();std::size_t logical=0,count=0;
        {std::lock_guard<std::mutex> lock(mutex);
            for(const auto& weak:accounts)if(auto a=weak.lock()){
                const auto bytes=a->logical.load();logical+=bytes;
                count+=bytes || a->target.load() || a->restore_target.load();
            }}
        const auto overhead=physical>logical?physical-logical:0;
        const auto maximum=std::numeric_limits<std::size_t>::max();
        if(!count || per_cache>(maximum-overhead)/count)throw std::overflow_error("QSA RAM protected capacity overflow");
        const auto target=per_cache*count+overhead;
        const auto free=mfq::sample_host_memory().available_host_bytes();
        const auto shortage=tq::ram_kv_reclaim_bytes(free,physical,target);
        return reclaim_qsa_ram_capacity(device,stream,shortage);
    }
    std::size_t last_reservation=0,last_reservation_free=0;
    std::size_t budget() {
        std::size_t physical=pool_account->physical.load(),logical=0,target=0,count=0;
        {std::lock_guard<std::mutex> lock(mutex);
            for(auto it=accounts.begin();it!=accounts.end();)if(auto a=it->lock()) {
                const auto bytes=a->logical.load(),wanted=std::max(a->target.load(),a->restore_target.load());
                logical+=bytes;
                // SSD-backed KV competes for spare RAM, not for expert RAM.
                // Its complete-context target still counts it as an active
                // layer, but may not reserve space by evicting experts.
                if(!a->can_spill)target+=wanted;
                // An unused MTP head or empty session owns no KV share. Its
                // small staging allocation remains in the physical overhead.
                count+=bytes || wanted;++it;
            }else it=accounts.erase(it);}
        auto free=mfq::sample_host_memory().available_host_bytes();
        const auto workspace=workspace_bytes.load();
        const auto kv_free=tq::ram_kv_reclaim_bytes(0,physical,logical);
        const auto required_free=workspace>std::numeric_limits<std::size_t>::max()-kv_free?
            std::numeric_limits<std::size_t>::max():workspace+kv_free;
        const auto shortage=required_free>free?required_free-free:0;
        // Called on the model thread, outside store/pool locks and graph
        // capture. Release disposable prefix RAM before quantizing live KV.
        if(shortage && reclaim_qsa_ram(device,stream,shortage)) {
            free=mfq::sample_host_memory().available_host_bytes();
            physical=pool_account->physical.load();
        }
        const auto overhead=physical>logical?physical-logical:0;
        const auto maximum=std::numeric_limits<std::size_t>::max();
        if(target>maximum-overhead)throw std::overflow_error("QSA shared RAM target overflow");
        if(const auto* cap=std::getenv("TYLOINFER_KV_RAM_LIMIT_BYTES");cap && *cap)target=std::min(target,setting("TYLOINFER_KV_RAM_LIMIT_BYTES"));
        const auto wanted=target?target+overhead:0;
        // Only RAM-only layers require expert reclamation. Existing spillable
        // images may exceed that target and must not inflate its reservation.
        const auto required=wanted?tq::ram_kv_reclaim_bytes(0,std::min(physical,wanted),wanted):0;
        constexpr std::size_t granule=4ULL<<20;
        const auto rounded=required>maximum-(granule-1)?maximum:
            (required+granule-1)/granule*granule;
        if(rounded!=last_reservation || (free<rounded && last_reservation_free>free && last_reservation_free-free>=granule)) {
            reserve_qsa_ram_budget(device,stream,rounded);last_reservation=rounded;
            free=mfq::sample_host_memory().available_host_bytes();
            physical=pool_account->physical.load();last_reservation_free=free;
        }
        // Pending prefill outputs and descriptor growth must remain possible
        // throughout the chunk, after every background KV maintenance pass.
        auto total=tq::ram_kv_budget(free-std::min(free,workspace),physical);
        if(const auto* cap=std::getenv("TYLOINFER_KV_RAM_LIMIT_BYTES");cap && *cap)total=std::min(total,setting("TYLOINFER_KV_RAM_LIMIT_BYTES"));
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
    static std::size_t image_stride(std::size_t key,std::size_t value) {
        constexpr auto maximum=std::numeric_limits<std::size_t>::max();
        if(!key || !value || key>maximum-15)throw std::invalid_argument("Invalid QSA RAM image extent");
        const auto aligned=(key+15)&~std::size_t(15);
        if(value>maximum-aligned || aligned+value>maximum-15)
            throw std::invalid_argument("QSA RAM image extent overflow");
        return (aligned+value+15)&~std::size_t(15);
    }
    static std::size_t image_stride(const tq::MicroblockImage& image) {
        return image_stride(image.key.size(),image.value.size());
    }
    tq::RamKvImagePtr reserve_image(const tq::RamKvImage& image,bool grow,std::size_t* remaining=nullptr) {
        const auto stride=image_stride(image.key_bytes,image.value_bytes);
        const auto key_extent=(image.key_bytes+15)&~std::size_t(15);
        std::shared_ptr<Allocation> owner;std::shared_ptr<tq::RamKvImage> result;
        const auto prepare=[&] {
            owner=std::make_shared<Allocation>();owner->pool=shared_from_this();
            result=std::make_shared<tq::RamKvImage>();result->owner=owner;
        };
        if(grow)prepare();
        std::shared_ptr<RamSlab> slab;unsigned slot=0;
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto found=available.find(stride);
            if(!grow && (found==available.end() || found->second.empty()))return {};
            if(!grow)prepare();
            // A known same-layout batch shares this counter under the pool
            // lock, including parallel CPU codec calls. Allocate only for
            // actual remaining images, not the model's maximum context.
            std::size_t pending=1;
            if(remaining) {
                if(!*remaining)throw std::logic_error("QSA RAM allocation batch exhausted");
                pending=(*remaining)--;
            }
            auto& list=available[stride];
            cached_empty_slots.try_emplace(stride,0);
            if(!list.empty()) {
                slab=list.back();
                if(!slab->occupied)cached_empty_slots[stride]-=slab->free.size();
            }
            if(!slab) {
                if(!grow)return {};
                // Slot returns run in a shared_ptr destructor. Reserve its
                // free-slab index here so returning a full slab cannot allocate.
                auto& total=total_slabs[stride];
                if(list.capacity()<=total)list.reserve(std::max(total+1,list.capacity()*2));
                slab=std::make_shared<RamSlab>();slab->stride=stride;
                const auto ordinary=std::max<std::size_t>(1,(1ULL<<20)/stride);
                const auto slots=std::max(ordinary,std::min<std::size_t>(pending,tq::RamPressureConfig{}.quantize_tokens/4));
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
        result->key=key;result->value=key+key_extent;result->key_bytes=image.key_bytes;result->value_bytes=image.value_bytes;
        result->key_bits=image.key_bits;result->value_bits=image.value_bits;
        result->key_state_bits=image.key_state_bits;result->value_state_bits=image.value_state_bits;
        return result;
    }
    tq::RamKvImagePtr allocate(const tq::MicroblockImage& image,std::size_t* remaining=nullptr) {
        tq::RamKvImage layout;
        layout.key=image.key.data();layout.value=image.value.data();
        layout.key_bytes=image.key.size();layout.value_bytes=image.value.size();
        layout.key_bits=image.key_bits;layout.value_bits=image.value_bits;
        layout.key_state_bits=image.key_state_bits;layout.value_state_bits=image.value_state_bits;
        return allocate_view(layout,remaining);
    }
    tq::RamKvImagePtr allocate_view(const tq::RamKvImage& image,std::size_t* remaining=nullptr) {
        auto result=reserve_image(image,true,remaining);
        std::memcpy(const_cast<std::uint8_t*>(result->key),image.key,image.key_bytes);
        std::memcpy(const_cast<std::uint8_t*>(result->value),image.value,image.value_bytes);
        return result;
    }
    tq::RamKvImagePtr try_reserve(const tq::RamKvSnapshotBlock& source) {
        if(source.image)return source.image;
        if(!source.backing)throw std::runtime_error("QSA SSD source has no retained backing");
        const auto& backing=*source.backing;tq::RamKvImage layout;
        layout.key_bytes=backing.key_bytes;layout.value_bytes=backing.value_bytes;
        layout.key_bits=backing.key_bits;layout.value_bits=backing.value_bits;
        layout.key_state_bits=backing.key_state_bits;layout.value_state_bits=backing.value_state_bits;
        // A graph may wait for this CPU task: consume only existing locked
        // slots, never call a CUDA allocator or trim pinned slabs here.
        return reserve_image(layout,false);
    }
    std::vector<tq::RamKvImagePtr> allocate_batch(const std::vector<tq::MicroblockImage>& images) {
        std::map<std::size_t,std::size_t> remaining;
        for(const auto& image:images)++remaining[image_stride(image)];
        std::vector<tq::RamKvImagePtr> result;result.reserve(images.size());
        for(const auto& image:images)result.push_back(allocate(image,&remaining.at(image_stride(image))));
        return result;
    }
    std::vector<tq::RamKvRelocation> compact(const std::vector<tq::RamKvImagePtr>& images,std::size_t goal) {
        struct Group {std::shared_ptr<RamSlab> slab;std::vector<std::size_t> indices;bool exclusive=true;};
        std::map<RamSlab*,Group> grouped;
        for(std::size_t i=0;i<images.size();++i) {
            const auto owner=std::static_pointer_cast<const Allocation>(images[i]->owner);
            if(owner->pool.get()!=this)continue;
            auto& group=grouped[owner->slab.get()];group.slab=owner->slab;group.indices.push_back(i);
            group.exclusive&=images[i].use_count()==2;
        }
        std::vector<Group*> order;order.reserve(grouped.size());
        for(auto& [key,group]:grouped)if(group.exclusive)order.push_back(&group);
        std::sort(order.begin(),order.end(),[](const auto* a,const auto* b) {
            return a->indices.size()*b->slab->host.numel()<b->indices.size()*a->slab->host.numel();
        });
        struct Excluded {
            RamPinnedPool& pool;std::vector<std::shared_ptr<RamSlab>> slabs;
            ~Excluded(){std::lock_guard<std::mutex> lock(pool.mutex);
                for(auto& slab:slabs)pool.available.at(slab->stride).push_back(std::move(slab));}
        } excluded{*this,{}};
        {
            std::lock_guard<std::mutex> lock(mutex);std::size_t available_slabs=0;
            for(const auto& [stride,list]:available)available_slabs+=list.size();
            excluded.slabs.reserve(available_slabs);
            // Keep compaction within this store. Mixing different layers in
            // destination slabs would prevent later whole-slab evacuation.
            for(auto& [stride,list]:available)for(auto it=list.begin();it!=list.end();) {
                const auto found=grouped.find(it->get());
                const bool owned=found!=grouped.end() && found->second.exclusive &&
                    found->second.indices.size()==(*it)->occupied;
                if((*it)->occupied && !owned) {
                    excluded.slabs.push_back(std::move(*it));*it=std::move(list.back());list.pop_back();
                }else ++it;
            }
        }
        std::vector<tq::RamKvRelocation> result;result.reserve(images.size());
        std::size_t evacuated=0;
        for(const auto* group:order) {
            const auto& slab=group->slab;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if(slab->occupied!=group->indices.size() || slab->free.empty())continue;
                auto& list=available.at(slab->stride);std::size_t slots=0;
                for(const auto& candidate:list)if(candidate!=slab)slots+=candidate->free.size();
                if(slots<slab->occupied)continue;
                const auto found=std::find(list.begin(),list.end(),slab);
                if(found==list.end())continue;
                // Source images remain alive until descriptor publication.
                // Keep every evacuated slab out of subsequent allocations.
                *found=std::move(list.back());list.pop_back();excluded.slabs.push_back(slab);
            }
            const auto first=result.size();
            for(const auto index:group->indices) {
                const auto& old=*images[index];auto replacement=reserve_image(old,false);
                if(!replacement){result.resize(first);break;}
                std::memcpy(const_cast<std::uint8_t*>(replacement->key),old.key,old.key_bytes);
                std::memcpy(const_cast<std::uint8_t*>(replacement->value),old.value,old.value_bytes);
                result.push_back({index,std::move(replacement)});
            }
            if(result.size()-first==group->indices.size())evacuated+=slab->host.nbytes();
            if(evacuated>=goal)break;
        }
        return result;
    }
    void trim(std::size_t retain=2048/4) {
        std::vector<std::shared_ptr<RamSlab>> unused;
        {
            std::lock_guard<std::mutex> lock(mutex);
            for(auto& [stride,cached]:cached_empty_slots) {
                if(cached<=retain)continue;
                auto& list=available.at(stride);
                for(auto it=list.begin();it!=list.end() && cached>retain;) {
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
struct SsdGraphArena {
    tb::Tensor payload,ids,images,backup,saved;
    std::uint8_t* mapped_payload=nullptr;
    std::int32_t* mapped_ids=nullptr;
    tq::RamBlockView* mapped_images=nullptr;
    std::shared_ptr<RamAccount> account;
    std::size_t bytes=0,stride=0;
    ~SsdGraphArena(){if(bytes)account->physical.fetch_sub(bytes);}
};
struct StageArena {
    StageArena()=default;
    StageArena(const StageArena&)=delete;
    StageArena& operator=(const StageArena&)=delete;
    tb::Tensor payload,first,views,missing,host_missing,host_selected;
    std::int32_t* mapped_missing=nullptr;
    tq::SelectedBlock* mapped_selected=nullptr;
    // CPU tasks run in stream order. Preserve their small selection list
    // before releasing the GPU to overwrite the shared mapped export.
    std::vector<tq::SelectedBlock> served_selected;
    std::shared_ptr<SsdGraphArena> ssd;
    std::shared_ptr<RamAccount> pinned_account;
    std::size_t stride=0,host_bytes=0;
    ~StageArena(){if(host_bytes)pinned_account->physical.fetch_sub(host_bytes);}
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
        const auto mapped=[](const tb::Tensor& tensor) {
            void* pointer=nullptr;const auto status=cudaHostGetDevicePointer(&pointer,tensor.data_ptr(),0);
            if(status==cudaSuccess)return pointer;
            if(status!=cudaErrorInvalidValue && status!=cudaErrorNotSupported && status!=cudaErrorMemoryAllocation)check(status);
            (void)cudaGetLastError();return static_cast<void*>(nullptr);
        };
        arena->mapped_missing=static_cast<std::int32_t*>(mapped(arena->host_missing));
        arena->mapped_selected=static_cast<tq::SelectedBlock*>(mapped(arena->host_selected));
        arena->views=tb::empty({int64_t(8ULL*count*sizeof(tq::TqBlockView))},options.dtype(tb::kUInt8));weak=arena;
        arena->served_selected.resize(std::size_t(8)*count);
        arena->host_bytes=arena->host_missing.nbytes()+arena->host_selected.nbytes()+arena->served_selected.capacity()*sizeof(tq::SelectedBlock);
        pinned_account->physical.fetch_add(arena->host_bytes);
    }
    return arena;
}
}
static std::shared_ptr<void> register_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim,unsigned kind) {
    if(!reclaim)return {};
    const auto registry=ram_reclaim_registry();
    auto registration=std::make_shared<RamReclaimRegistration>();
    registration->key={device,stream};registration->reclaim=std::move(reclaim);registration->registry=registry;registration->kind=kind;
    {std::lock_guard<std::mutex> lock(registry->mutex);registry->entries[registration->key].push_back(registration);}
    return registration;
}
std::shared_ptr<void> register_qsa_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim) {
    return register_ram_reclaimer(device,stream,std::move(reclaim),0);
}
std::shared_ptr<void> register_qsa_ram_capacity_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim) {
    return register_ram_reclaimer(device,stream,std::move(reclaim),1);
}
static std::size_t reclaim_ram(int device,cudaStream_t stream,std::size_t bytes,unsigned kind) {
    if(!bytes && kind!=2)return 0;
    const auto registry=ram_reclaim_registry();
    std::vector<std::shared_ptr<RamReclaimRegistration>> callbacks;
    {
        std::lock_guard<std::mutex> lock(registry->mutex);
        const auto it=registry->entries.find({device,stream});
        if(it==registry->entries.end())return 0;
        // Reserve before locking weak entries: an allocation failure must
        // not destroy their last owner while the registry mutex is held.
        callbacks.reserve(it->second.size());
        for(const auto& weak:it->second)if(auto owner=weak.lock();owner && owner->kind==kind)callbacks.push_back(std::move(owner));
    }
    std::size_t released=0;
    for(const auto& callback:callbacks) {
        const auto n=callback->reclaim(kind==2?bytes:bytes-released);
        released+=std::min(n,kind==2?std::numeric_limits<std::size_t>::max()-released:bytes-released);
        if(kind!=2 && released==bytes)break;
    }
    return released;
}
std::size_t reclaim_qsa_ram(int device,cudaStream_t stream,std::size_t bytes) {
    return reclaim_ram(device,stream,bytes,0);
}
std::size_t reclaim_qsa_ram_capacity(int device,cudaStream_t stream,std::size_t bytes) {
    return reclaim_ram(device,stream,bytes,1);
}
std::shared_ptr<void> register_qsa_ram_budget_reserver(int device,cudaStream_t stream,QsaRamReclaimer reserve) {
    return register_ram_reclaimer(device,stream,std::move(reserve),2);
}
std::size_t reserve_qsa_ram_budget(int device,cudaStream_t stream,std::size_t free_bytes) {
    return reclaim_ram(device,stream,free_bytes,2);
}
static std::shared_ptr<void> hold_worker_workspace(std::shared_ptr<RamWorker> owner,std::size_t bytes) {
    if(!bytes)return {};
    struct Reservation {
        std::shared_ptr<RamWorker> worker;std::size_t bytes;
        Reservation(std::shared_ptr<RamWorker> owner,std::size_t size):worker(std::move(owner)),bytes(size) {
            auto previous=worker->workspace_bytes.load();
            do {
                if(bytes>std::numeric_limits<std::size_t>::max()-previous)
                    throw std::overflow_error("QSA host workspace overflow");
            }while(!worker->workspace_bytes.compare_exchange_weak(previous,previous+bytes));
        }
        ~Reservation(){worker->workspace_bytes.fetch_sub(bytes);}
    };
    return std::make_shared<Reservation>(std::move(owner),bytes);
}
void prepare_qsa_ram_workspace(std::size_t bytes,
        const std::function<QsaRamWorkspaceProgress()>& reclaim) {
    auto previous=reclaim();bool retried=false;
    while(mfq::sample_host_memory().available_host_bytes()<bytes) {
        const auto current=reclaim();
        if(mfq::sample_host_memory().available_host_bytes()>=bytes)return;
        const bool progress=current.logical<previous.logical || current.physical<previous.physical;
        if(!progress && retried)
            throw std::runtime_error("Insufficient host memory for prefill workspace: required="+
                std::to_string(bytes)+" available="+
                std::to_string(mfq::sample_host_memory().available_host_bytes())+
                " logical_kv="+std::to_string(current.logical)+" pinned_kv="+std::to_string(current.physical));
        retried=!progress;previous=current;
    }
}
struct QsaRamCache::Impl {
    int capacity,rows,count,bits,device=0;std::size_t fp16_bytes,affine_bytes,cache_stride,spill_stride;
    bool ssd_enabled=false,mapped_capability=false;
    bool reported_ssd_fetch=false;
    std::size_t sampled_budget=0;
    bool budget_sampled=false;
    std::atomic<bool> foreground_codec{false};
    // CPU transfer/flag builders must not change subsequent GPU allocations.
    const tb::TensorOptions options;
    Encode encode;
    std::shared_ptr<RamWorker> worker;
    std::shared_ptr<RamAccount> account=std::make_shared<RamAccount>();
    std::shared_ptr<RamPinnedPool> pool;
    std::unique_ptr<tq::RamKvStore> store;
    std::future<void> maintenance;
    std::uint64_t maintenance_sequence=0;
    std::atomic<std::size_t> required_capacity{0};
    tb::Tensor sources,live,cache_slots,cache_locks,cache_frame,cache_payload;
    tq::RamStageCache gpu_cache;
    tb::Tensor cache_parameters,host_cache_parameters;
    std::uint64_t publication_revision=0,prepared_revision=0;
    std::shared_ptr<StageArena> arena;
    std::uint64_t graph_epoch=0;
    std::vector<tq::RamBlockView> host_sources;
    std::mutex queue_mutex;std::vector<tq::RamKvChange> queued;
    void queue(const std::vector<tq::RamKvChange>& changes) {
        std::lock_guard<std::mutex> lock(queue_mutex);const auto needed=queued.size()+changes.size();
        if(needed>queued.capacity())queued.reserve(std::max(needed,queued.capacity()+queued.capacity()/2));
        queued.insert(queued.end(),changes.begin(),changes.end());
    }
    using Pending=RamRetirement;
    Impl(int c,int r,int n,int b,const tb::TensorOptions& o,const tb::Tensor& logical,bool ssd,const std::string& directory,Encode e)
        :capacity(c),rows(r),count(n),bits(b),fp16_bytes(std::size_t(rows)*512),affine_bytes(((3*rows+3)&~3)+std::size_t(rows)*256),options(o),encode(std::move(e)),live(logical) {
        if(c<=0 || r<=0 || n<=0 || (b!=16 && b!=8 && b!=4))throw std::invalid_argument("Invalid QSA RAM cache geometry");
        ssd_enabled=ssd;account->can_spill=ssd;
        check(cudaGetDevice(&device));
        int mapped=0;check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,device));mapped_capability=mapped!=0;
        worker=worker_for(device,mfq_current_cuda_stream());
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
            [this](const auto& changes){queue(changes);},
            [this](const auto& image){
                auto result=pool->allocate_view(image);
                static thread_local bool reported=false;
                if(!reported && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                    reported=true;std::fprintf(stderr,
                        "native_qsa_ram_admit_view actual_match=1 temporary_kv_vectors=0\n");
                }
                return result;
            });
        sources=tb::zeros({capacity,int64_t(sizeof(tq::RamBlockView))},options.dtype(tb::kUInt8));
        const auto tq4=tq4_bytes(rows);
        cache_stride=((b==16?2*fp16_bytes:b==8?2*affine_bytes:2*tq4)+31)&~std::size_t(15);
        spill_stride=(2*tq4+31)&~std::size_t(15);
        const auto cache_bytes=setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES");
        const auto sets=std::min<std::size_t>(INT_MAX/tq::ram_cache_ways,cache_bytes/cache_stride/tq::ram_cache_ways);
        if(sets) {
            const auto payload_bytes=sets*tq::ram_cache_ways*cache_stride;
            // RAM-only caches also acquire compressed tiers under pressure.
            // Reserve their metadata capacity without enlarging the payload.
            const auto max_sets=std::min<std::size_t>(INT_MAX/tq::ram_cache_ways,
                payload_bytes/std::min(cache_stride,spill_stride)/tq::ram_cache_ways);
            cache_slots=tb::empty({int64_t(max_sets*tq::ram_cache_ways*sizeof(tq::RamCacheSlot))},options.dtype(tb::kUInt8));
            cache_locks=tb::empty({int64_t(max_sets)},options.dtype(tb::kInt32));cache_frame=tb::empty({8},options.dtype(tb::kUInt8));
            cache_payload=tb::empty({int64_t(payload_bytes)},options.dtype(tb::kUInt8));
            gpu_cache={cache_slots.data_ptr<tq::RamCacheSlot>(),cache_locks.data_ptr<std::uint32_t>(),cache_frame.data_ptr<std::uint64_t>(),
                cache_payload.data_ptr<std::uint8_t>(),std::size_t(cache_payload.numel()),cache_stride,int(sets)};
            check(tq::tq_reset_ram_stage_cache(gpu_cache,mfq_current_cuda_stream()));
        }
        const auto parameter_options=options;
        cache_parameters=tb::empty({int64_t(sizeof(tq::RamStageCache))},parameter_options.dtype(tb::kUInt8));
        host_cache_parameters=tb::empty({int64_t(sizeof(tq::RamStageCache))},parameter_options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        upload_cache_parameters();
        worker->pool_account->physical.fetch_add(host_cache_parameters.nbytes());
    }
    void upload_cache_parameters() {
        std::memcpy(host_cache_parameters.data_ptr(),&gpu_cache,sizeof(gpu_cache));
        check(cudaMemcpyAsync(cache_parameters.data_ptr(),host_cache_parameters.data_ptr(),sizeof(gpu_cache),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
    }
    void configure_cache(bool has_ssd) {
        if(!gpu_cache.sets)return;
        // All tiers share the payload byte budget. Use the smallest present
        // tier as a base slot; larger images own aligned groups of slots.
        // Ordinary RAM graphs read this layout through the device descriptor.
        const auto usage=store->stats();
        auto stride=cache_stride;
        if(has_ssd || usage.tokens[2] || usage.tokens[3])stride=std::min(stride,spill_stride);
        else if(usage.tokens[1])stride=std::min(stride,(2*affine_bytes+31)&~std::size_t(15));
        if(gpu_cache.slot_bytes==stride)return;
        gpu_cache.slot_bytes=stride;
        gpu_cache.sets=int(std::min<std::size_t>(cache_locks.numel(),gpu_cache.payload_bytes/stride/tq::ram_cache_ways));
        check(tq::tq_reset_ram_stage_cache(gpu_cache,mfq_current_cuda_stream()));
        upload_cache_parameters();++publication_revision;
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
            "native_qsa_ram_cache_layout actual_match=1 ssd_priority=%d slot_bytes=%zu slots=%d payload_bytes=%zu\n",
            int(has_ssd),stride,gpu_cache.sets*tq::ram_cache_ways,gpu_cache.payload_bytes);
    }
    tq::RamStageCache cache_envelope()const {
        auto result=gpu_cache;result.slot_bytes=std::min(cache_stride,spill_stride);
        result.sets=int(cache_locks.numel());return result;
    }
    ~Impl() {
        try {wait();}catch(...){}
        (void)cudaStreamSynchronize(mfq_current_cuda_stream());
        try {worker->collect();}catch(...){}
        if(host_cache_parameters.defined())worker->pool_account->physical.fetch_sub(host_cache_parameters.nbytes());
    }
    std::vector<tq::RamKvImagePtr> convert(const std::vector<tq::RamKvImagePtr>& input,tq::RamKvTier tier) {
        check(cudaSetDevice(device));std::vector<tq::RamKvImagePtr> result(input.size());
        if(tier==tq::RamKvTier::Affine8) {
            std::size_t remaining=input.size();
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
                tq::CpuTqBlockFitter fitter(copy(*input[i]),rows,1);result[i]=pool->allocate(fitter.image({8,8}),&remaining);
            }});
        }else {
            std::vector<tq::MicroblockImage> source;source.reserve(input.size());for(const auto& image:input)source.push_back(copy(*image));
            const auto images=encode(source,std::vector<tq::Precision>(input.size(),{4,4}));
            if(images.size()!=input.size())throw std::runtime_error("QSA RAM GPU codec changed block count");
            result=pool->allocate_batch(images);
        }
        return result;
    }
    std::size_t target(std::size_t bytes)const {
        const auto cap=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");
        return cap?std::min(bytes,cap):bytes;
    }
    std::size_t budget()const {
        const auto available=worker->budget(),cap=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");
        return cap?std::min(cap,available):available;
    }
    std::size_t maintenance_budget(std::size_t limit)const {
        if(!ssd_enabled)return limit;
        // Leave one encoding batch for appends and hot recalls. Bound the
        // margin for small caches so pressure does not retrigger per token.
        const auto blocks=std::size_t(tq::RamPressureConfig{}.quantize_tokens/4);
        const auto margin=std::min(limit/32,2*fp16_bytes*blocks);
        return limit-margin;
    }
    std::size_t restore_append_blocks(std::size_t incoming)const {
        return incoming && incoming<std::size_t(capacity)?std::min(std::size_t(capacity)-incoming,
            std::size_t(tq::RamPressureConfig{}.quantize_tokens/4)):0;
    }
    struct GraphRead {
        std::size_t id;std::uint64_t generation;tq::RamKvImage image;
        tq::RamKvImagePtr resident;
    };
    struct GraphTask {
        Impl* owner;
        std::shared_ptr<StageArena> arena;
        std::shared_ptr<RamAccount> account;
        tb::Tensor flags;
        std::uint32_t* mapped=nullptr;
        int queries,count;
        std::vector<GraphRead> reads;
        std::vector<tq::RamKvChange> leases;
        std::vector<tq::RamKvSnapshotBlock> incoming;
        std::vector<tq::RamKvImagePtr> locked;
        std::vector<tq::RamKvReadInto> load_requests;
        std::future<void> lease_job;
        std::uint64_t lease_sequence=0;
        std::shared_ptr<std::vector<std::uint8_t>> fallback=std::make_shared<std::vector<std::uint8_t>>();
        std::size_t fallback_charge=0;
        GraphTask(Impl& value,int q,int n):owner(&value),arena(value.arena),account(value.worker->pool_account),queries(q),count(n) {
            flags=tb::zeros({16},value.options.device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true));
            void* pointer=nullptr;check(cudaHostGetDevicePointer(&pointer,flags.data_ptr(),0));mapped=static_cast<std::uint32_t*>(pointer);
            account->physical.fetch_add(flags.nbytes());
        }
        ~GraphTask(){clear();account->physical.fetch_sub(flags.nbytes()+fallback_charge);}
        std::uint32_t* host(){return flags.data_ptr<std::uint32_t>();}
        void settle_leases(){if(lease_job.valid())lease_job.get();}
        void clear()noexcept {
            try{settle_leases();}catch(...){}
            reads.clear();leases.clear();load_requests.clear();incoming.clear();locked.clear();
        }
        void reset(){clear();std::memset(flags.data_ptr(),0,flags.nbytes());std::atomic_thread_fence(std::memory_order_seq_cst);}
        void cancel()noexcept {
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *reinterpret_cast<volatile std::uint32_t*>(host()+tq::RamGraphReady)=2;
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        void serve() {
            wait_route_publication(host()+tq::RamGraphRequest,mfq_current_cuda_stream());
            auto& s=*owner;auto& a=*arena;auto& disk=*a.ssd;
            const auto* missing=a.host_missing.data_ptr<std::int32_t>();
            if(missing[0]<0 || missing[0]>queries*count)throw std::runtime_error("Invalid graph QSA SSD miss count");
            const auto stored=s.store->stats().blocks;
            std::memcpy(a.served_selected.data(),a.host_selected.data_ptr(),std::size_t(queries)*count*sizeof(tq::SelectedBlock));
            auto* records=disk.images.data_ptr<tq::RamBlockView>();auto* fetched=disk.ids.data_ptr<std::int32_t>();
            reads.reserve(missing[0]);
            load_requests.reserve(missing[0]);
            incoming.reserve(missing[0]);locked.reserve(missing[0]);std::size_t required=0;
            for(int at=0;at<missing[0];++at) {
                const auto id=std::size_t(missing[at+1]);
                if(id>=stored)throw std::runtime_error("Graph QSA SSD selection has no archived source");
                auto source=s.store->source(id);
                const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                const auto extent=(kb+15)&~std::size_t(15);
                if(extent>disk.stride || vb>disk.stride-extent || required>SIZE_MAX-extent-vb)
                    throw std::runtime_error("Graph QSA SSD image exceeds its prepared slot");
                auto resident=s.pool->try_reserve(source);
                if(!resident)required+=extent+vb;
                locked.push_back(std::move(resident));incoming.push_back(std::move(source));
            }
            // The mapped GPU stage is shared between layers. Its bytes may
            // be overwritten before deferred RAM admission runs. One compact
            // task-owned buffer preserves fallback images through that join.
            if(fallback->size()<required) {
                fallback->resize(required);
                const auto extra=fallback->capacity()-fallback_charge;
                account->physical.fetch_add(extra);fallback_charge+=extra;
            }
            std::size_t fallback_at=0;
            for(int at=0;at<missing[0];++at) {
                const auto id=std::size_t(missing[at+1]);
                if(id>=stored)throw std::runtime_error("Graph QSA SSD selection has no archived source");
                const auto& source=incoming[at];
                const auto& resident=locked[at];tq::RamKvImage image;
                const auto index=reads.size(),offset=index*disk.stride;
                if(resident) {
                    const auto extent=std::size_t(resident->value-resident->key);
                    if(extent>disk.stride || resident->value_bytes>disk.stride-extent)
                        throw std::runtime_error("Graph QSA SSD image exceeds its prepared slot");
                    if(!source.image)load_requests.push_back({&source,const_cast<std::uint8_t*>(resident->key),
                        extent+resident->value_bytes});
                    image=*resident;records[index]=s.record({id,source.generation,resident,{},source.image?source.tier:source.backing->tier});
                }else {
                    const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                    const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                    const auto key_extent=(kb+15)&~std::size_t(15);
                    if(key_extent>disk.stride || vb>disk.stride-key_extent)throw std::runtime_error("Graph QSA SSD image exceeds its prepared slot");
                    auto* retained=fallback->data()+fallback_at;fallback_at+=key_extent+vb;
                    load_requests.push_back({&source,retained,key_extent+vb});
                    image.owner=fallback;image.key=retained;image.value=retained+key_extent;image.key_bytes=kb;image.value_bytes=vb;
                    image.key_bits=source.image?source.image->key_bits:source.backing->key_bits;
                    image.value_bits=source.image?source.image->value_bits:source.backing->value_bits;
                    image.key_state_bits=source.image?source.image->key_state_bits:source.backing->key_state_bits;
                    image.value_state_bits=source.image?source.image->value_state_bits:source.backing->value_state_bits;
                    records[index]={{disk.mapped_payload+offset,disk.mapped_payload+offset+key_extent,
                        tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)},
                        std::uint32_t(image.key_bytes),std::uint32_t(image.value_bytes),source.generation};
                }
                fetched[index]=std::int32_t(id);
                reads.push_back({id,source.generation,std::move(image),std::move(resident)});
            }
            s.store->read_cpu_into_batch(load_requests,false);
            for(std::size_t at=0;at<reads.size();++at)if(!reads[at].resident) {
                const auto& image=reads[at].image;auto* payload=disk.payload.data_ptr<std::uint8_t>()+at*disk.stride;
                const auto extent=std::size_t(image.value-image.key);
                std::memcpy(payload,image.key,image.key_bytes);std::memcpy(payload+extent,image.value,image.value_bytes);
            }
            host()[tq::RamGraphImages]=std::uint32_t(reads.size());
            publish_mapped_flag(host()+tq::RamGraphReady);
            if(!reads.empty() && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                const auto direct=std::count_if(reads.begin(),reads.end(),[](const auto& read){return bool(read.resident);});
                const unsigned kinds=(direct?1u:0u)|(std::size_t(direct)<reads.size()?2u:0u);
                static thread_local unsigned reported[9]{};
                if(kinds&~reported[queries]) {
                    reported[queries]|=kinds;
                    std::fprintf(stderr,"native_qsa_ram_read_into actual_match=1 queries=%d direct_slots=%zu fallback_views=%zu no_pinned_growth=1\n",
                        queries,std::size_t(direct),reads.size()-std::size_t(direct));
                }
            }
            // Publication queues retain replaced RAM owners until their GPU
            // readers finish. Admission runs only after the whole graph has
            // joined, so touching and leasing need not stall this copy.
            RamClaims ids(s.worker->claims,s.capacity,std::size_t(queries)*count);
            const auto* selected=a.served_selected.data();
            for(int at=0;at<queries*count;++at)if(selected[at].id>=0 && std::size_t(selected[at].id)<stored && selected[at].token_mask)
                ids.add(std::size_t(selected[at].id));
            ids.clear();
            const auto epoch=s.graph_epoch;
            // The ready flag releases the GPU before LRU work. Keep that work
            // off the host data-service thread and join it before admission.
            // Pending descriptor publications retain replaced RAM images.
            lease_job=s.worker->submit([this,selected=std::move(ids.ids),epoch] {
                leases=owner->store->lease(selected,epoch);
            },lease_sequence);
        }
        void finish() {
            settle_leases();
            if(host()[tq::RamGraphError])throw std::runtime_error("Graph QSA SSD source changed during its read");
            // DecodeWindow joins every GPU reader before calling finish.
            // External prefix/snapshot readers keep their independent leases.
            leases.clear();
            if(reads.empty())return;
            auto& s=*owner;const auto limit=s.budget();
            for(const auto& read:reads) {
                if(read.resident)s.store->admit_read(read.id,read.generation,read.resident,limit);
                else s.store->admit_read(read.id,read.generation,read.image,limit);
            }
            if(!reads.empty() && !s.reported_ssd_fetch && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                s.reported_ssd_fetch=true;const auto stats=s.store->stats();
                std::fprintf(stderr,"qsa_ram_ssd_fetch actual_match=1 misses=%zu read_bytes=%llu promotions=%llu graph_task=1 packed_hits_skip_ssd=1\n",
                    reads.size(),static_cast<unsigned long long>(stats.ssd_read_bytes),static_cast<unsigned long long>(stats.promoted_blocks));
            }
            s.update_account();clear();
        }
    };
    void update_account() {
        const auto stats=store->stats();account->logical.store(stats.ram_bytes);
        account->target.store(target(stats.resident_target_bytes));
    }
    std::size_t compacted_physical=0,compacted_logical=0;
    void resolve_capacity() {
        auto needed=required_capacity.load();if(!needed)return;
        const auto before=budget();auto after=before;std::size_t released=0;
        required_capacity.store(0);
        if(needed>after) {
            // Finish already-submitted finite work before treating leases as
            // unavailable capacity. No graph can wait for this foreground path.
            const auto retire=[&] {
                drain();check(cudaStreamSynchronize(mfq_current_cuda_stream()));drain();
            };
            retire();
            pool->trim(0);needed=store->stats().ram_bytes;after=budget();
            if(needed<=after){sampled_budget=after;budget_sampled=true;return;}
            RamForegroundCodec foreground(foreground_codec);
            // Quantizing images cannot release pages stranded in partial
            // pinned slabs. Pack exact images into existing holes first.
            const auto physical=worker->pool_account->physical.load();
            const auto free=mfq::sample_host_memory().available_host_bytes();
            const auto shortage=std::max(needed-after,tq::ram_kv_reclaim_bytes(free,physical,physical));
            std::size_t logical=0;
            {std::lock_guard<std::mutex> lock(worker->mutex);
                for(const auto& weak:worker->accounts)if(const auto account=weak.lock())logical+=account->logical.load();}
            const auto quantum=std::max<std::size_t>(4ULL<<20,physical/64);
            const auto changed=[](auto a,auto b){return a>b?a-b:b-a;};
            if(shortage && (!compacted_physical || changed(physical,compacted_physical)>=quantum ||
                    changed(logical,compacted_logical)>=quantum)) {
                // Foreground compaction must not race another layer's codec
                // allocating from the same set of pinned destination holes.
                std::uint64_t sequence=0;auto idle=worker->submit([]{},sequence);
                worker->prioritize(sequence);idle.get();
                retire();pool->trim(0);needed=store->stats().ram_bytes;after=budget();
                if(needed<=after){sampled_budget=after;budget_sampled=true;return;}
                const auto compact_before=worker->pool_account->physical.load();
                const auto moved=store->relocate([&](const auto& images){return pool->compact(images,shortage);});
                if(moved) {
                    retire();pool->trim(0);needed=store->stats().ram_bytes;after=budget();
                    if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
                        "native_qsa_ram_compact actual_match=1 images=%zu physical_before=%zu physical_after=%zu\n",
                        moved,compact_before,worker->pool_account->physical.load());
                }
                compacted_physical=worker->pool_account->physical.load();compacted_logical=logical;
            }
            if(needed<=after) {sampled_budget=after;budget_sampled=true;return;}
            store->maintain(maintenance_budget(after),false);update_account();retire();
            needed=store->stats().ram_bytes;after=budget();
            // Explicit per-layer caps cannot borrow optional expert RAM.
            const auto cap=setting("TYLOINFER_QSA_RAM_BUDGET_BYTES");
            if(needed>after && !ssd_enabled && (!cap || after<cap)) {
                released=worker->reclaim_capacity(target(needed));after=budget();
            }
            if(needed>after) {
                // Spill cold KV before displacing RAM experts. RAM-only
                // configurations instead reclaim capacity or report exhaustion.
                const auto result=store->maintain(maintenance_budget(after),ssd_enabled);update_account();
                if(!result.fits() && !result.needs_readers)throw std::runtime_error(ssd_enabled?
                    "QSA RAM cannot hold the protected high-precision KV tiers; reduce the context":
                    "QSA RAM KV exhausted; enable session SSD KV or reduce the context");
            }
        }
        sampled_budget=after;budget_sampled=true;
        if(released && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
            "native_qsa_ram_capacity_reclaim actual_match=1 required=%zu before=%zu after=%zu released=%zu foreground=1\n",
            needed,before,after,released);
    }
    void wait(){
        if(maintenance.valid()) {
            if(maintenance.wait_for(std::chrono::seconds(0))!=std::future_status::ready)
                worker->prioritize(maintenance_sequence);
            maintenance.get();
        }
        resolve_capacity();
    }
    tq::RamBlockView record(const tq::RamKvChange& change) {
        auto result=host_sources.at(change.id);result.generation=change.generation;
        const auto& image=change.image;
        if(!image){
            result.image.key=result.image.value=nullptr;
            if(const auto& source=change.backing) {
                result.image.key_bits=tq::tq_precision_flags(source->key_bits,source->key_state_bits);
                result.image.value_bits=tq::tq_precision_flags(source->value_bits,source->value_state_bits);
                result.key_bytes=std::uint32_t(source->key_bytes);result.value_bytes=std::uint32_t(source->value_bytes);
            }
            return result;
        }
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
        const auto last=std::max_element(p.owners.begin(),p.owners.end(),[](const auto& a,const auto& b){return a.id<b.id;})->id;
        if(last>=std::size_t(capacity))throw std::out_of_range("QSA RAM publication exceeds capacity");
        if(host_sources.size()<=last)host_sources.resize(last+1);
        // Several precision transitions can replace an ID. Retain only its
        // final descriptor, without allocating a tree node for every block.
        RamClaims latest(worker->claims,capacity,p.owners.size());
        for(const auto& change:p.owners){auto rec=record(change);host_sources[change.id]=rec;latest.add(change.id);}
        latest.clear();const auto n=latest.ids.size();
        auto publication=worker->publication(n,options,mapped_capability);publication.owners=std::move(p.owners);p=std::move(publication);
        auto* ids=p.host_ids.data_ptr<std::int32_t>();auto* descriptors=p.host_descriptors.data_ptr<tq::RamBlockView>();
        std::size_t at=0;for(auto id:latest.ids){ids[at]=std::int32_t(id);descriptors[at++]=host_sources[id];}
        const auto* ids_source=p.mapped_ids;const auto* descriptors_source=p.mapped_descriptors;
        if(!ids_source) {
            check(cudaMemcpyAsync(p.ids.data_ptr(),p.host_ids.data_ptr(),n*sizeof(std::int32_t),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            check(cudaMemcpyAsync(p.descriptors.data_ptr(),p.host_descriptors.data_ptr(),n*sizeof(tq::RamBlockView),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            ids_source=p.ids.data_ptr<std::int32_t>();descriptors_source=p.descriptors.data_ptr<tq::RamBlockView>();
        }
        // Consume these small immutable records directly on devices that can
        // map pinned host memory. The retirement owns both the records and
        // replaced images until the stream has passed their publication.
        check(tq::tq_publish_ram_sources(sources.data_ptr<tq::RamBlockView>(),capacity,ids_source,descriptors_source,int(n),mfq_current_cuda_stream(),live.data_ptr<tq::TqBlockView>()));
        ++publication_revision;
        const bool reused=p.done!=nullptr;
        if(!p.done)check(cudaEventCreateWithFlags(&p.done,cudaEventDisableTiming));
        check(cudaEventRecord(p.done,mfq_current_cuda_stream()));worker->retire(std::move(p));
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            static thread_local unsigned reported=0;const unsigned mask=(mapped_capability?1u:2u)|(reused?4u:8u);
            if((reported&mask)!=mask){reported|=mask;std::fprintf(stderr,
                "native_qsa_ram_publish actual_match=1 mapped_descriptors=%d h2d_copies=%d event_reused=%d\n",
                int(mapped_capability),mapped_capability?0:2,int(reused));}
        }
        update_account();
    }
};
QsaRamCache::QsaRamCache(int c,int r,int n,int bits,const tb::TensorOptions& options,const Tensor& live,bool ssd,const std::string& directory,Encode encode)
    :impl_(std::make_unique<Impl>(c,r,n,bits,options,live,ssd,directory,std::move(encode))){
    std::lock_guard lock(impl_->account->owner_mutex);impl_->account->owner=this;
}
QsaRamCache::~QsaRamCache(){
    const auto account=impl_->account;
    std::lock_guard lock(account->owner_mutex);account->owner=nullptr;
    impl_.reset();account->logical.store(0);account->target.store(0);account->restore_target.store(0);
}
std::shared_ptr<void> QsaRamCache::hold_workspace(std::size_t bytes) {
    return hold_worker_workspace(impl_->worker,bytes);
}
QsaRamWorkspaceProgress QsaRamCache::reclaim_shared_workspace() {
    const auto worker=impl_->worker;
    std::vector<std::shared_ptr<RamAccount>> accounts;
    {
        std::lock_guard lock(worker->mutex);
        for(const auto& weak:worker->accounts)if(auto account=weak.lock())accounts.push_back(std::move(account));
    }
    // The predictor can share this pool without appearing in the target's
    // layer list. Refresh and maintain every live participant of the budget.
    std::size_t accounted_before=0,actual_before=0;
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(account->owner) {
            accounted_before+=account->logical.load();account->owner->impl_->update_account();
            actual_before+=account->logical.load();
        }
    }
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(account->owner)account->owner->wait();
    }
    check(cudaStreamSynchronize(mfq_current_cuda_stream()));
    QsaRamWorkspaceProgress progress;
    for(const auto& account:accounts) {
        std::lock_guard lock(account->owner_mutex);
        if(!account->owner)continue;
        account->owner->drain();account->owner->impl_->update_account();
        progress.logical+=account->logical.load();
    }
    progress.physical=worker->pool_account->physical.load();
    if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_shared_workspace actual_match=1 participants=%zu workspace_bytes=%zu accounted_before=%zu actual_before=%zu logical_bytes=%zu physical_bytes=%zu free_bytes=%llu\n",
        accounts.size(),worker->workspace_bytes.load(),accounted_before,actual_before,progress.logical,progress.physical,
        static_cast<unsigned long long>(mfq::sample_host_memory().available_host_bytes()));
    return progress;
}
void QsaRamCache::prepare() {
    auto& s=*impl_;if(!s.arena)s.arena=arena_for(s.device,mfq_current_cuda_stream(),s.capacity,s.count,2*s.fp16_bytes,s.options,s.worker->pool_account);
    s.drain();s.configure_cache(s.store->stats().tokens[3]!=0);
    s.prepared_revision=s.publication_revision;
}
QsaRamCache::Tensor QsaRamCache::prepared_views()const {
    if(!impl_->arena)throw std::logic_error("QSA RAM staging arena was not prepared");
    return impl_->arena->views;
}
std::optional<QsaRamPrefillStage> QsaRamCache::stage_prefill(std::size_t blocks) {
    auto& s=*impl_;if(!blocks)return {};
    const auto stream=mfq_current_cuda_stream();cudaStreamCaptureStatus capture;
    check(cudaStreamIsCapturing(stream,&capture));
    if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("Layer KV staging requires uncaptured prefill");
    constexpr auto maximum_io_records=(1ULL<<20)/sizeof(tq::RamKvReadInto);
    const auto io_bytes=tq::RamKvStore::sequential_read_workspace(std::min(blocks,std::size_t(maximum_io_records)));
    const auto io_reservation=hold_worker_workspace(s.worker,io_bytes);
    if(mfq::sample_host_memory().available_host_bytes()<io_bytes)
        prepare_qsa_ram_workspace(io_bytes,[this]{return reclaim_shared_workspace();});
    // Re-evaluate physical pressure before taking leases that prevent KV
    // reclamation. Waiting only for an earlier codec plan uses a stale budget.
    const auto snapshot=this->snapshot(blocks);
    std::size_t bytes=0,largest=0;
    const auto image_bytes=[](const tq::RamKvSnapshotBlock& source) {
        const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
        const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
        if(kb>SIZE_MAX-15)throw std::overflow_error("Prefill KV image overflows");
        const auto extent=(kb+15)&~std::size_t(15);
        if(extent>SIZE_MAX-15 || vb>SIZE_MAX-15-extent)throw std::overflow_error("Prefill KV image overflows");
        return (extent+vb+15)&~std::size_t(15);
    };
    for(const auto& source:snapshot->blocks) {
        if(!source.image && !source.backing)throw std::runtime_error("Prefill KV source is absent");
        const auto count=image_bytes(source);
        if(count>SIZE_MAX-bytes)throw std::overflow_error("Prefill KV layer overflows");
        bytes+=count;largest=std::max(largest,count);
    }
    auto context=default_context(s.device);
    const auto free=context->available_memory(),workspace=context->workspace_available();
    const auto available=free>SIZE_MAX-workspace?SIZE_MAX:free+workspace;
    const auto table_bytes=s.live.nbytes();
    if(table_bytes>available || bytes>available-table_bytes)return {};
    constexpr auto maximum_records=(1ULL<<20)/sizeof(tq::RamKvReadInto);
    const auto packet_bytes=std::max(largest,std::size_t(8ULL<<20));
    auto& retained=s.worker->prefill_transfer;
    if(!retained || retained->packet_bytes<packet_bytes) {
        if(retained)for(auto event:retained->done)if(event)check(cudaEventSynchronize(event));
        auto next=std::make_shared<PrefillTransfer>();next->account=s.worker->pool_account;
        next->packet_bytes=packet_bytes;
        const auto cpu=s.options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true);
        for(int slot=0;slot<2;++slot) {
            next->packets[slot]=tb::empty({int64_t(packet_bytes)},cpu);
            next->records[slot]=tb::empty({int64_t(maximum_records*sizeof(tq::TqBlockView))},cpu);
        }
        for(int slot=0;slot<2;++slot)next->charge+=next->packets[slot].nbytes()+next->records[slot].nbytes();
        next->account->physical.fetch_add(next->charge);retained=std::move(next);
    }
    // All layers on this model stream share the ring. Each slot's event
    // protects both its encoded packet and its bounded descriptor packet.
    const auto host=retained;
    QsaRamPrefillStage result;
    result.payload=tb::empty({int64_t(bytes)},s.options.dtype(tb::kUInt8));result.views=s.live.clone();
    result.payload.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    result.views.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    constexpr int ring=2;
    std::size_t first=0,offset=0,turn=0;bool submitted=false;
    try {
        while(first<blocks) {
            const auto slot=turn++%ring;
            if(host->done[slot])check(cudaEventSynchronize(host->done[slot]));
            auto* records=host->records[slot].data_ptr<tq::TqBlockView>();
            std::vector<tq::RamKvReadInto> reads;
            std::size_t used=0,end=first,packed=0;
            // Bound the CPU request descriptors independently of KV image
            // sizes; this packet limit never limits the GPU attention tile.
            while(end<blocks && reads.size()<maximum_records) {
                const auto& source=snapshot->blocks[end];const auto count=image_bytes(source);
                if(count>packet_bytes-used)break;
                const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                const auto key_bits=source.image?source.image->key_bits:source.backing->key_bits;
                const auto value_bits=source.image?source.image->value_bits:source.backing->value_bits;
                const auto key_states=source.image?source.image->key_state_bits:source.backing->key_state_bits;
                const auto value_states=source.image?source.image->value_state_bits:source.backing->value_state_bits;
                auto* device=result.payload.data_ptr<std::uint8_t>()+offset+used;
                records[end-first]={device,device+((kb+15)&~std::size_t(15)),
                    tq::tq_precision_flags(key_bits,key_states),tq::tq_precision_flags(value_bits,value_states)};
                reads.push_back({&source,host->packets[slot].data_ptr<std::uint8_t>()+packed,kb+vb});
                packed+=kb+vb;
                used+=count;++end;
            }
            s.store->read_packed_cpu_into_batch(reads);
            // Expand backwards in the existing packet so unread source bytes
            // survive overlap while device descriptors retain their alignment.
            std::size_t expanded=used;
            for(auto i=reads.size();i-->0;) {
                const auto& source=*reads[i].source;
                const auto kb=source.image?source.image->key_bytes:source.backing->key_bytes;
                const auto vb=source.image?source.image->value_bytes:source.backing->value_bytes;
                const auto extent=(kb+15)&~std::size_t(15);
                expanded-=image_bytes(source);
                auto* destination=host->packets[slot].data_ptr<std::uint8_t>()+expanded;
                if(destination!=reads[i].destination || extent!=kb) {
                    std::memmove(destination+extent,reads[i].destination+kb,vb);
                    std::memmove(destination,reads[i].destination,kb);
                }
            }
            check(cudaMemcpyAsync(result.payload.data_ptr<std::uint8_t>()+offset,
                host->packets[slot].data_ptr(),used,cudaMemcpyHostToDevice,stream));submitted=true;
            check(cudaMemcpyAsync(result.views.data_ptr<tq::TqBlockView>()+first,records,
                (end-first)*sizeof(*records),cudaMemcpyHostToDevice,stream));
            if(!host->done[slot])check(cudaEventCreateWithFlags(&host->done[slot],cudaEventDisableTiming));
            check(cudaEventRecord(host->done[slot],stream));offset+=used;first=end;
        }
    }catch(...){
        if(submitted)(void)cudaStreamSynchronize(stream);
        throw;
    }
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1')std::fprintf(stderr,
        "native_qsa_prefill_layer_stage actual_match=1 blocks=%zu encoded_bytes=%zu original_precision=1 packet_bytes=%zu payload_cuda=%d descriptors_cuda=%d pinned_ring_bytes=%zu shared_host_ring=1\n",
        blocks,bytes,packet_bytes,int(result.payload.is_cuda()),int(result.views.is_cuda()),host->charge);
    return result;
}
bool QsaRamCache::collect_graph_selection(const Tensor& selected,int queries,int count) {
    auto& s=*impl_;
    if(!s.arena || queries<1 || queries>8 || count!=s.count)throw std::logic_error("QSA graph selection was not prepared");
    auto& a=*s.arena;const auto stream=mfq_current_cuda_stream();
    if(!a.mapped_missing || !a.mapped_selected)return false;
    check(tq::tq_collect_ram_missing(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
        selected.data_ptr<tq::SelectedBlock>(),queries,count,a.first.data_ptr<std::uint32_t>(),
        a.missing.data_ptr<std::int32_t>()+1,a.missing.data_ptr<int>(),
        static_cast<const tq::RamStageCache*>(s.cache_parameters.data_ptr()),stream));
    check(tq::tq_export_ram_selection(a.missing.data_ptr<std::int32_t>(),selected.data_ptr<tq::SelectedBlock>(),
        a.mapped_missing,a.mapped_selected,queries,count,stream));
    return true;
}
bool QsaRamCache::graph_task_supported()const {
    const auto& s=*impl_;
    return s.mapped_capability && (!s.arena || (s.arena->mapped_missing && s.arena->mapped_selected)) && s.store->graph_reads_supported();
}
void QsaRamCache::prepare_graph_task(std::uint64_t epoch) {
    auto& s=*impl_;
    if(!s.arena)prepare();
    wait();
    if(!graph_task_supported())throw std::logic_error("QSA SSD graph task requires mapped control memory");
    // The task graph keeps one compact geometry across RAM/SSD changes.
    // Larger images own slot groups; only images exceeding a set bypass it.
    s.configure_cache(true);s.graph_epoch=epoch;
    if(!s.arena->ssd) {
        // Payloads are opaque and may retain alignment padding from a prefix.
        // Match the existing bounded transient slot instead of assuming an
        // exact codec byte count for every externally retained backing.
        auto disk=std::make_shared<SsdGraphArena>();disk->account=s.worker->pool_account;disk->stride=s.arena->stride;
        const auto n=int64_t(8)*s.count;const auto host=s.options.device(tb::kCPU).pinned_memory(true);
        disk->payload=tb::empty({n*int64_t(disk->stride)},host.dtype(tb::kUInt8));
        disk->ids=tb::empty({n},host.dtype(tb::kInt32));
        disk->images=tb::empty({n*int64_t(sizeof(tq::RamBlockView))},host.dtype(tb::kUInt8));
        disk->backup=tb::empty({n*int64_t(sizeof(tq::RamBlockView))},s.options.dtype(tb::kUInt8));
        disk->saved=tb::empty({1},s.options.dtype(tb::kInt32));
        void* pointer=nullptr;check(cudaHostGetDevicePointer(&pointer,disk->payload.data_ptr(),0));disk->mapped_payload=static_cast<std::uint8_t*>(pointer);
        check(cudaHostGetDevicePointer(&pointer,disk->ids.data_ptr(),0));disk->mapped_ids=static_cast<std::int32_t*>(pointer);
        check(cudaHostGetDevicePointer(&pointer,disk->images.data_ptr(),0));disk->mapped_images=static_cast<tq::RamBlockView*>(pointer);
        disk->bytes=disk->payload.nbytes()+disk->ids.nbytes()+disk->images.nbytes();disk->account->physical.fetch_add(disk->bytes);
        s.arena->ssd=std::move(disk);
    }
}
QsaRamCache::Tensor QsaRamCache::stage_graph_task(const Tensor& selected,int queries,int count,const Tensor& live,const Tensor& position,int appended) {
    auto& s=*impl_;auto* window=DecodeWindow::recording();
    if(!window || !s.arena || !s.arena->ssd || queries<1 || queries>8 || count!=s.count)
        throw std::logic_error("QSA SSD graph task was not prepared");
    auto state=window->workspace<Impl::GraphTask>(&s,[&]{return std::make_shared<Impl::GraphTask>(s,queries,count);});
    if(state->queries!=queries || state->count!=count)throw std::logic_error("QSA SSD graph task query geometry changed");
    const auto stream=mfq_current_cuda_stream();cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(stream,&capture));
    const bool warm=capture==cudaStreamCaptureStatusNone;
    if(warm)state->reset();
    window->enroll({state.get(),[state]{state->reset();},[state]{state->serve();},[state]{state->finish();},
        [state]{state->cancel();},[state]{state->clear();},[state]{state->clear();}});
    auto& a=*s.arena;auto& disk=*a.ssd;
    if(appended>0)check(tq::tq_publish_ram_append(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,live.data_ptr<tq::TqBlockView>(),
        position.defined()?position.data_ptr<int64_t>():nullptr,int64_t(s.graph_epoch),appended,unsigned(s.fp16_bytes),unsigned(s.affine_bytes),stream));
    if(!collect_graph_selection(selected,queries,count))throw std::logic_error("QSA graph task lost mapped selection memory");
    signal_mapped_flag(state->mapped+tq::RamGraphRequest,stream);
    const auto stage=[&](tq::RamStageStep step) {
        check(tq::tq_stage_ram_selected_step(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
            a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,step,nullptr,nullptr,stream));
    };
    stage(tq::RamStageStep::Available);
    if(warm)state->serve();
    wait_mapped_flag(state->mapped+tq::RamGraphReady,stream);
    check(tq::tq_publish_ram_graph(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,disk.mapped_ids,disk.mapped_images,
        state->mapped,disk.backup.data_ptr<tq::RamBlockView>(),disk.saved.data_ptr<std::uint32_t>(),queries*count,stream));
    check(tq::tq_cancel_ram_graph_selection(selected.data_ptr<tq::SelectedBlock>(),queries*count,disk.saved.data_ptr<std::uint32_t>(),stream));
    stage(tq::RamStageStep::Hydrated);
    check(tq::tq_restore_ram_graph(s.sources.data_ptr<tq::RamBlockView>(),disk.mapped_ids,disk.backup.data_ptr<tq::RamBlockView>(),
        disk.saved.data_ptr<std::uint32_t>(),queries*count,stream));
    if(std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH")) {
        static thread_local unsigned reported=0;const unsigned mask=1u<<(queries-1);
        if(!(reported&mask)){reported|=mask;std::fprintf(stderr,"native_qsa_ssd_graph_task actual_match=1 queries=%d single_submission=1 bounded_cpu_reads=1 deferred_lease=1\n",queries);}
    }
    return a.views;
}
void QsaRamCache::append(std::vector<tq::MicroblockImage> images,std::uint64_t epoch,bool fresh) {
    auto& s=*impl_;
    if(s.bits==4 && !fresh) {
        RamForegroundCodec foreground(s.foreground_codec);
        images=s.encode(images,std::vector<tq::Precision>(images.size(),{4,4}));
    }
    const auto tq4=tq4_bytes(s.rows);
    const auto tier=s.bits==16?tq::RamKvTier::Fp16:s.bits==8 || fresh?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
    for(const auto& image:s.pool->allocate_batch(images))s.store->append_at_tier(image,{2*s.fp16_bytes,2*s.affine_bytes,2*tq4},tier,epoch);
    s.update_account();s.drain();
    const auto usage=s.store->stats();
    if(!s.budget_sampled || usage.ram_bytes>s.sampled_budget || (s.bits==4 && usage.tokens[1]>=2048))
        observe({},epoch);
}
void QsaRamCache::drain(){impl_->drain();}
void QsaRamCache::release_idle() {
    auto& s=*impl_;const auto& worker=s.worker;
    {
        std::lock_guard<std::mutex> lock(worker->mutex);
        for(const auto& weak:worker->accounts)if(auto account=weak.lock())
            if(account->logical.load() || account->restore_target.load())return;
    }
    bool pending=false;
    {std::lock_guard<std::mutex> lock(worker->retired_mutex);pending=!worker->retired.empty();}
    const auto before=worker->pool_account->physical.load();
    MfqCudaGuard device(s.device);
    // The last reset publishes null descriptors asynchronously. Join only its
    // finite stream work, then release the retired images across all layers.
    if(pending)check(cudaStreamSynchronize(worker->stream));
    worker->collect();
    worker->prefill_transfer.reset();
    auto metadata=s.store->release_idle_metadata()+s.host_sources.capacity()*sizeof(tq::RamBlockView);
    std::vector<tq::RamBlockView>().swap(s.host_sources);
    {
        std::lock_guard<std::mutex> lock(s.queue_mutex);
        if(s.queued.empty()) {metadata+=s.queued.capacity()*sizeof(tq::RamKvChange);std::vector<tq::RamKvChange>().swap(s.queued);}
    }
    for(auto& publication:worker->publications) {
        metadata+=publication.owners.capacity()*sizeof(tq::RamKvChange);
        std::vector<tq::RamKvChange>().swap(publication.owners);
    }
    worker->budget();
    const auto after=worker->pool_account->physical.load();
    if(before>after && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_ram_idle_release actual_match=1 before_bytes=%zu after_bytes=%zu released_bytes=%zu\n",before,after,before-after);
    if(metadata && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_ram_idle_metadata actual_match=1 released_bytes=%zu\n",metadata);
}
QsaRamCache::Tensor QsaRamCache::stage(const Tensor& selected,int queries,int count,const Tensor& live,
        const Tensor& position,std::int64_t host_position,int appended,bool selection_ready) {
    auto& s=*impl_;if(queries<1 || queries>8 || count!=s.count)throw std::invalid_argument("QSA RAM staging requires its prepared query window");
    if(!s.arena)prepare();
    std::vector<tq::RamKvChange> leased,temporary;
    struct CopyGuard {
        cudaStream_t stream;bool pending=false;
        // SSD errors can unwind after the first copy was submitted. Its RAM
        // leases must outlive that finite copy even when no attention runs.
        ~CopyGuard(){if(pending)(void)cudaStreamSynchronize(stream);}
    } copy_guard{mfq_current_cuda_stream()};
    bool split_copy=false,hydrate_missing=false;
    const auto publish_append=[&] {
        if(appended>0)check(tq::tq_publish_ram_append(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,live.data_ptr<tq::TqBlockView>(),
            position.defined()?position.data_ptr<int64_t>():nullptr,host_position,appended,unsigned(s.fp16_bytes),unsigned(s.affine_bytes),mfq_current_cuda_stream()));
    };
    const auto stage_step=[&](tq::RamStageStep step) {
        auto& a=*s.arena;
        check(tq::tq_stage_ram_selected_step(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
            a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.gpu_cache,step,nullptr,nullptr,mfq_current_cuda_stream()));
    };
    const bool has_ssd=s.store->stats().tokens[3]!=0;
    if(!DecodeWindow::recording())s.configure_cache(has_ssd);
    if(has_ssd) {
        cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&status));
        if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("SSD KV fetch requires an eager graph boundary");
        // Finish this layer's outstanding replacements before collecting its
        // actual SSD misses. The new selected-hit protection keeps packed
        // hits readable even while incoming misses exceed cache capacity.
        s.wait();s.drain();
        auto& a=*s.arena;
        // The preceding graph segment has already joined these pinned copies.
        // Background precision/placement changes invalidate that snapshot;
        // repeat only the small collector when a publication raced it.
        s.configure_cache(s.store->stats().tokens[3]!=0);
        const bool reuse_selection=selection_ready && s.prepared_revision==s.publication_revision;
        if(!reuse_selection) {
        check(tq::tq_collect_ram_missing(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,
            selected.data_ptr<tq::SelectedBlock>(),queries,count,a.first.data_ptr<std::uint32_t>(),
            a.missing.data_ptr<std::int32_t>()+1,a.missing.data_ptr<int>(),s.gpu_cache,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(a.host_missing.data_ptr(),a.missing.data_ptr(),std::size_t(1+queries*count)*sizeof(int),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        check(cudaMemcpyAsync(a.host_selected.data_ptr(),selected.data_ptr(),std::size_t(queries)*count*sizeof(tq::SelectedBlock),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        if(selection_ready && std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            static thread_local unsigned reported=0;const unsigned mask=reuse_selection?1u:2u;
            if(!(reported&mask)){reported|=mask;std::fprintf(stderr,"native_qsa_graph_selection actual_match=1 reused=%d revision_check=1\n",int(reuse_selection));}
        }
        const auto* blocks=a.host_selected.data_ptr<tq::SelectedBlock>();
        // The copy above also retires the descriptor upload. Its publication
        // owner is no longer a reader lease and must not prevent admission.
        s.drain();
        const auto stored=s.store->stats().blocks;
        const auto* missing=a.host_missing.data_ptr<std::int32_t>();
        if(missing[0]<0 || missing[0]>queries*count)throw std::runtime_error("Invalid QSA RAM miss count");
        for(int at=0;at<missing[0];++at)if(std::size_t(missing[at+1])<stored){hydrate_missing=true;break;}
        {
            // This layer's producer was joined above. Its store retains the
            // immutable sources while the selected leases are assembled;
            // admission cannot start until all those leases are acquired.
            publish_append();copy_guard.pending=true;stage_step(tq::RamStageStep::Available);
            split_copy=true;
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
    if(split_copy) {if(hydrate_missing)stage_step(tq::RamStageStep::Hydrated);}
    else if(s.gpu_cache.sets)check(tq::tq_stage_ram_selected_cached(s.sources.data_ptr<tq::RamBlockView>(),s.capacity,selected.data_ptr<tq::SelectedBlock>(),queries,count,
        a.first.data_ptr<std::uint32_t>(),a.views.data_ptr<tq::TqBlockView>(),a.payload.data_ptr(),a.payload.numel(),a.stride,s.cache_envelope(),nullptr,nullptr,mfq_current_cuda_stream(),
        static_cast<const tq::RamStageCache*>(s.cache_parameters.data_ptr())));
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
    s.resolve_capacity();
    const auto physical_budget=s.budget();s.sampled_budget=physical_budget;s.budget_sampled=true;
    auto target=physical_budget;const auto usage=s.store->stats();
    const bool pressure=usage.ram_bytes>physical_budget;
    if(pressure)target=s.maintenance_budget(physical_budget);
    if(!pressure && s.bits==4 && usage.tokens[1]>=2048) {
        // Keep a partial batch of fresh Affine8 images off the serial decode
        // path. Fit only complete2048-token batches on the background worker.
        const auto saving=512*(2*s.affine_bytes-2*tq4_bytes(s.rows));
        target=usage.ram_bytes-saving;
    }
    if(usage.ram_bytes<=target)return;
    s.maintenance=s.worker->submit([&s,target,physical_budget,pressure]{
        check(cudaSetDevice(s.device));s.worker->release_readers();
        auto result=s.store->maintain(target,false);
        // A completed GPU publication must stop pinning its old images even
        // when the model has moved on to other layers. Retry only after
        // releasing completed leases; long-lived readers remain untouched.
        while(result.needs_readers && s.worker->release_readers())
            result=s.store->maintain(target,false);
        // Fresh blocks can arrive during optional format conversion. Only
        // the actual RAM limit permits SSD eviction or a capacity error.
        if(!pressure && s.store->stats().ram_bytes>physical_budget)result=s.store->maintain(physical_budget,false);
        const auto stats=s.store->stats();s.update_account();
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
            std::fprintf(stderr,"qsa_ram_pressure actual_match=1 budget_bytes=%zu ram_bytes=%zu pooled_pinned_bytes=%zu fp16_tokens=%llu affine8_tokens=%llu tq4_tokens=%llu ssd_tokens=%llu quantized=%llu spilled=%llu maintenance_budget=%zu\n",
                physical_budget,stats.ram_bytes,s.pool->account->physical.load(),static_cast<unsigned long long>(stats.tokens[0]),static_cast<unsigned long long>(stats.tokens[1]),
                static_cast<unsigned long long>(stats.tokens[2]),static_cast<unsigned long long>(stats.tokens[3]),
                static_cast<unsigned long long>(stats.quantized_blocks),static_cast<unsigned long long>(stats.spilled_blocks),target);
        }
        // Expert cache reclamation may synchronize/free pinned allocations;
        // hand the capacity request back to the model thread, outside this
        // background codec job and every store/pool/graph wait lock.
        if(stats.ram_bytes>physical_budget)
            s.required_capacity.store(stats.ram_bytes);
    },s.maintenance_sequence);
}
void QsaRamCache::wait(){
    auto& s=*impl_;s.wait();s.drain();
    bool retried_readers=false;
    while(s.store->stats().ram_bytes>s.budget()) {
        const auto logical=s.store->stats().ram_bytes;
        const auto physical=s.worker->pool_account->physical.load();
        s.required_capacity.store(logical);s.resolve_capacity();
        // Recompute pressure after publication releases its original images.
        // Logical eviction alone can leave the next pinned transfer without
        // physical pages; the resulting overhead changes the shared budget.
        check(cudaStreamSynchronize(mfq_current_cuda_stream()));s.drain();
        const bool progress=s.store->stats().ram_bytes<logical ||
            s.worker->pool_account->physical.load()<physical;
        if(!progress && retried_readers)break;
        // A completed publication can release a lease without releasing any
        // pages. Retry once, while external long-lived readers stay bounded.
        retried_readers=!progress;
    }
}
void QsaRamCache::truncate(std::size_t blocks){
    impl_->wait();impl_->store->truncate(blocks);impl_->update_account();impl_->drain();
    impl_->worker->budget();
}
bool QsaRamCache::graph_safe()const{return (!impl_->ssd_enabled || !impl_->maintenance.valid()) && impl_->store->stats().tokens[3]==0;}
bool QsaRamCache::encoding_urgent()const{
    if(impl_->foreground_codec.load())return true;
    const auto& worker=*impl_->worker;const auto job=worker.running_job.load();
    return job && job<=worker.urgent_through.load();
}
std::shared_ptr<tq::RamKvSnapshot> QsaRamCache::snapshot(std::size_t blocks) {
    wait();return impl_->store->snapshot(blocks);
}
std::shared_ptr<void> QsaRamCache::reserve_restore(const tq::RamKvSnapshot& state) {
    auto& s=*impl_;std::array<long double,3> precision{};
    if(state.blocks.size()>std::size_t(s.capacity))
        throw std::invalid_argument("QSA restore metadata exceeds cache capacity");
    // All layers reserve before any payload is hydrated. Include descriptor
    // pages and one policy batch of appends in the physical RAM sample. Exact
    // prefix capacity would force every layer to grow at its first new block.
    const auto incoming=state.blocks.size();
    const auto append=s.restore_append_blocks(incoming);
    const auto metadata_blocks=incoming+append;
    s.store->reserve_metadata(metadata_blocks);
    if(metadata_blocks>s.host_sources.capacity()) {
        const auto live=s.host_sources.size();
        s.host_sources.resize(metadata_blocks);s.host_sources.resize(live);
    }
    for(const auto& block:state.blocks)for(unsigned tier=0;tier<3;++tier)
        precision[tier]+=block.image && unsigned(block.tier)==tier?block.image->bytes():block.bytes[tier];
    const auto canonical=s.bits==16?tq::RamKvTier::Fp16:s.bits==8?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
    const auto target=s.target(tq::ram_precision_target(state.blocks.size(),precision,canonical));
    struct Reservation {
        std::shared_ptr<RamAccount> account;std::shared_ptr<RamWorker> worker;
        std::size_t previous;
        Reservation(std::shared_ptr<RamAccount> a,std::shared_ptr<RamWorker> w,std::size_t bytes)
            :account(std::move(a)),worker(std::move(w)),previous(account->restore_target.exchange(bytes)){}
        ~Reservation(){account->restore_target.store(previous);try{worker->budget();}catch(...){}}
    };
    return std::make_shared<Reservation>(s.account,s.worker,target);
}
void QsaRamCache::restore(const tq::RamKvSnapshot& state,bool borrow_prefix) {
    restore(state,borrow_prefix,{});
}
void QsaRamCache::restore(const tq::RamKvSnapshot& state,bool borrow_prefix,
                          const std::function<void(std::size_t)>& block_read) {
    auto& s=*impl_;s.wait();check(cudaStreamSynchronize(mfq_current_cuda_stream()));s.drain();
    if(s.gpu_cache.sets)check(tq::tq_reset_ram_stage_cache(s.gpu_cache,mfq_current_cuda_stream()));
    RamForegroundCodec foreground(s.foreground_codec);
    const auto written=s.store->stats().ssd_written_bytes;
    const auto reservation=reserve_restore(state);
    const auto physical_budget=s.budget();
    // Leave a bounded append transaction when cold blocks can remain on SSD.
    // Filling the last byte with restored payload otherwise forces a complete
    // codec batch as soon as the first new microblock arrives. This reservation
    // uses spare KV RAM and never requests expert-cache reclamation.
    const auto append=s.restore_append_blocks(state.blocks.size());
    const auto fresh_bytes=2*(s.bits==16?s.fp16_bytes:s.affine_bytes);
    const auto growth=append && fresh_bytes>SIZE_MAX/append?SIZE_MAX:append*fresh_bytes;
    const auto append_bytes=s.ssd_enabled?growth:0;
    const auto budget=physical_budget;
    try{s.store->restore(state,budget,borrow_prefix,false,block_read,
        [&](const auto& image,std::size_t remaining){return s.pool->allocate(image,&remaining);},append_bytes);
    }catch(...){s.update_account();throw;}
    s.update_account();s.drain();
    if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
        const auto stats=s.store->stats();
        std::fprintf(stderr,"qsa_ram_restore actual_match=1 borrow_prefix=%d ram_bytes=%zu budget_bytes=%zu target_bytes=%zu fp16_tokens=%llu affine8_tokens=%llu tq4_tokens=%llu ssd_tokens=%llu session_ssd_written_bytes=%llu\n",
            int(borrow_prefix),stats.ram_bytes,budget,s.account->restore_target.load(),
            static_cast<unsigned long long>(stats.tokens[0]),static_cast<unsigned long long>(stats.tokens[1]),
            static_cast<unsigned long long>(stats.tokens[2]),static_cast<unsigned long long>(stats.tokens[3]),
            static_cast<unsigned long long>(stats.ssd_written_bytes-written));
    }
}
tq::MicroblockImage QsaRamCache::reconstruction_image(std::size_t id){
    auto& s=*impl_;RamForegroundCodec foreground(s.foreground_codec);return copy(*s.store->read(id,s.budget(),false));
}
QsaRamMemory QsaRamCache::memory()const {
    const auto& s=*impl_;const auto stats=s.store->stats();
    return {s.worker->pool_account.get(),stats.ram_bytes,s.worker->pool_account->physical.load(),
        stats.resident_target_bytes,stats.session_disk_bytes,0,stats.tokens,s.ssd_enabled};
}
std::vector<QsaRamCache::Tensor> QsaRamCache::resident_tensors()const {
    const auto& s=*impl_;return {s.sources,s.cache_slots,s.cache_locks,s.cache_frame,s.cache_payload,s.cache_parameters};
}
tq::AdaptiveCacheStats QsaRamCache::stats()const {
    const auto a=impl_->store->stats();tq::AdaptiveCacheStats result;result.blocks=a.blocks;result.resident_bytes=a.ram_bytes;
    result.blocks_per_tier[5]=std::size_t(a.tokens[0]+a.tokens[1])/4;result.blocks_per_tier[0]=std::size_t(a.tokens[2])/4;
    result.demotions=a.quantized_blocks;result.promotions=a.promoted_blocks;result.canonical_read_bytes=a.ssd_read_bytes;
    return result;
}
} // namespace mfq::cuda
