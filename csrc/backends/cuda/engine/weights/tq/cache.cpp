#include "backends/cuda/engine/weights/tq/cache.h"

#include "backends/cuda/engine/weights/tq/image.h"
#include "formats/tq_table.h"
#include "backends/cuda/engine/executor/tq.h"
#include "backends/cuda/engine/executor/execution.h"
#include "env.h"
#include "weight_memory.h"
#include "backends/cpu/native/host_memory.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <unordered_set>

namespace tb = mfq_tensor_backend;

namespace {
std::size_t available_host_bytes(std::size_t process_limit) {
    try {
        const auto memory=mfq::sample_host_memory();
        auto available=memory.available_host_bytes();
        if(process_limit)available=std::min<std::uint64_t>(available,
            memory.resident_bytes>=process_limit?0:process_limit-memory.resident_bytes);
        return static_cast<std::size_t>(std::min<std::uint64_t>(available,SIZE_MAX));
    } catch(const std::runtime_error&) {return 0;}
}
}

struct TqCache::HostImage {
    mfq::TqLayout layout;
    std::vector<std::uint8_t> bytes;
    std::size_t source_bytes;
    bool pin_attempted=false,registered=false;
    int pin_device=0;
    mfq::cuda::HostRegistrationRetirement retirement;
    explicit HostImage(const std::vector<std::uint8_t>& wire)
        :layout(wire.data(),wire.size()),bytes(tq_image(wire,layout)),source_bytes(wire.size()) {
        retirement.owner=this;
        retirement.dispose=[](void* owner)noexcept{delete static_cast<HostImage*>(owner);};
    }
    HostImage(const HostImage&)=delete;
    HostImage& operator=(const HostImage&)=delete;
    static std::shared_ptr<HostImage> create(const std::vector<std::uint8_t>& wire) {
        return std::shared_ptr<HostImage>(new HostImage(wire),[](HostImage* image)noexcept {
            if(image->registered)mfq::cuda::retire_host_registration(&image->retirement);
            else delete image;
        });
    }
    // Pin only a reused host image. One-use SSD reads retain bounded
    // staging; repeated RAM transfers avoid another full CPU copy.
    void try_pin() {
        if(pin_attempted)return;
        pin_attempted=true;
        TYLOINFER_CUDA_CHECK(cudaGetDevice(&pin_device));
        const auto result=cudaHostRegister(bytes.data(),bytes.size(),cudaHostRegisterPortable);
        if(result==cudaSuccess){registered=true;return;}
        if(mfq::cuda::pinned_allocation_rejected(result) || result==cudaErrorNotSupported ||
           result==cudaErrorOperatingSystem || result==cudaErrorNotPermitted ||
           result==cudaErrorHostMemoryAlreadyRegistered) {
            (void)cudaGetLastError();return;
        }
        TYLOINFER_CUDA_CHECK(result);
    }
    ~HostImage()noexcept {
        if(registered)try {
            const mfq::cuda::DeviceGuard guard(pin_device);
            if(cudaHostUnregister(bytes.data())!=cudaSuccess)(void)cudaGetLastError();
        }catch(...) {}
    }
};

struct TqCache::Stage {
    mfq::cuda::HostBuffer host;
    mfq::cuda::Event uploaded;
    std::size_t cursor=0;
    bool pending=false;
    cudaStream_t stream=nullptr;
    explicit Stage(std::size_t bytes):host(bytes),uploaded(cudaEventDisableTiming) {
        mfq::cuda::charge_tensor_host_bytes(bytes);
    }
    ~Stage(){
        if(pending)try{uploaded.synchronize();}catch(...) {}
        mfq::cuda::tensor_host_bytes.fetch_sub(host.size());
    }
    void reuse(){if(pending){uploaded.synchronize();pending=false;}cursor=0;}
};

TqCache::HostTransfer::~HostTransfer() noexcept { try { ready->synchronize(); } catch (...) {} }

std::size_t TqCache::reserve_host_locked(std::size_t bytes) {
    if(!bytes || host_.empty())return 0;
    const auto available=available_host_bytes(process_host_limit_);
    return available>=bytes?0:host_.trim(bytes-available);
}

void TqCache::report()const {
    if(profile_)std::fprintf(stderr,"tq_cache calls=%llu gpu_hits=%llu host_hits=%llu source_bytes=%llu h2d_bytes=%llu fetch_ms=%.3f upload_submit_ms=%.3f gpu_bytes=%zu host_bytes=%zu host_limit=%zu direct_h2d_bytes=%llu pin_attempts=%llu pin_failures=%llu\n",
        static_cast<unsigned long long>(calls_),static_cast<unsigned long long>(gpu_hits_),
        static_cast<unsigned long long>(host_hits_),static_cast<unsigned long long>(source_bytes_),
        static_cast<unsigned long long>(transfer_bytes_),fetch_ms_,upload_submit_ms_,device_.used(),host_.used(),*host_.limit(),
        static_cast<unsigned long long>(direct_transfer_bytes_),static_cast<unsigned long long>(pin_attempts_),
        static_cast<unsigned long long>(pin_failures_));
}

void TqCache::upload(const tb::Tensor& destination,const std::vector<std::uint8_t>& bytes) {
    const auto started=std::chrono::steady_clock::now();
    auto& staging=stages_[destination.get_device()];
    auto* stage=staging.stages[staging.index].get();
    if(!stage || bytes.size()>stage->host.size()-stage->cursor) {
        staging.index^=1;
        auto& selected=staging.stages[staging.index];
        if(selected)selected->reuse();
        // Bound normal staging, while allowing any valid projection larger
        // than the normal chunk. This is a host DMA arena, not a GPU limit.
        const auto capacity=std::max<std::size_t>(std::size_t(32)<<20,bytes.size());
        if(!selected || selected->host.size()<capacity)selected=std::make_unique<Stage>(capacity);
        stage=selected.get();
    }
    auto* source=static_cast<std::uint8_t*>(stage->host.data())+stage->cursor;
    std::memcpy(source,bytes.data(),bytes.size());
    const auto stream=mfq_current_cuda_stream();
    // A staging arena can serve different request streams. Chain its
    // copies so the final reuse fence covers every outstanding DMA.
    if(stage->pending && stage->stream!=stream)
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,stage->uploaded.get(),0));
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(destination.data_ptr(),source,bytes.size(),cudaMemcpyHostToDevice,stream));
    stage->uploaded.record(stream);stage->pending=true;stage->stream=stream;
    stage->cursor+=bytes.size();transfer_bytes_+=bytes.size();
    upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
}

TqCache::TqCache(std::size_t budget,std::size_t host_budget,std::size_t process_limit,bool retain_all,
              std::shared_ptr<TqExec> tq)
    :tq_(std::move(tq)),
     host_(host_budget?host_budget:available_host_bytes(process_limit)/5*4),
     device_(retain_all?std::nullopt:std::optional<std::size_t>(budget)),
     process_host_limit_(process_limit),
     profile_(mfq::cuda::env_lookup("TYLOINFER_TQ_PROFILE_CACHE")!=nullptr) {
    if(!tq_)throw std::invalid_argument("TQ expert cache requires its execution pool");
}

TqCache::~TqCache(){
    host_transfers_.clear();
    report();
}

std::uint64_t TqCache::add(std::shared_ptr<const mfq::TqTable> table, std::int64_t rows) {
    if (!table || rows < 1 || table->width() < 1 || table->rows() % rows ||
        table->rows() / rows < 1 || table->rows() / rows > UINT32_MAX)
        throw std::invalid_argument("invalid TQ expert source geometry");
    std::lock_guard<std::mutex> guard(mutex_);
    const auto bytes = table->storage_bytes();
    if (bytes > SIZE_MAX - payload_bytes_)
        throw std::overflow_error("TQ source payload exceeds size limits");
    if (sources_.size() == UINT32_MAX)
        throw std::overflow_error("too many TQ projection sources");
    sources_.push_back({std::move(table), rows});
    payload_bytes_ += bytes;
    return sources_.size();
}

std::size_t TqCache::budget() {
    std::lock_guard<std::mutex> guard(mutex_);
    return device_.limit().value_or(std::max(payload_bytes_,device_.used()));
}

void TqCache::resize(std::size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);
    device_.limit(bytes);
}

std::size_t TqCache::reserve(std::size_t bytes) {
    std::lock_guard<std::mutex> guard(mutex_);return reserve_host_locked(bytes);
}

std::vector<std::pair<std::string,double>> TqCache::metrics() {
    std::lock_guard<std::mutex> guard(mutex_);
    std::size_t covered=0,staging=0;
    std::unordered_set<std::uint64_t> covered_images;
    covered_images.reserve(device_.entries().size()+host_.entries().size());
    for(const auto& [key,entry]:device_.entries())
        if(covered_images.insert(key.image).second)covered+=entry.value.source_bytes;
    for(const auto& [key,entry]:host_.entries())
        if(covered_images.insert(key).second)covered+=entry.value->source_bytes;
    for(const auto& [_,arenas]:stages_)for(const auto& stage:arenas.stages)
        if(stage)staging+=stage->host.size();
    const auto cold=payload_bytes_>covered?payload_bytes_-covered:0;
    const auto misses=calls_-gpu_hits_;
    return {{"resident_expert_cache_bytes",double(device_.used())},
        {"expert_cache_budget_bytes",double(device_.limit().value_or(std::max(payload_bytes_,device_.used())))},
        {"ram_expert_enabled",host_.used()?1.0:0.0},
        {"ram_expert_payload_bytes",double(host_.used())},
        {"ram_expert_cache_budget_bytes",double(*host_.limit())},
        {"ram_expert_pcie_read_bytes",double(transfer_bytes_)},
        {"ram_expert_direct_pcie_read_bytes",double(direct_transfer_bytes_)},
        {"ram_expert_cache_hits",double(host_hits_)},
        {"ram_expert_cache_misses",double(misses-host_hits_)},
        {"ram_expert_cache_evictions",double(host_.evictions())},
        {"ssd_expert_enabled",cold?1.0:0.0},
        {"ssd_expert_payload_bytes",double(cold)},
        {"ssd_expert_bytes_read",double(source_bytes_)},
        {"ssd_expert_read_count",double(source_reads_)},
        {"expert_logical_transfer_bytes",double(transfer_bytes_)},
        {"expert_host_fetch_ms",fetch_ms_},{"expert_host_copy_ms",upload_submit_ms_},
        {"expert_decode_transfer_bytes",double(staging)},
        {"expert_cache_hits",double(gpu_hits_)},
        {"expert_cache_misses",double(misses)},
        {"expert_cache_evictions",double(device_.evictions())},
        {"host_tensor_bytes",double(mfq::cuda::tensor_host_bytes.load())}};
}

tb::Tensor TqCache::views(std::uint64_t source,
        const std::int64_t* ids,const std::vector<TqWeight>& weights,int device) {
    const MfqCudaGuard device_guard(device);
    std::lock_guard<std::mutex> guard(mutex_);
    if (!source || source > sources_.size()) throw std::out_of_range("unknown TQ source");
    const auto& bank = sources_[source - 1];
    const auto experts = static_cast<std::int64_t>(bank.table->rows() / bank.rows);
    for (std::size_t i = 0; i < weights.size(); ++i)
        if (!ids || ids[i] < 0 || ids[i] >= experts)
            throw std::out_of_range("TQ view expert ID is out of bounds");
    const auto stream=mfq_current_cuda_stream();
    auto& tables=view_tables_[source];
    auto found=std::find_if(tables.begin(),tables.end(),[&](const ViewTable& table) {
        return table.device==device && table.stream==stream;
    });
    if(found==tables.end()) {
        auto views=tb::empty({experts*static_cast<std::int64_t>(sizeof(TrellisProjectionView))},
            tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        tables.push_back({device,stream,std::move(views),std::vector<std::uintptr_t>(experts),
            std::vector<std::uint8_t>(experts*sizeof(TrellisProjectionView))});
        found=std::prev(tables.end());
    }
    // Each stream has independent metadata, so a concurrent upload cannot
    // overwrite a view another request is reading. Tables own no expert
    // images; the caller's selected weights retain their normal lifetime.
    bool changed=false;
    for(std::size_t i=0;i<weights.size();++i) {
        const auto image=reinterpret_cast<std::uintptr_t>(weights[i].image.data_ptr());
        if(found->images[ids[i]]==image)continue;
        std::memcpy(found->bytes.data()+ids[i]*sizeof(TrellisProjectionView),&weights[i].view,sizeof(TrellisProjectionView));
        found->images[ids[i]]=image;changed=true;
    }
    // One small DMA for changed metadata, no per-expert transfers and no
    // host synchronization. Resident hits submit no metadata work at all.
    if(changed)try {upload(found->views,found->bytes);}
    catch(...) {std::fill(found->images.begin(),found->images.end(),0);throw;}
    return found->views;
}

std::vector<TqWeight> TqCache::get(std::uint64_t source,const std::int64_t* ids,std::size_t count,int device) {
    const MfqCudaGuard device_guard(device);
    std::lock_guard<std::mutex> guard(mutex_);
    if (!source || source > sources_.size()) throw std::out_of_range("unknown TQ source");
    const auto& bank = sources_[source - 1];
    const auto& table = *bank.table;
    const auto out = bank.rows;
    for (std::size_t i = 0; i < count; ++i)
        if (!ids || ids[i] < 0 || std::uint64_t(ids[i]) >= table.rows() / out)
            throw std::out_of_range("TQ routed expert ID is out of bounds");
    const auto resources=tq_->get(device);
    std::vector<TqWeight> result;result.reserve(count);
    const auto stream=mfq_get_current_cuda_stream(device);
    std::shared_ptr<mfq::cuda::Event> uploaded;
    for(auto it=host_transfers_.begin();it!=host_transfers_.end();)
        if(it->ready->ready())it=host_transfers_.erase(it);else ++it;
    auto transfer=host_transfers_.end();
    std::unordered_set<cudaEvent_t> waited;
    try {
    for(std::size_t index=0;index<count;++index) {
    const auto expert=static_cast<std::uint32_t>(ids[index]);
    const auto key=(source<<32)|expert;
    const DeviceKey device_key{key,device};
    ++calls_;
    if(const auto* resident=device_.get(device_key)) {
        if(resident->producer!=stream.stream()) {
            const auto event=resident->uploaded->get();
            if(waited.insert(event).second)TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),event,0));
            resources->retain(stream);
            mfq_cuda_record_stream(resident->weight.image,stream);
        }
        ++gpu_hits_;result.push_back(resident->weight);
        if((calls_&4095)==0)report();
        continue;
    }
    const auto fetch_started=std::chrono::steady_clock::now();
    std::shared_ptr<HostImage> image;
    if(const auto* cached=host_.get(key)) {
        image=*cached;++host_hits_;
        if(resources->host_register() && !image->pin_attempted) {
            if(auto registration=mfq::cuda::try_host_registration();registration.owns_lock()) {
                image->try_pin();++pin_attempts_;
                if(!image->registered)++pin_failures_;
            }
        }
    } else {
        auto wire=table.slice_rows(std::uint64_t(expert)*out,std::uint64_t(expert+1)*out);
        // Only source misses allocate new host images. Reclaim optional
        // images against current RAM/commit headroom before packing one;
        // GPU hits and ordinary RAM streaming do not poll process memory.
        reserve_host_locked(wire.size());
        const bool admit=available_host_bytes(process_host_limit_)>=wire.size();
        source_bytes_+=wire.size();++source_reads_;image=HostImage::create(wire);
        host_.reserve(image->bytes.size());
        if(admit)host_.put(key,image,image->bytes.size());
    }
    fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fetch_started).count();
    const auto needed=image->bytes.size();
    device_.reserve(needed);
    auto storage=tb::empty({static_cast<std::int64_t>(needed)},tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
    auto weight=tq_view(std::move(storage),image->layout,table.width());
    if(image->registered) {
        if(!uploaded)uploaded=std::make_shared<mfq::cuda::Event>(cudaEventDisableTiming);
        if(transfer==host_transfers_.end()) {
            host_transfers_.emplace_back();transfer=std::prev(host_transfers_.end());transfer->ready=uploaded;
        }
        // A host-cache eviction must not unpin a source still consumed
        // by asynchronous DMA. One event retains the whole call's images.
        transfer->images.push_back(image);
        const auto started=std::chrono::steady_clock::now();
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(weight.image.data_ptr(),image->bytes.data(),image->bytes.size(),
            cudaMemcpyHostToDevice,stream.stream()));
        transfer_bytes_+=image->bytes.size();direct_transfer_bytes_+=image->bytes.size();
        upload_submit_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }else upload(weight.image,image->bytes);
    if(!device_.limit() || needed<=*device_.limit()) {
        if(!uploaded)uploaded=std::make_shared<mfq::cuda::Event>(cudaEventDisableTiming);
        device_.put(device_key,DeviceEntry{weight,uploaded,stream.stream(),image->source_bytes},needed);
    }
    result.push_back(std::move(weight));
    if((calls_&4095)==0)report();
    }
    }catch(...) {
        // Publish readiness even when a later expert read/allocation
        // fails, before another request can observe the inserted entries.
        if(uploaded)uploaded->record(stream.stream());
        throw;
    }
    if(uploaded)uploaded->record(stream.stream());
    return result;
}
