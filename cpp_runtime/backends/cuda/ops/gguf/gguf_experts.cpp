#include "gguf_weight.h"
#include "gguf_rotation.h"
#include "gguf_cpu_executor.h"
#include "gguf_decode_dma.h"
#include "gguf_admission_executor.h"
#include "gguf_cpu_balance.h"
#include "gguf_adaptive_slots.h"
#include "gguf_route_trace.h"
#include "gguf_host_snapshot.h"
#include "gguf_prefill_stager.h"
#include "gguf_speculative_journal.h"
#include "quant_linear.h"
#include "cuda_execution.h"
#include "storage/moe_expert_cache.h"
#include "storage/moe_pinned_expert_arena.h"
#include "storage/pinned_copy_plan.h"
#include "strata/core/expert_cache.hpp"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/native_dma_batch.h"
#include "runtime/execution_options.h"
#include "tyloinfer/host_memory.h"
#include "tyloinfer/cpu_expert_pool.h"
#include "tyloinfer/runtime.h"
#include "tyloinfer/cpu_wait.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq_weight_memory.h"
#include "mfq_cuda_moe_ops.h"
#include "../../kernels/gguf.h"
#include "../../kernels/gguf_input.h"
#include "../../kernels/gguf_route_groups.h"
#include "../../kernels/gguf_mmq.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cuda.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <future>
#include <list>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace tb = mfq_tensor_backend;

struct GgufPrefillWeightStage {
    tb::Tensor image;
    std::size_t cursor=0;
    cudaEvent_t gate_up_ready=nullptr;
    bool gate_up_submitted=false;
    std::unordered_map<std::uint64_t,tb::Tensor> mmq_destinations;
};

// Fixed slots are replaced within their own layer. A mapped ready word lets
// that layer consume its new view after its copies, while earlier layers run.
// General LRU/aliased graph work retains whole-job synchronization.
struct GgufResidentView {
    std::array<std::uint64_t,3> sources,permanent{},strides{};
    std::int64_t experts;
    int device;
    std::vector<std::uint64_t> raw;
    std::vector<std::uint64_t> pending_ready;
    std::unique_ptr<mfq::cuda::HostBuffer> host;
    std::size_t admission_offset;
    void* mapped=nullptr;
    bool rebuild=false;
    GgufResidentView(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,int ordinal)
        :sources{banks[0]->source_id,banks[1]->source_id,banks[2]->source_id},experts(banks[0]->experts),device(ordinal),raw(3*std::size_t(experts),0),
         host(std::make_unique<mfq::cuda::HostBuffer>(((3*std::size_t(experts)*sizeof(std::uint64_t)+63)&~std::size_t(63))+64,true)),
         admission_offset(host->size()-64) {
        TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host->data(),0));
        std::memset(host->data(),0,host->size());
        mfq::cuda::publish_mapped_flag(admission_host());
        for(int role=0;role<3;++role)if(banks[role]->resident_packed.defined()) {
            if(banks[role]->resident_packed.get_device()!=device)throw std::invalid_argument("GGUF resident view device mismatch");
            permanent[role]=reinterpret_cast<std::uintptr_t>(banks[role]->resident_packed.data_ptr());
            strides[role]=banks[role]->expert_bytes;
        }
        mfq::cuda::charge_tensor_host_bytes(host->size()+raw.size()*sizeof(std::uint64_t));
    }
    ~GgufResidentView(){mfq::cuda::tensor_host_bytes.fetch_sub(host->size()+raw.size()*sizeof(std::uint64_t));}
    std::uint64_t* pointers(int role){return static_cast<std::uint64_t*>(host->data())+role*experts;}
    std::uint32_t* admission_host() const {
        return reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host->data())+admission_offset);
    }
    std::uint32_t* admission_device() const {
        return reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped)+admission_offset);
    }
    void update(int role,std::int64_t expert,std::uint64_t address) {
        raw[role*experts+expert]=address;
        const bool complete=raw[expert] && raw[experts+expert] && raw[2*experts+expert];
        for(int bank=0;bank<3;++bank)pointers(bank)[expert]=complete?raw[bank*experts+expert]:0;
    }
};

// The graph already owns this transfer arena. Preserve unchanged expert
// images between consecutive uses by the same bank; another layer, a resize
// or cancellation invalidates them. This adds no resident weight allocation.
struct GgufDirectWorkspaceImages {
    struct Slot {std::int64_t expert=-1;unsigned roles=0;};
    std::array<std::uint64_t,3> sources{UINT64_MAX,UINT64_MAX,UINT64_MAX};
    void* allocation=nullptr;
    std::size_t stride=0,bytes=0;
    std::array<std::size_t,3> offsets{};
    std::vector<Slot> slots;
    std::vector<std::size_t> selected_slots;
    void invalidate() {sources.fill(UINT64_MAX);slots.clear();selected_slots.clear();}
    void prepare(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            const std::vector<std::int64_t>& experts,void* workspace,std::size_t capacity) {
        std::size_t next_stride=0;
        for(int role=0;role<3;++role) {
            offsets[role]=next_stride;
            next_stride+=(banks[role]->expert_bytes+15)&~std::size_t(15);
        }
        const auto count=std::min(std::size_t(banks[0]->experts),capacity/next_stride);
        if(count<experts.size())throw std::logic_error("GGUF transfer reuse exceeds its graph workspace");
        const std::array ids{banks[0]->source_id,banks[1]->source_id,banks[2]->source_id};
        if(sources!=ids || allocation!=workspace || stride!=next_stride || bytes!=capacity) {
            sources=ids;allocation=workspace;stride=next_stride;bytes=capacity;
            slots.assign(count,Slot{});
        }
        selected_slots.clear();selected_slots.reserve(experts.size());
        for(const auto expert:experts) {
            auto found=std::find_if(slots.begin(),slots.end(),[&](const auto& slot){return slot.expert==expert;});
            if(found==slots.end()) {
                found=std::find_if(slots.begin(),slots.end(),[&](const auto& slot) {
                    return std::find(experts.begin(),experts.end(),slot.expert)==experts.end();
                });
                if(found==slots.end())throw std::logic_error("GGUF transfer reuse has no replaceable slot");
                *found={expert,0};
            }
            selected_slots.push_back(std::size_t(found-slots.begin()));
        }
    }
    void* destination(std::size_t selected,int role) const {
        return static_cast<std::uint8_t*>(allocation)+selected_slots.at(selected)*stride+offsets[role];
    }
    bool contains(std::size_t selected,int role) const {return slots[selected_slots.at(selected)].roles&(1u<<role);}
    void copied(std::size_t selected,int role) {slots[selected_slots.at(selected)].roles|=1u<<role;}
};

struct GgufLayerAdmission {
    int device;
    tb::Tensor ready_word;
    std::shared_ptr<mfq::cuda::Event> submitted_done;
    bool cancelled=false;
    std::exception_ptr error;
    std::vector<std::shared_ptr<GgufResidentView>> views;
};

// Windows on one consumer stream execute serially. Keep their transfer images
// in one high-water arena; an older, smaller allocation remains owned by any
// graph that already captured it. Image ownership follows the allocation, so
// changing windows cannot mistake another window's bytes for a cache hit.
struct GgufDirectArena {
    tb::Tensor storage;
    GgufDirectWorkspaceImages images;
};

// Cache the original GGUF blocks. Formats share capacity and eviction policy;
// only the projection kernel interprets their encoding.
class GgufExpertCache {
    struct HostImage {
        std::shared_ptr<MoePinnedExpertArena::Block> block;
        std::size_t block_offset=0;
        tb::Tensor fallback;
        std::shared_ptr<GgufHostSnapshot> snapshot;
        std::size_t snapshot_offset=0;
        std::size_t bytes;
        std::shared_ptr<mfq::cuda::Event> ready;
        cudaStream_t producer = nullptr;
        bool ready_on_host=false;
        void* data() const { return snapshot ? snapshot->data(snapshot_offset) :
            block ? static_cast<void*>(block->data()+block_offset) : fallback.data_ptr(); }
        bool pinned() const { return block || (snapshot && snapshot->registration(snapshot_offset,bytes)); }
        const void* registration_id() const { return block ? block->allocation_id() :
            snapshot ? snapshot->registration(snapshot_offset,bytes) : nullptr; }
        std::uint8_t* mapped_data(int device) const { return block ? block->mapped_data()+block_offset :
            snapshot->mapped_data(snapshot_offset,bytes,device); }
    };
    struct HostEntry {
        std::shared_ptr<HostImage> image;
        std::list<std::uint64_t>::iterator recent;
        std::optional<std::list<std::uint64_t>::iterator> duplicate;
    };
    struct Key {
        std::uint64_t image;
        int device;
        bool operator==(const Key&) const = default;
    };
    struct Hash {
        std::size_t operator()(const Key& key) const noexcept {
            return std::hash<std::uint64_t>{}(key.image) ^ (std::hash<int>{}(key.device) << 1);
        }
    };
    struct DeviceEntry {
        GgufWeight weight;
        std::shared_ptr<mfq::cuda::Event> ready;
        cudaStream_t producer;
        std::list<Key>::iterator recent;
        // This immutable image cannot become pending again. Replacing a slot
        // inserts a fresh entry with its new producer event and false here.
        bool ready_on_host=false;
    };
    struct Transfer {
        std::shared_ptr<HostImage> image;
        std::shared_ptr<mfq::cuda::Event> ready;
        tb::Tensor device_image;
    };
    struct PendingUpload { Key key; GgufWeight weight; std::shared_ptr<HostImage> source; bool retain; };
    struct UploadBatch { std::vector<PendingUpload> images; };
    enum class UploadPurpose {Streamed,CacheAdmission};
    struct UploadStage {
        std::unique_ptr<mfq::cuda::HostBuffer> host;
        tb::Tensor device;
        std::shared_ptr<mfq::cuda::Event> ready;
        int ordinal = -1;
    };
    static constexpr std::size_t upload_capacity = 16ULL << 20;
    std::size_t decode_workspace_reserve_=0;
    bool decode_workspace_prepared_=false;
    std::size_t decode_growth_reserve_=0;
    std::optional<std::size_t> decode_transient_reserve_;
    std::array<UploadStage, 2> upload_stages_;
    static constexpr std::size_t direct_capacity = 256ULL << 20;
    std::unique_ptr<mfq::cuda::HostBuffer> direct_host_;
    void* direct_mapped_ = nullptr;
    bool serving_graph_ = false;
    bool reuse_learning_ = true;
    std::size_t upload_cursor_ = 0;
    int upload_blocks_=0,upload_threads_=0;
    std::mutex mutex_;
    std::size_t host_limit_, host_used_ = 0, host_reserved_ = 0, physical_limit_;
    std::size_t pinned_ceiling_ = SIZE_MAX;
    std::shared_ptr<GgufHostSnapshot> snapshot_;
    std::unordered_map<std::uint64_t,GgufHostSnapshot::Bank> snapshot_sources_;
    struct PreloadedGate {std::uint64_t source;std::size_t bytes;std::int64_t experts;ggml_type type;};
    std::unordered_map<std::string,PreloadedGate> preloaded_gates_;
    std::uint64_t host_gate_up_pair_bytes_=0;
    std::size_t prefill_loan_required_=0,prefill_loan_last_=0,prefill_loan_retained_=0;
    std::uint64_t prefill_dma_batches_=0,prefill_dma_batched_ranges_=0,prefill_dma_overlap_batches_=0;
    std::uint64_t transfer_reap_images_=0,transfer_reap_queries_=0;
    std::uint64_t prefill_phased_groups_=0;
    std::size_t ordinary_host_limit_=0;
    std::unique_ptr<GgufPrefillStager> prefill_host_stager_;
    std::unordered_map<std::uint64_t,int> staged_host_jobs_;
    std::uint64_t prefill_host_staged_bytes_=0;
    double prefill_host_staging_wait_ms_=0;
    std::optional<std::size_t> configured_device_limit_;
    std::map<int, std::size_t> device_limit_, device_used_;
    std::uint64_t sources_ = 0;
    std::size_t payload_bytes_ = 0, predictor_resident_bytes_ = 0;
    std::uint64_t calls_ = 0, gpu_hits_ = 0, host_hits_ = 0, source_bytes_ = 0,
        source_reads_ = 0, transfer_bytes_ = 0, host_evictions_ = 0, device_evictions_ = 0,
        writeback_bytes_ = 0, pin_failures_ = 0, pin_capacity_misses_ = 0,
        batch_packets_ = 0, grouped_prefills_ = 0, source_batches_=0,
        immutable_recovery_bytes_=0;
    struct DirectTraffic {std::uint64_t layers=0,hits=0,misses=0,bytes=0;};
    std::map<int,DirectTraffic> direct_by_rows_;
    std::uint64_t direct_calls_ = 0, direct_bytes_ = 0, batch_mapped_bytes_=0,
        prefill_direct_mmq_bytes_=0,prefill_dma_bytes_=0,prefill_dma_copies_=0,prefill_dma_images_=0,prefill_staging_images_=0,prefill_staging_bytes_=0;
    double fetch_ms_ = 0, upload_enqueue_ms_ = 0, writeback_enqueue_ms_ = 0, allocation_ms_ = 0,
        batch_copy_ms_ = 0, batch_wait_ms_ = 0;
    struct GraphPool { MfqCudaStream stream; std::shared_ptr<mfq::cuda::Context> context; };
    std::map<cudaStream_t, GraphPool> graph_pools_;
    std::map<std::pair<int,cudaStream_t>,std::weak_ptr<GgufDirectArena>> direct_arenas_;
    std::vector<std::weak_ptr<GgufDirectArena>> direct_allocations_;
    std::uint64_t direct_arena_reuses_=0;
    std::vector<std::shared_ptr<MoePinnedExpertArena>> arenas_;
    std::unordered_map<std::uint64_t, HostEntry> host_;
    std::list<std::uint64_t> host_recent_;
    std::list<std::uint64_t> host_duplicates_;
    std::unordered_map<std::uint64_t,mfq::ModelSource::TensorReader> source_readers_;
    std::unordered_map<std::uint64_t, std::size_t> device_copies_;
    std::unordered_map<Key, DeviceEntry, Hash> device_;
    std::map<std::array<std::uint64_t,4>,std::shared_ptr<GgufResidentView>> resident_views_;
    struct ResidentRole {GgufResidentView* view;int role;};
    std::unordered_map<Key,std::vector<ResidentRole>,Hash> resident_roles_;
    std::map<int,std::unordered_set<std::uint64_t>> dirty_resident_images_;
    std::uint64_t resident_version_=1,resident_view_refreshes_=0;
    std::uint64_t resident_view_updates_=0;
    double resident_view_prepare_ms_=0;
    std::map<int,std::uint64_t> prepared_resident_version_;
    std::list<Key> device_recent_;
    std::vector<Transfer> transfers_;
    struct DirectDemand {
        std::array<std::weak_ptr<GgufRoutedWeight>,3> banks;
        std::unordered_map<std::int64_t,std::uint64_t> scores;
    };
    struct SpeculativeDemand {
        std::array<std::weak_ptr<GgufRoutedWeight>,3> banks;
        std::vector<GgufSpeculativeTouch> touches;
        // unordered_map keeps element pointers stable across rehashes. The
        // residency version rejects this snapshot before any erased entry can
        // be read; collection and resolution both hold the cache mutex.
        std::vector<std::array<DeviceEntry*,3>> entries;
        std::uint64_t resident_version=0;
        std::size_t rows=0;
        int device=0;
    };
    GgufSpeculativeJournal speculative_journal_;
    std::vector<GgufSpeculativeTouch> ordinary_touches_;
    std::map<std::uint64_t,SpeculativeDemand> speculative_demand_;
    int speculative_rows_=0;
    std::uint64_t speculative_committed_=0,speculative_discarded_=0;
    std::uint64_t speculative_route_visits_=0,speculative_unique_touches_=0;
    std::uint64_t speculative_snapshot_touches_=0,speculative_refind_touches_=0;
    std::map<std::uint64_t,DirectDemand> direct_demand_;
    bool direct_window_pending_=false;
    struct PrefillDemand {
        std::array<std::weak_ptr<GgufRoutedWeight>,3> banks;
        std::vector<std::uint64_t> counts;
    };
    std::map<std::uint64_t,PrefillDemand> prefill_demand_;
    bool prefill_rebalance_pending_=false;
    std::size_t prefill_resident_experts_=0;
    double prefill_rebalance_ms_=0;
    CudaProfiler& profiler_;
    std::map<int,MfqCudaStream> prefill_streams_;
    std::map<int,MfqCudaStream> admission_streams_;
    std::map<int,tb::Tensor> admission_ready_words_;
    // Joined before a new decode graph can wait on host-published routes.
    // Only post-window cache admission runs here; speculative rows never do.
    std::future<void> admission_job_;
    std::unique_ptr<GgufAdmissionExecutor> admission_executor_;
    std::shared_ptr<GgufLayerAdmission> layer_admission_;
    std::uint64_t layer_admission_epochs_=0,layer_admission_signals_=0,layer_admission_served_=0;
    std::uint64_t layer_admission_ready_updates_=0;
    double layer_admission_wait_ms_=0;
    std::uint64_t admission_dma_bytes_=0,admission_dma_copies_=0,admission_dma_images_=0;
    std::uint64_t async_admission_batches_=0;
    double admission_join_ms_=0;
    double admission_submission_join_ms_=0;
    double admission_selection_ms_=0;
    std::uint64_t adaptive_rounds_=0,adaptive_fast_rounds_=0,direct_workspace_reused_bytes_=0;
    std::unique_ptr<GgufCpuExecutor> cpu_executor_;
    std::map<int,std::shared_ptr<GgufDecodeDma>> decode_dma_;
    // Discover before a decode window temporarily pins the serving caller.
    // On Linux children inherit that affinity; rediscovering here would see
    // only the reserved host core rather than the process's worker cores.
    const std::vector<int> cpu_worker_cores_=[] {
        const auto* setting=std::getenv("TYLOINFER_GGUF_CPU_EXPERTS");
        if(!setting || (std::strcmp(setting,"auto")!=0 && std::atoi(setting)<=0) ||
                !mfq::cpu::gguf::available())return std::vector<int>{};
        auto cores=mfq::cpu::physical_cores(false);
        std::erase(cores,mfq::cpu::planned_host_core());
        return cores;
    }();
    using CpuBalanceKey=std::array<std::int64_t,12>;
    std::map<CpuBalanceKey,GgufCpuBalanceModel> cpu_balance_;
    std::uint64_t cpu_auto_layers_=0,cpu_auto_selected_=0,cpu_auto_gpu_only_=0,cpu_auto_samples_=0,cpu_auto_fitted_layers_=0;
    std::uint64_t cpu_clock_samples_=0,cpu_exposed_waits_=0;
    double cpu_exposed_wait_ms_=0,cpu_clock_adjust_ms_=0;
    static CpuBalanceKey cpu_balance_key(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            int device,int tokens,bool half) {
        return {device,tokens,int(half),int(banks[0]->type),int(banks[1]->type),int(banks[2]->type),
            banks[0]->input,banks[0]->out,banks[1]->input,banks[1]->out,banks[2]->input,banks[2]->out};
    }
    std::unique_ptr<mfq::cuda::gguf::RouteTrace> route_trace_;
    std::uint64_t cpu_experts_=0;
    double cpu_work_ms_=0;
    double cpu_queue_ms_=0;
    double cpu_queue_peak_ms_=0;
    std::uint64_t cpu_jobs_=0,cpu_queue_delayed_jobs_=0;
    std::uint64_t host_route_group_packets_=0,host_route_group_slots_=0,host_route_group_count_=0;
    std::array<double,6> host_prepare_ms_{};
    std::uint64_t host_prepare_calls_=0;
    double resident_lock_ms_=0,resident_events_ms_=0,resident_body_ms_=0;
    std::uint64_t prefill_inplace_mmq_bytes_=0;
    std::uint64_t host_device_ready_waits_=0,prefill_device_ready_skips_=0;
    std::uint64_t speculative_cache_observed_=0;
    std::vector<std::shared_ptr<mfq::cuda::Event>> deferred_events_;
    std::uint64_t deferred_event_releases_=0;
    struct Serving {
        bool& active;bool previous;
        explicit Serving(bool& value):active(value),previous(value){active=true;}
        ~Serving(){active=previous;}
    };
    struct SlotOwner {
        std::array<std::weak_ptr<GgufRoutedWeight>,3> banks;
        std::int64_t expert;
    };
    struct SlotRole {int layer,role;std::size_t offset;};
    struct SlotTier {
        std::shared_ptr<strata::core::ExpertCache> cache;
        std::unordered_map<std::uint64_t,SlotRole> roles;
        std::vector<SlotOwner> owners;
        std::vector<std::array<std::weak_ptr<GgufRoutedWeight>,3>> banks;
        // Cache-free source metadata may be used by a background selection
        // job without becoming the cache's final owner on that same worker.
        std::vector<std::array<std::shared_ptr<GgufRoutedWeight>,3>> admission_banks;
        std::vector<std::vector<float>> usage;
        std::uint64_t windows=0,last_decay=0;
        std::int64_t loan_first=0;
        bool loan_active=false;
    };
    std::map<int,SlotTier> slot_tiers_;
    // Weak references also see arenas retained by older tensor/graph views,
    // without extending any allocation's lifetime themselves.
    std::vector<std::weak_ptr<strata::core::ExpertCache>> slot_arenas_;
    std::size_t slot_touch_free_before_=0,slot_touch_free_after_=0,slot_touch_reserve_=0;
    std::uint64_t slot_touch_retries_=0;
    static bool strata_slots_requested() {
        const auto* setting=std::getenv("TYLOINFER_GGUF_STRATA_CACHE");
        return !setting || setting[0]!='0';
    }
    static std::size_t slot_part_bytes(std::size_t bytes) {
        if(bytes>std::size_t(INT64_MAX)-255)throw std::overflow_error("GGUF slot size overflows");
        return (bytes+255)&~std::size_t(255);
    }
    tb::Tensor slot_image(const GgufRoutedWeight& bank,std::int64_t expert,int device) const {
        const auto tier=slot_tiers_.find(device);if(tier==slot_tiers_.end())return {};
        const auto role=tier->second.roles.find(bank.source_id);if(role==tier->second.roles.end())return {};
        const auto slot=tier->second.cache->slot_of(role->second.layer,expert);
        if(slot<0 || slot>=tier->second.loan_first)return {};
        auto storage=std::make_shared<mfq::cuda::TensorStorage>();
        storage->owner=tier->second.cache;storage->base=tier->second.cache->device_slot(slot)+role->second.offset;
        storage->bytes=bank.expert_bytes;storage->device=mfq::cuda::Device{mfq::cuda::DeviceType::cuda,device};
        const std::array<std::int64_t,1> shape{std::int64_t(bank.expert_bytes)};
        return tb::Tensor(storage,mfq::cuda::make_contiguous_view(storage->base,shape,mfq::cuda::ScalarType::uint8,storage->device));
    }
    bool slot_admitted(const GgufRoutedWeight& bank,std::int64_t expert,int device) const {
        const auto tier=slot_tiers_.find(device);if(tier==slot_tiers_.end())return false;
        const auto role=tier->second.roles.find(bank.source_id);if(role==tier->second.roles.end())return false;
        const auto slot=tier->second.cache->slot_of(role->second.layer,expert);
        return slot>=0 && slot<tier->second.loan_first;
    }
    bool prepare_slot_expert(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            std::int64_t expert,int device,std::int64_t replace_expert=-1) {
        std::lock_guard lock(mutex_);
        const auto found=slot_tiers_.find(device);if(found==slot_tiers_.end())return true;
        auto& tier=found->second;
        const auto role=tier.roles.find(banks[0]->source_id);if(role==tier.roles.end())return false;
        if(slot_admitted(*banks[0],expert,device))return true;
        const auto victim=replace_expert<0?-1:tier.cache->slot_of(role->second.layer,replace_expert);
        if(victim<0 || victim>=tier.loan_first)return false;
        auto& owner=tier.owners[victim];const auto old_expert=owner.expert;
        // A fixed slot is replaced within its layer; these immutable bank
        // identifiers also describe the victim, without locking owner.banks.
        for(const auto& bank:banks) {
            const Key key{(bank->source_id<<32)|std::uint64_t(old_expert),device};
            const auto entry=device_.find(key);if(entry==device_.end())continue;
            demote(key,entry->second);device_recent_.erase(entry->second.recent);
            device_used_[device]-=entry->second.weight.packed.nbytes();remove_device_copy(key.image,key.device);
            device_.erase(entry);++device_evictions_;
        }
        tier.cache->replace(role->second.layer,std::int32_t(old_expert),std::int32_t(expert));
        owner.expert=expert;return true;
    }
    template<class Candidate> bool select_slot_swaps(std::vector<Candidate>& selected,int device,bool frequent) {
        const auto found=slot_tiers_.find(device);if(found==slot_tiers_.end())return false;
        auto& tier=found->second;
        // Ordinary decode keeps Strata's four-window cadence. Verification
        // can learn each window while selection/copies overlap the next draft.
        // Preserve the same heat lifetime when that cadence changes.
        if(tier.loan_active)return true;
        ++adaptive_rounds_;if(frequent)++adaptive_fast_rounds_;
        const auto swaps=mfq::cuda::gguf::adaptive_slot_swaps(tier.usage,[&](auto layer,auto expert) {
            return tier.cache->slot_of(layer,expert)>=0;
        });
        for(const auto& swap:swaps) {
            auto banks=tier.admission_banks[swap.layer];
            const auto bytes=banks[0]->expert_bytes+banks[1]->expert_bytes+banks[2]->expert_bytes;
            selected.push_back({std::move(banks),swap.in,0,bytes,swap.out});
        }
        const auto elapsed=tier.windows-tier.last_decay;
        const float decay=elapsed==4?.7f:std::pow(.7f,float(elapsed)/4.f);
        for(auto& layer:tier.usage)for(auto& count:layer)count*=decay;
        tier.last_decay=tier.windows;
        return true;
    }
    template<class Candidate> void initialize_slot_tier(std::vector<Candidate>& selected,int device) {
        if(selected.empty())return;
        // Strata's open_sized and admit preserve the profile's slot order.
        // Each slot holds three unchanged GGUF byte ranges, not a new format.
        SlotTier tier;tier.cache=std::make_shared<strata::core::ExpertCache>();
        std::int64_t max_experts=0;int layer=0;
        for(const auto& [source,demand]:prefill_demand_) {
            std::size_t offset=0;bool alive=true;
            for(int role=0;role<3;++role) {
                const auto bank=demand.banks[role].lock();if(!bank){alive=false;break;}
                tier.roles.emplace(bank->source_id,SlotRole{layer,role,offset});
                offset+=slot_part_bytes(bank->expert_bytes);max_experts=std::max(max_experts,bank->experts);
            }
            if(alive) {
                tier.banks.push_back(demand.banks);
                std::array<std::shared_ptr<GgufRoutedWeight>,3> metadata;
                for(int role=0;role<3;++role) {
                    metadata[role]=std::make_shared<GgufRoutedWeight>(*demand.banks[role].lock());
                    metadata[role]->cache.reset();
                }
                tier.admission_banks.push_back(std::move(metadata));
                tier.usage.emplace_back(std::size_t(demand.counts.size()),0.f);
                ++layer;
            }
        }
        // Retire the former separate allocations before opening the one arena.
        for(auto next=device_.begin();next!=device_.end();) {
            if(next->first.device!=device){++next;continue;}
            const auto key=next->first;demote(key,next->second);
            device_recent_.erase(next->second.recent);device_used_[device]-=next->second.weight.packed.nbytes();
            remove_device_copy(key.image,key.device);next=device_.erase(next);++device_evictions_;
        }
        const auto context=mfq::cuda::default_context(device);context->reclaim_idle_memory();reap();
        context->reclaim_idle_memory(); // Retire GPU owners released by the completed D2H transfers.
        slot_touch_free_before_=context->available_memory();
        const auto available=std::min(device_limit_.at(device),slot_touch_free_before_);
        std::vector<std::int64_t> sizes;std::size_t total=0,retained=0;
        for(auto& candidate:selected) {
            std::size_t bytes=0;for(const auto& bank:candidate.banks)bytes+=slot_part_bytes(bank->expert_bytes);
            if(bytes>available-std::min(total,available))continue;
            total+=bytes;sizes.push_back(std::int64_t(bytes));
            tier.owners.push_back({{candidate.banks[0],candidate.banks[1],candidate.banks[2]},candidate.expert});
            if(&selected[retained]!=&candidate)selected[retained]=std::move(candidate);
            ++retained;
        }
        selected.resize(retained);if(selected.empty())return;
        context->check_memory_limit(total);
        std::string error;
        // Match Strata's post-write residency check: an allocation accepted by
        // the driver can leave less physical room once all its pages are touched.
        // No tensor aliases exist yet, so a smaller arena can be reopened safely.
        slot_touch_reserve_=decode_workspace_reserve_;
        for(;;) {
            if(!tier.cache->open_sized(sizes,layer,max_experts,error))throw std::runtime_error(error);
            // ExpertCache's initial zero-fill is on CUDA's legacy stream.
            TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
            slot_touch_free_after_=context->available_memory();
            if(slot_touch_free_after_>=slot_touch_reserve_)break;
            const auto shortage=slot_touch_reserve_-slot_touch_free_after_;
            const auto target=total>shortage?total-shortage:0;
            tier.cache->close();
            do {total-=std::size_t(sizes.back());sizes.pop_back();}while(!sizes.empty() && total>target);
            selected.resize(sizes.size());tier.owners.resize(sizes.size());
            ++slot_touch_retries_;
            device_limit_[device]=std::min(device_limit_[device],total);
            context->reclaim_idle_memory();
            if(sizes.empty())return;
        }
        slot_arenas_.push_back(tier.cache);
        tier.loan_first=tier.cache->slots();
        for(std::size_t i=0;i<selected.size();++i) {
            const auto& candidate=selected[i];
            const auto role=tier.roles.at(candidate.banks[0]->source_id);
            if(tier.cache->admit(role.layer,candidate.expert)!=std::int32_t(i))
                throw std::logic_error("Strata GGUF slot order changed");
        }
        slot_tiers_.emplace(device,std::move(tier));
    }

    void release_slot_tier(int device) {
        const auto tier=slot_tiers_.find(device);if(tier==slot_tiers_.end())return;
        if(tier->second.loan_active)throw std::logic_error("Cannot release an active GGUF prefill loan");
        const MfqCudaGuard guard(device);
        for(auto entry=device_.begin();entry!=device_.end();) {
            if(entry->first.device!=device){++entry;continue;}
            const auto key=entry->first;demote(key,entry->second);
            device_recent_.erase(entry->second.recent);device_used_[device]-=entry->second.weight.packed.nbytes();
            remove_device_copy(key.image,key.device);entry=device_.erase(entry);++device_evictions_;
        }
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));reap();
        slot_tiers_.erase(tier);
        mfq::cuda::default_context(device)->reclaim_idle_memory();
        prefill_rebalance_pending_=!prefill_demand_.empty();
    }

    std::size_t reusable_resident_memory(int device) const {
#if defined(TYLOINFER_NATIVE_CUDA_RUNTIME) && CUDA_VERSION >= 11030
        for(const auto& [key,entry]:device_)if(key.device==device) {
            CUmemoryPool pool=nullptr;
            if(cuPointerGetAttribute(&pool,CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
                    CUdeviceptr(reinterpret_cast<std::uintptr_t>(entry.weight.packed.data_ptr())))!=CUDA_SUCCESS || !pool)return 0;
            std::uint64_t used=0,reserved=0;
            const auto native=reinterpret_cast<cudaMemPool_t>(pool);
            if(cudaMemPoolGetAttribute(native,cudaMemPoolAttrUsedMemCurrent,&used)!=cudaSuccess ||
                    cudaMemPoolGetAttribute(native,cudaMemPoolAttrReservedMemCurrent,&reserved)!=cudaSuccess) {
                (void)cudaGetLastError();return 0;
            }
            return reserved>used?reserved-used:0;
        }
#endif
        return 0;
    }

    std::size_t host_available() const {
        const auto sample = mfq::sample_host_memory();
        const auto available = sample.available_host_bytes();
        if (!physical_limit_) return available;
        return std::min<std::uint64_t>(available,
            physical_limit_ > sample.resident_bytes ? physical_limit_ - sample.resident_bytes : 0);
    }
    void reap() {
        const mfq::cuda::Event* previous=nullptr;bool complete=false;
        transfer_reap_images_+=transfers_.size();
        std::erase_if(transfers_, [this,&previous,&complete](const Transfer& transfer) {
            // One upload fence covers many consecutive immutable images.
            // A false result can safely keep the whole run until the next
            // reap; it must never cause a later image to retire earlier.
            const bool first=transfer.ready.get()!=previous;
            if(first) {
                previous=transfer.ready.get();complete=transfer.ready->ready();++transfer_reap_queries_;
            }
            if(!complete)return false;
            if(serving_graph_ || layer_admission_) {
                if(first)deferred_events_.push_back(transfer.ready);
                if(transfer.image && transfer.image->ready && transfer.image->ready!=transfer.ready)
                    deferred_events_.push_back(transfer.image->ready);
            }
            return true;
        });
    }
    void release_empty_arenas() {
        if (serving_graph_ || layer_admission_) return; // cudaFreeHost cannot run behind a waiting graph.
        std::erase_if(arenas_, [&](const auto& arena) {
            if (arena->used_bytes()) return false;
            host_reserved_ -= arena->bytes(); return true;
        });
    }
    std::size_t host_allocated() const {
        auto bytes = host_reserved_+(snapshot_?snapshot_->bytes:0);
        for (const auto& [key, entry] : host_) if (!entry.image->block && !entry.image->snapshot) bytes += entry.image->bytes;
        return bytes;
    }
    void evict_host() {
        // Spare RAM may keep upload sources, avoiding a return transfer when
        // an expert leaves VRAM. Under pressure discard those duplicates
        // before any expert whose sole resident copy is in RAM.
        const auto old = host_duplicates_.empty() ? host_recent_.back() : host_duplicates_.back();
        erase_host(old); ++host_evictions_;
    }
    std::size_t reclaim_host(std::size_t bytes) {
        reap(); const auto before = host_allocated();
        release_empty_arenas();
        for(auto it=host_recent_.end();before-host_allocated()<bytes && it!=host_recent_.begin();) {
            --it;const auto found=host_.find(*it);
            if(found->second.image->snapshot)continue;
            const auto key=*it;++it;erase_host(key);++host_evictions_;release_empty_arenas();
        }
        return before - host_allocated();
    }
    std::shared_ptr<MoePinnedExpertArena::Block> allocate_pinned(std::size_t bytes) {
        for (const auto& arena : arenas_)
            if (auto block = arena->allocate(bytes)) return block;
        // Strata prioritizes CUDA registration for experts absent from VRAM
        // (PinnedArena::ByNeed). A VRAM duplicate must not leave a cold expert
        // pageable merely because the OS/driver's pinning budget is full.
        for (auto next = host_duplicates_.end(); next != host_duplicates_.begin();) {
            const auto old = std::prev(next);
            const auto found = host_.find(*old);
            if (!found->second.image->block || found->second.image.use_count() != 1) { next = old; continue; }
            const auto key = *old;
            erase_host(key);++host_evictions_;
            for (const auto& arena : arenas_)
                if (auto block = arena->allocate(bytes)) return block;
        }
        if (serving_graph_ || layer_admission_) return {}; // Reuse intervals; never register memory while the GPU waits.
        const auto pinned_limit = std::min(host_limit_, pinned_ceiling_);
        if (bytes <= pinned_limit - std::min(pinned_limit, host_reserved_)) {
            const auto available = host_available();
            const auto capacity = std::min<std::size_t>(pinned_limit - host_reserved_,
                std::max<std::size_t>(bytes, std::min<std::size_t>(256ULL << 20, available / 8)));
            if (capacity >= bytes && capacity <= available) {
                try {
                    auto arena = std::make_shared<MoePinnedExpertArena>(capacity);
                    auto block = arena->allocate(bytes);
                    arenas_.push_back(std::move(arena)); host_reserved_ += capacity;
                    return block;
                } catch (const std::exception& error) {
                    if (!std::string(error.what()).starts_with("CUDA pinned allocation failed:")) throw;
                    ++pin_failures_;
                    pinned_ceiling_ = host_reserved_;
                    // Pageable source storage remains valid when the host
                    // cannot pin another region. Its transfer is still fenced.
                    (void)cudaGetLastError();
                }
            }
        }
        return {};
    }
    std::shared_ptr<HostImage> allocate_host(std::size_t bytes) {
        while (bytes > host_limit_ - std::min(host_used_, host_limit_) && !host_recent_.empty())
            evict_host();
        const auto available = host_available();
        if (available < bytes) reclaim_host(bytes - available);
        auto image = std::make_shared<HostImage>(); image->bytes = bytes;
        image->block = allocate_pinned(bytes);
        if (!image->block) {
            ++pin_capacity_misses_;
            image->fallback = tb::empty({static_cast<std::int64_t>(image->bytes)},
                tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
        }
        return image;
    }
    void insert_host(std::uint64_t key, const std::shared_ptr<HostImage>& image) {
        if (image->bytes <= host_limit_) {
            host_recent_.push_front(key);
            auto [found, inserted] = host_.emplace(key, HostEntry{image, host_recent_.begin(), {}});
            if (!inserted) throw std::logic_error("GGUF duplicate host cache insertion");
            host_used_ += image->bytes;
            if (device_copies_.count(key)) {
                host_duplicates_.push_front(key); found->second.duplicate = host_duplicates_.begin();
            }
        }
    }
    void erase_host(std::uint64_t key) {
        const auto found = host_.find(key);
        if (found == host_.end()) return;
        if(serving_graph_ && found->second.image->ready)deferred_events_.push_back(found->second.image->ready);
        host_used_ -= found->second.image->bytes;
        if (found->second.duplicate) host_duplicates_.erase(*found->second.duplicate);
        host_recent_.erase(found->second.recent); host_.erase(found);
    }
    void add_device_copy(std::uint64_t key,int device) {
        ++resident_version_;
        if(resident_roles_.count(Key{key>>32,device}))dirty_resident_images_[device].insert(key);
        ++device_copies_[key];
        if (auto found = host_.find(key); found != host_.end() && !found->second.duplicate) {
            host_duplicates_.push_front(key); found->second.duplicate = host_duplicates_.begin();
        }
    }
    void remove_device_copy(std::uint64_t key,int device) {
        ++resident_version_;
        if(resident_roles_.count(Key{key>>32,device}))dirty_resident_images_[device].insert(key);
        const auto copy = device_copies_.find(key);
        if (copy == device_copies_.end()) throw std::logic_error("GGUF device cache accounting differs");
        if (--copy->second) return;
        device_copies_.erase(copy);
        if (auto found = host_.find(key); found != host_.end() && found->second.duplicate) {
            host_duplicates_.erase(*found->second.duplicate); found->second.duplicate.reset();
        }
    }
    std::shared_ptr<HostImage> read_host(const GgufRoutedWeight& bank, std::int64_t expert) {
        const auto key = (bank.source_id << 32) | std::uint64_t(expert);
        if (auto found = host_.find(key); found != host_.end()) {
            host_recent_.splice(host_recent_.begin(), host_recent_, found->second.recent);
            if (found->second.duplicate)
                host_duplicates_.splice(host_duplicates_.begin(), host_duplicates_, *found->second.duplicate);
            ++host_hits_;
            auto image = found->second.image;
            const bool available=image->ready_on_host || !image->ready || image->ready->ready();
            if(available)image->ready_on_host=true;
            // Upgrade an earlier pageable fallback when hot VRAM duplicates
            // have released pinning capacity. Never mutate an in-flight image.
            if (!image->block && !image->snapshot && available) {
                if (auto block = allocate_pinned(image->bytes)) {
                    auto pinned = std::make_shared<HostImage>(); pinned->bytes = image->bytes;
                    pinned->block = std::move(block); std::memcpy(pinned->data(), image->data(), image->bytes);
                    if(serving_graph_ && image->ready)deferred_events_.push_back(image->ready);
                    found->second.image = pinned; image = std::move(pinned);
                }
            }
            if (serving_graph_) {
                if(!available){image->ready->synchronize();image->ready_on_host=true;}
            } else if (image->ready && !image->ready_on_host && image->producer != mfq_current_cuda_stream())
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(mfq_current_cuda_stream(), image->ready->get(), 0));
            return image;
        }
        if(snapshot_)if(const auto found=snapshot_sources_.find(bank.source_id);found!=snapshot_sources_.end()) {
            auto image=std::make_shared<HostImage>();image->snapshot=snapshot_;image->bytes=bank.expert_bytes;
            image->snapshot_offset=found->second.expert_offset(std::size_t(expert));
            insert_host(key,image);++host_hits_;return image;
        }
        auto image = allocate_host(bank.expert_bytes);
        bank.reader(std::uint64_t(expert) * bank.expert_bytes,
            static_cast<std::byte*>(image->data()), image->bytes);
        source_bytes_ += image->bytes; ++source_reads_;
        insert_host(key, image);
        return image;
    }
    void read_host_batch(const GgufRoutedWeight& bank,const std::vector<std::int64_t>& experts,int device) {
        if((snapshot_ && snapshot_sources_.count(bank.source_id)) || !bank.batch_reader || experts.size()<2)return;
        std::vector<std::int64_t> missing;
        std::unordered_set<std::int64_t> unique;
        for(const auto expert:experts) {
            if(expert<0 || expert>=bank.experts)throw std::out_of_range("GGUF expert id");
            const Key key{(bank.source_id<<32)|std::uint64_t(expert),device};
            if(!host_.count(key.image) && !device_.count(key) && unique.insert(expert).second)missing.push_back(expert);
        }
        if(missing.size()<2 || missing.size()>host_limit_/bank.expert_bytes)return;
        std::vector<std::pair<std::uint64_t,std::shared_ptr<HostImage>>> images;
        std::vector<mfq::ReadSpan> spans;
        images.reserve(missing.size());spans.reserve(missing.size());
        try {
            for(const auto expert:missing) {
                const auto key=(bank.source_id<<32)|std::uint64_t(expert);
                auto image=allocate_host(bank.expert_bytes);
                images.emplace_back(key,image);
                insert_host(key,image);
                spans.push_back({std::uint64_t(expert)*bank.expert_bytes,
                    static_cast<std::byte*>(image->data()),image->bytes});
            }
            bank.batch_reader(spans);
            source_bytes_+=missing.size()*bank.expert_bytes;source_reads_+=missing.size();++source_batches_;
        } catch(...) {
            // No partially read image may be used after an I/O failure. The
            // mutex keeps these placeholders private until the batch succeeds.
            for(const auto& [key,image]:images)
                if(auto found=host_.find(key);found!=host_.end() && found->second.image==image)erase_host(key);
            throw;
        }
    }
    void demote(const Key& key, const DeviceEntry& entry) {
        if (host_.count(key.image) || (snapshot_ && snapshot_sources_.count(key.image>>32)) ||
                entry.weight.packed.nbytes() > host_limit_) return;
        const auto image = allocate_host(entry.weight.packed.nbytes());
        if(!image->block)if(const auto source=source_readers_.find(key.image>>32);source!=source_readers_.end()) {
            // Pageable D2H can block the host until earlier compute on the
            // stream ends. Experts are immutable: a bounded source mapping
            // can restore these same bytes without a device synchronization.
            const auto expert=std::uint32_t(key.image);
            source->second(std::uint64_t(expert)*image->bytes,static_cast<std::byte*>(image->data()),image->bytes);
            source_bytes_+=image->bytes;++source_reads_;immutable_recovery_bytes_+=image->bytes;
            insert_host(key.image,image);return;
        }
        const auto stream = mfq_current_cuda_stream();
        if (entry.producer != stream) TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream, entry.ready->get(), 0));
        image->ready = std::make_shared<mfq::cuda::Event>(); image->producer = stream;
        transfers_.push_back({image, image->ready, entry.weight.packed});
        try {
            entry.weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            const auto before = std::chrono::steady_clock::now();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(image->data(), entry.weight.packed.data_ptr(), image->bytes,
                cudaMemcpyDeviceToHost, stream));
            writeback_enqueue_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-before).count();
            image->ready->record(stream);
            writeback_bytes_ += image->bytes;
        } catch (...) { (void)cudaStreamSynchronize(stream); transfers_.pop_back(); throw; }
        insert_host(key.image, image);
    }
    void evict_device(int device, std::size_t bytes) {
        const auto limit = device_limit_.at(device);
        while (device_used_[device] > limit || bytes > limit - device_used_[device]) {
            auto old = std::find_if(device_recent_.rbegin(), device_recent_.rend(),
                [device](const Key& key) { return key.device == device; });
            if (old == device_recent_.rend()) break;
            const auto key = *old;
            demote(key, device_.at(key));
            device_recent_.erase(std::next(old).base());
            device_used_[device] -= device_.at(key).weight.packed.nbytes(); device_.erase(key);
            remove_device_copy(key.image,key.device);
            ++device_evictions_;
        }
    }
public:
    std::size_t lendable_bytes() {
        std::lock_guard lock(mutex_);
        const auto tier=slot_tiers_.find(mfq_current_cuda_device());
        return tier==slot_tiers_.end()?0:std::size_t(tier->second.cache->bytes());
    }
    std::size_t begin_prefill_loan(std::size_t required) {
        wait_direct_admission();
        const auto device=mfq_current_cuda_device();const auto context=mfq::cuda::default_context(device);
        std::lock_guard lock(mutex_);
        prefill_loan_required_=required;prefill_loan_last_=0;prefill_loan_retained_=0;
        const auto found=slot_tiers_.find(device);if(found==slot_tiers_.end())return 0;
        auto& tier=found->second;
        if(tier.loan_active)throw std::logic_error("GGUF prefill workspace already borrowed");
        context->reclaim_idle_memory();reap();
        const auto available=context->available_memory();if(available>=required)return 0;
        const auto needed=required-available,capacity=std::size_t(tier.cache->bytes());
        if(needed>capacity)return 0;
        const auto first=tier.cache->slots_within(std::int64_t(capacity-needed));
        const auto offset=std::size_t(tier.cache->bytes_of(first));
        for(auto next=device_.begin();next!=device_.end();) {
            const auto role=tier.roles.find(next->first.image>>32);
            const auto slot=role==tier.roles.end()?-1:tier.cache->slot_of(role->second.layer,std::uint32_t(next->first.image));
            if(next->first.device!=device || slot<first){++next;continue;}
            const auto key=next->first;demote(key,next->second);
            device_recent_.erase(next->second.recent);device_used_[device]-=next->second.weight.packed.nbytes();
            remove_device_copy(key.image,key.device);next=device_.erase(next);
        }
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));reap();
        tier.loan_first=first;tier.loan_active=true;
        try {
            context->begin_workspace_loan({{tier.cache->device_slot(0)+offset,capacity-offset}},mfq_current_cuda_stream());
        } catch(...) {tier.loan_first=tier.cache->slots();tier.loan_active=false;throw;}
        prefill_loan_last_=capacity-offset;
        for(const auto& [key,entry]:device_)if(key.device==device)prefill_loan_retained_+=entry.weight.packed.nbytes();
        return capacity-offset;
    }
    void suspend_prefill_loan() {mfq::cuda::default_context(mfq_current_cuda_device())->suspend_workspace_loan();}
    void end_prefill_loan() {
        wait_direct_admission();const auto device=mfq_current_cuda_device();
        std::vector<SlotOwner> restore;
        {
            std::lock_guard lock(mutex_);
            const auto found=slot_tiers_.find(device);if(found==slot_tiers_.end() || !found->second.loan_active)return;
            auto& tier=found->second;
            // The last compressed PP arenas are retained by in-flight upload
            // owners. Retire them before checking the loan's live tensors.
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));reap();
            mfq::cuda::default_context(device)->end_workspace_loan();
            restore.assign(tier.owners.begin()+tier.loan_first,tier.owners.end());
            tier.loan_first=tier.cache->slots();tier.loan_active=false;
        }
        // As in Strata, this refill follows computation_finished and remains
        // part of TTFT. It never changes the PP or stable-decode timer.
        for(const auto& owner:restore) {
            std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
            for(int role=0;role<3;++role) {
                banks[role]=owner.banks[role].lock();
                if(!banks[role])throw std::runtime_error("GGUF borrowed expert owner expired");
            }
            (void)acquire_moe(banks,{owner.expert},device);
        }
    }
    void wait_direct_admission() {
        if(!admission_job_.valid() && !layer_admission_)return;
        const auto before=std::chrono::steady_clock::now();
        try {
            if(admission_job_.valid())admission_job_.get();
            if(layer_admission_ && layer_admission_->submitted_done)layer_admission_->submitted_done->synchronize();
        }
        catch(...) {
            std::lock_guard lock(mutex_);
            admission_join_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
            layer_admission_.reset();
            throw;
        }
        std::lock_guard lock(mutex_);
        admission_join_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
        if(layer_admission_)for(const auto& view:layer_admission_->views) {
            for(const auto image:view->pending_ready)
                if(auto entry=device_.find(Key{image,view->device});entry!=device_.end()) {
                    entry->second.ready_on_host=true;++layer_admission_ready_updates_;
                }
            view->pending_ready.clear();
        }
        layer_admission_.reset();
    }
    void prepare_admission_window(bool immutable) {
        bool submitted_only=false;
        {
            std::lock_guard lock(mutex_);
            if(immutable && layer_admission_ && !layer_admission_->cancelled) {
                if(layer_admission_->error)std::rethrow_exception(layer_admission_->error);
                submitted_only=true;
            }
        }
        if(submitted_only) {
            if(!admission_job_.valid())return;
            // Freeze cache metadata and finish enqueueing copies before the
            // next graph. Waiting for DMA remains layer-local; the planner
            // must not contend with a worker still preparing later banks.
            const auto before=std::chrono::steady_clock::now();
            std::exception_ptr error;
            try{if(admission_job_.valid())admission_job_.get();}catch(...){error=std::current_exception();}
            {
                std::lock_guard lock(mutex_);
                admission_submission_join_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
            }
            if(error) {
                try{wait_direct_admission();}catch(...){}
                std::rethrow_exception(error);
            }
            return;
        }
        wait_direct_admission();
    }
    std::shared_ptr<GgufLayerAdmission> begin_layer_admission(int device) {
        const auto* setting=std::getenv("TYLOINFER_GGUF_LAYER_ADMISSION");
        if(setting && setting[0]=='0')return {};
        const auto* dma=std::getenv("TYLOINFER_GGUF_ADMISSION_DMA");
        if(dma && dma[0]=='0')return {}; // SM-scatter admission cannot precede a waiting graph's compute queue.
        // Complete older pointer journals before starting this epoch. Only a
        // fixed, same-layer replacement tier permits concurrent graph readers.
        prepare_resident_views(device,mfq_current_cuda_stream());
        auto epoch=std::make_shared<GgufLayerAdmission>();epoch->device=device;
        epoch->submitted_done=std::make_shared<mfq::cuda::Event>();
        std::lock_guard lock(mutex_);
        const auto tier=slot_tiers_.find(device);
        if(tier==slot_tiers_.end() || tier->second.loan_active)return {};
        for(const auto& [key,view]:resident_views_) {
            if(view->device!=device)continue;
            int layer=-1,matches=0;
            for(int role=0;role<3;++role)if(const auto found=tier->second.roles.find(view->sources[role]);found!=tier->second.roles.end()) {
                if(layer<0)layer=found->second.layer;
                if(found->second.layer!=layer || found->second.role!=role)return {};
                ++matches;
            }
            // A view aliasing only some of a layer's banks needs the existing
            // whole-job barrier; no layer-local ownership proof is available.
            if(matches && matches!=3)return {};
            if(matches)epoch->views.push_back(view);
        }
        if(epoch->views.empty())return {};
        auto& ready_word=admission_ready_words_[device];
        if(!ready_word.defined()) {
            ready_word=tb::full({1},1,tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kInt32));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            const auto stream=admission_streams_.at(device).stream();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(epoch->views.front()->admission_host(),ready_word.data_ptr(),
                sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        epoch->ready_word=ready_word;
        for(const auto& view:epoch->views)
            *reinterpret_cast<volatile std::uint32_t*>(view->admission_host())=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        layer_admission_=epoch;++layer_admission_epochs_;
        return epoch;
    }
    void abort_layer_admission(const std::shared_ptr<GgufLayerAdmission>& epoch,std::exception_ptr error={}) {
        if(!epoch)return;
        std::lock_guard lock(mutex_);
        if(error)epoch->error=std::move(error);
        if(epoch->cancelled)return;
        epoch->cancelled=true;
        // Never release a waiting GPU onto a partially overwritten slot.
        // An aborted window sees zero hits; its normal cancellation clears the
        // streamed-copy descriptors too. A later window rebuilds the journal.
        for(const auto& view:epoch->views) {
            std::memset(view->host->data(),0,view->raw.size()*sizeof(std::uint64_t));
            view->rebuild=true;
        }
        ++resident_version_;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for(const auto& view:epoch->views)mfq::cuda::publish_mapped_flag(view->admission_host());
    }
    void release_unchanged_layers(const std::shared_ptr<GgufLayerAdmission>& epoch,
            const std::unordered_set<std::uint64_t>& pending) {
        if(!epoch)return;
        std::lock_guard lock(mutex_);
        if(epoch->cancelled)return;
        for(const auto& view:epoch->views)if(!pending.count(view->sources[0]))
            mfq::cuda::publish_mapped_flag(view->admission_host());
    }
    void publish_admission_layer(const std::shared_ptr<GgufLayerAdmission>& epoch,
            const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,cudaStream_t stream) {
        if(!epoch)return;
        std::vector<std::uint32_t*> ready;
        {
            std::lock_guard lock(mutex_);
            if(epoch->cancelled)return;
            auto& dirty=dirty_resident_images_[epoch->device];
            for(const auto& view:epoch->views)if(view->sources[0]==banks[0]->source_id) {
                for(auto it=dirty.begin();it!=dirty.end();) {
                    const auto image=*it;
                    const auto role=std::find(view->sources.begin(),view->sources.end(),image>>32);
                    if(role==view->sources.end()){++it;continue;}
                    std::uint64_t pointer=0;
                    if(const auto entry=device_.find(Key{image,epoch->device});entry!=device_.end())
                        pointer=reinterpret_cast<std::uintptr_t>(entry->second.weight.packed.data_ptr());
                    if(pointer)view->pending_ready.push_back(image);
                    view->update(int(role-view->sources.begin()),std::uint32_t(image),pointer);
                    it=dirty.erase(it);++resident_view_updates_;
                }
                ready.push_back(view->admission_host());
            }
            layer_admission_signals_+=ready.size();
            std::atomic_thread_fence(std::memory_order_seq_cst);
        }
        // The table is published now, but the graph cannot read it until these
        // stream-ordered signals follow all three banks' DMA/event records.
        // A copy-engine landing word can progress while the serving graph
        // waits. A separate signal kernel can be serialized behind that graph
        // by WDDM's compute scheduling, creating a circular wait.
        for(auto* flag:ready)TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(flag,epoch->ready_word.data_ptr(),
            sizeof(std::uint32_t),cudaMemcpyDeviceToHost,stream));
        // Submit landing signals before the host prepares a later layer.
        // This is also the flush used by Strata for WDDM's command batching.
        const auto progress=cudaStreamQuery(stream);
        if(progress!=cudaSuccess && progress!=cudaErrorNotReady)TYLOINFER_CUDA_CHECK(progress);
    }
    void wait_layer_admission(const std::shared_ptr<GgufResidentView>& view) {
        if(!view)return;
        std::shared_ptr<GgufLayerAdmission> epoch;
        {
            std::lock_guard lock(mutex_);epoch=layer_admission_;
            if(!epoch || epoch->device!=view->device)return;
        }
        const auto started=std::chrono::steady_clock::now();
        const auto* ready=reinterpret_cast<const volatile std::uint32_t*>(view->admission_host());
        std::uint32_t spins=0;
        while(!*ready) {
            mfq::cpu::spin_pause();
            if((++spins&1023u)!=0)continue;
            if(std::chrono::steady_clock::now()-started>std::chrono::seconds(20)) {
                auto error=std::make_exception_ptr(std::runtime_error("GGUF layer admission timed out"));
                abort_layer_admission(epoch,error);std::rethrow_exception(error);
            }
        }
        std::atomic_thread_fence(std::memory_order_acquire);
        std::lock_guard lock(mutex_);
        if(epoch->error)std::rethrow_exception(epoch->error);
        if(epoch->cancelled)throw std::runtime_error("GGUF layer admission cancelled");
        // The mapped signal proves these per-layer events have completed.
        // Avoid querying each new image's event again in the host plan.
        for(const auto image:view->pending_ready)
            if(auto entry=device_.find(Key{image,view->device});entry!=device_.end()) {
                entry->second.ready_on_host=true;++layer_admission_ready_updates_;
            }
        view->pending_ready.clear();
        ++layer_admission_served_;
        layer_admission_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    void prepare_device(int device) {
        wait_direct_admission();
        const MfqCudaGuard guard(device);
        std::lock_guard lock(mutex_);
        if(snapshot_)snapshot_->prepare_device(device);
        if(device_limit_.count(device))return;
        std::size_t available=0,total=0;
        TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available,&total));
        const auto reserve=mfq::cuda::runtime_options::workspace_reserve_bytes();
        device_limit_[device]=configured_device_limit_.value_or(available>reserve?available-reserve:0);
    }
    std::shared_ptr<GgufDecodeDma> decode_dma(int device) {
        const MfqCudaGuard guard(device);
        std::lock_guard lock(mutex_);
        auto& lane=decode_dma_[device];
        if(!lane)lane=std::make_shared<GgufDecodeDma>(device);
        return lane;
    }
    MfqCudaStream prefill_transfer_stream(int device) {
        std::lock_guard lock(mutex_);
        auto [found,inserted]=prefill_streams_.try_emplace(device);
        if(inserted)found->second=mfq_get_stream_from_pool(false,device);
        return found->second;
    }
    MfqCudaStream admission_transfer_stream(int device) {
        std::lock_guard lock(mutex_);
        auto [found,inserted]=admission_streams_.try_emplace(device);
        if(inserted)found->second=mfq_get_stream_from_pool(false,device);
        return found->second;
    }
    std::shared_ptr<GgufDirectArena> direct_arena(int device,cudaStream_t stream,std::size_t bytes) {
        std::lock_guard lock(mutex_);
        auto& slot=direct_arenas_[{device,stream}];
        if(auto arena=slot.lock();arena && arena->storage.nbytes()>=bytes) {
            ++direct_arena_reuses_;return arena;
        }
        const mfq::cuda::ResidentAllocationScope resident;
        auto arena=std::make_shared<GgufDirectArena>();
        arena->storage=tb::empty({std::int64_t(bytes)},
            tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        direct_allocations_.erase(std::remove_if(direct_allocations_.begin(),direct_allocations_.end(),
            [](const auto& allocation){return allocation.expired();}),direct_allocations_.end());
        direct_allocations_.push_back(arena);slot=arena;return arena;
    }
    tb::Tensor upload_packet(tb::Tensor host, int device) {
        const MfqCudaGuard guard(device);
        std::lock_guard lock(mutex_); reap();
        auto packet = tb::empty(host.sizes(), host.options().device(tb::Device(tb::kCUDA, device)));
        auto image = std::make_shared<HostImage>(); image->fallback = std::move(host); image->bytes = image->fallback.nbytes();
        auto ready = std::make_shared<mfq::cuda::Event>();
        transfers_.push_back({image, ready, {}});
        try {
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(packet.data_ptr(), image->data(), image->bytes,
                cudaMemcpyHostToDevice, mfq_current_cuda_stream()));
            ready->record(mfq_current_cuda_stream());
        } catch (...) { (void)cudaStreamSynchronize(mfq_current_cuda_stream()); transfers_.pop_back(); throw; }
        packet.record_stream(reinterpret_cast<std::uintptr_t>(mfq_current_cuda_stream()));
        return packet;
    }
    template<typename Fn> auto profile_prefill(const std::string& name,Fn&& fn) {
        return profiler_.measure(name,std::forward<Fn>(fn));
    }
    explicit GgufExpertCache(CudaExecutionContext& execution,const mfq::ModelSource& source)
        : physical_limit_(execution.config.moe_host_physical_bytes),
          configured_device_limit_(execution.config.moe_device_cache_bytes),profiler_(execution.profiler) {
        route_trace_=mfq::cuda::gguf::RouteTrace::from_environment();
        try {
            // Keep a small transfer ring available before bulk weight pinning
            // reaches the OS/driver limit. Every format shares these buffers.
            for (auto& stage : upload_stages_) {
                stage.host = std::make_unique<mfq::cuda::HostBuffer>(upload_capacity);
                mfq::cuda::charge_tensor_host_bytes(upload_capacity);
            }
        } catch (const std::exception& error) {
            if (!std::string(error.what()).starts_with("CUDA pinned allocation failed:")) throw;
            for (auto& stage : upload_stages_) if (stage.host) {
                stage.host.reset(); mfq::cuda::tensor_host_bytes.fetch_sub(upload_capacity);
            }
        }
        int can_map=0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,mfq_current_cuda_device()));
        const auto* direct=std::getenv("TYLOINFER_GGUF_DIRECT_EXPERTS");
        if (can_map && (!direct || direct[0]!='0')) {
            try {
                direct_host_=std::make_unique<mfq::cuda::HostBuffer>(direct_capacity,true);
                TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&direct_mapped_,direct_host_->data(),0));
                mfq::cuda::charge_tensor_host_bytes(direct_capacity);
            } catch (const std::exception& error) {
                if(!std::string(error.what()).starts_with("CUDA pinned allocation failed:"))throw;
                direct_host_.reset();direct_mapped_=nullptr;
            }
        }
        host_limit_ = execution.config.moe_host_cache_bytes;
        TYLOINFER_CUDA_CHECK(mfq::cuda::moe_mapped_copy_launch_geometry(&upload_blocks_,&upload_threads_));
        if (!host_limit_) host_limit_ = host_available() / 5 * 4;
        ordinary_host_limit_=host_limit_;
        const auto* keep_original=std::getenv("TYLOINFER_GGUF_HOST_SNAPSHOT");
        if(keep_original && keep_original[0]=='1') {
            auto planned=GgufHostSnapshot::plan(source);
            const auto available=host_available();
            // Honor an explicit RAM cache cap. Automatic mode may retain the
            // complete original image only while leaving 10% of available RAM
            // for later allocations; under pressure reserve_host drops it.
            const auto budget=execution.config.moe_host_cache_bytes?host_limit_:available-available/10;
            if(planned && planned->bytes<=budget && planned->bytes<=available-available/10) {
                try {
                    planned->load(source,mfq_current_cuda_device());
                    source_bytes_+=planned->payload;source_reads_+=planned->banks.size();
                    host_limit_=std::max(host_limit_,planned->bytes);snapshot_=std::move(planned);
                } catch(const std::exception& error) {
                    (void)cudaGetLastError();
                    std::fprintf(stderr,"native_gguf_host_snapshot unavailable=%s bounded_cache=1\n",error.what());
                }
            } else std::fprintf(stderr,"native_gguf_host_snapshot skipped=ram_budget required=%zu available=%zu budget=%zu\n",
                planned?planned->bytes:0,available,budget);
        }
        if (!configured_device_limit_ && execution.moe_expert_cache)
            configured_device_limit_ = moe_expert_cache_budget_bytes(execution.moe_expert_cache);
        // Create before a graph temporarily pins its serving caller. Linux
        // threads otherwise inherit that narrow affinity at the first update.
        admission_executor_=std::make_unique<GgufAdmissionExecutor>();
    }
    ~GgufExpertCache() {
        // CPU workers retain RAM images; finish them before freeing arenas.
        cpu_executor_.reset();
        try {wait_direct_admission();}catch(const std::exception& error) {
            std::fprintf(stderr,"GGUF cache admission failed during shutdown: %s\n",error.what());
        }catch(...) {
            std::fprintf(stderr,"GGUF cache admission failed during shutdown\n");
        }
        admission_executor_.reset();
        for (const auto& transfer : transfers_) try { transfer.ready->synchronize(); } catch (...) {}
        for (auto& stage : upload_stages_) {
            if (stage.ready) try { stage.ready->synchronize(); } catch (...) {}
            if (stage.host) mfq::cuda::tensor_host_bytes.fetch_sub(upload_capacity);
        }
        // Tensor record_stream keeps consumer handles for its final fence.
        // Retire those tensors while the owned PP upload streams still live;
        // reverse member destruction would otherwise destroy the streams first.
        transfers_.clear();device_.clear();
        for(auto& stage:upload_stages_)stage.device={};
        for (auto& [stream, pool] : graph_pools_) pool.context->end_graph_pool(pool.stream.stream());
        if (direct_host_) mfq::cuda::tensor_host_bytes.fetch_sub(direct_capacity);
    }
    bool direct_supported(std::size_t bytes) const {return direct_mapped_ && bytes<=direct_capacity;}
    bool set_reuse_learning(bool enabled) {
        std::lock_guard lock(mutex_);return std::exchange(reuse_learning_,enabled);
    }
    void record_prefill(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            const std::vector<std::vector<std::int64_t>>& routes) {
        if(banks[0]->resident_packed.defined())return;
        if(banks[0]->cache!=banks[1]->cache || banks[0]->cache!=banks[2]->cache)return;
        std::lock_guard lock(mutex_);
        if(!reuse_learning_)return;
        auto& demand=prefill_demand_[banks[0]->source_id];
        for(int role=0;role<3;++role)demand.banks[role]=banks[role];
        demand.counts.resize(routes.size());
        for(std::size_t expert=0;expert<routes.size();++expert)demand.counts[expert]=routes[expert].size();
        prefill_rebalance_pending_=true;
    }
    void rebalance_prefill() {
        wait_direct_admission();
        struct Candidate {
            std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
            std::int64_t expert;
            std::uint64_t count;
            std::size_t bytes;
        };
        const auto started=std::chrono::steady_clock::now();
        const int ordinal=mfq_current_cuda_device();
        std::vector<Candidate> selected;
        {
            std::lock_guard lock(mutex_);
            if(!prefill_rebalance_pending_)return;
            prefill_rebalance_pending_=false;
            if(slot_tiers_.count(ordinal))return;
            if(!device_limit_.count(ordinal) || !device_limit_.at(ordinal))return;
            std::vector<Candidate> candidates;
            for(auto& [source,demand]:prefill_demand_) {
                std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
                bool alive=true;
                for(int role=0;role<3;++role){banks[role]=demand.banks[role].lock();alive&=bool(banks[role]);}
                if(!alive)continue;
                const auto bytes=banks[0]->expert_bytes+banks[1]->expert_bytes+banks[2]->expert_bytes;
                for(std::size_t expert=0;expert<demand.counts.size();++expert)
                    if(demand.counts[expert])candidates.push_back({banks,std::int64_t(expert),demand.counts[expert],bytes});
            }
            // Transfer time scales with compressed bytes, so the prompt's
            // token frequency ranks saved transfer time per resident byte.
            // Allocate across layers instead of assigning every layer the
            // same fraction of its expert count.
            std::stable_sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b){return a.count>b.count;});
            std::size_t remaining=device_limit_.at(ordinal);
            std::unordered_set<std::uint64_t> wanted;
            for(auto& candidate:candidates)if(candidate.bytes<=remaining) {
                remaining-=candidate.bytes;
                for(const auto& bank:candidate.banks)wanted.insert((bank->source_id<<32)|std::uint64_t(candidate.expert));
                selected.push_back(std::move(candidate));
            }
            if(strata_slots_requested())initialize_slot_tier(selected,ordinal);
            bool released=false;
            for(auto next=device_.begin();next!=device_.end();) {
                const auto& key=next->first;
                if(key.device!=ordinal || wanted.count(key.image)){++next;continue;}
                demote(key,next->second);
                device_recent_.erase(next->second.recent);
                device_used_[ordinal]-=next->second.weight.packed.nbytes();
                remove_device_copy(key.image,key.device);next=device_.erase(next);
                ++device_evictions_;released=true;
            }
            if(released) {
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
                reap();mfq::cuda::default_context(ordinal)->reclaim_idle_memory();
            }
        }
        // This is prompt cleanup, after the PP timer stops and before the
        // first generated token. It remains part of TTFT.
        std::map<std::uint64_t,std::vector<const Candidate*>> groups;
        for(const auto& candidate:selected)groups[candidate.banks[0]->source_id].push_back(&candidate);
        for(const auto& [source,group]:groups) {
            std::vector<std::int64_t> experts;experts.reserve(group.size());
            for(const auto* candidate:group)experts.push_back(candidate->expert);
            (void)acquire_moe(group.front()->banks,experts,ordinal);
        }
        std::lock_guard lock(mutex_);
        direct_demand_.clear();direct_window_pending_=false;
        prefill_resident_experts_=selected.size();
        prefill_rebalance_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    void finish_direct(bool frequent=false) {
        // Window completion precedes this callback. Destroying even a ready
        // event in read_host can block WDDM's driver while the GPU is waiting
        // for that callback to publish its route plan. Release outside both
        // the waiting graph and the cache mutex.
        std::vector<std::shared_ptr<mfq::cuda::Event>> retired;
        {
            std::lock_guard lock(mutex_);retired.swap(deferred_events_);
            deferred_event_releases_+=retired.size();
        }
        retired.clear();
        struct Candidate {
            std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
            std::int64_t expert;
            std::uint64_t score;
            std::size_t bytes;
            std::int64_t victim=-1;
        };
        std::vector<Candidate> selected;
        const int device=mfq_current_cuda_device();
        bool deferred_selection=false;
        {
            std::lock_guard lock(mutex_);
            if(speculative_rows_)return;
            if(!direct_window_pending_)return;
            direct_window_pending_=false;
            if(route_trace_) {
                const auto tier=slot_tiers_.find(device);
                route_trace_->complete(-1,true,device,tier==slot_tiers_.end()?0:tier->second.windows);
            }
            if(!device_limit_.count(device) || !device_limit_.at(device))return;
            deferred_selection=slot_tiers_.count(device)!=0;
            if(deferred_selection) {
                auto& tier=slot_tiers_.at(device);
                if(tier.loan_active)return;
                ++tier.windows;
                if(!frequent && tier.windows%4)return;
            }
            if(!deferred_selection) {
                std::vector<Candidate> candidates;
                for(auto& [source,demand]:direct_demand_) {
                    std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
                    bool alive=true;
                    for(int role=0;role<3;++role){banks[role]=demand.banks[role].lock();alive&=bool(banks[role]);}
                    if(!alive){demand.scores.clear();continue;}
                    const auto bytes=banks[0]->expert_bytes+banks[1]->expert_bytes+banks[2]->expert_bytes;
                    for(const auto& [expert,score]:demand.scores)candidates.push_back({banks,expert,score,bytes});
                }
                std::sort(candidates.begin(),candidates.end(),[](const auto& a,const auto& b) {
                    if(a.score!=b.score)return a.score>b.score;
                    if(a.banks[0]->source_id!=b.banks[0]->source_id)return a.banks[0]->source_id<b.banks[0]->source_id;
                    return a.expert<b.expert;
                });
                const auto limit=device_limit_.at(device),used=device_used_[device];
                auto free=limit-std::min(limit,used);
                std::size_t remaining=std::max<std::size_t>(16ULL<<20,std::min<std::size_t>(free,512ULL<<20));
                for(auto& candidate:candidates) {
                    if(candidate.bytes>free && candidate.score<2)continue;
                    if(candidate.bytes>remaining || candidate.bytes>device_limit_.at(device))continue;
                    remaining-=candidate.bytes;
                    free-=std::min(free,candidate.bytes);
                    direct_demand_[candidate.banks[0]->source_id].scores.erase(candidate.expert);
                    selected.push_back(std::move(candidate));
                }
            }
        }
        // The window has finished: allocation, eviction and DMA are safe here.
        // Frequently missed experts receive bounded admission without adding
        // any driver calls to the graph's per-layer CPU callbacks.
        if(selected.empty() && !deferred_selection)return;
        wait_direct_admission();
        // The fallback tier selects on the caller. Detach its source metadata
        // before the job, too: a worker must never retain a bank -> cache cycle.
        for(auto& candidate:selected)for(auto& bank:candidate.banks)if(bank->cache) {
            bank=std::make_shared<GgufRoutedWeight>(*bank);bank->cache.reset();
        }
        const auto* option=std::getenv("TYLOINFER_GGUF_ASYNC_ADMISSION");
        const bool asynchronous=!option || option[0]!='0';
        const auto transfer=asynchronous?admission_transfer_stream(device):mfq_get_current_cuda_stream();
        const auto epoch=asynchronous && deferred_selection && transfer.stream()!=prefill_transfer_stream(device).stream()
            ?begin_layer_admission(device):std::shared_ptr<GgufLayerAdmission>{};
        auto run=[this,selected=std::move(selected),deferred_selection,frequent,asynchronous,transfer,device,epoch]() mutable {
            const MfqCudaStreamGuard stream(transfer);
            try {
            if(deferred_selection) {
                const auto started=std::chrono::steady_clock::now();
                std::lock_guard lock(mutex_);
                select_slot_swaps(selected,device,frequent);
                admission_selection_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
            }
            if(selected.empty()){
                release_unchanged_layers(epoch,{});
                if(epoch)epoch->submitted_done->record(transfer.stream());
                return;
            }
            if(asynchronous){std::lock_guard lock(mutex_);++async_admission_batches_;}
            // A layer's three banks share one upload batch. Avoid a separate
            // packet/event and three cache acquisitions for every admitted expert.
            std::map<std::uint64_t,std::vector<const Candidate*>> groups;
            for(const auto& item:selected)groups[item.banks[0]->source_id].push_back(&item);
            struct Batch {
                std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
                std::vector<std::int64_t> experts;
                std::vector<std::int64_t> victims;
            };
            std::vector<Batch> batches;batches.reserve(groups.size());
            for(const auto& [source,items]:groups) {
                Batch batch;batch.experts.reserve(items.size());
                batch.banks=items.front()->banks;
                for(const auto* item:items){batch.experts.push_back(item->expert);batch.victims.push_back(item->victim);}
                batches.push_back(std::move(batch));
            }
            std::unordered_set<std::uint64_t> pending;
            for(const auto& batch:batches)pending.insert(batch.banks[0]->source_id);
            release_unchanged_layers(epoch,pending);
            for(const auto& batch:batches) {
                std::vector<std::int64_t> admitted;
                for(std::size_t i=0;i<batch.experts.size();++i)
                    if(prepare_slot_expert(batch.banks,batch.experts[i],device,batch.victims[i]))admitted.push_back(batch.experts[i]);
                if(!admitted.empty())(void)acquire_moe(batch.banks,admitted,device,true,nullptr,nullptr,UploadPurpose::CacheAdmission);
                publish_admission_layer(epoch,batch.banks,transfer.stream());
            }
            // Completing the host job alone only proves DMA was enqueued.
            // Finish those copies and allocator dependencies while the draft
            // runs, before another graph can wait on host-published routes.
            if(epoch)epoch->submitted_done->record(transfer.stream());
            else TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(transfer.stream()));
            }catch(...) {
                const auto error=std::current_exception();
                abort_layer_admission(epoch,error);
                // Release graph gates before draining the independent stream.
                // Failed epochs must not leave partial copies for a retry.
                (void)cudaStreamSynchronize(transfer.stream());
                std::rethrow_exception(error);
            }
        };
        if(asynchronous) {
            try{admission_job_=admission_executor_->submit(std::move(run));}
            catch(...){abort_layer_admission(epoch,std::current_exception());throw;}
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_admission_worker actual_match=1 persistent=1\n");
            }
            if(epoch)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true))std::fprintf(stderr,
                    "native_gguf_layer_admission enabled=1 fixed_layer_slots=1 separate_stream=1 layers=%zu\n",epoch->views.size());
            }
        } else run();
    }
    bool speculative_routes() {
        std::lock_guard lock(mutex_);return speculative_rows_>0;
    }
    void begin_speculative_routes(int rows) {
        prepare_admission_window(true);
        if(rows<1)throw std::invalid_argument("GGUF speculative route window must be positive");
        std::lock_guard lock(mutex_);
        speculative_rows_=rows;speculative_demand_.clear();
    }
    void observe_routes(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            const std::int64_t* ids,std::int64_t tokens,std::int64_t routes,int device) {
        if(banks[0]->resident_packed.defined())return;
        std::lock_guard lock(mutex_);
        if(!reuse_learning_)return;
        // A fixed arena can exchange experts only within its registered
        // layers. A separate bank (e.g. an attention-only-primed predictor)
        // cannot occupy these slots. Counting its drafts as arena windows
        // would decay the target's route heat and schedule useless exchanges.
        if(const auto tier=slot_tiers_.find(device);tier!=slot_tiers_.end() &&
                !tier->second.roles.count(banks[0]->source_id)) {
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_cache_observation actual_match=1 unmanaged_bank_skipped=1 source=%llu\n",
                    static_cast<unsigned long long>(banks[0]->source_id));
            }
            return;
        }
        auto& demand=direct_demand_[banks[0]->source_id];
        for(int role=0;role<3;++role)demand.banks[role]=banks[role];
        const auto missed=[&](std::int64_t expert) {
            for(const auto& bank:banks)if(!device_.count(Key{(bank->source_id<<32)|std::uint64_t(expert),device}))return true;
            return false;
        };
        if(route_trace_)route_trace_->observe(banks[0]->source_id,device,tokens,routes,ids,banks[0]->experts,
            banks[0]->expert_bytes+banks[1]->expert_bytes+banks[2]->expert_bytes,
            speculative_rows_>0,[&](auto expert){return !missed(expert);});
        if(speculative_rows_) {
            auto& pending=speculative_demand_[banks[0]->source_id];
            pending.banks=demand.banks;pending.device=device;
            // Graph preparation may execute this layer several times. Keep
            // its latest window, never learn the warmup routes repeatedly.
            pending.rows=std::size_t(tokens);
            pending.entries.clear();
            pending.entries.reserve(std::min(std::size_t(tokens*routes),std::size_t(banks[0]->experts)));
            speculative_journal_.collect({ids,std::size_t(tokens*routes)},std::size_t(banks[0]->experts),
                pending.touches,[&](auto expert) {
                    auto& entries=pending.entries.emplace_back();
                    bool missing=false;
                    for(std::size_t role=0;role<banks.size();++role) {
                        const auto found=device_.find({(banks[role]->source_id<<32)|std::uint64_t(expert),device});
                        entries[role]=found==device_.end()?nullptr:&found->second;
                        missing|=!entries[role];
                    }
                    return missing;
                });
            pending.resident_version=resident_version_;
            return;
        }
        // The same bounded journal serves ordinary routes without allocating
        // a hash node for every expert/layer/token. Heat still receives each
        // original +1, including FP32 saturation; demand counts each unique
        // missed expert once per window, as the former set-based loop did.
        speculative_journal_.collect({ids,std::size_t(tokens*routes)},std::size_t(banks[0]->experts),ordinary_touches_,missed);
        std::vector<float>* usage=nullptr;
        if(auto tier=slot_tiers_.find(device);tier!=slot_tiers_.end())
            if(auto role=tier->second.roles.find(banks[0]->source_id);role!=tier->second.roles.end())
                usage=&tier->second.usage[role->second.layer];
        for(const auto& touch:ordinary_touches_) {
            if(usage && touch.expert>=0 && std::uint64_t(touch.expert)<usage->size())
                for(std::uint64_t i=0;i<touch.uses;++i)(*usage)[touch.expert]+=1.f;
            if(touch.missed)++demand.scores[touch.expert];else demand.scores.erase(touch.expert);
        }
        direct_window_pending_=true;
    }
    void resolve_speculative_routes(int rows) {
        {
            std::lock_guard lock(mutex_);
            if(!speculative_rows_)return;
            if(!reuse_learning_){speculative_demand_.clear();speculative_rows_=0;return;}
            const int committed=std::clamp(rows,0,speculative_rows_);
            if(route_trace_)route_trace_->complete(committed,committed>0,mfq_current_cuda_device(),
                slot_tiers_.count(mfq_current_cuda_device())?slot_tiers_.at(mfq_current_cuda_device()).windows:0);
            for(auto& [source,pending]:speculative_demand_) {
                const auto count=std::min(std::size_t(committed),pending.rows);
                speculative_committed_+=count;speculative_discarded_+=pending.rows-count;
                // Strata counts every executed verifier row in expert heat
                // (expert_source.cpp, begin_layer). Rejected drafts consumed
                // real weights too. KV/state still keeps only the accepted
                // prefix; graph preparation/cancellation discards the journal.
                const auto observed=count?pending.rows:0;
                speculative_cache_observed_+=observed;
                auto& demand=direct_demand_[source];demand.banks=pending.banks;
                // Residency and bank identities stay fixed under this lock.
                // Resolve them once per layer rather than once per route row.
                std::vector<float>* usage=nullptr;
                if(auto tier=slot_tiers_.find(pending.device);tier!=slot_tiers_.end())
                    if(auto role=tier->second.roles.find(source);role!=tier->second.roles.end())
                        usage=&tier->second.usage[role->second.layer];
                std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
                for(std::size_t role=0;role<banks.size();++role)banks[role]=pending.banks[role].lock();
                const bool snapshot_current=pending.resident_version==resident_version_;
                if(observed)for(std::size_t index=pending.touches.size();index-->0;) {
                    const auto* it=&pending.touches[index];
                    const auto expert=it->expert;
                    // Repeated +1 preserves the original FP32 heat rounding,
                    // including saturation. Integer demand counts can combine.
                    if(usage && expert>=0 && std::uint64_t(expert)<usage->size())
                        for(std::uint64_t i=0;i<it->uses;++i)(*usage)[expert]+=1.f;
                    if(it->missed)demand.scores[expert]+=it->uses;else demand.scores.erase(expert);
                    speculative_route_visits_+=it->uses;++speculative_unique_touches_;
                    if(snapshot_current)++speculative_snapshot_touches_;else ++speculative_refind_touches_;
                    for(std::size_t role=0;role<banks.size();++role)if(banks[role]) {
                        DeviceEntry* entry=nullptr;
                        if(snapshot_current)entry=pending.entries[index][role];
                        else {
                            const Key key{(banks[role]->source_id<<32)|std::uint64_t(expert),pending.device};
                            const auto found=device_.find(key);
                            if(found!=device_.end())entry=&found->second;
                        }
                        if(entry)device_recent_.splice(device_recent_.begin(),device_recent_,entry->recent);
                    }
                }
                if(observed)direct_window_pending_=true;
            }
            speculative_demand_.clear();speculative_rows_=0;
        }
        const auto* option=std::getenv("TYLOINFER_GGUF_SPEC_ADAPT");
        finish_direct(!option || option[0]!='0');
    }
    void discard_speculative_routes() noexcept {
        std::lock_guard lock(mutex_);
        for(const auto& [source,pending]:speculative_demand_)speculative_discarded_+=pending.rows;
        if(route_trace_ && reuse_learning_)route_trace_->complete(0,false,mfq_current_cuda_device(),0);
        speculative_demand_.clear();speculative_rows_=0;
    }
    void prepare_graph_pool() {
        wait_direct_admission();
        const auto stream = mfq_get_current_cuda_stream();
        std::lock_guard lock(mutex_);
        if (graph_pools_.count(stream.stream())) return;
        const auto context = mfq::cuda::default_context(stream.device());
        context->begin_graph_pool(stream.stream());
        try { graph_pools_.emplace(stream.stream(), GraphPool{stream, context}); }
        catch (...) { context->end_graph_pool(stream.stream()); throw; }
    }
    std::uint64_t register_source(const std::string& name,std::size_t bytes,mfq::ModelSource::TensorReader recovery,bool resident=false) {
        std::lock_guard lock(mutex_);
        if (bytes > SIZE_MAX - (resident?predictor_resident_bytes_:payload_bytes_)) throw std::overflow_error("GGUF source payload overflow");
        if (++sources_ > UINT32_MAX) throw std::overflow_error("too many GGUF expert sources");
        if(resident)predictor_resident_bytes_+=bytes;
        else payload_bytes_ += bytes;
        if(snapshot_ && !resident) {
            const auto bank=snapshot_->banks.find(name);
            if(bank==snapshot_->banks.end() || bank->second.bytes!=bytes)
                throw std::invalid_argument("GGUF host snapshot has no routed bank: "+name);
            snapshot_sources_.emplace(sources_,bank->second);
        }
        if(recovery)source_readers_.emplace(sources_,std::move(recovery));
        return sources_;
    }
    std::size_t budget_bytes() {
        std::lock_guard lock(mutex_);
        std::size_t bytes = 0;
        for (const auto& [device, limit] : device_limit_) bytes += limit;
        return device_limit_.empty() ? configured_device_limit_.value_or(payload_bytes_) : bytes;
    }
    std::uint64_t streamed_bytes() {
        std::lock_guard lock(mutex_); return transfer_bytes_;
    }
    double fetch_ms() {
        std::lock_guard lock(mutex_);return fetch_ms_;
    }
    std::size_t prefill_quota(std::size_t experts, int device) {
        std::lock_guard lock(mutex_);
        ++grouped_prefills_;
        const auto budget = device_limit_.count(device) ? device_limit_.at(device) : configured_device_limit_.value_or(0);
        return payload_bytes_ ? std::min(experts, std::size_t(static_cast<long double>(experts)*budget/payload_bytes_)) : 0;
    }
    void finalize(std::size_t reserve) {
        wait_direct_admission();
        const auto device = mfq_current_cuda_device();
        const auto context = mfq::cuda::default_context(device);
        context->reclaim_idle_memory();
        const auto available = context->available_memory();
        const auto workspace = mfq::cuda::runtime_options::workspace_reserve_bytes();
        const auto total_reserve = reserve > SIZE_MAX - workspace ? SIZE_MAX : reserve + workspace;
        std::lock_guard lock(mutex_);
        const auto capacity = available > total_reserve ? available - total_reserve : 0;
        device_limit_[device] = std::min(configured_device_limit_.value_or(capacity), capacity);
        evict_device(device, 0);
    }
    void reserve_workspace(std::size_t bytes) {
        wait_direct_admission();
        const auto device = mfq_current_cuda_device();
        const auto context = mfq::cuda::default_context(device);
        std::lock_guard lock(mutex_);
        if(slot_tiers_.count(device)) {
            context->reclaim_idle_memory();reap();
            if(context->available_memory()>=bytes)return;
            // KV growth needs physical pages outside the PP loan. Retire the
            // fixed arena before reserving them; restore_workspace can seed
            // the remaining capacity again after the new allocation exists.
            release_slot_tier(device);
        }
        reap();
        context->reclaim_idle_memory();
        auto available = context->available_memory();
        const auto before=available,resident_before=device_used_[device];
        // Reclaim adjacent virtual allocation regions together. Evicting a
        // few scattered objects can leave every backing page partly live.
        std::size_t granularity=0;
#if CUDA_VERSION >= 10020
        CUmemAllocationProp properties{};properties.type=CU_MEM_ALLOCATION_TYPE_PINNED;
        properties.location.type=CU_MEM_LOCATION_TYPE_DEVICE;properties.location.id=device;
        if(cuMemGetAllocationGranularity(&granularity,&properties,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)!=CUDA_SUCCESS)
            granularity=0;
#endif
        struct Allocation {std::uintptr_t begin,end;Key key;};
        std::vector<Allocation> allocations;
        if(granularity && available<bytes) {
            for(const auto& [key,entry]:device_)if(key.device==device) {
                const auto start=reinterpret_cast<std::uintptr_t>(entry.weight.packed.data_ptr());
                allocations.push_back({start,start+entry.weight.packed.nbytes(),key});
            }
            std::sort(allocations.begin(),allocations.end(),[](const auto& a,const auto& b){return a.begin<b.begin;});
        }
        // Async-pool pages can contain several expert allocations. Removing
        // their logical payload does not guarantee that those pages are free.
        // Recheck physical availability after each batch before accepting the
        // workspace reservation.
        while(available<bytes) {
            const auto needed=bytes-available;
            std::size_t released=0;
            while(released<needed) {
                auto old=std::find_if(device_recent_.rbegin(),device_recent_.rend(),
                    [device](const Key& key){return key.device==device;});
                if(old==device_recent_.rend())break;
                const auto erase=[&](const Key& key) {
                    auto found=device_.find(key);if(found==device_.end())return;
                    demote(key,found->second);device_recent_.erase(found->second.recent);
                    const auto count=found->second.weight.packed.nbytes();
                    released+=count;device_used_[device]-=count;device_.erase(found);
                    remove_device_copy(key.image,key.device);++device_evictions_;
                };
                const auto key=*old;
                if(granularity) {
                    const auto address=reinterpret_cast<std::uintptr_t>(device_.at(key).weight.packed.data_ptr());
                    const auto first=address/granularity*granularity,last=first+granularity;
                    auto next=std::lower_bound(allocations.begin(),allocations.end(),first,
                        [](const auto& allocation,auto begin){return allocation.begin<begin;});
                    if(next!=allocations.begin() && std::prev(next)->end>first)--next;
                    for(;next!=allocations.end() && next->begin<last;++next)erase(next->key);
                } else erase(key);
            }
            if(!released)break;
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            reap();
            context->reclaim_idle_memory();
            available=context->available_memory();
        }
        // Do not refill the returned workspace with PP cache admissions.
        // restore_workspace recomputes the configured decode capacity later.
        if(auto limit=device_limit_.find(device);limit!=device_limit_.end())
            limit->second=std::min(limit->second,device_used_[device]+(available>bytes?available-bytes:0));
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_LAYER_TIMINGS");trace && trace[0]!='0')
            std::fprintf(stderr,"runtime_gguf_workspace_reserve requested_bytes=%zu available_before=%zu available_after=%zu resident_before=%zu resident_after=%zu\n",
                bytes,before,available,resident_before,device_used_[device]);
    }
    void restore_workspace(std::size_t growth,std::optional<std::size_t> transient) {
        wait_direct_admission();
        const auto device=mfq_current_cuda_device();
        const auto context=mfq::cuda::default_context(device);
        context->reclaim_idle_memory();
        // Captured graph buffers are already charged to the allocator. Keep
        // only the prepared decoder's transient allowance after prompt tensors
        // retire; unprepared callers retain their original capture allowance.
        // Explicit user reservations always take precedence.
        const auto add=[](std::size_t a,std::size_t b){return b>SIZE_MAX-a?SIZE_MAX:a+b;};
        const bool configured=std::getenv("TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB")!=nullptr;
        // Concrete operator workspaces do not price all driver residency costs.
        // Retain Strata's general 700 MiB free-memory allowance after capture;
        // explicit reservations still take precedence on every device.
        constexpr auto residency_allowance=std::size_t(700)<<20;
        const auto base=configured?mfq::cuda::runtime_options::workspace_reserve_bytes():
            add(std::max(transient.value_or(std::size_t(1280)<<20),residency_allowance),2*upload_capacity);
        const auto reserve=add(base,growth);
        {
            std::lock_guard lock(mutex_);reap();context->reclaim_idle_memory();
            decode_workspace_reserve_=reserve;decode_workspace_prepared_=transient.has_value() && !configured;
            decode_growth_reserve_=growth;decode_transient_reserve_=transient;
            const auto available=context->available_memory(),reusable=reusable_resident_memory(device);
            // Cache allocations can reuse free space inside their own pool
            // without increasing physical residency. PP workspace uses the
            // temporary pool and therefore cannot count this space as free.
            const auto existing=slot_tiers_.find(device);
            const auto physical=existing==slot_tiers_.end()?device_used_[device]:std::size_t(existing->second.cache->bytes());
            const auto capacity=physical+reusable+(available>reserve?available-reserve:0);
            if(const auto tier=slot_tiers_.find(device);tier!=slot_tiers_.end() &&
                    std::uint64_t(tier->second.cache->bytes())>capacity)release_slot_tier(device);
            device_limit_[device]=std::min(configured_device_limit_.value_or(capacity),capacity);
            evict_device(device,0);
        }
        rebalance_prefill();
        if(route_trace_) {
            std::lock_guard lock(mutex_);
            const auto found=slot_tiers_.find(device);
            if(found!=slot_tiers_.end())route_trace_->record([&](auto& out) {
                using Trace=mfq::cuda::gguf::RouteTrace;
                const auto& tier=found->second;
                out << "{\"event\":\"snapshot\",\"device\":" << device << ",\"clock\":" << tier.windows
                    << ",\"last_decay\":" << tier.last_decay
                    << ",\"budget\":" << tier.cache->bytes() << ",\"layers\":[";
                bool first=true;
                for(std::size_t layer=0;layer<tier.banks.size();++layer) {
                    const auto bank=tier.banks[layer][0].lock();if(!bank)continue;
                    if(!first)out << ',';first=false;
                    std::size_t bytes=0;for(const auto& weak:tier.banks[layer])if(const auto b=weak.lock())bytes+=b->expert_bytes;
                    out << "{\"source\":" << bank->source_id << ",\"bytes\":" << bytes << ",\"usage\":";
                    Trace::array(out,tier.usage[layer]);out << ",\"resident\":[";
                    bool resident_first=true;
                    for(std::int64_t expert=0;expert<bank->experts;++expert)if(slot_admitted(*bank,expert,device)) {
                        if(!resident_first)out << ',';resident_first=false;out << expert;
                    }
                    out << "]}";
                }
                out << "]}";
            });
        }
    }
    void restore_after_capture() {
        std::size_t growth;
        std::optional<std::size_t> transient;
        {
            std::lock_guard lock(mutex_);
            growth=decode_growth_reserve_;transient=decode_transient_reserve_;
        }
        restore_workspace(growth,transient);
    }
    void resize(std::size_t bytes) {
        wait_direct_admission();
        std::lock_guard lock(mutex_);
        std::vector<int> release;
        for(const auto& [device,tier]:slot_tiers_)if(std::uint64_t(tier.cache->bytes())>bytes)release.push_back(device);
        for(const auto device:release)release_slot_tier(device);
        configured_device_limit_ = bytes;
        for (auto& [device, limit] : device_limit_) { limit = bytes; evict_device(device, 0); }
    }
    std::size_t reserve_host(std::size_t bytes) {
        wait_direct_admission();
        std::lock_guard lock(mutex_);
        const auto available = host_available();
        if(available>=bytes)return 0;
        auto released=reclaim_host(bytes-available);
        if(snapshot_ && available+released<bytes) {
            // This is a request-planning boundary, never a host callback of a
            // waiting graph. Do not unregister RAM still owned by a consumer.
            bool busy=serving_graph_;
            for(const auto& [key,entry]:host_)if(entry.image->snapshot && entry.image.use_count()!=1)busy=true;
            for(const auto& transfer:transfers_)if(transfer.image->snapshot)busy=true;
            if(!busy) {
                const auto original=snapshot_->bytes;
                std::vector<std::uint64_t> discard;
                for(const auto& [key,entry]:host_)if(entry.image->snapshot)discard.push_back(key);
                for(const auto key:discard)erase_host(key);
                snapshot_sources_.clear();snapshot_.reset();host_limit_=ordinary_host_limit_;
                released+=original;
                std::fprintf(stderr,"native_gguf_host_snapshot released_bytes=%zu reason=ram_workspace\n",original);
            }
        }
        return released;
    }
    std::vector<std::pair<std::string, double>> memory_metrics() {
        std::lock_guard lock(mutex_);
        std::uint64_t decode_dma_bytes=0,decode_dma_copies=0,decode_dma_batches=0,decode_dma_signals=0;
        for(const auto& [device,lane]:decode_dma_) {
            decode_dma_bytes+=lane->bytes();decode_dma_copies+=lane->copies();decode_dma_batches+=lane->batches();
            decode_dma_signals+=lane->signals();
        }
        std::size_t device_bytes = 0, budget = 0, covered = host_used_;
        std::size_t slot_bytes=0,borrowed_bytes=0;
        for(const auto& [device,tier]:slot_tiers_) {
            slot_bytes+=tier.cache->bytes();
            if(tier.loan_active)borrowed_bytes+=tier.cache->bytes()-tier.cache->bytes_of(tier.loan_first);
        }
        std::size_t live_arena_bytes=0,live_arenas=0;
        std::erase_if(slot_arenas_,[](const auto& arena){return arena.expired();});
        for(const auto& weak:slot_arenas_)if(const auto arena=weak.lock()) {
            live_arena_bytes+=std::size_t(arena->bytes());++live_arenas;
        }
        std::size_t pinned = 0;
        for (const auto& [key, entry] : host_) if (entry.image->pinned()) pinned += entry.image->bytes;
        std::unordered_set<std::uint64_t> covered_images;
        for (const auto& [key, entry] : device_) {
            if (!host_.count(key.image) && covered_images.insert(key.image).second)
                covered += entry.weight.packed.nbytes();
        }
        for (const auto& [device, bytes] : device_used_) device_bytes += bytes;
        for (const auto& [device, bytes] : device_limit_) budget += bytes;
        double cpu_fit[5]{},tail_risk=0,symmetric_risk=0;std::uint64_t deferred_probes=0,risk_recovery_probes=0;
        for(const auto& [key,model]:cpu_balance_) {
            cpu_fit[0]+=model.a;cpu_fit[1]+=model.c;cpu_fit[2]+=model.d;
            cpu_fit[3]+=model.g0;cpu_fit[4]+=model.g;
            deferred_probes+=model.deferred_probes;
            risk_recovery_probes+=model.risk_recovery_probes;
            tail_risk+=model.tail_variance;symmetric_risk+=model.symmetric_variance;
        }
        if(!cpu_balance_.empty())for(auto& value:cpu_fit)value/=cpu_balance_.size();
        if(!cpu_balance_.empty()){tail_risk/=cpu_balance_.size();symmetric_risk/=cpu_balance_.size();}
        const auto cold = payload_bytes_ - std::min(payload_bytes_, covered);
        std::size_t direct_arena_bytes=0;
        for(const auto& allocation:direct_allocations_)if(auto arena=allocation.lock())direct_arena_bytes+=arena->storage.nbytes();
        std::vector<std::pair<std::string,double>> result{{"resident_expert_cache_bytes", double(device_bytes)},
            {"gguf_decode_workspace_reserve_bytes",double(decode_workspace_reserve_)},
            {"gguf_decode_workspace_prepared",double(decode_workspace_prepared_)},
            {"gguf_direct_arena_bytes",double(direct_arena_bytes)},
            {"gguf_direct_arena_reuses",double(direct_arena_reuses_)},
            {"gguf_gpu_resident_view_refreshes",double(resident_view_refreshes_)},
            {"gguf_gpu_resident_view_updates",double(resident_view_updates_)},
            {"gguf_gpu_resident_view_prepare_ms",resident_view_prepare_ms_},
            {"gguf_predictor_resident_expert_bytes",double(predictor_resident_bytes_)},
            {"gguf_prefill_inplace_mmq_bytes",double(prefill_inplace_mmq_bytes_)},
            {"gguf_host_device_ready_waits",double(host_device_ready_waits_)},
            {"gguf_prefill_device_ready_skips",double(prefill_device_ready_skips_)},
            {"gguf_speculative_cache_observed_rows",double(speculative_cache_observed_)},
            {"gguf_deferred_events",double(deferred_events_.size())},
            {"gguf_transfer_reap_images",double(transfer_reap_images_)},
            {"gguf_transfer_reap_queries",double(transfer_reap_queries_)},
            {"gguf_deferred_event_releases",double(deferred_event_releases_)},
            {"gguf_strata_slot_cache_bytes",double(slot_bytes)},
            {"gguf_slot_live_arena_bytes",double(live_arena_bytes)},
            {"gguf_slot_live_arenas",double(live_arenas)},
            {"gguf_slot_retired_arena_bytes",double(live_arena_bytes>slot_bytes?live_arena_bytes-slot_bytes:0)},
            {"gguf_slot_touch_free_before_bytes",double(slot_touch_free_before_)},
            {"gguf_slot_touch_free_after_bytes",double(slot_touch_free_after_)},
            {"gguf_slot_touch_reserve_bytes",double(slot_touch_reserve_)},
            {"gguf_slot_touch_retries",double(slot_touch_retries_)},
            {"gguf_prefill_borrowed_bytes",double(borrowed_bytes)},
            {"gguf_prefill_last_loan_required_bytes",double(prefill_loan_required_)},
            {"gguf_prefill_last_loan_bytes",double(prefill_loan_last_)},
            {"gguf_prefill_last_loan_retained_bytes",double(prefill_loan_retained_)},
            {"expert_cache_budget_bytes", double(device_limit_.empty() ? configured_device_limit_.value_or(0) : budget)},
            {"ram_expert_enabled", host_used_ ? 1. : 0.},
            {"ram_expert_payload_bytes", double(host_used_)},
            {"ram_expert_cache_budget_bytes", double(host_limit_)},
            {"ram_expert_cache_allocated_bytes", double(host_allocated())},
            {"ram_expert_pinned_payload_bytes", double(pinned)},
            {"ram_expert_pageable_payload_bytes", double(host_used_-pinned)},
            {"ram_expert_pinned_reserved_bytes", double(host_reserved_+(snapshot_?snapshot_->arena->registered_bytes:0))},
            {"gguf_host_snapshot_bytes",double(snapshot_?snapshot_->bytes:0)},
            {"gguf_host_gate_up_pair_bytes",double(host_gate_up_pair_bytes_)},
            {"gguf_host_snapshot_locked_bytes",double(snapshot_?snapshot_->arena->locked_bytes:0)},
            {"gguf_prefill_direct_mmq_bytes",double(prefill_direct_mmq_bytes_)},
            {"gguf_prefill_host_staged_bytes",double(prefill_host_staged_bytes_)},
            {"gguf_prefill_host_staging_wait_ms",prefill_host_staging_wait_ms_},
            {"ram_expert_pin_failures", double(pin_failures_)},
            {"ram_expert_pinned_capacity_misses", double(pin_capacity_misses_)},
            {"gguf_immutable_recovery_bytes",double(immutable_recovery_bytes_)},
            {"ram_expert_cache_hits", double(host_hits_)},
            {"ram_expert_cache_evictions", double(host_evictions_)},
            {"ssd_expert_enabled", cold ? 1. : 0.}, {"ssd_expert_payload_bytes", double(cold)},
            {"ssd_expert_bytes_read", double(source_bytes_)}, {"ssd_expert_read_count", double(source_reads_)},
            {"gguf_expert_source_batches",double(source_batches_)},
            {"expert_logical_transfer_bytes", double(transfer_bytes_)},
            {"ram_expert_pcie_read_bytes", double(transfer_bytes_)},
            {"expert_writeback_bytes", double(writeback_bytes_)},
            {"expert_upload_enqueue_ms", upload_enqueue_ms_},
            {"expert_writeback_enqueue_ms", writeback_enqueue_ms_},
            {"expert_allocation_ms", allocation_ms_},
            {"gguf_async_admission_batches",double(async_admission_batches_)},
            {"gguf_admission_join_ms",admission_join_ms_},
            {"gguf_admission_submission_join_ms",admission_submission_join_ms_},
            {"gguf_layer_admission_epochs",double(layer_admission_epochs_)},
            {"gguf_layer_admission_signals",double(layer_admission_signals_)},
            {"gguf_layer_admission_served",double(layer_admission_served_)},
            {"gguf_layer_admission_ready_updates",double(layer_admission_ready_updates_)},
            {"gguf_layer_admission_wait_ms",layer_admission_wait_ms_},
            {"gguf_admission_selection_ms",admission_selection_ms_},
            {"gguf_decode_dma_bytes",double(decode_dma_bytes)},
            {"gguf_decode_dma_copies",double(decode_dma_copies)},
            {"gguf_decode_dma_batches",double(decode_dma_batches)},
            {"gguf_decode_dma_signals",double(decode_dma_signals)},
            {"gguf_admission_dma_bytes",double(admission_dma_bytes_)},
            {"gguf_admission_dma_copies",double(admission_dma_copies_)},
            {"gguf_admission_dma_images",double(admission_dma_images_)},
            {"gguf_adaptive_rounds",double(adaptive_rounds_)},
            {"gguf_direct_workspace_reused_bytes",double(direct_workspace_reused_bytes_)},
            {"gguf_adaptive_fast_rounds",double(adaptive_fast_rounds_)},
            {"gguf_cpu_experts",double(cpu_experts_)},
            {"gguf_cpu_auto_layers",double(cpu_auto_layers_)},
            {"gguf_cpu_auto_selected_experts",double(cpu_auto_selected_)},
            {"gguf_cpu_auto_gpu_only_layers",double(cpu_auto_gpu_only_)},
            {"gguf_cpu_auto_samples",double(cpu_auto_samples_)},
            {"gguf_cpu_clock_samples",double(cpu_clock_samples_)},
            {"gguf_cpu_exposed_waits",double(cpu_exposed_waits_)},
            {"gguf_cpu_exposed_wait_ms",cpu_exposed_wait_ms_},
            {"gguf_cpu_clock_adjust_ms",cpu_clock_adjust_ms_},
            {"gguf_cpu_auto_fitted_layers",double(cpu_auto_fitted_layers_)},
            {"gguf_cpu_auto_models",double(cpu_balance_.size())},
            {"gguf_cpu_auto_cpu_offset_ms",cpu_fit[0]},
            {"gguf_cpu_auto_probe_deferrals",double(deferred_probes)},
            {"gguf_cpu_auto_probe_policy_version",2.},
            {"gguf_cpu_auto_risk_recovery_probes",double(risk_recovery_probes)},
            {"gguf_cpu_auto_cpu_expert_ms",cpu_fit[1]},
            {"gguf_cpu_auto_cpu_contention_ms",cpu_fit[2]},
            {"gguf_cpu_auto_gpu_offset_ms",cpu_fit[3]},
            {"gguf_cpu_auto_pcie_expert_ms",cpu_fit[4]},
            {"gguf_cpu_auto_tail_risk_version",2.},
            {"gguf_cpu_auto_tail_variance_ms2",tail_risk},
            {"gguf_cpu_auto_symmetric_variance_ms2",symmetric_risk},
            {"gguf_cpu_work_ms",cpu_work_ms_},
            {"gguf_cpu_queue_ms",cpu_queue_ms_},
            {"gguf_cpu_jobs",double(cpu_jobs_)},
            {"gguf_cpu_queue_peak_ms",cpu_queue_peak_ms_},
            {"gguf_cpu_queue_delayed_jobs",double(cpu_queue_delayed_jobs_)},
            {"gguf_host_route_group_packets",double(host_route_group_packets_)},
            {"gguf_host_route_group_slots",double(host_route_group_slots_)},
            {"gguf_host_route_group_count",double(host_route_group_count_)},
            {"gguf_host_prepare_calls",double(host_prepare_calls_)},
            {"gguf_host_prepare_resident_lock_ms",resident_lock_ms_},
            {"gguf_host_prepare_resident_events_ms",resident_events_ms_},
            {"gguf_host_prepare_resident_body_ms",resident_body_ms_},
            {"gguf_host_prepare_unique_ms",host_prepare_ms_[0]},
            {"gguf_host_prepare_resident_ms",host_prepare_ms_[1]},
            {"gguf_host_prepare_groups_ms",host_prepare_ms_[2]},
            {"gguf_host_prepare_cpu_ms",host_prepare_ms_[3]},
            {"gguf_host_prepare_fetch_ms",host_prepare_ms_[4]},
            {"gguf_host_prepare_observe_ms",host_prepare_ms_[5]},
            {"expert_batched_upload_packets", double(batch_packets_)},
            {"expert_batched_cpu_copy_ms", batch_copy_ms_},
            {"expert_batched_stage_wait_ms", batch_wait_ms_},
            {"expert_batched_mapped_bytes", double(batch_mapped_bytes_)},
            {"gguf_prefill_dma_bytes",double(prefill_dma_bytes_)},
            {"gguf_prefill_staging_images",double(prefill_staging_images_)},
            {"gguf_prefill_staging_bytes",double(prefill_staging_bytes_)},
            {"gguf_prefill_dma_copies",double(prefill_dma_copies_)},
            {"gguf_prefill_dma_images",double(prefill_dma_images_)},
            {"gguf_prefill_dma_batches",double(prefill_dma_batches_)},
            {"gguf_prefill_dma_batched_ranges",double(prefill_dma_batched_ranges_)},
            {"gguf_prefill_dma_overlap_batches",double(prefill_dma_overlap_batches_)},
            {"gguf_prefill_phased_groups",double(prefill_phased_groups_)},
            {"gguf_speculative_route_rows_committed",double(speculative_committed_)},
            {"gguf_speculative_route_rows_discarded",double(speculative_discarded_)},
            {"gguf_speculative_route_visits",double(speculative_route_visits_)},
            {"gguf_speculative_unique_touches",double(speculative_unique_touches_)},
            {"gguf_speculative_snapshot_touches",double(speculative_snapshot_touches_)},
            {"gguf_speculative_refind_touches",double(speculative_refind_touches_)},
            {"gguf_grouped_prefill_calls", double(grouped_prefills_)},
            {"gguf_direct_expert_calls", double(direct_calls_)},
            {"gguf_direct_expert_bytes", double(direct_bytes_)},
            {"gguf_prefill_resident_experts",double(prefill_resident_experts_)},
            {"gguf_prefill_rebalance_ms",prefill_rebalance_ms_},
            {"expert_cache_hits", double(gpu_hits_)}, {"expert_cache_misses", double(calls_ - gpu_hits_)},
            {"expert_cache_evictions", double(device_evictions_)}, {"expert_host_fetch_ms", fetch_ms_},
            {"host_tensor_bytes", double(mfq::cuda::tensor_host_bytes.load())}};
        for(const auto& [rows,traffic]:direct_by_rows_) {
            const auto prefix="gguf_direct_rows_"+std::to_string(rows)+"_";
            result.emplace_back(prefix+"layer_calls",double(traffic.layers));
            result.emplace_back(prefix+"hits",double(traffic.hits));
            result.emplace_back(prefix+"misses",double(traffic.misses));
            result.emplace_back(prefix+"transfer_bytes",double(traffic.bytes));
        }
        return result;
    }
    void preload(const GgufRoutedWeight& bank,const std::string& name) {
        if(bank.resident_packed.defined())return;
        std::lock_guard lock(mutex_);
        if (bank.expert_bytes > (host_limit_ - std::min(host_used_, host_limit_)) /
                std::uint64_t(bank.experts)) return;
        constexpr auto gate_suffix=".mlp.experts.gate.weight";
        constexpr auto up_suffix=".mlp.experts.up.weight";
        const PreloadedGate* gate=nullptr;
        if(!snapshot_ && name.ends_with(up_suffix)) {
            const auto stem=name.substr(0,name.size()-std::char_traits<char>::length(up_suffix));
            const auto found=preloaded_gates_.find(stem+gate_suffix);
            if(found!=preloaded_gates_.end() && found->second.experts==bank.experts &&
                    found->second.type==bank.type && found->second.bytes%16==0 && bank.expert_bytes%16==0 &&
                    found->second.bytes<=SIZE_MAX-bank.expert_bytes)gate=&found->second;
        }
        for (std::int64_t expert = 0; expert < bank.experts; ++expert) {
            bool paired=false;
            if(gate) {
                const auto key=(gate->source<<32)|std::uint64_t(expert);
                if(const auto found=host_.find(key);found!=host_.end() && found->second.image->block &&
                        !found->second.image->ready) {
                    const auto old=found->second.image;
                    // Only the immutable startup preload moves an image. Both
                    // roles share one bounded allocation; there is no second
                    // persistent RAM copy and graph-visible images never move.
                    if(auto block=allocate_pinned(gate->bytes+bank.expert_bytes)) {
                        auto g=std::make_shared<HostImage>(),u=std::make_shared<HostImage>();
                        g->block=block;g->bytes=gate->bytes;
                        u->block=std::move(block);u->block_offset=gate->bytes;u->bytes=bank.expert_bytes;
                        std::memcpy(g->data(),old->data(),old->bytes);
                        bank.reader(std::uint64_t(expert)*bank.expert_bytes,static_cast<std::byte*>(u->data()),u->bytes);
                        source_bytes_+=u->bytes;++source_reads_;
                        erase_host(key);insert_host(key,g);
                        insert_host((bank.source_id<<32)|std::uint64_t(expert),u);
                        host_gate_up_pair_bytes_+=g->bytes+u->bytes;paired=true;
                    }
                }
            }
            if(!paired)read_host(bank,expert);
        }
        if(name.ends_with(gate_suffix))preloaded_gates_.insert_or_assign(name,
            PreloadedGate{bank.source_id,std::size_t(bank.expert_bytes),bank.experts,bank.type});
    }
    std::vector<GgufWeight> acquire(const GgufRoutedWeight& bank,
        const std::vector<std::int64_t>& selected, int device, UploadBatch* batch = nullptr, bool retain = true,
        const std::vector<std::uint8_t>* admission = nullptr,GgufPrefillWeightStage* staging = nullptr) {
        MfqCudaGuard guard(device);
        std::lock_guard lock(mutex_); reap();
        const auto stream = mfq_current_cuda_stream();
        if(bank.resident_packed.defined()) {
            std::vector<GgufWeight> result;result.reserve(selected.size());
            for(const auto expert:selected) {
                result.push_back(bank.resident_expert(expert,device));
                result.back().packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
                ++calls_;++gpu_hits_;
            }
            return result;
        }
        if (!device_limit_.count(device)) {
            std::size_t available = 0, total = 0;
            TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&available, &total));
            const auto reserve = mfq::cuda::runtime_options::workspace_reserve_bytes();
            device_limit_[device] = configured_device_limit_.value_or(available > reserve ? available - reserve : 0);
        }
        std::vector<GgufWeight> result; result.reserve(selected.size());
        const auto batch_started=std::chrono::steady_clock::now();
        read_host_batch(bank,selected,device);
        fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-batch_started).count();
        for (const auto expert : selected) {
            ++calls_;
            if (expert < 0 || expert >= bank.experts) throw std::out_of_range("GGUF expert id");
            bool keep = retain && (!admission || admission->at(expert));
            if(slot_tiers_.count(device))keep=keep && slot_admitted(bank,expert,device);
            const Key key{(bank.source_id << 32) | std::uint64_t(expert), device};
            if (auto found = device_.find(key); found != device_.end()) {
                if (keep) device_recent_.splice(device_recent_.begin(), device_recent_, found->second.recent);
                ++gpu_hits_;
                if (found->second.producer != stream) {
                    auto& entry=found->second;
                    // Cache entries are immutable until replacement. Once
                    // complete on the host, their producer fence never needs
                    // to be inserted into another PP stream again.
                    if(!entry.ready_on_host && entry.ready->ready())entry.ready_on_host=true;
                    if(entry.ready_on_host)++prefill_device_ready_skips_;
                    else TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,entry.ready->get(),0));
                }
                result.push_back(found->second.weight);
                result.back().packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
                continue;
            }
            const auto started = std::chrono::steady_clock::now();
            const auto image = read_host(bank, expert);
            fetch_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
            // Free the promoted image's logical RAM slot before demoting the
            // VRAM victim, otherwise a full cache discards a different cold
            // image even when the two tiers can hold the entire working set.
            if (keep && image->bytes <= device_limit_.at(device) && host_duplicates_.empty() &&
                image->bytes > host_limit_ - std::min(host_used_, host_limit_)) erase_host(key.image);
            if (keep) evict_device(device, image->bytes);
            tb::Tensor packed;
            const auto allocation_started = std::chrono::steady_clock::now();
            if(keep && slot_tiers_.count(device))packed=slot_image(bank,expert,device);
            else if(!keep && staging && staging->image.defined() &&
                    image->bytes<=staging->image.nbytes()-std::min(staging->cursor,staging->image.nbytes())) {
                if(const auto destination=staging->mmq_destinations.find(key.image);destination!=staging->mmq_destinations.end()) {
                    packed=destination->second;prefill_direct_mmq_bytes_+=image->bytes;
                    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                        static std::atomic<bool> reported{false};
                        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_prefill_direct_mmq actual_match=1 format=%d\n",int(bank.type));
                    }
                }
                else {
                    packed=staging->image.narrow(0,std::int64_t(staging->cursor),std::int64_t(image->bytes));
                    staging->cursor=(staging->cursor+image->bytes+15)&~std::size_t(15);
                }
                ++prefill_staging_images_;prefill_staging_bytes_+=image->bytes;
            } else {
                // Cache images can change with warmup routes. Their lifetime
                // must not consume a temporary slot reserved for graph math.
                // PP images excluded from admission are temporary. Mixing
                // them into the resident pool fragments the decode cache's
                // pages and prevents workspace from returning after PP.
                const mfq::cuda::RoutedWeightAllocationScope resident(!(batch && admission && !keep));
                packed = tb::empty({static_cast<std::int64_t>(image->bytes)},
                    tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
            }
            allocation_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-allocation_started).count();
            if (batch) {
                GgufWeight weight{std::move(packed), bank.type, bank.out, bank.input, bank.row_bytes, bank.rotation};
                weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
                batch->images.push_back({key, weight, image, keep}); result.push_back(std::move(weight));
                continue;
            }
            auto ready = std::make_shared<mfq::cuda::Event>();
            // Retain the host allocation even if a later cache eviction occurs
            // while the upload is still in flight.
            transfers_.push_back({image, ready, {}});
            try {
                const auto before = std::chrono::steady_clock::now();
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(packed.data_ptr(), image->data(), image->bytes,
                    cudaMemcpyHostToDevice, stream));
                upload_enqueue_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-before).count();
                ready->record(stream);
                transfer_bytes_ += image->bytes;
            } catch (...) { (void)cudaStreamSynchronize(stream); transfers_.pop_back(); throw; }
            GgufWeight weight{std::move(packed), bank.type, bank.out, bank.input, bank.row_bytes, bank.rotation};
            weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            if (keep && image->bytes <= device_limit_.at(device)) {
                device_recent_.push_front(key);
                device_.emplace(key, DeviceEntry{weight, ready, stream, device_recent_.begin()});
                device_used_[device] += image->bytes;
                add_device_copy(key.image,key.device);
            }
            result.push_back(std::move(weight));
        }
        return result;
    }
    std::vector<GgufCpuExpert> cpu_candidates(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            const std::int64_t* slots,int count,int device,int limit,int tokens=0,bool half=false,int* misses=nullptr) {
        std::lock_guard lock(mutex_);
        if(cpu_executor_)cpu_executor_->keep_awake();
        const Serving serving(serving_graph_);
        std::vector<std::pair<std::int64_t,int>> ranking;
        for(int i=0;i<count;++i) {
            const auto id=slots[i];
            if(id<0 || id>=banks[0]->experts)throw std::out_of_range("GGUF CPU expert id");
            auto found=std::find_if(ranking.begin(),ranking.end(),[&](auto p){return p.first==id;});
            if(found==ranking.end())ranking.emplace_back(id,1);else ++found->second;
        }
        // Copying one expert costs the same regardless of how many tokens
        // use it. Give the CPU the least-referenced misses so its bounded
        // share finishes sooner; the GPU retains the most weight reuse.
        std::stable_sort(ranking.begin(),ranking.end(),[](auto a,auto b){return a.second<b.second;});
        std::erase_if(ranking,[&](const auto& candidate) {
            for(const auto& bank:banks)if(bank->resident_packed.defined() || device_.count({(bank->source_id<<32)|std::uint64_t(candidate.first),device}))return true;
            return false;
        });
        if(misses) {
            *misses=int(ranking.size());
            auto& model=cpu_balance_[cpu_balance_key(banks,device,tokens,half)];
            cpu_auto_fitted_layers_+=model.ready();
            limit=std::min(limit,model.cpu_count(*misses));
            ++cpu_auto_layers_;cpu_auto_selected_+=limit;cpu_auto_gpu_only_+=limit==0;
        }
        std::vector<GgufCpuExpert> result;
        if(limit<=0)return result;
        bool batch_attempted=false;
        for(const auto [id,uses]:ranking) {
            GgufCpuExpert expert;
            for(int role=0;role<3;++role) {
                const auto& bank=*banks[role];const auto started=std::chrono::steady_clock::now();
                if(!batch_attempted && bank.batch_reader &&
                    !(snapshot_ && snapshot_sources_.count(bank.source_id)) &&
                    !host_.count((bank.source_id<<32)|std::uint64_t(id))) {
                    batch_attempted=true;
                    std::vector<std::int64_t> selected;
                    for(const auto& candidate:ranking) {
                        selected.push_back(candidate.first);if(int(selected.size())==limit)break;
                    }
                    // All selected CPU experts need all three immutable banks.
                    // Allocation while serving only reuses existing pinned
                    // arenas or pageable memory; CUDA registration stays out.
                    for(const auto& candidate:banks)read_host_batch(*candidate,selected,device);
                }
                auto image=read_host(bank,id);
                // CPU consumers require completed D2H data even on the same stream.
                if(image->ready && !image->ready->ready())image->ready->synchronize();
                fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                expert.owners[role]=image;
                expert.projections[role]={static_cast<const std::uint8_t*>(image->data()),bank.row_bytes,
                    int(bank.type),int(bank.out),int(bank.input)};
            }
            for(int i=0;i<count;++i)if(slots[i]==id)expert.slots.push_back(i);
            calls_+=3;
            result.push_back(std::move(expert));if(int(result.size())==limit)break;
        }
        return result;
    }
    void record_cpu_balance(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            int device,int tokens,bool half,int cpu,int pcie,double cpu_ms,double gpu_ms,
            double exposed_ms,double clock_adjust_ms) {
        if(cpu<0 || pcie<0 || !std::isfinite(cpu_ms) || !std::isfinite(gpu_ms) || cpu_ms<0 || gpu_ms<=0 || gpu_ms>=1000)return;
        std::lock_guard lock(mutex_);
        if(!reuse_learning_)return;
        ++cpu_clock_samples_;cpu_exposed_waits_+=exposed_ms>0;
        cpu_exposed_wait_ms_+=exposed_ms;cpu_clock_adjust_ms_+=clock_adjust_ms;
        cpu_balance_[cpu_balance_key(banks,device,tokens,half)].observe(cpu,pcie,cpu_ms,gpu_ms);
        ++cpu_auto_samples_;
    }
    void submit_cpu(std::shared_ptr<GgufCpuWork> work) {
        if(!cpu_executor_)cpu_executor_=std::make_unique<GgufCpuExecutor>(cpu_worker_cores_);
        cpu_executor_->submit(std::move(work));
    }
    std::shared_ptr<void> prepare_host_staging(const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
            const std::vector<std::int64_t>& active,const std::vector<std::size_t>& ends,int device) {
        std::lock_guard lock(mutex_);
        const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_HOST_STAGE");
        const auto* dma=std::getenv("TYLOINFER_GGUF_PREFILL_DMA");
        if(!snapshot_ || (setting && setting[0]=='0') || (dma && dma[0]=='0'))return {};
        if(!staged_host_jobs_.empty())throw std::logic_error("overlapping GGUF host staging");
        std::vector<GgufPrefillStager::Job> jobs;
        std::size_t first=0,maximum=0;
        for(const auto end:ends) {
            for(const auto& bank:banks)for(auto i=first;i<end;++i) {
                if(bank->resident_packed.defined())continue;
                const auto expert=active[i];const Key key{(bank->source_id<<32)|std::uint64_t(expert),device};
                if(device_.count(key))continue;
                auto image=read_host(*bank,expert);if(image->pinned())continue;
                staged_host_jobs_.emplace(key.image,int(jobs.size()));maximum=std::max(maximum,image->bytes);
                jobs.push_back({static_cast<const std::uint8_t*>(image->data()),image->bytes,image});
            }
            first=end;
        }
        if(jobs.empty())return {};
        if(jobs.size()>65535){staged_host_jobs_.clear();return {};}
        if(!prefill_host_stager_ || prefill_host_stager_->device!=device || prefill_host_stager_->capacity<maximum) {
            prefill_host_stager_.reset();
            auto stager=std::make_unique<GgufPrefillStager>();
            const auto hw=std::thread::hardware_concurrency();
            const auto threads=std::max(2u,std::min(4u,hw/4));
            if(!stager->init(maximum,threads)){staged_host_jobs_.clear();return {};}
            prefill_host_stager_=std::move(stager);
        }
        prefill_host_stager_->start(std::move(jobs));
        static std::once_flag report;
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')std::call_once(report,[&] {
            std::fprintf(stderr,"native_gguf_prefill_host_stage actual_match=1 ring=%d jobs=%zu\n",
                prefill_host_stager_->kRing,staged_host_jobs_.size());
        });
        return std::shared_ptr<void>(this,[this](void*) {
            std::lock_guard lock(mutex_);prefill_host_stager_->finish();
            prefill_host_stager_->jobs.clear();staged_host_jobs_.clear();
        });
    }
    void record_cpu(std::size_t experts,double ms,double queue_ms) {
        std::lock_guard lock(mutex_);cpu_experts_+=experts;cpu_work_ms_+=ms;cpu_queue_ms_+=queue_ms;
        ++cpu_jobs_;cpu_queue_peak_ms_=std::max(cpu_queue_peak_ms_,queue_ms);
        cpu_queue_delayed_jobs_+=queue_ms>=1.;
    }
    void record_host_prepare(const std::array<double,6>& times) {
        std::lock_guard lock(mutex_);++host_prepare_calls_;
        for(std::size_t i=0;i<times.size();++i)host_prepare_ms_[i]+=times[i];
    }
    void record_route_groups(std::size_t slots,std::size_t groups) {
        std::lock_guard lock(mutex_);++host_route_group_packets_;host_route_group_slots_+=slots;host_route_group_count_+=groups;
    }
    void record_prefill_inplace(std::size_t bytes) {
        std::lock_guard lock(mutex_);prefill_inplace_mmq_bytes_+=bytes;
    }
    void await_device_host(DeviceEntry& entry,cudaStream_t consumer) {
        // Callers already own their stream guard. Reuse that handle rather
        // than querying the device and global stream registry for every hit.
        if(entry.ready_on_host || entry.producer==consumer)return;
        entry.ready->synchronize();entry.ready_on_host=true;++host_device_ready_waits_;
    }
    std::shared_ptr<GgufResidentView> resident_view(
            const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,int device) {
        wait_direct_admission();
        std::lock_guard lock(mutex_);
        const std::array<std::uint64_t,4> key{banks[0]->source_id,banks[1]->source_id,banks[2]->source_id,std::uint64_t(device)};
        auto& view=resident_views_[key];
        if(!view) {
            view=std::make_shared<GgufResidentView>(banks,device);
            for(int role=0;role<3;++role) {
                for(std::int64_t expert=0;expert<view->experts;++expert) {
                    std::uint64_t pointer=0;
                    if(view->permanent[role])pointer=view->permanent[role]+expert*view->strides[role];
                    else if(auto entry=device_.find(Key{(banks[role]->source_id<<32)|std::uint64_t(expert),device});entry!=device_.end()) {
                        await_device_host(entry->second,mfq_current_cuda_stream());
                        pointer=reinterpret_cast<std::uintptr_t>(entry->second.weight.packed.data_ptr());
                    }
                    view->update(role,expert,pointer);
                }
                resident_roles_[Key{banks[role]->source_id,device}].push_back({view.get(),role});
            }
            ++resident_version_;
        }
        return view;
    }
    void prepare_resident_views(int device,cudaStream_t consumer) {
        {
            std::lock_guard lock(mutex_);
            if(layer_admission_ && layer_admission_->device==device && !layer_admission_->cancelled) {
                if(layer_admission_->error)std::rethrow_exception(layer_admission_->error);
                return;
            }
        }
        wait_direct_admission();
        std::lock_guard lock(mutex_);
        if(prepared_resident_version_[device]==resident_version_)return;
        const auto started=std::chrono::steady_clock::now();
        for(const auto& [key,view]:resident_views_)if(view->device==device && view->rebuild) {
            for(int role=0;role<3;++role)for(std::int64_t expert=0;expert<view->experts;++expert) {
                std::uint64_t pointer=view->permanent[role]?view->permanent[role]+expert*view->strides[role]:0;
                if(!pointer)if(auto entry=device_.find(Key{(view->sources[role]<<32)|std::uint64_t(expert),device});entry!=device_.end()) {
                    await_device_host(entry->second,consumer);
                    pointer=reinterpret_cast<std::uintptr_t>(entry->second.weight.packed.data_ptr());
                }
                view->update(role,expert,pointer);
            }
            view->rebuild=false;
            view->pending_ready.clear();
        }
        // Admission, eviction and workspace loans append changed images to
        // a deduplicated journal, bounded by the bank images. Updating those
        // images avoids an L1-sized scan
        // and a full pointer-table rewrite on every verification window.
        auto& dirty=dirty_resident_images_[device];
        for(const auto image:dirty) {
            const auto roles=resident_roles_.find(Key{image>>32,device});
            if(roles==resident_roles_.end())continue;
            std::uint64_t pointer=0;
            if(auto entry=device_.find(Key{image,device});entry!=device_.end()) {
                await_device_host(entry->second,consumer);
                pointer=reinterpret_cast<std::uintptr_t>(entry->second.weight.packed.data_ptr());
            }
            for(const auto& role:roles->second)role.view->update(role.role,std::uint32_t(image),pointer);
            ++resident_view_updates_;
        }
        dirty.clear();
        prepared_resident_version_[device]=resident_version_;++resident_view_refreshes_;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        resident_view_prepare_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    }
    void acquire_resident_moe(
        const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
        const std::vector<std::int64_t>& experts,int device,cudaStream_t consumer,
        const std::array<std::int64_t*,3>& pointers,std::vector<GgufWeight>& keep,bool profile=false) {
        using Clock=std::chrono::steady_clock;
        const auto entered=profile?Clock::now():Clock::time_point{};
        std::lock_guard lock(mutex_);
        const auto locked=profile?Clock::now():Clock::time_point{};
        double event_ms=0;
        if(profile)resident_lock_ms_+=std::chrono::duration<double,std::milli>(locked-entered).count();
        for(const auto expert:experts) {
            // The resident pass must own all three projections. Keep partial
            // hits in the streamed pass and look each entry up only once.
            std::array<DeviceEntry*,3> entries{};
            bool complete=true;
            for(int role=0;role<3;++role) {
                const auto found=device_.find({(banks[role]->source_id<<32)|std::uint64_t(expert),device});
                entries[role]=found==device_.end()?nullptr:&found->second;
                complete&=banks[role]->resident_packed.defined() || entries[role]!=nullptr;
            }
            for(int role=0;role<3;++role) {
                pointers[role][expert]=0;
                if(!complete)continue;
                if(banks[role]->resident_packed.defined()) {
                    auto weight=banks[role]->resident_expert(expert,device);
                    weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(consumer));
                    pointers[role][expert]=reinterpret_cast<std::intptr_t>(weight.packed.data_ptr());
                    keep.push_back(std::move(weight));continue;
                }
                const bool timed=profile && !entries[role]->ready_on_host && entries[role]->producer!=consumer;
                const auto before=timed?Clock::now():Clock::time_point{};
                await_device_host(*entries[role],consumer);
                if(timed)event_ms+=std::chrono::duration<double,std::milli>(Clock::now()-before).count();
                const auto& weight=entries[role]->weight;
                pointers[role][expert]=reinterpret_cast<std::intptr_t>(weight.packed.data_ptr());
                keep.push_back(weight);
            }
        }
        if(profile) {
            resident_events_ms_+=event_ms;
            resident_body_ms_+=std::chrono::duration<double,std::milli>(Clock::now()-locked).count()-event_ms;
        }
    }
    void acquire_direct(
        const std::array<std::shared_ptr<GgufRoutedWeight>,3>& banks,
        const std::vector<std::int64_t>& experts,int device,cudaStream_t consumer,void* workspace,
        const std::array<std::int64_t*,3>& pointers,
        const std::array<const std::int64_t*,3>& resident,
        std::vector<GgufWeight>& keep,std::vector<std::shared_ptr<void>>& host_keep,
        mfq::cuda::MoeMappedCopyDescriptor* copies,std::size_t& copy_count,std::size_t& copy_bytes,int rows,
        std::size_t* gate_up_bytes=nullptr,GgufDirectWorkspaceImages* workspace_images=nullptr) {
        std::lock_guard lock(mutex_);
        auto& traffic=direct_by_rows_[rows];++traffic.layers;
        const Serving serving(serving_graph_);
        std::size_t pageable_bytes=0;copy_count=copy_bytes=0;
        ++direct_calls_;
        for(int role=0;role<3;++role) {
            if(role==2 && gate_up_bytes)*gate_up_bytes=copy_bytes;
            const auto& bank=*banks[role];
            bool batch_attempted=!bank.batch_reader || (snapshot_ && snapshot_sources_.count(bank.source_id));
            for(std::size_t selected=0;selected<experts.size();++selected) {
                const auto expert=experts[selected];
                ++calls_;
                // Already owned by the resident pass. Preserve the existing
                // counters without another readiness wait or weight handle.
                if(resident[role] && resident[role][expert]) {
                    if(!speculative_rows_) {
                        const Key key{(bank.source_id<<32)|std::uint64_t(expert),device};
                        const auto found=device_.find(key);
                        if(found!=device_.end())device_recent_.splice(device_recent_.begin(),device_recent_,found->second.recent);
                    }
                    pointers[role][expert]=0;++gpu_hits_;++traffic.hits;continue;
                }
                if(bank.resident_packed.defined()) {
                    auto weight=bank.resident_expert(expert,device);
                    weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(consumer));
                    pointers[role][expert]=reinterpret_cast<std::intptr_t>(weight.packed.data_ptr());
                    keep.push_back(std::move(weight));++gpu_hits_;++traffic.hits;continue;
                }
                const Key key{(bank.source_id<<32)|std::uint64_t(expert),device};
                if(auto found=device_.find(key);found!=device_.end()) {
                    // Uploads precede this graph; never enqueue work behind its
                    // host wait. Cross-stream producers are fenced on the CPU.
                    await_device_host(found->second,consumer);
                    if(!speculative_rows_)device_recent_.splice(device_recent_.begin(),device_recent_,found->second.recent);
                    ++gpu_hits_;++traffic.hits;keep.push_back(found->second.weight);
                    pointers[role][expert]=reinterpret_cast<std::intptr_t>(found->second.weight.packed.data_ptr());
                    continue;
                }
                if(workspace_images && workspace_images->contains(selected,role)) {
                    pointers[role][expert]=reinterpret_cast<std::intptr_t>(workspace_images->destination(selected,role));
                    ++gpu_hits_;++traffic.hits;direct_workspace_reused_bytes_+=bank.expert_bytes;
                    continue;
                }
                const auto started=std::chrono::steady_clock::now();
                if(!batch_attempted && !host_.count(key.image)) {
                    batch_attempted=true;
                    std::vector<std::int64_t> pending;
                    for(std::size_t at=selected;at<experts.size();++at)
                        if(!workspace_images || !workspace_images->contains(at,role))pending.push_back(experts[at]);
                    read_host_batch(bank,pending,device);
                }
                auto image=read_host(bank,expert);
                fetch_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                const void* source=nullptr;
                if(image->pinned())source=image->mapped_data(device);
                else {
                    if(image->bytes>direct_capacity-pageable_bytes)throw std::logic_error("GGUF direct host packet overflow");
                    std::memcpy(static_cast<std::uint8_t*>(direct_host_->data())+pageable_bytes,image->data(),image->bytes);
                    source=static_cast<const std::uint8_t*>(direct_mapped_)+pageable_bytes;
                    pageable_bytes=(pageable_bytes+image->bytes+15)&~std::size_t(15);
                }
                if(image->bytes>direct_capacity-copy_bytes)throw std::logic_error("GGUF direct GPU packet overflow");
                auto* destination=workspace_images?workspace_images->destination(selected,role):
                    static_cast<std::uint8_t*>(workspace)+copy_bytes;
                copies[copy_count++]={reinterpret_cast<std::uint64_t>(destination),
                    reinterpret_cast<std::uint64_t>(source),copy_bytes,image->bytes};
                copy_bytes=(copy_bytes+image->bytes+15)&~std::size_t(15);
                if(workspace_images)workspace_images->copied(selected,role);
                // The graph owns the shared destination arena. Only the source
                // image needs a per-window reference; a TensorStorage wrapper
                // for each copied projection adds no lifetime guarantee.
                host_keep.push_back(std::move(image));
                pointers[role][expert]=reinterpret_cast<std::intptr_t>(destination);
                direct_bytes_+=bank.expert_bytes;transfer_bytes_+=bank.expert_bytes;
                ++traffic.misses;traffic.bytes+=bank.expert_bytes;
            }
        }
    }
    std::array<std::vector<GgufWeight>, 3> acquire_moe(
        const std::array<std::shared_ptr<GgufRoutedWeight>, 3>& banks,
        const std::vector<std::int64_t>& experts, int device, bool retain = true,
        const std::vector<std::uint8_t>* admission = nullptr,GgufPrefillWeightStage* staging = nullptr,
        UploadPurpose purpose=UploadPurpose::Streamed) {
        std::array<std::vector<GgufWeight>, 3> result;
        UploadBatch batch;
        const bool staged = bool(upload_stages_[0].host);
        for (int role=0;role<3;++role) result[role]=acquire(*banks[role],experts,device,staged?&batch:nullptr,retain,admission,staging);
        if (batch.images.empty()) return result;
        const MfqCudaGuard guard(device);
        std::lock_guard lock(mutex_); reap();
        const auto stream=mfq_current_cuda_stream();
        std::size_t index=0, offset=0;
        try {
        const bool cache_admission=purpose==UploadPurpose::CacheAdmission;
        const auto* dma_setting=std::getenv(cache_admission?"TYLOINFER_GGUF_ADMISSION_DMA":"TYLOINFER_GGUF_PREFILL_DMA");
        // PP runs outside the host-wait decode graph. Its pinned sources can
        // use the copy engine while the previous expert group computes,
        // instead of competing with GEMM for SMs and mapped PCIe reads.
        // Post-window cache fills also run outside the serving graph;
        // host-served graph packets retain their original mapped-copy path.
        const bool dma=(admission || cache_admission) && (!dma_setting || dma_setting[0]!='0');
        if(dma) {
            if(cache_admission)for(const auto& image:batch.images)if(!image.source->pinned()) {
                // Pageable H2D may synchronize its stream before staging.
                // Reuse the already pinned upload ring and copy directly into
                // the fixed slot; neither pinning nor a scatter kernel may be
                // introduced while a serving graph waits for this layer.
                if(image.source->ready && !image.source->ready->ready())image.source->ready->synchronize();
                for(std::size_t offset=0;offset<image.source->bytes;) {
                    auto& stage=upload_stages_[upload_cursor_++%upload_stages_.size()];
                    const auto waited=std::chrono::steady_clock::now();
                    if(stage.ready && !stage.ready->ready())stage.ready->synchronize();
                    batch_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-waited).count();
                    const auto count=std::min(upload_capacity,image.source->bytes-offset);
                    const auto copied=std::chrono::steady_clock::now();
                    std::memcpy(stage.host->data(),static_cast<const std::uint8_t*>(image.source->data())+offset,count);
                    batch_copy_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-copied).count();
                    const auto issued=std::chrono::steady_clock::now();
                    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(image.weight.packed.data_ptr<std::uint8_t>()+offset,
                        stage.host->data(),count,cudaMemcpyHostToDevice,stream));
                    upload_enqueue_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-issued).count();
                    if(!stage.ready)stage.ready=std::make_shared<mfq::cuda::Event>();
                    stage.ready->record(stream);admission_dma_bytes_+=count;++admission_dma_copies_;offset+=count;
                }
                ++admission_dma_images_;
            }
            const auto down_source=banks[2]->source_id;
            const bool phased=staging && staging->gate_up_ready && !cache_admission &&
                std::all_of(batch.images.begin(),batch.images.end(),[](const auto& image){return image.source->pinned();}) &&
                std::any_of(batch.images.begin(),batch.images.end(),[&](const auto& image){return (image.key.image>>32)==down_source;});
            std::array<std::vector<mfq::cuda::PinnedUploadRange>,2> phases;
            phases[0].reserve(batch.images.size());
            if(phased)phases[1].reserve(experts.size());
            for(const auto& image:batch.images)if(const auto found=staged_host_jobs_.find(image.key.image);found!=staged_host_jobs_.end()) {
                const auto before=std::chrono::steady_clock::now();
                const auto* source=prefill_host_stager_->wait(found->second);
                prefill_host_staging_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(image.weight.packed.data_ptr(),source,image.source->bytes,cudaMemcpyHostToDevice,stream));
                prefill_host_stager_->issued_one(found->second,stream);prefill_host_staged_bytes_+=image.source->bytes;
            }
            for(const auto& image:batch.images)if(image.source->pinned()) {
                if(image.source->ready && image.source->producer!=stream)
                    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,image.source->ready->get(),0));
                auto* destination=static_cast<std::byte*>(image.weight.packed.data_ptr());
                // Only PP slices inside this single known-live allocation
                // share a destination root. Permanent cache images do not.
                const void* destination_root=destination;
                if(staging && staging->image.defined()) {
                    const auto base=reinterpret_cast<std::uintptr_t>(staging->image.data_ptr());
                    const auto address=reinterpret_cast<std::uintptr_t>(destination);
                    const auto capacity=staging->image.nbytes();
                    if(address>=base && address-base<=capacity && image.source->bytes<=capacity-(address-base))
                        destination_root=staging->image.data_ptr();
                }
                const int phase=phased && (image.key.image>>32)==down_source;
                phases[phase].push_back({static_cast<const std::byte*>(image.source->data()),
                    destination,image.source->bytes,image.source->registration_id(),destination_root});
            }
            auto& dma_images=cache_admission?admission_dma_images_:prefill_dma_images_;
            auto& dma_bytes=cache_admission?admission_dma_bytes_:prefill_dma_bytes_;
            auto& dma_copies=cache_admission?admission_dma_copies_:prefill_dma_copies_;
            for(int phase=0;phase<(phased?2:1);++phase) {
                auto& copies=phases[phase];
                dma_images+=copies.size();
                mfq::cuda::coalesce_pinned_uploads(copies);
                const auto before=std::chrono::steady_clock::now();
                bool submitted=false;
                const auto* native=std::getenv("TYLOINFER_MOE_NATIVE_DMA_BATCH");
                const auto mode=native?std::clamp(std::atoi(native),0,2):2;
                // A PP group's active expert IDs are unique, so these ranges are
                // independent. All source/destination owners and the existing
                // transfer fences remain live. CUDA's batch ABI excludes the
                // legacy NULL stream; older drivers use the same individual DMA.
                if(!cache_admission && mode && copies.size()>1 && stream && stream!=cudaStreamLegacy &&
                        mfq::cuda::NativeDmaBatch::available()) {
                    std::vector<mfq::cuda::DmaCopy> commands;commands.reserve(copies.size());
                    for(const auto& span:copies)commands.push_back({span.destination,span.source,span.bytes});
                    submitted=mfq::cuda::NativeDmaBatch::submit(commands,stream,mode,
                        [&](const void* source,std::size_t) -> CUdeviceptr {
                            // Coalescing keeps the first image's source address.
                            // Its registration already provides the mapped alias;
                            // avoid querying CUDA again for every expert upload.
                            for(const auto& image:batch.images)if(image.source->data()==source && image.source->pinned())
                                return reinterpret_cast<CUdeviceptr>(image.source->mapped_data(device));
                            return 0;
                        });
                    if(submitted) {
                        prefill_dma_batches_+=(commands.size()+127)/128;
                        prefill_dma_batched_ranges_+=commands.size();
                        if(mode>=2)prefill_dma_overlap_batches_+=(commands.size()+127)/128;
                    }
                }
                if(!submitted)for(const auto& span:copies)
                    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(span.destination,span.source,span.bytes,cudaMemcpyHostToDevice,stream));
                for(const auto& span:copies){dma_bytes+=span.bytes;++dma_copies;}
                upload_enqueue_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
                if(phased && phase==0) {
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(staging->gate_up_ready,stream));
                    staging->gate_up_submitted=true;++prefill_phased_groups_;
                }
                if(!copies.empty())if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                    static std::atomic<unsigned> reported{0};
                    const unsigned bit=cache_admission?2:1;
                    if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))
                        std::fprintf(stderr,"%s pinned_dma=%d copy_ranges=%zu pageable_stage=retained copy_engine=1\n",
                            cache_admission?"native_gguf_cache_admission_upload actual_match=1":"native_gguf_prefill_upload",
                            int(std::all_of(batch.images.begin(),batch.images.end(),[](const auto& image){return image.source->pinned();})),copies.size());
                }
            }
        }
        while(index<batch.images.size()) {
            if(dma && (cache_admission || batch.images[index].source->pinned() || staged_host_jobs_.count(batch.images[index].key.image))){++index;continue;}
            auto& stage=upload_stages_[upload_cursor_++%upload_stages_.size()];
            auto waited=std::chrono::steady_clock::now();
            if(stage.ready && !stage.ready->ready()) stage.ready->synchronize();
            batch_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-waited).count();
            if(!stage.device.defined() || stage.ordinal!=device) {
                const mfq::cuda::ResidentAllocationScope resident;
                stage.device=tb::empty({std::int64_t(upload_capacity)},
                    tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
                stage.ordinal=device;stage.ready=std::make_shared<mfq::cuda::Event>();
            }
            const auto descriptors=std::min<std::size_t>(256,batch.images.size()-index);
            std::size_t cursor=32+descriptors*sizeof(mfq::cuda::MoeMappedCopyDescriptor), chunks=0, total=0;
            auto* packet=static_cast<std::uint8_t*>(stage.host->data());
            auto* header=reinterpret_cast<std::uint64_t*>(packet);
            auto* jobs=reinterpret_cast<mfq::cuda::MoeMappedCopyDescriptor*>(packet+32);
            while(index<batch.images.size() && chunks<descriptors && cursor+16<=upload_capacity) {
                const auto& image=batch.images[index];
                if(dma && (image.source->pinned() || staged_host_jobs_.count(image.key.image))){++index;continue;}
                std::size_t count=image.source->bytes-offset;
                const void* source=nullptr;
                if(image.source->pinned()) {
                    if(image.source->ready && image.source->producer!=stream)
                        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,image.source->ready->get(),0));
                    source=image.source->mapped_data(device)+offset;batch_mapped_bytes_+=count;
                } else {
                    waited=std::chrono::steady_clock::now();
                    if(image.source->ready && !image.source->ready->ready())image.source->ready->synchronize();
                    batch_wait_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-waited).count();
                    count=std::min(count,upload_capacity-cursor);
                    const auto copied=std::chrono::steady_clock::now();
                    std::memcpy(packet+cursor,static_cast<const std::uint8_t*>(image.source->data())+offset,count);
                    batch_copy_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-copied).count();
                    source=stage.device.data_ptr<std::uint8_t>()+cursor;cursor=(cursor+count+15)&~std::size_t(15);
                }
                jobs[chunks++]={reinterpret_cast<std::uint64_t>(image.weight.packed.data_ptr<std::uint8_t>()+offset),
                    reinterpret_cast<std::uint64_t>(source),total,count};
                total=(total+count+15)&~std::size_t(15);
                offset+=count;
                if(offset==image.source->bytes) {offset=0;++index;}
            }
            header[0]=chunks;header[1]=total;header[2]=header[3]=0;
            const auto before=std::chrono::steady_clock::now();
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage.device.data_ptr(),packet,cursor,cudaMemcpyHostToDevice,stream));
            upload_enqueue_ms_+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-before).count();
            mfq::cuda::moe_copy_mapped(stage.device.data_ptr(),stage.device.data_ptr<std::uint32_t>()+4,
                upload_blocks_,upload_threads_,stream);
            TYLOINFER_CUDA_CHECK(cudaGetLastError());
            ++batch_packets_;
            stage.device.record_stream(reinterpret_cast<std::uintptr_t>(stream));stage.ready->record(stream);
        }
        auto ready=std::make_shared<mfq::cuda::Event>();ready->record(stream);
        for(const auto& image:batch.images) {
            transfer_bytes_+=image.source->bytes;
            if(image.source->pinned() || (cache_admission && dma))transfers_.push_back({image.source,ready,image.weight.packed});
            if(!image.retain) continue;
            evict_device(device,image.source->bytes);
            if(image.source->bytes<=device_limit_.at(device) && !device_.count(image.key)) {
                device_recent_.push_front(image.key);
                device_.emplace(image.key,DeviceEntry{image.weight,ready,stream,device_recent_.begin()});
                device_used_[device]+=image.source->bytes;add_device_copy(image.key.image,image.key.device);
            }
        }
        } catch(...) {
            // All source and destination owners must survive any already
            // queued scatter, including a failure in a later packet.
            (void)cudaStreamSynchronize(stream);throw;
        }
        return result;
    }
};

std::size_t gguf_expert_cache_budget_bytes(const std::shared_ptr<GgufExpertCache>& cache) {
    return cache ? cache->budget_bytes() : 0;
}
std::size_t gguf_prefill_lendable_bytes(const std::shared_ptr<GgufExpertCache>& cache) {
    return cache?cache->lendable_bytes():0;
}
GgufPrefillWorkspaceLoan::~GgufPrefillWorkspaceLoan() {
    try {finish();}catch(const std::exception& error) {std::fprintf(stderr,"GGUF prefill restore failed: %s\n",error.what());}
}
std::size_t GgufPrefillWorkspaceLoan::begin(std::size_t bytes) {
    const auto lent=cache_?cache_->begin_prefill_loan(bytes):0;active_=lent>0;return lent;
}
void GgufPrefillWorkspaceLoan::suspend() {if(active_)cache_->suspend_prefill_loan();}
void GgufPrefillWorkspaceLoan::finish() {
    if(!active_)return;
    cache_->end_prefill_loan();active_=false;
}
std::uint64_t gguf_expert_streamed_bytes(const std::shared_ptr<GgufExpertCache>& cache) {
    return cache ? cache->streamed_bytes() : 0;
}
double gguf_expert_fetch_ms(const std::shared_ptr<GgufExpertCache>& cache) {
    return cache?cache->fetch_ms():0;
}
void resize_gguf_expert_cache(const std::shared_ptr<GgufExpertCache>& cache, std::size_t bytes) {
    if (cache) cache->resize(bytes);
}
std::size_t reserve_gguf_expert_host(const std::shared_ptr<GgufExpertCache>& cache, std::size_t bytes) {
    return cache ? cache->reserve_host(bytes) : 0;
}
std::vector<std::pair<std::string, double>> gguf_expert_memory_metrics(const std::shared_ptr<GgufExpertCache>& cache) {
    return cache ? cache->memory_metrics() : std::vector<std::pair<std::string, double>>{};
}
bool prepare_gguf_graph_pool(const std::shared_ptr<GgufExpertCache>& cache) {
    if (!cache) return false;
    cache->prepare_graph_pool(); return true;
}
void finalize_gguf_expert_cache(const std::shared_ptr<GgufExpertCache>& cache, std::size_t reserve) {
    if (cache) cache->finalize(reserve);
}
void reserve_gguf_expert_workspace(const std::shared_ptr<GgufExpertCache>& cache, std::size_t bytes) {
    if (cache) cache->reserve_workspace(bytes);
}
void restore_gguf_decode_workspace(const std::shared_ptr<GgufExpertCache>& cache, std::size_t growth,
        std::optional<std::size_t> transient) {
    if(cache)cache->restore_workspace(growth,transient);
}
void restore_gguf_decode_workspace_after_capture(const std::shared_ptr<GgufExpertCache>& cache) {
    if(cache)cache->restore_after_capture();
}
void begin_gguf_speculative_routes(const std::shared_ptr<GgufExpertCache>& cache,int rows) {
    if(cache)cache->begin_speculative_routes(rows);
}
void commit_gguf_speculative_routes(const std::shared_ptr<GgufExpertCache>& cache,int rows) {
    if(cache)cache->resolve_speculative_routes(rows);
}
void wait_gguf_expert_admission(const std::shared_ptr<GgufExpertCache>& cache) {
    if(cache)cache->wait_direct_admission();
}
void discard_gguf_speculative_routes(const std::shared_ptr<GgufExpertCache>& cache) noexcept {
    if(cache)cache->discard_speculative_routes();
}
bool set_gguf_reuse_learning(const std::shared_ptr<GgufExpertCache>& cache,bool enabled) {
    return cache?cache->set_reuse_learning(enabled):false;
}

GgufRoutedWeight::GgufRoutedWeight(CudaExecutionContext& execution,
    const mfq::ModelSource& source, const std::string& name) {
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 3)
        throw std::invalid_argument("GGUF expert bank must be [experts,out,in]: " + name);
    type = mfq::gguf::quant_type(record->dtype);
    experts = record->shape->at(0); out = record->shape->at(1); input = record->shape->at(2);
    rotation=load_gguf_input_rotation(source,name,input,0,input,mfq_current_cuda_device(),true);
    row_bytes = mfq::gguf::row_bytes(type, input);
    if (experts <= 0 || experts > UINT32_MAX || out <= 0 ||
            std::uint64_t(out) > UINT64_MAX / row_bytes)
        throw std::invalid_argument("GGUF expert bank dimensions overflow");
    expert_bytes = std::uint64_t(out) * row_bytes;
    if (expert_bytes > INT64_MAX) throw std::overflow_error("GGUF expert exceeds tensor size limits");
    if (std::uint64_t(experts) > UINT64_MAX / expert_bytes ||
            record->nbytes != std::uint64_t(experts) * expert_bytes)
        throw std::invalid_argument("GGUF expert bank payload size mismatch");
    reader = source.tensor_reader(name);
    batch_reader = source.tensor_batch_reader(name);
    const bool resident=name.starts_with("predictor.");
    if(resident) {
        if(record->nbytes>INT64_MAX)throw std::overflow_error("GGUF resident expert bank exceeds tensor size limits");
        const mfq::cuda::WeightAllocationScope weight_memory;
        const mfq::cuda::ResidentAllocationScope permanent;
        const int device=mfq_current_cuda_device();
        resident_packed=tb::empty({std::int64_t(record->nbytes)},
            tb::TensorOptions().device(tb::Device(tb::kCUDA,device)).dtype(tb::kUInt8));
        // Preserve the original compressed encoding. Bounded staging avoids
        // keeping a second host copy of a fully resident predictor bank.
        constexpr std::size_t chunk_bytes=4ULL<<20;
        auto host=tb::empty({std::int64_t(std::min<std::uint64_t>(record->nbytes,chunk_bytes))},
            tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
        for(std::uint64_t offset=0;offset<record->nbytes;) {
            const auto bytes=std::min<std::uint64_t>(record->nbytes-offset,host.nbytes());
            reader(offset,reinterpret_cast<std::byte*>(host.data_ptr()),bytes);
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(resident_packed.data_ptr<std::uint8_t>()+offset,
                host.data_ptr(),bytes,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            offset+=bytes;
        }
        std::fprintf(stderr,"native_gguf_predictor_resident actual_match=1 bank=%s bytes=%llu device=%d\n",
            name.c_str(),static_cast<unsigned long long>(record->nbytes),device);
    }
    if (!execution.gguf_expert_cache) execution.gguf_expert_cache = std::make_shared<GgufExpertCache>(execution,source);
    cache = execution.gguf_expert_cache;
    if(resident) {
        const mfq::cuda::ResidentAllocationScope permanent;
        auto addresses=tb::empty({experts},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64));
        for(std::int64_t expert=0;expert<experts;++expert)
            addresses.data_ptr<std::int64_t>()[expert]=reinterpret_cast<std::intptr_t>(resident_packed.data_ptr<std::uint8_t>()+expert*expert_bytes);
        resident_addresses=addresses.to(resident_packed.device());
    }
    // Lab path: cold file-backed page faults can cost more than pageable D2H.
    // Keep the measured transfer path until mapped residency is accounted for.
    const auto* mapped_recovery=std::getenv("TYLOINFER_GGUF_MAPPED_RECOVERY");
    auto recovery=mapped_recovery && mapped_recovery[0]=='1'?source.mapped_tensor_reader(name):mfq::ModelSource::TensorReader{};
    source_id=cache->register_source(name,record->nbytes,std::move(recovery),resident);cache->preload(*this,name);
}

GgufWeight GgufRoutedWeight::resident_expert(std::int64_t expert,int device) const {
    if(!resident_packed.defined() || resident_packed.get_device()!=device)
        throw std::invalid_argument("GGUF resident expert requested on a different device");
    if(expert<0 || expert>=experts)throw std::out_of_range("GGUF resident expert id");
    return {resident_packed.narrow(0,std::int64_t(std::uint64_t(expert)*expert_bytes),std::int64_t(expert_bytes)),
        type,out,input,row_bytes,rotation};
}

namespace {
struct GgufGraphLease {
    std::vector<GgufWeight> weights;
    std::vector<std::shared_ptr<void>> host_owners;
    tb::Tensor direct_workspace;
    std::shared_ptr<GgufDirectArena> direct_arena;
    std::size_t direct_stages=0;
    // Segmented projection uploads may evict L1 between graph segments.
    // Such a window uses the per-layer host-owned resident table instead.
    bool mutable_residency=false;
    void clear() {weights.clear();host_owners.clear();}
};
const char gguf_graph_lease_identity = 0;
const std::int8_t* rotation_signs(const std::shared_ptr<const GgufInputRotation>& rotation,
        const tb::Tensor& input,cudaStream_t stream) {
    if(!rotation || !rotation->signs_device.defined())return nullptr;
    if(rotation->signs_device.device()!=input.device())throw std::invalid_argument("GGUF expert Hadamard device differs");
    rotation->signs_device.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    return rotation->signs_device.data_ptr<std::int8_t>();
}
void prepare_bank_input(const std::shared_ptr<const GgufInputRotation>& rotation,tb::Tensor input,
        tb::Tensor& packed,float* cpu,cudaStream_t stream) {
    using namespace mfq::cuda::gguf;
    if(input.scalar_type()!=tb::kFloat16 && input.scalar_type()!=tb::kFloat32 && input.scalar_type()!=tb::kBFloat16)
        input=input.to(tb::kFloat32);
    const auto type=input.scalar_type()==tb::kFloat16?InputType::Float16:
        input.scalar_type()==tb::kBFloat16?InputType::BFloat16:InputType::Float32;
    if(rotation && try_prepare_rotated_moe_input(input.data_ptr(),type,input.size(0),input.size(1),
            input.stride(0),input.stride(1),rotation_signs(rotation,input,stream),rotation->block,packed.data_ptr(),cpu,stream))return;
    if(rotation)input=rotation->apply(input);
    prepare_moe_input(input.data_ptr(),input.scalar_type()==tb::kFloat16?InputType::Float16:
        input.scalar_type()==tb::kBFloat16?InputType::BFloat16:InputType::Float32,input.size(0),input.size(1),
        input.stride(0),input.stride(1),packed.data_ptr(),cpu,stream);
    input.record_stream(reinterpret_cast<std::uintptr_t>(stream));
}
void prepare_bank_hidden(const std::shared_ptr<const GgufInputRotation>& rotation,
        const tb::Tensor& gate,const tb::Tensor& up,tb::ScalarType dtype,float limit,tb::Tensor& packed,cudaStream_t stream) {
    using namespace mfq::cuda::gguf;
    const auto columns=gate.size(-1),rows=gate.numel()/columns;
    if(dtype==tb::kFloat16 || dtype==tb::kFloat32) {
        if(!rotation) {quantize_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,
            dtype==tb::kFloat16,limit,packed.data_ptr(),stream);return;}
        if(try_quantize_rotated_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,
                dtype==tb::kFloat16,limit,rotation_signs(rotation,gate,stream),rotation->block,packed.data_ptr(),stream))return;
        auto hidden=tb::empty(gate.sizes(),gate.options().dtype(dtype));
        materialize_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,columns,dtype==tb::kFloat16,limit,hidden.data_ptr(),stream);
        hidden=rotation->apply(hidden).to(tb::kFloat32);
        quantize_input(hidden.data_ptr<float>(),rows,columns,packed.data_ptr(),stream);
        hidden.record_stream(reinterpret_cast<std::uintptr_t>(stream));return;
    }
    auto g=gate.to(dtype),u=up.to(dtype);
    if(limit>0){g=tb::clamp_max(g.to(tb::kFloat32),limit);u=tb::clamp(u.to(tb::kFloat32),-limit,limit);}
    auto hidden=((g*tb::sigmoid(g))*u).to(dtype);
    if(rotation)hidden=rotation->apply(hidden);
    hidden=hidden.to(tb::kFloat32).contiguous();
    quantize_input(hidden.data_ptr<float>(),rows,columns,packed.data_ptr(),stream);
    hidden.record_stream(reinterpret_cast<std::uintptr_t>(stream));
}
struct GgufExpertStage {
    const GgufRoutedWeight& bank;
    std::shared_ptr<GgufGraphLease> lease;
    MfqCudaStream stream;
    tb::Tensor addresses, host_addresses, ids, input, quantized, output;
    std::unique_ptr<mfq::cuda::HostBuffer> host_ids;
    std::int64_t tokens, routes;
    bool routed;
    GgufExpertStage(const GgufRoutedWeight& bank, const tb::Tensor& source,
        std::int64_t tokens, std::int64_t routes, std::shared_ptr<GgufGraphLease> lease)
        : bank(bank), lease(std::move(lease)), stream(mfq_get_current_cuda_stream(source.get_device())),
          tokens(tokens), routes(routes), routed(source.dim() == 3) {
        if(!bank.resident_packed.defined())this->lease->mutable_residency=true;
        const auto options = source.options();
        addresses = tb::zeros({bank.experts}, options.dtype(tb::kInt64));
        host_addresses = tb::zeros({bank.experts}, options.device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
        ids = tb::zeros({tokens, routes}, options.dtype(tb::kInt64));
        input = tb::zeros(source.sizes(), options.dtype(tb::kFloat32));
        quantized = tb::empty({static_cast<std::int64_t>((routed ? tokens * routes : tokens) *
            mfq::cuda::gguf::q8_row_bytes(bank.input))}, options.dtype(tb::kUInt8));
        output = tb::zeros({tokens, routes, bank.out}, options.dtype(tb::kFloat32));
        host_ids = std::make_unique<mfq::cuda::HostBuffer>(tokens * routes * sizeof(std::int64_t));
    }
    ~GgufExpertStage() { try { MfqCudaGuard guard(stream.device()); (void)cudaStreamSynchronize(stream.stream()); } catch (...) {} }
    void load() {
        MfqCudaStreamGuard guard(stream);
        const auto* values = static_cast<const std::int64_t*>(host_ids->data());
        std::vector<std::int64_t> unique;
        std::unordered_set<std::int64_t> visited;
        for (std::int64_t i = 0; i < tokens * routes; ++i)
            if (values[i] >= 0 && values[i] < bank.experts && visited.insert(values[i]).second) unique.push_back(values[i]);
        lease->clear();
        lease->weights = bank.cache->acquire(bank, unique, input.get_device());
        auto* pointers = host_addresses.data_ptr<std::int64_t>();
        for (std::size_t i = 0; i < unique.size(); ++i)
            pointers[unique[i]] = reinterpret_cast<std::intptr_t>(lease->weights[i].packed.data_ptr());
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(addresses.data_ptr(), host_addresses.data_ptr(), addresses.nbytes(),
            cudaMemcpyHostToDevice, stream.stream()));
    }
    void compute() {
        mfq::cuda::gguf::quantize_input(input.data_ptr<float>(), routed ? tokens * routes : tokens,
            bank.input, quantized.data_ptr(), stream.stream());
        mfq::cuda::gguf::routed_matvec_q8(bank.type,
            reinterpret_cast<const std::uint64_t*>(addresses.data_ptr()), bank.row_bytes,
            bank.experts, bank.out, bank.input, ids.data_ptr<std::int64_t>(), tokens, routes,
            quantized.data_ptr(), routed, output.data_ptr<float>(), stream.stream());
    }
};

// A route selects a complete expert, rather than three independent projections.
// Share its host handoff and the Q8 activation packet across Gate and Up. As in
// the TQ/Metal path, all three banks retain their own quantization and layout.
struct GgufMoeStage {
    std::array<std::shared_ptr<GgufRoutedWeight>, 3> banks;
    std::array<tb::Tensor, 3> addresses, host_addresses;
    std::shared_ptr<GgufGraphLease> lease;
    MfqCudaStream stream;
    std::unique_ptr<mfq::cuda::HostBuffer> host_ids;
    tb::Tensor input, ids, weights, q8_input, q8_up, q8_hidden, gate, up, pairs,route_groups;
    std::vector<std::int64_t> observation_routes;
    bool shared_input_basis=true;
    std::unique_ptr<mfq::cuda::HostBuffer> cpu_buffer;
    void* cpu_mapped=nullptr;
    std::size_t cpu_input_offset=0,cpu_up_offset=0,cpu_output_offset=0,cpu_slots_offset=0;
    int cpu_max_experts=0;
    bool cpu_auto=false,cpu_balance_pending=false;
    int cpu_auto_misses=0,cpu_auto_count=0;
    std::chrono::steady_clock::time_point cpu_reference_at;
    std::shared_ptr<GgufCpuWork> cpu_work;
    void* mapped_routes = nullptr;
    bool direct=false;
    bool resident_overlap=false;
    std::shared_ptr<GgufDecodeDma> decode_dma;
    const bool profile_host_prepare=[] {
        const auto* value=std::getenv("TYLOINFER_PROFILE_GGUF_HOST_PREPARE");
        return value && value[0]=='1';
    }();
    MfqCudaStream transfer_stream;
    std::unique_ptr<mfq::cuda::Event> copy_start,copy_gate_up_done,copy_done;
    tb::Tensor resident_groups,resident_device_addresses,streamed_device_addresses;
    std::shared_ptr<GgufResidentView> resident_view;
    std::size_t resident_table_offset=0;
    tb::Tensor copy_descriptors;
    int copy_blocks=0,copy_threads=0;
    std::size_t copy_offset=0,route_group_offset=0;
    std::int64_t tokens, routes;
    std::vector<std::int64_t> selected;
    tb::ScalarType output_type;
    GgufMoeStage(const GgufMoeBanks& banks, const tb::Tensor& source, std::int64_t routes,
        std::shared_ptr<GgufGraphLease> lease)
        : banks{banks.gate, banks.up, banks.down}, lease(std::move(lease)),
          stream(mfq_get_current_cuda_stream(source.get_device())), tokens(source.size(0)),
          routes(routes), output_type(source.scalar_type()) {
        const auto options = source.options();
        selected.reserve(std::size_t(std::min(tokens*routes,this->banks[0]->experts)));
        // Initialize admission before any graph host-wait callback. A short
        // first prompt may reach direct decode without a preceding PP acquire.
        for(const auto& bank:this->banks)bank->cache->prepare_device(source.get_device());
        for (int role = 0; role < 3; ++role) {
            addresses[role] = tb::zeros({this->banks[role]->experts}, options.dtype(tb::kInt64));
            host_addresses[role] = tb::zeros({this->banks[role]->experts},
                options.device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
        }
        input = source;
        shared_input_basis=same_gguf_rotation(banks.gate->rotation,banks.up->rotation);
        q8_input = tb::empty({std::int64_t(tokens * mfq::cuda::gguf::q8_row_bytes(banks.up->input))},
            options.dtype(tb::kUInt8));
        q8_up=shared_input_basis?q8_input:tb::empty_like(q8_input);
        q8_hidden = tb::empty({std::int64_t(tokens * routes * mfq::cuda::gguf::q8_row_bytes(banks.up->out))},
            options.dtype(tb::kUInt8));
        gate = tb::empty({tokens, routes, banks.up->out}, options.dtype(tb::kFloat32));
        up = tb::empty_like(gate);
        pairs = tb::empty({tokens, routes, banks.down->out}, options.dtype(tb::kFloat32));
        // The compact-group counter uses a signed 32-bit atomic.
        const auto* reuse=std::getenv("TYLOINFER_GGUF_ROUTED_REUSE");
        const bool prepared=std::any_of(this->banks.begin(),this->banks.end(),[](const auto& bank){
            return mfq::cuda::gguf::supports_shared_routed_decode(bank->type);});
        // Verification windows share a bounded four-entry tile per expert;
        // wider windows add groups without increasing the kernel's live tile.
        const bool shared_decode=reuse?reuse[0]=='1':tokens<=8 && prepared;
        if(tokens>1 && routes<=INT_MAX/tokens && shared_decode)
            route_groups=tb::empty({std::int64_t((tokens*routes+1)*sizeof(mfq::cuda::gguf::RouteGroup))},options.dtype(tb::kUInt8));
        int can_map = 0;
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, source.get_device()));
        const auto unique=std::min(tokens*routes,banks.gate->experts);
        std::size_t expert_bytes=0;
        bool fits=true;
        for(const auto& bank:this->banks) {
            if(bank->expert_bytes>SIZE_MAX-15 || ((bank->expert_bytes+15)&~std::uint64_t(15))>SIZE_MAX-expert_bytes){fits=false;break;}
            expert_bytes+=(bank->expert_bytes+15)&~std::uint64_t(15);
        }
        direct=fits && unique>0 && can_map && banks.gate->cache==banks.up->cache && banks.up->cache==banks.down->cache &&
            expert_bytes<=SIZE_MAX/std::size_t(unique) && banks.gate->cache->direct_supported(expert_bytes*unique);
        if(direct) {
            try {
            const auto capacity=std::int64_t(expert_bytes*unique);
            if(!this->lease->direct_workspace.defined() || this->lease->direct_workspace.numel()<capacity) {
                this->lease->direct_arena=this->banks[0]->cache->direct_arena(source.get_device(),stream.stream(),std::size_t(capacity));
                this->lease->direct_workspace=this->lease->direct_arena->storage;
            }
            this->lease->direct_workspace.record_stream(reinterpret_cast<std::uintptr_t>(stream.stream()));
            TYLOINFER_CUDA_CHECK(mfq::cuda::moe_mapped_copy_launch_geometry(&copy_blocks,&copy_threads));
            copy_descriptors=tb::empty({std::int64_t(32+3*unique*sizeof(mfq::cuda::MoeMappedCopyDescriptor))},options.dtype(tb::kUInt8));
            ++this->lease->direct_stages;
            if(this->lease->direct_stages>1)this->lease->direct_arena->images.invalidate();
            } catch(const MfqResourceExhausted&) {
                // The segmented path can operate without a shared transfer
                // workspace. Keep it available under a tighter VRAM budget.
                direct=false;copy_descriptors={};(void)cudaGetLastError();
            }
        }
        const auto* overlap=std::getenv("TYLOINFER_GGUF_RESIDENT_OVERLAP");
        // Mapped route publication and a separate stream are the requirements
        // for the same fork/join in eager execution and captured graphs.
        resident_overlap=direct && (!overlap || overlap[0]!='0');
        if(!direct)this->lease->mutable_residency=true;
        if(resident_overlap) {
            transfer_stream=this->banks[0]->cache->prefill_transfer_stream(source.get_device());
            resident_overlap=transfer_stream.stream()!=stream.stream();
            if(resident_overlap) {
                // A PCIe reader needs outstanding loads, not every resident
                // warp. The standalone occupancy grid otherwise blocks the
                // expert kernels that this stream is meant to overlap.
                int multiprocessors=0;
                TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors,cudaDevAttrMultiProcessorCount,source.get_device()));
                copy_blocks=std::min(copy_blocks,multiprocessors);
                copy_threads=std::min(copy_threads,128);
                copy_start=std::make_unique<mfq::cuda::Event>();copy_gate_up_done=std::make_unique<mfq::cuda::Event>();copy_done=std::make_unique<mfq::cuda::Event>();
                if(route_groups.defined())resident_groups=tb::empty_like(route_groups);
                resident_device_addresses=tb::empty({3*this->banks[0]->experts},options.dtype(tb::kInt64));
                streamed_device_addresses=tb::empty_like(resident_device_addresses);
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                    static std::atomic<bool> reported{false};
                    if(!reported.exchange(true,std::memory_order_relaxed))
                        std::fprintf(stderr,"native_gguf_resident_overlap actual_match=1 separate_stream=1 split_ffn=1 copy_blocks=%d copy_threads=%d\n",copy_blocks,copy_threads);
                }
            }
        }
        const auto* dma_setting=std::getenv("TYLOINFER_GGUF_DECODE_DMA");
        if(resident_overlap && (dma_setting?dma_setting[0]=='1':GgufDecodeDma::default_enabled()))
            decode_dma=this->banks[0]->cache->decode_dma(source.get_device());
        copy_offset=128+tokens*routes*sizeof(std::int64_t)+(can_map?3*this->banks[0]->experts*sizeof(std::int64_t):0);
        const auto descriptor_bytes=direct?32+3*unique*sizeof(mfq::cuda::MoeMappedCopyDescriptor):0;
        route_group_offset=copy_offset+descriptor_bytes;
        resident_table_offset=route_group_offset+(direct && route_groups.defined()?route_groups.nbytes():0);
        resident_table_offset=(resident_table_offset+7)&~std::size_t(7);
        host_ids = std::make_unique<mfq::cuda::HostBuffer>(resident_table_offset+
            (resident_overlap?3*this->banks[0]->experts*sizeof(std::int64_t):0),can_map!=0);
        std::memset(host_ids->data(),0,host_ids->size());
        if (can_map) TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped_routes, host_ids->data(), 0));
        if(resident_overlap)resident_view=this->banks[0]->cache->resident_view(this->banks,stream.device());
        mfq::cuda::charge_tensor_host_bytes(host_ids->size());
    }
    ~GgufMoeStage() {
        try {wait_cpu();}catch(...) {}
        try { MfqCudaGuard guard(stream.device()); (void)cudaStreamSynchronize(stream.stream()); } catch (...) {}
        if(cpu_buffer)mfq::cuda::tensor_host_bytes.fetch_sub(cpu_buffer->size());
        mfq::cuda::tensor_host_bytes.fetch_sub(host_ids->size());
    }
    void enable_cpu() {
        // Experimental until its CPU/GPU overlap wins in complete inference.
        const auto* setting=std::getenv("TYLOINFER_GGUF_CPU_EXPERTS");
        // Explicit tuning may reserve CPU work for windows that reuse weight
        // decoding. The default retains all supported token counts; CPU speed
        // and transfer bandwidth are not inferred from the development host.
        const auto* minimum_setting=std::getenv("TYLOINFER_GGUF_CPU_EXPERTS_MIN_TOKENS");
        const int minimum_tokens=minimum_setting?std::max(1,std::atoi(minimum_setting)):1;
        const bool automatic=setting && std::strcmp(setting,"auto")==0;
        const int requested=automatic?int(std::min(tokens*routes,banks[0]->experts)):setting?std::atoi(setting):0;
        if(requested<=0 || !direct || !mfq::cpu::gguf::available() || tokens<minimum_tokens || tokens>4 ||
            routes>INT_MAX/tokens || (output_type!=tb::kFloat16 && output_type!=tb::kFloat32))return;
        for(int role=0;role<3;++role)if(banks[role]->input>INT_MAX || banks[role]->out>INT_MAX ||
            !(role==2?mfq::cpu::gguf::down_supported(banks[role]->type):mfq::cpu::gguf::input_supported(banks[role]->type)))return;
        if(banks[0]->input%256 || banks[1]->input%256 || banks[2]->input%32 ||
            (banks[2]->type==GGML_TYPE_Q2_0 && banks[2]->input%64))return;
        cpu_max_experts=std::min<int>(requested,int(std::min(tokens*routes,banks[0]->experts)));
        cpu_slots_offset=copy_descriptors.numel();
        copy_descriptors=tb::empty({std::int64_t(cpu_slots_offset+tokens*routes*sizeof(std::int32_t))},copy_descriptors.options());
        cpu_input_offset=(64+tokens*routes*sizeof(std::int32_t)+127)&~std::size_t(127);
        cpu_up_offset=shared_input_basis?cpu_input_offset:(cpu_input_offset+input.numel()*sizeof(float)+127)&~std::size_t(127);
        cpu_output_offset=(cpu_up_offset+input.numel()*sizeof(float)+127)&~std::size_t(127);
        const auto bytes=cpu_output_offset+pairs.numel()*sizeof(float);
        try {cpu_buffer=std::make_unique<mfq::cuda::HostBuffer>(bytes,true);}
        catch(const MfqResourceExhausted&) {
            // CPU offload is optional. Preserve the GPU path when late graph
            // preparation cannot pin another handoff buffer beside KV/weights.
            cpu_max_experts=0;return;
        }
        std::memset(cpu_buffer->data(),0,bytes);*static_cast<std::uint32_t*>(cpu_buffer->data())=1;
        mfq::cuda::charge_tensor_host_bytes(bytes);
        TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&cpu_mapped,cpu_buffer->data(),0));
        cpu_auto=automatic;
    }
    void wait_cpu() {
        double elapsed=0,latency=0;
        if(cpu_work) {
            auto work=std::move(cpu_work);
            try{work->future.get();}catch(...){cpu_balance_pending=false;throw;}
            elapsed=work->elapsed_ms;
            // Resident GPU work starts when its table is published, before
            // CPU selection and host-weight preparation. Fit both lanes from
            // that same origin, including work before the CPU queue submit.
            // With no resident overlap the table is published after submit;
            // already-completed CPU work then has zero remaining latency.
            latency=std::max(0.,std::chrono::duration<double,std::milli>(work->completed_at-cpu_reference_at).count());
            banks[0]->cache->record_cpu(work->experts.size(),elapsed,work->queue_ms);
        }
        if(cpu_auto && cpu_balance_pending) {
            cpu_balance_pending=false;
            const auto* clocks=reinterpret_cast<const volatile std::uint64_t*>(static_cast<const std::uint8_t*>(cpu_buffer->data())+8);
            const auto begin=clocks[0],end=clocks[1];
            const auto waited=clocks[2];
            if(begin && end>=begin) {
                const auto host_latency=latency;
                const auto exposed=waited>=end?double(waited-end)*1e-6:0.;
                // Cached GPU routing can start before host table publication.
                // A blocked GPU consumer observes the CPU verdict on the same
                // GPU clock as the resident/copy lane, avoiding a late CPU
                // origin which would underprice exposed CPU work.
                if(waited>=end && cpu_auto_count>0)latency=double(waited-begin)*1e-6;
                banks[0]->cache->record_cpu_balance(banks,stream.device(),int(tokens),output_type==tb::kFloat16,
                    cpu_auto_count,cpu_auto_misses-cpu_auto_count,latency,double(end-begin)*1e-6,
                    exposed,latency-host_latency);
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                    static std::atomic<unsigned> reported{0};
                    const auto bit=1u<<unsigned(tokens);
                    if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                        "native_gguf_cpu_tail_clock actual_match=1 rows=%lld gpu_clock_origin=1 blocked=%d\n",
                        static_cast<long long>(tokens),int(waited>=end));
                }
            }
        }
    }
    void prepare_cpu(float limit) {
        if(!cpu_buffer)return;
        wait_cpu();
        auto* ready=static_cast<std::uint32_t*>(cpu_buffer->data());
        auto* mask=reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(cpu_buffer->data())+64);
        std::fill_n(mask,tokens*routes,0);
        auto experts=banks[0]->cache->cpu_candidates(banks,static_cast<const std::int64_t*>(route_data()),
            int(tokens*routes),stream.device(),cpu_max_experts,int(tokens),output_type==tb::kFloat16,cpu_auto?&cpu_auto_misses:nullptr);
        cpu_auto_count=int(experts.size());
        // Eager preparation may load kernels, pin source images and start the
        // CPU pool between table publication and completion. Those one-time
        // costs do not describe replay, and would suppress CPU exploration
        // for hundreds of layers. Only replayed windows train the cost fit.
        cpu_balance_pending=cpu_auto && !mfq::cuda::DecodeWindow::recording();
        if(experts.empty()){mfq::cuda::publish_mapped_flag(ready);return;}
        auto work=std::make_shared<GgufCpuWork>();
        work->tokens=int(tokens);work->routes=int(routes);work->input_width=int(banks[0]->input);
        work->hidden_width=int(banks[2]->input);work->output_width=int(banks[2]->out);
        work->half=output_type==tb::kFloat16;work->limit=limit;
        work->input=reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(cpu_buffer->data())+cpu_input_offset);
        work->up_input=reinterpret_cast<const float*>(static_cast<const std::uint8_t*>(cpu_buffer->data())+cpu_up_offset);
        work->down_rotation=banks[2]->rotation;
        work->output=reinterpret_cast<float*>(static_cast<std::uint8_t*>(cpu_buffer->data())+cpu_output_offset);
        work->ready=ready;work->experts=std::move(experts);
        for(const auto& expert:work->experts)for(int slot:expert.slots)mask[slot]=1;
        *ready=0;std::atomic_thread_fence(std::memory_order_seq_cst);
        banks[0]->cache->submit_cpu(work);cpu_work=std::move(work);
    }
    void* route_data() const { return static_cast<std::uint8_t*>(host_ids->data()) + 128; }
    std::int64_t* host_table(int role) const {
        return mapped_routes ? reinterpret_cast<std::int64_t*>(static_cast<std::uint8_t*>(route_data()) +
            tokens * routes * sizeof(std::int64_t)) + role * banks[0]->experts : host_addresses[role].data_ptr<std::int64_t>();
    }
    const std::uint64_t* device_table(int role) const {
        if(resident_overlap)return reinterpret_cast<const std::uint64_t*>(streamed_device_addresses.data_ptr())+role*banks[0]->experts;
        return mapped_routes ? reinterpret_cast<const std::uint64_t*>(static_cast<const std::uint8_t*>(mapped_routes) +
            128 + tokens * routes * sizeof(std::int64_t)) + role * banks[0]->experts :
            reinterpret_cast<const std::uint64_t*>(addresses[role].data_ptr());
    }
    std::int64_t* resident_host_table(int role) const {
        return reinterpret_cast<std::int64_t*>(static_cast<std::uint8_t*>(host_ids->data())+resident_table_offset)+role*banks[0]->experts;
    }
    const std::uint64_t* resident_device_table(int role) const {
        return reinterpret_cast<const std::uint64_t*>(resident_device_addresses.data_ptr())+role*banks[0]->experts;
    }
    void reset_routes() {
        wait_cpu();
        if(cpu_auto)std::memset(static_cast<std::uint8_t*>(cpu_buffer->data())+8,0,3*sizeof(std::uint64_t));
        for(const auto& bank:banks)bank->cache->prepare_admission_window(!lease->mutable_residency);
        if(resident_view && !lease->mutable_residency)banks[0]->cache->prepare_resident_views(stream.device(),stream.stream());
        if (!mapped_routes) return;
        *static_cast<volatile std::uint32_t*>(host_ids->data()) = 0;
        if(direct)*reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+64)=0;
        if(decode_dma)std::memset(static_cast<std::uint8_t*>(host_ids->data())+72,0,2*sizeof(std::uint32_t));
        if(resident_overlap)*reinterpret_cast<volatile std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+96)=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }
    void publish_routes() {
        auto* cpu_input=cpu_buffer?reinterpret_cast<float*>(static_cast<std::uint8_t*>(cpu_mapped)+cpu_input_offset):nullptr;
        auto* cpu_up=cpu_buffer?reinterpret_cast<float*>(static_cast<std::uint8_t*>(cpu_mapped)+cpu_up_offset):nullptr;
        if(mapped_routes)reset_routes();
        prepare_bank_input(banks[0]->rotation,input,q8_input,cpu_input,stream.stream());
        if(!shared_input_basis)prepare_bank_input(banks[1]->rotation,input,q8_up,cpu_up,stream.stream());
        if(mapped_routes) {
            if(auto* window=mfq::cuda::DecodeWindow::recording())window->mark_current("cpu_input_publish_end");
            mfq::cuda::moe_publish_routes(nullptr,reinterpret_cast<const std::int32_t*>(ids.data_ptr()),nullptr,
                reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(mapped_routes)+128),
                static_cast<std::uint32_t*>(mapped_routes),0,int(tokens*routes*2),stream.stream());
            TYLOINFER_CUDA_CHECK(cudaGetLastError());
            if(auto* window=mfq::cuda::DecodeWindow::recording())window->mark_current("route_publish_end");
        } else TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(route_data(),ids.data_ptr(),ids.nbytes(),cudaMemcpyDeviceToHost,stream.stream()));
    }
    void wait_routes() {
        if (mapped_routes) mfq::cuda::wait_route_publication(static_cast<const std::uint32_t*>(host_ids->data()), stream.stream());
        else TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream.stream()));
    }
    void load(float limit) {
        const MfqCudaStreamGuard guard(stream);
        banks[0]->cache->wait_layer_admission(resident_view);
        std::array<double,6> times{};
        using Clock=std::chrono::steady_clock;
        auto before=profile_host_prepare?Clock::now():Clock::time_point{};
        const auto phase=[&](int index) {
            if(!profile_host_prepare)return;
            const auto now=Clock::now();
            times[index]+=std::chrono::duration<double,std::milli>(now-before).count();before=now;
        };
        const auto* values = static_cast<const std::int64_t*>(route_data());
        selected.clear();
        for (std::int64_t slot = 0; slot < tokens * routes; ++slot) {
            const auto id=values[slot];
            if (id < 0 || id >= banks[0]->experts)
                throw std::out_of_range("GGUF MoE expert id");
            if(std::find(selected.begin(),selected.end(),id)==selected.end())selected.push_back(id);
        }
        // The GPU may publish the next layer as soon as this copy plan is
        // released. Later metadata learning must not retain its mapped IDs.
        if(direct)observation_routes.assign(values,values+tokens*routes);
        lease->clear();phase(0);
        if(resident_overlap) {
            banks[0]->cache->acquire_resident_moe(banks,selected,stream.device(),stream.stream(),
                {resident_host_table(0),resident_host_table(1),resident_host_table(2)},lease->weights,profile_host_prepare);
            if(cpu_auto)cpu_reference_at=std::chrono::steady_clock::now();
            mfq::cuda::publish_mapped_flag(reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+96));
        }
        phase(1);
        if(direct && route_groups.defined()) {
            auto* plan=reinterpret_cast<mfq::cuda::gguf::RouteGroup*>(static_cast<std::uint8_t*>(host_ids->data())+route_group_offset);
            plan[0].count=0;
            const int capacity=mfq::cuda::gguf::route_group_capacity(int(tokens));
            for(int slot=0;slot<tokens*routes;++slot) {
                int group=1;
                for(;group<=plan[0].count;++group)if(plan[group].expert==values[slot] && plan[group].count<capacity)break;
                if(group>plan[0].count){++plan[0].count;plan[group].expert=values[slot];plan[group].count=0;}
                plan[group].slots[plan[group].count++]=slot;
            }
            banks[0]->cache->record_route_groups(tokens*routes,plan[0].count);
        }
        phase(2);
        prepare_cpu(limit);
        if(cpu_work) {
            const auto* mask=reinterpret_cast<const std::int32_t*>(static_cast<const std::uint8_t*>(cpu_buffer->data())+64);
            selected.erase(std::remove_if(selected.begin(),selected.end(),[&](auto id){
                for(std::int64_t slot=0;slot<tokens*routes;++slot)if(values[slot]==id && mask[slot])return true;
                return false;
            }),selected.end());
            for(int role=0;role<3;++role)for(std::int64_t slot=0;slot<tokens*routes;++slot)
                if(mask[slot])host_table(role)[values[slot]]=0;
        }
        phase(3);
        std::array<std::vector<GgufWeight>,3> acquired;
        const bool shared=banks[0]->cache==banks[1]->cache && banks[0]->cache==banks[2]->cache;
        const bool speculative=shared && banks[0]->cache->speculative_routes();
        if(direct) {
            auto* header=reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(host_ids->data())+copy_offset);
            std::size_t count=0,bytes=0,gate_up_bytes=0;
            // Cross-layer windows share this arena and overwrite its contents.
            // Preserve their compact transfer layout; a single writer can
            // safely retain stable expert slots between consecutive replays.
            auto& images=lease->direct_arena->images;
            auto* reusable=lease->direct_stages==1?&images:nullptr;
            if(!reusable)images.invalidate();
            if(reusable)reusable->prepare(banks,selected,lease->direct_workspace.data_ptr(),lease->direct_workspace.nbytes());
            try {banks[0]->cache->acquire_direct(banks,selected,stream.device(),stream.stream(),lease->direct_workspace.data_ptr(),
                {host_table(0),host_table(1),host_table(2)},
                {resident_overlap?resident_host_table(0):nullptr,resident_overlap?resident_host_table(1):nullptr,
                    resident_overlap?resident_host_table(2):nullptr},lease->weights,lease->host_owners,
                reinterpret_cast<mfq::cuda::MoeMappedCopyDescriptor*>(header+4),count,bytes,int(tokens),&gate_up_bytes,
                reusable);}
            catch(...) {images.invalidate();throw;}
            header[0]=count;header[1]=bytes;header[3]=gate_up_bytes;
        } else if(shared)
            acquired=banks[0]->cache->acquire_moe(banks,selected,stream.device(),!speculative);
        else for(int role=0;role<3;++role)
            acquired[role]=banks[role]->cache->acquire(*banks[role],selected,stream.device());
        phase(4);
        if(!direct && speculative)banks[0]->cache->observe_routes(banks,values,tokens,routes,stream.device());
        if(!direct)for (int role = 0; role < 3; ++role) {
            const auto& bank = *banks[role];
            auto* pointers = host_table(role);
            for (std::size_t i = 0; i < selected.size(); ++i)
                pointers[selected[i]] = resident_overlap && resident_host_table(role)[selected[i]]?0:
                    reinterpret_cast<std::intptr_t>(acquired[role][i].packed.data_ptr());
            lease->weights.insert(lease->weights.end(), acquired[role].begin(), acquired[role].end());
            if (!mapped_routes) TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(addresses[role].data_ptr(), pointers,
                addresses[role].nbytes(), cudaMemcpyHostToDevice, stream.stream()));
        }
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if(direct) {
            if(cpu_auto && !resident_overlap)cpu_reference_at=std::chrono::steady_clock::now();
            mfq::cuda::publish_mapped_flag(reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+64));
        }
        if(decode_dma) {
            const auto* header=reinterpret_cast<const std::uint64_t*>(static_cast<const std::uint8_t*>(host_ids->data())+copy_offset);
            decode_dma->submit(reinterpret_cast<const mfq::cuda::MoeMappedCopyDescriptor*>(header+4),header[0],header[3],
                reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+72),
                reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+76),
                reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped_routes)+72),
                reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped_routes)+76));
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<std::uint64_t> reported{0};const auto bit=std::uint64_t(1)<<std::min<std::int64_t>(tokens,63);
                if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                    "native_gguf_decode_dma actual_match=1 rows=%lld captured_flag_wait=1 copy_sm_blocks=0 copies=%llu bytes=%llu\n",
                    static_cast<long long>(tokens),static_cast<unsigned long long>(header[0]),static_cast<unsigned long long>(header[1]));
            }
        }
        // DMA owns its submitted plan now. Metadata changes neither weight bytes nor
        // device-residency tables and completes before the serving callback
        // returns, so window admission/commit sees the same learned routes.
        if(direct) {
            banks[0]->cache->observe_routes(banks,observation_routes.data(),tokens,routes,stream.device());
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<std::uint64_t> seen{};
                const auto bit=std::uint64_t(1)<<std::min(tokens,std::int64_t(63));
                if(!(seen.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                    "native_gguf_route_learning actual_match=1 rows=%lld snapshot_ids=1 after_plan_publish=1 after_dma_submit=%d compact_observation=1\n",
                    static_cast<long long>(tokens),int(bool(decode_dma)));
            }
        }
        phase(5);
        if(profile_host_prepare)banks[0]->cache->record_host_prepare(times);
    }
    void cancel() {
        if(decode_dma) {
            mfq::cuda::publish_mapped_flag(reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+72));
            mfq::cuda::publish_mapped_flag(reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+76));
            try{decode_dma->synchronize();}catch(...){}
        }
        // Admission owns an independent stream and releases its gates on both
        // completion and failure. Drain it naturally: destroying an idle graph
        // must not abort a different graph's current cache-admission epoch.
        if(!direct)return;
        lease->direct_arena->images.invalidate();
        try{wait_cpu();}catch(...){}
        if(cpu_buffer)mfq::cuda::publish_mapped_flag(static_cast<std::uint32_t*>(cpu_buffer->data()));
        if(route_groups.defined())reinterpret_cast<mfq::cuda::gguf::RouteGroup*>(
            static_cast<std::uint8_t*>(host_ids->data())+route_group_offset)[0].count=0;
        for(int role=0;role<3;++role)std::memset(host_table(role),0,banks[0]->experts*sizeof(std::int64_t));
        if(resident_overlap) {
            for(int role=0;role<3;++role)std::memset(resident_host_table(role),0,banks[0]->experts*sizeof(std::int64_t));
            mfq::cuda::publish_mapped_flag(reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+96));
        }
        *reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(host_ids->data())+copy_offset)=0;
        mfq::cuda::publish_mapped_flag(reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host_ids->data())+64));
    }
    void prepare_direct_copy(cudaStream_t copy_stream) {
        if(!direct)return;
        auto* flags=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped_routes)+64);
        mfq::cuda::wait_mapped_flag(flags,copy_stream);
        if(resident_overlap) {
            streamed_device_addresses.record_stream(reinterpret_cast<std::uintptr_t>(copy_stream));
            mfq::cuda::gguf::snapshot_routed_addresses(reinterpret_cast<const std::uint64_t*>(
                static_cast<const std::uint8_t*>(mapped_routes)+128+tokens*routes*sizeof(std::int64_t)),
                reinterpret_cast<std::uint64_t*>(streamed_device_addresses.data_ptr()),ids.data_ptr<std::int64_t>(),
                tokens*routes,banks[0]->experts,copy_stream);
        }
        copy_descriptors.record_stream(reinterpret_cast<std::uintptr_t>(copy_stream));
        const auto* host_groups=route_groups.defined()?reinterpret_cast<const mfq::cuda::gguf::RouteGroup*>(
            static_cast<const std::uint8_t*>(mapped_routes)+route_group_offset):nullptr;
        mfq::cuda::gguf::snapshot_direct_plan(static_cast<std::uint8_t*>(mapped_routes)+copy_offset,flags+1,
            copy_descriptors.data_ptr(),3*std::min(tokens*routes,banks[0]->experts),
            cpu_buffer?reinterpret_cast<const std::int32_t*>(static_cast<const std::uint8_t*>(cpu_mapped)+64):nullptr,
            cpu_buffer?reinterpret_cast<std::int32_t*>(copy_descriptors.data_ptr<std::uint8_t>()+cpu_slots_offset):nullptr,
            cpu_buffer?tokens*routes:0,host_groups,route_groups.defined()?static_cast<mfq::cuda::gguf::RouteGroup*>(route_groups.data_ptr()):nullptr,
            tokens*routes,copy_stream);
        if(decode_dma) {
            mfq::cuda::wait_mapped_flag(reinterpret_cast<const std::uint32_t*>(static_cast<const std::uint8_t*>(mapped_routes)+72),copy_stream);
            copy_gate_up_done->record(copy_stream);
            mfq::cuda::wait_mapped_flag(reinterpret_cast<const std::uint32_t*>(static_cast<const std::uint8_t*>(mapped_routes)+76),copy_stream);
        } else if(resident_overlap) {
            mfq::cuda::moe_copy_mapped_phase(copy_descriptors.data_ptr(),copy_descriptors.data_ptr<std::uint32_t>()+4,
                copy_blocks,copy_threads,0,copy_stream);
            copy_gate_up_done->record(copy_stream);
            mfq::cuda::moe_copy_mapped_phase(copy_descriptors.data_ptr(),copy_descriptors.data_ptr<std::uint32_t>()+4,
                copy_blocks,copy_threads,1,copy_stream);
        }else mfq::cuda::moe_copy_mapped(copy_descriptors.data_ptr(),copy_descriptors.data_ptr<std::uint32_t>()+4,
            copy_blocks,copy_threads,copy_stream);
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
    }
    tb::Tensor compute(float limit,const std::function<void()>& independent_work={}) {
        using namespace mfq::cuda::gguf;
        auto* window = mfq::cuda::DecodeWindow::recording();
        auto* groups=route_groups.defined()?static_cast<RouteGroup*>(route_groups.data_ptr()):nullptr;
        if(groups && !direct)prepare_route_groups(ids.data_ptr<std::int64_t>(),tokens*routes,int(tokens),groups,stream.stream());
        if(resident_overlap) {
            copy_start->record(stream.stream());
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(transfer_stream.stream(),copy_start->get(),0));
            prepare_direct_copy(transfer_stream.stream());copy_done->record(transfer_stream.stream());
            // Route publication has already released the host planner. Start
            // its copy stream before independent shared-expert computation,
            // giving both host preparation and DMA useful work to overlap.
            if(independent_work) {
                independent_work();
                if(window)window->mark_current("shared_end");
            }
            const bool gpu_residency=!lease->mutable_residency;
            if(!gpu_residency)mfq::cuda::wait_mapped_flag(reinterpret_cast<const std::uint32_t*>(
                static_cast<const std::uint8_t*>(mapped_routes)+96),stream.stream());
            const auto* resident_source=gpu_residency?static_cast<const std::uint64_t*>(resident_view->mapped):
                reinterpret_cast<const std::uint64_t*>(static_cast<const std::uint8_t*>(mapped_routes)+resident_table_offset);
            auto* early_groups=resident_groups.defined()?static_cast<RouteGroup*>(resident_groups.data_ptr()):nullptr;
            auto* start_clock=cpu_auto?reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(cpu_mapped)+8):nullptr;
            const bool prepared=try_prepare_resident_routes(resident_source,
                reinterpret_cast<std::uint64_t*>(resident_device_addresses.data_ptr()),ids.data_ptr<std::int64_t>(),
                tokens*routes,banks[0]->experts,route_group_capacity(int(tokens)),early_groups,
                gpu_residency?resident_view->admission_device():nullptr,start_clock,stream.stream());
            if(!prepared) {
                snapshot_routed_addresses(resident_source,reinterpret_cast<std::uint64_t*>(resident_device_addresses.data_ptr()),
                    ids.data_ptr<std::int64_t>(),tokens*routes,banks[0]->experts,stream.stream(),
                    gpu_residency?resident_view->admission_device():nullptr);
                if(start_clock)mfq::cuda::moe_stamp_clock(start_clock,stream.stream());
                if(early_groups)prepare_route_groups(ids.data_ptr<std::int64_t>(),tokens*routes,int(tokens),early_groups,stream.stream());
            }
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");gpu_residency && trace && trace[0]=='1') {
                static std::atomic<std::uint64_t> reported{0};
                const auto bit=std::uint64_t(1)<<tokens;
                if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                    "native_gguf_gpu_resident_routes actual_match=1 rows=%lld host_plan_wait=0 fused_prepare=%d\n",static_cast<long long>(tokens),int(prepared));
            }
            if(window)window->mark_current("hot_begin");
            compute_experts(limit,true,early_groups,false);
            if(window)window->mark_current("resident_down_end");
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),copy_gate_up_done->get(),0));
            if(window)window->mark_current("stream_gate_up_join");
        } else {
            if(independent_work) {
                independent_work();
                if(window)window->mark_current("shared_end");
            }
            if(cpu_auto)mfq::cuda::moe_stamp_clock(reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(cpu_mapped)+8),stream.stream());
            prepare_direct_copy(stream.stream());
            if(window)window->mark_current("hot_begin");
        }
        compute_experts(limit,false,groups,resident_overlap,resident_overlap?copy_done->get():nullptr);
        if(cpu_auto)mfq::cuda::moe_stamp_clock(reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(cpu_mapped)+16),stream.stream());
        if (window) window->mark_current("down_end");
        tb::Tensor result;
        if(output_type==tb::kFloat16 || output_type==tb::kFloat32) {
            result=tb::empty({tokens,banks[2]->out},input.options().dtype(output_type));
            const auto* mapped=static_cast<const std::uint8_t*>(cpu_mapped);
            reduce_moe_output(pairs.data_ptr<float>(),weights.data_ptr<float>(),tokens,routes,banks[2]->out,
                output_type==tb::kFloat16,result.data_ptr(),
                cpu_buffer?reinterpret_cast<const float*>(mapped+cpu_output_offset):nullptr,
                cpu_buffer?reinterpret_cast<const std::int32_t*>(copy_descriptors.data_ptr<std::uint8_t>()+cpu_slots_offset):nullptr,
                cpu_buffer?reinterpret_cast<const std::uint32_t*>(mapped):nullptr,stream.stream(),
                cpu_auto?reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(cpu_mapped)+24):nullptr);
        } else {
            result=moe_weighted_reduce_cuda(pairs.to(output_type),weights);
        }
        if (window) window->mark_current("hot_end");
        return result;
    }
    void compute_experts(float limit,bool resident,mfq::cuda::gguf::RouteGroup* groups,bool preserve_missing,cudaEvent_t down_ready=nullptr) {
        using namespace mfq::cuda::gguf;
        auto* window=mfq::cuda::DecodeWindow::recording();
        const bool joined=shared_input_basis && banks[0]->type==banks[1]->type && banks[0]->input==banks[1]->input &&
            banks[0]->out==banks[1]->out && try_gate_up_matvec_q8(banks[0]->type,
                resident?resident_device_table(0):device_table(0),resident?resident_device_table(1):device_table(1),
                banks[0]->row_bytes,banks[1]->row_bytes,banks[0]->experts,banks[0]->out,banks[0]->input,
                ids.data_ptr<std::int64_t>(),groups,tokens,routes,q8_input.data_ptr(),gate.data_ptr<float>(),
                up.data_ptr<float>(),stream.stream());
        if(!joined)for (int role = 0; role < 2; ++role) {
            const auto& bank = *banks[role];
            grouped_routed_matvec_q8(bank.type, resident?resident_device_table(role):device_table(role),
                bank.row_bytes, bank.experts, bank.out, bank.input, ids.data_ptr<std::int64_t>(),groups,
                tokens, routes, (role?q8_up:q8_input).data_ptr(), false, (role ? up : gate).data_ptr<float>(), stream.stream());
        }
        if (window) window->mark_current(resident?"resident_gate_up_end":"gate_up_end");
        prepare_bank_hidden(banks[2]->rotation,gate,up,output_type,limit,q8_hidden,stream.stream());
        if(down_ready) {
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream.stream(),down_ready,0));
            if(window)window->mark_current("stream_down_join");
        }
        const auto& bank = *banks[2];
        grouped_routed_matvec_q8(bank.type, resident?resident_device_table(2):device_table(2),
            bank.row_bytes, bank.experts, bank.out, bank.input, ids.data_ptr<std::int64_t>(),groups,
            tokens, routes, q8_hidden.data_ptr(), true, pairs.data_ptr<float>(), stream.stream(),preserve_missing);
    }
};

tb::Tensor grouped_moe_prefill(const GgufMoeBanks& banks, tb::Tensor source,
    const tb::Tensor& ids, const tb::Tensor& weights, float limit) {
    const std::array<std::shared_ptr<GgufRoutedWeight>, 3> projections{banks.gate, banks.up, banks.down};
    const auto tokens = ids.size(0), routes = ids.size(1), slots = tokens * routes;
    namespace mmq=mfq::cuda::gguf::mmq;
    const auto* mmq_setting=std::getenv("TYLOINFER_GGUF_PREFILL_MMQ");
    bool use_mmq=(!mmq_setting || mmq_setting[0]!='0') && slots<=INT_MAX && banks.up->input<=INT_MAX;
    if(use_mmq)for(const auto& bank:projections)use_mmq&=mmq::fits(bank->type,bank->out,source.get_device());
    const bool shared_basis=same_gguf_rotation(banks.gate->rotation,banks.up->rotation);
    const bool joined_mmq=use_mmq && shared_basis && banks.gate->type==banks.up->type;
    const auto* direct_setting=std::getenv("TYLOINFER_GGUF_PREFILL_DIRECT");
    bool expert_major_mmq=use_mmq && (!direct_setting || direct_setting[0]!='0');
    std::array<bool,3> direct_mmq{};
    std::array<std::size_t,3> expert_offsets{};
    std::size_t expert_stride=0;
    for(int role=0;role<3;++role) {
        const auto& bank=*projections[role];expert_offsets[role]=expert_stride;
        expert_stride+=(bank.expert_bytes+15)&~std::size_t(15);
        if(joined_mmq && role==1)continue;
        direct_mmq[role]=use_mmq && mmq::direct_fits(int(bank.type),bank.out*(joined_mmq && role==0?2:1),bank.input,source.get_device());
        expert_major_mmq&=direct_mmq[role];
    }
    const auto selected = ids.cpu().contiguous();
    std::vector<std::vector<std::int64_t>> grouped(banks.up->experts);
    for (std::int64_t slot = 0; slot < slots; ++slot) {
        const auto expert = selected.data_ptr<std::int64_t>()[slot];
        if (expert < 0 || expert >= banks.up->experts) throw std::out_of_range("GGUF prefill expert id");
        grouped[expert].push_back(slot);
    }
    auto host = use_mmq?tb::Tensor{}:tb::empty({slots},source.options().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
    std::vector<std::int64_t> offsets(banks.up->experts + 1, 0), active;
    for (std::int64_t expert = 0; expert < banks.up->experts; ++expert) {
        offsets[expert + 1] = offsets[expert] + grouped[expert].size();
        if (!grouped[expert].empty()) {
            active.push_back(expert);
            if(host.defined())std::copy(grouped[expert].begin(),grouped[expert].end(),host.data_ptr<std::int64_t>()+offsets[expert]);
        }
    }
    const auto plan=host.defined()?banks.up->cache->upload_packet(std::move(host),source.get_device()):tb::Tensor{};
    banks.up->cache->record_prefill(projections,grouped);
    std::vector<std::uint8_t> admission(banks.up->experts,0);
    auto ranking=active;
    std::stable_sort(ranking.begin(),ranking.end(),[&](auto a,auto b){return grouped[a].size()>grouped[b].size();});
    const auto quota=banks.up->cache->prefill_quota(banks.up->experts,source.get_device());
    for(std::size_t rank=0;rank<std::min(quota,ranking.size());++rank) admission[ranking[rank]]=1;
    // Similar row counts keep both the BMM padding and MMQ's shared tile
    // width efficient. Preserve those groups when scheduling their uploads.
    active=std::move(ranking);
    auto pairs = tb::empty({tokens, routes, banks.down->out}, source.options());
    const auto stream = mfq_current_cuda_stream();
    const auto transfer_stream=banks.up->cache->prefill_transfer_stream(source.get_device());
    long double compressed_bytes=0,decoded_bytes=0,aligned_compressed_bytes=0;
    for(const auto& bank:projections) {
        compressed_bytes+=bank->expert_bytes;
        aligned_compressed_bytes+=std::ceil(static_cast<long double>(bank->expert_bytes)/16)*16;
        decoded_bytes+=2.L*bank->out*bank->input;
    }
    // Expand each original weight once for the whole expert's token batch.
    // Temporary decoded storage stays bounded independently of model size.
    auto group=std::max<std::size_t>(1,std::size_t(std::min({65535.L,static_cast<long double>(active.size()),
        (64ULL<<20)/compressed_bytes,(256ULL<<20)/decoded_bytes})));
    if(use_mmq)group=std::min(group,std::size_t(mmq::group_capacity));
    std::vector<std::size_t> group_ends;
    for(std::size_t begin=0;begin<active.size();) {
        auto end=std::min(active.size(),begin+group);
        const auto maximum=grouped[active[begin]].size();
        for(auto next=begin+1;next<end;++next)if(grouped[active[next]].size()<(maximum+1)/2){end=next;break;}
        group_ends.push_back(end);begin=end;
    }
    auto host_staging=banks.up->cache->prepare_host_staging(projections,active,group_ends,source.get_device());
    std::size_t mmq_guard_bytes=0;
    if(use_mmq)for(const auto& bank:projections)mmq_guard_bytes+=128*ggml_type_size(bank->type)+15;
    const auto arena_bytes=aligned_compressed_bytes*group+mmq_guard_bytes;
    if(arena_bytes>INT64_MAX)throw std::overflow_error("GGUF prefill staging capacity overflow");
    std::array<GgufPrefillWeightStage,2> weight_stages;
    std::array<mfq::cuda::Event,2> weight_consumed,weight_ready,gate_up_ready;
    {
        // Strata's PP uploads reuse a bounded expert workspace. Temporary
        // images never enter the resident cache: two fenced arenas replace
        // thousands of per-expert device allocations during the prompt.
        const mfq::cuda::RoutedWeightAllocationScope temporary(false);
        for(auto& stage:weight_stages) {
            stage.image=tb::empty({std::int64_t(arena_bytes)},source.options().dtype(tb::kUInt8));
            stage.image.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            stage.image.record_stream(reinterpret_cast<std::uintptr_t>(transfer_stream.stream()));
        }
    }
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true,std::memory_order_relaxed))
            std::fprintf(stderr,"native_gguf_prefill_weight_arena arenas=2 bytes_per_arena=%lld expert_group_cap=%zu\n",
                static_cast<long long>(arena_bytes),group);
    }
    // Each descriptor is written once. Sharing one pinned slab across the
    // layer avoids registering/freeing three tiny host allocations per group,
    // which can synchronize otherwise asynchronous uploads and GEMMs.
    tb::Tensor job_metadata,mmq_metadata,mmq_packet,mmq_inverse;
    std::vector<std::int64_t> mmq_offsets,mmq_row_offsets;
    banks.up->cache->profile_prefill("gguf.pp.metadata",[&] {
        job_metadata=use_mmq?tb::Tensor{}:tb::empty({std::int64_t(3*active.size()*sizeof(mfq::cuda::gguf::GemmJob))},
            source.options().device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        mmq_metadata=use_mmq?tb::empty({3*slots+2*std::int64_t(active.size())+std::int64_t(group_ends.size())*mmq::id_guard+1},
            source.options().device(tb::kCPU).dtype(tb::kInt32).pinned_memory(true)):tb::Tensor{};
        std::int64_t mmq_cursor=0;
        if(use_mmq) {
            // Group bounds and token permutations depend only on routing, not on
            // the uploaded weights. Build the whole layer once, then keep device
            // views instead of allocating/copying a packet and event per group.
            std::size_t begin=0;
            std::int64_t compact_row=0;
            auto* inverse=mmq_metadata.data_ptr<std::int32_t>();
            mmq_cursor=slots;
            for(const auto end:group_ends) {
                mmq_offsets.push_back(mmq_cursor);
                mmq_row_offsets.push_back(compact_row);
                const auto n=std::int64_t(end-begin);
                std::int64_t count=0;
                for(auto i=begin;i<end;++i)count+=grouped[active[i]].size();
                auto* bounds=mmq_metadata.data_ptr<std::int32_t>()+mmq_cursor;bounds[0]=0;
                auto* identity=bounds+n+1;auto* output_ids=identity+count+mmq::id_guard;
                std::fill_n(identity+count,mmq::id_guard,0);
                std::int64_t at=0;
                for(std::int64_t i=0;i<n;++i) {
                    for(const auto slot:grouped[active[begin+i]]) {
                        inverse[slot]=std::int32_t(compact_row+at);
                        identity[at]=std::int32_t(at);output_ids[at++]=std::int32_t(slot);
                    }
                    bounds[i+1]=std::int32_t(at);
                }
                mmq_cursor+=n+1+2*count+mmq::id_guard;compact_row+=count;begin=end;
            }
            mmq_offsets.push_back(mmq_cursor);
            mmq_packet=banks.up->cache->upload_packet(mmq_metadata.narrow(0,0,mmq_cursor),source.get_device());
            mmq_inverse=mmq_packet.narrow(0,0,slots);
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_prefill_metadata actual_match=1 packets_per_layer=1 groups=%zu unused_plan=0 unused_jobs=0 activation_quantize_once=1\n",group_ends.size());
            }
        }
        return true;
    });
    std::array<tb::Tensor,2> mmq_input;
    if(use_mmq)for(int role=0;role<2;++role) {
        if(role==1 && shared_basis){mmq_input[1]=mmq_input[0];continue;}
        mmq_input[role]=(projections[role]->rotation?projections[role]->rotation->apply(source):source).to(tb::kFloat32).contiguous();
    }
    auto mmq_context=use_mmq?std::make_unique<mmq::Context>():nullptr;
    std::array<tb::Tensor,2> mmq_layer_input;
    if(use_mmq)for(int role=0;role<2;++role) {
        if(role==1 && shared_basis && banks.gate->type==banks.up->type){mmq_layer_input[1]=mmq_layer_input[0];continue;}
        const auto& bank=*projections[role];
        auto& activation=mmq_layer_input[role];
        activation=tb::empty({std::int64_t(mmq::activation_bytes(slots,bank.input))},source.options().dtype(tb::kUInt8));
        banks.up->cache->profile_prefill("gguf.pp.mmq.quantize",[&] {
            mmq::quantize_scatter(mmq_input[role].data_ptr<float>(),mmq_inverse.data_ptr<std::int32_t>(),activation.data_ptr(),
                int(bank.type),bank.input,tokens,int(routes),stream);return true;
        });
        activation.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    }
    source = source.contiguous();
    for (const auto& tensor : {source, pairs, plan,mmq_packet})
        if(tensor.defined())tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    struct UploadGroup {
        GgufPrefillWeightStage stage;
        std::vector<std::int64_t> experts;
        std::array<tb::Tensor,3> packed;
        std::array<std::vector<GgufWeight>,3> acquired;
    };
    std::vector<UploadGroup> upload_groups(group_ends.size());
    std::size_t group_index=0;
    for (std::size_t begin = 0; begin < active.size();++group_index) {
        const auto stage_index=group_index%weight_stages.size();
        auto& job=upload_groups[group_index];job.stage=weight_stages[stage_index];
        auto& weight_stage=job.stage;weight_stage.cursor=0;
        weight_stage.gate_up_ready=gate_up_ready[stage_index].get();weight_stage.gate_up_submitted=false;
        const auto end=group_ends[group_index];
        job.experts.assign(active.begin()+begin,active.begin()+end);
        const auto& experts=job.experts;
        auto& mmq_packed=job.packed;
        weight_stage.mmq_destinations.clear();
        if(use_mmq) {
            std::size_t cursor=0;
            const auto n=std::int64_t(experts.size());
            for(int role=0;role<3;++role) {
                const auto& bank=*projections[role];
                const auto stride=expert_major_mmq?expert_stride:bank.expert_bytes*(joined_mmq && role<2?2:1);
                if(role==1 && joined_mmq)mmq_packed[role]=mmq_packed[0];
                else if(expert_major_mmq) {
                    const auto offset=expert_offsets[role];
                    mmq_packed[role]=weight_stage.image.narrow(0,std::int64_t(offset),std::int64_t(arena_bytes-offset));
                }
                else {
                    const auto size=n*stride+128*ggml_type_size(bank.type);
                    mmq_packed[role]=weight_stage.image.narrow(0,std::int64_t(cursor),std::int64_t(size));
                    cursor=(cursor+size+15)&~std::size_t(15);
                }
                for(std::int64_t i=0;i<n;++i) {
                    const auto offset=i*stride+(joined_mmq && role==1?bank.expert_bytes:0);
                    weight_stage.mmq_destinations.emplace((bank.source_id<<32)|std::uint64_t(experts[i]),
                        mmq_packed[role].narrow(0,std::int64_t(offset),std::int64_t(bank.expert_bytes)));
                }
            }
        }
        begin=end;
    }
    // Two reusable weight arenas. CUDA events guard reuse and order each
    // projection on the compute stream; submissions use the serving thread.
    const auto acquire_group=[&](std::size_t index) {
        if(index>=upload_groups.size())return;
        const MfqCudaStreamGuard transfer_guard(transfer_stream);
        if(index>=weight_stages.size())
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(transfer_stream.stream(),
                weight_consumed[index%weight_stages.size()].get(),0));
        auto& job=upload_groups[index];
        banks.up->cache->profile_prefill("gguf.pp.acquire",[&] {
            if(projections[0]->cache==projections[1]->cache && projections[0]->cache==projections[2]->cache)
                job.acquired=projections[0]->cache->acquire_moe(projections,job.experts,source.get_device(),true,&admission,&job.stage);
            else for(int role=0;role<3;++role)
                job.acquired[role]=projections[role]->cache->acquire(*projections[role],job.experts,source.get_device(),nullptr,true,&admission,&job.stage);
            return true;
        });
        weight_ready[index%weight_stages.size()].record(transfer_stream.stream());
    };
    acquire_group(0);
    group_index=0;
    for(std::size_t begin=0;begin<active.size();++group_index) {
        const auto stage_index=group_index%2,end=group_ends[group_index];
        auto& job=upload_groups[group_index];auto& weight_stage=job.stage;
        const auto& experts=job.experts;auto& mmq_packed=job.packed;
        acquire_group(group_index+1);
        auto& acquired=job.acquired;
        for(const auto& bank:acquired)for(const auto& weight:bank)
            weight.packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        if(use_mmq) {
            std::int64_t count=0,maximum=0;
            for(const auto expert:experts){count+=grouped[expert].size();maximum=std::max(maximum,std::int64_t(grouped[expert].size()));}
            const auto n=std::int64_t(experts.size());
            auto packet=mmq_packet.narrow(0,mmq_offsets[group_index],mmq_offsets[group_index+1]-mmq_offsets[group_index]);
            const auto* device_bounds=packet.data_ptr<std::int32_t>();
            const auto* device_identity=device_bounds+n+1;const auto* device_output=device_identity+count+mmq::id_guard;
            const bool joined=joined_mmq;
            auto packed=mmq_packed;
            mmq::Copy copies[3];
            for(int role=0;role<3;++role) {
                const auto& bank=*projections[role];
                const auto guard=std::size_t(128)*ggml_type_size(bank.type);
                const auto stride=expert_major_mmq?expert_stride:bank.expert_bytes*(joined && role<2?2:1);
                auto& copy=copies[role];copy.destination=packed[role].data_ptr();copy.bytes=bank.expert_bytes;copy.stride=stride;
                copy.offset=joined && role==1?bank.expert_bytes:0;
                copy.guard_bytes=expert_major_mmq || (joined && role==1)?0:guard;
                for(std::int64_t i=0;i<n;++i)copy.source[i]=acquired[role][i].packed.data_ptr();
                packed[role].record_stream(reinterpret_cast<std::uintptr_t>(stream));
            }
            std::array<std::array<const void*,mmq::group_capacity>,3> weight_tables{};
            std::array<bool,3> use_table{};
            std::size_t inplace_bytes=0;
            if(!direct_setting || direct_setting[0]!='0')for(int role=0;role<3;++role) {
                if(joined && role==1)continue;
                use_table[role]=expert_major_mmq;
                const auto& bank=*projections[role];
                if(!direct_mmq[role])continue;
                for(std::int64_t i=0;i<n;++i) {
                    auto* destination=static_cast<std::uint8_t*>(copies[role].destination)+i*copies[role].stride;
                    const auto* candidate=static_cast<const std::uint8_t*>(acquired[role][i].packed.data_ptr());
                    const bool contiguous=!joined || role!=0 ||
                        acquired[1][i].packed.data_ptr()==candidate+bank.expert_bytes;
                    weight_tables[role][i]=contiguous?candidate:destination;
                    if(!contiguous || candidate==destination)continue;
                    copies[role].source[i]=destination;use_table[role]=true;
                    inplace_bytes+=bank.expert_bytes;
                    if(joined && role==0) {
                        copies[1].source[i]=static_cast<std::uint8_t*>(copies[1].destination)+i*copies[1].stride+copies[1].offset;
                        inplace_bytes+=projections[1]->expert_bytes;
                    }
                }
            }
            if(inplace_bytes)banks.up->cache->record_prefill_inplace(inplace_bytes);
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,weight_stage.gate_up_submitted?
                gate_up_ready[stage_index].get():weight_ready[stage_index].get(),0));
            banks.up->cache->profile_prefill("gguf.pp.mmq.gather",[&] {
                if(weight_stage.gate_up_submitted) {
                    mmq::Copy gu_copies[3]={copies[0],copies[1],copies[2]};
                    gu_copies[2].bytes=gu_copies[2].guard_bytes=0;
                    mmq::gather(gu_copies,int(n),stream);
                } else mmq::gather(copies,int(n),stream);
                return true;
            });
            auto gu=tb::empty({count,2*banks.up->out},source.options().dtype(tb::kFloat32));
            const auto product=[&](int role,std::int64_t rows,std::size_t expert_bytes,const tb::Tensor& activation,
                    float* output,std::int64_t stride) {
                const auto& bank=*projections[role];
                banks.up->cache->profile_prefill("gguf.pp.mmq.gemm",[&] {
                    mmq_context->run({packed[role].data_ptr(),int(bank.type),rows,bank.input,expert_bytes,int(n),activation.data_ptr(),
                        device_bounds,device_identity,count,maximum,output,stride,use_table[role]?weight_tables[role].data():nullptr,
                        role<2?slots:0,role<2?mmq_row_offsets[group_index]:0},stream);return true;
                });
            };
            const auto& activation=mmq_layer_input[0];
            product(0,banks.gate->out*(joined?2:1),banks.gate->expert_bytes*(joined?2:1),activation,gu.data_ptr<float>(),gu.size(1));
            if(!joined) {
                const auto& up_activation=mmq_layer_input[1];
                product(1,banks.up->out,banks.up->expert_bytes,up_activation,gu.data_ptr<float>()+banks.gate->out,gu.size(1));
            }
            auto hidden_q=tb::empty({std::int64_t(mmq::activation_bytes(count,banks.down->input))},source.options().dtype(tb::kUInt8));
            banks.up->cache->profile_prefill("gguf.pp.mmq.quantize",[&] {
                if(banks.down->rotation) {
                    auto hidden=tb::empty({count,banks.down->input},gu.options());
                    mmq::swiglu(gu.data_ptr<float>(),hidden.data_ptr<float>(),count,banks.down->input,limit,stream);
                    banks.down->rotation->apply_inplace(hidden);
                    mmq::quantize(hidden.data_ptr<float>(),nullptr,hidden_q.data_ptr(),int(banks.down->type),banks.down->input,count,stream);
                    hidden.record_stream(reinterpret_cast<std::uintptr_t>(stream));
                } else mmq::swiglu_quantize(gu.data_ptr<float>(),hidden_q.data_ptr(),int(banks.down->type),
                    banks.down->input,count,limit,stream);return true;
            });
            hidden_q.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            if(weight_stage.gate_up_submitted)banks.up->cache->profile_prefill("gguf.pp.mmq.gather_down",[&] {
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,weight_ready[stage_index].get(),0));
                mmq::Copy down_copies[3]={copies[0],copies[1],copies[2]};
                for(int role=0;role<2;++role)down_copies[role].bytes=down_copies[role].guard_bytes=0;
                mmq::gather(down_copies,int(n),stream);return true;
            });
            auto result=tb::empty({count,banks.down->out},gu.options());
            product(2,banks.down->out,banks.down->expert_bytes,hidden_q,result.data_ptr<float>(),banks.down->out);
            mmq::scatter(result.data_ptr<float>(),pairs.data_ptr(),device_output,count,banks.down->out,stream);
            weight_consumed[stage_index].record(stream);
            for(auto& weights:acquired)weights.clear();
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_prefill_mmq actual_match=1 gate=%d up=%d down=%d experts=%lld joined_gate_up=%d expert_major=%d expert_order=frequency_groups\n",
                    int(banks.gate->type),int(banks.up->type),int(banks.down->type),static_cast<long long>(n),int(joined),int(expert_major_mmq));
            }
            begin=end;continue;
        }
        std::array<tb::Tensor, 3> packets;
        std::int64_t maximum = 0;
        for (const auto expert : experts) {
            maximum = std::max(maximum, std::int64_t(grouped[expert].size()));
        }
        for (int role = 0; role < 3; ++role) {
            const auto& bank = *projections[role];
            auto packet=job_metadata.narrow(0,
                std::int64_t((role*active.size()+begin)*sizeof(mfq::cuda::gguf::GemmJob)),
                std::int64_t(experts.size()*sizeof(mfq::cuda::gguf::GemmJob)));
            auto* jobs = reinterpret_cast<mfq::cuda::gguf::GemmJob*>(packet.data_ptr());
            for (std::size_t i = 0; i < experts.size(); ++i) jobs[i] = {acquired[role][i].packed.data_ptr(),
                plan.data_ptr<std::int64_t>() + offsets[experts[i]], std::int64_t(grouped[experts[i]].size())};
            packets[role] = bank.cache->upload_packet(std::move(packet), source.get_device());
        }
        const auto batch=std::int64_t(experts.size());
        auto gathered=tb::empty({batch,maximum,banks.up->input},source.options());
        gathered.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        mfq::cuda::gguf::grouped_gather_f16(reinterpret_cast<const mfq::cuda::gguf::GemmJob*>(packets[0].data_ptr()),
            int(batch),maximum,banks.up->input,source.data_ptr(),routes,gathered.data_ptr(),stream);
        const auto decode=[&](int role,tb::Tensor& decoded,std::int64_t stride=0) {
            const auto& bank=*projections[role];
            decoded.record_stream(reinterpret_cast<std::uintptr_t>(stream));
            bank.cache->profile_prefill("gguf.pp.decode",[&] {
                mfq::cuda::gguf::grouped_decode_f16(bank.type,
                    reinterpret_cast<const mfq::cuda::gguf::GemmJob*>(packets[role].data_ptr()),
                    int(batch),bank.row_bytes,bank.out,bank.input,decoded.data_ptr(),stream,stride);
                return true;
            });
        };
        tb::Tensor gate_up;
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,weight_stage.gate_up_submitted?
                gate_up_ready[stage_index].get():weight_ready[stage_index].get(),0));
        {
            // Formats remain independent. Only their bounded decoded images
            // share one GEMM, so the activation is read once for Gate and Up.
            const auto& bank=*projections[0];
            auto decoded=tb::empty({batch,2*bank.out,bank.input},source.options());
            auto gate=decoded.narrow(1,0,bank.out),up=decoded.narrow(1,bank.out,bank.out);
            const auto stride=2*bank.out*bank.input;
            decode(0,gate,stride);decode(1,up,stride);
            gate_up=bank.cache->profile_prefill("gguf.pp.gemm",[&] {
                auto gate_input=banks.gate->rotation?banks.gate->rotation->apply(gathered):gathered;
                if(shared_basis)return tb::bmm(gate_input,decoded.transpose(1,2));
                auto up_input=banks.up->rotation?banks.up->rotation->apply(gathered):gathered;
                return tb::cat({tb::bmm(gate_input,gate.transpose(1,2)),tb::bmm(up_input,up.transpose(1,2))},-1);
            });
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))
                    std::fprintf(stderr,"native_gguf_prefill_gate_up_fused enabled=1 gate_format=%d up_format=%d experts=%lld max_tokens=%lld\n",
                        int(banks.gate->type),int(banks.up->type),static_cast<long long>(batch),static_cast<long long>(maximum));
            }
        }
        auto hidden=tb::empty({batch,maximum,banks.up->out},source.options());
        gate_up.record_stream(reinterpret_cast<std::uintptr_t>(stream));hidden.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        mfq::cuda::gguf::gate_up_swiglu_f16(gate_up.data_ptr(),batch*maximum,banks.up->out,hidden.data_ptr(),limit,stream);
        if(banks.down->rotation)banks.down->rotation->apply_inplace(hidden);
        auto decoded=tb::empty({batch,banks.down->out,banks.down->input},source.options());
        if(weight_stage.gate_up_submitted)
            TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(stream,weight_ready[stage_index].get(),0));
        decode(2,decoded);
        auto result=banks.down->cache->profile_prefill("gguf.pp.gemm",[&] {
            return tb::bmm(hidden,decoded.transpose(1,2));
        });
        mfq::cuda::gguf::grouped_scatter_f16(reinterpret_cast<const mfq::cuda::gguf::GemmJob*>(packets[2].data_ptr()),
            int(batch),maximum,banks.down->out,result.data_ptr(),pairs.data_ptr(),stream);
        weight_consumed[stage_index].record(stream);
        for(auto& weights:acquired)weights.clear();
        begin=end;
    }
    return moe_weighted_reduce_cuda(pairs, weights.to(tb::kFloat32).contiguous());
}
}

// A fully resident bank needs no host route publication, cache lookup or
// transfer-stream handshake. Routing and all three projections stay in the
// captured device graph, using the same quantization and compute kernels.
static tb::Tensor resident_moe(const GgufMoeBanks& banks,tb::Tensor source,
        const tb::Tensor& ids,tb::Tensor weights,float limit) {
    using namespace mfq::cuda::gguf;
    const std::array bank{banks.gate,banks.up,banks.down};
    const auto tokens=ids.size(0),routes=ids.size(1),slots=tokens*routes;
    const auto stream=mfq_current_cuda_stream();
    for(const auto& b:bank) {
        if(b->resident_packed.device()!=source.device())throw std::invalid_argument("GGUF resident MoE device mismatch");
        b->resident_packed.record_stream(reinterpret_cast<std::uintptr_t>(stream));
        b->resident_addresses.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    }
    const auto address=[&](int role){return reinterpret_cast<const std::uint64_t*>(bank[role]->resident_addresses.data_ptr());};
    auto input=source.to(tb::kFloat32).contiguous();
    weights=weights.to(tb::kFloat32).contiguous();
    const auto bytes=input.options().dtype(tb::kUInt8);
    auto q8_input=tb::empty({std::int64_t(tokens*q8_row_bytes(bank[0]->input))},bytes);
    auto q8_hidden=tb::empty({std::int64_t(slots*q8_row_bytes(bank[2]->input))},bytes);
    auto gate=tb::empty({tokens,routes,bank[0]->out},input.options()),up=tb::empty_like(gate);
    auto pairs=tb::empty({tokens,routes,bank[2]->out},input.options());
    tb::Tensor group_storage;RouteGroup* groups=nullptr;
    // A single row has no cross-token reuse. Both paired Gate/Up and Down
    // already consume its routed IDs directly, without a grouping pass.
    if(tokens>1) {
        group_storage=tb::empty({std::int64_t((slots+1)*sizeof(RouteGroup))},bytes);
        groups=static_cast<RouteGroup*>(group_storage.data_ptr());
        prepare_route_groups(ids.data_ptr<std::int64_t>(),slots,int(tokens),groups,stream);
    }
    const bool shared_basis=same_gguf_rotation(bank[0]->rotation,bank[1]->rotation);
    auto q8_up=shared_basis?q8_input:tb::empty_like(q8_input);
    prepare_bank_input(bank[0]->rotation,source,q8_input,nullptr,stream);
    if(!shared_basis)prepare_bank_input(bank[1]->rotation,source,q8_up,nullptr,stream);
    auto* window=mfq::cuda::DecodeWindow::recording();
    if(window)window->mark_current("hot_begin");
    const bool joined=shared_basis && bank[0]->type==bank[1]->type && try_gate_up_matvec_q8(bank[0]->type,
        address(0),address(1),bank[0]->row_bytes,bank[1]->row_bytes,bank[0]->experts,bank[0]->out,bank[0]->input,
        ids.data_ptr<std::int64_t>(),groups,tokens,routes,q8_input.data_ptr(),gate.data_ptr<float>(),up.data_ptr<float>(),stream);
    if(!joined)for(int role=0;role<2;++role)
        grouped_routed_matvec_q8(bank[role]->type,address(role),bank[role]->row_bytes,bank[role]->experts,
            bank[role]->out,bank[role]->input,ids.data_ptr<std::int64_t>(),groups,tokens,routes,
            (role?q8_up:q8_input).data_ptr(),false,(role?up:gate).data_ptr<float>(),stream);
    if(window)window->mark_current("gate_up_end");
    const auto dtype=source.scalar_type();
    prepare_bank_hidden(bank[2]->rotation,gate,up,dtype,limit,q8_hidden,stream);
    grouped_routed_matvec_q8(bank[2]->type,address(2),bank[2]->row_bytes,bank[2]->experts,
        bank[2]->out,bank[2]->input,ids.data_ptr<std::int64_t>(),groups,tokens,routes,
        q8_hidden.data_ptr(),true,pairs.data_ptr<float>(),stream);
    if(window)window->mark_current("down_end");
    tb::Tensor result;
    if(dtype==tb::kFloat16 || dtype==tb::kFloat32) {
        result=tb::empty({tokens,bank[2]->out},source.options());
        reduce_moe_output(pairs.data_ptr<float>(),weights.data_ptr<float>(),tokens,routes,bank[2]->out,
            dtype==tb::kFloat16,result.data_ptr(),nullptr,nullptr,nullptr,stream);
    } else result=moe_weighted_reduce_cuda(pairs.to(dtype),weights);
    if(window)window->mark_current("hot_end");
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{false};
        if(!reported.exchange(true))std::fprintf(stderr,"native_gguf_resident_moe actual_match=1 host_routes=0 expert_transfers=0\n");
    }
    return result;
}

namespace {
// Same ready/fork/join graph structure as Strata's default CUDA shared expert
// stream. Keep the main MoE on its original stream, including host doorbells
// and transfer events; only the independent shared projection changes stream.
template<class Main>
std::optional<tb::Tensor> try_shared_branch(CudaIndependentBranchExecutor* executor,
        const std::function<void()>& side,Main&& main,std::int64_t tokens,
        cudaStream_t transfer=nullptr) {
    auto* window=mfq::cuda::DecodeWindow::recording();
    if(!executor || !side || (window && window->layer_profiling()))return std::nullopt;
    const auto parent=mfq_get_current_cuda_stream();
    if(!executor->ensure(1,parent))return std::nullopt;
    const auto branch=executor->streams[0];
    if(branch.stream()==parent.stream() || branch.stream()==transfer)return std::nullopt;
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(parent.stream(),&capture));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if(window)window->prepare_branch_stream(branch,executor->graph_pools[0]);
#endif
    TYLOINFER_CUDA_CHECK(cudaEventRecord(executor->ready,parent.stream()));
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(branch.stream(),executor->ready,0));
    const char* phase="shared";
    try {
        {const MfqCudaStreamGuard guard(branch);side();}
        phase="shared_complete";
        TYLOINFER_CUDA_CHECK(cudaEventRecord(executor->completed[0],branch.stream()));
        phase="routed";
        auto output=main();
        phase="join";
        TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(parent.stream(),executor->completed[0],0));
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<std::uint64_t> seen[2]{};
            const auto bit=std::uint64_t(1)<<std::min(tokens,std::int64_t(63));
            const int captured=capture!=cudaStreamCaptureStatusNone;
            if(!(seen[captured].fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
                "native_gguf_shared_stream actual_match=1 rows=%lld fork_join=1 shared_parallel_resident=1 captured=%d\n",
                static_cast<long long>(tokens),captured);
        }
        return output;
    }catch(const std::exception& error) {
        std::fprintf(stderr,"native_gguf_shared_stream_error phase=%s rows=%lld captured=%d error=%s\n",
            phase,static_cast<long long>(tokens),int(capture!=cudaStreamCaptureStatusNone),error.what());
        // Preserve the dependency even if shared/main preparation throws;
        // temporary branch allocations must not race the parent's recovery.
        (void)cudaEventRecord(executor->completed[0],branch.stream());
        (void)cudaStreamWaitEvent(parent.stream(),executor->completed[0],0);
        throw;
    }
}
}
tb::Tensor GgufMoeBanks::forward(tb::Tensor source, tb::Tensor ids, tb::Tensor weights, float limit,
        const std::function<void()>& independent_work,CudaIndependentBranchExecutor* shared_executor) const {
    if (!gate || !up || !down || gate->experts != up->experts || up->experts != down->experts ||
        gate->input != up->input || gate->out != up->out || down->input != up->out ||
        source.dim() != 2 || !source.is_cuda() || source.size(1) != up->input || ids.dim() != 2 ||
        source.size(0) != ids.size(0) || ids.size(1) <= 0 || weights.sizes().vec() != ids.sizes().vec() ||
        ids.device() != source.device() || weights.device() != source.device() || !std::isfinite(limit) || limit < 0)
        throw std::invalid_argument("invalid GGUF three-bank MoE geometry");
    const MfqCudaGuard guard(source.get_device());
    ids = ids.to(tb::kInt64).contiguous();
    if (!source.size(0)) {
        if(independent_work)independent_work();
        return tb::empty({0, down->out}, source.options());
    }
    if(source.size(0)<=32 && gate->resident_packed.defined() && up->resident_packed.defined() && down->resident_packed.defined()) {
        if(auto output=try_shared_branch(shared_executor,independent_work,
                [&]{return resident_moe(*this,source,ids,weights,limit);},source.size(0)))return *output;
        if(independent_work)independent_work();
        return resident_moe(*this,source,ids,weights,limit);
    }
    if (source.size(0) > 32) {
        if(independent_work)independent_work();
        if (source.scalar_type() == tb::kFloat16 && mfq::cuda::gguf::grouped_f16_available(source.get_device()))
            return grouped_moe_prefill(*this, source, ids, weights, limit);
        auto g = gate->forward(source, ids), u = up->forward(source, ids);
        if (limit > 0) {
            g = tb::clamp_max(g.to(tb::kFloat32), limit);
            u = tb::clamp(u.to(tb::kFloat32), -limit, limit);
        }
        auto hidden = ((g * tb::sigmoid(g)) * u).to(source.scalar_type());
        return moe_weighted_reduce_cuda(down->forward(hidden, ids), weights.to(tb::kFloat32).contiguous());
    }
    auto* window = mfq::cuda::DecodeWindow::recording();
    auto lease = window ? window->workspace<GgufGraphLease>(&gguf_graph_lease_identity,
        [] { return std::make_shared<GgufGraphLease>(); }) : std::make_shared<GgufGraphLease>();
    auto create = [&] {
        auto stage=std::make_shared<GgufMoeStage>(*this, source, ids.size(1), lease);
        stage->enable_cpu();return stage;
    };
    auto stage = window ? window->workspace<GgufMoeStage>(this, create) : create();
    if (stage->tokens != ids.size(0) || stage->routes != ids.size(1) || stage->output_type != source.scalar_type())
        throw std::invalid_argument("GGUF MoE graph geometry changed");
    // Keep graph inputs alive without recording three redundant device copies.
    // Unsupported scalar types retain the original conversion to FP32.
    stage->input=(source.scalar_type()==tb::kFloat16 || source.scalar_type()==tb::kFloat32 ||
        source.scalar_type()==tb::kBFloat16)?source:source.to(tb::kFloat32);
    stage->ids=ids;
    stage->weights=weights.to(tb::kFloat32).contiguous();
    cudaStreamCaptureStatus capture;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stage->stream.stream(), &capture));
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<std::uint64_t> reported[2]{};
        const auto bit=std::uint64_t(1)<<std::min<std::int64_t>(source.size(0),63);
        const int captured=capture!=cudaStreamCaptureStatusNone;
        if(!(reported[captured].fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_gguf_input_prepare actual_match=1 rows=%lld direct_input=1 fused_quant_cpu=1 route_copies=0 captured=%d\n",
            static_cast<long long>(source.size(0)),captured);
    }
    if (window) window->mark_current("cpu_wait_begin");
    stage->publish_routes();
    if (capture == cudaStreamCaptureStatusNone) {
        stage->wait_routes(); stage->load(limit);
    }
    if (window) {
        window->enroll({stage.get(), [stage] { stage->reset_routes(); }, [stage,limit] {
                if (stage->mapped_routes) { stage->wait_routes(); stage->load(limit); }
            }, [lease,stage] { stage->wait_cpu();lease->clear();if(stage->direct)stage->banks[0]->cache->finish_direct(); },
            [stage] {stage->cancel();}, [lease] { lease->clear(); }, [stage] {
                // The synchronized warmup has completed all consumers. Drop
                // its borrowed inputs now, before capture: their side-stream
                // lifetime fences cannot be inserted into an unrelated graph.
                stage->input={};stage->ids={};stage->weights={};
            }});
        if(!stage->direct)window->host_barrier([stage,limit] { if (!stage->mapped_routes) stage->load(limit); });
    }
    if(auto output=try_shared_branch(shared_executor,independent_work,[&]{return stage->compute(limit);},
            source.size(0),stage->resident_overlap?stage->transfer_stream.stream():nullptr))return *output;
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");independent_work && trace && trace[0]=='1') {
        static std::atomic<std::uint64_t> reported{0};
        const auto bit=std::uint64_t(1)<<source.size(0);
        if(!(reported.fetch_or(bit,std::memory_order_relaxed)&bit))std::fprintf(stderr,
            "native_gguf_shared_overlap rows=%lld route_before_shared=1 transfer_before_shared=%d\n",
            static_cast<long long>(source.size(0)),int(stage->resident_overlap));
    }
    return stage->compute(limit,independent_work);
}

tb::Tensor GgufRoutedWeight::forward(tb::Tensor source, tb::Tensor ids) const {
    cache->wait_direct_admission();
    if (!source.is_cuda() || ids.dim() != 2 || source.size(0) != ids.size(0) ||
            source.size(-1) != input || (source.dim() != 2 && source.dim() != 3) ||
            (source.dim() == 3 && source.size(1) != ids.size(1)))
        throw std::invalid_argument("GGUF routed projection input shape");
    MfqCudaGuard guard(source.get_device());
    ids = ids.to(tb::kInt64).contiguous();
    const auto tokens = ids.size(0), routes = ids.size(1);
    if (tokens <= 32) {
        auto* window = mfq::cuda::DecodeWindow::recording();
        auto lease = window ? window->workspace<GgufGraphLease>(&gguf_graph_lease_identity,
            [] { return std::make_shared<GgufGraphLease>(); }) : std::make_shared<GgufGraphLease>();
        auto create = [&] { return std::make_shared<GgufExpertStage>(*this, source, tokens, routes, lease); };
        auto stage = window ? window->workspace<GgufExpertStage>(this, create) : create();
        if (stage->tokens != tokens || stage->routes != routes || stage->input.sizes() != source.sizes())
            throw std::invalid_argument("GGUF graph geometry changed");
        stage->input.copy_(rotation?rotation->apply(source):source); stage->ids.copy_(ids);
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->host_ids->data(), stage->ids.data_ptr(),
            stage->ids.nbytes(), cudaMemcpyDeviceToHost, stage->stream.stream()));
        cudaStreamCaptureStatus capture;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stage->stream.stream(), &capture));
        if (capture == cudaStreamCaptureStatusNone) {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stage->stream.stream())); stage->load();
        }
        if (window) {
            window->enroll({stage.get(), [] {}, [] {}, [lease] { lease->clear(); },
                [] {}, [lease] { lease->clear(); }, {}});
            window->host_barrier([stage] { stage->load(); });
        }
        stage->compute();
        return stage->output.to(source.scalar_type());
    }
    // Prefill groups all occurrences of one expert, transferring its compressed
    // matrix once and sharing every decoded GEMM tile across those tokens.
    auto selected = ids.cpu().contiguous();
    std::vector<std::vector<std::int64_t>> slots(experts);
    for (std::int64_t i = 0; i < selected.numel(); ++i) {
        const auto expert = selected.data_ptr<std::int64_t>()[i];
        if (expert >= 0 && expert < experts) slots[expert].push_back(i);
    }
    auto host = tb::zeros({2, selected.numel()},
        tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64).pinned_memory(true));
    auto* positions = host.data_ptr<std::int64_t>();
    std::int64_t offset = 0;
    for (const auto& group : slots) for (const auto slot : group) {
        positions[offset] = slot;
        positions[selected.numel() + offset] = source.dim() == 2 ? slot / routes : slot;
        ++offset;
    }
    auto plan = host.to(source.device());
    auto output = tb::zeros({tokens * routes, out}, source.options());
    auto flat = source.reshape({-1, input}); offset = 0;
    for (std::int64_t expert = 0; expert < experts; ++expert) {
        const auto count = static_cast<std::int64_t>(slots[expert].size());
        if (!count) continue;
        auto weights = cache->acquire(*this, {expert}, source.get_device(),nullptr,false);
        auto x = flat.index_select(0, plan.select(0, 1).narrow(0, offset, count));
        // An unsupported MoE group and its independent projections use the
        // same FP16/FP32 arithmetic. Routed MMQ is selected for the whole FFN.
        auto values = weights[0].forward(x,false);
        output.index_copy_(0, plan.select(0, 0).narrow(0, offset, count), values);
        offset += count;
    }
    return output.reshape({tokens, routes, out});
}
