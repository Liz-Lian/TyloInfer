#include "kernels/qsa_ram.h"
#include "qsa_tq_cache.h"
#include "qsa_ram_cache.h"
#include "qsa_tq_session.h"
#include "kernels/qsa_tq_fa.h"
#include "mfq_cuda_context.h"
#include "decode_window.h"
#include "kernels/qsa_tq.h"
#include "kernels/qsa_tq_encode.h"
#include "mfq_cuda_resident_allocations.h"
#include "tyloinfer/kernels/cuda/qwen4_exp.h"
#include "tyloinfer/file_read_window.h"
#include <atomic>
#include <array>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <map>
#include <numeric>
#include <unordered_map>

namespace mfq::cuda {
namespace tb=mfq_tensor_backend;
namespace tq=tyloinfer::qsa;
namespace {
void check(int code) {TYLOINFER_CUDA_CHECK(cudaError_t(code));}
std::size_t option(const char* name,std::size_t fallback) {
    const auto* s=std::getenv(name);if(!s || !*s)return fallback;
    std::size_t used=0;const auto value=std::stoull(s,&used);
    if(used!=std::strlen(s) || s[0]=='-')throw std::invalid_argument(std::string("invalid ")+name);
    return std::size_t(value);
}
struct TqCodecBuffer {
    void* pointer=nullptr;std::size_t bytes=0;
    TqCodecBuffer()=default;TqCodecBuffer(const TqCodecBuffer&)=delete;TqCodecBuffer& operator=(const TqCodecBuffer&)=delete;
    ~TqCodecBuffer(){if(pointer)(void)cudaFree(pointer);}
    void reserve(std::size_t n) {if(n<=bytes)return;if(pointer)check(cudaFree(pointer));pointer=nullptr;bytes=0;
        check(cudaMalloc(&pointer,n));bytes=n;}
    template<class T> T* as(std::size_t offset=0)const{return reinterpret_cast<T*>(static_cast<std::uint8_t*>(pointer)+offset);}
};
struct TqCodecHostBuffer {
    void* pointer=nullptr;std::size_t bytes=0;
    TqCodecHostBuffer()=default;TqCodecHostBuffer(const TqCodecHostBuffer&)=delete;
    TqCodecHostBuffer& operator=(const TqCodecHostBuffer&)=delete;
    ~TqCodecHostBuffer(){if(pointer)(void)cudaFreeHost(pointer);}
    void reserve(std::size_t n) {
        if(n<=bytes)return;
        void* replacement=nullptr;check(cudaHostAlloc(&replacement,n,cudaHostAllocDefault));
        if(pointer)check(cudaFreeHost(pointer));pointer=replacement;bytes=n;
    }
    template<class T>T* as(std::size_t offset=0)const{return reinterpret_cast<T*>(static_cast<std::uint8_t*>(pointer)+offset);}
};
struct TqEncoderStream {
    cudaStream_t value=nullptr;
    ~TqEncoderStream(){if(value)(void)cudaStreamDestroy(value);}
};
struct TqCodecArena {
    std::mutex mutex;std::once_flag initialized;
    TqEncoderStream owner;cudaStream_t stream=nullptr;
    TqCodecBuffer source,source_views,decoded,encoded,sizes,flags,status,targets,workspace;
    TqCodecHostBuffer input,output;
    std::size_t reported_bytes=0;
    std::size_t device_bytes()const {
        return source.bytes+source_views.bytes+decoded.bytes+encoded.bytes+sizes.bytes+flags.bytes+status.bytes+targets.bytes+workspace.bytes;
    }
};
std::shared_ptr<TqCodecArena> shared_codec_arena(int device,cudaStream_t stream) {
    // Foreground imports and the background worker can run on different host
    // threads. Both must find the same arena for this model execution stream.
    static std::mutex mutex;
    static std::map<std::pair<int,cudaStream_t>,std::weak_ptr<TqCodecArena>> arenas;
    std::lock_guard<std::mutex> lock(mutex);auto& slot=arenas[{device,stream}];auto arena=slot.lock();
    if(!arena){arena=std::make_shared<TqCodecArena>();slot=arena;}return arena;
}
tb::Tensor upload(const void* data,std::size_t bytes,const tb::TensorOptions& options) {
    auto host=tb::empty({int64_t(bytes)},options.device(tb::kCPU).dtype(tb::kUInt8));
    if(bytes)std::memcpy(host.data_ptr(),data,bytes);return host.to(options.dtype(tb::kUInt8));
}
tb::Tensor signs(int width,bool value,const tb::TensorOptions& options) {
    std::uint32_t state=0x6d2b79f5u^(std::uint32_t(width)*7919u+unsigned(value));
    auto host=tb::empty({width},options.device(tb::kCPU).dtype(tb::kFloat32));
    for(int d=0;d<width;++d){state^=state<<13;state^=state>>17;state^=state<<5;host.data_ptr<float>()[d]=(state&1)?1.f:-1.f;}
    return host.to(options.dtype(tb::kFloat32));
}
}
bool qsa_tq_cache_enabled() {const auto* s=std::getenv("TYLOINFER_QSA_ADAPTIVE_TQ");return (s && std::string(s)=="1") || option("TYLOINFER_QSA_RAM_KV_BITS",0)!=0;}

struct QsaTqCache::Impl {
    // Adjacent layers share one pinned transfer image on their worker thread.
    // Weak pooling ties its lifetime to the live model caches. A lease also
    // serializes reuse if a cache later migrates to another worker thread.
    struct ArchiveArena {
        tb::Tensor host;
        mfq::cuda::StreamHandle stream;
        mfq::cuda::Event ready,done;
        std::mutex mutex;
        explicit ArchiveArena(int device):stream(mfq::cuda::stream_from_pool(false,device)){}
    };
    struct ArchiveLease {
        std::shared_ptr<ArchiveArena> arena;
        std::unique_lock<std::mutex> lock;
        bool submitted=false;
        explicit ArchiveLease(std::shared_ptr<ArchiveArena> a):arena(std::move(a)),lock(arena->mutex){}
        ~ArchiveLease(){if(submitted)(void)cudaStreamSynchronize(arena->stream.stream());}
        void wait(){if(submitted){arena->done.synchronize();submitted=false;}}
    };
    std::shared_ptr<ArchiveArena> archive_arena;
    std::unique_ptr<ArchiveLease> lease_archive() {
        if(!archive_arena) {
            static thread_local std::unordered_map<int,std::weak_ptr<ArchiveArena>> arenas;
            auto& slot=arenas[device];archive_arena=slot.lock();
            if(!archive_arena){archive_arena=std::make_shared<ArchiveArena>(device);slot=archive_arena;}
        }
        return std::make_unique<ArchiveLease>(archive_arena);
    }
    struct ImageBatch {tb::Tensor payload;std::vector<std::uint8_t> host;std::vector<tq::TqBlockView> records;};
    struct Retired {cudaEvent_t done=nullptr;std::vector<std::shared_ptr<ImageBatch>> images;};
    struct Upload {cudaEvent_t done=nullptr;std::shared_ptr<ImageBatch> image;};
    int64_t maximum=0,tokens=0;int heads,kv_heads,width,padded,index_heads,index_width,budget,rows,capacity,device=0;
    std::size_t byte_budget=0,canonical_bytes=0;unsigned interval=128;
    const tb::TensorOptions options;
    // The next use (or wait/reset) consumes this snapshot before appending
    // new blocks, preserving its observation length and token timeline.
    struct HistoryTransfer {
        tb::Tensor host;
        cudaEvent_t done=nullptr;
        bool pending=false;
        std::uint64_t tokens=0;
        std::size_t budget=0;
        int pools=0;
        ~HistoryTransfer() {
            if(pending)(void)cudaEventSynchronize(done);
            if(done)(void)cudaEventDestroy(done);
        }
    } history_transfer;
    tb::Tensor ks,vs,relative,views,index_codes,index_scales,index_zero,hits,mass;
    tb::Tensor tail_k,tail_v,tail_index,pending_k,pending_v;
    tb::Tensor tail_storage_k,tail_storage_v,tail_storage_index,pending_modes_k,pending_modes_v;
    tb::Tensor rotated_tail_k,rotated_tail_v;
    tb::Tensor append_k,append_v,append_index,archive;
    int append_capacity=0;bool append_small=true,pending_graph=false,ssd_cache=false,graph_complete=false;
    int graph_queries=1,graph_phase=0,graph_blocks=0;
    bool batch_prepared=false;
    tb::Tensor seed_k,seed_v,seed_index,seed_rotated_k,seed_rotated_v;
    tb::Tensor graph_destination,graph_archive,graph_archive_host;
    std::uint8_t* graph_archive_mapped=nullptr;
    std::shared_ptr<ImageBatch> graph_image;
    tb::Tensor canonical_slab;
    struct AttentionWorkspace {
        int queries=0,pools=0,parts=0;
        tb::Tensor ranked,blocks,rotated,partial,stats,logmass,norm,indexer;
    } decode_attention,prefill_attention,pending_attention;
    std::array<AttentionWorkspace,7> batch_attention;
    struct BatchAppendWorkspace {
        tb::Tensor key,value,index,modes_key,modes_value,fp16_key,fp16_value,rotated_key,rotated_value,packed;
    };
    std::array<BatchAppendWorkspace,7> batch_append;
    AttentionWorkspace& graph_workspace(int queries) {
        return queries==1?pending_attention:batch_attention.at(std::size_t(queries-2));
    }
    tq::PendingTail committed_tail() {
        return {tail_storage_k.data_ptr<float>(),tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
            rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>()};
    }
    tq::PendingTail seeded_tail() {
        return {seed_k.data_ptr<float>(),seed_v.data_ptr<float>(),seed_index.data_ptr<float>(),
            seed_rotated_k.data_ptr<float>(),seed_rotated_v.data_ptr<float>()};
    }
    void cancel_batch_warmup() {
        if(!batch_prepared)return;
        check(tq::tq_copy_pending_tail(seeded_tail(),committed_tail(),kv_heads,width,padded,index_width,mfq_current_cuda_stream()));
        batch_prepared=false;
    }
    std::unique_ptr<tq::TqPrefillGraph> prefill_graph;
    std::unique_ptr<tq::AdaptiveCache> cache;
    std::unique_ptr<QsaRamCache> ram;
    int ram_bits=0;
    std::shared_ptr<mfq::cache::QsaKvStore> raw_index_store;
    std::vector<std::shared_ptr<QsaIndexArchive>> index_archive;
    std::vector<std::shared_ptr<ImageBatch>> owners;
    std::vector<Retired> retired;
    std::vector<Upload> uploads;
    tq::TqBlockView pending_view;
    std::mutex queue_mutex;std::vector<tq::BlockReplacement> queued;
    std::size_t resident_append_begin=0,resident_append_end=0;
    struct ResidentAppendScope {
        Impl& owner;
        ResidentAppendScope(Impl& value,std::size_t first,std::size_t count):owner(value) {
            std::lock_guard<std::mutex> lock(owner.queue_mutex);
            owner.resident_append_begin=first;owner.resident_append_end=first+count;
        }
        ~ResidentAppendScope() {
            std::lock_guard<std::mutex> lock(owner.queue_mutex);
            owner.resident_append_begin=owner.resident_append_end=0;
        }
    };
    std::string directory;
    cudaStream_t codec_stream_key=nullptr;std::once_flag bind_encoder;
    std::shared_ptr<TqCodecArena> encoder;

    Impl(int64_t max,int h,int kh,int w,int ih,int iw,int b,const tb::TensorOptions& opts,bool ssd)
        :maximum(max),heads(h),kv_heads(kh),width(w),padded(w<=256?256:w<=512?512:1024),index_heads(ih),
         index_width(iw),budget(b),rows(0),capacity(0),options(opts),ssd_cache(ssd) {
        if(max<=0 || max>INT_MAX-3 || h<=0 || kh<=0 || h%kh || w<=0 || w>1024 || ih<=0 || iw<=0 || b<=0 || b%4)
            throw std::invalid_argument("invalid adaptive QSA geometry");
        const auto row_count=4LL*kh*(padded/256);
        // Native compact images have 32-bit lengths and integer row strides.
        if(row_count>(INT_MAX-16)/259)throw std::invalid_argument("adaptive QSA image stride exceeds 32 bits");
        rows=int(row_count);capacity=int((max+3)/4);
        ram_bits=int(option("TYLOINFER_QSA_RAM_KV_BITS",0));
        if(ram_bits && ram_bits!=16 && ram_bits!=8 && ram_bits!=4)throw std::invalid_argument("QSA RAM KV bits must be 16, 8 or 4");
        check(cudaGetDevice(&device));codec_stream_key=mfq_current_cuda_stream();
        const auto update=option("TYLOINFER_QSA_TQ_UPDATE_TOKENS",128);
        if(update!=128 && update!=256)throw std::invalid_argument("adaptive QSA update interval must be 128 or 256");
        interval=unsigned(update);
        byte_budget=option("TYLOINFER_QSA_TQ_BUDGET_BYTES",0);
        const auto* path=std::getenv("TYLOINFER_QSA_TQ_DIRECTORY");
        if(ssd_cache && (!path || !*path))throw std::invalid_argument("adaptive QSA requires TYLOINFER_QSA_TQ_DIRECTORY for its original KV cache");
        static std::atomic<std::uint64_t> next{0};
        if(ssd_cache)directory=(std::filesystem::path(path)/("sequence-"+std::to_string(++next))).string();
        ks=signs(padded,false,options);vs=signs(padded,true,options);
        std::array<__half,16> rel;for(int i=0;i<16;++i)rel[i]=__float2half_rn(std::exp2(-2.f+4.f*i/15.f));
        relative=upload(rel.data(),sizeof(rel),options);
        views=tb::zeros({capacity,int64_t(sizeof(tq::TqBlockView))},options.dtype(tb::kUInt8));
        index_codes=tb::empty({capacity,index_width},options.dtype(tb::kInt16));
        index_scales=tb::empty({capacity},options.dtype(tb::kFloat16));index_zero=tb::empty({capacity},options.dtype(tb::kInt16));
        hits=tb::zeros({capacity},options.dtype(tb::kFloat32));mass=tb::zeros({capacity},options.dtype(tb::kFloat32));
        canonical_bytes=ram_bits==16?std::size_t(rows)*512:((3*rows+3)&~3)+256*rows;
        tail_storage_k=tb::empty({1,4,kv_heads,width},options.dtype(tb::kFloat32));tail_storage_v=tb::empty_like(tail_storage_k);
        tail_storage_index=tb::empty({1,4,index_width},options.dtype(tb::kFloat32));
        rotated_tail_k=tb::empty({4,kv_heads,padded},options.dtype(tb::kFloat32));rotated_tail_v=tb::empty_like(rotated_tail_k);
        pending_modes_k=tb::empty({rows,256},options.dtype(tb::kFloat32));pending_modes_v=tb::empty_like(pending_modes_k);
        pending_k=tb::empty({rows,256},options.dtype(tb::kFloat16));pending_v=tb::empty_like(pending_k);
        initialize_cache();
    }
    ~Impl() {
        (void)cudaSetDevice(device);
        if(history_transfer.pending){(void)cudaEventSynchronize(history_transfer.done);history_transfer.pending=false;}
        ram.reset();cache.reset();(void)cudaStreamSynchronize(mfq_current_cuda_stream());
        for(auto& old:retired)if(old.done)(void)cudaEventDestroy(old.done);
        for(auto& upload:uploads)if(upload.done)(void)cudaEventDestroy(upload.done);
    }
    void initialize_cache() {
        if(ram_bits) {
            ram=std::make_unique<QsaRamCache>(capacity,rows,(budget+6)/4,ram_bits,options,views,ssd_cache,directory+"/ram-kv",
                [this](const auto& images,const auto& precision){return reencode(images,precision);});
            mfq::cache::QsaKvStoreConfig index;index.budget_bytes=index.buffer_bytes=4ULL<<20;index.directory=directory+"/index-tail";
            if(ssd_cache)raw_index_store=std::make_shared<mfq::cache::QsaKvStore>(index);
            return;
        }
        tq::AdaptiveCacheConfig config;config.directory=directory+"/kv";config.rows_per_kind=rows;
        config.ssd_cache=ssd_cache;
        config.tq_state_bits=14;config.policy.update_tokens=interval;
        cache=std::make_unique<tq::AdaptiveCache>(config,tq::AdaptiveCache::ReencodeBatch(
            [this](const auto& images,const auto& precision){return reencode(images,precision);}),
            [this](const auto& batch){
                std::lock_guard<std::mutex> lock(queue_mutex);
                for(const auto& change:batch) {
                    if(change.tier==5 && change.block>=resident_append_begin && change.block<resident_append_end)continue;
                    queued.push_back(change);
                }
            });
        mfq::cache::QsaKvStoreConfig index;index.budget_bytes=index.buffer_bytes=4ULL<<20;index.directory=directory+"/index-tail";
        if(ssd_cache)raw_index_store=std::make_shared<mfq::cache::QsaKvStore>(index);
    }
    std::size_t index_archive_bytes()const {return std::size_t(index_width)*8+(ram?std::size_t(index_width+2)*2:0);}
    void save_index(std::vector<std::uint8_t> bytes) {
        auto image=std::make_shared<QsaIndexArchive>();
        if(ssd_cache){image->store=raw_index_store;image->block=raw_index_store->write(std::move(bytes));}
        else {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            auto resident=std::make_unique<const std::vector<std::uint8_t>>(std::move(bytes));
            const auto capacity=resident->capacity();charge_tensor_host_bytes(capacity);
            image->resident=std::shared_ptr<const std::vector<std::uint8_t>>(resident.release(),
                [capacity](const std::vector<std::uint8_t>* value){delete value;tensor_host_bytes.fetch_sub(capacity);});
#else
            image->resident=std::make_shared<const std::vector<std::uint8_t>>(std::move(bytes));
#endif
        }
        index_archive.push_back(std::move(image));
    }
    std::vector<tq::MicroblockImage> reencode(const std::vector<tq::MicroblockImage>& input,const std::vector<tq::Precision>& precision) {
        if(input.size()!=precision.size() || input.size()>std::size_t(INT_MAX/2))
            throw std::invalid_argument("QSA encoder precision count mismatch");
        if(input.empty())return {};
        check(cudaSetDevice(device));
        const auto encoder_budget=option("TYLOINFER_QSA_TQ_WORKSPACE_BYTES",32ULL<<20);
        // Keep fitting at the policy's2048-token transaction granularity.
        // Larger head counts may require smaller chunks to respect the actual
        // encoder workspace; no model geometry or device whitelist is used.
        auto batch=std::min(input.size(),std::size_t(2048/4));
        tq::TqEncodePlan checked_plan;
        for(;;) {
            const auto error=tq::tq_make_encode_plan(&checked_plan,int(2*batch),rows,tq::TqEncodeProfile::Fast,encoder_budget);
            if(error==int(cudaErrorMemoryAllocation) && batch>1){batch=(batch+1)/2;continue;}
            check(error);break;
        }
        if(batch<input.size()) {
            std::vector<tq::MicroblockImage> result;result.reserve(input.size());
            for(std::size_t begin=0;begin<input.size();begin+=batch) {
                const auto end=std::min(input.size(),begin+batch);
                auto part=reencode({input.begin()+begin,input.begin()+end},{precision.begin()+begin,precision.begin()+end});
                for(auto& image:part)result.push_back(std::move(image));
            }
            return result;
        }
        std::call_once(bind_encoder,[this]{encoder=ram_bits?shared_codec_arena(device,codec_stream_key):std::make_shared<TqCodecArena>();});
        auto& arena=*encoder;std::lock_guard<std::mutex> encoder_guard(arena.mutex);
        auto& encoder_stream=arena.stream;auto& encoder_owner=arena.owner;
        auto& source=arena.source;auto& source_views=arena.source_views;auto& decoded=arena.decoded;
        auto& encoded=arena.encoded;auto& sizes=arena.sizes;auto& flags=arena.flags;
        auto& status=arena.status;auto& targets=arena.targets;auto& workspace=arena.workspace;
        auto& encoder_input=arena.input;auto& encoder_output=arena.output;
        check(cudaSetDevice(device));
        // Short contexts use only canonical Q8. Create an encoder stream when
        // the background worker first needs secondary quantization.
        std::call_once(arena.initialized,[&]{
            int least=0,greatest=0;check(cudaDeviceGetStreamPriorityRange(&least,&greatest));
            check(cudaStreamCreateWithPriority(&encoder_stream,cudaStreamNonBlocking,least));
            encoder_owner.value=encoder_stream;
        });
        const int n=int(input.size()),count=2*n;
        const auto stride=(3*rows+7)/8+3*rows+256*rows+16;
        std::size_t bytes=0;std::vector<std::size_t> offsets;
        for(int kind=0;kind<2;++kind)for(const auto& image:input) {
            const auto& src=kind?image.value:image.key;offsets.push_back(bytes);bytes+=src.size();
        }
        source.reserve(bytes);source_views.reserve(n*sizeof(tq::TqBlockView));
        decoded.reserve(std::size_t(count)*rows*256*sizeof(float));encoded.reserve(std::size_t(count)*stride);
        sizes.reserve(count*4);flags.reserve(count*4);status.reserve(4);targets.reserve(count*4);
        const auto records_at=(bytes+15)&~std::size_t(15),budgets_at=records_at+n*sizeof(tq::TqBlockView);
        encoder_input.reserve(budgets_at+count*4);
        auto* records=encoder_input.as<tq::TqBlockView>(records_at);auto* budgets=encoder_input.as<int>(budgets_at);
        for(int kind=0;kind<2;++kind)for(int i=0;i<n;++i) {
            const auto& image=kind?input[i].value:input[i].key;
            std::memcpy(encoder_input.as<std::uint8_t>(offsets[kind*n+i]),image.data(),image.size());
        }
        for(int i=0;i<n;++i) {
            records[i]={source.as<std::uint8_t>(offsets[i]),source.as<std::uint8_t>(offsets[n+i]),
                tq::tq_precision_flags(input[i].key_bits,input[i].key_state_bits),tq::tq_precision_flags(input[i].value_bits,input[i].value_state_bits)};
            budgets[i]=rows*precision[i].key_bits;budgets[n+i]=rows*precision[i].value_bits;
        }
        check(cudaMemcpyAsync(source.pointer,encoder_input.pointer,bytes,cudaMemcpyHostToDevice,encoder_stream));
        check(cudaMemcpyAsync(source_views.pointer,records,n*sizeof(records[0]),cudaMemcpyHostToDevice,encoder_stream));
        check(cudaMemcpyAsync(targets.pointer,budgets,count*4,cudaMemcpyHostToDevice,encoder_stream));
        check(tq::tq_decode_modes(source_views.as<tq::TqBlockView>(),decoded.as<float>(),decoded.as<float>(std::size_t(n)*rows*256*4),
            relative.data_ptr<std::uint16_t>(),n,kv_heads,padded,encoder_stream));
        auto plan=checked_plan;
        if(ram) {
            const auto* explicit_limit=std::getenv("TYLOINFER_QSA_BACKGROUND_ROWS");
            // The low-priority stream and actual workspace/kernel resources
            // bound asynchronous fitting. Artificially limiting it to a few
            // persistent CTAs leaves codecs queued until speculative rollback.
            const bool urgent=ram->encoding_urgent();
            const auto limit=option("TYLOINFER_QSA_BACKGROUND_ROWS",0);
            if(limit>INT_MAX)throw std::invalid_argument("QSA encoder row limit exceeds 32 bits");
            if(limit)check(tq::tq_limit_encode_waves(&plan,int(limit)));
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ")) {
                static thread_local unsigned reported=0;const auto category=urgent?1u:2u;
                if(!(reported&category)){reported|=category;
                    std::fprintf(stderr,"native_qsa_ram_codec actual_match=1 urgent=%d active_row_limit=%zu explicit_limit=%d\n",
                        int(urgent),limit,int(explicit_limit && *explicit_limit));}
            }
        }
        workspace.reserve(plan.workspace_bytes);
        check(tq::tq_encode_modes(plan,decoded.as<float>(),targets.as<int>(),0,1.1f,encoded.as<std::uint8_t>(),stride,
            sizes.as<std::uint32_t>(),flags.as<std::uint32_t>(),status.as<int>(),workspace.pointer,workspace.bytes,encoder_stream));
        const auto packed_bytes=std::size_t(count)*stride,lengths_at=(packed_bytes+15)&~std::size_t(15);
        const auto tags_at=lengths_at+count*4,error_at=tags_at+count*4;
        encoder_output.reserve(error_at+4);
        if(std::getenv("TYLOINFER_TRACE_QSA_TQ") && arena.reported_bytes!=arena.device_bytes()) {
            arena.reported_bytes=arena.device_bytes();
            std::fprintf(stderr,"native_qsa_codec_workspace actual_match=1 shared_stream_arena=%d device_bytes=%zu host_bytes=%zu\n",
                int(ram_bits!=0),arena.reported_bytes,encoder_input.bytes+encoder_output.bytes);
        }
        auto* packed=encoder_output.as<std::uint8_t>();auto* lengths=encoder_output.as<std::uint32_t>(lengths_at);
        auto* tags=encoder_output.as<std::uint32_t>(tags_at);auto* error=encoder_output.as<int>(error_at);
        // Pageable D2H may wait inside the driver's context lock. A model
        // graph can meanwhile wait for a host-served expert DMA that needs
        // that same lock. Pinned endpoints let every copy submit before the
        // encoder waits, preserving progress on both streams.
        check(cudaMemcpyAsync(packed,encoded.pointer,packed_bytes,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(lengths,sizes.pointer,count*4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(tags,flags.pointer,count*4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(error,status.pointer,4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaStreamSynchronize(encoder_stream));check(*error);
        std::vector<tq::MicroblockImage> result(n);
        for(int i=0;i<n;++i) {
            if(lengths[i]>stride || lengths[n+i]>stride)throw std::runtime_error("QSA encoder returned an invalid packed length");
            auto& dst=result[i];dst.key.assign(packed+std::size_t(i)*stride,packed+std::size_t(i)*stride+lengths[i]);
            dst.value.assign(packed+std::size_t(n+i)*stride,packed+std::size_t(n+i)*stride+lengths[n+i]);
            dst.key_bits=std::uint8_t(tags[i]);dst.value_bits=std::uint8_t(tags[n+i]);dst.key_state_bits=dst.value_state_bits=14;
        }
        return result;
    }
    void retire(std::vector<std::shared_ptr<ImageBatch>> previous) {
        if(previous.empty())return;Retired old;old.images=std::move(previous);
        check(cudaEventCreateWithFlags(&old.done,cudaEventDisableTiming));check(cudaEventRecord(old.done,mfq_current_cuda_stream()));retired.push_back(std::move(old));
    }
    void ensure_owner_slots(std::size_t count) {
        if(count>std::size_t(capacity))throw std::out_of_range("QSA append owner range exceeds capacity");
        if(owners.size()<count)owners.resize(count);
    }
    void retain_images(const std::shared_ptr<ImageBatch>& group,const std::vector<std::size_t>& ids) {
        if(!ids.empty())ensure_owner_slots(*std::max_element(ids.begin(),ids.end())+1);
        std::vector<std::shared_ptr<ImageBatch>> previous;
        previous.reserve(ids.size());
        for(const auto id:ids) {
            if(owners[id])previous.push_back(std::move(owners[id]));owners[id]=group;
        }
        retire(std::move(previous));
    }
    void publish_images(const std::shared_ptr<ImageBatch>& group,const std::vector<std::size_t>& ids) {
        bool contiguous=!ids.empty();
        for(std::size_t i=1;i<ids.size();++i)contiguous&=ids[i]==ids[0]+i;
        if(contiguous)check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+ids[0]*sizeof(tq::TqBlockView),
            group->records.data(),ids.size()*sizeof(tq::TqBlockView),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        for(std::size_t i=0;i<ids.size();++i) {
            const auto id=ids[i];const auto& view=group->records[i];
            if(!contiguous)check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+id*sizeof(view),&view,sizeof(view),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        }
        Upload submitted;submitted.image=group;check(cudaEventCreateWithFlags(&submitted.done,cudaEventDisableTiming));
        check(cudaEventRecord(submitted.done,mfq_current_cuda_stream()));uploads.push_back(std::move(submitted));
        retain_images(group,ids);
    }
    AttentionWorkspace& prepare_workspace(int queries,int pools,int parts,bool pending=false) {
        const bool small=queries<=8;
        // Captured GPU-only calls retain these addresses across eager prefill
        // and multi-row verification, which may resize the other arenas.
        auto& a=pending?graph_workspace(queries):small?decode_attention:prefill_attention;
        if(a.queries>=queries && a.pools>=pools && a.parts==parts)return a;
        // These buffers survive this call and may coexist with a borrowed
        // expert workspace. Never retain an expert-loan allocation here.
        ResidentAllocationScope resident;
        const int count=(budget+6)/4;
        int p=small?capacity:std::max(1,a.pools);
        while(p<pools)p=std::min(capacity,p>capacity/2?capacity:2*p);
        const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
        const auto score_queries=std::max<std::size_t>(16,score_budget/(sizeof(float)*std::size_t(p)));
        const int q=small?std::max(queries,a.queries):std::max(queries,int(std::min<std::size_t>(a.queries,score_queries)));
        a.queries=q;a.pools=p;a.parts=parts;
        const auto f=options.dtype(tb::kFloat32);
        const auto indexer_bytes=tq::indexer_affine16_score_scratch(q,index_heads,index_width);
        a.indexer=indexer_bytes?tb::empty({int64_t(indexer_bytes)},options.dtype(tb::kUInt8)):tb::Tensor{};
        a.ranked=tb::empty({int64_t(q)*p},f);a.blocks=tb::empty({int64_t(q)*count*int64_t(sizeof(tq::SelectedBlock))},options.dtype(tb::kUInt8));
        a.rotated=tb::empty({int64_t(q)*heads*padded},f);
        auto partial_elements=int64_t(q)*heads*parts*padded;
        if(!small && padded==256) {
            const auto bytes=tq::tq_prefill_attention_scratch(q,heads,4*p);
            partial_elements=std::max(partial_elements,int64_t((bytes+3)/4));
        }
        a.partial=tb::empty({partial_elements},f);
        a.stats=tb::empty({int64_t(q)*heads*parts*2},f);a.logmass=tb::empty({int64_t(q)*heads*count},f);a.norm=tb::empty({int64_t(q)*heads},f);
        return a;
    }
    void indexer_scores(const tb::Tensor& query,const tb::Tensor& positions,tb::Tensor output,
                        int queries,int pools,AttentionWorkspace& scratch,bool batched_prefill=false) {
        // Small forward calls preserve decode numerics, including eager warmup.
        // Batched prefill retains its caller-owned integer query workspace.
        if(batched_prefill) {
            check(tq::indexer_affine16_scores(query.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),
                index_scales.data_ptr<std::uint16_t>(),index_zero.data_ptr<std::int16_t>(),positions.data_ptr<int64_t>(),
                output.data_ptr<float>(),queries,index_heads,index_width,pools,mfq_current_cuda_stream(),
                scratch.indexer.defined()?scratch.indexer.data_ptr():nullptr,
                scratch.indexer.defined()?scratch.indexer.nbytes():0));
            return;
        }
        check(tq::indexer_affine16_scores(query.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),
            index_scales.data_ptr<std::uint16_t>(),index_zero.data_ptr<std::int16_t>(),positions.data_ptr<int64_t>(),
            output.data_ptr<float>(),queries,index_heads,index_width,pools,mfq_current_cuda_stream()));
    }
    void drain() {
        for(auto it=uploads.begin();it!=uploads.end();) {
            const auto result=cudaEventQuery(it->done);
            if(result==cudaSuccess) {
                check(cudaEventDestroy(it->done));std::vector<std::uint8_t>().swap(it->image->host);
                std::vector<tq::TqBlockView>().swap(it->image->records);it=uploads.erase(it);
            }else {if(result!=cudaErrorNotReady)check(result);++it;}
        }
        for(auto it=retired.begin();it!=retired.end();) {
            const auto result=cudaEventQuery(it->done);
            if(result==cudaSuccess){check(cudaEventDestroy(it->done));it=retired.erase(it);}
            else {if(result!=cudaErrorNotReady)check(result);++it;}
        }
        if(ram){ram->drain();return;}
        std::vector<tq::BlockReplacement> batch;
        {std::lock_guard<std::mutex> lock(queue_mutex);batch.swap(queued);}
        if(batch.empty())return;
        auto group=std::make_shared<ImageBatch>();std::vector<std::size_t> key_offsets,value_offsets;
        for(const auto& entry:batch) {
            key_offsets.push_back(group->host.size());group->host.insert(group->host.end(),entry.image->key.begin(),entry.image->key.end());
            value_offsets.push_back(group->host.size());group->host.insert(group->host.end(),entry.image->value.begin(),entry.image->value.end());
        }
        group->payload=upload(group->host.data(),group->host.size(),options);
        group->records.resize(batch.size());
        std::vector<std::size_t> ids;
        for(std::size_t i=0;i<batch.size();++i) {
            const auto& entry=batch[i];const auto id=entry.block;
            ids.push_back(id);
            auto& view=group->records[i];view={group->payload.data_ptr<std::uint8_t>()+key_offsets[i],group->payload.data_ptr<std::uint8_t>()+value_offsets[i],
                tq::tq_precision_flags(entry.image->key_bits,entry.image->key_state_bits),tq::tq_precision_flags(entry.image->value_bits,entry.image->value_state_bits)};
        }
        // The upload owner retains every host descriptor until its event.
        publish_images(group,ids);
    }
    void publish_pending(const tb::Tensor& k,const tb::Tensor& v,int64_t id) {
        pending_k=k;pending_v=v;
        pending_view={k.data_ptr<std::uint8_t>(),v.data_ptr<std::uint8_t>(),0,0};
        check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+id*sizeof(pending_view),&pending_view,sizeof(pending_view),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
    }
    bool graph_host_boundary=false,graph_ssd_task=false;
    tb::Tensor stage_pending(const tb::Tensor& selected,int queries,int count,const tb::Tensor& positions) {
        if(!ram)return views;
        if(graph_ssd_task)return ram->stage_graph_task(selected,queries,count,views,positions,queries);
        if(!graph_host_boundary)return ram->stage(selected,queries,count,views,positions,tokens,queries);
        auto* window=DecodeWindow::recording();
        if(!window)throw std::logic_error("SSD QSA graph requires its decode window host boundary");
        cudaStreamCaptureStatus capture;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture));
        const bool selection_in_graph=ram->collect_graph_selection(selected,queries,count);
        // Only the selected RAM/SSD images need a host boundary. Append,
        // index scoring, selection and attention keep their captured GPU work.
        // Read the live epoch when replaying, rather than retaining the epoch
        // of the first capture in LRU updates.
        window->host_barrier([this,selected,queries,count,positions,selection_in_graph] {
            ram->stage(selected,queries,count,views,positions,tokens,queries,selection_in_graph);
        });
        if(capture==cudaStreamCaptureStatusNone) {
            check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            ram->stage(selected,queries,count,views,positions,tokens,queries,selection_in_graph);
        }
        return ram->prepared_views();
    }
    tb::Tensor forward_pending_batch(const tb::Tensor& query,const tb::Tensor& key,const tb::Tensor& value,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const tb::Tensor& gate,tb::Tensor output,
        const QsaTqCache::Pool& pool,const tb::Tensor& absolute_positions) {
        const int n=int(key.size(1));
        if(!pending_graph || !batch_prepared || n!=graph_queries || !output.defined() || !pool)
            throw std::logic_error("QSA batch graph was not prepared");
        const auto convert=[](const tb::Tensor& x) {
            auto y=(x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)?x:x.to(tb::kFloat32);
            return y.stride(y.dim()-1)==1?y:y.contiguous();
        };
        auto k=convert(key),v=convert(value),r=convert(raw),q=convert(query),g=convert(gate);
        auto& a=graph_workspace(n);auto& batch=batch_append.at(std::size_t(n-2));
        const int staged=int(batch.key.size(1)),blocks=staged/4;
        check(tq::tq_stage_graph_batch(k.data_ptr(),v.data_ptr(),r.data_ptr(),k.scalar_type()==tb::kFloat16,v.scalar_type()==tb::kFloat16,r.scalar_type()==tb::kFloat16,
            seeded_tail(),positions.data_ptr<int64_t>(),batch.key.data_ptr<float>(),batch.value.data_ptr<float>(),batch.index.data_ptr<float>(),
            n,staged,kv_heads,width,index_width,k.stride(1),k.stride(2),v.stride(1),v.stride(2),r.stride(1),mfq_current_cuda_stream()));
        check(tq::tq_prepare_modes(batch.key.data_ptr(),batch.value.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
            batch.modes_key.data_ptr<float>(),batch.modes_value.data_ptr<float>(),batch.fp16_key.data_ptr<std::uint16_t>(),batch.fp16_value.data_ptr<std::uint16_t>(),
            staged,kv_heads,width,padded,mfq_current_cuda_stream(),batch.rotated_key.data_ptr<float>(),batch.rotated_value.data_ptr<float>()));
        const auto* canonical_k=reinterpret_cast<const std::uint8_t*>(batch.fp16_key.data_ptr());
        const auto* canonical_v=reinterpret_cast<const std::uint8_t*>(batch.fp16_value.data_ptr());
        if(ram_bits!=16) {
            auto* packed=batch.packed.data_ptr<std::uint8_t>();
            check(tq::tq_prepare_affine8(batch.key.data_ptr(),batch.value.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                packed,packed+blocks*canonical_bytes,staged,kv_heads,width,padded,mfq_current_cuda_stream()));
            canonical_k=packed;canonical_v=packed+blocks*canonical_bytes;
        }
        auto absolute=absolute_positions.defined()?absolute_positions:positions;
        if(absolute.dim()==1)absolute=absolute.reshape({1,n});
        auto first=absolute.narrow(-1,0,1)-positions.reshape({-1}).narrow(0,0,1).remainder(4);
        auto starts=tb::arange(blocks,positions.options())*4+first;
        auto last=absolute.narrow(-1,n-1,1);
        starts=tb::where(starts<=last,starts,last);
        auto pooled=pool(batch.index,starts).to(tb::kFloat32).contiguous();
        check(tq::indexer_affine16_append_batch(pooled.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),
            index_zero.data_ptr<std::int16_t>(),positions.data_ptr<int64_t>(),blocks,index_width,capacity,mfq_current_cuda_stream()));
        tq::PendingBatchPublication publication{
            positions.data_ptr<int64_t>(),canonical_k,canonical_v,batch.fp16_key.data_ptr<std::uint16_t>(),batch.fp16_value.data_ptr<std::uint16_t>(),
            batch.key.data_ptr<float>(),batch.value.data_ptr<float>(),batch.index.data_ptr<float>(),batch.rotated_key.data_ptr<float>(),batch.rotated_value.data_ptr<float>(),
            index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),index_zero.data_ptr<std::int16_t>(),graph_destination.data_ptr<tq::TqBlockView>(),
            views.data_ptr<tq::TqBlockView>(),graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>(),committed_tail(),
            n,kv_heads,width,padded,index_width,int(canonical_bytes),int(interval/4),bool(ram)};
        check(tq::tq_publish_graph_batch(publication,mfq_current_cuda_stream()));
        if(!graph_archive_mapped)check(cudaMemcpyAsync(graph_archive_host.data_ptr(),graph_archive.data_ptr(),graph_archive.numel(),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        const int count=(budget+6)/4;
        auto ranked=a.ranked.narrow(0,0,int64_t(n)*capacity).reshape({1,n,capacity});auto index_query=iq.to(tb::kFloat32).contiguous();
        indexer_scores(index_query,positions,ranked,n,capacity,a);
        auto selected=mfq_qwen4_exp::live_topk_selected_blocks(ranked,positions,4,std::min(budget/4,capacity),budget);
        cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&status));const bool observe=status!=cudaStreamCaptureStatusNone;
        const auto operands=stage_pending(selected,n,count,positions);
        check(tq::tq_sparse_attention(q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),
            selected.data_ptr<tq::SelectedBlock>(),positions.data_ptr<int64_t>(),output.data_ptr(),a.rotated.data_ptr<float>(),a.partial.data_ptr<float>(),a.stats.data_ptr<float>(),
            a.logmass.data_ptr<float>(),a.norm.data_ptr<float>(),observe?hits.data_ptr<float>():nullptr,observe?mass.data_ptr<float>():nullptr,n,heads,kv_heads,width,padded,count,a.parts,
            q.stride(2),q.stride(1),mfq_current_cuda_stream(),bool(ram),g.data_ptr(),g.scalar_type()==tb::kFloat16,output.scalar_type()==tb::kFloat16,
            g.stride(1),g.stride(2),q.scalar_type()==tb::kFloat16,true,false));
        return output;
    }
    tb::Tensor forward_pending_graph(const tb::Tensor& query,const tb::Tensor& key,const tb::Tensor& value,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const tb::Tensor& gate,tb::Tensor output,
        const QsaTqCache::Pool& pool,const tb::Tensor& absolute_positions) {
        if(key.size(1)>1)return forward_pending_batch(query,key,value,iq,raw,positions,gate,std::move(output),pool,absolute_positions);
        if(!pending_graph || key.size(1)!=1 || !output.defined())throw std::logic_error("QSA device graph was not prepared");
        const auto mark=[](const char* phase){if(auto* window=DecodeWindow::recording())window->mark_current(phase);};
        mark("qsa_cache_begin");
        const auto convert=[](const tb::Tensor& x) {
            auto y=(x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32)?x:x.to(tb::kFloat32);
            return y.stride(y.dim()-1)==1?y:y.contiguous();
        };
        auto k=convert(key),v=convert(value),r=convert(raw),q=convert(query),g=convert(gate);
        auto& a=pending_attention;
        const tq::PendingQuery prepared_query{q.data_ptr(),a.rotated.data_ptr<float>(),heads,q.stride(1),q.scalar_type()==tb::kFloat16};
        check(tq::tq_append_pending(k.data_ptr(),v.data_ptr(),r.data_ptr(),k.scalar_type()==tb::kFloat16,v.scalar_type()==tb::kFloat16,r.scalar_type()==tb::kFloat16,
            positions.data_ptr<int64_t>(),tail_storage_k.data_ptr<float>(),tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
            rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>(),
            ks.data_ptr<float>(),vs.data_ptr<float>(),pending_k.data_ptr<std::uint16_t>(),pending_v.data_ptr<std::uint16_t>(),views.data_ptr<tq::TqBlockView>(),
            kv_heads,width,padded,index_width,k.stride(2),v.stride(2),mfq_current_cuda_stream(),
            graph_complete?pending_modes_k.data_ptr<float>():nullptr,graph_complete?pending_modes_v.data_ptr<float>():nullptr,&prepared_query));
        if(graph_complete) {
            if(!pool)throw std::logic_error("complete QSA graph requires indexer pooling");
            check(tq::tq_finalize_pending(pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),
                graph_destination.data_ptr<tq::TqBlockView>(),views.data_ptr<tq::TqBlockView>(),positions.data_ptr<int64_t>(),
                graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>(),rows,index_width,int(interval/4),mfq_current_cuda_stream()));
            auto starts=absolute_positions.defined()?absolute_positions:positions;
            if(starts.dim()==1)starts=starts.reshape({1,1});
            starts=starts-3;
            auto pooled=pool(tail_storage_index,starts).to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_append(pooled.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),
                index_zero.data_ptr<std::int16_t>(),positions.data_ptr<int64_t>(),index_width,mfq_current_cuda_stream()));
            if(ram)check(tq::tq_archive_indexer(index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),index_zero.data_ptr<std::int16_t>(),
                graph_archive_mapped?graph_archive_mapped:graph_archive.data_ptr<std::uint8_t>(),2*canonical_bytes+8*index_width,
                1,index_width,0,positions.data_ptr<int64_t>(),mfq_current_cuda_stream()));
            if(!graph_archive_mapped)check(cudaMemcpyAsync(graph_archive_host.data_ptr(),graph_archive.data_ptr(),graph_archive.numel(),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        }
        const int columns=budget+3,count=(columns+3)/4,parts=a.parts;
        mark("qsa_cache_append_end");
        auto ranked=a.ranked.narrow(0,0,capacity).reshape({1,1,capacity});auto index_query=iq.to(tb::kFloat32).contiguous();
        indexer_scores(index_query,positions,ranked,1,capacity,a);
        mark("qsa_cache_scores_end");
        auto blocks=mfq_qwen4_exp::live_topk_selected_blocks(ranked,positions,4,std::min(budget/4,capacity),budget);
        mark("qsa_cache_select_end");
        // Graph preparation runs this body repeatedly without committing a
        // token. Only the recorded replay may update its hit/attention history.
        cudaStreamCaptureStatus capture_status;
        check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture_status));
        const bool observe=capture_status!=cudaStreamCaptureStatusNone;
        const auto operands=stage_pending(blocks,1,count,positions);
        if(ram)mark("qsa_cache_ram_staging_end");
        check(tq::tq_sparse_attention(q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),
            blocks.data_ptr<tq::SelectedBlock>(),positions.data_ptr<int64_t>(),output.data_ptr(),a.rotated.data_ptr<float>(),a.partial.data_ptr<float>(),a.stats.data_ptr<float>(),
            a.logmass.data_ptr<float>(),a.norm.data_ptr<float>(),observe?hits.data_ptr<float>():nullptr,observe?mass.data_ptr<float>():nullptr,1,heads,kv_heads,width,padded,count,parts,
            q.stride(2),q.stride(1),mfq_current_cuda_stream(),bool(ram),g.data_ptr(),g.scalar_type()==tb::kFloat16,output.scalar_type()==tb::kFloat16,
            g.stride(1),g.stride(2),q.scalar_type()==tb::kFloat16,true,true));
        mark("qsa_cache_attention_end");
        return output;
    }
    void append_graph_image() {
        finish_history();
        const auto id=std::size_t(tokens/4);
        // DecodeWindow has completed the entire model, including this pinned
        // archive copy. No layer boundary or additional GPU wait is required.
        if(!ram){ensure_owner_slots(id+1);owners[id]=std::move(graph_image);}
        const auto* bytes=graph_archive_host.data_ptr<std::uint8_t>();
        tq::MicroblockImage image;image.key.assign(bytes,bytes+canonical_bytes);
        image.value.assign(bytes+canonical_bytes,bytes+2*canonical_bytes);
        if(ram_bits==16)image.key_bits=image.value_bits=0;
        if(ram){ram->append({std::move(image)},std::uint64_t(id)*4,true);graph_image.reset();}
        else {ResidentAppendScope publication(*this,id,1);cache->append(std::move(image),std::uint64_t(id)*4);}
        const auto* raw=bytes+2*canonical_bytes;
        save_index(std::vector<std::uint8_t>(raw,raw+index_archive_bytes()));
    }
    void append_graph_batch() {
        finish_history();
        const auto first=std::size_t(tokens/4),record_bytes=2*canonical_bytes+index_archive_bytes();
        const auto* archived=graph_archive_host.data_ptr<std::uint8_t>();
        std::vector<tq::MicroblockImage> images;images.reserve(graph_blocks);
        for(int i=0;i<graph_blocks;++i) {
            const auto* source=archived+i*record_bytes;tq::MicroblockImage image;
            image.key.assign(source,source+canonical_bytes);image.value.assign(source+canonical_bytes,source+2*canonical_bytes);
            if(ram_bits==16)image.key_bits=image.value_bits=0;
            images.push_back(std::move(image));
            const auto* raw=source+2*canonical_bytes;save_index(std::vector<std::uint8_t>(raw,raw+index_archive_bytes()));
            if(!ram){ensure_owner_slots(first+i+1);owners[first+i]=graph_image;}
        }
        if(ram)ram->append(std::move(images),std::uint64_t(first)*4,true);
        else {ResidentAppendScope publication(*this,first,images.size());cache->append_batch(std::move(images),std::uint64_t(first)*4);}
        graph_image.reset();
    }
    void finish_history(bool wait=true) {
        auto& h=history_transfer;
        if(!h.pending)return;
        const auto ready=wait?cudaEventSynchronize(h.done):cudaEventQuery(h.done);
        if(!wait && ready==cudaErrorNotReady)return;
        check(ready);
        std::vector<double> hits(h.pools),attention(h.pools);
        const auto* source=h.host.data_ptr<float>();
        for(int i=0;i<h.pools;++i){hits[i]=source[i];attention[i]=source[h.pools+i];}
        if(ram)ram->observe(hits,h.tokens);
        else cache->update(h.tokens,h.budget,std::move(hits),std::move(attention));
        h.pending=false;
    }
    void update_history(int64_t start) {
        if((byte_budget || ram) && tokens/interval!=start/interval) {
            finish_history();
            auto& h=history_transfer;const int pools=int(tokens/4);
            if(!h.host.defined() || h.host.numel()<int64_t(2)*pools)
                h.host=tb::empty({int64_t(2)*pools},options.device(tb::kCPU).dtype(tb::kFloat32).pinned_memory(true));
            if(!h.done)check(cudaEventCreateWithFlags(&h.done,cudaEventDisableTiming));
            auto* destination=h.host.data_ptr<float>();const auto bytes=std::size_t(pools)*sizeof(float);
            const auto stream=mfq_current_cuda_stream();
            check(cudaMemcpyAsync(destination,hits.data_ptr(),bytes,cudaMemcpyDeviceToHost,stream));
            check(cudaMemcpyAsync(destination+pools,mass.data_ptr(),bytes,cudaMemcpyDeviceToHost,stream));
            check(cudaEventRecord(h.done,stream));
            h.tokens=tokens;h.budget=byte_budget;h.pools=pools;h.pending=true;
            hits.zero_();mass.zero_();
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> printed{false};
                if(!printed.exchange(true))std::fprintf(stderr,"native_qsa_history_transfer actual_match=1 pinned_async=1 deferred_update=1\n");
            }
        }
    }
    tb::Tensor forward(const tb::Tensor& query,const tb::Tensor& keys,const tb::Tensor& values,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const QsaTqCache::Pool& pool,bool history_only,const tb::Tensor& gate,tb::Tensor output,bool fp16_math) {
        cancel_batch_warmup();finish_history();
        const auto t=keys.size(1),start=tokens;
        pending_graph=false;
        append_small=t<=8;
        if(keys.size(0)!=1 || t<=0 || t>maximum-tokens || keys.size(2)!=kv_heads || keys.size(3)!=width ||
           values.sizes()!=keys.sizes() || raw.size(0)!=1 || raw.size(1)!=t || raw.size(2)!=index_width)
            throw std::invalid_argument("adaptive QSA input geometry mismatch");
        const auto supported=[](const tb::Tensor& x){return x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32;};
        auto input_k=supported(keys)?keys:keys.to(tb::kFloat32);if(input_k.stride(3)!=1)input_k=input_k.contiguous();
        auto input_v=supported(values)?values:values.to(tb::kFloat32);if(input_v.stride(3)!=1)input_v=input_v.contiguous();
        auto input_r=supported(raw)?raw:raw.to(tb::kFloat32);if(input_r.stride(2)!=1)input_r=input_r.contiguous();
        const int prior=int(tokens%4),staged=int(t)+prior;
        if(append_capacity<staged) {
            ResidentAllocationScope resident;
            append_capacity=staged;
            append_k=tb::empty({1,append_capacity,kv_heads,width},options.dtype(tb::kFloat32));append_v=tb::empty_like(append_k);
            append_index=tb::empty({1,append_capacity,index_width},options.dtype(tb::kFloat32));
        }
        auto k=append_k.narrow(1,0,staged),v=append_v.narrow(1,0,staged),r=append_index.narrow(1,0,staged);
        check(tq::tq_stage_append(input_k.data_ptr(),input_v.data_ptr(),input_r.data_ptr(),input_k.scalar_type()==tb::kFloat16,
            input_v.scalar_type()==tb::kFloat16,input_r.scalar_type()==tb::kFloat16,prior?tail_k.data_ptr<float>():nullptr,
            prior?tail_v.data_ptr<float>():nullptr,prior?tail_index.data_ptr<float>():nullptr,k.data_ptr<float>(),v.data_ptr<float>(),r.data_ptr<float>(),
            int(t),prior,kv_heads,width,index_width,input_k.stride(1),input_k.stride(2),input_v.stride(1),input_v.stride(2),input_r.stride(1),mfq_current_cuda_stream()));
        const auto base=tokens/4,complete=k.size(1)/4,remainder=k.size(1)%4;
        tb::Tensor cpu_archive;
        std::unique_ptr<ArchiveLease> archive_lease;
        if(complete) {
            const auto span=complete*4;
            auto group=std::make_shared<ImageBatch>();
            group->payload=tb::empty({int64_t(2*complete*canonical_bytes)},options.dtype(tb::kUInt8));
            auto* canonical_k=group->payload.data_ptr<std::uint8_t>();auto* canonical_v=canonical_k+complete*canonical_bytes;
            if(ram_bits==16) {
                auto modes=tb::empty({2*complete*rows*256},options.dtype(tb::kFloat32));
                check(tq::tq_prepare_modes(k.data_ptr(),v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                    modes.data_ptr<float>(),modes.data_ptr<float>()+complete*rows*256,
                    reinterpret_cast<std::uint16_t*>(canonical_k),reinterpret_cast<std::uint16_t*>(canonical_v),
                    int(span),kv_heads,width,padded,mfq_current_cuda_stream()));
            }else check(tq::tq_prepare_affine8(k.data_ptr(),v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                canonical_k,canonical_v,int(span),kv_heads,width,padded,mfq_current_cuda_stream()));
            const auto archive_bytes=2*complete*canonical_bytes+complete*index_archive_bytes();
            if(!archive.defined() || archive.numel()<archive_bytes) {
                ResidentAllocationScope resident;
                archive=tb::empty({int64_t(archive_bytes)},options.dtype(tb::kUInt8));
            }
            check(tq::tq_pack_archive(canonical_k,canonical_v,r.data_ptr<float>(),archive.data_ptr<std::uint8_t>(),int(complete),int(canonical_bytes),index_width,mfq_current_cuda_stream()));
            auto origin=positions.narrow(-1,0,1)-start;
            if(origin.dim()==1)origin=origin.reshape({1,1});
            auto block_positions=tb::arange(complete,positions.options().dtype(tb::kInt64))*4+base*4+origin;
            auto pooled=pool(r.narrow(1,0,span),block_positions).to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_encode(pooled.data_ptr<float>(),index_codes.data_ptr<std::int16_t>()+base*index_width,
                index_scales.data_ptr<std::uint16_t>()+base,index_zero.data_ptr<std::int16_t>()+base,int(complete),index_width,mfq_current_cuda_stream()));
            if(ram)check(tq::tq_archive_indexer(index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),index_zero.data_ptr<std::int16_t>(),
                archive.data_ptr<std::uint8_t>(),2*complete*canonical_bytes+8*complete*index_width,int(complete),index_width,base,nullptr,mfq_current_cuda_stream()));
            {
                archive_lease=lease_archive();auto& arena=*archive_lease->arena;
                if(!arena.host.defined() || arena.host.numel()<archive_bytes)
                    arena.host=tb::empty({int64_t(archive_bytes)},options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
                cpu_archive=arena.host.narrow(0,0,archive_bytes);
                arena.ready.record(mfq_current_cuda_stream());
                check(cudaStreamWaitEvent(arena.stream.stream(),arena.ready.get(),0));
                archive_lease->submitted=true;
                check(cudaMemcpyAsync(cpu_archive.data_ptr(),archive.data_ptr(),archive_bytes,cudaMemcpyDeviceToHost,arena.stream.stream()));
                arena.done.record(arena.stream.stream());mfq_cuda_record_stream(archive,arena.stream);
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                    static std::atomic<unsigned> printed{0};const unsigned category=t<=8?1u:2u;
                    if(!(printed.fetch_or(category)&category))std::fprintf(stderr,"native_qsa_archive_overlap actual_match=1 queries=%lld bytes=%lld shared_pinned_arena=1 side_stream=1 device_descriptors=1\n",static_cast<long long>(t),static_cast<long long>(archive_bytes));
                }
            }
            // Fixed-size canonical planes need no host descriptor upload.
            // Publishing on the model stream lets the pinned archive copy run
            // while selection and attention consume these same GPU images.
            const unsigned tag=ram_bits==16?0:254;
            check(tq::tq_publish_contiguous_views({canonical_k,canonical_v,tag,tag},views.data_ptr<tq::TqBlockView>()+base,
                int(canonical_bytes),int(complete),mfq_current_cuda_stream()));
            std::vector<std::size_t> ids;ids.reserve(complete);
            for(int64_t i=0;i<complete;++i)ids.push_back(std::size_t(base+i));
            retain_images(group,ids);
        }
        bool archive_committed=false;
        const auto commit_archive=[&] {
            if(archive_committed)return;
            archive_committed=true;
            if(archive_lease)archive_lease->wait();
            // Bound transient images when SSD storage does not retain them.
            // GPU attention and its compute chunk are independent of this batch.
            const auto batch_blocks=ssd_cache
                ?std::max<std::size_t>(1,(8ULL<<20)/(2*canonical_bytes))
                :std::max<std::size_t>(1,std::size_t(complete));
            std::vector<tq::MicroblockImage> images;
            images.reserve(std::min<std::size_t>(std::size_t(complete),batch_blocks));
            for(int64_t i=0;i<complete;++i) {
                tq::MicroblockImage image;
                const auto* a=cpu_archive.data_ptr<std::uint8_t>()+i*canonical_bytes;const auto* b=cpu_archive.data_ptr<std::uint8_t>()+(complete+i)*canonical_bytes;
                image.key.assign(a,a+canonical_bytes);image.value.assign(b,b+canonical_bytes);
                if(ram_bits==16)image.key_bits=image.value_bits=0;
                images.push_back(std::move(image));
                const auto* index=cpu_archive.data_ptr<std::uint8_t>()+2*complete*canonical_bytes+i*4*index_width*2;
                // Allocate the complete image once. Appending pooled codes to
                // a raw-index vector previously grew every microblock's capacity.
                std::vector<std::uint8_t> bytes(index_archive_bytes());
                std::memcpy(bytes.data(),index,8*index_width);
                if(ram) {
                    const auto* codes=cpu_archive.data_ptr<std::uint8_t>()+2*complete*canonical_bytes+8*complete*index_width+i*(index_width+2)*2;
                    std::memcpy(bytes.data()+8*index_width,codes,(index_width+2)*2);
                }
                save_index(std::move(bytes));
                if(images.size()==batch_blocks || i+1==complete) {
                    const auto first=std::size_t(base+i+1)-images.size();
                    if(ram) {
                        const auto n=images.size();ram->append(std::move(images),std::uint64_t(first)*4,t<=8);
                        std::vector<std::shared_ptr<ImageBatch>> previous;
                        for(std::size_t id=first;id<first+n;++id)if(owners[id])previous.push_back(std::move(owners[id]));
                        retire(std::move(previous));
                    }else {
                        ResidentAppendScope publication(*this,first,images.size());cache->append_batch(std::move(images),std::uint64_t(first)*4);
                    }
                    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                        static std::atomic<bool> printed{false};
                        if(!printed.exchange(true))std::fprintf(stderr,"native_qsa_cpu_archive_batch actual_match=1 blocks=%lld ssd=%d prepublished_range=1\n",
                            static_cast<long long>(complete),int(ssd_cache));
                    }
                    images.clear();
                    if(i+1<complete)images.reserve(std::min<std::size_t>(std::size_t(complete-i-1),batch_blocks));
                }
            }
            if(archive_lease)archive_lease->lock.unlock();
        };
        tokens+=t;drain();
        check(tq::tq_commit_tail(k.data_ptr<float>(),v.data_ptr<float>(),r.data_ptr<float>(),tail_storage_k.data_ptr<float>(),
            tail_storage_v.data_ptr<float>(),tail_storage_index.data_ptr<float>(),staged,int(remainder),kv_heads,width,index_width,mfq_current_cuda_stream()));
        tail_k=remainder?tail_storage_k.narrow(1,0,remainder):tb::Tensor{};
        tail_v=remainder?tail_storage_v.narrow(1,0,remainder):tb::Tensor{};
        tail_index=remainder?tail_storage_index.narrow(1,0,remainder):tb::Tensor{};
        if(remainder) {
            check(tq::tq_prepare_modes(tail_k.data_ptr(),tail_v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),
                pending_k.data_ptr<std::uint16_t>(),pending_v.data_ptr<std::uint16_t>(),int(remainder),kv_heads,width,padded,mfq_current_cuda_stream(),
                rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>()));
            publish_pending(pending_k,pending_v,tokens/4);
        }
        if(history_only){commit_archive();drain();return {};}
        // Bound indexer scores independently of the projection/FFN chunk.
        // A long target context must not multiply every layer's score arena
        // by the complete 8K compute tile. Query tiles preserve causal positions.
        const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
        auto score_pools=std::max(1,prefill_attention.pools);
        while(score_pools<tokens/4)score_pools=std::min(capacity,score_pools>capacity/2?capacity:2*score_pools);
        const auto row_bytes=std::size_t(score_pools)*sizeof(float);
        const auto query_limit=std::max<int64_t>(16,int64_t(score_budget/row_bytes));
        // Until the first forward is complete, every canonical KV image is
        // already on the GPU. Do not publish RAM replacements between score
        // tiles: their query offsets are not evidence of pre-existing history.
        const bool fresh_prefill=ram && t>8 && start==0;
        std::optional<QsaRamPrefillStage> layer_stage;
        bool layer_stage_attempted=false;
        tb::Tensor decoded_operands,requested_blocks;
        bool decode_staging_attempted=false;
        const auto prefill_queries=t;
        const auto attend=[&](const tb::Tensor& query,const tb::Tensor& iq,
                const tb::Tensor& gate,tb::Tensor output,int64_t start) {
            const auto t=query.size(2);
            auto q=supported(query)?query:query.to(tb::kFloat32);if(q.stride(3)!=1)q=q.contiguous();
            auto absolute=tb::arange(start,start+t,options.dtype(tb::kInt64));
            tb::Tensor blocks;
            const int pools=int(tokens/4),columns=budget+3,count=(columns+3)/4;
            const int parts=tq::tq_attention_partitions(int(t),heads,kv_heads,padded,count,q.scalar_type()==tb::kFloat16,fp16_math,bool(ram && t<=8));
            auto& scratch=prepare_workspace(int(t),pools,parts);
            if(pools) {
                auto ranked=scratch.ranked.narrow(0,0,t*pools).reshape({1,t,pools});auto index_query=iq.to(tb::kFloat32).contiguous();
                indexer_scores(index_query,absolute,ranked,int(t),pools,scratch,!append_small);
                blocks=mfq_qwen4_exp::live_topk_selected_blocks(ranked,absolute,4,std::min(budget/4,pools),budget);
            } else {
                auto ids=tb::arange(columns,absolute.options()).reshape({1,1,columns}).expand({1,t,columns});
                auto selected=tb::where(ids<=absolute.reshape({1,t,1}),ids,tb::full_like(ids,-1)).to(tb::kInt32).contiguous();
                blocks=scratch.blocks.narrow(0,0,t*count*int64_t(sizeof(tq::SelectedBlock)));
                check(tq::tq_group_selected(selected.data_ptr<int32_t>(),blocks.data_ptr<tq::SelectedBlock>(),int(t),columns,mfq_current_cuda_stream()));
            }
            if(!output.defined())output=tb::empty({t,heads,width},options.dtype(tb::kFloat32));
            if(!supported(output) || !output.is_contiguous() || output.numel()!=t*heads*width)
                throw std::invalid_argument("adaptive QSA output must be contiguous FP16 or FP32");
            auto g=gate.defined() && !supported(gate)?gate.to(tb::kFloat32):gate;
            if(g.defined() && (g.dim()!=4 || g.size(0)!=1 || g.size(1)!=t || g.size(2)!=heads || g.size(3)!=width))
                throw std::invalid_argument("adaptive QSA gate geometry mismatch");
            if(g.defined() && g.stride(3)!=1)g=g.contiguous();
            if(ram && t>8 && !fresh_prefill) {
                // Preserve original mixed precision and copy one layer once for
                // all query tiles. If available workspace cannot hold it,
                // stream each selected union through the bounded RAM arena.
                commit_archive();drain();
                if(!layer_stage_attempted) {
                    layer_stage_attempted=true;
                    layer_stage=ram->stage_prefill(std::size_t(tokens/4));
                }
                if(!layer_stage) {
                    const auto grouped=blocks.reshape({t,-1});
                    for(int64_t begin=0;begin<t;begin+=8) {
                        const int n=int(std::min<int64_t>(8,t-begin));
                        auto selected=grouped.narrow(0,begin,n);
                        auto operands=ram->stage(selected,n,count,views,{},tokens-remainder,int(remainder));
                        const int split=tq::tq_attention_partitions(n,heads,kv_heads,padded,count,
                            q.scalar_type()==tb::kFloat16,fp16_math,true);
                        auto& tile=prepare_workspace(n,pools,split);
                        auto tile_query=q.narrow(2,begin,n),tile_positions=absolute.narrow(0,begin,n);
                        auto tile_output=output.reshape({t,heads,width}).narrow(0,begin,n);
                        auto tile_gate=g.defined()?g.narrow(1,begin,n):tb::Tensor{};
                        check(tq::tq_sparse_attention_staged(nullptr,nullptr,tile_query.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),
                            relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),selected.data_ptr<tq::SelectedBlock>(),
                            tile_positions.data_ptr<int64_t>(),tile_output.data_ptr(),tile.rotated.data_ptr<float>(),tile.partial.data_ptr<float>(),
                            tile.stats.data_ptr<float>(),tile.logmass.data_ptr<float>(),tile.norm.data_ptr<float>(),hits.data_ptr<float>(),mass.data_ptr<float>(),
                            n,heads,kv_heads,width,padded,count,split,tile_query.stride(2),tile_query.stride(1),mfq_current_cuda_stream(),true,
                            tile_gate.defined()?tile_gate.data_ptr():nullptr,tile_gate.defined() && tile_gate.scalar_type()==tb::kFloat16,
                            tile_output.scalar_type()==tb::kFloat16,tile_gate.defined()?tile_gate.stride(1):0,tile_gate.defined()?tile_gate.stride(2):0,
                            tile_query.scalar_type()==tb::kFloat16,fp16_math));
                    }
                    return output;
                }
            }
            // Decode the layer once and reuse it across every score tile.
            // Long histories must not lose tensor-core attention solely
            // because they exceed a fixed small temporary allocation.
            const auto block_count=(tokens+3)/4;
            const auto staged_elements=2*block_count*kv_heads*4LL*padded;
            const auto* staging_option=std::getenv("TYLOINFER_QSA_PREFILL_STAGING");
            const bool use_staging=prefill_queries>8 && heads/kv_heads>=4 &&
                (fp16_math || q.scalar_type()==tb::kFloat16) &&
                (!staging_option || staging_option[0]!='0') && block_count<=INT_MAX/4 &&
                block_count<prefill_queries*count/2 && tq::tq_prefill_staging_available(heads,kv_heads,padded);
            if(use_staging && !decode_staging_attempted) {
                decode_staging_attempted=true;
                const auto context=default_context(mfq_current_cuda_device());
                const auto available=context->available_memory()+context->workspace_available();
                const auto limit=option("TYLOINFER_QSA_PREFILL_STAGING_BYTES",SIZE_MAX);
                const auto bytes=std::size_t(staged_elements)*sizeof(std::uint16_t);
                const auto metadata=std::size_t(block_count)*(sizeof(tq::SelectedBlock)+5*sizeof(std::uint32_t));
                if(bytes<=limit && bytes<=available && metadata<=available-bytes) {
                    decoded_operands=tb::empty({staged_elements},options.dtype(tb::kFloat16));
                    requested_blocks=tb::empty({block_count},options.dtype(tb::kInt32));
                    auto ids=tb::arange(block_count*4,options.dtype(tb::kInt32));
                    auto selected=tb::empty({int64_t(block_count*sizeof(tq::SelectedBlock))},options.dtype(tb::kUInt8));
                    check(tq::tq_group_selected(ids.data_ptr<int32_t>(),selected.data_ptr<tq::SelectedBlock>(),
                        1,int(block_count*4),mfq_current_cuda_stream()));
                    const auto& source_views=layer_stage?layer_stage->views:views;
                    check(tq::tq_stage_attention_blocks(source_views.data_ptr<tq::TqBlockView>(),relative.data_ptr<std::uint16_t>(),
                        decoded_operands.data_ptr<std::uint16_t>(),decoded_operands.data_ptr<std::uint16_t>()+staged_elements/2,
                        int(block_count),kv_heads,padded,mfq_current_cuda_stream(),selected.data_ptr<tq::SelectedBlock>(),
                        1,int(block_count),requested_blocks.data_ptr<std::uint32_t>()));
                    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1')
                        std::fprintf(stderr,"native_qsa_prefill_staging actual_match=1 queries=%lld blocks=%lld bytes=%zu shared_score_tiles=1\n",
                            (long long)prefill_queries,(long long)block_count,bytes);
                }
            }
            bool fa_used=false;
            const auto* fa_setting=std::getenv("TYLOINFER_QSA_PREFILL_FA");
            if((!fa_setting || fa_setting[0]!='0') && decoded_operands.defined() && padded==256) {
                tq::TqPrefillAttention a;
                a.keys=static_cast<const std::uint16_t*>(decoded_operands.data_ptr());a.values=a.keys+staged_elements/2;
                a.query=q.data_ptr();a.key_signs=ks.data_ptr<float>();a.value_signs=vs.data_ptr<float>();
                a.selected=blocks.data_ptr<tq::SelectedBlock>();a.positions=absolute.data_ptr<int64_t>();a.output=output.data_ptr();
                a.rotated=scratch.rotated.data_ptr<float>();a.temporary=scratch.partial.data_ptr<float>();a.temporary_bytes=scratch.partial.nbytes();
                a.logmass=scratch.logmass.data_ptr<float>();a.norm=scratch.norm.data_ptr<float>();a.hits=hits.data_ptr<float>();a.mass=mass.data_ptr<float>();
                a.queries=int(t);a.heads=heads;a.kv_heads=kv_heads;a.width=width;a.kv_rows=int(block_count*4);a.count=count;
                a.query_token_stride=q.stride(2);a.query_head_stride=q.stride(1);a.query_half=q.scalar_type()==tb::kFloat16;
                a.gate=g.defined()?g.data_ptr():nullptr;a.gate_half=g.defined() && g.scalar_type()==tb::kFloat16;a.output_half=output.scalar_type()==tb::kFloat16;
                a.gate_token_stride=g.defined()?g.stride(1):0;a.gate_head_stride=g.defined()?g.stride(2):0;a.stream=mfq_current_cuda_stream();
                if(!prefill_graph)prefill_graph=std::make_unique<tq::TqPrefillGraph>();
                const auto status=prefill_graph->run(a);
                if(status!=int(cudaErrorNotSupported)){check(status);fa_used=true;}
            }
            const bool streamed_ram=ram && t<=8 && !layer_stage;
            const auto operands=streamed_ram?ram->stage(blocks,int(t),count,views,{},
                archive_committed?tokens-remainder:start,archive_committed?int(remainder):int(t)):
                layer_stage?layer_stage->views:views;
            if(!fa_used)check(tq::tq_sparse_attention_staged(decoded_operands.defined()?static_cast<const std::uint16_t*>(decoded_operands.data_ptr()):nullptr,
                decoded_operands.defined()?static_cast<const std::uint16_t*>(decoded_operands.data_ptr())+staged_elements/2:nullptr,q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),operands.data_ptr<tq::TqBlockView>(),
                blocks.data_ptr<tq::SelectedBlock>(),absolute.data_ptr<int64_t>(),output.data_ptr(),scratch.rotated.data_ptr<float>(),scratch.partial.data_ptr<float>(),scratch.stats.data_ptr<float>(),
                scratch.logmass.data_ptr<float>(),scratch.norm.data_ptr<float>(),hits.data_ptr<float>(),mass.data_ptr<float>(),int(t),heads,kv_heads,width,padded,count,parts,
                q.stride(2),q.stride(1),mfq_current_cuda_stream(),streamed_ram,g.defined()?g.data_ptr():nullptr,g.defined() && g.scalar_type()==tb::kFloat16,
                output.scalar_type()==tb::kFloat16,g.defined()?g.stride(1):0,g.defined()?g.stride(2):0,q.scalar_type()==tb::kFloat16,fp16_math));
            if(!fresh_prefill){commit_archive();drain();}
            return output;
        };
        if(t>8 && t>query_limit) {
            if(!output.defined())output=tb::empty({t,heads,width},options.dtype(tb::kFloat32));
            if(!supported(output) || !output.is_contiguous() || output.numel()!=t*heads*width)
                throw std::invalid_argument("adaptive QSA output must be contiguous FP16 or FP32");
            const auto tiles=(t+query_limit-1)/query_limit;
            int64_t begin=0;
            for(int64_t tile=0;tile<tiles;++tile) {
                const auto n=(t-begin+tiles-tile-1)/(tiles-tile);
                attend(query.narrow(2,begin,n),iq.narrow(1,begin,n),
                    gate.defined()?gate.narrow(1,begin,n):tb::Tensor{},
                    output.reshape({t,heads,width}).narrow(0,begin,n),start+begin);
                begin+=n;
            }
            if(std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
                "native_qsa_prefill_score_tiles actual_match=1 queries=%lld limit=%lld score_budget=%zu pools=%d\n",
                (long long)t,(long long)query_limit,score_budget,capacity);
        }else output=attend(query,iq,gate,std::move(output),start);
        commit_archive();drain();
        update_history(start);
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1' && (start==0 || tokens/interval!=start/interval)) {
            const auto s=ram?ram->stats():cache->stats();std::fprintf(stderr,"qsa_adaptive_tq tokens=%lld resident=%zu demotions=%llu promotions=%llu ssd_read=%llu state_bits=14 actual_match=1 workspace_reuse=1 direct_canonical=1 fused_gate=%d gqa_group=%d\n",
                static_cast<long long>(tokens),s.resident_bytes,static_cast<unsigned long long>(s.demotions),static_cast<unsigned long long>(s.promotions),static_cast<unsigned long long>(s.canonical_read_bytes),int(gate.defined()),heads/kv_heads>4?8:4);
        }
        return output;
    }
    void truncate(int64_t keep) {
        cancel_batch_warmup();
        if(keep<0 || keep>tokens)throw std::invalid_argument("invalid adaptive QSA truncation");
        if(keep==tokens)return;
        finish_history();
        if(ram)ram->wait();else cache->wait();drain();
        // Empty reset only retires owners and enqueues zeroes on this stream.
        // Reconstruction retains the existing completion barrier.
        if(keep)check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        const auto complete=keep/4,remainder=keep%4;
        if(remainder && keep/4==tokens/4 && tail_k.defined()) {
            tail_k=tail_k.narrow(1,0,remainder).contiguous();tail_v=tail_v.narrow(1,0,remainder).contiguous();tail_index=tail_index.narrow(1,0,remainder).contiguous();
        }else if(remainder) {
            const auto image=ram?ram->reconstruction_image(complete):cache->reconstruction_image(complete);
            std::vector<std::uint8_t> data=image.key;data.insert(data.end(),image.value.begin(),image.value.end());
            auto bytes=upload(data.data(),data.size(),options);
            tq::TqBlockView record{bytes.data_ptr<std::uint8_t>(),bytes.data_ptr<std::uint8_t>()+image.key.size(),
                tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)};
            auto view=upload(&record,sizeof(record),options);
            auto km=tb::empty({rows,256},options.dtype(tb::kFloat32)),vm=tb::empty_like(km);
            auto k=tb::empty({1,4,kv_heads,width},options.dtype(tb::kFloat32)),v=tb::empty_like(k);
            check(tq::tq_decode_modes(view.data_ptr<tq::TqBlockView>(),km.data_ptr<float>(),vm.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),1,kv_heads,padded,mfq_current_cuda_stream()));
            check(tq::tq_restore_modes(km.data_ptr<float>(),vm.data_ptr<float>(),ks.data_ptr<float>(),vs.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),1,kv_heads,width,padded,mfq_current_cuda_stream()));
            tail_k=k.narrow(1,0,remainder).contiguous();tail_v=v.narrow(1,0,remainder).contiguous();
            auto index=index_archive.at(complete)->read();
            auto host=tb::empty({1,4,index_width},options.device(tb::kCPU).dtype(tb::kFloat16));
            if(index->size()!=index_archive_bytes())throw std::runtime_error("QSA raw index archive length changed");
            std::memcpy(host.data_ptr(),index->data(),8*index_width);tail_index=host.to(options.dtype(tb::kFloat32)).narrow(1,0,remainder).contiguous();
        }else {tail_k={};tail_v={};tail_index={};}
        if(ram)ram->truncate(complete);else cache->truncate(complete);
        index_archive.resize(complete);
        {std::lock_guard<std::mutex> lock(queue_mutex);queued.clear();resident_append_begin=resident_append_end=0;}
        for(std::size_t i=std::size_t(complete);i<owners.size();++i)owners[i].reset();
        tokens=keep;hits.zero_();mass.zero_();
        if(!keep){canonical_slab={};graph_image.reset();}
        if(remainder) {
            check(tq::tq_prepare_modes(tail_k.data_ptr(),tail_v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),pending_modes_k.data_ptr<float>(),pending_modes_v.data_ptr<float>(),
                pending_k.data_ptr<std::uint16_t>(),pending_v.data_ptr<std::uint16_t>(),int(remainder),kv_heads,width,padded,mfq_current_cuda_stream(),
                rotated_tail_k.data_ptr<float>(),rotated_tail_v.data_ptr<float>()));publish_pending(pending_k,pending_v,complete);
        }
    }
};
QsaTqCache::QsaTqCache(int64_t max,int heads,int kv_heads,int width,int index_heads,int index_width,int budget,const tb::TensorOptions& options,bool ssd)
    :impl_(std::make_unique<Impl>(max,heads,kv_heads,width,index_heads,index_width,budget,options,ssd)){}
QsaTqCache::~QsaTqCache()=default;
int64_t QsaTqCache::position()const{return impl_->tokens;}
void QsaTqCache::reset(){impl_->truncate(0);}
void QsaTqCache::truncate(int64_t keep){impl_->truncate(keep);}
void QsaTqCache::wait(){impl_->finish_history();if(impl_->ram)impl_->ram->wait();else impl_->cache->wait();impl_->drain();}
void QsaTqCache::collect(){impl_->drain();}
void QsaTqCache::release_idle(){
    if(impl_->tokens)return;
    if(impl_->ram)impl_->ram->release_idle();
    impl_->drain();
    // These are CPU ownership/index records. Retained CUDA graphs reference
    // their unchanged device tensors and independent immutable image leases.
    const auto bytes=impl_->index_archive.capacity()*sizeof(std::shared_ptr<QsaIndexArchive>)+
        impl_->owners.capacity()*sizeof(std::shared_ptr<Impl::ImageBatch>);
    std::vector<std::shared_ptr<QsaIndexArchive>>().swap(impl_->index_archive);
    std::vector<std::shared_ptr<Impl::ImageBatch>>().swap(impl_->owners);
    if(bytes && std::getenv("TYLOINFER_TRACE_QSA_TQ"))std::fprintf(stderr,
        "native_qsa_tq_idle_metadata actual_match=1 released_bytes=%zu\n",bytes);
}
void QsaTqCache::set_budget(std::size_t bytes){impl_->byte_budget=bytes;}
int64_t QsaTqCache::prefill_rows(int64_t requested)const {
    if(requested<=8)return requested;
    const auto& a=*impl_;
    const auto context=mfq::cuda::default_context(a.ks.get_device());
    const auto free=context->available_memory(),reusable=context->reusable_workspace_bytes();
    const auto available=free>SIZE_MAX-reusable?SIZE_MAX:free+reusable;
    const auto loan=context->workspace_available();
    const auto bytes=available>SIZE_MAX-loan?SIZE_MAX:available+loan;
    const auto count=(a.budget+6)/4;
    auto rows=requested;
    // Scores and attention scratch are already query-tiled internally.
    // Charge that bounded tile once, then price only append work per input
    // row; charging every row for the complete context splits weight GEMMs
    // and repeatedly reconstructs the same historical KV layer.
    const auto score_budget=option("TYLOINFER_QSA_PREFILL_SCORE_BYTES",64ULL<<20);
    const auto allowance=bytes/4;
    const auto append_row_bytes=std::max<int64_t>(1,int64_t(a.kv_heads)*a.padded*16+int64_t(a.index_width)*4);
    for(int attempt=0;attempt<3;++attempt) {
        auto score_pools=std::max(1,a.prefill_attention.pools);
        const auto active=(a.tokens+rows+3)/4;
        while(score_pools<active)score_pools=std::min(a.capacity,score_pools>a.capacity/2?a.capacity:2*score_pools);
        const auto queries=std::min(rows,std::max<int64_t>(16,int64_t(score_budget/(4ULL*score_pools))));
        const auto parts=tq::tq_attention_partitions(int(queries),a.heads,a.kv_heads,a.padded,count,false,true);
        const auto partial=std::max(int64_t(a.heads)*parts*a.padded*4,int64_t(a.heads)*a.padded*4+7*score_pools+8);
        const auto query_bytes=partial+int64_t(a.heads)*a.padded*4+int64_t(a.heads)*parts*8+
            int64_t(a.heads)*count*4+8LL*score_pools+int64_t(count)*sizeof(tq::SelectedBlock);
        const auto fixed=std::size_t(queries)*query_bytes+
            tq::indexer_affine16_score_scratch(int(queries),a.index_heads,a.index_width);
        const auto next=fixed<allowance?std::min(requested,std::max<int64_t>(1,int64_t((allowance-fixed)/append_row_bytes))):
            std::min(rows,std::max<int64_t>(1,int64_t(allowance/std::max<int64_t>(1,query_bytes+append_row_bytes))));
        if(next>=rows)break;
        rows=next>=32?next/32*32:next;
    }
    return rows;
}
void QsaTqCache::release_prefill_workspace() {
    impl_->prefill_attention=Impl::AttentionWorkspace{};
    if(!impl_->append_small) {
        impl_->append_k={};impl_->append_v={};impl_->append_index={};
        impl_->append_capacity=0;impl_->append_small=true;
    }
}

int QsaTqCache::device_graph_kind(int64_t tokens)const{
    if(tokens<1 || tokens>8 || tokens>impl_->maximum-impl_->tokens)return 0;
    const int kind=tokens==1?(impl_->tokens%4==3?2:1):3;
    return kind+(impl_->ram && !impl_->ram->graph_safe()?4:0);
}
bool QsaTqCache::pending_graph_enabled()const{return impl_->pending_graph;}
bool QsaTqCache::graph_task_supported()const {
    // All stored encodings have immutable readers. Capability does not depend
    // on finishing history, fitting KV or reclaiming its current RAM budget.
    // Actual graph preparation settles and publishes pending versions.
    return impl_->ram && impl_->ram->graph_task_supported();
}
void QsaTqCache::prepare_decode(bool allow_pending){prepare_decode_tokens(1,allow_pending);}
void QsaTqCache::prepare_decode_tokens(int64_t queries,bool allow_pending,int host_mode){
    impl_->cancel_batch_warmup();
    impl_->finish_history();
    const bool prefill=impl_->prefill_attention.queries>0 || !impl_->append_small;
    impl_->prefill_attention=Impl::AttentionWorkspace{};
    if(prefill){impl_->append_k={};impl_->append_v={};impl_->append_index={};impl_->append_capacity=0;impl_->append_small=true;}
    impl_->drain();impl_->pending_graph=allow_pending && device_graph_kind(queries);
    impl_->graph_ssd_task=impl_->pending_graph && impl_->ram && host_mode==8;
    impl_->graph_host_boundary=!impl_->graph_ssd_task && impl_->pending_graph && impl_->ram && (host_mode || !impl_->ram->graph_safe());
    impl_->graph_queries=int(queries);impl_->graph_phase=int(impl_->tokens%4);
    impl_->graph_blocks=impl_->pending_graph?int((impl_->graph_phase+queries)/4):0;
    if(impl_->ram) {
        if(impl_->graph_ssd_task)impl_->ram->prepare_graph_task(std::uint64_t(impl_->tokens));
        else impl_->ram->prepare();
    }
    impl_->graph_complete=impl_->graph_blocks!=0;
    if(impl_->pending_graph) {
        ResidentAllocationScope resident;
        // This arena belongs to one fixed cache geometry and CUDA device.
        // Keep its plan across token replays, avoiding a device query per layer.
        auto& scratch=impl_->graph_workspace(int(queries));
        const int parts=scratch.parts>0?scratch.parts:
            tq::tq_attention_partitions(int(queries),impl_->heads,impl_->kv_heads,impl_->padded,(impl_->budget+6)/4,false,true);
        impl_->prepare_workspace(int(queries),impl_->capacity,parts,true);
        if(queries>1) {
            auto& batch=impl_->batch_append.at(std::size_t(queries-2));
            if(!batch.key.defined()) {
                const int staged=int((queries+6)/4*4),blocks=staged/4;
                const auto f=impl_->options.dtype(tb::kFloat32),h=impl_->options.dtype(tb::kFloat16);
                batch.key=tb::empty({1,staged,impl_->kv_heads,impl_->width},f);batch.value=tb::empty_like(batch.key);
                batch.index=tb::empty({1,staged,impl_->index_width},f);
                batch.modes_key=tb::empty({blocks,impl_->rows,256},f);batch.modes_value=tb::empty_like(batch.modes_key);
                batch.fp16_key=tb::empty({blocks,impl_->rows,256},h);batch.fp16_value=tb::empty_like(batch.fp16_key);
                batch.rotated_key=tb::empty({staged,impl_->kv_heads,impl_->padded},f);batch.rotated_value=tb::empty_like(batch.rotated_key);
                if(impl_->ram_bits!=16)batch.packed=tb::empty({int64_t(2*blocks*impl_->canonical_bytes)},impl_->options.dtype(tb::kUInt8));
            }
            if(!impl_->seed_k.defined()) {
                impl_->seed_k=tb::empty_like(impl_->tail_storage_k);impl_->seed_v=tb::empty_like(impl_->tail_storage_v);
                impl_->seed_index=tb::empty_like(impl_->tail_storage_index);
                impl_->seed_rotated_k=tb::empty_like(impl_->rotated_tail_k);impl_->seed_rotated_v=tb::empty_like(impl_->rotated_tail_v);
            }
        }
        if(impl_->graph_complete || queries>1) {
            if(!impl_->graph_destination.defined()) {
                impl_->graph_destination=tb::empty({int64_t(sizeof(tq::TqBlockView)*(impl_->interval/4))},impl_->options.dtype(tb::kUInt8));
                // M<=8 can complete two four-token blocks. Keep addresses
                // stable when single-row and verification graphs alternate.
                const int64_t bytes=2*(2*impl_->canonical_bytes+impl_->index_archive_bytes());
                impl_->graph_archive=tb::empty({bytes},impl_->options.dtype(tb::kUInt8));
                impl_->graph_archive_host=tb::empty({bytes},impl_->options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
                // Coalesced encoder stores avoid a tiny copy-engine transfer
                // between this layer and its attention kernels. Completion of
                // the model window makes the mapped archive readable by CPU.
                void* mapped=nullptr;
                if(cudaHostGetDevicePointer(&mapped,impl_->graph_archive_host.data_ptr(),0)==cudaSuccess)
                    impl_->graph_archive_mapped=static_cast<std::uint8_t*>(mapped);
                else (void)cudaGetLastError();
                if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1')
                    std::fprintf(stderr,"native_qsa_tq_archive mapped=%d bytes=%lld\n",int(mapped!=nullptr),static_cast<long long>(bytes));
            }
            const int slots=int(impl_->interval/4),slot=int(impl_->tokens/4)%slots;
            // Prepare a policy interval's destinations on-device, removing
            // the small pageable H2D descriptor copy from each fourth step.
            if(!impl_->canonical_slab.defined() || !slot || slot+impl_->graph_blocks>slots) {
                impl_->canonical_slab=tb::empty({int64_t(2*impl_->canonical_bytes*slots)},impl_->options.dtype(tb::kUInt8));
                const auto initialize=impl_->ram_bits==16?tq::tq_initialize_fp16_destinations:tq::tq_initialize_destinations;
                check(initialize(impl_->canonical_slab.data_ptr<std::uint8_t>(),impl_->graph_destination.data_ptr<tq::TqBlockView>(),
                    int(impl_->canonical_bytes),slots,mfq_current_cuda_stream()));
            }
            impl_->graph_image=std::make_shared<Impl::ImageBatch>();
            impl_->graph_image->payload=queries>1?impl_->canonical_slab:
                impl_->canonical_slab.narrow(0,int64_t(2*impl_->canonical_bytes*slot),int64_t(2*impl_->canonical_bytes));
        }
        const auto prior=impl_->tokens%4;
        if(prior) {
            if(impl_->tail_k.data_ptr()!=impl_->tail_storage_k.data_ptr())impl_->tail_storage_k.narrow(1,0,prior).copy_(impl_->tail_k);
            if(impl_->tail_v.data_ptr()!=impl_->tail_storage_v.data_ptr())impl_->tail_storage_v.narrow(1,0,prior).copy_(impl_->tail_v);
            if(impl_->tail_index.data_ptr()!=impl_->tail_storage_index.data_ptr())impl_->tail_storage_index.narrow(1,0,prior).copy_(impl_->tail_index);
        }
        if(queries>1) {
            check(tq::tq_copy_pending_tail(impl_->committed_tail(),impl_->seeded_tail(),impl_->kv_heads,impl_->width,impl_->padded,impl_->index_width,mfq_current_cuda_stream()));
            impl_->batch_prepared=true;
        }
    }
}
void QsaTqCache::advance_graph(int64_t tokens){
    if(!impl_->pending_graph)return;
    if(tokens!=impl_->graph_queries || (tokens==1?impl_->graph_complete!=(impl_->tokens%4==3):impl_->graph_phase!=impl_->tokens%4))
        throw std::logic_error("QSA graph replay changed block phase");
    const auto start=impl_->tokens;
    if(impl_->graph_complete){if(tokens==1)impl_->append_graph_image();else impl_->append_graph_batch();}
    impl_->batch_prepared=false;
    impl_->tokens+=tokens;const auto remainder=impl_->tokens%4;
    impl_->tail_k=remainder?impl_->tail_storage_k.narrow(1,0,remainder):Tensor{};
    impl_->tail_v=remainder?impl_->tail_storage_v.narrow(1,0,remainder):Tensor{};
    impl_->tail_index=remainder?impl_->tail_storage_index.narrow(1,0,remainder):Tensor{};
    impl_->update_history(start);
}
QsaTqCache::Tensor QsaTqCache::forward_pending_graph(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& iq,const Tensor& raw,
    const Tensor& positions,const Tensor& gate,Tensor output,const Pool& pool,const Tensor& absolute){
    return impl_->forward_pending_graph(q,k,v,iq,raw,positions,gate,std::move(output),pool,absolute);}
QsaTqCache::Tensor QsaTqCache::forward(const Tensor& q,const Tensor& k,const Tensor& v,const Tensor& iq,const Tensor& raw,
    const Tensor& positions,const Pool& pool,bool history_only,const Tensor& gate,Tensor output,bool fp16_math){return impl_->forward(q,k,v,iq,raw,positions,pool,history_only,gate,std::move(output),fp16_math);}
std::shared_ptr<void> QsaTqCache::hold_host_workspace(std::size_t bytes) {
    return impl_->ram?impl_->ram->hold_workspace(bytes):std::shared_ptr<void>{};
}
QsaRamWorkspaceProgress QsaTqCache::reclaim_shared_host_workspace() {
    return impl_->ram?impl_->ram->reclaim_shared_workspace():QsaRamWorkspaceProgress{};
}
std::optional<QsaRamMemory> QsaTqCache::ram_memory()const {
    if(!impl_->ram)return {};auto result=impl_->ram->memory();
    if(impl_->raw_index_store)result.index_disk=impl_->raw_index_store->stats().disk_bytes;
    return result;
}
std::vector<QsaTqCache::Tensor> QsaTqCache::resident_tensors()const {
    const auto& s=*impl_;
    std::vector<Tensor> result{s.ks,s.vs,s.relative,s.views,s.index_codes,s.index_scales,s.index_zero,s.hits,s.mass,
        s.tail_storage_k,s.tail_storage_v,s.tail_storage_index,s.rotated_tail_k,s.rotated_tail_v,s.canonical_slab};
    if(s.ram) {
        auto ram=s.ram->resident_tensors();result.insert(result.end(),ram.begin(),ram.end());
    }else for(const auto& owner:s.owners)if(owner)result.push_back(owner->payload);
    return result;
}
tq::AdaptiveCacheStats QsaTqCache::stats()const{impl_->finish_history(false);return impl_->ram?impl_->ram->stats():impl_->cache->stats();}
std::uint64_t QsaTqCache::ssd_read_bytes()const {
    return (impl_->ram?impl_->ram->stats():impl_->cache->stats()).canonical_read_bytes;
}
bool QsaTqCache::supports_session()const{return bool(impl_->ram);}
std::shared_ptr<QsaTqSession> QsaTqCache::capture_session(std::int64_t count) {
    auto& s=*impl_;
    if(!s.ram || count<=0 || count>s.tokens)throw std::invalid_argument("Invalid RAM QSA session boundary");
    wait();const auto complete=std::size_t(count/4);const auto remainder=count%4;
    auto state=std::make_shared<QsaTqSession>();state->tokens=count;state->heads=s.heads;state->kv_heads=s.kv_heads;
    state->width=s.width;state->padded=s.padded;state->index_width=s.index_width;state->ram_bits=s.ram_bits;
    state->kv=s.ram->snapshot(complete);state->index.assign(s.index_archive.begin(),s.index_archive.begin()+complete);
    const auto host=[](const Tensor& input){return input.to(tb::kCPU).contiguous();};
    if(!remainder)return state;
    if(count/4==s.tokens/4 && s.tail_k.defined()) {
        state->tail_keys=host(s.tail_k.narrow(1,0,remainder));state->tail_values=host(s.tail_v.narrow(1,0,remainder));
        state->tail_index=host(s.tail_index.narrow(1,0,remainder));return state;
    }
    const auto image=s.ram->reconstruction_image(complete);
    std::vector<std::uint8_t> data=image.key;data.insert(data.end(),image.value.begin(),image.value.end());
    auto bytes=upload(data.data(),data.size(),s.options);
    tq::TqBlockView record{bytes.data_ptr<std::uint8_t>(),bytes.data_ptr<std::uint8_t>()+image.key.size(),
        tq::tq_precision_flags(image.key_bits,image.key_state_bits),tq::tq_precision_flags(image.value_bits,image.value_state_bits)};
    auto view=upload(&record,sizeof(record),s.options);
    auto km=tb::empty({s.rows,256},s.options.dtype(tb::kFloat32)),vm=tb::empty_like(km);
    auto k=tb::empty({1,4,s.kv_heads,s.width},s.options.dtype(tb::kFloat32)),v=tb::empty_like(k);
    check(tq::tq_decode_modes(view.data_ptr<tq::TqBlockView>(),km.data_ptr<float>(),vm.data_ptr<float>(),s.relative.data_ptr<std::uint16_t>(),1,s.kv_heads,s.padded,mfq_current_cuda_stream()));
    check(tq::tq_restore_modes(km.data_ptr<float>(),vm.data_ptr<float>(),s.ks.data_ptr<float>(),s.vs.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),1,s.kv_heads,s.width,s.padded,mfq_current_cuda_stream()));
    state->tail_keys=host(k.narrow(1,0,remainder));state->tail_values=host(v.narrow(1,0,remainder));
    const auto index=s.index_archive.at(complete)->read();
    if(index->size()!=s.index_archive_bytes())throw std::runtime_error("QSA snapshot raw index length changed");
    auto raw=tb::empty({1,4,s.index_width},s.options.device(tb::kCPU).dtype(tb::kFloat16));
    std::memcpy(raw.data_ptr(),index->data(),8*s.index_width);
    state->tail_index=raw.to(tb::kFloat32).narrow(1,0,remainder).contiguous();return state;
}
std::shared_ptr<void> QsaTqCache::reserve_session_restore(const QsaTqSession& state) {
    const auto& s=*impl_;
    if(!s.ram || state.tokens<=0 || state.tokens>s.maximum || state.heads!=s.heads || state.kv_heads!=s.kv_heads ||
            state.width!=s.width || state.padded!=s.padded || state.index_width!=s.index_width || state.ram_bits!=s.ram_bits ||
            !state.kv || state.kv->blocks.size()!=std::size_t(state.tokens/4) || state.index.size()!=state.kv->blocks.size())
        throw std::invalid_argument("Incompatible RAM QSA session geometry");
    return s.ram->reserve_restore(*state.kv);
}
void QsaTqCache::restore_session(const QsaTqSession& state,bool borrow_prefix) {
    auto& s=*impl_;const auto count=state.tokens,remainder=count%4;
    if(!s.ram || count<=0 || count>s.maximum || state.heads!=s.heads || state.kv_heads!=s.kv_heads ||
            state.width!=s.width || state.padded!=s.padded || state.index_width!=s.index_width || state.ram_bits!=s.ram_bits ||
            !state.kv || state.kv->blocks.size()!=std::size_t(count/4) || state.index.size()!=state.kv->blocks.size())
        throw std::invalid_argument("Incompatible RAM QSA session geometry");
    if(remainder)for(const auto* tail:{&state.tail_keys,&state.tail_values,&state.tail_index})
        if(!tail->defined() || tail->size(0)!=1 || tail->size(1)!=remainder || tail->scalar_type()!=tb::kFloat32)
            throw std::invalid_argument("Incompatible RAM QSA session tail");
    if(remainder && (state.tail_keys.dim()!=4 || state.tail_keys.size(2)!=s.kv_heads || state.tail_keys.size(3)!=s.width ||
            state.tail_values.sizes()!=state.tail_keys.sizes() || state.tail_index.dim()!=3 || state.tail_index.size(2)!=s.index_width))
        throw std::invalid_argument("Incompatible RAM QSA session tail width");
    s.truncate(0);
    try {
        const auto stride=std::size_t(s.index_width+2)*2;
        const auto step=std::max<std::size_t>(1,std::min(state.index.size(),(2U<<20)/stride));
        auto staging=tb::empty({int64_t(step*stride)},s.options.device(tb::kCPU).dtype(tb::kUInt8).pinned_memory(true));
        std::size_t begin=0,filled=0;
        const auto flush=[&] {
            if(!filled)return;
            const auto* input=staging.data_ptr<std::uint8_t>();const auto n=filled;
            check(cudaMemcpy2DAsync(s.index_codes.data_ptr<std::int16_t>()+begin*s.index_width,s.index_width*2,
                input,stride,s.index_width*2,n,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            check(cudaMemcpy2DAsync(s.index_scales.data_ptr<std::uint16_t>()+begin,2,
                input+2*s.index_width,stride,2,n,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            check(cudaMemcpy2DAsync(s.index_zero.data_ptr<std::int16_t>()+begin,2,
                input+2*s.index_width+2,stride,2,n,cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            check(cudaStreamSynchronize(mfq_current_cuda_stream()));
            begin+=filled;filled=0;
        };
        tyloinfer::ScopedFileReadWindow reads(borrow_prefix);
        s.ram->restore(*state.kv,borrow_prefix,[&](std::size_t id) {
            // KV and its index archive are adjacent on disk. Reuse that window,
            // but do not prefetch unneeded KV when the block remains SSD-only.
            tyloinfer::ScopedFileReadWindow index_reads(false);
            if(!state.index[id])throw std::runtime_error("Missing QSA snapshot index archive");
            auto* destination=staging.data_ptr<std::uint8_t>()+filled*stride;
            if(borrow_prefix) {
                state.index[id]->read_slice_into(8*s.index_width,destination,stride,s.index_archive_bytes());
                s.index_archive.push_back(state.index[id]);
            }else {
                const auto bytes=state.index[id]->read();
                if(bytes->size()!=s.index_archive_bytes())throw std::runtime_error("Incompatible QSA snapshot index archive length");
                std::memcpy(destination,bytes->data()+8*s.index_width,stride);
                s.save_index(std::vector<std::uint8_t>(*bytes));
            }
            if(++filled==step)flush();
        });
        flush();
        s.tokens=count;s.pending_graph=false;s.graph_image.reset();s.hits.zero_();s.mass.zero_();
        if(remainder) {
            s.tail_k=state.tail_keys.to(s.options.dtype(tb::kFloat32)).contiguous();
            s.tail_v=state.tail_values.to(s.options.dtype(tb::kFloat32)).contiguous();
            s.tail_index=state.tail_index.to(s.options.dtype(tb::kFloat32)).contiguous();
            check(tq::tq_prepare_modes(s.tail_k.data_ptr(),s.tail_v.data_ptr(),false,s.ks.data_ptr<float>(),s.vs.data_ptr<float>(),
                s.pending_modes_k.data_ptr<float>(),s.pending_modes_v.data_ptr<float>(),s.pending_k.data_ptr<std::uint16_t>(),s.pending_v.data_ptr<std::uint16_t>(),
                int(remainder),s.kv_heads,s.width,s.padded,mfq_current_cuda_stream(),s.rotated_tail_k.data_ptr<float>(),s.rotated_tail_v.data_ptr<float>()));
            s.publish_pending(s.pending_k,s.pending_v,count/4);
        }
    }catch(...) {
        // A failed companion read can follow already published KV batches.
        // Keep an empty, retryable cache even though tokens was still zero.
        try{s.ram->truncate(0);}catch(...){}
        s.index_archive.clear();s.tokens=0;s.pending_graph=false;s.graph_image.reset();
        s.tail_k={};s.tail_v={};s.tail_index={};
        throw;
    }
}
} // namespace mfq::cuda
