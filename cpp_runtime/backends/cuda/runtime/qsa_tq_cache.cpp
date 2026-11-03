#include "qsa_tq_cache.h"
#include "kernels/qsa_tq_fa.h"
#include "mfq_cuda_context.h"
#include "decode_window.h"
#include "kernels/qsa_tq.h"
#include "kernels/qsa_tq_encode.h"
#include "mfq_cuda_resident_allocations.h"
#include "tyloinfer/kernels/cuda/qwen4_exp.h"
#include <atomic>
#include <cmath>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
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
struct TqEncoderStream {
    cudaStream_t value=nullptr;
    ~TqEncoderStream(){if(value)(void)cudaStreamDestroy(value);}
};
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
bool qsa_tq_cache_enabled() {const auto* s=std::getenv("TYLOINFER_QSA_ADAPTIVE_TQ");return s && std::string(s)=="1";}

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
    tb::Tensor graph_destination,graph_archive,graph_archive_host;
    std::uint8_t* graph_archive_mapped=nullptr;
    std::shared_ptr<ImageBatch> graph_image;
    tb::Tensor canonical_slab;
    struct AttentionWorkspace {
        int queries=0,pools=0,parts=0;
        tb::Tensor ranked,blocks,rotated,partial,stats,logmass,norm;
    } decode_attention,prefill_attention,pending_attention;
    std::unique_ptr<tq::TqPrefillGraph> prefill_graph;
    std::unique_ptr<tq::AdaptiveCache> cache;
    std::shared_ptr<mfq::cache::QsaKvStore> raw_index_store;
    std::vector<mfq::cache::QsaKvStore::BlockPtr> raw_index_blocks;
    std::vector<std::vector<std::uint8_t>> ram_index_blocks;
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
    TqEncoderStream encoder_owner;cudaStream_t encoder_stream=nullptr;std::once_flag encoder_once;
    TqCodecBuffer source,source_views,decoded,encoded,sizes,flags,status,targets,workspace;

    Impl(int64_t max,int h,int kh,int w,int ih,int iw,int b,const tb::TensorOptions& opts,bool ssd)
        :maximum(max),heads(h),kv_heads(kh),width(w),padded(w<=256?256:w<=512?512:1024),index_heads(ih),
         index_width(iw),budget(b),rows(0),capacity(0),options(opts),ssd_cache(ssd) {
        if(max<=0 || max>INT_MAX-3 || h<=0 || kh<=0 || h%kh || w<=0 || w>1024 || ih<=0 || iw<=0 || b<=0 || b%4)
            throw std::invalid_argument("invalid adaptive QSA geometry");
        const auto row_count=4LL*kh*(padded/256);
        // Native compact images have 32-bit lengths and integer row strides.
        if(row_count>(INT_MAX-16)/259)throw std::invalid_argument("adaptive QSA image stride exceeds 32 bits");
        rows=int(row_count);capacity=int((max+3)/4);
        check(cudaGetDevice(&device));
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
        owners.resize(capacity);canonical_bytes=((3*rows+3)&~3)+256*rows;
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
        cache.reset();(void)cudaStreamSynchronize(mfq_current_cuda_stream());
        for(auto& old:retired)if(old.done)(void)cudaEventDestroy(old.done);
        for(auto& upload:uploads)if(upload.done)(void)cudaEventDestroy(upload.done);
    }
    void initialize_cache() {
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
    std::vector<tq::MicroblockImage> reencode(const std::vector<tq::MicroblockImage>& input,const std::vector<tq::Precision>& precision) {
        check(cudaSetDevice(device));
        // Short contexts use only canonical Q8. Create an encoder stream when
        // the background worker first needs secondary quantization.
        std::call_once(encoder_once,[this]{
            check(cudaStreamCreateWithFlags(&encoder_stream,cudaStreamNonBlocking));
            encoder_owner.value=encoder_stream;
        });
        const int n=int(input.size()),count=2*n;
        const auto stride=(3*rows+7)/8+3*rows+256*rows+16;
        std::vector<std::uint8_t> bytes;std::vector<std::size_t> offsets;
        for(int kind=0;kind<2;++kind)for(const auto& image:input) {
            const auto& src=kind?image.value:image.key;offsets.push_back(bytes.size());bytes.insert(bytes.end(),src.begin(),src.end());
        }
        source.reserve(bytes.size());source_views.reserve(n*sizeof(tq::TqBlockView));
        decoded.reserve(std::size_t(count)*rows*256*sizeof(float));encoded.reserve(std::size_t(count)*stride);
        sizes.reserve(count*4);flags.reserve(count*4);status.reserve(4);targets.reserve(count*4);
        std::vector<tq::TqBlockView> records(n);std::vector<int> budgets(count);
        for(int i=0;i<n;++i) {
            records[i]={source.as<std::uint8_t>(offsets[i]),source.as<std::uint8_t>(offsets[n+i]),
                tq::tq_precision_flags(input[i].key_bits,input[i].key_state_bits),tq::tq_precision_flags(input[i].value_bits,input[i].value_state_bits)};
            budgets[i]=rows*precision[i].key_bits;budgets[n+i]=rows*precision[i].value_bits;
        }
        check(cudaMemcpyAsync(source.pointer,bytes.data(),bytes.size(),cudaMemcpyHostToDevice,encoder_stream));
        check(cudaMemcpyAsync(source_views.pointer,records.data(),records.size()*sizeof(records[0]),cudaMemcpyHostToDevice,encoder_stream));
        check(cudaMemcpyAsync(targets.pointer,budgets.data(),count*4,cudaMemcpyHostToDevice,encoder_stream));
        check(tq::tq_decode_modes(source_views.as<tq::TqBlockView>(),decoded.as<float>(),decoded.as<float>(std::size_t(n)*rows*256*4),
            relative.data_ptr<std::uint16_t>(),n,kv_heads,padded,encoder_stream));
        tq::TqEncodePlan plan;
        check(tq::tq_make_encode_plan(&plan,count,rows,tq::TqEncodeProfile::Fast,option("TYLOINFER_QSA_TQ_WORKSPACE_BYTES",32ULL<<20)));
        workspace.reserve(plan.workspace_bytes);
        check(tq::tq_encode_modes(plan,decoded.as<float>(),targets.as<int>(),0,1.1f,encoded.as<std::uint8_t>(),stride,
            sizes.as<std::uint32_t>(),flags.as<std::uint32_t>(),status.as<int>(),workspace.pointer,workspace.bytes,encoder_stream));
        std::vector<std::uint8_t> packed(std::size_t(count)*stride);std::vector<std::uint32_t> lengths(count),tags(count);int error=0;
        check(cudaMemcpyAsync(packed.data(),encoded.pointer,packed.size(),cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(lengths.data(),sizes.pointer,count*4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(tags.data(),flags.pointer,count*4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaMemcpyAsync(&error,status.pointer,4,cudaMemcpyDeviceToHost,encoder_stream));
        check(cudaStreamSynchronize(encoder_stream));check(error);
        std::vector<tq::MicroblockImage> result(n);
        for(int i=0;i<n;++i) {
            auto& dst=result[i];dst.key.assign(packed.begin()+std::size_t(i)*stride,packed.begin()+std::size_t(i)*stride+lengths[i]);
            dst.value.assign(packed.begin()+std::size_t(n+i)*stride,packed.begin()+std::size_t(n+i)*stride+lengths[n+i]);
            dst.key_bits=std::uint8_t(tags[i]);dst.value_bits=std::uint8_t(tags[n+i]);dst.key_state_bits=dst.value_state_bits=14;
        }
        return result;
    }
    void retire(std::vector<std::shared_ptr<ImageBatch>> previous) {
        if(previous.empty())return;Retired old;old.images=std::move(previous);
        check(cudaEventCreateWithFlags(&old.done,cudaEventDisableTiming));check(cudaEventRecord(old.done,mfq_current_cuda_stream()));retired.push_back(std::move(old));
    }
    void publish_images(const std::shared_ptr<ImageBatch>& group,const std::vector<std::size_t>& ids) {
        std::vector<std::shared_ptr<ImageBatch>> previous;
        bool contiguous=!ids.empty();
        for(std::size_t i=1;i<ids.size();++i)contiguous&=ids[i]==ids[0]+i;
        if(contiguous)check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+ids[0]*sizeof(tq::TqBlockView),
            group->records.data(),ids.size()*sizeof(tq::TqBlockView),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
        for(std::size_t i=0;i<ids.size();++i) {
            const auto id=ids[i];const auto& view=group->records[i];
            if(!contiguous)check(cudaMemcpyAsync(views.data_ptr<std::uint8_t>()+id*sizeof(view),&view,sizeof(view),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            if(owners[id])previous.push_back(std::move(owners[id]));owners[id]=group;
        }
        Upload submitted;submitted.image=group;check(cudaEventCreateWithFlags(&submitted.done,cudaEventDisableTiming));
        check(cudaEventRecord(submitted.done,mfq_current_cuda_stream()));uploads.push_back(std::move(submitted));
        retire(std::move(previous));
    }
    AttentionWorkspace& prepare_workspace(int queries,int pools,int parts,bool pending=false) {
        const bool small=queries<=8;
        // Captured GPU-only calls retain these addresses across eager prefill
        // and multi-row verification, which may resize the other arenas.
        auto& a=pending?pending_attention:small?decode_attention:prefill_attention;
        if(a.queries>=queries && a.pools>=pools && a.parts==parts)return a;
        const int q=std::max(queries,a.queries),count=(budget+6)/4;
        int p=small?capacity:std::max(1,a.pools);
        while(p<pools)p=std::min(capacity,p>capacity/2?capacity:2*p);
        a.queries=q;a.pools=p;a.parts=parts;
        const auto f=options.dtype(tb::kFloat32);
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
    tb::Tensor forward_pending_graph(const tb::Tensor& query,const tb::Tensor& key,const tb::Tensor& value,const tb::Tensor& iq,
        const tb::Tensor& raw,const tb::Tensor& positions,const tb::Tensor& gate,tb::Tensor output,
        const QsaTqCache::Pool& pool,const tb::Tensor& absolute_positions) {
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
            auto starts=((absolute_positions.defined()?absolute_positions:positions).reshape({1,1})-3);
            auto pooled=pool(tail_storage_index,starts).to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_append(pooled.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),
                index_zero.data_ptr<std::int16_t>(),positions.data_ptr<int64_t>(),index_width,mfq_current_cuda_stream()));
            if(!graph_archive_mapped)check(cudaMemcpyAsync(graph_archive_host.data_ptr(),graph_archive.data_ptr(),graph_archive.numel(),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        }
        const int columns=budget+3,count=(columns+3)/4,parts=a.parts;
        mark("qsa_cache_append_end");
        auto ranked=a.ranked.narrow(0,0,capacity).reshape({1,1,capacity});auto index_query=iq.to(tb::kFloat32).contiguous();
        check(tq::indexer_affine16_scores(index_query.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),
            index_zero.data_ptr<std::int16_t>(),positions.data_ptr<int64_t>(),ranked.data_ptr<float>(),1,index_heads,index_width,capacity,mfq_current_cuda_stream()));
        mark("qsa_cache_scores_end");
        auto blocks=mfq_qwen4_exp::live_topk_selected_blocks(ranked,positions,4,std::min(budget/4,capacity),budget);
        mark("qsa_cache_select_end");
        // Graph preparation runs this body repeatedly without committing a
        // token. Only the recorded replay may update its hit/attention history.
        cudaStreamCaptureStatus capture_status;
        check(cudaStreamIsCapturing(mfq_current_cuda_stream(),&capture_status));
        const bool observe=capture_status!=cudaStreamCaptureStatusNone;
        check(tq::tq_sparse_attention(q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),views.data_ptr<tq::TqBlockView>(),
            blocks.data_ptr<tq::SelectedBlock>(),positions.data_ptr<int64_t>(),output.data_ptr(),a.rotated.data_ptr<float>(),a.partial.data_ptr<float>(),a.stats.data_ptr<float>(),
            a.logmass.data_ptr<float>(),a.norm.data_ptr<float>(),observe?hits.data_ptr<float>():nullptr,observe?mass.data_ptr<float>():nullptr,1,heads,kv_heads,width,padded,count,parts,
            q.stride(2),q.stride(1),mfq_current_cuda_stream(),false,g.data_ptr(),g.scalar_type()==tb::kFloat16,output.scalar_type()==tb::kFloat16,
            g.stride(1),g.stride(2),q.scalar_type()==tb::kFloat16,true,true));
        mark("qsa_cache_attention_end");
        return output;
    }
    void append_graph_image() {
        finish_history();
        const auto id=std::size_t(tokens/4);
        // DecodeWindow has completed the entire model, including this pinned
        // archive copy. No layer boundary or additional GPU wait is required.
        owners[id]=std::move(graph_image);
        const auto* bytes=graph_archive_host.data_ptr<std::uint8_t>();
        tq::MicroblockImage image;image.key.assign(bytes,bytes+canonical_bytes);
        image.value.assign(bytes+canonical_bytes,bytes+2*canonical_bytes);
        {ResidentAppendScope publication(*this,id,1);cache->append(std::move(image),std::uint64_t(id)*4);}
        const auto* raw=bytes+2*canonical_bytes;
        auto index=std::vector<std::uint8_t>(raw,raw+4*index_width*2);
        if(ssd_cache)raw_index_blocks.push_back(raw_index_store->write(std::move(index)));
        else ram_index_blocks.push_back(std::move(index));
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
        cache->update(h.tokens,h.budget,std::move(hits),std::move(attention));
        h.pending=false;
    }
    void update_history(int64_t start) {
        if(byte_budget && tokens/interval!=start/interval) {
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
        finish_history();
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
            check(tq::tq_prepare_affine8(k.data_ptr(),v.data_ptr(),false,ks.data_ptr<float>(),vs.data_ptr<float>(),
                canonical_k,canonical_v,int(span),kv_heads,width,padded,mfq_current_cuda_stream()));
            const auto archive_bytes=2*complete*canonical_bytes+complete*4*index_width*2;
            if(!archive.defined() || archive.numel()<archive_bytes)archive=tb::empty({int64_t(archive_bytes)},options.dtype(tb::kUInt8));
            check(tq::tq_pack_archive(canonical_k,canonical_v,r.data_ptr<float>(),archive.data_ptr<std::uint8_t>(),int(complete),int(canonical_bytes),index_width,mfq_current_cuda_stream()));
            if(t>8) {
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
                    static std::atomic<bool> printed{false};
                    if(!printed.exchange(true))std::fprintf(stderr,"native_qsa_prefill_archive_overlap actual_match=1 bytes=%lld shared_pinned_arena=1 side_stream=1 contiguous_descriptors=1\n",static_cast<long long>(archive_bytes));
                }
            } else cpu_archive=archive.narrow(0,0,archive_bytes).to(tb::kCPU).contiguous();
            group->records.resize(complete);std::vector<std::size_t> ids;
            for(int64_t i=0;i<complete;++i) {
                group->records[i]={canonical_k+i*canonical_bytes,canonical_v+i*canonical_bytes,254,254};ids.push_back(std::size_t(base+i));
            }
            publish_images(group,ids);
            auto block_positions=(tb::arange(complete,positions.options().dtype(tb::kInt64))*4+base*4+
                (positions.reshape({-1}).narrow(0,0,1)-start)).reshape({1,complete});
            auto pooled=pool(r.narrow(1,0,span),block_positions).to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_encode(pooled.data_ptr<float>(),index_codes.data_ptr<std::int16_t>()+base*index_width,
                index_scales.data_ptr<std::uint16_t>()+base,index_zero.data_ptr<std::int16_t>()+base,int(complete),index_width,mfq_current_cuda_stream()));
        }
        const auto commit_archive=[&] {
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
                images.push_back(std::move(image));
                const auto* index=cpu_archive.data_ptr<std::uint8_t>()+2*complete*canonical_bytes+i*4*index_width*2;
                auto bytes=std::vector<std::uint8_t>(index,index+4*index_width*2);
                if(ssd_cache)raw_index_blocks.push_back(raw_index_store->write(std::move(bytes)));
                else ram_index_blocks.push_back(std::move(bytes));
                if(images.size()==batch_blocks || i+1==complete) {
                    const auto first=std::size_t(base+i+1)-images.size();
                    ResidentAppendScope publication(*this,first,images.size());
                    cache->append_batch(std::move(images),std::uint64_t(first)*4);
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
        auto q=supported(query)?query:query.to(tb::kFloat32);if(q.stride(3)!=1)q=q.contiguous();
        auto absolute=tb::arange(start,tokens,options.dtype(tb::kInt64));
        tb::Tensor blocks;
        const int pools=int(tokens/4),columns=budget+3,count=(columns+3)/4;
        const int parts=tq::tq_attention_partitions(int(t),heads,kv_heads,padded,count,q.scalar_type()==tb::kFloat16,fp16_math);
        auto& scratch=prepare_workspace(int(t),pools,parts);
        if(pools) {
            auto ranked=scratch.ranked.narrow(0,0,t*pools).reshape({1,t,pools});auto index_query=iq.to(tb::kFloat32).contiguous();
            check(tq::indexer_affine16_scores(index_query.data_ptr<float>(),index_codes.data_ptr<std::int16_t>(),index_scales.data_ptr<std::uint16_t>(),
                index_zero.data_ptr<std::int16_t>(),absolute.data_ptr<int64_t>(),ranked.data_ptr<float>(),int(t),index_heads,index_width,pools,mfq_current_cuda_stream()));
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
        // Reuse one decoded operand tile across many queries. The cap is the
        // same 32 MiB attention-workspace budget used by the QSA batch planner;
        // long contexts keep the existing compressed streaming path.
        tb::Tensor decoded_operands,requested_blocks;
        const auto block_count=(tokens+3)/4;
        const auto staged_elements=2*block_count*kv_heads*4LL*padded;
        const auto* staging_option=std::getenv("TYLOINFER_QSA_PREFILL_STAGING");
        const bool use_staging=t>8 && heads/kv_heads>=4 && (fp16_math || q.scalar_type()==tb::kFloat16) &&
            (!staging_option || staging_option[0]!='0') &&
            staged_elements<=int64_t((32ULL<<20)/sizeof(std::uint16_t)) && block_count<t*count/2 &&
            tq::tq_prefill_staging_available(heads,kv_heads,padded);
        if(use_staging) {
            decoded_operands=tb::empty({staged_elements},options.dtype(tb::kFloat16));
            requested_blocks=tb::empty({block_count},options.dtype(tb::kInt32));
            check(tq::tq_stage_attention_blocks(views.data_ptr<tq::TqBlockView>(),relative.data_ptr<std::uint16_t>(),
                static_cast<std::uint16_t*>(decoded_operands.data_ptr()),static_cast<std::uint16_t*>(decoded_operands.data_ptr())+staged_elements/2,
                int(block_count),kv_heads,padded,mfq_current_cuda_stream(),blocks.data_ptr<tq::SelectedBlock>(),
                int(t),count,requested_blocks.data_ptr<std::uint32_t>()));
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true))std::fprintf(stderr,"native_qsa_prefill_staging actual_match=1 queries=%lld blocks=%lld bytes=%lld\n",
                    (long long)t,(long long)block_count,(long long)(staged_elements*2));
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
        if(!fa_used)check(tq::tq_sparse_attention_staged(decoded_operands.defined()?static_cast<const std::uint16_t*>(decoded_operands.data_ptr()):nullptr,
            decoded_operands.defined()?static_cast<const std::uint16_t*>(decoded_operands.data_ptr())+staged_elements/2:nullptr,q.data_ptr(),ks.data_ptr<float>(),vs.data_ptr<float>(),relative.data_ptr<std::uint16_t>(),views.data_ptr<tq::TqBlockView>(),
            blocks.data_ptr<tq::SelectedBlock>(),absolute.data_ptr<int64_t>(),output.data_ptr(),scratch.rotated.data_ptr<float>(),scratch.partial.data_ptr<float>(),scratch.stats.data_ptr<float>(),
            scratch.logmass.data_ptr<float>(),scratch.norm.data_ptr<float>(),hits.data_ptr<float>(),mass.data_ptr<float>(),int(t),heads,kv_heads,width,padded,count,parts,
            q.stride(2),q.stride(1),mfq_current_cuda_stream(),false,g.defined()?g.data_ptr():nullptr,g.defined() && g.scalar_type()==tb::kFloat16,
            output.scalar_type()==tb::kFloat16,g.defined()?g.stride(1):0,g.defined()?g.stride(2):0,q.scalar_type()==tb::kFloat16,fp16_math));
        commit_archive();drain();
        update_history(start);
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_QSA_TQ");trace && trace[0]=='1' && (start==0 || tokens/interval!=start/interval)) {
            const auto s=cache->stats();std::fprintf(stderr,"qsa_adaptive_tq tokens=%lld resident=%zu demotions=%llu promotions=%llu ssd_read=%llu state_bits=14 actual_match=1 workspace_reuse=1 direct_canonical=1 fused_gate=%d gqa_group=%d\n",
                static_cast<long long>(tokens),s.resident_bytes,static_cast<unsigned long long>(s.demotions),static_cast<unsigned long long>(s.promotions),static_cast<unsigned long long>(s.canonical_read_bytes),int(g.defined()),heads/kv_heads>4?8:4);
        }
        return output;
    }
    void truncate(int64_t keep) {
        if(keep<0 || keep>tokens)throw std::invalid_argument("invalid adaptive QSA truncation");
        if(keep==tokens)return;
        finish_history();
        cache->wait();drain();
        // Empty reset only retires owners and enqueues zeroes on this stream.
        // Reconstruction retains the existing completion barrier.
        if(keep)check(cudaStreamSynchronize(mfq_current_cuda_stream()));
        const auto complete=keep/4,remainder=keep%4;
        if(remainder && keep/4==tokens/4 && tail_k.defined()) {
            tail_k=tail_k.narrow(1,0,remainder).contiguous();tail_v=tail_v.narrow(1,0,remainder).contiguous();tail_index=tail_index.narrow(1,0,remainder).contiguous();
        }else if(remainder) {
            const auto image=cache->reconstruction_image(complete);
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
            auto index=ssd_cache?raw_index_store->read(raw_index_blocks.at(complete)):
                std::make_shared<const std::vector<std::uint8_t>>(ram_index_blocks.at(complete));
            auto host=tb::empty({1,4,index_width},options.device(tb::kCPU).dtype(tb::kFloat16));
            std::memcpy(host.data_ptr(),index->data(),index->size());tail_index=host.to(options.dtype(tb::kFloat32)).narrow(1,0,remainder).contiguous();
        }else {tail_k={};tail_v={};tail_index={};}
        cache->truncate(complete);
        if(ssd_cache)raw_index_blocks.resize(complete);else ram_index_blocks.resize(complete);
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
void QsaTqCache::wait(){impl_->finish_history();impl_->cache->wait();impl_->drain();}
void QsaTqCache::set_budget(std::size_t bytes){impl_->byte_budget=bytes;}
int64_t QsaTqCache::prefill_rows(int64_t requested)const {
    if(requested<=8)return requested;
    const auto& a=*impl_;
    const auto context=mfq::cuda::default_context(a.ks.get_device());
    const auto free=context->available_memory(),reusable=context->reusable_workspace_bytes();
    const auto available=free>SIZE_MAX-reusable?SIZE_MAX:free+reusable;
    const auto loan=context->workspace_available();
    const auto bytes=available>SIZE_MAX-loan?SIZE_MAX:available+loan;
    const auto pools=(a.tokens+requested+3)/4;
    const auto count=(a.budget+6)/4;
    auto rows=requested;
    // Reserve three quarters for live activations, weight decoding and BLAS.
    // Include geometric score capacity, history, legacy partials and FA masks.
    for(int attempt=0;attempt<3;++attempt) {
        const auto parts=tq::tq_attention_partitions(int(rows),a.heads,a.kv_heads,a.padded,count,false,true);
        const auto partial=std::max(int64_t(a.heads)*parts*a.padded*4,int64_t(a.heads)*a.padded*4+7*pools+8);
        const auto row_bytes=partial+int64_t(a.heads)*a.padded*4+int64_t(a.heads)*parts*8+
            int64_t(a.heads)*count*4+8*pools+int64_t(count)*sizeof(tq::SelectedBlock)+
            int64_t(a.kv_heads)*a.padded*16+int64_t(a.index_width)*4;
        const auto next=std::min(requested,std::max<int64_t>(1,int64_t(bytes/4/std::max<int64_t>(1,row_bytes))));
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

int QsaTqCache::device_graph_kind(int64_t tokens)const{return tokens!=1 || impl_->tokens>=impl_->maximum?0:impl_->tokens%4==3?2:1;}
bool QsaTqCache::pending_graph_enabled()const{return impl_->pending_graph;}
void QsaTqCache::prepare_decode(bool allow_pending){
    impl_->finish_history();
    const bool prefill=impl_->prefill_attention.queries>0 || !impl_->append_small;
    impl_->prefill_attention=Impl::AttentionWorkspace{};
    if(prefill){impl_->append_k={};impl_->append_v={};impl_->append_index={};impl_->append_capacity=0;impl_->append_small=true;}
    impl_->drain();impl_->pending_graph=allow_pending && device_graph_kind(1);
    impl_->graph_complete=impl_->pending_graph && device_graph_kind(1)==2;
    if(impl_->pending_graph) {
        ResidentAllocationScope resident;
        // This arena belongs to one fixed cache geometry and CUDA device.
        // Keep its plan across token replays, avoiding a device query per layer.
        const int parts=impl_->pending_attention.parts>0?impl_->pending_attention.parts:
            tq::tq_attention_partitions(1,impl_->heads,impl_->kv_heads,impl_->padded,(impl_->budget+6)/4,false,true);
        impl_->prepare_workspace(1,impl_->capacity,parts,true);
        if(impl_->graph_complete) {
            if(!impl_->graph_destination.defined()) {
                impl_->graph_destination=tb::empty({int64_t(sizeof(tq::TqBlockView)*(impl_->interval/4))},impl_->options.dtype(tb::kUInt8));
                const int64_t bytes=2*impl_->canonical_bytes+8*impl_->index_width;
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
            if(!impl_->canonical_slab.defined() || !slot) {
                impl_->canonical_slab=tb::empty({int64_t(2*impl_->canonical_bytes*slots)},impl_->options.dtype(tb::kUInt8));
                check(tq::tq_initialize_destinations(impl_->canonical_slab.data_ptr<std::uint8_t>(),impl_->graph_destination.data_ptr<tq::TqBlockView>(),
                    int(impl_->canonical_bytes),slots,mfq_current_cuda_stream()));
            }
            impl_->graph_image=std::make_shared<Impl::ImageBatch>();
            impl_->graph_image->payload=impl_->canonical_slab.narrow(0,int64_t(2*impl_->canonical_bytes*slot),int64_t(2*impl_->canonical_bytes));
        }
        const auto prior=impl_->tokens%4;
        if(prior) {
            if(impl_->tail_k.data_ptr()!=impl_->tail_storage_k.data_ptr())impl_->tail_storage_k.narrow(1,0,prior).copy_(impl_->tail_k);
            if(impl_->tail_v.data_ptr()!=impl_->tail_storage_v.data_ptr())impl_->tail_storage_v.narrow(1,0,prior).copy_(impl_->tail_v);
            if(impl_->tail_index.data_ptr()!=impl_->tail_storage_index.data_ptr())impl_->tail_storage_index.narrow(1,0,prior).copy_(impl_->tail_index);
        }
    }
}
void QsaTqCache::advance_graph(int64_t tokens){
    if(!impl_->pending_graph)return;
    if(tokens!=1 || impl_->graph_complete!=(impl_->tokens%4==3))throw std::logic_error("QSA graph replay changed block phase");
    const auto start=impl_->tokens;
    if(impl_->graph_complete)impl_->append_graph_image();
    ++impl_->tokens;const auto remainder=impl_->tokens%4;
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
tq::AdaptiveCacheStats QsaTqCache::stats()const{impl_->finish_history(false);return impl_->cache->stats();}
} // namespace mfq::cuda
