#include "storage/session_state.h"
#include "runtime/qsa_tq_cache.h"
#include "runtime/qsa_tq_session.h"
#include "mfq_cuda_context.h"
#include "mfq_paged_prefix_cache.h"
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace tb=mfq_tensor_backend;
void setting(const char* name,const std::string& value) {
#ifdef _WIN32
    _putenv_s(name,value.c_str());
#else
    setenv(name,value.c_str(),1);
#endif
}
void require(bool b,const char* message) {if(!b)throw std::runtime_error(message);}
void failed_restore(mfq::cuda::QsaTqCache& runtime,const mfq::cuda::QsaTqSession& state,bool borrow) {
    auto broken=state;const auto id=std::min<std::size_t>(513,broken.index.size()-1);
    broken.index[id]=std::make_shared<mfq::cuda::QsaIndexArchive>();
    broken.index[id]->resident=std::make_shared<const std::vector<std::uint8_t>>(1);
    bool rejected=false;
    try{runtime.restore_session(broken,borrow);}catch(const std::runtime_error&){rejected=true;}
    require(rejected && runtime.stats().resident_bytes==0,"Failed prefix restore retained partial RAM KV");
}
void same(const tb::Tensor& a,const tb::Tensor& b) {
    require(a.sizes()==b.sizes() && !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),
        "Compressed prefix changed exact index selection or attention output");
}
void index_archive_slices(const std::filesystem::path& directory) {
    constexpr std::size_t width=128,raw=8*width,stride=2*(width+2),length=raw+stride,padding=13;
    auto payload=std::make_shared<std::vector<std::uint8_t>>(padding+length+9);
    for(std::size_t i=0;i<payload->size();++i)(*payload)[i]=std::uint8_t(i*29+7);
    std::vector<std::uint8_t> output(stride+8,0xa5);
    const auto verify=[&] {
        require(std::equal(output.begin()+4,output.end()-4,payload->begin()+padding+raw),"Index suffix bytes changed");
        for(int i=0;i<4;++i)require(output[i]==0xa5 && output[output.size()-1-i]==0xa5,"Index slice overwrote destination guards");
    };
    mfq::cuda::QsaIndexArchive archive;
    archive.resident=std::make_shared<const std::vector<std::uint8_t>>(payload->begin()+padding,payload->begin()+padding+length);
    archive.read_slice_into(raw,output.data()+4,stride,length);verify();
    auto source=std::make_shared<tyloinfer::qsa::RamKvReadSource>();source->bytes=payload->size();
    unsigned direct=0,vector=0;
    source->read_range=[&](std::size_t,std::size_t)->std::vector<std::uint8_t>{++vector;throw std::runtime_error("Direct index slice allocated a full archive");};
    source->read_range_into=[&](std::size_t at,std::uint8_t* destination,std::size_t count) {
        ++direct;require(at==padding+raw && count==stride,"Index restore read its unused FP32 companion");
        std::memcpy(destination,payload->data()+at,count);
    };
    archive.bind(source,padding,length);archive.read_slice_into(raw,output.data()+4,stride,length);verify();
    require(direct==1 && vector==0,"Index slice did not use direct retained I/O");
    for(int bad=0;bad<3;++bad) {
        bool rejected=false;try {
            if(bad==0)archive.read_slice_into(length,output.data()+4,1,length);
            if(bad==1)archive.read_slice_into(raw,nullptr,stride,length);
            if(bad==2)archive.read_slice_into(raw,output.data()+4,stride,length+1);
        }catch(const std::exception&){rejected=true;}
        require(rejected && direct==1 && vector==0,"Invalid index slice reached its reader");
    }
    archive.read_slice_into(length,nullptr,0,length);require(direct==1,"Empty index slice performed I/O");
    source->bytes=padding+length-1;bool truncated=false;
    try{archive.read_slice_into(raw,output.data()+4,stride,length);}catch(const std::out_of_range&){truncated=true;}
    require(truncated && direct==1,"Truncated retained index extent was accepted");source->bytes=payload->size();
    source->read_range_into={};source->read_range=[&](std::size_t at,std::size_t count) {
        ++vector;require(at==padding+raw && count==stride,"Vector-only index reader received the wrong slice");
        return std::vector<std::uint8_t>(payload->begin()+at,payload->begin()+at+count);
    };
    archive.read_slice_into(raw,output.data()+4,stride,length);verify();require(vector==1,"Vector-only reader was not retained");
    mfq::cache::QsaKvStoreConfig config;config.directory=directory.string();config.buffer_bytes=config.budget_bytes=4096;
    auto store=std::make_shared<mfq::cache::QsaKvStore>(config);mfq::cuda::QsaIndexArchive session;
    session.store=store;session.block=store->write({payload->begin()+padding,payload->begin()+padding+length});store->flush();
    session.read_slice_into(raw,output.data()+4,stride,length);verify();
    std::cout<<"Index archive slices: exact INT16 suffix, direct/vector/session/resident sources, full geometry and destination bounds PASS\n";
}
void run(int bits,size_t page,const std::filesystem::path& directory) {
    setting("TYLOINFER_QSA_RAM_KV_BITS",std::to_string(bits));
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(1<<20));
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    constexpr int heads=6,kh=2,width=63,iw=7,length=160,prefix=129;
    mfq::cuda::QsaTqCache runtime(length,heads,kh,width,1,iw,32,f,false);
    std::vector<float> kv(length*kh*width),raw_data(length*iw),query_data(length*heads*width),iq_data(length*iw);
    for(size_t i=0;i<kv.size();++i)kv[i]=.13f*std::sin(float(i)*.017f);
    for(size_t i=0;i<raw_data.size();++i){raw_data[i]=std::sin(float(i)*.071f);iq_data[i]=std::cos(float(i)*.029f);}
    for(size_t i=0;i<query_data.size();++i)query_data[i]=.2f*std::cos(float(i)*.011f);
    auto values=tb::tensor(kv,f).reshape({1,length,kh,width}),raw=tb::tensor(raw_data,f).reshape({1,length,iw});
    auto query=tb::tensor(query_data,f).reshape({1,length,heads,width}).permute({0,2,1,3});
    auto iq=tb::tensor(iq_data,f).reshape({1,length,1,iw});
    const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
    const auto append=[&](int begin,int count) {
        auto positions=tb::arange(begin,begin+count,f.dtype(tb::kInt64)).reshape({1,count});
        return runtime.forward(query.narrow(2,begin,count),values.narrow(1,begin,count),values.narrow(1,begin,count),
            iq.narrow(1,begin,count),raw.narrow(1,begin,count),positions,pool,false,tb::zeros({1,count,heads,width},f)).to(tb::kCPU).contiguous();
    };
    append(0,prefix);
    TextSessionState state;state.cache_pos=prefix;for(int i=0;i<prefix;++i)state.tokens.push_back(i+1);
    std::vector<Qwen4BlockSessionState> layers(3);
    layers[0].adaptive_qsa=runtime.capture_session(prefix);
    layers[1].convolution=tb::full({1,3,7},.25,f);layers[1].recurrent=tb::full({2,5,5},.125,f);
    layers[1].ple_convolution=tb::full({1,2,7},.375,f);layers[1].ple_tokens={11,19};layers[1].ple_batch=1;
    layers[2].keys=tb::full({1,prefix,13},.25,f);layers[2].values=tb::full({1,prefix,13},.5,f);
    layers[2].index=tb::full({1,prefix,7},.75,f);
    state.payload=std::move(layers);state.qwen4_positions=tb::arange(prefix,f.dtype(tb::kInt64)).reshape({1,prefix});
    state.mtp.emplace();state.mtp->cache_pos=prefix;state.mtp->last_target_hidden=tb::zeros({1,1,5},f);
    Qwen4BlockSessionState head;head.adaptive_qsa=runtime.capture_session(prefix);
    // Real predictor layers own independent index archives. This fixture uses
    // the same values for both layers, but must preserve that ownership rule.
    for(auto& archive:head.adaptive_qsa->index)archive=std::make_shared<mfq::cuda::QsaIndexArchive>(*archive);
    state.mtp->qwen4_blocks.push_back(std::move(head));state.mtp->qwen4_positions.push_back(state.qwen4_positions.clone());
    state.mtp->qwen4_lengths.push_back(prefix);
    refresh_cuda_qwen4_session_bytes(state);const auto old_bytes=state.bytes;
    const auto expected=append(prefix,3);
    {
        auto sliced=*std::get<std::vector<Qwen4BlockSessionState>>(state.payload)[0].adaptive_qsa;
        auto reads=std::make_shared<std::size_t>(0);
        for(auto& archive:sliced.index) {
            const auto bytes=archive->read();
            auto reader=std::make_shared<tyloinfer::qsa::RamKvReadSource>();reader->bytes=bytes->size();
            reader->read_range_into=[bytes,reads](std::size_t at,std::uint8_t* destination,std::size_t count) {
                require(at==8*iw && count==2*(iw+2),"Borrowed index restore requested unused FP32 data");
                ++*reads;std::memcpy(destination,bytes->data()+at,count);
            };
            archive=std::make_shared<mfq::cuda::QsaIndexArchive>();archive->bind(reader,0,bytes->size());
        }
        runtime.reset();runtime.restore_session(sliced,true);same(expected,append(prefix,3));
        require(*reads==sliced.index.size(),"Borrowed index restore did not read each exact INT16 slice once");
    }
    std::vector<CudaPagedPayload> payloads;
    for(size_t i=0;i<(prefix+page-1)/page;++i)payloads.push_back(encode_cuda_qwen4_paged_block(state,page,i));
    auto decoded=decode_cuda_qwen4_paged_session(payloads,state.tokens,page);
    auto packed=std::get<std::vector<Qwen4BlockSessionState>>(decoded.payload)[0].adaptive_qsa;
    require(packed && !packed->borrow_prefix,"Standalone codec unexpectedly enabled SSD borrowing");
    runtime.reset();failed_restore(runtime,*packed,false);
    runtime.restore_session(*packed,false);same(expected,append(prefix,3));
    require(decoded.mtp && decoded.mtp->qwen4_blocks.size()==1 && decoded.mtp->qwen4_blocks[0].adaptive_qsa,
        "Compressed native MTP prefix omitted its head KV");
    runtime.reset();runtime.restore_session(*decoded.mtp->qwen4_blocks[0].adaptive_qsa,false);same(expected,append(prefix,3));
    {
        // A normal checkpoint may already be on SSD when the same token
        // boundary gains MTP state in disposable RAM. Read the new version
        // before RAM pressure eventually writes it to the same SSD key.
        auto ordinary=state;ordinary.mtp.reset();
        mfq::cache::PagedPrefixCacheConfig upgrade;
        upgrade.cache_dir=directory/"ram-upgrade";
        upgrade.compatibility_key="mtp-upgrade-"+std::to_string(bits)+"-"+std::to_string(page);
        upgrade.block_size_tokens=page;upgrade.max_disk_bytes=upgrade.max_hot_bytes=8<<20;
        upgrade.max_pending_bytes=0;upgrade.min_disk_free_bytes=0;upgrade.spill_hot_on_eviction=true;
        mfq::cache::PagedPrefixCache cache(upgrade);
        std::vector<mfq::cache::BlockHash> blocks;mfq::cache::BlockHash parent{};
        for(size_t i=0;i<payloads.size();++i) {
            const auto at=i*page,count=std::min(page,state.tokens.size()-at);
            parent=cache.store(parent,state.tokens.data()+at,count,encode_cuda_qwen4_paged_block(ordinary,page,i));
            blocks.push_back(parent);
        }
        cache.trim_hot();cache.flush();
        const auto previous=cache.borrow(blocks.back());require(bool(previous),"Missing ordinary SSD checkpoint");
        const auto i=payloads.size()-1,at=i*page,count=state.tokens.size()-at;
        cache.replace(i?blocks[i-1]:mfq::cache::BlockHash{},state.tokens.data()+at,count,payloads.back());
        require(!cache.borrow(blocks.back()),"RAM MTP upgrade borrowed an obsolete ordinary SSD checkpoint");
        auto prompt=state.tokens;prompt.push_back(999);
        auto upgraded=decode_cuda_qwen4_paged_prefix(cache,blocks,prompt,page);
        require(upgraded.mtp && upgraded.mtp->cache_pos==prefix && upgraded.tokens.size()==prefix &&
                upgraded.mtp->qwen4_blocks.size()==1 && upgraded.mtp->qwen4_blocks[0].adaptive_qsa,
            "Deferred RAM MTP checkpoint was lost behind an older SSD file");
        runtime.reset();runtime.restore_session(*upgraded.mtp->qwen4_blocks[0].adaptive_qsa,false);
        same(expected,append(prefix,3));
        cache.trim_hot();cache.flush();const auto current=cache.borrow(blocks.back());
        require(current && current!=previous && current->payload_bytes>previous->payload_bytes,
            "RAM MTP upgrade did not become the new durable file version");
        std::cout<<"RAM MTP upgrade supersedes older SSD checkpoint without eager persistence PASS bits="<<bits<<" page="<<page<<"\n";
    }
    mfq::cache::PagedPrefixCacheConfig cfg;cfg.cache_dir=directory;cfg.compatibility_key="packed-prefix-"+std::to_string(bits)+"-"+std::to_string(page);
    cfg.block_size_tokens=page;cfg.max_disk_bytes=8<<20;cfg.max_hot_bytes=0;cfg.max_pending_bytes=0;cfg.min_disk_free_bytes=0;
    mfq::cache::PagedPrefixCache cache(cfg);std::vector<mfq::cache::BlockHash> blocks;mfq::cache::BlockHash parent{};
    for(size_t i=0;i<payloads.size();++i) {
        const auto at=i*page,count=std::min(page,state.tokens.size()-at);
        parent=cache.store(parent,state.tokens.data()+at,count,payloads[i]);blocks.push_back(parent);
    }
    cache.flush();bind_cuda_qwen4_paged_prefix(state,cache,blocks);refresh_cuda_qwen4_session_bytes(state);
    require(state.bytes<old_bytes,"Prefix binding retained duplicate compressed RAM images");
    const auto bound=std::get<std::vector<Qwen4BlockSessionState>>(state.payload)[0].adaptive_qsa;
    const auto retained=bound->kv->blocks.front().backing;
    bind_cuda_qwen4_paged_prefix(state,cache,blocks);
    require(bound->kv->blocks.front().backing==retained,"Repeated binding recreated an unchanged prefix reader");
    // Replacing the same token hash creates a new immutable file version.
    // Old snapshots must remain readable, while the next bind uses that version.
    cache.replace({},state.tokens.data(),page,payloads.front());cache.flush();
    bind_cuda_qwen4_paged_prefix(state,cache,blocks);
    require(bound->kv->blocks.front().backing!=retained,"Prefix replacement reused a stale file lease");
    for(size_t i=0;i<bound->kv->blocks.size();++i)
        require(!bound->kv->blocks[i].image && bound->kv->blocks[i].backing && !bound->index[i]->resident,
            "Prefix binding failed to release compressed RAM owners");
    auto prompt=state.tokens;prompt.push_back(999);
    auto imported=decode_cuda_qwen4_paged_prefix(cache,blocks,prompt,page);
    require(imported.tokens==state.tokens,"Streamed prefix selected the wrong recurrent boundary");
    auto restored=std::get<std::vector<Qwen4BlockSessionState>>(imported.payload)[0].adaptive_qsa;
    require(restored->borrow_prefix && restored->durable_prefix,"Streamed prefix lost its SSD file leases");
    for(size_t start=0;start<size_t(prefix);start+=page) {
        const auto first=start/4,last=std::min(start+page,size_t(prefix))/4;
        for(auto id=first;id<last;++id) {
            const auto& b=restored->kv->blocks[id].backing;const auto& index=restored->index[id];
            require(b && b->source && b->source==restored->kv->blocks[first].backing->source &&
                    b->source==index->source && index->offset==b->offset+b->key_bytes+b->value_bytes,
                "Prefix page did not share its immutable KV/index range reader");
        }
    }
    prepare_cuda_qwen4_session_restore(imported,false);require(!restored->borrow_prefix,"Disabled SSD prefix remained enabled");
    require(!imported.mtp->qwen4_blocks[0].adaptive_qsa->borrow_prefix,"Disabled SSD prefix still lent MTP head KV");
    prepare_cuda_qwen4_session_restore(imported,true);require(restored->borrow_prefix,"Enabled SSD prefix did not lend its files");
    cache.clear();runtime.reset();runtime.restore_session(*restored,true);same(expected,append(prefix,3));
    runtime.reset();runtime.restore_session(*imported.mtp->qwen4_blocks[0].adaptive_qsa,true);same(expected,append(prefix,3));
    require(cache.metrics().writes==payloads.size()+1,"Prefix reuse wrote pages beyond the explicit replacement");
    auto truncated=std::make_shared<std::vector<uint8_t>>(*payloads.back());truncated->resize(truncated->size()-1);
    payloads.back()=truncated;bool rejected=false;
    try{(void)decode_cuda_qwen4_paged_session(payloads,state.tokens,page);}catch(const CudaSessionStateError&){rejected=true;}
    require(rejected,"Truncated packed prefix was accepted");
    std::cout<<"Packed QSA prefix, exact INT16 index, unaligned pages, retained SSD file and continuation PASS bits="<<bits<<" page="<<page<<"\n";
}
void pressure(const std::filesystem::path& directory) {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(7<<20));
    setting("TYLOINFER_QSA_TQ_DIRECTORY",(directory/"session").string());
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    constexpr int n=8192,heads=6,kh=2,width=63,iw=4;
    mfq::cuda::QsaTqCache runtime(n+16,heads,kh,width,1,iw,64,f,true);
    const auto append=[&](int start,int count) {
        auto values=tb::full({1,count,kh,width},.03125,f),query=tb::zeros({1,heads,count,width},f.dtype(tb::kFloat16));
        auto raw=tb::zeros({1,count,iw},f),iq=tb::zeros({1,count,1,iw},f);
        auto positions=tb::arange(start,start+count,f.dtype(tb::kInt64)).reshape({1,count});
        const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
        auto destination=tb::empty({1,count,heads,width},f);
        return runtime.forward(query,values,values,iq,raw,positions,pool,false,tb::zeros({1,count,heads,width},f),destination).to(tb::kCPU).contiguous();
    };
    append(0,n);runtime.wait();
    TextSessionState state;state.cache_pos=n;for(int i=0;i<n;++i)state.tokens.push_back(i+1);
    Qwen4BlockSessionState layer;layer.adaptive_qsa=runtime.capture_session(n);
    std::array<size_t,4> tiers{};for(const auto& b:layer.adaptive_qsa->kv->blocks)++tiers[unsigned(b.tier)];
    require(tiers[0] && tiers[1] && tiers[2] && tiers[3],"Pressure prefix did not cover all four real RAM/SSD tiers");
    state.payload=std::vector<Qwen4BlockSessionState>{layer};state.qwen4_positions=tb::arange(n,f.dtype(tb::kInt64)).reshape({1,n});
    const auto expected=append(n,3);
    const auto expected_continuation=append(n+3,12);
    mfq::cache::PagedPrefixCacheConfig cfg;cfg.cache_dir=directory/"prefix";cfg.compatibility_key="four-real-tiers";
    cfg.block_size_tokens=256;cfg.max_disk_bytes=64<<20;cfg.max_hot_bytes=0;cfg.max_pending_bytes=0;cfg.min_disk_free_bytes=0;
    mfq::cache::PagedPrefixCache cache(cfg);mfq::cache::BlockHash parent{};std::vector<mfq::cache::BlockHash> blocks;
    for(size_t i=0;i<n/256;++i){parent=cache.store(parent,state.tokens.data()+i*256,256,encode_cuda_qwen4_paged_block(state,256,i));blocks.push_back(parent);}
    cache.flush();bind_cuda_qwen4_paged_prefix(state,cache,blocks);
    auto prompt=state.tokens;prompt.push_back(999);
    auto restored=decode_cuda_qwen4_paged_prefix(cache,blocks,prompt,256);
    auto snapshot=std::get<std::vector<Qwen4BlockSessionState>>(restored.payload)[0].adaptive_qsa;
    runtime.reset();failed_restore(runtime,*snapshot,true);
    runtime.restore_session(*snapshot,true);same(expected,append(n,3));runtime.wait();
    same(expected_continuation,append(n+3,12));runtime.wait();
    require(runtime.stats().resident_bytes<=size_t(7<<20),"Prefix restore exceeded the RAM pressure budget");
    std::cout<<"Real mixed FP16/A8/TQ4/session-SSD prefix restore under RAM budget PASS\n";
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
}
void shared_codec(bool fingerprints=false) {
    setting("TYLOINFER_QSA_RAM_KV_BITS","4");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    mfq::cuda::QsaTqCache first(160,6,2,63,1,4,64,f,false),second(160,12,4,257,1,4,64,f,false);
    const auto append=[&](auto& cache,int begin,int count,int heads,int kh,int width,float value) {
        auto kv=tb::full({1,count,kh,width},value,f),query=tb::zeros({1,heads,count,width},f.dtype(tb::kFloat16));
        auto raw=tb::zeros({1,count,4},f),iq=tb::zeros({1,count,1,4},f);
        const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
        auto pos=tb::arange(begin,begin+count,f.dtype(tb::kInt64)).reshape({1,count});
        auto result=cache.forward(query,kv,kv,iq,raw,pos,pool,false,tb::zeros({1,count,heads,width},f)).to(tb::kCPU).contiguous();
        double error=0,energy=0;
        for(int64_t i=0;i<result.numel();++i) {
            const double x=result.data_ptr<float>()[i],expected=.5f*value;
            require(std::isfinite(x),"Shared codec produced a nonfinite value");error+=(x-expected)*(x-expected);energy+=expected*expected;
        }
        require(error<std::max(energy,1e-30)*.02,"Shared codec lost constant-value reconstruction");
        if(fingerprints) {
            uint64_t hash=1469598103934665603ULL;const auto* bytes=static_cast<const uint8_t*>(result.data_ptr());
            for(size_t i=0;i<result.nbytes();++i){hash^=bytes[i];hash*=1099511628211ULL;}
            std::cout<<"QSA shared codec sample start="<<begin<<" heads="<<heads<<" width="<<width<<" checksum="<<hash<<"\n";
        }
    };
    append(first,0,128,6,2,63,.03125f);append(second,0,128,12,4,257,-.0625f);
    append(first,128,4,6,2,63,.03125f);append(second,128,8,12,4,257,-.0625f);
    std::cout<<"Shared codec across live caches with different geometry and M4/M8 continuation PASS\n";
}
void large_tq4_prefill(int heads=6,int kh=2,int width=63) {
    setting("TYLOINFER_QSA_RAM_KV_BITS","4");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    constexpr int n=8192,iw=4;
    mfq::cuda::QsaTqCache runtime(n+16,heads,kh,width,1,iw,64,f,false);
    const auto append=[&](int start,int count) {
        auto values=tb::full({1,count,kh,width},.03125,f),query=tb::zeros({1,heads,count,width},f.dtype(tb::kFloat16));
        auto raw=tb::zeros({1,count,iw},f),iq=tb::zeros({1,count,1,iw},f);
        auto positions=tb::arange(start,start+count,f.dtype(tb::kInt64)).reshape({1,count});
        const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
        return runtime.forward(query,values,values,iq,raw,positions,pool,false,tb::zeros({1,count,heads,width},f)).to(tb::kCPU).contiguous();
    };
    const auto begin=std::chrono::steady_clock::now();append(0,n);runtime.wait();
    const auto elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    require(runtime.stats().blocks_per_tier[0]==n/4,"Forced TQ4 append did not actually encode all completed KV blocks");
    const auto result=append(n,4);
    for(int64_t i=0;i<result.numel();++i)require(std::isfinite(result.data_ptr<float>()[i]) && std::abs(result.data_ptr<float>()[i]-.015625f)<.001f,
        "Large TQ4 append lost constant-value reconstruction");
    std::cout<<"8192-token forced TQ4 encoding and M4 continuation PASS kv_heads="<<kh<<" width="<<width<<" elapsed_ms="<<elapsed<<"\n";
}
int main(int argc,char** argv)try {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    const auto directory=argc>1?std::filesystem::path(argv[1]):std::filesystem::current_path()/"packed-prefix-test-data";
    if(argc>2 && std::string(argv[2])=="shared-codec"){shared_codec(true);return 0;}
    if(argc>2 && std::string(argv[2])=="large-tq4"){large_tq4_prefill();large_tq4_prefill(24,4,256);return 0;}
    index_archive_slices(directory/"index-slices");
    for(int bits:{16,8,4})for(size_t page:{size_t(64),size_t(65)})run(bits,page,directory);
    pressure(directory/"pressure");
    shared_codec();
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
