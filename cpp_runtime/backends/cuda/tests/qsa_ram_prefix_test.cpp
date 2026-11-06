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
void same(const tb::Tensor& a,const tb::Tensor& b) {
    require(a.sizes()==b.sizes() && !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),
        "Compressed prefix changed exact index selection or attention output");
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
    state.mtp->qwen4_blocks.push_back(std::move(head));state.mtp->qwen4_positions.push_back(state.qwen4_positions.clone());
    state.mtp->qwen4_lengths.push_back(prefix);
    refresh_cuda_qwen4_session_bytes(state);const auto old_bytes=state.bytes;
    const auto expected=append(prefix,3);
    std::vector<CudaPagedPayload> payloads;
    for(size_t i=0;i<(prefix+page-1)/page;++i)payloads.push_back(encode_cuda_qwen4_paged_block(state,page,i));
    auto decoded=decode_cuda_qwen4_paged_session(payloads,state.tokens,page);
    auto packed=std::get<std::vector<Qwen4BlockSessionState>>(decoded.payload)[0].adaptive_qsa;
    require(packed && !packed->borrow_prefix,"Standalone codec unexpectedly enabled SSD borrowing");
    runtime.reset();runtime.restore_session(*packed,false);same(expected,append(prefix,3));
    require(decoded.mtp && decoded.mtp->qwen4_blocks.size()==1 && decoded.mtp->qwen4_blocks[0].adaptive_qsa,
        "Compressed native MTP prefix omitted its head KV");
    runtime.reset();runtime.restore_session(*decoded.mtp->qwen4_blocks[0].adaptive_qsa,false);same(expected,append(prefix,3));
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
    for(size_t i=0;i<bound->kv->blocks.size();++i)
        require(!bound->kv->blocks[i].image && bound->kv->blocks[i].backing && !bound->index[i]->resident,
            "Prefix binding failed to release compressed RAM owners");
    auto prompt=state.tokens;prompt.push_back(999);
    auto imported=decode_cuda_qwen4_paged_prefix(cache,blocks,prompt,page);
    require(imported.tokens==state.tokens,"Streamed prefix selected the wrong recurrent boundary");
    auto restored=std::get<std::vector<Qwen4BlockSessionState>>(imported.payload)[0].adaptive_qsa;
    require(restored->borrow_prefix && restored->durable_prefix,"Streamed prefix lost its SSD file leases");
    prepare_cuda_qwen4_session_restore(imported,false);require(!restored->borrow_prefix,"Disabled SSD prefix remained enabled");
    require(!imported.mtp->qwen4_blocks[0].adaptive_qsa->borrow_prefix,"Disabled SSD prefix still lent MTP head KV");
    prepare_cuda_qwen4_session_restore(imported,true);require(restored->borrow_prefix,"Enabled SSD prefix did not lend its files");
    cache.clear();runtime.reset();runtime.restore_session(*restored,true);same(expected,append(prefix,3));
    runtime.reset();runtime.restore_session(*imported.mtp->qwen4_blocks[0].adaptive_qsa,true);same(expected,append(prefix,3));
    require(cache.metrics().writes==payloads.size(),"Prefix reuse unexpectedly duplicated disk writes");
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
    runtime.reset();runtime.restore_session(*snapshot,true);same(expected,append(n,3));runtime.wait();
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
    for(int bits:{16,8,4})for(size_t page:{size_t(64),size_t(65)})run(bits,page,directory);
    pressure(directory/"pressure");
    shared_codec();
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
