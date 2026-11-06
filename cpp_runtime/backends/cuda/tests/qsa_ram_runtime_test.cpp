#include "runtime/qsa_tq_cache.h"
#include "runtime/qsa_ram_cache.h"
#include "kernels/qsa_ram.h"
#include "runtime/qsa_tq_session.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tb=mfq_tensor_backend;
void setting(const char* name,const std::string& value) {
#ifdef _WIN32
    _putenv_s(name,value.c_str());
#else
    setenv(name,value.c_str(),1);
#endif
}
void run(int bits,int heads,int width,std::size_t cache_bytes) {
    setting("TYLOINFER_QSA_RAM_KV_BITS",std::to_string(bits));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(cache_bytes));
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    constexpr int kh=2,iw=13,length=96;
    mfq::cuda::QsaTqCache cache(128,heads,kh,width,3,iw,128,f,false);
    std::vector<float> source(length*kh*width);
    for(std::size_t i=0;i<source.size();++i)source[i]=.04f*std::sin(float(i)*.13f)+.01f*std::cos(float(i)*.007f);
    auto values=tb::tensor(source,f).reshape({1,length,kh,width});
    const auto pool=[&](const auto& x,const auto&){return tb::zeros({1,x.size(1)/4,iw},f);};
    const auto verify=[&](const auto& output,int start,int count) {
        const auto actual=output.to(tb::kCPU,tb::kFloat32).contiguous();double error=0,energy=0;
        for(int row=0;row<count;++row)for(int head=0;head<heads;++head)for(int d=0;d<width;++d) {
            double mean=0,power=0;
            for(int t=0;t<=start+row;++t){const auto x=source[(t*kh+head/(heads/kh))*width+d];mean+=x;power+=x*x;}
            mean*=.5/(start+row+1);power*=.25/(start+row+1);
            const double difference=actual.data_ptr<float>()[(row*heads+head)*width+d]-mean;
            if(!std::isfinite(difference))throw std::runtime_error("Nonfinite RAM attention output");
            error+=difference*difference;energy+=power;
        }
        const auto tolerance=bits==16?2e-6:bits==8?2e-4:.02;
        if(error>tolerance*std::max(energy,1e-30))throw std::runtime_error("RAM attention failed independent causal reference: bits="+std::to_string(bits)+" NMSE="+std::to_string(error/energy));
        if(cache.position()!=start+count)throw std::runtime_error("RAM append position mismatch");
    };
    const auto append=[&](int start,int count) {
        auto query=tb::zeros({1,heads,count,width},h),raw=tb::zeros({1,count,iw},f),iq=tb::zeros({1,count,3,iw},f);
        auto gate=tb::zeros({1,count,heads,width},h),destination=tb::empty({count,heads,width},h);
        auto positions=tb::arange(start,start+count,f.dtype(tb::kInt64)).reshape({1,count});
        auto v=values.narrow(1,start,count);
        verify(cache.forward(query,v,v,iq,raw,positions,pool,false,gate,destination),start,count);
    };
    int token=0;for(int count:{11,1,2,3,4,5,6,7,8}){append(token,count);token+=count;}
    // Retain both graphs while the exact live position and input change.
    mfq::cuda::DecodeWindow partial(mfq_current_cuda_stream()),complete(mfq_current_cuda_stream());
    auto query=tb::zeros({1,heads,1,width},h),raw=tb::zeros({1,1,iw},f),iq=tb::zeros({1,1,3,iw},f);
    auto gate=tb::zeros({1,1,heads,width},h),destination=tb::empty({1,heads,width},h);
    auto v=tb::empty({1,1,kh,width},f),positions=tb::empty({1,1},f.dtype(tb::kInt64));
    for(int i=0;i<9;++i,++token) {
        v.copy_(values.narrow(1,token,1));positions.copy_(tb::tensor(std::vector<int64_t>{token},f.dtype(tb::kInt64)).reshape({1,1}));
        cache.prepare_decode(true);const auto kind=cache.device_graph_kind(1);
        if(!kind)throw std::runtime_error("All-RAM path unexpectedly disabled its CUDA Graph");
        auto& graph=kind==2?complete:partial;
        if(!graph.valid())graph.capture([&]{cache.forward_pending_graph(query,v,v,iq,raw,positions,gate,destination,pool);},[]{});
        graph.run();cache.advance_graph(1);verify(destination,token,1);
    }
    cache.wait();const auto before=cache.stats();
    if(before.blocks!=std::size_t(token/4) || !before.resident_bytes)throw std::runtime_error("RAM cache accounting lost completed blocks");
    cache.truncate(5);append(5,7);cache.reset();append(0,3);append(3,5);cache.wait();
    std::cout<<"QSA RAM runtime causal M1..8, FP16 gate, immutable archive, Graph phases and rollback PASS bits="<<bits<<" heads="<<heads<<" width="<<width<<" cache="<<cache_bytes<<"\n";
}
void snapshot_roundtrip(int bits) {
    setting("TYLOINFER_QSA_RAM_KV_BITS",std::to_string(bits));setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(1<<20));
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    constexpr int heads=6,kh=2,width=63,iw=7,length=160;
    mfq::cuda::QsaTqCache cache(length,heads,kh,width,1,iw,32,f,false);
    std::vector<float> kv(length*kh*width),raw_values(length*iw),queries(length*heads*width),iq_values(length*iw);
    for(std::size_t i=0;i<kv.size();++i)kv[i]=.13f*std::sin(float(i)*.017f);
    for(std::size_t i=0;i<raw_values.size();++i){raw_values[i]=std::sin(float(i)*.071f);iq_values[i]=std::cos(float(i)*.029f);}
    for(std::size_t i=0;i<queries.size();++i)queries[i]=.2f*std::cos(float(i)*.011f);
    const auto values=tb::tensor(kv,f).reshape({1,length,kh,width}),raw=tb::tensor(raw_values,f).reshape({1,length,iw});
    const auto query=tb::tensor(queries,f).reshape({1,length,heads,width}).permute({0,2,1,3});
    const auto iq=tb::tensor(iq_values,f).reshape({1,length,1,iw});
    const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
    const auto append=[&](int begin,int count){
        auto positions=tb::arange(begin,begin+count,f.dtype(tb::kInt64)).reshape({1,count});
        auto gate=tb::zeros({1,count,heads,width},f);
        return cache.forward(query.narrow(2,begin,count),values.narrow(1,begin,count),values.narrow(1,begin,count),
            iq.narrow(1,begin,count),raw.narrow(1,begin,count),positions,pool,false,gate).to(tb::kCPU).contiguous();
    };
    const auto same=[](const auto& a,const auto& b){
        if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))
            throw std::runtime_error("Packed RAM session changed index selection or attention output");
    };
    append(0,129);auto snapshot=cache.capture_session(129);const auto expected=append(129,3);
    cache.reset();cache.restore_session(*snapshot,false);same(expected,append(129,3));
    auto older=cache.capture_session(125);cache.truncate(125);const auto expected_older=append(125,4);
    cache.reset();cache.restore_session(*older,false);same(expected_older,append(125,4));
    if(snapshot->kv->blocks.size()!=32 || snapshot->index.size()!=32 || snapshot->tail_keys.size(1)!=1)
        throw std::runtime_error("Packed RAM session omitted its partial tail");
    std::cout<<"QSA RAM packed snapshot, exact nonzero INT16 index and partial/older-tail continuation PASS bits="<<bits<<"\n";
}
void pressure(const std::string& directory) {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(1<<20));
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(7<<20));setting("TYLOINFER_QSA_TQ_DIRECTORY",directory);
    constexpr int prompt=8192,heads=6,kh=2,width=63,iw=4;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    mfq::cuda::QsaTqCache cache(prompt+64,heads,kh,width,1,iw,64,f,true);
    const auto pool=[&](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
    const auto append=[&](int start,int count,bool move_query) {
        auto values=tb::full({1,count,kh,width},.03125,f),query=tb::zeros({1,heads,count,width},h);
        std::vector<float> raw_values(count*iw),iq_values(count*iw);
        for(int i=0;i<count;++i) {
            const float x=float((start+i)/4)/float(prompt/4);
            raw_values[i*iw]=x;raw_values[i*iw+1]=x*x;
            if(move_query){iq_values[i*iw]=.25f;iq_values[i*iw+1]=-1.f;}
        }
        auto raw=tb::tensor(raw_values,f).reshape({1,count,iw}),iq=tb::tensor(iq_values,f).reshape({1,count,1,iw});
        auto positions=tb::arange(start,start+count,f.dtype(tb::kInt64)).reshape({1,count});
        auto gate=tb::zeros({1,count,heads,width},h),destination=tb::empty({count,heads,width},h);
        auto result=cache.forward(query,values,values,iq,raw,positions,pool,false,gate,destination).to(tb::kCPU,tb::kFloat32).contiguous();
        for(int64_t i=0;i<result.numel();++i)if(std::abs(result.data_ptr<float>()[i]-.015625f)>.001f)
            throw std::runtime_error("RAM/SSD selected attention lost constant-value reconstruction");
    };
    append(0,prompt,false);append(prompt,1,false);cache.wait();
    const auto cold=cache.stats();std::size_t in_ram=0;for(auto n:cold.blocks_per_tier)in_ram+=n;
    if(!cold.demotions || in_ram>=cold.blocks || cold.resident_bytes>7<<20)
        throw std::runtime_error("Runtime RAM pressure did not quantize and evict to session SSD: demotions="+
            std::to_string(cold.demotions)+" in_ram="+std::to_string(in_ram)+" blocks="+std::to_string(cold.blocks)+" bytes="+std::to_string(cold.resident_bytes));
    append(prompt+1,1,true);cache.wait();const auto hot=cache.stats();
    if(!hot.promotions || hot.canonical_read_bytes<=cold.canonical_read_bytes)
        throw std::runtime_error("Runtime cold SSD selection did not promote hot KV into RAM");
    append(prompt+2,1,true);cache.wait();
    cache.reset();append(0,3,false);cache.wait();
    if(cache.stats().blocks || cache.position()!=3)throw std::runtime_error("RAM/SSD reset retained an old logical block");
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    std::cout<<"QSA RAM runtime 8192-token pressure16->8->4, session SSD, changed indexer query, hot promotion and reset PASS\n";
}
void reclaimer_lifetime() {
    using namespace mfq::cuda;
    const auto stream=mfq_current_cuda_stream();const auto device=mfq_current_cuda_device();
    const auto require=[](bool value){if(!value)throw std::runtime_error("RAM prefix reclaimer ownership/routing failed");};
    require(!register_qsa_ram_reclaimer(device,stream,{}));
    std::size_t first_bytes=0,second_bytes=0;
    auto first=register_qsa_ram_reclaimer(device,stream,[&](std::size_t n){first_bytes+=n;return std::min(n,std::size_t(3));});
    auto second=register_qsa_ram_reclaimer(device,stream,[&](std::size_t n){second_bytes+=n;return n;});
    require(reclaim_qsa_ram(device,stream,0)==0 && first_bytes==0);
    require(reclaim_qsa_ram(device+1,stream,10)==0);
    require(reclaim_qsa_ram(device,stream,10)==10 && first_bytes==10 && second_bytes==7);
    first.reset();first_bytes=second_bytes=0;
    require(reclaim_qsa_ram(device,stream,8)==8 && first_bytes==0 && second_bytes==8);
    second.reset();require(reclaim_qsa_ram(device,stream,8)==0);
    auto nested=register_qsa_ram_reclaimer(device,stream,[&](std::size_t n){
        // Re-enter registration and query on a separate key. No callback may
        // execute under the registry mutex, nor retain an unloaded engine.
        auto inner=register_qsa_ram_reclaimer(device+1,stream,[](std::size_t bytes){return bytes;});
        return reclaim_qsa_ram(device+1,stream,n);
    });
    require(reclaim_qsa_ram(device,stream,7)==7);nested.reset();
    require(reclaim_qsa_ram(device,stream,7)==0 && reclaim_qsa_ram(device+1,stream,7)==0);
    std::cout<<"RAM prefix reclamation registration, lifetime, isolation and callback re-entry PASS\n";
}
void failed_ssd_copy(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    for(std::size_t cache_bytes:{std::size_t(0),std::size_t(8)<<20}) {
        setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(cache_bytes));
        const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
        auto live=tb::zeros({16,int64_t(sizeof(tq::TqBlockView))},options);
        mfq::cuda::QsaRamCache cache(16,64,2,16,options,live,true,directory,
            [](const auto&,const auto&)->std::vector<tq::MicroblockImage>{throw std::runtime_error("Unexpected codec in SSD failure check");});
        auto bytes=std::make_shared<std::vector<std::uint8_t>>(65536);
        auto resident=std::make_shared<tq::RamKvImage>();resident->owner=bytes;
        resident->key=bytes->data();resident->value=bytes->data()+32768;resident->key_bytes=resident->value_bytes=32768;
        bool fail=true;
        auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=8400;
        backing->key_bits=backing->value_bits=4;backing->read=[&] {
            if(fail)throw std::runtime_error("Expected SSD read failure");
            return std::vector<std::uint8_t>(16800);
        };
        tq::RamKvSnapshot state;state.blocks.resize(2);
        for(auto& block:state.blocks)block.bytes={65536,33200,16800};
        state.blocks[0].image=resident;
        state.blocks[1].tier=tq::RamKvTier::Ssd;state.blocks[1].backing=backing;
        cache.restore(state,true);
        auto host=tb::empty({int64_t(2*sizeof(tq::SelectedBlock))},options.device(tb::kCPU));
        host.data_ptr<tq::SelectedBlock>()[0]={0,15};host.data_ptr<tq::SelectedBlock>()[1]={1,15};
        auto selected=host.to(options);
        bool rejected=false;
        try{cache.stage(selected,1,2,live,{},8,0);}
        catch(const std::runtime_error& error){rejected=std::string(error.what())=="Expected SSD read failure";}
        if(!rejected || cudaStreamQuery(mfq_current_cuda_stream())!=cudaSuccess)
            throw std::runtime_error("SSD failure left early RAM readers in flight");
        fail=false;cache.stage(selected,1,2,live,{},8,0);
        cache.truncate(0);cache.wait();
        if(cache.stats().blocks)throw std::runtime_error("SSD retry retained truncated RAM blocks");
    }
    std::cout<<"Early RAM copy with failing SSD: source lifetime, completed stream, retry and reset PASS\n";
}
int main(int argc,char** argv)try {
    reclaimer_lifetime();
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::cout<<"SKIP: CUDA unavailable\n";return 77;}
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(mfq_current_cuda_stream());
    for(int bits:{16,8,4})for(std::size_t bytes:{std::size_t(0),std::size_t(1)<<20}) {
        run(bits,6,63,bytes);run(bits,24,256,bytes);run(bits,12,257,bytes);
    }
    for(int bits:{16,8,4})snapshot_roundtrip(bits);
    pressure(argc>1?argv[1]:(std::filesystem::current_path()/"qsa-ram-runtime-ssd-test-data").string());
    failed_ssd_copy((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"failure").string());
    std::cout<<"QSA RAM runtime integration checks PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
