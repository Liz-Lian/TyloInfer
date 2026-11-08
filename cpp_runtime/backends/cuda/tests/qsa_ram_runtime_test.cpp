#include "../../../tests/qsa_reader_fixture.h"
#include "runtime/qsa_tq_cache.h"
#include "runtime/qsa_ram_cache.h"
#include "kernels/qsa_ram.h"
#include "runtime/qsa_tq_session.h"
#include "runtime/decode_window.h"
#include "tyloinfer/qsa_ram_policy.h"
#include "tyloinfer/qsa_tq_cpu.h"
#include "tyloinfer/host_memory.h"
#include "mfq_cuda_context.h"
#include <array>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tb=mfq_tensor_backend;
void setting(const char* name,const std::string& value) {
#ifdef _WIN32
    _putenv_s(name,value.c_str());
#else
    setenv(name,value.c_str(),1);
#endif
}
void run(int bits,int heads,int width,std::size_t cache_bytes,int initial_count=11) {
    setting("TYLOINFER_QSA_RAM_KV_BITS",std::to_string(bits));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(cache_bytes));
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    constexpr int kh=2,iw=13,length=192;
    mfq::cuda::QsaTqCache cache(192,heads,kh,width,3,iw,192,f,false);
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
    // The second large chunk must reuse its complete layer staging, including
    // a partial microblock and different query tiles, before Graph decode.
    int token=0;for(int count:{initial_count,67,1,2,3,4,5,6,7,8}){append(token,count);token+=count;}
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
    const auto memory=cache.ram_memory();
    if(!memory || memory->payload!=before.resident_bytes || memory->allocated<memory->payload || memory->disk)
        throw std::runtime_error("QSA RAM telemetry lost payload, physical pages or placement");
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
    // Score tiling must keep the first canonical GPU image alive across all
    // queries, including nonconstant index selection and a partial last block.
    setting("TYLOINFER_QSA_PREFILL_SCORE_BYTES",std::to_string(64ULL<<20));
    const auto expected_initial=append(0,129);cache.reset();
    setting("TYLOINFER_QSA_PREFILL_SCORE_BYTES","512");
    const auto actual_initial=append(0,129);
    for(std::int64_t i=0;i<expected_initial.numel();++i)
        if(std::abs(expected_initial.data_ptr<float>()[i]-actual_initial.data_ptr<float>()[i])>2e-5f)
            throw std::runtime_error("Score tiling changed fresh canonical attention");
    setting("TYLOINFER_QSA_PREFILL_SCORE_BYTES",std::to_string(64ULL<<20));
    auto snapshot=cache.capture_session(129);const auto expected=append(129,17);
    cache.reset();cache.release_idle();cache.restore_session(*snapshot,false);same(expected,append(129,17));
    auto older=cache.capture_session(125);cache.truncate(125);const auto expected_older=append(125,4);
    cache.reset();cache.release_idle();cache.restore_session(*older,false);same(expected_older,append(125,4));
    if(snapshot->kv->blocks.size()!=32 || snapshot->index.size()!=32 || snapshot->tail_keys.size(1)!=1)
        throw std::runtime_error("Packed RAM session omitted its partial tail");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    cache.reset();
    std::map<const mfq::cuda::QsaIndexArchive*,std::shared_ptr<mfq::cuda::QsaIndexArchive>> archives;
    for(const auto& image:snapshot->index)archives.emplace(image.get(),image);
    for(const auto& image:older->index)archives.emplace(image.get(),image);
    std::size_t expected_release=0;
    for(const auto& [identity,image]:archives)if(image->resident)expected_release+=image->resident->capacity();
    const auto host_before=mfq::cuda::tensor_host_bytes.load();
    auto released=std::make_shared<tyloinfer::qsa::RamKvReadSource>();
    released->read_range=[](std::size_t,std::size_t){return std::vector<std::uint8_t>{};};
    for(const auto& [identity,image]:archives)image->bind(released,0,0);
    if(!expected_release || host_before-mfq::cuda::tensor_host_bytes.load()!=expected_release)
        throw std::runtime_error("Indexer archive RAM was uncharged or retained after prefix binding");
#endif
    std::cout<<"QSA RAM packed snapshot, exact nonzero INT16 index and partial/older-tail continuation PASS bits="<<bits<<"\n";
}
void pressure(const std::string& directory) {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(1<<20));
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(7<<20));setting("TYLOINFER_QSA_TQ_DIRECTORY",directory);
    constexpr int prompt=8192,heads=6,kh=2,width=63,iw=4;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    mfq::cuda::QsaTqCache cache(prompt+128,heads,kh,width,1,iw,64,f,true);
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
    if(cold.resident_bytes>(7<<20)-(7<<20)/64)
        throw std::runtime_error("SSD pressure left no batch margin for fresh KV and hot recalls");
    append(prompt+1,1,true);cache.wait();const auto hot=cache.stats();
    if(!hot.promotions || hot.canonical_read_bytes<=cold.canonical_read_bytes)
        throw std::runtime_error("Runtime cold SSD selection did not promote hot KV into RAM");
    append(prompt+2,1,true);cache.wait();
    for(int host_mode:{4,8}) {
        // Restore the same mixed-precision images for the eager reference.
        // Exercise changing selections and token data, retained graphs,
        // partial/complete blocks and rollback while SSD images remain live.
        mfq::cuda::QsaTqCache reference(prompt+128,heads,kh,width,1,iw,64,f,true);
        struct Inputs {tb::Tensor q,k,v,iq,raw,pos,gate,out;};
        std::map<int,Inputs> inputs;
        std::map<std::pair<int,int>,std::unique_ptr<mfq::cuda::DecodeWindow>> windows;
        for(int n=1;n<=8;++n) {
            auto& x=inputs[n];x.q=tb::zeros({1,heads,n,width},h);
            x.k=tb::zeros({1,n,kh,width},h);x.v=tb::zeros_like(x.k);
            x.iq=tb::zeros({1,n,1,iw},f);x.raw=tb::zeros({1,n,iw},f);
            x.pos=tb::zeros({1,n},f.dtype(tb::kInt64));
            x.gate=tb::full({1,n,heads,width},.31,h);x.out=tb::empty({n,heads,width},h);
        }
        int iteration=0;
        for(int n:{1,2,3,4,5,6,7,8,1,1,1,4}) {
            if(iteration==6)cache.truncate(cache.position()-3);
            const int64_t start=cache.position();auto& x=inputs[n];
            auto state=cache.capture_session(start);reference.reset();reference.restore_session(*state,true);
            std::vector<float> k(n*kh*width),v(k.size()),q(n*heads*width),r(n*iw),iq(n*iw);
            for(std::size_t i=0;i<k.size();++i){k[i]=.08f*std::sin(float(i)*.13f+iteration);v[i]=.1f*std::cos(float(i)*.07f+iteration);}
            for(std::size_t i=0;i<q.size();++i)q[i]=.2f*std::sin(float(i)*.03f+iteration);
            for(int i=0;i<n;++i){r[i*iw]=float(start+i)/prompt;r[i*iw+1]=r[i*iw]*r[i*iw];iq[i*iw]=iteration%2?.25f:-.25f;iq[i*iw+1]=iteration%2?-1.f:1.f;}
            x.k.copy_(tb::tensor(k,f).reshape(x.k.sizes()));x.v.copy_(tb::tensor(v,f).reshape(x.v.sizes()));
            x.q.copy_(tb::tensor(q,f).reshape(x.q.sizes()));x.raw.copy_(tb::tensor(r,f).reshape(x.raw.sizes()));
            x.iq.copy_(tb::tensor(iq,f).reshape(x.iq.sizes()));x.pos.copy_(tb::arange(start,start+n,x.pos.options()).reshape({1,n}));
            auto expected=reference.forward(x.q,x.k,x.v,x.iq,x.raw,x.pos,pool,false,x.gate).to(tb::kCPU,tb::kFloat32).contiguous();
            const int kind=cache.device_graph_kind(n);if(!kind)throw std::runtime_error("SSD disabled its prepared device work");
            if(host_mode==8 && !cache.graph_task_supported())throw std::runtime_error("SSD graph task was unavailable for codec-free backing");
            cache.prepare_decode_tokens(n,true,host_mode);auto& window=windows[{n,kind&3}];
            if(!window)window=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
            if(!window->valid())window->capture([&]{cache.forward_pending_graph(x.q,x.k,x.v,x.iq,x.raw,x.pos,x.gate,x.out,pool);},[]{});
            if(window->segments()!=(host_mode==8?1:2))throw std::runtime_error("SSD QSA graph submission count changed");
            window->run();cache.advance_graph(n);
            auto actual=x.out.to(tb::kCPU,tb::kFloat32).contiguous();double error=0,energy=0,maximum=0;
            for(int64_t i=0;i<actual.numel();++i){const double a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i],d=a-b;error+=d*d;energy+=b*b;maximum=std::max(maximum,std::abs(d));}
            if(!std::isfinite(error) || maximum>.002 || error>5e-4*std::max(energy,1e-30))
                throw std::runtime_error("SSD graph/eager mismatch M="+std::to_string(n)+" max="+std::to_string(maximum));
            if(cache.position()!=start+n)throw std::runtime_error("SSD graph replay changed its logical epoch");
            ++iteration;
        }
        cache.wait();reference.wait();
        std::cout<<"QSA RAM/SSD M1..8 host_mode="<<host_mode<<" nonconstant Graph/eager parity, retained graphs and rollback PASS\n";
    }
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
    require(!register_qsa_ram_capacity_reclaimer(device,stream,{}));
    std::size_t critical_bytes=0;
    const auto caller=std::this_thread::get_id();
    auto critical=register_qsa_ram_capacity_reclaimer(device,stream,[&](std::size_t n){
        require(std::this_thread::get_id()==caller);critical_bytes+=n;return n;
    });
    std::size_t first_bytes=0,second_bytes=0;
    auto first=register_qsa_ram_reclaimer(device,stream,[&](std::size_t n){first_bytes+=n;return std::min(n,std::size_t(3));});
    auto second=register_qsa_ram_reclaimer(device,stream,[&](std::size_t n){second_bytes+=n;return n;});
    require(reclaim_qsa_ram(device,stream,0)==0 && first_bytes==0);
    require(reclaim_qsa_ram(device+1,stream,10)==0);
    require(reclaim_qsa_ram(device,stream,10)==10 && first_bytes==10 && second_bytes==7);
    require(critical_bytes==0);
    require(reclaim_qsa_ram_capacity(device+1,stream,9)==0);
    require(reclaim_qsa_ram_capacity(device,stream,9)==9 && critical_bytes==9 &&
        first_bytes==10 && second_bytes==7);
    first.reset();first_bytes=second_bytes=0;
    require(reclaim_qsa_ram(device,stream,8)==8 && first_bytes==0 && second_bytes==8);
    second.reset();require(reclaim_qsa_ram(device,stream,8)==0);
    require(critical_bytes==9);critical.reset();require(reclaim_qsa_ram_capacity(device,stream,8)==0);
    auto nested=register_qsa_ram_reclaimer(device,stream,[&](std::size_t n){
        // Re-enter registration and query on a separate key. No callback may
        // execute under the registry mutex, nor retain an unloaded engine.
        auto inner=register_qsa_ram_reclaimer(device+1,stream,[](std::size_t bytes){return bytes;});
        return reclaim_qsa_ram(device+1,stream,n);
    });
    require(reclaim_qsa_ram(device,stream,7)==7);nested.reset();
    require(reclaim_qsa_ram(device,stream,7)==0 && reclaim_qsa_ram(device+1,stream,7)==0);
    std::vector<std::size_t> reservation1,reservation2;
    auto r1=register_qsa_ram_budget_reserver(device,stream,[&](std::size_t n){reservation1.push_back(n);return n;});
    auto r2=register_qsa_ram_budget_reserver(device,stream,[&](std::size_t n){reservation2.push_back(n);return n/2;});
    require(reserve_qsa_ram_budget(device,stream,10)==15);
    require(reservation1==std::vector<std::size_t>{10} && reservation2==reservation1);
    require(reserve_qsa_ram_budget(device,stream,0)==0 && reservation1.back()==0 && reservation2.back()==0);
    r1.reset();r2.reset();require(reserve_qsa_ram_budget(device,stream,7)==0);
    std::cout<<"RAM prefix/expert reclamation priority, foreground execution, lifetime, isolation and callback re-entry PASS\n";
}
void combined_ram_caps(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(4<<20));
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES",std::to_string(1<<20));
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto live=tb::zeros({4096,int64_t(sizeof(tq::TqBlockView))},options);
    mfq::cuda::QsaRamCache cache(4096,2,1,4,options,live,true,directory,
        [](const auto& images,const auto&){return images;});
    tq::MicroblockImage image;image.key.resize(264);image.value=image.key;
    image.key_bits=image.value_bits=4;
    cache.append(std::vector<tq::MicroblockImage>(4096,image),16384,false);cache.wait();
    if(cache.memory().payload>(1<<20) || !cache.memory().tokens[3])
        throw std::runtime_error("Per-layer RAM cap bypassed the smaller shared RAM cap");
    cache.truncate(0);
    // Clearing an override must restore automatic capacity on Windows/Linux.
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    cache.append(std::vector<tq::MicroblockImage>(4096,image),16384,false);cache.wait();
    if(cache.memory().tokens[3] || cache.memory().payload!=4096*528)
        throw std::runtime_error("Cleared shared RAM override retained a zero-byte cap");
    cache.truncate(0);setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    std::cout<<"Combined shared/per-layer RAM caps and cleared override PASS\n";
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
        backing->key_bits=backing->value_bits=4;qsa_test::reader(*backing).read_range=[&](std::size_t,std::size_t){
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
        // Capture with a working reader, then force a real miss to fail during
        // replay. All mapped GPU waiters must be released before propagating it.
        cache.restore(state,true);cache.prepare();cache.prepare_graph_task(8);
        {
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{cache.stage_graph_task(selected,1,2,live,{},0);},[]{});
            if(graph.segments()!=1)throw std::runtime_error("SSD task inserted a host graph boundary");
            cache.restore(state,true);cache.prepare();cache.prepare_graph_task(8);fail=true;rejected=false;
            try{graph.run();}catch(const std::runtime_error& error){rejected=std::string(error.what())=="Expected SSD read failure";}
            if(!rejected || cudaStreamQuery(mfq_current_cuda_stream())!=cudaSuccess)
                throw std::runtime_error("SSD graph failure did not cancel and join its GPU readers");
        }
        fail=false;selected.copy_(host.to(options));cache.prepare();cache.prepare_graph_task(8);
        {
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{cache.stage_graph_task(selected,1,2,live,{},0);},[]{});graph.run();
        }
        cache.truncate(0);cache.wait();
        if(cache.stats().blocks)throw std::runtime_error("SSD retry retained truncated RAM blocks");
    }
    std::cout<<"Early RAM copy with failing SSD: source lifetime, completed stream, retry and reset PASS\n";
}
void prefill_layer_bytes(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(2*65536));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    constexpr int blocks=768;
    auto live_cpu=tb::zeros({blocks+1,int64_t(sizeof(tq::TqBlockView))},options.device(tb::kCPU));
    live_cpu.data_ptr<tq::TqBlockView>()[blocks].key_bits=77;
    auto live=live_cpu.to(options);
    mfq::cuda::QsaRamCache cache(blocks+1,64,16,16,options,live,true,directory,
        [](const auto&,const auto&)->std::vector<tq::MicroblockImage>{throw std::runtime_error("Unexpected prefill-layer codec");});
    tq::RamKvSnapshot state;state.blocks.resize(blocks);std::size_t reads=0,expected_bytes=0;
    const auto key_size=[](unsigned id){return id<2 || id%3==0?32768u:id%3==1?16600u:8380u;};
    const auto bits=[](unsigned id){return id<2 || id%3==0?0u:id%3==1?unsigned(tq::affine8_tag):4u;};
    for(unsigned id=0;id<blocks;++id) {
        auto& block=state.blocks[id];block.bytes={65536,33200,16760};block.access=1;
        const auto size=key_size(id);expected_bytes+=(((size+15)&~std::size_t(15))+size+15)&~std::size_t(15);
        if(id<2) {
            auto owner=std::make_shared<std::vector<std::uint8_t>>(2*size,std::uint8_t(id));
            auto image=std::make_shared<tq::RamKvImage>();image->owner=owner;
            image->key=owner->data();image->value=owner->data()+size;image->key_bytes=image->value_bytes=size;
            block.image=image;
        }else {
            auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=size;
            backing->key_bits=backing->value_bits=std::uint8_t(bits(id));
            backing->key_state_bits=backing->value_state_bits=bits(id)==4?14:0;
            backing->tier=bits(id)==0?tq::RamKvTier::Fp16:bits(id)==tq::affine8_tag?tq::RamKvTier::Affine8:tq::RamKvTier::Tq4;
            qsa_test::reader(*backing).read_range_into=[&,id](std::size_t,std::uint8_t* p,std::size_t n){++reads;std::fill(p,p+n,std::uint8_t(id));};
            block.tier=tq::RamKvTier::Ssd;block.backing=backing;
        }
    }
    cache.restore(state,true);auto stage=cache.stage_prefill(blocks);
    if(!stage || stage->payload.nbytes()!=expected_bytes || reads!=blocks-2)
        throw std::runtime_error("Original prefill-layer staging lost bytes or reread a block");
    if(!stage->payload.is_cuda() || !stage->views.is_cuda())
        throw std::runtime_error("QSA prefill payload and descriptors must stay in VRAM after CPU transfer allocation");
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));cache.wait();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    const auto host_before=mfq::cuda::tensor_host_bytes.load();
    mfq::cuda::tensor_host_peak_bytes.store(host_before);
#endif
    auto repeated=cache.stage_prefill(blocks);
    if(!repeated || reads!=2*(blocks-2))throw std::runtime_error("Repeated prefill staging lost a source");
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if(mfq::cuda::tensor_host_peak_bytes.load()>host_before+(32ULL<<10))
        throw std::runtime_error("Repeated prefill staging must reuse pinned transfer packets");
#endif
    cache.truncate(0); // The submitted staging must outlive all original owners.
    auto payload=stage->payload.to(tb::kCPU),views=stage->views.to(tb::kCPU);
    auto repeated_payload=repeated->payload.to(tb::kCPU),repeated_views=repeated->views.to(tb::kCPU);
    if(repeated_payload.nbytes()!=payload.nbytes())
        throw std::runtime_error("Reused prefill packets changed the layer extent");
    const auto base=reinterpret_cast<std::uintptr_t>(stage->payload.data_ptr());
    const auto repeated_base=reinterpret_cast<std::uintptr_t>(repeated->payload.data_ptr());
    for(unsigned id=0;id<blocks;++id) {
        const auto& v=views.data_ptr<tq::TqBlockView>()[id];const auto size=key_size(id);
        const auto flag=tq::tq_precision_flags(bits(id),bits(id)==4?14:0);
        if(v.key_bits!=flag || v.value_bits!=flag || reinterpret_cast<std::uintptr_t>(v.key)%16 || reinterpret_cast<std::uintptr_t>(v.value)%16)
            throw std::runtime_error("Prefill staging changed precision flags or alignment");
        const auto& again=repeated_views.data_ptr<tq::TqBlockView>()[id];
        for(int kind=0;kind<2;++kind) {
            const auto pointer=kind?v.value:v.key;
            const auto offset=reinterpret_cast<std::uintptr_t>(pointer)-base;
            if(offset>payload.nbytes() || size>payload.nbytes()-offset)
                throw std::runtime_error("Prefill staging pointer exceeds its payload");
            const auto* p=payload.data_ptr<std::uint8_t>()+offset;
            for(unsigned i=0;i<size;++i)if(p[i]!=std::uint8_t(id))throw std::runtime_error("Prefill staging changed original bytes");
            const auto repeated_offset=reinterpret_cast<std::uintptr_t>(kind?again.value:again.key)-repeated_base;
            if(repeated_offset>repeated_payload.nbytes() || size>repeated_payload.nbytes()-repeated_offset ||
                    std::memcmp(p,repeated_payload.data_ptr<std::uint8_t>()+repeated_offset,size))
                throw std::runtime_error("Reused prefill packets changed an earlier layer's bytes");
        }
    }
    if(views.data_ptr<tq::TqBlockView>()[blocks].key_bits!=77)
        throw std::runtime_error("Prefill staging overwrote the pending tail");
    cache.wait();
    std::cout<<"Prefill layer staging: mixed original FP16/Affine8/TQ4, bounded reusable pinned ring, padded views, lifetime and untouched tail PASS\n";
}
void prefill_pressure_recheck(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    constexpr std::size_t key_bytes=8376,image_bytes=2*key_bytes;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(4*image_bytes));
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto live=tb::zeros({4,int64_t(sizeof(tq::TqBlockView))},options);
    mfq::cuda::QsaRamCache cache(4,64,4,4,options,live,true,directory,
        [](const auto& images,const auto&){return images;});
    std::vector<tq::MicroblockImage> images(4);
    for(unsigned id=0;id<images.size();++id) {
        auto& image=images[id];image.key.assign(key_bytes,std::uint8_t(id+3));image.value=image.key;
        image.key_bits=image.value_bits=4;image.key_state_bits=image.value_state_bits=14;
    }
    cache.append(std::move(images),1);cache.wait();
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(image_bytes));
    auto stage=cache.stage_prefill(4);
    if(!stage || cache.stats().resident_bytes>image_bytes || cache.stats().canonical_read_bytes<3*image_bytes)
        throw std::runtime_error("Prefill did not apply the current RAM budget before retaining its snapshot");
    auto views=stage->views.to(tb::kCPU);
    for(unsigned id=0;id<4;++id) {
        const auto& view=views.data_ptr<tq::TqBlockView>()[id];
        for(const auto* pointer:{view.key,view.value}) {
            std::vector<std::uint8_t> actual(key_bytes);
            TYLOINFER_CUDA_CHECK(cudaMemcpy(actual.data(),pointer,actual.size(),cudaMemcpyDefault));
            for(auto byte:actual)if(byte!=std::uint8_t(id+3))
                throw std::runtime_error("Pressure-aware prefill changed a spilled KV image");
        }
    }
    cache.truncate(0);cache.wait();
    std::cout<<"Prefill rechecks reduced RAM budget, spills cold KV and retains exact original bytes PASS\n";
}
void packed_spill_cache(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(2*65536));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(1<<20));
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    constexpr int capacity=128,count=16;
    auto live=tb::zeros({capacity,int64_t(sizeof(tq::TqBlockView))},options);
    mfq::cuda::QsaRamCache cache(capacity,64,count,16,options,live,true,directory,
        [](const auto&,const auto&)->std::vector<tq::MicroblockImage>{throw std::runtime_error("Unexpected spill-cache codec");});
    std::size_t reads=0;tq::RamKvSnapshot state;state.blocks.resize(capacity);
    for(unsigned id=0;id<capacity;++id) {
        auto& block=state.blocks[id];block.bytes={65536,33200,16760};block.access=1;
        if(id<2) {
            auto bytes=std::make_shared<std::vector<std::uint8_t>>(65536,std::uint8_t(id+5));
            auto image=std::make_shared<tq::RamKvImage>();image->owner=bytes;
            image->key=bytes->data();image->value=bytes->data()+32768;image->key_bytes=image->value_bytes=32768;
            block.image=image;
        }else {
            auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=8380;
            backing->key_bits=backing->value_bits=4;
            qsa_test::reader(*backing).read_range=[&,id](std::size_t,std::size_t){++reads;return std::vector<std::uint8_t>(16760,std::uint8_t(id+5));};
            block.tier=tq::RamKvTier::Ssd;block.backing=backing;
        }
    }
    // The same payload fits eight full-size images or 24 compact TQ4 images.
    // Include FP16 images occupying four compact base slots each. Build a
    // mixed working set whose total slot footprint fits each hash set.
    std::vector<int> ids{0,1};unsigned used[3]{};
    for(unsigned id=0;id<2;++id)used[tq::ram_cache_set(id,3)]+=4;
    for(unsigned id=2;id<capacity && ids.size()<count;++id) {
        const auto set=tq::ram_cache_set(id,3);
        if(used[set]<8){++used[set];ids.push_back(int(id));}
    }
    if(ids.size()!=count)throw std::runtime_error("Spill-cache test could not populate its sets");
    auto host=tb::empty({int64_t(count*sizeof(tq::SelectedBlock))},options.device(tb::kCPU));
    const auto selections=[&] {
        for(int at=0;at<count;++at)host.data_ptr<tq::SelectedBlock>()[at]={ids[at],ids[at]<0?0u:15u};
        return host.to(options);
    };
    auto selected=selections();
    const auto verify=[&](const auto& views,unsigned seed) {
        auto records=views.to(tb::kCPU).contiguous();
        for(int at=0;at<count;++at)if(ids[at]>=0) {
            const auto& view=records.data_ptr<tq::TqBlockView>()[at];
            std::vector<std::uint8_t> bytes(ids[at]<2?32768:8380);
            for(const auto* source:{view.key,view.value}) {
                if(cudaMemcpy(bytes.data(),source,bytes.size(),cudaMemcpyDefault)!=cudaSuccess)
                    throw std::runtime_error("Packed spill-cache payload download failed");
                for(auto value:bytes)if(value!=std::uint8_t(ids[at]+seed))
                    throw std::runtime_error("Packed spill-cache payload changed");
            }
        }
    };
    cache.restore(state,true);cache.prepare();
    for(int repeat=0;repeat<2;++repeat)verify(cache.stage(selected,1,count,live,{},8,0),5);
    if(reads!=count-2)throw std::runtime_error("Compact spill cache reread selected SSD images: reads="+std::to_string(reads));
    cache.truncate(0);
    std::vector<tq::MicroblockImage> fresh(2);
    for(unsigned id=0;id<2;++id) {
        fresh[id].key.assign(32768,std::uint8_t(id+11));fresh[id].value=fresh[id].key;
        fresh[id].key_bits=fresh[id].value_bits=0;
    }
    cache.append(std::move(fresh),1);cache.prepare();
    const auto cold_ids=ids;ids.assign(count,-1);ids[0]=1;ids[1]=0;selected.copy_(selections());
    // Capture while the cache has full-size slots, then retain the collector
    // across compact SSD-slot transitions. Its descriptor must remain live.
    mfq::cuda::DecodeWindow collector(mfq_current_cuda_stream());
    bool collected=false;
    auto collector_fallback=tb::zeros({1},options.dtype(tb::kUInt8));
    collector.capture([&]{collected=cache.collect_graph_selection(selected,1,count);if(!collected)collector_fallback.zero_();},[]{});
    {
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        tb::Tensor views;
        graph.capture([&]{views=cache.stage(selected,1,count,live,{},8,0);},[]{});
        graph.run();verify(views,11);
        std::swap(ids[0],ids[1]);selected.copy_(selections());graph.run();verify(views,11);
    }
    ids=cold_ids;selected.copy_(selections());cache.restore(state,true);cache.prepare();
    for(int repeat=0;repeat<2;++repeat){collector.run();verify(cache.stage(selected,1,count,live,{},8,0,collected),5);}
    if(reads!=2*(count-2))throw std::runtime_error("Reset spill cache retained stale versions or missed packed hits");
    collector.run(); // The hot selection currently needs no SSD reads.
    auto changed=state;
    for(unsigned id=0;id<capacity;++id) {
        auto& block=changed.blocks[id];
        if(block.image) {
            auto bytes=std::make_shared<std::vector<std::uint8_t>>(65536,std::uint8_t(id+23));
            auto image=std::make_shared<tq::RamKvImage>();image->owner=bytes;
            image->key=bytes->data();image->value=bytes->data()+32768;image->key_bytes=image->value_bytes=32768;
            block.image=std::move(image);
        }else {
            auto backing=std::make_shared<tq::RamKvBacking>(*block.backing);
            qsa_test::reader(*backing).read_range=[&,id](std::size_t,std::size_t){++reads;return std::vector<std::uint8_t>(16760,std::uint8_t(id+23));};
            block.backing=std::move(backing);
        }
    }
    // Replace the sources after the collector joins, without preparing a new
    // snapshot. The revision check must reject its old empty miss list.
    cache.restore(changed,true);
    verify(cache.stage(selected,1,count,live,{},8,0,collected),23);
    if(reads!=3*(count-2))throw std::runtime_error("Stale captured miss set escaped publication revision validation");
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    std::cout<<"Same-budget packed SSD cache, live-layout Graph collector, stale-publication retry and exact SSD-read counts PASS\n";
}
void graph_source_transaction() {
    namespace tq=tyloinfer::qsa;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8),host=options.device(tb::kCPU).pinned_memory(true);
    auto payload=tb::zeros({128},options),original=tb::zeros({int64_t(2*sizeof(tq::RamBlockView))},host);
    auto* records=original.data_ptr<tq::RamBlockView>();
    for(int id=0;id<2;++id)records[id]={{payload.data_ptr<std::uint8_t>()+id*32,payload.data_ptr<std::uint8_t>()+id*32+16,4,4},16,16,std::uint64_t(id+5)};
    auto sources=original.to(options),images=tb::empty_like(original),ids=tb::empty({2},host.dtype(tb::kInt32));
    auto flags=tb::zeros({16},host.dtype(tb::kInt32)),saved=tb::zeros({1},options.dtype(tb::kInt32)),backup=tb::empty_like(sources);
    auto selected_host=tb::empty({int64_t(2*sizeof(tq::SelectedBlock))},host);
    selected_host.data_ptr<tq::SelectedBlock>()[0]={0,15};selected_host.data_ptr<tq::SelectedBlock>()[1]={1,7};
    auto selected=selected_host.to(options);
    ids.data_ptr<std::int32_t>()[0]=0;ids.data_ptr<std::int32_t>()[1]=1;
    const auto map=[](const auto& tensor){void* result=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&result,tensor.data_ptr(),0));return result;};
    auto* mapped_ids=static_cast<std::int32_t*>(map(ids));auto* mapped_images=static_cast<tq::RamBlockView*>(map(images));
    auto* mapped_flags=static_cast<std::uint32_t*>(map(flags));auto* control=flags.data_ptr<std::uint32_t>();
    const auto stream=mfq_current_cuda_stream();
    const auto publish=[&] {
        TYLOINFER_CUDA_CHECK(cudaError_t(tq::tq_publish_ram_graph(sources.data_ptr<tq::RamBlockView>(),2,mapped_ids,mapped_images,mapped_flags,
            backup.data_ptr<tq::RamBlockView>(),saved.data_ptr<std::uint32_t>(),2,stream)));
    };
    const auto restore=[&] {
        TYLOINFER_CUDA_CHECK(cudaError_t(tq::tq_restore_ram_graph(sources.data_ptr<tq::RamBlockView>(),mapped_ids,backup.data_ptr<tq::RamBlockView>(),saved.data_ptr<std::uint32_t>(),2,stream)));
        auto current=sources.to(tb::kCPU).contiguous();
        if(std::memcmp(current.data_ptr(),original.data_ptr(),original.nbytes()))throw std::runtime_error("SSD graph did not restore its exact source descriptors");
    };
    images.copy_(original);
    for(int id=0;id<2;++id)images.data_ptr<tq::RamBlockView>()[id].image.key=payload.data_ptr<std::uint8_t>()+64+id*16;
    control[tq::RamGraphReady]=1;control[tq::RamGraphImages]=2;publish();
    auto current=sources.to(tb::kCPU).contiguous();
    if(std::memcmp(current.data_ptr(),images.data_ptr(),images.nbytes()))throw std::runtime_error("SSD graph did not publish its immutable images");
    // Cancellation after publication cannot erase the device-owned rollback count.
    control[tq::RamGraphReady]=2;restore();
    for(int invalid:{0,1,2}) {
        images.copy_(original);control[tq::RamGraphError]=0;
        control[tq::RamGraphReady]=invalid==2?2:1;control[tq::RamGraphImages]=invalid==1?3:2;
        if(!invalid)++images.data_ptr<tq::RamBlockView>()[1].generation;
        selected.copy_(selected_host.to(options));publish();
        TYLOINFER_CUDA_CHECK(cudaError_t(tq::tq_cancel_ram_graph_selection(selected.data_ptr<tq::SelectedBlock>(),2,saved.data_ptr<std::uint32_t>(),stream)));
        restore();auto masks=selected.to(tb::kCPU).contiguous();
        for(int at=0;at<2;++at)if(masks.data_ptr<tq::SelectedBlock>()[at].token_mask)throw std::runtime_error("SSD graph cancellation retained an active selection");
        if(invalid!=2 && !control[tq::RamGraphError])throw std::runtime_error("SSD graph accepted an invalid source generation or count");
    }
    std::cout<<"SSD Graph publication, exact rollback after cancellation, generation/count rejection and safe masks PASS\n";
}
void ram_pressure_retained_graph() {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(1<<20));
    constexpr int prompt=8192,heads=6,kh=2,width=63,iw=4,n=4;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    mfq::cuda::QsaTqCache cache(prompt+64,heads,kh,width,1,iw,64,f,false);
    const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
    const auto prefill=[&] {
        auto values=tb::full({1,prompt,kh,width},.03125,f);
        auto q=tb::zeros({1,heads,prompt,width},h),iq=tb::zeros({1,prompt,1,iw},f),raw=tb::zeros({1,prompt,iw},f);
        auto position=tb::arange(0,prompt,f.dtype(tb::kInt64)).reshape({1,prompt});
        auto gate=tb::zeros({1,prompt,heads,width},h);
        cache.forward(q,values,values,iq,raw,position,pool,false,gate);cache.wait();
    };
    prefill();
    auto q=tb::zeros({1,heads,n,width},h),kv=tb::full({1,n,kh,width},.03125,f);
    auto iq=tb::zeros({1,n,1,iw},f),raw=tb::zeros({1,n,iw},f);
    auto position=tb::zeros({1,n},f.dtype(tb::kInt64)),gate=tb::zeros({1,n,heads,width},h),out=tb::empty({n,heads,width},h);
    mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
    for(int iteration=0;iteration<4;++iteration) {
        if(iteration==3){cache.reset();cache.release_idle();prefill();}
        const auto start=cache.position();
        position.copy_(tb::arange(start,start+n,position.options()).reshape({1,n}));
        if(iteration==1) {
            setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(9<<20));cache.wait();
            const auto usage=cache.stats();
            const auto state=cache.capture_session(start);std::array<std::size_t,4> tiers{};
            for(const auto& block:state->kv->blocks)++tiers[unsigned(block.tier)];
            if(!tiers[0] || !tiers[1] || !tiers[2] || tiers[3] || usage.resident_bytes>9<<20)
                throw std::runtime_error("RAM-only pressure did not retain three encoded tiers without SSD");
        }
        cache.prepare_decode_tokens(n,true);
        if(!window.valid())window.capture([&]{cache.forward_pending_graph(q,kv,kv,iq,raw,position,gate,out,pool);},[]{});
        window.run();cache.advance_graph(n);
        const auto actual=out.to(tb::kCPU,tb::kFloat32).contiguous();
        for(int64_t i=0;i<actual.numel();++i)if(std::abs(actual.data_ptr<float>()[i]-.015625f)>.001f)
            throw std::runtime_error("Retained RAM Graph lost its dynamic compact layout after quantization");
    }
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    std::cout<<"RAM-only16/8/4 pressure, no SSD, retained M4 Graph across slot shrink and idle metadata release PASS\n";
}

void restore_reservation_and_idle_release() {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    constexpr int count=1024;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    const auto make_state=[](int rows) {
        tq::RamKvSnapshot result;
        const auto bytes=((2*rows+(rows+1)/2+(3*rows+7)/8+3)&~std::size_t(3))+std::size_t(rows)*128;
        for(int id=0;id<count;++id) {
            auto data=std::make_shared<std::vector<std::uint8_t>>(2*bytes,std::uint8_t(id));
            auto image=std::make_shared<tq::RamKvImage>();image->owner=data;
            image->key=data->data();image->value=data->data()+bytes;image->key_bytes=image->value_bytes=bytes;
            image->key_bits=image->value_bits=4;
            result.blocks.push_back({image,{},{2*bytes,2*bytes,2*bytes},std::uint64_t(id+1),{},tq::RamKvTier::Tq4});
        }
        return result;
    };
    auto first_state=make_state(32),second_state=make_state(64);
    const auto bytes=[](const auto& state){return state.blocks.size()*state.blocks.front().image->bytes();};
    const auto total=bytes(first_state)+bytes(second_state);
    auto first_live=tb::zeros({count,int64_t(sizeof(tq::TqBlockView))},options),second_live=tb::zeros_like(first_live);
    const auto unexpected_codec=[](const auto&,const auto&)->std::vector<tq::MicroblockImage>{
        throw std::runtime_error("Unpressured restore unexpectedly invoked the codec");
    };
    mfq::cuda::QsaRamCache first(count,32,16,4,options,first_live,false,"",unexpected_codec);
    mfq::cuda::QsaRamCache second(count,64,16,4,options,second_live,false,"",unexpected_codec);
    auto idle_live=tb::zeros_like(first_live);
    mfq::cuda::QsaRamCache unused(count,32,16,4,options,idle_live,false,"",unexpected_codec);
    // Enough for the two active layers. Dividing by an unused third cache
    // would incorrectly quantize/spill the larger layer during restore.
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES",std::to_string(2*bytes(second_state)+(1<<20)));
    const auto stream=mfq_current_cuda_stream();const auto device=mfq_current_cuda_device();
    std::vector<std::size_t> reservations;
    auto registration=mfq::cuda::register_qsa_ram_budget_reserver(device,stream,[&](auto value){reservations.push_back(value);return 0;});
    {
        auto a=first.reserve_restore(first_state),b=second.reserve_restore(second_state);
        first.restore(first_state,false);
        if(reservations.empty() || reservations.front()<tq::ram_kv_reclaim_bytes(0,0,total))
            throw std::runtime_error("Prefix restore reserved only the first layer instead of all incoming geometry");
        second.restore(second_state,false);
        if(first.memory().payload!=bytes(first_state) || second.memory().payload!=bytes(second_state))
            throw std::runtime_error("Unpressured multi-layer restore dropped incoming KV");
    }
    first.truncate(0);second.truncate(0);
    // Production model construction and request execution use different host
    // threads. Cleanup must follow the cache owner, not thread-local lookup.
    const auto model_stream=mfq_get_current_cuda_stream();std::exception_ptr error;
    std::thread cleanup([&]{try {
        MfqCudaGuard guard(device);MfqCudaStreamGuard stream_guard(model_stream);first.release_idle();second.release_idle();
    }catch(...){error=std::current_exception();}});
    cleanup.join();if(error)std::rethrow_exception(error);
    if(first.memory().payload || second.memory().payload || first.memory().allocated>total/2+(1<<20))
        throw std::runtime_error("Idle reset retained completed KV image readers");
    if(reservations.empty() || reservations.back())throw std::runtime_error("Reset retained an expert RAM reservation");
    try {
        auto pending=first.reserve_restore(first_state);
        throw std::runtime_error("Expected interrupted restore");
    }catch(const std::runtime_error&){}
    if(reservations.back())throw std::runtime_error("Interrupted restore retained its incoming RAM reservation");
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    std::cout<<"All-layer prefix reservation with different geometry, unused cache excluded, interrupted restore and cross-thread idle KV release PASS\n";
}

void fragmented_ram_capacity(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_get_stream_from_pool();const MfqCudaStreamGuard guard(stream);
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");setting("TYLOINFER_KV_RAM_LIMIT_BYTES",std::to_string(64u<<20));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    constexpr int rows=32,blocks=2048,count=16;
    const std::size_t part=((2*rows+(rows+1)/2+(3*rows+7)/8+3)&~std::size_t(3))+std::size_t(rows)*128;
    tq::RamKvSnapshot state;state.blocks.resize(blocks);
    for(int id=0;id<blocks;++id) {
        auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=part;
        backing->key_bits=backing->value_bits=4;backing->tier=tq::RamKvTier::Tq4;
        qsa_test::reader(*backing).read_range=[id,part](auto,auto){return std::vector<std::uint8_t>(part*2,std::uint8_t(id));};
        auto& block=state.blocks[id];block.backing=backing;block.bytes.fill(part*2);block.access=id+1;block.tier=tq::RamKvTier::Tq4;
    }
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto live=tb::zeros({blocks,int64_t(sizeof(tq::TqBlockView))},options);
    const auto fixture=[](const auto& input,const auto&){return input;};
    mfq::cuda::QsaRamCache cache(blocks,rows,count,4,options,live,true,directory,fixture);
    cache.restore(state,true);cache.wait();const auto before=cache.memory();
    std::vector<const std::uint8_t*> old_keys(blocks);std::vector<std::uint64_t> generations(blocks);
    {const auto snapshot=cache.snapshot(blocks);for(int id=0;id<blocks;++id){
        old_keys[id]=snapshot->blocks[id].image->key;generations[id]=snapshot->blocks[id].generation;}}
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES",std::to_string(before.allocated*3/4+(1u<<20)));
    std::vector<double> hits(blocks);for(int id=0;id<blocks;id+=2)hits[id]=1;
    cache.observe(hits,10000);cache.wait();const auto after=cache.memory();
    if(after.allocated>=before.allocated*9/10 || after.payload<blocks/2*part*2)
        throw std::runtime_error("Fragmented pinned RAM did not release physical slabs while retaining hot KV: before="+
            std::to_string(before.allocated)+" after="+std::to_string(after.allocated)+" payload="+std::to_string(after.payload));
    std::size_t moved=0;
    {const auto snapshot=cache.snapshot(blocks);for(int id=0;id<blocks;id+=2){
        const auto& block=snapshot->blocks[id];
        if(!block.image || block.generation!=generations[id] || block.image->key[0]!=std::uint8_t(id))
            throw std::runtime_error("RAM compaction changed a hot image or its generation");
        moved+=block.image->key!=old_keys[id];}}
    if(!moved)throw std::runtime_error("Pinned fragmentation regression did not exercise relocation");
    cache.prepare();std::vector<tq::SelectedBlock> selection(count);
    for(int i=0;i<count;++i)selection[i]={i*2,15};
    auto selected=tb::empty({count,int64_t(sizeof(tq::SelectedBlock))},options);
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(selected.data_ptr(),selection.data(),selection.size()*sizeof(tq::SelectedBlock),cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
    auto staged=cache.stage(selected,1,count,live,{},blocks*4,0,true);
    std::vector<tq::TqBlockView> views(count);
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(views.data(),staged.data_ptr(),views.size()*sizeof(tq::TqBlockView),cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    for(int i=0;i<count;++i) {
        std::vector<std::uint8_t> bytes(part);
        TYLOINFER_CUDA_CHECK(cudaMemcpy(bytes.data(),views[i].key,bytes.size(),cudaMemcpyDeviceToHost));
        if(!std::all_of(bytes.begin(),bytes.end(),[&](auto b){return b==std::uint8_t(i*2);}))
            throw std::runtime_error("GPU source descriptors retained a relocated RAM address");
    }
    cache.truncate(0);cache.release_idle();setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    std::cout<<"Fragmented pinned KV: physical slab release, hot LRU/version preservation and GPU address publication PASS\n";
}
void ssd_kv_preserves_expert_ram(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES",std::to_string(1<<20));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    constexpr int rows=32,blocks=2048;
    const auto part=((2*rows+(rows+1)/2+(3*rows+7)/8+3)&~std::size_t(3))+std::size_t(rows)*128;
    tq::RamKvSnapshot state;state.blocks.resize(blocks);
    for(int id=0;id<blocks;++id) {
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->key_bytes=backing->value_bytes=part;
        backing->key_bits=backing->value_bits=4;backing->tier=tq::RamKvTier::Tq4;
        qsa_test::reader(*backing).read_range=[id,part](std::size_t,std::size_t){return std::vector<std::uint8_t>(2*part,std::uint8_t(id));};
        auto& block=state.blocks[id];block.backing=std::move(backing);
        block.bytes={2*part,2*part,2*part};block.access=id+1;block.tier=tq::RamKvTier::Tq4;
    }
    const auto device=mfq_current_cuda_device();const auto stream=mfq_current_cuda_stream();
    std::vector<std::size_t> reservations;std::size_t expert_reclaims=0;
    auto reservation=mfq::cuda::register_qsa_ram_budget_reserver(device,stream,[&](auto n){reservations.push_back(n);return 0;});
    auto reclaimer=mfq::cuda::register_qsa_ram_capacity_reclaimer(device,stream,[&](auto n){expert_reclaims+=n;return 0;});
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto live=tb::zeros({blocks+128,int64_t(sizeof(tq::TqBlockView))},options);
    // append() accepts canonical inputs through its encoder. These fixture
    // bytes are already TQ4; real encoding is covered by the numeric tests.
    const auto encoded_fixture=[](const auto& images,const auto& precision) {
        for(const auto& image:images)if(image.key_bits!=4 || image.value_bits!=4)
            throw std::runtime_error("TQ4 placement unexpectedly changed encoding");
        for(const auto& p:precision)if(p.key_bits!=4 || p.value_bits!=4)
            throw std::runtime_error("TQ4 placement requested another precision");
        return images;
    };
    mfq::cuda::QsaRamCache cache(blocks+128,rows,16,4,options,live,true,directory,encoded_fixture);
    cache.restore(state,true);cache.wait();
    const auto usage=cache.memory();
    if(!usage.tokens[3] || usage.payload>1<<20 || usage.disk)
        throw std::runtime_error("SSD prefix did not preserve cold backing within spare RAM");
    const auto image=cache.reconstruction_image(7);
    if(image.key!=std::vector<std::uint8_t>(part,7) || image.value!=image.key)
        throw std::runtime_error("Cold KV placement changed its original bytes");
    tq::MicroblockImage appended;appended.key.resize(part,19);appended.value=appended.key;
    appended.key_bits=appended.value_bits=4;
    cache.append(std::vector<tq::MicroblockImage>(128,appended),(blocks+128)*4,false);cache.wait();
    if(expert_reclaims || std::any_of(reservations.begin(),reservations.end(),[](auto n){return n!=0;}))
        throw std::runtime_error("SSD-backed KV restore/growth displaced RAM experts");
    // Sharing a worker with a RAM-only layer must reserve only that layer's
    // target, even while the SSD-backed layer has a much larger context.
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES",std::to_string(32<<20));
    auto ram_state=state;ram_state.blocks.resize(64);
    auto ram_live=tb::zeros({64,int64_t(sizeof(tq::TqBlockView))},options);
    {
        mfq::cuda::QsaRamCache ram(64,rows,16,4,options,ram_live,false,"",encoded_fixture);
        ram.restore(ram_state,false);ram.wait();
        // Existing shared pinned pages can cover this small RAM-only target
        // without requesting another expert reservation.
        if(ram.memory().payload!=64*2*part || std::any_of(reservations.begin(),reservations.end(),[](auto n){return n>4<<20;}))
            throw std::runtime_error("Mixed RAM/SSD layers reserved expert RAM for spillable KV");
        ram.truncate(0);ram.release_idle();
    }
    cache.truncate(0);cache.release_idle();
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    std::cout<<"SSD KV restore/growth retains RAM experts, exact cold read and mixed RAM-only reservation PASS\n";
}

void shared_stage_admission_lifetime(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    constexpr int rows=19,blocks=4;const auto part=((2*rows+(rows+1)/2+(3*rows+7)/8+3)&~std::size_t(3))+std::size_t(rows)*128;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    auto first_live=tb::zeros({blocks,int64_t(sizeof(tq::TqBlockView))},options),second_live=tb::zeros_like(first_live);
    const auto codec=[](const auto&,const auto&)->std::vector<tq::MicroblockImage>{throw std::runtime_error("Unexpected codec in shared-stage lifetime test");};
    mfq::cuda::QsaRamCache first(blocks,rows,blocks,16,options,first_live,true,directory+"/first",codec);
    mfq::cuda::QsaRamCache second(blocks,rows,blocks,16,options,second_live,true,directory+"/second",codec);
    auto batch_calls=std::make_shared<std::size_t>(0);
    const auto state=[&](std::uint8_t byte) {
        tq::RamKvSnapshot result;
        auto source=std::make_shared<tq::RamKvReadSource>();source->bytes=blocks*2*part;
        source->read_range=[=](std::size_t,std::size_t n){return std::vector<std::uint8_t>(n,byte);};
        source->read_range_into=[=](std::size_t,std::uint8_t* output,std::size_t n){if(n!=2*part)throw std::runtime_error("Changed shared-stage extent");std::memset(output,byte,n);};
        source->read_ranges_into=[=](const std::vector<mfq::ReadSpan>& spans){
            ++*batch_calls;for(const auto& span:spans){if(span.size!=2*part)throw std::runtime_error("Changed shared-stage batch extent");std::memset(span.destination,byte,span.size);}
        };
        for(int id=0;id<blocks;++id) {
            auto backing=std::make_shared<tq::RamKvBacking>();backing->key_bytes=backing->value_bytes=part;
            backing->key_bits=backing->value_bits=4;backing->tier=tq::RamKvTier::Tq4;
            backing->source=source;backing->offset=id*2*part;
            const auto affine=((3*rows+3)&~3)+std::size_t(rows)*256;
            result.blocks.push_back({{},backing,{2*std::size_t(rows)*512,2*affine,2*part},0,std::uint64_t(id),tq::RamKvTier::Ssd});
        }
        return result;
    };
    first.restore(state(43),true);second.restore(state(117),true);
    auto host=tb::zeros({blocks,int64_t(sizeof(tq::SelectedBlock))},options.device(tb::kCPU));
    for(int id=0;id<blocks;++id)host.data_ptr<tq::SelectedBlock>()[id]={id,15};
    auto selected=host.to(options);first.prepare();second.prepare();first.prepare_graph_task(100);second.prepare_graph_task(100);
    mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{first.stage_graph_task(selected,1,blocks,first_live,{},0);second.stage_graph_task(selected,1,blocks,second_live,{},0);},[]{});
    if(graph.segments()!=1)throw std::runtime_error("Shared-stage test added a graph boundary");
    for(int replay=0;replay<2;++replay) {
        first.restore(state(43),true);second.restore(state(117),true);
        first.prepare_graph_task(100);second.prepare_graph_task(100);
        const auto before=*batch_calls;
        graph.run();
        if(*batch_calls<=before)throw std::runtime_error("Retained SSD Graph skipped random source batch dispatch");
        for(auto pair:{std::pair{&first,std::uint8_t(43)},std::pair{&second,std::uint8_t(117)}}) {
            auto snapshot=pair.first->snapshot(blocks);
            for(const auto& block:snapshot->blocks) {
                if(!block.image)throw std::runtime_error("Shared-stage test did not admit its fallback data");
                const auto image=block.read();
                if(image.key!=std::vector<std::uint8_t>(part,pair.second) || image.value!=image.key)
                    throw std::runtime_error("Later layer overwrote deferred RAM KV admission");
            }
        }
    }
    if(!*batch_calls)throw std::runtime_error("SSD Graph did not dispatch random source batches");
    std::cout<<"Two QSA layers sharing mapped staging: batch dispatch, fallback lifetime, deferred admission and retained Graph replay PASS\n";
}

void mixed_prefix_graph(const std::string& directory) {
    namespace tq=tyloinfer::qsa;
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    setting("TYLOINFER_KV_RAM_LIMIT_BYTES","");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    constexpr int prompt=128,heads=6,kh=2,width=63,iw=7;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
    mfq::cuda::QsaTqCache source(256,heads,kh,width,1,iw,256,f,false);
    std::vector<float> values(prompt*kh*width),queries(prompt*heads*width),raw(prompt*iw);
    for(std::size_t i=0;i<values.size();++i)values[i]=.13f*std::sin(i*.031f)+.02f*std::cos(i*.007f);
    for(std::size_t i=0;i<queries.size();++i)queries[i]=.09f*std::cos(i*.013f);
    for(std::size_t i=0;i<raw.size();++i)raw[i]=.07f*std::sin(i*.043f);
    auto kv=tb::tensor(values,f).reshape({1,prompt,kh,width});
    auto iq=tb::zeros({1,prompt,1,iw},f),position=tb::arange(prompt,f.dtype(tb::kInt64)).reshape({1,prompt});
    source.forward(tb::tensor(queries,f).reshape({1,prompt,heads,width}).permute({0,2,1,3}),kv,kv,
        iq,tb::tensor(raw,f).reshape({1,prompt,iw}),position,pool,false,tb::zeros({1,prompt,heads,width},h));
    source.wait();auto state=source.capture_session(prompt);source.reset();source.release_idle();
    auto ram_state=*state;ram_state.kv=std::make_shared<tq::RamKvSnapshot>(*state->kv);
    auto ssd_state=ram_state;ssd_state.kv=std::make_shared<tq::RamKvSnapshot>(*state->kv);
    std::array<unsigned,3> reads{};
    for(std::size_t id=0;id<state->kv->blocks.size();++id) {
        const unsigned tier=id%3;const auto input=state->kv->blocks[id].read();auto encoded=input;
        if(tier)encoded=tq::CpuTqBlockFitter(input,int(input.key.size()/512),1).image(tier==1?tq::Precision{8,8}:tq::Precision{4,4});
        auto bytes=std::make_shared<std::vector<std::uint8_t>>(encoded.key);
        bytes->insert(bytes->end(),encoded.value.begin(),encoded.value.end());
        auto image=std::make_shared<tq::RamKvImage>();image->owner=bytes;
        image->key=bytes->data();image->value=bytes->data()+encoded.key.size();
        image->key_bytes=encoded.key.size();image->value_bytes=encoded.value.size();
        image->key_bits=encoded.key_bits;image->value_bits=encoded.value_bits;
        image->key_state_bits=encoded.key_state_bits;image->value_state_bits=encoded.value_state_bits;
        auto& memory=ram_state.kv->blocks[id];memory.image=image;memory.backing.reset();memory.tier=tq::RamKvTier(tier);
        auto backing=std::make_shared<tq::RamKvBacking>();
        backing->key_bytes=image->key_bytes;backing->value_bytes=image->value_bytes;
        backing->key_bits=image->key_bits;backing->value_bits=image->value_bits;
        backing->key_state_bits=image->key_state_bits;backing->value_state_bits=image->value_state_bits;
        backing->tier=tq::RamKvTier(tier);qsa_test::reader(*backing).read_range=[bytes,tier,&reads](std::size_t,std::size_t){++reads[tier];return *bytes;};
        if(id%2)qsa_test::reader(*backing).read_range_into=[bytes,tier,&reads](std::size_t,std::uint8_t* output,std::size_t count) {
            if(count!=bytes->size())throw std::runtime_error("Direct mixed prefix size changed");
            ++reads[tier];std::memcpy(output,bytes->data(),count);
        };
        auto& disk=ssd_state.kv->blocks[id];disk.image.reset();disk.backing=std::move(backing);disk.tier=tq::RamKvTier::Ssd;
    }
    for(std::size_t cache_bytes:{std::size_t(0),std::size_t(1)<<20}) {
        setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES",std::to_string(cache_bytes));
        setting("TYLOINFER_QSA_TQ_DIRECTORY",directory+"/"+std::to_string(cache_bytes));
        mfq::cuda::QsaTqCache actual(256,heads,kh,width,1,iw,256,f,true),reference(256,heads,kh,width,1,iw,256,f,false);
        for(int n=1;n<=8;++n) {
            actual.reset();actual.restore_session(ssd_state,true);reference.reset();reference.restore_session(ram_state,false);
            auto keys=tb::full({1,n,kh,width},.042,h),q=tb::full({1,heads,n,width},.061,h);
            auto raw_next=tb::full({1,n,iw},.033,f),index_query=tb::full({1,n,1,iw},-.023,f);
            auto next=tb::arange(prompt,prompt+n,f.dtype(tb::kInt64)).reshape({1,n});
            auto gate=tb::full({1,n,heads,width},.31,h),out=tb::empty({n,heads,width},h);
            const auto expected=reference.forward(q,keys,keys,index_query,raw_next,next,pool,false,gate).to(tb::kCPU,tb::kFloat32).contiguous();
            if(!actual.graph_task_supported())throw std::runtime_error("Original-precision prefix disabled SSD Graph reads");
            actual.prepare_decode_tokens(n,true,8);mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{actual.forward_pending_graph(q,keys,keys,index_query,raw_next,next,gate,out,pool);},[]{});
            if(window.segments()!=1)throw std::runtime_error("Mixed prefix added a CUDA Graph boundary");
            window.run();actual.advance_graph(n);
            const auto result=out.to(tb::kCPU,tb::kFloat32).contiguous();
            double error=0,energy=0,maximum=0;
            for(int64_t i=0;i<result.numel();++i) {
                const double a=result.data_ptr<float>()[i],b=expected.data_ptr<float>()[i],d=a-b;
                error+=d*d;energy+=b*b;maximum=std::max(maximum,std::abs(d));
            }
            if(maximum>2e-4 || !std::isfinite(error))
                throw std::runtime_error("Mixed prefix Graph/RAM mismatch M="+std::to_string(n)+" cache="+std::to_string(cache_bytes)+
                    " max="+std::to_string(maximum)+" NMSE="+std::to_string(error/std::max(energy,1e-30)));
            actual.wait();reference.wait();
        }
    }
    for(auto n:reads)if(!n)throw std::runtime_error("Mixed prefix test did not recall every encoding");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    std::cout<<"Original FP16/Affine8/TQ4 prefix M1..8, single SSD Graph submission, exact RAM reference and VRAM LRU on/off PASS\n";
}

void graph_capability_is_read_only(const std::string& directory) {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");setting("TYLOINFER_QSA_TQ_DIRECTORY",directory);
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    mfq::cuda::QsaTqCache cache(64,2,1,64,1,8,8,f,true);
    auto q=tb::zeros({1,2,8,64},f),kv=tb::full({1,8,1,64},.03125,f);
    auto iq=tb::zeros({1,8,1,8},f),raw=tb::zeros({1,8,8},f),gate=tb::zeros({1,8,2,64},f);
    auto positions=tb::arange(8,f.dtype(tb::kInt64)).reshape({1,8});
    const auto pool=[](const auto& x,const auto&){return x.slice(1,0,x.size(1),4).contiguous();};
    cache.forward(q,kv,kv,iq,raw,positions,pool,false,gate);cache.wait();
    cache.prepare_decode_tokens(1,true,8);
    const auto before=cache.stats();
    struct RestoreBudget {~RestoreBudget(){setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");}} restore;
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","1");
    tb::Tensor result;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
    window.capture([&] {
        if(!cache.graph_task_supported())throw std::runtime_error("Graph capability changed with the RAM budget");
        result=tb::ones({1},f)+1;
    },[]{});
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");window.run();
    const auto after=cache.stats();
    if(before.resident_bytes!=after.resident_bytes || before.demotions!=after.demotions || before.promotions!=after.promotions ||
            before.canonical_read_bytes!=after.canonical_read_bytes || cache.position()!=8 || result.cpu().data_ptr<float>()[0]!=2)
        throw std::runtime_error("Graph capability query changed cached KV");
    std::cout<<"Graph capability query under changing RAM budget and CUDA capture performs no fitting, reclamation or KV mutation PASS\n";
}

void prefill_loan_lifetime() {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();
    auto cache=std::make_unique<mfq::cuda::QsaTqCache>(128,6,2,64,3,13,128,f,false);
    const auto append=[&](int start,int count) {
        auto q=tb::zeros({1,6,count,64},h),v=tb::full({1,count,2,64},.03125,f);
        auto raw=tb::zeros({1,count,13},f),iq=tb::zeros({1,count,3,13},f),gate=tb::zeros({1,count,6,64},h);
        auto positions=tb::arange(start,start+count,f.dtype(tb::kInt64)).reshape({1,count});
        auto out=cache->forward(q,v,v,iq,raw,positions,[&](const auto& x,const auto&){return tb::zeros({1,x.size(1)/4,13},f);},false,gate);
        const auto actual=out.to(tb::kCPU,tb::kFloat32).contiguous();
        for(int64_t i=0;i<actual.numel();++i)if(std::abs(actual.data_ptr<float>()[i]-.015625f)>.00001f)
            throw std::runtime_error("QSA workspace loan changed causal attention");
    };
    append(0,4);cache->wait();
    auto storage=tb::empty({8<<20},f.dtype(tb::kUInt8));
    context->begin_workspace_loan({{storage.data_ptr(),storage.nbytes()}},stream);
    try {
        append(4,33);TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));cache->collect();
        context->suspend_workspace_loan();context->end_workspace_loan();
    }catch(...) {
        context->suspend_workspace_loan();cache.reset();
        try{context->end_workspace_loan();}catch(...){}
        throw;
    }
    append(37,7);cache->wait();
    std::cout<<"QSA growing prefill workspace survives expert-loan release and later continuation PASS\n";
}
void long_prefill_operand_reuse() {
    setting("TYLOINFER_QSA_RAM_KV_BITS","16");setting("TYLOINFER_QSA_RAM_BUDGET_BYTES",std::to_string(128ULL<<20));
    setting("TYLOINFER_QSA_RAM_VRAM_CACHE_BYTES","0");setting("TYLOINFER_QSA_PREFILL_SCORE_BYTES",std::to_string(512ULL<<10));
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    constexpr int history=17000,queries=128,heads=8,kh=2,width=256,iw=13;
    mfq::cuda::QsaTqCache cache(history+8192,heads,kh,width,3,iw,2048,f,false);
    auto host=tb::empty({1,history+queries,kh,width},f.device(tb::kCPU));
    for(int i=0;i<host.numel();++i)host.data_ptr<float>()[i]=i%width<2?.03125f:.04f*std::sin(float(i)*.013f);
    auto values=host.to(tb::kCUDA);
    const auto pool=[&](const auto& x,const auto&){return tb::zeros({1,x.size(1)/4,iw},f);};
    {
        auto q=tb::zeros({1,heads,history,width},h),k=tb::zeros({1,history,kh,width},f);
        auto iq=tb::zeros({1,history,3,iw},f),raw=tb::zeros({1,history,iw},f);
        auto positions=tb::arange(history,f.dtype(tb::kInt64)).reshape({1,history});
        (void)cache.forward(q,k,values.narrow(1,0,history),iq,raw,positions,pool,true);cache.wait();
    }
    {
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        const auto limit=context->memory_stats().limit;
        context->set_memory_limit(std::max(context->resident_memory_usage(),context->memory_stats().allocated)+(512ULL<<20));
        const auto available=context->available_memory()+context->reusable_workspace_bytes();
        const auto rows=cache.prefill_rows(8192);context->set_memory_limit(limit);
        if(rows!=8192)throw std::runtime_error("Tiled QSA scores unnecessarily split the outer PP/GEMM batch: rows="+
            std::to_string(rows)+" available="+std::to_string(available));
    }
    auto q=tb::zeros({1,heads,queries,width},h),k=tb::zeros({1,queries,kh,width},f);
    auto v=values.narrow(1,history,queries),iq=tb::zeros({1,queries,3,iw},f),raw=tb::zeros({1,queries,iw},f);
    auto positions=tb::arange(history,history+queries,f.dtype(tb::kInt64)).reshape({1,queries});
    auto gate=tb::zeros({1,queries,heads,width},h);
    std::array<tb::Tensor,2> results;std::array<double,2> elapsed{};
    for(int staged=0;staged<2;++staged) {
        setting("TYLOINFER_QSA_PREFILL_STAGING",std::to_string(staged));
        const auto started=std::chrono::steady_clock::now();
        results[staged]=cache.forward(q,k,v,iq,raw,positions,pool,false,gate).to(tb::kCPU,tb::kFloat32).contiguous();
        elapsed[staged]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
        cache.wait();if(!staged)cache.truncate(history);
    }
    double maximum=0;
    for(int64_t i=0;i<results[0].numel();++i) {
        const double difference=std::abs(results[0].data_ptr<float>()[i]-results[1].data_ptr<float>()[i]);
        if(!std::isfinite(difference))throw std::runtime_error("Long QSA prefill produced nonfinite output");
        maximum=std::max(maximum,difference);
        if(i%width<2 && std::abs(results[1].data_ptr<float>()[i]-.015625f)>2e-5f)
            throw std::runtime_error("Long staged QSA changed constant-value causal attention");
    }
    if(maximum>2e-4)throw std::runtime_error("Long staged QSA differs from direct mixed-precision attention");
    setting("TYLOINFER_QSA_PREFILL_STAGING","1");setting("TYLOINFER_QSA_PREFILL_SCORE_BYTES",std::to_string(64ULL<<20));
    setting("TYLOINFER_QSA_RAM_BUDGET_BYTES","0");
    std::cout<<"Long QSA prefill beyond32MiB, multiple score tiles and direct-reference parity PASS max="<<maximum
        <<" direct_ms="<<elapsed[0]<<" staged_ms="<<elapsed[1]<<'\n';
}
int main(int argc,char** argv)try {
    reclaimer_lifetime();
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::cout<<"SKIP: CUDA unavailable\n";return 77;}
    restore_reservation_and_idle_release();
    ssd_kv_preserves_expert_ram((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"expert-priority").string());
    fragmented_ram_capacity((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"fragmented-capacity").string());
    prefill_loan_lifetime();
    long_prefill_operand_reuse();
    prefill_layer_bytes((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"prefill-layer").string());
    prefill_pressure_recheck((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"prefill-pressure").string());
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(mfq_current_cuda_stream());
    graph_capability_is_read_only((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"capability").string());
    shared_stage_admission_lifetime((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"shared-stage").string());
    mixed_prefix_graph((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"mixed-prefix").string());
    graph_source_transaction();
    for(int bits:{16,8,4})for(std::size_t bytes:{std::size_t(0),std::size_t(1)<<20}) {
        run(bits,6,63,bytes);run(bits,24,256,bytes);run(bits,12,257,bytes);
    }
    for(int bits:{16,8,4})snapshot_roundtrip(bits);
    ram_pressure_retained_graph();
    pressure(argc>1?argv[1]:(std::filesystem::current_path()/"qsa-ram-runtime-ssd-test-data").string());
    combined_ram_caps((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"combined-caps").string());
    failed_ssd_copy((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"failure").string());
    packed_spill_cache((std::filesystem::path(argc>1?argv[1]:"qsa-ram-runtime-ssd-test-data")/"packed-spill").string());
    std::cout<<"QSA RAM runtime integration checks PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}
