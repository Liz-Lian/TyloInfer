#include <limits>
#include "runtime/qsa_tq_cache.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include "kernels/qsa_tq.h"
#include "tyloinfer/kernels/cuda/qwen4_exp.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>

namespace tb=mfq_tensor_backend;
void check_attention_partitions() {
    namespace tq=tyloinfer::qsa;
    int cases=0;
    for(int width:{256,512,1024})for(int ratio:{2,4,6,8,12,24})for(int queries:{1,2,3,4,8,9,64})for(int count:{1,3,7,65,129,513}) {
        const int heads=3*ratio;
        const int exact=tq::tq_attention_partitions(queries,heads,3,width,count,false,false);
        if(exact!=std::min(queries>8?4:64,count))throw std::runtime_error("FP32 partition fallback changed");
        const int single=tq::tq_attention_partitions(1,heads,3,width,count,true,false);
        for(bool snapshots:{false,true}) {
            const int half=tq::tq_attention_partitions(queries,heads,3,width,count,true,false,snapshots);
            const int allowed=tq::tq_attention_partitions(queries,heads,3,width,count,false,true,snapshots);
            if(half!=allowed || half<=0 || half>std::min(count,128))throw std::runtime_error("QSA partition bounds/math mismatch");
            if(queries>8 && half!=std::min(4,count))throw std::runtime_error("Prefill partitions changed");
            if(!snapshots && queries<=8 && half>single)throw std::runtime_error("More queries increased partition storage");
            ++cases;
        }
    }
    if(tq::tq_attention_partitions(0,6,2,256,513,true,false)!=0 ||
       tq::tq_attention_partitions(1,5,2,256,513,true,false)!=0 ||
       tq::tq_attention_partitions(1,6,2,255,513,true,false)!=0)
        throw std::runtime_error("Invalid QSA partition geometry accepted");
    std::cerr<<"adaptive QSA resource partition bounds, FP32 fallback, all head widths and query counts PASS cases="<<cases<<'\n';
}

void check_mass_observation() {
    namespace tq=tyloinfer::qsa;
    constexpr int queries=2,count=37,blocks=2*count,width=17,padded=256,parts=4;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    const auto bytes=f.dtype(tb::kUInt8);
    const std::vector<int64_t> positions{9,111};
    auto pos=tb::tensor(positions,f.dtype(tb::kInt64));
    for(int heads:{1,3,10,24}) {
        const int kv_heads=heads==10?2:1;
        auto payload=tb::zeros({4*kv_heads*padded},f.dtype(tb::kFloat16));
        std::vector<tq::TqBlockView> views(blocks);
        for(auto& view:views) {
            view.key=view.value=static_cast<const uint8_t*>(payload.data_ptr());
            view.key_bits=view.value_bits=0;
        }
        std::vector<tq::SelectedBlock> selected(queries*count);
        std::vector<float> expected_hits(blocks),expected_mass(blocks);
        for(int q=0;q<queries;++q) {
            int denominator=0;
            for(int slot=0;slot<count;++slot) {
                auto& b=selected[q*count+slot];
                b.id=slot%7==0?-1:2*slot;b.token_mask=slot%5==0?5u:15u;
                if(slot==count-1){b.id=2;b.token_mask=15;}
                for(int t=0;t<4;++t)if(b.id>=0 && (b.token_mask&(1u<<t)) && b.id*4+t<=positions[q])++denominator;
            }
            for(int slot=0;slot<count;++slot) {
                const auto b=selected[q*count+slot];int visible=0;
                for(int t=0;t<4;++t)if(b.id>=0 && (b.token_mask&(1u<<t)) && b.id*4+t<=positions[q])++visible;
                if(visible){expected_hits[b.id]+=1;expected_mass[b.id]+=float(visible)/denominator;}
            }
        }
        const auto upload=[&](const auto& values) {
            auto host=tb::empty({int64_t(values.size()*sizeof(values[0]))},bytes.device(tb::kCPU));
            std::memcpy(host.data_ptr(),values.data(),size_t(host.numel()));return host.to(tb::kCUDA);
        };
        auto table=upload(views),selection=upload(selected);
        auto query=tb::zeros({queries,heads,width},f),signs=tb::ones({padded},f),relative=tb::zeros({65536},f.dtype(tb::kFloat16));
        auto output=tb::empty_like(query),rotated=tb::empty({queries*heads*padded},f);
        auto partial=tb::empty({queries*heads*parts*padded},f),stats=tb::empty({queries*heads*parts*2},f);
        auto block_mass=tb::empty({queries*heads*count},f),norm=tb::empty({queries*heads},f);
        auto hits=tb::zeros({blocks},f),mass=tb::zeros({blocks},f);
        const int status=tq::tq_sparse_attention(query.data_ptr(),signs.data_ptr<float>(),signs.data_ptr<float>(),
            static_cast<const uint16_t*>(relative.data_ptr()),static_cast<const tq::TqBlockView*>(table.data_ptr()),
            static_cast<const tq::SelectedBlock*>(selection.data_ptr()),pos.data_ptr<int64_t>(),output.data_ptr(),
            rotated.data_ptr<float>(),partial.data_ptr<float>(),stats.data_ptr<float>(),block_mass.data_ptr<float>(),norm.data_ptr<float>(),
            hits.data_ptr<float>(),mass.data_ptr<float>(),queries,heads,kv_heads,width,padded,count,parts,heads*width,width,mfq_current_cuda_stream());
        if(status)throw std::runtime_error(cudaGetErrorString(cudaError_t(status)));
        auto actual_hits=hits.cpu(),actual_mass=mass.cpu();
        for(int b=0;b<blocks;++b)
            if(actual_hits.data_ptr<float>()[b]!=expected_hits[b] || std::abs(actual_mass.data_ptr<float>()[b]-expected_mass[b])>2e-6)
                throw std::runtime_error("QSA history mass differs from uniform causal reference");
    }
    std::cerr<<"adaptive QSA history head reduction, partial masks, invalid/future and duplicate blocks PASS\n";
}
void check_selected_blocks() {
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);int cases=0;
    using Pair=std::pair<int,uint32_t>;
    const auto test=[&](int pools,int pool,int budget,const std::vector<int64_t>& position,int pattern) {
        const int count=std::min(pools,budget/pool),columns=budget+pool-1,slots=(columns+pool-1)/pool,rows=int(position.size());
        std::vector<float> input(rows*pools);
        for(int i=0;i<int(input.size());++i)input[i]=pattern==0?std::sin(float(i)*.31f):pattern==1?float((i*17)%7):
            pattern==2?(i&1?-0.f:0.f):i%7==0?std::numeric_limits<float>::infinity():i%11==0?-std::numeric_limits<float>::infinity():float(i%3);
        auto scores=tb::tensor(input,options).reshape({1,rows,pools});
        auto positions=tb::tensor(position,options.dtype(tb::kInt64));
        auto old=mfq_qwen4_exp::live_topk_selected_tokens(scores,positions,pool,count,budget).cpu().contiguous();
        auto blocks=mfq_qwen4_exp::live_topk_selected_blocks(scores,positions,pool,count,budget).cpu().contiguous();
        for(int row=0;row<rows;++row) {
            std::vector<Pair> expected,actual;
            for(int slot=0;slot<slots;++slot) {
                int id=-1;uint32_t mask=0;
                for(int t=0;t<pool && slot*pool+t<columns;++t) {
                    const int token=old.data_ptr<int32_t>()[row*columns+slot*pool+t];
                    if(token>=0){id=token/pool;mask|=1u<<(token%pool);}
                }
                if(id>=0 && mask)expected.emplace_back(id,mask);
                const auto* x=blocks.data_ptr<int32_t>()+(row*slots+slot)*2;
                if(x[0]>=0 && x[1])actual.emplace_back(x[0],uint32_t(x[1]));
            }
            std::sort(expected.begin(),expected.end());std::sort(actual.begin(),actual.end());
            if(actual!=expected) {
                std::cerr<<"Mismatch pools="<<pools<<" pool="<<pool<<" budget="<<budget<<" pattern="<<pattern<<" row="<<row<<" pos="<<position[row]<<" expected="<<expected.size()<<" actual="<<actual.size()<<std::endl;
                std::vector<Pair> missing,extra;std::set_difference(expected.begin(),expected.end(),actual.begin(),actual.end(),std::back_inserter(missing));std::set_difference(actual.begin(),actual.end(),expected.begin(),expected.end(),std::back_inserter(extra));
                for(int i=0;i<int(std::min(std::size_t(8),missing.size()));++i)std::cerr<<" missing "<<missing[i].first<<":"<<missing[i].second;
                for(int i=0;i<int(std::min(std::size_t(8),extra.size()));++i)std::cerr<<" extra "<<extra[i].first<<":"<<extra[i].second;std::cerr<<std::endl;
                throw std::runtime_error("threshold top-k set differs from original sorted token selection");
            }
            if(count<=1024) {
                expected.clear();const auto p=position[row];const int available=std::min(pools,int((p+1)/pool));
                if(p+1<=budget) {
                    for(int id=0;int64_t(id)*pool<=p;++id) {
                        const int n=int(std::min(int64_t(pool),p+1-int64_t(id)*pool));
                        expected.emplace_back(id,n==32?0xffffffffu:(1u<<n)-1);
                    }
                }else {
                    std::vector<int> ranked(available);for(int i=0;i<available;++i)ranked[i]=i;
                    std::stable_sort(ranked.begin(),ranked.end(),[&](int a,int b){return input[row*pools+a]>input[row*pools+b];});
                    for(int i=0;i<count;++i)expected.emplace_back(ranked[i],pool==32?0xffffffffu:(1u<<pool)-1);
                    if((p+1)%pool)expected.emplace_back(int((p+1)/pool),(1u<<int((p+1)%pool))-1);
                }
                std::sort(expected.begin(),expected.end());
                if(actual!=expected)throw std::runtime_error("threshold selected set differs from independent CPU stable sort");
            }
        }
        ++cases;
    };
    for(int pattern:{0,1,2,3})for(int pool:{1,4,8,32})for(int pools:{1,129,4096,8193})for(int budget:{4,32,2048,6144}) {
        if(budget%pool)continue;std::vector<int64_t> position;
        for(int64_t p:{0,1,2,3,31,32,2047,2048,2050,2051,12289})position.push_back(std::min(p,int64_t(pool)*pools-1));
        test(pools,pool,budget,position,pattern);
    }
    // Exercise both sides of every per-thread threshold storage capacity.
    for(int pools:{257,511,513,1024,1025,2048,2049})for(int pattern:{0,1})
        for(int budget:{32,2048})test(pools,4,budget,{int64_t(pools)*4-1},pattern);
    for(int pools:{65536,262144})for(int pattern:{0,1})test(pools,4,2048,{int64_t(pools)*4-2},pattern);
    for(int pattern:{0,1,2,3})test(8193,4,4096,{32770},pattern);
    std::cout<<"threshold top-k independent CPU/reference sets,stable ties,signed zero,Inf,causal tail,multi-stage,1M context PASS cases="<<cases<<std::endl;
}

void set_directory(const std::string& path) {
#ifdef _WIN32
    _putenv_s("TYLOINFER_QSA_TQ_DIRECTORY",path.c_str());
    _putenv_s("TYLOINFER_QSA_TQ_BUDGET_BYTES","0");
#else
    setenv("TYLOINFER_QSA_TQ_DIRECTORY",path.c_str(),1);
    setenv("TYLOINFER_QSA_TQ_BUDGET_BYTES","0",1);
#endif
}
void check(int heads,int width,bool gated=false,bool half=false,bool ssd=true) {
    constexpr int kv_heads=2,index_width=13;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    mfq::cuda::QsaTqCache cache(64,heads,kv_heads,width,3,index_width,32,options,ssd);
    std::vector<float> source(20*kv_heads*width);
    for(std::size_t i=0;i<source.size();++i)source[i]=std::sin(float(i)*.13f)*.04f;
    auto values=tb::tensor(source,options).reshape({1,20,kv_heads,width});
    if(half) {
        auto strided=tb::zeros({1,20,kv_heads,2*width},options.dtype(tb::kFloat16)).narrow(3,width,width);
        strided.copy_(values);values=strided;
    }
    const auto append=[&](int start,int count) {
        auto q=tb::zeros({1,heads,count,width},options.dtype(half?tb::kFloat16:tb::kFloat32));
        auto v=values.narrow(1,start,count);
        auto raw=tb::zeros({1,count,index_width},options),iq=tb::zeros({1,count,3,index_width},options);
        auto positions=tb::arange(start,start+count,options.dtype(tb::kInt64)).reshape({1,count});
        auto gate=gated?tb::full({1,count,heads,2*width},1.5,options.dtype(half?tb::kFloat16:tb::kFloat32)).narrow(3,width,width):tb::Tensor{};
        auto destination=tb::empty({1,count,heads*width},options.dtype(half?tb::kFloat16:tb::kFloat32));
        auto output=cache.forward(q,v,v,iq,raw,positions,[&](const auto& x,const auto&) {
            return tb::zeros({1,x.size(1)/4,index_width},options);
        },false,gate,destination).to(tb::kCPU,tb::kFloat32).contiguous();
        double error_energy=0,source_energy=0;
        for(int row=0;row<count;++row)for(int h=0;h<heads;++h)for(int d=0;d<width;++d) {
            double expected=0,energy=0;for(int token=0;token<=start+row;++token) {
                const double x=source[(token*kv_heads+h/(heads/kv_heads))*width+d];expected+=x;energy+=x*x;
            }
            expected/=start+row+1;
            energy/=start+row+1;
            if(gated){const auto scale=1./(1.+std::exp(-1.5));expected*=scale;energy*=scale*scale;}
            const auto actual=output.data_ptr<float>()[(row*heads+h)*width+d];
            const double difference=actual-expected;
            error_energy+=difference*difference;source_energy+=energy;
            // Affine8 is approximate. Check its aggregate error and bound
            // outliers while independently checking token/head/causal layout.
            if(!std::isfinite(actual) || std::abs(difference)>1e-3) {
                std::cerr<<"reference mismatch width="<<width<<" append="<<start<<'+'<<count<<" row="<<row<<" head="<<h<<" channel="<<d<<" expected="<<expected<<" actual="<<actual<<'\n';
                throw std::runtime_error("adaptive QSA differs from causal uniform-attention reference");
            }
        }
        // Normalize by input energy: a uniform mean can nearly cancel while
        // its Affine8 reconstruction still has the expected quantization noise.
        if(error_energy>1e-4*std::max(source_energy,1e-30))
            throw std::runtime_error("adaptive QSA uniform-attention NMSE exceeds the Affine8 tolerance");
        if(cache.position()!=start+count)throw std::runtime_error("adaptive QSA append position mismatch");
    };
    append(0,5);append(5,6);
    // This restores part of an older, completed block from its SSD original.
    cache.truncate(3);append(3,4);
    cache.reset();append(0,3);append(3,1);cache.wait();
    std::cout<<"adaptive QSA causal attention, pending block, SSD partial rollback and reset PASS heads="<<heads<<" width="<<width<<" gated="<<gated<<" fp16_strided="<<half<<'\n';
}
void check_pending_graph(int heads,int width,bool ssd=true,bool wrap=false,bool nonzero=false) {
    constexpr int kh=2,iw=13;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    const auto half=options.dtype(tb::kFloat16);
    const int length=wrap?160:20;
    mfq::cuda::QsaTqCache cache(wrap?256:64,heads,kh,width,3,iw,wrap?256:32,options,ssd);
    std::vector<float> source(length*kh*width);
    for(std::size_t i=0;i<source.size();++i)source[i]=std::sin(float(i)*.13f)*.04f;
    auto values=tb::tensor(source,options).reshape({1,length,kh,width});
    auto q=tb::zeros({1,heads,1,width},half);
    std::vector<float> query(heads*width,0.f);
    if(nonzero) {
        for(int i=0;i<int(query.size());++i)query[i]=4.f*std::sin(float(i)*.283f+.7f);
        q.copy_(tb::tensor(query,options).reshape({1,heads,1,width}));
        auto rounded=q.to(tb::kCPU,tb::kFloat32).contiguous();
        std::copy_n(rounded.data_ptr<float>(),query.size(),query.begin());
    }
    auto v=tb::zeros({1,1,kh,2*width},half).narrow(3,width,width);
    auto raw=tb::zeros({1,1,iw},options),iq=tb::zeros({1,1,3,iw},options);
    auto positions=tb::zeros({1,1},options.dtype(tb::kInt64));
    auto gate=tb::full({1,1,heads,2*width},1.5,half).narrow(3,width,width);
    auto destination=tb::empty({1,1,heads*width},half);
    const auto pool=[&](const auto& x,const auto&) {return tb::zeros({1,x.size(1)/4,iw},options);};
    mfq::cuda::DecodeWindow partial(mfq_current_cuda_stream()),complete(mfq_current_cuda_stream());
    int replays=0;
    const auto append=[&](int start,int count) {
        for(int token=start;token<start+count;++token) {
            v.copy_(values.narrow(1,token,1));
            positions.copy_(tb::tensor(std::vector<int64_t>{token},options.dtype(tb::kInt64)).reshape({1,1}));
            if(auto kind=cache.device_graph_kind(1)) {
                auto& window=kind==2?complete:partial;
                cache.prepare_decode(true);
                if(!window.valid())window.capture([&]{
                    cache.forward_pending_graph(q,v,v,iq,raw,positions,gate,destination,pool);
                },[]{});
                if(window.segments()!=1)throw std::runtime_error("pending QSA graph retained a host boundary");
                window.run();cache.advance_graph(1);++replays;
            }else {
                cache.forward(q,v,v,iq,raw,positions,pool,false,gate,destination);
            }
            auto actual=destination.to(tb::kCPU,tb::kFloat32).contiguous();
            double error=0,energy=0;
            const double scale=1./(1.+std::exp(-1.5));
            for(int h=0;h<heads;++h) {
                std::vector<double> weights(token+1,1.);
                double denominator=token+1;
                if(nonzero) {
                    double peak=-1e100;
                    for(int t=0;t<=token;++t) {
                        double score=0;
                        for(int d=0;d<width;++d)score+=query[h*width+d]*source[(t*kh+h/(heads/kh))*width+d];
                        weights[t]=score/std::sqrt(double(width));peak=std::max(peak,weights[t]);
                    }
                    denominator=0;for(auto& weight:weights){weight=std::exp(weight-peak);denominator+=weight;}
                }
                for(int d=0;d<width;++d) {
                    double expected=0;
                    for(int t=0;t<=token;++t) {
                        const double x=source[(t*kh+h/(heads/kh))*width+d];
                        expected+=weights[t]*x;energy+=weights[t]*x*x*scale*scale/denominator;
                    }
                    expected*=scale/denominator;
                    const double difference=actual.data_ptr<float>()[h*width+d]-expected;
                    error+=difference*difference;
                    if(!std::isfinite(difference) || std::abs(difference)>1e-3)
                        throw std::runtime_error("pending QSA graph differs from causal reference at token "+std::to_string(token));
                }
            }
            if(error>1e-4*std::max(energy,1e-30) || cache.position()!=token+1)
                throw std::runtime_error("pending QSA graph changed tail precision or position");
        }
    };
    // One graph handles every incomplete phase, including repeated warmups.
    append(0,11);
    cache.truncate(5);append(5,7);
    // Verification can resize its own small-row arena while this graph is
    // retained. The next ordinary decode must keep its captured addresses.
    cache.reset();
    auto many_q=tb::zeros({1,heads,6,width},half);
    auto many_raw=tb::zeros({1,6,iw},options),many_iq=tb::zeros({1,6,3,iw},options);
    cache.forward(many_q,values.narrow(1,0,6),values.narrow(1,0,6),many_iq,many_raw,
                  tb::arange(0,6,positions.options()).reshape({1,6}),pool);
    append(6,6);
    cache.reset();append(0,8);
    if(wrap){append(8,132);cache.truncate(129);append(129,7);}
    cache.wait();
    std::cerr<<"adaptive QSA single-segment graph reuse, archive boundaries, rollback and reset PASS heads="
             <<heads<<" width="<<width<<" replays="<<replays<<'\n';
}
void check_batch_graph(int heads,int width,int bits) {
    const auto* saved=std::getenv("TYLOINFER_QSA_RAM_KV_BITS");const std::string previous=saved?saved:"";
    const auto setting=[](const std::string& value) {
#ifdef _WIN32
        _putenv_s("TYLOINFER_QSA_RAM_KV_BITS",value.c_str());
#else
        if(value.empty())unsetenv("TYLOINFER_QSA_RAM_KV_BITS");else setenv("TYLOINFER_QSA_RAM_KV_BITS",value.c_str(),1);
#endif
    };
    setting(std::to_string(bits));
    constexpr int maximum=512,kh=2,iw=13;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32),h=f.dtype(tb::kFloat16);
    mfq::cuda::QsaTqCache cache(maximum,heads,kh,width,3,iw,64,f,false),reference(maximum,heads,kh,width,3,iw,64,f,false);
    struct Inputs {tb::Tensor q,k,v,iq,raw,positions,absolute,gate,out;};
    std::map<int,Inputs> inputs;
    std::map<std::pair<int,int>,std::unique_ptr<mfq::cuda::DecodeWindow>> windows;
    const auto pool=[&](const auto& x,const auto& starts) {
        if(starts.dim()!=3 || starts.size(0)!=3)throw std::runtime_error("QSA archive discarded MRoPE axes");
        auto shift=starts.to(tb::kFloat32).sum(0).reshape({1,starts.size(-1),1})*.0001;
        return x.reshape({1,x.size(1)/4,4,iw}).mean(2)+shift;
    };
    for(int n=1;n<=8;++n) {
        auto& x=inputs[n];x.q=tb::zeros({1,heads,n,width},h);x.k=tb::zeros({1,n,kh,width},h);x.v=tb::zeros_like(x.k);
        x.iq=tb::zeros({1,n,3,iw},f);x.raw=tb::zeros({1,n,iw},f);x.positions=tb::zeros({1,n},f.dtype(tb::kInt64));
        x.absolute=tb::zeros({3,1,n},f.dtype(tb::kInt64));
        x.gate=tb::full({1,n,heads,width},.7,h);x.out=tb::empty({1,n,heads*width},h);
    }
    int position=0,replays=0;
    const auto append=[&](int n,int iteration,bool graph) {
        auto& x=inputs.at(n);std::vector<float> k(n*kh*width),v(k.size()),q(n*heads*width),raw(n*iw),iq(n*3*iw);
        for(std::size_t i=0;i<k.size();++i){k[i]=.1f*std::sin(float(i+position*kh*width)*.13f+iteration*.07f);v[i]=.1f*std::cos(float(i+position*kh*width)*.17f+iteration*.11f);}
        for(std::size_t i=0;i<q.size();++i)q[i]=std::sin(float(i)*.023f+iteration*.07f);
        for(std::size_t i=0;i<raw.size();++i)raw[i]=std::sin(float(i)*.071f+position*.13f);
        for(std::size_t i=0;i<iq.size();++i)iq[i]=std::cos(float(i)*.093f+position*.17f);
        x.k.copy_(tb::tensor(k,f).reshape(x.k.sizes()));x.v.copy_(tb::tensor(v,f).reshape(x.v.sizes()));
        x.q.copy_(tb::tensor(q,f).reshape(x.q.sizes()));x.raw.copy_(tb::tensor(raw,f).reshape(x.raw.sizes()));
        x.iq.copy_(tb::tensor(iq,f).reshape(x.iq.sizes()));x.positions.copy_(tb::arange(position,position+n,x.positions.options()).reshape({1,n}));
        for(int axis=0;axis<3;++axis)x.absolute.narrow(0,axis,1).copy_((x.positions+axis*7).reshape({1,1,n}));
        auto expected=reference.forward(x.q,x.k,x.v,x.iq,x.raw,x.absolute,pool,false,x.gate).to(tb::kCPU,tb::kFloat32).contiguous();
        if(graph && iteration==6) {
            cache.prepare_decode_tokens(n,true);
            cache.forward_pending_graph(x.q,x.k,x.v,x.iq,x.raw,x.positions,x.gate,x.out,pool,x.absolute);
            graph=false; // Abandon a warmup that already crossed block boundaries.
        }
        if(graph) {
            const int kind=cache.device_graph_kind(n);if(!kind)throw std::runtime_error("RAM batch graph dispatch was not enabled");
            cache.prepare_decode_tokens(n,true);auto& window=windows[{n,kind}];
            if(!window)window=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
            if(!window->valid())window->capture([&]{cache.forward_pending_graph(x.q,x.k,x.v,x.iq,x.raw,x.positions,x.gate,x.out,pool,x.absolute);},[]{});
            if(window->segments()!=1)throw std::runtime_error("RAM batch graph retained a CPU boundary");
            window->run();cache.advance_graph(n);++replays;
        }else cache.forward(x.q,x.k,x.v,x.iq,x.raw,x.absolute,pool,false,x.gate,x.out);
        auto actual=x.out.to(tb::kCPU,tb::kFloat32).contiguous();double error=0,energy=0,maximum_error=0;
        for(int64_t i=0;i<actual.numel();++i) {
            const double a=actual.data_ptr<float>()[i],b=expected.data_ptr<float>()[i],d=a-b;
            error+=d*d;energy+=b*b;maximum_error=std::max(maximum_error,std::abs(d));
        }
        if(!std::isfinite(error) || maximum_error>.002 || error>5e-4*std::max(energy,1e-30))
            throw std::runtime_error("RAM batch graph parity failure: M="+std::to_string(n)+" position="+std::to_string(position)+" max="+std::to_string(maximum_error)+" nmse="+std::to_string(error/std::max(energy,1e-30)));
        position+=n;if(cache.position()!=position)throw std::runtime_error("RAM batch graph changed the token timeline");
    };
    append(3,0,false);
    for(int i=0;i<48;++i) {
        if(i==17){position-=5;cache.truncate(position);reference.truncate(position);}
        if(i==32){position=0;cache.reset();reference.reset();}
        append(i%8+1,i+1,true);
    }
    while(position<maximum)append(std::min(8,maximum-position),49+position,true);
    if(cache.device_graph_kind(1))throw std::runtime_error("RAM graph exceeded its final context position");
    cache.wait();reference.wait();setting(previous);
    std::cerr<<"adaptive QSA M1..8 single-segment graph parity/phase/warmup/rollback/reset PASS heads="<<heads<<" width="<<width<<" bits="<<bits<<" replays="<<replays<<'\n';
}
void check_async_history() {
    constexpr int heads=8,kh=2,width=17,iw=13,length=260;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    mfq::cuda::QsaTqCache cache(512,heads,kh,width,3,iw,512,f);
    cache.set_budget(16ULL<<20);
    std::vector<float> source(length*kh*width);
    for(std::size_t i=0;i<source.size();++i)source[i]=.04f*std::sin(float(i)*.13f);
    const auto values=tb::tensor(source,f).reshape({1,length,kh,width});
    const auto append=[&](int start,int count) {
        auto q=tb::zeros({1,heads,count,width},f);
        auto v=values.narrow(1,start,count);
        return cache.forward(q,v,v,tb::zeros({1,count,3,iw},f),tb::zeros({1,count,iw},f),
            tb::arange(start,start+count,f.dtype(tb::kInt64)).reshape({1,count}),
            [&](const auto& x,const auto&){return tb::zeros({1,x.size(1)/4,iw},f);});
    };
    const auto validate=[&](const tb::Tensor& output,int end) {
        const auto cpu=output.to(tb::kCPU,tb::kFloat32).contiguous();
        const auto row=output.numel()/heads/width-1;
        for(int h=0;h<heads;++h)for(int d=0;d<width;++d) {
            double expected=0;
            for(int t=0;t<end;++t)expected+=source[(t*kh+h/(heads/kh))*width+d]/end;
            const auto actual=cpu.data_ptr<float>()[(row*heads+h)*width+d];
            if(!std::isfinite(actual) || std::abs(actual-expected)>1.e-3)
                throw std::runtime_error("Deferred history/reset changed causal attention");
        }
    };
    // Consume an outstanding observation before appending new blocks; no
    // output download or explicit GPU barrier separates these forwards.
    auto first=append(0,128);
    auto next=append(128,4);
    validate(next,132);cache.wait();
    const auto updates=cache.stats().updates;
    if(updates!=1)throw std::runtime_error("Deferred history was lost or consumed twice");
    cache.wait();
    if(cache.stats().updates!=updates)throw std::runtime_error("Repeated wait consumed history twice");
    auto pending=append(132,128);
    cache.reset();
    if(cache.position()!=0)throw std::runtime_error("Reset retained an old observation timeline");
    auto restarted=append(0,128);
    validate(restarted,128);cache.wait();
    if(cache.stats().updates!=updates+2)throw std::runtime_error("History was lost across reset");
    // A reset also safely retires queued attention without an observation.
    auto queued=append(128,4);
    cache.reset();
    auto final=append(0,4);
    validate(final,4);cache.wait();
    std::cerr<<"adaptive QSA deferred history, exactly-once wait, append growth and queued reset PASS\n";
}
void check_nonzero_attention(int heads,int kh,int width) {
    constexpr int t=17,iw=13;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    std::vector<float> key(t*kh*width),value(key.size()),query(heads*t*width);
    const auto half=[](float x){return __half2float(__float2half_rn(x));};
    for(std::size_t i=0;i<key.size();++i){key[i]=half(.05f*std::sin(float(i)*.173f));value[i]=half(.1f*std::cos(float(i)*.131f));}
    for(std::size_t i=0;i<query.size();++i)query[i]=half(4.f*std::sin(float(i)*.283f+.7f));
    auto k=tb::tensor(key,f).reshape({1,t,kh,width}).to(tb::kFloat16);
    auto v=tb::tensor(value,f).reshape({1,t,kh,width}).to(tb::kFloat16);
    auto q=tb::tensor(query,f).reshape({1,heads,t,width}).to(tb::kFloat16);
    mfq::cuda::QsaTqCache cache(64,heads,kh,width,3,iw,32,f);
    // Normalization emits FP32 Q in the real model. The cache explicitly
    // permits tensor-core operands while retaining FP32 accumulation.
    auto output=cache.forward(q.to(tb::kFloat32),k,v,tb::zeros({1,t,3,iw},f),tb::zeros({1,t,iw},f),
        tb::arange(0,t,f.dtype(tb::kInt64)).reshape({1,t}),
        [&](const auto& x,const auto&){return tb::zeros({1,x.size(1)/4,iw},f);}).to(tb::kCPU).contiguous();
    mfq::cuda::QsaTqCache reference_cache(64,heads,kh,width,3,iw,32,f);
    auto fp32=reference_cache.forward(q.to(tb::kFloat32),k,v,tb::zeros({1,t,3,iw},f),tb::zeros({1,t,iw},f),
        tb::arange(0,t,f.dtype(tb::kInt64)).reshape({1,t}),
        [&](const auto& x,const auto&){return tb::zeros({1,x.size(1)/4,iw},f);},false,{},{},false).to(tb::kCPU).contiguous();
    double maximum_error=0,tensor_core_error=0;
    for(int row=0;row<t;++row)for(int h=0;h<heads;++h) {
        std::vector<double> scores(row+1);double peak=-1e100,total=0;
        for(int at=0;at<=row;++at) {
            double score=0;
            for(int d=0;d<width;++d)score+=query[(h*t+row)*width+d]*key[(at*kh+h/(heads/kh))*width+d];
            scores[at]=score/std::sqrt(double(width));peak=std::max(peak,scores[at]);
        }
        for(auto& score:scores){score=std::exp(score-peak);total+=score;}
        for(int d=0;d<width;++d) {
            double expected=0;
            for(int at=0;at<=row;++at)expected+=scores[at]*value[(at*kh+h/(heads/kh))*width+d]/total;
            const auto actual=output.data_ptr<float>()[(row*heads+h)*width+d];
            const auto baseline=fp32.data_ptr<float>()[(row*heads+h)*width+d];
            tensor_core_error=std::max(tensor_core_error,double(std::abs(actual-baseline)));
            if(std::abs(actual-baseline)>2e-4)throw std::runtime_error("tensor-core QSA differs from FP32 attention on the same quantized KV");
            const double error=std::abs(actual-expected);maximum_error=std::max(maximum_error,error);
            // Same relative Affine8 outlier bound as the smaller-amplitude
            // uniform test; the tensor-core-only error is checked separately.
            if(!std::isfinite(actual) || error>2.5e-3) {
                std::cerr<<"nonzero mismatch token="<<row<<" head="<<h<<" channel="<<d<<" expected="<<expected<<" actual="<<actual<<" error="<<error<<'\n';
                throw std::runtime_error("nonzero adaptive attention differs from CPU causal reference");
            }
        }
    }
    std::cerr<<"adaptive QSA nonzero FP16 attention and padded tensor-core tile PASS heads="<<heads<<" kv_heads="<<kh<<" width="<<width<<" max_quantization_error="<<maximum_error
             <<" max_tensor_core_error="<<tensor_core_error<<'\n';
}
void check_nonzero_attention_geometries() {
    for(int width:{256,257,1024})check_nonzero_attention(24,2,width);
    // Exercise a partial second eight-head tile, multiple sixteen-head
    // groups, and padded channels against the independent CPU reference.
    check_nonzero_attention(20,2,256);
    check_nonzero_attention(48,2,256);
    check_nonzero_attention(36,2,257);
    check_nonzero_attention(24,4,256);
    check_nonzero_attention(10,2,257);
    check_nonzero_attention(12,2,1024);
}
void check_packed_decode_windows() {
    namespace tq=tyloinfer::qsa;
    constexpr int heads=2,width=256,rows=4*heads,images=8*17;
    constexpr int metadata=(2*rows+(rows+1)/2+3)&~3;
    constexpr int stride=(metadata+rows*256+15)&~15;
    const auto f=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    const auto host=f.device(tb::kCPU).dtype(tb::kUInt8);
    std::vector<std::uint8_t> encoded(std::size_t(images)*2*stride);
    std::vector<tq::TqBlockView> views(images);
    std::vector<tq::SelectedBlock> selected(images);
    std::uint32_t rng=0x713b9e21u;
    for(auto& x:encoded){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;x=std::uint8_t(rng);}
    for(int image=0;image<images;++image)for(int kind=0;kind<2;++kind) {
        auto* record=encoded.data()+(image*2+kind)*stride;
        std::memset(record+2*rows,0,metadata-2*rows);
        for(int row=0;row<rows;++row) {
            const auto scale=__float2half_rn(row%5==0?0.f:std::ldexp(1.f,-3-row%5));
            std::memcpy(record+2*row,&scale,2);
        }
    }
    auto payload=tb::empty({int64_t(encoded.size())},host);
    std::memcpy(payload.data_ptr(),encoded.data(),encoded.size());payload=payload.to(tb::kCUDA);
    const auto upload=[&](const auto& values) {
        auto cpu=tb::empty({int64_t(values.size()*sizeof(values[0]))},host);
        std::memcpy(cpu.data_ptr(),values.data(),size_t(cpu.numel()));return cpu.to(tb::kCUDA);
    };
    for(int image=0;image<images;++image) {
        const unsigned bits=unsigned(image/17+1),profile=unsigned(image%17),flags=bits|(profile<<8);
        const auto* start=static_cast<const std::uint8_t*>(payload.data_ptr())+image*2*stride;
        views[image]={start,start+stride,flags,flags};selected[image]={image,15};
    }
    auto table=upload(views),selection=upload(selected),relative=tb::ones({16},f.dtype(tb::kFloat16));
    auto keys=tb::empty({images,heads,4,width},f.dtype(tb::kFloat16)),values=tb::empty_like(keys);
    auto requested=tb::empty({images},f.dtype(tb::kInt32));
    const int status=tq::tq_stage_attention_blocks(static_cast<const tq::TqBlockView*>(table.data_ptr()),
        static_cast<const std::uint16_t*>(relative.data_ptr()),static_cast<std::uint16_t*>(keys.data_ptr()),
        static_cast<std::uint16_t*>(values.data_ptr()),images,heads,width,mfq_current_cuda_stream(),
        static_cast<const tq::SelectedBlock*>(selection.data_ptr()),1,images,requested.data_ptr<std::uint32_t>());
    if(status)throw std::runtime_error(cudaGetErrorString(cudaError_t(status)));
    const auto actual_keys=keys.cpu().contiguous(),actual_values=values.cpu().contiguous();
    std::size_t checked=0;
    for(int image=0;image<images;++image)for(int kind=0;kind<2;++kind) {
        const int bits=image/17+1,profile=image%17,state_bits=profile?profile:16;
        const auto* record=encoded.data()+(image*2+kind)*stride;
        const auto* actual=static_cast<const std::uint16_t*>((kind?actual_values:actual_keys).data_ptr());
        for(int head=0;head<heads;++head)for(int token=0;token<4;++token)for(int d=0;d<width;++d) {
            float expected=0;
            for(int mode=0;mode<4;++mode) {
                const int row=head*4+mode,total=256*bits;std::uint32_t state=0;
                const auto* stream=record+metadata+row*32*bits;
                // Read individual circular bits, independently of the GPU's
                // adjacent-coordinate extraction and word-boundary handling.
                for(int j=0;j<state_bits;++j) {
                    const int bit=((255-d)*bits+j)%total;
                    state|=((stream[bit/8]>>(bit&7))&1u)<<j;
                }
                const std::uint32_t product=state*0x83dcd12du;
                const unsigned sum=(product&255)+((product>>8)&255)+((product>>16)&255)+(product>>24);
                const auto base=__float2half_rn(std::fma(float(sum+1024),
                    __half2float(__ushort_as_half(0x1eee)),__half2float(__ushort_as_half(0xc931))));
                __half scale;std::memcpy(&scale,record+2*row,2);
                const float value=__half2float(__float2half_rn(__half2float(base)*__half2float(scale)));
                const int parity=((token&mode)&1)^(((token&mode)>>1)&1);
                expected+=(parity?-.5f:.5f)*value;
            }
            const auto offset=((std::size_t(image)*heads+head)*4+token)*width+d;
            if(actual[offset]!=__half_as_ushort(__float2half_rn(expected)))
                throw std::runtime_error("Packed KV decode differs from circular-bit/H4 CPU reference");
            ++checked;
        }
    }
    std::cerr<<"packed KV adjacent decode bits1..8/state0..16, wrap/three-word windows, zero scales and H4 exact PASS values="<<checked<<'\n';
}
int main(int argc,char** argv)try {
    int devices=0;const auto available=cudaGetDeviceCount(&devices);
    if(available!=cudaSuccess || !devices) {
        std::cerr<<"adaptive QSA cache test SKIP: "<<cudaGetErrorString(available)<<" devices="<<devices<<'\n';return 77;
    }
    set_directory(argc>1?argv[1]:(std::filesystem::current_path()/"qsa-tq-test-cache").string());
    auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    if(argc>2 && std::string(argv[2])=="--attention-only") {
        check_mass_observation();
        check_packed_decode_windows();
        check_nonzero_attention_geometries();
        std::cerr<<"adaptive QSA attention-only checks PASS\n";
        return 0;
    }
    check_attention_partitions();check_mass_observation();check_selected_blocks();check_packed_decode_windows();
    check_async_history();
    check(6,17);check(10,257);check(10,257,true,true);check(24,256,true,true);
    check_pending_graph(6,17);check_pending_graph(10,257);check_pending_graph(24,256);
    check_pending_graph(24,257);
    check(10,257,true,true,false);check_pending_graph(24,256,false);
    check_pending_graph(6,17,false,true);
    check_pending_graph(24,257,false,false,true);
    check_batch_graph(6,17,16);
    check_batch_graph(24,256,16);
    check_batch_graph(10,257,8);
    check_batch_graph(6,17,4);
    check_batch_graph(20,257,8);
    check_batch_graph(48,256,4);
    check_batch_graph(24,1024,16);
    check_nonzero_attention_geometries();
    context->end_graph_pool(stream);
    std::cerr<<"adaptive QSA cache checks PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
