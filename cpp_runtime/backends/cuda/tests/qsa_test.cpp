#include "models/qwen4_exp/model.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_context.h"
#include "selected_attention.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
void qsa_tile_pair_benchmark(int rows,int heads,int kv_heads,int capacity,int selected);
namespace tb=mfq_tensor_backend;
using tb::Tensor;
using namespace mfq::cuda;
namespace {
Tensor data(std::vector<int64_t> shape,int salt,float scale=.03f) {
    int64_t count=1;for(auto x:shape)count*=x;
    std::vector<float> values(count);
    for(int64_t i=0;i<count;++i)values[i]=std::sin(float(i)*.173f+salt)*scale;
    return tb::tensor(values).reshape(shape).to(tb::kCUDA);
}
void setting(const char* value) {
#ifdef _WIN32
    _putenv_s("TYLOINFER_QSA_SPARSE_QUERY_FUSED",value?value:"");
#else
    if(value)setenv("TYLOINFER_QSA_SPARSE_QUERY_FUSED",value,1);else unsetenv("TYLOINFER_QSA_SPARSE_QUERY_FUSED");
#endif
}
struct QueryMode {
    bool present;std::string previous;
    explicit QueryMode(const char* mode):present(std::getenv("TYLOINFER_QSA_SPARSE_QUERY_FUSED")!=nullptr) {
        if(present)previous=std::getenv("TYLOINFER_QSA_SPARSE_QUERY_FUSED");setting(mode);
    }
    ~QueryMode(){setting(present?previous.c_str():nullptr);}
};
void general_query() {
    constexpr int width=256,capacity=769,count=513;
    int cases=0;
    for(int heads:{6,10,24,40,48})for(int tokens:{1,2,3,4})for(bool half:{false,true}) {
        auto query=data({1,tokens,heads,2*width},510,.3f).to(half?tb::kFloat16:tb::kFloat32)
            .narrow(3,0,width).permute({0,2,1,3});
        auto gate=data({1,tokens,heads,2*width},511,9.f).narrow(3,width,width);
        auto key=data({1,2,capacity,width},512,.17f),value=data(key.sizes().vec(),513,.19f);
        auto indices=tb::arange(tokens*count,query.options().dtype(tb::kInt32)).remainder(capacity).reshape({1,tokens,count});
        QueryMode mode("0");const auto expected=mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,value,indices,gate,false);
        setting("1");const auto actual=mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,value,indices,gate,false);
        const float error=(actual-expected).abs().max().item<float>(),maximum=expected.abs().max().item<float>();
        if(!std::isfinite(error) || error>2e-6f+2e-5f*maximum)throw std::runtime_error("QSA direct query changed head tails or strided activations");
        ++cases;
    }
    std::cout<<"QSA arbitrary GQA ratios/strided f16,f32 queries M1..4 cases="<<cases<<" PASS\n";
}
void gate_reference() {
    for(int width:{17,256,512})for(int tokens:{1,4,129})for(bool half:{false,true}) {
        auto attended=data({2,tokens,6,width},520,3.f);
        auto gate=data({2,tokens,6,2*width},521,100.f).narrow(3,width,width);
        const auto a=attended.cpu(),g=gate.contiguous().cpu();
        auto reference=tb::empty(a.sizes(),a.options());
        for(int64_t i=0;i<a.numel();++i)reference.data_ptr<float>()[i]=a.data_ptr<float>()[i]/(1.f+std::exp(-g.data_ptr<float>()[i]));
        const auto actual=mfq_selected_attention::gated(attended,gate,half).to(tb::kFloat32);
        const auto expected=reference.to(half?tb::kFloat16:tb::kFloat32).to(tb::kFloat32).to(tb::kCUDA);
        const float error=(actual-expected).abs().max().item<float>();
        if(!std::isfinite(error) || error>(half?.002f:1e-6f))throw std::runtime_error("QSA gate differs from FP32 CPU reference");
    }
    std::cout<<"QSA strided FP32 gate CPU reference + extreme sigmoid PASS\n";
}
void sparse_uniform_reference() {
    // Zero queries give a uniform distribution over valid selections. A CPU
    // average independently checks both 32/64-key tails and fully masked rows.
    constexpr int width=256,capacity=69;
    int cases=0;float largest=0;
    for(int heads:{6,24})for(int tokens:{1,2,3,4})for(int count:{1,31,32,33,63,64,65}) {
        std::vector<float> values(2*capacity*width);
        for(size_t i=0;i<values.size();++i)values[i]=float(int(i%15)-7)/4096.f;
        std::vector<int32_t> ids(tokens*count);
        for(int t=0;t<tokens;++t)for(int c=0;c<count;++c)
            ids[t*count+c]=t==tokens-1 && tokens>1?-1:c%7==0?-1:c%11==0?capacity+3:(c*13+t)%capacity;
        auto query=tb::zeros({1,heads,tokens,width},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32));
        auto key=tb::zeros({1,2,capacity,width},query.options().dtype(tb::kFloat16));
        auto value=tb::tensor(values).reshape(key.sizes()).to(tb::kCUDA).to(tb::kFloat16);
        auto selected=tb::tensor(ids).reshape({1,tokens,count}).to(tb::kCUDA);
        auto gate=tb::zeros({1,tokens,heads,width},query.options());
        const auto output=mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,value,selected,gate,false).contiguous().cpu();
        for(int t=0;t<tokens;++t)for(int h=0;h<heads;++h)for(int d=0;d<width;++d) {
            double sum=0;int valid=0;
            for(int c=0;c<count;++c)if(const auto row=ids[t*count+c];row>=0 && row<capacity) {
                sum+=values[(size_t(h/(heads/2))*capacity+row)*width+d];++valid;
            }
            const float expected=valid?float(.5*sum/valid):0.f;
            const float error=std::abs(output.data_ptr<float>()[(t*heads+h)*width+d]-expected);
            if(!std::isfinite(error) || error>1e-7f)throw std::runtime_error("QSA uniform attention differs from independent CPU average");
            largest=std::max(largest,error);
        }
        ++cases;
    }
    std::cout<<"QSA uniform sparse attention CPU reference 32/64-key tails and fully masked rows cases="<<cases<<" max_error="<<largest<<" PASS\n";
}
void resolved_hot_reference() {
    constexpr int width=256,capacity=69,cells=17;
    int cases=0;float largest=0;
    for(int heads:{6,24})for(int tokens:{1,2,3,4})for(int count:{33,65})
    for(bool exclusive:{false,true})for(bool half:{false,true}) {
        auto query=data({1,heads,tokens,2*width},710,.3f).slice(-1,0,2*width,2).to(half?tb::kFloat16:tb::kFloat32);
        auto key=data({1,2,capacity,width},711,.17f).to(tb::kFloat16);
        auto value=data(key.sizes().vec(),712,.19f).to(tb::kFloat16);
        auto gate=data({1,tokens,heads,2*width},713,.7f).slice(-1,0,2*width,2);
        std::vector<int32_t> selected(tokens*count),mapping(capacity,-1),cold_mapping(capacity),hot_ids(cells),cold_ids(capacity);
        for(size_t i=0;i<selected.size();++i)selected[i]=i%7==0?-1:i%11==0?capacity+3:int((i*13+i/count)%capacity);
        for(int i=0;i<capacity;++i){cold_mapping[i]=capacity-1-i;cold_ids[i]=capacity-1-i;}
        for(int i=0;i<cells;++i){mapping[i]=cells-1-i;hot_ids[i]=cells-1-i;if(exclusive)cold_mapping[i]=-1;}
        auto ids=tb::tensor(selected).reshape({1,tokens,count}).to(tb::kCUDA);
        auto map=tb::tensor(mapping).to(tb::kCUDA),cold_map=tb::tensor(cold_mapping).to(tb::kCUDA);
        auto hot_index=tb::tensor(hot_ids).to(tb::kInt64).to(tb::kCUDA),cold_index=tb::tensor(cold_ids).to(tb::kInt64).to(tb::kCUDA);
        auto hot_key=key.index_select(2,hot_index).contiguous(),hot_value=value.index_select(2,hot_index).contiguous();
        auto cold_key=exclusive?key.index_select(2,cold_index).contiguous():key.clone();
        auto cold_value=exclusive?value.index_select(2,cold_index).contiguous():value.clone();
        const HotKvView hot{hot_key.data_ptr(),hot_value.data_ptr(),map.data_ptr<int32_t>(),cells,
            exclusive?cold_map.data_ptr<int32_t>():nullptr,exclusive?capacity:0};
        Tensor output;
        DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{output=mfq_qwen4_exp::sparse_gqa_attention_gate(query,cold_key,cold_value,ids,gate,false,hot);},[]{},[&]{output={};});
        for(int replay=0;replay<3;++replay) {
            if(replay==1) {
                // Keep all addresses fixed while both tiers change layout.
                for(int i=0;i<cells;++i)mapping[i]=i;
                for(int i=0;i<capacity;++i)cold_mapping[i]=i<cells && exclusive?-1:i;
                map.copy_(tb::tensor(mapping).to(tb::kCUDA));cold_map.copy_(tb::tensor(cold_mapping).to(tb::kCUDA));
                hot_key.copy_(key.narrow(2,0,cells));hot_value.copy_(value.narrow(2,0,cells));
                cold_key.copy_(key);cold_value.copy_(value);
            }else if(replay==2)ids.fill_(-1);
            window.run();
            const auto expected=mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,value,ids,gate,false);
            const float error=(output-expected).abs().max().item<float>();
            const float maximum=expected.abs().max().item<float>();
            if(!std::isfinite(error) || error>2e-6f+2e-5f*maximum)
                throw std::runtime_error("QSA resolved hot/cold row descriptors changed canonical attention");
            largest=std::max(largest,error);++cases;
        }
    }
    std::cout<<"QSA resolved VRAM/RAM descriptors, remapped tiers, masked tails and changing graphs cases="<<cases<<" max_error="<<largest<<" PASS\n";
}
void history_tail_graph() {
    using namespace mfq::cuda::qwen4_exp;
    constexpr int hidden=32,heads=6,kv=2,iw=8,ih=2,pool=4,maximum=64,prefix=6;
    int cases=0;
    const auto close=[](const Tensor& actual,const Tensor& expected) {
        const auto a=actual.to(tb::kCPU,tb::kFloat32).contiguous();
        const auto b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
        if(a.sizes()!=b.sizes())throw std::runtime_error("QSA tail output shape differs");
        for(int64_t i=0;i<a.numel();++i) {
            const auto error=std::abs(a.data_ptr<float>()[i]-b.data_ptr<float>()[i]);
            if(!std::isfinite(error) || error>2e-5f+2e-4f*std::abs(b.data_ptr<float>()[i]))
                throw std::runtime_error("QSA tail output differs from complete-row attention");
        }
    };
    for(int width:{32,48})for(int budget:{8,maximum})for(bool half:{false,true}) {
        CudaExecutionContext execution;
        const auto dtype=half?tb::kFloat16:tb::kFloat32;
        const auto linear=[&](int in,int out,int salt)->Linear {
            auto w=data({out,in},salt).to(dtype);
            return [w](CudaExecutionContext&,const Tensor& x){return tb::matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
        };
        auto norm=data({width},1,0),inorm=data({iw},2,0);
        QsaWeights weights{linear(hidden,heads*2*width,1),linear(hidden,kv*width,2),linear(hidden,kv*width,3),
            linear(heads*width,hidden,4),linear(hidden,(ih+1)*iw,5),norm,norm,inorm,inorm};
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.,std::vector<int64_t>{1,1,2});
        QsaConfig config{heads,kv,width,ih,iw,pool,budget,maximum,1e-6};
        int query_rows=0;
        auto tail_weights=weights;
        tail_weights.query=[base=weights.query,&query_rows](CudaExecutionContext& context,const Tensor& x) {
            query_rows=std::max(query_rows,int(x.size(1)));return base(context,x);
        };
        Qsa full(weights,config,rotary),tail(tail_weights,config,rotary);
        auto inputs=data({1,maximum,hidden},27,.4f).to(dtype);
        auto history=tb::arange(prefix,inputs.options().dtype(tb::kInt64));
        full.forward(execution,inputs.narrow(1,0,prefix),history,history,true);
        tail.forward(execution,inputs.narrow(1,0,prefix),history,history,true);
        full.prepare_graph(history,maximum);tail.prepare_graph(history,maximum);
        for(int rows:{1,2,3,4,8}) {
            query_rows=0;
            auto x=inputs.narrow(1,prefix,rows).clone();
            auto positions=tb::arange(prefix,prefix+rows,history.options());
            auto axes=positions.reshape({1,1,rows}).expand({3,1,rows}).contiguous();
            Tensor expected,actual;
            DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&] {
                expected=full.forward_graph(execution,x,axes,positions);
                actual=tail.forward_graph(execution,x,axes,positions,{},true,false,true);
            },[]{},[&]{expected={};actual={};});
            if(query_rows!=1)throw std::runtime_error("QSA tail still projected unused history queries");
            for(int replay=0;replay<3;++replay) {
                full.truncate(prefix);tail.truncate(prefix);
                x.copy_(data({1,rows,hidden},71+replay,.4f).to(dtype));
                axes.copy_((positions+replay*3).reshape({1,1,rows}).expand({3,1,rows}));
                axes.narrow(0,1,1).copy_(axes.narrow(0,1,1)+5);
                axes.narrow(0,2,1).copy_(axes.narrow(0,2,1)+11);
                graph.run();full.advance_graph(rows);tail.advance_graph(rows);
                close(actual,expected.narrow(1,rows-1,1));
                Qwen4BlockSessionState a,b;std::size_t bytes=0;
                tail.capture_session(a,prefix+rows,bytes);full.capture_session(b,prefix+rows,bytes);
                for(const auto& pair:{std::pair{a.keys,b.keys},std::pair{a.values,b.values},std::pair{a.index,b.index}}) {
                    const auto av=pair.first.cpu().contiguous(),bv=pair.second.cpu().contiguous();
                    if(av.sizes()!=bv.sizes() || av.scalar_type()!=bv.scalar_type() ||
                        std::memcmp(av.data_ptr(),bv.data_ptr(),av.nbytes()))
                        throw std::runtime_error("QSA history tail changed complete history KV/index");
                }
                ++cases;
            }
            auto next_pos=tb::arange(prefix+rows,prefix+rows+1,history.options());
            auto next_x=inputs.narrow(1,prefix+rows,1);
            DecodeWindow next(mfq_current_cuda_stream());
            next.capture([&] {
                expected=full.forward_graph(execution,next_x,next_pos,next_pos);
                actual=tail.forward_graph(execution,next_x,next_pos,next_pos);
            },[]{},[&]{expected={};actual={};});
            next.run();full.advance_graph(1);tail.advance_graph(1);close(actual,expected);++cases;
        }
    }
    std::cout<<"QSA batched-history tail dense/sparse FP16/FP32 axes/replay/rollback/following-row KV+index exact cases="<<cases<<" PASS\n";
}
void draft_window_graph() {
    using namespace mfq::cuda::qwen4_exp;
    constexpr int width=32,heads=2,maximum=37,prefix=5;
    int cases=0;
    for(int window:{1,3,7,maximum})for(bool half:{false,true})for(bool managed:{false,true}) {
        CudaExecutionContext execution;
        const auto dtype=half?tb::kFloat16:tb::kFloat32;
        const auto zeros=[](int columns)->Linear {
            return [columns](CudaExecutionContext&,const Tensor& x) {
                return tb::zeros({x.size(0),x.size(1),columns},x.options());
            };
        };
        // Q/K zero gives uniform attention. Independent CPU window averages
        // catch missing causal masks, old-prefix leakage and dropped history.
        QsaWeights weights{zeros(heads*width*2),zeros(width),
            [](CudaExecutionContext&,const Tensor& x){return x;},
            [](CudaExecutionContext&,const Tensor& x){return x.narrow(-1,0,width);},zeros(16),
            data({width},1,0),data({width},1,0),data({8},1,0),data({8},1,0)};
        QsaConfig config{heads,1,width,1,8,1,maximum,maximum,1e-6,window};
        auto rotary=std::make_shared<RotaryEmbedding>(8,maximum,10000.);
        std::shared_ptr<KvMemoryManager> memory;
        if(managed){KvMemoryOptions options;options.placement="vram";memory=std::make_shared<KvMemoryManager>(options);}
        Qsa full(weights,config,rotary,true,true,{},memory),tail(weights,config,rotary,true,true,{},memory);
        if(memory)memory->finalize();
        auto inputs=data({1,maximum,width},170,.4f).to(dtype);
        auto history=tb::arange(prefix,inputs.options().dtype(tb::kInt64));
        full.forward(execution,inputs.narrow(1,0,prefix),history,history,true);
        tail.forward(execution,inputs.narrow(1,0,prefix),history,history,true);
        full.prepare_graph(history,maximum);tail.prepare_graph(history,maximum);
        const auto original=inputs.to(tb::kFloat16).to(tb::kFloat32).contiguous().cpu();
        for(int rows:{1,2,3,4,8}) {
            full.truncate(prefix);tail.truncate(prefix);
            auto x=inputs.narrow(1,prefix,rows).clone();
            auto pos=tb::arange(prefix,prefix+rows,history.options());
            Tensor full_output,tail_output;
            DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&] {
                full_output=full.forward_graph(execution,x,pos,pos);
                tail_output=tail.forward_graph(execution,x,pos,pos,{},true,false,true);
            },[]{},[&]{full_output={};tail_output={};});
            for(int replay=0;replay<3;++replay) {
                full.truncate(prefix);tail.truncate(prefix);
                x.copy_(data({1,rows,width},180+replay,.4f).to(dtype));
                graph.run();full.advance_graph(rows);tail.advance_graph(rows);
                const auto fresh=x.to(tb::kFloat16).to(tb::kFloat32).contiguous().cpu();
                const auto a=full_output.to(tb::kFloat32).contiguous().cpu();
                const auto b=tail_output.to(tb::kFloat32).contiguous().cpu();
                for(int row=0;row<rows;++row)for(int column=0;column<width;++column) {
                    const int end=prefix+row,begin=std::max(0,end+1-window);
                    double expected=0;
                    for(int index=begin;index<=end;++index)
                        expected+=index<prefix?original.data_ptr<float>()[index*width+column]:
                            fresh.data_ptr<float>()[(index-prefix)*width+column];
                    expected*=.5/(end-begin+1);
                    const double tolerance=half?.0003:.00001;
                    if(std::abs(a.data_ptr<float>()[row*width+column]-expected)>tolerance ||
                        (row==rows-1 && std::abs(b.data_ptr<float>()[column]-expected)>tolerance))
                        throw std::runtime_error("draft dense-window attention differs from CPU causal-window average");
                }
                if(full.position()!=prefix+rows || tail.position()!=prefix+rows)
                    throw std::runtime_error("draft dense window lost history KV positions");
                ++cases;
            }
        }
    }
    std::cout<<"Draft dense-window/full-context graph FP16/FP32 CPU reference, managed/ordinary KV, causal edges, rollback and history-tail cases="<<cases<<" PASS\n";
}
void draft_window_options() {
    const auto set=[](const char* value) {
#ifdef _WIN32
        _putenv_s("TYLOINFER_MTP_ATTENTION_WINDOW",value?value:"");
#else
        if(value)setenv("TYLOINFER_MTP_ATTENTION_WINDOW",value,1);else unsetenv("TYLOINFER_MTP_ATTENTION_WINDOW");
#endif
    };
    const auto* previous=std::getenv("TYLOINFER_MTP_ATTENTION_WINDOW");
    const std::optional<std::string> saved=previous?std::optional<std::string>(previous):std::nullopt;
    try {
        set(nullptr);
        if(runtime_options::mtp_attention_window())throw std::runtime_error("draft attention window changed its unset default");
        for(const auto* value:{"0","7","32768"}) {
            set(value);
            if(runtime_options::mtp_attention_window()!=std::optional<int64_t>(std::atoll(value)))
                throw std::runtime_error("explicit draft attention window was not preserved");
        }
        for(const auto* value:{"-1","abc","1x","9223372036854775808"}) {
            set(value);bool rejected=false;
            try{(void)runtime_options::mtp_attention_window();}catch(const std::invalid_argument&){rejected=true;}
            if(!rejected)throw std::runtime_error("invalid draft attention window was accepted");
        }
    }catch(...){set(saved?saved->c_str():nullptr);throw;}
    set(saved?saved->c_str():nullptr);
    std::cout<<"Draft attention window default/full/explicit/invalid options CPU-only PASS\n";
}
void causal_live(bool benchmark_only, const char* reference_path=nullptr, bool export_reference=false) {
    std::ifstream reference;
    std::ofstream exported;
    if(reference_path) {
        if(export_reference)exported.open(reference_path,std::ios::binary|std::ios::trunc);
        else reference.open(reference_path,std::ios::binary);
        if(export_reference?!exported:!reference)throw std::runtime_error("could not open causal attention reference");
    }
    constexpr int capacity=32768,width=256,kv_heads=2;
    auto keys=data({1,kv_heads,capacity,width},710,.17f).to(tb::kFloat16);
    auto values=data({1,kv_heads,capacity,width},711,.19f).to(tb::kFloat16);
    const auto stream=mfq_current_cuda_stream();
    int cases=0;
    for(int heads:benchmark_only?std::vector<int>{24}:std::vector<int>{6,24})
    for(int rows:benchmark_only?std::vector<int>{1}:std::vector<int>{1,2,3,4})
    for(bool half:benchmark_only?std::vector<bool>{false}:std::vector<bool>{false,true})
    for(int live:benchmark_only?std::vector<int>{512,4096,16384,capacity}:std::vector<int>{1,31,32,33,511,4096,capacity}) {
        auto query=data({1,heads,rows,width},712,.3f).to(half?tb::kFloat16:tb::kFloat32);
        auto gate=data({1,rows,heads,width},713,.7f);
        std::vector<int64_t> boundaries(rows);
        for(int row=0;row<rows;++row)boundaries[row]=std::max(0,live-rows+row);
        auto positions=tb::tensor(boundaries).to(tb::kCUDA);
        Tensor output;
        DecodeWindow graph(stream);
        graph.capture([&] {
            output=mfq_qwen4_exp::causal_gqa_attention_gate(query,keys,values,positions,gate,capacity,half);
        },[]{},[&]{output={};});
        if(benchmark_only) {
            for(int repeat=0;repeat<16;++repeat)graph.run();
            Event begin(0),end(0);begin.record(stream);
            for(int repeat=0;repeat<128;++repeat)graph.run();
            end.record(stream);end.synchronize();float elapsed=0;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin.get(),end.get()));
            std::cout<<"causal_live_prefix capacity="<<capacity<<" live="<<live<<" heads="<<heads
                <<" rows="<<rows<<" gpu_us="<<double(elapsed)*1000/128<<"\n";
            continue;
        }
        for(int replay=0;replay<2;++replay) {
            for(int row=0;row<rows;++row)boundaries[row]=std::max(0,live-rows+row-replay*3);
            positions.copy_(tb::tensor(boundaries).to(tb::kCUDA));
            graph.run();
            const auto actual=output.to(tb::kFloat32).contiguous().cpu();
            if(export_reference) {
                exported.write(reinterpret_cast<const char*>(actual.data_ptr<float>()),actual.numel()*sizeof(float));
                if(!exported)throw std::runtime_error("could not write causal attention reference");
                ++cases;continue;
            }
            Tensor expected;
            if(reference_path) {
                std::vector<float> saved(actual.numel());
                reference.read(reinterpret_cast<char*>(saved.data()),saved.size()*sizeof(float));
                if(!reference)throw std::runtime_error("causal attention reference is truncated");
                expected=tb::tensor(saved).reshape(actual.sizes().vec());
            }else expected=mfq_qwen4_exp::causal_gqa_attention_gate(query,keys,values,positions,gate,live,half)
                .to(tb::kFloat32).contiguous().cpu();
            for(int64_t i=0;i<actual.numel();++i) {
                const auto error=std::abs(double(actual.data_ptr<float>()[i])-expected.data_ptr<float>()[i]);
                const auto tolerance=reference_path?1.e-7:(half?.0002:.00002)+.001*std::abs(expected.data_ptr<float>()[i]);
                if(!std::isfinite(error) || error>tolerance) {
                    std::cerr<<"causal prefix mismatch heads="<<heads<<" rows="<<rows<<" half="<<half
                        <<" live="<<live<<" replay="<<replay<<" index="<<i<<" actual="<<actual.data_ptr<float>()[i]
                        <<" expected="<<expected.data_ptr<float>()[i]<<" error="<<error<<"\n";
                    throw std::runtime_error("captured causal attention differs from bounded-prefix reference");
                }
            }
            ++cases;
        }
    }
    if(reference_path&&!export_reference&&reference.peek()!=std::char_traits<char>::eof())
        throw std::runtime_error("causal attention reference has unused entries");
    if(!benchmark_only)std::cout<<"Causal MMA reserved/live prefix FP16/FP32 M1..4 changing graph replay cases="<<cases
        <<" frozen_reference="<<(reference_path!=nullptr)<<" export="<<export_reference<<" PASS\n";
}
void benchmark(bool half_cache) {
    constexpr int width=256,capacity=4096,count=2048,replays=128;
    for(int tokens:{1,2,3,4}) {
        auto query=data({1,tokens,24,2*width},530,.3f).narrow(3,0,width).permute({0,2,1,3});
        auto gate=data({1,tokens,24,2*width},531,.7f).narrow(3,width,width);
        auto key=data({1,2,capacity,width},532,.17f),value=data(key.sizes().vec(),533,.19f);
        if(half_cache){key=key.to(tb::kFloat16);value=value.to(tb::kFloat16);}
        auto indices=tb::arange(tokens*count,query.options().dtype(tb::kInt32)).remainder(capacity).reshape({1,tokens,count});
        Tensor output;auto compute=[&]{output=mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,value,indices,gate,true);};
        const auto stream=mfq_current_cuda_stream();
        {GraphWarmupScope warmup(default_context(mfq_current_cuda_device()),stream);compute();}
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));output={};
        Graph graph;graph.capture_begin_shared();compute();graph.capture_end();
        for(int i=0;i<32;++i)graph.replay();
        Event begin(0),end(0);begin.record(stream);for(int i=0;i<replays;++i)graph.replay();end.record(stream);end.synchronize();
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
        std::cout<<"qsa_strided_gate half_cache="<<half_cache<<" rows="<<tokens<<" us="<<double(ms)*1000/replays<<" K="<<count<<" replays="<<replays<<"\n";
    }
}
void prefill_benchmark() {
    constexpr int width=256,capacity=4096,selected=2048;
    for(int tokens:{512,4096}) {
        auto query=data({1,24,tokens,width},610,.3f);
        auto key=data({1,2,capacity,width},611,.17f).to(tb::kFloat16);
        auto value=data({1,2,capacity,width},612,.19f).to(tb::kFloat16);
        std::vector<int32_t> rows(tokens*selected);
        for(int t=0;t<tokens;++t)for(int k=0;k<selected;++k)
            rows[t*selected+k]=k<t+1?(t*13+k*17)%(t+1):-1;
        auto ids=tb::tensor(rows).reshape({1,tokens,selected}).to(tb::kCUDA);
        Tensor output;const auto compute=[&]{output=mfq_selected_attention::sparse(query,key,value,ids,1./16,false);};
        const auto stream=mfq_current_cuda_stream();
        {GraphWarmupScope warm(default_context(mfq_current_cuda_device()),stream);compute();}
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));output={};
        Graph graph;graph.capture_begin_shared();compute();graph.capture_end();
        for(int i=0;i<4;++i)graph.replay();
        Event begin(0),end(0);std::vector<float> samples(5);
        for(auto& sample:samples) {
            begin.record(stream);graph.replay();graph.replay();end.record(stream);end.synchronize();
            TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&sample,begin.get(),end.get()));sample/=2;
        }
        std::sort(samples.begin(),samples.end());
        std::cout<<"qsa_prefill rows="<<tokens<<" selected="<<selected<<" median_ms="<<samples[2]<<" trials=5\n";
    }
}
void qsa_sparse_gate_case() {
    int ordinary=0,changing=0;
    const auto bits=[](const Tensor& actual,const Tensor& expected) {
        auto x=actual.contiguous().cpu(),y=expected.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.numel()*x.element_size()))
            throw std::runtime_error("QSA sparse fused gate dtype/shape/bytes differ");
    };
    for(int tokens:{1,3,8})for(int count:{1,31,32,33,127,512,2051})for(int mode=0;mode<8;++mode) {
        const int batch=tokens==3?2:1,heads=count<127 || count==2051?24:48,kv=2,width=256,capacity=257;
        const auto gate_dtype=mode&1?tb::kFloat32:tb::kFloat16;
        const bool strided=mode&2,half_output=mode&4;
        auto query=data({batch,heads,tokens,width*2},381,.3f).slice(-1,0,width*2,2);
        if(mode&1)query=query.to(tb::kFloat32);
        auto key=data({batch,kv,capacity,width},382,.17f),value=data(key.sizes().vec(),383,.19f);
        if(mode&1){key=key.to(tb::kFloat16);value=value.to(tb::kFloat16);}
        auto gate=data({batch,tokens,heads,width*(strided?2:1)},384,88.f).to(gate_dtype);
        if(strided)gate=gate.slice(-1,0,width*2,2);
        std::vector<int64_t> host_indices(batch*tokens*count);
        for(size_t i=0;i<host_indices.size();++i)
            host_indices[i]=i%7==0?-1:i%11==0?capacity+3:int64_t((i*13+i/count)%capacity);
        auto indices=tb::tensor(host_indices).reshape({batch,tokens,count}).to(tb::kCUDA);
        if(!(mode&1))indices=indices.to(tb::kInt32);
        const auto reference=[&]{return mfq_qwen4_exp::attention_gate(
            mfq_qwen4_exp::sparse_gqa_attention(query,key,value,indices),gate,half_output);};
        const auto fused=[&]{return mfq_qwen4_exp::sparse_gqa_attention_gate(
            query,key,value,indices,gate,half_output);};
        bits(fused(),reference());++ordinary;
        Tensor result;DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{result=fused();},[]{},[&]{result={};});
        for(int replay=0;replay<3;++replay) {
            query.copy_(data(query.sizes().vec(),385+replay,replay==0?0.f:.31f).to(query.scalar_type()));
            gate.copy_(data(gate.sizes().vec(),388+replay,replay==2?0.f:88.f).to(gate_dtype));
            if(replay==0)indices.fill_(-1);
            else if(replay==1) {
                indices.copy_(tb::tensor(host_indices).reshape(indices.sizes()).to(tb::kCUDA).to(indices.scalar_type()));
                // Reuse the exact cache addresses with new contents before
                // graph replay, including the FP16 production cache layout.
                key.copy_(data(key.sizes().vec(),590,.13f).to(key.scalar_type()));
                value.copy_(data(value.sizes().vec(),591,.23f).to(value.scalar_type()));
            }
            else indices.fill_(capacity+7);
            window.run();bits(result,reference());++changing;
        }
    }
    // Preserve fallback behavior for other widths, BF16 gates and an empty KV cache.
    for(int width:{64,256})for(int capacity:{0,17}) {
        auto query=data({1,4,1,width},397,.3f),key=data({1,2,capacity,width},398,.17f);
        auto gate=data({1,1,4,width},399,88.f).to(tb::kBFloat16);
        auto indices=tb::zeros({1,1,33},query.options().dtype(tb::kInt32));
        for(bool half_output:{false,true}) {
            bits(mfq_qwen4_exp::sparse_gqa_attention_gate(query,key,key,indices,gate,half_output),
                mfq_qwen4_exp::attention_gate(mfq_qwen4_exp::sparse_gqa_attention(query,key,key,indices),gate,half_output));
            ++ordinary;
        }
    }
    std::cout<<"QSA sparse fused gate ordinary="<<ordinary<<" changing_graph="<<changing
        <<" bytes exact; partial/whole tiles, invalid/repeated indices, Half/Float strided query/gates, BF16/empty-cache fallbacks PASS\n";
}

void topk_selection(bool sanitizer=false) {
    const auto stream=mfq_current_cuda_stream();
    int cases=0;
    const auto run=[&](int pools,int rows,int pool,int count,int budget) {
        auto scores=data({1,rows,pools},733,.7f);
        auto positions=tb::zeros({rows},scores.options().dtype(tb::kInt64));
        Tensor output[2];Graph graphs[2];
        const auto compute=[&](int mode) {
            if(mode)output[1]=mfq_qwen4_exp::live_topk_selected_tokens(scores,positions,pool,count,budget);
            else output[0]=mfq_qwen4_exp::qsa_selected_tokens(
                mfq_qwen4_exp::live_topk_blocks(scores,positions,pool,count),positions,pool,budget);
        };
        for(int mode=0;mode<2;++mode) {
            {GraphWarmupScope warm(default_context(mfq_current_cuda_device()),stream);compute(mode);}
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));output[mode]={};
            graphs[mode].capture_begin_shared();compute(mode);graphs[mode].capture_end();
        }
        for(int ties:{0,1})for(int64_t start:{int64_t(-1),int64_t(0),int64_t(count)*pool-1,int64_t(count)*pool,
                int64_t(129)*pool-1,int64_t(257)*pool,int64_t(pools)*pool-2}) {
            if(ties)scores.zero_();
            std::vector<int64_t> live(rows);
            for(int row=0;row<rows;++row)live[row]=std::min(int64_t(pools)*pool-1,std::max(int64_t(-1),start+row));
            positions.copy_(tb::tensor(live).to(tb::kCUDA));
            for(int mode=0;mode<2;++mode){output[mode].fill_(-123456);graphs[mode].replay();}
            auto a=output[0].cpu().contiguous(),b=output[1].cpu().contiguous();
            if(a.sizes()!=b.sizes() || std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))
                throw std::runtime_error("Fused QSA top-k selection changed token IDs");
            ++cases;
        }

    };
    if(sanitizer) {
        run(1024,4,32,64,2048);
        run(8193,3,4,1024,4096);
    } else {
        for(int pools:{17,257,1024,8193})for(int rows:{1,4})for(int pool:{1,3,4,7,32}) {
            const int count=std::min(64,pools);
            run(pools,rows,pool,count,count*pool+(pool>1?3:0));
        }
        for(int rows:{1,4})run(8192,rows,4,512,2048);
        for(int count:{512,1024,1025})run(8193,3,4,count,count*4);
    }
    std::cout<<"QSA fused top-k/token selection cases="<<cases
        <<" changing graphs/ties/empty visibility/causal tails/multiple stages/large-count fallback exact PASS\n";
}

}
#include "rotary_shared_angles_test.h"
int main(int argc,char** argv) {
    if(argc==2 && std::string(argv[1])=="--draft-window-options-check") {
        try{draft_window_options();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess || !count)return 77;
    try {
        std::cout<<std::unitbuf;
        auto context=default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
        struct Release {std::shared_ptr<Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
        if(argc==2 && std::string(argv[1])=="--topk-select-check")topk_selection(false);
        else if(argc==2 && std::string(argv[1])=="--topk-select-sanitizer-check")topk_selection(true);
        else if(argc==2 && std::string(argv[1])=="--rotary-angle-check")rotary_angles_test::check();
        else if(argc==7 && std::string(argv[1])=="--rotary-angle-bench")rotary_angles_test::benchmark(
            std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]));
        else if(argc==7 && std::string(argv[1])=="--tile-pair-bench")qsa_tile_pair_benchmark(
            std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]));
        else if(argc==2 && (std::string(argv[1])=="--bench" || std::string(argv[1])=="--bench-f16"))benchmark(std::string(argv[1])=="--bench-f16");
        else if(argc==2 && std::string(argv[1])=="--prefill-bench")prefill_benchmark();
        else if(argc==2 && std::string(argv[1])=="--history-tail-check")history_tail_graph();
        else if(argc==2 && std::string(argv[1])=="--draft-window-check")draft_window_graph();
        else if(argc==2 && std::string(argv[1])=="--causal-live-check")causal_live(false);
        else if(argc==2 && std::string(argv[1])=="--resolved-hot-check")resolved_hot_reference();
        else if(argc==2 && std::string(argv[1])=="--causal-live-bench")causal_live(true);
        else if(argc==3 && std::string(argv[1])=="--causal-live-export")causal_live(false,argv[2],true);
        else if(argc==3 && std::string(argv[1])=="--causal-live-reference")causal_live(false,argv[2]);
        else{gate_reference();sparse_uniform_reference();general_query();qsa_sparse_gate_case();resolved_hot_reference();}
        return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
