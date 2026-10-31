#pragma once
#include "mfq_cuda_attention_ops.h"

namespace rotary_angles_test {
void setting(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value?value:"");
#else
    if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
struct Mode {
    bool present,reduce_present;std::string previous,reduce_previous;
    explicit Mode(const char* value):present(std::getenv("TYLOINFER_ROTARY_SHARED_ANGLES")!=nullptr),
        reduce_present(std::getenv("TYLOINFER_ROTARY_FUSED_REDUCTION")!=nullptr) {
        if(present)previous=std::getenv("TYLOINFER_ROTARY_SHARED_ANGLES");
        if(reduce_present)reduce_previous=std::getenv("TYLOINFER_ROTARY_FUSED_REDUCTION");
        setting("TYLOINFER_ROTARY_SHARED_ANGLES",value);setting("TYLOINFER_ROTARY_FUSED_REDUCTION",value);
    }
    ~Mode(){setting("TYLOINFER_ROTARY_SHARED_ANGLES",present?previous.c_str():nullptr);
        setting("TYLOINFER_ROTARY_FUSED_REDUCTION",reduce_present?reduce_previous.c_str():nullptr);}
};
void exact(const Tensor& actual,const Tensor& expected) {
    if(actual.defined()!=expected.defined())throw std::runtime_error("shared rotary output presence differs");
    if(!actual.defined())return;
    if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type())
        throw std::runtime_error("shared rotary output geometry differs");
    const auto a=actual.contiguous().cpu(),b=expected.contiguous().cpu();
    if(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))throw std::runtime_error("shared rotary output bits differ");
}
struct Job {
    std::vector<Tensor> inputs,weights,entry_positions;
    Tensor positions,frequencies,axes,key,value,projected_value,cache_positions;
    struct Result {std::vector<Tensor> outputs;Tensor key,value;};
    Result run(const char* setting) const {
        Mode selected(setting);Result result;
        if(key.defined()){result.key=key.clone();result.value=value.clone();}
        result.outputs=rotary_normalized_grouped_cuda(inputs,weights,positions,frequencies,axes,64,1e-6,1,
            result.key,cache_positions,projected_value,result.value,entry_positions);
        return result;
    }
};
Job make(int batches,int tokens,int width,bool half,bool mixed,bool cache,bool tail,
         const std::vector<int>& heads={4,2,3}) {
    Job job;const int count=3,pairs=width/4,position_axes=mixed?3:1,position_batches=batches==2?2:1;
    const auto type=half?tb::kFloat16:tb::kFloat32;
    for(int i=0;i<count;++i) {
        const int rows=tail && i!=1?1:tokens,columns=i==2?width/2:width;
        const auto selected=mixed && i==1?(half?tb::kFloat32:tb::kFloat16):type;
        job.inputs.push_back(data({batches,rows,heads[i],2*columns},700+i,.7f).to(selected).narrow(-1,0,columns));
        job.weights.push_back(data({columns},704+i,.13f));
    }
    std::vector<int64_t> positions(position_axes*position_batches*tokens);
    for(size_t i=0;i<positions.size();++i)positions[i]=i%4==0?-9:i%4==1?7:i%4==2?65536:((int64_t(1)<<40)+17);
    const bool wide=!mixed;
    if(!wide)for(auto& p:positions)if(p>INT32_MAX)p=INT32_MAX;
    job.positions=tb::tensor(positions).reshape({position_axes,position_batches,tokens}).to(tb::kCUDA).to(wide?tb::kInt64:tb::kInt32);
    std::vector<float> frequencies(pairs);std::vector<int32_t> axes(pairs);
    for(int i=0;i<pairs;++i){frequencies[i]=std::pow(10000.f,-float(i)/pairs);axes[i]=i%position_axes;}
    job.frequencies=tb::tensor(frequencies).to(tb::kCUDA);job.axes=tb::tensor(axes).to(tb::kCUDA);
    if(tail) {
        auto last=job.positions.narrow(-1,tokens-1,1).contiguous();
        job.entry_positions={last,job.positions,last};
    }
    if(cache) {
        const int capacity=tokens+3;
        job.key=tb::full({batches,heads[1],capacity,width},-.25,job.inputs[0].options().dtype(tb::kFloat16));
        job.value=tb::full(job.key.sizes(),-.375,job.key.options());
        job.projected_value=data({batches,tokens,heads[1],2*width},710,.3f).to(type).narrow(-1,0,width);
        std::vector<int64_t> slots(tokens);for(int i=0;i<tokens;++i)slots[i]=i%3==0?-1:i%3==1?1:capacity+3;
        job.cache_positions=tb::tensor(slots).to(tb::kCUDA).to(wide?tb::kInt32:tb::kInt64);
    }
    return job;
}
void compare(const Job::Result& actual,const Job::Result& expected) {
    if(actual.outputs.size()!=expected.outputs.size())throw std::runtime_error("shared rotary group count differs");
    for(size_t i=0;i<actual.outputs.size();++i)exact(actual.outputs[i],expected.outputs[i]);
    exact(actual.key,expected.key);exact(actual.value,expected.value);
}
void check() {
    int cases=0;
    for(bool half:{false,true})for(int width:{32,96,128,256,512})for(int tokens:{1,2,4})for(int batches:{1,2}) {
        auto job=make(batches,tokens,width,half,width==96,tokens==4,batches==2 && tokens==4);
        const auto reference=job.run("0");
        compare(job.run("1"),reference);compare(job.run(nullptr),reference);++cases;
        // Also exercise the single-input API, including a one-head input
        // which has no redundant angle work to share.
        for(int heads:{1,4}) {
            auto input=data({batches,tokens,heads,width},720,.4f).to(half?tb::kFloat16:tb::kFloat32);
            Tensor expected,actual;
            {Mode old("0");expected=rotary_normalized_cached_cuda(input,job.weights[0],job.positions,
                job.frequencies,job.axes,64,1e-6,{},{},{},{});}
            {Mode defaults(nullptr);actual=rotary_normalized_cached_cuda(input,job.weights[0],job.positions,
                job.frequencies,job.axes,64,1e-6,{},{},{},{});}
            exact(actual,expected);++cases;
        }
    }
    auto job=make(1,4,128,true,false,false,false);
    const auto stream=mfq_current_cuda_stream();const auto context=default_context(mfq_current_cuda_device());
    const bool owns_pool=!context->has_graph_pool(stream);
    if(owns_pool)context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;bool owns;
        ~Release(){if(owns)context->end_graph_pool(stream);}} release{context,stream,owns_pool};
    Job::Result reference,actual;
    DecodeWindow old_graph(stream),new_graph(stream);
    {Mode old("0");old_graph.capture([&]{reference=job.run("0");},[]{},[&]{reference={};});}
    {Mode forced("1");new_graph.capture([&]{actual=job.run("1");},[]{},[&]{actual={};});}
    if(new_graph.nodes()!=old_graph.nodes()+1)throw std::runtime_error("shared rotary did not capture exactly one shared angle producer");
    for(int replay=0;replay<4;++replay) {
        std::vector<int64_t> positions(4);for(int i=0;i<4;++i)positions[i]=replay*9+i;
        job.positions.copy_(tb::tensor(positions).reshape(job.positions.sizes()).to(tb::kCUDA));
        old_graph.run();new_graph.run();compare(actual,reference);
    }
    std::cout<<"Rotary fused reduction/shared angles exact eager cases="<<cases
        <<" mixed rows/types/positions/strides/KV and changing graph replay PASS\n"<<std::flush;
}
void benchmark(int tokens,int query_heads,int kv_heads,int index_heads,int width) {
    if(tokens<1 || query_heads<1 || kv_heads<1 || index_heads<1 || width<4 || width%4)
        throw std::invalid_argument("rotary angle benchmark geometry");
    auto job=make(1,tokens,width,true,false,false,false,{query_heads,kv_heads,index_heads});
    const auto stream=mfq_current_cuda_stream();const auto context=default_context(mfq_current_cuda_device());
    const bool owns_pool=!context->has_graph_pool(stream);
    if(owns_pool)context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;bool owns;
        ~Release(){if(owns)context->end_graph_pool(stream);}} release{context,stream,owns_pool};
    Job::Result old_output,new_output;DecodeWindow old_graph(stream),new_graph(stream);
    {Mode old("0");old_graph.capture([&]{old_output=job.run("0");},[]{},[&]{old_output={};});}
    {Mode defaults(nullptr);new_graph.capture([&]{new_output=job.run(nullptr);},[]{},[&]{new_output={};});}
    old_graph.run();new_graph.run();compare(new_output,old_output);
    for(int i=0;i<16;++i){old_graph.run();new_graph.run();}
    cudaEvent_t begin,end;
    TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples[2];
    for(int trial=0;trial<7;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;auto& graph=mode?new_graph:old_graph;
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<64;++i)if(!graph.enqueue_if_device_only())throw std::runtime_error("rotary angle benchmark acquired host tasks");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));samples[mode].push_back(1000.*elapsed/64);
    }
    for(auto& sample:samples)std::sort(sample.begin(),sample.end());
    std::cout<<"rotary_angles M="<<tokens<<" q_heads="<<query_heads<<" kv_heads="<<kv_heads
        <<" index_heads="<<index_heads<<" width="<<width<<" original_us="<<samples[0][3]
        <<" shared_us="<<samples[1][3]<<" exact=1 graph_nodes="<<old_graph.nodes()<<','<<new_graph.nodes()
        <<" median7=1 PASS\n"<<std::flush;
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}
}
