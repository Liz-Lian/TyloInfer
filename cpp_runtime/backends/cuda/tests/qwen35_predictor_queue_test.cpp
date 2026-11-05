#include "storage/model_loader.h"
#include "models/qwen35/mtp.h"
#include "mfq_cuda_sampling_ops.h"
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

namespace tb=mfq_tensor_backend;
using ReferencePredictor=Qwen35Mtp;
static void require(bool condition,const char* message) {
    if(!condition)throw std::runtime_error(message);
}
static void exact(const tb::Tensor& a,const tb::Tensor& b,const char* message) {
    auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
    require(x.sizes()==y.sizes() && x.scalar_type()==y.scalar_type() &&
        std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())==0,message);
}

int main(int argc,char** argv)try {
    // Real weights are optional test fixtures, supplied explicitly by the caller.
    if(argc!=3) {std::cout<<"Supply target GGUF and compatible native MTP GGUF\n";return 77;}
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    std::cout<<std::unitbuf;tb::NoGradGuard no_grad;
    auto context=mfq::cuda::default_context();
    CudaExecutionContext execution;constexpr int capacity=64;
    execution.kl_mmq.kv_cache_capacity=capacity;
    auto source=std::make_shared<mfq::GgufModelSource>(argv[1],
        std::vector<std::filesystem::path>{argv[2]});
    auto model=mfq::cuda::load_causal_lm<mfq::cuda::Qwen35CausalLm>(execution,
        argv[1],"",capacity,false,false,source,true);
    model.config.legacy_tensor_layout.norm_weight_offset=0.;
    model.config.legacy_tensor_layout.linear_attention_a_is_log=false;
    model.config.legacy_tensor_layout.qwen_gdn_gguf_layout=true;
    model.metadata.norm_weight_offset=0.;
    model.adapter_load_final_state(*source,model.output_norm);model.adapter_prepare_blocks(*source);
    auto reference=ReferencePredictor::load_if_present(*source,model.config,execution);
    auto candidate=Qwen35Mtp::load_if_present(*source,model.config,execution);
    require(reference.has_value() && candidate.has_value(),"Native predictor weights unavailable");
    auto& sync=*reference;auto& queued=*candidate;MtpTarget target(model);
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    auto host=tb::empty({1,4,model.hidden_size()},cpu);
    for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.3f*std::sin(float(i)*.017f);
    auto hidden=host.to(tb::kCUDA);
    auto ids=tb::tensor(std::vector<int64_t>{17,29,53,71},cpu.dtype(tb::kInt64)).reshape({1,4}).to(tb::kCUDA);
    sync.reset(1);queued.reset(1);
    sync.history_step(target,hidden,ids);queued.history_step(target,hidden,ids);
    auto base=sync.capture_session_state(5,hidden.narrow(1,3,1));
    auto queued_base=queued.capture_session_state(5,hidden.narrow(1,3,1));
    int cases=0;
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int history:{1,2,4})
    for(int depth:{2,3,4,7})for(bool positioned:{false,true}) {
        auto h=hidden.narrow(1,0,history).to(dtype),input=ids.narrow(1,0,history);
        auto pos=positioned?tb::arange(11,11+history,ids.options()):tb::Tensor{};
        sync.restore_session_state(base);queued.restore_session_state(queued_base);
        auto tail=sync.history_tail_step(target,h,input,pos);
        auto step=tail?std::move(*tail):sync.step_positioned(target,h,input,pos);
        std::vector<int32_t> expected;
        for(int i=0;i<depth;++i) {
            // Each scalar proposal reaches the CPU before the next forward.
            auto token=sample_greedy_cuda(step.sample_logits.contiguous()).cpu();
            const int64_t value=*token.data_ptr<int64_t>();expected.push_back(int32_t(value));
            if(i+1<depth) {
                auto next=tb::tensor(std::vector<int64_t>{value},cpu.dtype(tb::kInt64)).reshape({1,1}).to(tb::kCUDA);
                auto pp=positioned?tb::arange(11+history+i,12+history+i,ids.options()):tb::Tensor{};
                step=sync.step_positioned(target,step.chain_hidden.narrow(1,step.chain_hidden.size(1)-1,1),next,pp);
            }
        }
        const auto end=sync.cache_pos;
        tb::Tensor verification_ids;
        auto proposals=queued.enqueue_greedy_chain(target,h,input,pos,depth,&verification_ids);
        require(proposals.has_value() && proposals->is_cuda(),"Queued proposals left the device");
        // A device consumer precedes the combined readback, as in target
        // verification; the predictor may not require an intermediate wait.
        auto packet=tb::cat({*proposals,verification_ids.reshape({-1})},0);
        if(cases%3==0)queued.predictor_graphs.reset();
        auto host=packet.cpu().contiguous();
        require(host.numel()==2*depth+1,"Queued device consumer returned an incomplete packet");
        const auto* values=host.data_ptr<int64_t>();
        require(values[depth]==std::array<int64_t,4>{17,29,53,71}[history-1],"Queued target anchor differs");
        for(int i=0;i<depth;++i)require(values[i]==expected[i] && values[i+depth+1]==expected[i],
            "Queued proposals or their device consumer differ from scalar proposals");
        require(queued.cache_pos==end,"Queued cache position differs");
        auto a=sync.capture_session_state(end+1,hidden.narrow(1,3,1));
        auto b=queued.capture_session_state(end+1,hidden.narrow(1,3,1));
        require(a.blocks.size()==b.blocks.size(),"Predictor state count differs");
        for(size_t j=0;j<a.blocks.size();++j) {
            exact(a.blocks[j].k,b.blocks[j].k,"Queued K state differs");
            exact(a.blocks[j].v,b.blocks[j].v,"Queued V state differs");
        }
        auto follow_hidden=hidden.narrow(1,3,1).to(dtype),follow_ids=ids.narrow(1,0,1);
        auto expected_logits=sync.step_positioned(target,follow_hidden,follow_ids,{}).sample_logits.cpu();
        auto actual_logits=queued.step_positioned(target,follow_hidden,follow_ids,{}).sample_logits;
        if(cases%2==0)queued.predictor_graphs.reset(); // Release a graph with queued consumers.
        exact(expected_logits,actual_logits,"Follow-up logits differ after queued replay or graph release");
        ++cases;
        if(cases%8==0)std::cout<<"Real predictor scalar/queued exact cases="<<cases<<'\n';
    }
    queued.restore_session_state(queued_base);const auto before=queued.cache_pos;
    require(!queued.queued_greedy_chain(target,hidden,ids,{},capacity),"Invalid depth accepted");
    require(queued.cache_pos==before,"Invalid queued call changed state");
    // Reset an outstanding single-row replay and immediately reuse its graph.
    (void)queued.step_positioned(target,hidden.narrow(1,0,1),ids.narrow(1,0,1),{});
    queued.reset(1);sync.reset(1);
    queued.history_step(target,hidden,ids);sync.history_step(target,hidden,ids);
    exact(sync.step_positioned(target,hidden.narrow(1,0,1),ids.narrow(1,0,1),{}).sample_logits.cpu(),
        queued.step_positioned(target,hidden.narrow(1,0,1),ids.narrow(1,0,1),{}).sample_logits,
        "Reset with an outstanding predictor replay changed logits");
    std::cout<<"Real predictor queue PASS exact_cases="<<cases<<" invalid_depth=1 pending_reset=1\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
