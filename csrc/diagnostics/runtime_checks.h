#pragma once
#include "env.h"

#include "engine/generation/cuda.h"
#include "engine/config.h"
#include "engine/backends/cuda/native/scope.h"
#include "generation_result.h"
#include "engine/models/deepseek_v4/ops.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/nemotron_h/mtp.h"
#include "engine/models/common/mtp.h"
#include "engine/models/nemotron_h/radio_vision.h"
#include "diagnostics/token_generation.h"
#include "engine/models/qwen35/mtp.h"
#include "engine/models/qwen35/forward.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/weights/loader.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/models/session/cache.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/qwen35/load.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

namespace mfq::cuda::diagnostics {

using namespace mfq::cuda::internal;

static int run_qwen4_decode_hidden_check(mfq::cuda::Qwen4CausalLm& model) {
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(model.max_position_embeddings()>=24 && model.vocab_size()>=8,
        "Qwen4 hidden graph check requires context>=24 and vocab>=8");
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    const auto prompt=tb::tensor(std::vector<int64_t>{1,2},options).reshape({1,2});
    const auto ids_for=[&](int step) {
        return tb::tensor(std::vector<int64_t>{3+step%5},options).reshape({1,1});
    };
    const auto host=[](const tb::Tensor& value) {return value.to(tb::kFloat32).cpu().contiguous();};
    std::vector<tb::Tensor> reference_hidden,reference_raw,retained_raw;
    model.reset(1);model.hidden_forward(prompt);
    for(int step=0;step<20;++step) {
        tb::Tensor raw;
        auto hidden=model.hidden_forward(ids_for(step),mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&raw);
        reference_hidden.push_back(host(hidden));reference_raw.push_back(host(raw));
    }
    double maximum_error=0;
    const auto compare=[&](const tb::Tensor& value,const tb::Tensor& reference) {
        const auto actual=host(value);
        TYLOINFER_RUNTIME_CHECK(actual.sizes().vec()==reference.sizes().vec(),"hidden graph shape mismatch");
        const auto* a=actual.data_ptr<float>();const auto* b=reference.data_ptr<float>();
        for(int64_t i=0;i<actual.numel();++i) {
            const double error=std::abs(double(a[i])-b[i]);
            maximum_error=std::max(maximum_error,error);
            TYLOINFER_RUNTIME_CHECK(std::isfinite(a[i]) && error<=.015+.015*std::abs(b[i]),
                "hidden graph differs from eager execution");
        }
    };
    model.reset(1);model.hidden_forward(prompt);
    int graph_steps=0;
    for(int step=0;step<20;++step) {
        tb::Tensor raw,hidden;
        if(step==7) {
            // Equal-width scalar history also checks eager/graph transitions.
            // Verification windows are compared independently below: scalar
            // and batched quantized projections may round differently.
            hidden=model.hidden_forward(ids_for(step),mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&raw);
        } else {
            auto result=qwen4_decode_graph_hidden(model,ids_for(step),&raw);
            TYLOINFER_RUNTIME_CHECK(result.has_value(),"hidden graph was not exercised");
            hidden=std::move(*result);++graph_steps;
        }
        compare(hidden,reference_hidden[step]);compare(raw,reference_raw[step]);
        retained_raw.push_back(std::move(raw));
        TYLOINFER_RUNTIME_CHECK(model.cache_pos==step+3,"hidden graph advanced KV more than once");
    }
    for(int step=0;step<20;++step)compare(retained_raw[step],reference_raw[step]);
    TYLOINFER_RUNTIME_CHECK(model.graph_registry!=nullptr,"hidden graph did not survive KV growth");
    std::cout<<"qwen4_decode_hidden_check PASS graph_steps="<<graph_steps
        <<" retained_steps="<<retained_raw.size()<<" max_abs_error="<<maximum_error<<'\n';
    int transactions=0;
    for(int tokens=2;tokens<=6;++tokens) {
        std::vector<int64_t> values;for(int i=0;i<tokens;++i)
            values.push_back(i==1 && model.vocab_size()>15?15:3+i%5);
        const auto input=tb::tensor(values,options).reshape({1,tokens});
        std::vector<tb::Tensor> expected,expected_raw,expected_next;
        const std::vector<int> actions{0,-1,tokens>2?1:0,tokens>3?2:-1};
        model.reset(1);model.hidden_forward(prompt);
        for(int accepted:actions) {
            tb::Tensor raw;
            expected.push_back(host(model.hidden_forward(input,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&raw,1)));
            expected_raw.push_back(host(raw));
            TYLOINFER_RUNTIME_CHECK(!qwen4_decode_graph_hidden(model,ids_for(0),nullptr),
                "hidden graph accepted an active speculative transaction");
            if(accepted<0)model.commit_speculative();else if(model.speculative_prefix_retention) {
                model.rollback_speculative(accepted);
            } else {
                model.rollback_speculative();
                if(accepted)model.hidden_forward(input.narrow(1,1,accepted));
            }
            expected_next.push_back(host(model.hidden_forward(ids_for(0))));
        }
        model.reset(1);model.hidden_forward(prompt);
        for(std::size_t i=0;i<actions.size();++i) {
            tb::Tensor raw,logits;std::vector<int32_t> greedy;
            const bool project=i%2==0;
            auto hidden=qwen4_decode_graph_hidden(model,input,&raw,1,nullptr,input.cpu(),project?&logits:nullptr,
                project?&greedy:nullptr);
            TYLOINFER_RUNTIME_CHECK(hidden.has_value() && model.speculative_start>=0 && model.speculative_confirmed==1,
                "MTP graph did not open its verification transaction");
            compare(*hidden,expected[i]);compare(raw,expected_raw[i]);
            if(project) {
                compare(logits,host(model.logits_from_hidden(*hidden)));
                auto tokens=logits.argmax(-1).cpu().contiguous();
                TYLOINFER_RUNTIME_CHECK(greedy.size()==static_cast<size_t>(input.size(1)),"graph greedy row count changed");
                for(size_t row=0;row<greedy.size();++row)
                    TYLOINFER_RUNTIME_CHECK(greedy[row]==tokens.data_ptr<int64_t>()[row],"graph returned stale greedy tokens");
            }
            if(actions[i]<0)model.commit_speculative();else {
                TYLOINFER_RUNTIME_CHECK(model.speculative_prefix_retention,"MTP graph omitted prefix checkpoints");
                model.rollback_speculative(actions[i]);
            }
            auto next=qwen4_decode_graph_hidden(model,ids_for(0),nullptr);
            TYLOINFER_RUNTIME_CHECK(next.has_value(),"MTP graph did not resume after commit/rollback");
            compare(*next,expected_next[i]);++transactions;
        }
    }
    std::cout<<"qwen4_mtp_graph_check PASS transactions="<<transactions
        <<" max_abs_error="<<maximum_error<<'\n';
    return 0;
}

    struct MtpProfileEnvironment {
        const char* name;std::string previous;bool present;
        explicit MtpProfileEnvironment(const char* key):name(key),present(mfq::cuda::env_lookup(key)!=nullptr) {
            if(present)previous=mfq::cuda::env_lookup(key);
        }
        void set(const char* value) {
            if(value)mfq::cuda::env_set(name,value);else mfq::cuda::env_clear(name);
        }
        ~MtpProfileEnvironment() {
            if(present)mfq::cuda::env_set(name,previous.c_str());else mfq::cuda::env_clear(name);
        }
    };

static int run_qwen4_mtp_generation_variants(mfq::cuda::Qwen4CausalLm& model,MtpModule& mtp,
        const std::vector<int64_t>& prompt) {
    TYLOINFER_RUNTIME_CHECK(!prompt.empty(),"MTP generation variants require explicit prompt IDs");
    mfq::engine::mtp::State mtp_state;
    MtpProfileEnvironment compact("TYLOINFER_MFE_TOKEN_COMPACT_PLAN"),aligned("TYLOINFER_MFE_TOKEN_ALIGNED_GU"),
        down("TYLOINFER_GR_BATCH_SHARED_DOWN"),up("TYLOINFER_GR_BATCH_SHARED_UP"),capacity("TYLOINFER_MFE_TOKEN_GROUP_CAPACITY"),
        exact_batch("TYLOINFER_NINT_EXACT_BATCH"),separated("TYLOINFER_MFE_SEPARATE_HIDDEN_QUANT"),
        direct_dma("TYLOINFER_MOE_DIRECT_CACHE_DMA"),gdn_short("TYLOINFER_GDN_SHORT_FUSION"),
        gdn_inplace("TYLOINFER_GDN_SHORT_INPLACE"),short_interval("TYLOINFER_MOE_SHORT_DECODE_INTERVAL"),
        q8_decode("TYLOINFER_NINT_Q8_DECODE"),gr_integer("TYLOINFER_GR_INT_DOT"),gr_up_integer("TYLOINFER_GR_UP_INT_DOT"),
        gr_pair("TYLOINFER_GR_INT_PAIR_DOWN");
    struct Variant {const char* name;const char* compact;const char* aligned;const char* up;const char* capacity;
        const char* exact=nullptr;const char* separated=nullptr;const char* direct=nullptr;
        const char* gdn=nullptr;const char* interval=nullptr;const char* q8=nullptr;const char* gr=nullptr;const char* gr_up=nullptr;const char* gr_pair=nullptr;};
    const Variant variants[]={
        {"baseline","0","0","0","2"},
        {"compact","1",nullptr,"0","2"},
        {"paired_up","1",nullptr,"1","2"},
        {"even_up","1",nullptr,"2","2"},
        {"capacity4","1",nullptr,"2","4"},
        {"capacity8","1",nullptr,"2","8"},
        {"baseline_repeat","0","0","0","2"}};
    const Variant nint_variants[]={
        {"nint_original","1",nullptr,"1","2","0"},
        {"nint_exact_batch","1",nullptr,"1","2","1"},
        {"nint_original_repeat","1",nullptr,"1","2","0"}};
    const Variant short_capacity_variants[]={
        {"short_capacity2","1",nullptr,nullptr,"2"},
        {"short_capacity_auto","1",nullptr,nullptr,"auto"},
        {"short_capacity2_repeat","1",nullptr,nullptr,"2"},
        {"short_capacity_auto_repeat","1",nullptr,nullptr,"auto"}};
    const Variant shared_capacity_variants[]={
        {"routed2_shared2","1",nullptr,nullptr,"2"},
        {"routed2_shared4","1",nullptr,nullptr,"shared4"},
        {"routed2_shared8","1",nullptr,nullptr,"shared8"},
        {"routed2_shared2_repeat","1",nullptr,nullptr,"2"}};
    const Variant quant_variants[]={
        {"inline_hidden_quant","1",nullptr,nullptr,"2",nullptr,"0"},
        {"separate_hidden_quant","1",nullptr,nullptr,"2",nullptr,"1"},
        {"inline_hidden_quant_repeat","1",nullptr,nullptr,"2",nullptr,"0"},
        {"separate_hidden_quant_repeat","1",nullptr,nullptr,"2",nullptr,"1"}};
    const Variant direct_variants[]={
        {"wire_payload_dma","1",nullptr,nullptr,"2",nullptr,"1","0"},
        {"direct_cache_dma","1",nullptr,nullptr,"2",nullptr,"1","1"},
        {"wire_payload_dma_repeat","1",nullptr,nullptr,"2",nullptr,"1","0"},
        {"direct_cache_dma_repeat","1",nullptr,nullptr,"2",nullptr,"1","1"}};
    const Variant short_path_variants[]={
        {"short_path_legacy","1",nullptr,nullptr,"2",nullptr,"1","0","0","0"},
        {"short_path_fused","1",nullptr,nullptr,"2",nullptr,"1","0","1","1"},
        {"short_path_legacy_repeat","1",nullptr,nullptr,"2",nullptr,"1","0","0","0"},
        {"short_path_fused_repeat","1",nullptr,nullptr,"2",nullptr,"1","0","1","1"}};
    const Variant dense_variants[]={
        {"q8_chunk","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","0"},
        {"q8_group","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","2"},
        {"q8_chunk_repeat","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","0"},
        {"q8_group_repeat","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","2"}};
    const Variant gr_integer_variants[]={
        {"gr_float","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","2","0","0"},
        {"gr_integer24","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","2","3","0","0"},
        {"gr_integer_best","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","2",nullptr,nullptr},
        {"gr_float_repeat","1",nullptr,nullptr,"2",nullptr,"1",nullptr,"1","1","2","0","0"}};
    const auto* gr_option=mfq::cuda::env_lookup("TYLOINFER_GR_INTEGER_VARIANTS");
    const bool gr_only=gr_option && gr_option[0]=='1';
    const auto* dense_option=mfq::cuda::env_lookup("TYLOINFER_NINT_DENSE_VARIANTS");
    const bool dense_only=dense_option && dense_option[0]=='1';
    const auto* option=mfq::cuda::env_lookup("TYLOINFER_MTP_NINT_VARIANTS");
    const bool nint_only=option && option[0]=='1';
    const auto* short_option=mfq::cuda::env_lookup("TYLOINFER_MTP_SHORT_CAPACITY_VARIANTS");
    const bool short_capacity=short_option && short_option[0]>='1' && short_option[0]<='5';
    const auto* capacity_variants=short_option && short_option[0]=='5'?short_path_variants:
        short_option && short_option[0]=='4'?direct_variants:
        short_option && short_option[0]=='3'?quant_variants:
        short_option && short_option[0]=='2'?shared_capacity_variants:short_capacity_variants;
    const auto* active=gr_only?gr_integer_variants:dense_only?dense_variants:nint_only?nint_variants:short_capacity?capacity_variants:variants;
    const auto variant_count=gr_only?std::size(gr_integer_variants):dense_only?std::size(dense_variants):nint_only?std::size(nint_variants):
        short_capacity?std::size(short_capacity_variants):std::size(variants);
    const auto config=tyloi::engine::resolve_config();
    std::vector<int64_t> expected;
    std::cout<<std::unitbuf;
    const auto* decode_option=mfq::cuda::env_lookup("TYLOINFER_SHORT_DECODE_VARIANTS");
    const bool mtp_batch2=decode_option && std::strcmp(decode_option,"mtp-batch2")==0;
    const bool batch2=decode_option && (std::strcmp(decode_option,"batch2")==0 || std::strcmp(decode_option,"all")==0 || mtp_batch2);
    const auto modes=decode_option && (std::strcmp(decode_option,"both")==0 || std::strcmp(decode_option,"all")==0)?std::vector<bool>{false,true}:
        mtp_batch2?std::vector<bool>{true}:batch2?std::vector<bool>{}:
        decode_option && std::strcmp(decode_option,"ordinary")==0?std::vector<bool>{false}:std::vector<bool>{true};
    for(bool enable_mtp:modes)
    for(std::size_t index=0;index<variant_count;++index) {
        const auto& variant=active[index];
        compact.set(variant.compact);aligned.set(variant.aligned);up.set(variant.up);
        capacity.set(variant.capacity);down.set("0");exact_batch.set(variant.exact);separated.set(variant.separated);direct_dma.set(variant.direct);
        gdn_short.set(variant.gdn);gdn_inplace.set(variant.gdn);short_interval.set(variant.interval);
        q8_decode.set(variant.q8);gr_integer.set(variant.gr);gr_up_integer.set(variant.gr_up);gr_pair.set(variant.gr_pair);
        DecodeGraphCache graph(model.max_position_embeddings());
        TextSessionCache cache(config.session_cache,config.prefix_cache);
        // Use precisely the production generation dispatch. Resetting the
        // model on each request also invalidates kernels captured for an old
        // variant; neither draft depth nor acceptance is forced.
        for(int repeat=0;repeat<(index==0 || (nint_only || short_capacity) && index==1?2:1);++repeat) {
            mfq::engine::InferenceRequest request;request.prompt=prompt;
            request.sampling.max_tokens=1024;request.sampling.temperature=0.;
            request.sampling.top_k=1;request.sampling.top_p=1.;request.sampling.enable_mtp=enable_mtp;
            request.sampling.mtp_max_draft_tokens=5;
            mfq::engine::InferenceOutput output(request,nullptr,"mtp-operator-variants");
            auto result=collect_generation(generate(
                model, model, nullptr, model.execution->collectives,
                model.execution->forward_mode(), model.execution->profiler,
                model.execution->weights,
                model.execution->decode_graph_serial_branches,
                model.execution->config.report_cuda_memory,
                model.execution->dsv4_cpu_offload_layers.empty() &&
                    model.execution->dense_cpu_layer_count == 0 &&
                    model_parallel_cuda_graph_enabled(*model.execution),
                [owner = model.execution] { owner->drain(); },
                model.metadata, graph, cache, config, request, output,
                mtp_state, enable_mtp ? &mtp : nullptr));
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
            const auto decode_ms=std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-result.first_token).count();
            TYLOINFER_RUNTIME_CHECK(result.tokens.size()==1024 && decode_ms>0.,"MTP variant generation was incomplete");
            if(expected.empty())expected=result.tokens;
            const bool exact=result.tokens==expected;
            const auto* allow_difference=mfq::cuda::env_lookup("TYLOINFER_BENCH_ALLOW_MTP_OUTPUT_DIFFERENCE");
            TYLOINFER_RUNTIME_CHECK(exact || (enable_mtp && allow_difference && allow_difference[0]=='1'),
                "MTP generation variant changed output: ",variant.name);
            nlohmann::json sample={{"variant",variant.name},{"repeat",repeat},{"tokens",result.tokens.size()},{"mtp",enable_mtp},
                {"decode_ms",decode_ms},{"decode_tps",1023.*1000./decode_ms},{"exact",exact},
                {"cycles",mtp_state.cycles},{"accepted",mtp_state.accepted},{"drafted",mtp_state.stats.drafted_tokens},
                {"selected_depth",mtp_state.stats.selected_depth},{"depth_cycles",mtp_state.stats.depth_cycles}};
            std::cout<<"mtp_generation_variant "<<sample.dump()<<'\n';
        }
    }
    if(batch2) {
        namespace tb=mfq_tensor_backend;
        std::array<std::vector<int64_t>,2> batch_expected;
        const auto options=tb::TensorOptions().dtype(tb::kInt64);
        const auto device=tb::Device(tb::kCUDA,model.execution->layer_placement.primary_device());
        const auto prompt_tensor=tb::tensor(prompt,options).reshape({1,-1}).to(device);
        for(std::size_t index=0;index<variant_count;++index) {
            const auto& variant=active[index];
            compact.set(variant.compact);aligned.set(variant.aligned);up.set(variant.up);capacity.set(variant.capacity);
            down.set("0");exact_batch.set(variant.exact);separated.set(variant.separated);direct_dma.set(variant.direct);
            gdn_short.set(variant.gdn);gdn_inplace.set(variant.gdn);short_interval.set(variant.interval);
            q8_decode.set(variant.q8);gr_integer.set(variant.gr);gr_up_integer.set(variant.gr_up);gr_pair.set(variant.gr_pair);
            for(int repeat=0;repeat<(index<2?2:1);++repeat) {
                model.reset(1);
                std::array<std::unique_ptr<mfq::cuda::Qwen4BatchRequestState>,2> states;
                std::vector<mfq::cuda::Qwen4BatchRequestState*> requests;
                std::vector<int64_t> next;
                std::array<std::vector<int64_t>,2> generated;
                for(int row=0;row<2;++row) {
                    states[row]=std::make_unique<mfq::cuda::Qwen4BatchRequestState>(model);
                    requests.push_back(states[row].get());
                    auto hidden=states[row]->prefill(model,prompt_tensor);
                    next.push_back(model.adapter_next_token(model.lm_head,model.last_hidden(std::move(hidden))).item<int64_t>());
                    generated[row].push_back(next.back());
                }
                mfq::cuda::Qwen4BatchDecodeGraph graph(model);
                cudaStreamSynchronize(::tyloi::cuda::current_stream());
                const auto started=std::chrono::steady_clock::now();
                for(int step=1;step<512;++step) {
                    auto host_ids=tb::tensor(next,options).reshape({2,1});
                    auto ids=host_ids.to(device);
                    auto result=graph.run(requests,ids,host_ids,true);
                    TYLOINFER_RUNTIME_CHECK(result.has_value(),"short-batch variant must use a CUDA graph");
                    auto host=result->cpu().contiguous();
                    next.assign(host.data_ptr<int64_t>(),host.data_ptr<int64_t>()+2);
                    for(int row=0;row<2;++row)generated[row].push_back(next[row]);
                }
                const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                for(int row=0;row<2;++row) {
                    TYLOINFER_RUNTIME_CHECK(states[row]->position()==int64_t(prompt.size())+511,"short batch position changed");
                    if(!expected.empty())TYLOINFER_RUNTIME_CHECK(std::equal(generated[row].begin(),generated[row].end(),expected.begin()),
                        "short batch changed the ordinary decode response");
                    if(batch_expected[row].empty())batch_expected[row]=generated[row];
                    else TYLOINFER_RUNTIME_CHECK(batch_expected[row]==generated[row],"short batch variant changed output: ",variant.name);
                }
                nlohmann::json sample={{"variant",variant.name},{"repeat",repeat},{"tokens",1024},{"batch",2},
                    {"mtp",false},{"decode_ms",ms},{"decode_tps",1022.*1000./ms},{"exact",true}};
                std::cout<<"mtp_generation_variant "<<sample.dump()<<'\n';
            }
        }
    }
    return 0;
}

static int run_qwen4_mtp_variant_profile(mfq::cuda::Qwen4CausalLm& model,const std::vector<int64_t>& prompt_ids) {
    namespace tb=mfq_tensor_backend;
    MtpProfileEnvironment compact("TYLOINFER_MFE_TOKEN_COMPACT_PLAN"),aligned("TYLOINFER_MFE_TOKEN_ALIGNED_GU"),
      gr("TYLOINFER_GR_BATCH_SHARED_DOWN"),capacity("TYLOINFER_MFE_TOKEN_GROUP_CAPACITY"),
      gdn("TYLOINFER_GDN_SHORT_FUSION"),inplace("TYLOINFER_GDN_SHORT_INPLACE");
    struct Variant {const char* name;const char* compact;const char* aligned;const char* gr;const char* capacity;const char* gdn=nullptr;};
    const Variant variants[]={
        {"baseline","0","0","0","2"},
        {"compact","1","0","0","2"},
        {"aligned","1","1","0","2"},
        {"shared_gr","1","1","1","2"},
        {"capacity4","1","1","1","4"},
        {"compact_shared_gr","1","0","1","2"},
        {"baseline_repeat","0","0","0","2"}};
    const Variant short_variants[]={
        {"short_legacy","1",nullptr,nullptr,"2","0"},
        {"short_fused","1",nullptr,nullptr,"2","1"},
        {"short_fused_repeat","1",nullptr,nullptr,"2","1"},
        {"short_legacy_repeat","1",nullptr,nullptr,"2","0"}};
    const auto* short_option=mfq::cuda::env_lookup("TYLOINFER_MTP_PROFILE_VARIANTS");
    const bool short_path=short_option && short_option[0]=='2';
    const auto* active=short_path?short_variants:variants;
    const auto count=short_path?std::size(short_variants):std::size(variants);
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    const auto prompt=tb::tensor(prompt_ids.empty()?std::vector<int64_t>{1,2,3,4,5,6,7}:prompt_ids,options).reshape({1,-1});
    std::array<tb::Tensor,5> expected_hidden,expected_logits;
    for(std::size_t arm=0;arm<count;++arm)for(int tokens=2;tokens<=6;++tokens) {
        const auto& variant=active[arm];
        compact.set(variant.compact);aligned.set(variant.aligned);gr.set(variant.gr);capacity.set(variant.capacity);
        gdn.set(variant.gdn);inplace.set(variant.gdn);
        model.reset(1);model.execution->profiler.enabled=false;model.hidden_forward(prompt);
        std::vector<int64_t> values;for(int i=0;i<tokens;++i)values.push_back(3+i%5);
        const auto input=tb::tensor(values,options).reshape({1,tokens});
        double elapsed=0;tb::Tensor raw,logits;
        for(int repeat=0;repeat<10;++repeat) {
            cudaStreamSynchronize(::tyloi::cuda::current_stream());const auto start=std::chrono::steady_clock::now();
            std::vector<int32_t> greedy;
            auto result=qwen4_decode_graph_hidden(model,input,&raw,1,nullptr,{},&logits,&greedy);
            TYLOINFER_RUNTIME_CHECK(result.has_value(),"MTP variant profile did not use the graph");
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(repeat>=3)elapsed+=ms;
            model.rollback_speculative();
        }
        const auto index=tokens-2;
        auto hidden_cpu=raw.cpu().contiguous(),logits_cpu=logits.cpu().contiguous();
        if(arm==0) {
            expected_hidden[index]=hidden_cpu.clone();expected_logits[index]=logits_cpu.clone();
        }else {
            TYLOINFER_RUNTIME_CHECK(std::memcmp(hidden_cpu.data_ptr(),expected_hidden[index].data_ptr(),
                hidden_cpu.numel()*hidden_cpu.element_size())==0,"MTP variant changed raw hidden: ",variant.name);
            TYLOINFER_RUNTIME_CHECK(std::memcmp(logits_cpu.data_ptr(),expected_logits[index].data_ptr(),
                logits_cpu.numel()*logits_cpu.element_size())==0,"MTP variant changed target logits: ",variant.name);
        }
        std::cout<<"mtp_verify_variant name="<<variant.name<<" tokens="<<tokens
            <<" repeats=7 wall_avg_ms="<<elapsed/7<<" exact=1\n";
    }
    return 0;
}

static int run_qwen4_mtp_profile(mfq::cuda::Qwen4CausalLm& model,const std::vector<int64_t>& prompt_ids) {
    if(const auto* variants=mfq::cuda::env_lookup("TYLOINFER_MTP_PROFILE_VARIANTS");variants && (variants[0]=='1' || variants[0]=='2'))
        return run_qwen4_mtp_variant_profile(model,prompt_ids);
    namespace tb=mfq_tensor_backend;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
    auto prompt=tb::tensor(prompt_ids.empty()?std::vector<int64_t>{1,2,3,4,5,6,7}:prompt_ids,options).reshape({1,-1});
    auto& profiler=model.execution->profiler;
    std::cout<<std::unitbuf;
    for(bool graph:{false,true})for(int tokens=2;tokens<=6;++tokens) {
        model.reset(1);profiler.enabled=false;model.hidden_forward(prompt);
        std::vector<int64_t> values;for(int i=0;i<tokens;++i)values.push_back(3+i%5);
        const auto input=tb::tensor(values,options).reshape({1,tokens});
        double elapsed=0;
        for(int repeat=0;repeat<8;++repeat) {
            profiler.enabled=!graph && repeat>=3;profiler.reset();
            cudaStreamSynchronize(::tyloi::cuda::current_stream());const auto start=std::chrono::steady_clock::now();
            tb::Tensor raw;
            if(graph) {
                auto result=qwen4_decode_graph_hidden(model,input,&raw,1);
                TYLOINFER_RUNTIME_CHECK(result.has_value(),"MTP profile did not use the graph");
            } else model.hidden_forward(input,mfq_nullopt,mfq_nullopt,nullptr,mfq_nullopt,&raw,1);
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            if(repeat>=3)elapsed+=ms;
            if(profiler.enabled)profiler.report("mtp_verify_m"+std::to_string(tokens));
            model.rollback_speculative();
        }
        std::cout<<"mtp_verify_profile tokens="<<tokens<<" graph="<<graph<<" repeats=5 wall_avg_ms="<<elapsed/5<<'\n';
        if(graph)qwen4_report_decode_graph(model,std::cout);
    }
    profiler.enabled=false;profiler.reset();return 0;
}

} // namespace mfq::cuda::diagnostics
