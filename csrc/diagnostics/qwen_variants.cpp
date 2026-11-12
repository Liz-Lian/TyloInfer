#include "diagnostics/qwen_variants.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/config.h"
#include "engine/generation/batch_driver.h"
#include "engine/models/causal_model.h"
#include "engine/models/load.h"
#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/requests/output.h"
#include "engine/weights/loader.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
/** Restore one temporarily selected runtime option. */
struct Environment {
    const char* name; std::string previous; bool present;
    explicit Environment(const char* key):name(key),present(env_lookup(key)!=nullptr) {
        if(present)previous=env_lookup(key);
    }
    void set(const char* value) { if(value)env_set(name,value);else env_clear(name); }
    ~Environment() noexcept { try { if(present)env_set(name,previous.c_str());else env_clear(name); } catch(...) {} }
};

/** Retire each batch and its outstanding tensors after draining the program. */
struct BatchWork {
    Program& program;
    bool active=false;
    std::vector<Tensor> tensors;
    ~BatchWork() noexcept {
        if(active)try { program.end_batch(); }
        catch(const std::exception& error) { std::cerr<<"variant_batch_cleanup error="<<error.what()<<'\n'; }
        catch(...) { std::cerr<<"variant_batch_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) { tensors.push_back(value); return value; }
    void finish() { program.end_batch(); active=false; tensors.clear(); }
};
/** Keep one submitted batch's inputs alive until failures have drained. */
struct BatchStep {
    Program& program;
    int exceptions=std::uncaught_exceptions();
    std::vector<Tensor> tensors;
    ~BatchStep() noexcept {
        if(std::uncaught_exceptions()>exceptions)try { program.drain(); } catch(...) {}
    }
    Tensor keep(Tensor value) { tensors.push_back(value); return value; }
};
}

int run_qwen4_generation_variants(Program& program,const std::vector<std::int64_t>& prompt) {
    if(prompt.empty())throw std::runtime_error("MTP generation variants require explicit prompt IDs");
    const auto loader=program.loader();
    const auto* source=dynamic_cast<const ModelSource*>(&loader.source);
    if(!source || models::graph(*source).backbone!="qwen4_exp" || !program.model().mtp())
        throw std::runtime_error("Qwen4 MTP generation variants require a loaded predictor");
    auto scope=program.executor().enter(program.config().layer_placement.primary_device());
    Environment compact("TYLOINFER_MFE_TOKEN_COMPACT_PLAN"),aligned("TYLOINFER_MFE_TOKEN_ALIGNED_GU"),
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
    const auto* gr_option=tyloi::cuda::env_lookup("TYLOINFER_GR_INTEGER_VARIANTS");
    const bool gr_only=gr_option && gr_option[0]=='1';
    const auto* dense_option=tyloi::cuda::env_lookup("TYLOINFER_NINT_DENSE_VARIANTS");
    const bool dense_only=dense_option && dense_option[0]=='1';
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_NINT_VARIANTS");
    const bool nint_only=option && option[0]=='1';
    const auto* short_option=tyloi::cuda::env_lookup("TYLOINFER_MTP_SHORT_CAPACITY_VARIANTS");
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
    const auto* decode_option=tyloi::cuda::env_lookup("TYLOINFER_SHORT_DECODE_VARIANTS");
    const bool mtp_batch2=decode_option && std::strcmp(decode_option,"mtp-batch2")==0;
    const bool batch2=decode_option && (std::strcmp(decode_option,"batch2")==0 || std::strcmp(decode_option,"all")==0 || mtp_batch2);
    const auto modes=decode_option && (std::strcmp(decode_option,"both")==0 || std::strcmp(decode_option,"all")==0)?std::vector<bool>{false,true}:
        mtp_batch2?std::vector<bool>{true}:batch2?std::vector<bool>{}:
        decode_option && std::strcmp(decode_option,"ordinary")==0?std::vector<bool>{false}:std::vector<bool>{true};

    for(bool enable_mtp:modes)for(std::size_t index=0;index<variant_count;++index) {
        const auto& variant=active[index];
        compact.set(variant.compact);aligned.set(variant.aligned);up.set(variant.up);
        capacity.set(variant.capacity);down.set("0");exact_batch.set(variant.exact);separated.set(variant.separated);direct_dma.set(variant.direct);
        gdn_short.set(variant.gdn);gdn_inplace.set(variant.gdn);short_interval.set(variant.interval);
        q8_decode.set(variant.q8);gr_integer.set(variant.gr);gr_up_integer.set(variant.gr_up);gr_pair.set(variant.gr_pair);
        program.configure(config);
        for(int repeat=0;repeat<(index==0 || (nint_only || short_capacity) && index==1?2:1);++repeat) {
            engine::InferenceRequest request;request.prompt=prompt;
            request.sampling.max_tokens=1024;request.sampling.temperature=0.;
            request.sampling.top_k=1;request.sampling.top_p=1.;request.sampling.enable_mtp=enable_mtp;
            request.sampling.mtp_max_draft_tokens=5;
            engine::InferenceOutput output(request,nullptr,"mtp-operator-variants");
            std::vector<std::int64_t> generated;
            std::chrono::steady_clock::time_point first_token;
            auto job=program.generate(request,output,enable_mtp,"mtp-operator-variants");
            while(auto step=job.next())if(step.value) {
                if(auto* delta=std::get_if<engine::OutputDelta>(&*step.value)) {
                    if(generated.empty())first_token=std::chrono::steady_clock::now();
                    generated.insert(generated.end(),delta->token_ids.begin(),delta->token_ids.end());
                }
            }
            if(output.cleanup_failure)std::rethrow_exception(output.cleanup_failure);
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
            const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-first_token).count();
            if(generated.size()!=1024 || ms<=0.)throw std::runtime_error("MTP variant generation was incomplete");
            if(expected.empty())expected=generated;
            const bool exact=generated==expected;
            const auto* allowed=env_lookup("TYLOINFER_BENCH_ALLOW_MTP_OUTPUT_DIFFERENCE");
            if(!exact && !(enable_mtp && allowed && allowed[0]=='1'))
                throw std::runtime_error(std::string("MTP generation variant changed output: ")+variant.name);
            const auto stats=program.stats();
            nlohmann::json sample={{"variant",variant.name},{"repeat",repeat},{"tokens",generated.size()},{"mtp",enable_mtp},
                {"decode_ms",ms},{"decode_tps",1023.*1000./ms},{"exact",exact},
                {"cycles",stats.cycles},{"accepted",stats.accepted},{"drafted",stats.request.drafted_tokens},
                {"selected_depth",stats.request.selected_depth},{"depth_cycles",stats.request.depth_cycles}};
            std::cout<<"mtp_generation_variant "<<sample.dump()<<'\n';
        }
    }
    if(batch2) {
        std::array<std::vector<std::int64_t>,2> batch_expected;
        const auto prompt_ids=::tyloi::ops::reshape(::tyloi::ops::tensor(prompt,program.executor().device()),{1,-1});
        for(std::size_t index=0;index<variant_count;++index) {
            const auto& variant=active[index];
            compact.set(variant.compact);aligned.set(variant.aligned);up.set(variant.up);capacity.set(variant.capacity);
            down.set("0");exact_batch.set(variant.exact);separated.set(variant.separated);direct_dma.set(variant.direct);
            gdn_short.set(variant.gdn);gdn_inplace.set(variant.gdn);short_interval.set(variant.interval);
            q8_decode.set(variant.q8);gr_integer.set(variant.gr);gr_up_integer.set(variant.gr_up);gr_pair.set(variant.gr_pair);
            for(int repeat=0;repeat<(index<2?2:1);++repeat) {
                program.drain();program.state().reset(1);
                BatchWork work{program};
                models::BatchConfig options; options.slots=2;options.graph=true;options.greedy=true;
                auto& driver=program.begin_batch(options);work.active=true;
                std::array<models::BatchState*,2> states{};
                std::vector<std::int64_t> next;
                std::array<std::vector<std::int64_t>,2> generated;
                for(int row=0;row<2;++row) {
                    states[row]=&driver.add(prompt.size());
                    auto hidden=work.keep(driver.prefill(*states[row],prompt_ids));
                    auto last=work.keep(::tyloi::ops::select(hidden,1,-1));
                    auto logits=work.keep(program.model().logits(last,false));
                    auto sampled=work.keep(::tyloi::ops::argmax(logits,-1));
                    next.push_back(::tyloi::ops::item(sampled));generated[row].push_back(next.back());
                }
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
                const auto started=std::chrono::steady_clock::now();
                for(int step=1;step<512;++step) {
                    BatchStep step_owner{program};
                    auto host=step_owner.keep(::tyloi::ops::reshape(step_owner.keep(::tyloi::ops::tensor(next,Device(kCPU))),{2,1}));
                    auto ids=step_owner.keep(::tyloi::ops::copy(host,program.executor().device()));
                    std::vector<models::BatchRow> rows;
                    for(int row=0;row<2;++row)rows.push_back({states[row],next[row],states[row]->position(),512-step,true,true});
                    auto result=driver.capture(rows,ids,host,true);
                    if(!result)throw std::runtime_error("short-batch variant must use a CUDA graph");
                    next=::tyloi::ops::integers(step_owner.keep(*result));
                    if(next.size()!=2)throw std::runtime_error("short-batch graph returned invalid output");
                    for(int row=0;row<2;++row)generated[row].push_back(next[row]);
                }
                const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
                for(int row=0;row<2;++row) {
                    if(states[row]->position()!=std::int64_t(prompt.size())+511)throw std::runtime_error("short batch position changed");
                    if(!expected.empty() && !std::equal(generated[row].begin(),generated[row].end(),expected.begin()))
                        throw std::runtime_error("short batch changed the ordinary decode response");
                    if(batch_expected[row].empty())batch_expected[row]=generated[row];
                    else if(batch_expected[row]!=generated[row])throw std::runtime_error(std::string("short batch variant changed output: ")+variant.name);
                }
                nlohmann::json sample={{"variant",variant.name},{"repeat",repeat},{"tokens",1024},{"batch",2},
                    {"mtp",false},{"decode_ms",ms},{"decode_tps",1022.*1000./ms},{"exact",true}};
                std::cout<<"mtp_generation_variant "<<sample.dump()<<'\n';
                work.finish();
            }
        }
    }
    return 0;
}
}
