#include "diagnostics/tokens.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/generation/decode.h"
#include "engine/models/causal_model.h"
#include "engine/models/load.h"
#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/requests/output.h"
#include "engine/weights/cache.h"
#include "engine/weights/loader.h"
#include "components/io/file_range_reader.h"

#include <cuda_profiler_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace tyloi::cuda::diagnostics {
namespace {
using Clock = std::chrono::steady_clock;

/** Retain graph, inputs and outputs through graph retirement and owner drain. */
struct Work {
    Program& program;
    std::vector<Tensor> values;
    std::unique_ptr<tyloi::Executor::Graph> graph;
    ~Work() noexcept {
        if(graph) try { graph->invalidate(); } catch (...) {}
        try { program.drain(); }
        catch(const std::exception& error) { std::cerr<<"token_cleanup error="<<error.what()<<'\n'; }
        catch(...) { std::cerr<<"token_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) { values.push_back(value); return value; }
};

/** Retain one eager step and drain before its buffers unwind on failure. */
struct Step {
    Program& program;
    int exceptions = std::uncaught_exceptions();
    std::vector<Tensor> values;
    ~Step() noexcept {
        if(std::uncaught_exceptions()>exceptions) {
            try { program.drain(); }
            catch(const std::exception& error) { std::cerr<<"token_step_cleanup error="<<error.what()<<'\n'; }
            catch(...) { std::cerr<<"token_step_cleanup error=unknown\n"; }
        }
    }
    Tensor keep(Tensor value) { values.push_back(value); return value; }
};

/** Print source I/O counters without making optional reporting fatal. */
void report_source(const ModelSource& source,const char* phase) noexcept {
    const auto* setting=env_lookup("TYLOINFER_REPORT_FILE_IO");
    if(!setting || setting[0]!='1')return;
    const auto s=source.file_read_stats();
    try {
        std::cout<<"source_file_io phase="<<phase<<" mode="<<file_read_mode_name(s.mode)
            <<" files="<<s.files<<" calls="<<s.calls<<" logical_bytes="<<s.logical_bytes
            <<" physical_bytes="<<s.physical_bytes<<" errors="<<s.errors<<" staging_bytes="<<s.staging_bytes
            <<" staging_peak_sum_bytes="<<s.staging_peak_bytes<<" staging_cached_bytes="<<s.staging_cached_bytes
            <<" staging_allocations="<<s.staging_allocations<<" read_ms="<<double(s.read_nanoseconds)/1e6<<std::endl;
    } catch(...) {}
}

/** Flush the optional CUDA activity collector after diagnostic generation. */
void finish_activity() noexcept {
    if(!env_lookup("TYLOINFER_CUDA_ACTIVITY_PREFIX"))return;
#ifdef _WIN32
    const auto module=GetModuleHandleW(L"cuda_activity_collector.dll");
    const auto finish=module?reinterpret_cast<int(*)()>(GetProcAddress(module,"MFQCompleteCudaActivityTrace")):nullptr;
    std::fprintf(stderr,"native_cuda_activity_export status=%d\n",finish?finish():0);
#endif
}

/** Project the last hidden row with the original family-specific rounding and sampling. */
template<class Owner>
Tensor next_token(Owner& work, Tensor hidden, models::Backbone backbone) {
    auto last=work.keep(hidden.dense().select(1,hidden.size(1)-1));
    const bool mini=backbone==models::Backbone::minicpmo45;
    const bool direct=backbone==models::Backbone::qwen4_exp || backbone==models::Backbone::glm5_next;
    if(mini)last=work.keep(last.dense().to(kBFloat16).contiguous());
    else if(!direct)last=work.keep(last.dense().to(kFloat16).contiguous());
    auto logits=work.keep(work.program.model().logits(last,false));
    if(direct || mini)return work.keep(::tyloi::ops::argmax(logits,-1));
    return work.keep(::tyloi::ops::sample_greedy(work.keep(::tyloi::ops::reshape(::tyloi::ops::contiguous(logits),{logits.size(0),-1}))));
}

/** Follow the family's prefill and model-owned decode dispatch before eager fallback. */
Tensor eager(Step& work,Tensor ids,models::Backbone backbone,bool production_required = false) {
    auto& p=work.program;
    auto mode=p.executor().forward_mode(p.config().layer_placement.primary_device());
    models::Input input; input.ids=work.keep(std::move(ids)); input.advance=true;
    const models::ForwardOptions forward{mode.kv_capacity,mode.trace,mode.trace_layer,mode.primary};
    if(backbone==models::Backbone::qwen4_exp && p.state().position()>0) {
        models::DecodeOptions decode;
        decode.projection=models::DecodeOptions::Projection::greedy;
        decode.routed=p.weights().decode()==weights::Cache::Decode::routed;
        if(auto result=engine::decode(p.model(),p.state(),p.executor(),p.weights(),forward,input,decode)) {
            if(!result->ids)throw std::runtime_error("model decode returned no token IDs");
            auto token=work.keep(std::move(result->ids));
            return token.device()==p.executor().device() ? token : work.keep(::tyloi::ops::copy(token,p.executor().device()));
        }
    }
    if(backbone==models::Backbone::qwen4_exp && p.state().position()==0 &&
       input.ids.size(1)>128 && input.ids.size(1)<=16384) {
        const auto count=input.ids.size(1);
        const auto cache_mode=p.weights().prefill_mode(count,false);
        const models::PrefillOptions plan{count,1024,0,0,false,
            cache_mode==weights::Cache::Prefill::retained,
            p.weights().decode()==weights::Cache::Decode::routed,
            p.weights().transfer()==weights::Cache::Transfer::coordinated,
            p.weights().split_prefill(count)};
        if(!p.model().prefill_plan(p.state(),plan).layered) {
            if(production_required)
                throw std::invalid_argument("router audit requires production prefill; enable TYLOINFER_MOE_PIPELINE and TYLOINFER_MOE_PREFILL_LAYER");
            auto hidden=work.keep(p.model().forward(std::move(input),p.state(),forward));
            return next_token(work,hidden,backbone);
        }
        engine::InferenceRequest request;
        engine::InferenceOutput output(request,nullptr,"diagnostic-prefill");
        auto result=p.prefill(input,1024,output);
        auto hidden=work.keep(result.hidden); if(result.raw)work.keep(result.raw);
        if(production_required && result.mode!=models::Prefill::Mode::layered)
            throw std::invalid_argument("router audit requires production prefill; enable TYLOINFER_MOE_PIPELINE and TYLOINFER_MOE_PREFILL_LAYER");
        if(production_required)std::cout<<"diagnostic_prefill path=production_layer_major tokens="<<input.ids.size(1)<<std::endl;
        return next_token(work,hidden,backbone);
    }
    if(production_required && backbone==models::Backbone::qwen4_exp && input.ids.size(1)>128)
        throw std::invalid_argument("router audit requires production prefill; enable TYLOINFER_MOE_PIPELINE and TYLOINFER_MOE_PREFILL_LAYER");
    auto hidden=work.keep(p.model().forward(std::move(input),p.state(),
        forward));
    return next_token(work,hidden,backbone);
}
}

void warmup_prefill(Program& program,const Tensor& ids,bool profile,int rounds,bool final_sync) {
    auto scope=program.executor().enter(program.config().layer_placement.primary_device());
    auto loader=program.loader();
    const auto* source=dynamic_cast<const ModelSource*>(&loader.source);
    if(!source)throw std::runtime_error("diagnostics model source is unavailable");
    const auto backbone=models::model_plan(models::graph(*source)).backbone;
    auto& profiler=program.executor().profiler();
    struct Restore { CudaProfiler& p; bool enabled; ~Restore(){p.enabled=enabled;} } restore{profiler,profiler.enabled};
    profiler.enabled=false;
    if(rounds<0)rounds=profile?1:0;
    const auto start=Clock::now();
    for(int round=0;round<rounds;++round) {
        Step step{program};
        program.state().reset(1);
        (void)eager(step,ids,backbone);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
    }
    if(rounds || final_sync) { program.state().reset(1); if(final_sync) TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream())); profiler.reset(); }
    std::cout<<"warmup_prefill_rounds="<<rounds<<"\nwarmup_prefill_tokens="<<rounds*ids.size(1)
        <<"\nwarmup_prefill_sec="<<std::chrono::duration<double>(Clock::now()-start).count()<<'\n';
}

int generate_diagnostic_tokens(Program& program,Tensor ids,int gen,bool profile,
    Clock::time_point t0,Clock::time_point t1,const std::vector<std::int64_t>& forced,
    const std::unordered_set<std::int64_t>& stops,bool production_prefill,TokenOptions options) {
    if(gen<0)throw std::invalid_argument("diagnostic generation count must be nonnegative");
    if(!forced.empty() && (gen<=1 || forced.size()!=std::size_t(gen-1)))
        throw std::invalid_argument("diagnostic decode input count differs from gen-1");
    const auto loader=program.loader();
    const auto* source=dynamic_cast<const ModelSource*>(&loader.source);
    if(!source)throw std::runtime_error("diagnostics model source is unavailable");
    const auto plan=models::model_plan(models::graph(*source));
    const bool flash=plan.backbone==models::Backbone::qwen4_exp ||
        plan.backbone==models::Backbone::glm5_next;
    const auto info=program.model().info();
    auto scope=program.executor().enter(program.config().layer_placement.primary_device());
    Work work{program};
    ids=work.keep(std::move(ids));
    Tensor forced_input;
    auto& profiler=program.executor().profiler();
    struct Restore { CudaProfiler& p; bool enabled,graph; ~Restore(){p.enabled=enabled;p.graph_events=graph;} } restore{profiler,profiler.enabled,profiler.graph_events};
    if(!options.already_warmed)warmup_prefill(program,ids,profile,options.warmup);
    if(!forced.empty())forced_input=work.keep(::tyloi::ops::tensor(forced,program.executor().device()));
    std::cout<<"decode_input_mode="<<(forced_input?"forced":"generated")<<"\n";
    profiler.enabled=profile;
    report_source(*source,"prefill_begin"); profiler.reset();
    Tensor next;
    {
    Step step{program};
    next=eager(step,ids,plan.backbone,production_prefill);
    }
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
    const auto t2=Clock::now();
    report_cuda_memory(program.config().report_cuda_memory,"prefill"); report_source(*source,"prefill");
    if(const auto* setting=env_lookup("TYLOINFER_EMPTY_CACHE_BEFORE_GRAPH");setting && std::atoi(setting)!=0) {
        mfq_cuda_empty_cache(); report_cuda_memory(program.config().report_cuda_memory,"prefill_empty_cache");
    }
    profiler.report("prefill"); profiler.reset();
    if(gen==0)return 0;
    int produced=1;
    bool eos=!stops.empty() && stops.contains(::tyloi::ops::item(next));
    auto generated=work.keep(::tyloi::ops::empty({gen},TensorOptions().dtype(kInt64).device(program.executor().device())));
    const auto stream=current_stream().stream();
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(generated.dense().data_ptr<std::int64_t>(),next.dense().data_ptr<std::int64_t>(),sizeof(std::int64_t),cudaMemcpyDeviceToDevice,stream));
    const auto* graph_option=env_lookup("TYLOINFER_CUDA_GRAPH");
    const auto* profile_option=env_lookup("TYLOINFER_PROFILE_CUDA_GRAPH");
    const bool profile_graph=profile && profile_option && std::atoi(profile_option)!=0;
    const auto& config=program.config();
    const bool graph_requested=!graph_option || graph_option[0]!='0';
    const bool graph_eligible=graph_requested && !forced_input && plan.backbone!=models::Backbone::generic_qwen && !flash &&
        mfq_cuda_graph_capture_supported() &&
        config.dsv4_cpu_offload_layers.empty() && config.dense_cpu_layer_count==0 &&
        program.weights().transfer()!=weights::Cache::Transfer::coordinated &&
        info.graph==models::GraphOwner::executor &&
        ::tyloi::cuda::internal::graph_allowed(config,program.executor(),program.weights(),info) &&
        (!profile || profile_graph) && gen>1;
    if(graph_requested && graph_eligible && !eos)work.graph=program.executor().graph();
    const bool use_graph=bool(work.graph);
    const auto* range_option=env_lookup("TYLOINFER_CUDA_PROFILER_RANGE");
    const bool range=range_option && std::atoi(range_option)!=0;
    if(range)TYLOINFER_CUDA_CHECK(cudaProfilerStart());
    struct StopProfiler { bool active; ~StopProfiler(){if(active)cudaProfilerStop();} } stop_profiler{range};
    auto replay_start=t2, first=t2, halfway=t2, last_window=t2;
    bool measured_first=false,measured_half=false,measured_last=false;
    if(use_graph) {
        auto graph_scope=work.graph->enter();
        auto options=TensorOptions().dtype(kInt64).device(program.executor().device());
        auto static_input=work.keep(::tyloi::ops::empty({1,1},options));
        auto static_pos=work.keep(::tyloi::ops::empty({1},options));
        auto static_len=work.keep(::tyloi::ops::empty({1},options));
        auto static_step=work.keep(::tyloi::ops::empty({1},options));
        auto graph_stream=current_stream().stream();
        const std::int64_t pos=program.state().position(),len=pos+1,one=1;
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_input.dense().data_ptr<std::int64_t>(),next.dense().data_ptr<std::int64_t>(),sizeof(pos),cudaMemcpyDeviceToDevice,graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_pos.dense().data_ptr<std::int64_t>(),&pos,sizeof(pos),cudaMemcpyHostToDevice,graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_len.dense().data_ptr<std::int64_t>(),&len,sizeof(len),cudaMemcpyHostToDevice,graph_stream));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(static_step.dense().data_ptr<std::int64_t>(),&one,sizeof(one),cudaMemcpyHostToDevice,graph_stream));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream));
        auto states=program.state().graph_state();
        work.graph->prepare(states);
        const auto planned=pos+gen;
        const auto parts=::tyloi::ops::attention_parts(planned,program.executor().device());
        auto mode=program.executor().forward_mode(config.layer_placement.primary_device());
        const auto static_forward=[&] {
            models::Input input; input.ids=static_input; input.positions=static_pos; input.lengths=static_len;
            input.cache_positions=static_pos; input.kv_length=planned; input.attention_parts=parts;
            return next_token(work,work.keep(program.model().forward(std::move(input),program.state(),
                {mode.kv_capacity,mode.trace,mode.trace_layer,mode.primary})),plan.backbone);
        };
        const auto warmup_checkpoint=work.values.size();
        if(!states.empty() || info.double_warmup)for(int pass=0;pass<2;++pass) {
            try { (void)static_forward(); work.graph->restore(); }
            catch(...) { work.graph->restore(); throw; }
            if(program.state().position()!=pos)throw std::runtime_error("static decode warmup changed cache position");
            work.values.resize(warmup_checkpoint);
        }
        profiler.reset(); profiler.graph_events=profile_graph;
        work.graph->begin();
        auto static_next=profiler.measure("decode.model_total",[&]{return static_forward();});
        profiler.measure("decode.commit",[&]{::tyloi::ops::decode_commit(static_next,generated,static_step,static_input,static_pos,static_len);return 0;});
        work.graph->end(program.state().storage());
        report_cuda_memory(config.report_cuda_memory,"graph_captured");
        replay_start=Clock::now();
        for(int i=1;i<gen;++i) {
            work.graph->replay(); produced=i+1;
            if(!stops.empty()) { work.graph->wait(); if(stops.contains(::tyloi::ops::item(static_next))) { eos=true; break; } }
        }
        work.graph->wait();
    } else if(!eos) {
        for(int i=1;i<gen;++i) {
            Step step{program};
            auto input=step.keep(::tyloi::ops::reshape(forced_input?step.keep(::tyloi::ops::slice(forced_input,0,i-1,i)):next,{1,1}));
            next=profiler.measure("decode.eager_model",[&]{return eager(step,input,plan.backbone);});
            profiler.measure("decode.eager_commit",[&]{
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(generated.dense().data_ptr<std::int64_t>()+i,
                    next.dense().data_ptr<std::int64_t>(),sizeof(std::int64_t),cudaMemcpyDeviceToDevice,stream)); return 0;});
            produced=i+1;
            if(!stops.empty() && stops.contains(::tyloi::ops::item(next))) { eos=true; break; }
            if(i==1 && flash && stops.empty()) { TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream())); first=Clock::now(); measured_first=true; }
            if(gen>=4 && i==gen/2 && flash && stops.empty()) { TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream())); halfway=Clock::now(); measured_half=true; }
            if(gen>256 && i==gen-1-128 && flash && stops.empty()) { TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream())); last_window=Clock::now(); measured_last=true; }
        }
    }
    std::cout<<"diagnostic_cuda_graph requested="<<int(graph_requested)
        <<" eligible="<<int(graph_eligible)<<" used="<<int(use_graph)<<'\n';
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
    if(range) { TYLOINFER_CUDA_CHECK(cudaProfilerStop()); stop_profiler.active=false; }
    const auto t3=Clock::now();
    report_cuda_memory(config.report_cuda_memory,"decode_complete"); profiler.report("decode");
    auto host=generated.dense().narrow(0,0,produced).to(kCPU).contiguous();
    const auto* data=host.data_ptr<std::int64_t>();
    const double load_s=std::chrono::duration<double>(t1-t0).count();
    const double prefill_s=std::chrono::duration<double>(t2-t1).count();
    const double decode_s=std::chrono::duration<double>(t3-t2).count();
    const double replay_s=std::chrono::duration<double>(t3-replay_start).count();
    std::cout<<"load_sec="<<load_s<<"\nprefill_sec="<<prefill_s<<"\nfinish_reason="<<(eos?"stop":"length")
        <<"\ndecode_tokens="<<std::max(0,produced-1)<<"\ndecode_setup_sec="<<decode_s-replay_s<<"\ndecode_replay_sec="<<replay_s<<'\n';
    if(measured_first) { const auto head=std::chrono::duration<double>(first-t2).count(), tail=std::chrono::duration<double>(t3-first).count();
        std::cout<<"decode_first_step_sec="<<head<<"\ndecode_after_first_tokens="<<std::max(0,produced-2)<<"\ndecode_after_first_sec="<<tail<<'\n';
        if(produced>2)std::cout<<"decode_after_first_tok_per_s="<<double(produced-2)/tail<<'\n'; }
    if(measured_half) { const int head=produced/2,tail=produced-1-head; const auto hs=std::chrono::duration<double>(halfway-t2).count(),ts=std::chrono::duration<double>(t3-halfway).count();
        std::cout<<"decode_head_tokens="<<head<<"\ndecode_head_sec="<<hs<<"\ndecode_tail_tokens="<<tail<<"\ndecode_tail_sec="<<ts<<"\ndecode_tail_tok_per_s="<<double(tail)/ts<<'\n'; }
    if(measured_last) { const auto seconds=std::chrono::duration<double>(t3-last_window).count();
        std::cout<<"decode_last_128_tokens=128\ndecode_last_128_begin_cache_position="<<ids.size(1)+produced-1-128<<"\ndecode_last_128_sec="<<seconds<<"\ndecode_last_128_tok_per_s="<<128./seconds<<'\n'; }
    std::cout<<"decode_sec="<<decode_s<<'\n';
    if(produced>1)std::cout<<"decode_tok_per_s="<<double(produced-1)/decode_s<<"\ndecode_steady_tok_per_s="<<double(produced-1)/replay_s<<'\n';
    std::cout<<"generated_ids="; for(int i=0;i<produced;++i) { if(i)std::cout<<',';std::cout<<data[i]; } std::cout<<'\n';
    report_source(*source,"decode"); program.weights().report(std::cout);
    finish_activity();
    return 0;
}
} // namespace tyloi::cuda::diagnostics
