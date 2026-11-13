#include "diagnostics/prefill.h"

#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/config.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/loader.h"
#include "engine/requests/result.h"
#include "engine/requests/output.h"
#include "engine/ops.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <variant>

namespace {
using tyloi::Tensor;
using tyloi::cuda::Program;
using Clock = std::chrono::steady_clock;

/** Retain asynchronous inputs and outputs until their owning executor drains. */
struct Work {
    Program& program;
    std::vector<Tensor> values;
    tyloi::ops::Trace trace;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& e) { std::cerr << "prefill_cleanup error=" << e.what() << '\n'; }
        catch (...) { std::cerr << "prefill_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) { values.push_back(value); return value; }
    void clear() { program.drain(); values.clear(); trace.clear(); }
};

/** Compute the original last-token capped logits from normalized hidden rows. */
Tensor project(Work& work, Tensor hidden) {
    auto last=work.keep(hidden.dense().select(1,hidden.size(1)-1).to(tyloi::kFloat16).contiguous());
    return work.keep(work.program.model().logits(last,true));
}

/** Run one ordinary forward and retain its inputs and results. */
Tensor forward(Work& work, const Tensor& ids, tyloi::ops::Trace* trace=nullptr) {
    auto& program=work.program;
    auto mode=program.executor().forward_mode(program.config().layer_placement.primary_device());
    tyloi::models::Input input; input.ids=ids; input.advance=true;
    return work.keep(program.model().forward(std::move(input),program.state(),
        {mode.kv_capacity,trace?trace:mode.trace,trace?0:mode.trace_layer,mode.primary}));
}

/** Synchronize the primary stream at the original timing boundary. */
void sync() {
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream()));
}

/** Name the exact scalar encoding used by a contiguous native observation. */
const char* dtype(tyloi::ScalarType type) {
    switch(type) {
        case tyloi::ScalarType::boolean: return "bool";
        case tyloi::ScalarType::uint8: return "uint8";
        case tyloi::ScalarType::int8: return "int8";
        case tyloi::ScalarType::int16: return "int16";
        case tyloi::ScalarType::int32: return "int32";
        case tyloi::ScalarType::int64: return "int64";
        case tyloi::ScalarType::float16: return "float16";
        case tyloi::ScalarType::bfloat16: return "bfloat16";
        case tyloi::ScalarType::float32: return "float32";
        case tyloi::ScalarType::float64: return "float64";
        default: throw std::runtime_error("unsupported prefill trace dtype");
    }
}

/** Export one final-chunk observation set without replacing any existing result. */
void write_trace(Work& work, const std::filesystem::path& root,
        std::int64_t tokens, std::int64_t chunk, int repeat) {
    std::filesystem::create_directories(root.parent_path());
    if(!std::filesystem::create_directory(root))
        throw std::runtime_error("prefill trace directory already exists: "+root.string());
    nlohmann::json metadata={{"format","tyloinfer.prefill-trace.v1"},{"tokens",tokens},
        {"chunk",chunk},{"repeat",repeat},{"layer",0},{"stages",nlohmann::json::array()}};
    for(const auto& [name,tensor]:work.trace) {
        auto host=work.keep(tensor.dense().contiguous().to(tyloi::kCPU));
        work.program.drain();
        const auto value=host.dense();
        const auto file=std::to_string(metadata["stages"].size())+".bin";
        std::ofstream output(root/file,std::ios::binary);
        if(!output)throw std::runtime_error("cannot create prefill trace tensor: "+file);
        output.write(static_cast<const char*>(value.data_ptr()),
            static_cast<std::streamsize>(value.nbytes()));
        output.close();
        if(!output)throw std::runtime_error("cannot write prefill trace tensor: "+file);
        metadata["stages"].push_back({{"name",name},{"file",file},{"dtype",dtype(value.scalar_type())},
            {"shape",value.sizes().vec()},{"nbytes",value.nbytes()}});
    }
    std::ofstream output(root/"trace.json");
    output<<metadata.dump(2)<<'\n';
    output.close();
    if(!output)throw std::runtime_error("cannot write prefill trace metadata: "+root.string());
    std::cout<<"prefill_trace path="<<root.string()<<" stages="<<work.trace.size()<<std::endl;
}

/** Read and validate every logit before deriving the first maximum. */
struct Checked {
    std::vector<float> values;
    std::vector<std::int64_t> shape;
};

Checked checked(const Tensor& logits) {
    Checked result;
    result.values=tyloi::ops::floats(logits);
    for(std::int64_t i=0;i<logits.dim();++i)result.shape.push_back(logits.size(i));
    for(std::size_t i=0;i<result.values.size();++i)
        if(!std::isfinite(result.values[i]))throw std::runtime_error("nonfinite prefill logits at index "+std::to_string(i));
    return result;
}

/** Return the first maximum token from finite logits. */
std::int64_t top(const Checked& input) {
    const auto& values=input.values;
    if(values.empty())throw std::runtime_error("prefill logits are empty");
    return std::max_element(values.begin(),values.end())-values.begin();
}

/** Compute the original relative RMS and maximum absolute error. */
std::pair<double,double> error(const Checked& got,const Checked& expected,bool check_shape=true) {
    const auto& actual=got.values; const auto& reference=expected.values;
    if(actual.size()!=reference.size() || reference.empty() || (check_shape && got.shape!=expected.shape))
        throw std::runtime_error("layer-major logit reference missing");
    double sum=0,scale=0,maximum=0;
    for(std::size_t i=0;i<actual.size();++i) {
        const double delta=double(actual[i])-reference[i];
        sum+=delta*delta; scale+=double(reference[i])*reference[i]; maximum=std::max(maximum,std::abs(delta));
    }
    return {std::sqrt(sum/std::max(1e-30,scale)),maximum};
}

/** Reject retired transfer experiments that no longer have a production pipeline. */
void reject_retired() {
    for(const char* name : {"TYLOINFER_MOE_PREFILL_LAYER_FLUSH", "TYLOINFER_MOE_PREFILL_LAYER_ASYNC",
            "TYLOINFER_MOE_PREFILL_LAYER_PHASED"}) {
        const auto* value=tyloi::cuda::env_lookup(name);
        if(value && std::atoi(value)!=0)throw std::runtime_error(std::string("prefill transfer mode is unavailable: ")+name);
    }
}
}

int tyloi::cuda::diagnostics::run_prefill_sweep(Program& program,const std::vector<std::int64_t>& sizes,int repeats,bool profile) {
    if(repeats<1)throw std::runtime_error("--prefill-sweep-reps must be positive");
    if(sizes.empty() || *std::min_element(sizes.begin(),sizes.end())<=0)
        throw std::invalid_argument("--prefill-sweep values must be positive");
    const auto info=program.model().info();
    const auto loader=program.loader();
    const auto* source=dynamic_cast<const tyloi::ModelSource*>(&loader.source);
    if(!source)throw std::runtime_error("prefill sweep requires a model source");
    const bool qwen4=tyloi::models::graph(*source).backbone=="qwen4_exp";
    const auto maximum=*std::max_element(sizes.begin(),sizes.end());
    if(maximum>info.max_context)throw std::runtime_error("--prefill-sweep exceeds the configured context size");
    std::vector<std::int64_t> tokens(static_cast<std::size_t>(maximum));
    const auto span=std::max<std::int64_t>(1,std::min<std::int64_t>(1024,info.vocab_size-2));
    for(std::int64_t i=0;i<maximum;++i)tokens[std::size_t(i)]=1+i%span;
    auto scope=program.executor().enter(program.config().layer_placement.primary_device());
    Work work{program};
    program.drain();
    auto& profiler=program.executor().profiler();
    const bool prior_profile=profiler.enabled;
    struct RestoreProfile { CudaProfiler& profiler; bool enabled; ~RestoreProfile(){profiler.enabled=enabled;} } restore{profiler,prior_profile};
    profiler.enabled=profile;
    auto ids=work.keep(tyloi::ops::reshape(work.keep(tyloi::ops::tensor(tokens,program.executor().device())),{1,maximum}));
    const char* chunks_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_SWEEP_CHUNKS");
    const auto* trace_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_TRACE");
    const std::string trace_root=trace_option?trace_option:"";
    if(!trace_root.empty() && (!chunks_option || !chunks_option[0]))
        throw std::invalid_argument("prefill trace requires explicit chunk sizes");
    reject_retired();
    if(chunks_option && chunks_option[0]) {
        if(auto p=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_LAYER_FLUSH_SWEEP");p && std::atoi(p)!=0)
            throw std::runtime_error("prefill layer flush sweep is unavailable");
        if(auto p=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_LAYER_ASYNC_SWEEP");p && std::atoi(p)!=0)
            throw std::runtime_error("prefill layer async sweep is unavailable");
        if(auto p=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_LAYER_PHASE_SWEEP");p && std::atoi(p)!=0)
            throw std::runtime_error("prefill layer phase sweep is unavailable");
        const auto* variants_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_VARIANTS_JSON");
        const auto variants=variants_option?nlohmann::json::parse(variants_option):nlohmann::json::array();
        const bool variant_sweep=!variants.empty();
        if(variant_sweep && (!variants.is_array() || variants.size()>8))
            throw std::invalid_argument("prefill variants must be an array of one to eight environments");
        const auto apply_variant=[&](int index) {
            const auto& variant=variants.at(index);
            for(const auto& [name,value]:variant.at("environment").items()) {
                const auto option=value.get<std::string>();
                if(name.rfind("TYLOINFER_",0)!=0)throw std::invalid_argument("prefill variant must use MFQ options");
                tyloi::cuda::env_set(name.c_str(),option.c_str());
            }
        };
        const auto* layer_major_sweep_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_LAYER_MAJOR_SWEEP");
        const bool layer_major_sweep=layer_major_sweep_option && std::atoi(layer_major_sweep_option)!=0;
        const auto* layer_major_option=tyloi::cuda::env_lookup("TYLOINFER_MOE_PREFILL_LAYER_MAJOR");
        const bool layer_major_requested=layer_major_option && std::atoi(layer_major_option)!=0;
        const auto* kernel_sweep_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_KERNEL_SWEEP");
        const bool kernel_sweep=kernel_sweep_option && std::atoi(kernel_sweep_option)!=0;
        if(!trace_root.empty() && (variant_sweep || layer_major_sweep || layer_major_requested || kernel_sweep))
            throw std::invalid_argument("prefill trace requires an ordinary chunk sweep");
        const bool gr_gdn_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==2;
        const bool exact_sweep=kernel_sweep && std::atoi(kernel_sweep_option)>=3;
        const bool exact_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)>=4;
        const bool default_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==5;
        const bool quant_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==6;
        const bool qsa_score_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==12;
        const bool gr_batch_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==13;
        const bool gdn_batch_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==14;
        const bool gr_vector_pair_sweep=kernel_sweep && (std::atoi(kernel_sweep_option)==11 || qsa_score_pair_sweep || gr_batch_pair_sweep || gdn_batch_pair_sweep);
        const bool gr_tile_pair_sweep=kernel_sweep && (std::atoi(kernel_sweep_option)==7 || gr_vector_pair_sweep);
        const bool gr_mix_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==8;
        const bool nvq_cohort_pair_sweep=kernel_sweep && std::atoi(kernel_sweep_option)==10;
        const bool nvq_k_pair_sweep=kernel_sweep && (std::atoi(kernel_sweep_option)==9 || nvq_cohort_pair_sweep);
        const char* nvq_trial_option=nvq_cohort_pair_sweep?"TYLOINFER_NVQ_PREFILL_COHORTS":"TYLOINFER_NVQ_PREFILL_NARROW_G2";
        const auto* chunk_oracle_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_CHUNK_SWEEP_ORACLE");
        const bool chunk_sweep_oracle=chunk_oracle_option && std::atoi(chunk_oracle_option)!=0;
        std::vector<int64_t> chunks;
        std::stringstream options(chunks_option);std::string term;
        while(std::getline(options,term,',')) {
            std::size_t parsed=0;const auto value=std::stoll(term,&parsed);
            if(parsed!=term.size())throw std::invalid_argument("invalid prefill chunk size");
            chunks.push_back(value);
        }
        if(chunks.empty() || std::any_of(chunks.begin(),chunks.end(),[](auto n){return n<=0;}))
            throw std::invalid_argument("TYLOINFER_PREFILL_SWEEP_CHUNKS must contain positive chunk sizes");
        Checked chunk_reference, first_reference;
        std::int64_t first_top=-1;
        bool numerical_failure=false;
        tyloi::engine::InferenceRequest prefill_request;
        tyloi::engine::InferenceOutput prefill_output(prefill_request,nullptr,"prefill-sweep");
        if(layer_major_sweep || layer_major_requested || kernel_sweep)
            program.configure(tyloi::engine::resolve_config());
        for(const auto m:sizes) for(const auto chunk:chunks) {
            std::vector<double> elapsed_ms;
            Checked reference_logits;
            for(int repeat=0;repeat<repeats;++repeat) {
                if(variant_sweep) {
                    const int variant=repeat%int(variants.size()); apply_variant(variant);
                    reject_retired();
                    std::cout<<"prefill_variant_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" variant="<<variant<<" name="<<variants.at(variant).at("name").get<std::string>()<<std::endl;
                }
                const bool layer_major=(kernel_sweep?(exact_sweep || repeat%4>0):
                    layer_major_sweep?bool(repeat&1):layer_major_requested) && m>1024;
                if(kernel_sweep) {
                    const auto* exact_gr=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_EXACT_GR_KERNEL");
                    const auto* gr=gdn_batch_pair_sweep?(exact_gr?exact_gr:"21"):qsa_score_pair_sweep || gr_batch_pair_sweep?"21":gr_vector_pair_sweep?(repeat%2?"21":"17"):gr_mix_pair_sweep || nvq_k_pair_sweep?"17":gr_tile_pair_sweep?(repeat%2?(exact_gr?exact_gr:"12"):"6"):quant_pair_sweep?"6":exact_sweep?(repeat%2?(exact_gr?exact_gr:"8"):"1"):(repeat%4==3?"3":"1");
                    const auto* batch=quant_pair_sweep || gr_tile_pair_sweep || gr_mix_pair_sweep || nvq_k_pair_sweep?"1":exact_pair_sweep?(repeat%2?"1":"0"):(!gr_gdn_sweep && repeat%4>=2?"1":"0");
                    const auto* columns=exact_sweep || gr_gdn_sweep && repeat%4>=2?"4":"0";
                    const bool use_defaults=default_pair_sweep && repeat%2;
                    if(use_defaults) {
                        tyloi::cuda::env_clear("TYLOINFER_GR_PREFILL_MATMUL");tyloi::cuda::env_clear("TYLOINFER_MOE_PREFILL_EXPERT_BATCH");
                    } else {
                        tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_MATMUL",gr);tyloi::cuda::env_set("TYLOINFER_MOE_PREFILL_EXPERT_BATCH",batch);
                    }
                    tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_COLUMNS",columns);
                    if(quant_pair_sweep) {
                        const auto* wide_rows=repeat%2?"1":"0";
                        tyloi::cuda::env_set("TYLOINFER_NVQ_PREFILL_M256",wide_rows);
                        std::cout<<"prefill_quant_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                            <<" m256="<<wide_rows<<std::endl;
                    }
                    if(nvq_k_pair_sweep) {
                        const auto* narrow_g2=repeat%2?"1":"0";
                        tyloi::cuda::env_set(nvq_trial_option,narrow_g2);
                        if(nvq_cohort_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_PIPELINED",narrow_g2);
                        std::cout<<(nvq_cohort_pair_sweep?"prefill_nvq_cohort_mode_m=":"prefill_nvq_k_mode_m=")
                            <<m<<" chunk="<<chunk<<" repeat="<<repeat
                            <<(nvq_cohort_pair_sweep?" cohorts=":" narrow_g2=")<<narrow_g2<<std::endl;
                        if(nvq_cohort_pair_sweep)std::cout<<"prefill_gdn_pipeline_mode_m="<<m<<" chunk="<<chunk
                            <<" repeat="<<repeat<<" pipelined="<<narrow_g2<<std::endl;
                    }
                    if(gr_mix_pair_sweep) {
                        const auto* mix_option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_MIX_KERNEL");
                        const auto* mix_baseline=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_MIX_BASELINE");
                        const auto* mixed=repeat%2?(mix_option?mix_option:"1"):(mix_baseline?mix_baseline:"0");
                        tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_FUSED_MIX",mixed);
                        std::cout<<"prefill_mix_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                            <<" mode="<<mixed<<std::endl;
                    }
                    std::cout<<"prefill_kernel_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" gr_kernel="<<gr<<" expert_batch="<<batch<<std::endl;
                    std::cout<<"prefill_gdn_mode_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat
                        <<" columns="<<columns<<std::endl;
                    if(gr_vector_pair_sweep) {
                        tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_FUSED_MIX",qsa_score_pair_sweep || gr_batch_pair_sweep || gdn_batch_pair_sweep || repeat%2?"6":"3");
                        if(qsa_score_pair_sweep)tyloi::cuda::env_set("TYLOINFER_QSA_PREFILL_SCORE_GEMM",repeat%2?"1":"0");
                        if(gr_batch_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_LAYER_BATCH",repeat%2?"1":"0");
                        if(gdn_batch_pair_sweep) {
                            tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_LAYER_BATCH","1");
                            tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_LAYER_BATCH",repeat%2?"1":"0");
                        }
                    }
                }
                if(layer_major_sweep || layer_major_requested)
                    std::cout<<"prefill_layer_major_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat<<" layer_major="<<int(layer_major)<<std::endl;
                work.clear(); program.state().reset(1); sync(); profiler.reset();
                const auto started=Clock::now();
                Tensor logits;
                if(layer_major) {
                    profiler.enabled=profile;
                    auto part=work.keep(ids.dense().narrow(1,0,m));
                    tyloi::models::Input input; input.ids=part; input.advance=true;
                    auto result=program.prefill(input,chunk,prefill_output);
                    auto hidden=work.keep(result.hidden);
                    if(result.raw)work.keep(result.raw);
                    if(result.mode!=tyloi::models::Prefill::Mode::layered)
                        throw std::runtime_error("requested layer-major prefill was not executed");
                    logits=project(work,hidden); sync();
                    if(profile) { profiler.report("prefill_layer_major_m="+std::to_string(m)+" chunk="+std::to_string(chunk)+" repeat="+std::to_string(repeat)); profiler.reset(); }
                } else for(std::int64_t offset=0;offset<m;offset+=chunk) {
                    profiler.enabled=profile && offset==0;
                    const auto count=std::min(chunk,m-offset);
                    auto part=work.keep(ids.dense().narrow(1,offset,count));
                    const bool last=offset+count==m;
                    auto hidden=forward(work,part,!trace_root.empty() && last?&work.trace:nullptr);
                    if(last) {
                        logits=project(work,hidden);
                        if(!trace_root.empty()) {
                            work.trace.emplace_back("ids",work.keep(ids.dense().narrow(1,0,m).contiguous()));
                            work.trace.emplace_back("final_norm",work.keep(hidden.dense().narrow(1,count-1,1).contiguous()));
                            work.trace.emplace_back("logits",logits);
                        }
                    }
                    sync();
                    if(profiler.enabled) { profiler.report("prefill_m="+std::to_string(m)+" chunk="+std::to_string(chunk)+" repeat="+std::to_string(repeat)); profiler.reset(); }
                    std::cout<<"prefill_progress_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat<<" completed="<<offset+count
                        <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(Clock::now()-started).count()<<std::endl;
                }
                profiler.enabled=false;
                const auto ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
                const auto actual=checked(logits);
                if(!trace_root.empty())write_trace(work,std::filesystem::path(trace_root)/
                    ("m"+std::to_string(m))/("chunk"+std::to_string(chunk))/("repeat"+std::to_string(repeat)),m,chunk,repeat);
                if(layer_major_sweep || chunk_sweep_oracle) {
                    const bool reference=chunk_sweep_oracle ? chunk==chunks.front() && repeat==0 : exact_sweep?repeat%4==0:!layer_major;
                    if(reference) { reference_logits=actual; if(chunk_sweep_oracle)chunk_reference=actual; }
                    else {
                        if(chunk_sweep_oracle)reference_logits=chunk_reference;
                        const auto [rms,maximum_error]=error(actual,reference_logits);
                        std::cout<<"prefill_layer_major_oracle_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat<<" relative_rms="<<rms<<" maximum_absolute="<<maximum_error<<std::endl;
                        const bool accepted=rms<=1e-4;
                        if(kernel_sweep || chunk_sweep_oracle) { numerical_failure|=!accepted; std::cout<<"prefill_numerical_validation_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat<<" accepted="<<int(accepted)<<std::endl; }
                        else if(!accepted)throw std::runtime_error("layer-major full-prompt logits differ from chunk-major oracle");
                        if(program.state().position()!=m)throw std::runtime_error("layer-major logical cache did not advance");
                    }
                }
                elapsed_ms.push_back(ms);
                const auto best=top(actual);
                if((layer_major_sweep && (!layer_major || exact_sweep && repeat%4==0) || chunk_sweep_oracle && repeat==0) && m==sizes.front() && chunk==chunks.front()) { first_top=best; first_reference=actual; }
                std::cout<<"prefill_chunk_sample_m="<<m<<" chunk="<<chunk<<" repeat="<<repeat<<" elapsed_ms="<<ms<<" tok_per_s="<<1000.0*m/ms<<" top="<<best<<std::endl;
                work.clear();
            }
            std::sort(elapsed_ms.begin(),elapsed_ms.end());
            std::cout<<(layer_major_sweep?"prefill_mode_comparison_m=":"prefill_chunk_summary_m=")<<m<<" chunk="<<chunk<<" repeats="<<repeats
                <<" median_ms="<<elapsed_ms[elapsed_ms.size()/2]<<" min_ms="<<elapsed_ms.front()<<" max_ms="<<elapsed_ms.back()
                <<" tok_per_s="<<1000.0*m/elapsed_ms[elapsed_ms.size()/2]<<std::endl;
        }
        if((layer_major_sweep || chunk_sweep_oracle) && qwen4) {
            if(numerical_failure && !chunk_sweep_oracle) {
                tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_MATMUL","1");
                tyloi::cuda::env_set("TYLOINFER_MOE_PREFILL_EXPERT_BATCH","0");
            }
            const auto engine_chunks=chunk_sweep_oracle?chunks:std::vector<std::int64_t>{chunks.front()};
                const std::string engine_gr=tyloi::cuda::env_lookup("TYLOINFER_GR_PREFILL_MATMUL")?tyloi::cuda::env_lookup("TYLOINFER_GR_PREFILL_MATMUL"):"17";
                const std::string engine_batch=tyloi::cuda::env_lookup("TYLOINFER_MOE_PREFILL_EXPERT_BATCH")?tyloi::cuda::env_lookup("TYLOINFER_MOE_PREFILL_EXPERT_BATCH"):"1";
                const std::string engine_mix=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_MIX_KERNEL")?tyloi::cuda::env_lookup("TYLOINFER_PREFILL_MIX_KERNEL"):"1";
                const std::string engine_mix_baseline=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_MIX_BASELINE")?tyloi::cuda::env_lookup("TYLOINFER_PREFILL_MIX_BASELINE"):"0";
            for(const auto engine_chunk:engine_chunks) {
                const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_PUBLIC_PAIR_REPEATS");
                const int public_repeats=(exact_sweep || chunk_sweep_oracle) && option?std::clamp(std::atoi(option),1,8):1;
                std::vector<std::int64_t> reference_tokens;
                for(int public_repeat=0;public_repeat<public_repeats;++public_repeat) {
                    for(int order=0;order<(variant_sweep?int(variants.size()):exact_sweep?2:1);++order) {
                        const int engine_mode=variant_sweep?order:exact_sweep?(order^(public_repeat&1)):0;
                        if(variant_sweep)apply_variant(engine_mode);
                        if(variant_sweep)reject_retired();
                        std::cout<<"prefill_public_pair_repeat prompt="<<sizes.front()<<" chunk="<<engine_chunk
                            <<" repeat="<<public_repeat<<" mode="<<engine_mode<<std::endl;
                        work.clear(); program.state().reset(1); sync();
                        cudaMemPool_t pool=nullptr;
                        TYLOINFER_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,program.executor().device().index));
                        TYLOINFER_CUDA_CHECK(cudaMemPoolTrimTo(pool,0));
                if(exact_sweep) {
                    tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_MATMUL",gdn_batch_pair_sweep?engine_gr.c_str():qsa_score_pair_sweep || gr_batch_pair_sweep?"21":gr_vector_pair_sweep?(engine_mode?"21":"17"):gr_mix_pair_sweep || nvq_k_pair_sweep?"17":gr_tile_pair_sweep?(engine_mode?engine_gr.c_str():"6"):quant_pair_sweep?"6":engine_mode?engine_gr.c_str():"1");
                    if(gr_vector_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_FUSED_MIX",qsa_score_pair_sweep || gr_batch_pair_sweep || gdn_batch_pair_sweep || engine_mode?"6":"3");
                    if(qsa_score_pair_sweep)tyloi::cuda::env_set("TYLOINFER_QSA_PREFILL_SCORE_GEMM",engine_mode?"1":"0");
                    if(gr_batch_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_LAYER_BATCH",engine_mode?"1":"0");
                    if(gdn_batch_pair_sweep) {
                        tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_LAYER_BATCH","1");
                        tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_LAYER_BATCH",engine_mode?"1":"0");
                    }
                    tyloi::cuda::env_set("TYLOINFER_MOE_PREFILL_EXPERT_BATCH",quant_pair_sweep || gr_tile_pair_sweep || gr_mix_pair_sweep || nvq_k_pair_sweep?"1":engine_mode?engine_batch.c_str():"0");
                    if(nvq_k_pair_sweep)tyloi::cuda::env_set(nvq_trial_option,engine_mode?"1":"0");
                    if(nvq_cohort_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_PIPELINED",engine_mode?"1":"0");
                    if(quant_pair_sweep)tyloi::cuda::env_set("TYLOINFER_NVQ_PREFILL_M256",engine_mode?"1":"0");
                    if(gr_mix_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_FUSED_MIX",engine_mode?engine_mix.c_str():engine_mix_baseline.c_str());
                }
                        auto config=tyloi::engine::resolve_config();
                        config.generation.prefill_chunk_size=engine_chunk;
                        program.configure(std::move(config));
                        tyloi::engine::InferenceRequest request;
                        request.prompt.assign(tokens.begin(),tokens.begin()+sizes.front());
                        request.sampling.max_tokens=exact_sweep || variant_sweep?3:1;
                        request.sampling.temperature=0.; request.sampling.top_k=1;
                        request.sampling.top_p=1.; request.sampling.enable_mtp=false;
                        tyloi::engine::InferenceOutput output(request,nullptr,"layer-major-check");
                        const auto started=Clock::now();
                        std::vector<std::int64_t> generated;
                        double prefill_ms=0;
                        auto job=program.generate(request,output,false,"layer-major-check");
                        while(auto step=job.next()) if(step.value) {
                            const auto& event=*step.value;
                            if(const auto* delta=std::get_if<tyloi::engine::OutputDelta>(&event))
                                generated.insert(generated.end(),delta->token_ids.begin(),delta->token_ids.end());
                            if(const auto* progress=std::get_if<tyloi::engine::PrefillProgress>(&event)) prefill_ms=progress->timing.model_ms;
                        }
                        if(output.cleanup_failure)std::rethrow_exception(output.cleanup_failure);
                        sync();
                        const auto ms=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
                        std::cout<<"prefill_ordinary_generation_timing prompt="<<sizes.front()<<" chunk="<<engine_chunk<<" mode="<<engine_mode
                            <<" generated_tokens="<<generated.size()<<" elapsed_ms="<<ms<<" tok_per_s="<<1000.0*sizes.front()/ms<<std::endl;
                        std::cout<<"prefill_ordinary_prefill_timing prompt="<<sizes.front()<<" chunk="<<engine_chunk<<" mode="<<engine_mode
                            <<" elapsed_ms="<<prefill_ms<<" tok_per_s="<<1000.0*sizes.front()/prefill_ms<<std::endl;
                        if(generated.empty() || generated.front()!=first_top)
                            throw std::runtime_error("layer-major ordinary generation disagrees with chunk-major oracle");
                        if(exact_sweep || variant_sweep) {
                            if(engine_mode==0)reference_tokens=generated;
                            else if(generated!=reference_tokens)throw std::runtime_error("optimized prefill/decode continuation differs from original generation");
                            std::cout<<"prefill_decode_continuation_check prompt="<<sizes.front()<<" mode="<<engine_mode<<" generated_tokens="<<generated.size()<<" PASS"<<std::endl;
                        } else if(generated.size()!=1)throw std::runtime_error("generation first-token check returned extra tokens");
                        std::cout<<"prefill_layer_major_generation_check prompt="<<sizes.front()<<" chunk="<<engine_chunk<<" top="<<first_top<<" PASS"<<std::endl;
                    }
                }
            }
            if(exact_sweep) {
                tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_MATMUL",engine_gr.c_str());
                tyloi::cuda::env_set("TYLOINFER_MOE_PREFILL_EXPERT_BATCH",engine_batch.c_str());
                if(gr_mix_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_FUSED_MIX",engine_mix.c_str());
                if(quant_pair_sweep)tyloi::cuda::env_set("TYLOINFER_NVQ_PREFILL_M256","1");
                if(nvq_k_pair_sweep)tyloi::cuda::env_set(nvq_trial_option,"1");
                if(nvq_cohort_pair_sweep)tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_PIPELINED","1");
            }
            if(const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_POST_TIMING_PROFILE");option && std::atoi(option)!=0) {
                work.clear(); program.state().reset(1); profiler.reset(); profiler.enabled=true;
                tyloi::models::Input input; input.ids=work.keep(ids.dense().narrow(1,0,sizes.front())); input.advance=true;
                auto result=program.prefill(input,chunks.front(),prefill_output);
                auto hidden=work.keep(result.hidden);
                if(result.raw)work.keep(result.raw);
                if(result.mode!=tyloi::models::Prefill::Mode::layered)throw std::runtime_error("requested layer-major prefill was not executed");
                (void)project(work,hidden); sync();
                profiler.report("prefill_optimized_tail_profile_m="+std::to_string(sizes.front()));
                profiler.enabled=false; profiler.reset();
            }
            if(const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PREFILL_EXPERT_BATCH_AUDIT_RUN");option && std::atoi(option)!=0) {
                tyloi::cuda::env_set("TYLOINFER_GR_PREFILL_MATMUL","1");
                tyloi::cuda::env_set("TYLOINFER_GDN_PREFILL_COLUMNS","0");
                tyloi::cuda::env_set("TYLOINFER_MOE_PREFILL_EXPERT_BATCH","1");
                tyloi::cuda::env_set("TYLOINFER_PREFILL_EXPERT_BATCH_AUDIT","1");
                work.clear(); program.state().reset(1);
                tyloi::models::Input input; input.ids=work.keep(ids.dense().narrow(1,0,sizes.front())); input.advance=true;
                auto result=program.prefill(input,chunks.front(),prefill_output);
                auto hidden=work.keep(result.hidden);
                if(result.raw)work.keep(result.raw);
                if(result.mode!=tyloi::models::Prefill::Mode::layered)throw std::runtime_error("requested layer-major prefill was not executed");
                auto actual=checked(project(work,hidden));
                const auto [rms,maximum_error]=error(actual,first_reference,false); (void)maximum_error;
                std::cout<<"prefill_expert_batch_audit_reference_logits relative_rms="<<rms<<std::endl;
                if(!std::isfinite(rms) || rms>1e-4)throw std::runtime_error("FFN batching audit changed non-FFN math");
            }
        }
        return numerical_failure?2:0;
    }
    for(auto m:sizes) {
        auto part=work.keep(ids.dense().narrow(1,0,m));
        for(int i=0;i<2;++i) {
            program.state().reset(1); (void)project(work,forward(work,part)); sync();
            program.drain(); work.values.resize(2);
        }
        std::vector<double> times;
        std::int64_t best=-1;
        for(int i=0;i<repeats;++i) {
            program.state().reset(1);
            const auto started=Clock::now();
            auto logits=project(work,forward(work,part)); sync();
            const auto elapsed=std::chrono::duration<double,std::milli>(Clock::now()-started).count();
            times.push_back(elapsed);
            if(i==0)best=logits.dense().argmax(-1).item<std::int64_t>();
            program.drain(); work.values.resize(2);
        }
        std::sort(times.begin(),times.end());
        const auto median=times[times.size()/2];
        std::cout<<"prefill_sweep_m="<<m<<" median_ms="<<median<<" min_ms="<<times.front()
            <<" max_ms="<<times.back()<<" tok_per_s="<<1000.0*m/median<<" top="<<best<<'\n';
    }
    return 0;
}
