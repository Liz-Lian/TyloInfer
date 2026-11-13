#include "diagnostics/policy.h"
#include "diagnostics/tokens.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/weights/cache.h"
#include "engine/weights/loader.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
/** Preserve unset and empty environment values independently. */
struct Environment {
    std::string name, previous;
    bool present;
    explicit Environment(const char* value) : name(value), present(env_lookup(value)!=nullptr) {
        if(present) previous=env_lookup(value);
    }
    void set(const char* value) { env_set(name.c_str(),value); }
    ~Environment() noexcept { try { if(present)env_set(name.c_str(),previous.c_str()); else env_clear(name.c_str()); } catch(...) {} }
};

/** Drain and retire graph resources before restoring environment and policy. */
struct Retire {
    Program& program;
    bool done = false;
    void finish() {
        if(done)return;
        program.drain(); program.state().release_graphs(); program.drain();
        done=true;
    }
    ~Retire() noexcept {
        try { finish(); }
        catch(const std::exception& error) { std::cerr << "policy_cleanup error=" << error.what() << '\n'; }
        catch(...) { std::cerr << "policy_cleanup error=unknown\n"; }
    }
};

} // namespace

const char* policy_name(Policy policy) {
    switch(policy) {
    case Policy::route_poll:return "route_poll_comparison";
    case Policy::window_flush:return "window_flush_comparison";
    case Policy::window_input_views:return "window_input_views_comparison";
    case Policy::route_spin:return "route_spin_comparison";
    case Policy::host_pin:return "host_pin_comparison";
    case Policy::shared_gate_fusion:return "shared_gate_fusion_comparison";
    case Policy::router_topk:return "router_topk_comparison";
    case Policy::warp_multi_sum:return "warp_multi_sum_comparison";
    case Policy::mhc_shared_norm:return "mhc_shared_norm_comparison";
    case Policy::mhc_dense_fusion:return "mhc_dense_fusion_comparison";
    case Policy::mhc_packed_groups:return "mhc_packed_groups_comparison";
    case Policy::mhc_fixed_formats:return "mhc_fixed_formats_comparison";
    case Policy::mhc_async_normalized:return "mhc_async_normalized_comparison";
    case Policy::gdn_fixed_geometry:return "gdn_fixed_geometry_comparison";
    case Policy::gdn_inplace_state:return "gdn_inplace_state_comparison";
    case Policy::qsa_fusion:return "qsa_fusion_comparison";
    case Policy::rotary_fusion:return "rotary_fusion_comparison";
    case Policy::attention_grouping:return "attention_grouping_comparison";
    case Policy::mhc_fusion:return "mhc_fusion_comparison";
    }
    throw std::invalid_argument("unknown policy comparison");
}

namespace {

bool allows_free(Policy policy) {
    return policy==Policy::route_spin || policy==Policy::shared_gate_fusion ||
        policy==Policy::mhc_fixed_formats || policy==Policy::mhc_async_normalized ||
        policy==Policy::gdn_fixed_geometry || policy==Policy::gdn_inplace_state || policy==Policy::qsa_fusion;
}

bool allows_resident_legacy(Policy policy) {
    return policy==Policy::mhc_fusion || policy==Policy::mhc_shared_norm ||
        policy==Policy::mhc_dense_fusion || policy==Policy::mhc_packed_groups ||
        policy==Policy::mhc_fixed_formats || policy==Policy::mhc_async_normalized ||
        policy==Policy::gdn_fixed_geometry || policy==Policy::gdn_inplace_state;
}

/** Own temporary environment changes until the execution resources retire. */
struct Settings {
    std::vector<std::unique_ptr<Environment>> values;
    void set(const char* key, const char* value) {
        for(auto& entry:values) if(entry->name==key) { entry->set(value); return; }
        auto entry=std::make_unique<Environment>(key); entry->set(value); values.push_back(std::move(entry));
    }
};

void select(Settings& s,Policy p,bool on) {
    const auto value=on?"1":"0";
    switch(p) {
    case Policy::route_poll:s.set("TYLOINFER_MOE_ROUTE_POLL_US",on?"0":"2000");break;
    case Policy::window_flush:s.set("TYLOINFER_MOE_SINGLE_WINDOW_FLUSH",value);break;
    case Policy::window_input_views:s.set("TYLOINFER_MFE_WINDOW_INPUT_VIEWS",value);break;
    case Policy::route_spin:s.set("TYLOINFER_MOE_ROUTE_SPIN",value);break;
    case Policy::host_pin:s.set("TYLOINFER_MOE_DECODE_HOST_PIN",value);break;
    case Policy::shared_gate_fusion:s.set("TYLOINFER_SHARED_GATE_FUSED",value);break;
    case Policy::router_topk:s.set("TYLOINFER_MOE_TOPK_CACHE",value);break;
    case Policy::warp_multi_sum:s.set("TYLOINFER_CUDA_WARP_MULTI_SUM",value);break;
    case Policy::mhc_shared_norm:s.set("TYLOINFER_GR_SHARE_NORM",value);break;
    case Policy::mhc_dense_fusion:s.set("TYLOINFER_GR_SHARE_NORM","1");s.set("TYLOINFER_GR_DENSE_FUSION",value);break;
    case Policy::mhc_packed_groups:s.set("TYLOINFER_GR_FIXED_GROUPS",value);s.set("TYLOINFER_GR_UP_THREADS",on?"512":"256");break;
    case Policy::mhc_fixed_formats:s.set("TYLOINFER_GR_FIXED_FORMATS",value);break;
    case Policy::mhc_async_normalized:s.set("TYLOINFER_GR_ASYNC_NORMALIZED",value);break;
    case Policy::gdn_fixed_geometry:s.set("TYLOINFER_GDN_FIXED_GEOMETRY",value);break;
    case Policy::gdn_inplace_state:s.set("TYLOINFER_GDN_INPLACE_STATE",value);break;
    case Policy::qsa_fusion:
        s.set("TYLOINFER_QSA_SELECT_FUSED",value);s.set("TYLOINFER_QSA_SPARSE_GATE_FUSED",value);
        s.set("TYLOINFER_QSA_PARALLEL_REDUCE",value);s.set("TYLOINFER_QSA_SPARSE_QUERY_FUSED",value);break;
    case Policy::rotary_fusion:case Policy::attention_grouping:case Policy::mhc_fusion:break;
    }
}
} // namespace

int run_policy(Program& program, Tensor ids, const PolicyOptions& options) {
    const auto* name=policy_name(options.policy);
    if(options.rounds<1 || options.rounds>8)throw std::invalid_argument("comparison rounds must be between one and eight");
    if(options.warmup < -1 || options.warmup>100)throw std::invalid_argument("warmup count must be between zero and 100");
    if(options.gen<=1 || (!options.forced.empty() && options.forced.size()!=std::size_t(options.gen-1)))
        throw std::invalid_argument("runtime comparison requires valid fixed decode inputs and gen>1");
    const auto& cache=program.config().cache;
    const bool routed=program.weights().decode()==weights::Cache::Decode::routed;
    if(!routed && options.policy==Policy::qsa_fusion && program.weights().legacy_experts_resident())
        throw std::invalid_argument("QSA fusion comparison requires the routed graph path");
    const bool legacy=allows_resident_legacy(options.policy) && program.weights().legacy_experts_resident();
    if(cache.mapped || !program.config().weights.preload || (!routed && !legacy))
        throw std::invalid_argument("runtime comparison requires preloaded eligible expert weights and unmapped cache");
    if(options.forced.empty() && !(allows_free(options.policy) && cache.ram_fraction && *cache.ram_fraction==1.0))
        throw std::invalid_argument("runtime comparison requires fixed inputs or an eligible GPU-only free-decode comparison");
    auto loader=program.loader();
    const auto* source=dynamic_cast<const ModelSource*>(&loader.source);
    if(!source || models::graph(*source).backbone!="qwen4_exp")
        throw std::invalid_argument("runtime comparison requires Qwen4 Flash-Next");
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<=0 || ids.scalar_type()!=kInt64)
        throw std::invalid_argument("runtime comparison requires int64 token IDs with shape [1,tokens]");
    auto& executor=program.executor();
    auto scope=executor.enter(program.config().layer_placement.primary_device());
    const auto& residual=executor.options().residual;
    if(options.policy==Policy::attention_grouping && !program.config().weights.grouped)
        throw std::invalid_argument("attention-grouping comparison requires grouped NINT enabled");
    if(options.policy==Policy::mhc_fusion && (!residual.fused_projections || !residual.native_input))
        throw std::invalid_argument("residual fusion comparison requires native packed projections");
    if(options.policy==Policy::mhc_shared_norm && !residual.two_stage)
        throw std::invalid_argument("MHC norm comparison requires fused residual projection");
    if(options.policy==Policy::mhc_dense_fusion && (!residual.two_stage || !residual.dense_injection))
        throw std::invalid_argument("MHC dense fusion comparison requires fused residual BF16 injection");
    if((options.policy==Policy::mhc_packed_groups || options.policy==Policy::mhc_fixed_formats ||
        options.policy==Policy::mhc_async_normalized) && !residual.two_stage)
        throw std::invalid_argument("MHC packed group comparison requires fused residual projection");
    if((options.policy==Policy::gdn_fixed_geometry || options.policy==Policy::gdn_inplace_state) &&
        (!executor.options().gdn_fused_core || !executor.options().gdn_fused_preparation ||
         !executor.options().gdn_fused_output || !executor.mode().transposed))
        throw std::invalid_argument("GDN comparison requires the complete fused transposed core");
    std::cout << std::unitbuf << name << "_model_load_sec="
        << std::chrono::duration<double>(options.load_end-options.load_start).count() << '\n';
    Settings settings;
    Retire final_retire{program};
    const int performance_passes=3*options.rounds;
    for(int pass=0;pass<performance_passes+(options.phase_timings?3:0);++pass) {
        const bool on=pass%3==1;
        const bool timed=options.phase_timings && pass>=performance_passes;
        program.drain(); program.state().release_graphs(); program.state().reset(1);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(current_stream()));
        if(options.phase_timings) {
            settings.set("TYLOINFER_TRACE_LAYER_TIMINGS",timed?"2":"0");
            settings.set("TYLOINFER_TRACE_MHC_TIMINGS","0");
        }
        select(settings,options.policy,on);
        auto compute=executor.mode().compute;
        if(options.policy==Policy::rotary_fusion)compute.rotary=on;
        if(options.policy==Policy::attention_grouping) { compute.rotary=true; compute.attention_grouped=on; }
        if(options.policy==Policy::mhc_fusion) {
            compute.rotary=true; compute.attention_grouped=true;
            compute.residual_two_stage=on; compute.residual_dense_injection=on && residual.dense_injection;
        }
        PolicyScope policy(executor,std::move(compute));
        Retire pass_retire{program};
        std::cout << name << "_pass=" << pass << " phase=" << on << " begin=1";
        if(options.phase_timings)std::cout << " mfe_phase_timings=" << timed;
        std::cout << '\n';
        warmup_prefill(program,ids,false,options.warmup<0?1:options.warmup,true);
        const auto started=std::chrono::steady_clock::now();
        const auto status=generate_diagnostic_tokens(program,ids,options.gen,false,started,started,
            options.forced,{},false,TokenOptions{.already_warmed=true});
        if(status)return status;
        if(options.policy==Policy::mhc_dense_fusion || options.policy==Policy::mhc_packed_groups ||
           options.policy==Policy::mhc_fixed_formats || options.policy==Policy::mhc_async_normalized ||
           options.policy==Policy::gdn_fixed_geometry || options.policy==Policy::gdn_inplace_state ||
           options.policy==Policy::qsa_fusion)
            program.weights().report(std::cout);
        pass_retire.finish();
        std::cout << name << "_pass=" << pass << " phase=" << on << " end=1\n";
    }
    final_retire.finish();
    return 0;
}
} // namespace tyloi::cuda::diagnostics
