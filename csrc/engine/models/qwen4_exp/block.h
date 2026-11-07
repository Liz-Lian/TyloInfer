#pragma once

// The Qwen4-Exp layer record: what one transformer block owns. Weights and
// per-layer runtime caches stay here until the state split moves the caches
// into the request State; the definition is public so the shared layer loop
// can iterate typed blocks without a virtual interface.

#include "engine/weights/cache_internal.h"
#include "engine/models/qwen4_exp/model.h"
#include "engine/models/qwen4_exp/config.h"
#include "engine/models/common/causal_model_ops.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/ops/ffn/gated.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "diagnostics/router_lookahead.h"
#include "engine/weights/moe/expert_cache.h"

#include <functional>
#include <memory>
#include <vector>

namespace tyloi::cuda::qwen4_exp {
struct BlockLoader;


struct RecurrentGraphCheckpoint {
    std::vector<DenseTensor> gdn_conv,gdn_state,ple_conv;
};

struct Qwen4Block final {
    using DenseTensor = ::tyloi::DenseTensor;
    // Placement, as the layer loop sets it for CPU-offloaded blocks.
    int cuda_device = 0;
    bool cpu_offloaded = false;
    tyloi::models::qwen4_exp::Config config;
    GatedResidual attention_gr, ffn_gr;
    std::function<DenseTensor(CudaExecutionContext&, const DenseTensor&)> ffn;
    Linear router;
    Linear next_router;
    MoeFfnPrefetch prefetch,next_prefetch;
    bool tq_graph_ffn=false;
    std::shared_ptr<mfq::cuda::RouterLookaheadInputs> router_audit;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    DenseTensor batch_ffn(CudaExecutionContext& execution,const DenseTensor& input) {
        if(input.size(0)<=8)return ffn(execution,input);
        const auto option=::tyloi::cuda::options::continuous_ffn_tile();
        const int64_t rows=option?*option:8;
        TYLOINFER_RUNTIME_CHECK(rows==8 || rows==16,"continuous FFN tile must be 8 or 16 rows");
        auto* window=DecodeWindow::recording();
        const int previous_profile_tile=window ? window->set_profile_tile(-1) : -1;
        auto& mode = execution.executor().mode().ffn;
        struct TileScope {
            tyloi::cuda::Ffn::Mode& mode;
            int tile,tiles;
            DecodeWindow* window;
            int profile_tile;
            ~TileScope() {mode.tile=tile;mode.tiles=tiles;
                if(window)window->set_profile_tile(profile_tile);}
        } scope{mode,mode.tile,mode.tiles,window,previous_profile_tile};
        mode.tiles=int((input.size(0)+rows-1)/rows);
        std::vector<DenseTensor> outputs;
        for(int64_t start=0;start<input.size(0);start+=rows) {
            mode.tile=int(start/rows);
            if(window)window->set_profile_tile(int(start/rows));
            outputs.push_back(ffn(execution,input.narrow(0,start,std::min(rows,input.size(0)-start))));
        }
        return ::tyloi::cuda::cat(outputs,0);
    }
    // Defined in ops.cpp, next to the loader it uses.
    Qwen4Block(CudaExecutionContext &execution, const tyloi::weights::Loader& weights,
               const tyloi::models::qwen4_exp::Config &c, int layer, bool predictor = false);
    bool retain_graph_storage=false,retained_prefix=false;
    void reset(int64_t) {
        if (gdn)
            gdn->reset(retain_graph_storage);
        if (qsa)
            qsa->reset();
        if (ple)
            ple->reset(retain_graph_storage);
    }
    bool supports_speculation() const noexcept { return true; }
    void prefetch_token_ids(const DenseTensor& ids) { if(ple)ple->prefetch_tokens(ids); }
    void commit_speculative() {
        if (gdn)
            gdn->commit();
        if (ple)
            ple->commit();
    }
    void rollback_speculative(int64_t keep) {
        if (gdn)
            gdn->rollback(keep);
        if (qsa)
            qsa->truncate(keep);
        if (ple)
            ple->rollback(keep);
    }
    DenseTensor execute(CudaExecutionContext &execution, DenseTensor x, const DenseTensor &ids,
                   const DenseTensor &positions, const DenseTensor &full_positions, int64_t confirmed = 0,const DenseTensor& host_ids = {}) {
        retained_prefix=confirmed>0 && ids.size(1)<=8;
        auto& profiler=execution.executor().profiler();
        if(ple)x=x+profiler.measure("prefill.ple",[&] {
            return ple->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),x,ids,true,confirmed,host_ids,full_positions.size(-1)-ids.size(1));
        });
        auto [attention_input,attention_residual,attention_gate]=profiler.measure("prefill.gr_attention",[&] {
            return attention_gr.pre(x);
        });
        auto attention_output=gdn
            ? profiler.measure("prefill.gdn",[&] {
                return gdn->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),execution.executor().profiler(),attention_input,true,confirmed,full_positions.size(-1)-ids.size(1));
            })
            : profiler.measure("prefill.qsa",[&] {
                return qsa->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),attention_input,positions,full_positions,true);
            });
        auto [ffn_input,ffn_residual,ffn_gate]=profiler.measure("prefill.gr_ffn_pre",[&] {
            return ffn_gr.pre_after(attention_output,attention_residual,attention_gate);
        });
        attention_output={};
        auto ffn_output=profiler.measure("prefill.ffn",[&] {return ffn(execution,ffn_input);});
        return profiler.measure("prefill.gr_ffn_post",[&] {return ffn_gr.post(ffn_output,ffn_residual,ffn_gate);});
    }
    DenseTensor forward(CudaExecutionContext &, DenseTensor, DenseTensor, int64_t, const MfqOptional<DenseTensor> &,
                   const RopeCache &, const MfqOptional<DenseTensor> & = mfq_nullopt,
                   const MfqOptional<DenseTensor> & = mfq_nullopt) {
        throw std::runtime_error(
            "Qwen4 block requires the unified model position/PLE input lifecycle");
    }
    DenseTensor forward_context(CudaExecutionContext &execution, DenseTensor x, const BlockContext &context,
                           const RopeCache &) {
        return execute(execution, std::move(x), context.token_ids, context.positions,
                       context.full_positions, context.confirmed_prefix,context.host_token_ids);
    }
    DenseTensor prefill_ple(CudaExecutionContext& execution,DenseTensor x,const BlockContext& context) {
        if(ple)x=x+execution.executor().profiler().measure("prefill.ple",[&] {
            return ple->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),x,context.token_ids,true,0,context.host_token_ids);
        });
        return x;
    }
    DenseTensor prefill_attention_core(CudaExecutionContext& execution,const DenseTensor& input,const BlockContext& context) {
        return gdn?
            execution.executor().profiler().measure("prefill.gdn",[&] {return gdn->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),execution.executor().profiler(),input,true,0);}):
            execution.executor().profiler().measure("prefill.qsa",[&] {return qsa->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),input,context.positions,context.full_positions,true);});
    }
    GatedResidual::Result prefill_attention(CudaExecutionContext& execution,DenseTensor x,const BlockContext& context) {
        x=prefill_ple(execution,std::move(x),context);
        auto mix=execution.executor().profiler().measure("prefill.gr_attention",[&] {return attention_gr.pre(x);});
        auto attention_output=prefill_attention_core(execution,mix.mixed,context);
        return execution.executor().profiler().measure("prefill.gr_ffn_pre",[&] {
            return ffn_gr.pre_after(attention_output,mix.residual,mix.gate);
        });
    }
    std::vector<DenseTensor*> graph_warmup_state() {
        std::vector<DenseTensor*> result;
        if(gdn)for(auto* value:gdn->graph_state())if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state())if(value->defined())result.push_back(value);
        return result;
    }
    DenseTensor execute_graph(CudaExecutionContext& execution,DenseTensor x,const DenseTensor& ids,
                         const DenseTensor& positions,const DenseTensor& cache_positions,const DenseTensor& ple_rows,int layer,
                         int64_t confirmed=0,RecurrentGraphCheckpoint* checkpoint=nullptr,
                         const std::optional<GatedResidual::Result>& pending_residual={},
                         std::optional<GatedResidual::Result>* deferred_residual=nullptr) {
        if(auto* window=DecodeWindow::recording())window->set_profile_layer_kind(layer,gdn?"gdn":"qsa");
        const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
        const bool detailed=::tyloi::cuda::options::mhc_timings();
        const auto marker=[&](const char* role)->GatedResidualMarker {
            if(!detailed || !DecodeWindow::recording())return {};
            return [&,role](const char* phase) {
                const auto name=std::string(role)+phase;mark(name.c_str());
            };
        };
        if(ple)x=x+(confirmed
            ? ple->forward_graph(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),x,ids,ple_rows,confirmed,checkpoint->ple_conv)
            : ple->forward_chunk(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),x,ids,true,{},ple_rows));
        auto [attention_input,attention_residual,attention_gate]=pending_residual
            ? attention_gr.pre_after(x,pending_residual->residual,pending_residual->gate,marker("attention_mhc_"))
            : attention_gr.pre(x,marker("attention_mhc_"));
        mark("attention_core_begin");
        auto attention_output=gdn
            ? (confirmed ? gdn->forward_graph(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),execution.executor().profiler(),attention_input,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                         : gdn->forward_chunk(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),execution.executor().profiler(),attention_input,true))
            : qsa->forward_graph(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),attention_input,positions,cache_positions);
        mark("attention_core_end");
        auto ffn_result=ffn_gr.pre_after(attention_output,attention_residual,attention_gate,marker("ffn_mhc_"));
        mark("ffn_pre_end");
        attention_output={};
        const auto& [ffn_input,ffn_residual,ffn_gate]=ffn_result;
        DecodeWindow::Task prediction{};
        if(next_prefetch && ffn_input.numel()==config.hidden && DecodeWindow::recording()->prediction_enabled()) {
            const auto routing=config.routing();
            auto logits=execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*next_router,ffn_input.reshape({1,config.hidden}).to(::tyloi::kFloat16)).to(::tyloi::kFloat32).contiguous();
            const int prediction_count=std::min<int64_t>(logits.size(-1),
                moe_expert_cache_ssd_experts(::tyloi::weights::detail::cache(execution.weights).moe)?4:2);
            auto selected=moe_topk_cuda(logits,prediction_count,
                routing.activation==tyloi::ops::RouterActivation::sigmoid,
                routing.activation==tyloi::ops::RouterActivation::sqrt_softplus,
                routing.normalize,routing.delayed_softmax,mfq_nullopt,1e-20,routing.scale);
            prediction=next_prefetch(selected[0].to(::tyloi::kInt32).contiguous());
        }
        if(router_audit)router_audit->input(layer,ffn_input);
        const bool early_prediction=prediction.identity && moe_expert_cache_ssd_experts(::tyloi::weights::detail::cache(execution.weights).moe);
        if(early_prediction)DecodeWindow::recording()->enroll(std::move(prediction));
        auto ffn_output=ffn(execution,ffn_input);
        // Reserve current-layer readers before selecting speculative
        // slots. The route export itself precedes current FFN math.
        if(prediction.identity && !early_prediction)DecodeWindow::recording()->enroll(std::move(prediction));
        if(deferred_residual) {
            *deferred_residual=std::move(ffn_result);
            return ffn_output;
        }
        return ffn_gr.post(ffn_output,ffn_residual,ffn_gate);
    }
};

} // namespace tyloi::cuda::qwen4_exp
