#pragma once

// Shared layer weights and the per-request layer state.

#include "engine/models/qwen4_exp/model.h"
#include "engine/models/qwen4_exp/moe.h"
#include "engine/models/qwen4_exp/config.h"
#include "engine/ops.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/ops/ffn/gated.h"
#include "diagnostics/router_lookahead.h"

#include <functional>
#include <memory>
#include <utility>
#include <vector>

namespace tyloi::cuda::qwen4_exp {
struct Qwen4Block;

struct LayerState {
    Gdn::State gdn;
    std::unique_ptr<Qsa::State> qsa;
    Ple::State ple;
    bool retain_graph_storage = false, retained_prefix = false;
    Linear next_router;
    MoeFfnPrefetch next_prefetch;
    std::shared_ptr<mfq::cuda::RouterLookaheadInputs> router_audit;
    explicit LayerState(const Qwen4Block& layer);
};

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
    Moe ffn;
    Linear router;
    MoeFfnPrefetch prefetch;
    bool tq_graph_ffn=false;
    std::unique_ptr<Gdn> gdn;
    std::unique_ptr<Qsa> qsa;
    std::unique_ptr<Ple> ple;
    DenseTensor batch_ffn(CudaExecutionContext& execution,const DenseTensor& input) {
        if(input.size(0)<=8)return ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
            execution,
#endif
            input);
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
#ifdef TYLOINFER_NATIVE_CUDA
            auto tile_mode=Executor::current_ffn();
            tile_mode.tile=mode.tile;
            tile_mode.tiles=mode.tiles;
            auto tile_scope=Executor::current().enter(tile_mode);
#endif
            outputs.push_back(ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                execution,
#endif
                input.narrow(0,start,std::min(rows,input.size(0)-start))));
        }
        return ::tyloi::cuda::cat(outputs,0);
    }
    Qwen4Block(CudaExecutionContext &execution, const tyloi::weights::Loader& weights,
               const tyloi::models::qwen4_exp::Config &c, int layer, bool predictor = false);
    void reset(LayerState& state, int64_t) {
        if (gdn)
            gdn->reset(state.gdn, state.retain_graph_storage);
        if (qsa)
            qsa->reset(*state.qsa);
        if (ple)
            ple->reset(state.ple, state.retain_graph_storage);
    }
    bool supports_speculation() const noexcept { return true; }
    void prefetch_token_ids(LayerState& state, const DenseTensor& ids) { if(ple)ple->prefetch_tokens(state.ple, ids); }
    void commit_speculative(LayerState& state) {
        if (gdn)
            gdn->commit(state.gdn);
        if (ple)
            ple->commit(state.ple);
    }
    void rollback_speculative(LayerState& state, int64_t keep) {
        if (gdn)
            gdn->rollback(state.gdn, keep);
        if (qsa)
            qsa->truncate(*state.qsa, keep);
        if (ple)
            ple->rollback(state.ple, keep);
    }
    DenseTensor execute(LayerState& state,
#ifndef TYLOINFER_NATIVE_CUDA
                   CudaExecutionContext &execution,
#endif
                   DenseTensor x, const DenseTensor &ids,
                   const DenseTensor &positions, const DenseTensor &full_positions, int64_t confirmed = 0,const DenseTensor& host_ids = {}) {
        state.retained_prefix=confirmed>0 && ids.size(1)<=8;
        const auto measure=[&](const char* name,auto&& operation) {
#ifdef TYLOINFER_NATIVE_CUDA
            return ::tyloi::ops::measure(name,std::forward<decltype(operation)>(operation));
#else
            return execution.executor().profiler().measure(name,std::forward<decltype(operation)>(operation));
#endif
        };
        if(ple)x=x+measure("prefill.ple",[&] {
            return ple->forward(state.ple, x,ids,true,confirmed,host_ids,full_positions.size(-1)-ids.size(1));
        });
        auto [attention_input,attention_residual,attention_gate]=measure("prefill.gr_attention",[&] {
            return attention_gr.pre(x);
        });
        auto attention_output=gdn
            ? measure("prefill.gdn",[&] {
                return gdn->forward(state.gdn, attention_input,true,confirmed,full_positions.size(-1)-ids.size(1));
            })
            : measure("prefill.qsa",[&] {
                return qsa->forward(*state.qsa, attention_input,positions,full_positions,true);
            });
        auto [ffn_input,ffn_residual,ffn_gate]=measure("prefill.gr_ffn_pre",[&] {
            return ffn_gr.pre_after(attention_output,attention_residual,attention_gate);
        });
        attention_output={};
        auto ffn_output=measure("prefill.ffn",[&] {return ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
            execution,
#endif
            ffn_input);});
        return measure("prefill.gr_ffn_post",[&] {return ffn_gr.post(ffn_output,ffn_residual,ffn_gate);});
    }
    DenseTensor prefill_ple(LayerState& state,
#ifndef TYLOINFER_NATIVE_CUDA
                           CudaExecutionContext& execution,
#endif
                           DenseTensor x,const DenseTensor& ids,const DenseTensor& host_ids) {
        if(ple)x=x+
#ifdef TYLOINFER_NATIVE_CUDA
            ::tyloi::ops::measure
#else
            execution.executor().profiler().measure
#endif
            ("prefill.ple",[&] {
            return ple->forward(state.ple, x,ids,true,0,host_ids);
        });
        return x;
    }
    DenseTensor prefill_attention_core(LayerState& state,
#ifndef TYLOINFER_NATIVE_CUDA
                                      CudaExecutionContext& execution,
#endif
                                      const DenseTensor& input,const DenseTensor& positions,const DenseTensor& full_positions) {
        const auto measure=[&](const char* name,auto&& operation) {
#ifdef TYLOINFER_NATIVE_CUDA
            return ::tyloi::ops::measure(name,std::forward<decltype(operation)>(operation));
#else
            return execution.executor().profiler().measure(name,std::forward<decltype(operation)>(operation));
#endif
        };
        return gdn?measure("prefill.gdn",[&] {return gdn->forward(state.gdn, input,true,0);}):
            measure("prefill.qsa",[&] {return qsa->forward(*state.qsa, input,positions,full_positions,true);});
    }
    GatedResidual::Result prefill_attention(LayerState& state,
#ifndef TYLOINFER_NATIVE_CUDA
                                            CudaExecutionContext& execution,
#endif
                                            DenseTensor x,const DenseTensor& ids,const DenseTensor& host_ids,
                                            const DenseTensor& positions,const DenseTensor& full_positions) {
        const auto measure=[&](const char* name,auto&& operation) {
#ifdef TYLOINFER_NATIVE_CUDA
            return ::tyloi::ops::measure(name,std::forward<decltype(operation)>(operation));
#else
            return execution.executor().profiler().measure(name,std::forward<decltype(operation)>(operation));
#endif
        };
        x=prefill_ple(state,
#ifndef TYLOINFER_NATIVE_CUDA
                      execution,
#endif
                      std::move(x),ids,host_ids);
        auto mix=measure("prefill.gr_attention",[&] {return attention_gr.pre(x);});
        auto attention_output=prefill_attention_core(state,
#ifndef TYLOINFER_NATIVE_CUDA
                                                    execution,
#endif
                                                    mix.mixed,positions,full_positions);
        return measure("prefill.gr_ffn_pre",[&] {
            return ffn_gr.pre_after(attention_output,mix.residual,mix.gate);
        });
    }
    std::vector<::tyloi::Tensor*> graph_warmup_state(LayerState& state) {
        std::vector<::tyloi::Tensor*> result;
        if(gdn)for(auto* value:gdn->graph_state(state.gdn))if(value->defined())result.push_back(value);
        if(ple)for(auto* value:ple->graph_state(state.ple))if(value->defined())result.push_back(value);
        return result;
    }
    GatedResidual::Result graph_attention(LayerState& state,DenseTensor x,const DenseTensor& ids,
                         const DenseTensor& positions,const DenseTensor& cache_positions,const DenseTensor& ple_rows,int layer,
                         int64_t confirmed=0,RecurrentGraphCheckpoint* checkpoint=nullptr,
                         const std::optional<GatedResidual::Result>& pending_residual={}) {
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
            ? ple->forward_graph(state.ple, x,ids,ple_rows,confirmed,checkpoint->ple_conv)
            : ple->forward_chunk(state.ple, x,ids,true,{},ple_rows));
        auto [attention_input,attention_residual,attention_gate]=pending_residual
            ? attention_gr.pre_after(x,pending_residual->residual,pending_residual->gate,marker("attention_mhc_"))
            : attention_gr.pre(x,marker("attention_mhc_"));
        mark("attention_core_begin");
        auto attention_output=gdn
            ? (confirmed ? gdn->forward_graph(state.gdn, attention_input,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                         : gdn->forward_chunk(state.gdn, attention_input,true))
            : qsa->forward_graph(*state.qsa, attention_input,positions,cache_positions);
        mark("attention_core_end");
        auto ffn_result=ffn_gr.pre_after(attention_output,attention_residual,attention_gate,marker("ffn_mhc_"));
        mark("ffn_pre_end");
        attention_output={};
        return ffn_result;
    }
};

inline LayerState::LayerState(const Qwen4Block& layer) {
    if (layer.qsa) qsa = std::make_unique<Qsa::State>(*layer.qsa);
}

} // namespace tyloi::cuda::qwen4_exp
