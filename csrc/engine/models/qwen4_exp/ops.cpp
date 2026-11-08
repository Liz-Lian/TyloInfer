#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"
#include "engine/models/session/snapshot.h"
#include "env.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/tq.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_shared_gate.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "components/error.h"
#include "context.h"
#include "engine/backends/cuda/native/memory.h"
#include "engine/backends/cuda/native/stream.h"
#include "ops.h"
#include "model.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include "predictor.h"
#include "engine/models/common/load.h"
#include "engine/weights/group.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/tq/rows.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/generation/prefill_budget.h"
#include "engine/weights/nint_row_pipeline.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include "diagnostics/router_lookahead.h"
using mfq::cuda::RouterLookaheadInputs;
#include <fstream>
#include <iostream>
#include <atomic>
#include "engine/ops.h"
#include "engine/backends/cuda/ops/ffn/gated.h"
#include "engine/weights/nint_cache.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "matmul_rows.h"
#include "engine/backends/cuda/native/graph.h"
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>

namespace tyloi::cuda {
static bool streamed_decode_graphs(const CudaExecutionContext& execution) {
    if(!execution.offload)return true;
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_KV_STREAM_GRAPHS");
    return execution.kv && execution.kv->offload() && (!option || option[0]!='0');
}
static ::tyloi::DenseTensor join_request_outputs(
        const std::vector<::tyloi::DenseTensor>& outputs,int64_t axis) {
    const auto option=::tyloi::cuda::options::single_output_view();
    if(outputs.size()==1 && outputs.front().is_contiguous() && (!option || *option))
        return outputs.front();
    return ::tyloi::cuda::cat(outputs,axis);
}
static ::tyloi::DenseTensor append_decode_positions(
        const ::tyloi::DenseTensor& previous,const ::tyloi::DenseTensor& current,
        ::tyloi::DenseTensor& storage,int64_t maximum,
        const char* option_name="TYLOINFER_QWEN4_POSITION_BUFFER") {
    if(const auto option=::tyloi::cuda::options::position_buffer(option_name))if(!*option)
        return previous.defined()?::tyloi::cuda::cat({previous,current},-1):current.clone();
    TYLOINFER_RUNTIME_CHECK(current.dim()>=1 && (!previous.defined() || (previous.dim()==current.dim() &&
        previous.scalar_type()==current.scalar_type() && previous.device()==current.device())),
        "invalid decode position history");
    const int64_t count=previous.defined()?previous.size(-1):0,required=count+current.size(-1);
    TYLOINFER_RUNTIME_CHECK(required<=maximum,"decode position history exceeds context");
    for(int axis=0;previous.defined() && axis+1<current.dim();++axis)
        TYLOINFER_RUNTIME_CHECK(previous.size(axis)==current.size(axis),"decode position history axes changed");
    // Rollback retains a prefix view; restored histories receive a new arena.
    // The same last-axis append serves 1D target and 3D MTP rotary positions.
    const bool owns_prefix=storage.defined() && previous.defined() &&
        previous.data_ptr()==storage.data_ptr();
    if(!owns_prefix || storage.size(-1)<required) {
        int64_t capacity=owns_prefix?storage.size(-1):std::min<int64_t>(256,maximum);
        while(capacity<required)capacity=capacity>maximum/2?maximum:capacity*2;
        auto shape=current.sizes().vec();shape.back()=capacity;
        auto expanded=::tyloi::cuda::empty(shape,current.options());
        if(count)expanded.narrow(-1,0,count).copy_(previous);
        storage=std::move(expanded);
    }
    storage.narrow(-1,count,current.size(-1)).copy_(current);
    return storage.narrow(-1,0,required);
}
}

namespace tyloi::cuda::qwen4_exp {
using Config = tyloi::models::qwen4_exp::Config;

// Preserve router and shared-expert arithmetic while regrouping routed experts.
static thread_local int64_t qwen_prefill_reference_rows=0;
struct QwenPrefillReferenceRowsScope {
    int64_t previous;
    explicit QwenPrefillReferenceRowsScope(int64_t rows):previous(qwen_prefill_reference_rows) {
        qwen_prefill_reference_rows=rows;
    }
    ~QwenPrefillReferenceRowsScope(){qwen_prefill_reference_rows=previous;}
};
template<class Projection>
static DenseTensor qwen_prefill_preserve_rows(const DenseTensor& source,Projection project) {
    const auto rows=qwen_prefill_reference_rows;
    if(rows<=0 || source.size(0)<=rows)return project(source);
    DenseTensor output;
    for(int64_t begin=0;begin<source.size(0);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(0)-begin);
        auto part=project(source.narrow(0,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[0]=source.size(0);
            output=::tyloi::cuda::empty(shape,part.options());
        }
        output.narrow(0,begin,count).copy_(part);
    }
    return output;
}

// This scope is active only while an entire GDN layer is evaluated. Retain
// each projection's original token count and therefore its GEMM arithmetic.
static thread_local int64_t qwen_gdn_prefill_projection_rows=0;
struct QwenGdnPrefillProjectionRowsScope {
    int64_t previous;
    explicit QwenGdnPrefillProjectionRowsScope(int64_t rows):previous(qwen_gdn_prefill_projection_rows) {
        qwen_gdn_prefill_projection_rows=rows;
    }
    ~QwenGdnPrefillProjectionRowsScope(){qwen_gdn_prefill_projection_rows=previous;}
};
template<class Project>
static DenseTensor qwen_gdn_project_rows(const DenseTensor& source,Project project) {
    const auto rows=qwen_gdn_prefill_projection_rows;
    if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)return project(source);
#ifdef TYLOINFER_NATIVE_CUDA
    const auto parallel_option=::tyloi::cuda::options::gdn_prefill_projection_streams();
    const int streams=parallel_option?*parallel_option:2;
    const auto tail=source.size(1)%rows;
    if(streams>0 && (tail==0 || tail>8)) {
        const tyloi::cuda::MatmulRowsScope gemm_rows(rows,streams);
        return project(source);
    }
#endif
    DenseTensor output;
    for(int64_t begin=0;begin<source.size(1);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(1)-begin);
        auto part=project(source.narrow(1,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[1]=source.size(1);
            output=::tyloi::cuda::empty(shape,part.options());
        }
        output.narrow(1,begin,count).copy_(part);
    }
    return output;
}

DenseTensor Gdn::project_weight(const ::tyloi::Tensor& weight,const DenseTensor& source) const {
    return qwen_gdn_project_rows(source,[&](const DenseTensor& input){return ::tyloi::ops::project(input,weight).dense();});
}

DenseTensor Gdn::project_qkv(const DenseTensor& source) const {
    if(w_->qkv_count==1)return project_weight(w_->input_projection[0],source);
    return qwen_gdn_project_rows(source,[&](const DenseTensor& input) {
        return ::tyloi::cuda::cat({::tyloi::ops::project(input,w_->input_projection[0]).dense(),
            ::tyloi::ops::project(input,w_->input_projection[1]).dense(),
            ::tyloi::ops::project(input,w_->input_projection[2]).dense()},-1);
    });
}

std::vector<DenseTensor> Gdn::project_inputs(const DenseTensor& source) const {
    if(!(grouped_projection_ && ::tyloi::ops::grouped(w_->input_projection)))
        return {project_qkv(source),
            project_weight(w_->input_projection[w_->qkv_count],source),
            project_weight(w_->input_projection[w_->qkv_count+1],source),
            project_weight(w_->input_projection[w_->qkv_count+2],source)};
    auto* window=tyloi::cuda::DecodeWindow::recording();
    if(window)window->mark_current("gdn_input_begin");
    const auto rows=qwen_gdn_prefill_projection_rows;
    const auto project=[&](const DenseTensor& input) {
        const auto option=::tyloi::cuda::options::gdn_packed_qkv();
        const size_t prefix=w_->qkv_count==3 && (!option || *option)?3:1;
        auto handles=::tyloi::ops::project_group(input,w_->input_projection,prefix);
        std::vector<DenseTensor> result;
        for(const auto& handle:handles)result.push_back(handle.dense());
        if(w_->qkv_count==3 && prefix==1) {
            TYLOINFER_RUNTIME_CHECK(result.size()==6,"GDN input projection count mismatch");
            result={::tyloi::cuda::cat({result[0],result[1],result[2]},-1),result[3],result[4],result[5]};
        }
        TYLOINFER_RUNTIME_CHECK(result.size()==4,"GDN input projection count mismatch");
        return result;
    };
    std::vector<DenseTensor> output;
    const auto slice=[&] {
        for(int64_t begin=0;begin<source.size(1);begin+=rows) {
            const auto count=std::min<int64_t>(rows,source.size(1)-begin);
            auto part=project(source.narrow(1,begin,count));
            if(output.empty())for(const auto& value:part) {
                auto shape=value.sizes().vec();shape[1]=source.size(1);
                output.push_back(::tyloi::cuda::empty(shape,value.options()));
            }
            TYLOINFER_RUNTIME_CHECK(part.size()==output.size(),"GDN projection count changed across chunks");
            for(size_t i=0;i<part.size();++i)output[i].narrow(1,begin,count).copy_(part[i]);
        }
    };
    if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows) {
        output=project(source);
    } else {
#ifdef TYLOINFER_NATIVE_CUDA
        const auto parallel_option=::tyloi::cuda::options::gdn_prefill_projection_streams();
        const int streams=parallel_option?*parallel_option:2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const tyloi::cuda::MatmulRowsScope gemm_rows(rows,streams);
            output=project(source);
        } else {
            slice();
        }
#else
        slice();
#endif
    }
    if(window)window->mark_current("gdn_input_end");
    return output;
}

DenseTensor Gdn::project_output(const DenseTensor& source) const {
    auto* window=tyloi::cuda::DecodeWindow::recording();
    if(window)window->mark_current("gdn_output_begin");
    auto output=project_weight(w_->output,source);
    if(window)window->mark_current("gdn_output_end");
    return output;
}



Moe::Moe(Pipelines& pipelines, std::vector<Linear> gate_up, Linear down,
         Linear router, Linear shared_gate, Linear sg, Linear su, Linear sd,
         const Config &c, MoeFfnPrefetch* prefetch)
    : gate_up_(std::move(gate_up)), down_(std::move(down)), router_(std::move(router)), config_(c) {
    auto shared = std::make_shared<MoeSharedExpert>(std::move(sg),std::move(su),
                                                    std::move(sd),std::move(shared_gate));
    shared_=shared;
    MoeFfnSharedWeights shared_weights;
    shared_weights.projections=shared->projections;
    shared_weights.scalar_gate=shared->scalar_gate;
    shared_weights.gate=[shared](const Projector& project,const DenseTensor& input) {
        return shared->gate(project,input);
    };
    auto banks = tq_banks(gate_up_.empty() ? nullptr : gate_up_.front(),
        gate_up_.empty() ? nullptr : gate_up_.back(), down_);
    if(banks.gate || banks.up || banks.down) {
        TYLOINFER_RUNTIME_CHECK(gate_up_.size()==2 && down_ &&
            banks.gate && banks.up && banks.down,
            "TQ MoE requires separate Gate, Up and Down banks");
#ifdef TYLOINFER_NATIVE_CUDA
        tq_=true;
#else
        tq_=std::make_shared<TqMoeBanks>(std::move(banks));
#endif
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::shared_ptr<MoeFfnPipeline> owned_pipeline;
    MoeFfnPrefetch prefetch_fn;
    if(!tq_) {
        owned_pipeline=pipelines.make(
            gate_up_,{down_},[shared](const Projector& project,const DenseTensor& source) {
                return shared->forward(project,source,qwen_prefill_reference_rows);
            },{},std::move(shared_weights));
        if(owned_pipeline) {
            const std::weak_ptr<MoeFfnPipeline> borrowed=owned_pipeline;
            if(prefetch)prefetch_fn=[borrowed](const DenseTensor& ids) {
                if(auto pipeline=borrowed.lock())return prefetch_moe_ffn_pipeline(pipeline,ids);
                return tyloi::cuda::DecodeWindow::Task{};
            };
            pipeline_=borrowed;
        }
    }
#endif
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(owned_pipeline) {
        pipelines.add(std::move(owned_pipeline));
        if(prefetch)*prefetch=std::move(prefetch_fn);
    }
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
DenseTensor Moe::forward(const DenseTensor &x) const {
#else
DenseTensor Moe::forward(CudaExecutionContext &execution,const DenseTensor &x) const {
#endif
    const auto& c=config_;
    const auto& gate_up=gate_up_;
    const auto& down=down_;
    const auto& router=router_;
    auto source = x.reshape({-1, c.hidden}).to(::tyloi::kFloat16);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    auto borrowed=pipeline_.lock();
#endif
#ifndef TYLOINFER_NATIVE_CUDA
    const auto& tq=tq_;
    const auto layer_prefill = ::tyloi::cuda::options::moe_prefill_layer();
    const bool combined_prefill = source.size(0)>8 &&
        execution.executor().mmq().mode==KlMmqMode::Default &&
        execution.weights.prefill_ready(layer_prefill);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const bool use_pipeline=moe_ffn_use_pipeline(borrowed,source.size(0),gate_up.size());
#else
    const bool use_pipeline=false;
#endif
#elif TYLOINFER_ENABLE_MFQ_COMPAT
    const bool use_pipeline=borrowed && (gate_up.size()!=1 ||
        combined_prefill ||
        ((source.size(0)<=8 || (source.size(0)<=32 && execution.executor().mode().ffn.batch_serial)) &&
         execution.executor().options().moe_ffn.two_stage &&
         execution.executor().mmq().mode==KlMmqMode::Default));
#else
    const bool use_pipeline=false;
#endif
    const auto routing = c.routing();
    auto selected = moe_topk_cuda(
#ifdef TYLOINFER_NATIVE_CUDA
        qwen_prefill_preserve_rows(source,[&](const DenseTensor& part){return tyloi::ops::project(part,router).dense();}).contiguous(),
#else
        qwen_prefill_preserve_rows(source,[&](const DenseTensor& part){return execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*router,part);}).contiguous(),
#endif
        c.topk, routing.activation == tyloi::ops::RouterActivation::sigmoid,
        routing.activation == tyloi::ops::RouterActivation::sqrt_softplus,
        routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
#ifdef TYLOINFER_NATIVE_CUDA
    auto shared_output = use_pipeline ? DenseTensor{} : shared_->forward(source,qwen_prefill_reference_rows);
#else
    auto shared_output = use_pipeline ? DenseTensor{} : shared_->forward(execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),source,qwen_prefill_reference_rows);
#endif
    DenseTensor routed;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(use_pipeline) routed = run_moe_ffn_pipeline(borrowed,
#ifdef TYLOINFER_NATIVE_CUDA
                source,selected[0].to(::tyloi::kInt32).contiguous(),selected[1]);
#else
                execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)),source,
                selected[0].to(::tyloi::kInt32).contiguous(),selected[1],execution.executor().mode().ffn.batch_serial,
                execution.executor().mode().ffn.tile,execution.executor().mode().ffn.tiles);
#endif
    else
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    routed = tyloi::ops::experts(source,selected[0],selected[1],gate_up,down,c.moe_width).dense();
#else
    if(tq) routed = execution.executor().profiler().measure("tq.moe",[&] {
                return execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*tq,source,selected[0],selected[1]);
            });
    else {
        DenseTensor hidden;
        {
            const auto& project = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config));
            auto gu = project.run(*gate_up[0], source, selected[0]);
            if(gate_up.size()==2) {
                auto up = project.run(*gate_up[1], source, selected[0]);
                gu = ::tyloi::cuda::cat({gu, up}, -1);
            }
            auto gate = gu.narrow(-1, 0, c.moe_width),
                 up = gu.narrow(-1, c.moe_width, c.moe_width);
            hidden = (gate * ::tyloi::cuda::sigmoid(gate)) * up;
        }
        auto pairs = execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*down, hidden, selected[0]);
        hidden = {};
        auto reduced = ::tyloi::cuda::zeros({source.size(0), c.hidden},
                                            source.options().dtype(::tyloi::kFloat32));
        for (int64_t r = 0; r < c.topk; ++r)
            reduced = reduced + pairs.select(1, r).to(::tyloi::kFloat32) *
                                selected[1].select(1, r).unsqueeze(-1);
        routed = reduced.to(pairs.scalar_type());
    }
#endif
    if(use_pipeline)return routed.reshape(x.sizes());
    return (routed + shared_output).reshape(x.sizes());
}


// One word changes the captured readout's work without retaining a second
// graph or its scratch. DecodeWindow waits for the previous replay before the
// host changes this flag. Greedy requests do not consume compact top-k.
struct MtpCompactControl {
    tyloi::cuda::HostBuffer host{128};
    DenseTensor storage=::tyloi::cuda::zeros({1},::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt32));
    const volatile int32_t* device=nullptr;
    int previous=-1;
    MtpCompactControl() {
        device=storage.data_ptr<int32_t>();
        tyloi::cuda::charge_tensor_host_bytes(host.size());
    }
    ~MtpCompactControl(){tyloi::cuda::tensor_host_bytes.fetch_sub(host.size());}
    void set(bool enabled,const char* role,int rows) {
        if(previous!=int(enabled)) {
            *static_cast<int32_t*>(host.data())=enabled?1:0;
            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(storage.data_ptr(),host.data(),sizeof(int32_t),
                cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            std::cerr<<"runtime_mtp_compact_control role="<<role<<" rows="<<rows<<" enabled="<<int(enabled)<<'\n';
            previous=int(enabled);
        }
    }
};
struct Qwen4MtpGraph {
    struct Cell {
        bool used_in_request=false;
        int profile_replays=0;
        DenseTensor ids,hidden,positions,cache_positions,normalized,multi,logits,greedy_host,compact_host;
        MtpCompactControl compact_control;
        DecodeWindow window;
        Cell(const DenseTensor& source,const DenseTensor& token_ids,cudaStream_t stream)
            :ids(::tyloi::cuda::empty_like(token_ids)),hidden(::tyloi::cuda::empty_like(source)),
             positions(::tyloi::cuda::empty({3,1,source.size(1)},token_ids.options().dtype(::tyloi::kInt32))),
             cache_positions(::tyloi::cuda::empty({source.size(1)},token_ids.options().dtype(::tyloi::kInt64))),
             greedy_host(::tyloi::cuda::empty({1},token_ids.options().device(::tyloi::kCPU).dtype(::tyloi::kInt64).pinned_memory(true))),
             compact_host(::tyloi::cuda::empty({1,64},token_ids.options().device(::tyloi::kCPU).dtype(::tyloi::kInt64).pinned_memory(true))),
             window(stream,::tyloi::cuda::options::layer_timings()) {
            window.set_profile_geometry("mtp_predictor",static_cast<int>(source.size(1)));
        }
    };
    MfqCudaStream stream=mfq_get_current_cuda_stream();
    std::map<std::tuple<int,int,int>,std::unique_ptr<Cell>> cells;
    DenseTensor position_storage;
    int64_t capacity=0;
    bool failed=false;
    DenseTensor last_compact;
    bool reusable(const Predictor& predictor, const PredictorState& request) const {
        return request.reuse_decode_graphs && !failed && streamed_decode_graphs(*predictor.execution) &&
            !predictor.layers.empty() && predictor.layers[0]->qsa && predictor.layers[0]->qsa->graph_capacity(*request.layer_states[0]->qsa)>0;
    }
    void restore_legacy(const Predictor& predictor, PredictorState& request) {
        last_compact={};cells.clear();predictor.layers[0]->qsa->leave_graph(*request.layer_states[0]->qsa);
    }
    tyloi::models::MtpStep run(const Predictor& predictor,PredictorState& request,const DenseTensor& hidden,const DenseTensor& ids,
               const DenseTensor& positions = {},bool history_only=false,bool scalar_history=false) {
        using ProfileTime=std::chrono::steady_clock;
        const auto profile_started=ProfileTime::now();
        const auto profile_transfer_before=predictor.execution->weights.activity().streamed_bytes;
        TYLOINFER_RUNTIME_CHECK(!failed && stream.stream()==mfq_current_cuda_stream(),"MTP predictor graph is unavailable");
        auto& execution=*predictor.execution;auto& block=*predictor.layers[0];
        auto& layer_state=*request.layer_states[0];
        last_compact={};
        // Predictor prefill may retain a multi-GiB expert image. Retire it
        // before graph replay so the first draft does not page graph buffers.
        execution.weights.retire_prefill();
        const auto profile_released=ProfileTime::now();
        const int64_t tokens=ids.size(1),start=request.lengths[0],required=start+tokens;
        TYLOINFER_RUNTIME_CHECK(required<=predictor.config.maximum,"MTP predictor exceeds context capacity");
        if(block.qsa->graph_capacity(*layer_state.qsa)) {
            execution.kv->ensure(required);
            capacity=block.qsa->graph_capacity(*layer_state.qsa);
        }
        if(required>capacity) {
            execution.weights.finish_exchange();restore_legacy(predictor,request);
            capacity=std::min<int64_t>(predictor.config.maximum,std::max<int64_t>(2048,capacity));
            while(capacity<required)capacity=std::min(predictor.config.maximum,capacity*2);
        }
        const bool scalar_chain=history_only && scalar_history && tokens>1;
        const auto key=std::make_tuple(int(tokens),int(hidden.scalar_type()),history_only?(scalar_chain?2:1):0);
        auto& owner=cells[key];if(!owner)owner=std::make_unique<Cell>(hidden,ids,stream.stream());
        auto& cell=*owner;cell.hidden.copy_(hidden);cell.ids.copy_(ids);
        if(!history_only)cell.compact_control.set(request.compact_graph_sampling,"predictor",int(tokens));
        cell.window.set_profile_geometry(scalar_chain?"mtp_history_scalar_chain":history_only?"mtp_history":"mtp_predictor",int(tokens));
        request.graph_initialized=request.graph_initialized || !cell.used_in_request;
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,start);
        if (positions.defined()) cell.positions.copy_(positions);
        try {
            const auto profile_exchange_begin=ProfileTime::now();
            // A warmed history graph touches attention KV and dense weights
            // only. Let the expert exchange proceed until the full draft.
            if(!cell.window.valid())
                execution.weights.finish_exchange();
            else if(!history_only)execution.weights.prepare_predictor();
            const auto profile_exchange_end=ProfileTime::now();
            block.qsa->prepare_graph(*layer_state.qsa,request.positions[0],capacity);
            if(!cell.window.valid()) {
                DecodeGraphBranchScope branch_scope(execution.executor().mode().ffn.serial);
                request.graph_initialized=true;
                const auto capture_started=std::chrono::steady_clock::now();
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    const auto make_input=[&](const DenseTensor& hidden_rows,const DenseTensor& token_rows,int64_t rows) {
                        auto embeddings=predictor.embed(token_rows);
#ifdef TYLOINFER_NATIVE_CUDA
                        auto e=execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*predictor.embedding_fusion,rms_norm(embeddings,dynamic_cast<const DenseTensor&>(*predictor.embedding_norm)+1,predictor.config.eps));
                        auto h=execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*predictor.hidden_fusion,rms_norm(hidden_rows,dynamic_cast<const DenseTensor&>(*predictor.hidden_norm)+1,predictor.config.eps)
#else
                        auto e=execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*predictor.embedding_fusion,rms_norm(embeddings,predictor.embedding_norm+1,predictor.config.eps));
                        auto h=execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)).run(*predictor.hidden_fusion,rms_norm(hidden_rows,predictor.hidden_norm+1,predictor.config.eps)
#endif
                            .reshape({1,rows,predictor.config.streams,predictor.config.hidden}));
                        return (h+e.unsqueeze(-2)).reshape({1,rows,predictor.config.streams*predictor.config.hidden});
                    };
                    if(scalar_chain) {
                        // Keep scalar projection/reduction order while replaying
                        // all accepted history rows with one set of submissions.
                        for(int64_t row=0;row<tokens;++row) {
                            auto input=make_input(cell.hidden.narrow(1,row,1),cell.ids.narrow(1,row,1),1);
                            auto mix=block.attention_gr.pre(input);
                            block.qsa->forward_graph(*layer_state.qsa,mix.mixed,cell.positions.narrow(-1,row,1).contiguous(),
                                cell.cache_positions.narrow(0,row,1),{},true,true);
                        }
                    } else if(history_only) {
                        auto input=make_input(cell.hidden,cell.ids,tokens);
                        auto mix=block.attention_gr.pre(input);
                        block.qsa->forward_graph(*layer_state.qsa,mix.mixed,cell.positions,cell.cache_positions,{},true,true);
                    } else {
                        auto input=make_input(cell.hidden,cell.ids,tokens);
                        cell.multi=block.execute_graph(layer_state,execution,std::move(input),cell.ids,cell.positions,cell.cache_positions,{},0);
                        cell.window.mark(-1,"head_begin");
                        cell.normalized=predictor.final_mixer->pre(cell.multi).mixed;
                        cell.logits=predictor.logits(cell.normalized.narrow(1,tokens-1,1).contiguous());
                        if(cell.logits.size(-1)>=64) {
                            auto compact=sample_top_k_compact_if_cuda(cell.logits.reshape({1,-1}).contiguous(),64,
                                cell.compact_control.device);
                            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.compact_host.data_ptr(),compact.data_ptr(),64*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
                        }
                        auto greedy=sample_greedy_cuda(cell.logits.reshape({1,-1}).contiguous());
                        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),sizeof(int64_t),
                            cudaMemcpyDeviceToHost,stream.stream()));
                    }
                    cell.window.mark(-1,"window_end");
                },[]{},[&]{cell.multi={};cell.normalized={};cell.logits={};});
                ++request.graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=predictor rows="<<tokens
                    <<" history_only="<<history_only<<" scalar_chain="<<scalar_chain<<" segments="<<cell.window.segments()
                    <<" count="<<request.graph_captures<<" ms="
                    <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            const auto profile_run_begin=ProfileTime::now();
            cell.window.run();
            const auto profile_run_end=ProfileTime::now();
            cell.used_in_request=true;++request.graph_replays;block.qsa->advance_graph(*layer_state.qsa,tokens);
            request.positions[0]=append_decode_positions(request.positions[0],cell.positions,
                position_storage,predictor.config.maximum,"TYLOINFER_MTP_POSITION_BUFFER");
            request.lengths[0]=required;request.batch=1;
            const auto* profile_option=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_MTP");
            if(profile_option && profile_option[0]=='1' && ++cell.profile_replays<=(history_only?2:16)) {
                const auto ms=[](auto begin,auto end){return std::chrono::duration<double,std::milli>(end-begin).count();};
                std::fprintf(stderr,"runtime_mtp_predictor_profile rows=%lld history=%d replay=%d release_ms=%.4f setup_ms=%.4f exchange_ms=%.4f prepare_ms=%.4f run_ms=%.4f tail_ms=%.4f transfer_mb=%.4f\n",
                    static_cast<long long>(tokens),int(history_only),cell.profile_replays,
                    ms(profile_started,profile_released),ms(profile_released,profile_exchange_begin),
                    ms(profile_exchange_begin,profile_exchange_end),ms(profile_exchange_end,profile_run_begin),
                    ms(profile_run_begin,profile_run_end),ms(profile_run_end,ProfileTime::now()),
                    (execution.weights.activity().streamed_bytes-profile_transfer_before)/1048576.0);
            }
            if(history_only)return {};
            if(request.compact_graph_sampling && cell.logits.size(-1)>=64)last_compact=cell.compact_host;
            return {std::make_shared<DenseTensor>(cell.normalized.clone()),
                std::make_shared<DenseTensor>(cell.multi.clone()), std::make_shared<DenseTensor>(cell.logits.clone()),
                static_cast<int32_t>(*cell.greedy_host.data_ptr<int64_t>())};
        } catch(...) {failed=true;throw;}
    }
};
DenseTensor Predictor::graph_compact_topk(const tyloi::models::MtpState& value) const {
    const auto& request=state<PredictorState>(value);
    return request.graph_registry?request.graph_registry->last_compact:DenseTensor{};
}
void Predictor::set_compact_graph_sampling(tyloi::models::MtpState& value, bool enabled) const {
    state<PredictorState>(value).compact_graph_sampling=enabled;
}
PredictorState::PredictorState(const Predictor& owner)
    : MtpState(owner), positions(owner.layers.size()), lengths(owner.layers.size(),0) {
    for (const auto& layer : owner.layers)
        layer_states.push_back(std::make_unique<LayerState>(*layer));
}
PredictorState::~PredictorState() = default;
int64_t PredictorState::position() const noexcept {
    return lengths.empty() ? 0 : lengths.front();
}
std::unique_ptr<tyloi::models::MtpState> Predictor::make_state() const {
    return std::make_unique<PredictorState>(*this);
}
Predictor::Predictor() = default;
Predictor::~Predictor() = default;
Predictor::Predictor(Predictor&&) noexcept = default;
Predictor& Predictor::operator=(Predictor&&) noexcept = default;

DenseTensor Predictor::embed(DenseTensor ids) const {
    return tyloi::cuda::embed(*embedding, std::move(ids), execution->executor().collectives(),
                           execution->executor().profiler(), primary);
}

DenseTensor Predictor::logits(DenseTensor hidden) const {
    return tyloi::cuda::logits(*output, std::move(hidden), execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)),
                            execution->executor().profiler(), softcap);
}

void Predictor::reset(tyloi::models::MtpState& value, int64_t next_batch) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    const bool retain=next_batch==1 && request.graph_registry && request.graph_registry->reusable(*this,request);
    if(!retain)request.graph_registry.reset();
    else for(auto& [key,cell]:request.graph_registry->cells) {cell->used_in_request=false;cell->profile_replays=0;}
    request.graph_initialized=false;
    for (std::size_t i=0; i<layers.size(); ++i) {
        request.layer_states[i]->retain_graph_storage=retain;
        layers[i]->reset(*request.layer_states[i], next_batch);
    }
    for (auto &pos : request.positions)
        pos = DenseTensor();
    std::fill(request.lengths.begin(), request.lengths.end(), 0);
    request.batch = next_batch;
}

std::pair<DenseTensor, DenseTensor> Predictor::evaluate(tyloi::models::MtpState& value, const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth, bool cache,
                                       const DenseTensor &supplied_positions,
                                       const DenseTensor &supplied_embeddings, bool history_only) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(!history_only || (cache && layers.size()==1),
        "cache-only MTP priming requires a single predictor layer");
    if(request.graph_registry) {
        if(request.graph_registry->reusable(*this,request) && ids.size(0)==1 && cache)layers[0]->qsa->leave_graph(*request.layer_states[0]->qsa);
        else {request.graph_registry->restore_legacy(*this,request);request.graph_registry.reset();}
    }
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm->numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
#else
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
                      "Qwen4-Exp MTP input geometry mismatch");
#endif
    const auto b = ids.size(0), t = ids.size(1);
    const auto capacity = config.maximum;
    const auto count = int64_t(request.lengths.size());
    tyloi::models::require_model(count > 0 && b > 0 && t > 0, "invalid predictor geometry");
    const auto layer = (depth % count + count) % count;
    if (cache && request.batch && request.batch != b)
        reset(value,b);
    const auto start = cache ? request.lengths[layer] : 0;
    tyloi::models::require_model(start >= 0 && t <= capacity - start,
                                 "predictor history exceeds context capacity");
    auto embeds =
        supplied_embeddings.defined() ? supplied_embeddings : embed(ids);
    TYLOINFER_RUNTIME_CHECK(embeds.sizes().vec() ==
                          std::vector<int64_t>({b, t, config.hidden}),
                      "Qwen4-Exp MTP embedding shape mismatch");
    std::array<DenseTensor, 2> pos;
    {
        // Full positions persist across predictor priming tiles and must
        // not occupy expert storage borrowed by this prefill phase.
        tyloi::cuda::ResidentAllocationScope persistent_positions;
        auto current = supplied_positions.defined()
                           ? supplied_positions
                           : ::tyloi::cuda::arange(start, start + t,
                                        ids.options().dtype(::tyloi::kInt32));
        if (current.dim() == 1)
            current = current.reshape({1, 1, t}).expand({3, 1, t}).contiguous();
        else if (current.dim() == 2)
            current = current.unsqueeze(1);
        if (current.dim() == 3 && current.size(0) == 4)
            current = current.narrow(0, 1, 3);
        TYLOINFER_RUNTIME_CHECK(current.dim() == 3 && current.size(0) == 3 &&
                              current.size(-1) == t &&
                              (current.size(1) == 1 || current.size(1) == b),
                          "Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        current = current.to(::tyloi::kInt32).expand({3, b, t}).contiguous();
        auto full = cache && request.positions[layer].defined()
                        ? ::tyloi::cuda::cat({request.positions[layer], current}, -1)
                        : current;
        pos = {current, full};
    }
    DenseTensor x;
    {
        DenseTensor e;
        {
            e = execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*embedding_fusion,
#ifdef TYLOINFER_NATIVE_CUDA
                                    rms_norm(embeds, dynamic_cast<const DenseTensor&>(*embedding_norm) + 1, config.eps));
#else
                                    rms_norm(embeds, embedding_norm + 1, config.eps));
#endif
        }
        // Release prior-stage tensors before later projections allocate.
        embeds = {};
        DenseTensor streams;
        streams = execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*hidden_fusion,
#ifdef TYLOINFER_NATIVE_CUDA
                                         rms_norm(hidden, dynamic_cast<const DenseTensor&>(*hidden_norm) + 1, config.eps)
#else
                                         rms_norm(hidden, hidden_norm + 1, config.eps)
#endif
                                             .reshape({b, t, config.streams, config.hidden}));
        x = (streams + e.unsqueeze(-2)).reshape({b, t, config.streams * config.hidden});
    }
    DenseTensor multi;
    {
        auto &block = *layers[layer];
        // Teacher-forced rows each receive the target hidden state. Only
        // the attention input contributes persistent predictor history;
        // attention readout, FFN and final mixer have no consumer here.
        if(history_only) {
            auto mix=block.attention_gr.pre(x);
            block.qsa->forward(*request.layer_states[layer]->qsa,mix.mixed,pos[0],pos[1],true,nullptr,{},true,true);
            multi = DenseTensor{};
        } else {
            auto [attention_input,attention_residual,attention_gate]=block.attention_gr.pre(x);
            auto attention_output=block.qsa->forward(*request.layer_states[layer]->qsa,attention_input,pos[0],pos[1],cache);
            x=block.attention_gr.post(attention_output,attention_residual,attention_gate);
            attention_output={};
            auto [ffn_input,ffn_residual,ffn_gate]=block.ffn_gr.pre(x);
            auto ffn_output=block.ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                *execution,
#endif
                ffn_input);
            multi = block.ffn_gr.post(ffn_output,ffn_residual,ffn_gate);
        }
    }
    x = {};
    auto result = history_only?DenseTensor{}:final_mixer->pre(multi).mixed;
    if (cache) {
        request.positions[layer] = pos[1];
        request.batch = b;
        request.lengths[layer] = start + t;
    }
    return {std::move(result), std::move(multi)};
}

tyloi::models::MtpStep Predictor::step(tyloi::models::MtpState& value, const tyloi::Tensor& input_hidden, const tyloi::Tensor& input_ids,
                                      const tyloi::Tensor& input_positions) const {
    auto& request=state<PredictorState>(value);
    const auto& hidden=input_hidden.dense();
    const auto& ids=input_ids.dense();
    auto positions=input_positions.defined()?input_positions.dense():DenseTensor{};
    if (positions.defined()) {
        if (positions.dim()==1) positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if (positions.dim()==2) positions=positions.unsqueeze(1);
        if (positions.dim()!=3 || positions.size(0)!=3 || positions.size(1)!=ids.size(0) || positions.size(2)!=ids.size(1))
            throw std::invalid_argument("Qwen4 MTP positions require [T]/[3,T]/[3,B,T]");
        positions=positions.to(::tyloi::kInt32).contiguous();
    }
    const auto graph=::tyloi::cuda::options::mtp_predictor_graph();
    if((!graph || *graph) && execution->config.cache.pipeline && execution->executor().options().moe_ffn.two_stage &&
        execution->executor().mmq().mode==KlMmqMode::Default &&
        execution->weights.transfer()==::tyloi::weights::Cache::Transfer::coordinated &&
        streamed_decode_graphs(*execution) && request.batch==1 && ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
#ifdef TYLOINFER_NATIVE_CUDA
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm->numel() &&
#else
        hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) && hidden.size(2)==hidden_norm.numel() &&
#endif
        ids.is_cuda() && hidden.is_cuda() && !request.lengths.empty() && request.lengths[0]>0 &&
        layers[0]->qsa && layers[0]->prefetch && request.positions[0].dim()==3 && request.positions[0].size(1)==1) {
        if(!request.graph_registry)request.graph_registry=std::make_unique<Qwen4MtpGraph>();
        return request.graph_registry->run(*this,request,hidden,ids,positions);
    }
    auto result = evaluate(value,hidden, ids, 0, true, positions);
    return {std::make_shared<DenseTensor>(std::move(result.first)),
            std::make_shared<DenseTensor>(std::move(result.second))};
}

void Predictor::history_step(tyloi::models::MtpState& value,DenseTensor hidden,DenseTensor ids,DenseTensor positions,bool scalar_chain) const {
    auto& request=state<PredictorState>(value);
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        TYLOINFER_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 &&
            positions.size(1)==ids.size(0) && positions.size(2)==ids.size(1),"invalid MTP history positions");
        positions=positions.to(::tyloi::kInt32).contiguous();
    }
    const auto graph=::tyloi::cuda::options::mtp_predictor_graph();
    if(request.reuse_decode_graphs && (!graph || *graph) && execution->config.cache.pipeline &&
            execution->executor().options().moe_ffn.two_stage && execution->executor().mmq().mode==KlMmqMode::Default &&
            execution->weights.transfer()==::tyloi::weights::Cache::Transfer::coordinated && streamed_decode_graphs(*execution) && layers.size()==1 && request.batch==1 &&
            ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 && ids.size(1)<=8 &&
            hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==ids.size(1) &&
#ifdef TYLOINFER_NATIVE_CUDA
            hidden.size(2)==hidden_norm->numel() && ids.is_cuda() && hidden.is_cuda() &&
#else
            hidden.size(2)==hidden_norm.numel() && ids.is_cuda() && hidden.is_cuda() &&
#endif
            !request.lengths.empty() && request.lengths[0]>0 && layers[0]->qsa && layers[0]->prefetch &&
            request.positions[0].dim()==3 && request.positions[0].size(1)==1) {
        if(!request.graph_registry)request.graph_registry=std::make_unique<Qwen4MtpGraph>();
        (void)request.graph_registry->run(*this,request,hidden,ids,positions,true,scalar_chain);
    } else if(scalar_chain && ids.size(1)>1) {
        for(int64_t row=0;row<ids.size(1);++row) {
            auto row_positions=positions.defined()?positions.narrow(-1,row,1):DenseTensor{};
            (void)step(value,hidden.narrow(1,row,1),ids.narrow(1,row,1),
                       row_positions.defined()?&row_positions:nullptr);
        }
    } else (void)step(value,hidden,ids,positions.defined()?&positions:nullptr);
}

void Predictor::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= request.position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(*request.layer_states[index]->qsa,position);
    if (request.positions[index].defined())
        request.positions[index] = request.positions[index].narrow(-1, 0, position);
    request.lengths[index] = position;
}

bool Predictor::teacher_forced_prompt_prime() const noexcept { return true; }

bool Predictor::target_bootstrap_decode() const noexcept { return true; }

bool Predictor::preserve_output_dtype() const noexcept { return true; }

std::unique_ptr<tyloi::engine::mtp::DepthPolicy> Predictor::depth_policy(
        int maximum_depth, tyloi::engine::mtp::PolicyState& state) const {
    const auto* option = tyloi::cuda::env_lookup("TYLOINFER_MTP_METAL_POLICY");
    const bool batch = !option || option[0] != '0';
    return std::make_unique<tyloi::engine::mtp::AdaptivePolicy>(maximum_depth, &state, true, batch);
}

tyloi::models::qwen4_exp::BlockSnapshot capture_block_session(
        const Qwen4Block& block,const LayerState& layer,int64_t position,std::size_t& bytes) {
    tyloi::models::qwen4_exp::BlockSnapshot state;
    const auto save=[&](const DenseTensor& tensor) {
        TYLOINFER_RUNTIME_CHECK(tensor.defined(),"Qwen4 recurrent session state is unavailable");
        auto saved=std::make_shared<DenseTensor>(tensor.clone());bytes+=session_tensor_bytes(*saved);return saved;
    };
    if(block.qsa)block.qsa->capture_session(*layer.qsa,state,position,bytes);
    if(block.gdn) {
        state.convolution=save(layer.gdn.conv.dense());state.recurrent=save(layer.gdn.recurrent.dense());
    }
    if(block.ple) {
        state.ple_convolution=save(block.ple->conv_state(layer.ple).dense());
        const auto& context=block.ple->session_context(layer.ple);
        state.ple_batch=context.batch;state.ple_tokens=context.tokens;
        bytes+=state.ple_tokens.size()*sizeof(int64_t);
    }
    return state;
}

void restore_block_session(Qwen4Block& block,LayerState& layer,
        const tyloi::models::qwen4_exp::BlockSnapshot& state,int64_t position) {
    block.reset(layer,1);
    if(block.qsa)block.qsa->restore_session(*layer.qsa,state,position);
    if(block.gdn) {
        const auto* convolution=dynamic_cast<const DenseTensor*>(state.convolution.get());
        const auto* recurrent=dynamic_cast<const DenseTensor*>(state.recurrent.get());
        TYLOINFER_RUNTIME_CHECK(convolution && recurrent,"missing Qwen4 GDN session state");
        block.gdn->restore_session(layer.gdn,*convolution,*recurrent,layer.retain_graph_storage);
    }
    if(block.ple) {
        const auto* convolution=dynamic_cast<const DenseTensor*>(state.ple_convolution.get());
        TYLOINFER_RUNTIME_CHECK(convolution,"missing Qwen4 PLE session state");
        block.ple->restore_session(layer.ple,*convolution,{state.ple_batch,state.ple_tokens},layer.retain_graph_storage);
    }
}

bool Predictor::supports_session_state() const noexcept {
    return execution && !execution->offload && !layers.empty();
}

MtpSessionState Predictor::capture_session_state(const tyloi::models::MtpState& value, int64_t target_position,const tyloi::Tensor& input_hidden) const {
    const auto& request=this->state<PredictorState>(value);
    const auto& hidden=input_hidden.dense();
    const auto position=target_position-1;
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && position>0 && position<=request.position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm->numel(),
        "Qwen4 MTP session boundary is unavailable");
#else
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && position>0 && position<=request.position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
#endif
    MtpSessionState state;state.cache_pos=position;
    auto& snapshot=state.payload.emplace<tyloi::models::qwen4_exp::MtpSnapshot>();
    for(std::size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,request.lengths[i]);
        snapshot.lengths.push_back(count);
        snapshot.blocks.push_back(count?capture_block_session(*layers[i],*request.layer_states[i],count,state.bytes):tyloi::models::qwen4_exp::BlockSnapshot{});
        ::tyloi::Tensor pos;
        if(count) {
            pos=std::make_shared<DenseTensor>(request.positions[i].narrow(-1,0,count).clone());
            state.bytes+=session_tensor_bytes(*pos);
        }
        snapshot.positions.push_back(std::move(pos));
    }
    auto saved_hidden=hidden.clone();state.bytes+=session_tensor_bytes(saved_hidden);
    state.last_target_hidden=std::make_shared<DenseTensor>(std::move(saved_hidden));
    return state;
}

void Predictor::restore_session_state(tyloi::models::MtpState& value, const MtpSessionState& state) const {
    auto& request=this->state<PredictorState>(value);
    const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::MtpSnapshot>(&state.payload);
    const auto* hidden=dynamic_cast<const DenseTensor*>(state.last_target_hidden.get());
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && state.cache_pos>0 &&
        snapshot && snapshot->blocks.size()==layers.size() && snapshot->positions.size()==layers.size() &&
        snapshot->lengths.size()==layers.size() && snapshot->lengths.front()==state.cache_pos &&
        hidden && hidden->defined() && hidden->dim()==3 &&
        hidden->size(0)==1 && hidden->size(1)==1 &&
        hidden->size(2)==hidden_norm->numel(),"incompatible Qwen4 MTP session state");
#else
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && state.cache_pos>0 &&
        snapshot && snapshot->blocks.size()==layers.size() && snapshot->positions.size()==layers.size() &&
        snapshot->lengths.size()==layers.size() && snapshot->lengths.front()==state.cache_pos &&
        hidden && hidden->defined() && hidden->dim()==3 &&
        hidden->size(0)==1 && hidden->size(1)==1 &&
        hidden->size(2)==hidden_norm.numel(),"incompatible Qwen4 MTP session state");
#endif
    reset(value,1);
    for(std::size_t i=0;i<layers.size();++i)if(snapshot->lengths[i]) {
        restore_block_session(*layers[i],*request.layer_states[i],snapshot->blocks[i],snapshot->lengths[i]);
        const auto* position=snapshot->positions[i].representation<DenseTensor>();
        TYLOINFER_RUNTIME_CHECK(position && position->defined(),"incompatible Qwen4 MTP session state");
        request.positions[i]=position->clone();request.lengths[i]=snapshot->lengths[i];
    }
}

} // namespace tyloi::cuda::qwen4_exp

namespace tyloi::cuda {
using qwen4_exp::Model;
namespace {

::tyloi::DenseTensor qwen4_last_hidden(::tyloi::DenseTensor hidden) {
    return hidden.index({::tyloi::indexing::Slice(), -1, ::tyloi::indexing::Slice()});
}
::tyloi::DenseTensor qwen4_logits_from_hidden(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden) {
    const auto cap = model.metadata.final_logit_softcapping;
    auto logits = model.logits(hidden).dense();
    return cap > 0.0 ? ::tyloi::cuda::tanh(logits / cap) * cap : logits;
}
::tyloi::DenseTensor qwen4_finalize_hidden(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden,
                                           std::int64_t, std::int64_t) {
    auto* window=DecodeWindow::recording();
    GatedResidualMarker marker;
    if(window && ::tyloi::cuda::options::mhc_timings())marker=[window](const char* phase) {
        window->mark(-1,(std::string("final_mhc_")+phase).c_str());
    };
    return model.final_mixer->pre(hidden,marker).mixed;
}

::tyloi::DenseTensor qwen4_next_token(const qwen4_exp::Model& model, ::tyloi::DenseTensor hidden) {
    auto* window=DecodeWindow::recording();
    if(window)window->mark(-1,"lm_head_begin");
    auto projected=model.logits(hidden).dense();
    if(window)window->mark(-1,"lm_head_projection_end");
    auto output=::tyloi::cuda::argmax(projected,-1).to(::tyloi::kInt64);
    if(window)window->mark(-1,"lm_head_sample_end");
    return output;
}

void qwen4_validate_forward(const qwen4_exp::Model& model, std::int64_t tokens,
                            std::int64_t position, bool cache_override) {
    tyloi::models::require_model(tokens > 0 &&
        position + tokens <= model.metadata.max_position_embeddings && !cache_override,
        "Qwen4 requires unpadded causal cache positions");
}

::tyloi::DenseTensor qwen4_forward(qwen4_exp::Model& model, qwen4_exp::State& state,
                                   tyloi::models::Input input) {
    auto scope = model.execution_scope();
    const auto primary = model.execution->config.layer_placement.primary_device();
    tyloi::models::require_model(bool(input.ids), "Qwen4 forward requires token ids");
    auto ids = model.device_ids(dynamic_cast<const DenseTensor&>(*input.ids));
    auto embeddings = input.embeddings
        ? model.execution->executor().collectives().copy(dynamic_cast<const DenseTensor&>(*input.embeddings), primary).contiguous()
        : model.embed_tokens(ids);
    const auto B = ids.size(0), T = ids.size(1);
    tyloi::models::require_model(embeddings.dim() == 3 && embeddings.size(0) == B &&
                                     embeddings.size(1) == T && embeddings.size(2) == model.hidden_size(),
                                 "inputs_embeds shape must match [batch,tokens,hidden_size]");
    qwen4_validate_forward(model, T, state.cache_pos, bool(input.cache_positions));
    tyloi::models::require_model(state.speculative_start < 0 || state.speculative_suffix_forward,
                                 "commit or roll back the pending speculative pass before forwarding");
    const auto confirmed = input.confirmed;
    tyloi::models::require_model(
        confirmed >= 0 && (confirmed == 0 ||
            (confirmed < T && B == 1 && state.cache_pos > 0 &&
             !input.cache_positions &&
             model.supports_speculation())),
        "unsupported speculative backbone geometry");
    if (confirmed > 0) {
        tyloi::models::require_model(state.cache_pos + T <= model.max_position_embeddings(),
                                     "speculative pass exceeds context capacity");
        state.speculative_start = state.cache_pos;
        state.speculative_confirmed = confirmed;
    }
    if ((state.batch != 0 && state.batch != B) || state.cache_pos == 0)
        state.reset(B);
    auto cache_positions = ::tyloi::cuda::arange(state.cache_pos, state.cache_pos + T,
                                                 ::tyloi::TensorOptions()
                                                     .device(::tyloi::kCUDA)
                                                     .dtype(::tyloi::kInt64));
    auto current = input.positions
        ? model.device_ids(dynamic_cast<const DenseTensor&>(*input.positions))
        : (state.decode_position_delta == 0 ? cache_positions
                                            : cache_positions + state.decode_position_delta);
    if ((current.dim() == 2 || current.dim() == 3) && current.size(0) == 4)
        current = current.narrow(0, 1, 3);
    auto full_positions = state.positions.defined()
        ? ::tyloi::cuda::cat({state.positions, current}, -1)
        : current;
    tyloi::models::require_model((current.dim() == 1 || (current.dim() == 2 && current.size(0) == 3) ||
                                     (current.dim() == 3 && current.size(0) == 3 && current.size(1) == B)) &&
                                     current.size(-1) == T,
                                 "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    DenseTensor host_ids, cpu_positions, cpu_cache_positions;
    const bool pinned = model.execution->config.cache.pipeline ||
                        model.execution->config.dense_cpu_layer_count > 0;
    if (pinned) {
        host_ids = ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        cpu_positions = current.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        cpu_cache_positions = cache_positions.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        for (std::size_t i=0;i<model.blocks.size();++i)
            model.blocks[i]->prefetch_token_ids(*state.layers[i], host_ids);
    }
    auto hidden = model.repeat_hidden(
        model.execution->executor().collectives().copy(embeddings, primary).contiguous(), model.metadata.hc_mult);
    state.begin_forward();
    const auto trace_layer = [&] {
        if (!input.trace) return;
        input.trace->push_back(
            std::make_shared<DenseTensor>(model.execution->executor().collectives().copy(hidden, primary).to(::tyloi::kFloat32).clone()));
    };
    if (input.trace)
        input.trace->push_back(std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).clone()));
    for (std::size_t i=0;i<model.blocks.size();++i) {
        auto& block = model.blocks[i];
        MfqCudaGuard guard(block->cuda_device);
        auto local_ids = block->cpu_offloaded
            ? host_ids
            : model.execution->executor().collectives().copy(ids, block->cuda_device);
        auto local_pos = block->cpu_offloaded
            ? cpu_positions
            : model.execution->executor().collectives().copy(current, block->cuda_device);
        auto local_cache_positions = block->cpu_offloaded
            ? cpu_cache_positions
            : model.execution->executor().collectives().copy(cache_positions, block->cuda_device);
        tyloi::cuda::BlockContext context;
        context.token_ids = local_ids;
        context.host_token_ids = host_ids;
        context.positions = local_pos;
        context.full_positions = model.execution->executor().collectives().copy(full_positions, block->cuda_device);
        context.cache_position = state.cache_pos;
        context.confirmed_prefix = confirmed;
        context.planned_kv_length = input.kv_length;
        context.decode_attention_parts = input.attention_parts;
        context.cache_positions = local_cache_positions;
        hidden = block->cpu_offloaded
            ? hidden.to(::tyloi::kCPU).contiguous()
            : model.execution->executor().collectives().copy(hidden, block->cuda_device);
        hidden = block->forward_context(*state.layers[i], *model.execution, std::move(hidden), context);
        trace_layer();
    }
    state.finish_forward(full_positions, B, T);
    state.cache_pos += T;
    hidden = model.execution->executor().collectives().copy(hidden, primary);
    if (input.raw_hidden) *input.raw_hidden = hidden;
    auto finalized = qwen4_finalize_hidden(model, std::move(hidden), B, T);
    return finalized;
}

} // namespace
bool Model::prefetch_token_rows(State& state,const ::tyloi::Tensor& host_ids) {
    if (!state.reuse_decode_graphs || !state.graph_registry) return false;
    bool issued=false;
    for(std::size_t i=0;i<blocks.size();++i) {
        const auto& block=blocks[i];
        if(block->ple && block->ple->graph_supported()) {
            block->ple->prefetch_tokens(state.layers[i]->ple,host_ids.dense(),true);
            issued=true;
        }
    }
    return issued;
}
struct Qwen4DecodeGraph {
    static inline thread_local bool startup_warmup=false;
    struct PleStage {
        qwen4_exp::Ple* ple;
        qwen4_exp::Ple::State* state;
        ::tyloi::DenseTensor rows;
        std::shared_ptr<RowStage> transfer;
        std::vector<qwen4_exp::NgramEmbedding::Context> prefix_context;
        HostBuffer flags{128,true};
        uint32_t* device=nullptr;
        bool profile=false;
        uint64_t calls=0,route_wait_ns=0,upload_ns=0,notify_ns=0;
        PleStage(qwen4_exp::Ple* p,qwen4_exp::Ple::State& request,int tokens,int hidden):ple(p),state(&request) {
            const auto timing=::tyloi::cuda::options::trace_ple_timings();profile=timing && *timing;
            charge_tensor_host_bytes(128);
            TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),flags.data(),0));
            std::memset(flags.data(),0,128);publish_mapped_flag(host()+16);
            transfer=ple->graph_stage(*state,tokens,::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()});
            rows=transfer->output().dense().reshape({1,tokens,hidden});
        }
        ~PleStage(){
            if(profile)std::cerr<<"ple_stage_host mapped_rows="<<transfer->mapped_rows()<<" calls="<<calls
                <<" route_wait_ns="<<route_wait_ns<<" upload_ns="<<upload_ns<<" notify_ns="<<notify_ns<<'\n';
            tensor_host_bytes.fetch_sub(128);
        }
        uint32_t* host(){return static_cast<uint32_t*>(flags.data());}
    };
    struct Cell {
        ::tyloi::DenseTensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host,compact_host;
        qwen4_exp::MtpCompactControl compact_control;
        unsigned used_kinds=0;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<qwen4_exp::RecurrentGraphCheckpoint> checkpoints;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<qwen4_exp::Qwen4Block>>& blocks,
                const std::vector<std::unique_ptr<qwen4_exp::LayerState>>& layers,cudaStream_t stream,int confirmed,
                std::vector<qwen4_exp::RecurrentGraphCheckpoint>& shared_checkpoints,bool prediction)
            :window(stream,::tyloi::cuda::options::layer_timings()) {
            window.set_prediction_enabled(prediction);
            const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
            ids=::tyloi::cuda::zeros({1,tokens},options);positions=::tyloi::cuda::zeros({tokens},options);cache_positions=::tyloi::cuda::zeros({tokens},options);
            host_ids=::tyloi::cuda::zeros({1,tokens},options.device(::tyloi::kCPU));
            greedy_host=::tyloi::cuda::empty({tokens},options.device(::tyloi::kCPU).pinned_memory(true));
            compact_host=::tyloi::cuda::empty({tokens,64},options.device(::tyloi::kCPU).pinned_memory(true));
            if(confirmed && shared_checkpoints.empty())shared_checkpoints.resize(blocks.size());
            std::size_t layer=0;
            for(const auto& block:blocks) {
                auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),layers[layer]->ple,tokens,hidden));
                if(confirmed) {
                    auto& shared=shared_checkpoints[layer];
                    qwen4_exp::RecurrentGraphCheckpoint checkpoint;
                    if(b.gdn) {
                        const auto state=b.gdn->graph_state(layers[layer]->gdn);
                        while(shared.gdn_conv.size()<std::size_t(tokens-confirmed)) {
                            shared.gdn_conv.push_back(::tyloi::cuda::empty_like(state[0]->dense()));
                            shared.gdn_state.push_back(::tyloi::cuda::empty_like(state[1]->dense()));
                        }
                        checkpoint.gdn_conv.assign(shared.gdn_conv.begin(),shared.gdn_conv.begin()+tokens-confirmed);
                        checkpoint.gdn_state.assign(shared.gdn_state.begin(),shared.gdn_state.begin()+tokens-confirmed);
                    }
                    if(b.ple) {
                        while(shared.ple_conv.size()<std::size_t(tokens-confirmed))
                            shared.ple_conv.push_back(::tyloi::cuda::empty_like(b.ple->conv_state(layers[layer]->ple).dense()));
                        checkpoint.ple_conv.assign(shared.ple_conv.begin(),shared.ple_conv.begin()+tokens-confirmed);
                    }
                    checkpoints.push_back(std::move(checkpoint));
                }
                ++layer;
            }
        }
    };
    MfqCudaStream stream,ple_stream;
    std::map<std::tuple<int,int,int,bool>,std::unique_ptr<Cell>> cells;
    // Verification widths execute serially and commit/rollback before the next
    // window. Their snapshots can share stable allocations across graph shapes.
    std::vector<qwen4_exp::RecurrentGraphCheckpoint> shared_checkpoints;
    qwen4_exp::State& request;
    std::vector<std::pair<qwen4_exp::Qsa*,qwen4_exp::Qsa::State*>> qsa;
    ::tyloi::DenseTensor position_storage;
    int64_t kv_capacity=0;
    bool failed=false;
    ::tyloi::DenseTensor last_compact;
    explicit Qwen4DecodeGraph(qwen4_exp::Model& model,qwen4_exp::State& state)
        :stream(mfq_get_current_cuda_stream()),ple_stream(mfq_get_stream_from_pool()),request(state) {
        const bool enabled=model.execution->weights.lookahead(
            ::tyloi::weights::Cache::Lookahead::link,request.router_lookahead);
        for(std::size_t i=0;i<model.blocks.size();++i) {
            auto& current=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
            auto& layer=*request.layers[i];
            layer.next_router={};layer.next_prefetch={};
            if(enabled && i+1<model.blocks.size()) {
                auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                TYLOINFER_RUNTIME_CHECK(next.router && next.prefetch,"router lookahead requires the tiered FFN pipeline");
                layer.next_router=next.router;layer.next_prefetch=next.prefetch;
            }
        }
        for(std::size_t i=0;i<model.blocks.size();++i)
            if(auto* attention=model.blocks[i]->qsa.get())qsa.emplace_back(attention,request.layers[i]->qsa.get());
    }
    ~Qwen4DecodeGraph() {(void)cudaStreamSynchronize(ple_stream.stream());cells.clear();}
    void restore_legacy() {
        last_compact={};
        cells.clear();shared_checkpoints.clear();
        for(auto [attention,state]:qsa)attention->leave_graph(*state);
        for(auto& layer:request.layers)layer->retain_graph_storage=false;
    }
    bool reusable() const {
        return !failed && !qsa.empty() && std::all_of(qsa.begin(),qsa.end(),
            [](const auto& attention){return attention.first->graph_capacity(*attention.second)>0;});
    }

    ::tyloi::DenseTensor run(qwen4_exp::Model& model,::tyloi::DenseTensor ids,int kind,
                                  ::tyloi::DenseTensor* raw_hidden=nullptr,int confirmed=0,bool* initialized=nullptr,
                                  const ::tyloi::DenseTensor& host_ids={},::tyloi::DenseTensor* logits=nullptr,
                                  std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true) {
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        last_compact={};
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
        const int tokens=static_cast<int>(ids.size(1));
        // These consumers use the same single-row backbone. Keep its hidden
        // state, logits and selected token in one graph instead of reserving
        // a complete set of layer buffers for each output representation.
        const int result_kind=kind;
        if(tokens==1 && kind>=1 && kind<=4)kind=4;
        model.execution->weights.retire_prefill();
        const auto required=request.cache_pos+tokens;
        TYLOINFER_RUNTIME_CHECK(required<=model.max_position_embeddings(),"Qwen4 graph exceeds context capacity");
        if(!qsa.empty() && qsa.front().first->graph_capacity(*qsa.front().second)) {
            model.execution->kv->ensure(required);
            kv_capacity=qsa.front().first->graph_capacity(*qsa.front().second);
        }
        if(required>kv_capacity) {
            model.execution->weights.finish_exchange();
            restore_legacy();
            // Small growing chunks preserve dynamic KV allocation without
            // recapturing all MTP batch shapes at every few decode tokens.
            const int64_t chunk=raw_hidden?2048:256;
            kv_capacity=std::min<int64_t>(model.max_position_embeddings(),std::max<int64_t>(chunk,kv_capacity));
            const auto planned=std::max(required,request.decode_reserve_until);
            while(kv_capacity<planned)
                kv_capacity=std::min(model.max_position_embeddings(),kv_capacity>model.max_position_embeddings()/2
                    ? model.max_position_embeddings() : kv_capacity*2);
            if(const auto* trace=tyloi::cuda::env_lookup("TYLOINFER_RUNTIME_TRACE_CUDA_GRAPH");trace && trace[0]=='1')
                std::cerr<<"runtime_qwen4_graph_capacity position="<<request.cache_pos
                    <<" required="<<required<<" planned="<<planned<<" capacity="<<kv_capacity<<'\n';
        }
        const bool prediction=tokens==1 && model.execution->weights.lookahead(
            ::tyloi::weights::Cache::Lookahead::capture,request.router_lookahead);
        auto& owner=cells[{tokens,kind,confirmed,prediction}];
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.blocks,request.layers,stream.stream(),confirmed,shared_checkpoints,prediction);
        owner->window.set_profile_geometry(raw_hidden?"mtp_target":"target_decode",tokens);
        auto& cell=*owner;
        if(kind==4 && tokens>1)cell.compact_control.set(compact_sampling,"target",tokens);
        // A retained graph's first replay can still pay cold transfer/cache
        // costs after prefill. Exclude that replay from steady MTP policy data.
        if(initialized)*initialized=!cell.window.valid() || !(cell.used_kinds&(1u<<result_kind));
        if(host_ids.defined()) {
            TYLOINFER_RUNTIME_CHECK(!host_ids.is_cuda() && host_ids.scalar_type()==::tyloi::kInt64 &&
                host_ids.sizes()==ids.sizes(),"Qwen4 graph host token geometry changed");
            cell.host_ids.copy_(host_ids);
        } else cell.host_ids.copy_(ids.to(::tyloi::kCPU,::tyloi::kInt64));
        cell.ids.copy_(ids);
        mfq_qwen4_exp::fill_decode_positions(cell.cache_positions,cell.positions,request.cache_pos,request.decode_position_delta);
        try {
            // PLE depends only on token IDs and the committed ngram context.
            // Start its read before waiting for adaptive expert exchanges,
            // rather than leaving that CPU interval outside the I/O overlap.
            // Task reset repeats issue() idempotently; it never advances the
            // context, which is still committed by upload_graph().
            const auto* early_ple=tyloi::cuda::env_lookup("TYLOINFER_PLE_EARLY_ISSUE");
            if(!early_ple || early_ple[0]!='0')
                for(const auto& stage:cell.ple)stage->ple->prefetch_tokens(*stage->state,cell.host_ids);
            model.execution->weights.finish_exchange();
            // Prefix restoration and eager prefill update the same KV/index
            // storage; refresh pooled keys even when this graph is already valid.
            for(auto [attention,state]:qsa)attention->prepare_graph(*state,request.positions,kv_capacity);
            if(!cell.window.valid()) {
                model.execution->weights.prepare_graph(::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()});
                DecodeGraphBranchScope branch_scope(
                    model.execution->executor().mode().ffn.serial);
                const auto capture_started=std::chrono::steady_clock::now();
                struct Saved {::tyloi::Tensor* target;::tyloi::DenseTensor saved;const void* address;};
                std::vector<Saved> states;
                if(!startup_warmup)
                    for(std::size_t i=0;i<model.blocks.size();++i)
                        for(auto* value:model.blocks[i]->graph_warmup_state(*request.layers[i]))
                            states.push_back({value,value->dense().clone(),value->dense().data_ptr()});
                auto restore=[&] {
                    for(const auto& state:states) {
                        TYLOINFER_RUNTIME_CHECK(state.target->dense().data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->dense().copy_(state.saved);
                    }
                };
                cell.window.capture([&] {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.repeat_hidden(model.embed_tokens(cell.ids),model.metadata.hc_mult);
                    std::size_t ple_index=0;int layer=0;
                    std::optional<GatedResidual::Result> pending_residual;
                    const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
                    const bool chain_post=(!chain_setting || *chain_setting) && tokens<=8;
                    for(const auto& block:model.blocks) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);::tyloi::DenseTensor rows;
                        if(b.ple) {
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();rows=stage->rows;
                            cell.window.enroll({stage,
                                [&cell,stage] {std::memset(stage->flags.data(),0,128);stage->ple->prefetch_tokens(*stage->state,cell.host_ids);},
                                [this,&cell,stage,confirmed] {
                                    using Clock=std::chrono::steady_clock;
                                    auto point=stage->profile?Clock::now():Clock::time_point{};
                                    const auto sample=[&](uint64_t& total) {
                                        if(!stage->profile)return;
                                        const auto now=Clock::now();
                                        total+=std::chrono::duration_cast<std::chrono::nanoseconds>(now-point).count();point=now;
                                    };
                                    wait_route_publication(stage->host(),stream.stream());
                                    sample(stage->route_wait_ns);
                                    MfqCudaStreamGuard guard(ple_stream);
                                    if(confirmed) {
                                        stage->prefix_context.clear();
                                        for(int i=confirmed;i<cell.ids.size(1);++i)
                                            stage->prefix_context.push_back(stage->ple->graph_prefix_context(*stage->state,cell.host_ids,i));
                                    }
                                    stage->ple->upload_graph(*stage->state,cell.host_ids,*stage->transfer);
                                    sample(stage->upload_ns);
                                    if(stage->transfer->mapped_rows())publish_mapped_flag(stage->host()+16);
                                    else {
                                        TYLOINFER_CUDA_CHECK(cudaLaunchHostFunc(ple_stream.stream(),[](void* p){publish_mapped_flag(static_cast<uint32_t*>(p));},stage->host()+16));
                                        check_stream_progress(ple_stream.stream());
                                    }
                                    sample(stage->notify_ns);if(stage->profile)++stage->calls;
                                },{},[stage]{publish_mapped_flag(stage->host()+16);}});
                            signal_mapped_flag(stage->device,stream.stream());wait_mapped_flag(stage->device+16,stream.stream());
                            stage->transfer->decode();
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
                        std::optional<GatedResidual::Result> next_residual;
                        bool defer=false;
                        if(chain_post && size_t(layer+1)<model.blocks.size()) {
                            auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer+1]);
                            defer=!next.ple && next.attention_gr.can_fuse(pending_residual?pending_residual->residual:hidden);
                        }
                        hidden=b.execute_graph(*request.layers[layer],*model.execution,std::move(hidden),cell.ids,cell.positions,
                            cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr,
                            pending_residual,defer?&next_residual:nullptr);
                        pending_residual=std::move(next_residual);
                        cell.window.mark(layer++,"block_end");
                    }
                    if(kind>=3)cell.raw_hidden=hidden.clone();
                    hidden=qwen4_finalize_hidden(model,std::move(hidden),1,tokens);
                    if(kind>=3) {
                        cell.output=std::move(hidden);
                        if(kind==4) {
                            cell.logits=qwen4_logits_from_hidden(model, cell.output);
                            if(tokens>1 && cell.logits.size(-1)>=64) {
                                auto compact=sample_top_k_compact_if_cuda(cell.logits.reshape({tokens,-1}).contiguous(),64,
                                    cell.compact_control.device);
                                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.compact_host.data_ptr(),compact.data_ptr(),tokens*64*sizeof(int64_t),
                                    cudaMemcpyDeviceToHost,stream.stream()));
                            }
                            auto greedy=sample_greedy_cuda(cell.logits.reshape({tokens,-1}).contiguous());
                            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),tokens*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
                        }
                    }
                    else if(kind==2)cell.output=qwen4_next_token(model,qwen4_last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.logits(qwen4_last_hidden(std::move(hidden))).dense();
                    else cell.output=qwen4_logits_from_hidden(model, std::move(hidden));
                    cell.window.mark(-1,"window_end");
                    auto audit=request.layers.front()->router_audit;
                    if(audit)cell.window.enroll({audit.get(),[]{},[]{},[audit]{audit->finish_sample();},{}});
                },restore,[&]{cell.output={};cell.raw_hidden={};cell.logits={};});
                ++request.decode_graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=target rows="<<tokens<<" kind="<<kind
                    <<" confirmed="<<confirmed<<" segments="<<cell.window.segments()<<" count="<<request.decode_graph_captures
                    <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            cell.window.run();cell.used_kinds|=1u<<result_kind;
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.blocks) {
                    auto& b=static_cast<qwen4_exp::Qwen4Block&>(*block);
                    const auto& checkpoint=cell.checkpoints[layer];
                    auto& layer_state=*request.layers[layer++];
                    const auto position=request.cache_pos+confirmed;
                    if(b.gdn)b.gdn->bind_graph_checkpoint(layer_state.gdn,checkpoint.gdn_conv,checkpoint.gdn_state,position);
                    if(b.ple)b.ple->bind_graph_checkpoint(layer_state.ple,checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context,position);
                }
                request.speculative_start=request.cache_pos;request.speculative_confirmed=confirmed;
                request.speculative_prefix_retention=true;
            }
            for(std::size_t i=0;i<model.blocks.size();++i)
                if(model.blocks[i]->qsa)model.blocks[i]->qsa->advance_graph(*request.layers[i]->qsa,tokens);
            request.cache_pos+=tokens;
            request.positions=append_decode_positions(request.positions,cell.positions,position_storage,
                model.max_position_embeddings());request.batch=1;
            // MTP can retain history until a later probe. Each step must own its
            // raw residual rather than alias the next replay's capture buffer.
            if(raw_hidden)*raw_hidden=cell.raw_hidden.clone();
            if(kind==4 && tokens>1 && compact_sampling && cell.logits.size(-1)>=64)last_compact=cell.compact_host;
            if(logits)*logits=cell.logits.clone();
            if(greedy_tokens) {
                const auto* first=cell.greedy_host.data_ptr<int64_t>();
                greedy_tokens->assign(first,first+tokens);
            }
            if(tokens==1 && result_kind==1)return cell.logits.reshape({1,-1});
            if(tokens==1 && result_kind==2)return cell.greedy_host;
            return cell.output;
        } catch(...) {failed=true;throw;}
    }
};
::tyloi::Tensor qwen4_decode_graph_compact_topk(const qwen4_exp::Model& model,const qwen4_exp::State& state) {
    return state.graph_registry?state.graph_registry->last_compact: ::tyloi::DenseTensor{};
}

std::optional<::tyloi::Tensor> qwen4_decode_graph_logits(qwen4_exp::Model& model,qwen4_exp::State& state,::tyloi::Tensor ids_handle,int kind,
        const ::tyloi::Tensor& host_ids) {
    auto ids = ids_handle.dense();
    if (!streamed_decode_graphs(*model.execution)) return {};
    const auto decode=model.execution->weights.decode(model.execution->config.cache.pipeline);
    const bool tq=decode==::tyloi::weights::Cache::Decode::routed;
    if(decode==::tyloi::weights::Cache::Decode::none ||
            state.cache_pos==0 || state.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || state.positions.dim()!=1)return {};
    if(tq) {
        if(!tq_graph_supports_tokens(ids.size(1)))return {};
        for(const auto& block:model.blocks) {
            const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
            if(!b.tq_graph_ffn || (b.ple && !b.ple->graph_supported()))return {};
        }
    }
    TYLOINFER_RUNTIME_CHECK(state.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(model,state);
    return ::tyloi::Tensor(state.graph_registry->run(model,model.device_ids(std::move(ids)),kind,nullptr,0,nullptr,
        host_ids ? host_ids.dense() : ::tyloi::DenseTensor{}));
}

void qwen4_report_decode_graph(const qwen4_exp::Model& model,const qwen4_exp::State& state,std::ostream& output) {
    if(!state.graph_registry)return;
    for(const auto& [key,cell]:state.graph_registry->cells)
        for(const auto& phase:cell->window.gpu_phases())if(phase.samples)
            output<<"mtp_graph_phase tokens="<<std::get<0>(key)<<" confirmed="<<std::get<2>(key)
                <<" prediction="<<std::get<3>(key)
                <<" layer="<<phase.layer<<" from="<<phase.from<<" to="<<phase.to
                <<" samples="<<phase.samples<<" total_ns="<<phase.total_ns<<'\n';
}

std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(
        const qwen4_exp::Model& model,const qwen4_exp::State& state,const tyloi::models::Mtp* mtp) {
    std::vector<std::pair<std::string,double>> result;
    if(!::tyloi::cuda::options::layer_timings())return result;
    const auto append=[&](const auto& cells,const char* role,const char* attribute,const char* state) {
        for(const auto& [key,cell]:cells) {
            auto prefix=std::string("cuda_graph_profile_")+role+"_rows_"+
                std::to_string(std::get<0>(key))+"_"+attribute+"_"+std::to_string(std::get<1>(key))+
                "_"+state+"_"+std::to_string(std::get<2>(key))+"_";
            if constexpr(std::tuple_size_v<std::remove_cvref_t<decltype(key)>> == 4)
                prefix+="prediction_"+std::to_string(std::get<3>(key))+"_";
            struct Sample {int64_t total_ns=0,max_ns=0;uint64_t samples=0;};
            std::map<std::string,Sample> phases;
            uint64_t windows=0;
            for(const auto& phase:cell->window.gpu_phases())if(phase.samples) {
                auto& sample=phases[phase.from+"_to_"+phase.to];
                sample.total_ns+=phase.total_ns;sample.samples+=phase.samples;
                sample.max_ns=std::max(sample.max_ns,phase.max_ns);
                windows=std::max(windows,phase.samples);
            }
            if(!windows)continue;
            result.emplace_back(prefix+"windows",double(windows));
            for(const auto& [name,sample]:phases) {
                result.emplace_back(prefix+name+"_ms",sample.total_ns/1e6);
                result.emplace_back(prefix+name+"_samples",double(sample.samples));
                result.emplace_back(prefix+name+"_max_ms",sample.max_ns/1e6);
            }
        }
    };
    if(state.graph_registry)append(state.graph_registry->cells,"target","kind","confirmed");
    const auto* predictor=dynamic_cast<const qwen4_exp::Predictor*>(mtp);
    if(predictor && state.predictor && state.predictor->belongs_to(*predictor)) {
        const auto& request=static_cast<const qwen4_exp::PredictorState&>(*state.predictor);
        if(request.graph_registry)append(request.graph_registry->cells,"predictor","dtype","mode");
    }
    return result;
}

std::optional<::tyloi::Tensor> qwen4_decode_graph_hidden(
        qwen4_exp::Model& model,qwen4_exp::State& state,::tyloi::Tensor ids_handle,::tyloi::Tensor* raw_hidden,int64_t confirmed,
        bool* initialized,const ::tyloi::Tensor& host_ids,::tyloi::Tensor* logits,
        std::vector<int32_t>* greedy_tokens,bool compact_sampling) {
    auto ids = ids_handle.dense();
    if(initialized)*initialized=false;
    const auto decode=model.execution->weights.decode(model.execution->config.cache.pipeline);
    const bool tq=decode==::tyloi::weights::Cache::Decode::routed;
    if(!streamed_decode_graphs(*model.execution) ||
            decode==::tyloi::weights::Cache::Decode::none ||
            state.speculative_start>=0)return {};
    // Retained windows own initialized, reset recurrent/KV storage. The
    // first row of a new prompt can replay them without an eager FFN copy.
    if(state.cache_pos==0 && (!state.graph_registry || !state.graph_registry->reusable()))return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
        (state.cache_pos!=0 && state.positions.dim()!=1) ||
        confirmed<0 || (confirmed && confirmed>=ids.size(1)))return {};
    if(tq && !tq_graph_supports_tokens(ids.size(1)))return {};
    for(const auto& block:model.blocks) {
        const auto& b=static_cast<const qwen4_exp::Qwen4Block&>(*block);
        if((!b.prefetch && !b.tq_graph_ffn) || (b.ple && !b.ple->graph_supported()))return {};
    }
    TYLOINFER_RUNTIME_CHECK(state.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(model,state);
    ::tyloi::DenseTensor raw_result, logits_result;
    auto result = state.graph_registry->run(model,model.device_ids(std::move(ids)),(logits || greedy_tokens)?4:3,
        raw_hidden ? &raw_result : nullptr, static_cast<int>(confirmed),initialized,
        host_ids ? host_ids.dense() : ::tyloi::DenseTensor{}, logits ? &logits_result : nullptr,
        greedy_tokens,compact_sampling);
    if (raw_hidden) *raw_hidden = std::move(raw_result);
    if (logits) *logits = std::move(logits_result);
    return ::tyloi::Tensor(std::move(result));
}

void qwen4_release_decode_graphs(qwen4_exp::Model& model,qwen4_exp::State& state,tyloi::models::Mtp* mtp) {
    model.execution->weights.finish_exchange();
    if(state.graph_registry)state.graph_registry->restore_legacy();
    state.graph_registry.reset();
    for(auto& layer:state.layers)layer->retain_graph_storage=false;
    if(auto* predictor=dynamic_cast<qwen4_exp::Predictor*>(mtp); predictor && state.predictor && state.predictor->belongs_to(*predictor)) {
        auto& request=static_cast<qwen4_exp::PredictorState&>(*state.predictor);
        if(request.graph_registry)request.graph_registry->restore_legacy(*predictor,request);
        request.graph_registry.reset();
        for(auto& layer:request.layer_states)layer->retain_graph_storage=false;
    }
}

std::size_t qwen4_mtp_graph_reserve(const qwen4_exp::Model& model,const tyloi::models::Mtp* mtp,int maximum_draft_depth) {
    const auto* predictor=dynamic_cast<const qwen4_exp::Predictor*>(mtp);
    const auto* reuse=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_REUSE_GRAPHS");
    const auto* warmup=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!predictor || (reuse && reuse[0]=='0') || (warmup && warmup[0]=='0') ||
            !model.execution->config.cache.pipeline || !streamed_decode_graphs(*model.execution))return 0;
    // The common decode allowance covers verification through four rows.
    // Wider windows retain extra shared recurrent checkpoints and per-shape
    // graph/staging allocations. Price those before admitting expert arenas.
    const int extra_rows=std::max(0,std::min({8,predictor->maximum_draft_depth()+1,maximum_draft_depth+1})-4);
    const auto& c=model.config;
    std::size_t recurrent_bytes=0;
    for(int64_t layer=0;layer<c.layers;++layer)if(c.linear_layer(layer))
        recurrent_bytes+=(std::size_t(c.kernel-1)*(2*c.key_heads+c.value_heads)*c.linear_width+
            std::size_t(c.value_heads)*c.linear_width*c.linear_width)*sizeof(float);
    return std::size_t(extra_rows)*(recurrent_bytes+(std::size_t(256)<<20));
}

void qwen4_warmup_decode_graphs(qwen4_exp::Model& model,qwen4_exp::State& state,tyloi::models::Mtp* mtp,int maximum_draft_depth) {
    const auto* reuse=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_REUSE_GRAPHS");
    state.reuse_decode_graphs=!reuse || reuse[0]!='0';
    auto* predictor=dynamic_cast<qwen4_exp::Predictor*>(mtp);
    auto* predictor_state=predictor && state.predictor && state.predictor->belongs_to(*predictor)
        ? static_cast<qwen4_exp::PredictorState*>(state.predictor.get()) : nullptr;
    if(predictor_state)predictor_state->reuse_decode_graphs=state.reuse_decode_graphs;
    if(!predictor_state)predictor=nullptr;
    const auto decode=model.execution->weights.decode(model.execution->config.cache.pipeline);
    const bool tq=decode==::tyloi::weights::Cache::Decode::routed;
    // Keep TQ startup to the target decode window. Wider verification shapes
    // are captured on demand; predictor graph preparation remains separate.
    if(tq)predictor=nullptr;
    const auto* warmup=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!state.reuse_decode_graphs || (warmup && warmup[0]=='0') ||
        model.max_position_embeddings()<32 || !streamed_decode_graphs(*model.execution) ||
        decode==::tyloi::weights::Cache::Decode::none)return;
    for(std::size_t i=0;i<model.blocks.size();++i) {
        const auto& b=*model.blocks[i];
        if((!b.prefetch && !b.tq_graph_ffn) || (b.qsa && !b.qsa->graph_capacity(*state.layers[i]->qsa)) || (b.ple && !b.ple->graph_supported()))return;
    }
    // Startup state contains synthetic tokens and is discarded before serving.
    // Saving every recurrent state during each capture needlessly doubles its
    // transient footprint. Request-time captures still restore their real state.
    tyloi::weights::Cache::Warmup cache_warmup(model.execution->weights, true);
    struct StartupWarmup {
        StartupWarmup(){Qwen4DecodeGraph::startup_warmup=true;}
        ~StartupWarmup(){Qwen4DecodeGraph::startup_warmup=false;}
    } startup_scope;
    const auto start=std::chrono::steady_clock::now();
    auto scope=model.execution_scope();
    const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
    // Diverse synthetic IDs exercise more routed experts than sixteen copies
    // of one token. Only loading uses these rows; all request state is reset.
    std::vector<int64_t> warm_ids(16);
    for(std::size_t i=0;i<warm_ids.size();++i)warm_ids[i]=int64_t((i*7919+17)%model.metadata.vocab_size);
    auto ids=::tyloi::cuda::tensor(warm_ids,options.device(::tyloi::kCPU)).to(::tyloi::kCUDA).reshape({1,16});
    state.reset(1);
    if(tyloi::cuda::env_lookup("TYLOINFER_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=bootstrap\n";
    // Bootstrap with single-token rows. A sixteen-row eager forward otherwise
    // materializes a whole expert layer just to initialize graph state, before
    // the prompt's workspace loan exists, and defeats tight expert residency.
    std::vector<::tyloi::DenseTensor> raw_rows;
    // Eager FFN cells retain their transfer wire per layer until this pass
    // completes; eight-row bootstrap cells multiply that transient footprint.
    for(int row=0;row<(tq?1:16);++row) {
        auto token=ids.narrow(1,row,1);
        std::shared_ptr<::tyloi::Tensor> hidden;
        {
            tyloi::models::Input in;
            in.ids = std::make_shared<::tyloi::DenseTensor>(token);
            in.raw_hidden = &hidden;
            (void)model.forward(in, state);
        }
        raw_rows.push_back(dynamic_cast<const ::tyloi::DenseTensor&>(*hidden).clone());
    }
    auto raw=::tyloi::cuda::cat(raw_rows,1);
    raw_rows.clear();
    // Bootstrap cells are not the retained whole-model windows. Retire their
    // staging and captured scratch before preparing predictor windows.
    cache_warmup.release(tyloi::Device(tyloi::kCUDA, model.execution->config.layer_placement.primary_device()), true);
    if(tyloi::cuda::env_lookup("TYLOINFER_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=predictor\n";
    // Capture the normal greedy path and the common MTP verification widths.
    // Synthetic state is cleared afterwards; no prompt or sampling history is saved.
    if(predictor) {
        predictor->reset(*predictor_state,1);
        // Full predictor drafts consume one row. Bootstrap the same shape so
        // startup does not retain multi-row expert and pinned transfer arenas.
        for(int row=0;row<16;++row)
            (void)predictor->evaluate(*predictor_state,raw.narrow(1,row,1),ids.narrow(1,row,1));
        cache_warmup.release(tyloi::Device(tyloi::kCUDA, model.execution->config.layer_placement.primary_device()), true);
        // Production drafts run one full predictor row. Committed history
        // appends KV without an FFN/head; multi-row full-predictor graphs
        // retain scratch and driver metadata with no production consumer.
        predictor->trim_cache_to(*predictor_state,16);
        (void)predictor->step(*predictor_state,raw.narrow(1,0,1),ids.narrow(1,0,1));
        const int verification_rows=std::min({8,predictor->maximum_draft_depth()+1,maximum_draft_depth+1});
        for(int rows=4;rows>=1;--rows) {
            predictor->trim_cache_to(*predictor_state,16);
            predictor->history_step(*predictor_state,raw.narrow(1,0,rows),ids.narrow(1,0,rows));
        }
        const auto* chain_option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HISTORY_CHAIN");
        const bool history_chain=!chain_option || chain_option[0]!='0';
        const auto* chain_check=tyloi::cuda::env_lookup("TYLOINFER_VERIFY_MTP_HISTORY_CHAIN");
        const bool verify_chain=chain_check && chain_check[0]=='1' && predictor->layers[0]->qsa->supports_session(*predictor_state->layer_states[0]->qsa);
        if(history_chain)for(int rows=2;rows<=std::min(4,verification_rows-1);++rows) {
            tyloi::models::qwen4_exp::BlockSnapshot expected;
            if(verify_chain) {
                predictor->trim_cache_to(*predictor_state,16);
                for(int row=0;row<rows;++row)
                    predictor->history_step(*predictor_state,raw.narrow(1,row,1),ids.narrow(1,row,1));
                std::size_t bytes=0;
                predictor->layers[0]->qsa->capture_session(*predictor_state->layer_states[0]->qsa,expected,16+rows,bytes);
            }
            predictor->trim_cache_to(*predictor_state,16);
            predictor->history_step(*predictor_state,raw.narrow(1,0,rows),ids.narrow(1,0,rows),{},true);
            if(verify_chain) {
                tyloi::models::qwen4_exp::BlockSnapshot actual;std::size_t bytes=0;
                predictor->layers[0]->qsa->capture_session(*predictor_state->layer_states[0]->qsa,actual,16+rows,bytes);
                for(const auto& pair:{std::pair{actual.keys,expected.keys},std::pair{actual.values,expected.values},
                                     std::pair{actual.index,expected.index}}) {
                    const auto a=dynamic_cast<const DenseTensor&>(*pair.first).cpu().contiguous(),b=dynamic_cast<const DenseTensor&>(*pair.second).cpu().contiguous();
                    TYLOINFER_RUNTIME_CHECK(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
                        std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"MTP scalar history chain changed KV/index bytes");
                }
                std::cerr<<"runtime_mtp_history_chain_check rows="<<rows<<" KV/index exact PASS\n";
            }
        }
        // A depth-D draft verifies D+1 target rows. Prepare every supported
        // width before requests lend spare VRAM to resident experts; capturing
        // widths five/six afterwards also needs live recurrent-state clones.
        for(int rows=verification_rows;rows>=1;--rows) {
            if(state.cache_pos+rows+2>model.max_position_embeddings())state.reset(1);
            ::tyloi::Tensor hidden,logits;
            (void)qwen4_decode_graph_hidden(model,state,ids.narrow(1,0,rows),&hidden,
                rows>1?1:0,nullptr,{},&logits);
            if(rows>1)state.commit_speculative();
        }
        // Capturing and replaying once leaves cold per-width staging paths.
        // Exercise both single-row drafting and batched history/verification
        // again with different IDs before accepting the first request.
        for(int pass=0;pass<2;++pass)for(int rows=1;rows<=verification_rows;++rows) {
            if(state.cache_pos+rows+2>model.max_position_embeddings())break;
            const auto offset=4+pass*4;
            auto tokens=ids.narrow(1,offset,rows);
            predictor->trim_cache_to(*predictor_state,16);
            if(history_chain && rows>2) {
                for(int first=0;first<rows-1;first+=4) {
                    const auto count=std::min(4,rows-1-first);
                    predictor->history_step(*predictor_state,raw.narrow(1,offset+first,count),tokens.narrow(1,first,count),{},true);
                }
            }
            else for(int row=0;row+1<rows;++row)
                predictor->history_step(*predictor_state,raw.narrow(1,offset+row,1),tokens.narrow(1,row,1));
            (void)predictor->step(*predictor_state,raw.narrow(1,offset+rows-1,1),tokens.narrow(1,rows-1,1));
            ::tyloi::Tensor hidden,logits;
            (void)qwen4_decode_graph_hidden(model,state,tokens,&hidden,rows>1?1:0,nullptr,{},&logits);
            if(rows>1)state.commit_speculative();
        }
        predictor->reset(*predictor_state,1);
    }
    (void)qwen4_decode_graph_logits(model,state,ids.narrow(1,0,1),2);
    (void)qwen4_decode_graph_logits(model,state,ids.narrow(1,0,1),1);
    if(model.execution->weights.lookahead(
            ::tyloi::weights::Cache::Lookahead::warmup,state.router_lookahead)) {
        struct Restore {
            qwen4_exp::State& state;std::optional<bool> previous;
            ~Restore(){state.router_lookahead=previous;}
        } restore{state,state.router_lookahead};
        state.router_lookahead=true;
        if(state.cache_pos+4>=model.max_position_embeddings())state.reset(1);
        (void)qwen4_decode_graph_logits(model,state,ids.narrow(1,0,1),1);
    }
    state.reset(1);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    std::cerr<<"runtime_qwen4_graph_warmup target_captures="<<state.decode_graph_captures
        <<" maximum_draft_depth="<<(predictor?maximum_draft_depth:0)
        <<" predictor_captures="<<(predictor?predictor_state->graph_captures:0)<<" ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
    cache_warmup.release(tyloi::Device(tyloi::kCUDA, model.execution->config.layer_placement.primary_device()), false);
}

void qwen4_begin_router_lookahead_audit(qwen4_exp::Model& model,qwen4_exp::State& state,int samples) {
    if(model.blocks.size()<2 || state.graph_registry || samples<=0)
        throw std::invalid_argument("router lookahead audit requires a fresh multilayer decode session");
    auto audit=std::make_shared<RouterLookaheadInputs>(static_cast<int>(model.blocks.size()),
        static_cast<int>(model.config.hidden),samples);
    for(std::size_t i=0;i<model.blocks.size();++i) {
        if(!model.blocks[i]->router)throw std::runtime_error("router lookahead audit has no retained router");
        state.layers[i]->router_audit=audit;
    }
}
void qwen4_finish_router_lookahead_audit(qwen4_exp::Model& model,qwen4_exp::State& state,const std::string& prefix) {
    auto audit=state.layers.front()->router_audit;
    if(!audit || !audit->samples())throw std::runtime_error("router lookahead audit captured no decode inputs");
    std::ofstream file(prefix+"-logits.f32",std::ios::binary);
    if(!file)throw std::runtime_error("cannot write router lookahead audit logits");
    for(std::size_t layer=1;layer<model.blocks.size();++layer) {
        auto& b=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[layer]);
        for(int input_layer:{static_cast<int>(layer)-1,static_cast<int>(layer)}) {
            auto input=audit->rows(input_layer).to(::tyloi::kCUDA);
            auto logits=model.execution->executor().projector(model.execution->executor().mode().ffn, model_parallel_primary_device(model.execution->config)).run(*b.router,input).to(::tyloi::kFloat32).cpu().contiguous();
            if(logits.dim()!=2 || logits.size(0)!=audit->samples() || logits.size(1)!=model.config.experts)
                throw std::runtime_error("router lookahead audit router shape mismatch");
            file.write(reinterpret_cast<const char*>(logits.data_ptr()),logits.numel()*logits.element_size());
            if(!file)throw std::runtime_error("router lookahead audit logits write failed");
        }
    }
    std::cout<<"router_lookahead_audit_samples="<<audit->samples()<<" layers="<<model.blocks.size()
        <<" hidden="<<model.config.hidden<<" experts="<<model.config.experts<<" topk="<<model.config.topk
        <<" scores=predicted_then_same_layer_oracle row_dtype=FP32\n";
}

std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(
        const qwen4_exp::Model& model,const qwen4_exp::State& state,const tyloi::models::Mtp* mtp) {
    // Shared checkpoint views and recurrent state aliases count once. These
    // are existing cache tensors; collecting telemetry never copies GPU data.
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add=[&](const ::tyloi::DenseTensor& tensor) {
        if(!tensor.defined() || !tensor.is_cuda())return;
        auto& bytes=allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes=std::max(bytes,std::size_t(tensor.numel())*tensor.element_size());
    };
    const auto block_cache=[&](const qwen4_exp::Qwen4Block& block,const qwen4_exp::LayerState& layer) {
        if(block.qsa)for(const auto& tensor:block.qsa->resident_cache_tensors(*layer.qsa))add(tensor);
        if(block.gdn) { add(layer.gdn.conv.dense()); add(layer.gdn.recurrent.dense()); }
        if(block.ple && layer.ple.conv.defined())add(layer.ple.conv.dense());
    };
    for(std::size_t i=0;i<model.blocks.size();++i)block_cache(*model.blocks[i],*state.layers[i]);
    add(state.positions);
    if(state.graph_registry)add(state.graph_registry->position_storage);
    if(state.graph_registry)for(const auto& [key,cell]:state.graph_registry->cells)
        for(const auto& checkpoint:cell->checkpoints) {
            for(const auto& tensor:checkpoint.gdn_conv)add(tensor);
            for(const auto& tensor:checkpoint.gdn_state)add(tensor);
            for(const auto& tensor:checkpoint.ple_conv)add(tensor);
        }
    if(const auto* predictor=dynamic_cast<const qwen4_exp::Predictor*>(mtp); predictor && state.predictor && state.predictor->belongs_to(*predictor)) {
        const auto& request=static_cast<const qwen4_exp::PredictorState&>(*state.predictor);
        for(std::size_t i=0;i<predictor->layers.size();++i)
            block_cache(*predictor->layers[i],*request.layer_states[i]);
        for(const auto& positions:request.positions)add(positions);
        if(request.graph_registry)add(request.graph_registry->position_storage);
    }
    std::size_t bytes=0;
    std::map<int,std::size_t> devices;
    for(const auto& [address,size]:allocations) { bytes+=size; devices[address.first]+=size; }
    if(model.execution->kv)
        for(const auto& metric:model.execution->kv->metrics())
            if(metric.first=="kv_memory_gpu_bytes") {
                bytes+=std::size_t(metric.second);
                devices[model.execution->config.layer_placement.primary_device()]+=std::size_t(metric.second);
            }
    if(model.execution->offload)
        for(const auto& metric:model.execution->offload->metrics())
            if(metric.first=="qsa_kv_resident_bytes") {
                bytes+=std::size_t(metric.second);
                devices[model.execution->config.layer_placement.primary_device()]+=std::size_t(metric.second);
            }
    std::vector<std::pair<std::string,double>> result{{"kv_cache_bytes",double(bytes)},
        {"kv_cache_contexts",state.cache_pos>0?double(state.batch):0.0}};
    for(const auto& [device,size]:devices)result.emplace_back("gpu_"+std::to_string(device)+"_live_kv_bytes",double(size));
    return result;
}

Model::Model() = default;
Model::~Model() = default;

} // namespace tyloi::cuda

namespace tyloi::cuda {
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

struct Qwen4BatchRequestState::Impl {
    struct Layer {
        qwen4_exp::Gdn* gdn;
        qwen4_exp::Qsa* qsa;
        qwen4_exp::Ple* ple;
        qwen4_exp::Gdn::State gdn_state;
        std::unique_ptr<qwen4_exp::Qsa::State> qsa_state;
        qwen4_exp::Ple::State ple_state;
        explicit Layer(qwen4_exp::Qwen4Block& block)
            : gdn(block.gdn.get()), qsa(block.qsa.get()), ple(block.ple.get()) {
            if(qsa)qsa_state=std::make_unique<qwen4_exp::Qsa::State>(*qsa);
        }
    };
    std::vector<Layer> layers;
    ::tyloi::DenseTensor positions, position_storage;
    int64_t cache_position = 0, decode_position_delta = 0;
    std::size_t growth_bytes(int64_t tokens) const {
        std::size_t bytes = 0;
        for (const auto& layer : layers) {
            if (layer.qsa) bytes += layer.qsa->growth_bytes(*layer.qsa_state,tokens);
            if (layer.gdn) bytes += layer.gdn->growth_bytes(layer.gdn_state);
        }
        return bytes;
    }
};

Qwen4BatchRequestState::Qwen4BatchRequestState(qwen4_exp::Model& model)
    : state_(std::make_unique<Impl>()) {
    const auto primary = model.execution->config.layer_placement.primary_device();
    for (const auto& block : model.blocks) {
        TYLOINFER_RUNTIME_CHECK(!block->cpu_offloaded && block->cuda_device == primary,
                          "Flash-Next continuous batching requires one CUDA device");
        state_->layers.emplace_back(*block);
    }
}
Qwen4BatchRequestState::~Qwen4BatchRequestState() = default;
int64_t Qwen4BatchRequestState::position() const { return state_->cache_position; }
std::size_t Qwen4BatchRequestState::initial_growth_bytes() const {
    std::size_t fixed=0;
    for(const auto& layer:state_->layers)
        if(layer.gdn)fixed+=layer.gdn->request_cache_growth_bytes(layer.gdn_state);
    return fixed;
}
void Qwen4BatchRequestState::initialize() {
    for(auto& layer:state_->layers)if(layer.gdn)layer.gdn->initialize_request_cache(layer.gdn_state,false);
}
std::size_t Qwen4BatchRequestState::prefill_growth_bytes(int64_t tokens) const {
    return state_->growth_bytes(tokens);
}
std::size_t Qwen4BatchRequestState::resident_bytes() const {
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add = [&](const ::tyloi::DenseTensor& tensor) {
        if (!tensor.defined() || !tensor.is_cuda()) return;
        auto& bytes = allocations[{tensor.get_device(),reinterpret_cast<std::uintptr_t>(tensor.data_ptr())}];
        bytes = std::max(bytes, std::size_t(tensor.numel()) * tensor.element_size());
    };
    add(state_->positions);
    add(state_->position_storage);
    for (const auto& layer : state_->layers) {
        if (layer.qsa) for (const auto& tensor : layer.qsa->resident_cache_tensors(*layer.qsa_state)) add(tensor);
        if (layer.gdn) { add(layer.gdn_state.conv.dense()); add(layer.gdn_state.recurrent.dense()); }
        if (layer.ple && layer.ple_state.conv.defined()) add(layer.ple_state.conv.dense());
    }
    std::size_t bytes = 0;
    for (const auto& [address,size] : allocations) bytes += size;
    return bytes;
}
struct Qwen4BatchDecodeGraph::Impl {
    using DenseTensor = ::tyloi::DenseTensor;
    qwen4_exp::Model& model;
    MfqCudaStream stream;
    Event ready;
    std::unique_ptr<Graph> memory_pool;
    std::unique_ptr<DecodeWindow> window;
    std::vector<Qwen4BatchRequestState*> requests;
    std::vector<int64_t> capacities;
    std::vector<DenseTensor> positions, cache_positions;
    std::vector<DenseTensor> gdn_addresses;
    std::vector<std::vector<std::shared_ptr<RowStage>>> ple;
    DenseTensor ids, logits;
    bool greedy = false;
    int64_t captures = 0, replays = 0;
    explicit Impl(qwen4_exp::Model& owner) : model(owner), stream(mfq_get_current_cuda_stream()) {}
    void clear() {
        if (requests.empty() && !window) return;
        model.execution->weights.finish_exchange();
        window.reset();
        memory_pool.reset();
        for (auto* request : requests)
            for (auto& layer : request->state_->layers) if (layer.qsa) layer.qsa->leave_graph(*layer.qsa_state);
        requests.clear(); capacities.clear(); positions.clear(); cache_positions.clear(); ple.clear();gdn_addresses.clear();
        ids = {}; logits = {};
    }
};
Qwen4BatchDecodeGraph::Qwen4BatchDecodeGraph(qwen4_exp::Model& model)
    : graph_(std::make_unique<Impl>(model)) {}
Qwen4BatchDecodeGraph::~Qwen4BatchDecodeGraph() = default;
void Qwen4BatchDecodeGraph::clear() { graph_->clear(); }
int64_t Qwen4BatchDecodeGraph::captures() const { return graph_->captures; }
int64_t Qwen4BatchDecodeGraph::replays() const { return graph_->replays; }
std::optional<Qwen4BatchDecodeGraph::Plan> Qwen4BatchDecodeGraph::plan(
        const std::vector<Qwen4BatchRequestState*>& requests, bool greedy) {
    auto& graph = *graph_;
    auto& model = graph.model;
    if (model.execution->offload || !mfq_cuda_graph_capture_supported()) return {};
    for (auto* request : requests) for (const auto& layer : request->state_->layers)
        if (layer.ple && !layer.ple->graph_supported()) return {};
    // Whole-layer prefill snapshots the GPU/RAM expert maps. Adaptive decode
    // can exchange those experts, so retire the snapshots before replay just
    // as the single-request window does.
    model.execution->weights.retire_prefill();
    const auto batch = static_cast<int64_t>(requests.size());
    if(model.execution->kv) {
        int64_t required=0;
        for(auto* request:requests)required=std::max(required,request->position()+1);
        model.execution->kv->ensure(required);
    }
    graph.ready.record(mfq_get_current_cuda_stream());
    MfqCudaStreamGuard stream_guard(graph.stream);
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(graph.stream.stream(),graph.ready.get(),0));
    bool reuse = graph.requests == requests && graph.window && graph.greedy == greedy;
    if (reuse) for (int64_t row = 0; row < batch; ++row)
        reuse = reuse && requests[row]->position() + 1 <= graph.capacities[row];
    if (!reuse) {
        graph.clear();
        // Prefill and previous widths leave asynchronously freed pool pages
        // charged to WDDM. Retire transfers and trim before preparing a graph.
        model.execution->weights.finish_exchange();
        default_context(model.execution->config.layer_placement.primary_device())->reclaim_idle_memory();
    }
    Plan plan;
    plan.rebuild = !graph.window;
    if (plan.rebuild) {
        for (auto* request : requests) {
            const auto maximum = model.max_position_embeddings();
            const auto required = request->position() + 1;
            TYLOINFER_RUNTIME_CHECK(required <= maximum,"Flash-Next batch graph exceeds max_context");
            // Keep the initial capture interval. Large independent batches
            // subsequently grow by 128 tokens instead of doubling every
            // request's KV allocation at once under a tight device budget.
            int64_t capacity = std::min<int64_t>(batch>8 ?
                ::tyloi::cuda::options::continuous_batch_kv_initial_capacity() : 256,maximum);
            if(batch>16 && capacity<required) {
                const auto chunks=1+(required-1)/128;
                capacity=chunks>maximum/128?maximum:chunks*128;
            } else while (capacity < required)
                capacity = std::min(maximum,capacity > maximum/2 ? maximum : capacity*2);
            for(const auto& layer:request->state_->layers)
                if(layer.qsa && layer.qsa->graph_capacity(*layer.qsa_state))capacity=layer.qsa->graph_capacity(*layer.qsa_state);
            plan.capacities.push_back(capacity);
            for (const auto& layer : request->state_->layers) {
                if (layer.qsa) plan.growth += layer.qsa->graph_growth_bytes(*layer.qsa_state,capacity);
                if (layer.gdn) plan.growth += layer.gdn->request_cache_growth_bytes(layer.gdn_state);
            }
        }
    }
    return plan;
}

std::optional<::tyloi::Tensor> Qwen4BatchDecodeGraph::run(
        const std::vector<Qwen4BatchRequestState*>& requests,
        const ::tyloi::Tensor& ids_handle,const ::tyloi::Tensor& host_ids_handle,bool greedy,const Plan& plan) {
    const auto& ids = ids_handle.dense();
    const auto& host_ids = host_ids_handle.dense();
    auto& graph = *graph_;
    auto& model = graph.model;
    const auto batch = static_cast<int64_t>(requests.size());
    graph.ready.record(mfq_get_current_cuda_stream());
    MfqCudaStreamGuard stream_guard(graph.stream);
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(graph.stream.stream(),graph.ready.get(),0));
    graph.greedy = greedy;
    if (plan.rebuild) {
        graph.requests = requests;
        graph.capacities = plan.capacities;
        graph.ids = ::tyloi::cuda::empty_like(ids);
        graph.ple.resize(batch);
        for (int64_t row = 0; row < batch; ++row) {
            auto& state = *requests[row]->state_;
            graph.positions.push_back(::tyloi::cuda::empty({1},ids.options()));
            graph.cache_positions.push_back(::tyloi::cuda::empty({1},ids.options()));
            graph.ple[row].resize(state.layers.size());
            for (std::size_t i = 0; i < state.layers.size(); ++i) {
                auto& layer = state.layers[i];
                if (layer.gdn) layer.gdn->initialize_request_cache(layer.gdn_state);
                if (layer.qsa) layer.qsa->prepare_graph(*layer.qsa_state,state.positions,graph.capacities[row]);
                if (layer.ple) graph.ple[row][i] = layer.ple->graph_stage(layer.ple_state,1,ids.device());
            }
        }
        graph.gdn_addresses.resize(model.blocks.size());
        if(batch>1 && ::tyloi::cuda::options::batch_attention_projections() && ::tyloi::cuda::options::gdn_independent_batch())
            for(std::size_t i=0;i<model.blocks.size();++i) {
                std::vector<std::pair<const qwen4_exp::Gdn*,qwen4_exp::Gdn::State*>> rows;
                for(auto* request:requests)if(request->state_->layers[i].gdn)
                    rows.emplace_back(request->state_->layers[i].gdn,&request->state_->layers[i].gdn_state);
                if(rows.size()==requests.size())graph.gdn_addresses[i]=qwen4_exp::Gdn::prepare_independent_batch(rows);
            }
        // Cached MFE owns its graph pool and initializes it during preparation.
        // Dense fixtures need a pool owned by this decoder instead.
        if (!model.execution->weights.prepare_graph(::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()})) {
            graph.memory_pool = std::make_unique<Graph>();
            graph.memory_pool->prepare_memory();
            default_context(mfq_current_cuda_device())->end_graph_warmup(mfq_current_cuda_stream());
        }
        graph.window = std::make_unique<DecodeWindow>(
            mfq_current_cuda_stream(),::tyloi::cuda::options::layer_timings(),true);
        graph.window->set_profile_geometry("batch_decode",static_cast<int>(batch));
    }
    graph.ids.copy_(ids);
    std::vector<std::tuple<qwen4_exp::Ple*,qwen4_exp::Ple::State*,qwen4_exp::NgramEmbedding::Context>> ple_contexts;
    for (int64_t row = 0; row < batch; ++row) {
        auto& state = *requests[row]->state_;
        mfq_qwen4_exp::fill_decode_positions(graph.cache_positions[row],graph.positions[row],
                                            state.cache_position,state.decode_position_delta);
        for (std::size_t i = 0; i < state.layers.size(); ++i) if (state.layers[i].ple) {
            if(!graph.window->valid())ple_contexts.emplace_back(state.layers[i].ple,&state.layers[i].ple_state,state.layers[i].ple->session_context(state.layers[i].ple_state));
            state.layers[i].ple->upload_graph(state.layers[i].ple_state,host_ids.narrow(0,row,1),*graph.ple[row][i]);
        }
    }
    if (!graph.window->valid()) {
        struct Saved { ::tyloi::Tensor* tensor; ::tyloi::DenseTensor snapshot; };
        std::vector<Saved> recurrent;
        const auto save=[&](::tyloi::Tensor* tensor) {
            // Capture restores these values after warmup. They are never read
            // by replay, so large batches can keep this temporary copy in RAM.
            if(tensor->defined())recurrent.push_back({tensor,batch>8 ? tensor->dense().to(::tyloi::kCPU) : tensor->dense().clone()});
        };
        for (auto* request : requests) for (auto& layer : request->state_->layers) {
            if (layer.gdn) for (auto* tensor : layer.gdn->graph_state(layer.gdn_state))
                save(tensor);
            if (layer.ple) for (auto* tensor : layer.ple->graph_state(layer.ple_state))
                save(tensor);
        }
        DecodeGraphBranchScope branch_scope(model.execution->executor().mode().ffn.serial);
        try { graph.window->capture([&] {
            auto& execution = *model.execution;
            graph.window->mark(-1,"window_begin");
            auto hidden = model.repeat_hidden(model.embed_tokens(graph.ids),model.metadata.hc_mult);
            std::optional<GatedResidual::Result> pending_residual;
            const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
            const bool chain_post=(!chain_setting || *chain_setting) && batch<=8;
            for (std::size_t i = 0; i < model.blocks.size(); ++i) {
                graph.window->mark(static_cast<int>(i),"block_begin");
                auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
                graph.window->set_profile_layer_kind(static_cast<int>(i),block.gdn?"gdn":"qsa");
                const auto marker=[&](const char* role)->GatedResidualMarker {
                    if(!::tyloi::cuda::options::mhc_timings())return {};
                    return [&,role](const char* phase) {
                        graph.window->mark(static_cast<int>(i),(std::string(role)+phase).c_str());
                    };
                };
                if (block.ple) {
                    graph.window->mark(static_cast<int>(i),"ple_begin");
                    std::vector<::tyloi::DenseTensor> positional;
                    for (int64_t row = 0; row < batch; ++row) {
                        auto x = hidden.narrow(0,row,1);
                        graph.ple[row][i]->decode();
                        auto rows = dynamic_cast<const DenseTensor&>(graph.ple[row][i]->output()).reshape({1,1,model.hidden_size()});
                        positional.push_back(x + requests[row]->state_->layers[i].ple->forward_chunk(
                            requests[row]->state_->layers[i].ple_state,x,graph.ids.narrow(0,row,1),true,{},rows));
                    }
                    hidden = join_request_outputs(positional,0);
                    graph.window->mark(static_cast<int>(i),"ple_end");
                }
                auto mix = pending_residual
                    ? block.attention_gr.pre_after(hidden,pending_residual->residual,pending_residual->gate,marker("attention_mhc_"))
                    : block.attention_gr.pre(hidden,marker("attention_mhc_"));
                graph.window->mark(static_cast<int>(i),"attention_core_begin");
                const bool project_batch=batch>1 && ::tyloi::cuda::options::batch_attention_projections();
                const auto projections=project_batch ? (block.gdn ? block.gdn->project_inputs(mix.mixed)
                    : block.qsa->project_inputs(mix.mixed)) : std::vector<::tyloi::DenseTensor>{};
                std::vector<::tyloi::DenseTensor> attended;
                DenseTensor attention;
                if(project_batch && block.gdn && graph.gdn_addresses[i].defined())
                    attention=requests[0]->state_->layers[i].gdn->forward_independent_batch(requests[0]->state_->layers[i].gdn_state,mix.mixed,projections,graph.gdn_addresses[i]);
                if(!attention.defined())for (int64_t row = 0; row < batch; ++row) {
                    auto& layer = requests[row]->state_->layers[i];
                    auto attention_input = mix.mixed.narrow(0,row,1);
                    std::vector<::tyloi::DenseTensor> projected;
                    for(const auto& value:projections)projected.push_back(value.narrow(0,row,1));
                    attended.push_back(layer.gdn ? layer.gdn->forward_chunk(layer.gdn_state,attention_input,true,std::move(projected),!project_batch)
                        : layer.qsa->forward_graph(*layer.qsa_state,attention_input,graph.positions[row],graph.cache_positions[row],
                            std::move(projected),!project_batch));
                }
                if(!attention.defined())attention=join_request_outputs(attended,0);
                if(project_batch)attention=block.gdn ? block.gdn->project_output(attention)
                    : block.qsa->project_output(attention);
                graph.window->mark(static_cast<int>(i),"attention_core_end");
                auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate,marker("ffn_mhc_"));
                graph.window->mark(static_cast<int>(i),"ffn_pre_end");
                auto ffn_output=block.batch_ffn(execution,ffn_mix.mixed);
                bool defer=false;
                if(chain_post && i+1<model.blocks.size()) {
                    auto& next=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i+1]);
                    defer=!next.ple && next.attention_gr.can_fuse(ffn_mix.residual);
                }
                if(defer){hidden=std::move(ffn_output);pending_residual=std::move(ffn_mix);}
                else {hidden=block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate);pending_residual.reset();}
                graph.window->mark(static_cast<int>(i),"block_end");
            }
            auto finalized=qwen4_finalize_hidden(model, std::move(hidden),batch,1);
            graph.logits = greedy ? qwen4_next_token(model,qwen4_last_hidden(std::move(finalized)))
                .reshape({batch,1}) : qwen4_logits_from_hidden(model, std::move(finalized)).reshape({batch,-1});
            graph.window->mark(-1,"window_end");
        },[&] { for (const auto& saved : recurrent) saved.tensor->dense().copy_(saved.snapshot); },
        [&] { graph.logits = {}; }); }
        catch(const std::exception& error) {
            const std::string message=error.what();
            if(!dynamic_cast<const MfqResourceExhausted*>(&error) &&
               !message.starts_with("CUDA process VRAM limit exceeded:") &&
               !message.starts_with("CUDA VRAM limit exceeded:") &&
               !message.starts_with("CUDA pinned allocation failed:"))throw;
            // Capture has not accepted a token. Retire its consumers before
            // restoring state, then let admission reject only the newest row.
            graph.clear();
            (void)cudaGetLastError();
            for(const auto& saved:recurrent)saved.tensor->dense().copy_(saved.snapshot);
            for(auto& [layer,layer_state,context]:ple_contexts)layer->restore_graph_context(*layer_state,std::move(context));
            throw;
        }
        ++graph.captures;
    }
    graph.window->run();
    ++graph.replays;
    for (std::size_t row=0;row<requests.size();++row) {
        auto& state = *requests[row]->state_;
        for (auto& layer : state.layers) if (layer.qsa) layer.qsa->advance_graph(*layer.qsa_state,1);
        state.positions=append_decode_positions(state.positions,graph.cache_positions[row],
            state.position_storage,model.max_position_embeddings());
        ++state.cache_position;
    }
    return ::tyloi::Tensor(graph.logits);
}

std::vector<::tyloi::Tensor> qwen4_batch_prefill(qwen4_exp::Model& model,
        const std::vector<Qwen4BatchRequestState*>& requests,
        const std::vector<::tyloi::Tensor>& chunk_handles) {
    std::vector<::tyloi::DenseTensor> chunks;
    chunks.reserve(chunk_handles.size());
    for (const auto& chunk : chunk_handles) chunks.push_back(chunk.dense());
    TYLOINFER_RUNTIME_CHECK(!requests.empty() && requests.size()==chunks.size(),"invalid packed prefill rows");
    auto& execution = *model.execution;
    const bool packed=requests.size()>1;
    std::vector<int64_t> offsets{0};
    std::vector<::tyloi::DenseTensor> current,full;
    for (std::size_t row=0;row<requests.size();++row) {
        const auto& state=*requests[row]->state_;
        const auto& ids=chunks[row];
        TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 && ids.size(1)>0 &&
            ids.size(1)<=model.max_position_embeddings()-state.cache_position,"invalid packed prefill chunk");
        auto positions=::tyloi::cuda::arange(ids.size(1),ids.options())+state.cache_position;
        current.push_back(positions);
        full.push_back(state.positions.defined()?::tyloi::cuda::cat({state.positions,positions},-1):positions);
        offsets.push_back(offsets.back()+ids.size(1));
    }
    auto ids=::tyloi::cuda::cat(chunks,1);
    auto host_ids=ids.to(::tyloi::kCPU);
    const auto tokens=offsets.back();
    if(!packed && execution.config.cache.pipeline) {
        for(std::size_t row=0;row<requests.size();++row) {
            const auto row_ids=host_ids.narrow(1,offsets[row],chunks[row].size(1));
            for(auto& layer:requests[row]->state_->layers)
                if(layer.ple)layer.ple->prefetch_tokens(layer.ple_state,row_ids);
        }
    }
    auto hidden=model.repeat_hidden(model.embed_tokens(ids),model.metadata.hc_mult);
    for (std::size_t i=0;i<model.blocks.size();++i) {
        NintCache decode_reuse(packed);
        auto& block=static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if (block.ple) {
            std::vector<::tyloi::DenseTensor> positional;
            for (std::size_t row=0;row<requests.size();++row) {
                auto x=hidden.narrow(1,offsets[row],chunks[row].size(1));
                auto& layer=requests[row]->state_->layers[i];
                auto ple=packed ? layer.ple->forward(layer.ple_state,x,chunks[row],true,0,
                    host_ids.narrow(1,offsets[row],chunks[row].size(1))) :
                    execution.executor().profiler().measure("prefill.ple",[&] {
                        return layer.ple->forward(layer.ple_state,x,chunks[row],true,0,
                            host_ids.narrow(1,offsets[row],chunks[row].size(1)));
                    });
                positional.push_back(x+ple);
            }
            hidden=join_request_outputs(positional,1);
        }
        auto mix=packed ? block.attention_gr.pre(hidden) :
            execution.executor().profiler().measure("prefill.gr_attention",[&] { return block.attention_gr.pre(hidden); });
        ::tyloi::DenseTensor attention;
        if(packed) {
            const auto projections=block.gdn?block.gdn->project_inputs(mix.mixed)
                                             :block.qsa->project_inputs(mix.mixed);
            std::vector<::tyloi::DenseTensor> attended;
            for (std::size_t row=0;row<requests.size();++row) {
                auto& layer=requests[row]->state_->layers[i];
                auto attention_input=mix.mixed.narrow(1,offsets[row],chunks[row].size(1));
                std::vector<::tyloi::DenseTensor> projected;
                for (const auto& value:projections)
                    projected.push_back(value.narrow(1,offsets[row],chunks[row].size(1)));
                attended.push_back(layer.gdn?layer.gdn->forward_chunk(layer.gdn_state,attention_input,true,std::move(projected),false)
                    :layer.qsa->forward(*layer.qsa_state,attention_input,current[row],full[row],true,nullptr,std::move(projected),false));
            }
            attention=join_request_outputs(attended,1);
            attention=block.gdn?block.gdn->project_output(attention):block.qsa->project_output(attention);
        } else {
            auto& layer=requests.front()->state_->layers[i];
            if(layer.gdn)attention=execution.executor().profiler().measure("prefill.gdn",[&] {
                return layer.gdn->forward(layer.gdn_state,mix.mixed,true,0);
            });
            else attention=execution.executor().profiler().measure("prefill.qsa",[&] {
                return layer.qsa->forward(*layer.qsa_state,mix.mixed,current.front(),full.front(),true);
            });
        }
        auto ffn_mix=packed ? block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate) :
            execution.executor().profiler().measure("prefill.gr_ffn_pre",[&] {
                return block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate);
            });
        auto ffn_output=packed ? block.ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
            execution,
#endif
            ffn_mix.mixed) :
            execution.executor().profiler().measure("prefill.ffn",[&] { return block.ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                execution,
#endif
                ffn_mix.mixed); });
        hidden=packed ? block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate) :
            execution.executor().profiler().measure("prefill.gr_ffn_post",[&] {
                return block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate);
            });
    }
    hidden=qwen4_finalize_hidden(model, std::move(hidden),1,tokens);
    std::vector<::tyloi::Tensor> result;
    for (std::size_t row=0;row<requests.size();++row) {
        result.push_back(hidden.narrow(1,offsets[row+1]-1,1));
        requests[row]->state_->positions=std::move(full[row]);
        requests[row]->state_->cache_position+=chunks[row].size(1);
    }
    return result;
}

::tyloi::Tensor qwen4_batch_decode(qwen4_exp::Model& model,
        const std::vector<Qwen4BatchRequestState*>& requests, ::tyloi::Tensor ids_handle,
        const ::tyloi::Tensor& host_ids_handle) {
    const auto& ids = ids_handle.dense();
    const auto& host_ids = host_ids_handle.dense();
    const auto batch = static_cast<int64_t>(requests.size());
    TYLOINFER_RUNTIME_CHECK(batch > 0 && ids.dim() == 2 && ids.size(0) == batch && ids.size(1) == 1,
                      "invalid Flash-Next batch input");
    std::vector<::tyloi::DenseTensor> current, full;
    for (const auto* request : requests) {
        const auto& state = *request->state_;
        TYLOINFER_RUNTIME_CHECK(state.cache_position < model.max_position_embeddings(), "batch request exceeds max_context");
        auto position = ::tyloi::cuda::full({1},state.cache_position,ids.options());
        current.push_back(position);
        full.push_back(state.positions.defined() ? ::tyloi::cuda::cat({state.positions,position},-1) : position);
    }
    auto& execution = *model.execution;
    auto hidden = model.repeat_hidden(model.embed_tokens(ids),model.metadata.hc_mult);
    for (std::size_t i = 0; i < model.blocks.size(); ++i) {
        auto& block = static_cast<qwen4_exp::Qwen4Block&>(*model.blocks[i]);
        if (block.ple) {
            std::vector<::tyloi::DenseTensor> positional;
            for (int64_t row = 0; row < batch; ++row) {
                auto& layer = requests[row]->state_->layers[i];
                auto x = hidden.narrow(0,row,1);
                positional.push_back(x + layer.ple->forward(layer.ple_state,x,ids.narrow(0,row,1),true,0,
                                                           host_ids.narrow(0,row,1)));
            }
            hidden = join_request_outputs(positional,0);
        }
        auto mix = block.attention_gr.pre(hidden);
        const bool project_batch=batch>1 && block.gdn && ::tyloi::cuda::options::batch_attention_projections();
        const auto projections=project_batch ? block.gdn->project_inputs(mix.mixed) : std::vector<::tyloi::DenseTensor>{};
        std::vector<::tyloi::DenseTensor> attended;
        for (int64_t row = 0; row < batch; ++row) {
            auto& layer = requests[row]->state_->layers[i];
            auto attention_input = mix.mixed.narrow(0,row,1);
            std::vector<::tyloi::DenseTensor> projected;
            for(const auto& value:projections)projected.push_back(value.narrow(0,row,1));
            attended.push_back(layer.gdn ? layer.gdn->forward_chunk(layer.gdn_state,attention_input,true,std::move(projected),!project_batch)
                : layer.qsa->forward(*layer.qsa_state,attention_input,current[row],full[row],true));
        }
        auto attention=join_request_outputs(attended,0);
        if(project_batch)attention=block.gdn->project_output(attention);
        auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate);
        // Tiles share the expert transfer cache across all eligible rows.
        hidden = block.ffn_gr.post(block.batch_ffn(execution,ffn_mix.mixed),ffn_mix.residual,ffn_mix.gate);
    }
    auto logits = qwen4_logits_from_hidden(model, qwen4_finalize_hidden(model, std::move(hidden),batch,1)).reshape({batch,-1});
    for (int64_t row = 0; row < batch; ++row) {
        requests[row]->state_->positions = std::move(full[row]);
        ++requests[row]->state_->cache_position;
    }
    return logits;
}

qwen4_exp::State::State(Model& model) : model_(model) {
    for (const auto& layer : model_.blocks)
        layers.push_back(std::make_unique<LayerState>(*layer));
}

qwen4_exp::State::~State() = default;

void qwen4_exp::State::reset(int64_t new_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    speculative_prefix_retention = false;
    if(model_.execution)model_.execution->weights.finish_exchange();
    const bool retain=reuse_decode_graphs && new_batch==1 && graph_registry && graph_registry->reusable();
    for(auto& layer:layers)layer->retain_graph_storage=retain;
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells)cell->used_kinds=0;
    positions = {};
    batch = new_batch;
    for(std::size_t i=0;i<layers.size();++i)model_.blocks[i]->reset(*layers[i],new_batch);
}

void qwen4_exp::State::begin_forward() {
    for(auto& layer:layers)layer->retained_prefix=false;
    if(graph_registry) {
        if(reuse_decode_graphs && graph_registry->reusable()) {
            for(auto [attention,state]:graph_registry->qsa)attention->leave_graph(*state);
        } else {graph_registry->restore_legacy();graph_registry.reset();}
    }
}

void qwen4_exp::State::finish_forward(const ::tyloi::DenseTensor &full_positions,
                                      int64_t new_batch, int64_t tokens) {
    positions = full_positions;
    batch = new_batch;
    speculative_prefix_retention=tokens<=8 && !layers.empty() &&
        std::all_of(layers.begin(),layers.end(),[](const auto& layer) {
            return layer->retained_prefix;
        });
}

// Keep each compressed expert image resident while advancing all prompt chunks
// through that layer. Attention and PLE still advance in causal token order.
bool qwen4_layer_major_prefill_eligible(const qwen4_exp::Model& model, const qwen4_exp::State& state, int64_t tokens) {
    const auto enabled=::tyloi::cuda::options::moe_prefill_layer_major();
    const auto whole=::tyloi::cuda::options::moe_prefill_layer();
    const bool tq=model.execution && model.execution->weights.decode(model.execution->config.cache.pipeline)==
        ::tyloi::weights::Cache::Decode::routed;
    // TQ already groups ordinary bank rows by expert. Its layer-major path
    // does not require registration in the legacy MFE transfer pipeline.
    // Keep explicit user overrides and the existing legacy selection policy.
    if((enabled && *enabled==0) || (whole && *whole==0) || (!tq && !whole) || tokens<=128 ||
       tokens>16384 || state.speculative_start>=0 || !model.execution ||
       (!tq && !model.execution->config.cache.pipeline) ||
       model.execution->weights.transfer()!=::tyloi::weights::Cache::Transfer::coordinated ||
       model.execution->config.dense_cpu_layer_count || model.execution->config.tensor_parallel.enabled() ||
       model.execution->config.expert_parallel.enabled())return false;
    const auto primary=model.execution->config.layer_placement.primary_device();
    for(const auto& block:model.blocks)
        if(block->cpu_offloaded || block->cuda_device!=primary)return false;
    return true;
}

std::int64_t prefill_span(const qwen4_exp::Model& model, const qwen4_exp::State& state, std::size_t tokens,
                          std::int64_t chunk, std::size_t reused,
                          std::size_t stable, bool transformed, bool) {
    // Bound residual/FFN storage as well as projection tiles. Long
    // prompts advance each bounded chunk through all layers.
    auto logical_chunk=std::min<int64_t>(chunk,8192);
    if(chunk>8 && reused==0 && (stable==0 || stable==tokens) && !transformed &&
        qwen4_layer_major_prefill_eligible(model,state,tokens))
        logical_chunk=tokens;
    return logical_chunk;
}

::tyloi::Tensor qwen4_layer_major_prefill_hidden(
        qwen4_exp::Model& model,qwen4_exp::State& state,::tyloi::Tensor ids_handle,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,::tyloi::Tensor* raw_hidden,
        bool raw_hidden_host) {
    return qwen4_layer_major_prefill_hidden(model,state,std::move(ids_handle),compute_chunk,last_only,
        cancelled,raw_hidden,raw_hidden_host,{});
}

::tyloi::Tensor qwen4_layer_major_prefill_hidden(
        qwen4_exp::Model& model,qwen4_exp::State& state,::tyloi::Tensor ids_handle,int64_t compute_chunk,
        bool last_only,const std::function<bool()>& cancelled,::tyloi::Tensor* raw_hidden,
        bool raw_hidden_host,const std::function<void()>& computation_finished) {
    using DenseTensor=::tyloi::DenseTensor;
    auto ids = ids_handle.dense();
    TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
        qwen4_layer_major_prefill_eligible(model,state,ids.size(1)),
        "layer-major prefill requires a bounded single-device Qwen4 chunk and whole-layer cache");
    TYLOINFER_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
    auto scope=model.execution_scope();
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    const auto cache_start=state.cache_pos;
    auto& profiler=model.execution->executor().profiler();
    const auto* profile_once=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_PREFILL_ONCE");
    static std::atomic<uint64_t> profile_requests{0};
    const bool profile_enabled=profile_once && profile_once[0]=='1';
    const auto* profile_skip=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_PREFILL_SKIP");
    const auto skipped=profile_skip?std::max<long long>(0,std::atoll(profile_skip)):0;
    const auto profile_request=profile_enabled?profile_requests.fetch_add(1):0;
    const bool collect=profile_enabled && profile_request==uint64_t(skipped);
    if(profile_enabled)std::cerr<<"prefill_profile_request index="<<profile_request
        <<" tokens="<<tokens<<" collect="<<int(collect)<<'\n';
    const bool previous_profile=profiler.enabled;
    if(collect){profiler.reset();profiler.enabled=true;}
    struct ProfileRestore {
        CudaProfiler& profiler;bool enabled,previous,collect;
        ~ProfileRestore() {
            if(collect && std::uncaught_exceptions()==0) {
                try {profiler.report("prefill_regression_once");profiler.reset();}
                catch(const std::exception& error){std::cerr<<"prefill_profile_error "<<error.what()<<'\n';}
            }
            profiler.enabled=previous;
        }
    } restore{profiler,profiler.enabled,previous_profile,collect};
    compute_chunk=std::min(compute_chunk,tokens);
    tyloi::weights::Loan workspace_loan(model.execution->weights);
    bool borrowed_workspace=false;
    {
        // Persistent decode graphs and prefill now coexist. Bound the compute
        // tile by actual remaining workspace instead of discarding those graphs
        // or allocating an 8K projection tile on top of their live buffers.
        const auto context=tyloi::cuda::default_context(mfq_current_cuda_device());
        const auto reserve=std::size_t(1)<<30;
        const auto row_bytes=std::size_t(model.config.hidden)*model.config.streams*sizeof(float)*8;
        // Whole-prompt residual and FFN injection remain live across tiles.
        // Their previous hidden input and the assembled FFN output can overlap
        // them at a layer boundary. Include that third sequence-sized buffer.
        const auto sequence_reserve=std::size_t(tokens)*model.config.hidden*model.config.streams*sizeof(float)*3;
        const auto budget=model.execution->weights.prefill_budget(true);
        const auto planned_fixed=tyloi::engine::prefill_budget_add(reserve+sequence_reserve,budget.transfer_bytes);
        const auto planned_available=tyloi::engine::prefill_budget_add(
            context->available_memory()+context->workspace_available(),budget.lendable_bytes);
        compute_chunk=tyloi::engine::select_prefill_chunk(tokens,compute_chunk,planned_available,planned_fixed,row_bytes);
        TYLOINFER_RUNTIME_CHECK(compute_chunk>0,"insufficient CUDA prefill workspace before expert loan");
        borrowed_workspace=workspace_loan.begin(planned_fixed+std::size_t(compute_chunk)*row_bytes)>0;
        const auto available=context->available_memory()+context->workspace_available();
        const auto requested=compute_chunk;
        const auto fixed=reserve+sequence_reserve+workspace_loan.transfer_bytes();
        compute_chunk=tyloi::engine::select_prefill_chunk(tokens,compute_chunk,available,fixed,row_bytes);
        TYLOINFER_RUNTIME_CHECK(compute_chunk>0,"insufficient CUDA prefill workspace after expert loan");
        if(compute_chunk!=requested)std::cerr<<"runtime_prefill_chunk_budget requested="<<requested
            <<" compute="<<compute_chunk<<" available_bytes="<<available<<'\n';
    }
    tyloi::cuda::ResidentAllocationScope persistent_preamble;
    qwen4_validate_forward(model,tokens,cache_start,false);
    if(cache_start==0)state.reset(1);
    state.begin_forward();
    auto current=::tyloi::cuda::arange(cache_start,cache_start+tokens,ids.options());
    if(state.decode_position_delta)current=current+state.decode_position_delta;
    auto prepared=std::move(current);
    if ((prepared.dim() == 2 || prepared.dim() == 3) && prepared.size(0) == 4)
        prepared = prepared.narrow(0,1,3);
    auto prepared_full=state.positions.defined()?
        ::tyloi::cuda::cat({state.positions,prepared},-1):prepared;
    tyloi::models::require_model((prepared.dim() == 1 || (prepared.dim() == 2 && prepared.size(0) == 3) ||
        (prepared.dim() == 3 && prepared.size(0) == 3 && prepared.size(1) == 1)) && prepared.size(-1) == tokens,
        "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    auto cpu_ids=ids.to(::tyloi::kCPU,::tyloi::kInt64).contiguous();
    persistent_preamble.finish();
    DenseTensor hidden=::tyloi::cuda::empty({1,tokens,model.metadata.hidden_size*model.metadata.hc_mult},
                           ids.options().dtype(::tyloi::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
        if(cancelled && cancelled())return {};
        const auto count=std::min(compute_chunk,tokens-begin);
        auto input=model.repeat_hidden(model.embed_tokens(ids.narrow(1,begin,count)),model.metadata.hc_mult);
        TYLOINFER_RUNTIME_CHECK(input.scalar_type()==hidden.scalar_type() &&
            input.size(-1)==hidden.size(-1),"layer-major embedding geometry changed");
        hidden.narrow(1,begin,count).copy_(input);
    }
    const auto trace_option=::tyloi::cuda::options::prefill_layer_major_trace();
    const bool trace=trace_option && *trace_option!=0;
    const auto batch_option=::tyloi::cuda::options::moe_prefill_expert_batch();
    const bool batch_experts=!batch_option || *batch_option!=0;
    const auto bulk_gr_option=::tyloi::cuda::options::gr_prefill_layer_batch();
    // Long prompts retain the established memory footprint unless explicitly
    // requested; the combined residual tensors have been verified to 16k.
    const bool bulk_gr=batch_experts && (bulk_gr_option?*bulk_gr_option:tokens<=16384);
    const auto bulk_gdn_option=::tyloi::cuda::options::gdn_prefill_layer_batch();
#ifdef TYLOINFER_NATIVE_CUDA
    // Allow 8k chat framing while keeping the extra whole-sequence QKV and
    // gate tensors out of the 16k service path, which also retains MTP weights.
    const bool bulk_gdn=bulk_gr && (bulk_gdn_option?*bulk_gdn_option:tokens<=9216);
#else
    const bool bulk_gdn=bulk_gr && bulk_gdn_option && *bulk_gdn_option;
#endif
    const qwen4_exp::QwenPrefillReferenceRowsScope reference_rows(batch_experts?std::min<int64_t>(1024,compute_chunk):0);
    const auto batch_tokens_option=::tyloi::cuda::options::moe_prefill_expert_batch_tokens();
    const int64_t batch_tokens=batch_tokens_option?*batch_tokens_option:8192;
    const auto append_tail_option=::tyloi::cuda::options::moe_prefill_append_tail();
    const bool append_tail=append_tail_option && *append_tail_option;
    const auto expert_batch_count=[&](int64_t remaining) {
        // Chat framing can leave a few tokens beyond the batch boundary.
        // Absorb at most one reference chunk instead of running a second
        // almost-empty routed GEMM for every layer.
        return append_tail && remaining<=batch_tokens+1024?remaining:std::min(batch_tokens,remaining);
    };
    TYLOINFER_RUNTIME_CHECK(!batch_experts || batch_tokens>=1024,"expert prefill batch must cover at least 1024 tokens");
    const auto audit_option=::tyloi::cuda::options::prefill_expert_batch_audit();
    const bool audit_batch=audit_option && *audit_option!=0;
    const auto started=std::chrono::steady_clock::now();
    int layer=0;
    for(auto& block:model.blocks) {
        auto& layer_state=*state.layers[layer];
        const auto decode_reuse_option=::tyloi::cuda::options::nint_prefill_decode_reuse();
        NintCache decode_reuse(!decode_reuse_option || *decode_reuse_option);
        DenseTensor next_hidden=hidden;
        DenseTensor ffn_inputs,ffn_gate;
        GatedResidual::Result attention_result;
        DenseTensor attention_output;
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            DenseTensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    if(cancelled && cancelled())return {};
                    const auto count=std::min(compute_chunk,tokens-begin);
                    BlockContext context;
                    context.token_ids=ids.narrow(1,begin,count);
                    context.host_token_ids=cpu_ids.narrow(1,begin,count);
                    block->prefetch_token_ids(layer_state,context.host_token_ids);
                    auto input=qwen_block.prefill_ple(layer_state,*model.execution,hidden.narrow(1,begin,count),context);
                    if(begin==0)source=::tyloi::cuda::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            attention_result=profiler.measure("prefill.gr_attention",[&] {
                return qwen_block.attention_gr.pre(source);
            });
            if(bulk_gdn && qwen_block.gdn) {
                if(cancelled && cancelled())return {};
                const qwen4_exp::QwenGdnPrefillProjectionRowsScope projection_rows(compute_chunk);
                attention_output=profiler.measure("prefill.gdn",[&] {
                    return qwen_block.gdn->forward(layer_state.gdn,attention_result.mixed,true,0);
                });
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            if(cancelled && cancelled())return {};
            if(bulk_gdn && static_cast<qwen4_exp::Qwen4Block&>(*block).gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            profiler.enabled=restore.enabled;
            BlockContext context;
            context.token_ids=ids.narrow(1,begin,count);
            context.host_token_ids=cpu_ids.narrow(1,begin,count);
            context.positions=prepared.narrow(-1,begin,count);
            context.full_positions=prepared_full.narrow(-1,0,cache_start+begin+count);
            context.cache_position=cache_start+begin;
            block->prefetch_token_ids(layer_state,context.host_token_ids);
            auto input=hidden.narrow(1,begin,count);
            DenseTensor output;
            if(bulk_gr) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto attention_chunk=qwen_block.prefill_attention_core(layer_state,*model.execution,attention_result.mixed.narrow(1,begin,count),context);
                if(!attention_output.defined()) {
                    auto shape=attention_chunk.sizes().vec();shape[1]=tokens;
                    attention_output=::tyloi::cuda::empty(shape,attention_chunk.options());
                }
                attention_output.narrow(1,begin,count).copy_(attention_chunk);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
                auto mix=qwen_block.prefill_attention(layer_state,*model.execution,input,context);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=::tyloi::cuda::empty({1,tokens,model.hidden_size()},mix.mixed.options().dtype(::tyloi::kFloat16));
                    ffn_gate=::tyloi::cuda::empty({1,tokens,mix.gate.size(-1)},mix.gate.options());
                }
                // The MoE casts the router/FFN source to half at this boundary.
                ffn_inputs.narrow(1,begin,count).copy_(mix.mixed.to(::tyloi::kFloat16));
                ffn_gate.narrow(1,begin,count).copy_(mix.gate);
                output=std::move(mix.residual);
            } else output=block->forward_context(layer_state,*model.execution,input,context);
            // Retain the original rounding at every layer boundary. The next
            // chunk is independent of this input once its cache writes retire.
            TYLOINFER_RUNTIME_CHECK(output.sizes()==input.sizes(),"layer-major decoder changed hidden shape");
            if(begin==0 && output.scalar_type()!=hidden.scalar_type()) {
                next_hidden=::tyloi::cuda::empty(hidden.sizes(),output.options());
                if(trace)std::cout<<"prefill_layer_major_hidden layer="<<layer
                    <<" input_bytes="<<hidden.nbytes()<<" output_bytes="<<next_hidden.nbytes()<<std::endl;
            }
            TYLOINFER_RUNTIME_CHECK(output.scalar_type()==next_hidden.scalar_type(),
                "layer-major decoder output dtype changed within a layer");
            next_hidden.narrow(1,begin,count).copy_(output);
        }
        // Attention's decoded matrices are no longer needed by FFN. Keep
        // reuse within each phase, without retaining both sets at peak memory.
        if(tokens>9216)decode_reuse.clear();
        if(bulk_gr) {
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            auto mix=profiler.measure("prefill.gr_ffn_pre",[&] {
                return qwen_block.ffn_gr.pre_after(attention_output,attention_result.residual,attention_result.gate);
            });
            ffn_inputs=mix.mixed.to(::tyloi::kFloat16);ffn_gate=std::move(mix.gate);
            next_hidden=std::move(mix.residual);
            attention_result={};attention_output={};
        }
        if(tokens>9216)decode_reuse.clear();
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
            if(cancelled && cancelled())return {};
            auto& qwen_block=static_cast<qwen4_exp::Qwen4Block&>(*block);
            profiler.enabled=restore.enabled;
            auto ffn_output=profiler.measure("prefill.ffn",[&] {
                if(expert_batch_count(tokens)==tokens)return qwen_block.ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                    *model.execution,
#endif
                    ffn_inputs);
                DenseTensor result;
                for(int64_t begin=0;begin<tokens;) {
                    if(cancelled && cancelled())return DenseTensor{};
                    auto count=expert_batch_count(tokens-begin);
                    const auto tail=tokens-begin-count;
                    // A tiny final batch would enter the decode FFN path and
                    // release the shared prefill images halfway through every
                    // layer. Keep both batches in prefill, including chat inputs
                    // just above the expert-batch boundary.
                    if(tail>0 && tail<=8)count-=16-tail;
                    auto part=qwen_block.ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                        *model.execution,
#endif
                        ffn_inputs.narrow(1,begin,count));
                    if(!result.defined()) {
                        auto shape=part.sizes().vec();shape[1]=tokens;
                        result=::tyloi::cuda::empty(shape,part.options());
                    }
                    result.narrow(1,begin,count).copy_(part);
                    begin+=count;
                }
                return result;
            });
            if(!ffn_output.defined())return {};
            if(audit_batch) {
                auto expected=::tyloi::cuda::empty(ffn_output.sizes(),ffn_output.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
                        qwen_block.ffn.forward(
#ifndef TYLOINFER_NATIVE_CUDA
                            *model.execution,
#endif
                            ffn_inputs.narrow(1,begin,count)));
                }
                auto a=ffn_output.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
                auto b=expected.to(::tyloi::kCPU,::tyloi::kFloat32).contiguous();
                double error=0,scale=0,maximum=0;
                for(int64_t i=0;i<a.numel();++i) {
                    const double delta=double(a.data_ptr<float>()[i])-b.data_ptr<float>()[i];
                    error+=delta*delta;scale+=double(b.data_ptr<float>()[i])*b.data_ptr<float>()[i];
                    maximum=std::max(maximum,std::abs(delta));
                }
                std::cout<<"prefill_expert_batch_layer_oracle layer="<<layer<<" tokens="<<tokens
                    <<" relative_rms="<<std::sqrt(error/std::max(1e-30,scale))
                    <<" maximum_absolute="<<maximum<<std::endl;
                // Diagnostic only: continue with the reference so later
                // layers are tested with their original input distribution.
                ffn_output=std::move(expected);
            }
            DenseTensor post_hidden=next_hidden;
            for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                const auto count=std::min(compute_chunk,tokens-begin);
                auto output=qwen_block.ffn_gr.post(ffn_output.narrow(1,begin,count),
                    next_hidden.narrow(1,begin,count),ffn_gate.narrow(1,begin,count));
                if(begin==0 && output.scalar_type()!=next_hidden.scalar_type())
                    post_hidden=::tyloi::cuda::empty(next_hidden.sizes(),output.options());
                TYLOINFER_RUNTIME_CHECK(output.scalar_type()==post_hidden.scalar_type(),
                    "expert-batched residual dtype changed");
                post_hidden.narrow(1,begin,count).copy_(output);
            }
            next_hidden=std::move(post_hidden);
        }
        hidden=std::move(next_hidden);
        ++layer;
        if(trace && (layer%4==0 || layer==int(model.blocks.size()))) {
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
#ifdef TYLOINFER_NATIVE_CUDA
            const auto memory=tyloi::cuda::default_context(ids.get_device())->memory_stats();
            const auto local=tyloi::cuda::default_context(ids.get_device())->local_memory_usage();
            cudaMemPool_t pool=nullptr;uint64_t reserved=0,used=0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool,ids.get_device()));
            TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
            TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
            std::cout<<"prefill_layer_major_memory tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" owned_bytes="<<memory.allocated
                <<" process_local_bytes="<<local<<" pool_used_bytes="<<used<<" pool_reserved_bytes="<<reserved<<std::endl;
#endif
            std::cout<<"prefill_layer_major_progress tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" total_layers="<<model.blocks.size()
                <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count()<<std::endl;
        }
    }
    profiler.enabled=restore.enabled;
    if(last_only && !raw_hidden) {
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        DenseTensor result;
        {
            auto finalized=qwen4_finalize_hidden(model, hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
            result=qwen4_last_hidden(std::move(finalized)).unsqueeze(1);
        }
        if(computation_finished)computation_finished();
        if(borrowed_workspace) {
            workspace_loan.suspend();
            result=result.clone();
            hidden={};
            workspace_loan.finish();
        }
        state.finish_forward(prepared_full,1,tokens);
        state.cache_pos=cache_start+tokens;
        return result;
    }
    DenseTensor finalized;
    if(last_only) {
        // Use the same final GR dispatch and rounding as chunk-major prefill.
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        finalized=qwen4_finalize_hidden(model, hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
        finalized=qwen4_last_hidden(std::move(finalized)).unsqueeze(1);
    } else {
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
            const auto count=std::min(compute_chunk,tokens-begin);
            auto part=qwen4_finalize_hidden(model, hidden.narrow(1,begin,count),1,count);
            if(!finalized.defined())finalized=::tyloi::cuda::empty({1,tokens,part.size(-1)},part.options());
            finalized.narrow(1,begin,count).copy_(part);
        }
    }
    if(raw_hidden && raw_hidden_host)hidden=hidden.to(::tyloi::kCPU);
    if(computation_finished)computation_finished();
    if(borrowed_workspace) {
        // The final mixer also needs sequence-sized scratch. Finish it inside
        // the loan, then retain only the outputs consumed by MTP/the caller.
        workspace_loan.suspend();
        if(raw_hidden) {if(!raw_hidden_host)hidden=hidden.clone();}else hidden={};
        finalized=finalized.clone();
        workspace_loan.finish();
    }
    state.finish_forward(prepared_full,1,tokens);
    state.cache_pos=cache_start+tokens;
    if(raw_hidden)*raw_hidden=hidden;
    return finalized;
}

namespace qwen4_exp {

tyloi::models::Info Model::info() const {
    return {metadata.vocab_size, config.maximum, "qwen4_exp",
            std::size_t(metadata.hidden_size) * sizeof(float) * 8, true, true};
}

MfqModelCapabilities Model::capabilities() const {
    const auto state = cuda_component_state(graph, plan, false, false);
    const auto caps = cuda_runtime_capabilities(graph, plan, state, false, false);
    return {graph.architecture, caps.text, caps.image_input, caps.video_input,
        caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp,
        "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
    const auto state = cuda_component_state(graph, plan, false, false);
    return {{"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0},
        {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)},
        {"vision_available", double(state.vision_available)},
        {"mtp_declared", double(state.mtp_declared)},
        {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", double(state.mtp_available)}};
}

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    return std::make_unique<State>(const_cast<Model&>(*this));
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>&, const MfqMultimodalInput&, tyloi::models::State&) const {
    throw std::invalid_argument("model has no multimodal component");
}

std::shared_ptr<::tyloi::Tensor> Model::embedding(const ::tyloi::Tensor& ids, int) const {
    auto scope = execution_scope();
    return std::make_shared<::tyloi::DenseTensor>(embed_tokens(device_ids(dynamic_cast<const ::tyloi::DenseTensor&>(ids))));
}

::tyloi::Tensor Model::normalize(const ::tyloi::Tensor& hidden, std::int64_t, std::int64_t) const {
    return hidden;
}

::tyloi::Tensor Model::softcap(const ::tyloi::Tensor& logits) const {
    return logits;
}

::tyloi::Tensor Model::logits(const ::tyloi::Tensor& hidden, bool) const {
    return execution->executor().profiler().measure("model.lm_head", [&] {
        return execution->executor().projector(execution->executor().mode().ffn,
            model_parallel_primary_device(execution->config)).run(lm_head, hidden.dense());
    });
}

std::shared_ptr<::tyloi::Tensor> Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                    const tyloi::models::ForwardOptions&) const {
    return std::make_shared<::tyloi::DenseTensor>(qwen4_forward(const_cast<Model&>(*this), static_cast<State&>(state_value), std::move(input)));
}

::tyloi::Tensor Model::hidden_forward_speculative_suffix(State& state, ::tyloi::Tensor ids,
                                                       ::tyloi::Tensor* raw_hidden) {
    tyloi::models::require_model(state.speculative_start >= 0 && state.speculative_confirmed == 0 &&
                                     !state.speculative_suffix_forward,
                                 "speculative suffix is not active");
    state.speculative_suffix_forward = true;
    try {
        auto result = qwen4_forward(*this, state,
                                    tyloi::models::Input{std::move(ids), {}, {}, {}, {}, nullptr,
                                                         raw_hidden, false, 0, 0, 0});
        state.speculative_suffix_forward = false;
        return result;
    } catch (...) {
        state.speculative_suffix_forward = false;
        throw;
    }
}

void State::rollback_speculative(std::int64_t accepted) {
    tyloi::models::require_model(speculative_start >= 0 && accepted >= 0,
                                 "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    speculative_prefix_retention=false;
    for (std::size_t i=0;i<model_.blocks.size();++i)
        model_.blocks[i]->rollback_speculative(*layers[i],keep);
    if (positions.defined()) positions=positions.narrow(-1,0,keep);
    cache_pos = keep;
    speculative_start = -1;
    speculative_confirmed = 0;
}

void Model::load_config(std::string_view payload) {
    auto& config = this->config;
    auto& metadata = this->metadata;
    config = CausalConfig::from_json(payload);
    metadata.vocab_size = config.vocab;
    metadata.hidden_size = config.hidden;
    metadata.num_hidden_layers = config.layers;
    metadata.num_attention_heads = config.heads;
    metadata.num_key_value_heads = config.kv_heads;
    metadata.head_dim = config.width;
    metadata.max_position_embeddings = config.maximum;
    metadata.rotary_dim = config.rotary;
    metadata.num_experts = config.experts;
    metadata.hc_mult = config.streams;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.eps;
    metadata.tie_word_embeddings = config.tied_embeddings;
    metadata.multi_axis_positions = true;
    metadata.flash_next = true;
    metadata.model_type = "qwen4_exp";
    metadata.layer_types = config.layer_types;
}


} // namespace qwen4_exp
} // namespace tyloi::cuda
