#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"
#include "engine/models/session/snapshot.h"
#include "env.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/execution.h"
#endif
#include "engine/backends/cuda/executor/decode_graph.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/state/kv_memory.h"
#endif
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
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
#include "engine/models/qwen4_exp/components.h"
#include "predictor.h"
#include "engine/models/common/load.h"
#include "engine/models/load.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/source.h"
#include "engine/weights/tq/rows.h"
#include "engine/generation/prefill_budget.h"
#include "engine/weights/nint_row_pipeline.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/kernels/moe/pipeline.h"
#include <iostream>
#include <atomic>
#include "engine/ops.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops/compact.h"
#include "engine/ops/positions.h"
#endif
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/ffn/gated.h"
#endif
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/weights/nint_cache.h"
#endif
#ifdef TYLOINFER_NATIVE_CUDA
#include "matmul_rows.h"
#include "engine/backends/cuda/native/graph.h"
#endif
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <exception>

namespace tyloi::cuda {
namespace {

struct GraphCheckpoint {
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<::tyloi::Tensor> gdn_conv, gdn_state;
#else
    std::vector<::tyloi::DenseTensor> gdn_conv, gdn_state;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<::tyloi::Tensor> ple_conv;
#else
    std::vector<::tyloi::DenseTensor> ple_conv;
#endif
};

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::ops::Residual::Result graph_attention(const qwen4_exp::Layer& weights, qwen4_exp::LayerState& state,
#else
GatedResidual::Result graph_attention(const qwen4_exp::Layer& weights, qwen4_exp::LayerState& state,
#endif
        ::tyloi::DenseTensor x, const ::tyloi::DenseTensor& ids, const ::tyloi::DenseTensor& positions,
        const ::tyloi::DenseTensor& cache_positions, const ::tyloi::DenseTensor& ple_rows, int layer,
        int64_t confirmed = 0, GraphCheckpoint* checkpoint = nullptr,
#ifdef TYLOINFER_NATIVE_CUDA
        const std::optional<tyloi::ops::Residual::Result>& pending_residual = {}) {
#else
        const std::optional<GatedResidual::Result>& pending_residual = {}) {
#endif
    if(auto* window=DecodeWindow::recording())window->set_profile_layer_kind(layer,weights.gdn?"gdn":"qsa");
    const auto mark=[&](const char* phase){if(auto* window=DecodeWindow::recording())window->mark(layer,phase);};
#ifndef TYLOINFER_NATIVE_CUDA
    const bool detailed=::tyloi::cuda::options::mhc_timings();
    const auto marker=[&](const char* role)->GatedResidualMarker {
        if(!detailed || !DecodeWindow::recording())return {};
        return [&,role](const char* phase) {
            const auto name=std::string(role)+phase;mark(name.c_str());
        };
    };
#endif
    if(weights.ple) {
        auto ple=confirmed ? weights.ple->forward_graph(state.ple, x,ids,ple_rows,confirmed,checkpoint->ple_conv)
            : weights.ple->forward_chunk(state.ple, x,ids,true,{},ple_rows);
#ifdef TYLOINFER_NATIVE_CUDA
        x=tyloi::ops::add(x,ple).dense();
#else
        x=x+ple;
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA
    auto [attention_input,attention_residual,attention_gate]=pending_residual
        ? weights.attention_gr.pre_after(x,pending_residual->residual,pending_residual->gate,tyloi::ops::ResidualRole::attention)
        : weights.attention_gr.pre(x,tyloi::ops::ResidualRole::attention);
#else
    auto [attention_input,attention_residual,attention_gate]=pending_residual
        ? weights.attention_gr.pre_after(x,pending_residual->residual,pending_residual->gate,marker("attention_mhc_"))
        : weights.attention_gr.pre(x,marker("attention_mhc_"));
#endif
    mark("attention_core_begin");
    auto attention_output=weights.gdn
        ? (confirmed ? weights.gdn->forward_graph(state.gdn, attention_input,confirmed,checkpoint->gdn_conv,checkpoint->gdn_state)
                     : weights.gdn->forward_chunk(state.gdn, attention_input,true))
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
        : weights.qsa->forward_graph(*state.qsa, attention_input,positions,cache_positions)
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
            ;
    mark("attention_core_end");
#ifdef TYLOINFER_NATIVE_CUDA
    auto ffn_result=weights.ffn_gr.pre_after(attention_output,attention_residual,attention_gate,tyloi::ops::ResidualRole::ffn);
#else
    auto ffn_result=weights.ffn_gr.pre_after(attention_output,attention_residual,attention_gate,marker("ffn_mhc_"));
#endif
    mark("ffn_pre_end");
    attention_output={};
    return ffn_result;
}

} // namespace

#ifndef TYLOINFER_NATIVE_CUDA
static bool streamed_decode_graphs(const CudaExecutionContext& execution) {
    if(!execution.offload)return true;
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_KV_STREAM_GRAPHS");
    return execution.kv && execution.kv->offload() && (!option || option[0]!='0');
}
#endif
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
#ifndef TYLOINFER_NATIVE_CUDA
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
#endif

#ifndef TYLOINFER_NATIVE_CUDA
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
#endif



#ifndef TYLOINFER_NATIVE_CUDA
Moe::Moe(std::vector<Linear> gate_up, Linear down, Linear router,
         Linear shared_gate, Linear sg, Linear su, Linear sd, const Config &c)
    : config(c), gate_up(std::move(gate_up)), down(std::move(down)), router(std::move(router)),
      shared_gate(std::move(shared_gate)), shared{std::move(sg), std::move(su), std::move(sd)} {
    TYLOINFER_RUNTIME_CHECK(shared[0] && shared[1] && shared[2] && this->shared_gate,
        "Qwen shared expert requires projection weights");
    auto banks = tq_banks(this->gate_up.empty() ? nullptr : this->gate_up.front(),
        this->gate_up.empty() ? nullptr : this->gate_up.back(), this->down);
    if (banks.gate || banks.up || banks.down) {
        TYLOINFER_RUNTIME_CHECK(this->gate_up.size() == 2 && this->down &&
            banks.gate && banks.up && banks.down,
            "TQ MoE requires separate Gate, Up and Down banks");
        tq = true;
    }
}

DenseTensor Moe::forward(const DenseTensor &x) const {
    const auto& c = config;
    auto source = x.reshape({-1, c.hidden}).to(::tyloi::kFloat16);
    const auto routing = c.routing();
    auto selected = moe_topk_cuda(
        qwen_prefill_preserve_rows(source,[&](const DenseTensor& part) {
            return tyloi::ops::project(part, router).dense();
        }).contiguous(),
        c.topk, routing.activation == tyloi::ops::RouterActivation::sigmoid,
        routing.activation == tyloi::ops::RouterActivation::sqrt_softplus,
        routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
    return tyloi::ops::moe(source, selected[0], selected[1], gate_up, down, shared,
                           shared_gate, c.moe_width, qwen_prefill_reference_rows)
        .dense().reshape(x.sizes());
}
#endif


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
#ifdef TYLOINFER_NATIVE_CUDA
    auto normalized=::tyloi::ops::contiguous(::tyloi::ops::copy(
        ::tyloi::ops::cast(ids,::tyloi::kInt64),::tyloi::Device(::tyloi::kCUDA,primary)));
    return ::tyloi::ops::measure("model.embed",normalized.device(),[&] {
        return ::tyloi::ops::embedding(normalized,embedding);
    }).dense();
#else
    return tyloi::cuda::embed(*embedding, std::move(ids), execution->executor().collectives(),
                           execution->executor().profiler(), primary);
#endif
}

DenseTensor Predictor::logits(DenseTensor hidden) const {
#ifdef TYLOINFER_NATIVE_CUDA
    auto projected=::tyloi::ops::measure("model.lm_head",hidden.device(),[&] {
        return ::tyloi::ops::project(hidden,output);
    });
    return ::tyloi::ops::softcap(projected,softcap).dense();
#else
    return tyloi::cuda::logits(*output, std::move(hidden), execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)),
                            execution->executor().profiler(), softcap);
#endif
}

void Predictor::reset(tyloi::models::MtpState& value, int64_t next_batch) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(next_batch > 0, "Qwen4-Exp MTP batch must be positive");
    for (std::size_t i=0; i<layers.size(); ++i)
        request.layer_states[i]->reset(*layers[i]);
    for (auto &pos : request.positions)
        pos = {};
    std::fill(request.lengths.begin(), request.lengths.end(), 0);
    request.batch = next_batch;
}

#ifdef TYLOINFER_NATIVE_CUDA
std::vector<std::pair<::tyloi::Device, std::size_t>> Predictor::growth(
        const tyloi::models::MtpState& value, const tyloi::models::Mtp::Input& input) const {
    const auto* step = std::get_if<tyloi::models::Mtp::Step>(&input);
    if (!step) throw std::invalid_argument("Qwen4 MTP supports step prediction only");
    const auto& request = state<PredictorState>(value);
    const auto& ids = step->ids;
    if (!ids.defined() || ids.dim() != 2 || ids.size(0) <= 0 || ids.size(1) <= 0 ||
        !ids.is_cuda() || request.layer_states.size() != layers.size() ||
        request.positions.size() != layers.size() || request.lengths.size() != layers.size() || layers.empty())
        throw std::invalid_argument("invalid Qwen4 MTP predictor geometry");
    const auto batch = ids.size(0), tokens = ids.size(1);
    const bool resets = request.batch && request.batch != batch;
    const auto start = resets ? 0 : request.lengths.front();
    if (start < 0 || start > config.maximum || tokens > config.maximum - start)
        throw std::invalid_argument("predictor history exceeds context capacity");
    const auto multiply = [](std::size_t a, std::size_t b) {
        if (b && a > std::numeric_limits<std::size_t>::max() / b)
            throw std::overflow_error("Qwen4 MTP allocation size overflow");
        return a * b;
    };
    std::vector<std::pair<::tyloi::Device, std::size_t>> result;
    const auto add = [&](::tyloi::Device device, std::size_t bytes) {
        if (!bytes) return;
        auto found = std::find_if(result.begin(), result.end(),
            [device](const auto& item) { return item.first == device; });
        if (found == result.end()) result.emplace_back(device, bytes);
        else {
            if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
                throw std::overflow_error("Qwen4 MTP allocation size overflow");
            found->second += bytes;
        }
    };
    const auto& block = *layers.front();
    if (!block.qsa || !request.layer_states.front() || !request.layer_states.front()->qsa)
        throw std::invalid_argument("Qwen4 MTP QSA state is unavailable");
    const auto& cache = *request.layer_states.front()->qsa;
    const auto target = ::tyloi::Device(::tyloi::kCUDA, block.cuda_device);
    add(target,block.qsa->allocation_bytes(cache,batch,tokens,resets));
    const auto rows = multiply(std::size_t(batch), std::size_t(start + tokens));
    add(ids.device(), multiply(multiply(rows, 3), sizeof(std::int32_t)));
    return result;
}
#endif

std::pair<DenseTensor, DenseTensor> Predictor::evaluate(tyloi::models::MtpState& value, const DenseTensor &hidden,
                                       const DenseTensor &ids, int64_t depth, bool cache,
                                       const DenseTensor &supplied_positions,
                                       const DenseTensor &supplied_embeddings, bool history_only) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(!history_only || (cache && layers.size()==1),
        "cache-only MTP priming requires a single predictor layer");
#ifdef TYLOINFER_NATIVE_CUDA
    TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0 &&
                          hidden.dim() == 3 && hidden.size(0) == ids.size(0) &&
                          hidden.size(1) == ids.size(1) &&
                          hidden.size(2) == hidden_norm.numel(),
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
#ifdef TYLOINFER_NATIVE_CUDA
    {
        auto positions = tyloi::ops::position_axes(start,t,b,3,supplied_positions,
            cache ? request.positions[layer] : tyloi::Tensor{},ids.options().dtype(tyloi::kInt32));
        pos = {positions[0].dense(),positions[1].dense()};
    }
#else
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
#ifdef TYLOINFER_NATIVE_CUDA
                        ? ::tyloi::cuda::cat({request.positions[layer].dense(), current}, -1)
#else
                        ? ::tyloi::cuda::cat({request.positions[layer], current}, -1)
#endif
                        : current;
        pos = {current, full};
    }
#endif
    DenseTensor x;
    {
        DenseTensor e;
        {
#ifdef TYLOINFER_NATIVE_CUDA
            e = ::tyloi::ops::project(rms_norm(embeds, embedding_norm.dense() + 1, config.eps),
                                      embedding_fusion).dense();
#else
            e = execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*embedding_fusion,
                                    rms_norm(embeds, embedding_norm + 1, config.eps));
#endif
        }
        // Release prior-stage tensors before later projections allocate.
        embeds = {};
        DenseTensor streams;
#ifdef TYLOINFER_NATIVE_CUDA
        streams = ::tyloi::ops::project(rms_norm(hidden, hidden_norm.dense() + 1, config.eps)
                                           .reshape({b, t, config.streams, config.hidden}),hidden_fusion).dense();
#else
        streams = execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)).run(*hidden_fusion,
                                         rms_norm(hidden, hidden_norm + 1, config.eps)
                                             .reshape({b, t, config.streams, config.hidden}));
#endif
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
            auto attention_output=block.qsa->forward(*request.layer_states[layer]->qsa,attention_input,pos[0],pos[1],cache)
#ifdef TYLOINFER_NATIVE_CUDA
                .dense()
#endif
                ;
            x=block.attention_gr.post(attention_output,attention_residual,attention_gate)
#ifdef TYLOINFER_NATIVE_CUDA
                .dense()
#endif
                ;
            attention_output={};
            auto [ffn_input,ffn_residual,ffn_gate]=block.ffn_gr.pre(x);
#ifdef TYLOINFER_NATIVE_CUDA
            auto ffn_output=tyloi::ops::ffn(block.ffn,ffn_input).dense();
#else
            auto ffn_output=block.ffn.forward(ffn_input);
#endif
            multi = block.ffn_gr.post(ffn_output,ffn_residual,ffn_gate)
#ifdef TYLOINFER_NATIVE_CUDA
                .dense()
#endif
                ;
        }
    }
    x = {};
    auto result = history_only?DenseTensor{}:final_mixer->pre(multi).mixed
#ifdef TYLOINFER_NATIVE_CUDA
        .dense()
#endif
        ;
    if (cache) {
        request.positions[layer] = pos[1];
        request.batch = b;
        request.lengths[layer] = start + t;
    }
    return {std::move(result), std::move(multi)};
}

tyloi::models::MtpStep Predictor::step(tyloi::models::MtpState& value, const tyloi::Tensor& input_hidden, const tyloi::Tensor& input_ids,
                                      const tyloi::Tensor& input_positions) const {
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
    auto result = evaluate(value,hidden, ids, 0, true, positions);
    return {std::make_shared<DenseTensor>(std::move(result.first)),
            std::make_shared<DenseTensor>(std::move(result.second))};
}

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::models::Mtp::History Predictor::history(
        std::int64_t tokens, std::int64_t requested, std::size_t available) const {
    History result;
    result.stream = history_stream && tokens > 16384;
    const auto bytes = std::size_t(tokens) * config.hidden * config.streams * sizeof(std::uint16_t);
    result.host = history_host < 0 ? bytes + (std::size_t(32) << 20) > available : history_host != 0;
    result.bootstrap = history_bootstrap;
    result.cache_only = history_cache_only && layers.size() == 1;
    result.chunk = std::min<std::int64_t>(1024, requested);
    result.merge = 128;
    result.flush = history_early_flush ? 4 : 512;
    result.parked = history_early_flush ? 16 : 512;
    result.workspace_fixed = std::size_t(64) << 20;
    result.workspace_per_row = std::size_t(config.hidden) * config.streams * sizeof(float) * 12;
    return result;
}

void Predictor::prime(tyloi::models::MtpState& value, const tyloi::Tensor& hidden,
        const tyloi::Tensor& ids, const tyloi::Tensor& positions, bool cache_only) const {
    if (!cache_only) {
        (void)step(value, hidden, ids, positions);
        return;
    }
    (void)evaluate(value, hidden.dense(), ids.dense(), 0, true,
        positions ? positions.dense() : DenseTensor{}, {}, true);
}
#endif

void Predictor::history_step(tyloi::models::MtpState& value,DenseTensor hidden,DenseTensor ids,DenseTensor positions,bool scalar_chain) const {
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(positions.defined()) {
        if(positions.dim()==1)positions=positions.reshape({1,1,-1}).expand({3,ids.size(0),ids.size(1)});
        else if(positions.dim()==2)positions=positions.unsqueeze(1);
        TYLOINFER_RUNTIME_CHECK(positions.dim()==3 && positions.size(0)==3 &&
            positions.size(1)==ids.size(0) && positions.size(2)==ids.size(1),"invalid MTP history positions");
        positions=positions.to(::tyloi::kInt32).contiguous();
    }
    if(scalar_chain && ids.size(1)>1) {
        for(int64_t row=0;row<ids.size(1);++row) {
            auto row_positions=positions.defined()?positions.narrow(-1,row,1):DenseTensor{};
            (void)step(value,hidden.narrow(1,row,1),ids.narrow(1,row,1),
                       row_positions.defined() ? ::tyloi::Tensor(row_positions) : ::tyloi::Tensor{});
        }
    } else (void)step(value,hidden,ids,positions.defined() ? ::tyloi::Tensor(positions) : ::tyloi::Tensor{});
}

void Predictor::trim_cache_to(tyloi::models::MtpState& value, int64_t position) const {
    auto& request=state<PredictorState>(value);
    TYLOINFER_RUNTIME_CHECK(position >= 0 && position <= request.position(),
                      "Qwen4-Exp MTP cache trim position is invalid");
    constexpr size_t index = 0;
    if (index < layers.size() && layers[index]->qsa)
        layers[index]->qsa->truncate(*request.layer_states[index]->qsa,position);
    if (request.positions[index].defined())
#ifdef TYLOINFER_NATIVE_CUDA
        request.positions[index] = tyloi::ops::slice(request.positions[index], -1, 0, position);
#else
        request.positions[index] = request.positions[index].narrow(-1, 0, position);
#endif
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

#ifdef TYLOINFER_NATIVE_CUDA
bool Predictor::supports_session_state(const tyloi::models::MtpState& value) const noexcept {
    if (!value.belongs_to(*this) || layers.empty()) return false;
    const auto* request = dynamic_cast<const PredictorState*>(&value);
    if (!request || request->layer_states.size() != layers.size() ||
        request->positions.size() != layers.size() || request->lengths.size() != layers.size()) return false;
    for (std::size_t i=0; i<layers.size(); ++i) {
        if (!layers[i] || !request->layer_states[i]) return false;
        if (layers[i]->qsa && (!request->layer_states[i]->qsa ||
            !layers[i]->qsa->supports_session(*request->layer_states[i]->qsa))) return false;
    }
    return true;
}
#else
bool Predictor::supports_session_state() const noexcept {
    return execution && !execution->offload && !layers.empty();
}
#endif

MtpSessionState Predictor::capture_session_state(const tyloi::models::MtpState& value, int64_t target_position,const tyloi::Tensor& input_hidden) const {
    const auto& request=this->state<PredictorState>(value);
    const auto& hidden=input_hidden.dense();
    const auto position=target_position-1;
    TYLOINFER_RUNTIME_CHECK(
#ifdef TYLOINFER_NATIVE_CUDA
        supports_session_state(value) &&
#else
        supports_session_state() &&
#endif
        position>0 && position<=request.position() &&
        hidden.defined() && hidden.dim()==3 && hidden.size(0)==1 && hidden.size(1)==1 && hidden.size(2)==hidden_norm.numel(),
        "Qwen4 MTP session boundary is unavailable");
    MtpSessionState state;state.cache_pos=position;
    auto& snapshot=state.payload.emplace<tyloi::models::qwen4_exp::MtpSnapshot>();
    for(std::size_t i=0;i<layers.size();++i) {
        const auto count=std::min(position,request.lengths[i]);
        snapshot.lengths.push_back(count);
        snapshot.layers.push_back(count?capture_layer_session(*layers[i],*request.layer_states[i],count,state.bytes):tyloi::models::qwen4_exp::LayerSnapshot{});
        ::tyloi::Tensor pos;
        if(count) {
#ifdef TYLOINFER_NATIVE_CUDA
            pos=tyloi::ops::clone(tyloi::ops::slice(request.positions[i],-1,0,count));
#else
            pos=std::make_shared<DenseTensor>(request.positions[i].narrow(-1,0,count).clone());
#endif
            state.bytes+=session_tensor_bytes(pos);
        }
        snapshot.positions.push_back(std::move(pos));
    }
    auto saved_hidden=hidden.clone();state.bytes+=session_tensor_bytes(saved_hidden);
    state.last_target_hidden=std::make_shared<DenseTensor>(std::move(saved_hidden));
    return state;
}

#ifdef TYLOINFER_NATIVE_CUDA
std::unique_ptr<tyloi::models::Restore> Predictor::prepare_restore(
        tyloi::models::MtpState& value, const MtpSessionState& saved) const {
    auto& request = state<PredictorState>(value);
    struct Candidate final : tyloi::models::Restore {
        const Predictor& owner;
        PredictorState& request;
        MtpSessionState saved;
        std::vector<std::unique_ptr<LayerState>> layers;
        std::vector<::tyloi::Tensor> positions;
        std::vector<std::int64_t> lengths;
        std::vector<std::pair<::tyloi::Device, std::size_t>> allocations;

        Candidate(const Predictor& owner, PredictorState& request, const MtpSessionState& snapshot)
            : owner(owner), request(request), saved(snapshot), layers(owner.layers.size()),
              positions(owner.layers.size()) {
            const auto* data = std::get_if<tyloi::models::qwen4_exp::MtpSnapshot>(&saved.payload);
            const auto* hidden = saved.last_target_hidden.representation<DenseTensor>();
            const auto device = ::tyloi::Device(::tyloi::kCUDA, owner.primary);
            if (!owner.supports_session_state(request) || saved.cache_pos <= 0 || saved.cache_pos > owner.config.maximum ||
                !data || data->layers.size() != owner.layers.size() || data->positions.size() != owner.layers.size() ||
                data->lengths.size() != owner.layers.size() || data->lengths.empty() ||
                data->lengths.front() != saved.cache_pos || request.layer_states.size() != owner.layers.size() ||
                !hidden || !hidden->defined() || hidden->dim() != 3 || hidden->size(0) != 1 ||
                hidden->size(1) != 1 || hidden->size(2) != owner.hidden_norm.numel() || hidden->device() != device ||
                (hidden->scalar_type() != ::tyloi::kFloat16 && hidden->scalar_type() != ::tyloi::kBFloat16 &&
                 hidden->scalar_type() != ::tyloi::kFloat32))
                throw std::invalid_argument("incompatible Qwen4 MTP session state");
            lengths = data->lengths;
            const auto add = [&](::tyloi::Device target, std::size_t bytes) {
                if (!bytes) return;
                auto found = std::find_if(allocations.begin(), allocations.end(),
                    [target](const auto& item) { return item.first == target; });
                if (found == allocations.end()) allocations.emplace_back(target, bytes);
                else {
                    if (found->second > std::numeric_limits<std::size_t>::max() - bytes)
                        throw std::overflow_error("Qwen4 MTP session allocation overflow");
                    found->second += bytes;
                }
            };
            for (std::size_t i = 0; i < owner.layers.size(); ++i) {
                const auto count = lengths[i];
                if (count < 0 || count > saved.cache_pos || !request.layer_states[i])
                    throw std::invalid_argument("incompatible Qwen4 MTP session length");
                if (!count) {
                    const auto& layer = data->layers[i];
                    if (data->positions[i] || layer.keys || layer.values || layer.index ||
                        layer.convolution || layer.recurrent || layer.ple_convolution ||
                        layer.ple_batch || !layer.ple_tokens.empty())
                        throw std::invalid_argument("incompatible empty Qwen4 MTP session layer");
                    continue;
                }
                const auto& pos = data->positions[i];
                if (!pos.representation<DenseTensor>() || !pos.defined() || pos.dim() != 3 ||
                    pos.size(0) != 3 || pos.size(1) != 1 || pos.size(2) != count ||
                    pos.scalar_type() != ::tyloi::kInt32 || pos.device() != device)
                    throw std::invalid_argument("incompatible Qwen4 MTP session positions");
                validate_layer_session(*owner.layers[i], *request.layer_states[i], data->layers[i], count);
                add(::tyloi::Device(::tyloi::kCUDA, owner.layers[i]->cuda_device),
                    layer_restore_growth(*owner.layers[i], *request.layer_states[i], data->layers[i], count));
                add(device, restore_tensor_bytes({}, pos));
            }
        }
        std::vector<std::pair<::tyloi::Device, std::size_t>> growth() const override { return allocations; }
        void stage() override {
            const auto& data = std::get<tyloi::models::qwen4_exp::MtpSnapshot>(saved.payload);
            for (std::size_t i = 0; i < owner.layers.size(); ++i) {
                layers[i] = std::make_unique<LayerState>(*owner.layers[i]);
                if (!lengths[i]) continue;
                stage_layer_session(*owner.layers[i], *layers[i], data.layers[i], lengths[i]);
                const auto& source = data.positions[i];
                positions[i] = tyloi::ops::empty(source.dense().sizes(), source.options());
                tyloi::ops::copy_into(positions[i], source);
            }
        }
        void publish() noexcept override {
            request.layer_states.swap(layers);
            request.positions.swap(positions);
            request.lengths.swap(lengths);
            request.batch = 1;
        }
    };
    return std::make_unique<Candidate>(*this, request, saved);
}
#else
void Predictor::restore_session_state(tyloi::models::MtpState& value, const MtpSessionState& state) const {
    auto& request=this->state<PredictorState>(value);
    const auto* snapshot=std::get_if<tyloi::models::qwen4_exp::MtpSnapshot>(&state.payload);
    const auto* hidden=state.last_target_hidden.representation<DenseTensor>();
    TYLOINFER_RUNTIME_CHECK(supports_session_state() && state.cache_pos>0 &&
        snapshot && snapshot->layers.size()==layers.size() && snapshot->positions.size()==layers.size() &&
        snapshot->lengths.size()==layers.size() && snapshot->lengths.front()==state.cache_pos &&
        hidden && hidden->defined() && hidden->dim()==3 &&
        hidden->size(0)==1 && hidden->size(1)==1 &&
        hidden->size(2)==hidden_norm.numel(),"incompatible Qwen4 MTP session state");
    reset(value,1);
    for(std::size_t i=0;i<layers.size();++i)if(snapshot->lengths[i]) {
        restore_layer_session(*layers[i],*request.layer_states[i],snapshot->layers[i],snapshot->lengths[i]);
        const auto* position=snapshot->positions[i].representation<DenseTensor>();
        TYLOINFER_RUNTIME_CHECK(position && position->defined(),"incompatible Qwen4 MTP session state");
        request.positions[i]=position->clone();request.lengths[i]=snapshot->lengths[i];
    }
}
#endif

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
#ifdef TYLOINFER_NATIVE_CUDA
    return model.final_mixer->pre(hidden,tyloi::ops::ResidualRole::final).mixed.dense();
#else
    auto* window=DecodeWindow::recording();
    GatedResidualMarker marker;
    if(window && ::tyloi::cuda::options::mhc_timings())marker=[window](const char* phase) {
        window->mark(-1,(std::string("final_mhc_")+phase).c_str());
    };
    return model.final_mixer->pre(hidden,marker).mixed;
#endif
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

::tyloi::DenseTensor qwen4_forward(const qwen4_exp::Model& model, qwen4_exp::State& state,
                                   tyloi::models::Input input) {
#ifndef TYLOINFER_NATIVE_CUDA
    auto scope = model.execution_scope();
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    const auto primary = model.primary_device;
#else
    const auto primary = model.execution->config.layer_placement.primary_device();
#endif
    tyloi::models::require_model(bool(input.ids), "Qwen4 forward requires token ids");
    auto ids = model.device_ids(input.ids.dense());
    auto embeddings = input.embeddings
        ? tyloi::ops::contiguous(tyloi::ops::copy(input.embeddings,
            ::tyloi::Device(::tyloi::kCUDA, primary))).dense()
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
        ? model.device_ids(input.positions.dense())
        : (state.decode_position_delta == 0 ? cache_positions
                                            : cache_positions + state.decode_position_delta);
    if ((current.dim() == 2 || current.dim() == 3) && current.size(0) == 4)
        current = current.narrow(0, 1, 3);
    auto full_positions = state.positions.defined()
#ifdef TYLOINFER_NATIVE_CUDA
        ? ::tyloi::cuda::cat({state.positions.dense(), current}, -1)
#else
        ? ::tyloi::cuda::cat({state.positions, current}, -1)
#endif
        : current;
    tyloi::models::require_model((current.dim() == 1 || (current.dim() == 2 && current.size(0) == 3) ||
                                     (current.dim() == 3 && current.size(0) == 3 && current.size(1) == B)) &&
                                     current.size(-1) == T,
                                 "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    DenseTensor host_ids, cpu_positions;
#ifdef TYLOINFER_NATIVE_CUDA
    const bool pinned = model.needs_host_tokens;
#else
    const bool pinned = model.execution->config.cache.pipeline ||
                        model.execution->config.dense_cpu_layer_count > 0;
#endif
    if (pinned) {
        host_ids = ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        cpu_positions = current.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
        for (std::size_t i=0;i<model.layers.size();++i)
            if (model.layers[i]->ple)
                model.layers[i]->ple->prefetch_tokens(state.layers[i]->ple, host_ids);
    }
    auto hidden = model.repeat_hidden(
        tyloi::ops::contiguous(tyloi::ops::copy(embeddings,
            ::tyloi::Device(::tyloi::kCUDA, primary))).dense(), model.metadata.hc_mult);
    state.begin_forward();
    const auto trace_layer = [&] {
        if (!input.trace) return;
        input.trace->push_back(
            std::make_shared<DenseTensor>(tyloi::ops::copy(hidden,
                ::tyloi::Device(::tyloi::kCUDA, primary)).dense().to(::tyloi::kFloat32).clone()));
    };
    if (input.trace)
        input.trace->push_back(std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).clone()));
    for (std::size_t i=0;i<model.layers.size();++i) {
        auto& block = model.layers[i];
#ifndef TYLOINFER_NATIVE_CUDA
        MfqCudaGuard guard(block->cuda_device);
#endif
        auto local_ids = block->cpu_offloaded
            ? host_ids
            : tyloi::ops::copy(ids, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        auto local_pos = block->cpu_offloaded
            ? cpu_positions
            : tyloi::ops::copy(current, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        auto local_full_positions = tyloi::ops::copy(full_positions,
            ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        hidden = block->cpu_offloaded
            ? hidden.to(::tyloi::kCPU).contiguous()
            : tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, block->cuda_device)).dense();
        hidden = qwen4_exp::forward(*block, *state.layers[i],
                                std::move(hidden), local_ids, local_pos, local_full_positions,
                                confirmed, host_ids);
        trace_layer();
    }
    state.finish_forward(full_positions, B, T);
    state.cache_pos += T;
    hidden = tyloi::ops::copy(hidden, ::tyloi::Device(::tyloi::kCUDA, primary)).dense();
    if (input.raw_hidden) *input.raw_hidden = hidden;
    auto finalized = qwen4_finalize_hidden(model, std::move(hidden), B, T);
    return finalized;
}

} // namespace
#ifdef TYLOINFER_NATIVE_CUDA
bool Model::prefetch(tyloi::models::State& value,const ::tyloi::Tensor& host_ids) const {
    auto* target=dynamic_cast<State*>(&value);
    tyloi::models::require_model(target && target->belongs_to(*this), "Qwen4 state belongs to another model");
    auto& state=*target;
#else
bool Model::prefetch_token_rows(State& state,const ::tyloi::Tensor& host_ids) {
#endif
    if (!state.reuse_decode_graphs || !state.graph_registry) return false;
    bool issued=false;
    for(std::size_t i=0;i<layers.size();++i) {
        const auto& block=layers[i];
        if(block->ple && block->ple->graph_supported()) {
            block->ple->prefetch_tokens(state.layers[i]->ple,host_ids.dense(),true);
            issued=true;
        }
    }
    return issued;
}

namespace qwen4_exp {
#ifndef TYLOINFER_NATIVE_CUDA
// Target decode graphs use one compact-readout control buffer. This is shared
// by the target graph only; predictor graph sampling no longer exists.
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
#endif
}

struct Qwen4DecodeGraph {
#ifndef TYLOINFER_NATIVE_CUDA
    static inline thread_local bool startup_warmup=false;
#endif
    struct PleStage {
        qwen4_exp::Ple* ple;
        qwen4_exp::Ple::State* state;
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor rows;
#else
        ::tyloi::DenseTensor rows;
#endif
        std::shared_ptr<tyloi::weights::RowStage> transfer;
        std::vector<qwen4_exp::NgramEmbedding::Context> prefix_context;
#ifdef TYLOINFER_NATIVE_CUDA
        qwen4_exp::NgramEmbedding::Context next_context;
        PleStage(qwen4_exp::Ple* p,qwen4_exp::Ple::State& request,int tokens,int hidden):ple(p),state(&request) {
            transfer=ple->graph_stage(*state,tokens,::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()});
            transfer->prepare_graph();
            rows=tyloi::ops::reshape(transfer->output(),{1,tokens,hidden});
        }
#else
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
#endif
    };
    struct Cell {
        ::tyloi::DenseTensor ids,positions,cache_positions,host_ids,output,raw_hidden,logits,greedy_host,compact_host;
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::Compact compact_control;
        ::tyloi::Tensor compact_source,greedy_source;
#else
        qwen4_exp::MtpCompactControl compact_control;
#endif
        unsigned used_kinds=0;
        std::vector<std::unique_ptr<PleStage>> ple;
        std::vector<GraphCheckpoint> checkpoints;
        DecodeWindow window;
        Cell(int tokens,int hidden,const std::vector<std::unique_ptr<qwen4_exp::Layer>>& weights,
                const std::vector<std::unique_ptr<qwen4_exp::LayerState>>& layers,
#ifndef TYLOINFER_NATIVE_CUDA
                cudaStream_t stream,
#endif
                int confirmed,
                std::vector<GraphCheckpoint>& shared_checkpoints)
#ifndef TYLOINFER_NATIVE_CUDA
            :window(stream,::tyloi::cuda::options::layer_timings())
#endif
            {
            const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
            ids=::tyloi::cuda::zeros({1,tokens},options);positions=::tyloi::cuda::zeros({tokens},options);cache_positions=::tyloi::cuda::zeros({tokens},options);
            host_ids=::tyloi::cuda::zeros({1,tokens},options.device(::tyloi::kCPU));
            greedy_host=::tyloi::cuda::empty({tokens},options.device(::tyloi::kCPU).pinned_memory(true));
            compact_host=::tyloi::cuda::empty({tokens,64},options.device(::tyloi::kCPU).pinned_memory(true));
            if(confirmed && shared_checkpoints.empty())shared_checkpoints.resize(weights.size());
            std::size_t layer=0;
            for(const auto& block:weights) {
                auto& b=*block;
                if(b.ple)ple.push_back(std::make_unique<PleStage>(b.ple.get(),layers[layer]->ple,tokens,hidden));
                if(confirmed) {
                    auto& shared=shared_checkpoints[layer];
                    GraphCheckpoint checkpoint;
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
#ifndef TYLOINFER_NATIVE_CUDA
    MfqCudaStream stream,ple_stream;
#endif
    std::map<std::tuple<int,int,int>,std::unique_ptr<Cell>> cells;
    // Verification widths execute serially and commit/rollback before the next
    // window. Their snapshots can share stable allocations across graph shapes.
    std::vector<GraphCheckpoint> shared_checkpoints;
    qwen4_exp::State& request;
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<std::pair<tyloi::ops::Qsa*,tyloi::ops::Qsa::State*>> qsa;
#else
    std::vector<std::pair<qwen4_exp::Qsa*,qwen4_exp::Qsa::State*>> qsa;
#endif
    ::tyloi::DenseTensor position_storage;
    int64_t kv_capacity=0;
    bool failed=false;
    ::tyloi::DenseTensor last_compact;
    explicit Qwen4DecodeGraph(const qwen4_exp::Model& model,qwen4_exp::State& state)
#ifndef TYLOINFER_NATIVE_CUDA
        :stream(mfq_get_current_cuda_stream()),ple_stream(mfq_get_stream_from_pool()),request(state) {
#else
        :request(state) {
#endif
        for(std::size_t i=0;i<model.layers.size();++i)
            if(auto* attention=model.layers[i]->qsa.get())qsa.emplace_back(attention,request.layers[i]->qsa.get());
    }
    ~Qwen4DecodeGraph() {
#ifndef TYLOINFER_NATIVE_CUDA
        (void)cudaStreamSynchronize(ple_stream.stream());
#endif
        cells.clear();
    }
    void restore_legacy() {
        last_compact={};
        cells.clear();shared_checkpoints.clear();
        for(auto [attention,state]:qsa)attention->leave_graph(*state);
        for(auto& layer:request.layers)layer->retain_graph_storage=false;
    }
    bool reusable() const {
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::Qsa::Graph::Request row;
        for(auto [attention,state]:qsa)row.layers.emplace_back(attention,state);
        return !failed && tyloi::ops::Qsa::Graph({std::move(row)},false).reusable();
#else
        return !failed && !qsa.empty() && std::all_of(qsa.begin(),qsa.end(),
            [](const auto& attention){return attention.first->graph_capacity(*attention.second)>0;});
#endif
    }

#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::StepSequence<tyloi::models::Decoded> run(const qwen4_exp::Model& model,::tyloi::DenseTensor ids,int kind,
                                  ::tyloi::DenseTensor* raw_hidden,int confirmed,bool* initialized,
                                  ::tyloi::DenseTensor host_ids,::tyloi::DenseTensor* logits,
                                  std::vector<int32_t>* greedy_tokens,bool compact_sampling) {
#else
    ::tyloi::DenseTensor run(const qwen4_exp::Model& model,::tyloi::DenseTensor ids,int kind,
                                  ::tyloi::DenseTensor* raw_hidden=nullptr,int confirmed=0,bool* initialized=nullptr,
                                  const ::tyloi::DenseTensor& host_ids={},::tyloi::DenseTensor* logits=nullptr,
                                  std::vector<int32_t>* greedy_tokens=nullptr,bool compact_sampling=true) {
#endif
        if(failed)throw std::runtime_error("MFQ model session requires recovery after its previous error");
        last_compact={};
#ifndef TYLOINFER_NATIVE_CUDA
        if(stream.stream()!=mfq_current_cuda_stream())throw std::runtime_error("MFQ model window changed execution stream");
#else
        if(!cells.empty()) cells.begin()->second->window.validate();
#endif
        const int tokens=static_cast<int>(ids.size(1));
        // These consumers use the same single-row backbone. Keep its hidden
        // state, logits and selected token in one graph instead of reserving
        // a complete set of layer buffers for each output representation.
        const int result_kind=kind;
        if(tokens==1 && kind>=1 && kind<=4)kind=4;
#ifndef TYLOINFER_NATIVE_CUDA
        model.execution->weights.retire_prefill();
#endif
        const auto required=request.cache_pos+tokens;
        TYLOINFER_RUNTIME_CHECK(required<=model.max_position_embeddings(),"Qwen4 graph exceeds context capacity");
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::Qsa::Graph::Request row;
        for(auto [attention,state]:qsa)row.layers.emplace_back(attention,state);
        row.position=request.cache_pos;row.maximum=model.max_position_embeddings();row.reserve=request.decode_reserve_until;
        tyloi::ops::Qsa::Graph attention({std::move(row)},false);
        attention.admit(tokens,kv_capacity,raw_hidden!=nullptr);
        kv_capacity=attention.rows().front().capacity;
        if(attention.clear_required()) {
            co_yield tyloi::GraphPhase::ready;
            restore_legacy();
            attention.layout();
            kv_capacity=attention.rows().front().capacity;
        }
#else
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
#endif
        auto& owner=cells[{tokens,kind,confirmed}];
        if(!owner)owner=std::make_unique<Cell>(tokens,model.config.hidden,model.layers,request.layers,
#ifndef TYLOINFER_NATIVE_CUDA
            stream.stream(),
#endif
            confirmed,shared_checkpoints);
        owner->window.set_profile_geometry(raw_hidden?"mtp_target":"target_decode",tokens);
        auto& cell=*owner;
#ifdef TYLOINFER_NATIVE_CUDA
        cell.window.validate();
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
            co_yield tyloi::GraphPhase::ready;
            for(const auto& stage:cell.ple) {
                stage->prefix_context.clear();
                if(confirmed) for(int i=confirmed;i<tokens;++i)
                    stage->prefix_context.push_back(stage->ple->graph_prefix_context(*stage->state,cell.host_ids,i));
                auto hashes=stage->ple->graph_rows(*stage->state,cell.host_ids);
                const auto columns=static_cast<int64_t>(hashes.ids.size())/tokens;
                stage->transfer->set_rows(hashes.ids,{1,tokens,columns});
                stage->next_context=std::move(hashes.next);
            }
#else
            model.execution->weights.finish_exchange();
#endif
            // Prefix restoration and eager prefill update the same KV/index
            // storage; refresh pooled keys even when this graph is already valid.
#ifdef TYLOINFER_NATIVE_CUDA
            attention.prepare(0,request.positions);
#else
            for(auto [attention,state]:qsa)attention->prepare_graph(*state,request.positions,kv_capacity);
#endif
            if(!cell.window.valid()) {
#ifdef TYLOINFER_NATIVE_CUDA
                cell.window.prepare();
                DecodeGraphBranchScope branch_scope;
#else
                model.execution->weights.prepare_graph(::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()});
                DecodeGraphBranchScope branch_scope(model.execution->executor().mode().ffn.serial);
#endif
                const auto capture_started=std::chrono::steady_clock::now();
                struct Saved {::tyloi::Tensor* target;::tyloi::DenseTensor saved;const void* address;};
                std::vector<Saved> states;
                if(!
#ifdef TYLOINFER_NATIVE_CUDA
                   request.warming
#else
                   startup_warmup
#endif
                )
                    for(std::size_t i=0;i<model.layers.size();++i)
                        for(auto* value:request.layers[i]->graph_state(*model.layers[i]))
                            states.push_back({value,value->dense().clone(),value->dense().data_ptr()});
                auto restore=[&] {
                    for(const auto& state:states) {
                        TYLOINFER_RUNTIME_CHECK(state.target->dense().data_ptr()==state.address,"MFQ capture changed recurrent storage");
                        state.target->dense().copy_(state.saved);
                    }
                };
                auto body=[&] {
                    cell.window.mark(-1,"window_begin");
                    auto hidden=model.repeat_hidden(model.embed_tokens(cell.ids),model.metadata.hc_mult);
                    std::size_t ple_index=0;int layer=0;
#ifdef TYLOINFER_NATIVE_CUDA
                    std::optional<tyloi::ops::Residual::Result> pending_residual;
#else
                    std::optional<GatedResidual::Result> pending_residual;
#endif
                    const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
                    const bool chain_post=(!chain_setting || *chain_setting) && tokens<=8;
                    for(const auto& block:model.layers) {
                        cell.window.mark(layer,"block_begin");
                        auto& b=*block;::tyloi::DenseTensor rows;
                        if(b.ple) {
                            cell.window.mark(layer,"ple_begin");
                            auto* stage=cell.ple[ple_index++].get();
#ifdef TYLOINFER_NATIVE_CUDA
                            rows=stage->rows.dense();
                            stage->transfer->decode_graph();
#else
                            rows=stage->rows;
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
#endif
                            cell.window.mark(layer,"ple_end");
                        }
                        cell.window.mark(layer,"attention_begin");
#ifdef TYLOINFER_NATIVE_CUDA
                        std::optional<tyloi::ops::Residual::Result> next_residual;
#else
                        std::optional<GatedResidual::Result> next_residual;
#endif
                        bool defer=false;
                        if(chain_post && size_t(layer+1)<model.layers.size()) {
                            auto& next=*model.layers[layer+1];
#ifdef TYLOINFER_NATIVE_CUDA
                            defer=!next.ple && next.attention_gr.can_chain(pending_residual?pending_residual->residual:tyloi::Tensor(hidden));
#else
                            defer=!next.ple && next.attention_gr.can_fuse(pending_residual?pending_residual->residual:hidden);
#endif
                        }
                        {
                            auto& layer_state=*request.layers[layer];
                            auto ffn_result=graph_attention(b,layer_state,std::move(hidden),cell.ids,cell.positions,
                                cell.cache_positions,rows,layer,confirmed,confirmed?&cell.checkpoints[layer]:nullptr,pending_residual);
                            const auto& ffn_input=ffn_result.mixed;
#ifdef TYLOINFER_NATIVE_CUDA
                            auto ffn_output=tyloi::ops::ffn(b.ffn,ffn_input).dense();
#else
                            auto ffn_output=b.ffn.forward(ffn_input);
#endif
                            if(defer) {
                                next_residual=std::move(ffn_result);
                                hidden=std::move(ffn_output);
                            } else hidden=b.ffn_gr.post(ffn_output,ffn_result.residual,ffn_result.gate)
#ifdef TYLOINFER_NATIVE_CUDA
                                .dense()
#endif
                                ;
                        }
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
#ifdef TYLOINFER_NATIVE_CUDA
                                cell.compact_source=cell.compact_control.forward(cell.logits.reshape({tokens,-1}).contiguous());
                                tyloi::ops::copy_into(cell.compact_host,cell.compact_source,true);
#else
                                auto compact=sample_top_k_compact_if_cuda(cell.logits.reshape({tokens,-1}).contiguous(),64,
                                    cell.compact_control.device);
                                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.compact_host.data_ptr(),compact.data_ptr(),tokens*64*sizeof(int64_t),
                                    cudaMemcpyDeviceToHost,stream.stream()));
#endif
                            }
#ifdef TYLOINFER_NATIVE_CUDA
                            cell.greedy_source=tyloi::ops::sample_greedy(cell.logits.reshape({tokens,-1}).contiguous());
                            tyloi::ops::copy_into(cell.greedy_host,cell.greedy_source,true);
#else
                            auto greedy=sample_greedy_cuda(cell.logits.reshape({tokens,-1}).contiguous());
                            TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(cell.greedy_host.data_ptr(),greedy.data_ptr(),tokens*sizeof(int64_t),
                                cudaMemcpyDeviceToHost,stream.stream()));
#endif
                        }
                    }
                    else if(kind==2)cell.output=qwen4_next_token(model,qwen4_last_hidden(std::move(hidden)));
                    else if(kind==1)cell.output=model.logits(qwen4_last_hidden(std::move(hidden))).dense();
                    else cell.output=qwen4_logits_from_hidden(model, std::move(hidden));
                    cell.window.mark(-1,"window_end");
                };
#ifdef TYLOINFER_NATIVE_CUDA
                {
                    auto capture=cell.window.capture();
                    try {
                        while(capture->next()) {
                            body();
                            capture->submitted();
                            restore();
                            if(capture->warmup()) {
                                cell.output={};cell.raw_hidden={};cell.logits={};
                                cell.compact_source={};cell.greedy_source={};
                            }
                            capture->restored();
                        }
                    } catch(...) {
                        auto failure=capture->fail(std::current_exception());
                        restore();
                        std::rethrow_exception(failure);
                    }
                }
#else
                cell.window.capture(body,restore,[&]{cell.output={};cell.raw_hidden={};cell.logits={};});
#endif
                ++request.decode_graph_captures;
                std::cerr<<"runtime_qwen4_graph_capture role=target rows="<<tokens<<" kind="<<kind
                    <<" confirmed="<<confirmed<<" segments="<<cell.window.segments()<<" count="<<request.decode_graph_captures
                    <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-capture_started).count()<<'\n';
            }
            cell.window.run();
#ifdef TYLOINFER_NATIVE_CUDA
            for(const auto& stage:cell.ple)
                stage->ple->restore_graph_context(*stage->state,std::move(stage->next_context));
#endif
            cell.used_kinds|=1u<<result_kind;
            if(confirmed) {
                // Captured copies save the confirmed prefix. Bind their owned
                // storage only after replay, so generic commit/rollback keeps
                // exactly the same transaction lifecycle as eager execution.
                std::size_t ple_index=0,layer=0;
                for(const auto& block:model.layers) {
                    auto& b=*block;
                    const auto& checkpoint=cell.checkpoints[layer];
                    auto& layer_state=*request.layers[layer++];
                    const auto position=request.cache_pos+confirmed;
                    if(b.gdn)b.gdn->bind_graph_checkpoint(layer_state.gdn,checkpoint.gdn_conv,checkpoint.gdn_state,position);
                    if(b.ple)b.ple->bind_graph_checkpoint(layer_state.ple,checkpoint.ple_conv,cell.ple[ple_index++]->prefix_context,position);
                }
                request.speculative_start=request.cache_pos;request.speculative_confirmed=confirmed;
                request.speculative_prefix_retention=true;
            }
            for(std::size_t i=0;i<model.layers.size();++i)
                if(model.layers[i]->qsa)model.layers[i]->qsa->advance_graph(*request.layers[i]->qsa,tokens);
            request.cache_pos+=tokens;
            request.positions=append_decode_positions(
#ifdef TYLOINFER_NATIVE_CUDA
                request.positions.defined() ? request.positions.dense() : DenseTensor{},
#else
                request.positions,
#endif
                cell.positions,position_storage,
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
#ifdef TYLOINFER_NATIVE_CUDA
            tyloi::models::Decoded result;
            if(tokens==1 && result_kind==1)result.hidden=cell.logits.reshape({1,-1});
            else if(tokens==1 && result_kind==2)result.hidden=cell.greedy_host;
            else result.hidden=cell.output;
            co_yield std::move(result);
#else
            if(tokens==1 && result_kind==1)return cell.logits.reshape({1,-1});
            if(tokens==1 && result_kind==2)return cell.greedy_host;
            return cell.output;
#endif
        } catch(...) {failed=true;throw;}
    }
};
#ifndef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor qwen4_decode_graph_compact_topk(const qwen4_exp::Model& model,const qwen4_exp::State& state) {
    return state.graph_registry?state.graph_registry->last_compact: ::tyloi::DenseTensor{};
}

std::optional<::tyloi::Tensor> qwen4_decode_graph_logits(const qwen4_exp::Model& model,qwen4_exp::State& state,::tyloi::Tensor ids_handle,int kind,
        const ::tyloi::Tensor& host_ids) {
    auto ids = ids_handle.dense();
    if (!streamed_decode_graphs(*model.execution)) return {};
    const auto decode=model.execution->weights.decode();
    const bool tq=decode==::tyloi::weights::Cache::Decode::routed;
    if(decode==::tyloi::weights::Cache::Decode::none ||
            state.cache_pos==0 || state.speculative_start>=0)return {};
    if(ids.dim()==1)ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 || state.positions.dim()!=1)return {};
    if(tq) {
        if(!tq_graph_supports_tokens(ids.size(1)))return {};
        for(const auto& block:model.layers) {
            const auto& b=*block;
            if(!b.ffn.tq || (b.ple && !b.ple->graph_supported()))return {};
        }
    }
    TYLOINFER_RUNTIME_CHECK(state.cache_pos+ids.size(1)<=model.max_position_embeddings(),"MFQ window exceeds context");
    if(!state.graph_registry)state.graph_registry=std::make_unique<Qwen4DecodeGraph>(model,state);
    return ::tyloi::Tensor(state.graph_registry->run(model,model.device_ids(std::move(ids)),kind,nullptr,0,nullptr,
        host_ids ? host_ids.dense() : ::tyloi::DenseTensor{}));
}
#endif

void qwen4_report_decode_graph(const qwen4_exp::Model& model,const qwen4_exp::State& state,std::ostream& output) {
    if(!state.graph_registry)return;
    for(const auto& [key,cell]:state.graph_registry->cells)
        for(const auto& phase:cell->window.gpu_phases())if(phase.samples)
            output<<"mtp_graph_phase tokens="<<std::get<0>(key)<<" confirmed="<<std::get<2>(key)
                <<" layer="<<phase.layer<<" from="<<phase.from<<" to="<<phase.to
                <<" samples="<<phase.samples<<" total_ns="<<phase.total_ns<<'\n';
}

std::vector<std::pair<std::string,double>> qwen4_decode_graph_profile_metrics(
        const qwen4_exp::Model& model,const qwen4_exp::State& state) {
    std::vector<std::pair<std::string,double>> result;
    if(!::tyloi::cuda::options::layer_timings())return result;
    const auto append=[&](const auto& cells,const char* role,const char* attribute,const char* state) {
        for(const auto& [key,cell]:cells) {
            auto prefix=std::string("cuda_graph_profile_")+role+"_rows_"+
                std::to_string(std::get<0>(key))+"_"+attribute+"_"+std::to_string(std::get<1>(key))+
                "_"+state+"_"+std::to_string(std::get<2>(key))+"_";
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
    return result;
}

#ifndef TYLOINFER_NATIVE_CUDA
std::optional<::tyloi::Tensor> qwen4_decode_graph_hidden(
        const qwen4_exp::Model& model,qwen4_exp::State& state,::tyloi::Tensor ids_handle,::tyloi::Tensor* raw_hidden,int64_t confirmed,
        bool* initialized,const ::tyloi::Tensor& host_ids,::tyloi::Tensor* logits,
        std::vector<int32_t>* greedy_tokens,bool compact_sampling) {
    auto ids = ids_handle.dense();
    if(initialized)*initialized=false;
    const auto decode=model.execution->weights.decode();
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
    for(const auto& block:model.layers) {
        const auto& b=*block;
        if(!b.ffn.tq || (b.ple && !b.ple->graph_supported()))return {};
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
#else
tyloi::StepSequence<tyloi::models::Decoded> qwen4_exp::Model::decode(
        tyloi::models::Input input, tyloi::models::State& target,
        tyloi::models::ForwardOptions, tyloi::models::DecodeOptions options) const {
    auto* state=dynamic_cast<qwen4_exp::State*>(&target);
    tyloi::models::require_model(state && state->belongs_to(*this), "Qwen4 state belongs to another model");
    if (!input.ids || input.embeddings || input.positions || input.lengths || input.cache_positions ||
        input.image_mask || input.suffix || input.trace || input.kv_length || input.attention_parts)
        co_return;
    const bool tq=options.routed;
    tyloi::ops::Qsa::Graph::Request graph_row;
    for(std::size_t i=0;i<layers.size();++i)
        if(layers[i]->qsa)graph_row.layers.emplace_back(layers[i]->qsa.get(),state->layers[i]->qsa.get());
    tyloi::ops::Qsa::Graph graph_policy({std::move(graph_row)},false);
    if (!info().decode_graph || !options.routed || !graph_policy.eligible() || state->speculative_start>=0) co_return;
    const bool hidden=options.projection==tyloi::models::DecodeOptions::Projection::hidden;
    if (hidden) {
        if(state->cache_pos==0 && (!state->graph_registry || !state->graph_registry->reusable())) co_return;
    } else if(state->cache_pos==0 || input.confirmed!=0) co_return;
    auto ids=input.ids.dense();
    if(ids.dim()==1) ids=ids.unsqueeze(0);
    if(ids.dim()!=2 || ids.size(0)!=1 || ids.size(1)<1 || ids.size(1)>8 ||
       (state->cache_pos!=0 && state->positions.dim()!=1) || input.confirmed<0 ||
       (input.confirmed && input.confirmed>=ids.size(1))) co_return;
    if(options.projection==tyloi::models::DecodeOptions::Projection::greedy && ids.size(1)!=1) co_return;
    if(!hidden && input.raw_hidden && ids.size(1)!=1) co_return;
    if(tq && !tq_graph_supports_tokens(ids.size(1))) co_return;
    if(hidden || tq) for(const auto& block:layers)
        if(!block->ffn.qwen_tq || (block->ple && !block->ple->graph_supported())) co_return;
    TYLOINFER_RUNTIME_CHECK(state->cache_pos+ids.size(1)<=max_position_embeddings(), "MFQ window exceeds context");
    co_yield tyloi::GraphPhase::prepare;
    if(!state->graph_registry) state->graph_registry=std::make_unique<Qwen4DecodeGraph>(*this,*state);
    tyloi::models::Decoded result;
    ::tyloi::DenseTensor raw, logits;
    const int kind=hidden?4:(options.projection==tyloi::models::DecodeOptions::Projection::greedy?2:1);
    const auto* compact_option=hidden ? tyloi::cuda::env_lookup("TYLOINFER_MTP_CONDITIONAL_TOPK") : nullptr;
    const bool compact=options.compact || (compact_option && compact_option[0]=='0');
    auto job=state->graph_registry->run(*this,device_ids(std::move(ids)),kind,
        input.raw_hidden ? &raw : nullptr, static_cast<int>(input.confirmed),&result.initialized,
        input.host_ids ? input.host_ids.dense() : ::tyloi::DenseTensor{}, hidden ? &logits : nullptr,
        kind==1 ? nullptr : &result.tokens, compact);
    tyloi::Tensor output;
    while(auto step=job.next()) {
        if(step.layer)throw std::logic_error("unexpected decode layer event");
        if(step.graph) co_yield *step.graph;
        if(step.value) output=std::move(step.value->hidden);
    }
    if(hidden) {
        result.hidden=std::move(output); result.logits=std::move(logits);
        result.compact=state->graph_registry->last_compact;
    } else if(kind==1) result.logits=std::move(output);
    else result.ids=std::move(output);
    if(input.raw_hidden) *input.raw_hidden=std::move(raw);
    co_yield std::move(result);
}
#endif

#ifndef TYLOINFER_NATIVE_CUDA
void qwen4_release_decode_graphs(const qwen4_exp::Model& model,qwen4_exp::State& state) {
    model.execution->weights.finish_exchange();
    if(state.graph_registry)state.graph_registry->restore_legacy();
    state.graph_registry.reset();
    for(auto& layer:state.layers)layer->retain_graph_storage=false;
}
#endif

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::StepSequence<tyloi::models::Decoded> qwen4_exp::Model::warmup(
        tyloi::models::State& target, tyloi::models::DecodeOptions decode_options) const {
    auto* request=dynamic_cast<qwen4_exp::State*>(&target);
    tyloi::models::require_model(request && request->belongs_to(*this), "Qwen4 state belongs to another model");
    auto& state=*request;
    const auto& model=*this;
#else
void qwen4_warmup_decode_graphs(const qwen4_exp::Model& model,qwen4_exp::State& state) {
#endif
    const auto* reuse=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_REUSE_GRAPHS");
    state.reuse_decode_graphs=!reuse || reuse[0]!='0';
#ifdef TYLOINFER_NATIVE_CUDA
    const bool tq=decode_options.routed;
    tyloi::ops::Qsa::Graph::Request graph_row;
    for(std::size_t i=0;i<model.layers.size();++i)
        if(model.layers[i]->qsa)graph_row.layers.emplace_back(model.layers[i]->qsa.get(),state.layers[i]->qsa.get());
    tyloi::ops::Qsa::Graph graph_policy({std::move(graph_row)},false);
#else
    const auto decode=model.execution->weights.decode();
    const bool tq=decode==::tyloi::weights::Cache::Decode::routed;
#endif
    const auto* warmup=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
    if(!state.reuse_decode_graphs || (warmup && warmup[0]=='0') ||
        model.max_position_embeddings()<32 ||
#ifdef TYLOINFER_NATIVE_CUDA
        !model.info().decode_graph || !decode_options.routed || !graph_policy.eligible() || !graph_policy.warmup_supported())co_return;
#else
        !streamed_decode_graphs(*model.execution) || decode==::tyloi::weights::Cache::Decode::none)return;
#endif
    for(std::size_t i=0;i<model.layers.size();++i) {
        const auto& b=*model.layers[i];
        if(!b.ffn.
#ifdef TYLOINFER_NATIVE_CUDA
            qwen_tq ||
#else
            tq ||
#endif
#ifndef TYLOINFER_NATIVE_CUDA
            (b.qsa && !b.qsa->graph_capacity(*state.layers[i]->qsa)) ||
#endif
            (b.ple && !b.ple->graph_supported()))
#ifdef TYLOINFER_NATIVE_CUDA
            co_return;
#else
            return;
#endif
    }
    // Startup state contains synthetic tokens and is discarded before serving.
    // Saving every recurrent state during each capture needlessly doubles its
    // transient footprint. Request-time captures still restore their real state.
#ifdef TYLOINFER_NATIVE_CUDA
    co_yield tyloi::GraphPhase::warmup;
#else
    tyloi::weights::Cache::Warmup cache_warmup(model.execution->weights, true);
#endif
    struct StartupWarmup {
#ifdef TYLOINFER_NATIVE_CUDA
        qwen4_exp::State& state;
        bool previous;
        explicit StartupWarmup(qwen4_exp::State& state) : state(state), previous(state.warming) { state.warming=true; }
        ~StartupWarmup(){state.warming=previous;}
#else
        StartupWarmup(){Qwen4DecodeGraph::startup_warmup=true;}
        ~StartupWarmup(){Qwen4DecodeGraph::startup_warmup=false;}
#endif
    } startup_scope
#ifdef TYLOINFER_NATIVE_CUDA
        {state}
#endif
        ;
    const auto start=std::chrono::steady_clock::now();
#ifndef TYLOINFER_NATIVE_CUDA
    auto scope=model.execution_scope();
#endif
    const auto options=::tyloi::TensorOptions().device(::tyloi::kCUDA).dtype(::tyloi::kInt64);
    // Diverse synthetic IDs exercise more routed experts than sixteen copies
    // of one token. Only loading uses these rows; all request state is reset.
    std::vector<int64_t> warm_ids(16);
    for(std::size_t i=0;i<warm_ids.size();++i)warm_ids[i]=int64_t((i*7919+17)%model.metadata.vocab_size);
    auto ids=::tyloi::cuda::tensor(warm_ids,options.device(::tyloi::kCPU)).to(::tyloi::kCUDA).reshape({1,16});
#ifdef TYLOINFER_NATIVE_CUDA
    co_yield tyloi::GraphPhase::ready;
#endif
    state.reset(1);
    if(tyloi::cuda::env_lookup("TYLOINFER_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=bootstrap\n";
    // Bootstrap with single-token rows. A sixteen-row eager forward otherwise
    // materializes a whole expert layer just to initialize graph state, before
    // the prompt's workspace loan exists, and defeats tight expert residency.
    // Eager FFN cells retain their transfer wire per layer until this pass
    // completes; eight-row bootstrap cells multiply that transient footprint.
    for(int row=0;row<(tq?1:16);++row) {
        auto token=ids.narrow(1,row,1);
        {
            tyloi::models::Input in;
            in.ids = std::make_shared<::tyloi::DenseTensor>(token);
            (void)model.forward(in, state);
        }
    }
    // Bootstrap cells are not the retained whole-model windows. Retire their
    // staging and captured scratch before preparing target windows.
#ifdef TYLOINFER_NATIVE_CUDA
    co_yield tyloi::GraphPhase::target;
#else
    cache_warmup.release(tyloi::Device(tyloi::kCUDA, model.execution->config.layer_placement.primary_device()), true);
#endif
    if(tyloi::cuda::env_lookup("TYLOINFER_TRACE_RESIDENCY_MEMORY"))std::cerr<<"warmup_memory phase=target\n";
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::models::Input graph_input;
    graph_input.ids=ids.narrow(1,0,1);
    tyloi::models::DecodeOptions graph_options;
    graph_options.routed=decode_options.routed;
    graph_options.projection=tyloi::models::DecodeOptions::Projection::greedy;
    auto greedy=model.decode(graph_input,state,{},graph_options);
    while(auto step=greedy.next()) {
        if(step.layer)throw std::logic_error("unexpected warmup decode layer event");
        if(step.graph) co_yield *step.graph;
    }
    graph_options.projection=tyloi::models::DecodeOptions::Projection::logits;
    auto logits=model.decode(std::move(graph_input),state,{},graph_options);
    while(auto step=logits.next()) {
        if(step.layer)throw std::logic_error("unexpected warmup decode layer event");
        if(step.graph) co_yield *step.graph;
    }
    co_yield tyloi::GraphPhase::ready;
#else
    (void)qwen4_decode_graph_logits(model,state,ids.narrow(1,0,1),2);
    (void)qwen4_decode_graph_logits(model,state,ids.narrow(1,0,1),1);
#endif
    state.reset(1);
#ifdef TYLOINFER_NATIVE_CUDA
    co_yield tyloi::GraphPhase::settle;
#else
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
#endif
    std::cerr<<"runtime_qwen4_graph_warmup target_captures="<<state.decode_graph_captures<<" ms="
        <<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<'\n';
#ifdef TYLOINFER_NATIVE_CUDA
    co_return;
#else
    cache_warmup.release(tyloi::Device(tyloi::kCUDA, model.execution->config.layer_placement.primary_device()), false);
#endif
}
#ifdef TYLOINFER_NATIVE_CUDA
std::optional<tyloi::engine::Metrics> qwen4_exp::State::metrics() const {
    const auto& model=model_;
    const auto& state=*this;
    const auto* mtp=model.mtp();
#else
std::vector<std::pair<std::string,double>> qwen4_cache_memory_metrics(
        const qwen4_exp::Model& model,const qwen4_exp::State& state,const tyloi::models::Mtp* mtp) {
#endif
    // Shared checkpoint views and recurrent state aliases count once. These
    // are existing cache tensors; collecting telemetry never copies GPU data.
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add=[&](const ::tyloi::Tensor& tensor) {
        for (const auto& storage : tensor.backings()) {
            if (!storage->device.is_cuda()) continue;
            const void* identity = storage->owner ? storage->owner.get() : storage->base;
            if (!identity) continue;
            auto& bytes=allocations[{storage->device.index,reinterpret_cast<std::uintptr_t>(identity)}];
            bytes=std::max(bytes,storage->bytes);
        }
    };
    const auto block_cache=[&](const qwen4_exp::Layer& block,const qwen4_exp::LayerState& layer) {
        if(block.qsa)for(const auto& tensor:block.qsa->resident_cache_tensors(*layer.qsa))add(tensor);
        if(block.gdn) { add(layer.gdn.conv); add(layer.gdn.recurrent); }
        if(block.ple && block.ple->conv_state(layer.ple).defined())add(block.ple->conv_state(layer.ple));
    };
    for(std::size_t i=0;i<model.layers.size();++i)block_cache(*model.layers[i],*state.layers[i]);
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
    }
    std::size_t bytes=0;
    std::map<int,std::size_t> devices;
    for(const auto& [address,size]:allocations) { bytes+=size; devices[address.first]+=size; }
    std::vector<std::pair<std::string,double>> result{{"kv_cache_bytes",double(bytes)},
        {"kv_cache_contexts",state.cache_pos>0?double(state.batch):0.0}};
    for(const auto& [device,size]:devices)result.emplace_back("gpu_"+std::to_string(device)+"_live_kv_bytes",double(size));
#ifdef TYLOINFER_NATIVE_CUDA
    auto profile=qwen4_decode_graph_profile_metrics(model,state);
    result.insert(result.end(),profile.begin(),profile.end());
#endif
    return result;
}

Model::Model() = default;
Model::~Model() = default;

} // namespace tyloi::cuda

namespace tyloi::cuda {
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

#ifdef TYLOINFER_NATIVE_CUDA
static ::tyloi::DenseTensor batch_ffn(const tyloi::ops::FfnWeights& ffn,
#else
static ::tyloi::DenseTensor batch_ffn(const qwen4_exp::Moe& ffn,
#endif
#ifndef TYLOINFER_NATIVE_CUDA
                                      CudaExecutionContext& execution,
#endif
                                      const ::tyloi::DenseTensor& input) {
#ifdef TYLOINFER_NATIVE_CUDA
    if(input.size(0)<=8)return tyloi::ops::ffn(ffn,input).dense();
#else
    if(input.size(0)<=8)return ffn.forward(input);
#endif
    const auto option=::tyloi::cuda::options::continuous_ffn_tile();
    const int64_t rows=option?*option:8;
    TYLOINFER_RUNTIME_CHECK(rows==8 || rows==16,"continuous FFN tile must be 8 or 16 rows");
    auto* window=DecodeWindow::recording();
    const int previous_profile_tile=window ? window->set_profile_tile(-1) : -1;
    struct TileScope {
#ifndef TYLOINFER_NATIVE_CUDA
        tyloi::cuda::FfnMode& mode;
        int tile,tiles;
#endif
        DecodeWindow* window;
        int profile_tile;
        ~TileScope() {
#ifndef TYLOINFER_NATIVE_CUDA
            mode.tile=tile;mode.tiles=tiles;
#endif
            if(window)window->set_profile_tile(profile_tile);}
    };
#ifndef TYLOINFER_NATIVE_CUDA
    auto& mode = execution.executor().mode().ffn;
    TileScope scope{mode,mode.tile,mode.tiles,window,previous_profile_tile};
    mode.tiles=int((input.size(0)+rows-1)/rows);
#else
    TileScope scope{window,previous_profile_tile};
#endif
    std::vector<::tyloi::DenseTensor> outputs;
    for(int64_t start=0;start<input.size(0);start+=rows) {
        if(window)window->set_profile_tile(int(start/rows));
#ifndef TYLOINFER_NATIVE_CUDA
        mode.tile=int(start/rows);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        outputs.push_back(tyloi::ops::ffn(ffn,input.narrow(0,start,std::min(rows,input.size(0)-start))).dense());
#else
        outputs.push_back(ffn.forward(input.narrow(0,start,std::min(rows,input.size(0)-start))));
#endif
    }
    return ::tyloi::cuda::cat(outputs,0);
}

struct Qwen4BatchRequestState::Impl {
    const qwen4_exp::Model& model;
    std::vector<std::unique_ptr<qwen4_exp::LayerState>> layers;
    ::tyloi::DenseTensor positions, position_storage;
    int64_t cache_position = 0, decode_position_delta = 0;
    explicit Impl(const qwen4_exp::Model& owner) : model(owner) {}
    std::size_t growth_bytes(int64_t tokens) const {
        std::size_t bytes = 0;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            const auto& state = *layers[i];
            if (weights.qsa) bytes += weights.qsa->growth_bytes(*state.qsa,tokens);
            if (weights.gdn) bytes += weights.gdn->growth_bytes(state.gdn);
#ifdef TYLOINFER_NATIVE_CUDA
            if (weights.ple) bytes += weights.ple->growth_bytes(state.ple,1);
#endif
        }
        return bytes;
    }
};

Qwen4BatchRequestState::Qwen4BatchRequestState(qwen4_exp::Model& model)
    : state_(std::make_unique<Impl>(model)) {
#ifdef TYLOINFER_NATIVE_CUDA
    const auto primary = model.primary_device;
#else
    const auto primary = model.execution->config.layer_placement.primary_device();
#endif
    for (const auto& block : model.layers) {
        TYLOINFER_RUNTIME_CHECK(!block->cpu_offloaded && block->cuda_device == primary,
                          "Flash-Next continuous batching requires one CUDA device");
        state_->layers.push_back(std::make_unique<qwen4_exp::LayerState>(*block));
    }
}
Qwen4BatchRequestState::~Qwen4BatchRequestState() = default;
#ifdef TYLOINFER_NATIVE_CUDA
bool Qwen4BatchRequestState::belongs_to(const qwen4_exp::Model& model) const noexcept {
    return &state_->model == &model;
}
#endif
int64_t Qwen4BatchRequestState::position() const { return state_->cache_position; }
std::size_t Qwen4BatchRequestState::initial_growth_bytes() const {
    std::size_t fixed=0;
    for(std::size_t i=0; i<state_->layers.size(); ++i) {
        const auto& weights = *state_->model.layers[i];
        const auto& state = *state_->layers[i];
        if(weights.gdn)fixed+=weights.gdn->request_cache_growth_bytes(state.gdn);
    }
    return fixed;
}
void Qwen4BatchRequestState::initialize() {
    for(std::size_t i=0; i<state_->layers.size(); ++i) {
        const auto& weights = *state_->model.layers[i];
        auto& state = *state_->layers[i];
        if(weights.gdn)weights.gdn->initialize_request_cache(state.gdn,false);
    }
}
std::size_t Qwen4BatchRequestState::prefill_growth_bytes(int64_t tokens) const {
    return state_->growth_bytes(tokens);
}
std::size_t Qwen4BatchRequestState::resident_bytes() const {
    std::map<std::pair<int,std::uintptr_t>,std::size_t> allocations;
    const auto add = [&](const ::tyloi::Tensor& tensor) {
        for (const auto& storage : tensor.backings()) {
            if (!storage->device.is_cuda()) continue;
            const void* identity = storage->owner ? storage->owner.get() : storage->base;
            if (!identity) continue;
            auto& bytes = allocations[{storage->device.index,reinterpret_cast<std::uintptr_t>(identity)}];
            bytes = std::max(bytes, storage->bytes);
        }
    };
    add(state_->positions);
    add(state_->position_storage);
    for (std::size_t i = 0; i < state_->layers.size(); ++i) {
        const auto& weights = *state_->model.layers[i];
        const auto& state = *state_->layers[i];
        if (weights.qsa) for (const auto& tensor : weights.qsa->resident_cache_tensors(*state.qsa)) add(tensor);
        if (weights.gdn) { add(state.gdn.conv); add(state.gdn.recurrent); }
        if (weights.ple && weights.ple->conv_state(state.ple).defined()) add(weights.ple->conv_state(state.ple));
    }
    std::size_t bytes = 0;
    for (const auto& [address,size] : allocations) bytes += size;
    return bytes;
}
struct Qwen4BatchDecodeGraph::Impl {
    using DenseTensor = ::tyloi::DenseTensor;
    qwen4_exp::Model& model;
#ifndef TYLOINFER_NATIVE_CUDA
    MfqCudaStream stream;
    Event ready;
    std::unique_ptr<Graph> memory_pool;
#endif
    std::unique_ptr<DecodeWindow> window;
    std::vector<Qwen4BatchRequestState*> requests;
    std::vector<int64_t> capacities;
    std::vector<DenseTensor> positions, cache_positions;
    std::vector<DenseTensor> gdn_addresses;
    std::vector<std::vector<std::shared_ptr<tyloi::weights::RowStage>>> ple;
    DenseTensor ids, logits;
    bool greedy = false;
    int64_t captures = 0, replays = 0;
    explicit Impl(qwen4_exp::Model& owner) : model(owner)
#ifdef TYLOINFER_NATIVE_CUDA
        ,window(DecodeWindow::batch())
#else
        ,stream(mfq_get_current_cuda_stream())
#endif
        {}
    void clear() {
#ifdef TYLOINFER_NATIVE_CUDA
        if (requests.empty() && !ids.defined()) return;
        auto scope=window->enter(false);
        window->clear();
#else
        if (requests.empty() && !window) return;
        model.execution->weights.finish_exchange();
        window.reset();
        memory_pool.reset();
#endif
        for (auto* request : requests) {
            for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& state = *request->state_->layers[i];
                if (weights.qsa) weights.qsa->leave_graph(*state.qsa);
            }
        }
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
#ifdef TYLOINFER_NATIVE_CUDA
tyloi::StepSequence<Qwen4BatchDecodeGraph::Plan> Qwen4BatchDecodeGraph::plan(
        std::vector<Qwen4BatchRequestState*> requests, bool greedy) {
#else
std::optional<Qwen4BatchDecodeGraph::Plan> Qwen4BatchDecodeGraph::plan(
        const std::vector<Qwen4BatchRequestState*>& requests, bool greedy) {
#endif
    auto& graph = *graph_;
    auto& model = graph.model;
#ifdef TYLOINFER_NATIVE_CUDA
    std::vector<tyloi::ops::Qsa::Graph::Request> attention_rows;
    attention_rows.reserve(requests.size());
    for(auto* request:requests) {
        tyloi::ops::Qsa::Graph::Request row;
        row.position=request->position();row.maximum=model.max_position_embeddings();
        for(std::size_t i=0;i<request->state_->layers.size();++i)
            if(model.layers[i]->qsa)
                row.layers.emplace_back(model.layers[i]->qsa.get(),request->state_->layers[i]->qsa.get());
        attention_rows.push_back(std::move(row));
    }
    auto attention=std::make_unique<tyloi::ops::Qsa::Graph>(std::move(attention_rows),true);
    if (!attention->eligible()) co_return;
#else
    if (model.execution->offload || !mfq_cuda_graph_capture_supported()) return {};
#endif
    for (auto* request : requests) {
        for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            if (weights.ple && !weights.ple->graph_supported())
#ifdef TYLOINFER_NATIVE_CUDA
                co_return;
#else
                return {};
#endif
        }
    }
    // Whole-layer prefill snapshots the GPU/RAM expert maps. Adaptive decode
    // can exchange those experts, so retire the snapshots before replay just
    // as the single-request window does.
#ifdef TYLOINFER_NATIVE_CUDA
    co_yield tyloi::GraphPhase::prepare;
#else
    model.execution->weights.retire_prefill();
#endif
    const auto batch = static_cast<int64_t>(requests.size());
#ifdef TYLOINFER_NATIVE_CUDA
    attention->admit();
#else
    if(model.execution->kv) {
        int64_t required=0;
        for(auto* request:requests)required=std::max(required,request->position()+1);
        model.execution->kv->ensure(required);
    }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope=graph.window->enter();
#else
    graph.ready.record(mfq_get_current_cuda_stream());
    MfqCudaStreamGuard stream_guard(graph.stream);
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(graph.stream.stream(),graph.ready.get(),0));
#endif
    bool reuse = graph.requests == requests &&
#ifdef TYLOINFER_NATIVE_CUDA
        graph.ids.defined() &&
#else
        graph.window &&
#endif
        graph.greedy == greedy;
    if (reuse) for (int64_t row = 0; row < batch; ++row)
        reuse = reuse && requests[row]->position() + 1 <= graph.capacities[row];
    if (!reuse) {
#ifdef TYLOINFER_NATIVE_CUDA
        co_yield tyloi::GraphPhase::ready;
#endif
        graph.clear();
        // Prefill and previous widths leave asynchronously freed pool pages
        // charged to WDDM. Retire transfers and trim before preparing a graph.
#ifndef TYLOINFER_NATIVE_CUDA
        model.execution->weights.finish_exchange();
        default_context(model.execution->config.layer_placement.primary_device())->reclaim_idle_memory();
#endif
    }
    Plan plan;
#ifdef TYLOINFER_NATIVE_CUDA
    plan.rebuild = !graph.ids.defined();
#else
    plan.rebuild = !graph.window;
#endif
    if (plan.rebuild) {
#ifdef TYLOINFER_NATIVE_CUDA
        attention->layout();
        for(const auto& row:attention->rows()) {
            plan.capacities.push_back(row.capacity);
            plan.growth+=row.growth;
        }
        for(auto* request:requests) for(std::size_t i=0;i<request->state_->layers.size();++i) {
            const auto& weights=*model.layers[i];
            if(weights.gdn)plan.growth+=weights.gdn->request_cache_growth_bytes(request->state_->layers[i]->gdn);
        }
#else
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
            for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                const auto& state = *request->state_->layers[i];
                if (weights.qsa && weights.qsa->graph_capacity(*state.qsa))
                    capacity = weights.qsa->graph_capacity(*state.qsa);
            }
            plan.capacities.push_back(capacity);
            for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                const auto& state = *request->state_->layers[i];
                if (weights.qsa) plan.growth += weights.qsa->graph_growth_bytes(*state.qsa,capacity);
                if (weights.gdn) plan.growth += weights.gdn->request_cache_growth_bytes(state.gdn);
            }
        }
#endif
    }
#ifdef TYLOINFER_NATIVE_CUDA
    plan.attention=std::move(attention);
    co_yield std::move(plan);
#else
    return plan;
#endif
}

std::optional<::tyloi::Tensor> Qwen4BatchDecodeGraph::run(
        const std::vector<Qwen4BatchRequestState*>& requests,
        const ::tyloi::Tensor& ids_handle,const ::tyloi::Tensor& host_ids_handle,bool greedy,const Plan& plan) {
    const auto& ids = ids_handle.dense();
    const auto& host_ids = host_ids_handle.dense();
    auto& graph = *graph_;
    auto& model = graph.model;
    const auto batch = static_cast<int64_t>(requests.size());
#ifdef TYLOINFER_NATIVE_CUDA
    auto scope=graph.window->enter();
#else
    graph.ready.record(mfq_get_current_cuda_stream());
    MfqCudaStreamGuard stream_guard(graph.stream);
    TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(graph.stream.stream(),graph.ready.get(),0));
#endif
    graph.greedy = greedy;
    if (plan.rebuild) {
        graph.requests = requests;
        graph.capacities = plan.capacities;
        graph.ids = ::tyloi::cuda::empty_like(ids);
        graph.ple.resize(batch);
        for (int64_t row = 0; row < batch; ++row) {
            auto& state = *requests[row]->state_;
#ifdef TYLOINFER_NATIVE_CUDA
            std::size_t qsa_index=0;
#endif
            graph.positions.push_back(::tyloi::cuda::empty({1},ids.options()));
            graph.cache_positions.push_back(::tyloi::cuda::empty({1},ids.options()));
            graph.ple[row].resize(state.layers.size());
            for (std::size_t i = 0; i < state.layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& layer = *state.layers[i];
                if (weights.gdn) weights.gdn->initialize_request_cache(layer.gdn);
                if (weights.qsa)
#ifdef TYLOINFER_NATIVE_CUDA
                    plan.attention->prepare(std::size_t(row),qsa_index++,state.positions);
#else
                    weights.qsa->prepare_graph(*layer.qsa,state.positions,graph.capacities[row]);
#endif
                if (weights.ple) graph.ple[row][i] = weights.ple->graph_stage(layer.ple,1,ids.device());
            }
        }
        graph.gdn_addresses.resize(model.layers.size());
        if(batch>1 && ::tyloi::cuda::options::batch_attention_projections() && ::tyloi::cuda::options::gdn_independent_batch())
            for(std::size_t i=0;i<model.layers.size();++i) {
#ifdef TYLOINFER_NATIVE_CUDA
                std::vector<std::pair<const tyloi::ops::Gdn*,tyloi::ops::Gdn::State*>> rows;
#else
                std::vector<std::pair<const qwen4_exp::Gdn*,qwen4_exp::Gdn::State*>> rows;
#endif
                for(auto* request:requests) {
                    const auto& weights = *model.layers[i];
                    auto& state = *request->state_->layers[i];
                    if(weights.gdn) rows.emplace_back(weights.gdn.get(),&state.gdn);
                }
#ifdef TYLOINFER_NATIVE_CUDA
                if(rows.size()==requests.size()) {
                    auto addresses=tyloi::ops::Gdn::prepare_independent_batch(rows);
                    if(addresses.defined()) graph.gdn_addresses[i]=addresses.dense();
                }
#else
                if(rows.size()==requests.size())graph.gdn_addresses[i]=qwen4_exp::Gdn::prepare_independent_batch(rows);
#endif
            }
#ifndef TYLOINFER_NATIVE_CUDA
        // Cached MFE owns its graph pool and initializes it during preparation.
        // Dense fixtures need a pool owned by this decoder instead.
        if (!model.execution->weights.prepare_graph(::tyloi::Device{::tyloi::kCUDA,mfq_current_cuda_device()})) {
            graph.memory_pool = std::make_unique<Graph>();
            graph.memory_pool->prepare_memory();
            default_context(mfq_current_cuda_device())->end_graph_warmup(mfq_current_cuda_stream());
        }
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        graph.window->prepare();
#else
        graph.window = std::make_unique<DecodeWindow>(
            mfq_current_cuda_stream(),::tyloi::cuda::options::layer_timings(),true
        );
#endif
        graph.window->set_profile_geometry("batch_decode",static_cast<int>(batch));
    }
    graph.ids.copy_(ids);
    std::vector<std::tuple<qwen4_exp::Ple*,qwen4_exp::Ple::State*,qwen4_exp::NgramEmbedding::Context>> ple_contexts;
    for (int64_t row = 0; row < batch; ++row) {
        auto& state = *requests[row]->state_;
        mfq_qwen4_exp::fill_decode_positions(graph.cache_positions[row],graph.positions[row],
                                            state.cache_position,state.decode_position_delta);
        for (std::size_t i = 0; i < state.layers.size(); ++i) {
            const auto& weights = *model.layers[i];
            auto& layer = *state.layers[i];
            if (weights.ple) {
                if(!graph.window->valid())ple_contexts.emplace_back(weights.ple.get(),&layer.ple,weights.ple->session_context(layer.ple));
                weights.ple->upload_graph(layer.ple,host_ids.narrow(0,row,1),*graph.ple[row][i]);
            }
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
        for (auto* request : requests) {
            for (std::size_t i = 0; i < request->state_->layers.size(); ++i) {
                const auto& weights = *model.layers[i];
                auto& state = *request->state_->layers[i];
                if (weights.gdn) for (auto* tensor : weights.gdn->graph_state(state.gdn))
                    save(tensor);
                if (weights.ple) for (auto* tensor : weights.ple->graph_state(state.ple))
                    save(tensor);
            }
        }
#ifdef TYLOINFER_NATIVE_CUDA
        DecodeGraphBranchScope branch_scope;
#else
        DecodeGraphBranchScope branch_scope(model.execution->executor().mode().ffn.serial);
#endif
        auto body=[&] {
#ifndef TYLOINFER_NATIVE_CUDA
            auto& execution = *model.execution;
#endif
            graph.window->mark(-1,"window_begin");
            auto hidden = model.repeat_hidden(model.embed_tokens(graph.ids),model.metadata.hc_mult);
#ifdef TYLOINFER_NATIVE_CUDA
            std::optional<tyloi::ops::Residual::Result> pending_residual;
#else
            std::optional<GatedResidual::Result> pending_residual;
#endif
            const auto chain_setting=::tyloi::cuda::options::gr_chain_ffn_post();
            const bool chain_post=(!chain_setting || *chain_setting) && batch<=8;
            for (std::size_t i = 0; i < model.layers.size(); ++i) {
                graph.window->mark(static_cast<int>(i),"block_begin");
                const auto& block = *model.layers[i];
                graph.window->set_profile_layer_kind(static_cast<int>(i),block.gdn?"gdn":"qsa");
#ifndef TYLOINFER_NATIVE_CUDA
                const auto marker=[&](const char* role)->GatedResidualMarker {
                    if(!::tyloi::cuda::options::mhc_timings())return {};
                    return [&,role](const char* phase) {
                        graph.window->mark(static_cast<int>(i),(std::string(role)+phase).c_str());
                    };
                };
#endif
                if (block.ple) {
                    graph.window->mark(static_cast<int>(i),"ple_begin");
                    std::vector<::tyloi::DenseTensor> positional;
                    for (int64_t row = 0; row < batch; ++row) {
                        auto x = hidden.narrow(0,row,1);
                        auto& layer = *requests[row]->state_->layers[i];
                        graph.ple[row][i]->decode();
                        auto rows = graph.ple[row][i]->output().dense().reshape({1,1,model.hidden_size()});
                        auto ple=block.ple->forward_chunk(layer.ple,x,graph.ids.narrow(0,row,1),true,{},rows);
#ifdef TYLOINFER_NATIVE_CUDA
                        positional.push_back(tyloi::ops::add(x,ple).dense());
#else
                        positional.push_back(x+ple);
#endif
                    }
                    hidden = join_request_outputs(positional,0);
                    graph.window->mark(static_cast<int>(i),"ple_end");
                }
#ifdef TYLOINFER_NATIVE_CUDA
                auto mix = pending_residual
                    ? block.attention_gr.pre_after(hidden,pending_residual->residual,pending_residual->gate,tyloi::ops::ResidualRole::attention)
                    : block.attention_gr.pre(hidden,tyloi::ops::ResidualRole::attention);
#else
                auto mix = pending_residual
                    ? block.attention_gr.pre_after(hidden,pending_residual->residual,pending_residual->gate,marker("attention_mhc_"))
                    : block.attention_gr.pre(hidden,marker("attention_mhc_"));
#endif
                graph.window->mark(static_cast<int>(i),"attention_core_begin");
                const bool project_batch=batch>1 && ::tyloi::cuda::options::batch_attention_projections();
#ifdef TYLOINFER_NATIVE_CUDA
                std::vector<::tyloi::DenseTensor> projections;
                if(project_batch) {
                    if(block.gdn) for(const auto& value:block.gdn->project_inputs(mix.mixed)) projections.push_back(value.dense());
                    else for(const auto& value:block.qsa->project_inputs(mix.mixed)) projections.push_back(value.dense());
                }
#else
                const auto projections=project_batch ? (block.gdn ? block.gdn->project_inputs(mix.mixed)
                    : block.qsa->project_inputs(mix.mixed)) : std::vector<::tyloi::DenseTensor>{};
#endif
                std::vector<::tyloi::DenseTensor> attended;
                DenseTensor attention;
#ifdef TYLOINFER_NATIVE_CUDA
                if(project_batch && block.gdn && graph.gdn_addresses[i].defined()) {
                    auto result=block.gdn->forward_independent_batch(requests[0]->state_->layers[i]->gdn,mix.mixed,
                        std::vector<::tyloi::Tensor>(projections.begin(),projections.end()),graph.gdn_addresses[i]);
                    if(result.defined()) attention=result.dense();
                }
#else
                if(project_batch && block.gdn && graph.gdn_addresses[i].defined())
                    attention=block.gdn->forward_independent_batch(requests[0]->state_->layers[i]->gdn,mix.mixed,projections,graph.gdn_addresses[i]);
#endif
                if(!attention.defined())for (int64_t row = 0; row < batch; ++row) {
                    auto& layer = *requests[row]->state_->layers[i];
                    auto attention_input = mix.mixed
#ifdef TYLOINFER_NATIVE_CUDA
                        .dense()
#endif
                        .narrow(0,row,1);
                    std::vector<::tyloi::DenseTensor> projected;
                    for(const auto& value:projections)projected.push_back(value.narrow(0,row,1));
#ifdef TYLOINFER_NATIVE_CUDA
                    if(block.gdn) attended.push_back(block.gdn->forward_chunk(layer.gdn,attention_input,true,
                        std::vector<::tyloi::Tensor>(projected.begin(),projected.end()),!project_batch).dense());
                    else attended.push_back(block.qsa->forward_graph(*layer.qsa,attention_input,graph.positions[row],graph.cache_positions[row],
                        std::vector<::tyloi::Tensor>(projected.begin(),projected.end()),!project_batch).dense());
#else
                    attended.push_back(block.gdn ? block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),!project_batch)
                        : block.qsa->forward_graph(*layer.qsa,attention_input,graph.positions[row],graph.cache_positions[row],
                            std::move(projected),!project_batch));
#endif
                }
                if(!attention.defined())attention=join_request_outputs(attended,0);
                if(project_batch)attention=block.gdn ? block.gdn->project_output(attention)
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    : block.qsa->project_output(attention)
#ifdef TYLOINFER_NATIVE_CUDA
                        .dense()
#endif
                        ;
                graph.window->mark(static_cast<int>(i),"attention_core_end");
#ifdef TYLOINFER_NATIVE_CUDA
                auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate,tyloi::ops::ResidualRole::ffn);
#else
                auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate,marker("ffn_mhc_"));
#endif
                graph.window->mark(static_cast<int>(i),"ffn_pre_end");
                auto ffn_output=batch_ffn(block.ffn,
#ifndef TYLOINFER_NATIVE_CUDA
                    execution,
#endif
                    ffn_mix.mixed
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    );
                bool defer=false;
                if(chain_post && i+1<model.layers.size()) {
                    const auto& next=*model.layers[i+1];
#ifdef TYLOINFER_NATIVE_CUDA
                    defer=!next.ple && next.attention_gr.can_chain(ffn_mix.residual);
#else
                    defer=!next.ple && next.attention_gr.can_fuse(ffn_mix.residual);
#endif
                }
                if(defer){hidden=std::move(ffn_output);pending_residual=std::move(ffn_mix);}
                else {hidden=block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate)
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    ;pending_residual.reset();}
                graph.window->mark(static_cast<int>(i),"block_end");
            }
            auto finalized=qwen4_finalize_hidden(model, std::move(hidden),batch,1);
            graph.logits = greedy ? qwen4_next_token(model,qwen4_last_hidden(std::move(finalized)))
                .reshape({batch,1}) : qwen4_logits_from_hidden(model, std::move(finalized)).reshape({batch,-1});
            graph.window->mark(-1,"window_end");
        };
        const auto restore=[&] { for (const auto& saved : recurrent) saved.tensor->dense().copy_(saved.snapshot); };
        try {
#ifdef TYLOINFER_NATIVE_CUDA
            auto capture=graph.window->capture();
            try {
                while(capture->next()) {
                    body();
                    capture->submitted();
                    restore();
                    if(capture->warmup()) graph.logits={};
                    capture->restored();
                }
            } catch(...) {
                auto failure=capture->fail(std::current_exception());
                restore();
                std::rethrow_exception(failure);
            }
#else
            graph.window->capture(body,restore,[&] { graph.logits = {}; });
#endif
        }
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
        for (std::size_t i = 0; i < state.layers.size(); ++i)
            if (model.layers[i]->qsa) model.layers[i]->qsa->advance_graph(*state.layers[i]->qsa,1);
        state.positions=append_decode_positions(state.positions,graph.cache_positions[row],
            state.position_storage,model.max_position_embeddings());
        ++state.cache_position;
    }
    return ::tyloi::Tensor(graph.logits);
}

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::StepSequence<std::vector<::tyloi::Tensor>> qwen4_batch_prefill(qwen4_exp::Model& model,
        std::vector<Qwen4BatchRequestState*> requests,
        std::vector<::tyloi::Tensor> chunk_handles) {
#else
std::vector<::tyloi::Tensor> qwen4_batch_prefill(qwen4_exp::Model& model,
        const std::vector<Qwen4BatchRequestState*>& requests,
        const std::vector<::tyloi::Tensor>& chunk_handles) {
#endif
    std::vector<::tyloi::DenseTensor> chunks;
    chunks.reserve(chunk_handles.size());
    for (const auto& chunk : chunk_handles) chunks.push_back(chunk.dense());
    TYLOINFER_RUNTIME_CHECK(!requests.empty() && requests.size()==chunks.size(),"invalid packed prefill rows");
#ifdef TYLOINFER_NATIVE_CUDA
    const auto pipeline = model.needs_host_tokens;
    const auto measure = [](const char* name, const auto& operation) {
        return tyloi::ops::measure(name, operation);
    };
#else
    auto& execution = *model.execution;
    const auto pipeline = execution.config.cache.pipeline;
    const auto measure = [&](const char* name, const auto& operation) {
        return execution.executor().profiler().measure(name, operation);
    };
#endif
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
    if(!packed && pipeline) {
        for(std::size_t row=0;row<requests.size();++row) {
            const auto row_ids=host_ids.narrow(1,offsets[row],chunks[row].size(1));
            for(std::size_t i=0;i<requests[row]->state_->layers.size();++i) {
                const auto& weights=*model.layers[i];
                auto& state=*requests[row]->state_->layers[i];
                if(weights.ple)weights.ple->prefetch_tokens(state.ple,row_ids);
            }
        }
    }
    auto hidden=model.repeat_hidden(model.embed_tokens(ids),model.metadata.hc_mult);
    for (std::size_t i=0;i<model.layers.size();++i) {
#ifdef TYLOINFER_NATIVE_CUDA
        co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::begin,
            packed?tyloi::LayerEvent::Mode::packed:tyloi::LayerEvent::Mode::single};
#else
        NintCache decode_reuse(packed);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        {
#endif
        const auto& block=*model.layers[i];
        if (block.ple) {
            std::vector<::tyloi::DenseTensor> positional;
            for (std::size_t row=0;row<requests.size();++row) {
                auto x=hidden.narrow(1,offsets[row],chunks[row].size(1));
                auto& layer=*requests[row]->state_->layers[i];
                auto ple=packed ? block.ple->forward(layer.ple,x,chunks[row],true,0,
                    host_ids.narrow(1,offsets[row],chunks[row].size(1))) :
                    measure("prefill.ple",[&] {
                        return block.ple->forward(layer.ple,x,chunks[row],true,0,
                            host_ids.narrow(1,offsets[row],chunks[row].size(1)));
                    });
#ifdef TYLOINFER_NATIVE_CUDA
                positional.push_back(tyloi::ops::add(x,ple).dense());
#else
                positional.push_back(x+ple);
#endif
            }
            hidden=join_request_outputs(positional,1);
        }
        auto mix=packed ? block.attention_gr.pre(hidden) :
            measure("prefill.gr_attention",[&] { return block.attention_gr.pre(hidden); });
        ::tyloi::DenseTensor attention;
        if(packed) {
#ifdef TYLOINFER_NATIVE_CUDA
            std::vector<::tyloi::DenseTensor> projections;
            if(block.gdn) for(const auto& value:block.gdn->project_inputs(mix.mixed)) projections.push_back(value.dense());
            else for(const auto& value:block.qsa->project_inputs(mix.mixed)) projections.push_back(value.dense());
#else
            const auto projections=block.gdn?block.gdn->project_inputs(mix.mixed)
                                             :block.qsa->project_inputs(mix.mixed);
#endif
            std::vector<::tyloi::DenseTensor> attended;
            for (std::size_t row=0;row<requests.size();++row) {
                auto& layer=*requests[row]->state_->layers[i];
                auto attention_input=mix.mixed
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    .narrow(1,offsets[row],chunks[row].size(1));
                std::vector<::tyloi::DenseTensor> projected;
                for (const auto& value:projections)
                    projected.push_back(value.narrow(1,offsets[row],chunks[row].size(1)));
#ifdef TYLOINFER_NATIVE_CUDA
                if(block.gdn) attended.push_back(block.gdn->forward_chunk(layer.gdn,attention_input,true,
                    std::vector<::tyloi::Tensor>(projected.begin(),projected.end()),false).dense());
                else attended.push_back(block.qsa->forward(*layer.qsa,attention_input,current[row],full[row],true,nullptr,
                    std::vector<::tyloi::Tensor>(projected.begin(),projected.end()),false).dense());
#else
                attended.push_back(block.gdn?block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),false)
                    :block.qsa->forward(*layer.qsa,attention_input,current[row],full[row],true,nullptr,std::move(projected),false));
#endif
            }
            attention=join_request_outputs(attended,1);
            attention=block.gdn?block.gdn->project_output(attention)
#ifdef TYLOINFER_NATIVE_CUDA
                .dense()
#endif
                :block.qsa->project_output(attention)
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    ;
        } else {
            auto& layer=*requests.front()->state_->layers[i];
            if(block.gdn)attention=measure("prefill.gdn",[&] {
                return block.gdn->forward(layer.gdn,mix.mixed,true,0)
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    ;
            });
            else attention=measure("prefill.qsa",[&] {
                return block.qsa->forward(*layer.qsa,mix.mixed,current.front(),full.front(),true)
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    ;
            });
        }
        auto ffn_mix=packed ? block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate) :
            measure("prefill.gr_ffn_pre",[&] {
                return block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate);
            });
#ifdef TYLOINFER_NATIVE_CUDA
        auto ffn_output=packed ? tyloi::ops::ffn(block.ffn,ffn_mix.mixed).dense() :
            measure("prefill.ffn",[&] { return tyloi::ops::ffn(block.ffn,ffn_mix.mixed).dense(); });
#else
        auto ffn_output=packed ? block.ffn.forward(ffn_mix.mixed) :
            measure("prefill.ffn",[&] { return block.ffn.forward(ffn_mix.mixed); });
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        hidden=packed ? block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate).dense() :
            measure("prefill.gr_ffn_post",[&] {
                return block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate).dense();
            });
#else
        hidden=packed ? block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate) :
            measure("prefill.gr_ffn_post",[&] {
                return block.ffn_gr.post(ffn_output,ffn_mix.residual,ffn_mix.gate);
            });
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        }
        co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::end,
            packed?tyloi::LayerEvent::Mode::packed:tyloi::LayerEvent::Mode::single};
#endif
    }
    hidden=qwen4_finalize_hidden(model, std::move(hidden),1,tokens);
    std::vector<::tyloi::Tensor> result;
    for (std::size_t row=0;row<requests.size();++row) {
        result.push_back(hidden.narrow(1,offsets[row+1]-1,1));
        requests[row]->state_->positions=std::move(full[row]);
        requests[row]->state_->cache_position+=chunks[row].size(1);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    co_yield std::move(result);
#else
    return result;
#endif
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
    auto hidden = model.repeat_hidden(model.embed_tokens(ids),model.metadata.hc_mult);
    for (std::size_t i = 0; i < model.layers.size(); ++i) {
        const auto& block = *model.layers[i];
        if (block.ple) {
            std::vector<::tyloi::DenseTensor> positional;
            for (int64_t row = 0; row < batch; ++row) {
                auto& layer = *requests[row]->state_->layers[i];
                auto x = hidden.narrow(0,row,1);
                auto ple=block.ple->forward(layer.ple,x,ids.narrow(0,row,1),true,0,host_ids.narrow(0,row,1));
#ifdef TYLOINFER_NATIVE_CUDA
                positional.push_back(tyloi::ops::add(x,ple).dense());
#else
                positional.push_back(x+ple);
#endif
            }
            hidden = join_request_outputs(positional,0);
        }
        auto mix = block.attention_gr.pre(hidden);
        const bool project_batch=batch>1 && block.gdn && ::tyloi::cuda::options::batch_attention_projections();
#ifdef TYLOINFER_NATIVE_CUDA
        std::vector<::tyloi::DenseTensor> projections;
        if(project_batch) for(const auto& value:block.gdn->project_inputs(mix.mixed)) projections.push_back(value.dense());
#else
        const auto projections=project_batch ? block.gdn->project_inputs(mix.mixed) : std::vector<::tyloi::DenseTensor>{};
#endif
        std::vector<::tyloi::DenseTensor> attended;
        for (int64_t row = 0; row < batch; ++row) {
            auto& layer = *requests[row]->state_->layers[i];
            auto attention_input = mix.mixed
#ifdef TYLOINFER_NATIVE_CUDA
                .dense()
#endif
                .narrow(0,row,1);
            std::vector<::tyloi::DenseTensor> projected;
            for(const auto& value:projections)projected.push_back(value.narrow(0,row,1));
#ifdef TYLOINFER_NATIVE_CUDA
            if(block.gdn) attended.push_back(block.gdn->forward_chunk(layer.gdn,attention_input,true,
                std::vector<::tyloi::Tensor>(projected.begin(),projected.end()),!project_batch).dense());
            else attended.push_back(block.qsa->forward(*layer.qsa,attention_input,current[row],full[row],true).dense());
#else
            attended.push_back(block.gdn ? block.gdn->forward_chunk(layer.gdn,attention_input,true,std::move(projected),!project_batch)
                : block.qsa->forward(*layer.qsa,attention_input,current[row],full[row],true));
#endif
        }
        auto attention=join_request_outputs(attended,0);
        if(project_batch)attention=block.gdn->project_output(attention)
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
            ;
        auto ffn_mix = block.ffn_gr.pre_after(std::move(attention),mix.residual,mix.gate);
        // Tiles share the expert transfer cache across all eligible rows.
        hidden = block.ffn_gr.post(batch_ffn(block.ffn,
#ifndef TYLOINFER_NATIVE_CUDA
            *model.execution,
#endif
            ffn_mix.mixed
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
            ),ffn_mix.residual,ffn_mix.gate)
#ifdef TYLOINFER_NATIVE_CUDA
            .dense()
#endif
            ;
    }
    auto logits = qwen4_logits_from_hidden(model, qwen4_finalize_hidden(model, std::move(hidden),batch,1)).reshape({batch,-1});
    for (int64_t row = 0; row < batch; ++row) {
        requests[row]->state_->positions = std::move(full[row]);
        ++requests[row]->state_->cache_position;
    }
    return logits;
}

qwen4_exp::State::State(const Model& model) : model_(model) {
    for (const auto& layer : model_.layers)
        layers.push_back(std::make_unique<LayerState>(*layer));
}

qwen4_exp::State::~State() = default;

#ifdef TYLOINFER_NATIVE_CUDA
void qwen4_exp::State::retire_graphs() noexcept { graph_registry.reset(); }

void qwen4_exp::State::release_graphs() {
    std::exception_ptr failure;
    try { if(graph_registry) graph_registry->restore_legacy(); }
    catch (...) { if(!failure) failure=std::current_exception(); }
    graph_registry.reset();
    for(auto& layer:layers) layer->retain_graph_storage=false;
    if(failure) std::rethrow_exception(failure);
}
#endif

void qwen4_exp::State::reset(int64_t new_batch) {
    cache_pos = decode_position_delta = 0;
    speculative_start = -1;
    speculative_confirmed = 0;
    speculative_suffix_forward = false;
    speculative_prefix_retention = false;
#ifndef TYLOINFER_NATIVE_CUDA
    if(model_.execution)model_.execution->weights.finish_exchange();
#endif
    const bool retain=reuse_decode_graphs && new_batch==1 && graph_registry && graph_registry->reusable();
    for(auto& layer:layers)layer->retain_graph_storage=retain;
    if(!retain)graph_registry.reset();
    else for(auto& [key,cell]:graph_registry->cells)cell->used_kinds=0;
    positions = {};
    batch = new_batch;
    for(std::size_t i=0;i<layers.size();++i)layers[i]->reset(*model_.layers[i]);
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
#ifdef TYLOINFER_NATIVE_CUDA
namespace {
bool layer_prefill(const qwen4_exp::Model& model, const qwen4_exp::State& state,
                   const tyloi::models::PrefillOptions& options, int64_t tokens) {
    const auto enabled=::tyloi::cuda::options::moe_prefill_layer_major();
    const auto whole=::tyloi::cuda::options::moe_prefill_layer();
    if((enabled && *enabled==0) || (whole && *whole==0) || (!options.routed && !whole) ||
       tokens<=128 || tokens>16384 || state.speculative_start>=0 ||
       (!options.routed && !model.needs_host_tokens) || !options.coordinated || model.parallel) return false;
    for(const auto& block:model.layers)
        if(block->cpu_offloaded || block->cuda_device!=model.primary_device)return false;
    return true;
}
}
#else
bool qwen4_layer_major_prefill_eligible(const qwen4_exp::Model& model, const qwen4_exp::State& state, int64_t tokens) {
    const auto enabled=::tyloi::cuda::options::moe_prefill_layer_major();
    const auto whole=::tyloi::cuda::options::moe_prefill_layer();
    const bool tq=model.execution && model.execution->weights.decode()==
        ::tyloi::weights::Cache::Decode::routed;
    // Preserve explicit overrides and the existing expert-cache layout selection.
    if((enabled && *enabled==0) || (whole && *whole==0) || (!tq && !whole) || tokens<=128 ||
       tokens>16384 || state.speculative_start>=0 || !model.execution ||
       (!tq && !model.execution->config.cache.pipeline) ||
       model.execution->weights.transfer()!=::tyloi::weights::Cache::Transfer::coordinated ||
       model.execution->config.dense_cpu_layer_count || model.execution->config.tensor_parallel.enabled() ||
       model.execution->config.expert_parallel.enabled())return false;
    const auto primary=model.execution->config.layer_placement.primary_device();
    for(const auto& block:model.layers)
        if(block->cpu_offloaded || block->cuda_device!=primary)return false;
    return true;
}
#endif

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::models::PrefillPlan qwen4_exp::Model::prefill_plan(const tyloi::models::State& value,
                                            const tyloi::models::PrefillOptions& options) const {
    const auto* request = dynamic_cast<const qwen4_exp::State*>(&value);
    tyloi::models::require_model(request && request->belongs_to(*this), "Qwen4 state belongs to a different model");
    auto span=std::min<int64_t>(options.compute,8192);
    if(options.compute>8 && options.reused==0 && (options.stable==0 || options.stable==static_cast<std::size_t>(options.tokens)) &&
       !options.transformed && layer_prefill(*this,*request,options,options.tokens)) span=options.tokens;
    return {span,!options.transformed && layer_prefill(*this,*request,options,std::min<int64_t>(options.tokens,8192))};
}
#else
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
#endif

#ifndef TYLOINFER_NATIVE_CUDA
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
#else
tyloi::StepSequence<tyloi::models::Prefill> qwen4_exp::Model::prefill(
        tyloi::models::Input input, tyloi::models::State& state_value,
        const tyloi::models::ForwardOptions& options, tyloi::models::PrefillOptions prefill) const {
    auto* target = dynamic_cast<qwen4_exp::State*>(&state_value);
    tyloi::models::require_model(target && target->belongs_to(*this),
        "Qwen4 state belongs to a different model");
    auto& state = *target;
    const auto& model = *this;
    if (input.embeddings || input.positions || input.cache_positions || input.lengths || input.image_mask ||
        input.confirmed || input.suffix || !input.ids || input.ids.dim()!=2 || input.ids.size(0)!=1) co_return;
    const auto count=input.ids.size(1);
    if (state.reuse_decode_graphs && state.graph_registry && count>0 && count<=128 &&
        (input.raw_hidden || prefill.split)) {
        std::vector<tyloi::Tensor> hidden_rows, raw_rows;
        tyloi::Tensor last;
        for(std::int64_t offset=0;offset<count;) {
            const auto rows=state.cache_pos==0?std::int64_t(1):std::min<std::int64_t>(4,count-offset);
            auto ids=tyloi::ops::slice(input.ids,1,offset,offset+rows);
            tyloi::Tensor raw;
            tyloi::models::Input window;
            window.ids=ids; window.confirmed=rows>1?1:0;
            window.raw_hidden=input.raw_hidden?&raw:nullptr;
            tyloi::models::DecodeOptions decode_options;
            decode_options.routed=prefill.routed;
            auto job=decode(window,state,options,decode_options);
            std::optional<tyloi::models::Decoded> decoded;
            while(auto step=job.next()) {
                if(step.layer)throw std::logic_error("unexpected prefill decode layer event");
                if(step.graph) co_yield *step.graph;
                if(step.value) decoded=std::move(*step.value);
            }
            tyloi::Tensor hidden;
            if(decoded) hidden=std::move(decoded->hidden);
            else {
                if(state.position()==0) co_yield tyloi::GraphPhase::ready;
                hidden=forward(std::move(window),state,options);
            }
            if(input.raw_hidden) {
                hidden_rows.push_back(tyloi::ops::clone(hidden));
                raw_rows.push_back(std::move(raw));
            } else last=std::move(hidden);
            if(rows>1)state.commit_speculative();
            offset+=rows;
            co_yield tyloi::StepState::advanced;
        }
        tyloi::Tensor raw;
        if(input.raw_hidden) {
            last=hidden_rows.size()==1?std::move(hidden_rows.front()):tyloi::ops::cat(hidden_rows,1);
            raw=raw_rows.size()==1?std::move(raw_rows.front()):tyloi::ops::cat(raw_rows,1);
            if(input.host_raw) raw=tyloi::ops::copy(raw,tyloi::Device(tyloi::kCPU));
            *input.raw_hidden=raw;
        }
        co_yield tyloi::models::Prefill{tyloi::models::Prefill::Phase::computed};
        co_yield tyloi::models::Prefill{tyloi::models::Prefill::Phase::complete,0,0,std::move(last),std::move(raw)};
        co_return;
    }
    if (!layer_prefill(model,state,prefill,count)) co_return;
    auto compute_chunk=prefill.compute;
    auto ids_handle = std::move(input.ids);
    auto* raw_hidden = input.raw_hidden;
    const bool raw_hidden_host = input.host_raw;
    const bool last_only = true;
#endif
    using DenseTensor=::tyloi::DenseTensor;
    auto ids = ids_handle.dense();
    TYLOINFER_RUNTIME_CHECK(ids.dim()==2 && ids.size(0)==1 &&
#ifdef TYLOINFER_NATIVE_CUDA
        layer_prefill(model,state,prefill,ids.size(1)),
#else
        qwen4_layer_major_prefill_eligible(model,state,ids.size(1)),
#endif
        "layer-major prefill requires a bounded single-device Qwen4 chunk and whole-layer cache");
    TYLOINFER_RUNTIME_CHECK(compute_chunk>8,"layer-major compute chunk must exceed decode width");
#ifndef TYLOINFER_NATIVE_CUDA
    auto scope=model.execution_scope();
#endif
    ids=model.device_ids(std::move(ids));
    const auto tokens=ids.size(1);
    const auto cache_start=state.cache_pos;
    const auto* profile_once=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_PREFILL_ONCE");
    static std::atomic<uint64_t> profile_requests{0};
    const bool profile_enabled=profile_once && profile_once[0]=='1';
    const auto* profile_skip=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_PREFILL_SKIP");
    const auto skipped=profile_skip?std::max<long long>(0,std::atoll(profile_skip)):0;
    const auto profile_request=profile_enabled?profile_requests.fetch_add(1):0;
    const bool collect=profile_enabled && profile_request==uint64_t(skipped);
    if(profile_enabled)std::cerr<<"prefill_profile_request index="<<profile_request
        <<" tokens="<<tokens<<" collect="<<int(collect)<<'\n';
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::ops::ProfileCapture capture(collect,"prefill_regression_once");
    const auto profile_measure=[&](const char* name, const auto& operation) {
        return tyloi::ops::measure(name, operation);
    };
    const auto resume_profile=[&] { capture.resume(); };
#else
    auto& profiler=model.execution->executor().profiler();
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
    const auto profile_measure=[&](const char* name, const auto& operation) {
        return profiler.measure(name, operation);
    };
    const auto resume_profile=[&] { profiler.enabled=restore.enabled; };
#endif
    compute_chunk=std::min(compute_chunk,tokens);
#ifdef TYLOINFER_NATIVE_CUDA
    const auto row_bytes=std::size_t(model.config.hidden)*model.config.streams*sizeof(float)*8;
    const auto fixed=(std::size_t(1)<<30)+
        std::size_t(tokens)*model.config.hidden*model.config.streams*sizeof(float)*3;
    co_yield tyloi::Workspace{fixed,nullptr,tyloi::Workspace::Tile{tokens,row_bytes,&compute_chunk}};
#else
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
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    if(cache_start==0) co_yield tyloi::GraphPhase::ready;
#endif
#ifndef TYLOINFER_NATIVE_CUDA
    tyloi::cuda::ResidentAllocationScope persistent_preamble;
#endif
    qwen4_validate_forward(model,tokens,cache_start,false);
    if(cache_start==0) state.reset(1);
    state.begin_forward();
#ifdef TYLOINFER_NATIVE_CUDA
    auto positions=tyloi::ops::position_span(cache_start,tokens,state.decode_position_delta,state.positions,ids.options());
    auto prepared=positions[0].dense();
    auto prepared_full=positions[1].dense();
#else
    auto current=::tyloi::cuda::arange(cache_start,cache_start+tokens,ids.options());
    if(state.decode_position_delta)current=current+state.decode_position_delta;
    auto prepared=std::move(current);
    if ((prepared.dim() == 2 || prepared.dim() == 3) && prepared.size(0) == 4)
        prepared = prepared.narrow(0,1,3);
    auto prepared_full=state.positions.defined()?::tyloi::cuda::cat({state.positions,prepared},-1):prepared;
#endif
    tyloi::models::require_model((prepared.dim() == 1 || (prepared.dim() == 2 && prepared.size(0) == 3) ||
        (prepared.dim() == 3 && prepared.size(0) == 3 && prepared.size(1) == 1)) && prepared.size(-1) == tokens,
        "Qwen4 positions require [T], [3,T], [4,T], [3,B,T] or [4,B,T]");
    auto cpu_ids=ids.to(::tyloi::kCPU,::tyloi::kInt64).contiguous();
#ifndef TYLOINFER_NATIVE_CUDA
    persistent_preamble.finish();
#endif
    DenseTensor hidden=::tyloi::cuda::empty({1,tokens,model.metadata.hidden_size*model.metadata.hc_mult},
                           ids.options().dtype(::tyloi::kFloat16));
    for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
#ifdef TYLOINFER_NATIVE_CUDA
        co_yield tyloi::StepState::advanced;
#else
        if(cancelled && cancelled())return {};
#endif
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
#ifdef TYLOINFER_NATIVE_CUDA
    const auto reference_rows=batch_experts?std::min<int64_t>(1024,compute_chunk):0;
#else
    const qwen4_exp::QwenPrefillReferenceRowsScope reference_rows(batch_experts?std::min<int64_t>(1024,compute_chunk):0);
#endif
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
    for(auto& block:model.layers) {
#ifdef TYLOINFER_NATIVE_CUDA
        co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::begin};
#else
        const auto decode_reuse_option=::tyloi::cuda::options::nint_prefill_decode_reuse();
        NintCache decode_reuse(!decode_reuse_option || *decode_reuse_option);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        {
#endif
        auto& layer_state=*state.layers[layer];
        DenseTensor next_hidden=hidden;
        DenseTensor ffn_inputs,ffn_gate;
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::ops::Residual::Result attention_result;
#else
        GatedResidual::Result attention_result;
#endif
        DenseTensor attention_output;
        if(bulk_gr) {
            auto& qwen_block=*block;
            DenseTensor source=hidden;
            if(qwen_block.ple) {
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
#ifdef TYLOINFER_NATIVE_CUDA
                    co_yield tyloi::StepState::advanced;
#else
                    if(cancelled && cancelled())return {};
#endif
                    const auto count=std::min(compute_chunk,tokens-begin);
                    auto chunk_ids=ids.narrow(1,begin,count);
                    auto host_ids=cpu_ids.narrow(1,begin,count);
                    qwen_block.ple->prefetch_tokens(layer_state.ple, host_ids);
                    auto input=qwen4_exp::prefill_ple(qwen_block,layer_state,
                        hidden.narrow(1,begin,count),chunk_ids,host_ids);
                    if(begin==0)source=::tyloi::cuda::empty(hidden.sizes(),input.options());
                    source.narrow(1,begin,count).copy_(input);
                }
            }
            attention_result=profile_measure("prefill.gr_attention",[&] {
                return qwen_block.attention_gr.pre(source);
            });
            if(bulk_gdn && qwen_block.gdn) {
#ifdef TYLOINFER_NATIVE_CUDA
                co_yield tyloi::StepState::advanced;
#else
                if(cancelled && cancelled())return {};
#endif
#ifdef TYLOINFER_NATIVE_CUDA
                const tyloi::ops::Gdn::ProjectionScope projection_rows(compute_chunk);
#else
                const qwen4_exp::QwenGdnPrefillProjectionRowsScope projection_rows(compute_chunk);
#endif
                attention_output=profile_measure("prefill.gdn",[&] {
                    return qwen_block.gdn->forward(layer_state.gdn,attention_result.mixed,true,0)
#ifdef TYLOINFER_NATIVE_CUDA
                        .dense()
#endif
                        ;
                });
            }
        }
        for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
#ifdef TYLOINFER_NATIVE_CUDA
            co_yield tyloi::StepState::advanced;
#else
            if(cancelled && cancelled())return {};
#endif
            if(bulk_gdn && block->gdn)continue;
            const auto count=std::min(compute_chunk,tokens-begin);
            resume_profile();
            auto chunk_ids=ids.narrow(1,begin,count);
            auto host_ids=cpu_ids.narrow(1,begin,count);
            auto positions=prepared.narrow(-1,begin,count);
            auto full_positions=prepared_full.narrow(-1,0,cache_start+begin+count);
            if (block->ple)
                block->ple->prefetch_tokens(layer_state.ple, host_ids);
            auto input=hidden.narrow(1,begin,count);
            DenseTensor output;
            if(bulk_gr) {
                auto& qwen_block=*block;
                auto attention_chunk=qwen4_exp::prefill_attention_core(qwen_block,layer_state,
                    attention_result.mixed
#ifdef TYLOINFER_NATIVE_CUDA
                        .dense()
#endif
                        .narrow(1,begin,count),positions,full_positions);
                if(!attention_output.defined()) {
                    auto shape=attention_chunk.sizes().vec();shape[1]=tokens;
                    attention_output=::tyloi::cuda::empty(shape,attention_chunk.options());
                }
                attention_output.narrow(1,begin,count).copy_(attention_chunk);
                continue;
            } else if(batch_experts) {
                auto& qwen_block=*block;
                auto mix=qwen4_exp::prefill_attention(qwen_block,layer_state,
                    input,chunk_ids,host_ids,positions,full_positions);
                if(!ffn_inputs.defined()) {
                    ffn_inputs=::tyloi::cuda::empty({1,tokens,model.hidden_size()},mix.mixed.options().dtype(::tyloi::kFloat16));
                    ffn_gate=::tyloi::cuda::empty({1,tokens,mix.gate.size(-1)},mix.gate.options());
                }
                // The MoE casts the router/FFN source to half at this boundary.
#ifdef TYLOINFER_NATIVE_CUDA
                ffn_inputs.narrow(1,begin,count).copy_(mix.mixed.dense().to(::tyloi::kFloat16));
                ffn_gate.narrow(1,begin,count).copy_(mix.gate.dense());
                output=mix.residual.dense();
#else
                ffn_inputs.narrow(1,begin,count).copy_(mix.mixed.to(::tyloi::kFloat16));
                ffn_gate.narrow(1,begin,count).copy_(mix.gate);
                output=std::move(mix.residual);
#endif
            } else output=qwen4_exp::forward(*block,layer_state,
                input,chunk_ids,positions,full_positions,0,host_ids);
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
        if(tokens>9216)
#ifdef TYLOINFER_NATIVE_CUDA
            co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::clear};
#else
            decode_reuse.clear();
#endif
        if(bulk_gr) {
            auto& qwen_block=*block;
            auto mix=profile_measure("prefill.gr_ffn_pre",[&] {
                return qwen_block.ffn_gr.pre_after(attention_output,attention_result.residual,attention_result.gate);
            });
#ifdef TYLOINFER_NATIVE_CUDA
            ffn_inputs=mix.mixed.dense().to(::tyloi::kFloat16);ffn_gate=mix.gate.dense();
            next_hidden=mix.residual.dense();
#else
            ffn_inputs=mix.mixed.to(::tyloi::kFloat16);ffn_gate=std::move(mix.gate);
            next_hidden=std::move(mix.residual);
#endif
            attention_result={};attention_output={};
        }
        if(tokens>9216)
#ifdef TYLOINFER_NATIVE_CUDA
            co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::clear};
#else
            decode_reuse.clear();
#endif
        // The new residual owns the next layer's values. Prior hidden readers
        // have already been submitted before this stream-ordered release.
        if(tokens>9216)hidden={};
        if(batch_experts) {
#ifdef TYLOINFER_NATIVE_CUDA
            co_yield tyloi::StepState::advanced;
#else
            if(cancelled && cancelled())return {};
#endif
            auto& qwen_block=*block;
            resume_profile();
#ifdef TYLOINFER_NATIVE_CUDA
            DenseTensor ffn_output;
            if(expert_batch_count(tokens)==tokens) {
                ffn_output=profile_measure("prefill.ffn",[&] { return tyloi::ops::ffn(qwen_block.ffn,ffn_inputs,{}, {},reference_rows).dense(); });
            } else {
                for(int64_t begin=0;begin<tokens;) {
                    co_yield tyloi::StepState::advanced;
                    auto count=expert_batch_count(tokens-begin);
                    const auto tail=tokens-begin-count;
                    if(tail>0 && tail<=8)count-=16-tail;
                    auto part=profile_measure("prefill.ffn",[&] {
                        return tyloi::ops::ffn(qwen_block.ffn,ffn_inputs.narrow(1,begin,count),{}, {},reference_rows).dense();
                    });
                    if(!ffn_output.defined()) {
                        auto shape=part.sizes().vec();shape[1]=tokens;
                        ffn_output=::tyloi::cuda::empty(shape,part.options());
                    }
                    ffn_output.narrow(1,begin,count).copy_(part);
                    begin+=count;
                }
            }
#else
            auto ffn_output=profile_measure("prefill.ffn",[&] {
                if(expert_batch_count(tokens)==tokens)return qwen_block.ffn.forward(ffn_inputs);
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
                    auto part=qwen_block.ffn.forward(ffn_inputs.narrow(1,begin,count));
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
#endif
            if(audit_batch) {
                auto expected=::tyloi::cuda::empty(ffn_output.sizes(),ffn_output.options());
                for(int64_t begin=0;begin<tokens;begin+=compute_chunk) {
                    const auto count=std::min(compute_chunk,tokens-begin);
                    expected.narrow(1,begin,count).copy_(
#ifdef TYLOINFER_NATIVE_CUDA
                        tyloi::ops::ffn(qwen_block.ffn,ffn_inputs.narrow(1,begin,count),{}, {},reference_rows).dense());
#else
                        qwen_block.ffn.forward(ffn_inputs.narrow(1,begin,count)));
#endif
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
                    next_hidden.narrow(1,begin,count),ffn_gate.narrow(1,begin,count))
#ifdef TYLOINFER_NATIVE_CUDA
                    .dense()
#endif
                    ;
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
        if(trace && (layer%4==0 || layer==int(model.layers.size()))) {
#ifdef TYLOINFER_NATIVE_CUDA
            tyloi::models::Prefill report;
            report.phase=tyloi::models::Prefill::Phase::progress;
            report.progress=tyloi::models::Prefill::Progress{ids.device(),tokens,compute_chunk,layer,
                static_cast<int>(model.layers.size()),started};
            co_yield std::move(report);
#else
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_get_current_cuda_stream()));
            std::cout<<"prefill_layer_major_progress tokens="<<tokens<<" chunk="<<compute_chunk
                <<" completed_layers="<<layer<<" total_layers="<<model.layers.size()
                <<" elapsed_ms="<<std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-started).count()<<std::endl;
#endif
        }
#ifdef TYLOINFER_NATIVE_CUDA
        }
        co_yield tyloi::LayerEvent{tyloi::LayerEvent::Phase::end};
#endif
    }
    resume_profile();
    if(last_only && !raw_hidden) {
        const auto begin=((tokens-1)/compute_chunk)*compute_chunk;
        DenseTensor result;
        {
            auto finalized=qwen4_finalize_hidden(model, hidden.narrow(1,begin,tokens-begin),1,tokens-begin);
            result=qwen4_last_hidden(std::move(finalized)).unsqueeze(1);
        }
#ifndef TYLOINFER_NATIVE_CUDA
        if(computation_finished)computation_finished();
#else
        co_yield tyloi::models::Prefill{tyloi::models::Prefill::Phase::computed};
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::Tensor final_output(std::move(result));
        hidden={};
        co_yield tyloi::Workspace{0,&final_output};
#else
        if(borrowed_workspace) {
            workspace_loan.suspend();
            result=result.clone();
            hidden={};
            workspace_loan.finish();
        }
#endif
        state.finish_forward(prepared_full,1,tokens);
        state.cache_pos=cache_start+tokens;
#ifdef TYLOINFER_NATIVE_CUDA
        co_yield tyloi::models::Prefill{tyloi::models::Prefill::Phase::complete, 0, 0, std::move(final_output), {}};
        co_return;
#else
        return result;
#endif
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
#ifndef TYLOINFER_NATIVE_CUDA
    if(computation_finished)computation_finished();
#else
    co_yield tyloi::models::Prefill{tyloi::models::Prefill::Phase::computed};
#endif
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::Tensor final_output(std::move(finalized));
    tyloi::Tensor raw_output;
    if(raw_hidden) raw_output=std::move(hidden);
    else hidden={};
    co_yield tyloi::Workspace{0,&final_output,std::nullopt,
        raw_hidden && !raw_hidden_host ? &raw_output : nullptr};
#else
    if(borrowed_workspace) {
        // The final mixer also needs sequence-sized scratch. Finish it inside
        // the loan, then retain only the outputs consumed by MTP/the caller.
        workspace_loan.suspend();
        if(raw_hidden) {if(!raw_hidden_host)hidden=hidden.clone();}else hidden={};
        finalized=finalized.clone();
        workspace_loan.finish();
    }
#endif
    state.finish_forward(prepared_full,1,tokens);
    state.cache_pos=cache_start+tokens;
#ifdef TYLOINFER_NATIVE_CUDA
    if(raw_hidden)*raw_hidden=raw_output;
    co_yield tyloi::models::Prefill{tyloi::models::Prefill::Phase::complete, 0, 0,
        std::move(final_output), std::move(raw_output)};
#else
    if(raw_hidden)*raw_hidden=hidden;
    return finalized;
#endif
}

namespace qwen4_exp {

tyloi::models::Info Model::info() const {
    auto result=tyloi::models::Info{metadata.vocab_size, config.maximum, "qwen4_exp",
            std::size_t(metadata.hidden_size) * metadata.hc_mult * sizeof(float) * 8, true, true};
#ifdef TYLOINFER_NATIVE_CUDA
    result.decode_graph=std::none_of(layers.begin(),layers.end(),
        [](const auto& layer) { return layer->cpu_offloaded; });
    result.graph=tyloi::models::GraphOwner::model;
    result.row_prefetch=true;
    result.streamed_kv=true;
#endif
    return result;
}

MfqModelCapabilities Model::capabilities() const {
#ifdef TYLOINFER_NATIVE_CUDA
    const bool vision = components && components->grid_vision.has_value();
    const auto state = cuda_component_state(graph, plan, vision, mtp() != nullptr);
    const auto caps = cuda_runtime_capabilities(graph, plan, state, vision, false);
#else
    const auto state = cuda_component_state(graph, plan, false, false);
    const auto caps = cuda_runtime_capabilities(graph, plan, state, false, false);
#endif
    return {graph.architecture, caps.text, caps.image_input, caps.video_input,
        caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp,
        "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
#ifdef TYLOINFER_NATIVE_CUDA
    const auto state = cuda_component_state(graph, plan,
        components && components->grid_vision.has_value(), mtp() != nullptr);
    const auto quantization = cuda_kv_quantization();
    return {{"kv_memory_supported", 1.0},
        {"kv_quantization_supported", 1.0},
        {"kv_quantization_bits", double(quantization.bits)},
#else
    const auto state = cuda_component_state(graph, plan, false, false);
    return {{"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 0.0},
        {"kv_quantization_bits", 0.0},
#endif
        {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)},
        {"vision_available", double(state.vision_available)},
        {"mtp_declared", double(state.mtp_declared)},
        {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", double(state.mtp_available)}};
}

tyloi::models::Mtp* Model::mtp() const noexcept {
#ifdef TYLOINFER_NATIVE_CUDA
    return components ? components->mtp.get() : nullptr;
#else
    return nullptr;
#endif
}

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    auto state=std::make_unique<State>(*this);
#ifdef TYLOINFER_NATIVE_CUDA
    if (auto* predictor=mtp()) state->predictor=predictor->make_state();
#endif
    return state;
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
    const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media, tyloi::models::State& state) const {
#ifdef TYLOINFER_NATIVE_CUDA
    auto* target=dynamic_cast<State*>(&state);
    tyloi::models::require_model(target && target->belongs_to(*this), "Qwen4 state belongs to another model");
    if (components)
        return components->prepare(*this, target->images, primary_device, tokens, media);
#endif
    throw std::invalid_argument("model has no multimodal component");
}

::tyloi::Tensor Model::embedding(const ::tyloi::Tensor& ids, int) const {
#ifndef TYLOINFER_NATIVE_CUDA
    auto scope = execution_scope();
#endif
    return std::make_shared<::tyloi::DenseTensor>(embed_tokens(device_ids(ids.dense())));
}

::tyloi::Tensor Model::normalize(const ::tyloi::Tensor& hidden, std::int64_t, std::int64_t) const {
    return hidden;
}

::tyloi::Tensor Model::softcap(const ::tyloi::Tensor& logits) const {
    return logits;
}

::tyloi::Tensor Model::logits(const ::tyloi::Tensor& hidden, bool) const {
#ifdef TYLOINFER_NATIVE_CUDA
    return tyloi::ops::measure("model.lm_head", hidden.device(), [&] {
        return tyloi::ops::project(hidden, lm_head);
    });
#else
    return execution->executor().profiler().measure("model.lm_head", [&] {
        return execution->executor().projector(execution->executor().mode().ffn,
            model_parallel_primary_device(execution->config)).run(lm_head, hidden.dense());
    });
#endif
}

::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& state_value,
                                    const tyloi::models::ForwardOptions&) const {
#ifdef TYLOINFER_NATIVE_CUDA
    auto* target = dynamic_cast<State*>(&state_value);
    tyloi::models::require_model(target && target->belongs_to(*this),
                                 "Qwen4 state belongs to a different model");
    auto& state = *target;
    if (input.suffix) {
        tyloi::models::require_model(state.speculative_start >= 0 && state.speculative_confirmed == 0 &&
                                         !state.speculative_suffix_forward,
                                     "speculative suffix is not active");
        state.speculative_suffix_forward = true;
        try {
            auto result = qwen4_forward(*this, state, std::move(input));
            state.speculative_suffix_forward = false;
            return result;
        } catch (...) {
            state.speculative_suffix_forward = false;
            throw;
        }
    }
    return qwen4_forward(*this, state, std::move(input));
#else
    return std::make_shared<::tyloi::DenseTensor>(qwen4_forward(*this, static_cast<State&>(state_value), std::move(input)));
#endif
}

#ifndef TYLOINFER_NATIVE_CUDA
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
#endif

void State::rollback_speculative(std::int64_t accepted) {
    tyloi::models::require_model(speculative_start >= 0 && accepted >= 0,
                                 "no speculative transaction");
    const auto keep = speculative_start + speculative_confirmed + accepted;
    speculative_prefix_retention=false;
    for (std::size_t i=0;i<model_.layers.size();++i)
        layers[i]->rollback(*model_.layers[i],keep);
#ifdef TYLOINFER_NATIVE_CUDA
    if (positions.defined()) positions=tyloi::ops::slice(positions,-1,0,keep);
#else
    if (positions.defined()) positions=positions.narrow(-1,0,keep);
#endif
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

#ifdef TYLOINFER_NATIVE_CUDA
std::unique_ptr<tyloi::models::CausalModel> load(std::shared_ptr<const tyloi::ModelSource> source,
        std::string_view definition, const tyloi::weights::Loader& weights,
        LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options) {
    if (!options.cpu_offload_layers.empty())
        throw std::invalid_argument("Qwen4 does not support --cpu-offload-layers");
    auto model=std::make_unique<Model>();
    read(*model,std::move(source),definition,options.context);
    load(*model,weights,placement,options);
    model->components=std::make_unique<Components>(load_components(*model,weights,true,definition,options.yarn));
    return model;
}
#endif

} // namespace qwen4_exp
} // namespace tyloi::cuda
