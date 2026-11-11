#include "env.h"
#include "engine/backends/cuda/executor/options.h"
#pragma once
#include "resident_allocations.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/models/qwen4_exp/causal_lm.h"
#include "engine/models/qwen4_exp/ngram.h"
#include "engine/models/common/attention_ops.h"
#include "engine/ops.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/kernels/ple_fused.h"
#include "engine/weights/nint_row_pipeline.h"
#include "engine/weights/row_stage.h"
#include "engine/state/kv_offload.h"
#include "engine/state/kv_memory.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/qwen4_exp/snapshot.h"
#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <unordered_map>

namespace tyloi::cuda::qwen4_exp {
using attention_ops::DenseTensor;
using attention_ops::Linear;
using attention_ops::Embedding;
using attention_ops::rms_norm;
using attention_ops::SequenceCache;
using attention_ops::select_pooled_blocks;

struct GdnWeights {
    std::vector<::tyloi::Tensor> input_projection;
    ::tyloi::Tensor output;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor conv, dt_bias, a_log, norm;
#else
    DenseTensor conv, dt_bias, a_log, norm;
#endif
    std::size_t qkv_count = 1;
};

class Gdn {
  public:
    struct State {
        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        ::tyloi::Tensor conv{DenseTensor{}}, recurrent{DenseTensor{}}, core_workspace{DenseTensor{}};
        DenseTensor saved_conv, saved_state;
        std::vector<DenseTensor> saved_prefix_conv, saved_prefix_state;
        int64_t saved_prefix_position = -1;
        bool resident = false;
    };
    Gdn(std::shared_ptr<const GdnWeights> weights, int64_t key_heads, int64_t value_heads, int64_t width, int64_t kernel,
        double eps, bool silu_gate, bool fused_decode = true, bool transposed_state = true,
        bool fused_output = true, bool grouped_projection = true, bool fused_preparation = true,
        bool fused_core = true)
        : w_(std::move(weights)), nk_(key_heads), nv_(value_heads), d_(width), kernel_(kernel),
          eps_(eps), silu_gate_(silu_gate), fused_decode_(fused_decode),
          transposed_state_(fused_decode && transposed_state && (width==32 || width==64 || width==128)),
          fused_output_(fused_output), grouped_projection_(grouped_projection), fused_preparation_(fused_preparation),
          fused_core_(fused_core) {
        TYLOINFER_RUNTIME_CHECK(nk_ > 0 && nv_ > 0 && nv_ % nk_ == 0 && d_ > 0 && kernel_ > 1 && eps_ > 0,
                          "invalid Qwen4 GDN head/convolution geometry");
    }
    void reset(State& request, bool retain_storage=false) {
        if(retain_storage) {
            if(request.conv.dense().defined())request.conv.dense().zero_();
            if(request.recurrent.dense().defined())request.recurrent.dense().zero_();
            if(request.core_workspace.dense().defined())request.core_workspace.dense().zero_();
        } else {
            request.conv.dense() = DenseTensor();
            request.recurrent.dense() = DenseTensor();
            request.core_workspace.dense() = DenseTensor();
        }
        commit(request);
    }
    void commit(State& request) {
        request.saved_conv = DenseTensor();
        request.saved_state = DenseTensor();
        request.saved_prefix_conv.clear();request.saved_prefix_state.clear();request.saved_prefix_position=-1;
    }
    void rollback(State& request, int64_t keep=-1) {
        TYLOINFER_RUNTIME_CHECK(request.saved_conv.defined(), "Qwen4 GDN has no speculative checkpoint");
        if(keep>=0 && request.saved_prefix_position>=0) {
            const auto index=keep-request.saved_prefix_position;
            TYLOINFER_RUNTIME_CHECK(index>=0 && index<int64_t(request.saved_prefix_conv.size()),
                "Qwen4 GDN prefix checkpoint is out of range");
            request.saved_conv=request.saved_prefix_conv[index];request.saved_state=request.saved_prefix_state[index];
        }
        if(request.conv.dense().defined())request.conv.dense().copy_(request.saved_conv);else request.conv.dense()=request.saved_conv;
        if(request.recurrent.dense().defined())request.recurrent.dense().copy_(request.saved_state);else request.recurrent.dense()=request.saved_state;
        commit(request);
    }
    const DenseTensor &conv_state(const State& request) const { return request.conv.dense(); }
    void initialize_request_cache(State& request, bool allocate_counters=true) {
        // Recurrence is allocated on first use, outside temporary pool pages.
        // Keep the reusable decode counters out of graph scratch as well.
        request.resident=true;
#ifdef TYLOINFER_NATIVE_CUDA
        ResidentAllocationScope resident;
#endif
#ifdef TYLOINFER_NATIVE_CUDA
        auto options=w_->norm.options().dtype(::tyloi::kFloat32);
#else
        auto options=w_->norm.options().dtype(::tyloi::kFloat32);
#endif
        if(allocate_counters && !request.core_workspace.dense().defined() && fused_core_ && fused_decode_ &&
            fused_preparation_ && fused_output_ && transposed_state_)
            request.core_workspace.dense()=::tyloi::cuda::zeros({1,nv_,d_+1},options);
    }
    std::size_t growth_bytes(const State& request) const {
        return (request.conv.dense().defined() ? 0 : std::size_t(kernel_-1)*(2*nk_+nv_)*d_*sizeof(float)) +
            (request.recurrent.dense().defined() ? 0 : std::size_t(nv_)*d_*d_*sizeof(float));
    }
    std::size_t request_cache_growth_bytes(const State& request) const {
        return growth_bytes(request) + (fused_core_ && fused_decode_ && fused_preparation_ &&
            fused_output_ && transposed_state_ && !request.core_workspace.dense().defined()
            ? std::size_t(nv_)*(d_+1)*sizeof(float) : 0);
    }
    void restore_session(State& request, const DenseTensor& convolution,const DenseTensor& recurrent,bool retain_storage=false) {
        TYLOINFER_RUNTIME_CHECK(convolution.defined() && recurrent.defined(),"missing Qwen4 GDN session state");
        if(retain_storage) {
            TYLOINFER_RUNTIME_CHECK(request.conv.dense().defined() && request.recurrent.dense().defined() && request.conv.dense().sizes()==convolution.sizes() &&
                request.recurrent.dense().sizes()==recurrent.sizes(),"Qwen4 GDN graph session geometry changed");
            commit(request);request.conv.dense().copy_(convolution);request.recurrent.dense().copy_(recurrent);
        } else {reset(request);request.conv.dense()=convolution.clone();request.recurrent.dense()=recurrent.clone();}
    }
    DenseTensor recurrent_state(const State& request) const {
        return request.recurrent.dense().defined() && transposed_state_ ? request.recurrent.dense().transpose(-1,-2) : request.recurrent.dense();
    }
    std::vector<::tyloi::Tensor*> graph_state(State& request) { return {&request.conv, &request.recurrent}; }
    std::vector<DenseTensor> project_inputs(const DenseTensor& x) const;
    DenseTensor project_output(const DenseTensor& x) const;
    static DenseTensor prepare_independent_batch(const std::vector<std::pair<const Gdn*, State*>>& rows) {
        if(rows.size()<2 || rows.size()>32)return {};
        const auto inplace=::tyloi::cuda::options::gdn_inplace_state();
        if(inplace && !*inplace)return {};
        const auto& first=*rows.front().first;
        const auto& first_state=*rows.front().second;
        const auto eligible=[](const auto& entry) {
            const auto& row=*entry.first;
            const auto& request=*entry.second;
            return row.fused_core_ && row.fused_decode_ && row.fused_preparation_ &&
                row.fused_output_ && row.transposed_state_ &&
                (row.d_==32 || row.d_==64 || row.d_==128) &&
                request.conv.dense().defined() && request.recurrent.dense().defined() && request.core_workspace.dense().defined() &&
                request.conv.dense().is_cuda() && request.recurrent.dense().is_cuda() && request.core_workspace.dense().is_cuda() &&
                request.conv.dense().is_contiguous() && request.recurrent.dense().is_contiguous() && request.core_workspace.dense().is_contiguous() &&
                request.conv.dense().scalar_type()==::tyloi::kFloat32 && request.recurrent.dense().scalar_type()==::tyloi::kFloat32 &&
                request.core_workspace.dense().scalar_type()==::tyloi::kFloat32 &&
                request.conv.dense().sizes().vec()==std::vector<int64_t>({1,row.kernel_-1,(2*row.nk_+row.nv_)*row.d_}) &&
                request.recurrent.dense().sizes().vec()==std::vector<int64_t>({1,row.nv_,row.d_,row.d_}) &&
                request.core_workspace.dense().sizes().vec()==std::vector<int64_t>({1,row.nv_,row.d_+1});
        };
        if(::tyloi::cuda::options::gdn_independent_batch_trace())
            std::fprintf(stderr,"gdn_batch_state eligible=%d flags=%d%d%d%d%d conv=%d state=%d counters=%d geometry=%lld/%lld/%lld/%lld\n",
                int(eligible(rows.front())),int(first.fused_core_),int(first.fused_decode_),int(first.fused_preparation_),
                int(first.fused_output_),int(first.transposed_state_),int(first_state.conv.dense().defined()),int(first_state.recurrent.dense().defined()),
                int(first_state.core_workspace.dense().defined()),(long long)first.nk_,(long long)first.nv_,(long long)first.d_,(long long)first.kernel_);
        if(!eligible(rows.front()))return {};
        for(const auto& row:rows)if(!eligible(row) || row.first->nk_!=first.nk_ || row.first->nv_!=first.nv_ ||
            row.first->d_!=first.d_ || row.first->kernel_!=first.kernel_ || row.first->eps_!=first.eps_ ||
            row.first->silu_gate_!=first.silu_gate_ || row.second->recurrent.dense().device()!=first_state.recurrent.dense().device() ||
            row.second->conv.dense().device()!=first_state.recurrent.dense().device() || row.second->core_workspace.dense().device()!=first_state.recurrent.dense().device() ||
#ifdef TYLOINFER_NATIVE_CUDA
            row.first->w_->conv.dense().data_ptr()!=first.w_->conv.dense().data_ptr() || row.first->w_->norm.dense().data_ptr()!=first.w_->norm.dense().data_ptr() ||
            row.first->w_->dt_bias.dense().data_ptr()!=first.w_->dt_bias.dense().data_ptr() || row.first->w_->a_log.dense().data_ptr()!=first.w_->a_log.dense().data_ptr())return {};
#else
            row.first->w_->conv.data_ptr()!=first.w_->conv.data_ptr() || row.first->w_->norm.data_ptr()!=first.w_->norm.data_ptr() ||
            row.first->w_->dt_bias.data_ptr()!=first.w_->dt_bias.data_ptr() || row.first->w_->a_log.data_ptr()!=first.w_->a_log.data_ptr())return {};
#endif
        auto addresses=::tyloi::cuda::empty({int64_t(rows.size()),3},first_state.recurrent.dense().options().device(::tyloi::kCPU).dtype(::tyloi::kInt64));
        auto* values=addresses.data_ptr<int64_t>();
        for(std::size_t i=0;i<rows.size();++i) {
            values[3*i]=reinterpret_cast<int64_t>(rows[i].second->conv.dense().data_ptr());
            values[3*i+1]=reinterpret_cast<int64_t>(rows[i].second->recurrent.dense().data_ptr());
            values[3*i+2]=reinterpret_cast<int64_t>(rows[i].second->core_workspace.dense().data_ptr());
        }
        return addresses.to(first_state.recurrent.dense().device());
    }
    DenseTensor forward_independent_batch(State& request, const DenseTensor& x,const std::vector<DenseTensor>& projections,
                                     const DenseTensor& addresses) {
        if(::tyloi::cuda::options::gdn_independent_batch_trace() && projections.size()==4)
            std::fprintf(stderr,"gdn_batch_projection input=%d qkv=%d gate=%d alpha=%d beta=%d rows=%lld\n",
                int(x.scalar_type()),int(projections[0].scalar_type()),int(projections[1].scalar_type()),
                int(projections[2].scalar_type()),int(projections[3].scalar_type()),(long long)x.size(0));
        if(!addresses.defined() || projections.size()!=4 || x.size(1)!=1 ||
            (x.scalar_type()!=::tyloi::kFloat16 && x.scalar_type()!=::tyloi::kFloat32) ||
            projections[0].scalar_type()!=::tyloi::kFloat16 ||
            projections[1].scalar_type()!=projections[2].scalar_type() ||
            projections[2].scalar_type()!=projections[3].scalar_type() ||
            (projections[1].scalar_type()!=::tyloi::kFloat16 && projections[1].scalar_type()!=::tyloi::kFloat32))return {};
        if(::tyloi::cuda::options::gdn_independent_batch_trace())
            std::fprintf(stderr,"gdn_batch_core rows=%lld\n",(long long)x.size(0));
#ifdef TYLOINFER_NATIVE_CUDA
        auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
#else
        auto weight=w_->conv.to(::tyloi::kFloat32).contiguous();
#endif
        if(weight.dim()==2 && weight.size(0)==(2*nk_+nv_)*d_ && weight.size(1)==kernel_)
            weight=weight.reshape({(2*nk_+nv_)*d_,1,kernel_});
        const auto stride_setting=::tyloi::cuda::options::gdn_strided_projections();
        const auto direct=[&](const DenseTensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
        auto result=gdn_decode_core_cuda(direct(projections[0]),direct(projections[1]),
            projections[2].reshape({x.size(0),1,nv_}),projections[3].reshape({x.size(0),1,nv_}),
#ifdef TYLOINFER_NATIVE_CUDA
            request.conv.dense(),request.recurrent.dense(),weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
            w_->norm.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
#else
            request.conv.dense(),request.recurrent.dense(),weight,w_->dt_bias.to(::tyloi::kFloat32).contiguous(),w_->a_log.to(::tyloi::kFloat32).contiguous(),
            w_->norm.to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
#endif
            x.scalar_type()==::tyloi::kFloat16,true,request.core_workspace.dense(),true,addresses);
        return result[0];
    }
    DenseTensor forward_graph(State& request, const DenseTensor& x,int64_t confirmed,
                         std::vector<DenseTensor>& checkpoint_conv,std::vector<DenseTensor>& checkpoint_state) {
        if(!confirmed)return forward_chunk(request, x,true);
        const auto batch=::tyloi::cuda::options::gdn_batch_projection();
        const bool grouped=grouped_projection_ && ::tyloi::ops::grouped(w_->input_projection) && x.size(1)<=8 && (!batch || *batch);
        // Only recurrence is split at the confirmation boundary. Both parts
        // share the input projections and one output projection.
        std::vector<DenseTensor> projected;if(grouped)projected=project_inputs(x);
        const auto* window_option=tyloi::cuda::env_lookup("TYLOINFER_MTP_GDN_REGISTER_WINDOW");
        const auto inplace_option=::tyloi::cuda::options::gdn_inplace_state();
        const auto* checkpoint_option=tyloi::cuda::env_lookup("TYLOINFER_MTP_GDN_FUSED_CHECKPOINT");
        if(grouped && confirmed==1 && x.size(0)==1 && x.size(1)>=2 && x.size(1)<=8 &&
            (!window_option || window_option[0]!='0') && nv_<=65535 &&
            (!inplace_option || *inplace_option) && (!checkpoint_option || checkpoint_option[0]!='0') &&
            d_==128 && kernel_==4 && fused_core_ && fused_decode_ && fused_preparation_ &&
            fused_output_ && transposed_state_ && request.conv.dense().defined() && request.recurrent.dense().defined() &&
            projected.size()==4 && projected[0].scalar_type()==::tyloi::kFloat16 &&
            projected[1].scalar_type()==projected[2].scalar_type() &&
            projected[2].scalar_type()==projected[3].scalar_type() &&
            (projected[1].scalar_type()==::tyloi::kFloat16 || projected[1].scalar_type()==::tyloi::kFloat32) &&
            (x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32)) {
#ifdef TYLOINFER_NATIVE_CUDA
            auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
#else
            auto weight=w_->conv.to(::tyloi::kFloat32).contiguous();
#endif
            const auto channels=(2*nk_+nv_)*d_;
            if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                weight=weight.reshape({channels,1,kernel_});
            auto decoded=gdn_decode_window_cuda(projected[0],projected[1],projected[2].reshape({1,x.size(1),nv_}),
#ifdef TYLOINFER_NATIVE_CUDA
                projected[3].reshape({1,x.size(1),nv_}),request.conv.dense(),request.recurrent.dense(),weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),
                w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),w_->norm.dense().to(::tyloi::kFloat32).contiguous(),int(nk_),int(nv_),int(confirmed),
#else
                projected[3].reshape({1,x.size(1),nv_}),request.conv.dense(),request.recurrent.dense(),weight,w_->dt_bias.to(::tyloi::kFloat32).contiguous(),
                w_->a_log.to(::tyloi::kFloat32).contiguous(),w_->norm.to(::tyloi::kFloat32).contiguous(),int(nk_),int(nv_),int(confirmed),
#endif
                1e-6,eps_,silu_gate_,x.scalar_type()==::tyloi::kFloat16,true,checkpoint_conv,checkpoint_state);
            request.conv.dense().copy_(decoded[1]);
            return project_output(decoded[0]);
        }
        const auto slice=[&](int64_t begin,int64_t count) {
            std::vector<DenseTensor> result;for(const auto& value:projected)result.push_back(value.narrow(1,begin,count));
            return result;
        };
        // Project the whole verification window together, but preserve the
        // recurrent state after each prefix. Partial rejection then selects a
        // checkpoint instead of replaying accepted tokens through every layer.
        std::vector<DenseTensor> outputs;
        for(int64_t begin=0;begin<x.size(1);) {
            const auto count=begin==0?confirmed:1;
            const auto end=begin+count,index=end-confirmed;
            bool snapshot_written=false;
            outputs.push_back(forward_chunk(request, x.narrow(1,begin,count),true,
                slice(begin,count),!grouped,
                end<x.size(1)?checkpoint_conv[index]:DenseTensor{},
                end<x.size(1)?checkpoint_state[index]:DenseTensor{},&snapshot_written));
            begin=end;
            if(begin<x.size(1) && !snapshot_written) {
                checkpoint_conv[index].copy_(request.conv.dense());checkpoint_state[index].copy_(request.recurrent.dense());
            }
        }
        auto result=::tyloi::cuda::cat(outputs,1);
        return grouped?project_output(result):result;
    }
    void bind_graph_checkpoint(State& request, const std::vector<DenseTensor>& conv,const std::vector<DenseTensor>& state,
                               int64_t position) {
        TYLOINFER_RUNTIME_CHECK(!request.saved_conv.defined(),"resolve previous GDN checkpoint");
        TYLOINFER_RUNTIME_CHECK(!conv.empty() && conv.size()==state.size(),"invalid GDN prefix checkpoints");
        request.saved_conv=conv.front();request.saved_state=state.front();
        request.saved_prefix_conv=conv;request.saved_prefix_state=state;request.saved_prefix_position=position;
    }
    DenseTensor forward(State& request, const DenseTensor &x, bool cache,
                   int64_t confirmed = 0, int64_t cache_position = -1) {
        TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.dim() == 3 && x.size(0) > 0,
                          "recurrent attention requires nonempty [B,T,H] input");
        if(cache && cache_position>=0 && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8) {
            TYLOINFER_RUNTIME_CHECK(!request.saved_conv.defined() && request.conv.dense().defined() && request.recurrent.dense().defined(),
                              "GDN verification requires committed recurrent state");
            std::vector<DenseTensor> conv,state;
            for(int64_t i=confirmed;i<x.size(1);++i) {
                conv.push_back(::tyloi::cuda::empty_like(request.conv.dense()));state.push_back(::tyloi::cuda::empty_like(request.recurrent.dense()));
            }
            auto result=forward_graph(request, x,confirmed,conv,state);
            bind_graph_checkpoint(request, conv,state,cache_position+confirmed);
            return result;
        }
        const auto batch=::tyloi::cuda::options::gdn_batch_projection();
        const bool grouped=cache && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8 &&
            grouped_projection_ && ::tyloi::ops::grouped(w_->input_projection) && (!batch || *batch);
        std::vector<DenseTensor> projected;
        if(grouped)projected=project_inputs(x);
        auto result=tyloi::models::recurrent_window(
            x.size(1), cache, confirmed, request.saved_conv.defined(),
            [&](int64_t start, int64_t count) {
                if(!grouped)return forward_chunk(request, x.narrow(1,start,count),cache);
                // Share both projections across the verification window and
                // preserve the same per-prefix recurrent order as the graph.
                std::vector<DenseTensor> outputs;
                for(int64_t begin=start;begin<start+count;) {
                    const auto rows=begin==0?count:int64_t(1);
                    std::vector<DenseTensor> inputs;
                    for(const auto& value:projected)inputs.push_back(value.narrow(1,begin,rows));
                    outputs.push_back(forward_chunk(request, x.narrow(1,begin,rows),cache,std::move(inputs),false));
                    begin+=rows;
                }
                return outputs.size()==1?std::move(outputs.front()): ::tyloi::cuda::cat(outputs,1);
            },
            [&] {
                request.saved_conv = request.conv.dense().clone();
                request.saved_state = request.recurrent.dense().clone();
            },
            [&] { rollback(request); },
            [](DenseTensor prefix, DenseTensor suffix) { return ::tyloi::cuda::cat({prefix, suffix}, 1); });
        return grouped?project_output(result):result;
    }
    DenseTensor forward_chunk(State& request, const DenseTensor &x, bool cache,
                         std::vector<DenseTensor> projections = {},bool project_output=true,
                         DenseTensor checkpoint_conv={},DenseTensor checkpoint_state={},bool* snapshot_written=nullptr) {
        if(snapshot_written)*snapshot_written=false;
        const auto b = x.size(0), t = x.size(1), kw = nk_ * d_, vw = nv_ * d_,
                   channels = 2 * kw + vw;
        auto options = x.options().dtype(::tyloi::kFloat32);
        TYLOINFER_RUNTIME_CHECK(!cache || !request.conv.dense().defined() ||
                              (request.conv.dense().size(0) == b && request.conv.dense().device() == x.device()),
                          "reset Qwen4 GDN before changing batch/device");
        auto previous =
            cache && request.conv.dense().defined() ? request.conv.dense() : ::tyloi::cuda::zeros({b, kernel_ - 1, channels}, options);
        std::array<DenseTensor,2> prepared_gates;
        const auto prefill_option=::tyloi::cuda::options::gdn_prefill_fused();
        const bool prefill_fused=t>1 && fused_preparation_ && d_<=256 && kernel_<=8 &&
            (!prefill_option || *prefill_option!=0);
        if(cache && request.conv.dense().defined() && request.recurrent.dense().defined() && fused_core_ && fused_decode_ &&
            fused_preparation_ && fused_output_ && transposed_state_ && t==1 &&
            (d_==32 || d_==64 || d_==128) &&
            (x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32)) {
            if(projections.empty()) {
                projections=project_inputs(x);
            }
            TYLOINFER_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
            if(projections[0].scalar_type()==::tyloi::kFloat16 &&
                projections[1].scalar_type()==projections[2].scalar_type() &&
                projections[2].scalar_type()==projections[3].scalar_type() &&
                (projections[1].scalar_type()==::tyloi::kFloat16 || projections[1].scalar_type()==::tyloi::kFloat32)) {
#ifdef TYLOINFER_NATIVE_CUDA
                auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
#else
                auto weight=w_->conv.to(::tyloi::kFloat32).contiguous();
#endif
                if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                    weight=weight.reshape({channels,1,kernel_});
                if(!request.core_workspace.dense().defined())request.core_workspace.dense()=::tyloi::cuda::zeros({b,nv_,d_+1},options);
                const auto stride_setting=::tyloi::cuda::options::gdn_strided_projections();
                const auto direct=[&](const DenseTensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
                const auto decode=[&](bool checkpoint) {
                  auto qkv=direct(projections[0]),gate=direct(projections[1]);
                  auto alpha=projections[2].reshape({b,t,nv_}),beta=projections[3].reshape({b,t,nv_});
                  if(checkpoint)return gdn_decode_core_checkpoint_cuda(qkv,gate,alpha,beta,previous,request.recurrent.dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                    weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
                    w_->norm.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
#else
                    weight,w_->dt_bias.to(::tyloi::kFloat32).contiguous(),w_->a_log.to(::tyloi::kFloat32).contiguous(),
                    w_->norm.to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
#endif
                    x.scalar_type()==::tyloi::kFloat16,true,request.core_workspace.dense(),true,checkpoint_conv,checkpoint_state);
                  return gdn_decode_core_cuda(qkv,gate,
                    projections[2].reshape({b,t,nv_}),projections[3].reshape({b,t,nv_}),previous,request.recurrent.dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                    weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
                    w_->norm.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
#else
                    weight,w_->dt_bias.to(::tyloi::kFloat32).contiguous(),w_->a_log.to(::tyloi::kFloat32).contiguous(),
                    w_->norm.to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
#endif
                    x.scalar_type()==::tyloi::kFloat16,true,request.core_workspace.dense(),
                    [] {const auto value=::tyloi::cuda::options::gdn_inplace_state();return !value || *value;}());
                };
                const auto* checkpoint_option=tyloi::cuda::env_lookup("TYLOINFER_MTP_GDN_FUSED_CHECKPOINT");
                const bool fused_snapshot=checkpoint_conv.defined() && checkpoint_state.defined() &&
                    (!checkpoint_option || checkpoint_option[0]!='0') &&
                    [] {const auto value=::tyloi::cuda::options::gdn_inplace_state();return !value || *value;}();
                auto decoded=decode(fused_snapshot);
                if(snapshot_written)*snapshot_written=fused_snapshot;
                if(fused_snapshot) {
                    static thread_local bool reported=false;
                    if(!reported) {
                        reported=true;
                        std::fprintf(stderr,"runtime_mtp_gdn_checkpoint fused=1 width=%lld heads=%lld\n",
                            static_cast<long long>(d_),static_cast<long long>(nv_));
                    }
                }
                auto output=project_output?this->project_output(decoded[0]):decoded[0];
                if(request.conv.dense().data_ptr()!=decoded[1].data_ptr())request.conv.dense().copy_(decoded[1]);
                if(request.recurrent.dense().data_ptr()!=decoded[2].data_ptr())request.recurrent.dense().copy_(decoded[2]);
                return output;
            }
        }
        return tyloi::models::gated_delta_attention(
            cache,
            [&] {
                DenseTensor projected;
                if(!projections.empty())projected=projections[0];
                else if(grouped_projection_ && ::tyloi::ops::grouped(w_->input_projection)) {
                    projections=project_inputs(x);
                    TYLOINFER_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
                    projected=projections[0];
                } else projected=project_qkv(x);
                TYLOINFER_RUNTIME_CHECK(projected.sizes().vec() == std::vector<int64_t>({b, t, channels}),
                                  "Qwen4 GDN projection width mismatch");
                return projected;
            },
            [&](DenseTensor projected) {
                if(prefill_fused && projected.scalar_type()==::tyloi::kFloat16) {
                    return ::tyloi::ops::measure("prefill.gdn_conv_qkv",[&] {
#ifdef TYLOINFER_NATIVE_CUDA
                        auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
#else
                        auto weight=w_->conv.to(::tyloi::kFloat32).contiguous();
#endif
                        if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                            weight=weight.reshape({channels,1,kernel_});
                        const auto short_option=::tyloi::cuda::options::gdn_short_fusion();
                        if(t<=8 && (!short_option || *short_option) && projections.size()==4 &&
                            projections[2].scalar_type()==projections[3].scalar_type() &&
                            (projections[2].scalar_type()==::tyloi::kFloat16 || projections[2].scalar_type()==::tyloi::kFloat32)) {
                            const auto stride_setting=::tyloi::cuda::options::gdn_strided_projections();
                            const auto direct=[&](const DenseTensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
                            auto qkv=linear_conv_qkv_gate_short_cuda(previous,direct(projected),weight,
                                direct(projections[2].reshape({b,t,nv_})),direct(projections[3].reshape({b,t,nv_})),
#ifdef TYLOINFER_NATIVE_CUDA
                                w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6);
#else
                                w_->dt_bias.to(::tyloi::kFloat32).contiguous(),w_->a_log.to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6);
#endif
                            prepared_gates={qkv[4],qkv[5]};
                            return std::array<DenseTensor,4>{qkv[0],qkv[1],qkv[2],qkv[3]};
                        }
                        auto qkv=linear_conv_qkv_prefill_cuda(previous,
                            projected.narrow(-1,0,2*kw).contiguous(),
                            projected.narrow(-1,2*kw,vw).contiguous(),weight,
                            ::tyloi::cuda::empty({0},options),nk_,nv_,d_,d_,1e-6);
                        return std::array<DenseTensor,4>{qkv[0],qkv[1],qkv[2],qkv[3]};
                    });
                }
                if (fused_decode_ && t == 1 && d_ <= 256 && projected.scalar_type() == ::tyloi::kFloat16) {
#ifdef TYLOINFER_NATIVE_CUDA
                    auto weight = w_->conv.dense().to(::tyloi::kFloat32).contiguous();
#else
                    auto weight = w_->conv.to(::tyloi::kFloat32).contiguous();
#endif
                    if (weight.dim() == 2 && weight.size(0) == channels && weight.size(1) == kernel_)
                        weight = weight.reshape({channels, 1, kernel_});
                    auto next_conv = previous.clone();
                    auto qk=projected.narrow(-1,0,2*kw).contiguous(),v=projected.narrow(-1,2*kw,vw).contiguous();
                    std::vector<DenseTensor> qkv;
                    if(fused_preparation_) {
                        auto alpha=projections.empty()
                            ?project_weight(w_->input_projection[w_->qkv_count+1],x):projections[2];
                        auto beta=projections.empty()
                            ?project_weight(w_->input_projection[w_->qkv_count+2],x):projections[3];
                        if(alpha.scalar_type()!=beta.scalar_type() ||
                            (alpha.scalar_type()!=::tyloi::kFloat16 && alpha.scalar_type()!=::tyloi::kFloat32)) {
                            alpha=alpha.to(::tyloi::kFloat32);beta=beta.to(::tyloi::kFloat32);
                        }
                        qkv=linear_conv_qkv_gate_decode_cuda(next_conv,qk,v,weight,
                            alpha.reshape({b,t,nv_}),beta.reshape({b,t,nv_}),
#ifdef TYLOINFER_NATIVE_CUDA
                            w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
#else
                            w_->dt_bias.to(::tyloi::kFloat32).contiguous(),w_->a_log.to(::tyloi::kFloat32).contiguous(),
#endif
                            nk_,nv_,d_,d_,1e-6);
                        prepared_gates={qkv[3],qkv[4]};
                    } else qkv=linear_conv_qkv_decode_cuda(next_conv,qk,v,weight,
                        ::tyloi::cuda::empty({0}, options),nk_,nv_,d_,d_,1e-6);
                    return std::array<DenseTensor, 4>{qkv[0], qkv[1], qkv[2], next_conv};
                }
                auto joined = ::tyloi::cuda::cat({previous, projected.to(::tyloi::kFloat32)}, 1).contiguous();
#ifdef TYLOINFER_NATIVE_CUDA
                auto convolved = ssm_conv_silu_cuda(joined, w_->conv.dense().to(::tyloi::kFloat32).contiguous(),
#else
                auto convolved = ssm_conv_silu_cuda(joined, w_->conv.to(::tyloi::kFloat32).contiguous(),
#endif
                                                    ::tyloi::cuda::empty({0}, options), t);
                auto next_conv = joined.narrow(1, t, kernel_ - 1).contiguous();
                const auto normalize = [](const DenseTensor &v) {
                    return (v / ::tyloi::cuda::clamp_min((v * v).sum(-1, true).sqrt(), 1e-6)).contiguous();
                };
                auto q = normalize(
                    convolved.narrow(-1, 0, kw).reshape({b, t, nk_, d_}).permute({0, 2, 1, 3}));
                auto k = normalize(
                    convolved.narrow(-1, kw, kw).reshape({b, t, nk_, d_}).permute({0, 2, 1, 3}));
                auto v = convolved.narrow(-1, 2 * kw, vw)
                             .reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .contiguous();
                return std::array<DenseTensor, 4>{q, k, v, next_conv};
            },
            [&] {
                if(prepared_gates[0].defined())return prepared_gates;
                auto alpha=projections.empty()
                    ?project_weight(w_->input_projection[w_->qkv_count+1],x):projections[2];
                auto raw_beta=projections.empty()
                    ?project_weight(w_->input_projection[w_->qkv_count+2],x):projections[3];
                if ((fused_decode_ && t == 1) || prefill_fused) {
                    auto gates = linear_gate_beta_cuda(
                        alpha.to(::tyloi::kFloat32).reshape({b, t, nv_}),
                        raw_beta.to(::tyloi::kFloat32).reshape({b, t, nv_}),
#ifdef TYLOINFER_NATIVE_CUDA
                        w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),
                        w_->a_log.dense().to(::tyloi::kFloat32).contiguous(), true);
#else
                        w_->dt_bias.to(::tyloi::kFloat32).contiguous(),
                        w_->a_log.to(::tyloi::kFloat32).contiguous(), true);
#endif
                    return std::array<DenseTensor, 2>{gates[0], gates[1]};
                }
                auto gate_input = alpha.to(::tyloi::kFloat32).reshape({b, t, nv_}) +
#ifdef TYLOINFER_NATIVE_CUDA
                                  w_->dt_bias.dense().to(::tyloi::kFloat32).reshape({1, 1, nv_});
#else
                                  w_->dt_bias.to(::tyloi::kFloat32).reshape({1, 1, nv_});
#endif
                auto softplus =
                    ::tyloi::cuda::clamp_min(gate_input, 0) + ::tyloi::cuda::log1p(::tyloi::cuda::exp(-gate_input.abs()));
#ifdef TYLOINFER_NATIVE_CUDA
                auto decay = (-w_->a_log.dense().to(::tyloi::kFloat32).exp().reshape({1, 1, nv_}) * softplus)
#else
                auto decay = (-w_->a_log.to(::tyloi::kFloat32).exp().reshape({1, 1, nv_}) * softplus)
#endif
                                 .transpose(1, 2)
                                 .contiguous();
                auto beta = ::tyloi::cuda::sigmoid(raw_beta.to(::tyloi::kFloat32).reshape({b, t, nv_}))
                                .transpose(1, 2)
                                .contiguous();
                return std::array<DenseTensor, 2>{decay, beta};
            },
            [&](const auto &convolution, const auto &gates) {
                auto q = convolution[0], k = convolution[1], v = convolution[2];
                auto decay = gates[0], beta = gates[1];
                auto initial = cache && request.recurrent.dense().defined() ? request.recurrent.dense() : DenseTensor{};
                DenseTensor attended, next_state;
                if (d_ == 32 || d_ == 64 || d_ == 128) {
                    auto result = ::tyloi::ops::measure("prefill.gdn_recurrence",[&] {
                        const auto short_option=::tyloi::cuda::options::gdn_short_inplace();
                        const auto inplace=::tyloi::cuda::options::gdn_inplace_state();
                        if(cache && initial.defined() && t<=8 && (!short_option || *short_option) && (!inplace || *inplace))
                            return (transposed_state_?gdn_inplace_transposed_cuda:gdn_inplace_cuda)(q,k,v,decay,beta,initial);
                        return (transposed_state_ ? gdn_transposed_cuda : gdn_cuda)(
                            q, k, v, decay, beta,initial.defined() ? MfqOptional<DenseTensor>(initial) : mfq_nullopt);
                    });
                    attended = result[0];
                    next_state = result[1];
                } else {
                    auto ids = ::tyloi::cuda::arange(nv_, options.dtype(::tyloi::kInt64)) / (nv_ / nk_);
                    q = q.index_select(1, ids.to(::tyloi::kInt64));
                    k = k.index_select(1, ids.to(::tyloi::kInt64));
                    next_state = initial.defined() ? initial : ::tyloi::cuda::zeros({b, nv_, d_, d_}, options);
                    std::vector<DenseTensor> out;
                    for (int64_t i = 0; i < t; ++i) {
                        auto ki = k.select(2, i), rate = decay.select(2, i).exp().unsqueeze(-1);
                        auto projected_key = (next_state * ki.unsqueeze(-1)).sum(-2);
                        auto delta = (v.select(2, i) - rate * projected_key) *
                                     beta.select(2, i).unsqueeze(-1);
                        next_state = rate.unsqueeze(-1) * next_state +
                                     ki.unsqueeze(-1) * delta.unsqueeze(-2);
                        out.push_back(((next_state * q.select(2, i).unsqueeze(-1)).sum(-2) /
                                       std::sqrt(double(d_)))
                                          .unsqueeze(2));
                    }
                    attended = ::tyloi::cuda::cat(out, 2);
                }
                return std::array<DenseTensor, 2>{attended, next_state};
            },
            [&](DenseTensor attended) {
                const auto stride_setting=::tyloi::cuda::options::gdn_strided_projections();
                const bool strided=!stride_setting || *stride_setting;
                auto projected_gate=projections.empty()
                    ?project_weight(w_->input_projection[w_->qkv_count],x)
                    :strided?projections[1]:projections[1].contiguous();
                if(fused_decode_ && fused_output_ && attended.is_contiguous() &&
                    attended.scalar_type()==::tyloi::kFloat32 && (projected_gate.is_contiguous() ||
                        (strided && projected_gate.dim()==3 && projected_gate.stride(2)==1)) &&
                    (projected_gate.scalar_type()==::tyloi::kFloat16 || projected_gate.scalar_type()==::tyloi::kFloat32) &&
                    (x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32))
                    return gdn_rms_norm_gate_cuda(attended,projected_gate,
#ifdef TYLOINFER_NATIVE_CUDA
                        w_->norm.dense().to(::tyloi::kFloat32).contiguous(),eps_,silu_gate_,x.scalar_type()==::tyloi::kFloat16);
#else
                        w_->norm.to(::tyloi::kFloat32).contiguous(),eps_,silu_gate_,x.scalar_type()==::tyloi::kFloat16);
#endif
                projected_gate=projected_gate.contiguous();
                auto z = projected_gate.reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .to(::tyloi::kFloat32);
                auto gate = ::tyloi::cuda::sigmoid(z);
                if (silu_gate_)
                    gate = z * gate;
#ifdef TYLOINFER_NATIVE_CUDA
                return rms_norm(attended, w_->norm.dense(), eps_) * gate;
#else
                return rms_norm(attended, w_->norm, eps_) * gate;
#endif
            },
            [&](DenseTensor normalized) {
                if(normalized.dim()==3)return project_output?this->project_output(normalized):normalized;
                normalized=normalized.permute({0,2,1,3}).reshape({b,t,vw}).to(x.scalar_type());
                if(!project_output)return normalized;
                auto result = this->project_output(normalized);
                return result;
            },
            [&](const auto &convolution, const DenseTensor &next) {
                if(request.conv.dense().defined())request.conv.dense().copy_(convolution[3]);
                else {
#ifdef TYLOINFER_NATIVE_CUDA
                    ResidentAllocationScope resident(request.resident);
#endif
                    request.conv.dense()=request.resident?convolution[3].clone():convolution[3];
                }
                if(request.recurrent.dense().defined()) {if(request.recurrent.dense().data_ptr()!=next.data_ptr())request.recurrent.dense().copy_(next);}
                else {
#ifdef TYLOINFER_NATIVE_CUDA
                    ResidentAllocationScope resident(request.resident);
#endif
                    request.recurrent.dense()=request.resident?next.clone():next;
                }
            });
    }

  private:
    DenseTensor project_weight(const ::tyloi::Tensor& weight,const DenseTensor& source) const;
    DenseTensor project_qkv(const DenseTensor& source) const;
    std::shared_ptr<const GdnWeights> w_;
    int64_t nk_, nv_, d_, kernel_;
    double eps_;
    bool silu_gate_, fused_decode_,transposed_state_,fused_output_,grouped_projection_,fused_preparation_,fused_core_;
};

// The upstream PLE hash is a CPU uint64 algorithm, including signed remainder
// after bitwise mixing and EOS segment boundaries. Keep shards independent.
class NgramEmbedding {
  public:
    using Context = tyloi::models::qwen4_exp::NgramContext;
    using RowLookup = std::function<::tyloi::Tensor(const std::vector<int64_t>&,const ::tyloi::Device&)>;
    using StageFactory = std::function<std::shared_ptr<RowStage>(const std::vector<int64_t>&,const ::tyloi::Device&)>;
    struct State {
        State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        Context context;
        std::shared_ptr<NintRowPipeline> pipeline;
        bool hash_ready=false,hash_reported=false;
        int cached_tokens=0;
        std::array<int64_t,8> cached_ids{};
        tyloi::models::qwen4_exp::NgramHashes cached_hashes;
    };
    NgramEmbedding(std::vector<Embedding> shards, int64_t rows, int64_t dimension, int64_t ngram,
                   int64_t heads_per_ngram, int64_t eos, std::vector<int64_t> multipliers,
                   std::vector<int64_t> offsets, std::vector<int64_t> vocab,std::vector<std::shared_ptr<tyloi::NintRows>> ranges = {},
                   RowLookup row_lookup = {},StageFactory stage_factory = {})
        : shards_(std::move(shards)), rows_(rows), dimension_(dimension), ngram_(ngram),
          heads_(heads_per_ngram), eos_(eos), multipliers_(std::move(multipliers)),
          offsets_(std::move(offsets)), vocab_(std::move(vocab)),row_lookup_(std::move(row_lookup)),stage_factory_(std::move(stage_factory)) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(!ranges.empty())source_=std::make_shared<NintRowSource>(std::move(ranges));
#else
        (void)ranges;
#endif
        TYLOINFER_RUNTIME_CHECK(!shards_.empty() && rows_ > 0 && dimension_ > 0 && ngram_ > 1 &&
                              heads_ > 0 && multipliers_.size() == size_t(ngram_) &&
                              offsets_.size() == size_t((ngram_ - 1) * heads_) &&
                              vocab_.size() == offsets_.size(),
                          "invalid Qwen4 ngram metadata geometry");
        for (size_t h = 0; h < vocab_.size(); ++h)
            TYLOINFER_RUNTIME_CHECK(vocab_[h] > 0 && offsets_[h] >= 0 &&
                                  offsets_[h] <= int64_t(shards_.size()) * rows_ - vocab_[h],
                              "Qwen4 ngram vocabulary lies outside embedding shards");
    }
    void reset(State& state) const { state.context = {};state.hash_ready=false; }
    bool graph_supported() const { return bool(source_) || bool(stage_factory_); }
    const Context &context(const State& state) const { return state.context; }
    void restore(State& state,Context context) const { state.context = std::move(context);state.hash_ready=false; }
    Context prefix_context(const State& state,const DenseTensor& host_ids,int64_t count) const {
        auto host=host_ids.narrow(1,0,count).to(::tyloi::kInt64).contiguous();
        return tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),host.size(0),count,
            ngram_,heads_,eos_,multipliers_,offsets_,vocab_,state.context,true).next;
    }
    void prefetch(State& state,const DenseTensor& host_ids,bool early_prefix=false) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT

        if(!source_)return;
        if(!state.pipeline)state.pipeline=std::make_shared<NintRowPipeline>(source_);
        TYLOINFER_RUNTIME_CHECK(host_ids.is_cpu() && host_ids.dim()==2,"PLE prefetch requires host token IDs");
        auto host=host_ids.to(::tyloi::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            const auto& hashes=graph_hashes(state,host);
            state.pipeline->issue(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);
            return;
        }
        auto hashes=tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        state.pipeline->issue(hashes.ids,{b,t,(ngram_-1)*heads_},early_prefix);

#else
    (void)state; (void)host_ids; (void)early_prefix;
#endif
}
    std::shared_ptr<RowStage> graph_stage(State& state,int64_t b,int64_t t,const ::tyloi::Device& device) const {
        const std::vector<int64_t> shape{b,t,(ngram_-1)*heads_};
        if(stage_factory_)return stage_factory_(shape,device);
        TYLOINFER_RUNTIME_CHECK(bool(source_),"PLE graph requires prepared range rows");
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(!state.pipeline)state.pipeline=std::make_shared<NintRowPipeline>(source_);
        return std::make_shared<NintStage>(state.pipeline,shape,device);
#else
        throw std::runtime_error("PLE graph requires TQ row stages");
#endif
    }
    void upload_graph(State& state,const DenseTensor& host_ids,RowStage& stage) const {
        TYLOINFER_RUNTIME_CHECK(graph_supported() && host_ids.is_cpu() && host_ids.dim()==2,"PLE graph requires host token IDs");
        auto host=host_ids.to(::tyloi::kInt64).contiguous();const auto b=host.size(0),t=host.size(1);
        if(cacheable_hashes(b,t)) {
            auto& hashes=graph_hashes(state,host);
            stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
            restore(state,std::move(hashes.next));
            return;
        }
        auto hashes=tyloi::models::qwen4_exp::ngram_hashes(host.data_ptr<int64_t>(),b,t,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        stage.upload(hashes.ids,{b,t,(ngram_-1)*heads_});
        restore(state,std::move(hashes.next));
    }
    DenseTensor forward(State& state,const DenseTensor &ids, bool cache, DenseTensor *hashed_ids = nullptr,const DenseTensor& host_ids = {}) const {
        TYLOINFER_RUNTIME_CHECK(ids.dim() == 2 && ids.size(0) > 0 && ids.size(1) > 0,
                          "Qwen4 ngram IDs must be [B,T]");
        const auto b = ids.size(0), t = ids.size(1), nh = (ngram_ - 1) * heads_;
        auto host = host_ids.defined() ? host_ids.to(::tyloi::kInt64).contiguous() : ids.to(::tyloi::kInt64).contiguous().cpu();
        TYLOINFER_RUNTIME_CHECK(host.is_cpu() && host.sizes()==ids.sizes(),"PLE host token mirror differs");
        const auto *source = host.data_ptr<int64_t>();
        auto hashes = tyloi::models::qwen4_exp::ngram_hashes(
            source, b, t, ngram_, heads_, eos_, multipliers_, offsets_, vocab_, state.context, cache);
        auto &global = hashes.ids;
        if(row_lookup_) {
            auto result=row_lookup_(global,ids.device());
            if(hashed_ids)*hashed_ids=::tyloi::cuda::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(state,std::move(hashes.next));
            return result.dense().reshape({b,t,nh*dimension_});
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if(bool(source_)) {
            if(!state.pipeline)state.pipeline=std::make_shared<NintRowPipeline>(source_);
            auto result=state.pipeline->collect(global,{b,t,nh},ids.device());
            if(hashed_ids)*hashed_ids=::tyloi::cuda::tensor(global).reshape({b,t,nh}).to(ids.device());
            if(cache)restore(state,std::move(hashes.next));
            return result.dense().reshape({b,t,nh*dimension_});
        }
#endif

        auto global_tensor =
            ::tyloi::cuda::from_blob(global.data(), {b, t, nh}, ::tyloi::TensorOptions().dtype(::tyloi::kInt64))
                .clone()
                .to(ids.device());
        if (hashed_ids)
            *hashed_ids = global_tensor;
        std::set<int64_t> used;
        for (auto value : global)
            used.insert(value / rows_);
        DenseTensor result;
        for (auto shard : used) {
            auto mask = (global_tensor >= shard * rows_) & (global_tensor < (shard + 1) * rows_);
            auto local =
                ::tyloi::cuda::where(mask, global_tensor - shard * rows_, ::tyloi::cuda::zeros_like(global_tensor));
            auto values = shards_.at(shard)(local);
            TYLOINFER_RUNTIME_CHECK(values.sizes().vec() == std::vector<int64_t>({b, t, nh, dimension_}),
                              "Qwen4 ngram embedding shape mismatch");
            values = values * mask.unsqueeze(-1).to(values.scalar_type());
            result = result.defined() ? result + values : values;
        }
        if (cache)
            restore(state,std::move(hashes.next));
        return result.reshape({b, t, nh * dimension_});
    }

  private:
#if TYLOINFER_ENABLE_MFQ_COMPAT
    struct NintStage final:RowStage {
        std::shared_ptr<NintRowPipeline> pipeline;
        std::shared_ptr<NintRowStage> stage;
        NintStage(std::shared_ptr<NintRowPipeline> p,const std::vector<int64_t>& shape,const ::tyloi::Device& device)
            :pipeline(std::move(p)),stage(pipeline->make_stage(shape,device)) {}
        const ::tyloi::Tensor& output()const override{return stage->output();}
        bool mapped_rows()const override{return stage->mapped_rows();}
        void decode()override{stage->decode();}
        void upload(const std::vector<int64_t>& ids,const std::vector<int64_t>& shape)override {
            pipeline->upload(*stage,ids,shape);
        }
    };
#endif

    std::vector<Embedding> shards_;
    int64_t rows_, dimension_, ngram_, heads_, eos_;
    std::vector<int64_t> multipliers_, offsets_, vocab_;
    std::shared_ptr<const NintRowSource> source_;
    RowLookup row_lookup_;
    StageFactory stage_factory_;
    bool reuse_hashes_=[] {
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PLE_HASH_REUSE");return !option || option[0]!='0';
    }();
    bool cacheable_hashes(int64_t b,int64_t t) const {return reuse_hashes_ && b==1 && t>0 && t<=8;}
    tyloi::models::qwen4_exp::NgramHashes& graph_hashes(State& state,const DenseTensor& host) const {
        const auto tokens=host.size(1);const auto* ids=host.data_ptr<int64_t>();
        if(state.hash_ready && state.cached_tokens==tokens && std::equal(ids,ids+tokens,state.cached_ids.begin())) {
            if(!state.hash_reported) {
                state.hash_reported=true;
                std::fprintf(stderr,"ple_ngram_hash_reuse enabled=1 tokens=%lld\n",static_cast<long long>(tokens));
            }
            return state.cached_hashes;
        }
        state.cached_hashes=tyloi::models::qwen4_exp::ngram_hashes(ids,1,tokens,ngram_,heads_,eos_,
            multipliers_,offsets_,vocab_,state.context,true);
        std::copy_n(ids,tokens,state.cached_ids.begin());state.cached_tokens=int(tokens);state.hash_ready=true;
        return state.cached_hashes;
    }
};

struct PleWeights {
    Linear key, value;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor key_norm, query_norm, conv_norm, conv;
#else
    DenseTensor key_norm, query_norm, conv_norm, conv;
#endif
};
class Ple {
  public:
    struct State {
        NgramEmbedding::State embedding;
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor conv;
#else
        DenseTensor conv;
#endif
        DenseTensor saved_conv;
        NgramEmbedding::Context saved_context;
        std::vector<DenseTensor> saved_prefix_conv;
        std::vector<NgramEmbedding::Context> saved_prefix_context;
        int64_t saved_prefix_position=-1;
        bool saved=false;
    };
    Ple(NgramEmbedding embedding, PleWeights weights, int64_t hidden, int64_t streams,
        int64_t dilation, double eps)
        : embedding_(std::move(embedding)), w_(std::move(weights)), hidden_(hidden),
          streams_(streams), dilation_(dilation), eps_(eps) {}
    void reset(State& state,bool retain_storage=false) const {
        embedding_.reset(state.embedding);
#ifdef TYLOINFER_NATIVE_CUDA
        if(retain_storage && state.conv.defined())state.conv.dense().zero_();else state.conv = {};
#else
        if(retain_storage && state.conv.defined())state.conv.zero_();else state.conv = DenseTensor();
#endif
        commit(state);
    }
    void commit(State& state) const {
        state.saved = false;
        state.saved_conv = DenseTensor();
        state.saved_context = {};
        state.saved_prefix_conv.clear();state.saved_prefix_context.clear();state.saved_prefix_position=-1;
    }
    void rollback(State& state,int64_t keep=-1) const {
        TYLOINFER_RUNTIME_CHECK(state.saved, "Qwen4 PLE has no speculative checkpoint");
        if(keep>=0 && state.saved_prefix_position>=0) {
            const auto index=keep-state.saved_prefix_position;
            TYLOINFER_RUNTIME_CHECK(index>=0 && index<int64_t(state.saved_prefix_conv.size()),
                "Qwen4 PLE prefix checkpoint is out of range");
            state.saved_conv=state.saved_prefix_conv[index];state.saved_context=state.saved_prefix_context[index];
        }
#ifdef TYLOINFER_NATIVE_CUDA
        if(state.conv.defined())state.conv.dense().copy_(state.saved_conv);else state.conv=state.saved_conv;
#else
        if(state.conv.defined())state.conv.copy_(state.saved_conv);else state.conv=state.saved_conv;
#endif
        embedding_.restore(state.embedding,state.saved_context);
        commit(state);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    const ::tyloi::Tensor &conv_state(const State& state) const { return state.conv; }
    std::vector<::tyloi::Tensor*> graph_state(State& state) const { return {&state.conv}; }
#else
    const DenseTensor &conv_state(const State& state) const { return state.conv; }
    std::vector<DenseTensor*> graph_state(State& state) const { return {&state.conv}; }
#endif
    bool graph_supported() const { return embedding_.graph_supported(); }
    const NgramEmbedding::Context& session_context(const State& state) const { return embedding_.context(state.embedding); }
    void restore_graph_context(State& state,NgramEmbedding::Context context) const { embedding_.restore(state.embedding,std::move(context)); }
    void restore_session(State& state,const DenseTensor& convolution,NgramEmbedding::Context context,bool retain_storage=false) const {
        TYLOINFER_RUNTIME_CHECK(convolution.defined(),"missing Qwen4 PLE session state");
        if(retain_storage) {
#ifdef TYLOINFER_NATIVE_CUDA
            TYLOINFER_RUNTIME_CHECK(state.conv.defined() && state.conv.dense().sizes()==convolution.sizes(),
#else
            TYLOINFER_RUNTIME_CHECK(state.conv.defined() && state.conv.sizes()==convolution.sizes(),
#endif
                "Qwen4 PLE graph session geometry changed");
            commit(state);
#ifdef TYLOINFER_NATIVE_CUDA
            state.conv.dense().copy_(convolution);
#else
            state.conv.copy_(convolution);
#endif
        } else {reset(state);state.conv=convolution.clone();}
        embedding_.restore(state.embedding,std::move(context));
    }
    std::shared_ptr<RowStage> graph_stage(State& state,int64_t tokens,const ::tyloi::Device& device) const {
        return embedding_.graph_stage(state.embedding,1,tokens,device);
    }
    void upload_graph(State& state,const DenseTensor& host_ids,RowStage& stage) const {embedding_.upload_graph(state.embedding,host_ids,stage);}
    NgramEmbedding::Context graph_prefix_context(const State& state,const DenseTensor& host_ids,int64_t count) const {
        return embedding_.prefix_context(state.embedding,host_ids,count);
    }
    void bind_graph_checkpoint(State& state,const std::vector<DenseTensor>& conv,
                               std::vector<NgramEmbedding::Context> context,int64_t position) const {
        TYLOINFER_RUNTIME_CHECK(!state.saved,"resolve previous PLE checkpoint");
        TYLOINFER_RUNTIME_CHECK(!conv.empty() && conv.size()==context.size(),"invalid PLE prefix checkpoints");
        state.saved=true;state.saved_conv=conv.front();state.saved_context=context.front();
        state.saved_prefix_conv=conv;state.saved_prefix_context=std::move(context);state.saved_prefix_position=position;
    }
    DenseTensor forward_graph(State& state,const DenseTensor& x,const DenseTensor& ids,
                         const DenseTensor& rows,int64_t confirmed,std::vector<DenseTensor>& checkpoint_conv) const {
        if(!confirmed)return forward_chunk(state,x,ids,true,{},rows);
        auto prefix=forward_chunk(state,x.narrow(1,0,confirmed),ids.narrow(1,0,confirmed),
            true,{},rows.narrow(1,0,confirmed));
#ifdef TYLOINFER_NATIVE_CUDA
        checkpoint_conv.front().copy_(state.conv.dense());
#else
        checkpoint_conv.front().copy_(state.conv);
#endif
        auto suffix=forward_chunk(state,x.narrow(1,confirmed,x.size(1)-confirmed),
            ids.narrow(1,confirmed,x.size(1)-confirmed),true,{},
            rows.narrow(1,confirmed,x.size(1)-confirmed),&checkpoint_conv,1);
        return ::tyloi::cuda::cat({prefix,suffix},1);
    }
    void prefetch_tokens(State& state,const DenseTensor& ids,bool early_prefix=false) const { embedding_.prefetch(state.embedding,ids,early_prefix); }
    DenseTensor forward(State& state, const DenseTensor &x, const DenseTensor &ids, bool cache,
                   int64_t confirmed = 0,const DenseTensor& host_ids = {},int64_t cache_position = -1) const {
        TYLOINFER_RUNTIME_CHECK(x.dim() == 3 && ids.dim() == 2 && x.size(0) == ids.size(0) &&
                              x.size(1) == ids.size(1) && x.size(2) == hidden_ * streams_ &&
                              confirmed >= 0 && confirmed <= x.size(1) && (!confirmed || cache),
                          "invalid Qwen4 PLE input");
        if(cache && cache_position>=0 && confirmed>0 && confirmed<x.size(1) && x.size(1)<=8) {
            TYLOINFER_RUNTIME_CHECK(!state.saved && state.conv.defined(),"PLE verification requires committed convolution state");
            auto host=host_ids.defined()?host_ids:ids.to(::tyloi::kCPU,::tyloi::kInt64);
            std::vector<DenseTensor> conv;
            std::vector<NgramEmbedding::Context> context;
            for(int64_t i=confirmed;i<x.size(1);++i) {
#ifdef TYLOINFER_NATIVE_CUDA
                conv.push_back(::tyloi::cuda::empty_like(state.conv.dense()));
#else
                conv.push_back(::tyloi::cuda::empty_like(state.conv));
#endif
                context.push_back(embedding_.prefix_context(state.embedding,host,i));
            }
            auto rows=embedding_.forward(state.embedding,ids,true,nullptr,host);
            auto result=forward_graph(state,x,ids,rows,confirmed,conv);
            bind_graph_checkpoint(state,conv,std::move(context),cache_position+confirmed);
            return result;
        }
        return tyloi::models::recurrent_window(
            x.size(1), cache, confirmed, state.saved,
            [&](int64_t start, int64_t count) {
                return forward_chunk(state, x.narrow(1, start, count),
                                     ids.narrow(1, start, count), cache,
                                     host_ids.defined() ? host_ids.narrow(1,start,count) : DenseTensor{});
            },
            [&] {
                state.saved = true;
#ifdef TYLOINFER_NATIVE_CUDA
                state.saved_conv = state.conv.dense().clone();
#else
                state.saved_conv = state.conv.clone();
#endif
                state.saved_context = embedding_.context(state.embedding);
            },
            [&] { rollback(state); },
            [](DenseTensor first, DenseTensor second) { return ::tyloi::cuda::cat({first, second}, 1); });
    }
    DenseTensor forward_chunk(State& state, const DenseTensor &x, const DenseTensor &ids,
                         bool cache,const DenseTensor& host_ids = {},const DenseTensor& graph_embeddings = {},
                         std::vector<DenseTensor>* checkpoints=nullptr,std::size_t checkpoint_offset=0) const {
        auto context = embedding_.context(state.embedding);
#ifdef TYLOINFER_NATIVE_CUDA
        auto conv = state.conv.defined() ? state.conv.dense() : DenseTensor{};
#else
        auto conv = state.conv;
#endif
        const auto b = x.size(0), t = x.size(1);
        const auto save_prefixes=[&](const DenseTensor& normalized) {
            if(!checkpoints)return;
            TYLOINFER_RUNTIME_CHECK(cache && conv.defined(),"PLE prefix history is unavailable");
            const auto lookback=conv.size(1);
            // The FIR state is a tail of normalized inputs. Recover every
            // intermediate tail while keeping key/value projections batched.
            for(int64_t count=1;count<t;++count) {
                auto& saved=checkpoints->at(checkpoint_offset+std::size_t(count-1));
                if(count>=lookback)saved.copy_(normalized.narrow(1,count-lookback,lookback));
                else {
                    saved.narrow(1,0,lookback-count).copy_(conv.narrow(1,count,lookback-count));
                    saved.narrow(1,lookback-count,count).copy_(normalized.narrow(1,0,count));
                }
            }
        };
#ifdef TYLOINFER_NATIVE_CUDA
        const auto native=::tyloi::cuda::options::ple_native_fusion();
        if((!native || *native) && x.is_contiguous() && t<=32 && hidden_>128 && hidden_<=8192 &&
            (x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32)) {
            try {
                auto embeddings=graph_embeddings.defined()?graph_embeddings:
                    embedding_.forward(state.embedding,ids,cache,nullptr,host_ids);
#ifdef TYLOINFER_NATIVE_CUDA
                auto key=mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(embeddings,w_.key).dense(),w_.key_norm.dense(),hidden_,eps_);
                auto query=mfq_qwen4_exp::grouped_rms_norm(x,w_.query_norm.dense(),hidden_,eps_);
#else
                auto key=mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(embeddings,w_.key).dense(),w_.key_norm,hidden_,eps_);
                auto query=mfq_qwen4_exp::grouped_rms_norm(x,w_.query_norm,hidden_,eps_);
#endif
                auto value=::tyloi::ops::project(embeddings,w_.value).dense();
                auto prepared=mfq_qwen4_exp::ple_gate_norm_cuda(key,query,value,
#ifdef TYLOINFER_NATIVE_CUDA
                    w_.conv_norm.dense().to(::tyloi::kFloat32).contiguous(),hidden_,streams_,eps_);
#else
                    w_.conv_norm.to(::tyloi::kFloat32).contiguous(),hidden_,streams_,eps_);
#endif
                save_prefixes(prepared[1]);
#ifdef TYLOINFER_NATIVE_CUDA
                auto output=mfq_qwen4_exp::ple_conv_add_cuda(prepared[1],w_.conv.dense(),
#else
                auto output=mfq_qwen4_exp::ple_conv_add_cuda(prepared[1],w_.conv,
#endif
                    cache && conv.defined()?std::optional<DenseTensor>(conv):std::nullopt,dilation_,prepared[0],
                    x.scalar_type()==::tyloi::kFloat16);
                if(cache) {if(conv.defined())conv.copy_(output[1]);else state.conv=output[1];}
                return output[0];
            }catch(...) {embedding_.restore(state.embedding,std::move(context));throw;}
        }
#endif
        return tyloi::models::qwen4_exp::position_embedding(
            cache, [&] { return graph_embeddings.defined() ? graph_embeddings : embedding_.forward(state.embedding,ids, cache,nullptr,host_ids); },
            [&](const DenseTensor &embeddings) {
                auto key = mfq_qwen4_exp::grouped_rms_norm(::tyloi::ops::project(embeddings,w_.key).dense(),
#ifdef TYLOINFER_NATIVE_CUDA
                                                           w_.key_norm.dense(), hidden_, eps_)
#else
                                                           w_.key_norm, hidden_, eps_)
#endif
                               .reshape({b, t, streams_, hidden_});
                return key;
            },
            [&] {
#ifdef TYLOINFER_NATIVE_CUDA
                auto query = mfq_qwen4_exp::grouped_rms_norm(x, w_.query_norm.dense(), hidden_, eps_)
#else
                auto query = mfq_qwen4_exp::grouped_rms_norm(x, w_.query_norm, hidden_, eps_)
#endif
                                 .reshape({b, t, streams_, hidden_});
                return query;
            },
            [&](const DenseTensor &key, const DenseTensor &query, const DenseTensor &embeddings) {
                auto score = (key.to(::tyloi::kFloat32) * query.to(::tyloi::kFloat32)).sum(-1) /
                             std::sqrt(double(hidden_));
                auto sign = (score > 0).to(::tyloi::kFloat32) - (score < 0).to(::tyloi::kFloat32);
                auto root = sign * ::tyloi::cuda::clamp_min(score.abs(), 1e-6).sqrt();
                auto gated = (::tyloi::cuda::sigmoid(root).unsqueeze(-1) *
                              ::tyloi::ops::project(embeddings,w_.value).dense().to(::tyloi::kFloat32).unsqueeze(-2))
                                 .reshape({b, t, streams_ * hidden_});
                return gated;
            },
            [&](const DenseTensor &gated) {
#ifdef TYLOINFER_NATIVE_CUDA
                return mfq_qwen4_exp::grouped_rms_norm(gated, w_.conv_norm.dense(), hidden_, eps_);
#else
                return mfq_qwen4_exp::grouped_rms_norm(gated, w_.conv_norm, hidden_, eps_);
#endif
            },
            [&](DenseTensor normalized) {
                save_prefixes(normalized);
                return mfq_qwen4_exp::ple_dilated_conv_silu(
#ifdef TYLOINFER_NATIVE_CUDA
                    normalized, w_.conv.dense(),
#else
                    normalized, w_.conv,
#endif
                    cache && conv.defined() ? std::optional<DenseTensor>(conv) : std::nullopt,
                    dilation_);
            },
            [&](DenseTensor gated, DenseTensor convolved) { return (gated + convolved).to(x.scalar_type()); },
            [&](DenseTensor next) { if(conv.defined())conv.copy_(next);else state.conv=std::move(next); },
            [&] { embedding_.restore(state.embedding,std::move(context)); });
    }

  private:
    NgramEmbedding embedding_;
    PleWeights w_;
    int64_t hidden_, streams_, dilation_;
    double eps_;
};

struct QsaWeights {
    std::vector<::tyloi::Tensor> input_projection;
    ::tyloi::Tensor output;
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor query_norm, key_norm, index_query_norm, index_key_norm;
#else
    DenseTensor query_norm, key_norm, index_query_norm, index_key_norm;
#endif
    bool has_gate = false;
};
struct QsaConfig {
    int64_t heads, kv_heads, width, index_heads, index_width, pool, budget, maximum;
    double eps;
};

class Qsa {
  public:
    struct State {
        explicit State(const Qsa& model)
            : keys(model.c_.maximum, model.c_.kv_heads * model.key_width(), model.cache_dtype()),
              values(model.c_.maximum, model.c_.kv_heads * model.value_width(), model.cache_dtype()),
              index(model.c_.maximum, model.c_.index_width), offload(model.offload_) {
            if(model.memory_manager_ && !offload)
                memory=model.memory_manager_->add(model.c_.maximum,model.c_.kv_heads,model.key_width(),model.value_width(),
                    model.c_.index_width,model.c_.pool,model.w_->query_norm.options().dtype(model.cache_dtype()),
                    model.c_.maximum<=model.c_.budget?model.c_.maximum:model.c_.budget+model.c_.pool-1);
            if(offload) {
                offloaded_kv=std::make_unique<KvOffloadSequence>(offload,model.c_.kv_heads*(model.key_width()+model.value_width()),model.c_.pool,model.cache_dtype());
                offloaded_index=std::make_unique<KvOffloadSequence>(offload,model.c_.index_width,model.c_.pool);
                offloaded_pooled=std::make_unique<KvOffloadSequence>(offload,model.c_.index_width,16);
            }
        }
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        State(State&&) = default;
        State& operator=(State&&) = default;
        SequenceCache keys,values,index;
        std::shared_ptr<KvOffloadStore> offload;
        std::unique_ptr<KvOffloadSequence> offloaded_kv,offloaded_index,offloaded_pooled;
        ::tyloi::Tensor graph_keys,graph_values,graph_pooled,graph_index;
        std::shared_ptr<KvMemoryAllocation> memory;
        bool graph_prepared=false;
    };
    Qsa(std::shared_ptr<const QsaWeights> weights, QsaConfig config, std::shared_ptr<RotaryEmbedding> rotary,
        bool fused_projection, bool grouped_projection,
        std::shared_ptr<KvOffloadStore> offload, std::shared_ptr<KvMemoryManager> memory)
        : w_(std::move(weights)), c_(config), rotary_(std::move(rotary)), fused_projection_(fused_projection), grouped_projection_(grouped_projection),
          offload_(std::move(offload)), memory_manager_(std::move(memory)) {
        TYLOINFER_RUNTIME_CHECK(c_.heads > 0 && c_.kv_heads > 0 && c_.heads % c_.kv_heads == 0 &&
                              c_.width > 0 && c_.index_heads > 0 && c_.pool > 0 && c_.budget > 0 &&
                              c_.budget % c_.pool == 0 && rotary_,
                          "invalid Qwen4 QSA configuration");
    }
    int64_t position(const State& request) const { return request.offload && request.offloaded_kv ? request.offloaded_kv->position() : request.keys.position(); }
    bool managed_memory(const State& request) const {return request.memory && request.memory->ready;}
    int64_t graph_capacity(const State& request) const {return managed_memory(request)?c_.maximum:0;}
    std::vector<::tyloi::Tensor*> graph_state(State& request) const {
        return {&request.graph_keys,&request.graph_values,&request.graph_pooled,&request.graph_index};
    }
    void activate_memory(State& request) {
        if(!managed_memory(request))return;
        if(!request.index.storage().defined())request.index.replace_storage(request.memory->index);
        request.graph_index=request.index.storage();
        request.graph_pooled=request.memory->pooled;
        if(request.memory->offload) {
            request.offload=request.memory->offload;
            if(!request.offloaded_kv)request.offloaded_kv=std::make_unique<KvOffloadSequence>(request.offload,
                c_.kv_heads*(key_width()+value_width()),c_.pool,cache_dtype());
        } else {request.graph_keys=request.memory->keys;request.graph_values=request.memory->values;}
    }
    std::vector<DenseTensor> project_inputs(const DenseTensor& x) const {
        std::vector<DenseTensor> projected;
        const bool grouped=grouped_projection_ && ::tyloi::ops::grouped(w_->input_projection);
        if(grouped)
            { auto values=::tyloi::ops::project_group(x,w_->input_projection,1); for(const auto& value:values)projected.push_back(value.dense()); }
        else
            projected={project_query(x),
                ::tyloi::ops::project(x,w_->input_projection[1]).dense(),
                ::tyloi::ops::project(x,w_->input_projection[2]).dense(),
                ::tyloi::ops::project(x,w_->input_projection[3]).dense()};
        if(grouped && w_->has_gate) {
            TYLOINFER_RUNTIME_CHECK(projected.size()==5,"QSA input projection count mismatch");
            const auto b=x.size(0),t=x.size(1);
            projected[0]=::tyloi::cuda::cat({projected[0].reshape({b,t,c_.heads,c_.width}),
                projected[4].reshape({b,t,c_.heads,c_.width})},-1).reshape({b,t,2*c_.heads*c_.width});
            projected.erase(projected.begin()+4);
        }
        TYLOINFER_RUNTIME_CHECK(projected.size()==4,"QSA input projection count mismatch");
        return projected;
    }
    DenseTensor project_output(const DenseTensor& x) const {
        return ::tyloi::ops::project(x,w_->output).dense();
    }
    DenseTensor project_query(const DenseTensor& x) const {
        auto query=::tyloi::ops::project(x,w_->input_projection[0]).dense();
        if(!w_->has_gate)return query;
        const auto b=x.size(0),t=x.size(1);
        auto gate=::tyloi::ops::project(x,w_->input_projection[4]).dense();
        return ::tyloi::cuda::cat({query.reshape({b,t,c_.heads,c_.width}),
            gate.reshape({b,t,c_.heads,c_.width})},-1).reshape({b,t,2*c_.heads*c_.width});
    }
    std::size_t growth_bytes(const State& request,int64_t tokens) const {
        if(managed_memory(request))return request.memory->growth_bytes(position(request)+tokens);
        if (request.offload) return 0;
        const auto required = position(request) + tokens;
        TYLOINFER_RUNTIME_CHECK(tokens > 0 && required <= c_.maximum, "QSA growth exceeds max_context");
        std::size_t bytes = 0;
        for (const auto* cache : {&request.keys, &request.values, &request.index}) {
            const auto& storage = cache->storage();
            const int64_t current = request.graph_keys.defined() && cache != &request.index
                ? request.graph_keys.size(2) : (storage.defined() ? storage.size(1) : 0);
            if (required <= current) continue;
            int64_t capacity = current ? current : std::min<int64_t>(16, c_.maximum);
            while (capacity < required)
                capacity = std::min(c_.maximum, capacity > c_.maximum / 2 ? c_.maximum : capacity * 2);
            const auto width = cache == &request.index ? c_.index_width : c_.kv_heads * (cache == &request.keys ? key_width() : value_width());
            bytes += std::size_t(capacity) * width * (cache != &request.index && quantization_.enabled() ? 4 : 2);
        }
        return bytes;
    }
    std::size_t graph_growth_bytes(const State& request,int64_t capacity) const {
        if(managed_memory(request))return request.memory->growth_bytes(capacity);
        if (request.graph_keys.defined()) return 0;
        auto bytes = std::size_t(capacity) * c_.kv_heads * (key_width()+value_width()) * (quantization_.enabled()?4:2);
        if (!request.index.storage().defined() || request.index.storage().size(1) != capacity)
            bytes += std::size_t(capacity) * c_.index_width * 2;
        bytes += std::size_t((capacity+c_.pool-1)/c_.pool) * c_.index_width * 2;
        return bytes;
    }
    bool supports_session(const State& request) const { return !request.offload && !(request.memory && request.memory->offload); }
#ifdef TYLOINFER_NATIVE_CUDA
    /** Validate a QSA snapshot against this layer's cache geometry and device. */
    void validate_session(const State& request, const tyloi::models::qwen4_exp::LayerSnapshot& state,
                          int64_t count, tyloi::Device device) const {
        const auto check = [&](const tyloi::Tensor& tensor, int64_t width, tyloi::ScalarType type) {
            return tensor.defined() && tensor.representation<DenseTensor>() && tensor.device()==device &&
                tensor.scalar_type()==type && tensor.dim()==3 && tensor.size(0)==1 &&
                tensor.size(1)==count && tensor.size(2)==width;
        };
        TYLOINFER_RUNTIME_CHECK(supports_session(request) && count>0 && count<=c_.maximum &&
            check(state.keys,c_.kv_heads*key_width(),cache_dtype()) &&
            check(state.values,c_.kv_heads*value_width(),cache_dtype()) &&
            check(state.index,c_.index_width,::tyloi::kFloat16), "incompatible Qwen4 QSA session state");
    }
    /** Estimate fresh unmanaged sequence backing; managed allocation has its own admission. */
    std::size_t fresh_session_growth(const State& current, int64_t count) const {
        if (memory_manager_) {
            TYLOINFER_RUNTIME_CHECK(managed_memory(current), "Qwen4 managed session storage is not ready");
            return 0;
        }
        SequenceCache keys(c_.maximum,c_.kv_heads*key_width(),cache_dtype());
        SequenceCache values(c_.maximum,c_.kv_heads*value_width(),cache_dtype());
        SequenceCache index(c_.maximum,c_.index_width);
        const auto a=keys.growth(1,count), b=values.growth(1,count), c=index.growth(1,count);
        if (a>std::numeric_limits<std::size_t>::max()-b || a+b>std::numeric_limits<std::size_t>::max()-c)
            throw std::overflow_error("Qwen4 session allocation planning overflow");
        return a+b+c;
    }
    /** Stage a snapshot into independently owned sequence or managed KV storage. */
    void stage_session(State& request, const tyloi::models::qwen4_exp::LayerSnapshot& saved,
                       int64_t count) {
        if (managed_memory(request)) { restore_session(request,saved,count); return; }
        const auto stage = [&](SequenceCache& cache, const ::tyloi::Tensor& source) {
            int64_t capacity=std::min<int64_t>(16,c_.maximum);
            while(capacity<count)
                capacity=std::min(c_.maximum,capacity>c_.maximum/2?c_.maximum:capacity*2);
            auto shape=source.dense().sizes().vec(); shape[1]=capacity;
            cache.replace_storage(tyloi::ops::empty(shape,source.options()));
            auto view=tyloi::ops::slice(cache.storage(),1,0,count);
            tyloi::ops::copy_into(view,source);
            cache.advance_fixed(count);
        };
        stage(request.keys,saved.keys);
        stage(request.values,saved.values);
        stage(request.index,saved.index);
    }
#endif
    std::size_t session_bytes(const State& request,int64_t count) const {
        if(count<=0)return 0;
        return std::size_t(count)*(c_.kv_heads*(key_width()+value_width())*(quantization_.enabled()?4:2)+
            c_.index_width*request.index.storage().element_size());
    }
    void capture_session(const State& request,tyloi::models::qwen4_exp::LayerSnapshot& state,int64_t count,std::size_t& bytes) const {
        TYLOINFER_RUNTIME_CHECK(!request.offload && count>0 && count<=position(request),"invalid Qwen4 QSA session boundary");
        const auto save=[&](const DenseTensor& tensor) {
            auto saved=std::make_shared<DenseTensor>(tensor.contiguous().clone());bytes+=session_tensor_bytes(*saved);return saved;
        };
        if(managed_memory(request) && request.memory->hot.exclusive()) {
            auto pair=request.memory->hot.gather(count);
            state.keys=save(pair.first.permute({0,2,1,3}).reshape({1,count,c_.kv_heads*key_width()}));
            state.values=save(pair.second.permute({0,2,1,3}).reshape({1,count,c_.kv_heads*value_width()}));
        } else if(request.graph_keys.defined()) {
            state.keys=save((request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}).narrow(2,0,count).permute({0,2,1,3}).reshape({1,count,c_.kv_heads*key_width()}));
            state.values=save((request.graph_values.defined()?request.graph_values.dense():DenseTensor{}).narrow(2,0,count).permute({0,2,1,3}).reshape({1,count,c_.kv_heads*value_width()}));
        } else {
            state.keys=save(request.keys.storage().dense().narrow(1,0,count));
            state.values=save(request.values.storage().dense().narrow(1,0,count));
        }
        state.index=save(request.index.storage().dense().narrow(1,0,count));
    }
    void restore_session(State& request,const tyloi::models::qwen4_exp::LayerSnapshot& state,int64_t count) {
        const auto* keys=state.keys.representation<DenseTensor>();
        const auto* values=state.values.representation<DenseTensor>();
        const auto* index=state.index.representation<DenseTensor>();
        TYLOINFER_RUNTIME_CHECK(!request.offload && keys && values && index && keys->defined() && values->defined() && index->defined() &&
            keys->dim()==3 && keys->size(0)==1 && keys->size(1)==count &&
            keys->size(2)==c_.kv_heads*key_width() && keys->scalar_type()==cache_dtype() &&
            values->dim()==3 && values->size(0)==1 && values->size(1)==count &&
            values->size(2)==c_.kv_heads*value_width() && values->scalar_type()==cache_dtype() &&
            index->dim()==3 && index->size(0)==1 && index->size(1)==count && index->size(2)==c_.index_width,
            "incompatible Qwen4 QSA session state");
        reset(request);activate_memory(request);
        if(managed_memory(request)) {
            request.memory->ensure(count);
            auto k=keys->reshape({1,count,c_.kv_heads,key_width()}).permute({0,2,1,3});
            auto v=values->reshape({1,count,c_.kv_heads,value_width()}).permute({0,2,1,3});
            auto positions=::tyloi::cuda::arange(count,request.graph_keys.dense().options().dtype(::tyloi::kInt64));
            if(request.memory->hot.exclusive())request.memory->hot.append(k,v,positions);
            else {
                (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}).narrow(2,0,count).copy_(k);(request.graph_values.defined()?request.graph_values.dense():DenseTensor{}).narrow(2,0,count).copy_(v);
                request.memory->hot.publish((request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}),(request.graph_values.defined()?request.graph_values.dense():DenseTensor{}),positions);
            }
            request.keys.advance_fixed(count);request.values.advance_fixed(count);request.index.append(*index);request.graph_prepared=false;
        } else {request.keys.append(*keys);request.values.append(*values);request.index.append(*index);}
    }
    /** Return owning handles for the request's unmanaged resident cache. */
    std::vector<::tyloi::Tensor> resident_cache_tensors(const State& request) const {
        if(managed_memory(request))return {}; // Count physical GPU/RAM tiers through KvMemoryManager.
        return {request.keys.storage(), request.values.storage(), request.index.storage(),
            request.graph_keys, request.graph_values, request.graph_pooled};
    }
    void leave_graph(State& request) {
        if(managed_memory(request)){request.graph_prepared=false;return;}
        if(!request.graph_keys.defined())return;
        const auto n=position(request);
        request.keys.replace_storage((request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{}).narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*key_width()}));
        request.values.replace_storage((request.graph_values.defined()?request.graph_values.dense():DenseTensor{}).narrow(2,0,n).permute({0,2,1,3}).contiguous().reshape({1,n,c_.kv_heads*value_width()}));
        request.graph_keys={};request.graph_values={};request.graph_pooled={};request.graph_index={};
    }
    void reset(State& request) {
        if (request.offload) {
            if(request.offloaded_kv)request.offloaded_kv->reset();
            if(request.offloaded_index)request.offloaded_index->reset();
            if(request.offloaded_pooled)request.offloaded_pooled->reset();
        }
        request.keys.reset();
        request.values.reset();
        request.index.reset();
        request.graph_prepared=false;
        if(managed_memory(request)){request.memory->hot.reset();activate_memory(request);return;}
        request.graph_keys={};request.graph_values={};request.graph_pooled={};request.graph_index={};
    }
    void prepare_graph(State& request,const DenseTensor& full_positions, int64_t capacity) {
        activate_memory(request);
        TYLOINFER_RUNTIME_CHECK(!request.offload || managed_memory(request), "streamed QSA graphs require a resident indexer");
        if(managed_memory(request)) {
            if(request.graph_prepared)return;
            const auto n=position(request),pools=n/c_.pool;
            request.memory->ensure(n);
            if(pools) {
                auto pooled=request.index.storage().dense().narrow(1,0,pools*c_.pool).reshape({1,pools,c_.pool,c_.index_width}).to(::tyloi::kFloat32).mean(2).to(request.index.storage().scalar_type());
#ifdef TYLOINFER_NATIVE_CUDA
                pooled=rms_norm(pooled,w_->index_key_norm.dense().to(::tyloi::kFloat32)+1,c_.eps);
#else
                pooled=rms_norm(pooled,w_->index_key_norm.to(::tyloi::kFloat32)+1,c_.eps);
#endif
                auto starts=::tyloi::cuda::arange(pools,full_positions.options().dtype(::tyloi::kInt64))*c_.pool;
                (request.graph_pooled.defined()?request.graph_pooled.dense():DenseTensor{}).narrow(1,0,pools).copy_(rotary_->forward(pooled.unsqueeze(1),full_positions.index_select(-1,starts)).squeeze(1));
            }
            request.graph_prepared=true;return;
        }
        if(request.graph_keys.defined())return;
        TYLOINFER_RUNTIME_CHECK(position(request)>0 && request.keys.storage().size(0)==1,
            "QSA graph requires a prefilled single sequence");
        const auto n=position(request);
        TYLOINFER_RUNTIME_CHECK(capacity>=n && capacity<=c_.maximum,"invalid QSA graph capacity");
        auto graph_keys=::tyloi::cuda::zeros({1,c_.kv_heads,capacity,key_width()},request.keys.storage().options());
        auto graph_values=::tyloi::cuda::zeros({1,c_.kv_heads,capacity,value_width()},request.values.storage().options());
        graph_keys.narrow(2,0,n).copy_(request.keys.storage().dense().narrow(1,0,n).reshape({1,n,c_.kv_heads,key_width()}).permute({0,2,1,3}));
        graph_values.narrow(2,0,n).copy_(request.values.storage().dense().narrow(1,0,n).reshape({1,n,c_.kv_heads,value_width()}).permute({0,2,1,3}));
        request.index.prepare_fixed(capacity);
        request.graph_index=request.index.storage();
        auto graph_pooled=::tyloi::cuda::zeros({1,(capacity+c_.pool-1)/c_.pool,c_.index_width},request.index.storage().options());
        const auto pools=n/c_.pool;
        if(pools) {
            auto pooled=request.index.storage().dense().narrow(1,0,pools*c_.pool).reshape({1,pools,c_.pool,c_.index_width}).to(::tyloi::kFloat32).mean(2).to(request.index.storage().scalar_type());
#ifdef TYLOINFER_NATIVE_CUDA
            pooled=rms_norm(pooled,w_->index_key_norm.dense().to(::tyloi::kFloat32)+1,c_.eps);
#else
            pooled=rms_norm(pooled,w_->index_key_norm.to(::tyloi::kFloat32)+1,c_.eps);
#endif
            auto starts=::tyloi::cuda::arange(pools,full_positions.options().dtype(::tyloi::kInt64))*c_.pool;
            graph_pooled.narrow(1,0,pools).copy_(rotary_->forward(pooled.unsqueeze(1),full_positions.index_select(-1,starts)).squeeze(1));
        }
        // Commit only after all allocation/copy submissions succeed. Replay
        // reads graph KV; retaining the old layout duplicates the full prefix
        // at every growth boundary. leave_graph recreates it when needed.
        request.graph_keys=std::move(graph_keys);request.graph_values=std::move(graph_values);request.graph_pooled=std::move(graph_pooled);
        request.keys.release_storage();request.values.release_storage();
    }
    DenseTensor forward_graph(State& request,const DenseTensor& hidden,
                         const DenseTensor& positions,const DenseTensor& cache_positions,
                         std::vector<DenseTensor> projections={},bool project_output=true,bool history_only=false) {
        const auto mark=[](const char* phase) {
            if(auto* window=DecodeWindow::recording())window->mark_current(phase);
        };
        const bool streamed=bool(request.offload);
        TYLOINFER_RUNTIME_CHECK((request.graph_keys.defined() || (streamed && managed_memory(request))) && hidden.size(0)==1 && cache_positions.dim()==1,
            "QSA graph cache is not prepared");
        auto graph_keys = (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{});
        auto graph_values = (request.graph_values.defined()?request.graph_values.dense():DenseTensor{});
        auto graph_pooled = (request.graph_pooled.defined()?request.graph_pooled.dense():DenseTensor{});
        const auto t=hidden.size(1),pools=request.graph_pooled.size(1);
        const bool dense_capacity=(streamed?c_.maximum:request.graph_keys.size(2))<=c_.budget;
        const bool exclusive=managed_memory(request) && request.memory->hot.exclusive();
        auto p=history_only?project_history(hidden,positions):
            project(hidden,positions,!dense_capacity,cache_positions,
                       quantization_.enabled() || exclusive || streamed?DenseTensor{}:graph_keys,quantization_.enabled() || exclusive || streamed?DenseTensor{}:graph_values,
                       std::move(projections));
        mark("qsa_input_end");
        const auto finish=[&](DenseTensor output) {
            mark("qsa_attention_end");
            output=output.reshape({1,t,c_.heads*c_.width});
            return project_output?this->project_output(output):output;
        };
        bool index_written=false;
        const auto* prepare_option=tyloi::cuda::env_lookup("TYLOINFER_KV_PREPARE_FUSED");
        const bool begin_prepare=!prepare_option || prepare_option[0]!='0';
        if(exclusive) {
            const auto keys=(quantization_.enabled()?encode_key(p.key):p.key).permute({0,2,1,3});
            const auto values=(quantization_.enabled()?encode_value(p.value):p.value).permute({0,2,1,3});
            index_written=request.memory->hot.append_decode(keys,values,cache_positions,p.raw,request.index.storage().dense(),begin_prepare);
            if(!index_written)request.memory->hot.append(keys.to(cache_dtype()),values.to(cache_dtype()),cache_positions);
        }
        else if (!p.cache_written && !streamed) {
            graph_keys.index_copy_(2,cache_positions,encode_key(p.key).permute({0,2,1,3}));
            graph_values.index_copy_(2,cache_positions,encode_value(p.value).permute({0,2,1,3}));
        }
        if(managed_memory(request) && !exclusive && !streamed)request.memory->hot.publish(graph_keys,graph_values,cache_positions);
        if(!index_written)request.index.append_fixed(p.raw,cache_positions);
        mark("qsa_cache_end");
        if(history_only && streamed) {
            auto* window=DecodeWindow::recording();
            TYLOINFER_RUNTIME_CHECK(window,"streamed QSA history requires a decode window");
            struct HistoryStage {DenseTensor raw;};
            const auto stage=window->workspace<HistoryStage>(&request,[&] {
                ResidentAllocationScope resident;
                auto value=std::make_shared<HistoryStage>();
                value->raw=::tyloi::cuda::empty({t,c_.kv_heads*(key_width()+value_width())},hidden.options().dtype(cache_dtype()));
                return value;
            });
            stage->raw.copy_(::tyloi::cuda::cat({encode_key(p.key).reshape({t,c_.kv_heads*key_width()}),
                encode_value(p.value).reshape({t,c_.kv_heads*value_width()})},1));
            window->host_barrier([this,&request,stage] {request.offloaded_kv->append(stage->raw);});
        }
        if(dense_capacity) {
            if(history_only)return {};
            if(streamed) {
                auto ids=::tyloi::cuda::arange(c_.maximum,cache_positions.options()).reshape({1,1,c_.maximum}).expand({1,t,c_.maximum});
                ids=::tyloi::cuda::where(ids<=cache_positions.reshape({1,t,1}),ids,::tyloi::cuda::full_like(ids,-1)).to(::tyloi::kInt32).contiguous();
                return finish(streamed_graph_attention(request,p,ids,hidden.scalar_type()));
            }
            const auto columns=std::min(request.graph_keys.size(2),c_.budget+c_.pool-1);
            const auto hot=prepare_hot(request,{},cache_positions,columns,index_written && begin_prepare);
            if(quantization_.enabled()) {
                auto attended=cuda_kv_attention(p.query,graph_keys,graph_values,quantization_,{},cache_positions,0,{},0,{},{},0,hot);
                return finish((attended*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(hidden.scalar_type()));
            }
            if(fused_projection_ && rotary_->fused() &&
                (hidden.scalar_type()==::tyloi::kFloat16 || hidden.scalar_type()==::tyloi::kFloat32)) {
                auto gated=mfq_qwen4_exp::causal_gqa_attention_gate(p.query,graph_keys,graph_values,
                    cache_positions,p.gate,columns,hidden.scalar_type()==::tyloi::kFloat16,hot);
                return finish(std::move(gated));
            }
            auto absolute=cache_positions.reshape({1,t,1});
            auto indices=::tyloi::cuda::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
            indices=::tyloi::cuda::where(indices<=absolute,indices,::tyloi::cuda::full_like(indices,-1)).to(::tyloi::kInt32).contiguous();
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys,graph_values,indices,hot);
            auto gated=attended.to(::tyloi::kFloat32)*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
            return finish(gated.to(hidden.scalar_type()));
        }
        DenseTensor block,pooled;
        const auto* pool_fusion=tyloi::cuda::env_lookup("TYLOINFER_QSA_POOL_FUSED");
        const auto* pool_write=tyloi::cuda::env_lookup("TYLOINFER_QSA_POOL_WRITE_FUSED");
        const bool pool_written=(!pool_fusion || pool_fusion[0]!='0') && (!pool_write || pool_write[0]!='0') &&
#ifdef TYLOINFER_NATIVE_CUDA
            request.index.storage().size(1)>=c_.maximum && rotary_->pool_normalized_into(request.index.storage().dense(),w_->index_key_norm.dense(),
#else
            request.index.storage().size(1)>=c_.maximum && rotary_->pool_normalized_into(request.index.storage().dense(),w_->index_key_norm,
#endif
                positions,cache_positions,graph_pooled,c_.pool,c_.maximum,c_.eps);
        if(!pool_written && rotary_->fused() && (!pool_fusion || pool_fusion[0]!='0') && request.index.storage().size(1)>=c_.maximum) {
            auto prepared=mfq_qwen4_exp::live_index_pool(request.index.storage().dense(),positions,cache_positions,c_.pool,c_.maximum);
            block=prepared[1];
#ifdef TYLOINFER_NATIVE_CUDA
            pooled=rotary_->forward_normalized(prepared[0].unsqueeze(2),w_->index_key_norm.dense(),prepared[2],c_.eps).squeeze(1);
#else
            pooled=rotary_->forward_normalized(prepared[0].unsqueeze(2),w_->index_key_norm,prepared[2],c_.eps).squeeze(1);
#endif
        }else if(!pool_written) {
            block=(cache_positions/c_.pool).to(::tyloi::kInt64);
            auto row=(block.unsqueeze(-1)*c_.pool+::tyloi::cuda::arange(c_.pool,block.options())).clamp(0,c_.maximum-1);
            pooled=request.index.storage().dense().index_select(1,row.reshape({-1})).reshape({1,t,c_.pool,c_.index_width}).to(::tyloi::kFloat32).mean(2).to(request.index.storage().scalar_type());
#ifdef TYLOINFER_NATIVE_CUDA
            pooled=rms_norm(pooled,w_->index_key_norm.dense().to(::tyloi::kFloat32)+1,c_.eps);
#else
            pooled=rms_norm(pooled,w_->index_key_norm.to(::tyloi::kFloat32)+1,c_.eps);
#endif
            auto delta=positions-cache_positions;
            auto block_positions=block*c_.pool+delta;
            pooled=rotary_->forward(pooled.unsqueeze(1),block_positions).squeeze(1);
        }
        if(!pool_written)graph_pooled.index_copy_(1,block,pooled);
        mark("qsa_pool_end");
        if(history_only)return {};
        auto ranked=mfq_qwen4_exp::live_block_scores(p.iq,graph_pooled,cache_positions,c_.pool);
        mark("qsa_scores_end");
        auto absolute=cache_positions.reshape({1,t,1});
        const auto count=std::min(c_.budget/c_.pool,pools),columns=c_.budget+c_.pool-1;
        auto indices=mfq_qwen4_exp::live_topk_blocks(ranked,cache_positions,c_.pool,count);
        mark("qsa_topk_end");
        DenseTensor selected;
        const auto fused_selection=::tyloi::cuda::options::qsa_select_fused();
        if(!fused_selection || *fused_selection) {
            selected=mfq_qwen4_exp::qsa_selected_tokens(indices,cache_positions,c_.pool,c_.budget);
        } else {
        // Live top-k pads inactive slots with an out-of-range block ID.
        // Check positions directly rather than gathering a visibility table
        // with those sentinels when the unfused selector is requested.
        auto valid=((indices>=0)&(indices*c_.pool+c_.pool-1<=absolute)).unsqueeze(-1).expand({1,t,count,c_.pool});
        auto expanded=indices.unsqueeze(-1)*c_.pool+::tyloi::cuda::arange(c_.pool,indices.options());
        expanded=::tyloi::cuda::where(valid,expanded,::tyloi::cuda::full_like(expanded,-1));
        selected=::tyloi::cuda::full({1,t,columns},-1,cache_positions.options());
        selected.narrow(-1,0,count*c_.pool).copy_(expanded.reshape({1,t,count*c_.pool}));
        if(c_.pool>1) {
            auto tail_count=(absolute+1).remainder(c_.pool);
            auto offsets=::tyloi::cuda::arange(c_.pool-1,cache_positions.options()).reshape({1,1,c_.pool-1});
            auto tail=absolute+1-tail_count+offsets;
            selected.narrow(-1,c_.budget,c_.pool-1).copy_(::tyloi::cuda::where(offsets<tail_count,tail,::tyloi::cuda::full_like(tail,-1)));
        }
        // Dense-budget queries select their complete causal prefix. The same
        // native selected-attention kernel serves both paths inside one graph.
        auto dense=::tyloi::cuda::arange(columns,cache_positions.options()).reshape({1,1,columns}).expand({1,t,columns});
        dense=::tyloi::cuda::where(dense<=absolute,dense,::tyloi::cuda::full_like(dense,-1));
        selected=::tyloi::cuda::where(absolute+1<=c_.budget,dense,selected).to(::tyloi::kInt32).contiguous();
        }
        mark("qsa_select_end");
        if(streamed)return finish(streamed_graph_attention(request,p,selected,hidden.scalar_type()));
        DenseTensor gated;
        const auto hot=prepare_hot(request,selected,cache_positions,0,index_written && begin_prepare);
        mark("qsa_hot_end");
        if(quantization_.enabled()) {
            auto attended=cuda_kv_attention(p.query,graph_keys,graph_values,quantization_,selected,cache_positions,0,{},0,{},{},0,hot);
            gated=(attended*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(hidden.scalar_type());
        } else if(fused_projection_ && rotary_->fused() &&
            (hidden.scalar_type()==::tyloi::kFloat16 || hidden.scalar_type()==::tyloi::kFloat32)) {
            const auto sparse_gate=::tyloi::cuda::options::qsa_sparse_gate_fused();
            if(!sparse_gate || *sparse_gate)
                gated=mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,graph_keys,graph_values,selected,
                    p.gate,hidden.scalar_type()==::tyloi::kFloat16,hot);
            else gated=mfq_qwen4_exp::attention_gate(
                mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys,graph_values,selected,hot),
                p.gate,hidden.scalar_type()==::tyloi::kFloat16);
        } else {
            auto attended=mfq_qwen4_exp::sparse_gqa_attention(p.query,graph_keys,graph_values,selected,hot);
            gated=(attended.to(::tyloi::kFloat32)*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(hidden.scalar_type());
        }
        return finish(std::move(gated));
    }
    void advance_graph(State& request,int64_t tokens) {
        if(!request.offload){request.keys.advance_fixed(tokens);request.values.advance_fixed(tokens);}
        request.index.advance_fixed(tokens);
    }
    void truncate(State& request,int64_t keep) {
        if (request.offload) {
            TYLOINFER_RUNTIME_CHECK(keep >= 0 && keep <= position(request), "invalid offloaded QSA truncation");
            request.offloaded_kv->truncate(keep);
            if(managed_memory(request))request.index.truncate(keep);
            else {request.offloaded_index->truncate(keep);request.offloaded_pooled->truncate(keep/c_.pool);}
            return;
        }
        TYLOINFER_RUNTIME_CHECK(keep >= 0 && keep <= request.keys.position() && keep <= request.values.position() &&
                              keep <= request.index.position(),
                          "invalid Qwen4 QSA cache truncation");
        request.keys.truncate(keep);
        request.values.truncate(keep);
        request.index.truncate(keep);
    }
    DenseTensor forward(State& request, const DenseTensor &hidden,
                   const DenseTensor &current_positions, const DenseTensor &full_positions, bool use_cache,
                   std::vector<DenseTensor> *selection_trace = nullptr,
                   std::vector<DenseTensor> projections = {}, bool project_output = true,
                   bool history_only = false) {
        TYLOINFER_RUNTIME_CHECK(hidden.is_cuda() && hidden.dim() == 3 && hidden.size(0) > 0 &&
                              hidden.size(1) > 0,
                          "Qwen4 QSA requires nonempty [B,T,H] input");
        activate_memory(request);
        auto graph_keys = (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{});
        auto graph_values = (request.graph_values.defined()?request.graph_values.dense():DenseTensor{});
        TYLOINFER_RUNTIME_CHECK(!history_only || use_cache,"QSA history priming requires a cache");
        if(use_cache && hidden.size(1)>8 && !selection_trace && !history_only) {
            // Index scores grow with both query rows and the accumulated
            // context. Bound that workspace independently of the FFN tile.
            // Keep per-head products, reduction and top-k scratch below the
            // same budget even at the final chunk of a long prompt.
            const auto offset=position(request),length=offset+hidden.size(1);
            const auto pools=std::max<int64_t>(1,length/c_.pool);
            const auto row_bytes=std::size_t(hidden.size(0))*pools*(c_.index_heads+4)*sizeof(float);
            const auto rows=std::max<int64_t>(32,int64_t((128ULL<<20)/row_bytes)/32*32);
            if(hidden.size(1)>rows) {
                std::vector<DenseTensor> outputs;
                for(int64_t begin=0;begin<hidden.size(1);begin+=rows) {
                    const auto count=std::min(rows,hidden.size(1)-begin);
                    std::vector<DenseTensor> projected;
                    for(const auto& value:projections)projected.push_back(value.narrow(1,begin,count));
                    outputs.push_back(forward(request,hidden.narrow(1,begin,count),
                        current_positions.narrow(-1,begin,count),full_positions.narrow(-1,0,offset+begin+count),
                        true,nullptr,std::move(projected),project_output));
                }
                return ::tyloi::cuda::cat(outputs,1);
            }
        }
        if (request.offload && use_cache) {
            if(managed_memory(request))request.memory->ensure(position(request)+hidden.size(1));
            return forward_offloaded(request, hidden, current_positions, full_positions, selection_trace,
                                     std::move(projections),project_output,history_only);
        }
        const auto b = hidden.size(0), t = hidden.size(1), offset = use_cache ? position(request) : 0;
        const auto prefill_option=::tyloi::cuda::options::qsa_prefill_fused();
        // Scalar decode and verification use the same grouped projections and
        // normalization/RoPE preparation as captured execution. Keeping short
        // eager calls on separate norms introduces extra FP16 rounding steps.
        const bool prepared_inputs=!projections.empty() || (!prefill_option || *prefill_option!=0);
        TYLOINFER_RUNTIME_CHECK(
            t <= c_.maximum - offset && full_positions.size(-1) == offset + t &&
                (!use_cache || (offset == request.values.position() && offset == request.index.position())),
            "Qwen4 QSA cache/position geometry mismatch");
        auto project_inputs = [&] {
                if(prepared_inputs)
                    return project(hidden,current_positions,!history_only && offset+t>c_.budget,{},{},{},std::move(projections));
                auto pair = this->project_query(hidden).reshape({b, t, c_.heads, 2 * c_.width});
                auto gate = pair.narrow(-1, c_.width, c_.width);
                auto query = rms_norm(pair.narrow(-1, 0, c_.width),
#ifdef TYLOINFER_NATIVE_CUDA
                                      w_->query_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps)
#else
                                      w_->query_norm.to(::tyloi::kFloat32) + 1, c_.eps)
#endif
                                 .permute({0, 2, 1, 3});
                auto key =
                    rms_norm(::tyloi::ops::project(hidden,w_->input_projection[1]).dense().reshape({b, t, c_.kv_heads, c_.width}),
#ifdef TYLOINFER_NATIVE_CUDA
                             w_->key_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps)
#else
                             w_->key_norm.to(::tyloi::kFloat32) + 1, c_.eps)
#endif
                        .permute({0, 2, 1, 3});
                auto value = ::tyloi::ops::project(hidden,w_->input_projection[2]).dense().reshape({b, t, c_.kv_heads, c_.width});
                query = rotary_->forward(query, current_positions);
                key = rotary_->forward(key, current_positions).permute({0, 2, 1, 3});
                auto iqk = ::tyloi::ops::project(hidden,w_->input_projection[3]).dense();
                auto iq = rms_norm(iqk.narrow(-1, 0, c_.index_heads * c_.index_width)
                                       .reshape({b, t, c_.index_heads, c_.index_width}),
#ifdef TYLOINFER_NATIVE_CUDA
                                   w_->index_query_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps)
#else
                                   w_->index_query_norm.to(::tyloi::kFloat32) + 1, c_.eps)
#endif
                              .permute({0, 2, 1, 3});
                iq = rotary_->forward(iq, current_positions).permute({0, 2, 1, 3});
                auto raw = iqk.narrow(-1, c_.index_heads * c_.index_width, c_.index_width);
                return Projection{gate, query, key, value, iq, raw};
            };
        auto append_inputs = [&](Projection &p) {
                if(managed_memory(request)) {
                    TYLOINFER_RUNTIME_CHECK(b==1,"preallocated QSA expects one sequence per cache");
                    request.memory->ensure(offset+t);
                    if(request.memory->hot.exclusive())request.memory->hot.append(encode_key(p.key).permute({0,2,1,3}),
                        encode_value(p.value).permute({0,2,1,3}),::tyloi::cuda::arange(offset,offset+t,hidden.options().dtype(::tyloi::kInt64)));
                    else {
                        graph_keys.narrow(2,offset,t).copy_(encode_key(p.key).permute({0,2,1,3}));
                        graph_values.narrow(2,offset,t).copy_(encode_value(p.value).permute({0,2,1,3}));
                    }
                    if(request.memory->hot.defined()) {
                        if(!request.memory->hot.exclusive())request.memory->hot.publish(graph_keys,graph_values,::tyloi::cuda::arange(offset,offset+t,hidden.options().dtype(::tyloi::kInt64)));
                        p.key=graph_keys.permute({0,2,1,3});p.value=graph_values.permute({0,2,1,3});
                    }else {
                        p.key=graph_keys.narrow(2,0,offset+t).permute({0,2,1,3});
                        p.value=graph_values.narrow(2,0,offset+t).permute({0,2,1,3});
                    }
                    request.keys.advance_fixed(t);request.values.advance_fixed(t);request.graph_prepared=false;
                } else {
                    p.key = request.keys.append(encode_key(p.key).reshape({b, t, -1}))
                                .reshape({b, offset + t, c_.kv_heads, key_width()});
                    p.value = request.values.append(encode_value(p.value).reshape({b, t, -1}))
                                  .reshape({b, offset + t, c_.kv_heads, value_width()});
                }
                p.raw = request.index.append(p.raw);
            };
        if(history_only) {
            auto p=project_history(hidden,current_positions);
            try {append_inputs(p);}catch(...) {truncate(request,offset);throw;}
            return {};
        }
        const bool short_gate=prepared_inputs && fused_projection_ && rotary_->fused() &&
            t<=8 && !quantization_.enabled() &&
            (hidden.scalar_type()==::tyloi::kFloat16 || hidden.scalar_type()==::tyloi::kFloat32);
        const auto attend_projected=[&](const Projection& p,const DenseTensor& selected) {
            auto key=p.key.permute({0,2,1,3}),value=p.value.permute({0,2,1,3});
            if(!short_gate)return attend(request,p.query,key,value,selected,offset);
            const auto positions=::tyloi::cuda::arange(offset,offset+t,hidden.options().dtype(::tyloi::kInt64));
            const auto hot=prepare_hot(request,selected,positions,offset+t);
            // Match the decode graph's fused gate and FP32 attention
            // accumulation for short eager/correction calls as well.
            if(selected.defined())return mfq_qwen4_exp::sparse_gqa_attention_gate(
                p.query,key,value,selected,p.gate,hidden.scalar_type()==::tyloi::kFloat16,hot);
            return mfq_qwen4_exp::causal_gqa_attention_gate(p.query,key,value,positions,
                p.gate,offset+t,hidden.scalar_type()==::tyloi::kFloat16,hot);
        };
        return tyloi::models::qwen4_exp::sparse_attention(
            offset + t, c_.budget, use_cache, project_inputs, append_inputs,
            [&](const Projection &p) {
                return attend_projected(p,{});
            },
            [&](const Projection &p) {
                const auto &raw = p.raw;
                const auto pools = (offset + t) / c_.pool;
                auto pooled = raw.narrow(1, 0, pools * c_.pool)
                                  .reshape({b, pools, c_.pool, c_.index_width})
                                  .to(::tyloi::kFloat32)
                                  .mean(2)
                                  .to(raw.scalar_type());
#ifdef TYLOINFER_NATIVE_CUDA
                pooled = rms_norm(pooled, w_->index_key_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps);
#else
                pooled = rms_norm(pooled, w_->index_key_norm.to(::tyloi::kFloat32) + 1, c_.eps);
#endif
                auto starts = ::tyloi::cuda::arange(pools, raw.options().dtype(::tyloi::kInt64)) * c_.pool;
                auto positions = full_positions.index_select(-1, starts);
                pooled = rotary_->forward(pooled.unsqueeze(1), positions).squeeze(1);
                return pooled;
            },
            [](const Projection &p, const DenseTensor &pooled) {
                return mfq_qwen4_exp::block_scores(p.iq, pooled);
            },
            [&](const DenseTensor &scores, const Projection &p, const DenseTensor &pooled) {
                auto ids =
                    select_pooled_blocks(scores, offset, offset + t, c_.pool, c_.budget, true);
                if (selection_trace)
                    *selection_trace = {scores, ids, p.iq, pooled};
                return ids;
            },
            [&](const Projection &p, DenseTensor ids) {
                return attend_projected(p,ids);
            },
            [&](DenseTensor attended, const Projection &p) {
                auto gated = short_gate?std::move(attended):
                    attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
                auto result = gated.reshape({b, t, c_.heads * c_.width}).to(hidden.scalar_type());
                return project_output ? this->project_output(result) : result;
            },
            [&] { truncate(request,offset); });
    }

  private:
    HotKvView prepare_hot(State& request,const DenseTensor& selected,const DenseTensor& positions={},int64_t columns=0,bool epoch_ready=false) const {
        if(!managed_memory(request) || !request.memory->hot.defined())return {};
        auto graph_keys = (request.graph_keys.defined()?request.graph_keys.dense():DenseTensor{});
        auto graph_values = (request.graph_values.defined()?request.graph_values.dense():DenseTensor{});
        request.memory->hot.prepare(graph_keys,graph_values,selected,positions,columns,true,epoch_ready);
        return request.memory->hot.view();
    }
    DenseTensor attend(State& request,const DenseTensor& query, DenseTensor keys, DenseTensor values, const DenseTensor& selected, int64_t offset) const {
        if(managed_memory(request) && request.memory->hot.defined()) {
            const auto positions=::tyloi::cuda::arange(offset,offset+query.size(2),query.options().dtype(::tyloi::kInt64));
            const auto hot=offset+query.size(2)<=request.memory->hot.keys.size(2)?request.memory->hot.view():prepare_hot(request,selected,positions,offset+query.size(2));
            if(quantization_.enabled())return cuda_kv_attention(query,keys,values,quantization_,selected,positions,offset,{},0,{},{},0,hot);
            if(selected.defined())return mfq_qwen4_exp::sparse_gqa_attention(query,keys,values,selected,hot);
            return mfq_qwen4_exp::dense_gqa_attention(query,keys,values,offset,hot);
        }
        if (keys.scalar_type() == ::tyloi::kInt32) {
            // Large query tiles amortize one decode per KV row and use the
            // existing matrix attention kernels. Persistent caches stay packed;
            // floating workspace is bounded and lives only for this layer.
            const auto floating_bytes = std::size_t(keys.size(0)) * keys.size(1) * keys.size(2) * c_.width * 4;
            if (query.size(2) <= 8 || floating_bytes > (128ULL << 20))
                return cuda_kv_attention(query, keys, values, quantization_, selected, {}, offset);
            keys = cuda_kv_decode(keys, int(c_.width), quantization_.key_bits(), false);
            values = cuda_kv_decode(values, int(c_.width), quantization_.value_bits(), true);
        }
        if (selected.defined())
            return mfq_qwen4_exp::sparse_gqa_attention(query, keys, values, selected);
        return mfq_qwen4_exp::dense_gqa_attention(query, keys, values, offset);
    }
    int64_t key_width() const { return quantization_.enabled()?cuda_kv_packed_width(int(c_.width),quantization_.key_bits()):c_.width; }
    int64_t value_width() const { return quantization_.enabled()?cuda_kv_packed_width(int(c_.width),quantization_.value_bits()):c_.width; }
    ::tyloi::ScalarType cache_dtype() const { return quantization_.enabled()?::tyloi::kInt32: ::tyloi::kFloat16; }
    DenseTensor encode_key(const DenseTensor& x) const { return quantization_.enabled()?cuda_kv_encode(x,quantization_.key_bits(),false):x.to(::tyloi::kFloat16); }
    DenseTensor encode_value(const DenseTensor& x) const { return quantization_.enabled()?cuda_kv_encode(x,quantization_.value_bits(),true):x.to(::tyloi::kFloat16); }
    struct Projection {DenseTensor gate,query,key,value,iq,raw;bool cache_written=false;};
    DenseTensor streamed_graph_attention(State& request,const Projection& p,const DenseTensor& selected,::tyloi::ScalarType output_type) {
        auto* window=DecodeWindow::recording();
        TYLOINFER_RUNTIME_CHECK(window && selected.size(1)<=8,"streamed QSA requires a decode window of at most eight rows");
        struct Stage {
            DenseTensor raw,host_ids,ids;
            std::shared_ptr<KvOffloadStore::GatherWorkspace> gather;
        };
        const auto tokens=selected.size(1),columns=selected.size(2);
        const auto key_row=c_.kv_heads*key_width(),value_row=c_.kv_heads*value_width();
        const auto stage=window->workspace<Stage>(&request,[&] {
            ResidentAllocationScope resident;
            auto result=std::make_shared<Stage>();
            result->raw=::tyloi::cuda::empty({tokens,key_row+value_row},p.key.options().dtype(cache_dtype()));
            result->host_ids=::tyloi::cuda::empty(selected.sizes(),selected.options().device(::tyloi::kCPU).pinned_memory(true));
            result->ids=::tyloi::cuda::zeros(selected.sizes(),selected.options());
            // Every layer and MTP width shares the same bounded I/O workspace.
            // A later segment may reuse it only after the prior consumer ends.
            result->gather=request.offloaded_kv->gather_workspace(8*columns,c_.kv_heads,key_width());
            return result;
        });
        stage->raw.copy_(::tyloi::cuda::cat({encode_key(p.key).reshape({tokens,key_row}),encode_value(p.value).reshape({tokens,value_row})},1));
        TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->host_ids.data_ptr(),selected.data_ptr(),selected.nbytes(),
            cudaMemcpyDeviceToHost,mfq_current_cuda_stream()));
        window->host_barrier([this,&request,stage] {
            const auto start=position(request);
            try {
                request.offloaded_kv->append(stage->raw);
                auto* ids=stage->host_ids.data_ptr<int32_t>();
                std::unordered_map<int32_t,int32_t> compact;
                std::vector<int64_t> requested;
                requested.reserve(stage->host_ids.numel());compact.reserve(stage->host_ids.numel());
                for(int64_t i=0;i<stage->host_ids.numel();++i)if(ids[i]>=0) {
                    auto [item,added]=compact.emplace(ids[i],int32_t(requested.size()));
                    if(added)requested.push_back(ids[i]);ids[i]=item->second;
                }
                request.offloaded_kv->gather_into(requested,*stage->gather);
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(stage->ids.data_ptr(),ids,stage->ids.nbytes(),
                    cudaMemcpyHostToDevice,mfq_current_cuda_stream()));
            } catch(...) {request.offloaded_kv->truncate(start);throw;}
        });
        const auto rows=stage->gather->rows.size(0);
        const auto& gathered=stage->gather->rows.dense();
        auto flat=gathered.reshape({-1});
        auto keys=flat.narrow(0,0,rows*key_row).reshape({1,c_.kv_heads,rows,key_width()});
        auto values=flat.narrow(0,rows*key_row,rows*value_row).reshape({1,c_.kv_heads,rows,value_width()});
        if(!quantization_.enabled() && fused_projection_ && rotary_->fused() && (output_type==::tyloi::kFloat16 || output_type==::tyloi::kFloat32))
            return mfq_qwen4_exp::sparse_gqa_attention_gate(p.query,keys,values,stage->ids,p.gate,output_type==::tyloi::kFloat16);
        auto attended=attend(request,p.query,keys,values,stage->ids,rows);
        return (attended.to(::tyloi::kFloat32)*::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32))).to(output_type);
    }
    DenseTensor forward_offloaded(State& request, const DenseTensor& hidden,
        const DenseTensor& positions, const DenseTensor& full_positions, std::vector<DenseTensor>* trace,
        std::vector<DenseTensor> projections={},bool project_output=true,bool history_only=false) {
        const auto offset = position(request), tokens = hidden.size(1), length = offset + tokens;
        auto graph_pooled = (request.graph_pooled.defined()?request.graph_pooled.dense():DenseTensor{});
        const auto pool_position=[&]{return managed_memory(request)?request.index.position()/c_.pool:request.offloaded_pooled->position();};
        const auto pool_range=[&](int64_t start,int64_t count,::tyloi::Tensor& owner) -> DenseTensor {
            if(managed_memory(request)) return graph_pooled.narrow(1,start,count).reshape({count,c_.index_width});
            owner=request.offloaded_pooled->range(start,count);
            return owner.dense();
        };
        TYLOINFER_RUNTIME_CHECK(hidden.size(0) == 1 && length <= c_.maximum && full_positions.size(-1) == length,
            "streamed QSA requires one sequence within its configured context");
        const auto key_row = c_.kv_heads * key_width(), value_row = c_.kv_heads * value_width();
        const auto max_keys = std::min(length, c_.budget + c_.pool - 1);
        const auto row_bytes = std::size_t(key_row+value_row) * (quantization_.enabled()?4:2);
        if (std::size_t(max_keys) * row_bytes > request.offload->config().buffer_bytes / 4)
            throw std::invalid_argument("QSA I/O buffer cannot hold one selected context; increase the resident KV budget");
        auto p = history_only?project_history(hidden,positions):
            project( hidden, positions, length > c_.budget,{},{},{},std::move(projections));
        try {
            request.offloaded_kv->append(::tyloi::cuda::cat({encode_key(p.key).reshape({tokens,key_row}), encode_value(p.value).reshape({tokens,value_row})}, 1));
            const auto first_pool = pool_position();
            if(managed_memory(request))request.index.append(p.raw);
            else request.offloaded_index->append(p.raw.reshape({tokens,c_.index_width}));
            const auto new_pools = length / c_.pool - first_pool;
            if (new_pools) {
                ::tyloi::Tensor raw_owner;
                DenseTensor raw;
                if(managed_memory(request))
                    raw=request.index.storage().dense().narrow(1,first_pool*c_.pool,new_pools*c_.pool).reshape({new_pools*c_.pool,c_.index_width});
                else {
                    raw_owner=request.offloaded_index->range(first_pool * c_.pool, new_pools * c_.pool);
                    raw=raw_owner.dense();
                }
                auto pooled = raw.reshape({1, new_pools, c_.pool, c_.index_width}).to(::tyloi::kFloat32).mean(2).to(::tyloi::kFloat16);
#ifdef TYLOINFER_NATIVE_CUDA
                pooled = rms_norm(pooled, w_->index_key_norm.dense().to(::tyloi::kFloat32) + 1, c_.eps);
#else
                pooled = rms_norm(pooled, w_->index_key_norm.to(::tyloi::kFloat32) + 1, c_.eps);
#endif
                auto starts = (::tyloi::cuda::arange(new_pools, positions.options().dtype(::tyloi::kInt64)) + first_pool) * c_.pool;
                pooled = rotary_->forward(pooled.unsqueeze(1), full_positions.index_select(-1, starts)).squeeze(1);
                if(managed_memory(request))graph_pooled.narrow(1,first_pool,new_pools).copy_(pooled);
                else request.offloaded_pooled->append(pooled.reshape({new_pools, c_.index_width}));
            }
            if(history_only)return {};
            // When one layer fits in staging, reuse that layer across the
            // entire prefill tile instead of gathering the same rows per token.
            // Other layers remain under the GPU/RAM/SSD eviction policy.
            if (tokens > 8 && std::size_t(length) * row_bytes <= request.offload->config().buffer_bytes / 2 &&
                    std::size_t(length / c_.pool) * c_.index_width * 2 <= request.offload->config().buffer_bytes / 4) {
                auto packed_owner = request.offloaded_kv->range(0, length);
                const auto& packed = packed_owner.dense();
                auto key = packed.narrow(1,0,key_row).reshape({1,length,c_.kv_heads,key_width()}).permute({0,2,1,3});
                auto value = packed.narrow(1,key_row,value_row).reshape({1,length,c_.kv_heads,value_width()}).permute({0,2,1,3});
                DenseTensor ids;
                if (length > c_.budget) {
                    ::tyloi::Tensor pooled_owner;
                    auto pooled = pool_range(0,length / c_.pool,pooled_owner).reshape({1,length / c_.pool,c_.index_width});
                    auto scores = mfq_qwen4_exp::block_scores(p.iq, pooled);
                    ids = select_pooled_blocks(scores, offset, length, c_.pool, c_.budget, true);
                    if (trace) *trace = {scores,ids,p.iq,pooled};
                }
                auto attended = attend(request,p.query, key, value, ids, offset);
                auto gated = attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
                auto result = gated.reshape({1,tokens,c_.heads*c_.width}).to(hidden.scalar_type());
                return project_output ? this->project_output(result) : result;
            }
            const auto queries = std::max<int64_t>(1, std::min<int64_t>(32,
                request.offload->config().buffer_bytes / 4 / (std::size_t(max_keys) * row_bytes)));
            std::vector<DenseTensor> outputs;
            for (int64_t begin = 0; begin < tokens; begin += queries) {
                const auto count = std::min(queries, tokens - begin);
                auto cache_positions = ::tyloi::cuda::arange(offset + begin, offset + begin + count, positions.options().dtype(::tyloi::kInt64));
                DenseTensor selected;
                if (length <= c_.budget) {
                    auto ids = ::tyloi::cuda::arange(max_keys, cache_positions.options()).reshape({1, 1, max_keys}).expand({1, count, max_keys});
                    selected = ::tyloi::cuda::where(ids <= cache_positions.reshape({1, count, 1}), ids, ::tyloi::cuda::full_like(ids, -1)).to(::tyloi::kInt32).contiguous();
                } else {
                    const auto pools = pool_position(), keep = std::min(c_.budget / c_.pool, pools);
                    DenseTensor best_scores, best_ids;
                    std::vector<DenseTensor> traced_scores, traced_pools;
                    const auto pool_chunk = std::max<int64_t>(16, std::min<int64_t>(1024,
                        request.offload->config().buffer_bytes / 8 / (c_.index_width * 2 + count * 4)));
                    for (int64_t start = 0; start < pools; start += pool_chunk) {
                        const auto n = std::min(pool_chunk, pools - start);
                        ::tyloi::Tensor pooled_owner;
                        auto pooled = pool_range(start, n, pooled_owner).reshape({1, n, c_.index_width});
                        auto scores = mfq_qwen4_exp::block_scores(p.iq.narrow(1, begin, count), pooled);
                        auto ids = ::tyloi::cuda::arange(start, start + n, cache_positions.options());
                        auto visible = (ids * c_.pool + c_.pool - 1).reshape({1, 1, n}) <= cache_positions.reshape({1, count, 1});
                        auto ranked = ::tyloi::cuda::where(visible, scores.to(::tyloi::kFloat32), ::tyloi::cuda::full_like(scores, -1e30).to(::tyloi::kFloat32));
                        auto top = ::tyloi::cuda::topk(ranked, std::min(keep, n), -1, true, false);
                        auto values = std::get<0>(top), indices = std::get<1>(top) + start;
                        if (best_scores.defined()) {
                            values = ::tyloi::cuda::cat({best_scores, values}, -1);
                            indices = ::tyloi::cuda::cat({best_ids, indices}, -1);
                            top = ::tyloi::cuda::topk(values, std::min(keep, values.size(-1)), -1, true, false);
                            best_scores = std::get<0>(top); best_ids = indices.gather(-1, std::get<1>(top));
                        } else { best_scores = values; best_ids = indices; }
                        if (trace) { traced_scores.push_back(scores); traced_pools.push_back(pooled); }
                    }
                    selected = mfq_qwen4_exp::qsa_selected_tokens(best_ids, cache_positions, c_.pool, c_.budget);
                    if (trace && begin == 0) *trace = {::tyloi::cuda::cat(traced_scores, -1), selected, p.iq.narrow(1, begin, count), ::tyloi::cuda::cat(traced_pools, 1)};
                }
                auto cpu_ids = selected.to(::tyloi::kCPU).contiguous();
                auto* mapping = static_cast<int32_t*>(cpu_ids.data_ptr());
                std::unordered_map<int32_t, int32_t> compact;
                std::vector<int64_t> requested;
                for (int64_t index = 0; index < cpu_ids.numel(); ++index) {
                    if (mapping[index] < 0) continue;
                    auto [found, added] = compact.emplace(mapping[index], static_cast<int32_t>(requested.size()));
                    if (added) requested.push_back(mapping[index]);
                    mapping[index] = found->second;
                }
                auto packed_owner = request.offloaded_kv->gather(requested);
                const auto& packed = packed_owner.dense();
                const auto n = static_cast<int64_t>(requested.size());
                auto key = packed.narrow(1,0,key_row).reshape({1,n,c_.kv_heads,key_width()}).permute({0,2,1,3}).contiguous();
                auto value = packed.narrow(1,key_row,value_row).reshape({1,n,c_.kv_heads,value_width()}).permute({0,2,1,3}).contiguous();
                // IDs now address a compact union; selection already contains the causal mask.
                outputs.push_back(attend(request,p.query.narrow(2,begin,count),key,value,cpu_ids.to(selected.device()),n));
            }
            auto attended = ::tyloi::cuda::cat(outputs, 1);
            auto gated = attended.to(::tyloi::kFloat32) * ::tyloi::cuda::sigmoid(p.gate.to(::tyloi::kFloat32));
            auto result=gated.reshape({1, tokens, c_.heads * c_.width}).to(hidden.scalar_type());
            return project_output?this->project_output(result):result;
        } catch (...) { truncate(request,offset); throw; }
    }
    DenseTensor project_norm(const DenseTensor& value,const DenseTensor& weight) {
        if(value.scalar_type()==::tyloi::kFloat16 || value.scalar_type()==::tyloi::kFloat32)
            return grouped_rms_norm_cuda(value.contiguous(),weight.to(::tyloi::kFloat32).contiguous(),value.size(-1),c_.eps,1.0);
        return rms_norm(value,weight.to(::tyloi::kFloat32)+1,c_.eps);
    }
    Projection project_history(const DenseTensor& hidden,const DenseTensor& positions) {
        const auto b=hidden.size(0),t=hidden.size(1);
        auto source=::tyloi::ops::project(hidden,w_->input_projection[1]).dense().reshape({b,t,c_.kv_heads,c_.width});
        DenseTensor key;
        if(fused_projection_ && rotary_->fused() &&
                (source.scalar_type()==::tyloi::kFloat16 || source.scalar_type()==::tyloi::kFloat32))
#ifdef TYLOINFER_NATIVE_CUDA
            key=rotary_->forward_normalized_grouped({source},{w_->key_norm.dense()},positions,c_.eps,1)[0].permute({0,2,1,3});
        else key=rotary_->forward(project_norm(source,w_->key_norm.dense()).permute({0,2,1,3}),positions).permute({0,2,1,3});
#else
            key=rotary_->forward_normalized_grouped({source},{w_->key_norm},positions,c_.eps,1)[0].permute({0,2,1,3});
        else key=rotary_->forward(project_norm(source,w_->key_norm).permute({0,2,1,3}),positions).permute({0,2,1,3});
#endif
        auto value=::tyloi::ops::project(hidden,w_->input_projection[2]).dense().reshape({b,t,c_.kv_heads,c_.width});
        auto raw=::tyloi::ops::project(hidden,w_->input_projection[3]).dense().narrow(-1,c_.index_heads*c_.index_width,c_.index_width);
        return {{},{},std::move(key),std::move(value),{},std::move(raw)};
    }
    Projection project(const DenseTensor& hidden,const DenseTensor& positions,
                       bool index_query_needed=true,const DenseTensor& cache_positions={},
                       const DenseTensor& key_cache={},const DenseTensor& value_cache={},
                       std::vector<DenseTensor> projections={}) {
        const auto b=hidden.size(0),t=hidden.size(1);
        if(projections.empty()) {
            if(grouped_projection_ && ::tyloi::ops::grouped(w_->input_projection))
                projections=project_inputs(hidden);
            else projections={project_query(hidden),
                ::tyloi::ops::project(hidden,w_->input_projection[1]).dense(),
                ::tyloi::ops::project(hidden,w_->input_projection[2]).dense(),
                ::tyloi::ops::project(hidden,w_->input_projection[3]).dense()};
        }
        TYLOINFER_RUNTIME_CHECK(projections.empty() || projections.size()==4,"QSA input projection count mismatch");
        auto pair=projections[0].reshape({b,t,c_.heads,2*c_.width});
        auto gate=pair.narrow(-1,c_.width,c_.width);
        auto key_source=projections[1].reshape({b,t,c_.kv_heads,c_.width});
        auto value=projections[2].reshape({b,t,c_.kv_heads,c_.width});
        const auto supported=[](const DenseTensor& x) { return x.scalar_type()==::tyloi::kFloat16 || x.scalar_type()==::tyloi::kFloat32; };
        const bool fused=fused_projection_ && rotary_->fused() && supported(pair) && supported(key_source) && supported(value);
        const bool cache_written=fused && key_cache.defined();
        auto iqk=projections[3];
        DenseTensor query,key,iq;
        if(fused) {
            std::vector<DenseTensor> inputs{pair.narrow(-1,0,c_.width),key_source};
#ifdef TYLOINFER_NATIVE_CUDA
            std::vector<DenseTensor> weights{w_->query_norm.dense(),w_->key_norm.dense()};
#else
            std::vector<DenseTensor> weights{w_->query_norm,w_->key_norm};
#endif
            if(index_query_needed) {
                auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
#ifdef TYLOINFER_NATIVE_CUDA
                if(supported(source)){inputs.push_back(source);weights.push_back(w_->index_query_norm.dense());}
#else
                if(supported(source)){inputs.push_back(source);weights.push_back(w_->index_query_norm);}
#endif
            }
            auto normalized=rotary_->forward_normalized_grouped(inputs,weights,positions,c_.eps,1,
                cache_written?key_cache:DenseTensor{},cache_written?cache_positions:DenseTensor{},
                cache_written?value:DenseTensor{},cache_written?value_cache:DenseTensor{});
            query=normalized[0];key=normalized[1];
            if(key.defined())key=key.permute({0,2,1,3});
            if(normalized.size()==3)iq=normalized[2].permute({0,2,1,3});
        } else {
#ifdef TYLOINFER_NATIVE_CUDA
            query=project_norm(pair.narrow(-1,0,c_.width),w_->query_norm.dense()).permute({0,2,1,3});
            key=project_norm(key_source,w_->key_norm.dense()).permute({0,2,1,3});
#else
            query=project_norm(pair.narrow(-1,0,c_.width),w_->query_norm).permute({0,2,1,3});
            key=project_norm(key_source,w_->key_norm).permute({0,2,1,3});
#endif
            query=rotary_->forward(query,positions);key=rotary_->forward(key,positions).permute({0,2,1,3});
        }
        if(index_query_needed && !iq.defined()) {
            auto source=iqk.narrow(-1,0,c_.index_heads*c_.index_width).reshape({b,t,c_.index_heads,c_.index_width});
#ifdef TYLOINFER_NATIVE_CUDA
            if(fused && supported(source))iq=rotary_->forward_normalized(source,w_->index_query_norm.dense(),positions,c_.eps).permute({0,2,1,3});
#else
            if(fused && supported(source))iq=rotary_->forward_normalized(source,w_->index_query_norm,positions,c_.eps).permute({0,2,1,3});
#endif
            else {
#ifdef TYLOINFER_NATIVE_CUDA
                iq=project_norm(source,w_->index_query_norm.dense()).permute({0,2,1,3});
#else
                iq=project_norm(source,w_->index_query_norm).permute({0,2,1,3});
#endif
                iq=rotary_->forward(iq,positions).permute({0,2,1,3});
            }
        }
        return {gate,query,key,value,iq,iqk.narrow(-1,c_.index_heads*c_.index_width,c_.index_width),cache_written};
    }
    std::shared_ptr<const QsaWeights> w_;
    QsaConfig c_;
    tyloi::state::KvQuantization quantization_=cuda_kv_quantization();
    std::shared_ptr<RotaryEmbedding> rotary_;
    bool fused_projection_;
    bool grouped_projection_;
    std::shared_ptr<KvOffloadStore> offload_;
    std::shared_ptr<KvMemoryManager> memory_manager_;
};
} // namespace tyloi::cuda::qwen4_exp
