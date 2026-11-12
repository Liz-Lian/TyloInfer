#pragma once

#include "engine/ops/gdn.h"
#include "engine/ops.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_window.h"
#include "engine/backends/cuda/native/resident_allocations.h"
#include "engine/backends/cuda/native/matmul_rows.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/native/env.h"
#include <algorithm>
#include <array>
#include <cstdio>

namespace tyloi::ops {
struct Gdn::State::Impl {
    Tensor core_workspace{DenseTensor{}};
    // Keep failed graph bridge allocations alive until the runtime drains and resets the request.
    std::vector<DenseTensor> failed_conv, failed_state;
    DenseTensor failed_output;
    bool resident = false;
};
namespace detail {
inline thread_local int64_t gdn_projection_rows = 0;
class GdnCore {
  public:
    using State = ::tyloi::ops::Gdn::State;
    GdnCore(std::shared_ptr<const GdnWeights> weights, int64_t key_heads, int64_t value_heads, int64_t width, int64_t kernel,
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
        request.impl_->failed_conv.clear();
        request.impl_->failed_state.clear();
        request.impl_->failed_output = {};
        if(retain_storage) {
            if(request.conv.dense().defined())request.conv.dense().zero_();
            if(request.recurrent.dense().defined())request.recurrent.dense().zero_();
            if(request.impl_->core_workspace.dense().defined())request.impl_->core_workspace.dense().zero_();
        } else {
            request.conv.dense() = DenseTensor();
            request.recurrent.dense() = DenseTensor();
            request.impl_->core_workspace.dense() = DenseTensor();
        }
        commit(request);
    }
    void commit(State& request) {
        request.saved_conv = Tensor{};
        request.saved_state = Tensor{};
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
        if(request.conv.dense().defined())request.conv.dense().copy_(request.saved_conv.dense());else request.conv.dense()=request.saved_conv.dense();
        if(request.recurrent.dense().defined())request.recurrent.dense().copy_(request.saved_state.dense());else request.recurrent.dense()=request.saved_state.dense();
        commit(request);
    }
    const DenseTensor &conv_state(const State& request) const { return request.conv.dense(); }
    void initialize_request_cache(State& request, bool allocate_counters=true) {
        // Recurrence is allocated on first use, outside temporary pool pages.
        // Keep the reusable decode counters out of graph scratch as well.
        request.impl_->resident=true;
        ::tyloi::cuda::ResidentAllocationScope resident;
        auto options=w_->norm.dense().options().dtype(::tyloi::kFloat32);
        if(allocate_counters && !request.impl_->core_workspace.dense().defined() && fused_core_ && fused_decode_ &&
            fused_preparation_ && fused_output_ && transposed_state_)
            request.impl_->core_workspace.dense()=::tyloi::cuda::zeros({1,nv_,d_+1},options);
    }
    std::size_t growth_bytes(const State& request) const {
        return (request.conv.dense().defined() ? 0 : std::size_t(kernel_-1)*(2*nk_+nv_)*d_*sizeof(float)) +
            (request.recurrent.dense().defined() ? 0 : std::size_t(nv_)*d_*d_*sizeof(float));
    }
    std::size_t request_cache_growth_bytes(const State& request) const {
        return growth_bytes(request) + (fused_core_ && fused_decode_ && fused_preparation_ &&
            fused_output_ && transposed_state_ && !request.impl_->core_workspace.dense().defined()
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
    static DenseTensor prepare_independent_batch(const std::vector<std::pair<const GdnCore*, State*>>& rows) {
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
                request.conv.dense().defined() && request.recurrent.dense().defined() && request.impl_->core_workspace.dense().defined() &&
                request.conv.dense().is_cuda() && request.recurrent.dense().is_cuda() && request.impl_->core_workspace.dense().is_cuda() &&
                request.conv.dense().is_contiguous() && request.recurrent.dense().is_contiguous() && request.impl_->core_workspace.dense().is_contiguous() &&
                request.conv.dense().scalar_type()==::tyloi::kFloat32 && request.recurrent.dense().scalar_type()==::tyloi::kFloat32 &&
                request.impl_->core_workspace.dense().scalar_type()==::tyloi::kFloat32 &&
                request.conv.dense().sizes().vec()==std::vector<int64_t>({1,row.kernel_-1,(2*row.nk_+row.nv_)*row.d_}) &&
                request.recurrent.dense().sizes().vec()==std::vector<int64_t>({1,row.nv_,row.d_,row.d_}) &&
                request.impl_->core_workspace.dense().sizes().vec()==std::vector<int64_t>({1,row.nv_,row.d_+1});
        };
        if(::tyloi::cuda::options::gdn_independent_batch_trace())
            std::fprintf(stderr,"gdn_batch_state eligible=%d flags=%d%d%d%d%d conv=%d state=%d counters=%d geometry=%lld/%lld/%lld/%lld\n",
                int(eligible(rows.front())),int(first.fused_core_),int(first.fused_decode_),int(first.fused_preparation_),
                int(first.fused_output_),int(first.transposed_state_),int(first_state.conv.dense().defined()),int(first_state.recurrent.dense().defined()),
                int(first_state.impl_->core_workspace.dense().defined()),(long long)first.nk_,(long long)first.nv_,(long long)first.d_,(long long)first.kernel_);
        if(!eligible(rows.front()))return {};
        for(const auto& row:rows)if(!eligible(row) || row.first->nk_!=first.nk_ || row.first->nv_!=first.nv_ ||
            row.first->d_!=first.d_ || row.first->kernel_!=first.kernel_ || row.first->eps_!=first.eps_ ||
            row.first->silu_gate_!=first.silu_gate_ || row.second->recurrent.dense().device()!=first_state.recurrent.dense().device() ||
            row.second->conv.dense().device()!=first_state.recurrent.dense().device() || row.second->impl_->core_workspace.dense().device()!=first_state.recurrent.dense().device() ||
            row.first->w_->conv.dense().data_ptr()!=first.w_->conv.dense().data_ptr() || row.first->w_->norm.dense().data_ptr()!=first.w_->norm.dense().data_ptr() ||
            row.first->w_->dt_bias.dense().data_ptr()!=first.w_->dt_bias.dense().data_ptr() || row.first->w_->a_log.dense().data_ptr()!=first.w_->a_log.dense().data_ptr())return {};
        auto addresses=::tyloi::cuda::empty({int64_t(rows.size()),3},first_state.recurrent.dense().options().device(::tyloi::kCPU).dtype(::tyloi::kInt64));
        auto* values=addresses.data_ptr<int64_t>();
        for(std::size_t i=0;i<rows.size();++i) {
            values[3*i]=reinterpret_cast<int64_t>(rows[i].second->conv.dense().data_ptr());
            values[3*i+1]=reinterpret_cast<int64_t>(rows[i].second->recurrent.dense().data_ptr());
            values[3*i+2]=reinterpret_cast<int64_t>(rows[i].second->impl_->core_workspace.dense().data_ptr());
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
        auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
        if(weight.dim()==2 && weight.size(0)==(2*nk_+nv_)*d_ && weight.size(1)==kernel_)
            weight=weight.reshape({(2*nk_+nv_)*d_,1,kernel_});
        const auto stride_setting=::tyloi::cuda::options::gdn_strided_projections();
        const auto direct=[&](const DenseTensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
        auto result=gdn_decode_core_cuda(direct(projections[0]),direct(projections[1]),
            projections[2].reshape({x.size(0),1,nv_}),projections[3].reshape({x.size(0),1,nv_}),
            request.conv.dense(),request.recurrent.dense(),weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
            w_->norm.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
            x.scalar_type()==::tyloi::kFloat16,true,request.impl_->core_workspace.dense(),true,addresses);
        return result[0];
    }
    DenseTensor forward_graph(State& request, const DenseTensor& x,int64_t confirmed,
                         std::vector<DenseTensor>& checkpoint_conv,std::vector<DenseTensor>& checkpoint_state) {
        if(!confirmed)return forward_chunk(request, x,true);
        const auto batch=::tyloi::cuda::options::gdn_batch_projection();
        const bool grouped=tyloi::cuda::Executor::current().mode().compute.attention_grouped.value_or(grouped_projection_) && ::tyloi::ops::grouped(w_->input_projection) && x.size(1)<=8 && (!batch || *batch);
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
            auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
            const auto channels=(2*nk_+nv_)*d_;
            if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                weight=weight.reshape({channels,1,kernel_});
            auto decoded=gdn_decode_window_cuda(projected[0],projected[1],projected[2].reshape({1,x.size(1),nv_}),
                projected[3].reshape({1,x.size(1),nv_}),request.conv.dense(),request.recurrent.dense(),weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),
                w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),w_->norm.dense().to(::tyloi::kFloat32).contiguous(),int(nk_),int(nv_),int(confirmed),
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
        request.saved_prefix_conv.clear();request.saved_prefix_state.clear();
        for(const auto& value:conv)request.saved_prefix_conv.emplace_back(value);
        for(const auto& value:state)request.saved_prefix_state.emplace_back(value);
        request.saved_prefix_position=position;
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
            tyloi::cuda::Executor::current().mode().compute.attention_grouped.value_or(grouped_projection_) && ::tyloi::ops::grouped(w_->input_projection) && (!batch || *batch);
        std::vector<DenseTensor> projected;
        if(grouped)projected=project_inputs(x);
        const auto run_window = [&](int64_t start, int64_t count) {
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
            };
        const auto checkpoint = [&] {
                request.saved_conv = request.conv.dense().clone();
                request.saved_state = request.recurrent.dense().clone();
            };
        TYLOINFER_RUNTIME_CHECK(x.size(1)>0 && confirmed>=0 && confirmed<=x.size(1) && (!confirmed || cache),
            "invalid recurrent verification window");
        DenseTensor result;
        if(confirmed==0 || confirmed==x.size(1)) result=run_window(0,x.size(1));
        else {
            TYLOINFER_RUNTIME_CHECK(!request.saved_conv.defined(), "resolve previous recurrent checkpoint");
            auto prefix=run_window(0,confirmed);
            checkpoint();
            try {
                auto suffix=run_window(confirmed,x.size(1)-confirmed);
                result=::tyloi::cuda::cat({prefix,suffix},1);
            } catch (...) { rollback(request); throw; }
        }
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
                auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
                if(weight.dim()==2 && weight.size(0)==channels && weight.size(1)==kernel_)
                    weight=weight.reshape({channels,1,kernel_});
                if(!request.impl_->core_workspace.dense().defined())request.impl_->core_workspace.dense()=::tyloi::cuda::zeros({b,nv_,d_+1},options);
                const auto stride_setting=::tyloi::cuda::options::gdn_strided_projections();
                const auto direct=[&](const DenseTensor& value){return !stride_setting || *stride_setting?value:value.contiguous();};
                const auto decode=[&](bool checkpoint) {
                  auto qkv=direct(projections[0]),gate=direct(projections[1]);
                  auto alpha=projections[2].reshape({b,t,nv_}),beta=projections[3].reshape({b,t,nv_});
                  if(checkpoint)return gdn_decode_core_checkpoint_cuda(qkv,gate,alpha,beta,previous,request.recurrent.dense(),
                    weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
                    w_->norm.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
                    x.scalar_type()==::tyloi::kFloat16,true,request.impl_->core_workspace.dense(),true,checkpoint_conv,checkpoint_state);
                  return gdn_decode_core_cuda(qkv,gate,
                    projections[2].reshape({b,t,nv_}),projections[3].reshape({b,t,nv_}),previous,request.recurrent.dense(),
                    weight,w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
                    w_->norm.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6,eps_,silu_gate_,
                    x.scalar_type()==::tyloi::kFloat16,true,request.impl_->core_workspace.dense(),
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
                const auto project = [&] {
                DenseTensor projected;
                if(!projections.empty())projected=projections[0];
                else if(tyloi::cuda::Executor::current().mode().compute.attention_grouped.value_or(grouped_projection_) && ::tyloi::ops::grouped(w_->input_projection)) {
                    projections=project_inputs(x);
                    TYLOINFER_RUNTIME_CHECK(projections.size()==4,"GDN input projection count mismatch");
                    projected=projections[0];
                } else projected=project_qkv(x);
                TYLOINFER_RUNTIME_CHECK(projected.sizes().vec() == std::vector<int64_t>({b, t, channels}),
                                  "Qwen4 GDN projection width mismatch");
                return projected;
            };
        const auto convolve = [&](DenseTensor projected) {
                if(prefill_fused && projected.scalar_type()==::tyloi::kFloat16) {
                    return ::tyloi::ops::measure("prefill.gdn_conv_qkv",[&] {
                        auto weight=w_->conv.dense().to(::tyloi::kFloat32).contiguous();
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
                                w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),nk_,nv_,d_,1e-6);
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
                    auto weight = w_->conv.dense().to(::tyloi::kFloat32).contiguous();
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
                            w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),w_->a_log.dense().to(::tyloi::kFloat32).contiguous(),
                            nk_,nv_,d_,d_,1e-6);
                        prepared_gates={qkv[3],qkv[4]};
                    } else qkv=linear_conv_qkv_decode_cuda(next_conv,qk,v,weight,
                        ::tyloi::cuda::empty({0}, options),nk_,nv_,d_,d_,1e-6);
                    return std::array<DenseTensor, 4>{qkv[0], qkv[1], qkv[2], next_conv};
                }
                auto joined = ::tyloi::cuda::cat({previous, projected.to(::tyloi::kFloat32)}, 1).contiguous();
                auto convolved = ssm_conv_silu_cuda(joined, w_->conv.dense().to(::tyloi::kFloat32).contiguous(),
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
            };
        const auto gates = [&] {
                if(prepared_gates[0].defined())return prepared_gates;
                auto alpha=projections.empty()
                    ?project_weight(w_->input_projection[w_->qkv_count+1],x):projections[2];
                auto raw_beta=projections.empty()
                    ?project_weight(w_->input_projection[w_->qkv_count+2],x):projections[3];
                if ((fused_decode_ && t == 1) || prefill_fused) {
                    auto gates = linear_gate_beta_cuda(
                        alpha.to(::tyloi::kFloat32).reshape({b, t, nv_}),
                        raw_beta.to(::tyloi::kFloat32).reshape({b, t, nv_}),
                        w_->dt_bias.dense().to(::tyloi::kFloat32).contiguous(),
                        w_->a_log.dense().to(::tyloi::kFloat32).contiguous(), true);
                    return std::array<DenseTensor, 2>{gates[0], gates[1]};
                }
                auto gate_input = alpha.to(::tyloi::kFloat32).reshape({b, t, nv_}) +
                                  w_->dt_bias.dense().to(::tyloi::kFloat32).reshape({1, 1, nv_});
                auto softplus =
                    ::tyloi::cuda::clamp_min(gate_input, 0) + ::tyloi::cuda::log1p(::tyloi::cuda::exp(-gate_input.abs()));
                auto decay = (-w_->a_log.dense().to(::tyloi::kFloat32).exp().reshape({1, 1, nv_}) * softplus)
                                 .transpose(1, 2)
                                 .contiguous();
                auto beta = ::tyloi::cuda::sigmoid(raw_beta.to(::tyloi::kFloat32).reshape({b, t, nv_}))
                                .transpose(1, 2)
                                .contiguous();
                return std::array<DenseTensor, 2>{decay, beta};
            };
        const auto recur = [&](const auto &convolution, const auto &gates) {
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
            };
        const auto norm_gate = [&](DenseTensor attended) {
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
                        w_->norm.dense().to(::tyloi::kFloat32).contiguous(),eps_,silu_gate_,x.scalar_type()==::tyloi::kFloat16);
                projected_gate=projected_gate.contiguous();
                auto z = projected_gate.reshape({b, t, nv_, d_})
                             .permute({0, 2, 1, 3})
                             .to(::tyloi::kFloat32);
                auto gate = ::tyloi::cuda::sigmoid(z);
                if (silu_gate_)
                    gate = z * gate;
                auto f = attended.to(::tyloi::kFloat32);
                auto normalized = (f * ::tyloi::cuda::rsqrt((f * f).mean(-1, true) + eps_) *
                    w_->norm.dense().to(::tyloi::kFloat32)).to(attended.scalar_type());
                return normalized * gate;
            };
        const auto output = [&](DenseTensor normalized) {
                if(normalized.dim()==3)return project_output?this->project_output(normalized):normalized;
                normalized=normalized.permute({0,2,1,3}).reshape({b,t,vw}).to(x.scalar_type());
                if(!project_output)return normalized;
                auto result = this->project_output(normalized);
                return result;
            };
        const auto commit_state = [&](const auto &convolution, const DenseTensor &next) {
                if(request.conv.dense().defined())request.conv.dense().copy_(convolution[3]);
                else {
                    ::tyloi::cuda::ResidentAllocationScope resident(request.impl_->resident);
                    request.conv.dense()=request.impl_->resident?convolution[3].clone():convolution[3];
                }
                if(request.recurrent.dense().defined()) {if(request.recurrent.dense().data_ptr()!=next.data_ptr())request.recurrent.dense().copy_(next);}
                else {
                    ::tyloi::cuda::ResidentAllocationScope resident(request.impl_->resident);
                    request.recurrent.dense()=request.impl_->resident?next.clone():next;
                }
            };
        auto projected=project();
        auto convolution=convolve(std::move(projected));
        auto decay=gates();
        auto recurrent=recur(convolution,std::move(decay));
        auto result=output(norm_gate(recurrent[0]));
        if(cache)commit_state(convolution,recurrent[1]);
        return result;
    }

  private:
    DenseTensor project_weight(const ::tyloi::Tensor& weight,const DenseTensor& source) const;
    DenseTensor project_qkv(const DenseTensor& source) const;
    std::shared_ptr<const GdnWeights> w_;
    int64_t nk_, nv_, d_, kernel_;
    double eps_;
    const bool silu_gate_, fused_decode_,transposed_state_,fused_output_,grouped_projection_,fused_preparation_,fused_core_;
};

template<class Project>
inline DenseTensor gdn_project_rows(const DenseTensor& source,Project project) {
    const auto rows=gdn_projection_rows;
    if(rows<=0 || source.dim()!=3 || source.size(0)!=1 || source.size(1)<=rows)return project(source);
    const auto parallel_option=::tyloi::cuda::options::gdn_prefill_projection_streams();
    const int streams=parallel_option?*parallel_option:2;
    const auto tail=source.size(1)%rows;
    if(streams>0 && (tail==0 || tail>8)) {
        const tyloi::cuda::MatmulRowsScope gemm_rows(rows,streams);
        return project(source);
    }
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

inline DenseTensor GdnCore::project_weight(const ::tyloi::Tensor& weight,const DenseTensor& source) const {
    return gdn_project_rows(source,[&](const DenseTensor& input){return ::tyloi::ops::project(input,weight).dense();});
}

inline DenseTensor GdnCore::project_qkv(const DenseTensor& source) const {
    if(w_->qkv_count==1)return project_weight(w_->input_projection[0],source);
    return gdn_project_rows(source,[&](const DenseTensor& input) {
        return ::tyloi::cuda::cat({::tyloi::ops::project(input,w_->input_projection[0]).dense(),
            ::tyloi::ops::project(input,w_->input_projection[1]).dense(),
            ::tyloi::ops::project(input,w_->input_projection[2]).dense()},-1);
    });
}

inline std::vector<DenseTensor> GdnCore::project_inputs(const DenseTensor& source) const {
    if(!(tyloi::cuda::Executor::current().mode().compute.attention_grouped.value_or(grouped_projection_) && ::tyloi::ops::grouped(w_->input_projection)))
        return {project_qkv(source),
            project_weight(w_->input_projection[w_->qkv_count],source),
            project_weight(w_->input_projection[w_->qkv_count+1],source),
            project_weight(w_->input_projection[w_->qkv_count+2],source)};
    auto* window=tyloi::cuda::DecodeWindow::recording();
    if(window)window->mark_current("gdn_input_begin");
    const auto rows=gdn_projection_rows;
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
        const auto parallel_option=::tyloi::cuda::options::gdn_prefill_projection_streams();
        const int streams=parallel_option?*parallel_option:2;
        const auto tail=source.size(1)%rows;
        if(streams>0 && (tail==0 || tail>8)) {
            const tyloi::cuda::MatmulRowsScope gemm_rows(rows,streams);
            output=project(source);
        } else {
            slice();
        }
    }
    if(window)window->mark_current("gdn_input_end");
    return output;
}

inline DenseTensor GdnCore::project_output(const DenseTensor& source) const {
    auto* window=tyloi::cuda::DecodeWindow::recording();
    if(window)window->mark_current("gdn_output_begin");
    auto output=project_weight(w_->output,source);
    if(window)window->mark_current("gdn_output_end");
    return output;
}
} // namespace detail
} // namespace tyloi::ops
