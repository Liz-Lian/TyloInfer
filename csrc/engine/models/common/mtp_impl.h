#pragma once
#include "engine/weights/workspace.h"
#include "engine/budget.h"

#include "env.h"
#include "engine/weights/cache_internal.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/generation/mtp/state.h"
#include "engine/models/info.h"
#include "engine/models/source.h"
#include "engine/models/common/prefetch.h"
#include "engine/models/common/mtp.h"
#include "engine/models/common/prompt.h"
#include "engine/backends/cuda/executor/sampling.h"
#include "engine/requests/policy.h"
#include "engine/requests/output.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/models/common/mtp_ops.h"
#include "engine/models/common/generation_ops.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/models/common/input.h"
#include "engine/models/causal_model.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>
using tyloi::cuda::internal::PrefillCudaTimer;

using namespace tyloi::engine;

namespace mtp_detail {
using tyloi::cuda::generation_workspace;
using tyloi::cuda::mtp_stream_prompt_history;
using tyloi::cuda::mtp_host_prompt_history;
using tyloi::cuda::mtp_bootstrap_before_publication;
using tyloi::cuda::mtp_retains_partial_target_prefix;
using tyloi::cuda::mtp_history_merge_limit;
using tyloi::cuda::mtp_history_flush_threshold;
using tyloi::cuda::mtp_parked_history_flush_threshold;
using tyloi::cuda::mtp_predictor_prefill_chunk_size;
using tyloi::cuda::mtp_cache_only_prompt_history;
using tyloi::cuda::mtp_initialize;
using tyloi::cuda::mtp_target_forward;
using tyloi::cuda::mtp_prefill;
using tyloi::cuda::mtp_predictor_step;
using tyloi::cuda::mtp_predictor_compact;
using tyloi::cuda::mtp_predictor_history_step;
using tyloi::cuda::mtp_begin_predictor_prefill;
using tyloi::cuda::mtp_predictor_prefill_begin;
using tyloi::cuda::mtp_predictor_prefill;

using DenseTensor = ::tyloi::DenseTensor;

template <typename Model, typename State>
constexpr bool has_forward =
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
  requires(Model& model, State& state, tyloi::cuda::Input input,
           const tyloi::models::ForwardOptions& options) {
    static_cast<const tyloi::models::CausalModel&>(model).forward(std::move(input), state, options);
  } ||
#endif
  requires(
    Model& model, State& state, tyloi::cuda::Executor& executor,
    tyloi::cuda::Collectives& collectives, const tyloi::cuda::Executor::Mode& mode,
    tyloi::cuda::Input input) {
  forward(model, state, executor, collectives, mode, std::move(input));
};

template <typename Model, typename State>
constexpr bool has_replay = requires(
    Model& model, State& state, tyloi::cuda::Executor& executor) {
  replay(model, state, executor, std::int64_t{});
};

template <typename Model, typename State>
DenseTensor hidden(Model& model, State& state, tyloi::cuda::Executor* executor,
              tyloi::cuda::Collectives& collectives, const tyloi::cuda::Executor::Mode& mode,
              tyloi::cuda::Input input) {
  static_assert(has_forward<Model, State>);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
  if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(model).forward(
                             std::move(input), state, tyloi::cuda::forward_options(mode)); }) {
    const tyloi::models::CausalModel& causal = model;
    auto scope = tyloi::cuda::Executor::current().enter(mode.ffn);
    auto result = causal.forward(std::move(input), state, tyloi::cuda::forward_options(mode));
    TYLOINFER_RUNTIME_CHECK(result != nullptr, "CUDA model forward returned no hidden tensor");
    return dynamic_cast<const DenseTensor&>(*result);
  } else
#endif
  {
    TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                            "CUDA model forward requires an executor");
    return forward(model, state, *executor, collectives, mode, std::move(input));
  }
}
template <typename Model, typename State>
void replay_target(Model& model, State& state, tyloi::cuda::Executor* executor,
                   std::int64_t accepted) {
  static_assert(has_replay<Model, State>);
  TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                          "CUDA model replay requires an executor");
  replay(model, state, *executor, accepted);
}

template <class Model, class State> struct CudaMtpOps {
  using DenseTensor = ::tyloi::DenseTensor;
  Model &model;
  State &state;
  tyloi::cuda::Executor* executor;
  tyloi::cuda::Collectives& collectives;
  const tyloi::cuda::Executor::Mode mode;
  ::tyloi::weights::Cache& weights;
  const tyloi::models::Info &info;
  const tyloi::ModelSource* source;
  tyloi::models::Mtp &mtp;
  tyloi::engine::mtp::State &mtp_state;
  const tyloi::cuda::PreparedPrompt *prepared;
  bool transformed_prompt;
  int position_axes;
  int64_t prefill_compute_chunk;
  InferenceOutput* output;
  tyloi::engine::Budget* budget=nullptr;
  // Destroy after local/result tensors, including cancellation of priming.
  std::unique_ptr<tyloi::weights::Loan> predictor_loan;
  const ::tyloi::TensorOptions options =
      ::tyloi::TensorOptions()
          .device(::tyloi::kCUDA)
          .dtype(::tyloi::kInt64);
  DenseTensor input_ids, counts;
  DenseTensor known_ids_host, known_ids_device;
  DenseTensor early_ple_ids_host;
  DenseTensor projected_target_hidden, projected_target_logits;
  DenseTensor projected_predictor_hidden, projected_predictor_logits;
  DenseTensor projected_target_compact,projected_predictor_compact;
  int graph_sampling_reuses=0;
  int32_t projected_predictor_greedy=-1;
  std::vector<int32_t> projected_target_greedy;
  std::optional<tyloi::cuda::Sampler> sampler_;
  bool penalties = false;
  bool target_maintenance = false;
  std::uint64_t target_graph_captures = 0;
  std::uint64_t target_transfer_bytes=0;
  std::array<double,6> target_profile{};
  std::shared_ptr<MoeExpertCache> target_route_cache;
  static bool profile_target() {
    static const bool enabled=[] {
      const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_MTP");
      return option && option[0]=='1';
    }();
    return enabled;
  }
  std::array<double,6> target_costs() const {
    std::array<double,6> result{};
    for(const auto& [name,value]:weights.metrics()) {
      if(name=="expert_route_wait_ms")result[0]=value;
      else if(name=="expert_host_fetch_ms")result[1]=value;
      else if(name=="expert_host_copy_ms")result[2]=value;
      else if(name=="ssd_expert_read_wait_ms")result[3]=value;
    }
    if(source) {
      const auto io=source->file_read_stats();
      result[4]=double(io.read_nanoseconds)/1e6;
      result[5]=double(io.physical_bytes)/1048576.0;
    }
    return result;
  }
  CudaMtpOps(CudaMtpOps&&)=default;
  ~CudaMtpOps() {
    if(target_route_cache)discard_moe_speculative_routes(target_route_cache);
  }
  void begin_target_routes(int rows) {
    begin_moe_speculative_routes(::tyloi::weights::detail::cache(weights).moe,rows);
    target_route_cache=::tyloi::weights::detail::cache(weights).moe;
  }
  void resolve_target_routes(int rows,bool cancelled) {
    if(!target_route_cache)return;
    if(cancelled)discard_moe_speculative_routes(target_route_cache);
    else commit_moe_speculative_routes(target_route_cache,rows);
    target_route_cache.reset();
  }
  bool take_target_maintenance() {
    const auto captures=moe_expert_graph_captures(::tyloi::weights::detail::cache(weights).moe);
    const bool initialized=captures!=target_graph_captures;
    target_graph_captures=captures;
    return std::exchange(target_maintenance,false) || initialized;
  }
  bool prefetch_target_ids(int32_t token,const std::vector<int32_t>& drafts) {
    if constexpr(requires {model.adapter_prefetch_token_rows(model,early_ple_ids_host);}) {
      const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_EARLY_PLE");
      if(option && option[0]=='0')return false;
      const int tokens=mtp.split_target_verification()?1:1+int(drafts.size());
      if(!early_ple_ids_host.defined() || early_ple_ids_host.size(1)<tokens)
        early_ple_ids_host=::tyloi::cuda::empty({1,tokens},options.device(::tyloi::kCPU));
      auto* ids=early_ple_ids_host.template data_ptr<int64_t>();ids[0]=token;
      for(int i=1;i<tokens;++i)ids[i]=drafts[std::size_t(i-1)];
      if(tyloi::cuda::internal::prefetch_token_rows(model,early_ple_ids_host.narrow(1,0,tokens))) {
        static bool reported=false;
        if(!reported) {
          reported=true;std::cerr<<"runtime_mtp_early_token_rows enabled=1 prefix_tokens=1\n";
        }
        if(tokens>1 && tokens<64) {
          static uint64_t reported_prefixes=0;const auto bit=uint64_t(1)<<tokens;
          if(!(reported_prefixes&bit)) {
            reported_prefixes|=bit;
            std::cerr<<"runtime_mtp_token_row_draft_prefix enabled=1 prefix_tokens="<<tokens<<'\n';
          }
        }
        return true;
      }
    }
    return false;
  }
  void prefetch_target_first(int32_t token) {prefetch_target_ids(token,{});}
  void prefetch_target_prefix(int32_t token,const std::vector<int32_t>& drafts) {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_PLE_DRAFT_PREFIX");
    if(!option || option[0]!='0')prefetch_target_ids(token,drafts);
  }
  void prefetch_committed_token(int32_t token) {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_PLE_PUBLICATION");
    if(option && option[0]=='0')return;
    if(prefetch_target_ids(token,{})) {
      static bool reported=false;
      if(!reported) {
        reported=true;std::cerr<<"runtime_mtp_token_row_publication enabled=1\n";
      }
    }
  }
  bool stream_prompt_history(int64_t tokens) const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_STREAM_PROMPT_HISTORY");
    return mtp_stream_prompt_history(model,tokens,option);
  }
  bool host_prompt_history(int64_t tokens) const {
    return mtp_host_prompt_history(model,tokens);
  }
  void trace_prompt_prefill(double target_ms,double predictor_ms,double workspace_ms) const {
    // Block predictors seed only their bounded target window and skip the
    // teacher-forced priming hook that restores workspace for chain predictors.
    if(!mtp.teacher_forced_prompt_prime() && budget)
      budget->prepare(tyloi::engine::Budget::Stage::decode, source);
    std::cerr<<"runtime_mtp_prefill target_ms="<<target_ms<<" predictor_ms="<<predictor_ms
        <<" predictor_workspace_ms="<<workspace_ms<<" streamed="<<stream_prompt_history(input_ids.size(1))<<std::endl;
  }
  bool bootstrap_before_publication() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_EARLY_BOOTSTRAP");
    return mtp_bootstrap_before_publication(model,option);
  }
  bool retains_partial_target_prefix(int accepted) const {
    return mtp_retains_partial_target_prefix(model,mtp,accepted);
  }
  std::size_t predictor_history_merge_limit() const {
    return mtp_history_merge_limit(model);
  }
  std::size_t predictor_history_flush_threshold() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return mtp_history_flush_threshold(model,option);
  }
  std::size_t predictor_parked_history_flush_threshold() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_HISTORY_EARLY_FLUSH");
    return mtp_parked_history_flush_threshold(model,option);
  }
  void trace_cycle(uint64_t cycle,int depth,int accepted,double draft_ms,double verify_ms,double commit_ms) {
      const auto* option=tyloi::cuda::env_lookup("TYLOINFER_PROFILE_MTP");
      const bool profile=option && option[0]=='1';
      if(profile || (cycle<=32 && draft_ms+verify_ms+commit_ms>=55.0)) {
        std::cerr<<(profile?"runtime_mtp_cycle_profile cycle=":"runtime_mtp_slow_cycle cycle=")<<cycle<<" depth="<<depth<<" accepted="<<accepted
            <<" draft_ms="<<draft_ms<<" verify_ms="<<verify_ms<<" commit_ms="<<commit_ms
            <<" target_transfer_mb="<<target_transfer_bytes/1048576.0;
        if(profile)
          std::cerr<<" target_route_ms="<<target_profile[0]<<" target_fetch_ms="<<target_profile[1]
              <<" target_host_copy_ms="<<target_profile[2]<<" target_ssd_wait_ms="<<target_profile[3]
              <<" target_parallel_read_ms="<<target_profile[4]<<" target_physical_read_mb="<<target_profile[5];
        std::cerr<<'\n';
      }
  }

  CudaMtpOps(Model &model, State &state, tyloi::cuda::Executor* executor,
             tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
             ::tyloi::weights::Cache& weights,
             const tyloi::models::Info &info, const tyloi::ModelSource* source,
             tyloi::models::Mtp &mtp, tyloi::engine::mtp::State &mtp_state,
             const tyloi::cuda::PreparedPrompt *prepared,
             int64_t compute_chunk, InferenceOutput* output,
             tyloi::engine::Budget* budget = nullptr)
      : model(model), state(state), executor(executor), collectives(collectives), mode(mode),
        weights(weights), info(info), source(source), mtp(mtp), mtp_state(mtp_state), prepared(prepared),
        transformed_prompt(prepared && prepared->transformed()),
        position_axes(transformed_prompt &&
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                              prepared->positions && prepared->positions->defined() && prepared->positions->dim() == 2 &&
                              prepared->positions->size(0) == 3
#else
                              prepared->positions.defined() && prepared->positions.dim() == 2 &&
                              prepared->positions.size(0) == 3
#endif
                          ? 3
                          : 1), prefill_compute_chunk(compute_chunk), output(output),
        budget(budget) {
    target_graph_captures=moe_expert_graph_captures(::tyloi::weights::detail::cache(weights).moe);
  }
  void rollback_target(std::int64_t accepted) {
    if constexpr (has_replay<Model, State>)
      replay_target(model, state, executor, accepted);
    else
      model.rollback_speculative(accepted);
  }
  void initialize(const std::vector<int64_t> &prompt,
                  const MfqSamplingParams &sampling) {
    input_ids = ids_for(prompt);
    sampler_.emplace(sampling);
    penalties = sampler().has_penalties();
    mtp_initialize(model,*this);
    if (penalties) {
      counts = tyloi::cuda::SamplingOps::token_counts(input_ids, info.vocab_size);
    }
  }
  auto &sampler() { return *sampler_; }
  bool captured_compact_requested() const {
    const auto* option=tyloi::cuda::env_lookup("TYLOINFER_MTP_CONDITIONAL_TOPK");
    if(option && option[0]=='0')return true;
    const auto& parameters=sampler_->params();
    return !sampler_->greedy() && !penalties && parameters.top_k>0 && parameters.top_k<=64;
  }
  static void add_counts(DenseTensor counts, const DenseTensor &ids) {
    sample_token_counts_add_cuda(counts, ids);
  }
  static bool defined(const DenseTensor &value) { return value.defined(); }
  static int64_t size(const DenseTensor &value, int axis) {
    return value.size(axis);
  }
  static int rank(const DenseTensor &value) { return value.dim(); }
  static DenseTensor clone(const DenseTensor &value) {
    return value.is_cuda()?value.clone():value.to(::tyloi::kCUDA);
  }
  static DenseTensor slice(const DenseTensor &value, int axis, int64_t offset,
                      int64_t length) {
    return value.narrow(axis, offset, length).contiguous();
  }
  static DenseTensor reshape(DenseTensor value, std::initializer_list<int64_t> shape) {
    return value.reshape(shape);
  }
  static DenseTensor concatenate(const std::vector<DenseTensor> &values, int axis) {
    if(std::any_of(values.begin(),values.end(),[](const auto& value){return !value.is_cuda();})) {
      auto host=values;
      for(auto& value:host)value=value.to(::tyloi::kCPU);
      return ::tyloi::cuda::cat(host,axis).contiguous();
    }
    return ::tyloi::cuda::cat(values, axis).contiguous();
  }
  DenseTensor target_forward(DenseTensor ids, DenseTensor *raw = nullptr,
                        int64_t confirmed = 0) {
    projected_target_hidden = {}; projected_target_logits = {};
    projected_target_compact={};
    projected_target_greedy.clear();
    target_transfer_bytes = 0;
    const auto transfer_before=moe_expert_streamed_bytes(::tyloi::weights::detail::cache(weights).moe);
    const auto costs_before=profile_target()?target_costs():std::array<double,6>{};
    const auto finish_profile=[&] {
      if(!profile_target())return;
      const auto costs_after=target_costs();
      for(std::size_t i=0;i<target_profile.size();++i)target_profile[i]=costs_after[i]-costs_before[i];
    };
    if(auto result=mtp_target_forward(model,*this,ids,raw,confirmed,transfer_before,finish_profile))
      return std::move(*result);
    DenseTensor hidden;
    if constexpr (has_forward<Model, State>) {
      tyloi::cuda::Input input;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
      std::shared_ptr<tyloi::Tensor> raw_result;
      input.ids = std::make_shared<DenseTensor>(std::move(ids));
      input.raw_hidden = raw ? &raw_result : nullptr;
#else
      input.ids = std::move(ids);
      input.raw_hidden = raw;
#endif
      input.confirmed = confirmed;
      hidden = mtp_detail::hidden(model, state, executor, collectives, mode, std::move(input));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
      if (raw && raw_result) *raw = dynamic_cast<const DenseTensor&>(*raw_result);
#endif
    } else {
      hidden = model.hidden_forward(ids, mfq_nullopt, mfq_nullopt, nullptr,
                                    mfq_nullopt, raw, confirmed);
    }
    target_transfer_bytes=moe_expert_streamed_bytes(::tyloi::weights::detail::cache(weights).moe)-transfer_before;
    finish_profile();
    return hidden;
  }
  DenseTensor target_suffix(DenseTensor ids, DenseTensor *raw) {
    if constexpr (requires(Model& value) { value.supports_suffix_speculation(); }) {
      TYLOINFER_RUNTIME_CHECK(model.supports_suffix_speculation(),
                              "model does not support speculative suffix forwarding");
    }
    if constexpr (requires(Model& value, DenseTensor value_ids, DenseTensor* value_raw) {
                    value.hidden_forward_speculative_suffix(std::move(value_ids), value_raw);
                  }) {
      projected_target_hidden = {}; projected_target_logits = {};
      projected_target_compact={};
      projected_target_greedy.clear();
      const auto transfer_before=moe_expert_streamed_bytes(::tyloi::weights::detail::cache(weights).moe);
      const auto costs_before=profile_target()?target_costs():std::array<double,6>{};
      auto hidden=model.hidden_forward_speculative_suffix(ids, raw);
      target_transfer_bytes+=moe_expert_streamed_bytes(::tyloi::weights::detail::cache(weights).moe)-transfer_before;
      if(profile_target()) {
        const auto costs_after=target_costs();
        for(std::size_t i=0;i<target_profile.size();++i)target_profile[i]+=costs_after[i]-costs_before[i];
      }
      return hidden;
    } else {
      throw std::runtime_error("model does not support speculative suffix forwarding");
    }
  }
  struct PrefillResult {
    DenseTensor hidden, raw;
    double elapsed;
  };
  double blockwise_prompt_context(DenseTensor hidden,int64_t position,bool seed) {
    PrefillCudaTimer timer;
    if(seed)mtp.seed_target_context(hidden,position);
    else {
      const auto start=position-hidden.size(1);
      mtp.append_target_context(hidden,start);
    }
    TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  PrefillResult prefill(tyloi::engine::PrefillChunk chunk) {
    PrefillCudaTimer timer;
    bool prefill_finished=false;
    const auto finish_prefill=[&] {
      if(prefill_finished)return;
      TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
      prefill_finished=true;
    };
    DenseTensor raw;
    auto ids = input_ids.narrow(1, chunk.offset, chunk.count).contiguous();
    tyloi::cuda::ImageMaskScope image_scope(model,prepared,chunk.offset,chunk.count);
    if(auto result=mtp_prefill(model,*this,chunk,ids,raw,timer,finish_prefill))
      return std::move(*result);
    tyloi::weights::Loan workspace(weights);
    const bool host_history=host_prompt_history(chunk.count);
    const bool borrowed=transformed_prompt && generation_workspace(model,workspace,chunk.count);
    DenseTensor hidden;
    if (transformed_prompt) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
      auto embeddings = dynamic_cast<const DenseTensor&>(*prepared->embeddings).narrow(1, chunk.offset, chunk.count)
                            .contiguous();
      auto positions = prepared->positions && prepared->positions->defined()
                           ? MfqOptional<DenseTensor>(dynamic_cast<const DenseTensor&>(*prepared->positions).narrow(
                                 -1, chunk.offset, chunk.count).contiguous())
                           : mfq_nullopt;
#else
      auto embeddings = DenseTensor(prepared->embeddings).narrow(1, chunk.offset, chunk.count)
                            .contiguous();
      auto positions = prepared->positions.defined()
                           ? MfqOptional<DenseTensor>(DenseTensor(prepared->positions).narrow(
                                 -1, chunk.offset, chunk.count).contiguous())
                           : mfq_nullopt;
#endif
      if constexpr (has_forward<Model, State>) {
        tyloi::cuda::Input input;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        std::shared_ptr<tyloi::Tensor> raw_result;
        input.ids = std::make_shared<DenseTensor>(std::move(ids));
        input.embeddings = std::make_shared<DenseTensor>(std::move(embeddings));
        if (positions) input.positions = std::make_shared<DenseTensor>(std::move(*positions));
        input.raw_hidden = &raw_result;
#else
        input.ids = std::move(ids);
        input.embeddings = std::move(embeddings);
        if (positions) input.positions = std::move(*positions);
        input.raw_hidden = &raw;
#endif
        input.advance = true;
        hidden = mtp_detail::hidden(model, state, executor, collectives, mode, std::move(input));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (raw_result) raw = dynamic_cast<const DenseTensor&>(*raw_result);
#endif
      } else {
        hidden = model.hidden_forward_inputs(
            ids, std::move(embeddings), std::move(positions), mfq_nullopt, nullptr,
            mfq_nullopt, true, mfq_nullopt, &raw);
      }
    } else {
      hidden = target_forward(std::move(ids), &raw);
    }
    if(host_history)raw=raw.to(::tyloi::kCPU);
    finish_prefill();
    if(borrowed) {
      workspace.suspend();
      hidden=hidden.narrow(1,hidden.size(1)-1,1).clone();
      if(!host_history)raw=raw.clone();
      workspace.finish();
    }
    return {std::move(hidden), std::move(raw), timer.elapsed_ms()};
  }
  DenseTensor ids_for(std::vector<int64_t> tokens) {
    known_ids_host = ::tyloi::cuda::tensor(tokens, options.device(::tyloi::kCPU))
        .reshape({1, -1}).contiguous();
    known_ids_device = known_ids_host.to(::tyloi::kCUDA);
    return known_ids_device;
  }
  DenseTensor decode_positions(int64_t cache_start, int64_t tokens) {
    TYLOINFER_RUNTIME_CHECK(tokens > 0, "CUDA MTP position span must be positive");
    const int64_t logical_start =
        cache_start +
        (transformed_prompt ? prepared->decode_position_delta : 0);
    TYLOINFER_RUNTIME_CHECK(logical_start >= 0 && logical_start + tokens <=
                                                info.max_context,
                      "CUDA MTP position span exceeds context capacity");
    auto result = ::tyloi::cuda::arange(logical_start,
                                             logical_start + tokens, options);
    return position_axes == 3
               ? result.reshape({1, tokens}).expand({3, tokens}).contiguous()
               : result;
  }
  tyloi::models::MtpStep predictor_step(DenseTensor hidden, DenseTensor ids, DenseTensor positions) {
    if(auto result=mtp_predictor_step(model,*this,hidden,ids,positions))
      return std::move(*result);
    auto result=mtp.step(hidden, ids, positions.defined() ? &positions : nullptr);
    const bool initialized=mtp.take_graph_initialization();
    target_maintenance=target_maintenance || initialized;
    projected_predictor_logits = result.sample_logits
        ? dynamic_cast<const DenseTensor&>(*result.sample_logits) : DenseTensor{};
    projected_predictor_compact={};
    mtp_predictor_compact(model,*this,result);
    projected_predictor_greedy = result.greedy_token;
    if (projected_predictor_logits.defined()) {
      const auto& sample_hidden = dynamic_cast<const DenseTensor&>(*result.sample_hidden);
      projected_predictor_hidden = sample_hidden.narrow(1, sample_hidden.size(1)-1, 1).contiguous();
    } else projected_predictor_hidden = {};
    return result;
  }
  void predictor_history_step(DenseTensor hidden, DenseTensor ids, DenseTensor positions,bool scalar_chain=false) {
    if(mtp_predictor_history_step(model,*this,hidden,ids,positions,scalar_chain))return;
    (void)predictor_step(std::move(hidden),std::move(ids),std::move(positions));
  }
  int64_t predictor_prefill_chunk_size(int64_t requested) const {
    return mtp_predictor_prefill_chunk_size(model,requested);
  }
  bool cache_only_prompt_history() const {
    return mtp_cache_only_prompt_history(model,*this);
  }
  double begin_predictor_prefill(int64_t pairs) {
    const auto started=std::chrono::steady_clock::now();
    return mtp_begin_predictor_prefill(model,*this,pairs,started);
  }
  double end_predictor_prefill() {
    if(predictor_loan){predictor_loan->finish();predictor_loan.reset();}
    if (budget) budget->prepare(tyloi::engine::Budget::Stage::decode, source);
    // Every predictor tile has already synchronized its compute-end event.
    // Returning the expert loan is TTFT maintenance, not prompt computation.
    return 0;
  }
  double predictor_prefill(DenseTensor hidden, DenseTensor ids, DenseTensor positions) {
    mtp_predictor_prefill_begin(model,*this,ids);
    PrefillCudaTimer timer;
    bool prefill_finished=false;
    if(!cache_only_prompt_history())hidden=hidden.to(::tyloi::kCUDA);
    mtp_predictor_prefill(model,*this,hidden,ids,positions,timer,prefill_finished);
    if(!prefill_finished)
      TYLOINFER_CUDA_CHECK(cudaEventRecord(timer.finished_event(),mfq_get_current_cuda_stream()));
    return timer.elapsed_ms();
  }
  int32_t sample_normal(DenseTensor logits, DenseTensor token_counts) {
    if(sampler().greedy() && !penalties) {
      if(projected_predictor_greedy>=0 && projected_predictor_logits.defined() &&
          logits.data_ptr()==projected_predictor_logits.data_ptr() && logits.numel()==projected_predictor_logits.numel())
        return projected_predictor_greedy;
      if(projected_target_greedy.size()==1 && projected_target_logits.defined() &&
          logits.data_ptr()==projected_target_logits.data_ptr() && logits.numel()==projected_target_logits.numel())
        return projected_target_greedy.front();
    }
    if (penalties)
      logits = logits.clone();
    return static_cast<int32_t>(
        tyloi::engine::sample_logits(sampler(), std::move(logits), token_counts)
            .template item<int64_t>());
  }
  std::vector<int32_t> greedy_tokens(DenseTensor logits) {
    TYLOINFER_RUNTIME_CHECK(!penalties && sampler().greedy() && logits.dim() == 2,
                      "batched greedy MTP sampling geometry is invalid");
    if(!projected_target_greedy.empty() && projected_target_logits.defined() &&
        logits.data_ptr()==projected_target_logits.data_ptr() && logits.numel()==projected_target_logits.numel())
      return projected_target_greedy;
    auto host = sampler().ops().sample_greedy(logits.contiguous()).cpu().contiguous();
    const auto* tokens = host.template data_ptr<int64_t>();
    return std::vector<int32_t>(tokens, tokens + host.numel());
  }
  int32_t sample_constrained(DenseTensor logits, DenseTensor token_counts,
                             const MfqTokenConstraintPtr &constraint) {
    if(!constraint && sampler().greedy() && !penalties) return sample_normal(logits,token_counts);
    if (penalties)
      logits = logits.clone();
    auto sampler_constraint =
        constraint ? constraint->clone() : MfqTokenConstraintPtr{};
    return static_cast<int32_t>(
        tyloi::engine::sample_logits(sampler(), std::move(logits), token_counts,
                                 sampler_constraint)
            .template item<int64_t>());
  }
  std::vector<float> probabilities(DenseTensor logits, DenseTensor token_counts,
                                   const MfqSamplingParams &parameters) {
    logits = logits.contiguous().reshape({1, -1});
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    return tyloi::cuda::SamplingOps::probabilities(std::move(logits), parameters);
  }
  DenseTensor captured_compact(const DenseTensor& logits,const MfqSamplingParams& parameters) {
    for(const auto& pair:std::array<std::pair<DenseTensor,DenseTensor>,2>{{
        {projected_predictor_logits,projected_predictor_compact},{projected_target_logits,projected_target_compact}}})
      if(pair.first.defined() && pair.second.defined() && pair.first.data_ptr()==logits.data_ptr() &&
          pair.first.numel()==logits.numel() && pair.second.size(0)==logits.size(0)) {
        if(++graph_sampling_reuses==1)std::cerr<<"runtime_mtp_graph_sampling captured_topk=64 host_rows="<<logits.size(0)<<'\n';
#ifdef TYLOINFER_MTP_VERIFY_GRAPH_TOPK
        static thread_local std::array<bool,33> checked{};
        if(!checked[logits.size(0)]) {
            auto expected=tyloi::cuda::SamplingOps::compact_radix_rows(logits,parameters);
            auto actual=tyloi::cuda::SamplingOps::compact_rows_from_host(pair.second,parameters);
            TYLOINFER_RUNTIME_CHECK(expected.size()==actual.size(),"captured MTP top-k row count differs");
            for(std::size_t row=0;row<actual.size();++row) {
                TYLOINFER_RUNTIME_CHECK(expected[row].tokens==actual[row].tokens,"captured MTP top-k token IDs differ");
                for(std::size_t i=0;i<actual[row].probabilities.size();++i)
                    TYLOINFER_RUNTIME_CHECK(std::abs(actual[row].probabilities[i]-expected[row].probabilities[i])<=1e-6,
                        "captured MTP top-k probabilities differ");
            }
            checked[logits.size(0)]=true;target_maintenance=true;
            std::cerr<<"runtime_mtp_graph_sampling_check rows="<<logits.size(0)<<" PASS\n";
        }
#endif
        return pair.second;
      }
    return {};
  }
  tyloi::engine::mtp::CompactDistribution
  compact_probabilities(DenseTensor logits, DenseTensor token_counts,
                        const MfqSamplingParams &parameters) {
    TYLOINFER_RUNTIME_CHECK(parameters.top_k > 0 && parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "compact CUDA MTP sampling requires top-k 1-64");
    logits = logits.contiguous().reshape({1, -1});
    if(!penalties)if(auto host=captured_compact(logits,parameters);host.defined())
        return std::move(tyloi::cuda::SamplingOps::compact_rows_from_host(host,parameters).front());
    if (penalties) {
      logits = logits.clone();
      sample_apply_penalties_cuda(
          logits, token_counts, parameters.presence_penalty,
          parameters.frequency_penalty, parameters.repetition_penalty);
    }
    return tyloi::cuda::SamplingOps::compact_probabilities(std::move(logits), parameters);
  }
  std::vector<tyloi::engine::mtp::CompactDistribution>
  compact_probability_rows(DenseTensor logits, const MfqSamplingParams &parameters) {
    TYLOINFER_RUNTIME_CHECK(!penalties && logits.dim() == 2 && parameters.top_k > 0 &&
                          parameters.top_k <= 64 &&
                          parameters.temperature > 0.0,
                      "batched compact CUDA MTP sampling geometry is invalid");
    logits = logits.contiguous();
    if(auto host=captured_compact(logits,parameters);host.defined())return tyloi::cuda::SamplingOps::compact_rows_from_host(host,parameters);
    return tyloi::cuda::SamplingOps::compact_probability_rows(std::move(logits), parameters);
  }
  DenseTensor logits_for(DenseTensor normalized) {
    const auto matches = [&](const DenseTensor& hidden) {
        return hidden.defined() && hidden.data_ptr()==normalized.data_ptr() && hidden.sizes()==normalized.sizes();
    };
    if(matches(projected_predictor_hidden)) return projected_predictor_logits;
    if(matches(projected_target_hidden)) return projected_target_logits;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if constexpr (requires(Model& value, DenseTensor value_hidden) {
                    static_cast<const tyloi::models::CausalModel&>(value).logits(std::move(value_hidden), true);
                  }) {
      auto input = (mtp.preserve_output_dtype()
                        ? normalized
                        : normalized.to(::tyloi::kFloat16))
                       .contiguous();
      const tyloi::models::CausalModel& causal = model;
      auto scope = tyloi::cuda::Executor::current().enter(tyloi::cuda::Ffn::Mode{});
      auto result = causal.logits(input, true);
      TYLOINFER_RUNTIME_CHECK(result != nullptr, "CUDA model logits returned no tensor");
      return dynamic_cast<const DenseTensor&>(*result);
    } else
#endif
    if constexpr (requires(Model& value, tyloi::cuda::Executor& value_executor, DenseTensor value_hidden) {
                    logits(value, value_executor, std::move(value_hidden), true);
                  }) {
      TYLOINFER_RUNTIME_CHECK(executor != nullptr,
                              "CUDA model logits requires an executor");
      auto input = (mtp.preserve_output_dtype()
                        ? normalized
                        : normalized.to(::tyloi::kFloat16))
                       .contiguous();
      return logits(model, *executor, std::move(input), true);
    } else {
      auto input = (mtp.preserve_output_dtype()
                        ? normalized
                        : normalized.to(::tyloi::kFloat16))
                       .contiguous();
      return model.logits_from_hidden(std::move(input));
    }
  }
};

} // namespace mtp_detail
