#pragma once

#include "engine/generate.h"
#include "engine/generation/mtp/sequence.h"
#include "engine/models/common/mtp_impl.h"

using tyloi::engine::Generation;

template <typename Model, typename State>
Generation
run_mtp_generation(Model &model, State &state, tyloi::cuda::Executor* executor,
                   tyloi::cuda::Collectives& collectives, tyloi::cuda::Executor::Mode mode,
                   tyloi::cuda::RoutedCache& weights,
                   const tyloi::models::Info &info, const tyloi::ModelSource* source,
                   MtpModule &mtp, tyloi::engine::mtp::State &mtp_state,
                   InferenceRequest &request,
                   InferenceOutput &output, int64_t prefill_chunk_size,
                   const tyloi::cuda::PreparedPrompt *prepared,
                   std::size_t reused_tokens,
                   ::tyloi::DenseTensor restored_last_hidden,
                   ::tyloi::DenseTensor *session_last_hidden,
                   std::size_t stable_prefix_tokens, std::function<void()> restore) {
  auto logical_chunk=prefill_chunk_size;
  if constexpr(requires { prefill_span(model,request.prompt.size(),prefill_chunk_size,reused_tokens,0,false,false); }) {
    logical_chunk=prefill_span(model,request.prompt.size(),prefill_chunk_size,reused_tokens,0,
        prepared && prepared->transformed(),reused_tokens>0 && prefill_chunk_size<=8);
  }
  return tyloi::engine::speculative_sequence(
      mtp_detail::CudaMtpOps<Model, State>{model, state, executor, collectives, mode, weights, info, source, mtp, mtp_state, prepared,
                               prefill_chunk_size, &output, std::move(restore)},
      request, output,
      logical_chunk, reused_tokens, std::move(restored_last_hidden),
      session_last_hidden,stable_prefix_tokens);
}
