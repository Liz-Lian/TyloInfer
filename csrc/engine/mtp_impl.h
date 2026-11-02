#pragma once

#include "engine/generate.h"
#include "engine/generation/mtp/sequence.h"
#include "engine/models/common/mtp_impl.h"

using mfq::engine::Generation;

template <typename Model, typename State>
Generation
run_mtp_generation(Model &model, State &state, mfq::cuda::Executor* executor,
                   mfq::cuda::Collectives& collectives, mfq::cuda::Executor::Mode mode,
                   mfq::cuda::RoutedCache& weights,
                   const mfq::models::CausalLmMetadata &metadata,
                   MtpModule &mtp, mfq::engine::mtp::State &mtp_state,
                   InferenceRequest &request,
                   InferenceOutput &output, int64_t prefill_chunk_size,
                   const mfq::cuda::PreparedPrompt *prepared,
                   std::size_t reused_tokens,
                   mfq_tensor_backend::Tensor restored_last_hidden,
                   mfq_tensor_backend::Tensor *session_last_hidden,
                   std::size_t stable_prefix_tokens, std::function<void()> restore) {
  auto logical_chunk=prefill_chunk_size;
  if constexpr(requires { prefill_span(model,request.prompt.size(),prefill_chunk_size,reused_tokens,0,false,false); }) {
    logical_chunk=prefill_span(model,request.prompt.size(),prefill_chunk_size,reused_tokens,0,
        prepared && prepared->transformed(),reused_tokens>0 && prefill_chunk_size<=8);
  }
  return mfq::engine::speculative_sequence(
      mtp_detail::CudaMtpOps<Model, State>{model, state, executor, collectives, mode, weights, metadata, mtp, mtp_state, prepared,
                               prefill_chunk_size, &output, std::move(restore)},
      request, output,
      logical_chunk, reused_tokens, std::move(restored_last_hidden),
      session_last_hidden,stable_prefix_tokens);
}
