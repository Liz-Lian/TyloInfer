#pragma once

#include "ops/attention/linear.h"
#include "backends/cuda/engine/state/recurrent.h"
#include "models/common/block.h"
#include "engine/models/ffn.h"
#include "engine/models/qwen35/config.h"

namespace mfq::cuda::qwen35 {

struct LinearAttentionBlock final : ::Block, LinearAttention {
    mfq::models::qwen35::Config qwen_config;
    mfq_tensor_backend::Tensor ffn_norm;
    mfq::cuda::Ffn ffn;
    Recurrent mixer;
    int64_t speculative_ffn_batches = 0;
    int64_t speculative_projection_batches = 0;
    LinearInputs speculative_inputs;

    bool supports_speculation() const noexcept override { return true; }
    std::vector<mfq_tensor_backend::Tensor*> graph_warmup_state() override {
        TYLOINFER_RUNTIME_CHECK(!mixer.pending() && mixer.defined(),
                              "decode warmup requires confirmed recurrent state");
        return {&mixer.view().convolution, &mixer.view().recurrent};
    }
    void reset(int64_t batch) override;
    void clear_speculative() noexcept;
    void commit_speculative() noexcept override;
    void rollback_speculative(int64_t keep_position) override;
    void prepare(const mfq_tensor_backend::Tensor& input, bool transposed);
    mfq_tensor_backend::Tensor forward_ffn(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor attention_output);
    mfq_tensor_backend::Tensor forward_context(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor input,
        const Block::Context& context, const RopeCache& rope) override;
    mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor positions, int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask = mfq_nullopt) override;
};

} // namespace mfq::cuda::qwen35
