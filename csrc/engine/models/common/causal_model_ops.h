#pragma once

#include "engine/models/common/causal_metadata.h"

#include "engine/models/common/capabilities.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tyloi { class ModelSource; }
namespace tyloi::cuda::weight_loader { struct Loader; }

enum class TextSessionStateKind : int;
struct TextSessionState;

struct CudaExecutionContext;

namespace tyloi::cuda {

// What the layer loop hands each block: the family-local geometry of one
// forward step. Blocks are plain structs now; this bag belongs to the loop.
struct BlockContext {
    ::tyloi::DenseTensor token_ids;
    ::tyloi::DenseTensor host_token_ids;
    ::tyloi::DenseTensor positions;
    ::tyloi::DenseTensor full_positions;
    std::int64_t cache_position = 0;
    std::int64_t confirmed_prefix = 0;
    std::int64_t planned_kv_length = 0;
    std::int64_t decode_attention_parts = 0;
    MfqOptional<::tyloi::DenseTensor> sequence_lengths = mfq_nullopt;
    MfqOptional<::tyloi::DenseTensor> cache_positions = mfq_nullopt;
    MfqOptional<::tyloi::DenseTensor> attention_mask = mfq_nullopt;
};

struct CausalResources {
    tyloi::models::CausalLmMetadata metadata;
    CudaExecutionContext *execution = nullptr;
    std::function<void()> restore_decode_workspace;
    template<class Model> std::optional<::tyloi::DenseTensor>
    adapter_graph_logits(Model&,::tyloi::DenseTensor,int) {return {};}

    void adapter_validate_load_options() const;
    bool adapter_uses_common_rope() const noexcept;
    void adapter_configure_rope(RopeCache &rope, ::tyloi::Device device) const;
    void adapter_load_final_state(const tyloi::cuda::weight_loader::Loader& weights,
                                  ::tyloi::DenseTensor &output_norm);
    void adapter_prepare_blocks(const tyloi::cuda::weight_loader::Loader& weights);
    bool adapter_supports_dense_cpu_offload() const noexcept;

    ::tyloi::DenseTensor adapter_embed(::tyloi::DenseTensor output) const;
    void adapter_reset(int64_t batch);
    bool adapter_requires_batch_reset(int64_t batch) const noexcept;

    ::tyloi::DenseTensor adapter_prepare_hidden(::tyloi::DenseTensor hidden,
                                                      int64_t batch, int64_t tokens) const;
    void adapter_begin_forward(bool capture_raw_hidden);

    ::tyloi::DenseTensor
    adapter_block_positions(const ::tyloi::DenseTensor &full_positions,
                            const ::tyloi::DenseTensor &local_positions, int device) const;

    void adapter_finish_forward(const ::tyloi::DenseTensor &full_positions, int64_t batch,
                                int64_t tokens);

    ::tyloi::DenseTensor
    adapter_finalize_hidden(::tyloi::DenseTensor hidden,
                            const ::tyloi::DenseTensor &output_norm, int64_t batch,
                            int64_t tokens) const;
    ::tyloi::DenseTensor
    adapter_raw_hidden(const ::tyloi::DenseTensor &hidden,
                       const ::tyloi::DenseTensor &finalized) const;
    ::tyloi::DenseTensor adapter_logits(const ::tyloi::Tensor &lm_head,
                                              ::tyloi::DenseTensor hidden) const;
    ::tyloi::DenseTensor adapter_last_logits(const ::tyloi::Tensor &lm_head,
                                                   ::tyloi::DenseTensor hidden) const;
    ::tyloi::DenseTensor adapter_next_token(const ::tyloi::Tensor &lm_head,
                                                  ::tyloi::DenseTensor hidden) const;

    bool adapter_supports_prepared_prompt() const noexcept;

    void adapter_begin_speculative();
    void adapter_commit_speculative();
    void adapter_rollback_speculative(int64_t keep);
};
} // namespace tyloi::cuda
