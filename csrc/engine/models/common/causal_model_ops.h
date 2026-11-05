#pragma once

#include "engine/models/common/causal_metadata.h"

#include "engine/models/common/capabilities.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tyloi { class ModelSource; }
namespace tyloi::weights { struct Loader; }

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
    bool adapter_uses_common_rope() const noexcept;
    void adapter_configure_rope(RopeCache &rope, ::tyloi::Device device) const;
    void adapter_load_final_state(const tyloi::weights::Loader& weights,
                                  ::tyloi::DenseTensor &output_norm);
    void adapter_prepare_blocks(const tyloi::weights::Loader& weights);
    ::tyloi::DenseTensor adapter_embed(::tyloi::DenseTensor output) const;
    ::tyloi::DenseTensor
    adapter_raw_hidden(const ::tyloi::DenseTensor &hidden,
                       const ::tyloi::DenseTensor &finalized) const;
    ::tyloi::DenseTensor adapter_logits(const ::tyloi::Tensor &lm_head,
                                              ::tyloi::DenseTensor hidden) const;
};
} // namespace tyloi::cuda
