#pragma once

#include "engine/models/common/capabilities.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/models/common/block.h"
#include "engine/models/common/causal_model.h"
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

namespace tyloi::models::qwen4_exp {
template <class Backend> struct CausalLm;
}
namespace tyloi::models::deepseek_v41 {
template <class Backend> struct CausalLm;
}

namespace tyloi::cuda {

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
    ::tyloi::DenseTensor adapter_logits(const QuantLinear &lm_head,
                                              ::tyloi::DenseTensor hidden) const;
    ::tyloi::DenseTensor adapter_last_logits(const QuantLinear &lm_head,
                                                   ::tyloi::DenseTensor hidden) const;
    ::tyloi::DenseTensor adapter_next_token(const QuantLinear &lm_head,
                                                  ::tyloi::DenseTensor hidden) const;

    bool adapter_supports_prepared_prompt() const noexcept;

    void adapter_begin_speculative();
    void adapter_commit_speculative();
    void adapter_rollback_speculative(int64_t keep);
};

struct Qwen4Model;
struct DeepseekV41Model;

template <typename Model> struct CudaCausalOps;
template <typename Model>
using CausalLm = typename Model::template CausalModel<CudaCausalOps<Model>>;

template <typename Model> struct CudaSessionCodec {
    using CausalModel = CausalLm<Model>;
    static TextSessionStateKind kind(const CausalModel &model);
    static bool supports_paged(const CausalModel &model);
    static TextSessionState capture(const CausalModel &model, const std::vector<int64_t> &tokens);
    static void restore(CausalModel &model, const TextSessionState &state);
};

struct CudaForwardPlan {
    int64_t kv_length = 0;
    int64_t attention_parts = 0;
};

template <typename Model> struct CudaCausalOps : Model {
    using DenseTensor = ::tyloi::DenseTensor;
    using ForwardPlan = CudaForwardPlan;
    using CausalModel = CausalLm<Model>;
    using SessionState = TextSessionState;
    using SessionStateKind = TextSessionStateKind;
    using SessionCodec = CudaSessionCodec<Model>;
    std::shared_ptr<const tyloi::ModelSource> source;
    tyloi::ModelGraph graph;
    CudaModelPlan plan;
    RopeCache rope;
    RopeCache cpu_rope;
    std::unordered_map<int, RopeCache> device_ropes;
    QuantLinear embed;
    std::vector<std::unique_ptr<Block>> blocks;
    ::tyloi::DenseTensor output_norm;
    QuantLinear lm_head;
    std::vector<DenseTensor*> graph_state() {
        std::vector<DenseTensor*> warmup_states;
        for (const auto& block : blocks) {
            for (auto* state : block->graph_warmup_state()) {
                warmup_states.push_back(state);
            }
        }
        return warmup_states;
    }
    std::optional<DenseTensor> graph_logits(CausalModel& model,DenseTensor ids,int kind) {
        return static_cast<Model&>(model).adapter_graph_logits(model,std::move(ids),kind);
    }

    auto execution_scope() const {
        return CudaScope(this->execution->config.layer_placement.primary_device(), this->execution->executor().contexts());
    }
    static auto block_scope(const std::unique_ptr<Block> &block) {
        return MfqCudaGuard(block->cuda_device);
    }
    static bool defined(const DenseTensor &value) { return value.defined(); }
    static DenseTensor position_axes(DenseTensor value, int64_t start, int64_t count) {
        return value.narrow(0, start, count);
    }
    static DenseTensor concat_positions(DenseTensor first, DenseTensor second) {
        // Generated text advances all semantic axes equally after an image.
        if (first.dim() > 1 && second.dim() == 1) {
            auto shape = first.sizes().vec();
            shape.back() = second.size(0);
            std::vector<int64_t> view(shape.size(),1);
            view.back() = second.size(0);
            second = second.reshape(view).expand(shape);
        }
        return ::tyloi::cuda::cat({first, second}, -1);
    }
    static int64_t rank(const DenseTensor &value) { return value.dim(); }
    static int64_t size(const DenseTensor &value, int axis) { return value.size(axis); }
    static DenseTensor batch_ids(DenseTensor ids) { return ids.unsqueeze(0); }
    DenseTensor device_ids(DenseTensor ids) const {
        return this->execution->executor().collectives().copy(ids.to(::tyloi::kInt64), this->execution->config.layer_placement.primary_device())
            .contiguous();
    }
    DenseTensor embed_tokens(DenseTensor ids) const {
        return this->execution->executor().profiler().measure(
            "model.embed", [&] { return this->adapter_embed(quant_embedding_lookup(embed, ids)); });
    }
    DenseTensor sequence_lengths(int64_t batch, int64_t end) const {
        auto scope = execution_scope();
        return ::tyloi::cuda::full({batch}, end,
                                        ::tyloi::TensorOptions()
                                            .dtype(::tyloi::kInt64)
                                            .device(::tyloi::kCUDA));
    }
    static DenseTensor last_hidden(DenseTensor hidden) { return hidden.index({Slice(), -1, Slice()}); }
    static DenseTensor softcap(DenseTensor logits, double cap) {
        return ::tyloi::cuda::tanh(logits / cap) * cap;
    }
    DenseTensor scale_hidden(DenseTensor hidden, double scale) const {
        return this->execution->executor().profiler().measure("model.embed_scale",
                                                 [&] { return hidden * scale; });
    }
    static DenseTensor repeat_hidden(DenseTensor hidden, int64_t streams) {
        return hidden.to(::tyloi::kFloat16).repeat({1, 1, streams});
    }
    static DenseTensor expand_hidden(DenseTensor hidden, int64_t batch, int64_t tokens, int64_t streams) {
        const auto width = hidden.size(-1);
        return hidden.to(::tyloi::kFloat16)
            .unsqueeze(2)
            .expand({batch, tokens, streams, width})
            .contiguous();
    }
    struct ForwardOps : tyloi::models::CausalForwardInputs<DenseTensor, ForwardPlan> {
        using Inputs = tyloi::models::CausalForwardInputs<DenseTensor, ForwardPlan>;
        CausalModel &model;
        ForwardOps(CausalModel &model, Inputs input)
            : Inputs(std::move(input)), model(model),
              primary(model.execution->config.layer_placement.primary_device()) {}
        int primary;
        DenseTensor pos, full_positions, cache_positions, cpu_ids, cpu_pos, cpu_cache_positions;
        MfqOptional<DenseTensor> effective_attention_mask, cpu_attention_mask, cpu_seq_len;

        static int64_t size(const DenseTensor &value, int axis) { return value.size(axis); }
        static int64_t rank(const DenseTensor &value) { return value.dim(); }
        static int64_t elements(const DenseTensor &value) { return value.numel(); }

        DenseTensor device_ids(DenseTensor value) {
            return model.execution->executor().collectives().copy(value.to(::tyloi::kInt64), primary)
                .contiguous();
        }
        void prepare_inputs() { ids = device_ids(ids); }
        bool has_mrope() const { return model.rope.sections.numel() > 0; }
        DenseTensor position_range(int64_t start, int64_t tokens) {
            return ::tyloi::cuda::arange(start, start + tokens,
                                              ::tyloi::TensorOptions()
                                                  .device(::tyloi::kCUDA)
                                                  .dtype(::tyloi::kInt64));
        }
        static DenseTensor offset_positions(DenseTensor positions, int64_t delta) {
            return positions + delta;
        }
        DenseTensor prepare_hidden(int64_t B, int64_t T) {
            if(model.execution->config.moe_pipeline) {
                cpu_ids=ids.to(::tyloi::kCPU,::tyloi::kInt64).contiguous();
                for(const auto& block:model.blocks)block->prefetch_token_ids(cpu_ids);
            }
            if (model.execution->config.dense_cpu_layer_count > 0) {
                cpu_ids = ids.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
                cpu_pos = pos.to(::tyloi::kCPU, ::tyloi::kInt64).contiguous();
                cpu_cache_positions =
                    cache_positions.to(::tyloi::kCPU, ::tyloi::kInt64)
                        .contiguous();
                if (seq_len.has_value()) {
                    cpu_seq_len = seq_len.value()
                                      .to(::tyloi::kCPU, ::tyloi::kInt64)
                                      .contiguous();
                }
                if (effective_attention_mask.has_value()) {
                    cpu_attention_mask =
                        effective_attention_mask.value().to(::tyloi::kCPU).contiguous();
                }
            }
            return model.adapter_prepare_hidden(
                model.execution->executor().collectives().copy(input_embeddings, primary)
                    .contiguous(),
                B, T);
        }
        DenseTensor layer(std::unique_ptr<Block> &b, DenseTensor x) {
            MfqCudaGuard block_guard(b->cuda_device);
            auto local_ids =
                b->cpu_offloaded
                    ? cpu_ids
                    : model.execution->executor().collectives().copy(ids, b->cuda_device);
            auto local_pos =
                b->cpu_offloaded
                    ? cpu_pos
                    : model.execution->executor().collectives().copy(pos, b->cuda_device);
            MfqOptional<::tyloi::DenseTensor> local_cache_positions = mfq_nullopt;
            // Grid-MRoPE semantic coordinates never double as physical KV
            // slots, during either prepared prefill or delta-adjusted decode.
            if (model.adapter_pass_cache_positions(model.rope.sections.numel() > 0,
                                                   cache_positions_override.has_value())) {
                local_cache_positions =
                    b->cpu_offloaded
                        ? cpu_cache_positions
                        : model.execution->executor().collectives().copy(cache_positions, b->cuda_device);
            }
            MfqOptional<::tyloi::DenseTensor> local_seq_len = mfq_nullopt;
            if (seq_len.has_value()) {
                local_seq_len =
                    b->cpu_offloaded
                        ? cpu_seq_len.value()
                        : model.execution->executor().collectives().copy(seq_len.value(), b->cuda_device);
            }
            MfqOptional<::tyloi::DenseTensor> local_attention_mask = mfq_nullopt;
            if (effective_attention_mask.has_value()) {
                local_attention_mask =
                    b->cpu_offloaded
                        ? cpu_attention_mask.value()
                        : model.execution->executor().collectives().copy(effective_attention_mask.value(), b->cuda_device);
            }
            x = b->cpu_offloaded
                    ? x.to(::tyloi::kCPU).contiguous()
                    : model.execution->executor().collectives().copy(x, b->cuda_device);
            b->set_token_ids(local_ids);
            const RopeCache &active_rope =
                b->cpu_offloaded
                    ? model.cpu_rope
                    : (model.device_ropes.empty() ? model.rope
                                                  : model.device_ropes.at(b->cuda_device));
            Block::Context context;
            context.token_ids = local_ids;
            context.host_token_ids=cpu_ids;
            context.positions = local_pos;
            context.full_positions =
                model.adapter_block_positions(full_positions, local_pos, b->cuda_device);
            context.cache_position = model.cache_pos;
            context.confirmed_prefix = confirmed_prefix;
            context.planned_kv_length = plan.kv_length;
            context.decode_attention_parts = plan.attention_parts;
            context.sequence_lengths = local_seq_len;
            context.cache_positions = local_cache_positions;
            if (model.adapter_pass_attention_mask()) {
                context.attention_mask = local_attention_mask;
            }
            x = b->forward_context(*model.execution, std::move(x), context, active_rope);
            return x;
        }
        void trace(const DenseTensor &x) {
            if (block_trace)
                block_trace->push_back(
                    model.execution->executor().collectives().copy(x, primary)
                        .to(::tyloi::kFloat32)
                        .clone());
        }
        DenseTensor finish(DenseTensor x, int64_t B, int64_t T) {
            x = model.execution->executor().collectives().copy(x, primary);
            auto finalized = model.finalize_hidden(x, B, T);
            if (raw_hidden)
                *raw_hidden = model.adapter_raw_hidden(x, finalized);
            return finalized;
        }
    };
    static ForwardOps forward_ops(CausalModel &model, typename ForwardOps::Inputs input) {
        return ForwardOps(model, std::move(input));
    }
};

using Qwen4CausalLm = tyloi::models::qwen4_exp::CausalLm<CudaCausalOps<Qwen4Model>>;
using DeepseekV41CausalLm = tyloi::models::deepseek_v41::CausalLm<CudaCausalOps<DeepseekV41Model>>;

} // namespace tyloi::cuda
