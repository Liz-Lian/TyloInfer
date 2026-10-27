#include "backends/cuda/engine/weights/load.h"
#include "engine/weights/group.h"
#include "ops.h"
#include "models/common/session_codec.h"
#include "linear_attention.h"
#include "engine/executor/ffn.h"

#include <type_traits>

#include "backends/cuda/engine/models/load.h"

namespace mfq::cuda::qwen35 {
struct BlockLoader : TransformerWeightLoader {
    using Ffn = mfq::cuda::Ffn;
    auto full_block() const {
        auto block=std::make_unique<FullBlock>();block->kv_quantization=cuda_kv_quantization();return block;
    }
    auto linear_block() const { return std::make_unique<LinearAttentionBlock>(); }
    template <class T> static std::unique_ptr<::Block> finish(std::unique_ptr<T> block) {
        if constexpr (std::is_same_v<T, LinearAttentionBlock>) {
            const auto& config = block->qwen_config;
            block->key_heads = config.linear_num_key_heads;
            block->value_heads = config.linear_num_value_heads;
            block->key_dim = config.linear_key_head_dim;
            block->value_dim = config.linear_value_head_dim;
            block->conv_kernel = config.linear_conv_kernel_dim;
            block->rms_norm_eps = config.rms_norm_eps;
            block->norm_weight_offset = config.legacy_tensor_layout.norm_weight_offset;
        }
        return block;
    }
    static auto log_negative(mfq_tensor_backend::Tensor value) { return mfq_tensor_backend::log(-value); }
    void linear_projections(LinearAttentionBlock& b,
        const mfq::models::qwen35::LinearWeightNames& names, bool split_qkv) const {
        if (split_qkv) {
            b.split_in_proj = true;
            b.qkv_proj =
                load_quant_group(execution, source, {names.qk, names.value});
            if (is_quant_dtype(require_tensor(source, names.gate).dtype)) {
                b.z_proj = load_quant_linear(execution, source, names.gate);
                const bool a_nint = is_quant_dtype(require_tensor(source, names.alpha).dtype);
                const bool b_nint = is_quant_dtype(require_tensor(source, names.beta).dtype);
                if (a_nint != b_nint)
                    throw std::runtime_error("linear_attn a/b must use the same storage kind");
                b.ab_is_nint = a_nint;
                if (b.ab_is_nint) {
                    const auto scalar_axis =
                        execution.config.tensor_parallel_mirror_linear_attention_scalars
                            ? std::optional<TensorParallelAxis>(TensorParallelAxis::Mirrored)
                            : std::nullopt;
                    b.ab_nint_proj = make_quant_group(
                        {
                            load_quant_linear(execution, source, names.alpha, scalar_axis),
                            load_quant_linear(execution, source, names.beta, scalar_axis),
                        }, execution.config.diagnostic_nint_group, execution.loading_cpu_layer);
                } else {
                    b.ab_proj = make_dense_group({
                        load_dense_gpu(execution, source, names.alpha),
                        load_dense_gpu(execution, source, names.beta),
                    });
                }
            } else {
                b.split_dense_zab = true;
                b.zab_proj = make_dense_group({
                    load_dense_gpu(execution, source, names.gate),
                    load_dense_gpu(execution, source, names.alpha),
                    load_dense_gpu(execution, source, names.beta),
                });
            }
        } else {
            const bool alpha_quant = is_quant_dtype(require_tensor(source, names.alpha).dtype);
            const bool beta_quant = is_quant_dtype(require_tensor(source, names.beta).dtype);
            if (alpha_quant != beta_quant) {
                throw std::runtime_error(
                    "linear_attention alpha/beta must use the same storage kind");
            }
            if (alpha_quant) {
                b.in_proj = load_quant_group(
                    execution, source,
                    {names.qkv, names.gate, names.alpha, names.beta});
            } else {
                b.dense_ab_tail = true;
                b.in_proj =
                    load_quant_group(execution, source, {names.qkv, names.gate});
                b.ab_proj = make_dense_group({
                    load_dense_gpu(execution, source, names.alpha),
                    load_dense_gpu(execution, source, names.beta),
                });
            }
        }
    }
    void linear_output(LinearAttentionBlock& b, const std::string& name) const {
        if (is_quant_dtype(require_tensor(source, name).dtype)) {
            b.out_proj = load_quant_linear(execution, source, name);
        } else {
            b.dense_out_proj = true;
            b.out_proj_dense = load_dense_gpu(execution, source, name);
            if (require_tensor(source, name).dtype == "F16") {
                b.out_proj_dense = b.out_proj_dense.to(mfq_tensor_backend::kFloat16).contiguous();
            }
            if (b.out_proj_dense.dim() != 2) {
                throw std::runtime_error("dense linear_attention output projection must be 2D");
            }
        }
    }
};

std::unique_ptr<::Block> load_block(CudaExecutionContext &execution, const mfq::ModelSource &source,
                                    const Config &config, int layer, const std::string &type,
                                    std::string_view tensor_root) {
    BlockLoader loader{{execution, source}};
    return mfq::models::qwen35::load_block(loader, config, layer, type, tensor_root);
}

void LinearAttentionBlock::reset(int64_t batch) {
    clear_speculative();
    mixer.reset(batch);
}

void LinearAttentionBlock::prepare(const mfq_tensor_backend::Tensor& input, bool transposed) {
    if (mixer.defined()) return;
    const auto batch = input.size(0);
    const auto width = 2 * key_heads * key_dim + value_heads * value_dim;
    auto options = input.options().dtype(mfq_tensor_backend::kFloat32);
    mixer.view() = {
        mfq_tensor_backend::zeros({batch, conv_kernel - 1, width}, options),
        mfq_tensor_backend::zeros({batch, value_heads, value_dim, value_dim}, options),
        input.is_cuda() && transposed};
}

mfq_tensor_backend::Tensor LinearAttentionBlock::forward_ffn(
    CudaExecutionContext& execution, mfq_tensor_backend::Tensor residual,
    mfq_tensor_backend::Tensor output) {
    return execution.ffn().linear(
        ffn, ffn_norm, *this, std::move(residual), std::move(output));
}

mfq_tensor_backend::Tensor LinearAttentionBlock::forward(
        CudaExecutionContext& execution, mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor, int64_t,
        const MfqOptional<mfq_tensor_backend::Tensor>&, const RopeCache&,
        const MfqOptional<mfq_tensor_backend::Tensor>&,
        const MfqOptional<mfq_tensor_backend::Tensor>&) {
    prepare(input, execution.config.gdn_transposed_state);
    auto& state = mixer.view();
    auto attention = LinearAttention::forward(
            execution.projector(), execution.profiler,
            {execution.config.tasks.projection.grouped &&
                 execution.config.tensor_parallel_shared_linear_attention_input,
             execution.config.linear_conv_prefill_fused},
            std::move(input),
        {state.convolution, state.recurrent, state.transposed});
    return forward_ffn(execution, std::move(attention[0]), std::move(attention[1]));
}

void LinearAttentionBlock::clear_speculative() noexcept {
    speculative_inputs = LinearInputs();
    mixer.commit();
}

mfq_tensor_backend::Tensor LinearAttentionBlock::forward_context(CudaExecutionContext &execution,
                                                                 mfq_tensor_backend::Tensor input,
                                                                 const Block::Context &context,
                                                                 const RopeCache &rope) {
    if (context.confirmed_prefix == 0) {
        return Block::forward_context(execution, std::move(input), context, rope);
    }
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && !mixer.pending() &&
                          context.confirmed_prefix > 0 &&
                          context.confirmed_prefix < input.size(1) &&
                          mixer.defined(),
                      "invalid Qwen3.5 speculative linear-attention transaction");

    mixer.begin(context.cache_position, context.confirmed_prefix, input.size(1));

    try {
        auto& state = mixer.view();
        // Match Metal: evaluate the complete [confirmed, drafts...] window
        // once so projections and FFN stay batched. Rollback replays only the
        // recurrent state prefix from this layer's retained projections.
        auto attention = LinearAttention::forward(
            execution.projector(), execution.profiler,
            {execution.config.tasks.projection.grouped &&
                 execution.config.tensor_parallel_shared_linear_attention_input,
             execution.config.linear_conv_prefill_fused},
            std::move(input),
            {state.convolution, state.recurrent, state.transposed}, &speculative_inputs);
        auto result = forward_ffn(execution, std::move(attention[0]), std::move(attention[1]));
        ++speculative_projection_batches;
        ++speculative_ffn_batches;
        return result;
    } catch (...) {
        mixer.abort();
        clear_speculative();
        throw;
    }
}

void LinearAttentionBlock::commit_speculative() noexcept { clear_speculative(); }

void LinearAttentionBlock::rollback_speculative(int64_t keep_position) {
    try {
        const int64_t retained = mixer.rewind(keep_position);
        auto& state = mixer.view();
        // Restore only convolution/GDN state from the retained projected
        // rows; target projections, output projection and FFN are not rerun.
        LinearAttention::replay({state.convolution, state.recurrent, state.transposed},
                                speculative_inputs, retained);
        clear_speculative();
    } catch (...) {
        mixer.abort();
        clear_speculative();
        throw;
    }
}

bool supports_text_session_state(const std::vector<std::unique_ptr<::Block>> &blocks) {
    return !blocks.empty() &&
           std::all_of(blocks.begin(), blocks.end(), [](const std::unique_ptr<::Block> &block) {
               return dynamic_cast<const FullBlock *>(block.get()) != nullptr ||
                      dynamic_cast<const LinearAttentionBlock *>(block.get()) != nullptr;
           });
}

TextSessionState capture_text_session_state(const std::vector<std::unique_ptr<::Block>> &blocks,
                                            const std::vector<std::int64_t> &tokens,
                                            std::int64_t cache_position) {
    if (!supports_text_session_state(blocks) || cache_position <= 0 ||
        static_cast<std::size_t>(cache_position) != tokens.size()) {
        throw std::runtime_error("Qwen hybrid session token count does not match the cache");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_position;
    state.payload = std::vector<HybridBlockSessionState>{};
    auto &layers = std::get<std::vector<HybridBlockSessionState>>(state.payload);
    layers.reserve(blocks.size());
    for (const auto &block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        HybridBlockSessionState saved;
        if (const auto *full = dynamic_cast<const FullBlock *>(block.get())) {
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention =
                capture_kv(full->cache, cache_position, state.bytes);
        } else if (const auto *linear = dynamic_cast<const LinearAttentionBlock *>(block.get())) {
            if (linear->mixer.pending() || !linear->mixer.defined()) {
                throw std::runtime_error("Qwen recurrent session state is unavailable");
            }
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.mixer = linear->mixer.snapshot();
            state.bytes += session_tensor_bytes(saved.mixer.convolution);
            state.bytes += session_tensor_bytes(saved.mixer.recurrent);
        } else {
            throw std::runtime_error("Qwen hybrid session layer type changed");
        }
        layers.push_back(std::move(saved));
    }
    return state;
}

void restore_text_session_state(std::vector<std::unique_ptr<::Block>> &blocks,
                                const TextSessionState &state) {
    const auto *layers = std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
    if (state.kind() != TextSessionStateKind::HybridAttention || state.cache_pos <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_pos) || layers == nullptr ||
        layers->size() != blocks.size()) {
        throw CudaSessionStateError("Qwen hybrid session state is incompatible");
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto &block = blocks[index];
        const auto &saved = (*layers)[index];
        MfqCudaGuard guard(block->cuda_device);
        if (saved.kind == HybridBlockSessionStateKind::FullAttention) {
            auto *full = dynamic_cast<FullBlock *>(block.get());
            if (full == nullptr) {
                throw CudaSessionStateError("Qwen hybrid full-attention layer changed");
            }
            validate_kv(saved.full_attention);
            continue;
        }
        auto *linear = dynamic_cast<LinearAttentionBlock *>(block.get());
        const auto convolution_width = linear != nullptr ? 2 * linear->qwen_config.linear_k_size() +
                                                               linear->qwen_config.linear_v_size()
                                                         : 0;
        if (saved.kind != HybridBlockSessionStateKind::Recurrent)
            throw CudaSessionStateError("Qwen hybrid recurrent layer kind changed");
        if (linear == nullptr || !saved.mixer.convolution.defined() ||
            !saved.mixer.recurrent.defined() ||
            saved.mixer.convolution.scalar_type() != mfq_tensor_backend::kFloat32 ||
            saved.mixer.recurrent.scalar_type() != mfq_tensor_backend::kFloat32 ||
            saved.mixer.convolution.dim() != 3 || saved.mixer.convolution.size(0) != 1 ||
            saved.mixer.convolution.size(1) != linear->qwen_config.linear_conv_kernel_dim - 1 ||
            saved.mixer.convolution.size(2) != convolution_width ||
            saved.mixer.recurrent.dim() != 4 || saved.mixer.recurrent.size(0) != 1 ||
            saved.mixer.recurrent.size(1) != linear->qwen_config.linear_num_value_heads ||
            saved.mixer.recurrent.size(2) != linear->qwen_config.linear_value_head_dim ||
            saved.mixer.recurrent.size(3) != linear->qwen_config.linear_value_head_dim ||
            !saved.mixer.convolution.is_cuda() || !saved.mixer.recurrent.is_cuda() ||
            saved.mixer.convolution.get_device() != block->cuda_device ||
            saved.mixer.recurrent.get_device() != block->cuda_device) {
            throw CudaSessionStateError("Qwen recurrent session topology changed");
        }
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto& block = blocks[index];
        const auto& saved = (*layers)[index];
        MfqCudaGuard guard(block->cuda_device);
        if (auto* full = dynamic_cast<FullBlock*>(block.get())) {
            restore_kv(full->cache, saved.full_attention, full->attention_head_dim,
                full->official_bf16 ? mfq_tensor_backend::kBFloat16 : mfq_tensor_backend::kFloat16,
                full->kv_quantization);
        } else if (auto* linear = dynamic_cast<LinearAttentionBlock*>(block.get())) {
            linear->mixer.restore(saved.mixer);
            linear->clear_speculative();
        }
    }
}

} // namespace mfq::cuda::qwen35

namespace mfq::cuda {

void Qwen35Model::adapter_validate_components(const mfq::ModelGraph &graph) const {
    if (graph.component("vision") != nullptr &&
        cuda_model_plan(graph).vision != CudaVisionAdapter::grid_vit) {
        throw std::runtime_error("Qwen CUDA vision requires grid_vit/grid_vision.v1/grid_mrope");
    }
}

void Qwen35Model::adapter_configure_rope(RopeCache &rope, mfq_tensor_backend::Device device) const {
    rope.configure_mrope(config.mrope_sections, config.mrope_interleaved, config.rotary_dim,
                         device);
}

bool Qwen35Model::adapter_uses_common_rope() const noexcept { return true; }

bool Qwen35Model::adapter_supports_dense_cpu_offload() const noexcept { return true; }

std::unique_ptr<Block> Qwen35Model::adapter_load_block(const mfq::ModelSource &source, int layer,
                                                       int, const std::string &type) {
    return qwen35::load_block(*execution, source, config, layer, type);
}

TextSessionStateKind CudaSessionCodec<Qwen35Model>::kind(const Model &model) {
    const auto full = FullAttentionSessionCodec<Qwen35Model>::kind(model);
    if (full != TextSessionStateKind::Unsupported)
        return full;
    return qwen35::supports_text_session_state(model.blocks) ? TextSessionStateKind::HybridAttention
                                                             : TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<Qwen35Model>::supports_paged(const Model &model) {
    return FullAttentionSessionCodec<Qwen35Model>::supports_paged(model);
}

TextSessionState CudaSessionCodec<Qwen35Model>::capture(const Model &model,
                                                        const std::vector<int64_t> &tokens) {
    if (kind(model) == TextSessionStateKind::HybridAttention) {
        if (model.cache_pos <= 0 || static_cast<size_t>(model.cache_pos) != tokens.size()) {
            throw std::runtime_error("text session token count does not match the model cache");
        }
        return qwen35::capture_text_session_state(model.blocks, tokens, model.cache_pos);
    }
    return FullAttentionSessionCodec<Qwen35Model>::capture(model, tokens);
}

void CudaSessionCodec<Qwen35Model>::restore(Model &model, const TextSessionState &state) {
    if (kind(model) == TextSessionStateKind::HybridAttention &&
        state.kind() == TextSessionStateKind::HybridAttention) {
        qwen35::restore_text_session_state(model.blocks, state);
        model.cache_pos = state.cache_pos;
        return;
    }
    FullAttentionSessionCodec<Qwen35Model>::restore(model, state);
}

mfq_tensor_backend::Tensor Qwen35Model::adapter_embed(mfq_tensor_backend::Tensor output) const {
    return output.to(mfq_tensor_backend::kFloat16).contiguous();
}

bool Qwen35Model::adapter_supports_prepared_prompt() const noexcept { return true; }

} // namespace mfq::cuda

namespace mfq::cuda {

template struct FullAttentionSessionCodec<Qwen35Model>;

} // namespace mfq::cuda

namespace mfq::models {
template struct qwen35::CausalLm<cuda::CudaCausalOps<cuda::Qwen35Model>>;
} // namespace mfq::models
