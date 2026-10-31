#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/executor/tq_moe.h"
#include "backends/cuda/engine/models/load.h"
#include "engine/executor/ffn.h"
#include "engine/weights/loader.h"
#include "ops.h"
#include "model.h"
#include "models/common/session_codec.h"

#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_selective_state_scan_ops.h"
#include "engine/models/common/recurrent_attention.h"
#include "engine/models/common/transformer_layer.h"

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda::nemotron_h {
namespace {

using Tensor = mfq_tensor_backend::Tensor;

// The canonical graph returns the last K-1 convolution inputs for the next
// call. Runtime CUDA graphs retain this storage address across requests.
Tensor next_convolution_state(const Tensor& conv_input, int64_t kernel) {
    return conv_input.narrow(1, conv_input.size(1) - (kernel - 1), kernel - 1).contiguous();
}

} // namespace

Tensor Mamba2MixerBlock::forward(CudaExecutionContext& execution, Tensor x, Tensor, int64_t,
                                 const MfqOptional<Tensor>&, const RopeCache&,
                                 const MfqOptional<Tensor>&, const MfqOptional<Tensor>&) {
    auto& profiler = execution.profiler;
    auto& state = mixer.view();
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    const int64_t heads = mamba_heads;
    const int64_t head_dim = mamba_head_dim;
    const int64_t groups = mamba_groups;
    const int64_t state_width = ssm_state_size;
    const int64_t inner = heads * head_dim;
    const int64_t conv_width = inner + 2 * groups * state_width;
    const int64_t projection_width = conv_width + inner + heads;
    const auto f32 = mfq_tensor_backend::TensorOptions()
                         .device(mfq_tensor_backend::kCUDA)
                         .dtype(mfq_tensor_backend::kFloat32);
    if (!mixer.defined()) {
        state = {
            mfq_tensor_backend::zeros({B, conv_kernel - 1, conv_width}, f32),
            mfq_tensor_backend::zeros({B, heads, head_dim, state_width}, f32), false};
    }
    TYLOINFER_RUNTIME_CHECK(state.convolution.size(0) == B &&
                          state.convolution.size(1) == conv_kernel - 1 &&
                          state.convolution.size(2) == conv_width &&
                          state.recurrent.size(0) == B && state.recurrent.size(1) == heads &&
                          state.recurrent.size(2) == head_dim &&
                          state.recurrent.size(3) == state_width,
                      "mamba state geometry changed");
    const auto bias = conv_bias.defined()
                          ? conv_bias
                          : mfq_tensor_backend::empty({0}, f32);
    return mfq::models::attention_layer(
        std::move(x),
        [&](Tensor x) {
            return profiler.measure("mamba.norm", [&] {
                return qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                     mixer_norm, rms_norm_eps, 0.0)
                    .reshape({B, T, H});
            });
        },
        [&](Tensor xn) {
            return mfq::models::selective_state_attention<Tensor>(
                true,
                [&]() {
                    auto projected = profiler.measure("mamba.in_proj", [&] {
                        return execution.projector().run(in_proj, xn).to(mfq_tensor_backend::kFloat32);
                    });
                    TYLOINFER_RUNTIME_CHECK(projected.size(-1) == projection_width,
                                      "mamba fused projection width disagrees with config");
                    auto gate = projected.narrow(2, 0, inner).contiguous();
                    auto xbc = projected.narrow(2, inner, conv_width).contiguous();
                    auto dt = projected.narrow(2, inner + conv_width, heads).contiguous();
                    if (mixer.pending()) {
                        speculative_xbc = xbc;
                        speculative_dt = dt;
                    }
                    return std::array<Tensor, 3>{std::move(gate), std::move(xbc), std::move(dt)};
                },
                [&](const std::array<Tensor, 3>& projected) {
                    const auto& xbc = projected[1];
                    auto convolved = profiler.measure("mamba.conv", [&] {
                        if (T == 1)
                            return ssm_conv_silu_decode_cuda(state.convolution, xbc, conv_weight,
                                                             bias);
                        auto conv_input = mfq_tensor_backend::cat({state.convolution, xbc}, 1);
                        state.convolution.copy_(next_convolution_state(conv_input, conv_kernel));
                        return ssm_conv_silu_cuda(conv_input, conv_weight, bias, T);
                    });
                    auto x = convolved.narrow(2, 0, inner)
                                 .reshape({B, T, heads, head_dim})
                                 .contiguous();
                    auto b = convolved.narrow(2, inner, groups * state_width)
                                 .reshape({B, T, groups, state_width})
                                 .contiguous();
                    auto c = convolved.narrow(2, inner + groups * state_width, groups * state_width)
                                 .reshape({B, T, groups, state_width})
                                 .contiguous();
                    return std::array<Tensor, 3>{std::move(x), std::move(b), std::move(c)};
                },
                [&](const std::array<Tensor, 3>& signals, const Tensor& dt) {
                    return profiler.measure("mamba.scan", [&] {
                        if (T == 1) {
                            auto y = selective_state_scan_decode_cuda(
                                state.recurrent, signals[0].select(1, 0).contiguous(),
                                dt.select(1, 0).contiguous(), dt_bias, a_log,
                                signals[1].select(1, 0).contiguous(),
                                signals[2].select(1, 0).contiguous(), d, time_step_limit_lo,
                                time_step_limit_hi);
                            return std::array<Tensor, 2>{y.unsqueeze(1), Tensor()};
                        }
                        auto scanned = selective_state_scan_prefill_cuda(
                            signals[0], dt, dt_bias, a_log, signals[1], signals[2], d,
                            state.recurrent, time_step_limit_lo, time_step_limit_hi);
                        return std::array<Tensor, 2>{std::move(scanned[0]), std::move(scanned[1])};
                    });
                },
                [&](Tensor scanned, Tensor gate) {
                    return profiler.measure("mamba.gated_norm", [&] {
                        return grouped_gated_rms_norm_cuda(
                                   scanned.reshape({B * T, inner}),
                                   gate.reshape({B * T, inner}), scan_norm,
                                   inner / groups, rms_norm_eps)
                            .reshape({B, T, inner});
                    });
                },
                [&](Tensor normalized) {
                    return profiler.measure(
                        "mamba.out_proj",
                        [&] { return execution.projector().run(out_proj, normalized); });
                },
                [&](const Tensor& final_state) {
                    if (final_state.defined())
                        state.recurrent.copy_(final_state);
                });
        },
        [&](Tensor residual, Tensor mixed) {
            auto rr = residual.reshape({-1, H});
            auto mm = mixed.reshape({-1, H});
            if (mm.scalar_type() != rr.scalar_type())
                mm = mm.to(rr.scalar_type()).contiguous();
            return acc_cuda(rr, mm).reshape({B, T, H});
        });
}

Tensor Mamba2MixerBlock::forward_context(CudaExecutionContext& execution, Tensor x,
                                         const Block::Context& context, const RopeCache& rope) {
    if (context.confirmed_prefix == 0)
        return Block::forward_context(execution, std::move(x), context, rope);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && !mixer.pending() && context.confirmed_prefix > 0 &&
                          context.confirmed_prefix < x.size(1) && mixer.defined(),
                      "invalid Nemotron-H speculative mamba transaction");
    mixer.begin(context.cache_position, context.confirmed_prefix, x.size(1));
    try {
        return forward(execution, std::move(x), context.full_positions, context.cache_position,
                       context.sequence_lengths, rope, context.cache_positions,
                       context.attention_mask);
    } catch (...) {
        mixer.abort();
        clear_speculative();
        throw;
    }
}

void Mamba2MixerBlock::clear_speculative() noexcept {
    speculative_xbc = Tensor();
    speculative_dt = Tensor();
    mixer.commit();
}

void Mamba2MixerBlock::replay_recurrent(int64_t retained_tokens) {
    TYLOINFER_RUNTIME_CHECK(retained_tokens > 0 &&
                          speculative_xbc.defined() && speculative_dt.defined() &&
                          retained_tokens <= speculative_xbc.size(1) &&
                          retained_tokens <= speculative_dt.size(1),
                      "invalid mamba recurrent replay");
    auto& state = mixer.view();
    const int64_t B = speculative_xbc.size(0);
    const int64_t inner = mamba_heads * mamba_head_dim;
    const int64_t groups = mamba_groups;
    const int64_t state_width = ssm_state_size;
    const auto xbc = speculative_xbc.narrow(1, 0, retained_tokens).contiguous();
    const auto dt = speculative_dt.narrow(1, 0, retained_tokens).contiguous();
    auto bias = conv_bias.defined() ? conv_bias
                                    : mfq_tensor_backend::empty(
                                          {0}, mfq_tensor_backend::TensorOptions()
                                                   .device(mfq_tensor_backend::kCUDA)
                                                   .dtype(mfq_tensor_backend::kFloat32));
    auto conv_input = mfq_tensor_backend::cat({state.convolution, xbc}, 1);
    state.convolution.copy_(next_convolution_state(conv_input, conv_kernel));
    auto convolved = ssm_conv_silu_cuda(conv_input, conv_weight, bias, retained_tokens);
    auto x = convolved.narrow(2, 0, inner)
                 .reshape({B, retained_tokens, mamba_heads, mamba_head_dim})
                 .contiguous();
    auto b = convolved.narrow(2, inner, groups * state_width)
                 .reshape({B, retained_tokens, groups, state_width})
                 .contiguous();
    auto c = convolved.narrow(2, inner + groups * state_width, groups * state_width)
                 .reshape({B, retained_tokens, groups, state_width})
                 .contiguous();
    auto scanned = selective_state_scan_prefill_cuda(x, dt, dt_bias, a_log, b, c, d,
                                                     state.recurrent, time_step_limit_lo,
                                                     time_step_limit_hi);
    state.recurrent.copy_(scanned[1]);
}

void Mamba2MixerBlock::rollback_speculative(int64_t keep_position) {
    try {
        const int64_t retained = mixer.rewind(keep_position);
        // Only the convolution and scan states are replayed; the normalized
        // output and the projection are not recomputed.
        replay_recurrent(retained);
        clear_speculative();
    } catch (...) {
        mixer.abort();
        clear_speculative();
        throw;
    }
}

Tensor MoeMixerBlock::forward(CudaExecutionContext& execution, Tensor x, Tensor, int64_t,
                              const MfqOptional<Tensor>&, const RopeCache&,
                              const MfqOptional<Tensor>&, const MfqOptional<Tensor>&) {
    const int64_t B = x.size(0), T = x.size(1), H = x.size(2);
    return mfq::models::attention_layer(
        std::move(x),
        [&](Tensor x) {
            return execution.profiler.measure("moe.norm", [&] {
                return qwen_rms_norm(x.reshape({B * T, H}).to(mfq_tensor_backend::kFloat32),
                                     ffn_norm, rms_norm_eps, 0.0)
                    .reshape({B, T, H});
            });
        },
        [&](Tensor xn) {
            return execution.profiler.measure("moe.ffn", [&] {
                return execution.ffn().run(ffn, xn.reshape({B * T, H})).reshape({B, T, H});
            });
        },
        [&](Tensor residual, Tensor mixed) {
            auto rr = residual.reshape({-1, H});
            auto mm = mixed.reshape({-1, H});
            if (mm.scalar_type() != rr.scalar_type())
                mm = mm.to(rr.scalar_type()).contiguous();
            return acc_cuda(rr, mm).reshape({B, T, H});
        });
}

Tensor AttentionMixerBlock::forward_ffn(CudaExecutionContext& execution, Tensor residual,
                                       Tensor attention_output, int64_t batch, int64_t tokens,
                                       int64_t hidden) {
    auto rr = residual.reshape({-1, hidden});
    auto aa = attention_output.reshape({-1, hidden});
    if (execution.config.tasks.dense.fp32_residual) {
        rr = rr.to(mfq_tensor_backend::kFloat32);
        aa = aa.to(mfq_tensor_backend::kFloat32);
    } else if (aa.scalar_type() != rr.scalar_type()) {
        aa = aa.to(rr.scalar_type()).contiguous();
    }
    return execution.profiler.measure("attention.residual", [&] {
        return acc_cuda(rr, aa).reshape({batch, tokens, hidden});
    });
}

struct BlockLoader : TransformerWeightLoader {
    using Ffn = mfq::cuda::Ffn;
    auto full_block() const {
        auto block = std::make_unique<AttentionMixerBlock>();
        block->kv_quantization = cuda_kv_quantization();
        return block;
    }
    auto mamba_block() const { return std::make_unique<Mamba2MixerBlock>(); }
    auto moe_block() const { return std::make_unique<MoeMixerBlock>(); }
    template <class T> static std::unique_ptr<::Block> finish(std::unique_ptr<T> block) {
        return block;
    }
};

std::unique_ptr<::Block> load_block(Pipelines& pipelines,
                                    const weight_loader::Loader& weights, const Config& config, int layer,
                                    const std::string& type, std::string_view tensor_root) {
    BlockLoader loader{{weights}};
    switch (Config::mixer_kind(type)) {
        case mfq::models::nemotron_h::MixerKind::linear_attention: {
            auto block = loader.mamba_block();
            mfq::models::nemotron_h::load_linear_attention(*block, loader, config, layer,
                                                           tensor_root);
            block->conv_weight = block->conv_weight.to(mfq_tensor_backend::kFloat32).contiguous();
            block->a_log = block->a_log.to(mfq_tensor_backend::kFloat32).contiguous();
            block->d = block->d.to(mfq_tensor_backend::kFloat32).contiguous();
            block->dt_bias = block->dt_bias.to(mfq_tensor_backend::kFloat32).contiguous();
            block->scan_norm = block->scan_norm.to(mfq_tensor_backend::kFloat32).contiguous();
            if (block->conv_bias.defined())
                block->conv_bias = block->conv_bias.to(mfq_tensor_backend::kFloat32).contiguous();
            return BlockLoader::finish(std::move(block));
        }
        case mfq::models::nemotron_h::MixerKind::full_attention: {
            auto block = loader.full_block();
            mfq::models::nemotron_h::load_attention(*block, loader, config, layer, tensor_root);
            return BlockLoader::finish(std::move(block));
        }
        case mfq::models::nemotron_h::MixerKind::moe: {
            auto block = loader.moe_block();
            mfq::models::nemotron_h::load_moe(*block, loader, config, layer, tensor_root);
            // An expert cache turns the routed bank into the two-stage MFE
            // pipeline; without one the same FFN keeps the direct path.
            pipelines.prepare(block->ffn);
            return BlockLoader::finish(std::move(block));
        }
        case mfq::models::nemotron_h::MixerKind::mlp: {
            auto block = loader.moe_block();
            mfq::models::nemotron_h::load_dense_mlp(*block, loader, config, layer, tensor_root);
            return BlockLoader::finish(std::move(block));
        }
    }
    throw std::runtime_error("unsupported Nemotron-H mixer type: " + type);
}

bool supports_text_session_state(const std::vector<std::unique_ptr<::Block>>& blocks) {
    return !blocks.empty() &&
           std::all_of(blocks.begin(), blocks.end(), [](const std::unique_ptr<::Block>& block) {
               return dynamic_cast<const Mamba2MixerBlock*>(block.get()) != nullptr ||
                      dynamic_cast<const MoeMixerBlock*>(block.get()) != nullptr ||
                      dynamic_cast<const FullBlock*>(block.get()) != nullptr;
           });
}

TextSessionState capture_text_session_state(const std::vector<std::unique_ptr<::Block>>& blocks,
                                            const std::vector<std::int64_t>& tokens,
                                            std::int64_t cache_position) {
    if (!supports_text_session_state(blocks) || cache_position <= 0 ||
        static_cast<std::size_t>(cache_position) != tokens.size()) {
        throw std::runtime_error(
            "Nemotron-H session token count does not match the cache");
    }
    TextSessionState state;
    state.tokens = tokens;
    state.cache_pos = cache_position;
    state.payload = std::vector<HybridBlockSessionState>{};
    auto& layers = std::get<std::vector<HybridBlockSessionState>>(state.payload);
    layers.reserve(blocks.size());
    for (const auto& block : blocks) {
        MfqCudaGuard guard(block->cuda_device);
        HybridBlockSessionState saved;
        if (const auto* full = dynamic_cast<const FullBlock*>(block.get())) {
            saved.kind = HybridBlockSessionStateKind::FullAttention;
            saved.full_attention =
                capture_kv(full->cache, cache_position, state.bytes);
        } else if (const auto* mamba = dynamic_cast<const Mamba2MixerBlock*>(block.get())) {
            if (mamba->mixer.pending() || !mamba->mixer.defined()) {
                throw std::runtime_error("Nemotron-H mamba session state is unavailable");
            }
            saved.kind = HybridBlockSessionStateKind::Recurrent;
            saved.mixer = mamba->mixer.snapshot();
            state.bytes += session_tensor_bytes(saved.mixer.convolution);
            state.bytes += session_tensor_bytes(saved.mixer.recurrent);
        } else if (dynamic_cast<const MoeMixerBlock*>(block.get()) != nullptr) {
            saved.kind = HybridBlockSessionStateKind::Stateless;
        } else {
            throw std::runtime_error("Nemotron-H session layer type changed");
        }
        layers.push_back(std::move(saved));
    }
    return state;
}

void restore_text_session_state(std::vector<std::unique_ptr<::Block>>& blocks,
                                const TextSessionState& state) {
    const auto* layers = std::get_if<std::vector<HybridBlockSessionState>>(&state.payload);
    if (state.kind() != TextSessionStateKind::HybridAttention || state.cache_pos <= 0 ||
        state.tokens.size() != static_cast<std::size_t>(state.cache_pos) || layers == nullptr ||
        layers->size() != blocks.size()) {
        throw CudaSessionStateError("Nemotron-H session state is incompatible");
    }
    for (std::size_t index = 0; index < blocks.size(); ++index) {
        auto& block = blocks[index];
        const auto& saved = (*layers)[index];
        MfqCudaGuard guard(block->cuda_device);
        if (saved.kind == HybridBlockSessionStateKind::FullAttention) {
            auto* full = dynamic_cast<FullBlock*>(block.get());
            if (full == nullptr)
                throw CudaSessionStateError("Nemotron-H full-attention layer changed");
            validate_kv(saved.full_attention);
            continue;
        }
        if (saved.kind == HybridBlockSessionStateKind::Stateless) {
            if (dynamic_cast<MoeMixerBlock*>(block.get()) == nullptr)
                throw CudaSessionStateError("Nemotron-H stateless layer changed");
            continue;
        }
        auto* mamba = dynamic_cast<Mamba2MixerBlock*>(block.get());
        const auto convolution_width =
            mamba != nullptr ? mamba->mamba_heads * mamba->mamba_head_dim +
                                   2 * mamba->mamba_groups * mamba->ssm_state_size
                             : 0;
        if (saved.kind != HybridBlockSessionStateKind::Recurrent || mamba == nullptr ||
            !saved.mixer.convolution.defined() || !saved.mixer.recurrent.defined() ||
            saved.mixer.convolution.scalar_type() != mfq_tensor_backend::kFloat32 ||
            saved.mixer.recurrent.scalar_type() != mfq_tensor_backend::kFloat32 ||
            saved.mixer.convolution.dim() != 3 || saved.mixer.convolution.size(0) != 1 ||
            saved.mixer.convolution.size(1) != mamba->conv_kernel - 1 ||
            saved.mixer.convolution.size(2) != convolution_width ||
            saved.mixer.recurrent.dim() != 4 || saved.mixer.recurrent.size(0) != 1 ||
            saved.mixer.recurrent.size(1) != mamba->mamba_heads ||
            saved.mixer.recurrent.size(2) != mamba->mamba_head_dim ||
            saved.mixer.recurrent.size(3) != mamba->ssm_state_size ||
            !saved.mixer.convolution.is_cuda() || !saved.mixer.recurrent.is_cuda() ||
            saved.mixer.convolution.get_device() != block->cuda_device ||
            saved.mixer.recurrent.get_device() != block->cuda_device) {
            throw CudaSessionStateError("Nemotron-H mamba session topology changed");
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
        } else if (auto* mamba = dynamic_cast<Mamba2MixerBlock*>(block.get())) {
            mamba->mixer.restore(saved.mixer);
            mamba->clear_speculative();
        }
    }
}

} // namespace mfq::cuda::nemotron_h

namespace mfq::cuda {

void NemotronHModel::adapter_validate_components(const mfq::ModelGraph& graph) const {
    // A declared vision component must resolve to the RADIO adapter this
    // backend implements; anything else stays reported as unsupported instead
    // of loading a partial graph.
    if (graph.component("vision") != nullptr &&
        cuda_model_plan(graph).vision != CudaVisionAdapter::radio)
        throw std::runtime_error(
            "Nemotron-H CUDA vision requires the radio/radio_cpe.v1 vision component");
    if (graph.component("predictor") != nullptr &&
        cuda_model_plan(graph).predictor != CudaPredictorAdapter::nemotron_h)
        throw std::runtime_error(
            "Nemotron-H CUDA multi-token prediction requires the nemotron_h_mtp predictor "
            "implementation under the predictor tensor root");
}

std::unique_ptr<Block> NemotronHModel::adapter_load_block(const mfq::cuda::weight_loader::Loader& weights, int layer,
                                                          int, const std::string& type) {
    return nemotron_h::load_block(execution->pipelines, weights, config, layer, type);
}

mfq_tensor_backend::Tensor
NemotronHModel::adapter_raw_hidden(const mfq_tensor_backend::Tensor&,
                                   const mfq_tensor_backend::Tensor& finalized) const {
    // llama.cpp (src/models/nemotron-h-moe.cpp), vLLM and SGLang all hand the
    // MTP head the backbone hidden state after output_norm, unlike DeepSeek-V3.
    // The predictor therefore receives the same normalized hidden that the
    // lm_head sees, and its own hnorm renormalizes it.
    return finalized;
}

TextSessionStateKind CudaSessionCodec<NemotronHModel>::kind(const Model& model) {
    const auto full = FullAttentionSessionCodec<NemotronHModel>::kind(model);
    if (full != TextSessionStateKind::Unsupported)
        return full;
    return nemotron_h::supports_text_session_state(model.blocks)
               ? TextSessionStateKind::HybridAttention
               : TextSessionStateKind::Unsupported;
}

bool CudaSessionCodec<NemotronHModel>::supports_paged(const Model& model) {
    return FullAttentionSessionCodec<NemotronHModel>::supports_paged(model);
}

TextSessionState CudaSessionCodec<NemotronHModel>::capture(const Model& model,
                                                           const std::vector<int64_t>& tokens) {
    if (kind(model) == TextSessionStateKind::HybridAttention) {
        if (model.cache_pos <= 0 || static_cast<size_t>(model.cache_pos) != tokens.size()) {
            throw std::runtime_error("text session token count does not match the model cache");
        }
        return nemotron_h::capture_text_session_state(model.blocks, tokens, model.cache_pos);
    }
    return FullAttentionSessionCodec<NemotronHModel>::capture(model, tokens);
}

void CudaSessionCodec<NemotronHModel>::restore(Model& model, const TextSessionState& state) {
    if (kind(model) == TextSessionStateKind::HybridAttention &&
        state.kind() == TextSessionStateKind::HybridAttention) {
        nemotron_h::restore_text_session_state(model.blocks, state);
        model.cache_pos = state.cache_pos;
        return;
    }
    FullAttentionSessionCodec<NemotronHModel>::restore(model, state);
}

template struct FullAttentionSessionCodec<NemotronHModel>;

} // namespace mfq::cuda

namespace mfq::models {
template struct nemotron_h::CausalLm<cuda::CudaCausalOps<cuda::NemotronHModel>>;
} // namespace mfq::models
