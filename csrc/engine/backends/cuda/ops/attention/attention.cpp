#include "engine/backends/cuda/ops/attention/attention.h"

#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_paged_kv.h"
#include "engine/backends/cuda/kernels/deepseek_v4_attention.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/attention/scratch.h"
#endif

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda {
#ifdef TYLOINFER_NATIVE_CUDA
std::int64_t ops::attention_parts(std::int64_t planned_len) {
    return std::min<std::int64_t>(planned_len >= 192 ? (planned_len + 127) / 128 : 1,
                                  AttentionScratch::kMaxParts);
}
#endif
namespace {

using ::tyloi::indexing::Slice;

#ifdef TYLOINFER_NATIVE_CUDA
struct NativeAttentionInput {
    ::tyloi::DenseTensor query, key, value, positions, write_positions;
    MfqOptional<::tyloi::DenseTensor> sequence_lengths, attention_mask;
    std::int64_t cache_position, planned_kv_length, decode_attention_parts;
};
struct NativeKvView {
    ::tyloi::DenseTensor visible_key, visible_value, physical_key, physical_value;
    ::tyloi::DenseTensor key_chunk_ptrs, value_chunk_ptrs, page_table;
    tyloi::state::KvQuantization quantization;
    std::int64_t page_size, pages_per_chunk, heads;
    bool paged() const noexcept { return page_size > 0; }
};
#else
using NativeAttentionInput = AttentionInput;
#endif

MfqOptional<::tyloi::DenseTensor> minicpmo45_attention_mask(
    const NativeAttentionInput& request, const AttentionConfig& config,
    std::int64_t batch, std::int64_t tokens, std::int64_t visible_len,
    bool explicit_causal) {
    MfqOptional<::tyloi::DenseTensor> result = mfq_nullopt;
    if (!config.official_bf16 ||
        (!explicit_causal && !request.sequence_lengths.has_value() &&
         !request.attention_mask.has_value())) {
        return result;
    }
    auto options = ::tyloi::TensorOptions()
                       .device(request.query.device())
                       .dtype(::tyloi::kInt64);
    auto key_positions = ::tyloi::cuda::arange(visible_len, options);
    auto query_positions = request.write_positions.to(
        request.query.device(), ::tyloi::kInt64);
    if (query_positions.dim() == 1)
        query_positions = query_positions.reshape({1, tokens}).expand({batch, tokens});
    auto allowed = key_positions.reshape({1, 1, visible_len}) <=
                   query_positions.unsqueeze(-1);
    if (request.sequence_lengths.has_value()) {
        auto lengths = request.sequence_lengths.value()
                           .to(request.query.device(), ::tyloi::kInt64)
                           .reshape({batch, 1, 1});
        allowed = allowed &
                  (key_positions.reshape({1, 1, visible_len}) < lengths);
    }
    if (request.attention_mask.has_value()) {
        auto valid = request.attention_mask.value().to(request.query.device());
        if (valid.dim() != 2 || valid.size(0) != batch || valid.size(1) < visible_len) {
            throw std::runtime_error(
                "MiniCPM-o attention_mask must cover [batch,visible_tokens]");
        }
        valid = valid.narrow(1, 0, visible_len).ne(0);
        allowed = allowed & valid.unsqueeze(1);
        auto attended = allowed.any(-1, true);
        allowed = ::tyloi::cuda::where(attended, allowed,
                                            ::tyloi::cuda::ones_like(allowed));
    }
    const auto mask_dtype = config.official_bf16
                                ? ::tyloi::kBFloat16
                                : request.query.scalar_type();
    auto additive = ::tyloi::cuda::zeros(
        {batch, 1, tokens, visible_len},
        ::tyloi::TensorOptions().device(request.query.device()).dtype(mask_dtype));
    const double mask_min =
        static_cast<double>(std::numeric_limits<mfq_bfloat16>::lowest());
    additive.masked_fill_(allowed.logical_not().unsqueeze(1), mask_min);
    result = additive.contiguous();
    return result;
}

} // namespace

#ifdef TYLOINFER_NATIVE_CUDA
namespace {
const tyloi::DenseTensor& dense(const tyloi::Tensor& tensor) {
    const auto* result = tensor.representation<tyloi::DenseTensor>();
    if (!result) throw std::invalid_argument("attention requires dense tensors");
    return *result;
}
bool rope_kv_supported(const tyloi::ops::AttentionInput& input,
                               const tyloi::state::KvView& kv, const tyloi::ops::RopeTable& table) {
    return input.query.defined() && input.key.defined() && input.positions.defined() &&
        input.write_positions.defined() && table.cos && table.sin &&
        input.query.dim() == 4 && input.key.dim() == 4 &&
        input.query.device().is_cuda() &&
        input.write_positions.dim() == 1 && input.positions.dim() == 1 && input.query.size(2) == 1 &&
        !kv.paged() && !kv.ring && table.rotary_dim == 128 &&
        (!table.sections || !table.sections.defined() || table.sections.numel() == 0) &&
        input.query.size(1) == 32 && input.key.size(1) == 8 && input.query.size(3) == 128 &&
        kv.dtype == tyloi::ScalarType::bfloat16;
}
void check_device(const tyloi::Tensor& query, const tyloi::Tensor& tensor) {
    if (tensor.defined() && tensor.numel() != 0 && tensor.device() != query.device())
        throw std::invalid_argument("fused attention tensor devices do not match");
}
void check_rope_kv_devices(const tyloi::ops::AttentionInput& input,
                           const tyloi::state::KvView& kv, const tyloi::ops::RopeTable& table) {
    for (const auto& tensor : std::array<tyloi::Tensor, 7>{
             input.key, input.value, input.write_positions, table.cos, table.sin,
             kv.physical_key, kv.physical_value})
        if (tensor.defined()) check_device(input.query, tensor);
}
}

std::optional<tyloi::DenseTensor> ops::norm_rope_kv(const tyloi::ops::AttentionInput& input,
    const tyloi::state::KvView& kv, const tyloi::ops::RopeTable& table,
    const tyloi::DenseTensor& q_norm, const tyloi::DenseTensor& k_norm, double eps, double offset) {
    if (!rope_kv_supported(input, kv, table) || kv.quantization.enabled() ||
        !q_norm.defined() || !k_norm.defined() ||
        !Executor::current().attention_options().fused_qk_rope_kv)
        return std::nullopt;
    check_rope_kv_devices(input, kv, table);
    check_device(input.query, q_norm);
    check_device(input.query, k_norm);
    MfqCudaGuard guard(input.query.device().index);
    using T = ::tyloi::DenseTensor;
    return tyloi::ops::measure("full.qk_norm_rope_kv_write", [&]() -> tyloi::DenseTensor {
        auto q = dense(input.query);
        return minicpm_qk_norm_rope_cache_write_bf16_cuda(
            q.contiguous(), dense(input.key).contiguous(), dense(input.value).contiguous(), T(q_norm), T(k_norm),
            dense(input.positions).contiguous().to(q.device(), ::tyloi::kInt64),
            dense(input.write_positions), dense(table.cos), dense(table.sin), dense_kv_buffer(kv.physical_key), dense_kv_buffer(kv.physical_value),
            eps, offset);
    });
}

std::optional<tyloi::DenseTensor> ops::rope_kv(const tyloi::ops::AttentionInput& input,
    const tyloi::state::KvView& kv, const tyloi::ops::RopeTable& table) {
    if (!rope_kv_supported(input, kv, table) ||
        !Executor::current().attention_options().fused_rope_kv)
        return std::nullopt;
    check_rope_kv_devices(input, kv, table);
    MfqCudaGuard guard(input.query.device().index);
    return tyloi::ops::measure("full.rope_kv_write", [&]() -> tyloi::DenseTensor {
        auto q = dense(input.query);
        return minicpm_bf16_rope_cache_write_cuda(
            q.contiguous(), dense(input.key).contiguous(), dense(input.value).contiguous(),
            dense(input.positions).contiguous().to(q.device(), ::tyloi::kInt64),
            dense(input.write_positions), dense(table.cos), dense(table.sin), dense_kv_buffer(kv.physical_key), dense_kv_buffer(kv.physical_value),
            table.rotary_dim);
    });
}

tyloi::ops::AttentionResult ops::attention(const tyloi::ops::AttentionInput& input,
                                        const tyloi::state::KvView& public_kv,
                                        const tyloi::ops::AttentionConfig& public_config) {
    MfqCudaGuard guard(input.query.device().index);
    using DenseTensor = ::tyloi::DenseTensor;
    const NativeAttentionInput request{
        dense(input.query), dense(input.key), dense(input.value), dense(input.positions),
        dense(input.write_positions),
        input.sequence_lengths ? MfqOptional<DenseTensor>{dense(input.sequence_lengths)} : mfq_nullopt,
        input.attention_mask ? MfqOptional<DenseTensor>{dense(input.attention_mask)} : mfq_nullopt,
        input.cache_position, input.planned_kv_length, input.decode_attention_parts};
    const NativeKvView kv{dense_kv_buffer(public_kv.visible_key), dense_kv_buffer(public_kv.visible_value),
        dense_kv_buffer(public_kv.physical_key), dense_kv_buffer(public_kv.physical_value),
        dense_kv_buffer(public_kv.key_chunk_ptrs), dense_kv_buffer(public_kv.value_chunk_ptrs),
        dense_kv_buffer(public_kv.page_table), public_kv.quantization, public_kv.page_size,
        public_kv.pages_per_chunk, public_kv.heads};
    auto& executor = Executor::current();
    const AttentionConfig config{public_config.bf16, public_config.sliding, public_config.window,
                                 public_config.scale, executor.attention_kernels()};
    auto& profiler = executor.profiler();
    auto& scratch = executor.attention_scratch(request.query.device(),
        {request.query.size(1), request.key.size(1), request.query.size(3),
         config.sliding ? config.window : 0});
    if (request.query.is_cuda())
        scratch.split(request.query.size(0) * request.query.size(1), AttentionScratch::kMaxParts,
                      request.query.size(3), request.query.device());
    const auto pages = quantized_pages(public_kv);
#else
AttentionResult dispatch_attention(const AttentionInput& request,
                                   const tyloi::state::KvView& kv,
                                   const AttentionConfig& config,
                                   AttentionScratch& scratch,
                                   CudaProfiler& profiler) {
    const auto pages = quantized_pages(kv);
#endif
    const auto& q = request.query;
    const auto& k = request.key;
    const auto& v = request.value;
    const std::int64_t batch = q.size(0);
    const std::int64_t tokens = q.size(2);
    const std::int64_t heads = q.size(1);
    const std::int64_t head_dim = q.size(3);
    const std::int64_t kv_heads = k.size(1);
    const double attn_scale = config.scale > 0.0
                                  ? config.scale
                                  : 1.0 / std::sqrt(static_cast<double>(head_dim));
    bool attention_token_major = false;

    auto attention = profiler.measure("full.attention", [&]() -> ::tyloi::DenseTensor {
        if (kv.quantization.enabled()) {
            attention_token_major = true;
            const auto capacity = request.planned_kv_length > 0
                                      ? request.planned_kv_length
                                      : request.cache_position + tokens;
            return cuda_kv_attention(
                q, kv.physical_key, kv.physical_value, kv.quantization, {},
                request.write_positions, request.cache_position, pages, capacity,
                request.sequence_lengths.value_or(::tyloi::DenseTensor{}),
                request.attention_mask.value_or(::tyloi::DenseTensor{}), attn_scale);
        }

        if (!q.is_cuda()) {
            TYLOINFER_RUNTIME_CHECK(
                !config.sliding, "CPU dense offload does not support sliding attention");
            const auto cpu_attention_dtype = config.official_bf16
                                                 ? ::tyloi::kBFloat16
                                                 : ::tyloi::kFloat32;
            auto qh = q.to(cpu_attention_dtype).contiguous();
            auto kh = kv.visible_key.to(cpu_attention_dtype).contiguous();
            auto vh = kv.visible_value.to(cpu_attention_dtype).contiguous();
            if (kv_heads != heads) {
                TYLOINFER_RUNTIME_CHECK(kv_heads > 0 && heads % kv_heads == 0,
                                        "CPU attention head ratio is invalid");
                const std::int64_t repeat = heads / kv_heads;
                kh = kh.repeat_interleave(repeat, 1);
                vh = vh.repeat_interleave(repeat, 1);
            }
            MfqOptional<::tyloi::DenseTensor> mask = mfq_nullopt;
            if (config.official_bf16) {
                mask = minicpmo45_attention_mask(request, config, batch, tokens,
                                                 kh.size(2), tokens > 1);
            } else if (tokens > 1 || request.sequence_lengths.has_value()) {
                const std::int64_t visible = kh.size(2);
                auto key_positions = ::tyloi::cuda::arange(
                    visible, ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kInt64));
                ::tyloi::DenseTensor allowed;
                if (tokens > 1) {
                    auto query_positions =
                        request.positions.dim() == 1
                            ? request.positions.to(::tyloi::kCPU, ::tyloi::kInt64)
                            : request.positions.select(0, 0).to(::tyloi::kCPU, ::tyloi::kInt64);
                    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1);
                    allowed = allowed.unsqueeze(0).expand({batch, tokens, visible});
                } else {
                    allowed = ::tyloi::cuda::ones(
                        {batch, tokens, visible},
                        ::tyloi::TensorOptions().device(::tyloi::kCPU).dtype(::tyloi::kBool));
                }
                if (request.sequence_lengths.has_value()) {
                    auto lengths = request.sequence_lengths.value()
                                       .to(::tyloi::kCPU, ::tyloi::kInt64)
                                       .reshape({batch, 1, 1});
                    allowed = allowed &
                              (key_positions.reshape({1, 1, visible}) < lengths);
                }
                mask = allowed.unsqueeze(1);
            }
            return mfq_scaled_dot_product_attention(qh, kh, vh, mask, 0.0, false,
                                                    attn_scale, false);
        }

        const auto attention_dtype = config.official_bf16 ? ::tyloi::kBFloat16 : ::tyloi::kFloat16;
        auto qh = q.to(attention_dtype).contiguous();
        auto kh = k.to(attention_dtype).contiguous();
        auto vh = v.to(attention_dtype).contiguous();
        if (request.cache_position == 0 && tokens > 1) {
            if (config.official_bf16) {
                const bool bf16_flash128 =
                    !config.sliding && head_dim == 128 && heads == 4 * kv_heads &&
                    !request.sequence_lengths.has_value() &&
                    !request.attention_mask.has_value() && config.kernels.bf16_flash128;
                if (bf16_flash128) {
                    const bool specialized_casts = config.kernels.flash128_casts;
                    auto flash_q = profiler.measure("full.flash128_q_cast", [&]() {
                        return specialized_casts ? minicpm_flash128_q_cast_cuda(qh)
                                                 : qh.to(::tyloi::kFloat32).contiguous();
                    });
                    auto flash_kv = profiler.measure("full.flash128_kv_cast", [&]() {
                        if (specialized_casts)
                            return minicpm_flash128_kv_cast_cuda(kh, vh);
                        return std::vector<::tyloi::DenseTensor>{kh.to(::tyloi::kFloat16).contiguous(),
                                                       vh.to(::tyloi::kFloat16).contiguous()};
                    });
                    auto output = profiler.measure("full.flash128_kernel", [&]() {
                        return mfq_attention_mma128_cuda(flash_q, flash_kv[0], flash_kv[1],
                                                         attn_scale);
                    });
                    output = profiler.measure("full.flash128_output_cast", [&]() {
                        return specialized_casts ? minicpm_flash128_output_cast_cuda(output)
                                                 : output.to(::tyloi::kBFloat16).contiguous();
                    });
                    attention_token_major = true;
                    return output;
                }
                auto repeated_k = kh;
                auto repeated_v = vh;
                if (kv_heads != heads) {
                    TYLOINFER_RUNTIME_CHECK(
                        kv_heads > 0 && heads % kv_heads == 0,
                        "MiniCPM-o Qwen3 attention head ratio is invalid");
                    const std::int64_t repeat = heads / kv_heads;
                    repeated_k = kh.repeat_interleave(repeat, 1).contiguous();
                    repeated_v = vh.repeat_interleave(repeat, 1).contiguous();
                }
                auto mask = minicpmo45_attention_mask(request, config, batch, tokens, tokens,
                                                      false);
                return mfq_scaled_dot_product_attention(qh, repeated_k, repeated_v, mask, 0.0,
                                                        !mask.has_value(), attn_scale, false);
            }

            const bool mma_attention_enabled = config.kernels.mma_prefill;
            if (!config.sliding && tokens % 256 == 0 && head_dim == 512 &&
                heads == 8 * kv_heads && mma_attention_enabled) {
                attention_token_major = true;
                return mfq_attention_mma512_cuda(q.to(::tyloi::kFloat32).contiguous(), kh, vh,
                                                 attn_scale);
            }
            if (config.sliding && tokens >= 32 && head_dim == 256 && heads == 2 * kv_heads &&
                mma_attention_enabled) {
                attention_token_major = true;
                return mfq_attention_mma256_swa_cuda(q.to(::tyloi::kFloat32).contiguous(), kh, vh,
                                                     attn_scale, config.window);
            }
            if (!config.sliding && tokens >= 32 && head_dim == 256 &&
                (heads == 4 * kv_heads || heads == 8 * kv_heads) && mma_attention_enabled) {
                attention_token_major = true;
                return mfq_attention_mma256_cuda(q.to(::tyloi::kFloat32).contiguous(), kh, vh,
                                                 attn_scale);
            }
            if (config.sliding)
                return attention_swa_cuda(qh, kh, vh, attn_scale, config.window);
            return mfq_scaled_dot_product_attention(qh, kh, vh, std::nullopt, 0.0, true,
                                                    attn_scale, true);
        }

        if (request.sequence_lengths.has_value()) {
            const std::int64_t planned_len = request.planned_kv_length > 0
                                                  ? request.planned_kv_length
                                                  : request.cache_position + tokens;
            const bool bf16_gqa_decode = config.official_bf16 && tokens == 1 &&
                                         config.kernels.bf16_gqa_decode;
            const bool aten_decode_enabled =
                (config.official_bf16 && !bf16_gqa_decode) || config.kernels.decode_aten;
            const bool mma_decode_enabled = config.kernels.mma_decode;
            if (kv.paged()) {
                const bool split_enabled = config.kernels.split_k;
                std::int64_t parts = split_enabled && request.cache_position >= 192
                                         ? (request.cache_position + 127) / 128
                                         : 1;
                const bool dynamic_parts = split_enabled && request.decode_attention_parts > 1;
                if (dynamic_parts)
                    parts = request.decode_attention_parts;
                parts = std::min<std::int64_t>(parts, scratch.partial_o.size(1));
                return attention_paged_cache_decode_cuda(
                    qh, kv.key_chunk_ptrs, kv.value_chunk_ptrs, kv.page_table,
                    request.sequence_lengths.value(), attn_scale, kv.page_size,
                    kv.pages_per_chunk, kv.heads, scratch.partial_o, scratch.partial_m,
                    scratch.partial_l, parts, dynamic_parts);
            }
            if (aten_decode_enabled) {
                const std::int64_t visible_len =
                    config.sliding ? std::min<std::int64_t>(config.window,
                                                            request.cache_position + tokens)
                                   : request.cache_position + tokens;
                auto cached_k = kv.physical_key
                                    .index({Slice(), Slice(), Slice(0, visible_len), Slice()})
                                    .contiguous();
                auto cached_v = kv.physical_value
                                    .index({Slice(), Slice(), Slice(0, visible_len), Slice()})
                                    .contiguous();
                auto mask = minicpmo45_attention_mask(request, config, batch, tokens,
                                                      visible_len,
                                                      request.cache_position > 0 && tokens > 1);
                if (config.official_bf16 && kv_heads != heads) {
                    TYLOINFER_RUNTIME_CHECK(
                        kv_heads > 0 && heads % kv_heads == 0,
                        "MiniCPM-o Qwen3 attention head ratio is invalid");
                    const std::int64_t repeat = heads / kv_heads;
                    cached_k = cached_k.repeat_interleave(repeat, 1).contiguous();
                    cached_v = cached_v.repeat_interleave(repeat, 1).contiguous();
                }
                return mfq_scaled_dot_product_attention(qh, cached_k, cached_v, mask, 0.0,
                                                        false, attn_scale, !config.official_bf16);
            }
            if (config.sliding && tokens == 1 && mma_decode_enabled && head_dim == 256 &&
                heads == 2 * kv_heads) {
                const std::int64_t visible_len =
                    std::min<std::int64_t>(config.window, planned_len);
                scratch.mma(batch, kv_heads, head_dim, visible_len, 64, q.device());
                attention_token_major = true;
                return mfq_attention_mma256_swa_decode_cuda(
                    q.to(::tyloi::kFloat32).contiguous(), kv.physical_key, kv.physical_value,
                    request.sequence_lengths.value(), attn_scale, visible_len, scratch.mask,
                    scratch.kv_max, scratch.meta);
            }
            if (!config.sliding && tokens == 1 && mma_decode_enabled && heads == 8 * kv_heads &&
                (head_dim == 256 || head_dim == 512)) {
                const std::int64_t kv_tile = head_dim == 512 ? 32 : 64;
                scratch.mma(batch, kv_heads, head_dim, planned_len, kv_tile, q.device());
                attention_token_major = true;
                return head_dim == 512
                           ? mfq_attention_mma512_decode_cuda(
                                 q.to(::tyloi::kFloat32).contiguous(), kv.physical_key,
                                 kv.physical_value, request.sequence_lengths.value(), attn_scale,
                                 planned_len, scratch.mask, scratch.kv_max, scratch.meta)
                           : mfq_attention_mma256_decode_cuda(
                                 q.to(::tyloi::kFloat32).contiguous(), kv.physical_key,
                                 kv.physical_value, request.sequence_lengths.value(), attn_scale,
                                 planned_len, scratch.mask, scratch.kv_max, scratch.meta);
            }
            if (config.sliding)
                return attention_cache_swa_planned_cuda(
                    qh, kv.physical_key, kv.physical_value, request.sequence_lengths.value(),
                    attn_scale, config.window, planned_len);

            const bool split_enabled = config.kernels.split_k;
            std::int64_t parts = split_enabled && request.cache_position >= 192
                                     ? (request.cache_position + 127) / 128
                                     : 1;
            if (split_enabled && request.decode_attention_parts > 0)
                parts = request.decode_attention_parts;
            parts = std::min<std::int64_t>(parts, scratch.partial_o.size(1));
            if (request.decode_attention_parts > 1)
                return attention_cache_decode_dynamic_cuda(
                    qh, kv.physical_key, kv.physical_value,
                    request.sequence_lengths.value(), attn_scale, scratch.partial_o,
                    scratch.partial_m, scratch.partial_l, parts);
            if (parts > 1)
                return attention_cache_decode_split_cuda(
                    qh, kv.physical_key, kv.physical_value,
                    request.sequence_lengths.value(), attn_scale, scratch.partial_o,
                    scratch.partial_m, scratch.partial_l, parts);
            return attention_cache_decode_cuda(qh, kv.physical_key, kv.physical_value,
                                              request.sequence_lengths.value(), attn_scale);
        }

        if (config.official_bf16) {
            auto cached_k = kv.visible_key.contiguous();
            auto cached_v = kv.visible_value.contiguous();
            if (kv_heads != heads) {
                TYLOINFER_RUNTIME_CHECK(
                    kv_heads > 0 && heads % kv_heads == 0,
                    "MiniCPM-o Qwen3 attention head ratio is invalid");
                const std::int64_t repeat = heads / kv_heads;
                cached_k = cached_k.repeat_interleave(repeat, 1).contiguous();
                cached_v = cached_v.repeat_interleave(repeat, 1).contiguous();
            }
            auto mask = minicpmo45_attention_mask(
                request, config, batch, tokens, cached_k.size(2),
                request.cache_position > 0 && tokens > 1);
            return mfq_scaled_dot_product_attention(
                qh, cached_k, cached_v, mask, 0.0,
                !mask.has_value() && request.cache_position == 0 && tokens > 1,
                attn_scale, false);
        }
        return config.sliding
                   ? attention_swa_cuda(qh, kv.visible_key.contiguous(),
                                        kv.visible_value.contiguous(), attn_scale,
                                        config.window)
                   : attention_cuda(qh, kv.visible_key.contiguous(),
                                    kv.visible_value.contiguous(), attn_scale, true);
    });

#ifdef TYLOINFER_NATIVE_CUDA
    return AttentionResult{std::make_shared<::tyloi::DenseTensor>(std::move(attention)), attention_token_major};
#else
    return AttentionResult{std::move(attention), attention_token_major};
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::DenseTensor ops::sparse_attention(
    ::tyloi::DenseTensor query, ::tyloi::DenseTensor cache,
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor mask,
    ::tyloi::DenseTensor sinks, double scale) {
    const auto device = query.device();
    MfqCudaGuard guard(device.index);
    auto& executor = Executor::current();
    auto& scratch = executor.attention_scratch(
        device, {query.size(1), 1, query.size(3), 0});
    scratch.ensure_meta(8 * 1024 * 1024, device);
    return attention_dsv4_sparse_cuda(
        std::move(query), std::move(cache), std::move(indices), std::move(mask),
        std::move(sinks), scratch.meta, scale);
}
#endif

} // namespace tyloi::cuda
