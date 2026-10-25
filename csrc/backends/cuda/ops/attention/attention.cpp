#include "ops/attention/attention.h"

#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_paged_kv.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {
namespace {

using mfq_tensor_backend::indexing::Slice;

MfqOptional<mfq_tensor_backend::Tensor> minicpmo45_attention_mask(
    const AttentionInput& request, const AttentionConfig& config,
    std::int64_t batch, std::int64_t tokens, std::int64_t visible_len,
    bool explicit_causal) {
    MfqOptional<mfq_tensor_backend::Tensor> result = mfq_nullopt;
    if (!config.official_bf16 ||
        (!explicit_causal && !request.sequence_lengths.has_value() &&
         !request.attention_mask.has_value())) {
        return result;
    }
    auto options = mfq_tensor_backend::TensorOptions()
                       .device(request.query.device())
                       .dtype(mfq_tensor_backend::kInt64);
    auto key_positions = mfq_tensor_backend::arange(visible_len, options);
    auto query_positions = request.write_positions.to(
        request.query.device(), mfq_tensor_backend::kInt64);
    if (query_positions.dim() == 1)
        query_positions = query_positions.reshape({1, tokens}).expand({batch, tokens});
    auto allowed = key_positions.reshape({1, 1, visible_len}) <=
                   query_positions.unsqueeze(-1);
    if (request.sequence_lengths.has_value()) {
        auto lengths = request.sequence_lengths.value()
                           .to(request.query.device(), mfq_tensor_backend::kInt64)
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
        allowed = mfq_tensor_backend::where(attended, allowed,
                                            mfq_tensor_backend::ones_like(allowed));
    }
    const auto mask_dtype = config.official_bf16
                                ? mfq_tensor_backend::kBFloat16
                                : request.query.scalar_type();
    auto additive = mfq_tensor_backend::zeros(
        {batch, 1, tokens, visible_len},
        mfq_tensor_backend::TensorOptions().device(request.query.device()).dtype(mask_dtype));
    const double mask_min =
        static_cast<double>(std::numeric_limits<mfq_bfloat16>::lowest());
    additive.masked_fill_(allowed.logical_not().unsqueeze(1), mask_min);
    result = additive.contiguous();
    return result;
}

} // namespace

AttentionResult dispatch_attention(const AttentionInput& request,
                                   const AttentionKvView& kv,
                                   const AttentionConfig& config,
                                   AttentionScratch& scratch,
                                   CudaProfiler& profiler) {
    namespace tb = mfq_tensor_backend;
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

    auto attention = profiler.measure("full.attention", [&]() -> tb::Tensor {
        if (kv.quantization.enabled()) {
            attention_token_major = true;
            const auto capacity = request.planned_kv_length > 0
                                      ? request.planned_kv_length
                                      : request.cache_position + tokens;
            return cuda_kv_attention(
                q, kv.physical_key, kv.physical_value, kv.quantization, {},
                request.write_positions, request.cache_position, kv.quantized_pages(), capacity,
                request.sequence_lengths.value_or(tb::Tensor{}),
                request.attention_mask.value_or(tb::Tensor{}), attn_scale);
        }

        if (!q.is_cuda()) {
            TYLOINFER_RUNTIME_CHECK(
                !config.sliding, "CPU dense offload does not support sliding attention");
            const auto cpu_attention_dtype = config.official_bf16
                                                 ? tb::kBFloat16
                                                 : tb::kFloat32;
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
            MfqOptional<tb::Tensor> mask = mfq_nullopt;
            if (config.official_bf16) {
                mask = minicpmo45_attention_mask(request, config, batch, tokens,
                                                 kh.size(2), tokens > 1);
            } else if (tokens > 1 || request.sequence_lengths.has_value()) {
                const std::int64_t visible = kh.size(2);
                auto key_positions = tb::arange(
                    visible, tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64));
                tb::Tensor allowed;
                if (tokens > 1) {
                    auto query_positions =
                        request.positions.dim() == 1
                            ? request.positions.to(tb::kCPU, tb::kInt64)
                            : request.positions.select(0, 0).to(tb::kCPU, tb::kInt64);
                    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1);
                    allowed = allowed.unsqueeze(0).expand({batch, tokens, visible});
                } else {
                    allowed = tb::ones(
                        {batch, tokens, visible},
                        tb::TensorOptions().device(tb::kCPU).dtype(tb::kBool));
                }
                if (request.sequence_lengths.has_value()) {
                    auto lengths = request.sequence_lengths.value()
                                       .to(tb::kCPU, tb::kInt64)
                                       .reshape({batch, 1, 1});
                    allowed = allowed &
                              (key_positions.reshape({1, 1, visible}) < lengths);
                }
                mask = allowed.unsqueeze(1);
            }
            return mfq_scaled_dot_product_attention(qh, kh, vh, mask, 0.0, false,
                                                    attn_scale, false);
        }

        const auto attention_dtype = config.official_bf16 ? tb::kBFloat16 : tb::kFloat16;
        auto qh = q.to(attention_dtype).contiguous();
        auto kh = k.to(attention_dtype).contiguous();
        auto vh = v.to(attention_dtype).contiguous();
        if (request.cache_position == 0 && tokens > 1) {
            if (config.official_bf16) {
                const bool bf16_flash128 =
                    !config.sliding && head_dim == 128 && heads == 4 * kv_heads &&
                    !request.sequence_lengths.has_value() &&
                    !request.attention_mask.has_value() && config.bf16_flash128;
                if (bf16_flash128) {
                    const bool specialized_casts = config.flash128_casts;
                    auto flash_q = profiler.measure("full.flash128_q_cast", [&]() {
                        return specialized_casts ? minicpm_flash128_q_cast_cuda(qh)
                                                 : qh.to(tb::kFloat32).contiguous();
                    });
                    auto flash_kv = profiler.measure("full.flash128_kv_cast", [&]() {
                        if (specialized_casts)
                            return minicpm_flash128_kv_cast_cuda(kh, vh);
                        return std::vector<tb::Tensor>{kh.to(tb::kFloat16).contiguous(),
                                                       vh.to(tb::kFloat16).contiguous()};
                    });
                    auto output = profiler.measure("full.flash128_kernel", [&]() {
                        return mfq_attention_mma128_cuda(flash_q, flash_kv[0], flash_kv[1],
                                                         attn_scale);
                    });
                    output = profiler.measure("full.flash128_output_cast", [&]() {
                        return specialized_casts ? minicpm_flash128_output_cast_cuda(output)
                                                 : output.to(tb::kBFloat16).contiguous();
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

            const bool mma_attention_enabled = config.mma_prefill;
            if (!config.sliding && tokens % 256 == 0 && head_dim == 512 &&
                heads == 8 * kv_heads && mma_attention_enabled) {
                attention_token_major = true;
                return mfq_attention_mma512_cuda(q.to(tb::kFloat32).contiguous(), kh, vh,
                                                 attn_scale);
            }
            if (config.sliding && tokens >= 32 && head_dim == 256 && heads == 2 * kv_heads &&
                mma_attention_enabled) {
                attention_token_major = true;
                return mfq_attention_mma256_swa_cuda(q.to(tb::kFloat32).contiguous(), kh, vh,
                                                     attn_scale, config.window);
            }
            if (!config.sliding && tokens >= 32 && head_dim == 256 &&
                (heads == 4 * kv_heads || heads == 8 * kv_heads) && mma_attention_enabled) {
                attention_token_major = true;
                return mfq_attention_mma256_cuda(q.to(tb::kFloat32).contiguous(), kh, vh,
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
                                         config.bf16_gqa_decode;
            const bool aten_decode_enabled =
                (config.official_bf16 && !bf16_gqa_decode) || config.decode_aten;
            const bool mma_decode_enabled = config.mma_decode;
            if (kv.paged()) {
                const bool split_enabled = config.split_k;
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
                    q.to(tb::kFloat32).contiguous(), kv.physical_key, kv.physical_value,
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
                                 q.to(tb::kFloat32).contiguous(), kv.physical_key,
                                 kv.physical_value, request.sequence_lengths.value(), attn_scale,
                                 planned_len, scratch.mask, scratch.kv_max, scratch.meta)
                           : mfq_attention_mma256_decode_cuda(
                                 q.to(tb::kFloat32).contiguous(), kv.physical_key,
                                 kv.physical_value, request.sequence_lengths.value(), attn_scale,
                                 planned_len, scratch.mask, scratch.kv_max, scratch.meta);
            }
            if (config.sliding)
                return attention_cache_swa_planned_cuda(
                    qh, kv.physical_key, kv.physical_value, request.sequence_lengths.value(),
                    attn_scale, config.window, planned_len);

            const bool split_enabled = config.split_k;
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

    return AttentionResult{std::move(attention), attention_token_major};
}

} // namespace mfq::cuda
