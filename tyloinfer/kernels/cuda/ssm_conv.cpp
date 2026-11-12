#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "ssm_conv_raw.h"
#include <cstdint>

torch::Tensor ssm_conv_silu_cuda(torch::Tensor conv_input, torch::Tensor weight, torch::Tensor bias, int64_t n_tokens)
{
    TORCH_CHECK(conv_input.is_cuda() && conv_input.is_contiguous() && conv_input.scalar_type() == at::kFloat,
                "ssm_conv_silu: conv_input must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat,
                "ssm_conv_silu: weight must be cuda contiguous f32");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == at::kFloat,
                "ssm_conv_silu: bias must be cuda contiguous f32");
    TORCH_CHECK(conv_input.dim() == 3, "ssm_conv_silu: conv_input must be [B,K-1+T,C]");
    int B = (int)conv_input.size(0);
    int C = (int)conv_input.size(2);
    int has_bias = (int)(bias.numel() > 0);
    TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu: bias must be [C]");
    int K = 0;
    int layout = 0; // 0: channel-major [C,1,K]/[C,K], 1: [K,C]
    if (weight.dim() == 3) {
        TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1, "ssm_conv_silu: [C,1,K] weight mismatch");
        K = (int)weight.size(2);
        layout = 0;
    } else {
        TORCH_CHECK(weight.dim() == 2, "ssm_conv_silu: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C) {
            K = (int)weight.size(1);
            layout = 0;
        } else {
            TORCH_CHECK(weight.size(1) == C, "ssm_conv_silu: 2D weight must be [C,K] or [K,C]");
            K = (int)weight.size(0);
            layout = 1;
        }
    }
    int T = (int)n_tokens;
    TORCH_CHECK(T > 0 && conv_input.size(1) == T + K - 1, "ssm_conv_silu: conv_input length must be K-1+T");
    auto out = torch::empty({B, T, C}, conv_input.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.input=conv_input.data_ptr<float>(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>(); a.output=out.data_ptr<float>();
    a.batch=B; a.tokens=T; a.channels=C; a.kernel=K; a.layout=layout; a.has_bias=has_bias;
    tyloi::cuda::ssm_conv_raw::prefill(a, at::cuda::getCurrentCUDAStream().stream());
    return out;
}

torch::Tensor ssm_conv_silu_decode_cuda(torch::Tensor state, torch::Tensor x, torch::Tensor weight, torch::Tensor bias)
{
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat,
                "ssm_conv_silu_decode: state must be cuda contiguous f32");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat,
                "ssm_conv_silu_decode: x must be cuda contiguous f32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat,
                "ssm_conv_silu_decode: weight must be cuda contiguous f32");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == at::kFloat,
                "ssm_conv_silu_decode: bias must be cuda contiguous f32");
    TORCH_CHECK(state.dim() == 3, "ssm_conv_silu_decode: state must be [B,K-1,C]");
    TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "ssm_conv_silu_decode: x must be [B,1,C]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int C = (int)state.size(2);
    TORCH_CHECK(x.size(0) == B && x.size(2) == C, "ssm_conv_silu_decode: x shape mismatch");
    int has_bias = (int)(bias.numel() > 0);
    TORCH_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TORCH_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "ssm_conv_silu_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TORCH_CHECK(weight.dim() == 2,
                    "ssm_conv_silu_decode: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C && weight.size(1) == K) {
            layout = 0;
        } else {
            TORCH_CHECK(weight.size(0) == K && weight.size(1) == C,
                        "ssm_conv_silu_decode: 2D weight must be [C,K] or [K,C]");
            layout = 1;
        }
    }
    auto out = torch::empty({B, 1, C}, x.options());
    tyloi::cuda::ssm_conv_raw::Args a;
    a.state=state.data_ptr<float>(); a.input=x.data_ptr<float>(); a.weight=weight.data_ptr<float>(); a.bias=bias.data_ptr<float>(); a.output=out.data_ptr<float>();
    a.batch=B; a.channels=C; a.kernel=K; a.layout=layout; a.has_bias=has_bias;
    tyloi::cuda::ssm_conv_raw::decode(a, at::cuda::getCurrentCUDAStream().stream());
    return out;
}
