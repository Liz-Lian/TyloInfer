#include "selected_attention.h"
#include "selected_raw.h"
#include "env.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>

namespace mfq_selected_jit {
namespace {
namespace raw = tyloi::cuda::selected_raw;
namespace mma = tyloi::cuda::sparse_mma_raw;
cudaStream_t stream() { return at::cuda::getCurrentCUDAStream().stream(); }
bool half_or_float(const Tensor& x) { return x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat; }
raw::Layout layout(const Tensor& x) { return {x.stride(0), x.stride(1), x.stride(2), x.stride(3)}; }
void shapes(const Tensor& q, const Tensor& k, const Tensor& v) {
    values({&q, &k, &v});
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(), "selected attention requires [B,H,T,D] query/key/value");
    TORCH_CHECK(q.size(0) == k.size(0) && k.size(1) > 0 && q.size(1) > 0 &&
        q.size(1) % k.size(1) == 0 && q.size(3) == k.size(3) && q.size(3) > 0,
        "selected-attention dimensions disagree");
}

Tensor mma_attention(const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& indices, const Tensor& mask, const Tensor& positions, int64_t columns,
    const Tensor& gate, bool half_output, double scale, Hot hot) {
    const int64_t B = q.size(0), H = q.size(1), M = q.size(2), D = q.size(3);
    TORCH_CHECK(B <= INT_MAX && H <= INT_MAX && M <= INT_MAX && k.size(1) <= INT_MAX && k.size(2) <= INT_MAX,
        "attention dimensions exceed 32-bit indexing");
    const bool causal = positions.defined();
    TORCH_CHECK(!causal || (columns > 0 && columns <= INT_MAX - 32), "causal selection exceeds 32-bit indexing");
    TORCH_CHECK(B > 0 && M > 0 && H > 0 && k.size(1) > 0 && H % k.size(1) == 0 && k.size(2) > 0 &&
        q.dim() == 4 && k.dim() == 4 && v.dim() == 4 && k.size(0) == B && k.size(3) == D &&
        v.size(0) == B && v.size(1) == k.size(1) && v.size(2) == k.size(2) && v.size(3) == D &&
        q.is_cuda() && k.is_cuda() && v.is_cuda() && q.device() == k.device() && q.device() == v.device() &&
        half_or_float(q) && k.scalar_type() == at::kHalf && v.scalar_type() == at::kHalf &&
        k.is_contiguous() && v.is_contiguous(), "invalid attention MMA tensors");
    if (causal) {
        TORCH_CHECK(positions.is_cuda() && positions.device() == q.device() && positions.is_contiguous() &&
            positions.dim() == 1 && positions.numel() == M &&
            (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong) &&
            !indices.defined() && !mask.defined(), "invalid implicit causal selection");
    } else {
        TORCH_CHECK(indices.defined() && indices.is_cuda() && indices.device() == q.device() &&
            indices.is_contiguous() && indices.scalar_type() == at::kInt && indices.dim() == 3 &&
            indices.size(0) == B && indices.size(1) == M, "invalid attention MMA indices");
        if (mask.defined()) TORCH_CHECK(mask.is_cuda() && mask.device() == q.device() &&
            mask.is_contiguous() && mask.scalar_type() == at::kHalf && mask.sizes() == indices.sizes(),
            "invalid attention MMA mask");
    }
    const auto selected = causal ? ((columns + 31) / 32) * 32 : indices.size(2);
    TORCH_CHECK(selected > 0 && selected <= INT_MAX, "invalid attention selection");
    auto plan = mma::plan(int(B), int(M), int(H), int(k.size(1)), int(k.size(2)), int(selected), int(D), hot.mapping != nullptr);
    auto partial = torch::empty({B, M, H, D}, q.options().dtype(at::kFloat));
    auto metadata = torch::empty({int64_t(plan.workspace_bytes / sizeof(float))}, q.options().dtype(at::kFloat));
    const int64_t strides[4]{q.stride(0), q.stride(1), q.stride(2), q.stride(3)};
    mma::launch(plan, q.data_ptr(), k.data_ptr(), v.data_ptr(),
        indices.defined() ? indices.data_ptr<int32_t>() : nullptr, mask.defined() ? mask.data_ptr() : nullptr,
        nullptr, causal ? positions.data_ptr() : nullptr, strides, q.scalar_type() == at::kHalf,
        q.is_contiguous(), causal && positions.scalar_type() == at::kLong, int(columns), hot,
        float(scale), partial.data_ptr<float>(), metadata.data_ptr(), plan.workspace_bytes,
        gate.defined(), stream());
    if (!gate.defined()) return partial;
    const auto shared = size_t(plan.parts - 1) * sizeof(float) * 2;
    TORCH_CHECK(shared + sizeof(float) <= 48 * 1024, "MMA gate reduction shared memory exceeded");
    auto output = torch::empty(partial.sizes(), partial.options().dtype(half_output ? at::kHalf : at::kFloat));
    const char* option = tyloi::cuda::env_lookup("TYLOINFER_QSA_PARALLEL_REDUCE");
    raw::reduce_gated(partial.data_ptr<float>(), metadata.data_ptr(), gate.data_ptr(), output.data_ptr(),
        layout(gate), output.numel() / D, int(M), int(H), int(k.size(1)), int(D), plan.blocks,
        plan.parts, 1, 16, gate.scalar_type() == at::kFloat, half_output,
        raw::Sigmoid::torch_float, !option || option[0] != '0', stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor sparse_impl(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& selected,
    double scale, bool key_view, const Tensor& gate, bool half_output, bool fused_query, Hot hot) {
    shapes(q, k, v);
    TORCH_CHECK(selected.defined() && selected.is_cuda() && selected.device() == q.device() &&
        selected.dim() == 3 && selected.size(0) == q.size(0) && selected.size(1) == q.size(2) &&
        selected.size(2) > 0 && (selected.scalar_type() == at::kInt || selected.scalar_type() == at::kLong),
        "sparse indices must have integer [B,T,topk] shape");
    TORCH_CHECK(std::isfinite(scale), "attention scale must be finite");
    c10::cuda::CUDAGuard guard(q.device());
    const bool direct = fused_query && !key_view && q.size(3) == 256 &&
        (q.size(1) == 24 || selected.size(2) <= 512) && half_or_float(q);
    auto query = direct ? q : q.to(at::kFloat).contiguous();
    auto key = k.to(at::kHalf).contiguous();
    auto value = key_view ? key : v.to(at::kHalf).contiguous();
    auto indices = selected.to(at::kInt).contiguous();
    const std::vector<int64_t> output_shape{q.size(0), q.size(2), q.size(1), q.size(3)};
    if (!q.size(0) || !q.size(2)) return torch::empty(output_shape, q.options().dtype(at::kFloat));
    if (!key.size(2)) return torch::zeros(output_shape, q.options().dtype(at::kFloat));
    const bool qsa = !key_view && q.size(3) == 256;
    const bool mla = key_view && q.size(3) == 512;
    if (qsa || mla) {
        const int64_t count = indices.size(2);
        TORCH_CHECK(count <= INT_MAX - 32 && key.size(2) <= INT_MAX, "sparse attention exceeds 32-bit index geometry");
        const int64_t padded = ((count + 31) / 32) * 32;
        TORCH_CHECK(indices.size(0) <= INT64_MAX / indices.size(1), "sparse attention row count is too large");
        const int64_t rows = indices.size(0) * indices.size(1);
        TORCH_CHECK(rows <= INT64_MAX / padded, "sparse attention index workspace is too large");
        auto safe = torch::empty({indices.size(0), indices.size(1), padded}, indices.options());
        auto mask = torch::empty(safe.sizes(), q.options().dtype(at::kHalf));
        raw::prepare_indices(indices.data_ptr<int32_t>(), safe.data_ptr<int32_t>(), mask.data_ptr(),
            rows, int(count), int(padded), int(key.size(2)), stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return mma_attention(query, key, value, safe, mask, {}, 0, gate, half_output, scale, mla ? Hot{} : hot);
    }
    auto output = torch::empty(output_shape, q.options().dtype(at::kFloat));
    if (!output.numel()) return output;
    TORCH_CHECK(output.numel() / q.size(3) <= INT_MAX, "sparse attention exceeds CUDA grid size");
    raw::sparse(query.data_ptr(), key.data_ptr(), value.data_ptr(), indices.data_ptr<int32_t>(), output.data_ptr<float>(),
        q.size(0), q.size(1), key.size(1), q.size(2), q.size(3), key.size(2), indices.size(2),
        float(scale), false, nullptr, false, {}, hot, stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
}

void values(std::initializer_list<const Tensor*> tensors) {
    const auto* first = *tensors.begin();
    for (const auto* x : tensors) {
        TORCH_CHECK(x->defined() && x->is_cuda(), "selected attention requires CUDA tensors");
        TORCH_CHECK(x->device() == first->device(), "selected-attention tensors must share one CUDA device");
        TORCH_CHECK(x->scalar_type() == at::kFloat || x->scalar_type() == at::kHalf || x->scalar_type() == at::kBFloat16,
            "selected attention requires F32/F16/BF16 values");
    }
}
Tensor promoted_matmul(const Tensor& left, const Tensor& right) {
    auto dtype = left.scalar_type() == right.scalar_type() ? left.scalar_type() : at::kFloat;
    return torch::matmul(left.to(dtype), right.to(dtype));
}
Tensor dots(const Tensor& query, const Tensor& keys) {
    values({&query, &keys});
    TORCH_CHECK(query.dim() == 4 && keys.dim() == 3 && query.size(0) == keys.size(0) &&
        query.size(3) == keys.size(2) && query.size(2) > 0 && query.size(3) > 0,
        "selected-attention score dimensions disagree");
    return torch::matmul(query.to(at::kFloat).contiguous(), keys.to(at::kFloat).transpose(-1, -2).unsqueeze(1)).clamp_min(0.0);
}
Tensor sparse(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& indices,
    double scale, bool key_view, Hot hot) {
    return sparse_impl(q, k, v, indices, scale, key_view, {}, false, false, hot);
}
Tensor gated(const Tensor& attended, const Tensor& gate, bool half_output) {
    values({&attended, &gate});
    TORCH_CHECK(attended.dim() == 4 && attended.sizes() == gate.sizes(), "attention gate requires matching [B,T,H,D] inputs");
    if (!half_or_float(attended) || !half_or_float(gate))
        return (attended.to(at::kFloat) * torch::sigmoid(gate.to(at::kFloat))).to(half_output ? at::kHalf : at::kFloat);
    c10::cuda::CUDAGuard guard(attended.device());
    auto output = torch::empty(attended.sizes(), attended.options().dtype(half_output ? at::kHalf : at::kFloat));
    if (!output.numel()) return output;
    raw::Epilogue ep{gate.data_ptr(), output.data_ptr(), gate.scalar_type() == at::kFloat,
        half_output, raw::Sigmoid::torch_float, layout(gate)};
    raw::gated(attended.data_ptr(), attended.scalar_type() == at::kHalf, layout(attended), ep,
        output.numel(), output.size(2), output.size(1), output.size(3), stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
Tensor sparse_gated(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& indices,
    const Tensor& gate, double scale, bool half_output, Hot hot) {
    shapes(q, k, v); values({&q, &gate});
    TORCH_CHECK(gate.dim() == 4 && gate.size(0) == q.size(0) && gate.size(1) == q.size(2) &&
        gate.size(2) == q.size(1) && gate.size(3) == q.size(3), "sparse attention gate geometry mismatch");
    if (q.size(3) != 256 || !half_or_float(gate) || !q.numel() || !k.size(2))
        return gated(sparse(q, k, v, indices, scale, false, hot), gate, half_output);
    const char* option = tyloi::cuda::env_lookup("TYLOINFER_QSA_SPARSE_QUERY_FUSED");
    return sparse_impl(q, k, v, indices, scale, false, gate, half_output, !option || option[0] != '0', hot);
}
Tensor block_indices_to_tokens(const Tensor& blocks, const Tensor& positions, int64_t pool, int64_t budget) {
    TORCH_CHECK(blocks.defined() && positions.defined() && blocks.is_cuda() && positions.is_cuda() &&
        blocks.device() == positions.device() && blocks.dim() == 3 && positions.dim() == 1 &&
        positions.numel() == blocks.size(1) && (blocks.scalar_type() == at::kInt || blocks.scalar_type() == at::kLong) &&
        (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong) && pool > 0 && budget > 0 &&
        pool <= INT_MAX && budget <= INT_MAX - pool && blocks.size(1) <= INT_MAX && blocks.size(2) <= budget / pool,
        "block-selected token geometry mismatch");
    c10::cuda::CUDAGuard guard(blocks.device());
    const auto columns = budget + pool - 1;
    auto output = torch::empty({blocks.size(0), blocks.size(1), columns}, blocks.options().dtype(at::kInt));
    if (!output.numel()) return output;
    auto compact = blocks.contiguous(), pos = positions.contiguous();
    raw::block_indices(compact.data_ptr(), blocks.scalar_type() == at::kLong, pos.data_ptr(), positions.scalar_type() == at::kLong,
        output.data_ptr<int32_t>(), blocks.size(0) * blocks.size(1), int(blocks.size(1)), int(blocks.size(2)),
        int(pool), int(budget), int(columns), stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor causal_gated(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& gate, int64_t columns, double scale, bool half_output,
    bool fused_gate, bool fused_prepare, Hot hot) {
    shapes(q, k, v); values({&q, &gate});
    TORCH_CHECK(gate.dim() == 4 && gate.size(0) == q.size(0) && gate.size(1) == q.size(2) &&
        gate.size(2) == q.size(1) && gate.size(3) == q.size(3) && positions.defined() &&
        positions.is_cuda() && positions.device() == q.device() && positions.dim() == 1 &&
        positions.numel() == q.size(2) && (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong) &&
        columns > 0 && columns <= INT_MAX && std::isfinite(scale), "causal attention gate geometry mismatch");
    c10::cuda::CUDAGuard guard(q.device());
    auto pos = positions.contiguous();
    if (!half_or_float(q) || !half_or_float(gate) || (q.size(3) == 256 && !k.size(2))) {
        auto selected = torch::arange(columns, positions.options()).reshape({1, 1, columns}).expand({q.size(0), q.size(2), columns});
        selected = torch::where(selected <= pos.reshape({1, q.size(2), 1}), selected, torch::full_like(selected, -1));
        return gated(sparse(q, k, v, selected, scale, false, hot), gate, half_output);
    }
    if (q.size(3) == 256) {
        auto key = k.to(at::kHalf).contiguous(), value = v.to(at::kHalf).contiguous();
        if ((hot.mapping || fused_prepare) && q.numel()) {
            auto attended = mma_attention(q, key, value, {}, {}, pos, columns,
                fused_gate ? gate : Tensor{}, half_output, scale, hot);
            return fused_gate ? attended : gated(attended, gate, half_output);
        }
        TORCH_CHECK(columns <= INT_MAX - 32, "causal MMA index geometry exceeds 32 bits");
        const auto padded = ((columns + 31) / 32) * 32;
        auto query = torch::empty(q.sizes(), q.options().dtype(at::kFloat));
        auto indices = torch::empty({q.size(0), q.size(2), padded}, q.options().dtype(at::kInt));
        auto mask = torch::empty(indices.sizes(), q.options().dtype(at::kHalf));
        if (!query.numel() && !indices.numel())
            return torch::empty({q.size(0), q.size(2), q.size(1), q.size(3)}, q.options().dtype(half_output ? at::kHalf : at::kFloat));
        const raw::Layout query_layout{q.stride(0), q.stride(2), q.stride(1), q.stride(3)};
        raw::prepare_causal(q.data_ptr(), q.scalar_type() == at::kHalf, query_layout, query.data_ptr<float>(), query.numel(),
            q.size(1), q.size(2), q.size(3), pos.data_ptr(), pos.scalar_type() == at::kLong,
            indices.data_ptr<int32_t>(), mask.data_ptr(), indices.numel(), columns, padded, k.size(2), stream());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        auto attended = mma_attention(query, key, value, indices, mask, {}, 0,
            fused_gate ? gate : Tensor{}, half_output, scale, hot);
        return fused_gate ? attended : gated(attended, gate, half_output);
    }
    auto query = q.contiguous();
    auto key = k.to(at::kHalf).contiguous(), value = v.to(at::kHalf).contiguous();
    auto output = torch::empty({q.size(0), q.size(2), q.size(1), q.size(3)},
        q.options().dtype(half_output ? at::kHalf : at::kFloat));
    if (!output.numel()) return output;
    auto scratch = q.size(3) > 1024 ? torch::empty(output.sizes(), q.options().dtype(at::kFloat)) : Tensor{};
    TORCH_CHECK(output.numel() / q.size(3) <= INT_MAX, "causal attention exceeds CUDA grid size");
    raw::Epilogue ep{gate.data_ptr(), output.data_ptr(), gate.scalar_type() == at::kFloat,
        half_output, raw::Sigmoid::torch_float, layout(gate)};
    raw::sparse(query.data_ptr(), key.data_ptr(), value.data_ptr(), nullptr,
        scratch.defined() ? scratch.data_ptr<float>() : nullptr, q.size(0), q.size(1), key.size(1),
        q.size(2), q.size(3), key.size(2), columns, float(scale), q.scalar_type() == at::kHalf,
        pos.data_ptr(), pos.scalar_type() == at::kLong, ep, hot, stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

Tensor dense(const Tensor& q, const Tensor& k, const Tensor& v, int64_t offset, double scale, Hot hot) {
    shapes(q, k, v);
    const auto count = k.size(2);
    TORCH_CHECK(count > 0 && offset >= 0 && offset <= count && q.size(2) <= count - offset,
        "attention query range is outside the cache");
    TORCH_CHECK(std::isfinite(scale), "attention scale must be finite");
    c10::cuda::CUDAGuard guard(q.device());
    if (hot.mapping) {
        auto pos = torch::arange(offset, offset + q.size(2), q.options().dtype(at::kLong));
        if (q.size(3) == 256) return mma_attention(q, k, v, {}, {}, pos, offset + q.size(2), {}, false, scale, hot);
        auto selected = torch::arange(offset + q.size(2), pos.options()).reshape({1, 1, -1}).expand({1, q.size(2), offset + q.size(2)});
        selected = torch::where(selected <= pos.reshape({1, -1, 1}), selected, torch::full_like(selected, -1));
        return sparse(q, k, v, selected, scale, false, hot);
    }
    const auto options = q.options().dtype(at::kLong);
    const auto kp = torch::arange(count, options).unsqueeze(0);
    const auto qp = torch::arange(q.size(2), options).unsqueeze(1) + offset;
    auto mask = (kp <= qp).unsqueeze(0).unsqueeze(0);
    return at::scaled_dot_product_attention(q, k.to(q.scalar_type()), v.to(q.scalar_type()), mask,
        0.0, false, scale, true).permute({0, 2, 1, 3});
}
} // namespace mfq_selected_jit
