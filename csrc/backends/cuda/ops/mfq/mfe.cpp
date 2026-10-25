#include "mfe.h"
#include "mfq_cuda_moe_ops.h"

#include <stdexcept>
#include <unordered_set>

namespace mfq::cuda {

using Tensor = mfq_tensor_backend::Tensor;

MoeActivationWorkspace MfeScratch::get(Tensor x, int input_rows, int groups, int gs, bool stable) {
    // A producer may have already quantized this exact view for us.
    const MoeActivationKey exact{input_rows, groups, gs, x.get_device()};
    auto found = values.find(exact);
    if (found != values.end()) return found->second;
    stable = stable && input_rows <= MoeActivationWorkspace::stable_row_limit;
    const int capacity = MoeActivationWorkspace::capacity_rows(input_rows, stable);
    // Each negative key owns a permanent capacity bucket. Larger eager
    // prefill uses key zero and cannot replace a captured bucket address.
    const MoeActivationKey key{stable ? -capacity : 0, groups, gs, x.get_device()};
    auto it = values.find(key);
    if (it == values.end() || it->second.qx.size(0) < input_rows) {
        MoeActivationWorkspace workspace;
        workspace.qx = mfq_tensor_backend::empty(
            {capacity, groups * gs}, x.options().dtype(mfq_tensor_backend::kInt8));
        workspace.xscale = mfq_tensor_backend::empty(
            {capacity, groups}, x.options().dtype(mfq_tensor_backend::kFloat32));
        it = values.insert_or_assign(key, std::move(workspace)).first;
    }
    return {it->second.qx.narrow(0, 0, input_rows),
            it->second.xscale.narrow(0, 0, input_rows)};
}

Tensor Mfe::forward(MfeScratch& scratch, Tensor x, const MoeRoutePlan& route,
                    bool input_prequantized, int epilogue_mode) const {
    if (epilogue_mode < 0 || epilogue_mode > 2 ||
        (epilogue_mode != 0 && out_per_expert % 2 != 0))
        throw std::runtime_error("invalid MFE NINT projection epilogue");
    if (!x.is_cuda() || !x.is_contiguous() || x.scalar_type() != mfq_tensor_backend::kFloat16 ||
        (x.dim() != 2 && x.dim() != 3) || x.size(-1) != neuron_len) {
        throw std::runtime_error("MFE input must be contiguous CUDA f16 with exact K");
    }
    int tokens = (int)route.ids.size(0);
    int routes = (int)route.ids.size(1);
    if (route.n_experts != n_experts || x.size(0) != tokens ||
        (x.dim() == 3 && x.size(1) != routes)) {
        throw std::runtime_error("MFE input and route shape mismatch");
    }
    int input_rows = x.dim() == 3 ? tokens * routes : tokens;
    const int result_width = epilogue_mode == 0
        ? out_per_expert
        : out_per_expert / 2;
    auto output = partial_experts
        ? mfq_tensor_backend::zeros(
            {tokens, routes, result_width},
            x.options().dtype(mfq_tensor_backend::kFloat16))
        : mfq_tensor_backend::empty(
            {tokens, routes, result_width},
            x.options().dtype(mfq_tensor_backend::kFloat16));
    std::unordered_set<MoeActivationKey, MoeActivationKeyHash> quantized;
    int nint_pool_phase = 0;
    const bool stable = input_rows <= 2048;
    for (const auto & pool : pools) {
        int groups = (int)pool.weight.ng;
        int gs = (int)pool.weight.gs;
        MoeActivationKey key{input_rows, groups, gs, x.get_device()};
        auto workspace = scratch.get(x, input_rows, groups, gs, stable);
        bool input_quantized = input_prequantized ||
            quantized.find(key) != quantized.end();
        const int route_tile_m = select_nint_prefill_route_tile(
            route, tokens, routes, n_experts);
        mfe_nint_matmul_ws_cuda(
            pool.weight.q_packed, pool.weight.row_q_bits,
            pool.weight.row_q_bit_offsets, pool.weight.sub_scale,
            pool.weight.sub_min, pool.weight.neuron_scale,
            pool.weight.neuron_min, x, route.ids, pool.expert_local,
            n_experts, pool.local_experts, out_per_expert, gs,
            epilogue_mode, route.map_ready, input_quantized,
            output, workspace.qx,
            workspace.xscale, route.ids_dst, route.expert_bounds,
            nint_route_tile_bounds(route, route_tile_m),
            nint_route_tile_experts(route, route_tile_m),
            route_tile_m, nint_pool_phase++);
        if (input_quantized || route_tile_m == 8) {
            quantized.insert(key);
        }
    }
    return output;
}

Tensor Mfe::clamped(MfeScratch& scratch, Tensor gate_up, const MoeRoutePlan& route, double limit) const {
    if (pools.empty() || !gate_up.is_cuda() ||
            !gate_up.is_contiguous() ||
            gate_up.scalar_type() != mfq_tensor_backend::kFloat16 ||
            gate_up.dim() != 3 ||
            gate_up.size(2) != 2 * neuron_len || limit <= 0.0) {
        throw std::runtime_error(
            "clamped SwiGLU is unavailable for this MFE tensor");
    }
    const int64_t width = gate_up.size(2) / 2;
    auto gate = mfq_tensor_backend::clamp_max(
        gate_up.slice(2, 0, width)
            .to(mfq_tensor_backend::kFloat32),
        limit);
    auto up = mfq_tensor_backend::clamp(
        gate_up.slice(2, width, 2 * width)
            .to(mfq_tensor_backend::kFloat32),
        -limit, limit);
    auto activation = (mfq_tensor_backend::silu(gate) * up)
        .to(mfq_tensor_backend::kFloat16).contiguous();
    return forward(scratch, activation, route);
}

} // namespace mfq::cuda
