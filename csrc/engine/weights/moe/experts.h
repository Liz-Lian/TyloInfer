#pragma once

#include <memory>

namespace tyloi { class MfeTensor; }
struct MoeRoutePlan;

namespace tyloi::cuda {

bool supports_projection_glu_epilogue(const MfeTensor& weight);
bool can_reuse_activation_for(const MfeTensor& producer, const MfeTensor& consumer);
bool supports_clamped_swiglu(const MfeTensor& weight);
void prefetch_experts(const MfeTensor& weight, const MoeRoutePlan& route, bool begin = false);

bool prefetch_cached_moe_projection_bundle(
    const MfeTensor& gate, const MfeTensor& up, const MfeTensor& down, const MoeRoutePlan& route);
bool prefetch_cached_moe_projection_bundle(
    const MfeTensor& gate_up, const MfeTensor& down, const MoeRoutePlan& route);

} // namespace tyloi::cuda

std::shared_ptr<const tyloi::MfeTensor> stage_cpu_mixed_moe(
    const std::shared_ptr<const tyloi::MfeTensor>& runtime,
    bool heterogeneous);
