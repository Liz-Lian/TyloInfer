#pragma once

#include "ops/mfq/include/mfe.h"

class MoeCachedSource;
struct MixedMoeRuntime;

namespace mfq::cuda {

struct Experts {
    struct Shard {
        int device = 0;
        int64_t expert_begin = 0;
        int64_t expert_end = 0;
        std::shared_ptr<Experts> weight;
    };

    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    std::vector<MfePoolWeight> pools;
    int64_t mixed_weight_bytes = 0;
    bool partial_experts = false;
    bool unified_nint_projection = false;
    std::vector<Shard> expert_parallel_shards;
    std::shared_ptr<MixedMoeRuntime> resident;
    std::shared_ptr<MoeCachedSource> cached_source;
    std::vector<MoeActivationGeometry> activation_geometries;
    int activation_workspace_domain = 0;
    std::shared_ptr<MfeScratch> scratch = std::make_shared<MfeScratch>();

    bool expert_parallel() const { return !expert_parallel_shards.empty(); }
    bool supports_projection_glu_epilogue() const {
        if (!expert_parallel()) {
            return unified_nint_projection || !pools.empty();
        }
        return std::all_of(
            expert_parallel_shards.begin(),
            expert_parallel_shards.end(),
            [](const Shard & shard) {
                return shard.weight &&
                    shard.weight->supports_projection_glu_epilogue();
            });
    }
    bool supports_prequantized_input() const {
        return !expert_parallel() && (!pools.empty() || resident || cached_source);
    }
    bool can_reuse_activation_for(const Experts& consumer) const {
        if (!supports_prequantized_input() ||
                !consumer.supports_prequantized_input() ||
                neuron_len != consumer.neuron_len ||
                activation_workspace_domain == 0 ||
                activation_workspace_domain !=
                    consumer.activation_workspace_domain) {
            return false;
        }
        // Equal geometry cannot share quantization without a common backing.
        const bool shared = (cached_source && cached_source == consumer.cached_source) ||
                            (resident && resident == consumer.resident) ||
                            (!cached_source && !resident && !consumer.cached_source &&
                             !consumer.resident && scratch == consumer.scratch);
        if (!shared)
            return false;
        if (activation_geometries.empty() ||
                consumer.activation_geometries.empty()) {
            return false;
        }
        return std::all_of(
            consumer.activation_geometries.begin(),
            consumer.activation_geometries.end(),
            [&](const MoeActivationGeometry & required) {
                return std::find(
                    activation_geometries.begin(),
                    activation_geometries.end(),
                    required) != activation_geometries.end();
            });
    }
    bool supports_clamped_swiglu() const;
};

bool prefetch_cached_moe_projection_bundle(
    const Experts& gate, const Experts& up, const Experts& down, const MoeRoutePlan& route);
bool prefetch_cached_moe_projection_bundle(
    const Experts& gate_up, const Experts& down, const MoeRoutePlan& route);

} // namespace mfq::cuda
