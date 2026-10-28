#include "moe_ffn_pipeline.h"
#include "formats/format_features.h"
#include "engine/executor/ffn.h"

#include <cstdio>
#include <exception>
#include <stdexcept>

namespace mfq::cuda {

#if !TYLOINFER_ENABLE_MFQ_COMPAT
std::shared_ptr<MoeFfnPipeline> Pipelines::make(
    const std::vector<std::shared_ptr<Experts>>&,
    const std::vector<std::shared_ptr<Experts>>&,
    MoeFfnShared, MoeFfnDispatchObserver, MoeFfnSharedWeights, double) const {
    return {};
}
#endif

void Pipelines::add(std::shared_ptr<MoeFfnPipeline> pipeline) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!pipeline) throw std::invalid_argument("cannot register an empty MoE FFN pipeline");
    items_.push_back(pipeline);
    try { activate_moe_ffn_pipeline(std::move(pipeline)); }
    catch (...) { items_.pop_back(); throw; }
#else
    throw std::invalid_argument("MoE FFN pipeline support is disabled");
#endif
}

void Pipelines::drain() {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::exception_ptr failure;
    for (const auto& pipeline : items_) {
        try { drain_moe_ffn_pipeline(pipeline); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
#endif
}

void Pipelines::prepare(Ffn& f) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(!f.moe_pipeline.expired() || !mfq::cuda::uses_expert_cache(f) || mfq::cuda::expert_parallel(f) ||
            f.cpu_moe_gate || f.cpu_moe_up || f.cpu_moe_gate_up || f.cpu_moe_down ||
            routes_.enabled())return;
    std::vector<std::shared_ptr<mfq::cuda::Experts>> gate_up;
    // An ungated bank publishes Up in both slots; the pipeline detects the
    // single-image layout from the projection geometry, not the model name.
    if(f.moe_ungated_experts)
        gate_up={std::make_shared<mfq::cuda::Experts>(f.moe_up),std::make_shared<mfq::cuda::Experts>(f.moe_up)};
    else if(f.moe_split_gate_up)gate_up={std::make_shared<mfq::cuda::Experts>(f.moe_gate),std::make_shared<mfq::cuda::Experts>(f.moe_up)};
    else gate_up={std::make_shared<mfq::cuda::Experts>(f.moe_gate_up)};
    auto pipeline=make(
        gate_up,{std::make_shared<mfq::cuda::Experts>(f.moe_down)},
        {},{},{},f.swiglu_limit);
    if(!pipeline)return;
    add(pipeline);
    f.moe_pipeline=pipeline;
    std::fprintf(stderr,"moe_generic_pipeline_dispatch layer=%d clipped_swiglu=%g\n",
        f.moe_layer,f.swiglu_limit);
#else
    return;
#endif
}

} // namespace mfq::cuda
