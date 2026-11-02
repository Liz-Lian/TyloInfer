#pragma once

#include <utility>

namespace mfq::models::gemma4 {

template <class Tensor> struct FfnInputs {
    Tensor residual, dense, router, experts;
};

template <class Tensor, class Fused, class PostNorm, class Add, class DenseNorm, class RouterNorm,
          class ExpertNorm>
auto prepare_ffn(Tensor residual, Tensor attention, bool moe, Fused fused, PostNorm post_norm,
                 Add add, DenseNorm dense_norm, RouterNorm router_norm, ExpertNorm expert_norm) {
    if (auto prepared = fused(residual, attention))
        return std::move(*prepared);
    auto hidden = add(std::move(residual), post_norm(std::move(attention)));
    FfnInputs<Tensor> result{hidden, dense_norm(hidden), {}, {}};
    if (moe) {
        result.router = router_norm(hidden);
        result.experts = expert_norm(hidden);
    }
    return result;
}

enum class FfnNorm { dense, experts, output };

// Fusion replaces the same normalization/merge span. The ordinary graph stays
// here so a backend cannot silently choose different residual or scale order.
template <class Tensor, class Dense, class Experts, class Normalize, class Add, class Residual,
          class Scale, class Fused>
auto feed_forward(bool moe, bool has_scale, bool fused_merge, Tensor residual, Dense dense,
                  Experts experts, Normalize normalize, Add add, Residual add_residual, Scale scale,
                  Fused fused) {
    auto output = dense();
    if (moe) {
        if (!fused_merge)
            output = normalize(std::move(output), FfnNorm::dense);
        auto routed = experts();
        if (fused_merge)
            return fused(std::move(output), std::move(routed), std::move(residual));
        output = add(std::move(output), normalize(std::move(routed), FfnNorm::experts));
    }
    output = add_residual(std::move(residual), normalize(std::move(output), FfnNorm::output));
    return has_scale ? scale(std::move(output)) : output;
}

template <class Router, class Select, class Scale, class Project, class Reduce>
auto experts(Router router, Select select, Scale scale, Project project, Reduce reduce) {
    auto selected = select(router());
    scale(selected);
    auto pairs = project(selected);
    return reduce(std::move(pairs), selected);
}

} // namespace mfq::models::gemma4
