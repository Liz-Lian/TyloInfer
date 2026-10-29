#pragma once

#include <memory>
#include <type_traits>

template <class Model>
struct RuntimeComponents;

namespace mfq::cuda {

class Executor;

template <class M, class S = M, class P = RuntimeComponents<M>>
struct Loaded {
    using Model = M;
    using State = S;
    using Parts = P;

    std::shared_ptr<Model> model;
    Parts components;
    std::unique_ptr<State> owned_state;
    std::unique_ptr<Executor> executor;

    Model& language() {
        if constexpr (requires { components.language(*model); })
            return components.language(*model);
        else
            return *model;
    }
    State& state() {
        if constexpr (std::is_same_v<Model, State>) return language();
        else return *owned_state;
    }
};

} // namespace mfq::cuda
