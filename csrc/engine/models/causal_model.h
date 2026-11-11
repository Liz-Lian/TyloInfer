#pragma once

#include "engine/tensor.h"
#include "engine/metrics.h"
#include "engine/models/metadata.h"
#include "engine/models/input.h"
#include "engine/models/state.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>
#include <vector>

struct MfqMultimodalInput;

namespace tyloi::models {

/** Own shared model weights while requests exclusively own the State passed to each call. */
class CausalModel {
public:
    virtual ~CausalModel() = default;
    /** Return model dimensions, context limits and execution capabilities. */
    virtual Info info() const = 0;
    /** Return the capabilities provided by this assembled model. */
    virtual MfqModelCapabilities capabilities() const = 0;
    /** Return model-owned capability and resource metrics. */
    virtual tyloi::engine::Metrics metrics() const = 0;
    /** Create request state that must be retired before this model is replaced or destroyed. */
    virtual std::unique_ptr<State> make_state() const = 0;
    /** Prepare supported media for the owning request, yielding progress and owned Tensor payloads. */
    virtual tyloi::StepSequence<Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, State& state) const = 0;
    /** Embed borrowed token IDs without changing their owning handle. */
    virtual Tensor embedding(const Tensor& ids, int primary = 0) const = 0;
    /** Return normalized output hidden states with the supplied batch and token geometry. */
    virtual Tensor normalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const = 0;
    /** Return logits with this model's configured soft cap. */
    virtual Tensor softcap(const Tensor& logits) const = 0;
    /** Project borrowed hidden states to an owning logits handle. */
    virtual Tensor logits(const Tensor& hidden, bool capped = true) const = 0;
    /** Run a model step in the owner's execution scope, updating only the supplied request State. */
    virtual Tensor forward(Input input, State& state, const ForwardOptions& options = {}) const = 0;
};

} // namespace tyloi::models
