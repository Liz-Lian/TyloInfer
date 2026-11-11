#pragma once

#include "engine/tensor.h"
#include "engine/metrics.h"
#include "engine/models/metadata.h"
#include "engine/models/input.h"
#include "engine/models/state.h"
#include "engine/models/batch.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>
#include <optional>
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
    /** Borrow assembled predictor weights; each request owns its mutable predictor State. */
    virtual Mtp* mtp() const noexcept { return nullptr; }
    /** Create request state that must be retired before this model is replaced or destroyed. */
    virtual std::unique_ptr<State> make_state() const = 0;
    /** Initialize supported startup graphs using disposable request state in the owner's scope. */
    virtual void warmup(State&) const {}
    /** Create supported batch computation, or return null for single-request execution. */
    virtual std::unique_ptr<Batch> batch(bool) { return {}; }
    /** Prefetch rows for host token IDs using the supplied owning request state. */
    virtual bool prefetch(State&, const Tensor&) const { return false; }
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
    /** Run a model step in the owner's execution scope; reject unsupported speculative suffixes. */
    virtual Tensor forward(Input input, State& state, const ForwardOptions& options = {}) const = 0;
    /** Select the logical prefill span for the full prompt and its reuse boundaries. */
    virtual std::int64_t prefill_span(const State&, const PrefillOptions& options) const {
        return info().prefill == PrefillPolicy::merged && !options.transformed && !options.retained
            ? options.tokens - static_cast<std::int64_t>(options.reused) : options.compute;
    }
    /** Run supported model-owned decode, or return no result so generation uses ordinary execution. */
    virtual std::optional<Decoded> decode(Input, State&, const ForwardOptions&, const DecodeOptions& = {}) const {
        return {};
    }
    /** Run model-owned prefill phases; an empty sequence selects the ordinary forward path. */
    virtual tyloi::StepSequence<Prefill> prefill(Input, State&, const ForwardOptions&, std::int64_t) const { return {}; }
};

} // namespace tyloi::models
