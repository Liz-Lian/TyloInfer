#pragma once

#include "engine/tensor.h"
#include "engine/metrics.h"
#include "engine/models/capabilities.h"
#include "engine/models/input.h"
#include "engine/models/info.h"
#include "engine/models/options.h"
#include "engine/models/state.h"
#include "engine/models/prepared.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>
#include <vector>

struct MfqMultimodalInput;

namespace mfq::models {

class CausalModel {
public:
    virtual ~CausalModel() = default;
    virtual Info info() const = 0;
    virtual MfqModelCapabilities capabilities() const = 0;
    virtual mfq::engine::Metrics metrics() const = 0;
    virtual std::unique_ptr<State> make_state() const = 0;
    virtual mfq::StepSequence<Prepared> prepare(const std::vector<std::int64_t>& tokens,
        const MfqMultimodalInput& media, State& state) const = 0;
    virtual Tensor embedding(Tensor ids, int primary = 0) const = 0;
    virtual Tensor normalize(Tensor hidden, std::int64_t batch, std::int64_t tokens) const = 0;
    virtual Tensor softcap(Tensor logits) const = 0;
    virtual Tensor logits(Tensor hidden, bool capped = true) const = 0;
    virtual Tensor forward(Input input, State& state, const ForwardOptions& options = {}) const = 0;
};

} // namespace mfq::models
