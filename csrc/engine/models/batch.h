#pragma once

#include "engine/tensor.h"
#include "engine/requests/result.h"
#include "engine/metrics.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace tyloi::models {

struct BatchSample {
    ::tyloi::Tensor next;
    int64_t token = 0;
    MfqPrefillTiming timing;
};

/** Own one request's numerical state independently of the batch scheduler. */
class BatchState {
public:
    virtual ~BatchState() = default;
    /** Return the number of committed tokens. */
    virtual std::int64_t position() const = 0;
    /** Return resident bytes owned by this request. */
    virtual std::size_t resident_bytes() const = 0;
    /** Return backing required before initializing this request. */
    virtual std::size_t initial_growth_bytes() const = 0;
    /** Initialize request backing after runtime admission. */
    virtual void initialize() = 0;
    /** Return backing required for the next token span. */
    virtual std::size_t prefill_growth_bytes(std::int64_t tokens) const = 0;
};

/** Own batch computation and graph state; request scheduling stays in the Engine. */
class Batch {
public:
    /** Keep a model-specific graph layout alive between admission and execution. */
    struct Plan {
        virtual ~Plan() = default;
        /** New device backing admitted before capture. */
        std::size_t growth = 0;
        /** Whether the captured layout must be rebuilt. */
        bool rebuild = false;
    };
    virtual ~Batch() = default;
    /** Create uninitialized numerical state for one request. */
    virtual std::unique_ptr<BatchState> make_state() = 0;
    /** Return numerical workspace bytes per prompt row for runtime admission. */
    virtual std::size_t prefill_row_bytes() const = 0;
    /** Compute independent prompt spans in row order. */
    virtual std::vector<Tensor> prefill(std::span<BatchState* const> states,
        const std::vector<Tensor>& ids) = 0;
    /** Compute one ordinary decode step for each supplied row. */
    virtual Tensor decode(std::span<BatchState* const> states, Tensor ids,
        const Tensor& host_ids) = 0;
    /** Prepare an eligible graph layout without executing the numerical step. */
    virtual tyloi::StepSequence<std::unique_ptr<Plan>> plan(std::span<BatchState* const> states, bool greedy) = 0;
    /** Execute an admitted graph, returning no result when ordinary decode is required. */
    virtual std::optional<Tensor> capture(std::span<BatchState* const> states,
        const Tensor& ids, const Tensor& host_ids, bool greedy, const Plan& plan) = 0;
    /** Retire graphs before releasing or changing request backing. */
    virtual void clear() = 0;
    /** Report numerical graph capture and replay metrics. */
    virtual tyloi::engine::Metrics metrics() const = 0;
};

} // namespace tyloi::models
