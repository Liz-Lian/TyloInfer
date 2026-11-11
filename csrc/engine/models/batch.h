#pragma once

#include "engine/tensor.h"
#include "engine/requests/result.h"
#include "engine/metrics.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace tyloi::models {

/** Per-device allocation requirements for batch state. */
using BatchGrowth = std::vector<std::pair<Device, std::size_t>>;

/** Configure batch layout and numerical execution policy. */
struct BatchConfig {
    std::size_t slots = 0;
    bool paged = false, preallocate = false, cpu_offload = false;
    int cached_slots = 0;
    bool graph = false, greedy = false;
    std::int64_t minimum = 0;
};

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
    virtual BatchGrowth initial_growth_bytes() const = 0;
    /** Initialize request backing after runtime admission. */
    virtual void initialize() = 0;
    /** Return backing required for the next token span. */
    virtual BatchGrowth prefill_growth_bytes(std::int64_t tokens) const = 0;
};

/** Describe one active request, including rows paused by the scheduler. */
struct BatchRow {
    BatchState* state;
    std::int64_t token, position, remaining;
    bool eligible, greedy;
};

/** Own batch computation and graph state; request scheduling stays in the Engine. */
class Batch {
public:
    /** Keep a model-specific graph layout alive between admission and execution. */
    struct Plan {
        virtual ~Plan() = default;
        /** New device backing admitted before capture. */
        BatchGrowth growth;
        /** Whether the captured layout must be rebuilt. */
        bool rebuild = false;
    };
    virtual ~Batch() = default;
    /** Plan batch-wide backing before initialization. */
    virtual BatchGrowth initial_growth_bytes() const { return {}; }
    /** Initialize batch-wide backing after admission. */
    virtual void initialize() {}
    /** Create uninitialized numerical state for one request. */
    virtual std::unique_ptr<BatchState> make_state(std::int64_t prompt_tokens) = 0;
    /** Report whether independent prompt spans can be computed together. */
    virtual bool supports_packed() const { return false; }
    /** Plan backing needed to admit a prefetched request into decode. */
    virtual BatchGrowth activate_growth(const BatchState&) const { return {}; }
    /** Admit a prefetched request to the active batch. */
    virtual void activate(BatchState&) {}
    /** Release a request before its numerical state is destroyed. */
    virtual void release(BatchState&) {}
    /** Resume the aggregate state after prompt processing. */
    virtual void resume(std::int64_t) {}
    /** Update aggregate state after requests are retired. */
    virtual void retire(std::int64_t) {}
    /** Reset aggregate state after runtime has drained failed work. */
    virtual void recover() {}
    /** Plan backing needed before preparing the active decode rows. */
    virtual BatchGrowth decode_growth(std::span<const BatchRow>) const { return {}; }
    /** Prepare active rows after their backing has been admitted. */
    virtual void prepare_decode(std::span<const BatchRow>) {}
    /** Return numerical workspace bytes per prompt row for runtime admission. */
    virtual std::size_t prefill_row_bytes() const = 0;
    /** Compute independent prompt spans in row order, yielding resource boundaries. */
    virtual StepSequence<std::vector<Tensor>> prefill(std::vector<BatchState*> states,
        std::vector<Tensor> ids) = 0;
    /** Compute one ordinary decode step for each supplied row. */
    virtual Tensor decode(std::span<const BatchRow> rows, Tensor ids,
        const Tensor& host_ids) = 0;
    /** Prepare an eligible graph layout without executing the numerical step. */
    virtual tyloi::StepSequence<std::unique_ptr<Plan>> plan(std::span<const BatchRow> rows, bool greedy) = 0;
    /** Execute an admitted graph, returning no result when ordinary decode is required. */
    virtual std::optional<Tensor> capture(std::span<const BatchRow> rows,
        const Tensor& ids, const Tensor& host_ids, bool greedy, const Plan& plan) = 0;
    /** Retire graphs before releasing or changing request backing. */
    virtual void clear() = 0;
    /** Report numerical graph capture and replay metrics. */
    virtual tyloi::engine::Metrics metrics() const = 0;
};

} // namespace tyloi::models
