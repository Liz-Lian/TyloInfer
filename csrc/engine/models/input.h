#pragma once

#include "engine/tensor.h"
#include "engine/ops.h"

#include <cstdint>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tyloi::models {

/** Describe a full prompt and the compute and reuse boundaries for prefill. */
struct PrefillOptions {
    /** Full prompt length and preferred compute quantum, in tokens. */
    std::int64_t tokens = 0, compute = 0;
    /** Reused prefix and the exact boundary requested for a session checkpoint. */
    std::size_t reused = 0, stable = 0;
    bool transformed = false, retained = false;
    /** Routed weight access is available for this execution owner. */
    bool routed = false;
    /** Weight transfers are coordinated by the runtime cache. */
    bool coordinated = false;
    /** Permit bounded verification windows for this input span. */
    bool split = true;
};

/** Select logical prompt work and whether numerical workspace can be borrowed. */
struct PrefillPlan {
    /** Logical token span advanced before publishing prompt progress. */
    std::int64_t span = 0;
    /** Admit compute tiles using reclaimable expert workspace. */
    bool workspace = false;
};

struct Input {
    using Trace = std::vector<Tensor>;

    Tensor ids, embeddings;
    /** Optional semantic positions, per-sequence lengths and physical cache write positions. */
    Tensor positions, lengths, cache_positions;
    /** Optional observation output owned by the synchronous caller. */
    Trace* trace = nullptr;
    /** Optional output handle for hidden states before the final normalization. */
    Tensor* raw_hidden = nullptr;
    /** Commit completed input positions even when explicit positions are supplied. */
    bool advance = false;
    /** Confirmed prefix of a speculative verification window, or zero for ordinary execution. */
    std::int64_t confirmed = 0;
    /** Planned KV length used by static execution; zero selects the model's ordinary path. */
    std::int64_t kv_length = 0;
    /** Optional decode attention partition count selected for the planned KV length. */
    std::int64_t attention_parts = 0;
    /** Optional CPU uint8 participation mask for this input's media token rows. */
    Tensor image_mask;
    /** Execute an uncommitted suffix inside an active speculative transaction. */
    bool suffix = false;
    /** Return raw prefill hidden states in host memory when requested. */
    bool host_raw = false;
    /** Optional known host token IDs for model-owned decode prefetch and graph uploads. */
    Tensor host_ids;
    /** Optional MiniCPM key-validity mask [batch, visible tokens or more], with nonzero entries visible. */
    Tensor attention_mask;
};

struct ForwardOptions {
    std::int64_t kv_capacity = 0;
    ops::Trace* trace = nullptr;
    int trace_layer = -1;
    int primary = 0;
};

/** Select the projection required from a model-owned decode operation. */
struct DecodeOptions {
    enum class Projection { hidden, logits, greedy };
    Projection projection = Projection::hidden;
    /** Allow a reusable compact top-k projection for stochastic verification. */
    bool compact = false;
    /** Routed weight access is available for this execution owner. */
    bool routed = false;
};

/** Hold decode results, which may alias replay buffers until the next model step. */
struct Decoded {
    Tensor hidden, logits, compact, ids;
    /** Greedy verification tokens copied to host memory. */
    std::vector<std::int32_t> tokens;
    /** Report graph preparation so generation can exclude maintenance from its policy timing. */
    bool initialized = false;
};

/** Model-owned prefill phase boundaries and final owning outputs. */
struct Prefill {
    enum class Phase { encode, decode, computed, complete, progress };
    enum class Mode { ordinary, layered };
    /** Describe a layerwise diagnostic boundary without observing execution resources. */
    struct Progress {
        Device device;
        std::int64_t tokens = 0, chunk = 0;
        int completed = 0, total = 0;
        std::chrono::steady_clock::time_point started;
    };
    Phase phase = Phase::complete;
    /** Half-open weight layer range prioritised during this phase, or an empty range. */
    int first = 0, last = 0;
    /** Final normalized hidden rows used for the output projection. */
    Tensor hidden;
    /** Model-specific predictor context, which may be a bounded tail of the input span. */
    Tensor raw;
    /** Driver-measured computation time before workspace restoration, or a negative value if unmarked. */
    double elapsed_ms = -1;
    /** Present only for a progress event; the driver owns synchronization and observation. */
    std::optional<Progress> progress;
    /** Identify the model path that actually produced a completed result. */
    Mode mode = Mode::ordinary;
};

struct Prepared {
    std::vector<std::int64_t> token_ids;
    Tensor embeddings;
    Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    Tensor image_mask;

    /** Return whether prepared embeddings or positions change ordinary text prefill. */
    bool transformed() const noexcept {
        return (embeddings && embeddings.defined()) ||
            (positions && positions.defined()) ||
            decode_position_delta != 0;
    }
};

} // namespace tyloi::models
