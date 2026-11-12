#pragma once

#include "engine/models/input.h"

#include <cstdint>
#include <memory>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi { class ModelSource; }
namespace tyloi::weights { struct Loader; }

namespace tyloi::models {
struct LoadOptions;

/** Own a bounded group of layers while the evaluation tool controls chunk batches. */
class Groups {
public:
    struct Info {
        int layers;
        std::int64_t width, vocab, streams;
    };
    /** Own a fresh chunk batch for the currently loaded group. */
    class State {
    public:
        virtual ~State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        /** Verify the creating model before any state mutation. */
        bool belongs_to(const Groups& owner) const noexcept { return owner_ == &owner; }
    protected:
        explicit State(const Groups& owner) : owner_(&owner) {}
    private:
        const Groups* owner_;
    };
    virtual ~Groups() = default;
    /** Return the full model dimensions and per-token hidden stream count. */
    virtual Info info() const noexcept = 0;
    /** Load shared tables after the execution owner has retained this object for failure cleanup. */
    virtual void prepare() = 0;
    /** Load a consecutive layer group after earlier work and requests have drained. */
    virtual void load(int first, int count) = 0;
    /** Release loaded layers after the caller has drained and retired their requests. */
    virtual void release() = 0;
    /** Create zero-history state for one batch of at most sixteen chunks. */
    virtual std::unique_ptr<State> make_state(std::int64_t batch) const = 0;
    /** Embed IDs as F16 [batch,tokens,streams,width] for the staged evaluation tool. */
    virtual Tensor embedding(const Tensor& ids) const = 0;
    /** Run the loaded group once with explicit token IDs and positions in the caller's scope. */
    virtual Tensor forward(const Tensor& ids, const Tensor& hidden, State& state,
        const Tensor& positions, const ForwardOptions& options) const = 0;
    /** Apply the original stream reduction and final normalization in their original order. */
    virtual Tensor finalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const = 0;
    /** Project selected normalized hidden states and apply the original final logit cap. */
    virtual Tensor logits(const Tensor& hidden) const = 0;
};

/** Assemble grouped computation without loading its layers; source, cache and placement outlive it. */
std::unique_ptr<Groups> load_groups(std::shared_ptr<const ModelSource> source,
    std::string_view definition, const weights::Loader& weights,
    LayerPlacementConfig& placement, const LoadOptions& options);

} // namespace tyloi::models
