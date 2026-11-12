#pragma once

#include "engine/models/input.h"
#include "engine/tensor.h"

#include <cstdint>
#include <memory>
#include <string_view>

struct LayerPlacementConfig;

namespace tyloi {
class ModelSource;
namespace weights { struct Loader; }

namespace models {
struct LoadOptions;

/** Own model weights for resident or one-layer-at-a-time evaluation. */
class Layers {
public:
    struct Info {
        int layers;
        std::int64_t width;
        std::int64_t vocab;
    };

    /** Own one fresh layer/chunk request; its creating Layers must outlive it. */
    class State {
    public:
        virtual ~State() = default;
        State(const State&) = delete;
        State& operator=(const State&) = delete;
        /** Reject a request created for another model or layer before mutation. */
        bool belongs_to(const Layers& model, int layer) const noexcept {
            return model_ == &model && layer_ == layer;
        }
    protected:
        State(const Layers& model, int layer) : model_(&model), layer_(layer) {}
    private:
        const Layers* model_;
        int layer_;
    };

    virtual ~Layers() = default;
    /** Return dimensions without exposing model-family weights or configuration. */
    virtual Info info() const noexcept = 0;
    /** Retain all layers; on failure the caller drains before release and optional OOM fallback. */
    virtual void load_resident() = 0;
    /** Load one layer after the caller has drained and retired the previous layer's requests. */
    virtual void load_layer(int index) = 0;
    /** Release all loaded layers after the caller has drained their work and retired requests. */
    virtual void release() = 0;
    /** Create zero-history state for one loaded layer and one chunk. */
    virtual std::unique_ptr<State> make_state(int index) const = 0;
    /** Run F16 [1,tokens,width] with I64 [tokens] positions in the caller's scope; use each State once. */
    virtual Tensor forward(int index, const Tensor& hidden, State& state,
        const Tensor& positions, const ForwardOptions& options) const = 0;
    /** Read only selected source rows, round to F16 and apply the model's embedding scale. */
    virtual Tensor embedding(const Tensor& ids) const = 0;
    /** Apply the model's original final normalization without projecting logits. */
    virtual Tensor normalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const = 0;
    /** Project F16 [tokens,width] against selected vocabulary rows, then apply the original soft cap. */
    virtual Tensor head(const Tensor& hidden, std::int64_t first, std::int64_t count) const = 0;
};

/** Assemble supported layer-wise computation; the source, cache and placement outlive the result. */
std::unique_ptr<Layers> load_layers(std::shared_ptr<const ModelSource> source,
    std::string_view definition, const weights::Loader& weights,
    LayerPlacementConfig& placement, const LoadOptions& options);

} // namespace models
} // namespace tyloi
