#pragma once

#include "engine/tensor.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>

struct LayerPlacementConfig;
namespace tyloi { class ModelSource; }
namespace tyloi::weights { struct Loader; }
namespace tyloi::models {
struct LoadOptions;

/** Own one model layer's routed FFN for controlled diagnostics. */
class Ffn {
public:
    struct Info {
        /** Routed projection width used by the original diagnostic log. */
        std::int64_t hidden;
        /** Backbone input width before any latent routed projection. */
        std::int64_t input_width;
        std::int64_t experts, intermediate, routes;
        /** Original compact NINT pool counts, excluding mixed or cached records. */
        std::size_t gate_pools, down_pools;
        double routed_bytes;
    };
    virtual ~Ffn() = default;
    /** Load weights after the caller has retained this owner for failure cleanup. */
    virtual void prepare() = 0;
    /** Return the geometry and original logical byte metric after preparation. */
    virtual Info info() const = 0;
    /** Execute the original FFN in the active scope with optional real token IDs. */
    virtual Tensor forward(const Tensor& input, const Tensor& ids = {}) const = 0;
};

/** Construct an unloaded routed FFN view borrowing the loader's cache owner. */
std::unique_ptr<Ffn> load_ffn(std::shared_ptr<const ModelSource> source,
    std::string_view definition, const weights::Loader& weights,
    LayerPlacementConfig& placement, const LoadOptions& options, int layer);
}
