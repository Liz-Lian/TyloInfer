#include "engine/models/gemma4/model.h"
#include "engine/models/common/check.h"
#include "engine/models/common/load.h"

#include <bit>
#include <cmath>
#include <utility>

namespace mfq::cuda::gemma4 {

mfq::models::Info Model::info() const {
    return {config.vocab_size, config.max_position_embeddings, config.model_type,
            std::size_t(config.hidden_size) * sizeof(float) * 8, true, true};
}

Model read(std::shared_ptr<const mfq::ModelSource> source, std::string_view definition,
           std::int64_t context) {
    validate_model_source(*source);
    Model model;
    model.source = std::move(source);
    model.graph = mfq::models::graph(*model.source);
    model.plan = cuda_model_plan(model.graph);
    mfq::models::require_model(model.graph.backbone == "gemma4",
                               "model backbone does not match the requested causal model");
    model.config = Config::from_json(definition);
    auto bits = std::bit_cast<std::uint32_t>(
        static_cast<float>(std::sqrt(static_cast<double>(model.config.hidden_size))));
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    model.embed_scale = std::bit_cast<float>(bits & 0xffff0000U);
    mfq::models::require_model(model.config.num_hidden_layers == model.graph.topology.text_layers,
                               "model graph/config text-layer topology mismatch");
    if (context > 0) {
        mfq::models::require_model(context <= model.config.max_position_embeddings,
                                   "--ctx-size exceeds max_position_embeddings");
        model.config.max_position_embeddings = context;
    }
    return model;
}

} // namespace mfq::cuda::gemma4
