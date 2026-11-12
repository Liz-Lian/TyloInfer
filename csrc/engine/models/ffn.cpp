#include "engine/models/ffn.h"
#include "engine/models/load.h"
#include "engine/models/source/model_graph.h"
#include "engine/models/common/load.h"
#include "engine/models/glm_dsa/weights.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/ops/ffn.h"
#include "engine/weights/loader.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/experts.h"

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>

namespace tyloi::models {
namespace {
using Glm = glm_dsa::Config;
using V4 = deepseek_v4::Config;
using V41 = deepseek_v41::Config;

class LayerFfn final : public Ffn {
public:
    LayerFfn(std::shared_ptr<const ModelSource> source, weights::Loader weights,
             LayerPlacementConfig& placement, const LoadOptions& options, int layer,
             std::variant<Glm, V4, V41> config)
        : source_(std::move(source)), weights_(std::move(weights)),
          offload_(options.cpu_offload_layers.count(layer) != 0), layer_(layer),
          primary_(placement.primary_device()), config_(std::move(config)) {
        if (weights_.placement.device != primary_) throw std::invalid_argument("FFN primary device mismatch");
    }

    void prepare() override {
        if (attempted_) throw std::logic_error("FFN preparation already attempted");
        attempted_ = true;
        const auto weights = weights_.for_layer(primary_, false);
        MfqCudaGuard guard(primary_);
        if (const auto* config = std::get_if<Glm>(&config_)) {
            TransformerWeightLoader loader{weights};
            glm_dsa::load_ffn(loader, *config, layer_, native_);
            hidden_ = config->hidden_size;
        } else if (const auto* config = std::get_if<V4>(&config_)) {
            cuda::deepseek_v4::load_ffn(native_, weights, *config, layer_, offload_);
            hidden_ = config->hidden_size;
        } else {
            const auto& config41 = std::get<V41>(config_);
            cuda::deepseek_v41_runtime::load_moe_at_into(ops_, weights, config41,
                "model.block." + std::to_string(layer_) + ".mlp.", layer_, config41.top_k);
            hidden_ = config41.hidden;
        }
        if (std::holds_alternative<V41>(config_)) inspect(ops_);
        else inspect(native_);
        prepared_ = true;
    }

    Info info() const override {
        if (!prepared_) throw std::logic_error("FFN is not prepared");
        return info_;
    }

    Tensor forward(const Tensor& input, const Tensor& ids) const override {
        if (!prepared_) throw std::logic_error("FFN is not prepared");
        if (!input || input.dim() != 2 || input.size(1) != hidden_ ||
            input.device() != Device(kCUDA, primary_))
            throw std::invalid_argument("FFN input must have shape [tokens,hidden] on the primary device");
        if (std::holds_alternative<V41>(config_)) return ops::ffn(ops_, input, ids);
        std::optional<DenseTensor> token_ids;
        if (ids) token_ids = ids.dense();
        return std::make_shared<DenseTensor>(native_.forward(input.dense(), token_ids));
    }

private:
    template<class Weights> void inspect(const Weights& ffn) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (!ffn.is_moe) throw std::runtime_error("selected layer does not contain MFE MoE weights");
        const auto* gate = ffn.moe_gate_up.template representation<MfeTensor>();
        const auto* down = ffn.moe_down.template representation<MfeTensor>();
        if (!gate || !down) throw std::runtime_error("selected layer does not contain fused MFE MoE weights");
        info_ = {gate->neuron_len, hidden_, gate->n_experts, down->neuron_len, ffn.moe_top_k,
            gate->compact ? gate->compact->size() : 0, down->compact ? down->compact->size() : 0,
            cuda::logical_bytes(*gate) + cuda::logical_bytes(*down)};
#else
        (void)ffn;
        throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
    }
    std::shared_ptr<const ModelSource> source_;
    weights::Loader weights_;
    bool offload_;
    int layer_, primary_;
    std::variant<Glm, V4, V41> config_;
    cuda::Ffn native_;
    ops::FfnWeights ops_;
    std::int64_t hidden_ = 0;
    Info info_{};
    bool attempted_ = false, prepared_ = false;
};
}

std::unique_ptr<Ffn> load_ffn(std::shared_ptr<const ModelSource> source,
    std::string_view definition, const weights::Loader& weights,
    LayerPlacementConfig& placement, const LoadOptions& options, int layer) {
    if (!source || &weights.source != source.get()) throw std::invalid_argument("FFN source and loader must match");
    if (layer < 0) throw std::invalid_argument("MoE benchmark layer must be nonnegative");
    const auto name = graph(*source).backbone;
    if (name == "glm_dsa") {
        auto config = Glm::from_json(definition);
        if (layer >= config.num_hidden_layers) throw std::out_of_range("MoE benchmark layer is out of range");
        return std::make_unique<LayerFfn>(std::move(source), weights, placement, options, layer, std::move(config));
    }
    if (name == "deepseek_v4") {
        auto config = V4::from_json(definition);
        if (layer >= config.num_hidden_layers) throw std::out_of_range("MoE benchmark layer is out of range");
        return std::make_unique<LayerFfn>(std::move(source), weights, placement, options, layer, std::move(config));
    }
    if (name == "deepseek_v41") {
        auto config = V41::from_json(definition);
        if (layer >= config.n_layers) throw std::out_of_range("MoE benchmark layer is out of range");
        return std::make_unique<LayerFfn>(std::move(source), weights, placement, options, layer, std::move(config));
    }
    throw std::runtime_error("selected CUDA backbone has no FFN MoE benchmark adapter");
}
}
