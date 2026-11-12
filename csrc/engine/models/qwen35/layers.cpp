#include "engine/models/qwen35/layers.h"

#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/qwen35/state.h"
#include "engine/ops.h"
#include "engine/weights/loader.h"
#include "engine/weights/rows.h"

#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::qwen35 {
namespace {

class LayerRequest final : public tyloi::models::Layers::State {
public:
    LayerRequest(const tyloi::models::Layers& owner, int index, std::size_t generation,
                 bool full)
        : State(owner, index), generation(generation) {
        if (!full) state.emplace<LinearState>();
    }

    LayerState state;
    std::size_t generation;
    bool used = false;
};

class QwenLayers final : public tyloi::models::Layers {
public:
    QwenLayers(const tyloi::weights::Loader& weights,
               LayerPlacementConfig& placement, bool mirror_scalars)
        : model_(std::make_unique<Model>()), weights_(weights), placement_(placement),
          mirror_scalars_(mirror_scalars) {}

    void initialize(std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
                    const tyloi::models::LoadOptions& options) {
        load_deferred(*model_, std::move(source), definition, weights_, placement_, options);
        layers_.resize(model_->config.num_hidden_layers);
    }

    /** Describe the source model's fixed dimensions. */
    Info info() const noexcept override {
        return {static_cast<int>(model_->config.num_hidden_layers), model_->config.hidden_size,
                model_->config.vocab_size};
    }

    /** Load every layer, leaving partial results owned on failure. */
    void load_resident() override {
        require_empty();
        for (int index = 0; index < info().layers; ++index) load_one(index);
        ready_ = true;
    }

    /** Load exactly one layer after previously owned layers were released. */
    void load_layer(int index) override {
        check_index(index);
        require_empty();
        load_one(index);
        ready_ = true;
    }

    /** Release layers after the caller has drained and retired their states. */
    void release() override {
        for (auto& layer : layers_) layer.reset();
        ready_ = false;
        ++generation_;
    }

    /** Create a fresh zero-history state for the selected loaded layer. */
    std::unique_ptr<State> make_state(int index) const override {
        const auto& layer = selected(index);
        return std::make_unique<LayerRequest>(*this, index, generation_,
                                             std::holds_alternative<Full>(layer.weights));
    }

    /** Execute the selected layer once using its request-owned state. */
    Tensor forward(int index, const Tensor& hidden, State& state, const Tensor& positions,
                   const tyloi::models::ForwardOptions& options) const override {
        const auto& layer = selected(index);
        auto* request = dynamic_cast<LayerRequest*>(&state);
        if (!request || !state.belongs_to(*this, index) || request->generation != generation_ || request->used)
            throw std::invalid_argument("Qwen layer state does not belong to this fresh request");
        if (options.primary != model_->primary_device)
            throw std::invalid_argument("Qwen layer primary device differs from its owner");
        if (!hidden.defined() || hidden.scalar_type() != kFloat16 || hidden.dim() != 3 || hidden.size(0) != 1 ||
            hidden.size(1) < 1 || hidden.size(2) != model_->config.hidden_size ||
            !positions.defined() || positions.scalar_type() != kInt64 ||
            positions.dim() != 1 || positions.size(0) != hidden.size(1))
            throw std::invalid_argument("Qwen layer input or position geometry is invalid");
        const auto* rope = &model_->rope;
        if (layer.cpu) rope = &model_->cpu_rope;
        else if (!model_->device_ropes.empty()) {
            const auto found = model_->device_ropes.find(layer.device);
            if (found == model_->device_ropes.end())
                throw std::logic_error("Qwen layer has no rope table for its device");
            rope = &found->second;
        }
        const AttentionStep step{positions, {}, {}, 0, 0, 0};
        request->used = true;
        return qwen35::forward_layer(layer, request->state, hidden, step, 0, *rope, options);
    }

    /** Read token rows individually and return the original F16 embedding. */
    Tensor embedding(const Tensor& ids) const override {
        if (!ids.defined() || ids.dim() != 2 || ids.size(0) < 1 || ids.size(1) < 1 ||
            (ids.scalar_type() != kInt64 && ids.scalar_type() != kInt32))
            throw std::invalid_argument("Qwen reference token IDs require [batch,tokens] integers");
        const auto indices = ::tyloi::ops::integers(ids);
        auto embedded = ::tyloi::ops::empty(
            {static_cast<std::int64_t>(indices.size()), model_->config.hidden_size},
            TensorOptions().device(kCPU).dtype(kFloat16));
        for (std::size_t index = 0; index < indices.size(); ++index) {
            const auto id = indices[index];
            if (id < 0 || id >= model_->config.vocab_size)
                throw std::out_of_range("Qwen reference token is outside the vocabulary");
            auto row = ::tyloi::ops::cast(tyloi::weights::read_rows(weights_.source,
                "model.token_embedding.weight", id, 1), kFloat16);
            const auto position = static_cast<std::int64_t>(index);
            ::tyloi::ops::copy_into(::tyloi::ops::slice(embedded, 0, position, position + 1), row);
        }
        embedded = ::tyloi::ops::reshape(embedded, {ids.size(0), ids.size(1), model_->config.hidden_size});
        return ::tyloi::ops::contiguous(::tyloi::ops::copy(embedded, Device(kCUDA, model_->primary_device)));
    }

    /** Apply the model's shared output normalization. */
    Tensor normalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override {
        return qwen35::normalize(*model_, hidden, batch, tokens);
    }

    /** Project a bounded vocabulary range with the original F16 source rounding. */
    Tensor head(const Tensor& hidden, std::int64_t first, std::int64_t count) const override {
        if (!hidden.defined() || hidden.dim() != 2 || hidden.size(1) != model_->config.hidden_size ||
            hidden.scalar_type() != kFloat16 || hidden.device() != Device(kCUDA, model_->primary_device) ||
            first < 0 || count < 1 ||
            count > model_->config.vocab_size || first > model_->config.vocab_size - count)
            throw std::invalid_argument("Qwen reference head input or row range is invalid");
        const char* name = !model_->config.tie_word_embeddings && weights_.has("model.output.weight")
            ? "model.output.weight" : "model.token_embedding.weight";
        auto weight = tyloi::weights::read_rows(weights_.source, name, first, count);
        weight = ::tyloi::ops::cast(weight, kFloat16, Device(kCUDA, model_->primary_device));
        return qwen35::softcap(*model_, ::tyloi::ops::matmul(hidden, ::tyloi::ops::transpose(weight, 0, 1)));
    }

private:
    void check_index(int index) const {
        if (index < 0 || index >= info().layers) throw std::out_of_range("Qwen layer index is invalid");
    }

    void require_empty() const {
        for (const auto& layer : layers_)
            if (layer) throw std::logic_error("Qwen layers must be released before loading");
    }

    const Layer& selected(int index) const {
        check_index(index);
        if (!ready_ || !layers_[index]) throw std::logic_error("Qwen layer is not loaded");
        return *layers_[index];
    }

    void load_one(int index) {
        const int device = placement_.device_for_layer(index);
        const bool cpu = index < model_->cpu_layers;
        auto placed = weights_.for_layer(device, cpu);
        auto& layer = layers_[index].emplace();
        qwen35::load_layer(layer, placed, model_->config, index,
            model_->config.layer_types.at(static_cast<std::size_t>(index)), mirror_scalars_);
    }

    std::unique_ptr<Model> model_;
    tyloi::weights::Loader weights_;
    LayerPlacementConfig& placement_;
    bool mirror_scalars_;
    std::vector<std::optional<Layer>> layers_;
    std::size_t generation_ = 0;
    bool ready_ = false;
};

} // namespace

void load_layers(std::unique_ptr<tyloi::models::Layers>& owner,
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options) {
    owner = std::make_unique<QwenLayers>(weights, placement, options.mirror_scalars);
    static_cast<QwenLayers*>(owner.get())->initialize(std::move(source), definition, options);
}

} // namespace tyloi::cuda::qwen35
