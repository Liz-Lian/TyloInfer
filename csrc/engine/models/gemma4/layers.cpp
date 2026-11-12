#include "engine/models/gemma4/layers.h"

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/gemma4/forward.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/layers.h"
#include "engine/models/load.h"
#include "engine/weights/loader.h"
#include "engine/weights/rows.h"

#include "nlohmann/json.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace tyloi::cuda::gemma4 {
namespace {

void check_definition(std::string_view definition) {
    const auto root = nlohmann::json::parse(definition.begin(), definition.end(), nullptr, false);
    if (!root.is_object()) throw std::runtime_error("Gemma4 model config must be an object");
    const auto& text = root.contains("text_config") ? root.at("text_config") : root;
    if (!text.is_object()) throw std::runtime_error("Gemma4 text_config must be an object");
    for (const char* name : {"num_kv_shared_layers", "hidden_size_per_layer_input"}) {
        if (text.contains(name) && (!text.at(name).is_number_integer() || text.at(name) != 0))
            throw std::runtime_error(std::string("Gemma4 layer reference does not support ") + name);
    }
}

class Request final : public tyloi::models::Layers::State {
public:
    Request(const tyloi::models::Layers& owner, int index, std::uint64_t generation)
        : State(owner, index), generation(generation) {}
    tyloi::state::KvView cache;
    std::uint64_t generation;
    bool used = false;
};

class GemmaLayers final : public tyloi::models::Layers {
public:
    GemmaLayers(Model model, const tyloi::weights::Loader& weights,
                LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options)
        : model_(std::move(model)), weights_(weights), placement_(placement),
          primary_(placement.primary_device()) {
        if (weights.placement.device != primary_)
            throw std::invalid_argument("Gemma4 weight and layer primary devices differ");
        gemma4::load(model_, weights_, placement_, options.gpu_layers,
                     options.expert_parallel, false, false);
    }

    Info info() const noexcept override {
        return {static_cast<int>(model_.config.num_hidden_layers),
                model_.config.hidden_size, model_.config.vocab_size};
    }

    void load_resident() override {
        require_empty();
        mode_ = Mode::loading;
        model_.layers.reserve(static_cast<std::size_t>(info().layers));
        for (int index = 0; index < info().layers; ++index)
            model_.layers.push_back(read_layer(index));
        mode_ = Mode::resident;
    }

    void load_layer(int index) override {
        require_empty();
        check_index(index);
        mode_ = Mode::loading;
        streamed_.emplace(read_layer(index));
        streamed_index_ = index;
        mode_ = Mode::streamed;
    }

    void release() override {
        model_.layers.clear();
        streamed_.reset();
        streamed_index_ = -1;
        mode_ = Mode::empty;
        ++generation_;
    }

    std::unique_ptr<State> make_state(int index) const override {
        selected(index);
        return std::make_unique<Request>(*this, index, generation_);
    }

    Tensor forward(int index, const Tensor& hidden, State& state,
                   const Tensor& positions, const tyloi::models::ForwardOptions& options) const override {
        const auto& layer = selected(index);
        auto* request = dynamic_cast<Request*>(&state);
        if (!request || !request->belongs_to(*this, index) ||
            request->generation != generation_ || request->used)
            throw std::invalid_argument("Gemma4 layer state is foreign, stale or already used");
        if (options.primary != primary_)
            throw std::invalid_argument("Gemma4 layer primary device mismatch");
        if (!hidden || hidden.dim() != 3 || hidden.size(0) != 1 ||
            hidden.size(1) <= 0 || hidden.size(1) > model_.config.max_position_embeddings ||
            hidden.size(2) != info().width || hidden.scalar_type() != kFloat16)
            throw std::invalid_argument("Gemma4 layer expects F16 [1,tokens,width] within context");
        if (!positions || positions.scalar_type() != kInt64 || positions.dim() != 1 ||
            positions.size(0) != hidden.size(1))
            throw std::invalid_argument("Gemma4 positions must be int64 [tokens]");
        for (const auto position : ::tyloi::ops::integers(positions))
            if (position < 0 || position >= model_.config.max_position_embeddings)
                throw std::out_of_range("Gemma4 position outside context");
        request->used = true;
        MfqCudaGuard guard(layer.device);
        const Device device(kCUDA, layer.device);
        const AttentionStep step{::tyloi::ops::copy(positions, device), {}, {}, 0, 0, 0};
        auto result = forward_layer(layer, request->cache, ::tyloi::ops::copy(hidden, device), step, options);
        return ::tyloi::ops::copy(result, Device(kCUDA, primary_));
    }

    Tensor embedding(const Tensor& ids) const override {
        if (!ids || ids.scalar_type() != kInt64 || (ids.dim() != 1 && ids.dim() != 2))
            throw std::invalid_argument("Gemma4 embedding expects rank one or two int64 IDs");
        const auto batch = ids.dim() == 1 ? 1 : ids.size(0);
        const auto tokens = ids.dim() == 1 ? ids.size(0) : ids.size(1);
        if (batch <= 0 || tokens <= 0) throw std::invalid_argument("Gemma4 embedding IDs are empty");
        const auto values = ::tyloi::ops::integers(ids);
        auto host = ::tyloi::ops::empty({batch * tokens, info().width},
                               TensorOptions().device(kCPU).dtype(kFloat16));
        for (std::int64_t row = 0; row < batch * tokens; ++row) {
            const auto token = values.at(static_cast<std::size_t>(row));
            if (token < 0 || token >= info().vocab)
                throw std::out_of_range("Gemma4 embedding ID outside vocabulary");
            auto source = weights::read_rows(*model_.source, "model.token_embedding.weight", token, 1);
            ::tyloi::ops::copy_into(::tyloi::ops::slice(host, 0, row, row + 1), ::tyloi::ops::cast(source, kFloat16));
        }
        auto embedded = ::tyloi::ops::copy(::tyloi::ops::reshape(host, {batch, tokens, info().width}), Device(kCUDA, primary_));
        return ::tyloi::ops::scale(embedded, model_.embed_scale);
    }

    Tensor normalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override {
        return model_.normalize(hidden, batch, tokens);
    }

    Tensor head(const Tensor& hidden, std::int64_t first, std::int64_t count) const override {
        if (!hidden || hidden.dim() != 2 || hidden.size(1) != info().width ||
            hidden.scalar_type() != kFloat16 || hidden.device() != Device(kCUDA, primary_))
            throw std::invalid_argument("Gemma4 head expects F16 [tokens,width]");
        if (first < 0 || count <= 0 || first > info().vocab || count > info().vocab - first)
            throw std::out_of_range("Gemma4 head rows outside vocabulary");
        const bool separate = !model_.config.tie_word_embeddings && weights_.has("model.output.weight");
        const char* name = separate ? "model.output.weight" : "model.token_embedding.weight";
        auto rows = weights::read_rows(*model_.source, name, first, count);
        auto weight = ::tyloi::ops::cast(rows, kFloat16, Device(kCUDA, primary_));
        return model_.softcap(::tyloi::ops::matmul(hidden, ::tyloi::ops::transpose(weight, 0, 1)));
    }

private:
    enum class Mode { empty, loading, resident, streamed };

    void require_empty() const {
        if (mode_ != Mode::empty) throw std::logic_error("Gemma4 layers must be released before loading");
    }
    void check_index(int index) const {
        if (index < 0 || index >= info().layers) throw std::out_of_range("Gemma4 layer index outside model");
    }
    Layer read_layer(int index) const {
        const int device = placement_.device_for_layer(index);
        return gemma4::load_layer(weights_.for_layer(device, false), model_.config, index,
                                  model_.config.layer_types.at(static_cast<std::size_t>(index)));
    }
    const Layer& selected(int index) const {
        check_index(index);
        if (mode_ == Mode::resident) return model_.layers.at(static_cast<std::size_t>(index));
        if (mode_ == Mode::streamed && streamed_index_ == index) return *streamed_;
        throw std::logic_error("Gemma4 layer is not loaded");
    }

    Model model_;
    const tyloi::weights::Loader weights_;
    LayerPlacementConfig& placement_;
    int primary_;
    Mode mode_ = Mode::empty;
    std::optional<Layer> streamed_;
    int streamed_index_ = -1;
    std::uint64_t generation_ = 0;
};

} // namespace

std::unique_ptr<tyloi::models::Layers> load_layers(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options) {
    if (!source || &weights.source != source.get())
        throw std::invalid_argument("Gemma4 source and weight loader must match");
    if (!options.cpu_offload_layers.empty())
        throw std::invalid_argument("Gemma4 layer reference does not support CPU offload");
    check_definition(definition);
    return std::make_unique<GemmaLayers>(read(std::move(source), definition, options.context),
                                         weights, placement, options);
}

} // namespace tyloi::cuda::gemma4
