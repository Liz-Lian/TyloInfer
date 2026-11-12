#include "engine/models/deepseek_v4/groups.h"

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/deepseek_v4/forward.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/groups.h"
#include "engine/models/load.h"
#include "engine/ops.h"
#include "engine/weights/loader.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace tyloi::cuda::deepseek_v4 {
namespace {

class Request final : public tyloi::models::Groups::State {
public:
    Request(const tyloi::models::Groups& owner, const Model& model,
            std::uint64_t generation, std::int64_t batch)
        : State(owner), state(model), generation(generation), batch(batch) {}
    deepseek_v4::State state;
    std::uint64_t generation;
    std::int64_t batch;
    bool used = false;
};

class DeepseekGroups final : public tyloi::models::Groups {
public:
    DeepseekGroups(Model model, const tyloi::weights::Loader& weights,
                   LayerPlacementConfig& placement, const tyloi::models::LoadOptions& options)
        : model_(std::move(model)), weights_(weights), placement_(placement),
          offload_(options.cpu_offload_layers), primary_(placement.primary_device()),
          gpu_layers_(options.gpu_layers) {
        if (weights_.placement.device != primary_)
            throw std::invalid_argument("DeepSeek V4 weight and layer primary devices differ");
        validate_offload(model_.config, offload_);
    }

    Info info() const noexcept override {
        return {static_cast<int>(model_.config.num_hidden_layers), model_.config.hidden_size,
                model_.config.vocab_size, model_.config.hc_mult};
    }

    void prepare() override {
        if (attempted_) throw std::logic_error("DeepSeek V4 group preparation already attempted");
        attempted_ = true;
        load_deferred(model_, weights_, placement_, gpu_layers_);
        prepared_ = true;
    }

    void load(int first, int count) override {
        require_prepared();
        if (loading_) throw std::logic_error("DeepSeek V4 group must be released before loading");
        if (first < 0 || count <= 0 || first >= info().layers || count > info().layers - first)
            throw std::out_of_range("DeepSeek V4 group range outside model");
        loading_ = true;
        auto& executor = Executor::current();
        model_.layers.reserve(static_cast<std::size_t>(count));
        for (int index = first; index < first + count; ++index) {
            const int device = placement_.device_for_layer(index);
            std::cerr << "stream loading layer " << index << ' '
                << model_.metadata.layer_types.at(static_cast<std::size_t>(index)) << std::endl;
            model_.layers.emplace_back();
            load_layer(model_.layers.back(), weights_.for_layer(device, false), model_.config, index,
                model_.metadata.layer_types.at(static_cast<std::size_t>(index)),
                executor.collectives(), primary_, offload_, host_bytes_);
        }
        ready_ = true;
    }

    void release() override {
        model_.layers.clear();
        ready_ = false;
        loading_ = false;
        ++generation_;
    }

    std::unique_ptr<State> make_state(std::int64_t batch) const override {
        if (!ready_) throw std::logic_error("DeepSeek V4 group is not loaded");
        if (batch <= 0 || batch > 16) throw std::invalid_argument("DeepSeek V4 group batch must be 1..16");
        return std::make_unique<Request>(*this, model_, generation_, batch);
    }

    Tensor embedding(const Tensor& ids) const override {
        require_prepared();
        auto embedded = model_.embedding(ids, primary_);
        const auto batch = embedded.size(0), tokens = embedded.size(1);
        auto half = ::tyloi::ops::cast(embedded, kFloat16);
        return ::tyloi::ops::contiguous(::tyloi::ops::expand(::tyloi::ops::unsqueeze(half, 2),
            {batch, tokens, info().streams, info().width}));
    }

    Tensor forward(const Tensor& ids, const Tensor& hidden, State& state,
                   const Tensor& positions, const tyloi::models::ForwardOptions& options) const override {
        auto* request = dynamic_cast<Request*>(&state);
        if (!ready_ || !request || !request->belongs_to(*this) || request->generation != generation_ || request->used)
            throw std::invalid_argument("DeepSeek V4 group state is foreign, stale or already used");
        if (options.primary != primary_) throw std::invalid_argument("DeepSeek V4 group primary mismatch");
        if (!ids || ids.scalar_type() != kInt64 || ids.dim() != 2 || ids.size(0) != request->batch ||
            ids.size(1) <= 0 || ids.size(1) > model_.config.max_position_embeddings)
            throw std::invalid_argument("DeepSeek V4 group IDs must be int64 [batch,tokens] within context");
        const auto batch = ids.size(0), tokens = ids.size(1);
        if (!hidden || hidden.scalar_type() != kFloat16 || hidden.dim() != 4 ||
            hidden.size(0) != batch || hidden.size(1) != tokens ||
            hidden.size(2) != info().streams || hidden.size(3) != info().width)
            throw std::invalid_argument("DeepSeek V4 group hidden must be F16 [batch,tokens,streams,width]");
        if (!positions || positions.scalar_type() != kInt64 ||
            !((positions.dim() == 1 && positions.size(0) == tokens) ||
              (positions.dim() == 2 && positions.size(0) == batch && positions.size(1) == tokens)))
            throw std::invalid_argument("DeepSeek V4 group positions must be int64 [tokens] or [batch,tokens]");
        request->used = true;
        request->state.reset(batch);
        Tensor result = hidden;
        for (std::size_t index = 0; index < model_.layers.size(); ++index) {
            const auto& layer = model_.layers[index];
            MfqCudaGuard guard(layer.device);
            const Device device(kCUDA, layer.device);
            const AttentionStep step{::tyloi::ops::copy(positions, device), {}, {}, 0, 0, 0};
            result = forward_layer(layer, request->state.layers[index],
                ::tyloi::ops::copy(result, device), ::tyloi::ops::copy(ids, device), step, options);
        }
        return ::tyloi::ops::copy(result, Device(kCUDA, primary_));
    }

    Tensor finalize(const Tensor& hidden, std::int64_t batch, std::int64_t tokens) const override {
        require_prepared();
        if (!hidden || hidden.scalar_type() != kFloat16 || hidden.dim() != 4 ||
            batch <= 0 || tokens <= 0 || hidden.size(0) != batch || hidden.size(1) != tokens ||
            hidden.size(2) != info().streams || hidden.size(3) != info().width)
            throw std::invalid_argument("DeepSeek V4 finalize hidden geometry differs");
        return deepseek_v4::finalize(model_, hidden, batch, tokens);
    }

    Tensor logits(const Tensor& hidden) const override { require_prepared(); return model_.logits(hidden); }

private:
    void require_prepared() const {
        if (!prepared_) throw std::logic_error("DeepSeek V4 shared weights are not prepared");
    }
    Model model_;
    const tyloi::weights::Loader weights_;
    LayerPlacementConfig& placement_;
    const std::unordered_set<int> offload_;
    const int primary_;
    const int gpu_layers_;
    std::int64_t host_bytes_ = 0;
    std::uint64_t generation_ = 0;
    bool attempted_ = false, prepared_ = false, loading_ = false, ready_ = false;
};

} // namespace

std::unique_ptr<tyloi::models::Groups> load_groups(
    std::shared_ptr<const tyloi::ModelSource> source, std::string_view definition,
    const tyloi::weights::Loader& weights, LayerPlacementConfig& placement,
    const tyloi::models::LoadOptions& options) {
    if (!source || &weights.source != source.get())
        throw std::invalid_argument("DeepSeek V4 source and weight loader must match");
    auto model = read(std::move(source), definition, options.context);
    if (model.plan.vision != ::tyloi::models::Vision::none ||
        model.plan.predictor != ::tyloi::models::Predictor::none)
        throw std::runtime_error("CUDA component adapter is unsupported for this model");
    return std::make_unique<DeepseekGroups>(std::move(model), weights, placement, options);
}

} // namespace tyloi::cuda::deepseek_v4
