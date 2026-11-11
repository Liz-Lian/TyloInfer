#include "engine/models/qwen35/model.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/qwen35/components.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/state.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#endif
#include "engine/ops.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

#ifdef TYLOINFER_NATIVE_CUDA
Model::Model() = default;
Model::~Model() = default;
Model::Model(Model&&) noexcept = default;
Model& Model::operator=(Model&&) noexcept = default;

tyloi::models::Info Model::info() const {
    auto result = tyloi::models::model_info(metadata);
    result.decode_graph = result.decode_graph && cpu_layers == 0 &&
        std::none_of(layers.begin(), layers.end(), [](const Layer& layer) { return layer.cpu; });
    result.graph_weights = tyloi::models::GraphWeights::resident;
    result.graph = tyloi::models::GraphOwner::executor;
    return result;
}

MfqModelCapabilities Model::capabilities() const {
    const bool vision = components && components->grid_vision.has_value();
    const auto state = ::tyloi::models::component_state(graph, plan, vision, mtp() != nullptr);
    const auto caps = ::tyloi::models::runtime_capabilities(graph, plan, state, vision, false);
    return {graph.architecture, caps.text, caps.image_input, caps.video_input,
        caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp,
        "model-graph+cuda-adapters"};
}

tyloi::engine::Metrics Model::metrics() const {
    const auto state = ::tyloi::models::component_state(graph, plan,
        components && components->grid_vision.has_value(), mtp() != nullptr);
    const auto quantization = cuda_kv_quantization();
    return {{"kv_memory_supported", 0.0},
        {"kv_quantization_supported", 1.0},
        {"kv_quantization_bits", quantization.bits},
        {"vision_declared", double(state.vision_declared)},
        {"vision_supported", double(state.vision_supported)},
        {"vision_available", double(state.vision_available)},
        {"mtp_declared", double(state.mtp_declared)},
        {"mtp_supported", double(state.mtp_supported)},
        {"mtp_available", double(state.mtp_available)}};
}

tyloi::models::Mtp* Model::mtp() const noexcept {
    return components ? components->mtp.get() : nullptr;
}

std::unique_ptr<tyloi::models::State> Model::make_state() const {
    auto state = std::make_unique<State>(*this, transposed);
    if (auto* predictor = mtp()) state->predictor = predictor->make_state();
    return state;
}

tyloi::StepSequence<tyloi::models::Prepared> Model::prepare(
        const std::vector<std::int64_t>& tokens, const MfqMultimodalInput& media,
        tyloi::models::State& value) const {
    auto* state = dynamic_cast<State*>(&value);
    if (!state || !state->belongs_to(*this))
        throw std::invalid_argument("Qwen3.5 state belongs to another model");
    if (!components) throw std::invalid_argument("model has no multimodal component");
    return components->prepare(*this, state->images, primary_device, tokens, media);
}

::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int primary) const {
    if (ids.dim() != 1 && ids.dim() != 2)
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::Tensor(tyloi::ops::unsqueeze(ids, 0)) : tyloi::Tensor{};
    const auto& input = shaped ? shaped : ids;
    if (input.size(0) <= 0 || input.size(1) <= 0)
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(tyloi::ops::copy(
        tyloi::ops::cast(input, tyloi::ScalarType::int64),
        tyloi::Device{tyloi::DeviceType::cuda, primary}));
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::ScalarType::float16));
}

::tyloi::Tensor Model::normalize(const tyloi::Tensor& hidden, std::int64_t batch,
                                 std::int64_t tokens) const {
    return qwen35::normalize(*this, hidden, batch, tokens);
}

::tyloi::Tensor Model::softcap(const tyloi::Tensor& value) const {
    return qwen35::softcap(*this, value);
}

::tyloi::Tensor Model::logits(const tyloi::Tensor& hidden, bool capped) const {
    return qwen35::logits(*this, hidden, capped);
}

::tyloi::Tensor Model::forward(tyloi::models::Input input, tyloi::models::State& value,
                               const tyloi::models::ForwardOptions& options) const {
    auto* state = dynamic_cast<State*>(&value);
    if (!state || !state->belongs_to(*this))
        throw std::invalid_argument("Qwen3.5 state belongs to another model");
    if (input.suffix || input.host_raw)
        throw std::invalid_argument("Qwen3.5 forward does not support this input mode");
    return qwen35::forward(*this, *state, std::move(input), options);
}
#endif

} // namespace tyloi::cuda::qwen35
