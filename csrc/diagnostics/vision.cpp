#include "diagnostics/vision.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/causal_model.h"
#include "engine/models/nemotron_h/vision.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/ops.h"
#include "engine/requests/media.h"
#include "engine/weights/loader.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
/** Keep the input, coroutine frames and produced tensors alive until drain. */
struct Work {
    tyloi::cuda::Program& program;
    MfqMultimodalInput media;
    std::vector<std::int64_t> ids, short_ids;
    tyloi::StepSequence<tyloi::models::Prepared> sequence, short_sequence;
    std::optional<tyloi::models::Prepared> prepared, short_prepared;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "vision_check_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "vision_check_cleanup error=unknown\n"; }
    }
};

/** Consume the model's preparation sequence while retaining its last output. */
void finish(tyloi::StepSequence<tyloi::models::Prepared>& sequence,
            std::optional<tyloi::models::Prepared>& result) {
    while (auto step = sequence.next()) {
        if (step.workspace || step.graph || step.layer)
            throw std::logic_error("resource requests require an execution owner");
        if (step.value) result = std::move(step.value);
    }
    if (!result) throw std::logic_error("vision preparation returned no result");
}

/** Produce the original exactly representable deterministic patch value. */
float pixel(std::int64_t index) {
    return static_cast<float>((index * 37 + 11) % 101) / 64.0f;
}

/** Export one observed tensor as the original F32 JSON representation. */
nlohmann::json json_tensor(const tyloi::Tensor& value) {
    std::vector<std::int64_t> shape;
    for (std::int64_t dim = 0; dim < value.dim(); ++dim) shape.push_back(value.size(dim));
    return {{"shape", shape}, {"data", tyloi::ops::floats(value)}};
}
}

int run_nemotron_h_vision_check(tyloi::cuda::Program& program) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    auto& model = program.model();
    auto& state = program.state();
    auto loader = program.loader();
    const auto* source = dynamic_cast<const tyloi::ModelSource*>(&loader.source);
    if (!source) throw std::runtime_error("vision check requires a model source");
    const auto graph = tyloi::models::graph(*source);
    if (graph.backbone != "nemotron_h")
        throw std::runtime_error("--check-nemotron-h-vision requires a Nemotron-H causal LM");
    const auto* component = graph.component("vision");
    if (!component || !model.capabilities().image_input)
        throw std::runtime_error("--check-nemotron-h-vision requires a model containing RADIO vision weights");
    const auto config = tyloi::models::nemotron_h::RadioVisionConfig::from_json(
        tyloi::models::config_json(*source));
    if (config.image_token_id < 1 || model.info().vocab_size <= 3)
        throw std::runtime_error("Nemotron-H vision diagnostic requires image/vocab token ids");

    const std::vector<std::pair<std::int64_t, std::int64_t>> grids{{4, 6}, {6, 2}};
    work.media.processor = MfqMultimodalProcessor::grid_vision;
    work.media.processor_name = component->input_contract;
    work.ids = {1, 2};
    std::vector<std::int32_t> image_grids;
    std::int64_t patches = 0, pixel_index = 0, expected_tokens = 0;
    for (const auto& [rows, columns] : grids) {
        const auto count = rows * columns;
        for (std::int64_t index = 0; index < count * config.patch_width(); ++index)
            work.media.pixel_values.push_back(pixel(pixel_index++));
        image_grids.insert(image_grids.end(), {1, static_cast<std::int32_t>(rows), static_cast<std::int32_t>(columns)});
        work.media.vision_types.push_back(1);
        patches += count;
        const auto tokens = count / (config.downsample_factor * config.downsample_factor);
        work.ids.insert(work.ids.end(), static_cast<std::size_t>(tokens), config.image_token_id);
        work.ids.push_back(3);
        expected_tokens += tokens;
    }
    work.media.pixel_shape = {patches, config.patch_width()};
    work.media.vision_grid = image_grids;
    work.media.vision_grid_shape = {static_cast<std::int64_t>(grids.size()), 3};
    work.media.image_grid = image_grids;
    work.media.image_grid_shape = work.media.vision_grid_shape;

    state.reset(1);
    work.sequence = model.prepare(work.ids, work.media, state);
    finish(work.sequence, work.prepared);
    if (work.prepared->embeddings.dim() != 3 ||
        work.prepared->embeddings.size(1) != static_cast<std::int64_t>(work.ids.size()))
        throw std::runtime_error("Nemotron-H vision prompt shape disagrees with the prompt length");
    if (std::count(work.ids.begin(), work.ids.end(), config.image_token_id) != expected_tokens)
        throw std::runtime_error("Nemotron-H vision diagnostic placeholder count is inconsistent");

    nlohmann::json result;
    result["architecture"] = graph.backbone;
    result["image_grids"] = nlohmann::json::array();
    for (const auto& [rows, columns] : grids) result["image_grids"].push_back({rows, columns});
    result["patch_width"] = config.patch_width();
    result["hidden_size"] = config.hidden_size;
    result["downsample_factor"] = config.downsample_factor;
    result["position_side"] = config.position_side();
    result["summary_tokens"] = config.summary_tokens();
    result["heads"] = config.num_heads;
    result["depth"] = config.depth;
    result["image_token_id"] = config.image_token_id;
    result["layer_norm_eps"] = config.layer_norm_eps;
    result["projector_norm_eps"] = config.projector_norm_eps;
    result["ids"] = work.ids;
    result["pixels"] = work.media.pixel_values;
    result["embeddings"] = json_tensor(work.prepared->embeddings);
    result["positions"] = json_tensor(work.prepared->positions);

    bool rejected = false;
    try {
        work.short_ids = work.ids;
        work.short_ids.erase(std::remove(work.short_ids.begin(), work.short_ids.end(), config.image_token_id));
        work.short_sequence = model.prepare(work.short_ids, work.media, state);
        finish(work.short_sequence, work.short_prepared);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    result["placeholder_mismatch_rejected"] = rejected;
    program.drain();
    std::cout << "nemotron_h_vision_check " << result.dump() << '\n';
    return 0;
}
