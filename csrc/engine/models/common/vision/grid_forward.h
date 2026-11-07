#pragma once

#include "engine/generation/sequence.h"
#include "engine/ops.h"
#include "grid_vision.h"
#include "../transformer_layer.h"
#include <cmath>
#include <utility>

namespace tyloi::models::grid_vision {

template <class Value, class Project, class Split, class Rope, class Attend, class Output>
auto attention(Value hidden, Project project, Split split, Rope rope, Attend attend,
               Output output) {
    auto qkv = split(project(std::move(hidden)));
    auto query = rope(std::move(qkv[0]));
    auto key = rope(std::move(qkv[1]));
    return output(attend(std::move(query), std::move(key), std::move(qkv[2])));
}

template <class Value, class Layers, class Patch, class Position, class Add, class Layer>
tyloi::StepSequence<Value> encode(Value pixels, const GridVisionConfig &config, std::vector<GridShape> grids,
            Layers &layers, Patch patch, Position position, Add add, Layer layer) {
    const auto layout =
        make_grid_vision_layout(grids, static_cast<int32_t>(config.spatial_merge_size));
    const auto side =
        static_cast<int32_t>(std::sqrt(static_cast<double>(config.num_position_embeddings)));
    const auto interpolation = make_learned_position_interpolation(
        grids, static_cast<int32_t>(config.spatial_merge_size), side);
    auto hidden = patch(std::move(pixels), layout.patch_count);
    hidden = add(std::move(hidden), position(interpolation));
    co_yield tyloi::StepState::advanced;
    for (const auto& block : layers) {
        hidden = layer(block, std::move(hidden), layout);
        co_yield tyloi::StepState::advanced;
    }
    co_yield std::move(hidden);
}

template <class Value, class Norm, class Pack, class Up, class Gelu, class Down>
auto merge(Value patches, int64_t patch_count, const GridVisionConfig &config, Norm norm,
           Pack pack, Up up, Gelu gelu, Down down) {
    const auto unit = config.spatial_merge_size * config.spatial_merge_size;
    require_model(unit > 0 && patch_count > 0 && patch_count % unit == 0,
                  "grid-Vision patch count is not divisible by merge unit");
    auto grouped = pack(norm(std::move(patches)), patch_count / unit, unit * config.hidden_size);
    return down(gelu(up(std::move(grouped))));
}

template <class Value> struct Prepared {
    Value embeddings, positions;
    int decode_delta = 0;
};

template <class Value, class Embed, class Scatter, class Positions>
auto prepare(const std::vector<int64_t> &ids, Value merged, int64_t merged_count,
             int64_t image_token, int64_t video_token, const GridVisionConfig &config,
             const std::vector<GridShape> &grids, Embed embed, Scatter scatter,
             Positions positions) {
    auto embeddings = embed(ids);
    std::vector<int64_t> placeholders;
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == image_token)
            placeholders.push_back(static_cast<int64_t>(i));
        else if (ids[i] == video_token)
            throw std::invalid_argument(
                "grid-Vision image preparation does not accept video placeholders");
    }
    if (static_cast<int64_t>(placeholders.size()) != merged_count)
        throw std::invalid_argument("image placeholders disagree with merged vision patches");
    scatter(embeddings, merged, placeholders);
    auto semantic = build_grid_mrope_positions(
        ids, image_token, video_token, static_cast<int32_t>(config.spatial_merge_size), grids, {});
    return Prepared<Value>{std::move(embeddings), positions(semantic), semantic.decode_delta};
}

} // namespace tyloi::models::grid_vision
