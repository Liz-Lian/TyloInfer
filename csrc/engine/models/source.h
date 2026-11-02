#pragma once

#include "engine/weights/source/source.h"
#include "common/legacy_names.h"
#include "common/model_graph.h"

namespace mfq {

// Canonical model names, layout and graph interpretation over retained input bytes.
class ModelSource : public source::Source {
public:
    virtual std::string_view architecture() const noexcept = 0;
    virtual TensorLayout tensor_layout() const noexcept { return {}; }
    virtual std::optional<ModelGraph> model_graph() const = 0;
};

std::shared_ptr<const ModelSource> open_model_source(
    const std::filesystem::path& path);

} // namespace mfq
