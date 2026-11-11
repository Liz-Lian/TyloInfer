#include "engine/weights/tq/load.h"

#include "engine/weights/tq/source.h"
#include "engine/weights/cache_resources.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/tq/cache.h"
#include "engine/weights/tq/image.h"

#include <climits>
#include <cstring>
#include <stdexcept>
#include <utility>

::tyloi::encoding::TqTensor load_tq_weight(const tyloi::source::Source& source,
                         const std::string& name, int device, bool cuda) {
    const auto shape = tyloi::tq_projection_shape(source, name);
    std::vector<std::uint8_t> blob;
    if (source.find_tensor(name)) {
        const auto bytes = source.read(name);
        blob.resize(bytes.size());
        std::memcpy(blob.data(), bytes.data(), bytes.size());
    } else {
        const auto table = tyloi::tq_table(source, name);
        blob = table->slice_rows(0, table->rows());
    }
    auto result = to_device_tq(blob, shape[1], device, cuda);
    if (result.size(0) != shape[0]) {
        throw std::invalid_argument("TQ projection output shape differs: " + name);
    }
    return result;
}

tyloi::Tensor tyloi::weights::Cache::routed_tq(
        const tyloi::source::Source& source, const std::string& name,
        const ParallelConfig& parallel) {
    const auto shape = tyloi::tq_shape(source, name);
    if (shape.size() != 3)
        throw std::invalid_argument("TQ expert bank must be [experts,out,in]: " + name);
    return routed_tq(source, name, parallel, shape[0], shape[1], shape[2]);
}

tyloi::Tensor tyloi::weights::Cache::routed_tq(
        const tyloi::source::Source& source, const std::string& name,
        const ParallelConfig& parallel, std::int64_t experts, std::int64_t output,
        std::int64_t input) {
    auto table = tyloi::tq_table(source, name);
    if (experts < 1 || experts > UINT32_MAX || output < 1 || input < 1 ||
        table->rows() % experts || table->rows() / experts != std::uint64_t(output) ||
        table->width() != input) {
        throw std::invalid_argument("TQ routed projection shape differs: " + name);
    }
    auto slices = parallel.enabled()
        ? plan_slices(experts, 1, parallel)
        : std::vector<tyloi::TensorParallelSlice>{};
    if (!data_->tq) {
        const auto bytes = data_->device_bytes.value_or(moe_expert_cache_budget_bytes(data_->moe));
        const bool explicit_budget = data_->device_bytes.has_value() || bool(data_->moe);
        data_->tq = std::make_shared<TqCache>(bytes, data_->host_bytes, data_->host_limit, !explicit_budget, data_->exec);
    }
    auto cache = data_->tq;
    auto result = std::make_shared<tyloi::TqBank>();
    result->cache = cache;
    result->experts = experts;
    result->out = output;
    result->input = input;
    result->slices = std::move(slices);
    result->source_id = cache->add(std::move(table), output);
    return result;
}
