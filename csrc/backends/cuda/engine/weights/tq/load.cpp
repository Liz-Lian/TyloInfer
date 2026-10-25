#include "backends/cuda/engine/weights/tq/load.h"

#include "model_source/tq_source.h"
#include "execution/execution.h"
#include "backends/cuda/engine/weights/tq/cache.h"
#include "backends/cuda/engine/weights/tq/image.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"

#include <climits>
#include <cstring>
#include <stdexcept>
#include <utility>

TqWeight load_tq_weight(const mfq::ModelSource& source,
                         const std::string& name, int device, bool cuda) {
    const auto shape = mfq::tq_projection_shape(source, name);
    const auto bytes = source.read(name);
    std::vector<std::uint8_t> blob(bytes.size());
    std::memcpy(blob.data(), bytes.data(), bytes.size());
    auto result = to_device_tq(blob, shape[1], device, cuda);
    if (result.out != shape[0]) {
        throw std::invalid_argument("TQ projection output shape differs: " + name);
    }
    return result;
}

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
        CudaExecutionContext& execution, const mfq::ModelSource& source,
        const std::string& name) {
    const auto shape = mfq::tq_shape(source, name);
    if (shape.size() != 3) {
        throw std::invalid_argument(
            "TQ expert bank must be [experts,out,in]: " + name);
    }
    return load_tq_routed_weight(execution, source, name, shape[0], shape[1], shape[2]);
}

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
        CudaExecutionContext& execution, const mfq::ModelSource& source,
        const std::string& name, std::int64_t experts, std::int64_t output,
        std::int64_t input) {
    auto table = mfq::tq_table(source, name);
    if (experts < 1 || experts > UINT32_MAX || output < 1 || input < 1 ||
        table->rows() % experts || table->rows() / experts != std::uint64_t(output) ||
        table->width() != input) {
        throw std::invalid_argument("TQ routed projection shape differs: " + name);
    }
    auto& cache = execution.stores.tq_expert_cache;
    if (!cache) {
        const auto budget = execution.config.moe_device_cache_bytes.value_or(
            moe_expert_cache_budget_bytes(execution.stores.moe_expert_cache));
        const bool explicit_budget = execution.config.moe_device_cache_bytes.has_value() ||
            bool(execution.stores.moe_expert_cache);
        cache = std::make_shared<TqCache>(budget, execution.config.moe_host_cache_bytes,
            execution.config.moe_host_physical_bytes, !explicit_budget, execution.tq);
    }
    auto result = std::make_shared<TqRoutedWeight>();
    result->cache = cache;
    result->experts = experts;
    result->out = output;
    result->input = input;
    result->source_id = cache->add(std::move(table), output);
    return result;
}
