#include "engine/weights/tq/load.h"

#include "engine/weights/tq/source.h"
#include "engine/weights/cache_internal.h"
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
    const auto bytes = source.read(name);
    std::vector<std::uint8_t> blob(bytes.size());
    std::memcpy(blob.data(), bytes.data(), bytes.size());
    auto result = to_device_tq(blob, shape[1], device, cuda);
    if (result.size(0) != shape[0]) {
        throw std::invalid_argument("TQ projection output shape differs: " + name);
    }
    return result;
}

std::shared_ptr<tyloi::TqBank> load_tq_routed_weight(
        tyloi::weights::Cache& weights, const std::shared_ptr<TqExec>& exec,
        const tyloi::source::Source& source,
        const std::string& name) {
    const auto shape = tyloi::tq_shape(source, name);
    if (shape.size() != 3) {
        throw std::invalid_argument(
            "TQ expert bank must be [experts,out,in]: " + name);
    }
    return load_tq_routed_weight(weights, exec, source, name, shape[0], shape[1], shape[2]);
}

std::shared_ptr<tyloi::TqBank> load_tq_routed_weight(
        tyloi::weights::Cache& weights, const std::shared_ptr<TqExec>& exec,
        const tyloi::source::Source& source,
        const std::string& name, std::int64_t experts, std::int64_t output,
        std::int64_t input) {
    auto table = tyloi::tq_table(source, name);
    if (experts < 1 || experts > UINT32_MAX || output < 1 || input < 1 ||
        table->rows() % experts || table->rows() / experts != std::uint64_t(output) ||
        table->width() != input) {
        throw std::invalid_argument("TQ routed projection shape differs: " + name);
    }
    auto& data = tyloi::weights::detail::cache(weights);
    if (!data.tq) {
        const auto bytes = data.device_bytes.value_or(moe_expert_cache_budget_bytes(data.moe));
        const bool explicit_budget = data.device_bytes.has_value() || bool(data.moe);
        data.tq = std::make_shared<TqCache>(bytes, data.host_bytes, data.host_limit, !explicit_budget, exec);
    }
    auto cache = data.tq;
    auto result = std::make_shared<tyloi::TqBank>();
    result->cache = cache;
    result->experts = experts;
    result->out = output;
    result->input = input;
    result->source_id = cache->add(std::move(table), output);
    return result;
}
