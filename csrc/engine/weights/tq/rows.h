#pragma once

#include "engine/weights/row_stage.h"
#include "engine/weights/encoding/tq_table.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

::tyloi::DenseTensor lookup_tq_table(
    const tyloi::TqTable&, const std::int64_t*, std::size_t,
    const ::tyloi::Device&);

::tyloi::DenseTensor lookup_tq_table(
    const tyloi::TqTable&, ::tyloi::DenseTensor ids);

::tyloi::DenseTensor lookup_tq_shards(
    const std::vector<std::shared_ptr<tyloi::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& ids,
    const ::tyloi::Device& device);

std::shared_ptr<RowStage> make_tq_row_stage(
    const std::vector<std::shared_ptr<tyloi::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
    const ::tyloi::Device& device);
