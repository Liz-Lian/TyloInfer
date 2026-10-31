#pragma once

#include "backends/cuda/engine/executor/layer_inputs.h"
#include "formats/tq_table.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

mfq_tensor_backend::Tensor lookup_tq_table(
    const mfq::TqTable&, const std::int64_t*, std::size_t,
    const mfq_tensor_backend::Device&);

mfq_tensor_backend::Tensor lookup_tq_table(
    const mfq::TqTable&, mfq_tensor_backend::Tensor ids);

mfq_tensor_backend::Tensor lookup_tq_shards(
    const std::vector<std::shared_ptr<mfq::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& ids,
    const mfq_tensor_backend::Device& device);

std::shared_ptr<RowStage> make_tq_row_stage(
    const std::vector<std::shared_ptr<mfq::TqTable>>& tables,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
    const mfq_tensor_backend::Device& device);
