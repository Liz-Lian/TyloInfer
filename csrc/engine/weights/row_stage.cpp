#include "engine/weights/row_stage.h"
#include "engine/weights/encoding/rows.h"
#include "engine/weights/tq/rows.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/nint_row_pipeline.h"
#endif
#include <limits>
#include <stdexcept>

namespace tyloi::weights {
#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
struct NintStage final : RowStage {
    std::shared_ptr<NintRowPipeline> pipeline;
    std::shared_ptr<NintRowStage> stage;
    NintStage(std::shared_ptr<NintRowPipeline> value, const std::vector<std::int64_t>& shape, const Device& device)
        : pipeline(std::move(value)), stage(pipeline->make_stage(shape, device)) {}
    const Tensor& output() const override { return stage->output(); }
    bool mapped_rows() const override { return stage->mapped_rows(); }
    void decode() override { stage->decode(); }
    void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) override {
        pipeline->upload(*stage, ids, shape);
    }
};
}
#endif
struct RowPipeline::Source {
    std::vector<std::shared_ptr<TqTable>> tq;
    std::int64_t rows;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::shared_ptr<const NintRowSource> nint;
#endif
};
struct RowPipeline::Impl {
    std::shared_ptr<const Source> source;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::shared_ptr<NintRowPipeline> nint;
#endif
};
std::shared_ptr<const RowPipeline::Source> RowPipeline::source(std::vector<Tensor> tables, std::int64_t rows_per_shard) {
    if (tables.empty()) return {};
    bool tq = true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    bool nint_supported = true;
#endif
    for (const auto& table : tables) {
        const auto* rows = table.representation<Rows>();
        tq = tq && rows && bool(rows->tq());
#if TYLOINFER_ENABLE_MFQ_COMPAT
        nint_supported = nint_supported && rows && bool(rows->nint());
#endif
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!tq && !nint_supported) return {};
#else
    if (!tq) return {};
#endif
    if (rows_per_shard <= 0) throw std::invalid_argument("invalid row pipeline geometry");
    auto result = std::make_shared<Source>();
    result->rows = rows_per_shard;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::vector<std::shared_ptr<NintRows>> nint;
#endif
    for (const auto& table : tables) {
        const auto& rows = *table.representation<Rows>();
        if (rows.tq()) result->tq.push_back(rows.tq());
#if TYLOINFER_ENABLE_MFQ_COMPAT
        if (rows.nint()) nint.push_back(rows.nint());
#endif
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!nint.empty()) result->nint = std::make_shared<NintRowSource>(std::move(nint));
#endif
    return result;
}
RowPipeline::RowPipeline(std::shared_ptr<const Source> source) : impl_(std::make_unique<Impl>()) {
    if (!source) throw std::invalid_argument("row pipeline requires a source");
    impl_->source = std::move(source);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->source->nint) impl_->nint = std::make_shared<NintRowPipeline>(impl_->source->nint);
#endif
}
RowPipeline::~RowPipeline() = default;
bool RowPipeline::prefetch(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, bool early_prefix) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint) { impl_->nint->issue(ids, shape, early_prefix); return true; }
#endif
    return false;
}
Tensor RowPipeline::collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, const Device& device) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint) return impl_->nint->collect(ids, shape, device);
#endif
    std::size_t count = 1;
    for (auto extent : shape) {
        if (extent < 0 || (extent && count > std::numeric_limits<std::size_t>::max() / std::size_t(extent)))
            throw std::invalid_argument("invalid row pipeline shape");
        count *= std::size_t(extent);
    }
    if (count != ids.size()) throw std::invalid_argument("row pipeline shape mismatch");
    auto output = lookup_tq_shards(impl_->source->tq, impl_->source->rows, ids, device);
    auto result_shape = shape;
    result_shape.push_back(impl_->source->tq.front()->width());
    return output.dense().reshape(result_shape);
}
std::shared_ptr<RowStage> RowPipeline::stage(const std::vector<std::int64_t>& shape, const Device& device) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (impl_->nint) return std::make_shared<NintStage>(impl_->nint, shape, device);
#endif
    return make_tq_row_stage(impl_->source->tq, impl_->source->rows, shape, device);
}
}
