#pragma once
#include "engine/tensor.h"
#include <cstdint>
#include <memory>
#include <vector>

// A prepared table-row transfer. Encodings stay in their own loader; graph
// consumers use stable output storage and never allocate inside upload/decode.
class RowStage {
public:
    virtual ~RowStage()=default;
    virtual const ::tyloi::Tensor& output()const=0;
    virtual bool mapped_rows()const=0;
    virtual void upload(const std::vector<std::int64_t>& ids,const std::vector<std::int64_t>& shape)=0;
    virtual void decode()=0;
};

namespace tyloi::weights {
/** Own selected-row transfers and stable graph staging for compatible tables. */
class RowPipeline {
public:
    struct Source;
    /** Share immutable readers and cache identity, or return null for incompatible tables. */
    static std::shared_ptr<const Source> source(std::vector<Tensor> tables, std::int64_t rows_per_shard);
    /** Create independent transfer state for an immutable row source. */
    explicit RowPipeline(std::shared_ptr<const Source> source);
    ~RowPipeline();
    /** Start an eligible early transfer, returning whether it was issued. */
    bool prefetch(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, bool early_prefix = false);
    /** Collect selected rows on the requested device. */
    Tensor collect(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape, const Device& device);
    /** Create stable graph storage for later upload and decode. */
    std::shared_ptr<RowStage> stage(const std::vector<std::int64_t>& shape, const Device& device);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
