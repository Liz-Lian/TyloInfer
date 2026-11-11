#pragma once
#include "engine/tensor.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::weights {
/** Own a prepared table-row transfer and its stable device output. */
class RowStage : public std::enable_shared_from_this<RowStage> {
public:
    RowStage();
    virtual ~RowStage();
    RowStage(const RowStage&) = delete;
    RowStage& operator=(const RowStage&) = delete;
    /** Return the stable output tensor for this stage. */
    virtual const Tensor& output() const = 0;
    /** Report whether the transfer uses mapped host rows. */
    virtual bool mapped_rows() const = 0;
    /** Upload the selected host row IDs into the prepared staging storage. */
    virtual void upload(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape) = 0;
    /** Decode the staged rows on the active execution stream. */
    virtual void decode() = 0;
    /** Prepare graph transfer resources before entering capture. */
    void prepare_graph();
    /** Store host row IDs for the next graph replay without starting I/O. */
    void set_rows(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape);
    /** Register the row transfer and decode within an active graph window. */
    void decode_graph();
protected:
    /** Begin the selected row read during graph task reset when supported. */
    virtual void prefetch_rows(const std::vector<std::int64_t>& ids, const std::vector<std::int64_t>& shape);
private:
    struct Graph;
    void init_graph(const Device& device);
    std::unique_ptr<Graph> graph_;
};

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
