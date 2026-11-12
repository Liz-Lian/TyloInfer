#pragma once

#include "engine/load.h"

#include <memory>

namespace tyloi {
class ModelSource;
namespace models { class CausalModel; class State; class Layers; }
namespace weights { class Cache; }
}

namespace tyloi::cuda {

/** Own native execution resources and one independently loaded model or layer view. */
class Program final {
public:
    /** Prepare execution resources without loading model weights. */
    static std::unique_ptr<Program> open(CudaLoadOptions options, bool adapt = true);
    ~Program() noexcept;
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) = delete;
    Program& operator=(Program&&) = delete;

    /** Load a complete model and its request state. */
    void load(bool components = true);
    /** Load the independent layer-wise model view. */
    void load_layers();
    /** Borrow the execution resources while this owner remains alive. */
    Executor& executor();
    /** Borrow the weight cache while this owner remains alive. */
    weights::Cache& weights();
    /** Borrow the resolved execution configuration. */
    CudaLoadConfig& config();
    /** Borrow the opened model source. */
    const ModelSource& source() const;
    /** Borrow the loaded complete model. */
    models::CausalModel& model();
    /** Borrow the request state of the loaded complete model. */
    models::State& state();
    /** Borrow the loaded layer-wise model view. */
    models::Layers& layers();
    /** Wait for outstanding work and propagate any drain failure. */
    void drain();

private:
    Program();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace internal {
/** Configure the shared native KV owner and optional offload store. */
void configure_model_kv(Executor&, std::shared_ptr<KvMemoryManager>&,
    std::shared_ptr<KvOffloadStore>&);
/** Assemble a native model with the shared loader and policy. */
std::unique_ptr<models::CausalModel> load_model(
    const CudaLoadOptions&, CudaLoadConfig&, Executor&, weights::Cache&,
    const std::shared_ptr<const ModelSource>&, bool components);
}
} // namespace tyloi::cuda
