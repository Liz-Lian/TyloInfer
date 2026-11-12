#pragma once

#include "engine/load.h"
#include "engine/generation/sequence.h"
#include "engine/generation/mtp/metrics.h"
#include "engine/requests/result.h"

#include <memory>

namespace tyloi {
class ModelSource;
namespace models { class CausalModel; class State; class Layers; class Groups; class Ffn; struct Info; struct Input; struct Prefill; struct BatchConfig; }
namespace weights { class Cache; struct Loader; }
namespace engine { struct Config; class InferenceOutput; class Batch; class BatchDriver; }
}

namespace tyloi::cuda {

/** Own native execution resources and one independently loaded model or layer view. */
class Program final {
public:
    /** Own one exclusive numerical batch; requests and sequences must retire first. */
    class BatchLease final {
    public:
        ~BatchLease() noexcept;
        BatchLease(const BatchLease&) = delete;
        BatchLease& operator=(const BatchLease&) = delete;
        /** Borrow the real numerical batch while the lease remains alive. */
        engine::Batch& batch();
    private:
        friend class Program;
        explicit BatchLease(Program& owner);
        Program& owner_;
        std::unique_ptr<engine::Batch> batch_;
    };
    /** Observe the real request policy and executor-owned graph counters. */
    struct Stats {
        engine::mtp::GenerationStats request;
        std::uint64_t cycles = 0, accepted = 0, rejected = 0, captures = 0, reuses = 0;
    };
    /** Prepare operator execution without applying model or weight-loading options. */
    static std::unique_ptr<Program> bare();
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
    /** Load the grouped model view using the reference context length. */
    void load_groups(std::int64_t context);
    /** Load the controlled routed FFN view for one model layer. */
    void load_ffn(int layer);
    /** Borrow the execution resources while this owner remains alive. */
    Executor& executor();
    /** Borrow the weight cache while this owner remains alive. */
    weights::Cache& weights();
    /** Open the configured source lazily and return a loader borrowing this owner. */
    weights::Loader loader();
    /** Borrow the resolved execution configuration. */
    CudaLoadConfig& config();
    /** Borrow the loaded complete model. */
    models::CausalModel& model();
    /** Borrow the request state of the loaded complete model. */
    models::State& state();
    /** Borrow the loaded layer-wise model view. */
    models::Layers& layers();
    /** Borrow the loaded grouped model view. */
    models::Groups& groups();
    /** Borrow the prepared routed FFN view. */
    models::Ffn& ffn();
    /** Configure a quiescent loaded model, replacing its request graph and session owners. */
    void configure(engine::Config config);
    /** Create an exclusive numerical batch; this program must outlive the lease. */
    std::unique_ptr<BatchLease> make_batch();
    /** Begin an exclusive numerical batch owned by this program until end_batch. */
    engine::BatchDriver& begin_batch(models::BatchConfig config);
    /** Drain and retire the active numerical batch and its request states. */
    void end_batch();
    /** Run one controlled prefill; the caller retains its inputs and result until draining. */
    models::Prefill prefill(models::Input input, std::int64_t compute, engine::InferenceOutput& output);
    /** Run one exclusive request in the caller's scope; its owners outlive the sequence and drain after it. */
    StepSequence<engine::EventData> generate(engine::InferenceRequest& request,
        engine::InferenceOutput& output, bool predictor = true, std::string id = "check");
    /** Return request and cumulative generation counters without borrowing mutable policy state. */
    Stats stats() const;
    /** Wait for outstanding work and propagate any drain failure. */
    void drain();

private:
    Program();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

namespace internal {
/** Apply the owner's runtime options and actual graph eligibility to generation. */
void configure_generation(engine::Config&, const CudaLoadConfig&, Executor&,
    const weights::Cache&, const models::Info&);
/** Configure the shared native KV owner and optional offload store. */
void configure_model_kv(Executor&, std::shared_ptr<KvMemoryManager>&,
    std::shared_ptr<KvOffloadStore>&);
/** Assemble a native model with the shared loader and policy. */
std::unique_ptr<models::CausalModel> load_model(
    const CudaLoadOptions&, CudaLoadConfig&, Executor&, weights::Cache&,
    const std::shared_ptr<const ModelSource>&, bool components);
}
} // namespace tyloi::cuda
