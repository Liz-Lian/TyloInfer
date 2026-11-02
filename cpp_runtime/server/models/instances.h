#pragma once

#include "catalog.h"
#include "factory.h"
#include "scheduler.h"

#include <atomic>
#include <future>
#include <memory>
#include <map>
#include <mutex>

namespace mfq::server {
class Storage;
struct LoadPolicy {
    bool pin = false;
    std::optional<std::chrono::seconds> idle_ttl;
};
struct InstanceInfo {
    enum class State { loaded, unloading, failed } state;
    std::string id, model, artifact_id, backend;
    engine::EngineInfo capabilities;
    MfqScheduler::Snapshot execution;
    LoadPolicy policy;
};

class ModelInstance : public std::enable_shared_from_this<ModelInstance> {
public:
    const std::string id, model, artifact_id, backend;
    ModelInstance(const ModelCatalog::Artifact& artifact, std::string backend, std::unique_ptr<engine::Engine> engine);
    InstanceInfo snapshot() const;
    const MfqSamplingParams& sampling() const { return sampling_; }
    // The returned native outbox keeps this instance alive and cancels on abandonment.
    std::shared_ptr<MfqScheduledRequest> submit(engine::EngineRequest request);
    bool cancel(const engine::RequestId& id);
private:
    friend class ModelInstances;
    void close(bool force);
    std::unique_ptr<engine::Engine> engine_;
    MfqScheduler scheduler_;
    MfqSamplingParams sampling_;
    std::atomic<bool> stopping_{false}, close_failed_{false};
    std::mutex admission_;
    bool accepting_ = true;
    LoadPolicy policy_;
    engine::Clock::time_point last_activity_ = engine::Clock::now();
    bool idle_expired(engine::Clock::time_point now);
};

struct LoadControl {
    enum class Phase { resolving, loading, initializing };
    const std::atomic<bool>* cancelled = nullptr;
    std::function<void(Phase)> progress;
    // Native construction is synchronous; cancellation there is observed on return.
    void checkpoint(Phase phase) const;
};

class ModelInstances {
public:
    using Aliases = std::map<std::string, std::string>;
    struct Support { bool supported; std::string reason; };
    ModelInstances(ModelCatalog& catalog, Storage& storage, EngineFactory factory);
    ~ModelInstances();
    Support support(const ModelCatalog::Artifact& artifact) const;
    InstanceInfo load(const std::string& model, LoadOptions options = {}, const LoadControl& control = {}, LoadPolicy policy = {});
    void unload(const std::string& instance_id, bool force = false);
    std::vector<InstanceInfo> list() const;
    std::shared_ptr<ModelInstance> get(const std::string& model) const;
    std::shared_ptr<ModelInstance> acquire(const std::string& model, const std::function<bool()>& cancelled = {});
    Aliases aliases() const;
    Aliases set_aliases(Aliases values);
    std::vector<std::string> names() const;
    void request_stop();
    void shutdown();
    void reap_idle();
private:
    struct Load {
        std::string id;
        std::shared_future<std::shared_ptr<ModelInstance>> result;
    };
    std::shared_ptr<ModelInstance> open(const ModelCatalog::Artifact& artifact, LoadOptions options,
        const LoadControl& control, LoadPolicy policy, bool reuse, const std::function<bool()>& cancelled);
    std::shared_ptr<ModelInstance> create(const ModelCatalog::Artifact& artifact, LoadOptions options,
        const LoadControl& control, LoadPolicy policy);
    Aliases read_aliases();
    ModelCatalog& catalog_;
    Storage& storage_;
    EngineFactory factory_;
    mutable std::mutex mutex_;
    std::mutex operations_;
    std::map<std::string, std::shared_ptr<ModelInstance>> instances_;
    std::optional<Load> loading_;
    Aliases aliases_;
    std::atomic<bool> stopping_{false};
};
}
