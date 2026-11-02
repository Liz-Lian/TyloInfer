#include "instances.h"
#include "errors.h"
#include "runtime_profile.h"
#include "instance.h"
#include "tyloinfer/model_source.h"

#include <algorithm>

namespace mfq::server {
namespace {
struct RequestLifetime {
    std::shared_ptr<ModelInstance> instance;
    std::string id;
    std::shared_ptr<MfqScheduledRequest> outbox;
    ~RequestLifetime() {
        if (outbox && !outbox->done()) try { instance->cancel(id); } catch (...) {}
    }
};
}

ModelInstance::ModelInstance(const ModelCatalog::Artifact& artifact, std::string selected_backend,
        std::unique_ptr<engine::Engine> loaded)
    : id(normalize_uuid({})), model(artifact.name), artifact_id(artifact.id),
      backend(std::move(selected_backend)), engine_(std::move(loaded)), scheduler_(*engine_) {
    auto source = mfq::open_model_source(artifact.path);
    const auto embedded = source->metadata().find("runtime.sampling.v1");
    const auto profile = resolve_mfq_runtime_profile(path_utf8(artifact.path), std::string(source->architecture()),
        scheduler_.info().model_type, model,
        embedded == source->metadata().end() ? std::string() : embedded->second, source->model_config_json());
    sampling_ = default_sampling_params(profile);
}

InstanceInfo ModelInstance::snapshot() const {
    const auto execution = scheduler_.snapshot();
    const auto state = close_failed_ || !execution.status.healthy ? InstanceInfo::State::failed :
        stopping_ ? InstanceInfo::State::unloading : InstanceInfo::State::loaded;
    return {state, id, model, artifact_id, backend, scheduler_.info(), execution, policy_};
}

std::shared_ptr<MfqScheduledRequest> ModelInstance::submit(engine::EngineRequest request) {
    std::lock_guard guard(admission_);
    if (!accepting_ || stopping_) throw ModelError(ModelError::Kind::stopping, "model_stopping", "model instance is stopping");
    last_activity_ = engine::Clock::now();
    auto lifetime = std::make_shared<RequestLifetime>();
    lifetime->instance = shared_from_this();
    lifetime->id = request.id;
    lifetime->outbox = scheduler_.submit(std::move(request));
    return {lifetime, lifetime->outbox.get()};
}

bool ModelInstance::cancel(const engine::RequestId& id) {
    std::lock_guard guard(admission_);
    return !stopping_ && scheduler_.cancel_request(id);
}

void ModelInstance::close(bool force) {
    {
        std::lock_guard admission(admission_);
        if (!stopping_) {
            // Serialize admission and use a scheduler barrier before observing pending requests.
            (void)scheduler_.status();
            if (!force && scheduler_.snapshot().requests)
                throw ModelError(ModelError::Kind::conflict, "model_busy", "model has active requests; cancel them or use force");
            stopping_ = true;
        }
    }
    try { scheduler_.shutdown(); }
    catch (...) { close_failed_ = true; throw; }
}

void LoadControl::checkpoint(Phase phase) const {
    const auto check = [&] {
        if (cancelled && cancelled->load())
            throw ModelError(ModelError::Kind::cancelled, "model_load_cancelled", "model loading was cancelled");
    };
    check();
    if (progress) progress(phase);
    check();
}

ModelInstances::ModelInstances(ModelCatalog& catalog, Storage& storage, EngineFactory factory)
    : catalog_(catalog), storage_(storage), factory_(std::move(factory)), aliases_(read_aliases()) {}
ModelInstances::~ModelInstances() { try { shutdown(); } catch (...) {} }

ModelInstances::Support ModelInstances::support(const ModelCatalog::Artifact& artifact) const {
    if (!artifact.complete) return {false, artifact.error};
    if (!factory_.load) return {false, "this server has no model execution backend"};
    try {
        auto source = mfq::open_model_source(artifact.path);
        if (!factory_.unavailable_reason) return {false, "execution backend does not provide a support query"};
        auto reason = factory_.unavailable_reason(*source);
        if (!reason.empty()) return {false, std::move(reason)};
        if (!source->has_asset(mfq::kTokenizerGgufAsset) && !source->has_asset("__mfq_asset__/hf/tokenizer.json"))
            return {false, "model has no native tokenizer assets"};
        return {true, {}};
    } catch (const std::exception&) { return {false, "execution support could not be determined"}; }
}

InstanceInfo ModelInstances::load(const std::string& model, LoadOptions options, const LoadControl& control, LoadPolicy policy) {
    control.checkpoint(LoadControl::Phase::resolving);
    return open(catalog_.resolve(model), options, control, policy, false, {})->snapshot();
}

std::shared_ptr<ModelInstance> ModelInstances::acquire(const std::string& model, const std::function<bool()>& cancelled) {
    if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
    if (cancelled && cancelled()) throw ModelError(ModelError::Kind::cancelled, "request_cancelled", "model acquisition was cancelled");
    try { return get(model); }
    catch (const ModelError& error) { if (error.kind != ModelError::Kind::not_found) throw; }
    auto source = model;
    {
        std::lock_guard guard(mutex_);
        for (const auto& [name, alias] : aliases_) if (alias == model) { source = name; break; }
    }
    return open(catalog_.resolve(source), {}, {}, {}, true, cancelled);
}

std::shared_ptr<ModelInstance> ModelInstances::open(const ModelCatalog::Artifact& artifact, LoadOptions options,
        const LoadControl& control, LoadPolicy policy, bool reuse, const std::function<bool()>& cancelled) {
    if (cancelled && cancelled()) throw ModelError(ModelError::Kind::cancelled, "request_cancelled", "model acquisition was cancelled");
    std::shared_future<std::shared_ptr<ModelInstance>> pending;
    std::optional<std::promise<std::shared_ptr<ModelInstance>>> promise;
    std::unique_lock operation(operations_, std::defer_lock);
    {
        std::lock_guard guard(mutex_);
        if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
        if (reuse) {
            for (const auto& [id, instance] : instances_) if (instance->artifact_id == artifact.id) {
                if (instance->stopping_) throw ModelError(ModelError::Kind::stopping, "model_stopping", "model instance is stopping");
                return instance;
            }
            if (loading_ && loading_->id == artifact.id) pending = loading_->result;
        }
        if (!pending.valid()) {
            if (!operation.try_lock()) throw ModelError(ModelError::Kind::conflict, "model_operation_busy", "another model operation is in progress");
            promise.emplace(); pending = promise->get_future().share();
            loading_ = Load{artifact.id, pending};
        }
    }
    if (promise) {
        try { promise->set_value(create(artifact, options, control, policy)); }
        catch (...) { promise->set_exception(std::current_exception()); }
        { std::lock_guard guard(mutex_); loading_.reset(); }
        operation.unlock();
    }
    do {
        if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
        if (cancelled && cancelled()) throw ModelError(ModelError::Kind::cancelled, "request_cancelled", "model acquisition was cancelled");
    } while (pending.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready);
    if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
    if (cancelled && cancelled()) throw ModelError(ModelError::Kind::cancelled, "request_cancelled", "model acquisition was cancelled");
    return pending.get();
}

std::shared_ptr<ModelInstance> ModelInstances::create(const ModelCatalog::Artifact& artifact, LoadOptions options,
        const LoadControl& control, LoadPolicy policy) {
    if (!factory_.load) throw ModelError(ModelError::Kind::unavailable, "backend_unavailable", "this server has no model execution backend");
    const auto supported = support(artifact);
    if (!supported.supported) throw ModelError(ModelError::Kind::invalid, "model_not_loadable", supported.reason);
    const auto capacity = artifact.context_capacity.value_or(2147483647);
    if (options.context_size < 0 || options.context_size > capacity)
        throw ModelError(ModelError::Kind::invalid, "invalid_request", "model loading options are outside their supported ranges");
    if (policy.idle_ttl && (policy.pin || policy.idle_ttl->count() < 1))
        throw ModelError(ModelError::Kind::invalid, "invalid_request", "idle timeout requires an unpinned model and a positive duration");
    {
        std::lock_guard guard(mutex_);
        for (const auto& [id, instance] : instances_)
            if (instance->artifact_id == artifact.id)
                throw ModelError(ModelError::Kind::conflict, "model_already_loaded", "model already has a loaded instance");
        for (const auto& [name, alias] : aliases_) {
            if (alias == artifact.name || alias == artifact.id)
                throw ModelError(ModelError::Kind::conflict, "model_alias_conflict", "model identity conflicts with a saved alias");
            if (name == artifact.name)
                for (const auto& [id, instance] : instances_) if (instance->model == name)
                    throw ModelError(ModelError::Kind::conflict, "model_alias_conflict", "model name with a saved alias must identify one artifact");
        }
    }
    control.checkpoint(LoadControl::Phase::loading);
    auto engine = factory_.load(artifact.path, options);
    if (!engine) throw std::runtime_error("engine factory returned no engine");
    auto instance = std::make_shared<ModelInstance>(artifact, factory_.backend, std::move(engine));
    instance->policy_ = policy;
    try {
        control.checkpoint(LoadControl::Phase::initializing);
        if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
        auto result = instance->snapshot();
        if (options.context_size && result.capabilities.max_context != options.context_size)
            throw ModelError(ModelError::Kind::invalid, "unsupported_requirement", "native factory did not satisfy the requested context size");
        {
            std::lock_guard guard(mutex_);
            if (stopping_) throw ModelError(ModelError::Kind::stopping, "server_stopping", "server is stopping");
            for (const auto& [name, alias] : aliases_) if (alias == instance->id)
                throw ModelError(ModelError::Kind::conflict, "model_alias_conflict", "instance identity conflicts with a saved alias");
            instances_.emplace(instance->id, instance);
        }
        return instance;
    } catch (...) {
        try { instance->close(true); }
        catch (...) {
            std::lock_guard guard(mutex_); instances_.emplace(instance->id, instance);
            throw;
        }
        throw;
    }
}

std::shared_ptr<ModelInstance> ModelInstances::get(const std::string& model) const {
    std::lock_guard guard(mutex_);
    const auto require_running = [&](const std::shared_ptr<ModelInstance>& instance) {
        if (instance->stopping_)
            throw ModelError(ModelError::Kind::stopping, "model_stopping", "model instance is stopping");
        return instance;
    };
    if (const auto found = instances_.find(model); found != instances_.end()) return require_running(found->second);
    for (const auto& [id, instance] : instances_)
        if (instance->artifact_id == model) return require_running(instance);
    std::shared_ptr<ModelInstance> named;
    for (const auto& [id, instance] : instances_) {
        const auto alias = aliases_.find(instance->model);
        if (instance->model != model && (alias == aliases_.end() || alias->second != model)) continue;
        if (named) throw ModelError(ModelError::Kind::conflict, "ambiguous_model", "model name selects more than one instance; use an instance ID");
        named = instance;
    }
    if (named) return require_running(named);
    throw ModelError(ModelError::Kind::not_found, "model_not_loaded", "model has no loaded instance");
}

void ModelInstances::unload(const std::string& instance_id, bool force) {
    std::unique_lock operation(operations_, std::try_to_lock);
    if (!operation.owns_lock()) throw ModelError(ModelError::Kind::conflict, "model_operation_busy", "another model operation is in progress");
    std::shared_ptr<ModelInstance> instance;
    {
        std::lock_guard guard(mutex_);
        const auto found = instances_.find(instance_id);
        if (found != instances_.end()) instance = found->second;
    }
    if (!instance)
        throw ModelError(ModelError::Kind::not_found, "model_not_loaded", "model has no loaded instance");
    instance->close(force);
    { std::lock_guard guard(mutex_); instances_.erase(instance_id); }
}

std::vector<InstanceInfo> ModelInstances::list() const {
    std::vector<std::shared_ptr<ModelInstance>> instances;
    { std::lock_guard guard(mutex_); for (const auto& [id, instance] : instances_) instances.push_back(instance); }
    std::vector<InstanceInfo> result;
    for (const auto& instance : instances) result.push_back(instance->snapshot());
    return result;
}

void ModelInstances::request_stop() {
    stopping_ = true;
    std::vector<std::shared_ptr<ModelInstance>> instances;
    { std::lock_guard guard(mutex_); for (const auto& [id, instance] : instances_) instances.push_back(instance); }
    std::exception_ptr failure;
    for (const auto& instance : instances) {
        std::lock_guard guard(instance->admission_);
        instance->accepting_ = false;
        if (!instance->stopping_) try { instance->scheduler_.cancel_all(); }
        catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}

void ModelInstances::shutdown() {
    std::exception_ptr failure;
    try { request_stop(); } catch (...) { failure = std::current_exception(); }
    std::lock_guard operation(operations_);
    std::vector<std::shared_ptr<ModelInstance>> instances;
    { std::lock_guard guard(mutex_); for (const auto& [id, instance] : instances_) instances.push_back(instance); }
    for (const auto& instance : instances) {
        try {
            instance->close(true);
            std::lock_guard guard(mutex_); instances_.erase(instance->id);
        } catch (...) { if (!failure) failure = std::current_exception(); }
    }
    if (failure) std::rethrow_exception(failure);
}

bool ModelInstance::idle_expired(engine::Clock::time_point now) {
    std::lock_guard guard(admission_);
    if (stopping_ || policy_.pin || !policy_.idle_ttl) return false;
    if (scheduler_.snapshot().requests) { last_activity_ = now; return false; }
    return now - last_activity_ >= *policy_.idle_ttl;
}
void ModelInstances::reap_idle() {
    if (stopping_) return;
    std::vector<std::shared_ptr<ModelInstance>> instances;
    { std::lock_guard guard(mutex_); for (const auto& [id, instance] : instances_) instances.push_back(instance); }
    for (const auto& instance : instances) if (instance->idle_expired(engine::Clock::now())) {
        // Only an actual eviction competes with explicit model operations.
        std::unique_lock operation(operations_, std::try_to_lock);
        if (!operation.owns_lock() || stopping_) return;
        if (!instance->idle_expired(engine::Clock::now())) continue;
        try { instance->close(false); }
        catch (const ModelError& error) { if (error.kind == ModelError::Kind::conflict) continue; throw; }
        { std::lock_guard guard(mutex_); instances_.erase(instance->id); }
    }
}
}
