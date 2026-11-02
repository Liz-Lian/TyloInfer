#include "server.h"
#include "instance.h"
#include "engine.h"
#include "hf_tokenizer.h"
#include "tokenizer.h"
#include "tyloinfer/hf_safetensors_source.h"
#include "server_test_process.h"
#include "httplib.h"
#include "models/instances.h"
#include "models/errors.h"
#include "auth/auth.h"
#include "storage/sqlite.h"
#include "jobs/jobs.h"
#include "api/failure.h"
#include "sessions/sessions.h"

#include <cassert>
#include <cmath>
#include <future>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

using namespace mfq::engine;
using namespace mfq::server;
using namespace std::chrono_literals;

using Json = nlohmann::json;
namespace fs = std::filesystem;

namespace {
#define CHECK(value) do { if (!(value)) throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) + ": " #value); } while (false)
struct FixtureState {
    std::atomic<int> alive{0}, cleaned{0}, requests{0};
    std::atomic<bool> fail_close{false};
};
struct Directory {
    fs::path root = fs::temp_directory_path() / ("mfq-modules-" + normalize_uuid({}));
    ~Directory() { std::error_code ignored; fs::remove_all(root, ignored); }
};
struct FixtureEngine final : Engine {
    struct Work { EngineRequest request; bool cancelled = false; int steps = 0; };
    std::map<std::string, Work> active;
    bool loaded = true;
    std::int64_t context_size = 128;
    std::shared_ptr<FixtureState> state;
    explicit FixtureEngine(std::shared_ptr<FixtureState> state = {}) : state(std::move(state)) {
        if (this->state) ++this->state->alive;
    }
    ~FixtureEngine() override { if (state) --state->alive; }
    EngineInfo info() const override {
        EngineInfo result;
        result.max_context = context_size; result.vocab_size = 8; result.model_type = "qwen3_5";
        result.capabilities.family = "fixture";
        return result;
    }
    Admission admit(EngineRequest&& request) override {
        if (!active.empty()) return Admission::deferred;
        if (state) ++state->requests;
        const auto id = request.id;
        active.emplace(id, Work{std::move(request)});
        return Admission::accepted;
    }
    void cancel(const RequestId& id) override { if (active.count(id)) active.at(id).cancelled = true; }
    EngineStatus status() const override { return {loaded && active.empty() ? 1U : 0U, loaded}; }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        EngineStepResult result;
        for (auto it = active.begin(); it != active.end();) {
            auto& work = it->second;
            const auto id = it->first;
            if (work.cancelled) {
                it = active.erase(it);
                result.events.push_back({id, Cancelled{{2, 0}, {}}}); continue;
            }
            if (std::find(eligible.begin(), eligible.end(), id) == eligible.end()) { ++it; continue; }
            const auto& input = work.request.input;
            const auto prompt = input.chat_input ? input.chat_input->template_inputs.messages.back().render_content() : input.prompt;
            if (prompt == "fail") throw std::runtime_error("fixture execution failure");
            result.advanced.push_back(id);
            std::this_thread::sleep_for(2ms);
            if (prompt == "slow") { ++it; continue; }
            common_chat_msg_diff diff;
            diff.content_delta = "hi";
            if (prompt == "tool") {
                diff.reasoning_content_delta = "thinking";
                diff.tool_call_index = 0; diff.tool_call_delta = {"lookup", "{}", "call-1"};
            }
            if (prompt == "inspect") {
                auto observed = mfq::protocol::sampling_params_json(input.sampling);
                observed["messages"] = Json::array();
                for (const auto& message : input.chat_input->template_inputs.messages)
                    observed["messages"].push_back({{"role", message.role}, {"content", message.render_content()}, {"reasoning", message.reasoning_content}, {"tools", message.tool_calls.size()}});
                observed["tools"] = input.chat_input->template_inputs.tools.size();
                diff.content_delta = observed.dump();
            }
            if (prompt == "fragmented-tool" || prompt == "bad-tool") {
                diff = {}; diff.tool_call_index = 0;
                if (work.steps++ == 0) {
                    diff.reasoning_content_delta = "plan"; diff.content_delta = "checking";
                    diff.tool_call_delta = {"look", "", "call-1"};
                    result.events.push_back({id, OutputDelta{{2}, {diff}}}); ++it; continue;
                }
                if (work.steps == 2) {
                    diff.tool_call_delta = {"lookup", "{\"x\":", "call-1"};
                    result.events.push_back({id, OutputDelta{{2}, {diff}}}); ++it; continue;
                }
                diff.tool_call_delta.arguments = prompt == "bad-tool" ? "bad}" : "1}";
                result.events.push_back({id, OutputDelta{{2}, {diff}}});
                if (prompt != "bad-tool") {
                    diff = {}; diff.tool_call_index = 1; diff.tool_call_delta = {"lookup", "{\"x\":2}", "call-2"};
                    result.events.push_back({id, OutputDelta{{2}, {diff}}});
                }
                it = active.erase(it);
                result.events.push_back({id, Completed{"tool_calls", {12, 4}, {}}}); continue;
            }
            result.events.push_back({id, OutputDelta{{2}, {diff}}});
            it = active.erase(it);
            result.events.push_back({id, Completed{prompt == "tool" ? "tool_calls" : prompt == "limit" ? "length" : "stop", {2, 1}, {}}});
        }
        result.status = status(); result.has_work = !active.empty(); return result;
    }
    SessionResult session(const SessionCommand&) override { return {}; }
    std::int64_t reload(std::int64_t) override { throw std::runtime_error("fixture reload unsupported"); }
    void shutdown() override {
        active.clear();
        if (state && state->fail_close) throw std::runtime_error("fixture shutdown failure");
        if (state && loaded) ++state->cleaned;
        loaded = false;
    }
    ControlResult control(ControlRequest) override { return Metrics{{"fixture_allocated_bytes", loaded ? 1024.0 : 0.0}}; }
};

void write_file(const fs::path& path, const std::string& value) {
    std::ofstream stream(path, std::ios::binary); stream << value; CHECK(stream.good());
}
void fixture_model(const fs::path& root) {
    fs::create_directories(root);
    write_file(root / "config.json", R"({"model_type":"qwen3_5","vocab_size":8,"max_position_embeddings":8192})");
    const std::string header = R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}})";
    std::ofstream weights(root / "model.safetensors", std::ios::binary);
    const std::uint64_t size = header.size();
    weights.write(reinterpret_cast<const char*>(&size), sizeof(size));
    weights << header; weights.put(0); weights.put(0); weights.close();
    write_file(root / "tokenizer.json", R"({"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":[["h","i"]]},"added_tokens":[{"id":3,"content":"<eos>","special":true},{"id":4,"content":"<end>","special":true}]})");
    write_file(root / "tokenizer_config.json", R"({"eos_token":"<eos>","add_bos_token":false})");
    write_file(root / "generation_config.json", R"({"eos_token_id":[3,4]})");
    write_file(root / "chat_template.jinja", "{{ messages[0]['content'] }}");
}

void tokenizer_checks(const fs::path& root) {
    mfq::HfSafetensorsSource source(root);
    MfqTokenizer tokenizer(hf_tokenizer_gguf(source));
    CHECK(tokenizer.vocab_size() == 8 && tokenizer.eos_token() == 3);
    CHECK(tokenizer.is_eog(3) && tokenizer.is_eog(4));
    CHECK(tokenizer.tokenize("hi", true) == std::vector<std::int64_t>{2});
    CHECK(tokenizer.chat_template() == "{{ messages[0]['content'] }}");
    write_file(root / "generation_config.json", R"({"eos_token_id":[8]})");
    bool invalid = false;
    try { (void)hf_tokenizer_gguf(mfq::HfSafetensorsSource(root)); } catch (const std::exception&) { invalid = true; }
    CHECK(invalid);
    write_file(root / "generation_config.json", R"({"eos_token_id":[3,4]})");
    write_file(root / "tokenizer.json", R"({"model":{"type":"BPE","vocab":{"h":2000001},"merges":[]}})");
    invalid = false;
    try { (void)hf_tokenizer_gguf(mfq::HfSafetensorsSource(root)); } catch (const std::exception&) { invalid = true; }
    CHECK(invalid);
    fixture_model(root);
}

template<class Action> void model_error(const char* code, Action action) {
    try { action(); }
    catch (const ModelError& error) { CHECK(std::string(error.code) == code); return; }
    throw std::runtime_error(std::string("expected model error: ") + code);
}

void module_checks() {
    Directory directory;
    Storage storage(directory.root / "models.sqlite3");
    fixture_model(directory.root / "model");
    ModelCatalog catalog({directory.root});
    const auto artifact = catalog.list(true).at(0);
    CHECK(artifact.complete && artifact.error.empty() && artifact.context_capacity == 8192);
    ModelInstances unavailable(catalog, storage, {});
    CHECK(!unavailable.support(artifact).supported);
    CHECK(catalog.list(false).at(0).complete);

    auto state = std::make_shared<FixtureState>();
    LoadOptions received;
    int loads = 0;
    bool reject_support = false;
    std::function<void()> during_load;
    EngineFactory factory;
    factory.backend = "fixture";
    factory.unavailable_reason = [&](const mfq::ModelSource&) -> std::string {
        if (reject_support) throw std::runtime_error("fixture support unavailable");
        return {};
    };
    factory.load = [&](const fs::path&, const LoadOptions& options) -> std::unique_ptr<Engine> {
        ++loads; received = options;
        auto engine = std::make_unique<FixtureEngine>(state);
        if (options.context_size) engine->context_size = options.context_size;
        if (options.context_size == 13) throw std::runtime_error("fixture load failure after allocation");
        if (during_load) during_load();
        return engine;
    };
    ModelInstances models(catalog, storage, factory);
    reject_support = true;
    CHECK(!models.support(artifact).supported);
    CHECK(catalog.list(false).at(0).error.empty());
    reject_support = false;
    CHECK(models.support(artifact).supported);
    model_error("invalid_request", [&] { models.load(artifact.id, {-1}); });
    model_error("invalid_request", [&] { models.load(artifact.id, {8193}); });
    std::atomic<bool> cancelled{true};
    model_error("model_load_cancelled", [&] { models.load(artifact.id, {}, {&cancelled}); });
    CHECK(loads == 0);
    try { models.load(artifact.id, {13}); CHECK(false); }
    catch (const std::runtime_error& error) { CHECK(std::string(error.what()) == "fixture load failure after allocation"); }
    CHECK(state->alive == 0 && models.list().empty());

    cancelled = false;
    during_load = [&] { cancelled = true; };
    model_error("model_load_cancelled", [&] { models.load(artifact.id, {}, {&cancelled}); });
    CHECK(state->alive == 0 && state->cleaned == 1 && models.list().empty());
    during_load = {};
    cancelled = false;
    try {
        models.load(artifact.id, {}, {nullptr, [](LoadControl::Phase phase) {
            if (phase == LoadControl::Phase::initializing) throw std::runtime_error("fixture progress failure");
        }});
        CHECK(false);
    } catch (const std::runtime_error& error) { CHECK(std::string(error.what()) == "fixture progress failure"); }
    CHECK(state->alive == 0 && models.list().empty());

    std::promise<void> entered, release;
    auto ready = entered.get_future();
    auto gate = release.get_future().share();
    during_load = [&] { entered.set_value(); CHECK(gate.wait_for(5s) == std::future_status::ready); };
    auto loading = std::async(std::launch::async, [&] {
        model_error("model_load_cancelled", [&] { models.load(artifact.id, {}, {&cancelled}); });
    });
    CHECK(ready.wait_for(5s) == std::future_status::ready);
    model_error("model_operation_busy", [&] { models.load(artifact.id); });
    model_error("model_operation_busy", [&] { models.unload("missing"); });
    CHECK(models.list().empty());
    cancelled = true;
    release.set_value();
    CHECK(loading.wait_for(5s) == std::future_status::ready); loading.get();
    CHECK(state->alive == 0 && models.list().empty());
    during_load = {};
    cancelled = false;

    std::vector<LoadControl::Phase> phases;
    auto loaded = models.load(artifact.id, {}, {&cancelled, [&](LoadControl::Phase phase) { phases.push_back(phase); }});
    CHECK(received.context_size == 0 && phases.size() == 3);
    CHECK(loaded.state == InstanceInfo::State::loaded);
    auto instance = models.get(loaded.id);
    auto input = [] { InferenceInput value; value.chat = false; value.prompt = "slow"; return value; };
    auto request = instance->submit({"direct-cancel", input()});
    model_error("model_busy", [&] { models.unload(loaded.id); });
    CHECK(instance->cancel("direct-cancel"));
    int terminals = 0;
    const auto deadline = Clock::now() + 5s;
    while (!terminals && Clock::now() < deadline)
        for (const auto& event : request->wait_for(100ms)) {
            CHECK(event.id == "direct-cancel");
            if (terminal(event.data)) { ++terminals; CHECK(std::holds_alternative<Cancelled>(event.data)); }
        }
    CHECK(terminals == 1 && request->done());
    request.reset();
    request = instance->submit({"abandoned", input()});
    request.reset();
    const auto abandoned = Clock::now() + 5s;
    while (instance->snapshot().execution.requests && Clock::now() < abandoned) std::this_thread::sleep_for(1ms);
    CHECK(instance->snapshot().execution.requests == 0);
    instance.reset();
    models.unload(loaded.id);
    CHECK(models.list().empty() && state->alive == 0);

    std::weak_ptr<ModelInstance> retained;
    {
        ModelInstances owner(catalog, storage, factory);
        const auto info = owner.load(artifact.name);
        auto instance = owner.get(info.id);
        retained = instance;
        request = instance->submit({"outlives-owner", input()});
        owner.unload(info.id, true);
        CHECK(owner.list().empty());
    }
    CHECK(!retained.expired());
    const auto events = request->wait_for(100ms);
    CHECK(std::count_if(events.begin(), events.end(), [](const auto& event) { return terminal(event.data); }) == 1);
    CHECK(std::holds_alternative<Cancelled>(events.back().data));
    request.reset();
    CHECK(retained.expired() && state->alive == 0);

    loaded = models.load(artifact.id);
    state->fail_close = true;
    try { models.unload(loaded.id, true); CHECK(false); }
    catch (const std::runtime_error& error) { CHECK(std::string(error.what()) == "fixture shutdown failure"); }
    CHECK(models.list().size() == 1 && models.list()[0].state == InstanceInfo::State::failed);
    CHECK(state->alive == 1);
    model_error("model_stopping", [&] { models.get(loaded.id); });
    model_error("model_already_loaded", [&] { models.load(artifact.id); });

    Authentication auth(storage, directory.root / "auth" / "policy.json", "root-key", true);
    const auto created = auth.create_key({{"name", "transaction"}, {"scopes", {"inference"}}});
    const auto key_id = created["key"]["id"].get<std::string>(), token = created["token"].get<std::string>();
    auth.require(token, Scope::inference);
    try { auth.require(token, Scope::admin); CHECK(false); }
    catch (const AuthError& error) { CHECK(error.kind == AuthError::forbidden); }
    {
        auto access = storage.access();
        sqlite::execute(access.database, "UPDATE api_keys SET scopes_json='invalid'");
    }
    try { auth.rotate_key(key_id); CHECK(false); }
    catch (const Json::parse_error&) {}
    {
        auto access = storage.access();
        sqlite::execute(access.database, "UPDATE api_keys SET scopes_json='[\"inference\"]'");
    }
    auth.require(token, Scope::inference);
    CHECK(auth.list_keys()["data"][0]["prefix"] == created["key"]["prefix"]);
}

template<class Terminal> void expect_terminal(const std::shared_ptr<MfqScheduledRequest>& request) {
    const auto deadline = Clock::now() + 5s;
    int terminals = 0;
    while (!terminals && Clock::now() < deadline)
        for (const auto& event : request->wait_for(100ms))
            if (terminal(event.data)) { ++terminals; CHECK(std::holds_alternative<Terminal>(event.data)); }
    CHECK(terminals == 1 && request->done());
}

void multi_instance_checks() {
    Directory directory;
    Storage storage(directory.root / "models.sqlite3");
    for (const auto* name : {"alpha", "beta", "broken"}) fixture_model(directory.root / name);
    ModelCatalog catalog({directory.root});
    auto alpha_state = std::make_shared<FixtureState>(), beta_state = std::make_shared<FixtureState>();
    EngineFactory factory;
    factory.backend = "fixture";
    factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
    factory.load = [&](const fs::path& path, const LoadOptions&) -> std::unique_ptr<Engine> {
        auto engine = std::make_unique<FixtureEngine>(path.filename() == "alpha" ? alpha_state : beta_state);
        if (path.filename() == "broken") throw MfqResourceExhausted("fixture load budget exhausted");
        return engine;
    };
    ModelInstances models(catalog, storage, factory);
    auto alpha = models.load("alpha"), beta = models.load("beta");
    CHECK(models.list().size() == 2 && alpha.id != beta.id);
    auto a = models.get(alpha.id), b = models.get(beta.id);
    CHECK(a == models.get(alpha.artifact_id) && a == models.get("alpha"));
    CHECK(b == models.get(beta.artifact_id) && b == models.get("beta"));
    const auto input = [](std::string prompt) { InferenceInput value; value.chat = false; value.prompt = std::move(prompt); return value; };
    auto first = a->submit({"same-request-id", input("slow")});
    auto second = b->submit({"same-request-id", input("slow")});
    CHECK(a->cancel("same-request-id"));
    expect_terminal<Cancelled>(first);
    CHECK(b->snapshot().state == InstanceInfo::State::loaded && b->snapshot().execution.requests == 1);
    model_error("model_busy", [&] { models.unload(beta.id); });
    models.unload(alpha.id);
    first.reset(); a.reset();
    CHECK(models.list().size() == 1 && models.get("beta") == b);
    try { models.load("broken"); CHECK(false); }
    catch (const MfqResourceExhausted&) {}
    CHECK(models.list().size() == 1 && b->snapshot().execution.requests == 1);

    alpha = models.load("alpha"); a = models.get(alpha.id);
    first = a->submit({"execution-failure", input("fail")});
    expect_terminal<Failed>(first);
    const auto failed = Clock::now() + 5s;
    while (a->snapshot().state != InstanceInfo::State::failed && Clock::now() < failed) std::this_thread::sleep_for(1ms);
    CHECK(a->snapshot().state == InstanceInfo::State::failed);
    CHECK(b->snapshot().state == InstanceInfo::State::loaded && b->snapshot().execution.requests == 1);
    models.unload(alpha.id, true);
    first.reset(); a.reset();
    alpha = models.load("alpha"); a = models.get(alpha.id);
    first = a->submit({"force-unload", input("slow")});
    models.unload(alpha.id, true);
    expect_terminal<Cancelled>(first);
    CHECK(b->snapshot().state == InstanceInfo::State::loaded && b->snapshot().execution.requests == 1);
    first.reset(); a.reset();
    CHECK(b->cancel("same-request-id"));
    expect_terminal<Cancelled>(second);
    second = b->submit({"still-usable", input("hi")});
    expect_terminal<Completed>(second);
    second = b->submit({"shutdown-survivor", input("slow")});

    alpha = models.load("alpha"); a = models.get(alpha.id);
    alpha_state->fail_close = true;
    try { models.unload(alpha.id, true); CHECK(false); }
    catch (const std::runtime_error& error) { CHECK(std::string(error.what()) == "fixture shutdown failure"); }
    CHECK(models.list().size() == 2 && models.get("beta") == b);
    models.request_stop();
    CHECK(models.get("beta") == b);
    model_error("model_stopping", [&] { b->submit({"after-stop", input("slow")}); });
    try { models.shutdown(); CHECK(false); }
    catch (const std::runtime_error& error) { CHECK(std::string(error.what()) == "fixture shutdown failure"); }
    CHECK(models.list().size() == 1 && models.list()[0].id == alpha.id);
    CHECK(models.list()[0].state == InstanceInfo::State::failed);
    expect_terminal<Cancelled>(second);
    CHECK(b->snapshot().execution.requests == 0);
    model_error("model_not_loaded", [&] { models.get(beta.id); });
    model_error("server_stopping", [&] { models.load("beta"); });
}

void residency_checks() {
    Directory directory; fixture_model(directory.root / "idle"); fixture_model(directory.root / "pinned");
    Storage storage(directory.root / "models.sqlite3");
    ModelCatalog catalog({directory.root});
    EngineFactory factory;
    factory.backend = "fixture";
    factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
    factory.load = [](const fs::path&, const LoadOptions&) { return std::make_unique<FixtureEngine>(); };
    ModelInstances instances(catalog, storage, factory);
    const auto idle = instances.load("idle", {}, {}, {false, 1s});
    const auto pinned = instances.load("pinned", {}, {}, {true, {}});
    auto model = instances.get(idle.id);
    InferenceInput input; input.chat = false; input.prompt = "slow";
    auto request = model->submit({"active", input});
    std::this_thread::sleep_for(1100ms);
    instances.reap_idle(); CHECK(instances.list().size() == 2);
    CHECK(model->cancel("active")); expect_terminal<Cancelled>(request);
    std::this_thread::sleep_for(1100ms);
    instances.reap_idle();
    CHECK(instances.list().size() == 1 && instances.list()[0].id == pinned.id);
    model_error("model_not_loaded", [&] { instances.get(idle.id); });
    std::atomic<bool> scanning{true};
    auto scans = std::async(std::launch::async, [&] {
        while (scanning) { instances.reap_idle(); std::this_thread::yield(); }
    });
    try {
        for (int i = 0; i < 100; ++i) {
            const auto loaded = instances.load("idle");
            instances.unload(loaded.id);
        }
    } catch (...) { scanning = false; scans.get(); throw; }
    scanning = false; scans.get();
    instances.unload(pinned.id);
}

void acquire_checks() {
    for (int mode = 0; mode < 5; ++mode) {
        Directory directory; Storage storage(directory.root / "models.sqlite3");
        fixture_model(directory.root / "alpha"); fixture_model(directory.root / "beta");
        ModelCatalog catalog({directory.root});
        std::promise<void> entered, release;
        auto ready = entered.get_future();
        auto gate = release.get_future().share();
        auto state = std::make_shared<FixtureState>();
        std::atomic<int> loads{0}; std::atomic<bool> fail{mode == 2}, cancelled{false};
        EngineFactory factory; factory.backend = "fixture";
        factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
        factory.load = [&](const fs::path&, const LoadOptions&) -> std::unique_ptr<Engine> {
            auto engine = std::make_unique<FixtureEngine>(state);
            if (++loads == 1) entered.set_value();
            gate.wait();
            if (fail) throw std::runtime_error("shared load failure");
            return engine;
        };
        ModelInstances models(catalog, storage, factory);
        models.set_aliases({{"alpha", "fast"}});
        model_error("request_cancelled", [&] { models.acquire("fast", [] { return true; }); });
        CHECK(loads == 0 && models.names() == (std::vector<std::string>{"beta", "fast"}));
        auto first = std::async(std::launch::async, [&] {
            return mode == 1 ? models.load("alpha").id : models.acquire("fast", [&] { return mode == 4 && cancelled.load(); })->id;
        });
        if (ready.wait_for(5s) != std::future_status::ready) { release.set_value(); CHECK(false); }
        auto second = std::async(std::launch::async, [&] { return models.acquire("alpha")->id; });
        auto third = std::async(std::launch::async, [&] { return models.acquire("fast", [&] { return mode == 0 && cancelled.load(); })->id; });
        const bool waiting = second.wait_for(50ms) == std::future_status::timeout && third.wait_for(50ms) == std::future_status::timeout;
        std::string conflict;
        try { models.acquire("beta"); } catch (const ModelError& error) { conflict = error.code; }
        cancelled = true;
        bool stopped = false;
        if (mode == 0) stopped = third.wait_for(1s) == std::future_status::ready;
        if (mode == 3) {
            models.request_stop();
            stopped = second.wait_for(1s) == std::future_status::ready && third.wait_for(1s) == std::future_status::ready;
        }
        const bool owner_waits = first.wait_for(0ms) == std::future_status::timeout;
        release.set_value();
        CHECK(waiting && owner_waits && conflict == "model_operation_busy" && loads == 1);
        if (mode == 2) {
            for (auto* result : {&first, &second, &third}) {
                try { result->get(); CHECK(false); }
                catch (const std::runtime_error& error) { CHECK(std::string(error.what()) == "shared load failure"); }
            }
            CHECK(models.list().empty() && state->alive == 0);
            fail = false; CHECK(!models.acquire("fast")->id.empty() && loads == 2);
        } else if (mode == 3) {
            CHECK(stopped);
            for (auto* result : {&first, &second, &third}) model_error("server_stopping", [&] { result->get(); });
            CHECK(models.list().empty() && state->alive == 0);
        } else {
            const auto id = second.get();
            if (mode == 4) model_error("request_cancelled", [&] { first.get(); });
            else CHECK(first.get() == id);
            if (mode == 0) { CHECK(stopped); model_error("request_cancelled", [&] { third.get(); }); }
            else CHECK(third.get() == id);
            CHECK(models.list().size() == 1 && models.get("fast")->id == id && loads == 1);
            model_error("model_already_loaded", [&] { models.load("alpha"); });
        }
        models.shutdown(); CHECK(state->alive == 0);
    }
}

void names_checks() {
    Directory directory;
    fixture_model(directory.root / "alpha"); fixture_model(directory.root / "beta");
    ModelCatalog catalog({directory.root});
    EngineFactory factory; factory.backend = "fixture";
    factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
    factory.load = [](const fs::path&, const LoadOptions&) { return std::make_unique<FixtureEngine>(); };
    const auto database = directory.root / "models.sqlite3";
    {
        Storage storage(database);
        {
            auto access = storage.access();
            sqlite::execute(access.database, R"(INSERT INTO schema_meta(key,value) VALUES('runtime_model_aliases','{"alpha":"fast"}'))");
        }
        ModelInstances models(catalog, storage, factory);
        CHECK(models.aliases().at("alpha") == "fast" && models.list().empty());
        CHECK(models.names() == (std::vector<std::string>{"beta", "fast"}));
        model_error("model_not_loaded", [&] { models.get("fast"); });
        auto alpha = models.load("alpha"); models.load("beta");
        CHECK(models.names() == (std::vector<std::string>{"beta", "fast"}));
        CHECK(models.get("fast") == models.get("alpha") && models.get("fast") == models.get(alpha.id));
        CHECK(models.get("fast") == models.get(alpha.artifact_id));
        CHECK(models.set_aliases({{"alpha", " fast "}, {"beta", "beta"}}).at("alpha") == "fast");
        model_error("unknown_model", [&] { models.set_aliases({{"missing", "unknown"}}); });
        for (const auto& values : std::vector<ModelInstances::Aliases>{
                {{"alpha", "beta"}}, {{"alpha", "same"}, {"beta", "same"}},
                {{"alpha", alpha.id}}, {{"alpha", alpha.artifact_id}}})
            model_error("model_alias_conflict", [&] { models.set_aliases(values); });
        for (const auto& value : {std::string("bad\nname"), std::string(256, 'x'), std::string("bad\x7f"), std::string("\xff")})
            model_error("invalid_model_alias", [&] { models.set_aliases({{"alpha", value}}); });
        std::string unicode;
        for (int i = 0; i < 255; ++i) unicode += "模";
        models.set_aliases({{"alpha", unicode}}); CHECK(models.get(unicode)->id == alpha.id);
        model_error("invalid_model_alias", [&] { models.set_aliases({{"alpha", unicode + "模"}}); });
        models.set_aliases({{"alpha", "fast"}});
        {
            auto access = storage.access();
            sqlite::execute(access.database, "CREATE TRIGGER reject_alias BEFORE UPDATE ON schema_meta WHEN NEW.key='runtime_model_aliases' "
                "BEGIN SELECT RAISE(ABORT,'fixture alias save failure'); END");
        }
        model_error("model_alias_save_failed", [&] { models.set_aliases({{"alpha", "changed"}}); });
        CHECK(models.aliases().at("alpha") == "fast" && models.get("fast")->id == alpha.id);
        {
            auto access = storage.access();
            sqlite::Statement saved(access.database, "SELECT value FROM schema_meta WHERE key='runtime_model_aliases'");
            CHECK(saved.step() && Json::parse(saved.column(0).get<std::string>())["alpha"] == "fast");
            sqlite::execute(access.database, "DROP TRIGGER reject_alias");
        }
        // A caller holding Storage must still be able to resolve models during an alias write.
        auto access = storage.access();
        const auto deadline = Clock::now() + 5s;
        auto update = std::async(std::launch::async, [&] {
            while (Clock::now() < deadline) {
                try { return models.set_aliases({{"alpha", "fast"}}); }
                catch (const ModelError& error) { if (std::string(error.code) != "model_operation_busy") throw; }
                std::this_thread::yield();
            }
            throw std::runtime_error("alias update never acquired the operation lock");
        });
        bool busy = false;
        do {
            try { models.load(alpha.artifact_id); }
            catch (const ModelError& error) { busy = std::string(error.code) == "model_operation_busy"; }
            if (!busy) std::this_thread::sleep_for(1ms);
        } while (!busy && Clock::now() < deadline);
        auto lookup = std::async(std::launch::async, [&] { return models.get("fast")->id; });
        const bool resolved = lookup.wait_for(1s) == std::future_status::ready;
        access.lock.unlock();
        CHECK(update.get().at("alpha") == "fast" && lookup.get() == alpha.id && busy && resolved);
        auto instance = models.get("fast");
        InferenceInput input; input.chat = false; input.prompt = "slow";
        auto request = instance->submit({"rename", input});
        models.set_aliases({{"alpha", "faster"}});
        model_error("model_not_loaded", [&] { models.get("fast"); });
        CHECK(instance == models.get("faster") && instance->cancel("rename")); expect_terminal<Cancelled>(request);
        request.reset(); models.unload(alpha.id); instance.reset();
        CHECK(models.aliases().at("alpha") == "faster" && models.names() == (std::vector<std::string>{"beta", "faster"}));
        alpha = models.load("alpha"); CHECK(models.get("faster")->id == alpha.id);
        fixture_model(directory.root / "faster"); catalog.list(true);
        model_error("model_alias_conflict", [&] { models.load("faster"); });
        models.set_aliases({{"alpha", ""}}); CHECK(models.aliases().empty());
        const std::string header = R"({"model.embed_tokens.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}})";
        {
            std::ofstream weights(directory.root / "alpha" / "model.safetensors", std::ios::binary);
            const std::uint64_t size = header.size(); weights.write(reinterpret_cast<const char*>(&size), sizeof(size));
            weights << header; for (int i = 0; i < 4; ++i) weights.put(0);
        }
        catalog.list(true);
        const auto replacement = models.load("alpha"); CHECK(replacement.artifact_id != alpha.artifact_id);
        model_error("ambiguous_model", [&] { models.get("alpha"); });
        model_error("model_alias_conflict", [&] { models.set_aliases({{"alpha", "fast"}}); });
        const auto names = models.names();
        CHECK(std::find(names.begin(), names.end(), alpha.id) != names.end());
        CHECK(std::find(names.begin(), names.end(), replacement.id) != names.end());
        models.unload(alpha.id); models.set_aliases({{"alpha", "fast"}});
    }
    {
        Storage storage(database); ModelInstances restored(catalog, storage, factory);
        CHECK(restored.aliases().at("alpha") == "fast" && restored.list().empty());
        const auto loaded = restored.load("alpha"); CHECK(restored.get("fast")->id == loaded.id);
    }
    Storage storage(database);
    {
        auto access = storage.access();
        sqlite::execute(access.database, "UPDATE schema_meta SET value='[]' WHERE key='runtime_model_aliases'");
    }
    model_error("invalid_saved_aliases", [&] { ModelInstances invalid(catalog, storage, factory); });
}

void jobs_checks() {
    Directory directory;
    Storage storage(directory.root / "jobs.sqlite3");
    const auto failure = [](std::exception_ptr exception) { return describe_failure(exception).detail; };
    const auto rejected = [](Jobs::Error::Kind kind, auto action) {
        try { action(); CHECK(false); } catch (const Jobs::Error& error) { CHECK(error.kind == kind); }
    };
    std::string successful;
    {
        Jobs jobs(storage, failure);
        const auto wait = [&](const std::string& id) {
            const auto deadline = Clock::now() + 5s;
            while (Clock::now() < deadline) {
                auto job = jobs.get(id);
                if (Jobs::terminal(job)) return job;
                std::this_thread::sleep_for(1ms);
            }
            throw std::runtime_error("job did not terminate");
        };
        std::promise<void> entered;
        auto ready = entered.get_future();
        const auto running = jobs.submit("fixture.wait", {}, [&](const Jobs::Control& control) -> Json {
            entered.set_value();
            while (!control.cancelled) std::this_thread::sleep_for(1ms);
            control.checkpoint(); return {};
        });
        CHECK(ready.wait_for(5s) == std::future_status::ready);
        std::vector<std::string> queued;
        for (int i = 0; i < 16; ++i) queued.push_back(jobs.submit("fixture.queued", {}, [](const Jobs::Control&) { return Json::object(); }));
        rejected(Jobs::Error::conflict, [&] { jobs.erase(running); });
        rejected(Jobs::Error::conflict, [&] { jobs.erase(queued.front()); });
        CHECK(jobs.clear() == 0);
        const auto listing = jobs.list()["data"];
        CHECK(listing.size() == 17 && jobs.list(1, 0)["data"][0] == listing[0] && jobs.list(1, 1)["data"][0] == listing[1]);
        CHECK(jobs.list(100, 0, "running")["data"][0]["id"] == running);
        CHECK(jobs.list(100, 0, "queued", "fixture.queued")["data"].size() == 16);
        CHECK(jobs.list(100, 0, "running", "fixture.queued")["data"].empty());
        CHECK(jobs.list(100, 100)["data"].empty());
        rejected(Jobs::Error::invalid, [&] { jobs.list(0); });
        rejected(Jobs::Error::invalid, [&] { jobs.list(1001); });
        rejected(Jobs::Error::invalid, [&] { jobs.list(100, 0, "other"); });
        rejected(Jobs::Error::invalid, [&] { jobs.list(100, 0, {}, "a' OR 1=1"); });
        try { jobs.submit("fixture.full", {}, [](const Jobs::Control&) { return Json(); }); CHECK(false); }
        catch (const Jobs::Error& error) { CHECK(error.kind == Jobs::Error::full); }
        CHECK(jobs.cancel(queued.front())["status"] == "cancelled");
        CHECK(jobs.cancel(running)["cancel_requested"] == true);
        CHECK(wait(running)["status"] == "cancelled");
        for (const auto& id : queued) wait(id);
        successful = jobs.submit("fixture.progress", {}, [](const Jobs::Control& control) {
            for (int i = 0; i < 270; ++i) control.progress("loading");
            return Json{{"instance_id", "result"}};
        });
        const auto done = wait(successful);
        CHECK(done["status"] == "succeeded" && done["result"]["instance_id"] == "result");
        CHECK(jobs.cancel(successful)["status"] == "succeeded");
        try { jobs.events(successful, 0); CHECK(false); }
        catch (const Jobs::Error& error) { CHECK(error.kind == Jobs::Error::expired); }
        const auto cursor = done["last_sequence"].get<std::int64_t>();
        CHECK(jobs.events(successful, cursor - 256)["data"].size() == 256);
        CHECK(jobs.events(successful, cursor)["data"].empty());
        CHECK(jobs.events(successful, cursor - 1)["data"][0]["progress"] == 1);
        const auto failed = jobs.submit("fixture.fail", {}, [](const Jobs::Control&) -> Json {
            throw ModelError(ModelError::Kind::invalid, "unsupported_requirement", "unsupported context");
        });
        CHECK(wait(failed)["error"]["code"] == "unsupported_requirement");
        {
            auto access = storage.access();
            sqlite::execute(access.database, "CREATE TRIGGER reject_archive BEFORE UPDATE OF archived_at ON jobs "
                "BEGIN SELECT RAISE(ABORT,'archive failure'); END");
        }
        const auto count = jobs.list()["data"].size();
        try { jobs.erase(failed); CHECK(false); } catch (const sqlite::Error&) {}
        try { jobs.clear(); CHECK(false); } catch (const sqlite::Error&) {}
        CHECK(jobs.list()["data"].size() == count && jobs.get(failed)["status"] == "failed");
        { auto access = storage.access(); sqlite::execute(access.database, "DROP TRIGGER reject_archive"); }
        jobs.erase(failed);
        rejected(Jobs::Error::not_found, [&] { jobs.get(failed); });
        rejected(Jobs::Error::not_found, [&] { jobs.events(failed, 0); });
        rejected(Jobs::Error::not_found, [&] { jobs.cancel(failed); });
        rejected(Jobs::Error::not_found, [&] { jobs.erase(failed); });
        std::promise<void> stopping_entered;
        auto stopping_ready = stopping_entered.get_future();
        const auto stopping = jobs.submit("fixture.stop", {}, [&](const Jobs::Control& control) -> Json {
            stopping_entered.set_value();
            while (!control.cancelled) std::this_thread::sleep_for(1ms);
            control.checkpoint(); return {};
        });
        CHECK(stopping_ready.wait_for(5s) == std::future_status::ready);
        bool executed = false;
        const auto pending = jobs.submit("fixture.pending", {}, [&](const Jobs::Control&) { executed = true; return Json(); });
        jobs.request_stop();
        try { jobs.submit("fixture.rejected", {}, [](const Jobs::Control&) { return Json(); }); CHECK(false); }
        catch (const Jobs::Error& error) { CHECK(error.kind == Jobs::Error::stopping); }
        jobs.shutdown();
        CHECK(!executed && jobs.get(pending)["status"] == "cancelled");
        CHECK(jobs.get(stopping)["status"] == "cancelled");
    }
    const auto interrupted = normalize_uuid({});
    {
        auto access = storage.access();
        sqlite::Statement insert(access.database, "INSERT INTO jobs(id,kind,status,payload_json,progress,cancel_requested,created_at,updated_at) VALUES(?,'model.load','running','{}',0,0,'now','now')");
        insert.bind(1, interrupted); insert.step();
    }
    {
        Jobs restarted(storage, failure);
        CHECK(restarted.get(interrupted)["status"] == "interrupted");
        CHECK(restarted.get(successful)["status"] == "succeeded");
        {
            auto access = storage.access();
            sqlite::Statement lineage(access.database, "INSERT INTO artifact_lineage VALUES('fixture','workspace://model','model',?,'fixture.progress','[]','{}','{}','now')");
            lineage.bind(1, successful); lineage.step();
        }
        const auto count = restarted.list()["data"].size();
        CHECK(restarted.clear() == count && restarted.clear() == 0 && restarted.list()["data"].empty());
        rejected(Jobs::Error::not_found, [&] { restarted.get(interrupted); });
        rejected(Jobs::Error::not_found, [&] { restarted.get(successful); });
        auto access = storage.access();
        sqlite::Statement preserved(access.database, "SELECT COUNT(*) FROM jobs j JOIN job_events e ON e.job_id=j.id "
            "JOIN artifact_lineage a ON a.producer_job_id=j.id WHERE j.id=? AND j.archived_at IS NOT NULL");
        preserved.bind(1, successful); preserved.step(); CHECK(preserved.column(0) == "256");
    }
    {
        Jobs restarted(storage, failure);
        CHECK(restarted.list()["data"].empty());
        rejected(Jobs::Error::not_found, [&] { restarted.get(successful); });
    }
}

template<class Action> void session_error(const char* code, Action action) {
    try { action(); }
    catch (const Sessions::Error& error) { CHECK(error.code == code); return; }
    throw std::runtime_error(std::string("expected session error: ") + code);
}
Json reply_input(const char* prompt = "hello", std::int64_t revision = 0) {
    return {{"request_id", normalize_uuid({})}, {"expected_revision", revision},
        {"input", Json::array({{{"type", "text"}, {"text", prompt}}})}, {"stream", false}};
}
void session_checks() {
    Directory directory;
    Storage storage(directory.root / "sessions.sqlite3");
    fixture_model(directory.root / "model");
    write_file(directory.root / "model.runtime.json", R"({"chat":{"top_k":7,"temperature":0.4,"enable_thinking":false}})");
    ModelCatalog catalog({directory.root});
    EngineFactory factory; factory.backend = "fixture";
    factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
    factory.load = [](const fs::path&, const LoadOptions&) { return std::make_unique<FixtureEngine>(); };
    ModelInstances models(catalog, storage, factory);
    const auto loaded = models.load(catalog.list(true)[0].id);
    std::string preserved, interrupted;
    {
        Sessions sessions(storage, models);
        const auto created = sessions.create({{"model", loaded.id}, {"title", "original"}});
        const auto id = preserved = created.at("id").get<std::string>();
        const auto body = reply_input();
        auto reply = sessions.start(id, body);
        CHECK(reply->snapshot()["status"] == "running");
        CHECK(sessions.get(id)["revision"] == 1 && sessions.messages(id)["data"].size() == 1);
        session_error("response_in_progress", [&] { sessions.start(id, body); });
        session_error("response_in_progress", [&] { sessions.erase(id); });
        const auto result = reply->finish();
        CHECK(result["status"] == "completed" && result["output"][0]["text"] == "hi");
        CHECK(result["output_message_id"] == sessions.messages(id)["data"][1]["id"]);
        CHECK(result["settings"]["sampling"].empty());
        CHECK(result["performance"]["sampling"]["top_k"] == 7);
        CHECK(std::abs(result["performance"]["sampling"]["temperature"].get<double>() - 0.4) < 0.0001);
        CHECK(result["performance"]["sampling"]["enable_thinking"] == false);
        CHECK(sessions.get(id)["revision"] == 2 && sessions.get(id)["state"] == "idle");
        auto streamed = body; streamed["stream"] = true;
        CHECK(sessions.start(id, streamed)->finish() == result);
        auto changed = body; changed["input"][0]["text"] = "different";
        session_error("idempotency_conflict", [&] { sessions.start(id, changed); });
        session_error("revision_conflict", [&] { sessions.start(id, reply_input()); });
        CHECK(sessions.messages(id)["data"].size() == 2);
        const auto chained_id = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        for (int i = 0; i < 10; ++i) {
            auto first = sessions.start(chained_id, reply_input("hello", sessions.get(chained_id)["revision"]));
            auto finishing = std::async(std::launch::async, [&] { return first->finish(); });
            const auto deadline = Clock::now() + 5s;
            while (sessions.get(chained_id)["state"] == "processing" && Clock::now() < deadline) std::this_thread::yield();
            auto second = sessions.start(chained_id, reply_input("slow", sessions.get(chained_id)["revision"]));
            CHECK(finishing.get()["status"] == "completed");
            CHECK(sessions.cancel(chained_id)["status"] == "running");
            CHECK(second->finish()["status"] == "cancelled");
        }
        const auto original_messages = sessions.messages(id)["data"];
        auto forked = sessions.fork(id, {{"at_message_id", original_messages[0]["id"]}});
        CHECK(forked["revision"] == 1);
        CHECK(sessions.messages(forked["id"])["data"][0] == original_messages[0]);
        sessions.rewind(id, {{"expected_revision", 2}, {"at_message_id", original_messages[0]["id"]}, {"include_message", false}});
        CHECK(sessions.get(id)["revision"] == 0 && sessions.responses(id)["data"].empty());
        CHECK(sessions.messages(forked["id"])["data"].size() == 1);
        session_error("message_not_found", [&] { sessions.rewind(id, {{"expected_revision", 0}, {"at_message_id", original_messages[0]["id"]}}); });
        auto appended = sessions.append(id, {{"expected_revision", 0}, {"role", "user"}, {"parts", body["input"]}});
        CHECK(appended["session"]["revision"] == 1);
        auto archive = sessions.export_archive(id);
        const auto imported = sessions.import_archive(archive);
        CHECK(imported["messages_imported"] == 1 && imported["media_imported"] == 0);
        CHECK(sessions.messages(imported["session"]["id"])["data"][0]["parts"] == body["input"]);
        const auto before = sessions.list()["data"].size();
        archive["messages"].push_back({{"role", "invalid"}, {"parts", body["input"]}, {"created_at", "bad"}});
        session_error("invalid_request", [&] { sessions.import_archive(archive); });
        CHECK(sessions.list()["data"].size() == before);
        auto invalid_date = sessions.export_archive(id);
        invalid_date["messages"][0]["created_at"] = "2026-02-30T00:00:00Z";
        session_error("invalid_request", [&] { sessions.import_archive(invalid_date); });
        CHECK(sessions.list()["data"].size() == before);
        auto media = sessions.export_archive(id); media["media"] = Json::array({Json::object()});
        session_error("media_unavailable", [&] { sessions.import_archive(media); });
        const auto tool_session = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        auto tool_input = reply_input("tool");
        tool_input["tools"] = Json::array({{{"type", "function"}, {"function", {{"name", "lookup"}}}}});
        const auto tool_reply = sessions.start(tool_session, tool_input)->finish();
        CHECK(tool_reply["status"] == "completed" && tool_reply["output"][0]["type"] == "reasoning");
        CHECK(tool_reply["output"][2]["type"] == "tool_call" && tool_reply["output"][2]["arguments"].is_object());
        auto tool_result = reply_input("hello", 2); tool_result["input_role"] = "tool";
        tool_result["input"] = Json::array({{{"type", "tool_result"}, {"call_id", "call-1"}, {"result", "hello"}}});
        CHECK(sessions.start(tool_session, tool_result)->finish()["status"] == "completed");
        {
            auto access = storage.access();
            sqlite::Statement large(access.database, "UPDATE responses SET output_json=? WHERE session_id=?");
            large.bind(1, Json::array({{{"type", "text"}, {"text", std::string(9 * 1024 * 1024, 'x')}}}).dump());
            large.bind(2, tool_session); large.step();
        }
        session_error("result_limit", [&] { sessions.responses(tool_session); });
        CHECK(sessions.responses(tool_session, 1)["data"].size() == 1);
        auto preset = sessions.put_preset({{"name", "text"}, {"settings", Json::object()}});
        CHECK(preset["settings"]["sampling"].empty());
        session_error("generation_preset_conflict", [&] { sessions.put_preset({{"name", "text"}, {"settings", Json::object()}}); });
        preset = sessions.put_preset({{"name", "renamed"}, {"settings", {{"sampling", {{"top_k", 3}, {"temperature", 0}}}}}}, preset["id"]);
        CHECK(sessions.presets()["data"][0]["name"] == "renamed");
        CHECK(preset["settings"]["sampling"] == Json({{"top_k", 3}, {"temperature", 0}}));
        auto overridden = reply_input(); overridden["sampling"] = preset["settings"]["sampling"];
        const auto overridden_id = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        const auto explicit_result = sessions.start(overridden_id, overridden)->finish();
        CHECK(explicit_result["performance"]["sampling"]["top_k"] == 3);
        CHECK(explicit_result["performance"]["sampling"]["temperature"] == 0);
        CHECK(explicit_result["performance"]["sampling"]["enable_thinking"] == false);
        sessions.erase_preset(preset["id"]);
        CHECK(sessions.presets()["data"].empty());
        const auto cancelled_id = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        auto running = sessions.start(cancelled_id, reply_input("slow"));
        CHECK(sessions.cancel(cancelled_id)["status"] == "running");
        CHECK(running->finish()["status"] == "cancelled");
        CHECK(sessions.get(cancelled_id)["state"] == "interrupted");
        CHECK(sessions.messages(cancelled_id)["data"].size() == 1);
        const auto same_key_a = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        const auto same_key_b = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        const auto shared_key = reply_input("slow");
        auto request_a = sessions.start(same_key_a, shared_key);
        auto request_b = sessions.start(same_key_b, shared_key);
        CHECK(request_a->snapshot()["status"] == "running" && request_b->snapshot()["status"] == "running");
        sessions.cancel(same_key_a); CHECK(request_a->finish()["status"] == "cancelled");
        sessions.cancel(same_key_b); CHECK(request_b->finish()["status"] == "cancelled");
        const auto abandoned = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        sessions.start(abandoned, reply_input("slow")).reset();
        CHECK(sessions.responses(abandoned)["data"][0]["status"] == "cancelled");
        const auto sid = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        {
            auto access = storage.access();
            sqlite::execute(access.database, "CREATE TRIGGER fail_begin BEFORE INSERT ON responses BEGIN SELECT RAISE(ABORT,'begin failure'); END");
        }
        try { sessions.start(sid, reply_input()); CHECK(false); } catch (const sqlite::Error&) {}
        CHECK(sessions.messages(sid)["data"].empty() && sessions.get(sid)["revision"] == 0);
        {
            auto access = storage.access(); sqlite::execute(access.database, "DROP TRIGGER fail_begin");
        }
        auto unsaved = sessions.start(sid, reply_input());
        {
            auto access = storage.access();
            sqlite::execute(access.database, "CREATE TRIGGER fail_save BEFORE UPDATE ON responses WHEN NEW.status='completed' BEGIN SELECT RAISE(ABORT,'save failure'); END");
        }
        session_error("response_save_failed", [&] { unsaved->finish(); });
        CHECK(sessions.messages(sid)["data"].size() == 1);
        CHECK(sessions.responses(sid)["data"][0]["status"] == "running");
        interrupted = sid;
        {
            auto access = storage.access(); sqlite::execute(access.database, "DROP TRIGGER fail_save");
        }
        session_error("response_save_failed", [&] { unsaved->next(); });
        session_error("response_save_failed", [&] { unsaved->snapshot(); });
        try { unsaved->finish(); CHECK(false); }
        catch (const Sessions::Error& error) {
            CHECK(error.code == "response_save_failed" && error.details["execution_status"] == "completed");
            CHECK(error.details["saved"] == false);
        }
        CHECK(sessions.messages(sid)["data"].size() == 1);
        auto active_id = sessions.create({{"model", loaded.id}})["id"].get<std::string>();
        auto active = sessions.start(active_id, reply_input("slow"));
        sessions.request_stop();
        session_error("server_stopping", [&] { sessions.start(active_id, reply_input()); });
        CHECK(sessions.get(active_id)["state"] == "processing");
        sessions.shutdown();
        CHECK(active->snapshot()["status"] == "cancelled");
    }
    {
        Sessions recovered(storage, models);
        CHECK(recovered.get(preserved)["revision"] == 1);
        CHECK(recovered.get(interrupted)["state"] == "interrupted");
        CHECK(recovered.responses(interrupted)["data"][0]["error"]["code"] == "response_interrupted");
        CHECK(recovered.start(interrupted, reply_input("hello", 1))->finish()["status"] == "completed");
        const auto failed_id = recovered.create({{"model", loaded.id}})["id"].get<std::string>();
        CHECK(recovered.start(failed_id, reply_input("fail"))->finish()["status"] == "failed");
        CHECK(recovered.get(failed_id)["state"] == "error");
        CHECK(recovered.messages(failed_id)["data"].size() == 1);
    }
    models.shutdown();
}

void reply_load_checks() {
    for (int mode = 0; mode < 4; ++mode) {
        Directory directory; Storage storage(directory.root / "db");
        fixture_model(directory.root / "model");
        write_file(directory.root / "model.runtime.json", R"({"chat":{"top_k":7,"temperature":0.4}})");
        ModelCatalog catalog({directory.root});
        std::promise<void> entered, release;
        auto ready = entered.get_future(); auto gate = release.get_future().share();
        std::atomic<int> loads{0}; std::atomic<bool> fail{mode == 2};
        auto state = std::make_shared<FixtureState>();
        EngineFactory factory; factory.backend = "fixture";
        factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
        factory.load = [&](const fs::path&, const LoadOptions&) -> std::unique_ptr<Engine> {
            auto engine = std::make_unique<FixtureEngine>(state);
            if (++loads == 1) entered.set_value();
            gate.wait(); if (fail) throw std::runtime_error("fixture load failed");
            return engine;
        };
        ModelInstances models(catalog, storage, factory); models.set_aliases({{"model", "fast"}});
        Sessions sessions(storage, models);
        const auto a = sessions.create({{"model", "fast"}})["id"].get<std::string>();
        auto invalid = reply_input(); invalid["sampling"] = {{"top_k", 1025}};
        session_error("invalid_request_error", [&] { sessions.start(a, invalid); });
        CHECK(sessions.get(a)["revision"] == 0 && sessions.messages(a)["data"].empty() && loads == 0);
        const auto abandoned = sessions.create({{"model", "fast"}})["id"].get<std::string>();
        auto pending = sessions.start(abandoned, reply_input());
        sessions.cancel(abandoned);
        CHECK(pending->finish()["status"] == "cancelled" && pending->snapshot()["performance"].is_null() && loads == 0);
        const auto body = reply_input(); auto first = sessions.start(a, body);
        CHECK(loads == 0 && sessions.get(a)["runtime_instance_id"].is_null());
        auto one = std::async(std::launch::async, [&] { return first->finish(); });
        if (ready.wait_for(5s) != std::future_status::ready) { release.set_value(); CHECK(false); }
        const auto b = sessions.create({{"model", "fast"}})["id"].get<std::string>();
        auto second = sessions.start(b, reply_input());
        auto two = std::async(std::launch::async, [&] { return second->finish(); });
        const bool waiting = two.wait_for(100ms) == std::future_status::timeout;
        const bool processing = sessions.get(a)["state"] == "processing" && first->snapshot()["status"] == "running";
        session_error("response_in_progress", [&] { sessions.start(a, body); });
        session_error("response_in_progress", [&] { sessions.erase(a); });
        bool cancelled = false;
        if (mode == 1 || mode == 3) {
            if (mode == 1) { sessions.cancel(a); sessions.cancel(b); }
            else { sessions.request_stop(); models.request_stop(); }
            cancelled = two.wait_for(1s) == std::future_status::ready;
        }
        const bool owner_waits = one.wait_for(0ms) == std::future_status::timeout;
        release.set_value();
        const auto x = one.get(), y = two.get();
        CHECK(waiting && processing && owner_waits && loads == 1);
        const auto status = mode == 0 ? "completed" : mode == 2 ? "failed" : "cancelled";
        CHECK(x["status"] == status && y["status"] == status);
        if (mode == 0) {
            CHECK(state->requests == 2 && x["performance"]["sampling"]["top_k"] == 7);
            CHECK(sessions.get(a)["revision"] == 2 && !sessions.get(a)["runtime_instance_id"].is_null());
        } else {
            CHECK((mode == 2 || cancelled) && state->requests == 0);
            CHECK(x["usage"].is_null() && x["performance"].is_null() && sessions.messages(a)["data"].size() == 1);
        }
        if (mode != 3) {
            CHECK(sessions.start(a, body)->finish() == x && loads == 1);
            session_error("revision_conflict", [&] { sessions.start(a, reply_input()); });
        }
        if (mode == 2) {
            CHECK(models.list().empty() && state->alive == 0 && x["error"]["code"] == "request_start_failed");
            fail = false; CHECK(sessions.start(a, reply_input("hello", 1))->finish()["status"] == "completed" && loads == 2);
        }
        sessions.shutdown(); first.reset(); second.reset(); models.shutdown(); CHECK(state->alive == 0);
    }
}

struct Server {
    fs::path root = fs::temp_directory_path() / ("mfq-inference-" + normalize_uuid({}));
    std::unique_ptr<server_test::Process> process;
    int port = 0;
    Server(const fs::path& executable, bool fixture, const std::string& allocation_limit_gib = "8") {
        fs::create_directories(root);
        std::vector<std::string> arguments;
        if (fixture) arguments = {"--fixture-server", path_utf8(root / "data")};
        else arguments = {"--port", "0", "--data-dir", path_utf8(root / "data"), "--exit-on-stdin-close"};
        process = std::make_unique<server_test::Process>(executable, arguments, root / "server.log",
            std::map<std::string, std::string>{{"TYLOINFER_CUDA_VRAM_LIMIT_GIB", allocation_limit_gib},
                {"TYLOINFER_RUNTIME_PREFIX_CACHE_DIR", path_utf8(root / "prefix")}});
        const auto deadline = Clock::now() + 10s;
        while (!port && Clock::now() < deadline && process->running()) {
            std::istringstream lines(process->output()); std::string line;
            while (std::getline(lines, line)) {
                const auto event = Json::parse(line, nullptr, false);
                if (event.is_object() && event.value("event", "") == "listening") port = event.at("port");
            }
            std::this_thread::sleep_for(10ms);
        }
        if (!port) throw std::runtime_error(process->output());
    }
    ~Server() {
        if (process) { process->close_input(); try { process->wait(); } catch (...) {} }
        process.reset(); std::error_code ignored; fs::remove_all(root, ignored);
    }
    httplib::Result post(const std::string& path, const Json& body, const httplib::Headers& headers = {}) {
        httplib::Client client("127.0.0.1", port); client.set_read_timeout(60);
        return client.Post(path, headers, body.dump(), "application/json");
    }
    Json complete(const httplib::Result& response) {
        CHECK(response && response->status == 202);
        const auto accepted = Json::parse(response->body);
        CHECK(accepted.at("status") == "accepted");
        const auto path = response->get_header_value("Location");
        CHECK(path == "/api/v1/jobs/" + accepted.at("operation_id").get<std::string>());
        const auto deadline = Clock::now() + 60s;
        while (Clock::now() < deadline) {
            auto job = get(path.c_str());
            if (Jobs::terminal(job)) return job;
            std::this_thread::sleep_for(10ms);
        }
        throw std::runtime_error("operation did not terminate");
    }
    Json get(const char* path) {
        httplib::Client client("127.0.0.1", port);
        auto response = client.Get(path); CHECK(response && response->status == 200);
        return Json::parse(response->body);
    }
};

std::string register_checkpoint(Server& server, const fs::path& source, const char* name) {
    const auto path = server.root / name;
    fs::create_directories(path);
    for (const auto& file : fs::directory_iterator(source)) if (file.is_regular_file()) {
        std::error_code error;
        fs::create_hard_link(file.path(), path / file.path().filename(), error);
        if (error) fs::copy_file(file.path(), path / file.path().filename());
    }
    const auto response = server.post("/api/v1/models/directories/register", {{"path", path_utf8(path)}});
    CHECK(response && response->status == 200);
    return Json::parse(response->body).at("data").at(0).at("id");
}

void shutdown_checks(const fs::path& executable) {
    Server server(executable, true);
    const auto alpha = server.root / "alpha", beta = server.root / "beta";
    fixture_model(alpha); fixture_model(beta);
    auto registered = server.post("/api/v1/models/directories/register", {{"path", path_utf8(server.root)}});
    CHECK(registered && registered->status == 200);
    std::map<std::string, std::string> artifacts;
    const auto listing = Json::parse(registered->body);
    for (const auto& artifact : listing["data"]) artifacts.emplace(artifact["name"], artifact["id"]);
    const auto loaded = server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifacts.at("alpha")}}));
    CHECK(loaded["status"] == "succeeded");
    const auto created = server.post("/api/v1/sessions", {{"model", "alpha"}});
    CHECK(created && created->status == 201);
    const auto session_path = "/api/v1/sessions/" + Json::parse(created->body)["id"].get<std::string>();
    auto reply = std::async(std::launch::async, [&] { return server.post(session_path + "/responses", reply_input("slow")); });
    const auto deadline = Clock::now() + 5s;
    while (server.get(session_path.c_str())["state"] != "processing" && Clock::now() < deadline) std::this_thread::sleep_for(1ms);
    CHECK(server.get(session_path.c_str())["state"] == "processing");
    const auto message = [&](bool stream) {
        return server.post("/v1/messages", {{"model", "alpha"}, {"max_tokens", 32}, {"stream", stream},
            {"messages", {{{"role", "user"}, {"content", "slow"}}}}});
    };
    auto ordinary = std::async(std::launch::async, [&] { return message(false); });
    auto stream = std::async(std::launch::async, [&] { return message(true); });
    while (server.get("/api/v1/runtime/instances")["data"][0]["active_requests"] != 3 && Clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    CHECK(server.get("/api/v1/runtime/instances")["data"][0]["active_requests"] == 3);
    const auto loading = server.post("/api/v1/models/load", {{"artifact_id", artifacts.at("beta")}, {"requirements", {{"context_size", 15}}}});
    CHECK(loading && loading->status == 202);
    while (!fs::exists(beta / "entered") && Clock::now() < deadline) std::this_thread::sleep_for(1ms);
    CHECK(fs::exists(beta / "entered"));
    server.process->close_input();
    httplib::Result rejected;
    do {
        rejected = server.post("/api/v1/sessions", {{"model", "alpha"}});
        if (rejected && rejected->status == 503) break;
        std::this_thread::sleep_for(1ms);
    } while (Clock::now() < deadline);
    CHECK(rejected && rejected->status == 503 && Json::parse(rejected->body)["error"]["code"] == "server_stopping");
    const auto inference = server.post("/v1/chat/completions", {{"model", "alpha"}});
    CHECK(inference && inference->status == 503);
    CHECK(Json::parse(inference->body)["error"]["type"] == "server_stopping");
    const auto refused = server.post("/v1/messages", {{"model", "alpha"}});
    CHECK(refused && refused->status == 503 && Json::parse(refused->body)["error"]["type"] == "overloaded_error");
    CHECK(refused->get_header_value("Retry-After") == "1");
    CHECK(server.get("/health")["status"] == "ok");
    const auto operation = loading->get_header_value("Location");
    const auto cancelled = server.post(operation + "/cancel", Json::object());
    CHECK(cancelled && cancelled->status == 200);
    CHECK(reply.wait_for(2s) == std::future_status::ready);
    const auto response = reply.get();
    CHECK(response && response->status == 200 && Json::parse(response->body)["status"] == "cancelled");
    CHECK(ordinary.wait_for(2s) == std::future_status::ready && stream.wait_for(2s) == std::future_status::ready);
    const auto ended = ordinary.get(), streamed = stream.get();
    CHECK(ended && ended->status == 409 && Json::parse(ended->body)["error"]["type"] == "invalid_request_error");
    CHECK(streamed && streamed->status == 200 && streamed->body.find("event: error") != std::string::npos);
    CHECK(streamed->body.find("message_stop") == std::string::npos);
    CHECK(server.process->running());
    write_file(beta / "release", "");
    CHECK(server.process->wait() == 0);
}

Json message_events(const std::string& stream) {
    Json events = Json::array();
    std::istringstream lines(stream); std::string line, type;
    while (std::getline(lines, line)) {
        if (line.rfind("event: ", 0) == 0) type = line.substr(7);
        if (line.rfind("data: ", 0) == 0) {
            auto event = Json::parse(line.substr(6)); CHECK(event["type"] == type);
            events.push_back(std::move(event));
        }
    }
    return events;
}

void compat_checks(const fs::path& executable) {
    Server server(executable, true);
    const auto contract = server.get("/openapi.json");
    CHECK(contract["paths"].contains("/v1/messages") && contract["paths"].contains("/v1/models/{model_id}"));
    fixture_model(server.root / "alpha"); fixture_model(server.root / "beta");
    write_file(server.root / "alpha.runtime.json", R"({"chat":{"top_k":7,"temperature":0.4,"enable_thinking":false}})");
    const auto registered = server.post("/api/v1/models/directories/register", {{"path", path_utf8(server.root)}});
    CHECK(registered && registered->status == 200);
    const auto artifacts = Json::parse(registered->body)["data"];
    for (const auto& artifact : artifacts) CHECK(server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifact["id"]}}))["status"] == "succeeded");
    httplib::Client client("127.0.0.1", server.port);
    const httplib::Headers anthropic{{"Anthropic-Version", "2023-06-01"}};
    CHECK(server.get("/v1")["endpoints"][1] == "/v1/chat/completions");
    auto response = client.Get("/v1", anthropic);
    CHECK(response && Json::parse(response->body)["endpoints"][1] == "/v1/messages");
    CHECK(server.get("/v1/models")["object"] == "list");
    response = client.Get("/v1/models?limit=1", anthropic);
    CHECK(response && response->status == 200);
    auto page = Json::parse(response->body);
    CHECK(page["has_more"] == true && page["first_id"] == "alpha" && page["last_id"] == "alpha");
    response = client.Get("/v1/models?after_id=alpha", {{"X-Api-Key", "unused-local-key"}});
    page = Json::parse(response->body); CHECK(page["data"].size() == 1 && page["first_id"] == "beta" && page["has_more"] == false);
    response = client.Get("/v1/models?before_id=beta", anthropic);
    CHECK(response && Json::parse(response->body)["last_id"] == "alpha");
    CHECK(server.get("/v1/models/alpha")["type"] == "model");
    response = client.Get("/v1/models/missing");
    CHECK(response && response->status == 404 && Json::parse(response->body)["error"]["type"] == "not_found_error");
    for (const auto* query : {"limit=0", "limit=1001", "limit=true", "after_id=missing", "before_id="}) {
        response = client.Get(std::string("/v1/models?") + query, anthropic);
        CHECK(response && response->status == 400 && Json::parse(response->body)["type"] == "error");
    }
    const auto request_body = [](const char* prompt, bool stream = false) {
        return Json{{"model", "alpha"}, {"max_tokens", 32}, {"stream", stream},
            {"messages", Json::array({{{"role", "user"}, {"content", prompt}}})}};
    };
    const auto lookup = Json::array({{{"name", "lookup"}, {"input_schema", {{"type", "object"}}}}});
    auto body = request_body("inspect");
    response = server.post("/v1/messages", body);
    CHECK(response && response->status == 200);
    auto observed = Json::parse(Json::parse(response->body)["content"][0]["text"].get<std::string>());
    CHECK(observed["top_k"] == 7 && std::abs(observed["temperature"].get<double>() - 0.4) < 0.0001);
    CHECK(observed["enable_thinking"] == false);
    body["temperature"] = 0; body["enable_mtp"] = false; body["system"] = "system"; body["tools"] = lookup;
    body["messages"] = Json::array({
        {{"role", "assistant"}, {"content", Json::array({{{"type", "thinking"}, {"thinking", "plan"}, {"signature", ""}},
            {{"type", "tool_use"}, {"id", "call-1"}, {"name", "lookup"}, {"input", {{"x", 1}}}}})}},
        {{"role", "user"}, {"content", Json::array({{{"type", "tool_result"}, {"tool_use_id", "call-1"}, {"content", "failed lookup"}, {"is_error", true}},
            {{"type", "text"}, {"text", "inspect"}}})}}});
    response = server.post("/v1/messages", body);
    CHECK(response && response->status == 200);
    observed = Json::parse(Json::parse(response->body)["content"][0]["text"].get<std::string>());
    CHECK(observed["temperature"] == 0 && observed["enable_mtp"] == false && observed["tools"] == 1);
    CHECK(observed["messages"][0]["role"] == "system" && observed["messages"][1]["reasoning"] == "plan");
    CHECK(observed["messages"][1]["tools"] == 1 && observed["messages"][2]["role"] == "tool");
    CHECK(observed["messages"][2]["content"].get<std::string>().find("Tool execution failed") != std::string::npos);
    body["messages"][1]["content"][0]["content"] = Json::array();
    body["messages"][1]["content"][0]["is_error"] = false;
    body["thinking"] = {{"type", "adaptive"}};
    response = server.post("/v1/messages", body); CHECK(response && response->status == 200);
    observed = Json::parse(Json::parse(response->body)["content"][0]["text"].get<std::string>());
    CHECK(observed["enable_thinking"] == true && observed["messages"][2]["content"] == "");

    for (const auto* prompt : {"hello", "limit", "fragmented-tool"}) {
        body = request_body(prompt); body["tools"] = lookup;
        auto ordinary = server.post("/v1/messages", body); CHECK(ordinary && ordinary->status == 200);
        const auto result = Json::parse(ordinary->body);
        CHECK(result["id"].get<std::string>() == "msg_" + ordinary->get_header_value("request-id"));
        CHECK(result["stop_reason"] == (std::string(prompt) == "limit" ? "max_tokens" : std::string(prompt) == "fragmented-tool" ? "tool_use" : "end_turn"));
        body["stream"] = true;
        response = server.post("/v1/messages", body); CHECK(response && response->status == 200);
        CHECK(response->body.find("[DONE]") == std::string::npos);
        const auto events = message_events(response->body);
        CHECK(events.front()["type"] == "message_start" && events.back()["type"] == "message_stop");
        Json content = Json::array(); std::set<std::size_t> closed; std::map<std::size_t, std::string> arguments;
        int starts = 0, ends = 0, deltas = 0;
        for (const auto& event : events) {
            const auto kind = event["type"];
            if (kind == "message_start") ++starts;
            if (kind == "message_stop") ++ends;
            if (kind == "content_block_start") {
                CHECK(event["index"] == content.size()); content.push_back(event["content_block"]);
            } else if (kind == "content_block_delta") {
                const auto index = event["index"].get<std::size_t>(); CHECK(index < content.size() && !closed.count(index));
                const auto& delta = event["delta"]; const auto type = delta["type"];
                if (type == "input_json_delta") arguments[index] += delta["partial_json"].get<std::string>();
                else {
                    const std::string field = type == "text_delta" ? "text" : type == "thinking_delta" ? "thinking" : "signature";
                    content[index][field] = content[index].value(field, "") + delta[field].get<std::string>();
                }
            } else if (kind == "content_block_stop") {
                const auto index = event["index"].get<std::size_t>(); CHECK(closed.insert(index).second);
                if (content[index]["type"] == "tool_use") content[index]["input"] = Json::parse(arguments[index]);
            } else if (kind == "message_delta") {
                ++deltas; CHECK(event["delta"]["stop_reason"] == result["stop_reason"] && event["usage"] == result["usage"]);
            }
        }
        CHECK(starts == 1 && ends == 1 && deltas == 1 && closed.size() == content.size());
        CHECK(content == result["content"]);
        if (std::string(prompt) == "fragmented-tool") {
            CHECK(content.size() == 4 && content[2]["input"]["x"] == 1 && content[3]["input"]["x"] == 2);
            CHECK(result["usage"]["input_tokens"] == 12 && result["usage"]["output_tokens"] == 4);
        }
    }
    for (const auto& invalid : std::vector<Json>{
        {{"max_tokens", 0}}, {{"max_tokens", true}}, {{"max_tokens", 2147483648LL}}, {{"stream", "yes"}}, {{"temperature", 2}},
        {{"temperature", nullptr}}, {{"top_k", 4294967296LL}}, {{"top_k", nullptr}}, {{"top_p", nullptr}},
        {{"messages", Json::array({{{"role", "system"}, {"content", "invalid"}}})}}, {{"messages", Json::array({{{"role", "user"}, {"content", Json::array({1})}}})}},
        {{"tools", Json::array({{{"name", "lookup"}}})}}, {{"tool_choice", {{"type", "tool"}, {"name", "missing"}}}},
        {{"thinking", {{"type", "enabled"}, {"budget_tokens", 128}}}}, {{"stop_sequences", Json::array({""})}},
        {{"seed", -1}}, {{"unknown", 1}}}) {
        body = request_body("hello"); body.update(invalid);
        response = server.post("/v1/messages", body);
        CHECK(response && response->status == 400 && Json::parse(response->body)["error"]["type"] == "invalid_request_error");
    }
    body = request_body("hello"); body["stop_sequences"] = {"STOP"};
    response = server.post("/v1/messages", body);
    CHECK(response && response->status == 501 && Json::parse(response->body)["error"]["type"] == "api_error");
    body = request_body("hello"); body["max_tokens"] = 4096; body["thinking"] = {{"type", "enabled"}, {"budget_tokens", 1024}};
    response = server.post("/v1/messages", body); CHECK(response && response->status == 501);
    response = server.post("/v1/messages/count_tokens", request_body("hello")); CHECK(response && response->status == 501);
    response = server.post("/v1/messages", request_body("hello"), {{"Anthropic-Version", "future"}}); CHECK(response && response->status == 400);
    body = request_body("hello"); body["model"] = "missing";
    response = server.post("/v1/messages", body); CHECK(response && response->status == 404 && Json::parse(response->body)["type"] == "error");
    response = client.Post("/v1/messages", "not-json", "application/json");
    CHECK(response && response->status == 400 && Json::parse(response->body)["type"] == "error");
    response = client.Post("/v1/messages", "{}", "text/plain");
    CHECK(response && response->status == 415 && Json::parse(response->body)["type"] == "error");
    for (bool stream : {false, true}) {
        body = request_body("bad-tool", stream); body["tools"] = lookup;
        response = server.post("/v1/messages", body); CHECK(response);
        if (stream) {
            const auto events = message_events(response->body);
            CHECK(events.back()["type"] == "error" && response->body.find("message_stop") == std::string::npos);
        } else CHECK(response->status == 502 && Json::parse(response->body)["error"]["type"] == "api_error");
    }
    const auto instance = server.get("/api/v1/runtime/instances")["data"];
    std::string alpha;
    for (const auto& item : instance) if (item["model"] == "alpha") alpha = item.at("instance_id").get<std::string>();
    CHECK(!alpha.empty());
    const auto start = [&](std::promise<std::string>& signal, bool disconnect = false) {
        return std::async(std::launch::async, [&, disconnect] {
            httplib::Client peer("127.0.0.1", server.port); peer.set_read_timeout(30);
            httplib::Request request; request.method = "POST"; request.path = "/v1/messages";
            request.body = request_body("slow", true).dump(); request.set_header("Content-Type", "application/json");
            request.response_handler = [&](const httplib::Response& headers) { signal.set_value(headers.get_header_value("X-Request-ID")); return true; };
            std::string output;
            request.content_receiver = [&](const char* data, std::size_t size, std::uint64_t, std::uint64_t) { output.append(data, size); return !disconnect; };
            peer.send(request); return output;
        });
    };
    std::array<std::promise<std::string>, 3> signals;
    std::vector<std::future<std::string>> streams; std::vector<std::string> ids;
    for (auto& signal : signals) {
        streams.push_back(start(signal)); auto ready = signal.get_future(); CHECK(ready.wait_for(5s) == std::future_status::ready); ids.push_back(ready.get());
    }
    response = server.post("/v1/messages", request_body("hello"));
    CHECK(response && response->status == 429 && Json::parse(response->body)["error"]["type"] == "rate_limit_error");
    CHECK(response->get_header_value("Retry-After") == "1");
    for (const auto& id : ids) {
        response = server.post("/api/v1/inference/requests/" + id + "/cancel", {{"instance_id", alpha}});
        CHECK(response && response->status == 200);
    }
    for (auto& stream : streams) {
        CHECK(stream.wait_for(5s) == std::future_status::ready); const auto output = stream.get();
        CHECK(message_events(output).back()["type"] == "error" && output.find("message_stop") == std::string::npos);
    }
    std::promise<std::string> signal;
    auto dropped = start(signal, true); CHECK(dropped.wait_for(5s) == std::future_status::ready); dropped.get();
    const auto deadline = Clock::now() + 5s;
    bool idle = false;
    do {
        idle = true;
        const auto current = server.get("/api/v1/runtime/instances");
        for (const auto& item : current["data"]) if (item["active_requests"] != 0) idle = false;
        if (!idle) std::this_thread::sleep_for(1ms);
    } while (!idle && Clock::now() < deadline);
    CHECK(idle);
    response = client.Put("/api/v1/auth/policy", R"({"enabled":true,"api_key":"root"})", "application/json"); CHECK(response && response->status == 200);
    response = server.post("/v1/messages", request_body("hello"));
    CHECK(response && response->status == 401 && Json::parse(response->body)["error"]["type"] == "authentication_error");
    const httplib::Headers root{{"X-Api-Key", "root"}, {"Anthropic-Version", "2023-06-01"}};
    response = server.post("/v1/messages", request_body("hello"), root); CHECK(response && response->status == 200);
    response = server.post("/api/v1/auth/keys", {{"name", "models-only"}, {"scopes", {"models"}}}, {{"Authorization", "Bearer root"}});
    CHECK(response && response->status == 201);
    response = server.post("/v1/messages", request_body("hello"), {{"X-Api-Key", Json::parse(response->body)["token"].get<std::string>()}});
    CHECK(response && response->status == 403 && Json::parse(response->body)["error"]["type"] == "permission_error");
    body = {{"model", "alpha"}, {"messages", {{{"role", "user"}, {"content", "hello"}}}}, {"n", 2}};
    response = server.post("/v1/chat/completions", body, {{"Authorization", "Bearer root"}, {"Anthropic-Version", "2023-06-01"}});
    CHECK(response && response->status == 400);
    CHECK(Json::parse(response->body)["error"]["param"] == "n" && !Json::parse(response->body).contains("type"));
    for (bool stream : {false, true}) {
        body = request_body("fail", stream); body["model"] = stream ? "alpha" : "beta";
        response = server.post("/v1/messages", body, root); CHECK(response);
        const auto error = stream ? message_events(response->body).back() : Json::parse(response->body);
        CHECK(response->status == (stream ? 200 : 500));
        CHECK(error["type"] == "error" && error["error"]["message"] == "fixture execution failure");
        CHECK(response->body.find("message_stop") == std::string::npos);
    }
}

void jobs_http_checks(const fs::path& executable) {
    Server server(executable, true);
    fixture_model(server.root / "source");
    const auto artifact = register_checkpoint(server, server.root / "source", "chat");
    httplib::Client client("127.0.0.1", server.port);
    auto accepted = server.post("/api/v1/models/load", {{"artifact_id", artifact}});
    const auto loaded = server.complete(accepted);
    CHECK(loaded["status"] == "succeeded");
    const auto path = "/api/v1/jobs/" + loaded["id"].get<std::string>();
    auto response = client.Delete(path); CHECK(response && response->status == 204);
    CHECK(server.get("/api/v1/runtime/instances")["data"].size() == 1 && fs::exists(server.root / "chat" / "model.safetensors"));
    for (const auto& suffix : {"", "/events", "/events/stream"}) {
        response = client.Get(path + suffix); CHECK(response && response->status == 404);
    }
    response = client.Post(path + "/cancel"); CHECK(response && response->status == 404);
    response = client.Delete(path); CHECK(response && response->status == 404);
    CHECK(server.complete(server.post("/api/v1/models/unload", {{"instance_id", loaded["result"]["instance_id"]}}))["status"] == "succeeded");
    auto running = server.post("/api/v1/models/load", {{"artifact_id", artifact}, {"requirements", {{"context_size", 15}}}});
    CHECK(running && running->status == 202);
    const auto active = running->get_header_value("Location");
    const auto deadline = Clock::now() + 5s;
    while (!fs::exists(server.root / "chat" / "entered") && Clock::now() < deadline) std::this_thread::sleep_for(1ms);
    CHECK(fs::exists(server.root / "chat" / "entered"));
    auto queued = server.post("/api/v1/models/load", {{"artifact_id", artifact}, {"requirements", {{"context_size", 13}}}});
    CHECK(queued && queued->status == 202);
    for (const auto& id : {active, queued->get_header_value("Location")}) {
        response = client.Delete(id); CHECK(response && response->status == 409 && Json::parse(response->body)["error"]["code"] == "job_state_conflict");
    }
    CHECK(server.get("/api/v1/jobs?status=running&kind=model.load")["data"].size() == 1);
    CHECK(server.get("/api/v1/jobs?status=queued&kind=model.load")["data"].size() == 1);
    const auto all = server.get("/api/v1/jobs")["data"];
    CHECK(all.size() == 3 && server.get("/api/v1/jobs?limit=1&offset=1")["data"][0] == all[1]);
    for (const auto* query : {"limit=0", "limit=1001", "offset=-1", "offset=9223372036854775808", "status=other", "kind=Bad", "limit=", "offset=", "status=", "kind="}) {
        response = client.Get(std::string("/api/v1/jobs?") + query); CHECK(response && response->status == 400);
    }
    response = client.Delete("/api/v1/jobs/completed"); CHECK(response && response->status == 204);
    CHECK(server.get("/api/v1/jobs")["data"].size() == 2);
    response = client.Post(active + "/cancel"); CHECK(response && response->status == 200 && Json::parse(response->body)["status"] == "cancelling");
    response = client.Delete(active); CHECK(response && response->status == 409);
    response = client.Delete("/api/v1/jobs/completed"); CHECK(response && response->status == 204);
    CHECK(server.get("/api/v1/jobs")["data"].size() == 2);
    write_file(server.root / "chat" / "release", "");
    CHECK(server.complete(running)["status"] == "cancelled");
    CHECK(server.complete(queued)["status"] == "failed");
    response = client.Put("/api/v1/auth/policy", R"({"enabled":true,"api_key":"root"})", "application/json");
    CHECK(response && response->status == 200);
    response = client.Delete("/api/v1/jobs/completed"); CHECK(response && response->status == 401);
    for (const auto* scope : {"inference", "jobs"}) {
        const auto key = server.post("/api/v1/auth/keys", {{"name", scope}, {"scopes", {scope}}}, {{"Authorization", "Bearer root"}});
        CHECK(key && key->status == 201);
        const httplib::Headers token{{"Authorization", "Bearer " + Json::parse(key->body)["token"].get<std::string>()}};
        response = client.Delete(queued->get_header_value("Location"), token);
        CHECK(response && response->status == (std::string_view(scope) == "jobs" ? 204 : 403));
        response = client.Delete("/api/v1/jobs/completed", token);
        CHECK(response && response->status == (std::string_view(scope) == "jobs" ? 204 : 403));
        if (std::string_view(scope) == "jobs") {
            response = client.Get("/api/v1/jobs", token); CHECK(response && response->status == 200 && Json::parse(response->body)["data"].empty());
        }
    }
    CHECK(fs::exists(server.root / "chat" / "model.safetensors"));
}

void load_http_checks(const fs::path& executable) {
    {
        Server server(executable, true);
        fixture_model(server.root / "source");
        register_checkpoint(server, server.root / "source", "chat");
        write_file(server.root / "chat.runtime.json", R"({"chat":{"top_k":7,"temperature":0.4,"enable_thinking":false}})");
        httplib::Client client("127.0.0.1", server.port);
        auto response = client.Put("/api/v1/runtime/model-aliases", R"({"aliases":{"chat":"fast"}})", "application/json");
        CHECK(response && response->status == 200);
        CHECK(server.get("/v1/models")["data"][0]["id"] == "fast");
        CHECK(server.get("/v1/models/fast")["id"] == "fast");
        const auto path = server.root / "chat";
        Json body{{"model", "fast"}, {"messages", {{{"role", "user"}, {"content", "inspect"}}}}, {"max_tokens", 8}};
        for (const auto* route : {"/v1/chat/completions", "/v1/messages"}) {
            for (const auto& fields : std::vector<Json>{{{"max_tokens", 0}}, {{"top_k", 4294967296LL}}, {{"stream", "yes"}}}) {
                auto invalid = body; invalid.update(fields);
                response = server.post(route, invalid); CHECK(response && response->status == 400);
            }
        }
        auto invalid = body; invalid["mtp_max_draft_tokens"] = 4294967297LL;
        response = server.post("/v1/chat/completions", invalid); CHECK(response && response->status == 400);
        invalid = body; invalid["stop_sequences"] = {"stop"};
        response = server.post("/v1/messages", invalid); CHECK(response && response->status == 501);
        CHECK(!fs::exists(path / "loads") && server.get("/api/v1/runtime/instances")["data"].empty());
        int loads = 0;
        for (const auto* route : {"/v1/chat/completions", "/v1/messages"}) for (bool stream : {false, true}) {
            body["stream"] = stream;
            response = server.post(route, body); CHECK(response && response->status == 200);
            if (!stream) {
                const auto result = Json::parse(response->body);
                const auto content = result.contains("choices") ? result["choices"][0]["message"]["content"] : result["content"][0]["text"];
                const auto observed = Json::parse(content.get<std::string>());
                CHECK(result["model"] == "fast" && observed["top_k"] == 7);
                CHECK(std::abs(observed["temperature"].get<double>() - 0.4) < 0.0001);
            } else CHECK(response->get_header_value("Content-Type").find("text/event-stream") == 0 && response->body.find("fast") != std::string::npos);
            CHECK(fs::file_size(path / "loads") == static_cast<std::uintmax_t>(++loads));
            const auto instances = server.get("/api/v1/runtime/instances")["data"];
            CHECK(instances.size() == 1);
            CHECK(server.complete(server.post("/api/v1/models/unload", {{"instance_id", instances[0]["instance_id"]}}))["status"] == "succeeded");
        }
    }
    for (int mode = 0; mode < 3; ++mode) {
        Server server(executable, true);
        fixture_model(server.root / "source");
        register_checkpoint(server, server.root / "source", "chat");
        const auto path = server.root / "chat";
        write_file(path / "wait", "");
        if (mode == 1) write_file(path / "fail-load", "");
        Json body{{"model", "chat"}, {"messages", {{{"role", "user"}, {"content", "hello"}}}}, {"max_tokens", 8}, {"stream", true}};
        auto first = std::async(std::launch::async, [&] {
            httplib::Client client("127.0.0.1", server.port);
            client.set_read_timeout(mode == 2 ? 200ms : 10s);
            return client.Post("/v1/chat/completions", body.dump(), "application/json");
        });
        const auto deadline = Clock::now() + 5s;
        while (!fs::exists(path / "entered") && Clock::now() < deadline) std::this_thread::sleep_for(1ms);
        const bool entered = fs::exists(path / "entered");
        auto second = std::async(std::launch::async, [&] { return server.post("/v1/messages", body); });
        const bool waiting = second.wait_for(100ms) == std::future_status::timeout;
        std::future<httplib::Result> third, fourth;
        bool limited = true;
        if (mode == 0) {
            third = std::async(std::launch::async, [&] { return server.post("/v1/messages", body); });
            fourth = std::async(std::launch::async, [&] { return server.post("/v1/chat/completions", body); });
            limited = third.wait_for(1s) == std::future_status::ready || fourth.wait_for(1s) == std::future_status::ready;
        }
        const bool empty = server.get("/api/v1/runtime/instances")["data"].empty();
        const bool listed = server.get("/v1/models")["data"].size() == 1;
        const bool disconnected = mode != 2 || first.wait_for(2s) == std::future_status::ready;
        write_file(path / "release", "");
        auto a = first.get(), b = second.get();
        CHECK(entered && waiting && limited && empty && listed && disconnected && fs::file_size(path / "loads") == 1);
        if (mode == 0) {
            auto c = third.get(), d = fourth.get();
            CHECK(c && d && ((c->status == 429 && d->status == 200) || (c->status == 200 && d->status == 429)));
        }
        if (mode == 1) {
            CHECK(a && b && a->status == 500 && b->status == 500);
            CHECK(a->get_header_value("Content-Type").find("application/json") == 0);
            CHECK(b->get_header_value("Content-Type").find("application/json") == 0 && Json::parse(b->body)["type"] == "error");
            CHECK(server.get("/api/v1/runtime/instances")["data"].empty());
            fs::remove(path / "fail-load");
            auto retry = server.post("/v1/messages", body); CHECK(retry && retry->status == 200);
            CHECK(fs::file_size(path / "loads") == 2);
        } else {
            CHECK(mode == 2 ? !a : a && a->status == 200);
            CHECK(b && b->status == 200 && server.get("/api/v1/runtime/instances")["data"].size() == 1);
        }
    }
}

void reply_load_http(const fs::path& executable) {
    for (int mode = 0; mode < 5; ++mode) {
        Server server(executable, true);
        fixture_model(server.root / "source"); register_checkpoint(server, server.root / "source", "chat");
        const auto model = server.root / "chat";
        write_file(model / "wait", "");
        if (mode == 4) write_file(model / "fail-load", "");
        const auto created = server.post("/api/v1/sessions", {{"model", "chat"}});
        CHECK(created && created->status == 201);
        const auto path = "/api/v1/sessions/" + Json::parse(created->body)["id"].get<std::string>();
        auto body = reply_input(); body["stream"] = mode == 1 || mode == 3 || mode == 4;
        const bool disconnect = mode == 2 || mode == 3;
        auto running = std::async(std::launch::async, [&] {
            httplib::Client client("127.0.0.1", server.port);
            client.set_read_timeout(disconnect ? 200ms : 10s);
            return client.Post(path + "/responses", body.dump(), "application/json");
        });
        const auto deadline = Clock::now() + 5s;
        while (!fs::exists(model / "entered") && Clock::now() < deadline) std::this_thread::sleep_for(1ms);
        const bool entered = fs::exists(model / "entered");
        const auto pending = server.get(path.c_str());
        bool stopped = true;
        if (mode < 2) {
            auto cancel = server.post(path + "/responses/cancel", Json::object());
            stopped = cancel && cancel->status == 200 && Json::parse(cancel->body)["status"] == "running";
        } else if (disconnect) stopped = running.wait_for(2s) == std::future_status::ready;
        write_file(model / "release", "");
        const auto result = running.get();
        CHECK(entered && stopped && pending["state"] == "processing" && pending["runtime_instance_id"].is_null());
        if (disconnect) CHECK(!result);
        else {
            CHECK(result && result->status == 200);
            if (mode == 0) CHECK(Json::parse(result->body)["status"] == "cancelled");
            else CHECK(result->body.find(mode == 4 ? "event: error" : "event: response.interrupted") != std::string::npos);
        }
        Json saved; const auto finished = Clock::now() + 5s;
        do {
            saved = server.get((path + "/responses").c_str())["data"][0];
            if (saved["status"] != "running") break;
            std::this_thread::sleep_for(1ms);
        } while (Clock::now() < finished);
        CHECK(saved["status"] == (mode == 4 ? "failed" : "cancelled") && saved["performance"].is_null());
        const auto instances = server.get("/api/v1/runtime/instances")["data"];
        if (mode == 4) CHECK(instances.empty());
        else {
            CHECK(instances.size() == 1);
            CHECK(server.complete(server.post("/api/v1/models/unload", {{"instance_id", instances[0]["instance_id"]}}))["status"] == "succeeded");
        }
        body["stream"] = false;
        auto replay = server.post(path + "/responses", body);
        CHECK(replay && replay->status == 200 && Json::parse(replay->body) == saved);
        CHECK(fs::file_size(model / "loads") == 1 && server.get("/api/v1/runtime/instances")["data"].empty());
    }
}

void names_http_checks(const fs::path& executable) {
    Server server(executable, true);
    CHECK(server.get("/openapi.json")["paths"].contains("/api/v1/runtime/model-aliases"));
    fixture_model(server.root / "alpha");
    const auto artifact = register_checkpoint(server, server.root / "alpha", "beta");
    const auto loaded = server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifact}}));
    CHECK(loaded["status"] == "succeeded");
    httplib::Client client("127.0.0.1", server.port);
    const auto configure = [&](const Json& body, const httplib::Headers& headers = {}) {
        return client.Put("/api/v1/runtime/model-aliases", headers, body.dump(), "application/json");
    };
    auto response = configure({{"aliases", {{"beta", "fast"}}}}); CHECK(response && response->status == 200);
    CHECK(server.get("/api/v1/runtime/model-aliases")["aliases"]["beta"] == "fast");
    CHECK(server.get("/v1/models")["data"][0]["id"] == "fast");
    response = client.Get("/v1/models", {{"Anthropic-Version", "2023-06-01"}});
    CHECK(response && Json::parse(response->body)["first_id"] == "fast");
    CHECK(server.get("/v1/models/fast")["id"] == "fast");
    for (const auto* path : {"/v1/chat/completions", "/v1/messages"}) for (bool stream : {false, true}) {
        response = server.post(path, {{"model", "fast"}, {"messages", {{{"role", "user"}, {"content", "hello"}}}},
            {"max_tokens", 8}, {"stream", stream}});
        CHECK(response && response->status == 200);
        if (!stream) CHECK(Json::parse(response->body)["model"] == "fast");
        else {
            std::istringstream lines(response->body); std::string line; int found = 0;
            while (std::getline(lines, line)) if (line.rfind("data: ", 0) == 0 && line != "data: [DONE]") {
                const auto event = Json::parse(line.substr(6));
                if (event.contains("model")) { CHECK(event["model"] == "fast"); ++found; }
                if (event.contains("message")) { CHECK(event["message"]["model"] == "fast"); ++found; }
            }
            CHECK(found > 0);
        }
    }
    for (const auto& body : std::vector<Json>{Json::object(), Json::array(), {{"aliases", nullptr}}, {{"aliases", {1}}},
            {{"aliases", {{"beta", 1}}}}, {{"aliases", {{"beta", "next"}}}, {"unknown", true}}}) {
        response = configure(body); CHECK(response && response->status == 400);
        CHECK(server.get("/api/v1/runtime/model-aliases")["aliases"]["beta"] == "fast");
    }
    response = configure({{"aliases", {{"beta", "team/fast"}}}}); CHECK(response && response->status == 200);
    CHECK(server.get("/v1/models/team%2Ffast")["id"] == "team/fast");
    response = client.Put("/api/v1/auth/policy", R"({"enabled":true,"api_key":"root"})", "application/json");
    CHECK(response && response->status == 200);
    response = configure({{"aliases", Json::object()}}); CHECK(response && response->status == 401);
    response = server.post("/api/v1/auth/keys", {{"name", "infer"}, {"scopes", {"inference"}}}, {{"Authorization", "Bearer root"}});
    CHECK(response && response->status == 201);
    const httplib::Headers token{{"Authorization", "Bearer " + Json::parse(response->body)["token"].get<std::string>()}};
    response = client.Get("/api/v1/runtime/model-aliases", token); CHECK(response && response->status == 200);
    response = configure({{"aliases", Json::object()}}, token); CHECK(response && response->status == 403);
    response = configure({{"aliases", {{"beta", "next"}}}}, {{"Authorization", "Bearer root"}});
    CHECK(response && response->status == 200 && Json::parse(response->body)["aliases"]["beta"] == "next");
    server.process->close_input(); CHECK(server.process->wait() == 0); server.process.reset();
    Storage storage(server.root / "data" / "server.sqlite3");
    ModelCatalog catalog({server.root / "data" / "models"}); ModelInstances restored(catalog, storage, {});
    CHECK(restored.aliases().at("beta") == "next");
}

void session_http_checks(const fs::path& executable) {
    Server server(executable, true);
    const auto contract = server.get("/openapi.json");
    CHECK(contract["paths"].contains("/api/v1/sessions/{session_id}/responses"));
    CHECK(contract["components"]["schemas"].contains("SessionResponse"));
    fixture_model(server.root / "model");
    const auto artifact = register_checkpoint(server, server.root / "model", "chat");
    CHECK(server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifact}}))["status"] == "succeeded");
    auto create = server.post("/api/v1/sessions", {{"model", "chat"}});
    CHECK(create && create->status == 201);
    const auto id = Json::parse(create->body)["id"].get<std::string>();
    const auto path = "/api/v1/sessions/" + id;
    const auto input = reply_input();
    auto result = server.post(path + "/responses", input);
    CHECK(result && result->status == 200 && Json::parse(result->body)["status"] == "completed");
    CHECK(server.get((path + "/messages").c_str())["data"].size() == 2);
    auto streaming = reply_input("hello", 2); streaming["stream"] = true;
    result = server.post(path + "/responses", streaming);
    CHECK(result && result->status == 200);
    const auto stream_data = result->body;
    CHECK(stream_data.find("event: response.completed") != std::string::npos);
    CHECK(stream_data.find("event: response.text.delta") != std::string::npos);
    std::istringstream lines(stream_data); std::string line; std::size_t sequence = 0, terminal = 0;
    while (std::getline(lines, line)) if (line.rfind("data: ", 0) == 0) {
        const auto event = Json::parse(line.substr(6));
        CHECK(event["session_id"] == id && event["sequence"] == sequence++);
        if (event["payload"]["type"] == "response.completed") ++terminal;
    }
    CHECK(terminal == 1 && sequence >= 4);
    const auto replay = server.post(path + "/responses", streaming);
    CHECK(replay && replay->status == 200 && replay->body.find("response.completed") != std::string::npos);
    CHECK(server.get((path + "/messages").c_str())["data"].size() == 4);
    auto slow = reply_input("slow", 4); slow["stream"] = true;
    httplib::Client client("127.0.0.1", server.port);
    httplib::Request request; request.method = "POST"; request.path = path + "/responses";
    request.set_header("Content-Type", "application/json"); request.body = slow.dump();
    request.content_receiver = [](const char*, std::size_t, std::uint64_t, std::uint64_t) { return false; };
    client.send(request);
    const auto deadline = Clock::now() + 5s;
    while (Clock::now() < deadline && server.get(path.c_str())["state"] == "processing") std::this_thread::sleep_for(10ms);
    CHECK(server.get(path.c_str())["state"] == "interrupted");
    const auto responses = server.get((path + "/responses").c_str())["data"];
    CHECK(std::any_of(responses.begin(), responses.end(), [](const Json& value) { return value.at("status") == "cancelled"; }));
    result = server.post("/api/v1/presets", {{"name", "chat"}, {"settings", Json::object()}});
    CHECK(result && result->status == 201);
    CHECK(server.get("/api/v1/presets")["data"].size() == 1);
    result = client.Get("/api/v1/sessions?limit=-1"); CHECK(result && result->status == 400);
    result = server.post(path + "/messages", {{"expected_revision", false}, {"role", "user"}, {"parts", input["input"]}});
    CHECK(result && result->status == 400);
    result = server.post(path + "/messages", {{"expected_revision", 0}, {"role", "user"}, {"parts", input["input"]}});
    CHECK(result && result->status == 409 && Json::parse(result->body)["error"]["details"]["actual_revision"] == 5);
    server.process->close_input(); CHECK(server.process->wait() == 0);
    server.process.reset();
    Storage reopened(server.root / "data" / "server.sqlite3");
    ModelCatalog catalog({server.root / "data" / "models"}); ModelInstances models(catalog, reopened, {});
    Sessions restored(reopened, models);
    CHECK(restored.messages(id)["data"].size() == 5 && restored.presets()["data"].size() == 1);
    CHECK(restored.start(id, input)->finish()["status"] == "completed");
}

Json fresh_instance(Server& server, const std::string& id) {
    const auto after = Clock::now() + 2ms;
    const auto deadline = after + 5s;
    Json snapshot;
    while (Clock::now() < deadline) {
        const auto before_read = Clock::now();
        const auto instances = server.get("/api/v1/runtime/instances").at("data");
        for (const auto& instance : instances) if (instance.at("instance_id") == id) {
            snapshot = instance;
            const auto age = instance.at("metrics_age_ms").get<std::int64_t>();
            CHECK(age >= 0);
            if (before_read - std::chrono::milliseconds(age) >= after) return instance;
        }
        std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error("native metrics did not refresh: " + snapshot.dump());
}

double byte_metric(const Json& instance, const char* name) {
    for (const auto& metric : instance.at("metrics")) if (metric.at(0) == name) {
        const auto value = metric.at(1).get<double>();
        CHECK(std::isfinite(value) && value >= 0);
        return value;
    }
    throw std::runtime_error(std::string("native resource metric is missing: ") + name);
}

void cuda_resource_checks(const fs::path& executable, const fs::path& model) {
    const auto load = [](Server& server, const std::string& artifact) {
        return server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifact}, {"requirements", {{"context_size", 512}}}}));
    };
    const auto rejected = [](const Json& job) {
        CHECK(job.at("status") == "failed" && job.at("result").is_null());
        CHECK(job.at("error").at("code") == "resource_exhausted");
        CHECK(job.at("error").at("retryable") == true);
    };
    {
        Server server(executable, false, "0.0625");
        const auto artifact = register_checkpoint(server, model, "limited");
        for (int attempt = 0; attempt < 2; ++attempt) {
            rejected(load(server, artifact));
            CHECK(server.get("/api/v1/runtime/instances").at("data").empty());
            CHECK(server.get("/health").at("status") == "ok");
        }
        server.process->close_input(); CHECK(server.process->wait() == 0);
    }
    Server server(executable, false, "2");
    const auto alpha_artifact = register_checkpoint(server, model, "alpha");
    std::vector<std::string> rejected_artifacts;
    for (const auto* name : {"beta", "gamma", "delta"}) rejected_artifacts.push_back(register_checkpoint(server, model, name));
    auto loaded = load(server, alpha_artifact);
    if (loaded.at("status") != "succeeded") throw std::runtime_error(loaded.dump());
    auto id = loaded.at("result").at("instance_id").get<std::string>();
    const auto bounded = [&](const Json& instance) {
        CHECK(instance.at("backend") == "cuda" && instance.at("state") == "loaded");
        CHECK(instance.at("active_requests") == 0);
        const auto allocated = byte_metric(instance, "cuda_allocated_bytes");
        const auto reserved = byte_metric(instance, "cuda_reserved_bytes");
        CHECK(byte_metric(instance, "resident_weight_bytes") > 0);
        CHECK(allocated >= byte_metric(instance, "resident_weight_bytes") && allocated <= 2.0 * (1ULL << 30));
        CHECK(reserved >= allocated);
        CHECK(byte_metric(instance, "resident_memory_used_bytes") == reserved);
        CHECK(byte_metric(instance, "device_free_bytes") <= byte_metric(instance, "device_total_bytes"));
    };
    const auto baseline = fresh_instance(server, id);
    bounded(baseline);
    const auto before = byte_metric(baseline, "cuda_allocated_bytes");
    for (const auto& artifact : rejected_artifacts) {
        rejected(load(server, artifact));
        const auto instances = server.get("/api/v1/runtime/instances").at("data");
        CHECK(instances.size() == 1 && instances.at(0).at("instance_id") == id);
        const auto reclaimed = fresh_instance(server, id);
        bounded(reclaimed);
        if (byte_metric(reclaimed, "cuda_allocated_bytes") != before)
            throw std::runtime_error("rejected load retained tracked CUDA allocations: " + reclaimed.dump());
    }
    const auto generate = [&] {
        const auto response = server.post("/v1/chat/completions", {{"model", id},
            {"messages", Json::array({{{"role", "user"}, {"content", "Say hello."}}})},
            {"max_tokens", 16}, {"temperature", 0}, {"enable_thinking", false}});
        CHECK(response && response->status == 200);
        CHECK(Json::parse(response->body).at("usage").at("completion_tokens") > 0);
    };
    generate(); bounded(fresh_instance(server, id));
    CHECK(server.complete(server.post("/api/v1/models/unload", {{"instance_id", id}})).at("status") == "succeeded");
    CHECK(server.get("/api/v1/runtime/instances").at("data").empty());
    loaded = load(server, rejected_artifacts.front());
    if (loaded.at("status") != "succeeded") throw std::runtime_error(loaded.dump());
    id = loaded.at("result").at("instance_id");
    bounded(fresh_instance(server, id)); generate();
    CHECK(server.complete(server.post("/api/v1/models/unload", {{"instance_id", id}})).at("status") == "succeeded");
    server.process->close_input(); CHECK(server.process->wait() == 0);
    std::cout << "CUDA resource checks passed: 64 MiB refusal; 2 GiB limit; three failed loads reclaimed "
        << "to " << before << " tracked bytes; survivor generation and replacement reload succeeded\n";
}

void multi_model_checks(const fs::path& executable, const fs::path& real_model) {
    const bool fixture = real_model.empty();
    Server server(executable, fixture);
    for (const auto* name : {"alpha", "beta"}) {
        if (!fixture) { register_checkpoint(server, real_model, name); continue; }
        const auto path = server.root / name;
        fixture_model(path);
        const auto response = server.post("/api/v1/models/directories/register", {{"path", path_utf8(path)}});
        CHECK(response && response->status == 200);
    }
    const auto load = [&](const char* model, int context) {
        std::string artifact;
        const auto entries = server.get("/api/v1/models");
        for (const auto& entry : entries["data"]) if (entry["name"] == model) artifact = entry["id"];
        CHECK(!artifact.empty());
        const auto job = server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifact}, {"requirements", {{"context_size", context}}}}));
        if (job["status"] != "succeeded") throw std::runtime_error(job.dump() + server.process->output());
        return job.at("result");
    };
    auto alpha = load("alpha", 512);
    const auto beta = load("beta", 2048);
    CHECK(server.get("/api/v1/runtime/instances")["data"].size() == 2);
    CHECK(server.get("/v1/models")["data"].size() == 2);
    if (!fixture) CHECK(alpha["context_size"] == 512 && beta["context_size"] == 2048);
    const auto input = [](const std::string& model, const std::string& prompt, bool stream = false) {
        return Json{{"model", model}, {"messages", Json::array({{{"role", "user"}, {"content", prompt}}})},
            {"max_tokens", stream ? 256 : 16}, {"temperature", 0}, {"enable_thinking", false}, {"stream", stream}};
    };
    const auto start = [&](const std::string& model, std::promise<std::string>& ready, int tokens = 256) {
        auto body = input(model, fixture ? "slow" : "Print the integers from 1 to 10000, one per line. Do not skip numbers.", true);
        body["max_tokens"] = tokens;
        return std::async(std::launch::async, [&, body = std::move(body)] {
            httplib::Client client("127.0.0.1", server.port); client.set_read_timeout(30);
            httplib::Request request;
            request.method = "POST"; request.path = "/v1/chat/completions"; request.body = body.dump();
            request.set_header("Content-Type", "application/json");
            request.response_handler = [&](const httplib::Response& response) { ready.set_value(response.get_header_value("X-Request-ID")); return true; };
            std::string output;
            request.content_receiver = [&](const char* data, std::size_t size, std::uint64_t, std::uint64_t) { output.append(data, size); return true; };
            client.send(request);
            return output;
        });
    };
    std::promise<std::string> alpha_ready, beta_ready;
    auto first = start(alpha.at("instance_id"), alpha_ready), second = start(beta.at("artifact_id"), beta_ready, 1024);
    auto first_id = alpha_ready.get_future(), second_id = beta_ready.get_future();
    CHECK(first_id.wait_for(10s) == std::future_status::ready && second_id.wait_for(10s) == std::future_status::ready);
    const auto id_a = first_id.get(), id_b = second_id.get();
    CHECK(!id_a.empty() && !id_b.empty());
    auto response = server.post("/api/v1/inference/requests/" + id_a + "/cancel", {{"instance_id", alpha.at("instance_id")}});
    CHECK(response && response->status == 200 && Json::parse(response->body)["cancel_requested"] == true);
    CHECK(first.wait_for(10s) == std::future_status::ready);
    const auto first_output = first.get();
    CHECK(first_output.find("request_cancelled") != std::string::npos && first_output.find("[DONE]") == std::string::npos);
    CHECK(second.wait_for(0ms) == std::future_status::timeout);

    std::string oversized;
    for (int i = 0; i < 1024; ++i) oversized += "hello ";
    response = server.post("/v1/chat/completions", input("alpha", fixture ? "fail" : oversized));
    CHECK(response && response->status == (fixture ? 500 : 400));
    CHECK(second.wait_for(0ms) == std::future_status::timeout);
    response = server.post("/api/v1/models/unload", {{"instance_id", alpha.at("instance_id")}, {"force", true}});
    CHECK(server.complete(response)["status"] == "succeeded");
    const auto remaining = server.get("/api/v1/runtime/instances")["data"];
    CHECK(remaining.size() == 1 && remaining[0]["instance_id"] == beta["instance_id"] && remaining[0]["active_requests"] == 1);
    response = server.post("/api/v1/inference/requests/" + id_b + "/cancel", {{"instance_id", beta.at("instance_id")}});
    CHECK(response && response->status == 200 && Json::parse(response->body)["cancel_requested"] == true);
    CHECK(second.wait_for(10s) == std::future_status::ready);
    const auto second_output = second.get();
    CHECK(second_output.find("request_cancelled") != std::string::npos && second_output.find("[DONE]") == std::string::npos);

    alpha = load("alpha", 256);
    auto query = [&](const std::string& model) {
        const auto response = server.post("/v1/chat/completions", input(model, "Say hello."));
        CHECK(response && response->status == 200);
        CHECK(!Json::parse(response->body)["choices"][0]["message"]["content"].get<std::string>().empty());
    };
    auto parallel = std::async(std::launch::async, [&] { query("alpha"); });
    query("beta"); parallel.get();
    std::promise<std::string> exiting_a, exiting_b;
    first = start("alpha", exiting_a); second = start("beta", exiting_b);
    CHECK(exiting_a.get_future().wait_for(10s) == std::future_status::ready);
    CHECK(exiting_b.get_future().wait_for(10s) == std::future_status::ready);
    server.process->close_input(); CHECK(server.process->wait() == 0);
    CHECK(first.wait_for(10s) == std::future_status::ready && second.wait_for(10s) == std::future_status::ready);
    first.get(); second.get();
}

void run_checks(const fs::path& executable, const fs::path& real_model) {
    const bool fixture = real_model.empty();
    Server server(executable, fixture);
    const auto model_path = fixture ? server.root / "model" : real_model;
    if (fixture) { fixture_model(model_path); tokenizer_checks(model_path); }
    auto registration = server.post("/api/v1/models/directories/register", {{"path", path_utf8(model_path)}});
    CHECK(registration && registration->status == 200);
    const auto artifact = Json::parse(registration->body).at("data").at(0);
    CHECK(artifact.at("loadable"));
    CHECK(artifact.at("error").is_null() && artifact.at("support_error").is_null());
    const auto model = artifact.at("name").get<std::string>();
    CHECK(server.get("/v1/models")["data"].size() == 1);
    const auto artifact_id = artifact.at("id").get<std::string>();
    auto response = server.post("/api/v1/models/load", {{"artifact_id", artifact_id}, {"requirements", {{"context_size", false}}}});
    CHECK(response && response->status == 400);
    for (const auto& option : std::vector<Json>{
            {{"requirements", {{"context_size", 0}}}}, {{"requirements", {{"context_size", nullptr}}}},
            {{"requirements", {{"context_size", 2147483648LL}}}}, {{"requirements", nullptr}},
            {{"prefill_chunk_size", 1}}, {{"continuous_batching", 1}},
            {{"moe_gpu_cache_gb", 1}}, {{"gpu_layers", 1}}, {{"unknown", 1}},
            {{"requirements", {{"gpu_layers", 1}}}}, {{"policy", {{"pin", true}, {"idle_ttl_seconds", 1}}}}}) {
        auto invalid = option; invalid["artifact_id"] = artifact_id;
        response = server.post("/api/v1/models/load", invalid);
        CHECK(response && response->status == 400);
        const auto error = Json::parse(response->body).at("error");
        CHECK(error.contains("retryable") && error["details"].is_object());
    }
    CHECK(server.get("/api/v1/runtime/instances")["data"].empty());
    if (fixture) {
        response = server.post("/api/v1/models/load", {{"artifact_id", artifact_id}, {"requirements", {{"context_size", 13}}}});
        CHECK(server.complete(response)["status"] == "failed");
        CHECK(server.get("/api/v1/runtime/instances")["data"].empty());
        response = server.post("/api/v1/models/load", {{"artifact_id", artifact_id}, {"requirements", {{"context_size", 14}}}});
        CHECK(response && response->status == 202);
        const auto operation = Json::parse(response->body)["operation_id"].get<std::string>();
        const auto operation_path = "/api/v1/jobs/" + operation;
        const auto deadline = Clock::now() + 5s;
        while (server.get(operation_path.c_str())["progress_data"].value("phase", "") != "loading" && Clock::now() < deadline)
            std::this_thread::sleep_for(1ms);
        const auto cancellation = server.post(operation_path + "/cancel", Json::object());
        CHECK(cancellation && cancellation->status == 200 && Json::parse(cancellation->body)["cancel_requested"] == true);
        CHECK(server.complete(response)["status"] == "cancelled");
        CHECK(server.get("/api/v1/runtime/instances")["data"].empty());
    }
    auto load = [&] {
        auto job = server.complete(server.post("/api/v1/models/load", {{"artifact_id", artifact_id}, {"requirements", {{"context_size", 512}}}}));
        if (job["status"] != "succeeded") throw std::runtime_error(job.dump() + server.process->output());
        return job.at("result").at("instance_id").get<std::string>();
    };
    auto instance = load();
    const auto jobs = server.get("/api/v1/jobs").at("data");
    CHECK(!jobs.empty());
    const auto job_id = jobs[0]["id"].get<std::string>();
    httplib::Client event_client("127.0.0.1", server.port);
    const auto event_stream = event_client.Get("/api/v1/jobs/" + job_id + "/events/stream");
    CHECK(event_stream && event_stream->status == 200 && event_stream->body.find("id: 1\n") != std::string::npos);
    CHECK(event_stream->body.find("succeeded") != std::string::npos);
    const auto caps = server.get(("/api/v1/models/" + artifact_id + "/capabilities").c_str());
    CHECK(caps["supported"] == true && caps["load"]["cancellation"] == "after_return");
    CHECK(server.get("/v1/models")["data"].at(0).at("id") == model);
    CHECK(fresh_instance(server, instance).at("metrics_age_ms").get<std::int64_t>() >= 0);
    response = server.post("/api/v1/models/load", {{"artifact_id", artifact_id}});
    CHECK(server.complete(response)["error"]["code"] == "model_already_loaded");
    Json body{{"model", model}, {"messages", Json::array({{{"role", "user"}, {"content", "Say hello."}}})},
        {"max_tokens", 16}, {"temperature", 0}, {"enable_thinking", false}};
    response = server.post("/v1/chat/completions", body);
    if (!response || response->status != 200) throw std::runtime_error(response ? response->body + server.process->output() : server.process->output());
    const auto result = Json::parse(response->body);
    CHECK(!result["choices"][0]["message"]["content"].get<std::string>().empty());
    CHECK(result["usage"]["completion_tokens"] > 0);
    std::cout << "completion: " << result["choices"][0]["message"]["content"] << '\n';
    body["stream"] = true; body["stream_options"] = {{"include_usage", true}};
    response = server.post("/v1/chat/completions", body);
    CHECK(response && response->status == 200 && response->body.find("data: [DONE]\n\n") != std::string::npos);
    CHECK(response->body.find("\"completion_tokens\"") != std::string::npos);
    CHECK(response->body.find("\"role\":\"assistant\"") != std::string::npos);

    body["messages"][0]["content"] = fixture ? "slow" : "Write a long detailed essay about astronomy.";
    body["max_tokens"] = fixture ? 4096 : 128;
    auto start_stream = [&](std::promise<std::string>& ready, bool disconnect = false) {
        return std::async(std::launch::async, [&, disconnect] {
            httplib::Client client("127.0.0.1", server.port); client.set_read_timeout(30);
            httplib::Request request;
            request.method = "POST"; request.path = "/v1/chat/completions"; request.body = body.dump();
            request.set_header("Content-Type", "application/json");
            request.response_handler = [&](const httplib::Response& headers) { ready.set_value(headers.get_header_value("X-Request-ID")); return true; };
            std::string output;
            request.content_receiver = [&](const char* data, std::size_t size, std::uint64_t, std::uint64_t) { output.append(data, size); return !disconnect; };
            client.send(request);
            return output;
        });
    };
    std::promise<std::string> ready;
    auto streaming = start_stream(ready); auto ready_id = ready.get_future();
    CHECK(ready_id.wait_for(10s) == std::future_status::ready);
    const auto id = ready_id.get(); CHECK(!id.empty());
    response = server.post("/api/v1/models/unload", {{"instance_id", instance}});
    CHECK(server.complete(response)["error"]["code"] == "model_busy");
    response = server.post("/api/v1/inference/requests/" + id + "/cancel", {{"instance_id", instance}});
    CHECK(response && response->status == 200 && Json::parse(response->body).at("cancel_requested"));
    CHECK(streaming.wait_for(10s) == std::future_status::ready);
    const auto cancelled_stream = streaming.get();
    CHECK(cancelled_stream.find("request_cancelled") != std::string::npos);
    CHECK(cancelled_stream.find("[DONE]") == std::string::npos);
    std::promise<std::string> disconnected;
    auto dropped = start_stream(disconnected, true); CHECK(dropped.wait_for(10s) == std::future_status::ready); dropped.get();
    const auto deadline = Clock::now() + 5s;
    while (server.get("/api/v1/runtime/instances")["data"][0]["active_requests"] != 0 && Clock::now() < deadline)
        std::this_thread::sleep_for(20ms);
    CHECK(server.get("/api/v1/runtime/instances")["data"][0]["active_requests"] == 0);
    if (fixture) {
        std::array<std::promise<std::string>, 3> saturated;
        std::vector<std::future<std::string>> streams;
        std::vector<std::string> ids;
        for (auto& signal : saturated) {
            streams.push_back(start_stream(signal));
            auto started = signal.get_future();
            CHECK(started.wait_for(10s) == std::future_status::ready);
            ids.push_back(started.get());
        }
        response = server.post("/v1/chat/completions", body);
        CHECK(response && response->status == 429);
        for (const auto& request_id : ids) {
            response = server.post("/api/v1/inference/requests/" + request_id + "/cancel", {{"instance_id", instance}});
            CHECK(response && response->status == 200);
        }
        for (auto& stream : streams) { CHECK(stream.wait_for(10s) == std::future_status::ready); stream.get(); }
    }
    std::promise<std::string> forced;
    streaming = start_stream(forced); CHECK(forced.get_future().wait_for(10s) == std::future_status::ready);
    response = server.post("/api/v1/models/unload", {{"instance_id", instance}, {"force", true}});
    CHECK(server.complete(response)["status"] == "succeeded");
    CHECK(streaming.wait_for(10s) == std::future_status::ready); streaming.get();
    CHECK(server.get("/api/v1/runtime/instances")["data"].empty());
    instance = load();
    if (fixture) {
        body["messages"][0]["content"] = "fail";
        response = server.post("/v1/chat/completions", body);
        CHECK(response && response->body.find("fixture execution failure") != std::string::npos);
        const auto failed = server.get("/api/v1/runtime/instances")["data"][0];
        CHECK(failed["state"] == "failed");
        response = server.post("/api/v1/models/unload", {{"instance_id", instance}, {"force", true}});
        CHECK(server.complete(response)["status"] == "succeeded");
        instance = load(); body["messages"][0]["content"] = "slow";
    }
    std::promise<std::string> exiting;
    streaming = start_stream(exiting); CHECK(exiting.get_future().wait_for(10s) == std::future_status::ready);
    server.process->close_input(); CHECK(server.process->wait() == 0);
    CHECK(streaming.wait_for(10s) == std::future_status::ready); streaming.get();
}
}

int main(int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--fixture-server") {
            Config config; config.port = 0; config.data_dir = fs::u8path(argv[2]); config.exit_on_stdin_close = true;
            EngineFactory factory;
            factory.backend = "fixture";
            factory.unavailable_reason = [](const mfq::ModelSource&) { return std::string(); };
            factory.load = [](const fs::path& path, const LoadOptions& request) -> std::unique_ptr<Engine> {
                std::ofstream(path / "loads", std::ios::app).put('1');
                auto engine = std::make_unique<FixtureEngine>();
                if (request.context_size) engine->context_size = request.context_size;
                if (request.context_size == 14) std::this_thread::sleep_for(200ms);
                if (request.context_size == 15 || fs::exists(path / "wait")) {
                    write_file(path / "entered", "");
                    const auto deadline = Clock::now() + 10s;
                    while (!fs::exists(path / "release") && Clock::now() < deadline) std::this_thread::sleep_for(1ms);
                    CHECK(fs::exists(path / "release"));
                }
                if (request.context_size == 13 || fs::exists(path / "fail-load")) throw std::runtime_error("fixture load failure after allocation");
                return engine;
            };
            std::atomic<bool> stop{false}; return run(config, stop, factory);
        }
        if (argc == 2 && std::string(argv[1]) == "--sessions") {
            session_checks();
            session_http_checks(fs::absolute(fs::u8path(argv[0])));
            shutdown_checks(fs::absolute(fs::u8path(argv[0])));
            std::cout << "native session checks passed\n";
            return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--compatibility") {
            compat_checks(fs::absolute(fs::u8path(argv[0])));
            std::cout << "native compatibility checks passed\n"; return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--names") {
            names_checks(); names_http_checks(fs::absolute(fs::u8path(argv[0])));
            std::cout << "native model name checks passed\n"; return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--load") {
            acquire_checks(); reply_load_checks(); load_http_checks(fs::absolute(fs::u8path(argv[0])));
            reply_load_http(fs::absolute(fs::u8path(argv[0])));
            std::cout << "native on-demand load checks passed\n"; return 0;
        }
        if (argc == 2 && std::string(argv[1]) == "--jobs") {
            jobs_checks(); jobs_http_checks(fs::absolute(fs::u8path(argv[0])));
            std::cout << "native job checks passed\n"; return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--cuda-resources") {
            cuda_resource_checks(fs::absolute(fs::u8path(argv[2])), fs::u8path(argv[3]));
            return 0;
        }
        if (argc != 1 && argc != 3) throw std::runtime_error("usage: server-inference-test [SERVER MODEL | --cuda-resources SERVER MODEL]");
        if (argc == 1) { module_checks(); multi_instance_checks(); residency_checks(); }
        run_checks(fs::absolute(fs::u8path(argc == 3 ? argv[1] : argv[0])), argc == 3 ? fs::u8path(argv[2]) : fs::path{});
        multi_model_checks(fs::absolute(fs::u8path(argc == 3 ? argv[1] : argv[0])), argc == 3 ? fs::u8path(argv[2]) : fs::path{});
        std::cout << "native inference checks passed\n";
    } catch (const std::exception& failure) { std::cerr << failure.what() << '\n'; return 1; }
}
