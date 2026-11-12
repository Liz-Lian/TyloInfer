#include "diagnostics/text.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/causal_model.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/weights/loader.h"
#include "nlohmann/json.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using tyloi::Tensor;
namespace ops = tyloi::ops;
using tyloi::models::Input;
using tyloi::models::ForwardOptions;

struct Work {
    tyloi::cuda::Program& program;
    std::vector<Tensor> tensors;
    std::optional<TextSessionState> snapshot;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "text_check_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "text_check_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) {
        tensors.push_back(value);
        return value;
    }
    void reset(tyloi::models::State& state, std::int64_t batch) {
        program.drain();
        tensors.clear();
        state.reset(batch);
    }
};

nlohmann::json json_tensor(const Tensor& tensor) {
    std::vector<std::int64_t> shape;
    for (std::int64_t i = 0; i < tensor.dim(); ++i) shape.push_back(tensor.size(i));
    return {{"shape", shape}, {"data", ops::floats(tensor)}};
}

const tyloi::ModelSource& source(tyloi::cuda::Program& program) {
    auto loader = program.loader();
    const auto* model = dynamic_cast<const tyloi::ModelSource*>(&loader.source);
    if (!model) throw std::runtime_error("text check requires a model source");
    return *model;
}

Tensor ids_for(Work& work, const std::vector<std::int64_t>& ids, tyloi::Device device) {
    return work.keep(ops::reshape(work.keep(ops::tensor(ids, device)), {1, std::int64_t(ids.size())}));
}

Tensor slice(Work& work, const Tensor& value, std::int64_t begin, std::int64_t count) {
    return work.keep(ops::slice(value, 1, begin, begin + count));
}

Tensor hidden(Work& work, tyloi::models::CausalModel& model, tyloi::models::State& state,
    const Tensor& ids, const ForwardOptions& options, const Tensor& positions = {}, std::int64_t confirmed = 0) {
    Input input;
    input.ids = ids;
    input.positions = positions;
    input.confirmed = confirmed;
    return work.keep(model.forward(std::move(input), state, options));
}

Tensor logits(Work& work, tyloi::models::CausalModel& model, const Tensor& hidden_states) {
    return work.keep(model.logits(hidden_states, true));
}

Tensor forward(Work& work, tyloi::models::CausalModel& model, tyloi::models::State& state,
    const Tensor& ids, const ForwardOptions& options, const Tensor& positions = {}, std::int64_t confirmed = 0) {
    return logits(work, model, hidden(work, model, state, ids, options, positions, confirmed));
}

Tensor joined(Work& work, const std::vector<Tensor>& pieces) {
    return work.keep(ops::cat(pieces, 1));
}

void transactions(Work& work, tyloi::models::CausalModel& model, tyloi::models::State& state,
    const Tensor& ids, const ForwardOptions& options, nlohmann::json& result, const char* family) {
    for (bool accept : {false, true}) {
        const std::string name = accept ? "committed" : "rejected";
        work.reset(state, 1);
        (void)forward(work, model, state, slice(work, ids, 0, 2), options);
        auto verify = hidden(work, model, state, slice(work, ids, 2, 2), options, {}, 1);
        result[name + "_verify"] = json_tensor(logits(work, model, verify));
        if (accept) state.commit_speculative();
        else state.rollback_speculative(0);
        if (state.position() != (accept ? 4 : 3))
            throw std::runtime_error(std::string(family) + " transaction cache position mismatch");
        result[name] = json_tensor(forward(work, model, state, slice(work, ids, 5, 1), options));
        work.reset(state, 1);
        (void)forward(work, model, state, slice(work, ids, 0, 2), options);
        (void)forward(work, model, state, slice(work, ids, 2, accept ? 2 : 1), options);
        result[name + "_reference"] = json_tensor(forward(work, model, state, slice(work, ids, 5, 1), options));
    }
}
}

int run_flash_next_check(tyloi::cuda::Program& program) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    auto& model = program.model();
    auto& state = program.state();
    const auto graph = tyloi::models::graph(source(program));
    const bool multi_axis = graph.backbone == "qwen4_exp";
    if ((graph.backbone != "glm5_next" && !multi_axis) || model.info().vocab_size < 8 || model.info().max_context < 16)
        throw std::runtime_error("Flash-Next diagnostic requires a Flash-Next text graph, vocab>=8 and context>=16");
    const auto device = executor.device();
    const auto mode = executor.forward_mode(device.index);
    const ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, device.index};
    auto ids = ids_for(work, {1, 2, 3, 4, 5, 6, 7}, device);
    nlohmann::json result;
    work.reset(state, 1);
    result["full"] = json_tensor(forward(work, model, state, ids, options));
    work.reset(state, 1);
    std::vector<Tensor> pieces;
    for (const auto [begin, count] : std::array<std::pair<int64_t, int64_t>, 3>{{{0,2}, {2,1}, {3,4}}})
        pieces.push_back(forward(work, model, state, slice(work, ids, begin, count), options));
    result["chunked"] = json_tensor(joined(work, pieces));
    transactions(work, model, state, ids, options, result, "Flash-Next");
    work.reset(state, 1);
    result["reset"] = json_tensor(forward(work, model, state, ids, options));
    if (multi_axis) {
        std::vector<int64_t> axis;
        for (int offset : {0, 2, 4}) for (int i = 0; i < 7; ++i) axis.push_back(i + offset);
        auto positions = work.keep(ops::reshape(work.keep(ops::tensor(axis, device)), {3, 7}));
        work.reset(state, 1);
        result["axis_full"] = json_tensor(forward(work, model, state, ids, options, positions));
        std::vector<int64_t> axis4;
        for (int i = 0; i < 7; ++i) axis4.push_back(i + 11);
        axis4.insert(axis4.end(), axis.begin(), axis.end());
        auto positions4 = work.keep(ops::reshape(work.keep(ops::tensor(axis4, device)), {4, 7}));
        work.reset(state, 1);
        pieces.clear();
        for (const auto [begin, count] : std::array<std::pair<int64_t, int64_t>, 3>{{{0,2}, {2,1}, {3,4}}}) {
            auto part = work.keep(ops::slice(positions4, -1, begin, begin + count));
            pieces.push_back(forward(work, model, state, slice(work, ids, begin, count), options, part));
        }
        result["axis_chunked"] = json_tensor(joined(work, pieces));
        work.reset(state, 1);
        auto repeated = work.keep(ops::repeat(ids, {2, 1}));
        result["batch"] = json_tensor(forward(work, model, state, repeated, options));
        result["batch_reset"] = json_tensor(forward(work, model, state, ids, options));
        work.reset(state, 1);
        auto last_hidden = hidden(work, model, state, ids, options);
        result["last"] = json_tensor(logits(work, model, work.keep(ops::select(last_hidden, 1, -1))));
    }
    result["architecture"] = graph.backbone;
    std::cout << "flash_next_check " << result.dump() << '\n';
    return 0;
}

int run_nemotron_h_check(tyloi::cuda::Program& program) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    auto& model = program.model();
    auto& state = program.state();
    const auto graph = tyloi::models::graph(source(program));
    if (graph.backbone != "nemotron_h")
        throw std::runtime_error("--check-nemotron-h requires a Nemotron-H causal LM");
    if (model.info().vocab_size < 8 || model.info().max_context < 16)
        throw std::runtime_error("Nemotron-H diagnostic requires vocab>=8 and context>=16");
    const auto device = executor.device();
    const auto mode = executor.forward_mode(device.index);
    const ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, device.index};
    auto ids = ids_for(work, {1, 2, 3, 4, 5, 6, 7}, device);
    nlohmann::json result;
    work.reset(state, 1);
    auto full_hidden = hidden(work, model, state, ids, options);
    result["full_hidden"] = json_tensor(full_hidden);
    result["full"] = json_tensor(logits(work, model, work.keep(ops::clone(full_hidden))));
    work.reset(state, 1);
    std::vector<Tensor> pieces;
    for (const auto [begin, count] : std::array<std::pair<int64_t, int64_t>, 3>{{{0,2}, {2,1}, {3,4}}})
        pieces.push_back(forward(work, model, state, slice(work, ids, begin, count), options));
    result["chunked"] = json_tensor(joined(work, pieces));
    work.reset(state, 1);
    (void)forward(work, model, state, slice(work, ids, 0, 5), options);
    pieces.clear();
    for (int64_t i = 5; i < 7; ++i) pieces.push_back(forward(work, model, state, slice(work, ids, i, 1), options));
    result["decode"] = json_tensor(joined(work, pieces));
    work.reset(state, 1);
    (void)forward(work, model, state, slice(work, ids, 0, 4), options);
    work.snapshot.emplace(state.capture_text_session_state({1, 2, 3, 4}));
    auto continued = forward(work, model, state, slice(work, ids, 4, 1), options);
    program.drain();
    state.restore_text_session_state(*work.snapshot);
    result["restored"] = json_tensor(forward(work, model, state, slice(work, ids, 4, 1), options));
    result["continued"] = json_tensor(continued);
    transactions(work, model, state, ids, options, result, "Nemotron-H");
    work.reset(state, 2);
    result["batch"] = json_tensor(forward(work, model, state, work.keep(ops::repeat(ids, {2, 1})), options));
    work.reset(state, 1);
    result["reset"] = json_tensor(forward(work, model, state, ids, options));
    result["architecture"] = graph.backbone;
    std::cout << "nemotron_h_check " << result.dump() << '\n';
    return 0;
}
