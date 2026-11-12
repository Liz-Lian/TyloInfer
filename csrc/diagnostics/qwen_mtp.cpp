#include "diagnostics/qwen_mtp.h"

#include "engine/program.h"
#include "engine/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"
#include "engine/ops.h"
#include "engine/requests/output.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {

struct Work {
    Program& program;
    std::vector<Tensor> tensors;
    std::vector<std::vector<float>> floats;
    std::vector<std::vector<std::int64_t>> ids;
    models::Input::Trace verify_trace, serial_trace;
    ::tyloi::ops::Trace verify_stages, serial_stages;
    Tensor raw;
    std::optional<TextSessionState> actual_state, reference_state;
    std::optional<engine::InferenceRequest> request;
    std::optional<engine::InferenceOutput> output;
    StepSequence<engine::EventData> sequence;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& e) { std::cerr << "mtp_check_cleanup error=" << e.what() << '\n'; }
        catch (...) { std::cerr << "mtp_check_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor tensor) { tensors.push_back(tensor); return tensor; }
    void retire() {
        program.drain();
        sequence = {}; output.reset(); request.reset();
        tensors.clear(); floats.clear(); ids.clear();
        verify_trace.clear(); serial_trace.clear(); verify_stages.clear(); serial_stages.clear();
        actual_state.reset(); reference_state.reset(); raw = {};
    }
};

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

Tensor rows(Work& w, const Tensor& x, std::int64_t start, std::int64_t count) {
    return w.keep(::tyloi::ops::slice(x, 1, start, start + count));
}

void compare(const Tensor& actual, const Tensor& reference, double tolerance,
             const std::string& name, bool& mismatch) {
    require(actual.dim() == reference.dim(), "MTP gate tensor shape mismatch");
    for (std::int64_t i = 0; i < actual.dim(); ++i)
        require(actual.size(i) == reference.size(i), "MTP gate tensor shape mismatch");
    const auto a = ::tyloi::ops::floats(actual), r = ::tyloi::ops::floats(reference);
    double squared = 0., norm = 0.;
    for (std::size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(r[i]), "nonfinite MTP gate value");
        squared += (double(a[i]) - r[i]) * (double(a[i]) - r[i]);
        norm += double(r[i]) * r[i];
    }
    const double rel = std::sqrt(squared / std::max(norm, 1.e-30));
    std::cout << "mtp_check " << name << " relative_l2=" << rel << '\n';
    mismatch = mismatch || rel > tolerance;
}
} // namespace

int run_qwen35_mtp_check(Program& program) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work w{program};
    auto& model = program.model();
    auto& state = program.state();
    auto* mtp = model.mtp();
    auto loader = program.loader();
    const auto* source = dynamic_cast<const ModelSource*>(&loader.source);
    require(source && models::graph(*source).backbone == "qwen3_5" && mtp && state.predictor,
            "Qwen3.5 MTP diagnostic requires a loaded predictor");
    auto& predictor = *state.predictor;
    const auto device = executor.device();
    const auto mode = executor.forward_mode(device.index);
    models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, device.index};
    auto project = executor.projector(Executor::current_ffn());
    w.floats.resize(5);
    auto& identity = w.floats[0]; auto& input = w.floats[1]; auto& gate = w.floats[2];
    identity.assign(33 * 33, 0.f); input.resize(6 * 33); gate.resize(6 * 33);
    for (int i = 0; i < 33; ++i) identity[i * 33 + i] = 1.f;
    for (int i = 0; i < 6 * 33; ++i) {
        input[i] = static_cast<float>(i % 33 - 16) / 16.f;
        gate[i] = static_cast<float>(i % 31 - 15) / 4.f;
    }
    auto x = w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::tensor(input, device)), {1, 6, 33}));
    auto g = w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::tensor(gate, device)), {1, 6, 33}));
    auto ident = w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::tensor(identity, device)), {33, 33}));
    for (auto dtype : {kFloat32, kFloat16, kBFloat16}) {
        auto weight = w.keep(::tyloi::ops::cast(ident, dtype));
        const double tolerance = dtype == kBFloat16 ? .008 : dtype == kFloat16 ? .001 : 2.e-6;
        for (int m = 1; m <= 6; ++m) for (auto activation : {InputGateMode::Sigmoid, InputGateMode::Silu}) {
            auto actual = w.keep(project.gated(weight, rows(w, x, 0, m).dense(), rows(w, g, 0, m).dense(), activation));
            require(actual.scalar_type() == dtype && actual.size(1) == m, "dense gate output dtype or shape mismatch");
            auto values = ::tyloi::ops::floats(actual);
            for (int i = 0; i < m * 33; ++i) {
                const double activated = (activation == InputGateMode::Sigmoid ? 1. : gate[i]) / (1. + std::exp(-double(gate[i])));
                const double reference = double(input[i]) * activated;
                require(std::isfinite(values[i]) && std::abs(values[i] - reference) <= tolerance * (1. + std::abs(reference)),
                        "dense gate differs from CPU double identity oracle");
            }
        }
    }
    std::cout << "mtp_check dense_gate_cases=36 PASS\n";
    auto& pi = w.floats[3]; auto& pw = w.floats[4]; pi.resize(6 * 257); pw.resize(131 * 257);
    for (std::size_t i = 0; i < pi.size(); ++i) pi[i] = static_cast<float>(std::sin(double(i) * .37));
    for (std::size_t i = 0; i < pw.size(); ++i) pw[i] = static_cast<float>(std::cos(double(i) * .19) / 17.);
    auto px = w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::tensor(pi, device)), {1, 6, 257}));
    auto pweight = w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::tensor(pw, device)), {131, 257}));
    for (auto dtype : {kFloat32, kFloat16, kBFloat16}) {
        auto weight = w.keep(::tyloi::ops::cast(pweight, dtype));
        std::vector<Tensor> serial;
        for (int row = 0; row < 6; ++row) serial.push_back(w.keep(project.run(weight, rows(w, px, row, 1).dense())));
        auto reference = w.keep(::tyloi::ops::cat(serial, 1));
        for (int m = 2; m <= 6; ++m) {
            auto actual = w.keep(project.run(weight, rows(w, px, 0, m).dense()));
            require(::tyloi::ops::equal(actual, rows(w, reference, 0, m)), "short dense projection differs from per-token decode");
        }
    }
    std::cout << "mtp_check dense_decode_geometry_cases=15 PASS\n";
    w.retire();
    const auto ids = [&](std::vector<std::int64_t> tokens) {
        w.ids.push_back(std::move(tokens));
        return w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::tensor(w.ids.back(), device)), {1, -1}));
    };
    const auto forward = [&](const Tensor& tokens, std::int64_t confirmed, models::Input::Trace* trace,
                             Tensor* raw, const models::ForwardOptions& forward_options) {
        models::Input arguments; arguments.ids = tokens; arguments.advance = true;
        arguments.confirmed = confirmed; arguments.trace = trace; arguments.raw_hidden = raw;
        return w.keep(model.forward(std::move(arguments), state, forward_options));
    };
    const auto logits = [&](const Tensor& tokens, bool capped) {
        auto hidden = forward(tokens, 0, nullptr, nullptr, options);
        auto last = rows(w, hidden, hidden.size(1) - 1, 1);
        last = w.keep(::tyloi::ops::contiguous(w.keep(::tyloi::ops::cast(w.keep(::tyloi::ops::reshape(last, {hidden.size(0), hidden.size(2)})), kFloat16))));
        return w.keep(model.logits(last, capped));
    };
    const std::vector<std::int64_t> prompt{100, 200, 300, 400, 500, 600, 700};
    bool mismatch = false;
    for (int accepted = 0; accepted <= 2; ++accepted) {
        w.retire(); state.reset(1);
        (void)forward(ids(prompt), 0, nullptr, nullptr, options);
        auto traced = options; traced.trace_layer = 3; traced.trace = &w.verify_stages;
        (void)forward(ids({37, 41, 43}), 1, &w.verify_trace, nullptr, traced);
        if (accepted == 2) state.commit_speculative(); else state.rollback_speculative(accepted);
        require(state.position() == 8 + accepted, "MTP resolution retained an incorrect logical cache length");
        auto actual = w.keep(::tyloi::ops::clone(logits(ids({47}), true)));
        auto retained = prompt; retained.push_back(37); if (accepted >= 1) retained.push_back(41);
        if (accepted >= 2) retained.push_back(43);
        retained.push_back(47);
        w.actual_state = state.capture_text_session_state(retained);
        program.drain(); state.reset(1);
        (void)forward(ids(prompt), 0, nullptr, nullptr, options);
        traced.trace = &w.serial_stages;
        (void)forward(ids({37}), 0, &w.serial_trace, nullptr, traced);
        require(w.verify_stages.size() == w.serial_stages.size() && !w.verify_stages.empty(), "MTP full-attention stage trace mismatch");
        for (std::size_t i = 0; i < w.verify_stages.size(); ++i) {
            require(w.verify_stages[i].first == w.serial_stages[i].first, "MTP full-attention stage names differ");
            compare(rows(w, w.verify_stages[i].second, 0, 1), w.serial_stages[i].second, .005, w.verify_stages[i].first, mismatch);
        }
        require(w.verify_trace.size() == w.serial_trace.size(), "MTP diagnostic block trace size mismatch");
        for (std::size_t i = 0; i < w.verify_trace.size(); ++i)
            compare(rows(w, w.verify_trace[i], 0, 1), w.serial_trace[i], .005, "confirmed_prefix_block_" + std::to_string(i), mismatch);
        if (accepted >= 1) (void)forward(ids({41}), 0, nullptr, nullptr, options);
        if (accepted >= 2) (void)forward(ids({43}), 0, nullptr, nullptr, options);
        auto expected = logits(ids({47}), true);
        compare(actual, expected, .005, "resolution_" + std::to_string(accepted) + "_logits", mismatch);
        w.reference_state = state.capture_text_session_state(retained);
        const auto* a = std::get_if<std::vector<HybridBlockSessionState>>(&w.actual_state->payload);
        const auto* b = std::get_if<std::vector<HybridBlockSessionState>>(&w.reference_state->payload);
        require((a == nullptr) == (b == nullptr) && (!a || a->size() == b->size()), "MTP recurrent state topology mismatch");
        std::size_t index = 0;
        for (std::size_t i = 0; a && i < a->size(); ++i) if ((*a)[i].kind == HybridBlockSessionStateKind::Recurrent) {
            require((*b)[i].kind == (*a)[i].kind, "MTP recurrent state topology mismatch");
            compare((*a)[i].convolution, (*b)[i].convolution, .002, "conv_continuation_" + std::to_string(index), mismatch);
            compare((*a)[i].recurrent, (*b)[i].recurrent, .002, "gdn_continuation_" + std::to_string(index), mismatch);
            ++index;
        }
    }
    require(!mismatch, "MTP gate numerical mismatch (see per-layer diagnostics)");
    w.retire(); state.reset(1);
    (void)forward(ids(prompt), 0, nullptr, &w.raw, options);
    for (int tokens = 2; tokens <= 6; ++tokens) {
        auto next = rows(w, ids(prompt), 1, tokens);
        auto traced = options; traced.trace_layer = 0; traced.trace = tokens == 2 ? &w.verify_stages : nullptr;
        program.drain(); mtp->reset(predictor, 1);
        auto batched = w.keep(::tyloi::ops::clone(mtp->step(predictor, rows(w, w.raw, 0, tokens), next, {}, traced).sample_hidden));
        program.drain(); mtp->reset(predictor, 1);
        traced.trace = tokens == 2 ? &w.serial_stages : nullptr;
        std::vector<Tensor> serial;
        for (int t = 0; t < tokens; ++t)
            serial.push_back(w.keep(mtp->step(predictor, rows(w, w.raw, t, 1), rows(w, next, t, 1), {}, traced).sample_hidden));
        if (tokens == 2) {
            require(w.serial_stages.size() == 2 * w.verify_stages.size(), "MTP predictor diagnostic stage count mismatch");
            for (std::size_t i = 0; i < w.verify_stages.size(); ++i) {
                auto expected = w.keep(::tyloi::ops::cat(std::vector<Tensor>{w.serial_stages[i].second, w.serial_stages[i + w.verify_stages.size()].second}, 1));
                compare(w.verify_stages[i].second, expected, std::numeric_limits<double>::infinity(), "predictor_stage_" + w.verify_stages[i].first, mismatch);
            }
        }
        compare(batched, w.keep(::tyloi::ops::cat(serial, 1)), .005, "predictor_batched_vs_serial", mismatch);
    }
    require(!mismatch, "MTP predictor gate numerical mismatch");
    w.retire();
    program.configure(engine::resolve_config());
    auto generate = [&](const std::vector<std::int64_t>& tokens, const MfqSamplingParams& sampling, int stop) {
        w.retire(); state.reset(1); mtp->reset(predictor, 1);
        w.request.emplace(); w.request->prompt = tokens; w.request->sampling = sampling; w.request->sampling.enable_mtp = true;
        w.output.emplace(*w.request, nullptr, "mtp-check");
        w.sequence = program.generate(*w.request, *w.output, true, "mtp-check");
        std::vector<std::int64_t> result; std::size_t largest = 0;
        while (auto step = w.sequence.next()) {
            if (!step.value) continue;
            if (auto* delta = std::get_if<engine::OutputDelta>(&*step.value)) {
                require(!state.speculating(), "MTP published an uncommitted transaction");
                largest = std::max(largest, delta->token_ids.size());
                result.insert(result.end(), delta->token_ids.begin(), delta->token_ids.end());
                if (stop) w.request->sampling.max_tokens = stop;
            }
        }
        if (w.output->cleanup_failure) std::rethrow_exception(w.output->cleanup_failure);
        const auto stats = program.stats(); const auto& metrics = w.output->metrics.mtp;
        if (!stop && result.size() >= 16 && stats.accepted > 1)
            require(largest > 1, "MTP split a verification delta into per-token events");
        require(metrics.available && metrics.cycles == stats.request.cycles && metrics.drafted_tokens == stats.request.drafted_tokens &&
            metrics.accepted_tokens == stats.request.accepted_tokens, "request did not retain its MTP statistics");
        return result;
    };
    std::uint64_t total_cycles = 0;
    for (const auto& prompt_ids : std::vector<std::vector<std::int64_t>>{{1,2,3}, prompt, std::vector<std::int64_t>(17,10)}) {
        MfqSamplingParams params; params.max_tokens = 32; params.temperature = 0.; params.top_k = 1; params.seed = 20260907;
        w.retire(); state.reset(1); auto current = ids(prompt_ids); std::vector<std::int64_t> expected;
        for (int i = 0; i < 32; ++i) {
            auto scores = logits(current, false);
            auto next = w.keep(::tyloi::ops::sample_greedy(w.keep(::tyloi::ops::reshape(w.keep(::tyloi::ops::contiguous(scores)), {scores.size(0), -1}))));
            expected.push_back(::tyloi::ops::item(next)); current = w.keep(::tyloi::ops::reshape(next, {1,1}));
        }
        auto got = generate(prompt_ids, params, 0); auto stats = program.stats(); total_cycles += stats.cycles;
        std::cout << "mtp_check greedy_prompt_tokens=" << prompt_ids.size() << " produced=" << got.size()
                  << " exact=" << (got == expected) << " accepted=" << stats.accepted << " rejected=" << stats.rejected << '\n';
        require(got.size() == 32 && got == expected, "MTP greedy generation differs from ordinary incremental decode");
        got = generate(prompt_ids, params, 3);
        require(got.size() == 3 && got == std::vector<std::int64_t>(expected.begin(), expected.begin()+3), "MTP step emitted extra or incorrect tokens");
    }
    MfqSamplingParams stochastic; stochastic.max_tokens = 8; stochastic.temperature = .8; stochastic.top_k = 100;
    stochastic.top_p = .95; stochastic.presence_penalty = .2; stochastic.frequency_penalty = .1;
    stochastic.repetition_penalty = 1.05; stochastic.seed = 20260907;
    auto produced = generate(prompt, stochastic, 0); auto stats = program.stats();
    require(produced.size() == 8 && stats.cycles > 0 && total_cycles > 0, "MTP runtime gate did not execute speculative cycles");
    auto counters = state.metrics();
    require(counters.has_value(), "MTP batching gate found no recurrent layers");
    std::int64_t layers = 0, ffn = 0, projection = 0;
    for (const auto& [key, value] : *counters) {
        if (!key.starts_with("linear.") || !key.ends_with(".ffn_batches")) continue;
        const auto projection_key = key.substr(0, key.size() - 11) + "projection_batches";
        const auto found = std::find_if(counters->begin(), counters->end(), [&](const auto& metric) {
            return metric.first == projection_key;
        });
        require(value > 0 && found != counters->end() && found->second > 0,
                "MTP gate did not exercise full-window recurrent batching");
        ++layers;
        ffn += static_cast<std::int64_t>(value);
        projection += static_cast<std::int64_t>(found->second);
    }
    require(layers > 0, "MTP batching gate found no recurrent layers");
    std::cout << "mtp_check batched_linear_layers=" << layers << " ffn_calls=" << ffn << " projection_calls=" << projection << '\n';
    std::cout << "mtp_check PASS full_chain=1 greedy_tokens=96 stochastic_smoke_tokens=8\n";
    return 0;
}
} // namespace tyloi::cuda::diagnostics
