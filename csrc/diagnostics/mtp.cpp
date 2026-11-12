#include "diagnostics/mtp.h"

#include "engine/program.h"
#include "engine/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/causal_model.h"
#include "engine/models/common/mtp.h"
#include "engine/models/session/snapshot.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"
#include "engine/ops.h"
#include "engine/requests/output.h"
#include "engine/weights/loader.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
namespace ops = ::tyloi::ops;

/** Retain diagnostic inputs, request state and staged copies until drain. */
struct Work {
    Program& program;
    std::vector<Tensor> tensors;
    std::vector<std::int64_t> id_values;
    std::vector<float> previous_values;
    Tensor target_hidden;
    std::optional<MtpSessionState> snapshot;
    std::unique_ptr<models::Restore> restore;
    std::optional<engine::InferenceRequest> request;
    std::optional<engine::InferenceOutput> output;
    StepSequence<engine::EventData> sequence;

    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "mtp_check_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "mtp_check_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) {
        tensors.push_back(value);
        return value;
    }
    void retire_request() {
        program.drain();
        sequence = {};
        output.reset();
        request.reset();
    }
};

nlohmann::json json_tensor(const Tensor& value) {
    std::vector<std::int64_t> shape;
    for (std::int64_t i = 0; i < value.dim(); ++i) shape.push_back(value.size(i));
    return {{"shape", shape}, {"data", ops::floats(value)}};
}

void close(const Tensor& actual, const Tensor& expected, const char* label) {
    if (actual.dim() != expected.dim())
        throw std::runtime_error(std::string("Nemotron-H MTP ") + label + " changed its shape");
    for (std::int64_t i = 0; i < actual.dim(); ++i)
        if (actual.size(i) != expected.size(i))
            throw std::runtime_error(std::string("Nemotron-H MTP ") + label + " changed its shape");
    const auto a = ops::floats(actual), b = ops::floats(expected);
    double worst = 0.;
    for (std::size_t i = 0; i < a.size(); ++i)
        worst = std::max(worst, std::abs(double(a[i]) - b[i]) / (.003 + .003 * std::abs(b[i])));
    if (worst > 1.)
        throw std::runtime_error(std::string("Nemotron-H MTP ") + label +
            " changed its value normalized_error=" + std::to_string(worst));
}

Tensor slice(Work& work, const Tensor& tensor, std::int64_t begin, std::int64_t count) {
    return work.keep(ops::slice(tensor, 1, begin, begin + count));
}
}  // namespace

int run_nemotron_h_mtp_check(Program& program) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    auto& model = program.model();
    auto& state = program.state();
    auto* mtp = model.mtp();
    const auto info = model.info();
    auto loader = program.loader();
    const auto* source = dynamic_cast<const ModelSource*>(&loader.source);
    if (!source || models::graph(*source).backbone != "nemotron_h" || !mtp || !state.predictor)
        throw std::runtime_error("Nemotron-H MTP diagnostic requires a loaded predictor");
    if (info.vocab_size < 8 || info.max_context < 32)
        throw std::runtime_error("Nemotron-H MTP diagnostic requires vocab>=8 and context>=32");
    auto& predictor = *state.predictor;
    const auto device = executor.device();
    const auto mode = executor.forward_mode(device.index);
    const models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, device.index};
    work.id_values = {1, 2, 3, 4, 5, 6, 7};
    const auto ids = work.keep(ops::reshape(work.keep(ops::tensor(work.id_values, device)), {1, 7}));
    const auto embedding = work.keep(model.embedding(ids, device.index));
    const auto width = embedding.size(2);
    work.previous_values.resize(static_cast<std::size_t>(7 * width));
    for (std::int64_t i = 0; i < 7 * width; ++i) {
        const volatile float product = static_cast<float>(i % 31) * .01f;
        work.previous_values[static_cast<std::size_t>(i)] = product - .15f;
    }
    const auto previous = work.keep(ops::reshape(work.keep(ops::tensor(work.previous_values, device)), {1, 7, width}));
    const auto reset = [&] {
        program.drain();
        mtp->reset(predictor, 1);
    };
    const auto head = [&](const Tensor& hidden, const Tensor& next_ids) {
        auto value = mtp->step(predictor, hidden, next_ids, {}, options);
        work.keep(value.chain_hidden);
        return work.keep(value.sample_hidden);
    };
    nlohmann::json result;
    result["architecture"] = models::graph(*source).backbone;
    result["previous"] = json_tensor(previous);
    result["ids"] = json_tensor(ids);
    result["embedding"] = json_tensor(embedding);
    reset();
    const auto full = head(previous, ids);
    result["head"] = json_tensor(full);
    result["logits"] = json_tensor(work.keep(model.logits(full)));

    reset();
    std::vector<Tensor> pieces;
    for (auto [begin, count] : std::vector<std::pair<std::int64_t, std::int64_t>>{{0, 2}, {2, 1}, {3, 4}})
        pieces.push_back(head(slice(work, previous, begin, count), slice(work, ids, begin, count)));
    close(work.keep(ops::cat(pieces, 1)), full, "chunked forward");

    reset();
    (void)head(slice(work, previous, 0, 4), slice(work, ids, 0, 4));
    const auto committed = predictor.position();
    if (committed != 4) throw std::runtime_error("Nemotron-H MTP history did not advance");
    const auto rejected = head(slice(work, previous, 4, 1), slice(work, ids, 4, 1));
    mtp->trim_cache_to(predictor, committed);
    const auto replayed = head(slice(work, previous, 4, 1), slice(work, ids, 4, 1));
    close(replayed, rejected, "rollback replay");
    const auto tail = work.keep(ops::cat(std::vector<Tensor>{replayed,
        head(slice(work, previous, 5, 2), slice(work, ids, 5, 2))}, 1));
    close(tail, slice(work, full, 4, 3), "continuation after rollback");
    result["replayed"] = json_tensor(replayed);
    result["rollback"] = json_tensor(slice(work, full, 4, 1));

    reset();
    (void)head(slice(work, previous, 0, 4), slice(work, ids, 0, 4));
    const auto draft = head(slice(work, previous, 4, 1), slice(work, ids, 4, 1));
    const auto following = head(slice(work, previous, 5, 2), slice(work, ids, 5, 2));
    const auto committed_rows = work.keep(ops::cat(std::vector<Tensor>{draft, following}, 1));
    close(committed_rows, slice(work, full, 4, 3), "continuation after commit");
    result["committed"] = json_tensor(committed_rows);

    reset();
    const auto first = mtp->step(predictor, slice(work, previous, 0, 1), slice(work, ids, 0, 1), {}, options);
    work.keep(first.sample_hidden); work.keep(first.chain_hidden);
    close(first.chain_hidden, first.sample_hidden, "depth chain value");
    close(first.sample_hidden, slice(work, full, 0, 1), "first depth step");
    const auto second = mtp->step(predictor, first.chain_hidden, slice(work, ids, 1, 1), {}, options);
    work.keep(second.sample_hidden); work.keep(second.chain_hidden);
    close(second.chain_hidden, second.sample_hidden, "depth chain value");
    result["chain_first"] = json_tensor(first.sample_hidden);
    result["chain_second"] = json_tensor(second.sample_hidden);

    program.drain();
    state.reset(1);
    models::Input input;
    input.ids = ids; input.advance = true; input.raw_hidden = &work.target_hidden;
    const auto normalized = work.keep(model.forward(std::move(input), state, options));
    close(work.target_hidden, normalized, "post-output_norm target hidden");
    result["target_hidden"] = json_tensor(work.target_hidden);
    result["target_normalized"] = json_tensor(normalized);
    reset();
    result["head_from_target"] = json_tensor(head(slice(work, work.target_hidden, 0, 6), slice(work, ids, 1, 6)));

    reset();
    (void)head(slice(work, previous, 0, 3), slice(work, ids, 1, 3));
    work.snapshot = mtp->capture_session_state(predictor, 4, slice(work, previous, 3, 1));
    const auto expected = head(slice(work, previous, 3, 1), slice(work, ids, 4, 1));
    (void)head(slice(work, previous, 4, 2), slice(work, ids, 5, 2));
    work.restore = mtp->prepare_restore(predictor, *work.snapshot);
    if (!work.restore) throw std::runtime_error("Nemotron-H MTP session restore is unavailable");
    work.restore->stage();
    program.drain();
    work.restore->publish();
    if (predictor.position() != 3) throw std::runtime_error("Nemotron-H MTP restore position changed");
    close(head(slice(work, previous, 3, 1), slice(work, ids, 4, 1)), expected, "session restore");

    program.configure(engine::resolve_config());
    MfqSamplingParams sampling;
    sampling.max_tokens = 32; sampling.temperature = 0.; sampling.top_k = 1; sampling.top_p = 1.;
    const std::vector<std::int64_t> prompt{1, 2, 3, 4, 5};
    const auto generate = [&](bool enabled, bool use_predictor) {
        work.retire_request();
        if (!enabled) program.configure(engine::resolve_config());
        state.reset(1); mtp->reset(predictor, 1);
        work.request.emplace();
        work.request->prompt = prompt;
        work.request->sampling = sampling;
        work.request->sampling.enable_mtp = enabled;
        work.output.emplace(*work.request, nullptr, "nemotron-h-mtp");
        work.sequence = program.generate(*work.request, *work.output, use_predictor, "nemotron-h-mtp");
        std::vector<std::int64_t> tokens;
        std::size_t largest = 0;
        while (auto step = work.sequence.next()) {
            if (!step.value) continue;
            if (auto* delta = std::get_if<engine::OutputDelta>(&*step.value)) {
                if (enabled && state.speculating())
                    throw std::runtime_error("MTP published an uncommitted transaction");
                largest = std::max(largest, delta->token_ids.size());
                tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
            }
        }
        const auto stats = program.stats();
        if (enabled) {
            const auto& metrics = work.output->metrics.mtp;
            if (tokens.size() >= 16 && stats.accepted > 1 && largest <= 1)
                throw std::runtime_error("MTP split a verification delta into per-token events");
            if (!metrics.available || metrics.cycles != stats.request.cycles ||
                metrics.drafted_tokens != stats.request.drafted_tokens || metrics.accepted_tokens != stats.request.accepted_tokens)
                throw std::runtime_error("request did not retain its MTP statistics");
        }
        return tokens;
    };
    const auto speculative = generate(true, true);
    const auto stats = program.stats();
    result["greedy_mtp"] = speculative;
    result["cycles"] = stats.request.cycles;
    result["accepted"] = stats.accepted;
    result["rejected"] = stats.rejected;
    if (!stats.request.cycles) throw std::runtime_error("Nemotron-H MTP diagnostic ran no speculative cycle");
    const auto plain = generate(false, false);
    result["greedy_plain"] = plain;
    if (speculative != plain) throw std::runtime_error("Nemotron-H MTP greedy generation differs from ordinary decode");
    const auto disabled = generate(false, true);
    result["greedy_disabled"] = disabled;
    if (disabled != plain || program.stats().request.cycles != 0)
        throw std::runtime_error("disabled Nemotron-H MTP still ran speculative cycles");
    program.drain();
    std::cout << "nemotron_h_mtp_check " << result.dump() << '\n';
    return 0;
}

int run_flash_next_mtp_check(Program& program) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    auto& model = program.model();
    auto& state = program.state();
    auto* mtp = model.mtp();
    auto loader = program.loader();
    const auto* source = dynamic_cast<const ModelSource*>(&loader.source);
    if (!source || !mtp || !state.predictor)
        throw std::runtime_error("Flash-Next MTP diagnostic requires a loaded predictor");
    const auto backbone = models::graph(*source).backbone;
    const bool multi_axis = backbone == "qwen4_exp";
    const auto info = model.info();
    if ((!multi_axis && backbone != "glm5_next") || info.vocab_size < 8 || info.max_context < 24)
        throw std::runtime_error("Flash-Next MTP diagnostic requires vocab>=8 and context>=24");
    auto& predictor = *state.predictor;
    const auto device = executor.device();
    const auto mode = executor.forward_mode(device.index);
    const models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, device.index};
    work.id_values = {1, 2, 3, 4, 5, 6, 7};
    const auto ids = work.keep(ops::reshape(work.keep(ops::tensor(work.id_values, device)), {1, 7}));
    const auto width = mtp->width();
    work.previous_values.resize(static_cast<std::size_t>(7 * width));
    for (std::int64_t i = 0; i < 7 * width; ++i) {
        const volatile float product = static_cast<float>(i % 31) * .01f;
        work.previous_values[static_cast<std::size_t>(i)] = product - .15f;
    }
    const auto previous = work.keep(ops::reshape(work.keep(ops::tensor(work.previous_values, device)), {1, 7, width}));
    const auto reset = [&] { program.drain(); mtp->reset(predictor, 1); };
    const auto evaluate = [&](const Tensor& hidden, const Tensor& next_ids,
            std::int64_t depth = 0, bool cache = true, const Tensor& positions = Tensor{}) {
        auto output = mtp->evaluate(predictor, {hidden, next_ids, positions, depth, cache, options});
        work.keep(output.sample_hidden); work.keep(output.chain_hidden);
        return output;
    };
    nlohmann::json result;
    result["architecture"] = backbone;
    result["previous"] = json_tensor(previous);
    const auto heads = predictor.head_positions().size();
    for (std::size_t i = 0; i < heads; ++i) {
        nlohmann::json row;
        reset();
        const auto full = evaluate(previous, ids, i);
        row["full"] = json_tensor(full.sample_hidden);
        row["multi"] = json_tensor(full.chain_hidden);
        row["logits"] = json_tensor(work.keep(model.logits(full.sample_hidden)));
        const auto saved = predictor.head_positions();
        row["uncached"] = json_tensor(evaluate(previous, ids, i, false).sample_hidden);
        if (saved != predictor.head_positions()) throw std::runtime_error("uncached MTP changed layer positions");
        reset();
        std::vector<Tensor> pieces;
        for (auto [begin, count] : std::vector<std::pair<std::int64_t, std::int64_t>>{{0, 2}, {2, 1}, {3, 4}})
            pieces.push_back(evaluate(slice(work, previous, begin, count), slice(work, ids, begin, count), i).sample_hidden);
        row["chunked"] = json_tensor(work.keep(ops::cat(pieces, 1)));
        reset();
        row["reset"] = json_tensor(evaluate(previous, ids, i).sample_hidden);
        row["batch"] = json_tensor(evaluate(work.keep(ops::repeat(previous, {2, 1, 1})),
            work.keep(ops::repeat(ids, {2, 1})), i).sample_hidden);
        row["batch_reset"] = json_tensor(evaluate(previous, ids, i).sample_hidden);
        auto positions = work.keep(ops::add(work.keep(ops::reshape(ids, {7})), 2));
        if (multi_axis) {
            std::vector<Tensor> axes;
            for (int offset : {10, 0, 2, 4})
                axes.push_back(work.keep(ops::unsqueeze(work.keep(ops::add(positions, offset)), 0)));
            positions = work.keep(ops::cat(axes, 0));
        }
        row["axis"] = json_tensor(evaluate(previous, ids, i, false, positions).sample_hidden);
        reset();
        for (std::size_t other = 0; other < heads; ++other)
            if (other != i) (void)evaluate(previous, ids, other);
        row["independent"] = json_tensor(evaluate(previous, ids, i).sample_hidden);
        result["layers"].push_back(std::move(row));
    }
    program.drain();
    state.reset(1);
    models::Input input;
    input.ids = ids; input.advance = true; input.raw_hidden = &work.target_hidden;
    const auto normalized = work.keep(model.forward(std::move(input), state, options));
    result["target_raw"] = json_tensor(work.target_hidden);
    result["target_normalized"] = json_tensor(normalized);

    program.configure(engine::resolve_config());
    MfqSamplingParams sampling;
    sampling.max_tokens = 12; sampling.temperature = 0.; sampling.top_k = 1;
    program.drain(); state.reset(1);
    auto current = ids;
    std::vector<std::int64_t> expected_tokens;
    for (int i = 0; i < sampling.max_tokens; ++i) {
        models::Input current_input; current_input.ids = current; current_input.advance = true;
        auto hidden = work.keep(model.forward(std::move(current_input), state, options));
        auto last = work.keep(ops::reshape(slice(work, hidden, hidden.size(1) - 1, 1), {hidden.size(0), hidden.size(2)}));
        auto logits = work.keep(model.logits(work.keep(ops::contiguous(last)), false));
        auto next = work.keep(ops::sample_greedy(work.keep(ops::reshape(work.keep(ops::contiguous(logits)), {logits.size(0), -1}))));
        expected_tokens.push_back(ops::item(next));
        current = work.keep(ops::reshape(next, {1, 1}));
    }
    const auto generate = [&](MfqSamplingParams params, int stop_after) {
        work.retire_request(); state.reset(1); mtp->reset(predictor, 1);
        work.request.emplace(); work.request->prompt = work.id_values; work.request->sampling = params;
        work.output.emplace(*work.request, nullptr, "flash-next-mtp");
        work.sequence = program.generate(*work.request, *work.output, true, "flash-next-mtp");
        std::vector<std::int64_t> tokens;
        std::size_t largest = 0;
        while (auto step = work.sequence.next()) {
            if (!step.value) continue;
            if (auto* delta = std::get_if<engine::OutputDelta>(&*step.value)) {
                if (state.speculating()) throw std::runtime_error("MTP published an uncommitted transaction");
                largest = std::max(largest, delta->token_ids.size());
                tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                if (stop_after) work.request->sampling.max_tokens = stop_after;
            }
        }
        const auto stats = program.stats();
        const auto& metrics = work.output->metrics.mtp;
        if (!stop_after && tokens.size() >= 16 && stats.accepted > 1 && largest <= 1)
            throw std::runtime_error("MTP split a verification delta into per-token events");
        if (!metrics.available || metrics.cycles != stats.request.cycles ||
            metrics.drafted_tokens != stats.request.drafted_tokens || metrics.accepted_tokens != stats.request.accepted_tokens)
            throw std::runtime_error("request did not retain its MTP statistics");
        return tokens;
    };
    const auto generated = generate(sampling, 0);
    const auto stats = program.stats();
    if (generated.size() != static_cast<std::size_t>(sampling.max_tokens) || generated != expected_tokens || !stats.cycles)
        throw std::runtime_error("Flash-Next MTP greedy tokens disagree with incremental target");
    result["greedy"] = generated; result["cycles"] = stats.cycles;
    result["accepted"] = stats.accepted; result["rejected"] = stats.rejected;
    result["drafted"] = stats.request.drafted_tokens; result["selected_depth"] = stats.request.selected_depth;
    result["depth_cycles"] = stats.request.depth_cycles;
    const auto stopped = generate(sampling, 3);
    if (stopped.size() != 3 || stopped != std::vector<std::int64_t>(expected_tokens.begin(), expected_tokens.begin() + 3))
        throw std::runtime_error("Flash-Next MTP step emitted extra or incorrect tokens");
    sampling.max_tokens = 8; sampling.temperature = .8; sampling.top_k = 16; sampling.top_p = .95;
    sampling.presence_penalty = .2; sampling.frequency_penalty = .1; sampling.repetition_penalty = 1.05; sampling.seed = 20260907;
    if (generate(sampling, 0).size() != 8) throw std::runtime_error("Flash-Next stochastic MTP failed to generate requested tokens");
    program.drain();
    std::cout << "flash_next_mtp_check " << result.dump() << '\n';
    return 0;
}
}  // namespace tyloi::cuda::diagnostics
