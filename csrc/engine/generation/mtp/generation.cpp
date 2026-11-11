#include "engine/generation/mtp/generation.h"

#include "engine/budget.h"
#include "engine/config.h"
#include "engine/generation/mtp/sampling.h"
#include "engine/generation/mtp/sequence.h"
#include "engine/generation/prefill_budget.h"
#include "engine/generation/prefill.h"
#include "engine/models/common/mtp.h"
#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/weights/cache.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace tyloi::engine::mtp {
namespace {

struct Ops : Sampling {
    models::CausalModel& model;
    models::State& state;
    Executor& executor;
    Budget& budget;
    ::tyloi::weights::Cache& weights;
    State& mtp_state;
    const models::ForwardOptions options;
    models::Info info;
    models::Mtp& mtp;
    models::MtpState& predictor;
    const models::Prepared* prepared;
    const ModelSource* source;
    const bool profiling;
    const PrefetchConfig& prefetch;
    const std::int64_t compute_chunk;
    InferenceOutput& output;
    ::tyloi::weights::Cache::Routes routes;
    Tensor projected_hidden, projected_logits, host_ids;
    int32_t projected_greedy = -1;
    std::uint64_t target_transfer_bytes = 0;
    std::array<double, 6> target_profile{};
    std::uint64_t graph_captures = 0;

    Ops(models::CausalModel& model, models::State& state, Executor& executor,
        Budget& budget, ::tyloi::weights::Cache& weights, State& policy, const models::ForwardOptions& options,
        models::Mtp& mtp, models::MtpState& predictor, const models::Prepared* prepared,
        const ModelSource* source, bool profiling, const PrefetchConfig& prefetch,
        std::int64_t compute_chunk, InferenceOutput& output)
        : Sampling(executor.device()), model(model), state(state), executor(executor), budget(budget),
          weights(weights), mtp_state(policy), options(options), info(model.info()), mtp(mtp),
          predictor(predictor), prepared(prepared), source(source), profiling(profiling), prefetch(prefetch),
          compute_chunk(compute_chunk), output(output),
          graph_captures(weights.activity().graph_captures) {}

    void initialize(const std::vector<int64_t>& prompt, const MfqSamplingParams& sampling) {
        Sampling::initialize(prompt, sampling);
        count_prompt(info.vocab_size);
    }
    void reserve(const std::vector<std::pair<Device, std::size_t>>& growth) {
        for (const auto& [device, bytes] : growth)
            if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
    }
    std::array<double, 6> costs() const {
        std::array<double, 6> result{};
        for (const auto& [name, value] : weights.metrics()) {
            if (name == "expert_route_wait_ms") result[0] = value;
            else if (name == "expert_host_fetch_ms") result[1] = value;
            else if (name == "expert_host_copy_ms") result[2] = value;
            else if (name == "ssd_expert_read_wait_ms") result[3] = value;
        }
        if (source) {
            const auto io = source->file_read_stats();
            result[4] = double(io.read_nanoseconds) / 1e6;
            result[5] = double(io.physical_bytes) / 1048576.0;
        }
        return result;
    }
    Tensor forward(models::Input input) {
        reserve(state.growth(input.ids.size(1), options, input.confirmed));
        const auto before = weights.activity().streamed_bytes;
        const auto old = profiling ? costs() : std::array<double, 6>{};
        auto result = model.forward(std::move(input), state, options);
        target_transfer_bytes = weights.activity().streamed_bytes - before;
        if (!result) throw std::runtime_error("model forward returned no hidden tensor");
        if (profiling) {
            const auto current = costs();
            for (std::size_t i = 0; i < current.size(); ++i) target_profile[i] = current[i] - old[i];
        }
        return result;
    }
    Tensor target_forward(Tensor ids, Tensor* raw = nullptr, int64_t confirmed = 0) {
        models::Input input;
        input.ids = std::move(ids);
        input.raw_hidden = raw;
        input.confirmed = confirmed;
        return forward(std::move(input));
    }
    struct PrefillResult { Tensor hidden, raw; double elapsed; };
    PrefillResult prefill(PrefillChunk chunk) {
        auto timer = executor.timer();
        Tensor raw;
        models::Input input;
        input.ids = slice(input_ids, 1, chunk.offset, chunk.count);
        input.raw_hidden = &raw;
        if (prepared && prepared->transformed()) {
            if (prepared->embeddings) input.embeddings = slice(prepared->embeddings, 1, chunk.offset, chunk.count);
            if (prepared->positions) input.positions = slice(prepared->positions, -1, chunk.offset, chunk.count);
            input.advance = true;
        }
        if (prepared && prepared->image_mask)
            input.image_mask = slice(prepared->image_mask, 1, chunk.offset, chunk.count);
        auto result = engine::prefill(model, state, executor, weights, budget, options,
            std::move(input), compute_chunk, output);
        timer->stop();
        return {std::move(result.hidden), std::move(result.raw), timer->elapsed_ms()};
    }
    Tensor decode_positions(int64_t start, int64_t tokens) {
        if (tokens <= 0) throw std::runtime_error("MTP position span must be positive");
        const auto logical = start + (prepared && prepared->transformed() ? prepared->decode_position_delta : 0);
        if (logical < 0 || logical + tokens > info.max_context)
            throw std::runtime_error("MTP position span exceeds context capacity");
        auto positions = ops::arange(logical, logical + tokens, executor.device());
        if (prepared && prepared->positions && prepared->positions.dim() == 2 && prepared->positions.size(0) == 3)
            return ops::contiguous(ops::expand(ops::reshape(positions, {1, tokens}), {3, tokens}));
        return positions;
    }
    models::MtpStep predictor_step(Tensor hidden, Tensor ids, Tensor positions) {
        reserve(mtp.growth(predictor, models::Mtp::Step{ids}));
        auto result = mtp.step(predictor, hidden, ids, positions);
        projected_logits = result.sample_logits;
        projected_greedy = result.greedy_token;
        projected_hidden = projected_logits ? slice(result.sample_hidden, 1, result.sample_hidden.size(1) - 1, 1) : Tensor{};
        return result;
    }
    double predictor_prefill(Tensor hidden, Tensor ids, Tensor positions) {
        auto timer = executor.timer();
        hidden = ops::copy(hidden, executor.device());
        (void)predictor_step(std::move(hidden), std::move(ids), std::move(positions));
        timer->stop();
        return timer->elapsed_ms();
    }
    /** Reserve and append or seed the blockwise predictor's target context. */
    double blockwise_prompt_context(Tensor hidden, std::int64_t position, bool seed) {
        auto timer = executor.timer();
        if (seed) {
            reserve(mtp.growth(predictor, models::Mtp::Context{hidden, position, true}));
            mtp.seed_target_context(predictor, hidden, position);
        } else append_target_context(hidden, position - hidden.size(1));
        timer->stop();
        return timer->elapsed_ms();
    }
    /** Reserve and append committed target rows to the blockwise predictor. */
    void append_target_context(const Tensor& hidden, std::int64_t start) {
        reserve(mtp.growth(predictor, models::Mtp::Context{hidden, start, false}));
        mtp.append_target_context(predictor, hidden, start);
    }
    /** Reserve predictor storage before blockwise drafting. */
    Tensor draft_block(Tensor ids, int depth) {
        reserve(mtp.growth(predictor, models::Mtp::Draft{ids, depth}));
        return mtp.draft_block(predictor, ids, depth);
    }
    Tensor target_suffix(Tensor ids, Tensor* raw) {
        models::Input input;
        input.ids = std::move(ids);
        input.raw_hidden = raw;
        input.suffix = true;
        return forward(std::move(input));
    }
    /** Reserve the owning state's checkpoint before beginning a suffix transaction. */
    void begin_target_suffix(int64_t tokens) {
        reserve(state.suffix_growth(tokens));
        state.begin_speculative_suffix(tokens);
    }
    double end_predictor_prefill() {
        weights.prepare(::tyloi::weights::Cache::Stage::decode, executor.device(), source);
        return 0;
    }
    void rollback_target(std::int64_t accepted) { state.rollback_speculative(accepted); }
    bool prefetch_target_ids(int32_t token, const std::vector<int32_t>& drafts) {
        if (!info.row_prefetch || !prefetch.mtp || (prepared && prepared->transformed()) || output.stopped()) return false;
        const auto count = std::int64_t(1 + (mtp.split_target_verification() ? 0 : drafts.size()));
        if (!host_ids || host_ids.size(1) < count)
            host_ids = ops::empty({1, count}, TensorOptions().device(kCPU).dtype(kInt64));
        auto first = ops::slice(host_ids, 1, 0, 1);
        ops::fill(first, token);
        for (std::int64_t i = 1; i < count; ++i) {
            auto next = ops::slice(host_ids, 1, i, i + 1);
            ops::fill(next, drafts[static_cast<std::size_t>(i - 1)]);
        }
        auto host = ops::slice(host_ids, 1, 0, count);
        return model.prefetch(state, host);
    }
    void prefetch_target_first(int32_t token) { prefetch_target_ids(token, {}); }
    void prefetch_target_prefix(int32_t token, const std::vector<int32_t>& drafts) {
        if (prefetch.draft) prefetch_target_ids(token, drafts);
    }
    void prefetch_committed_token(int32_t token) {
        if (prefetch.publication) prefetch_target_ids(token, {});
    }
    void begin_target_routes(int rows) { routes.begin(weights, rows); }
    void resolve_target_routes(int rows, bool cancelled) { routes.resolve(rows, cancelled); }
    bool take_target_maintenance() {
        const auto captures = weights.activity().graph_captures;
        const bool changed = captures != graph_captures;
        graph_captures = captures;
        return changed;
    }
    void trace_prompt_prefill(double target, double predictor_ms, double workspace) {
        if (!mtp.teacher_forced_prompt_prime()) weights.prepare(::tyloi::weights::Cache::Stage::decode, executor.device(), source);
        std::cerr << "runtime_mtp_prefill target_ms=" << target << " predictor_ms=" << predictor_ms
                  << " predictor_workspace_ms=" << workspace << " streamed=0" << std::endl;
    }
    void trace_cycle(uint64_t cycle, int depth, int accepted, double draft, double verify, double commit) {
        const bool enabled = profiling;
        if (enabled || (cycle <= 32 && draft + verify + commit >= 55.0)) {
            std::cerr << (enabled ? "runtime_mtp_cycle_profile cycle=" : "runtime_mtp_slow_cycle cycle=")
                << cycle << " depth=" << depth << " accepted=" << accepted << " draft_ms=" << draft
                << " verify_ms=" << verify << " commit_ms=" << commit
                << " target_transfer_mb=" << target_transfer_bytes / 1048576.0;
            if (enabled) std::cerr << " target_route_ms=" << target_profile[0] << " target_fetch_ms=" << target_profile[1]
                << " target_host_copy_ms=" << target_profile[2] << " target_ssd_wait_ms=" << target_profile[3]
                << " target_parallel_read_ms=" << target_profile[4] << " target_physical_read_mb=" << target_profile[5];
            std::cerr << '\n';
        }
    }
    Tensor logits_for(Tensor hidden) {
        if (projected_hidden && hidden.same_view(projected_hidden)) return projected_logits;
        auto input = ops::contiguous(mtp.preserve_output_dtype() ? hidden : ops::cast(hidden, kFloat16));
        auto result = model.logits(input, true);
        if (!result) throw std::runtime_error("model logits returned no tensor");
        return result;
    }
    int32_t sample_normal(Tensor logits, Tensor token_counts) {
        if (sampler().greedy() && !penalties && projected_greedy >= 0 && projected_logits &&
            logits.numel() == projected_logits.numel() &&
            ops::reshape(logits, {1, -1}).same_view(ops::reshape(projected_logits, {1, -1})))
            return projected_greedy;
        return Sampling::sample_normal(std::move(logits), std::move(token_counts));
    }
    int32_t sample_constrained(Tensor logits, Tensor token_counts, const MfqTokenConstraintPtr& constraint) {
        if (!constraint && sampler().greedy() && !penalties)
            return sample_normal(std::move(logits), std::move(token_counts));
        return Sampling::sample_constrained(std::move(logits), std::move(token_counts), constraint);
    }
};

} // namespace

Generation generate(models::CausalModel& model, models::State& state, Executor& executor,
    Budget& budget, ::tyloi::weights::Cache& weights, State& policy, const models::ForwardOptions& options,
    InferenceRequest& input, InferenceOutput& output, std::int64_t chunk,
    const models::Prepared* prepared, std::size_t reused, Tensor restored,
    Tensor* committed, std::size_t stable, const ModelSource* source, bool profile,
    const PrefetchConfig& prefetch) {
    auto* predictor = model.mtp();
    if (!predictor || !state.predictor || !state.predictor->belongs_to(*predictor))
        throw std::invalid_argument("predictor state is unavailable or belongs to another model");
    const auto logical = prefill_span(model.info(), input.prompt.size() - reused, chunk,
        prepared && prepared->transformed(),
        weights.prefill_mode(input.prompt.size() - reused, state.position() > 0) == ::tyloi::weights::Cache::Prefill::retained);
    return speculative_sequence(Ops(model, state, executor, budget, weights, policy, options,
        *predictor, *state.predictor, prepared, source, profile, prefetch, chunk, output), input, output, logical, reused,
        std::move(restored), committed, stable);
}

} // namespace tyloi::engine::mtp
