#include "mlx_dflash2_serving.h"
#include "../models/dflash2/mlx_dflash2.h"
#include "mlx_resident_budget.h"
#include "mlx_kernel_prepare.h"

#include <stdexcept>

namespace mfq::metal {

bool mlx_dflash2_sampling_supported(const MfqSamplingParams& sampling,
    const MfqTokenConstraintPtr& constraint) noexcept {
    return sampling.enable_mtp && sampling.max_tokens > 1 &&
        sampling.presence_penalty == 0 && sampling.frequency_penalty == 0 &&
        sampling.repetition_penalty == 1 && !constraint;
}

void configure_mlx_dflash2(MlxEngineComponents& components,
    std::shared_ptr<std::optional<MlxQwen4CausalLm>> runtime,
    const std::filesystem::path& directory, const MlxSamplingParams& sampling) {
    auto draft = std::make_shared<MlxDFlash2>(MlxDFlash2::load(directory));
    components.prepare_speculative = [runtime, draft, sampling] {
        if (!runtime || !runtime->has_value()) throw std::logic_error("DFlash2 target is unavailable");
        auto& target = runtime->value();
        const auto& config = draft->config();
        if (config.hidden != target.config().hidden_size || config.vocabulary != target.config().vocab_size ||
            config.target_layers.back() >= target.layer_count())
            throw std::invalid_argument("DFlash2 checkpoint is incompatible with target geometry");
        draft->track_weights();
        target.set_hidden_capture_layers(config.target_layers);
        draft->prepare([&](const auto& hidden) { return target.project_hidden_logits(hidden); });
        if (auto* preparation = MlxKernelPreparation::current()) preparation->collect([&] {
            for (const auto dtype : {mlx::core::float16, mlx::core::bfloat16, mlx::core::float32}) {
                const auto logits = mlx::core::zeros({8, config.vocabulary}, dtype);
                preparation->add(sample_greedy(logits));
                preparation->add(sample(logits, sampling, mlx::core::full({8}, .5f, mlx::core::float32)));
            }
        });
        target.set_hidden_capture_layers({});
    };
    components.speculative_generate = [runtime, draft](const auto& prompt, const auto& sampling,
        MlxGenerationJob& job, const auto& constraint, int chunk_size) {
        if (!mlx_dflash2_sampling_supported(sampling, constraint)) return false;
        if (!runtime || !runtime->has_value()) throw std::logic_error("DFlash2 target is unavailable");
        auto& target = runtime->value();
        target.reset_generation_state();
        MlxDFlash2GenerationStats stats;
        MlxSamplingParams parameters;
        parameters.temperature = sampling.temperature; parameters.top_k = sampling.top_k;
        parameters.top_p = sampling.top_p; parameters.seed = sampling.seed;
        draft->generate(target, prompt, parameters, sampling.max_tokens,
            [&](auto token) { MlxResidentBudgetScope::reserve(0); return job.token(token); },
            [&](auto tokens, auto ms) { MlxResidentBudgetScope::reserve(0); job.prefill({tokens, ms, 0, ms}); },
            chunk_size, std::nullopt, &stats, sampling.mtp_max_draft_tokens);
        job.mtp = {};
        job.mtp.available = true;
        job.mtp.used = stats.cycles > 0;
        job.mtp.cycles = stats.cycles;
        job.mtp.drafted_tokens = stats.verified;
        job.mtp.accepted_tokens = stats.accepted;
        job.mtp.depth_cycles = stats.depth_cycles;
        job.mtp.position_drafted = stats.position_verified;
        job.mtp.position_accepted = stats.position_accepted;
        job.mtp.selected_depth = sampling.mtp_max_draft_tokens;
        job.mtp.standard_tokens = stats.standard_tokens;
        target.record_mtp_stats({true, job.mtp.used, stats.cycles, stats.verified, stats.accepted,
            stats.depth_cycles, stats.position_verified, stats.position_accepted, {},
            job.mtp.selected_depth, stats.standard_tokens, 0, 0});
        return true;
    };
    components.mtp_available = true;
}

}
