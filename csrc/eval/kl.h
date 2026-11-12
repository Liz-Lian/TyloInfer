#pragma once

#include "kl_reference_contract.h"

#include <cstdint>
#include <string>

namespace tyloi::cuda { class Program; }

enum class KlEvaluator {
    Legacy,
    Optimized,
};

/** Return the stable name written by the existing evaluator logs. */
const char* kl_evaluator_name(KlEvaluator evaluator);

/** Score complete context batches with the original reference geometry and metrics. */
int run_kl_eval_batched(
    tyloi::cuda::Program& program,
    const std::string& reference_path,
    int max_chunks,
    std::int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract& reference_contract);

/** Run the selected resident evaluator using the program's model and request state. */
int run_selected_kl_eval(
    tyloi::cuda::Program& program,
    const std::string& reference_path,
    int max_chunks,
    KlEvaluator evaluator,
    std::int64_t requested_n_batch,
    int score_override,
    const KlReferenceContract& reference_contract);

/** Score DeepSeek V4 layer groups through pinned F16 hidden storage and optional logits output. */
int run_kl_eval_streamed(
    tyloi::cuda::Program& program,
    const std::string& reference_path,
    const std::string& logits_output_path,
    int max_chunks,
    int layer_group,
    int chunk_batch,
    int score_override,
    const KlReferenceContract& reference_contract);
