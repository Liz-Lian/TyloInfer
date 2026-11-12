#pragma once

#include <string>

namespace tyloi::cuda { class Program; }

/** Report the configured model source's shard, tensor and asset counts. */
int run_container_check(tyloi::cuda::Program& program);

/** Check the embedded tokenizer against model configuration and an optional external GGUF. */
int run_assets_check(tyloi::cuda::Program& program, const std::string& config_path,
    const std::string& tokenizer_model, const std::string& text);
