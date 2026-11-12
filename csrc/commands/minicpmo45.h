#pragma once

#include <cstdint>
#include <string>
#include <string_view>

struct CudaExecutionContext;

namespace tyloi::cuda {
struct CudaLoadOptions;
class Program;
namespace commands { class ArgCursor; }

namespace minicpmo45 {

struct CommandOptions {
    std::string input_prefix, output_prefix;
    std::string duplex_input_prefix, duplex_output_prefix;
    std::int64_t tts_steps = 0;
    std::int64_t duplex_steps = 0;
    std::int64_t duplex_max_speak_tokens = 20;
    std::int64_t duplex_seed = 0;
    bool duplex_greedy = false;
};

bool parse_command_option(
    std::string_view option,
    commands::ArgCursor& args,
    CommandOptions& result);
void validate_command_options(const CommandOptions& options);

#ifdef TYLOINFER_NATIVE_CUDA
/** Run the MiniCPM-o composite diagnostic using an owned native program. */
int run_composite(Program& program, const CommandOptions& options);
/** Run a MiniCPM-o duplex diagnostic using an owned native program. */
int run_duplex(Program& program, const CommandOptions& options);
/** Run the MiniCPM-o JSONL batch evaluator using an owned native program. */
int run_eval_batch(Program& program, std::int64_t context_size, std::int64_t vision_batch_size);
#else
int run_composite(
    CudaExecutionContext& execution,
    const CudaLoadOptions& load,
    const CommandOptions& options);
int run_duplex(
    CudaExecutionContext& execution,
    const CudaLoadOptions& load,
    const CommandOptions& options);
#endif

} // namespace minicpmo45
} // namespace tyloi::cuda
