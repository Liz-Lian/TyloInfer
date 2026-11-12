#pragma once

#include <cstdint>
#include <string>

namespace tyloi::cuda { class Program; }

/** Export the native gated-delta output and updated state for fixed F32 inputs. */
int run_gdn_operator_check(tyloi::cuda::Program& program, const std::string& input_dir,
    const std::string& output_path, const std::string& state_path, std::int64_t tokens,
    std::int64_t query_heads, std::int64_t value_heads, std::int64_t head_dim);
/** Export native linear-convolution outputs and state for fixed F32 inputs. */
int run_linear_conv_operator_check(tyloi::cuda::Program& program, const std::string& input_dir,
    const std::string& output_dir, std::int64_t tokens, std::int64_t query_heads,
    std::int64_t value_heads, std::int64_t key_dim, std::int64_t value_dim,
    std::int64_t kernel_size, double eps);
