#pragma once

#include <cstdint>
#include <string>
#include <vector>

enum class TensorParallelAxis;
enum class InputGateMode : int;
struct CudaExecutionContext;
namespace tyloi::cuda { class Program; }

/** Compare a native projection with its decoded dense reference. */
int run_linear_check(tyloi::cuda::Program& program, const std::string& name, int rows, InputGateMode gate_mode, int repetitions);
/** Compare CPU and CUDA execution of the same projection. */
int run_cpu_linear_check(tyloi::cuda::Program& program, const std::string& name, int rows, InputGateMode gate_mode, int repetitions);
/** Compare a tensor-parallel projection with its unsharded result. */
int run_tensor_parallel_linear_check(tyloi::cuda::Program& program, const std::string& name, TensorParallelAxis axis, int rows);
std::vector<std::string> parse_tensor_names(const std::string& value);
/** Compare grouped projections, graph replay and decoded dense outputs. */
int run_linear_group_check(tyloi::cuda::Program& program, const std::string& names, int rows, int repetitions);
/** Compare NINT8-zero embedding lookup with decoded rows. */
int run_q8_embedding_check(tyloi::cuda::Program& program, const std::string& name);
/** Compare DeepSeek output-A groupwise and full projection paths. */
int run_dsv4_output_a_check(tyloi::cuda::Program& program, const std::string& name, int batch, int repetitions);
/** Compare Gemma GeGLU activation and output paths. */
int run_gemma_geglu_check(tyloi::cuda::Program& program, int layer, int repetitions);
int run_expert_parallel_moe_check(CudaExecutionContext& execution, const std::string& model_path, const std::string& tensor_name, int tokens, int routes);
int run_mfe_tensor_check(CudaExecutionContext& execution, const std::string& model_path, const std::string& tensor_name, int tokens, int routes, int repetitions, int split_width, bool routed_input, bool benchmark_only);
int run_moe_check(CudaExecutionContext& execution, const std::string& model_path, const std::string& config_path, int layer, const std::vector<std::int64_t>& token_sizes, int repetitions);
int run_attention_decode_check(int length, int repetitions, int dimension, bool sliding, int window);
int run_gemma4_swa_check(int repetitions);
int run_glm_dsa_check(int repetitions);
int run_dsv4_hc_check(int repetitions);
int run_dsv4_attention_check(int repetitions);
int run_text_session_state_check();
