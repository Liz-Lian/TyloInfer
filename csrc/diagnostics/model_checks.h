#pragma once

#include "diagnostics/experts.h"
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
int run_moe_check(CudaExecutionContext& execution, const std::string& model_path, const std::string& config_path, int layer, const std::vector<std::int64_t>& token_sizes, int repetitions);
/** Compare native full or sliding-window decode attention. */
int run_attention_decode_check(tyloi::cuda::Program& program, int length, int repetitions, int dimension, bool sliding, int window);
/** Check native Gemma sliding-window and full attention paths. */
int run_gemma4_swa_check(tyloi::cuda::Program& program, int repetitions);
/** Check native GLM DSA attention and routing paths. */
int run_glm_dsa_check(tyloi::cuda::Program& program, int repetitions);
/** Compare DeepSeek V4 hyper-connection operations. */
int run_dsv4_hc_check(tyloi::cuda::Program& program, int repetitions);
/** Check DeepSeek V4 attention, compression and routing paths. */
int run_dsv4_attention_check(tyloi::cuda::Program& program, int repetitions);
/** Exercise exact synthetic snapshot restoration through the production request states. */
int run_text_session_state_check(tyloi::cuda::Program& program);
