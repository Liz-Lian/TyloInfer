#include "run.h"
#include "engine/models/source/model_config.h"
#include "env.h"
#include "engine/weights/loader.h"
#include "engine/weights/cache.h"
#include "cli.h"
#include "engine/models/prepare.h"
#include "diagnostics/backend_checks.h"
#include "diagnostics/model_checks.h"
#include "diagnostics/operator_checks.h"
#include "diagnostics/trace.h"
#include "diagnostics/engines.h"
#include "diagnostics/source.h"
#include "diagnostics/text.h"
#include "diagnostics/vision.h"
#include "diagnostics/mtp_bench.h"
#include "diagnostics/mtp.h"
#include "diagnostics/prefill.h"
#include "diagnostics/qwen_mtp.h"
#include "diagnostics/batch.h"
#include "diagnostics/tokens.h"
#include "diagnostics/qwen_decode.h"
#include "diagnostics/qwen_variants.h"
#include "diagnostics/compare.h"
#include "diagnostics/policy.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/ops/projection/mmq.h"
#include "minicpmo45.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/kernels/deepseek_v41.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tyloi::cuda {

struct DiagnosticsCommandOptions : CudaLoadOptions, TokenInputOptions {
    std::string check_linear, check_linear_cpu, check_linear_gate;
    std::string check_tp_linear, check_ep_moe;
    std::string check_tp_axis_arg = "output";
    std::string check_linear_group, check_q8_embedding;
    std::string check_gdn_input, check_gdn_output, check_gdn_state;
    std::string check_linear_conv_input, check_linear_conv_output;
    std::string check_mfe_tensor, check_dsv4_output_a;
    std::string prefill_sweep_arg;
    std::string check_moe_tokens = "1,2,4,8,16,32,64,128,256";
    std::string block_trace_reference, block_trace_output;
    std::string check_tokenizer_text =
        "MFQ tokenizer check: hello, world! <think>";
    std::string bench_qwen35_mtp;
    std::string kl_mmq_arg = "default";
    int bench_qwen35_mtp_reps = 3;
    int64_t minicpmo_eval_vision_batch_size = 16;
    int64_t block_trace_start = 0;
    int64_t block_trace_count = 0;
    int prefill_repeat = 0;
    int prefill_sweep_reps = 5;
    int check_linear_m = 1;
    int check_linear_reps = 200;
    int check_tp_m = 1;
    int check_ep_moe_tokens = 1;
    int check_ep_moe_routes = 2;
    int check_gdn_tokens = 512;
    int check_gdn_q_heads = 16;
    int check_gdn_v_heads = 32;
    int check_gdn_head_dim = 128;
    int check_linear_conv_tokens = 512;
    int check_linear_conv_q_heads = 16;
    int check_linear_conv_v_heads = 32;
    int check_linear_conv_key_dim = 128;
    int check_linear_conv_value_dim = 128;
    int check_linear_conv_kernel = 4;
    int check_gemma_geglu_layer = -1;
    int check_gemma_geglu_reps = 100;
    int check_moe_layer = -1;
    int check_moe_reps = 100;
    int check_mfe_tokens = 1;
    int check_mfe_routes = 2;
    int check_mfe_reps = 100;
    int check_mfe_split_width = 0;
    int check_dsv4_output_a_batch = 1;
    int check_dsv4_output_a_reps = 200;
    int check_attention_decode = 0;
    int check_attention_reps = 200;
    int check_attention_head_dim = 256;
    int check_attention_window = 4096;
    int compare_mma_decode_steps = 1;
    int compare_mma_decode_planned_len = 0;
    bool profile = false;
    bool ignore_eos = false;
    bool compare_ffn_transfer_phases = false;
    bool compare_router_lookahead = false;
    bool compare_rotary_fusion = false;
    bool compare_attention_grouping = false;
    bool compare_mhc_fusion = false;
    bool compare_mhc_shared_norm = false;
    bool compare_mhc_dense_fusion = false;
    bool compare_mhc_packed_groups = false;
    bool compare_mhc_fixed_formats = false;
    bool compare_mhc_async_normalized = false;
    bool compare_gdn_fixed_geometry = false;
    bool compare_gdn_inplace_state = false;
    bool compare_qsa_fusion = false;
    bool compare_mfe_parallel_gu = false;
    bool compare_mfe_gpu_bundle = false;
    bool compare_mfe_default_math = false;
    bool compare_mfe_e8_narrow = false;
    bool compare_mfe_shared_first = false;
    bool compare_mfe_early_gu = false;
    bool compare_cpu_transfer_budget = false;
    bool compare_cpu_background_calibration = false;
    bool compare_router_topk = false;
    bool compare_route_poll = false;
    bool compare_window_flush = false;
    bool compare_early_no_cpu_ready = false;
    bool compare_gpu_only_cpu_wait = false;
    bool compare_fixed_gpu_timing = false;
    bool compare_window_input_views = false;
    bool compare_route_spin = false;
    bool compare_moe_plan_reuse = false;
    bool compare_shared_gate_fusion = false;
    bool compare_graph_optimizations = false;
    int comparison_rounds = 1;
    bool comparison_phase_timings = false;
    bool compare_host_pin = false;
    bool compare_resident_overlap = false;
    bool compare_dma_graph_batch = false;
    bool compare_ram_field_packing = false;
    bool compare_warp_multi_sum = false;
    int warmup_prefill = -1;
    std::string router_lookahead_audit;
    std::string decode_inputs_file;
    std::vector<int64_t> decode_inputs;
    bool check_backend_bf16_add = false;
    bool check_backend_argmax = false;
    bool compare_mma_attention = false;
    bool compare_decode_splitk = false;
    bool compare_mma_decode = false;
    bool compare_nvq_vec4 = false;
    bool check_gemma4_swa = false;
    bool check_glm_dsa = false;
    bool check_dsv4_attention = false;
    bool check_dsv4_hc = false;
    bool check_deepseek_v41 = false;
    bool check_text_session_state = false;
    bool check_qwen35_mtp = false;
    bool check_continuous_batching = false;
    int check_continuous_batching_requests = 2;
    bool check_engine_isolation = false;
    bool check_flash_next = false;
    bool check_qwen4_decode_hidden = false;
    bool check_flash_next_mtp = false;
    bool check_nemotron_h = false;
    bool check_nemotron_h_mtp = false;
    bool check_nemotron_h_vision = false;
    bool compare_dsv4_hc_ops = false;
    bool compare_dsv4_hc_model = false;
    bool check_attention_swa_decode = false;
    bool check_mfe_routed_input = false;
    bool check_mfe_benchmark_only = false;
    bool check_runtime_assets = false;
    bool check_mfq_container = false;
    bool minicpmo_eval_batch = false;

    int runtime_comparison_modes() const {
        return int(compare_ffn_transfer_phases) + int(compare_router_lookahead) +
               int(compare_rotary_fusion) + int(compare_attention_grouping) + int(compare_mhc_fusion) + int(compare_mhc_shared_norm) + int(compare_mhc_dense_fusion) + int(compare_mhc_packed_groups) + int(compare_mhc_fixed_formats) + int(compare_mhc_async_normalized) + int(compare_gdn_fixed_geometry) + int(compare_gdn_inplace_state) + int(compare_qsa_fusion) + int(compare_mfe_parallel_gu) + int(compare_mfe_gpu_bundle) + int(compare_mfe_default_math) + int(compare_mfe_e8_narrow) + int(compare_mfe_shared_first) + int(compare_mfe_early_gu) + int(compare_cpu_transfer_budget) + int(compare_cpu_background_calibration) +
               int(compare_router_topk) + int(compare_route_poll) + int(compare_window_flush) + int(compare_early_no_cpu_ready) + int(compare_gpu_only_cpu_wait) + int(compare_fixed_gpu_timing) + int(compare_window_input_views) + int(compare_route_spin) + int(compare_moe_plan_reuse) + int(compare_shared_gate_fusion) + int(compare_graph_optimizations) + int(compare_host_pin) + int(compare_resident_overlap) + int(compare_dma_graph_batch) + int(compare_ram_field_packing) + int(compare_warp_multi_sum);
    }
    /** Resolve only comparisons backed by current production computation. */
    std::optional<diagnostics::Policy> selected_policy() const {
        using Policy = diagnostics::Policy;
        static constexpr std::pair<bool DiagnosticsCommandOptions::*, Policy> flags[] = {
            {&DiagnosticsCommandOptions::compare_route_poll, Policy::route_poll},
            {&DiagnosticsCommandOptions::compare_window_flush, Policy::window_flush},
            {&DiagnosticsCommandOptions::compare_window_input_views, Policy::window_input_views},
            {&DiagnosticsCommandOptions::compare_route_spin, Policy::route_spin},
            {&DiagnosticsCommandOptions::compare_host_pin, Policy::host_pin},
            {&DiagnosticsCommandOptions::compare_shared_gate_fusion, Policy::shared_gate_fusion},
            {&DiagnosticsCommandOptions::compare_router_topk, Policy::router_topk},
            {&DiagnosticsCommandOptions::compare_warp_multi_sum, Policy::warp_multi_sum},
            {&DiagnosticsCommandOptions::compare_mhc_shared_norm, Policy::mhc_shared_norm},
            {&DiagnosticsCommandOptions::compare_mhc_dense_fusion, Policy::mhc_dense_fusion},
            {&DiagnosticsCommandOptions::compare_mhc_packed_groups, Policy::mhc_packed_groups},
            {&DiagnosticsCommandOptions::compare_mhc_fixed_formats, Policy::mhc_fixed_formats},
            {&DiagnosticsCommandOptions::compare_mhc_async_normalized, Policy::mhc_async_normalized},
            {&DiagnosticsCommandOptions::compare_gdn_fixed_geometry, Policy::gdn_fixed_geometry},
            {&DiagnosticsCommandOptions::compare_gdn_inplace_state, Policy::gdn_inplace_state},
            {&DiagnosticsCommandOptions::compare_qsa_fusion, Policy::qsa_fusion},
            {&DiagnosticsCommandOptions::compare_rotary_fusion, Policy::rotary_fusion},
            {&DiagnosticsCommandOptions::compare_attention_grouping, Policy::attention_grouping},
            {&DiagnosticsCommandOptions::compare_mhc_fusion, Policy::mhc_fusion},
        };
        for (const auto& [flag, policy] : flags) if (this->*flag) return policy;
        return std::nullopt;
    }
};

} // namespace tyloi::cuda

using namespace tyloi::cuda::internal;

using namespace tyloi::cuda::diagnostics;

namespace {
struct DiagnosticsCommand : tyloi::cuda::DiagnosticsCommandOptions {
    explicit DiagnosticsCommand(tyloi::cuda::DiagnosticsCommandOptions options)
        : tyloi::cuda::DiagnosticsCommandOptions(std::move(options)) {}
    /** Resolve model-mode context and validate combined diagnostic options. */
    std::vector<int64_t> prepare_model() {
        if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
        if (!bench_qwen35_mtp.empty()) {
            TYLOINFER_RUNTIME_CHECK(bench_qwen35_mtp == "ordinary" || bench_qwen35_mtp == "mtp",
                "--bench-qwen35-mtp expects ordinary or mtp");
            TYLOINFER_RUNTIME_CHECK(!check_qwen35_mtp && ids_arg.empty() && ids_file.empty() &&
                prefill_sweep_arg.empty(), "MTP benchmark cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            TYLOINFER_RUNTIME_CHECK(gen >= 2 && context_size >= static_cast<int64_t>(gen) + 17 &&
                bench_qwen35_mtp_reps > 0 && bench_qwen35_mtp_reps <= 100,
                "MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");
            std::cout << std::unitbuf;
        }
        if (check_nemotron_h_vision) {
            if (context_size == 0) context_size = 128;
            std::cout << std::unitbuf;
        }
        if (check_nemotron_h_mtp) {
            if (context_size == 0) context_size = 128;
            if (context_size < 64)
                throw std::runtime_error(
                    "Nemotron-H MTP correctness gate requires --ctx-size >= 64");
            std::cout << std::unitbuf;
        }
        if (check_qwen35_mtp) {
            if (context_size == 0) context_size = 512;
            if (context_size < 64) throw std::runtime_error("MTP correctness gate requires --ctx-size >= 64");
            std::cout << std::unitbuf;
        }
        if (check_continuous_batching) {
            TYLOINFER_RUNTIME_CHECK(
                !check_qwen35_mtp &&
                !check_flash_next && !check_flash_next_mtp &&
                bench_qwen35_mtp.empty() && ids_arg.empty() &&
                ids_file.empty() &&
                prefill_sweep_arg.empty(),
                "continuous batching check cannot combine execution modes");
            if (context_size == 0) context_size = 512;
            if (context_size < 32) {
                throw std::runtime_error(
                    "continuous batching check requires --ctx-size >= 32");
            }
            std::cout << std::unitbuf;
        }
        std::vector<int64_t> prefill_sweep_sizes;
        if (!prefill_sweep_arg.empty()) {
            prefill_sweep_sizes = parse_ids(prefill_sweep_arg);
            if (std::any_of(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end(),
                            [](int64_t value) { return value <= 0; })) {
                throw std::runtime_error("--prefill-sweep values must be positive");
            }
            if (context_size == 0) {
                context_size = *std::max_element(prefill_sweep_sizes.begin(), prefill_sweep_sizes.end());
            }
        }
        if ((!block_trace_reference.empty() || !block_trace_output.empty()) &&
                context_size == 0) {
            context_size = (int64_t)(
                ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file)).size();
        }
        return prefill_sweep_sizes;
    }
    /** Select the optional weights used by the original model-mode combination. */
    bool needs_parts() const {
        return check_qwen35_mtp || check_flash_next_mtp || check_nemotron_h_mtp ||
            check_nemotron_h_vision || !bench_qwen35_mtp.empty() || (check_qwen4_decode_hidden && profile);
    }
    int run() { return with_command_errors([&]() -> int {
        if (check_engine_isolation) {
            tyloi::cuda::CudaEngineOptions options;
            static_cast<tyloi::cuda::CudaLoadOptions&>(options) =
                static_cast<const tyloi::cuda::CudaLoadOptions&>(*this);
            return run_cuda_engine_isolation_check(std::move(options));
        }
        if (check_backend_bf16_add || check_backend_argmax) {
            auto program = tyloi::cuda::Program::bare();
            auto scope = program->executor().enter(program->config().layer_placement.primary_device());
            if (check_backend_bf16_add) return run_backend_bf16_add_check(*program, 4096, 10000);
            return run_backend_argmax_check(*program, 248320, 2000);
        }
        enum class Tool { linear_cpu, linear, tp, ep, group, q8, gdn, conv, geglu, container, assets, moe,
            mfe, output_a, attention, swa, glm, dsv4_attention, hc, dsv41, policy, session, minicpmo, flash, nemo, vision, nemo_mtp, flash_mtp, qwen_mtp, qwen_decode, qwen_variants, batch, mtp_bench, prefill, hc_compare, trace, compare, tokens };
        Tool selected = Tool::tokens;
        if (!check_linear_cpu.empty()) selected = Tool::linear_cpu;
        else if (!check_linear.empty()) selected = Tool::linear;
        else if (!check_tp_linear.empty()) selected = Tool::tp;
        else if (!check_ep_moe.empty()) selected = Tool::ep;
        else if (!check_linear_group.empty()) selected = Tool::group;
        else if (!check_q8_embedding.empty()) selected = Tool::q8;
        else if (!check_gdn_input.empty()) selected = Tool::gdn;
        else if (!check_linear_conv_input.empty()) selected = Tool::conv;
        else if (check_gemma_geglu_layer >= 0) selected = Tool::geglu;
        else if (check_mfq_container) selected = Tool::container;
        else if (check_runtime_assets) selected = Tool::assets;
        else if (check_moe_layer >= 0) selected = Tool::moe;
        else if (!check_mfe_tensor.empty()) selected = Tool::mfe;
        else if (!check_dsv4_output_a.empty()) selected = Tool::output_a;
        else if (check_attention_decode > 0) selected = Tool::attention;
        else if (check_gemma4_swa) selected = Tool::swa;
        else if (check_glm_dsa) selected = Tool::glm;
        else if (check_dsv4_attention) selected = Tool::dsv4_attention;
        else if (check_dsv4_hc) selected = Tool::hc;
        else if (check_deepseek_v41) selected = Tool::dsv41;
        else if (check_text_session_state) selected = Tool::session;
        else if (minicpmo_eval_batch) selected = Tool::minicpmo;
        else if (check_continuous_batching) selected = Tool::batch;
        else if (check_qwen4_decode_hidden) {
            const auto* variants = tyloi::cuda::env_lookup("TYLOINFER_MTP_GENERATION_VARIANTS");
            selected = profile && variants && variants[0] == '1' ? Tool::qwen_variants : Tool::qwen_decode;
        }
        else if (check_flash_next) selected = Tool::flash;
        else if (check_nemotron_h) selected = Tool::nemo;
        else if (check_nemotron_h_vision) selected = Tool::vision;
        else if (check_nemotron_h_mtp) selected = Tool::nemo_mtp;
        else if (check_flash_next_mtp) selected = Tool::flash_mtp;
        else if (check_qwen35_mtp) selected = Tool::qwen_mtp;
        else if (!bench_qwen35_mtp.empty()) selected = Tool::mtp_bench;
        else if (!prefill_sweep_arg.empty()) selected = Tool::prefill;
        else if (compare_dsv4_hc_ops || compare_dsv4_hc_model) selected = Tool::hc_compare;
        else if (!check_continuous_batching && !check_qwen4_decode_hidden && !check_flash_next &&
                 !check_nemotron_h && !check_nemotron_h_vision && !check_nemotron_h_mtp &&
                 !check_flash_next_mtp && !check_qwen35_mtp && bench_qwen35_mtp.empty() &&
                 prefill_sweep_arg.empty() && !compare_dsv4_hc_ops && !compare_dsv4_hc_model &&
                 (!block_trace_reference.empty() || !block_trace_output.empty())) selected = Tool::trace;
        else if (runtime_comparison_modes()) selected = Tool::policy;
        else if (compare_decode_splitk || compare_mma_decode || compare_nvq_vec4 ||
                 prefill_repeat > 0 || compare_mma_attention) selected = Tool::compare;
        else if (!runtime_comparison_modes() && !comparison_phase_timings && router_lookahead_audit.empty() &&
                 !compare_decode_splitk && !compare_mma_decode && !compare_nvq_vec4 &&
                 prefill_repeat <= 0 && !compare_mma_attention) selected = Tool::tokens;

        {
            if (comparison_phase_timings && !runtime_comparison_modes())
                throw std::runtime_error("comparison phase timings require a runtime comparison");
            if (runtime_comparison_modes() && selected != Tool::policy)
                throw std::runtime_error("runtime comparison cannot combine execution modes");
            if (!router_lookahead_audit.empty())
                throw std::runtime_error("router lookahead audit has no current CUDA producer");
            const auto policy = selected_policy();
            if (selected == Tool::policy) {
                if (runtime_comparison_modes() != 1 || profile)
                    throw std::runtime_error("runtime comparison requires one policy and profiling disabled");
                if (!policy) throw std::runtime_error("the selected runtime comparison uses retired CUDA scheduling policies");
            }
            auto options = static_cast<const tyloi::cuda::CudaLoadOptions&>(*this);
            std::vector<int64_t> prefill_sizes;
            if (selected == Tool::flash || selected == Tool::nemo || selected == Tool::vision ||
                    selected == Tool::nemo_mtp || selected == Tool::flash_mtp || selected == Tool::qwen_mtp ||
                    selected == Tool::batch || selected == Tool::mtp_bench || selected == Tool::prefill ||
                    selected == Tool::qwen_decode || selected == Tool::qwen_variants ||
                    selected == Tool::hc_compare || selected == Tool::tokens || selected == Tool::compare || selected == Tool::policy) {
                if (model_path.empty()) {
                    std::cerr << "missing model or execution mode (see --help)\n";
                    return 2;
                }
                prefill_sizes = prepare_model();
                options.context_size = context_size;
            }
            std::vector<int64_t> trace_ids;
            if (selected == Tool::trace || selected == Tool::hc_compare || selected == Tool::tokens ||
                    selected == Tool::compare || selected == Tool::policy || selected == Tool::qwen_variants || (selected == Tool::qwen_decode && profile)) {
                if (context_size < 0) throw std::runtime_error("--ctx-size must be positive");
                trace_ids = ids_file.empty() ? parse_ids(ids_arg) : load_ids_file(ids_file);
                if (selected == Tool::trace && options.context_size == 0)
                    options.context_size = static_cast<int64_t>(trace_ids.size());
            }
            if (selected == Tool::batch)
                return run_continuous_batching_check(std::move(options), check_continuous_batching_requests);
            if (selected == Tool::minicpmo && options.context_size <= 0) options.context_size = 8192;
            const auto loaded_context = options.context_size;
            auto program = tyloi::cuda::Program::open(std::move(options), selected != Tool::policy);
            auto scope = program->executor().enter(program->config().layer_placement.primary_device());
            if (selected == Tool::tokens || selected == Tool::compare || selected == Tool::policy) {
                program->executor().profiler().enabled = false;
                auto loader = program->loader();
                if (selected == Tool::policy) {
                    std::cout << std::unitbuf << policy_name(*policy) << "_setup\n";
                    program->weights().report(std::cout);
                }
                const auto started = std::chrono::steady_clock::now();
                program->load(needs_parts());
                const auto loaded = std::chrono::steady_clock::now();
                auto ids = tyloi::ops::unsqueeze(tyloi::ops::tensor(trace_ids,
                    program->executor().device()), 0);
                if (selected == Tool::policy) {
                    PolicyOptions run{*policy};
                    run.rounds = comparison_rounds; run.warmup = warmup_prefill; run.gen = gen;
                    run.phase_timings = comparison_phase_timings; run.forced = decode_inputs;
                    run.load_start = started; run.load_end = loaded;
                    return run_policy(*program, std::move(ids), run);
                }
                if (selected == Tool::compare) {
                    tyloi::cuda::diagnostics::warmup_prefill(*program, ids, profile, warmup_prefill);
                    program->executor().profiler().enabled = profile;
                    if (compare_decode_splitk) return tyloi::cuda::diagnostics::compare_decode_splitk(*program, std::move(ids));
                    if (compare_mma_decode) return tyloi::cuda::diagnostics::compare_mma_decode(*program,
                        std::move(ids), compare_mma_decode_steps, compare_mma_decode_planned_len);
                    if (compare_nvq_vec4)
                        throw std::runtime_error("NVQ vec4 comparison has no live kernel selector");
                    if (prefill_repeat > 0) {
                        const auto* trace = tyloi::cuda::env_lookup("TYLOINFER_CHECK_PREFILL_REPEAT_TRACE");
                        return repeat_prefill(*program, std::move(ids), prefill_repeat, trace && std::atoi(trace) != 0);
                    }
                    return tyloi::cuda::diagnostics::compare_mma_attention(*program, std::move(ids));
                }
                std::unordered_set<int64_t> stops;
                if (!ignore_eos && tokenizer_model.empty() && decode_inputs.empty()) {
                    const auto* source = dynamic_cast<const tyloi::ModelSource*>(&loader.source);
                    if (!source) throw std::runtime_error("diagnostics model source is unavailable");
                    stops = tyloi::models::stop_token_ids(
                        tyloi::models::config_json(*source, config_path), program->model().info().vocab_size);
                    if (stops.empty())
                        std::cerr << "warning: no model EOS token IDs; generation is limited only by --gen\n";
                }
                return generate_diagnostic_tokens(*program, std::move(ids), gen, profile,
                    started, loaded, decode_inputs, stops, false, TokenOptions{.warmup=warmup_prefill});
            }
            if (selected == Tool::dsv41) return run_backend_dsv41_check(*program);
            if (selected == Tool::attention) return run_attention_decode_check(*program,
                check_attention_decode, check_attention_reps, check_attention_head_dim,
                check_attention_swa_decode, check_attention_window);
            if (selected == Tool::swa) return run_gemma4_swa_check(*program, check_attention_reps);
            if (selected == Tool::glm) return run_glm_dsa_check(*program, check_attention_reps);
            if (selected == Tool::dsv4_attention) return run_dsv4_attention_check(*program, check_attention_reps);
            if (selected == Tool::hc) return run_dsv4_hc_check(*program, check_attention_reps);
            if (selected == Tool::session) return run_text_session_state_check(*program);
            if (selected == Tool::flash || selected == Tool::nemo || selected == Tool::vision ||
                    selected == Tool::nemo_mtp || selected == Tool::flash_mtp || selected == Tool::qwen_mtp ||
                    selected == Tool::qwen_decode || selected == Tool::qwen_variants || selected == Tool::mtp_bench || selected == Tool::prefill) {
                program->load(needs_parts());
                if (selected == Tool::qwen_variants) return run_qwen4_generation_variants(*program, trace_ids);
                if (selected == Tool::qwen_decode) return profile
                    ? run_qwen4_mtp_profile(*program, trace_ids) : run_qwen4_decode_hidden_check(*program);
                if (selected == Tool::prefill) {
                    const auto status = run_prefill_sweep(*program, prefill_sizes, prefill_sweep_reps, profile);
                    program->weights().report(std::cout);
                    return status;
                }
                if (selected == Tool::nemo_mtp) return run_nemotron_h_mtp_check(*program);
                if (selected == Tool::flash_mtp) return run_flash_next_mtp_check(*program);
                if (selected == Tool::qwen_mtp) return run_qwen35_mtp_check(*program);
                if (selected == Tool::mtp_bench) return run_qwen35_mtp_bench(*program,
                    bench_qwen35_mtp == "mtp", gen, bench_qwen35_mtp_reps);
                if (selected == Tool::vision) return run_nemotron_h_vision_check(*program);
                return selected == Tool::flash ? run_flash_next_check(*program) : run_nemotron_h_check(*program);
            }
            if (selected == Tool::trace || selected == Tool::hc_compare) {
                if (model_path.empty()) {
                    std::cerr << "missing model or execution mode (see --help)\n";
                    return 2;
                }
                program->load(selected == Tool::hc_compare && needs_parts());
                auto ids = tyloi::ops::unsqueeze(tyloi::ops::tensor(trace_ids,
                    tyloi::Device(tyloi::kCUDA, program->config().layer_placement.primary_device())), 0);
                if (selected == Tool::hc_compare)
                    return run_hc_compare(*program, std::move(ids), compare_dsv4_hc_ops);
                if (!block_trace_reference.empty()) {
                    auto reference = static_cast<const tyloi::cuda::CudaLoadOptions&>(*this);
                    reference.model_path = block_trace_reference;
                    reference.context_size = loaded_context;
                    return run_block_trace_compare(*program, std::move(reference), std::move(ids));
                }
                return run_block_trace_dump(*program, block_trace_output, std::move(ids),
                    block_trace_start, block_trace_count);
            }
            if (selected == Tool::linear_cpu || selected == Tool::linear) {
                if (model_path.empty()) throw std::runtime_error(selected == Tool::linear_cpu
                    ? "--check-linear-cpu requires --model" : "--check-linear requires --model");
                auto gate = InputGateMode::None;
                if (check_linear_gate == "sigmoid") gate = InputGateMode::Sigmoid;
                else if (check_linear_gate == "silu") gate = InputGateMode::Silu;
                else if (!check_linear_gate.empty()) throw std::runtime_error("--check-linear-gate must be sigmoid or silu");
                return selected == Tool::linear_cpu
                    ? run_cpu_linear_check(*program, check_linear_cpu, check_linear_m, gate, check_linear_reps)
                    : run_linear_check(*program, check_linear, check_linear_m, gate, check_linear_reps);
            }
            if (selected == Tool::tp) {
                if (model_path.empty()) throw std::runtime_error("--check-tp-linear requires --model");
                const TensorParallelAxis axis = check_tp_axis_arg == "output" ? TensorParallelAxis::Output
                    : check_tp_axis_arg == "input" ? TensorParallelAxis::Input
                    : throw std::runtime_error("--check-tp-axis must be output or input");
                return run_tensor_parallel_linear_check(*program, check_tp_linear, axis, check_tp_m);
            }
            if (selected == Tool::ep) {
                if (model_path.empty()) throw std::runtime_error("--check-ep-moe requires --model");
                return run_expert_parallel_moe_check(*program, check_ep_moe, check_ep_moe_tokens, check_ep_moe_routes);
            }
            if (selected == Tool::mfe) {
                if (model_path.empty()) throw std::runtime_error("--check-mfe-tensor requires --model");
                KlMmqScope mmq(parse_kl_mmq_mode(kl_mmq_arg), program->executor().mmq());
                const auto names = parse_tensor_names(check_mfe_tensor);
                if (moe_gpu_cache_gb > 0 && names.size() != 1)
                    throw std::runtime_error("cached --check-mfe-tensor accepts one tensor per invocation");
                for (const auto& name : names) {
                    const int result = run_mfe_tensor_check(*program, name, check_mfe_tokens,
                        check_mfe_routes, check_mfe_reps, check_mfe_split_width,
                        check_mfe_routed_input, check_mfe_benchmark_only);
                    if (result != 0) return result;
                }
                return 0;
            }
            if (selected == Tool::group) {
                if (model_path.empty()) throw std::runtime_error("--check-linear-group requires --model");
                return run_linear_group_check(*program, check_linear_group, check_linear_m, check_linear_reps);
            }
            if (selected == Tool::q8) {
                if (model_path.empty()) throw std::runtime_error("--check-q8-embedding requires --model");
                return run_q8_embedding_check(*program, check_q8_embedding);
            }
            if (selected == Tool::gdn) {
                if (check_gdn_output.empty() || check_gdn_state.empty())
                    throw std::runtime_error("--check-gdn-input requires --check-gdn-output and --check-gdn-state");
                return run_gdn_operator_check(*program, check_gdn_input, check_gdn_output, check_gdn_state,
                    check_gdn_tokens, check_gdn_q_heads, check_gdn_v_heads, check_gdn_head_dim);
            }
            if (selected == Tool::conv) {
                if (check_linear_conv_output.empty())
                    throw std::runtime_error("--check-linear-conv-input requires --check-linear-conv-output");
                return run_linear_conv_operator_check(*program, check_linear_conv_input, check_linear_conv_output,
                    check_linear_conv_tokens, check_linear_conv_q_heads, check_linear_conv_v_heads,
                    check_linear_conv_key_dim, check_linear_conv_value_dim, check_linear_conv_kernel, 1.0e-6);
            }
            if (selected == Tool::geglu) {
                if (model_path.empty()) throw std::runtime_error("--check-gemma-geglu-layer requires --model");
                return run_gemma_geglu_check(*program, check_gemma_geglu_layer, check_gemma_geglu_reps);
            }
            if (selected == Tool::moe) {
                if (model_path.empty()) throw std::runtime_error("--check-moe-layer requires --model");
                return run_moe_check(*program, config_path, check_moe_layer,
                    parse_ids(check_moe_tokens), check_moe_reps);
            }
            if (selected == Tool::container) {
                if (model_path.empty()) throw std::runtime_error("--check-mfq-container requires --model");
                return run_container_check(*program);
            }
            if (selected == Tool::assets) {
                if (model_path.empty()) throw std::runtime_error("--check-runtime-assets requires --model");
                return run_assets_check(*program, config_path, tokenizer_model, check_tokenizer_text);
            }
            if (selected == Tool::output_a) {
                if (model_path.empty()) throw std::runtime_error("--check-dsv4-output-a requires --model");
                return run_dsv4_output_a_check(*program, check_dsv4_output_a,
                    check_dsv4_output_a_batch, check_dsv4_output_a_reps);
            }
            if (model_path.empty()) throw std::runtime_error("--minicpmo-eval-batch requires --model");
            if (!ids_arg.empty() || !ids_file.empty())
                throw std::runtime_error("MiniCPM-o eval mode cannot be combined with composite, duplex, token, transport, KL, or prefill modes");
            if (minicpmo_eval_vision_batch_size <= 0)
                throw std::runtime_error("--minicpmo-eval-vision-batch-size must be positive");
            return tyloi::cuda::minicpmo45::run_eval_batch(
                *program, context_size <= 0 ? 8192 : context_size, minicpmo_eval_vision_batch_size);
        }
    }); }
};

} // namespace

namespace tyloi::cuda {
int execute_diagnostics_command(DiagnosticsCommandOptions options) {
    return DiagnosticsCommand(std::move(options)).run();
}
} // namespace tyloi::cuda

namespace tyloi::cuda::commands {
namespace {

void print_diagnostics_help() {
    std::cout
        << "MFQ CUDA diagnostics\n\n"
        << "Usage:\n"
        << "  mfq-diagnostics [--model MODEL] CHECK [OPTIONS]\n\n"
        << "Checks and benchmarks:\n"
        << "  --check-linear NAME             check one GPU linear\n"
        << "  --check-linear-cpu NAME         check one CPU linear\n"
        << "  --check-linear-group NAMES      check a tensor group\n"
        << "  --check-q8-embedding NAME       check a Q8 embedding\n"
        << "  --check-tp-linear NAME          check tensor-parallel linear\n"
        << "  --check-ep-moe NAME             check expert-parallel MoE\n"
        << "  --check-gdn-input PATH          check GDN operator fixtures\n"
        << "  --check-linear-conv-input PATH  check linear-conv fixtures\n"
        << "  --check-moe-layer N             check a model MoE layer\n"
        << "  --check-mfe-tensor NAME         check an MFE tensor\n"
        << "  --check-dsv4-output-a NAME      check DeepSeek output-A\n"
        << "  --check-attention-decode N      check decode attention\n"
        << "  --check-gemma4-swa              check Gemma4 SWA\n"
        << "  --check-glm-dsa                 check GLM DSA\n"
        << "  --check-dsv4-attention          check DeepSeek-V4 attention\n"
        << "  --check-dsv4-hc                 check DeepSeek-V4 HC\n"
        << "  --check-deepseek-v41            run DeepSeek-V4.1 self-check\n"
        << "  --check-text-session-state      check session state\n"
        << "  --check-qwen35-mtp              check Qwen3.5 MTP\n"
        << "  --check-flash-next              check Flash-Next\n"
        << "  --check-qwen4-decode-hidden     check decode graph raw hidden and KV transitions\n"
        << "  --check-flash-next-mtp          check Flash-Next MTP\n"
        << "  --check-nemotron-h              check Nemotron-H text backbone\n"
        << "  --check-nemotron-h-mtp          check Nemotron-H multi-token prediction\n"
        << "  --check-nemotron-h-vision       check RADIO vision tower and projection\n"
        << "  --check-continuous-batching     check continuous batching\n"
        << "  --continuous-batching-check-requests N  check 2..32 concurrent requests\n"
        << "  --check-engine-isolation        check two concurrent CUDA engines\n"
        << "  --check-backend-bf16-add        check backend BF16 add\n"
        << "  --check-backend-argmax          check backend argmax\n"
        << "  --check-runtime-assets          validate embedded assets\n"
        << "  --check-mfq-container           validate the MFQ container\n"
        << "  --minicpmo-eval-batch           run MiniCPM-o batch evaluation\n"
        << "  --bench-qwen35-mtp MODE         benchmark ordinary or mtp mode\n"
        << "  --prefill-sweep LIST            benchmark prefill sizes\n"
        << "  --prefill-repeat N              repeat prefill N times\n"
        << "  --compare-block-trace PATH      compare a block trace\n"
        << "  --dump-block-trace PATH         write a block trace\n"
        << "  --compare-dsv4-hc-ops           compare DeepSeek HC operators\n"
        << "  --compare-dsv4-hc-model         compare DeepSeek HC model\n"
        << "  --compare-mma-attention         compare MMA attention\n"
        << "  --compare-decode-splitk         compare decode split-K\n"
        << "  --compare-mma-decode            compare MMA decode\n"
        << "  --compare-nvq-vec4              compare NVQ vec4\n"
        << "  --profile                       profile token generation\n\n"
        << "Common options:\n"
        << "  --model PATH                    model or split-model shard\n"
        << "  --config PATH                   external model config\n"
        << "  --ids LIST | --ids-file PATH    diagnostic token input\n"
        << "  --router-lookahead-audit PREFIX record actual FFN inputs and next-router logits; diagnostic only\n"
        << "  --compare-ffn-transfer-phases     one load, fixed primary tiers and CPU routes; baseline/phased/baseline\n"
        << "  --compare-router-lookahead       one load, fixed primary tiers and CPU routes; baseline/prefetch/baseline\n"
        << "  --compare-rotary-fusion          one load, fixed primary tiers and CPU routes; original/fused/original\n"
        << "  --compare-attention-grouping     one load, fixed primary tiers and CPU routes; separate/grouped/separate\n"
        << "  --compare-mhc-fusion             one load, fixed primary tiers and CPU routes; original/fused/original; honors TYLOINFER_GR_TWO_STAGE_DENSE_INJECTION\n"
        << "  --compare-mhc-shared-norm        one load, fixed primary tiers and CPU routes; fused/shared-norm/fused\n"
        << "  --compare-mhc-dense-fusion       one load, fixed primary tiers and CPU routes; original/fused/original BF16 injection\n"
        << "  --compare-mhc-packed-groups      one load, fixed routes; original/warp-cached metadata and 512-thread Up/original\n"
        << "  --compare-mfe-parallel-gu        one load, fixed primary tiers and CPU routes; serial/parallel/serial Gate-Up\n"
        << "  --compare-mfe-gpu-bundle         one load, fixed primary tiers and CPU routes; current/bundle/current FFN\n"
        << "  --compare-mfe-shared-first       one load, fixed primary tiers and CPU routes; off/on/off shared expert CTA order\n"
        << "  --compare-mfe-early-gate-up      one load, fixed primary tiers and CPU routes; off/on/off resident Gate-Up with cold Down\n"
        << "  --compare-cpu-background-calibration one load; synchronous/background/synchronous, fresh CPU costs\n"
        << "  --compare-cpu-transfer-budget    one load; original/conservative/original CPU policy; report natural route changes\n"
        << "  --compare-router-topk            one load, fixed primary tiers and CPU routes; original/cached/original\n"
        << "  --decode-inputs-file FILE        fixed decode input ids, raw int32, gen-1 entries\n"
        << "  --gen N                         generated tokens\n"
        << "  --ignore-eos                    fixed-length token diagnostics\n"
        << "  --compare-route-poll            compare route notification polling, 2000/0/2000 us\n"
        << "  --compare-window-flush          compare per-layer / one token / per-layer WDDM submission flush\n"
        << "  --compare-early-no-cpu-ready     compare late / early / late GPU-only CPU completion\n"
        << "  --compare-gpu-only-cpu-wait      compare kept / skipped / kept CPU wait with a full GPU quota\n"
        << "  --compare-fixed-gpu-timing      compare kept / skipped / kept rate events with a full GPU quota\n"
        << "  --compare-window-input-views    compare staged / producer / staged graph input addresses\n"
        << "  --compare-route-spin           compare yielding / Strata pause / yielding route waits\n"
        << "  --compare-moe-plan-reuse       one load, fixed primary tiers; allocating/reused/allocating host plans\n"
        << "  --compare-graph-optimizations  compare graph, shared-gate and residual optimizations off / on / off\n"
        << "  --compare-mhc-fixed-formats     compare generic / specialized / generic GR Down formats\n"
        << "  --compare-mhc-async-normalized  compare sync / async / sync GR Float activation copies\n"
        << "  --compare-gdn-fixed-geometry   compare generic / fixed / generic GDN geometry\n"
        << "  --compare-gdn-inplace-state    compare copied / inplace / copied GDN state\n"
        << "  --compare-shared-gate-fusion    compare cuBLAS / fused / cuBLAS shared scalar gates\n"
        << "  --compare-mfe-default-math      compare complete / default-only / complete MFE arithmetic kernels\n"
        << "  --compare-mfe-e8-narrow         compare wide / proven narrow / wide MFE E8 metadata kernels\n"
        << "  --comparison-rounds N           repeat the off/on/off runtime comparison (1..8)\n"
        << "  --comparison-phase-timings      append a separate off/on/off MFE timing round after performance passes\n"
        << "  --compare-host-pin              compare decode host placement, free/pinned/free\n"
        << "  --compare-resident-overlap      compare resident GPU work before host planning, off/on/off\n"
        << "  --compare-dma-graph-batch       compare ordinary/batched/ordinary weight DMA submissions\n"
        << "  --compare-ram-field-packing     compare direct/packed/direct small RAM fields, fixed decode inputs\n"
        << "  --compare-warp-multi-sum       compare original / Strata multi-row reduction / original\n"
        << "  --warmup-prefill N              prefill warmups, 0-100; default profile=1, otherwise=0\n"
        << "  --ctx-size N                    context size; 0 selects a default\n"
        << "  -t, --threads N                 positive CPU thread count\n"
        << "  -ngl, --n-gpu-layers N          non-negative GPU layer count\n"
        << "  --cpu-offload-layers RANGES     explicit CPU-offloaded layers\n"
        << "  --moe-gpu-cache-gb N            bounded MoE GPU cache\n"
        << "  --tensor-parallel DEVICES       tensor-parallel devices\n"
        << "  --expert-parallel DEVICES       expert-parallel devices\n"
        << "  --layer-parallel DEVICES        layer-placement devices\n"
        << "  -h, --help                      show this help\n";
}

bool has_diagnostic_action(const DiagnosticsCommandOptions& value) {
    return !value.check_linear.empty() || !value.check_linear_cpu.empty() ||
        !value.check_tp_linear.empty() || !value.check_ep_moe.empty() ||
        !value.check_linear_group.empty() || !value.check_q8_embedding.empty() ||
        !value.check_gdn_input.empty() || !value.check_linear_conv_input.empty() ||
        !value.check_mfe_tensor.empty() || !value.check_dsv4_output_a.empty() ||
        value.check_moe_layer >= 0 || value.check_gemma_geglu_layer >= 0 ||
        value.check_attention_decode > 0 || value.check_backend_bf16_add ||
        value.check_backend_argmax || value.check_gemma4_swa || value.check_glm_dsa ||
        value.check_dsv4_attention || value.check_dsv4_hc ||
        value.check_deepseek_v41 || value.check_text_session_state ||
        value.check_qwen35_mtp || value.check_continuous_batching ||
        value.check_engine_isolation ||
        value.check_flash_next || value.check_qwen4_decode_hidden || value.check_flash_next_mtp ||
        value.check_nemotron_h || value.check_nemotron_h_mtp ||
        value.check_nemotron_h_vision ||
        value.check_runtime_assets || value.check_mfq_container ||
        value.compare_dsv4_hc_ops || value.compare_dsv4_hc_model ||
        value.compare_mma_attention || value.compare_decode_splitk ||
        value.compare_mma_decode || value.compare_nvq_vec4 ||
        !value.block_trace_reference.empty() || !value.block_trace_output.empty() ||
        !value.bench_qwen35_mtp.empty() || !value.prefill_sweep_arg.empty() ||
        !value.ids_arg.empty() || !value.ids_file.empty() ||
        value.prefill_repeat > 0 || value.profile || value.minicpmo_eval_batch;
}

// Keep flag dispatch flat: MSVC limits nesting in long else-if chains.
bool parse_diagnostic_switch(std::string_view option, DiagnosticsCommandOptions& result) {
    static constexpr std::pair<std::string_view, bool DiagnosticsCommandOptions::*> switches[] = {
        {"--check-qwen4-decode-hidden", &DiagnosticsCommandOptions::check_qwen4_decode_hidden},
        {"--compare-ram-field-packing", &DiagnosticsCommandOptions::compare_ram_field_packing},
        {"--compare-qsa-fusion", &DiagnosticsCommandOptions::compare_qsa_fusion},
        {"--compare-moe-plan-reuse", &DiagnosticsCommandOptions::compare_moe_plan_reuse},
        {"--compare-mfe-default-math", &DiagnosticsCommandOptions::compare_mfe_default_math},
        {"--compare-mfe-e8-narrow", &DiagnosticsCommandOptions::compare_mfe_e8_narrow},
        {"--minicpmo-eval-batch", &DiagnosticsCommandOptions::minicpmo_eval_batch},
        {"--check-mfe-routed-input", &DiagnosticsCommandOptions::check_mfe_routed_input},
        {"--check-mfe-benchmark-only", &DiagnosticsCommandOptions::check_mfe_benchmark_only},
        {"--check-attention-swa-decode", &DiagnosticsCommandOptions::check_attention_swa_decode},
        {"--check-gemma4-swa", &DiagnosticsCommandOptions::check_gemma4_swa},
        {"--check-glm-dsa", &DiagnosticsCommandOptions::check_glm_dsa},
        {"--check-dsv4-attention", &DiagnosticsCommandOptions::check_dsv4_attention},
        {"--check-dsv4-hc", &DiagnosticsCommandOptions::check_dsv4_hc},
        {"--check-deepseek-v41", &DiagnosticsCommandOptions::check_deepseek_v41},
        {"--check-text-session-state", &DiagnosticsCommandOptions::check_text_session_state},
        {"--check-qwen35-mtp", &DiagnosticsCommandOptions::check_qwen35_mtp},
        {"--check-flash-next", &DiagnosticsCommandOptions::check_flash_next},
        {"--check-flash-next-mtp", &DiagnosticsCommandOptions::check_flash_next_mtp},
        {"--check-nemotron-h", &DiagnosticsCommandOptions::check_nemotron_h},
        {"--check-nemotron-h-mtp", &DiagnosticsCommandOptions::check_nemotron_h_mtp},
        {"--check-nemotron-h-vision", &DiagnosticsCommandOptions::check_nemotron_h_vision},
        {"--compare-dsv4-hc-ops", &DiagnosticsCommandOptions::compare_dsv4_hc_ops},
        {"--compare-dsv4-hc-model", &DiagnosticsCommandOptions::compare_dsv4_hc_model},
        {"--check-continuous-batching", &DiagnosticsCommandOptions::check_continuous_batching},
        {"--check-engine-isolation", &DiagnosticsCommandOptions::check_engine_isolation},
        {"--check-runtime-assets", &DiagnosticsCommandOptions::check_runtime_assets},
        {"--check-mfq-container", &DiagnosticsCommandOptions::check_mfq_container},
        {"--compare-ffn-transfer-phases", &DiagnosticsCommandOptions::compare_ffn_transfer_phases},
        {"--compare-route-poll", &DiagnosticsCommandOptions::compare_route_poll},
        {"--compare-window-flush", &DiagnosticsCommandOptions::compare_window_flush},
        {"--compare-early-no-cpu-ready", &DiagnosticsCommandOptions::compare_early_no_cpu_ready},
        {"--compare-gpu-only-cpu-wait", &DiagnosticsCommandOptions::compare_gpu_only_cpu_wait},
        {"--compare-fixed-gpu-timing", &DiagnosticsCommandOptions::compare_fixed_gpu_timing},
        {"--compare-window-input-views", &DiagnosticsCommandOptions::compare_window_input_views},
        {"--compare-route-spin", &DiagnosticsCommandOptions::compare_route_spin},
        {"--compare-shared-gate-fusion", &DiagnosticsCommandOptions::compare_shared_gate_fusion},
        {"--compare-graph-optimizations", &DiagnosticsCommandOptions::compare_graph_optimizations},
        {"--comparison-phase-timings", &DiagnosticsCommandOptions::comparison_phase_timings},
        {"--compare-host-pin", &DiagnosticsCommandOptions::compare_host_pin},
        {"--compare-resident-overlap", &DiagnosticsCommandOptions::compare_resident_overlap},
        {"--compare-dma-graph-batch", &DiagnosticsCommandOptions::compare_dma_graph_batch},
        {"--compare-warp-multi-sum", &DiagnosticsCommandOptions::compare_warp_multi_sum},
        {"--compare-router-lookahead", &DiagnosticsCommandOptions::compare_router_lookahead},
        {"--compare-rotary-fusion", &DiagnosticsCommandOptions::compare_rotary_fusion},
        {"--compare-attention-grouping", &DiagnosticsCommandOptions::compare_attention_grouping},
        {"--compare-mhc-fusion", &DiagnosticsCommandOptions::compare_mhc_fusion},
        {"--compare-mhc-shared-norm", &DiagnosticsCommandOptions::compare_mhc_shared_norm},
        {"--compare-mhc-dense-fusion", &DiagnosticsCommandOptions::compare_mhc_dense_fusion},
        {"--compare-mhc-packed-groups", &DiagnosticsCommandOptions::compare_mhc_packed_groups},
        {"--compare-mhc-fixed-formats", &DiagnosticsCommandOptions::compare_mhc_fixed_formats},
        {"--compare-mhc-async-normalized", &DiagnosticsCommandOptions::compare_mhc_async_normalized},
        {"--compare-gdn-fixed-geometry", &DiagnosticsCommandOptions::compare_gdn_fixed_geometry},
        {"--compare-gdn-inplace-state", &DiagnosticsCommandOptions::compare_gdn_inplace_state},
        {"--compare-mfe-parallel-gu", &DiagnosticsCommandOptions::compare_mfe_parallel_gu},
        {"--compare-mfe-gpu-bundle", &DiagnosticsCommandOptions::compare_mfe_gpu_bundle},
        {"--compare-mfe-shared-first", &DiagnosticsCommandOptions::compare_mfe_shared_first},
        {"--compare-mfe-early-gate-up", &DiagnosticsCommandOptions::compare_mfe_early_gu},
        {"--compare-cpu-background-calibration", &DiagnosticsCommandOptions::compare_cpu_background_calibration},
        {"--compare-cpu-transfer-budget", &DiagnosticsCommandOptions::compare_cpu_transfer_budget},
        {"--compare-router-topk", &DiagnosticsCommandOptions::compare_router_topk},
        {"--profile", &DiagnosticsCommandOptions::profile},
        {"--ignore-eos", &DiagnosticsCommandOptions::ignore_eos},
        {"--check-backend-bf16-add", &DiagnosticsCommandOptions::check_backend_bf16_add},
        {"--check-backend-argmax", &DiagnosticsCommandOptions::check_backend_argmax},
        {"--compare-mma-attention", &DiagnosticsCommandOptions::compare_mma_attention},
        {"--compare-decode-splitk", &DiagnosticsCommandOptions::compare_decode_splitk},
        {"--compare-mma-decode", &DiagnosticsCommandOptions::compare_mma_decode},
        {"--compare-nvq-vec4", &DiagnosticsCommandOptions::compare_nvq_vec4},
        {"--compare-niq-vec4", &DiagnosticsCommandOptions::compare_nvq_vec4},
    };
    for (const auto& [name, field] : switches) {
        if (option == name) {
            result.*field = true;
            return true;
        }
    }
    return false;
}

DiagnosticsCommandOptions parse_diagnostics(ArgCursor& args) {
    DiagnosticsCommandOptions result;
    while (!args.empty()) {
        const std::string_view option = args.next();
        if (option == "--help" || option == "-h") {
            print_diagnostics_help();
            throw HelpRequested{};
        }
        if (parse_cuda_load_option(option, args, result) ||
                parse_token_input_option(option, args, result) ||
                parse_diagnostic_switch(option, result)) {
            continue;
        }
        if (option == "--continuous-batching-check-requests") {
            result.check_continuous_batching_requests = integer<int>(args.value(option),option);
            continue;
        }
        if (option == "--minicpmo-eval-vision-batch-size") {
            result.minicpmo_eval_vision_batch_size =
                integer<int64_t>(args.value(option), option);
            if (result.minicpmo_eval_vision_batch_size <= 0) {
                usage_error("--minicpmo-eval-vision-batch-size must be positive");
            }
        }
        else if (option == "--check-linear") result.check_linear = args.value(option);
        else if (option == "--check-linear-cpu") result.check_linear_cpu = args.value(option);
        else if (option == "--check-tp-linear") result.check_tp_linear = args.value(option);
        else if (option == "--check-tp-axis") result.check_tp_axis_arg = args.value(option);
        else if (option == "--check-tp-m") result.check_tp_m = integer<int>(args.value(option), option);
        else if (option == "--check-ep-moe" || option == "--check-tp-moe") {
            result.check_ep_moe = args.value(option);
        }
        else if (option == "--check-ep-moe-tokens" ||
                option == "--check-tp-moe-tokens") {
            result.check_ep_moe_tokens = integer<int>(args.value(option), option);
        }
        else if (option == "--check-ep-moe-routes" ||
                option == "--check-tp-moe-routes") {
            result.check_ep_moe_routes = integer<int>(args.value(option), option);
        }
        else if (option == "--check-linear-group") result.check_linear_group = args.value(option);
        else if (option == "--check-q8-embedding") result.check_q8_embedding = args.value(option);
        else if (option == "--check-gdn-input") result.check_gdn_input = args.value(option);
        else if (option == "--check-gdn-output") result.check_gdn_output = args.value(option);
        else if (option == "--check-gdn-state") result.check_gdn_state = args.value(option);
        else if (option == "--check-gdn-tokens") result.check_gdn_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-q-heads") result.check_gdn_q_heads = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-v-heads") result.check_gdn_v_heads = integer<int>(args.value(option), option);
        else if (option == "--check-gdn-head-dim") result.check_gdn_head_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-input") result.check_linear_conv_input = args.value(option);
        else if (option == "--check-linear-conv-output") result.check_linear_conv_output = args.value(option);
        else if (option == "--check-linear-conv-tokens") result.check_linear_conv_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-q-heads") result.check_linear_conv_q_heads = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-v-heads") result.check_linear_conv_v_heads = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-key-dim") result.check_linear_conv_key_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-value-dim") result.check_linear_conv_value_dim = integer<int>(args.value(option), option);
        else if (option == "--check-linear-conv-kernel") result.check_linear_conv_kernel = integer<int>(args.value(option), option);
        else if (option == "--check-linear-m") result.check_linear_m = integer<int>(args.value(option), option);
        else if (option == "--check-linear-reps") result.check_linear_reps = integer<int>(args.value(option), option);
        else if (option == "--check-linear-gate") result.check_linear_gate = args.value(option);
        else if (option == "--check-gemma-geglu-layer") result.check_gemma_geglu_layer = integer<int>(args.value(option), option);
        else if (option == "--check-gemma-geglu-reps") result.check_gemma_geglu_reps = integer<int>(args.value(option), option);
        else if (option == "--check-moe-layer") result.check_moe_layer = integer<int>(args.value(option), option);
        else if (option == "--check-moe-tokens") {
            result.check_moe_tokens = args.value(option);
            validate_integer_list(result.check_moe_tokens, option);
        }
        else if (option == "--check-moe-reps") result.check_moe_reps = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-tensor") result.check_mfe_tensor = args.value(option);
        else if (option == "--check-mfe-tokens") result.check_mfe_tokens = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-routes") result.check_mfe_routes = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-reps") result.check_mfe_reps = integer<int>(args.value(option), option);
        else if (option == "--check-mfe-split-width") result.check_mfe_split_width = integer<int>(args.value(option), option);
        else if (option == "--check-dsv4-output-a") result.check_dsv4_output_a = args.value(option);
        else if (option == "--check-dsv4-output-a-batch") result.check_dsv4_output_a_batch = integer<int>(args.value(option), option);
        else if (option == "--check-dsv4-output-a-reps") result.check_dsv4_output_a_reps = integer<int>(args.value(option), option);
        else if (option == "--check-attention-decode") result.check_attention_decode = integer<int>(args.value(option), option);
        else if (option == "--check-attention-reps") result.check_attention_reps = integer<int>(args.value(option), option);
        else if (option == "--check-attention-head-dim") result.check_attention_head_dim = integer<int>(args.value(option), option);
        else if (option == "--check-attention-window") result.check_attention_window = integer<int>(args.value(option), option);
        else if (option == "--bench-qwen35-mtp") result.bench_qwen35_mtp = args.value(option);
        else if (option == "--bench-qwen35-mtp-reps") result.bench_qwen35_mtp_reps = integer<int>(args.value(option), option);
        else if (option == "--kl-mmq") {
            result.kl_mmq_arg = args.value(option);
            try { (void)parse_kl_mmq_mode(result.kl_mmq_arg); }
            catch (const std::runtime_error& error) {
                usage_error(error.what());
            }
        }
        else if (option == "--compare-block-trace") result.block_trace_reference = args.value(option);
        else if (option == "--dump-block-trace") result.block_trace_output = args.value(option);
        else if (option == "--dump-block-trace-start") result.block_trace_start = integer<int64_t>(args.value(option), option);
        else if (option == "--dump-block-trace-count") result.block_trace_count = integer<int64_t>(args.value(option), option);
        else if (option == "--prefill-repeat") result.prefill_repeat = integer<int>(args.value(option), option);
        else if (option == "--prefill-sweep") {
            result.prefill_sweep_arg = args.value(option);
            validate_integer_list(result.prefill_sweep_arg, option);
        }
        else if (option == "--prefill-sweep-reps") result.prefill_sweep_reps = integer<int>(args.value(option), option);
        else if (option == "--check-tokenizer-text") result.check_tokenizer_text = args.value(option);
        else if (option == "--router-lookahead-audit") result.router_lookahead_audit = args.value(option);
        else if (option == "--comparison-rounds") {
            result.comparison_rounds=integer<int>(args.value(option),option);
            if(result.comparison_rounds<1 || result.comparison_rounds>8)
                throw std::invalid_argument("comparison rounds must be between one and eight");
        }
        else if (option == "--decode-inputs-file") result.decode_inputs_file = args.value(option);
        else if (option == "--warmup-prefill") {
            result.warmup_prefill = integer<int>(args.value(option), option);
            if (result.warmup_prefill < 0 || result.warmup_prefill > 100)
                usage_error("--warmup-prefill requires a count from 0 to 100");
        }
        else if (option == "--compare-mma-decode-steps") result.compare_mma_decode_steps = integer<int>(args.value(option), option);
        else if (option == "--compare-mma-decode-planned-len") result.compare_mma_decode_planned_len = integer<int>(args.value(option), option);
        else usage_error("unknown option: " + std::string(option));
    }
    if (!result.decode_inputs_file.empty()) {
        result.decode_inputs = load_ids_file(result.decode_inputs_file);
        if (result.gen <= 1 || result.decode_inputs.size() != size_t(result.gen - 1))
            usage_error("--decode-inputs-file requires exactly gen-1 entries and gen greater than 1");
        if (std::any_of(result.decode_inputs.begin(),result.decode_inputs.end(),[](int64_t id){return id<0;}))
            usage_error("--decode-inputs-file requires nonnegative token ids");
    }
    if (!result.ids_arg.empty() && !result.ids_file.empty()) {
        usage_error("--ids and --ids-file are mutually exclusive");
    }
    if (!has_diagnostic_action(result)) {
        usage_error("mfq-diagnostics requires a check or benchmark");
    }
    return result;
}

} // namespace

int run_diagnostics(int argc, char** argv) {
    try {
        ArgCursor args(argc, argv);
        auto options = parse_diagnostics(args);
        return tyloi::cuda::execute_diagnostics_command(std::move(options));
    } catch (const HelpRequested&) {
        return 0;
    } catch (const UsageError& error) {
        return print_usage_error(error);
    } catch (const std::exception& error) {
        std::cerr << "mfq-diagnostics: " << error.what() << '\n';
        return 1;
    }
}

} // namespace tyloi::cuda::commands
