from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUDA_ROOT = ROOT / "csrc" / "backends" / "cuda"
SOURCE = "\n".join(
    path.read_text(encoding="utf-8")
    for path in sorted(CUDA_ROOT.rglob("*"))
    if path.suffix in {".h", ".cpp"}
)
CORE = (ROOT / "csrc" / "core" / "tensor_parallel.h").read_text(
    encoding="utf-8"
)
CMAKE = (ROOT / "csrc" / "tests" / "CMakeLists.txt").read_text(
    encoding="utf-8"
)
COMPONENTS = "\n".join(
    path.read_text(encoding="utf-8")
    for path in (
        CUDA_ROOT / "engine" / "model_components.h",
        CUDA_ROOT / "engine" / "model_components.cpp",
        CUDA_ROOT / "models" / "qwen35" / "ops.cpp",
    )
)


def test_tensor_parallel_cli_and_weighted_split_are_wired():
    assert '"--tensor-parallel"' in SOURCE
    assert '"--tensor-split"' in SOURCE
    assert "plan_tensor_parallel_slices" in SOURCE
    assert "cudaDeviceEnablePeerAccess" in SOURCE


def test_dense_ffn_keeps_intermediate_shards_local():
    assert "forward_tensor_parallel_dense" in SOURCE
    assert '"ffn.tensor_parallel"' in SOURCE
    assert "reduce_model_parallel_outputs" in SOURCE


def test_routed_moe_all_compact_families_are_expert_sharded():
    assert "to_cuda_device_moe_expert_slice" in SOURCE
    assert "select_nint_cpu_rows" in SOURCE
    assert "select_nint8_zero_cpu_rows" in SOURCE
    assert "select_nvq_cpu_rows" in SOURCE
    assert "select_nepq_cpu_rows" in SOURCE
    assert "partial_experts = true" in SOURCE
    assert "expert_parallel_shards" in SOURCE
    assert '"--check-ep-moe"' in SOURCE
    assert "run_expert_parallel_moe_check" in SOURCE


def test_expert_parallel_cli_and_hybrid_device_contract_are_wired():
    assert '"--expert-parallel"' in SOURCE
    assert '"--expert-split"' in SOURCE
    assert "execution.expert_parallel" in SOURCE
    assert "plan_moe_expert_parallel_slices" in SOURCE
    assert "execution.tensor_parallel.devices !=" in SOURCE
    assert "execution.expert_parallel.devices" in SOURCE


def test_tensor_parallel_has_a_native_partition_test_target():
    assert "mfq-tensor-parallel-test" in CMAKE
    assert "tensor_parallel_test.cpp" in CMAKE


def test_tq_parallel_preserves_compressed_rows_and_input_rings():
    linear = (CUDA_ROOT / "ops" / "common" / "quant_linear.cpp").read_text(encoding="utf-8")
    loader = (CUDA_ROOT / "storage" / "weights" / "loaders" / "weight.cpp").read_text(encoding="utf-8")
    header = (CUDA_ROOT / "ops" / "common" / "include" / "quant_linear.h").read_text(encoding="utf-8")
    shard = header.split("struct QuantLinearShard {", 1)[1].split("};", 1)[0]
    assert "TqWeight tq;" in shard
    assert "rows.slice_rows(slice.begin,slice.end)" in loader
    assert "rows.slice_columns(slice.begin*256,slice.end*256)" in loader
    assert "shard.tq=to_device_tq" in loader
    assert 'profiler.measure("tq.tensor_parallel"' in linear
    assert "if(is_tq() && !tensor_parallel())" in linear
    for method, logical in (("out", "logical_out"), ("neuron_len", "logical_neuron_len")):
        body = linear.split(f"int64_t QuantLinear::{method}() const {{", 1)[1].split("\n}", 1)[0]
        assert body.index(f"if (tensor_parallel()) return {logical}") < body.index("if(is_tq())")
    assert '"--tq-tp-check"' in SOURCE
    assert '"TQ compressed shard changed decoded weights"' in SOURCE


def test_tq_self_describing_headers_supply_missing_shape_metadata():
    tq = (CUDA_ROOT / "ops" / "tq" / "tq.cpp").read_text(encoding="utf-8")
    logical = tq.split("std::vector<std::int64_t> logical_shape(", 1)[1].split(
        "std::vector<std::string> row_parts(", 1
    )[0]
    assert "if(shape.empty() && record)" in logical
    assert "mfq::tq_tensor_shape(record->nbytes,source.tensor_reader(name))" in logical
    assert logical.index("if(shape.empty() && record)") < logical.index("if(shape.size()!=2")


def test_tq_residual_projection_uses_the_parallel_aware_linear_entry():
    loader = (CUDA_ROOT / "storage" / "weights" / "loaders" / "weight.cpp").read_text(encoding="utf-8")
    residual = loader.split("ResidualLinear residual_linear(", 1)[1]
    tq = residual.split("if(weight->is_tq()) {", 1)[1].split("if(weight->is_dense())", 1)[0]
    assert "weight->forward(context," in tq
    assert "weight->tq.forward(" not in tq
    assert "input.scalar_type()==tb::kFloat16?input:input.to(tb::kFloat32)" in tq
    assert '"TQ residual projection bypassed tensor-parallel shards or dtype promotion"' in SOURCE


def test_tq_expert_cache_isolates_device_images_and_upload_arenas():
    tq = (CUDA_ROOT / "ops" / "tq" / "tq.cpp").read_text(encoding="utf-8")
    cache = tq.split("class TqExpertCache {", 1)[1]
    assert "std::unordered_map<DeviceKey,DeviceEntry,DeviceKeyHash> device_;" in cache
    assert "const DeviceKey device_key{key,device};" in cache
    assert "device_.find(device_key)" in cache
    assert "device_.emplace(device_key," in cache
    assert "stages_[destination.get_device()]" in cache
    assert "gemm_.try_emplace(device,resources)" in cache
    assert "covered_images.insert(key.image)" in cache
    assert '"TQ cache shared images across devices or lost resident hits"' in SOURCE


def test_tq_default_residency_is_not_an_explicit_zero_cache_budget():
    tq = (CUDA_ROOT / "ops" / "tq" / "tq.cpp").read_text(encoding="utf-8")
    assert "execution.config.moe_device_cache_bytes.value_or(" in tq
    assert "execution.config.moe_device_cache_bytes.has_value()" in tq
    assert "bool(execution.moe_expert_cache)" in tq
    assert "execution.config.moe_host_physical_bytes,!explicit_budget)" in tq
    assert "if(retain_all_ || weight.image.numel()<=device_limit_)" in tq
    resize = tq.split("void resize(std::size_t bytes) {", 1)[1].split("std::size_t reclaim_host(", 1)[0]
    assert "retain_all_=false;" in resize
    assert '"TQ default loading disabled GPU expert residency"' in SOURCE
    assert '"TQ explicit zero budget did not disable expert residency"' in SOURCE


def test_tq_expert_partitions_keep_global_routes_and_zero_unowned_outputs():
    tq = (CUDA_ROOT / "ops" / "tq" / "tq.cpp").read_text(encoding="utf-8")
    assert "plan_moe_expert_parallel_slices(parallel,gate->experts" in tq
    assert "expert<expert_begin || expert>=expert_end" in tq
    assert "auto pairs=partitioned?tb::zeros(" in tq
    assert "if(unique.empty())return tb::zeros(" in tq
    assert "return reduce_model_parallel_outputs(execution,std::move(partials))" in tq
    ffn = (CUDA_ROOT / "models" / "common" / "ffn.cpp").read_text(encoding="utf-8")
    qwen = (CUDA_ROOT / "models" / "qwen4_exp" / "ops.cpp").read_text(encoding="utf-8")
    assert "moe_tq->forward_parallel(execution," in ffn
    assert "tq->forward_parallel(execution," in qwen
    assert '"TQ expert parallelism changed routed FFN values"' in SOURCE


def test_model_parallel_rejects_silent_moe_cache_bypass():
    assert (
        '"--moe-gpu-cache-gb cannot be combined with "'
        in SOURCE
    )
    assert "model_parallel_enabled(execution)" in SOURCE
    assert '"tensor/expert parallelism"' in SOURCE


def test_tensor_parallel_graph_capture_registers_all_participant_streams():
    assert '"TYLOINFER_MODEL_PARALLEL_CUDA_GRAPH"' in SOURCE
    assert '"TYLOINFER_TP_CUDA_GRAPH"' in SOURCE
    assert '"TYLOINFER_EP_CUDA_GRAPH"' in SOURCE
    assert "model_parallel_cuda_graph_enabled(" in SOURCE
    assert "execution.model_parallel_collectives" in SOURCE
    assert "execution.config.tensor_parallel_cuda_graph" in SOURCE
    assert "execution.config.expert_parallel_cuda_graph" in SOURCE
    assert "graph_participant_streams" in SOURCE
    assert "collectives.streams.begin()" in SOURCE
    assert "graph.compute_streams" in SOURCE
    assert "participant_streams" in SOURCE
    assert "bool graph_eligible() const" in SOURCE
    assert "bool use_cuda_graph" in SOURCE


def test_tensor_parallel_graph_primes_and_captures_nccl_peer_transfers():
    assert "p2p_warmup_buffers" in SOURCE
    assert "ncclSend(" in SOURCE
    assert "ncclRecv(" in SOURCE
    assert "runtime.streams[destination_rank].stream()" in SOURCE
    assert "runtime.ready[source_rank]" in SOURCE
    assert "cudaStreamSynchronize(\n                participant.stream())" in SOURCE
    assert "for (int pass = 0; pass < 2; ++pass)" in SOURCE


def test_expert_parallel_graph_uses_capture_safe_peer_transfers():
    assert "tensor_to_cuda_device(collectives, std::move(value), device)" in SOURCE
    assert (
        "destination = tensor_to_cuda_device(\n"
        "            collectives, source, device, std::move(destination));"
        in SOURCE
    )
    assert "auto destination = destination_tensor();" in SOURCE
    assert ".clone().to(pool.weight.q_packed.device()).contiguous()" in SOURCE


def test_two_rank_fp16_reduce_avoids_round_trip_casts():
    assert '"TYLOINFER_MODEL_PARALLEL_FP16_REDUCE"' in SOURCE
    assert '"TYLOINFER_TP_FP16_REDUCE"' in SOURCE
    assert "execution.config.model_parallel_fp16_reduce" in SOURCE
    assert "outputs.size() == 2" in SOURCE
    assert "output_dtype == mfq_tensor_backend::kFloat16" in SOURCE
    assert "fp16_reduce ? ncclFloat16 : ncclFloat32" in SOURCE
    assert "result = outputs[index]" in SOURCE


def test_tensor_parallel_projection_groups_share_each_rank_input_transfer():
    assert '"TYLOINFER_TP_GROUPED_PROJECTIONS"' in SOURCE
    assert "execution.config.tensor_parallel_grouped_projections" in SOURCE
    assert "tensor_parallel_output_compatible()" in SOURCE
    assert "forward_tensor_parallel_output_group(execution, x)" in SOURCE
    assert (
        "auto local_x = tensor_to_cuda_device(\n"
        "            execution.model_parallel_collectives, flat, device);"
        in SOURCE
    )


def test_batched_decode_graph_orders_tp_groups_by_projection():
    assert "decode_graph_tp_projection_major" in SOURCE
    assert "thread_local bool g_decode_graph_tp_projection_major" not in SOURCE
    assert "DecodeGraphTpProjectionScope" in SOURCE
    assert "std::vector<mfq_tensor_backend::Tensor> local_inputs" in SOURCE
    assert (
        "local_inputs[shard] = tensor_to_cuda_device(\n"
        "                execution.model_parallel_collectives, flat, device)"
        in SOURCE
    )


def test_tensor_parallel_peer_first_launch_preserves_rank_indexing():
    assert '"TYLOINFER_MODEL_PARALLEL_PEER_FIRST_LAUNCH"' in SOURCE
    assert '"TYLOINFER_TP_PEER_FIRST_LAUNCH"' in SOURCE
    assert "model_parallel_launch_index(" in SOURCE
    assert "config.model_parallel_peer_first_launch" in SOURCE
    assert "peer_first_parallel_launch_index" in SOURCE
    assert "launch_position < primary_rank" in CORE
    assert "local_outputs[index] =" in SOURCE
    assert "partials[index]=mfq::models::gated_mlp(" in "".join(SOURCE.split())


def test_qwen35_mtp_accepts_dense_tensor_parallel_placement():
    source = (CUDA_ROOT / "engine/model_components.cpp").read_text()
    qwen35 = source.split(
        "RuntimeComponents<mfq::cuda::Qwen35CausalLm> load_runtime_components(", 1
    )[1].split("RuntimeComponents<mfq::cuda::Qwen4CausalLm>", 1)[0]
    assert "const bool supported_placement" in qwen35
    assert "!model_execution.layer_placement.enabled()" in qwen35
    assert "!execution.tensor_parallel.enabled()" not in qwen35
    assert "dense GPU-resident Qwen blocks" in qwen35


def test_quantized_tp_weights_and_workspaces_follow_the_shard_device():
    assert SOURCE.count("mfq_current_cuda_device()") >= 3
    assert "const auto workspace_options = q_packed.options();" in SOURCE
    assert "const auto workspace_options = indices_packed.options();" in SOURCE


def test_deepseek_v4_split_gate_up_uses_existing_moe_runtime_and_cache():
    shared = (CUDA_ROOT.parents[1] / "models/common/weight_loading.h").read_text()
    assert 'prefix + "experts.gate.weight"' in shared
    assert 'prefix + "experts.up.weight"' in shared
    assert "split != ops.has(up)" in shared
    assert "moe_split_gate_up" in shared
    assert 'role = "gate"' in SOURCE
    assert 'role = "up"' in SOURCE
    assert '"moe.gate_up_split"' in SOURCE
    assert "mfq_tensor_backend::cat({gate, up}, -1).contiguous()" in SOURCE


def test_generic_ffn_loader_stays_dense_and_model_moe_loaders_are_typed():
    shared = (CUDA_ROOT.parents[1] / "models/common/weight_loading.h").read_text()
    dense = shared.split("Ffn load_dense_ffn", 1)[1].split("Ffn load_moe_weights", 1)[0]
    assert "experts.gate.weight" not in dense
    assert "const ModelConfig& config" in dense
    assert "deepseek_v4::load_block(" in SOURCE
    assert "glm_dsa::load_ffn(" in SOURCE
    assert "models::load_moe_weights<FFN>" in SOURCE


def test_native_float_linears_are_supported_without_forcing_tp_shards():
    assert "QuantLinearKind::Dense" in SOURCE
    assert 'dtype == "BF16" || dtype == "F16" || dtype == "F32"' in SOURCE
    assert '"TYLOINFER_TP_SHARD_NATIVE_FLOAT"' in SOURCE
    assert "execution.config.tensor_parallel_shard_native_float" in SOURCE
    assert (
        "result.dense = cpu.to(mfq_tensor_backend::kCUDA).contiguous()" in SOURCE
    )
