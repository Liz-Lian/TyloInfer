from pathlib import Path


METAL = Path(__file__).resolve().parents[1] / "csrc/backends/metal"


def test_native_kernel_preparation_does_not_dispatch_graphs():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    native = source.split("void prepare_mlx_kernel_gpu(", 1)[1].split("thread_local", 1)[0]
    for operation in (".eval(", "mlx::core::eval(", "get_command_encoder(",
                      "dispatch_threads(", "dispatch_threadgroups(", "allocator::malloc("):
        assert operation not in native
    assert "device.get_library(" in source
    assert "device.get_kernel(" in source
    assert "bounded(input_shapes, input_dtypes) && bounded(output_shapes, output_dtypes)" in source
    assert "compiled_jobs_.push_back" in source
    assert "mlx::core::zeros(input_shapes[index]" in source


def test_preparation_defaults_to_available_cpu_concurrency():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "std::max(1u, std::thread::hardware_concurrency())" in source
    assert "std::size_t workers = maximum_workers;" in source
    assert "requested > maximum_workers" in source
    assert "workers = std::min(workers, std::max<std::size_t>(1, parallel_total));" in source
    assert "[1, 16]" not in source
    assert "std::vector<std::thread> threads;" in source
    assert "thread.joinable()" in source
    assert "std::jthread" not in source


def test_compiled_preparation_preserves_bounded_view_layouts():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "preparation_view_input(input, key)" in source
    assert 'name == "Transpose" || name == "Reshape" || name == "Slice" || name == "Broadcast"' in source
    assert "input.inputs().front().nbytes() <= (4u << 20)" in source
    assert "inputs = std::move(view_inputs)" in source
    native = (METAL / "tests/mlx_compiled_prepare_test.cpp").read_text()
    assert "preparation executed original strided-view graphs" in native


def test_flash_next_prepares_mtp_float_stream_inputs():
    model = (METAL / "models/flash_next/mlx_qwen4_causal_lm.cpp").read_text()
    inputs = model.split("void prepare_inputs(const MlxEmbedding& embedding)", 1)[1].split("std::size_t layer_count()", 1)[0]
    assert "{preparation->activation_dtype(), mlx::core::float32}" in inputs
    assert "fuse_inputs(hidden, ids, embedding, false).streams" in inputs


def test_custom_kernel_cache_matches_source_and_compile_options():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "MLX_VERSION_NUMERIC >= 32001" in source
    assert "std::hash<std::string>{}(std::get<1>(state))" in source
    assert "std::get<10>(state)" in source
    cached = (METAL / "runtime/mlx_cached_kernel.cpp").read_text()
    assert "backend.get_library(library_name," in cached


def test_preparation_covers_host_and_device_sampling_inputs():
    source = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "mlx::core::zeros({rows}, mlx::core::float32)" in source
    assert "mlx::core::array(random_values.begin(), mlx::core::Shape{rows})" in source
    assert "mlx::core::bfloat16" in source


def test_flash_next_prepares_actual_storage_and_rollback_kernels():
    model = (METAL / "models/flash_next/mlx_qwen4_causal_lm.cpp").read_text()
    tensor = (METAL / "runtime/mlx_tensor.cpp").read_text()
    preparation = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    assert "if (activation_dtype) result.set_activation_dtype(*activation_dtype);" in tensor
    assert "result.preferred_activation_dtype()" in tensor
    assert "const auto& normalized = projected.second;" in model
    assert "compiled_replay_(transaction, 0)" in model
    assert "pool_index_keys(raw_index, positions, 0, blocks)" in model
    assert "for (const int blocks : {1, 8})" in model
    rows = (METAL / "storage/gguf/mlx_gguf_rows.cpp").read_text()
    assert "for (const int count : {1, 64})" in rows
    assert 'std::string_view(primitive->name()) == "ScatterAxis"' in preparation


def test_load_prepares_kernels_before_service_ready():
    source = (METAL / "apps/mfq_decode_mlx.cpp").read_text()
    loaded = source.split("int run_loaded_runtime(", 1)[1].split("int run_native_runtime(", 1)[0]
    assert loaded.index("preparation->finish();") < loaded.index("return runtime.run();")
    assert "preparation.collect_sampling(preparation_vocab, preparation_sampling);" in source


def test_warmup_is_internal_and_cleans_state_before_ready():
    source = (METAL / "apps/mfq_decode_mlx.cpp").read_text()
    warmup = source.split("void warm_runtime()", 1)[1].split("std::shared_ptr<std::mutex> runtime_mutex", 1)[0]
    assert "generate_with_prefill_metrics(" in warmup
    assert "session_cache->" not in warmup
    assert "runtime.reset_generation_state();" in warmup
    assert "release_model_load_staging_memory(runtime_stream);" in warmup
    for family, filename in (("flash_next", "mlx_qwen4_causal_lm.cpp"), ("qwen35", "mlx_qwen35_causal_lm.cpp")):
        model = (METAL / "models" / family / filename).read_text()
        reset = model.split("::reset_generation_state() noexcept", 1)[1].split("\n}", 1)[0]
        assert "clear_cache();" in reset and "policy_state" in reset and "last_mtp_stats" in reset


def test_flash_next_prepares_execution_paths_instead_of_weight_variants():
    preparation = (METAL / "runtime/mlx_kernel_prepare.cpp").read_text()
    tensor = (METAL / "runtime/mlx_tensor.cpp").read_text()
    moe = (METAL / "ops/mfq/mlx_moe.cpp").read_text()
    model = (METAL / "models/flash_next/mlx_qwen4_causal_lm.cpp").read_text()
    assert tensor.count("!preparation->model_paths()") == 2
    assert "1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 25, 31, 32, 33, 64, 65, 96, 128, chunk_size" in preparation
    assert "!preparation->model_paths() && preparation->routes() > 0" in moe
    load = model.split("MlxQwen4CausalLm MlxQwen4CausalLm::load(", 1)[1].split("\n}\n", 1)[0]
    assert load.index("preparation->use_model_paths();") < load.index("Impl::load(")
    assert "const auto projected = project(input, embeddings);" in model
    assert "preparation->add(fuse_inputs(hidden, ids, embedding, false).streams);" in model
    assert "preparation->add(project_logits(result->output, result->embedding, hidden));" in model
