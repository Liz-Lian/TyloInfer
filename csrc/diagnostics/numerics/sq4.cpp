#include "sq4.h"
#include "tensor.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/backends/cuda/kernels/mxfp4_sq.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/graph.h"
#include "engine/weights/encoding/mxfp4_sq_blob.h"
#include "engine/weights/encoding/mxfp4_sq_decode.h"
#endif

namespace tyloi::diagnostics::numerics {
namespace {
#if TYLOINFER_ENABLE_MFQ_COMPAT
using tyloi::DenseTensor;

struct ForceTensorCore {
    std::optional<std::string> previous;
    ForceTensorCore() {
        if (const char* value = tyloi::cuda::env_lookup("TYLOINFER_FORCE_SQ_TENSOR_CORE"))
            previous = value;
        tyloi::cuda::env_set("TYLOINFER_FORCE_SQ_TENSOR_CORE", "1");
    }
    ~ForceTensorCore() noexcept {
        try {
            if (previous)
                tyloi::cuda::env_set("TYLOINFER_FORCE_SQ_TENSOR_CORE", previous->c_str());
            else
                tyloi::cuda::env_clear("TYLOINFER_FORCE_SQ_TENSOR_CORE");
        } catch (const std::exception& error) {
            std::cerr << "SQ4 environment cleanup: " << error.what() << '\n';
        } catch (...) {
            std::cerr << "SQ4 environment cleanup: unknown error\n";
        }
    }
};

struct GraphRun {
    explicit GraphRun(tyloi::cuda::StreamHandle value) : stream(std::move(value)) {}
    ~GraphRun() noexcept {
        graph.reset();
        (void)cudaStreamSynchronize(stream.stream());
    }
    tyloi::cuda::StreamHandle stream;
    tyloi::cuda::Graph graph;
    DenseTensor ordinary, warmup, result;
};

struct Weight {
    tyloi::sq::Layout layout;
    DenseTensor blob, row_q, offsets, auxiliary;
    std::int64_t q_sum = 0, sq4_rows = 0;
};

DenseTensor upload(TensorWork& work, void* data, std::int64_t count, tyloi::ScalarType dtype) {
    auto host = work.keep(
        tyloi::cuda::from_blob(
            data, {count}, tyloi::TensorOptions().dtype(dtype).device(tyloi::Device(tyloi::kCPU)))
            .clone());
    return work.keep(host.to(work.device()));
}

Weight load(TensorWork& work, std::vector<std::uint8_t>& blob, const Json& config) {
    Weight result;
    result.layout = tyloi::sq::parse(blob.data(), blob.size());
    const auto& q = result.layout;
    if (config.at("bits") != q.bits || config.at("base") != q.base ||
        config.at("weight_shape") != Json::array({q.outputs, q.width}))
        throw std::invalid_argument("SQ4 payload geometry differs from case");
    const auto rows = tyloi::sq::row_metadata(blob.data(), q);
    std::vector<std::int32_t> offsets(rows.q.size()), auxiliary(rows.q.size());
    for (std::size_t i = 0; i < rows.q.size(); ++i) {
        if (rows.symbol_byte_offsets[i] > std::uint32_t(std::numeric_limits<std::int32_t>::max()) ||
            rows.auxiliary_rows[i] > std::uint32_t(std::numeric_limits<std::int32_t>::max()))
            throw std::overflow_error("SQ4 row metadata exceeds int32");
        offsets[i] = static_cast<std::int32_t>(rows.symbol_byte_offsets[i]);
        auxiliary[i] = static_cast<std::int32_t>(rows.auxiliary_rows[i]);
        result.q_sum += rows.q[i];
        result.sq4_rows += rows.q[i] == 4;
    }
    result.blob = upload(work, blob.data(), blob.size(), tyloi::kUInt8);
    result.row_q = upload(work, const_cast<std::uint8_t*>(rows.q.data()), q.outputs, tyloi::kUInt8);
    result.offsets = upload(work, offsets.data(), q.outputs, tyloi::kInt32);
    result.auxiliary = upload(work, auxiliary.data(), q.outputs, tyloi::kInt32);
    return result;
}

DenseTensor decode(const Weight& w, bool fp32) {
    const auto& q = w.layout;
    return mxfp4_sq_dequant_cuda(w.blob, w.row_q, w.offsets, w.auxiliary, q.bits, q.outputs,
                                 q.width, q.base, w.q_sum, w.sq4_rows, fp32);
}

DenseTensor project(const Weight& w, const DenseTensor& x) {
    const auto& q = w.layout;
    return mxfp4_sq_matmul_cuda(w.blob, w.row_q, w.offsets, w.auxiliary, x, q.bits, q.outputs,
                                q.width, q.base, w.q_sum, w.sq4_rows);
}

void check(Json& checks, int index, bool passed, Json evidence) {
    checks["native_" + std::to_string(index)] = {{"status", passed ? "pass" : "fail"},
                                                 {"evidence", std::move(evidence)}};
}

bool rejects_parse(const std::vector<std::uint8_t>& bytes, std::size_t length) {
    try {
        (void)tyloi::sq::parse(bytes.data(), length);
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

bool rejects_decode(const Weight& w, DenseTensor blob, int width, int base) {
    const auto& q = w.layout;
    try {
        (void)mxfp4_sq_dequant_cuda(blob, w.row_q, w.offsets, w.auxiliary, q.bits, q.outputs, width,
                                    base, w.q_sum, w.sq4_rows, true);
    } catch (const tyloi::cuda::Error&) {
        throw;
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

void decode_case(Case& value, TensorWork& work, Weight& w, std::vector<std::uint8_t>& raw,
                 Json& checks) {
    check(checks, 0, rejects_parse(raw, raw.size() - 1), {{"mutation", "truncate_last_byte"}});
    auto damaged = raw;
    damaged.push_back(0);
    check(checks, 1, rejects_parse(damaged, damaged.size()), {{"mutation", "append_zero_byte"}});
    int index = 2;
    for (int offset : {0, 3, 4, 5, 6, 7, 15, 23}) {
        damaged = raw;
        damaged[static_cast<std::size_t>(offset)] = 255;
        check(checks, index++, rejects_parse(damaged, damaged.size()),
              {{"offset", offset}, {"value", 255}});
    }
    auto host = work.keep(tyloi::cuda::from_blob(
        raw.data(), {std::int64_t(raw.size())},
        tyloi::TensorOptions().dtype(tyloi::kUInt8).device(tyloi::Device(tyloi::kCPU))));
    const auto& q = w.layout;
    check(checks, index++, rejects_decode(w, host, q.width, q.base), {{"mutation", "host_blob"}});
    check(checks, index++, rejects_decode(w, w.blob, q.width + 1, q.base),
          {{"mutation", "width_plus_one"}});
    check(checks, index++, rejects_decode(w, w.blob, q.width, 252), {{"mutation", "base_252"}});
    work.output("f32", work.keep(decode(w, true)));
    work.output("f16", work.keep(decode(w, false)));
    check(checks, index++, true, {{"observations", {"f32", "f16"}}});
    if (q.width >= 64) {
        const auto& spec = value.data().at("inputs").at("slice_rows");
        if (spec.at("dtype") != "int64" || spec.at("shape").size() != 1 ||
            spec.at("layout") != "contiguous")
            throw std::invalid_argument("SQ slice rows require contiguous rank-1 int64");
        const auto bytes = value.input("slice_rows");
        if (bytes.size() % sizeof(std::int64_t) != 0)
            throw std::invalid_argument("invalid SQ slice rows");
        std::vector<std::int64_t> selected(bytes.size() / sizeof(std::int64_t));
        std::memcpy(selected.data(), bytes.data(), bytes.size());
        const auto& config = value.data().at("config");
        const auto begin = config.at("slice_start").get<std::int64_t>();
        const auto end = config.at("slice_end").get<std::int64_t>();
        const auto sliced = tyloi::sq::select_rows(raw, selected, begin, end);
        const auto layout = tyloi::sq::parse(sliced.data(), sliced.size());
        const auto rows = tyloi::sq::row_metadata(sliced.data(), layout);
        std::vector<float> result(std::size_t(layout.outputs) * layout.width);
        for (int r = 0; r < layout.outputs; ++r)
            for (int c = 0; c < layout.width; ++c)
                result[std::size_t(r) * layout.width + c] =
                    tyloi::sq::decode_cpu(sliced.data(), layout, rows.q[r],
                                          rows.symbol_byte_offsets[r], rows.auxiliary_rows[r], c);
        value.output("slice_cpu", Json::array({layout.outputs, layout.width}), "float32",
                     result.data(), result.size() * sizeof(float));
        check(checks, index, true, {{"observation", "slice_cpu"}});
    }
}

void forward_case(Case& value, TensorWork& work, const Weight& w, Json& checks) {
    auto input = work.input("native_input");
    const auto& config = value.data().at("config");
    const auto dtype = config.at("input_dtype").get<std::string>();
    if ((dtype != "float16" && dtype != "float32") || input.dim() != 2 ||
        input.size(0) != config.at("rows").get<std::int64_t>() || input.size(1) != w.layout.width ||
        input.scalar_type() != (dtype == "float16" ? tyloi::kFloat16 : tyloi::kFloat32))
        throw std::invalid_argument("SQ4 input geometry or dtype differs from case");
    auto output = work.keep(project(w, input));
    work.output("output", output);
    check(checks, 0,
          output.dim() == 2 && output.size(0) == input.size(0) &&
              output.size(1) == w.layout.outputs && output.scalar_type() == input.scalar_type(),
          {{"observation", "output"}});
    if (!value.data().at("outputs").contains("graph_replay_output"))
        return;
    cuda_check(cudaDeviceSynchronize(), "SQ4 graph device synchronization");
    const auto stream = tyloi::cuda::stream_from_pool(false, work.device().index);
    tyloi::cuda::StreamGuard guard(stream);
    GraphRun run(stream);
    run.ordinary = work.keep(project(w, input));
    work.output("graph_ordinary_output", run.ordinary);
    if (!tyloi::cuda::default_context(work.device().index)->supports_async_allocations())
        throw std::runtime_error("SQ4 graph requires async allocations");
    run.graph.prepare_memory();
    run.warmup = project(w, input);
    cuda_check(cudaStreamSynchronize(stream.stream()), "SQ4 graph warmup synchronization");
    run.warmup = {};
    run.graph.capture_begin();
    run.result = project(w, input);
    run.graph.capture_end();
    run.graph.replay();
    run.graph.replay();
    cuda_check(cudaStreamSynchronize(stream.stream()), "SQ4 graph replay synchronization");
    work.output("graph_replay_output", run.result);
    check(
        checks, 1, true,
        {{"ordinary", "graph_ordinary_output"}, {"replay", "graph_replay_output"}, {"replays", 2}});
}

void moe_case(Case& value, TensorWork& work, const Weight& w, Json& checks) {
    const auto& config = value.data().at("config");
    const auto tokens = config.at("tokens").get<std::int64_t>();
    const auto routes = config.at("routes").get<std::int64_t>();
    const auto experts = config.at("n_experts").get<std::int64_t>();
    const auto local_experts = config.at("local_experts").get<std::int64_t>();
    const auto outputs = config.at("out_per_expert").get<std::int64_t>();
    if (tokens <= 0 || routes <= 0 || experts <= 0 || local_experts <= 0 || outputs <= 0 ||
        local_experts > std::numeric_limits<std::int64_t>::max() / outputs ||
        local_experts * outputs != w.layout.outputs)
        throw std::invalid_argument("invalid SQ4 expert geometry");
    auto input = work.input("native_input");
    auto ids = work.input("route_ids");
    auto local = work.input("expert_local");
    auto output = work.input("initial_output");
    const auto& q = w.layout;
    if (input.dim() != 2 || input.size(0) != tokens || input.size(1) != q.width || ids.dim() != 2 ||
        ids.size(0) != tokens || ids.size(1) != routes || local.dim() != 1 ||
        local.size(0) != experts || output.dim() != 3 || output.size(0) != tokens ||
        output.size(1) != routes || output.size(2) != outputs)
        throw std::invalid_argument("SQ4 expert inputs differ from case geometry");
    mxfp4_sq_moe_matmul_cuda(w.blob, w.row_q, w.offsets, w.auxiliary, input, ids, local, q.bits,
                             experts, local_experts, outputs, q.width, q.base, w.q_sum, w.sq4_rows,
                             output);
    work.output("routed_output", output);
    check(checks, 0,
          output.dim() == 3 && output.size(0) == tokens && output.size(1) == routes &&
              output.size(2) == outputs && output.scalar_type() == tyloi::kFloat16,
          {{"observation", "routed_output"}});
}
#endif
} // namespace

bool run_sq4(Case& value) {
    const auto op = value.data().at("op").get<std::string>();
    if (op != "sq4_decode" && op != "sq4_forward" && op != "sq4_backward" && op != "sq4_moe")
        return false;
    if (op == "sq4_backward")
        throw std::invalid_argument("SQ4 backward requires the Python JIT candidate");
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (value.data().at("source").value("generator", std::string()) !=
            "tests/numerics/sq4_ref.py" ||
        value.data().value("actual_role", std::string("native")) != "native")
        throw std::invalid_argument("unsupported native SQ4 case");
    const auto& environment = value.data().at("config").at("native_environment");
    if (!environment.is_object() || environment.size() != 1 ||
        environment.at("TYLOINFER_FORCE_SQ_TENSOR_CORE") != "1")
        throw std::invalid_argument("unsupported SQ4 native environment");
    ForceTensorCore force;
    auto raw = value.asset("weight");
    TensorWork work(value);
    auto weight = load(work, raw, value.data().at("config"));
    Json checks = Json::object();
    if (op == "sq4_decode")
        decode_case(value, work, weight, raw, checks);
    else if (op == "sq4_forward")
        forward_case(value, work, weight, checks);
    else
        moe_case(value, work, weight, checks);
    work.finish(op, {{"environment", environment}}, std::move(checks));
    return true;
#else
    throw std::runtime_error("SQ4 numerical cases require an MFQ compatibility build");
#endif
}
} // namespace tyloi::diagnostics::numerics
