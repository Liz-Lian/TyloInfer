#include "mfq.h"
#include "tensor.h"

#include <limits>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/backends/cuda/kernels/float_projection.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/ops/projection/nint.h"
#include "engine/backends/cuda/ops/projection/vq.h"
#include "engine/models/source.h"
#include "engine/weights/encoding/quant/mfe_blob.h"
#include "engine/weights/encoding/quant/mfe_quant_expert_store.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/encoding/quant/nvq_rows.h"
#include "engine/weights/encoding/nint.h"
#include "engine/weights/quant.h"
#include "engine/weights/source/read.h"
#endif

namespace tyloi::diagnostics::numerics {
namespace {
#if TYLOINFER_ENABLE_MFQ_COMPAT
void input_contract(const Case& value, const char* name, const Json& shape, const char* dtype) {
    const auto& spec = value.data().at("inputs").at(name);
    if (spec.at("shape") != shape || spec.at("dtype") != dtype || spec.at("layout") != "contiguous")
        throw std::invalid_argument(std::string("synthetic NINT field differs: ") + name);
}

struct CountingReader {
    tyloi::source::Source::TensorReader reader;
    std::size_t* bytes;
    void operator()(std::size_t offset, std::uint8_t* output, std::size_t count) const {
        if (*bytes > std::numeric_limits<std::size_t>::max() - count)
            throw std::overflow_error("row read count overflow");
        *bytes += count;
        reader(offset, reinterpret_cast<std::byte*>(output), count);
    }
};

tyloi::DenseTensor cpu_project(const std::string& dtype, const std::vector<std::uint8_t>& payload,
                               const tyloi::DenseTensor& input, int rows, int width) {
    CudaProfiler profiler;
    if (dtype == "NINT") {
        const auto weight = to_device_nint(tyloi::nint::unpack(payload), false);
        if (weight.out != rows || weight.neuron_len != width)
            throw std::invalid_argument("CPU NINT payload geometry differs");
        return nint_matmul(profiler, weight, input).to(tyloi::kFloat32).contiguous();
    }
    const auto weight = to_device_nvq(tyloi::nvq::unpack(payload, dtype), false);
    if (weight.out != rows || weight.neuron_len != width)
        throw std::invalid_argument("CPU NVQ payload geometry differs");
    return nvq_matmul(profiler, weight, input).to(tyloi::kFloat32).contiguous();
}

void cpu_output(Case& value, const std::string& name, const tyloi::DenseTensor& tensor,
                std::int64_t batch, std::int64_t rows) {
    if (!tensor.defined() || tensor.is_cuda() || tensor.scalar_type() != tyloi::kFloat32 ||
        !tensor.is_contiguous() || tensor.dim() != 2 || tensor.size(0) != batch ||
        tensor.size(1) != rows)
        throw std::runtime_error("CPU MFQ projection returned unexpected shape, dtype or device");
    value.output(name, Json::array({batch, rows}), "float32", tensor.data_ptr(),
                 static_cast<std::size_t>(tensor.numel()) * sizeof(float));
}

void cpu_mfq(Case& value) {
    const auto& config = value.data().at("config");
    const int rows = config.at("rows"), width = config.at("width"), experts = config.at("experts");
    if (rows <= 17 || width <= 0 || experts <= 0 ||
        std::int64_t(rows) * experts > std::numeric_limits<int>::max())
        throw std::invalid_argument("invalid CPU MFQ geometry");
    input_contract(value, "input", Json::array({3, width}), "float32");
    auto raw_input = value.input("input");
    std::vector<float> input_values(raw_input.size() / sizeof(float));
    std::memcpy(input_values.data(), raw_input.data(), raw_input.size());
    auto input = tyloi::cuda::from_blob(
        input_values.data(), {3, width},
        tyloi::TensorOptions().dtype(tyloi::kFloat32).device(tyloi::Device(tyloi::kCPU)));
    const auto source = tyloi::open_model_source(value.asset_path("model"));
    const auto& record = tyloi::require_tensor(*source, "linear.weight");
    const auto dtype = config.at("payload_dtype").get<std::string>();
    if (record.dtype != config.at("model_dtype").get<std::string>() ||
        source->tensors().size() != 1)
        throw std::invalid_argument("MFQ source tensor identity differs");
    const auto model_bytes = tyloi::read_tensor(*source, "linear.weight");
    const auto payload = value.asset("weight");
    std::vector<std::uint8_t> actual_payload;
    if (experts == 1) {
        actual_payload = model_bytes;
    } else {
        const auto model = tyloi::mfe::unpack(model_bytes);
        if (model.n_experts != experts || model.out_per_expert != rows ||
            model.neuron_len != width || model.pools.size() != 1 || model.pools[0].dtype != dtype ||
            model.pools[0].expert_ids.size() != std::size_t(experts))
            throw std::invalid_argument("MFQ expert container geometry differs");
        const auto order = config.at("expert_order").get<std::vector<int>>();
        if (order != model.pools[0].expert_ids)
            throw std::invalid_argument("MFQ expert order differs");
        for (int i = 0; i < experts; ++i)
            if (order[std::size_t(i)] != i)
                throw std::invalid_argument("unsupported MFQ expert order");
        actual_payload = model.pools[0].payload;
    }
    if (actual_payload != payload)
        throw std::invalid_argument("MFQ model and weight payload differ");
    std::vector<float> experts_output;
    std::size_t expert_reads = 0;
    if (experts == 1) {
        auto output = cpu_project(dtype, actual_payload, input, rows, width);
        cpu_output(value, "output", output, 3, rows);
    } else {
        CountingReader read{source->tensor_reader("linear.weight"), &expert_reads};
        tyloi::MfeQuantExpertStore store(record.nbytes, read);
        if (store.num_experts() != experts || store.out_per_expert() != rows ||
            store.neuron_len() != width || store.pool_count() != 1)
            throw std::runtime_error("loaded MFQ expert container differs");
        experts_output.resize(std::size_t(3) * rows * experts);
        for (int expert = 0; expert < experts; ++expert) {
            if (store.expert_pool(expert) != 0 || store.expert_local(expert) != expert)
                throw std::runtime_error("loaded MFQ expert mapping differs");
            auto part = store.read_expert(expert);
            if (part.dtype != dtype)
                throw std::runtime_error("MFQ expert dtype differs");
            auto projected = cpu_project(dtype, part.payload, input, rows, width);
            if (projected.dim() != 2 || projected.size(0) != 3 || projected.size(1) != rows ||
                projected.scalar_type() != tyloi::kFloat32 || !projected.is_contiguous() ||
                projected.is_cuda())
                throw std::runtime_error("MFQ expert projection shape differs");
            for (int batch = 0; batch < 3; ++batch)
                std::memcpy(experts_output.data() + (std::size_t(batch) * experts + expert) * rows,
                            projected.data_ptr<float>() + std::size_t(batch) * rows,
                            std::size_t(rows) * sizeof(float));
        }
        value.output("output", Json::array({3, std::int64_t(rows) * experts}), "float32",
                     experts_output.data(), experts_output.size() * sizeof(float));
    }
    Json checks = Json::object();
    if (experts == 1) {
        std::size_t reads = 0;
        CountingReader read{source->tensor_reader("linear.weight"), &reads};
        std::unique_ptr<tyloi::NintRows> nint;
        std::unique_ptr<tyloi::NvqRows> nvq;
        if (dtype == "NINT")
            nint = std::make_unique<tyloi::NintRows>(record.nbytes, read);
        else
            nvq = std::make_unique<tyloi::NvqRows>(record.nbytes, read);
        const auto index_reads = reads;
        if (index_reads >= record.nbytes)
            throw std::runtime_error("range index read entire weight payload");
        const std::pair<int, int> ranges[] = {
            {1, 17}, {rows / 2, rows / 2 + 9}, {rows - 11, rows}, {0, rows}};
        for (int i = 0; i < 4; ++i) {
            const auto [begin, end] = ranges[i];
            auto slice =
                nint ? nint->slice_rows_blob(begin, end) : nvq->slice_rows_blob(begin, end);
            const auto size =
                nint ? nint->row_range_nbytes(begin, end) : nvq->row_range_nbytes(begin, end);
            if (slice.size() != size || (begin == 0 && end == rows && slice != model_bytes))
                throw std::runtime_error("MFQ canonical row slice identity differs");
            auto result = cpu_project(dtype, slice, input, end - begin, width);
            cpu_output(value, "slice_" + std::to_string(i), result, 3, end - begin);
        }
        checks["native_0"] = {
            {"status", "pass"},
            {"evidence", {{"dtype", dtype}, {"device", "cpu"}, {"compressed", true}}}};
        checks["native_1"] = {
            {"status", "pass"},
            {"evidence",
             {{"index_bytes", index_reads},
              {"total_read_bytes", reads},
              {"ranges", {{1, 17}, {rows / 2, rows / 2 + 9}, {rows - 11, rows}, {0, rows}}},
              {"canonical_full_slice", true}}}};
    } else {
        checks["native_0"] = {{"status", "pass"},
                              {"evidence",
                               {{"category", "expert_container_extension"},
                                {"expert_order", config.at("expert_order")},
                                {"payload_bytes", payload.size()},
                                {"source_read_bytes", expert_reads}}}};
    }
    value.finish({{"target", "mfq-numerics"},
                  {"entrypoint", "cpu_quant_projection"},
                  {"device", "cpu"},
                  {"result_precision", "float16 expanded to float32"},
                  {"build", {{"mfq_compat", true}}}},
                 std::move(checks));
}

void synthetic(Case& value) {
    const auto& config = value.data().at("config");
    const auto rows = config.at("outputs").get<std::int64_t>();
    const auto width = config.at("width").get<std::int64_t>();
    const auto group = config.at("group_size").get<std::int64_t>();
    const auto batch = config.at("batch").get<std::int64_t>();
    const auto zero = config.at("zero").get<bool>();
    if (rows <= 0 || width <= 0 || group <= 0 || batch <= 0 ||
        width > std::numeric_limits<std::int64_t>::max() - group + 1)
        throw std::invalid_argument("invalid synthetic NINT geometry");
    const auto groups = (width + group - 1) / group;
    const auto& packed_spec = value.data().at("inputs").at("q_packed");
    if (packed_spec.at("dtype") != "uint8" || packed_spec.at("layout") != "contiguous" ||
        !packed_spec.at("shape").is_array() || packed_spec.at("shape").size() != 1)
        throw std::invalid_argument("invalid synthetic NINT packed stream");
    const auto packed = value.input("q_packed");
    if (packed.size() < 8)
        throw std::invalid_argument("synthetic NINT packed stream lacks its guard bytes");
    for (std::size_t i = packed.size() - 8; i < packed.size(); ++i)
        if (packed[i] != 0)
            throw std::invalid_argument("synthetic NINT packed stream lacks its guard bytes");
    input_contract(value, "input", Json::array({batch, width}), "float32");
    if (zero) {
        input_contract(value, "zero_scale", Json::array({rows, groups}), "float16");
    } else {
        input_contract(value, "row_q_bits", Json::array({rows}), "uint8");
        input_contract(value, "row_q_bit_offsets", Json::array({rows}), "int64");
        input_contract(value, "sub_scale", Json::array({rows, groups}), "uint8");
        input_contract(value, "sub_min", Json::array({rows, groups}), "uint8");
        input_contract(value, "neuron_scale", Json::array({rows}), "float32");
        input_contract(value, "neuron_min", Json::array({rows}), "float32");
    }
    TensorWork work(value);
    tyloi::NintTensor weight;
    weight.out = rows;
    weight.neuron_len = width;
    weight.gs = group;
    weight.ng = groups;
    weight.q8_zero = zero;
    weight.q_packed = work.input("q_packed");
    if (zero) {
        weight.q8_zero_scale = work.input("zero_scale");
    } else {
        weight.row_q_bits = work.input("row_q_bits");
        weight.row_q_bit_offsets = work.input("row_q_bit_offsets");
        weight.sub_scale = work.input("sub_scale");
        weight.sub_min = work.input("sub_min");
        weight.neuron_scale = work.input("neuron_scale");
        weight.neuron_min = work.input("neuron_min");
    }
    const auto input = work.input("input");
    if (input.dim() != 2 || input.size(0) != batch || input.size(1) != width ||
        input.scalar_type() != tyloi::kFloat32)
        throw std::invalid_argument("synthetic NINT input geometry differs");
    auto split = work.keep(nint_float_projection_cuda(weight, input));
    auto unsplit = work.keep(nint_float_projection_cuda(weight, input, false));
    work.output("split", split);
    work.output("unsplit", unsplit);
    Json checks = {
        {"native_0", {{"status", "pass"}, {"evidence", {{"parallel_groups", true}}}}},
        {"native_1", {{"status", "pass"}, {"evidence", {{"parallel_groups", false}}}}},
    };
    work.finish("nint_float_projection_cuda", {{"q8_zero", zero}}, std::move(checks));
}
#endif
} // namespace

bool run_mfq(Case& value) {
    const auto op = value.data().at("op").get<std::string>();
    if (op != "mfq_cpu_quant_linear" && op != "nint_synthetic_projection")
        return false;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (op == "nint_synthetic_projection") {
        synthetic(value);
        return true;
    }
    cpu_mfq(value);
    return true;
#else
    throw std::runtime_error("MFQ numerical cases require an MFQ compatibility build");
#endif
}
} // namespace tyloi::diagnostics::numerics
