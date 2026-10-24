#include "storage/weights/loaders/weight.h"
#include "memory_tensor_source.h"

#include <iostream>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

static void check_dense_loading() {
    using namespace mfq_tensor_backend;
    MemoryTensorSource source;
    CudaExecutionContext execution;
    const uint32_t rank = 2;
    const int64_t shape[] = {2, 2};
    const size_t header = sizeof(rank) + sizeof(shape);
    auto same = [](Tensor value, const Tensor &expected) {
        value = value.cpu().contiguous();
        return value.scalar_type() == expected.scalar_type() && value.sizes() == expected.sizes() &&
               std::memcmp(value.data_ptr(), expected.data_ptr(), expected.nbytes()) == 0;
    };
    for (const auto &[name, dtype] : std::vector<std::pair<std::string, ScalarType>>{
             {"BF16", kBFloat16}, {"F16", kFloat16}, {"F32", kFloat32}, {"I64", kInt64}, {"I32", kInt32}}) {
        const auto expected = tensor(std::vector<float>{-3.25, 0, 1.125, 7}).reshape({2, 2}).to(dtype);
        source.records[0].dtype = name;
        source.bytes.resize(header + expected.nbytes());
        std::memcpy(source.bytes.data(), &rank, sizeof(rank));
        std::memcpy(source.bytes.data() + sizeof(rank), shape, sizeof(shape));
        std::memcpy(source.bytes.data() + header, expected.data_ptr(), expected.nbytes());
        source.records[0].nbytes = source.bytes.size();
        if (name == "F32" || name == "F16" || name == "BF16") {
            for (bool cpu_layer : {false, true}) {
                execution.loading_cpu_layer = cpu_layer;
                const auto linear = load_quant_linear(execution, source, "weight");
                check(linear.is_dense() && linear.dense.is_cuda() != cpu_layer &&
                          same(quant_linear_reference_weight(linear), expected),
                      "linear loader changed dtype, values or placement");
                if (!cpu_layer && name == "F32") {
                    auto input = tensor(std::vector<float>{1, 2}).reshape({1, 2}).to(kCUDA);
                    auto reference = tensor(std::vector<float>{-3.25, 15.125}).reshape({1, 2});
                    check(same(linear.forward(execution, input), reference), "loaded linear output changed");
                    auto group = load_quant_group(execution, source, {"weight", "weight"});
                    auto outputs = group.forward(execution, input);
                    check(outputs.size() == 2 && same(outputs[0], reference) && same(outputs[1], reference),
                          "loaded linear group output changed");
                    auto paired = load_paired_gate_up(execution, source, {"weight", "weight"}, linear);
                    outputs = paired.forward(execution, input);
                    check(outputs.size() == 2 && same(outputs[0], reference) && same(outputs[1], reference),
                          "paired projection loading changed");
                }
            }
            execution.loading_cpu_layer = false;
        }
        auto native = load_dense_native_gpu(execution, source, "weight");
        check(same(native, expected), "dense native dtype/value changed");
        const auto promoted = name == "BF16" || name == "F16" ? kFloat32 : dtype;
        for (bool cpu_layer : {false, true}) {
            execution.loading_cpu_layer = cpu_layer;
            auto value = load_dense_gpu(execution, source, "weight");
            check(value.is_cuda() != cpu_layer && value.scalar_type() == promoted &&
                      same(value, expected.to(promoted)), "dense load placement/promotion changed");
        }
    }
    const auto valid = source.bytes;
    auto rejects = [&] {
        source.records[0].nbytes = source.bytes.size();
        bool rejected = false;
        try { (void)load_dense_cpu(execution, source, "weight"); }
        catch (const std::exception&) { rejected = true; }
        check(rejected, "malformed dense tensor accepted");
    };
    for (size_t size : {size_t(0), size_t(3), header - 1, valid.size() - 1}) {
        source.bytes = valid;
        source.bytes.resize(size);
        rejects();
    }
    for (int64_t extent : {int64_t(-1), std::numeric_limits<int64_t>::max()}) {
        source.bytes = valid;
        std::memcpy(source.bytes.data() + sizeof(rank), &extent, sizeof(extent));
        rejects();
    }
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    try {
        check_dense_loading();
        TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
        std::cout << "PASS: dense and linear loading preserve dtype, placement, outputs and validation\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
