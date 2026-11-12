#include "tensor.h"
#include "engine/backends/cuda/executor/executor.h"

#include <cuda_runtime_api.h>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
/** Map the supported fixed-input binary encodings to native scalar types. */
tyloi::ScalarType scalar(const std::string& name) {
    if (name == "float32")
        return tyloi::kFloat32;
    if (name == "float16")
        return tyloi::kFloat16;
    if (name == "int64")
        return tyloi::kInt64;
    throw std::invalid_argument("unsupported basic tensor dtype: " + name);
}
/** Map an observed native scalar type to its logical name and byte width. */
std::pair<std::string, std::size_t> dtype(tyloi::ScalarType value) {
    if (value == tyloi::kFloat32)
        return {"float32", 4};
    if (value == tyloi::kFloat16)
        return {"float16", 2};
    if (value == tyloi::kInt64)
        return {"int64", 8};
    throw std::invalid_argument("unsupported basic output dtype");
}
} // namespace

struct TensorWork::Impl {
    Case& value;
    int ordinal;
    tyloi::cuda::Executor executor;
    std::unique_ptr<tyloi::Executor::Scope> scope;
    std::vector<std::vector<std::uint8_t>> host;
    std::vector<tyloi::DenseTensor> tensors;
    bool finished = false;
    Impl(Case& source, int device)
        : value(source), ordinal(device), executor(tyloi::cuda::Executor::Options{}, device),
          scope(executor.enter(device)) {}
    ~Impl() noexcept {
        if (!finished) {
            try {
                executor.drain();
            } catch (const std::exception& error) {
                std::cerr << "numerics cleanup: " << error.what() << '\n';
            } catch (...) {
                std::cerr << "numerics cleanup: unknown error\n";
            }
        }
    }
};

TensorWork::TensorWork(Case& value) {
    int ordinal = 0;
    cuda_check(cudaGetDevice(&ordinal), "cudaGetDevice");
    impl_ = std::make_unique<Impl>(value, ordinal);
}
TensorWork::~TensorWork() noexcept = default;
tyloi::Device TensorWork::device() const { return tyloi::Device(tyloi::kCUDA, impl_->ordinal); }
tyloi::DenseTensor TensorWork::keep(tyloi::DenseTensor value) {
    if (value.defined())
        impl_->tensors.push_back(value);
    return value;
}
tyloi::DenseTensor TensorWork::input(const std::string& name) {
    const auto& spec = impl_->value.data().at("inputs").at(name);
    if (spec.at("layout") != "contiguous")
        throw std::invalid_argument("basic worker requires contiguous fixed inputs");
    const auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
    const auto type = scalar(spec.at("dtype").get<std::string>());
    impl_->host.push_back(impl_->value.input(name));
    auto& bytes = impl_->host.back();
    if (bytes.empty())
        throw std::invalid_argument("basic worker requires nonempty input tensors");
    auto cpu = keep(tyloi::cuda::from_blob(
        bytes.data(), shape,
        tyloi::TensorOptions().dtype(type).device(tyloi::Device(tyloi::kCPU))));
    auto device_value = keep(cpu.to(device()));
    return keep(device_value.contiguous());
}
void TensorWork::output(const std::string& name, tyloi::DenseTensor value) {
    if (!value.defined())
        throw std::invalid_argument("undefined native observation");
    value = keep(std::move(value));
    auto copied = keep(value.to(tyloi::kCPU));
    auto host = keep(copied.contiguous());
    impl_->executor.drain();
    const auto [logical, element] = dtype(host.scalar_type());
    const Json shape = host.sizes().vec();
    impl_->value.output(name, shape, logical, host.data_ptr(), std::size_t(host.numel()) * element);
}
void TensorWork::finish(const std::string& entrypoint, Json details) {
    impl_->executor.drain();
    cudaDeviceProp properties{};
    int runtime = 0, driver = 0;
    cuda_check(cudaGetDeviceProperties(&properties, impl_->ordinal), "cudaGetDeviceProperties");
    cuda_check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    cuda_check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    Json execution = {{"target", "mfq-numerics"},
                      {"entrypoint", entrypoint},
                      {"device", impl_->ordinal},
                      {"device_name", properties.name},
                      {"capability", {properties.major, properties.minor}},
                      {"cuda_runtime", runtime},
                      {"cuda_driver", driver},
                      {"build", {{"native_cuda", true}, {"cuda_runtime_version", CUDART_VERSION}}},
                      {"details", std::move(details)}};
    impl_->value.finish(std::move(execution), Json::object());
    impl_->finished = true;
}
} // namespace tyloi::diagnostics::numerics
