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
    if (name == "bool")
        return tyloi::kBool;
    if (name == "float32")
        return tyloi::kFloat32;
    if (name == "float16")
        return tyloi::kFloat16;
    if (name == "bfloat16")
        return tyloi::kBFloat16;
    if (name == "int8")
        return tyloi::kInt8;
    if (name == "uint8")
        return tyloi::kUInt8;
    if (name == "int32")
        return tyloi::kInt32;
    if (name == "int64")
        return tyloi::kInt64;
    throw std::invalid_argument("unsupported input tensor dtype: " + name);
}
/** Map an observed native scalar type to its logical name and byte width. */
std::pair<std::string, std::size_t> dtype(tyloi::ScalarType value) {
    if (value == tyloi::kBool)
        return {"bool", 1};
    if (value == tyloi::kFloat32)
        return {"float32", 4};
    if (value == tyloi::kFloat16)
        return {"float16", 2};
    if (value == tyloi::kBFloat16)
        return {"bfloat16", 2};
    if (value == tyloi::kUInt8)
        return {"uint8", 1};
    if (value == tyloi::kInt32)
        return {"int32", 4};
    if (value == tyloi::kInt64)
        return {"int64", 8};
    throw std::invalid_argument("unsupported output tensor dtype");
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
    const auto shape = spec.at("shape").get<std::vector<std::int64_t>>();
    const auto type = scalar(spec.at("dtype").get<std::string>());
    impl_->host.push_back(impl_->value.input(name));
    auto& bytes = impl_->host.back();
    if (type == tyloi::kBFloat16)
        bytes = pack_bf16(bytes);
    const auto options = tyloi::TensorOptions().dtype(type);
    tyloi::DenseTensor result;
    if (bytes.empty())
        result = keep(tyloi::cuda::empty(shape, options.device(device())));
    else {
        auto cpu = keep(tyloi::cuda::from_blob(bytes.data(), shape,
                                               options.device(tyloi::Device(tyloi::kCPU))));
        result = keep(cpu.to(device()));
        result = keep(result.contiguous());
    }
    const auto& layout = spec.at("layout");
    if (layout == "contiguous")
        return result;
    if (!layout.is_object() || shape.size() < 2 || layout.at("offset") != 0)
        throw std::invalid_argument("unsupported fixed input layout");
    // Rebuild the source transpose-contiguous-transpose layout from its logical values.
    auto transposed = keep(result.transpose(-1, -2));
    auto backing = keep(transposed.contiguous());
    result = keep(backing.transpose(-1, -2));
    const auto strides = layout.at("strides").get<std::vector<std::int64_t>>();
    if (strides.size() != shape.size() || result.byte_offset() != 0)
        throw std::invalid_argument("fixed input layout differs from source");
    for (std::size_t i = 0; i < strides.size(); ++i)
        if (result.stride(static_cast<std::int64_t>(i)) != strides[i])
            throw std::invalid_argument("fixed input strides differ from source");
    return result;
}
void TensorWork::output(const std::string& name, tyloi::DenseTensor value) {
    if (!value.defined())
        throw std::invalid_argument("undefined native observation");
    value = keep(std::move(value));
    auto contiguous = keep(value.contiguous());
    auto copied = keep(contiguous.to(tyloi::kCPU));
    auto host = keep(copied.contiguous());
    impl_->executor.drain();
    const auto [logical, element] = dtype(host.scalar_type());
    const Json shape = host.sizes().vec();
    const auto bytes = std::size_t(host.numel()) * element;
    if (host.scalar_type() == tyloi::kBFloat16) {
        const auto values = unpack_bf16({static_cast<const std::uint8_t*>(host.data_ptr()), bytes});
        impl_->value.output(name, shape, logical, values.data(), values.size());
    } else
        impl_->value.output(name, shape, logical, host.data_ptr(), bytes);
}
void TensorWork::finish(const std::string& entrypoint, Json details, Json checks) {
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
    impl_->value.finish(std::move(execution), std::move(checks));
    impl_->finished = true;
}
} // namespace tyloi::diagnostics::numerics
