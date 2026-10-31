#include "gguf_rotation.h"
#include "mfq_cuda_context.h"
#include "mfq_weight_memory.h"
#include "nepq.h"
#include "ggml.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace tb = mfq_tensor_backend;
namespace {
// Same normalized Sylvester butterfly as Strata's CPU hadamard_rotate
// (MIT, third_party/strata.LICENSE). The basis is per weight, with no global
// model state; adjoint and intermediate dtype rounding are explicit.
void rotate_cpu(float* values, std::int64_t rows, std::int64_t width, int block,
        const std::vector<std::int8_t>& signs, bool adjoint) {
    const float normalization = 1.f / std::sqrt(float(block));
    for (std::int64_t row = 0; row < rows; ++row) {
        auto* x = values + row * width;
        if (!adjoint && !signs.empty()) for (std::int64_t i = 0; i < width; ++i) x[i] *= signs[i];
        for (std::int64_t start = 0; start < width; start += block)
            for (int half = 1; half < block; half *= 2)
                for (int first = 0; first < block; first += 2 * half)
                    for (int i = 0; i < half; ++i) {
                        const float a = x[start + first + i], b = x[start + first + half + i];
                        x[start + first + i] = a + b; x[start + first + half + i] = a - b;
                    }
        for (std::int64_t i = 0; i < width; ++i)
            x[i] *= normalization * (adjoint && !signs.empty() ? float(signs[i]) : 1.f);
    }
}
void rotate_gpu(const GgufInputRotation& rotation, const tb::Tensor& input, const tb::Tensor& output, bool adjoint) {
    const MfqCudaGuard guard(input.get_device());
    const auto stream = mfq_get_current_cuda_stream(input.get_device());
    TYLOINFER_RUNTIME_CHECK(rotation.signs_cpu.empty() || (rotation.signs_device.defined() && rotation.signs_device.device() == input.device()),
        "GGUF Hadamard signs are not on the activation device");
    for (const auto& tensor : {input, output}) mfq_cuda_record_stream(tensor, stream);
    const std::int8_t* signs = nullptr;
    if (rotation.signs_device.defined()) { mfq_cuda_record_stream(rotation.signs_device, stream); signs = rotation.signs_device.data_ptr<std::int8_t>(); }
    const auto width = input.size(-1), rows = input.numel() / width;
    if (input.scalar_type() == tb::kFloat16)
        mfq::cuda::hadamard::apply_f16(input.data_ptr(), signs, output.data_ptr(), rows, width, rotation.block, adjoint, stream.stream());
    else mfq::cuda::hadamard::apply_f32(input.data_ptr<float>(), signs, output.data_ptr<float>(), rows, width, rotation.block, adjoint, stream.stream());
}
}

bool GgufInputRotation::equivalent(const GgufInputRotation& other) const {
    return block == other.block && signs_cpu == other.signs_cpu;
}
bool same_gguf_rotation(const std::shared_ptr<const GgufInputRotation>& a,
        const std::shared_ptr<const GgufInputRotation>& b) {
    return a==b || (a && b && a->equivalent(*b));
}
void GgufInputRotation::apply_cpu(float* values,std::int64_t rows,std::int64_t width,bool half,bool adjoint) const {
    if(block<=0 || (block&(block-1)) || rows<0 || width<=0 || width%block || rows>INT64_MAX/width ||
            (!signs_cpu.empty() && int64_t(signs_cpu.size())!=width))
        throw std::invalid_argument("GGUF CPU Hadamard geometry");
    rotate_cpu(values,rows,width,block,signs_cpu,adjoint);
    if(half)for(int64_t i=0;i<rows*width;++i)values[i]=ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i]));
}
void GgufInputRotation::apply_inplace(tb::Tensor input, bool adjoint) const {
    TYLOINFER_RUNTIME_CHECK(block > 0 && !(block & (block - 1)) && input.dim() >= 1 && input.is_contiguous() && input.size(-1) > 0 &&
        input.size(-1) % block == 0 && (signs_cpu.empty() || int64_t(signs_cpu.size()) == input.size(-1)) &&
        (input.scalar_type() == tb::kFloat32 || input.scalar_type() == tb::kFloat16),
        "GGUF Hadamard activation geometry/precision differs");
    const auto width = input.size(-1), rows = input.numel() / width;
    if (!rows) return;
    if (input.is_cuda()) {
        rotate_gpu(*this, input, input, adjoint);
    } else {
        if (input.scalar_type() == tb::kFloat32) rotate_cpu(input.data_ptr<float>(), rows, width, block, signs_cpu, adjoint);
        else {
            auto values = input.to(tb::kFloat32).contiguous();
            rotate_cpu(values.data_ptr<float>(), rows, width, block, signs_cpu, adjoint);
            input.copy_(values.to(tb::kFloat16));
        }
    }
}
tb::Tensor GgufInputRotation::apply(tb::Tensor input, bool adjoint) const {
    auto source = (input.scalar_type() == tb::kFloat16 ? input : input.to(tb::kFloat32)).contiguous();
    TYLOINFER_RUNTIME_CHECK(block > 0 && !(block & (block - 1)) && source.dim() >= 1 && source.size(-1) > 0 &&
        source.size(-1) % block == 0 && (signs_cpu.empty() || int64_t(signs_cpu.size()) == source.size(-1)),
        "GGUF Hadamard activation geometry differs");
    if (source.is_cuda()) {
        auto result = tb::empty_like(source);
        if (source.numel()) rotate_gpu(*this, source, result, adjoint);
        return result;
    }
    auto result = source.clone(); apply_inplace(result, adjoint);
    return result;
}

std::shared_ptr<const GgufInputRotation> load_gguf_input_rotation(const mfq::ModelSource& source,
        const std::string& name, std::int64_t width, std::int64_t first, std::int64_t last,
        int device, bool cuda) {
    mfq::HadamardTransform parsed;
    const mfq::HadamardTransform* transform=nullptr;
    if(const auto* cached=source.hadamard_transforms()) {
        const auto found=cached->find(name);
        if(found==cached->end())return {};
        transform=&found->second;
    } else {
        const auto& metadata=source.metadata();
        const auto found=metadata.find("gguf_hadamard");
        if(found==metadata.end()) {
            if(const auto required=metadata.find("gguf_requires_hadamard");required!=metadata.end() && required->second=="true")
                throw std::invalid_argument("GGUF Hadamard descriptors are missing");
            return {};
        }
        const auto transforms=nlohmann::json::parse(found->second);
        if(!transforms.contains(name))return {};
        parsed.block_size=transforms.at(name).at("block_size").get<int64_t>();
        const auto signs=transforms.at(name).at("signs").get<std::vector<int>>();
        for(int sign:signs) {
            if(sign!=1 && sign!=-1)throw std::invalid_argument("GGUF Hadamard sign value differs");
            parsed.signs.push_back(std::int8_t(sign));
        }
        transform=&parsed;
    }
    const auto block=transform->block_size;
    const auto& signs=transform->signs;
    if (block < 1 || block > INT_MAX || (block & (block - 1)) || first < 0 || last <= first || last > width ||
            first % block || (last - first) % block || (!signs.empty() && int64_t(signs.size()) != width) ||
            std::any_of(signs.begin(), signs.end(), [](int sign) { return sign != 1 && sign != -1; }))
        throw std::invalid_argument("GGUF Hadamard basis is incompatible with the projection slice");
    auto result = std::make_shared<GgufInputRotation>(); result->block = int(block);
    if (!signs.empty()) result->signs_cpu.assign(signs.begin() + first, signs.begin() + last);
    if (std::all_of(result->signs_cpu.begin(), result->signs_cpu.end(), [](int sign) { return sign == 1; }))
        result->signs_cpu.clear();
    if (block == 1 && result->signs_cpu.empty()) return {};
    if (cuda) {
        const MfqCudaGuard guard(device);
        if (block > 128) {
            int regular = 0, optin = 0;
            TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&regular, cudaDevAttrMaxSharedMemoryPerBlock, device));
            TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            if (std::uint64_t(block) * sizeof(float) > std::uint64_t(std::max(regular, optin)))
                throw std::invalid_argument("GGUF Hadamard block exceeds CUDA shared-memory capacity");
        }
        if (!result->signs_cpu.empty()) {
            const mfq::cuda::WeightAllocationScope weight_memory;
            auto host = tb::empty({int64_t(result->signs_cpu.size())}, tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt8));
            std::memcpy(host.data_ptr(), result->signs_cpu.data(), result->signs_cpu.size());
            result->signs_device = host.to(tb::Device(tb::kCUDA, device));
        }
    }
    return result;
}
