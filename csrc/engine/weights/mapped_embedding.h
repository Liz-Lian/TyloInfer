#pragma once
#include <atomic>
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/native/memory.h"
#include <cstring>
#include <memory>

// Storage strategy adapted from Strata NativeEmbed::load/gather_dev:
// references/strata/src/core/native_head.cpp, MIT license in
// LICENSES/Strata-MIT.txt. Preserve MFQ's native floating dtype
// and gather kernel; only the read-only embedding table changes placement.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda {
inline std::atomic<std::uint64_t> mapped_embedding_host_bytes{0};
}
#endif
inline ::tyloi::DenseTensor map_embedding_storage(const ::tyloi::DenseTensor& cpu,int device,
        bool device_read_control=false) {
    TYLOINFER_RUNTIME_CHECK(cpu.is_cpu() && cpu.is_contiguous(),
        "mapped embedding requires contiguous CPU storage");
    const auto bytes=std::size_t(cpu.numel())*cpu.element_size();
    TYLOINFER_RUNTIME_CHECK(bytes>0,"mapped embedding table is empty");
    struct Owner {
        tyloi::cuda::HostBuffer host;
        ::tyloi::DenseTensor device_copy;
        std::size_t bytes;
        explicit Owner(std::size_t count):host(count,true),bytes(count) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            tyloi::cuda::charge_tensor_host_bytes(bytes);
            tyloi::cuda::mapped_embedding_host_bytes.fetch_add(bytes,std::memory_order_relaxed);
#endif
        }
        ~Owner() {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            tyloi::cuda::tensor_host_bytes.fetch_sub(bytes);
            tyloi::cuda::mapped_embedding_host_bytes.fetch_sub(bytes,std::memory_order_relaxed);
#endif
        }
    };
    MfqCudaGuard guard(device);
    auto owner=std::make_shared<Owner>(bytes);
    std::memcpy(owner->host.data(),cpu.data_ptr(),bytes);
    void* alias=nullptr;
    TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&alias,owner->host.data(),0));
    TYLOINFER_RUNTIME_CHECK(alias,"mapped embedding has no device alias");
    const auto gpu_device=::tyloi::Device(::tyloi::kCUDA,device);
    // Diagnostic control: retain the identical mapped allocation, but gather
    // from a GPU copy. This separates pinning/placement from remote GPU reads.
    if(device_read_control) {
        owner->device_copy=cpu.to(gpu_device);
        alias=owner->device_copy.data_ptr();
    }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto storage=std::make_shared<::tyloi::TensorStorage>();
    storage->owner=std::move(owner);storage->base=alias;storage->bytes=bytes;storage->device=gpu_device;
    return ::tyloi::DenseTensor(storage,::tyloi::make_contiguous_view(alias,cpu.sizes(),cpu.scalar_type(),gpu_device));
#else
    return ::tyloi::cuda::from_blob(alias,cpu.sizes(),[owner](void*){},cpu.options().device(gpu_device));
#endif
}

inline std::shared_ptr<const ::tyloi::Tensor> map_dense_embedding(const ::tyloi::DenseTensor& cpu,int device,
        bool device_read_control=false) {
    TYLOINFER_RUNTIME_CHECK(cpu.dim()==2 &&
        (cpu.scalar_type()==::tyloi::kFloat16 || cpu.scalar_type()==::tyloi::kBFloat16 || cpu.scalar_type()==::tyloi::kFloat32),
        "mapped embedding requires a rank-2 floating table");
    return std::make_shared<::tyloi::DenseTensor>(map_embedding_storage(cpu,device,device_read_control));
}

inline std::shared_ptr<const ::tyloi::Tensor> map_nint_embedding(const std::shared_ptr<const ::tyloi::Tensor>& result,int device,bool device_read_control=false) {
    TYLOINFER_RUNTIME_CHECK(
        dynamic_cast<const tyloi::NintTensor*>(result.get()) != nullptr,
        "mapped NINT embedding requires an unsharded CPU table");
    auto mapped=std::make_shared<tyloi::NintTensor>(
        static_cast<const tyloi::NintTensor&>(*result));
    auto* w=mapped.get();
    TYLOINFER_RUNTIME_CHECK(w->q_packed.is_cpu(),
        "mapped NINT embedding requires an unsharded CPU table");
    // Only selected vocabulary rows cross PCIe. Keep small row descriptors on
    // the GPU so their dependent loads do not add host round trips.
    for (auto* field : {&w->q_packed, &w->sub_scale, &w->sub_min, &w->q8_zero_scale})
        if(field->defined())*field=map_embedding_storage(*field,device,device_read_control);
    for (auto* field : {&w->row_q_bits, &w->row_q_bit_offsets, &w->neuron_scale, &w->neuron_min})
        if(field->defined())*field=field->to(::tyloi::Device(::tyloi::kCUDA,device));
    return mapped;
}
