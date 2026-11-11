// Adapted from Strata (MIT); see LICENSES/Strata-MIT.txt.
#include "engine/backends/cuda/native/vmm.h"

#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include <cuda.h>
#include <cuda_runtime.h>

#include <mutex>

namespace tyloi::cuda::vmm {
namespace {

struct Api {
    bool ok = false;
    decltype(&cuDeviceGetAttribute) attr = nullptr;
    decltype(&cuMemGetAllocationGranularity) granularity = nullptr;
    decltype(&cuMemAddressReserve) reserve = nullptr;
    decltype(&cuMemAddressFree) free_va = nullptr;
    decltype(&cuMemCreate) create = nullptr;
    decltype(&cuMemRelease) release = nullptr;
    decltype(&cuMemMap) map = nullptr;
    decltype(&cuMemUnmap) unmap = nullptr;
    decltype(&cuMemSetAccess) access = nullptr;
};

template <class F> bool resolve(const char* name, F& f) {
    cudaDriverEntryPointQueryResult q{};
    void* p = nullptr;
#if CUDART_VERSION >= 12050
    const cudaError_t e = cudaGetDriverEntryPointByVersion(name, &p, 12000, cudaEnableDefault, &q);
#else   // the versioned query arrived in CUDA 12.5
    const cudaError_t e = cudaGetDriverEntryPoint(name, &p, cudaEnableDefault, &q);
#endif
    if (e != cudaSuccess || q != cudaDriverEntryPointSuccess || p == nullptr) {
        (void) cudaGetLastError();
        return false;
    }
    f = (F) p;
    return true;
}

// never destroyed: the K/V pools and the cache release their chunks from static destructors at exit
const Api& api() {
    static Api& a = *new Api;
    static std::once_flag once;
    std::call_once(once, [] {
        if (!resolve("cuDeviceGetAttribute", a.attr) || !resolve("cuMemGetAllocationGranularity", a.granularity) ||
            !resolve("cuMemAddressReserve", a.reserve) || !resolve("cuMemAddressFree", a.free_va) ||
            !resolve("cuMemCreate", a.create) || !resolve("cuMemRelease", a.release) || !resolve("cuMemMap", a.map) ||
            !resolve("cuMemUnmap", a.unmap) || !resolve("cuMemSetAccess", a.access))
            return;
        a.ok = true;
    });
    return a;
}

}  // namespace

namespace {
uint64_t device_granularity(int device) {
    const Api& a = api();
    if (!a.ok) return 0;
    int supported = 0;
    if (a.attr(&supported, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, (CUdevice) device) != CUDA_SUCCESS || !supported)
        return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    size_t gran = 0;
    return a.granularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS ? gran : 0;
}
VmmChunk allocate_chunk(int device, uint64_t gran) {
    const Api& a = api();
    if (!a.ok || !gran) return 0;
    CUmemAllocationProp prop{};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    CUmemGenericAllocationHandle h = 0;
    if (a.create(&h, (size_t) gran, &prop, 0) != CUDA_SUCCESS) return 0;
    return (VmmChunk) h;
}
}

bool vmm_available() { return vmm_granularity() != 0; }
uint64_t vmm_granularity() {
    int device = -1;
    return cudaGetDevice(&device) == cudaSuccess ? device_granularity(device) : 0;
}
VmmChunk vmm_chunk_new() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 0;
    return allocate_chunk(device, device_granularity(device));
}
VmmChunk VmmRange::new_chunk() const { return allocate_chunk(device_, gran_); }

void vmm_chunk_free(VmmChunk h) {
    if (h != 0 && api().ok) api().release((CUmemGenericAllocationHandle) h);
}

bool VmmRange::reserve(uint64_t bytes) {
    release();
    const Api& a = api();
    int device = -1;
    if (!a.ok || bytes == 0 || cudaGetDevice(&device) != cudaSuccess) return false;
    const uint64_t gran = device_granularity(device);
    if (!gran) return false;
    const uint64_t n = (bytes + gran - 1) / gran;
    CUdeviceptr p = 0;
    if (a.reserve(&p, (size_t) (n * gran), 0, 0, 0) != CUDA_SUCCESS) return false;
    base_ = (unsigned long long) p;
    device_ = device;
    gran_ = gran;
    h_.assign((size_t) n, 0);
    return true;
}

void VmmRange::release() {
    if (base_ == 0) return;
    const Api& a = api();
    for (int64_t i = 0; i < chunks(); ++i) vmm_chunk_free(unmap(i));
    a.free_va((CUdeviceptr) base_, (size_t) ((uint64_t) h_.size() * gran_));
    base_ = 0;
    gran_ = 0;
    device_ = -1;
    h_.clear();
}

int64_t VmmRange::mapped_count() const {
    int64_t n = 0;
    for (const VmmChunk h : h_) n += h != 0;
    return n;
}

bool VmmRange::map_one(int64_t i, VmmChunk h) {
    const Api& a = api();
    if (a.map((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_, 0, (CUmemGenericAllocationHandle) h, 0) !=
        CUDA_SUCCESS)
        return false;
    h_[(size_t) i] = h;
    return true;
}

bool VmmRange::set_access(int64_t lo, int64_t hi) {
    if (hi <= lo) return true;
    const Api& a = api();
    CUmemAccessDesc d{};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = device_;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return a.access((CUdeviceptr) (base_ + (uint64_t) lo * gran_), (size_t) ((uint64_t) (hi - lo) * gran_), &d, 1) ==
           CUDA_SUCCESS;
}

bool VmmRange::commit_run(int64_t lo, int64_t hi) {
    if (set_access(lo, hi)) return true;
    for (int64_t c = lo; c < hi; ++c) vmm_chunk_free(unmap(c));
    return false;
}

VmmChunk VmmRange::unmap(int64_t i) {
    if (!mapped(i)) return 0;
    const Api& a = api();
    const VmmChunk h = h_[(size_t) i];
    if (a.unmap((CUdeviceptr) (base_ + (uint64_t) i * gran_), (size_t) gran_) != CUDA_SUCCESS) return 0;
    h_[(size_t) i] = 0;
    return h;
}

}  // namespace tyloi::cuda::vmm

#else  // HIP: no virtual memory here; the elastic K/V stays off

namespace tyloi::cuda::vmm {
bool vmm_available() { return false; }
uint64_t vmm_granularity() { return 0; }
VmmChunk vmm_chunk_new() { return 0; }
VmmChunk VmmRange::new_chunk() const { return 0; }
void vmm_chunk_free(VmmChunk) {}
bool VmmRange::reserve(uint64_t) { return false; }
void VmmRange::release() {}
int64_t VmmRange::mapped_count() const { return 0; }
bool VmmRange::map_one(int64_t, VmmChunk) { return false; }
bool VmmRange::set_access(int64_t, int64_t) { return false; }
bool VmmRange::commit_run(int64_t, int64_t) { return false; }
VmmChunk VmmRange::unmap(int64_t) { return 0; }
}  // namespace tyloi::cuda::vmm

#endif
