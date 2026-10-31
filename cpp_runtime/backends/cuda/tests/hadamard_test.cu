#include "../../../../tyloinfer/kernels/cuda/nepq.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Device {
    T* data = nullptr;
    explicit Device(std::size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
// Original shared-memory butterfly order, independent of the warp layout.
template<class T>
__global__ void reference(const T* input, const std::int8_t* signs, T* output,
        int width, int block, bool adjoint) {
    extern __shared__ float values[];
    const auto start = std::int64_t(blockIdx.x) * block;
    for (int j = threadIdx.x; j < block; j += blockDim.x) {
        float value = float(input[start + j]);
        if (!adjoint && signs) value *= signs[(start + j) % width];
        values[j] = value;
    }
    __syncthreads();
    for (int half = 1; half < block; half *= 2) {
        for (int j = threadIdx.x; j < block / 2; j += blockDim.x) {
            const int a = (j / half) * (2 * half) + j % half, b = a + half;
            const float x = values[a], y = values[b];
            values[a] = x + y; values[b] = x - y;
        }
        __syncthreads();
    }
    for (int j = threadIdx.x; j < block; j += blockDim.x) {
        float value = values[j] * rsqrtf(float(block));
        if (adjoint && signs) value *= signs[(start + j) % width];
        output[start + j] = T(value);
    }
}

template<class T> void run(int rows, int width, int block, bool adjoint, bool identity, bool in_place, bool bench = false) {
    const auto count = std::size_t(rows) * width;
    std::vector<T> input(count + 2), expected(count + 2), actual(count + 2);
    std::vector<std::int8_t> signs(width + 2);
    const auto guard = T(-30001.f);
    std::fill(input.begin(), input.end(), guard);
    for (std::size_t i = 0; i < count; ++i) input[i + 1] = T(std::sin(float(i) * .073f) * 17.f);
    for (int i = 0; i < width + 2; ++i) signs[i] = (i * 17 + i / 7) % 3 ? -1 : 1;
    Device<T> source(count + 2), old(count + 2), dest(count + 2);
    Device<std::int8_t> sign(width + 2);
    check(cudaMemcpy(source.data, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    check(cudaMemcpy(dest.data, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    check(cudaMemcpy(old.data, input.data(), input.size() * sizeof(T), cudaMemcpyHostToDevice));
    check(cudaMemcpy(sign.data, signs.data(), signs.size(), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    int active_device = 0, regular_shared = 0;
    check(cudaGetDevice(&active_device));
    check(cudaDeviceGetAttribute(&regular_shared, cudaDevAttrMaxSharedMemoryPerBlock, active_device));
    if (block * sizeof(float) > std::size_t(regular_shared))
        check(cudaFuncSetAttribute(reference<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, block * sizeof(float)));
    const auto* sign_ptr = identity ? nullptr : sign.data + 1;
    auto* result = in_place ? source.data : dest.data;
    const auto launch = [&](bool next) {
        if (next) {
            if constexpr (std::is_same_v<T, __half>) mfq::cuda::hadamard::apply_f16(source.data + 1, sign_ptr, result + 1, rows, width, block, adjoint, stream);
            else mfq::cuda::hadamard::apply_f32(source.data + 1, sign_ptr, result + 1, rows, width, block, adjoint, stream);
        }
        else reference<<<unsigned(count / block), 256, block * sizeof(float), stream>>>(
            source.data + 1, sign_ptr, old.data + 1, width, block, adjoint);
        check(cudaGetLastError());
    };
    launch(false); launch(true);
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(expected.data(), old.data, expected.size() * sizeof(T), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), result, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
    if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(T))) {
        for (std::size_t i = 0; i < actual.size(); ++i) if (std::memcmp(&expected[i], &actual[i], sizeof(T))) {
            std::fprintf(stderr, "Hadamard mismatch rows=%d K=%d block=%d adjoint=%d identity=%d in_place=%d i=%zu old=%g new=%g\n",
                rows, width, block, int(adjoint), int(identity), int(in_place), i,
                float(expected[i]), float(actual[i]));
            break;
        }
        throw std::runtime_error("Hadamard differs from original transform or overwrites guard");
    }
    if (bench) {
        cudaGraph_t graphs[2]{}; cudaGraphExec_t execs[2]{};
        for (int mode = 0; mode < 2; ++mode) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            launch(mode != 0);
            check(cudaStreamEndCapture(stream, &graphs[mode]));
            check(cudaGraphInstantiate(&execs[mode], graphs[mode], 0));
            std::size_t nodes = 0; check(cudaGraphGetNodes(graphs[mode], nullptr, &nodes));
            if (nodes != 1) throw std::runtime_error("Hadamard graph should contain one transform");
            cudaGraphNode_t node; check(cudaGraphGetNodes(graphs[mode], &node, &nodes));
            cudaKernelNodeParams params{}; check(cudaGraphKernelNodeGetParams(node, &params));
            cudaFuncAttributes attributes{}; check(cudaFuncGetAttributes(&attributes, params.func));
            std::printf("Hadamard actual_graph_kernel mode=%d threads=%u registers=%d local_bytes=%zu\n",
                mode, params.blockDim.x, attributes.numRegs, attributes.localSizeBytes);
            if (mode && block <= 128 && params.blockDim.x != 128)
                throw std::runtime_error("Hadamard warp implementation was not captured");
        }
        for (int i = 0; i < 512; ++i) for (auto exec : execs) check(cudaGraphLaunch(exec, stream));
        cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        std::vector<float> times[2];
        for (int trial = 0; trial < 7; ++trial) for (int order = 0; order < 2; ++order) {
            const int mode = (trial + order) % 2;
            check(cudaEventRecord(begin, stream));
            for (int i = 0; i < 512; ++i) check(cudaGraphLaunch(execs[mode], stream));
            check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
            float ms = 0; check(cudaEventElapsedTime(&ms, begin, end)); times[mode].push_back(ms * 1000 / 512);
        }
        for (auto& t : times) std::sort(t.begin(), t.end());
        std::printf("Hadamard rows=%d K=%d block=%d old_us=%.6f new_us=%.6f ratio=%.6f\n",
            rows, width, block, times[0][3], times[1][3], times[1][3] / times[0][3]);
        for (int mode = 0; mode < 2; ++mode) { cudaGraphExecDestroy(execs[mode]); cudaGraphDestroy(graphs[mode]); }
        cudaEventDestroy(begin); cudaEventDestroy(end);
    }
    cudaStreamDestroy(stream);
}
}

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) {
            std::puts("SKIP: CUDA device unavailable"); return 77;
        }
        if (argc == 5 && std::string(argv[1]) == "--bench-f32") {
            run<float>(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), false, false, false, true);
        } else if (argc == 5 && std::string(argv[1]) == "--bench") {
            run<__half>(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), false, false, false, true);
        } else {
            for (int block = 1; block <= 8192; block *= 2)
                for (int rows : {1, 2, 17}) for (int blocks : {1, 3})
                    for (bool adjoint : {false, true}) for (bool identity : {false, true})
                        for (bool in_place : {false, true})
                            {run<__half>(rows, blocks * block, block, adjoint, identity, in_place);
                             run<float>(rows, blocks * block, block, adjoint, identity, in_place);}
            run<__half>(65537, 2, 2, false, false, false);
            run<float>(65537, 1, 1, false, false, false);
            int device = 0, regular = 0, optin = 0;
            check(cudaGetDevice(&device));
            check(cudaDeviceGetAttribute(&regular, cudaDevAttrMaxSharedMemoryPerBlock, device));
            check(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            if (std::max(regular, optin) >= 16384 * int(sizeof(float))) {
                for (bool adjoint : {false, true}) {
                    run<__half>(2, 32768, 16384, adjoint, false, true);
                    run<float>(2, 32768, 16384, adjoint, false, true);
                }
            } else std::puts("SKIP: 16384-value Hadamard requires 65536 bytes shared memory");
            int oversized = 1;
            while (std::size_t(oversized) * sizeof(float) <= std::size_t(std::max(regular, optin))) oversized *= 2;
            Device<float> one(1);
            bool capacity_rejected = false;
            try { mfq::cuda::hadamard::apply_f32(one.data, nullptr, one.data, 1, oversized, oversized, false, nullptr); }
            catch (const std::invalid_argument& error) { capacity_rejected = std::string(error.what()).find("capacity") != std::string::npos; }
            if (!capacity_rejected) throw std::runtime_error("Hadamard accepted excess shared-memory requirement");
            mfq::cuda::hadamard::apply_f16(nullptr, nullptr, nullptr, 0, 128, 128, false, nullptr);
            mfq::cuda::hadamard::apply_f16(nullptr, nullptr, nullptr, 3, 0, 128, false, nullptr);
            for (int block : {0, -1, 3}) {
                bool rejected = false;
                try { mfq::cuda::hadamard::apply_f16(nullptr, nullptr, nullptr, 0, 128, block, false, nullptr); }
                catch (const std::invalid_argument&) { rejected = true; }
                if (!rejected) throw std::runtime_error("Invalid Hadamard block accepted");
            }
        }
        std::puts("Hadamard FP16/FP32 original-order/guards/in-place/identity/block1..8192/large-batch PASS");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
