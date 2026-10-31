#include "../kernels/gguf_compact.h"
#include "../kernels/gguf.h"
#include "ggml.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace mfq::cuda::gguf;
const std::uint16_t book[4096] = {
#include "../../../third_party/agention-gyro/codebook_f16.inc"
};
struct alignas(4) Q8 { __half d, s; std::int8_t qs[32]; };
void check(cudaError_t status) { if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
template<class T> struct Device {
    T* data = nullptr;
    explicit Device(std::size_t n) { check(cudaMalloc(&data, n * sizeof(T))); }
    ~Device() { cudaFree(data); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
float half_at(const void* at) { __half value; std::memcpy(&value, at, 2); return __half2float(value); }
std::vector<float> reference(const std::vector<std::uint8_t>& weights, std::uint64_t stride,
        int format, int rows, int columns) {
    const std::array<int, 9> bits{0, 8, 6, 7, 5, 6, 7, 5, 4}, steps{0, 32, 32, 32, 64, 64, 64, 32, 32};
    const int b = bits.at(format - 143), s = steps.at(format - 143), bytes = b ? 2 + b * s / 8 : 28;
    std::vector<float> decoded(std::size_t(rows) * columns);
    for (int row = 0; row < rows; ++row) for (int block = 0; block < columns / 128; ++block) {
        const auto* q = weights.data() + 1 + row * stride + block * bytes;
        const float scale = half_at(q + (b ? 0 : 26));
        auto* output = decoded.data() + row * columns + block * 128;
        if (!b) {
            const unsigned powers[] = {1, 3, 9, 27, 81};
            int at = 0;
            for (int tile = 0; tile < 3; ++tile) {
                const int first = tile == 0 ? 0 : tile == 1 ? 16 : 24;
                const int width = tile == 0 ? 16 : tile == 1 ? 8 : 2;
                for (int digit = 0; digit < (tile == 2 ? 4 : 5); ++digit)
                    for (int byte = 0; byte < width; ++byte) {
                        const auto shifted = std::uint8_t(q[first + byte] * powers[digit]);
                        output[at++] = scale * (int(unsigned(shifted) * 3 >> 8) - 1);
                    }
            }
        } else {
            q += 2;
            for (int step = 0; step < s; ++step) {
                std::uint32_t state = 0;
                if (b == 8 && s == 32) state = (std::uint32_t(q[(step + 31) % 32]) << 8) | q[step];
                else for (int j = 0; j < 15; ++j) {
                    const int at = ((s - 1 - step) * b + j) % (s * b);
                    state |= std::uint32_t((q[at / 8] >> (at % 8)) & 1) << j;
                }
                const auto hash = (state & 32767u) * 0x9e3779b1u;
                for (int j = 0; j < 128 / s; ++j) {
                    const auto index = j < 2 ? hash >> 21 : (hash >> 10) & 2047u;
                    output[step * (128 / s) + j] = half_at(book + 2 * index + j % 2) * scale;
                }
            }
        }
    }
    return decoded;
}
template<class T> constexpr CompactInput input_type = std::is_same_v<T, Q8> ? CompactInput::Q8_1 :
    std::is_same_v<T, __half> ? CompactInput::Float16 : CompactInput::Float32;
template<class T> float input_at(const std::vector<T>& x, int at) {
    if constexpr (std::is_same_v<T, Q8>) return __half2float(x[1 + at / 32].d) * x[1 + at / 32].qs[at % 32];
    else return float(x[1 + at]);
}

template<class T>
void run(int format, int tokens, int rows, int columns, bool misaligned, bool benchmark, bool native_pair = false) {
    const auto type = CompactFormat(format);
    const auto row_bytes = compact_row_bytes(type, columns), stride = row_bytes + (misaligned ? 3 : 0);
    const auto blocks = columns / 128;
    const auto bytes = row_bytes / blocks;
    std::vector<std::uint8_t> weights(rows * stride + 2);
    for (std::size_t i = 0; i < weights.size(); ++i) weights[i] = std::uint8_t((i * 29 + i / 17 + 13) & 255);
    for (int row = 0; row < rows; ++row) for (int block = 0; block < blocks; ++block) {
        const auto index = row * blocks + block;
        const auto scale = __float2half(index % 7 ? std::ldexp(index % 2 ? -1.f : 1.f, -int(5 + index % 3)) : 0.f);
        std::memcpy(weights.data() + 1 + row * stride + block * bytes + (format == 143 ? 26 : 0), &scale, 2);
    }
    const auto values = std::size_t(tokens) * columns;
    std::vector<T> input((std::is_same_v<T, Q8> ? values / 32 : values) + 2);
    for (std::size_t i = 1; i + 1 < input.size(); ++i) {
        if constexpr (std::is_same_v<T, Q8>) {
            input[i].d = __float2half(std::ldexp(1.f, -int(7 + (i - 1) % 3))); input[i].s = __float2half(0.f);
            for (int j = 0; j < 32; ++j) input[i].qs[j] = std::int8_t(((i - 1) * 19 + j * 31) % 255 - 127);
        } else {
            const auto at = i - 1;
            input[i] = T(float(int(((at / 32) * 19 + (at % 32) * 31) % 255) - 127) * std::ldexp(1.f, -int(7 + (at / 32) % 3)));
        }
    }
    const auto decoded = reference(weights, stride, format, rows, columns);
    const auto* traits = ggml_get_type_traits(ggml_type(format));
    if (!traits->to_float || traits->blck_size != 128 || traits->type_size != bytes)
        throw std::runtime_error("Compact GGML traits disagree with file layout");
    std::vector<float> shared_decoded(decoded.size());
    for (int row = 0; row < rows; ++row)
        traits->to_float(weights.data() + 1 + row * stride, shared_decoded.data() + row * columns, columns);
    if (std::memcmp(shared_decoded.data(), decoded.data(), decoded.size() * sizeof(float)))
        throw std::runtime_error("Compact test oracle differs from upstream Metal/GGML decoder");
    Device<std::uint8_t> w(weights.size()); Device<T> x(input.size() + 4);
    const auto* input_ptr = x.data + (misaligned ? 1 : 4);
    Device<float> expanded(decoded.size() + 2), out(std::size_t(tokens) * rows + 2);
    Device<__half> half_expanded(decoded.size() + 2), half_out(std::size_t(tokens) * rows + 2);
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    check(cudaMemcpyAsync(w.data, weights.data(), weights.size(), cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(const_cast<T*>(input_ptr), input.data() + 1, (input.size() - 2) * sizeof(T), cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(out.data, 0x5a, (std::size_t(tokens) * rows + 2) * sizeof(float), stream));
    check(cudaMemsetAsync(half_out.data, 0x5a, (std::size_t(tokens) * rows + 2) * sizeof(__half), stream));
    compact_dequantize(type, w.data + 1, stride, rows, columns, expanded.data + 1, false, stream);
    compact_dequantize(type, w.data + 1, stride, rows, columns, half_expanded.data + 1, true, stream);
    std::vector<float> actual_decoded(decoded.size()); std::vector<__half> actual_half(decoded.size());
    check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(actual_decoded.data(), expanded.data + 1, decoded.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual_half.data(), half_expanded.data + 1, decoded.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    if (std::memcmp(decoded.data(), actual_decoded.data(), decoded.size() * sizeof(float)))
        throw std::runtime_error("Compact GPU dequantization differs from independent scalar decoder");
    for (std::size_t i = 0; i < decoded.size(); ++i) {
        const auto h = __float2half(decoded[i]);
        if (std::memcmp(&h, &actual_half[i], 2)) throw std::runtime_error("Compact FP16 dequantization differs");
    }
    if constexpr(std::is_same_v<T,float>)if(tokens==1 && !benchmark) {
        // This PP entry point has a separate vectorized decoder. Check the
        // scalar/GGML oracle, odd compressed strides and joined GU batch holes.
        const auto matrix=std::int64_t(rows)*columns,group_stride=matrix+16;
        Device<GemmJob> jobs(2);Device<__half> grouped(std::size_t(2*group_stride+16));
        const std::array<GemmJob,2> descriptors{{{w.data+1,nullptr,0},{w.data+1,nullptr,0}}};
        check(cudaMemcpyAsync(jobs.data,descriptors.data(),sizeof(descriptors),cudaMemcpyHostToDevice,stream));
        check(cudaMemsetAsync(grouped.data,0x7f,std::size_t(2*group_stride+16)*sizeof(__half),stream));
        grouped_decode_f16(ggml_type(format),jobs.data,2,stride,rows,columns,grouped.data+8,stream,group_stride);
        check(cudaStreamSynchronize(stream));
        std::vector<std::uint16_t> result(std::size_t(2*group_stride+16));
        check(cudaMemcpy(result.data(),grouped.data,result.size()*sizeof(std::uint16_t),cudaMemcpyDeviceToHost));
        for(std::size_t i=0;i<result.size();++i) {
            const auto at=std::int64_t(i)-8;
            const bool live=at>=0 && at<2*group_stride && at%group_stride<matrix;
            std::uint16_t expected=0x7f7f;
            if(live){const auto h=__float2half(decoded[std::size_t(at%group_stride)]);std::memcpy(&expected,&h,2);}
            if(result[i]!=expected)throw std::runtime_error("Compact grouped FP16 decode changed a scalar value or batch gap");
        }
    }
    const auto launch = [&](bool reuse, bool half_output = false) {
        if (reuse) compact_project(type, w.data + 1, stride, rows, columns, input_ptr, input_type<T>, tokens,
            half_output ? static_cast<void*>(half_out.data + 1) : static_cast<void*>(out.data + 1), half_output, stream);
        else for (int m = 0; m < tokens; ++m) compact_project(type, w.data + 1, stride, rows, columns,
            input_ptr + m * (std::is_same_v<T, Q8> ? columns / 32 : columns), input_type<T>, 1,
            out.data + 1 + m * rows, false, stream);
    };
    std::vector<float> scalar(std::size_t(tokens) * rows + 2), actual(scalar.size());
    launch(false); check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(scalar.data(), out.data, scalar.size() * sizeof(float), cudaMemcpyDeviceToHost));
    launch(true); launch(true, true); check(cudaStreamSynchronize(stream));
    check(cudaMemcpy(actual.data(), out.data, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    if (std::memcmp(scalar.data(), actual.data(), actual.size() * sizeof(float)))
        throw std::runtime_error("Compact multi-token reuse differs from separate M1 calls");
    std::vector<__half> half_result(actual.size());
    check(cudaMemcpy(half_result.data(), half_out.data, half_result.size() * sizeof(__half), cudaMemcpyDeviceToHost));
    if (std::uint32_t guard; (std::memcpy(&guard, actual.data(), 4), guard != 0x5a5a5a5a) ||
        (std::memcpy(&guard, &actual.back(), 4), guard != 0x5a5a5a5a)) throw std::runtime_error("Compact output guard overwritten");
    const auto launch_native = [&] {
        matvec_q8(ggml_type(format), w.data + 1, stride, rows, columns, input_ptr,
            tokens, out.data + 1, stream);
    };
    std::vector<float> native;
    if constexpr (std::is_same_v<T, Q8>) {
        launch_native(); check(cudaStreamSynchronize(stream)); native.resize(actual.size());
        check(cudaMemcpy(native.data(), out.data, native.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (std::memcmp(native.data(), actual.data(), 4) || std::memcmp(&native.back(), &actual.back(), 4))
            throw std::runtime_error("Native compact projection output guard overwritten");
    }
    for (int m = 0; m < tokens; ++m) for (int row = 0; row < rows; ++row) {
        double expected = 0, absolute = 0;
        for (int k = 0; k < columns; ++k) {
            const double value = double(decoded[row * columns + k]) * input_at(input, m * columns + k);
            expected += value; absolute += std::abs(value);
        }
        const auto i = 1 + m * rows + row;
        if (!std::isfinite(actual[i]) || std::abs(double(actual[i]) - expected) > 2e-6 * absolute + 1e-6)
            throw std::runtime_error("Compact dot differs from scalar double reference");
        if (!native.empty() && (!std::isfinite(native[i]) || std::abs(double(native[i]) - expected) > 2e-6 * absolute + 1e-6))
            throw std::runtime_error("Native prepared compact dot differs from scalar double reference");
        const auto h = __float2half(actual[i]);
        if (std::memcmp(&h, &half_result[i], 2)) throw std::runtime_error("Compact FP16 projection differs from FP32 rounding");
    }
    if (benchmark) {
        cudaGraph_t graphs[2]{}; cudaGraphExec_t execs[2]{};
        for (int mode = 0; mode < 2; ++mode) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            if (!mode && native_pair) launch_native(); else launch(mode != 0);
            check(cudaStreamEndCapture(stream, &graphs[mode])); check(cudaGraphInstantiate(&execs[mode], graphs[mode], 0));
        }
        if (native_pair) {
            std::size_t count[2]{};
            for (int mode = 0; mode < 2; ++mode) check(cudaGraphGetNodes(graphs[mode], nullptr, &count[mode]));
            if (count[0] == 1 && count[1] == 1) {
                cudaGraphNode_t nodes[2]{}; cudaKernelNodeParams params[2]{};
                for (int mode = 0; mode < 2; ++mode) {
                    check(cudaGraphGetNodes(graphs[mode], &nodes[mode], &count[mode]));
                    check(cudaGraphKernelNodeGetParams(nodes[mode], &params[mode]));
                }
                if (params[0].func == params[1].func && params[0].gridDim.x == params[1].gridDim.x &&
                        params[0].gridDim.y == params[1].gridDim.y && params[0].blockDim.x == params[1].blockDim.x)
                    throw std::runtime_error("Native wrapper already selects the compact kernel; identical benchmark paths refused");
            }
        }
        for (int i = 0; i < 512; ++i) for (auto g : execs) check(cudaGraphLaunch(g, stream));
        cudaEvent_t begin, end; check(cudaEventCreate(&begin)); check(cudaEventCreate(&end)); std::vector<float> times[2];
        for (int trial = 0; trial < 7; ++trial) for (int order = 0; order < 2; ++order) {
            const int mode = (trial + order) % 2; check(cudaEventRecord(begin, stream));
            for (int i = 0; i < 512; ++i) check(cudaGraphLaunch(execs[mode], stream));
            check(cudaEventRecord(end, stream)); check(cudaEventSynchronize(end));
            float ms = 0; check(cudaEventElapsedTime(&ms, begin, end)); times[mode].push_back(ms * 1000 / 512);
        }
        for (auto& t : times) std::sort(t.begin(), t.end());
        std::printf("compact format=%d input=%d M=%d N=%d K=%d baseline=%s baseline_us=%.6f reuse_us=%.6f ratio=%.6f\n",
            format, int(input_type<T>), tokens, rows, columns, native_pair ? "native_batch" : "separate_M1",
            times[0][3], times[1][3], times[1][3] / times[0][3]);
        for (int mode = 0; mode < 2; ++mode) { cudaGraphExecDestroy(execs[mode]); cudaGraphDestroy(graphs[mode]); }
        cudaEventDestroy(begin); cudaEventDestroy(end);
    }
    cudaStreamDestroy(stream);
}
}

int main(int argc, char** argv) {
    try {
        int devices = 0;
        if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) { std::puts("SKIP: CUDA unavailable"); return 77; }
        if (argc == 6 || argc == 7) {
            const int f = std::stoi(argv[1]), m = std::stoi(argv[2]), n = std::stoi(argv[3]), k = std::stoi(argv[4]), kind = std::stoi(argv[5]);
            const bool native_pair = argc == 7;
            if (native_pair && (std::string(argv[6]) != "native" || kind != 2))
                throw std::invalid_argument("Native pair requires Q8 activation input");
            if (kind == 0) run<__half>(f, m, n, k, false, true);
            else if (kind == 1) run<float>(f, m, n, k, false, true);
            else run<Q8>(f, m, n, k, false, true, native_pair);
        }
        else {
            for (int format = 143; format <= 151; ++format) {
                for (int m : {1, 2, 3, 4, 5}) for (int rows : {1, 7}) for (int width : {128, 640}) {
                    run<__half>(format, m, rows, width, true, false);
                    run<float>(format, m, rows, width, true, false);
                    run<Q8>(format, m, rows, width, true, false);
                }
                run<__half>(format, 4, 129, 2560, false, false);
                run<float>(format, 17, 7, 128, false, false);
                run<Q8>(format, 32, 7, 128, false, false);
            }
        }
        if(argc!=6 && argc!=7)std::puts("compact9 strided grouped FP16 decode/scalar oracle/unaligned packed bytes/batch gaps PASS");
        std::puts("compact9 original bytes/FP16-FP32-Q8/M1..32/decode reuse/reference/guards PASS");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
