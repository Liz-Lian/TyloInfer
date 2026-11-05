#include "kernels/dflash2.h"
#include "mfq_cuda_context.h"
#include "models/dflash2/weight_quant.h"
#include "runtime/decode_window.h"
#include "runtime/graph_pool_lease.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string_view>
#include <vector>

namespace tb = mfq_tensor_backend;
namespace df = mfq::cuda::dflash2;
namespace {
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
tb::Tensor values(int rows, int columns, float phase = 0.f) {
    auto result =
        tb::empty({rows, columns}, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for (int64_t i = 0; i < result.numel(); ++i)
        result.data_ptr<float>()[i] =
            i / columns % 7 == 0 ? 0.f : .3f * std::sin(float(i) * .017f + phase);
    return result;
}
void check_cpu() {
    const int threads = mfq_get_num_threads();
    int cases = 0;
    for (int columns : {32, 96, 512, 1536})
        for (int rows : {1, 7, 33})
            for (auto dtype : {tb::kFloat32, tb::kFloat16, tb::kBFloat16}) {
                auto input = values(rows, columns).to(dtype);
                auto reference = input.to(tb::kFloat32);
                for (auto format : {GGML_TYPE_Q4_0, GGML_TYPE_Q8_0}) {
                    auto expected =
                        tb::empty({rows * int64_t(mfq::gguf::row_bytes(format, columns))},
                                  input.options().dtype(tb::kUInt8));
                    if (format == GGML_TYPE_Q4_0)
                        quantize_row_q4_0_ref(reference.data_ptr<float>(),
                                              static_cast<block_q4_0 *>(expected.data_ptr()),
                                              int64_t(rows) * columns);
                    else
                        quantize_row_q8_0_ref(reference.data_ptr<float>(),
                                              static_cast<block_q8_0 *>(expected.data_ptr()),
                                              int64_t(rows) * columns);
                    for (int count : {1, threads}) {
                        mfq_set_num_threads(count);
                        auto actual = format == GGML_TYPE_Q4_0 ? df::pack_q4_weight_cpu(input)
                                                               : df::pack_q8_weight_cpu(input);
                        require(actual.nbytes() == expected.nbytes() &&
                                    !std::memcmp(actual.data_ptr(), expected.data_ptr(),
                                                 expected.nbytes()),
                                "Draft weight packing differs from GGML reference");
                        ++cases;
                    }
                }
            }
    mfq_set_num_threads(threads);
    std::cout << "Q4/Q8 CPU F32/F16/BF16 reference and thread overrides: " << cases << " PASS\n";
}
void check_gpu() {
    const auto context = mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream = mfq_current_cuda_stream();
    const auto lease = mfq::cuda::GraphPoolLease::acquire(context, stream);
    int cases = 0, replays = 0;
    // Non-tile dimensions exercise padded columns, rows and segment boundaries.
    for (auto [columns, first_rows] : {std::pair{32, 1}, std::pair{544, 33}, std::pair{1536, 65}}) {
        const int pk = (columns + 511) / 512 * 512;
        std::vector<tb::Tensor> weights[2];
        std::vector<int> rows;
        for (int segment = 0; segment < 3; ++segment) {
            const int n = first_rows + segment * 17, pn = (n + 31) / 32 * 32;
            auto packed = df::pack_q4_weight_cpu(values(n, columns, float(segment)));
            auto expanded =
                tb::empty({int64_t(n) * (columns / 32) * sizeof(block_q8_0)}, packed.options());
            const auto *q4 = static_cast<const block_q4_0 *>(packed.data_ptr());
            auto *q8 = static_cast<block_q8_0 *>(expanded.data_ptr());
            // An exact Q4 expansion shares the original FP16 scales and codes.
            // The existing Q8 GPU dot is an independent numerical reference.
            for (int64_t b = 0; b < int64_t(n) * (columns / 32); ++b) {
                q8[b].d = q4[b].d;
                for (int j = 0; j < 16; ++j) {
                    q8[b].qs[j] = int(q4[b].qs[j] & 15) - 8;
                    q8[b].qs[j + 16] = int(q4[b].qs[j] >> 4) - 8;
                }
            }
            auto g4 = packed.to(tb::kCUDA), g8 = expanded.to(tb::kCUDA);
            auto t4 = tb::empty({int64_t(pn) * (pk / 32) * sizeof(block_q4_0)}, g4.options());
            auto t8 = tb::empty({int64_t(pn) * (pk / 32) * sizeof(block_q8_0)}, g8.options());
            df::q4_tile(g4.data_ptr(), t4.data_ptr(), n, columns, stream);
            df::q8_tile(g8.data_ptr(), t8.data_ptr(), n, columns, stream);
            auto restored = tb::empty_like(g4);
            df::q4_untile(t4.data_ptr(), restored.data_ptr(), n, columns, stream);
            const auto host = restored.cpu();
            require(!std::memcmp(host.data_ptr(), packed.data_ptr(), packed.nbytes()),
                    "Q4 tile roundtrip changed a code, scale or padded tail");
            weights[0].push_back(t8);
            weights[1].push_back(t4);
            rows.push_back(n);
        }
        for (int count : {1, 2, 3})
            for (int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 32}) {
                auto input = values(tokens, columns, .7f).to(tb::kCUDA);
                const int padded = (tokens + 7) / 8 * 8;
                auto codes = tb::empty({int64_t(padded) * pk}, input.options().dtype(tb::kUInt8));
                auto scales = tb::empty({int64_t(padded) * (pk / 32)}, input.options());
                std::vector<tb::Tensor> outputs[2];
                std::vector<df::Q8TiledMatrix> descriptors[2];
                for (int mode = 0; mode < 2; ++mode)
                    for (int i = 0; i < count; ++i) {
                        outputs[mode].push_back(
                            tb::full({int64_t(tokens) * rows[i] + 2}, -12345.f, input.options()));
                        descriptors[mode].push_back({weights[mode][i].data_ptr(),
                                                     outputs[mode].back().data_ptr<float>() + 1,
                                                     rows[i], (rows[i] + 31) / 32 * 32});
                    }
                const auto run = [&] {
                    df::q8_quantize(input.data_ptr<float>(), codes.data_ptr(),
                                    scales.data_ptr<float>(), tokens, columns, stream);
                    df::q8_gemm(descriptors[0].data(), count, codes.data_ptr(),
                                scales.data_ptr<float>(), tokens, columns, stream);
                    df::q4_gemm(descriptors[1].data(), count, codes.data_ptr(),
                                scales.data_ptr<float>(), tokens, columns, stream);
                };
                const auto compare = [&] {
                    for (int i = 0; i < count; ++i) {
                        const auto ref = outputs[0][i].cpu(), actual = outputs[1][i].cpu();
                        require(!std::memcmp(ref.data_ptr(), actual.data_ptr(), ref.nbytes()),
                                "Q4 projection differs from exact expanded Q8 reference");
                        const auto *y = actual.data_ptr<float>();
                        require(y[0] == -12345.f && y[actual.numel() - 1] == -12345.f,
                                "Q4 projection overwrote an output guard");
                        for (int64_t j = 1; j + 1 < actual.numel(); ++j)
                            require(y[j] != -12345.f, "Q4 projection left an output unwritten");
                    }
                };
                run();
                compare();
                ++cases;
                if (count == 3 && (tokens == 3 || tokens == 7 || tokens == 17)) {
                    mfq::cuda::DecodeWindow graph(stream);
                    graph.capture(run, [] {});
                    for (float phase : {1.1f, -2.3f, 0.f}) {
                        input.copy_(values(tokens, columns, phase));
                        graph.run();
                        compare();
                        ++replays;
                    }
                }
            }
    }
    std::cout << "Q4 tails, heterogeneous segments, output guards, M=1..9/17/32: " << cases
              << ", changing graph replays: " << replays << " PASS\n";
}
} // namespace
int main(int argc, char **argv) try {
    tb::NoGradGuard guard;
    check_cpu();
    if (argc == 2 && std::string_view(argv[1]) == "--cpu")
        return 0;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: CUDA device unavailable\n";
        return 77;
    }
    check_gpu();
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
