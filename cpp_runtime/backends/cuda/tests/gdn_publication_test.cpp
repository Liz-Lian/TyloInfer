#include "mfq_cuda_context.h"
#include "mfq_cuda_linear_attention_ops.h"
#include "runtime/decode_window.h"
#include "runtime/graph_pool_lease.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace tb = mfq_tensor_backend;
using Tensor = tb::Tensor;
using mfq::cuda::DecodeWindow;

static Tensor data(std::vector<int64_t> shape, int salt, float scale) {
    auto host = tb::empty(shape, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for (int64_t i = 0; i < host.numel(); ++i)
        host.data_ptr<float>()[i] = scale * std::sin(float(i) * .0173f + float(salt) * .31f);
    return host.to(tb::kCUDA);
}

static void equal(const Tensor &actual, const Tensor &expected, const char *message) {
    const auto a = actual.contiguous().cpu(), b = expected.contiguous().cpu();
    if (a.sizes() != b.sizes() || a.scalar_type() != b.scalar_type() ||
        std::memcmp(a.data_ptr(), b.data_ptr(), a.nbytes()))
        throw std::runtime_error(message);
}

// Existing coverage includes contiguous state, prepared Q8 output, snapshots,
// all three column-tile counts, FP16/FP32 gates and output rounding.
#include "gdn_core_q8_test.h"

static void set_option(const char *name, const char *value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}

class SavedOptions {
    std::vector<std::pair<std::string, std::optional<std::string>>> values_;

  public:
    SavedOptions(std::initializer_list<const char *> names) {
        for (auto name : names) {
            const auto *value = std::getenv(name);
            values_.emplace_back(name, value ? std::optional<std::string>(value) : std::nullopt);
        }
    }
    ~SavedOptions() {
        for (const auto &[name, value] : values_)
            set_option(name.c_str(), value ? value->c_str() : nullptr);
    }
};

static void independent_history_case() {
    SavedOptions saved{"TYLOINFER_GDN_INPLACE_CONV", "TYLOINFER_GDN_DECODE_TILES",
                       "TYLOINFER_GDN_BROADCAST_NORMALIZATION"};
    int cases = 0, replays = 0;
    for (int width : {32, 64, 128})
        for (int batch : {2, 3})
            for (int group : {1, 3, 24, 25})
                for (int tiles : {1, 2, 4})
                    for (bool half : {false, true})
                        for (bool quantized : {false, true}) {
                            const int nk = 2, nv = nk * group, taps = width == 64 ? 7 : 4;
                            const int channels = (2 * nk + nv) * width, owners = batch + 1;
                            const bool transpose = width != 32, output_half = width == 64;
                            const auto dtype = half ? tb::kFloat16 : tb::kFloat32;
                            auto qkv = data({batch, 1, channels + 17}, 1301, .2f)
                                           .to(dtype)
                                           .narrow(2, 0, channels);
                            auto gate = data({batch, 1, nv * width + 13}, 1302, .3f)
                                            .to(dtype)
                                            .narrow(2, 0, nv * width);
                            auto alpha =
                                data({batch, 1, nv + 3}, 1303, .1f).to(dtype).narrow(2, 0, nv);
                            auto beta =
                                data({batch, 1, nv + 3}, 1304, .1f).to(dtype).narrow(2, 0, nv);
                            auto weight = data({channels, 1, taps}, 1305, .1f);
                            auto bias = data({nv}, 1306, .2f), a = data({nv}, 1307, .1f);
                            auto norm = data({width}, 1308, .4f);
                            std::vector<Tensor> conv[2], state[2], counters[2];
                            for (int owner = 0; owner < owners; ++owner) {
                                auto c = data({1, taps - 1, channels}, 1400 + owner, .1f);
                                auto s = data({1, nv, width, width}, 1500 + owner, .01f);
                                for (int mode = 0; mode < 2; ++mode) {
                                    conv[mode].push_back(c.clone());
                                    state[mode].push_back(s.clone());
                                    counters[mode].push_back(
                                        tb::zeros({1, nv, width + 1}, c.options()));
                                }
                            }
                            Tensor addresses[2], output[2], returned_history[2];
                            for (auto &address : addresses)
                                address = tb::empty({batch, 3}, weight.options().dtype(tb::kInt64));
                            const auto change_owners = [&](int step) {
                                for (int mode = 0; mode < 2; ++mode) {
                                    auto host = tb::empty(
                                        {batch, 3},
                                        weight.options().device(tb::kCPU).dtype(tb::kInt64));
                                    for (int row = 0; row < batch; ++row) {
                                        const int owner = (row + step) % owners;
                                        host.data_ptr<int64_t>()[3 * row] =
                                            reinterpret_cast<int64_t>(conv[mode][owner].data_ptr());
                                        host.data_ptr<int64_t>()[3 * row + 1] =
                                            reinterpret_cast<int64_t>(
                                                state[mode][owner].data_ptr());
                                        host.data_ptr<int64_t>()[3 * row + 2] =
                                            reinterpret_cast<int64_t>(
                                                counters[mode][owner].data_ptr());
                                    }
                                    addresses[mode].copy_(host);
                                }
                            };
                            change_owners(0);
                            set_option("TYLOINFER_GDN_DECODE_TILES", std::to_string(tiles).c_str());
                            const char *normalization = group == 1    ? "1"
                                                        : group == 24 ? "3"
                                                        : group == 25 ? "4"
                                                                      : "2";
                            set_option("TYLOINFER_GDN_BROADCAST_NORMALIZATION", normalization);
                            DecodeWindow graphs[2] = {DecodeWindow(mfq_current_cuda_stream()),
                                                      DecodeWindow(mfq_current_cuda_stream())};
                            for (int mode = 0; mode < 2; ++mode) {
                                set_option("TYLOINFER_GDN_INPLACE_CONV", mode ? "1" : "0");
                                graphs[mode].capture(
                                    [&, mode] {
                                        auto r = gdn_decode_core_cuda(
                                            qkv, gate, alpha, beta, conv[mode][0], state[mode][0],
                                            weight, bias, a, norm, nk, nv, width, 1e-6, 1e-6, half,
                                            output_half, transpose, counters[mode][0], true,
                                            addresses[mode], half, quantized);
                                        output[mode] = r[0];
                                        returned_history[mode] = r[1];
                                    },
                                    [&] {
                                        // Restore capture warmups to the reference owners. Owner
                                        // tables change only after capture and do not affect graph
                                        // addresses.
                                        for (int owner = 0; owner < owners; ++owner) {
                                            auto c =
                                                data({1, taps - 1, channels}, 1400 + owner, .1f);
                                            auto s =
                                                data({1, nv, width, width}, 1500 + owner, .01f);
                                            for (int m = 0; m < 2; ++m) {
                                                conv[m][owner].copy_(c);
                                                state[m][owner].copy_(s);
                                                counters[m][owner].zero_();
                                            }
                                        }
                                    },
                                    [&, mode] {
                                        output[mode] = {};
                                        returned_history[mode] = {};
                                    });
                            }
                            for (int step = 0; step < 3; ++step) {
                                change_owners(step);
                                qkv.copy_(data(qkv.sizes().vec(), 1600 + step, .1f * (step + 1))
                                              .to(dtype));
                                const int repeat = width == 128 && batch == 3 && group == 3 &&
                                                           tiles == 4 && half && quantized
                                                       ? 128
                                                       : 3;
                                for (int n = 0; n < repeat; ++n) {
                                    graphs[0].run();
                                    graphs[1].run();
                                    ++replays;
                                }
                                equal(output[1], output[0], "Independent GDN output changed");
                                equal(returned_history[1], returned_history[0],
                                      "Independent GDN returned convolution history changed");
                                for (int owner = 0; owner < owners; ++owner) {
                                    equal(conv[1][owner], conv[0][owner],
                                          "Independent GDN owner convolution history changed");
                                    equal(state[1][owner], state[0][owner],
                                          "Independent GDN owner recurrent state changed");
                                    auto zero = tb::zeros({1, nv}, weight.options());
                                    equal(counters[1][owner].select(2, width), zero,
                                          "Independent GDN publication counter did not reset");
                                    equal(counters[0][owner].select(2, width), zero,
                                          "Reference GDN publication counter did not reset");
                                }
                            }
                            ++cases;
                        }
    std::cout << "GDN indirect history publication cases=" << cases
              << " changed-graph replays=" << replays << " EXACT PASS\n";
}

int main() try {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count)
        return 77;
    const auto context = mfq::cuda::default_context(mfq_current_cuda_device());
    const auto pool = mfq::cuda::GraphPoolLease::acquire(context, mfq_current_cuda_stream());
    tb::NoGradGuard no_grad;
    gdn_core_q8_case();
    independent_history_case();
    return 0;
} catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
}
