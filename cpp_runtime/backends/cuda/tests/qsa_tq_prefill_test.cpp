
#include "kernels/qsa_tq.h"
#include "mfq_cuda_context.h"
#include "mfq_tensor_backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
namespace tb = mfq_tensor_backend;
namespace tq = tyloinfer::qsa;
using Tensor = tb::Tensor;
void ck(int e) {
    if (e)
        throw std::runtime_error(cudaGetErrorString(cudaError_t(e)));
}
Tensor upload(const void *p, size_t bytes) {
    auto h = tb::empty({int64_t(bytes)}, tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
    std::memcpy(h.data_ptr(), p, bytes);
    return h.to(tb::kCUDA);
}
Tensor random(std::vector<int64_t> shape, unsigned seed) {
    auto h = tb::empty(shape, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for (int64_t i = 0; i < h.numel(); ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        h.data_ptr<float>()[i] = (int(seed & 65535) - 32768) * (1.f / 262144);
    }
    return h.to(tb::kCUDA);
}
void exact(const Tensor &a, const Tensor &b, const char *label) {
    auto x = a.cpu().contiguous(), y = b.cpu().contiguous();
    if (x.nbytes() != y.nbytes() || std::memcmp(x.data_ptr(), y.data_ptr(), x.nbytes()))
        throw std::runtime_error(std::string("Staged attention changed ") + label);
}

struct Fixture {
    int queries, heads, kh, width, padded, blocks, count, parts;
    Tensor payload, views, selected, pos, query, signs, relative, decoded;
    Tensor output, rotated, partial, stats, logmass, norm, hits, mass, gate, requested;
    Fixture(int nq, int h, int k, int w, int nb, int ns, int format, bool half)
        : queries(nq), heads(h), kh(k), width(w), padded(w <= 256   ? 256
                                                         : w <= 512 ? 512
                                                                    : 1024),
          blocks(nb), count(ns), parts(std::min(4, ns)) {
        int rows = kh * 4 * (padded / 256); // Leave room for any codec, including a variable-width row map.
        const int bytes = rows * 512;
        std::vector<uint8_t> data(size_t(2) * blocks * bytes);
        unsigned seed = 9173;
        for (auto &b : data) {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            b = uint8_t(seed);
        }
        for (int b = 0; b < 2 * blocks; ++b) {
            auto *p = data.data() + b * bytes;
            if (!format) {
                for (int i = 0; i < rows * 256; ++i)
                    reinterpret_cast<__half *>(p)[i] = __float2half_rn(.125f * std::sin(float(i + b)));
            } else {
                for (int i = 0; i < rows; ++i)
                    reinterpret_cast<__half *>(p)[i] =
                        __float2half_rn(format == 254 ? .0037f + float(i) * 1.e-6f : .25f);
                if (format != 254) {
                    std::memset(p + 2 * rows, 0, (rows + 1) / 2);
                    if (format == 255) {
                        auto *map = p + 2 * rows + (rows + 1) / 2;
                        std::memset(map, 0, (3 * rows + 7) / 8);
                        for (int r = 0; r < rows; ++r) {
                            const unsigned code = r % 8;
                            const int bit = 3 * r;
                            map[bit / 8] |= uint8_t(code << (bit % 8));
                            if (bit % 8 > 5)
                                map[bit / 8 + 1] |= uint8_t(code >> (8 - bit % 8));
                        }
                    }
                }
            }
        }
        payload = upload(data.data(), data.size());
        std::vector<tq::TqBlockView> table(blocks);
        for (int b = 0; b < blocks; ++b) {
            table[b] = {
                payload.data_ptr<uint8_t>() + b * bytes, payload.data_ptr<uint8_t>() + (blocks + b) * bytes,
                tq::tq_precision_flags(uint8_t(format), 14), tq::tq_precision_flags(uint8_t(format), 14)};
        } // A cold, unselected descriptor must never be dereferenced by staging.
        table.back().key = table.back().value = reinterpret_cast<const uint8_t *>(uintptr_t(1));
        views = upload(table.data(), table.size() * sizeof(table[0]));
        std::vector<tq::SelectedBlock> selection(size_t(queries) * count);
        std::vector<int64_t> positions(queries);
        for (int q = 0; q < queries; ++q) {
            positions[q] = std::min(blocks * 4 - 1, (q + 2) * blocks * 4 / queries);
            for (int i = 0; i < count; ++i)
                selection[q * count + i] = {i % 23 == 0 ? -1 : (i * 13 + q * 7) % (blocks - 1),
                                            i % 11 == 0 ? 5u : 15u};
        }
        selected = upload(selection.data(), selection.size() * sizeof(selection[0]));
        pos = tb::tensor(positions, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64));
        query = random({queries, heads, width}, 97);
        if (half)
            query = query.to(tb::kFloat16);
        gate = random({queries, heads, width}, 139);
        const auto f = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        signs = tb::ones({padded}, f);
        relative = tb::ones({16}, f.dtype(tb::kFloat16));
        decoded = tb::empty({2LL * blocks * kh * 4 * padded}, f.dtype(tb::kFloat16));
        output = tb::empty({queries, heads, width}, f);
        rotated = tb::empty({queries * heads * padded}, f);
        partial = tb::empty({queries * heads * parts * padded}, f);
        stats = tb::empty({queries * heads * parts * 2}, f);
        logmass = tb::empty({queries * heads * count}, f);
        norm = tb::empty({queries * heads}, f);
        hits = tb::zeros({blocks}, f);
        mass = tb::zeros({blocks}, f);
        requested = tb::empty({blocks}, f.dtype(tb::kInt32));
    }
    void run(bool staged) {
        auto *first = static_cast<uint16_t *>(decoded.data_ptr());
        auto *second = first + decoded.numel() / 2;
        if (staged)
            ck(tq::tq_stage_attention_blocks(views.data_ptr<tq::TqBlockView>(), relative.data_ptr<uint16_t>(),
                                             first, second, blocks, kh, padded, mfq_current_cuda_stream(),
                                             selected.data_ptr<tq::SelectedBlock>(), queries, count,
                                             requested.data_ptr<uint32_t>()));
        ck(tq::tq_sparse_attention_staged(
            staged ? first : nullptr, staged ? second : nullptr, query.data_ptr(), signs.data_ptr<float>(),
            signs.data_ptr<float>(), relative.data_ptr<uint16_t>(), views.data_ptr<tq::TqBlockView>(),
            selected.data_ptr<tq::SelectedBlock>(), pos.data_ptr<int64_t>(), output.data_ptr(),
            rotated.data_ptr<float>(), partial.data_ptr<float>(), stats.data_ptr<float>(),
            logmass.data_ptr<float>(), norm.data_ptr<float>(), hits.data_ptr<float>(), mass.data_ptr<float>(),
            queries, heads, kh, width, padded, count, parts, heads * width, width, mfq_current_cuda_stream(),
            false, gate.data_ptr(), false, false, heads * width, width, query.scalar_type() == tb::kFloat16,
            true));
    }
    void validate() {
        hits.zero_();
        mass.zero_();
        run(false);
        auto a = output.clone(), b = partial.clone(), c = stats.clone(), d = logmass.clone(),
             old_hits = hits.clone(), old_mass = mass.clone();
        hits.zero_();
        mass.zero_();
        run(true);
        exact(a, output, "output");
        exact(b, partial, "partial");
        exact(c, stats, "stats");
        exact(d, logmass, "logmass");
        exact(old_hits, hits, "history hits");
        auto expected = old_mass.cpu(), actual = mass.cpu();
        for (int i = 0; i < blocks; ++i)
            if (std::abs(expected.data_ptr<float>()[i] - actual.data_ptr<float>()[i]) > 1.e-5f)
                throw std::runtime_error("Staged attention changed history mass");
    }
};
int main() try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices)
        return 77;
    auto stream = mfq::cuda::stream_from_pool(false);
    mfq::cuda::StreamGuard guard(stream);
    tb::NoGradGuard nograd;
    int cases = 0;
    for (int w : {17, 257, 1024})
        for (int ratio : {4, 5, 8, 12})
            for (int format : {0, 254, 1, 2, 3, 4, 5, 6, 7, 8, 255})
                for (bool half : {false, true}) {
                    Fixture f(9, 2 * ratio, 2, w, 7, 11, format, half);
                    try {
                        f.validate();
                    } catch (const std::exception &e) {
                        std::cerr << "width=" << w << " ratio=" << ratio << " format=" << format
                                  << " half=" << half << "\n";
                        throw;
                    }
                    ++cases;
                }
    // Capture the actual preparation and attention calls, then change Q and
    // replay. Allocations remain outside capture; requested IDs are regenerated.
    {
        Fixture f(9, 10, 2, 257, 7, 11, 254, false);
        f.run(true);
        ck(cudaStreamSynchronize(mfq_current_cuda_stream()));
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        ck(cudaStreamBeginCapture(mfq_current_cuda_stream(), cudaStreamCaptureModeThreadLocal));
        f.run(true);
        ck(cudaStreamEndCapture(mfq_current_cuda_stream(), &graph));
        ck(cudaGraphInstantiate(&executable, graph, 0));
        auto original_selection = f.selected.clone();
        for (int i = 0; i < 3; ++i) {
            if (i == 1)
                f.selected.zero_();
            else
                f.selected.copy_(original_selection);
            f.query.fill_(.0625 * (i + 1));
            ck(cudaGraphLaunch(executable, mfq_current_cuda_stream()));
            auto expected = f.output.clone();
            f.run(false);
            exact(expected, f.output, "graph_output");
        }
        ck(cudaGraphExecDestroy(executable));
        ck(cudaGraphDestroy(graph));
    }
    std::cout << nlohmann::json({{"exact_cases", cases}, {"changing_graph_replays", 3}}).dump() << std::endl;
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
}
