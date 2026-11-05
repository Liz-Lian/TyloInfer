#pragma once
#include "../kernels/gguf.h"
#include "../kernels/gguf_head_gate.h"
#include "core/rope.h"
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_cache_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "mfq_cuda_paged_kv.h"
#include "runtime/decode_window.h"
#include <cmath>
#include <cstring>
#include <iostream>

template <class Data, class Equal>
inline void qk_prepare_paged_case(const Data &data, const Equal &equal) {
    namespace tb = mfq_tensor_backend;
    int checked = 0;
    for (auto dtype : {tb::kFloat16, tb::kFloat32})
        for (int width : {31, 128, 256, 513})
            for (int batch : {1, 2, 3})
                for (int tokens : {1, 3, 8})
                    for (int page_size : {5, 16})
                        for (bool per_batch : {false, true}) {
                            constexpr int query_heads = 6, key_heads = 2, chunks = 3,
                                          pages_per_chunk = 4, logical_pages = 4;
                            const int capacity = logical_pages * page_size,
                                      rotary = 2 * (width / 4);
                            auto q = data({batch, tokens, query_heads, 2 * width + 3}, 1)
                                         .to(dtype)
                                         .narrow(3, 0, width)
                                         .transpose(1, 2);
                            auto k = data({batch, tokens, key_heads, width + 5}, 2)
                                         .to(dtype)
                                         .narrow(3, 0, width)
                                         .transpose(1, 2);
                            auto v = data({batch, tokens, key_heads, width + 7}, 3)
                                         .to(dtype)
                                         .narrow(3, 0, width)
                                         .transpose(1, 2);
                            auto qw = data({width}, 4), kw = data({width}, 5);
                            RopeCache rope(17, rotary, 10000.);
                            const auto integer =
                                tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
                            const auto shape = per_batch ? std::vector<int64_t>{batch, tokens}
                                                         : std::vector<int64_t>{tokens};
                            auto position = tb::zeros(shape, integer),
                                 slot = tb::zeros(shape, integer);
                            auto pages =
                                tb::zeros({batch, logical_pages}, integer.dtype(tb::kInt32));
                            auto initial_pages = tb::empty(
                                {batch, logical_pages}, integer.device(tb::kCPU).dtype(tb::kInt32));
                            for (int b = 0; b < batch; ++b)
                                for (int l = 0; l < logical_pages; ++l)
                                    initial_pages.data_ptr<int32_t>()[b * logical_pages + l] =
                                        (b * logical_pages + l) * 5 % (chunks * pages_per_chunk);
                            pages.copy_(initial_pages);
                            auto actual_k_ptrs = tb::empty({chunks}, integer),
                                 actual_v_ptrs = tb::empty({chunks}, integer);
                            auto expected_k_ptrs = tb::empty({chunks}, integer),
                                 expected_v_ptrs = tb::empty({chunks}, integer);
                            std::vector<tb::Tensor> initial, actual_k, actual_v, expected_k,
                                expected_v;
                            for (int c = 0; c < chunks; ++c) {
                                auto seed =
                                    data({pages_per_chunk, key_heads, page_size, width}, 23 + c)
                                        .to(tb::kFloat16);
                                initial.push_back(seed);
                                actual_k.push_back(seed.clone());
                                actual_v.push_back(seed.clone());
                                expected_k.push_back(seed.clone());
                                expected_v.push_back(seed.clone());
                            }
                            const auto reset = [&] {
                                for (int c = 0; c < chunks; ++c) {
                                    actual_k[c].copy_(initial[c]);
                                    actual_v[c].copy_(initial[c]);
                                    expected_k[c].copy_(initial[c]);
                                    expected_v[c].copy_(initial[c]);
                                }
                            };
                            const auto pointer_tables = [&](int repeat) {
                                auto ak = tb::empty({chunks}, integer.device(tb::kCPU)),
                                     av = tb::empty_like(ak);
                                auto ek = tb::empty_like(ak), ev = tb::empty_like(ak);
                                for (int c = 0; c < chunks; ++c) {
                                    const int physical = (c + repeat) % chunks;
                                    ak.data_ptr<int64_t>()[c] = int64_t(
                                        reinterpret_cast<intptr_t>(actual_k[physical].data_ptr()));
                                    av.data_ptr<int64_t>()[c] = int64_t(
                                        reinterpret_cast<intptr_t>(actual_v[physical].data_ptr()));
                                    ek.data_ptr<int64_t>()[c] = int64_t(reinterpret_cast<intptr_t>(
                                        expected_k[physical].data_ptr()));
                                    ev.data_ptr<int64_t>()[c] = int64_t(reinterpret_cast<intptr_t>(
                                        expected_v[physical].data_ptr()));
                                }
                                if (repeat == 3)
                                    ak.data_ptr<int64_t>()[2] = ek.data_ptr<int64_t>()[2] = 0;
                                actual_k_ptrs.copy_(ak);
                                actual_v_ptrs.copy_(av);
                                expected_k_ptrs.copy_(ek);
                                expected_v_ptrs.copy_(ev);
                            };
                            pointer_tables(0);
                            tb::Tensor actual;
                            const double offset = batch == 1 ? 0. : 1.;
                            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                            graph.capture(
                                [&] {
                                    actual = qk_norm_rope_paged_cache_cuda(
                                        q, k, v, qw, kw, position, slot, rope.cos, rope.sin,
                                        actual_k_ptrs, actual_v_ptrs, pages, page_size,
                                        pages_per_chunk, rotary, 1e-6, offset);
                                },
                                reset, [&] { actual = {}; });
                            for (int repeat = 0; repeat < 4; ++repeat) {
                                auto ph = tb::empty(shape, integer.device(tb::kCPU)),
                                     sh = tb::empty_like(ph);
                                for (int b = 0; b < (per_batch ? batch : 1); ++b)
                                    for (int t = 0; t < tokens; ++t) {
                                        ph.data_ptr<int64_t>()[b * tokens + t] =
                                            (repeat == 0   ? -3
                                             : repeat == 1 ? 23
                                                           : 7) +
                                            t + b;
                                        int64_t at = (repeat == 1 ? capacity - tokens : 2) + t;
                                        if (repeat == 2 && t == 0)
                                            at = -1;
                                        if (repeat == 2 && t == tokens - 1)
                                            at = capacity;
                                        sh.data_ptr<int64_t>()[b * tokens + t] = at;
                                    }
                                auto table = tb::empty({batch, logical_pages},
                                                       integer.device(tb::kCPU).dtype(tb::kInt32));
                                for (int b = 0; b < batch; ++b)
                                    for (int l = 0; l < logical_pages; ++l)
                                        table.data_ptr<int32_t>()[b * logical_pages + l] =
                                            ((b * logical_pages + l) * 5 + repeat) %
                                            (chunks * pages_per_chunk);
                                if (repeat == 3) {
                                    table.data_ptr<int32_t>()[0] = -1;
                                    table.data_ptr<int32_t>()[1] = chunks * pages_per_chunk;
                                }
                                position.copy_(ph);
                                slot.copy_(sh);
                                pages.copy_(table);
                                pointer_tables(repeat);
                                q.copy_(data(q.sizes().vec(), 7 + repeat).to(dtype));
                                k.copy_(data(k.sizes().vec(), 11 + repeat).to(dtype));
                                v.copy_(data(v.sizes().vec(), 15 + repeat).to(dtype));
                                auto normalized =
                                    dtype == tb::kFloat16
                                        ? rms_norm_pair_f16_f32_offset_cuda(
                                              q.contiguous(), k.contiguous(), qw, kw, 1e-6, offset)
                                        : std::vector<tb::Tensor>{
                                              rms_norm_offset_cuda(q.contiguous(), qw, 1e-6,
                                                                   offset),
                                              rms_norm_offset_cuda(k.contiguous(), kw, 1e-6,
                                                                   offset)};
                                auto expected = rope.apply(normalized[0], position);
                                auto expected_key = rope.apply(normalized[1], position)
                                                        .to(tb::kFloat16)
                                                        .contiguous();
                                reset();
                                paged_kv_cache_write_cuda(expected_k_ptrs, expected_v_ptrs, pages,
                                                          expected_key,
                                                          v.to(tb::kFloat16).contiguous(), slot,
                                                          page_size, pages_per_chunk);
                                graph.run();
                                equal(actual, expected,
                                      "Paged fused QK preparation changed FP32 query");
                                for (int c = 0; c < chunks; ++c) {
                                    equal(actual_k[c], expected_k[c],
                                          "Paged fused QK preparation changed physical key bytes");
                                    equal(
                                        actual_v[c], expected_v[c],
                                        "Paged fused QK preparation changed physical value bytes");
                                }
                                ++checked;
                            }
                        }
    std::cout << "Paged QK fused changing-graph physical-cache bitwise cases=" << checked
              << " PASS\n";
}

inline void qk_prepare_case() {
    namespace tb = mfq_tensor_backend;
    const auto data = [](std::vector<int64_t> shape, int salt) {
        auto host = tb::empty(shape, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for (int64_t i = 0; i < host.numel(); ++i)
            host.data_ptr<float>()[i] = std::sin(float(i) * .173f + salt) * .4f;
        return host.to(tb::kCUDA);
    };
    const auto equal = [](const tb::Tensor &a, const tb::Tensor &b, const char *message) {
        const auto x = a.cpu().contiguous(), y = b.cpu().contiguous();
        if (x.sizes() != y.sizes() || x.scalar_type() != y.scalar_type() ||
            std::memcmp(x.data_ptr(), y.data_ptr(), x.nbytes())) {
            if (x.scalar_type() == tb::kFloat32) {
                int64_t differences = 0;
                double error = 0.;
                for (int64_t i = 0; i < x.numel(); ++i)
                    if (x.data_ptr<float>()[i] != y.data_ptr<float>()[i]) {
                        if (differences < 8)
                            std::cerr << "diff " << i << " actual=" << x.data_ptr<float>()[i]
                                      << " expected=" << y.data_ptr<float>()[i] << '\n';
                        ++differences;
                        error = std::max(error, std::abs(double(x.data_ptr<float>()[i]) -
                                                         y.data_ptr<float>()[i]));
                    }
                std::cerr << "differing=" << differences << " max_abs=" << error << '\n';
            }
            throw std::runtime_error(message);
        }
    };
    int checked = 0;
    for (auto dtype : {tb::kFloat16, tb::kFloat32})
        for (int width : {31, 64, 128, 255, 256, 320, 513})
            for (int batch : {1, 2})
                for (int query_heads : {5, 6})
                    for (int tokens : {1, 3, 8, 17})
                        for (bool per_batch : {false, true}) {
                            const int key_heads = query_heads == 6 ? 2 : 1, capacity = tokens + 13,
                                      rotary = 2 * (width / 4);
                            auto q = data({batch, tokens, query_heads, 2 * width + 3}, 1)
                                         .to(dtype)
                                         .narrow(3, 0, width)
                                         .transpose(1, 2);
                            auto k = data({batch, tokens, key_heads, width + 5}, 2)
                                         .to(dtype)
                                         .narrow(3, 0, width)
                                         .transpose(1, 2);
                            auto v = data({batch, tokens, key_heads, width + 7}, 3)
                                         .to(dtype)
                                         .narrow(3, 0, width)
                                         .transpose(1, 2);
                            auto qw = data({width}, 4), kw = data({width}, 5);
                            RopeCache rope(17, rotary, 10000.);
                            const auto integer =
                                tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
                            const auto shape = per_batch ? std::vector<int64_t>{batch, tokens}
                                                         : std::vector<int64_t>{tokens};
                            auto position = tb::full(shape, 0, integer),
                                 slot = tb::full(shape, 0, integer);
                            auto initial =
                                data({batch, key_heads, capacity, width}, 6).to(tb::kFloat16);
                            auto kc = initial.clone(), vc = initial.clone(),
                                 reference_k = initial.clone(), reference_v = initial.clone();
                            tb::Tensor actual;
                            const auto reset = [&] {
                                kc.copy_(initial);
                                vc.copy_(initial);
                            };
                            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                            const double offset = batch == 1 ? 0. : 1.;
                            graph.capture(
                                [&] {
                                    actual = qk_norm_rope_cache_cuda(q, k, v, qw, kw, position,
                                                                     slot, rope.cos, rope.sin, kc,
                                                                     vc, rotary, 1e-6, offset);
                                },
                                reset, [&] { actual = {}; });
                            for (int repeat = 0; repeat < 3; ++repeat) {
                                const int target = repeat == 0   ? 2
                                                   : repeat == 1 ? capacity - tokens
                                                                 : 0;
                                auto ph = tb::empty(shape, integer.device(tb::kCPU)),
                                     sh = tb::empty_like(ph);
                                for (int b = 0; b < (per_batch ? batch : 1); ++b)
                                    for (int t = 0; t < tokens; ++t) {
                                        ph.data_ptr<int64_t>()[b * tokens + t] =
                                            (repeat == 0   ? -3
                                             : repeat == 1 ? 23
                                                           : 7) +
                                            t + b;
                                        sh.data_ptr<int64_t>()[b * tokens + t] = target + t;
                                        if (repeat == 2 && t == 0)
                                            sh.data_ptr<int64_t>()[b * tokens + t] = -1;
                                        else if (repeat == 2 && t == tokens - 1)
                                            sh.data_ptr<int64_t>()[b * tokens + t] = capacity;
                                    }
                                position.copy_(ph);
                                slot.copy_(sh);
                                q.copy_(data(q.sizes().vec(), 7 + repeat).to(dtype));
                                k.copy_(data(k.sizes().vec(), 11 + repeat).to(dtype));
                                v.copy_(data(v.sizes().vec(), 15 + repeat).to(dtype));
                                auto normalized =
                                    dtype == tb::kFloat16
                                        ? rms_norm_pair_f16_f32_offset_cuda(
                                              q.contiguous(), k.contiguous(), qw, kw, 1e-6, offset)
                                        : std::vector<tb::Tensor>{
                                              rms_norm_offset_cuda(q.contiguous(), qw, 1e-6,
                                                                   offset),
                                              rms_norm_offset_cuda(k.contiguous(), kw, 1e-6,
                                                                   offset)};
                                auto expected = rope.apply(normalized[0], position);
                                auto expected_key = rope.apply(normalized[1], position)
                                                        .to(tb::kFloat16)
                                                        .contiguous();
                                reference_k.copy_(initial);
                                reference_v.copy_(initial);
                                reset();
                                kv_cache_write_cuda(reference_k, reference_v, expected_key,
                                                    v.to(tb::kFloat16).contiguous(), slot);
                                graph.run();
                                equal(actual, expected, "Fused QK preparation changed FP32 query");
                                equal(
                                    kc, reference_k,
                                    "Fused QK preparation changed KV key bytes or untouched slots");
                                equal(vc, reference_v,
                                      "Fused QK preparation changed KV value bytes or untouched "
                                      "slots");
                                ++checked;
                            }
                        }
    std::cout << "QK fused normalization/rotary/cache changing-graph bitwise cases=" << checked
              << " PASS\n";
    int gate_checked = 0;
    for (auto dtype : {tb::kFloat16, tb::kFloat32})
        for (int batch : {1, 2})
            for (int tokens : {1, 3})
                for (int heads : {2, 4})
                    for (int width : {32, 80, 96})
                        for (bool silu : {false, true}) {
                            auto input = data({batch, tokens, heads * width}, 19).to(dtype);
                            auto gate = data({batch, tokens, heads, 2 * width + 5}, 23)
                                            .to(dtype)
                                            .narrow(3, 0, width);
                            auto flat = gate.contiguous().reshape_as(input);
                            const auto bytes =
                                batch * tokens *
                                int64_t(mfq::cuda::gguf::q8_row_bytes(heads * width));
                            auto expected = tb::empty({bytes}, input.options().dtype(tb::kUInt8)),
                                 actual = tb::empty_like(expected);
                            for (bool aligned : {false, true}) {
                                if (aligned && batch * tokens != 1)
                                    continue;
                                const auto quantize =
                                    aligned ? mfq::cuda::gguf::quantize_gated_input_m1_aligned
                                            : mfq::cuda::gguf::quantize_gated_input;
                                quantize(input.data_ptr(), flat.data_ptr(), dtype == tb::kFloat16,
                                         batch * tokens, heads * width, silu, expected.data_ptr(),
                                         mfq_current_cuda_stream());
                                mfq::cuda::gguf::quantize_gated_head_input(
                                    input.data_ptr(), gate.data_ptr(), dtype == tb::kFloat16, batch,
                                    tokens, heads, width, gate.stride(0), gate.stride(1),
                                    gate.stride(2), silu, aligned, actual.data_ptr(),
                                    mfq_current_cuda_stream());
                                equal(actual, expected,
                                      "Head-gated preparation changed Q8 codes/scales");
                                ++gate_checked;
                            }
                        }
    std::cout << "Strided head-gate native/aligned Q8 bitwise cases=" << gate_checked << " PASS\n";
    qk_prepare_paged_case(data, equal);
}
