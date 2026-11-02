#include "mlx_gguf.h"
#include "mlx_gguf_prefill.h"
#include "mlx_gguf_staged.h"
#include "mlx_gguf_expert_cache.h"
#include "native/mlx_native_gguf.h"
#include "mlx_moe_ops.h"
#include "mlx_kernel_prepare.h"
#include "mfq_container.h"
#include "gguf.h"
#include "ggml-quants.h"
#include <mlx/memory.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <chrono>
#include <climits>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <random>
#include <unordered_set>
#include <unistd.h>

namespace {
using namespace mlx::core;
using namespace mfq::metal;
using Clock = std::chrono::steady_clock;

void compare(const array& actual, const array& expected, const std::string& label) {
    const float error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    const float scale = max(abs(astype(expected, float32))).item<float>();
    const float tolerance = actual.dtype() == bfloat16 ? .025f : .003f;
    if (!std::isfinite(error) || error > tolerance * std::max(.01f, scale))
        throw std::runtime_error(label + " error=" + std::to_string(error) + " scale=" + std::to_string(scale));
}

bool has_prefill_gather(const array& output) {
    std::vector<array> pending{output};
    std::unordered_set<const Primitive*> visited;
    while (!pending.empty()) {
        const auto node = pending.back(); pending.pop_back();
        if (!node.has_primitive() || !visited.insert(node.primitive_ptr().get()).second) continue;
        if (std::string_view(node.primitive().name()) == "CustomKernel" || dynamic_cast<const MlxPreparableKernel*>(&node.primitive())) {
            const auto key = mlx_kernel_preparation_key(node.primitive());
            if (key.find("tinfer_gguf_prefill_gather") != std::string::npos ||
                key.find("tinfer_gguf_rotated_gate_up_gather") != std::string::npos) return true;
        }
        for (const auto& input : node.inputs()) pending.push_back(input);
    }
    return false;
}

std::vector<uint8_t> payload(ggml_type type, int k, int rows, unsigned seed_offset = 0, bool variable_scales = false) {
    std::vector<uint8_t> bytes(ggml_row_size(type, k) * rows);
    std::mt19937 rng(711 + int(type) + seed_offset);
    for (auto& b : bytes) b = uint8_t(rng());
    const float scale = type >= GGML_TYPE_TQ2_T ? .03125f : .00025f;
    for (std::size_t at = 0; at < bytes.size(); at += ggml_type_size(type)) {
        if (type == GGML_TYPE_MXFP4) {
            bytes[at] = 115u + (variable_scales ? rng() % 3u : 0u);
            continue;
        }
        if (type == GGML_TYPE_Q8_K) {
            const float d = scale * (variable_scales ? .75f + float(rng() % 19) / 16.f : 1.f);
            std::memcpy(bytes.data() + at, &d, sizeof(d));
            continue;
        }
        const auto d = ggml_fp32_to_fp16(scale * (variable_scales ? .75f + float(rng() % 19) / 16.f : 1.f));
        const int offset = type == GGML_TYPE_Q6_K ? 208 : type == GGML_TYPE_Q3_K ? 108 : type == GGML_TYPE_Q2_K ? 80
            : type == GGML_TYPE_PTQ1_0 ? 26 : 0;
        std::memcpy(bytes.data() + at + offset, &d, 2);
        if (type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
            const auto minimum = variable_scales ? ggml_fp32_to_fp16(scale * (.5f + float(rng() % 23) / 16.f)) : d;
            std::memcpy(bytes.data() + at + offset + 2, &minimum, 2);
        }
    }
    return bytes;
}

struct Fixture {
    std::filesystem::path path;
    Fixture(const std::array<ggml_type, 3>& formats, int h, int i, int e, unsigned rotations, bool signed_rotation = false, bool variable_scales = false) {
        path = std::filesystem::temp_directory_path() /
            ("tinfer-projection-moe-" + std::to_string(Clock::now().time_since_epoch().count()) + ".gguf");
        auto* context = ggml_init({4 << 20, nullptr, true});
        auto* file = gguf_init_empty();
        gguf_set_val_str(file, "general.architecture", "test");
        gguf_set_val_u32(file, "agention.trellis.state_bits", 15);
        const char* names[]{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
        std::vector<const char*> rotated;
        std::vector<std::vector<uint8_t>> payloads;
        for (int p = 0; p < 3; ++p) {
            const int k = p == 2 ? i : h, n = p == 2 ? h : i;
            auto* tensor = ggml_new_tensor_3d(context, formats[p], k, n, e);
            ggml_set_name(tensor, names[p]);
            auto& bytes = payloads.emplace_back(payload(formats[p], k, n * e, p * 104729u, variable_scales));
            tensor->data = bytes.data();
            gguf_add_tensor(file, tensor);
            if (rotations & (1u << p)) rotated.push_back(names[p]);
        }
        if (!rotated.empty()) {
            gguf_set_val_u32(file, "prism.hadamard.version", 1);
            gguf_set_val_u32(file, "prism.hadamard.block_size", 128);
            gguf_set_val_str(file, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
            gguf_set_val_str(file, "prism.hadamard.axis", "input-last-dimension");
            gguf_set_val_str(file, "prism.hadamard.sign_mode", signed_rotation ? "explicit" : "identity");
            if (signed_rotation) {
                const int32_t widths[]{h, i};
                std::vector<int32_t> signs(h + i);
                for (int col = 0; col < h + i; ++col) signs[col] = (col * 13 + col / 7) % 3 ? 1 : -1;
                gguf_set_arr_data(file, "prism.hadamard.sign_widths", GGUF_TYPE_INT32, widths, 2);
                gguf_set_arr_data(file, "prism.hadamard.sign_values", GGUF_TYPE_INT32, signs.data(), signs.size());
            }
            gguf_set_arr_str(file, "prism.hadamard.weight_names", rotated.data(), rotated.size());
        }
        const bool ok = gguf_write_to_file(file, path.string().c_str(), false);
        gguf_free(file); ggml_free(context);
        if (!ok) throw std::runtime_error("cannot write mixed-projection fixture");
    }
    ~Fixture() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};

array apply(const std::array<MlxGgufWeight, 3>& w, const array& x, const array& ids, const array& p) {
    return gguf_routed_ffn(w, x, ids, p, 0.f, true);
}

struct Comparison {
    int hidden = 2560, intermediate = 768;
    int experts = 64, top_k = 8, tile = 0, layers = 1, tokens = 0, repeats = 0, pair_a = -1, pair_b = -1, fixed_down = -1;
    bool rotated = false, inspect = false, prefill_only = false, decode_only = false, all_small = false, ffn = false, indirect = false;
    bool matched = false;
    bool breakdown = false;
    bool raw_layout = false;
    bool show_kernels = false;
    std::string routing = "uniform";
    std::array<ggml_type, 4> types{GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_TQK6, GGML_TYPE_TQP6};
};

void check_compact_routes() {
    constexpr int r = 3;
    for (const int m : {1, 2, 6, 8, 17, 257}) for (const int e : {1, 3, 64, 512})
      for (const int tile : {8, 32, 48}) for (const bool shared : {false, true}) {
        const auto ids = reshape(remainder(arange(m * r, int32), array(e + 3, int32)) - 1, {m, r});
        const auto routes = gguf_group_routes(ids, e, tile, shared, true);
        eval(ids, routes.descriptors, routes.count, routes.grouped_routes, routes.route_rows);
        std::vector<int> seen(m * (r + int(shared)));
        int end = 0;
        const auto count = routes.count.item<int32_t>();
        for (int at = 0; at < count; ++at) {
            const auto* desc = routes.descriptors.data<int32_t>() + at * 3;
            const int expert = desc[0], live = desc[1], start = desc[2];
            const int capacity = tile == 8 || live <= 8 ? 8 : live <= 16 ? 16 : live <= 32 ? 32 : 48;
            if (expert < 0 || expert >= e + int(shared) || live < 1 || live > tile || start != end || start + capacity > routes.row_capacity)
                throw std::runtime_error("compact GGUF descriptor bounds/contiguity");
            end = start + capacity;
            for (int local = 0; local < capacity; ++local) {
                const int original = routes.grouped_routes.data<int32_t>()[start + local];
                if (local >= live) {
                    if (original != -1) throw std::runtime_error("compact GGUF padding was not cleared");
                    continue;
                }
                if (original < 0 || original >= int(seen.size()) || ++seen[original] != 1 ||
                    routes.route_rows.data<int32_t>()[original] != start + local)
                    throw std::runtime_error("compact GGUF forward/reverse map");
                const int row = original / (r + int(shared)), rank = original % (r + int(shared));
                const int expected = rank == r ? e : ids.data<int32_t>()[row * r + rank];
                if (expert != expected) throw std::runtime_error("compact GGUF expert association");
            }
        }
        for (int row = 0; row < m; ++row) for (int rank = 0; rank < r + int(shared); ++rank) {
            const int expert = rank == r ? e : ids.data<int32_t>()[row * r + rank];
            const bool valid = rank == r || (expert >= 0 && expert < e);
            if (seen[row * (r + int(shared)) + rank] != int(valid)) throw std::runtime_error("compact GGUF route coverage");
        }
    }
}

void check_raw_down_routes() {
    constexpr int experts = 32, n = 513;
    for (const auto type : {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
            GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ2_S, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS})
        for (const int k : {256, 2048}) {
            const auto bytes = payload(type, k, n * experts, 917, true);
            const auto stride = ggml_row_size(type, k), expert_bytes = stride * n;
            std::vector<int32_t> physical;
            std::vector<array> storage;
            for (int e = 0; e < experts; ++e) {
                physical.push_back(e);
                storage.emplace_back(bytes.begin() + expert_bytes * e, Shape{n, int(stride)}, uint8);
            }
            auto name = "GGUF_" + std::string(ggml_type_name(type));
            for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
            const auto weight = MlxGgufWeight::from_experts(name, {experts, n, k}, physical, storage);
            for (const int m : {1, 2, 6}) for (const int r : {0, 1, 3, 8, 10, 17, 32}) {
                const auto ranks = reshape(arange(m * r, int32), {m, r});
                const auto ids = where(remainder(ranks, array(7, int32)) == array(6, int32), array(-1, int32),
                    where(remainder(ranks, array(11, int32)) == array(10, int32), array(experts + 1, int32),
                        remainder(ranks, array(experts, int32))));
                const auto probabilities = r ? softmax(astype(ranks, float32) * .017f, -1) : zeros({m, r}, float32);
                for (const auto dtype : {float16, bfloat16, float32}) {
                    const auto input = astype(reshape(sin(arange(m * r * k, float32) * .013f), {m, r, k}), dtype);
                    compare(weight.routed_weighted_matmul(input, ids, probabilities),
                        r ? moe_weighted_reduce(weight.routed_matmul(input, ids, false), probabilities) : zeros({m, n}, dtype),
                        "raw weighted Down routes=" + std::to_string(r) + " K=" + std::to_string(k) + " " + name);
                }
            }
        }
}

void check_raw_gate_up_dispatch() {
    constexpr int h = 2560, n = 769, r = 10;
    for (const auto high_type : {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_K})
      for (const auto type : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ4_XS}) {
        const auto load = [&](ggml_type format, unsigned seed, bool dense = false) {
            auto name = "GGUF_" + std::string(ggml_type_name(format));
            for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
            const auto stride = ggml_row_size(format, h);
            const auto bytes = payload(format, h, n, seed, true);
            if (dense) return MlxGgufWeight::from_bytes(name, bytes, {n, h});
            return MlxGgufWeight::from_experts(name, {1, n, h}, {0},
                {array(bytes.begin(), Shape{n, int(stride)}, uint8)});
        };
        const auto low = load(type, 37), high = load(high_type, 791);
        const auto low_dense = load(type, 37, true), high_dense = load(high_type, 791, true);
        for (const bool reverse : {false, true}) {
            const auto& gate = reverse ? high : low;
            const auto& up = reverse ? low : high;
            const auto& dense_gate = reverse ? high_dense : low_dense;
            const auto& dense_up = reverse ? low_dense : high_dense;
            for (const int m : {1, 4}) for (const auto dtype : {float16, bfloat16, float32}) {
                const auto input = astype(reshape(sin(arange(m * h, float32) * .017f), {m, h}), dtype);
                const auto ids = zeros({m, r}, int32);
                const auto dense = dense_gate.swiglu(dense_up, input);
                if (!mlx_kernel_preparation_key(dense.primitive()).ends_with("_independent"))
                    throw std::runtime_error("dense mixed Gate/Up SIMD dispatch");
                compare(dense, moe_swiglu_pair(dense_gate.matmul(input), dense_up.matmul(input)), "dense independent Gate/Up");
                const auto expected_gate = gate.routed_matmul(input, ids, true);
                const auto expected_up = up.routed_matmul(input, ids, true);
                for (const float limit : {0.f, -0.f, .125f, .25f}) {
                    const auto actual = gate.routed_swiglu(up, input, ids, nullptr, limit);
                    const auto key = mlx_kernel_preparation_key(actual.primitive());
                    if (!key.ends_with("_independent"))
                        throw std::runtime_error("raw mixed Gate/Up SIMD dispatch");
                    compare(actual, limit > 0.f ? moe_limited_swiglu_pair(expected_gate, expected_up, limit)
                        : moe_swiglu_pair(expected_gate, expected_up), "raw mixed independent Gate/Up");
                }
                if (m == 4) {
                    const auto compiled = compile([&](const std::vector<array>& values) {
                        return std::vector{gate.routed_swiglu(up, values[0], ids, nullptr, .125f),
                            gate.routed_swiglu(up, values[0], ids, nullptr, .25f)};
                    });
                    std::vector<array> outputs, expected;
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        const auto next = input + array(.03125f * repeat, dtype);
                        const auto result = compiled({next});
                        outputs.insert(outputs.end(), result.begin(), result.end());
                        const auto g = gate.routed_matmul(next, ids, true), u = up.routed_matmul(next, ids, true);
                        expected.push_back(moe_limited_swiglu_pair(g, u, .125f));
                        expected.push_back(moe_limited_swiglu_pair(g, u, .25f));
                    }
                    auto together = outputs;
                    together.insert(together.end(), expected.begin(), expected.end());
                    eval(together);
                    for (std::size_t at = 0; at < outputs.size(); ++at)
                        compare(outputs[at], expected[at], "compiled raw mixed Gate/Up limits/workspace");
                }
            }
        }
    }
    const auto partial = [&](ggml_type format, int slot) {
        auto name = "GGUF_" + std::string(ggml_type_name(format));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        const auto stride = ggml_row_size(format, h);
        const auto bytes = payload(format, h, n, 179, true);
        return MlxGgufWeight::from_experts(name, {2, n, h}, {slot},
            {array(bytes.begin(), Shape{n, int(stride)}, uint8)});
    };
    for (const auto high_type : {GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_K}) {
    const auto gate = partial(GGML_TYPE_Q4_K, 0), up = partial(high_type, 1);
    for (const int m : {1, 4}) for (const auto dtype : {float16, bfloat16, float32}) {
        const auto input = full({m, h}, 60000.f, dtype);
        const auto ids = reshape(remainder(arange(m * r, int32), array(4, int32)) - 1, {m, r});
        compare(gate.routed_swiglu(up, input, ids), zeros({m, r, n}, dtype), "disjoint raw Gate/Up slots");
        compare(up.routed_swiglu(gate, input, ids), zeros({m, r, n}, dtype), "reversed disjoint raw Gate/Up slots");
    }
    }
}

void check_raw_iq_gate_up() {
    constexpr int h = 2560, n = 769, r = 3;
    for (const auto pair : {std::pair{GGML_TYPE_Q2_K, GGML_TYPE_IQ3_S}, std::pair{GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS},
            std::pair{GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS}, std::pair{GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS},
            std::pair{GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_S}, std::pair{GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S},
            std::pair{GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S}, std::pair{GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS},
            std::pair{GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S}}) {
        std::vector<MlxGgufWeight> direct, indirect, dense;
        std::vector<array> reference;
        for (const auto type : {pair.first, pair.second}) {
            const auto bytes = payload(type, h, n, 131u + unsigned(direct.size()) * 104729u, true);
            auto name = "GGUF_" + std::string(ggml_type_name(type));
            for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
            const auto previous = std::getenv("TYLOINFER_METAL_DISABLE_NAX");
            const auto saved = previous ? std::optional<std::string>(previous) : std::nullopt;
            setenv("TYLOINFER_METAL_DISABLE_NAX", "1", 1);
            direct.push_back(MlxGgufWeight::from_bytes(name, bytes, {1, n, h}));
            dense.push_back(MlxGgufWeight::from_bytes(name, bytes, {n, h}));
            if (saved) setenv("TYLOINFER_METAL_DISABLE_NAX", saved->c_str(), 1);
            else unsetenv("TYLOINFER_METAL_DISABLE_NAX");
            indirect.push_back(MlxGgufWeight::from_experts(name, {1, n, h}, {0},
                {array(bytes.begin(), Shape{n, int(ggml_row_size(type, h))}, uint8)}));
            std::vector<float> values(h * n);
            ggml_get_type_traits(type)->to_float(bytes.data(), values.data(), values.size());
            reference.push_back(array(values.begin(), Shape{n, h}, float32));
        }
        for (const int m : {1, 2, 6}) for (const auto dtype : {float16, bfloat16, float32}) {
            const auto input = astype(reshape(sin(arange(m * h, float32) * .017f), {m, h}), dtype);
            const auto ids = reshape(remainder(arange(m * r, int32), array(3, int32)) - 1, {m, r});
            const auto g = astype(matmul(astype(input, float32), transpose(reference[0]), Device::cpu), dtype);
            const auto u = astype(matmul(astype(input, float32), transpose(reference[1]), Device::cpu), dtype);
            const auto context = " " + std::string(ggml_type_name(pair.first)) + "/" + ggml_type_name(pair.second) +
                " M=" + std::to_string(m) + " dtype=" + (dtype == float16 ? "f16" : dtype == bfloat16 ? "bf16" : "f32");
            compare(dense[0].matmul(input), g, "IQ gate GGML reference" + context);
            compare(dense[1].matmul(input), u, "IQ up GGML reference" + context);
            for (const bool reverse : {false, true}) for (const float limit : {0.f, .125f}) {
                const int a = int(reverse), b = 1 - a;
                const auto expected = limit > 0.f ? moe_limited_swiglu_pair(reverse ? u : g, reverse ? g : u, limit)
                    : moe_swiglu_pair(reverse ? u : g, reverse ? g : u);
                const auto routed = where(expand_dims(equal(ids, array(0, int32)), -1),
                    broadcast_to(reshape(expected, {m, 1, n}), {m, r, n}), zeros({m, r, n}, dtype));
                const auto detail = context + " reverse=" + std::to_string(reverse) + " limit=" + std::to_string(limit);
                compare(direct[a].routed_swiglu(direct[b], input, ids, nullptr, limit), routed, "raw IQ Gate/Up tail" + detail);
                compare(indirect[a].routed_swiglu(indirect[b], input, ids, nullptr, limit), routed, "indirect IQ Gate/Up tail" + detail);
                if (!limit) compare(dense[a].swiglu(dense[b], input), expected, "dense IQ Gate/Up tail" + detail);
            }
        }
    }
}

void check_raw_compact_mixed(bool extended_rows = false) {
    constexpr int h = 512, n = 256, e = 64, r = 10;
    if (!gguf_expert_prefill_available(zeros({7, h}, float16))) return;
    const auto load = [&](ggml_type type, int outputs, int width, unsigned seed) {
        auto name = "GGUF_" + std::string(ggml_type_name(type));
        for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
        const auto bytes = payload(type, width, outputs * e, seed, true);
        const auto previous = std::getenv("TYLOINFER_METAL_DISABLE_NAX");
        const auto saved = previous ? std::optional<std::string>(previous) : std::nullopt;
        setenv("TYLOINFER_METAL_DISABLE_NAX", "1", 1);
        auto weight = MlxGgufWeight::from_bytes(name, bytes, {e, outputs, width});
        if (saved) setenv("TYLOINFER_METAL_DISABLE_NAX", saved->c_str(), 1);
        else unsetenv("TYLOINFER_METAL_DISABLE_NAX");
        if (weight.has_staged_planes()) throw std::runtime_error("raw compact fixture unexpectedly staged weights");
        return weight;
    };
    for (const auto pair : {std::pair{GGML_TYPE_Q3_K, GGML_TYPE_IQ2_S}, std::pair{GGML_TYPE_Q3_K, GGML_TYPE_IQ2_XS},
            std::pair{GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XS}, std::pair{GGML_TYPE_Q3_K, GGML_TYPE_IQ2_XXS},
            std::pair{GGML_TYPE_Q3_K, GGML_TYPE_IQ3_XXS}, std::pair{GGML_TYPE_Q3_K, GGML_TYPE_IQ3_S},
            std::pair{GGML_TYPE_Q3_K, GGML_TYPE_TQK5}, std::pair{GGML_TYPE_Q3_K, GGML_TYPE_TQP6},
            std::pair{GGML_TYPE_Q8_K, GGML_TYPE_Q4_K}, std::pair{GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS}}) {
        const std::array weights{load(pair.first, n, h, 17), load(pair.second, n, h, 197), load(pair.first, h, n, 791)};
        for (const int m : extended_rows ? std::vector<int>{7, 8} : std::vector<int>{1, 2, 3, 4, 5, 6})
            for (const bool repeat : {false, true}) for (const auto dtype : {float16, bfloat16, float32}) {
            const auto input = astype(reshape(sin(arange(m * h, float32) * .013f), {m, h}), dtype);
            const auto ids = reshape(remainder(arange(m * r, int32), array(repeat ? r : e, int32)), {m, r});
            const auto probabilities = softmax(astype(ids, float32) * .017f, -1);
            const bool iq2 = pair.second == GGML_TYPE_IQ2_XXS || pair.second == GGML_TYPE_IQ2_XS || pair.second == GGML_TYPE_IQ2_S;
            if ((m > 1 && m <= 6 && iq2) != gguf_compact_ffn_available(weights[0], weights[1], weights[2], ids))
                throw std::runtime_error("raw IQ2 mixed compact FFN was not selected");
            const MlxGgufPrefillPlan compact(input, ids, e, 8);
            for (const bool reverse : {false, true}) {
                const std::array ordered{weights[reverse ? 1 : 0], weights[reverse ? 0 : 1], weights[2]};
                const auto gate = ordered[0].routed_matmul(input, ids, true), up = ordered[1].routed_matmul(input, ids, true);
                for (const float limit : {0.f, .125f}) {
                    const auto hidden = limit > 0.f ? moe_limited_swiglu_pair(gate, up, limit) : moe_swiglu_pair(gate, up);
                    const auto expected = moe_weighted_reduce(ordered[2].routed_matmul(hidden, ids, false), probabilities);
                    compare(gguf_routed_ffn(ordered, input, ids, probabilities, limit, true), expected,
                        "raw IQ2 compact mixed FFN M=" + std::to_string(m));
                    compare(gguf_routed_ffn(ordered, input, ids, probabilities, limit), expected,
                        "raw IQ2 generic compact mixed FFN M=" + std::to_string(m));
                    const auto grouped_hidden = compact.gate_up(ordered[0], ordered[1], nullptr, nullptr, limit);
                    if (extended_rows && dtype == float16 && has_prefill_gather(grouped_hidden))
                        throw std::runtime_error("extended source rows still gather the input");
                    compare(compact.down(ordered[2], grouped_hidden, probabilities),
                        expected, "raw mixed explicit compact FFN M=" + std::to_string(m));
                }
            }
            if (m == 6 && repeat && dtype == float16) {
                const auto bad_ids = where(remainder(ids, array(5, int32)) == array(4, int32), array(-1, int32),
                    where(remainder(ids, array(7, int32)) == array(6, int32), array(e + 2, int32), ids));
                const auto hidden = moe_swiglu_pair(weights[0].routed_matmul(input, bad_ids, true), weights[1].routed_matmul(input, bad_ids, true));
                compare(gguf_routed_ffn(weights, input, bad_ids, probabilities),
                    moe_weighted_reduce(weights[2].routed_matmul(hidden, bad_ids, false), probabilities), "raw compact invalid expert IDs");
            }
        }
    }
}

void check_mixed_shared() {
    constexpr int h = 512, i = 256, e = 4, r = 3;
    for (const auto type : {GGML_TYPE_Q8_0, GGML_TYPE_Q8_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S})
      for (const bool reverse : {false, true}) {
        const std::array formats{reverse ? GGML_TYPE_Q4_K : type, reverse ? type : GGML_TYPE_Q4_K, type};
        Fixture fixture(formats, h, i, e, 0, false, true);
        const MfqContainer model(fixture.path);
        const std::array weights{
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.gate.weight"),
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.up.weight"),
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.down.weight")};
        const auto shared = [&](int n, int k, unsigned seed) {
            const auto format = type == GGML_TYPE_IQ2_XS || type == GGML_TYPE_IQ2_S ? type : GGML_TYPE_Q4_K;
            auto name = "GGUF_" + std::string(ggml_type_name(format));
            for (auto& c : name) c = std::toupper(static_cast<unsigned char>(c));
            auto weight = MlxGgufWeight::from_bytes(name, payload(format, k, n, seed, true), {n, k});
            weight.prepare_grouped();
            return weight;
        };
        const std::array share{shared(i, h, 179), shared(i, h, 991), shared(h, i, 7717)};
        for (const int m : {1, 2, 6, 65}) for (const auto dtype : {float16, bfloat16, float32}) {
            const auto input = astype(reshape(sin(arange(m * h, float32) * .013f), {m, h}), dtype);
            const auto ids = reshape(remainder(arange(m * r, int32), array(e, int32)), {m, r});
            const auto probabilities = concatenate({softmax(astype(ids, float32) * .017f, -1) * array(.75f),
                full({m, 1}, .25f)}, 1);
            const auto gate = weights[0].routed_matmul(input, ids, true), up = weights[1].routed_matmul(input, ids, true);
            const auto sg = share[0].matmul(input), su = share[1].matmul(input);
            const auto plan = MlxGgufPrefillPlan::from_topk(input, ids, e, m <= 6 ? 8 : 32, true);
            for (const float limit : {0.f, .125f}) {
                const auto hidden = limit > 0.f ? moe_limited_swiglu_pair(gate, up, limit) : moe_swiglu_pair(gate, up);
                const auto shared_hidden = limit > 0.f ? moe_limited_swiglu_pair(sg, su, limit) : moe_swiglu_pair(sg, su);
                const auto expected = moe_weighted_reduce(concatenate({weights[2].routed_matmul(hidden, ids, false),
                    expand_dims(share[2].matmul(shared_hidden), 1)}, 1), probabilities);
                compare(plan.down(weights[2], plan.gate_up(weights[0], weights[1], &share[0], &share[1], limit),
                    probabilities, false, &share[2]), expected, "mixed routed/shared FFN M=" + std::to_string(m));
            }
        }
    }
}

void check_projection_alignments() {
    constexpr int h = 1280, i = 640, e = 4, r = 3;
    for (const auto formats : {std::array{GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_TQK6},
            std::array{GGML_TYPE_IQ2_S, GGML_TYPE_TQK4, GGML_TYPE_TQP5},
            std::array{GGML_TYPE_TQP6, GGML_TYPE_Q5_K, GGML_TYPE_TQK7}}) {
        unsigned rotations = 0;
        for (int part = 0; part < 3; ++part) if (formats[part] >= GGML_TYPE_TQ2_T) rotations |= 1u << part;
        Fixture fixture(formats, h, i, e, rotations, true, true);
        const MfqContainer model(fixture.path);
        const std::array weights{
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.gate.weight"),
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.up.weight"),
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.down.weight")};
        const std::array dense{weights[0].dequantize(float32), weights[1].dequantize(float32), weights[2].dequantize(float32)};
        MlxGgufExpertCache cache(model, {"model.block.0"}, std::size_t(64) << 20, 2);
        for (const int m : {1, 2, 6, 7, 8, 17}) {
            const auto ids = reshape(remainder(arange(m * r, int32), array(e, int32)), {m, r});
            const auto probabilities = softmax(astype(ids, float32) * .017f, -1);
            for (const auto dtype : {float16, bfloat16, float32}) {
                const auto input = astype(reshape(sin(arange(m * h, float32) * .013f), {m, h}), dtype);
                const auto project = [&](const array& x, int part) {
                    return astype(squeeze(matmul(expand_dims(astype(x, float32), -2),
                        swapaxes(take(dense[part], ids, 0), -1, -2)), -2), dtype);
                };
                const auto gate = project(expand_dims(input, 1), 0), up = project(expand_dims(input, 1), 1);
                for (const float limit : {0.f, .125f}) {
                    const auto hidden = limit > 0.f ? moe_limited_swiglu_pair(gate, up, limit) : moe_swiglu_pair(gate, up);
                    const auto expected = moe_weighted_reduce(project(hidden, 2), probabilities);
                    compare(gguf_routed_ffn(weights, input, ids, probabilities, limit, true),
                        expected, "independent projection block alignments M=" + std::to_string(m));
                    if (dtype == float16) compare(cache.apply_ffn(0, input, ids, probabilities, nullptr, limit),
                        expected, "SSD independent projection block alignments M=" + std::to_string(m));
                }
            }
        }
    }
}

void compare_formats(const Comparison& options) {
    const int h = options.hidden, i = options.intermediate;
    const int e = options.experts, r = options.top_k, tile = options.tile;
    const bool rotated = options.rotated, inspect = options.inspect;
    if (h < 256 || i < 256 || h % 256 || i % 256 || std::int64_t(h) * i * e > INT_MAX ||
        e < 2 || e > 512 || r < 1 || r > e || (tile != 0 && tile != 32 && tile != 48) ||
        options.layers < 1 || options.layers > 128 || (options.layers > 1 && (!options.ffn || inspect || tile ||
            (!options.decode_only && !(options.matched && options.prefill_only && options.layers == 2)))) ||
        (options.matched && (options.layers < 2 || options.layers % 2)) ||
        options.tokens < 0 || options.tokens > 4096 || options.repeats < 0 || options.repeats > 64 || (options.decode_only && options.tokens > 6) ||
        options.pair_a < -1 || options.pair_a > 3 || options.pair_b < -1 || options.pair_b > 3 ||
        ((options.pair_a < 0) != (options.pair_b < 0)) ||
        options.fixed_down < -1 || options.fixed_down > 3 ||
        (options.fixed_down >= 0 && (!options.ffn || !options.matched)) ||
        (options.routing != "uniform" && options.routing != "repeat" && options.routing != "hot"))
        throw std::invalid_argument("invalid comparison configuration");
    std::cout << "geometry H=" << h << " I=" << i << " E=" << e << " topk=" << r << " common_rotation=" << rotated
        << " routing=" << options.routing << " ffn=" << options.ffn << " dependent_layers=" << options.layers
        << " indirect=" << options.indirect << " matched_homogeneous=" << options.matched << " raw_layout=" << options.raw_layout
        << " fixed_down=" << options.fixed_down
        << " distinct_projection_seeds=1 bounded_chain_input=1 layer_route_stride=token_rows*topk" << std::endl;
    const auto& types = options.types;
    std::cout << "formats=";
    for (const auto type : types) std::cout << ggml_type_name(type) << ',';
    std::cout << std::endl;
    std::vector<std::unique_ptr<Fixture>> fixtures;
    std::vector<std::array<MlxGgufWeight, 3>> banks;
    for (const auto type : types) {
        fixtures.push_back(std::make_unique<Fixture>(std::array{type, type, type}, h, i, 1,
            rotated || type >= GGML_TYPE_TQ2_T ? 7 : 0));
        const MfqContainer model(fixtures.back()->path);
        const auto load = [&](const char* role, int n, int k) {
            const auto name = "model.block.0.mlp.experts." + std::string(role) + ".weight";
            auto dtype = "GGUF_" + std::string(ggml_type_name(type));
            for (auto& c : dtype) c = std::toupper(static_cast<unsigned char>(c));
            const unsigned salt = role[0] == 'g' ? 0u : role[0] == 'u' ? 104729u : 209458u;
            if (options.indirect) {
                const auto packed = payload(type, k, n * e, salt);
                const auto stride = ggml_row_size(type, k), bytes = stride * n;
                std::vector<array> experts;
                std::vector<int32_t> ids;
                for (int expert = 0; expert < e; ++expert) {
                    experts.emplace_back(packed.begin() + bytes * expert, Shape{n, int(stride)}, uint8);
                    ids.push_back(expert);
                }
                auto weight = MlxGgufWeight::from_experts(dtype, {e, n, k}, ids, experts);
                weight.configure_hadamard(model, name);
                return weight;
            }
            const auto previous = std::getenv("TYLOINFER_METAL_DISABLE_NAX");
            const auto saved = previous ? std::optional<std::string>(previous) : std::nullopt;
            setenv("TYLOINFER_METAL_DISABLE_NAX", "1", 1);
            auto weight = MlxGgufWeight::from_bytes(dtype, payload(type, k, n * e, salt), {e, n, k});
            if (saved) setenv("TYLOINFER_METAL_DISABLE_NAX", saved->c_str(), 1);
            else unsetenv("TYLOINFER_METAL_DISABLE_NAX");
            weight.configure_hadamard(model, name);
            if (!options.raw_layout && gguf_expert_prefill_available(zeros({8, h}, float16)))
                MlxGgufStagedPrefill::prepare(weight);
            std::cout << "weight=" << role << " format=" << ggml_type_name(type) << " staged_planes=" << weight.has_staged_planes()
                << " hadamard=" << weight.has_hadamard() << std::endl;
            return weight;
        };
        banks.push_back({load("gate", i, h), load("up", i, h), load("down", h, i)});
    }
    const auto selected = [&](int a, int b) { return options.pair_a < 0 ||
        ((a == options.pair_a || a == options.pair_b) && (b == options.pair_a || b == options.pair_b)); };
    for (const int m : options.tokens ? std::vector<int>{options.tokens} : options.all_small ? std::vector<int>{1, 2, 3, 4, 5, 6, 4096} : std::vector<int>{1, 6, 4096}) {
        if ((tile || inspect || options.prefill_only) && m <= 6) continue;
        if (options.decode_only && m > 6) continue;
        const auto x = astype(reshape(sin(arange(m * h, float32) * .013f), {m, h}), float16);
        auto ids = reshape(remainder(arange(m * r, int32), array(options.routing == "repeat" ? r : e, int32)), {m, r});
        if (options.routing == "hot") {
            const int hot = std::min(64, e / 2), selected_hot = std::min(hot, std::max(1, r * 4 / 5));
            std::vector<int> values(m * r);
            for (int token = 0; token < m; ++token) for (int slot = 0; slot < r; ++slot)
                values[token * r + slot] = slot < selected_hot ? (token * 7 + slot) % hot
                    : hot + (token * 11 + slot) % (e - hot);
            ids = array(values.begin(), Shape{m, r}, int32);
        }
        const auto p = full({m, r}, 1.f / r, float32);
        const auto hidden = astype(reshape(cos(arange(m * r * i, float32) * .017f), {m, r, i}), float16);
        eval(x, ids, p, hidden);
        std::vector<array> layer_ids;
        for (int layer = 0; layer < options.layers; ++layer)
            layer_ids.push_back(remainder(ids + array(layer * m * r, int32), array(e, int32)));
        eval(layer_ids);
        const auto upper = [&](int a, int b, bool reference = false) {
            auto input = x;
            for (int layer = 0; layer < options.layers; ++layer) {
                const auto& routes = layer_ids[layer];
                const int gate = options.matched && layer % 2 ? b : a;
                const int up = options.matched ? (reference ? gate : layer % 2 ? a : b) : b;
                const int down = options.fixed_down >= 0 ? options.fixed_down : gate;
                auto output = [&]() {
                    if (options.ffn) {
                        if (reference && !options.matched) return banks[a][2].routed_weighted_matmul(banks[a][0].routed_swiglu(banks[b][1], input, routes), routes, p);
                        return gguf_routed_ffn(std::array{banks[gate][0], banks[up][1], banks[down][2]}, input, routes, p, 0.f, true);
                    }
                    if (!tile) return banks[a][0].routed_swiglu(banks[b][1], input, routes);
                    const MlxGgufPrefillPlan plan(input, routes, e, tile);
                    return plan.restore(plan.gate_up(banks[a][0], banks[b][1]));
                }();
                input = options.layers == 1 ? std::move(output) : x + tanh(output) * array(.01f, input.dtype());
            }
            return input;
        };
        for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
            if (!selected(a, b)) continue;
            const auto output = upper(a, b);
            std::map<std::string, int> names;
            if (options.show_kernels) {
                std::vector<array> pending{output};
                std::unordered_set<const Primitive*> visited;
                while (!pending.empty()) {
                    const auto node = pending.back(); pending.pop_back();
                    if (!node.has_primitive() || !visited.insert(node.primitive_ptr().get()).second) continue;
                    if (std::string_view(node.primitive().name()) == "CustomKernel" || dynamic_cast<const MlxPreparableKernel*>(&node.primitive()))
                        ++names[mlx_kernel_preparation_key(node.primitive())];
                    for (const auto& input : node.inputs()) pending.push_back(input);
                }
                for (const auto& [name, count] : names) std::cout << "dispatch a=" << a << " b=" << b << " count=" << count << " kernel=" << name << std::endl;
            }
            eval(output);
            if (!std::isfinite(max(abs(astype(output, float32))).item<float>()))
                throw std::runtime_error("nonfinite mixed GGUF comparison output");
            if (options.ffn) {
                const auto reference = upper(a, b, true);
                if (!std::isfinite(max(abs(astype(reference, float32))).item<float>()))
                    throw std::runtime_error("nonfinite homogeneous GGUF comparison output");
            }
            if (options.layers > 1 && !options.matched) compare(upper(a, b), upper(a, b, true), "dependent mixed GGUF FFN chain");
        }
        for (int a = 0; a < 4; ++a) eval(banks[a][2].routed_weighted_matmul(hidden, ids, p));
        synchronize();
        if (inspect) {
            for (const auto pair : {std::pair{0, 0}, {1, 1}, {2, 2}, {3, 3}, {0, 2}, {2, 0}, {1, 3}, {3, 1}}) {
                for (int pass = 0; pass < 4; ++pass) {
                    const auto output = upper(pair.first, pair.second);
                    std::vector<array> pending{output};
                    std::vector<array> core;
                    std::unordered_set<const Primitive*> visited;
                    while (!pending.empty()) {
                        const auto node = pending.back(); pending.pop_back();
                        if (!node.has_primitive() || !visited.insert(node.primitive_ptr().get()).second) continue;
                        if ((std::string_view(node.primitive().name()) == "CustomKernel" ||
                            dynamic_cast<const MlxPreparableKernel*>(&node.primitive())) &&
                            (mlx_kernel_preparation_key(node.primitive()).find("tinfer_gguf_prefill_staged_gate_up") != std::string::npos ||
                             mlx_kernel_preparation_key(node.primitive()).find("tinfer_gguf_prefill_staged_down") != std::string::npos)) {
                            core.push_back(node);
                        }
                        for (const auto& input : node.inputs()) pending.push_back(input);
                    }
                    if (core.empty()) throw std::runtime_error("grouped Gate/Up was not exercised");
                    const auto start = Clock::now();
                    std::vector<array> inputs;
                    for (const auto& node : core) for (const auto& input : node.inputs())
                        if (!input.has_primitive() || std::none_of(core.begin(), core.end(), [&](const auto& value) {
                            return value.primitive_ptr() == input.primitive_ptr();
                        })) inputs.push_back(input);
                    eval(inputs);
                    const auto ready = Clock::now();
                    eval(core);
                    const auto finished = Clock::now();
                    eval(output);
                    const auto restored = Clock::now();
                    std::cout << "a=" << pair.first << " b=" << pair.second << " pass=" << pass
                        << " dependencies_ms=" << std::chrono::duration<double, std::milli>(ready - start).count()
                        << " core_ms=" << std::chrono::duration<double, std::milli>(finished - ready).count()
                        << " restore_ms=" << std::chrono::duration<double, std::milli>(restored - finished).count() << std::endl;
                }
            }
            continue;
        }
        const int repeats = options.repeats ? options.repeats : options.layers > 2 ? 8 : m > 6 || options.layers > 1 ? 3 : options.ffn ? 32 : 64;
        const int pair_count = options.ffn ? 32 : 16, cases = pair_count + 4;
        for (int pass = 0; pass < 6; ++pass) {
            std::vector<int> order;
            if (options.ffn) {
                for (int at = 0; at < 20; ++at) {
                    const int pair = (at * 7 + pass * 3) % 20;
                    if (pair >= 16) order.push_back(pair + 16);
                    else {
                        order.push_back(pair + (pass % 2 ? 16 : 0));
                        order.push_back(pair + (pass % 2 ? 0 : 16));
                    }
                }
            } else for (int at = 0; at < cases; ++at) order.push_back((at * 7 + pass * 3) % cases);
            for (const int slot : order) {
                const int a = slot < pair_count ? (slot % 16) / 4 : slot - pair_count, b = slot % 4;
                if (!selected(a, b)) continue;
                const auto start = Clock::now();
                double graph_ms = 0., execute_ms = 0.;
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    if (!options.breakdown) {
                        if (slot < pair_count) eval(upper(a, b, slot >= 16));
                        else eval(banks[a][2].routed_weighted_matmul(hidden, ids, p));
                        synchronize();
                        continue;
                    }
                    const auto graph_start = Clock::now();
                    const auto output = slot < pair_count ? upper(a, b, slot >= 16) : banks[a][2].routed_weighted_matmul(hidden, ids, p);
                    const auto built = Clock::now();
                    eval(output);
                    synchronize();
                    graph_ms += std::chrono::duration<double, std::milli>(built - graph_start).count();
                    execute_ms += std::chrono::duration<double, std::milli>(Clock::now() - built).count();
                }
                const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count() / repeats /
                    (slot < pair_count ? options.layers : 1);
                std::cout << "stage=" << (slot < pair_count ? (options.ffn ? (slot >= 16 ? (options.matched ? (options.fixed_down >= 0 ? "homogeneous-gu" : "homogeneous") : "ffn-ref") : "ffn") : "gu") : "down") << " a=" << a << " b=" << b
                    << " M=" << m << " pass=" << pass << " ms=" << ms << " unix_ms="
                    << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
                    << (options.breakdown ? " graph_ms=" + std::to_string(graph_ms / repeats / (slot < pair_count ? options.layers : 1)) +
                        " execute_ms=" + std::to_string(execute_ms / repeats / (slot < pair_count ? options.layers : 1)) : "") << std::endl;
            }
        }
    }
}

void compare_native_formats(const Comparison& options) {
    const int h = options.hidden, i = options.intermediate, e = options.experts, r = options.top_k;
    if (h < e || h % 256 || i < 256 || i % 256 || e < 2 || e > 512 || r < 1 || r > e ||
        std::int64_t(h) * i * e > INT_MAX || !options.ffn || !options.matched ||
        options.layers < 2 || options.layers > 128 || options.layers % 2 || options.rotated || options.indirect ||
        options.raw_layout || options.tile || options.inspect || options.show_kernels ||
        options.tokens < 0 || options.tokens > 4096 || options.repeats < 0 || options.repeats > 64 ||
        options.pair_a < -1 || options.pair_a > 3 || options.pair_b < -1 || options.pair_b > 3 ||
        ((options.pair_a < 0) != (options.pair_b < 0)) || options.fixed_down < -1 || options.fixed_down > 3 ||
        (options.routing != "uniform" && options.routing != "repeat"))
        throw std::invalid_argument("invalid native complete MoE comparison configuration");
    std::cout << "geometry native_complete_moe=1 shared=q4_K router=BF16 H=" << h << " I=" << i
        << " E=" << e << " topk=" << r << " routing=" << options.routing << " dependent_layers=" << options.layers
        << " matched_homogeneous=1 layer_slot_stride=token_rows*topk" << std::endl;
    const auto pack = [&](ggml_type type, int n, int k, int experts, unsigned seed) {
        if (!mlx_native_gguf_supported(type, n, k)) throw std::invalid_argument("unsupported native MoE comparison format");
        if (std::size_t(n) * experts * ggml_row_size(type, k) > INT_MAX)
            throw std::invalid_argument("native MoE comparison weight exceeds the array limit");
        const auto bytes = payload(type, k, n * experts, seed);
        auto weight = mlx_native_gguf_pack(array(bytes.begin(), Shape{int(bytes.size())}, uint8),
            type, n * experts, k, int(ggml_row_size(type, k)));
        weight.outputs = n; weight.experts = experts;
        eval(std::vector<array>(weight.planes.begin(), weight.planes.end()));
        return weight;
    };
    std::vector<std::array<MlxNativeGgufWeight, 3>> banks;
    for (const auto type : options.types) {
        banks.push_back({pack(type, i, h, e, 0), pack(type, i, h, e, 104729), pack(type, h, i, e, 209458)});
        std::cout << "native_format=" << ggml_type_name(type) << std::endl;
    }
    const std::array shared{pack(GGML_TYPE_Q4_K, i, h, 1, 314159), pack(GGML_TYPE_Q4_K, i, h, 1, 314160),
        pack(GGML_TYPE_Q4_K, h, i, 1, 314161)};
    const auto router = astype(equal(expand_dims(arange(e, int32), 1), expand_dims(arange(h, int32), 0)), bfloat16);
    const auto scalar = zeros({1, h}, float32);
    eval(router, scalar);
    const auto selected = [&](int a, int b) { return options.pair_a < 0 ||
        ((a == options.pair_a || a == options.pair_b) && (b == options.pair_a || b == options.pair_b)); };
    for (const int m : options.tokens ? std::vector<int>{options.tokens} : options.all_small ?
        std::vector<int>{1, 2, 3, 4, 5, 6, 4096} : std::vector<int>{1, 6, 4096}) {
        if ((options.decode_only && m > 6) || (options.prefill_only && m <= 6)) continue;
        auto x = reshape(sin(arange(m * h, float32) * .013f), {m, h});
        const auto logical = expand_dims(arange(e, int32), 0);
        const auto starts = options.routing == "repeat" ? zeros({m, 1}, int32) : reshape(arange(m, int32) * r, {m, 1});
        const auto scores = where(remainder(logical - starts + array(e, int32), array(e, int32)) < array(r, int32),
            array(1.f), array(-1.f));
        x = astype(h == e ? scores : concatenate({scores, slice(x, {0, e}, {m, h})}, 1), float16);
        std::vector<array> slots;
        for (int layer = 0; layer < options.layers; ++layer)
            slots.push_back(remainder(arange(e, uint32) + array(uint32_t(layer * m * r)), array(uint32_t(e))));
        eval(x); eval(slots);
        const auto execute = [&](int a, int b, bool reference) {
            auto input = x;
            for (int layer = 0; layer < options.layers; ++layer) {
                const int gate = layer % 2 ? b : a, up = reference ? gate : layer % 2 ? a : b;
                const int down = options.fixed_down >= 0 ? options.fixed_down : gate;
                const MlxNativeGgufMoe weights{{banks[gate][0], banks[up][1], banks[down][2]}, shared,
                    router, scalar, slots[layer], r};
                input = x + tanh(mlx_native_gguf_moe(weights, input)) * array(.01f, input.dtype());
            }
            return input;
        };
        for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) if (selected(a, b)) {
            for (const bool reference : {false, true}) {
                const auto output = execute(a, b, reference);
                if (!std::isfinite(max(abs(astype(output, float32))).item<float>()))
                    throw std::runtime_error("nonfinite native complete MoE comparison output");
            }
        }
        synchronize();
        const int repeats = options.repeats ? options.repeats : 1;
        for (int pass = 0; pass < 6; ++pass) for (int at = 0; at < 16; ++at) {
            const int pair = (at * 7 + pass * 3) % 16, a = pair / 4, b = pair % 4;
            if (!selected(a, b)) continue;
            for (int order = 0; order < 2; ++order) {
                const bool reference = (order + pass) % 2;
                const auto start = Clock::now();
                double graph_ms = 0., execute_ms = 0.;
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    if (!options.breakdown) { eval(execute(a, b, reference)); synchronize(); continue; }
                    const auto graph_start = Clock::now();
                    const auto output = execute(a, b, reference);
                    const auto built = Clock::now();
                    eval(output);
                    synchronize();
                    graph_ms += std::chrono::duration<double, std::milli>(built - graph_start).count();
                    execute_ms += std::chrono::duration<double, std::milli>(Clock::now() - built).count();
                }
                std::cout << "stage=" << (reference ? (options.fixed_down >= 0 ? "homogeneous-gu" : "homogeneous") : "ffn")
                    << " a=" << a << " b=" << b << " M=" << m << " pass=" << pass << " ms="
                    << std::chrono::duration<double, std::milli>(Clock::now() - start).count() / repeats / options.layers
                    << " unix_ms=" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
                    << (options.breakdown ? " graph_ms=" + std::to_string(graph_ms / repeats / options.layers) +
                        " execute_ms=" + std::to_string(execute_ms / repeats / options.layers) : "") << std::endl;
            }
        }
    }
}

void run(bool benchmark, bool breakdown = false, bool decode_only = false, bool small_decode = false,
    int selected_case = -1, bool wide = false, bool cache_only = false) {
    const int h = wide ? 7168 : benchmark ? 2560 : 512, i = wide ? 2048 : benchmark ? 768 : 256;
    const int e = wide ? 2 : small_decode ? 64 : decode_only ? 16 : benchmark ? 64 : 4;
    const std::array<std::array<ggml_type, 3>, 64> cases{{
        {GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_TQK6},
        {GGML_TYPE_TQP6, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_TQK4, GGML_TYPE_TQP5},
        {GGML_TYPE_TQK6, GGML_TYPE_TQP5, GGML_TYPE_Q6_K},
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL},
        {GGML_TYPE_Q2_K, GGML_TYPE_TQP5, GGML_TYPE_Q3_K},
        {GGML_TYPE_TQK4, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K},
        {GGML_TYPE_IQ2_XS, GGML_TYPE_Q5_K, GGML_TYPE_TQK7},
        {GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_TQP7},
        {GGML_TYPE_TQP7, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS},
        {GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL},
        {GGML_TYPE_Q5_K, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL},
        {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ2_S},
        {GGML_TYPE_TQK4, GGML_TYPE_TQP5, GGML_TYPE_IQ2_XS},
        {GGML_TYPE_TQP5, GGML_TYPE_TQK4, GGML_TYPE_Q5_K},
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_TQP6, GGML_TYPE_Q4_K},
        {GGML_TYPE_TQP5, GGML_TYPE_IQ2_XS, GGML_TYPE_TQK4},
        {GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_S},
        {GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q2_K},
        {GGML_TYPE_Q3_K, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K},
        {GGML_TYPE_Q8_0, GGML_TYPE_TQK6, GGML_TYPE_Q4_K},
        {GGML_TYPE_TQP5, GGML_TYPE_Q8_0, GGML_TYPE_IQ3_XXS},
        {GGML_TYPE_IQ3_S, GGML_TYPE_TQK7, GGML_TYPE_Q2_K},
        {GGML_TYPE_Q5_K, GGML_TYPE_TQK5, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_TQK7, GGML_TYPE_TQP7, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K},
        {GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_Q5_K},
        {GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_Q4_K},
        {GGML_TYPE_Q3_K, GGML_TYPE_IQ2_S, GGML_TYPE_Q6_K},
        {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, GGML_TYPE_Q4_K},
        {GGML_TYPE_Q8_0, GGML_TYPE_IQ2_XS, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ2_S, GGML_TYPE_Q8_0, GGML_TYPE_TQP5},
        {GGML_TYPE_TQ2_T, GGML_TYPE_Q3_K, GGML_TYPE_IQ3_S},
        {GGML_TYPE_TQK5, GGML_TYPE_TQ2_T, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_TQ2_T, GGML_TYPE_TQP7, GGML_TYPE_Q5_K},
        {GGML_TYPE_TQ2_T, GGML_TYPE_TQK5, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_Q2_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ2_XS},
        {GGML_TYPE_PTQ1_0, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q5_K},
        {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K},
        {GGML_TYPE_Q5_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS},
        {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q5_K},
        {GGML_TYPE_Q2_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_S},
        {GGML_TYPE_Q8_K, GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS},
        {GGML_TYPE_Q4_K, GGML_TYPE_Q8_K, GGML_TYPE_TQP6},
        {GGML_TYPE_Q4_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q8_K},
        {GGML_TYPE_Q8_0, GGML_TYPE_Q8_K, GGML_TYPE_TQK4},
        {GGML_TYPE_Q8_K, GGML_TYPE_Q8_K, GGML_TYPE_Q8_K},
        {GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0},
        {GGML_TYPE_Q8_0, GGML_TYPE_Q5_K, GGML_TYPE_Q4_K},
        {GGML_TYPE_Q8_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q4_K},
        {GGML_TYPE_Q2_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS},
        {GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_S},
        {GGML_TYPE_Q2_K, GGML_TYPE_IQ3_S, GGML_TYPE_Q6_K},
        {GGML_TYPE_Q6_K, GGML_TYPE_IQ3_S, GGML_TYPE_Q2_K},
        {GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_S}}};
    if (selected_case < -1 || selected_case >= int(cases.size())) throw std::invalid_argument("invalid numeric fixture index");
    for (std::size_t c = 0; c < cases.size(); ++c) {
        if (selected_case >= 0 && c != std::size_t(selected_case)) continue;
        unsigned rotations = 0;
        for (int p = 0; p < 3; ++p) if (cases[c][p] >= GGML_TYPE_TQ2_T) rotations |= 1u << p;
        if (!benchmark && c == 0) rotations |= 1;
        if (!benchmark && c == 1) rotations |= 2;
        Fixture fixture(cases[c], h, i, e, rotations, !benchmark && c % 2, !benchmark);
        const MfqContainer model(fixture.path);
        const std::array weights{
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.gate.weight"),
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.up.weight"),
            MlxGgufWeight::load(model, "model.block.0.mlp.experts.down.weight")};
        const bool grouped_supported = std::all_of(weights.begin(), weights.end(), [](const auto& w) {
            return MlxGgufStagedPrefill::supports_grouped(w);
        });
        std::vector<array> dense, cache_matrices;
        if (!benchmark) for (int part = 0; part < 3; ++part) {
            dense.push_back(weights[part].dequantize(float32));
            const int k = part == 2 ? i : h, n = part == 2 ? h : i;
            const auto bytes = payload(cases[c][part], k, n * e, part * 104729u, true);
            std::vector<float> values(std::size_t(e) * n * k);
            if (cases[c][part] == GGML_TYPE_Q8_K)
                dequantize_row_q8_K(reinterpret_cast<const block_q8_K*>(bytes.data()), values.data(), values.size());
            else {
                const auto decode = ggml_get_type_traits(cases[c][part])->to_float;
                if (!decode) throw std::runtime_error("numeric fixture has no reference decoder");
                decode(bytes.data(), values.data(), values.size());
            }
            cache_matrices.emplace_back(values.begin(), Shape{e, n, k}, float32);
        }
        for (const int m : cache_only ? std::vector<int>{1, 2, 6, 65}
            : decode_only ? (small_decode ? std::vector<int>{1, 2, 3, 4, 5, 6} : wide ? std::vector<int>{1, 2, 6} : std::vector<int>{1})
            : benchmark ? std::vector<int>{1, 6, 4096} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 65}) {
            const int r = wide ? 1 : decode_only ? 10 : benchmark ? 8 : 3;
            const auto x = astype(reshape(sin(arange(m * h, float32) * .013f), {m, h}), float16);
            const auto ids = reshape(remainder(arange(m * r, int32), array(e, int32)), {m, r});
            const auto p = full({m, r}, 1.f / r, float32);
            eval(x, ids, p);
            if (!benchmark) {
                if (c == 0 && m == 1) {
                    bool rejected = false;
                    try { (void)gguf_routed_ffn(std::array{weights[0], weights[1], weights[1]}, x, ids, p); }
                    catch (const std::invalid_argument&) { rejected = true; }
                    if (!rejected) throw std::runtime_error("mismatched FFN Down geometry was accepted");
                }
                const auto dense_mm = [&](const array& input, int index) {
                    auto w = take(dense[index], ids, 0);
                    return squeeze(matmul(expand_dims(astype(input, float32), -2), swapaxes(w, -1, -2)), -2);
                };
                const auto g = astype(dense_mm(expand_dims(x, 1), 0), float16);
                const auto u = astype(dense_mm(expand_dims(x, 1), 1), float16);
                const auto hidden = moe_swiglu_pair(g, u);
                const auto expected = moe_weighted_reduce(astype(dense_mm(hidden, 2), float16), p);
                if (!cache_only) {
                compare(apply(weights, x, ids, p), expected, "mixed-projection FFN case=" + std::to_string(c) + " M=" + std::to_string(m));
                if (c == 10 && m <= 6) {
                    const auto empty = zeros_like(x);
                    compare(weights[0].routed_swiglu(weights[1], empty, ids), zeros({m, r, i}, float16), "zero mixed Gate/Up publication");
                    compare(apply(weights, empty, ids, p), zeros_like(x), "zero mixed FFN publication");
                }
                if (c == 10 && m == 2) {
                    const auto duplicate_ids = zeros_like(ids);
                    const auto duplicate_hidden = moe_swiglu_pair(weights[0].routed_matmul(x, duplicate_ids, true),
                        weights[1].routed_matmul(x, duplicate_ids, true));
                    compare(gguf_routed_ffn(weights, x, duplicate_ids, p),
                        weights[2].routed_weighted_matmul(duplicate_hidden, duplicate_ids, p), "duplicate-route generic FFN");
                    compare(gguf_routed_ffn(weights, x, full_like(ids, -1), p, 0.f, true),
                        zeros_like(x), "empty mixed FFN route set");
                }
                for (const auto dtype : {float16, bfloat16, float32}) {
                    const auto context = " case=" + std::to_string(c) + " M=" + std::to_string(m) +
                        " dtype=" + (dtype == float16 ? "F16" : dtype == bfloat16 ? "BF16" : "F32");
                    const auto input = astype(x, dtype);
                    const auto gate = weights[0].routed_matmul(input, ids, true);
                    const auto up = weights[1].routed_matmul(input, ids, true);
                    if (c == 12 && m == 1 && !weights[0].has_staged_planes() && !weights[1].has_staged_planes()) {
                        const auto fused = weights[0].routed_swiglu(weights[1], input, ids);
                        if (mlx_kernel_preparation_key(fused.primitive()).find("_8_32_") == std::string::npos)
                            throw std::runtime_error("raw contracting Gate/Up did not select the projection geometry");
                    }
                    const auto clipped = moe_limited_swiglu_pair(gate, up, .125f);
                    if ((c == 8 || c == 10) && m <= 6) {
                        const auto unrestricted = moe_swiglu_pair(gate, up);
                        compare(weights[0].routed_swiglu(weights[1], input, ids, nullptr, -0.f), unrestricted,
                            "negative-zero Gate/Up limit" + context);
                        compare(gguf_routed_ffn(weights, input, ids, p, -0.f, true),
                            weights[2].routed_weighted_matmul(unrestricted, ids, p), "negative-zero FFN limit" + context);
                    }
                    compare(weights[0].routed_swiglu(weights[1], input, ids, nullptr, .125f), clipped, "mixed-projection clipped Gate/Up" + context);
                    if (c == 8 && m == 2 && !weights[0].has_staged_planes() && !weights[1].has_staged_planes()) {
                        const auto fused = weights[0].routed_swiglu(weights[1], input, ids, nullptr, .125f);
                        if (!mlx_kernel_preparation_key(fused.primitive()).ends_with("_independent"))
                            throw std::runtime_error("raw Q6 mixed Gate/Up did not select independent SIMD groups");
                    }
                    const auto reference = weights[2].routed_weighted_matmul(clipped, ids, p);
                    compare(gguf_routed_ffn(weights, input, ids, p, .125f), reference, "mixed-projection clipped FFN" + context);
                    compare(gguf_routed_ffn(weights, input, ids, p, .125f, true), reference, "top-k mixed-projection clipped FFN" + context);
                    if (c == 24 && m <= 6) {
                        const auto reversed = moe_limited_swiglu_pair(up, gate, .125f);
                        compare(gguf_routed_ffn(std::array{weights[1], weights[0], weights[2]}, input, ids, p, .125f, true),
                            weights[2].routed_weighted_matmul(reversed, ids, p), "reversed high-bit mixed FFN" + context);
                    }
                    if ((c == 8 || c == 10 || c == 13 || c == 24) && (m == 1 || m == 2 || m == 6) && dtype != float32 &&
                        (c == 8 || gguf_expert_prefill_available(zeros({7, h}, dtype)))) {
                        const auto compiled = compile([&](const std::vector<array>& values) {
                            return std::vector{gguf_routed_ffn(weights, values[0], values[1], p, .125f, true)};
                        });
                        std::vector<array> outputs, references;
                        for (int repeat = 0; repeat < 4; ++repeat) {
                            const auto next = repeat == 0 ? zeros_like(input) : input + array(.01f * repeat, dtype);
                            const auto routes = remainder(ids + array(repeat, int32), array(e, int32));
                            outputs.push_back(compiled({next, routes}).front());
                            references.push_back(gguf_routed_ffn(weights, next, routes, p, .125f, true));
                        }
                        auto together = outputs;
                        together.insert(together.end(), references.begin(), references.end());
                        eval(together);
                        for (std::size_t at = 0; at < outputs.size(); ++at)
                            compare(outputs[at], references[at], "simultaneous compiled mixed FFN workspace" + context);
                    }
                    if (grouped_supported && m <= 6 && gguf_expert_prefill_available(zeros({7, h}, dtype))) {
                        const MlxGgufPrefillPlan compact(input, ids, e, 8);
                        compare(compact.down(weights[2], compact.gate_up(weights[0], weights[1], nullptr, nullptr, .125f), p),
                            reference, "compact mixed-projection clipped FFN" + context);
                        const auto topk = MlxGgufPrefillPlan::from_topk(input, ids, e, 8);
                        compare(topk.down(weights[2], topk.gate_up(weights[0], weights[1], nullptr, nullptr, .125f), p),
                            reference, "top-k compact mixed-projection clipped FFN" + context);
                    }
                }
                if (grouped_supported && gguf_expert_prefill_available(zeros({7, h}, x.dtype()))) {
                    const MlxGgufPrefillPlan plan(x, ids, e, m <= 8 ? 8 : gguf_prefill_tile_rows(weights[0], weights[1], weights[2], ids));
                    compare(plan.down(weights[2], plan.gate_up(weights[0], weights[1]), p), expected, "mixed-projection grouped FFN");
                }
                }
                if (m == 1 || m == 2 || m == 6 || m == 65) {
                    const auto context = " case=" + std::to_string(c) + " M=" + std::to_string(m);
                    std::size_t cache_budget = 1 << 20;
                    if (wide) {
                        cache_budget = 0;
                        const auto page = std::size_t(getpagesize());
                        for (int part = 0; part < 3; ++part) {
                            const auto bytes = ggml_row_size(cases[c][part], part == 2 ? i : h) * (part == 2 ? h : i);
                            cache_budget += (bytes + page - 1) / page * page;
                        }
                    }
                    MlxGgufExpertCache cache(model, {"model.block.0"}, cache_budget, 2);
                    compare(cache.apply_ffn(0, x, ids, p), expected, "mixed-projection SSD FFN" + context);
                    const bool half_weights = m > 6;
                    const auto cache_mm = [&](const array& input, int part) {
                        auto transformed = input;
                        if (rotations & (1u << part)) {
                            auto source = astype(input, float32);
                            if (c % 2) {
                                const int width = input.shape(-1), offset = part == 2 ? h : 0;
                                std::vector<float> signs(width);
                                for (int col = 0; col < width; ++col) {
                                    const int at = offset + col;
                                    signs[col] = (at * 13 + at / 7) % 3 ? 1.f : -1.f;
                                }
                                source = source * array(signs.begin(), Shape{width}, float32);
                            }
                            transformed = astype(reshape(hadamard_transform(reshape(source, {-1, 128}),
                                .08838834764831845f), input.shape()), input.dtype());
                        }
                        const auto matrix = take(half_weights ? astype(cache_matrices[part], float16) : cache_matrices[part], ids, 0);
                        return squeeze(matmul(expand_dims(astype(transformed, float32), -2),
                            swapaxes(astype(matrix, float32), -1, -2)), -2);
                    };
                    const auto clipped = moe_limited_swiglu_pair(astype(cache_mm(expand_dims(x, 1), 0), float16),
                        astype(cache_mm(expand_dims(x, 1), 1), float16), .125f);
                    const auto actual_clipped = cache.apply_ffn(0, x, ids, p, nullptr, .125f);
                    const auto reference_clipped = moe_weighted_reduce(astype(cache_mm(clipped, 2), float16), p);
                    compare(actual_clipped, reference_clipped, "mixed-projection clipped SSD FFN" + context);
                    if ((c == 18 || c == 19 || c >= 25) && (m == 2 || m == 6)) {
                        const auto page = std::size_t(getpagesize());
                        std::size_t budget = 0;
                        for (int part = 0; part < 3; ++part) {
                            const auto bytes = ggml_row_size(cases[c][part], part == 2 ? i : h) * (part == 2 ? h : i);
                            budget += (bytes + page - 1) / page * page;
                        }
                        MlxGgufExpertCache partial(model, {"model.block.0"}, budget, 2);
                        compare(partial.apply_ffn(0, x, ids, p), expected, "mixed-projection single-expert SSD batches");
                        compare(partial.apply_ffn(0, x, ids, p, nullptr, .125f),
                            reference_clipped, "clipped single-expert SSD batches" + context);
                    }
                }
                continue;
            }
            eval(apply(weights, x, ids, p)); synchronize();
            for (int pass = 0; pass < 6; ++pass) {
                const auto begin = Clock::now();
                double graph_ms = 0., eval_ms = 0., drain_ms = 0.;
                for (int repeat = 0; repeat < 3; ++repeat) {
                    if (!breakdown) { eval(apply(weights, x, ids, p)); synchronize(); continue; }
                    const auto start = Clock::now();
                    const auto output = apply(weights, x, ids, p);
                    const auto built = Clock::now();
                    eval(output);
                    const auto ready = Clock::now();
                    synchronize();
                    const auto drained = Clock::now();
                    graph_ms += std::chrono::duration<double, std::milli>(built - start).count();
                    eval_ms += std::chrono::duration<double, std::milli>(ready - built).count();
                    drain_ms += std::chrono::duration<double, std::milli>(drained - ready).count();
                }
                const auto ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count() / 3.;
                std::cout << "case=" << c << " M=" << m << " pass=" << pass << " ms=" << ms;
                if (breakdown) std::cout << " graph_ms=" << graph_ms / 3. << " eval_ms=" << eval_ms / 3. << " drain_ms=" << drain_ms / 3.;
                std::cout << std::endl;
            }
        }
        if (!benchmark) std::cout << "mixed-projection numeric case=" << c << " passed" << std::endl;
    }
}
}

int main(int argc, char** argv) {
    try {
        set_default_device(Device::gpu);
        set_cache_limit(std::size_t(8) << 30);
        if (argc > 1 && std::string_view(argv[1]) == "check-shared") {
            if (!gguf_expert_prefill_available(zeros({7, 512}, float16))) return 77;
            check_mixed_shared();
            std::cout << "mixed routed/shared FFN passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-alignment") {
            check_projection_alignments();
            std::cout << "independent projection block alignments passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-gate-up") {
            check_raw_gate_up_dispatch();
            std::cout << "raw mixed Gate/Up dispatch passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-routes") {
            check_compact_routes();
            std::cout << "compact route maps passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-raw-gate-up") {
            check_raw_gate_up_dispatch();
            std::cout << "raw mixed Gate/Up dispatch passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-raw-iq-gate-up") {
            check_raw_iq_gate_up();
            std::cout << "raw IQ Gate/Up dispatch passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-raw-compact") {
            check_raw_compact_mixed();
            std::cout << "raw mixed compact FFN passed\n";
            return 0;
        }
        if (argc > 1 && std::string_view(argv[1]) == "check-source-rows") {
            check_raw_compact_mixed(true);
            std::cout << "extended compact source rows passed\n";
            return 0;
        }
        if (argc > 1 && (std::string_view(argv[1]) == "compare" || std::string_view(argv[1]) == "compare-native")) {
            const bool native = std::string_view(argv[1]) == "compare-native";
            Comparison options;
            if (native) options.types = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS};
            for (int at = 2; at < argc; ++at) {
                const std::string_view arg(argv[at]);
                if (arg == "--common-rotation") options.rotated = true;
                else if (arg == "--inspect") options.inspect = true;
                else if (arg == "--prefill-only") options.prefill_only = true;
                else if (arg == "--decode-only") options.decode_only = true;
                else if (arg == "--all-small") options.all_small = true;
                else if (arg == "--ffn") options.ffn = true;
                else if (arg == "--indirect") options.indirect = true;
                else if (arg == "--matched") options.matched = true;
                else if (arg == "--breakdown") options.breakdown = true;
                else if (arg == "--raw-layout") options.raw_layout = true;
                else if (arg == "--show-kernels") options.show_kernels = true;
                else if (at + 1 < argc && arg == "--experts") options.experts = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--hidden") options.hidden = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--intermediate") options.intermediate = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--top-k") options.top_k = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--tile") options.tile = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--layers") options.layers = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--tokens") options.tokens = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--repeats") options.repeats = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--fixed-down") options.fixed_down = std::stoi(argv[++at]);
                else if (at + 1 < argc && arg == "--pair") {
                    const std::string pair(argv[++at]);
                    const auto colon = pair.find(':');
                    if (colon == std::string::npos) throw std::invalid_argument("--pair requires two format indices, e.g. 0:1");
                    options.pair_a = std::stoi(pair.substr(0, colon));
                    options.pair_b = std::stoi(pair.substr(colon + 1));
                }
                else if (at + 1 < argc && arg == "--routing") options.routing = argv[++at];
                else if (at + 1 < argc && arg == "--formats") {
                    std::string_view names(argv[++at]);
                    for (int index = 0; index < 4; ++index) {
                        const auto comma = names.find(',');
                        const auto name = names.substr(0, comma);
                        bool found = false;
                        for (int type = 0; type < GGML_TYPE_COUNT; ++type) {
                            const char* candidate = ggml_type_name(ggml_type(type));
                            if (candidate && name == candidate) { options.types[index] = ggml_type(type); found = true; break; }
                        }
                        if (!found || (index == 3) != (comma == std::string_view::npos))
                            throw std::invalid_argument("--formats requires four supported GGUF format names");
                        if (comma != std::string_view::npos) names.remove_prefix(comma + 1);
                    }
                }
                else throw std::invalid_argument("unknown comparison option");
            }
            if (native) compare_native_formats(options); else compare_formats(options);
            return 0;
        }
        const bool breakdown = argc > 1 && std::string_view(argv[1]) == "benchmark-breakdown";
        const bool benchmark = breakdown || (argc > 1 && std::string_view(argv[1]) == "benchmark");
        const bool small_decode = argc > 1 && std::string_view(argv[1]) == "check-small";
        const bool wide = argc > 1 && std::string_view(argv[1]) == "check-wide";
        const bool cache_only = argc > 1 && std::string_view(argv[1]) == "check-cache";
        const bool decode_only = small_decode || wide || (argc > 1 && std::string_view(argv[1]) == "check-decode");
        int selected_case = -1;
        for (int at = 2; at < argc; ++at) {
            if (std::string_view(argv[at]) == "--case" && at + 1 < argc) selected_case = std::stoi(argv[++at]);
            else throw std::invalid_argument("unknown numeric fixture option");
        }
        if (!benchmark && !decode_only && !cache_only) {
            check_compact_routes(); check_raw_down_routes(); check_raw_gate_up_dispatch(); check_raw_iq_gate_up();
            check_raw_compact_mixed(); check_raw_compact_mixed(true);
        }
        if (wide && selected_case < 0) throw std::invalid_argument("check-wide requires --case");
        run(benchmark, breakdown, decode_only, small_decode, selected_case, wide, cache_only);
        std::cout << "mixed-projection GGUF MoE passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
