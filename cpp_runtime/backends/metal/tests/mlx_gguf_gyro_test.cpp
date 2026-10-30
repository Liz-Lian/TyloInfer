#include "mlx_gguf.h"
#include "mlx_gguf_expert_cache.h"
#include "mlx_gguf_prefill.h"
#include "mlx_moe_ops.h"
#include "mlx_kernel_prepare.h"
#include "mfq_container.h"
#include "gguf.h"
#include <mlx/primitives.h>

#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <unistd.h>

namespace {
using namespace mfq::metal;
using namespace mlx::core;

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void compare(const array& actual, const array& expected, float tolerance, const char* message) {
    const auto error = max(abs(astype(actual, float32) - astype(expected, float32))).item<float>();
    const auto scale = max(abs(astype(expected, float32))).item<float>();
    require(std::isfinite(error) && error <= tolerance * std::max(1.f, scale), message);
}

std::vector<float> inverse_rows(std::vector<float> values, int width) {
    for (std::size_t start = 0; start < values.size(); start += 128) {
        for (int size = 1; size < 128; size *= 2) for (int group = 0; group < 128; group += size * 2)
            for (int j = 0; j < size; ++j) {
                const auto a = values[start + group + j], b = values[start + group + size + j];
                values[start + group + j] = a + b; values[start + group + size + j] = a - b;
            }
        for (int j = 0; j < 128; ++j) {
            const auto col = int((start + j) % width);
            const float sign = col % (width == 256 ? 3 : 5) == 0 ? -1.f : 1.f;
            values[start + j] *= sign / std::sqrt(128.f);
        }
    }
    return values;
}
}

int main() {
    using namespace mlx::core;
    using namespace mfq::metal;
    const auto path = std::filesystem::temp_directory_path() /
        ("tinfer-gyro-rotation-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    const auto identity_path = std::filesystem::path(path.string() + ".identity.gguf");
    try {
        set_default_device(Device::gpu);
        auto* tensors = ggml_init({4 << 20, nullptr, false});
        auto* file = gguf_init_empty();
        gguf_set_val_str(file, "general.architecture", "test");
        gguf_set_val_u32(file, "agention.trellis.state_bits", 15);
        gguf_set_val_u32(file, "prism.hadamard.version", 1);
        gguf_set_val_u32(file, "prism.hadamard.block_size", 128);
        gguf_set_val_str(file, "prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
        gguf_set_val_str(file, "prism.hadamard.axis", "input-last-dimension");
        gguf_set_val_str(file, "prism.hadamard.sign_mode", "explicit");
        const int32_t widths[]{256, 128};
        std::vector<int32_t> signs;
        for (const auto width : widths) for (int c = 0; c < width; ++c) signs.push_back(c % (width == 256 ? 3 : 5) == 0 ? -1 : 1);
        gguf_set_arr_data(file, "prism.hadamard.sign_widths", GGUF_TYPE_INT32, widths, 2);
        gguf_set_arr_data(file, "prism.hadamard.sign_values", GGUF_TYPE_INT32, signs.data(), signs.size());
        const char* names[]{"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.attn_q.weight"};
        gguf_set_arr_str(file, "prism.hadamard.weight_names", names, 4);
        std::vector<array> original;
        std::size_t largest = 0;
        const auto page = std::size_t(getpagesize());
        for (int p = 0; p < 4; ++p) {
            const int experts = p == 3 ? 1 : 64, n = p == 2 ? 256 : p == 3 ? 64 : 128, k = p == 2 ? 128 : 256;
            const auto type = std::array{GGML_TYPE_TQK4, GGML_TYPE_TQP5, GGML_TYPE_TQK6, GGML_TYPE_Q8_0}[p];
            auto* tensor = experts == 1 ? ggml_new_tensor_2d(tensors, type, k, n) : ggml_new_tensor_3d(tensors, type, k, n, experts);
            ggml_set_name(tensor, names[p]);
            auto* bytes = static_cast<uint8_t*>(tensor->data);
            std::mt19937 rng(81 + p);
            for (std::size_t i = 0; i < ggml_nbytes(tensor); ++i) bytes[i] = uint8_t(rng());
            const auto scale = ggml_fp32_to_fp16(.03125f);
            for (std::size_t b = 0; b < ggml_nbytes(tensor); b += ggml_type_size(type)) std::memcpy(bytes + b, &scale, 2);
            std::vector<float> decoded(experts * n * k);
            ggml_get_type_traits(type)->to_float(bytes, decoded.data(), decoded.size());
            decoded = inverse_rows(std::move(decoded), k);
            original.emplace_back(decoded.begin(), experts == 1 ? Shape{n, k} : Shape{experts, n, k});
            if (p < 3) {
                const auto extent = ggml_nbytes(tensor) / experts;
                largest += (extent + page - 1) / page * page;
            }
            gguf_add_tensor(file, tensor);
        }
        const auto reject = [&] {
            require(gguf_write_to_file(file, path.string().c_str(), false), "cannot write invalid Gyro fixture");
            bool rejected = false;
            try { MfqContainer invalid(path); } catch (const std::exception&) { rejected = true; }
            require(rejected, "invalid Gyro transform metadata was accepted");
        };
        gguf_set_val_u32(file, "agention.trellis.state_bits", 14); reject();
        gguf_set_val_u32(file, "agention.trellis.state_bits", 15);
        gguf_set_val_u32(file, "prism.hadamard.block_size", 96); reject();
        gguf_set_val_u32(file, "prism.hadamard.block_size", 128);
        signs[0] = 0;
        gguf_set_arr_data(file, "prism.hadamard.sign_values", GGUF_TYPE_INT32, signs.data(), signs.size()); reject();
        signs[0] = -1;
        gguf_set_arr_data(file, "prism.hadamard.sign_values", GGUF_TYPE_INT32, signs.data(), signs.size());
        const auto written = gguf_write_to_file(file, path.string().c_str(), false);
        gguf_set_val_str(file, "prism.hadamard.sign_mode", "identity");
        require(gguf_write_to_file(file, identity_path.string().c_str(), false), "cannot write identity Gyro fixture");
        gguf_free(file); ggml_free(tensors);
        require(written, "cannot write Gyro rotation fixture");
        const MfqContainer model(path);
        const auto gate = MlxGgufWeight::load(model, "model.block.0.mlp.experts.gate.weight");
        const auto up = MlxGgufWeight::load(model, "model.block.0.mlp.experts.up.weight");
        const auto down = MlxGgufWeight::load(model, "model.block.0.mlp.experts.down.weight");
        const auto query = MlxGgufWeight::load(model, "model.block.0.attention.query.weight", true);
        require(query.has_hadamard() && !query.native_weight_ref() && !query.has_staged_planes(), "native projection bypassed Hadamard");
        compare(gate.dequantize(float32), original[0], 2e-6f, "Gyro Gate inverse rotation differs");
        compare(up.dequantize(float32), original[1], 2e-6f, "Gyro Up inverse rotation differs");
        compare(down.dequantize(float32), original[2], 2e-6f, "Gyro Down inverse rotation differs");
        compare(query.dequantize(float32), original[3], 2e-6f, "plain GGUF inverse rotation differs");
        const MfqContainer identity_model(identity_path);
        const auto identity_query = MlxGgufWeight::load(identity_model, "model.block.0.attention.query.weight");
        std::vector<float> query_signs;
        for (int col = 0; col < 256; ++col) query_signs.push_back(col % 3 == 0 ? -1.f : 1.f);
        const auto identity_reference = original[3] * array(query_signs.begin(), Shape{256});
        compare(identity_query.dequantize(float32), identity_reference, 2e-6f, "identity Gyro inverse rotation differs");
        {
            MlxKernelPreparation preparation(4096, 4096);
            const auto first = query.embedding(array({0}, int32), float32);
            const auto many = query.embedding(arange(8, int32), float32);
            const auto rotation_key = [](const array& output) {
                std::vector<array> pending{output};
                while (!pending.empty()) {
                    const auto value = pending.back(); pending.pop_back();
                    if (!value.has_primitive()) continue;
                    if (std::string_view(value.primitive().name()) == "CustomKernel") {
                        const auto key = mlx_kernel_preparation_key(value.primitive());
                        if (key.find("tinfer_gguf_hadamard128") != std::string::npos) return key;
                    }
                    for (const auto& input : value.inputs()) pending.push_back(input);
                }
                throw std::runtime_error("Gyro fused rotation was not exercised");
            };
            require(rotation_key(first) == rotation_key(many), "Gyro inverse rotation compiled per token count");
            preparation.add(first); preparation.add(many);
            require(preparation.size() > 0, "Gyro inverse rotation preparation was empty");
            preparation.finish();
            require(preparation.unsupported().empty(), "Gyro inverse rotation was not prepared at load time");
        }
        for (int rows : {1, 7, 8, 65}) {
            const auto input = reshape(sin(arange(rows * 256, float32) * .013f), {rows, 256});
            compare(query.matmul(input), matmul(input, transpose(original[3])), .003f, "rotated dense projection differs");
            compare(identity_query.matmul(input), matmul(input, transpose(identity_reference)), .003f, "identity rotated dense projection differs");
            const auto ids = reshape(remainder(arange(rows * 2, int32), array(4, int32)), {rows, 2});
            const auto hidden = gate.routed_swiglu(up, astype(input, float16), ids);
            compare(hidden, moe_swiglu_pair(gate.routed_matmul(astype(input, float16), ids, true),
                up.routed_matmul(astype(input, float16), ids, true)), .003f, "rotated Gate/Up did not reuse the right transform");
            const auto p = full({rows, 2}, .5f);
            const auto expected = moe_weighted_reduce(down.routed_matmul(hidden, ids, false), p);
            if (gguf_expert_prefill_available(input)) {
                const MlxGgufPrefillPlan plan(astype(input, float16), ids, 64,
                    gguf_prefill_tile_rows(gate, up, down, ids));
                compare(plan.down(down, plan.gate_up(gate, up), p), expected, .003f, "Gyro register prefill changed rotation order");
            }
            for (const int capacity : {1, 4}) {
                MlxGgufExpertCache cache(model, {"model.block.0"}, largest * capacity, 4);
                compare(cache.apply_ffn(0, astype(input, float16), ids, p), expected, .003f, "SSD Gyro rotation differs from resident");
                require(cache.resident_bytes() <= cache.limit_bytes(), "SSD Gyro exceeded its cache budget");
            }
        }
        std::filesystem::remove(path);
        std::filesystem::remove(identity_path);
        std::cout << "Gyro Hadamard dequantization/dense/routed/GateUp/SSD M=1,7,8,65 passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove(path);
        std::filesystem::remove(identity_path);
        std::cerr << error.what() << '\n'; return 1;
    }
}
