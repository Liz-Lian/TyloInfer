#include "mlx_gguf.h"
#include "mfq_mlx_utils_embedded.h"
#include "mlx_gguf_prefill.h"
#include "mlx_gguf_staged.h"
#include "mlx_gguf_rotation.h"
#include "native/mlx_native_gguf.h"

#include "mfq_container.h"
#include "gguf_embedded.h"
#include "mlx_weight_residency.h"
#include "mlx_indirect_weight_residency.h"
#include "mlx_buffer_view.h"
#include "mlx_platform.h"
#include "mlx_moe_ops.h"
#include "nlohmann/json.hpp"

#include <mlx/allocator.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace mfq::metal {
namespace {
using namespace mlx::core;
struct Format { const char* name; int id, block, bytes; };
constexpr Format formats[]{
    {"GGUF_Q4_0", 2, 32, 18}, {"GGUF_Q4_1", 3, 32, 20},
    {"GGUF_Q5_0", 6, 32, 22}, {"GGUF_Q5_1", 7, 32, 24},
    {"GGUF_Q8_0", 8, 32, 34}, {"GGUF_Q2_K", 10, 256, 84},
    {"GGUF_Q3_K", 11, 256, 110}, {"GGUF_Q4_K", 12, 256, 144},
    {"GGUF_Q5_K", 13, 256, 176}, {"GGUF_Q6_K", 14, 256, 210},
    {"GGUF_Q8_K", 15, 256, 292}, {"GGUF_IQ2_XXS", 16, 256, 66},
    {"GGUF_IQ2_XS", 17, 256, 74}, {"GGUF_IQ3_XXS", 18, 256, 98},
    {"GGUF_IQ4_NL", 20, 32, 18}, {"GGUF_IQ3_S", 21, 256, 110},
    {"GGUF_IQ2_S", 22, 256, 82}, {"GGUF_IQ4_XS", 23, 256, 136},
    {"GGUF_MXFP4", 39, 32, 17}, {"GGUF_Q2_0", 42, 64, 18},
    {"GGUF_PTQ1_0", 143, 128, 28},
    {"GGUF_TQ2_T", 144, 128, 34}, {"GGUF_TQK6", 145, 128, 26},
    {"GGUF_TQK7", 146, 128, 30}, {"GGUF_TQP5", 147, 128, 42},
    {"GGUF_TQP6", 148, 128, 50}, {"GGUF_TQP7", 149, 128, 58},
    {"GGUF_TQK5", 150, 128, 22}, {"GGUF_TQK4", 151, 128, 18},
};

constexpr const char* project_source = R"METAL(
uint lane = thread_index_in_simdgroup;
uint local = lane % L;
uint row = thread_position_in_grid.x / L;
uint route = thread_position_in_grid.y;
uint K = params[0], N = params[1], M = params[2], stride = params[3];
if (row >= N) return;
int expert = GROUPED ? int(route) : ROUTED ? ids[route] : 0;
float acc[GROUPED ? 8 : 6] = {};
uint count = ROUTED ? 1u : M;
auto source_row = [&](uint m) { return GROUPED ? m * params[4] + route
    : ROUTED ? (params[6] ? route / params[5] : route) : m; };
if (expert >= 0 && uint(expert) < params[4]) {
    auto matrix = gguf_matrix<INDIRECT>(packed, uint(expert), N, stride);
    auto weight = matrix ? matrix + ulong(row) * stride : nullptr;
    for (uint k = local * V; weight && k < K; k += L * V) {
        if constexpr (F == 143) {
            for (uint m = 0u; m < count; ++m) {
                const uint token = source_row(m);
                acc[m] += gguf_ptq_dot128<8>(weight, x + ulong(token) * K, k - local * V, local);
            }
        } else if constexpr (F == 8) {
            auto block = reinterpret_cast<device const block_q8_0*>(weight) + k / 32u;
            const float scale = float(block->d);
            for (uint m = 0u; m < count; ++m) {
                const uint token = source_row(m);
                auto source = x + ulong(token) * K + k;
                float sum = 0.f;
                for (uint j = 0u; j < 32u; j += 4u) {
                    const float4 codes(block->qs[j], block->qs[j + 1u], block->qs[j + 2u], block->qs[j + 3u]);
                    sum += dot(codes, float4(source[j], source[j + 1u], source[j + 2u], source[j + 3u]));
                }
                acc[m] += scale * sum;
            }
        } else if constexpr (F == 7) {
            auto block = reinterpret_cast<device const block_q5_1*>(weight) + k / 32u;
            uint high = 0u;
            for (uint j = 0u; j < 4u; ++j) high |= uint(block->qh[j]) << (8u * j);
            for (uint m = 0u; m < count; ++m) {
                const uint token = source_row(m);
                auto source = x + ulong(token) * K + k;
                float dot_value = 0.f, sum = 0.f;
                for (uint j = 0u; j < 32u; j += 4u) {
                    float4 codes;
                    for (uint item = 0u; item < 4u; ++item) {
                        const uint c = j + item;
                        codes[item] = float(((block->qs[c % 16u] >> (4u * (c / 16u))) & 15u) |
                            (((high >> c) & 1u) << 4u));
                    }
                    const float4 values(source[j], source[j + 1u], source[j + 2u], source[j + 3u]);
                    dot_value += dot(codes, values);
                    sum += values.x + values.y + values.z + values.w;
                }
                acc[m] += float(block->d) * dot_value + float(block->m) * sum;
            }
        } else if constexpr (F == 12 || F == 13) {
            for (uint m = 0u; m < count; ++m) {
                const uint token = source_row(m);
                acc[m] += gguf_dot32<F>(weight, x + ulong(token) * K, k);
            }
        } else if constexpr (F == 14) {
            for (uint m = 0u; m < count; ++m) {
                const uint token = source_row(m);
                acc[m] += gguf_q6_dot16(weight, x + ulong(token) * K, k);
            }
        } else if constexpr (V == 1) {
            float w = gguf_value<F>(weight, k);
            for (uint m = 0u; m < count; ++m) {
                uint token = source_row(m);
                acc[m] += w * float(x[ulong(token) * K + k]);
            }
        } else {
            if constexpr (F == 20 || F == 23) {
                if (count == 1u) {
                    const uint token = source_row(0u);
                    acc[0] += gguf_iq4_dot32<F>(weight, x + ulong(token) * K, k);
                    continue;
                }
            }
            if constexpr (F == 16 || F == 17 || F == 18 || F == 21 || F == 22) {
                if (count == 1u) {
                    const uint token = source_row(0u);
                    acc[0] += gguf_iq_dot16<F>(weight, x + ulong(token) * K, k, GgufConstantLookup<F>{});
                    continue;
                }
            }
            for (uint block = 0u; block < V / 4u; ++block) {
                float4 weights = gguf_values4<F>(weight, k + block * 4u);
                for (uint m = 0u; m < count; ++m) {
                    uint token = source_row(m);
                    auto source = x + ulong(token) * K + k + block * 4u;
                    acc[m] += dot(weights, float4(source[0], source[1], source[2], source[3]));
                }
            }
        }
    }
}
for (uint m = 0u; m < count; ++m) {
    float value = acc[m];
    if constexpr (L == 32) value = simd_sum(value);
    else for (ushort shift = L / 2; shift > 0; shift /= 2) value += simd_shuffle_xor(value, shift);
    if (local == 0u) y[(GROUPED ? ulong(m) * params[4] + route : ROUTED ? ulong(route) : ulong(m)) * N + row] = T(value);
}
)METAL";

constexpr const char* embedding_source = R"METAL(
uint col = thread_position_in_grid.x * 4u, token = thread_position_in_grid.y;
uint K = params[0], ROWS = params[1], stride = params[3];
if (col >= K) return;
int row = ids[token];
float4 values = row >= 0 && uint(row) < ROWS ? gguf_values4<F>(packed + ulong(row) * stride, col) : float4(0.f);
for (uint j = 0u; j < 4u; ++j) y[ulong(token) * K + col + j] = T(values[j]);
)METAL";

constexpr const char* prefill_source = R"METAL(
threadgroup half a_tile[16 * 32];
threadgroup half b_tile[32 * 32];
threadgroup float result[16 * 32];
uint t = thread_index_in_threadgroup;
uint sg = simdgroup_index_in_threadgroup;
uint K = params[0], N = params[1], M = params[2], stride = params[3];
uint base_n = threadgroup_position_in_grid.x * 32u;
uint base_m = threadgroup_position_in_grid.y * 16u;
simdgroup_float8x8 c0(0.f), c1(0.f);
for (uint base_k = 0u; base_k < K; base_k += 32u) {
    for (uint i = t; i < 16u * 32u; i += 128u) {
        uint m = base_m + i / 32u, k = base_k + i % 32u;
        a_tile[i] = m < M && k < K ? half(x[ulong(m) * K + k]) : half(0);
    }
    for (uint i = t; i < 32u * 32u; i += 128u) {
        uint k = base_k + i / 32u, n = base_n + i % 32u;
        b_tile[i] = n < N && k < K ? half(gguf_value<F>(packed + ulong(n) * stride, k)) : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0u; k < 32u; k += 8u) {
        simdgroup_half8x8 a, b0, b1;
        simdgroup_load(a, a_tile + (sg / 2u) * 8u * 32u + k, 32);
        simdgroup_load(b0, b_tile + k * 32u + (sg % 2u) * 16u, 32);
        simdgroup_load(b1, b_tile + k * 32u + (sg % 2u) * 16u + 8u, 32);
        simdgroup_multiply_accumulate(c0, a, b0, c0);
        simdgroup_multiply_accumulate(c1, a, b1, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(c0, result + (sg / 2u) * 8u * 32u + (sg % 2u) * 16u, 32);
simdgroup_store(c1, result + (sg / 2u) * 8u * 32u + (sg % 2u) * 16u + 8u, 32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = t; i < 16u * 32u; i += 128u) {
    uint m = base_m + i / 32u, n = base_n + i % 32u;
    if (m < M && n < N) y[ulong(m) * N + n] = T(result[i]);
}
)METAL";

const std::string& indirect_decode_header() {
    static const auto source = std::string(kGgufDecode) + R"METAL(
template<bool INDIRECT, typename P>
device const uchar* gguf_matrix(P packed, uint expert, uint rows, uint stride) {
    if constexpr (INDIRECT) return reinterpret_cast<device const uchar*>(packed[expert]);
    else return reinterpret_cast<device const uchar*>(packed) + ulong(expert) * rows * stride;
}
template<int F, int BM, typename T, class InputRow, class OutputRow>
void gguf_sparse_rows(device const uchar* matrix, device const T* x, device T* y,
    uint K, uint N, uint stride, uint rows, uint first, uint base_n,
    uint lane, uint group, InputRow input_row, OutputRow output_row) {
    const uint local = lane % 8u, column = lane / 8u;
    for (uint j = 0u; j < 8u; ++j) {
        uint n = base_n + group * 32u + j * 4u + column;
        if (n >= N) continue;
        auto weight = matrix + ulong(n) * stride;
        for (uint r = first; r < min(rows, first + uint(BM)); ++r) {
            auto source = x + ulong(input_row(r)) * K;
            float sum = 0.f;
            for (uint k = local * 4u; k < K; k += 32u)
                sum += dot(float4(half4(gguf_values4<F>(weight, k))),
                    float4(half4(half(source[k]), half(source[k + 1u]), half(source[k + 2u]), half(source[k + 3u]))));
            sum += simd_shuffle_xor(sum, 4u);
            sum += simd_shuffle_xor(sum, 2u);
            sum += simd_shuffle_xor(sum, 1u);
            if (!local) y[ulong(output_row(r)) * N + n] = T(sum);
        }
    }
}
)METAL";
    return source;
}
class GgufProjection final : public Primitive, public MlxPreparableKernel {
public:
    GgufProjection(Stream stream, std::array<std::uint32_t, 7> params, int format, Dtype dtype,
        int values, int lanes, bool routed, bool indirect, const array& packed, bool grouped = false)
        : Primitive(stream), params_(params), format_(format), dtype_(dtype), values_(values), lanes_(lanes),
          routed_(routed), indirect_(indirect), grouped_(grouped), key_("tinfer_gguf_project_fast_" + std::to_string(format) + "_" +
              type_to_name(dtype) + "_" + std::to_string(values) + "_" + std::to_string(lanes) +
              (routed ? "_routed" : "") + (indirect ? "_indirect" : "") + (grouped ? "_grouped" : "")) {
        if (indirect) {
            auto& registry = indirect_weight_registry();
            std::lock_guard lock(registry.mutex);
            const auto found = registry.entries.find(packed.data_shared_ptr().get());
            if (found != registry.entries.end()) residency_ = found->second.residency.lock();
        }
    }
    const char* name() const override { return "GgufProjection"; }
    std::string preparation_key() const override { return key_; }
    void prepare_gpu() override { (void)pipeline(); }
    MTL::ComputePipelineState* pipeline() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(key_, options, [this] {
            const std::string type = dtype_ == float16 ? "half" : dtype_ == bfloat16 ? "bfloat16_t" : "float";
            auto source = std::string(detail::kMlxUtilsSource) + indirect_decode_header();
            source += "\nusing T = " + type + ";\n";
            source += "constant constexpr int F=" + std::to_string(format_) + ", V=" + std::to_string(values_) +
                ", L=" + std::to_string(lanes_) + ";\n";
            source += "constant constexpr bool ROUTED=" + std::to_string(routed_) +
                ", INDIRECT=" + std::to_string(indirect_) + ", GROUPED=" + std::to_string(grouped_) + ";\n";
            source += "kernel void " + key_ + "(device const " + (indirect_ ? "ulong" : "uchar") + R"METAL(* packed [[buffer(0)]],
    device const T* x [[buffer(1)]], device const int* ids [[buffer(2)]],
    device T* y [[buffer(3)]], constant uint* params [[buffer(4)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
)METAL";
            source += project_source;
            source += "}\n";
            return source;
        });
        return device.get_kernel(key_, library);
    }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("GGUF projection requires Metal");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& output = outputs.front();
        output.set_data(allocator::malloc(output.nbytes()));
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        if (residency_) {
            auto* command = encoder.get_command_buffer();
            residency_->use_command_buffer(command);
            command->addCompletedHandler([owner = residency_](MTL::CommandBuffer*) {});
        }
        encoder.set_compute_pipeline_state(pipeline());
        encoder.set_input_array(inputs[0], 0); encoder.set_input_array(inputs[1], 1);
        encoder.set_input_array(inputs[routed_ ? 2 : 0], 2);
        encoder.set_output_array(output, 3); encoder.set_bytes(params_, 4);
        const auto rows = 128u / lanes_;
        encoder.dispatch_threads(MTL::Size(((params_[1] + rows - 1) / rows) * 128,
            grouped_ ? params_[4] : routed_ ? params_[2] * params_[5] : 1, 1), MTL::Size(128, 1, 1));
    }
    bool is_equivalent(const Primitive& other) const override {
        const auto& value = static_cast<const GgufProjection&>(other);
        return key_ == value.key_ && params_ == value.params_;
    }
private:
    std::array<std::uint32_t, 7> params_;
    int format_; Dtype dtype_; int values_, lanes_; bool routed_, indirect_, grouped_;
    std::string key_;
    std::shared_ptr<MlxWeightResidency> residency_;
};
constexpr const char* swiglu_source = R"METAL(
uint lane = thread_index_in_simdgroup;
uint local = lane % L;
uint row = thread_position_in_grid.x / L, route = thread_position_in_grid.y;
uint K = params[0], N = params[1], gate_stride = params[3], up_stride = params[7];
threadgroup uint gate_table[GgufConstantLookup<FG>::storage_size];
threadgroup uint up_table[FG == FU ? 1u : GgufConstantLookup<FU>::storage_size];
gguf_load_lookup<FG>(gate_table, thread_index_in_threadgroup, 128u);
if constexpr (FG != FU) gguf_load_lookup<FU>(up_table, thread_index_in_threadgroup, 128u);
if constexpr (GgufConstantLookup<FG>::words || GgufConstantLookup<FU>::words)
    threadgroup_barrier(mem_flags::mem_threadgroup);
const GgufThreadgroupLookup<FG> gate_lookup{gate_table};
const GgufThreadgroupLookup<FU> up_lookup{FG == FU ? gate_table : up_table};
if (row >= N) return;
int expert = ROUTED ? ids[route] : 0;
float g = 0.f, u = 0.f;
if (expert >= 0 && uint(expert) < params[4]) {
    auto wg = gguf_matrix<IG>(gate, uint(expert), N, gate_stride);
    auto wu = gguf_matrix<IU>(up, uint(expert), N, up_stride);
    if (wg && wu) {
        wg += ulong(row) * gate_stride; wu += ulong(row) * up_stride;
        auto input = x + ulong(ROUTED ? route / params[5] : route) * K;
        for (uint k = local * V; k < K; k += L * V) {
            if constexpr (FG == 143 && FU == 143) {
                g += gguf_ptq_dot128<8>(wg, input, k - local * V, local);
                u += gguf_ptq_dot128<8>(wu, input, k - local * V, local);
            } else if constexpr ((FG == 12 || FG == 13) && (FU == 12 || FU == 13)) {
                g += gguf_dot32<FG>(wg, input, k);
                u += gguf_dot32<FU>(wu, input, k);
            } else if constexpr ((FG == 20 || FG == 23) && (FU == 20 || FU == 23)) {
                g += gguf_iq4_dot32<FG>(wg, input, k);
                u += gguf_iq4_dot32<FU>(wu, input, k);
            } else if constexpr ((FG == 16 || FG == 17 || FG == 18 || FG == 21 || FG == 22)
                && (FU == 16 || FU == 17 || FU == 18 || FU == 21 || FU == 22)) {
                g += gguf_iq_dot16<FG>(wg, input, k, gate_lookup);
                u += gguf_iq_dot16<FU>(wu, input, k, up_lookup);
            } else {
                for (uint offset = 0u; offset < V; offset += 4u) {
                    float4 gate_value = gguf_values4<FG>(wg, k + offset, gate_lookup);
                    float4 up_value = gguf_values4<FU>(wu, k + offset, up_lookup);
                    const float4 value(input[k + offset], input[k + offset + 1u], input[k + offset + 2u], input[k + offset + 3u]);
                    g += dot(gate_value, value);
                    u += dot(up_value, value);
                }
            }
        }
    }
}
if constexpr (L == 32) { g = simd_sum(g); u = simd_sum(u); }
else for (ushort shift = L / 2; shift > 0; shift /= 2) {
    g += simd_shuffle_xor(g, shift); u += simd_shuffle_xor(u, shift);
}
g = float(T(g)); u = float(T(u));
if (params[8]) { float limit = as_type<float>(params[8]); g = min(g, limit); u = clamp(u, -limit, limit); }
if (!local) y[ulong(route) * N + row] = T((g / (1.f + exp(-g))) * u);
)METAL";

class GgufSwiGLU final : public Primitive, public MlxPreparableKernel {
public:
    GgufSwiGLU(Stream stream, std::array<std::uint32_t, 9> params, int gate_format, int up_format,
        Dtype dtype, int lanes, int values, bool indirect_gate, bool indirect_up, const array& gate, const array& up,
        bool routed = true)
        : Primitive(stream), params_(params), gate_format_(gate_format), up_format_(up_format), dtype_(dtype),
          lanes_(lanes), values_(values), indirect_gate_(indirect_gate), indirect_up_(indirect_up), routed_(routed),
          key_("tinfer_gguf_swiglu_fast_" + std::to_string(gate_format) + "_" + std::to_string(up_format) + "_" +
              type_to_name(dtype) + "_" + std::to_string(lanes) + "_" + std::to_string(values) + "_" +
              std::to_string(indirect_gate) + std::to_string(indirect_up) + (routed ? "_routed" : "_dense")) {
        if (indirect_gate || indirect_up) {
            auto& registry = indirect_weight_registry();
            std::lock_guard lock(registry.mutex);
            for (const auto* weight : {indirect_gate ? &gate : nullptr, indirect_up ? &up : nullptr}) {
                if (!weight) continue;
                const auto found = registry.entries.find(weight->data_shared_ptr().get());
                if (found != registry.entries.end()) {
                    if (auto owner = found->second.residency.lock(); owner &&
                        std::find(owners_.begin(), owners_.end(), owner) == owners_.end()) owners_.push_back(std::move(owner));
                }
            }
        }
    }
    const char* name() const override { return "GgufSwiGLU"; }
    std::string preparation_key() const override { return key_; }
    void prepare_gpu() override { (void)pipeline(); }
    MTL::ComputePipelineState* pipeline() {
        auto& device = kernel_device(stream().device);
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        auto* library = device.get_library(key_, options, [this] {
            const std::string type = dtype_ == float16 ? "half" : dtype_ == bfloat16 ? "bfloat16_t" : "float";
            auto source = std::string(detail::kMlxUtilsSource) + indirect_decode_header();
            source += "\nusing T = " + type + ";\n";
            source += "constant constexpr int FG=" + std::to_string(gate_format_) + ", FU=" + std::to_string(up_format_) +
                ", L=" + std::to_string(lanes_) + ", V=" + std::to_string(values_) + ";\n";
            source += "constant constexpr bool IG=" + std::to_string(indirect_gate_) +
                ", IU=" + std::to_string(indirect_up_) + ", ROUTED=" + std::to_string(routed_) + ";\n";
            source += "kernel void " + key_ + "(device const " + (indirect_gate_ ? "ulong" : "uchar") +
                "* gate [[buffer(0)]], device const " + (indirect_up_ ? "ulong" : "uchar") + R"METAL(* up [[buffer(1)]],
    device const T* x [[buffer(2)]], device const int* ids [[buffer(3)]], device T* y [[buffer(4)]],
    constant uint* params [[buffer(5)]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
)METAL";
            source += swiglu_source;
            source += "}\n";
            return source;
        });
        return device.get_kernel(key_, library);
    }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("GGUF Gate/Up requires Metal");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& output = outputs.front();
        output.set_data(allocator::malloc(output.nbytes()));
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        if (!owners_.empty()) {
            auto* command = encoder.get_command_buffer();
            for (const auto& owner : owners_) owner->use_command_buffer(command);
            command->addCompletedHandler([owners = owners_](MTL::CommandBuffer*) {});
        }
        encoder.set_compute_pipeline_state(pipeline());
        for (int index = 0; index < 4; ++index) encoder.set_input_array(inputs[index], index);
        encoder.set_output_array(output, 4);
        encoder.set_bytes(params_, 5);
        const auto rows = 128u / lanes_;
        encoder.dispatch_threads(MTL::Size(((params_[1] + rows - 1) / rows) * 128, params_[2] * params_[5], 1),
            MTL::Size(128, 1, 1));
    }
    bool is_equivalent(const Primitive& other) const override {
        const auto& value = static_cast<const GgufSwiGLU&>(other);
        return key_ == value.key_ && params_ == value.params_;
    }
private:
    std::array<std::uint32_t, 9> params_;
    int gate_format_, up_format_; Dtype dtype_; int lanes_, values_;
    bool indirect_gate_, indirect_up_, routed_;
    std::string key_;
    std::vector<std::shared_ptr<MlxWeightResidency>> owners_;
};
const fast::CustomKernelFunction& embedding_kernel() {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_embedding",
        {"packed", "ids", "params"}, {"y"}, embedding_source, kGgufDecode);
    return kernel;
}
const fast::CustomKernelFunction& dequantize_kernel() {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_dequantize_rows32",
        {"packed", "params"}, {"y"}, R"METAL(
uint block = thread_position_in_grid.x;
uint K = params[0], ROWS = params[1], stride = params[3];
uint blocks_per_row = K / 32u;
uint row = block / blocks_per_row, col = (block % blocks_per_row) * 32u;
if (row >= ROWS) return;
auto source = packed + ulong(row) * stride;
auto destination = y + ulong(row) * K + col;
for (uint offset = 0u; offset < 32u; offset += 4u) {
    const float4 values = gguf_values4<F>(source, col + offset);
    for (uint j = 0u; j < 4u; ++j) destination[offset + j] = T(values[j]);
}
)METAL", kGgufDecode);
    return kernel;
}
const fast::CustomKernelFunction& prefill_kernel() {
    static const auto kernel = fast::metal_kernel("tinfer_gguf_prefill",
        {"packed", "x", "params"}, {"y"}, prefill_source,
        std::string("#include <metal_simdgroup_matrix>\n") + kGgufDecode);
    return kernel;
}
const fast::CustomKernelFunction& routed_prefill_kernel() {
    static const auto kernel = indirect_weight_metal_kernel("tinfer_gguf_routed_prefill",
        {"packed", "x", "order", "bounds", "tile_offsets", "params"}, {"y"}, R"METAL(
uint block = threadgroup_position_in_grid.y;
if (block >= uint(tile_offsets[params[4]])) return;
uint low = 0u, high = params[4];
while (low < high) {
    uint mid = low + (high - low) / 2u;
    if (uint(tile_offsets[mid + 1u]) <= block) low = mid + 1u; else high = mid;
}
uint expert = low;
uint first = uint(bounds[expert * 2u]) + (block - uint(tile_offsets[expert])) * 16u;
uint end = uint(bounds[expert * 2u + 1u]);
uint K = params[0], N = params[1], stride = params[3];
auto matrix = gguf_matrix<INDIRECT>(packed, expert, N, stride);
if (!matrix) return;
threadgroup half a_tile[16 * 32];
threadgroup half b_tile[32 * 32];
threadgroup float result[16 * 32];
uint t = thread_index_in_threadgroup, sg = simdgroup_index_in_threadgroup;
uint base_n = threadgroup_position_in_grid.x * 32u;
simdgroup_float8x8 c0(0.f), c1(0.f);
for (uint base_k = 0u; base_k < K; base_k += 32u) {
    for (uint i = t; i < 16u * 32u; i += 128u) {
        uint route_index = first + i / 32u, k = base_k + i % 32u;
        uint route = route_index < end ? uint(order[route_index]) : 0u;
        uint row = params[6] ? route / params[5] : route;
        a_tile[i] = route_index < end && k < K ? half(x[ulong(row) * K + k]) : half(0);
    }
    for (uint i = t; i < 32u * 32u; i += 128u) {
        uint k = base_k + i / 32u, n = base_n + i % 32u;
        b_tile[i] = n < N && k < K ? half(gguf_value<F>(matrix + ulong(n) * stride, k)) : half(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0u; k < 32u; k += 8u) {
        simdgroup_half8x8 a, b0, b1;
        simdgroup_load(a, a_tile + (sg / 2u) * 8u * 32u + k, 32);
        simdgroup_load(b0, b_tile + k * 32u + (sg % 2u) * 16u, 32);
        simdgroup_load(b1, b_tile + k * 32u + (sg % 2u) * 16u + 8u, 32);
        simdgroup_multiply_accumulate(c0, a, b0, c0);
        simdgroup_multiply_accumulate(c1, a, b1, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(c0, result + (sg / 2u) * 8u * 32u + (sg % 2u) * 16u, 32);
simdgroup_store(c1, result + (sg / 2u) * 8u * 32u + (sg % 2u) * 16u + 8u, 32);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint i = t; i < 16u * 32u; i += 128u) {
    uint route_index = first + i / 32u, n = base_n + i % 32u;
    if (route_index < end && n < N) y[ulong(order[route_index]) * N + n] = T(result[i]);
}
)METAL", std::string("#include <metal_simdgroup_matrix>\n") + indirect_decode_header());
    return kernel;
}
const fast::CustomKernelFunction& nax_prefill_kernel(bool routed) {
    static const auto ordinary = fast::metal_kernel("tinfer_gguf_prefill_nax",
        {"packed", "x", "params"}, {"y"}, R"METAL(
uint K = params[0], N = params[1], M = params[2], stride = params[3];
uint n0 = threadgroup_position_in_grid.x * 128u + simdgroup_index_in_threadgroup * 32u;
uint m0 = threadgroup_position_in_grid.y * BM;
auto identity = [](uint row) { return row; };
gguf_matmul<F, BM>(packed, x, y, K, N, stride, M, m0, n0, identity, identity);
)METAL", std::string(kGgufDecode) + kGgufNax, true, false);
    static const auto experts = indirect_weight_metal_kernel("tinfer_gguf_routed_prefill_nax",
        {"packed", "x", "order", "bounds", "tile_offsets", "params"}, {"y"}, R"METAL(
uint block = threadgroup_position_in_grid.y;
if (block >= uint(tile_offsets[params[4]])) return;
uint low = 0u, high = params[4];
while (low < high) {
    uint mid = low + (high - low) / 2u;
    if (uint(tile_offsets[mid + 1u]) <= block) low = mid + 1u; else high = mid;
}
uint expert = low;
uint begin = uint(bounds[expert * 2u]);
uint rows = uint(bounds[expert * 2u + 1u]) - begin;
uint m0 = (block - uint(tile_offsets[expert])) * BM;
uint n0 = threadgroup_position_in_grid.x * 128u + simdgroup_index_in_threadgroup * 32u;
uint K = params[0], N = params[1], stride = params[3];
auto matrix = gguf_matrix<INDIRECT>(packed, expert, N, stride);
if (!matrix) return;
auto input_row = [&](uint row) { uint route = uint(order[begin + row]); return params[6] ? route / params[5] : route; };
auto output_row = [&](uint row) { return uint(order[begin + row]); };
if (SPARSE && rows <= 2u) {
    gguf_sparse_rows<F, BM>(matrix, x, y, K, N, stride, rows, m0,
        threadgroup_position_in_grid.x * 128u, thread_index_in_simdgroup,
        simdgroup_index_in_threadgroup, input_row, output_row);
    return;
}
gguf_matmul<F, BM>(matrix, x, y, K, N, stride, rows, m0, n0, input_row, output_row);
)METAL", indirect_decode_header() + kGgufNax, true, false);
    return routed ? experts : ordinary;
}
bool use_nax() {
    return !mlx_metal_nax_disabled() && mlx_apple_chip_starts_with("Apple M5");
}
int checked_int(std::size_t value) {
    if (value > INT_MAX) throw std::overflow_error("GGUF tensor dimension exceeds runtime limits");
    return static_cast<int>(value);
}
array parameters(int k, int n, int m, int stride, int experts, int routes = 1, bool shared = false) {
    return array({uint32_t(k), uint32_t(n), uint32_t(m), uint32_t(stride),
                  uint32_t(experts), uint32_t(routes), uint32_t(shared)});
}
void check_input(const array& input, int width) {
    if (input.ndim() == 0 || input.shape(-1) != width ||
        (input.dtype() != float16 && input.dtype() != bfloat16 && input.dtype() != float32))
        throw std::invalid_argument("GGUF projection requires matching floating-point input");
}
int projection_values(int format) {
    if (format == 39) return 32;
    if (format >= 144 && format <= 151) return 16;
    if (format == 8 || format == 7 || format == 12 || format == 13 || format == 20 || format == 23) return 32;
    if (format == 14 || format == 16 || format == 17 || format == 18 || format == 21 || format == 22 || format == 42 || format == 143) return 16;
    return 1;
}
int projection_lanes(int format, int width, int outputs) {
    if (format == 39) return width > outputs ? 8 : 4;
    if (format >= 147 && format <= 149) return 8;
    if (format >= 144 && format <= 151) return width > outputs ? 8 : 4;
    if (format == 143) return 8;
    if (format == 7 || format == 12 || format == 13 || format == 14 || format == 20 || format == 23)
        return width > outputs ? 8 : 4;
    if (format == 16 || format == 17 || format == 18 || format == 21 || format == 22 || format == 42 || format == 143)
        return width <= 1024 ? 4 : 8;
    return format == 8 ? (width <= 512 ? 4 : 8) : 32;
}
std::pair<int, int> swiglu_geometry(int gate, int up) {
    if (gate == 39 && up == 39) return {4, 32};
    if (gate == 143 && up == 143) return {8, 16};
    const auto iq = [](int format) { return format == 16 || format == 17 || format == 18 || format == 21 || format == 22; };
    const auto kquant = [](int format) { return format == 12 || format == 13; };
    if (kquant(gate) && kquant(up)) return {4, 32};
    if ((gate == 20 || gate == 23) && (up == 20 || up == 23)) return {4, 32};
    if (iq(gate) && iq(up)) return {8, 16};
    return {32, 4};
}
} // namespace

bool is_gguf_dtype(std::string_view dtype) noexcept { return dtype.substr(0, 5) == "GGUF_"; }
std::size_t gguf_packed_row_bytes(std::string_view dtype, int width) {
    const auto format = std::find_if(std::begin(formats), std::end(formats),
        [&](const Format& f) { return dtype == f.name; });
    if (format == std::end(formats) || width <= 0 || width % format->block)
        throw std::invalid_argument("unsupported GGUF row format or block alignment");
    return std::size_t(width / format->block) * format->bytes;
}
MlxGgufRoutePlan gguf_route_plan(const array& ids, int experts, bool wide_mixed) {
    if (wide_mixed && use_nax() && experts > 0 && ids.size() / experts >= 64)
        return mlx_expert_route_tiles(ids, experts, 64);
    return mlx_expert_route_tiles(ids, experts,
        use_nax() && experts > 0 && ids.size() / experts >= 32 ? 32 : 16);
}
MlxGgufWeight::MlxGgufWeight(array bytes, Shape shape, int format, int row_bytes, bool indirect)
    : bytes_(std::move(bytes)), shape_(std::move(shape)), format_(format), row_bytes_(row_bytes), indirect_(indirect) {}

void MlxGgufWeight::prepare_grouped() {
    if (native_ || mixed_layout_ || rotation_ || !use_nax()) return;
    MlxGgufStagedPrefill::prepare(*this);
    if (planes_) {
        MlxWeightResidency::track(bytes_);
        for (const auto& plane : *planes_) MlxWeightResidency::track(plane);
    }
}

MlxGgufWeight MlxGgufWeight::allocate_packed(std::string_view dtype, Shape shape, std::size_t bytes) {
    const auto format = std::find_if(std::begin(formats), std::end(formats),
        [&](const Format& f) { return dtype == f.name; });
    if (format == std::end(formats)) throw std::invalid_argument("unsupported native GGUF format: " + std::string(dtype));
    if ((shape.size() != 2 && shape.size() != 3) ||
        std::any_of(shape.begin(), shape.end(), [](int d) { return d <= 0; }) || shape.back() % format->block)
        throw std::invalid_argument("invalid GGUF matrix shape or block alignment");
    const auto row_bytes = checked_int(std::size_t(shape.back() / format->block) * format->bytes);
    const auto rows = std::size_t(shape[shape.size() - 2]) * (shape.size() == 3 ? shape[0] : 1);
    if (rows > SIZE_MAX / row_bytes || rows * row_bytes != bytes)
        throw std::invalid_argument("GGUF packed tensor length mismatch");
    auto packed = array(allocator::malloc(bytes), Shape{checked_int(rows), row_bytes}, uint8);
    return MlxGgufWeight(std::move(packed), std::move(shape), format->id, row_bytes);
}
MlxGgufWeight MlxGgufWeight::from_bytes(std::string_view dtype,
    std::span<const std::uint8_t> bytes, Shape shape) {
    auto result = allocate_packed(dtype, std::move(shape), bytes.size());
    std::memcpy(result.bytes_.data<std::uint8_t>(), bytes.data(), bytes.size());
    if (use_nax() && result.expert_count() <= 512)
        MlxGgufStagedPrefill::prepare(result);
    MlxWeightResidency::track(result.bytes_);
    if (result.planes_) for (const auto& plane : *result.planes_) MlxWeightResidency::track(plane);
    return result;
}
MlxGgufWeight MlxGgufWeight::from_experts(std::string_view dtype, Shape shape,
    const std::vector<std::int32_t>& ids, const std::vector<array>& packed,
    std::shared_ptr<MlxWeightResidency> residency) {
    if (shape.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || ids.empty() || ids.size() != packed.size())
        throw std::invalid_argument("invalid GGUF resident expert table");
    const auto stride = gguf_packed_row_bytes(dtype, shape[2]);
    std::vector<std::uint64_t> addresses(shape[0], 0);
    std::size_t allocated = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
        const auto& value = packed[i];
        if (ids[i] < 0 || ids[i] >= shape[0] || addresses[ids[i]] || value.dtype() != uint8 ||
            value.nbytes() != std::size_t(shape[1]) * stride || !value.flags().row_contiguous)
            throw std::invalid_argument("invalid GGUF resident expert allocation");
        const auto* buffer = static_cast<const MTL::Buffer*>(value.buffer().ptr());
        addresses[ids[i]] = buffer->gpuAddress() + value.offset();
        allocated += buffer->allocatedSize();
    }
    struct Lifetime {
        std::vector<array> experts;
        array table;
        std::shared_ptr<MlxWeightResidency> residency;
    };
    auto lifetime = std::make_shared<Lifetime>(Lifetime{packed,
        array(addresses.begin(), Shape{shape[0]}, uint64),
        residency ? std::move(residency) : MlxWeightResidency::retain_arrays(packed, allocated, true)});
    auto table = mlx_buffer_view(lifetime->table.buffer(), lifetime->table.shape(), uint64, lifetime);
    register_indirect_weight_pool(table, lifetime->residency);
    const auto format = std::find_if(std::begin(formats), std::end(formats),
        [&](const Format& f) { return dtype == f.name; });
    return MlxGgufWeight(std::move(table), std::move(shape), format->id, checked_int(stride), true);
}
bool MlxGgufWeight::supports_native(const MfqContainer& model, const std::string& name) {
    const auto rotations = model.header().extra_json.find("gguf_hadamard");
    if (rotations != model.header().extra_json.end() && nlohmann::json::parse(rotations->second).contains(name)) return false;
    const auto& dtype = model.record(name).dtype;
    const auto format = std::find_if(std::begin(formats), std::end(formats), [&](const Format& f) { return dtype == f.name; });
    if (format == std::end(formats)) return false;
    const auto shape = model.tensor_shape(name);
    if ((shape.size() != 2 && shape.size() != 3) ||
        std::any_of(shape.begin(), shape.end(), [](auto d) { return d < 1 || d > INT_MAX; })) return false;
    return mlx_native_gguf_supported(format->id, int(shape[shape.size() - 2]), int(shape.back()));
}

MlxGgufWeight MlxGgufWeight::load(const MfqContainer& model, const std::string& name, bool native_projection) {
    Shape shape;
    for (const auto d : model.tensor_shape(name)) {
        if (d <= 0 || d > INT_MAX) throw std::invalid_argument("invalid GGUF tensor dimension");
        shape.push_back(static_cast<int>(d));
    }
    const auto& record = model.record(name);
    if (record.dtype == "GGUF_MIXED") {
        if (shape.size() != 3) throw std::invalid_argument("mixed GGUF weights require an expert bank");
        const auto layouts = nlohmann::json::parse(model.header().extra_json.at("gguf_mixed_experts"));
        const auto& entries = layouts.at(name);
        if (!entries.is_array() || entries.size() != std::size_t(shape[0]))
            throw std::invalid_argument("mixed GGUF expert layout length mismatch");
        std::vector<std::int32_t> ids;
        std::vector<std::string> dtypes;
        std::vector<array> storage;
        std::size_t next = 0;
        for (const auto& entry : entries) {
            const auto dtype = entry.at("dtype").get<std::string>();
            const auto offset = entry.at("offset").get<std::size_t>();
            const auto bytes = entry.at("nbytes").get<std::size_t>();
            if (offset != next || offset > record.nbytes || bytes > record.nbytes - offset ||
                bytes != std::size_t(shape[1]) * gguf_packed_row_bytes(dtype, shape[2]))
                throw std::invalid_argument("mixed GGUF expert byte ranges disagree");
            auto value = array(allocator::malloc(bytes), Shape{checked_int(bytes)}, uint8);
            model.read_range_into(name, offset, {reinterpret_cast<std::byte*>(value.data<std::uint8_t>()), bytes});
            ids.push_back(int(ids.size())); dtypes.push_back(dtype); storage.push_back(std::move(value));
            next += bytes;
        }
        if (next != record.nbytes) throw std::invalid_argument("mixed GGUF expert payload has trailing bytes");
        auto result = from_mixed_experts(std::move(shape), ids, dtypes, storage);
        result.configure_hadamard(model, name);
        model.record_prepared(name);
        return result;
    }
    auto result = allocate_packed(record.dtype, std::move(shape), record.nbytes);
    result.configure_hadamard(model, name);
    model.read_range_into(name, 0, {reinterpret_cast<std::byte*>(result.bytes_.data<std::uint8_t>()), record.nbytes});
    if (native_projection && result.shape_.size() == 2 && supports_native(model, name)) {
        auto prepared = mlx_native_gguf_pack(result.bytes_, result.format_, result.output_size(), result.input_size(), result.row_bytes_);
        eval(std::vector<array>(prepared.planes.begin(), prepared.planes.end()));
        for (const auto& plane : prepared.planes) MlxWeightResidency::track(plane);
        result.native_ = std::make_shared<MlxNativeGgufWeight>(std::move(prepared));
    }
    if (!result.native_ && use_nax() && result.expert_count() <= 512)
        MlxGgufStagedPrefill::prepare(result);
    MlxWeightResidency::track(result.bytes_);
    if (result.planes_) for (const auto& plane : *result.planes_) MlxWeightResidency::track(plane);
    model.record_prepared(name);
    return result;
}
MlxGgufWeight MlxGgufWeight::from_address_table(std::string_view dtype, Shape shape,
    array table, std::shared_ptr<MlxWeightResidency> residency) {
    if (shape.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || table.dtype() != uint64 ||
        table.shape() != Shape{shape[0]} || !table.flags().row_contiguous || !residency)
        throw std::invalid_argument("invalid GGUF pooled expert address table");
    const auto stride = gguf_packed_row_bytes(dtype, shape[2]);
    table.eval();
    struct Lifetime { array table; std::shared_ptr<MlxWeightResidency> residency; };
    auto owner = std::make_shared<Lifetime>(Lifetime{table, std::move(residency)});
    auto handle = mlx_buffer_view(table.buffer(), table.shape(), uint64, owner);
    register_indirect_weight_pool(handle, owner->residency);
    const auto format = std::find_if(std::begin(formats), std::end(formats),
        [&](const Format& f) { return dtype == f.name; });
    return MlxGgufWeight(std::move(handle), std::move(shape), format->id, checked_int(stride), true);
}
std::size_t MlxGgufWeight::packed_nbytes() const noexcept {
    const auto transform_bytes = rotation_ ? rotation_->signs.nbytes() : 0;
    if (mixed_layout_) return mixed_payload_bytes_ + bytes_.nbytes() + mixed_layout_->nbytes() + transform_bytes;
    std::size_t bytes = bytes_.size() + transform_bytes + (planes_ ? (*planes_)[0].nbytes() + (*planes_)[1].nbytes() : 0);
    if (native_) for (const auto& plane : native_->planes) bytes += plane.nbytes();
    return bytes;
}

array MlxGgufWeight::matmul(const array& input) const {
    check_input(input, input_size());
    if (shape_.size() != 2) throw std::invalid_argument("GGUF expert weights require routed_matmul");
    const auto m = checked_int(input.size() / input_size());
    auto shape = input.shape(); shape.back() = output_size();
    if (!m) return zeros(shape, input.dtype());
    if (rotation_) { auto plain = *this; plain.rotation_.reset(); return plain.matmul(rotate_input(input)); }
    if (native_) return mlx_native_gguf_project(std::span(native_.get(), 1), input)[0];
    const auto x = contiguous(input);
    const auto dense_prefill_fits = [&] {
        if (m < 64 || !use_nax()) return false;
        const auto bytes = std::size_t(input_size()) * output_size() * sizeof(float16_t);
        const auto active = get_active_memory(), limit = get_memory_limit();
        return bytes <= (256u << 20) && active < limit && bytes <= (limit - active) / 2;
    };
    if (planes_) {
        if (m >= 7 && m <= 8 && use_nax() && MlxGgufStagedPrefill::prefers_small_matrix(*this))
            return MlxGgufStagedPrefill::dense_small(*this, x);
        if (m <= 8) return MlxGgufStagedPrefill::project(*this, x);
        if (format_ >= 144 && format_ <= 151 && dense_prefill_fits())
            return gguf_dense_prefill(x, dequantize(float16));
        if (input_size() % 64 == 0) return MlxGgufStagedPrefill::dense(*this, x);
        auto expert = *this;
        expert.shape_.insert(expert.shape_.begin(), 1);
        const MlxGgufPrefillPlan grouped(reshape(x, {m, input_size()}), zeros({m, 1}, int32), 1, m == 1 ? 8 : 32);
        return reshape(grouped.project(expert), shape);
    }
    if (m <= 6) return array(shape, input.dtype(), std::make_shared<GgufProjection>(default_stream(Device::gpu),
        std::array<std::uint32_t, 7>{uint32_t(input_size()), uint32_t(output_size()), uint32_t(m), uint32_t(row_bytes_), 1, 1, 0},
        format_, input.dtype(), projection_values(format_), projection_lanes(format_, input_size(), output_size()), false, false, bytes_),
        std::vector<array>{bytes_, x});
    if (m >= 64 && use_nax() && input_size() % 64 == 0 && MlxGgufStagedPrefill::supports(*this))
        return MlxGgufStagedPrefill::dense(*this, x);
    if (dense_prefill_fits())
        return gguf_dense_prefill(x, dequantize(float16));
    const auto params = parameters(input_size(), output_size(), m, row_bytes_, 1);
    if (use_nax()) {
        const int tile_rows = m >= 32 ? 32 : 16;
        return nax_prefill_kernel(false)({bytes_, x, params}, {shape}, {input.dtype()},
            {((output_size() + 127) / 128) * 128, (m + tile_rows - 1) / tile_rows, 1}, {128, 1, 1},
            {{"F", format_}, {"T", input.dtype()}, {"BM", tile_rows}}, std::nullopt, false, {}).front();
    }
    return prefill_kernel()({bytes_, x, params}, {shape}, {input.dtype()},
        {((output_size() + 31) / 32) * 128, (m + 15) / 16, 1}, {128, 1, 1},
        {{"F", format_}, {"T", input.dtype()}}, std::nullopt, false, {}).front();
}
array MlxGgufWeight::embedding(const array& ids, Dtype dtype) const {
    if (shape_.size() != 2 || (dtype != float16 && dtype != bfloat16 && dtype != float32))
        throw std::invalid_argument("GGUF embedding requires a matrix and floating output");
    auto shape = ids.shape(); shape.push_back(input_size());
    if (!ids.size()) return zeros(shape, dtype);
    if (rotation_) { auto plain = *this; plain.rotation_.reset(); return unrotate_rows(plain.embedding(ids, float32), dtype); }
    if (planes_) return MlxGgufStagedPrefill::embedding(*this, ids, dtype);
    return embedding_kernel()({bytes_, contiguous(astype(ids, int32)),
        parameters(input_size(), output_size(), checked_int(ids.size()), row_bytes_, 1)},
        {shape}, {dtype}, {((input_size() / 4 + 127) / 128) * 128, checked_int(ids.size()), 1}, {128, 1, 1},
        {{"F", format_}, {"T", dtype}}, std::nullopt, false, {}).front();
}
array MlxGgufWeight::swiglu(const MlxGgufWeight& up, const array& input) const {
    check_input(input, input_size());
    if (shape_.size() != 2 || up.shape_ != shape_ || indirect_ || up.indirect_)
        throw std::invalid_argument("GGUF dense Gate/Up requires matching matrices");
    const auto rows = checked_int(input.size() / input_size());
    auto shape = input.shape(); shape.back() = output_size();
    if (!rows) return zeros(shape, input.dtype());
    if (rotation_ || up.rotation_) {
        if (!same_rotation(up)) return moe_swiglu_pair(matmul(input), up.matmul(input));
        auto gate_plain = *this, up_plain = up;
        gate_plain.rotation_.reset(); up_plain.rotation_.reset();
        return gate_plain.swiglu(up_plain, rotate_input(input));
    }
    if (native_ && up.native_) return mlx_native_gguf_swiglu(*native_, *up.native_, input);
    if (planes_ && up.planes_ && rows <= 6) {
        const std::array weights{*this, up};
        auto values = MlxGgufStagedPrefill::project_many(weights, input);
        return moe_swiglu_pair(values[0], values[1]);
    }
    if (planes_ || up.planes_) return moe_swiglu_pair(matmul(input), up.matmul(input));
    if (rows > 6) return moe_swiglu_pair(matmul(input), up.matmul(input));
    const auto [lanes, values] = swiglu_geometry(format_, up.format_);
    return array(shape, input.dtype(), std::make_shared<GgufSwiGLU>(default_stream(Device::gpu),
        std::array<std::uint32_t, 9>{uint32_t(input_size()), uint32_t(output_size()), uint32_t(rows), uint32_t(row_bytes_),
            1, 1, 1, uint32_t(up.row_bytes_), 0},
        format_, up.format_, input.dtype(), lanes, values, false, false, bytes_, up.bytes_, false),
        std::vector<array>{bytes_, up.bytes_, contiguous(input), bytes_});
}
array MlxGgufWeight::dequantize(Dtype dtype) const {
    if (indirect_) throw std::invalid_argument("GGUF streamed expert tables cannot be expanded as one dense tensor");
    if (dtype != float16 && dtype != bfloat16 && dtype != float32)
        throw std::invalid_argument("GGUF dequantization requires floating output");
    if (rotation_) { auto plain = *this; plain.rotation_.reset(); return unrotate_rows(plain.dequantize(float32), dtype); }
    if (planes_) return MlxGgufStagedPrefill::dequantize(*this, dtype);
    const auto rows = checked_int(std::size_t(output_size()) * expert_count());
    const auto blocks = std::size_t(rows) * input_size() / 32;
    return dequantize_kernel()({bytes_, parameters(input_size(), rows, 1, row_bytes_, 1)},
        {shape_}, {dtype}, {checked_int((blocks + 127) / 128 * 128), 1, 1}, {128, 1, 1},
        {{"F", format_}, {"T", dtype}}, std::nullopt, false, {}).front();
}
array MlxGgufWeight::grouped_row_matmul(const array& input, int groups) const {
    check_input(input, input_size());
    if (shape_.size() != 2 || indirect_ || groups < 1 || output_size() % groups ||
        input.ndim() < 2 || input.shape(-2) != groups)
        throw std::invalid_argument("GGUF grouped-row projection geometry differs");
    const int rows = checked_int(input.size() / input_size() / groups);
    auto shape = input.shape(); shape.back() = output_size() / groups;
    if (!rows) return zeros(shape, input.dtype());
    if (rotation_) { auto plain = *this; plain.rotation_.reset(); return plain.grouped_row_matmul(rotate_input(input), groups); }
    if (planes_ && (rows <= 8 || (output_size() / groups) % 64))
        return MlxGgufStagedPrefill::project(*this, input, groups);
    if (rows <= 8 && !planes_) {
        return array(shape, input.dtype(), std::make_shared<GgufProjection>(default_stream(Device::gpu),
            std::array<std::uint32_t, 7>{uint32_t(input_size()), uint32_t(output_size() / groups), uint32_t(rows),
                uint32_t(row_bytes_), uint32_t(groups), 1, 0}, format_, input.dtype(), projection_values(format_),
            projection_lanes(format_, input_size(), output_size() / groups), false, false, bytes_, true),
            std::vector<array>{bytes_, contiguous(input)});
    }
    auto grouped = *this;
    grouped.shape_ = Shape{groups, output_size() / groups, input_size()};
    const auto ids = broadcast_to(arange(groups, int32), {rows, groups});
    auto output = grouped.routed_matmul(reshape(input, {rows, groups, input_size()}), ids, false);
    return reshape(output, shape);
}
array MlxGgufWeight::routed_matmul(const array& input, const array& expert_ids, bool shared_input,
    const MlxGgufRoutePlan* plan) const {
    check_input(input, input_size());
    if (shape_.size() != 3 || expert_ids.ndim() != 2 ||
        (expert_ids.dtype() != int32 && expert_ids.dtype() != uint32))
        throw std::invalid_argument("GGUF routed projection requires [E,N,K] weights and integer [M,R] routes");
    const auto tokens = expert_ids.shape(0), routes = expert_ids.shape(1);
    if (input.size() / input_size() != std::size_t(tokens) * (shared_input ? 1 : routes))
        throw std::invalid_argument("GGUF routed projection input/route mismatch");
    Shape shape{tokens, routes, output_size()};
    if (!expert_ids.size()) return zeros(shape, input.dtype());
    if (rotation_) { auto plain = *this; plain.rotation_.reset(); return plain.routed_matmul(rotate_input(input), expert_ids, shared_input, plan); }
    if (mixed_layout_) return mixed_matmul(input, expert_ids, shared_input, plan);
    if (planes_) {
        const MlxGgufPrefillPlan grouped(input, expert_ids, expert_count(),
            expert_ids.size() <= std::size_t(expert_count()) ? 8 : gguf_prefill_tile_rows(*this, *this, *this, expert_ids), shared_input);
        return grouped.project(*this);
    }
    if (tokens > 8 && use_nax() && MlxGgufStagedPrefill::supports_grouped(*this)) {
        const MlxGgufPrefillPlan grouped(input, expert_ids, expert_count(),
            expert_ids.size() <= std::size_t(expert_count()) ? 8 : 32, shared_input);
        return grouped.project(*this);
    }
    if (tokens > 6) {
        std::optional<MlxGgufRoutePlan> owned;
        if (!plan) { owned = gguf_route_plan(expert_ids, expert_count()); plan = &*owned; }
        const bool sparse = gguf_short_sparse_routes(expert_ids, expert_count());
        if (use_nax()) return nax_prefill_kernel(true)({bytes_, contiguous(input), plan->order, plan->bounds, plan->tile_offsets,
            parameters(input_size(), output_size(), tokens, row_bytes_, expert_count(), routes, shared_input)},
            {shape}, {input.dtype()}, {((output_size() + 127) / 128) * 128, plan->tiles, 1}, {128, 1, 1},
            {{"F", format_}, {"T", input.dtype()}, {"INDIRECT", indirect_}, {"BM", plan->tile_rows},
                {"SPARSE", sparse}}, 0.f, false, {}).front();
        return routed_prefill_kernel()({bytes_, contiguous(input), plan->order, plan->bounds, plan->tile_offsets,
            parameters(input_size(), output_size(), tokens, row_bytes_, expert_count(), routes, shared_input)},
            {shape}, {input.dtype()}, {((output_size() + 31) / 32) * 128, plan->tiles, 1}, {128, 1, 1},
            {{"F", format_}, {"T", input.dtype()}, {"INDIRECT", indirect_}}, 0.f, false, {}).front();
    }
    return array(shape, input.dtype(), std::make_shared<GgufProjection>(default_stream(Device::gpu),
        std::array<std::uint32_t, 7>{uint32_t(input_size()), uint32_t(output_size()), uint32_t(tokens), uint32_t(row_bytes_),
            uint32_t(expert_count()), uint32_t(routes), uint32_t(shared_input)},
        format_, input.dtype(), projection_values(format_), projection_lanes(format_, input_size(), output_size()), true, indirect_, bytes_),
        std::vector<array>{bytes_, contiguous(input), contiguous(astype(expert_ids, int32))});
}
array MlxGgufWeight::routed_swiglu(const MlxGgufWeight& up, const array& input,
    const array& expert_ids, const MlxGgufRoutePlan* plan, float limit) const {
    check_input(input, input_size());
    if (shape_.size() != 3 || up.shape_ != shape_ || expert_ids.ndim() != 2 ||
        (expert_ids.dtype() != int32 && expert_ids.dtype() != uint32) ||
        input.size() / input_size() != std::size_t(expert_ids.shape(0)))
        throw std::invalid_argument("GGUF Gate/Up inputs and routes disagree");
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid GGUF SwiGLU limit");
    const auto activate = [&](const array& g, const array& u) {
        return limit > 0.f ? moe_limited_swiglu_pair(g, u, limit) : moe_swiglu_pair(g, u);
    };
    const auto tokens = expert_ids.shape(0), routes = expert_ids.shape(1);
    const Shape shape{tokens, routes, output_size()};
    if (!expert_ids.size()) return zeros(shape, input.dtype());
    if (rotation_ || up.rotation_) {
        if (!same_rotation(up)) return activate(routed_matmul(input, expert_ids, true, plan),
            up.routed_matmul(input, expert_ids, true, plan));
        auto gate_plain = *this, up_plain = up;
        gate_plain.rotation_.reset(); up_plain.rotation_.reset();
        return gate_plain.routed_swiglu(up_plain, rotate_input(input), expert_ids, plan, limit);
    }
    if ((mixed_layout_ && up.mixed_layout_) || ((mixed_layout_ || up.mixed_layout_) &&
        (tokens == 1 || (tokens <= 4 && use_nax()))))
        return mixed_swiglu(up, input, expert_ids, limit, plan);
    if (mixed_layout_ || up.mixed_layout_ || (limit > 0.f && (planes_ || up.planes_))) return activate(routed_matmul(input, expert_ids, true, plan),
        up.routed_matmul(input, expert_ids, true, plan));
    if (planes_ && up.planes_) {
        const MlxGgufPrefillPlan grouped(input, expert_ids, expert_count(),
            expert_ids.size() <= std::size_t(expert_count()) ? 8 : gguf_prefill_tile_rows(*this, up, *this, expert_ids));
        return grouped.restore(grouped.gate_up(*this, up));
    }
    if (planes_ || up.planes_) return activate(routed_matmul(input, expert_ids, true, plan),
        up.routed_matmul(input, expert_ids, true, plan));
    if (tokens > 8 && use_nax() && MlxGgufStagedPrefill::supports_grouped(*this) && MlxGgufStagedPrefill::supports_grouped(up)) {
        const MlxGgufPrefillPlan grouped(input, expert_ids, expert_count(),
            expert_ids.size() <= std::size_t(expert_count()) ? 8 : 32);
        return limit > 0.f ? activate(grouped.project(*this), grouped.project(up)) : grouped.restore(grouped.gate_up(*this, up));
    }
    if (tokens > 6) {
        std::optional<MlxGgufRoutePlan> owned;
        if (!plan) { owned = gguf_route_plan(expert_ids, expert_count()); plan = &*owned; }
        return activate(routed_matmul(input, expert_ids, true, plan),
            up.routed_matmul(input, expert_ids, true, plan));
    }
    const auto [lanes, values] = swiglu_geometry(format_, up.format_);
    return array(shape, input.dtype(), std::make_shared<GgufSwiGLU>(default_stream(Device::gpu),
        std::array<std::uint32_t, 9>{uint32_t(input_size()), uint32_t(output_size()), uint32_t(tokens), uint32_t(row_bytes_),
            uint32_t(expert_count()), uint32_t(routes), 1, uint32_t(up.row_bytes_), std::bit_cast<uint32_t>(limit)},
        format_, up.format_, input.dtype(), lanes, values, indirect_, up.indirect_, bytes_, up.bytes_),
        std::vector<array>{bytes_, up.bytes_, contiguous(input), contiguous(astype(expert_ids, int32))});
}
} // namespace mfq::metal
