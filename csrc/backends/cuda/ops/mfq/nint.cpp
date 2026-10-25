#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_quant_ops.h"
#include "mfq_cuda_nint_route_hint.h"
#include "resident_allocations.h"
#include "nint.h"
#include "nint_prefill_cache.h"
#include "backends/cuda/engine/executor/execution.h"
#include "format.h"
#include "packed_nint.h"
#include "cpu_projection_rows.h"
#include "backends/cuda/engine/executor/decode_window.h"
#include "backends/cuda/engine/executor/options.h"

#include <cstring>
#include <limits>
#include <array>
#include <algorithm>
#include <map>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define TYLOINFER_CPU_X86_GNU 1
#endif

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

struct NintPrefillDecodeCacheState {
    using Key=std::array<uintptr_t,11>;
    std::map<Key,mfq_tensor_backend::Tensor> decoded;
    NintPrefillDecodeCacheState* previous=nullptr;
};
namespace {
Workspace& resident_nint_workspace(const NintWeight& weight,int rows) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    mfq::cuda::ResidentAllocationScope resident;
#endif
    return weight.workspace(rows);
}
thread_local NintPrefillDecodeCacheState* nint_prefill_decode_cache=nullptr;
NintPrefillDecodeCacheState::Key nint_prefill_decode_key(const NintWeight& w) {
    return {reinterpret_cast<uintptr_t>(w.q_packed.data_ptr()),
        reinterpret_cast<uintptr_t>(w.row_q_bits.data_ptr()),
        reinterpret_cast<uintptr_t>(w.row_q_bit_offsets.data_ptr()),
        reinterpret_cast<uintptr_t>(w.sub_scale.data_ptr()),
        reinterpret_cast<uintptr_t>(w.sub_min.data_ptr()),
        reinterpret_cast<uintptr_t>(w.neuron_scale.data_ptr()),
        reinterpret_cast<uintptr_t>(w.neuron_min.data_ptr()),
        uintptr_t(w.neuron_len),uintptr_t(w.gs),uintptr_t(w.out),uintptr_t(w.q_packed.get_device())};
}
}
NintPrefillDecodeCacheScope::NintPrefillDecodeCacheScope(bool enabled) {
    if(!enabled)return;
    state_=std::make_unique<NintPrefillDecodeCacheState>();
    state_->previous=nint_prefill_decode_cache;
    nint_prefill_decode_cache=state_.get();
}
NintPrefillDecodeCacheScope::~NintPrefillDecodeCacheScope() {
    if(state_)nint_prefill_decode_cache=state_->previous;
}
void NintPrefillDecodeCacheScope::clear() {
    if(state_)state_->decoded.clear();
}

mfq_tensor_backend::Tensor nint_row_embedding_lookup(
        const mfq::NintRows& table, const mfq_tensor_backend::Tensor& ids) {
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.scalar_type() == mfq_tensor_backend::kInt64,
        "NINT row IDs must be CUDA int64");
    const MfqCudaGuard guard(ids.device());
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(mfq_current_cuda_stream(), &capture));
    TYLOINFER_RUNTIME_CHECK(capture == cudaStreamCaptureStatusNone,
        "range-backed NINT row lookup cannot run inside CUDA graph capture");
    auto shape = ids.sizes().vec();
    shape.push_back(table.width());
    if (ids.numel() == 0)
        return mfq_tensor_backend::empty(shape, ids.options().dtype(mfq_tensor_backend::kFloat16));
    TYLOINFER_RUNTIME_CHECK(ids.numel() <= std::numeric_limits<int>::max() / 6 &&
        static_cast<uint64_t>(ids.numel()) * table.width() <= std::numeric_limits<uint32_t>::max(),
        "NINT row lookup batch exceeds bounds");
    auto host = ids.contiguous().cpu();
    mfq::NintRowBatch selected;
    table.append_rows(host.data_ptr<int64_t>(), static_cast<std::size_t>(host.numel()),
        selected, mfq_get_num_threads());
    selected.validate();
    auto packed = cpu_u8_tensor(selected.packed(), {static_cast<int64_t>(selected.packed_nbytes())}).to(ids.device());
    std::vector<int32_t> words(selected.descriptors().size());
    std::memcpy(words.data(), selected.descriptors().data(), words.size() * sizeof(uint32_t));
    auto descriptors = cpu_i32_tensor(words, {static_cast<int64_t>(selected.rows()), 6}).to(ids.device());
    return nint_selected_rows_cuda(packed, descriptors, table.width()).reshape(shape);
}

mfq_tensor_backend::Tensor pad_last(mfq_tensor_backend::Tensor x, int64_t target) {
    if (x.size(1) == target) return x;
    if (x.size(1) > target) throw std::runtime_error("activation width exceeds neuron_len");
    return mfq_tensor_backend::constant_pad_nd(x, {0, target - x.size(1)}, 0);
}
uint32_t cpu_load_packed_bits(
        const uint8_t * data,
        int64_t nbytes,
        int64_t bit,
        int bits) {
    const int64_t byte = bit >> 3;
    const int shift = static_cast<int>(bit & 7);
    uint32_t word = data[byte];
    if (byte + 1 < nbytes) word |= static_cast<uint32_t>(data[byte + 1]) << 8;
    if (byte + 2 < nbytes) word |= static_cast<uint32_t>(data[byte + 2]) << 16;
    return (word >> shift) & ((1u << bits) - 1u);
}

#ifdef TYLOINFER_CPU_X86_GNU
__attribute__((target("avx512f,avx512bw,avx512vnni")))
int32_t cpu_dot_s8_s8_vnni(
        const int8_t * left,
        const int8_t * right,
        int32_t right_sum) {
    const __m512i signed_a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(left));
    const __m512i a = _mm512_xor_si512(
        signed_a, _mm512_set1_epi8(static_cast<char>(-128)));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(right));
    const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, b);
    return _mm512_reduce_add_epi32(dot) - 128 * right_sum;
}

__attribute__((target("avx512f,avx512bw,avx512vnni")))
int32_t cpu_dot_u8_s8_vnni(
        const uint8_t * left,
        const int8_t * right) {
    const __m512i a = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(left));
    const __m512i b = _mm512_loadu_si512(
        reinterpret_cast<const __m512i *>(right));
    const __m512i dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a, b);
    return _mm512_reduce_add_epi32(dot);
}

bool cpu_has_avx512_vnni() {
    static const bool supported = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vnni");
    }();
    return supported;
}
#endif

int32_t cpu_dot_s8_s8_64(
        const int8_t * left,
        const int8_t * right,
        int32_t right_sum) {
#ifdef TYLOINFER_CPU_X86_GNU
    if (cpu_has_avx512_vnni()) {
        return cpu_dot_s8_s8_vnni(left, right, right_sum);
    }
#endif
    int32_t result = 0;
    for (int index = 0; index < 64; ++index) {
        result += static_cast<int32_t>(left[index]) *
            static_cast<int32_t>(right[index]);
    }
    return result;
}

int32_t cpu_dot_u8_s8_64(
        const uint8_t * left,
        const int8_t * right) {
#ifdef TYLOINFER_CPU_X86_GNU
    if (cpu_has_avx512_vnni()) return cpu_dot_u8_s8_vnni(left, right);
#endif
    int32_t result = 0;
    for (int index = 0; index < 64; ++index) {
        result += static_cast<int32_t>(left[index]) *
            static_cast<int32_t>(right[index]);
    }
    return result;
}

void cpu_unpack_nint_group(
        const uint8_t * packed,
        int bits,
        int valid,
        uint8_t * values) {
    std::fill(values, values + 64, static_cast<uint8_t>(0));
    if (bits == 8) {
        std::memcpy(values, packed, static_cast<size_t>(valid));
        return;
    }
    const uint32_t mask = (1u << bits) - 1u;
    uint32_t reservoir = 0;
    int available = 0;
    int source = 0;
    for (int index = 0; index < valid; ++index) {
        while (available < bits) {
            reservoir |= static_cast<uint32_t>(packed[source++]) << available;
            available += 8;
        }
        values[index] = static_cast<uint8_t>(reservoir & mask);
        reservoir >>= bits;
        available -= bits;
    }
}

void cpu_unpack_nint_group_at_bit_offset(
        const uint8_t * packed,
        uint64_t bit_offset,
        int bits,
        int valid,
        uint8_t * values) {
    std::fill(values, values + 64, static_cast<uint8_t>(0));
    const uint32_t mask = bits == 8 ? 255u : ((1u << bits) - 1u);
    for (int index = 0; index < valid; ++index) {
        const uint64_t value_bit =
            bit_offset + static_cast<uint64_t>(index) * static_cast<uint64_t>(bits);
        const size_t byte = static_cast<size_t>(value_bit >> 3);
        const int shift = static_cast<int>(value_bit & 7u);
        const uint32_t pair = static_cast<uint32_t>(packed[byte]) |
            (static_cast<uint32_t>(packed[byte + 1]) << 8);
        values[index] = static_cast<uint8_t>((pair >> shift) & mask);
    }
}

CpuQuantizedActivation cpu_quantize_activation(
        mfq_tensor_backend::Tensor x,
        int64_t width,
        int64_t group_size,
        bool with_sums) {
    TYLOINFER_RUNTIME_CHECK(!x.is_cuda(), "CPU activation quantization requires a CPU tensor");
    x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat16), width).contiguous();
    const int64_t rows = x.size(0);
    const int64_t groups = (width + group_size - 1) / group_size;
    TYLOINFER_RUNTIME_CHECK(group_size > 0 && group_size <= 64,
        "CPU compact GEMV group size must be in [1, 64]");
    CpuQuantizedActivation result;
    result.rows = rows;
    result.groups = groups;
    result.group_size = group_size;
    result.values.resize(static_cast<size_t>(rows * groups * 64));
    result.scales.resize(static_cast<size_t>(rows * groups));
    if (with_sums) {
        result.sums.resize(static_cast<size_t>(rows * groups));
    }
    const mfq_half * input = x.data_ptr<mfq_half>();
    mfq_parallel_for(0, rows * groups, 1, [&](int64_t begin, int64_t end) {
        for (int64_t linear = begin; linear < end; ++linear) {
            const int64_t row = linear / groups;
            const int64_t group = linear - row * groups;
            const int64_t k0 = group * group_size;
            const int64_t valid = std::min(group_size, width - k0);
            float maximum = 0.0f;
            for (int64_t index = 0; index < valid; ++index) {
                maximum = std::max(
                    maximum,
                    std::fabs(static_cast<float>(input[row * width + k0 + index])));
            }
            const float scale = maximum > 0.0f ? maximum / 127.0f : 1.0f;
            result.scales[static_cast<size_t>(linear)] = scale;
            int32_t sum = 0;
            int8_t * quantized = result.values.data() + linear * 64;
            for (int64_t index = 0; index < valid; ++index) {
                int value = static_cast<int>(std::round(
                    static_cast<float>(input[row * width + k0 + index]) / scale));
                value = std::max(-127, std::min(127, value));
                quantized[index] = static_cast<int8_t>(value);
                sum += value;
            }
            std::fill(
                quantized + valid,
                quantized + 64,
                static_cast<int8_t>(0));
            if (with_sums) {
                result.sums[static_cast<size_t>(linear)] = sum;
            }
        }
    });
    return result;
}

float cpu_half_from_bytes(const int8_t * bytes, int64_t offset) {
    uint16_t raw = 0;
    std::memcpy(&raw, bytes + offset, sizeof(raw));
    mfq_half value;
    std::memcpy(&value, &raw, sizeof(raw));
    return static_cast<float>(value);
}
CpuProjectionRows make_cpu_projection_rows(const NintWeight& w) {
    TYLOINFER_RUNTIME_CHECK(!w.q8_zero && w.q_packed.is_cpu(),"original-FP32 rows require CPU NINTv2 storage");
    CpuProjectionRows p;
    p.width=static_cast<int>(w.neuron_len); p.outputs=static_cast<int>(w.out);
    p.nint={w.q_packed.data_ptr<uint8_t>(),static_cast<size_t>(w.q_packed.numel()),
        w.row_q_bits.data_ptr<uint8_t>(),w.row_q_bit_offsets.data_ptr<int64_t>(),
        w.sub_scale.data_ptr<uint8_t>(),w.sub_min.data_ptr<uint8_t>(),
        w.neuron_scale.data_ptr<float>(),w.neuron_min.data_ptr<float>(),w.out,w.neuron_len,w.gs};
    p.nint_rows=mfq::cpu::packed_nint_rows_kernel();
    return p;
}
mfq_tensor_backend::Tensor nint_sharded_embedding_lookup(
    const std::vector<std::shared_ptr<mfq::NintRows>>& tables,const std::vector<int64_t>& ids,
    const std::vector<int64_t>& shape,const mfq_tensor_backend::Device& device) {
    namespace tb=mfq_tensor_backend;
    if(tables.empty() || !device.is_cuda())throw std::invalid_argument("sharded NINT embeddings require CUDA tables");
    const MfqCudaGuard guard(device);const int64_t rows=tables.front()->rows();const int width=tables.front()->width();
    int64_t count=1;for(auto n:shape){if(n<0 || (n && count>std::numeric_limits<int64_t>::max()/n))throw std::overflow_error("embedding shape overflow");count*=n;}
    if(count!=static_cast<int64_t>(ids.size()))throw std::invalid_argument("sharded embedding ID shape mismatch");
    std::vector<std::vector<int64_t>> selected(tables.size()),positions(tables.size());
    for(std::size_t i=0;i<ids.size();++i) {
        const auto id=ids[i];
        if(id<0 || id/rows>=static_cast<int64_t>(tables.size()))throw std::out_of_range("sharded embedding row ID");
        selected[id/rows].push_back(id%rows);positions[id/rows].push_back(static_cast<int64_t>(i));
    }
    std::vector<mfq::NintRowBatch> pieces(tables.size());
    mfq::host_parallel_for(0,static_cast<int64_t>(tables.size()),1,mfq_get_num_threads(),[&](int64_t first,int64_t last) {
        for(auto s=first;s<last;++s) {
            if(tables[s]->rows()!=rows || tables[s]->width()!=width)throw std::invalid_argument("NINT shard geometry mismatch");
            if(!selected[s].empty())tables[s]->append_rows(selected[s].data(),selected[s].size(),pieces[s],1);
        }
    });
    mfq::NintRowBatch batch;std::vector<int64_t> inverse(ids.size());int64_t at=0;
    for(std::size_t s=0;s<tables.size();++s) {
        batch.append_batch(pieces[s]);for(auto p:positions[s])inverse[p]=at++;
    }
    auto out_shape=shape;out_shape.push_back(width);
    if(ids.empty())return tb::empty(out_shape,tb::TensorOptions().device(device).dtype(tb::kFloat16));
    batch.validate();
    auto packed=cpu_u8_tensor(batch.packed(),{static_cast<int64_t>(batch.packed_nbytes())}).to(device);
    std::vector<int32_t> words(batch.descriptors().size());std::memcpy(words.data(),batch.descriptors().data(),words.size()*sizeof(uint32_t));
    auto descriptors=cpu_i32_tensor(words,{static_cast<int64_t>(batch.rows()),6}).to(device);
    return nint_selected_rows_cuda(packed,descriptors,width).index_select(0,tb::tensor(inverse).to(device)).reshape(out_shape);
}
static mfq_tensor_backend::Tensor nint_matmul_cpu(
        const NintWeight & w,
        mfq_tensor_backend::Tensor x) {
    TYLOINFER_RUNTIME_CHECK(!x.is_cuda(), "CPU NINT GEMV requires CPU activations");
    TYLOINFER_RUNTIME_CHECK(!w.q_packed.is_cuda(), "CPU NINT GEMV requires CPU-resident weights");
    if (!w.q8_zero) {
        TYLOINFER_RUNTIME_CHECK(!w.row_q_bits.is_cuda() && !w.row_q_bit_offsets.is_cuda() &&
            !w.sub_scale.is_cuda() && !w.sub_min.is_cuda() &&
            !w.neuron_scale.is_cuda() && !w.neuron_min.is_cuda(),
            "CPU NINT GEMV requires CPU-resident metadata");
        x = pad_last(x.contiguous().to(mfq_tensor_backend::kFloat32), w.neuron_len).contiguous();
        auto result = mfq_tensor_backend::empty({x.size(0), w.out},
            x.options().dtype(mfq_tensor_backend::kFloat32));
        mfq::cpu::PackedNintView view;
        view.packed = w.q_packed.data_ptr<uint8_t>();
        view.packed_bytes = static_cast<size_t>(w.q_packed.numel());
        view.row_bits = w.row_q_bits.data_ptr<uint8_t>();
        view.row_bit_offsets = w.row_q_bit_offsets.data_ptr<int64_t>();
        view.group_scale = w.sub_scale.data_ptr<uint8_t>();
        view.group_min = w.sub_min.data_ptr<uint8_t>();
        view.row_scale = w.neuron_scale.data_ptr<float>();
        view.row_min = w.neuron_min.data_ptr<float>();
        view.outputs = w.out;
        view.width = w.neuron_len;
        view.group_size = w.gs;
        mfq::cpu::packed_nint_matmul(view, x.data_ptr<float>(), x.size(0),
            w.neuron_len, result.data_ptr<float>(), w.out, mfq_get_num_threads());
        return result.to(mfq_tensor_backend::kFloat16);
    }
    auto activation = cpu_quantize_activation(
        std::move(x), w.neuron_len, w.gs, true);
    const int64_t rows = activation.rows;
    const int64_t width = w.neuron_len;
    const int64_t outputs = w.out;
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    mfq_half * output = result.data_ptr<mfq_half>();
    const uint8_t * quant = w.q_packed.data_ptr<uint8_t>();
    const uint8_t * row_q_bits = w.q8_zero
        ? nullptr : w.row_q_bits.data_ptr<uint8_t>();
    const int64_t * row_q_bit_offsets = w.q8_zero
        ? nullptr : w.row_q_bit_offsets.data_ptr<int64_t>();
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            const int neuron_bits = w.q8_zero
                ? 8 : static_cast<int>(row_q_bits[neuron]);
            for (int64_t group = 0; group < w.ng; ++group) {
                const int64_t meta = neuron * w.ng + group;
                const uint8_t * packed = w.q8_zero
                    ? quant + meta * 32
                    : quant;
                float scale = 0.0f;
                float minimum = 0.0f;
                if (w.q8_zero) {
                    scale = static_cast<float>(
                        w.q8_zero_scale.data_ptr<mfq_half>()[meta]);
                } else {
                    scale = w.neuron_scale.data_ptr<float>()[neuron] *
                        static_cast<float>(w.sub_scale.data_ptr<uint8_t>()[meta]);
                    minimum = w.neuron_min.data_ptr<float>()[neuron] *
                        static_cast<float>(w.sub_min.data_ptr<uint8_t>()[meta]);
                }
                const int64_t valid = std::min<int64_t>(
                    w.gs, width - group * w.gs);
                alignas(64) uint8_t weights[64];
                if (w.q8_zero) {
                    std::fill(weights, weights + 64, static_cast<uint8_t>(0));
                    std::memcpy(weights, packed, static_cast<size_t>(valid));
                } else {
                    const uint64_t bit_offset =
                        static_cast<uint64_t>(row_q_bit_offsets[neuron]) +
                        static_cast<uint64_t>(group * w.gs) *
                            static_cast<uint64_t>(neuron_bits);
                    cpu_unpack_nint_group_at_bit_offset(
                        packed, bit_offset, neuron_bits,
                        static_cast<int>(valid), weights);
                }
                for (int64_t row = 0; row < rows; ++row) {
                    const int64_t activation_group =
                        row * activation.groups + group;
                    const int8_t * quantized = activation.values.data() +
                        activation_group * 64;
                    int32_t dot = 0;
                    if (w.q8_zero) {
                        dot = cpu_dot_s8_s8_64(
                            reinterpret_cast<const int8_t *>(weights), quantized,
                            activation.sums[
                                static_cast<size_t>(activation_group)]);
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            scale * activation.scales[
                                static_cast<size_t>(activation_group)],
                            static_cast<float>(dot),
                            accumulators[static_cast<size_t>(row)]);
                    } else {
                        dot = cpu_dot_u8_s8_64(weights, quantized);
                        const float activation_scale = activation.scales[
                            static_cast<size_t>(activation_group)];
                        const float dot_term = scale * activation_scale *
                            static_cast<float>(dot);
                        const float min_term = minimum * activation_scale *
                            static_cast<float>(activation.sums[
                                static_cast<size_t>(activation_group)]);
                        accumulators[static_cast<size_t>(row)] +=
                            dot_term - min_term;
                    }
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] =
                    mfq_half(accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}
namespace {
bool nint_packed_decode_rows(const NintWeight& w,int64_t rows) {
    // Batched decode projects directly from packed weights. Preserve
    // prefill's GEMM path and its layer decode cache.
    return !w.q8_zero && (rows <= 8 || (rows <= 32 && w.gs>=4 && w.gs<=64 &&
        !nint_prefill_decode_cache && mfq::cuda::DecodeWindow::recording() &&
        (mfq::cuda::runtime_options::nint_compressed_decode_batch() ||
         (rows>16 && mfq::cuda::runtime_options::nint_compressed_large_batch(w.out>=65536)))));
}
}
mfq_tensor_backend::Tensor nint_matmul(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x) {
    if (!x.is_cuda()) return nint_matmul_cpu(w, std::move(x));
    x = x.contiguous();
    const bool packed_rows=nint_packed_decode_rows(w,x.size(0));
    if (!packed_rows && !w.q8_zero && w.out>=65536 && x.size(0)>8 && x.size(0)<=32 &&
        !nint_prefill_decode_cache && mfq::cuda::runtime_options::nint_large_output_tiling()) {
        // Large vocabulary heads otherwise materialize over a GiB of FP16
        // weights. Reuse compressed GEMV across up to eight activation rows.
        x=x.to(mfq_tensor_backend::kFloat16);
        return profiler.measure("nint.matmul.tiled_output",[&] {
            std::vector<mfq_tensor_backend::Tensor> output;
            for(int64_t row=0;row<x.size(0);row+=8)
                output.push_back(nint_matmul(profiler,w,x.narrow(0,row,std::min<int64_t>(8,x.size(0)-row))));
            return mfq_tensor_backend::cat(output,0);
        });
    }
    if (x.size(0) > 8 || (x.scalar_type() != mfq_tensor_backend::kFloat16 &&
            x.scalar_type() != mfq_tensor_backend::kFloat32))
        x = x.to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    int M = (int)x.size(0);
    if (!w.q8_zero) {
        if(packed_rows && M>8) {
            return profiler.measure("nint.matmul.online_f16",[&] {
                return nint_matmul_online_f16_cuda(w.q_packed,w.row_q_bits,w.row_q_bit_offsets,
                    w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,x,w.gs);
            });
        }
        if (M <= 8) {
            Workspace & ws = resident_nint_workspace(w,M);
            return profiler.measure("nint.matmul", [&]() {
                if (w.aligned_q8) {
                    return nint_matmul_q8_ws_cuda(
                        w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                        w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                        x, w.gs, ws.qx, ws.xscale);
                }
                if((w.uniform_qbits==5 && w.gs==28) || (w.uniform_qbits==6 && (w.gs==24 || w.gs==28))) {
                    return nint_matmul_fixed_bits_ws_cuda(
                        w.q_packed,w.row_q_bits,w.row_q_bit_offsets,
                        w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,
                        x,w.gs,ws.qx,ws.xscale,w.uniform_qbits);
                }
                return nint_matmul_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, w.gs, ws.qx, ws.xscale);
            });
        }
        const auto decode=[&] {
            return profiler.measure("nint.dequant", [&]() {return nint_decode_cuda(
                w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                w.neuron_len, w.gs);});
        };
        mfq_tensor_backend::Tensor dense;
        if(nint_prefill_decode_cache) {
            const auto key=nint_prefill_decode_key(w);
            const auto found=nint_prefill_decode_cache->decoded.find(key);
            if(found!=nint_prefill_decode_cache->decoded.end())dense=found->second;
            else {
                dense=decode();nint_prefill_decode_cache->decoded.emplace(key,dense);
            }
        } else dense=decode();
        return profiler.measure("nint.gemm", [&]() {
            return mfq_tensor_backend::matmul(x, dense.transpose(0, 1));
        });
    }
    if (M <= 8) {
        Workspace & ws = resident_nint_workspace(w,M);
        return profiler.measure("nint8_zero.gemv", [&]() {
            return nint8_zero_gemv_ws_cuda(
                w.q_packed, w.q8_zero_scale, x, ws.qx, ws.xscale);
        });
    }
    return profiler.measure("nint8_zero.packed_mmq", [&]() {
        return nint8_zero_mmq_f16_packed_cuda(
            w.q_packed, w.q8_zero_scale, x, w.neuron_len);
    });
}

static mfq_tensor_backend::Tensor nint_matmul_f32(
        CudaProfiler& profiler,
        const NintWeight & w,
        mfq_tensor_backend::Tensor x) {
    x = pad_last(
        x.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    if (!w.q8_zero) {
        return nint_matmul(
            profiler, w, x).to(mfq_tensor_backend::kFloat32);
    }
    TYLOINFER_RUNTIME_CHECK(
        x.size(0) >= 16,
        "FP32-output NINT MMQ requires at least 16 activation rows");
    return profiler.measure(
        "kld_mmq.nint8_zero.fp32_output", [&]() {
            return nint8_zero_mmq_f32_packed_cuda(
                w.q_packed, w.q8_zero_scale,
                x, w.neuron_len);
        });
}

mfq_tensor_backend::Tensor nint_matmul_input_mul_f32(
        CudaProfiler& profiler,
        const NintWeight & w,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) {
    TYLOINFER_RUNTIME_CHECK(
        mode == 1 || mode == 2,
        "FP32-output NINT input gate mode must be sigmoid or SiLU");
    x = pad_last(
        x.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    gate = pad_last(
        gate.contiguous().to(mfq_tensor_backend::kFloat16),
        w.neuron_len);
    TYLOINFER_RUNTIME_CHECK(
        x.sizes() == gate.sizes(),
        "FP32-output NINT x and gate shapes must match");
    auto activation = mode == 1
        ? x * mfq_tensor_backend::sigmoid(gate)
        : x * mfq_tensor_backend::silu(gate);
    return nint_matmul_f32(
        profiler, w, activation.contiguous());
}

mfq_tensor_backend::Tensor nint_matmul_groupwise_u8(
        CudaProfiler& profiler,
        const NintWeight & w,
        mfq_tensor_backend::Tensor x,
        int64_t groups) {
    TYLOINFER_RUNTIME_CHECK(
        w.bits == 8 && w.gs == 48,
        "groupwise NINT projection requires NINT8 gs48");
    TYLOINFER_RUNTIME_CHECK(
        x.dim() == 3 && x.size(1) == groups && x.size(2) == w.neuron_len,
        "groupwise NINT projection expects [B, groups, K]");
    TYLOINFER_RUNTIME_CHECK(
        w.out % groups == 0,
        "groupwise NINT projection output rows must divide groups");
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    auto dense = profiler.measure("nint.groupwise_dequant", [&]() {
        return nint_decode_cuda(
            w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
            w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
            w.neuron_len, w.gs);
    });
    const int64_t rows_per_group = w.out / groups;
    return profiler.measure("nint.groupwise_gemm", [&]() {
        return mfq_tensor_backend::bmm(
            x.transpose(0, 1),
            dense.reshape({groups, rows_per_group, w.neuron_len})
                .transpose(1, 2))
            .transpose(0, 1)
            .contiguous()
            .reshape({x.size(0), w.out});
    });
}

mfq_tensor_backend::Tensor nint_matmul_input_mul(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor gate,
        int mode) {
    TYLOINFER_RUNTIME_CHECK(
        mode == 1 || mode == 2,
        "NINT input gate mode must be sigmoid or SiLU");
    if (!x.is_cuda()) {
        x = x.contiguous().to(mfq_tensor_backend::kFloat16);
        gate = gate.contiguous().to(mfq_tensor_backend::kFloat16);
        TYLOINFER_RUNTIME_CHECK(x.sizes() == gate.sizes(), "NINT x and gate shapes must match");
        auto value = mode == 1
            ? x * mfq_tensor_backend::sigmoid(gate)
            : x * mfq_tensor_backend::silu(gate);
        return nint_matmul_cpu(w, value);
    }
    x = x.contiguous();
    gate = gate.contiguous();
    const bool fused_input = !w.q8_zero && x.size(0) <= 8;
    if (!fused_input || (x.scalar_type() != mfq_tensor_backend::kFloat16 &&
            x.scalar_type() != mfq_tensor_backend::kFloat32))
        x = x.to(mfq_tensor_backend::kFloat16);
    if (!fused_input || (gate.scalar_type() != mfq_tensor_backend::kFloat16 &&
            gate.scalar_type() != mfq_tensor_backend::kFloat32))
        gate = gate.to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(
        x.sizes() == gate.sizes(),
        "NINT x and gate shapes must match");
    x = pad_last(x, w.neuron_len);
    gate = pad_last(gate, w.neuron_len);
    if (!w.q8_zero && x.size(0) <= 8) {
        Workspace & ws = resident_nint_workspace(w,static_cast<int>(x.size(0)));
        return profiler.measure("nint.matmul.input_mul", [&]() {
            if (w.aligned_q8) {
                return nint_matmul_input_mul_q8_ws_cuda(
                    w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                    w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                    x, gate, mode, w.gs, ws.qx, ws.xscale);
            }
            const auto fixed_setting=mfq::cuda::runtime_options::nint_fused_fixed_bits();
            if((!fixed_setting || *fixed_setting) &&
                    ((w.uniform_qbits==5 && w.gs==28) || (w.uniform_qbits==6 && (w.gs==24 || w.gs==28)))) {
                return nint_matmul_input_mul_fixed_bits_ws_cuda(
                    w.q_packed,w.row_q_bits,w.row_q_bit_offsets,
                    w.sub_scale,w.sub_min,w.neuron_scale,w.neuron_min,
                    x,gate,mode,w.gs,ws.qx,ws.xscale,w.uniform_qbits);
            }
            return nint_matmul_input_mul_ws_cuda(
                w.q_packed, w.row_q_bits, w.row_q_bit_offsets,
                w.sub_scale, w.sub_min, w.neuron_scale, w.neuron_min,
                x, gate, mode, w.gs, ws.qx, ws.xscale);
        });
    }
    if (mode == 1) {
        return nint_matmul(
            profiler, w, x * mfq_tensor_backend::sigmoid(gate));
    }
    return nint_matmul(
        profiler, w, x * mfq_tensor_backend::silu(gate));
}

mfq_tensor_backend::Tensor nint_matmul_swiglu(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    auto parts = nint_matmul(profiler, w, x).chunk(2, -1);
    return mfq_tensor_backend::silu(parts[0]) * parts[1];
}

mfq_tensor_backend::Tensor nint_matmul_geglu(
        CudaProfiler& profiler,
        const NintWeight& w,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    x = pad_last(x, w.neuron_len);
    auto parts = nint_matmul(profiler, w, x).chunk(2, -1);
    return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
}


void require_tp_row_major_weight(
        const std::vector<int64_t> & shape,
        int axis,
        int out,
        int neuron_len,
        const char * format) {
    if (shape.size() != 2 || axis != 0 ||
        shape[0] != out || shape[1] != neuron_len) {
        throw std::runtime_error(
            std::string(format) +
            " tensor parallelism requires a row-major rank-2 weight");
    }
}

static uint64_t nint_row_payload_bits(
        int ng, int gs, int bits) {
    if (ng <= 0 || gs <= 0 || bits < 1 || bits > 8 ||
            static_cast<uint64_t>(ng) >
                std::numeric_limits<uint64_t>::max() /
                    static_cast<uint64_t>(gs) /
                    static_cast<uint64_t>(bits)) {
        throw std::runtime_error("invalid NINT row bit geometry");
    }
    return static_cast<uint64_t>(ng) *
        static_cast<uint64_t>(gs) * static_cast<uint64_t>(bits);
}

static void copy_nint_packed_bits(
        const std::vector<uint8_t> & source,
        uint64_t source_bit,
        std::vector<uint8_t> & destination,
        uint64_t destination_bit,
        uint64_t bit_count) {
    const uint64_t source_capacity =
        static_cast<uint64_t>(source.size()) * 8u;
    const uint64_t destination_capacity =
        static_cast<uint64_t>(destination.size()) * 8u;
    if (source_bit > source_capacity ||
            bit_count > source_capacity - source_bit ||
            destination_bit > destination_capacity ||
            bit_count > destination_capacity - destination_bit) {
        throw std::runtime_error("NINT packed bit range is out of bounds");
    }
    if (((source_bit | destination_bit | bit_count) & 7u) == 0u) {
        std::memcpy(
            destination.data() + static_cast<size_t>(destination_bit >> 3),
            source.data() + static_cast<size_t>(source_bit >> 3),
            static_cast<size_t>(bit_count >> 3));
        return;
    }
    while (bit_count != 0) {
        const int source_shift = static_cast<int>(source_bit & 7u);
        const int destination_shift = static_cast<int>(destination_bit & 7u);
        const int chunk = static_cast<int>(std::min<uint64_t>(
            bit_count,
            static_cast<uint64_t>(std::min(
                8 - source_shift, 8 - destination_shift))));
        const uint8_t mask = static_cast<uint8_t>((1u << chunk) - 1u);
        const uint8_t value = static_cast<uint8_t>(
            (source[static_cast<size_t>(source_bit >> 3)] >> source_shift) &
            mask);
        destination[static_cast<size_t>(destination_bit >> 3)] |=
            static_cast<uint8_t>(value << destination_shift);
        source_bit += static_cast<uint64_t>(chunk);
        destination_bit += static_cast<uint64_t>(chunk);
        bit_count -= static_cast<uint64_t>(chunk);
    }
}

static void require_canonical_nint_cpu(const mfq::nint::Data & source) {
    if (source.out <= 0 ||
            source.row_q_bits.size() != static_cast<size_t>(source.out) ||
            source.row_q_bit_offsets.size() != static_cast<size_t>(source.out) ||
            source.row_sub_bits.size() != static_cast<size_t>(source.out) ||
            source.q_packed.size() < 8) {
        throw std::runtime_error("NINT storage is missing canonical row metadata");
    }
    const uint64_t payload_capacity =
        static_cast<uint64_t>(source.q_packed.size() - 8) * 8u;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const int64_t signed_offset =
            source.row_q_bit_offsets[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        if (signed_offset < 0 ||
                static_cast<uint64_t>(signed_offset) > payload_capacity ||
                row_bits > payload_capacity -
                    static_cast<uint64_t>(signed_offset)) {
            throw std::runtime_error("invalid canonical NINT row metadata");
        }
    }
}

static mfq::nint::Data repack_nint_cpu_rows(
        const mfq::nint::Data & source,
        const std::vector<int64_t> & rows) {
    require_canonical_nint_cpu(source);
    if (rows.empty()) {
        throw std::runtime_error("cannot create an empty NINT row selection");
    }
    uint64_t total_bits = 0;
    for (int64_t row : rows) {
        if (row < 0 || row >= source.out) {
            throw std::runtime_error("NINT selected row is out of range");
        }
        const uint64_t row_bits = nint_row_payload_bits(
            source.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT selected row payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT selected row payload is too large");
    }
    mfq::nint::Data result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bits.resize(rows.size());
    result.row_q_bit_offsets.resize(rows.size());
    result.row_sub_bits.resize(rows.size());
    result.sub_scale.resize(rows.size() * static_cast<size_t>(source.ng));
    result.sub_min.resize(rows.size() * static_cast<size_t>(source.ng));
    result.neuron_scale_h.resize(rows.size());
    result.neuron_min_h.resize(rows.size());
    uint64_t destination_bit = 0;
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const size_t source_row = static_cast<size_t>(rows[destination]);
        const int bits = source.row_q_bits[source_row];
        const uint64_t row_bits =
            nint_row_payload_bits(source.ng, source.gs, bits);
        result.row_q_bits[destination] = static_cast<uint8_t>(bits);
        result.row_sub_bits[destination] = source.row_sub_bits[source_row];
        result.row_q_bit_offsets[destination] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed,
            static_cast<uint64_t>(source.row_q_bit_offsets[source_row]),
            result.q_packed, destination_bit, row_bits);
        const size_t source_group = source_row * static_cast<size_t>(source.ng);
        const size_t destination_group =
            destination * static_cast<size_t>(source.ng);
        std::copy_n(
            source.sub_scale.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_scale.begin() + static_cast<ptrdiff_t>(destination_group));
        std::copy_n(
            source.sub_min.begin() + static_cast<ptrdiff_t>(source_group),
            source.ng,
            result.sub_min.begin() + static_cast<ptrdiff_t>(destination_group));
        result.neuron_scale_h[destination] = source.neuron_scale_h[source_row];
        result.neuron_min_h[destination] = source.neuron_min_h[source_row];
        destination_bit += row_bits;
    }
    mfq::nint::describe(result);
    return result;
}

mfq::nint::Data slice_nint_cpu_output(
        const mfq::nint::Data & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT output shard");
    }
    std::vector<int64_t> rows(static_cast<size_t>(end - begin));
    std::iota(rows.begin(), rows.end(), begin);
    return repack_nint_cpu_rows(source, rows);
}

mfq::nint::Data slice_nint_cpu_input_groups(
        const mfq::nint::Data & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT input shard");
    }
    require_canonical_nint_cpu(source);
    mfq::nint::Data result = source;
    result.ng = static_cast<int>(end - begin);
    const int64_t element_begin = begin * source.gs;
    const int64_t element_end =
        std::min<int64_t>(end * source.gs, source.neuron_len);
    result.neuron_len = static_cast<int>(element_end - element_begin);
    result.shape[1] = result.neuron_len;
    uint64_t total_bits = 0;
    for (int row = 0; row < source.out; ++row) {
        const uint64_t row_bits = nint_row_payload_bits(
            result.ng, source.gs,
            source.row_q_bits[static_cast<size_t>(row)]);
        if (row_bits > std::numeric_limits<uint64_t>::max() - total_bits) {
            throw std::runtime_error("NINT input shard payload is too large");
        }
        total_bits += row_bits;
    }
    if (total_bits > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            (total_bits + 7u) / 8u >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max() - 8)) {
        throw std::runtime_error("NINT input shard payload is too large");
    }
    result.q_packed.assign(static_cast<size_t>((total_bits + 7u) / 8u) + 8, 0);
    result.row_q_bit_offsets.resize(static_cast<size_t>(source.out));
    result.sub_scale.resize(
        static_cast<size_t>(source.out) * result.ng);
    result.sub_min.resize(
        static_cast<size_t>(source.out) * result.ng);
    uint64_t destination_bit = 0;
    for (int row = 0; row < source.out; ++row) {
        const int bits = source.row_q_bits[static_cast<size_t>(row)];
        const uint64_t row_bits =
            nint_row_payload_bits(result.ng, source.gs, bits);
        const uint64_t source_bit = static_cast<uint64_t>(
            source.row_q_bit_offsets[static_cast<size_t>(row)]) +
            static_cast<uint64_t>(begin) *
                static_cast<uint64_t>(source.gs) *
                static_cast<uint64_t>(bits);
        result.row_q_bit_offsets[static_cast<size_t>(row)] =
            static_cast<int64_t>(destination_bit);
        copy_nint_packed_bits(
            source.q_packed, source_bit,
            result.q_packed, destination_bit, row_bits);
        destination_bit += row_bits;
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.sub_scale.data() + destination_group,
            source.sub_scale.data() + source_group,
            static_cast<size_t>(result.ng));
        std::memcpy(
            result.sub_min.data() + destination_group,
            source.sub_min.data() + source_group,
            static_cast<size_t>(result.ng));
    }
    mfq::nint::describe(result);
    return result;
}

mfq::nint8::Data slice_nint8_zero_cpu_output(
        const mfq::nint8::Data & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.out) {
        throw std::runtime_error("invalid NINT8-0 output shard");
    }
    mfq::nint8::Data result = source;
    result.out = static_cast<int>(end - begin);
    result.shape[0] = result.out;
    const size_t first_group =
        static_cast<size_t>(begin) * source.ng;
    const size_t group_count =
        static_cast<size_t>(result.out) * source.ng;
    result.q.assign(
        source.q.begin() +
            static_cast<ptrdiff_t>(first_group * 32),
        source.q.begin() +
            static_cast<ptrdiff_t>((first_group + group_count) * 32));
    result.scale_h.assign(
        source.scale_h.begin() + static_cast<ptrdiff_t>(first_group),
        source.scale_h.begin() +
            static_cast<ptrdiff_t>(first_group + group_count));
    return result;
}

mfq::nint8::Data slice_nint8_zero_cpu_input_groups(
        const mfq::nint8::Data & source, int64_t begin, int64_t end) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (begin < 0 || begin >= end || end > source.ng) {
        throw std::runtime_error("invalid NINT8-0 input shard");
    }
    mfq::nint8::Data result = source;
    result.ng = static_cast<int>(end - begin);
    result.neuron_len = result.ng * 32;
    result.shape[1] = result.neuron_len;
    result.q.resize(
        static_cast<size_t>(source.out) * result.ng * 32);
    result.scale_h.resize(
        static_cast<size_t>(source.out) * result.ng);
    for (int row = 0; row < source.out; ++row) {
        const size_t source_group =
            static_cast<size_t>(row) * source.ng +
            static_cast<size_t>(begin);
        const size_t destination_group =
            static_cast<size_t>(row) * result.ng;
        std::memcpy(
            result.q.data() + destination_group * 32,
            source.q.data() + source_group * 32,
            static_cast<size_t>(result.ng) * 32);
        std::memcpy(
            result.scale_h.data() + destination_group,
            source.scale_h.data() + source_group,
            static_cast<size_t>(result.ng) * sizeof(uint16_t));
    }
    return result;
}
NintWeight to_device_nint(const mfq::nint::Data & c, bool cuda) {
    if (c.bits < 1 || c.bits > 8) throw std::runtime_error("unsupported NINT bits");
    require_canonical_nint_cpu(c);
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = c.gs;
    w.bits = c.bits;
    w.neuron_len = c.neuron_len;
    w.format_version = c.format_version;
    w.aggregate_bpw = c.aggregate_bpw;
    w.distribution_entropy = c.distribution_entropy;
    w.shape = c.shape;
    w.aligned_q8 = c.gs % 4 == 0;
    w.uniform_qbits=c.row_q_bits.empty()?0:c.row_q_bits.front();
    for (size_t row = 0; row < c.row_q_bits.size(); ++row) {
        if(c.row_q_bits[row]!=w.uniform_qbits)w.uniform_qbits=0;
        w.aligned_q8 = w.aligned_q8 && c.row_q_bits[row] == 8 &&
            (c.row_q_bit_offsets[row] & 31) == 0;
    }
    w.q_packed = cpu_u8_tensor(
        c.q_packed, {static_cast<int64_t>(c.q_packed.size())});
    w.row_q_bits = cpu_u8_tensor(c.row_q_bits, {c.out});
    w.row_q_bit_offsets = cpu_i64_tensor(c.row_q_bit_offsets, {c.out});
    w.sub_scale = cpu_u8_tensor(c.sub_scale, {c.out, c.ng});
    w.sub_min = cpu_u8_tensor(c.sub_min, {c.out, c.ng});
    w.neuron_scale = cpu_f16_to_f32_tensor(c.neuron_scale_h, c.out);
    w.neuron_min = cpu_f16_to_f32_tensor(c.neuron_min_h, c.out);
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.q_packed = w.q_packed.to(target).contiguous();
        w.row_q_bits = w.row_q_bits.to(target).contiguous();
        w.row_q_bit_offsets = w.row_q_bit_offsets.to(target).contiguous();
        w.sub_scale = w.sub_scale.to(target).contiguous();
        w.sub_min = w.sub_min.to(target).contiguous();
        w.neuron_scale = w.neuron_scale.to(target).contiguous();
        w.neuron_min = w.neuron_min.to(target).contiguous();
    }
    return w;
}

NintWeight to_gpu_nint(const mfq::nint::Data & c) {
    return to_device_nint(c, true);
}

NintWeight to_cuda_device_nint(
        const mfq::nint::Data & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint(c, true);
}

NintWeight to_cpu_nint(const mfq::nint::Data & c) {
    return to_device_nint(c, false);
}

NintWeight to_device_mfe_nint(
        const mfq::nint::Data & source,
        int local_experts,
        int out_per_expert,
        bool cuda) {
    require_canonical_nint_cpu(source);
    if (local_experts <= 0 || out_per_expert <= 0 ||
            source.out != local_experts * out_per_expert) {
        throw std::runtime_error("invalid MFE NINT expert geometry");
    }
    uint64_t maximum_payload_bits = 0;
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t payload_bits = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const uint64_t row_bits = nint_row_payload_bits(
                source.ng, source.gs, source.row_q_bits[row]);
            if (row_bits > std::numeric_limits<uint64_t>::max() - payload_bits) {
                throw std::runtime_error("MFE NINT expert payload is too large");
            }
            payload_bits += row_bits;
        }
        maximum_payload_bits = std::max(maximum_payload_bits, payload_bits);
    }
    const uint64_t stride_u64 = (maximum_payload_bits + 7u) / 8u + 8u;
    if (stride_u64 > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            static_cast<uint64_t>(local_experts) >
                static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                    stride_u64) {
        throw std::runtime_error("MFE NINT expert payload is too large");
    }
    const size_t stride = static_cast<size_t>(stride_u64);
    std::vector<uint8_t> expert_stream(
        static_cast<size_t>(local_experts) * stride, 0);
    std::vector<int64_t> relative_offsets(static_cast<size_t>(source.out));
    for (int expert = 0; expert < local_experts; ++expert) {
        uint64_t destination_bit = 0;
        for (int local_row = 0; local_row < out_per_expert; ++local_row) {
            const size_t row = static_cast<size_t>(expert) * out_per_expert +
                static_cast<size_t>(local_row);
            const int bits = source.row_q_bits[row];
            const uint64_t row_bits =
                nint_row_payload_bits(source.ng, source.gs, bits);
            relative_offsets[row] = static_cast<int64_t>(destination_bit);
            copy_nint_packed_bits(
                source.q_packed,
                static_cast<uint64_t>(source.row_q_bit_offsets[row]),
                expert_stream,
                static_cast<uint64_t>(expert) * stride_u64 * 8u +
                    destination_bit,
                row_bits);
            destination_bit += row_bits;
        }
    }
    NintWeight result = to_device_nint(source, false);
    result.aligned_q8 = false;
    result.q_expert_stride = static_cast<int64_t>(stride);
    result.q_packed = cpu_u8_tensor(
        expert_stream, {local_experts, static_cast<int64_t>(stride)});
    result.row_q_bit_offsets = cpu_i64_tensor(
        relative_offsets, {source.out});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        result.q_packed = result.q_packed.to(target).contiguous();
        result.row_q_bits = result.row_q_bits.to(target).contiguous();
        result.row_q_bit_offsets =
            result.row_q_bit_offsets.to(target).contiguous();
        result.sub_scale = result.sub_scale.to(target).contiguous();
        result.sub_min = result.sub_min.to(target).contiguous();
        result.neuron_scale = result.neuron_scale.to(target).contiguous();
        result.neuron_min = result.neuron_min.to(target).contiguous();
    }
    return result;
}

NintWeight to_device_nint8_zero(
        const mfq::nint8::Data & c, bool cuda) {
    NintWeight w;
    w.out = c.out;
    w.ng = c.ng;
    w.gs = 32;
    w.bits = 8;
    w.neuron_len = c.neuron_len;
    w.q8_zero = true;
    w.shape = c.shape;
    w.q_packed = cpu_u8_tensor(c.q, {c.out, c.ng, 32});
    w.q8_zero_scale = cpu_f16_tensor(c.scale_h, {c.out, c.ng});
    if (cuda) {
        const auto target = mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
        w.q_packed = w.q_packed.to(target).contiguous();
        w.q8_zero_scale = w.q8_zero_scale.to(target).contiguous();
    }
    return w;
}

NintWeight to_gpu_nint8_zero(const mfq::nint8::Data & c) {
    return to_device_nint8_zero(c, true);
}

NintWeight to_cuda_device_nint8_zero(
        const mfq::nint8::Data & c, int device) {
    MfqCudaGuard guard(device);
    return to_device_nint8_zero(c, true);
}

NintWeight to_cpu_nint8_zero(const mfq::nint8::Data & c) {
    return to_device_nint8_zero(c, false);
}

mfq::nint::Data select_nint_cpu_rows(
        const mfq::nint::Data & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT");
    return repack_nint_cpu_rows(source, rows);
}

mfq::nint8::Data select_nint8_zero_cpu_rows(
        const mfq::nint8::Data & source,
        const std::vector<int64_t> & rows) {
    require_tp_row_major_weight(
        source.shape, source.axis, source.out,
        source.neuron_len, "NINT8-0");
    if (rows.empty()) {
        throw std::runtime_error(
            "cannot create an empty NINT8-0 MoE shard");
    }
    mfq::nint8::Data result = source;
    result.out = static_cast<int>(rows.size());
    result.shape[0] = result.out;
    const size_t q_row_bytes =
        static_cast<size_t>(source.ng) * 32;
    result.q.resize(rows.size() * q_row_bytes);
    result.scale_h.resize(
        rows.size() * source.ng);
    for (size_t destination = 0;
         destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 ||
            source_row >= source.out) {
            throw std::runtime_error(
                "NINT8-0 MoE shard row is out of range");
        }
        std::memcpy(
            result.q.data() +
                destination * q_row_bytes,
            source.q.data() +
                static_cast<size_t>(source_row) *
                    q_row_bytes,
            q_row_bytes);
        std::memcpy(
            result.scale_h.data() +
                destination * source.ng,
            source.scale_h.data() +
                static_cast<size_t>(source_row) *
                    source.ng,
            static_cast<size_t>(source.ng) *
                sizeof(uint16_t));
    }
    return result;
}

mfq_tensor_backend::Tensor dequant_nint8_zero_cpu(const mfq::nint8::Data & source) {
    TYLOINFER_RUNTIME_CHECK(source.shape.size() == 2,
        "CPU NINT8-0 dense tensor must be rank 2");
    auto packed = to_device_nint8_zero(source, false);
    auto result = mfq_tensor_backend::empty(
        source.shape,
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    const auto * quantized =
        reinterpret_cast<const int8_t *>(packed.q_packed.data_ptr<uint8_t>());
    const mfq_half * scales = packed.q8_zero_scale.data_ptr<mfq_half>();
    float * output = result.data_ptr<float>();
    mfq_parallel_for(0, source.out, 1, [&](int64_t begin, int64_t end) {
        for (int64_t row = begin; row < end; ++row) {
            for (int64_t group = 0; group < source.ng; ++group) {
                const int64_t meta = row * source.ng + group;
                const float scale = static_cast<float>(scales[meta]);
                const int64_t k0 = group * 32;
                const int64_t valid = std::min<int64_t>(
                    32, source.neuron_len - k0);
                for (int64_t index = 0; index < valid; ++index) {
                    output[row * source.neuron_len + k0 + index] =
                        scale * static_cast<float>(quantized[meta * 32 + index]);
                }
            }
        }
    });
    return result.contiguous();
}
