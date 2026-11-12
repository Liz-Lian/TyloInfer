#include "rope.h"
#include "tail.h"
#include "engine/ops.h"
#include "yarn.h"

#include "engine/backends/cuda/kernels/mfq_cuda_attention_ops.h"

#include <algorithm>
#include <array>
#include <optional>
#include <numeric>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"

tyloi::ops::RopeTable RopeCache::view() const {
    using T = ::tyloi::DenseTensor;
    return {std::make_shared<T>(cos), std::make_shared<T>(sin), std::make_shared<T>(sections),
            std::make_shared<T>(empty_sections), std::make_shared<T>(interleaved_order),
            std::make_shared<T>(interleaved_inverse), rotary_dim};
}

tyloi::ops::RopeTable tyloi::cuda::ops::rope_table(int64_t positions, int64_t dimension, double base,
    Device device, int64_t frequency_dim, int64_t active_pairs) {
    std::optional<DeviceGuard> guard;
    if (device.is_cuda()) guard.emplace(device.index);
    return RopeCache(positions, dimension, base, frequency_dim, active_pairs, device).view();
}

tyloi::DenseTensor tyloi::cuda::ops::rope(tyloi::DenseTensor input, tyloi::DenseTensor positions,
    const tyloi::ops::RopeTable& table, bool bf16, bool grid_mrope_positions) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    RopeCache cache;
    auto copy = [](const ::tyloi::Tensor& value, tyloi::DenseTensor& target) {
        if (!value) return;
        const auto* dense = value.representation<tyloi::DenseTensor>();
        if (!dense) throw std::invalid_argument("RoPE requires dense table tensors");
        target = *dense;
    };
    copy(table.cos, cache.cos);
    copy(table.sin, cache.sin);
    copy(table.sections, cache.sections);
    copy(table.empty_sections, cache.empty_sections);
    copy(table.interleaved_order, cache.interleaved_order);
    copy(table.interleaved_inverse, cache.interleaved_inverse);
    cache.rotary_dim = table.rotary_dim;
    auto x = input, pos = positions;
    return bf16 ? cache.apply_bf16(x.is_cuda() && Executor::current().attention_options().fused_rope, x, pos)
                : cache.apply(x, pos, grid_mrope_positions);
}
#endif

RopeCache::RopeCache(
        int64_t max_positions,
        int64_t dim,
        double base,
        int64_t frequency_dim,
        int64_t active_pairs,
        ::tyloi::Device device,
        bool official_reciprocal_frequencies, tyloi::ops::YarnScaling yarn)
    : rotary_dim(dim) {
    int64_t half = rotary_dim / 2;
    const int64_t denominator = frequency_dim > 0 ? frequency_dim : rotary_dim;
    if (active_pairs < 0) active_pairs = half;
    if (active_pairs > half) {
        throw std::runtime_error("RoPE active pair count exceeds rotary dimension");
    }
    auto opts = ::tyloi::TensorOptions().device(device).dtype(::tyloi::kFloat32);
    auto pos = ::tyloi::cuda::arange(max_positions, opts);
    auto ar = ::tyloi::cuda::arange(0, rotary_dim, 2, opts);
    auto freq = ::tyloi::cuda::pow(
        ::tyloi::cuda::full({half}, base, opts),
        -ar / (double)denominator);
    if (official_reciprocal_frequencies) {
        auto cpu_opts = ::tyloi::TensorOptions()
            .device(::tyloi::kCPU).dtype(::tyloi::kFloat32);
#ifdef TYLOINFER_NATIVE_CUDA
        std::vector<float> official_values(static_cast<size_t>(half));
        const float official_base = static_cast<float>(base);
        const float official_denominator = static_cast<float>(denominator);
        for (int64_t index = 0; index < half; ++index) {
            const float exponent =
                static_cast<float>(index * 2) / official_denominator;
            official_values[static_cast<size_t>(index)] =
                1.0f / std::pow(official_base, exponent);
        }
        auto official_freq = ::tyloi::cuda::tensor(
            official_values, cpu_opts);
#else
        auto exponent = ::tyloi::cuda::arange(0, rotary_dim, 2, cpu_opts) /
            (double)denominator;
        auto official_freq = ::tyloi::cuda::reciprocal(
            ::tyloi::cuda::pow(::tyloi::cuda::full({half}, base, cpu_opts), exponent));
#endif
        freq.copy_(official_freq);
    }
    if(yarn.enabled()) {
        std::vector<float> values(static_cast<std::size_t>(half));
        for(int pair=0;pair<half;++pair)values[pair]=float(tyloi::cuda::yarn_frequency(yarn, pair,denominator,base));
        freq.copy_(::tyloi::cuda::tensor(values).to(device));
    }
    if (active_pairs < half) {
        auto pair = ::tyloi::cuda::arange(half, opts);
        freq = ::tyloi::cuda::where(pair < active_pairs, freq, ::tyloi::cuda::zeros_like(freq));
    }
    auto ang = pos.unsqueeze(1) * freq.unsqueeze(0);
    cos = ::tyloi::cuda::cos(ang).contiguous();
    sin = ::tyloi::cuda::sin(ang).contiguous();
    if(yarn.enabled()) { cos=cos*tyloi::cuda::yarn_attention_factor(yarn);sin=sin*tyloi::cuda::yarn_attention_factor(yarn); }
    sections = ::tyloi::cuda::empty({0}, ::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::kCPU));
    empty_sections = sections;
}

void RopeCache::configure_mrope(
        const std::vector<int64_t>& configured_sections,
        bool interleaved,
        int64_t configured_rotary_dim,
        ::tyloi::Device device) {
        if (configured_sections.empty()) return;
        auto execution_sections = configured_sections;
        if (interleaved) {
            std::array<std::vector<int64_t>, 3> pairs;
            for (int64_t pair = 0; pair < configured_rotary_dim / 2; ++pair) {
                const int64_t residue = pair % 3;
                const int axis = residue == 1 &&
                        pair < configured_sections[1] * 3
                    ? 1
                    : (residue == 2 && pair < configured_sections[2] * 3
                        ? 2 : 0);
                pairs[static_cast<size_t>(axis)].push_back(pair);
            }
            execution_sections = {
                static_cast<int64_t>(pairs[0].size()),
                static_cast<int64_t>(pairs[1].size()),
                static_cast<int64_t>(pairs[2].size())};
            std::vector<int64_t> order;
            for (const auto& axis : pairs) {
                order.insert(order.end(), axis.begin(), axis.end());
            }
            std::vector<int64_t> inverse(order.size());
            for (size_t index = 0; index < order.size(); ++index) {
                inverse[static_cast<size_t>(order[index])] =
                    static_cast<int64_t>(index);
            }
            interleaved_order = ::tyloi::cuda::tensor(
                order, ::tyloi::TensorOptions()
                    .dtype(::tyloi::kInt64)
                    .device(::tyloi::kCPU)).to(device);
            interleaved_inverse = ::tyloi::cuda::tensor(
                inverse, ::tyloi::TensorOptions()
                    .dtype(::tyloi::kInt64)
                    .device(::tyloi::kCPU)).to(device);
        }
        sections = ::tyloi::cuda::tensor(
            execution_sections,
            ::tyloi::TensorOptions()
                .dtype(::tyloi::kInt64)
                .device(::tyloi::kCPU));
    }

::tyloi::DenseTensor RopeCache::apply(
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor pos,
        bool grid_mrope_positions) const {
        if (!x.is_cuda()) {
            TYLOINFER_RUNTIME_CHECK(
                !cos.is_cuda() && !sin.is_cuda(),
                "CPU RoPE requires a CPU-resident table");
            auto positions = pos.contiguous().to(::tyloi::kCPU, ::tyloi::kInt64)
                .clamp(0, cos.size(0) - 1);
            ::tyloi::DenseTensor selected_cos;
            ::tyloi::DenseTensor selected_sin;
            if (positions.dim() == 1) {
                selected_cos = cos.index_select(0, positions);
                selected_sin = sin.index_select(0, positions);
            } else if (!grid_mrope_positions && positions.dim() == 2) {
                const int64_t batches = positions.size(0);
                const int64_t tokens = positions.size(1);
                selected_cos = cos.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
                selected_sin = sin.index_select(0, positions.reshape({-1}))
                    .reshape({batches, tokens, rotary_dim / 2})
                    .unsqueeze(1);
            } else if (grid_mrope_positions && sections.numel() == 3) {
                const int64_t * section = sections.data_ptr<int64_t>();
                std::vector<::tyloi::DenseTensor> cos_parts;
                std::vector<::tyloi::DenseTensor> sin_parts;
                int64_t offset = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const int64_t count = section[axis];
                    auto active = positions.select(0, axis);
                    cos_parts.push_back(
                        cos.index_select(0, active).narrow(1, offset, count));
                    sin_parts.push_back(
                        sin.index_select(0, active).narrow(1, offset, count));
                    offset += count;
                }
                selected_cos = ::tyloi::cuda::cat(cos_parts, 1);
                selected_sin = ::tyloi::cuda::cat(sin_parts, 1);
            } else {
                throw std::runtime_error(
                    "multi-axis CPU RoPE requires three position sections");
            }
            auto xf = x.contiguous().to(::tyloi::kFloat32);
            const int64_t half = rotary_dim / 2;
            auto first = xf.narrow(-1, 0, half);
            auto second = xf.narrow(-1, half, half);
            auto output = xf.clone();
            output.narrow(-1, 0, half).copy_(
                first * selected_cos - second * selected_sin);
            output.narrow(-1, half, half).copy_(
                second * selected_cos + first * selected_sin);
            return output;
        }
        auto source = x.contiguous().to(::tyloi::kFloat32);
        auto active_cos = cos;
        auto active_sin = sin;
        const bool interleaved = grid_mrope_positions &&
            interleaved_order.defined();
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<::tyloi::DenseTensor> parts{
                source.narrow(-1, 0, half).index_select(-1, interleaved_order),
                source.narrow(-1, half, half).index_select(-1, interleaved_order)};
            if (source.size(-1) > rotary_dim) {
                parts.push_back(source.narrow(
                    -1, rotary_dim, source.size(-1) - rotary_dim));
            }
            source = ::tyloi::cuda::cat(parts, -1).contiguous();
            active_cos = cos.index_select(1, interleaved_order).contiguous();
            active_sin = sin.index_select(1, interleaved_order).contiguous();
        }
        auto output = rope_table_cuda(
            source,
            pos.contiguous().to(
                ::tyloi::kCUDA, ::tyloi::kInt64),
            active_cos, active_sin, rotary_dim,
            grid_mrope_positions ? sections : empty_sections);
        if (interleaved) {
            const int64_t half = rotary_dim / 2;
            std::vector<::tyloi::DenseTensor> parts{
                output.narrow(-1, 0, half).index_select(
                    -1, interleaved_inverse),
                output.narrow(-1, half, half).index_select(
                    -1, interleaved_inverse)};
            if (output.size(-1) > rotary_dim) {
                parts.push_back(output.narrow(
                    -1, rotary_dim, output.size(-1) - rotary_dim));
            }
            output = ::tyloi::cuda::cat(parts, -1).contiguous();
        }
        return output;
    }

::tyloi::DenseTensor RopeCache::apply_bf16(
        bool fused,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor pos) const {
        TYLOINFER_RUNTIME_CHECK(
            sections.numel() == 0,
            "Qwen3 BF16 RoPE does not support multi-axis sections");
        if (x.is_cuda() && pos.dim() == 1 &&
                fused) {
            return rope_table_bf16_cuda(
                x.contiguous().to(::tyloi::kBFloat16),
                pos.contiguous().to(cos.device(), ::tyloi::kInt64),
                cos, sin, rotary_dim);
        }
        auto positions = pos.contiguous().to(cos.device(), ::tyloi::kInt64)
            .clamp(0, cos.size(0) - 1);
        ::tyloi::DenseTensor selected_cos;
        ::tyloi::DenseTensor selected_sin;
        if (positions.dim() == 1) {
            selected_cos = cos.index_select(0, positions)
                .to(::tyloi::kBFloat16).unsqueeze(0).unsqueeze(0);
            selected_sin = sin.index_select(0, positions)
                .to(::tyloi::kBFloat16).unsqueeze(0).unsqueeze(0);
        } else if (positions.dim() == 2) {
            const int64_t batches = positions.size(0);
            const int64_t tokens = positions.size(1);
            selected_cos = cos.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(::tyloi::kBFloat16).unsqueeze(1);
            selected_sin = sin.index_select(0, positions.reshape({-1}))
                .reshape({batches, tokens, rotary_dim / 2})
                .to(::tyloi::kBFloat16).unsqueeze(1);
        } else {
            throw std::runtime_error(
                "Qwen3 BF16 RoPE positions must be rank one or two");
        }
        auto xb = x.contiguous().to(::tyloi::kBFloat16);
        const int64_t half = rotary_dim / 2;
        auto first = xb.narrow(-1, 0, half);
        auto second = xb.narrow(-1, half, half);
        auto output = xb.clone();
        output.narrow(-1, 0, half).copy_(
            first * selected_cos - second * selected_sin);
        output.narrow(-1, half, half).copy_(
            second * selected_cos + first * selected_sin);
        return output.contiguous();
    }

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA
tyloi::ops::TailRopeTable ops::tail_rope_table(int64_t max_position_embeddings, double rope_base,
                   int64_t compress_ratio, Device device, double compress_rope_base,
                   int64_t original_positions, double rope_factor, double beta_fast, double beta_slow) {
    MfqCudaGuard guard(device.index);
#else
TailRope::TailRope(int64_t max_position_embeddings, double rope_base, int64_t compress_ratio,
                   double compress_rope_base, int64_t original_positions, double rope_factor,
                   double beta_fast, double beta_slow) {
#endif
    constexpr int64_t rotary_dim = 64;
    const int64_t half = rotary_dim / 2;
    const int64_t positions = std::max<int64_t>(1, max_position_embeddings);
    const double base =
        compress_ratio > 0 && compress_rope_base > 0.0 ? compress_rope_base : rope_base;
    auto opts = ::tyloi::TensorOptions()
#ifdef TYLOINFER_NATIVE_CUDA
                    .device(device)
#else
                    .device(::tyloi::kCUDA)
#endif
                    .dtype(::tyloi::kFloat32);
    auto dims = ::tyloi::cuda::arange(0, rotary_dim, 2, opts);
    auto freqs = ::tyloi::cuda::pow(::tyloi::cuda::full({half}, base, opts),
                                         -dims / static_cast<double>(rotary_dim));
    if (compress_ratio > 0 && original_positions > 0) {
        auto correction = [&](double rotations) {
            return static_cast<double>(rotary_dim) *
                   std::log(static_cast<double>(original_positions) /
                            (rotations * 6.28318530717958647692)) /
                   (2.0 * std::log(base));
        };
        const double low = std::max(0.0, std::floor(correction(beta_fast)));
        const double high =
            std::min(static_cast<double>(rotary_dim - 1), std::ceil(correction(beta_slow)));
        const double denominator = high == low ? 0.001 : high - low;
        auto ramp = ::tyloi::cuda::clamp(
            (::tyloi::cuda::arange(half, opts) - low) / denominator, 0.0, 1.0);
        auto smooth = 1.0 - ramp;
        freqs = freqs / rope_factor * (1.0 - smooth) + freqs * smooth;
    }
    auto angles = ::tyloi::cuda::arange(positions, opts).unsqueeze(1) * freqs.unsqueeze(0);
#ifdef TYLOINFER_NATIVE_CUDA
    auto cos = ::tyloi::cuda::cos(angles).contiguous();
    auto sin = ::tyloi::cuda::sin(angles).contiguous();
    auto negative_sin = (-sin).contiguous();
    return {std::make_shared<DenseTensor>(std::move(cos)),
        std::make_shared<DenseTensor>(std::move(sin)),
        std::make_shared<DenseTensor>(std::move(negative_sin))};
#else
    cos_ = ::tyloi::cuda::cos(angles).contiguous();
    sin_ = ::tyloi::cuda::sin(angles).contiguous();
    negative_sin_ = (-sin_).contiguous();
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA
DenseTensor ops::tail_rope(DenseTensor x, DenseTensor positions,
        const tyloi::ops::TailRopeTable& table, bool inverse) {
    if (x.dim() != 4 || x.size(-1) < 64) {
        throw std::runtime_error("DeepSeek V4 RoPE expects contiguous [B,H,T,D>=64]");
    }
    MfqCudaGuard guard(x.device().index);
    const auto& cos = table.cos.dense();
    const auto& sin = (inverse ? table.negative_sin : table.sin).dense();
    return dsv4_tail_rope_cuda(
        x, positions.contiguous().to(x.device(), ::tyloi::kInt64), cos, sin, 64);
}
#else
DenseTensor TailRope::forward(DenseTensor x, DenseTensor positions, bool inverse) const {
    if (x.dim() != 4 || x.size(-1) < 64) {
        throw std::runtime_error("DeepSeek V4 RoPE expects contiguous [B,H,T,D>=64]");
    }
    return dsv4_tail_rope_cuda(
        x, positions.contiguous().to(::tyloi::kCUDA, ::tyloi::kInt64),
        cos_, inverse ? negative_sin_ : sin_, 64);
}
#endif

RotaryEmbedding::RotaryEmbedding(int64_t dimension, int64_t maximum, double base,
                               std::vector<int64_t> sections, bool interleaved,
                               bool fused,tyloi::ops::YarnScaling yarn)
    : dimension_(dimension), maximum_(maximum), base_(base), sections_(std::move(sections)),
      interleaved_(interleaved),yarn_(yarn) {
    fused_ = fused && !yarn_.enabled();
    TYLOINFER_RUNTIME_CHECK(dimension > 0 && dimension % 2 == 0 && maximum > 0 &&
                          std::isfinite(base) && base > 0,
                      "invalid rotary configuration");
    TYLOINFER_RUNTIME_CHECK(
        (sections_.empty() && !interleaved_) ||
            (sections_.size() == 3 &&
             std::all_of(sections_.begin(), sections_.end(), [](auto n) { return n >= 0; }) &&
             std::accumulate(sections_.begin(), sections_.end(), int64_t(0)) == dimension / 2),
        "MRoPE sections disagree with rotary width");
}

bool RotaryEmbedding::fused() const {
#ifdef TYLOINFER_NATIVE_CUDA
    if (const auto* executor = Executor::active())
        return executor->mode().compute.rotary.value_or(fused_) && !yarn_.enabled();
#endif
    return fused_;
}

RotaryEmbedding::Prepared RotaryEmbedding::prepare_parameters(const DenseTensor& value) const {
    const auto pairs = dimension_ / 2;
    std::lock_guard lock(prepared_mutex_);
    auto& cached = prepared_[value.get_device()];
    if (!cached.frequencies.defined()) {
        Prepared next;
        auto pair = ::tyloi::cuda::arange(pairs, value.options().dtype(::tyloi::kInt64));
        next.frequencies = ::tyloi::cuda::pow(::tyloi::cuda::full({pairs}, base_, value.options().dtype(::tyloi::kFloat32)),
            -(pair.to(::tyloi::kFloat32) * 2) / double(dimension_));
        if(yarn_.enabled()) {
            std::vector<float> frequencies(static_cast<std::size_t>(pairs));
            for(int j=0;j<pairs;++j)frequencies[j]=float(tyloi::cuda::yarn_frequency(yarn_, j,dimension_,base_));
            next.frequencies=::tyloi::cuda::tensor(frequencies).to(value.device());
        }
        std::vector<int32_t> selected(static_cast<size_t>(pairs), 0);
        for (int64_t j = 0; j < pairs; ++j) if (!sections_.empty()) {
            if (interleaved_) {
                if (j % 3 == 1 && j < sections_[1] * 3) selected[j] = 1;
                else if (j % 3 == 2 && j < sections_[2] * 3) selected[j] = 2;
            } else selected[j] = j < sections_[0] ? 0 : (j < sections_[0] + sections_[1] ? 1 : 2);
        }
        next.axes = ::tyloi::cuda::tensor(selected).to(value.device());
        TYLOINFER_RUNTIME_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()) == cudaSuccess,
            "warm cached rotary parameters before CUDA graph capture");
        cached = std::move(next);
    }
    return cached;
}

bool RotaryEmbedding::pool_normalized_into(const DenseTensor& index,const DenseTensor& weight,const DenseTensor& positions,
        const DenseTensor& cache_positions,const DenseTensor& output,int64_t pool,int64_t maximum,double eps) const {
    if(!fused() || index.dim()!=3 || index.size(0)!=1 || index.size(2)>256 || index.size(2)<dimension_ ||
        cache_positions.numel()>32 || (index.scalar_type()!=::tyloi::kFloat16 && index.scalar_type()!=::tyloi::kFloat32) ||
        (positions.scalar_type()!=::tyloi::kInt32 && positions.scalar_type()!=::tyloi::kInt64))return false;
    const auto prepared=prepare_parameters(index);
    qsa_pool_rotary_cache_cuda(index,weight.to(::tyloi::kFloat32).contiguous(),positions.contiguous(),cache_positions,
        prepared.frequencies,prepared.axes,output,pool,maximum,maximum_,eps);
    return true;
}

DenseTensor RotaryEmbedding::forward_normalized(const DenseTensor& value, const DenseTensor& weight,
        const DenseTensor& positions, double eps, const DenseTensor& key_cache,
        const DenseTensor& cache_positions, const DenseTensor& projected_value, const DenseTensor& value_cache) const {
    TYLOINFER_RUNTIME_CHECK(fused() && value.is_cuda() && value.dim() == 4 &&
        value.size(-1) >= dimension_, "normalized rotary requires fused [B,T,H,D] input");
    const auto prepared = prepare_parameters(value);
    auto active_positions = positions.scalar_type() == ::tyloi::kInt32 || positions.scalar_type() == ::tyloi::kInt64
        ? positions.contiguous() : positions.to(::tyloi::kInt32).contiguous();
    return rotary_normalized_cached_cuda(value, weight.to(::tyloi::kFloat32).contiguous(), active_positions,
        prepared.frequencies, prepared.axes, maximum_, eps, key_cache, cache_positions,
        projected_value, value_cache);
}

std::vector<DenseTensor> RotaryEmbedding::forward_normalized_grouped(
        const std::vector<DenseTensor>& values,const std::vector<DenseTensor>& weights,
        const DenseTensor& positions,double eps,int cache_entry,const DenseTensor& key_cache,
        const DenseTensor& cache_positions,const DenseTensor& projected_value,const DenseTensor& value_cache) const {
    TYLOINFER_RUNTIME_CHECK(fused() && !values.empty() && values.size()<=3 && weights.size()==values.size(),
        "normalized rotary requires a fused input/weight group");
    std::vector<DenseTensor> active_weights;active_weights.reserve(weights.size());
    for(size_t i=0;i<values.size();++i) {
        TYLOINFER_RUNTIME_CHECK(values[i].is_cuda() && values[i].dim()==4 && values[i].size(-1)>=dimension_,
            "normalized rotary group requires [B,T,H,D] inputs");
        active_weights.push_back(weights[i].to(::tyloi::kFloat32).contiguous());
    }
    const auto prepared=prepare_parameters(values[0]);
    auto active_positions=positions.scalar_type()==::tyloi::kInt32 || positions.scalar_type()==::tyloi::kInt64
        ? positions.contiguous():positions.to(::tyloi::kInt32).contiguous();
    return rotary_normalized_grouped_cuda(values,active_weights,active_positions,prepared.frequencies,
        prepared.axes,maximum_,eps,cache_entry,key_cache,cache_positions,projected_value,value_cache);
}

::tyloi::DenseTensor RotaryEmbedding::forward(const DenseTensor &value, const DenseTensor &positions) const {
    TYLOINFER_RUNTIME_CHECK(value.is_cuda() && value.dim() == 4 && value.size(-1) >= dimension_ &&
                          positions.is_cuda() && positions.device() == value.device() &&
                          positions.dim() >= 1 && positions.dim() <= 3 &&
                          positions.size(-1) == value.size(2),
                      "RoPE requires [B,H,T,D] with compatible positions");
    const auto b = value.size(0), t = value.size(2), pairs = dimension_ / 2;
    const auto axes = positions.dim() == 1 ? 1 : positions.size(0);
    const auto batches = positions.dim() < 3 ? 1 : positions.size(1);
    TYLOINFER_RUNTIME_CHECK(axes > 0 && (batches == 1 || batches == b),
                      "RoPE position batch mismatch");
    auto input = value.scalar_type() == ::tyloi::kFloat32 ? value : value.to(::tyloi::kFloat16);
    if(fused()) {
        const auto prepared = prepare_parameters(value);
        return rotary_embedding_cached_cuda(input.contiguous(),positions.to(::tyloi::kInt32).contiguous(),
            prepared.frequencies,prepared.axes,maximum_);
    }
    auto f = input.to(::tyloi::kFloat32);
    auto pair = ::tyloi::cuda::arange(pairs, value.options().dtype(::tyloi::kInt64));
    auto frequencies = ::tyloi::cuda::pow(::tyloi::cuda::full({pairs}, base_, f.options()),
                               -(pair.to(::tyloi::kFloat32) * 2) / double(dimension_));
    if(yarn_.enabled()) {
        frequencies=prepare_parameters(value).frequencies;
    }
    auto axis_ids = ::tyloi::cuda::zeros({pairs}, pair.options());
    if (!sections_.empty()) {
        if (interleaved_) {
            auto remainder = pair.remainder(3);
            auto a1 = (remainder == 1) & (pair < sections_[1] * 3);
            auto a2 = (remainder == 2) & (pair < sections_[2] * 3);
            axis_ids = ::tyloi::cuda::where(a1, ::tyloi::cuda::full_like(pair, 1),
                                 ::tyloi::cuda::where(a2, ::tyloi::cuda::full_like(pair, 2), axis_ids));
        } else {
            axis_ids = ::tyloi::cuda::where(pair < sections_[0], axis_ids,
                                 ::tyloi::cuda::where(pair < sections_[0] + sections_[1],
                                           ::tyloi::cuda::full_like(pair, 1), ::tyloi::cuda::full_like(pair, 2)));
        }
        axis_ids = ::tyloi::cuda::where(axis_ids < axes, axis_ids, ::tyloi::cuda::zeros_like(axis_ids));
    }
    DenseTensor cosine, sine;
    for (int64_t axis = 0; axis < std::min<int64_t>(axes, 3); ++axis) {
        auto pos = (positions.dim() == 1 ? positions : positions.select(0, axis))
                       .reshape({batches, t})
                       .to(::tyloi::kInt32)
                       .clamp(0, maximum_ - 1)
                       .to(::tyloi::kFloat32);
        auto angles = pos.unsqueeze(-1) * frequencies;
        auto ca = ::tyloi::cuda::cos(angles), sa = ::tyloi::cuda::sin(angles);
        if (axis == 0) {
            cosine = ca;
            sine = sa;
        } else {
            cosine = ::tyloi::cuda::where(axis_ids == axis, ca, cosine);
            sine = ::tyloi::cuda::where(axis_ids == axis, sa, sine);
        }
    }
    cosine = cosine.unsqueeze(1);
    sine = sine.unsqueeze(1);
    if(yarn_.enabled()) { cosine=cosine*tyloi::cuda::yarn_attention_factor(yarn_);sine=sine*tyloi::cuda::yarn_attention_factor(yarn_); }
    auto first = f.narrow(-1, 0, pairs), second = f.narrow(-1, pairs, pairs);
    return ::tyloi::cuda::cat({first * cosine - second * sine, second * cosine + first * sine,
                    f.narrow(-1, dimension_, f.size(-1) - dimension_)},
                   -1)
        .to(input.scalar_type());
}

} // namespace tyloi::cuda
