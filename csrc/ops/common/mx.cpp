#include "mfq_cuda_quant_ops.h"
#include "mx.h"
#include "mxfp8_grouped_gemm.h"
#include "format.h"
#include "cpu_projection_rows.h"
#include <cmath>

CpuProjectionRows make_cpu_projection_rows(const Mxfp4Weight& w) {
    TYLOINFER_RUNTIME_CHECK(!w.values.is_cuda() && !w.scales.is_cuda() && w.neuron_len>0 &&
        w.neuron_len%32==0 && w.values.numel()==w.out*(w.neuron_len/2) &&
        w.scales.numel()==w.out*(w.neuron_len/32),"invalid CPU MXFP4 projection fields");
    CpuProjectionRows plan;plan.width=int(w.neuron_len);plan.outputs=int(w.out);
    const auto* values=w.values.data_ptr<uint8_t>();const auto* scales=w.scales.data_ptr<uint8_t>();
    const int width=plan.width;
    plan.scalar_rows=[values,scales,width](const float* input,int batch,int input_stride,
            float* output,int output_stride,int begin,int end) {
        static constexpr float table[16]={0,.5f,1,1.5f,2,3,4,6,0,-.5f,-1,-1.5f,-2,-3,-4,-6};
        for(int row=begin;row<end;++row)for(int token=0;token<batch;++token) {
            float sum=0;
            for(int group=0;group<width/32;++group) {
                const float scale=std::ldexp(1.0f,int(scales[std::size_t(row)*(width/32)+group])-127);
                const auto* packed=values+std::size_t(row)*(width/2)+group*16;
                const auto* x=input+std::size_t(token)*input_stride+group*32;
                for(int i=0;i<16;++i) {
                    sum+=x[2*i]*float(mfq_half(table[packed[i]&15]*scale));
                    sum+=x[2*i+1]*float(mfq_half(table[packed[i]>>4]*scale));
                }
            }
            output[std::size_t(token)*output_stride+row]=sum;
        }
    };
    return plan;
}

using mfq_tensor_backend::indexing::Slice;
using namespace mfq::cuda::quant_format;

mfq_tensor_backend::Tensor mxfp8_matmul(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 activation width mismatch");
    if (!x.is_cuda()) {
        TYLOINFER_RUNTIME_CHECK(
            !weight.values.is_cuda() && !weight.scales.is_cuda(),
            "CPU MXFP8 matmul requires CPU-resident weights");
        const int64_t rows = x.size(0);
        const int64_t outputs = weight.out;
        const int64_t width = weight.neuron_len;
        const int64_t scale_columns = weight.scales.size(1);
        const int64_t block_columns=width/scale_columns;
        const int64_t block_rows=weight.scales.size(0)==outputs?1:block_columns;
        const auto * input = x.data_ptr<mfq_half>();
        const auto * values = weight.values.data_ptr<uint8_t>();
        const auto * scales = weight.scales.data_ptr<uint8_t>();
        auto result = mfq_tensor_backend::empty(
            {rows, outputs},
            mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
        auto * output = result.data_ptr<mfq_half>();
        mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
            std::vector<float> accumulators(static_cast<size_t>(rows));
            for (int64_t neuron = begin; neuron < end; ++neuron) {
                std::fill(accumulators.begin(), accumulators.end(), 0.0f);
                for (int64_t column = 0; column < width; ++column) {
                    const uint8_t raw = values[neuron * width + column];
                    const unsigned exponent =
                        (static_cast<unsigned>(raw) >> 3u) & 15u;
                    const unsigned mantissa = static_cast<unsigned>(raw) & 7u;
                    float decoded = exponent == 0u
                        ? std::ldexp(float(mantissa) * 0.125f, -6)
                        : std::ldexp(
                            1.0f + float(mantissa) * 0.125f,
                            static_cast<int>(exponent) - 7);
                    if ((raw & 128u) != 0u) decoded = -decoded;
                    const uint8_t raw_scale = scales[
                        (neuron / block_rows) * scale_columns + column / block_columns];
                    const float weight_value = decoded * std::ldexp(
                        1.0f, static_cast<int>(raw_scale) - 127);
                    for (int64_t row = 0; row < rows; ++row) {
                        accumulators[static_cast<size_t>(row)] = std::fma(
                            static_cast<float>(input[row * width + column]),
                            weight_value,
                            accumulators[static_cast<size_t>(row)]);
                    }
                }
                for (int64_t row = 0; row < rows; ++row) {
                    output[row * outputs + neuron] = mfq_half(
                        accumulators[static_cast<size_t>(row)]);
                }
            }
        });
        return result;
    }
    if (x.size(0) <= 8) {
        return profiler.measure("mxfp8.small_m", [&]() {
            return mxfp8_small_m_cuda(
                weight.values, weight.scales, x);
        });
    }
    return profiler.measure("mxfp8.packed_matmul", [&]() {
        return mxfp8_matmul_f16_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_matmul_f32(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP8 FP32-output activation width mismatch");
    if (x.size(0) <= 8) {
        return profiler.measure("mxfp8.small_m_f32", [&]() {
            return mxfp8_small_m_f32_cuda(
                weight.values, weight.scales, x);
        });
    }
    return profiler.measure("mxfp8.gemm_f32", [&]() {
        return mxfp8_gemm_f32_cuda(
            weight.values, weight.scales, x);
    });
}

mfq_tensor_backend::Tensor mxfp8_cpu_reference(
        const Mxfp8Weight & weight) {
    auto values = weight.values.to(mfq_tensor_backend::kCPU).contiguous();
    auto scales = weight.scales.to(mfq_tensor_backend::kCPU).contiguous();
    const auto * value_bytes = values.data_ptr<uint8_t>();
    const auto * scale_bytes = scales.data_ptr<uint8_t>();
    const int64_t scale_columns = scales.size(1);
    const int64_t block_columns=weight.neuron_len/scale_columns;
    const int64_t block_rows=scales.size(0)==weight.out?1:block_columns;
    std::vector<float> dense(
        static_cast<size_t>(weight.out * weight.neuron_len));
    for (int64_t row = 0; row < weight.out; ++row) {
        for (int64_t column = 0;
             column < weight.neuron_len; ++column) {
            const uint8_t raw = value_bytes[
                static_cast<size_t>(row * weight.neuron_len + column)];
            const unsigned exponent =
                (static_cast<unsigned>(raw) >> 3u) & 15u;
            const unsigned mantissa =
                static_cast<unsigned>(raw) & 7u;
            float value;
            if (exponent == 15u && mantissa == 7u) {
                value = std::numeric_limits<float>::quiet_NaN();
            } else {
                value = exponent == 0u
                    ? std::ldexp(float(mantissa) * 0.125f, -6)
                    : std::ldexp(
                        1.0f + float(mantissa) * 0.125f,
                        static_cast<int>(exponent) - 7);
                if ((raw & 128u) != 0u) value = -value;
            }
            const uint8_t raw_scale = scale_bytes[
                static_cast<size_t>(
                    (row / block_rows) * scale_columns + column / block_columns)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, int(raw_scale) - 127);
            dense[static_cast<size_t>(
                row * weight.neuron_len + column)] = value * scale;
        }
    }
    return mfq_tensor_backend::from_blob(
        dense.data(), {weight.out, weight.neuron_len},
        mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kFloat32))
        .clone().to(weight.values.device())
        .to(mfq_tensor_backend::kFloat16).contiguous();
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    TYLOINFER_RUNTIME_CHECK(
        outputs_per_group % (weight.scales.size(0)==weight.out?1:weight.neuron_len/weight.scales.size(1)) == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if(!grouped.is_cuda()) {
        std::vector<mfq_tensor_backend::Tensor> outputs;
        outputs.reserve(static_cast<size_t>(groups));
        const auto block_rows=weight.scales.size(0)==weight.out?1:
            weight.neuron_len/weight.scales.size(1);
        const auto scale_rows=outputs_per_group/block_rows;
        for(int64_t group=0;group<groups;++group) {
            const Mxfp8Weight shard{
                weight.values.narrow(0,group*outputs_per_group,outputs_per_group),
                weight.scales.narrow(0,group*scale_rows,scale_rows),outputs_per_group,weight.neuron_len};
            outputs.push_back(mxfp8_matmul(profiler,shard,grouped.select(1,group).contiguous()));
        }
        return mfq_tensor_backend::cat(outputs,-1).contiguous();
    }
    if (grouped.size(0) <= 8) {
        return profiler.measure("mxfp8.groupwise_small_m", [&]() {
            return mxfp8_groupwise_small_m_cuda(
                weight.values, weight.scales, grouped, groups);
        });
    }
    return profiler.measure("mxfp8.groupwise_gemm", [&]() {
        return mxfp8_groupwise_gemm_f16_cuda(weight.values,weight.scales,grouped,groups);
    });
}

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul_f32(
        CudaProfiler& profiler,
        const Mxfp8Weight & weight,
        mfq_tensor_backend::Tensor grouped,
        int64_t groups) {
    grouped = grouped.contiguous().to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(
        grouped.dim() == 3 && grouped.size(1) == groups &&
            grouped.size(2) == weight.neuron_len &&
            weight.out % groups == 0,
        "MXFP8 groupwise FP32-output projection geometry mismatch");
    const int64_t outputs_per_group = weight.out / groups;
    TYLOINFER_RUNTIME_CHECK(
        outputs_per_group % (weight.scales.size(0)==weight.out?1:weight.neuron_len/weight.scales.size(1)) == 0,
        "MXFP8 groupwise output width must preserve scale blocks");
    if (grouped.size(0) <= 8) {
        return profiler.measure(
            "mxfp8.groupwise_small_m_f32", [&]() {
                return mxfp8_groupwise_small_m_f32_cuda(
                    weight.values, weight.scales, grouped, groups);
            });
    }
    return profiler.measure("mxfp8.groupwise_gemm_f32", [&]() {
        return mxfp8_groupwise_gemm_f32_cuda(weight.values,weight.scales,grouped,groups);
    });
}

mfq_tensor_backend::Tensor mxfp4_matmul(
        const Mxfp4Weight & weight,
        mfq_tensor_backend::Tensor x) {
    x = x.contiguous().to(mfq_tensor_backend::kFloat16);
    TYLOINFER_RUNTIME_CHECK(
        x.dim() == 2 && x.size(1) == weight.neuron_len,
        "MXFP4 activation width mismatch");
    if (x.is_cuda()) {
        TYLOINFER_RUNTIME_CHECK(
            weight.values.is_cuda() && weight.scales.is_cuda(),
            "CUDA MXFP4 matmul requires CUDA-resident weights");
        return mxfp4_matmul_f16_cuda(
            weight.values, weight.scales, x);
    }
    TYLOINFER_RUNTIME_CHECK(
        !weight.values.is_cuda() && !weight.scales.is_cuda(),
        "CPU MXFP4 matmul requires CPU-resident weights");
    const int64_t rows = x.size(0);
    const int64_t outputs = weight.out;
    const int64_t width = weight.neuron_len;
    const int64_t packed_columns = width / 2;
    const int64_t scale_columns = width / 32;
    const auto * input = x.data_ptr<mfq_half>();
    const auto * values = weight.values.data_ptr<uint8_t>();
    const auto * scales = weight.scales.data_ptr<uint8_t>();
    auto result = mfq_tensor_backend::empty(
        {rows, outputs},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat16));
    auto * output = result.data_ptr<mfq_half>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    mfq_parallel_for(0, outputs, 1, [&](int64_t begin, int64_t end) {
        std::vector<float> accumulators(static_cast<size_t>(rows));
        for (int64_t neuron = begin; neuron < end; ++neuron) {
            std::fill(accumulators.begin(), accumulators.end(), 0.0f);
            for (int64_t column = 0; column < width; ++column) {
                const uint8_t packed = values[
                    neuron * packed_columns + column / 2];
                const uint8_t code = static_cast<uint8_t>(
                    (packed >> ((column & 1) * 4)) & 15u);
                float decoded = magnitude[code & 7u];
                if ((code & 8u) != 0u) decoded = -decoded;
                const uint8_t raw_scale = scales[
                    neuron * scale_columns + column / 32];
                const float weight_value = decoded * std::ldexp(
                    1.0f, static_cast<int>(raw_scale) - 127);
                for (int64_t row = 0; row < rows; ++row) {
                    accumulators[static_cast<size_t>(row)] = std::fma(
                        static_cast<float>(input[row * width + column]),
                        weight_value,
                        accumulators[static_cast<size_t>(row)]);
                }
            }
            for (int64_t row = 0; row < rows; ++row) {
                output[row * outputs + neuron] = mfq_half(
                    accumulators[static_cast<size_t>(row)]);
            }
        }
    });
    return result;
}

mfq_tensor_backend::Tensor dequant_mxfp4_cpu(const mfq::mxfp4::Data & source) {
    auto dense = mfq_tensor_backend::empty(
        {source.out, source.neuron_len},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCPU).dtype(mfq_tensor_backend::kFloat32));
    float * destination = dense.data_ptr<float>();
    static constexpr float magnitude[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    const int64_t packed_columns = source.neuron_len / 2;
    const int64_t scale_columns = source.neuron_len / 32;
    for (int64_t row = 0; row < source.out; ++row) {
        for (int64_t column = 0; column < source.neuron_len; ++column) {
            const uint8_t packed = source.values[
                static_cast<size_t>(row * packed_columns + column / 2)];
            const uint8_t code = static_cast<uint8_t>(
                (packed >> ((column & 1) * 4)) & 15u);
            const uint8_t raw_scale = source.scales[
                static_cast<size_t>(row * scale_columns + column / 32)];
            const float scale = raw_scale == 255u
                ? std::numeric_limits<float>::quiet_NaN()
                : std::ldexp(1.0f, static_cast<int>(raw_scale) - 127);
            const float value = magnitude[code & 7u] * scale;
            destination[row * source.neuron_len + column] =
                (code & 8u) == 0u ? value : -value;
        }
    }
    return dense.to(mfq_tensor_backend::kFloat16);
}

mfq::mxfp4::Data select_mxfp4_cpu_rows(
        const mfq::mxfp4::Data & source,
        const std::vector<int64_t> & rows) {
    mfq::mxfp4::Data result;
    result.out = static_cast<int64_t>(rows.size());
    result.neuron_len = source.neuron_len;
    const size_t value_row = static_cast<size_t>(source.neuron_len / 2);
    const size_t scale_row = static_cast<size_t>(source.neuron_len / 32);
    result.values.resize(rows.size() * value_row);
    result.scales.resize(rows.size() * scale_row);
    for (size_t destination = 0; destination < rows.size(); ++destination) {
        const int64_t source_row = rows[destination];
        if (source_row < 0 || source_row >= source.out) {
            throw std::runtime_error("MXFP4 selected row is out of range");
        }
        std::memcpy(
            result.values.data() + destination * value_row,
            source.values.data() + static_cast<size_t>(source_row) * value_row,
            value_row);
        std::memcpy(
            result.scales.data() + destination * scale_row,
            source.scales.data() + static_cast<size_t>(source_row) * scale_row,
            scale_row);
    }
    return result;
}

mfq::mxfp4::Data slice_mxfp4_cpu(
        const mfq::mxfp4::Data & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis == TensorParallelAxis::Output) {
        if (begin < 0 || begin >= end || end > source.out) {
            throw std::runtime_error("invalid MXFP4 output shard");
        }
        std::vector<int64_t> rows(static_cast<size_t>(end - begin));
        std::iota(rows.begin(), rows.end(), begin);
        return select_mxfp4_cpu_rows(source, rows);
    }
    if (axis != TensorParallelAxis::Input || begin < 0 || begin >= end ||
            end > source.neuron_len || begin % 32 != 0 || end % 32 != 0) {
        throw std::runtime_error(
            "MXFP4 input shards must preserve 32-value scale blocks");
    }
    mfq::mxfp4::Data result;
    result.out = source.out;
    result.neuron_len = end - begin;
    const size_t source_value_row =
        static_cast<size_t>(source.neuron_len / 2);
    const size_t destination_value_row =
        static_cast<size_t>(result.neuron_len / 2);
    const size_t source_scale_row =
        static_cast<size_t>(source.neuron_len / 32);
    const size_t destination_scale_row =
        static_cast<size_t>(result.neuron_len / 32);
    result.values.resize(static_cast<size_t>(result.out) * destination_value_row);
    result.scales.resize(static_cast<size_t>(result.out) * destination_scale_row);
    for (int64_t row = 0; row < result.out; ++row) {
        std::memcpy(
            result.values.data() + static_cast<size_t>(row) * destination_value_row,
            source.values.data() + static_cast<size_t>(row) * source_value_row +
                static_cast<size_t>(begin / 2),
            destination_value_row);
        std::memcpy(
            result.scales.data() + static_cast<size_t>(row) * destination_scale_row,
            source.scales.data() + static_cast<size_t>(row) * source_scale_row +
                static_cast<size_t>(begin / 32),
            destination_scale_row);
    }
    return result;
}
mfq::mxfp8::Data slice_mxfp8_cpu(
        const mfq::mxfp8::Data & source,
        TensorParallelAxis axis,
        int64_t begin,
        int64_t end) {
    if (axis != TensorParallelAxis::Output &&
            axis != TensorParallelAxis::Input) {
        throw std::runtime_error("MXFP8 slicing requires output or input axis");
    }
    const int64_t extent = axis == TensorParallelAxis::Output
        ? source.out : source.neuron_len;
    const int block=axis==TensorParallelAxis::Output?source.block_rows:source.block_columns;
    if (begin < 0 || begin >= end || end > extent ||
            begin % block != 0 || (end != extent && end % block != 0)) {
        throw std::runtime_error(
            "MXFP8 tensor-parallel shards must preserve scale blocks");
    }
    mfq::mxfp8::Data result;result.block_rows=source.block_rows;result.block_columns=source.block_columns;
    if (axis == TensorParallelAxis::Output) {
        result.out = end - begin;
        result.neuron_len = source.neuron_len;
        const size_t value_begin =
            static_cast<size_t>(begin * source.neuron_len);
        const size_t value_end =
            static_cast<size_t>(end * source.neuron_len);
        result.values.assign(
            source.values.begin() + static_cast<ptrdiff_t>(value_begin),
            source.values.begin() + static_cast<ptrdiff_t>(value_end));
        const int64_t scale_columns = source.neuron_len / source.block_columns;
        const size_t scale_begin =
            static_cast<size_t>((begin / source.block_rows) * scale_columns);
        const size_t scale_end = static_cast<size_t>(
            ((end + source.block_rows-1) / source.block_rows) * scale_columns);
        result.scales.assign(
            source.scales.begin() + static_cast<ptrdiff_t>(scale_begin),
            source.scales.begin() + static_cast<ptrdiff_t>(scale_end));
        return result;
    }

    result.out = source.out;
    result.neuron_len = end - begin;
    result.values.resize(
        static_cast<size_t>(result.out * result.neuron_len));
    for (int64_t row = 0; row < source.out; ++row) {
        std::memcpy(
            result.values.data() +
                static_cast<size_t>(row * result.neuron_len),
            source.values.data() +
                static_cast<size_t>(row * source.neuron_len + begin),
            static_cast<size_t>(result.neuron_len));
    }
    const int64_t source_scale_columns = source.neuron_len / source.block_columns;
    const int64_t result_scale_columns = result.neuron_len / source.block_columns;
    const int64_t scale_rows = (source.out + source.block_rows-1) / source.block_rows;
    result.scales.resize(
        static_cast<size_t>(scale_rows * result_scale_columns));
    for (int64_t row = 0; row < scale_rows; ++row) {
        std::memcpy(
            result.scales.data() +
                static_cast<size_t>(row * result_scale_columns),
            source.scales.data() + static_cast<size_t>(
                row * source_scale_columns + begin / source.block_columns),
            static_cast<size_t>(result_scale_columns));
    }
    return result;
}
Mxfp8Weight to_device_mxfp8(
        const mfq::mxfp8::Data & source,
        bool cuda,
        int device) {
    Mxfp8Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values,
        {source.out, source.neuron_len});
    result.scales = cpu_u8_tensor(
        source.scales,
        {(source.out + source.block_rows-1) / source.block_rows, source.neuron_len / source.block_columns});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}

Mxfp8Weight to_cuda_device_mxfp8(
        const mfq::mxfp8::Data & source,
        int device) {
    return to_device_mxfp8(source, true, device);
}
Mxfp4Weight to_device_mxfp4(
        const mfq::mxfp4::Data & source, bool cuda, int device) {
    Mxfp4Weight result;
    result.out = source.out;
    result.neuron_len = source.neuron_len;
    result.values = cpu_u8_tensor(
        source.values, {source.out, source.neuron_len / 2});
    result.scales = cpu_u8_tensor(
        source.scales, {source.out, source.neuron_len / 32});
    if (cuda) {
        const int target_device = device >= 0
            ? device : mfq_current_cuda_device();
        MfqCudaGuard guard(target_device);
        const auto target = mfq_tensor_backend::Device(mfq_tensor_backend::kCUDA, target_device);
        result.values = result.values.to(target, false, false).contiguous();
        result.scales = result.scales.to(target, false, false).contiguous();
    }
    return result;
}
