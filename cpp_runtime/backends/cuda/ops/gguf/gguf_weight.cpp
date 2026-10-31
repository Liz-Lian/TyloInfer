#include "gguf_weight.h"
#include "gguf_rotation.h"
#include "quant_linear.h"
#include "mfq_weight_memory.h"
#include "mfq_cuda_context.h"
#include "../../kernels/gguf.h"
#include "../../kernels/gguf_compact.h"
#include "../../kernels/gguf_mmq.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "gguf_dense_schedule.h"
#endif
#include "ggml-quants.h"

#include <algorithm>
#include <climits>
#include <cstring>
#include <limits>
#include <cstdlib>
#include <atomic>
#include <cstdio>

namespace tb = mfq_tensor_backend;

bool same_gguf_input_basis(const GgufWeight& a, const GgufWeight& b) {
    return (!a.rotation && !b.rotation) || (a.rotation && b.rotation && a.rotation->equivalent(*b.rotation));
}

GgufWeight combine_gguf_rows(const std::vector<GgufWeight*>& weights) {
    if(weights.empty() || !weights.front())throw std::invalid_argument("GGUF row combination requires weights");
    const auto& first=*weights.front();
    if(first.neuron_len<1 || !first.row_bytes || mfq::gguf::row_bytes(first.type,first.neuron_len)!=first.row_bytes)
        throw std::invalid_argument("invalid GGUF combined row encoding");
    std::int64_t rows=0;
    std::vector<tb::Tensor> images;images.reserve(weights.size());
    for(const auto* weight:weights) {
        if(!weight || weight->type!=first.type || weight->neuron_len!=first.neuron_len || weight->row_bytes!=first.row_bytes ||
                !same_gguf_input_basis(first,*weight) ||
                weight->out<1 || !weight->packed.defined() || weight->packed.device()!=first.packed.device() ||
                weight->packed.scalar_type()!=tb::kUInt8 || !weight->packed.is_contiguous() ||
                weight->out>INT64_MAX-rows || weight->row_bytes>std::uint64_t(INT64_MAX)/weight->out ||
                weight->packed.numel()!=std::int64_t(weight->row_bytes*weight->out))
            throw std::invalid_argument("incompatible GGUF row images");
        rows+=weight->out;images.push_back(weight->packed.reshape({-1}));
    }
    if(first.row_bytes>std::uint64_t(INT64_MAX)/rows)throw std::overflow_error("GGUF joined row image is too large");
    const mfq::cuda::WeightAllocationScope weight_memory;
    const auto image=tb::cat(images,0);
    GgufWeight joined{image,first.type,rows,first.neuron_len,first.row_bytes};
    joined.rotation=first.rotation;
    std::int64_t offset=0;
    for(auto* weight:weights) {
        const auto bytes=std::int64_t(weight->out*weight->row_bytes);
        weight->packed=image.narrow(0,offset,bytes);offset+=bytes;
    }
    return joined;
}

void decode_gguf_cpu(ggml_type type, const void* packed, float* output, std::int64_t columns) {
    if (type == GGML_TYPE_Q8_K)
        dequantize_row_q8_K(static_cast<const block_q8_K*>(packed), output, columns);
    else {
        const auto* traits = ggml_get_type_traits(type);
        if (!traits->to_float) throw std::invalid_argument("GGUF format has no CPU decoder");
        traits->to_float(packed, output, columns);
    }
}

GgufWeight load_gguf_weight(const mfq::ModelSource& source, const std::string& name,
    int device, bool cuda, std::int64_t row_begin, std::int64_t row_end,
    std::int64_t column_begin, std::int64_t column_end) {
    const mfq::cuda::WeightAllocationScope weight_memory;
    const auto* record = source.find_tensor(name);
    if (!record || !record->shape || record->shape->size() != 2)
        throw std::invalid_argument("GGUF projection requires a matrix: " + name);
    GgufWeight result;
    result.type = mfq::gguf::quant_type(record->dtype);
    const auto rows = record->shape->at(0), columns = record->shape->at(1);
    const auto source_stride = mfq::gguf::row_bytes(result.type, columns);
    if (rows <= 0 || source_stride > std::uint64_t(INT64_MAX) / rows ||
            record->nbytes != source_stride * rows)
        throw std::invalid_argument("invalid GGUF matrix payload: " + name);
    if (row_end == -1) row_end = rows;
    if (column_end == -1) column_end = columns;
    const auto block = ggml_blck_size(result.type);
    if (row_begin < 0 || row_begin >= row_end || row_end > rows || column_begin < 0 ||
            column_begin >= column_end || column_end > columns || column_begin % block || column_end % block)
        throw std::invalid_argument("GGUF slices must contain complete quantization blocks: " + name);
    result.out = row_end - row_begin;
    result.neuron_len = column_end - column_begin;
    result.row_bytes = mfq::gguf::row_bytes(result.type, result.neuron_len);
    result.rotation = load_gguf_input_rotation(source, name, columns, column_begin, column_end, device, cuda);
    auto host = tb::empty({static_cast<std::int64_t>(result.row_bytes * result.out)},
        tb::TensorOptions().dtype(tb::kUInt8).device(tb::kCPU));
    const auto read = source.tensor_reader(name);
    auto* destination = reinterpret_cast<std::byte*>(host.data_ptr());
    if (!column_begin && column_end == columns)
        read(row_begin * source_stride, destination, host.nbytes());
    else for (std::int64_t row = 0; row < result.out; ++row)
        read((row_begin + row) * source_stride + column_begin / block * ggml_type_size(result.type),
            destination + row * result.row_bytes, result.row_bytes);
    if (cuda) {
        MfqCudaGuard guard(device);
        result.packed = host.to(tb::Device(tb::kCUDA, device));
    } else result.packed = std::move(host);
    return result;
}

tb::Tensor GgufWeight::dequantize_stored() const {
    auto output = tb::empty({out, neuron_len}, packed.options().dtype(tb::kFloat32));
    if (packed.is_cuda()) {
        MfqCudaGuard guard(packed.get_device());
        const auto stream=mfq_get_current_cuda_stream(packed.get_device());
        mfq_cuda_record_stream(packed,stream);mfq_cuda_record_stream(output,stream);
        mfq::cuda::gguf::dequantize_rows(type, packed.data_ptr(), row_bytes, out, neuron_len,
            output.data_ptr(), false, mfq_current_cuda_stream());
    } else {
        for (std::int64_t row = 0; row < out; ++row)
            decode_gguf_cpu(type, packed.data_ptr<std::uint8_t>() + row * row_bytes,
                output.data_ptr<float>() + row * neuron_len, neuron_len);
    }
    return output;
}

tb::Tensor GgufWeight::dequantize() const {
    auto output = dequantize_stored();
    if (rotation) rotation->apply_inplace(output, true);
    return output;
}

tb::Tensor GgufWeight::embedding(tb::Tensor ids) const {
    auto shape = ids.sizes().vec();
    shape.push_back(neuron_len);
    ids = ids.to(packed.device()).to(tb::kInt64).contiguous().reshape({-1});
    if (!packed.is_cuda()) {
        auto output = tb::zeros({ids.numel(), neuron_len}, packed.options().dtype(tb::kFloat32));
        for (std::int64_t row = 0; row < ids.numel(); ++row) {
            const auto id = ids.data_ptr<std::int64_t>()[row];
            if (id < 0 || id >= out) continue;
            decode_gguf_cpu(type, packed.data_ptr<std::uint8_t>() + id * row_bytes,
                output.data_ptr<float>() + row * neuron_len, neuron_len);
        }
        return output.reshape(shape);
    }
    MfqCudaGuard guard(packed.get_device());
    auto output = tb::empty({ids.numel(), neuron_len}, packed.options().dtype(tb::kFloat32));
    const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    mfq_cuda_record_stream(packed,stream); mfq_cuda_record_stream(ids,stream); mfq_cuda_record_stream(output,stream);
    mfq::cuda::gguf::embedding_rows(type, packed.data_ptr(), row_bytes, out, neuron_len,
        ids.data_ptr<std::int64_t>(), ids.numel(), output.data_ptr<float>(), mfq_current_cuda_stream());
    if(rotation)rotation->apply_inplace(output,true);
    return output.reshape(shape);
}

GgufProjectionInput prepare_gguf_group_input(const std::vector<GgufWeight>& weights,tb::Tensor input) {
    if(weights.empty())return {};
    const auto columns=weights.front().neuron_len;
    TYLOINFER_RUNTIME_CHECK(input.dim()>=1 && columns>0 && input.size(-1)==columns && input.is_cuda(),
        "GGUF projection group input width/device mismatch");
    const auto tokens=input.numel()/columns;
    TYLOINFER_RUNTIME_CHECK(tokens<=32,"GGUF shared quantization requires a small token window");
    for(const auto& weight:weights)
        TYLOINFER_RUNTIME_CHECK(weight.neuron_len==columns && weight.packed.is_cuda() &&
            weight.packed.device()==input.device(),"GGUF projection group contains incompatible input dimensions/devices");
    const auto dtype=input.scalar_type()==tb::kBFloat16?tb::kFloat16:input.scalar_type();
    const MfqCudaGuard guard(input.get_device());
    const auto stream_handle=mfq_get_current_cuda_stream(input.get_device());
    const auto stream=stream_handle.stream();
    auto source=input.reshape({tokens,columns});
    auto original=source;
    if(source.scalar_type()!=tb::kFloat16)source=source.to(tb::kFloat32);
    source=source.contiguous();
    const auto supported=[&](const auto& weight){return !weight.prefers_original_input() && mfq::cuda::gguf::supports_interleaved_activation(weight.type,tokens);};
    const auto floating_codebook=[](const auto& weight){return weight.prefers_original_input();};
    const bool keep_original=std::any_of(weights.begin(),weights.end(),floating_codebook);
    const bool interleaved=std::any_of(weights.begin(),weights.end(),supported);
    const bool native=std::any_of(weights.begin(),weights.end(),[&](const auto& weight){return !supported(weight) && !floating_codebook(weight);});
    auto activation=native?tb::empty({std::int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(columns))},
        input.options().dtype(tb::kUInt8)):tb::Tensor{};
    auto position_major=interleaved?tb::empty({std::int64_t(mfq::cuda::gguf::interleaved_activation_bytes(tokens,columns))},
        input.options().dtype(tb::kUInt8)):tb::Tensor{};
    for(const auto& tensor:{source,activation,position_major})if(tensor.defined())mfq_cuda_record_stream(tensor,stream_handle);
    if(interleaved)
        mfq::cuda::gguf::quantize_input_interleaved(source.data_ptr(),source.scalar_type()==tb::kFloat16,tokens,columns,
            position_major.data_ptr(),activation.defined()?activation.data_ptr():nullptr,stream);
    else if(native && source.scalar_type()==tb::kFloat16)
        mfq::cuda::gguf::quantize_input_f16(source.data_ptr(),tokens,columns,activation.data_ptr(),stream);
    else if(native)mfq::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,columns,activation.data_ptr(),stream);
    return {std::move(activation),std::move(position_major),keep_original?std::move(original):tb::Tensor{},input.sizes().vec(),tokens,dtype};
}

tb::Tensor GgufProjectionInput::project(const GgufWeight& weight) const {
    TYLOINFER_RUNTIME_CHECK(!shape.empty(),"GGUF projection input was not prepared");
    const bool use_interleaved=mfq::cuda::gguf::supports_interleaved_activation(weight.type,tokens);
    auto output_shape=shape;output_shape.back()=weight.out;
    if(!tokens)return tb::empty(output_shape,weight.packed.options().dtype(dtype));
    if(weight.prefers_original_input()) {
        TYLOINFER_RUNTIME_CHECK(original.defined(),"GGUF floating codebook projection requires the original activation");
        return weight.forward(original).to(dtype).reshape(output_shape);
    }
    return weight.forward_q8(use_interleaved?interleaved:native,tokens,
        dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32,use_interleaved).to(dtype).reshape(output_shape);
}

std::vector<tb::Tensor> project_gguf_group(const std::vector<GgufWeight>& weights,tb::Tensor input) {
    if(weights.empty())return {};
    const auto prepared=prepare_gguf_group_input(weights,std::move(input));
    return prepared.project_all(weights);
}

std::vector<tb::Tensor> GgufProjectionInput::project_all(const std::vector<GgufWeight>& weights) const {
    std::vector<tb::Tensor> output;output.reserve(weights.size());
    if(weights.size()<2 || !tokens || tokens>4) {
        for(const auto& weight:weights)output.push_back(project(weight));
        return output;
    }
    const MfqCudaGuard guard(weights.front().packed.get_device());
    const auto stream=mfq_get_current_cuda_stream(weights.front().packed.get_device());
    for(const auto& packet:{native,interleaved,original})if(packet.defined())mfq_cuda_record_stream(packet,stream);
    std::vector<mfq::cuda::gguf::DenseProjection> projections;projections.reserve(weights.size());
    std::vector<std::size_t> projection_indices;projection_indices.reserve(weights.size());
    const auto result_type=dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    for(const auto& weight:weights) {
        if(weight.rotation || (weight.type>=GGML_TYPE_PTQ1_0 && weight.type<=GGML_TYPE_TQK4)) {
            output.push_back(project(weight));
            continue;
        }
        auto result=tb::empty({tokens,weight.out},weight.packed.options().dtype(result_type));
        mfq_cuda_record_stream(weight.packed,stream);mfq_cuda_record_stream(result,stream);
        projections.push_back({weight.packed.data_ptr(),weight.row_bytes,weight.out,result.data_ptr(),weight.type});
        projection_indices.push_back(output.size());
        output.push_back(std::move(result));
    }
    if(projections.empty())return output;
    const auto separate=[&] {
        for(const auto& p:projections) {
            if(mfq::cuda::gguf::supports_interleaved_activation(p.type,tokens))
                mfq::cuda::gguf::matvec_interleaved(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    interleaved.data_ptr(),tokens,p.output,result_type==tb::kFloat16,stream.stream());
            else if(result_type==tb::kFloat16)
                mfq::cuda::gguf::matvec_q8_f16(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    native.data_ptr(),tokens,p.output,stream.stream());
            else mfq::cuda::gguf::matvec_q8(p.type,p.packed,p.stride,p.rows,weights.front().neuron_len,
                    native.data_ptr(),tokens,static_cast<float*>(p.output),stream.stream());
        }
    };
    const auto batch=[&] {
        mfq::cuda::gguf::matvec_group(projections.data(),projections.size(),weights.front().neuron_len,
            native.defined()?native.data_ptr():nullptr,interleaved.defined()?interleaved.data_ptr():nullptr,
            tokens,result_type==tb::kFloat16,stream.stream());
    };
    bool selected=false;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    selected=mfq_gguf_dense_schedule::choose_batch(projections,weights.front().packed.get_device(),
        weights.front().neuron_len,tokens,result_type==tb::kFloat16,stream.stream(),separate,batch);
#endif
    if(selected)batch();else separate();
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
        TYLOINFER_CUDA_CHECK(cudaStreamIsCapturing(stream.stream(),&status));
        if(status==cudaStreamCaptureStatusActive)
            std::fprintf(stderr,"native_gguf_dense_schedule_use actual_match=1 M=%lld matrices=%zu selected=%s captured=1\n",
                static_cast<long long>(tokens),weights.size(),selected?"batch":"separate");
    }
    for(const auto i:projection_indices) {
        auto output_shape=shape;output_shape.back()=weights[i].out;
        output[i]=output[i].to(dtype).reshape(output_shape);
    }
    return output;
}

tb::Tensor GgufWeight::forward_q8(const tb::Tensor& activation,std::int64_t tokens,tb::ScalarType dtype,bool position_major) const {
    TYLOINFER_RUNTIME_CHECK(!rotation,"GGUF rotated projection requires its original activation before Q8 quantization");
    const auto bytes_per_row=mfq::cuda::gguf::q8_row_bytes(neuron_len);
    const auto bytes=position_major?mfq::cuda::gguf::interleaved_activation_bytes(tokens,neuron_len)
        :std::uint64_t(tokens)*bytes_per_row;
    TYLOINFER_RUNTIME_CHECK(packed.is_cuda() && activation.is_cuda() && activation.device()==packed.device() &&
        activation.is_contiguous() && activation.scalar_type()==tb::kUInt8 && tokens>=1 &&
        std::uint64_t(tokens)<=std::numeric_limits<std::uint64_t>::max()/bytes_per_row &&
        (dtype==tb::kFloat16 || dtype==tb::kFloat32) &&
        (!position_major || mfq::cuda::gguf::supports_interleaved_activation(type,tokens)) &&
        activation.nbytes()==bytes,"GGUF prepared activation geometry/device/type differs");
    const MfqCudaGuard guard(packed.get_device());const auto stream_handle=mfq_get_current_cuda_stream(packed.get_device());
    const auto stream=stream_handle.stream();
    auto result=tb::empty({tokens,out},packed.options().dtype(dtype));
    for(const auto& tensor:{activation,packed,result})mfq_cuda_record_stream(tensor,stream_handle);
    if(position_major)mfq::cuda::gguf::matvec_interleaved(type,packed.data_ptr(),row_bytes,out,neuron_len,
        activation.data_ptr(),tokens,result.data_ptr(),dtype==tb::kFloat16,stream);
    else if(dtype==tb::kFloat16)mfq::cuda::gguf::matvec_q8_f16(type,packed.data_ptr(),row_bytes,out,neuron_len,
        activation.data_ptr(),tokens,result.data_ptr(),stream);
    else mfq::cuda::gguf::matvec_q8(type,packed.data_ptr(),row_bytes,out,neuron_len,
        activation.data_ptr(),tokens,result.data_ptr<float>(),stream);
    return result;
}
tb::Tensor GgufWeight::forward(tb::Tensor input,bool allow_mmq) const {
    TYLOINFER_RUNTIME_CHECK(input.dim() >= 1 && input.size(-1) == neuron_len,
        "GGUF projection input width mismatch");
    auto shape = input.sizes().vec();
    shape.back() = out;
    // CUDA attention and sampling consume FP16/FP32 activations. BF16 norms
    // from a GGUF source must not propagate BF16 logits into the sampler.
    const auto dtype = input.scalar_type() == tb::kBFloat16 ? tb::kFloat16 : input.scalar_type();
    auto x = input.reshape({-1, neuron_len}).contiguous();
    if(rotation)x=rotation->apply(x);
    if (!packed.is_cuda())
        return tb::matmul(x.to(tb::kFloat32), dequantize_stored().transpose(0, 1)).to(dtype).reshape(shape);
    MfqCudaGuard guard(packed.get_device());
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.get_device() == packed.get_device(),
        "GGUF weight and input must share a device");
    const auto tokens = x.size(0);
    const auto output_dtype=dtype==tb::kFloat16?tb::kFloat16:tb::kFloat32;
    auto output = tb::empty({tokens, out}, packed.options().dtype(output_dtype));
    const auto stream=mfq_get_current_cuda_stream(packed.get_device());
    mfq_cuda_record_stream(packed,stream); mfq_cuda_record_stream(output,stream);
    // Trellis weights decode to floating-point codebook entries. Preserve the
    // original activation instead of requantizing it to Q8 for a float dot.
    // Large prefill batches retain the tensor-core path below.
    if (type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4 && tokens <= 32) {
        auto source = x.scalar_type() == tb::kFloat16 ? x : x.to(tb::kFloat32).contiguous();
        mfq_cuda_record_stream(source,stream);
        mfq::cuda::gguf::compact_project(mfq::cuda::gguf::CompactFormat(type), packed.data_ptr(), row_bytes,
            out, neuron_len, source.data_ptr(), source.scalar_type() == tb::kFloat16
                ? mfq::cuda::gguf::CompactInput::Float16 : mfq::cuda::gguf::CompactInput::Float32,
            tokens, output.data_ptr(), output_dtype == tb::kFloat16, mfq_current_cuda_stream());
        return output.to(dtype).reshape(shape);
    }
    if (tokens <= 32) {
        auto source=x.scalar_type()==tb::kFloat16?x:x.to(tb::kFloat32).contiguous();
        mfq_cuda_record_stream(source,stream);
        const bool interleaved=mfq::cuda::gguf::supports_interleaved_activation(type,tokens);
        auto activation = tb::empty({static_cast<std::int64_t>(interleaved
                ?mfq::cuda::gguf::interleaved_activation_bytes(tokens,neuron_len)
                :tokens * mfq::cuda::gguf::q8_row_bytes(neuron_len))},
            packed.options());
        mfq_cuda_record_stream(activation,stream);
        if(interleaved)
            mfq::cuda::gguf::quantize_input_interleaved(source.data_ptr(),source.scalar_type()==tb::kFloat16,
                tokens,neuron_len,activation.data_ptr(),nullptr,mfq_current_cuda_stream());
        else if(source.scalar_type()==tb::kFloat16)
            mfq::cuda::gguf::quantize_input_f16(source.data_ptr(),tokens,neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        else mfq::cuda::gguf::quantize_input(source.data_ptr<float>(),tokens,neuron_len,activation.data_ptr(),mfq_current_cuda_stream());
        if(interleaved)
            mfq::cuda::gguf::matvec_interleaved(type,packed.data_ptr(),row_bytes,out,neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),output_dtype==tb::kFloat16,mfq_current_cuda_stream());
        else if(output_dtype==tb::kFloat16)
            mfq::cuda::gguf::matvec_q8_f16(type,packed.data_ptr(),row_bytes,out,neuron_len,
                activation.data_ptr(),tokens,output.data_ptr(),mfq_current_cuda_stream());
        else mfq::cuda::gguf::matvec_q8(type,packed.data_ptr(),row_bytes,out,neuron_len,
            activation.data_ptr(),tokens,output.data_ptr<float>(),mfq_current_cuda_stream());
    } else {
        namespace mmq=mfq::cuda::gguf::mmq;
        const auto* mmq_setting=std::getenv("TYLOINFER_GGUF_DENSE_MMQ");
        const auto raw_mmq_rows=std::max<std::int64_t>(1,(16LL<<20)/row_bytes);
        const auto mmq_rows=std::min(out,raw_mmq_rows>=128?raw_mmq_rows/128*128:raw_mmq_rows);
        const auto final_rows=mmq_rows>0?out%mmq_rows:0;
        if(allow_mmq && (!mmq_setting || mmq_setting[0]!='0') && mmq_rows>0 && tokens<=INT_MAX && neuron_len<=INT_MAX &&
                (dtype==tb::kFloat16 || dtype==tb::kFloat32) && mmq::fits(type,mmq_rows,packed.get_device()) &&
                (!final_rows || mmq::fits(type,final_rows,packed.get_device()))) {
            auto input_float=x.to(tb::kFloat32).contiguous();
            tb::Tensor exponents,scaled;
            if(dtype==tb::kFloat32) {
                scaled=tb::empty_like(input_float);
                exponents=tb::empty({tokens},packed.options().dtype(tb::kInt32));
                mfq::cuda::gguf::pack_scaled_f32(input_float.data_ptr<float>(),scaled.data_ptr<float>(),
                    exponents.data_ptr<std::int32_t>(),tokens,neuron_len,mfq_current_cuda_stream());
                input_float=scaled;
            }
            auto activation=tb::empty({std::int64_t(mmq::activation_bytes(tokens,neuron_len))},packed.options());
            mmq::quantize(input_float.data_ptr<float>(),nullptr,activation.data_ptr(),int(type),neuron_len,tokens,mfq_current_cuda_stream());
            auto result=output_dtype==tb::kFloat32?output:tb::empty({tokens,out},packed.options().dtype(tb::kFloat32));
            const auto guard_bytes=std::size_t(128)*ggml_type_size(type);
            auto weight=tb::empty({std::int64_t(mmq_rows*row_bytes+guard_bytes)},packed.options());
            for(const auto& tensor:{input_float,exponents,activation,result,weight})if(tensor.defined())mfq_cuda_record_stream(tensor,stream);
            mmq::Context context;
            for(std::int64_t begin=0;begin<out;begin+=mmq_rows) {
                const auto count=std::min(mmq_rows,out-begin);
                const auto bytes=std::uint64_t(count)*row_bytes;
                TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(weight.data_ptr(),packed.data_ptr<std::uint8_t>()+begin*row_bytes,
                    bytes,cudaMemcpyDeviceToDevice,mfq_current_cuda_stream()));
                TYLOINFER_CUDA_CHECK(cudaMemsetAsync(weight.data_ptr<std::uint8_t>()+bytes,0,guard_bytes,mfq_current_cuda_stream()));
                context.run({weight.data_ptr(),int(type),count,neuron_len,std::size_t(bytes),1,activation.data_ptr(),nullptr,nullptr,
                    tokens,tokens,result.data_ptr<float>()+begin,out},mfq_current_cuda_stream());
            }
            if(exponents.defined())mfq::cuda::gguf::rescale_f32(result.data_ptr<float>(),exponents.data_ptr<std::int32_t>(),
                tokens,out,mfq_current_cuda_stream());
            if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
                static std::atomic<bool> reported{false};
                if(!reported.exchange(true,std::memory_order_relaxed))std::fprintf(stderr,
                    "native_gguf_dense_mmq actual_match=1 format=%d tokens=%lld rows=%lld columns=%lld scaled_fp32=%d\n",
                    int(type),static_cast<long long>(tokens),static_cast<long long>(out),static_cast<long long>(neuron_len),int(exponents.defined()));
            }
            return result.to(dtype).reshape(shape);
        }
        // Prefill shares each decoded tile across every token. Temporary weight
        // expansion is bounded and never changes permanent compressed storage.
        const bool half = dtype == tb::kFloat16;
        const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_TENSORCORE");
        const bool scaled_half=!half && dtype==tb::kFloat32 && (!setting || setting[0]!='0') &&
            mfq::cuda::gguf::grouped_f16_available(packed.get_device());
        x = x.to(half ? tb::kFloat16 : tb::kFloat32).contiguous();
        const auto raw_rows=std::max<std::int64_t>(1,(16LL<<20)/(neuron_len*(half?2:4)));
        // Keep GEMM leading dimensions aligned without growing the bounded
        // decoded workspace. The final output slice still handles arbitrary N.
        const auto tile_rows=std::min(out,raw_rows>=32?raw_rows/32*32:raw_rows);
        auto tile = tb::empty({tile_rows, neuron_len}, x.options());
        mfq_cuda_record_stream(tile,stream);
        tb::Tensor scaled_x,scaled_weight,x_exponents,weight_exponents;
        mfq::cuda::BlasHandle* blas=nullptr;
        if(scaled_half) {
            scaled_x=tb::empty(x.sizes(),x.options().dtype(tb::kFloat16));
            scaled_weight=tb::empty({tile_rows,neuron_len},scaled_x.options());
            x_exponents=tb::empty({tokens},x.options().dtype(tb::kInt32));
            weight_exponents=tb::empty({tile_rows},x_exponents.options());
            for(const auto& tensor:{scaled_x,scaled_weight,x_exponents,weight_exponents})mfq_cuda_record_stream(tensor,stream);
            mfq::cuda::gguf::pack_scaled_f16(x.data_ptr<float>(),scaled_x.data_ptr(),x_exponents.data_ptr<std::int32_t>(),
                tokens,neuron_len,mfq_current_cuda_stream());
            blas=&mfq::cuda::default_context(packed.get_device())->blas();blas->set_stream(mfq_current_cuda_stream());
        }
        for (std::int64_t begin = 0; begin < out; begin += tile_rows) {
            const auto count = std::min(tile_rows, out - begin);
            auto weight = tile.slice(0, 0, count);
            mfq::cuda::gguf::dequantize_rows(type,
                packed.data_ptr<std::uint8_t>() + begin * row_bytes, row_bytes,
                count, neuron_len, weight.data_ptr(), half, mfq_current_cuda_stream());
            if(scaled_half) {
                mfq::cuda::gguf::pack_scaled_f16(weight.data_ptr<float>(),scaled_weight.data_ptr(),weight_exponents.data_ptr<std::int32_t>(),
                    count,neuron_len,mfq_current_cuda_stream());
                auto result=tb::empty({tokens,count},x.options());mfq_cuda_record_stream(result,stream);
                mfq::cuda::gguf::gemm_scaled_f16(scaled_x.data_ptr(),scaled_weight.data_ptr(),x_exponents.data_ptr<std::int32_t>(),
                    weight_exponents.data_ptr<std::int32_t>(),result.data_ptr<float>(),tokens,count,neuron_len,blas->get(),mfq_current_cuda_stream());
                output.slice(1,begin,begin+count).copy_(result);
            } else output.slice(1, begin, begin + count).copy_(dense_projection(x, weight).to(output_dtype));
        }
    }
    return output.to(dtype).reshape(shape);
}
