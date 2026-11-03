#include "gguf_row_correction.h"
#include "../../kernels/gguf_row_residual.h"
#include "gguf_weight.h"
#include "mfq_cuda_context.h"
#include "mfq_weight_memory.h"
#include <climits>

namespace tb=mfq_tensor_backend;
struct GgufRowCorrection::Storage {
    std::shared_ptr<const mfq::RowResidualFile> file;
    tb::Tensor packed,offsets,types;
    void* device_data=nullptr;
    int device=0;
};
GgufRowCorrection::GgufRowCorrection(std::shared_ptr<const mfq::RowResidualFile> file,int device,bool host) {
    if(!file || file->row_count()>INT64_MAX || file->columns()>INT64_MAX || file->payload_bytes()>INT64_MAX)
        throw std::invalid_argument("invalid GGUF row correction extent");
    storage_=std::make_shared<Storage>();storage_->file=std::move(file);storage_->device=device;
    rows_=std::int64_t(storage_->file->row_count());columns_=std::int64_t(storage_->file->columns());
    if(device<0)return;
    const mfq::cuda::WeightAllocationScope memory;
    const MfqCudaGuard guard(device);
    auto options=tb::TensorOptions().device(tb::kCPU);
    auto offsets=tb::empty({rows_},options.dtype(tb::kInt64));
    auto types=tb::empty({rows_},options.dtype(tb::kInt32));
    for(std::int64_t row=0;row<rows_;++row) {
        offsets.data_ptr<std::int64_t>()[row]=std::int64_t(storage_->file->rows()[row].offset);
        types.data_ptr<std::int32_t>()[row]=std::int32_t(storage_->file->rows()[row].type);
    }
    const auto bytes=std::max<std::int64_t>(1,std::int64_t(storage_->file->payload_bytes()));
    auto packed=tb::empty({bytes},options.dtype(tb::kUInt8).pinned_memory(host));
    storage_->file->read_payload(0,packed.data_ptr(),std::size_t(storage_->file->payload_bytes()));
    const auto target=tb::Device(tb::kCUDA,device);
    storage_->offsets=offsets.to(target);storage_->types=types.to(target);
    storage_->packed=host?std::move(packed):packed.to(target);
    if(host)TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&storage_->device_data,storage_->packed.data_ptr(),0));
    else storage_->device_data=storage_->packed.data_ptr();
}
std::shared_ptr<const GgufRowCorrection> GgufRowCorrection::slice(std::int64_t begin,std::int64_t end,
        std::int64_t column_begin,std::int64_t column_end) const {
    if(begin<0 || begin>=end || end>rows_ || column_begin<0 || column_begin>=column_end || column_end>columns_)
        throw std::out_of_range("GGUF row correction slice is out of bounds");
    auto view=std::make_shared<GgufRowCorrection>(*this);
    view->first_+=begin;view->rows_=end-begin;view->column_first_+=column_begin;view->columns_=column_end-column_begin;
    return view;
}
tb::Tensor GgufRowCorrection::add(tb::Tensor input,tb::Tensor output,tb::Tensor ids,std::int64_t experts,bool routed) const {
    if(!input.is_cuda() && !output.is_cuda() && !ids.defined() && experts==1) {
        if(input.size(-1)!=columns_ || output.size(-1)!=rows_)
            throw std::invalid_argument("CPU residual projection geometry differs");
        auto x=input.to(tb::kFloat32).contiguous(), y=output.to(tb::kFloat32).contiguous(),delta=dense(-1);
        const auto tokens=x.numel()/columns_;
        if(y.numel()!=tokens*rows_)throw std::invalid_argument("CPU residual token geometry differs");
        for(std::int64_t token=0;token<tokens;++token)for(std::int64_t row=0;row<rows_;++row) {
            float sum=0;
            for(std::int64_t column=0;column<columns_;++column)
                sum+=x.data_ptr<float>()[token*columns_+column]*delta.data_ptr<float>()[row*columns_+column];
            y.data_ptr<float>()[token*rows_+row]+=sum;
        }
        return y.to(output.scalar_type());
    }
    if(input.size(-1)!=columns_ || experts<1 || rows_%experts || input.device()!=output.device() ||
        !input.is_cuda() || input.get_device()!=storage_->device)
        throw std::invalid_argument("GGUF row correction projection geometry differs");
    const auto dtype=output.scalar_type();
    auto x=input.to(tb::kFloat32).contiguous(), y=output.to(tb::kFloat32).contiguous();
    const auto rows=rows_/experts;
    if(y.size(-1)!=rows || (ids.defined() && (ids.dim()!=2 || ids.device()!=input.device())))
        throw std::invalid_argument("GGUF residual expert/output geometry differs");
    const auto routes=ids.defined()?ids.size(1):1;
    const auto tokens=ids.defined()?ids.size(0):input.numel()/columns_;
    if(tokens<0 || routes<1 || tokens>INT64_MAX/routes ||
        y.numel()/rows!=tokens*routes || x.numel()/columns_!=(routed?tokens*routes:tokens))
        throw std::invalid_argument("GGUF residual token geometry differs");
    if(ids.defined())ids=ids.to(tb::kInt64).contiguous();
    const auto stream=mfq_get_current_cuda_stream(storage_->device);
    for(const auto& v:{x,y,ids,storage_->packed,storage_->offsets,storage_->types})
        if(v.defined())mfq_cuda_record_stream(v,stream);
    mfq::cuda::gguf::add_row_residual(storage_->device_data,
        reinterpret_cast<const std::uint64_t*>(storage_->offsets.data_ptr()),storage_->types.data_ptr<std::int32_t>(),
        first_,rows,column_first_,columns_,x.data_ptr<float>(),y.data_ptr<float>(),tokens,
        ids.defined()?ids.data_ptr<std::int64_t>():nullptr,routes,experts,routed,stream.stream());
    return y.to(dtype);
}
tb::Tensor GgufRowCorrection::add_embedding(tb::Tensor ids,tb::Tensor output) const {
    if(!output.is_cuda()) {
        auto full=dense(-1),flat=ids.to(tb::kInt64).cpu().contiguous().reshape({-1});
        auto y=output.to(tb::kFloat32).contiguous();
        if(output.size(-1)!=columns_ || flat.numel()!=output.numel()/columns_)
            throw std::invalid_argument("CPU residual lookup geometry differs");
        for(std::int64_t i=0;i<flat.numel();++i) {
            const auto row=flat.data_ptr<std::int64_t>()[i];
            if(row>=0 && row<rows_)for(std::int64_t column=0;column<columns_;++column)
                y.data_ptr<float>()[i*columns_+column]+=full.data_ptr<float>()[row*columns_+column];
        }
        return y.to(output.scalar_type());
    }
    if(!output.is_cuda() || output.get_device()!=storage_->device || output.size(-1)!=columns_)
        throw std::invalid_argument("GGUF residual embedding device/width differs");
    const auto dtype=output.scalar_type();auto y=output.to(tb::kFloat32).contiguous();
    if(ids.defined()) {
        if(ids.device()!=output.device())throw std::invalid_argument("GGUF residual embedding IDs are on another device");
        ids=ids.to(tb::kInt64).contiguous();
        if(ids.numel()!=output.numel()/columns_)throw std::invalid_argument("GGUF residual embedding ID count differs");
    }
    const auto stream=mfq_get_current_cuda_stream(storage_->device);
    for(const auto& v:{y,ids,storage_->packed,storage_->offsets,storage_->types})if(v.defined())mfq_cuda_record_stream(v,stream);
    mfq::cuda::gguf::add_residual_embedding(storage_->device_data,
        reinterpret_cast<const std::uint64_t*>(storage_->offsets.data_ptr()),storage_->types.data_ptr<std::int32_t>(),
        first_,rows_,column_first_,columns_,ids.defined()?ids.data_ptr<std::int64_t>():nullptr,
        y.data_ptr<float>(),output.numel()/columns_,stream.stream());
    return y.to(dtype);
}
tb::Tensor GgufRowCorrection::dense(int device) const {
    if(device>=0) {
        if(device!=storage_->device)throw std::invalid_argument("GGUF residual belongs to another device");
        return add_embedding({},tb::zeros({rows_,columns_},storage_->offsets.options().dtype(tb::kFloat32)));
    }
    auto out=tb::zeros({rows_,columns_},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    std::vector<float> row((storage_->file->columns()+127)/128*128);
    for(std::int64_t i=0;i<rows_;++i) {
        const auto& descriptor=storage_->file->rows().at(std::size_t(first_+i));
        if(!descriptor.type)continue;
        auto bytes=storage_->file->read_row(first_+i);
        decode_gguf_cpu(ggml_type(descriptor.type),bytes.data(),row.data(),std::int64_t(row.size()));
        std::copy_n(row.data()+column_first_,columns_,out.data_ptr<float>()+i*columns_);
    }
    return out;
}
