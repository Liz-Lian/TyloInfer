#include "float_projection.h"
#include "gguf_weight.h"
#include "tyloinfer/gguf_model_source.h"
#include "gguf_cpu.h"
#include "tyloinfer/cpu_expert_pool.h"
#include "tyloinfer/host_memory.h"
#include "../ops/gguf/gguf_cpu_executor.h"
#include "../ops/gguf/gguf_adaptive_slots.h"
#include "../ops/gguf/gguf_host_snapshot.h"
#include "../ops/gguf/gguf_decode_dma.h"
#include "../ops/gguf/gguf_rotation.h"
#include "../ops/gguf/gguf_residual.h"
#include "../ops/gguf/gguf_row_correction.h"
#include "../../../tests/row_residual_fixture.h"
#include "../kernels/gguf_residual.h"
#include "qwen4_exp.h"
#include "nlohmann/json.hpp"
#include <bit>
#include "../kernels/gguf.h"
#include "../kernels/gguf_quad_storage.h"
#include "../kernels/gguf_quad_decode.h"
#include "../kernels/gguf_m1_residual.h"
#include "../kernels/gguf_compact.h"
#include "../kernels/gguf_mmq.h"
#include "cuda_execution.h"
#include "quant_linear.h"
#include "storage/weight_loader.h"
#include "storage/pinned_copy_plan.h"
#include "storage/moe_pinned_expert_arena.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "runtime/decode_window.h"
#include "runtime/moe_pipeline.h"
#include "runtime/native_dma_batch.h"
#include "ggml-quants.h"
#include "mfq_cuda_resident_allocations.h"
#include "mfq_cuda_process_memory.h"
#include "mfq_weight_memory.h"
#include "recurrent_copy_batch_test.h"
#include "gguf_cpu_activation_test.h"
#include "gguf_cpu_quantize_test.h"
#include "gguf_input_test.h"
#include "gguf_rotation_test.h"
#include "gguf_resident_prepare_test.h"
#include "gguf_speculative_journal_test.h"
#include <cuda.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <atomic>
#include <chrono>
#include <cstdlib>

void run_gguf_iq_codebooks();

namespace tb = mfq_tensor_backend;
namespace weight_loader=mfq::cuda::weight_loader;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void run_fused_expert_reduce() {
    int can_map=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,0));
    for(int tokens:{1,2,4})for(int routes:{1,3,10,33})for(int columns:{7,257})for(bool half:{false,true}) {
        const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
        auto host=tb::empty({tokens,routes,columns},options),weights=tb::empty({tokens,routes},options);
        for(int i=0;i<host.numel();++i)host.data_ptr<float>()[i]=5.f*std::sin(float(i)*.17f);
        for(int i=0;i<weights.numel();++i)weights.data_ptr<float>()[i]=.1f*std::cos(float(i)*.3f);
        const auto pairs=host.to(tb::kCUDA),coefficients=weights.to(tb::kCUDA);
        const auto dtype=half?tb::kFloat16:tb::kFloat32;
        auto actual=tb::empty({tokens,columns},pairs.options().dtype(dtype));
        for(int cpu=0;cpu<=(can_map?1:0);++cpu) {
            std::unique_ptr<mfq::cuda::HostBuffer> buffer;void* mapped=nullptr;
            tb::Tensor device_mask;
            const auto values_offset=(64+tokens*routes*sizeof(std::int32_t)+127)&~std::size_t(127);
            if(cpu) {
                buffer=std::make_unique<mfq::cuda::HostBuffer>(values_offset+host.nbytes(),true);
                TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,buffer->data(),0));
                auto* mask=reinterpret_cast<std::int32_t*>(static_cast<std::uint8_t*>(buffer->data())+64);
                auto host_mask=tb::empty({tokens*routes},options.dtype(tb::kInt32));
                auto* values=reinterpret_cast<float*>(static_cast<std::uint8_t*>(buffer->data())+values_offset);
                for(int slot=0;slot<tokens*routes;++slot) {
                    mask[slot]=slot%3==0;
                    host_mask.data_ptr<std::int32_t>()[slot]=mask[slot];
                    for(int col=0;col<columns;++col) {
                        const auto at=slot*columns+col;values[at]=-2.f*host.data_ptr<float>()[at]+.1f;
                        if(mask[slot])host.data_ptr<float>()[at]=values[at];
                    }
                }
                *static_cast<std::uint32_t*>(buffer->data())=1;
                device_mask=host_mask.to(tb::kCUDA);
            }
            const auto* base=static_cast<const std::uint8_t*>(mapped);
            mfq::cuda::gguf::reduce_moe_output(pairs.data_ptr<float>(),coefficients.data_ptr<float>(),tokens,routes,columns,
                half,actual.data_ptr(),cpu?reinterpret_cast<const float*>(base+values_offset):nullptr,
                cpu?device_mask.data_ptr<std::int32_t>():nullptr,cpu?reinterpret_cast<const std::uint32_t*>(base):nullptr,
                mfq_current_cuda_stream());
            const auto expected=moe_weighted_reduce_cuda(host.to(dtype).to(tb::kCUDA),coefficients);
            require((actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>()==0,
                "fused GGUF expert reduction changed FP16 rounding, CPU slots or output tails");
        }
    }
}
void run_resident_pool_separation() {
#if CUDA_VERSION >= 11030
    const auto context=mfq::cuda::default_context(0);
    if(!context->supports_async_allocations())return;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8);
    tb::Tensor permanent,routed,temporary;
    {const mfq::cuda::ResidentAllocationScope scope;permanent=tb::empty({65536},options);}
    {
        const mfq::cuda::RoutedWeightAllocationScope scope;
        routed=tb::empty({65536},options);
        {const mfq::cuda::RoutedWeightAllocationScope scratch(false);temporary=tb::empty({65536},options);}
    }
    const auto pool=[](const tb::Tensor& tensor) {
        CUmemoryPool value=nullptr;
        require(cuPointerGetAttribute(&value,CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
            CUdeviceptr(reinterpret_cast<std::uintptr_t>(tensor.data_ptr())))==CUDA_SUCCESS && value,
            "CUDA async allocation did not report its pool");
        return value;
    };
    require(pool(permanent)!=pool(routed) && pool(routed)!=pool(temporary) && pool(permanent)!=pool(temporary),
        "Routed weights shared allocator pages with model weights or temporary storage");
#endif
}
class Source final : public mfq::ModelSource {
    std::vector<mfq::TensorMetadata> records_;
    std::map<std::string, std::shared_ptr<std::vector<std::byte>>> images_;
    std::shared_ptr<std::atomic<std::size_t>> read_count_=std::make_shared<std::atomic<std::size_t>>(0);
    std::shared_ptr<std::atomic<std::size_t>> batch_count_=std::make_shared<std::atomic<std::size_t>>(0);
    std::shared_ptr<bool> fail_batch_=std::make_shared<bool>(false);
    std::shared_ptr<std::atomic<bool>> fail_read_=std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::atomic<bool>> fail_background_=std::make_shared<std::atomic<bool>>(false);
    std::shared_ptr<std::atomic<std::size_t>> fail_skip_=std::make_shared<std::atomic<std::size_t>>(0);
    std::map<std::string,std::shared_ptr<const mfq::RowResidualFile>> corrections_;
    bool batched_=false,mapped_=false,parallel_=false;
    std::unordered_map<std::string,std::string> metadata_;
public:
    bool supports_parallel_tensor_reads()const noexcept override{return parallel_;}
    void enable_parallel_reads(){parallel_=true;}
    void add_correction(const std::string& name,const std::filesystem::path& path) {
        corrections_[name]=std::make_shared<mfq::RowResidualFile>(path);
    }
    std::shared_ptr<const mfq::RowResidualFile> row_residual(std::string_view name) const override {
        const auto found=corrections_.find(std::string(name));
        return found==corrections_.end()?nullptr:found->second;
    }
    void add_hadamard(const std::string& name, int block=128, bool identity=false, int salt=0) {
        const auto* record=find_tensor(name);
        require(record && record->shape && record->shape->size()>=2,"Hadamard fixture tensor missing");
        std::vector<int> signs;
        if(!identity)for(int64_t i=0;i<record->shape->back();++i)signs.push_back((i*17+i/7+salt)%3?-1:1);
        auto transforms=metadata_.count("gguf_hadamard")?nlohmann::json::parse(metadata_.at("gguf_hadamard")):nlohmann::json::object();
        transforms[name]={{"block_size",block},{"signs",signs}};
        metadata_["gguf_hadamard"]=transforms.dump();metadata_["gguf_requires_hadamard"]="true";
    }
    std::size_t read_calls()const {return read_count_->load();}
    void enable_batches(){batched_=true;}
    void enable_mapping(){mapped_=true;}
    void fail_next_batch(){*fail_batch_=true;}
    void fail_next_read(bool background_only=false,std::size_t skip=0){fail_background_->store(background_only);fail_skip_->store(skip);fail_read_->store(true);}
    void clear_read_failure(){fail_read_->store(false);}
    std::size_t batch_calls()const{return batch_count_->load();}
    void large_q4_scale(const std::string& name) {
        const std::uint16_t scale=0x7bff;
        auto& image=*images_.at(name);
        for(std::size_t i=0;i<image.size();i+=sizeof(block_q4_0))std::memcpy(image.data()+i,&scale,sizeof(scale));
    }
    void full_q2_codes(const std::string& name) {
        auto& image=*images_.at(name);
        for(std::size_t i=0;i<image.size();++i)image[i]=std::byte(uint8_t(i*29+i/7+17));
        const float scales[]={0.f,1.f/1024,-1.f/512,1.f/2048};
        for(std::size_t i=0;i<image.size();i+=sizeof(block_q2_K)) {
            const auto block=i/sizeof(block_q2_K);
            for(int j=0;j<16;++j)image[i+offsetof(block_q2_K,scales)+j]=
                std::byte(uint8_t(((block+j*5)%16)|(((block*3+j*7)%16)<<4)));
            const auto d=ggml_fp32_to_fp16(scales[block%4]);
            const auto minimum=ggml_fp32_to_fp16(scales[(block+1)%4]);
            std::memcpy(image.data()+i+offsetof(block_q2_K,d),&d,sizeof(d));
            std::memcpy(image.data()+i+offsetof(block_q2_K,dmin),&minimum,sizeof(minimum));
        }
    }
    void full_q6_codes(const std::string& name) {
        auto& image=*images_.at(name);
        for(std::size_t i=0;i<image.size();++i)image[i]=std::byte(uint8_t(i*29+i/7+17));
        const float scales[]={0.f,1.f/1024,-1.f/512,1.f/2048};
        for(std::size_t i=0;i<image.size();i+=sizeof(block_q6_K)) {
            const auto scale=ggml_fp32_to_fp16(scales[(i/sizeof(block_q6_K))%4]);
            std::memcpy(image.data()+i+offsetof(block_q6_K,d),&scale,sizeof(scale));
        }
    }
    void full_iq3_codes(const std::string& name) {
        auto& image=*images_.at(name);
        for(std::size_t i=0;i<image.size();++i)image[i]=std::byte(uint8_t(i*29+i/7+17));
        const uint16_t scale=0x3000;
        for(std::size_t i=0;i<image.size();i+=ggml_type_size(GGML_TYPE_IQ3_S))
            std::memcpy(image.data()+i,&scale,sizeof(scale));
    }
    void full_compact_codes(const std::string& name, ggml_type type) {
        auto& image = *images_.at(name);
        for (std::size_t i = 0; i < image.size(); ++i) image[i] = std::byte(uint8_t(i * 29 + i / 7 + 17));
        const uint16_t scale = 0x3000;
        for (std::size_t i = 0; i < image.size(); i += ggml_type_size(type))
            std::memcpy(image.data() + i + (type == GGML_TYPE_PTQ1_0 ? 26 : 0), &scale, sizeof(scale));
    }
    void add(const std::string& name, const std::vector<std::int64_t>& shape, ggml_type type = GGML_TYPE_Q4_0) {
        const auto stride = mfq::gguf::row_bytes(type, shape.back());
        std::size_t bytes = stride;
        for (std::size_t i = 0; i + 1 < shape.size(); ++i) bytes *= shape[i];
        auto image = std::make_shared<std::vector<std::byte>>(bytes);
        for (std::size_t i = 0; i < bytes; ++i) (*image)[i] = std::byte((i * 29 + i / 11 + 7) & 63);
        images_.emplace(name, image);
        auto dtype = "GGUF_" + std::string(ggml_type_name(type));
        std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](char c) {
            return c >= 'a' && c <= 'z' ? char(c - 'a' + 'A') : c;
        });
        records_.push_back({name, dtype, dtype, shape, bytes});
    }
    void add_dense(const std::string& name,const std::vector<std::int64_t>& shape) {
        std::size_t count=1;for(auto extent:shape)count*=extent;
        std::vector<float> values(count);
        for(std::size_t i=0;i<count;++i)values[i]=std::sin(float(i)*.173f)*.03f;
        const auto payload=tb::tensor(values).to(tb::kBFloat16).contiguous();
        const std::uint32_t rank=std::uint32_t(shape.size());
        const auto header=sizeof(rank)+shape.size()*sizeof(int64_t);
        auto image=std::make_shared<std::vector<std::byte>>(header+payload.nbytes());
        std::memcpy(image->data(),&rank,sizeof(rank));
        std::memcpy(image->data()+sizeof(rank),shape.data(),shape.size()*sizeof(int64_t));
        std::memcpy(image->data()+header,payload.data_ptr(),payload.nbytes());
        images_.emplace(name,image);
        records_.push_back({name,"BF16","BF16",shape,image->size()});
    }
    const std::vector<std::filesystem::path>& source_paths() const noexcept override {
        static const std::vector<std::filesystem::path> value; return value;
    }
    std::string_view architecture() const noexcept override { return "test"; }
    const std::unordered_map<std::string, std::string>& metadata() const noexcept override {
        return metadata_;
    }
    const std::vector<mfq::TensorMetadata>& tensors() const noexcept override { return records_; }
    const mfq::TensorMetadata* find_tensor(std::string_view name) const noexcept override {
        for (const auto& record : records_) if (record.name == name) return &record;
        return nullptr;
    }
    TensorReader tensor_reader(std::string_view name) const override {
        const auto image = images_.at(std::string(name));
        return [image,reads=read_count_,fail=fail_read_,background=fail_background_,skip=fail_skip_,caller=std::this_thread::get_id()](std::uint64_t offset, std::byte* target, std::size_t count) {
            if((!background->load() || caller!=std::this_thread::get_id()) && fail->load()) {
                auto remaining=skip->load();
                while(remaining && !skip->compare_exchange_weak(remaining,remaining-1)){}
                if(!remaining && fail->exchange(false))throw std::runtime_error("fixture single read failure");
            }
            if (offset > image->size() || count > image->size() - offset)
                throw std::out_of_range("GGUF fixture read");
            ++*reads;std::memcpy(target, image->data() + offset, count);
        };
    }
    TensorReader mapped_tensor_reader(std::string_view name) const override {
        return mapped_?tensor_reader(name):TensorReader{};
    }
    TensorBatchReader tensor_batch_reader(std::string_view name) const override {
        if(!batched_)return {};
        const auto read=tensor_reader(name);
        return [read,batches=batch_count_,fail=fail_batch_](const std::vector<mfq::ReadSpan>& spans) {
            ++*batches;
            for(const auto& span:spans) {
                read(span.offset,span.destination,span.size);
                if(*fail){*fail=false;throw std::runtime_error("fixture batch read failure");}
            }
        };
    }
    void read_range_into(std::string_view name, std::uint64_t offset, std::byte* target,
        std::size_t count) const override { tensor_reader(name)(offset, target, count); }
    const std::vector<std::string>& assets() const noexcept override {
        static const std::vector<std::string> value; return value;
    }
    bool has_asset(std::string_view) const noexcept override { return false; }
    std::vector<std::byte> read_asset(std::string_view) const override { return {}; }
    std::optional<mfq::ModelGraph> model_graph() const override { return {}; }
    tb::Tensor decoded(const std::string& name) const {
        const auto& shape = *find_tensor(name)->shape;
        auto result = tb::empty(shape, tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU));
        const auto columns = shape.back(), rows = result.numel() / columns;
        const auto type = mfq::gguf::quant_type(find_tensor(name)->dtype);
        const auto stride = mfq::gguf::row_bytes(type, columns);
        for (std::int64_t row = 0; row < rows; ++row)
            decode_gguf_cpu(type, images_.at(name)->data() + row * stride,
                result.data_ptr<float>() + row * columns, columns);
        return result;
    }
};

void run_cpu_executor_queue() {
    Source source;source.add("queue",{256,256},GGML_TYPE_IQ3_XXS);
    source.add("down",{256,256},GGML_TYPE_IQ4_NL);
    auto image=std::make_shared<std::vector<std::byte>>(source.find_tensor("queue")->nbytes);
    source.tensor_reader("queue")(0,image->data(),image->size());
    auto down=std::make_shared<std::vector<std::byte>>(source.find_tensor("down")->nbytes);
    source.tensor_reader("down")(0,down->data(),down->size());
    struct Job {
        std::array<float,256> input{},output{};
        std::uint32_t ready=0;
        std::shared_ptr<GgufCpuWork> work=std::make_shared<GgufCpuWork>();
    };
    const auto make=[&](bool fail=false) {
        auto job=std::make_shared<Job>();auto& w=*job->work;
        for(int i=0;i<256;++i){job->input[i]=.1f*std::sin(i*.173f);job->output[i]=-13.f;}
        w.tokens=1;w.routes=1;w.input_width=w.hidden_width=w.output_width=256;w.half=true;
        w.input=fail?nullptr:job->input.data();w.output=job->output.data();w.ready=&job->ready;
        GgufCpuExpert expert;expert.slots={0};
        for(int i=0;i<3;++i) {
            const auto& owner=i==2?down:image;const auto type=i==2?GGML_TYPE_IQ4_NL:GGML_TYPE_IQ3_XXS;
            expert.owners[i]=owner;expert.projections[i]={reinterpret_cast<const uint8_t*>(owner->data()),
                mfq::gguf::row_bytes(type,256),int(type),256,256};
        }
        w.experts.push_back(std::move(expert));return job;
    };
    const auto initial=mfq::cuda::tensor_host_bytes.load();
    std::vector<std::shared_ptr<Job>> pending;
    auto failed=make(true);
    {
        GgufCpuExecutor executor({});
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        for(int i=0;i<500;++i)executor.keep_awake();
        executor.submit(failed->work);
        for(int i=0;i<32;++i){pending.push_back(make());executor.keep_awake();executor.submit(pending.back()->work);}
        // Destruction must wake the parked thread and drain every accepted
        // job, including recovery after the malformed work item above.
    }
    bool propagated=false;
    try{failed->work->future.get();}catch(const std::invalid_argument&){propagated=true;}
    require(propagated && failed->ready==1,"CPU queue failure did not publish completion");
    for(const auto& job:pending) {
        job->work->future.get();require(job->ready==1,"CPU queue lost a completion flag");
        for(int i=0;i<256;++i)require(std::isfinite(job->output[i]) && job->output[i]==pending.front()->output[i],
            "CPU queue changed repeated expert output");
    }
    require(mfq::cuda::tensor_host_bytes.load()==initial,"CPU queue retained charged workspaces after shutdown");
    std::cout<<"GGUF CPU activity wake/idle/destructor drain/failure recovery 32 real expert jobs PASS\n";
}

tb::Tensor input(std::int64_t tokens, std::int64_t routes, bool routed) {
    auto result = tb::empty(routed ? std::vector<std::int64_t>{tokens, routes, 256} :
        std::vector<std::int64_t>{tokens, 256}, tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU));
    for (std::int64_t i = 0; i < result.numel(); ++i) result.data_ptr<float>()[i] = std::sin(float(i) * .17f) * .1f;
    return result.to(tb::kCUDA);
}
tb::Tensor routes(std::int64_t tokens, int rotation) {
    auto result = tb::empty({tokens, 3}, tb::TensorOptions().dtype(tb::kInt64).device(tb::kCPU));
    for (std::int64_t i = 0; i < result.numel(); ++i) result.data_ptr<std::int64_t>()[i] = (i + rotation) % 3;
    return result.to(tb::kCUDA);
}
void check_projection(const tb::Tensor& actual, const tb::Tensor& x, const tb::Tensor& ids,
    const tb::Tensor& bank) {
    const auto output = actual.to(tb::kFloat32).cpu(), source = x.cpu(), selected = ids.cpu();
    for (std::int64_t slot = 0; slot < selected.numel(); ++slot) for (int row = 0; row < 7; ++row) {
        const auto expert = selected.data_ptr<std::int64_t>()[slot];
        const auto* values = bank.data_ptr<float>() + (expert * 7 + row) * 256;
        const auto* activation = source.data_ptr<float>() + (x.dim() == 3 ? slot : slot / 3) * 256;
        double expected = 0., magnitude = 0.;
        for (int k = 0; k < 256; ++k) { const double v = double(values[k]) * activation[k]; expected += v; magnitude += std::abs(v); }
        require(std::abs(output.data_ptr<float>()[slot * 7 + row] - expected) < magnitude * .008 + 1e-6,
            "GGUF routed runtime changed the projection");
    }
}
void check_moe(const tb::Tensor& actual, const GgufMoeBanks& banks, const tb::Tensor& x,
    const tb::Tensor& ids, const tb::Tensor& weights, float limit = 0) {
    auto g = banks.gate->forward(x, ids), u = banks.up->forward(x, ids);
    if (limit > 0) { g = tb::clamp_max(g.to(tb::kFloat32), limit); u = tb::clamp(u.to(tb::kFloat32), -limit, limit); }
    auto hidden = ((g * tb::sigmoid(g)) * u).to(x.scalar_type());
    auto expected = moe_weighted_reduce_cuda(banks.down->forward(hidden, ids), weights);
    const auto error = (actual.to(tb::kFloat32) - expected.to(tb::kFloat32)).abs().max().item<float>();
    if (!std::isfinite(error) || error >= 1e-5f + expected.abs().max().item<float>() * 1e-4f)
        std::cerr << "combined tokens=" << x.size(0) << " dtype=" << int(x.scalar_type())
            << " limit=" << limit << " error=" << error << " max=" << expected.abs().max().item<float>() << '\n';
    require(std::isfinite(error) && error < 1e-5f + expected.abs().max().item<float>() * 1e-4f,
        "GGUF combined MoE changed Gate/Up/Down or routing weights");
}
tb::Tensor mmq_quantized_reference(tb::Tensor input) {
    auto host=input.to(tb::kFloat32).cpu().contiguous();
    const auto columns=host.size(-1),rows=host.numel()/columns;
    auto* data=host.data_ptr<float>();
    for(int64_t row=0;row<rows;++row)for(int64_t first=0;first<columns;first+=32) {
        const auto count=std::min(int64_t(32),columns-first);
        float maximum=0.f;
        for(int64_t k=0;k<count;++k)maximum=std::max(maximum,std::abs(data[row*columns+first+k]));
        const float inverse=maximum>0?127.f/maximum:0.f,scale=inverse>0?1.f/inverse:0.f;
        for(int64_t k=0;k<count;++k)data[row*columns+first+k]=std::round(data[row*columns+first+k]*inverse)*scale;
    }
    return host.to(input.device());
}
void run_mmq_products() {
    namespace mmq=mfq::cuda::gguf::mmq;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    for(auto format:{GGML_TYPE_Q2_0,GGML_TYPE_Q2_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,
            GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS,GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q5_1})
    for(int rows:{17,256})for(int columns:{format==GGML_TYPE_Q2_0?640:768,1024}) {
        if(!mmq::fits(format,rows,mfq_current_cuda_device()))continue;
        Source source;source.add("w",{2*rows,columns},format);
        auto w=load_gguf_weight(source,"w",mfq_current_cuda_device());
        const auto guard=128*ggml_type_size(format);
        auto packed=tb::zeros({w.packed.numel()+int64_t(guard)},w.packed.options());
        packed.narrow(0,0,w.packed.numel()).copy_(w.packed);
        auto x=tb::empty({17,columns},options.device(tb::kCPU));
        for(int64_t i=0;i<x.numel();++i)x.data_ptr<float>()[i]=.1f*std::sin(float(i)*.17f);
        x=x.to(tb::kCUDA);
        auto ids=tb::arange(40,options.dtype(tb::kInt32)).remainder(17).to(tb::kInt32);
        auto output_ids=(tb::arange(40+mmq::id_guard,options.dtype(tb::kInt32))*17).remainder(40).to(tb::kInt32).narrow(0,0,40);
        auto bounds=tb::empty({3},options.device(tb::kCPU).dtype(tb::kInt32));
        bounds.data_ptr<int32_t>()[0]=0;bounds.data_ptr<int32_t>()[1]=29;bounds.data_ptr<int32_t>()[2]=40;
        bounds=bounds.to(tb::kCUDA);
        auto q=tb::empty({int64_t(mmq::activation_bytes(40,columns))},packed.options());
        mmq::quantize(x.data_ptr<float>(),ids.data_ptr<int32_t>(),q.data_ptr(),int(format),columns,40,mfq_current_cuda_stream());
        auto output=tb::full({40,rows},NAN,options);
        mmq::Context context;
        context.run({packed.data_ptr(),int(format),rows,columns,std::size_t(rows*w.row_bytes),2,q.data_ptr(),
            bounds.data_ptr<int32_t>(),output_ids.data_ptr<int32_t>(),40,29,output.data_ptr<float>(),rows},mfq_current_cuda_stream());
        if(mmq::direct_fits(format,rows,columns,mfq_current_cuda_device())) {
            const auto bytes=rows*w.row_bytes;
            // Separate allocations and nonzero tails expose accidental use of
            // a contiguous stride or an unguarded K over-read.
            std::array<tb::Tensor,2> scattered_weights;
            std::array<const void*,2> table;
            for(int e=0;e<2;++e) {
                scattered_weights[e]=tb::full({int64_t(bytes+guard)},0x7c,packed.options());
                scattered_weights[e].narrow(0,0,bytes).copy_(w.packed.narrow(0,e*bytes,bytes));
                table[e]=scattered_weights[e].data_ptr();
            }
            for(int maximum:{29,128,513}) {
                auto direct=tb::full({40,rows},NAN,options);
                context.run({packed.data_ptr(),int(format),rows,columns,std::size_t(bytes),2,q.data_ptr(),
                    bounds.data_ptr<int32_t>(),output_ids.data_ptr<int32_t>(),40,maximum,direct.data_ptr<float>(),rows,table.data()},mfq_current_cuda_stream());
                const auto actual=direct.cpu();
                // Each width is compared with the same upstream schedule.
                auto gathered=tb::full({40,rows},NAN,options);
                context.run({packed.data_ptr(),int(format),rows,columns,std::size_t(bytes),2,q.data_ptr(),
                    bounds.data_ptr<int32_t>(),output_ids.data_ptr<int32_t>(),40,maximum,gathered.data_ptr<float>(),rows},mfq_current_cuda_stream());
                const auto expected=gathered.cpu();
                require(actual.nbytes()==expected.nbytes() && std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
                    "GGUF direct MMQ differs from the contiguous schedule");
            }
        }
        auto dense=w.dequantize();
        const auto selected=mmq_quantized_reference(x).index_select(0,ids.to(tb::kInt64));
        const auto scattered=output.index_select(0,output_ids.to(tb::kInt64));
        for(const auto [first,count,expert]:{std::tuple{0,29,0},std::tuple{29,11,1}}) {
            const auto input=selected.narrow(0,first,count),weight=dense.narrow(0,expert*rows,rows).transpose(0,1);
            const auto expected=tb::matmul(input,weight),magnitude=tb::matmul(input.abs(),weight.abs());
            const auto error=(scattered.narrow(0,first,count)-expected).abs();
            const auto normalized=(error/(magnitude*.012f+1e-5f)).max().item<float>();
            if(!std::isfinite(normalized) || normalized>1.f)std::cerr<<"MMQ format="<<int(format)<<" rows="<<rows<<" normalized_error="<<normalized<<'\n';
            require(std::isfinite(normalized) && normalized<=1.f,"GGUF MMQ changed quantized multiplication, expert bounds, or output permutation");
        }
    }
}
void run_mmq_scatter() {
    namespace mmq=mfq::cuda::gguf::mmq;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    constexpr int tokens=17,routes=3,slots=tokens*routes,first=7,count=35,rows=256;
    for(auto format:{GGML_TYPE_Q2_0,GGML_TYPE_Q2_K,GGML_TYPE_IQ1_S,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,
            GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS,GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q5_1}) {
        if(!mmq::fits(format,rows,mfq_current_cuda_device()))continue;
        const int columns=format==GGML_TYPE_Q2_0?640:768;
        Source source;source.add("w",{2*rows,columns},format);
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
        auto packed=tb::zeros({weight.packed.numel()+128*int64_t(ggml_type_size(format))},weight.packed.options());
        packed.narrow(0,0,weight.packed.numel()).copy_(weight.packed);
        auto x=tb::empty({tokens,columns},options.device(tb::kCPU));
        for(int64_t i=0;i<x.numel();++i)x.data_ptr<float>()[i]=.1f*std::sin(float(i)*.17f);
        x=x.to(tb::kCUDA);
        auto inverse=tb::empty({slots},options.device(tb::kCPU).dtype(tb::kInt32));
        auto indices=tb::empty_like(inverse);
        for(int slot=0;slot<slots;++slot) {
            const int compact=(slot*13+5)%slots;
            inverse.data_ptr<int32_t>()[slot]=compact;indices.data_ptr<int32_t>()[compact]=slot/routes;
        }
        inverse=inverse.to(tb::kCUDA);indices=indices.to(tb::kCUDA);
        const auto qoptions=options.dtype(tb::kUInt8);
        auto scattered=tb::full({int64_t(mmq::activation_bytes(slots,columns))},0xa5,qoptions);
        auto gathered=tb::full(scattered.sizes(),0xa5,qoptions);
        mmq::quantize_scatter(x.data_ptr<float>(),inverse.data_ptr<int32_t>(),scattered.data_ptr(),int(format),columns,tokens,routes,mfq_current_cuda_stream());
        mmq::quantize(x.data_ptr<float>(),indices.data_ptr<int32_t>(),gathered.data_ptr(),int(format),columns,slots,mfq_current_cuda_stream());
        const auto sh=scattered.cpu(),gh=gathered.cpu();
        require(sh.nbytes()==gh.nbytes() && !std::memcmp(sh.data_ptr(),gh.data_ptr(),sh.nbytes()),
            "GGUF single-token scatter changed a Q8 code, scale, sum, padding or route permutation");
        const auto row_bytes=mmq::activation_bytes(2,columns)-mmq::activation_bytes(1,columns);
        for(auto at=std::size_t(slots)*row_bytes;at<sh.nbytes();++at)
            require(sh.data_ptr<std::uint8_t>()[at]==0 && gh.data_ptr<std::uint8_t>()[at]==0,
                "GGUF quantization left poison in its padded Q8 guard");
        auto compact=tb::full({int64_t(mmq::activation_bytes(count,columns))},0xa5,qoptions);
        mmq::quantize(x.data_ptr<float>(),indices.data_ptr<int32_t>()+first,compact.data_ptr(),int(format),columns,count,mfq_current_cuda_stream());
        auto bounds=tb::empty({3},options.device(tb::kCPU).dtype(tb::kInt32));
        bounds.data_ptr<int32_t>()[0]=0;bounds.data_ptr<int32_t>()[1]=19;bounds.data_ptr<int32_t>()[2]=count;
        bounds=bounds.to(tb::kCUDA);
        auto identity=tb::arange(count+mmq::id_guard,options.dtype(tb::kInt32));
        auto expected=tb::full({count,rows},NAN,options),actual=tb::full({count,rows},NAN,options);
        mmq::Context context;
        mmq::Product product{packed.data_ptr(),int(format),rows,columns,std::size_t(rows*weight.row_bytes),2,compact.data_ptr(),
            bounds.data_ptr<int32_t>(),identity.data_ptr<int32_t>(),count,19,expected.data_ptr<float>(),rows};
        context.run(product,mfq_current_cuda_stream());
        product.activation=scattered.data_ptr();product.activation_rows=slots;product.activation_offset=first;product.output=actual.data_ptr<float>();
        context.run(product,mfq_current_cuda_stream());
        const auto ah=actual.cpu(),eh=expected.cpu();
        require(!std::memcmp(ah.data_ptr(),eh.data_ptr(),ah.nbytes()),"GGUF global activation row stride/offset changed MMQ output");
        if(mmq::direct_fits(format,rows,columns,mfq_current_cuda_device())) {
            const void* table[2]={packed.data_ptr(),packed.data_ptr<std::uint8_t>()+rows*weight.row_bytes};
            product.weight_addresses=table;product.activation=compact.data_ptr();product.activation_rows=0;product.activation_offset=0;
            product.output=expected.data_ptr<float>();context.run(product,mfq_current_cuda_stream());
            product.activation=scattered.data_ptr();product.activation_rows=slots;product.activation_offset=first;
            product.output=actual.data_ptr<float>();context.run(product,mfq_current_cuda_stream());
            const auto direct=actual.cpu(),reference=expected.cpu();
            require(!std::memcmp(direct.data_ptr(),reference.data_ptr(),direct.nbytes()),
                "GGUF scattered-weight MMQ changed global activation view");
        }
    }
    std::cout<<"GGUF per-token quantize/scatter all layouts + nonzero-offset group views exact PASS\n";
}
void run_mmq_moe(bool wide=false) {
    const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_MMQ");
    if(setting && setting[0]=='0')return;
    for(bool joined:{false,true}) {
        const int experts=wide?33:3,tokens=wide?257:67;
        Source source;
        source.add("gate",{experts,256,256},joined?GGML_TYPE_Q8_0:GGML_TYPE_IQ4_XS);
        source.add("up",{experts,256,256},joined?GGML_TYPE_Q8_0:GGML_TYPE_IQ3_S);
        source.add("down",{experts,256,256},joined?GGML_TYPE_Q8_0:GGML_TYPE_Q2_0);
        CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4ULL<<20;execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        const auto x=input(tokens,3,false).to(tb::kFloat16);
        auto ids=wide?tb::arange(tokens*3,x.options().dtype(tb::kInt64)).remainder(experts).reshape({tokens,3}):routes(tokens,1);
        // A late expert is far busier than its neighbors; wide cases span
        // multiple fenced groups and exercise the final one-expert tail.
        ids.narrow(0,0,wide?tokens/2:11).fill_(experts-1);
        auto weights=tb::full({tokens,3},.13,x.options().dtype(tb::kFloat32));
        auto result=banks.forward(x,ids,weights,.2f).to(tb::kFloat32);
        auto expected=tb::zeros_like(result);
        auto gate=source.decoded("gate").to(tb::kCUDA),up=source.decoded("up").to(tb::kCUDA),down=source.decoded("down").to(tb::kCUDA);
        auto f=mmq_quantized_reference(x);
        for(int e=0;e<experts;++e) {
            auto g=tb::clamp_max(tb::matmul(f,gate.select(0,e).transpose(0,1)),.2f);
            auto u=tb::clamp(tb::matmul(f,up.select(0,e).transpose(0,1)),-.2f,.2f);
            auto h=g*tb::sigmoid(g)*u;
            auto d=tb::matmul(mmq_quantized_reference(h),down.select(0,e).transpose(0,1)).to(tb::kFloat16).to(tb::kFloat32);
            auto routing=(ids==e).to(tb::kFloat32)*weights;
            expected=expected+d*routing.sum(1,true);
        }
        const auto error=(result-expected).abs().max().item<float>(),maximum=expected.abs().max().item<float>();
        if(!std::isfinite(error) || error>maximum*.001f+2e-5f)std::cerr<<"MMQ MoE joined="<<joined<<" error="<<error<<" maximum="<<maximum<<'\n';
        require(std::isfinite(error) && error<=maximum*.001f+2e-5f,"GGUF MMQ MoE changed grouping, clamping, activation, or weighted reduction");
    }
}
void run_combined_moe(bool pageable=false,bool batched=false) {
    Source source;
    if(batched)source.enable_batches();
    source.add("gate", {3, 256, 256}, GGML_TYPE_Q4_0);
    source.add("up", {3, 256, 256}, GGML_TYPE_Q5_0);
    source.add("down", {3, 256, 256}, GGML_TYPE_Q8_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes = batched ? 3*256*8*34 : (pageable ? 1 : 2 * 1024 * 1024);
    execution.config.moe_device_cache_bytes = 0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution, source, "gate"),
        std::make_shared<GgufRoutedWeight>(execution, source, "up"),
        std::make_shared<GgufRoutedWeight>(execution, source, "down")};
    const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    for (int tokens : {1, 2, 4, 5, 8, 64}) {
        for (const auto type : {tb::kFloat16, tb::kFloat32}) {
            const auto x=input(tokens,3,false).to(type);
            auto ids=routes(tokens,0);
            if(tokens==64) {
                auto selected=ids.cpu();
                // Keep every reference expert on its FP16 GEMM path while
                // exercising nonuniform counts and a changed expert order.
                for(int i=0;i<selected.numel();++i)selected.data_ptr<int64_t>()[i]=i%16<3?0:(i%16<8?1:2);
                ids=selected.to(tb::kCUDA);
            }
            const auto weights = tb::full({tokens, 3}, .3, options);
            for (float limit : {0.f, .01f}) check_moe(banks.forward(x, ids, weights, limit), banks, x, ids, weights, limit);
        }
    }
    require(prepare_gguf_graph_pool(execution.gguf_expert_cache), "GGUF MoE graph pool missing");
    auto x = input(2, 3, false), ids = routes(2, 0);
    auto weights = tb::full({2, 3}, .3, options);
    tb::Tensor actual;
    mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&] { actual = banks.forward(x, ids, weights); }, [] {});
    bool direct=false;
    for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
        if(key=="gguf_direct_expert_calls")direct=value>0;
    require(graph.segments()==(direct?1:2),"GGUF MoE used unnecessary host barriers");
    for (int rotation : {1, 2}) {
        ids.copy_(routes(2, rotation)); weights.fill_(.1 * rotation);
        graph.run(); check_moe(actual, banks, x, ids, weights);
    }
    if(direct) {
        const auto* reuse=std::getenv("TYLOINFER_GGUF_ROUTED_REUSE");
        double packets=0;
        for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
            if(key=="gguf_host_route_group_packets")packets=value;
        require((packets>0)==(!reuse || reuse[0]=='1'),"GGUF shared route publication ignored its default or explicit override");
    }
    if(direct) {
        ids.fill_(99);
        bool rejected=false;
        try{graph.run();}catch(const std::out_of_range&){rejected=true;}
        require(rejected,"GGUF direct graph accepted an invalid route");
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
    }
    if(batched)require(source.batch_calls()>0,"GGUF expert batch reader was unused");
    if(!pageable && !batched) {
        std::map<std::string,double> metrics;
        for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))metrics[key]=value;
        const auto* setting=std::getenv("TYLOINFER_GGUF_PREFILL_DMA");
        if(setting && setting[0]=='0')require(metrics.at("gguf_prefill_dma_bytes")==0,"GGUF disabled DMA override was ignored");
        else if(metrics.at("ram_expert_pinned_payload_bytes")>0)
            require(metrics.at("gguf_prefill_dma_bytes")>0,"GGUF prefill did not use its pinned DMA path");
    }
}
void run_pinned_copy_plan() {
    using mfq::cuda::PinnedUploadRange;
    std::array<std::byte,128> source{},destination{};
    int source_root[2]{},device_root[2]{};
    const auto span=[&](int s,int d,int n,int sr=0,int dr=0) {
        return PinnedUploadRange{source.data()+s,destination.data()+d,std::size_t(n),
            &source_root[sr],&device_root[dr]};
    };
    std::vector<PinnedUploadRange> ranges{span(32,32,16),span(0,0,16),span(16,16,16)};
    mfq::cuda::coalesce_pinned_uploads(ranges);
    require(ranges.size()==1 && ranges[0].bytes==48 && ranges[0].source==source.data(),
        "GGUF continuous slices in common allocations did not coalesce");
    // Consecutive virtual addresses alone must never authorize a wider copy.
    for(int boundary=0;boundary<5;++boundary) {
        auto second=span(16,16,16);
        if(boundary==0)second.source_allocation=&source_root[1];
        if(boundary==1)second.destination_allocation=&device_root[1];
        if(boundary==2)second.source_allocation=nullptr;
        if(boundary==3)second.source+=1;
        if(boundary==4)second.destination-=8;
        ranges={span(0,0,16),second};
        mfq::cuda::coalesce_pinned_uploads(ranges);
        require(ranges.size()==2,"GGUF upload crossed an allocation boundary or a discontinuity");
    }
}
void run_decode_dma() {
    constexpr std::size_t count=259,stride=113,total=count*stride;
    mfq::cuda::HostBuffer source(total,true);
    auto* bytes=static_cast<std::uint8_t*>(source.data());
    for(std::size_t i=0;i<total;++i)bytes[i]=std::uint8_t(i*17+3);
    void* mapped=nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,source.data(),0));
    auto target=tb::empty({std::int64_t(total)},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kUInt8));
    GgufDecodeDma dma(mfq_current_cuda_device());
    mfq::cuda::HostBuffer flags(2*sizeof(std::uint32_t),true);void* flags_device=nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&flags_device,flags.data(),0));
    auto* ready=static_cast<std::uint32_t*>(flags.data());
    auto* ready_device=static_cast<std::uint32_t*>(flags_device);
    ready[0]=ready[1]=0;
    dma.submit(nullptr,0,0,&ready[0],&ready[1]);
    require(ready[0]==1 && ready[1]==1,"Empty decode DMA phases did not publish completion");
    for(const std::size_t split:{std::size_t(0),128*stride,total}) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemsetAsync(target.data_ptr(),0xa5,total,mfq_current_cuda_stream()));
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        std::vector<std::uint8_t> expected(total,0xa5);
        std::vector<mfq::cuda::MoeMappedCopyDescriptor> descriptors;
        for(std::size_t i=0;i<count;++i) {
            const auto offset=i*stride,size=1+i%(stride-1);
            descriptors.push_back({reinterpret_cast<std::uint64_t>(target.data_ptr())+offset,
                reinterpret_cast<std::uint64_t>(mapped)+offset,offset,size});
            std::memcpy(expected.data()+offset,bytes+offset,size);
        }
        std::reverse(descriptors.begin(),descriptors.end());
        ready[0]=ready[1]=0;
        dma.submit(descriptors.data(),descriptors.size(),split,&ready[0],&ready[1],ready_device,ready_device+1);
        dma.synchronize();
        require(ready[0]==1 && ready[1]==1,"Decode DMA omitted a phase completion signal");
        const auto actual=target.cpu();
        require(std::memcmp(actual.data_ptr(),expected.data(),total)==0,
            "Decode DMA changed compressed bytes, descriptor tails or destination gaps");
    }
    // An invalid later descriptor must reject the complete packet before any
    // copy is queued or any completion flag is published.
    const mfq::cuda::MoeMappedCopyDescriptor valid{reinterpret_cast<std::uint64_t>(target.data_ptr()),
        reinterpret_cast<std::uint64_t>(mapped),0,1};
    for(int invalid=0;invalid<5;++invalid) {
        auto bad=valid;
        if(invalid==0)bad.source=0;
        if(invalid==1)bad.destination=0;
        if(invalid==2)bad.bytes=0;
        if(invalid==3){bad.offset=UINT64_MAX;bad.bytes=2;}
        if(invalid==4){bad.offset=1;bad.bytes=2;}
        const std::array<mfq::cuda::MoeMappedCopyDescriptor,2> descriptors{valid,bad};
        ready[0]=ready[1]=0;const auto copied=dma.bytes();bool rejected=false;
        try{dma.submit(descriptors.data(),descriptors.size(),2,&ready[0],&ready[1]);}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected && !ready[0] && !ready[1] && copied==dma.bytes(),
            "Invalid decode DMA packet partially submitted or published completion");
    }
    dma.synchronize();
    require(dma.copies()==3*count,"Decode DMA copy accounting omitted a phase or packet");
    std::cout<<"GGUF decode DMA bytes/tails/empty phases/multiple packets/preflight rejection PASS stream_signals="
        <<dma.signals()<<" batches="<<dma.batches()<<'\n';
}
void run_prefill_arena_reuse() {
    Source source;
    source.add("gate",{9,256,256},GGML_TYPE_Q4_0);
    source.add("up",{9,256,256},GGML_TYPE_Q5_0);
    source.add("down",{9,256,256},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=4ULL<<20;execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const std::array<int,9> counts{512,384,256,192,128,96,64,48,48};
    int tokens=0;for(auto count:counts)tokens+=count;
    const auto x=input(tokens,1,false).to(tb::kFloat16);
    auto selected=tb::empty({tokens,1},x.options().device(tb::kCPU).dtype(tb::kInt64));
    const auto weights=tb::full({tokens,1},.3,x.options().dtype(tb::kFloat32));
    for(int rotation:{0,1}) {
        int offset=0;
        for(int expert=0;expert<9;++expert)for(int j=0;j<counts[expert];++j)
            selected.data_ptr<int64_t>()[offset++]=(expert+rotation)%9;
        const auto ids=selected.to(tb::kCUDA);
        check_moe(banks.forward(x,ids,weights),banks,x,ids,weights);
    }
    std::map<std::string,double> metrics;
    for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))metrics[key]=value;
    require(metrics.at("gguf_prefill_staging_images")>=54,"GGUF prefill did not reuse its fenced compressed arenas");
    const auto* dma=std::getenv("TYLOINFER_GGUF_PREFILL_DMA");
    if((!dma || dma[0]!='0') && metrics.at("gguf_prefill_dma_images")>0)
        require(metrics.at("gguf_prefill_dma_copies")<metrics.at("gguf_prefill_dma_images"),
            "GGUF PP did not coalesce slices in its shared allocations");
    require(metrics.at("resident_expert_cache_bytes")==0,"GGUF temporary arena images entered the resident cache");
}
void run_failed_expert_batch() {
    Source source;source.enable_batches();source.add("bank",{5,7,256});
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=3*7*8*18;
    execution.config.moe_device_cache_bytes=0;
    GgufRoutedWeight bank(execution,source,"bank");
    auto x=input(1,3,false),ids=routes(1,0);
    source.fail_next_batch();
    bool failed=false;
    try{(void)bank.forward(x,ids);}catch(const std::runtime_error& error){failed=std::string(error.what())=="fixture batch read failure";}
    require(failed,"GGUF batch fixture did not report its read failure");
    check_projection(bank.forward(x,ids),x,ids,source.decoded("bank"));
    require(source.batch_calls()==2,"GGUF retained partially read expert images");
}
void run_scaled_dense_prefill() {
    Source source;
    source.add("large",{7,256},GGML_TYPE_Q4_0);source.large_q4_scale("large");
    source.add("nonlinear",{7,256},GGML_TYPE_IQ3_S);
    for(const auto* name:{"large","nonlinear"}) {
        const auto weight=load_gguf_weight(source,name,0);
        const auto reference=source.decoded(name);
        for(const int tokens:{33,79}) {
            auto x=tb::empty({tokens,256},tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU));
            for(int row=0;row<tokens;++row)for(int k=0;k<256;++k) {
                const int exponent=row%4==0?90:(row%4==1?-110:(row%4==2?-140:0));
                x.data_ptr<float>()[row*256+k]=row==0?0.f:std::ldexp(std::sin(float(row*256+k)*.17f)*.1f,exponent);
            }
            const auto y=weight.forward(x.to(tb::kCUDA)).cpu();
            require(y.scalar_type()==tb::kFloat32,"GGUF scaled prefill lost FP32 output");
            for(int row=0;row<tokens;++row)for(int n=0;n<7;++n) {
                double expected=0,magnitude=0;
                for(int k=0;k<256;++k) {
                    const double term=double(x.data_ptr<float>()[row*256+k])*reference.data_ptr<float>()[n*256+k];
                    expected+=term;magnitude+=std::abs(term);
                }
                const auto actual=y.data_ptr<float>()[row*7+n];
                if(row==0)require(actual==0.f,"GGUF scaled prefill changed a zero row");
                const int exponent=row%4==0?90:(row%4==1?-110:(row%4==2?-140:0));
                require(std::isfinite(actual) && std::abs(actual-expected)<=.0015*magnitude+std::ldexp(1e-6,exponent),
                    "GGUF scaled prefill overflowed or lost the row exponent/precision");
            }
        }
    }
}
void run_streamed_dense_prefill() {
    const auto context=mfq::cuda::default_context(0);
    const auto old_limit=context->memory_stats().limit;
    Source source;source.add("stream",{2053,1792},GGML_TYPE_Q4_0);
    int cases=0;
    try {
        for(bool quad:{false,true})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            context->set_memory_limit(0);
            auto weight=load_gguf_weight(source,"stream",0);
            if(quad)weight.repack_inplace();
            auto host=tb::empty({79,1792},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.017f)*.1f;
            auto x=host.to(tb::kCUDA,dtype);
            auto reference=weight.forward(x,false).to(tb::kCPU,tb::kFloat32).contiguous();
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
            context->reclaim_idle_memory();
            // An explicit small allocator budget exercises streaming on any
            // sufficiently capable GPU; it is not a device capacity default.
            const auto live=std::max(context->resident_memory_usage(),context->memory_stats().allocated);
            context->set_memory_limit(live+(6ULL<<20));
            // Repeated calls must reuse their temporary pages under the same
            // small limit, rather than charging idle pool pages a second time.
            for(int repeat=0;repeat<3;++repeat) {
                auto actual=weight.forward(x,false).to(tb::kCPU,tb::kFloat32).contiguous();
                double error=0,norm=0;
                for(int64_t i=0;i<actual.numel();++i) {
                    const double r=reference.data_ptr<float>()[i],a=actual.data_ptr<float>()[i];
                    require(std::isfinite(a),"streamed prefill produced a nonfinite result");
                    error+=(a-r)*(a-r);norm+=r*r;
                }
                require(error<=4e-6*std::max(norm,1e-30),"streamed prefill changed canonical/quad tile results");
                ++cases;
            }
        }
        context->set_memory_limit(old_limit);
    }catch(...) {context->set_memory_limit(old_limit);throw;}
    std::cout<<"GGUF budgeted double-buffer FP16/FP32 prefill and incomplete row tiles PASS cases="<<cases<<'\n';
}
void run_prefill_precision() {
    int cases=0;
    const ggml_type formats[]={
#define PREFILL_TEST_FORMAT(F) GGML_TYPE_##F,
        TYLOINFER_GGUF_FORMATS(PREFILL_TEST_FORMAT)
#undef PREFILL_TEST_FORMAT
    };
    for(auto format:formats)for(bool quad:{false,true}) {
        if(quad && !mfq::cuda::gguf::supports_quad_storage(format))continue;
        Source source;source.add("weight",{17,512},format);
        if(format>=GGML_TYPE_PTQ1_0)source.full_compact_codes("weight",format);
        const auto bank=source.decoded("weight");
        auto weight=load_gguf_weight(source,"weight",0);
        if(quad)weight.repack_inplace();
        for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int tokens:{33,79}) {
            auto host=tb::empty({tokens,512},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.1f*std::sin(float(i)*.017f);
            const auto x=host.to(tb::kCUDA,dtype);
            const auto rounded=x.to(tb::kCPU,tb::kFloat32).contiguous();
            const auto decoded_bank=dtype==tb::kFloat16?bank.to(tb::kFloat16).to(tb::kFloat32):bank;
            const auto check=[&](const tb::Tensor& result) {
                const auto y=result.to(tb::kCPU,tb::kFloat32).contiguous();
                double squared_error=0,squared_norm=0;
                for(int row=0;row<tokens;++row)for(int n=0;n<17;++n) {
                    double expected=0,magnitude=0;
                    for(int k=0;k<512;++k) {
                        const double term=double(rounded.data_ptr<float>()[row*512+k])*decoded_bank.data_ptr<float>()[n*512+k];
                        expected+=term;magnitude+=std::abs(term);
                    }
                    const auto actual=y.data_ptr<float>()[row*17+n];
                    // COMPUTE_16F rounds partial accumulations, as llama.cpp's
                    // FP16 GEMM does. Check the complete CPU dot product with
                    // both an absolute-contribution bound and a global L2 bound.
                    if(!std::isfinite(actual) || std::abs(actual-expected)>.01*magnitude+1.e-6)
                        std::cerr<<"prefill precision format="<<int(format)<<" half="<<(dtype==tb::kFloat16)
                            <<" quad="<<quad<<" M="<<tokens<<" row="<<row<<" n="<<n
                            <<" expected="<<expected<<" actual="<<actual<<" magnitude="<<magnitude<<'\n';
                    require(std::isfinite(actual) && std::abs(actual-expected)<=.01*magnitude+1.e-6,
                        "FP16 tensor-core prefill differs from the independent CPU decoder/dot product");
                    squared_error+=(actual-expected)*(actual-expected);squared_norm+=expected*expected;
                }
                require(squared_error<=1.e-4*std::max(squared_norm,1.e-30),
                    "FP16 prefill relative L2 error exceeds one percent");
                ++cases;
            };
            check(weight.forward(x,false));
            // A cached scale initialized on one stream must be usable on another.
            if(dtype==tb::kFloat32 && tokens==33 &&
                    (format==GGML_TYPE_IQ2_XS || format==GGML_TYPE_Q4_0)) {
                mfq::cuda::Event ready;ready.record(mfq_current_cuda_stream());
                const auto other=mfq::cuda::stream_from_pool(false,0);
                TYLOINFER_CUDA_CHECK(cudaStreamWaitEvent(other.stream(),ready.get(),0));
                mfq::cuda::StreamGuard guard(other);check(weight.forward(x,false));
            }
        }
    }
    std::cout<<"GGUF all-format FP16/FP32 prefill, quad row tails and cross-stream metadata PASS cases="<<cases<<'\n';
}
void run_fused_dense_activation() {
    namespace gg=mfq::cuda::gguf;
    int cases=0;
    for(int rows:{1,3,32})for(int width:{32,96,16640})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(bool silu:{false,true}) {
        auto host=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto other=tb::empty_like(host);
        for(int64_t i=0;i<host.numel();++i){host.data_ptr<float>()[i]=std::sin(float(i)*.173f)*20.f;other.data_ptr<float>()[i]=std::cos(float(i)*.31f)*.73f;}
        auto gate=host.to(tb::kCUDA,dtype),up=other.to(tb::kCUDA,dtype);
        auto hidden=(up*(silu?tb::silu(gate):tb::sigmoid(gate))).to(tb::kFloat32).contiguous();
        auto expected=tb::empty({int64_t(rows*gg::q8_row_bytes(width))},gate.options().dtype(tb::kUInt8)),actual=tb::empty_like(expected);
        gg::quantize_input(hidden.data_ptr<float>(),rows,width,expected.data_ptr(),mfq_current_cuda_stream());
        gg::quantize_gated_input(up.data_ptr(),gate.data_ptr(),dtype==tb::kFloat16,rows,width,silu,actual.data_ptr(),mfq_current_cuda_stream());
        const auto a=actual.cpu(),b=expected.cpu();
        require(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"GGUF fused input gate changed ordinary Q8 packet bits");
        ++cases;
    }
    std::cout<<"GGUF fused sigmoid/SiLU product/Q8, FP16/FP32 dtype boundaries: "<<cases<<" cases PASS\n";
}

void run_weight_packet_preferences() {
    namespace gg=mfq::cuda::gguf;
    const auto exact=[](const tb::Tensor& a,const tb::Tensor& b) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        require(x.sizes()==y.sizes() && x.scalar_type()==y.scalar_type() &&
            !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()),"GGUF packet selection changed projection bits");
    };
    int cases=0;
    for(auto type:{GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_XXS})
    for(int width:{256,768})for(int rows:{24,260})for(int tokens:{2,3,4})
    for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        Source source;source.add("w",{rows,width},type);
        auto canonical=load_gguf_weight(source,"w",mfq_current_cuda_device());
        auto quad=canonical;quad.packed=canonical.packed.clone();quad.repack_inplace();
        require(quad.quad_projection_ready(),"Fixture did not prepare four-row weights");
        require(!quad.prefers_position_major_q8(tokens) && canonical.prefers_position_major_q8(tokens),
            "GGUF packet preference ignored compute storage");
        auto host=tb::empty({1,tokens,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.3f*std::sin(float(i)*.173f);
        auto input=host.to(tb::kCUDA,dtype);
        auto native=tb::empty({int64_t(tokens*gg::q8_row_bytes(width))},quad.packed.options());
        auto interleaved=tb::empty({int64_t(gg::interleaved_activation_bytes(tokens,width))},quad.packed.options());
        gg::quantize_input_interleaved(input.data_ptr(),dtype==tb::kFloat16,tokens,width,
            interleaved.data_ptr(),native.data_ptr(),mfq_current_cuda_stream());
        const auto quad_reference=quad.forward_q8_base(native,tokens,dtype,false);
        exact(quad_reference,quad.forward_q8_base(interleaved,tokens,dtype,true));
        const auto canonical_reference=canonical.forward_q8_base(interleaved,tokens,dtype,true);
        for(bool mixed:{false,true}) {
            const std::vector<GgufWeight> weights{quad,mixed?canonical:quad};
            auto prepared=prepare_gguf_group_input(weights,input);
            require(prepared.native.defined() && prepared.interleaved.defined()==mixed,
                "GGUF group omitted a required packet or retained an unused layout");
            auto outputs=prepared.project_all(weights);
            exact(outputs[0].reshape({tokens,rows}),quad_reference);
            exact(outputs[1].reshape({tokens,rows}),mixed?canonical_reference:quad_reference);
            QuantLinearGroup group;
            for(const auto& weight:weights){QuantLinear layer;layer.kind=QuantLinearKind::Gguf;layer.gguf=weight;group.layers.push_back(std::move(layer));}
            CudaExecutionContext execution;tb::Tensor normalized;
            const auto norm=tb::ones({width},input.options().dtype(tb::kFloat32));
            auto fused=group.prepare_rms_norm_input(execution,input,norm,1e-5,0.,normalized);
            require(!fused.shape.empty() && fused.native.defined() && fused.interleaved.defined()==mixed,
                "Fused RMS preparation disagreed with projection packet preference");
            auto fused_output=group.forward_prepared_rms(fused);
            auto separate=prepare_gguf_group_input(weights,normalized);
            auto expected=separate.project_all(weights);
            for(size_t i=0;i<expected.size();++i)exact(fused_output[i],expected[i]);
            if(type==GGML_TYPE_IQ3_XXS && width==768 && rows==24 && tokens==3 &&
                    dtype==tb::kFloat32 && mixed) {
                struct Pool {
                    std::shared_ptr<mfq::cuda::Context> context=mfq::cuda::default_context(mfq_current_cuda_device());
                    cudaStream_t stream=mfq_current_cuda_stream();bool owned=!context->has_graph_pool(stream);
                    Pool(){if(owned)context->begin_graph_pool(stream);}
                    ~Pool(){if(owned)context->end_graph_pool(stream);}
                } pool;
                mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                std::vector<tb::Tensor> captured;
                graph.capture([&]{
                    fused=group.prepare_rms_norm_input(execution,input,norm,1e-5,0.,normalized);
                    captured=group.forward_prepared_rms(fused);
                },[]{},[&]{captured.clear();fused={};normalized={};});
                for(int step=0;step<3;++step) {
                    input.copy_(input*-.9f);graph.run();
                    auto reference=group.prepare_rms_norm_input(execution,input,norm,1e-5,0.,normalized);
                    expected=group.forward_prepared_rms(reference);
                    for(size_t i=0;i<expected.size();++i)exact(captured[i],expected[i]);
                }
            }
            ++cases;
        }
    }
    std::cout<<"GGUF packed/canonical mixed packet groups and fused RMS: "<<cases<<" cases PASS\n";
}

void run_corrected_normalized_input() {
    namespace gg=mfq::cuda::gguf;
    using Tensor=tb::Tensor;
    std::cout.setf(std::ios::unitbuf);
    std::cout<<"GGUF correction-aware fused preparation entry\n";
    struct Correction final:GgufProjectionCorrection {
        Tensor pattern;mutable const void* prepared=nullptr;
        mutable int preparations=0,applications=0;
        explicit Correction(Tensor p):pattern(std::move(p)){}
        void prepare(const Tensor& input)const override{
            prepared=input.data_ptr();++preparations;
        }
        Tensor apply(const Tensor& input,Tensor output)const override{
            require(input.data_ptr()==prepared,"Fused normalization lost the correction's prepared activation");
            ++applications;
            const auto rows=input.numel()/input.size(-1);
            return (output.reshape({rows,pattern.numel()})+
                input.reshape({rows,input.size(-1)}).sum(-1,true)*pattern).reshape_as(output);
        }
    };
    const auto exact=[](const Tensor& a,const Tensor& b) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        require(x.sizes()==y.sizes() && x.scalar_type()==y.scalar_type() &&
            !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()),
            "Fused normalized correction differs from separate base and original-activation correction");
    };
    int checked=0,fallbacks=0;
    for(auto type:{GGML_TYPE_Q2_K,GGML_TYPE_Q4_K,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XS})
    for(int width:{256,768})for(int tokens:{1,3})for(int mode:{0,1,2,3}) {
        Source source;source.add("a",{16,width},type);source.add("b",{7,width},type);
        QuantLinearGroup group;std::vector<GgufWeight> bases;
        std::vector<std::shared_ptr<Correction>> corrections;
        const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
        const auto norm_dtype=mode<2?tb::kFloat32:tb::kFloat16;
        for(const auto* name:{"a","b"}) {
            QuantLinear layer;layer.kind=QuantLinearKind::Gguf;
            layer.gguf=load_gguf_weight(source,name,mfq_current_cuda_device());
            layer.gguf.repack_inplace();bases.push_back(layer.gguf);
            auto h=tb::empty({1,layer.gguf.out},cpu);
            for(int i=0;i<h.numel();++i)h.data_ptr<float>()[i]=.001f*float(i+1);
            auto c=std::make_shared<Correction>(h.to(tb::kCUDA,norm_dtype));
            corrections.push_back(c);layer.gguf.projection_correction=c;
            group.layers.push_back(std::move(layer));
        }
        auto h=tb::empty({tokens,width},cpu),p=tb::empty_like(h),w=tb::empty({width},cpu);
        for(int i=0;i<h.numel();++i){h.data_ptr<float>()[i]=.2f*std::sin(i*.173f);p.data_ptr<float>()[i]=.1f*std::cos(i*.31f);}
        for(int i=0;i<width;++i)w.data_ptr<float>()[i]=.7f*std::cos(i*.23f);
        const std::vector<int64_t> shape=mode%2?std::vector<int64_t>{tokens,1,width}:std::vector<int64_t>{1,tokens,width};
        auto input=h.to(tb::kCUDA,mode==0?tb::kFloat32:tb::kFloat16).reshape(shape);
        auto projection=p.to(tb::kCUDA,mode==3?tb::kFloat32:tb::kFloat16).reshape(shape),weight=w.to(tb::kCUDA);
        CudaExecutionContext execution;Tensor normalized,sum;GgufProjectionInput prepared;
        const auto prepare=[&]{
            prepared=mode<2?group.prepare_rms_norm_input(execution,input,weight,1e-5,0.,normalized)
                :group.prepare_acc_rms_norm_input(execution,input,projection,weight,1e-5,0.,sum,normalized);
        };
        prepare();
        if(tokens>1 && gg::supports_interleaved_activation(type,tokens)) {
            require(prepared.shape.empty() && !normalized.defined(),"Unsupported interleaved preparation changed its fallback");
            for(const auto& c:corrections)require(c->preparations==0,"Rejected preparation started neuron correction work");
            ++fallbacks;continue;
        }
        require(!prepared.shape.empty() && prepared.original.defined() &&
            prepared.original.data_ptr()==normalized.data_ptr() && prepared.original.scalar_type()==norm_dtype,
            "Fused preparation omitted the normalized original activation");
        require(prepared.original.sizes().vec()==std::vector<int64_t>{tokens,width},
            "Fused preparation changed the flattened original activation contract");
        auto actual=group.forward_prepared_rms(prepared),reference=prepared.project_all(bases);
        for(size_t i=0;i<actual.size();++i) {
            exact(actual[i],reference[i]+normalized.sum(-1,true)*corrections[i]->pattern);
            require(corrections[i]->preparations==1 && corrections[i]->applications==1,
                "Fused preparation skipped or repeated a neuron correction callback");
        }
        if(tokens==1 && width==768 && mode==3) {
            struct Pool {
                std::shared_ptr<mfq::cuda::Context> context=mfq::cuda::default_context(mfq_current_cuda_device());
                cudaStream_t stream=mfq_current_cuda_stream();bool owned=!context->has_graph_pool(stream);
                Pool(){if(owned)context->begin_graph_pool(stream);}
                ~Pool(){if(owned)context->end_graph_pool(stream);}
            } pool;
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{prepare();actual=group.forward_prepared_rms(prepared);},[]{},[&]{actual.clear();prepared={};normalized={};sum={};});
            for(int step=0;step<3;++step) {
                input.copy_(input*-.9f);graph.run();
                prepare();reference=prepared.project_all(bases);
                for(size_t i=0;i<actual.size();++i)
                    exact(actual[i],reference[i]+normalized.sum(-1,true)*corrections[i]->pattern);
            }
        }
        ++checked;
    }
    std::cout<<"GGUF correction-aware fused RMS/Q8 preparation: "<<checked
        <<" exact cases, "<<fallbacks<<" unchanged fallbacks, changing graphs PASS\n";
}

void run_fused_normalized_input() {
    namespace gg=mfq::cuda::gguf;
    int cases=0;
    for(int rows:{1,3,32})for(int width:{32,96,768,5120,16640})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float offset:{0.f,1.f}) {
        auto host=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto wh=tb::empty({width},host.options());
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=(rows>1 && i<width)?0.f:std::sin(float(i)*.173f)*2.f;
        for(int i=0;i<width;++i)wh.data_ptr<float>()[i]=std::cos(float(i)*.31f)*.73f;
        auto input=host.to(tb::kCUDA,dtype),weight=wh.to(tb::kCUDA);
        auto normalized=tb::empty(host.sizes(),weight.options());
        auto actual=tb::empty({int64_t(rows*gg::q8_row_bytes(width))},input.options().dtype(tb::kUInt8)),expected=tb::empty_like(actual);
        gg::quantize_rms_input(input.data_ptr(),dtype==tb::kFloat16,weight.data_ptr<float>(),rows,width,1e-5f,offset,
            normalized.data_ptr<float>(),actual.data_ptr(),mfq_current_cuda_stream());
        gg::quantize_input(normalized.data_ptr<float>(),rows,width,expected.data_ptr(),mfq_current_cuda_stream());
        auto a=actual.cpu(),e=expected.cpu(),xh=input.to(tb::kCPU,tb::kFloat32),y=normalized.cpu();
        require(std::memcmp(a.data_ptr(),e.data_ptr(),a.nbytes())==0,"fused RMS/Q8 packet differs from its normalized output");
        for(int row=0;row<rows;++row) {
            double square=0,norm=0,error=0;
            for(int i=0;i<width;++i){const double v=xh.data_ptr<float>()[row*width+i];square+=v*v;}
            const double inverse=1./std::sqrt(square/width+1e-5);
            for(int i=0;i<width;++i) {
                const double ref=xh.data_ptr<float>()[row*width+i]*inverse*(wh.data_ptr<float>()[i]+offset);
                const double d=y.data_ptr<float>()[row*width+i]-ref;norm+=ref*ref;error+=d*d;
            }
            require(std::isfinite(error) && std::sqrt(error/std::max(norm,1e-30))<1e-6,"fused RMS/Q8 normalization differs from double reference");
        }
        ++cases;
    }
    std::cout<<"GGUF fused ordinary RMSNorm/Q8, zero rows/FP16/FP32/offsets: "<<cases<<" cases PASS\n";
}

void run_fused_residual_normalized_input() {
    namespace gg=mfq::cuda::gguf;int cases=0;
    for(int rows:{1,3})for(int width:{32,96,768,5120,16640})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float offset:{0.f,1.f}) {
        auto host=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto bh=tb::empty_like(host),wh=tb::empty({width},host.options());
        for(int64_t i=0;i<host.numel();++i){host.data_ptr<float>()[i]=std::sin(float(i)*.173f)*2.f;bh.data_ptr<float>()[i]=std::cos(float(i)*.31f)*.73f;}
        for(int i=0;i<width;++i)wh.data_ptr<float>()[i]=std::cos(float(i)*.27f)*.73f;
        auto a=host.to(tb::kCUDA,tb::kFloat16),b=bh.to(tb::kCUDA,dtype),w=wh.to(tb::kCUDA);
        auto sum=tb::empty_like(a),norm=tb::empty_like(a);
        auto packed=tb::empty({int64_t(rows*gg::q8_row_bytes(width))},a.options().dtype(tb::kUInt8)),expected_q=tb::empty_like(packed);
        gg::quantize_acc_rms_input(a.data_ptr(),b.data_ptr(),dtype==tb::kFloat32,w.data_ptr<float>(),rows,width,1e-5f,offset,
            sum.data_ptr(),norm.data_ptr(),packed.data_ptr(),mfq_current_cuda_stream());
        auto expected=acc_rms_norm_f16_cuda(a,b,w,1e-5,offset);
        auto sh=sum.cpu(),esh=expected[0].cpu(),nh=norm.to(tb::kCPU,tb::kFloat32),enh=expected[1].to(tb::kCPU,tb::kFloat32);
        require(std::memcmp(sh.data_ptr(),esh.data_ptr(),sh.nbytes())==0,"fused residual RMS/Q8 changed residual rounding");
        double error=0,energy=0;
        for(int64_t i=0;i<nh.numel();++i){double r=enh.data_ptr<float>()[i],d=nh.data_ptr<float>()[i]-r;error+=d*d;energy+=r*r;}
        require(std::isfinite(error) && std::sqrt(error/std::max(energy,1e-30))<2e-4,"fused residual RMS/Q8 changed normalization");
        gg::quantize_input_f16(norm.data_ptr(),rows,width,expected_q.data_ptr(),mfq_current_cuda_stream());
        auto q=packed.cpu(),eq=expected_q.cpu();require(std::memcmp(q.data_ptr(),eq.data_ptr(),q.nbytes())==0,
            "fused residual RMS/Q8 packet differs from FP16 normalized values");++cases;
    }
    std::cout<<"GGUF fused residual RMSNorm/Q8, exact residual and packed output: "<<cases<<" cases PASS\n";
}

void run_residual_rms_order() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();int cases=0,graphs=0;
    const auto exact=[](const tb::Tensor& x,const tb::Tensor& y) {
        auto a=x.contiguous().cpu(),b=y.contiguous().cpu();
        require(a.nbytes()==b.nbytes() && std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,
            "Residual RMS/Q8 arithmetic changed with token count or packet layout");
    };
    for(int rows:{2,3,4,5,6,7,8})for(int width:{96,768,2048,4096,5120,6144,16640})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float offset:{0.f,1.f}) {
        auto h=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto ph=tb::empty_like(h),wh=tb::empty({width},h.options());
        for(int64_t i=0;i<h.numel();++i){h.data_ptr<float>()[i]=std::sin(float(i)*.173f)*2.f;ph.data_ptr<float>()[i]=std::cos(float(i)*.31f)*.73f;}
        for(int i=0;i<width;++i)wh.data_ptr<float>()[i]=std::cos(float(i)*.27f)*.73f;
        auto a=h.to(tb::kCUDA,tb::kFloat16),b=ph.to(tb::kCUDA,dtype),w=wh.to(tb::kCUDA);
        auto sum=tb::empty_like(a),norm=tb::empty_like(a),refsum=tb::empty_like(a),refnorm=tb::empty_like(a);
        const auto stride=gg::q8_row_bytes(width);
        auto q=tb::empty({int64_t(rows*stride)},a.options().dtype(tb::kUInt8)),refq=tb::empty_like(q);
        const bool interleaved=gg::supports_interleaved_activation(GGML_TYPE_IQ4_XS,rows);
        auto pos=interleaved?tb::empty({int64_t(gg::interleaved_activation_bytes(rows,width))},q.options()):tb::Tensor{};
        auto ps=tb::empty_like(a),pn=tb::empty_like(a),pq=tb::empty_like(q);
        const auto compute=[&] {
            gg::quantize_acc_rms_input(a.data_ptr(),b.data_ptr(),dtype==tb::kFloat32,w.data_ptr<float>(),rows,width,1e-5f,offset,sum.data_ptr(),norm.data_ptr(),q.data_ptr(),stream);
            if(interleaved)gg::quantize_acc_rms_input_interleaved(a.data_ptr(),b.data_ptr(),dtype==tb::kFloat32,w.data_ptr<float>(),rows,width,1e-5f,offset,ps.data_ptr(),pn.data_ptr(),pos.data_ptr(),pq.data_ptr(),stream);
        };
        const auto verify=[&] {
            for(int row=0;row<rows;++row)gg::quantize_acc_rms_input(
                a.data_ptr<mfq_half>()+row*width,static_cast<const uint8_t*>(b.data_ptr())+row*width*(dtype==tb::kFloat32?sizeof(float):sizeof(mfq_half)),dtype==tb::kFloat32,
                w.data_ptr<float>(),1,width,1e-5f,offset,refsum.data_ptr<mfq_half>()+row*width,refnorm.data_ptr<mfq_half>()+row*width,refq.data_ptr<uint8_t>()+row*stride,stream);
            exact(sum,refsum);exact(norm,refnorm);exact(q,refq);
            if(interleaved){exact(ps,refsum);exact(pn,refnorm);exact(pq,refq);}
        };
        compute();verify();++cases;
        if(width==5120 && (rows==3 || rows==8) && offset==0.f) {
            struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
                ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}} graph;
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));compute();
            TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph.graph));
            TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&graph.exec,graph.graph,nullptr,nullptr,0));
            for(int step=0;step<3;++step){a.copy_(a*-.9f);b.copy_(b*1.03125f);TYLOINFER_CUDA_CHECK(cudaGraphLaunch(graph.exec,stream));verify();++graphs;}
        }
    }
    std::cout<<"Residual RMS/Q8 exact M1 versus M2..8, native/interleaved: "<<cases<<" cases + "<<graphs<<" changed-input Graph replays PASS\n";
}

void run_fused_rms_activation() {
    namespace gg=mfq::cuda::gguf;
    int cases=0;
    for(int batch:{1,2})for(int tokens:{1,3,17})for(int width:{32,96,128,320})for(bool strided:{false,true}) {
        constexpr int heads=3;
        auto host=tb::empty({batch,heads,tokens,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto gh=tb::empty({batch,tokens,heads*width+(strided?64:0)},host.options()),wh=tb::empty({width},host.options());
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=i<tokens*width?0.f:std::sin(float(i)*.173f)*2.f;
        for(int64_t i=0;i<gh.numel();++i)gh.data_ptr<float>()[i]=std::cos(float(i)*.31f)*20.f;
        for(int i=0;i<width;++i)wh.data_ptr<float>()[i]=std::cos(float(i)*.27f)*.73f;
        auto input=host.to(tb::kCUDA),gate=gh.to(tb::kCUDA).narrow(2,strided?16:0,heads*width),weight=wh.to(tb::kCUDA);
        auto normalized=rms_norm_cuda(input.reshape({-1,width}),weight,1e-5).reshape_as(input)
            .transpose(1,2).contiguous().reshape({batch,tokens,heads*width});
        auto expected=tb::empty({int64_t(batch*tokens*gg::q8_row_bytes(heads*width))},input.options().dtype(tb::kUInt8)),actual=tb::empty_like(expected);
        auto contiguous_gate=gate.contiguous();
        gg::quantize_gated_input(normalized.data_ptr(),contiguous_gate.data_ptr(),false,batch*tokens,heads*width,true,expected.data_ptr(),mfq_current_cuda_stream());
        gg::quantize_rms_gated_input(input.data_ptr<float>(),gate.data_ptr<float>(),weight.data_ptr<float>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),1e-5f,actual.data_ptr(),mfq_current_cuda_stream());
        auto a=actual.cpu(),e=expected.cpu();
        require(std::memcmp(a.data_ptr(),e.data_ptr(),a.nbytes())==0,"GGUF fused RMSNorm/SiLU/Q8 changed packed activation bits");
        ++cases;
    }
    std::cout<<"GGUF grouped RMSNorm/SiLU/Q8, including zero heads and strided gates: "<<cases<<" exact cases PASS\n";
}

void run_small_f32_projection() {
    int cases=0,graphs=0;
    for(int tokens:{1,2,3,4,5,6,7,8})for(int rows:{3,23,96})for(int width:{127,768,5120})for(int offset:{0,1}) {
        auto xh=tb::empty({tokens*width+offset},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto wh=tb::empty({rows*width+offset},xh.options());
        for(int64_t i=0;i<xh.numel();++i)xh.data_ptr<float>()[i]=std::sin(float(i)*.173f)*2.f;
        for(int64_t i=0;i<wh.numel();++i)wh.data_ptr<float>()[i]=std::cos(float(i)*.31f)*.73f;
        auto xs=xh.to(tb::kCUDA),ws=wh.to(tb::kCUDA);
        auto x=xs.narrow(0,offset,tokens*width).reshape({tokens,width});
        auto weight=ws.narrow(0,offset,rows*width).reshape({rows,width});
        auto actual=dense_projection(x,weight);
        auto fused=try_dense_f32_projection_cuda(x,weight);
        // Three output rows always fit the short projection schedule. Wider
        // fixtures may legitimately use GEMM on a device with fewer SMs.
        require(rows!=3 || fused.defined(),"small F32 projection omitted a supported token window");
        const auto y=actual.cpu();
        if(fused.defined()) {
            const auto expected=fused.cpu();
            require(std::memcmp(y.data_ptr(),expected.data_ptr(),y.nbytes())==0,
                "dense projection did not use the available short F32 schedule");
        }
        for(int t=0;t<tokens;++t)for(int row=0;row<rows;++row) {
            double sum=0,magnitude=0;
            for(int k=0;k<width;++k) {
                const double term=double(xh.data_ptr<float>()[offset+t*width+k])*wh.data_ptr<float>()[offset+row*width+k];
                sum+=term;magnitude+=std::abs(term);
            }
            require(std::isfinite(y.data_ptr<float>()[t*rows+row]) &&
                std::abs(y.data_ptr<float>()[t*rows+row]-sum)<=1e-5+1e-6*magnitude,
                "small F32 projection differs from double reference");
        }
        for(int t=0;fused.defined() && t<tokens;++t) {
            auto single=dense_projection(x.narrow(0,t,1),weight).cpu();
            require(std::memcmp(single.data_ptr(),y.data_ptr<float>()+t*rows,single.nbytes())==0,
                "small F32 projection changed singleton accumulation in a short window");
        }
        if((tokens==3 || tokens==8) && rows==3 && width==768 && offset==0) {
            struct Pool {
                std::shared_ptr<mfq::cuda::Context> context=mfq::cuda::default_context(mfq_current_cuda_device());
                cudaStream_t stream=mfq_current_cuda_stream();
                bool owned=!context->has_graph_pool(stream);
                Pool(){if(owned)context->begin_graph_pool(stream);}
                ~Pool(){if(owned)context->end_graph_pool(stream);}
            } pool;
            mfq::cuda::DecodeWindow graph(pool.stream);
            tb::Tensor output;
            graph.capture([&]{output=dense_projection(x,weight);},[]{});
            const auto original=x.clone();
            for(float scale:{.3f,.8f,1.1f}) {
                x.copy_(original*scale);
                graph.run();
                const auto actual=output.cpu();
                for(int t=0;t<tokens;++t) {
                    const auto expected=dense_projection(x.narrow(0,t,1),weight).cpu();
                    require(std::memcmp(expected.data_ptr(),actual.data_ptr<float>()+t*rows,expected.nbytes())==0,
                        "small F32 projection Graph differs from changed singleton inputs");
                }
                ++graphs;
            }
            x.copy_(original);
        }
        ++cases;
    }
    std::cout<<"Small F32 projections, tails and singleton-equivalent windows: "<<cases<<" cases, "<<graphs<<" changing Graphs PASS\n";
}

void run_fused_hidden_quantization() {
    using namespace mfq::cuda::gguf;
    for(int rows:{1,3,7})for(int width:{32,96,640}) {
        auto gate=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto up=tb::empty_like(gate);
        for(int i=0;i<gate.numel();++i) {
            gate.data_ptr<float>()[i]=i%13==0?1.00048828125f:std::sin(float(i)*.17f)*20.f;
            up.data_ptr<float>()[i]=i%11==0?-.333251953125f:std::cos(float(i)*.31f)*.73f;
        }
        gate=gate.to(tb::kCUDA);up=up.to(tb::kCUDA);
        const auto bytes=rows*q8_row_bytes(width);
        auto expected=tb::empty({std::int64_t(bytes)},gate.options().dtype(tb::kUInt8));
        auto actual=tb::empty_like(expected);
        for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float limit:{0.f,.01f,2.f}) {
            auto g=gate.to(dtype),u=up.to(dtype);
            if(limit>0){g=tb::clamp_max(g.to(tb::kFloat32),limit);u=tb::clamp(u.to(tb::kFloat32),-limit,limit);}
            auto hidden=((g*tb::sigmoid(g))*u).to(dtype).to(tb::kFloat32).contiguous();
            quantize_input(hidden.data_ptr<float>(),rows,width,expected.data_ptr(),mfq_current_cuda_stream());
            quantize_moe_hidden(gate.data_ptr<float>(),up.data_ptr<float>(),rows,width,dtype==tb::kFloat16,
                limit,actual.data_ptr(),mfq_current_cuda_stream());
            const auto a=actual.cpu(),b=expected.cpu();
            require(std::memcmp(a.data_ptr(),b.data_ptr(),bytes)==0,"GGUF fused SwiGLU/Q8 changed activation rounding or packed bytes");
        }
    }
}
void run_exclusive_cache(bool mapped=false) {
    Source source; source.add("bank", {3, 7, 256});
    if(mapped)source.enable_mapping();
    constexpr std::size_t image_bytes = 7 * 8 * 18;
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes = 2 * image_bytes;
    execution.config.moe_device_cache_bytes = image_bytes;
    if(mapped)execution.config.moe_host_physical_bytes=1; // Simulate exhausted host registration capacity.
    GgufRoutedWeight bank(execution, source, "bank");
    const auto reference = source.decoded("bank"), x = input(1, 3, false);
    for (int rotation : {0, 1, 2, 0}) check_projection(bank.forward(x, routes(1, rotation)), x, routes(1, rotation), reference);
    std::map<std::string, double> metrics;
    for (auto [key, value] : gguf_expert_memory_metrics(execution.gguf_expert_cache)) metrics[key] = value;
    if(mapped) {
        const auto* option=std::getenv("TYLOINFER_GGUF_MAPPED_RECOVERY");
        const bool enabled=option && option[0]=='1';
        require((metrics.at("gguf_immutable_recovery_bytes")>0)==enabled,"GGUF mapped recovery default/override was ignored");
        if(enabled)require(metrics.at("expert_writeback_bytes")==0,"GGUF pageable victim blocked on a GPU writeback despite a retained mapping");
        return;
    }
    require(metrics.at("ssd_expert_read_count") == 3, "GGUF reread experts that fit across RAM and VRAM");
    require(metrics.at("ssd_expert_payload_bytes") == 0, "GGUF duplicate host/GPU images wasted cache capacity");
    require(metrics.at("expert_writeback_bytes") > 0, "GGUF eviction did not preserve compressed expert bytes in RAM");
}
void run_global_prefill_admission() {
    Source source;
    for(const auto* layer:{"a","b"})for(const auto* role:{"gate","up","down"})
        source.add(std::string(role)+layer,{3,256,256},GGML_TYPE_Q4_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=2ULL<<20;
    execution.config.moe_device_cache_bytes=2*source.find_tensor("gatea")->nbytes;
    const auto banks=[&](const char* layer) {
        return GgufMoeBanks{std::make_shared<GgufRoutedWeight>(execution,source,std::string("gate")+layer),
            std::make_shared<GgufRoutedWeight>(execution,source,std::string("up")+layer),
            std::make_shared<GgufRoutedWeight>(execution,source,std::string("down")+layer)};
    };
    auto a=banks("a"),b=banks("b");
    finalize_gguf_expert_cache(execution.gguf_expert_cache);
    auto x=input(64,3,false).to(tb::kFloat16);
    auto ids=tb::empty({64,3},x.options().device(tb::kCPU).dtype(tb::kInt64));
    for(int i=0;i<ids.numel();++i)ids.data_ptr<int64_t>()[i]=i%2;
    auto weights=tb::full({64,3},.3,x.options().dtype(tb::kFloat32));
    (void)a.forward(x,ids.to(tb::kCUDA),weights).cpu();
    for(int i=0;i<ids.numel();++i)ids.data_ptr<int64_t>()[i]=i%3;
    (void)b.forward(x,ids.to(tb::kCUDA),weights).cpu();
    restore_gguf_decode_workspace(execution.gguf_expert_cache);
    const auto before=gguf_expert_streamed_bytes(execution.gguf_expert_cache);
    auto selected=tb::tensor(std::vector<int64_t>{0,1,0},ids.options()).to(tb::kCUDA).reshape({1,3});
    (void)a.forward(x.narrow(0,0,1),selected,weights.narrow(0,0,1)).cpu();
    require(gguf_expert_streamed_bytes(execution.gguf_expert_cache)==before,
        "GGUF prefill did not allocate cache across layers by observed token frequency");
    const auto lendable=gguf_prefill_lendable_bytes(execution.gguf_expert_cache);
    if(lendable) {
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        const auto old_limit=context->memory_stats().limit;
        context->reclaim_idle_memory();
        context->set_memory_limit(std::max(context->resident_memory_usage(),context->memory_stats().allocated)+lendable/4);
        try {
            GgufPrefillWorkspaceLoan loan(execution.gguf_expert_cache);
            require(loan.begin(lendable/2)>0,"GGUF slot cache did not lend its tail under a memory budget");
            auto scratch=tb::empty({128},x.options().dtype(tb::kUInt8));scratch.fill_(73);
            const auto base=scratch.data_ptr();
            const auto touched=context->workspace_touched_regions();
            require(!touched.empty() && touched.back().first==base,"GGUF PP workspace allocated outside its cache loan");
            loan.suspend();scratch={};
            context->set_memory_limit(old_limit);
            loan.finish();
            check_moe(a.forward(x.narrow(0,0,1),selected,weights.narrow(0,0,1)),a,x.narrow(0,0,1),selected,weights.narrow(0,0,1));
            for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
                if(key=="gguf_prefill_borrowed_bytes")require(value==0,"GGUF PP left cache slots borrowed");
        }catch(...) {context->set_memory_limit(old_limit);throw;}
    }
}
void run_strata_adaptive_slots() {
    const std::vector<std::vector<float>> usage{{5.f,3.f,8.f,4.f},{0.f,7.f,1.f}};
    const auto resident=[](std::size_t layer,std::size_t expert){return layer==0?expert<2:expert==0;};
    const auto swaps=mfq::cuda::gguf::adaptive_slot_swaps(usage,resident);
    require(swaps.size()==2 && swaps[0].layer==1 && swaps[0].in==1 && swaps[0].out==0 &&
        swaps[1].layer==0 && swaps[1].in==2 && swaps[1].out==1,
        "Strata adaptive slots changed minimum gain, same-layer victims or global gain order");
    require(mfq::cuda::gguf::adaptive_slot_swaps(usage,resident,1).size()==1,
        "Strata adaptive slot swap cap was ignored");
}
void run_unmanaged_bank_learning() {
    Source source;
    for(const auto* layer:{"target","drafter"})for(const auto* role:{"gate","up","down"})
        source.add(std::string(role)+layer,{3,256,256},GGML_TYPE_Q4_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=2ULL<<20;
    execution.config.moe_device_cache_bytes=2*source.find_tensor("gatetarget")->nbytes;
    const auto banks=[&](const char* layer) {
        return GgufMoeBanks{std::make_shared<GgufRoutedWeight>(execution,source,std::string("gate")+layer),
            std::make_shared<GgufRoutedWeight>(execution,source,std::string("up")+layer),
            std::make_shared<GgufRoutedWeight>(execution,source,std::string("down")+layer)};
    };
    auto target=banks("target"),drafter=banks("drafter");
    const auto cache=execution.gguf_expert_cache;
    auto x=input(64,1,false).to(tb::kFloat16);
    auto ids=tb::arange(64,x.options().dtype(tb::kInt64)).remainder(2).reshape({64,1});
    auto weights=tb::full({64,1},1.,x.options().dtype(tb::kFloat32));
    (void)target.forward(x,ids,weights).cpu();
    restore_gguf_decode_workspace(cache);
    const auto metric=[&](const char* key) {
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))if(name==key)return value;
        throw std::runtime_error("GGUF unmanaged-bank metric missing");
    };
    require(metric("gguf_strata_slot_cache_bytes")>0,"GGUF unmanaged-bank test requires an initialized arena");
    prepare_gguf_graph_pool(cache);
    x=x.narrow(0,0,1).contiguous();ids=ids.narrow(0,0,1).contiguous();weights=weights.narrow(0,0,1).contiguous();
    const auto before=metric("gguf_adaptive_rounds");
    tb::Tensor output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=drafter.forward(x,ids,weights);},[]{},[&]{output={};});
    const auto streamed=gguf_expert_streamed_bytes(cache);
    for(int step=0;step<12;++step)graph.run();
    wait_gguf_expert_admission(cache);
    if(metric("gguf_direct_expert_calls")>0) {
        require(gguf_expert_streamed_bytes(cache)==streamed,
            "GGUF repeated draft experts were copied again despite unchanged transfer workspace");
        require(metric("gguf_direct_workspace_reused_bytes")>0,"GGUF transfer workspace reuse did not activate");
    }
    check_moe(output,drafter,x,ids,weights);
    require(metric("gguf_adaptive_rounds")==before,
        "A bank outside the fixed arena advanced its target cache's adaptation/decay clock");
}

void run_resident_predictor(bool uniform=false) {
    Source source;
    const std::string prefix="predictor.block.0.mlp.experts.";
    source.add(prefix+"gate.weight",{7,256,256},GGML_TYPE_Q2_0);
    const auto up_type=uniform?GGML_TYPE_Q2_0:GGML_TYPE_Q4_0,down_type=uniform?GGML_TYPE_Q2_0:GGML_TYPE_Q5_0;
    source.add(prefix+"up.weight",{7,256,256},up_type);
    source.add(prefix+"down.weight",{7,256,256},down_type);
    source.add("target",{7,256,256},GGML_TYPE_Q4_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=4ULL<<20;
    execution.config.moe_device_cache_bytes=0;
    mfq::cuda::WeightMemoryLedger ledger;
    mfq::cuda::WeightMemoryLoadScope loading(ledger);
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,prefix+"gate.weight"),
        std::make_shared<GgufRoutedWeight>(execution,source,prefix+"up.weight"),
        std::make_shared<GgufRoutedWeight>(execution,source,prefix+"down.weight")};
    const auto bytes=banks.gate->resident_packed.nbytes()+banks.up->resident_packed.nbytes()+banks.down->resident_packed.nbytes();
    require(ledger.resident_bytes(mfq_current_cuda_device())==bytes,"Resident predictor experts missing from weight budget");
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* key) {
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))if(name==key)return value;
        throw std::runtime_error("Resident predictor metric missing");
    };
    require(metric("gguf_predictor_resident_expert_bytes")==double(bytes),"Resident predictor payload accounting mismatch");
    require(metric("ram_expert_payload_bytes")==0 && metric("ssd_expert_payload_bytes")==0,
        "Resident predictor experts were duplicated in RAM or counted as streamed SSD experts");
    prepare_gguf_graph_pool(cache);
    auto x=input(2,3,false).to(tb::kFloat16);
    auto ids=tb::arange(6,x.options().dtype(tb::kInt64)).remainder(3).reshape({2,3});
    auto weights=tb::full({2,3},.3,x.options().dtype(tb::kFloat32));
    tb::Tensor output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
    const auto reads=source.read_calls();
    for(const int offset:{0,2,4,1}) {
        resize_gguf_expert_cache(cache,0);
        ids.copy_((tb::arange(6,ids.options()).remainder(3)+offset).reshape({2,3}));
        graph.run();check_moe(output,banks,x,ids,weights);
    }
    for(const int rows:{1,4})for(const auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16})for(float limit:{0.f,.2f}) {
        const auto small=input(rows,3,false).to(dtype);
        const auto si=tb::arange(rows*3,ids.options()).remainder(7).reshape({rows,3});
        const auto sw=tb::full({rows,3},.3,weights.options());
        check_moe(banks.forward(small,si,sw,limit),banks,small,si,sw,limit);
    }
    // Include grouped prefill and the non-MoE projection interface.
    const auto px=input(64,3,false).to(tb::kFloat16);
    const auto pi=tb::arange(192,ids.options()).remainder(7).reshape({64,3});
    const auto pw=tb::full({64,3},.3,weights.options());
    Source reference;
    reference.add("gate",{7,256,256},GGML_TYPE_Q2_0);
    reference.add("up",{7,256,256},up_type);
    reference.add("down",{7,256,256},down_type);
    CudaExecutionContext other;
    other.config.moe_host_cache_bytes=4ULL<<20;other.config.moe_device_cache_bytes=0;
    GgufMoeBanks streamed{std::make_shared<GgufRoutedWeight>(other,reference,"gate"),
        std::make_shared<GgufRoutedWeight>(other,reference,"up"),std::make_shared<GgufRoutedWeight>(other,reference,"down")};
    require((banks.forward(px,pi,pw)-streamed.forward(px,pi,pw)).abs().max().item<float>()==0,
        "Resident predictor grouped prefill differs from streamed prefill");
    require(source.read_calls()==reads,"Resident predictor reread its checkpoint after loading");
    require(gguf_expert_streamed_bytes(cache)==0,"Resident predictor transferred expert weights during inference");
    require(metric("gguf_adaptive_rounds")==0,"Resident predictor advanced target cache learning");
    GgufRoutedWeight target(execution,source,"target");
    require(!target.resident_packed.defined(),"Target experts became permanently resident");
    require(metric("gguf_predictor_resident_expert_bytes")==double(bytes),"Target registration changed predictor accounting");
    std::cout<<"Resident predictor mixed formats/changing graph routes/prefill/budget/no transfers PASS\n";
}

void run_row_graph_upload() {
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto compute=mfq_current_cuda_stream();
    context->begin_graph_pool(compute);
    struct PoolScope {
        std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~PoolScope(){context->end_graph_pool(stream);}
    } pool{context,compute};
    int can_map=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,mfq_current_cuda_device()));
    if(!can_map){std::cout<<"SKIP row graph publication: mapped host memory unavailable\n";return;}
    Source source;source.add("graph.rows",{23,256},GGML_TYPE_Q6_K);
    auto table=std::make_shared<GgufRowTable>(source,"graph.rows");auto expected=source.decoded("graph.rows");
    bool mapped_seen=false,dma_seen=false;
    struct Case {int count;bool preload;};
    for(const auto [count,preload]:{Case{4,false},Case{4,true},Case{256,false}}) {
        auto stage=make_gguf_row_stage({table},23,{count},tb::Device(tb::kCUDA,mfq_current_cuda_device()));
        mapped_seen|=stage->mapped_rows();dma_seen|=!stage->mapped_rows();
        std::vector<std::int64_t> ids(count,0);
        const auto verify=[&] {
            const auto result=stage->output().to(tb::kFloat32).cpu().contiguous();
            for(int row=0;row<count;++row)for(int column=0;column<256;++column)
                require(result.data_ptr<float>()[row*256+column]==float(mfq_half(expected.data_ptr<float>()[ids[row]*256+column])),
                    "Fenced graph row upload changed row order or bytes");
        };
        stage->upload(ids,{count});stage->decode();verify();
        const auto output=stage->output().data_ptr();
        mfq::cuda::HostBuffer flags(128,true);auto* host=static_cast<std::uint32_t*>(flags.data());std::memset(host,0,128);
        std::uint32_t* mapped=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&mapped),host,0));
        mfq::cuda::publish_mapped_flag(host+16);auto upload=mfq_get_stream_from_pool();
        {
            mfq::cuda::DecodeWindow window(compute);
            window.capture([&] {
                window.enroll({stage.get(),[&]{
                    std::memset(host,0,128);
                    if(preload)stage->upload_graph(ids,{count});
                },[&] {
                    if(preload)return;
                    mfq::cuda::wait_route_publication(host,compute);const MfqCudaStreamGuard guard(upload);
                    stage->upload_graph(ids,{count});
                    if(stage->mapped_rows())mfq::cuda::publish_mapped_flag(host+16);
                    else TYLOINFER_CUDA_CHECK(cudaLaunchHostFunc(upload.stream(),[](void* value){
                        mfq::cuda::publish_mapped_flag(static_cast<std::uint32_t*>(value));},host+16));
                },{},[&]{mfq::cuda::publish_mapped_flag(host+16);}});
                if(!preload){mfq::cuda::signal_mapped_flag(mapped,compute);mfq::cuda::wait_mapped_flag(mapped+16,compute);}
                stage->decode();
            },[]{});
            for(int revision=0;revision<12;++revision) {
                for(int i=0;i<count;++i)ids[i]=(revision*7+i*11)%23;
                window.run();verify();require(stage->output().data_ptr()==output,"Graph row storage changed across replay");
            }
            const auto saved=ids[0];ids[0]=-1;bool rejected=false;
            try{window.run();}catch(const std::out_of_range&){rejected=true;}
            require(rejected,"Invalid graph row was accepted");ids[0]=saved;
        }
        // Eager ownership remains valid after a graph (including its cancelled
        // last replay) has fenced the same mapped packet or DMA allocation.
        stage->upload(ids,{count});stage->decode();verify();
    }
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    require(mapped_seen && dma_seen,"Row graph tests did not cover both mapping and DMA");
    std::cout<<"Graph row upload mapped/DMA, changed payloads, stable storage, cancellation and eager transitions PASS\n";
}
void run_gpu_residency_snapshot(int tokens=2) {
    Source source;
    source.add("gate",{7,256,256},GGML_TYPE_Q4_0);
    source.add("up",{7,256,256},GGML_TYPE_Q5_0);
    source.add("down",{7,256,256},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=4ULL<<20;
    execution.config.moe_device_cache_bytes=2*source.find_tensor("gate")->nbytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    set_gguf_reuse_learning(cache,false);
    prepare_gguf_graph_pool(cache);
    auto x=input(tokens,3,false).to(tb::kFloat16);
    auto ids=tb::arange(tokens*3,x.options().dtype(tb::kInt64)).remainder(3).reshape({tokens,3});
    auto weights=tb::full({tokens,3},.3,x.options().dtype(tb::kFloat32));
    // Cache only Gate and Up projections before recording a full MoE. These
    // partial hits must stay in the streamed pass, without counting them twice.
    (void)banks.gate->forward(x,ids).cpu();
    (void)banks.up->forward(x,ids).cpu();
    tb::Tensor output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
    const auto metric=[&](const char* key) {
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))if(name==key)return value;
        throw std::runtime_error("GPU residency snapshot metric missing");
    };
    graph.run();check_moe(output,banks,x,ids,weights);
    const auto expert_bytes=banks.gate->expert_bytes+banks.up->expert_bytes+banks.down->expert_bytes;
    for(const bool resident:{false,true,false,true}) {
        resize_gguf_expert_cache(cache,resident?3*expert_bytes:0);
        ids.copy_((tb::arange(tokens*3,ids.options()).remainder(3)+(resident?4:0)).reshape({tokens,3}));
        if(resident) {
            (void)banks.gate->forward(x,ids).cpu();
            (void)banks.up->forward(x,ids).cpu();
            (void)banks.down->forward(x,ids).cpu();
        }
        graph.run();
        const auto refreshes=metric("gguf_gpu_resident_view_refreshes");
        graph.run();
        if(metric("gguf_direct_expert_calls")>0)
            require(metric("gguf_gpu_resident_view_refreshes")==refreshes,
                "Unchanged resident cache redundantly rebuilt its GPU routing view");
        check_moe(output,banks,x,ids,weights);
    }
    // A segmented single-projection upload can evict a later layer's weights
    // inside one window. It must retain the per-layer residency handoff.
    const auto first_ids=tb::zeros_like(ids),last_ids=tb::full(ids.sizes(),2,ids.options());
    const auto solo_ids=(tb::arange(tokens*3,ids.options()).remainder(3)+4).reshape({tokens,3});
    GgufMoeBanks later=banks;
    tb::Tensor first,last,solo;
    mfq::cuda::DecodeWindow mixed(mfq_current_cuda_stream());
    mixed.capture([&]{
        first=banks.forward(x,first_ids,weights);
        solo=banks.gate->forward(x,solo_ids);
        last=later.forward(x,last_ids,weights);
    },[]{},[&]{first={};last={};solo={};});
    for(const auto bytes:{expert_bytes,3*expert_bytes}) {
        resize_gguf_expert_cache(cache,bytes);
        mixed.run();check_moe(first,banks,x,first_ids,weights);check_moe(last,later,x,last_ids,weights);
        require((solo-banks.gate->forward(x,solo_ids)).abs().max().item<float>()==0,
            "Segmented projection changed while surrounding MoE stages replaced resident experts");
    }
    std::cout<<"GPU residency snapshot partial hits/eviction/refill/unchanged-view/segmented-window checks PASS M="<<tokens<<"\n";
}

void run_transfer_workspace_reuse() {
    Source source;
    source.add("gate",{7,256,256},GGML_TYPE_Q4_0);
    source.add("up",{7,256,256},GGML_TYPE_Q2_0);
    source.add("other_up",{7,256,256},GGML_TYPE_Q8_0);
    source.add("down",{7,256,256},GGML_TYPE_Q5_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=4ULL<<20;
    execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks a{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),
        std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    GgufMoeBanks b=a;b.up=std::make_shared<GgufRoutedWeight>(execution,source,"other_up");
    const auto cache=execution.gguf_expert_cache;
    prepare_gguf_graph_pool(cache);
    auto x=input(2,3,false).to(tb::kFloat16);
    auto ids=tb::arange(6,x.options().dtype(tb::kInt64)).remainder(3).reshape({2,3});
    auto weights=tb::full({2,3},.3,x.options().dtype(tb::kFloat32));
    tb::Tensor single;
    mfq::cuda::DecodeWindow first(mfq_current_cuda_stream());
    first.capture([&]{single=a.forward(x,ids,weights);},[]{},[&]{single={};});
    for(const int offset:{0,1,1,3,0,0}) {
        ids.copy_((tb::arange(6,ids.options()).remainder(3)+offset).reshape({2,3}));
        first.run();check_moe(single,a,x,ids,weights);
    }
    const auto metric=[&](const char* key) {
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))if(name==key)return value;
        throw std::runtime_error("GGUF shared transfer arena metric missing");
    };
    const auto arena_bytes=metric("gguf_direct_arena_bytes"),reuses=metric("gguf_direct_arena_reuses");
    auto narrow_x=x.narrow(0,0,1),narrow_ids=ids.narrow(0,0,1),narrow_weights=weights.narrow(0,0,1);
    tb::Tensor narrow_output;
    mfq::cuda::DecodeWindow narrow(mfq_current_cuda_stream());
    narrow.capture([&]{narrow_output=a.forward(narrow_x,narrow_ids,narrow_weights);},[]{},[&]{narrow_output={};});
    if(arena_bytes>0) {
        require(metric("gguf_direct_arena_bytes")==arena_bytes,"Narrow verification window duplicated the shared transfer arena");
        require(metric("gguf_direct_arena_reuses")>reuses,"Narrow verification window did not reuse a transfer arena");
    }
    for(const int offset:{2,0,3,1}) {
        ids.copy_((tb::arange(6,ids.options()).remainder(3)+offset).reshape({2,3}));
        narrow.run();check_moe(narrow_output,a,narrow_x,narrow_ids,narrow_weights);
        first.run();check_moe(single,a,x,ids,weights);
    }
    tb::Tensor left,right;
    mfq::cuda::DecodeWindow shared(mfq_current_cuda_stream());
    shared.capture([&]{left=a.forward(x,ids,weights);right=b.forward(x,ids,weights);},[]{},[&]{left={};right={};});
    for(const int offset:{1,2,0,2}) {
        ids.copy_((tb::arange(6,ids.options()).remainder(3)+offset).reshape({2,3}));
        shared.run();check_moe(left,a,x,ids,weights);check_moe(right,b,x,ids,weights);
    }
    // A larger graph grows the arena without changing the smaller graphs'
    // captured addresses. Another graph shares the new allocation and must
    // invalidate images overwritten by the multi-layer graph.
    tb::Tensor other_output;
    mfq::cuda::DecodeWindow other(mfq_current_cuda_stream());
    other.capture([&]{other_output=b.forward(narrow_x,narrow_ids,narrow_weights);},[]{},[&]{other_output={};});
    for(const int offset:{1,0,2}) {
        ids.copy_((tb::arange(6,ids.options()).remainder(3)+offset).reshape({2,3}));
        shared.run();check_moe(left,a,x,ids,weights);check_moe(right,b,x,ids,weights);
        other.run();check_moe(other_output,b,narrow_x,narrow_ids,narrow_weights);
    }
    first.run();check_moe(single,a,x,ids,weights);
    narrow.run();check_moe(narrow_output,a,narrow_x,narrow_ids,narrow_weights);
}

void run_borrowed_gguf_prefill(bool quantized_prefill=false) {
    const auto* enabled=std::getenv("TYLOINFER_GGUF_STRATA_CACHE");
    if(enabled && enabled[0]=='0')return;
    Source source;
    for(const auto* role:{"gate","up","down"})source.add(role,{64,256,256},quantized_prefill?GGML_TYPE_Q8_0:GGML_TYPE_Q4_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=16ULL<<20;
    execution.config.moe_device_cache_bytes=3*source.find_tensor("gate")->nbytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    auto x=input(256,1,false).to(tb::kFloat16);
    auto ids=tb::arange(256,x.options().dtype(tb::kInt64)).remainder(64).reshape({256,1});
    auto weights=tb::full({256,1},.3,x.options().dtype(tb::kFloat32));
    (void)banks.forward(x,ids,weights).cpu();
    restore_gguf_decode_workspace(execution.gguf_expert_cache);
    const auto lendable=gguf_prefill_lendable_bytes(execution.gguf_expert_cache);
    require(lendable>0,"GGUF prompt did not initialize Strata slots");
    x=x.narrow(0,0,64).contiguous();weights=weights.narrow(0,0,64).contiguous();
    ids=(tb::arange(64,ids.options()).remainder(3)+61).reshape({64,1});
    const auto expected=banks.forward(x,ids,weights).cpu();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto old_limit=context->memory_stats().limit;
    context->reclaim_idle_memory();
    context->set_memory_limit(std::max(context->resident_memory_usage(),context->memory_stats().allocated)+lendable/4);
    try {
        GgufPrefillWorkspaceLoan loan(execution.gguf_expert_cache);
        require(loan.begin(lendable)>0,"GGUF second prefill did not borrow its cache tail");
        auto output=banks.forward(x,ids,weights);
        loan.suspend();context->set_memory_limit(old_limit);
        auto preserved=output.clone();output={};
        // The last two staging arenas remain in the cache's transfer owners.
        // finish must retire those owners before checking live loan tensors.
        loan.finish();
        const auto recovered=preserved.cpu();
        require(recovered.nbytes()==expected.nbytes() &&
            std::memcmp(recovered.data_ptr(),expected.data_ptr(),expected.nbytes())==0,
            "GGUF borrowed prefill changed the unborrowed PP result");
        const auto decoded_x=x.narrow(0,0,1),decoded_ids=ids.narrow(0,0,1),decoded_weights=weights.narrow(0,0,1);
        check_moe(banks.forward(decoded_x,decoded_ids,decoded_weights),banks,decoded_x,decoded_ids,decoded_weights);
        context->reclaim_idle_memory();
        context->set_memory_limit(std::max(context->resident_memory_usage(),context->memory_stats().allocated)+lendable/4);
        reserve_gguf_expert_workspace(execution.gguf_expert_cache,lendable/2);
        const auto remaining=gguf_prefill_lendable_bytes(execution.gguf_expert_cache);
        require(context->available_memory()>=lendable/2 && remaining<lendable,
            "GGUF fixed slots retained physical VRAM needed for KV growth");
        if(remaining) {
            // A subsequent loan owns only the still-mapped slots, not the
            // permanently removed tail retained for possible later growth.
            context->reclaim_idle_memory();
            GgufPrefillWorkspaceLoan partial(execution.gguf_expert_cache);
            const auto borrowed=partial.begin(context->available_memory()+std::min(remaining/2,std::size_t(1)<<20));
            require(borrowed>0,"Shrunk GGUF cache did not lend its live tail");
            const auto transferred=gguf_expert_streamed_bytes(execution.gguf_expert_cache);
            partial.suspend();partial.finish();
            require(gguf_expert_streamed_bytes(execution.gguf_expert_cache)-transferred<=borrowed,
                "Shrunk GGUF loan restored experts outside its live mapping");
        }
        context->set_memory_limit(old_limit);
        check_moe(banks.forward(decoded_x,decoded_ids,decoded_weights),banks,decoded_x,decoded_ids,decoded_weights);
    }catch(...) {context->set_memory_limit(old_limit);throw;}
}
struct ScopedSnapshot {
    std::optional<std::string> previous;
    std::string key;
    explicit ScopedSnapshot(const char* value,const char* variable="TYLOINFER_GGUF_HOST_SNAPSHOT"):key(variable) {
        if(const auto* old=std::getenv(key.c_str()))previous=old;
        set(value);
    }
    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(key.c_str(),value?value:"");
#else
        if(value)setenv(key.c_str(),value,1);else unsetenv(key.c_str());
#endif
    }
    ~ScopedSnapshot(){set(previous?previous->c_str():nullptr);}
};
void run_graph_expert_batches() {
    ScopedSnapshot snapshot("0"),cpu_setting("0","TYLOINFER_GGUF_CPU_EXPERTS");
    int can_map=0;
    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&can_map,cudaDevAttrCanMapHostMemory,mfq_current_cuda_device()));
    if(!can_map){std::cout<<"GGUF graph expert batch SKIP: mapped host handoff unavailable\n";return;}
    for(bool cpu:{false,true}) {
        if(cpu && !mfq::cpu::gguf::available())continue;
        cpu_setting.set(cpu?"2":"0");
        Source source;source.enable_batches();
        source.add("gate",{12,256,256},GGML_TYPE_Q4_0);
        source.add("up",{12,256,256},GGML_TYPE_Q5_0);
        source.add("down",{12,256,256},GGML_TYPE_Q8_0);
        CudaExecutionContext execution;
        const auto expert_bytes=(source.find_tensor("gate")->nbytes+source.find_tensor("up")->nbytes+
            source.find_tensor("down")->nbytes)/12;
        execution.config.moe_host_cache_bytes=3*expert_bytes;
        execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        auto cache=execution.gguf_expert_cache;
        set_gguf_reuse_learning(cache,false);
        require(prepare_gguf_graph_pool(cache),"GGUF cold-batch graph pool missing");
        auto x=input(2,3,false).to(tb::kFloat16);
        auto ids=tb::arange(6,x.options().dtype(tb::kInt64)).remainder(3).reshape({2,3});
        auto weights=tb::full({2,3},.3,x.options().dtype(tb::kFloat32));
        tb::Tensor output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
        for(int rotation:{3,6}) {
            const auto before=source.batch_calls();
            ids.copy_((tb::arange(6,ids.options()).remainder(3)+rotation).reshape({2,3}));
            graph.run();check_moe(output,banks,x,ids,weights);
            require(source.batch_calls()>before,"GGUF cold graph experts did not use batch reads");
        }
        ids.copy_((tb::arange(6,ids.options()).remainder(3)+9).reshape({2,3}));
        source.fail_next_batch();
        bool failed=false;
        try{graph.run();}catch(const std::runtime_error& error){failed=std::string(error.what())=="fixture batch read failure";}
        require(failed,"GGUF cold graph batch failure was not propagated");
        // A failed window is deliberately discarded; the next request records
        // a fresh graph against the same cache and must reread partial images.
        mfq::cuda::DecodeWindow recovery(mfq_current_cuda_stream());
        recovery.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
        recovery.run();check_moe(output,banks,x,ids,weights);
        std::cout<<"GGUF cold graph batches, bounded RAM, partial-I/O rollback/recovery PASS cpu="<<cpu
            <<" batches="<<source.batch_calls()<<'\n';
    }
}
void run_slot_transition_ram_coverage() {
    ScopedSnapshot reserve("0","TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
    ScopedSnapshot snapshot("0"),slots("1","TYLOINFER_GGUF_STRATA_CACHE");
    Source source;
    for(const auto* role:{"gate","up","down"})source.add(role,{5,256,256},GGML_TYPE_Q4_0);
    const auto expert_bytes=3*source.find_tensor("gate")->nbytes/5;
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=3*expert_bytes;
    execution.config.moe_device_cache_bytes=2*expert_bytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),
        std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* name) {
        for(const auto& item:gguf_expert_memory_metrics(cache))if(item.first==name)return std::size_t(item.second);
        throw std::runtime_error("Slot transition coverage metric missing");
    };
    auto x=tb::full({1,256},.0001,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
    auto ids=tb::zeros({1,1},x.options().dtype(tb::kInt64));
    auto scales=tb::full({1,1},1.,x.options().dtype(tb::kFloat32));
    require(prepare_gguf_graph_pool(cache),"Slot transition fixture has no graph pool");
    tb::Tensor output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,scales);},[]{});
    for(int expert=0;expert<5;++expert){ids.fill_(expert);graph.run();check_moe(output,banks,x,ids,scales);}
    reclaim_gguf_expert_duplicates(cache,1);
    require(metric("ssd_expert_payload_bytes")==0,"Slot transition fixture does not initially cover every expert");
    restore_gguf_decode_workspace(cache,0,0);
    require(metric("gguf_strata_slot_cache_bytes")==2*expert_bytes,"Slot transition did not seed the fixed arena");
    require(metric("ssd_expert_payload_bytes")==0,
        "Fixed-slot migration lost experts that fit in combined RAM and VRAM");
    const auto reads=source.read_calls();
    for(int expert=0;expert<5;++expert){ids.fill_(expert);graph.run();check_moe(output,banks,x,ids,scales);}
    require(source.read_calls()==reads,"Warm fixed-slot replay reread an expert retained before migration");
    std::cout<<"GGUF fixed-slot migration preserves complete combined RAM/VRAM coverage and warm source hits PASS\n";
}
void run_decode_seeded_slots() {
    ScopedSnapshot reserve("0","TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
    for(bool enabled:{true,false}) {
        ScopedSnapshot slots(enabled?"1":"0","TYLOINFER_GGUF_STRATA_CACHE");
        Source source;
        for(const auto* layer:{"first","second"}) {
            source.add(std::string(layer)+".gate",{8,256,256},GGML_TYPE_Q4_0);
            source.add(std::string(layer)+".up",{8,256,256},GGML_TYPE_Q5_0);
            source.add(std::string(layer)+".down",{8,256,256},GGML_TYPE_Q8_0);
        }
        const auto bytes=(source.find_tensor("first.gate")->nbytes+source.find_tensor("first.up")->nbytes+
            source.find_tensor("first.down")->nbytes)/8;
        CudaExecutionContext execution;execution.config.moe_host_cache_bytes=16ULL<<20;
        execution.config.moe_device_cache_bytes=2*bytes;
        const auto banks=[&](const char* layer) {
            return GgufMoeBanks{std::make_shared<GgufRoutedWeight>(execution,source,std::string(layer)+".gate"),
                std::make_shared<GgufRoutedWeight>(execution,source,std::string(layer)+".up"),
                std::make_shared<GgufRoutedWeight>(execution,source,std::string(layer)+".down")};
        };
        auto first=banks("first"),second=banks("second");const auto cache=execution.gguf_expert_cache;
        const auto metric=[&](const char* name) {
            for(const auto& item:gguf_expert_memory_metrics(cache))if(item.first==name)return item.second;
            throw std::runtime_error("Decode-seeded cache metric missing");
        };
        const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
        auto x=tb::full({1,256},.0001,options),ids=tb::zeros({1,1},options.dtype(tb::kInt64));
        auto scales=tb::full({1,1},1.,options.dtype(tb::kFloat32));
        tb::Tensor a,b;require(prepare_gguf_graph_pool(cache),"Decode-seeded cache has no graph pool");
        mfq::cuda::DecodeWindow one(mfq_current_cuda_stream()),two(mfq_current_cuda_stream());
        one.capture([&]{a=first.forward(x,ids,scales);},[]{});one.run();
        restore_gguf_decode_workspace(cache,0,0);
        require(metric("gguf_strata_slot_cache_bytes")==0,"Partial decode history created an undersized slot arena");
        two.capture([&]{b=second.forward(x,ids,scales);},[]{});two.run();
        restore_gguf_decode_workspace(cache,0,0);
        require((metric("gguf_strata_slot_cache_bytes")>0)==enabled,"Decode history did not seed enabled fixed slots or ignored explicit disable");
        require(metric("gguf_decode_seed_rebalances")==double(enabled),"Decode seed did not initialize exactly once at the safe boundary");
        for(int expert:{0,1,1,1,1,2,2,2,2}) {
            ids.fill_(expert);one.run();check_moe(a,first,x,ids,scales);
            two.run();check_moe(b,second,x,ids,scales);
        }
        restore_gguf_decode_workspace(cache,0,0);
        require(metric("gguf_decode_seed_rebalances")==double(enabled),"Existing slot arena was needlessly rebuilt");
        require(metric("resident_expert_cache_bytes")<=2*bytes,"Decode-seeded slots exceeded configured capacity");
    }
    std::cout<<"GGUF prefix-only decode history, full-capacity seed, disable override, mixed-role retained Graph and adaptive exchanges PASS\n";
}
void run_lazy_decode_slots() {
    ScopedSnapshot reserve("0","TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
    ScopedSnapshot slots("1","TYLOINFER_GGUF_STRATA_CACHE");
    ScopedSnapshot snapshot("0");
    Source source;
    source.add("gate",{16,256,256},GGML_TYPE_Q4_0);
    source.add("up",{16,256,256},GGML_TYPE_Q5_0);
    source.add("down",{16,256,256},GGML_TYPE_Q8_0);
    const auto bytes=(source.find_tensor("gate")->nbytes+source.find_tensor("up")->nbytes+
        source.find_tensor("down")->nbytes)/16;
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=16ULL<<20;
    execution.config.moe_device_cache_bytes=4*bytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* name) {
        for(const auto& item:gguf_expert_memory_metrics(cache))if(item.first==name)return item.second;
        throw std::runtime_error("Lazy decode cache metric missing");
    };
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto x=tb::full({1,256},.0001,options),ids=tb::zeros({1,1},options.dtype(tb::kInt64));
    auto scales=tb::full({1,1},1.,options.dtype(tb::kFloat32));tb::Tensor output;
    require(prepare_gguf_graph_pool(cache),"Lazy decode cache has no graph pool");
    mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,scales);},[]{});graph.run();
    wait_gguf_expert_admission(cache);
    const auto reads=source.read_calls(),transferred=gguf_expert_streamed_bytes(cache);
    restore_gguf_decode_workspace(cache,0,0);
    require(metric("gguf_strata_slot_cache_bytes")==4*bytes,"Sparse decode history did not establish full slot capacity");
    require(metric("gguf_slot_untouched_slots")==3,"Cold slots were filled without observed demand");
    require(metric("resident_expert_cache_bytes")==bytes,"Empty slot addresses were published as resident experts");
    require(source.read_calls()==reads,"Lazy initialization read cold source weights");
    require(gguf_expert_streamed_bytes(cache)-transferred<=bytes,"Lazy initialization uploaded cold weights");
    require(metric("gguf_decode_seed_rebalances")==1,"Sparse decode history did not seed exactly once");
    ids.fill_(15);begin_gguf_speculative_routes(cache,1);graph.run();commit_gguf_speculative_routes(cache,1);
    wait_gguf_expert_admission(cache);check_moe(output,banks,x,ids,scales);
    const auto first_miss=gguf_expert_streamed_bytes(cache);
    for(const auto& bank:{banks.gate,banks.up,banks.down})(void)bank->forward(x,ids).cpu();
    require(gguf_expert_streamed_bytes(cache)==first_miss,"An empty slot imposed the occupied-expert replacement threshold");
    for(int expert:{3,15}) {
        ids.fill_(expert);
        for(int step=0;step<8;++step){graph.run();wait_gguf_expert_admission(cache);check_moe(output,banks,x,ids,scales);}
        const auto before=gguf_expert_streamed_bytes(cache);
        for(const auto& bank:{banks.gate,banks.up,banks.down})(void)bank->forward(x,ids).cpu();
        require(gguf_expert_streamed_bytes(cache)==before,"Lazy assigned or replacement slot failed to admit its observed expert");
    }
    require(metric("gguf_slot_untouched_slots")<3,"Observed cold slot remained empty");
    require(metric("resident_expert_cache_bytes")<=4*bytes,"Lazy slots exceeded the configured cache capacity");
    restore_gguf_decode_workspace(cache,0,0);
    require(metric("gguf_decode_seed_rebalances")==1,"Lazy slot cache was initialized again");
    std::cout<<"GGUF sparse-history full-capacity lazy slots, unpublished cold addresses and retained Graph admission PASS\n";
}
void run_lazy_admission_retry() {
    ScopedSnapshot reserve("0","TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
    ScopedSnapshot slots("1","TYLOINFER_GGUF_STRATA_CACHE");ScopedSnapshot snapshot("0");
    Source source;for(const auto* role:{"gate","up","down"})source.add(role,{4,256,256},GGML_TYPE_Q8_0);
    const auto bytes=3*source.find_tensor("gate")->nbytes/4;
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=1;execution.config.moe_device_cache_bytes=2*bytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* name){for(const auto& item:gguf_expert_memory_metrics(cache))if(item.first==name)return item.second;throw std::runtime_error("Retry metric missing");};
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto x=tb::full({1,256},.0001,options),ids=tb::zeros({1,1},options.dtype(tb::kInt64));
    auto scales=tb::full({1,1},1.,options.dtype(tb::kFloat32));tb::Tensor output;
    require(prepare_gguf_graph_pool(cache),"Lazy retry has no graph pool");mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,scales);},[]{});graph.run();wait_gguf_expert_admission(cache);
    restore_gguf_decode_workspace(cache,0,0);
    require(metric("gguf_slot_untouched_slots")==1,"Lazy retry fixture has no single empty slot");
    ids.fill_(1);begin_gguf_speculative_routes(cache,1);graph.run();source.fail_next_read(true,1);
    bool failed=false;
    try{commit_gguf_speculative_routes(cache,1);wait_gguf_expert_admission(cache);}
    catch(const std::runtime_error& error){failed=std::string(error.what())=="fixture single read failure";}
    require(failed,"Partial lazy admission did not propagate its second-role source error");
    require(metric("gguf_slot_untouched_slots")==0,"Partial failure did not prepare its last cold slot");
    begin_gguf_speculative_routes(cache,1);graph.run();commit_gguf_speculative_routes(cache,1);
    wait_gguf_expert_admission(cache);
    const auto before=gguf_expert_streamed_bytes(cache);
    for(const auto& bank:{banks.gate,banks.up,banks.down})(void)bank->forward(x,ids).cpu();
    require(gguf_expert_streamed_bytes(cache)==before,"Previously touched empty slot did not retry its missing projections");
    require(metric("resident_expert_cache_bytes")==2*bytes,"Retry lost an existing resident expert");
    check_moe(output,banks,x,ids,scales);
    std::cout<<"GGUF partial lazy admission error, touched-slot retry and retained Graph recovery PASS\n";
}
void run_reusable_expert_pool_budget() {
    ScopedSnapshot slots("0","TYLOINFER_GGUF_STRATA_CACHE");
    Source source;
    for(const auto* role:{"gate","up","down"})source.add(role,{4,256,256},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=2ULL<<20;
    execution.config.moe_device_cache_bytes=16ULL<<20;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto x=tb::full({1,256},.0001,options),ids=tb::zeros({1,1},options.dtype(tb::kInt64));
    auto weights=tb::full({1,1},1.,options.dtype(tb::kFloat32));
    check_moe(banks.forward(x,ids,weights),banks,x,ids,weights);
    std::vector<tb::Tensor> fragments;
    {
        const mfq::cuda::RoutedWeightAllocationScope resident;
        for(int i=0;i<128;++i)fragments.push_back(tb::empty({65536},options.dtype(tb::kUInt8)));
    }
    context->reclaim_idle_memory();
    CUmemoryPool handle=nullptr;
    if(cuPointerGetAttribute(&handle,CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,
            CUdeviceptr(reinterpret_cast<std::uintptr_t>(fragments.front().data_ptr())))!=CUDA_SUCCESS || !handle) {
        std::cout<<"SKIP expert pool budget: allocation is not backed by an async memory pool\n";
        return;
    }
    for(std::size_t i=1;i<fragments.size();i+=2)fragments[i]={};
    context->reclaim_idle_memory();
    const auto previous_limit=context->memory_stats().limit;
    try {
        context->set_memory_limit(std::max(context->resident_memory_usage(),context->memory_stats().allocated)+(1ULL<<20));
        std::uint64_t used=0,reserved=0;
        const auto pool=reinterpret_cast<cudaMemPool_t>(handle);
        TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrUsedMemCurrent,&used));
        TYLOINFER_CUDA_CHECK(cudaMemPoolGetAttribute(pool,cudaMemPoolAttrReservedMemCurrent,&reserved));
        const auto reusable=reserved>used?reserved-used:0,available=context->available_memory();
        if(reusable<=available) {
            context->set_memory_limit(previous_limit);
            std::cout<<"SKIP expert pool budget: driver returned all fragmented free pages\n";
            return;
        }
        std::size_t resident=0;
        for(const auto& item:gguf_expert_memory_metrics(cache))
            if(item.first=="resident_expert_cache_bytes")resident=std::size_t(item.second);
        require(resident>0,"Reusable expert pool fixture has no cache images");
        const auto before=gguf_expert_cache_budget_bytes(cache);
        reserve_gguf_expert_workspace(cache,available/2);
        require(gguf_expert_cache_budget_bytes(cache)>=std::min(before,resident+reusable),
            "Workspace reservation discarded reusable expert pool capacity");
        require(context->available_memory()>=available/2,"Cache reuse consumed the physical workspace reservation");
        context->set_memory_limit(previous_limit);
        check_moe(banks.forward(x,ids,weights),banks,x,ids,weights);
        std::cout<<"GGUF fragmented expert pool capacity retained across workspace reservation PASS reusable_bytes="<<reusable<<'\n';
    } catch(...) {context->set_memory_limit(previous_limit);throw;}
}
void run_decode_workspace_budget() {
    ScopedSnapshot reserve(nullptr,"TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
    ScopedSnapshot slots("1","TYLOINFER_GGUF_STRATA_CACHE");
    Source source;
    for(const auto* role:{"gate","up","down"})source.add(role,{48,256,256},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=32ULL<<20;
    execution.config.moe_device_cache_bytes=16ULL<<20;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto x=tb::full({48,256},.0001,options);
    auto ids=tb::arange(48,options.dtype(tb::kInt64)).reshape({48,1});
    auto weights=tb::full({48,1},1.,options.dtype(tb::kFloat32));
    (void)banks.forward(x,ids,weights).cpu();
    resize_gguf_expert_cache(cache,0);
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->reclaim_idle_memory();
    const auto previous_limit=context->memory_stats().limit;
    if(const auto physical=mfq::cuda::physical_device_memory_stats(mfq_current_cuda_device())) {
        require(physical->total>0 && physical->free<=physical->total && physical->used<=physical->total,
            "driver-wide VRAM accounting returned invalid sizes");
        std::cout<<"GGUF physical VRAM query total="<<physical->total<<" free="<<physical->free<<'\n';
    }
    const auto metric=[&](const char* name) {
        for(const auto& item:gguf_expert_memory_metrics(cache))if(item.first==name)return item.second;
        throw std::runtime_error("GGUF decode reserve metric missing");
    };
    constexpr auto prepared_reserve=std::size_t(732)<<20;
    if(context->available_memory()<prepared_reserve+(96ULL<<20)) {
        std::cout<<"SKIP automatic decode reserve: insufficient available memory for its residency allowance and test allocations\n";
        return;
    }
    const auto roomy_limit=std::max(context->resident_memory_usage(),context->memory_stats().allocated)+prepared_reserve+(96ULL<<20);
    context->set_memory_limit(roomy_limit);
    try {
        resize_gguf_expert_cache(cache,16ULL<<20);
        restore_gguf_decode_workspace(cache);
        require(gguf_expert_cache_budget_bytes(cache)==0,"GGUF unprepared caller lost its capture allowance");
        restore_gguf_decode_workspace(cache,0,16ULL<<20);
        require(metric("gguf_decode_workspace_prepared")==1,"GGUF prepared decoder reserve was ignored");
        require(metric("gguf_decode_workspace_reserve_bytes")==double(prepared_reserve),"GGUF physical residency or upload reserve was lost");
        require(gguf_expert_cache_budget_bytes(cache)>0 && gguf_expert_cache_budget_bytes(cache)<=16ULL<<20,
            "GGUF prepared decode did not reuse spare memory within configured cache budget");
        (void)banks.forward(x,ids,weights).cpu();
        restore_gguf_decode_workspace(cache,0,16ULL<<20);
        const auto one=x.narrow(0,0,1),chosen=ids.narrow(0,0,1),scales=weights.narrow(0,0,1);
        check_moe(banks.forward(one,chosen,scales),banks,one,chosen,scales);
        auto output=tb::Tensor{};
        require(prepare_gguf_graph_pool(cache),"GGUF budget test has no graph pool");
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{output=banks.forward(one,chosen,scales);},[]{});
        window.run();check_moe(output,banks,one,chosen,scales);
        auto tail_id=tb::full({1,1},47,ids.options());tb::Tensor tail_output;
        mfq::cuda::DecodeWindow tail_window(mfq_current_cuda_stream());
        begin_gguf_speculative_routes(cache,1);
        tail_window.capture([&]{tail_output=banks.forward(one,tail_id,scales);},[]{});
        discard_gguf_speculative_routes(cache);
        tail_window.run();check_moe(tail_output,banks,one,tail_id,scales);
        const auto available=context->available_memory();
        const auto arena_before=metric("gguf_strata_slot_cache_bytes");
        const auto segment=metric("gguf_slot_max_segment_bytes");
        const auto rebuilds=metric("gguf_slot_rebuilds");
        if(mfq::cuda::gguf::grouped_f16_available(mfq_current_cuda_device()))
            require(arena_before>0,"GGUF pressure fixture did not create its fixed slot arena");
        reserve_gguf_expert_workspace(cache,available+(1ULL<<20));
        require(context->available_memory()>=available+(1ULL<<20),"GGUF workspace pressure did not return physical pages");
        if(metric("gguf_slot_segmented_arenas") && arena_before>segment+(1ULL<<20)) {
            require(metric("gguf_strata_slot_cache_bytes")>0 && metric("gguf_strata_slot_cache_bytes")<arena_before,
                "Small workspace pressure discarded the complete segmented expert cache");
            require(metric("gguf_slot_shrinks")>0 && metric("gguf_slot_rebuilds")==rebuilds,
                "Segmented expert cache rebuilt instead of preserving its prefix");
        }else if(arena_before)require(gguf_prefill_lendable_bytes(cache)<std::size_t(arena_before),
            "GGUF workspace pressure retained all revocable expert slots");
        else require(gguf_prefill_lendable_bytes(cache)==0,"GGUF unseeded cache reported fixed slots");
        window.run();check_moe(output,banks,one,chosen,scales);
        if(metric("gguf_slot_segmented_arenas") && arena_before>segment+(1ULL<<20)) {
            // Admit an expert from the unmapped tail while it is absent. Its
            // former tail slot must then belong to the displaced live expert.
            for(int step=0;step<8;++step) {
                begin_gguf_speculative_routes(cache,1);tail_window.run();commit_gguf_speculative_routes(cache,1);
                wait_gguf_expert_admission(cache);check_moe(tail_output,banks,one,tail_id,scales);
            }
            const auto streamed=gguf_expert_streamed_bytes(cache);
            for(const auto& bank:{banks.gate,banks.up,banks.down})(void)bank->forward(one,tail_id).cpu();
            require(gguf_expert_streamed_bytes(cache)==streamed,"Shrunk cache did not adapt to its missing tail expert");
        }
        restore_gguf_decode_workspace_after_capture(cache);
        require(metric("gguf_decode_workspace_prepared")==1 &&
            metric("gguf_decode_workspace_reserve_bytes")==double(prepared_reserve) &&
            metric("resident_expert_cache_bytes")>0,
            "GGUF capture did not restore its previous prepared policy and resident experts");
        window.run();check_moe(output,banks,one,chosen,scales);
        tail_window.run();check_moe(tail_output,banks,one,tail_id,scales);
        if(metric("gguf_slot_segmented_arenas") && arena_before>segment+(1ULL<<20)) {
            require(metric("gguf_slot_grows")>0 && metric("gguf_slot_rebuilds")==rebuilds,
                "Returned workspace did not refill only the removed expert-cache tail");
            std::cout<<"GGUF segmented pressure, missing-tail adaptation, owner-preserving growth and retained Graph PASS\n";
        }
        for(int expert=0;expert<48;++expert) {
            const auto selected=tb::full({1,1},expert,ids.options());
            check_moe(banks.forward(one,selected,scales),banks,one,selected,scales);
        }
        restore_gguf_decode_workspace(cache,5ULL<<20,16ULL<<20);
        require(metric("gguf_decode_workspace_reserve_bytes")==double(prepared_reserve+(5ULL<<20)),"GGUF prepared allowance omitted KV growth");
        // Retained graph allocations can leave a positive amount of free
        // memory which is nevertheless below the decoder's reserve. Existing
        // experts must yield physical pages before that graph runs again.
        // External slot arenas need a driver process-usage meter for this
        // quota test; allocator-owned bytes alone cannot observe their frees.
        if(mfq::cuda::process_device_memory_stats(mfq_current_cuda_device())) {
        const auto needed=std::size_t(metric("gguf_decode_workspace_reserve_bytes"));
        const auto shortfall=std::min(std::size_t(1)<<20,std::size_t(metric("resident_expert_cache_bytes"))/2);
        require(shortfall>0,"Workspace pressure fixture has no resident expert pages to reclaim");
        context->reclaim_idle_memory();
        context->set_memory_limit(context->resident_memory_usage()+needed-shortfall);
        require(context->available_memory()<needed,"Workspace pressure fixture did not consume the promised allowance");
        restore_gguf_decode_workspace(cache,5ULL<<20,16ULL<<20);
        require(context->available_memory()>=needed,"Existing GPU experts consumed the retained decoder's free-memory reserve");
        window.run();check_moe(output,banks,one,chosen,scales);
        context->set_memory_limit(roomy_limit);
        }
        {
            ScopedSnapshot explicit_reserve("0.0625","TYLOINFER_CUDA_WORKSPACE_RESERVE_GIB");
            restore_gguf_decode_workspace(cache,5ULL<<20,16ULL<<20);
            require(metric("gguf_decode_workspace_prepared")==0 &&
                metric("gguf_decode_workspace_reserve_bytes")==double(69ULL<<20),
                "GGUF prepared reserve overrode explicit user allowance");
        }
        restore_gguf_decode_workspace(cache,SIZE_MAX,16ULL<<20);
        require(metric("gguf_decode_workspace_reserve_bytes")==double(SIZE_MAX),"GGUF reserve overflow wrapped");
        context->set_memory_limit(previous_limit);
    } catch(...) {context->set_memory_limit(previous_limit);throw;}
    std::cout<<"GGUF prepared/unprepared/explicit/growth reserve and graph-after-reclaim PASS\n";
}

void run_host_layout_plan() {
    Source source;
    const std::string stem="layers.0.mlp.experts.";
    source.add(stem+"gate.weight",{7,128,256},GGML_TYPE_IQ3_XXS);
    source.add(stem+"up.weight",{7,128,256},GGML_TYPE_IQ4_XS);
    source.add(stem+"down.weight",{7,256,128},GGML_TYPE_Q8_0);
    source.add("layers.1.mlp.experts.gate.weight",{3,128,256},GGML_TYPE_Q8_0);
    source.add("layers.1.mlp.experts.up.weight",{4,128,256},GGML_TYPE_Q8_0);
    source.add("predictor.layers.0.mlp.experts.gate.weight",{7,128,256},GGML_TYPE_Q8_0);
    const auto plan=GgufHostSnapshot::plan(source);
    require(plan && plan->banks.size()==5,"host layout included predictor or omitted a routed role");
    const auto& g=plan->banks.at(stem+"gate.weight");
    const auto& u=plan->banks.at(stem+"up.weight");
    const auto& d=plan->banks.at(stem+"down.weight");
    require(g.stride==u.stride && g.stride==d.stride && g.offset+g.expert_bytes==u.offset &&
        u.offset+u.expert_bytes==d.offset && d.offset+d.expert_bytes==g.offset+g.stride,
        "three routed roles do not share a contiguous expert image");
    for(const auto* role:{"gate","up"}) {
        const auto& bank=plan->banks.at(std::string("layers.1.mlp.experts.")+role+".weight");
        require(bank.stride==bank.expert_bytes,"incomplete/mismatched expert banks were interleaved");
    }
    std::cout<<"GGUF host layout mixed roles/partial banks/expert-count boundaries PASS\n";
}
void run_paired_host_cache() {
    ScopedSnapshot setting("0");
    Source source;
    const std::array<std::string,3> names{"layers.0.mlp.experts.gate.weight",
        "layers.0.mlp.experts.up.weight","layers.0.mlp.experts.down.weight"};
    source.add(names[0],{7,640,256},GGML_TYPE_IQ3_XXS);
    source.add(names[1],{7,640,256},GGML_TYPE_IQ3_XXS);
    source.add(names[2],{7,256,640},GGML_TYPE_IQ4_NL);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=8ULL<<20;
    execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,names[0]),
        std::make_shared<GgufRoutedWeight>(execution,source,names[1]),
        std::make_shared<GgufRoutedWeight>(execution,source,names[2])};
    const auto metric=[&](const char* key) {
        for(const auto& [name,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))if(name==key)return value;
        throw std::runtime_error("paired host metric missing");
    };
    require(metric("gguf_host_snapshot_bytes")==0 && metric("gguf_host_gate_up_pair_bytes")==14*banks.gate->expert_bytes,
        "bounded cache did not pair immutable Gate/Up images");
    auto x=(input(64,1,false)*.001).to(tb::kFloat16);
    auto ids=tb::arange(64*3,x.options().dtype(tb::kInt64)).remainder(7).reshape({64,3});
    auto weights=tb::full({64,3},.3,x.options().dtype(tb::kFloat32));
    ScopedSnapshot direct("0","TYLOINFER_GGUF_PREFILL_DIRECT");
    ScopedSnapshot native("0","TYLOINFER_MOE_NATIVE_DMA_BATCH");
    ScopedSnapshot dma("0","TYLOINFER_GGUF_PREFILL_DMA");
    const auto expected=banks.forward(x,ids,weights,.01f).cpu();
    direct.set("1");
    native.set("1");
    dma.set("1");
    const auto copies=metric("gguf_prefill_dma_copies"),images=metric("gguf_prefill_dma_images");
    const auto batches=metric("gguf_prefill_dma_batches");
    const auto actual=banks.forward(x,ids,weights,.01f).cpu();
    require(std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.nbytes())==0,
        "paired RAM cache changed guarded MMQ output");
    require(metric("gguf_prefill_dma_images")-images==21 && metric("gguf_prefill_dma_copies")-copies<=14,
        "paired RAM cache did not merge Gate/Up DMA");
    require(source.read_calls()==21,"paired cache reread a Gate matrix or retained a second source image");
    if(mfq::cuda::NativeDmaBatch::available())require(metric("gguf_prefill_dma_batches")>batches,
        "PP batch DMA was available but not submitted");
    native.set("2");
    const auto hinted=banks.forward(x,ids,weights,.01f).cpu();
    require(std::memcmp(expected.data_ptr(),hinted.data_ptr(),hinted.nbytes())==0,
        "DMA overlap hint changed guarded MMQ output");
    if(mfq::cuda::NativeDmaBatch::available())require(metric("gguf_prefill_dma_overlap_batches")>0,
        "DMA overlap hint did not reach the batch submission");
    require(metric("gguf_transfer_reap_queries")>0 &&
        metric("gguf_transfer_reap_queries")<metric("gguf_transfer_reap_images"),
        "shared upload fences were queried once per image");
    require(metric("gguf_prefill_phased_groups")>0,
        "pinned PP did not publish GU before the Down upload completed");
    std::cout<<"GGUF bounded paired cache/guarded Down/byte parity/21 images <=14 DMA PASS\n";
}
void run_prefill_partial_fence() {
    ScopedSnapshot snapshot("0");
    Source source;
    constexpr int experts=17,width=4096,hidden=128,tokens=129;
    source.add("gate",{experts,hidden,width},GGML_TYPE_IQ3_XXS);
    source.add("up",{experts,hidden,width},GGML_TYPE_IQ3_XXS);
    source.add("down",{experts,width,hidden},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=64ULL<<20;
    execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),
        std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    auto host=tb::empty({tokens,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for(int i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.001f*std::sin(i*.17f);
    auto x=host.to(tb::kCUDA).to(tb::kFloat16);
    auto ids=tb::arange(tokens*3,x.options().dtype(tb::kInt64)).remainder(experts).reshape({tokens,3});
    auto weights=tb::full({tokens,3},.3,x.options().dtype(tb::kFloat32));
    ScopedSnapshot dma("0","TYLOINFER_GGUF_PREFILL_DMA");
    const auto expected=banks.forward(x,ids,weights,.01f).cpu();
    dma.set("1");
    const auto actual=banks.forward(x,ids,weights,.01f).cpu();
    require(std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.nbytes())==0,
        "partial GU readiness changed large-Down upload output");
    double phased=0;
    for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
        if(key=="gguf_prefill_phased_groups")phased=value;
    require(phased>0,"large-Down fixture bypassed phased DMA");
    std::cout<<"GGUF partial GU/Down fences long K/large Down/multiple stages/tail byte parity PASS\n";
}
void run_host_kv_sharing() {
    ScopedSnapshot snapshot("0");Source source;
    for(const auto* name:{"gate","up","down"})source.add(name,{7,256,256},GGML_TYPE_Q4_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4ULL<<20;
    execution.config.moe_device_cache_bytes=3*source.find_tensor("gate")->nbytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto metric=[&](const char* name) {
        for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))if(key==name)return std::size_t(value);
        throw std::runtime_error("Missing RAM sharing metric");
    };
    auto x=input(1,1,false).to(tb::kFloat16),ids=routes(1,0);
    auto weights=tb::full({1,3},.3,x.options().dtype(tb::kFloat32));
    const auto expected=banks.forward(x,ids,weights).cpu();
    const auto unique=metric("ram_expert_unique_payload_bytes");
    reclaim_gguf_expert_duplicates(execution.gguf_expert_cache,SIZE_MAX/2);
    require(metric("ram_expert_unique_payload_bytes")==unique,"Duplicate-only reclamation discarded a unique RAM expert");
    const auto cap=metric("ram_expert_cache_budget_bytes");
    share_gguf_expert_host(execution.gguf_expert_cache,SIZE_MAX/2);
    require(metric("ram_expert_cache_allocated_bytes")==0 && metric("ram_expert_cache_budget_bytes")==0,
        "KV reservation left pinned expert arenas allocated or allowed immediate cache regrowth");
    const auto actual=banks.forward(x,ids,weights).cpu();
    require(actual.nbytes()==expected.nbytes() && std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
        "Expert RAM reclamation changed resident or streamed results");
    share_gguf_expert_host(execution.gguf_expert_cache,0);
    require(metric("ram_expert_cache_budget_bytes")==cap,"Returned KV loan did not restore the expert RAM ceiling");
    std::cout<<"GGUF duplicate-only protection, physical arena reclamation, KV soft ceiling, reuse and loan return PASS\n";
}
void run_pinned_hole_under_pressure() {
    ScopedSnapshot snapshot("0");
    ScopedSnapshot direct("0","TYLOINFER_GGUF_DIRECT_EXPERTS");
    ScopedSnapshot slots("0","TYLOINFER_GGUF_STRATA_CACHE");
    Source source;source.add("gate",{8,256,256},GGML_TYPE_Q4_0);
    (void)tb::empty({1},tb::TensorOptions().device(tb::kCUDA)).cpu();
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=4ULL<<20;
    execution.config.moe_host_physical_bytes=mfq::sample_host_memory().resident_bytes+(128ULL<<20);
    const auto expert_bytes=source.find_tensor("gate")->nbytes/8;
    execution.config.moe_device_cache_bytes=expert_bytes;
    auto bank=std::make_shared<GgufRoutedWeight>(execution,source,"gate");
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* name) {
        for(const auto& [key,value]:gguf_expert_memory_metrics(cache))if(key==name)return std::size_t(value);
        throw std::runtime_error("Missing pinned-hole cache metric");
    };
    auto x=input(1,1,false).to(tb::kFloat16);
    auto ids=tb::zeros({1,1},x.options().dtype(tb::kInt64));
    const auto expected=bank->forward(x,ids).cpu();
    require(metric("ram_expert_pinned_payload_bytes")==8*expert_bytes &&
        metric("ram_expert_duplicate_payload_bytes")==expert_bytes,
        "Pinned-hole fixture did not retain eight experts and one GPU backup");
    reclaim_gguf_expert_duplicates(cache,1);
    const auto unique=metric("ram_expert_unique_payload_bytes");
    const auto allocated=metric("ram_expert_cache_allocated_bytes");
    const auto reads=source.read_calls();
    require(unique==7*expert_bytes && metric("ram_expert_duplicate_payload_bytes")==0,
        "Pinned-hole fixture failed to leave a reusable interval beside unique RAM experts");
    {
        // Exceed only this cache's process-relative soft budget. The real
        // machine needs just 128MiB; no dependence on its capacity or model.
        mfq::cuda::HostBuffer pressure(128ULL<<20,true);
        std::memset(pressure.data(),0,128ULL<<20);
        require(mfq::sample_host_memory().resident_bytes>=execution.config.moe_host_physical_bytes,
            "Pinned-hole fixture did not exhaust its simulated physical allowance");
        resize_gguf_expert_cache(cache,0);
        require(metric("ram_expert_unique_payload_bytes")==unique+expert_bytes &&
            metric("ram_expert_cache_allocated_bytes")==allocated,
            "Reusable pinned interval discarded unique RAM experts under physical pressure");
    }
    const auto actual=bank->forward(x,ids).cpu();
    require(std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.nbytes())==0 &&
        source.read_calls()==reads,
        "Pinned interval reuse changed the result or reread an already resident expert");
    std::cout<<"GGUF physical pressure reuses pinned holes without unique eviction or SSD reread PASS\n";
}
void run_pinned_interval_compaction() {
    auto arena=std::make_shared<MoePinnedExpertArena>(1024);
    auto gap=arena->allocate(128),first=arena->allocate(256),middle=arena->allocate(128);
    auto second=arena->allocate(256),guard=arena->allocate(256);
    for(std::size_t i=0;i<256;++i){first->data()[i]=std::uint8_t(i);second->data()[i]=std::uint8_t(i^0xa5);}
    auto* fixed=guard->data();gap.reset();middle.reset();
    require(arena->largest_free_bytes()==128,"Separated holes were counted as one interval");
    require(first->compact_left()==256 && arena->used_bytes()==768 && arena->largest_free_bytes()==256,
        "In-place overlapping move lost pinned capacity accounting");
    require(second->compact_left()==256 && guard->data()==fixed,
        "In-place packing moved a protected neighbor");
    for(std::size_t i=0;i<256;++i)
        require(first->data()[i]==std::uint8_t(i) && second->data()[i]==std::uint8_t(i^0xa5),
            "In-place overlapping move changed expert bytes");
    auto joined=arena->allocate(256);
    require(joined && joined->offset()==512 && !arena->allocate(16),
        "Compacted holes did not provide the expected bounded allocation");
    first.reset();second.reset();joined.reset();guard.reset();
    require(arena->used_bytes()==0 && arena->largest_free_bytes()==1024,
        "Packed expert leases did not return their complete arena");
    std::cout<<"GGUF in-place pinned interval packing: overlapping byte parity/fixed neighbor/bounded reuse/coalescing PASS\n";
}
void run_prefill_host_scan_bypass(bool pressure=false) {
    ScopedSnapshot snapshot("0");
    ScopedSnapshot slots("0","TYLOINFER_GGUF_STRATA_CACHE");
    Source source;source.enable_batches();
    for(const auto* layer:{"warm","scan"})for(const auto* role:{"gate","up","down"})
        source.add(std::string(layer)+role,{pressure && std::string_view(layer)=="scan"?512:32,256,256},GGML_TYPE_Q4_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=pressure?16ULL<<20:3*source.find_tensor("warmgate")->nbytes;
    if(pressure)execution.config.moe_host_physical_bytes=mfq::sample_host_memory().resident_bytes+(512ULL<<20);
    execution.config.moe_device_cache_bytes=0;
    const auto make=[&](const char* layer) {
        return GgufMoeBanks{std::make_shared<GgufRoutedWeight>(execution,source,std::string(layer)+"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,std::string(layer)+"up"),
            std::make_shared<GgufRoutedWeight>(execution,source,std::string(layer)+"down")};
    };
    auto warm=make("warm"),scan=make("scan");
    const auto metric=[&](const char* name) {
        for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
            if(key==name)return static_cast<std::uint64_t>(value);
        throw std::runtime_error("Missing PP host bypass metric");
    };
    auto x=tb::full({64,256},.001,tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16));
    auto ids=tb::arange(64,x.options().dtype(tb::kInt64)).remainder(32).reshape({64,1});
    auto weights=tb::full({64,1},1.,x.options().dtype(tb::kFloat32));
    const auto baseline=warm.forward(x,ids,weights).to(tb::kFloat32).cpu();
    std::vector<tb::Tensor> physical_pressure;
    if(pressure)while(mfq::sample_host_memory().resident_bytes<execution.config.moe_host_physical_bytes)
        physical_pressure.push_back(tb::full({8<<20},19.,x.options().device(tb::kCPU).dtype(tb::kUInt8)));
    const auto payload=metric("ram_expert_payload_bytes");
    const auto evictions=metric("ram_expert_cache_evictions");
    (void)scan.forward(x,ids,weights).cpu();
    require(metric("gguf_prefill_host_bypass_bytes")==3*source.find_tensor("warmgate")->nbytes,
        ("Bounded PP misses did not use transient source storage: bypass="+
        std::to_string(metric("gguf_prefill_host_bypass_bytes"))+" expected="+
        std::to_string(3*source.find_tensor("warmgate")->nbytes)+" payload="+
        std::to_string(metric("ram_expert_payload_bytes"))+" pinned="+
        std::to_string(metric("ram_expert_pinned_reserved_bytes"))+" resident="+
        std::to_string(mfq::sample_host_memory().resident_bytes)+" physical_limit="+
        std::to_string(execution.config.moe_host_physical_bytes)).c_str());
    require(metric("ram_expert_cache_evictions")==evictions && metric("ram_expert_payload_bytes")==payload,
        "PP scan changed the persistent RAM working set");
    const auto reads=source.read_calls();
    const auto actual=warm.forward(x,ids,weights).to(tb::kFloat32).cpu();
    require(source.read_calls()==reads && actual.nbytes()==baseline.nbytes() &&
        std::memcmp(actual.data_ptr(),baseline.data_ptr(),actual.nbytes())==0,
        "PP scan changed cached expert bytes or reread retained RAM weights");
    std::cout<<"GGUF bounded PP host scan bypass preserves resident RAM and byte parity PASS pressure="<<pressure<<'\n';
}
void run_fragmented_pinned_fallback(bool batch=false) {
    ScopedSnapshot snapshot("0");
    ScopedSnapshot direct("0","TYLOINFER_GGUF_DIRECT_EXPERTS");
    ScopedSnapshot slots("0","TYLOINFER_GGUF_STRATA_CACHE");
    Source source;
    if(batch)source.enable_batches();
    for(const auto* name:{"small-a","small-b","small-c"})source.add(name,{1024,256,256},GGML_TYPE_Q4_0);
    source.add("large",{batch?16:1,8192,batch?256:4096},GGML_TYPE_Q4_0);
    if(batch) {
        source.add("large-up",{16,8192,256},GGML_TYPE_Q4_0);
        source.add("large-down",{16,256,8192},GGML_TYPE_Q4_0);
    }
    (void)tb::empty({1},tb::TensorOptions().device(tb::kCUDA)).cpu();
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=96ULL<<20;
    execution.config.moe_host_physical_bytes=mfq::sample_host_memory().resident_bytes+(384ULL<<20);
    const auto small_bytes=source.find_tensor("small-a")->nbytes/1024;
    execution.config.moe_device_cache_bytes=2048*small_bytes;
    std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
    banks[0]=std::make_shared<GgufRoutedWeight>(execution,source,"small-a");
    banks[1]=std::make_shared<GgufRoutedWeight>(execution,source,"small-b");
    banks[2]=std::make_shared<GgufRoutedWeight>(execution,source,"small-c");
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* name) {
        for(const auto& [key,value]:gguf_expert_memory_metrics(cache))if(key==name)return std::size_t(value);
        throw std::runtime_error("Missing fragmented allocation metric");
    };
    auto x=(input(512,1,false)*.001).to(tb::kFloat16);
    auto even=(tb::arange(512,x.options().dtype(tb::kInt64))*2).reshape({512,1}),odd=even+1;
    std::array<std::vector<tb::Tensor>,3> expected;
    for(unsigned i=0;i<3;++i)for(int begin=0;begin<512;begin+=32)
        expected[i].push_back(banks[i]->forward(x.narrow(0,begin,32),odd.narrow(0,begin,32)).cpu());
    for(const auto& bank:banks)for(int begin=0;begin<512;begin+=32)
        (void)bank->forward(x.narrow(0,begin,32),even.narrow(0,begin,32)).cpu();
    require(metric("ram_expert_pageable_payload_bytes")==0 && metric("ram_expert_duplicate_payload_bytes")>0,
        ("Fragmented fallback fixture did not create locked GPU backups: pageable="+
        std::to_string(metric("ram_expert_pageable_payload_bytes"))+" duplicates="+
        std::to_string(metric("ram_expert_duplicate_payload_bytes"))).c_str());
    auto large=std::make_shared<GgufRoutedWeight>(execution,source,"large");
    const auto before=metric("ram_expert_pinned_interval_compactions");
    auto large_x=tb::full({batch?16:1,1,batch?256:4096},.001,x.options());
    auto large_id=tb::arange(batch?16:1,x.options().dtype(tb::kInt64)).reshape({batch?16:1,1});
    if(batch) {
        GgufMoeBanks large_banks{large,std::make_shared<GgufRoutedWeight>(execution,source,"large-up"),
            std::make_shared<GgufRoutedWeight>(execution,source,"large-down")};
        large_x=tb::full({64,256},.001,x.options());
        large_id=tb::arange(64,x.options().dtype(tb::kInt64)).remainder(16).reshape({64,1});
        const auto coefficients=tb::full({64,1},1.,x.options().dtype(tb::kFloat32));
        const auto result=large_banks.forward(large_x,large_id,coefficients);
        const auto actual=result.to(tb::kFloat32).cpu();
        std::cout<<"Fragmented batch interval passes="<<metric("ram_expert_pinned_interval_compactions")-before
            <<" moved_bytes="<<metric("ram_expert_compacted_bytes")<<'\n';
        require(metric("ram_expert_pinned_interval_compactions")-before<=6,
            ("Fragmented read batch repeatedly compacted one expert at a time: count="+
            std::to_string(metric("ram_expert_pinned_interval_compactions")-before)).c_str());
        CudaExecutionContext reference;
        reference.config.moe_host_cache_bytes=256ULL<<20;reference.config.moe_device_cache_bytes=0;
        GgufMoeBanks reference_banks{std::make_shared<GgufRoutedWeight>(reference,source,"large"),
            std::make_shared<GgufRoutedWeight>(reference,source,"large-up"),
            std::make_shared<GgufRoutedWeight>(reference,source,"large-down")};
        const auto expected=reference_banks.forward(large_x,large_id,coefficients).to(tb::kFloat32).cpu();
        require(actual.sizes()==expected.sizes() && std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
            "Fragmented batched MoE bytes differ from an unfragmented cache");
    }else (void)large->forward(large_x,large_id).cpu();
    require(batch || metric("ram_expert_pinned_interval_compactions")>0,
        "Fragmented fallback did not exercise in-place interval packing");
    const auto transient=batch?3*source.find_tensor("large")->nbytes:0;
    require((batch || metric("ram_expert_pageable_payload_bytes")==0) &&
        metric("ram_expert_payload_bytes")<=execution.config.moe_host_cache_bytes &&
        metric("ram_expert_cache_allocated_bytes")<=execution.config.moe_host_cache_bytes+transient,
        ("Fragmented locked capacity caused an avoidable pageable allocation: compacted="+
        std::to_string(metric("ram_expert_compacted_bytes"))+" pageable="+
        std::to_string(metric("ram_expert_pageable_payload_bytes"))+" allocated="+
        std::to_string(metric("ram_expert_cache_allocated_bytes"))).c_str());
    const auto reads=source.read_calls();
    for(unsigned i=0;i<3;++i)for(int begin=0;begin<512;begin+=32) {
        const auto actual=banks[i]->forward(x.narrow(0,begin,32),odd.narrow(0,begin,32)).cpu();
        const auto& reference=expected[i][begin/32];
        require(actual.nbytes()==reference.nbytes() && !std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes()),
            "Fragmented allocation changed retained expert bytes");
    }
    if(!batch)require(source.read_calls()==reads,"Fragmented allocation discarded unique RAM experts");
    std::cout<<"GGUF fragmented locked capacity reused before pageable allocation, unique expert bytes and physical ceiling preserved PASS\n";
}
void run_host_vram_evacuation() {
    ScopedSnapshot snapshot("0");
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto previous_limit=context->memory_stats().limit;
    std::vector<tb::Tensor> baseline;
    try {
        for(int mode=0;mode<4;++mode) {
            Source source;
            for(const auto* name:{"gate","up","down"})source.add(name,{7,256,256},GGML_TYPE_Q4_0);
            const auto payload=3*source.find_tensor("gate")->nbytes;
            CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4ULL<<20;
            execution.config.moe_device_cache_bytes=mode==2?1:2*payload;
            GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
                std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
            const auto cache=execution.gguf_expert_cache;
            const auto metric=[&](const char* name) {
                for(const auto& [key,value]:gguf_expert_memory_metrics(cache))if(key==name)return std::size_t(value);
                throw std::runtime_error("Missing RAM evacuation metric");
            };
            require(metric("resident_expert_cache_bytes")==0 && metric("ram_expert_unique_payload_bytes")==payload,
                "Evacuation fixture did not start with unique RAM experts");
            if(mode!=0)finalize_gguf_expert_cache(cache);
            if(mode==3) {
                restore_gguf_decode_workspace(cache,0,16ULL<<20);
                const auto reserve=metric("gguf_decode_workspace_reserve_bytes");
                context->set_memory_limit(context->resident_memory_usage()+reserve/2);
                require(context->available_memory()<reserve,"Evacuation fixture did not consume its workspace allowance");
            }
            const auto reads=source.read_calls();
            share_gguf_expert_host(cache,SIZE_MAX/2);
            require(metric("ram_expert_cache_allocated_bytes")==0 && metric("ram_expert_cache_budget_bytes")==0,
                "Evacuation failed to return physical RAM for KV");
            require(metric("ram_expert_evacuated_to_vram_bytes")==std::size_t(mode==1?payload:0),
                "RAM evacuation ignored load completion, GPU budget or workspace headroom");
            require(metric("resident_expert_cache_bytes")<=execution.config.moe_device_cache_bytes && source.read_calls()==reads,
                "RAM evacuation overfilled VRAM or fetched uncached weights from SSD");
            context->set_memory_limit(previous_limit);
            for(int tokens:{1,4}) {
                auto x=input(tokens,1,false).to(tb::kFloat16);
                auto ids=tb::arange(tokens*3,x.options().dtype(tb::kInt64)).remainder(7).reshape({tokens,3});
                auto weights=tb::full({tokens,3},.3,x.options().dtype(tb::kFloat32));
                const auto before=source.read_calls();const auto actual=banks.forward(x,ids,weights);
                if(mode==1)require(source.read_calls()==before,"A migrated expert was fetched again from SSD");
                const auto cpu=actual.to(tb::kFloat32).cpu();
                if(mode==0)baseline.push_back(cpu);
                else {
                    // The first case used ordinary source reads, with migration
                    // disabled. Its independent bytes catch expert/key mixups.
                    const auto& expected=baseline.at(tokens==1?0:1);
                    float error=0,magnitude=0;
                    for(int64_t i=0;i<cpu.numel();++i) {
                        error=std::max(error,std::abs(cpu.data_ptr<float>()[i]-expected.data_ptr<float>()[i]));
                        magnitude=std::max(magnitude,std::abs(expected.data_ptr<float>()[i]));
                    }
                    require(std::isfinite(error) && error<1e-5f+magnitude*1e-4f,
                        "Migrated expert output differs from the ordinary source-read baseline");
                }
                check_moe(actual,banks,x,ids,weights);
            }
            if(mode==1) {
                auto x=input(1,0,false).to(tb::kFloat16),ids=routes(1,0);
                auto weights=tb::full({1,3},.3,x.options().dtype(tb::kFloat32));tb::Tensor result;
                require(prepare_gguf_graph_pool(cache),"Evacuation fixture has no graph pool");
                mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                graph.capture([&]{result=banks.forward(x,ids,weights);},[]{});
                graph.run();check_moe(result,banks,x,ids,weights);
            }
        }
    } catch(...) {context->set_memory_limit(previous_limit);throw;}
    std::cout<<"GGUF RAM evacuation: unused GPU capacity, no SSD rereads, load/budget/workspace guards and Graph PASS\n";
}
void run_pinned_range_trim() {
    MoePinnedExpertArena arena(256);auto block=arena.allocate(192);
    for(int i=0;i<192;++i)block->data()[i]=std::uint8_t(i);
    const auto kept=block->data()+64;
    require(block->retain_range(64,64)==128 && block->bytes()==64 && block->data()==kept && arena.used_bytes()==64,
        "Pinned range trim changed live address or used-byte accounting");
    for(int i=0;i<64;++i)require(block->data()[i]==i+64,"Pinned range trim changed live bytes");
    auto head=arena.allocate(64),tail=arena.allocate(128);
    require(head && tail && arena.used_bytes()==256,"Pinned range trim did not return both adjacent holes");
    bool rejected=false;try{block->retain_range(1,16);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected && block->data()==kept && arena.used_bytes()==256,"Invalid pinned trim mutated its lease");
    block.reset();head.reset();tail.reset();
    require(arena.used_bytes()==0 && bool(arena.allocate(256)),"Trimmed pinned intervals did not coalesce after release");
    auto padded=arena.allocate(256);const auto start=padded->data();
    require(padded->retain_range(16,17)==224 && padded->data()==start+16 && padded->bytes()==32 && arena.used_bytes()==32,
        "Pinned subrange lost its alignment padding or prefix interval");
    require(padded->retain_range(0,17)==0,"Repeated pinned subrange changed an already minimal lease");
}
void run_fragmented_host_reclaim(bool paired=false) {
    ScopedSnapshot snapshot("0");
    ScopedSnapshot direct("0","TYLOINFER_GGUF_DIRECT_EXPERTS");
    Source source;
    const std::array<std::string,3> names=paired?std::array<std::string,3>{"layers.0.mlp.experts.gate.weight",
        "layers.0.mlp.experts.up.weight","layers.0.mlp.experts.down.weight"}:std::array<std::string,3>{"gate","up","down"};
    for(const auto& name:names)source.add(name,{1024,256,256},GGML_TYPE_Q4_0);
    // Force several bounded arenas using a process-relative budget; this
    // exercises fragmentation independently of the test machine's RAM size.
    (void)tb::empty({1},tb::TensorOptions().device(tb::kCUDA)).cpu();
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=192ULL<<20;
    execution.config.moe_host_physical_bytes=mfq::sample_host_memory().resident_bytes+(256ULL<<20);
    const auto expert_bytes=source.find_tensor(names[0])->nbytes/1024;
    execution.config.moe_device_cache_bytes=3*512*expert_bytes;
    std::array<std::shared_ptr<GgufRoutedWeight>,3> banks;
    int role=0;for(const auto& name:names)
        banks[role++]=std::make_shared<GgufRoutedWeight>(execution,source,name);
    const auto cache=execution.gguf_expert_cache;
    const auto metric=[&](const char* name) {
        for(const auto& [key,value]:gguf_expert_memory_metrics(cache))if(key==name)return std::size_t(value);
        throw std::runtime_error("Missing fragmented RAM cache metric");
    };
    auto x=(input(512,1,false)*.001).to(tb::kFloat16);
    auto even=(tb::arange(512,x.options().dtype(tb::kInt64))*2).reshape({512,1});
    auto odd=even+1;
    std::array<std::vector<tb::Tensor>,3> expected;
    for(int i=0;i<3;++i)for(int start=0;start<512;start+=32)
        expected[i].push_back(banks[i]->forward(x.narrow(0,start,32),odd.narrow(0,start,32)).cpu());
    require(prepare_gguf_graph_pool(cache),"Fragmented expert test has no graph pool");
    tb::Tensor graph_output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{graph_output=banks[0]->forward(x.narrow(0,0,32),odd.narrow(0,0,32));},[]{},[&]{graph_output={};});
    for(int i=0;i<(paired?1:3);++i)for(int start=0;start<512;start+=32)
        (void)banks[i]->forward(x.narrow(0,start,32),even.narrow(0,start,32)).cpu();
    const auto unique=metric("ram_expert_unique_payload_bytes");
    const auto allocated=metric("ram_expert_cache_allocated_bytes");
    require(metric("ram_expert_duplicate_payload_bytes")>0,"Fragmented fixture has no GPU duplicates");
    const auto released=reclaim_gguf_expert_duplicates(cache,1);
    std::cout<<"Fragmented RAM reclaim: allocated="<<allocated<<" released="<<released
        <<" unique_before="<<unique<<" unique_after="<<metric("ram_expert_unique_payload_bytes")
        <<" copied="<<metric("ram_expert_compacted_bytes")<<'\n';
    require(released>0 && metric("ram_expert_cache_allocated_bytes")<allocated,
        "Interleaved GPU duplicates did not release physical pinned arenas");
    require(metric("ram_expert_duplicate_payload_bytes")>0,
        "A small RAM reservation discarded every unrelated GPU backup");
    require(metric("ram_expert_compacted_bytes")>0 && metric("ram_expert_unique_payload_bytes")==unique,
        "Pinned arena compaction did not preserve unique RAM experts");
    if(paired)require(metric("ram_expert_trimmed_pinned_bytes")>0,
        "Paired survivor did not return its unused compressed-image half");
    const auto reads=source.read_calls();
    for(int i=0;i<3;++i)for(int start=0;start<512;start+=32) {
        const auto actual=banks[i]->forward(x.narrow(0,start,32),odd.narrow(0,start,32)).cpu();
        const auto& reference=expected[i][start/32];
        require(actual.nbytes()==reference.nbytes() && !std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes()),
            "Pinned arena compaction changed expert bytes or output");
    }
    require(source.read_calls()==reads,"Compacted unique experts were fetched from their source again");
    graph.run();const auto graph_result=graph_output.cpu();
    require(graph_result.nbytes()==expected[0][0].nbytes() &&
        !std::memcmp(graph_result.data_ptr(),expected[0][0].data_ptr(),graph_result.nbytes()),
        "Retained expert Graph used stale RAM pointers after compaction");
    share_gguf_expert_host(cache,SIZE_MAX/2);
    require(metric("ram_expert_cache_allocated_bytes")==0,
        "Full RAM reservation left relocated expert blocks behind");
    std::cout<<"GGUF fragmented pinned arenas: duplicate-only physical release/unique byte parity/no SSD rereads PASS\n";
}
void run_host_snapshot(bool pageable=false,bool mixed=false) {
    ScopedSnapshot setting("1");
    ScopedSnapshot registration(pageable?"none":nullptr,"STRATA_ARENA_PIN_GIB");
    Source source;
    const std::array<std::string,3> names{"layers.0.mlp.experts.gate.weight",
        "layers.0.mlp.experts.up.weight","layers.0.mlp.experts.down.weight"};
    for(int role=0;role<3;++role)source.add(names[role],{7,256,256},
        mixed?(role==2?GGML_TYPE_IQ4_NL:GGML_TYPE_IQ3_XXS):GGML_TYPE_Q8_0);
    source.add("unrelated.projection",{2,256,256}); // shape never implies routed role
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4ULL<<20;
    execution.config.moe_device_cache_bytes=3*256*mfq::gguf::row_bytes(GGML_TYPE_Q8_0,256);
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,names[0]),
        std::make_shared<GgufRoutedWeight>(execution,source,names[1]),
        std::make_shared<GgufRoutedWeight>(execution,source,names[2])};
    const auto metrics=[&](const char* key) {
        for(const auto& [name,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))if(name==key)return value;
        throw std::runtime_error("GGUF snapshot metric missing");
    };
    require(metrics("gguf_host_snapshot_bytes")>0,"GGUF full host arena was not enabled");
    require(source.read_calls()==3,"GGUF snapshot reread individual experts or loaded a non-routed tensor");
    auto x=input(1,1,false).to(tb::kFloat16);
    auto weights=tb::full({1,3},.3,x.options().dtype(tb::kFloat32));
    auto ids=routes(1,0);
    const auto expected=banks.forward(x,ids,weights).cpu();
    resize_gguf_expert_cache(execution.gguf_expert_cache,0);
    const auto actual=banks.forward(x,ids,weights).cpu();
    require(actual.nbytes()==expected.nbytes() && std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
        "GGUF full host arena changed expert output after GPU eviction");
    require(source.read_calls()==3 && metrics("expert_writeback_bytes")==0,
        "GGUF immutable host arena caused expert rereads or GPU writeback");
    {
        if(pageable)require(metrics("ram_expert_pinned_reserved_bytes")==0,"GGUF fixture unexpectedly pinned the original arena");
        // 3 roles x 7 experts exceeds the 16-entry ring. Repeated layers check
        // wraparound and generation ownership while DMA remains asynchronous.
        for(int rotation:{0,2}) {
            auto prompt=(input(64,1,false)*.001).to(tb::kFloat16);
            auto prompt_ids=((tb::arange(64*3,prompt.options().dtype(tb::kInt64))+rotation).remainder(7)).reshape({64,3});
            auto prompt_weights=tb::full({64,3},.3,prompt.options().dtype(tb::kFloat32));
            ScopedSnapshot stage("0","TYLOINFER_GGUF_PREFILL_HOST_STAGE");
            ScopedSnapshot direct("0","TYLOINFER_GGUF_PREFILL_DIRECT");
            const auto reference=banks.forward(prompt,prompt_ids,prompt_weights,.01f).cpu();
            stage.set("1");
            direct.set("1");
            const auto copies=metrics("gguf_prefill_dma_copies"),images=metrics("gguf_prefill_dma_images");
            const auto staged=banks.forward(prompt,prompt_ids,prompt_weights,.01f).cpu();
            require(std::memcmp(reference.data_ptr(),staged.data_ptr(),staged.nbytes())==0,
                "GGUF host staging changed the original compressed experts or token routes");
            if(!pageable)require(metrics("gguf_prefill_dma_images")-images==21 &&
                metrics("gguf_prefill_dma_copies")-copies<=14,
                "GGUF expert-major PP failed to merge Gate/Up uploads");
        }
        if(pageable) {
            require(metrics("gguf_prefill_host_staged_bytes")>0,"GGUF pageable PP did not use parallel host staging");
            require(metrics("gguf_prefill_phased_groups")==0,"pageable PP used a pinned-source partial fence");
        }
    }
    const auto original=metrics("gguf_host_snapshot_bytes");
    const auto released=reserve_gguf_expert_host(execution.gguf_expert_cache,SIZE_MAX/2);
    require(released>=original && metrics("gguf_host_snapshot_bytes")==0,
        "GGUF host arena did not release RAM needed for KV workspace");
    check_moe(banks.forward(x,ids,weights),banks,x,ids,weights);
    require(source.read_calls()>3,"GGUF bounded host cache did not recover after arena release");
    CudaExecutionContext limited;limited.config.moe_host_cache_bytes=1;
    GgufRoutedWeight fallback(limited,source,names[0]);
    for(const auto& [key,value]:gguf_expert_memory_metrics(limited.gguf_expert_cache))
        if(key=="gguf_host_snapshot_bytes")require(value==0,"GGUF host arena ignored explicit RAM budget");
}
void run_prepared_dense_projection() {
    for(auto format:{GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS,GGML_TYPE_Q8_0,GGML_TYPE_Q6_K})for(int columns:{256,3072}) {
        Source source;source.add("w",{17,columns},format);auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
        for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int tokens:{1,2,3,4,8,16,32}) {
            auto host=tb::empty({tokens,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(i*.17f)*.3f;
            auto x=host.to(tb::kCUDA).to(dtype);
            auto packet=tb::empty({int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
            if(dtype==tb::kFloat16)mfq::cuda::gguf::quantize_input_f16(x.data_ptr(),tokens,columns,packet.data_ptr(),mfq_current_cuda_stream());
            else mfq::cuda::gguf::quantize_input(x.data_ptr<float>(),tokens,columns,packet.data_ptr(),mfq_current_cuda_stream());
            const auto actual=weight.forward_q8(packet,tokens,dtype).cpu(),expected=weight.forward(x).cpu();
            require(actual.scalar_type()==expected.scalar_type() && actual.sizes()==expected.sizes() &&
                std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,"prepared GGUF projection changed shape/type/output bits");
            if(format==GGML_TYPE_Q6_K && tokens>1) {
                // A prepared/full comparison can share the same faulty dot.
                // Independently check every multi-row result against M=1.
                const auto batch=actual.to(tb::kFloat32).contiguous();
                double error=0.0,norm=0.0;
                for(int token=0;token<tokens;++token) {
                    const auto scalar=weight.forward(x.narrow(0,token,1)).to(tb::kCPU,tb::kFloat32).contiguous();
                    for(int row=0;row<weight.out;++row) {
                        const double value=scalar.data_ptr<float>()[row];
                        const double difference=batch.data_ptr<float>()[token*weight.out+row]-value;
                        error+=difference*difference;norm+=value*value;
                    }
                }
                const double tolerance=dtype==tb::kFloat16?1e-3:1e-5;
                require(std::isfinite(error) && error<=tolerance*tolerance*std::max(norm,1e-30),
                    "Q6_K multi-row projection differs from singleton dot products");
            }
        }
    }
    std::cout<<"GGUF prepared Q8 activation -> FP16/FP32 projection exact checks PASS\n";
}
void run_interleaved_dense_projection() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    int cases=0;
    const auto bits=[](const tb::Tensor& a,const tb::Tensor& b,const char* message) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        require(x.scalar_type()==y.scalar_type() && x.sizes()==y.sizes() &&
            std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())==0,message);
    };
    // The same IQ4_XS row must survive a change of verification width or
    // activation packet layout without changing the next FP16 KV entry.
    for(int columns:{256,768,4096,6144})for(int rows:{1,17,129})
    for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        Source source;source.add("w",{rows,columns},GGML_TYPE_IQ4_XS);
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
        auto input=tb::empty({8,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int64_t i=0;i<input.numel();++i)input.data_ptr<float>()[i]=std::sin(i*.173f)*.37f;
        input=input.to(tb::kCUDA).to(dtype);
        const auto bytes=int64_t(gg::q8_row_bytes(columns));
        auto native=tb::empty({8*bytes},input.options().dtype(tb::kUInt8));
        if(dtype==tb::kFloat16)gg::quantize_input_f16(input.data_ptr(),8,columns,native.data_ptr(),stream);
        else gg::quantize_input(input.data_ptr<float>(),8,columns,native.data_ptr(),stream);
        for(auto output_type:{tb::kFloat16,tb::kFloat32}) {
            std::vector<tb::Tensor> serial;
            for(int t=0;t<8;++t)serial.push_back(weight.forward_q8(native.narrow(0,t*bytes,bytes),1,output_type));
            const auto reference=tb::cat(serial,0);
            for(int tokens=2;tokens<=8;++tokens) {
                bits(weight.forward_q8(native.narrow(0,0,tokens*bytes),tokens,output_type),
                    reference.narrow(0,0,tokens),"IQ4_XS native M1/window reduction differs");
                if(tokens<=4) {
                    auto packed=tb::empty({int64_t(gg::interleaved_activation_bytes(tokens,columns))},native.options());
                    gg::quantize_input_interleaved(input.data_ptr(),dtype==tb::kFloat16,tokens,columns,
                        packed.data_ptr(),nullptr,stream);
                    bits(weight.forward_q8(packed,tokens,output_type,true),reference.narrow(0,0,tokens),
                        "IQ4_XS interleaved M1/window reduction differs");
                }
                ++cases;
            }
        }
    }
    for(int columns:{256,768,4096,6144})for(int tokens:{2,3,4})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        const int capacity=tokens==2?2:4,blocks=columns/32;
        auto host=tb::empty({tokens,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(i*.17f)*.3f;
        auto input=host.to(tb::kCUDA).to(dtype);
        auto native=tb::empty({int64_t(tokens*gg::q8_row_bytes(columns))},input.options().dtype(tb::kUInt8));
        auto duplicate=tb::empty_like(native);
        auto packed=tb::empty({int64_t(gg::interleaved_activation_bytes(tokens,columns))},native.options());
        if(dtype==tb::kFloat16)gg::quantize_input_f16(input.data_ptr(),tokens,columns,native.data_ptr(),stream);
        else gg::quantize_input(input.data_ptr<float>(),tokens,columns,native.data_ptr(),stream);
        gg::quantize_input_interleaved(input.data_ptr(),dtype==tb::kFloat16,tokens,columns,packed.data_ptr(),duplicate.data_ptr(),stream);
        bits(duplicate,native,"dual-layout quantization changed original Q8 bytes");
        const auto source=native.cpu(),actual_packet=packed.cpu();
        std::vector<uint8_t> expected(packed.nbytes(),0);
        for(int t=0;t<tokens;++t)for(int b=0;b<blocks;++b) {
            const auto* q=source.data_ptr<uint8_t>()+(t*blocks+b)*36;
            for(int word=0;word<8;++word)for(int byte=0;byte<4;++byte)
                expected[((word*blocks+b)*capacity+t)*4+byte]=q[4+word*4+byte];
            ggml_fp16_t half_scale;std::memcpy(&half_scale,q,sizeof(half_scale));
            const float scale=ggml_fp16_to_fp32(half_scale);
            std::memcpy(expected.data()+columns*capacity+(b*capacity+t)*sizeof(float),&scale,sizeof(float));
        }
        require(std::memcmp(expected.data(),actual_packet.data_ptr(),expected.size())==0,
            "position-major activation changed codes/scales/padding");
        for(auto format:{GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS})for(int rows:{1,17,129}) {
            Source weights;weights.add("w",{rows,columns},format);
            auto weight=load_gguf_weight(weights,"w",mfq_current_cuda_device());
            for(auto out_type:{tb::kFloat16,tb::kFloat32}) {
                const auto reference=weight.forward_q8(native,tokens,out_type);
                bits(weight.forward_q8(packed,tokens,out_type,true),reference,"position-major dense projection changed output bits");
                if(dtype==out_type)bits(weight.forward(input),reference,"dense caller failed to preserve original Q8 result");
                ++cases;
            }
        }
    }
    std::cout<<"GGUF interleaved/dual-layout Q8 bytes and dense output bitwise cases="<<cases<<" PASS\n";
}
void run_projection_group(bool mixed=true) {
    Source source;
    source.add("q",{7,256},mixed?GGML_TYPE_Q4_0:GGML_TYPE_Q8_0);
    source.add("k",{9,256},mixed?GGML_TYPE_Q5_0:GGML_TYPE_Q8_0);
    source.add("v",{11,256},GGML_TYPE_Q8_0);
    source.add("gate",{13,256},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;
    std::vector<weight_loader::Linear> projections{
        weight_loader::linear(execution,source,"q"),weight_loader::linear(execution,source,"k"),
        weight_loader::linear(execution,source,"v"),weight_loader::linear(execution,source,"gate"),
        [](CudaExecutionContext&,const tb::Tensor& x){return x.sum(-1,true);}};
    for(bool combine:{true,false})for(std::size_t joined:{1,3}) {
        std::vector<const void*> pointers;
        for(std::size_t i=0;i<4;++i)pointers.push_back(weight_loader::linear_weight(projections[i])->gguf.packed.data_ptr());
        auto group=weight_loader::grouped_linear(execution,projections,joined,combine);
        require(bool(group),"GGUF projections did not advertise whole-window execution");
        if(!combine)for(std::size_t i=0;i<4;++i)
            require(pointers[i]==weight_loader::linear_weight(projections[i])->gguf.packed.data_ptr(),
                "GGUF shared-input subset copied an existing weight image");
        for(int rows:{1,2,3,4,8,33}) {
            auto x=input(rows,1,false).to(tb::kFloat16).reshape({1,rows,256});
            auto actual=group(execution,x);
            std::vector<tb::Tensor> expected;
            for(const auto& project:projections)expected.push_back(project(execution,x));
            if(joined>1)expected={tb::cat({expected[0],expected[1],expected[2]},-1),expected[3],expected[4]};
            require(actual.size()==expected.size(),"GGUF joined Q/K/V group changed output boundaries");
            if(!mixed && combine && rows>1 && rows<=32)
                require(!actual[0].is_contiguous() && actual[0].stride(-1)==1,
                    "GGUF grouped projection materialized its joined output slice");
            for(std::size_t i=0;i<actual.size();++i)
                require((actual[i].to(tb::kFloat32)-expected[i].to(tb::kFloat32)).abs().max().item<float>()<1e-6f,
                    "GGUF grouped projection changed a mixed-format or dense auxiliary output");
        }
    }
}
void run_compact_projection_runtime() {
    const auto context = mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream = mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    struct PoolEnd {
        std::shared_ptr<mfq::cuda::Context> context; cudaStream_t stream;
        ~PoolEnd() { context->end_graph_pool(stream); }
    } pool_end{context, stream};
    int cases = 0;
    for (int format = GGML_TYPE_TQ2_T; format <= GGML_TYPE_TQK4; ++format)
    for (int columns : {128, 640}) {
        const auto type = ggml_type(format);
        Source source; source.add("compact", {7, columns}, type); source.full_compact_codes("compact", type);
        const auto bytes = source.read("compact");
        std::vector<float> decoded(7 * columns);
        ggml_get_type_traits(type)->to_float(bytes.data(), decoded.data(), decoded.size());
        auto weight = load_gguf_weight(source, "compact", mfq_current_cuda_device());
        for (int rows : {1, 2, 4}) for (auto dtype : {tb::kFloat16, tb::kFloat32, tb::kBFloat16}) {
            auto host = tb::empty({1, rows, columns}, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for (int64_t i = 0; i < host.numel(); ++i) host.data_ptr<float>()[i] = .7f * std::sin(float(i) * .173f + .41f);
            auto x = host.to(dtype).to(tb::kCUDA);
            tb::Tensor actual;
            const auto verify = [&] {
                const auto values = x.to(tb::kCPU, tb::kFloat32).contiguous();
                const auto result = actual.to(tb::kCPU, tb::kFloat32).contiguous();
                require(actual.sizes().vec() == std::vector<int64_t>({1, rows, 7}), "Gyro projection changed leading dimensions");
                for (int row = 0; row < rows; ++row) for (int out = 0; out < 7; ++out) {
                    double expected = 0, magnitude = 0;
                    for (int k = 0; k < columns; ++k) {
                        const double product = double(values.data_ptr<float>()[row * columns + k]) * decoded[out * columns + k];
                        expected += product; magnitude += std::abs(product);
                    }
                    const double value = result.data_ptr<float>()[row * 7 + out];
                    const double round = dtype == tb::kFloat32 ? 0 : .0005 * std::abs(expected) + 1e-6;
                    require(std::isfinite(value) && std::abs(value - expected) <= 2e-6 * magnitude + 1e-6 + round,
                        "Gyro runtime changed direct FP16/FP32/BF16 activations or missed compact dispatch");
                }
                ++cases;
            };
            actual = weight.forward(x); verify();
            if (rows == 4 && dtype == tb::kFloat16) {
                mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                graph.capture([&] { actual = weight.forward(x); }, [] {}, [&] { actual = {}; });
                for (int replay = 0; replay < 3; ++replay) {
                    x.copy_(x * -.5f); graph.run(); verify();
                }
            }
        }
    }
    std::cout << "Gyro runtime direct-input/loading/leading-shape/FP16-FP32-BF16/changing-graph cases=" << cases << " PASS\n";
}

void run_hadamard_projection_runtime() {
    int cases=0;
    for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_IQ3_S,GGML_TYPE_PTQ1_0,GGML_TYPE_TQK6,GGML_TYPE_TQP6}) {
        constexpr int columns=256,outputs=7,block=128;
        Source source;source.add("rotated",{outputs,columns},type);source.add("table",{outputs,columns},GGML_TYPE_Q4_0);
        if(type>=GGML_TYPE_PTQ1_0)source.full_compact_codes("rotated",type);
        else if(type==GGML_TYPE_IQ3_S)source.full_iq3_codes("rotated");
        source.add_hadamard("rotated",block);
        auto weight=load_gguf_weight(source,"rotated",mfq_current_cuda_device());
        require(bool(weight.rotation),"Hadamard weight lost its activation basis");
        auto plain=weight;plain.rotation={};
        const auto raw=source.decoded("rotated").contiguous();
        const auto original=weight.dequantize().cpu().contiguous();
        // Explicit Walsh matrix, independent of the butterfly implementation.
        for(int row=0;row<outputs;++row)for(int k=0;k<columns;++k) {
            double expected=0,magnitude=0;
            for(int j=0;j<block;++j) {
                const double value=raw.data_ptr<float>()[row*columns+(k/block)*block+j];
                expected+=(std::popcount(unsigned((k%block)&j))&1)?-value:value;magnitude+=std::abs(value);
            }
            const double sign=weight.rotation->signs_cpu.empty()?1:weight.rotation->signs_cpu[k];
            expected*=sign/std::sqrt(double(block));magnitude/=std::sqrt(double(block));
            require(std::abs(double(original.data_ptr<float>()[row*columns+k])-expected)<=2e-6*magnitude+1e-6,
                "Hadamard public dequantization did not undo the stored weight basis");
        }
        const auto table=std::make_shared<GgufRowTable>(source,"table");
        require(table->width()==columns,"Unrotated row table was rejected by another weight's transform");
        auto lookup_ids=tb::arange(2,tb::TensorOptions().dtype(tb::kInt64).device(tb::kCUDA))*6;
        const auto embeddings=weight.embedding(lookup_ids).cpu().contiguous();
        for(int row=0;row<2;++row)for(int k=0;k<columns;++k)
            require(embeddings.data_ptr<float>()[row*columns+k]==original.data_ptr<float>()[row*6*columns+k],
                "Hadamard embedding did not restore the original row basis");
        const auto row_slice=load_gguf_weight(source,"rotated",mfq_current_cuda_device(),true,2,5);
        const auto row_values=row_slice.dequantize().cpu();
        require(std::memcmp(row_values.data_ptr(),original.data_ptr<float>()+2*columns,3*columns*sizeof(float))==0,
            "Hadamard row slice changed the input basis");++cases;
        if(ggml_blck_size(type)<=block) {
            const auto column_slice=load_gguf_weight(source,"rotated",mfq_current_cuda_device(),true,0,outputs,block,columns);
            const auto column_values=column_slice.dequantize().cpu();
            for(int row=0;row<outputs;++row)for(int k=0;k<block;++k)
                require(column_values.data_ptr<float>()[row*block+k]==original.data_ptr<float>()[row*columns+block+k],
                    "Hadamard column slice did not crop its sign vector at a transform boundary");++cases;
        }
        const auto cpu_weight=load_gguf_weight(source,"rotated",mfq_current_cuda_device(),false);
        const auto cpu_input=input(2,1,false).cpu();
        const auto cpu_rotated=cpu_weight.rotation->apply(cpu_input);
        const auto cpu_original=cpu_weight.dequantize().contiguous();
        for(int i=0;i<outputs*columns;++i)
            require(std::abs(cpu_original.data_ptr<float>()[i]-original.data_ptr<float>()[i])<=
                2e-6f*std::max(1.f,std::abs(original.data_ptr<float>()[i])),"CPU inverse Hadamard changed the weight basis");
        std::vector<double> walsh(2*columns);
        for(int row=0;row<2;++row)for(int k=0;k<columns;++k) {
            double value=0;
            for(int j=0;j<block;++j) {
                const int at=(k/block)*block+j;
                const double term=cpu_input.data_ptr<float>()[row*columns+at]*cpu_weight.rotation->signs_cpu[at];
                value+=(std::popcount(unsigned((k%block)&j))&1)?-term:term;
            }
            walsh[row*columns+k]=value/std::sqrt(double(block));
            require(std::abs(cpu_rotated.data_ptr<float>()[row*columns+k]-walsh[row*columns+k])<=
                3e-6*std::max(1.,std::abs(walsh[row*columns+k])),"CPU Hadamard activation differs from the explicit Walsh matrix");
        }
        ++cases;
        for(int rows:{1,2,4})for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16}) {
            auto x=input(rows,1,false).to(dtype).reshape({1,rows,columns});
            const auto rotated=weight.rotation->apply(x);
            auto actual=weight.forward(x),expected=plain.forward(rotated).to(actual.scalar_type());
            const float bound=(actual.scalar_type()==tb::kFloat16?.0015f:2e-6f)*expected.to(tb::kFloat32).abs().max().item<float>()+1e-6f;
            require((actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>()<=bound,
                "Hadamard dense projection rotated after quantization or changed its input basis");
            const auto grouped=project_gguf_group({weight,plain},x);
            require((grouped[0].to(tb::kFloat32)-actual.to(tb::kFloat32)).abs().max().item<float>()==0 &&
                (grouped[1].to(tb::kFloat32)-plain.forward(x).to(tb::kFloat32)).abs().max().item<float>()==0,
                "Mixed Hadamard/plain projection group shared incompatible activation bases");
            ++cases;
        }
        auto duplicate=weight;require(same_gguf_input_basis(weight,duplicate),"Identical Hadamard basis did not compare equal");
        auto joined=combine_gguf_rows({&weight,&duplicate});
        auto x=input(2,1,false).to(tb::kFloat16);
        const auto expected=tb::cat({weight.forward(x),duplicate.forward(x)},-1);
        require((joined.forward(x)-expected).abs().max().item<float>()==0,"Joined rows lost their Hadamard basis");
        bool rejected=false;
        try{(void)combine_gguf_rows({&weight,&plain});}catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"Row combination accepted incompatible activation bases");
        auto q=tb::empty({int64_t(mfq::cuda::gguf::q8_row_bytes(columns))},weight.packed.options());
        rejected=false;
        try{(void)weight.forward_q8(q,1,tb::kFloat16);}catch(const std::exception&){rejected=true;}
        require(rejected,"Rotated weight accepted an activation that was already quantized");
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
        {
            tb::Tensor graph_output;
            mfq::cuda::DecodeWindow graph(stream);
            graph.capture([&]{graph_output=weight.forward(x);},[]{},[&]{graph_output={};});
            for(int replay=0;replay<3;++replay) {
                x.copy_(x*-.5f);graph.run();
                require((graph_output-weight.forward(x)).abs().max().item<float>()==0,
                    "Hadamard graph reused a previous activation's transform");
                ++cases;
            }
        }
        context->end_graph_pool(stream);
        source.add_hadamard("rotated",1,true);
        const auto identity=load_gguf_weight(source,"rotated",mfq_current_cuda_device());
        require(!identity.rotation && weight.rotation,"Identity basis was not simplified or mutated a loaded basis");
    }
    std::cout<<"Hadamard dense/CPU-GPU basis/dequant/embedding/slicing/grouping/immutable-signs/changing-graph cases="<<cases<<" PASS\n";
}

class ScopedEnvironment {
    std::string key_,previous_;
    bool present_=false;
public:
    ScopedEnvironment(const char* key,const char* value):key_(key) {
        if(const auto* old=std::getenv(key_.c_str())){present_=true;previous_=old;}
        set(value);
    }
    ~ScopedEnvironment(){set(present_?previous_.c_str():nullptr);}
    void set(const char* value) {
#ifdef _WIN32
        _putenv_s(key_.c_str(),value?value:"");
#else
        if(value)setenv(key_.c_str(),value,1);else unsetenv(key_.c_str());
#endif
    }
};
void run_gguf_residual_activation(int bench_tokens=0,int bench_columns=0,int bench_streams=0) {
    namespace gg=mfq::cuda::gguf;
    const bool benchmark=bench_tokens>0;
    if(benchmark && (bench_columns<=0 || bench_columns%32 || bench_streams<=0))
        throw std::invalid_argument("Residual activation benchmark needs positive Q8 dimensions");
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();int cases=0,changed=0;
    const bool own_pool=!context->has_graph_pool(stream);
    if(own_pool)context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> c;cudaStream_t s;bool owned;
        ~PoolEnd(){if(owned)c->end_graph_pool(s);}} pool{context,stream,own_pool};
    for(int tokens:benchmark?std::vector<int>{bench_tokens}:std::vector<int>{1,2,3,4})
    for(int columns:benchmark?std::vector<int>{bench_columns}:std::vector<int>{32,96,320,544})
    for(int streams:benchmark?std::vector<int>{bench_streams}:std::vector<int>{1,2,3,5}) {
        auto host=tb::empty({tokens,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto ihost=tb::empty({tokens,streams},host.options());
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=i%7==0?0.f:float(streams)*20.f*std::sin(float(i)*.173f);
        for(int64_t i=0;i<ihost.numel();++i)ihost.data_ptr<float>()[i]=float(streams)*12.f*std::cos(float(i)*.371f);
        auto down=host.to(tb::kCUDA),injection=ihost.to(tb::kCUDA);
        const auto bytes=tokens*gg::q8_row_bytes(columns);
        auto actual=tb::full({int64_t(bytes+16)},165,down.options().dtype(tb::kUInt8));
        auto gates=tb::full({int64_t(tokens*streams+2)},42.5,down.options());
        auto expected=tb::empty({int64_t(bytes)},actual.options());
        tb::Tensor low,wanted_gates;
        const auto reference=[&] {
            low=mfq_qwen4_exp::gated_residual_bottleneck(down,streams);
            wanted_gates=mfq_qwen4_exp::gated_residual_injection(injection,streams);
            gg::quantize_input(low.data_ptr<float>(),tokens,columns,expected.data_ptr(),stream);
        };
        const auto fused=[&] {
            gg::prepare_residual_activation(down.data_ptr<float>(),injection.data_ptr<float>(),tokens,columns,streams,
                actual.data_ptr<uint8_t>()+8,gates.data_ptr<float>()+1,stream);
        };
        const auto verify=[&] {
            reference();const auto a=actual.cpu(),e=expected.cpu(),g=gates.cpu(),w=wanted_gates.cpu();
            require(std::memcmp(a.data_ptr<uint8_t>()+8,e.data_ptr(),bytes)==0,"Fused GR bottleneck changed Q8 packet bytes");
            require(std::memcmp(g.data_ptr<float>()+1,w.data_ptr(),w.nbytes())==0,"Fused GR injection changed float rounding");
            for(int i=0;i<8;++i)require(a.data_ptr<uint8_t>()[i]==165 && a.data_ptr<uint8_t>()[bytes+8+i]==165,
                "Fused GR activation overwrote Q8 guard");
            require(g.data_ptr<float>()[0]==42.5f && g.data_ptr<float>()[tokens*streams+1]==42.5f,
                "Fused GR activation overwrote injection guard");
        };
        fused();verify();++cases;
        if(benchmark || (tokens==3 && columns==320 && streams==3)) {
            mfq::cuda::Graph graphs[2];
            for(int mode=0;mode<2;++mode) {
                {mfq::cuda::GraphWarmupScope warm(context,stream);if(mode)fused();else reference();}
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));low={};wanted_gates={};
                graphs[mode].capture_begin_shared();if(mode)fused();else reference();graphs[mode].capture_end();
            }
            for(float factor:{0.f,.5f,-1.5f}) {
                down.copy_(host.to(tb::kCUDA)*factor);injection.copy_(ihost.to(tb::kCUDA)*factor);
                graphs[1].replay();verify();++changed;
            }
            if(benchmark) {
                for(int i=0;i<256;++i){graphs[0].replay();graphs[1].replay();}
                std::vector<float> times[2];
                for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
                    const int mode=(trial+order)%2;mfq::cuda::Event begin(0),end(0);begin.record(stream);
                    for(int i=0;i<512;++i)graphs[mode].replay();end.record(stream);end.synchronize();
                    float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));times[mode].push_back(ms*1000/512);
                }
                for(auto& values:times)std::sort(values.begin(),values.end());
                std::printf("gguf_residual_activation M=%d columns=%d streams=%d separate_us=%.6f fused_us=%.6f ratio=%.6f launches=3_to_1 bitexact=1\n",
                    tokens,columns,streams,times[0][3],times[1][3],times[1][3]/times[0][3]);
            }
        }
    }
    std::printf("GGUF fused residual activation/Q8/injection cases=%d changing_graph=%d exact/guards PASS\n",cases,changed);
}

void run_gguf_q6_group() {
    namespace gg=mfq::cuda::gguf;
    const auto stream=mfq_current_cuda_stream();int cases=0,graphs=0;
    for(int columns:{256,768,4096,10240,16640})for(int rows:{1,7,33,321}) {
        Source source;source.add("down",{rows,columns},GGML_TYPE_Q6_K);
        source.add("inject",{3,columns},GGML_TYPE_Q6_K);
        source.full_q6_codes("down");source.full_q6_codes("inject");
        const auto down=load_gguf_weight(source,"down",mfq_current_cuda_device());
        const auto inject=load_gguf_weight(source,"inject",mfq_current_cuda_device());
        const tb::Tensor decoded[]={source.decoded("down"),source.decoded("inject")};
        for(int tokens:{1,2,3,4})for(bool half:{false,true}) {
            const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
            auto host=tb::empty({tokens,columns},options.device(tb::kCPU));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.3f*std::sin(float(i)*.173f);
            auto x=host.to(tb::kCUDA),q=tb::empty({int64_t(tokens*gg::q8_row_bytes(columns))},options.dtype(tb::kUInt8));
            const auto dtype=half?tb::kFloat16:tb::kFloat32;
            auto a=tb::full({tokens*rows+2},-12344.f,options.dtype(dtype));
            auto b=tb::full({tokens*3+2},-12344.f,options.dtype(dtype));
            auto out=a.narrow(0,1,tokens*rows),aux=b.narrow(0,1,tokens*3);
            const gg::DenseProjection projections[]={
                {down.packed.data_ptr(),down.row_bytes,rows,out.data_ptr(),GGML_TYPE_Q6_K},
                {nullptr,0,0,nullptr,GGML_TYPE_Q6_K},
                {inject.packed.data_ptr(),inject.row_bytes,3,aux.data_ptr(),GGML_TYPE_Q6_K}};
            const auto compute=[&]{gg::quantize_input(x.data_ptr<float>(),tokens,columns,q.data_ptr(),stream);
                gg::matvec_group(projections,3,columns,q.data_ptr(),nullptr,tokens,half,stream);};
            const auto verify=[&] {
                const auto packed=q.cpu();const auto* input=static_cast<const block_q8_1*>(packed.data_ptr());
                const tb::Tensor actual[]={a.to(tb::kCPU,tb::kFloat32),b.to(tb::kCPU,tb::kFloat32)};
                for(int matrix=0;matrix<2;++matrix) {
                    const int n=matrix?3:rows;const auto* values=actual[matrix].data_ptr<float>();
                    require(values[0]==-12344.f && values[tokens*n+1]==-12344.f,"Q6 group overwrote an output boundary");
                    for(int i=0;i<tokens*n;++i)require(std::isfinite(values[i+1]),"Q6 group returned nonfinite output");
                    for(int row:{0,n/2,n-1})for(int token=0;token<tokens;++token) {
                        double reference=0,magnitude=0;
                        for(int k=0;k<columns;++k) {
                            const auto& block=input[token*(columns/32)+k/32];float scale;
                            ggml_fp16_to_fp32_row(&block.d,&scale,1);
                            const double term=double(decoded[matrix].data_ptr<float>()[row*columns+k])*scale*block.qs[k%32];
                            reference+=term;magnitude+=std::abs(term);
                        }
                        const double error=std::abs(values[1+token*n+row]-reference);
                        const double allowance=2.e-6*magnitude+1.e-7+(half?.001*std::abs(reference)+1.e-6:0);
                        require(error<=allowance,"Q6 group differs from independent FP64 canonical weight/Q8 reference");
                    }
                }
            };
            compute();verify();++cases;
            if(columns==10240 && rows==33 && tokens>=3 && !half) {
                mfq::cuda::DecodeWindow graph(stream);graph.capture(compute,[]{},[]{});
                for(float amplitude:{0.f,-.3f,2.f}){x.copy_(host.to(tb::kCUDA)*amplitude);graph.run();verify();++graphs;}
            }
        }
    }
    std::cout<<"Q6 grouped projections M1..4 FP16/FP32, signed/zero scales, empty member, tails, guards, independent FP64 cases="
        <<cases<<" changing_graph="<<graphs<<" PASS\n";
}

void run_gguf_residual_pre_runtime() {
    run_gguf_residual_activation();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> c;cudaStream_t s;~PoolEnd(){c->end_graph_pool(s);}} end{context,stream};
    run_gguf_q6_group();
    CudaExecutionContext execution;
    int cases=0,graphs=0;
    const auto check_type=[&](ggml_type type) {
        for(int streams:{1,3,5}) {
            const int hidden=256,width=hidden*streams,rank=96;
            Source source;source.add("down",{rank,width},type);source.add("up",{width,rank},GGML_TYPE_Q8_0);
            source.add("inject",{streams,width},GGML_TYPE_Q6_K);
            if(type>=GGML_TYPE_PTQ1_0 && type<=GGML_TYPE_TQK4)source.full_compact_codes("down",type);
            auto down=weight_loader::residual_linear(execution,source,"down");
            auto up=weight_loader::residual_linear(execution,source,"up");
            auto inject=weight_loader::residual_linear(execution,source,"inject");
            auto norm=tb::zeros({width},tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCUDA));
            for(int rows:{1,2,3,4})for(bool after:{false,true}) {
                auto x=tb::full({1,rows,width},.03125f,norm.options());
                auto branch=after?tb::full({1,rows,hidden},.125f,norm.options()):tb::Tensor{};
                auto prior=after?tb::full({1,rows,streams},.75f,norm.options()):tb::Tensor{};
                const auto reference=[&] {
                    auto updated=x,normalized=tb::Tensor{};
                    if(after){auto p=mfq_qwen4_exp::gated_residual_post_norm(branch,x,prior,norm,streams,1e-6);updated=p[0];normalized=p[1];}
                    else normalized=mfq_qwen4_exp::grouped_rms_norm(x,norm,hidden,1e-6);
                    auto low=mfq_qwen4_exp::gated_residual_bottleneck(down.forward(execution,normalized),streams);
                    auto mixed=up.mixed(execution,low,normalized,streams);
                    auto injection=mfq_qwen4_exp::gated_residual_injection(inject.forward(execution,normalized),streams);
                    return std::vector<tb::Tensor>{mixed,updated,injection};
                };
                const auto compute=[&]{return try_gguf_gated_residual_pre(x,norm,down.weight->gguf,up.weight->gguf,inject.weight->gguf,streams,1e-6,branch,prior);};
                const auto verify=[&](const auto& actual) {
                    const auto expected=reference();
                    require(actual.size()==expected.size(),"GGUF GR prepared tuple missing");
                    for(int i=0;i<3;++i){require(actual[i].sizes()==expected[i].sizes() && actual[i].scalar_type()==expected[i].scalar_type(),"GGUF GR prepared shape/dtype changed");
                        const auto a=actual[i].cpu().contiguous(),b=expected[i].cpu().contiguous();
                        for(int64_t k=0;k<a.numel();++k)require(std::isfinite(a.template data_ptr<float>()[k]) && std::abs(a.template data_ptr<float>()[k]-b.template data_ptr<float>()[k])<=4e-6f*(1.f+std::abs(b.template data_ptr<float>()[k])),"GGUF GR prepared differs from sequential projection reference");}
                };
                if(down.weight->gguf.prefers_original_input()) {
                    require(compute().empty(),"Original-input GGUF GR unexpectedly quantized activations");
                    continue;
                }
                verify(compute());++cases;
                if(type==GGML_TYPE_Q6_K && streams==3 && rows==3 && after) {
                    std::vector<tb::Tensor> actual;mfq::cuda::DecodeWindow graph(stream);
                    graph.capture([&]{actual=compute();},[]{},[&]{actual.clear();});
                    for(float value:{0.f,.0625f,-.125f}){x.fill_(value);branch.fill_(value+.1f);graph.run();verify(actual);++graphs;}
                }
                require(try_gguf_gated_residual_pre(x.to(tb::kFloat16),norm,down.weight->gguf,up.weight->gguf,inject.weight->gguf,streams,1e-6).empty(),"GGUF GR changed FP16 rounding path");
            }
        }
    };
#define CHECK_GR_PREPARED(name) check_type(GGML_TYPE_##name);
    TYLOINFER_GGUF_FORMATS(CHECK_GR_PREPARED)
#undef CHECK_GR_PREPARED
    std::cout<<"GGUF GR prepared Down+injection/shared Q8/original compressed weights M1..4 streams1,3,5 cases="<<cases<<" changing_graph="<<graphs<<" fallback/reference PASS\n";
}

void run_gguf_residual_mix_runtime() {
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~PoolEnd(){context->end_graph_pool(stream);}} pool_end{context,stream};
    CudaExecutionContext execution;
    int cases=0,graphs=0,fallbacks=0;
    const auto check_format=[&](ggml_type type) {
        for(int streams:{1,2,3,4,5,8,9}) {
            const int columns=3*ggml_blck_size(type),hidden=7;
            Source source;source.add("up",{hidden*streams,columns},type);
            if(type>=GGML_TYPE_PTQ1_0 && type<=GGML_TYPE_TQK4)source.full_compact_codes("up",type);
            auto projection=weight_loader::residual_linear(execution,source,"up");
            require(bool(projection.mixed),"GGUF residual mix callback absent");
            const auto& weight=projection.weight->gguf;
            const bool supported=mfq::cuda::gguf::residual_mix_supported(type);
            for(int rows:{1,2,3,4}) {
                auto host=tb::empty({1,columns,rows},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
                auto norm=tb::empty({1,hidden*streams,rows},host.options());
                for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.31f*std::sin(float(i)*.37f+.23f);
                for(int64_t i=0;i<norm.numel();++i)norm.data_ptr<float>()[i]=.23f*std::cos(float(i)*.71f+.51f);
                // Noncontiguous inputs also cover the wrapper's staging and
                // leading dimensions; all stream counts include odd tails.
                auto x=host.to(tb::kCUDA).transpose(1,2),normalized=norm.to(tb::kCUDA).transpose(1,2);
                const auto verify=[&](const tb::Tensor& actual) {
                    tb::Tensor expected;
                    {ScopedEnvironment disable("TYLOINFER_GGUF_GR_MIX","0");expected=projection.mixed(execution,x,normalized,streams);}
                    require(actual.sizes()==expected.sizes() && actual.scalar_type()==expected.scalar_type(),
                        "GGUF residual fusion changed shape/dtype");
                    const auto a=actual.to(tb::kCPU,tb::kFloat32).contiguous(),b=expected.to(tb::kCPU,tb::kFloat32).contiguous();
                    for(int64_t i=0;i<a.numel();++i)require(std::isfinite(a.data_ptr<float>()[i]) &&
                        std::abs(a.data_ptr<float>()[i]-b.data_ptr<float>()[i])<=3e-6f*(1.f+std::abs(b.data_ptr<float>()[i])),
                        "GGUF residual fusion exceeds FP32 reduction allowance");
                };
                const auto direct=try_gguf_residual_mix(weight,x,normalized,streams);
                require(direct.defined()==supported,"GGUF residual fusion capability disagrees");
                verify(projection.mixed(execution,x,normalized,streams));++cases;
                if(type==GGML_TYPE_Q8_0 && streams>=3 && rows==3) {
                    tb::Tensor actual;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                    graph.capture([&]{actual=projection.mixed(execution,x,normalized,streams);},[]{},[&]{actual={};});
                    for(float factor:{0.f,.3f,1.f}) {
                        x.copy_(host.transpose(1,2).contiguous().to(tb::kCUDA)*factor);
                        normalized.copy_(norm.transpose(1,2).contiguous().to(tb::kCUDA)*(factor+.1f));
                        graph.run();verify(actual);++graphs;
                    }
                }
                if(rows==2 && streams==3) {
                    x=x.to(tb::kFloat16);normalized=normalized.to(tb::kFloat16);
                    require(!try_gguf_residual_mix(weight,x,normalized,streams).defined(),
                        "FP16 residual must preserve its projection/sigmoid rounding through the general path");
                    verify(projection.mixed(execution,x,normalized,streams));++fallbacks;
                }
            }
        }
    };
#define CHECK_GGUF_RESIDUAL(name) check_format(GGML_TYPE_##name);
    TYLOINFER_GGUF_FORMATS(CHECK_GGUF_RESIDUAL)
#undef CHECK_GGUF_RESIDUAL
    std::cout<<"GGUF residual mix all28formats M1..4 streams1,2,3,4,5,8,9 strided/shared-activation/fallback cases="
        <<cases<<" changing_graph="<<graphs<<" FP16_fallback="<<fallbacks<<" PASS\n";
}

void bench_file_residual_mix(const std::filesystem::path& path,int streams) {
    mfq::GgufModelSource source(path);std::string name;
    for(const auto& record:source.tensors())if(record.name.ends_with(".attention.mhc.pre.up.weight")){name=record.name;break;}
    require(!name.empty(),"GGUF file has no GR Up projection");
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();
    context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~PoolEnd(){context->end_graph_pool(stream);}} end_pool{context,stream};
    CudaExecutionContext execution;auto projection=weight_loader::residual_linear(execution,source,name);
    const auto& weight=projection.weight->gguf;
    require(streams>0 && weight.out%streams==0,"Residual benchmark stream geometry");
    for(int tokens:{1,2,4}) {
        const auto host_options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
        auto h=tb::empty({1,tokens,weight.neuron_len},host_options),n=tb::empty({1,tokens,weight.out},host_options);
        for(int64_t i=0;i<h.numel();++i)h.data_ptr<float>()[i]=.1f*std::sin(float(i)*.371f+.12f);
        for(int64_t i=0;i<n.numel();++i)n.data_ptr<float>()[i]=.25f*std::cos(float(i)*.157f+.63f);
        auto input=h.to(tb::kCUDA),normalized=n.to(tb::kCUDA);
        struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;tb::Tensor result;
            ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}} graphs[2];
        std::vector<const void*> functions[2];std::vector<float> expected;
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        for(int mode=0;mode<2;++mode) {
            ScopedEnvironment setting("TYLOINFER_GGUF_GR_MIX",mode?"1":"0");
            require(!mode || try_gguf_residual_mix(weight,input,normalized,streams).defined(),"Residual benchmark bypassed its fused path");
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            for(int pass=0;pass<3;++pass) {
                {mfq::cuda::GraphWarmupScope warmup(context,stream);
                    graphs[mode].result=projection.mixed(execution,input,normalized,streams);}
                TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));graphs[mode].result={};
            }
            {
                context->begin_graph_capture(stream);
                struct CaptureEnd {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
                    ~CaptureEnd(){context->end_graph_capture(stream);}} capture_end{context,stream};
                TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                graphs[mode].result=projection.mixed(execution,input,normalized,streams);
                TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode].graph));
            }
            TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&graphs[mode].exec,graphs[mode].graph,0));
            TYLOINFER_CUDA_CHECK(cudaGraphLaunch(graphs[mode].exec,stream));
            const auto output=graphs[mode].result.cpu().contiguous();
            if(!mode)expected.assign(output.data_ptr<float>(),output.data_ptr<float>()+output.numel());
            else for(int64_t i=0;i<output.numel();++i)require(std::isfinite(output.data_ptr<float>()[i]) &&
                std::abs(output.data_ptr<float>()[i]-expected[i])<=3e-6f*(1.f+std::abs(expected[i])),"Real GGUF residual mix rounding bound");
            size_t count=0;TYLOINFER_CUDA_CHECK(cudaGraphGetNodes(graphs[mode].graph,nullptr,&count));
            std::vector<cudaGraphNode_t> nodes(count);TYLOINFER_CUDA_CHECK(cudaGraphGetNodes(graphs[mode].graph,nodes.data(),&count));
            for(auto node:nodes) {cudaGraphNodeType type;TYLOINFER_CUDA_CHECK(cudaGraphNodeGetType(node,&type));
                require(type==cudaGraphNodeTypeKernel,"Residual benchmark expected a device-only graph");
                cudaKernelNodeParams params{};TYLOINFER_CUDA_CHECK(cudaGraphKernelNodeGetParams(node,&params));functions[mode].push_back(params.func);}
        }
        require(functions[0].size()==3 && functions[1].size()==2,"Residual benchmark did not replace projection+mix with one kernel");
        require(std::any_of(functions[1].begin(),functions[1].end(),[&](const auto* f){return std::find(functions[0].begin(),functions[0].end(),f)==functions[0].end();}),
            "Residual benchmark reused only baseline kernels");
        for(int i=0;i<512;++i)for(auto& g:graphs)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(g.exec,stream));
        cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));std::vector<float> times[2];
        for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
            const int mode=(trial+order)%2;TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int i=0;i<512;++i)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(graphs[mode].exec,stream));
            TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));times[mode].push_back(ms*1000/512);
        }
        for(auto& values:times)std::sort(values.begin(),values.end());
        std::printf("real_gguf_residual_mix format=%d M=%d N=%lld K=%lld streams=%d old_us=%.6f new_us=%.6f ratio=%.6f kernels=3->2 actual_match=1\n",
            int(weight.type),tokens,static_cast<long long>(weight.out),static_cast<long long>(weight.neuron_len),streams,times[0][3],times[1][3],times[1][3]/times[0][3]);
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
    }
}

void run_hadamard_moe_runtime() {
    ScopedEnvironment gpu_only("TYLOINFER_GGUF_CPU_EXPERTS","0");
    int cases=0;
    for(int kind=0;kind<3;++kind)for(int basis=0;basis<4;++basis)for(bool resident:{false,true}) {
        Source source;source.enable_batches();
        const std::string prefix=resident?"predictor.test.":"";
        const auto gate_type=kind==0?GGML_TYPE_Q4_0:kind==1?GGML_TYPE_PTQ1_0:GGML_TYPE_IQ3_S;
        const auto up_type=kind==1?GGML_TYPE_TQK6:gate_type;
        const auto down_type=kind==1?GGML_TYPE_TQP6:kind==2?GGML_TYPE_IQ4_NL:GGML_TYPE_Q4_0;
        for(const auto& [name,type]:{std::pair{"gate",gate_type},std::pair{"up",up_type},std::pair{"down",down_type}}) {
            source.add(prefix+name,{3,256,256},type);
            if(type>=GGML_TYPE_PTQ1_0)source.full_compact_codes(prefix+name,type);
            else if(type==GGML_TYPE_IQ3_S)source.full_iq3_codes(prefix+name);
        }
        const int block=basis==3?256:128;
        if(basis!=2)source.add_hadamard(prefix+"gate",block);
        source.add_hadamard(prefix+"up",block,false,basis==1?2:0);
        source.add_hadamard(prefix+"down",block,false,1);
        CudaExecutionContext execution;
        execution.config.moe_host_cache_bytes=4<<20;execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,prefix+"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,prefix+"up"),std::make_shared<GgufRoutedWeight>(execution,source,prefix+"down")};
        require(same_gguf_rotation(banks.gate->rotation,banks.up->rotation)==(basis==0 || basis==3),"Rotation fixture did not exercise the requested input bases");
        for(int rows:{1,2,4})for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16}) {
            auto x=input(rows,3,false).to(dtype),ids=routes(rows,0);
            auto weights=tb::full({rows,3},.13,x.options().dtype(tb::kFloat32));
            check_moe(banks.forward(x,ids,weights,.2f),banks,x,ids,weights,.2f);++cases;
        }
        require(prepare_gguf_graph_pool(execution.gguf_expert_cache),"Rotated expert graph pool missing");
        auto x=input(4,3,false).to(tb::kFloat16),ids=routes(4,0);
        auto weights=tb::full({4,3},.13,x.options().dtype(tb::kFloat32));
        tb::Tensor actual;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{actual=banks.forward(x,ids,weights,.2f);},[]{},[&]{actual={};});
        for(int next:{1,2}) {
            x.copy_(x*-.5f);ids.copy_(routes(4,next));graph.run();
            check_moe(actual,banks,x,ids,weights,.2f);++cases;
        }
        auto px=input(64,3,false).to(tb::kFloat16),pi=routes(64,0);
        auto pw=tb::full({64,3},.13,px.options().dtype(tb::kFloat32));
        const auto result=banks.forward(px,pi,pw,.2f);
        if(kind<2)check_moe(result,banks,px,pi,pw,.2f);
        else {
            // A separate expert-by-expert MMQ pipeline verifies the cache,
            // grouped activation permutation and each rotation boundary.
            // Avoid a floating GEMM surrogate: tiny projection roundoff can
            // cross the next Q8 threshold, especially after a wide transform.
            namespace mmq=mfq::cuda::gguf::mmq;
            const auto stream=mfq_current_cuda_stream();
            const auto project=[&](const std::string& name,ggml_type type,int expert,tb::Tensor x) {
                const auto raw=source.read(prefix+name);
                const auto bytes=256*mfq::gguf::row_bytes(type,256),guard=128*ggml_type_size(type);
                auto host=tb::zeros({int64_t(bytes+guard)},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
                std::memcpy(host.data_ptr(),raw.data()+expert*bytes,bytes);
                auto packed=host.to(tb::kCUDA);
                auto activation=tb::empty({int64_t(mmq::activation_bytes(64,256))},packed.options());
                auto result=tb::empty({64,256},x.options().dtype(tb::kFloat32));
                mmq::quantize(x.to(tb::kFloat32).data_ptr<float>(),nullptr,activation.data_ptr(),int(type),256,64,stream);
                mmq::Context context;
                context.run({packed.data_ptr(),int(type),256,256,bytes,1,activation.data_ptr(),nullptr,nullptr,64,64,result.data_ptr<float>(),256},stream);
                return result;
            };
            auto gx=banks.gate->rotation?banks.gate->rotation->apply(px):px;
            const auto ux=banks.up->rotation->apply(px);
            auto expected=tb::zeros({64,256},px.options().dtype(tb::kFloat32));
            for(int e=0;e<3;++e) {
                auto g=project("gate",gate_type,e,gx),u=project("up",up_type,e,ux);
                auto gu=tb::cat({g,u},-1),hidden=tb::empty_like(g);
                mmq::swiglu(gu.data_ptr<float>(),hidden.data_ptr<float>(),64,256,.2f,stream);
                banks.down->rotation->apply_inplace(hidden);
                auto y=project("down",down_type,e,hidden).to(tb::kFloat16).to(tb::kFloat32);
                expected=expected+y*((pi==e).to(tb::kFloat32)*pw).sum(1,true);
            }
            const auto error=(result.to(tb::kFloat32)-expected.to(tb::kFloat16).to(tb::kFloat32)).abs();
            const auto normalized=(error/(expected.abs()*.001f+1e-6f)).max().item<float>();
            if(!(std::isfinite(normalized) && normalized<=1.f))
                std::cerr<<"Rotated PP kind="<<kind<<" basis="<<basis<<" resident="<<resident<<" normalized_error="<<normalized<<" max="<<expected.abs().max().item<float>()<<'\n';
            require(std::isfinite(normalized) && normalized<=1.f,
                "Rotated MMQ prefill changed the input/hidden basis or expert grouping");
        }
        ++cases;
    }
    std::printf("Hadamard MoE shared/different/absent bases, register/shared fallback, streaming/resident, MMQ/GEMM PP and changing graphs cases=%d PASS\n",cases);
    std::fflush(stdout);
}

void bench_file_moe_projection(const std::filesystem::path& path,bool shared_lookup=false,bool decoded_input=false) {
    namespace gg=mfq::cuda::gguf;
    mfq::GgufModelSource source(path);
    std::string prefix;
    for(const auto& tensor:source.tensors()) {
        constexpr std::string_view suffix="mlp.experts.gate.weight";
        if(tensor.name.ends_with(suffix)){prefix=tensor.name.substr(0,tensor.name.size()-suffix.size());break;}
    }
    require(!prefix.empty(),"GGUF file has no canonical expert bank");
    const auto stream=mfq_current_cuda_stream();
    const auto options=tb::TensorOptions().device(tb::kCUDA);
    for(bool paired:{true,false}) {
        const auto name=prefix+(paired?"mlp.experts.gate.weight":"mlp.experts.down.weight");
        const auto* meta=source.find_tensor(name);require(meta && meta->shape && meta->shape->size()==3,"Benchmark needs expert banks");
        const auto experts=meta->shape->at(0),rows=meta->shape->at(1),columns=meta->shape->at(2);
        const auto type=mfq::gguf::quant_type(meta->dtype);const auto stride=mfq::gguf::row_bytes(type,columns);
        if(paired) {
            const auto* other=source.find_tensor(prefix+"mlp.experts.up.weight");
            require(other && other->dtype==meta->dtype && other->shape==meta->shape,
                "Paired benchmark requires matching formats and dimensions");
        }
        const auto expert_bytes=rows*stride;
        const auto load=[&](const std::string& name) {
            const auto* record=source.find_tensor(name);require(record && record->nbytes<=INT64_MAX,"Benchmark weight bytes");
            auto host=tb::empty({int64_t(record->nbytes)},options.device(tb::kCPU).dtype(tb::kUInt8));
            source.read_range_into(name,0,static_cast<std::byte*>(host.data_ptr()),host.nbytes());
            return host.to(tb::kCUDA);
        };
        auto packed=load(name),second=paired?load(prefix+"mlp.experts.up.weight"):tb::Tensor{};
        const auto table=[&](const tb::Tensor& data) {
            auto host=tb::empty({experts},options.device(tb::kCPU).dtype(tb::kInt64));
            for(int64_t e=0;e<experts;++e)host.data_ptr<int64_t>()[e]=reinterpret_cast<intptr_t>(data.data_ptr<uint8_t>()+e*expert_bytes);
            return host.to(tb::kCUDA);
        };
        auto addresses=table(packed),second_addresses=paired?table(second):tb::Tensor{};
        for(int diversity:decoded_input?std::vector<int>{0,1}:std::vector<int>{0}) for(int tokens:{1,2,3,4}) {
            const auto route_count=std::min(int64_t(diversity?1:3),experts);
            const auto slots=tokens*route_count,input_rows=paired?tokens:slots;
            auto host=tb::empty({input_rows,columns},options.device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.173f+.41f)*.1f;
            auto x=host.to(tb::kCUDA),q8=tb::empty({int64_t(input_rows*gg::q8_row_bytes(columns))},options.dtype(tb::kUInt8));
            gg::quantize_input(x.data_ptr<float>(),input_rows,columns,q8.data_ptr(),stream);
            auto ids=tb::arange(slots,options.dtype(tb::kInt64)).remainder(experts);
            auto groups=tb::empty({int64_t((slots+1)*sizeof(gg::RouteGroup))},options.dtype(tb::kUInt8));
            auto* plan=tokens>1?static_cast<gg::RouteGroup*>(groups.data_ptr()):nullptr;
            if(plan)gg::prepare_route_groups(ids.data_ptr<int64_t>(),slots,tokens,plan,stream);
            auto out=tb::empty({slots,rows},options.dtype(tb::kFloat32)),up=tb::empty_like(out);
            struct Graph {cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
                ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);}} graphs[2];
            std::vector<tb::Tensor> reference;
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            for(int mode=0;mode<2;++mode) {
                ScopedEnvironment setting("TYLOINFER_GGUF_COMPACT_ROUTED",shared_lookup || decoded_input || mode?"1":"0");
                TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
                if(shared_lookup || decoded_input) {
                    require(gg::try_compact_routed_project(gg::CompactFormat(int(type)),
                        reinterpret_cast<const uint64_t*>(addresses.data_ptr()),stride,experts,rows,columns,
                        ids.data_ptr<int64_t>(),plan,tokens,route_count,q8.data_ptr(),!paired,out.data_ptr<float>(),stream,false,
                        paired?reinterpret_cast<const uint64_t*>(second_addresses.data_ptr()):nullptr,
                        paired?stride:0,paired?up.data_ptr<float>():nullptr,decoded_input || mode!=0,!decoded_input || mode!=0),"Shared lookup benchmark not supported");
                }
                else if(paired) {
                    if(!gg::try_gate_up_matvec_q8(type,reinterpret_cast<const uint64_t*>(addresses.data_ptr()),
                        reinterpret_cast<const uint64_t*>(second_addresses.data_ptr()),stride,stride,experts,rows,columns,
                        ids.data_ptr<int64_t>(),plan,tokens,route_count,q8.data_ptr(),out.data_ptr<float>(),up.data_ptr<float>(),stream)) {
                        gg::grouped_routed_matvec_q8(type,reinterpret_cast<const uint64_t*>(addresses.data_ptr()),stride,experts,rows,columns,
                            ids.data_ptr<int64_t>(),plan,tokens,route_count,q8.data_ptr(),false,out.data_ptr<float>(),stream);
                        gg::grouped_routed_matvec_q8(type,reinterpret_cast<const uint64_t*>(second_addresses.data_ptr()),stride,experts,rows,columns,
                            ids.data_ptr<int64_t>(),plan,tokens,route_count,q8.data_ptr(),false,up.data_ptr<float>(),stream);
                    }
                }
                else gg::grouped_routed_matvec_q8(type,reinterpret_cast<const uint64_t*>(addresses.data_ptr()),stride,experts,rows,columns,
                    ids.data_ptr<int64_t>(),plan,tokens,route_count,q8.data_ptr(),true,out.data_ptr<float>(),stream);
                TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[mode].graph));
                TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&graphs[mode].exec,graphs[mode].graph,0));
                TYLOINFER_CUDA_CHECK(cudaGraphLaunch(graphs[mode].exec,stream));
                for(const auto& tensor:paired?std::vector<tb::Tensor>{out,up}:std::vector<tb::Tensor>{out}) {
                    const auto result=tensor.cpu();
                    if(!mode)reference.push_back(result);
                    else {
                        const auto expected=reference.front();reference.erase(reference.begin());
                        if(shared_lookup || decoded_input)require(!std::memcmp(result.data_ptr(),expected.data_ptr(),result.nbytes()),
                            "Shared codebook changes normative FP16 projection bits");
                        float maximum=0;for(int64_t i=0;i<expected.numel();++i)maximum=std::max(maximum,std::abs(expected.data_ptr<float>()[i]));
                        for(int64_t i=0;i<result.numel();++i)require(std::isfinite(result.data_ptr<float>()[i]) &&
                            std::abs(result.data_ptr<float>()[i]-expected.data_ptr<float>()[i])<=
                            maximum*3e-6f+1e-6f,"Compact routed changed original-codebook projection");
                    }
                }
            }
            cudaKernelNodeParams params[2]{};
            for(int mode=0;mode<2;++mode) {
                size_t count=0;TYLOINFER_CUDA_CHECK(cudaGraphGetNodes(graphs[mode].graph,nullptr,&count));
                require(count==size_t(paired && !mode && !shared_lookup && !decoded_input?2:1),"Routed benchmark missed its expected separate/combined dispatch");
                std::vector<cudaGraphNode_t> nodes(count);TYLOINFER_CUDA_CHECK(cudaGraphGetNodes(graphs[mode].graph,nodes.data(),&count));
                TYLOINFER_CUDA_CHECK(cudaGraphKernelNodeGetParams(nodes[0],&params[mode]));
            }
            if((shared_lookup || decoded_input) && params[0].func==params[1].func) {
                std::printf("shared_lookup_not_amortized type=%d M=%d N=%lld K=%lld bitwise=1\n",int(type),tokens,
                    static_cast<long long>(rows),static_cast<long long>(columns));continue;
            }
            require(params[0].func!=params[1].func,"Routed benchmark captured the same dispatch twice");
            for(int i=0;i<512;++i)for(auto& graph:graphs)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(graph.exec,stream));
            cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
            std::vector<float> times[2];
            for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
                const int mode=(trial+order)%2;TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
                for(int i=0;i<512;++i)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(graphs[mode].exec,stream));
                TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
                float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));times[mode].push_back(ms*1000/512);
            }
            for(auto& values:times)std::sort(values.begin(),values.end());
            std::printf(decoded_input?"decoded_input_routed type=%d paired=%d M=%d N=%lld K=%lld experts=%lld old_us=%.6f new_us=%.6f ratio=%.6f actual_match=1 bitwise=1\n":shared_lookup?"shared_trellis_routed type=%d paired=%d M=%d N=%lld K=%lld experts=%lld old_us=%.6f new_us=%.6f ratio=%.6f actual_match=1 bitwise=1\n":"real_trellis_routed type=%d paired=%d M=%d N=%lld K=%lld experts=%lld old_us=%.6f new_us=%.6f ratio=%.6f actual_match=1\n",
                int(type),int(paired),tokens,static_cast<long long>(rows),static_cast<long long>(columns),static_cast<long long>(experts),times[0][3],times[1][3],times[1][3]/times[0][3]);
            std::printf("routed_fixture diversity=%d routes=%lld\n",diversity,static_cast<long long>(route_count));
            TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
        }
    }
}

void run_file_moe_runtime(const std::filesystem::path& path) {
    ScopedEnvironment gpu_only("TYLOINFER_GGUF_CPU_EXPERTS","0");
    mfq::GgufModelSource source(path);
    std::string prefix;
    for(const auto& tensor:source.tensors()) {
        constexpr std::string_view suffix="mlp.experts.gate.weight";
        if(tensor.name.ends_with(suffix)){prefix=tensor.name.substr(0,tensor.name.size()-suffix.size());break;}
    }
    require(!prefix.empty(),"GGUF file has no canonical expert bank");
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=16<<20;execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,prefix+"mlp.experts.gate.weight"),
        std::make_shared<GgufRoutedWeight>(execution,source,prefix+"mlp.experts.up.weight"),
        std::make_shared<GgufRoutedWeight>(execution,source,prefix+"mlp.experts.down.weight")};
    const auto route_count=std::min(int64_t(3),banks.gate->experts),width=banks.gate->input;
    int cases=0;
    for(int rows:{1,2,4,64})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto host=tb::empty({rows,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.173f+.41f)*.1f;
        auto x=host.to(dtype).to(tb::kCUDA);
        auto ids=tb::arange(rows*route_count,x.options().dtype(tb::kInt64)).remainder(banks.gate->experts).reshape({rows,route_count});
        auto weights=tb::full({rows,route_count},1./route_count,x.options().dtype(tb::kFloat32));
        check_moe(banks.forward(x,ids,weights,.2f),banks,x,ids,weights,.2f);++cases;
        if(rows!=4)continue;
        require(prepare_gguf_graph_pool(execution.gguf_expert_cache),"File-backed expert graph pool missing");
        tb::Tensor actual;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{actual=banks.forward(x,ids,weights,.2f);},[]{},[&]{actual={};});
        for(int replay=0;replay<3;++replay) {
            x.copy_(x*-.7f);ids.copy_((ids+1).remainder(banks.gate->experts));graph.run();
            check_moe(actual,banks,x,ids,weights,.2f);++cases;
        }
    }
    std::printf("GGUF file-backed expert source/metadata/rotation/streaming/decode/PP/changing-graph formats=%d,%d,%d cases=%d PASS\n",
        int(banks.gate->type),int(banks.up->type),int(banks.down->type),cases);
    std::fflush(stdout);
}

void run_compact_moe_runtime() {
    int cases = 0;
    for (int format = GGML_TYPE_PTQ1_0; format <= GGML_TYPE_TQK4 + 1; ++format) {
        Source source; source.enable_batches();
        const auto gate_type = ggml_type(format <= GGML_TYPE_TQK4 ? format : GGML_TYPE_PTQ1_0);
        const auto up_type = ggml_type(format <= GGML_TYPE_TQK4 ? format : GGML_TYPE_TQK6);
        const auto down_type = ggml_type(format <= GGML_TYPE_TQK4 ? format : GGML_TYPE_TQP6);
        source.add("gate", {3, 256, 256}, gate_type); source.full_compact_codes("gate", gate_type);
        source.add("up", {3, 256, 256}, up_type); source.full_compact_codes("up", up_type);
        source.add("down", {3, 256, 256}, down_type); source.full_compact_codes("down", down_type);
        CudaExecutionContext execution;
        execution.config.moe_host_cache_bytes = 2 * 1024 * 1024;
        execution.config.moe_device_cache_bytes = 0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution, source, "gate"),
            std::make_shared<GgufRoutedWeight>(execution, source, "up"),
            std::make_shared<GgufRoutedWeight>(execution, source, "down")};
        for (int rows : {1, 2, 4}) for (auto dtype : {tb::kFloat16, tb::kFloat32}) {
            auto x = input(rows, 3, false).to(dtype), ids = routes(rows, 0);
            auto weights = tb::full({rows, 3}, .13, x.options().dtype(tb::kFloat32));
            check_moe(banks.forward(x, ids, weights, .2f), banks, x, ids, weights, .2f); ++cases;
        }
        require(prepare_gguf_graph_pool(execution.gguf_expert_cache), "Compact expert graph pool missing");
        auto x = input(4, 3, false).to(tb::kFloat16), ids = routes(4, 0);
        auto weights = tb::full({4, 3}, .13, x.options().dtype(tb::kFloat32));
        tb::Tensor actual;
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&] { actual = banks.forward(x, ids, weights, .2f); }, [] {}, [&] { actual = {}; });
        for (int rotation : {1, 2}) {
            ids.copy_(routes(4, rotation)); weights.fill_(.1 * rotation); graph.run();
            check_moe(actual, banks, x, ids, weights, .2f); ++cases;
        }
    }
    std::cout << "Compact9 MoE streaming/mixed-format/GateUpDown/M1-2-4/FP16-FP32/changing-graph cases=" << cases << " PASS\n";
}

void run_projection_tail_rounding() {
    int cases=0;double largest=0;
    for(int columns:{256,4096})for(auto type:{GGML_TYPE_Q8_0,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ4_XS}) {
        Source source;source.add("query",{33,columns},type);
        auto weight=load_gguf_weight(source,"query",mfq_current_cuda_device());
        for(int rows:{2,3,4})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto host=tb::empty({1,rows,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(i*.17f)*.3f;
            auto x=host.to(tb::kCUDA).to(dtype);
            auto batch=weight.forward(x).narrow(1,rows-1,1).to(tb::kCPU,tb::kFloat32).contiguous();
            auto tail=weight.forward(x.narrow(1,rows-1,1)).to(tb::kCPU,tb::kFloat32).contiguous();
            for(int64_t i=0;i<batch.numel();++i) {
                const auto error=std::abs(double(batch.data_ptr<float>()[i])-tail.data_ptr<float>()[i]);
                const auto relative=dtype==tb::kFloat16?.002:.0001;
                largest=std::max(largest,error);
                require(std::isfinite(error) && error<=relative*(1.+std::abs(batch.data_ptr<float>()[i])),
                    "GGUF tail query exceeds scalar/batched projection rounding allowance");
            }
            ++cases;
        }
    }
    std::cout<<"GGUF M1 tail vs M2..4 complete query projection FP16/FP32 cases="<<cases<<" max_error="<<largest<<" PASS\n";
}
void run_projection_subset_check() {
    run_projection_group();run_projection_group(false);
    std::cout<<"GGUF joined/separate projection images, no-copy subsets and M1..33 outputs PASS\n";
    run_projection_tail_rounding();
}
void run_parallel_projection_group() {
    const auto native=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();native->begin_graph_pool(stream);
    struct EndPool {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;~EndPool(){context->end_graph_pool(stream);}} pool{native,stream};
    Source source;constexpr int width=768;
    source.add("qkv",{17,width},GGML_TYPE_Q6_K);source.add("gate",{31,width},GGML_TYPE_IQ3_S);
    source.add_dense("alpha",{3,width});source.add_dense("beta",{5,width});
    CudaExecutionContext execution;
    std::vector<weight_loader::Linear> projections;
    for(const char* name:{"qkv","gate","alpha","beta"})projections.push_back(weight_loader::linear(execution,source,name));
    auto group=weight_loader::grouped_linear(execution,projections,1);
    require(bool(group),"parallel GGUF/dense projection group missing");
    int checked=0;
    for(int rows:{1,2,3,4})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        std::vector<float> values(rows*width);
        for(std::size_t i=0;i<values.size();++i)values[i]=std::sin(float(i)*.173f)*.1f;
        const auto base=tb::tensor(values).reshape({1,rows,width}).to(tb::kCUDA).to(dtype);
        auto x=base.clone();std::vector<tb::Tensor> output;
        execution.config.decode_branch_parallel=true;
        mfq::cuda::DecodeWindow window(stream);
        window.capture([&]{output=group(execution,x);},[]{},[&]{output.clear();});
        for(float amplitude:{.3f,.8f,1.1f}) {
            x.copy_(base*amplitude);
            execution.config.decode_branch_parallel=false;
            const auto expected=group(execution,x);
            execution.config.decode_branch_parallel=true;
            require(window.enqueue_if_device_only(),"projection branches introduced host graph tasks");
            mfq_cuda_synchronize();
            require(expected.size()==output.size(),"parallel projection changed output boundaries");
            for(std::size_t i=0;i<output.size();++i) {
                const auto a=output[i].contiguous().cpu(),b=expected[i].contiguous().cpu();
                require(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
                    std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"parallel projection changed output bytes");
            }
            ++checked;
        }
        require(execution.projection_executors.size()==1,"projection groups did not share their per-device executor");
    }
    auto executor=execution.projection_executors.at(mfq_current_cuda_device());
    require(executor->streams.size()==3,"batched GGUF matrices and two auxiliaries did not use three shared streams");
    std::weak_ptr<void> lifetime=executor->streams[1].lifetime();
    auto retained=weight_loader::linear_weight(projections[2])->dense;
    group={};projections.clear();executor.reset();execution.reset();
    require(!lifetime.expired(),"projection stream died before an aliased dense weight");
    retained={};
    require(lifetime.expired(),"projection streams leaked after their last weight");
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    std::cout<<"GGUF/dense parallel projection changing graphs, shared streams and aliased teardown cases="<<checked<<" PASS\n";
}
#include "gguf_quad_storage_test.h"
#include "gguf_iq2_fractional_test.h"
#include "gguf_token_vector_test.h"
#include "gguf_q2_affine_test.h"
void run_combined_row_storage() {
    for(const auto type:{
#define TYLOINFER_GGUF_JOIN_FORMAT(name) GGML_TYPE_##name,
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_JOIN_FORMAT)
#undef TYLOINFER_GGUF_JOIN_FORMAT
    }) {
    // Odd row counts also check compressed formats whose next matrix begins
    // at a smaller alignment than a fresh CUDA allocation.
    Source source;source.add("a",{7,256},type);source.add("b",{9,256},type);
    auto a=load_gguf_weight(source,"a",0),b=load_gguf_weight(source,"b",0);
    auto x=input(4,1,false).to(tb::kFloat16);
    const auto expected_output=tb::cat({a.forward(x),b.forward(x)},-1);
    const auto original=tb::cat({a.canonical_packed().cpu(),b.canonical_packed().cpu()},0);
    const auto first_bytes=a.packed.numel();
    auto joined=combine_gguf_rows({&a,&b});
    require(a.packed.data_ptr()==joined.packed.data_ptr() &&
        b.packed.data_ptr()==(joined.quad_storage_rows?joined.packed.data_ptr():joined.packed.data_ptr<std::uint8_t>()+first_bytes) &&
        (!joined.quad_storage_rows || b.quad_row_begin==7),
        "GGUF row combination duplicated its component weight storage");
    const auto actual=joined.canonical_packed().cpu();
    require(actual.numel()==original.numel() && std::memcmp(actual.data_ptr(),original.data_ptr(),original.nbytes())==0,
        "GGUF row combination changed the original compressed bytes");
    require((joined.forward(x)-expected_output).abs().max().item<float>()==0,
        "GGUF joined small-row projection changed activation or weight rounding");
    auto wrong=b;wrong.type=type==GGML_TYPE_Q4_0?GGML_TYPE_Q8_0:GGML_TYPE_Q4_0;
    bool rejected=false;try{(void)combine_gguf_rows({&a,&wrong});}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"GGUF row combination accepted incompatible quantization types");
    }
}
// Independent scalar reference uses canonical GGUF dequantization and the
// actual Q8 activation packet, covering partial output tiles and K warps.
void run_m1_affine_k() {
    namespace gg=mfq::cuda::gguf;
    int cases=0;
    int device=0,processors=0,resident_threads=0;
    require(cudaGetDevice(&device)==cudaSuccess &&
        cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,device)==cudaSuccess &&
        cudaDeviceGetAttribute(&resident_threads,cudaDevAttrMaxThreadsPerMultiProcessor,device)==cudaSuccess,
        "M=1 test could not query resident-row capacity");
    for(auto format:{GGML_TYPE_Q2_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS})
    for(int columns:{256,768,2560,4096,6144,16640})for(int row_case:{0,1,2,3}) {
        // The extra case exceeds the device's possible resident-row grid.
        // It covers standalone IQ4 cooperation when occupancy is already full.
        if(row_case==3 && (format!=GGML_TYPE_IQ4_XS || columns!=4096))continue;
        const int rows=row_case==0?1:row_case==1?7:row_case==2?9:processors*(resident_threads/128)*8+1;
        try {
        Source source;source.add("w",{rows,columns},format);
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
        const auto canonical=load_gguf_weight(source,"w",mfq_current_cuda_device(),false).dequantize_stored().contiguous();
        auto host=tb::empty({1,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int i=0;i<columns;++i)host.data_ptr<float>()[i]=.13f*std::sin(float(i)*.173f);
        auto x=host.to(tb::kCUDA),q=tb::empty({int64_t(gg::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
        gg::quantize_input(x.data_ptr<float>(),1,columns,q.data_ptr(),mfq_current_cuda_stream());
        const auto packed=q.cpu();const auto* activation=static_cast<const block_q8_1*>(packed.data_ptr());
        auto guarded=tb::full({rows+2},-12345.f,x.options()),output=guarded.narrow(0,1,rows);
        gg::matvec_q8(format,weight.packed.data_ptr(),weight.row_bytes,rows,columns,q.data_ptr(),1,output.data_ptr<float>(),mfq_current_cuda_stream());
        const auto actual=guarded.cpu();
        require(actual.data_ptr<float>()[0]==-12345.f && actual.data_ptr<float>()[rows+1]==-12345.f,"M=1 affine K projection overwrote its output boundary");
        for(int r=0;r<rows;++r) {
            double reference=0,magnitude=0;
            for(int k=0;k<columns;++k) {
                float scale;ggml_fp16_to_fp32_row(&activation[k/32].d,&scale,1);
                const double term=double(canonical.data_ptr<float>()[r*columns+k])*scale*activation[k/32].qs[k%32];
                reference+=term;magnitude+=std::abs(term);
            }
            const double value=actual.data_ptr<float>()[r+1];
            require(std::isfinite(value) && std::abs(value-reference)<=2.e-6*magnitude+1.e-7,"M=1 affine K projection disagrees with canonical GGUF and Q8 activations");
        }
        ++cases;
        }catch(const std::exception& error) {
            throw std::runtime_error("M=1 format="+std::to_string(int(format))+" K="+
                std::to_string(columns)+" rows="+std::to_string(rows)+": "+error.what());
        }
    }
    std::cout<<"M=1 affine K/IQ4_XS, canonical CPU reference and output boundaries: "<<cases<<" cases PASS\n";
}





void run_m1_residual() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();int cases=0;
    const ggml_type formats[]={GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,
        GGML_TYPE_Q2_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS};
    for(auto type:formats)for(int columns:{256,768,2560,6144,16640})for(int rows:{1,7,9}) {
        Source source;source.add("w",{rows,columns},type);
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
        const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
        auto host=tb::empty({1,columns},cpu),rh=tb::empty({rows},cpu);
        auto input=tb::empty({1,columns},cpu.device(tb::kCUDA).dtype(tb::kFloat16));
        auto residual=tb::empty({rows},input.options());
        auto activation=tb::empty({int64_t(gg::q8_row_bytes(columns))},input.options().dtype(tb::kUInt8));
        auto projected=tb::empty({rows},input.options());
        auto guarded=tb::full({rows+2},-123.f,input.options()),output=guarded.narrow(0,1,rows);
        for(bool aligned:{false,true}) {
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            gg::matvec_m1_residual(type,weight.packed.data_ptr(),weight.row_bytes,rows,columns,
                activation.data_ptr(),aligned,residual.data_ptr(),output.data_ptr(),stream);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
            for(int step=0;step<3;++step) {
                for(int i=0;i<columns;++i)host.data_ptr<float>()[i]=.13f*std::sin(float(i)*.173f+step);
                for(int r=0;r<rows;++r)rh.data_ptr<float>()[r]=.17f*std::cos(float(r)*.29f-step);
                input.copy_(host.to(tb::kCUDA,tb::kFloat16));residual.copy_(rh.to(tb::kCUDA,tb::kFloat16));
                if(aligned) {
                    gg::quantize_input_m1_aligned(input.data_ptr(),true,columns,activation.data_ptr(),stream);
                    gg::matvec_m1_aligned(type,weight.packed.data_ptr(),weight.row_bytes,rows,columns,
                        activation.data_ptr(),projected.data_ptr(),true,stream);
                }else {
                    gg::quantize_input_f16(input.data_ptr(),1,columns,activation.data_ptr(),stream);
                    gg::matvec_q8_f16(type,weight.packed.data_ptr(),weight.row_bytes,rows,columns,
                        activation.data_ptr(),1,projected.data_ptr(),stream);
                }
                auto expected=acc_cuda(residual,projected).cpu();
                TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(executable,stream));
                auto actual=output.cpu(),guard=guarded.to(tb::kCPU,tb::kFloat32);
                require(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
                    "M=1 residual epilogue changed projection/addition rounding or Graph inputs");
                require(guard.data_ptr<float>()[0]==-123.f && guard.data_ptr<float>()[rows+1]==-123.f,
                    "M=1 residual epilogue overwrote its output boundaries");
                if(step==2) {
                    gg::matvec_m1_residual(type,weight.packed.data_ptr(),weight.row_bytes,rows,columns,
                        activation.data_ptr(),aligned,residual.data_ptr(),residual.data_ptr(),stream);
                    actual=residual.cpu();
                    require(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
                        "M=1 residual epilogue changed an aliased residual/output row");
                }
            }
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphExecDestroy(executable));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphDestroy(graph));++cases;
        }
    }
    std::cout<<"M=1 residual epilogue, changing Graph inputs, FP16 rounding, tails and alias: "<<cases<<" cases PASS\n";
}

void run_aligned_m1() {
    namespace gg=mfq::cuda::gguf;const auto context=mfq::cuda::default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();int packets=0,pairs=0;
    const auto equal_packet=[&](const tb::Tensor& native,const tb::Tensor& aligned,int width){
        auto n=native.cpu(),a=aligned.cpu();const auto* blocks=static_cast<const block_q8_1*>(n.data_ptr());const auto* bytes=a.data_ptr<uint8_t>();
        for(int i=0;i<width;++i)require(blocks[i/32].qs[i%32]==reinterpret_cast<const int8_t*>(bytes)[i],"Aligned M=1 changed Q8 coordinates");
        for(int b=0;b<width/32;++b)require(std::memcmp(&blocks[b].ds,bytes+width+4*b,4)==0,"Aligned M=1 changed half-rounded scale/sum");++packets;
    };
    const auto equal_tensor=[&](const tb::Tensor& a,const tb::Tensor& b){auto x=a.cpu(),y=b.cpu();require(x.nbytes()==y.nbytes()&&std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())==0,"Aligned M=1 producer changed normalized/residual values");};
    for(int width:{256,768,4096,16640})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto h=tb::empty({1,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));for(int j=0;j<width;++j)h.data_ptr<float>()[j]=.13f*std::sin(float(j)*.173f);
        auto x=h.to(tb::kCUDA,dtype);tb::Tensor gate;
        // Populate gates on CPU; avoid relying on scalar CPU tensor kernels.
        auto gate_host=h.clone();for(int j=0;j<width;++j)gate_host.data_ptr<float>()[j]=.3f*std::cos(float(j)*.217f);gate=gate_host.to(tb::kCUDA,dtype);
        auto n=tb::empty({int64_t(gg::q8_row_bytes(width))},x.options().dtype(tb::kUInt8)),a=tb::empty_like(n);
        if(dtype==tb::kFloat16)gg::quantize_input_f16(x.data_ptr(),1,width,n.data_ptr(),stream);else gg::quantize_input(x.data_ptr<float>(),1,width,n.data_ptr(),stream);
        gg::quantize_input_m1_aligned(x.data_ptr(),dtype==tb::kFloat16,width,a.data_ptr(),stream);equal_packet(n,a,width);
        for(bool silu:{false,true}) {gg::quantize_gated_input(x.data_ptr(),gate.data_ptr(),dtype==tb::kFloat16,1,width,silu,n.data_ptr(),stream);gg::quantize_gated_input_m1_aligned(x.data_ptr(),gate.data_ptr(),dtype==tb::kFloat16,1,width,silu,a.data_ptr(),stream);equal_packet(n,a,width);}
        auto weight=tb::ones({width},x.options().dtype(tb::kFloat32)),yn=tb::empty({1,width},weight.options()),ya=tb::empty_like(yn);
        gg::quantize_rms_input(x.data_ptr(),dtype==tb::kFloat16,weight.data_ptr<float>(),1,width,1.e-6f,.25f,yn.data_ptr<float>(),n.data_ptr(),stream);
        gg::quantize_rms_input_m1_aligned(x.data_ptr(),dtype==tb::kFloat16,weight.data_ptr<float>(),1,width,1.e-6f,.25f,ya.data_ptr<float>(),a.data_ptr(),stream);equal_packet(n,a,width);equal_tensor(yn,ya);
        auto residual=x.to(tb::kFloat16),sn=tb::empty_like(residual),sa=tb::empty_like(residual),rn=tb::empty_like(residual),ra=tb::empty_like(residual);
        gg::quantize_acc_rms_input(residual.data_ptr(),gate.data_ptr(),dtype==tb::kFloat32,weight.data_ptr<float>(),1,width,1.e-6f,.25f,sn.data_ptr(),rn.data_ptr(),n.data_ptr(),stream);
        gg::quantize_acc_rms_input_m1_aligned(residual.data_ptr(),gate.data_ptr(),dtype==tb::kFloat32,weight.data_ptr<float>(),1,width,1.e-6f,.25f,sa.data_ptr(),ra.data_ptr(),a.data_ptr(),stream);equal_packet(n,a,width);equal_tensor(sn,sa);equal_tensor(rn,ra);
    }
    const ggml_type formats[]={GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_Q2_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS};
    for(int width:{768,2560,4096,6144,16640}) {
        Source source;std::vector<GgufWeight> wa,wb;
        for(auto format:formats){const auto name=std::to_string(int(format));source.add("a"+name,{7,width},format);source.add("b"+name,{9,width},format);wa.push_back(load_gguf_weight(source,"a"+name,0));wb.push_back(load_gguf_weight(source,"b"+name,0));(void)combine_gguf_rows({&wa.back(),&wb.back()});}
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto h=tb::empty({1,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));for(int j=0;j<width;++j)h.data_ptr<float>()[j]=.13f*std::sin(float(j)*.173f);
            auto x=h.to(tb::kCUDA,dtype),n=tb::empty({int64_t(gg::q8_row_bytes(width))},x.options().dtype(tb::kUInt8)),a=tb::empty_like(n);
            if(dtype==tb::kFloat16)gg::quantize_input_f16(x.data_ptr(),1,width,n.data_ptr(),stream);else gg::quantize_input(x.data_ptr<float>(),1,width,n.data_ptr(),stream);
            gg::quantize_input_m1_aligned(x.data_ptr(),dtype==tb::kFloat16,width,a.data_ptr(),stream);
            std::vector<tb::Tensor> ea,eb;
            const auto reference=[&](const GgufWeight& w){auto y=tb::empty({w.out},x.options());if(dtype==tb::kFloat16)gg::matvec_q8_f16(w.type,w.packed.data_ptr(),w.row_bytes,w.out,width,n.data_ptr(),1,y.data_ptr(),stream);else gg::matvec_q8(w.type,w.packed.data_ptr(),w.row_bytes,w.out,width,n.data_ptr(),1,y.data_ptr<float>(),stream);return y.cpu();};
            for(const auto& w:wa)ea.push_back(reference(w));for(const auto& w:wb)eb.push_back(reference(w));
            auto guarded_a=tb::full({9},-123.,x.options()),guarded_b=tb::full({11},-123.,x.options());auto ya=guarded_a.narrow(0,1,7),yb=guarded_b.narrow(0,1,9);
            for(size_t i=0;i<wa.size();++i)for(size_t j=0;j<wb.size();++j){const gg::DenseProjection p[]={{wa[i].packed.data_ptr(),wa[i].row_bytes,7,ya.data_ptr(),wa[i].type},{wb[j].packed.data_ptr(),wb[j].row_bytes,9,yb.data_ptr(),wb[j].type}};ya.fill_(NAN);yb.fill_(NAN);gg::matvec_group_m1_aligned(p,2,width,a.data_ptr(),dtype==tb::kFloat16,stream);equal_tensor(ya,ea[i]);equal_tensor(yb,eb[j]);auto ga=guarded_a.to(tb::kCPU,tb::kFloat32),gb=guarded_b.to(tb::kCPU,tb::kFloat32);require(ga.data_ptr<float>()[0]==-123&&ga.data_ptr<float>()[8]==-123&&gb.data_ptr<float>()[0]==-123&&gb.data_ptr<float>()[10]==-123,"Aligned M=1 overwrote output boundaries");++pairs;}
            // More than one packet, with empty entries between live members.
            std::vector<tb::Tensor> extra;std::vector<gg::DenseProjection> descriptors;
            for(size_t i=0;i<wa.size();++i){extra.push_back(tb::empty({7},x.options()));descriptors.push_back({wa[i].packed.data_ptr(),wa[i].row_bytes,7,extra.back().data_ptr(),wa[i].type});if(i%3==0)descriptors.push_back({nullptr,0,0,nullptr,wa[i].type});}
            gg::matvec_group_m1_aligned(descriptors.data(),descriptors.size(),width,a.data_ptr(),dtype==tb::kFloat16,stream);for(size_t i=0;i<extra.size();++i)equal_tensor(extra[i],ea[i]);
        }
    }
    std::cout<<"Aligned M=1 native packet bits, fused producers, mixed-format tails and multi-packet groups: "<<packets<<" packets, "<<pairs<<" pairs PASS\n";
}

void run_dense_format_pairs() {
    namespace gg=mfq::cuda::gguf;
    const ggml_type formats[]={GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_Q2_K,
        GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS};
    int cases=0;
    for(int width:{768,2560,4096,6144,16640}) {
        Source source;std::vector<GgufWeight> first,second;
        for(auto type:formats) {
            const auto name=std::to_string(int(type));
            source.add("a"+name,{7,width},type);source.add("b"+name,{9,width},type);
            first.push_back(load_gguf_weight(source,"a"+name,0));second.push_back(load_gguf_weight(source,"b"+name,0));
            // This API consumes canonical GGUF rows. Loading/combining may
            // repack low-bit weights; retain shared-allocation tail coverage
            // without passing that internal layout to the canonical reader.
            auto& a=first.back();auto& b=second.back();
            const auto joined=tb::cat({a.canonical_packed(),b.canonical_packed()},0);
            const auto split=std::int64_t(a.out*a.row_bytes);
            a.packed=joined.narrow(0,0,split);
            b.packed=joined.narrow(0,split,std::int64_t(b.out*b.row_bytes));
            a.quad_storage_rows=b.quad_storage_rows=0;
            a.quad_row_begin=b.quad_row_begin=0;
        }
        for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto host=tb::empty({1,width},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int i=0;i<width;++i)host.data_ptr<float>()[i]=std::sin(float(i)*.173f)*.01f;
            auto x=host.to(tb::kCUDA,dtype),q=tb::empty({int64_t(gg::q8_row_bytes(width))},x.options().dtype(tb::kUInt8));
            if(dtype==tb::kFloat16)gg::quantize_input_f16(x.data_ptr(),1,width,q.data_ptr(),mfq_current_cuda_stream());
            else gg::quantize_input(x.data_ptr<float>(),1,width,q.data_ptr(),mfq_current_cuda_stream());
            std::vector<tb::Tensor> expected_first,expected_second;
            for(const auto& w:first)expected_first.push_back(w.forward(x).to(dtype).cpu());
            for(const auto& w:second)expected_second.push_back(w.forward(x).to(dtype).cpu());
            auto a=tb::empty({1,7},x.options()),b=tb::empty({1,9},x.options());
            for(std::size_t i=0;i<first.size();++i)for(std::size_t j=0;j<second.size();++j) {
                a.fill_(NAN);b.fill_(NAN);
                const auto& wa=first[i];const auto& wb=second[j];
                const gg::DenseProjection projections[]={
                    {wa.packed.data_ptr(),wa.row_bytes,wa.out,a.data_ptr(),wa.type},
                    {wb.packed.data_ptr(),wb.row_bytes,wb.out,b.data_ptr(),wb.type}};
                gg::matvec_group(projections,2,width,q.data_ptr(),nullptr,1,dtype==tb::kFloat16,mfq_current_cuda_stream());
                const auto ah=a.cpu(),bh=b.cpu();
                const auto matches=[&](const tb::Tensor& actual,const tb::Tensor& expected) {
                    if(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0)return true;
                    // Homogeneous underfilled Q6 groups deliberately use more
                    // K warps. Check their changed FP32 addition tree within
                    // FP32 error / one final FP16 ULP; other pairs stay exact.
                    if(wa.type!=GGML_TYPE_Q6_K || wb.type!=GGML_TYPE_Q6_K)return false;
                    const auto av=actual.to(tb::kFloat32),ev=expected.to(tb::kFloat32);
                    float scale=0,error=0;
                    for(int64_t r=0;r<av.numel();++r) {
                        const float value=av.data_ptr<float>()[r],reference=ev.data_ptr<float>()[r];
                        if(!std::isfinite(value) || !std::isfinite(reference))return false;
                        scale=std::max(scale,std::abs(reference));
                        error=std::max(error,std::abs(value-reference));
                    }
                    return error<=scale*(dtype==tb::kFloat16?1.f/1024:2.e-6f)+1.e-7f;
                };
                require(matches(ah,expected_first[i]) && matches(bh,expected_second[j]),
                    ("GGUF format-pair kernel changed projection bits/boundaries: K="+std::to_string(width)+
                        " formats="+std::to_string(int(wa.type))+"/"+std::to_string(int(wb.type))+
                        " half="+std::to_string(dtype==tb::kFloat16)).c_str());
                ++cases;
            }
        }
    }
    std::cout<<"GGUF format pairs, FP16/FP32, short/wide/long K and shared-allocation tails: "<<cases<<" cases PASS (Q6 homogeneous reductions tolerate roundoff)\n";
}

void run_long_dense_pairs() {
    namespace gg=mfq::cuda::gguf;
    int device=0,processors=0,resident_threads=0;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetDevice(&device));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,device));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&resident_threads,cudaDevAttrMaxThreadsPerMultiProcessor,device));
    const auto context=mfq::cuda::default_context(device);
    // Exceed the possible resident grid to exercise the ordinary typed pair,
    // independent of the test device's model or occupancy characteristics.
    const int rows=processors*(resident_threads/128)*8+1,columns=4096;
    const ggml_type formats[]={GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,
        GGML_TYPE_Q2_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS};
    int cases=0;double peak_relative=0;
    for(size_t i=0;i<std::size(formats);++i) {
        Source source;source.add("a",{rows,columns},formats[i]);
        source.add("b",{rows-2,columns},formats[(i+1)%std::size(formats)]);
        auto a=load_gguf_weight(source,"a",device),b=load_gguf_weight(source,"b",device);
        auto host=tb::empty({columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        auto x=host.to(tb::kCUDA),q=tb::empty({int64_t(gg::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8)),aligned=tb::empty_like(q);
        auto ra=tb::empty({a.out},x.options()),rb=tb::empty({b.out},x.options());
        for(auto dtype:{tb::kFloat32,tb::kFloat16}) {
            std::vector<tb::Tensor> guarded,output;
            for(int layout=0;layout<2;++layout)for(int j=0;j<2;++j) {
                const auto count=j?b.out:a.out;
                guarded.push_back(tb::full({count+2},-123.f,x.options().dtype(dtype)));
                output.push_back(guarded.back().narrow(0,1,count));
            }
            const auto stream_owner=mfq_get_current_cuda_stream(device);
            const auto stream=stream_owner.stream();
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
            for(int layout=0;layout<2;++layout) {
                const gg::DenseProjection p[]={{a.packed.data_ptr(),a.row_bytes,a.out,output[2*layout].data_ptr(),a.type},
                    {b.packed.data_ptr(),b.row_bytes,b.out,output[2*layout+1].data_ptr(),b.type}};
                if(layout)gg::matvec_group_m1_aligned(p,2,columns,aligned.data_ptr(),dtype==tb::kFloat16,stream);
                else gg::matvec_group(p,2,columns,q.data_ptr(),nullptr,1,dtype==tb::kFloat16,stream);
            }
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
            for(int step=0;step<2;++step) {
                for(int k=0;k<columns;++k)host.data_ptr<float>()[k]=.13f*std::sin(float(k)*.173f+step);
                x.copy_(host.to(tb::kCUDA));
                gg::quantize_input(x.data_ptr<float>(),1,columns,q.data_ptr(),stream);
                gg::quantize_input_m1_aligned(x.data_ptr(),false,columns,aligned.data_ptr(),stream);
                gg::matvec_q8(a.type,a.packed.data_ptr(),a.row_bytes,a.out,columns,q.data_ptr(),1,ra.data_ptr<float>(),stream);
                gg::matvec_q8(b.type,b.packed.data_ptr(),b.row_bytes,b.out,columns,q.data_ptr(),1,rb.data_ptr<float>(),stream);
                for(auto& value:output)value.fill_(NAN);
                TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(executable,stream));
                const tb::Tensor refs[]={ra.to(dtype).to(tb::kFloat32).cpu(),rb.to(dtype).to(tb::kFloat32).cpu()};
                for(size_t j=0;j<output.size();++j) {
                    const auto actual=output[j].to(tb::kFloat32).cpu(),guard=guarded[j].to(tb::kFloat32).cpu();
                    require(guard.data_ptr<float>()[0]==-123.f && guard.data_ptr<float>()[guard.numel()-1]==-123.f,"Long dense pair overwrote output guards");
                    double error=0,norm=0;
                    for(int64_t r=0;r<actual.numel();++r) {
                        const float ref=refs[j%2].data_ptr<float>()[r],value=actual.data_ptr<float>()[r];
                        require(std::isfinite(value),"Long dense pair produced nonfinite output");
                        const double delta=double(value)-ref;error+=delta*delta;norm+=double(ref)*ref;
                        if(dtype==tb::kFloat16) {
                            // A reassociated FP32 sum can cross one FP16 rounding
                            // boundary. Permit at most one adjacent FP16 value.
                            const float ulp=std::ldexp(1.f,std::max(-24,ref==0.f?-24:std::ilogb(std::abs(ref))-10));
                            require(std::abs(delta)<=ulp,"Long dense pair exceeded one FP16 rounding step");
                        }
                    }
                    if(dtype==tb::kFloat32) {
                        const double relative=std::sqrt(error/std::max(norm,1.e-30));
                        require(relative<=2.e-6,"Long dense pair changed projection beyond FP32 reduction error");
                        peak_relative=std::max(peak_relative,relative);
                    }
                    ++cases;
                }
            }
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphExecDestroy(executable));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    std::cout<<"Long dense pairs, native/aligned Q8, changing Graph inputs, FP32/FP16 and odd-row guards: "<<cases<<" cases max_FP32_relative_L2="<<peak_relative<<" PASS\n";
}

void run_iq4_group_consistency() {
    int cases=0;
    for(int columns:{256,768,4096,6144})for(auto peer:{GGML_TYPE_IQ4_XS,GGML_TYPE_IQ2_XXS}) {
        Source source;source.add("a",{17,columns},GGML_TYPE_IQ4_XS);source.add("b",{9,columns},peer);
        std::vector<GgufWeight> weights{load_gguf_weight(source,"a",mfq_current_cuda_device()),
            load_gguf_weight(source,"b",mfq_current_cuda_device())};
        for(int tokens:{1,2,3,4,8})for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16}) {
            auto x=tb::empty({1,tokens,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
            for(int64_t i=0;i<x.numel();++i)x.data_ptr<float>()[i]=.37f*std::sin(float(i)*.173f);
            x=x.to(tb::kCUDA).to(dtype);
            const auto actual=project_gguf_group(weights,x);
            const auto expected=weights[0].forward(x);
            require(actual.size()==2 && actual[0].sizes()==expected.sizes() && actual[0].scalar_type()==expected.scalar_type(),
                "IQ4_XS group lost shape/dtype");
            require((actual[0].to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>()==0,
                ("IQ4_XS native/mixed-quad group changed reduction K="+std::to_string(columns)+
                " peer="+std::to_string(int(peer))+" M="+std::to_string(tokens)+" dtype="+std::to_string(int(dtype))).c_str());++cases;
        }
    }
    std::cout<<"GGUF IQ4_XS native/mixed-quad group consistency cases="<<cases<<" PASS\n";
}
void run_shared_quantization(int columns=256) {
    // Empty descriptors occur between real members of a batched group;
    // accepting them must not require backing storage or activation packets.
    for(int tokens:{1,2,3,4}) {
        const mfq::cuda::gguf::DenseProjection empty{nullptr,0,0,nullptr,GGML_TYPE_Q8_0};
        mfq::cuda::gguf::matvec_group(&empty,1,columns,nullptr,nullptr,tokens,false,mfq_current_cuda_stream());
        auto invalid=empty;invalid.rows=1;
        bool rejected=false;
        try {mfq::cuda::gguf::matvec_group(&invalid,1,columns,nullptr,nullptr,tokens,false,mfq_current_cuda_stream());}
        catch(const std::invalid_argument&){rejected=true;}
        require(rejected,"GGUF nonempty group accepted absent weight/output storage");
    }
    Source source;
    std::vector<GgufWeight> weights;
    for(const auto type:{
#define TYLOINFER_GGUF_SHARED_FORMAT(name) GGML_TYPE_##name,
        TYLOINFER_GGUF_FORMATS(TYLOINFER_GGUF_SHARED_FORMAT)
#undef TYLOINFER_GGUF_SHARED_FORMAT
    }) {
        const auto name=std::string("shared_")+std::to_string(int(type));
        source.add(name,{7+int(weights.size())%3,columns},type);
        weights.push_back(load_gguf_weight(source,name,0));
    }
    for(int rows:{0,1,2,3,4,8,32})for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16}) {
        auto host=tb::empty({1,rows,columns},tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU));
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.1f*std::sin(float(i)*.173f);
        auto x=host.to(tb::kCUDA).to(dtype);
        const auto actual=project_gguf_group(weights,x);
        require(actual.size()==weights.size(),"GGUF shared activation lost a mixed-format projection");
        for(std::size_t i=0;i<weights.size();++i) {
            const auto expected=weights[i].forward(x);
            require(actual[i].scalar_type()==expected.scalar_type() && actual[i].sizes().vec()==expected.sizes().vec(),
                "GGUF shared activation changed projection shape or precision");
            if(rows)require((actual[i].to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>()==0,
                ("GGUF shared activation changed mixed-format output bits format="+std::to_string(int(weights[i].type))+
                " M="+std::to_string(rows)+" K="+std::to_string(columns)+" dtype="+std::to_string(int(dtype))).c_str());
        }
    }
    std::cout<<"GGUF mixed-format batched/shared projections K="<<columns<<" zero/tail rows, FP16/32/BF16 PASS\n";
}

void run_dense_projection_overlap() {
    const auto parent=mfq_get_current_cuda_stream();
    const auto context=mfq::cuda::default_context(parent.device_index());
    struct Pool {
        std::shared_ptr<mfq::cuda::Context> context;
        cudaStream_t stream;
        bool owned;
        Pool(std::shared_ptr<mfq::cuda::Context> c,cudaStream_t s):context(std::move(c)),
            stream(s),owned(!context->has_graph_pool(stream)) {
            if(owned)context->begin_graph_pool(stream);
        }
        ~Pool(){if(owned)context->end_graph_pool(stream);}
    } pool(context,parent.stream());
    const auto host=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    int cases=0;
    for(const auto [rows,columns]:{std::pair{17,256},std::pair{33,513}}) {
        auto weight=tb::empty({rows,columns},host);
        for(int64_t i=0;i<weight.numel();++i)weight.data_ptr<float>()[i]=.1f*std::sin(float(i)*.07f);
        DenseLinearGroup group;group.w=weight.to(tb::kCUDA);group.outs={rows/2,rows-rows/2};
        CudaIndependentBranchExecutor executor;
        for(int mode:{0,1,2}) {
            const int tokens=mode==2?2:1;
            auto input=tb::empty({tokens,columns},host);
            for(int64_t i=0;i<input.numel();++i)input.data_ptr<float>()[i]=std::cos(float(i)*.11f);
            const auto dtype=mode==1?tb::kFloat16:tb::kFloat32;
            auto base=input.to(tb::kCUDA,dtype);tb::Tensor current,combined;
            std::vector<tb::Tensor> main,side;
            int calls=0;
            const auto body=[&] {
                current=base*.75f;
                // Consecutive layers share the executor and release earlier
                // output views while graph work is still being recorded.
                for(int layer=0;layer<3;++layer) {
                    side=group.forward_parallel(current,[&]{++calls;main=group.forward(current);},executor);
                    combined=tb::cat(main,-1)+tb::cat(side,-1);
                }
            };
            const auto compare=[&] {
                const auto expected=(tb::cat(group.forward(current),-1)*2.f).cpu();
                const auto actual=combined.cpu();
                require(actual.sizes()==expected.sizes() && actual.scalar_type()==expected.scalar_type() &&
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
                    "Independent dense projection changed values or reused a live output");
            };
            body();require(calls==3,"Independent dense callback did not run exactly once per layer");compare();
            for(int window=0;window<2;++window) {
                mfq::cuda::DecodeWindow graph(parent.stream());
                graph.capture(body,[]{},[&]{current={};combined={};main.clear();side.clear();});
                const int captured_calls=calls;
                for(int replay=0;replay<4;++replay) {
                    base.copy_(input.to(tb::kCUDA,dtype)*(.5f+.25f*(window+replay)));
                    graph.run();compare();
                    require(calls==captured_calls,"Dense graph replay called the host projection callback");
                    require(mfq_current_cuda_stream()==parent.stream(),"Dense branch changed its caller's stream");
                    ++cases;
                }
            }
            bool caught=false;
            try {group.forward_parallel(base,[]{throw std::runtime_error("expected independent failure");},executor);}
            catch(const std::runtime_error&){caught=true;}
            require(caught && mfq_current_cuda_stream()==parent.stream(),"Dense branch exception lost its parent stream");
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(parent.stream()));
        }
    }
    std::cout<<"Independent dense projection graph replay, buffer reuse, dtype/batch fallback and exception cases="
        <<cases<<" PASS\n";
}

void run_shared_work_overlap(bool parallel=false,bool resident_only=false) {
    ScopedEnvironment cpu("TYLOINFER_GGUF_CPU_EXPERTS","0");
    int cases=0,forked=0;
    for(int mode:resident_only?std::vector<int>{3}:std::vector<int>{0,1,2,3}) {
        ScopedEnvironment overlap("TYLOINFER_GGUF_RESIDENT_OVERLAP",mode==1?"0":"1");
        const std::string prefix=mode==3?"predictor.block.0.mlp.experts.":"target.";
        Source source;
        source.add(prefix+"gate.weight",{7,256,256},GGML_TYPE_Q4_0);
        source.add(prefix+"up.weight",{7,256,256},GGML_TYPE_Q5_0);
        source.add(prefix+"down.weight",{7,256,256},GGML_TYPE_Q8_0);
        source.add("shared",{256,256},GGML_TYPE_Q8_0);
        CudaExecutionContext execution;
        execution.config.moe_host_cache_bytes=mode==2?1:4ULL<<20;
        execution.config.moe_device_cache_bytes=256*256;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,prefix+"gate.weight"),
            std::make_shared<GgufRoutedWeight>(execution,source,prefix+"up.weight"),
            std::make_shared<GgufRoutedWeight>(execution,source,prefix+"down.weight")};
        if(resident_only)require(banks.gate->resident_packed.defined() && banks.up->resident_packed.defined() &&
            banks.down->resident_packed.defined(),"Shared-stream resident check unexpectedly needs host expert I/O");
        auto shared_weight=load_gguf_weight(source,"shared",mfq_current_cuda_device());
        const auto cache=execution.gguf_expert_cache;
        prepare_gguf_graph_pool(cache);
        const auto parent_stream=mfq_get_current_cuda_stream();
        CudaIndependentBranchExecutor branch;
        if(parallel)require(branch.ensure(1,parent_stream),"GGUF shared branch eager preparation failed");
        for(int tokens:{1,2,3,4})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto x=input(tokens,3,false).to(dtype),ids=routes(tokens,0);
            tb::Tensor current_input=x;
            auto weights=tb::full({tokens,3},.3,x.options().dtype(tb::kFloat32));
            tb::Tensor shared;
            int calls=0;
            const auto work=[&]{++calls;
                forked+=mfq_current_cuda_stream()!=parent_stream.stream();
                shared=shared_weight.forward(current_input,false);mfq_cuda_record_stream(shared,parent_stream);};
            auto routed=banks.forward(x,ids,weights,0.f,work,parallel?&branch:nullptr);
            require(calls==1,"GGUF independent work was not executed exactly once in eager mode");
            check_moe(routed,banks,x,ids,weights);
            const auto compare_shared=[&] {
                const auto expected=shared_weight.forward(x,false).cpu(),actual=shared.cpu();
                require(actual.sizes()==expected.sizes() && actual.scalar_type()==expected.scalar_type() &&
                    std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,
                    "GGUF independent shared projection changed output bits");
            };
            compare_shared();++cases;
            tb::Tensor output,combined;
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&] {
                // The model produces this activation within the graph. Its
                // previous warmup allocation has a side-stream consumer.
                current_input=x*1.f;
                // Successive layers reuse one ready/completed event pair and
                // one branch graph pool after joining the preceding layer.
                for(int step=0;step<(resident_only?3:1);++step) {
                    output=banks.forward(current_input,ids,weights,0.f,work,parallel?&branch:nullptr);
                    combined=output+shared;
                }
            },[]{},[&]{output={};combined={};shared={};current_input=x;});
            const auto captured_calls=calls;
            if(resident_only)require(graph.can_enqueue_device_only(),"Shared-stream resident graph unexpectedly has a host task");
            for(int replay=0;replay<3;++replay) {
                x.copy_((input(tokens,3,false)*(.5f+.25f*replay)).to(dtype));
                ids.copy_((tb::arange(tokens*3,ids.options())+replay*2).remainder(7).reshape({tokens,3}));
                weights.fill_(.1+.1*replay);
                if(mode!=3)resize_gguf_expert_cache(cache,replay==1?0:256*256);
                graph.run();
                require(calls==captured_calls,"GGUF graph replay invoked independent work on the host");
                check_moe(output,banks,x,ids,weights);compare_shared();
                const auto expected=(output+shared).cpu(),actual=combined.cpu();
                require(std::memcmp(expected.data_ptr(),actual.data_ptr(),actual.nbytes())==0,
                    "GGUF combined routed/shared addition changed during graph replay");
                ++cases;
            }
        }
    }
    if(parallel)require(forked>0,"GGUF shared-stream test never ran work on its side stream");
    std::cout<<"GGUF independent shared work eager/graph replay M1..4 f16/f32 resident/streamed/pageable/serial cache eviction/refill cases="
        <<cases<<" parallel="<<parallel<<" forked="<<forked<<" PASS\n"<<std::flush;
}

void run_mapped_flag_wait() {
    const auto owner=mfq_get_stream_from_pool();
    const auto stream=owner.stream();
    mfq::cuda::HostBuffer buffer(256,true);
    auto* host=static_cast<std::uint32_t*>(buffer.data());
    std::uint32_t* alias=nullptr;
    TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&alias),host,0));
    auto output=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
    *host=1;mfq::cuda::wait_mapped_flag(alias,stream);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    mfq::cuda::wait_mapped_flag(alias,stream);
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(),alias+16,sizeof(std::uint32_t),cudaMemcpyDeviceToDevice,stream));
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    size_t count=0;TYLOINFER_CUDA_CHECK(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);TYLOINFER_CUDA_CHECK(cudaGraphGetNodes(graph,nodes.data(),&count));
    bool memop=false;
    for(auto node:nodes) {
        CUgraphNodeType type;
        if(cuGraphNodeGetType(reinterpret_cast<CUgraphNode>(node),&type)==CUDA_SUCCESS)
            memop|=type==CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
    }
    for(int i=0;i<48;++i) {
        host[0]=0;host[16]=~std::uint32_t(0);
        TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        // Match the serving loop's progress query: asynchronous driver command
        // buffering must be submitted before the host waits on mapped data.
        const auto progress=cudaStreamQuery(stream);
        if(progress!=cudaErrorNotReady)TYLOINFER_CUDA_CHECK(progress);
        // Publish after submission, so a missing wait reads the sentinel.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        host[16]=std::uint32_t(i*17+3);
        const std::uint32_t ready=i%3==0?1u:i%3==1?2u:0x80000000u;
        std::atomic_ref<std::uint32_t>(host[0]).store(ready,std::memory_order_release);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        require(output.cpu().data_ptr<std::int32_t>()[0]==std::int32_t(host[16]),"mapped wait passed before data publication");
    }
    TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
    std::cout<<"GGUF mapped flag graph replay/data visibility/nonzero flags PASS memop="<<memop<<'\n';
}
void run_timed_mapped_flag_wait() {
    const auto owner=mfq_get_stream_from_pool();const auto stream=owner.stream();
    mfq::cuda::HostBuffer buffer(256,true);
    auto* host=static_cast<std::uint32_t*>(buffer.data());
    void* mapped=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host,0));
    auto* flag=static_cast<std::uint32_t*>(mapped);
    auto* clock=reinterpret_cast<std::uint64_t*>(static_cast<std::uint8_t*>(mapped)+32);
    auto* observed=reinterpret_cast<volatile std::uint64_t*>(reinterpret_cast<std::uint8_t*>(host)+32);
    auto* canary=reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uint8_t*>(host)+40);
    constexpr auto sentinel=std::uint64_t(0x97abcdef12345678ULL);
    *canary=sentinel;*host=1;*observed=sentinel;
    mfq::cuda::wait_mapped_flag(flag,stream,clock);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
    require(*observed==0,"ready CPU flag recorded a blocking wait");
    auto output=tb::zeros({1},tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt32));
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    mfq::cuda::wait_mapped_flag(flag,stream,clock);
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(output.data_ptr(),flag+16,sizeof(std::uint32_t),cudaMemcpyDeviceToDevice,stream));
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    for(int i=0;i<12;++i) {
        *host=0;*observed=0;host[16]=~std::uint32_t(0);
        TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        mfq::cuda::check_route_submission(stream);
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(!*observed && std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
        const auto entered=*observed;
        // The GPU publishes entry only after observing ready==0. Release real
        // data from that handshake, without a scheduling-sensitive sleep.
        host[16]=std::uint32_t(i*19+7);
        std::atomic_ref<std::uint32_t>(host[0]).store(i%2?2u:0x80000000u,std::memory_order_release);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        require(entered>0 && *observed>=entered,"CPU wait completion did not use a monotonic GPU clock");
        require(output.cpu().data_ptr<std::int32_t>()[0]==std::int32_t(host[16]),"timed wait read output before CPU publication");
        require(*canary==sentinel,"timed CPU wait overwrote its clock canary");
        *observed=sentinel;
        TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        require(*observed==0,"replayed ready CPU flag retained a stale blocked timestamp");
    }
    TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
    std::cout<<"GGUF timed CPU flag changed-graph/blocked-ready transitions/canaries PASS\n";
}
void run_cpu_auto_balance() {
    if(!mfq::cpu::gguf::available()){std::cout<<"CPU auto balance skipped: required CPU ISA unavailable\n";return;}
    ScopedEnvironment mode("TYLOINFER_GGUF_CPU_EXPERTS","auto");
    ScopedEnvironment minimum("TYLOINFER_GGUF_CPU_EXPERTS_MIN_TOKENS","1");
    GgufCpuExecutor reference(mfq::cpu::physical_cores(true));
    for(int m:{1,2,4}) {
        Source source;source.add("gate",{3,256,256},GGML_TYPE_IQ3_XXS);
        source.add("up",{3,256,256},GGML_TYPE_IQ3_XXS);source.add("down",{3,256,256},GGML_TYPE_Q2_0);
        CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4<<20;execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        auto cache=execution.gguf_expert_cache;
        const auto x=input(m,3,false).to(tb::kFloat16),weights=tb::full({m,3},.3,x.options().dtype(tb::kFloat32));
        std::array<tb::Tensor,3> patterns,lower,upper;
        for(int pattern=0;pattern<3;++pattern) {
            patterns[pattern]=pattern==0?routes(m,0):tb::full({m,3},pattern==1?1:0,x.options().dtype(tb::kInt64));
            const auto ids=patterns[pattern];
            auto g=banks.gate->forward(x,ids),u=banks.up->forward(x,ids);
            const auto hidden=((g*tb::sigmoid(g))*u).to(tb::kFloat16);
            const auto gpu=banks.down->forward(hidden,ids).to(tb::kFloat32).cpu();
            auto cpu=tb::empty_like(gpu);const auto selected=ids.cpu(),raw=x.to(tb::kFloat32).cpu();
            auto work=std::make_shared<GgufCpuWork>();uint32_t ready=0;
            work->tokens=m;work->routes=3;work->input_width=work->hidden_width=work->output_width=256;
            work->input=raw.data_ptr<float>();work->output=cpu.data_ptr<float>();work->ready=&ready;work->half=true;
            for(int e=0;e<3;++e) {
                GgufCpuExpert expert;
                for(int slot=0;slot<m*3;++slot)if(selected.data_ptr<int64_t>()[slot]==e)expert.slots.push_back(slot);
                if(expert.slots.empty())continue;
                int role=0;for(const auto& bank:{banks.gate,banks.up,banks.down}) {
                    auto image=std::make_shared<std::vector<std::byte>>(bank->expert_bytes);
                    bank->reader(e*bank->expert_bytes,image->data(),image->size());expert.owners[role]=image;
                    expert.projections[role++]={reinterpret_cast<const uint8_t*>(image->data()),bank->row_bytes,int(bank->type),256,256};
                }
                work->experts.push_back(std::move(expert));
            }
            reference.submit(work);work->future.get();require(ready==1,"CPU auto reference did not finish");
            lower[pattern]=tb::zeros({m,256},gpu.options());upper[pattern]=tb::zeros_like(lower[pattern]);
            for(int token=0;token<m;++token)for(int route=0;route<3;++route)for(int col=0;col<256;++col) {
                const int at=(token*3+route)*256+col;
                const float a=float(mfq_half(gpu.data_ptr<float>()[at])),b=float(mfq_half(cpu.data_ptr<float>()[at]));
                lower[pattern].data_ptr<float>()[token*256+col]+=.3f*std::min(a,b);
                upper[pattern].data_ptr<float>()[token*256+col]+=.3f*std::max(a,b);
            }
        }
        // One resident expert gives two misses, one miss and no misses in the
        // three patterns. Keep that residency fixed while the cost fit learns.
        resize_gguf_expert_cache(cache,banks.gate->expert_bytes+banks.up->expert_bytes+banks.down->expert_bytes);
        const auto prompt=input(64,1,false).to(tb::kFloat16);
        (void)banks.forward(prompt,tb::zeros({64,1},prompt.options().dtype(tb::kInt64)),tb::ones({64,1},weights.options()));
        require(prepare_gguf_graph_pool(cache),"CPU auto graph pool unavailable");
        auto ids=patterns[0].clone();tb::Tensor output;
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        begin_gguf_speculative_routes(cache,m);graph.capture([&]{output=banks.forward(x,ids,weights);},[]{});discard_gguf_speculative_routes(cache);
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))
            if(name=="gguf_cpu_auto_samples")require(value==0,"CPU balance learned graph preparation instead of replay costs");
        require(set_gguf_reuse_learning(cache,false),"GGUF reuse learning was disabled by default");
        for(int step=0;step<4;++step) {
            begin_gguf_speculative_routes(cache,m);graph.run();commit_gguf_speculative_routes(cache,m);
        }
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))
            if(name=="gguf_cpu_auto_samples" || name=="gguf_speculative_cache_observed_rows")
                require(value==0,"synthetic warmup trained GGUF costs or expert heat");
        require(!set_gguf_reuse_learning(cache,true),"GGUF reuse learning scope did not restore its previous value");
        for(int step=0;step<48;++step) {
            const int pattern=step%3;ids.copy_(patterns[pattern]);
            begin_gguf_speculative_routes(cache,m);graph.run();discard_gguf_speculative_routes(cache);
            const auto actual=output.to(tb::kFloat32).cpu();
            for(int i=0;i<actual.numel();++i) {
                const float lo=lower[pattern].data_ptr<float>()[i],hi=upper[pattern].data_ptr<float>()[i],v=actual.data_ptr<float>()[i];
                const float tolerance=1e-6f+.001f*std::max(std::abs(lo),std::abs(hi));
                require(std::isfinite(v) && v>=lo-tolerance && v<=hi+tolerance,"CPU auto split changed an expert, token or routed output");
            }
        }
        std::map<std::string,double> metrics;for(const auto& [key,value]:gguf_expert_memory_metrics(cache))metrics[key]=value;
        require(metrics.at("gguf_cpu_auto_samples")>=48 && metrics.at("gguf_cpu_auto_selected_experts")>0 &&
            metrics.at("gguf_cpu_auto_gpu_only_layers")>0 && metrics.at("gguf_cpu_auto_fitted_layers")>0,
            "CPU auto test omitted its learned, measured or GPU-only path");
    }
    std::cout<<"GGUF automatic CPU/GPU split changing graphs, 0/1/2 misses and mixed output references PASS\n";
}
void run_cpu_experts(bool rotated=false) {
    namespace cpu=mfq::cpu::gguf;
    if(!cpu::available()) {std::cout<<"CPU GGUF rows skipped: AVX2/FMA/F16C unavailable\n";return;}
    for(const auto type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,
            GGML_TYPE_IQ2_S,GGML_TYPE_IQ4_XS,GGML_TYPE_IQ4_NL,GGML_TYPE_Q2_0})
    for(const int columns:type==GGML_TYPE_Q2_0?std::vector<int>{64,192,256,640}:std::vector<int>{256,768}) {
        Source source;source.add("projection",{17,columns},type);
        const auto decoded=source.decoded("projection");
        const auto* meta=source.find_tensor("projection");std::vector<std::byte> bytes(meta->nbytes);
        source.tensor_reader("projection")(0,bytes.data(),bytes.size());
        const bool iq=cpu::input_supported(type);
        const auto packet=iq?cpu::input_bytes(columns):cpu::hidden_bytes(columns,type);
        std::vector<std::uint8_t> quantized(4*packet);std::vector<float> x(4*columns),qx(4*columns),out(4*17);
        const void* act[4];float* result[4];
        for(int t=0;t<4;++t) {
            for(int k=0;k<columns;++k)x[t*columns+k]=.31f*std::sin(float(k*13+t*23)*.017f);
            act[t]=quantized.data()+t*packet;result[t]=out.data()+t*17;
            if(iq){cpu::quantize_input(x.data()+t*columns,quantized.data()+t*packet,columns);
                dequantize_row_q8_K((const block_q8_K*)act[t],qx.data()+t*columns,columns);
            }else{cpu::quantize_hidden(x.data()+t*columns,quantized.data()+t*packet,columns,type);
                dequantize_row_q8_0((const block_q8_0*)act[t],qx.data()+t*columns,columns);}
        }
        const cpu::Projection p{reinterpret_cast<const std::uint8_t*>(bytes.data()),mfq::gguf::row_bytes(type,columns),int(type),17,columns};
        for(int m=1;m<=4;++m) {
            cpu::project(p,act,m,result,0,17,{false,false,false});
            const auto plain=out;
            // Unsupported VNNI instructions must fall back; supported VEX or
            // EVEX256 rows must preserve the AVX2 accumulation bits.
            cpu::project(p,act,m,result,0,17,{false,true,true});
            require(std::memcmp(plain.data(),out.data(),m*17*sizeof(float))==0,
                "CPU VNNI changed AVX2 projection bits or output tails");
            for(int t=0;t<m;++t)for(int r=0;r<17;++r) {
                double dot=0,mag=0;
                for(int k=0;k<columns;++k){const double v=double(decoded.data_ptr<float>()[r*columns+k])*qx[t*columns+k];dot+=v;mag+=std::abs(v);}
                require(std::abs(dot-result[t][r])<2e-6*mag+1e-6,"CPU GGUF row differs from decoded quantized-activation reference");
            }
        }
    }
    ScopedEnvironment all_rows("TYLOINFER_GGUF_CPU_EXPERTS_MIN_TOKENS","1");
    GgufCpuExecutor reference_executor(mfq::cpu::physical_cores(true));
    for(const int quota:{1,3})for(const auto down_type:{GGML_TYPE_IQ4_NL,GGML_TYPE_Q2_0}) {
        ScopedEnvironment environment("TYLOINFER_GGUF_CPU_EXPERTS",std::to_string(quota).c_str());
        Source source;source.add("gate",{3,256,256},GGML_TYPE_IQ2_XS);
        source.add("up",{3,256,256},GGML_TYPE_IQ3_XXS);source.add("down",{3,256,256},down_type);
        if(rotated){source.add_hadamard("gate",128);source.add_hadamard("up",128,false,2);source.add_hadamard("down",128,false,1);}
        CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4<<20;execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        require(prepare_gguf_graph_pool(execution.gguf_expert_cache),"CPU expert graph pool missing");
        for(int m:{1,2,3,4})for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float limit:{0.f,.01f}) {
            const auto x=input(m,3,false).to(dtype);auto ids=routes(m,0);
            if(quota==3 && m>1) {
                // Different experts see different token subsets, and one
                // token repeats an expert in two slots of the same window.
                auto sparse=ids.cpu();
                sparse.data_ptr<std::int64_t>()[(m-1)*3]=0;
                sparse.data_ptr<std::int64_t>()[(m-1)*3+1]=0;
                sparse.data_ptr<std::int64_t>()[(m-1)*3+2]=2;
                ids.copy_(sparse);
            }
            const auto weights=tb::full({m,3},.3,x.options().dtype(tb::kFloat32));
            auto verify=[&](const tb::Tensor& actual,int first) {
                auto g=banks.gate->forward(x,ids),u=banks.up->forward(x,ids);
                if(limit>0){g=tb::clamp_max(g.to(tb::kFloat32),limit);u=tb::clamp(u.to(tb::kFloat32),-limit,limit);}
                auto hidden=((g*tb::sigmoid(g))*u).to(dtype);
                auto expected_pairs=banks.down->forward(hidden,ids).to(tb::kFloat32).cpu();
                const auto selected=ids.cpu();
                const auto raw=(banks.gate->rotation?banks.gate->rotation->apply(x):x).to(tb::kFloat32).cpu();
                const auto raw_up=(banks.up->rotation?banks.up->rotation->apply(x):x).to(tb::kFloat32).cpu();
                auto work=std::make_shared<GgufCpuWork>();std::uint32_t ready=0;
                work->tokens=m;work->routes=3;work->input_width=work->hidden_width=work->output_width=256;
                work->input=raw.data_ptr<float>();work->output=expected_pairs.data_ptr<float>();work->ready=&ready;
                work->half=dtype==tb::kFloat16;work->limit=limit;
                work->up_input=rotated?raw_up.data_ptr<float>():nullptr;work->down_rotation=banks.down->rotation;
                for(int e=0;e<3;++e) {
                    if(quota==1 && e!=first)continue;
                    GgufCpuExpert expert;
                    int role=0;for(const auto& bank:{banks.gate,banks.up,banks.down}) {
                        auto image=std::make_shared<std::vector<std::byte>>(bank->expert_bytes);
                        bank->reader(e*bank->expert_bytes,image->data(),image->size());expert.owners[role]=image;
                        expert.projections[role++]={reinterpret_cast<const std::uint8_t*>(image->data()),bank->row_bytes,int(bank->type),256,256};
                    }
                    for(int slot=0;slot<m*3;++slot)if(selected.data_ptr<std::int64_t>()[slot]==e)expert.slots.push_back(slot);
                    work->experts.push_back(std::move(expert));
                }
                // Keep owners independently of the executor's retirement and
                // compare one batched dispatch with separate expert jobs.
                const auto single_experts=work->experts;
                reference_executor.submit(work);work->future.get();require(ready==1,"CPU reference did not publish completion");
                if(quota==3) {
                    auto separate=tb::empty_like(expected_pairs);
                    for(const auto& expert:single_experts) {
                        auto one=std::make_shared<GgufCpuWork>();std::uint32_t one_ready=0;
                        one->tokens=m;one->routes=3;one->input_width=one->hidden_width=one->output_width=256;
                        one->input=raw.data_ptr<float>();one->output=separate.data_ptr<float>();one->ready=&one_ready;
                        one->half=dtype==tb::kFloat16;one->limit=limit;one->experts={expert};
                        one->up_input=rotated?raw_up.data_ptr<float>():nullptr;one->down_rotation=banks.down->rotation;
                        reference_executor.submit(one);one->future.get();
                        require(one_ready==1,"Single CPU expert did not publish completion");
                    }
                    require(std::memcmp(separate.data_ptr(),expected_pairs.data_ptr(),expected_pairs.nbytes())==0,
                        "Batched CPU experts changed separate expert output bytes");
                }
                const auto expected=moe_weighted_reduce_cuda(expected_pairs.to(dtype).to(tb::kCUDA),weights);
                const auto error=(actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>();
                require(error<1e-6+expected.abs().max().item<float>()*1e-5,"CPU/GPU expert merge changed routed FFN reference");
            };
            verify(banks.forward(x,ids,weights,limit),0);
            tb::Tensor actual;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{actual=banks.forward(x,ids,weights,limit);},[]{});
            for(int rotation:{1,2}){ids.copy_(routes(m,rotation));graph.run();verify(actual,rotation);}
            ids.fill_(99);bool rejected=false;try{graph.run();}catch(const std::out_of_range&){rejected=true;}
            require(rejected,"CPU expert graph accepted an invalid id");
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
        }
        std::map<std::string,double> metrics;for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))metrics[key]=value;
        require(metrics.at("gguf_cpu_experts")>0,"CPU offload was not actually dispatched");
        if(quota==3)require(metrics.at("gguf_direct_expert_bytes")==0,"CPU-offloaded experts were still copied to GPU");
    }
    {
        ScopedEnvironment quota("TYLOINFER_GGUF_CPU_EXPERTS","1");
        ScopedEnvironment minimum("TYLOINFER_GGUF_CPU_EXPERTS_MIN_TOKENS","2");
        Source source;source.add("gate",{3,256,256},GGML_TYPE_IQ3_XXS);
        source.add("up",{3,256,256},GGML_TYPE_IQ3_XXS);source.add("down",{3,256,256},GGML_TYPE_Q2_0);
        CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4<<20;
        execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        require(prepare_gguf_graph_pool(execution.gguf_expert_cache),"CPU threshold graph pool missing");
        const auto count=[&]{for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
            if(key=="gguf_cpu_experts")return value;return 0.;};
        for(int m:{1,2}) {
            const auto x=input(m,3,false).to(tb::kFloat16),ids=routes(m,0);
            const auto weights=tb::full({m,3},.3,x.options().dtype(tb::kFloat32));
            tb::Tensor actual;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{actual=banks.forward(x,ids,weights);},[]{});
            // The next replay retires and accounts for the prior CPU job.
            const auto before=count();graph.run();graph.run();
            require((count()>before)==(m==2),"Explicit CPU row threshold was not respected");
        }
    }
    auto work=std::make_shared<GgufCpuWork>();std::uint32_t ready=0;float dummy=0;
    work->ready=&ready;work->input=&dummy;work->output=&dummy;reference_executor.submit(work);
    bool failed=false;try{work->future.get();}catch(const std::invalid_argument&){failed=true;}
    require(failed && ready==1,"CPU expert failure stranded its GPU waiter");
}
void run_direct_chain() {
    Source source;
    source.add("gate",{3,256,256},GGML_TYPE_Q4_0);
    source.add("up",{3,256,256},GGML_TYPE_Q5_0);
    source.add("down",{3,256,256},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=2<<20;
    const auto budget=(source.find_tensor("gate")->nbytes+source.find_tensor("up")->nbytes+
        source.find_tensor("down")->nbytes)/3;
    execution.config.moe_device_cache_bytes=budget;
    GgufMoeBanks first{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    GgufMoeBanks second=first;
    finalize_gguf_expert_cache(execution.gguf_expert_cache);
    require(prepare_gguf_graph_pool(execution.gguf_expert_cache),"GGUF chained graph pool missing");
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto x=input(2,1,false),a=tb::zeros({2,1},options.dtype(tb::kInt64)),b=a.clone();
    auto weights=tb::full({2,1},.3,options);
    tb::Tensor actual;
    mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{actual=second.forward(first.forward(x,a,weights),b,weights);},[]{});
    for(int expert:{0,1,1,2,2}) {
        a.fill_(expert);b.fill_(expert);graph.run();
        const auto expected=second.forward(first.forward(x,a,weights),b,weights);
        require((actual-expected).abs().max().item<float>()<1e-5f,"GGUF graph shared a transfer workspace before its consumer finished");
    }
    bool direct=false;double resident=0,evictions=0;
    for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache)) {
        if(key=="resident_expert_cache_bytes")resident=value;
        if(key=="expert_cache_evictions")evictions=value;
        if(key=="gguf_direct_expert_calls")direct=value>0;
    }
    require(resident<=budget,"GGUF direct admission exceeded the expert budget");
    if(direct)require(resident>0 && evictions>0,"GGUF direct graph did not adapt its resident expert set");
    b.fill_(99);bool rejected=false;
    try{graph.run();}catch(const std::out_of_range&){rejected=true;}
    require(rejected,"GGUF chained graph accepted an invalid second-layer route");
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(mfq_current_cuda_stream()));
}
void run_multi_packet() {
    Source source;
    source.add("gate",{7,768,2560});source.add("up",{7,768,2560});source.add("down",{7,2560,768});
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=64ULL<<20;execution.config.moe_device_cache_bytes=2ULL<<20;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    auto x=tb::empty({1,2560},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for(int k=0;k<2560;++k)x.data_ptr<float>()[k]=std::sin(k*.17f)*.1f;
    x=x.to(tb::kCUDA);
    auto ids=tb::arange(7,x.options().dtype(tb::kInt64)).reshape({1,7});
    const auto weights=tb::full({1,7},1./7,x.options());
    check_moe(banks.forward(x,ids,weights),banks,x,ids,weights);
    require(prepare_gguf_graph_pool(execution.gguf_expert_cache),"GGUF large graph pool missing");
    tb::Tensor output;
    mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,weights);},[]{});
    restore_gguf_decode_workspace(execution.gguf_expert_cache);
    require(gguf_expert_cache_budget_bytes(execution.gguf_expert_cache)==2ULL<<20,
        "GGUF workspace restoration exceeded the configured expert budget");
    graph.run();check_moe(output,banks,x,ids,weights);
}
void run_prompt_admission() {
    Source source;
    for(const auto* name:{"gate","up","down"}) source.add(name,{3,256,256});
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=2ULL<<20;execution.config.moe_device_cache_bytes=3*256*144;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    auto ids=tb::empty({64,3},tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64));
    for(int t=0;t<64;++t) {
        ids.data_ptr<int64_t>()[t*3]=ids.data_ptr<int64_t>()[t*3+1]=1;
        ids.data_ptr<int64_t>()[t*3+2]=t%2?2:0;
    }
    auto x=input(64,3,false).to(tb::kFloat16);
    auto weights=tb::full({64,3},1./3,x.options().dtype(tb::kFloat32));
    (void)banks.forward(x,ids.to(tb::kCUDA),weights).cpu();
    const auto before=gguf_expert_streamed_bytes(execution.gguf_expert_cache);
    const auto decode_ids=tb::full({1,3},1,ids.options().device(tb::kCUDA));
    (void)banks.forward(x.narrow(0,0,1),decode_ids,weights.narrow(0,0,1)).cpu();
    require(gguf_expert_streamed_bytes(execution.gguf_expert_cache)==before,
        "GGUF prompt discarded its most-used expert before decode");
}
void run_speculative_admission() {
    Source source;
    for(const auto* name:{"gate","up","down"})source.add(name,{3,256,256});
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes=2ULL<<20;execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto bytes=banks.gate->expert_bytes+banks.up->expert_bytes+banks.down->expert_bytes;
    resize_gguf_expert_cache(cache,3*bytes);
    const auto x=input(3,1,false),ids=tb::arange(3,x.options().dtype(tb::kInt64)).reshape({3,1});
    const auto weights=tb::full({3,1},1.,x.options());
    require(prepare_gguf_graph_pool(cache),"GGUF speculative graph pool missing");
    const auto metrics=[&] {
        wait_gguf_expert_admission(cache);
        std::map<std::string,double> value;
        for(const auto& [key,number]:gguf_expert_memory_metrics(cache))value[key]=number;
        return value;
    };
    const auto cached=[&](int expert) {
        const auto selected=tb::full({1,1},expert,ids.options());
        const auto before=gguf_expert_streamed_bytes(cache);
        for(const auto& bank:{banks.gate,banks.up,banks.down})(void)bank->forward(x.narrow(0,0,1),selected).cpu();
        require(gguf_expert_streamed_bytes(cache)==before,"GGUF committed expert was not resident");
    };
    begin_gguf_speculative_routes(cache,3);
    tb::Tensor output;
    mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    graph.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
    discard_gguf_speculative_routes(cache);
    require(metrics().at("resident_expert_cache_bytes")==0,"GGUF graph warmup admitted speculative experts");
    begin_gguf_speculative_routes(cache,3);graph.run();
    require(metrics().at("resident_expert_cache_bytes")==0,"GGUF verification admitted rejected rows before acceptance");
    commit_gguf_speculative_routes(cache,1);
    require(metrics().at("resident_expert_cache_bytes")==3*bytes,"GGUF verifier cache ignored executed rejected rows");
    require(metrics().at("gguf_speculative_route_rows_committed")==1,"GGUF committed graph warmup or rejected route rows");
    require(metrics().at("gguf_speculative_cache_observed_rows")==3,"GGUF verifier cache did not count all executed rows");
    cached(0);cached(1);cached(2);
    const auto before=metrics();
    begin_gguf_speculative_routes(cache,3);graph.run();discard_gguf_speculative_routes(cache);
    cached(0);
    require(metrics().at("gguf_speculative_route_rows_committed")==1,"GGUF cancellation learned speculative routes");
    require(metrics().at("gguf_speculative_route_rows_discarded")==before.at("gguf_speculative_route_rows_discarded")+3,
        "GGUF cancellation retained speculative rows");
    resize_gguf_expert_cache(cache,3*bytes);
    begin_gguf_speculative_routes(cache,3);graph.run();commit_gguf_speculative_routes(cache,3);
    require(metrics().at("gguf_speculative_route_rows_committed")==4,"GGUF full acceptance lost committed routes");
    require(metrics().at("resident_expert_cache_bytes")==3*bytes,"GGUF full acceptance did not fill the allowed expert cache");
    const auto* async_setting=std::getenv("TYLOINFER_GGUF_ASYNC_ADMISSION");
    require((metrics().at("gguf_async_admission_batches")>0)==(!async_setting || async_setting[0]!='0'),
        "GGUF asynchronous admission default/override was ignored");
    const auto* dma_setting=std::getenv("TYLOINFER_GGUF_ADMISSION_DMA");
    require((metrics().at("gguf_admission_dma_bytes")>0)==(!dma_setting || dma_setting[0]!='0'),
        "GGUF cache admission ignored its copy-engine default/override");
    require(metrics().at("gguf_prefill_dma_bytes")==0,
        "GGUF decode cache admission was counted as prefill DMA");
    cached(0);cached(1);cached(2);
    // A zero-length commit keeps the cache and drops the complete journal.
    begin_gguf_speculative_routes(cache,3);graph.run();commit_gguf_speculative_routes(cache,0);
    require(metrics().at("gguf_speculative_route_rows_committed")==4,"GGUF zero acceptance learned routes");
    const auto compact_before=metrics();
    auto repeated_ids=ids;repeated_ids.fill_(1);
    begin_gguf_speculative_routes(cache,3);graph.run();commit_gguf_speculative_routes(cache,1);
    const auto compact_after=metrics();
    require(compact_after.at("gguf_speculative_route_visits")-compact_before.at("gguf_speculative_route_visits")==3,
        "GGUF compact journal lost repeated expert heat");
    require(compact_after.at("gguf_speculative_unique_touches")-compact_before.at("gguf_speculative_unique_touches")==1,
        "GGUF compact journal repeated an expert LRU update");
    require(compact_after.at("gguf_speculative_cache_observed_rows")-compact_before.at("gguf_speculative_cache_observed_rows")==3,
        "GGUF compact journal lost executed rejected rows");
    require(compact_after.at("gguf_speculative_snapshot_touches")-compact_before.at("gguf_speculative_snapshot_touches")==1,
        "GGUF unchanged residency repeated an expert lookup");
    // The saved entry handles must preserve the same final LRU. Only the most
    // recently used expert's three projections fit after shrinking the cache.
    resize_gguf_expert_cache(cache,bytes);cached(1);
    // Erase those handles after the graph has observed them. A changed
    // residency version must force fresh lookups before touching any entry.
    const auto stale_before=metrics();
    begin_gguf_speculative_routes(cache,3);graph.run();(void)output.cpu();
    resize_gguf_expert_cache(cache,0);resize_gguf_expert_cache(cache,3*bytes);
    commit_gguf_speculative_routes(cache,1);
    const auto stale_after=metrics();
    require(stale_after.at("gguf_speculative_refind_touches")-stale_before.at("gguf_speculative_refind_touches")==1,
        "GGUF changed residency reused an erased expert handle");
    require(stale_after.at("gguf_speculative_snapshot_touches")==stale_before.at("gguf_speculative_snapshot_touches"),
        "GGUF stale speculative snapshot was not invalidated");
}
void run_fixed_speculative_admission() {
    const auto* slots=std::getenv("TYLOINFER_GGUF_STRATA_CACHE");
    if(slots && slots[0]=='0')return;
    std::weak_ptr<GgufExpertCache> lifetime;
    {
        Source source;
        for(const auto* role:{"gate","up","down"})source.add(role,{8,256,256});
        CudaExecutionContext execution;
        execution.config.moe_host_cache_bytes=2ULL<<20;
        const auto expert_bytes=3*source.find_tensor("gate")->nbytes/8;
        execution.config.moe_device_cache_bytes=2*expert_bytes;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        const auto cache=execution.gguf_expert_cache;lifetime=cache;
        auto x=input(64,1,false).to(tb::kFloat16),host_ids=tb::zeros({64,1},x.options().device(tb::kCPU).dtype(tb::kInt64));
        for(int i=32;i<64;++i)host_ids.data_ptr<std::int64_t>()[i]=1;
        auto weights=tb::full({64,1},1.,x.options().dtype(tb::kFloat32));
        (void)banks.forward(x,host_ids.to(tb::kCUDA),weights).cpu();
        restore_gguf_decode_workspace(cache);
        require(gguf_prefill_lendable_bytes(cache)>0,"GGUF fixed-admission test did not create slots");
        require(prepare_gguf_graph_pool(cache),"GGUF fixed-admission graph pool missing");
        x=x.narrow(0,0,2);weights=weights.narrow(0,0,2);
        const auto ids=(tb::arange(2,x.options().dtype(tb::kInt64))+6).reshape({2,1});
        tb::Tensor output;
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        begin_gguf_speculative_routes(cache,2);
        graph.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
        discard_gguf_speculative_routes(cache);
        for(int step=0;step<4;++step) {
            begin_gguf_speculative_routes(cache,2);graph.run();commit_gguf_speculative_routes(cache,1);
            wait_gguf_expert_admission(cache);check_moe(output,banks,x,ids,weights);
        }
        std::map<std::string,double> metrics;
        for(const auto& [key,value]:gguf_expert_memory_metrics(cache))metrics[key]=value;
        const auto* fast=std::getenv("TYLOINFER_GGUF_SPEC_ADAPT");
        require((metrics.at("gguf_adaptive_fast_rounds")>0)==(!fast || fast[0]!='0'),
            "GGUF speculative adaptive cadence ignored its default/override");
        const auto before=gguf_expert_streamed_bytes(cache);
        for(const auto& bank:{banks.gate,banks.up,banks.down})(void)bank->forward(x,ids).cpu();
        require(gguf_expert_streamed_bytes(cache)==before,"GGUF adaptive slots did not retain executed experts");
        // A cache-free metadata snapshot must not make a worker its owner's
        // final strong reference while destruction joins that same worker.
        begin_gguf_speculative_routes(cache,2);graph.run();commit_gguf_speculative_routes(cache,1);
    }
    require(lifetime.expired(),"GGUF fixed adaptive worker retained a destroyed cache");
}
#include "gguf_layer_admission_test.h"
void run_host_event_retirement() {
    Source source;
    for(const auto* name:{"gate","up","down"})source.add(name,{17,256,256});
    CudaExecutionContext execution;
    const auto image_bytes=source.find_tensor("gate")->nbytes/17;
    execution.config.moe_host_cache_bytes=2*image_bytes;
    execution.config.moe_device_cache_bytes=12*image_bytes;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    require(prepare_gguf_graph_pool(cache),"GGUF event-retirement graph pool missing");
    const auto x=input(3,2,false);
    auto ids=tb::arange(6,x.options().dtype(tb::kInt64)).reshape({3,2});
    const auto weights=tb::full({3,2},.5,x.options());
    tb::Tensor output;mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
    begin_gguf_speculative_routes(cache,3);
    graph.capture([&]{output=banks.forward(x,ids,weights);},[]{});
    discard_gguf_speculative_routes(cache);
    for(int step=0;step<32;++step) {
        ids.copy_((tb::arange(6,ids.options())+step*3).remainder(17).reshape({3,2}));
        begin_gguf_speculative_routes(cache,3);graph.run();commit_gguf_speculative_routes(cache,1);
        wait_gguf_expert_admission(cache);
        if(step%4==0)check_moe(output,banks,x,ids,weights);
    }
    std::map<std::string,double> metrics;
    for(const auto& [key,value]:gguf_expert_memory_metrics(cache))metrics[key]=value;
    require(metrics.at("gguf_deferred_events")==0,"GGUF completed windows retained retired events");
    if(metrics.at("gguf_direct_expert_calls")>0)
        require(metrics.at("gguf_deferred_event_releases")>0,"GGUF event-retirement test did not recycle writebacks behind a waiting graph");
}
void run_admission_failure_and_lifetime() {
    std::weak_ptr<GgufExpertCache> observed;
    {
        Source source;
        for(const auto* name:{"gate","up","down"})source.add(name,{3,256,256});
        CudaExecutionContext execution;
        execution.config.moe_host_cache_bytes=1;execution.config.moe_device_cache_bytes=0;
        GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
        const auto cache=execution.gguf_expert_cache;observed=cache;
        const auto bytes=banks.gate->expert_bytes+banks.up->expert_bytes+banks.down->expert_bytes;
        resize_gguf_expert_cache(cache,3*bytes);
        const auto x=input(3,1,false),ids=tb::arange(3,x.options().dtype(tb::kInt64)).reshape({3,1});
        const auto weights=tb::full({3,1},1.,x.options());
        prepare_gguf_graph_pool(cache);
        tb::Tensor output;
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        begin_gguf_speculative_routes(cache,3);
        graph.capture([&]{output=banks.forward(x,ids,weights);},[]{},[&]{output={};});
        discard_gguf_speculative_routes(cache);
        begin_gguf_speculative_routes(cache,3);graph.run();source.fail_next_read();
        bool failed=false;
        try {commit_gguf_speculative_routes(cache,3);wait_gguf_expert_admission(cache);}
        catch(const std::runtime_error& error){failed=std::string(error.what())=="fixture single read failure";}
        require(failed,"GGUF background admission swallowed a source error");
        begin_gguf_speculative_routes(cache,3);graph.run();commit_gguf_speculative_routes(cache,3);
        // Leave the admission outstanding: destroying the model/cache must
        // join it without a bank -> cache -> future ownership cycle.
    }
    require(observed.expired(),"GGUF asynchronous admission retained a destroyed model cache");
}
void run_row_prefetch() {
    const auto device=tb::Device(tb::kCUDA,mfq_current_cuda_device());
    for(bool batched:{false,true}) {
        Source source;source.enable_parallel_reads();source.add("table",{11,256});if(batched)source.enable_batches();
        auto table=std::make_shared<GgufRowTable>(source,"table");
        const auto expected=source.decoded("table");
        for(int count:{9,17,33}) {
            std::vector<int64_t> ids(count);for(int i=0;i<count;++i)ids[i]=(i*7)%22;
            auto collect=prefetch_gguf_shards({table,table},11,ids,{1,count},device);
            const auto values=collect().to(tb::kFloat32).cpu();
            for(int row=0;row<count;++row)for(int k=0;k<256;++k)
                require(values.data_ptr<float>()[row*256+k]==float(mfq_half(expected.data_ptr<float>()[(ids[row]%11)*256+k])),
                    "GGUF asynchronous PLE changed shard, duplicate or row ordering");
        }
        const auto reads=source.read_calls();
        {auto cancelled=prefetch_gguf_shards({table},11,{0,1,1},{3},device);}
        require(source.read_calls()==reads,"Cancelling a cached PLE job repeated its rows");
        bool rejected=false;
        try{auto bad=prefetch_gguf_shards({table},11,{11},{1},device);bad();}
        catch(const std::out_of_range&){rejected=true;}
        require(rejected,"Asynchronous PLE accepted an invalid shard");
    }
    Source serial;serial.add("table",{11,256});
    auto serial_table=std::make_shared<GgufRowTable>(serial,"table");const auto serial_reads=serial.read_calls();
    require(!prefetch_gguf_shards({serial_table},11,{1,2},{2},device) && serial.read_calls()==serial_reads,
        "PLE scheduled concurrent reads for a provider with a shared seek cursor");
    Source source;source.enable_parallel_reads();source.add("table",{11,256});
    auto table=std::make_shared<GgufRowTable>(source,"table");source.fail_next_read(true);
    auto collect=prefetch_gguf_shards({table},11,{7,1,7},{3},device);
    bool rejected=false;try{collect();}catch(const std::runtime_error& error){rejected=std::string(error.what())=="fixture single read failure";}
    require(rejected,"PLE I/O did not run on its CPU worker or swallowed its error");
    collect=prefetch_gguf_shards({table},11,{7,1,7},{3},device);
    require(collect().numel()==3*256,"Asynchronous PLE could not recover after a read failure");
    std::cout<<"GGUF asynchronous PLE shard/duplicate/tail ordering, cancellation, background failure and retry PASS\n";
}
void run() {
    run_row_prefetch();
    Source source; source.add("bank", {3, 7, 256}); source.add("table", {11, 256});
    CudaExecutionContext execution;
    execution.config.moe_host_cache_bytes = 65536;
    execution.config.moe_device_cache_bytes = 8192;
    GgufRoutedWeight bank(execution, source, "bank");
    auto reference = source.decoded("bank");
    for (const int tokens : {1, 2, 4, 5, 8, 64}) for (const bool routed : {false, true}) {
        auto x = input(tokens, 3, routed), ids = routes(tokens, 0);
        check_projection(bank.forward(x, ids), x, ids, reference);
    }
    require(prepare_gguf_graph_pool(execution.gguf_expert_cache), "GGUF graph pool missing");
    auto x = input(2, 3, false), ids = routes(2, 0);
    tb::Tensor y;
    {
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&] { y = bank.forward(x, ids); }, [] {});
        for (int rotation : {1, 2}) {
            ids.copy_(routes(2, rotation));
            resize_gguf_expert_cache(execution.gguf_expert_cache, 0);
            graph.run();
            check_projection(y, x, ids, reference);
        }
    }
    require(gguf_expert_cache_budget_bytes(execution.gguf_expert_cache) == 0, "GGUF budget resize ignored");
    auto table = std::make_shared<GgufRowTable>(source, "table");
    auto stage = make_gguf_row_stage({table}, 11, {3}, tb::Device(tb::kCUDA, 0));
    auto expected = source.decoded("table");
    const auto reads_before=source.read_calls();
    cudaGraph_t row_graph = nullptr;
    cudaGraphExec_t row_executable = nullptr;
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(mfq_current_cuda_stream(), cudaStreamCaptureModeGlobal));
    stage->decode();
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(mfq_current_cuda_stream(), &row_graph));
    TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&row_executable, row_graph, 0));
    for (const std::vector<std::int64_t> row_ids : {std::vector<std::int64_t>{7, 1, 7}, {3, 10, 0}}) {
        stage->upload(row_ids, {3});
        TYLOINFER_CUDA_CHECK(cudaGraphLaunch(row_executable, mfq_current_cuda_stream()));
        const auto values = stage->output().to(tb::kFloat32).cpu();
        for (int row = 0; row < 3; ++row) for (int k = 0; k < 256; ++k)
            require(values.data_ptr<float>()[row * 256 + k] == float(mfq_half(expected.data_ptr<float>()[row_ids[row] * 256 + k])),
                "GGUF streamed row order changed");
    }
    stage->upload({7,1,7},{3});
    require(source.read_calls()==reads_before+5,"GGUF PLE repeated duplicate rows or ignored its bounded cache");
    TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(row_executable));
    TYLOINFER_CUDA_CHECK(cudaGraphDestroy(row_graph));
    auto projection = load_gguf_weight(source, "table", 0, true, 2, 5, 32, 128).dequantize().cpu();
    for (int row = 0; row < 3; ++row) for (int k = 0; k < 96; ++k)
        require(projection.data_ptr<float>()[row * 96 + k] == expected.data_ptr<float>()[(row + 2) * 256 + k + 32],
            "GGUF block slice changed values");
    auto output_head = load_gguf_weight(source, "table", 0);
    require(output_head.forward(input(1, 1, false).to(tb::kBFloat16)).scalar_type() == tb::kFloat16,
        "GGUF BF16 input produced logits unsupported by CUDA sampling");
    const auto auxiliary = tb::ones({7, 256}, tb::TensorOptions().device(tb::kCUDA).dtype(tb::kBFloat16));
    for (const int tokens : {1, 2, 4, 5, 8, 64}) {
        auto logits = dense_projection(input(tokens, 1, false), auxiliary).contiguous();
        require(logits.scalar_type() == tb::kFloat32, "BF16 auxiliary changed native activation precision");
        require(moe_topk_cuda(logits, 2, false, false, false, false, mfq_nullopt, 1e-20, 1.).at(0).numel() == tokens * 2,
            "BF16 auxiliary could not feed the CUDA router");
    }
}
void run_compact_mmq_gather() {
    namespace mmq=mfq::cuda::gguf::mmq;
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8);
    for(int experts:{1,5,16}) {
        mmq::Copy jobs[3];
        std::array<tb::Tensor,3> expected,destinations;
        std::vector<tb::Tensor> sources;
        for(int role=0;role<3;++role) {
            auto& job=jobs[role];job.bytes=role==0?4111:role==1?4099:19;
            job.offset=role+1;job.stride=job.bytes+job.offset+23;job.guard_bytes=7+role*6;
            expected[role]=tb::full({int64_t(experts*job.stride+job.guard_bytes)},0xa5,options);
            destinations[role]=expected[role].to(tb::kCUDA);job.destination=destinations[role].data_ptr();
            for(int e=0;e<experts;++e) {
                if(e%3==0) {job.source[e]=destinations[role].data_ptr<uint8_t>()+e*job.stride+job.offset;continue;}
                const auto offset=e%4+1;
                auto host=tb::empty({int64_t(job.bytes+offset)},options);
                for(int i=0;i<host.numel();++i)host.data_ptr<uint8_t>()[i]=uint8_t(i*29+e*7+role);
                sources.push_back(host.to(tb::kCUDA));job.source[e]=sources.back().data_ptr<uint8_t>()+offset;
                std::memcpy(expected[role].data_ptr<uint8_t>()+e*job.stride+job.offset,host.data_ptr<uint8_t>()+offset,job.bytes);
            }
            std::memset(expected[role].data_ptr<uint8_t>()+experts*job.stride,0,job.guard_bytes);
        }
        mmq::gather(jobs,experts,mfq_current_cuda_stream());
        for(int role=0;role<3;++role) {
            const auto actual=destinations[role].cpu();
            require(std::memcmp(actual.data_ptr(),expected[role].data_ptr(),actual.nbytes())==0,
                "GGUF compact MMQ copy changed a skipped range, unaligned tail, guard or gap");
        }
        for(auto& job:jobs){job.bytes=0;job.guard_bytes=0;}
        mmq::gather(jobs,experts,mfq_current_cuda_stream());
    }
}
void run_inplace_mmq_moe() {
    namespace mmq=mfq::cuda::gguf::mmq;
    if(!mmq::direct_fits(GGML_TYPE_IQ3_XXS,1024,1024,mfq_current_cuda_device()))return;
    ScopedEnvironment setting("TYLOINFER_GGUF_PREFILL_DIRECT","0");
    Source source;
    source.add("gate",{4,512,1024},GGML_TYPE_IQ3_XXS);
    source.add("up",{4,512,1024},GGML_TYPE_IQ3_XXS);
    source.add("down",{4,1024,512},GGML_TYPE_Q2_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=8ULL<<20;
    execution.config.moe_device_cache_bytes=8ULL<<20;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto options=tb::TensorOptions().device(tb::kCUDA);
    auto x=tb::full({67,1024},.0001,options.dtype(tb::kFloat16));
    auto ids=tb::arange(67*3,options.dtype(tb::kInt64)).remainder(4).reshape({67,3});
    auto weights=tb::full({67,3},1./3,options.dtype(tb::kFloat32));
    const auto expected=banks.forward(x,ids,weights).cpu();
    restore_gguf_decode_workspace(execution.gguf_expert_cache);
    setting.set("1");
    const auto actual=banks.forward(x,ids,weights).cpu();
    require(std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes())==0,"GGUF cached direct MMQ changed the PP output bits");
    bool used=false;
    for(const auto& [key,value]:gguf_expert_memory_metrics(execution.gguf_expert_cache))
        if(key=="gguf_prefill_inplace_mmq_bytes")used=value>0;
    require(used,"GGUF cached direct MMQ test did not exercise resident matrices");
}
void run_split_expert_copy() {
    using namespace mfq::cuda;
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8);
    for(const char* range: {"0","1"}) {
        ScopedEnvironment variant("TYLOINFER_MOE_COPY_RANGES",range);
        for(int boundary:{0,13,32,96,113,128})for(int cancelled:{0,1})for(bool empty:{false,true}) {
            std::vector<uint64_t> packet(16);packet[0]=empty?0:3;packet[1]=128;packet[3]=boundary;
            std::array<tb::Tensor,3> source,destination,expected;
            constexpr int sizes[3]={31,64,17},offsets[3]={0,32,96};
            for(int field=0;field<3;++field) {
                auto raw=tb::empty({sizes[field]+5},options);
                for(int i=0;i<raw.numel();++i)raw.data_ptr<uint8_t>()[i]=uint8_t(i*17+field*31);
                source[field]=raw.to(tb::kCUDA);
                expected[field]=tb::full({sizes[field]+7},0xa5,options);
                destination[field]=expected[field].to(tb::kCUDA);
                const int first=4+field*4;
                packet[first]=reinterpret_cast<uint64_t>(destination[field].data_ptr<uint8_t>()+3);
                packet[first+1]=reinterpret_cast<uint64_t>(source[field].data_ptr<uint8_t>()+1);
                packet[first+2]=offsets[field];packet[first+3]=sizes[field];
            }
            auto table=tb::empty({int64_t(packet.size()*sizeof(uint64_t))},options);
            std::memcpy(table.data_ptr(),packet.data(),table.nbytes());table=table.to(tb::kCUDA);
            auto snapshot=tb::empty_like(table);
            auto abort=tb::full({1},cancelled,options.dtype(tb::kInt32)).to(tb::kCUDA);
            mfq::cuda::gguf::snapshot_direct_plan(table.data_ptr(),reinterpret_cast<const uint32_t*>(abort.data_ptr()),
                snapshot.data_ptr(),3,nullptr,nullptr,0,nullptr,nullptr,0,mfq_current_cuda_stream());
            const auto guard=reinterpret_cast<const uint32_t*>(snapshot.data_ptr())+4;
            moe_copy_mapped_phase(snapshot.data_ptr(),guard,3,128,0,mfq_current_cuda_stream());
            for(int field=0;field<3;++field) {
                const auto raw=source[field].cpu();
                if(!empty && !cancelled)for(int i=0;i<sizes[field];++i)
                    if(offsets[field]+i<boundary)expected[field].data_ptr<uint8_t>()[3+i]=raw.data_ptr<uint8_t>()[1+i];
                const auto actual=destination[field].cpu();
                require(!std::memcmp(actual.data_ptr(),expected[field].data_ptr(),actual.nbytes()),
                    "Gate/Up copy wrote Down early, changed a guard or missed a split tail");
            }
            moe_copy_mapped_phase(snapshot.data_ptr(),guard,3,128,1,mfq_current_cuda_stream());
            for(int field=0;field<3;++field) {
                if(!empty && !cancelled) {
                    const auto raw=source[field].cpu();
                    std::memcpy(expected[field].data_ptr<uint8_t>()+3,raw.data_ptr<uint8_t>()+1,sizes[field]);
                }
                const auto actual=destination[field].cpu();
                require(!std::memcmp(actual.data_ptr(),expected[field].data_ptr(),actual.nbytes()),
                    "Down copy missed bytes or copied an empty/cancelled packet");
            }
        }
    }
    std::cout<<"GGUF split Gate/Up+Down copy range variants/partial tails/guards/empty/cancel PASS\n";
}
void bench_cpu_projection(int rows,int output_width,int input_width,int type) {
    namespace cpu=mfq::cpu::gguf;
    require(rows>=1 && rows<=4 && output_width>0,"CPU projection benchmark geometry");
    Source source;source.add("projection",{output_width,input_width},ggml_type(type));
    const auto* meta=source.find_tensor("projection");std::vector<std::uint8_t> bytes(meta->nbytes);
    source.tensor_reader("projection")(0,reinterpret_cast<std::byte*>(bytes.data()),bytes.size());
    const bool iq=cpu::input_supported(type);
    const auto packet=iq?cpu::input_bytes(input_width):cpu::hidden_bytes(input_width,type);
    std::vector<std::uint8_t> activation(rows*packet);
    std::vector<float> values(rows*input_width),output(rows*output_width);
    const void* input[4]{};float* out[4]{};
    for(int t=0;t<rows;++t) {
        for(int k=0;k<input_width;++k)values[t*input_width+k]=.3f*std::sin(float(t*53+k*17)*.013f);
        input[t]=activation.data()+t*packet;out[t]=output.data()+t*output_width;
        if(iq)cpu::quantize_input(values.data()+t*input_width,activation.data()+t*packet,input_width);
        else cpu::quantize_hidden(values.data()+t*input_width,activation.data()+t*packet,input_width,type);
    }
    const cpu::Projection projection{bytes.data(),mfq::gguf::row_bytes(ggml_type(type),input_width),type,output_width,input_width};
    const auto cores=mfq::cpu::physical_cores(true);
    mfq::cpu::ScopedHostAffinity caller(cores.empty()?-1:cores.back());
    mfq::HostParallelPool pool([&](std::size_t i){if(i+1<cores.size())(void)mfq::cpu::pin_current_thread(cores[i]);},false,20000000);
    const auto dispatch=cpu::projection_dispatch();
    const auto run=[&]{pool.run(0,output_width,32,std::max(1,int(cores.size())),
        [&](std::int64_t begin,std::int64_t end){cpu::project(projection,input,rows,out,int(begin),int(end),dispatch);});};
    for(int i=0;i<8;++i)run();std::vector<double> samples;
    constexpr int repetitions=100;
    for(int sample=0;sample<5;++sample) {
        const auto begin=std::chrono::steady_clock::now();for(int i=0;i<repetitions;++i)run();
        samples.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/repetitions);
    }
    for(float value:output)require(std::isfinite(value),"CPU projection benchmark nonfinite output");
    std::sort(samples.begin(),samples.end());
    std::printf("cpu_projection M=%d N=%d K=%d type=%d threads=%zu us=%.6f\n",rows,output_width,input_width,type,cores.size(),samples[2]);
}
void bench_dense_small_mmq(int tokens,int rows,int columns,int format) {
    namespace mmq=mfq::cuda::gguf::mmq;
    if(tokens<=0 || rows<=0 || columns<=0 || !mmq::fits(format,rows,mfq_current_cuda_device()))
        throw std::invalid_argument("dense small MMQ geometry");
    Source source;source.add("w",{rows,columns},ggml_type(format));
    auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto host=tb::empty({tokens,columns},options.device(tb::kCPU));
    for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.2f*std::sin(float(i)*.17f);
    auto input=host.to(tb::kCUDA),old_output=tb::empty({tokens,rows},options),new_output=tb::empty_like(old_output);
    auto q8=tb::empty({int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(columns))},options.dtype(tb::kUInt8));
    auto activation=tb::empty({int64_t(mmq::activation_bytes(tokens,columns))},q8.options());
    const bool direct=mmq::direct_fits(format,rows,columns,mfq_current_cuda_device());
    auto guarded=direct?tb::Tensor{}:tb::zeros({weight.packed.numel()+128*int64_t(ggml_type_size(ggml_type(format)))},q8.options());
    if(guarded.defined())guarded.narrow(0,0,weight.packed.numel()).copy_(weight.packed);
    const void* addresses[]={weight.packed.data_ptr()};mmq::Context mmq_context;
    const auto stream=mfq_current_cuda_stream();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
    struct ReleasePool {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~ReleasePool(){context->end_graph_pool(stream);}} release_pool{context,stream};
    mfq::cuda::DecodeWindow old_graph(stream),new_graph(stream);
    old_graph.capture([&] {
        mfq::cuda::gguf::quantize_input(input.data_ptr<float>(),tokens,columns,q8.data_ptr(),stream);
        mfq::cuda::gguf::matvec_q8(ggml_type(format),weight.packed.data_ptr(),weight.row_bytes,rows,columns,
            q8.data_ptr(),tokens,old_output.data_ptr<float>(),stream);
    },[]{});
    new_graph.capture([&] {
        mmq::quantize(input.data_ptr<float>(),nullptr,activation.data_ptr(),format,columns,tokens,stream);
        mmq_context.run({direct?nullptr:guarded.data_ptr(),format,rows,columns,size_t(rows*weight.row_bytes),1,
            activation.data_ptr(),nullptr,nullptr,tokens,tokens,new_output.data_ptr<float>(),rows,direct?addresses:nullptr},stream);
    },[]{});
    old_graph.run();new_graph.run();
    const auto expected=old_output.cpu(),actual=new_output.cpu();float max_error=0,max_value=0;
    for(int64_t i=0;i<expected.numel();++i) {
        const auto x=expected.data_ptr<float>()[i],y=actual.data_ptr<float>()[i];
        require(std::isfinite(x) && std::isfinite(y),"dense small MMQ produced nonfinite output");
        max_error=std::max(max_error,std::abs(x-y));max_value=std::max(max_value,std::abs(x));
    }
    require(max_error<=1e-3f+1e-3f*max_value,"dense small MMQ differs from Q8 DP4A reference");
    for(int i=0;i<8;++i){old_graph.run();new_graph.run();}
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples[2];
    for(int trial=0;trial<7;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int repeat=0;repeat<64;++repeat)
            require((mode?new_graph:old_graph).enqueue_if_device_only(),"dense MMQ timing graph has host tasks");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));samples[mode].push_back(elapsed/64);
    }
    for(auto& times:samples)std::sort(times.begin(),times.end());
    std::printf("dense_small_mmq format=%d M=%d N=%d K=%d direct=%d dp4a_us=%.6f mmq_us=%.6f max_error=%.8g max_value=%.8g median7=1 gpu_queued=1\n",
        format,tokens,rows,columns,int(direct),1000.*samples[0][3],1000.*samples[1][3],max_error,max_value);
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}
void bench_projection_branches(int tokens,int qkv_rows,int gate_rows,int columns,bool legacy_auxiliary=false) {
    if(tokens<1 || tokens>32 || qkv_rows<1 || gate_rows<1 || columns<256 || columns%256)
        throw std::invalid_argument("mixed projection benchmark geometry");
    Source source;
    source.add("qkv",{qkv_rows,columns},GGML_TYPE_IQ4_XS);
    source.add("gate",{gate_rows,columns},GGML_TYPE_IQ3_S);source.full_iq3_codes("gate");
    source.add_dense("alpha",{3,columns});source.add_dense("beta",{5,columns});
    CudaExecutionContext execution;std::vector<weight_loader::Linear> projections;
    for(const char* name:{"qkv","gate","alpha","beta"})
        projections.push_back(weight_loader::linear(execution,source,name));
    auto group=weight_loader::grouped_linear(execution,projections,1);
    require(bool(group),"mixed projection benchmark group unavailable");
    const std::vector<GgufWeight> legacy_weights={weight_loader::linear_weight(projections[0])->gguf,
        weight_loader::linear_weight(projections[1])->gguf};
    const auto legacy_executor=execution.projection_executors.at(mfq_current_cuda_device());
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto x=tb::full({1,tokens,columns},.003,options);
    const auto stream=mfq_current_cuda_stream();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<mfq::cuda::Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    std::vector<tb::Tensor> outputs[2];mfq::cuda::DecodeWindow serial(stream),parallel(stream);
    execution.config.decode_branch_parallel=false;
    serial.capture([&]{outputs[0]=group(execution,x);},[]{});
    execution.config.decode_branch_parallel=true;
    parallel.capture([&] {
        if(!legacy_auxiliary){outputs[1]=group(execution,x);return;}
        std::vector<tb::Tensor> branches;
        require(legacy_executor->run(3,[&](std::size_t branch) {
            mfq_cuda_record_stream(x,mfq_get_current_cuda_stream());
            if(!branch) {
                const auto prepared=prepare_gguf_group_input(legacy_weights,x);
                outputs[1].clear();
                for(const auto& weight:legacy_weights)outputs[1].push_back(prepared.project(weight));
                return tb::Tensor{};
            }
            return projections[branch+1](execution,x);
        },branches),"legacy auxiliary benchmark streams unavailable");
        outputs[1].push_back(branches[1]);outputs[1].push_back(branches[2]);
    },[]{});
    serial.run();parallel.run();
    for(std::size_t i=0;i<outputs[0].size();++i) {
        const auto a=outputs[0][i].contiguous().cpu(),b=outputs[1][i].contiguous().cpu();
        require(a.nbytes()==b.nbytes() && std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,
            "parallel mixed projection changed output bits");
    }
    for(int i=0;i<8;++i){serial.run();parallel.run();}
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples[2];
    for(int trial=0;trial<7;++trial)for(int pass=0;pass<(legacy_auxiliary?1:2);++pass) {
        const int mode=legacy_auxiliary?1:(trial+pass)%2;
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int repeat=0;repeat<64;++repeat)
            require((mode?parallel:serial).enqueue_if_device_only(),"mixed projection benchmark contains host graph tasks");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));samples[mode].push_back(ms/64);
    }
    for(auto& sample:samples)std::sort(sample.begin(),sample.end());
    if(legacy_auxiliary)
        std::printf("projection_auxiliary_baseline M=%d K=%d qkv_rows=%d gate_rows=%d old_default_us=%.6f bitwise=1 median7=1 PASS\n",
            tokens,columns,qkv_rows,gate_rows,1000.*samples[1][3]);
    else std::printf("projection_branches M=%d K=%d qkv_rows=%d gate_rows=%d serial_us=%.6f parallel_us=%.6f bitwise=1 median7=1 PASS\n",
        tokens,columns,qkv_rows,gate_rows,1000.*samples[0][3],1000.*samples[1][3]);std::fflush(stdout);
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}

void bench_projection_layout(int tokens,int columns,int key_width,int value_width,int gate_width) {
    if(tokens<1 || tokens>32 || columns<1 || key_width<1 || value_width<1 || gate_width<1)
        throw std::invalid_argument("projection layout benchmark geometry");
    Source source;
    source.add("q",{key_width,columns},GGML_TYPE_Q8_0);source.add("k",{key_width,columns},GGML_TYPE_Q8_0);
    source.add("v",{value_width,columns},GGML_TYPE_Q8_0);source.add("gate",{gate_width,columns},GGML_TYPE_Q8_0);
    CudaExecutionContext execution;
    std::vector<weight_loader::Linear> projections;
    for(const char* name:{"q","k","v","gate"})projections.push_back(weight_loader::linear(execution,source,name));
    auto group=weight_loader::grouped_linear(execution,projections,3);
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto x=tb::full({1,tokens,columns},.003,options);
    const auto stream=mfq_current_cuda_stream();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<mfq::cuda::Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    std::vector<tb::Tensor> outputs[2];mfq::cuda::DecodeWindow old_graph(stream),view_graph(stream);
    old_graph.capture([&]{outputs[0]=group(execution,x);for(auto& v:outputs[0])v=v.contiguous();},[]{});
    view_graph.capture([&]{outputs[1]=group(execution,x);},[]{});
    old_graph.run();view_graph.run();
    for(std::size_t i=0;i<outputs[0].size();++i) {
        const auto a=outputs[0][i].contiguous().cpu(),b=outputs[1][i].contiguous().cpu();
        require(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"projection layout benchmark changed output bits");
    }
    for(int i=0;i<8;++i){old_graph.run();view_graph.run();}
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples[2];
    for(int trial=0;trial<7;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int repeat=0;repeat<64;++repeat)
            require((mode?view_graph:old_graph).enqueue_if_device_only(),"projection layout benchmark acquired host tasks");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));samples[mode].push_back(ms/64);
    }
    for(auto& s:samples)std::sort(s.begin(),s.end());
    std::printf("projection_layout M=%d K=%d key=%d value=%d gate=%d contiguous_us=%.6f views_us=%.6f exact=1 median7=1\n",
        tokens,columns,key_width,value_width,gate_width,1000.*samples[0][3],1000.*samples[1][3]);std::fflush(stdout);
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}

void bench_dense_io(int tokens,int rows,int columns,int format,bool position_major=false) {
    if(tokens<=0 || rows<=0 || columns<=0)throw std::invalid_argument("dense I/O benchmark geometry");
    Source source;source.add("w",{rows,columns},ggml_type(format));
    auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    auto input=tb::full({tokens,columns},.003,options),fp32_input=tb::empty({tokens,columns},options.dtype(tb::kFloat32));
    auto fp32_output=tb::empty({tokens,rows},fp32_input.options());
    auto old_output=tb::empty({tokens,rows},options),new_output=tb::empty_like(old_output);
    auto activation=tb::empty({std::int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(columns))},options.dtype(tb::kUInt8));
    if(position_major)require(mfq::cuda::gguf::supports_interleaved_activation(ggml_type(format),tokens),"dense position-major timing format unsupported");
    auto interleaved=position_major?tb::empty({int64_t(mfq::cuda::gguf::interleaved_activation_bytes(tokens,columns))},activation.options()):tb::Tensor{};
    const auto stream=mfq_current_cuda_stream();
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());context->begin_graph_pool(stream);
    struct ReleasePool {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~ReleasePool(){context->end_graph_pool(stream);}} release_pool{context,stream};
    mfq::cuda::DecodeWindow old_graph(stream),new_graph(stream);
    old_graph.capture([&] {
        if(position_major) {
            mfq::cuda::gguf::quantize_input_f16(input.data_ptr(),tokens,columns,activation.data_ptr(),stream);
            mfq::cuda::gguf::matvec_q8_f16(ggml_type(format),weight.packed.data_ptr(),weight.row_bytes,rows,columns,
                activation.data_ptr(),tokens,old_output.data_ptr(),stream);
            return;
        }
        fp32_input=input.to(tb::kFloat32);
        mfq::cuda::gguf::quantize_input(fp32_input.data_ptr<float>(),tokens,columns,activation.data_ptr(),stream);
        mfq::cuda::gguf::matvec_q8(ggml_type(format),weight.packed.data_ptr(),weight.row_bytes,rows,columns,
            activation.data_ptr(),tokens,fp32_output.data_ptr<float>(),stream);
        old_output=fp32_output.to(tb::kFloat16);
    },[]{});
    new_graph.capture([&] {
        if(position_major) {
            mfq::cuda::gguf::quantize_input_interleaved(input.data_ptr(),true,tokens,columns,interleaved.data_ptr(),nullptr,stream);
            mfq::cuda::gguf::matvec_interleaved(ggml_type(format),weight.packed.data_ptr(),weight.row_bytes,rows,columns,
                interleaved.data_ptr(),tokens,new_output.data_ptr(),true,stream);
            return;
        }
        mfq::cuda::gguf::quantize_input_f16(input.data_ptr(),tokens,columns,activation.data_ptr(),stream);
        mfq::cuda::gguf::matvec_q8_f16(ggml_type(format),weight.packed.data_ptr(),weight.row_bytes,rows,columns,
            activation.data_ptr(),tokens,new_output.data_ptr(),stream);
    },[]{});
    old_graph.run();new_graph.run();
    const auto old_host=old_output.cpu(),new_host=new_output.cpu();
    require(std::memcmp(old_host.data_ptr(),new_host.data_ptr(),old_host.nbytes())==0,"dense I/O benchmark changed output bits");
    for(int i=0;i<4;++i){old_graph.run();new_graph.run();}
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples[2];
    for(int trial=0;trial<7;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int repeat=0;repeat<64;++repeat)
            require((mode?new_graph:old_graph).enqueue_if_device_only(),"dense I/O timing graph unexpectedly has host tasks");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));samples[mode].push_back(elapsed/64);
    }
    for(auto& times:samples)std::sort(times.begin(),times.end());
    std::printf("%s format=%d M=%d N=%d K=%d old_us=%.6f fused_us=%.6f bitwise=1 median7=1 gpu_queued=1\n",
        position_major?"dense_position_major":"dense_io",format,tokens,rows,columns,1000.*samples[0][3],1000.*samples[1][3]);
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}
void bench_mmq_layer_quant(int tokens,int experts,int routes,int rows,int columns,int format) {
    namespace mmq=mfq::cuda::gguf::mmq;
    if(tokens<=0 || experts<=0 || routes<=0 || routes>experts || tokens>INT_MAX/routes ||
            !mmq::fits(format,rows,mfq_current_cuda_device()))throw std::invalid_argument("MMQ layer benchmark geometry");
    const int slots=tokens*routes;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    Source source;source.add("w",{int64_t(experts)*rows,columns},ggml_type(format));
    auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
    auto packed=tb::zeros({weight.packed.numel()+128*int64_t(ggml_type_size(ggml_type(format)))},weight.packed.options());
    packed.narrow(0,0,weight.packed.numel()).copy_(weight.packed);
    auto x=tb::empty({tokens,columns},options.device(tb::kCPU));
    for(int64_t i=0;i<x.numel();++i)x.data_ptr<float>()[i]=.1f*std::sin(float(i)*.17f);
    x=x.to(tb::kCUDA);
    std::vector<std::vector<int32_t>> groups(experts);
    for(int slot=0;slot<slots;++slot)groups[slot%experts].push_back(slot);
    auto inverse=tb::empty({slots},options.device(tb::kCPU).dtype(tb::kInt32));
    auto indices=tb::empty_like(inverse);int at=0;
    std::vector<int> offsets{0};
    for(const auto& group:groups) {
        for(int slot:group){inverse.data_ptr<int32_t>()[slot]=at;indices.data_ptr<int32_t>()[at++]=slot/routes;}
        offsets.push_back(at);
    }
    inverse=inverse.to(tb::kCUDA);indices=indices.to(tb::kCUDA);
    struct Group {int first,count,maximum,experts;tb::Tensor bounds,ids,quantized;};
    std::vector<Group> work;
    for(int first=0;first<experts;first+=mmq::group_capacity) {
        const int end=std::min(experts,first+mmq::group_capacity),count=offsets[end]-offsets[first];
        if(!count)continue;
        Group group{first,count,0,end-first};
        group.bounds=tb::empty({group.experts+1},options.device(tb::kCPU).dtype(tb::kInt32));
        for(int e=first;e<=end;++e) {
            group.bounds.data_ptr<int32_t>()[e-first]=offsets[e]-offsets[first];
            if(e<end)group.maximum=std::max(group.maximum,offsets[e+1]-offsets[e]);
        }
        group.bounds=group.bounds.to(tb::kCUDA);
        group.ids=tb::arange(count+mmq::id_guard,options.dtype(tb::kInt32));
        group.quantized=tb::empty({int64_t(mmq::activation_bytes(count,columns))},packed.options());
        work.push_back(std::move(group));
    }
    auto global=tb::empty({int64_t(mmq::activation_bytes(slots,columns))},packed.options());
    auto output=tb::empty({slots,rows},options);
    mmq::Context context;
    const auto stream=mfq_current_cuda_stream();
    const auto run=[&](bool shared) {
        if(shared)mmq::quantize_scatter(x.data_ptr<float>(),inverse.data_ptr<int32_t>(),global.data_ptr(),format,columns,tokens,routes,stream);
        for(auto& group:work) {
            const auto start=offsets[group.first];
            if(!shared)mmq::quantize(x.data_ptr<float>(),indices.data_ptr<int32_t>()+start,group.quantized.data_ptr(),format,columns,group.count,stream);
            context.run({packed.data_ptr<std::uint8_t>()+int64_t(group.first)*rows*weight.row_bytes,format,rows,columns,
                std::size_t(rows*weight.row_bytes),group.experts,shared?global.data_ptr():group.quantized.data_ptr(),
                group.bounds.data_ptr<int32_t>(),group.ids.data_ptr<int32_t>(),group.count,group.maximum,
                output.data_ptr<float>()+int64_t(start)*rows,rows,nullptr,shared?slots:0,shared?start:0},stream);
        }
    };
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    for(bool shared:{false,true}) {
        for(int i=0;i<2;++i)run(shared);
        std::vector<float> samples;
        constexpr int repeats=5;
        for(int trial=0;trial<3;++trial) {
            TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int i=0;i<repeats;++i)run(shared);
            TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));samples.push_back(elapsed/repeats);
        }
        std::sort(samples.begin(),samples.end());
        std::printf("mmq_layer_quant shared=%d tokens=%d experts=%d routes=%d N=%d K=%d format=%d groups=%zu ms=%.6f median3=1\n",
            int(shared),tokens,experts,routes,rows,columns,format,work.size(),samples[1]);
    }
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}
void bench_resident_overlap(int only_rows,bool half=false,bool rotating=false) {
    // Hold residency fixed, isolating the compute/copy schedule from cache
    // adaptation and speculative acceptance. No device-specific dimensions.
    Source source;
    source.add("gate",{32,640,2560},GGML_TYPE_IQ3_XXS);
    source.add("up",{32,640,2560},GGML_TYPE_IQ3_XXS);
    source.add("down",{32,2560,640},GGML_TYPE_Q2_0);
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=128ULL<<20;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto cache=execution.gguf_expert_cache;
    const auto expert_bytes=banks.gate->expert_bytes+banks.up->expert_bytes+banks.down->expert_bytes;
    resize_gguf_expert_cache(cache,16*expert_bytes);
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto prompt=tb::full({64,2560},.0001,options.dtype(tb::kFloat16));
    auto prompt_ids=tb::arange(64,options.dtype(tb::kInt64)).remainder(16).reshape({64,1});
    (void)banks.forward(prompt,prompt_ids,tb::full({64,1},1.,options));
    restore_gguf_decode_workspace(cache);
    require(prepare_gguf_graph_pool(cache),"GGUF overlap benchmark graph pool missing");
    for(int rows:{1,2,3,4}) {
        if(only_rows && rows!=only_rows)continue;
        auto x=tb::full({rows,2560},.0001,options.dtype(half?tb::kFloat16:tb::kFloat32));
        auto selected=tb::empty({rows,10},options.device(tb::kCPU).dtype(tb::kInt64));
        for(int token=0;token<rows;++token)for(int route=0;route<10;++route)
            selected.data_ptr<int64_t>()[token*10+route]=route<8?(route+token*3)%16:16+(token+route)%3;
        auto ids=selected.to(tb::kCUDA),weights=tb::full({rows,10},.1,options);
        auto shifted=selected.clone();
        if(rotating)for(int token=0;token<rows;++token)for(int route=8;route<10;++route)
            shifted.data_ptr<int64_t>()[token*10+route]+=3;
        auto shifted_ids=shifted.to(tb::kCUDA);
        auto original_ids=ids.clone();

        begin_gguf_speculative_routes(cache,rows);
        tb::Tensor output;
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream(),true);
        graph.set_profile_geometry("moe_overlap",rows);
        graph.capture([&]{graph.mark(0,"window_begin");output=banks.forward(x,ids,weights);graph.mark(0,"window_end");},[]{});
        discard_gguf_speculative_routes(cache);
        const auto before=gguf_expert_streamed_bytes(cache);
        std::map<std::string,double> cpu_before;
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))cpu_before[name]=value;
        const auto started=std::chrono::steady_clock::now();
        constexpr int repetitions=50;
        for(int i=0;i<repetitions;++i) {
            if(rotating)ids.copy_(i&1?shifted_ids:original_ids);
            begin_gguf_speculative_routes(cache,rows);graph.run();discard_gguf_speculative_routes(cache);
        }
        const auto ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()/repetitions;
        if(rotating)require(gguf_expert_streamed_bytes(cache)>before,"rotating overlap benchmark transferred no expert weights");
        std::printf("moe_overlap_bench rows=%d half=%d ms=%.6f bytes_per_window=%llu\n",rows,int(half),ms,
            static_cast<unsigned long long>((gguf_expert_streamed_bytes(cache)-before)/repetitions));
        const auto host=output.to(tb::kFloat32).cpu();
        for(int i=0;i<host.numel();++i)require(std::isfinite(host.data_ptr<float>()[i]),"GGUF overlap benchmark nonfinite result");
        for(const auto& [name,value]:gguf_expert_memory_metrics(cache))
            if(name=="gguf_cpu_experts" || name=="gguf_cpu_work_ms" || name=="gguf_cpu_queue_ms" || name.starts_with("gguf_cpu_auto_") || name.starts_with("gguf_host_prepare_") || name.starts_with("gguf_decode_dma_"))
                std::printf("moe_overlap_cpu name=%s delta=%.6f\n",name.c_str(),value-cpu_before[name]);
    }
}
}
#include "gguf_mmq_swiglu_test.h"

void run_row_correction(const char* dense_file,const char* moe_file) {
    const mfq::RowResidualFile fixture(dense_file);
    const auto rows=std::int64_t(fixture.row_count());
    require(fixture.columns()==256,"GGUF Q3_K fixture must have complete 256-value blocks");
    const auto close=[](const tb::Tensor& actual,const tb::Tensor& expected,const char* message) {
        const auto error=(actual.to(tb::kFloat32)-expected.to(tb::kFloat32)).abs().max().item<float>();
        require(std::isfinite(error) && error<=2e-4f*(1+expected.abs().max().item<float>()),message);
    };
    for(auto format:{GGML_TYPE_Q3_K,GGML_TYPE_TQK6,GGML_TYPE_TQP6})for(bool rotated:{false,true}) {
        Source source;source.add("weight",{rows,256},format);source.add("auxiliary",{rows,256},format);
        if(format>=GGML_TYPE_TQ2_T) {source.full_compact_codes("weight",format);source.full_compact_codes("auxiliary",format);}
        if(rotated)source.add_hadamard("weight");
        const auto frozen=source.read("weight");source.add_correction("weight",dense_file);
        auto weight=load_gguf_weight(source,"weight",mfq_current_cuda_device());
        auto raw=weight;raw.correction.reset();
        require(bool(weight.correction),"GGUF loader did not attach the row correction");
        const auto delta=weight.correction->dense(mfq_current_cuda_device());
        close(weight.dequantize(),raw.dequantize()+delta,"GGUF corrected dequantization differs");
        auto cpu=load_gguf_weight(source,"weight",0,false);
        close(cpu.dequantize().to(tb::kCUDA),weight.dequantize(),"CPU and CUDA residual weights differ");
        for(int tokens:{1,2,3,4,33})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
            auto x=input(tokens,1,false).to(dtype);
            const auto expected=(raw.forward(x).to(tb::kFloat32)+tb::matmul(x.to(tb::kFloat32),delta.transpose(0,1))).to(dtype);
            close(weight.forward(x),expected,"GGUF residual forward bypassed or double-applied correction");
            if(tokens<=4) {
                auto auxiliary=load_gguf_weight(source,"auxiliary",mfq_current_cuda_device());
                const auto outputs=project_gguf_group({weight,auxiliary},x);
                close(outputs[0],expected,"parallel GGUF projection omitted correction");
                close(outputs[1],auxiliary.forward(x),"residual changed an uncorrected projection");
            }
        }
        auto ids=tb::tensor(std::vector<std::int64_t>{rows-1,0,rows/2},tb::TensorOptions().dtype(tb::kInt64)).to(tb::kCUDA);
        close(weight.embedding(ids),weight.dequantize().index_select(0,ids),"GGUF residual embedding differs");
        if(!rotated) {
            auto table=std::make_shared<GgufRowTable>(source,"weight");
            const auto expected=weight.dequantize().index_select(0,ids);
            close(table->lookup(ids),expected.to(tb::kFloat16),"staged GGUF lookup omitted neuron correction");
            close(table->lookup(ids.cpu()).to(tb::kCUDA),expected,"CPU GGUF lookup omitted neuron correction");
            auto stage=make_gguf_row_stage({table},rows,{3},ids.device());
            stage->upload({rows-1,0,rows/2},{3});stage->decode();
            close(stage->output(),expected.to(tb::kFloat16),"corrected row stage differs");
            const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
            const auto stream=mfq_current_cuda_stream();
            const bool owns_pool=!context->has_graph_pool(stream);
            if(owns_pool)context->begin_graph_pool(stream);
            struct PoolEnd {
                std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;bool own;
                ~PoolEnd(){if(own)context->end_graph_pool(stream);}
            } end_pool{context,stream,owns_pool};
            mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
            graph.capture([&]{stage->decode();},[]{});
            stage->upload({0,1,rows-1},{3});graph.run();
            auto next=tb::tensor(std::vector<std::int64_t>{0,1,rows-1},tb::TensorOptions().dtype(tb::kInt64)).to(tb::kCUDA);
            close(stage->output(),weight.dequantize().index_select(0,next).to(tb::kFloat16),
                "corrected row stage graph retained stale rows");
        }
        require(source.read("weight")==frozen,"residual fitting/loading changed the base checkpoint");
    }
    Source source;
    for(const auto* role:{"gate","up","down"}) {
        source.add(role,{3,256,256},GGML_TYPE_Q8_0);source.add_correction(role,moe_file);
    }
    CudaExecutionContext execution;execution.config.moe_host_cache_bytes=4ULL<<20;execution.config.moe_device_cache_bytes=0;
    GgufMoeBanks banks{std::make_shared<GgufRoutedWeight>(execution,source,"gate"),
        std::make_shared<GgufRoutedWeight>(execution,source,"up"),std::make_shared<GgufRoutedWeight>(execution,source,"down")};
    const auto reference=[&](const std::shared_ptr<GgufRoutedWeight>& bank,const tb::Tensor& x,const tb::Tensor& ids) {
        auto result=bank->forward_base(x,ids).to(tb::kFloat32).cpu().contiguous();
        auto selected=ids.cpu(),activation=x.to(tb::kFloat32).cpu().contiguous(),delta=bank->correction->dense(-1);
        for(std::int64_t slot=0;slot<ids.numel();++slot)for(std::int64_t row=0;row<256;++row) {
            const auto expert=selected.data_ptr<std::int64_t>()[slot];double sum=0;
            const auto* values=delta.data_ptr<float>()+(expert*256+row)*256;
            const auto* input=activation.data_ptr<float>()+(x.dim()==3?slot:slot/ids.size(1))*256;
            for(int column=0;column<256;++column)sum+=double(input[column])*values[column];
            result.data_ptr<float>()[slot*256+row]+=float(sum);
        }
        return result.to(x.device()).to(x.scalar_type());
    };
    for(int tokens:{1,2,3,4,33}) {
        auto x=input(tokens,3,false).to(tb::kFloat16),ids=routes(tokens,1);
        const auto weights=tb::full({tokens,3},.2,x.options().dtype(tb::kFloat32));
        const auto actual=banks.forward(x,ids,weights,.2f);
        auto g=tb::clamp_max(reference(banks.gate,x,ids).to(tb::kFloat32),.2f);
        auto u=tb::clamp(reference(banks.up,x,ids).to(tb::kFloat32),-.2f,.2f);
        auto hidden=((g*tb::sigmoid(g))*u).to(x.scalar_type());
        auto expected=moe_weighted_reduce_cuda(reference(banks.down,hidden,ids),weights);
        close(actual,expected,"GGUF three-bank MoE residual differs from independent neuron correction");
    }
    std::cout<<"GGUF row residual: Q3_K/TQK/TQP, rotations, grouped projections, embeddings and routed Gate/Up/Down PASS\n";
}

int main(int argc,char** argv) {
    if(argc==2 && std::strcmp(argv[1],"--q2-packed-affine-check")==0) {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        try{run_q2_packed_affine();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--quad-token-vector-check")==0) {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        try{run_quad_token_vectors();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--residual-rms-order-check")==0) {
        try{run_residual_rms_order();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--quad-decode-check")==0) {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        try{run_quad_decode_views();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--corrected-normalized-input-check")==0) {
        try{run_corrected_normalized_input();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--quad-storage-check")==0) {
        std::cout.setf(std::ios::unitbuf);
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        try{run_quad_compute_storage();run_iq2_fractional_cpu_reference();run_quad_token_vectors();run_q2_packed_affine();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--m1-affine-k-check")==0) {
        try{run_m1_affine_k();return 0;}
        catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--dense-parallel-projection-check")==0) {
        try{run_dense_projection_overlap();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--aligned-m1-check")==0) {
        try{run_aligned_m1();return 0;}
        catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--dense-format-pair-check")==0) {
        try{run_dense_format_pairs();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--long-dense-pair-check")==0) {
        try{run_long_dense_pairs();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--normalized-input-check")==0) {
        try{run_fused_normalized_input();run_fused_residual_normalized_input();run_corrected_normalized_input();run_weight_packet_preferences();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--row-correction-check")==0) {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        try {
            RowResidualFixture dense(10,256),moe(768,256);
            run_row_correction(dense.path().string().c_str(),moe.path().string().c_str());return 0;
        } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==4 && std::strcmp(argv[1],"--row-correction-check")==0) {
        try{run_row_correction(argv[2],argv[3]);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }

    if(argc==2 && std::strcmp(argv[1],"--cpu-queue-check")==0) {
        try{run_cpu_executor_queue();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--decode-budget-check")==0) {
        try{run_slot_transition_ram_coverage();run_decode_seeded_slots();run_lazy_decode_slots();run_lazy_admission_retry();run_reusable_expert_pool_budget();run_decode_workspace_budget();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--cpu-quantize-check")==0) {
        try{gguf_cpu_quantize_test::check();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--cpu-quantize-bench")==0) {
        try{gguf_cpu_quantize_test::bench();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--cpu-activation-check")==0) {
        try{gguf_cpu_activation_test::check();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--cpu-activation-bench")==0) {
        try{gguf_cpu_activation_test::bench();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--recurrent-copy-check")==0) {
        try{check_recurrent_copy_batch();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==5 && std::strcmp(argv[1],"--recurrent-copy-bench")==0) {
        try{bench_recurrent_copy_batch(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    // This CLI exercises only host projections and must not require a GPU.
    if(argc==6 && std::strcmp(argv[1],"--cpu-projection-bench")==0) {
        try{bench_cpu_projection(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::string(argv[1])=="--speculative-journal-check") {
        try{gguf_journal_test::check();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    std::cout<<std::unitbuf;
    if(argc==2 && std::strcmp(argv[1],"--iq-codebook-check")==0) {
        try{run_gguf_iq_codebooks();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }

    if(argc==2 && std::strcmp(argv[1],"--m1-residual-check")==0) {
        try{run_m1_residual();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }


    if(argc==1) {
        try{gguf_input_test::check();gguf_rotation_test::check();gguf_resident_prepare_test::check();run_graph_expert_batches();run_compact_projection_runtime();run_compact_moe_runtime();run_hadamard_projection_runtime();run_hadamard_moe_runtime();run_gguf_residual_mix_runtime();run_gguf_residual_pre_runtime();}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--decode-dma-check")==0) {
        try{run_decode_dma();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==5 && std::strcmp(argv[1],"--residual-activation-bench")==0) {
        try{run_gguf_residual_activation(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));return 0;}
        catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--compact-projection-check")==0) {
        try{run_compact_projection_runtime();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::string(argv[1])=="--residual-pre-check") {
        try{run_gguf_residual_pre_runtime();return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--gguf-residual-mix-check")==0) {
        try{run_gguf_residual_mix_runtime();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--compact-moe-check")==0) {
        try{run_compact_moe_runtime();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==4 && std::strcmp(argv[1],"--gguf-file-residual-bench")==0) {
        try{bench_file_residual_mix(argv[2],std::stoi(argv[3]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==3 && std::strcmp(argv[1],"--gguf-file-decoded-input-bench")==0) {
        try{bench_file_moe_projection(argv[2],false,true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==3 && std::strcmp(argv[1],"--gguf-file-shared-bench")==0) {
        try{bench_file_moe_projection(argv[2],true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==3 && std::strcmp(argv[1],"--gguf-file-routed-bench")==0) {
        try{bench_file_moe_projection(argv[2]);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==3 && std::strcmp(argv[1],"--gguf-file-moe-check")==0) {
        try{run_file_moe_runtime(argv[2]);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--hadamard-moe-gpu-check")==0) {
        try{run_hadamard_moe_runtime();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--hadamard-moe-check")==0) {
        try{run_hadamard_moe_runtime();run_cpu_experts(true);std::puts("Hadamard CPU experts and GPU merge PASS");std::fflush(stdout);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==4 && std::strcmp(argv[1],"--hadamard-fused-bench")==0) {
        try{gguf_rotation_test::bench(std::stoi(argv[2]),std::stoi(argv[3]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--hadamard-fused-check")==0) {
        try{gguf_rotation_test::check();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--hadamard-projection-check")==0) {
        try{run_hadamard_projection_runtime();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--moe-input-check")==0) {
        try{gguf_input_test::check();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--resident-prepare-check")==0) {
        try{gguf_resident_prepare_test::check();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--graph-expert-batch-check")==0) {
        try{run_graph_expert_batches();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==5 && std::strcmp(argv[1],"--resident-prepare-bench")==0) {
        try{gguf_resident_prepare_test::run(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==5 && std::strcmp(argv[1],"--moe-input-bench")==0) {
        try{gguf_input_test::bench(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && (std::strcmp(argv[1],"--mmq-swiglu-check")==0 || std::strcmp(argv[1],"--mmq-swiglu-bench")==0)) {
        try{run_mmq_swiglu_quantize(std::strcmp(argv[1],"--mmq-swiglu-bench")==0);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<std::endl;return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--resident-predictor-check")==0) {
        try{run_resident_predictor();run_resident_predictor(true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--prepared-projection-check")==0) {
        try{run_prepared_dense_projection();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--interleaved-projection-check")==0) {
        try{run_interleaved_dense_projection();run_shared_quantization();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--projection-batch-check")==0) {
        try{for(int width:{768,2560,2816,8192,16640})run_shared_quantization(width);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--shared-quantization-check")==0) {
        try{run_shared_quantization();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--iq4-decode-consistency-check")==0) {
        int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        try{run_interleaved_dense_projection();run_iq4_group_consistency();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--gated-input-check")==0) {
        try{run_fused_dense_activation();run_fused_rms_activation();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--small-f32-projection-check")==0) {
        try{run_small_f32_projection();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--parallel-projection-check")==0) {
        try {
            if(!mfq::cuda::default_context(mfq_current_cuda_device())->supports_async_allocations())return 77;
            run_parallel_projection_group();return 0;
        }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--projection-subset-check")==0) {
        try{run_projection_subset_check();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if((argc==6 || argc==7) && std::strcmp(argv[1],"--projection-branch-bench")==0) {
        try {
            if(argc==7 && std::strcmp(argv[6],"auxiliary-only"))throw std::invalid_argument("unknown mixed projection benchmark mode");
            bench_projection_branches(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),argc==7);return 0;
        }
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==6 && std::strcmp(argv[1],"--dense-io-bench")==0) {
        try{bench_dense_io(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==6 && std::strcmp(argv[1],"--dense-position-bench")==0) {
        try{bench_dense_io(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==6 && std::strcmp(argv[1],"--dense-small-mmq-bench")==0) {
        try{bench_dense_small_mmq(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }


    if(argc==8 && std::strcmp(argv[1],"--mmq-layer-quant-bench")==0) {
        try{bench_mmq_layer_quant(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--prefill-stream-check")==0) {
        try{run_scaled_dense_prefill();run_streamed_dense_prefill();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--prefill-precision-check")==0) {
        try{run_scaled_dense_prefill();run_streamed_dense_prefill();run_prefill_precision();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--mmq-scatter-check")==0) {
        try{run_mmq_products();run_mmq_scatter();run_mmq_moe();run_prefill_arena_reuse();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--prefill-metadata-check")==0) {
        try{run_prefill_arena_reuse();run_borrowed_gguf_prefill();run_borrowed_gguf_prefill(true);run_combined_moe();
            std::cout<<"GGUF prefill metadata/uneven groups/mixed formats/loans PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--trace-routing-check")==0) {
        try{run_strata_adaptive_slots();run_unmanaged_bank_learning();run_transfer_workspace_reuse();run_gpu_residency_snapshot();run_speculative_admission();run_fixed_speculative_admission();run_host_event_retirement();run_borrowed_gguf_prefill();run_admission_failure_and_lifetime();
            std::cout<<"GGUF routing/speculative/admission/loan checks PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--event-retirement-check")==0) {
        try{run_paired_host_cache();run_host_event_retirement();run_admission_failure_and_lifetime();
            std::cout<<"GGUF shared fence queries/graph retirement/admission failure/lifetime PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--prefill-phase-check")==0) {
        try{run_prefill_partial_fence();run_paired_host_cache();run_host_snapshot(true);run_mmq_moe(true);
            std::cout<<"GGUF phased PP pinned/pageable/guarded MMQ/decode fences PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--mmq-schedule-check")==0) {
        try{run_mmq_moe();run_mmq_moe(true);
            std::cout<<"GGUF MMQ unequal expert counts/multiple groups/tail/mixed+joined reference PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--expert-image-check")==0) {
        try{run_prefill_host_scan_bypass();run_host_layout_plan();run_paired_host_cache();run_host_kv_sharing();run_pinned_hole_under_pressure();run_pinned_interval_compaction();run_fragmented_pinned_fallback();run_fragmented_pinned_fallback(true);run_host_vram_evacuation();run_pinned_range_trim();run_fragmented_host_reclaim();run_fragmented_host_reclaim(true);run_host_snapshot();run_host_snapshot(true);run_host_snapshot(false,true);
            run_inplace_mmq_moe();run_pinned_copy_plan();
            std::cout<<"GGUF expert-major host/PP output/eviction/fallback checks PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==7 && std::strcmp(argv[1],"--projection-layout-bench")==0) {
        try{bench_projection_layout(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),std::stoi(argv[6]));return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--layer-admission-snapshot-check")==0) {
        try{run_layer_admission_snapshot();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--row-graph-check")==0) {
        try{run_row_graph_upload();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--layer-admission-check")==0) {
        try{run_layer_admission_snapshot();run_layer_admission_windows();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--layer-admission-smoke-check")==0) {
        try{run_layer_admission_snapshot();run_layer_admission_windows(true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--shared-overlap-check")==0) {
        try{run_shared_work_overlap();return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--shared-stream-check")==0) {
        try{run_shared_work_overlap(true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--shared-stream-resident-check")==0) {
        try{run_shared_work_overlap(true,true);return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--cpu-auto-check")==0) {
        try{run_cpu_auto_balance();return 0;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--mapped-wait-check")==0) {
        try{run_mapped_flag_wait();run_timed_mapped_flag_wait();return 0;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc==2 && std::strcmp(argv[1],"--cpu-experts-check")==0) {
        try{run_cpu_experts();std::cout<<"CPU experts precision/routes/graph/merge checks PASS\n";return 0;}
        catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    if(argc>=2 && (std::strcmp(argv[1],"--resident-overlap-bench")==0 || std::strcmp(argv[1],"--resident-overlap-rotating-bench")==0)) {
        try{bench_resident_overlap(argc>2?std::stoi(argv[2]):0,argc>3 && std::string(argv[3])=="half",std::strcmp(argv[1],"--resident-overlap-rotating-bench")==0);return 0;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    try{run_mapped_flag_wait();run_timed_mapped_flag_wait();run_cpu_auto_balance();run_mmq_scatter();run_split_expert_copy();run_host_snapshot();run_host_snapshot(true);run_mmq_products();run_mmq_moe();run_compact_mmq_gather();run_inplace_mmq_moe();run_strata_adaptive_slots();run_pinned_copy_plan();run_decode_dma();run_resident_pool_separation();run_fused_expert_reduce();run_cpu_experts();run_borrowed_gguf_prefill();run_borrowed_gguf_prefill(true);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    try { run_residual_rms_order();run();run_weight_packet_preferences();run_corrected_normalized_input();run_gguf_iq_codebooks();run_m1_residual();run_small_f32_projection();run_m1_affine_k();run_aligned_m1();run_dense_format_pairs();run_long_dense_pairs();run_dense_projection_overlap();run_fused_normalized_input();run_fused_residual_normalized_input();run_fused_dense_activation();run_fused_rms_activation();run_scaled_dense_prefill();run_fused_hidden_quantization(); run_combined_moe(); run_combined_moe(true);run_combined_moe(false,true);run_prefill_arena_reuse();run_failed_expert_batch(); run_global_prefill_admission(); run_projection_group();run_projection_group(false);run_combined_row_storage();run_shared_quantization(); run_direct_chain(); run_exclusive_cache();run_exclusive_cache(true); run_multi_packet(); run_prompt_admission();run_speculative_admission();run_fixed_speculative_admission();run_host_event_retirement();run_admission_failure_and_lifetime(); std::cout << "GGUF cache/prefill/graph/row-stage/slicing/combined-MoE/exclusive-cache/multi-packet/prompt+speculative-admission passed\n"; }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
