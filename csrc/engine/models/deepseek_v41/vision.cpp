#include "engine/models/source.h"
#include "engine/backends/cuda/executor/execution.h"
#include "vision.h"
#include "engine/models/common/vision/axial_patch_encoder.h"
#include "engine/state/session/prefix_cache.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

namespace tyloi::cuda::deepseek_v41_runtime {
namespace {

DenseTensor vector_weight(const weight_loader::Loader& weights,
                     const std::string& name,int64_t width) {
    auto value=dynamic_cast<const ::tyloi::DenseTensor&>(*weights.tensor(name));
    TYLOINFER_RUNTIME_CHECK(value.dim()==1 && value.size(0)==width,"DeepSeek-V4.1 vision vector shape mismatch: "+name);
    return value;
}

struct Span {int64_t begin,end,height,width,rows,columns;};
std::vector<Span> validate_media(const CommonConfig& config,const std::vector<int64_t>& tokens,
                                 const MfqMultimodalInput& media) {
    const auto count=media.image_bounds.size()/4;
    const auto patch_width=3*config.vision.patch_size*config.vision.patch_size;
    if(media.processor!=MfqMultimodalProcessor::deepseek_v41 || media.processor_name!="deepseek_v41" ||
            !count || media.image_bounds.size()%4 || media.vision_grid.size()!=count*4 ||
            media.pixel_shape.size()!=3 || media.pixel_shape[0]!=int64_t(count) ||
            media.pixel_shape[1]<=0 || media.pixel_shape[2]!=patch_width ||
            media.vision_grid_shape!=std::vector<int64_t>{int64_t(count),4} ||
            !media.video_grid.empty() || tokens.empty())
        throw std::invalid_argument("DeepSeek-V4.1 vision payload geometry is invalid");
    const auto maximum=uint64_t(media.pixel_shape[1]);
    if(maximum>SIZE_MAX/uint64_t(patch_width)/count ||
            media.pixel_values.size()!=count*maximum*uint64_t(patch_width))
        throw std::invalid_argument("DeepSeek-V4.1 vision patch storage is invalid");
    for(const auto token:tokens)if(token<0 || token>=config.vocab)
        throw std::invalid_argument("DeepSeek-V4.1 vision prompt token is outside the vocabulary");
    std::vector<Span> spans;spans.reserve(count);int64_t previous=0;
    const auto ratio=config.vision.downsample_ratio;
    for(std::size_t i=0;i<count;++i) {
        const auto* bound=media.image_bounds.data()+4*i;
        const auto* grid=media.vision_grid.data()+4*i;
        const int64_t h=grid[0],w=grid[1],rows=grid[2],columns=grid[3];
        if(bound[0]!=0 || bound[1]!=int64_t(i) || h<=0 || w<=0 || h>int64_t(maximum)/w ||
                rows!=(h+ratio-1)/ratio || columns!=(w+ratio-1)/ratio ||
                bound[2]<previous || bound[3]>int64_t(tokens.size()) ||
                bound[3]-bound[2]!=rows*(columns+1)+2 || rows*columns>config.vision.max_image_tokens)
            throw std::invalid_argument("DeepSeek-V4.1 image span is invalid");
        for(auto position=bound[2];position<bound[3];++position)if(tokens[position]!=config.image_token_id)
            throw std::invalid_argument("DeepSeek-V4.1 image span contains a text token");
        spans.push_back({bound[2],bound[3],h,w,rows,columns});previous=bound[3];
    }
    return spans;
}
} // namespace

Vision Vision::load(std::shared_ptr<const tyloi::ModelSource> source,weight_loader::Loader weights,const CommonConfig& config) {
    TYLOINFER_RUNTIME_CHECK(source && config.has_vision() && config.image_token_id>=0 &&
        config.image_token_id<config.vocab && config.vision.hidden%config.vision.n_heads==0 &&
        (config.vision.hidden/config.vision.n_heads)%4==0,"invalid DeepSeek-V4.1 CUDA vision config");
    for(const auto* name:{"vision.patch_embedding.weight","vision.output_norm.weight",
            "vision.aligner.input.weight","vision.aligner.output.weight","vision.special_token.start",
            "vision.special_token.newline","vision.special_token.end"})
        TYLOINFER_RUNTIME_CHECK(has_tensor(*source,name),std::string("DeepSeek-V4.1 vision tensor missing: ")+name);
    return Vision(std::move(source),std::move(weights),config);
}

tyloi::StepSequence<Vision::DenseTensor> Vision::encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        DenseTensor patches,int height,int width) const {
    const auto& v=config_.vision;
    const tyloi::models::axial_patch_vision::Config config{
        v.hidden,v.n_layers,v.n_heads,v.intermediate,v.patch_size,
        v.downsample_ratio,config_.hidden,v.rope_theta,1e-6};
    auto tower=tyloi::cuda::axial_patch_vision::CudaAxialPatchVisionTower::load(weights_,config);
    auto steps=tower.encode_steps(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        projector,
#endif
        std::move(patches),height,width);
    while(auto step=steps.next()) {
        if(step.value)co_yield std::move(*step.value);
        else co_yield step.state;
    }
}

tyloi::StepSequence<PreparedPrompt> Vision::prepare(deepseek_v41::Model& language, MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const {
    auto spans=validate_media(config_,tokens,media);
    auto options=::tyloi::TensorOptions().dtype(::tyloi::kInt64).device(::tyloi::kCUDA);
    auto ids=::tyloi::cuda::tensor(tokens,options).reshape({1,-1});
    auto embeddings=dynamic_cast<const DenseTensor&>(*language.embedding(ids)).contiguous();
    std::vector<uint8_t> mask(tokens.size(),0);
    std::string request_key="deepseek_v41|";
    for(std::size_t i=0;i<spans.size();++i) {
        const auto& span=spans[i];
        const auto patch_width=media.pixel_shape[2];
        const auto* first=media.pixel_values.data()+int64_t(i)*media.pixel_shape[1]*patch_width;
        auto key=tyloi::cache::block_hash_hex(tyloi::cache::sha256(first,
            std::size_t(span.height*span.width*patch_width)*sizeof(float)))+"|"+
            std::to_string(span.height)+"|"+std::to_string(span.width);
        DenseTensor visual(images.find(key));
        if(!visual.defined()) {
            auto patches=::tyloi::cuda::from_blob(const_cast<float*>(first),{span.height*span.width,patch_width},
                ::tyloi::TensorOptions().dtype(::tyloi::kFloat32).device(::tyloi::kCPU))
                .to(::tyloi::kCUDA).to(::tyloi::kFloat16);
            auto encoding=encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                projector,
#endif
                std::move(patches),int(span.height),int(span.width));
            while(auto step=encoding.next()){if(step.value)visual=std::move(*step.value);else co_yield step.state;}
            images.store(key,visual);
        }
        TYLOINFER_RUNTIME_CHECK(visual.dim()==2 && visual.size(0)==span.rows*span.columns && visual.size(1)==config_.hidden,
                          "DeepSeek-V4.1 aligned image shape mismatch");
        DenseTensor start,newline,end;
        {const auto host=weights_.for_layer(weights_.placement.device,true);
            start=vector_weight(host,"vision.special_token.start",config_.hidden).to(::tyloi::kCUDA);
            newline=vector_weight(host,"vision.special_token.newline",config_.hidden).to(::tyloi::kCUDA);
            end=vector_weight(host,"vision.special_token.end",config_.hidden).to(::tyloi::kCUDA);
        }
        std::vector<DenseTensor> rows{start.reshape({1,-1}).to(visual.scalar_type())};
        for(int64_t row=0;row<span.rows;++row) {
            rows.push_back(visual.narrow(0,row*span.columns,span.columns));
            rows.push_back(newline.reshape({1,-1}).to(visual.scalar_type()));
        }
        rows.push_back(end.reshape({1,-1}).to(visual.scalar_type()));
        embeddings.narrow(1,span.begin,span.end-span.begin).copy_(
            ::tyloi::cuda::cat(rows,0).unsqueeze(0).to(embeddings.scalar_type()));
        std::fill(mask.begin()+span.begin,mask.begin()+span.end,1);
        request_key+=key+"|"+std::to_string(span.begin)+"|"+std::to_string(span.end)+";";
        co_yield tyloi::StepState::advanced;
    }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    PreparedPrompt prepared{tokens,std::make_shared<DenseTensor>(std::move(embeddings)),{},0,std::move(request_key)};
    auto mask_tensor=tyloi::ops::tensor(mask,tyloi::Device{tyloi::DeviceType::cpu,0});
    prepared.image_mask=tyloi::ops::reshape(*mask_tensor,{1,-1});
#else
    PreparedPrompt prepared{tokens,std::move(embeddings),{},0,std::move(request_key)};
    prepared.image_mask=::tyloi::cuda::tensor(mask,::tyloi::TensorOptions()
        .dtype(::tyloi::kUInt8).device(::tyloi::kCPU)).reshape({1,-1});
#endif
    static std::atomic_flag reported=ATOMIC_FLAG_INIT;
    if(!reported.test_and_set())std::fprintf(stderr,"deepseek_v41_vision_dispatch enabled=1 images=%zu weights=streamed engram=text_only router=modality_bias\n",spans.size());
    co_yield std::move(prepared);
}
} // namespace tyloi::cuda::deepseek_v41_runtime
