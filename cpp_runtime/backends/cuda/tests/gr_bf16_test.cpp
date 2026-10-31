#include "float_projection.h"
#include "quant_linear.h"
#include "cuda_execution.h"
#include "gated_residual.h"
#include "qwen4_exp.h"
#include "runtime/decode_window.h"
#include "mfq_cuda_shared_gate.h"
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <array>
#include <cstring>

namespace tb=mfq_tensor_backend;
namespace {
tb::Tensor pattern(int64_t count,float phase) {
    auto x=tb::empty({count},tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for(int64_t i=0;i<count;++i)x.data_ptr<float>()[i]=std::sin(float(i)*.17f+phase)*.08f;
    return x.to(tb::kCUDA);
}
void check_prefill(int width,int outputs,int rows,bool strided) {
    auto x=pattern(int64_t(rows)*width+1,.21f).narrow(0,1,int64_t(rows)*width).reshape({1,rows,width});
    auto w=pattern(int64_t(outputs)*width+1,.43f).to(tb::kBFloat16).narrow(0,1,int64_t(outputs)*width).reshape({outputs,width});
    if(strided){x=x.transpose(1,2).contiguous().transpose(1,2);w=w.transpose(0,1).contiguous().transpose(0,1);}
    auto actual=try_gr_bf16_prefill_projection_cuda(x,w);
    int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,x.get_device()));
    const auto* setting=std::getenv("TYLOINFER_GR_PREFILL_BF16");
    if(major<8 || (setting && setting[0]=='0')) {
        if(actual.defined())throw std::runtime_error("BF16 GR prefill ignored the capability/explicit override");
        return;
    }
    if(!actual.defined() || actual.scalar_type()!=tb::kFloat32)
        throw std::runtime_error("BF16 GR prefill rejected a valid tensor or rounded its FP32 output");
    auto source=x.to(tb::kBFloat16).to(tb::kFloat32).contiguous().cpu();
    auto weight=w.to(tb::kFloat32).contiguous().cpu(),values=actual.cpu();
    for(int m=0;m<rows;++m)for(int n=0;n<outputs;++n) {
        double expected=0,magnitude=0;
        for(int k=0;k<width;++k) {
            const double term=double(source.data_ptr<float>()[m*width+k])*weight.data_ptr<float>()[n*width+k];
            expected+=term;magnitude+=std::abs(term);
        }
        if(std::abs(values.data_ptr<float>()[m*outputs+n]-expected)>magnitude*2e-5+1e-6)
            throw std::runtime_error("BF16 GR prefill differs from Strata-precision CPU dot");
    }
}
void check_dense(int width,int outputs,int rows,tb::ScalarType dtype) {
    auto x=pattern(int64_t(rows)*width+1,.27f).to(dtype).narrow(0,1,int64_t(rows)*width).reshape({1,rows,width});
    auto weight=pattern(int64_t(outputs)*width+1,.59f).to(tb::kBFloat16).narrow(0,1,int64_t(outputs)*width).reshape({outputs,width});
    const auto actual=try_dense_bf16_projection_cuda(x,weight);
    if(!actual.defined() || actual.scalar_type()!=(dtype==tb::kFloat32?tb::kFloat32:tb::kFloat16))
        throw std::runtime_error("BF16 dense projection rejected a window or changed its output precision");
    const auto source=x.to(tb::kBFloat16).to(tb::kFloat32).cpu();
    const auto weights=weight.to(tb::kFloat32).cpu(),values=actual.to(tb::kFloat32).cpu();
    for(int m=0;m<rows;++m)for(int n=0;n<outputs;++n) {
        double sum=0,magnitude=0;
        for(int k=0;k<width;++k) {
            const double v=double(source.data_ptr<float>()[m*width+k])*weights.data_ptr<float>()[n*width+k];
            sum+=v;magnitude+=std::abs(v);
        }
        if(std::abs(values.data_ptr<float>()[m*outputs+n]-sum)>.004*magnitude+1e-6)
            throw std::runtime_error("BF16 dense projection differs from rounded-input CPU dot");
    }
    if((actual.to(tb::kBFloat16).to(tb::kFloat32)-actual.to(tb::kFloat32)).abs().max().item<float>()!=0)
        throw std::runtime_error("BF16 dense projection skipped its original BF16 output rounding");
    const auto integrated=dense_projection(x,weight);
    if((integrated.to(tb::kFloat32)-actual.to(tb::kFloat32)).abs().max().item<float>()!=0)
        throw std::runtime_error("BF16 dense projection was not integrated into the common adapter");
    for(int m=0;m<rows;++m) {
        const auto single=try_dense_bf16_projection_cuda(x.narrow(1,m,1),weight);
        if((single.to(tb::kFloat32)-actual.narrow(1,m,1).to(tb::kFloat32)).abs().max().item<float>()!=0)
            throw std::runtime_error("BF16 dense projection changed a token when grouped with other tokens");
    }
}
void check_prefill_mix(int hidden,int rank,int streams,int rows,bool strided,bool graph=false) {
    auto input=pattern(int64_t(rows)*rank+1,.17f).narrow(0,1,int64_t(rows)*rank).reshape({1,rows,rank});
    auto weight=pattern(int64_t(hidden)*streams*rank+1,.31f).to(tb::kBFloat16)
        .narrow(0,1,int64_t(hidden)*streams*rank).reshape({hidden*streams,rank});
    auto norm=pattern(int64_t(rows)*hidden*streams+1,.53f).narrow(0,1,int64_t(rows)*hidden*streams)
        .reshape({1,rows,hidden*streams});
    if(strided) {
        input=input.transpose(1,2).contiguous().transpose(1,2);
        weight=weight.transpose(0,1).contiguous().transpose(0,1);
        norm=norm.transpose(1,2).contiguous().transpose(1,2);
    }
    for(bool half:{false,true}) {
        const auto normalized=half?norm.to(tb::kFloat16):norm;
        const auto compute=[&]{return try_gr_bf16_up_mix_cuda(input,normalized,weight,streams);};
        auto output=compute();
        int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
        const auto* setting=std::getenv("TYLOINFER_GR_PREFILL_BF16");
        if(major<8 || (setting && setting[0]=='0')) {
            if(output.defined())throw std::runtime_error("BF16 GR prefill mix ignored capability/explicit disable");
            continue;
        }
        const auto compare=[&](const tb::Tensor& actual) {
            const auto projected=try_gr_bf16_prefill_projection_cuda(input,weight);
            const auto expected=mfq_qwen4_exp::gated_residual_mix(projected,normalized.contiguous(),streams);
            if(!actual.defined() || actual.scalar_type()!=tb::kFloat32 || actual.sizes()!=expected.sizes())
                throw std::runtime_error("BF16 GR prefill mix rejected valid shape or changed output dtype");
            const auto error=(actual-expected).abs().max().item<float>();
            const auto scale=normalized.abs().max().item<float>();
            if(!std::isfinite(error) || error>scale*2e-5f+1e-7f)
                throw std::runtime_error("BF16 GR prefill fused projection/mix differs from independent GEMM/mix");
        };
        compare(output);
        if(graph) {
            const auto context=mfq::cuda::default_context(input.get_device());
            context->begin_graph_pool(mfq_current_cuda_stream());
            struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;
                ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}} pool_end{context};
            mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
            window.capture([&]{output=compute();},[]{});
            for(float value:{.001f,-.007f}) {input.fill_(value);window.run();compare(output);}
        }
    }
}
std::vector<tb::Tensor> prefill_read_reference(const tb::Tensor& input,const tb::Tensor& norm,
        const tb::Tensor& down,const tb::Tensor& up,const tb::Tensor& injection,int streams,
        const tb::Tensor& branch={},const tb::Tensor& prior={}) {
    tb::Tensor updated=input,normalized;
    if(branch.defined()) {
        auto p=mfq_qwen4_exp::gated_residual_post_norm(branch,input,prior,norm,streams,1e-6);
        updated=p[0];normalized=p[1];
    } else normalized=mfq_qwen4_exp::grouped_rms_norm(input,norm,input.size(-1)/streams,1e-6);
    const auto project=[](const tb::Tensor& x,const tb::Tensor& w) {
        if(auto result=try_gr_bf16_prefill_projection_cuda(x,w);result.defined())return result;
        return tb::matmul(x.to(tb::kFloat32),w.transpose(0,1).to(tb::kFloat32));
    };
    auto low=mfq_qwen4_exp::gated_residual_bottleneck(project(normalized,down),streams);
    auto mixed=mfq_qwen4_exp::gated_residual_mix(project(low,up),normalized,streams);
    return {mixed,updated,mfq_qwen4_exp::gated_residual_injection(project(normalized,injection),streams)};
}
void check_prefill_read(int hidden,int rank,int streams,int rows,bool after,bool strided,bool graph=false,int types=3) {
    const int width=hidden*streams;
    auto input=pattern(int64_t(rows)*width+1,.17f).narrow(0,1,int64_t(rows)*width).reshape({1,rows,width});
    if(!(types&2))input=input.to(tb::kFloat16);
    auto norm=pattern(width,.23f);
    const auto matrix=[&](int n,int k,float phase) {
        return pattern(int64_t(n)*k+1,phase).to(tb::kBFloat16).narrow(0,1,int64_t(n)*k).reshape({n,k});
    };
    auto down=matrix(rank,width,.31f),up=matrix(width,rank,.43f),injection=matrix(streams,width,.59f);
    auto branch=after?pattern(int64_t(rows)*hidden,.67f).to(types&4?tb::kFloat32:tb::kFloat16).reshape({1,rows,hidden}):tb::Tensor{};
    auto prior=after?pattern(int64_t(rows)*streams,.71f).to(types&1?tb::kFloat32:tb::kFloat16).reshape({1,rows,streams}):tb::Tensor{};
    if(strided) {
        input=input.transpose(1,2).contiguous().transpose(1,2);
        down=down.transpose(0,1).contiguous().transpose(0,1);
        up=up.transpose(0,1).contiguous().transpose(0,1);
    }
    const auto compute=[&]{return try_gated_residual_bf16_cuda(input,norm,down,up,injection,streams,1e-6,branch,prior);};
    auto actual=compute();
    int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    const auto* setting=std::getenv("TYLOINFER_GR_PREFILL_BF16");
    if(major<8 || (setting && setting[0]=='0')) {
        if(!actual.empty())throw std::runtime_error("compact GR prefill ignored capability/explicit disable");return;
    }
    const auto compare=[&](const auto& values) {
        const auto expected=prefill_read_reference(input,norm,down,up,injection,streams,branch,prior);
        if(values.size()!=3)throw std::runtime_error("compact GR prefill did not activate");
        for(int i=0;i<3;++i) {
            if(values[i].sizes()!=expected[i].sizes() || values[i].scalar_type()!=expected[i].scalar_type())
                throw std::runtime_error("compact GR prefill changed output shape/dtype");
            const auto error=(values[i]-expected[i]).abs().max().template item<float>();
            const auto scale=expected[i].abs().max().template item<float>();
            if(!std::isfinite(error) || error>(i==1?0.f:scale*1e-4f+2e-6f))
                throw std::runtime_error("compact GR prefill differs from independent normalization/projections/mix");
        }
    };
    compare(actual);
    if(graph) {
        const auto context=mfq::cuda::default_context(input.get_device());context->begin_graph_pool(mfq_current_cuda_stream());
        struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;
            ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}} pool_end{context};
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());window.capture([&]{actual=compute();},[]{});
        for(float value:{0.f,-.007f}) {input.fill_(value);window.run();compare(actual);}
    }
    norm.fill_(-1);actual=compute();
    if(actual[0].abs().max().item<float>()!=0 || (actual[2]-1).abs().max().item<float>()!=0)
        throw std::runtime_error("compact GR prefill lost norm offset/zero projection identity");
}
void benchmark_prefill_mix(int hidden,int rank,int streams,int rows,int sets,int iterations,bool full=false,bool after=false) {
    if(hidden<1 || rank<1 || streams<1 || rows<=32 || sets<1 || iterations<sets)
        throw std::invalid_argument("GR prefill mix benchmark dimensions must be positive with rows>32 and iterations>=sets");
    auto input=pattern(int64_t(rows)*(full?hidden*streams:rank),.17f).reshape({1,rows,full?hidden*streams:rank});
    auto normalized=full?tb::Tensor{}:pattern(int64_t(rows)*hidden*streams,.53f).reshape({1,rows,hidden*streams});
    auto norm=full?pattern(hidden*streams,.23f):tb::Tensor{};
    auto branch=after?pattern(int64_t(rows)*hidden,.67f).to(tb::kFloat16).reshape({1,rows,hidden}):tb::Tensor{};
    auto prior=after?pattern(int64_t(rows)*streams,.71f).reshape({1,rows,streams}):tb::Tensor{};
    std::vector<tb::Tensor> weights,old_output(sets),new_output(sets);
    std::vector<tb::Tensor> down,injection;
    std::vector<std::vector<tb::Tensor>> old_full(sets),new_full(sets);
    std::vector<std::unique_ptr<mfq::cuda::DecodeWindow>> old_graphs,new_graphs;
    for(int i=0;i<sets;++i)weights.push_back(pattern(int64_t(hidden)*streams*rank,.31f+i*.11f)
        .to(tb::kBFloat16).reshape({hidden*streams,rank}));
    if(full)for(int i=0;i<sets;++i) {
        down.push_back(pattern(int64_t(hidden)*streams*rank,.47f+i*.11f).to(tb::kBFloat16).reshape({rank,hidden*streams}));
        injection.push_back(pattern(int64_t(hidden)*streams*streams,.61f+i*.11f).to(tb::kBFloat16).reshape({streams,hidden*streams}));
    }
    const auto stream=mfq_current_cuda_stream();
    const auto context=mfq::cuda::default_context(input.get_device());context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~PoolEnd(){context->end_graph_pool(stream);}} pool_end{context,stream};
    for(int i=0;i<sets;++i) {
        auto old=std::make_unique<mfq::cuda::DecodeWindow>(stream);
        old->capture([&]{
            if(full){old_full[i]=prefill_read_reference(input,norm,down[i],weights[i],injection[i],streams,branch,prior);old_output[i]=old_full[i][0];}
            else old_output[i]=mfq_qwen4_exp::gated_residual_mix(
                try_gr_bf16_prefill_projection_cuda(input,weights[i]),normalized,streams);},[]{});
        auto current=std::make_unique<mfq::cuda::DecodeWindow>(stream);
        current->capture([&]{
            if(full){new_full[i]=try_gated_residual_bf16_cuda(input,norm,down[i],weights[i],injection[i],streams,1e-6,branch,prior);new_output[i]=new_full[i].at(0);}
            else new_output[i]=try_gr_bf16_up_mix_cuda(input,normalized,weights[i],streams);
            if(!new_output[i].defined())throw std::runtime_error("GR prefill mix benchmark did not select fused kernel");},[]{});
        old_graphs.push_back(std::move(old));new_graphs.push_back(std::move(current));
    }
    for(int i=0;i<sets;++i)if((old_output[i]-new_output[i]).abs().max().item<float>()>(full?1e-4f:3e-6f))
        throw std::runtime_error("GR prefill mix benchmark parity failed");
    std::array<std::vector<float>,2> times;
    for(int trial=0;trial<5;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;
        const auto& graphs=mode?new_graphs:old_graphs;
        for(int i=0;i<std::max(sets,iterations/4);++i)if(!graphs[i%sets]->enqueue_if_device_only())
            throw std::runtime_error("GR prefill mix warmup acquired a host task");
        mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
        for(int i=0;i<iterations;++i)if(!graphs[i%sets]->enqueue_if_device_only())
            throw std::runtime_error("GR prefill mix timing acquired a host task");
        end.record(stream);end.synchronize();float elapsed=0;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin.get(),end.get()));times[mode].push_back(elapsed/iterations);
    }
    for(auto& t:times)std::sort(t.begin(),t.end());
    std::printf("gr_prefill_%s M=%d hidden=%d K=%d streams=%d sets=%d iterations=%d old_ms=%.6f fused_ms=%.6f median5=1\n",
        after?"read_after":full?"read":"mix",rows,hidden,rank,streams,sets,iterations,times[0][2],times[1][2]);
}
void check_shared_gate_multiply(int width,int rows,tb::ScalarType input_type,tb::ScalarType expert_type,bool graph) {
    auto x=pattern(int64_t(rows)*width+1,.29f).to(input_type).narrow(0,1,int64_t(rows)*width).reshape({rows,width});
    auto weight=pattern(width+1,.57f).narrow(0,1,width).reshape({1,width});
    auto expert=pattern(int64_t(rows)*259+1,.41f).to(expert_type).narrow(0,1,int64_t(rows)*259).reshape({rows,259});
    const auto compute=[&] {
        auto result=try_shared_gate_multiply_cuda(x,weight,expert);
        if(!result || result->scalar_type()!=tb::kFloat32 || result->sizes()!=expert.sizes())
            throw std::runtime_error("FP32 shared gate scaling changed its dtype/shape or rejected a valid case");
        return *result;
    };
    const auto compare=[&](const tb::Tensor& actual) {
        const auto reference=tb::sigmoid(dense_projection(x,weight))*expert;
        if(reference.scalar_type()!=tb::kFloat32)throw std::runtime_error("Shared gate FP32 reference precision changed");
        const auto error=(actual-reference).abs().max().item<float>();
        if(!std::isfinite(error) || error>2e-6f)
            throw std::runtime_error("FP32 shared gate scaling exceeded float-addition allowance");
    };
    compare(compute());
    if(graph) {
        const auto context=mfq::cuda::default_context(x.get_device());context->begin_graph_pool(mfq_current_cuda_stream());
        struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;
            ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}} pool_end{context};
        tb::Tensor actual;mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{actual=compute();},[]{},[&]{actual={};});
        for(float value:{.01f,-.07f,100.f}) {x.fill_(value);expert.fill_(value*.001f);window.run();compare(actual);}
    }
    for(float sign:{-1.f,1.f}) {x.fill_(100.f);weight.fill_(sign*100.f);compare(compute());}
    if(try_shared_gate_multiply_cuda(x,weight.to(tb::kBFloat16),expert) ||
            try_shared_gate_multiply_cuda(x,weight,expert.narrow(0,0,rows-1)) ||
            (rows>1 && try_shared_gate_multiply_cuda(x,weight,expert.narrow(1,0,257))))
        throw std::runtime_error("FP32 shared gate accepted incompatible weight, row count or strided expert output");
}
void benchmark_shared_gate_multiply(int rows,int width,int outputs,int iterations) {
    auto x=pattern(int64_t(rows)*width,.29f).to(tb::kFloat16).reshape({rows,width});
    auto weight=pattern(width,.57f).reshape({1,width});
    auto expert=pattern(int64_t(rows)*outputs,.41f).to(tb::kFloat16).reshape({rows,outputs});
    const auto context=mfq::cuda::default_context(x.get_device());context->begin_graph_pool(mfq_current_cuda_stream());
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;
        ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}} pool_end{context};
    tb::Tensor result[2];mfq::cuda::DecodeWindow before(mfq_current_cuda_stream()),after(mfq_current_cuda_stream());
    before.capture([&]{result[0]=tb::sigmoid(dense_projection(x,weight))*expert;},[]{});
    after.capture([&]{auto output=try_shared_gate_multiply_cuda(x,weight,expert);
        if(!output)throw std::runtime_error("FP32 gate benchmark did not use the fused entry");result[1]=*output;},[]{});
    before.run();after.run();
    if((result[0]-result[1]).abs().max().item<float>()>2e-6f)throw std::runtime_error("FP32 gate benchmark parity failed");
    std::array<std::vector<float>,2> times;
    for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
        const int mode=(trial+order)%2;auto& window=mode?after:before;
        for(int i=0;i<iterations/4;++i)if(!window.enqueue_if_device_only())throw std::runtime_error("Gate benchmark has a host task");
        mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(mfq_current_cuda_stream());
        for(int i=0;i<iterations;++i)window.enqueue_if_device_only();end.record(mfq_current_cuda_stream());end.synchronize();
        float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));times[mode].push_back(ms*1000/iterations);
    }
    for(auto& t:times)std::sort(t.begin(),t.end());
    std::printf("shared_gate_multiply M=%d K=%d output=%d old_us=%.6f new_us=%.6f ratio=%.6f actual_match=1\n",
        rows,width,outputs,times[0][3],times[1][3],times[1][3]/times[0][3]);
}
void check_shared_gate(int width,int rows,bool graph) {
    auto x=pattern(int64_t(rows)*width+1,.29f).to(tb::kFloat16).narrow(0,1,int64_t(rows)*width).reshape({rows,width});
    auto weight=pattern(width+1,.57f).to(tb::kBFloat16).narrow(0,1,width).reshape({1,width});
    const auto compute=[&] {
        const auto result=try_shared_gate_sigmoid_cuda(x,weight);
        if(!result || result->scalar_type()!=tb::kFloat16 || result->sizes().vec()!=std::vector<int64_t>({rows,1}))
            throw std::runtime_error("shared gate fusion rejected arbitrary width/rows or changed output precision");
        return *result;
    };
    const auto compare=[&](const tb::Tensor& actual) {
        const auto reference=tb::sigmoid(dense_projection(x,weight));
        const auto error=(actual.to(tb::kFloat32)-reference.to(tb::kFloat32)).abs().max().item<float>();
        // Small windows use the dense adapter's identical dot/reduction tree.
        // Larger rows use BLAS there; the fused dot has the same precision.
        if(!std::isfinite(error) || error>(rows<=6?0.f:.002f))
            throw std::runtime_error("shared gate fusion differs from dense projection and sigmoid");
    };
    compare(compute());
    if(graph) {
        const auto context=mfq::cuda::default_context(x.get_device());
        context->begin_graph_pool(mfq_current_cuda_stream());
        struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;
            ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}} pool_end{context};
        tb::Tensor actual;
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{actual=compute();},[]{},[&]{actual={};});
        for(float value:{.01f,-.07f,100.f}) {x.fill_(value);window.run();compare(actual);}
    }
    for(float sign:{-1.f,1.f}) {
        x.fill_(100.f);weight.fill_(sign*100.f);compare(compute());
    }
    if(try_shared_gate_sigmoid_cuda(x.to(tb::kFloat32),weight))
        throw std::runtime_error("shared gate fusion accepted an unsupported activation precision");
}
void check(int hidden,int rank,int streams,int rows,tb::ScalarType type,bool unaligned) {
    const int width=hidden*streams,offset=int(unaligned);
    auto input=pattern(int64_t(rows)*width+offset,.1f).to(type).narrow(0,offset,int64_t(rows)*width).reshape({1,rows,width});
    auto down=pattern(int64_t(rank)*width+offset,.3f).to(tb::kBFloat16).narrow(0,offset,int64_t(rank)*width).reshape({rank,width});
    auto up=pattern(int64_t(width)*rank+offset,.7f).to(tb::kBFloat16).narrow(0,offset,int64_t(width)*rank).reshape({width,rank});
    auto low=try_gr_bf16_down_cuda(input,down,streams);
    if(!low.defined())throw std::runtime_error("generic BF16 residual Down rejected a valid shape");
    const auto x=input.to(tb::kFloat32).cpu(),w=down.to(tb::kFloat32).cpu(),actual_low=low.cpu();
    for(int m=0;m<rows;++m)for(int r=0;r<rank;++r) {
        double sum=0;
        for(int k=0;k<width;++k)sum+=double(w.data_ptr<float>()[r*width+k])*x.data_ptr<float>()[m*width+k];
        sum/=streams;const auto expected=sum/(1+std::exp(-sum));
        if(std::abs(actual_low.data_ptr<float>()[m*rank+r]-expected)>2e-5*(1+std::abs(expected)))
            throw std::runtime_error("generic BF16 residual Down differs from CPU projection and SiLU");
    }
    auto output=try_gr_bf16_up_mix_cuda(low,input,up,streams);
    if(!output.defined())throw std::runtime_error("generic BF16 residual Up rejected a valid shape");
    const auto v=up.to(tb::kFloat32).cpu(),actual=output.cpu();
    for(int m=0;m<rows;++m)for(int h=0;h<hidden;++h) {
        double expected=0;
        for(int s=0;s<streams;++s) {
            double sum=0;
            for(int r=0;r<rank;++r)sum+=double(v.data_ptr<float>()[(s*hidden+h)*rank+r])*actual_low.data_ptr<float>()[m*rank+r];
            expected+=x.data_ptr<float>()[m*width+s*hidden+h]/(1+std::exp(-sum));
        }
        expected/=streams;
        if(std::abs(actual.data_ptr<float>()[m*hidden+h]-expected)>2e-5*(1+std::abs(expected)))
            throw std::runtime_error("generic BF16 residual Up differs from CPU projection and stream mix");
    }
}
void check_combined(int hidden,int rank,int streams,int rows,tb::ScalarType type,bool after,bool graph) {
    const int width=hidden*streams;
    auto x=pattern(int64_t(rows)*width,.1f).to(type).reshape({1,rows,width});
    const auto norm=pattern(width,.2f),down=pattern(int64_t(rank)*width,.3f).to(tb::kBFloat16).reshape({rank,width});
    const auto up=pattern(int64_t(width)*rank,.4f).to(tb::kBFloat16).reshape({width,rank});
    const auto inject=pattern(int64_t(streams)*width,.5f).to(tb::kBFloat16).reshape({streams,width});
    tb::Tensor branch,prior;
    if(after) {
        branch=pattern(int64_t(rows)*hidden,.7f).to(tb::kFloat16).reshape({1,rows,hidden});
        prior=pattern(int64_t(rows)*streams,.9f).to(type).reshape({1,rows,streams});
    }
    const auto reference=[&] {
        auto residual=after?mfq_qwen4_exp::gated_residual_post(branch,x,prior,streams):x;
        return mfq_qwen4_exp::gated_residual_pre(residual,norm,down,up,inject,hidden,streams,1e-6);
    };
    const auto compare=[&](const std::vector<tb::Tensor>& actual) {
        if(actual.size()!=3)throw std::runtime_error("combined BF16 GR rejected valid inputs");
        const auto expected=reference();
        for(int i=0;i<3;++i) {
            const auto error=(actual[i].to(tb::kFloat32)-expected[i].to(tb::kFloat32)).abs().max().item<float>();
            const auto scale=expected[i].to(tb::kFloat32).abs().max().item<float>();
            if(!std::isfinite(error) || error>2e-5f*(1.f+scale))
                throw std::runtime_error("combined BF16 GR differs from independent norm/projections/post");
        }
    };
    const auto compute=[&]{return try_gated_residual_bf16_cuda(x,norm,down,up,inject,streams,1e-6,branch,prior);};
    compare(compute());
    if(graph) {
        const auto context=mfq::cuda::default_context(x.get_device());
        context->begin_graph_pool(mfq_current_cuda_stream());
        struct PoolEnd {
            std::shared_ptr<mfq::cuda::Context> context;
            ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}
        } pool_end{context};
        std::vector<tb::Tensor> actual;
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{actual=compute();},[]{},[&]{actual.clear();});
        for(float value:{.01f,-.07f}) {x.fill_(value);if(after)branch.fill_(value*2);window.run();compare(actual);}
    }
}
void check_post_mixed(int hidden,int streams,int rows,int mode,bool graph=false) {
    auto branch=pattern(int64_t(rows)*hidden,.17f).to(mode&4?tb::kFloat32:tb::kFloat16).reshape({1,rows,hidden});
    auto residual=pattern(int64_t(rows)*hidden*streams,.31f).to(mode&2?tb::kFloat32:tb::kFloat16).reshape({1,rows,hidden*streams});
    auto injection=pattern(int64_t(rows)*streams,.53f).to(mode&1?tb::kFloat32:tb::kFloat16).reshape({1,rows,streams});
    const auto reference=[&]{return residual+(branch.unsqueeze(-2)*injection.unsqueeze(-1)).reshape(residual.sizes());};
    const auto compare=[&](const tb::Tensor& actual,const tb::Tensor& expected) {
        if(actual.sizes()!=expected.sizes() || actual.scalar_type()!=expected.scalar_type())
            throw std::runtime_error("mixed GR residual post changed shape/dtype");
        auto a=actual.contiguous().cpu(),b=expected.contiguous().cpu();
        if(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))throw std::runtime_error("mixed GR residual post changed product/add rounding");
    };
    compare(mfq_qwen4_exp::gated_residual_post(branch,residual,injection,streams),reference());
    auto destination=tb::empty_like(reference());
    if(!mfq_qwen4_exp::try_gated_residual_post_out(branch,residual,injection,destination,streams))
        throw std::runtime_error("mixed GR residual post rejected compatible destination");
    compare(destination,reference());
    auto saved=residual.clone();const auto expected=reference();
    const bool inplace=mfq_qwen4_exp::try_gated_residual_post_out(branch,residual,injection,residual,streams);
    if(residual.scalar_type()==expected.scalar_type()) {
        if(!inplace)throw std::runtime_error("mixed GR residual post rejected exact alias");
        compare(residual,expected);
    } else {
        if(inplace)throw std::runtime_error("mixed GR residual post wrote to an incompatible output dtype");
        compare(residual,saved);
    }
    residual.copy_(saved);
    if(graph) {
        const auto context=mfq::cuda::default_context(residual.get_device());context->begin_graph_pool(mfq_current_cuda_stream());
        struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;
            ~PoolEnd(){context->end_graph_pool(mfq_current_cuda_stream());}} pool_end{context};
        tb::Tensor output;
        mfq::cuda::DecodeWindow window(mfq_current_cuda_stream());
        window.capture([&]{output=mfq_qwen4_exp::gated_residual_post(branch,residual,injection,streams);},[]{});
        for(float value:{.007f,-.003f}) {branch.fill_(value);window.run();compare(output,reference());}
    }
}
void check_post_alias_rejection() {
    auto storage=pattern(4*37*3+1,.17f);
    auto out=storage.narrow(0,0,4*37*3).reshape({1,4,37*3});
    auto branch=storage.narrow(0,0,4*37).reshape({1,4,37});
    auto residual=pattern(4*37*3,.31f).reshape({1,4,37*3});
    auto injection=pattern(4*3,.53f).reshape({1,4,3});
    if(mfq_qwen4_exp::try_gated_residual_post_out(branch,residual,injection,out,3))
        throw std::runtime_error("GR post accepted branch/output overlap");
    branch=pattern(4*37,.17f).reshape({1,4,37});
    injection=storage.narrow(0,0,4*3).reshape({1,4,3});
    if(mfq_qwen4_exp::try_gated_residual_post_out(branch,residual,injection,out,3))
        throw std::runtime_error("GR post accepted injection/output overlap");
    injection=pattern(4*3,.53f).reshape({1,4,3});residual=out;
    out=storage.narrow(0,1,4*37*3).reshape({1,4,37*3});
    if(mfq_qwen4_exp::try_gated_residual_post_out(branch,residual,injection,out,3))
        throw std::runtime_error("GR post accepted shifted residual/output overlap");
}
void benchmark_post(int hidden,int streams,int rows,int iterations) {
    if(hidden<1 || streams<1 || rows<1 || iterations<1)throw std::invalid_argument("GR post benchmark dimensions");
    auto initial=pattern(int64_t(rows)*hidden*streams,.31f).reshape({1,rows,hidden*streams});
    auto branch=pattern(int64_t(rows)*hidden,.17f).to(tb::kFloat16).reshape({1,rows,hidden});
    auto injection=pattern(int64_t(rows)*streams,.53f).reshape({1,rows,streams});
    auto old=initial.clone(),current=initial.clone();
    const auto stream=mfq_current_cuda_stream();auto context=mfq::cuda::default_context(initial.get_device());
    context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~PoolEnd(){context->end_graph_pool(stream);}} pool_end{context,stream};
    mfq::cuda::DecodeWindow old_graph(stream),new_graph(stream);
    old_graph.capture([&]{old.copy_(old+(branch.unsqueeze(-2)*injection.unsqueeze(-1)).reshape(old.sizes()));},[]{});
    new_graph.capture([&]{if(!mfq_qwen4_exp::try_gated_residual_post_out(branch,current,injection,current,streams))
        throw std::runtime_error("inplace mixed GR residual post did not activate");},[]{});
    old.copy_(initial);current.copy_(initial);
    std::array<std::vector<float>,2> times;
    for(int trial=0;trial<5;++trial)for(int pass=0;pass<2;++pass) {
        const int mode=(trial+pass)%2;auto& graph=mode?new_graph:old_graph;
        for(int i=0;i<4;++i)if(!graph.enqueue_if_device_only())throw std::runtime_error("GR post warmup has host work");
        mfq::cuda::Event begin(cudaEventDefault),end(cudaEventDefault);begin.record(stream);
        for(int i=0;i<iterations;++i)if(!graph.enqueue_if_device_only())throw std::runtime_error("GR post timing has host work");
        end.record(stream);end.synchronize();float elapsed=0;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin.get(),end.get()));times[mode].push_back(elapsed/iterations);
    }
    auto a=old.cpu(),b=current.cpu();
    if(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))throw std::runtime_error("GR post repeated alias updates changed rounding");
    for(auto& t:times)std::sort(t.begin(),t.end());
    std::printf("gr_post M=%d hidden=%d streams=%d iterations=%d old_ms=%.6f fused_ms=%.6f median5=1 bitwise=1\n",
        rows,hidden,streams,iterations,times[0][2],times[1][2]);
}
void benchmark_projection(int hidden,int rank,int streams,int rows,int sets,int iterations,bool full_read=false) {
    if(hidden<1 || rank<1 || streams<1 || rows<1 || rows>6 || sets<1 || iterations<sets)
        throw std::invalid_argument("GR projection benchmark requires positive dimensions, 1..6 rows and iterations>=sets");
    const auto width=int64_t(hidden)*streams;
    auto input=pattern(int64_t(rows)*width,.1f).reshape({1,rows,width});
    const auto norm=pattern(width,.2f)+1;
    const auto branch=pattern(int64_t(rows)*hidden,.4f).to(tb::kFloat16).reshape({1,rows,hidden});
    const auto prior=pattern(int64_t(rows)*streams,.6f).reshape({1,rows,streams});
    const auto context=mfq::cuda::default_context(input.get_device());
    const auto stream=mfq_current_cuda_stream();
    std::vector<tb::Tensor> down,up,inject,low,gates,mixed;
    for(int i=0;i<sets;++i) {
        down.push_back(pattern(int64_t(rank)*width,.3f+i*.07f).to(tb::kBFloat16).reshape({rank,width}));
        up.push_back(pattern(int64_t(rank)*width,.7f+i*.09f).to(tb::kBFloat16).reshape({width,rank}));
        inject.push_back(pattern(int64_t(streams)*width,.5f+i*.11f).to(tb::kBFloat16).reshape({streams,width}));
    }
    low.resize(sets);gates.resize(sets);mixed.resize(sets);
    context->begin_graph_pool(stream);
    struct PoolEnd {std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
        ~PoolEnd(){context->end_graph_pool(stream);}} pool_end{context,stream};
    using Window=mfq::cuda::DecodeWindow;
    std::vector<std::unique_ptr<Window>> down_graphs,up_graphs,combined_graphs;
    for(int i=0;i<sets;++i) {
        const auto make=[&](auto operation) {
            auto graph=std::make_unique<Window>(stream);
            graph->capture(operation,[]{},[]{});return graph;
        };
        if(!full_read) {
        down_graphs.push_back(make([&] {
            auto outputs=try_gr_bf16_down_injection_cuda(input,down[i],inject[i],streams);
            if(outputs.size()!=2)throw std::runtime_error("GR down benchmark did not select the fused path");
            low[i]=outputs[0];gates[i]=outputs[1];
        }));
        up_graphs.push_back(make([&] {
            mixed[i]=try_gr_bf16_up_mix_cuda(low[i],input,up[i],streams);
            if(!mixed[i].defined())throw std::runtime_error("GR up benchmark did not select the fused path");
        }));
        }
        combined_graphs.push_back(make([&] {
            if(full_read) {
                auto outputs=try_gated_residual_bf16_cuda(input,norm,down[i],up[i],inject[i],streams,1e-6,branch,prior);
                if(outputs.size()!=3)throw std::runtime_error("full GR benchmark did not select its fused path");
                mixed[i]=outputs[0];low[i]=outputs[1];gates[i]=outputs[2];
            } else {
                auto outputs=try_gr_bf16_down_injection_cuda(input,down[i],inject[i],streams);
                mixed[i]=try_gr_bf16_up_mix_cuda(outputs.at(0),input,up[i],streams);
            }
        }));
    }
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    const auto measure=[&](const auto& graphs,const char* stage) {
        // A full steady-state warmup avoids measuring the first few
        // milliseconds of clock ramp as a projection's cost.
        for(int i=0;i<std::max(sets,iterations/4);++i)if(!graphs[i%sets]->enqueue_if_device_only())
            throw std::runtime_error("GR benchmark acquired a host dependency");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<iterations;++i)if(!graphs[i%sets]->enqueue_if_device_only())
            throw std::runtime_error("GR benchmark lost its device-only graph");
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));
        std::cout<<"gr_projection stage="<<stage<<" M="<<rows<<" hidden="<<hidden<<" rank="<<rank
            <<" streams="<<streams<<" sets="<<sets<<" iterations="<<iterations<<" ms="<<elapsed/iterations<<'\n';
    };
    if(!full_read){measure(down_graphs,"down_injection");measure(up_graphs,"up_mix");}
    measure(combined_graphs,full_read?"full_read":"combined");
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
}
}
int main(int argc,char** argv) {
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
    try {
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        std::cout<<std::unitbuf;
        if(argc==8 && std::string(argv[1])=="--projection-bench") {
            benchmark_projection(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),
                std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]));return 0;
        }
        if(argc==6 && std::string(argv[1])=="--post-mixed-bench") {
            benchmark_post(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]));return 0;
        }
        if(argc==2 && std::string(argv[1])=="--post-mixed-check") {
            for(int mode=0;mode<8;++mode) {
                check_post_mixed(37,5,4,mode,mode==3);check_post_mixed(129,3,133,mode);
            }
            check_post_mixed(2560,4,4,3);check_post_alias_rejection();
            std::cout<<"GR post all8 dtype modes/bit-exact product rounding/exact-alias/overlap-rejection/changed graphs PASS\n";return 0;
        }
        if(argc==8 && std::string(argv[1])=="--gr-read-bench") {
            benchmark_projection(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),
                std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),true);return 0;
        }
        if(argc==8 && std::string(argv[1])=="--prefill-mix-bench") {
            benchmark_prefill_mix(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),
                std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]));return 0;
        }
        if(argc==8 && std::string(argv[1])=="--prefill-read-bench") {
            benchmark_prefill_mix(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),
                std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),true);return 0;
        }
        if(argc==8 && std::string(argv[1])=="--prefill-read-after-bench") {
            benchmark_prefill_mix(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),
                std::stoi(argv[5]),std::stoi(argv[6]),std::stoi(argv[7]),true,true);return 0;
        }
        if(argc==2 && std::string(argv[1])=="--prefill-read-check") {
            check_prefill_read(37,35,5,33,false,true);check_prefill_read(65,47,1,129,true,false,true);
            check_prefill_read(127,96,3,137,false,false);check_prefill_read(129,128,17,35,false,false);
            check_prefill_read(29,7,3,67,false,false);check_prefill_read(2560,320,4,133,true,false);
            for(int types=1;types<8;++types)check_prefill_read(37,35,3,33,true,false,false,types);
            std::cout<<"BF16 compact GR prefill reference/FP32 injection/narrow rank/arbitrary streams/views/graphs PASS\n";return 0;
        }
        if(argc==2 && std::string(argv[1])=="--prefill-mix-check") {
            check_prefill_mix(37,35,5,33,true);check_prefill_mix(65,47,1,129,false,true);
            check_prefill_mix(127,96,3,137,false);check_prefill_mix(129,128,7,257,false);
            check_prefill_mix(2560,320,4,133,false);
            std::cout<<"BF16 GR prefill up/mix arbitrary streams/tails/offsets/views/changed graphs PASS\n";return 0;
        }
        if(argc==6 && std::string(argv[1])=="--shared-gate-multiply-bench") {
            benchmark_shared_gate_multiply(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]));return 0;
        }
        if(argc==2 && std::string(argv[1])=="--shared-gate-multiply-check") {
            for(int width:{1,127,257,2560,4097,8193})for(int rows:{1,2,3,4,6})
                for(auto input:{tb::kFloat16,tb::kFloat32})for(auto expert:{tb::kFloat16,tb::kFloat32})
                    check_shared_gate_multiply(width,rows,input,expert,width==257 && rows==4);
            std::cout<<"FP32 shared gate120cases+changinggraphs+overflow/views/precision checks PASS\n";return 0;
        }
        if(argc==2 && std::string(argv[1])=="--shared-gate-check") {
            for(int width:{1,127,257,2560,4097,8193})for(int rows:{1,2,3,4,6,8,32,97})
                check_shared_gate(width,rows,width==257 && rows==4);
            check_shared_gate(8193,4,true);
            std::cout<<"shared gate48cases+changinggraphs+overflow/precision checks PASS\n";
            return 0;
        }
        check_prefill(129,17,33,true);check_prefill(768,129,97,false);check_prefill(10240,320,33,false);
        for(int rows:{1,2,3,4,5,6})for(auto dtype:{tb::kFloat16,tb::kFloat32,tb::kBFloat16})
            check_dense(257,17,rows,dtype);
        check_dense(2560,513,4,tb::kFloat16);check_dense(2560,48,2,tb::kFloat32);
        if(try_dense_bf16_projection_cuda(pattern(8*2560,.1f).reshape({8,2560}),
                pattern(48*2560,.2f).to(tb::kBFloat16).reshape({48,2560})).defined())
            throw std::runtime_error("BF16 dense dot replaced the existing larger-window GEMM");
        for(int rows:{1,2,3,4,5,6})for(auto type:{tb::kFloat16,tb::kFloat32}) {
            check(37,35,5,rows,type,true);
            check(64,48,3,rows,type,false);
            check_combined(37,35,5,rows,type,false,false);
            check_combined(64,48,3,rows,type,true,false);
        }
        check(2560,320,4,1,tb::kFloat32,false);
        check_combined(2560,320,4,1,tb::kFloat32,false,true);
        check_combined(2560,320,4,4,tb::kFloat32,true,true);
        std::cout<<"BF16 residual projections: row reuse, arbitrary streams, tails, unaligned views and combined graph replay passed\n";
    } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
