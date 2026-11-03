#include "gguf_prefill_fusion.h"
#include "gguf_weight.h"
#include "quant_linear.h"
#include "core/ffn.h"
#include "ggml-quants.h"
#include "../kernels/gguf.h"
#include "mfq_cuda_context.h"
#include "mfq_cuda_norm_ops.h"
#include "mfq_cuda_activation_ops.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
namespace tb=mfq_tensor_backend;
void require(bool p,const char* message){if(!p)throw std::runtime_error(message);}
int main(int argc,char** argv)try {
  int device_count=0;if(cudaGetDeviceCount(&device_count)!=cudaSuccess || !device_count)return 77;
  if(!mfq::cuda::gguf::grouped_f16_available(0))return 77;
  const MfqCudaGuard guard(0);const auto stream=mfq_get_current_cuda_stream(0);
  const bool elementwise_only=argc==2 && std::string_view(argv[1])=="--elementwise-check";
  if(argc==2 && std::string_view(argv[1])=="--disabled-path-check") {
    auto input=tb::zeros({512,32},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCUDA));
    GgufPrefillInputScope scope(input);tb::Tensor half,exponents;
    require(!GgufPrefillInputScope::lookup(input,half,exponents),"explicitly disabled prefill still packed the input");
    require(!gguf_prefill_product(GgufWeight{},input).defined(),"explicitly disabled prefill produced a scaled GEMM");
    auto weight=tb::ones({32},input.options());tb::Tensor normalized;
    require(!GgufPrefillInputScope::prepare_rms(input,weight,1e-6,0,normalized) && !normalized.defined(),
        "explicitly disabled prefill still fused RMSNorm");
    GgufWeight projection;projection.packed=input.to(tb::kUInt8);projection.out=32;projection.neuron_len=32;
    require(!projection.can_forward_prefill_residual(512,input.to(tb::kFloat16)),"explicitly disabled prefill still fused the residual");
    std::cout<<"Explicit prefill disable PASS"<<std::endl;return 0;
  }
  int residual_cases=0;
  for(int rows:{17,33,128})for(int tokens:{3,17,512}) {
    const int columns=128,stride=rows+7;
    auto host=tb::empty({tokens,columns},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCPU));
    auto weights=tb::empty({rows,columns},host.options()),residual=tb::empty({tokens,stride},host.options());
    for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=float(std::sin(i*.17)*.12);
    for(int64_t i=0;i<weights.numel();++i)weights.data_ptr<float>()[i]=float(std::cos(i*.13)*.08);
    for(int64_t i=0;i<residual.numel();++i)residual.data_ptr<float>()[i]=float(std::sin(i*.07)*.3);
    auto x=host.to(tb::kFloat16).to(tb::kCUDA),w=weights.to(tb::kFloat16).to(tb::kCUDA),r=residual.to(tb::kFloat16).to(tb::kCUDA);
    auto output=tb::empty_like(r);output.fill_(.4375);auto dot=tb::empty_like(r);
    const auto context=mfq::cuda::default_context(0);auto& blas=context->blas(stream.stream());
    mfq::cuda::gguf::prefill_gemm_f16_accum(x.data_ptr(),w.data_ptr(),nullptr,dot.data_ptr(),true,nullptr,nullptr,tokens,rows,columns,stride,blas.get(),stream.stream());
    auto baseline=acc_cuda(dot.narrow(1,0,rows).contiguous(),r.narrow(1,0,rows).contiguous()).to(tb::kCPU,tb::kFloat32);
    auto plan=mfq::cuda::gguf::prepare_prefill_residual_gemm(x.data_ptr(),w.data_ptr(),r.data_ptr(),output.data_ptr(),tokens,rows,columns,stride,stream.stream());
    tb::Tensor workspace;
    if(plan.workspace_bytes)workspace=tb::empty({int64_t(plan.workspace_bytes)},x.options().dtype(tb::kUInt8));
    mfq::cuda::gguf::prefill_gemm_f16_residual(x.data_ptr(),w.data_ptr(),r.data_ptr(),output.data_ptr(),tokens,rows,columns,stride,
        plan,workspace.defined()?workspace.data_ptr():nullptr,workspace.defined()?workspace.nbytes():0,blas.get(),stream.stream());
    auto actual=output.to(tb::kCPU,tb::kFloat32),original=r.to(tb::kCPU),expected_original=residual.to(tb::kFloat16);
    require(std::memcmp(original.data_ptr(),expected_original.data_ptr(),original.nbytes())==0,"residual GEMM changed its immutable C input");
    auto hx=x.to(tb::kCPU,tb::kFloat32),hw=w.to(tb::kCPU,tb::kFloat32),hr=r.to(tb::kCPU,tb::kFloat32);
    double old_sse=0,new_sse=0,energy=0;
    for(int t=0;t<tokens;++t)for(int row=0;row<stride;++row) {
      if(row>=rows){require(actual.data_ptr<float>()[t*stride+row]==.4375f,"residual GEMM wrote beyond its output tile");continue;}
      double truth=hr.data_ptr<float>()[t*stride+row];
      for(int k=0;k<columns;++k)truth+=double(hx.data_ptr<float>()[t*columns+k])*hw.data_ptr<float>()[row*columns+k];
      const double a=actual.data_ptr<float>()[t*stride+row],b=baseline.data_ptr<float>()[t*rows+row];
      require(std::isfinite(a),"residual GEMM produced nonfinite output");
      new_sse+=(a-truth)*(a-truth);old_sse+=(b-truth)*(b-truth);energy+=truth*truth;
    }
    require(new_sse<=old_sse*1.05+energy*1e-7,"residual GEMM worsened error against independent FP64 dot products");
    output.fill_(.4375);
    mfq::cuda::gguf::prefill_gemm_f16_residual(x.data_ptr(),w.data_ptr(),r.data_ptr(),output.data_ptr(),tokens,rows,columns,stride,
        {},nullptr,0,blas.get(),stream.stream());
    auto fallback=output.to(tb::kCPU,tb::kFloat32);
    for(int t=0;t<tokens;++t)for(int row=0;row<stride;++row)
        require(fallback.data_ptr<float>()[t*stride+row]==(row<rows?baseline.data_ptr<float>()[t*rows+row]:.4375f),
            "residual GEMM scratch fallback changed the ordinary result or output tail");
    bool rejected=false;
    try {mfq::cuda::gguf::prepare_prefill_residual_gemm(x.data_ptr(),w.data_ptr(),r.data_ptr(),r.data_ptr(),tokens,rows,columns,stride,stream.stream());}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"residual GEMM accepted an aliased immutable input");
    if(plan) {
      rejected=false;
      try {mfq::cuda::gguf::prefill_gemm_f16_residual(x.data_ptr(),w.data_ptr(),r.data_ptr(),output.data_ptr(),tokens+1,rows,columns,stride,
          plan,workspace.defined()?workspace.data_ptr():nullptr,workspace.defined()?workspace.nbytes():0,blas.get(),stream.stream());}
      catch(const std::invalid_argument&){rejected=true;}
      require(rejected,"residual GEMM accepted a plan for another geometry");
    }
    ++residual_cases;
  }
  {
    const int input_width=64,hidden=128,tokens=512;
    auto make_weight=[&](int rows,int columns,int seed) {
      auto host=tb::empty({rows,columns},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCPU));
      for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=float(std::sin(i*.031+seed)*.03);
      const auto row_bytes=ggml_row_size(GGML_TYPE_Q8_0,columns);auto packed=tb::empty({int64_t(rows*row_bytes)},host.options().dtype(tb::kUInt8));
      for(int row=0;row<rows;++row)quantize_row_q8_0_ref(host.data_ptr<float>()+row*columns,reinterpret_cast<block_q8_0*>(packed.data_ptr<uint8_t>()+row*row_bytes),columns);
      QuantLinear result;result.kind=QuantLinearKind::Gguf;result.logical_out=rows;result.logical_neuron_len=columns;
      result.gguf.type=GGML_TYPE_Q8_0;result.gguf.out=rows;result.gguf.neuron_len=columns;result.gguf.row_bytes=row_bytes;result.gguf.packed=packed.to(tb::kCUDA);return result;
    };
    FFN ffn;ffn.gate_up.layers={make_weight(hidden,input_width,1),make_weight(hidden,input_width,2)};ffn.gate_up.outs={hidden,hidden};ffn.down=make_weight(input_width,hidden,3);
    auto input=tb::empty({tokens,input_width},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCPU));
    for(int64_t i=0;i<input.numel();++i)input.data_ptr<float>()[i]=float(std::sin(i*.047)*.4);
    auto gpu=input.to(tb::kFloat16).to(tb::kCUDA),residual=(input.to(tb::kCUDA)*.25).to(tb::kFloat16);CudaExecutionContext execution;
    auto expected=acc_cuda(residual,ffn.forward(execution,gpu)).to(tb::kCPU,tb::kFloat32);
    auto result=ffn.try_forward_gguf_prefill_residual(execution,gpu,residual);require(result.has_value(),"FFN did not select the residual prefill path");
    auto actual=result->to(tb::kCPU,tb::kFloat32);double error=0,energy=0;
    for(int64_t i=0;i<actual.numel();++i){const double a=actual.data_ptr<float>()[i],r=expected.data_ptr<float>()[i];error+=(a-r)*(a-r);energy+=r*r;}
    require(error<energy*4e-6+1e-12,"fused FFN differs excessively from the ordinary quantized FFN");
    require(!ffn.try_forward_gguf_prefill_residual(execution,gpu.narrow(0,0,4),residual.narrow(0,0,4)),"MTP small M entered prefill residual projection");
    auto side=mfq::cuda::stream_from_pool(false,0);
    {mfq::cuda::StreamGuard use(side);mfq::cuda::Graph graph;graph.prepare_memory();
     {auto warm=ffn.try_forward_gguf_prefill_residual(execution,gpu,residual);require(warm.has_value(),"residual FFN graph warmup missing");}
     TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(side.stream()));
     graph.capture_begin();auto replayed=ffn.try_forward_gguf_prefill_residual(execution,gpu,residual);graph.capture_end();
     for(float value:{.125f,-.75f,1.5f}) {
       gpu.fill_(value);residual.fill_(value*.2f);graph.replay();
       auto reference=ffn.try_forward_gguf_prefill_residual(execution,gpu,residual);
       auto a=replayed->to(tb::kCPU),r=reference->to(tb::kCPU);
       require(std::memcmp(a.data_ptr(),r.data_ptr(),a.nbytes())==0,"residual FFN graph kept stale inputs or residuals");
     }}
    struct Correction final:GgufProjectionCorrection {Tensor apply(const Tensor&,Tensor y)const override{return y;}};
    ffn.down.gguf.projection_correction=std::make_shared<Correction>();
    require(!ffn.try_forward_gguf_prefill_residual(execution,gpu,residual),"projection correction was applied in the wrong residual order");
  }
  std::cout<<"Prefill residual GEMM independent FP64 reference PASS cases="<<residual_cases<<"; immutable input, output tails, scratch fallback, FFN Graph replay and correction/small-M fallback PASS"<<std::endl;
  if(argc==2 && std::string_view(argv[1])=="--prefill-residual-check")return 0;
  int input_cases=0;
  for(const auto dtype:{tb::kFloat32,tb::kFloat16,tb::kBFloat16})for(int width:{17,257,2048,4097,5120})for(double offset:{0.,1.}) {
    auto host=tb::empty({2,257,width},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCPU));
    auto weight=tb::empty({width},host.options());
    for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=float(std::sin(i*.013)*3.+std::cos(i*.017));
    for(int i=0;i<width;++i)weight.data_ptr<float>()[i]=float(.7+.3*std::cos(i*.037));
    auto input=host.to(dtype).to(tb::kCUDA),norm=weight.to(tb::kCUDA);tb::Tensor normalized,packed,exponents;
    auto scope=GgufPrefillInputScope::prepare_rms(input,norm,1e-6,offset,normalized);
    if(!mfq::cuda::gguf::rms_prefill_available(514,width,0)){require(!scope && !normalized.defined(),"RMS resource fallback materialized an unused input");continue;}
    require(bool(scope) && normalized.sizes()==input.sizes(),"prefill RMS input was not prepared");
    require(GgufPrefillInputScope::lookup(normalized,packed,exponents),"prefill RMS packet was not reused");
    auto reference=rms_norm_offset_cuda(input.reshape({514,width}),norm,1e-6,offset).reshape_as(input);
    auto expected_pack=tb::empty_like(packed),expected_exp=tb::empty_like(exponents);
    mfq::cuda::gguf::pack_scaled_f16_unit(reference.data_ptr<float>(),expected_pack.data_ptr(),expected_exp.data_ptr<int32_t>(),514,width,stream.stream());
    for(auto pair:{std::pair{normalized,reference},std::pair{packed,expected_pack},std::pair{exponents,expected_exp}}) {
        auto a=pair.first.to(tb::kCPU),r=pair.second.to(tb::kCPU);
        require(a.nbytes()==r.nbytes() && std::memcmp(a.data_ptr(),r.data_ptr(),a.nbytes())==0,"fused RMS/scaled-F16 packet changed ordinary arithmetic");
    }
    auto branch=mfq::cuda::stream_from_pool(false,0);
    {mfq::cuda::StreamGuard use(branch);tb::Tensor shared,scales;
     require(GgufPrefillInputScope::lookup(normalized,shared,scales),"fused RMS packet could not cross streams");
     auto a=shared.to(tb::kCPU),r=expected_pack.to(tb::kCPU);
     require(std::memcmp(a.data_ptr(),r.data_ptr(),a.nbytes())==0,"cross-stream RMS packet raced its producer");}
    scope.reset();require(!GgufPrefillInputScope::lookup(normalized,packed,exponents),"released RMS packet remained visible");
    ++input_cases;
  }
  {
    auto input=tb::ones({3,2048},tb::TensorOptions{}.dtype(tb::kFloat16).device(tb::kCUDA));
    auto weight=tb::ones({2048},input.options().dtype(tb::kFloat32));tb::Tensor normalized;
    require(!GgufPrefillInputScope::prepare_rms(input,weight,1e-6,0,normalized) && !normalized.defined(),
        "small-M path prepared an unused prefill packet");
    QuantLinearGroup dense;dense.layers.resize(1);dense.layers[0].kind=QuantLinearKind::Dense;
    CudaExecutionContext execution;auto large=tb::ones({512,2048},input.options());
    require(!dense.prepare_prefill_rms_norm_input(execution,large,weight,1e-6,0,normalized) && !normalized.defined(),
        "dense projection prepared an unused GGUF packet");
    require(!mfq::cuda::gguf::rms_prefill_available(512,INT_MAX,0),"impossible RMS shared memory request was accepted");
  }
  {
    auto graph_stream=mfq::cuda::stream_from_pool(false,0);mfq::cuda::StreamGuard use(graph_stream);
    auto input=tb::ones({512,257},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCUDA));
    auto weight=tb::ones({257},input.options());tb::Tensor normalized,packed,exponents;
    mfq::cuda::Graph graph;graph.prepare_memory();
    {auto warm=GgufPrefillInputScope::prepare_rms(input,weight,1e-6,1,normalized);require(bool(warm),"graph RMS warmup missing");}
    normalized={};
    auto graph_check=[](cudaError_t e,const char* phase){if(e!=cudaSuccess)throw std::runtime_error(std::string(phase)+": "+cudaGetErrorString(e));};
    graph_check(cudaStreamSynchronize(graph_stream.stream()),"RMS graph warmup sync");
    graph.capture_begin();
    try {auto scope=GgufPrefillInputScope::prepare_rms(input,weight,1e-6,1,normalized);
     require(bool(scope) && GgufPrefillInputScope::lookup(normalized,packed,exponents),"graph RMS packet missing");}
    catch(const std::exception& e){throw std::runtime_error(std::string("RMS capture producer: ")+e.what());}
    graph.capture_end();
    for(float value:{.125f,-.375f,1.5f}) {
      input.fill_(value);graph.replay();
      auto reference=rms_norm_offset_cuda(input,weight,1e-6,1);
      auto expected_pack=tb::empty_like(packed),expected_exp=tb::empty_like(exponents);
      mfq::cuda::gguf::pack_scaled_f16_unit(reference.data_ptr<float>(),expected_pack.data_ptr(),expected_exp.data_ptr<int32_t>(),512,257,graph_stream.stream());
      for(auto pair:{std::pair{normalized,reference},std::pair{packed,expected_pack},std::pair{exponents,expected_exp}}) {
        auto a=pair.first.to(tb::kCPU),r=pair.second.to(tb::kCPU);
        require(std::memcmp(a.data_ptr(),r.data_ptr(),a.nbytes())==0,"RMS Graph replay kept stale input data or scales");
      }
    }
  }
  std::cout<<"Prefill RMS/scaled input exact checks PASS cases="<<input_cases<<"; cross-stream reuse and small-M fallback PASS"<<std::endl;
  std::cout<<"Prefill RMS changing-input Graph replay, dense projection fallback and resource checks PASS"<<std::endl;
  if(argc==2 && std::string_view(argv[1])=="--rms-input-check")return 0;
  int cases=0,skipped=0;double worst=0;
  for(int heads:{3,7,48})for(int width:{17,32,64,128,192,320})for(bool silu:{false,true}) {
    const int batch=2,tokens=19,cols=heads*width,stride=cols+11;
    auto x=tb::empty({batch,heads,tokens,width},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCPU));
    auto g=tb::empty({batch,tokens,stride},x.options());auto norm=tb::empty({width},x.options());
    for(int64_t i=0;i<x.numel();++i)x.data_ptr<float>()[i]=float(std::sin(i*.017)*3+std::cos(i*.13));
    for(int64_t i=0;i<g.numel();++i)g.data_ptr<float>()[i]=float(4*std::sin(i*.023));
    for(int i=0;i<width;++i)norm.data_ptr<float>()[i]=float(.7+.5*std::cos(i*.31));
    for(int h=0;h<heads;++h)for(int d=0;d<width;++d)x.data_ptr<float>()[h*tokens*width+d]=0;
    auto gpu=x.to(tb::kCUDA),gate=g.to(tb::kCUDA).narrow(2,0,cols),weight=norm.to(tb::kCUDA);
    auto packed=tb::empty({batch*tokens,cols},gpu.options().dtype(tb::kFloat16));
    auto exp=tb::empty({batch*tokens},gpu.options().dtype(tb::kInt32));
    if(!mfq::cuda::gguf::rms_gated_prefill_available(cols,0)){++skipped;continue;}
    mfq::cuda::gguf::pack_rms_gated_prefill(gpu.data_ptr<float>(),gate.data_ptr<float>(),weight.data_ptr<float>(),
       batch,heads,tokens,width,gate.stride(0),gate.stride(1),1e-5f,silu,packed.data_ptr(),exp.data_ptr<int32_t>(),stream.stream());
    auto normalized=rms_norm_cuda(gpu.reshape({-1,width}),weight,1e-5).reshape_as(gpu)
        .transpose(1,2).contiguous().reshape({batch*tokens,cols});
    auto gated=tb::empty_like(normalized);auto reference=tb::empty_like(packed),reference_exp=tb::empty_like(exp);
    auto contiguous_gate=gate.contiguous();
    mfq::cuda::gguf::materialize_gated_input(normalized.data_ptr(),contiguous_gate.data_ptr(),false,normalized.numel(),silu,gated.data_ptr(),stream.stream());
    mfq::cuda::gguf::pack_scaled_f16_unit(gated.data_ptr<float>(),reference.data_ptr(),reference_exp.data_ptr<int32_t>(),batch*tokens,cols,stream.stream());
    if(width%32==0) {
        auto q8=tb::empty({int64_t(batch*tokens*mfq::cuda::gguf::q8_row_bytes(cols))},gpu.options().dtype(tb::kUInt8)),expected=tb::empty_like(q8);
        mfq::cuda::gguf::quantize_rms_gated_input(gpu.data_ptr<float>(),gate.data_ptr<float>(),weight.data_ptr<float>(),
            batch,heads,tokens,width,gate.stride(0),gate.stride(1),1e-5f,silu,q8.data_ptr(),stream.stream());
        mfq::cuda::gguf::quantize_input(gated.data_ptr<float>(),batch*tokens,cols,expected.data_ptr(),stream.stream());
        auto a=q8.to(tb::kCPU),r=expected.to(tb::kCPU);
        require(std::memcmp(a.data_ptr(),r.data_ptr(),a.nbytes())==0,"Sigmoid/SiLU RMS Q8 packet differs from reference");
    }
    auto packed_cpu=packed.to(tb::kCPU),reference_cpu=reference.to(tb::kCPU),reference_exp_cpu=reference_exp.to(tb::kCPU);
    auto actual=packed.to(tb::kFloat32).to(tb::kCPU),exps=exp.to(tb::kCPU);
    require(std::memcmp(packed_cpu.data_ptr(),reference_cpu.data_ptr(),packed_cpu.nbytes())==0 &&
        std::memcmp(exps.data_ptr(),reference_exp_cpu.data_ptr(),exps.nbytes())==0,"fused prefill differs from ordinary RMS/layout/gate/scale arithmetic");double error=0,energy=0,maxerr=0,peak=0;
    // Restore a scaled gate at consumption and preserve every FP32 operation
    // used by the ordinary gate path, including padded batch/token strides.
    auto gate_half=g.to(tb::kFloat16),decoded_gate=gate_half.to(tb::kFloat32);
    auto gr=tb::empty({batch*tokens},x.options().dtype(tb::kInt32));
    auto gc=tb::empty({cols},gr.options());
    for(int r=0;r<batch*tokens;++r)gr.data_ptr<int32_t>()[r]=r%5-2;
    for(int c=0;c<cols;++c)gc.data_ptr<int32_t>()[c]=c%7-3;
    for(int r=0;r<batch*tokens;++r)for(int c=0;c<cols;++c)
        decoded_gate.data_ptr<float>()[r*stride+c]=std::ldexp(decoded_gate.data_ptr<float>()[r*stride+c],
            gr.data_ptr<int32_t>()[r]+gc.data_ptr<int32_t>()[c]);
    auto encoded_gpu=gate_half.to(tb::kCUDA),decoded_gpu=decoded_gate.to(tb::kCUDA);
    auto gr_gpu=gr.to(tb::kCUDA),gc_gpu=gc.to(tb::kCUDA);
    mfq::cuda::gguf::pack_rms_gated_prefill(gpu.data_ptr<float>(),decoded_gpu.data_ptr<float>(),weight.data_ptr<float>(),
        batch,heads,tokens,width,tokens*stride,stride,1e-5f,silu,reference.data_ptr(),reference_exp.data_ptr<int32_t>(),stream.stream());
    mfq::cuda::gguf::pack_rms_scaled_gate_prefill(gpu.data_ptr<float>(),encoded_gpu.data_ptr(),weight.data_ptr<float>(),
        gr_gpu.data_ptr<int32_t>(),gc_gpu.data_ptr<int32_t>(),batch,heads,tokens,width,tokens*stride,stride,
        1e-5f,silu,packed.data_ptr(),exp.data_ptr<int32_t>(),stream.stream());
    auto encoded_result=packed.to(tb::kCPU),decoded_result=reference.to(tb::kCPU);
    auto encoded_exp=exp.to(tb::kCPU),decoded_exp=reference_exp.to(tb::kCPU);
    require(std::memcmp(encoded_result.data_ptr(),decoded_result.data_ptr(),encoded_result.nbytes())==0 &&
        std::memcmp(encoded_exp.data_ptr(),decoded_exp.data_ptr(),encoded_exp.nbytes())==0,
        "scaled gate consumption changed RMS/gate/scale arithmetic");
    for(int b=0;b<batch;++b)for(int t=0;t<tokens;++t)for(int h=0;h<heads;++h) {
      double square=0;for(int d=0;d<width;++d){const double v=x.data_ptr<float>()[((b*heads+h)*tokens+t)*width+d];square+=v*v;}
      const double inv=1/std::sqrt(square/width+1e-5);
      for(int d=0;d<width;++d) {
        const int col=h*width+d,row=b*tokens+t;const double gv=g.data_ptr<float>()[row*stride+col];
        const double expected=x.data_ptr<float>()[((b*heads+h)*tokens+t)*width+d]*inv*norm.data_ptr<float>()[d]*(silu?gv:1)/(1+std::exp(-gv));
        const double got=std::ldexp(double(actual.data_ptr<float>()[row*cols+col]),exps.data_ptr<int32_t>()[row]);
        error+=(got-expected)*(got-expected);energy+=expected*expected;maxerr=std::max(maxerr,std::abs(got-expected));peak=std::max(peak,std::abs(expected));
      }
    }
    const double relative=std::sqrt(error/std::max(energy,1e-30));worst=std::max(worst,relative);
    require(relative<.0005 && maxerr<std::max(.0006*peak,1e-7),"fused RMS/layout/gate/scale result differs from analytic reference");++cases;
  }
  auto x=tb::empty({2,512,127},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCUDA));x.fill_(.375);
  tb::Tensor half,exp;
  {GgufPrefillInputScope scope(x);require(GgufPrefillInputScope::lookup(x,half,exp),"scaled group input was not shared");
   auto y=tb::empty_like(x);y.fill_(.75);
   {GgufPrefillInputScope nested(y);require(!GgufPrefillInputScope::lookup(x,half,exp),"nested scope incorrectly matched an older tensor");
    require(GgufPrefillInputScope::lookup(y,half,exp),"nested input was not prepared");}
   require(GgufPrefillInputScope::lookup(x,half,exp),"nested scope did not restore its parent");
   const auto other=mfq::cuda::stream_from_pool(false,0);
   {mfq::cuda::StreamGuard use(other);require(GgufPrefillInputScope::lookup(x,half,exp),"cross-stream reuse failed");
    auto actual=half.to(tb::kFloat32).to(tb::kCPU),exps=exp.to(tb::kCPU);
    for(int64_t i=0;i<actual.numel();++i)require(std::ldexp(actual.data_ptr<float>()[i],exps.data_ptr<int32_t>()[i/127])==.375f,"shared input changed values");}
  }
  require(!GgufPrefillInputScope::lookup(x,half,exp),"destroyed scope remained visible");
  if(!elementwise_only)for(int columns:{128,256})for(int rows:{9,33}) {
    auto weights=tb::empty({rows,columns},tb::TensorOptions{}.device(tb::kCPU).dtype(tb::kFloat32));
    for(int64_t i=0;i<weights.numel();++i)weights.data_ptr<float>()[i]=float(std::sin(i*.013)*.4);
    const auto row_bytes=ggml_row_size(GGML_TYPE_Q8_0,columns);
    auto bytes=tb::empty({int64_t(rows*row_bytes)},weights.options().dtype(tb::kUInt8));
    for(int row=0;row<rows;++row)quantize_row_q8_0_ref(weights.data_ptr<float>()+row*columns,
        reinterpret_cast<block_q8_0*>(bytes.data_ptr<uint8_t>()+row*row_bytes),columns);
    GgufWeight w;w.type=GGML_TYPE_Q8_0;w.out=rows;w.neuron_len=columns;w.row_bytes=row_bytes;w.packed=bytes.to(tb::kCUDA);
    auto values=tb::empty({512,columns},weights.options());
    for(int64_t i=0;i<values.numel();++i)values.data_ptr<float>()[i]=float(std::cos(i*.071)*3.);
    auto input=values.to(tb::kCUDA);GgufPrefillInputScope scope(input);
    const auto product=gguf_prefill_product(w,input);require(product.defined(),"normalized FP16 prefill product missing");
    const auto raw=product.values.to(tb::kCPU,tb::kFloat32),re=product.row_exponents.to(tb::kCPU),ce=product.column_exponents.to(tb::kCPU);
    const auto reference=w.forward(input).to(tb::kCPU,tb::kFloat32);
    for(int64_t i=0;i<reference.numel();++i)require(std::ldexp(raw.data_ptr<float>()[i],
        re.data_ptr<int32_t>()[i/rows]+ce.data_ptr<int32_t>()[i%rows])==reference.data_ptr<float>()[i],"scaled GEMM product reconstruction changed values");
    auto attended=input.reshape({1,512,4,columns/4}).permute({0,2,1,3}).contiguous();
    auto norm=tb::empty({columns/4},input.options());norm.fill_(.75);
    GgufPrefillProduct scaled_gate{input.to(tb::kFloat16),
        tb::zeros({512},input.options().dtype(tb::kInt32)),tb::zeros({columns},input.options().dtype(tb::kInt32))};
    auto restored_gate=gguf_prefill_materialize(scaled_gate).reshape({1,512,columns});
    for(bool silu:{false,true}) {
      auto direct=gguf_rms_gated_product(w,attended,scaled_gate,norm,1e-5,silu);
      auto restored=gguf_rms_gated_product(w,attended,restored_gate,norm,1e-5,silu);
      require(direct.defined() && restored.defined(),"scaled gate product did not dispatch");
      for(auto pair:{std::pair{direct.values,restored.values},std::pair{direct.row_exponents,restored.row_exponents},
                     std::pair{direct.column_exponents,restored.column_exponents}}) {
        auto a=pair.first.cpu(),b=pair.second.cpu();
        require(std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,"scaled gate projection changed product values or exponents");
      }
    }
  }
  for(int columns:{17,127,2048,4096,4097,5120})for(float offset:{0.f,1.f}) {
    const int tokens=3;
    auto host=tb::empty({tokens,columns},tb::TensorOptions{}.dtype(tb::kFloat32).device(tb::kCPU));
    auto residual=tb::empty_like(host),weight=tb::empty({columns},host.options());
    auto re=tb::empty({tokens},host.options().dtype(tb::kInt32)),ce=tb::empty({columns},re.options());
    for(int r=0;r<tokens;++r)re.data_ptr<int32_t>()[r]=r-1;
    for(int c=0;c<columns;++c){ce.data_ptr<int32_t>()[c]=(c%7)-3;weight.data_ptr<float>()[c]=float(std::cos(c*.17)*.4);}
    for(int64_t i=0;i<host.numel();++i){host.data_ptr<float>()[i]=float(std::sin(i*.03)*.2);residual.data_ptr<float>()[i]=float(std::cos(i*.07)*.3);}
    auto raw=host.to(tb::kFloat16);auto restored=raw.to(tb::kFloat32);
    for(int64_t i=0;i<host.numel();++i)restored.data_ptr<float>()[i]=std::ldexp(restored.data_ptr<float>()[i],
        re.data_ptr<int32_t>()[i/columns]+ce.data_ptr<int32_t>()[i%columns]);
    const GgufPrefillProduct product{raw.to(tb::kCUDA),re.to(tb::kCUDA),ce.to(tb::kCUDA)};
    auto gpu_residual=residual.to(tb::kFloat16).to(tb::kCUDA),gpu_weight=weight.to(tb::kCUDA);
    const auto expected=acc_rms_norm_f16_cuda(gpu_residual,restored.to(tb::kCUDA),gpu_weight,1e-5,offset);
    const auto actual=gguf_prefill_acc_rms(product,gpu_residual,gpu_weight,1e-5,offset);
    for(int i=0;i<2;++i){auto a=actual[i].to(tb::kCPU),r=expected[i].to(tb::kCPU);
        require(std::memcmp(a.data_ptr(),r.data_ptr(),a.nbytes())==0,"fused product restoration/residual/RMS changed FP16 rounding or reduction");}
  }
  TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
  std::cout<<"Fused prefill RMS/layout/Sigmoid/SiLU/scale PASS cases="<<cases<<" resource_skips="<<skipped<<" worst_relative="<<worst<<"; nested and cross-stream input sharing PASS\n";
  std::cout<<"Scaled GEMM and fused restore/residual/RMS exact checks PASS"<<std::endl;
  std::cout<<"Scaled Sigmoid/SiLU gate restoration, padded strides and projected product exact checks PASS"<<std::endl;
  return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
