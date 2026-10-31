#include "models/dflash2/dflash2.h"
#include "models/dflash2/weight_quant.h"
#include "models/qwen4_exp/ops.h"
#include "mfq_cuda_sampling_ops.h"
#include "mfq_cuda_context.h"
#include "runtime/decode_window.h"
#include "kernels/dflash2.h"
#include "ggml-quants.h"
#include "tyloinfer/gguf_model_source.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <bit>
#include <cmath>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <limits>
#include <nlohmann/json.hpp>

namespace tb = mfq_tensor_backend;
namespace {
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void check_weight_quantization() {
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    int checked=0;const int threads=mfq_get_num_threads();
    for(int columns:{32,96,512,1536})for(int rows:{1,7,33,129})for(auto dtype:{tb::kFloat32,tb::kFloat16,tb::kBFloat16}) {
        auto raw=tb::empty({rows,columns},options);
        for(int i=0;i<raw.numel();++i)raw.data_ptr<float>()[i]=i%37==0?0.f:std::sin(i*.17f)*.2f;
        auto input=raw.to(dtype);const auto reference=input.to(tb::kFloat32).contiguous();
        auto expected=tb::empty({rows*int64_t(mfq::gguf::row_bytes(GGML_TYPE_Q8_0,columns))},options.dtype(tb::kUInt8));
        quantize_row_q8_0_ref(reference.data_ptr<float>(),static_cast<block_q8_0*>(expected.data_ptr()),rows*columns);
        for(int count:{1,threads}) {
            mfq_set_num_threads(count);
            const auto actual=mfq::cuda::dflash2::pack_q8_weight_cpu(input);
            require(actual.nbytes()==expected.nbytes() && !std::memcmp(actual.data_ptr(),expected.data_ptr(),expected.nbytes()),
                "Direct row conversion changed standard Q8_0 weight bytes");
            ++checked;
        }
    }
    mfq_set_num_threads(threads);
    std::cout<<"DFlash2 CPU F32/F16/BF16 weight packing independent reference, row tails and thread overrides cases="<<checked<<" PASS\n";
}
void benchmark_weight_quantization(int rows,int columns) {
    require(rows>0 && columns>0 && columns%32==0,"weight benchmark geometry differs");
    auto input=tb::empty({rows,columns},tb::TensorOptions().device(tb::kCPU).dtype(tb::kBFloat16));
    auto* bits=static_cast<std::uint16_t*>(input.data_ptr());
    for(int64_t i=0;i<input.numel();++i)bits[i]=std::uint16_t(0x3d00+(i%127));
    const auto before=std::chrono::steady_clock::now();
    const auto floats=input.to(tb::kFloat32).contiguous();
    auto expected=tb::empty({rows*int64_t(mfq::gguf::row_bytes(GGML_TYPE_Q8_0,columns))},input.options().dtype(tb::kUInt8));
    quantize_row_q8_0_ref(floats.data_ptr<float>(),static_cast<block_q8_0*>(expected.data_ptr()),int64_t(rows)*columns);
    const auto middle=std::chrono::steady_clock::now();
    const auto actual=mfq::cuda::dflash2::pack_q8_weight_cpu(input);
    const auto after=std::chrono::steady_clock::now();
    require(!std::memcmp(actual.data_ptr(),expected.data_ptr(),actual.nbytes()),"Weight benchmark differs from reference bytes");
    std::cout<<"dflash_weight_pack rows="<<rows<<" K="<<columns<<" threads="<<mfq_get_num_threads()
        <<" original_ms="<<std::chrono::duration<double,std::milli>(middle-before).count()
        <<" row_parallel_ms="<<std::chrono::duration<double,std::milli>(after-middle).count()<<" byte_exact=1\n";
}
void check_greedy_probability() {
    int cases=0;
    const auto cpu=tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU);
    for(auto dtype:{tb::kFloat32,tb::kFloat16,tb::kBFloat16})for(int vocabulary:{1,17,4097,248321}) {
        auto input=tb::empty({3,vocabulary},cpu);
        for(int pass=0;pass<3;++pass) {
            for(int row=0;row<3;++row)for(int token=0;token<vocabulary;++token) {
                float value=pass==0?.2f*std::sin(float(token+row)*.07f):0.f;
                if(pass==2)value=row==0?-INFINITY:row==1?(token%3==0?INFINITY:-2.f):NAN;
                input.data_ptr<float>()[int64_t(row)*vocabulary+token]=value;
            }
            auto device=input.to(dtype).to(tb::kCUDA);
            const auto rounded=device.to(tb::kCPU).to(tb::kFloat32).contiguous();
            const auto result=sample_greedy_probability_cuda(device).cpu().contiguous();
            const auto disabled=tb::zeros({1},device.options().dtype(tb::kInt32));
            const auto ids_only=sample_greedy_probability_cuda(device,{},disabled.data_ptr<int32_t>()).cpu().contiguous();
            for(int row=0;row<3;++row) {
                const auto* values=rounded.data_ptr<float>()+int64_t(row)*vocabulary;
                float maximum=-INFINITY;int best=0;
                for(int token=0;token<vocabulary;++token)if(values[token]>maximum){maximum=values[token];best=token;}
                double mass=0.;
                if(maximum==INFINITY)for(int token=0;token<vocabulary;++token)mass+=values[token]==INFINITY;
                else if(std::isfinite(maximum))for(int token=0;token<vocabulary;++token)
                    if(!std::isnan(values[token]))mass+=std::exp(double(values[token])-maximum);
                const float expected=mass>0.?float(1./mass):0.f;
                const auto packet=uint64_t(result.data_ptr<int64_t>()[row]);
                require(int(uint32_t(packet))==best,"GPU full-softmax greedy probability chose a different token");
                const auto without_probability=uint64_t(ids_only.data_ptr<int64_t>()[row]);
                require(uint32_t(without_probability)==uint32_t(packet) &&
                    std::bit_cast<float>(uint32_t(without_probability>>32))==-1.f,
                    "conditional greedy probability changed the selected token or missing-probability sentinel");
                require(std::abs(std::bit_cast<float>(uint32_t(packet>>32))-expected)<2e-6f,
                    "GPU full-softmax greedy probability differs from CPU reference");
                ++cases;
            }
        }
    }
    std::cout<<"GPU greedy token/full-softmax probability dtype/tails/ties/infinities cases="<<cases<<" PASS\n";
}
void check_combined_sampling_packet() {
    const auto cpu=tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU);
    constexpr int vocabulary=4097,k=64;
    auto host=tb::empty({1,vocabulary},cpu),logits=host.to(tb::kCUDA);
    auto packet=tb::full({k+3},-12345,logits.options().dtype(tb::kInt64));
    auto output=tb::empty({k+3},cpu.dtype(tb::kInt64).pinned_memory(true));
    void* packet_output=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&packet_output,output.data_ptr(),0));
    mfq::cuda::HostBuffer control(2*sizeof(std::int32_t),true);void* mapped=nullptr;
    TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,control.data(),0));
    auto* enabled=static_cast<volatile std::int32_t*>(control.data());*enabled=0;enabled[1]=1;
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    const auto stream=mfq_current_cuda_stream();
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    (void)sample_top_k_compact_if_cuda(logits,k,static_cast<const volatile std::int32_t*>(mapped),packet.narrow(0,1,k).reshape({1,k}));
    (void)sample_greedy_probability_cuda(logits,packet.narrow(0,k+1,1),static_cast<const volatile std::int32_t*>(mapped)+1);
    sample_publish_packets_cuda(packet,static_cast<int64_t*>(packet_output));
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    for(int pass=0;pass<4;++pass) {
        *enabled=pass%2;
        enabled[1]=(pass==0 || pass==3)?1:0;
        for(int i=0;i<vocabulary;++i)host.data_ptr<float>()[i]=std::sin(float(i+pass*37)*.17f);
        logits.copy_(host);
        TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto expected=sample_greedy_probability_cuda(logits).cpu();
        const auto actual=uint64_t(output.data_ptr<int64_t>()[k+1]),reference=uint64_t(expected.data_ptr<int64_t>()[0]);
        require(uint32_t(actual)==uint32_t(reference) && (enabled[1]?actual==reference:
            std::bit_cast<float>(uint32_t(actual>>32))==-1.f),"Captured greedy packet changed probability or token");
        require(output.data_ptr<int64_t>()[0]==-12345 && output.data_ptr<int64_t>()[k+2]==-12345,
            "Captured sampling packet wrote outside output views");
        if(*enabled) {
            const auto compact=sample_top_k_compact_cuda(logits,k).cpu();
            require(std::memcmp(output.data_ptr<int64_t>()+1,compact.data_ptr(),compact.nbytes())==0,
                "Captured compact packet differs from separate top-k");
        } else if(pass==0)require(output.data_ptr<int64_t>()[1]==-12345,"Disabled compact selector overwrote its packet");
    }
    TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));
    TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
    std::cout<<"Combined captured sampling packet/output views/replay controls PASS\n";
}
void check_greedy_packet_export() {
    const auto cpu=tb::TensorOptions().dtype(tb::kInt64).device(tb::kCPU);
    int cases=0;
    for(int count:{1,3,33,65,257}) {
        auto host=tb::empty({count},cpu),packets=host.to(tb::kCUDA);
        auto ids=tb::full({count+2},-99,packets.options()),unmapped=tb::empty({count},packets.options());
        auto published=tb::full({count+2},-99,cpu.dtype(tb::kInt32).pinned_memory(true));
        auto raw=tb::full({count+2},-99,cpu.pinned_memory(true));
        void* mapped=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,published.data_ptr(),0));
        void* raw_mapped=nullptr;TYLOINFER_CUDA_CHECK(cudaHostGetDevicePointer(&raw_mapped,raw.data_ptr(),0));
        const auto stream=mfq_current_cuda_stream();
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        sample_export_greedy_ids_cuda(packets,ids.narrow(0,1,count),static_cast<int32_t*>(mapped)+1);
        sample_export_greedy_ids_cuda(packets,unmapped);
        sample_publish_packets_cuda(packets,static_cast<int64_t*>(raw_mapped)+1);
        TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
        TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
        for(int pass=0;pass<4;++pass) {
            for(int i=0;i<count;++i) {
                const auto token=i%3==0?0:i%3==1?248319:int32_t(17+pass+i);
                host.data_ptr<int64_t>()[i]=int64_t((uint64_t(0xff123456u+pass)<<32)|uint32_t(token));
            }
            packets.copy_(host);std::fill_n(published.data_ptr<int32_t>(),count+2,int32_t(-99));
            TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
            TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            const auto output=ids.cpu(),other=unmapped.cpu();
            require(raw.data_ptr<int64_t>()[0]==-99 && raw.data_ptr<int64_t>()[count+1]==-99,
                "mapped raw packet publication wrote outside its view");
            require(output.data_ptr<int64_t>()[0]==-99 && output.data_ptr<int64_t>()[count+1]==-99 &&
                published.data_ptr<int32_t>()[0]==-99 && published.data_ptr<int32_t>()[count+1]==-99,
                "greedy packet export wrote outside output views");
            for(int i=0;i<count;++i) {
                const auto expected=int64_t(uint32_t(host.data_ptr<int64_t>()[i]));
                require(output.data_ptr<int64_t>()[i+1]==expected && other.data_ptr<int64_t>()[i]==expected &&
                    published.data_ptr<int32_t>()[i+1]==expected,"greedy packet export changed token or publication");
                require(raw.data_ptr<int64_t>()[i+1]==host.data_ptr<int64_t>()[i],
                    "mapped packet publication changed the original64bit payload");
                ++cases;
            }
        }
        TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
    }
    std::cout<<"Greedy device feedback/mapped publication/changed graph replay/canaries cases="<<cases<<" PASS\n";
}
void benchmark_greedy_packet(int vocabulary,int rows) {
    require(vocabulary>0 && rows>0,"greedy packet benchmark requires positive dimensions");
    auto host=tb::empty({rows,vocabulary},tb::TensorOptions().dtype(tb::kFloat32).device(tb::kCPU));
    for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.17f);
    auto logits=host.to(tb::kCUDA),output=tb::empty({rows},logits.options().dtype(tb::kInt64));
    auto flag=tb::zeros({1},logits.options().dtype(tb::kInt32));
    auto control=tb::zeros({1},host.options().dtype(tb::kInt32).pinned_memory(true));
    const auto stream=mfq_current_cuda_stream();
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;cudaEvent_t begin,end;
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    (void)sample_greedy_probability_cuda(logits,output,flag.data_ptr<int32_t>());
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    for(int enabled:{1,0}) {
        control.data_ptr<int32_t>()[0]=enabled;flag.copy_(control);
        for(int i=0;i<8;++i)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<256;++i)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));
        std::cout<<"greedy_packet rows="<<rows<<" vocabulary="<<vocabulary<<" probability="<<enabled
            <<" ms="<<elapsed/256<<'\n';
    }
    TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
    TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
}
void check_greedy_selector_walk() {
    namespace df=mfq::cuda::dflash2;
    const auto options=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    constexpr int vocabulary=1024;
    int cases=0;
    for(int rank:{1,33,256})for(int count:{1,17,64})for(int rows:{1,3,7}) {
        auto table=tb::empty({2,vocabulary,rank},options);
        for(int64_t i=0;i<table.numel();++i)table.data_ptr<float>()[i]=.05f*std::sin(float(i)*.17f);
        const auto books=table.to(tb::kBFloat16).to(tb::kCUDA);
        const auto pred=books.select(0,0),succ=books.select(0,1);
        auto compact_host=tb::empty({rows,count},options.dtype(tb::kInt64));
        auto compact=compact_host.to(tb::kCUDA);
        auto hidden_host=tb::empty({rows,rank},options),hidden=hidden_host.to(tb::kCUDA);
        auto anchor_host=tb::empty({1},options.dtype(tb::kInt64)),anchor=anchor_host.to(tb::kCUDA);
        auto staged=tb::empty({rows,count,rank},books.options()),selected=tb::empty({rows},anchor.options());
        for(bool packed:{false,true})for(bool probability:{false,true}) {
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            const auto stream=mfq_current_cuda_stream();
            TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            df::selector_greedy_walk(compact.data_ptr<int64_t>(),rows,count,anchor.data_ptr<int64_t>(),
                static_cast<const uint16_t*>(pred.data_ptr()),static_cast<const uint16_t*>(packed?staged.data_ptr():succ.data_ptr()),
                hidden.data_ptr<float>(),selected.data_ptr<int64_t>(),rank,vocabulary,stream,packed,probability);
            TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
            for(int pass=0;pass<3;++pass) {
                anchor_host.data_ptr<int64_t>()[0]=pass==2?vocabulary:17;
                for(int64_t i=0;i<hidden_host.numel();++i)hidden_host.data_ptr<float>()[i]=pass==1?0.f:.1f*std::cos(float(i)*.13f);
                std::vector<int64_t> ids(rows*count);
                for(int row=0;row<rows;++row)for(int c=0;c<count;++c) {
                    const int token=(c*13+row*17+7+pass*19)%vocabulary;
                    const float score=pass==1?0.f:.2f*std::sin(float(c+row*count)*.37f);
                    uint32_t bits;std::memcpy(&bits,&score,sizeof(bits));
                    const auto at=row*count+c;
                    compact_host.data_ptr<int64_t>()[at]=int64_t((uint64_t(bits)<<32)|uint32_t(token));ids[at]=token;
                }
                if(pass==2)compact_host.data_ptr<int64_t>()[0]=int64_t(uint64_t(0xffffffffu));
                compact.copy_(compact_host);hidden.copy_(hidden_host);anchor.copy_(anchor_host);
                if(packed)staged.copy_(succ.index_select(0,tb::tensor(ids,anchor.options().device(tb::kCPU)).to(tb::kCUDA)).reshape({rows,count,rank}));
                TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
                const auto actual=selected.cpu();
                auto previous=anchor.clone();
                for(int row=0;row<rows;++row) {
                    auto logits=tb::full({vocabulary},-std::numeric_limits<float>::infinity(),hidden.options());
                    df::selector_logits(compact.data_ptr<int64_t>()+row*count,count,previous.data_ptr<int64_t>(),
                        static_cast<const uint16_t*>(pred.data_ptr()),static_cast<const uint16_t*>(packed?staged.data_ptr():succ.data_ptr())+(packed?int64_t(row)*count*rank:0),
                        hidden.data_ptr<float>()+int64_t(row)*rank,logits.data_ptr<float>(),rank,vocabulary,stream,packed);
                    const auto scores=logits.cpu();int64_t best=0;float value=-std::numeric_limits<float>::infinity();
                    for(int token=0;token<vocabulary;++token)if(scores.data_ptr<float>()[token]>value){best=token;value=scores.data_ptr<float>()[token];}
                    const auto packet=uint64_t(actual.data_ptr<int64_t>()[row]);
                    require(int64_t(uint32_t(packet))==best,"DFlash2 GPU greedy walk differs from sequential dense selector or vocabulary tie rule");
                    if(probability) {
                        double mass=0.;
                        if(std::isfinite(value))for(int token=0;token<vocabulary;++token)
                            mass+=std::exp(double(scores.data_ptr<float>()[token])-value);
                        const float expected=mass>0.?float(1./mass):0.f;
                        const float reported=std::bit_cast<float>(uint32_t(packet>>32));
                        require(std::abs(reported-expected)<2e-6f,"DFlash2 greedy confidence differs from full selector softmax");
                    } else require(packet==uint64_t(best),"disabled DFlash2 confidence changed token packet");
                    auto next=tb::tensor(std::vector<int64_t>{best},anchor_host.options());previous.copy_(next);
                }
                ++cases;
            }
            TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    std::cout<<"DFlash2 GPU greedy walk exact sequential reference/ties/invalid IDs/changing graphs cases="<<cases<<" PASS\n";
}
float bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fff + ((bits >> 16) & 1);
    return std::bit_cast<float>(bits & 0xffff0000u);
}
tb::Tensor pattern(std::vector<int64_t> shape, float phase = 0) {
    auto x = tb::empty(shape, tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
    for (int64_t i = 0; i < x.numel(); ++i) x.data_ptr<float>()[i] = std::sin(float(i) * .13f + phase) * .35f;
    return x;
}
void check_mapped_selector(bool staged=false) {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    int capable=0;
    if(!staged) {
        TYLOINFER_NATIVE_CUDA_CHECK(cudaDeviceGetAttribute(&capable,cudaDevAttrCanMapHostMemory,0));
        if(!capable){std::cout<<"DFlash2 mapped selector skipped: host mapping unavailable\n";return;}
    }
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();
    int checked=0;
    for(int rank:{1,33,256})for(int count:{1,16,64}) {
        constexpr int vocab=1024;
        auto source=pattern({2,vocab,rank}).to(tb::kBFloat16);
        Tensor pinned;void* alias=nullptr;
        if(!staged) {
            pinned=tb::empty(source.sizes(),source.options().pinned_memory(true));pinned.copy_(source);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&alias,pinned.data_ptr(),0));
        }
        const auto* mapped=static_cast<const std::uint16_t*>(alias);
        auto resident=source.to(tb::kCUDA),hidden=pattern({rank},.7f).to(tb::kCUDA);
        auto staged_host=tb::empty({count,rank},source.options()),staged_device=staged_host.to(tb::kCUDA);
        auto compact_host=tb::empty({count},source.options().dtype(tb::kInt64));
        for(int i=0;i<count;++i) {
            const auto score=std::bit_cast<std::uint32_t>(float(i)*.01f);
            compact_host.data_ptr<std::int64_t>()[i]=std::int64_t((std::uint64_t(score)<<32)|std::uint32_t((i*13+7)%vocab));
        }
        auto compact=compact_host.to(tb::kCUDA),previous=tb::zeros({1},compact.options());
        auto actual=tb::full({vocab},-INFINITY,hidden.options()),expected=tb::empty_like(actual);
        const auto launch=[&](const std::uint16_t* pre,const std::uint16_t* post,Tensor& out,bool compact_successor=false) {
            selector_logits(compact.data_ptr<std::int64_t>(),count,previous.data_ptr<std::int64_t>(),pre,post,
                hidden.data_ptr<float>(),out.data_ptr<float>(),rank,vocab,stream,compact_successor);
        };
        const auto* gpu=static_cast<const std::uint16_t*>(resident.data_ptr());
        Graph graph;graph.capture_begin();
        if(staged)launch(gpu,static_cast<const std::uint16_t*>(staged_device.data_ptr()),actual,true);
        else launch(mapped,mapped+vocab*rank,actual);
        graph.capture_end();
        for(int prev:{0,17,vocab-1}) {
            for(int i=0;i<count;++i) {
                const auto score=std::bit_cast<std::uint32_t>(float(i)*.01f);
                int token=(i*13+prev+7)%vocab;
                if(staged && prev==17 && i==0)token=-1;
                if(staged && prev==vocab-1 && i==count-1)token=vocab;
                compact_host.data_ptr<int64_t>()[i]=int64_t((uint64_t(score)<<32)|uint32_t(token));
                auto* row=static_cast<std::uint16_t*>(staged_host.data_ptr())+i*rank;
                if(token>=0 && token<vocab)std::memcpy(row,static_cast<const std::uint16_t*>(source.data_ptr())+(vocab+token)*rank,rank*sizeof(std::uint16_t));
                else std::memset(row,0,rank*sizeof(std::uint16_t));
            }
            compact.copy_(compact_host);if(staged)staged_device.copy_(staged_host);
            previous.fill_(prev);actual.fill_(-INFINITY);expected.fill_(-INFINITY);graph.replay();
            launch(gpu,gpu+vocab*rank,expected);
            const auto a=actual.cpu(),b=expected.cpu();
            require(!std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),"DFlash2 alternate codebook storage changed selector logits");++checked;
        }
    }
    std::cout<<"DFlash2 "<<(staged?"staged":"mapped")<<" selector exact changing graph cases="<<checked<<" PASS\n";
}
void check_pointwise_q8() {
    using namespace mfq::cuda::dflash2;
    const auto context=mfq::cuda::default_context();
    const auto stream=mfq_current_cuda_stream();
    const auto bits=[](const tb::Tensor& a,const tb::Tensor& b) {
        const auto x=a.cpu(),y=b.cpu();return x.nbytes()==y.nbytes() && !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes());
    };
    for(int rows:{1,3,8})for(int width:{32,96,512,2560}) {
        const int padded=(width+511)/512*512,group=16;
        auto x=pattern({rows,width}).to(tb::kCUDA),up=pattern({rows,width},.7f).to(tb::kCUDA);
        auto dynamic=pattern({rows,4*(width/group)},.2f).to(tb::kCUDA),base=pattern({4*width},.9f).to(tb::kCUDA);
        auto residual=pattern({rows,width},.3f).to(tb::kCUDA);
        auto expected=tb::empty_like(x),actual=tb::empty_like(x);
        auto expected_codes=tb::empty({8*padded},x.options().dtype(tb::kUInt8)),codes=tb::empty_like(expected_codes);
        auto expected_scales=tb::empty({8*(padded/32)},x.options()),scales=tb::empty_like(expected_scales);
        const auto compare=[&] {
            q8_quantize(expected.data_ptr<float>(),expected_codes.data_ptr(),expected_scales.data_ptr<float>(),rows,width,stream);
            require(bits(actual,expected) && bits(codes,expected_codes) && bits(scales,expected_scales),
                "DFlash2 pointwise/Q8 fusion changed BF16 values, quantization or padded lanes");
        };
        swiglu_float(x.data_ptr<float>(),up.data_ptr<float>(),expected.data_ptr<float>(),x.numel(),stream);
        swiglu_float_q8(x.data_ptr<float>(),up.data_ptr<float>(),actual.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,width,stream);
        compare();
        for(bool finish:{false,true}) {
            cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
            TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
            convolution_q8(x.data_ptr<float>(),dynamic.data_ptr<float>(),base.data_ptr<float>(),
                finish?residual.data_ptr<float>():nullptr,actual.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),
                rows,width,group,finish,stream);
            TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
            TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
            for(float scale:{0.f,.3f,1.f}) {
                x.copy_(pattern({rows,width},scale).to(tb::kCUDA)*scale);
                TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
                convolution(x.data_ptr<float>(),dynamic.data_ptr<float>(),base.data_ptr<float>(),finish?residual.data_ptr<float>():nullptr,
                    expected.data_ptr<float>(),rows,width,group,finish,stream);
                compare();
            }
            TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
        }
    }
    std::cout<<"DFlash2 fused convolution/SwiGLU Q8 exact values/codes/scales/padding + changing graphs PASS\n";
}
void pointwise_benchmark(bool fused) {
    using namespace mfq::cuda::dflash2;
    constexpr int rows=8,width=2560,intermediate=7680,group=16,repetitions=64;
    const auto context=mfq::cuda::default_context();
    const auto stream=mfq_current_cuda_stream();
    auto x=pattern({rows,width}).to(tb::kCUDA),dynamic=pattern({rows,4*(width/group)}).to(tb::kCUDA);
    auto base=pattern({4*width}).to(tb::kCUDA),conv=tb::empty_like(x);
    auto gate=pattern({rows,intermediate}).to(tb::kCUDA),up=pattern({rows,intermediate},.3f).to(tb::kCUDA),activated=tb::empty_like(gate);
    auto codes=tb::empty({8*intermediate},x.options().dtype(tb::kUInt8)),scales=tb::empty({8*(intermediate/32)},x.options());
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    for(int r=0;r<repetitions;++r) {
        for(int i=0;i<2;++i) {
            if(fused)convolution_q8(x.data_ptr<float>(),dynamic.data_ptr<float>(),base.data_ptr<float>(),nullptr,
                conv.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,width,group,false,stream);
            else {
                convolution(x.data_ptr<float>(),dynamic.data_ptr<float>(),base.data_ptr<float>(),nullptr,conv.data_ptr<float>(),rows,width,group,false,stream);
                q8_quantize(conv.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,width,stream);
            }
        }
        if(fused)swiglu_float_q8(gate.data_ptr<float>(),up.data_ptr<float>(),activated.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,intermediate,stream);
        else {
            swiglu_float(gate.data_ptr<float>(),up.data_ptr<float>(),activated.data_ptr<float>(),gate.numel(),stream);
            q8_quantize(activated.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,intermediate,stream);
        }
    }
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    for(int warm=0;warm<8;++warm)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
    std::vector<float> samples(7);
    for(auto& sample:samples) {
        TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&sample,begin,end));
    }
    std::sort(samples.begin(),samples.end());
    std::cout<<"DFlash2 pointwise fused="<<fused<<" layer_us="<<samples[3]*1000/repetitions<<" median7=1\n";
    cudaEventDestroy(begin);cudaEventDestroy(end);cudaGraphExecDestroy(executable);cudaGraphDestroy(graph);
}
void check_device_enqueue() {
    using namespace mfq::cuda;
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    auto input=tb::full({32},1,tb::TensorOptions().device(tb::kCUDA));tb::Tensor output;
    DecodeWindow pure(stream);pure.capture([&]{output=input+2;},[]{},[&]{output={};});
    for(int value:{1,3,7}) {
        input.fill_(value);require(pure.enqueue_if_device_only(),"pure GPU window did not enqueue asynchronously");
        require(output.cpu().data_ptr<float>()[0]==value+2,"asynchronous graph returned stale activation contents");
    }
    int notifications=0;DecodeWindow host(stream);
    host.capture([&]{output=input+3;host.enroll({&notifications,[]{},[&]{++notifications;},[]{},[]{}});},[]{},[&]{output={};});
    const int before=notifications;
    require(!host.enqueue_if_device_only() && notifications==before,"asynchronous enqueue discarded required host tasks");
    host.run();require(notifications==before+1,"host window failed after declining asynchronous enqueue");
    std::cout<<"Pure-device graph asynchronous enqueue + host-task fallback PASS\n";
}
void check_rotary_table() {
    using namespace mfq::cuda::dflash2;
    const auto context=mfq::cuda::default_context();
    const auto stream=mfq_current_cuda_stream();int cases=0;
    for(int dimension:{6,64,128,512})for(int rows:{1,4,8})for(int heads:{1,8,32}) {
        auto x=pattern({rows,heads,dimension},.3f).to(tb::kCUDA);
        auto norm=pattern({dimension},.7f).to(tb::kCUDA)+1;
        auto positions=tb::arange(rows,tb::TensorOptions().dtype(tb::kInt64)).to(tb::kCUDA)+131071;
        auto table=tb::empty({rows,dimension},x.options()),actual=tb::empty_like(x),expected=tb::empty_like(x);
        for(float theta:{10000.f,1000000.f}) {
            norm_rope(x.data_ptr<float>(),norm.data_ptr<float>(),positions.data_ptr<int64_t>(),expected.data_ptr<float>(),
                rows,heads,dimension,theta,1e-6f,stream);
            rotary_table(positions.data_ptr<int64_t>(),table.data_ptr<float>(),rows,dimension,theta,stream);
            norm_rope_tabled(x.data_ptr<float>(),norm.data_ptr<float>(),table.data_ptr<float>(),actual.data_ptr<float>(),
                rows,heads,dimension,1e-6f,stream);
            const auto a=actual.cpu(),e=expected.cpu();
            require(!std::memcmp(a.data_ptr(),e.data_ptr(),a.nbytes()),"DFlash2 shared rotary table changed BF16 output bits");++cases;
        }
    }
    std::cout<<"DFlash2 shared rotary table cases="<<cases<<" exact BF16 bits PASS\n";
}
void rotary_benchmark() {
    using namespace mfq::cuda;
    using namespace mfq::cuda::dflash2;
    const auto context=default_context();
    constexpr int rows=8,dimension=128,layers=5,replays=64;
    auto x=pattern({rows,32,dimension},.3f).to(tb::kCUDA),key=pattern({rows,8,dimension},.5f).to(tb::kCUDA);
    auto norm=pattern({dimension},.7f).to(tb::kCUDA)+1;
    auto positions=tb::arange(rows,tb::TensorOptions().dtype(tb::kInt64)).to(tb::kCUDA)+4096;
    auto table=tb::empty({rows,dimension},x.options()),output=tb::empty_like(x),kout=tb::empty_like(key);
    const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<Context> context;cudaStream_t stream;~Release(){context->end_graph_pool(stream);}} release{context,stream};
    for(bool tabled:{false,true}) {
        const auto launch=[&] {
            if(tabled)rotary_table(positions.data_ptr<int64_t>(),table.data_ptr<float>(),rows,dimension,10000.f,stream);
            for(int i=0;i<layers;++i) {
                if(tabled) {
                    norm_rope_tabled(x.data_ptr<float>(),norm.data_ptr<float>(),table.data_ptr<float>(),output.data_ptr<float>(),rows,32,dimension,1e-6f,stream);
                    norm_rope_tabled(key.data_ptr<float>(),norm.data_ptr<float>(),table.data_ptr<float>(),kout.data_ptr<float>(),rows,8,dimension,1e-6f,stream);
                }else {
                    norm_rope(x.data_ptr<float>(),norm.data_ptr<float>(),positions.data_ptr<int64_t>(),output.data_ptr<float>(),rows,32,dimension,10000.f,1e-6f,stream);
                    norm_rope(key.data_ptr<float>(),norm.data_ptr<float>(),positions.data_ptr<int64_t>(),kout.data_ptr<float>(),rows,8,dimension,10000.f,1e-6f,stream);
                }
            }
        };
        launch();TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
        Graph graph;graph.capture_begin_shared();launch();graph.capture_end();for(int i=0;i<16;++i)graph.replay();
        Event begin(0),end(0);begin.record(stream);for(int i=0;i<replays;++i)graph.replay();end.record(stream);end.synchronize();
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
        std::cout<<"DFlash2 RoPE tabled="<<tabled<<" five_layers_us="<<ms*1000/replays<<" rows="<<rows<<"\n";
    }
}
void check_bf16_context() {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();int checked=0;
    for(bool compact:{false,true})for(int rows:{1,3,8})for(int heads:{1,3})for(int dimension:{6,64,128,512}) {
        constexpr int window=17;
        auto key=pattern({rows,heads,dimension},.1f).to(tb::kCUDA),value=pattern({rows,heads,dimension},.7f).to(tb::kCUDA);
        auto norm=pattern({dimension},1.1f).to(tb::kCUDA),rotated=tb::empty_like(key);
        auto positions=tb::arange(rows,key.options().dtype(tb::kInt64));
        auto table=tb::empty({rows,dimension},key.options());
        auto actual_k=tb::empty({window,heads,dimension},key.options().dtype(compact?tb::kBFloat16:tb::kFloat32)),actual_v=tb::empty_like(actual_k);
        auto expected_k=tb::empty_like(actual_k),expected_v=tb::empty_like(actual_k);
        Graph graph;graph.capture_begin();store_context(key.data_ptr<float>(),value.data_ptr<float>(),norm.data_ptr<float>(),
            table.data_ptr<float>(),positions.data_ptr<int64_t>(),actual_k.data_ptr(),actual_v.data_ptr(),
            rows,heads,dimension,window,1e-6f,stream,compact);graph.capture_end();
        for(int64_t start:{int64_t(0),int64_t(137),int64_t(131071)}) {
            positions.copy_(tb::arange(start,start+rows,positions.options()));
            rotary_table(positions.data_ptr<int64_t>(),table.data_ptr<float>(),rows,dimension,10000.f,stream);
            norm_rope_tabled(key.data_ptr<float>(),norm.data_ptr<float>(),table.data_ptr<float>(),rotated.data_ptr<float>(),
                rows,heads,dimension,1e-6f,stream);
            const auto slots=positions.remainder(double(window)).to(tb::kInt64);
            expected_k.fill_(.25);expected_v.fill_(-.5);actual_k.copy_(expected_k);actual_v.copy_(expected_v);
            expected_k.index_copy_(0,slots,rotated.to(actual_k.scalar_type()));expected_v.index_copy_(0,slots,value.to(actual_v.scalar_type()));
            graph.replay();
            for(const auto& pair:{std::pair{actual_k,expected_k},std::pair{actual_v,expected_v}}) {
                const auto a=pair.first.cpu(),b=pair.second.cpu();
                require(!std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),"Fused BF16 context changed RoPE, values or untouched ring cells");
            }
            ++checked;
        }
    }
    std::cout<<"DFlash2 fused F32/BF16 context exact changing graph cases="<<checked<<" PASS\n";
}
void check_tiled_attention() {
    using namespace mfq::cuda::dflash2;
    const auto context=mfq::cuda::default_context();int cases=0;float max_error=0;
    for(bool compact:{false,true})for(int dimension:{6,64,128,256})for(int gqa:{1,3,4,9})for(int rows:{1,4,8}) {
        constexpr int kv_heads=2,window=129;const int heads=kv_heads*gqa;
        auto q=pattern({rows,heads,dimension},.1f),k=pattern({window,kv_heads,dimension},.3f),v=pattern(k.sizes().vec(),.7f);
        auto bk=pattern({rows,kv_heads,dimension},.5f),bv=pattern(bk.sizes().vec(),.9f);
        if(compact){k=k.to(tb::kBFloat16).to(tb::kFloat32);v=v.to(tb::kBFloat16).to(tb::kFloat32);}
        auto gq=q.to(tb::kCUDA),gk=k.to(compact?tb::kBFloat16:tb::kFloat32).to(tb::kCUDA),
            gv=v.to(compact?tb::kBFloat16:tb::kFloat32).to(tb::kCUDA),gbk=bk.to(tb::kCUDA),gbv=bv.to(tb::kCUDA);
        auto output=tb::empty_like(gq),partial=tb::empty({(window-1+rows+127)/128,rows,heads,dimension+2},gq.options());
        for(int64_t position:{int64_t(1),int64_t(137),int64_t(300)}) {
            attention(gq.data_ptr<float>(),gk.data_ptr(),gv.data_ptr(),gbk.data_ptr<float>(),gbv.data_ptr<float>(),
                partial.data_ptr<float>(),output.data_ptr<float>(),rows,heads,kv_heads,dimension,window,position,mfq_current_cuda_stream(),nullptr,compact);
            const auto actual=output.cpu();
            if(compact) {
                auto float_k=gk.to(tb::kFloat32),float_v=gv.to(tb::kFloat32),reference=tb::empty_like(output);
                attention(gq.data_ptr<float>(),float_k.data_ptr(),float_v.data_ptr(),gbk.data_ptr<float>(),gbv.data_ptr<float>(),
                    partial.data_ptr<float>(),reference.data_ptr<float>(),rows,heads,kv_heads,dimension,window,position,mfq_current_cuda_stream());
                const auto host=reference.cpu();
                if(std::memcmp(actual.data_ptr(),host.data_ptr(),actual.nbytes())) {
                    float max_delta=0;int first=-1;
                    for(int64_t i=0;i<actual.numel();++i) {
                        max_delta=std::max(max_delta,std::abs(actual.data_ptr<float>()[i]-host.data_ptr<float>()[i]));
                        if(first<0 && actual.data_ptr<float>()[i]!=host.data_ptr<float>()[i])first=int(i);
                    }
                    std::cerr<<"BF16 KV mismatch dimension="<<dimension<<" gqa="<<gqa<<" rows="<<rows<<" pos="<<position
                        <<" index="<<first<<" max_delta="<<max_delta<<" actual="<<actual.data_ptr<float>()[first]
                        <<" expected="<<host.data_ptr<float>()[first]<<'\n';
                    throw std::runtime_error("BF16 attention differs from same-valued F32 history");
                }
            }
            for(int row=0;row<rows;++row)for(int head=0;head<heads;++head) {
                std::vector<double> sum(dimension,0);double mass=0;const int kv=head/gqa;
                const auto add=[&](const float* key,const float* value) {
                    double score=0;for(int d=0;d<dimension;++d)score+=double(q.data_ptr<float>()[(row*heads+head)*dimension+d])*key[d];
                    const double p=std::exp(score/std::sqrt(double(dimension)));mass+=p;
                    for(int d=0;d<dimension;++d)sum[d]+=p*value[d];
                };
                for(int64_t cell=std::max(int64_t(0),position+row-window+1);cell<position;++cell) {
                    const auto base=((cell%window)*kv_heads+kv)*dimension;add(k.data_ptr<float>()+base,v.data_ptr<float>()+base);
                }
                for(int cell=0;cell<rows;++cell) {
                    const auto base=(cell*kv_heads+kv)*dimension;add(bk.data_ptr<float>()+base,bv.data_ptr<float>()+base);
                }
                for(int d=0;d<dimension;++d) {
                    const float expected=bf16(float(sum[d]/mass)),value=actual.data_ptr<float>()[(row*heads+head)*dimension+d];
                    const float error=std::abs(value-expected);max_error=std::max(max_error,error);
                    require(std::isfinite(value) && error<1e-5f+.008f*std::abs(expected),
                        "DFlash2 tiled attention differs from independent FP64 sliding-window reference");
                }
            }
            ++cases;
        }
    }
    std::cout<<"DFlash2 tiled attention CPU reference cases="<<cases<<" max_error="<<max_error<<" GQA/tails/ring-wrap PASS\n";
}
void check_rms_q8() {
    using namespace mfq::cuda::dflash2;
    for(int width:{32,64,544,2560})for(int rows:{1,2,3,4,8}) {
        const int padded=(width+511)/512*512;
        auto x=pattern({rows,width},.4f),weight=pattern({width},.7f);
        auto gx=x.to(tb::kCUDA),gw=weight.to(tb::kCUDA),normalized=tb::empty_like(gx);
        auto codes=tb::empty({8*padded},gx.options().dtype(tb::kUInt8)),scales=tb::empty({8*(padded/32)},gx.options());
        rms_norm_q8(gx.data_ptr<float>(),gw.data_ptr<float>(),normalized.data_ptr<float>(),codes.data_ptr(),
            scales.data_ptr<float>(),rows,width,1e-6f,mfq_current_cuda_stream());
        const auto values=normalized.cpu(),q=codes.cpu(),d=scales.cpu();
        for(int row=0;row<rows;++row) {
            double squares=0;for(int c=0;c<width;++c)squares+=double(x.data_ptr<float>()[row*width+c])*x.data_ptr<float>()[row*width+c];
            const double inverse=1/std::sqrt(squares/width+1e-6);
            for(int c=0;c<width;++c) {
                const float expected=bf16(float(x.data_ptr<float>()[row*width+c]*inverse*weight.data_ptr<float>()[c]));
                require(std::abs(values.data_ptr<float>()[row*width+c]-expected)<1e-5f+.008f*std::abs(expected),
                    "DFlash2 fused block RMS differs from FP64 CPU normalization");
            }
        }
        for(int row=0;row<8;++row)for(int block=0;block<padded/32;++block) {
            float maximum=0;
            for(int lane=0;lane<32;++lane)if(row<rows && block*32+lane<width)
                maximum=std::max(maximum,std::abs(values.data_ptr<float>()[row*width+block*32+lane]));
            require(std::abs(d.data_ptr<float>()[row*(padded/32)+block]-maximum/127.f)<1e-9f,
                "DFlash2 fused RMS/Q8 changed quant scales or padding");
            for(int lane=0;lane<32;++lane) {
                const float v=row<rows && block*32+lane<width?values.data_ptr<float>()[row*width+block*32+lane]:0.f;
                const int expected=int(std::nearbyint(v*(maximum>0.f?127.f/maximum:0.f)));
                require(static_cast<signed char>(q.data_ptr<uint8_t>()[row*padded+block*32+lane])==expected,
                    "DFlash2 fused RMS/Q8 changed activation codes or padding");
            }
        }
    }
    std::cout<<"DFlash2 fused RMS FP64 reference/Q8 codes/scales/tails M1..8 PASS\n";
}
void normalization_benchmark(bool fused) {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    for(int rows:{1,4,8}) {
        constexpr int width=2560;auto input=pattern({rows,width},.1f).to(tb::kCUDA);
        auto weight=pattern({width},.3f).to(tb::kCUDA),output=tb::empty_like(input);
        auto codes=tb::empty({8*width},input.options().dtype(tb::kUInt8)),scales=tb::empty({8*(width/32)},input.options());
        const auto launch=[&]{
            if(fused)rms_norm_q8(input.data_ptr<float>(),weight.data_ptr<float>(),output.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,width,1e-6f,stream);
            else {rms_norm(input.data_ptr<float>(),weight.data_ptr<float>(),output.data_ptr<float>(),rows,width,1e-6f,stream);
                q8_quantize(output.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),rows,width,stream);}
        };
        launch();TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
        Graph graph;graph.capture_begin_shared();launch();graph.capture_end();for(int i=0;i<16;++i)graph.replay();
        Event begin(0),end(0);begin.record(stream);for(int i=0;i<128;++i)graph.replay();end.record(stream);end.synchronize();
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
        std::cout<<"DFlash2 RMS+Q8 fused="<<fused<<" M="<<rows<<" us="<<ms*1000/128<<" width="<<width<<"\n";
    }
}
void context_attention_benchmark(int mode,int layers,int rows,int committed,int heads,int kv_heads,int dimension,int window) {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    require((mode==0 || mode==16 || mode==32) && layers>0 && rows>0 && rows<=8 && committed>0 && committed<=window &&
        heads>0 && kv_heads>0 && heads%kv_heads==0 && dimension>0 && dimension<=512 && dimension%2==0,
        "Context/attention benchmark geometry invalid");
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto positions=tb::arange(int64_t(4096-committed),int64_t(4096),options.dtype(tb::kInt64));
    auto slots=positions.remainder(double(window)).to(tb::kInt64),table=tb::empty({committed,dimension},options);
    rotary_table(positions.data_ptr<int64_t>(),table.data_ptr<float>(),committed,dimension,10000.f,stream);
    struct Layer {tb::Tensor key,value,query,bk,bv,norm,input_k,input_v,rotated,partial,output,reference;};
    std::vector<Layer> blocks;
    for(int l=0;l<layers;++l) {
        Layer b;const auto dtype=mode==16?tb::kBFloat16:tb::kFloat32;
        b.key=pattern({window,kv_heads,dimension},l*.1f).to(tb::kBFloat16).to(dtype).to(tb::kCUDA);
        b.value=pattern({window,kv_heads,dimension},.7f+l*.1f).to(tb::kBFloat16).to(dtype).to(tb::kCUDA);
        b.query=pattern({rows,heads,dimension},.3f+l*.1f).to(tb::kCUDA);
        b.bk=pattern({rows,kv_heads,dimension},.5f+l*.1f).to(tb::kCUDA);
        b.bv=pattern({rows,kv_heads,dimension},.9f+l*.1f).to(tb::kCUDA);
        b.norm=pattern({dimension},1.1f+l*.1f).to(tb::kCUDA);
        b.input_k=pattern({committed,kv_heads,dimension},.2f+l*.1f).to(tb::kBFloat16).to(tb::kFloat32).to(tb::kCUDA);
        b.input_v=pattern({committed,kv_heads,dimension},.8f+l*.1f).to(tb::kBFloat16).to(tb::kFloat32).to(tb::kCUDA);
        b.rotated=tb::empty_like(b.input_k);b.partial=tb::empty({(window-1+rows+127)/128,rows,heads,dimension+2},options);
        b.output=tb::empty_like(b.query);b.reference=tb::empty_like(b.output);
        // Independent existing F32 context path, outside timed execution.
        auto k=b.key.to(tb::kFloat32),v=b.value.to(tb::kFloat32);
        norm_rope_tabled(b.input_k.data_ptr<float>(),b.norm.data_ptr<float>(),table.data_ptr<float>(),b.rotated.data_ptr<float>(),
            committed,kv_heads,dimension,1e-6f,stream);
        k.index_copy_(0,slots,b.rotated);v.index_copy_(0,slots,b.input_v);
        attention(b.query.data_ptr<float>(),k.data_ptr(),v.data_ptr(),b.bk.data_ptr<float>(),b.bv.data_ptr<float>(),b.partial.data_ptr<float>(),
            b.reference.data_ptr<float>(),rows,heads,kv_heads,dimension,window,4096,stream);
        blocks.push_back(std::move(b));
    }
    const auto launch=[&] {
        for(auto& b:blocks) {
            if(mode==0) {
                norm_rope_tabled(b.input_k.data_ptr<float>(),b.norm.data_ptr<float>(),table.data_ptr<float>(),b.rotated.data_ptr<float>(),
                    committed,kv_heads,dimension,1e-6f,stream);
                b.key.index_copy_(0,slots,b.rotated);b.value.index_copy_(0,slots,b.input_v);
            }else store_context(b.input_k.data_ptr<float>(),b.input_v.data_ptr<float>(),b.norm.data_ptr<float>(),table.data_ptr<float>(),
                positions.data_ptr<int64_t>(),b.key.data_ptr(),b.value.data_ptr(),committed,kv_heads,dimension,window,1e-6f,stream,mode==16);
            attention(b.query.data_ptr<float>(),b.key.data_ptr(),b.value.data_ptr(),b.bk.data_ptr<float>(),b.bv.data_ptr<float>(),b.partial.data_ptr<float>(),
                b.output.data_ptr<float>(),rows,heads,kv_heads,dimension,window,4096,stream,nullptr,mode==16);
        }
    };
    launch();
    for(const auto& b:blocks) {
        const auto actual=b.output.cpu(),reference=b.reference.cpu();
        require(!std::memcmp(actual.data_ptr(),reference.data_ptr(),actual.nbytes()),"Context/attention benchmark changed output bytes");
    }
    Graph graph;graph.capture_begin_shared();launch();graph.capture_end();for(int i=0;i<16;++i)graph.replay();
    std::vector<float> samples;
    for(int sample=0;sample<5;++sample) {
        Event begin(0),end(0);begin.record(stream);for(int r=0;r<128;++r)graph.replay();end.record(stream);end.synchronize();
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));samples.push_back(ms*1000/128);
    }
    std::sort(samples.begin(),samples.end());
    std::cout<<"dflash_context_attention mode="<<mode<<" layers="<<layers<<" M="<<rows<<" commit="<<committed
        <<" heads="<<heads<<" kv_heads="<<kv_heads<<" width="<<dimension<<" window="<<window
        <<" us="<<samples[2]<<" exact=1\n";
}
void attention_benchmark(bool compact=false,int only_rows=0) {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    for(int rows:{1,4,8}) {
        if(only_rows && rows!=only_rows)continue;
        constexpr int heads=32,kv_heads=8,dimension=128,window=2048;
        auto query=pattern({rows,heads,dimension},.1f).to(tb::kCUDA);
        auto key=pattern({window,kv_heads,dimension},.3f).to(tb::kCUDA),value=pattern(key.sizes().vec(),.7f).to(tb::kCUDA);
        if(compact){key=key.to(tb::kBFloat16);value=value.to(tb::kBFloat16);}
        auto bk=pattern({rows,kv_heads,dimension},.5f).to(tb::kCUDA),bv=pattern(bk.sizes().vec(),.9f).to(tb::kCUDA);
        auto output=tb::empty_like(query),partial=tb::empty({(window-1+rows+127)/128,rows,heads,dimension+2},query.options());
        const auto launch=[&]{attention(query.data_ptr<float>(),key.data_ptr(),value.data_ptr(),
            bk.data_ptr<float>(),bv.data_ptr<float>(),partial.data_ptr<float>(),output.data_ptr<float>(),rows,
            heads,kv_heads,dimension,window,4096,stream,nullptr,compact);};
        launch();TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
        Graph graph;graph.capture_begin_shared();launch();graph.capture_end();for(int i=0;i<16;++i)graph.replay();
        Event begin(0),end(0);begin.record(stream);for(int i=0;i<64;++i)graph.replay();end.record(stream);end.synchronize();
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
        std::cout<<"DFlash2 attention rows="<<rows<<" us="<<ms*1000/64<<" window="<<window<<" kv_bits="<<(compact?16:32)<<"\n";
    }
}
void projection_benchmark(int only_tokens=0) {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    const auto context=default_context();const auto stream=mfq_current_cuda_stream();context->begin_graph_pool(stream);
    struct Release {std::shared_ptr<Context> c;cudaStream_t s;~Release(){c->end_graph_pool(s);}} release{context,stream};
    for(auto [rows,columns]:{std::pair{1024,2560},std::pair{2560,12800},std::pair{256,2560}}) {
        const int pk=(columns+511)/512*512,pn=(rows+31)/32*32;
        auto fp=pattern({rows,columns},.3f),packed=tb::empty({int64_t(rows)*(columns/32)*34},fp.options().dtype(tb::kUInt8));
        quantize_row_q8_0_ref(fp.data_ptr<float>(),static_cast<block_q8_0*>(packed.data_ptr()),int64_t(rows)*columns);
        auto weight=packed.to(tb::kCUDA),tiles=tb::empty({int64_t(pn)*(pk/32)*34},weight.options());
        q8_tile(weight.data_ptr(),tiles.data_ptr(),rows,columns,stream);
        for(int tokens:{1,2,3,4,8}) {
            if(only_tokens && tokens!=only_tokens)continue;
            auto input=pattern({tokens,columns},.7f).to(tb::kCUDA),output=tb::empty({tokens,rows},input.options());
            auto codes=tb::empty({8*pk},weight.options()),scales=tb::empty({8*(pk/32)},input.options());
            Q8TiledMatrix matrix{tiles.data_ptr(),output.data_ptr<float>(),rows,pn};
            const auto launch=[&]{q8_quantize(input.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),tokens,columns,stream);
                q8_gemm(&matrix,1,codes.data_ptr(),scales.data_ptr<float>(),tokens,columns,stream);};
            launch();TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
            Graph graph;graph.capture_begin_shared();launch();graph.capture_end();for(int i=0;i<16;++i)graph.replay();
            Event begin(0),end(0);begin.record(stream);for(int i=0;i<64;++i)graph.replay();end.record(stream);end.synchronize();
            float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
            std::cout<<"DFlash2 Q8 projection M="<<tokens<<" N="<<rows<<" K="<<columns<<" us="<<ms*1000/64<<"\n";
        }
    }
}
void prompt_projection_benchmark(int tokens,int rows,int columns) {
    using namespace mfq::cuda;using namespace mfq::cuda::dflash2;
    if(tokens<1 || tokens>INT_MAX-7 || rows<1 || rows>INT_MAX-31 ||
            columns<1 || columns>INT_MAX-511 || columns%32)
        throw std::invalid_argument("DFlash2 prompt projection geometry differs");
    const auto stream=mfq_current_cuda_stream();
    const int pk=(columns+511)/512*512,pn=(rows+31)/32*32,pm=(tokens+7)/8*8;
    const auto fp=pattern({rows,columns},.3f);
    auto packed=tb::empty({int64_t(rows)*(columns/32)*34},fp.options().dtype(tb::kUInt8));
    quantize_row_q8_0_ref(fp.data_ptr<float>(),static_cast<block_q8_0*>(packed.data_ptr()),int64_t(rows)*columns);
    const auto weights=packed.to(tb::kCUDA);
    auto tiles=tb::empty({int64_t(pn)*(pk/32)*34},weights.options());
    q8_tile(weights.data_ptr(),tiles.data_ptr(),rows,columns,stream);
    const auto input=pattern({tokens,columns},.7f).to(tb::kCUDA);
    auto codes=tb::empty({int64_t(pm)*pk},weights.options()),scales=tb::empty({int64_t(pm)*(pk/32)},input.options());
    std::array<tb::Tensor,2> outputs{tb::full({tokens,rows},NAN,input.options()),tb::full({tokens,rows},INFINITY,input.options())};
    const auto launch=[&](int mode) {
        const int step=mode?tokens:kQ8TileRows;
        for(int first=0;first<tokens;first+=step) {
            const int count=std::min(step,tokens-first);
            Q8TiledMatrix matrix{tiles.data_ptr(),outputs[mode].data_ptr<float>()+int64_t(first)*rows,rows,pn};
            q8_quantize(input.data_ptr<float>()+int64_t(first)*columns,codes.data_ptr(),scales.data_ptr<float>(),count,columns,stream);
            q8_gemm(&matrix,1,codes.data_ptr(),scales.data_ptr<float>(),count,columns,stream);
        }
    };
    for(int i=0;i<3;++i){launch(0);launch(1);}
    const auto a=outputs[0].cpu().contiguous(),b=outputs[1].cpu().contiguous();
    for(int64_t i=0;i<a.numel();++i)require(std::isfinite(a.data_ptr<float>()[i]) &&
        a.data_ptr<float>()[i]==b.data_ptr<float>()[i],"DFlash2 batched prompt projection changed an output or missed a tail");
    std::vector<double> gpu[2],wall[2];
    for(int trial=0;trial<7;++trial)for(int order=0;order<2;++order) {
        const int mode=(trial+order)%2;
        Event begin(0),end(0);begin.record(stream);const auto start=std::chrono::steady_clock::now();
        launch(mode);end.record(stream);end.synchronize();
        wall[mode].push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));gpu[mode].push_back(ms*1000);
    }
    for(int mode=0;mode<2;++mode) {
        std::sort(gpu[mode].begin(),gpu[mode].end());std::sort(wall[mode].begin(),wall[mode].end());
        std::cout<<"DFlash2 prompt projection M="<<tokens<<" N="<<rows<<" K="<<columns
            <<" batched="<<mode<<" gpu_us="<<gpu[mode][3]<<" wall_us="<<wall[mode][3]
            <<" median7=1 output_bit_exact=1 kernel_launches="<<2*(mode?1:(tokens+7)/8)<<'\n';
    }
}
void check_tiled_gemm() {
    using namespace mfq::cuda::dflash2;
    const auto host=tb::TensorOptions().device(tb::kCPU);
    for(const auto [columns,rows]:{std::pair{32,1},std::pair{544,33},std::pair{2560,64},std::pair{7680,96}}) {
        const int pk=(columns+511)/512*512,pn=(rows+31)/32*32;
        auto source=pattern({rows,columns},.9f);
        auto packed=tb::empty({int64_t(rows)*(columns/32)*34},host.dtype(tb::kUInt8));
        quantize_row_q8_0_ref(source.data_ptr<float>(),static_cast<block_q8_0*>(packed.data_ptr()),int64_t(rows)*columns);
        auto weights=packed.to(tb::kCUDA),tiles=tb::empty({int64_t(pn)*(pk/32)*34},weights.options());
        auto restored=tb::empty_like(weights);
        q8_tile(weights.data_ptr(),tiles.data_ptr(),rows,columns,mfq_current_cuda_stream());
        q8_untile(tiles.data_ptr(),restored.data_ptr(),rows,columns,mfq_current_cuda_stream());
        auto bytes=restored.cpu();
        require(std::memcmp(bytes.data_ptr(),packed.data_ptr(),packed.nbytes())==0,
            "DFlash2 Q8_0 tiling changed a weight code, scale, or unaligned tail");
        for(int tokens:{1,2,3,4,5,6,7,8,9,17,35}) {
            auto input=pattern({tokens,columns},.4f),x=input.to(tb::kCUDA);
            const int padded_tokens=(tokens+7)/8*8;
            auto codes=tb::empty({int64_t(padded_tokens)*pk},weights.options()),scales=tb::empty({int64_t(padded_tokens)*(pk/32)},x.options());
            q8_quantize(x.data_ptr<float>(),codes.data_ptr(),scales.data_ptr<float>(),tokens,columns,mfq_current_cuda_stream());
            auto cq=codes.cpu(),cs=scales.cpu();
            for(int t=0;t<padded_tokens;++t)for(int block=0;block<pk/32;++block) {
                float maximum=0;
                for(int i=0;i<32;++i)if(t<tokens && block*32+i<columns)
                    maximum=std::max(maximum,std::abs(input.data_ptr<float>()[t*columns+block*32+i]));
                require(std::abs(cs.data_ptr<float>()[t*(pk/32)+block]-maximum/127.f)<=1e-9f,
                    "DFlash2 Q8 activation scale or padded row differs");
                for(int i=0;i<32;++i) {
                    const auto value=t<tokens && block*32+i<columns?input.data_ptr<float>()[t*columns+block*32+i]:0.f;
                    const int expected=int(std::nearbyint(value*(maximum>0?127.f/maximum:0.f)));
                    require(static_cast<signed char>(cq.data_ptr<std::uint8_t>()[t*pk+block*32+i])==expected,
                        "DFlash2 Q8 activation code or padding differs");
                }
            }
            std::vector<tb::Tensor> outputs;
            Q8TiledMatrix matrices[3];
            for(int i=0;i<3;++i) {
                outputs.push_back(tb::empty({tokens,rows},x.options()));
                matrices[i]={tiles.data_ptr(),outputs.back().data_ptr<float>(),rows,pn};
            }
            q8_gemm(matrices,3,codes.data_ptr(),scales.data_ptr<float>(),tokens,columns,mfq_current_cuda_stream());
            const auto actual=outputs.front().cpu();
            for(int i=1;i<3;++i)require((outputs[i]-outputs[0]).abs().max().item<float>()==0.f,
                "DFlash2 grouped Q8 matrices used another segment's rows");
            const auto* blocks=static_cast<const block_q8_0*>(packed.data_ptr());
            for(int t=0;t<tokens;++t)for(int row=0;row<rows;++row) {
                float lanes[32]{};
                for(int step=0;step<pk/512;++step)for(int lane=0;lane<32;++lane) {
                    const int block=step*16+lane/2;
                    if(block>=columns/32)continue;
                    const auto& w=blocks[row*(columns/32)+block];
                    int sum=0;
                    for(int i=0;i<16;++i)sum+=int(w.qs[lane%2*16+i])*
                        int(static_cast<signed char>(cq.data_ptr<std::uint8_t>()[t*pk+block*32+lane%2*16+i]));
                    const float scale=ggml_fp16_to_fp32(w.d)*cs.data_ptr<float>()[t*(pk/32)+block];
                    lanes[lane]=std::fma(scale,float(sum),lanes[lane]);
                }
                for(int offset=16;offset;offset>>=1) {
                    float next[32];
                    for(int lane=0;lane<32;++lane)next[lane]=lanes[lane]+lanes[lane^offset];
                    std::copy_n(next,32,lanes);
                }
                const auto expected=bf16(lanes[0]);
                require(actual.data_ptr<float>()[t*rows+row]==expected,
                    "DFlash2 tiled GEMM differs from quantized CPU dot and BF16 rounding");
            }
        }
    }
}
void check_kernels() {
    for(int width:{37,128}) {
        auto gate=pattern({5,width},.3f).to(tb::kCUDA).to(tb::kFloat16);
        auto up=pattern({5,width},.7f).to(tb::kCUDA).to(tb::kFloat16);
        const auto joined=tb::cat({gate,up},-1);
        auto a=tb::empty_like(gate),b=tb::empty_like(gate);
        mfq::cuda::dflash2::swiglu(gate.data_ptr(),up.data_ptr(),a.data_ptr(),a.numel(),mfq_current_cuda_stream());
        mfq::cuda::dflash2::swiglu_joined(joined.data_ptr(),b.data_ptr(),b.numel(),width,mfq_current_cuda_stream());
        require((a.to(tb::kFloat32)-b.to(tb::kFloat32)).abs().max().item<float>()==0.f,
            "DFlash2 joined Gate/Up changed BF16 activation rounding");
    }
    using namespace mfq::cuda::dflash2;
    for(int width:{37,2560}) {
        auto x=pattern({7,width}),weight=pattern({width},.3f);
        auto gx=x.to(tb::kCUDA),gw=weight.to(tb::kCUDA),result=tb::empty_like(gx);
        rms_norm(gx.data_ptr<float>(),gw.data_ptr<float>(),result.data_ptr<float>(),7,width,1e-6f,mfq_current_cuda_stream());
        auto actual=result.cpu();
        for(int row=0;row<7;++row) {
            double sum=0;for(int k=0;k<width;++k)sum+=double(x.data_ptr<float>()[row*width+k])*x.data_ptr<float>()[row*width+k];
            const float inverse=float(1/std::sqrt(sum/width+1e-6));
            for(int k=0;k<width;++k) {
                const float expected=bf16(x.data_ptr<float>()[row*width+k]*inverse*weight.data_ptr<float>()[k]);
                const float value=actual.data_ptr<float>()[row*width+k];
                require(value==bf16(value) && std::abs(value-expected)<.008f*std::max(.01f,std::abs(expected)),
                    "DFlash2 fused RMS changed its BF16 rounding or row reduction");
            }
        }
    }
    {
        auto gate=pattern({35},.2f).to(tb::kFloat16),up=pattern({35},.7f).to(tb::kFloat16);
        auto gg=gate.to(tb::kCUDA),gu=up.to(tb::kCUDA),out=tb::empty_like(gg);
        swiglu(gg.data_ptr(),gu.data_ptr(),out.data_ptr(),35,mfq_current_cuda_stream());
        auto expected=tb::empty({35},gate.options().dtype(tb::kFloat32));
        auto g=gate.to(tb::kFloat32),u=up.to(tb::kFloat32);
        for(int i=0;i<35;++i)expected.data_ptr<float>()[i]=bf16((g.data_ptr<float>()[i]/(1+std::exp(-g.data_ptr<float>()[i])))*u.data_ptr<float>()[i]);
        const auto actual=out.to(tb::kFloat32).cpu(),reference=expected.to(tb::kFloat16).to(tb::kFloat32);
        for(int i=0;i<35;++i)require(std::abs(actual.data_ptr<float>()[i]-reference.data_ptr<float>()[i])<.0005f,
            "DFlash2 fused SwiGLU changed BF16-to-FP16 rounding or its tail");
    }
    constexpr int rows=4, width=80, group=16, heads=6, kv_heads=2, dimension=96, window=129;
    constexpr int64_t position=300;
    auto x=pattern({rows,width}), dynamic=pattern({rows,width/group*4},.3f), base=pattern({4,width},.8f);
    auto residual=pattern({rows,width},1.2f);
    auto gx=x.to(tb::kCUDA), gd=dynamic.to(tb::kCUDA), gb=base.to(tb::kCUDA), gr=residual.to(tb::kCUDA);
    auto output=tb::empty_like(gx);
    for(bool finish : {false,true}) {
        convolution(gx.data_ptr<float>(),gd.data_ptr<float>(),gb.data_ptr<float>(),gr.data_ptr<float>(),
            output.data_ptr<float>(),rows,width,group,finish,mfq_current_cuda_stream());
        const auto actual=output.cpu();
        for(int row=0;row<rows;++row)for(int channel=0;channel<width;++channel) {
            const int d=row*(width/group)*4+int(finish)*(width/group)*2+channel/group;
            const int b=int(finish)*width*2+channel;
            float expected=x.data_ptr<float>()[row*width+channel]*(base.data_ptr<float>()[b]+dynamic.data_ptr<float>()[d]);
            if(row)expected+=x.data_ptr<float>()[(row-1)*width+channel]*(base.data_ptr<float>()[b+width]+dynamic.data_ptr<float>()[d+width/group]);
            if(finish)expected+=residual.data_ptr<float>()[row*width+channel];
            require(std::abs(actual.data_ptr<float>()[row*width+channel]-bf16(expected))<.004f,
                "DFlash2 grouped convolution differs from CPU arithmetic");
        }
    }
    auto q=pattern({rows,heads,dimension}), k=pattern({window,kv_heads,dimension},.7f), v=pattern({window,kv_heads,dimension},1.3f);
    auto bk=pattern({rows,kv_heads,dimension},.4f), bv=pattern({rows,kv_heads,dimension},.9f);
    auto gq=q.to(tb::kCUDA),gk=k.to(tb::kCUDA),gv=v.to(tb::kCUDA),g_bk=bk.to(tb::kCUDA),g_bv=bv.to(tb::kCUDA);
    auto partial=tb::empty({(window-1+rows+127)/128,rows,heads,dimension+2},gq.options());
    output=tb::empty_like(gq);
    attention(gq.data_ptr<float>(),gk.data_ptr<float>(),gv.data_ptr<float>(),g_bk.data_ptr<float>(),g_bv.data_ptr<float>(),
        partial.data_ptr<float>(),output.data_ptr<float>(),rows,heads,kv_heads,dimension,window,position,mfq_current_cuda_stream());
    auto actual=output.cpu();
    for(int row=0;row<rows;++row)for(int head=0;head<heads;++head) {
        std::vector<double> sum(dimension,0); double mass=0;
        const int kv=head/(heads/kv_heads);
        const auto add=[&](const float* key,const float* value) {
            double score=0;
            for(int d=0;d<dimension;++d)score+=double(q.data_ptr<float>()[(row*heads+head)*dimension+d])*key[d];
            const auto probability=std::exp(score/std::sqrt(double(dimension)));mass+=probability;
            for(int d=0;d<dimension;++d)sum[d]+=probability*value[d];
        };
        for(int64_t token=position+row-window+1;token<position;++token) {
            const auto offset=((token%window)*kv_heads+kv)*dimension;
            add(k.data_ptr<float>()+offset,v.data_ptr<float>()+offset);
        }
        for(int token=0;token<rows;++token) {
            const auto offset=(token*kv_heads+kv)*dimension;
            add(bk.data_ptr<float>()+offset,bv.data_ptr<float>()+offset);
        }
        for(int d=0;d<dimension;++d)require(std::abs(actual.data_ptr<float>()[(row*heads+head)*dimension+d]-bf16(float(sum[d]/mass)))<.002f,
            "DFlash2 ring/block attention differs from CPU softmax");
    }
    // A single captured drafter must read the updated position after every
    // committed window, including the first partial ring and wraparound.
    auto device_position=tb::full({1},position,gq.options().dtype(tb::kInt64));
    auto captured_output=tb::empty_like(gq);
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(mfq_current_cuda_stream(),cudaStreamCaptureModeThreadLocal));
    attention(gq.data_ptr<float>(),gk.data_ptr<float>(),gv.data_ptr<float>(),g_bk.data_ptr<float>(),g_bv.data_ptr<float>(),
        partial.data_ptr<float>(),captured_output.data_ptr<float>(),rows,heads,kv_heads,dimension,window,0,
        mfq_current_cuda_stream(),device_position.data_ptr<int64_t>());
    TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(mfq_current_cuda_stream(),&graph));
    TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,0));
    for(const int64_t next_position:{int64_t(1),int64_t(129),int64_t(300),int64_t(511)}) {
        device_position.fill_(next_position);
        TYLOINFER_CUDA_CHECK(cudaGraphLaunch(executable,mfq_current_cuda_stream()));
        attention(gq.data_ptr<float>(),gk.data_ptr<float>(),gv.data_ptr<float>(),g_bk.data_ptr<float>(),g_bv.data_ptr<float>(),
            partial.data_ptr<float>(),output.data_ptr<float>(),rows,heads,kv_heads,dimension,window,next_position,mfq_current_cuda_stream());
        require((captured_output-output).abs().max().item<float>()==0.f,
            "DFlash2 graph retained a previous window position");
    }
    TYLOINFER_CUDA_CHECK(cudaGraphExecDestroy(executable));TYLOINFER_CUDA_CHECK(cudaGraphDestroy(graph));
    auto norm=pattern({dimension},.2f);
    for(int d=0;d<dimension;++d)norm.data_ptr<float>()[d]+=1.f;
    auto positions=tb::arange(1000000,1000000+rows,tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64));
    auto gn=norm.to(tb::kCUDA),gp=positions.to(tb::kCUDA);
    norm_rope(gq.data_ptr<float>(),gn.data_ptr<float>(),gp.data_ptr<int64_t>(),output.data_ptr<float>(),
        rows,heads,dimension,10000.f,1e-6f,mfq_current_cuda_stream());
    actual=output.cpu();
    for(int row=0;row<rows;++row)for(int head=0;head<heads;++head) {
        const auto* values=q.data_ptr<float>()+(row*heads+head)*dimension;
        double square=0;for(int d=0;d<dimension;++d)square+=double(values[d])*values[d];
        const float inverse=float(1/std::sqrt(square/dimension+1e-6));
        for(int d=0;d<dimension;++d) {
            const int partner=d<dimension/2?d+dimension/2:d-dimension/2;
            const double angle=double(positions.data_ptr<int64_t>()[row])*std::pow(10000.,-2.*(d%(dimension/2))/dimension);
            const float direct=bf16(values[d]*inverse*norm.data_ptr<float>()[d]);
            const float other=bf16(values[partner]*inverse*norm.data_ptr<float>()[partner]);
            const auto expected=bf16(direct*float(std::cos(angle))+(d<dimension/2?-other:other)*float(std::sin(angle)));
            require(std::abs(actual.data_ptr<float>()[(row*heads+head)*dimension+d]-expected)<.02f,
                "DFlash2 RMS/RoPE differs from CPU arithmetic");
        }
    }
}
void fixture(const std::filesystem::path& directory,int layers=1,bool attention_sensitive=false) {
    using Json = nlohmann::json;
    std::filesystem::create_directories(directory);
    Json config{{"model_type", "qwen3"}, {"hidden_size", 64}, {"vocab_size", 1024}, {"num_hidden_layers", layers},
        {"intermediate_size", 64}, {"num_attention_heads", 3}, {"num_key_value_heads", 1}, {"head_dim", 64},
        {"sliding_window", 16}, {"rms_norm_eps", 1e-6}, {"rope_parameters", {{"rope_theta", 10000}}}};
    config["dflash_config"] = {{"block_size", 8}, {"conv_group_size", 16}, {"conv_kernel_size", 2},
        {"mask_token_id", 1023}, {"selector_rank", 33}, {"selector_top_k", 16}, {"target_layer_ids", {0}}};
    std::ofstream(directory / "config.json") << config;
    Json header = Json::object();
    std::vector<std::uint16_t> bytes;
    const auto add = [&](const std::string& name, const std::vector<int64_t>& shape, float value = 0.f) {
        int64_t count = 1; for (auto size : shape) count *= size;
        const auto begin = bytes.size() * 2;
        bytes.insert(bytes.end(), count, std::uint16_t(std::bit_cast<std::uint32_t>(value) >> 16));
        if(attention_sensitive && (name=="fc.weight" ||
                (name.find("layers.")==0 && name.find("norm.weight")==std::string::npos)))
            for(int64_t i=0;i<count;++i) {
                const float v=value+.025f*std::sin(float(i)*.17f+float(name.size()));
                bytes[begin/2+i]=std::uint16_t(std::bit_cast<std::uint32_t>(v)>>16);
            }
        header[name] = {{"dtype", "BF16"}, {"shape", shape}, {"data_offsets", {begin, bytes.size() * 2}}};
    };
    add("fc.weight", {64,64},.01f); add("hidden_norm.weight", {64}, 1); add("norm.weight", {64}, 1);
    add("candidate_selector.hidden_projection.weight", {33,64}, .01f);
    add("candidate_selector.predecessor_codebook", {1024,33}, 1);
    const auto successor_begin = bytes.size();
    add("candidate_selector.successor_codebook", {1024,33});
    std::fill_n(bytes.begin() + successor_begin + 777 * 33, 33, std::uint16_t(0x3f80));
    for(int layer=0;layer<layers;++layer) {
    const std::string p = "layers."+std::to_string(layer)+".";
    for (const auto* name : {"input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm"})
        add(p + name + ".weight", {64}, 1);
    for (const auto* name : {"attention_conv", "mlp_conv"}) {
        add(p + name + ".base_kernel", {2,2,64},attention_sensitive?.2f:0.f); add(p + name + ".kernel_projection.weight", {16,64});
    }
    add(p + "self_attn.q_proj.weight", {192,64});
    add(p + "self_attn.k_proj.weight", {64,64},.02f); add(p + "self_attn.v_proj.weight", {64,64},.03f);
    add(p + "self_attn.o_proj.weight", {64,192});
    for (const auto* name : {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}) add(p + name + ".weight", {64,64});
    }
    auto encoded = header.dump(); const std::uint64_t length = encoded.size();
    std::ofstream output(directory / "model.safetensors", std::ios::binary);
    output.write(reinterpret_cast<const char*>(&length), sizeof(length)); output.write(encoded.data(), encoded.size());
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size() * 2);
}
struct GraphMode {
    bool present;std::string old;
    explicit GraphMode(const char* value):present(std::getenv("TYLOINFER_DFLASH2_GRAPH")!=nullptr) {
        if(present)old=std::getenv("TYLOINFER_DFLASH2_GRAPH");set(value);
    }
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("TYLOINFER_DFLASH2_GRAPH",value?value:"");
#else
        if(value)setenv("TYLOINFER_DFLASH2_GRAPH",value,1);else unsetenv("TYLOINFER_DFLASH2_GRAPH");
#endif
    }
    ~GraphMode(){set(present?old.c_str():nullptr);}
};
struct ContextOverlapMode {
    bool present;std::string old;
    explicit ContextOverlapMode(const char* value):present(std::getenv("TYLOINFER_DFLASH2_CONTEXT_OVERLAP")!=nullptr) {
        if(present)old=std::getenv("TYLOINFER_DFLASH2_CONTEXT_OVERLAP");set(value);
    }
    static void set(const char* value) {
#ifdef _WIN32
        _putenv_s("TYLOINFER_DFLASH2_CONTEXT_OVERLAP",value?value:"");
#else
        if(value)setenv("TYLOINFER_DFLASH2_CONTEXT_OVERLAP",value,1);else unsetenv("TYLOINFER_DFLASH2_CONTEXT_OVERLAP");
#endif
    }
    ~ContextOverlapMode(){set(present?old.c_str():nullptr);}
};
void context_overlap_reference(const std::filesystem::path& directory) {
    fixture(directory,3,true);
    CudaExecutionContext execution;
    mfq::cuda::Qwen4CausalLm target;target.execution=&execution;
    target.metadata.hidden_size=64;target.metadata.vocab_size=1024;target.metadata.num_hidden_layers=1;
    target.embed.kind=target.lm_head.kind=QuantLinearKind::Dense;
    target.embed.dense=pattern({1024,64},.3f).to(tb::kCUDA);
    target.lm_head.dense=pattern({1024,64},.7f).to(tb::kCUDA);
    GraphMode graphs("0");ContextOverlapMode overlap("0");
    auto eager=mfq::cuda::dflash2::load(execution,target,directory);
    GraphMode::set("1");auto serial=mfq::cuda::dflash2::load(execution,target,directory);
    ContextOverlapMode::set("1");auto parallel=mfq::cuda::dflash2::load(execution,target,directory);
    const auto history=pattern({1,35,64},.1f).to(tb::kCUDA);
    GraphMode::set("0");eager->seed_target_context(history,35);
    GraphMode::set("1");serial->seed_target_context(history,35);parallel->seed_target_context(history,35);
    const auto anchor=tb::full({1,1},3,history.options().dtype(tb::kInt64));
    const MtpTarget projection(target);
    const auto exact=[](const tb::Tensor& a,const tb::Tensor& b,const char* message) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        require(x.scalar_type()==y.scalar_type() && x.sizes()==y.sizes() &&
            !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()),message);
    };
    // Prompt splitting must not switch the draft's arithmetic. The same raw
    // features and absolute positions must produce the same committed rings.
    auto partitioned=mfq::cuda::dflash2::load(execution,target,directory);
    for(int offset=0;offset<history.size(1);offset+=8)
        partitioned->append_target_context(history.narrow(1,offset,std::min<int64_t>(8,history.size(1)-offset)),offset);
    const auto seed_last=history.narrow(1,history.size(1)-1,1);
    const auto bulk_state=serial->capture_session_state(history.size(1),seed_last);
    const auto split_state=partitioned->capture_session_state(history.size(1),seed_last);
    for(std::size_t layer=0;layer<bulk_state.qwen4_blocks.size();++layer) {
        exact(bulk_state.qwen4_blocks[layer].keys,split_state.qwen4_blocks[layer].keys,
            "DFlash2 prompt splitting changed the committed K ring");
        exact(bulk_state.qwen4_blocks[layer].values,split_state.qwen4_blocks[layer].values,
            "DFlash2 prompt splitting changed the committed V ring");
    }
    partitioned.reset();
    int64_t position=35;int step=0;
    for(int rows:{1,4,2,3,8,1,7,4,2,3,1,8,5,6,2,4}) {
        const auto features=pattern({1,rows,64},.31f*(++step)).to(tb::kCUDA);
        GraphMode::set("0");eager->append_target_context(features,position);
        const auto expected=eager->draft_block(projection,anchor,3).clone();
        GraphMode::set("1");serial->append_target_context(features,position);
        const auto baseline=serial->draft_block(projection,anchor,3).clone();
        parallel->append_target_context_for_draft(features,position);
        const auto actual=parallel->draft_block(projection,anchor,3).clone();
        // No host copy between parallel append and block launch: the event
        // edges themselves must protect its attention reads of the KV rings.
        exact(expected,baseline,"DFlash2 serial graph changed the nonzero-attention eager result");
        exact(expected,actual,"DFlash2 overlapped block read stale or partially updated KV");
        position+=rows;const auto last=features.narrow(1,rows-1,1);
        const auto saved=eager->capture_session_state(position,last);
        const auto state=parallel->capture_session_state(position,last);
        for(int layer=0;layer<3;++layer) {
            exact(saved.qwen4_blocks[layer].keys,state.qwen4_blocks[layer].keys,"DFlash2 overlapped key ring differs");
            exact(saved.qwen4_blocks[layer].values,state.qwen4_blocks[layer].values,"DFlash2 overlapped value ring differs");
        }
        if(step==1) {
            auto empty=saved;
            for(auto& layer:empty.qwen4_blocks) {
                layer.keys=tb::zeros(layer.keys.sizes(),layer.keys.options());
                layer.values=tb::zeros(layer.values.sizes(),layer.values.options());
            }
            serial->restore_session_state(empty);
            const auto missing=serial->draft_block(projection,anchor,3);
            require((missing-expected).abs().max().item<float>()>1e-6f,
                "DFlash2 overlap fixture failed to make logits depend on committed KV");
            serial->restore_session_state(saved);
        }
        if(step==4 || step==9) {
            {
                const MfqCudaStreamGuard other(mfq_get_stream_from_pool(false,0));
                parallel->reset();parallel->restore_session_state(saved);
            }
            // Restore writes ran on another stream; the next block must wait
            // for those writes even though its last update used per-layer gates.
            exact(expected,parallel->draft_block(projection,anchor,3),"DFlash2 cross-stream prefix restore raced draft attention");
        }
        if(step==7) {
            const auto next=pattern({1,2,64},1.7f).to(tb::kCUDA);
            for(int row=0;row<2;++row) {
                GraphMode::set("0");eager->append_target_context(next.narrow(1,row,1),position+row);
                GraphMode::set("1");serial->append_target_context(next.narrow(1,row,1),position+row);
                parallel->append_target_context_for_draft(next.narrow(1,row,1),position+row);
            }
            position+=2;
        }
    }
    parallel.reset();serial.reset();eager.reset();
    std::cout<<"DFlash2 three-layer context overlap exact eager/serial logits and KV, ring wraps, consecutive commits, cross-stream prefix restore PASS\n";
}
void context_graph_reference(CudaExecutionContext& execution,mfq::cuda::Qwen4CausalLm& target,
    const std::filesystem::path& directory) {
    GraphMode mode("0");auto eager=mfq::cuda::dflash2::load(execution,target,directory);
    auto graphed=mfq::cuda::dflash2::load(execution,target,directory);
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    auto history=tb::full({1,33,64},.2,options);
    eager->seed_target_context(history,33);graphed->seed_target_context(history,33);
    int64_t position=33;int step=0;
    for(int rows:{1,4,2,1,4}) {
        auto features=tb::full({1,rows,64},(++step%2)?.3:-.4,options);
        GraphMode::set("0");eager->append_target_context(features,position);
        GraphMode::set("1");graphed->append_target_context(features,position);position+=rows;
        auto last=features.narrow(1,rows-1,1);
        const auto expected=eager->capture_session_state(position,last),actual=graphed->capture_session_state(position,last);
        const auto bits=[](const tb::Tensor& a,const tb::Tensor& b) {
            const auto x=a.cpu(),y=b.cpu();return x.nbytes()==y.nbytes() && !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes());
        };
        require(bits(expected.qwen4_blocks[0].keys,actual.qwen4_blocks[0].keys) &&
                bits(expected.qwen4_blocks[0].values,actual.qwen4_blocks[0].values),
                "DFlash2 context graph changed committed rings or reused stale positions/activations");
        require(actual.qwen4_blocks[0].values.abs().max().item<float>()>0,
                "DFlash2 context graph fixture did not exercise nonzero KV writes");
        if(step==2 || step==3) {
            const auto expected_keys=expected.qwen4_blocks[0].keys.cpu(),expected_values=expected.qwen4_blocks[0].values.cpu();
            const mfq::cuda::StreamGuard other(mfq::cuda::stream_from_pool());
            const auto snapshot=graphed->capture_session_state(position,last);
            require(bits(snapshot.qwen4_blocks[0].keys,expected_keys) && bits(snapshot.qwen4_blocks[0].values,expected_values),
                "DFlash2 asynchronous context snapshot raced another stream");
            if(step==3){graphed->reset();graphed->restore_session_state(expected);}
        }
        if(step==3)eager->restore_session_state(expected);
    }
    graphed.reset();eager.reset();
    std::cout<<"DFlash2 changing context graphs M1/2/4 + restored prefix exact KV bytes PASS\n";
}
void model_parity(const std::filesystem::path& directory,const std::filesystem::path& case_path,
                  const std::filesystem::path& real_head={},bool benchmark_overlap=false) {
    using namespace mfq::cuda;
    const auto config=mfq::cuda::dflash2::Config::load(directory);
    std::ifstream file(case_path,std::ios::binary);int32_t header[4]{};
    require(bool(file.read(reinterpret_cast<char*>(header),sizeof(header))),"DFlash2 parity case header missing");
    const int rows=header[0],anchor=header[1],vocabulary=header[2],width=config.hidden;
    require(rows>0 && anchor>=0 && anchor<config.vocabulary && vocabulary>0 && vocabulary<=config.vocabulary && config.vocabulary>=width,
        "DFlash2 parity case geometry differs");
    std::vector<float> taps(int64_t(rows)*width*config.target_layers.size()),embedding(2*width),reference(config.block*width);
    require(bool(file.read(reinterpret_cast<char*>(taps.data()),taps.size()*sizeof(float))) &&
        bool(file.read(reinterpret_cast<char*>(embedding.data()),embedding.size()*sizeof(float))),"DFlash2 parity features missing");
    const int64_t head_bytes=int64_t(vocabulary)*(width/32)*34;
    file.seekg(int64_t(vocabulary)*sizeof(int32_t)+head_bytes+(config.block-1)*(sizeof(int32_t)+sizeof(float)),std::ios::cur);
    require(bool(file.read(reinterpret_cast<char*>(reference.data()),reference.size()*sizeof(float))),"DFlash2 parity reference rows missing");
    CudaExecutionContext execution;Qwen4CausalLm target;target.execution=&execution;
    target.metadata.hidden_size=width;target.metadata.vocab_size=config.vocabulary;
    target.metadata.num_hidden_layers=config.target_layers.back()+1;
    target.embed.kind=target.lm_head.kind=QuantLinearKind::Dense;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat16);
    target.embed.dense=tb::zeros({config.vocabulary,width},options);
    auto embeddings=tb::tensor(embedding).reshape({2,width}).to(tb::kCUDA).to(tb::kFloat16);
    target.embed.dense.select(0,anchor).copy_(embeddings.select(0,0));
    target.embed.dense.select(0,config.mask).copy_(embeddings.select(0,1));
    // An identity target head reveals the already normalized draft rows in
    // its first H logits, without adding a production debugging API.
    if(real_head.empty()) {
        target.lm_head.dense=tb::zeros({config.vocabulary,width},options);
        auto identity=tb::zeros({width,width},options.device(tb::kCPU).dtype(tb::kFloat32));
        for(int i=0;i<width;++i)identity.data_ptr<float>()[int64_t(i)*(width+1)]=1.f;
        target.lm_head.dense.narrow(0,0,width).copy_(identity.to(tb::kCUDA).to(tb::kFloat16));
    } else {
        mfq::GgufModelSource source(real_head);
        target.lm_head.kind=QuantLinearKind::Gguf;
        target.lm_head.gguf=load_gguf_weight(source,"model.output.weight",target.embed.dense.get_device());
        require(target.lm_head.gguf.out==config.vocabulary && target.lm_head.gguf.neuron_len==width,
            "DFlash2 parity target head geometry differs");
        if(!benchmark_overlap) {
            const auto packed=target.lm_head.gguf.packed.cpu();
            std::ofstream output(case_path.parent_path()/"target-head.bin",std::ios::binary);
            output.write(static_cast<const char*>(packed.data_ptr()),packed.nbytes());
            require(bool(output),"Could not export target head for Strata parity");
        }
        std::cout<<"parity_head type="<<int(target.lm_head.gguf.type)<<" vocabulary="<<config.vocabulary<<'\n';
    }
    const auto history=tb::tensor(taps).reshape({1,rows,width*int64_t(config.target_layers.size())}).to(tb::kCUDA);
    auto id=tb::tensor(std::vector<int64_t>{anchor}).reshape({1,1}).to(tb::kCUDA);
    if(benchmark_overlap) {
        GraphMode graphs("1");ContextOverlapMode overlap("0");
        std::array<std::unique_ptr<MtpModule>,2> drafts;
        drafts[0]=mfq::cuda::dflash2::load(execution,target,directory);
        ContextOverlapMode::set("1");drafts[1]=mfq::cuda::dflash2::load(execution,target,directory);
        for(auto& draft:drafts)draft->seed_target_context(history,int64_t(rows)+header[3]);
        const MtpTarget projection(target);const auto stream=mfq_current_cuda_stream();
        std::vector<tb::Tensor> expected;float median[2]{};
        const auto round=[&](MtpModule& draft,int iteration) {
            const int count=1+iteration%4;
            const auto features=history.narrow(1,(iteration*13)%(rows-count+1),count);
            draft.append_target_context_for_draft(features,draft.cache_position());
            return draft.draft_block(projection,id,3);
        };
        for(int mode=0;mode<2;++mode) {
            for(int i=0;i<16;++i) {
                const auto logits=round(*drafts[mode],i);
                if(i>=12) {
                    const auto host=logits.cpu().contiguous();
                    if(!mode)expected.push_back(host.clone());
                    else require(std::memcmp(host.data_ptr(),expected[i-12].data_ptr(),host.nbytes())==0,
                        "Real-weight DFlash2 overlapped context changed draft logits");
                }
            }
            std::vector<float> samples;
            for(int sample=0;sample<7;++sample) {
                Event begin(0),end(0);begin.record(stream);
                for(int i=0;i<32;++i)(void)round(*drafts[mode],16+sample*32+i);
                end.record(stream);end.synchronize();
                float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));samples.push_back(ms*1000.f/32);
            }
            std::sort(samples.begin(),samples.end());median[mode]=samples[samples.size()/2];
            std::cout<<"dflash_context_draft_block mode="<<(mode?"overlap":"serial")<<" commit_rows=1..4 depth=3 median_us="<<median[mode]
                <<" sample_min_us="<<samples.front()<<" sample_max_us="<<samples.back()<<" real_weights=1 byte_exact=1\n";
        }
        drafts[1].reset();drafts[0].reset();
        std::cout<<"dflash_context_draft_block_gain percent="<<100.f*(median[0]-median[1])/median[0]<<'\n';return;
    }
    auto draft=mfq::cuda::dflash2::load(execution,target,directory);
    draft->seed_target_context(history,int64_t(rows)+header[3]);
    const int count=real_head.empty()?std::min(3,draft->maximum_draft_depth()):draft->maximum_draft_depth();
    if(!real_head.empty()) {
        const auto logits=draft->draft_block(MtpTarget(target),id,count).contiguous().cpu();
        const auto tokens=draft->draft_greedy_tokens(id);
        nlohmann::json report{{"tokens",tokens},{"position",int64_t(rows)+header[3]},{"requested",count}};
        std::ofstream(case_path.parent_path()/"native-head.json")<<report.dump(2)<<'\n';
        std::ofstream output(case_path.parent_path()/"native-head-logits.f32",std::ios::binary);
        output.write(static_cast<const char*>(logits.data_ptr()),logits.nbytes());
        require(bool(output),"Could not export native parity logits");
        std::cout<<"DFlash2 real input and target head "<<report.dump()<<'\n';
        return;
    }
    const auto logits=draft->draft_block(MtpTarget(target),id,count).narrow(-1,0,width).contiguous().cpu();
    double minimum=1.;
    for(int row=0;row<count;++row) {
        double dot=0,aa=0,bb=0;
        for(int d=0;d<width;++d) {
            const double a=logits.data_ptr<float>()[row*width+d],b=reference[(row+1)*width+d];
            dot+=a*b;aa+=a*a;bb+=b*b;
        }
        const double cosine=dot/std::sqrt(aa*bb+1e-30);minimum=std::min(minimum,cosine);
        std::cout<<"native draft row="<<row+1<<" cosine="<<cosine<<" norm_ratio="<<std::sqrt(aa/(bb+1e-30))<<'\n';
    }
    require(minimum>=.98,"DFlash2 real-weight block differs from the torch model reference");
    std::cout<<"DFlash2 real-weight full block numerical parity PASS\n";
}
void run(const std::filesystem::path& directory,bool shared_pool=false) {
    fixture(directory);
    CudaExecutionContext execution;
    mfq::cuda::Qwen4CausalLm target;
    target.execution = &execution;
    target.metadata.hidden_size = 64; target.metadata.vocab_size = 1024; target.metadata.num_hidden_layers = 1;
    target.embed.kind = target.lm_head.kind = QuantLinearKind::Dense;
    const auto options = tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    target.embed.dense = tb::full({1024,64}, .01, options);
    auto head = tb::full({1024,64}, -1000. / 64, options.device(tb::kCPU));
    for (int rank = 0; rank < 15; ++rank) head.select(0, 700 + rank).fill_((15.f - rank) / 64);
    head.select(0, 777).fill_(0);
    target.lm_head.dense = head.to(tb::kCUDA);
    const auto graph_context=mfq::cuda::default_context();
    if(shared_pool)graph_context->begin_graph_pool(mfq_current_cuda_stream());
    context_graph_reference(execution,target,directory);
    int workspace_notifications=0;
    auto draft = mfq::cuda::dflash2::load(execution, target, directory);
    draft->decode_workspace_changed=[&]{++workspace_notifications;};
    require(draft->blockwise_drafting(), "DFlash2 block adapter missing");
    require(!draft->extra_decode_workspace_bytes(3),"DFlash2 priced unprepared graph scratch as already resident");
    auto history = tb::zeros({1,33,64}, options);
    draft->seed_target_context(history, 33);
    require(draft->cache_position() == 33, "DFlash2 prompt tail did not advance");
    bool invalid = false;
    try { draft->append_target_context(history.narrow(1,0,1),34); } catch (const std::invalid_argument&) { invalid = true; }
    require(invalid && draft->cache_position() == 33, "DFlash2 accepted an uncommitted context gap");
    draft->append_target_context(history.narrow(1,0,1),33);
    require(draft->supports_session_state(), "DFlash2 disabled prefix reuse");
    const auto saved = draft->capture_session_state(34, history.narrow(1,0,1));
    draft->append_target_context(history.narrow(1,0,1),34);
    draft->restore_session_state(saved);
    auto legacy=saved;
    for(auto& block:legacy.qwen4_blocks){block.keys=block.keys.to(tb::kFloat32);block.values=block.values.to(tb::kFloat32);}
    draft->restore_session_state(legacy);
    const auto restored=draft->capture_session_state(34,saved.last_target_hidden);
    for(std::size_t i=0;i<restored.qwen4_blocks.size();++i) {
        for(const auto& pair:{std::pair{restored.qwen4_blocks[i].keys,saved.qwen4_blocks[i].keys},
                std::pair{restored.qwen4_blocks[i].values,saved.qwen4_blocks[i].values}}) {
            const auto a=pair.first.cpu(),b=pair.second.cpu();
            require(a.scalar_type()==b.scalar_type() && a.nbytes()==b.nbytes() &&
                !std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()),"Legacy F32 draft KV failed prefix restore");
        }
    }
    require(draft->cache_position() == 34 && saved.bytes > 64 * sizeof(float), "DFlash2 prefix snapshot lost its KV rings");
    const MtpTarget projection(target);
    auto previous = tb::full({1,1}, 3, options.dtype(tb::kInt64));
    auto logits = draft->draft_block(projection, previous, 3);
    const auto* graph_setting=std::getenv("TYLOINFER_DFLASH2_GRAPH");
    const bool graph_enabled=mfq_cuda_graph_capture_supported() && graph_context->supports_async_allocations() &&
        (!graph_setting || graph_setting[0]!='0');
    if(graph_enabled) {
        const auto bytes=draft->extra_decode_workspace_bytes(3),wide=draft->extra_decode_workspace_bytes(7);
        require(bytes && *bytes>0 && wide && *wide>=*bytes && workspace_notifications>0,
            "DFlash2 prepared graph did not publish its bounded transient requirement");
    } else require(!draft->extra_decode_workspace_bytes(3),"DFlash2 eager path discarded its conservative reserve");
    const auto prepared_notifications=workspace_notifications;
    require(logits.sizes().vec() == std::vector<int64_t>({1,3,1024}), "DFlash2 draft shape differs");
    for (int row = 0; row < 3; ++row) {
        auto selected = draft->draft_next(logits.narrow(1,row,1), previous);
        require(sample_greedy_cuda(selected.reshape({1,1024})).item<int64_t>() == 777,
            "DFlash2 selector ignored the predecessor/successor score");
        previous.fill_(777);
    }
    previous.fill_(3);
    (void)draft->draft_block(projection,previous,3);
    require(workspace_notifications==prepared_notifications,"DFlash2 restored workspace during an unchanged graph replay");
    require(draft->draft_greedy_tokens(previous)==std::vector<int32_t>({777,777,777}),
        "DFlash2 module greedy walk lost the current compact block or anchor");
    for(int depth:{4,6,7}) {
        require(draft->draft_block(projection,previous,depth).size(1)==depth,
            "DFlash2 eight-row block truncated a supported draft depth");
        require(draft->draft_greedy_tokens(previous)==std::vector<int32_t>(depth,777),
            "DFlash2 wide greedy walk lost a proposal or reused stale rows");
    }
    previous.fill_(777);
    draft->append_target_context(history.narrow(1,0,1),34);
    require(draft->draft_block(projection,previous,1).size(1)==1,
        "DFlash2 cached graph failed to change draft depth");
    draft->restore_session_state(saved);
    require(draft->draft_block(projection,previous,2).size(1)==2,
        "DFlash2 cached graph lost restored prefix rings");
    draft->reset();
    require(draft->cache_position() == 0, "DFlash2 reset retained context");
    draft->seed_target_context(history,33);
    require(draft->draft_block(projection,previous,3).size(1)==3,
        "DFlash2 cached graph could not restart a fresh request");
    draft.reset();
    require(graph_context->has_graph_pool(mfq_current_cuda_stream())==shared_pool,
        "DFlash2 released another model's graph pool or retained its own");
    if(shared_pool)graph_context->end_graph_pool(mfq_current_cuda_stream());
    mfq::cuda::Qwen4HiddenCapture capture;
    capture.layers = {0,2}; capture.begin(true);
    const auto hidden = tb::arange(48, options).reshape({1,2,24});
    capture.capture(0, hidden, 4); capture.capture(2, hidden + 10, 4);
    const auto combined = capture.result({});
    require(combined.sizes().vec() == std::vector<int64_t>({1,2,12}), "DFlash2 target feature width differs");
    const auto first = hidden.reshape({1,2,4,6}).mean(2);
    require((combined.narrow(-1,0,6)-first).abs().max().item<float>() == 0.f &&
        (combined.narrow(-1,6,6)-first-10).abs().max().item<float>() == 0.f,
        "DFlash2 target features are not the mean of the residual streams");
}
}
int main(int argc, char** argv) {
    // These checks exercise host packing and must work without a CUDA device.
    if(argc>=2 && (std::string(argv[1])=="--weight-quant-check" || std::string(argv[1])=="--weight-quant-bench")) {
        try {
            if(argc==2 && std::string(argv[1])=="--weight-quant-check")check_weight_quantization();
            else if(argc==4)benchmark_weight_quantization(std::stoi(argv[2]),std::stoi(argv[3]));
            else throw std::invalid_argument("weight quantization test arguments differ");
            return 0;
        }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
    }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || !count) return 77;
    try {
        // Keep the stream owner alive across short, independently allocated
        // fixtures and their graph executables, including instrumented runs.
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        std::cout<<std::unitbuf;
        if(argc==4 && std::string(argv[1])=="--greedy-packet-bench"){benchmark_greedy_packet(std::stoi(argv[2]),std::stoi(argv[3]));return 0;}
        if(argc==2 && std::string(argv[1])=="--greedy-selector-check"){check_greedy_selector_walk();return 0;}
        if(argc==2 && std::string(argv[1])=="--greedy-probability-check"){check_greedy_probability();check_combined_sampling_packet();check_greedy_packet_export();return 0;}
        if(argc==3 && std::string(argv[1])=="--context-overlap-check"){context_overlap_reference(argv[2]);return 0;}
        if(argc==2 && std::string(argv[1])=="--bf16-kv-check"){check_bf16_context();check_tiled_attention();return 0;}
        if(argc==2 && std::string(argv[1])=="--mapped-codebook-check"){check_mapped_selector();return 0;}
        if(argc==2 && std::string(argv[1])=="--staged-codebook-check"){check_mapped_selector(true);return 0;}
        if(argc==2 && std::string(argv[1])=="--pointwise-check"){check_pointwise_q8();return 0;}
        if(argc==3 && std::string(argv[1])=="--pointwise-bench"){pointwise_benchmark(std::stoi(argv[2])!=0);return 0;}
        if(argc==4 && std::string(argv[1])=="--model-parity"){model_parity(argv[2],argv[3]);return 0;}
        if(argc==5 && std::string(argv[1])=="--model-parity-head"){model_parity(argv[2],argv[3],argv[4]);return 0;}
        if(argc==5 && std::string(argv[1])=="--context-overlap-bench"){model_parity(argv[2],argv[3],argv[4],true);return 0;}
        if(argc==2 && std::string(argv[1])=="--rotary-bench"){rotary_benchmark();return 0;}
        if(argc>=2 && std::string(argv[1])=="--projection-bench"){projection_benchmark(argc>2?std::stoi(argv[2]):0);return 0;}
        if(argc==5 && std::string(argv[1])=="--prompt-projection-bench"){prompt_projection_benchmark(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]));return 0;}
        if(argc==10 && std::string(argv[1])=="--context-attention-bench"){
            context_attention_benchmark(std::stoi(argv[2]),std::stoi(argv[3]),std::stoi(argv[4]),std::stoi(argv[5]),
                std::stoi(argv[6]),std::stoi(argv[7]),std::stoi(argv[8]),std::stoi(argv[9]));return 0;}
        if(argc>=2 && std::string(argv[1])=="--attention-bench"){attention_benchmark(argc>2 && std::stoi(argv[2])==16,argc>3?std::stoi(argv[3]):0);return 0;}
        if(argc==2 && (std::string(argv[1])=="--normalization-bench" || std::string(argv[1])=="--normalization-fused-bench")){normalization_benchmark(std::string(argv[1])=="--normalization-fused-bench");return 0;}
        if (argc != 2) throw std::invalid_argument("DFlash2 test requires a workspace fixture directory");
        check_greedy_selector_walk();check_device_enqueue();check_mapped_selector();check_mapped_selector(true);check_bf16_context();check_pointwise_q8();check_rms_q8();check_tiled_attention();check_rotary_table();check_tiled_gemm();check_kernels(); run(argv[1]);run(std::filesystem::path(argv[1])/"shared-pool",true);context_overlap_reference(std::filesystem::path(argv[1])/"overlap");
        std::cout << "DFlash2 convolution/RMS/RoPE/attention/loader/block draft/selector/ring/target capture passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
