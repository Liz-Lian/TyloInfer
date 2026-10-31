#pragma once
#include "../kernels/gguf.h"

// Exercise automatic recorded updates with a real GGUF Q8 output projection.
// Full-state checkpoints supply an independent reference for every accepted
// prefix, including graph replay, eager execution and explicit overrides.
inline void gdn_prepared_replay_default_case() {
    using namespace mfq::cuda;
    using namespace mfq::cuda::qwen4_exp;
    namespace g=mfq::cuda::gguf;
    struct ReplaySetting {
        const std::optional<std::string> old=std::getenv("TYLOINFER_MTP_GDN_REPLAY")
            ?std::optional<std::string>(std::getenv("TYLOINFER_MTP_GDN_REPLAY")):std::nullopt;
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("TYLOINFER_MTP_GDN_REPLAY",value?value:"");
#else
            if(value)setenv("TYLOINFER_MTP_GDN_REPLAY",value,1);
            else unsetenv("TYLOINFER_MTP_GDN_REPLAY");
#endif
        }
        ReplaySetting(){set(nullptr);}
        ~ReplaySetting(){set(old?old->c_str():nullptr);}
    } setting;
    const auto bits=[](const Tensor& actual,const Tensor& expected,const char* label) {
        const auto a=actual.contiguous().cpu(),b=expected.contiguous().cpu();
        if(a.sizes()!=b.sizes() || a.scalar_type()!=b.scalar_type() ||
                std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes()))
            throw std::runtime_error(std::string(label)+" changed bytes");
    };
    CudaExecutionContext execution;
    constexpr int hidden=32,nk=2,nv=6,width=128,columns=nv*width,channels=(2*nk+nv)*width;
    constexpr int row_bytes=columns/32*34;
    std::vector<std::uint8_t> bytes(hidden*row_bytes);
    for(std::size_t block=0;block<bytes.size()/34;++block) {
        const std::uint16_t scale=0x211f;
        std::memcpy(bytes.data()+block*34,&scale,sizeof(scale));
        for(int j=0;j<32;++j)bytes[block*34+2+j]=std::uint8_t(int((block*13+j*17)%31)-15);
    }
    auto host=tb::empty({int64_t(bytes.size())},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
    std::memcpy(host.data_ptr(),bytes.data(),bytes.size());
    const auto packed=host.to(tb::kCUDA);
    const auto project=[packed](const Tensor& input,int64_t tokens,tb::ScalarType dtype) {
        auto out=tb::empty({1,tokens,hidden},input.options().dtype(dtype));
        if(dtype==tb::kFloat16)g::matvec_q8_f16(GGML_TYPE_Q8_0,packed.data_ptr(),row_bytes,
            hidden,columns,input.data_ptr(),tokens,out.data_ptr(),mfq_current_cuda_stream());
        else g::matvec_q8(GGML_TYPE_Q8_0,packed.data_ptr(),row_bytes,
            hidden,columns,input.data_ptr(),tokens,out.data_ptr<float>(),mfq_current_cuda_stream());
        return out;
    };
    int checked=0;
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(bool tiled:{false,true})for(int tokens:{2,3,4}) {
        ReplaySetting::set(nullptr);
        const auto linear=[&](int in,int out,int salt) -> Linear {
            auto weight=data({out,in},salt,.03f).to(dtype);
            return [weight](CudaExecutionContext&,const Tensor& x) {
                return tb::matmul(x.to(weight.scalar_type()),weight.transpose(-1,-2));
            };
        };
        auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        GdnWeights weights{linear(hidden,channels,551),linear(hidden,columns,552),
            linear(hidden,nv,553),linear(hidden,nv,554),{},data({channels,4},555,.03f),
            tb::zeros({nv},gpu),tb::zeros({nv},gpu),tb::ones({width},gpu)};
        weights.output=[project](CudaExecutionContext&,const Tensor& x) {
            auto contiguous=x.contiguous();
            auto q=tb::empty({int64_t(x.size(1)*g::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
            if(x.scalar_type()==tb::kFloat16)g::quantize_input_f16(contiguous.data_ptr(),x.size(1),columns,q.data_ptr(),mfq_current_cuda_stream());
            else g::quantize_input(contiguous.data_ptr<float>(),x.size(1),columns,q.data_ptr(),mfq_current_cuda_stream());
            return project(q,x.size(1),x.scalar_type());
        };
        weights.output_q8=[project](CudaExecutionContext&,const Tensor& input,int64_t count,tb::ScalarType output_type,bool position_major) {
            if(position_major)throw std::runtime_error("Q8_0 output received an unsupported activation layout");
            return project(input,count,output_type);
        };
        const auto separate=weights;
        weights.input_projection=[separate](CudaExecutionContext& e,const Tensor& x) {
            return std::vector<Tensor>{separate.qkv(e,x),separate.gate(e,x),separate.alpha(e,x),separate.beta(e,x)};
        };
        Gdn actual(weights,nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
        Gdn reference(weights,nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
        auto history=data({1,3,hidden},556,.4f).to(dtype),input=data({1,tokens,hidden},557,.3f).to(dtype);
        actual.forward_chunk(execution,history,true);
        const auto initial_conv=actual.conv_state().clone(),initial_state=actual.graph_state()[1]->clone();
        reference.restore_session(initial_conv,initial_state);
        if(!actual.replay_checkpoint_supported())throw std::runtime_error("prepared GDN output did not enable automatic recorded updates");
        std::vector<Tensor> prefix_conv,prefix_state;
        for(int i=1;i<tokens;++i) {
            prefix_conv.push_back(tb::empty_like(initial_conv));prefix_state.push_back(tb::empty_like(initial_state));
        }
        const auto expected=reference.forward_graph(execution,input,1,prefix_conv,prefix_state).clone();
        if(expected.abs().max().item<float>()==0)throw std::runtime_error("GGUF output fixture is all zero");
        std::vector<Tensor> empty_conv,empty_state,updates;Tensor output;
        DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{output=actual.forward_graph(execution,input,1,empty_conv,empty_state,&updates);},[&] {
            actual.graph_state()[0]->copy_(initial_conv);actual.graph_state()[1]->copy_(initial_state);
        },[&]{output={};});
        for(int keep=1;keep<=tokens;++keep) {
            graph.run();bits(output,expected,"automatic prepared GDN graph output");
            bits(actual.conv_state(),initial_conv,"automatic prepared GDN read-only convolution");
            bits(*actual.graph_state()[1],initial_state,"automatic prepared GDN read-only recurrence");
            actual.bind_graph_replay(updates,17);
            if(keep==tokens)actual.commit();else actual.rollback(17+keep);
            bits(actual.conv_state(),keep==tokens?reference.conv_state():prefix_conv[keep-1],"automatic prepared GDN accepted convolution");
            bits(*actual.graph_state()[1],keep==tokens?*reference.graph_state()[1]:prefix_state[keep-1],"automatic prepared GDN accepted recurrence");
            actual.restore_session(initial_conv,initial_state,true);++checked;
        }
        for(const char* option:{static_cast<const char*>(nullptr),"0","1"}) {
            ReplaySetting::set(option);
            auto eager=actual.forward(execution,input,true,1,17);
            bits(eager,expected,"automatic/override prepared GDN eager output");
            if(!option || *option=='1')bits(*actual.graph_state()[1],initial_state,"prepared GDN eager verification mutated committed state");
            actual.rollback(17+tokens-1);
            bits(actual.conv_state(),prefix_conv[tokens-2],"prepared GDN eager rejected convolution");
            bits(*actual.graph_state()[1],prefix_state[tokens-2],"prepared GDN eager rejected recurrence");
            actual.restore_session(initial_conv,initial_state,true);++checked;
        }
        ReplaySetting::set(nullptr);
        (void)actual.forward(execution,input,true,1,17);actual.reset(true);
        if(actual.conv_state().abs().max().item<float>()!=0 || actual.graph_state()[1]->abs().max().item<float>()!=0)
            throw std::runtime_error("automatic prepared GDN reset committed a discarded window");
    }
    std::cout<<"GDN automatic prepared GGUF output graph/eager/override/prefix/reset cases="<<checked<<" PASS\n";
}
