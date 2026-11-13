#include "moe.h"

#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/kernels/mfq_cuda_shared_gate.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/weights/encoding/tq.h"
#include "env.h"

#include <algorithm>
#include <utility>

#include "engine/backends/cuda/executor/executor.h"
#include "matmul_rows.h"

namespace tyloi::cuda {
namespace {
using DenseTensor = ::tyloi::DenseTensor;
template<class Project>
DenseTensor preserve_rows(const DenseTensor& source,int64_t rows,Project project) {
    if(rows<=0 || source.size(0)<=rows)return project(source);
    DenseTensor output;
    for(int64_t begin=0;begin<source.size(0);begin+=rows) {
        const auto count=std::min<int64_t>(rows,source.size(0)-begin);
        auto part=project(source.narrow(0,begin,count));
        if(!output.defined()) {
            auto shape=part.sizes().vec();shape[0]=source.size(0);
            output=::tyloi::cuda::empty(shape,part.options());
        }
        output.narrow(0,begin,count).copy_(part);
    }
    return output;
}
}

MoeSharedExpert::MoeSharedExpert(::tyloi::Tensor gate,::tyloi::Tensor up,
        ::tyloi::Tensor down,::tyloi::Tensor scalar)
    : projections{std::move(gate),std::move(up),std::move(down)},scalar_gate(std::move(scalar)) {
    TYLOINFER_RUNTIME_CHECK(projections[0] && projections[1] && projections[2] && scalar_gate,
        "Qwen shared expert requires projection weights");
    const auto* g=projections[0].representation<tyloi::encoding::TqTensor>();
    const auto* u=projections[1].representation<tyloi::encoding::TqTensor>();
    tq=g && u && g->image().device().is_cuda() && u->image().device().is_cuda();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    row_parallel=std::all_of(projections.begin(),projections.end(),[](const auto& weight) {
        const auto* nint=weight.representation<tyloi::NintTensor>();
        return nint && !nint->q8_zero;
    });
#else
    row_parallel=false;
#endif
}

DenseTensor MoeSharedExpert::raw(const Projector& project,const DenseTensor& source) const {
    const auto& shared_g=projections[0];
    const auto& shared_u=projections[1];
    const auto& shared_d=projections[2];
    if(tq) {
        const std::array<::tyloi::Tensor, 2> weights{shared_g, shared_u};
        const auto group = project.group(weights);
        auto output=project.run(shared_d,project.swiglu(group,source));
        return ::tyloi::cuda::sigmoid(project.run(scalar_gate,source))*output;
    }
    DenseTensor hidden;
    {
        auto g=project.run(shared_g,source);
        auto u=project.run(shared_u,source);
        hidden=(g*::tyloi::cuda::sigmoid(g))*u;
    }
    auto output=project.run(shared_d,hidden);
    hidden={};
    return ::tyloi::cuda::sigmoid(project.run(scalar_gate,source))*output;
}

DenseTensor MoeSharedExpert::forward(const Projector& project,const DenseTensor& source,int64_t rows) const {
    const auto parallel_option=::tyloi::cuda::options::moe_prefill_shared_streams();
    const int streams=parallel_option?*parallel_option:0;
    const auto chain_option=::tyloi::cuda::options::moe_prefill_shared_chain_streams();
    const int chain_streams=chain_option?*chain_option:4;
    const auto tail=rows>0?source.size(0)%rows:0;
    if(chain_streams>0 && row_parallel && rows>8 && source.size(0)>=8192 &&
            source.size(0)>rows && (tail==0 || tail>8)) {
        // The first original chunk prepares all three cached weight
        // matrices. Worker pipelines then read them after the ready fence.
        auto first=raw(project,source.narrow(0,0,rows));
        auto shape=first.sizes().vec();shape[0]=source.size(0);
        auto output=::tyloi::cuda::empty(shape,first.options());
        output.narrow(0,0,rows).copy_(first);
        tyloi::cuda::submit_parallel_rows(source.get_device(),rows,source.size(0),rows,chain_streams,
            [&](int64_t begin,int64_t count) {
                auto part=raw(project,source.narrow(0,begin,count));
                output.narrow(0,begin,count).copy_(part);
            });
        return output;
    }
    if(streams>0 && row_parallel && rows>8 && source.size(0)>rows && (tail==0 || tail>8)) {
        // Each GEMM retains its original row count. Only its submission
        // stream changes; activations remain in their canonical FP16 form.
        const auto projection=[&](int index,const DenseTensor& input) {
            const tyloi::cuda::MatmulRowsScope gemm_rows(rows,streams);
            return project.run(projections[index],input);
        };
        auto gate=projection(0,source),up=projection(1,source);
        auto output=projection(2,(gate*::tyloi::cuda::sigmoid(gate))*up);
        auto scale=preserve_rows(source,rows,[&](const DenseTensor& part) {
            return ::tyloi::cuda::sigmoid(project.run(scalar_gate,part));
        });
        return scale*output;
    }
    return preserve_rows(source,rows,[&](const DenseTensor& part){return raw(project,part);});
}

DenseTensor MoeSharedExpert::gate(const Projector& project,const DenseTensor& input) const {
    const auto option=::tyloi::cuda::options::shared_gate_fused();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* nint_option=tyloi::cuda::env_lookup("TYLOINFER_SHARED_GATE_NINT");
    const auto* gate_nint=scalar_gate.representation<tyloi::NintTensor>();
#endif
    const auto* gate_dense=scalar_gate.representation<tyloi::DenseTensor>();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if((!option || *option) && (!nint_option || nint_option[0]!='0') &&
            gate_nint)
        if(auto output=try_nint_shared_gate_sigmoid_cuda(input,*gate_nint))return *output;
#endif
    if((!option || *option) && gate_dense)
        if(auto output=try_shared_gate_sigmoid_cuda(input,*gate_dense))return *output;
    return ::tyloi::cuda::sigmoid(project.run(scalar_gate,input));
}

DenseTensor MoeSharedExpert::forward(const DenseTensor& source,int64_t rows) const {
    auto& executor=Executor::current();
    return forward(executor.projector(Executor::current_ffn()),source,rows);
}
} // namespace tyloi::cuda
