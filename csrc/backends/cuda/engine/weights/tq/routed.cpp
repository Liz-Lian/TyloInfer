#include "backends/cuda/engine/weights/tq/routed.h"
#include "backends/cuda/engine/weights/tq/cache.h"
#include "backends/cuda/ops/tq/include/moe.h"

#include "backend.h"

#include <stdexcept>
#include <utility>

namespace tb = mfq_tensor_backend;

tb::Tensor TqRoutedWeight::forward(tb::Tensor input_value,tb::Tensor ids)const {
    if(!cache || !source_id || experts<1 || out<1 || input<1 ||
       ids.dim()!=2 || ids.size(1)<1 || ids.size(1)>65535 || input_value.size(-1)!=input ||
       !input_value.is_cuda() || (input_value.dim()!=2 && input_value.dim()!=3))
        throw std::invalid_argument("invalid TQ routed activation geometry");
    const auto tokens=ids.size(0),routes=ids.size(1);
    if(input_value.size(0)!=tokens || (input_value.dim()==3 && input_value.size(1)!=routes))
        throw std::invalid_argument("TQ routed token/route count differs");
    const auto source=input_value.contiguous();
    auto result=tb::empty({tokens,routes,out},source.options());
    if(!tokens)return result;
    const MfqCudaGuard guard(source.get_device());
    auto host=ids.contiguous().cpu().to(tb::kInt64);
    const auto* selected=host.data_ptr<std::int64_t>();
    for(std::int64_t i=0;i<host.numel();++i) {
        if(selected[i]<0 || selected[i]>=experts)throw std::out_of_range("TQ routed expert ID is out of bounds");
    }
    const auto owners=cache->get(source_id,selected,host.numel(),source.get_device());
    tq_ops::routed_projection(owners,std::move(source),result);
    return result;
}
