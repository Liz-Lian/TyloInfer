#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/row_residual.h"
#include <memory>

class GgufRowCorrection {
    struct Storage;
    std::shared_ptr<Storage> storage_;
    std::int64_t first_=0,rows_=0,column_first_=0,columns_=0;
public:
    GgufRowCorrection(std::shared_ptr<const mfq::RowResidualFile>,int device,bool mapped_host=false);
    std::shared_ptr<const GgufRowCorrection> slice(std::int64_t begin,std::int64_t end,
        std::int64_t column_begin,std::int64_t column_end) const;
    mfq_tensor_backend::Tensor add(mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor output,
        mfq_tensor_backend::Tensor ids={},std::int64_t experts=1,bool routed_input=false) const;
    mfq_tensor_backend::Tensor add_embedding(mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor output) const;
    mfq_tensor_backend::Tensor dense(int device) const;
};
