#include "engine/weights/cache.h"
#include "engine/models/qwen35/forward.h"
#include "engine/models/qwen35/mtp.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/models/glm5_next/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/deepseek_v41/mtp_ops.h"
#endif
#include "engine/models/qwen4_exp/mtp_ops.h"

#include "engine/generation/mtp/cuda_impl.h"

#define TYLOINFER_INSTANTIATE_MTP(MODEL, STATE)                                      \
  template Generation run_mtp_generation<MODEL, STATE>(                         \
      MODEL &, STATE &, tyloi::cuda::Executor*, tyloi::cuda::Collectives&,           \
      tyloi::cuda::Executor::Mode, ::tyloi::weights::Cache&,                        \
      const tyloi::models::Info &, const tyloi::ModelSource*, tyloi::models::Mtp &,  \
      tyloi::engine::mtp::State &,                                                \
      InferenceRequest &, InferenceOutput &, int64_t,                            \
      const tyloi::models::Prepared *, std::size_t,                            \
      ::tyloi::Tensor, ::tyloi::Tensor *,      \
      std::size_t, tyloi::engine::Budget*)
#ifndef TYLOINFER_NATIVE_CUDA
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::glm5_next::Model, tyloi::cuda::glm5_next::State);
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::nemotron_h::Model, tyloi::cuda::nemotron_h::State);
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::deepseek_v41::Model, tyloi::cuda::deepseek_v41::State);
#endif
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::qwen4_exp::Model, tyloi::cuda::qwen4_exp::State);
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::qwen35::Model, tyloi::cuda::qwen35::State);
#undef TYLOINFER_INSTANTIATE_MTP
