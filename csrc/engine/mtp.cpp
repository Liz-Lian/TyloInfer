#include "engine/models/glm5_next/forward.h"
#include "engine/models/nemotron_h/forward.h"
#include "engine/models/qwen4_exp/mtp_ops.h"
#include "engine/models/deepseek_v41/mtp_ops.h"

#include "engine/mtp_impl.h"

#define TYLOINFER_INSTANTIATE_MTP(MODEL, STATE)                                      \
  template Generation run_mtp_generation<MODEL, STATE>(                         \
      MODEL &, STATE &, tyloi::cuda::Executor*, tyloi::cuda::Collectives&,           \
      tyloi::cuda::Executor::Mode, tyloi::cuda::RoutedCache&,                        \
      const tyloi::models::Info &, const tyloi::ModelSource*, tyloi::models::Mtp &,  \
      tyloi::engine::mtp::State &,                                                \
      InferenceRequest &, InferenceOutput &, int64_t,                            \
      const tyloi::cuda::PreparedPrompt *, std::size_t, ::tyloi::DenseTensor,     \
      ::tyloi::DenseTensor *, std::size_t, std::function<void()>)
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::glm5_next::Model, tyloi::cuda::glm5_next::State);
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::qwen4_exp::Model, tyloi::cuda::qwen4_exp::State);
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::deepseek_v41::Model, tyloi::cuda::deepseek_v41::State);
TYLOINFER_INSTANTIATE_MTP(tyloi::cuda::nemotron_h::Model, tyloi::cuda::nemotron_h::State);
#undef TYLOINFER_INSTANTIATE_MTP
