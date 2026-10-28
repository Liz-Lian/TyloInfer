#include "models/deepseek_v4/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/ops.h"

#include "backends/cuda/engine/executor/mtp_impl.h"

#define TYLOINFER_INSTANTIATE_MTP(MODEL)                                             \
  template Generation run_mtp_generation<MODEL, MODEL>(                         \
      MODEL &, MODEL &, mfq::cuda::Executor*, mfq::cuda::Collectives&,           \
      mfq::cuda::Executor::Mode, mfq::cuda::RoutedCache&,                        \
      const mfq::models::CausalLmMetadata &, MtpModule &,                       \
      InferenceRequest &, InferenceOutput &, int64_t,                            \
      const mfq::cuda::PreparedPrompt *, std::size_t, mfq_tensor_backend::Tensor,     \
      mfq_tensor_backend::Tensor *, std::size_t, std::function<void()>)
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::MiniCPMO45CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::Gemma4CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::GlmDsaCausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::Glm5CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::Qwen4CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::DeepseekV4CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::DeepseekV41CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::NemotronHCausalLm);
#undef TYLOINFER_INSTANTIATE_MTP
