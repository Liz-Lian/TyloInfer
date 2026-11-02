#include "engine/models/deepseek_v4/ops.h"
#include "engine/models/glm5_next/ops.h"
#include "engine/models/glm_dsa/ops.h"
#include "engine/models/minicpmo45/ops.h"
#include "engine/models/nemotron_h/ops.h"
#include "engine/models/qwen4_exp/mtp_ops.h"
#include "engine/models/deepseek_v41/mtp_ops.h"

#include "engine/mtp_impl.h"

#define TYLOINFER_INSTANTIATE_MTP(MODEL)                                             \
  template Generation run_mtp_generation<MODEL, MODEL>(                         \
      MODEL &, MODEL &, mfq::cuda::Executor*, mfq::cuda::Collectives&,           \
      mfq::cuda::Executor::Mode, mfq::cuda::RoutedCache&,                        \
      const mfq::models::CausalLmMetadata &, MtpModule &,                       \
      mfq::engine::mtp::State &,                                                \
      InferenceRequest &, InferenceOutput &, int64_t,                            \
      const mfq::cuda::PreparedPrompt *, std::size_t, mfq_tensor_backend::Tensor,     \
      mfq_tensor_backend::Tensor *, std::size_t, std::function<void()>)
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::MiniCPMO45CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::GlmDsaCausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::Glm5CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::Qwen4CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::DeepseekV4CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::DeepseekV41CausalLm);
TYLOINFER_INSTANTIATE_MTP(mfq::cuda::NemotronHCausalLm);
#undef TYLOINFER_INSTANTIATE_MTP
