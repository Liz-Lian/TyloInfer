#pragma once

#include "models/components.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/deepseek_v41/vision.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/minicpmo45/model.h"
#include "models/nemotron_h/ops.h"
#include "models/nemotron_h/radio_vision.h"
#include "models/qwen4_exp/ops.h"
#include "models/common/grid_vision.h"

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
    Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm>
load_runtime_components(
    mfq::cuda::Qwen4CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Glm5CausalLm>
load_runtime_components(
    mfq::cuda::Glm5CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm>
load_runtime_components(
    mfq::cuda::DeepseekV41CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::NemotronHCausalLm>
load_runtime_components(
    mfq::cuda::NemotronHCausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMO45CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::minicpmo45::Model>
load_runtime_components(
    mfq::cuda::minicpmo45::Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);
