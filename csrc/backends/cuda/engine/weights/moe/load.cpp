#include "backends/cuda/engine/weights/moe/load.h"

#include "engine/weights/moe/prepare.h"
#include "engine/weights/quant.h"
#include "backends/cuda/engine/weights/load.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_set>

static void initialize_mfe_nint_runtime(mfq::cuda::Experts & result) {
    if (result.pools.empty()) {
        throw std::runtime_error("MFE NINT runtime requires at least one pool");
    }
    result.activation_geometries.clear();
    result.activation_workspace_domain = 1;
    std::unordered_set<int64_t> quantized_shapes;
    const auto target = result.pools.front().weight.q_packed.device();
    for (const auto & pool : result.pools) {
        TYLOINFER_RUNTIME_CHECK(
            pool.weight.q_packed.is_cuda() &&
            pool.weight.q_packed.device() == target,
            "MFE NINT pools must share one CUDA device");
        const int64_t activation_key =
            (pool.weight.gs << 32) ^ pool.weight.ng;
        if (quantized_shapes.insert(activation_key).second) {
            result.activation_geometries.push_back({
                static_cast<int>(pool.weight.ng),
                static_cast<int>(pool.weight.gs),
                0,
                0,
            });
        }
    }
}

mfq::cuda::Experts to_gpu_mfe(const mfq::mfe::Data & cpu) {
    mfq::cuda::Experts result;
    result.unified_nint_projection = true;
    result.n_experts = cpu.n_experts;
    result.out_per_expert = cpu.out_per_expert;
    result.neuron_len = cpu.neuron_len;
    result.pools.reserve(cpu.pools.size());
    for (const auto & source_pool : cpu.pools) {
        if (source_pool.dtype != "NINT") {
            throw std::runtime_error("non-NINT cohort reached the NINT-only loader");
        }
        MfePoolWeight pool;
        pool.local_experts = (int)source_pool.expert_ids.size();
        pool.weight = to_device_mfe_nint(
            source_pool.weight, pool.local_experts,
            cpu.out_per_expert, true);
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            const int expert = source_pool.expert_ids.at(static_cast<size_t>(index));
            local[static_cast<size_t>(expert)] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()}, mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(pool.weight.q_packed.device()).contiguous();
        result.pools.push_back(std::move(pool));
    }
    initialize_mfe_nint_runtime(result);
    return result;
}

mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
        const mfq_tensor_backend::Tensor & source) {
    return source.defined()
        ? source.to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device())).contiguous()
        : mfq_tensor_backend::Tensor();
}

static int64_t tensor_storage_bytes(const mfq_tensor_backend::Tensor & value) {
    return value.defined()
        ? value.numel() * (int64_t)value.element_size()
        : 0;
}

int64_t mixed_moe_storage_bytes(const MixedMoeRuntime & runtime) {
    int64_t bytes = 0;
    if(runtime.nint_dispatch) {
        const auto& d=*runtime.nint_dispatch;
        for(const auto& t:{d.map_ptrs,d.pool_params,d.expert_pool,d.expert_local})bytes+=tensor_storage_bytes(t);
    }
    for(const auto& item:runtime.nint_input_plans) {
        bytes+=tensor_storage_bytes(item.second.weight_ptrs)+tensor_storage_bytes(item.second.quantize_descriptors);
    }
    if (runtime.nvq_dispatch) {
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_ptrs);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->weight_sizes);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->pool_params);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_pool);
        bytes += tensor_storage_bytes(runtime.nvq_dispatch->expert_local);
    }
    for (const auto & pool : runtime.pools) {
        bytes += tensor_storage_bytes(pool.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            bytes += tensor_storage_bytes(pool.nint.q_packed);
            bytes += tensor_storage_bytes(pool.nint.row_q_bits);
            bytes += tensor_storage_bytes(pool.nint.row_q_bit_offsets);
            bytes += tensor_storage_bytes(pool.nint.sub_scale);
            bytes += tensor_storage_bytes(pool.nint.sub_min);
            bytes += tensor_storage_bytes(pool.nint.neuron_scale);
            bytes += tensor_storage_bytes(pool.nint.neuron_min);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            bytes += tensor_storage_bytes(pool.q8_zero.q_packed);
            bytes += tensor_storage_bytes(pool.q8_zero.q8_zero_scale);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            bytes += tensor_storage_bytes(pool.mxfp4.values);
            bytes += tensor_storage_bytes(pool.mxfp4.scales);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            bytes += tensor_storage_bytes(pool.mxfp4_sq.blob);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.mxfp4_sq.row_symbol_byte_offsets);
            bytes += tensor_storage_bytes(pool.mxfp4_sq.row_auxiliary);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            bytes += tensor_storage_bytes(pool.fp8_sq.blob);
            bytes += tensor_storage_bytes(pool.fp8_sq.row_q);
            bytes += tensor_storage_bytes(
                pool.fp8_sq.row_symbol_byte_offsets);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            bytes += tensor_storage_bytes(pool.nvq.indices_packed);
            bytes += tensor_storage_bytes(pool.nvq.aux_packed);
            bytes += tensor_storage_bytes(pool.nvq.sub_scale_packed);
            bytes += tensor_storage_bytes(pool.nvq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nvq.codebook);
            bytes += tensor_storage_bytes(pool.nvq.integer_codebook);
            if(pool.nvq.decode_records.defined() && pool.nvq.decode_records.data_ptr()!=pool.nvq.indices_packed.data_ptr())
                bytes += tensor_storage_bytes(pool.nvq.decode_records);
        } else {
            bytes += tensor_storage_bytes(pool.nepq.indices_packed);
            bytes += tensor_storage_bytes(pool.nepq.aux_packed);
            bytes += tensor_storage_bytes(pool.nepq.state_packed);
            bytes += tensor_storage_bytes(pool.nepq.neuron_scale);
            bytes += tensor_storage_bytes(pool.nepq.table_pool);
            bytes += tensor_storage_bytes(pool.nepq.grouped_table_pool);
            bytes += tensor_storage_bytes(pool.nepq.bank_ids);
            bytes += tensor_storage_bytes(pool.nepq.rotation_signs);
            bytes += tensor_storage_bytes(pool.nepq.residual_codebook);
            bytes += tensor_storage_bytes(pool.nepq.residual_first);
            bytes += tensor_storage_bytes(pool.nepq.residual_second);
        }
    }
    return bytes;
}

std::shared_ptr<MixedMoeRuntime> make_mixed_moe_runtime(
        const mfq::mfe::Data & cpu,
        bool cuda,
        const MoeLoadOptions& options) {
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->pools.reserve(cpu.pools.size());
    for (const auto & source : cpu.pools) {
        MixedMoePool pool;
        pool.local_experts = (int)source.expert_ids.size();
        std::vector<int32_t> local((size_t)cpu.n_experts, -1);
        for (int index = 0; index < pool.local_experts; ++index) {
            local[(size_t)source.expert_ids[(size_t)index]] = index;
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local.data(), {(int64_t)local.size()},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone();
        if (cuda) {
            const auto target = mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device());
            pool.expert_local =
                pool.expert_local.to(target).contiguous();
        }
        const int expected_rows = pool.local_experts * cpu.out_per_expert;
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = to_device_nint8_zero(source.q8_zero, cuda);
            if (pool.q8_zero.out != expected_rows ||
                pool.q8_zero.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed NINT8-0 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(source.mxfp4, cuda);
            if (pool.mxfp4.out != expected_rows ||
                    pool.mxfp4.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4 cohort shape mismatch");
            }
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                source.payload, cuda);
            if (pool.mxfp4_sq.out != expected_rows ||
                    pool.mxfp4_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed MXFP4-SQ cohort shape mismatch");
            }
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, cuda);
            if (pool.fp8_sq.out != expected_rows ||
                    pool.fp8_sq.neuron_len != cpu.neuron_len) {
                throw std::runtime_error(
                    "mixed FP8-SQ cohort shape mismatch");
            }
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            pool.nint = to_device_mfe_nint(
                source.weight, pool.local_experts,
                cpu.out_per_expert, cuda);
            if (pool.nint.out != expected_rows ||
                pool.nint.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NINT cohort shape mismatch");
            }
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = mfq::nepq::unpack(
                source.payload, source.dtype, source.runtime_payload);
            if (parsed.n_experts != pool.local_experts ||
                parsed.out_per_expert != cpu.out_per_expert ||
                parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NEPQ cohort shape mismatch");
            }
            pool.nepq = to_device_nepq(parsed, cuda);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = mfq::nvq::unpack(source.payload, source.dtype);
            if (parsed.out != expected_rows || parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NVQ/NPQ cohort shape mismatch");
            }
            pool.nvq = cuda
                ? to_device_nvq(
                    parsed, true, options.nvq2,
                    options.nvq_groups)
                : to_device_nvq(parsed, false, false, false);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (cuda) {
        prepare_moe(*runtime, options.heterogeneous);
    }
    return runtime;
}

static mfq::cuda::Experts wrap_mixed_moe_runtime(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    mfq::cuda::Experts result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.partial_experts = runtime->partial_experts;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    result.resident = runtime;
    return result;
}

mfq::cuda::Experts to_gpu_mixed_moe(
        const mfq::mfe::Data & cpu,
        const MoeLoadOptions& options) {
    return wrap_mixed_moe_runtime(
        make_mixed_moe_runtime(cpu, true, options));
}

mfq::cuda::Experts to_cuda_device_moe_expert_slice(
        const mfq::mfe::Data & cpu,
        int64_t expert_begin,
        int64_t expert_end,
        int device,
        const MoeLoadOptions& options) {
    if (expert_begin < 0 || expert_begin >= expert_end ||
            expert_end > cpu.n_experts) {
        throw std::runtime_error(
            "invalid expert-parallel MoE shard");
    }
    const bool all_nint = std::all_of(
        cpu.pools.begin(), cpu.pools.end(),
        [](const mfq::mfe::Pool & pool) {
            return pool.dtype == "NINT";
        });
    MfqCudaGuard guard(device);
    if (all_nint) {
        mfq::mfe::Data sliced = cpu;
        sliced.pools.clear();
        for (const auto & source : cpu.pools) {
            mfq::mfe::Pool destination = source;
            destination.expert_ids.clear();
            std::vector<int64_t> rows;
            for (size_t local = 0;
                 local < source.expert_ids.size(); ++local) {
                const int expert = source.expert_ids[local];
                if (expert < expert_begin || expert >= expert_end) continue;
                destination.expert_ids.push_back(expert);
                for (int row = 0; row < cpu.out_per_expert; ++row) {
                    rows.push_back(
                        static_cast<int64_t>(local) * cpu.out_per_expert + row);
                }
            }
            if (rows.empty()) continue;
            destination.weight = select_nint_cpu_rows(
                source.weight, rows);
            sliced.pools.push_back(std::move(destination));
        }
        auto result = to_gpu_mfe(sliced);
        result.partial_experts = true;
        return result;
    }

    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu.n_experts;
    runtime->out_per_expert = cpu.out_per_expert;
    runtime->neuron_len = cpu.neuron_len;
    runtime->partial_experts = true;
    for (const auto & source : cpu.pools) {
        std::vector<int> expert_ids;
        std::vector<int64_t> rows;
        for (size_t local = 0;
             local < source.expert_ids.size(); ++local) {
            const int expert = source.expert_ids[local];
            if (expert < expert_begin || expert >= expert_end) continue;
            expert_ids.push_back(expert);
            for (int row = 0; row < cpu.out_per_expert; ++row) {
                rows.push_back(
                    static_cast<int64_t>(local) * cpu.out_per_expert + row);
            }
        }
        if (rows.empty()) continue;

        MixedMoePool pool;
        pool.local_experts = static_cast<int>(expert_ids.size());
        std::vector<int32_t> local_map(
            static_cast<size_t>(cpu.n_experts), -1);
        for (size_t local = 0; local < expert_ids.size(); ++local) {
            local_map[static_cast<size_t>(expert_ids[local])] =
                static_cast<int32_t>(local);
        }
        pool.expert_local = mfq_tensor_backend::from_blob(
            local_map.data(),
            {static_cast<int64_t>(local_map.size())},
            mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::Device(
                mfq_tensor_backend::kCUDA,
                mfq_current_cuda_device())).contiguous();
        if (source.dtype == "NINT8-0") {
            pool.family = MixedMoeFamily::Nint8Zero;
            pool.q8_zero = to_device_nint8_zero(
                select_nint8_zero_cpu_rows(source.q8_zero, rows), true);
        } else if (source.dtype == "NINT") {
            pool.family = MixedMoeFamily::Nint;
            const auto selected = select_nint_cpu_rows(source.weight, rows);
            pool.nint = to_device_mfe_nint(
                selected, pool.local_experts,
                cpu.out_per_expert, true);
        } else if (source.dtype == "MXFP4") {
            pool.family = MixedMoeFamily::Mxfp4;
            pool.mxfp4 = to_device_mxfp4(
                select_mxfp4_cpu_rows(source.mxfp4, rows), true);
        } else if (source.dtype == "MXFP4-SQ") {
            pool.family = MixedMoeFamily::Mxfp4Sq;
            pool.mxfp4_sq = to_device_mxfp4_sq(
                mfq::sq::select_rows(source.payload, rows), true);
        } else if (mfq::fp8sq::is_dtype(source.dtype)) {
            if (rows.size() != source.expert_ids.size() *
                    static_cast<size_t>(cpu.out_per_expert)) {
                throw std::runtime_error(
                    "FP8-SQ MFE pools must remain matrix-local across devices");
            }
            pool.family = MixedMoeFamily::Fp8Sq;
            pool.fp8_sq = to_device_fp8_sq(
                source.dtype, source.payload, true);
        } else if (source.dtype == "NEPQ") {
            pool.family = MixedMoeFamily::Nepq;
            auto parsed = mfq::nepq::unpack(
                source.payload, source.dtype, source.runtime_payload);
            pool.nepq = to_device_nepq(select_nepq_cpu_rows(
                parsed, rows, cpu.out_per_expert,
                pool.local_experts), true);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = mfq::nvq::unpack(source.payload, source.dtype);
            pool.nvq = to_device_nvq(
                select_nvq_cpu_rows(parsed, rows), true,
                options.nvq2, options.nvq_groups);
        }
        runtime->pools.push_back(std::move(pool));
    }
    if (runtime->pools.empty()) {
        throw std::runtime_error(
            "expert-parallel MoE shard has no owned experts");
    }
    prepare_moe(*runtime, options.heterogeneous);
    return wrap_mixed_moe_runtime(runtime);
}



static NintWeight copy_cpu_nint_to_cuda(const NintWeight & source) {
    NintWeight result = source;
    result.workspaces.clear();
    result.q_packed = copy_cpu_weight_to_cuda(source.q_packed);
    result.row_q_bits = copy_cpu_weight_to_cuda(source.row_q_bits);
    result.row_q_bit_offsets =
        copy_cpu_weight_to_cuda(source.row_q_bit_offsets);
    result.q8_zero_scale =
        copy_cpu_weight_to_cuda(source.q8_zero_scale);
    result.sub_scale = copy_cpu_weight_to_cuda(source.sub_scale);
    result.sub_min = copy_cpu_weight_to_cuda(source.sub_min);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.neuron_min = copy_cpu_weight_to_cuda(source.neuron_min);
    return result;
}

static NvqWeight copy_cpu_nvq_to_cuda(const NvqWeight & source) {
    NvqWeight result = source;
    result.workspaces.clear();
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.sub_scale_packed =
        copy_cpu_weight_to_cuda(source.sub_scale_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.codebook = copy_cpu_weight_to_cuda(source.codebook);
    result.integer_codebook = copy_cpu_weight_to_cuda(source.integer_codebook);
    result.decode_records = source.decode_records.defined() && source.decode_records.data_ptr()==source.indices_packed.data_ptr()
        ? result.indices_packed : copy_cpu_weight_to_cuda(source.decode_records);
    return result;
}

static Mxfp4Weight copy_cpu_mxfp4_to_cuda(const Mxfp4Weight & source) {
    Mxfp4Weight result = source;
    result.values = copy_cpu_weight_to_cuda(source.values);
    result.scales = copy_cpu_weight_to_cuda(source.scales);
    return result;
}

static Mxfp4SqWeight copy_cpu_mxfp4_sq_to_cuda(
        const Mxfp4SqWeight & source) {
    Mxfp4SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    result.row_auxiliary =
        copy_cpu_weight_to_cuda(source.row_auxiliary);
    return result;
}

static Fp8SqWeight copy_cpu_fp8_sq_to_cuda(
        const Fp8SqWeight & source) {
    Fp8SqWeight result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    return result;
}

static NepqWeight copy_cpu_nepq_to_cuda(const NepqWeight & source) {
    NepqWeight result = source;
    result.indices_packed =
        copy_cpu_weight_to_cuda(source.indices_packed);
    result.aux_packed = copy_cpu_weight_to_cuda(source.aux_packed);
    result.state_packed =
        copy_cpu_weight_to_cuda(source.state_packed);
    result.neuron_scale =
        copy_cpu_weight_to_cuda(source.neuron_scale);
    result.table_pool = copy_cpu_weight_to_cuda(source.table_pool);
    result.grouped_table_pool =
        copy_cpu_weight_to_cuda(source.grouped_table_pool);
    result.bank_ids = copy_cpu_weight_to_cuda(source.bank_ids);
    result.rotation_signs =
        copy_cpu_weight_to_cuda(source.rotation_signs);
    if (source.residual) {
        result.residual_codebook =
            copy_cpu_weight_to_cuda(source.residual_codebook);
        result.residual_first =
            copy_cpu_weight_to_cuda(source.residual_first);
        result.residual_second =
            copy_cpu_weight_to_cuda(source.residual_second);
    }
    return result;
}

mfq::cuda::Experts stage_cpu_mixed_moe(
        const std::shared_ptr<MixedMoeRuntime> & cpu,
        bool heterogeneous) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    auto runtime = std::make_shared<MixedMoeRuntime>();
    runtime->n_experts = cpu->n_experts;
    runtime->out_per_expert = cpu->out_per_expert;
    runtime->neuron_len = cpu->neuron_len;
    runtime->pools.reserve(cpu->pools.size());
    for (const auto & source : cpu->pools) {
        MixedMoePool pool;
        pool.family = source.family;
        pool.local_experts = source.local_experts;
        pool.expert_local =
            copy_cpu_weight_to_cuda(source.expert_local);
        if (pool.family == MixedMoeFamily::Nint) {
            pool.nint = copy_cpu_nint_to_cuda(source.nint);
        } else if (pool.family == MixedMoeFamily::Nint8Zero) {
            pool.q8_zero = copy_cpu_nint_to_cuda(source.q8_zero);
        } else if (pool.family == MixedMoeFamily::Mxfp4) {
            pool.mxfp4 = copy_cpu_mxfp4_to_cuda(source.mxfp4);
        } else if (pool.family == MixedMoeFamily::Mxfp4Sq) {
            pool.mxfp4_sq = copy_cpu_mxfp4_sq_to_cuda(
                source.mxfp4_sq);
        } else if (pool.family == MixedMoeFamily::Fp8Sq) {
            pool.fp8_sq = copy_cpu_fp8_sq_to_cuda(source.fp8_sq);
        } else if (pool.family == MixedMoeFamily::Nvq) {
            pool.nvq = copy_cpu_nvq_to_cuda(source.nvq);
        } else {
            pool.nepq = copy_cpu_nepq_to_cuda(source.nepq);
        }
        runtime->pools.push_back(std::move(pool));
    }
    prepare_moe(*runtime, heterogeneous);
    return wrap_mixed_moe_runtime(runtime);
}

mfq::cuda::Experts cpu_mixed_moe_metadata(
        const std::shared_ptr<MixedMoeRuntime> & runtime) {
    mfq::cuda::Experts result;
    result.unified_nint_projection = runtime->nint_only();
    result.n_experts = runtime->n_experts;
    result.out_per_expert = runtime->out_per_expert;
    result.neuron_len = runtime->neuron_len;
    result.mixed_weight_bytes = mixed_moe_storage_bytes(*runtime);
    result.activation_geometries = runtime->activation_geometry();
    result.activation_workspace_domain = 2;
    return result;
}

mfq::mfe::Data load_mfe_cpu(
        const mfq::ModelSource & mfq, const std::string & name) {
    const auto& dtype = require_tensor(mfq, name).dtype;
    if (dtype == "NINT") {
        mfq::mfe::Pool pool;
        pool.dtype = "NINT";
        pool.payload = read_tensor(mfq, name);
        pool.weight = mfq::nint::unpack(pool.payload);
        const auto& shape = pool.weight.shape;
        TYLOINFER_RUNTIME_CHECK(pool.weight.axis == 0 && shape.size() == 3 &&
            shape[0] > 0 && shape[0] <= std::numeric_limits<int>::max() &&
            shape[1] > 0 && shape[1] <= std::numeric_limits<int>::max() &&
            shape[2] == pool.weight.neuron_len && shape[0] * shape[1] == pool.weight.out,
            "routed NINT tensor must use [experts,out,in]: ", name);
        mfq::mfe::Data result;
        result.n_experts = static_cast<int>(shape[0]);
        result.out_per_expert = static_cast<int>(shape[1]);
        result.neuron_len = pool.weight.neuron_len;
        pool.expert_ids.resize(result.n_experts);
        std::iota(pool.expert_ids.begin(), pool.expert_ids.end(), 0);
        pool.weight.shape = {pool.weight.out, pool.weight.neuron_len};
        result.pools.push_back(std::move(pool));
        return result;
    }
    if (dtype != "MFE") {
        throw std::runtime_error("expert tensor must use MFE: " + name);
    }
    return mfq::mfe::unpack(read_tensor(mfq, name));
}

mfq_tensor_backend::Tensor mfe_dense_reference(
        const mfq::ModelSource& source,
        const std::string& name,
        mfq_tensor_backend::Tensor input,
        const std::vector<std::int32_t>& expert_ids,
        int tokens,
        int routes,
        bool routed_input) {
    auto cpu = mfq::mfe::unpack(read_tensor(source, name));
    auto reference = mfq_tensor_backend::empty(
        {tokens * routes, cpu.out_per_expert},
        input.options().dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor dense_flat;
        int rotation_block = 0;
        mfq_tensor_backend::Tensor rotation_signs;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_device_nint8_zero(pool.q8_zero, true);
            dense_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_device_nint(pool.weight, true);
            dense_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            dense_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else if (pool.dtype == "NEPQ") {
            auto packed = to_device_nepq(mfq::nepq::unpack(
                pool.payload, pool.dtype, pool.runtime_payload), true);
            dense_flat = nepq_dequant_cuda(
                packed.indices_packed, packed.aux_packed,
                packed.state_packed, packed.neuron_scale,
                packed.table_pool, packed.bank_ids,
                packed.neuron_len, packed.state_bits,
                packed.format);
            if (packed.residual) {
                dense_flat = nepq_sparse_residual_dequant_cuda(
                    packed.residual_codebook,
                    packed.residual_first,
                    packed.residual_second,
                    packed.residual_position_bits,
                    packed.residual_block_vectors,
                    dense_flat.reshape({
                        packed.n_experts * packed.out_per_expert,
                        packed.neuron_len}));
            }
            rotation_block = packed.rotation_block;
            rotation_signs = packed.rotation_signs;
        } else {
            dense_flat = nvq_dequant(to_device_nvq(
                mfq::nvq::unpack(pool.payload, pool.dtype), true));
        }
        auto dense = dense_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        for (size_t local = 0; local < pool.expert_ids.size(); ++local) {
            const int expert = pool.expert_ids[local];
            std::vector<int64_t> pair_indices;
            std::vector<int64_t> token_indices;
            for (int pair = 0; pair < tokens * routes; ++pair) {
                if (expert_ids[static_cast<size_t>(pair)] == expert) {
                    pair_indices.push_back(pair);
                    token_indices.push_back(pair / routes);
                }
            }
            if (pair_indices.empty()) continue;
            auto pair_index = mfq_tensor_backend::from_blob(
                pair_indices.data(),
                {static_cast<int64_t>(pair_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto token_index = mfq_tensor_backend::from_blob(
                token_indices.data(),
                {static_cast<int64_t>(token_indices.size())},
                mfq_tensor_backend::TensorOptions()
                    .dtype(mfq_tensor_backend::kInt64))
                .clone().to(mfq_tensor_backend::kCUDA);
            auto selected = routed_input
                ? input.reshape({tokens * routes, cpu.neuron_len})
                      .index_select(0, pair_index)
                : input.index_select(0, token_index);
            if (rotation_block != 0) {
                selected = nepq_hadamard_input_cuda(
                    selected.contiguous(), rotation_signs,
                    rotation_block);
            }
            auto expected = mfq_tensor_backend::matmul(
                selected,
                dense.index({static_cast<int64_t>(local)})
                    .transpose(0, 1));
            reference.index_copy_(0, pair_index, expected);
        }
    }
    return reference;
}

mfq_tensor_backend::Tensor materialize_mfe_dense(
        const mfq::ModelSource& source,
        const std::string& name) {
    auto cpu = mfq::mfe::unpack(read_tensor(source, name));
    auto dense = mfq_tensor_backend::empty(
        {cpu.n_experts, cpu.out_per_expert, cpu.neuron_len},
        mfq_tensor_backend::TensorOptions()
            .device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kFloat16));
    for (const auto& pool : cpu.pools) {
        mfq_tensor_backend::Tensor local_flat;
        if (pool.dtype == "NINT8-0") {
            auto packed = to_device_nint8_zero(pool.q8_zero, true);
            local_flat = nint8_zero_dequant_cuda(
                packed.q_packed, packed.q8_zero_scale,
                packed.neuron_len);
        } else if (pool.dtype == "NINT") {
            auto packed = to_device_nint(pool.weight, true);
            local_flat = nint_decode_cuda(
                packed.q_packed, packed.row_q_bits,
                packed.row_q_bit_offsets, packed.sub_scale,
                packed.sub_min, packed.neuron_scale,
                packed.neuron_min, packed.neuron_len, packed.gs);
        } else if (pool.dtype == "MXFP4") {
            local_flat = dequant_mxfp4_cpu(pool.mxfp4)
                .to(mfq_tensor_backend::kCUDA).contiguous();
        } else {
            throw std::runtime_error(
                "dense MoE reference requires NINT or MXFP4 cohorts");
        }
        auto local = local_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        auto expert_index = mfq_tensor_backend::from_blob(
            const_cast<int32_t*>(pool.expert_ids.data()),
            {static_cast<int64_t>(pool.expert_ids.size())},
            mfq_tensor_backend::TensorOptions()
                .dtype(mfq_tensor_backend::kInt32))
            .clone().to(mfq_tensor_backend::kCUDA)
            .to(mfq_tensor_backend::kInt64);
        dense.index_copy_(0, expert_index, local);
    }
    return dense;
}
