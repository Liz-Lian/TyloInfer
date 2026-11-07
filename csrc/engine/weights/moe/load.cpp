#include "engine/weights/moe/load.h"
#include "engine/backends/cuda/ops/ffn/moe.h"

#include "engine/weights/moe/prepare.h"
#include "engine/weights/quant.h"
#include "engine/weights/loader.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/moe/cached_source_internal.h"
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>

bool tyloi::MfeTensor::nint_only() const {
    return !pools.empty() && std::all_of(pools.begin(), pools.end(),
        [](const MixedMoePool& pool) { return pool.family == MixedMoeFamily::Nint; });
}

bool tyloi::MfeTensor::defined() const noexcept {
    if (n_experts <= 0 || out_per_expert <= 0 || neuron_len <= 0) return false;
    if (cached_source) return true;
    if (!shards.empty()) {
        for (const auto& shard : shards) if (!shard.weight || !shard.weight->defined()) return false;
        return true;
    }
    if (compact) {
        if (compact->empty()) return false;
        for (const auto& pool : *compact)
            if (!pool.weight.defined() || !pool.expert_local.defined()) return false;
        return true;
    }
    if (pools.empty()) return false;
    for (const auto& pool : pools) {
        const auto* payload = pool.payload();
        if (!(payload ? payload->defined() : pool.nepq.indices_packed.defined()) ||
            !pool.expert_local.defined()) return false;
    }
    return true;
}

std::int64_t tyloi::MfeTensor::numel() const {
    TensorView view;
    view.rank = 3;
    view.sizes[0] = n_experts;
    view.sizes[1] = out_per_expert;
    view.sizes[2] = neuron_len;
    return view.numel();
}

std::size_t tyloi::MfeTensor::nbytes() const {
    const auto elements = numel();
    if (static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("tensor byte count overflow");
    return static_cast<std::size_t>(elements) * 2;
}

std::size_t tyloi::MfeTensor::backing_nbytes() const {
    return backing_bytes(buffers());
}

std::int64_t tyloi::MfeTensor::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += 3;
    switch (dimension) {
        case 0: return n_experts;
        case 1: return out_per_expert;
        case 2: return neuron_len;
        default: throw std::out_of_range("tensor dimension is out of range");
    }
}

const tyloi::Device& tyloi::MfeTensor::device() const noexcept {
    if (cached_source) return source_device;
    if (!shards.empty() && shards.front().weight) return shards.front().weight->device();
    if (compact && !compact->empty()) return compact->front().expert_local.device();
    for (const auto& pool : pools) {
        const auto* payload = pool.payload();
        if (payload && payload->defined()) return payload->device();
        if (!payload && pool.nepq.indices_packed.defined()) return pool.nepq.indices_packed.device();
    }
    static const Device unprepared{};
    return unprepared;
}

std::vector<const tyloi::DenseTensor*> tyloi::MfeTensor::buffers() const {
    std::vector<const DenseTensor*> result;
    if (compact) for (const auto& pool : *compact) {
        auto values = pool.weight.buffers();
        result.insert(result.end(), values.begin(), values.end());
        result.push_back(&pool.expert_local);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (cached_source) {
        auto values = cached_source->source_tensor().buffers();
        result.insert(result.end(), values.begin(), values.end());
    }
#endif
    for (const auto& shard : shards) if (shard.weight) {
        auto values = shard.weight->buffers();
        result.insert(result.end(), values.begin(), values.end());
    }
    for (const auto& pool : pools) {
        if (const auto* payload = pool.payload()) {
            auto values = payload->buffers();
            result.insert(result.end(), values.begin(), values.end());
        } else {
            auto values = nepq_buffers(pool.nepq);
            result.insert(result.end(), values.begin(), values.end());
        }
        result.push_back(&pool.expert_local);
    }
    if (nvq_dispatch) {
        const auto& d = *nvq_dispatch;
        result.insert(result.end(), {&d.metadata_device, &d.metadata_host, &d.map_ptrs,
            &d.weight_ptrs, &d.weight_sizes, &d.pool_params,
            &d.initial_expert_pool, &d.initial_expert_local});
    }
    if (nint_dispatch) {
        const auto& d = *nint_dispatch;
        result.insert(result.end(), {&d.metadata_device, &d.metadata_host, &d.map_ptrs,
            &d.pool_params, &d.initial_expert_pool, &d.initial_expert_local});
    }
    return result;
}

void tyloi::MfeTensor::record_stream(std::uintptr_t stream) const {
    int cuda_device = -1;
    for (const auto& shard : shards) {
        if (cuda_device >= 0 && cuda_device != shard.device)
            throw std::invalid_argument("MFE stream recording across CUDA devices is unsupported");
        cuda_device = shard.device;
    }
    for (const auto* buffer : buffers()) if (buffer && buffer->defined() && buffer->is_cuda()) {
        if (cuda_device >= 0 && cuda_device != buffer->get_device())
            throw std::invalid_argument("MFE stream recording across CUDA devices is unsupported");
        cuda_device = buffer->get_device();
    }
    for (const auto* buffer : buffers())
        if (buffer && buffer->defined()) buffer->record_stream(stream);
}

std::shared_ptr<const tyloi::Tensor> tyloi::MfeTensor::relocated(Device target) const {
    if (!defined())
        throw std::invalid_argument("MFE relocation before preparation is unsupported");
    auto moved = std::make_shared<MfeTensor>(*this);
    if (device() == target) return moved;
    if (nvq_dispatch || nint_dispatch || cached_source || !shards.empty())
        throw std::invalid_argument("MFE relocation with dispatch, cached source, or shards is unsupported");
    if (compact) moved->compact = std::make_shared<std::vector<MfePoolWeight>>(*compact);
    for (const auto* buffer : moved->buffers()) {
        auto& writable = const_cast<DenseTensor&>(*buffer);
        if (writable.defined() && writable.device() != target) writable = writable.to(target);
    }
    return moved;
}

static void validate_mfe_nint_pools(const std::vector<MfePoolWeight>& pools) {
    if (pools.empty()) {
        throw std::runtime_error("MFE NINT runtime requires at least one pool");
    }
    const auto target = pools.front().weight.q_packed.device();
    for (const auto & pool : pools) {
        TYLOINFER_RUNTIME_CHECK(
            pool.weight.q_packed.is_cuda() &&
            pool.weight.q_packed.device() == target,
            "MFE NINT pools must share one CUDA device");
    }
}

std::shared_ptr<const tyloi::MfeTensor> to_gpu_mfe(const tyloi::mfe::Data & cpu) {
    auto result = std::make_shared<tyloi::MfeTensor>();
    result->n_experts = cpu.n_experts;
    result->out_per_expert = cpu.out_per_expert;
    result->neuron_len = cpu.neuron_len;
    std::vector<MfePoolWeight> pools;
    pools.reserve(cpu.pools.size());
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
        pool.expert_local = ::tyloi::cuda::from_blob(
            local.data(), {(int64_t)local.size()}, ::tyloi::TensorOptions().dtype(::tyloi::kInt32))
            .clone().to(pool.weight.q_packed.device()).contiguous();
        pools.push_back(std::move(pool));
    }
    validate_mfe_nint_pools(pools);
    result->compact = std::make_shared<const std::vector<MfePoolWeight>>(std::move(pools));
    return result;
}

::tyloi::DenseTensor copy_cpu_weight_to_cuda(
        const ::tyloi::DenseTensor & source) {
    return source.defined()
        ? source.to(::tyloi::Device(
            ::tyloi::kCUDA,
            mfq_current_cuda_device())).contiguous()
        : ::tyloi::DenseTensor();
}

static int64_t tensor_storage_bytes(const ::tyloi::DenseTensor & value) {
    return value.defined()
        ? value.numel() * (int64_t)value.element_size()
        : 0;
}

// Physical bytes of every backing an expert runtime keeps alive. Aliased
// buffers are counted once, the same rule the projection weights use.
int64_t mixed_moe_storage_bytes(const ::tyloi::MfeTensor & runtime) {
    std::vector<const ::tyloi::DenseTensor*> buffers;
    if (runtime.nint_dispatch) {
        const auto& d = *runtime.nint_dispatch;
        buffers.insert(buffers.end(), {&d.map_ptrs, &d.pool_params,
            &d.initial_expert_pool, &d.initial_expert_local});
    }
    if (runtime.nvq_dispatch) {
        const auto& d = *runtime.nvq_dispatch;
        buffers.insert(buffers.end(), {&d.weight_ptrs, &d.weight_sizes, &d.pool_params,
            &d.initial_expert_pool, &d.initial_expert_local});
    }
    for (const auto & pool : runtime.pools)
        pool.visit_tensors([&](const ::tyloi::DenseTensor& tensor) { buffers.push_back(&tensor); });
    return static_cast<int64_t>(::tyloi::backing_bytes(buffers));
}

std::shared_ptr<::tyloi::MfeTensor> make_mfe(
        const tyloi::mfe::Data & cpu,
        bool cuda,
        const MoeLoadOptions& options) {
    auto runtime = std::make_shared<::tyloi::MfeTensor>();
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
        pool.expert_local = ::tyloi::cuda::from_blob(
            local.data(), {(int64_t)local.size()},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt32))
            .clone();
        if (cuda) {
            const auto target = ::tyloi::Device(
                ::tyloi::kCUDA,
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
        } else if (tyloi::fp8sq::is_dtype(source.dtype)) {
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
            auto parsed = tyloi::nepq::unpack(
                source.payload, source.dtype, source.runtime_payload);
            if (parsed.n_experts != pool.local_experts ||
                parsed.out_per_expert != cpu.out_per_expert ||
                parsed.neuron_len != cpu.neuron_len) {
                throw std::runtime_error("mixed NEPQ cohort shape mismatch");
            }
            pool.nepq = to_device_nepq(parsed, cuda);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = tyloi::nvq::unpack(source.payload, source.dtype);
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

std::shared_ptr<const tyloi::MfeTensor> to_gpu_mixed_moe(
        const tyloi::mfe::Data & cpu,
        const MoeLoadOptions& options) {
    return make_mfe(cpu, true, options);
}

std::shared_ptr<const tyloi::MfeTensor> to_cuda_device_moe_expert_slice(
        const tyloi::mfe::Data & cpu,
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
        [](const tyloi::mfe::Pool & pool) {
            return pool.dtype == "NINT";
        });
    MfqCudaGuard guard(device);
    if (all_nint) {
        tyloi::mfe::Data sliced = cpu;
        sliced.pools.clear();
        for (const auto & source : cpu.pools) {
            tyloi::mfe::Pool destination = source;
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
        auto result = std::make_shared<tyloi::MfeTensor>(*to_gpu_mfe(sliced));
        result->partial_experts = true;
        return result;
    }

    auto runtime = std::make_shared<::tyloi::MfeTensor>();
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
        pool.expert_local = ::tyloi::cuda::from_blob(
            local_map.data(),
            {static_cast<int64_t>(local_map.size())},
            ::tyloi::TensorOptions().dtype(::tyloi::kInt32))
            .clone().to(::tyloi::Device(
                ::tyloi::kCUDA,
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
                tyloi::sq::select_rows(source.payload, rows), true);
        } else if (tyloi::fp8sq::is_dtype(source.dtype)) {
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
            auto parsed = tyloi::nepq::unpack(
                source.payload, source.dtype, source.runtime_payload);
            pool.nepq = to_device_nepq(select_nepq_cpu_rows(
                parsed, rows, cpu.out_per_expert,
                pool.local_experts), true);
        } else {
            pool.family = MixedMoeFamily::Nvq;
            auto parsed = tyloi::nvq::unpack(source.payload, source.dtype);
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
    return runtime;
}



static tyloi::NintTensor copy_cpu_nint_to_cuda(const tyloi::NintTensor & source) {
    tyloi::NintTensor result = source;
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

static tyloi::NvqTensor copy_cpu_nvq_to_cuda(const tyloi::NvqTensor & source) {
    tyloi::NvqTensor result = source;
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

static tyloi::Mxfp4Tensor copy_cpu_mxfp4_to_cuda(const tyloi::Mxfp4Tensor & source) {
    tyloi::Mxfp4Tensor result = source;
    result.values = copy_cpu_weight_to_cuda(source.values);
    result.scales = copy_cpu_weight_to_cuda(source.scales);
    return result;
}

static tyloi::Mxfp4SqTensor copy_cpu_mxfp4_sq_to_cuda(
        const tyloi::Mxfp4SqTensor & source) {
    tyloi::Mxfp4SqTensor result = source;
    result.blob = copy_cpu_weight_to_cuda(source.blob);
    result.row_q = copy_cpu_weight_to_cuda(source.row_q);
    result.row_symbol_byte_offsets =
        copy_cpu_weight_to_cuda(source.row_symbol_byte_offsets);
    result.row_auxiliary =
        copy_cpu_weight_to_cuda(source.row_auxiliary);
    return result;
}

static tyloi::Fp8SqTensor copy_cpu_fp8_sq_to_cuda(
        const tyloi::Fp8SqTensor & source) {
    tyloi::Fp8SqTensor result = source;
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

std::shared_ptr<const tyloi::MfeTensor> stage_cpu_mixed_moe(
        const std::shared_ptr<const ::tyloi::MfeTensor> & cpu,
        bool heterogeneous) {
    if (!cpu) throw std::runtime_error("missing CPU-offloaded MoE state");
    auto runtime = std::make_shared<::tyloi::MfeTensor>();
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
    return runtime;
}

tyloi::mfe::Data load_mfe_cpu(
        const tyloi::source::Source & mfq, const std::string & name) {
    const auto& dtype = tyloi::require_tensor(mfq, name).dtype;
    if (dtype == "NINT") {
        tyloi::mfe::Pool pool;
        pool.dtype = "NINT";
        pool.payload = tyloi::read_tensor(mfq, name);
        pool.weight = tyloi::nint::unpack(pool.payload);
        const auto& shape = pool.weight.shape;
        TYLOINFER_RUNTIME_CHECK(pool.weight.axis == 0 && shape.size() == 3 &&
            shape[0] > 0 && shape[0] <= std::numeric_limits<int>::max() &&
            shape[1] > 0 && shape[1] <= std::numeric_limits<int>::max() &&
            shape[2] == pool.weight.neuron_len && shape[0] * shape[1] == pool.weight.out,
            "routed NINT tensor must use [experts,out,in]: ", name);
        tyloi::mfe::Data result;
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
    return tyloi::mfe::unpack(tyloi::read_tensor(mfq, name));
}

::tyloi::DenseTensor mfe_dense_reference(
        const tyloi::source::Source& source,
        const std::string& name,
        ::tyloi::DenseTensor input,
        const std::vector<std::int32_t>& expert_ids,
        int tokens,
        int routes,
        bool routed_input) {
    auto cpu = tyloi::mfe::unpack(tyloi::read_tensor(source, name));
    auto reference = ::tyloi::cuda::empty(
        {tokens * routes, cpu.out_per_expert},
        input.options().dtype(::tyloi::kFloat16));
    for (const auto& pool : cpu.pools) {
        ::tyloi::DenseTensor dense_flat;
        int rotation_block = 0;
        ::tyloi::DenseTensor rotation_signs;
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
                .to(::tyloi::kCUDA).contiguous();
        } else if (pool.dtype == "NEPQ") {
            auto packed = to_device_nepq(tyloi::nepq::unpack(
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
                tyloi::nvq::unpack(pool.payload, pool.dtype), true));
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
            auto pair_index = ::tyloi::cuda::from_blob(
                pair_indices.data(),
                {static_cast<int64_t>(pair_indices.size())},
                ::tyloi::TensorOptions()
                    .dtype(::tyloi::kInt64))
                .clone().to(::tyloi::kCUDA);
            auto token_index = ::tyloi::cuda::from_blob(
                token_indices.data(),
                {static_cast<int64_t>(token_indices.size())},
                ::tyloi::TensorOptions()
                    .dtype(::tyloi::kInt64))
                .clone().to(::tyloi::kCUDA);
            auto selected = routed_input
                ? input.reshape({tokens * routes, cpu.neuron_len})
                      .index_select(0, pair_index)
                : input.index_select(0, token_index);
            if (rotation_block != 0) {
                selected = nepq_hadamard_input_cuda(
                    selected.contiguous(), rotation_signs,
                    rotation_block);
            }
            auto expected = ::tyloi::cuda::matmul(
                selected,
                dense.index({static_cast<int64_t>(local)})
                    .transpose(0, 1));
            reference.index_copy_(0, pair_index, expected);
        }
    }
    return reference;
}

::tyloi::DenseTensor materialize_mfe_dense(
        const tyloi::source::Source& source,
        const std::string& name) {
    auto cpu = tyloi::mfe::unpack(tyloi::read_tensor(source, name));
    auto dense = ::tyloi::cuda::empty(
        {cpu.n_experts, cpu.out_per_expert, cpu.neuron_len},
        ::tyloi::TensorOptions()
            .device(::tyloi::kCUDA)
            .dtype(::tyloi::kFloat16));
    for (const auto& pool : cpu.pools) {
        ::tyloi::DenseTensor local_flat;
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
                .to(::tyloi::kCUDA).contiguous();
        } else {
            throw std::runtime_error(
                "dense MoE reference requires NINT or MXFP4 cohorts");
        }
        auto local = local_flat.reshape({
            static_cast<int64_t>(pool.expert_ids.size()),
            cpu.out_per_expert, cpu.neuron_len});
        auto expert_index = ::tyloi::cuda::from_blob(
            const_cast<int32_t*>(pool.expert_ids.data()),
            {static_cast<int64_t>(pool.expert_ids.size())},
            ::tyloi::TensorOptions()
                .dtype(::tyloi::kInt32))
            .clone().to(::tyloi::kCUDA)
            .to(::tyloi::kInt64);
        dense.index_copy_(0, expert_index, local);
    }
    return dense;
}
