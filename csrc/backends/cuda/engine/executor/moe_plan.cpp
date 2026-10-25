#include "backends/cuda/engine/executor/moe_plan.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_quant_ops.h"

#include <algorithm>
#include <climits>
#include <limits>
#include <cstring>

static void upload_mixed_dispatch_metadata(
        std::initializer_list<std::pair<mfq_tensor_backend::Tensor*,mfq_tensor_backend::Tensor>> tables,
        mfq_tensor_backend::Device device,bool asynchronous,
        mfq_tensor_backend::Tensor& device_owner,mfq_tensor_backend::Tensor& host_owner) {
    namespace tb=mfq_tensor_backend;
    std::size_t bytes=0;
    for(const auto& [target,source]:tables) {
        bytes=(bytes+15)&~std::size_t(15);
        bytes+=source.nbytes();
    }
    // Small metadata packets are rebuilt while previous layers are running.
    // cudaHostAlloc here interrupts that overlap. Keep pageable storage alive
    // through its upload; cross-stream publication still synchronizes below.
    host_owner=tb::empty({int64_t(bytes)},tb::TensorOptions().dtype(tb::kUInt8));
    device_owner=tb::empty({int64_t(bytes)},host_owner.options().device(device).pinned_memory(false));
    std::memset(host_owner.data_ptr(),0,bytes);
    std::size_t offset=0;
    for(const auto& [target,source]:tables) {
        offset=(offset+15)&~std::size_t(15);
        std::memcpy(host_owner.data_ptr<uint8_t>()+offset,source.data_ptr(),source.nbytes());
        *target=tb::from_blob(device_owner.data_ptr<uint8_t>()+offset,source.sizes(),source.options().device(device));
        offset+=source.nbytes();
    }
    const auto stream=mfq_current_cuda_stream();
    TYLOINFER_CUDA_CHECK(cudaMemcpyAsync(device_owner.data_ptr(),host_owner.data_ptr(),bytes,cudaMemcpyHostToDevice,stream));
    // Prefill submits its consumers on this stream and retains both packets.
    // Other callers keep the existing cross-stream initialization guarantee.
    if(!asynchronous)TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
}

static void initialize_mixed_nint_dispatch(MixedMoeRuntime& runtime) {
    namespace tb=mfq_tensor_backend;
    runtime.nint_dispatch.reset();runtime.nint_input_plans.clear();
    if(runtime.n_experts<=0 || runtime.n_experts>4096 || runtime.neuron_len<=0 ||
        runtime.out_per_expert<=0 || runtime.out_per_expert>INT_MAX)return;
    auto dispatch=std::make_shared<MixedNintDispatch>();
    std::vector<int64_t> maps;std::vector<int32_t> params;
    tb::Device device(tb::kCUDA,mfq_current_cuda_device());
    for(std::size_t i=0;i<runtime.pools.size();++i) {
        const auto& pool=runtime.pools[i];if(pool.family!=MixedMoeFamily::Nint)continue;
        const auto& w=pool.nint;
        if(!w.q_packed.is_cuda() || pool.local_experts<=0 || w.q8_zero ||
            w.out!=int64_t(pool.local_experts)*runtime.out_per_expert || w.neuron_len!=runtime.neuron_len ||
            w.ng<=0 || w.ng>INT_MAX || w.gs<=0 || w.gs>128 || w.ng*w.gs>INT_MAX ||
            w.ng!=(runtime.neuron_len+w.gs-1)/w.gs ||
            w.q_packed.numel()%pool.local_experts || w.q_packed.numel()/pool.local_experts>INT_MAX)return;
        if(dispatch->pools.empty())device=w.q_packed.device();
        const auto field_ok=[&](const tb::Tensor& field,auto dtype,int64_t count) {
            return field.defined() && field.is_cuda() && field.device()==device && field.is_contiguous() &&
                field.scalar_type()==dtype && field.numel()>=count;
        };
        if(!field_ok(w.q_packed,tb::kUInt8,1) || !field_ok(pool.expert_local,tb::kInt32,runtime.n_experts) ||
            !field_ok(w.row_q_bits,tb::kUInt8,w.out) || !field_ok(w.row_q_bit_offsets,tb::kInt64,w.out) ||
            !field_ok(w.sub_scale,tb::kUInt8,w.out*w.ng) || !field_ok(w.sub_min,tb::kUInt8,w.out*w.ng) ||
            !field_ok(w.neuron_scale,tb::kFloat32,w.out) || !field_ok(w.neuron_min,tb::kFloat32,w.out))return;
        dispatch->pools.push_back(int(i));maps.push_back(reinterpret_cast<int64_t>(pool.expert_local.data_ptr<int32_t>()));
        params.insert(params.end(),{int32_t(w.ng),int32_t(w.gs),int32_t(w.q_packed.numel()/pool.local_experts)});
    }
    if(dispatch->pools.empty())return;
    std::vector<int32_t> missing(runtime.n_experts,-1);
    const auto i32=tb::TensorOptions().dtype(tb::kInt32),i64=tb::TensorOptions().dtype(tb::kInt64);
    upload_mixed_dispatch_metadata({
        {&dispatch->map_ptrs,tb::from_blob(maps.data(),{int64_t(maps.size())},i64)},
        {&dispatch->pool_params,tb::from_blob(params.data(),{int64_t(dispatch->pools.size()),3},i32)},
        {&dispatch->expert_pool,tb::from_blob(missing.data(),{runtime.n_experts},i32)},
        {&dispatch->expert_local,tb::from_blob(missing.data(),{runtime.n_experts},i32)}},
        device,runtime.prefill_only,dispatch->metadata_device,dispatch->metadata_host);
    runtime.nint_dispatch=std::move(dispatch);
}

void prepare_moe(
        MixedMoeRuntime & runtime,
        const CudaExecutionConfig& config,bool dense_prefill,
        const std::vector<std::vector<int32_t>>* expert_local_host) {
    if(expert_local_host && expert_local_host->size()!=runtime.pools.size())
        throw std::invalid_argument("mixed NVQ host-map pool count mismatch");
    std::unordered_map<const void*,mfq_tensor_backend::Tensor> integer_books;
    for(auto& pool:runtime.pools)if(pool.family==MixedMoeFamily::Nvq) {
        auto& w=pool.nvq;
        prepare_nvq_decode_records(w);
        const auto* key=w.codebook.data_ptr();
        const auto found=integer_books.find(key);
        if(found!=integer_books.end())w.integer_codebook=found->second;
        else {
            prepare_nvq_integer_codebook(w);
            if(w.integer_codebook.defined())integer_books.emplace(key,w.integer_codebook);
        }
    }
    initialize_mixed_nint_dispatch(runtime);
    runtime.nvq_dispatch.reset();
    if (!config.moe_nvq_heterogeneous) return;
    bool has_dense_groups=false;
    for(const auto& pool:runtime.pools)
        if(pool.family==MixedMoeFamily::Nvq && pool.nvq.dense_groups)has_dense_groups=true;
    // Prefill's FP16 weight reader handles compact records directly. Preserve
    // the existing decode dispatch until its separate readers are validated.
    if(has_dense_groups && !dense_prefill)return;

    int nvq_pools = 0;
    for (const auto & pool : runtime.pools) {
        if (pool.family == MixedMoeFamily::Nvq) ++nvq_pools;
    }
    if (nvq_pools == 0) return;

    std::vector<int64_t> weight_ptrs;
    std::vector<int64_t> map_ptrs;
    std::vector<int64_t> weight_sizes;
    std::vector<int32_t> pool_params;
    std::vector<int32_t> expert_pool(
        static_cast<size_t>(runtime.n_experts), -1);
    std::vector<int32_t> expert_local(
        static_cast<size_t>(runtime.n_experts), -1);
    weight_ptrs.reserve(static_cast<size_t>(nvq_pools) * 5);
    weight_sizes.reserve(static_cast<size_t>(nvq_pools) * 3);
    pool_params.reserve(static_cast<size_t>(nvq_pools) * 7);

    mfq_tensor_backend::Device target = mfq_tensor_backend::Device(
        mfq_tensor_backend::kCUDA, mfq_current_cuda_device());
    int dispatch_pool = 0;
    int owned_experts = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool first_nvq_format = true;
    for (std::size_t pool_index=0;pool_index<runtime.pools.size();++pool_index) {
        const auto& pool=runtime.pools[pool_index];
        if (pool.family != MixedMoeFamily::Nvq) continue;
        const auto & weight = pool.nvq;
        map_ptrs.push_back(static_cast<int64_t>(reinterpret_cast<uintptr_t>(pool.expert_local.data_ptr<int32_t>())));
        if (weight.gs != 24 || weight.ng <= 0 ||
                weight.ng > std::numeric_limits<int32_t>::max() ||
                weight.sub_bits < 1 || weight.sub_bits > 8 ||
                weight.kernel_format < 1 || weight.kernel_format > 17 ||
                weight.sign_mode < 0 || weight.sign_mode > 1) {
            return;
        }
        if (dispatch_pool == 0) target = weight.indices_packed.device();
        if (weight.indices_packed.device() != target ||
                weight.aux_packed.device() != target ||
                weight.sub_scale_packed.device() != target ||
                weight.neuron_scale.device() != target ||
                weight.codebook.device() != target) {
            return;
        }
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.indices_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.aux_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.sub_scale_packed.data_ptr<uint8_t>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.neuron_scale.data_ptr<float>())));
        weight_ptrs.push_back(static_cast<int64_t>(
            reinterpret_cast<uintptr_t>(
                weight.codebook.data_ptr<int8_t>())));
        weight_sizes.push_back(weight.indices_packed.numel());
        weight_sizes.push_back(weight.aux_packed.numel());
        weight_sizes.push_back(weight.sub_scale_packed.numel());
        const int format = static_cast<int>(weight.kernel_format);
        const auto pool_format_group = mixed_nvq_f16_format_group(format);
        if (first_nvq_format) {
            f16_format_group = pool_format_group;
            first_nvq_format = false;
        } else if (f16_format_group != pool_format_group) {
            f16_format_group = MixedNvqF16FormatGroup::All;
        }
        const bool d4 =
            format == 3 || format == 10 || format == 11 ||
            format == 12 || format == 15 || format == 17;
        const int nvec =
            (runtime.neuron_len + (d4 ? 3 : 7)) / (d4 ? 4 : 8);
        const int nsign = (runtime.neuron_len + 7) / 8;
        pool_params.push_back(pool.local_experts);
        pool_params.push_back(static_cast<int32_t>(weight.ng));
        pool_params.push_back(nvec);
        pool_params.push_back(nsign);
        pool_params.push_back(static_cast<int32_t>(weight.sub_bits));
        pool_params.push_back(static_cast<int32_t>(weight.sign_mode));
        pool_params.push_back(format);

        mfq_tensor_backend::Tensor local_host;
        const int32_t* local=nullptr;
        if(expert_local_host) {
            const auto& host=expert_local_host->at(pool_index);
            if(host.size()!=std::size_t(runtime.n_experts))
                throw std::invalid_argument("mixed NVQ host-map expert count mismatch");
            local=host.data();
        } else {
            local_host=pool.expert_local.to(mfq_tensor_backend::kCPU,mfq_tensor_backend::kInt32).contiguous();
            local=local_host.data_ptr<int32_t>();
        }
        for (int expert = 0; expert < runtime.n_experts; ++expert) {
            if (local[expert] < 0) continue;
            if (local[expert] >= pool.local_experts ||
                    expert_pool[static_cast<size_t>(expert)] >= 0) {
                throw std::runtime_error(
                    "mixed NVQ prefill has invalid expert ownership");
            }
            expert_pool[static_cast<size_t>(expert)] = dispatch_pool;
            expert_local[static_cast<size_t>(expert)] = local[expert];
            ++owned_experts;
        }
        ++dispatch_pool;
    }

    auto dispatch = std::make_shared<MixedNvqDispatch>();
    dispatch->pool_count = dispatch_pool;
    dispatch->f16_format_group = f16_format_group;
    dispatch->masked_experts = owned_experts < runtime.n_experts;
    dispatch->dense_groups = has_dense_groups;
    namespace tb=mfq_tensor_backend;
    const auto i32=tb::TensorOptions().dtype(tb::kInt32),i64=tb::TensorOptions().dtype(tb::kInt64);
    upload_mixed_dispatch_metadata({
        {&dispatch->map_ptrs,tb::from_blob(map_ptrs.data(),{int64_t(map_ptrs.size())},i64)},
        {&dispatch->weight_ptrs,tb::from_blob(weight_ptrs.data(),{dispatch_pool,5},i64)},
        {&dispatch->weight_sizes,tb::from_blob(weight_sizes.data(),{dispatch_pool,3},i64)},
        {&dispatch->pool_params,tb::from_blob(pool_params.data(),{dispatch_pool,7},i32)},
        {&dispatch->expert_pool,tb::from_blob(expert_pool.data(),{runtime.n_experts},i32)},
        {&dispatch->expert_local,tb::from_blob(expert_local.data(),{runtime.n_experts},i32)}},
        target,runtime.prefill_only,dispatch->metadata_device,dispatch->metadata_host);
    runtime.nvq_dispatch = std::move(dispatch);
}
