#pragma once

#include "mfq_tensor_backend.h"
#include "tyloinfer/model_source.h"
#include "tyloinfer/gguf_quant.h"
#include "runtime/row_stage.h"
#include <optional>
#include <functional>

struct GgufInputRotation;
class GgufRowCorrection;
// Activation-dependent neuron corrections retain the original input even
// when the base integer projection consumes a shared Q8 packet.
class GgufProjectionCorrection {
public:
    using Tensor=mfq_tensor_backend::Tensor;
    virtual ~GgufProjectionCorrection()=default;
    // Publish input-dependent asynchronous work before the low projection.
    virtual void prepare(const Tensor& input) const {}
    virtual Tensor apply(const Tensor& input,Tensor output) const=0;
};
struct GgufWeight {
    mfq_tensor_backend::Tensor packed;
    ggml_type type = GGML_TYPE_COUNT;
    std::int64_t out = 0, neuron_len = 0;
    std::uint64_t row_bytes = 0;
    std::shared_ptr<const GgufInputRotation> rotation;
    std::shared_ptr<const GgufRowCorrection> correction;
    std::shared_ptr<const GgufProjectionCorrection> projection_correction;
    // Internal lossless storage, independent of the on-disk GGML format.
    // Views retain the shared base allocation and address their original rows.
    std::int64_t quad_storage_rows = 0, quad_row_begin = 0;
    void repack_inplace();
    mfq_tensor_backend::Tensor canonical_packed() const;
    bool quad_projection_ready() const noexcept {
        return quad_storage_rows && quad_row_begin%4==0 && out%4==0 &&
            quad_row_begin+out<=quad_storage_rows/4*4;
    }
    bool base_prefers_original_input() const noexcept {
        return bool(correction) || bool(rotation) || (type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4);
    }
    bool prefers_original_input() const noexcept {return bool(projection_correction) || base_prefers_original_input();}
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor input,bool allow_mmq = true) const;
    mfq_tensor_backend::Tensor forward_base(mfq_tensor_backend::Tensor input,bool allow_mmq) const;
    bool can_forward_prefill_residual(std::int64_t tokens,const mfq_tensor_backend::Tensor& residual) const;
    mfq_tensor_backend::Tensor forward_base(mfq_tensor_backend::Tensor input,bool allow_mmq,
        const mfq_tensor_backend::Tensor& prefill_residual) const;
    // Consume the native Q8_1 packet produced by a fused upstream operator.
    mfq_tensor_backend::Tensor forward_q8(const mfq_tensor_backend::Tensor&,std::int64_t tokens,
        mfq_tensor_backend::ScalarType output_type,bool position_major=false) const;
    mfq_tensor_backend::Tensor forward_q8_base(const mfq_tensor_backend::Tensor&,std::int64_t tokens,
        mfq_tensor_backend::ScalarType output_type,bool position_major=false) const;
    // Internal aligned M=1 activation; native Q8 entry points stay unchanged.
    mfq_tensor_backend::Tensor forward_m1_aligned_base(const mfq_tensor_backend::Tensor&,
        mfq_tensor_backend::ScalarType output_type) const;
    mfq_tensor_backend::Tensor forward_gated_residual(
        const mfq_tensor_backend::Tensor& gate,const mfq_tensor_backend::Tensor& up,
        const mfq_tensor_backend::Tensor& residual) const;
    mfq_tensor_backend::Tensor embedding(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor embedding_base(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor dequantize() const;
    mfq_tensor_backend::Tensor dequantize_stored() const;
};
bool same_gguf_input_basis(const GgufWeight&, const GgufWeight&);
// Combine compatible immutable row images and replace each component with a
// view of the shared allocation. The total resident weight payload is unchanged.
GgufWeight combine_gguf_rows(const std::vector<GgufWeight*>& weights);
// Small-window projections share activation packets. Floating codebooks and
// transformed weights retain the original activation when it is required.
struct GgufProjectionInput {
    mfq_tensor_backend::Tensor native, interleaved, original;
    std::vector<std::int64_t> shape;
    std::int64_t tokens = 0;
    mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat32;
    // Selects the internal aligned coordinate/scale planes in native.
    bool aligned_m1 = false;
    mfq_tensor_backend::Tensor project(const GgufWeight& weight) const;
    std::vector<mfq_tensor_backend::Tensor> project_all(const std::vector<GgufWeight>& weights) const;
};
bool prefers_gguf_m1_aligned(const std::vector<GgufWeight>& weights,std::int64_t tokens);
GgufProjectionInput prepare_gguf_group_input(
    const std::vector<GgufWeight>& weights,mfq_tensor_backend::Tensor input);
std::vector<mfq_tensor_backend::Tensor> project_gguf_group(
    const std::vector<GgufWeight>& weights,mfq_tensor_backend::Tensor input);

struct CudaExecutionContext;
class GgufExpertCache;
std::size_t gguf_expert_cache_budget_bytes(const std::shared_ptr<GgufExpertCache>&);
std::uint64_t gguf_expert_streamed_bytes(const std::shared_ptr<GgufExpertCache>&);
double gguf_expert_fetch_ms(const std::shared_ptr<GgufExpertCache>&);
void resize_gguf_expert_cache(const std::shared_ptr<GgufExpertCache>&, std::size_t bytes);
std::size_t reserve_gguf_expert_host(const std::shared_ptr<GgufExpertCache>&, std::size_t bytes);
std::vector<std::pair<std::string, double>> gguf_expert_memory_metrics(const std::shared_ptr<GgufExpertCache>&);
bool prepare_gguf_graph_pool(const std::shared_ptr<GgufExpertCache>&);
void finalize_gguf_expert_cache(const std::shared_ptr<GgufExpertCache>&, std::size_t reserve_bytes = 0);
void reserve_gguf_expert_workspace(const std::shared_ptr<GgufExpertCache>&, std::size_t bytes);
// A prepared decoder can price its remaining transient tensors separately
// from graph/KV storage which is already allocated. No estimate preserves the
// conservative allowance used while graphs are still being prepared.
void restore_gguf_decode_workspace(const std::shared_ptr<GgufExpertCache>&, std::size_t growth_bytes = 0,
    std::optional<std::size_t> transient_bytes = {});
void restore_gguf_decode_workspace_after_capture(const std::shared_ptr<GgufExpertCache>&);
void begin_gguf_speculative_routes(const std::shared_ptr<GgufExpertCache>&,int rows);
void commit_gguf_speculative_routes(const std::shared_ptr<GgufExpertCache>&,int rows);
void wait_gguf_expert_admission(const std::shared_ptr<GgufExpertCache>&);
void discard_gguf_speculative_routes(const std::shared_ptr<GgufExpertCache>&) noexcept;
bool set_gguf_reuse_learning(const std::shared_ptr<GgufExpertCache>&,bool enabled);
std::size_t gguf_prefill_lendable_bytes(const std::shared_ptr<GgufExpertCache>&);
class GgufPrefillWorkspaceLoan {
public:
    explicit GgufPrefillWorkspaceLoan(std::shared_ptr<GgufExpertCache> cache):cache_(std::move(cache)) {}
    ~GgufPrefillWorkspaceLoan();
    std::size_t begin(std::size_t bytes);
    void suspend();
    void finish();
private:
    std::shared_ptr<GgufExpertCache> cache_;
    bool active_=false;
};
struct GgufRoutedWeight {
    std::shared_ptr<GgufExpertCache> cache;
    std::shared_ptr<const GgufInputRotation> rotation;
    std::shared_ptr<const GgufRowCorrection> correction;
    // Predictor experts are immutable model weights, outside the target's
    // evictable cache and its prefill workspace loans.
    mfq_tensor_backend::Tensor resident_packed;
    mfq_tensor_backend::Tensor resident_addresses;
    mfq::ModelSource::TensorReader reader;
    mfq::ModelSource::TensorBatchReader batch_reader;
    ggml_type type;
    std::uint64_t source_id, row_bytes, expert_bytes;
    std::int64_t experts, out, input;
    GgufRoutedWeight(CudaExecutionContext&, const mfq::ModelSource&, const std::string&);
    GgufWeight resident_expert(std::int64_t expert, int device) const;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor source,
        mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor forward_base(mfq_tensor_backend::Tensor source,
        mfq_tensor_backend::Tensor ids) const;
};
struct CudaIndependentBranchExecutor;
struct GgufMoeBanks {
    std::shared_ptr<GgufRoutedWeight> gate, up, down;
    // Independent shared-expert work runs after streamed routes are published.
    // The serial path starts it after starting the transfer stream.
    // A prepared branch executor can overlap it with routed computation;
    // the parent joins before consuming either result. Profiling is serial
    // so layer timestamps measure complete shared work, as in Strata.
    // Resident/prefill calls execute it once before their own computation.
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor source,
        mfq_tensor_backend::Tensor ids, mfq_tensor_backend::Tensor weights, float limit = 0,
        const std::function<void()>& independent_work = {},
        CudaIndependentBranchExecutor* shared_executor = nullptr) const;
};

GgufWeight load_gguf_weight(const mfq::ModelSource& source, const std::string& name,
    int device, bool cuda = true, std::int64_t row_begin = 0, std::int64_t row_end = -1,
    std::int64_t column_begin = 0, std::int64_t column_end = -1);
void decode_gguf_cpu(ggml_type type, const void* packed, float* output, std::int64_t columns);

class GgufRowTable : public std::enable_shared_from_this<GgufRowTable> {
    struct RowCache;
    std::shared_ptr<mfq::ModelSource::TensorReader> read_;
    mfq::ModelSource::TensorBatchReader batch_;
    std::unique_ptr<RowCache> cache_;
    ggml_type type_;
    std::int64_t rows_, columns_;
    std::uint64_t row_bytes_;
    std::shared_ptr<const mfq::RowResidualFile> correction_;
public:
    GgufRowTable(const mfq::ModelSource&, const std::string&);
    ~GgufRowTable();
    std::int64_t rows() const noexcept { return rows_; }
    std::int64_t width() const noexcept { return columns_; }
    std::uint64_t row_bytes() const noexcept { return row_bytes_; }
    ggml_type type() const noexcept { return type_; }
    void read_row(std::int64_t id, void* destination) const;
    void read_rows(const std::vector<mfq::ReadSpan>& rows) const;
    std::uint64_t staged_row_bytes() const noexcept { return correction_?std::uint64_t(columns_)*2:row_bytes_; }
    ggml_type staged_type() const noexcept { return correction_?GGML_TYPE_F16:type_; }
    void read_staged_rows(const std::vector<mfq::ReadSpan>& rows) const;
    void decode_row(std::int64_t id,float* output) const;
    mfq_tensor_backend::Tensor lookup(mfq_tensor_backend::Tensor ids) const;
};

std::shared_ptr<RowStage> make_gguf_row_stage(const std::vector<std::shared_ptr<GgufRowTable>>&,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& shape,
    const mfq_tensor_backend::Device& device);
mfq_tensor_backend::Tensor lookup_gguf_shards(const std::vector<std::shared_ptr<GgufRowTable>>&,
    std::int64_t rows_per_shard, const std::vector<std::int64_t>& ids,
    const mfq_tensor_backend::Device& device);
