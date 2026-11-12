#include "backend.h"
#include "packed_nvq.cuh"
#include "nvq_dense_group.cuh"
#include <cstdint>

namespace {
using namespace tyloi::cuda::packed_nvq;
void check_nepq_common(
    const ::tyloi::DenseTensor & indices,
    const ::tyloi::DenseTensor & aux,
    const ::tyloi::DenseTensor & sub_scale,
    const ::tyloi::DenseTensor & neuron_scale,
    const ::tyloi::DenseTensor & table_pool,
    const ::tyloi::DenseTensor & bank_ids,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    TYLOINFER_RUNTIME_CHECK(
        format == kNvq1L || format == kNpq0L ||
        format == kNvq1S || format == kNpq0S,
        "NEPQ base format must be NVQ1-L/NPQ0-L/NVQ1-S/NPQ0-S");
    TYLOINFER_RUNTIME_CHECK(neuron_len > 0 && neuron_len % 8 == 0,
                "NEPQ neuron_len must be a positive multiple of 8");
    const int expected_sub_bits = format == kNvq1S ? 4 :
        (format == kNpq0S ? 2 : 3);
    TYLOINFER_RUNTIME_CHECK(sub_bits == expected_sub_bits, "NEPQ state width mismatch");
    for (const auto * tensor : {&indices, &aux, &sub_scale, &bank_ids}) {
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->scalar_type() == ::tyloi::kUInt8 &&
                    tensor->is_contiguous(),
                    "NEPQ packed streams must be CUDA contiguous uint8");
    }
    TYLOINFER_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
                neuron_scale.is_contiguous() && neuron_scale.dim() == 1,
                "NEPQ neuron_scale must be CUDA contiguous float32 rank-1");
    TYLOINFER_RUNTIME_CHECK(table_pool.is_cuda() && table_pool.scalar_type() == ::tyloi::kInt8 &&
                table_pool.is_contiguous() && table_pool.dim() == 2 &&
                table_pool.size(0) >= 1 && table_pool.size(0) <= 256,
                "NEPQ table pool must be CUDA contiguous int8 [1..256,stride]");
    const int table_stride = format == kNvq1L ? 2048 * 8 :
        (format == kNvq1S ? 1024 * 8 :
         (format == kNpq0L ? kNpq0LTableBytes : kNepq0SCompactTableBytes));
    TYLOINFER_RUNTIME_CHECK(table_pool.size(1) == table_stride, "NEPQ table stride mismatch");
    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + kGroupSize - 1) / kGroupSize;
    const int64_t nvec = neuron_len / 8;
    const int64_t nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    TYLOINFER_RUNTIME_CHECK(bank_ids.dim() == 2 && bank_ids.size(0) == N &&
                bank_ids.size(1) == nsuper,
                "NEPQ bank selectors must have shape [rows,ceil(groups/4)]");
    const int index_bits = format == kNvq1L ? 11 :
        (format == kNvq1S ? 9 : (format == kNpq0L ? 7 : 6));
    const int64_t index_bytes = (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const int64_t aux_bytes = delta_format ? (N * ng + 7) / 8 : 0;
    const int64_t sub_bytes = (N * ng * sub_bits + 7) / 8;
    TYLOINFER_RUNTIME_CHECK(indices.numel() == index_bytes, "NEPQ index stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(aux.numel() == aux_bytes, "NEPQ aux stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(sub_scale.numel() == sub_bytes, "NEPQ state stream length mismatch");
}

void check_common(
    const ::tyloi::DenseTensor & indices,
    const ::tyloi::DenseTensor & aux,
    const ::tyloi::DenseTensor & sub_scale,
    const ::tyloi::DenseTensor & neuron_scale,
    const ::tyloi::DenseTensor & codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,bool allow_dense=false) {
    TYLOINFER_RUNTIME_CHECK(format >= kNvq1L && format <= kNvq3JscLGroupExec,
                "NVQ format must be in [1,17]");
    TYLOINFER_RUNTIME_CHECK(gs == kGroupSize, "NVQ CUDA kernels currently require gs=24");
    TYLOINFER_RUNTIME_CHECK(sub_bits >= 1 && sub_bits <= 8, "NVQ sub_bits must be in [1,8]");
    TYLOINFER_RUNTIME_CHECK(neuron_len > 0, "NVQ neuron_len must be positive");
    TYLOINFER_RUNTIME_CHECK(sign_mode == 0 ||
                (sign_mode == 1 && (format == kNvq2 || format == kNvq2Exec)),
                "index-parity sign mode is valid only for NVQ2");
    for (const auto * tensor : {&indices, &aux, &sub_scale}) {
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->scalar_type() == ::tyloi::kUInt8 && tensor->is_contiguous(),
                    "NVQ packed streams must be CUDA contiguous uint8");
    }
    TYLOINFER_RUNTIME_CHECK(neuron_scale.is_cuda() && neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
                neuron_scale.is_contiguous(), "NVQ neuron_scale must be CUDA contiguous float32");
    TYLOINFER_RUNTIME_CHECK(codebook.is_cuda() && codebook.scalar_type() == ::tyloi::kInt8 && codebook.is_contiguous(),
                "NVQ codebook must be CUDA contiguous int8");
    if (format == kNvq1L) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 2048 && codebook.size(1) == 8,
                    "NVQ1-L codebook must have shape [2048,8]");
    } else if (format == kNvq1S) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 1024 && codebook.size(1) == 8,
                    "NVQ1-S codebook must have shape [1024,8]");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ1-S requires a 4-bit scale and no sign mode");
    } else if (format == kNvq2 || format == kNvq2Exec) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 8,
                    "NVQ2 codebook must have shape [256,8]");
    } else if (format == kNvq2Jsc || format == kNvq2JscExec) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * kJscE8CodebookBytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * kJscE8CodebookBytes),
                    "NVQ2J codebook metadata has an invalid size");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq2JscL || format == kNvq2JscXL ||
        format == kNvq2JscXLGroupExec) {
        const int e8_bank_bytes = format == kNvq2JscL
            ? kJscE8Codebook1024Bytes
            : kJscE8Codebook4096Bytes;
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * e8_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * e8_bank_bytes),
                    "extended NVQ2J codebook metadata has an invalid size");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "extended NVQ2J requires a 4-bit state and even-parity signs");
    } else if (
        format == kNvq3Jsc || format == kNvq3Jsc2 ||
        format == kNvq3Jsc512 || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) {
        const int d4_bank_bytes = format == kNvq3Jsc512
            ? kJscD4Codebook512Bytes
            : (format == kNvq3JscL || format == kNvq3JscLGroupExec
               ? kJscD4Codebook1024Bytes
               : kJscD4CodebookBytes);
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 &&
                    (codebook.numel() == kJscMetadataBytes + d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 2 * d4_bank_bytes ||
                     codebook.numel() == kJscMetadataBytes + 4 * d4_bank_bytes),
                    "NVQ3J codebook metadata has an invalid size");
        if (format == kNvq3Jsc2) {
            TYLOINFER_RUNTIME_CHECK(
                codebook.numel() == kJscMetadataBytes + 2 * kJscD4CodebookBytes,
                "NVQ3J analytic-2 requires exactly two D4 banks");
        }
        TYLOINFER_RUNTIME_CHECK(sub_bits == 4 && sign_mode == 0,
                    "NVQ3J requires a 4-bit state and even-parity signs");
    } else if (format == kNpq0L) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0LTableBytes,
                    "NPQ0-L codebook metadata must contain 832 bytes");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 3 && sign_mode == 0,
                    "NPQ0-L requires a 3-bit state and no sign stream");
    } else if (format == kNpq0S) {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 1 && codebook.numel() == kNpq0STableBytes,
                    "NPQ0-S expanded PQ3+3 runtime LUT must contain 2112 bytes");
        TYLOINFER_RUNTIME_CHECK(sub_bits == 2 && sign_mode == 0,
                    "NPQ0-S requires a 2-bit state and no sign stream");
    } else {
        TYLOINFER_RUNTIME_CHECK(codebook.dim() == 2 && codebook.size(0) == 256 && codebook.size(1) == 4,
                    "NVQ3 codebook must have shape [256,4]");
    }

    const int64_t N = neuron_scale.numel();
    const int64_t ng = (neuron_len + gs - 1) / gs;
    const int64_t nvec = (neuron_len + (is_d4_format(format) ? 3 : 7)) /
                         (is_d4_format(format) ? 4 : 8);
    const int64_t nsign = (neuron_len + 7) / 8;
    const bool exec_layout = format == kNvq2Exec || format == kNvq2JscExec;
    const bool group_exec_layout =
        format == kNvq2JscXLGroupExec || format == kNvq3JscLGroupExec;
    const int index_bits = format_index_bits(format);
    const bool dense=allow_dense && dense_nvq_format(int(format)) && aux.numel()==0 && sub_scale.numel()==0;
    if(dense) {
        TYLOINFER_RUNTIME_CHECK(sub_bits==4 && (!is_d4_format(int(format)) || nvec==2*nsign),
            "compact NVQ requires complete D4 index pairs and 4-bit state");
        const int64_t bytes=(N*(nvec*index_bits+nsign*7+ng*4)+7)/8;
        TYLOINFER_RUNTIME_CHECK(indices.numel()==bytes,"compact NVQ stream length mismatch");
        return;
    }
    const bool nvq1_records=(format==kNvq1L || format==kNvq1S) && aux.numel()==0;
    if(nvq1_records)TYLOINFER_RUNTIME_CHECK(sub_bits==(format==kNvq1S?4:3),"NVQ1 record state width mismatch");
    const int64_t index_bytes = nvq1_records ? N*ng*(format==kNvq1S?4:5) : group_exec_layout
        ? N * ng * (format == kNvq2JscXLGroupExec ? 8 : 12)
        : (N * nvec * index_bits + 7) / 8;
    const bool delta_format = format == kNvq1L || format == kNvq1S;
    const bool no_aux_format = format == kNpq0L || format == kNpq0S;
    const int aux_bits = delta_format ? 1 : (no_aux_format ? 0 : 7);
    const int64_t aux_count = delta_format ? N * ng :
        (no_aux_format ? 0 : N * nsign);
    const int64_t aux_bytes = exec_layout || group_exec_layout || nvq1_records
        ? 0
        : (aux_count * aux_bits + 7) / 8;
    const int64_t sub_bytes = nvq1_records || (group_exec_layout && sub_scale.numel()==0)?
        0:(N * ng * sub_bits + 7) / 8;
    TYLOINFER_RUNTIME_CHECK(indices.numel() == index_bytes, "NVQ index stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(aux.numel() == aux_bytes, "NVQ aux stream length mismatch");
    TYLOINFER_RUNTIME_CHECK(sub_scale.numel() == sub_bytes, "NVQ sub-scale stream length mismatch");
}

} // namespace

/** Validate native NVQ tensors with the original compact-format checks. */
void nvq_check_native(const ::tyloi::DenseTensor& indices, const ::tyloi::DenseTensor& aux,
    const ::tyloi::DenseTensor& state, const ::tyloi::DenseTensor& scale,
    const ::tyloi::DenseTensor& table, int64_t width, int64_t gs, int64_t bits,
    int64_t format, int64_t sign) {
    check_common(indices, aux, state, scale, table, width, gs, bits, format, sign);
}
/** Validate native NEPQ tensors with the original shared-bank checks. */
void nepq_check_native(const ::tyloi::DenseTensor& indices, const ::tyloi::DenseTensor& aux,
    const ::tyloi::DenseTensor& state, const ::tyloi::DenseTensor& scale,
    const ::tyloi::DenseTensor& table, const ::tyloi::DenseTensor& banks,
    int64_t width, int64_t bits, int64_t format) {
    check_nepq_common(indices, aux, state, scale, table, banks, width, bits, format);
}
/** Validate native NVQ buffers with compact dense-group layouts enabled. */
void nvq_check_native_dense(const ::tyloi::DenseTensor& indices, const ::tyloi::DenseTensor& aux,
    const ::tyloi::DenseTensor& state, const ::tyloi::DenseTensor& scale,
    const ::tyloi::DenseTensor& table, int64_t width, int64_t gs, int64_t bits,
    int64_t format, int64_t sign) {
    check_common(indices, aux, state, scale, table, width, gs, bits, format, sign, true);
}
