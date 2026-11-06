#pragma once
#include "tyloinfer/qsa_adaptive_cache.h"
#include <memory>
#include <cstdint>
#include <vector>

namespace tyloinfer::qsa {
struct CpuTqRow {
    std::uint16_t anchor=0;
    std::uint8_t selector=7,bits=8;
    std::uint8_t state_bits=16;
    std::vector<std::uint8_t> payload;
    double sse=0;
};
// Closed rings with an explicit cache trellis profile. The default retains
// ordinary TQ16; smaller profiles keep the same bits-per-value payload budget.
CpuTqRow fit_cpu_tq_row(const float* input,int bits,int candidate_budget=1024,int state_bits=16);
std::vector<float> decode_cpu_tq_row(const CpuTqRow& row);

class CpuTqBlockFitter {
public:
    // Accept FP16 H4 modes, Affine8, or canonical TQ8. Decode once,
    // allocate each block's budget by row energy,
    // then fit only the selected precision of each row.
    CpuTqBlockFitter(const MicroblockImage& canonical,std::size_t rows_per_kind,int workers,
        RowPrecisionAllocation allocation=RowPrecisionAllocation::Energy,int state_bits=16);
    ~CpuTqBlockFitter();
    CpuTqBlockFitter(const CpuTqBlockFitter&)=delete;
    CpuTqBlockFitter& operator=(const CpuTqBlockFitter&)=delete;
    MicroblockImage image(Precision precision) const;
    std::size_t bytes() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloinfer::qsa
