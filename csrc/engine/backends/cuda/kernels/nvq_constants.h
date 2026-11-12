#pragma once

#ifdef __CUDACC__
#define NVQ_HOST_DEVICE __host__ __device__
#else
#define NVQ_HOST_DEVICE
#endif

namespace tyloi::cuda::packed_nvq {
constexpr int kNvq1L = 1;
constexpr int kNvq2 = 2;
constexpr int kNvq3 = 3;
constexpr int kNvq2Exec = 4;
constexpr int kNvq2Jsc = 5;
constexpr int kNvq2JscExec = 6;
constexpr int kNpq0L = 7;
constexpr int kNvq1S = 8;
constexpr int kNpq0S = 9;
constexpr int kNvq3Jsc = 10;
constexpr int kNvq3Jsc2 = 11;
constexpr int kNvq3Jsc512 = 12;
constexpr int kNvq2JscL = 13;
constexpr int kNvq2JscXL = 14;
constexpr int kNvq3JscL = 15;
constexpr int kNvq2JscXLGroupExec = 16;
constexpr int kNvq3JscLGroupExec = 17;
constexpr int kGroupSize = 24;
constexpr int kChunksPerGroup = 6;  // six dp4a chunks of four values
constexpr int kJscMetadataBytes = 64;
constexpr int kJscLutOffset = 4;
constexpr int kJscBankMapOffset = 36;
constexpr int kJscE8CodebookBytes = 256 * 8;
constexpr int kJscE8Codebook1024Bytes = 1024 * 8;
constexpr int kJscE8Codebook4096Bytes = 4096 * 8;
constexpr int kJscE8PartialEntries = 384;
constexpr int kJscE8PartialBytes = kJscE8PartialEntries * 8;
constexpr int kJscD4CodebookBytes = 256 * 4;
constexpr int kJscD4Codebook512Bytes = 512 * 4;
constexpr int kJscD4Codebook1024Bytes = 1024 * 4;
constexpr int kNpq0LMetadataBytes = 64;
constexpr int kNpq0LLutOffset = 8;
constexpr int kNpq0LStates = 8;
constexpr int kNpq0LFirstEntries = 8;
constexpr int kNpq0LSecondEntries = 16;
constexpr int kNpq0LFirstBytesPerState = kNpq0LFirstEntries * 4;
constexpr int kNpq0LSecondBytesPerState = kNpq0LSecondEntries * 4;
constexpr int kNpq0LSecondOffset =
    kNpq0LMetadataBytes + kNpq0LStates * kNpq0LFirstBytesPerState;
constexpr int kNpq0LTableBytes =
    kNpq0LSecondOffset + kNpq0LStates * kNpq0LSecondBytesPerState;
constexpr int kNvq1SBankEntries = 512;
constexpr int kNvq1SBankBytes = kNvq1SBankEntries * 8;
constexpr int kNpq0SMetadataBytes = 64;
constexpr int kNpq0SLutOffset = 8;
constexpr int kNpq0SStates = 4;
constexpr int kNpq0SEntries = 64;
constexpr int kNpq0SCodebookBytes = kNpq0SStates * kNpq0SEntries * 8;
constexpr int kNpq0STableBytes = kNpq0SMetadataBytes + kNpq0SCodebookBytes;
constexpr int kNepq0SEntriesPerHalf = 8;
constexpr int kNepq0SBytesPerStateHalf = kNepq0SEntriesPerHalf * 4;
constexpr int kNepq0SSecondOffset =
    kNpq0SMetadataBytes + kNpq0SStates * kNepq0SBytesPerStateHalf;
constexpr int kNepq0SCompactTableBytes =
    kNepq0SSecondOffset + kNpq0SStates * kNepq0SBytesPerStateHalf;
constexpr int kNepqGroupsPerSupergroup = 4;

NVQ_HOST_DEVICE constexpr bool is_d4_format(int format) {
    return format == kNvq3 || format == kNvq3Jsc ||
           format == kNvq3Jsc2 || format == kNvq3Jsc512 ||
           format == kNvq3JscL || format == kNvq3JscLGroupExec;
}

NVQ_HOST_DEVICE constexpr bool is_e8_format(int format) {
    return format == kNvq2 || format == kNvq2Exec ||
           format == kNvq2Jsc || format == kNvq2JscExec ||
           format == kNvq2JscL || format == kNvq2JscXL ||
           format == kNvq2JscXLGroupExec;
}

NVQ_HOST_DEVICE constexpr int format_index_bits(int format) {
    if (format == kNvq1L) return 11;
    if (format == kNvq1S || format == kNvq3Jsc512) return 9;
    if (format == kNpq0L) return 7;
    if (format == kNpq0S) return 6;
    if (format == kNvq2JscL || format == kNvq3JscL ||
        format == kNvq3JscLGroupExec) return 10;
    if (format == kNvq2JscXL || format == kNvq2JscXLGroupExec) return 12;
    if (format == kNvq2Exec || format == kNvq2JscExec) return 16;
    return 8;
}

NVQ_HOST_DEVICE constexpr bool dense_nvq_format(int format) {
    return format==5 || format==10 || format==11 || format==12 ||
        format==13 || format==14 || format==15;
}
} // namespace tyloi::cuda::packed_nvq

#undef NVQ_HOST_DEVICE
