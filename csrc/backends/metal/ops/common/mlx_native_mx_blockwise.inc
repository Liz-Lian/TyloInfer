std::string make_direct_small_m_blockwise_source(
    const std::vector<DirectProjectionLayout>& layouts,
    bool vectorized_fp16,
    bool vector_nvq1_inputs) {
    std::string source = R"METAL(
    constexpr uint SIMD_GROUPS = 2u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint global_tile = threadgroup_position_in_grid.x;
)METAL";

    for (std::size_t projection = 0;
         projection < layouts.size();
         ++projection) {
        const auto& layout = layouts[projection];
        const auto suffix = std::to_string(projection);
        source += projection == 0 ? "    if (" : "    else if (";
        source += "global_tile >= uint(P" + suffix
            + "_TILE_BEGIN) && global_tile < uint(P" + suffix
            + "_TILE_END)) {\n";
        source +=
            "        uint local_tile = global_tile - uint(P" + suffix
            + "_TILE_BEGIN);\n"
            "        uint output_index = local_tile * ROWS_PER_TG"
            " + simd_group * ROWS_PER_SIMD + simd_row;\n"
            "        uint output = min(output_index, uint(P" + suffix
            + "_OUT) - 1u);\n"
            "        float accumulators[ROWS];\n"
            "        for (uint row = 0u; row < uint(ROWS); ++row) {\n"
            "            accumulators[row] = 0.0f;\n"
            "        }\n";

            const int block = layout.bits == 4 ? 32 : 128;
            const int vectors = block / 4;
            source +=
                "        constexpr uint MX_BLOCK = "
                + std::to_string(block) + "u;\n"
                "        constexpr uint MX_BLOCKS = uint(K) / MX_BLOCK;\n"
                "        for (uint block = k_lane; block < MX_BLOCKS;"
                " block += K_LANES) {\n"
                "            uint column_base = block * MX_BLOCK;\n"
                "            float mx_accumulators[ROWS];\n"
                "            for (uint row = 0u; row < uint(ROWS); ++row) {\n"
                "                mx_accumulators[row] = 0.0f;\n"
                "            }\n";
            if (layout.bits == 4) {
                source +=
                    "            uint value_base = output * (uint(K) / 2u);\n";
            } else {
                source +=
                    "            uint value_base = output * uint(K);\n";
            }
            source +=
                "            for (uint vector = 0u; vector < "
                + std::to_string(vectors) + "u; ++vector) {\n"
                "                uint column = column_base + vector * 4u;\n";
            if (layout.bits == 4) {
                source +=
                    "                uint packed = uint(*(device const ushort*)("
                    "mx_values_" + suffix
                    + " + value_base + (column >> 1u)));\n"
                    "                float4 weights = float4(\n"
                    "                    mfq_grouped_mx_fp4(uchar(packed & 15u)),\n"
                    "                    mfq_grouped_mx_fp4(uchar((packed >> 4u) & 15u)),\n"
                    "                    mfq_grouped_mx_fp4(uchar((packed >> 8u) & 15u)),\n"
                    "                    mfq_grouped_mx_fp4(uchar((packed >> 12u) & 15u)));\n";
            } else {
                source +=
                    "                uchar4 codes = as_type<uchar4>("
                    "*(device const uint*)(mx_values_" + suffix
                    + " + value_base + column));\n"
                    "                float4 weights = float4(\n"
                    "                    mfq_grouped_mx_fp8(codes.x),\n"
                    "                    mfq_grouped_mx_fp8(codes.y),\n"
                    "                    mfq_grouped_mx_fp8(codes.z),\n"
                    "                    mfq_grouped_mx_fp8(codes.w));\n";
            }
            source +=
                "                for (uint row = 0u; row < uint(ROWS); ++row) {\n"
                "                    half4 activation = *(device const half4*)(\n"
                "                        x + row * uint(K) + column);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.x), weights.x,\n"
                "                        mx_accumulators[row]);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.y), weights.y,\n"
                "                        mx_accumulators[row]);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.z), weights.z,\n"
                "                        mx_accumulators[row]);\n"
                "                    mx_accumulators[row] = fma(\n"
                "                        float(activation.w), weights.w,\n"
                "                        mx_accumulators[row]);\n"
                "                }\n"
                "            }\n"
                "            float scale = mfq_grouped_mx_e8m0(";
            if (layout.bits == 4) {
                source +=
                    "mx_scales_" + suffix
                    + "[output * MX_BLOCKS + block]);\n";
            } else {
                source +=
                    "mx_scales_" + suffix
                    + "[(output / 128u) * MX_BLOCKS + block]);\n";
            }
            source +=
                "            for (uint row = 0u; row < uint(ROWS); ++row) {\n"
                "                accumulators[row] = fma(\n"
                "                    scale, mx_accumulators[row],\n"
                "                    accumulators[row]);\n"
                "            }\n"
                "        }\n";

        source += R"METAL(
        for (uint row = 0u; row < uint(ROWS); ++row) {
            accumulators[row] += simd_shuffle_down(accumulators[row], 4);
            accumulators[row] += simd_shuffle_down(accumulators[row], 2);
            accumulators[row] += simd_shuffle_down(accumulators[row], 1);
)METAL";
        source +=
            "            if (k_lane == 0u && output_index < uint(P" + suffix
            + "_OUT)) {\n"
            "                y[row * uint(TOTAL_OUT) + uint(P" + suffix
            + "_OUT_OFFSET) + output_index] = T(accumulators[row]);\n"
            "            }\n"
            "        }\n"
            "    }\n";
    }
    return source;
}
