std::string make_direct_source(
    const std::vector<DirectProjectionLayout>& layouts,
    bool batch_rows,
    int rows) {
    std::string source = batch_rows ? R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint global_tile = threadgroup_position_in_grid.x;

    uint projection = 0u;
    uint local_tile = 0u;
    uint output_width = 0u;
    uint output_offset = 0u;
)METAL" : R"METAL(
    constexpr uint ROWS_PER_SIMD = 4u;
    constexpr uint ROWS_PER_TG = 8u;

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint workgroup = threadgroup_position_in_grid.x;
    uint input_row = workgroup / uint(TOTAL_TILES);
    uint global_tile =
        workgroup - input_row * uint(TOTAL_TILES);
    if (input_row >= uint(ROWS)) {
        return;
    }

    uint projection = 0u;
    uint local_tile = 0u;
    uint output_width = 0u;
    uint output_offset = 0u;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (projection == 0) {
            source += "    if (";
        } else if (projection + 1 < layouts.size()) {
            source += " else if (";
        } else {
            source += " else {\n";
            source += "        projection = "
                + suffix + "u;\n";
            source += "        local_tile = global_tile"
                " - uint(P" + suffix + "_TILE_BEGIN);\n";
            source += "        output_width = uint(P"
                + suffix + "_OUT);\n";
            source += "        output_offset = uint(P"
                + suffix + "_OUT_OFFSET);\n";
            source += "    }\n";
            continue;
        }
        source += "global_tile < uint(P"
            + suffix + "_TILE_END)) {\n";
        source += "        projection = "
            + suffix + "u;\n";
        source += "        local_tile = global_tile"
            " - uint(P" + suffix + "_TILE_BEGIN);\n";
        source += "        output_width = uint(P"
            + suffix + "_OUT);\n";
        source += "        output_offset = uint(P"
            + suffix + "_OUT_OFFSET);\n";
        source += "    }";
    }

    source += batch_rows ? R"METAL(
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS][ROWS_PER_SIMD];
    for (uint input_row = 0u;
         input_row < uint(ROWS);
         ++input_row) {
        for (uint local_row = 0u;
             local_row < ROWS_PER_SIMD;
             ++local_row) {
            accumulators[input_row][local_row] = 0.0f;
        }
    }
    for (uint column = lane; column < uint(K); column += 32u) {
        float activations[ROWS];
        for (uint input_row = 0u;
             input_row < uint(ROWS);
             ++input_row) {
            activations[input_row] =
                float(x[input_row * uint(K) + column]);
        }
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            float weight = 0.0f;
)METAL" : R"METAL(
    uint output_base =
        local_tile * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD;

    float accumulators[ROWS_PER_SIMD] = {0.0f};
    for (uint column = lane; column < uint(K); column += 32u) {
        float activation =
            float(x[input_row * uint(K) + column]);
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            uint output = output_base + local_row;
            if (output >= output_width) {
                continue;
            }

            float weight = 0.0f;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source += projection == 0
            ? "            if ("
            : " else if (";
        source += "projection == " + suffix + "u) {\n";
        source +=
            "                weight = mfq_grouped_mx_weight(\n"
            "                    mx_values_" + suffix + ",\n"
            "                    mx_scales_" + suffix + ",\n"
            "                    output, column,\n"
            "                    uint(P" + suffix + "_BITS),\n"
            "                    uint(K));\n";
        source += "            }";
    }

    source += batch_rows ? R"METAL(
            for (uint input_row = 0u;
                 input_row < uint(ROWS);
                 ++input_row) {
                accumulators[input_row][local_row] = fma(
                    activations[input_row],
                    weight,
                    accumulators[input_row][local_row]);
            }
        }
    }

    for (uint input_row = 0u;
         input_row < uint(ROWS);
         ++input_row) {
        for (
            uint local_row = 0u;
            local_row < ROWS_PER_SIMD;
            ++local_row
        ) {
            float total = simd_sum(
                accumulators[input_row][local_row]);
            uint output = output_base + local_row;
            if (lane == 0u && output < output_width) {
                y[
                    input_row * uint(TOTAL_OUT)
                    + output_offset + output
                ] = T(total);
            }
        }
    }
)METAL" : R"METAL(
            accumulators[local_row] = fma(
                activation,
                weight,
                accumulators[local_row]);
        }
    }

    for (
        uint local_row = 0u;
        local_row < ROWS_PER_SIMD;
        ++local_row
    ) {
        float total = simd_sum(accumulators[local_row]);
        uint output = output_base + local_row;
        if (lane == 0u && output < output_width) {
            y[
                input_row * uint(TOTAL_OUT)
                + output_offset + output
            ] = T(total);
        }
    }
)METAL";
    return source;
}
