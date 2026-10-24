std::string make_single_row_mxfp8_source(
    const std::vector<DirectProjectionLayout>& layouts) {
    std::string source = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 8u;
    constexpr uint ROWS_PER_SIMD = 32u / K_LANES;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS * ROWS_PER_SIMD;
    constexpr uint MX_BLOCK = 128u;
    constexpr uint MX_BLOCKS = uint(K) / MX_BLOCK;
    constexpr uint Q8_GROUP = 32u;
    constexpr uint Q8_GROUPS_PER_MX = MX_BLOCK / Q8_GROUP;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint k_lane = lane & (K_LANES - 1u);
    uint simd_row = lane / K_LANES;
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simd_group * ROWS_PER_SIMD + simd_row;
)METAL";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    bool active_" + suffix
            + " = output_index < uint(P" + suffix + "_OUT);\n"
            "    uint output_" + suffix + " = min(\n"
            "        output_index, uint(P" + suffix + "_OUT) - 1u);\n"
            "    float accumulator_" + suffix + " = 0.0f;\n";
    }

    source += R"METAL(
    for (
        uint block = k_lane;
        block < MX_BLOCKS;
        block += K_LANES
    ) {
        uint column_base = block * MX_BLOCK;
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family == kFamilyMx) {
            source +=
                "        float mx_dot_" + std::to_string(projection)
                + " = 0.0f;\n";
        }
    }

    source += R"METAL(
        for (
            uint q8_local = 0u;
            q8_local < Q8_GROUPS_PER_MX;
            ++q8_local
        ) {
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family == kFamilyNint8Zero) {
            source +=
                "            float q8_dot_" + std::to_string(projection)
                + " = 0.0f;\n";
        }
    }

    source += R"METAL(
            uint q8_column_base =
                column_base + q8_local * Q8_GROUP;
            for (uint element = 0u; element < Q8_GROUP; element += 4u) {
                uint column = q8_column_base + element;
                half4 activation = *(device const half4*)(x + column);
)METAL";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        if (layouts[projection].family == kFamilyMx) {
            source +=
                "                if (active_" + suffix + ") {\n"
                "                    uint value_offset = output_" + suffix
                + " * uint(K) + column;\n"
                "                    uchar4 code = *(device const uchar4*)(\n"
                "                        mx_values_" + suffix
                + " + value_offset);\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.x),\n"
                "                        float(fp8_lut[uint(code.x)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.y),\n"
                "                        float(fp8_lut[uint(code.y)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.z),\n"
                "                        float(fp8_lut[uint(code.z)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                    mx_dot_" + suffix + " = fma(\n"
                "                        float(activation.w),\n"
                "                        float(fp8_lut[uint(code.w)]),\n"
                "                        mx_dot_" + suffix + ");\n"
                "                }\n";
        } else {
            source +=
                "                if (active_" + suffix + ") {\n"
                "                    uint value_offset = output_" + suffix
                + " * uint(K) + column;\n"
                "                    char4 quantized = *(device const char4*)(\n"
                "                        q8_q_" + suffix
                + " + value_offset);\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.x),\n"
                "                        float(quantized.x),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.y),\n"
                "                        float(quantized.y),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.z),\n"
                "                        float(quantized.z),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                    q8_dot_" + suffix + " = fma(\n"
                "                        float(activation.w),\n"
                "                        float(quantized.w),\n"
                "                        q8_dot_" + suffix + ");\n"
                "                }\n";
        }
    }

    source += "            }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family != kFamilyNint8Zero) {
            continue;
        }
        const auto suffix = std::to_string(projection);
        source +=
            "            if (active_" + suffix + ") {\n"
            "                uint group = block * Q8_GROUPS_PER_MX"
            " + q8_local;\n"
            "                float scale = float(q8_scales_" + suffix
            + "[output_" + suffix + " * uint(P" + suffix
            + "_NG) + group]);\n"
            "                accumulator_" + suffix + " = fma(\n"
            "                    scale, q8_dot_" + suffix + ",\n"
            "                    accumulator_" + suffix + ");\n"
            "            }\n";
    }
    source += "        }\n";
    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        if (layouts[projection].family != kFamilyMx) {
            continue;
        }
        const auto suffix = std::to_string(projection);
        source +=
            "        if (active_" + suffix + ") {\n"
            "            uint scale_offset =\n"
            "                (output_" + suffix
            + " / MX_BLOCK) * MX_BLOCKS + block;\n"
            "            accumulator_" + suffix + " = fma(\n"
            "                mfq_grouped_mx_e8m0(\n"
            "                    mx_scales_" + suffix
            + "[scale_offset]),\n"
            "                mx_dot_" + suffix + ",\n"
            "                accumulator_" + suffix + ");\n"
            "        }\n";
    }
    source += "    }\n";

    for (
        std::size_t projection = 0;
        projection < layouts.size();
        ++projection
    ) {
        const auto suffix = std::to_string(projection);
        source +=
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 4);\n"
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 2);\n"
            "    accumulator_" + suffix
            + " += simd_shuffle_down(accumulator_" + suffix + ", 1);\n"
            "    if (k_lane == 0u && active_" + suffix + ") {\n"
            "        y[uint(P" + suffix + "_OUT_OFFSET) + output_index] =\n"
            "            half(accumulator_" + suffix + ");\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction single_row_mxfp8_kernel(
    const std::vector<DirectProjectionLayout>& layouts) {
    static std::mutex mutex;
    static std::unordered_map<
        std::string,
        mlx::core::fast::CustomKernelFunction> kernels;
    const auto key = "mxfp8_m1_" + direct_kernel_key(layouts);
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(key);
    if (found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_single_row_grouped_" + key,
        direct_input_names(layouts),
        {"y"},
        make_single_row_mxfp8_source(layouts),
        kNativeMxProjectionHeader,
        true,
        false,
        options);
    kernels.emplace(key, kernel);
    return kernel;
}

constexpr const char* kSingleRowMxfp8PairSwiglu = R"METAL(
    constexpr uint SIMD_GROUPS = 4u;
    constexpr uint K_LANES = 16u;
    constexpr uint ROWS_PER_TG = SIMD_GROUPS;
    constexpr uint BLOCK = 128u;
    constexpr uint BLOCKS = uint(K) / BLOCK;

    threadgroup half fp8_lut[256];
    uint local_thread = thread_index_in_threadgroup;
    fp8_lut[local_thread] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread]);
    fp8_lut[local_thread + 128u] = as_type<half>(
        mfq_grouped_mx_fp8_half_lut[local_thread + 128u]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint lane = thread_index_in_simdgroup;
    uint projection = lane >> 4u;
    uint k_lane = lane & (K_LANES - 1u);
    uint output_index =
        threadgroup_position_in_grid.x * ROWS_PER_TG
        + simdgroup_index_in_threadgroup;
    uint output = min(output_index, uint(OUT) - 1u);
    uint value_base = output * uint(K);
    uint scale_base = (output / BLOCK) * BLOCKS;
    device const uchar* values = projection == 0u
        ? mx_values_0
        : mx_values_1;
    float accumulators[M];
    for (uint row = 0u; row < uint(M); ++row) {
        accumulators[row] = 0.0f;
    }

    // Gate and Up occupy separate SIMD half-groups. Both halves visit the
    // same 128-column block at once: the 16 lanes in each half read one
    // contiguous 128-byte MXFP8 row segment instead of maintaining 16
    // strided weight streams. The activation is tiny and remains cache-hot.
    for (uint block = 0u; block < BLOCKS; ++block) {
        uint column_base = block * BLOCK;
        uint column = column_base + k_lane * 8u;
        uchar4 code0 = *(device const uchar4*)(
            values + value_base + column);
        uchar4 code1 = *(device const uchar4*)(
            values + value_base + column + 4u);
        float4 weight0 = float4(
            float(fp8_lut[uint(code0.x)]),
            float(fp8_lut[uint(code0.y)]),
            float(fp8_lut[uint(code0.z)]),
            float(fp8_lut[uint(code0.w)]));
        float4 weight1 = float4(
            float(fp8_lut[uint(code1.x)]),
            float(fp8_lut[uint(code1.y)]),
            float(fp8_lut[uint(code1.z)]),
            float(fp8_lut[uint(code1.w)]));
        uchar scale = projection == 0u
            ? mx_scales_0[scale_base + block]
            : mx_scales_1[scale_base + block];
        float block_scale = mfq_grouped_mx_e8m0(scale);
        for (uint row = 0u; row < uint(M); ++row) {
            uint input_base = row * uint(K);
            activation4_t activation0 =
                *(device const activation4_t*)(x + input_base + column);
            activation4_t activation1 = *(device const activation4_t*)(
                x + input_base + column + 4u);
            float block_dot =
                dot(float4(activation0), weight0)
                + dot(float4(activation1), weight1);
            accumulators[row] = fma(
                block_scale,
                block_dot,
                accumulators[row]);
        }
    }

    // Each 16-lane half is an independent reduction tree. Only lanes 0 and
    // 16 are consumed, so shuffle-down traffic from inactive upper nodes
    // cannot cross-contaminate the two projection sums.
    for (uint row = 0u; row < uint(M); ++row) {
        float accumulator = accumulators[row];
        accumulator += simd_shuffle_down(accumulator, 8);
        accumulator += simd_shuffle_down(accumulator, 4);
        accumulator += simd_shuffle_down(accumulator, 2);
        accumulator += simd_shuffle_down(accumulator, 1);
        float gate = simd_shuffle(accumulator, 0u);
        float up = simd_shuffle(accumulator, 16u);
        if (lane == 0u && output_index < uint(OUT)) {
            // Match the unfused MXFP8 GEMV activation-dtype boundary.
            gate = float(activation_t(gate));
            up = float(activation_t(up));
            if (params[0] > 0.0f) {
                gate = min(gate, params[0]);
                up = clamp(up, -params[0], params[0]);
            }
            float activated = gate / (1.0f + exp(-gate));
            y[row * uint(OUT) + output_index] =
                activation_t(activated * up);
        }
    }
)METAL";

const mlx::core::fast::CustomKernelFunction&
single_row_mxfp8_pair_swiglu_kernel(Dtype dtype) {
    static const auto fp16_kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_single_row_mxfp8_pair_swiglu_f16",
            {
                "mx_values_0",
                "mx_scales_0",
                "mx_values_1",
                "mx_scales_1",
                "x",
                "params",
            },
            {"y"},
            std::string(
                "using activation_t = half;\n"
                "using activation4_t = half4;\n") +
                kSingleRowMxfp8PairSwiglu,
            kNativeMxProjectionHeader,
            true,
            false,
            options);
    }();
    static const auto bf16_kernel = [] {
        CompileOptions options;
        options.math_mode = MathMode::Fast;
        return mlx::core::fast::metal_kernel(
            "mfq_cpp_single_row_mxfp8_pair_swiglu_bf16",
            {
                "mx_values_0",
                "mx_scales_0",
                "mx_values_1",
                "mx_scales_1",
                "x",
                "params",
            },
            {"y"},
            std::string(
                "using activation_t = bfloat;\n"
                "using activation4_t = bfloat4;\n") +
                kSingleRowMxfp8PairSwiglu,
            kNativeMxProjectionHeader,
            true,
            false,
            options);
    }();
    return dtype == mlx::core::bfloat16
        ? bf16_kernel
        : fp16_kernel;
}
