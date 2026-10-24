std::vector<std::string> dense_projection_group_input_names(
    std::size_t projections) {
    std::vector<std::string> names;
    names.reserve(projections + 1);
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        names.push_back("w_" + std::to_string(projection));
    }
    names.push_back("x");
    return names;
}

constexpr const char* kDenseProjectionHeader = R"METAL(
template <typename Scalar, uint WIDTH>
inline float4 mfq_dense_projection_read4(
    device const Scalar* values, uint row, uint column) {
    const uint base = row * WIDTH + column;
    if constexpr (WIDTH % 4u == 0u) {
        return float4(*reinterpret_cast<device const vec<Scalar, 4>*>(values + base));
    } else {
        return float4(
            float(values[base]),
            column + 1u < WIDTH ? float(values[base + 1u]) : 0.0f,
            column + 2u < WIDTH ? float(values[base + 2u]) : 0.0f,
            column + 3u < WIDTH ? float(values[base + 3u]) : 0.0f);
    }
}

template <typename Scalar, uint WIDTH>
inline float4 mfq_dense_projection_read4(
    constant const Scalar* values, uint row, uint column) {
    const uint base = row * WIDTH + column;
    return float4(
        float(values[base]),
        column + 1u < WIDTH ? float(values[base + 1u]) : 0.0f,
        column + 2u < WIDTH ? float(values[base + 2u]) : 0.0f,
        column + 3u < WIDTH ? float(values[base + 3u]) : 0.0f);
}
)METAL";

std::string make_dense_projection_group_source(
    std::size_t projections) {
    std::string source = R"METAL(
    uint tile = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint output_lane = simdgroup_index_in_threadgroup;
)METAL";
    for (std::size_t projection = 0;
         projection < projections;
         ++projection) {
        const auto suffix = std::to_string(projection);
        source +=
            "    if (tile >= uint(P" + suffix + "_TILE_BEGIN) &&\n"
            "        tile < uint(P" + suffix + "_TILE_END)) {\n"
            "        uint output = (tile - uint(P" + suffix
            + "_TILE_BEGIN)) * 4u + output_lane;\n"
            "        if (output >= uint(P" + suffix + "_OUT)) return;\n"
            "        float accumulators[M];\n"
            "        for (uint row = 0u; row < uint(M); ++row) {\n"
            "            accumulators[row] = 0.0f;\n"
            "        }\n"
            "        for (uint column = lane * 4u; column < uint(K); "
            "column += 128u) {\n"
            "            float4 weight = mfq_dense_projection_read4<T, uint(K)>(w_" + suffix
            + ", output, column);\n"
            "            for (uint row = 0u; row < uint(M); ++row) {\n"
            "                float4 activation = mfq_dense_projection_read4<T, uint(K)>(x, row, column);\n"
            "                accumulators[row] += dot(activation, weight);\n"
            "            }\n"
            "        }\n"
            "        for (uint row = 0u; row < uint(M); ++row) {\n"
            "            float total = simd_sum(accumulators[row]);\n"
            "            if (lane == 0u) {\n"
            "                y[row * uint(TOTAL_OUT) + uint(P" + suffix
            + "_OFFSET) + output] = T(total);\n"
            "            }\n"
            "        }\n"
            "        return;\n"
            "    }\n";
    }
    return source;
}

mlx::core::fast::CustomKernelFunction dense_projection_group_kernel(
    std::size_t projections) {
    static std::mutex mutex;
    static std::unordered_map<
        std::size_t,
        mlx::core::fast::CustomKernelFunction> kernels;
    std::lock_guard<std::mutex> lock(mutex);
    const auto found = kernels.find(projections);
    if (found != kernels.end()) {
        return found->second;
    }
    CompileOptions options;
    options.math_mode = MathMode::Fast;
    auto kernel = mlx::core::fast::metal_kernel(
        "mfq_cpp_dense_projection_group_p" +
            std::to_string(projections),
        dense_projection_group_input_names(projections),
        {"y"},
        make_dense_projection_group_source(projections),
        kDenseProjectionHeader,
        true,
        false,
        options);
    kernels.emplace(projections, kernel);
    return kernel;
}
