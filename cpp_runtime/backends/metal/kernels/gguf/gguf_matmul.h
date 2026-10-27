template<int F, int BM, typename T, typename InputRow, typename OutputRow>
inline void gguf_matmul(device const uchar* matrix, device const T* x, device T* y,
    uint K, uint N, uint stride, uint rows, uint m0, uint n0, InputRow input_row, OutputRow output_row) {
    const short2 coordinate = mlx::steel::BaseNAXFrag::get_coord();
    const bool upper = (__metal_get_thread_index_in_simdgroup(ushort()) & 1u) != 0u;
    mlx::steel::NAXTile<float, BM / 16, 2> accumulator;
    accumulator.clear();
    for (uint k0 = 0u; k0 < K; k0 += 16u) {
        mlx::steel::NAXTile<half, BM / 16, 1> activation;
        mlx::steel::NAXTile<half, 2, 1> weight;
        for (uint tile = 0u; tile < BM / 16; ++tile) for (uint r = 0u; r < 2u; ++r) {
            const uint row = m0 + tile * 16u + uint(coordinate.y) + r * 8u;
            const uint k = k0 + uint(coordinate.x);
            for (uint j = 0u; j < 4u; ++j)
                activation.frag_at(tile, 0)[r * 4u + j] = row < rows && k + j < K
                    ? half(x[ulong(input_row(row)) * K + k + j]) : half(0);
        }
        for (uint block = 0u; block < 2u; ++block) {
            const uint row = n0 + block * 16u + uint(coordinate.y) + (upper ? 8u : 0u);
            const uint k = k0 + (uint(coordinate.x) & ~7u);
            half4 first(0.h), second(0.h);
            if (row < N) {
                auto source = matrix + ulong(row) * stride;
                first = half4(gguf_values4<F>(source, k));
                second = half4(gguf_values4<F>(source, k + 4u));
            }
            const half4 peer_first = simd_shuffle_xor(first, 1u);
            const half4 peer_second = simd_shuffle_xor(second, 1u);
            const half4 low = upper ? peer_second : first;
            const half4 high = upper ? second : peer_first;
            for (uint j = 0u; j < 4u; ++j) {
                weight.frag_at(block, 0)[j] = low[j];
                weight.frag_at(block, 0)[4u + j] = high[j];
            }
        }
        mlx::steel::tile_matmad_nax(accumulator, activation, metal::bool_constant<false>{},
            weight, metal::bool_constant<true>{});
    }
    for (uint tile = 0u; tile < BM / 16; ++tile) for (uint block = 0u; block < 2u; ++block) {
        for (uint r = 0u; r < 2u; ++r) {
            const uint row = m0 + tile * 16u + uint(coordinate.y) + r * 8u;
            const uint col = n0 + block * 16u + uint(coordinate.x);
            if (row < rows) {
                const auto offset = ulong(output_row(row)) * N;
                for (uint j = 0u; j < 4u; ++j)
                    if (col + j < N) y[offset + col + j] = T(accumulator.frag_at(tile, block)[r * 4u + j]);
            }
        }
    }
}
