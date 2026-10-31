#pragma once
#include <cstddef>
#include <cstdint>

namespace mfq::cpu::gguf {
struct Projection {
    const std::uint8_t* data=nullptr;
    std::size_t row_bytes=0;
    int type=0,rows=0,columns=0;
};
// The row implementation uses AVX2, FMA and F16C. The OS must preserve XMM/YMM.
bool available() noexcept;
bool avx512_available() noexcept;
bool input_supported(int type) noexcept;
bool down_supported(int type) noexcept;
std::size_t input_bytes(int columns);
std::size_t hidden_bytes(int columns,int type);
void quantize_input(const float* input,void* packed,int columns);
void quantize_hidden(const float* input,void* packed,int columns,int type);
struct ProjectionDispatch {bool avx512=false,trace=false,vnni=false;};
// Resolve configuration once before a parallel row job. Its tiles must not
// repeatedly consult process-wide environment state or update trace flags.
ProjectionDispatch projection_dispatch();
void project(const Projection&,const void* const* input,int tokens,float* const* output,int begin,int end);
void project(const Projection&,const void* const* input,int tokens,float* const* output,int begin,int end,ProjectionDispatch);
}
