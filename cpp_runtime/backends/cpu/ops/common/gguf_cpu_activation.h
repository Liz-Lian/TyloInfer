#pragma once
#include <cstddef>

namespace mfq::cpu::gguf {
// Preserve the expert's FP16 intermediate rounding, including the different
// unclipped and clipped SwiGLU expressions. Output may equal either input.
void swiglu(const float* gate,const float* up,float* output,std::size_t count,bool half,float limit);
}
