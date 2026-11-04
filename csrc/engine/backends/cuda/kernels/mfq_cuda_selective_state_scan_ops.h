#pragma once

#include "backend.h"

#include <vector>

// Selective state scan (SSD recurrence) shared by linear-attention mixers.
// See engine/backends/cuda/kernels/selective_state_scan.cu for the recurrence and for the
// dtype contract: the recurrent state, parameters and outputs are float32,
// while x/dt/B/C may be f32, f16 or bf16 (all four sharing one dtype).

// One decode step, state[B,H,P,N] updated in place; returns y[B,H,P] f32.
// dt[B,H] is the raw (pre-softplus) time step, dt_bias[H]/a_log[H]/D[H] are f32
// parameters and B/C[B,G,N] are the grouped projections (head h uses group
// h / (H/G)). dt_limit_lo/dt_limit_hi clamp softplus(dt + dt_bias).
::tyloi::Tensor selective_state_scan_decode_cuda(
    ::tyloi::Tensor state,
    ::tyloi::Tensor x,
    ::tyloi::Tensor dt,
    ::tyloi::Tensor dt_bias,
    ::tyloi::Tensor a_log,
    ::tyloi::Tensor b,
    ::tyloi::Tensor c,
    ::tyloi::Tensor d,
    double dt_limit_lo,
    double dt_limit_hi);

// Any-length prefill of the same recurrence: x[B,T,H,P], dt[B,T,H],
// B/C[B,T,G,N] and an optional initial_state[B,H,P,N]. Returns
// {y[B,T,H,P], final_state[B,H,P,N]}, both f32; the result equals T decode
// calls up to floating point rounding.
std::vector<::tyloi::Tensor> selective_state_scan_prefill_cuda(
    ::tyloi::Tensor x,
    ::tyloi::Tensor dt,
    ::tyloi::Tensor dt_bias,
    ::tyloi::Tensor a_log,
    ::tyloi::Tensor b,
    ::tyloi::Tensor c,
    ::tyloi::Tensor d,
    ::tyloi::Tensor initial_state,
    double dt_limit_lo,
    double dt_limit_hi);
