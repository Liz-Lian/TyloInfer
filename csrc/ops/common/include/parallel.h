#pragma once

#include "engine/weights/tensor_parallel.h"

enum class TensorParallelAxis {
    Mirrored,
    Output,
    Input,
};

inline size_t parallel_index(bool peer_first, size_t position, size_t count) {
    return peer_first ? mfq::peer_first_parallel_launch_index(position, count) : position;
}
