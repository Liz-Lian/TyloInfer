#pragma once

#include "engine/weights/tensor_parallel.h"


inline size_t parallel_index(bool peer_first, size_t position, size_t count) {
    return peer_first ? tyloi::peer_first_parallel_launch_index(position, count) : position;
}
