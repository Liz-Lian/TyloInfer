#pragma once

namespace tyloi::cuda {

struct FfnOptions {
    bool small_heterogeneous = true;
    bool delayed_readback = true;
    bool bundle_prefetch = true;
    bool swiglu_fusion = true;
    bool reduce_fusion = true;
};

struct FfnMode {
    bool materialized = false;
    bool unfused = false;
    bool serial = false;
    bool batch_serial = false;
    int tile = 0;
    int tiles = 1;
    bool reuse_input = true;
    bool projection_major = false;
    bool independent = false;
};

} // namespace tyloi::cuda
