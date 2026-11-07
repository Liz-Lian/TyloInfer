#pragma once

#include <optional>

namespace tyloi::weights {
struct CachePolicy {
    bool mapped = false;
    int copy_blocks = 64;
    int io_workers = 8;
    bool overlap = true;
    bool pipeline = false;
    bool hybrid_cpu = false;
    bool assert_resident = false;
    bool ram_pcie = false;
    bool direct_ram = true;
    bool adapt = true;
    bool warm = true;
    bool projection_heat = true;
    std::optional<double> ram_fraction;
};
}

struct MoeLoadOptions {
    bool nvq2 = false;
    bool nvq_groups = false;
    bool heterogeneous = true;
};
