#include "metal_device_info.h"

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

namespace mfq::metal {
int metal_apple_gpu_core_count() noexcept {
    static const int cores = [] {
        const auto service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
        if (!service) return 0;
        const auto value = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
        IOObjectRelease(service);
        if (!value) return 0;
        int count = 0;
        if (CFGetTypeID(value) == CFNumberGetTypeID())
            CFNumberGetValue(static_cast<CFNumberRef>(value), kCFNumberIntType, &count);
        CFRelease(value);
        return count > 0 && count <= 1024 ? count : 0;
    }();
    return cores;
}
}
