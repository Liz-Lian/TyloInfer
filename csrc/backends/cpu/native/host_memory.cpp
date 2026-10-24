#include "host_memory.h"
#include <stdexcept>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#include <unistd.h>
#include <fstream>
#endif

mfq::HostMemorySample mfq::sample_host_memory() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS process{};
    MEMORYSTATUSEX system{};
    system.dwLength = sizeof(system);
    if (!K32GetProcessMemoryInfo(GetCurrentProcess(), &process, sizeof(process)) ||
        !GlobalMemoryStatusEx(&system)) throw std::runtime_error("Cannot measure host memory");
    return {process.WorkingSetSize, system.ullAvailPhys, system.ullAvailPageFile};
#elif defined(__linux__)
    std::uint64_t virtual_pages = 0, resident_pages = 0;
    std::ifstream status("/proc/self/statm");
    struct sysinfo system{};
    if (!(status >> virtual_pages >> resident_pages) || sysinfo(&system) != 0)
        throw std::runtime_error("Cannot measure host memory");
    return {resident_pages * static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE)),
        std::uint64_t(system.freeram + system.bufferram) * system.mem_unit, std::nullopt};
#else
    throw std::runtime_error("Host memory measurement is unavailable on this platform");
#endif
}
