#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace mfq::cuda::gguf {
// Optional CPU-side diagnostic. No device reads or file I/O in route callbacks.
// The cache mutex serializes all access. Limits apply to bytes, not model shape.
class RouteTrace {
    struct Layer {
        std::uint64_t source;
        int device;
        std::int64_t rows, routes;
        std::size_t bytes;
        std::vector<std::int64_t> ids, resident;
    };
    static constexpr std::size_t limit = 64ULL << 20;
    std::ofstream file_;
    std::size_t written_ = 0, pending_bytes_ = 0;
    std::vector<Layer> pending_;
    void disable() noexcept { file_.close(); pending_.clear(); pending_bytes_ = 0; }
public:
    explicit RouteTrace(const std::string& prefix) {
        static std::atomic<std::uint64_t> serial{0};
#ifdef _WIN32
        const auto pid = _getpid();
#else
        const auto pid = getpid();
#endif
        file_.open(prefix + "-" + std::to_string(pid) + "-" + std::to_string(serial++) + ".jsonl",
                   std::ios::out | std::ios::trunc);
    }
    static std::unique_ptr<RouteTrace> from_environment() noexcept {
        const auto* prefix = std::getenv("TYLOINFER_GGUF_ROUTE_TRACE");
        if (!prefix || !*prefix) return {};
        try { return std::make_unique<RouteTrace>(prefix); } catch (...) { return {}; }
    }
    template<class Emit> void record(Emit emit) noexcept {
        if (!file_.is_open()) return;
        try {
            std::ostringstream line;
            line.imbue(std::locale::classic());
            line << std::setprecision(std::numeric_limits<float>::max_digits10);
            emit(line);
            const auto text = line.str();
            if (text.size() + 1 > limit - written_) { disable(); return; }
            file_ << text << '\n';
            file_.flush(); // Services may be terminated without destructing caches.
            written_ += text.size() + 1;
            if (!file_) disable();
        } catch (...) { disable(); }
    }
    template<class Values> static void array(std::ostream& out, const Values& values) {
        out << '[';
        bool first = true;
        for (const auto& value : values) { if (!first) out << ','; first = false; out << value; }
        out << ']';
    }
    template<class Resident> void observe(std::uint64_t source, int device, std::int64_t rows,
            std::int64_t routes, const std::int64_t* ids, std::int64_t experts,
            std::size_t bytes, bool speculative, Resident resident) noexcept {
        if (!file_.is_open()) return;
        try {
            if (rows <= 0 || routes <= 0 || experts < 0 ||
                std::uint64_t(rows) > limit / sizeof(std::int64_t) / std::uint64_t(routes) ||
                std::uint64_t(experts) > limit / sizeof(std::int64_t)) { disable(); return; }
            Layer layer{source, device, rows, routes, bytes, {ids, ids + rows * routes}, {}};
            for (std::int64_t e = 0; e < experts; ++e) if (resident(e)) layer.resident.push_back(e);
            const auto size = (layer.ids.size() + layer.resident.size()) * sizeof(std::int64_t);
            if (speculative) {
                const auto found = std::find_if(pending_.begin(), pending_.end(), [&](const auto& old) {
                    return old.source == source && old.device == device;
                });
                if (found != pending_.end()) {
                    pending_bytes_ -= (found->ids.size() + found->resident.size()) * sizeof(std::int64_t);
                    *found = std::move(layer); pending_bytes_ += size;
                    if (pending_bytes_ > limit) disable();
                    return;
                }
            }
            if (size > limit - pending_bytes_) { disable(); return; }
            pending_bytes_ += size; pending_.push_back(std::move(layer));
        } catch (...) { disable(); }
    }
    void complete(int kept, bool learned, int device, std::uint64_t clock) noexcept {
        if (pending_.empty()) return;
        record([&](auto& out) {
            out << "{\"event\":\"window\",\"device\":" << device << ",\"clock\":" << clock
                << ",\"kept\":" << kept << ",\"learned\":" << (learned ? "true" : "false") << ",\"layers\":[";
            bool first = true;
            for (const auto& layer : pending_) {
                if (!first) out << ','; first = false;
                out << "{\"source\":" << layer.source << ",\"device\":" << layer.device
                    << ",\"rows\":" << layer.rows << ",\"routes\":" << layer.routes
                    << ",\"bytes\":" << layer.bytes << ",\"ids\":";
                array(out, layer.ids); out << ",\"resident\":"; array(out, layer.resident); out << '}';
            }
            out << "]}";
        });
        pending_.clear(); pending_bytes_ = 0;
    }
};
}
