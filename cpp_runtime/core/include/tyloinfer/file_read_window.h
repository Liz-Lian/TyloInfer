#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>

namespace tyloinfer {

// Opt-in read-ahead for sequential recovery of immutable files. One bounded
// window serves the current thread, including nested scopes and all files.
// Random SSD reads outside recovery retain their exact requested ranges.
class ScopedFileReadWindow {
public:
    static constexpr std::size_t capacity=64*1024;
    explicit ScopedFileReadWindow(bool read_ahead=true)
        :previous_(active_),previous_read_ahead_(active_ && active_->read_ahead) {
        if(!active_ && read_ahead) {
            owned_=std::make_unique<State>();active_=owned_.get();
        }
        if(active_)active_->read_ahead=read_ahead;
    }
    ~ScopedFileReadWindow() {
        active_=previous_;
        if(active_)active_->read_ahead=previous_read_ahead_;
    }
    ScopedFileReadWindow(const ScopedFileReadWindow&)=delete;
    ScopedFileReadWindow& operator=(const ScopedFileReadWindow&)=delete;

    static bool read_ahead_enabled() noexcept {return active_ && active_->read_ahead;}

    // A source ID must identify an immutable open file version, never just its
    // pathname or an address which could be reused. The reader fills exactly
    // the requested range or throws; a failed fill invalidates the window.
    template<class Read>
    static bool read(std::uint64_t source,std::uint64_t file_bytes,
                     std::uint64_t offset,void* destination,std::size_t bytes,Read&& uncached) {
        if(offset>file_bytes || bytes>file_bytes-offset)
            throw std::out_of_range("file read window byte range");
        if(!bytes)return true;
        auto* state=active_;
        if(!state || bytes>=capacity)return false;
        if(state->valid && state->source==source && offset>=state->offset) {
            const auto relative=offset-state->offset;
            if(relative<=state->valid && bytes<=state->valid-relative) {
                std::memcpy(destination,state->data.data()+std::size_t(relative),bytes);return true;
            }
        }
        if(!state->read_ahead)return false;
        state->valid=0;
        const auto count=std::size_t(std::min<std::uint64_t>(capacity,file_bytes-offset));
        uncached(offset,state->data.data(),count);
        state->source=source;state->offset=offset;state->valid=count;
        std::memcpy(destination,state->data.data(),bytes);return true;
    }
private:
    struct State {
        std::array<std::uint8_t,capacity> data;
        std::uint64_t source=0,offset=0;
        std::size_t valid=0;
        bool read_ahead=true;
    };
    inline static thread_local State* active_=nullptr;
    State* previous_;
    bool previous_read_ahead_;
    std::unique_ptr<State> owned_;
};

} // namespace tyloinfer
