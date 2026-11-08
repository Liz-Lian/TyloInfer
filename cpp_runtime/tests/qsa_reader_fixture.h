#pragma once
#include "tyloinfer/qsa_ram_store.h"

namespace qsa_test {
// Isolate callback edits to this fake block even when it copied a backing.
inline tyloinfer::qsa::RamKvReadSource& reader(tyloinfer::qsa::RamKvBacking& backing) {
    auto source=backing.source?std::make_shared<tyloinfer::qsa::RamKvReadSource>(*backing.source):
        std::make_shared<tyloinfer::qsa::RamKvReadSource>();
    source->bytes=backing.key_bytes+backing.value_bytes;
    backing.offset=0;backing.source=source;return *source;
}
}
