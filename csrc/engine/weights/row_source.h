#pragma once

#include <memory>
#include <string>

namespace tyloi { class NintRows; namespace source { class Source; } }

namespace tyloi::weights {

/** Open a packed NINT row source and retain its reader and accounting. */
std::shared_ptr<NintRows> load_nint_rows(
    const source::Source& source, const std::string& name, bool prefer_mapping = false);

} // namespace tyloi::weights
