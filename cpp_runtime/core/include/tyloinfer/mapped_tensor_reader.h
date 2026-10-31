#pragma once
#include "model_source.h"

namespace mfq {
// Retains a bounded read-only file view independently of the model source.
// Uses the common mapped-page profiling/reclamation machinery on each OS.
ModelSource::TensorReader mapped_file_tensor_reader(const TensorFileRange& range);
}
