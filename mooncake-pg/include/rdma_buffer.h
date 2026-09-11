#pragma once

#include <cstddef>
#include <cuda_alike.h>

namespace mooncake {

// Communication buffers must be directly accessible by the RNIC. MACA's
// finegrained allocation is GPU-backed and can be released with cudaFree.
inline cudaError_t allocateRdmaBuffer(void** pointer, size_t bytes) {
#ifdef USE_MACA
    return mcExtMallocWithFlags(pointer, bytes, mcDeviceMallocFinegrained);
#else
    return cudaMalloc(pointer, bytes);
#endif
}

}  // namespace mooncake
