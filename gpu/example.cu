#include <cuda_runtime.h>
#include "nvcomp/lz4.h"

extern "C" void cuda_example() {
  void *ptr;
  cudaMalloc(&ptr, 4096);
  size_t temp_bytes;
  nvcompBatchedLZ4CompressGetTempSize(1, 4096,
                                      nvcompBatchedLZ4DefaultOpts, &temp_bytes);
}

