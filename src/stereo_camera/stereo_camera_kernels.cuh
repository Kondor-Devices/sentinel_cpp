#pragma once
#include <cstddef>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Host function that enqueues the kernel on 'stream'
void launch_bgra_to_rgb_fp16_nchw(const unsigned char* src_base,
                                  size_t src_pitch,
                                  int W, int H,
                                  __half* dst,
                                  cudaStream_t stream);
