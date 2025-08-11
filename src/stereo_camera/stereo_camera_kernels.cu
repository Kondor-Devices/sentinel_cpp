#include "stereo_camera_kernels.cuh"

static __global__ void bgra_to_rgb_fp16_nchw_kernel(
    const unsigned char* __restrict__ src_base,
    size_t src_pitch,
    int W, int H,
    __half* __restrict__ dst
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    const unsigned char* p = src_base + y * src_pitch + 4 * x; // BGRA
    float b = p[0] * (1.0f/255.0f);
    float g = p[1] * (1.0f/255.0f);
    float r = p[2] * (1.0f/255.0f);

    int idx = y * W + x;
    dst[idx]           = __float2half(r);
    dst[H*W + idx]     = __float2half(g);
    dst[2*H*W + idx]   = __float2half(b);
}

void launch_bgra_to_rgb_fp16_nchw(const unsigned char* src_base,
                                  size_t src_pitch,
                                  int W, int H,
                                  __half* dst,
                                  cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);
    bgra_to_rgb_fp16_nchw_kernel<<<grid, block, 0, stream>>>(src_base, src_pitch, W, H, dst);
}
