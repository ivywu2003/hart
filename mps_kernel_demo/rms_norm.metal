#include <metal_stdlib>

using namespace metal;

kernel void rms_norm_kernel(device float *a [[buffer(0)]],
                            device float *b [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            uint index [[thread_position_in_grid]]) {
  out[index] = a[index] + b[index];
}

