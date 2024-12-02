#include <metal_stdlib>

using namespace metal;

kernel void rms_norm_kernel(device float *output [[buffer(0)]],
                            constant float *input [[buffer(1)]],
                            constant float *weight [[buffer(2)]],
                            constant uint &hidden_size [[buffer(3)]],
                            constant float &epsilon [[buffer(4)]],
                            uint token_id [[thread_position_in_grid]],
                            uint thread_id [[thread_position_in_threadgroup]],
                            uint threadgroup_id [[threadgroup_position_in_grid]],
                            uint threads_per_group [[threads_per_threadgroup]]) {
  float variance = 0.0f;

  for (uint i = 0; i < hidden_size; i += 1) {
    float x = float(input[threadgroup_id * hidden_size + i]);
    variance += x * x;
  }

  float norm_factor = rsqrt(variance / float(hidden_size) + epsilon);

  for (uint i = thread_id; i < hidden_size; i += threads_per_group) {
    float x = float(input[threadgroup_id * hidden_size + i]);
    output[threadgroup_id * hidden_size + i] = x * norm_factor * weight[i];
  }

}

