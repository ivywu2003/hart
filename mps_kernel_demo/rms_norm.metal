#include <metal_stdlib>

using namespace metal;

inline int warpSum(int val, uint thread_id) {
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = simd_shuffle(val, thread_id ^ offset);
        val += neighbor;
    }
    return val;
}

kernel void rms_norm_kernel(device float *data [[buffer(0)]],
                            device float *b [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            uint thread_id [[thread_position_in_grid]]) {
    int val = data[thread_id];
    out[thread_id] = warpSum(val, thread_id);
}

// kernel void rms_norm_kernel(device float *data [[buffer(0)]],
//                             device float *b [[buffer(1)]],
//                             device float *out [[buffer(2)]],
//                             uint thread_id [[thread_position_in_grid]]) {
//   int val = data[thread_id];
// 
//   // Perform XOR reduction within a warp
//   for (int offset = 1; offset < 32; offset *= 2) {
//     // Broadcast value from a thread at thread_id ^ offset
//     int neighbor = simd_shuffle(val, thread_id ^ offset);
//     // XOR with the neighbor's value
//     val ^= neighbor;
//   }
// 
//   // Write the result back to the data array
//   out[thread_id] = val;
// }

