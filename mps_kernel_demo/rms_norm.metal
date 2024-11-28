#include <metal_stdlib>

using namespace metal;

inline int warpSum(int val, uint thread_id) {
    for (int offset = 1; offset < 32; offset *= 2) {
        int neighbor = simd_shuffle(val, thread_id ^ offset);
        val += neighbor;
    }
    return val;
}

template<typename T>
inline T threadgroupReduceSum(T val, threadgroup T* shared [[threadgroup(0)]], 
                            uint thread_index, uint threadgroup_size) {
    const uint lane = thread_index & 0x1f;
    const uint wid = thread_index >> 5;
    
    // First reduce within SIMD
    val = warpSum(val, thread_index);
    
    // Write to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Read from shared memory and reduce again
    val = (thread_index < (threadgroup_size / 32.0f)) ? shared[lane] : T(0.0f);
    val = warpSum(val, thread_index);
    
    return val;
}

kernel void rms_norm_kernel(device float *data [[buffer(0)]],
                            device float *b [[buffer(1)]],
                            device float *out [[buffer(2)]],
                            uint thread_id [[thread_position_in_grid]],
                            uint threads_per_group [[threads_per_threadgroup]]) {
    float val = data[thread_id];
    threadgroup float shared_mem[32];
    out[thread_id] = threadgroupReduceSum(val, shared_mem, thread_id, threads_per_group);
}

