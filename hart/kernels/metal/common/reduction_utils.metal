#include <metal_stdlib>

using namespace metal;

namespace hart {

// Metal equivalent of warpReduceSum
template<typename T>
inline T simdReduceSum(T val, uint simd_size = 32) {
    for (uint offset = simd_size/2; offset > 0; offset /= 2) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}

// Metal equivalent of warpReduceSumV2
template<typename T, int NUM>
inline T simdReduceSumV2(thread T* val) {
    for (int i = 0; i < NUM; i++) {
        for (uint offset = 16; offset > 0; offset /= 2) {
            val[i] += simd_shuffle_xor(val[i], offset);
        }
    }
    return T(0.0f);
}

// Metal equivalent of blockReduceSum
template<typename T>
inline T threadgroupReduceSum(T val, threadgroup T* shared [[threadgroup(0)]], 
                            uint thread_index, uint threadgroup_size) {
    const uint lane = thread_index & 0x1f;
    const uint wid = thread_index >> 5;
    
    // First reduce within SIMD
    val = simdReduceSum(val);
    
    // Write to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Read from shared memory and reduce again
    val = (thread_index < (threadgroup_size / 32.0f)) ? shared[lane] : T(0.0f);
    val = simdReduceSum(val);
    
    return val;
}

// Metal equivalent of blockAllReduceSum
template<typename T>
inline T threadgroupAllReduceSum(T val, threadgroup T* shared [[threadgroup(0)]], 
                               uint thread_index, uint threadgroup_size) {
    const uint lane = thread_index & 0x1f;
    const uint wid = thread_index >> 5;
    
    val = simdReduceSum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    val = (lane < (threadgroup_size / 32.0f)) ? shared[lane] : T(0.0f);
    val = simdReduceSum(val);
    
    return val;
}

// Metal equivalent of blockReduceSumV2
template<typename T, int NUM>
inline T threadgroupReduceSumV2(thread T* val, threadgroup T shared[][33] [[threadgroup(0)]], 
                               uint thread_index, uint threadgroup_size) {
    const uint lane = thread_index & 0x1f;
    const uint wid = thread_index >> 5;
    
    simdReduceSumV2<T, NUM>(val);
    
    if (lane == 0) {
        for (int i = 0; i < NUM; i++) {
            shared[i][wid] = val[i];
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    bool is_mask = thread_index < (threadgroup_size / 32.0f);
    for (int i = 0; i < NUM; i++) {
        val[i] = is_mask ? shared[i][lane] : T(0.0f);
    }
    
    simdReduceSumV2<T, NUM>(val);
    return T(0.0f);
}

} // namespace hart
