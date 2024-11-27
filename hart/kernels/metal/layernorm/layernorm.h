#pragma once

#include <torch/extension.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace hart {

// Forward declarations for Metal kernel functions
void rms_norm_metal(torch::Tensor& output,           // [..., hidden_size]
                   const torch::Tensor& input,      // [..., hidden_size]
                   const torch::Tensor& weight,     // [hidden_size]
                   uint32_t num_tokens,
                   uint32_t hidden_size,
                   float epsilon,
                   bool use_quant);

} // namespace hart

static char *LAYERNORM_KERNEL = R"LAYERNORM(
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

// RMS Norm kernel for Metal
template<typename scalar_t, typename out_type, bool use_quant>
kernel void rms_norm_kernel(
    device      out_type*   output      [[buffer(0)]],        // [..., hidden_size]
    constant    scalar_t*   input       [[buffer(1)]],        // [..., hidden_size]
    constant    scalar_t*   weight      [[buffer(2)]],        // [hidden_size]
    constant    uint&       hidden_size [[buffer(3)]],
    constant    float&      epsilon     [[buffer(4)]],
    uint token_idx [[thread_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    
    // Shared memory for variance reduction
    threadgroup float shared_mem[32];
    
    float variance = 0.0f;
    
    // Calculate variance
    for (uint idx = thread_idx; idx < hidden_size; idx += threads_per_group) {
        const float x = float(input[token_idx * hidden_size + idx]);
        variance += x * x;
    }
    
    // Reduce variance across threadgroup
    variance = threadgroupReduceSum(variance, shared_mem, thread_idx, threads_per_group);
    
    // Calculate normalization factor
    float norm_factor = 0.0f;
    if (thread_idx == 0) {
        norm_factor = rsqrt(variance / float(hidden_size) + epsilon);
        shared_mem[0] = norm_factor;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    norm_factor = shared_mem[0];
    
    // Apply normalization and weight
    for (uint idx = thread_idx; idx < hidden_size; idx += threads_per_group) {
        float x = float(input[token_idx * hidden_size + idx]);
        if (use_quant) {
            // Convert to int8 with rounding
            float normalized = x * norm_factor * float(weight[idx]);
            output[token_idx * hidden_size + idx] = 
                out_type(clamp(round(normalized * 127.0f), -128.0f, 127.0f));
        } else {
            output[token_idx * hidden_size + idx] = 
                scalar_t(x * norm_factor) * weight[idx];
        }
    }
}

// Explicit template instantiations
template 
[[host_name("rms_norm_kernel_float_float_false")]]
kernel void rms_norm_kernel<float, float, false>(
    device      float*   output      [[buffer(0)]], 
    constant    float*   input       [[buffer(1)]], 
    constant    float*   weight      [[buffer(2)]], 
    constant    uint&    hidden_size [[buffer(3)]],
    constant    float&   epsilon     [[buffer(4)]],
    uint token_idx [[thread_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);

template 
[[host_name("rms_norm_kernel_float_char_true")]]
kernel void rms_norm_kernel<float, char, true>(
    device      char*    output      [[buffer(0)]], 
    constant    float*   input       [[buffer(1)]], 
    constant    float*   weight      [[buffer(2)]], 
    constant    uint&    hidden_size [[buffer(3)]],
    constant    float&   epsilon     [[buffer(4)]],
    uint token_idx [[thread_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);

template 
[[host_name("rms_norm_kernel_half_half_false")]]
kernel void rms_norm_kernel<half, half, false>(
    device      half*    output      [[buffer(0)]], 
    constant    half*    input       [[buffer(1)]], 
    constant    half*    weight      [[buffer(2)]], 
    constant    uint&    hidden_size [[buffer(3)]],
    constant    float&   epsilon     [[buffer(4)]],
    uint token_idx [[thread_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);

template 
[[host_name("rms_norm_kernel_half_char_true")]]
kernel void rms_norm_kernel<half, char, true>(
    device      char*    output      [[buffer(0)]], 
    constant    half*    input       [[buffer(1)]], 
    constant    half*    weight      [[buffer(2)]], 
    constant    uint&    hidden_size [[buffer(3)]],
    constant    float&   epsilon     [[buffer(4)]],
    uint token_idx [[thread_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]);

}
)LAYERNORM";
