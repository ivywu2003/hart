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

at::Tensor fused_rope_with_pos_forward_func_metal(const at::Tensor &input,
                                          const at::Tensor &freqs,
                                          bool transpose_output_memory);

at::Tensor fused_rope_forward_metal(const at::Tensor &input,
                                   const at::Tensor &freqs,
                                   const bool transpose_output_memory);

at::Tensor fused_rope_backward_metal(const at::Tensor &output_grads,
                                    const at::Tensor &freqs,
                                    const bool transpose_output_memory);

} // namespace hart

static char *KERNEL = R"KERNEL(
#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;

namespace hart {

/************************************************/
/* COMMON                                       */
/************************************************/

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

/************************************************/
/* LAYERNORM                                    */
/************************************************/

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

/************************************************/
/* FUSED ROPE                                   */
/************************************************/

template<typename scalar_t>
kernel void fused_rope_with_pos_forward(device const scalar_t* src [[buffer(0)]],
                                      device const float* freqs [[buffer(1)]],
                                      device scalar_t* dst [[buffer(2)]],
                                      constant int& h [[buffer(3)]],
                                      constant int& d [[buffer(4)]],
                                      constant int& d2 [[buffer(5)]],
                                      constant int& stride_h [[buffer(6)]],
                                      constant int& stride_d [[buffer(7)]],
                                      constant int& o_stride_h [[buffer(8)]],
                                      constant int& o_stride_d [[buffer(9)]],
                                      constant int& s [[buffer(10)]],
                                      uint3 tid [[thread_position_in_threadgroup]],
                                      uint3 bid [[threadgroup_position_in_grid]],
                                      uint3 blockDim [[threads_per_threadgroup]]) {
    int s_id = bid.x;
    int b_id = bid.y;
    
    for (int d_id = tid.x; d_id < d2; d_id += blockDim.x) {
        float v_sin, v_cos;
        float freq = freqs[(b_id * s + s_id) * d2 + d_id];
        v_sin = sin(freq);
        v_cos = cos(freq);
        
        for (int h_id = tid.y; h_id < h; h_id += blockDim.y) {
            int offset_src = h_id * stride_h + d_id * stride_d;
            int offset_dst = h_id * o_stride_h + d_id * o_stride_d;
            float v_src = src[offset_src];
            float v_src_rotate = (d_id + d2/2 < d2) ? 
                -src[offset_src + (d2/2) * stride_d] :
                src[offset_src + (d2/2 - d2) * stride_d];
            
            dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
        }
    }

    // TODO: check if the d > d2 part is necessary
}

// Explicit template instantiations
template 
[[host_name("fused_rope_with_pos_forward_float")]]
kernel void fused_rope_with_pos_forward<float>(device const float* src [[buffer(0)]],
                                      device const float* freqs [[buffer(1)]],
                                      device float* dst [[buffer(2)]],
                                      constant int& h [[buffer(3)]],
                                      constant int& d [[buffer(4)]],
                                      constant int& d2 [[buffer(5)]],
                                      constant int& stride_h [[buffer(6)]],
                                      constant int& stride_d [[buffer(7)]],
                                      constant int& o_stride_h [[buffer(8)]],
                                      constant int& o_stride_d [[buffer(9)]],
                                      constant int& s [[buffer(10)]],
                                      uint3 tid [[thread_position_in_threadgroup]],
                                      uint3 bid [[threadgroup_position_in_grid]],
                                      uint3 blockDim [[threads_per_threadgroup]]);

// Explicit template instantiations
template 
[[host_name("fused_rope_with_pos_forward_half")]]
kernel void fused_rope_with_pos_forward<half>(device const half* src [[buffer(0)]],
                                      device const float* freqs [[buffer(1)]],
                                      device half* dst [[buffer(2)]],
                                      constant int& h [[buffer(3)]],
                                      constant int& d [[buffer(4)]],
                                      constant int& d2 [[buffer(5)]],
                                      constant int& stride_h [[buffer(6)]],
                                      constant int& stride_d [[buffer(7)]],
                                      constant int& o_stride_h [[buffer(8)]],
                                      constant int& o_stride_d [[buffer(9)]],
                                      constant int& s [[buffer(10)]],
                                      uint3 tid [[thread_position_in_threadgroup]],
                                      uint3 bid [[threadgroup_position_in_grid]],
                                      uint3 blockDim [[threads_per_threadgroup]]);

template<typename scalar_t>
kernel void fused_rope_forward(device const scalar_t* src [[buffer(0)]],
                                      device const float* freqs [[buffer(1)]],
                                      device scalar_t* dst [[buffer(2)]],
                                      constant int& h [[buffer(3)]],
                                      constant int& d [[buffer(4)]],
                                      constant int& d2 [[buffer(5)]],
                                      constant int& stride_h [[buffer(6)]],
                                      constant int& stride_d [[buffer(7)]],
                                      constant int& o_stride_h [[buffer(8)]],
                                      constant int& o_stride_d [[buffer(9)]],
                                      constant int& s [[buffer(10)]],
                                      uint3 tid [[thread_position_in_threadgroup]],
                                      uint3 bid [[threadgroup_position_in_grid]],
                                      uint3 blockDim [[threads_per_threadgroup]]) {
    int s_id = bid.x;
    int b_id = bid.y;
    
    for (int d_id = tid.x; d_id < d2; d_id += blockDim.x) {
        float v_sin, v_cos;
        float freq = freqs[(b_id * s + s_id) * d2 + d_id];
        v_sin = sin(freq);
        v_cos = cos(freq);
        
        for (int h_id = tid.y; h_id < h; h_id += blockDim.y) {
            int offset_src = h_id * stride_h + d_id * stride_d;
            int offset_dst = h_id * o_stride_h + d_id * o_stride_d;
            float v_src = src[offset_src];
            float v_src_rotate = (d_id + d2/2 < d2) ? 
                -src[offset_src + (d2/2) * stride_d] :
                src[offset_src + (d2/2 - d2) * stride_d];
            
            dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
        }
    }

    // TODO: check if the d > d2 part is necessary
}

// Explicit template instantiations
template 
[[host_name("fused_rope_forward_float")]]
kernel void fused_rope_forward<float>(device const float* src [[buffer(0)]],
                                      device const float* freqs [[buffer(1)]],
                                      device float* dst [[buffer(2)]],
                                      constant int& h [[buffer(3)]],
                                      constant int& d [[buffer(4)]],
                                      constant int& d2 [[buffer(5)]],
                                      constant int& stride_h [[buffer(6)]],
                                      constant int& stride_d [[buffer(7)]],
                                      constant int& o_stride_h [[buffer(8)]],
                                      constant int& o_stride_d [[buffer(9)]],
                                      constant int& s [[buffer(10)]],
                                      uint3 tid [[thread_position_in_threadgroup]],
                                      uint3 bid [[threadgroup_position_in_grid]],
                                      uint3 blockDim [[threads_per_threadgroup]]);

// Explicit template instantiations
template 
[[host_name("fused_rope_forward_half")]]
kernel void fused_rope_forward<half>(device const half* src [[buffer(0)]],
                                      device const float* freqs [[buffer(1)]],
                                      device half* dst [[buffer(2)]],
                                      constant int& h [[buffer(3)]],
                                      constant int& d [[buffer(4)]],
                                      constant int& d2 [[buffer(5)]],
                                      constant int& stride_h [[buffer(6)]],
                                      constant int& stride_d [[buffer(7)]],
                                      constant int& o_stride_h [[buffer(8)]],
                                      constant int& o_stride_d [[buffer(9)]],
                                      constant int& s [[buffer(10)]],
                                      uint3 tid [[thread_position_in_threadgroup]],
                                      uint3 bid [[threadgroup_position_in_grid]],
                                      uint3 blockDim [[threads_per_threadgroup]]);

template<typename scalar_t>
kernel void fused_rope_backward(device const scalar_t* src [[buffer(0)]],
                              device const float* freqs [[buffer(1)]],
                              device scalar_t* dst [[buffer(2)]],
                              constant int& h [[buffer(3)]],
                              constant int& d [[buffer(4)]],
                              constant int& d2 [[buffer(5)]],
                              constant int& stride_h [[buffer(6)]],
                              constant int& stride_d [[buffer(7)]],
                              constant int& o_stride_h [[buffer(8)]],
                              constant int& o_stride_d [[buffer(9)]],
                              uint3 tid [[thread_position_in_threadgroup]],
                              uint3 bid [[threadgroup_position_in_grid]],
                              uint3 blockDim [[threads_per_threadgroup]]) {
    int s_id = bid.x;
    
    for (int d_id = tid.x; d_id < d2; d_id += blockDim.x) {
        float v_sin, v_cos;
        float freq = freqs[s_id * d2 + d_id];
        v_sin = sin(freq);
        v_cos = cos(freq);
        
        for (int h_id = tid.y; h_id < h; h_id += blockDim.y) {
            int offset_src = h_id * stride_h + d_id * stride_d;
            int offset_dst = h_id * o_stride_h + d_id * o_stride_d;
            float v_src = src[offset_src];
            float v_src_rotate = (d_id + d2/2 < d2) ? 
                -src[offset_src + (d2/2) * stride_d] :
                src[offset_src + (d2/2 - d2) * stride_d];
            
            dst[offset_dst] = v_src * v_cos - v_src_rotate * v_sin;
        }
    }
}

template 
[[host_name("fused_rope_backward_float")]]
kernel void fused_rope_backward<float>(device const float* src [[buffer(0)]],
                              device const float* freqs [[buffer(1)]],
                              device float* dst [[buffer(2)]],
                              constant int& h [[buffer(3)]],
                              constant int& d [[buffer(4)]],
                              constant int& d2 [[buffer(5)]],
                              constant int& stride_h [[buffer(6)]],
                              constant int& stride_d [[buffer(7)]],
                              constant int& o_stride_h [[buffer(8)]],
                              constant int& o_stride_d [[buffer(9)]],
                              uint3 tid [[thread_position_in_threadgroup]],
                              uint3 bid [[threadgroup_position_in_grid]],
                              uint3 blockDim [[threads_per_threadgroup]]);

template 
[[host_name("fused_rope_backward_half")]]
kernel void fused_rope_backward<half>(device const half* src [[buffer(0)]],
                              device const float* freqs [[buffer(1)]],
                              device half* dst [[buffer(2)]],
                              constant int& h [[buffer(3)]],
                              constant int& d [[buffer(4)]],
                              constant int& d2 [[buffer(5)]],
                              constant int& stride_h [[buffer(6)]],
                              constant int& stride_d [[buffer(7)]],
                              constant int& o_stride_h [[buffer(8)]],
                              constant int& o_stride_d [[buffer(9)]],
                              uint3 tid [[thread_position_in_threadgroup]],
                              uint3 bid [[threadgroup_position_in_grid]],
                              uint3 blockDim [[threads_per_threadgroup]]);

}
)KERNEL";
