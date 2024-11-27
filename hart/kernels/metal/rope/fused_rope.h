// Consolidated version of rope_with_pos and rope kernels
#pragma once

#include <torch/extension.h>

namespace hart {

at::Tensor fused_rope_with_pos_forward_metal(const at::Tensor &input,
                                          const at::Tensor &freqs,
                                          bool transpose_output_memory);

at::Tensor fused_rope_forward_metal(const at::Tensor &input,
                                   const at::Tensor &freqs,
                                   const bool transpose_output_memory);

at::Tensor fused_rope_backward_metal(const at::Tensor &output_grads,
                                    const at::Tensor &freqs,
                                    const bool transpose_output_memory);

} // namespace hart


static char *ROPE_KERNEL = R"ROPE(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

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
)ROPE";