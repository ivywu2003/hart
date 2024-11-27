#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

kernel void fused_rope_with_pos_forward(device const float* src [[buffer(0)]],
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
}
