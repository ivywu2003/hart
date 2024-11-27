#pragma once

#include <torch/extension.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

namespace hart {

struct LayerNormParams {
    uint num_tokens;
    uint hidden_size;
    float epsilon;
    bool use_quant;
};

// Forward declarations for Metal kernel functions
void rms_norm_metal(torch::Tensor& output,           // [..., hidden_size]
                   const torch::Tensor& input,      // [..., hidden_size]
                   const torch::Tensor& weight,     // [hidden_size]
                   const LayerNormParams& params);

} // namespace hart

static char *LAYERNORM_KERNEL = R"LAYERNORM(
#include <metal_stdlib>

using namespace metal;

namespace hart {

// RMS Norm kernel for Metal
template<typename scalar_t, typename out_type, bool use_quant>
kernel void rms_norm_kernel(
    device out_type* output [[buffer(0)]],          // [..., hidden_size]
    device const scalar_t* input [[buffer(1)]],     // [..., hidden_size]
    device const scalar_t* weight [[buffer(2)]],    // [hidden_size]
    constant LayerNormParams& params [[buffer(3)]],
    uint token_idx [[thread_position_in_grid]],
    uint thread_idx [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]) {
    
    // Shared memory for variance reduction
    threadgroup float shared_mem[32];
    
    const uint hidden_size = params.hidden_size;
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
        norm_factor = rsqrt(variance / float(hidden_size) + params.epsilon);
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
    device const float*, device const float*, device float*,
    constant LayerNormParams&, uint, uint, uint);

template 
[[host_name("rms_norm_kernel_float_int8_true")]]
kernel void rms_norm_kernel<float, int8_t, true>(
    device const float*, device const float*, device int8_t*,
    constant LayerNormParams&, uint, uint, uint);

template 
[[host_name("rms_norm_kernel_half_half_false")]]
kernel void rms_norm_kernel<half, half, false>(
    device const half*, device const half*, device half*,
    constant LayerNormParams&, uint, uint, uint);

template 
[[host_name("rms_norm_kernel_half_int8_true")]]
kernel void rms_norm_kernel<half, int8_t, true>(
    device const half*, device const half*, device int8_t*,
    constant LayerNormParams&, uint, uint, uint);

}
)LAYERNORM";
