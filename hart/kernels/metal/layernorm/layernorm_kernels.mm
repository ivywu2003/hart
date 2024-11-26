#include <metal_stdlib>
#include "../common/reduction_utils.metal"
#include "../common/utils.metal"

using namespace metal;

namespace hart {

// RMS Norm kernel for Metal
template<typename scalar_t, typename out_type, bool use_quant>
kernel void rms_norm_kernel(
    device const scalar_t* input [[buffer(0)]],     // [..., hidden_size]
    device const scalar_t* weight [[buffer(1)]],    // [hidden_size]
    device out_type* output [[buffer(2)]],          // [..., hidden_size]
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

// Wrapper function to launch the kernel with appropriate parameters
void rms_norm_metal(const void* commandBuffer,
                   const void* input,
                   const void* weight,
                   void* output,
                   const LayerNormParams& params) {
    
    auto* buffer = (id<MTLCommandBuffer>)commandBuffer;
    
    // Configure the kernel
    auto computeEncoder = [buffer computeCommandEncoder];
    auto pipelineState = getPipelineState("rms_norm_kernel");
    [computeEncoder setComputePipelineState:pipelineState];
    
    // Set buffers
    [computeEncoder setBuffer:input offset:0 atIndex:0];
    [computeEncoder setBuffer:weight offset:0 atIndex:1];
    [computeEncoder setBuffer:output offset:0 atIndex:2];
    [computeEncoder setBytes:&params length:sizeof(LayerNormParams) atIndex:3];
    
    // Calculate grid and threadgroup sizes
    MTLSize gridSize = MTLSizeMake(params.num_tokens, 1, 1);
    MTLSize threadgroupSize = MTLSizeMake(min(params.hidden_size, 1024u), 1, 1);
    
    // Dispatch the kernel
    [computeEncoder dispatchThreadgroups:gridSize
                  threadsPerThreadgroup:threadgroupSize];
    
    [computeEncoder endEncoding];
}

// Explicit template instantiations
template kernel void rms_norm_kernel<float, float, false>(
    device const float*, device const float*, device float*,
    constant LayerNormParams&, uint, uint, uint);

template kernel void rms_norm_kernel<float, int8_t, true>(
    device const float*, device const float*, device int8_t*,
    constant LayerNormParams&, uint, uint, uint);

template kernel void rms_norm_kernel<half, half, false>(
    device const half*, device const half*, device half*,
    constant LayerNormParams&, uint, uint, uint);

template kernel void rms_norm_kernel<half, int8_t, true>(
    device const half*, device const half*, device int8_t*,
    constant LayerNormParams&, uint, uint, uint);

} // namespace hart
