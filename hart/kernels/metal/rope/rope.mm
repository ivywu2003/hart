#include "rope.h"
#include "../common/metal_utils.h"
#include "../common/dispatch_utils.h"

#include <Metal/Metal.h>
#include <torch/extension.h>

namespace {
    void nvte_fused_rope_forward_metal(const at::Tensor input, const at::Tensor freqs,
                                   at::Tensor output, const int s, const int b,
                                   const int h, const int d, const int d2,
                                   const int stride_s, const int stride_b,
                                   const int stride_h, const int stride_d,
                                   const int o_stride_s, const int o_stride_b,
                                   const int o_stride_h, const int o_stride_d) {
        auto device = getMetalDevice();
        auto commandQueue = device.newCommandQueue();
        auto library = getMetalLibrary(device);
        
        auto pipelineState = library.newComputePipelineState(withFunction: "fused_rope_forward");
        auto commandBuffer = commandQueue.commandBuffer();
        auto computeEncoder = commandBuffer.computeCommandEncoder();
        
        // Set buffers and parameters
        computeEncoder.setBuffer(input.data_ptr<float>(), offset: 0, index: 0);
        computeEncoder.setBuffer(freqs.data_ptr<float>(), offset: 0, index: 1);
        computeEncoder.setBuffer(output.data_ptr<float>(), offset: 0, index: 2);
        computeEncoder.setBytes(&h, length: sizeof(int), index: 3);
        computeEncoder.setBytes(&d, length: sizeof(int), index: 4);
        computeEncoder.setBytes(&d2, length: sizeof(int), index: 5);
        computeEncoder.setBytes(&stride_h, length: sizeof(int), index: 6);
        computeEncoder.setBytes(&stride_d, length: sizeof(int), index: 7);
        computeEncoder.setBytes(&o_stride_h, length: sizeof(int), index: 8);
        computeEncoder.setBytes(&o_stride_d, length: sizeof(int), index: 9);
        
        // Set grid and threadgroup size
        MTLSize gridSize = MTLSizeMake(s * b, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 32, 1);
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize);
        
        computeEncoder.endEncoding();
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
    }
    
    void nvte_fused_rope_backward_metal(const at::Tensor output_grads,
                                    const at::Tensor freqs, at::Tensor input_grads,
                                    const int s, const int b, const int h,
                                    const int d, const int d2, const int stride_s,
                                    const int stride_b, const int stride_h,
                                    const int stride_d, const int o_stride_s,
                                    const int o_stride_b, const int o_stride_h,
                                    const int o_stride_d) {
        auto device = getMetalDevice();
        auto commandQueue = device.newCommandQueue();
        auto library = getMetalLibrary(device);
        
        auto pipelineState = library.newComputePipelineState(withFunction: "fused_rope_backward");
        auto commandBuffer = commandQueue.commandBuffer();
        auto computeEncoder = commandBuffer.computeCommandEncoder();
        
        // Set buffers and parameters
        computeEncoder.setBuffer(output_grads.data_ptr<float>(), offset: 0, index: 0);
        computeEncoder.setBuffer(freqs.data_ptr<float>(), offset: 0, index: 1);
        computeEncoder.setBuffer(input_grads.data_ptr<float>(), offset: 0, index: 2);
        computeEncoder.setBytes(&h, length: sizeof(int), index: 3);
        computeEncoder.setBytes(&d, length: sizeof(int), index: 4);
        computeEncoder.setBytes(&d2, length: sizeof(int), index: 5);
        computeEncoder.setBytes(&stride_h, length: sizeof(int), index: 6);
        computeEncoder.setBytes(&stride_d, length: sizeof(int), index: 7);
        computeEncoder.setBytes(&o_stride_h, length: sizeof(int), index: 8);
        computeEncoder.setBytes(&o_stride_d, length: sizeof(int), index: 9);
        
        // Set grid and threadgroup size
        MTLSize gridSize = MTLSizeMake(s * b, 1, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 32, 1);
        computeEncoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize);
        
        computeEncoder.endEncoding();
        commandBuffer.commit();
        commandBuffer.waitUntilCompleted();
    }
}

at::Tensor fused_rope_forward_metal(const at::Tensor &input,
                                const at::Tensor &freqs,
                                const bool transpose_output_memory) {
    TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(freqs.dim() == 2, "expected 2D tensor");
    
    const int s = input.size(0);
    const int b = input.size(1);
    const int h = input.size(2);
    const int d = input.size(3);
    const int d2 = d / 2;
    
    auto output = at::empty_like(input);
    
    const int stride_s = input.stride(0);
    const int stride_b = input.stride(1);
    const int stride_h = input.stride(2);
    const int stride_d = input.stride(3);
    
    const int o_stride_s = output.stride(0);
    const int o_stride_b = output.stride(1);
    const int o_stride_h = output.stride(2);
    const int o_stride_d = output.stride(3);
    
    nvte_fused_rope_forward_metal(input, freqs, output, s, b, h, d, d2,
                               stride_s, stride_b, stride_h, stride_d,
                               o_stride_s, o_stride_b, o_stride_h, o_stride_d);
    
    return output;
}

at::Tensor fused_rope_backward_metal(const at::Tensor &output_grads,
                                 const at::Tensor &freqs,
                                 const bool transpose_output_memory) {
    TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(freqs.dim() == 2, "expected 2D tensor");
    
    const int s = output_grads.size(0);
    const int b = output_grads.size(1);
    const int h = output_grads.size(2);
    const int d = output_grads.size(3);
    const int d2 = d / 2;
    
    auto input_grads = at::empty_like(output_grads);
    
    const int stride_s = output_grads.stride(0);
    const int stride_b = output_grads.stride(1);
    const int stride_h = output_grads.stride(2);
    const int stride_d = output_grads.stride(3);
    
    const int o_stride_s = input_grads.stride(0);
    const int o_stride_b = input_grads.stride(1);
    const int o_stride_h = input_grads.stride(2);
    const int o_stride_d = input_grads.stride(3);
    
    nvte_fused_rope_backward_metal(output_grads, freqs, input_grads, s, b, h, d, d2,
                                stride_s, stride_b, stride_h, stride_d,
                                o_stride_s, o_stride_b, o_stride_h, o_stride_d);
    
    return input_grads;
}
