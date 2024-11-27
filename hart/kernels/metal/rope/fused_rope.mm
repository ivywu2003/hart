#include <Metal/Metal.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "fused_rope.h"

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

namespace hart {

at::Tensor fused_rope_with_pos_forward_metal(const at::Tensor &input,
                                        const at::Tensor &freqs,
                                        const bool transpose_output_memory) {
    TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(freqs.dim() == 2, "expected 2D tensor");
    
    const int s = input.size(0);
    const int b = input.size(1);
    const int h = input.size(2);
    const int d = input.size(3);
    
    auto output = at::empty_like(input);
    
    const int stride_s = input.stride(0);
    const int stride_b = input.stride(1);
    const int stride_h = input.stride(2);
    const int stride_d = input.stride(3);

    const int d2 = freqs.size(-1);
    
    const int o_stride_s = output.stride(0);
    const int o_stride_b = output.stride(1);
    const int o_stride_h = output.stride(2);
    const int o_stride_d = output.stride(3);

    @autoreleasepool {
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        id<MTLLibrary> fusedRopeKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:ROPE_KERNEL]
                                                                     options:nil
                                                                       error:&error];
        TORCH_CHECK(fusedRopeKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string input_type = (input.scalar_type() == torch::kFloat ? "float" : "half");
        std::string kernel_name = std::string("fused_rope_with_pos_forward_") + input_type;
        id<MTLFunction> customRopeFunction = [fusedRopeKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customRopeFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:customRopeFunction error:&error];
        TORCH_CHECK(pipelineState, error.localizedDescription.UTF8String);
        
        // Get a reference to the command buffer for the MPS stream.
        id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
        TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

        // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

        dispatch_sync(serialQueue, ^(){
            // Start a compute pass.
            id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
            TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

            // Set the compute pipeline state.
            [computeEncoder setComputePipelineState:pipelineState];

            // Set buffers and parameters
            [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(freqs) offset:freqs.storage_offset() * freqs.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];
            [computeEncoder setBytes:&h length:sizeof(int) atIndex:3];
            [computeEncoder setBytes:&d length:sizeof(int) atIndex:4];
            [computeEncoder setBytes:&d2 length:sizeof(int) atIndex:5];
            [computeEncoder setBytes:&stride_h length:sizeof(int) atIndex:6];
            [computeEncoder setBytes:&stride_d length:sizeof(int) atIndex:7];
            [computeEncoder setBytes:&o_stride_h length:sizeof(int) atIndex:8];
            [computeEncoder setBytes:&o_stride_d length:sizeof(int) atIndex:9];
            [computeEncoder setBytes:&s length:sizeof(int) atIndex:10];

            // Set grid and threadgroup size
            MTLSize gridSize = MTLSizeMake(s, b, 1);
            MTLSize threadgroupSize = MTLSizeMake(32, 32, 1);
            [computeEncoder dispatchThreadgroups:gridSize
                        threadsPerThreadgroup:threadgroupSize];
            
            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
    
    return output;
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rope_with_pos_forward_metal", &fused_rope_with_pos_forward_metal,
        py::arg("input"), 
        py::arg("freqs"),
        py::arg("transpose_output_memory"),
        "Fused RoPE with Position Embedding implementation using Metal"
    );
}

}