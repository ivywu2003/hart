#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <algorithm>  // for std::min

#include "layernorm.h"

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

namespace hart {

// Wrapper function to launch the kernel with appropriate parameters
void rms_norm_metal(torch::Tensor& output,
                   const torch::Tensor& input,
                   const torch::Tensor& weight,
                   const LayerNormParams& params) {
    
    @autoreleasepool {
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        id<MTLLibrary> layernormKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:LAYERNORM_KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(layernormKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string input_type = (input.scalar_type() == torch::kFloat ? "float" : "half");
        std::string output_type = (params.use_quant ? "int8_true" : input_type + "_false");
        std::string kernel_name = std::string("rms_norm_kernel_") + input_type + "_" + output_type;
        id<MTLFunction> customLayernormFunction = [layernormKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
        TORCH_CHECK(customLayernormFunction, "Failed to create function state object for ", kernel_name.c_str());

        // Create a compute pipeline state object for the soft shrink kernel.
        id<MTLComputePipelineState> pipelineState = [device newComputePipelineStateWithFunction:customLayernormFunction error:&error];
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

            // Set buffers
            [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(weight) offset:weight.storage_offset() * weight.element_size() atIndex:2];
            [computeEncoder setBytes:&params length:sizeof(LayerNormParams) atIndex:3];

            // Calculate grid and threadgroup sizes
            MTLSize gridSize = MTLSizeMake(params.num_tokens, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(std::min(params.hidden_size, 1024u), 1, 1);
            
            // Dispatch the kernel
            [computeEncoder dispatchThreadgroups:gridSize
                        threadsPerThreadgroup:threadgroupSize];
            
            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
}


// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm_metal", &rms_norm_metal, 
        py::arg("output"), py::arg("input"), py::arg("weight"), py::arg("params"),
        "Apply Root Mean Square (RMS) Normalization to the input tensor.");
}

} // namespace hart