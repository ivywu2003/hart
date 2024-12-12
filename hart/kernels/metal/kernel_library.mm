#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <algorithm>  // for std::min

#include "kernel_library.h"

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

namespace hart {

// Wrapper function to launch the kernel with appropriate parameters
void rms_norm_metal(torch::Tensor& output,
                   const torch::Tensor& input,
                   const torch::Tensor& weight,
                   float epsilon,
                   bool use_quant) {
    
    int hidden_size = input.size(-1);
    int num_tokens = (input.numel() + (hidden_size - 1)) / hidden_size;

    @autoreleasepool {
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        id<MTLLibrary> layernormKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:KERNEL]
                                                                  options:nil
                                                                    error:&error];
        TORCH_CHECK(layernormKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string input_type = (input.scalar_type() == torch::kFloat ? "float" : "half");
        std::string output_type = (use_quant ? "char_true" : input_type + "_false");
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
            [computeEncoder setBytes:&hidden_size length:sizeof(uint32_t) atIndex:3];
            [computeEncoder setBytes:&epsilon length:sizeof(float) atIndex:4];

            // Calculate grid and threadgroup sizes
            MTLSize gridSize = MTLSizeMake(num_tokens, 1, 1);
            MTLSize threadgroupSize = MTLSizeMake(std::min(hidden_size, 1024), 1, 1);
            
            // Dispatch the kernel
            [computeEncoder dispatchThreadgroups:gridSize
                        threadsPerThreadgroup:threadgroupSize];
            
            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
}

at::Tensor fused_rope_with_pos_forward_metal(const at::Tensor &input,
                                        const at::Tensor &freqs,
                                        const bool transpose_output_memory) {
    const int s = input.size(0);
    const int b = input.size(1);
    const int h = input.size(2);
    const int d = input.size(3);
    
    const int stride_s = input.stride(0);
    const int stride_b = input.stride(1);
    const int stride_h = input.stride(2);
    const int stride_d = input.stride(3);

    const int d2 = freqs.size(-1);
    
    auto act_options = input.options().requires_grad(false);
    at::Tensor output;
    if (transpose_output_memory) {
        output = torch::empty({b, s, h, d}, act_options).transpose(0, 1);
    } else {
        output = torch::empty({s, b, h, d}, act_options);
    }
    
    const int o_stride_s = output.stride(0);
    const int o_stride_b = output.stride(1);
    const int o_stride_h = output.stride(2);
    const int o_stride_d = output.stride(3);

    @autoreleasepool {
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        id<MTLLibrary> fusedRopeKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:KERNEL]
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
            [computeEncoder setBytes:&s length:sizeof(int) atIndex:3];
            [computeEncoder setBytes:&b length:sizeof(int) atIndex:4];
            [computeEncoder setBytes:&h length:sizeof(int) atIndex:5];
            [computeEncoder setBytes:&d length:sizeof(int) atIndex:6];
            [computeEncoder setBytes:&d2 length:sizeof(int) atIndex:7];
            [computeEncoder setBytes:&stride_s length:sizeof(int) atIndex:8];
            [computeEncoder setBytes:&stride_b length:sizeof(int) atIndex:9];
            [computeEncoder setBytes:&stride_h length:sizeof(int) atIndex:10];
            [computeEncoder setBytes:&stride_d length:sizeof(int) atIndex:11];
            [computeEncoder setBytes:&o_stride_s length:sizeof(int) atIndex:12];
            [computeEncoder setBytes:&o_stride_b length:sizeof(int) atIndex:13];
            [computeEncoder setBytes:&o_stride_h length:sizeof(int) atIndex:14];
            [computeEncoder setBytes:&o_stride_d length:sizeof(int) atIndex:15];

            // Set grid and threadgroup size
            MTLSize gridSize = MTLSizeMake(s, b, 1);
            MTLSize threadgroupSize = MTLSizeMake(32, h < 16 ? 4 : 8, 1);
            [computeEncoder dispatchThreadgroups:gridSize
                        threadsPerThreadgroup:threadgroupSize];
            
            [computeEncoder endEncoding];

            // Commit the work.
            torch::mps::commit();
        });
    }
    
    return output;
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

        id<MTLLibrary> fusedRopeKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:KERNEL]
                                                                     options:nil
                                                                       error:&error];
        TORCH_CHECK(fusedRopeKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string input_type = (input.scalar_type() == torch::kFloat ? "float" : "half");
        std::string kernel_name = std::string("fused_rope_forward_") + input_type;
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

at::Tensor fused_rope_backward_metal(const at::Tensor &output_grads,
                                        const at::Tensor &freqs,
                                        const bool transpose_output_memory) {
    TORCH_CHECK(output_grads.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(freqs.dim() == 2, "expected 2D tensor");
    
    const int s = output_grads.size(0);
    const int b = output_grads.size(1);
    const int h = output_grads.size(2);
    const int d = output_grads.size(3);
    
    auto input_grads = at::empty_like(output_grads);
    
    const int stride_s = output_grads.stride(0);
    const int stride_b = output_grads.stride(1);
    const int stride_h = output_grads.stride(2);
    const int stride_d = output_grads.stride(3);

    const int d2 = freqs.size(-1);
    
    const int o_stride_s = input_grads.stride(0);
    const int o_stride_b = input_grads.stride(1);
    const int o_stride_h = input_grads.stride(2);
    const int o_stride_d = input_grads.stride(3);

    @autoreleasepool {
        // Get the default Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        NSError *error = nil;

        id<MTLLibrary> fusedRopeKernelLibrary = [device newLibraryWithSource:[NSString stringWithUTF8String:KERNEL]
                                                                     options:nil
                                                                       error:&error];
        TORCH_CHECK(fusedRopeKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

        std::string input_type = (output_grads.scalar_type() == torch::kFloat ? "float" : "half");
        std::string kernel_name = std::string("fused_rope_backward_") + input_type;
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
            [computeEncoder setBuffer:getMTLBufferStorage(output_grads) offset:output_grads.storage_offset() * output_grads.element_size() atIndex:0];
            [computeEncoder setBuffer:getMTLBufferStorage(freqs) offset:freqs.storage_offset() * freqs.element_size() atIndex:1];
            [computeEncoder setBuffer:getMTLBufferStorage(input_grads) offset:input_grads.storage_offset() * input_grads.element_size() atIndex:2];
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
    
    return input_grads;
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm_metal,
        py::arg("output"), 
        py::arg("input"), 
        py::arg("weight"),
        py::arg("epsilon"),
        py::arg("use_quant"),
        "RMS LayerNorm implementation using Metal"
    );
    m.def("fused_rope_with_pos_forward_func", &fused_rope_with_pos_forward_metal,
        py::arg("input"), 
        py::arg("freqs"),
        py::arg("transpose_output_memory"),
        "Fused RoPE with Position Embedding implementation using Metal"
    );
    m.def("fused_rope_forward_func", &fused_rope_forward_metal,
        py::arg("input"), 
        py::arg("freqs"),
        py::arg("transpose_output_memory"),
        "Fused RoPE without Position Embedding implementation using Metal"
    );
    m.def("fused_rope_backward_func", &fused_rope_backward_metal,
        py::arg("output_grads"), 
        py::arg("freqs"),
        py::arg("transpose_output_memory"),
        "Fused RoPE without Position Embedding backwards implementation using Metal"
    );
}

} // namespace hart