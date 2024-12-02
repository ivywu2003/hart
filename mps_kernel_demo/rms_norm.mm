/*
   See the LICENSE.txt file for this sample’s licensing information.

Abstract:
The code that registers a PyTorch custom operation.
 */


#include <torch/extension.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Helper function to retrieve the `MTLBuffer` from a `torch::Tensor`.
static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor& tensor) {
  return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

torch::Tensor& dispatchKernel(torch::Tensor& output,
                              const torch::Tensor& input, 
                              const torch::Tensor& weight,
                              const float epsilon,
                              const bool use_quant) {
  uint32_t hidden_size = input.size(-1);
  printf("hidden size: %d\n", hidden_size);
  int num_tokens = input.numel() / hidden_size;
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    NSError *error = nil;

    // Set the number of threads equal to the number of elements within the input tensor.
    int numThreads = input.numel();

    // Load the custom soft shrink shader.
    const std::string& rms_norm_filepath = std::string("rms_norm.metal");
    NSString* shaderSource = [
      NSString stringWithContentsOfFile:[NSString stringWithUTF8String:rms_norm_filepath.c_str()]
      encoding:NSUTF8StringEncoding
      error:&error];

    printf("%s", [shaderSource cStringUsingEncoding:NSUTF8StringEncoding]);

    if (error) {
      throw std::runtime_error("Failed to load Metal shader: " + std::string(error.localizedDescription.UTF8String));
    }
    id<MTLLibrary> customKernelLibrary = [
      device newLibraryWithSource:shaderSource
      options:nil
      error:&error];
    TORCH_CHECK(customKernelLibrary, "Failed to to create custom kernel library, error: ", error.localizedDescription.UTF8String);

    std::string kernel_name = std::string("rms_norm_kernel");
    id<MTLFunction> customFunction = [customKernelLibrary newFunctionWithName:[NSString stringWithUTF8String:kernel_name.c_str()]];
    TORCH_CHECK(customFunction, "Failed to create function state object for ", kernel_name.c_str());

    // Create a compute pipeline state object for the soft shrink kernel.
    id<MTLComputePipelineState> PSO = [device newComputePipelineStateWithFunction:customFunction error:&error];
    TORCH_CHECK(PSO, error.localizedDescription.UTF8String);

    // Get a reference to the command buffer for the MPS stream.
    id<MTLCommandBuffer> commandBuffer = torch::mps::get_command_buffer();
    TORCH_CHECK(commandBuffer, "Failed to retrieve command buffer reference");

    // Get a reference to the dispatch queue for the MPS stream, which encodes the synchronization with the CPU.
    dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();

    dispatch_sync(serialQueue, ^(){
        // Start a compute pass.
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
        TORCH_CHECK(computeEncoder, "Failed to create compute command encoder");

        // Encode the pipeline state object and its parameters.
        [computeEncoder setComputePipelineState:PSO];
        [computeEncoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:0];
        [computeEncoder setBuffer:getMTLBufferStorage(input) offset:input.storage_offset() * input.element_size() atIndex:1];
        [computeEncoder setBuffer:getMTLBufferStorage(weight) offset:weight.storage_offset() * weight.element_size() atIndex:2];
        [computeEncoder setBytes:&hidden_size length:sizeof(uint32_t) atIndex:3];
        [computeEncoder setBytes:&epsilon length:sizeof(float) atIndex:4];

        MTLSize gridSize = MTLSizeMake(input.numel(), 1, 1);

        // Calculate a thread group size.
        NSUInteger threadGroupSize = PSO.maxTotalThreadsPerThreadgroup;
        NSUInteger hiddenSize = hidden_size;
        if (hiddenSize < threadGroupSize) {
          threadGroupSize = hiddenSize;
        }
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        // Encode the compute command.
        [computeEncoder dispatchThreadgroups:gridSize
          threadsPerThreadgroup:threadgroupSize];

        [computeEncoder endEncoding];

        // Commit the work.
        torch::mps::commit();
    });
  }

  return output;
}

// C++ op dispatching the Metal soft shrink shader.
torch::Tensor mps_rms_norm(const torch::Tensor& input, 
                           const torch::Tensor& weight,
                           const float epsilon,
                           const bool use_quant) {
  // Check whether the input tensor resides on the MPS device and whether it's contiguous.
  TORCH_CHECK(input.device().is_mps(), "input must be a MPS tensor");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

  // Allocate the output, same shape as the input.
  torch::Tensor output = torch::empty_like(input);

  return dispatchKernel(output, input, weight, epsilon, use_quant);
}

// Create Python bindings for the Objective-C++ code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mps_rms_norm", &mps_rms_norm);
}

