#include <pybind11/pybind11.h>
// #include <ATen/ATen.h>
#include <torch/extension.h>
#include <OpenCL/opencl.h>
#include <vector>
#include <iostream>

// Load and compile the OpenCL kernel
std::string loadKernelSource(const std::string &filename);
cl_program buildProgram(cl_context context, cl_device_id device, const std::string &source);


cl_device_id get_device() {
    cl_int err;

    // Step 1: Get the platform
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get platform ID\n");
        return nullptr;
    }

    // Step 2: Get the device (e.g., GPU)
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to get device ID\n");
        return nullptr;
    }
    return device;
}

cl_context get_context() {
    cl_int err;

    // Step 2: Get the device (e.g., GPU)
    cl_device_id device = get_device();

    // Step 3: Create a context for the device
    cl_context context = clCreateContext(
        NULL,        // Properties (NULL uses default)
        1,           // Number of devices
        &device,     // Array of devices
        NULL,        // Callback function (optional)
        NULL,        // User data for the callback (optional)
        &err);       // Error code
    if (err != CL_SUCCESS) {
        printf("Failed to create context\n");
        return nullptr;
    }
    return context;
}

void rms_norm(
    cl_mem out,
    cl_mem input,
    cl_mem weight,
    float epsilon,
    bool use_quant,
    int hidden_size = 32,
    int num_tokens = 2) {

    cl_int err;

    cl_context context = get_context();
    cl_device_id device = get_device();
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // Load the kernel source code
    std::string kernel_source = loadKernelSource("layernorm_kernels.cl");
    cl_program program = buildProgram(context, device, kernel_source);

    // Create kernel
    cl_kernel kernel = clCreateKernel(program, "layernorm_kernels", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &out);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &weight);
    clSetKernelArg(kernel, 3, sizeof(float), &epsilon);
    clSetKernelArg(kernel, 4, sizeof(int), &hidden_size);
    clSetKernelArg(kernel, 5, sizeof(int), &num_tokens);

    // Set up the execution configuration
    size_t global_size = num_tokens;
    size_t local_size = std::min(hidden_size, 1024);

    // Enqueue kernel
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

    // Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, 
        "RMS Norm function in C++",
        py::arg("output"),
        py::arg("input"),
        py::arg("weight"),
        py::arg("epsilon"),
        py::arg("use_quant"));
}