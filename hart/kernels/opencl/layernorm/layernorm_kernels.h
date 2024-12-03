#include <OpenCL/opencl.h>

void rms_norm(
    cl_mem out,
    cl_mem input,
    cl_mem weight,
    float epsilon,
    bool use_quant);