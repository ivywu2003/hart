__kernel void rms_norm_kernel(
    __global float *out,
    __global const float *input,
    __global const float *weight,
    const float epsilon,
    const int hidden_size,
    const int num_tokens) {
  
    // Calculate global and local indices
    int token_idx = get_global_id(0);
    int thread_idx = get_local_id(0);
    int local_size = get_local_size(0);
    __local float s_variance;

    // Initialize variance to 0
    float variance = 0.0f;

    // Step 1: Calculate the sum of squares for this token
    for (int i = thread_idx; i < hidden_size; i += local_size) {
        float x = input[token_idx * hidden_size + i];
        variance += x * x;
    }

    // Step 2: Perform reduction across workgroup to compute total variance
    for (int offset = local_size / 2; offset > 0; offset >>= 1) {
        // TODO
        // variance += work_group_reduce_add(variance);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Step 3: Compute RMS and store in shared memory
    if (thread_idx == 0) {
        s_variance = rsqrt(variance / hidden_size + epsilon);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Step 4: Normalize the input and compute the output
    for (int i = thread_idx; i < hidden_size; i += local_size) {
        float x = input[token_idx * hidden_size + i];
        out[token_idx * hidden_size + i] = x * s_variance * weight[i];
    }
}
