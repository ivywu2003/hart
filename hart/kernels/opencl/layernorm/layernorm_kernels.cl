__kernel void test_kernel(__global float *output) {
    int idx = get_global_id(0);
    output[idx] = (float)idx;
}
