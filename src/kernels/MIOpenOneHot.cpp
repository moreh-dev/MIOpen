#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
// #include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

// Output should be initialized with `off_value`.
extern "C" __global__ void
OneHotContiguous(const int* input, int* output, long input_size, int num_classes)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    int val = input[gid];

    output[gid * num_classes + val] = 1;
}
