#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
// #include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#define VDTYPE int
#define output_t VDTYPE

// Output should be initialized with `off_value`.
extern "C" __global__ void
OneHotContiguous(const int* input, output_t* output, long input_size, long num_classes)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    int val = input[gid];

    if(val >= num_classes)
    {
        printf("ERROR_CODE_LARGER_THAN_NUM_CLASS");
        return;
    }
    if(val < 0)
    {
        printf("ERROR_CODE_NEG_VALUE");
        return;
    }

    output_t on_val                 = output_t(1);
    output[gid * num_classes + val] = on_val;
}
