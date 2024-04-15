#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
// #include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename TI, typename TO>
__device__ void
oneHotContiguousKernel(const int* input, int* output, long input_size, int num_classes)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    TI val = input[gid];

    output[gid * num_classes + val] = 1;
}

extern "C" __global__ void
OneHotContiguous(const INPUT_TYPE* input, OUTPUT_TYPE* output, long input_size, int num_classes)
{
    oneHotContiguousKernel<INPUT_TYPE, OUTPUT_TYPE>(input, output, input_size, num_classes);
}
