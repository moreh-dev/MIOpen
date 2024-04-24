/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <cstdio>
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#endif

#define ONEHOT_ERROR_CODE_NEG_VALUE 1
#define ONEHOT_ERROR_CODE_LARGER_THAN_NUM_CLASS 2

template <typename TI, typename TO, typename TE>
__device__ void
oneHotContiguousKernel(const TI* input, TO* output, TE* err, long input_size, int num_classes)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= input_size)
        return;

    TI val = input[gid];
    if(val < 0)
    {
        *err = ONEHOT_ERROR_CODE_NEG_VALUE;
        return;
    }
    if(val >= num_classes)
    {
        *err = ONEHOT_ERROR_CODE_LARGER_THAN_NUM_CLASS;
        return;
    }

    output[gid * num_classes + val] = 1;
}

extern "C" __global__ void OneHotContiguous(
    const INPUT_TYPE* input, OUTPUT_TYPE* output, ERR_TYPE* err, long input_size, int num_classes)
{
    oneHotContiguousKernel<INPUT_TYPE, OUTPUT_TYPE, ERR_TYPE>(
        input, output, err, input_size, num_classes);
}
