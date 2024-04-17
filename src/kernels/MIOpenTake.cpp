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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

// TODO: edit like this https://github.com/ROCm/MIOpen/pull/2583/files#diff-7dd07357da54a672f7926d98e6219f9c9f93b16f977a836f670561f0af046b98
extern "C" __global__ void TakeFwdContiguous(const FLOAT* __restrict__ x,
                                             FLOAT* __restrict__ y,
                                             const int32_t* __restrict__ index,
                                             int64_t output_numel,
                                             int64_t input_numel) 
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid >= output_numel)
        return;

    int32_t index_v = index[gid];
    if (index_v < -input_numel || index_v >= input_numel) return;
    index_v += input_numel * static_cast<uint64_t>(index_v < 0);
    y[gid] = x[index_v];
}
