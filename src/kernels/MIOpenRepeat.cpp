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

__device__ void GET_NCDHW(uint64_t& n,
                          uint64_t& c,
                          uint64_t& d,
                          uint64_t& h,
                          uint64_t& w,
                          uint64_t gid,
                          uint64_t output_dim0,
                          uint64_t output_dim1,
                          uint64_t output_dim2,
                          uint64_t output_dim3,
                          uint64_t output_dim4)
{
    uint64_t ncdh = (gid) / output_dim4;
    w             = (gid) % output_dim4;
    uint64_t ncd  = (ncdh) / output_dim3;
    h             = (ncdh) % output_dim3;
    uint64_t nc   = (ncd) / output_dim2;
    d             = (ncd) % output_dim2;
    n             = (nc) / output_dim1;
    c             = (nc) % output_dim1;
}

__device__ void GET_5D_INDEX(const uint64_t input_dimensions[5],
                             uint64_t n,
                             uint64_t c,
                             uint64_t d,
                             uint64_t h,
                             uint64_t w,
                             uint64_t& output)
{
    output = (((n * input_dimensions[1] + c) * input_dimensions[2] + d) * input_dimensions[3] + h) *
                 input_dimensions[4] +
             w;
}

extern "C" __global__ void RepeatForward(const FLOAT* __restrict__ x,
                                         FLOAT* __restrict__ y,
                                         uint64_t inout_size,
                                         uint64_t offset,
                                         uint64_t input_dim0,
                                         uint64_t input_dim1,
                                         uint64_t input_dim2,
                                         uint64_t input_dim3,
                                         uint64_t input_dim4,
                                         uint64_t output_dim0,
                                         uint64_t output_dim1,
                                         uint64_t output_dim2,
                                         uint64_t output_dim3,
                                         uint64_t output_dim4)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    uint64_t input_dimensions[5] = {input_dim0, input_dim1, input_dim2, input_dim3, input_dim4};

    // get output index
    uint64_t o[5];
    GET_NCDHW(o[0],
              o[1],
              o[2],
              o[3],
              o[4],
              gid,
              output_dim0,
              output_dim1,
              output_dim2,
              output_dim3,
              output_dim4);

    // get input index
    uint64_t n[5] = {0, 0, 0, 0, 0};
    for(uint64_t i = offset; i < 5; i++)
    {
        n[i - offset] = o[i] % input_dimensions[i - offset];
    }

    uint64_t input_index = 0;
    GET_5D_INDEX(input_dimensions, n[0], n[1], n[2], n[3], n[4], input_index);
    y[gid] = x[input_index];
}

extern "C" __global__ void RepeatBackward(const FLOAT* __restrict__ dy,
                                          FLOAT* __restrict__ dx,
                                          uint64_t inout_size,
                                          uint64_t offset,
                                          uint64_t output_grad_dim0,
                                          uint64_t output_grad_dim1,
                                          uint64_t output_grad_dim2,
                                          uint64_t output_grad_dim3,
                                          uint64_t output_grad_dim4,
                                          uint64_t input_grad_dim0,
                                          uint64_t input_grad_dim1,
                                          uint64_t input_grad_dim2,
                                          uint64_t input_grad_dim3,
                                          uint64_t input_grad_dim4)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= inout_size)
        return;

    uint64_t input_grad_dimensions[5] = {
        input_grad_dim0, input_grad_dim1, input_grad_dim2, input_grad_dim3, input_grad_dim4};

    // get output index
    uint64_t o[5];
    GET_NCDHW(o[0],
              o[1],
              o[2],
              o[3],
              o[4],
              gid,
              output_grad_dim0,
              output_grad_dim1,
              output_grad_dim2,
              output_grad_dim3,
              output_grad_dim4);

    // get input index
    uint64_t n[5] = {0, 0, 0, 0, 0};
    for(uint64_t i = offset; i < 5; i++)
    {
        n[i - offset] = o[i] % input_grad_dimensions[i - offset];
    }

    FLOAT output_grad_v = dy[gid];

    uint64_t input_grad_index;
    GET_5D_INDEX(input_grad_dimensions, n[0], n[1], n[2], n[3], n[4], input_grad_index);
    atomicAdd(&dx[input_grad_index], output_grad_v);
}
