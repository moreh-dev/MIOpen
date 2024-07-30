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

extern "C" __global__ void IndexSelectForward(FLOAT* x,
                                              FLOAT* y,
                                              const size_t in_sz0,
                                              const size_t in_sz1,
                                              const size_t in_sz2,
                                              const size_t in_sz3,
                                              const size_t out_sz0,
                                              const size_t out_sz1,
                                              const size_t out_sz2,
                                              const size_t out_sz3,
                                              const size_t in_st0,
                                              const size_t in_st1,
                                              const size_t in_st2,
                                              const size_t in_st3,
                                              const size_t out_st0,
                                              const size_t out_st1,
                                              const size_t out_st2,
                                              const size_t out_st3,
                                              const size_t dim,
                                              int* indices)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[4];
    size_t n012;
    size_t n01;
    n[3] = gid % out_sz3;
    n012 = gid / out_sz3;
    n[2] = n012 % out_sz2;
    n01  = n012 / out_sz2;
    n[1] = n01 % out_sz1;
    n[0] = n01 / out_sz1;

    if(n[0] >= out_sz0)
        return;

    size_t output_idx = n[0] * out_st0 + n[1] * out_st1 + n[2] * out_st2 + n[3] * out_st3;

    n[dim] = indices[n[dim]];

    size_t input_idx = n[0] * in_st0 + n[1] * in_st1 + n[2] * in_st2 + n[3] * in_st3;

    y[output_idx] = x[input_idx];
}

extern "C" __global__ void IndexSelectForwardContiguous(FLOAT* x,
                                                        FLOAT* y,
                                                        const size_t in_sz0,
                                                        const size_t in_sz1,
                                                        const size_t in_sz2,
                                                        const size_t in_sz3,
                                                        const size_t out_sz0,
                                                        const size_t out_sz1,
                                                        const size_t out_sz2,
                                                        const size_t out_sz3,
                                                        const size_t in_st0,
                                                        const size_t in_st1,
                                                        const size_t in_st2,
                                                        const size_t in_st3,
                                                        const size_t out_st0,
                                                        const size_t out_st1,
                                                        const size_t out_st2,
                                                        const size_t out_st3,
                                                        const size_t dim,
                                                        int* indices)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t n[4];
    size_t n012;
    size_t n01;
    n[3] = gid % out_sz3;
    n012 = gid / out_sz3;
    n[2] = n012 % out_sz2;
    n01  = n012 / out_sz2;
    n[1] = n01 % out_sz1;
    n[0] = n01 / out_sz1;

    if(n[0] >= out_sz0)
        return;

    size_t output_idx = gid;

    n[dim] = indices[n[dim]];

    size_t input_idx = n[0] * in_st0 + n[1] * in_st1 + n[2] * in_st2 + n[3] * in_st3;

    y[output_idx] = x[input_idx];
}

extern "C" __global__ void IndexSelectBackward(FLOAT* inGrad,
                                               FLOAT* outGrad,
                                               const size_t inGrad_sz0,
                                                const size_t inGrad_sz1,
                                                const size_t inGrad_sz2,
                                                const size_t inGrad_sz3,
                                                const size_t outGrad_sz0,
                                                const size_t outGrad_sz1,
                                                const size_t outGrad_sz2,
                                                const size_t outGrad_sz3,
                                                const size_t inGrad_st0,
                                                const size_t inGrad_st1,
                                                const size_t inGrad_st2,
                                                const size_t inGrad_st3,
                                                const size_t outGrad_st0,
                                                const size_t outGrad_st1,
                                                const size_t outGrad_st2,
                                                const size_t outGrad_st3,
                                                const size_t dim,
                                                size_t N,
                                                size_t st,
                                                size_t iK,
                                                size_t oK,
                                               int* indices)
{
    size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
    if(gid >= N)
        return;

    size_t output_grad_base_idx = (gid / st) * st * oK + gid % st;
    size_t n[4], n012, n01;
    n[3] = output_grad_base_idx % outGrad_sz3;
    n012 = output_grad_base_idx / outGrad_sz3;
    n[2] = n012 % outGrad_sz2;
    n01  = n012 / outGrad_sz2;
    n[1] = n01 % outGrad_sz1;
    n[0] = n01 / outGrad_sz1;

    for(int i = 0; i < oK; i++)
    {
        n[dim]     = i;
        size_t idx = indices[i];
        size_t output_grad_idx =
            n[0] * outGrad_st0 + n[1] * outGrad_st1 + n[2] * outGrad_st2 + n[3] * outGrad_st3;
        n[dim] = idx;
        size_t input_grad_idx =
            n[0] * inGrad_st0 + n[1] * inGrad_st1 + n[2] * inGrad_st2 + n[3] * inGrad_st3;

        inGrad[input_grad_idx] = inGrad[input_grad_idx] + outGrad[output_grad_idx];
    }
}
